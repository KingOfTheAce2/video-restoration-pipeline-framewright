"""
Power Management - Prevent sleep and manage system power states.

Keeps the system awake during long processing jobs and provides
options for auto-hibernate when complete.
"""

import ctypes
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum


class PowerAction(Enum):
    """Actions to take when processing completes."""
    NONE = "none"
    SLEEP = "sleep"
    HIBERNATE = "hibernate"
    SHUTDOWN = "shutdown"


@dataclass
class PowerState:
    """Current power state information."""
    on_battery: bool
    battery_percent: Optional[float]
    is_charging: bool
    time_remaining_minutes: Optional[float]
    power_saver_active: bool


class PowerManager:
    """
    Manage system power states during processing.

    - Prevents system sleep during active processing
    - Monitors battery status
    - Executes power actions on completion
    - Works on Windows, with stubs for other platforms
    """

    # Windows constants
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_AWAYMODE_REQUIRED = 0x00000040

    def __init__(
        self,
        prevent_sleep: bool = True,
        keep_display_on: bool = False,
        completion_action: PowerAction = PowerAction.NONE,
        low_battery_callback: Optional[Callable[[float], None]] = None,
        low_battery_threshold: float = 10.0
    ):
        """
        Initialize power manager.

        Args:
            prevent_sleep: Prevent system sleep while active
            keep_display_on: Also prevent display sleep
            completion_action: Action when processing completes
            low_battery_callback: Called when battery is low
            low_battery_threshold: Battery % to trigger callback
        """
        self.prevent_sleep = prevent_sleep
        self.keep_display_on = keep_display_on
        self.completion_action = completion_action
        self.low_battery_callback = low_battery_callback
        self.low_battery_threshold = low_battery_threshold

        self._active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._low_battery_warned = False

    def _set_thread_execution_state(self, flags: int) -> bool:
        """Set Windows thread execution state."""
        if sys.platform != 'win32':
            return False

        try:
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
            return True
        except Exception:
            return False

    def _prevent_sleep_windows(self) -> bool:
        """Prevent sleep on Windows."""
        flags = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        if self.keep_display_on:
            flags |= self.ES_DISPLAY_REQUIRED
        return self._set_thread_execution_state(flags)

    def _allow_sleep_windows(self) -> bool:
        """Allow sleep on Windows."""
        return self._set_thread_execution_state(self.ES_CONTINUOUS)

    def get_power_state(self) -> PowerState:
        """Get current power state."""
        state = PowerState(
            on_battery=False,
            battery_percent=None,
            is_charging=False,
            time_remaining_minutes=None,
            power_saver_active=False
        )

        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                state.on_battery = not battery.power_plugged
                state.battery_percent = battery.percent
                state.is_charging = battery.power_plugged
                if battery.secsleft > 0:
                    state.time_remaining_minutes = battery.secsleft / 60
        except ImportError:
            pass

        # Check Windows power saver
        if sys.platform == 'win32':
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
                )
                # This is a simplified check
                state.power_saver_active = False
            except Exception:
                pass

        return state

    def _monitor_loop(self) -> None:
        """Monitor power state in background."""
        while not self._stop_event.is_set():
            state = self.get_power_state()

            # Check low battery
            if (state.battery_percent is not None and
                state.battery_percent < self.low_battery_threshold and
                state.on_battery and
                not self._low_battery_warned):

                self._low_battery_warned = True
                if self.low_battery_callback:
                    self.low_battery_callback(state.battery_percent)

            # Reset warning if charging
            if state.is_charging:
                self._low_battery_warned = False

            # Keep system awake
            if self._active and self.prevent_sleep:
                self._prevent_sleep_windows()

            self._stop_event.wait(30)  # Check every 30 seconds

    def start(self) -> bool:
        """
        Start power management.

        Returns True if successfully preventing sleep.
        """
        self._active = True
        self._stop_event.clear()

        success = False
        if self.prevent_sleep:
            success = self._prevent_sleep_windows()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        return success

    def stop(self) -> None:
        """Stop power management and execute completion action."""
        self._active = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        # Allow sleep again
        self._allow_sleep_windows()

        # Execute completion action
        self._execute_completion_action()

    def _execute_completion_action(self) -> None:
        """Execute the configured completion action."""
        if self.completion_action == PowerAction.NONE:
            return

        if sys.platform != 'win32':
            print(f"Power action {self.completion_action.value} not supported on this platform")
            return

        try:
            if self.completion_action == PowerAction.SLEEP:
                # SetSuspendState(hibernate, force, wakeup_events)
                ctypes.windll.powrprof.SetSuspendState(0, 0, 0)

            elif self.completion_action == PowerAction.HIBERNATE:
                ctypes.windll.powrprof.SetSuspendState(1, 0, 0)

            elif self.completion_action == PowerAction.SHUTDOWN:
                import subprocess
                subprocess.run(['shutdown', '/s', '/t', '60', '/c',
                              'FrameWright processing complete. Shutting down in 60 seconds.'],
                             check=False)

        except Exception as e:
            print(f"Error executing power action: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class KeepAwake:
    """
    Simple context manager to keep system awake.

    Usage:
        with KeepAwake():
            # Long processing here
            pass
    """

    def __init__(self, keep_display_on: bool = False):
        self.manager = PowerManager(
            prevent_sleep=True,
            keep_display_on=keep_display_on,
            completion_action=PowerAction.NONE
        )

    def __enter__(self):
        self.manager.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.stop()
        return False


def prevent_sleep_during(func: Callable) -> Callable:
    """
    Decorator to prevent sleep during function execution.

    Usage:
        @prevent_sleep_during
        def long_processing():
            # Processing here
            pass
    """
    def wrapper(*args, **kwargs):
        with KeepAwake():
            return func(*args, **kwargs)
    return wrapper
