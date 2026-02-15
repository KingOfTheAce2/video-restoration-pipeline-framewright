"""Thermal Monitoring and Throttling Detection for GPU Processing.

Monitors GPU temperatures and detects thermal throttling to automatically
adapt processing parameters for stable long-running jobs.

Features:
- Real-time temperature monitoring (NVIDIA, AMD, Intel)
- Thermal throttling detection
- Automatic batch size reduction when hot
- Performance degradation tracking
- Cool-down pause recommendations
- Temperature history and logging

Example:
    >>> monitor = ThermalMonitor()
    >>> with monitor.managed_processing():
    ...     for batch in batches:
    ...         batch_size = monitor.get_safe_batch_size(max_batch=8)
    ...         process_batch(batch[:batch_size])
    ...         if monitor.should_pause():
    ...             monitor.cool_down_pause()
"""

import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """GPU thermal states."""
    COOL = "cool"           # < 60C - Full performance
    WARM = "warm"           # 60-75C - Normal operation
    HOT = "hot"             # 75-85C - Consider reducing load
    CRITICAL = "critical"   # > 85C - Throttling likely
    UNKNOWN = "unknown"     # Can't read temperature


class ThrottleState(Enum):
    """GPU throttling states."""
    NONE = "none"                   # No throttling
    POWER_LIMIT = "power_limit"     # Power limit throttling
    THERMAL = "thermal"             # Thermal throttling
    RELIABILITY = "reliability"     # Reliability throttling
    UNKNOWN = "unknown"


@dataclass
class ThermalReading:
    """Single thermal reading from GPU."""
    timestamp: float
    temperature_celsius: float
    thermal_state: ThermalState
    throttle_state: ThrottleState = ThrottleState.NONE
    power_usage_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    clock_speed_mhz: Optional[int] = None
    clock_speed_max_mhz: Optional[int] = None


@dataclass
class ThermalProfile:
    """Thermal profile and history for a GPU."""
    device_id: int = 0
    device_name: str = "Unknown"

    # Current state
    current_temp: float = 0.0
    current_state: ThermalState = ThermalState.UNKNOWN
    is_throttling: bool = False
    throttle_reason: ThrottleState = ThrottleState.NONE

    # Thresholds (configurable)
    temp_cool: float = 60.0
    temp_warm: float = 75.0
    temp_hot: float = 85.0
    temp_critical: float = 90.0

    # History
    readings: List[ThermalReading] = field(default_factory=list)
    max_temp_observed: float = 0.0
    throttle_events: int = 0
    total_throttle_seconds: float = 0.0

    # Performance tracking
    performance_factor: float = 1.0  # 1.0 = full, 0.5 = 50%

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "current_temp": self.current_temp,
            "current_state": self.current_state.value,
            "is_throttling": self.is_throttling,
            "throttle_reason": self.throttle_reason.value,
            "max_temp_observed": self.max_temp_observed,
            "throttle_events": self.throttle_events,
            "total_throttle_seconds": self.total_throttle_seconds,
            "performance_factor": self.performance_factor,
            "readings_count": len(self.readings),
        }


class ThermalMonitor:
    """Monitors GPU thermal state and manages throttling.

    Provides real-time temperature monitoring with automatic
    adaptation of processing parameters to prevent overheating.
    """

    def __init__(
        self,
        device_id: int = 0,
        poll_interval: float = 5.0,
        max_history: int = 100,
        auto_adapt: bool = True,
    ):
        """Initialize thermal monitor.

        Args:
            device_id: GPU device ID to monitor
            poll_interval: Seconds between temperature readings
            max_history: Maximum readings to keep in history
            auto_adapt: Automatically reduce batch size when hot
        """
        self.device_id = device_id
        self.poll_interval = poll_interval
        self.max_history = max_history
        self.auto_adapt = auto_adapt

        self.profile = ThermalProfile(device_id=device_id)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[ThermalReading], None]] = []
        self._last_throttle_start: Optional[float] = None

        # Initialize device name
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect GPU and initialize profile."""
        try:
            from .gpu import get_all_gpu_info
            gpus = get_all_gpu_info()
            for gpu in gpus:
                if gpu.index == self.device_id:
                    self.profile.device_name = gpu.name
                    break
        except Exception:
            pass

    def read_temperature(self) -> Optional[ThermalReading]:
        """Read current GPU temperature.

        Returns:
            ThermalReading or None if unavailable
        """
        reading = None

        # Try NVIDIA first
        reading = self._read_nvidia_temperature()

        # Try AMD if NVIDIA failed
        if reading is None:
            reading = self._read_amd_temperature()

        # Try Intel if others failed
        if reading is None:
            reading = self._read_intel_temperature()

        if reading:
            self._update_profile(reading)

        return reading

    def _read_nvidia_temperature(self) -> Optional[ThermalReading]:
        """Read temperature from NVIDIA GPU via nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                f"--id={self.device_id}",
                "--query-gpu=temperature.gpu,power.draw,power.limit,"
                "clocks.current.graphics,clocks.max.graphics,"
                "gpu_power_profiles",
                "--format=csv,noheader,nounits"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )

            if result.returncode != 0:
                return None

            parts = [p.strip() for p in result.stdout.strip().split(",")]

            if len(parts) < 1 or parts[0] == "[N/A]":
                return None

            temp = float(parts[0])
            power = float(parts[1]) if len(parts) > 1 and parts[1] != "[N/A]" else None
            power_limit = float(parts[2]) if len(parts) > 2 and parts[2] != "[N/A]" else None
            clock = int(parts[3]) if len(parts) > 3 and parts[3] != "[N/A]" else None
            clock_max = int(parts[4]) if len(parts) > 4 and parts[4] != "[N/A]" else None

            # Determine thermal state
            thermal_state = self._classify_temperature(temp)

            # Check for throttling via separate query
            throttle_state = self._check_nvidia_throttling()

            return ThermalReading(
                timestamp=time.time(),
                temperature_celsius=temp,
                thermal_state=thermal_state,
                throttle_state=throttle_state,
                power_usage_watts=power,
                power_limit_watts=power_limit,
                clock_speed_mhz=clock,
                clock_speed_max_mhz=clock_max,
            )

        except Exception as e:
            logger.debug(f"NVIDIA temperature read failed: {e}")
            return None

    def _check_nvidia_throttling(self) -> ThrottleState:
        """Check NVIDIA GPU throttle reasons."""
        try:
            cmd = [
                "nvidia-smi",
                f"--id={self.device_id}",
                "--query-gpu=clocks_throttle_reasons.active",
                "--format=csv,noheader"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )

            if result.returncode != 0:
                return ThrottleState.UNKNOWN

            reasons = result.stdout.strip().lower()

            if "thermal" in reasons or "temp" in reasons:
                return ThrottleState.THERMAL
            elif "power" in reasons:
                return ThrottleState.POWER_LIMIT
            elif "reliability" in reasons:
                return ThrottleState.RELIABILITY
            elif reasons == "not active" or reasons == "" or "[not supported]" in reasons:
                return ThrottleState.NONE

            return ThrottleState.UNKNOWN

        except Exception:
            return ThrottleState.UNKNOWN

    def _read_amd_temperature(self) -> Optional[ThermalReading]:
        """Read temperature from AMD GPU."""
        if platform.system() != "Linux":
            # On Windows, we'd need ROCm tools
            return None

        try:
            # Try reading from hwmon (Linux)
            import glob

            hwmon_paths = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp1_input")

            for path in hwmon_paths:
                try:
                    with open(path) as f:
                        temp_milli = int(f.read().strip())
                        temp = temp_milli / 1000.0

                        return ThermalReading(
                            timestamp=time.time(),
                            temperature_celsius=temp,
                            thermal_state=self._classify_temperature(temp),
                        )
                except Exception:
                    continue

            return None

        except Exception as e:
            logger.debug(f"AMD temperature read failed: {e}")
            return None

    def _read_intel_temperature(self) -> Optional[ThermalReading]:
        """Read temperature from Intel GPU."""
        # Intel GPU temperature reading is limited
        # On Linux, might be available via i915 driver
        return None

    def _classify_temperature(self, temp: float) -> ThermalState:
        """Classify temperature into thermal state.

        Args:
            temp: Temperature in Celsius

        Returns:
            ThermalState classification
        """
        if temp < self.profile.temp_cool:
            return ThermalState.COOL
        elif temp < self.profile.temp_warm:
            return ThermalState.WARM
        elif temp < self.profile.temp_hot:
            return ThermalState.HOT
        else:
            return ThermalState.CRITICAL

    def _update_profile(self, reading: ThermalReading) -> None:
        """Update thermal profile with new reading.

        Args:
            reading: New thermal reading
        """
        self.profile.current_temp = reading.temperature_celsius
        self.profile.current_state = reading.thermal_state

        # Update max temperature
        if reading.temperature_celsius > self.profile.max_temp_observed:
            self.profile.max_temp_observed = reading.temperature_celsius

        # Track throttling
        is_throttling = reading.throttle_state not in (ThrottleState.NONE, ThrottleState.UNKNOWN)

        if is_throttling and not self.profile.is_throttling:
            # Started throttling
            self.profile.throttle_events += 1
            self._last_throttle_start = time.time()
            logger.warning(
                f"GPU thermal throttling detected: {reading.temperature_celsius}C "
                f"({reading.throttle_state.value})"
            )

        if not is_throttling and self.profile.is_throttling:
            # Stopped throttling
            if self._last_throttle_start:
                duration = time.time() - self._last_throttle_start
                self.profile.total_throttle_seconds += duration
                logger.info(f"GPU throttling ended (duration: {duration:.1f}s)")
            self._last_throttle_start = None

        self.profile.is_throttling = is_throttling
        self.profile.throttle_reason = reading.throttle_state

        # Update performance factor based on thermal state
        if reading.thermal_state == ThermalState.COOL:
            self.profile.performance_factor = 1.0
        elif reading.thermal_state == ThermalState.WARM:
            self.profile.performance_factor = 0.9
        elif reading.thermal_state == ThermalState.HOT:
            self.profile.performance_factor = 0.7
        else:  # CRITICAL
            self.profile.performance_factor = 0.5

        # Add to history
        self.profile.readings.append(reading)
        if len(self.profile.readings) > self.max_history:
            self.profile.readings.pop(0)

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.debug(f"Thermal callback failed: {e}")

    def start_monitoring(self) -> None:
        """Start background temperature monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ThermalMonitor"
        )
        self._monitor_thread.start()
        logger.debug("Thermal monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background temperature monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.debug("Thermal monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self.read_temperature()
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")

            time.sleep(self.poll_interval)

    def add_callback(self, callback: Callable[[ThermalReading], None]) -> None:
        """Add callback for thermal events.

        Args:
            callback: Function to call on each reading
        """
        self._callbacks.append(callback)

    def get_safe_batch_size(
        self,
        max_batch_size: int,
        min_batch_size: int = 1,
    ) -> int:
        """Get safe batch size based on thermal state.

        Args:
            max_batch_size: Maximum desired batch size
            min_batch_size: Minimum allowed batch size

        Returns:
            Adjusted batch size
        """
        if not self.auto_adapt:
            return max_batch_size

        # Apply performance factor
        adjusted = int(max_batch_size * self.profile.performance_factor)

        # Ensure within bounds
        return max(min_batch_size, min(max_batch_size, adjusted))

    def should_pause(self) -> bool:
        """Check if processing should pause for cooling.

        Returns:
            True if GPU is critically hot
        """
        return (
            self.profile.current_state == ThermalState.CRITICAL or
            self.profile.is_throttling
        )

    def cool_down_pause(
        self,
        target_temp: Optional[float] = None,
        max_wait_seconds: float = 300.0,
    ) -> float:
        """Pause until GPU cools down.

        Args:
            target_temp: Target temperature (default: warm threshold)
            max_wait_seconds: Maximum time to wait

        Returns:
            Seconds waited
        """
        if target_temp is None:
            target_temp = self.profile.temp_warm

        logger.info(
            f"Pausing for GPU cool-down (current: {self.profile.current_temp}C, "
            f"target: {target_temp}C)"
        )

        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            reading = self.read_temperature()

            if reading and reading.temperature_celsius <= target_temp:
                break

            time.sleep(5.0)

        wait_time = time.time() - start_time
        logger.info(f"Cool-down complete after {wait_time:.1f}s")

        return wait_time

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary.

        Returns:
            Summary dictionary
        """
        summary = self.profile.to_dict()

        if self.profile.readings:
            temps = [r.temperature_celsius for r in self.profile.readings]
            summary["avg_temp"] = sum(temps) / len(temps)
            summary["min_temp"] = min(temps)
        else:
            summary["avg_temp"] = 0
            summary["min_temp"] = 0

        return summary

    class ManagedProcessing:
        """Context manager for thermal-aware processing."""

        def __init__(self, monitor: "ThermalMonitor"):
            self.monitor = monitor

        def __enter__(self):
            self.monitor.start_monitoring()
            return self.monitor

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.monitor.stop_monitoring()
            return False

    def managed_processing(self) -> ManagedProcessing:
        """Context manager for thermal-aware processing.

        Example:
            >>> with monitor.managed_processing():
            ...     process_video()
        """
        return self.ManagedProcessing(self)


def get_gpu_temperature(device_id: int = 0) -> Optional[float]:
    """Convenience function to get GPU temperature.

    Args:
        device_id: GPU device ID

    Returns:
        Temperature in Celsius or None
    """
    monitor = ThermalMonitor(device_id=device_id)
    reading = monitor.read_temperature()
    return reading.temperature_celsius if reading else None


def is_gpu_throttling(device_id: int = 0) -> bool:
    """Convenience function to check if GPU is throttling.

    Args:
        device_id: GPU device ID

    Returns:
        True if throttling
    """
    monitor = ThermalMonitor(device_id=device_id)
    reading = monitor.read_temperature()
    return reading is not None and reading.throttle_state not in (
        ThrottleState.NONE, ThrottleState.UNKNOWN
    )
