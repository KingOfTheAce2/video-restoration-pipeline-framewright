"""Processing Safeguards for Long-Running Video Restoration Jobs.

Integrates thermal monitoring, disk space checking, and checkpoint
management to ensure stable and recoverable processing.

Features:
- Pre-flight disk space validation
- Real-time disk space monitoring during processing
- GPU thermal monitoring with auto-adaptation
- Automatic pause on critical conditions
- Graceful degradation (reduce batch size when constrained)
- Progress notifications

Example:
    >>> safeguards = ProcessingSafeguards(
    ...     video_path="video.mp4",
    ...     output_dir="./output",
    ... )
    >>> with safeguards.managed_processing() as context:
    ...     for frame in frames:
    ...         batch_size = context.get_safe_batch_size(8)
    ...         context.check_conditions()  # May pause if needed
    ...         process_frame(frame)
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SafeguardStatus(Enum):
    """Status of processing safeguards."""
    OK = "ok"                       # All systems nominal
    WARNING = "warning"             # Minor issues, continue with caution
    DEGRADED = "degraded"           # Reduced performance mode
    CRITICAL = "critical"           # Should pause/stop
    FAILED = "failed"               # Cannot continue


class ConstraintType(Enum):
    """Types of processing constraints."""
    DISK_SPACE = "disk_space"
    GPU_THERMAL = "gpu_thermal"
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"


@dataclass
class Constraint:
    """A processing constraint that was triggered."""
    type: ConstraintType
    severity: SafeguardStatus
    message: str
    recommendation: str
    value: float = 0.0
    threshold: float = 0.0


@dataclass
class SafeguardConfig:
    """Configuration for processing safeguards."""
    # Disk space
    min_disk_space_gb: float = 10.0         # Minimum free space to continue
    disk_warning_gb: float = 20.0           # Warning threshold
    disk_check_interval: float = 60.0       # Seconds between checks

    # GPU thermal
    temp_warning: float = 75.0              # Warning temperature
    temp_critical: float = 85.0             # Critical temperature
    thermal_check_interval: float = 10.0    # Seconds between checks
    cool_down_target: float = 70.0          # Target temp after cool-down

    # GPU memory
    min_vram_mb: int = 500                  # Minimum free VRAM
    vram_warning_percent: float = 85.0      # Warning usage percent

    # Performance adaptation
    enable_auto_adaptation: bool = True     # Auto-reduce batch size
    min_batch_size: int = 1                 # Minimum batch size
    batch_reduction_step: float = 0.5       # Reduce by this factor

    # Notifications
    enable_notifications: bool = True       # Send desktop notifications
    notify_on_pause: bool = True
    notify_on_complete: bool = True


@dataclass
class ProcessingContext:
    """Context provided to processing loop."""
    safeguards: "ProcessingSafeguards"

    # Current state
    current_batch_size: int = 8
    max_batch_size: int = 8
    performance_factor: float = 1.0

    # Statistics
    frames_processed: int = 0
    pauses_count: int = 0
    total_pause_seconds: float = 0.0
    constraints_hit: List[Constraint] = field(default_factory=list)

    def get_safe_batch_size(self, max_batch: int) -> int:
        """Get safe batch size based on current conditions.

        Args:
            max_batch: Maximum desired batch size

        Returns:
            Safe batch size
        """
        self.max_batch_size = max_batch
        self.current_batch_size = int(max_batch * self.performance_factor)
        self.current_batch_size = max(
            self.safeguards.config.min_batch_size,
            self.current_batch_size
        )
        return self.current_batch_size

    def check_conditions(self) -> SafeguardStatus:
        """Check all conditions and handle any issues.

        May pause processing if critical conditions detected.

        Returns:
            Current status
        """
        return self.safeguards.check_all()

    def report_progress(self, frames: int) -> None:
        """Report progress for tracking.

        Args:
            frames: Number of frames just processed
        """
        self.frames_processed += frames


class ProcessingSafeguards:
    """Safeguards for long-running video restoration.

    Monitors disk space, GPU temperature, and memory to ensure
    stable processing with automatic recovery from issues.
    """

    def __init__(
        self,
        video_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config: Optional[SafeguardConfig] = None,
    ):
        """Initialize safeguards.

        Args:
            video_path: Input video path (for estimation)
            output_dir: Output directory (for disk monitoring)
            config: Safeguard configuration
        """
        self.video_path = Path(video_path) if video_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or SafeguardConfig()

        self._status = SafeguardStatus.OK
        self._constraints: List[Constraint] = []
        self._context: Optional[ProcessingContext] = None

        self._last_disk_check = 0.0
        self._last_thermal_check = 0.0

        # Monitoring objects (lazy loaded)
        self._disk_monitor = None
        self._thermal_monitor = None

    def preflight_check(self) -> Tuple[bool, List[Constraint]]:
        """Run pre-flight checks before processing.

        Returns:
            Tuple of (can_proceed, list of constraints)
        """
        constraints = []

        # Check disk space
        if self.output_dir:
            disk_constraint = self._check_disk_space()
            if disk_constraint:
                constraints.append(disk_constraint)

        # Check GPU availability and temperature
        thermal_constraint = self._check_thermal()
        if thermal_constraint:
            constraints.append(thermal_constraint)

        # Check VRAM
        vram_constraint = self._check_vram()
        if vram_constraint:
            constraints.append(vram_constraint)

        # Determine if we can proceed
        can_proceed = all(
            c.severity not in (SafeguardStatus.CRITICAL, SafeguardStatus.FAILED)
            for c in constraints
        )

        return can_proceed, constraints

    def check_all(self) -> SafeguardStatus:
        """Check all safeguards and return current status.

        Returns:
            Current SafeguardStatus
        """
        self._constraints = []
        current_time = time.time()

        # Disk space check (less frequent)
        if current_time - self._last_disk_check >= self.config.disk_check_interval:
            constraint = self._check_disk_space()
            if constraint:
                self._constraints.append(constraint)
            self._last_disk_check = current_time

        # Thermal check (more frequent)
        if current_time - self._last_thermal_check >= self.config.thermal_check_interval:
            constraint = self._check_thermal()
            if constraint:
                self._constraints.append(constraint)
            self._last_thermal_check = current_time

        # Determine overall status
        if not self._constraints:
            self._status = SafeguardStatus.OK
        else:
            severities = [c.severity for c in self._constraints]
            if SafeguardStatus.CRITICAL in severities:
                self._status = SafeguardStatus.CRITICAL
                self._handle_critical()
            elif SafeguardStatus.WARNING in severities:
                self._status = SafeguardStatus.WARNING
                self._adapt_performance(0.8)
            else:
                self._status = SafeguardStatus.DEGRADED
                self._adapt_performance(0.6)

        return self._status

    def _check_disk_space(self) -> Optional[Constraint]:
        """Check available disk space.

        Returns:
            Constraint if disk space is low, None otherwise
        """
        if not self.output_dir:
            return None

        try:
            from ..utils.disk import get_disk_usage

            usage = get_disk_usage(self.output_dir)
            free_gb = usage.free_gb

            if free_gb < self.config.min_disk_space_gb:
                return Constraint(
                    type=ConstraintType.DISK_SPACE,
                    severity=SafeguardStatus.CRITICAL,
                    message=f"Disk space critically low: {free_gb:.1f}GB free",
                    recommendation="Free up disk space or change output directory",
                    value=free_gb,
                    threshold=self.config.min_disk_space_gb,
                )

            if free_gb < self.config.disk_warning_gb:
                return Constraint(
                    type=ConstraintType.DISK_SPACE,
                    severity=SafeguardStatus.WARNING,
                    message=f"Disk space low: {free_gb:.1f}GB free",
                    recommendation="Consider freeing up disk space",
                    value=free_gb,
                    threshold=self.config.disk_warning_gb,
                )

        except Exception as e:
            logger.debug(f"Disk check failed: {e}")

        return None

    def _check_thermal(self) -> Optional[Constraint]:
        """Check GPU temperature.

        Returns:
            Constraint if temperature is high, None otherwise
        """
        try:
            from ..utils.thermal_monitor import ThermalMonitor, ThermalState

            if self._thermal_monitor is None:
                self._thermal_monitor = ThermalMonitor()

            reading = self._thermal_monitor.read_temperature()
            if reading is None:
                return None

            temp = reading.temperature_celsius

            if temp >= self.config.temp_critical:
                return Constraint(
                    type=ConstraintType.GPU_THERMAL,
                    severity=SafeguardStatus.CRITICAL,
                    message=f"GPU critically hot: {temp:.0f}°C",
                    recommendation="Processing will pause for cool-down",
                    value=temp,
                    threshold=self.config.temp_critical,
                )

            if temp >= self.config.temp_warning:
                return Constraint(
                    type=ConstraintType.GPU_THERMAL,
                    severity=SafeguardStatus.WARNING,
                    message=f"GPU running hot: {temp:.0f}°C",
                    recommendation="Reducing batch size to lower temperature",
                    value=temp,
                    threshold=self.config.temp_warning,
                )

            # Check for throttling
            if reading.throttle_state.value not in ("none", "unknown"):
                return Constraint(
                    type=ConstraintType.GPU_THERMAL,
                    severity=SafeguardStatus.DEGRADED,
                    message=f"GPU throttling: {reading.throttle_state.value}",
                    recommendation="Performance may be reduced",
                    value=temp,
                    threshold=0,
                )

        except Exception as e:
            logger.debug(f"Thermal check failed: {e}")

        return None

    def _check_vram(self) -> Optional[Constraint]:
        """Check GPU VRAM usage.

        Returns:
            Constraint if VRAM is low, None otherwise
        """
        try:
            from ..utils.gpu import get_gpu_memory_info

            info = get_gpu_memory_info()
            if info is None:
                return None

            free_mb = info["free_mb"]
            usage_percent = info["usage_percent"]

            if free_mb < self.config.min_vram_mb:
                return Constraint(
                    type=ConstraintType.GPU_MEMORY,
                    severity=SafeguardStatus.CRITICAL,
                    message=f"VRAM critically low: {free_mb}MB free",
                    recommendation="Reduce batch size or close other applications",
                    value=free_mb,
                    threshold=self.config.min_vram_mb,
                )

            if usage_percent >= self.config.vram_warning_percent:
                return Constraint(
                    type=ConstraintType.GPU_MEMORY,
                    severity=SafeguardStatus.WARNING,
                    message=f"VRAM usage high: {usage_percent:.0f}%",
                    recommendation="Consider reducing batch size",
                    value=usage_percent,
                    threshold=self.config.vram_warning_percent,
                )

        except Exception as e:
            logger.debug(f"VRAM check failed: {e}")

        return None

    def _handle_critical(self) -> None:
        """Handle critical status by pausing or adapting."""
        if self._context is None:
            return

        # Find the critical constraint
        critical = next(
            (c for c in self._constraints if c.severity == SafeguardStatus.CRITICAL),
            None
        )

        if critical is None:
            return

        logger.warning(f"Critical constraint: {critical.message}")

        if critical.type == ConstraintType.GPU_THERMAL:
            # Cool-down pause
            self._cool_down_pause()

        elif critical.type == ConstraintType.DISK_SPACE:
            # Can't auto-recover from disk space
            self._notify("Disk Space Critical", critical.message)

        elif critical.type == ConstraintType.GPU_MEMORY:
            # Reduce batch size aggressively
            self._adapt_performance(0.25)
            # Clear GPU cache
            try:
                from ..utils.gpu_memory_optimizer import clear_gpu_memory
                clear_gpu_memory()
            except Exception:
                pass

    def _cool_down_pause(self) -> float:
        """Pause for GPU cool-down.

        Returns:
            Seconds waited
        """
        if self._context:
            self._context.pauses_count += 1

        self._notify("GPU Cool-Down", "Pausing for GPU to cool down")

        logger.info("Starting cool-down pause")
        start = time.time()
        target_temp = self.config.cool_down_target

        while True:
            if self._thermal_monitor:
                reading = self._thermal_monitor.read_temperature()
                if reading and reading.temperature_celsius <= target_temp:
                    break

            if time.time() - start > 300:  # Max 5 minutes
                logger.warning("Cool-down timeout")
                break

            time.sleep(5)

        wait_time = time.time() - start

        if self._context:
            self._context.total_pause_seconds += wait_time

        logger.info(f"Cool-down complete after {wait_time:.0f}s")
        return wait_time

    def _adapt_performance(self, factor: float) -> None:
        """Adapt performance by reducing batch size.

        Args:
            factor: Factor to multiply current performance (0-1)
        """
        if not self.config.enable_auto_adaptation:
            return

        if self._context:
            self._context.performance_factor *= factor
            self._context.performance_factor = max(
                0.1, self._context.performance_factor
            )

            new_batch = int(
                self._context.max_batch_size * self._context.performance_factor
            )
            new_batch = max(self.config.min_batch_size, new_batch)

            if new_batch != self._context.current_batch_size:
                logger.info(
                    f"Adapting batch size: {self._context.current_batch_size} -> {new_batch}"
                )
                self._context.current_batch_size = new_batch

    def _notify(self, title: str, message: str) -> None:
        """Send desktop notification.

        Args:
            title: Notification title
            message: Notification message
        """
        if not self.config.enable_notifications:
            return

        try:
            from ..cli_advanced import send_notification
            send_notification(title, message)
        except Exception:
            pass

    @contextmanager
    def managed_processing(self):
        """Context manager for managed processing.

        Yields:
            ProcessingContext for use in processing loop
        """
        self._context = ProcessingContext(
            safeguards=self,
            performance_factor=1.0,
        )

        # Run preflight
        can_proceed, constraints = self.preflight_check()

        if not can_proceed:
            for c in constraints:
                logger.error(f"Preflight failed: {c.message}")
            raise RuntimeError("Preflight checks failed")

        for c in constraints:
            if c.severity == SafeguardStatus.WARNING:
                logger.warning(f"Preflight warning: {c.message}")

        try:
            yield self._context
        finally:
            # Cleanup
            if self._thermal_monitor:
                try:
                    self._thermal_monitor.stop_monitoring()
                except Exception:
                    pass

            # Final notification
            if self.config.notify_on_complete:
                self._notify(
                    "Processing Complete",
                    f"Processed {self._context.frames_processed} frames"
                )

            self._context = None

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary.

        Returns:
            Summary dictionary
        """
        summary = {
            "status": self._status.value,
            "constraints": [
                {
                    "type": c.type.value,
                    "severity": c.severity.value,
                    "message": c.message,
                }
                for c in self._constraints
            ],
        }

        if self._context:
            summary.update({
                "frames_processed": self._context.frames_processed,
                "current_batch_size": self._context.current_batch_size,
                "performance_factor": self._context.performance_factor,
                "pauses_count": self._context.pauses_count,
                "total_pause_seconds": self._context.total_pause_seconds,
            })

        return summary


def create_safeguards(
    video_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> ProcessingSafeguards:
    """Create processing safeguards with custom config.

    Args:
        video_path: Input video path
        output_dir: Output directory
        **kwargs: Config overrides

    Returns:
        ProcessingSafeguards instance
    """
    config = SafeguardConfig(**kwargs)
    return ProcessingSafeguards(
        video_path=video_path,
        output_dir=output_dir,
        config=config,
    )
