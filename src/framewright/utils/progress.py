"""Rich progress reporting system for FrameWright video restoration pipeline.

Provides advanced progress tracking with GPU monitoring, ETA calculations,
and both terminal and log file output support.
"""
from __future__ import annotations

import logging
import subprocess
import shutil
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO, Tuple, Union

logger = logging.getLogger(__name__)


# Check for optional dependencies
RICH_AVAILABLE = False
PYNVML_AVAILABLE = False

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    logger.debug("Rich library not available, using fallback progress display")

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    logger.debug("pynvml not available, using nvidia-smi fallback for GPU monitoring")


class ProgressOutputMode(Enum):
    """Output modes for progress reporting."""

    TERMINAL = "terminal"
    LOG = "log"
    BOTH = "both"
    SILENT = "silent"


@dataclass
class GPUMetrics:
    """Real-time GPU metrics."""

    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_celsius: Optional[float] = None
    available: bool = False

    @property
    def memory_used_gb(self) -> float:
        """Get memory used in GB."""
        return self.memory_used_mb / 1024.0

    @property
    def memory_total_gb(self) -> float:
        """Get total memory in GB."""
        return self.memory_total_mb / 1024.0

    @property
    def memory_percent(self) -> float:
        """Get memory usage percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0


@dataclass
class StageInfo:
    """Information about a processing stage."""

    name: str
    total_frames: int
    completed_frames: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    stage_index: int = 0
    total_stages: int = 1

    @property
    def is_complete(self) -> bool:
        """Check if stage is complete."""
        return self.completed_frames >= self.total_frames

    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_frames == 0:
            return 100.0
        return (self.completed_frames / self.total_frames) * 100.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def fps(self) -> float:
        """Get processing speed in frames per second."""
        elapsed = self.elapsed_seconds
        if elapsed == 0 or self.completed_frames == 0:
            return 0.0
        return self.completed_frames / elapsed

    @property
    def eta_seconds(self) -> Optional[float]:
        """Get estimated time to completion in seconds."""
        if self.fps == 0:
            return None
        remaining = self.total_frames - self.completed_frames
        return remaining / self.fps


class GPUMonitor:
    """Monitor GPU utilization and memory usage.

    Supports both pynvml (preferred) and nvidia-smi fallback.
    Provides graceful fallback for CPU-only systems.
    """

    def __init__(self, device_id: int = 0, poll_interval: float = 1.0):
        """Initialize GPU monitor.

        Args:
            device_id: GPU device index to monitor
            poll_interval: How often to poll GPU stats (seconds)
        """
        self.device_id = device_id
        self.poll_interval = poll_interval
        self._handle: Optional[Any] = None
        self._initialized = False
        self._lock = threading.Lock()
        self._cached_metrics = GPUMetrics()
        self._last_poll_time = 0.0
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize the GPU monitoring backend."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                self._initialized = True
                logger.debug(f"GPU monitoring initialized with pynvml for device {self.device_id}")
                return
            except Exception as e:
                logger.debug(f"Failed to initialize pynvml: {e}")

        # Fallback to nvidia-smi
        if shutil.which("nvidia-smi") is not None:
            self._initialized = True
            logger.debug("GPU monitoring using nvidia-smi fallback")
        else:
            logger.debug("No GPU monitoring available (CPU-only system)")

    def get_metrics(self) -> GPUMetrics:
        """Get current GPU metrics.

        Returns:
            GPUMetrics with current utilization and memory stats
        """
        current_time = time.time()

        # Use cached metrics if polled recently
        if current_time - self._last_poll_time < self.poll_interval:
            return self._cached_metrics

        with self._lock:
            if not self._initialized:
                return GPUMetrics(available=False)

            try:
                if PYNVML_AVAILABLE and self._handle is not None:
                    metrics = self._get_metrics_pynvml()
                else:
                    metrics = self._get_metrics_nvidia_smi()

                self._cached_metrics = metrics
                self._last_poll_time = current_time
                return metrics

            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")
                return GPUMetrics(available=False)

    def _get_metrics_pynvml(self) -> GPUMetrics:
        """Get metrics using pynvml."""
        if not PYNVML_AVAILABLE or self._handle is None:
            return GPUMetrics(available=False)

        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

            temperature = None
            try:
                temperature = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass

            return GPUMetrics(
                utilization_percent=float(utilization.gpu),
                memory_used_mb=memory.used / (1024 * 1024),
                memory_total_mb=memory.total / (1024 * 1024),
                temperature_celsius=temperature,
                available=True,
            )
        except Exception as e:
            logger.debug(f"pynvml metrics failed: {e}")
            return GPUMetrics(available=False)

    def _get_metrics_nvidia_smi(self) -> GPUMetrics:
        """Get metrics using nvidia-smi command."""
        try:
            cmd = [
                "nvidia-smi",
                f"--id={self.device_id}",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return GPUMetrics(available=False)

            parts = [p.strip() for p in result.stdout.strip().split(",")]

            utilization = float(parts[0]) if parts[0] != "[N/A]" else 0.0
            memory_used = float(parts[1]) if parts[1] != "[N/A]" else 0.0
            memory_total = float(parts[2]) if parts[2] != "[N/A]" else 0.0
            temperature = float(parts[3]) if parts[3] != "[N/A]" else None

            return GPUMetrics(
                utilization_percent=utilization,
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                temperature_celsius=temperature,
                available=True,
            )

        except Exception as e:
            logger.debug(f"nvidia-smi metrics failed: {e}")
            return GPUMetrics(available=False)

    def start_background_monitoring(self) -> None:
        """Start background thread for continuous GPU monitoring."""
        if self._monitoring_thread is not None:
            return

        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._stop_event.set()
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            self.get_metrics()  # Updates cache
            self._stop_event.wait(self.poll_interval)

    def shutdown(self) -> None:
        """Clean up resources."""
        self.stop_background_monitoring()
        if PYNVML_AVAILABLE and self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class ProgressCallback(ABC):
    """Abstract base class for progress callbacks."""

    @abstractmethod
    def on_stage_start(self, stage: StageInfo) -> None:
        """Called when a stage starts."""
        pass

    @abstractmethod
    def on_progress_update(
        self,
        stage: StageInfo,
        gpu_metrics: Optional[GPUMetrics] = None,
    ) -> None:
        """Called on progress update."""
        pass

    @abstractmethod
    def on_stage_complete(self, stage: StageInfo) -> None:
        """Called when a stage completes."""
        pass

    @abstractmethod
    def on_pipeline_complete(self, stages: List[StageInfo]) -> None:
        """Called when entire pipeline completes."""
        pass


class LogFileCallback(ProgressCallback):
    """Progress callback that writes to a log file."""

    def __init__(
        self,
        log_path: Union[str, Path],
        include_gpu_metrics: bool = True,
    ):
        """Initialize log file callback.

        Args:
            log_path: Path to log file
            include_gpu_metrics: Whether to include GPU metrics
        """
        self.log_path = Path(log_path)
        self.include_gpu_metrics = include_gpu_metrics
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, message: str) -> None:
        """Write message to log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def on_stage_start(self, stage: StageInfo) -> None:
        """Log stage start."""
        self._write(
            f"STAGE_START: {stage.name} "
            f"({stage.stage_index + 1}/{stage.total_stages}) "
            f"- {stage.total_frames} frames"
        )

    def on_progress_update(
        self,
        stage: StageInfo,
        gpu_metrics: Optional[GPUMetrics] = None,
    ) -> None:
        """Log progress update."""
        msg = (
            f"PROGRESS: {stage.name} - "
            f"{stage.completed_frames}/{stage.total_frames} frames "
            f"({stage.progress_percent:.1f}%) - "
            f"{stage.fps:.2f} fps"
        )

        if stage.eta_seconds is not None:
            msg += f" - ETA: {self._format_time(stage.eta_seconds)}"

        if self.include_gpu_metrics and gpu_metrics and gpu_metrics.available:
            msg += (
                f" - GPU: {gpu_metrics.utilization_percent:.0f}% "
                f"VRAM: {gpu_metrics.memory_used_gb:.1f}/{gpu_metrics.memory_total_gb:.1f}GB"
            )

        self._write(msg)

    def on_stage_complete(self, stage: StageInfo) -> None:
        """Log stage completion."""
        self._write(
            f"STAGE_COMPLETE: {stage.name} - "
            f"{stage.total_frames} frames in "
            f"{self._format_time(stage.elapsed_seconds)} "
            f"({stage.fps:.2f} fps avg)"
        )

    def on_pipeline_complete(self, stages: List[StageInfo]) -> None:
        """Log pipeline completion."""
        total_frames = sum(s.total_frames for s in stages)
        total_time = sum(s.elapsed_seconds for s in stages)
        self._write(
            f"PIPELINE_COMPLETE: {len(stages)} stages, "
            f"{total_frames} total frames in "
            f"{self._format_time(total_time)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class FallbackProgress:
    """Fallback progress display when rich is not available.

    Uses simple print statements with tqdm-style progress bars.
    """

    def __init__(
        self,
        video_name: str = "",
        output_mode: ProgressOutputMode = ProgressOutputMode.TERMINAL,
    ):
        """Initialize fallback progress display.

        Args:
            video_name: Name of video being processed
            output_mode: Where to output progress
        """
        self.video_name = video_name
        self.output_mode = output_mode
        self.current_stage: Optional[StageInfo] = None
        self.gpu_monitor = GPUMonitor()
        self._last_print_time = 0.0
        self._print_interval = 0.5  # Update every 0.5 seconds

    def start_stage(
        self,
        name: str,
        total_frames: int,
        stage_index: int = 0,
        total_stages: int = 1,
    ) -> StageInfo:
        """Start a new processing stage.

        Args:
            name: Stage name
            total_frames: Total frames to process
            stage_index: Current stage index (0-based)
            total_stages: Total number of stages

        Returns:
            StageInfo object for the stage
        """
        self.current_stage = StageInfo(
            name=name,
            total_frames=total_frames,
            start_time=time.time(),
            stage_index=stage_index,
            total_stages=total_stages,
        )

        if self.output_mode in (ProgressOutputMode.TERMINAL, ProgressOutputMode.BOTH):
            print(f"\nProcessing: {self.video_name}")
            print(f"Stage: {name} ({stage_index + 1}/{total_stages})")

        return self.current_stage

    def update(self, completed: int) -> None:
        """Update progress.

        Args:
            completed: Number of frames completed
        """
        if self.current_stage is None:
            return

        self.current_stage.completed_frames = completed

        current_time = time.time()
        if current_time - self._last_print_time < self._print_interval:
            return

        self._last_print_time = current_time

        if self.output_mode in (ProgressOutputMode.TERMINAL, ProgressOutputMode.BOTH):
            self._print_progress()

    def _print_progress(self) -> None:
        """Print current progress to terminal."""
        if self.current_stage is None:
            return

        stage = self.current_stage
        total = stage.total_frames
        completed = stage.completed_frames
        percent = stage.progress_percent

        # Build progress bar
        bar_width = 20
        filled = int(bar_width * percent / 100)
        bar = "#" * filled + "-" * (bar_width - filled)

        # Build status line
        status = f"Progress: [{bar}] {percent:.0f}% ({completed}/{total} frames)"

        # Add speed and ETA
        fps = stage.fps
        if fps > 0:
            status += f" | {fps:.1f} fps"
            eta = stage.eta_seconds
            if eta is not None:
                status += f" | ETA: {self._format_eta(eta)}"

        # Add GPU metrics
        gpu = self.gpu_monitor.get_metrics()
        if gpu.available:
            status += (
                f" | GPU: {gpu.utilization_percent:.0f}% "
                f"| VRAM: {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB"
            )

        # Print with carriage return to update in place
        print(f"\r{status}", end="", flush=True)

    def complete_stage(self) -> None:
        """Mark current stage as complete."""
        if self.current_stage is None:
            return

        self.current_stage.end_time = time.time()

        if self.output_mode in (ProgressOutputMode.TERMINAL, ProgressOutputMode.BOTH):
            print()  # New line after progress bar
            print(
                f"Stage complete: {self.current_stage.elapsed_seconds:.1f}s "
                f"({self.current_stage.fps:.2f} fps avg)"
            )

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format ETA as human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class RichProgress:
    """Rich progress display with GPU monitoring.

    Provides a beautiful terminal UI with:
    - Current stage name and progress
    - Progress bar with percentage
    - Frame count (current/total)
    - Processing speed (fps)
    - ETA based on current speed
    - GPU utilization percentage
    - VRAM usage (used/total GB)
    """

    def __init__(
        self,
        video_name: str = "",
        device_id: int = 0,
        output_mode: ProgressOutputMode = ProgressOutputMode.TERMINAL,
        log_path: Optional[Union[str, Path]] = None,
        refresh_rate: float = 10.0,
    ):
        """Initialize rich progress display.

        Args:
            video_name: Name of video being processed
            device_id: GPU device ID to monitor
            output_mode: Where to output progress
            log_path: Optional path for log file output
            refresh_rate: Terminal refresh rate in Hz
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is required for RichProgress. "
                "Install with: pip install rich"
            )

        self.video_name = video_name
        self.output_mode = output_mode
        self.refresh_rate = refresh_rate

        self.console = Console()
        self.gpu_monitor = GPUMonitor(device_id=device_id)
        self.gpu_monitor.start_background_monitoring()

        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_id: Optional[TaskID] = None
        self.current_stage: Optional[StageInfo] = None
        self.stages: List[StageInfo] = []

        # Setup log callback if needed
        self.log_callback: Optional[LogFileCallback] = None
        if log_path and output_mode in (ProgressOutputMode.LOG, ProgressOutputMode.BOTH):
            self.log_callback = LogFileCallback(log_path)

    def _create_progress(self) -> Progress:
        """Create rich Progress instance with custom columns."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total} frames)"),
            TextColumn("[cyan]{task.fields[fps]:.1f} fps"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
        )

    def _create_display(self) -> Group:
        """Create the full display group with GPU stats."""
        if self._progress is None:
            return Group()

        gpu = self.gpu_monitor.get_metrics()

        # Create GPU status table
        gpu_table = Table.grid(padding=1)
        gpu_table.add_column(style="bold")
        gpu_table.add_column()

        if gpu.available:
            gpu_table.add_row(
                "GPU:",
                f"[green]{gpu.utilization_percent:.0f}%[/green]",
            )
            gpu_table.add_row(
                "VRAM:",
                f"[yellow]{gpu.memory_used_gb:.1f}[/yellow]/"
                f"[dim]{gpu.memory_total_gb:.1f}[/dim] GB",
            )
            if gpu.temperature_celsius is not None:
                temp_color = "green" if gpu.temperature_celsius < 70 else "yellow"
                if gpu.temperature_celsius >= 85:
                    temp_color = "red"
                gpu_table.add_row(
                    "Temp:",
                    f"[{temp_color}]{gpu.temperature_celsius:.0f}C[/{temp_color}]",
                )
        else:
            gpu_table.add_row("GPU:", "[dim]CPU-only mode[/dim]")

        # Create header
        header = Text()
        header.append("Processing: ", style="bold")
        header.append(self.video_name, style="cyan")

        if self.current_stage:
            header.append(" | Stage: ", style="bold")
            header.append(
                f"{self.current_stage.name} "
                f"({self.current_stage.stage_index + 1}/{self.current_stage.total_stages})",
                style="magenta",
            )

        return Group(
            Panel(
                Group(header, "", self._progress, "", gpu_table),
                title="[bold]FrameWright[/bold]",
                border_style="blue",
            )
        )

    @contextmanager
    def live_display(self) -> Iterator["RichProgress"]:
        """Context manager for live display updates.

        Yields:
            Self for method chaining
        """
        self._progress = self._create_progress()

        with Live(
            self._create_display(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            transient=False,
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None
                self.gpu_monitor.stop_background_monitoring()

    def start_stage(
        self,
        name: str,
        total_frames: int,
        stage_index: int = 0,
        total_stages: int = 1,
    ) -> StageInfo:
        """Start a new processing stage.

        Args:
            name: Stage name (e.g., "Upscaling", "Face Restoration")
            total_frames: Total frames to process
            stage_index: Current stage index (0-based)
            total_stages: Total number of stages

        Returns:
            StageInfo object for the stage
        """
        self.current_stage = StageInfo(
            name=name,
            total_frames=total_frames,
            start_time=time.time(),
            stage_index=stage_index,
            total_stages=total_stages,
        )

        if self._progress is not None:
            # Remove old task if exists
            if self._task_id is not None:
                try:
                    self._progress.remove_task(self._task_id)
                except Exception:
                    pass

            self._task_id = self._progress.add_task(
                name,
                total=total_frames,
                fps=0.0,
            )

        if self.log_callback:
            self.log_callback.on_stage_start(self.current_stage)

        self._update_display()
        return self.current_stage

    def update(self, completed: int) -> None:
        """Update progress for current stage.

        Args:
            completed: Number of frames completed
        """
        if self.current_stage is None:
            return

        self.current_stage.completed_frames = completed
        fps = self.current_stage.fps

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=completed,
                fps=fps,
            )

        self._update_display()

        # Log occasionally (not every frame)
        if self.log_callback and completed % max(1, self.current_stage.total_frames // 20) == 0:
            self.log_callback.on_progress_update(
                self.current_stage,
                self.gpu_monitor.get_metrics(),
            )

    def advance(self, amount: int = 1) -> None:
        """Advance progress by specified amount.

        Args:
            amount: Number of frames to advance by
        """
        if self.current_stage is not None:
            self.update(self.current_stage.completed_frames + amount)

    def complete_stage(self) -> None:
        """Mark current stage as complete."""
        if self.current_stage is None:
            return

        self.current_stage.end_time = time.time()
        self.current_stage.completed_frames = self.current_stage.total_frames
        self.stages.append(self.current_stage)

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.current_stage.total_frames,
                fps=self.current_stage.fps,
            )

        if self.log_callback:
            self.log_callback.on_stage_complete(self.current_stage)

        self._update_display()

    def complete_pipeline(self) -> None:
        """Mark entire pipeline as complete."""
        if self.log_callback:
            self.log_callback.on_pipeline_complete(self.stages)

        self.gpu_monitor.stop_background_monitoring()

    def _update_display(self) -> None:
        """Update the live display."""
        if self._live is not None:
            self._live.update(self._create_display())


class ProgressTracker:
    """Track progress across multiple stages in a pipeline.

    Provides:
    - Multi-stage progress tracking
    - Overall ETA calculation
    - Nested progress support (e.g., batch within stage)
    - Dual output to terminal and log files
    """

    def __init__(
        self,
        video_name: str = "",
        total_stages: int = 1,
        device_id: int = 0,
        output_mode: ProgressOutputMode = ProgressOutputMode.TERMINAL,
        log_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize progress tracker.

        Args:
            video_name: Name of video being processed
            total_stages: Total number of processing stages
            device_id: GPU device ID to monitor
            output_mode: Where to output progress
            log_path: Optional path for log file output
        """
        self.video_name = video_name
        self.total_stages = total_stages
        self.output_mode = output_mode

        self.stages: List[StageInfo] = []
        self.current_stage_index = 0
        self.callbacks: List[ProgressCallback] = []

        # Initialize appropriate progress display
        if RICH_AVAILABLE and output_mode != ProgressOutputMode.SILENT:
            self._rich_progress: Optional[RichProgress] = RichProgress(
                video_name=video_name,
                device_id=device_id,
                output_mode=output_mode,
                log_path=log_path,
            )
            self._fallback_progress: Optional[FallbackProgress] = None
        else:
            self._rich_progress = None
            self._fallback_progress = FallbackProgress(
                video_name=video_name,
                output_mode=output_mode,
            )

        # Setup log callback
        if log_path and output_mode in (ProgressOutputMode.LOG, ProgressOutputMode.BOTH):
            self.add_callback(LogFileCallback(log_path))

    @property
    def progress_display(self) -> Union[RichProgress, FallbackProgress]:
        """Get the active progress display."""
        if self._rich_progress is not None:
            return self._rich_progress
        if self._fallback_progress is not None:
            return self._fallback_progress
        raise RuntimeError("No progress display available")

    def add_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback.

        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)

    @contextmanager
    def track(self) -> Iterator["ProgressTracker"]:
        """Context manager for tracking progress.

        Yields:
            Self for method chaining
        """
        if self._rich_progress is not None:
            with self._rich_progress.live_display():
                try:
                    yield self
                finally:
                    self._rich_progress.complete_pipeline()
        else:
            try:
                yield self
            finally:
                pass

    def start_stage(self, name: str, total_frames: int) -> StageInfo:
        """Start a new processing stage.

        Args:
            name: Stage name
            total_frames: Total frames to process

        Returns:
            StageInfo object for the stage
        """
        stage = self.progress_display.start_stage(
            name=name,
            total_frames=total_frames,
            stage_index=self.current_stage_index,
            total_stages=self.total_stages,
        )

        for callback in self.callbacks:
            callback.on_stage_start(stage)

        return stage

    def update(self, completed: int) -> None:
        """Update current stage progress.

        Args:
            completed: Number of frames completed
        """
        self.progress_display.update(completed)

        stage = self.progress_display.current_stage
        if stage:
            gpu_metrics = None
            if self._rich_progress:
                gpu_metrics = self._rich_progress.gpu_monitor.get_metrics()
            elif self._fallback_progress:
                gpu_metrics = self._fallback_progress.gpu_monitor.get_metrics()

            for callback in self.callbacks:
                callback.on_progress_update(stage, gpu_metrics)

    def advance(self, amount: int = 1) -> None:
        """Advance progress by specified amount.

        Args:
            amount: Number of frames to advance by
        """
        if hasattr(self.progress_display, "advance"):
            self.progress_display.advance(amount)
        else:
            stage = self.progress_display.current_stage
            if stage:
                self.update(stage.completed_frames + amount)

    def complete_stage(self) -> None:
        """Mark current stage as complete."""
        self.progress_display.complete_stage()

        stage = self.progress_display.current_stage
        if stage:
            self.stages.append(stage)
            for callback in self.callbacks:
                callback.on_stage_complete(stage)

        self.current_stage_index += 1

    def complete_pipeline(self) -> None:
        """Mark pipeline as complete."""
        for callback in self.callbacks:
            callback.on_pipeline_complete(self.stages)

        if self._rich_progress:
            self._rich_progress.complete_pipeline()

    @contextmanager
    def stage(self, name: str, total_frames: int) -> Iterator[StageInfo]:
        """Context manager for a single stage.

        Args:
            name: Stage name
            total_frames: Total frames to process

        Yields:
            StageInfo for the stage
        """
        stage = self.start_stage(name, total_frames)
        try:
            yield stage
        finally:
            self.complete_stage()

    def get_overall_eta(self) -> Optional[float]:
        """Calculate overall ETA for remaining stages.

        Returns:
            Estimated time in seconds or None if cannot calculate
        """
        if not self.stages:
            return None

        # Calculate average fps across completed stages
        total_frames = sum(s.total_frames for s in self.stages)
        total_time = sum(s.elapsed_seconds for s in self.stages)

        if total_time == 0:
            return None

        avg_fps = total_frames / total_time

        # Estimate remaining work (assume similar frame counts)
        remaining_stages = self.total_stages - len(self.stages)
        if remaining_stages <= 0:
            return 0.0

        # Use average frames per stage
        avg_frames = total_frames / len(self.stages)
        remaining_frames = avg_frames * remaining_stages

        # Add current stage remaining if applicable
        current = self.progress_display.current_stage
        if current:
            remaining_frames += current.total_frames - current.completed_frames

        return remaining_frames / avg_fps if avg_fps > 0 else None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all stages.

        Returns:
            Dictionary with pipeline summary
        """
        return {
            "video_name": self.video_name,
            "total_stages": self.total_stages,
            "completed_stages": len(self.stages),
            "stages": [
                {
                    "name": s.name,
                    "total_frames": s.total_frames,
                    "elapsed_seconds": s.elapsed_seconds,
                    "fps": s.fps,
                }
                for s in self.stages
            ],
            "total_frames": sum(s.total_frames for s in self.stages),
            "total_time": sum(s.elapsed_seconds for s in self.stages),
            "average_fps": (
                sum(s.total_frames for s in self.stages)
                / max(1, sum(s.elapsed_seconds for s in self.stages))
            ),
        }


def create_progress_tracker(
    video_name: str = "",
    total_stages: int = 1,
    device_id: int = 0,
    output_mode: Union[str, ProgressOutputMode] = ProgressOutputMode.TERMINAL,
    log_path: Optional[Union[str, Path]] = None,
) -> ProgressTracker:
    """Factory function to create a progress tracker.

    Args:
        video_name: Name of video being processed
        total_stages: Total number of processing stages
        device_id: GPU device ID to monitor
        output_mode: Where to output progress ("terminal", "log", "both", "silent")
        log_path: Optional path for log file output

    Returns:
        Configured ProgressTracker instance
    """
    if isinstance(output_mode, str):
        output_mode = ProgressOutputMode(output_mode.lower())

    return ProgressTracker(
        video_name=video_name,
        total_stages=total_stages,
        device_id=device_id,
        output_mode=output_mode,
        log_path=log_path,
    )


# Convenience functions for simple use cases


def simple_progress(
    iterable: Iterator,
    total: int,
    description: str = "Processing",
) -> Iterator:
    """Simple progress wrapper for iterables.

    Args:
        iterable: Iterable to wrap
        total: Total number of items
        description: Description to display

    Yields:
        Items from the iterable
    """
    if RICH_AVAILABLE:
        from rich.progress import track as rich_track

        yield from rich_track(iterable, total=total, description=description)
    else:
        # Fallback to simple counter
        for i, item in enumerate(iterable):
            percent = (i + 1) / total * 100
            print(f"\r{description}: {percent:.0f}% ({i + 1}/{total})", end="", flush=True)
            yield item
        print()  # New line at end


def is_rich_available() -> bool:
    """Check if rich library is available.

    Returns:
        True if rich is available
    """
    return RICH_AVAILABLE


def is_pynvml_available() -> bool:
    """Check if pynvml is available.

    Returns:
        True if pynvml is available
    """
    return PYNVML_AVAILABLE
