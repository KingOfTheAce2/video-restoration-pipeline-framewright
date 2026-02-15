"""Beautiful progress displays for FrameWright.

Provides Apple-quality progress indicators with:
- Multi-stage progress tracking
- ETA calculations
- FPS statistics
- Live updating displays
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import time
import threading

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
        TransferSpeedColumn,
    )
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console, Group
    from rich.text import Text
    from rich.spinner import Spinner
    from rich.box import ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class StageProgress:
    """Progress information for a processing stage."""
    name: str
    display_name: str
    total: int = 0
    completed: int = 0
    status: str = "pending"  # pending, running, completed, error
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    fps: float = 0.0
    eta_seconds: Optional[float] = None

    @property
    def progress(self) -> float:
        """Get progress as 0-1 value."""
        if self.total == 0:
            return 0.0
        return min(1.0, self.completed / self.total)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        """Get elapsed time as MM:SS or HH:MM:SS."""
        elapsed = int(self.elapsed_seconds)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def eta_formatted(self) -> str:
        """Get ETA as MM:SS or HH:MM:SS."""
        if self.eta_seconds is None or self.eta_seconds < 0:
            return "calculating..."
        eta = int(self.eta_seconds)
        hours, remainder = divmod(eta, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


class ProgressDisplay:
    """Beautiful multi-stage progress display.

    Features:
    - Multiple concurrent progress bars
    - Real-time FPS calculation
    - ETA estimation
    - Stage status indicators
    - Statistics panel

    Example:
        >>> display = ProgressDisplay()
        >>> display.add_stage("extract", "Extracting Frames", total=1000)
        >>> display.add_stage("enhance", "Enhancing", total=1000)
        >>> display.start()
        >>> for i in range(1000):
        ...     display.update("extract", completed=i+1)
        >>> display.complete_stage("extract")
        >>> display.stop()
    """

    # Stage display names and icons
    STAGE_ICONS = {
        "download": "📥",
        "extract": "🎬",
        "analyze": "🔍",
        "qp_removal": "🧹",
        "denoise": "✨",
        "frame_gen": "🎞️",
        "enhance": "⬆️",
        "face": "👤",
        "interpolate": "🔄",
        "temporal": "⏱️",
        "colorize": "🎨",
        "reassemble": "📦",
        "validate": "✅",
    }

    def __init__(self, console: Optional["Console"] = None):
        """Initialize progress display.

        Args:
            console: Rich console instance (created if not provided)
        """
        self.stages: Dict[str, StageProgress] = {}
        self.stage_order: List[str] = []
        self.current_stage: Optional[str] = None
        self._live: Optional["Live"] = None
        self._lock = threading.Lock()
        self._running = False
        self._start_time: Optional[float] = None

        if RICH_AVAILABLE:
            self._console = console or Console()
        else:
            self._console = None

    def add_stage(
        self,
        name: str,
        display_name: str,
        total: int = 0,
    ) -> None:
        """Add a processing stage.

        Args:
            name: Internal stage name
            display_name: Human-readable name
            total: Total items to process (0 if unknown)
        """
        with self._lock:
            self.stages[name] = StageProgress(
                name=name,
                display_name=display_name,
                total=total,
            )
            self.stage_order.append(name)

    def start(self) -> None:
        """Start the progress display."""
        self._running = True
        self._start_time = time.time()

        if RICH_AVAILABLE and self._console:
            self._live = Live(
                self._create_display(),
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()

    def stop(self) -> None:
        """Stop the progress display."""
        self._running = False
        if self._live:
            self._live.stop()
            self._live = None

    def update(
        self,
        stage: str,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        """Update stage progress.

        Args:
            stage: Stage name
            completed: Number of items completed
            total: Total items (updates if provided)
            fps: Processing speed
        """
        with self._lock:
            if stage not in self.stages:
                return

            s = self.stages[stage]

            # Start stage if not already running
            if s.status == "pending":
                s.status = "running"
                s.start_time = time.time()
                self.current_stage = stage

            if total is not None:
                s.total = total

            if completed is not None:
                s.completed = completed

                # Calculate FPS
                if s.start_time and s.completed > 0:
                    elapsed = time.time() - s.start_time
                    s.fps = s.completed / elapsed if elapsed > 0 else 0

                    # Calculate ETA
                    if s.total > 0 and s.fps > 0:
                        remaining = s.total - s.completed
                        s.eta_seconds = remaining / s.fps

            if fps is not None:
                s.fps = fps

        # Update display
        if self._live:
            self._live.update(self._create_display())

    def complete_stage(self, stage: str) -> None:
        """Mark a stage as completed.

        Args:
            stage: Stage name
        """
        with self._lock:
            if stage in self.stages:
                s = self.stages[stage]
                s.status = "completed"
                s.end_time = time.time()
                s.completed = s.total
                s.eta_seconds = 0

        if self._live:
            self._live.update(self._create_display())

    def error_stage(self, stage: str, message: str = "") -> None:
        """Mark a stage as errored.

        Args:
            stage: Stage name
            message: Error message
        """
        with self._lock:
            if stage in self.stages:
                self.stages[stage].status = "error"
                self.stages[stage].end_time = time.time()

        if self._live:
            self._live.update(self._create_display())

    def _create_display(self) -> "Panel":
        """Create the Rich display panel."""
        if not RICH_AVAILABLE:
            return None

        # Create progress table
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Icon", width=3)
        table.add_column("Stage", min_width=20)
        table.add_column("Progress", min_width=30)
        table.add_column("Stats", min_width=15)
        table.add_column("Status", width=12)

        for name in self.stage_order:
            stage = self.stages[name]
            icon = self.STAGE_ICONS.get(name.split("_")[0], "▪️")

            # Status indicator
            if stage.status == "completed":
                status = "[green]✓ Done[/green]"
                style = "dim"
            elif stage.status == "running":
                status = f"[yellow]⏳ {stage.eta_formatted}[/yellow]"
                style = "bold"
            elif stage.status == "error":
                status = "[red]✗ Error[/red]"
                style = "red"
            else:
                status = "[dim]○ Waiting[/dim]"
                style = "dim"

            # Progress bar
            if stage.total > 0:
                filled = int(stage.progress * 20)
                bar = "█" * filled + "░" * (20 - filled)
                pct = f"{stage.progress * 100:.0f}%"
                progress = f"[{bar}] {pct}"
            else:
                progress = "[dim]—[/dim]"

            # Stats
            if stage.status == "running" and stage.fps > 0:
                stats = f"[cyan]{stage.fps:.1f}[/cyan] fps"
            elif stage.status == "completed":
                stats = f"[dim]{stage.elapsed_formatted}[/dim]"
            else:
                stats = ""

            table.add_row(
                icon,
                f"[{style}]{stage.display_name}[/{style}]",
                progress,
                stats,
                status,
            )

        # Overall stats
        total_elapsed = time.time() - self._start_time if self._start_time else 0
        hours, remainder = divmod(int(total_elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            elapsed_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            elapsed_str = f"{minutes}:{seconds:02d}"

        completed_stages = sum(1 for s in self.stages.values() if s.status == "completed")
        total_stages = len(self.stages)

        footer = Text()
        footer.append(f"\n  Total: ", style="dim")
        footer.append(f"{completed_stages}/{total_stages}", style="bold")
        footer.append(f" stages  |  Elapsed: ", style="dim")
        footer.append(elapsed_str, style="bold cyan")

        content = Group(table, footer)

        return Panel(
            content,
            title="[bold cyan]FrameWright[/bold cyan] [dim]Processing[/dim]",
            border_style="cyan",
            padding=(1, 2),
            box=ROUNDED,
        )

    def print_simple(self, stage: str, completed: int, total: int) -> None:
        """Print simple text progress (fallback when Rich unavailable)."""
        pct = (completed / total * 100) if total > 0 else 0
        print(f"\r{stage}: {completed}/{total} ({pct:.1f}%)", end="", flush=True)


class SpinnerContext:
    """Context manager for spinner display."""

    def __init__(self, message: str, console: Optional["Console"] = None):
        self.message = message
        self._console = console
        self._live: Optional["Live"] = None

    def __enter__(self):
        if RICH_AVAILABLE:
            from rich.spinner import Spinner
            spinner = Spinner("dots", text=f" {self.message}")
            self._live = Live(spinner, console=self._console, transient=True)
            self._live.start()
        else:
            print(f"{self.message}...", end="", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.stop()
        else:
            print(" done" if exc_type is None else " failed")
        return False

    def update(self, message: str) -> None:
        """Update spinner message."""
        if self._live and RICH_AVAILABLE:
            from rich.spinner import Spinner
            self._live.update(Spinner("dots", text=f" {message}"))
        self.message = message


def create_progress_display(console: Optional["Console"] = None) -> ProgressDisplay:
    """Create a progress display instance.

    Args:
        console: Optional Rich console

    Returns:
        Configured ProgressDisplay
    """
    return ProgressDisplay(console=console)


def create_spinner(message: str, console: Optional["Console"] = None) -> SpinnerContext:
    """Create a spinner context manager.

    Args:
        message: Spinner message
        console: Optional Rich console

    Returns:
        SpinnerContext for use with `with` statement

    Example:
        >>> with create_spinner("Analyzing video"):
        ...     analyze()
    """
    return SpinnerContext(message, console)


class ArchiveRestorationProgress(ProgressDisplay):
    """Specialized progress display for archive footage restoration.

    Pre-configured with all ultimate preset stages.
    """

    def __init__(self, console: Optional["Console"] = None):
        super().__init__(console)

        # Add all possible restoration stages
        self.add_stage("download", "Downloading Video")
        self.add_stage("analyze", "Analyzing Content")
        self.add_stage("extract", "Extracting Frames")
        self.add_stage("dedup", "Deduplicating Frames")
        self.add_stage("qp_removal", "Removing Compression Artifacts")
        self.add_stage("denoise", "Neural Denoising (TAP)")
        self.add_stage("frame_gen", "Generating Missing Frames")
        self.add_stage("enhance", "Super-Resolution")
        self.add_stage("face", "Face Enhancement")
        self.add_stage("interpolate", "Frame Interpolation (RIFE)")
        self.add_stage("temporal", "Temporal Consistency")
        self.add_stage("colorize", "Colorization")
        self.add_stage("reassemble", "Reassembling Video")
        self.add_stage("validate", "Validating Output")

    def skip_stage(self, stage: str) -> None:
        """Mark a stage as skipped (remove from display)."""
        with self._lock:
            if stage in self.stages:
                self.stage_order.remove(stage)
                del self.stages[stage]
