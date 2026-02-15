"""Rich-based terminal UI for FrameWright.

Provides Apple-quality terminal output with beautiful formatting,
consistent styling, and clear visual hierarchy.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

try:
    from rich.console import Console as RichConsole
    from rich.theme import Theme as RichTheme
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.style import Style
    from rich.box import ROUNDED, HEAVY, DOUBLE
    from rich.align import Align
    from rich.columns import Columns
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.rule import Rule
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# FrameWright brand colors and theme
FRAMEWRIGHT_THEME = {
    "brand": "bold cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "bold blue",
    "muted": "dim white",
    "highlight": "bold magenta",
    "path": "underline cyan",
    "number": "bold white",
    "stage": "bold cyan",
    "progress": "green",
    "eta": "yellow",
    "fps": "cyan",
    "quality": "magenta",
}


@dataclass
class Theme:
    """UI theme configuration."""
    brand_color: str = "cyan"
    success_color: str = "green"
    error_color: str = "red"
    warning_color: str = "yellow"
    info_color: str = "blue"

    def to_rich_theme(self) -> Optional["RichTheme"]:
        """Convert to Rich theme."""
        if not RICH_AVAILABLE:
            return None
        return RichTheme(FRAMEWRIGHT_THEME)


class Console:
    """Enhanced console for beautiful terminal output."""

    def __init__(self, theme: Optional[Theme] = None, quiet: bool = False):
        """Initialize console.

        Args:
            theme: UI theme configuration
            quiet: If True, suppress non-essential output
        """
        self.theme = theme or Theme()
        self.quiet = quiet

        if RICH_AVAILABLE:
            self._console = RichConsole(
                theme=self.theme.to_rich_theme(),
                highlight=True,
                markup=True,
            )
        else:
            self._console = None

    def print(self, *args, **kwargs) -> None:
        """Print with Rich formatting if available."""
        if self.quiet:
            return
        if self._console:
            self._console.print(*args, **kwargs)
        else:
            print(*args)

    def print_banner(self) -> None:
        """Print the FrameWright banner."""
        if self.quiet:
            return

        banner = """
[bold cyan]
  ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗██████╗ ██╗ ██████╗ ██╗  ██╗████████╗
  ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔══██╗██║██╔════╝ ██║  ██║╚══██╔══╝
  █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██████╔╝██║██║  ███╗███████║   ██║
  ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██╔══██╗██║██║   ██║██╔══██║   ██║
  ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝██║  ██║██║╚██████╔╝██║  ██║   ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
[/bold cyan]
[dim]  Ultimate AI Video Restoration Pipeline[/dim]
        """

        if self._console:
            self._console.print(banner)
            self._console.print()
        else:
            print("=" * 60)
            print("  FRAMEWRIGHT - Ultimate AI Video Restoration")
            print("=" * 60)
            print()

    def print_compact_banner(self) -> None:
        """Print a compact one-line banner."""
        if self.quiet:
            return

        if self._console:
            self._console.print(
                "[bold cyan]FrameWright[/bold cyan] [dim]|[/dim] "
                "[dim]Ultimate AI Video Restoration[/dim]"
            )
            self._console.print()
        else:
            print("FrameWright | Ultimate AI Video Restoration")
            print()

    def success(self, message: str) -> None:
        """Print success message."""
        if self._console:
            self._console.print(f"[success]✓[/success] {message}")
        else:
            print(f"✓ {message}")

    def error(self, message: str, hint: Optional[str] = None) -> None:
        """Print error message with optional hint."""
        if self._console:
            self._console.print(f"[error]✗[/error] {message}")
            if hint:
                self._console.print(f"  [dim]Hint: {hint}[/dim]")
        else:
            print(f"✗ {message}")
            if hint:
                print(f"  Hint: {hint}")

    def warning(self, message: str) -> None:
        """Print warning message."""
        if self._console:
            self._console.print(f"[warning]![/warning] {message}")
        else:
            print(f"! {message}")

    def info(self, message: str) -> None:
        """Print info message."""
        if self._console:
            self._console.print(f"[info]ℹ[/info] {message}")
        else:
            print(f"ℹ {message}")

    def step(self, number: int, total: int, message: str) -> None:
        """Print a step indicator."""
        if self._console:
            self._console.print(
                f"[stage]Step {number}/{total}[/stage] [dim]│[/dim] {message}"
            )
        else:
            print(f"Step {number}/{total} | {message}")

    def panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "cyan",
        padding: tuple = (1, 2),
    ) -> None:
        """Print content in a panel."""
        if self._console:
            self._console.print(
                Panel(
                    content,
                    title=title,
                    style=style,
                    padding=padding,
                    box=ROUNDED,
                )
            )
        else:
            if title:
                print(f"=== {title} ===")
            print(content)
            print()

    def table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Print data as a table."""
        if not data:
            return

        if columns is None:
            columns = list(data[0].keys())

        if self._console:
            table = Table(
                title=title,
                box=ROUNDED,
                header_style="bold cyan",
                row_styles=["", "dim"],
            )

            for col in columns:
                table.add_column(col.replace("_", " ").title())

            for row in data:
                table.add_row(*[str(row.get(col, "")) for col in columns])

            self._console.print(table)
        else:
            if title:
                print(f"\n{title}")
                print("-" * 40)
            for row in data:
                print(" | ".join(f"{k}: {v}" for k, v in row.items()))
            print()

    def rule(self, title: Optional[str] = None, style: str = "dim") -> None:
        """Print a horizontal rule."""
        if self._console:
            self._console.print(Rule(title, style=style))
        else:
            if title:
                print(f"--- {title} ---")
            else:
                print("-" * 40)

    def video_summary(
        self,
        path: Path,
        resolution: str,
        fps: float,
        duration: str,
        codec: str,
        size_mb: float,
    ) -> None:
        """Print a beautiful video summary panel."""
        if self._console:
            content = (
                f"[path]{path.name}[/path]\n\n"
                f"  [dim]Resolution:[/dim]  [number]{resolution}[/number]\n"
                f"  [dim]Frame Rate:[/dim]  [fps]{fps}[/fps] fps\n"
                f"  [dim]Duration:[/dim]    [number]{duration}[/number]\n"
                f"  [dim]Codec:[/dim]       [muted]{codec}[/muted]\n"
                f"  [dim]Size:[/dim]        [number]{size_mb:.1f}[/number] MB"
            )
            self._console.print(
                Panel(
                    content,
                    title="[bold]Video Analysis[/bold]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        else:
            print(f"\nVideo: {path.name}")
            print(f"  Resolution: {resolution}")
            print(f"  Frame Rate: {fps} fps")
            print(f"  Duration: {duration}")
            print(f"  Codec: {codec}")
            print(f"  Size: {size_mb:.1f} MB\n")

    def restoration_plan(
        self,
        preset: str,
        stages: List[str],
        estimated_time: str,
        quality_target: str,
    ) -> None:
        """Print the restoration plan."""
        if self._console:
            stages_text = "\n".join(f"  [dim]{i+1}.[/dim] {s}" for i, s in enumerate(stages))
            content = (
                f"[highlight]Preset:[/highlight] {preset}\n"
                f"[highlight]Quality Target:[/highlight] {quality_target}\n"
                f"[highlight]Estimated Time:[/highlight] {estimated_time}\n\n"
                f"[bold]Processing Pipeline:[/bold]\n{stages_text}"
            )
            self._console.print(
                Panel(
                    content,
                    title="[bold]Restoration Plan[/bold]",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )
        else:
            print(f"\nRestoration Plan")
            print(f"  Preset: {preset}")
            print(f"  Quality: {quality_target}")
            print(f"  Est. Time: {estimated_time}")
            print(f"  Stages: {', '.join(stages)}\n")

    def completion_summary(
        self,
        output_path: Path,
        duration: str,
        frames_processed: int,
        quality_metrics: Dict[str, float],
    ) -> None:
        """Print completion summary."""
        if self._console:
            metrics = "\n".join(
                f"  [dim]{k}:[/dim] [quality]{v:.2f}[/quality]"
                for k, v in quality_metrics.items()
            )
            content = (
                f"[success]Restoration Complete![/success]\n\n"
                f"  [dim]Output:[/dim]     [path]{output_path}[/path]\n"
                f"  [dim]Duration:[/dim]   [number]{duration}[/number]\n"
                f"  [dim]Frames:[/dim]     [number]{frames_processed:,}[/number]\n\n"
                f"[bold]Quality Metrics:[/bold]\n{metrics}"
            )
            self._console.print(
                Panel(
                    content,
                    title="[bold green]Done[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            print(f"\n✓ Restoration Complete!")
            print(f"  Output: {output_path}")
            print(f"  Duration: {duration}")
            print(f"  Frames: {frames_processed:,}")
            for k, v in quality_metrics.items():
                print(f"  {k}: {v:.2f}")
            print()


# Module-level convenience functions
_default_console: Optional[Console] = None


def create_console(theme: Optional[Theme] = None, quiet: bool = False) -> Console:
    """Create and cache a console instance."""
    global _default_console
    _default_console = Console(theme=theme, quiet=quiet)
    return _default_console


def get_console() -> Console:
    """Get the default console, creating if needed."""
    global _default_console
    if _default_console is None:
        _default_console = Console()
    return _default_console


def print_banner() -> None:
    """Print the FrameWright banner."""
    get_console().print_banner()


def print_compact_banner() -> None:
    """Print compact banner."""
    get_console().print_compact_banner()


def print_success(message: str) -> None:
    """Print success message."""
    get_console().success(message)


def print_error(message: str, hint: Optional[str] = None) -> None:
    """Print error message."""
    get_console().error(message, hint)


def print_warning(message: str) -> None:
    """Print warning message."""
    get_console().warning(message)


def print_info(message: str) -> None:
    """Print info message."""
    get_console().info(message)


def print_step(number: int, total: int, message: str) -> None:
    """Print step indicator."""
    get_console().step(number, total, message)


def create_panel(
    content: str,
    title: Optional[str] = None,
    style: str = "cyan",
) -> None:
    """Create and print a panel."""
    get_console().panel(content, title, style)


def create_table(
    data: List[Dict[str, Any]],
    title: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> None:
    """Create and print a table."""
    get_console().table(data, title, columns)
