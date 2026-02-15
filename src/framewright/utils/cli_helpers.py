"""CLI helper utilities for enhanced user experience."""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @classmethod
    def disable(cls):
        """Disable all colors."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, "")


def supports_color() -> bool:
    """Check if terminal supports colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except:
            return False
    return True


def colorize(text: str, *codes: str) -> str:
    """Apply color codes to text."""
    if not supports_color():
        return text
    return "".join(codes) + text + Colors.RESET


def success(text: str) -> str:
    """Format as success message."""
    return colorize(f"[OK] {text}", Colors.GREEN)


def warning(text: str) -> str:
    """Format as warning message."""
    return colorize(f"[!] {text}", Colors.YELLOW)


def error(text: str) -> str:
    """Format as error message."""
    return colorize(f"[X] {text}", Colors.RED)


def info(text: str) -> str:
    """Format as info message."""
    return colorize(f"[i] {text}", Colors.CYAN)


def header(text: str) -> str:
    """Format as header."""
    return colorize(text, Colors.BOLD, Colors.BRIGHT_CYAN)


def dim(text: str) -> str:
    """Format as dimmed text."""
    return colorize(text, Colors.DIM)


@dataclass
class StorageEstimate:
    """Storage requirement estimate."""
    input_size_gb: float
    frames_size_gb: float
    output_size_gb: float
    temp_size_gb: float
    total_required_gb: float

    def to_string(self) -> str:
        return (
            f"Storage Estimate:\n"
            f"  Input video:     {self.input_size_gb:.2f} GB\n"
            f"  Extracted frames: {self.frames_size_gb:.2f} GB\n"
            f"  Output video:    {self.output_size_gb:.2f} GB\n"
            f"  Temp space:      {self.temp_size_gb:.2f} GB\n"
            f"  Total required:  {self.total_required_gb:.2f} GB"
        )


def estimate_storage(
    input_path: Path,
    scale_factor: int = 1,
    crf: int = 18,
) -> StorageEstimate:
    """Estimate storage requirements for restoration.

    Args:
        input_path: Path to input video
        scale_factor: Upscaling factor
        crf: Output CRF (lower = bigger file)

    Returns:
        Storage estimate
    """
    input_path = Path(input_path)

    # Get input size
    input_size = input_path.stat().st_size / (1024**3)  # GB

    # Get video info
    info = _get_video_info(input_path)
    duration = info.get("duration", 60)
    fps = info.get("fps", 30)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    # Estimate frame count and size
    frame_count = int(duration * fps)
    bytes_per_pixel = 3  # RGB
    frame_size = width * height * bytes_per_pixel * (scale_factor ** 2)
    frames_size_gb = (frame_count * frame_size) / (1024**3)

    # Estimate output size based on CRF
    # CRF 18 ~ 60% of uncompressed, CRF 23 ~ 30%
    compression_ratio = 0.6 - (crf - 18) * 0.03
    compression_ratio = max(0.1, min(0.8, compression_ratio))
    output_pixels = frame_count * (width * scale_factor) * (height * scale_factor) * 3
    output_size_gb = (output_pixels * compression_ratio) / (1024**3)

    # Temp space for intermediate files
    temp_size_gb = frames_size_gb * 0.2  # 20% for temp

    total = input_size + frames_size_gb + output_size_gb + temp_size_gb

    return StorageEstimate(
        input_size_gb=input_size,
        frames_size_gb=frames_size_gb,
        output_size_gb=output_size_gb,
        temp_size_gb=temp_size_gb,
        total_required_gb=total,
    )


def _get_video_info(path: Path) -> Dict[str, Any]:
    """Get video information."""
    import json
    import subprocess

    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            info = {}

            fmt = data.get("format", {})
            info["duration"] = float(fmt.get("duration", 0))

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = stream.get("width", 1920)
                    info["height"] = stream.get("height", 1080)
                    fps_str = stream.get("r_frame_rate", "30/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        info["fps"] = float(num) / float(den) if float(den) > 0 else 30
                    break

            return info
    except:
        pass

    return {"duration": 60, "width": 1920, "height": 1080, "fps": 30}


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    td = timedelta(seconds=int(seconds))
    parts = []

    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_size(bytes_val: float) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
    fill: str = "█",
) -> None:
    """Print a progress bar to the terminal."""
    percent = current / total if total > 0 else 0
    filled_length = int(length * percent)
    bar = fill * filled_length + "-" * (length - filled_length)

    # Color based on progress
    if percent < 0.3:
        color = Colors.RED
    elif percent < 0.7:
        color = Colors.YELLOW
    else:
        color = Colors.GREEN

    bar_colored = colorize(bar, color) if supports_color() else bar
    line = f"\r{prefix} |{bar_colored}| {percent*100:.1f}% {suffix}"

    sys.stdout.write(line)
    sys.stdout.flush()

    if current >= total:
        print()


def print_table(
    headers: List[str],
    rows: List[List[str]],
    alignment: Optional[List[str]] = None,
) -> None:
    """Print a formatted table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Default alignment
    if alignment is None:
        alignment = ["<"] * len(headers)

    # Print header
    header_line = " | ".join(
        f"{h:{a}{w}}" for h, w, a in zip(headers, widths, alignment)
    )
    print(colorize(header_line, Colors.BOLD))
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        row_line = " | ".join(
            f"{str(c):{a}{w}}" for c, w, a in zip(row, widths, alignment)
        )
        print(row_line)


def confirm(prompt: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = input(colorize(prompt + suffix, Colors.YELLOW)).strip().lower()

    if not response:
        return default
    return response in ("y", "yes")


def select_option(
    prompt: str,
    options: List[str],
    default: int = 0,
) -> int:
    """Let user select from options."""
    print(colorize(prompt, Colors.CYAN))
    for i, opt in enumerate(options):
        marker = ">" if i == default else " "
        print(f"  {marker} {i+1}. {opt}")

    while True:
        try:
            choice = input(f"Enter choice [1-{len(options)}] (default: {default+1}): ").strip()
            if not choice:
                return default
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(error("Invalid choice"))


class DryRunContext:
    """Context manager for dry-run mode."""

    def __init__(self):
        self.actions: List[str] = []

    def log_action(self, action: str, details: str = "") -> None:
        """Log an action that would be performed."""
        self.actions.append(f"{action}: {details}" if details else action)
        print(dim(f"[DRY RUN] {action}"))
        if details:
            print(dim(f"          {details}"))

    def print_summary(self) -> None:
        """Print summary of actions that would be performed."""
        print("\n" + header("Dry Run Summary"))
        print("-" * 40)
        for i, action in enumerate(self.actions, 1):
            print(f"{i}. {action}")
        print("-" * 40)
        print(f"Total actions: {len(self.actions)}")


# Initialize colors based on terminal support
if not supports_color():
    Colors.disable()
