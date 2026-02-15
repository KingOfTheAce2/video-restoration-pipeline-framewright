"""Interactive restoration wizard for FrameWright.

Provides a step-by-step guided interface for video restoration,
making it easy for users to configure and run the restoration pipeline.

Example:
    >>> from framewright.ui.cli.wizard import run_wizard
    >>> config = run_wizard("my_video.mp4")
    >>> # User is guided through analysis, recommendation, customization
    >>> # Returns a Config object ready for restoration
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re


# =============================================================================
# WizardConfig
# =============================================================================

@dataclass
class WizardConfig:
    """Configuration for the restoration wizard.

    Attributes:
        show_preview: Show preview during wizard (default True)
        auto_detect: Auto-detect content type (default True)
        verbose: Show detailed analysis (default False)
        color_output: Use colored terminal output (default True)
    """
    show_preview: bool = True
    auto_detect: bool = True
    verbose: bool = False
    color_output: bool = True


# =============================================================================
# ANSI Color Support
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'

    # Colors
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # Background
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'
    BG_CYAN = '\033[46m'

    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports ANSI colors."""
        # Check for NO_COLOR environment variable
        if os.environ.get('NO_COLOR'):
            return False

        # Check for FORCE_COLOR
        if os.environ.get('FORCE_COLOR'):
            return True

        # Check if stdout is a TTY
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False

        # Windows-specific checks
        if os.name == 'nt':
            # Windows Terminal, VS Code, and modern terminals support ANSI
            return (
                os.environ.get('WT_SESSION') is not None or  # Windows Terminal
                os.environ.get('TERM_PROGRAM') == 'vscode' or  # VS Code
                'ANSICON' in os.environ or  # ANSICON
                os.environ.get('ConEmuANSI') == 'ON' or  # ConEmu
                os.environ.get('TERM') == 'xterm-256color'  # Git Bash
            )

        # Unix-like systems generally support colors
        return True

    @classmethod
    def enable_windows_ansi(cls) -> None:
        """Enable ANSI escape sequences on Windows."""
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences on Windows 10+
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-11),  # STD_OUTPUT_HANDLE
                    7  # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                )
            except Exception:
                pass


# Global color support flag
_color_enabled = True


def _c(text: str, *styles: str) -> str:
    """Apply color/styles to text if color is enabled."""
    if not _color_enabled:
        return text
    style_str = ''.join(styles)
    return f"{style_str}{text}{Colors.RESET}"


# =============================================================================
# Terminal UI Helpers
# =============================================================================

def clear_screen() -> None:
    """Clear the terminal screen."""
    if os.name == 'nt':
        os.system('cls')
    else:
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.flush()


def print_header(text: str) -> None:
    """Print a styled header."""
    width = min(70, _get_terminal_width())
    line = "=" * width
    print()
    print(_c(line, Colors.CYAN))
    print(_c(f"  {text}", Colors.BOLD, Colors.CYAN))
    print(_c(line, Colors.CYAN))
    print()


def print_step(num: int, text: str, total: int = 5) -> None:
    """Print a step indicator."""
    filled = num
    empty = total - num
    progress = _c("[", Colors.DIM) + _c("*" * filled, Colors.GREEN) + _c("-" * empty, Colors.DIM) + _c("]", Colors.DIM)
    print(f"{progress} {_c(f'Step {num}/{total}:', Colors.BOLD)} {text}")
    print()


def print_option(key: str, text: str, selected: bool = False) -> None:
    """Print a menu option."""
    if selected:
        indicator = _c(">", Colors.GREEN, Colors.BOLD)
        text_style = _c(text, Colors.GREEN, Colors.BOLD)
    else:
        indicator = " "
        text_style = text
    print(f"  {indicator} [{_c(key, Colors.CYAN, Colors.BOLD)}] {text_style}")


def print_progress(percent: float, message: str = "", width: int = 40) -> None:
    """Print a progress bar."""
    filled = int(width * percent)
    empty = width - filled
    bar = _c("█" * filled, Colors.GREEN) + _c("░" * empty, Colors.DIM)
    pct = _c(f"{int(percent * 100):3d}%", Colors.BOLD)

    # Use carriage return to update in place
    sys.stdout.write(f"\r  {bar} {pct}")
    if message:
        sys.stdout.write(f"  {_c(message, Colors.DIM)}")
    sys.stdout.write("   ")  # Clear any trailing characters
    sys.stdout.flush()


def print_comparison(label1: str, value1: str, label2: str, value2: str) -> None:
    """Print side-by-side comparison stats."""
    col_width = 25
    left = f"{_c(label1 + ':', Colors.DIM)} {value1}"
    right = f"{_c(label2 + ':', Colors.DIM)} {value2}"
    print(f"  {left:<{col_width + 10}}  {right}")


def print_info(label: str, value: str) -> None:
    """Print an info line."""
    print(f"  {_c(label + ':', Colors.DIM)} {_c(value, Colors.WHITE)}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"  {_c('[OK]', Colors.GREEN, Colors.BOLD)} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"  {_c('[!]', Colors.YELLOW, Colors.BOLD)} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"  {_c('[X]', Colors.RED, Colors.BOLD)} {message}")


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask a yes/no question and return the answer.

    Args:
        question: The question to ask
        default: Default answer if user just presses Enter

    Returns:
        True for yes, False for no
    """
    default_hint = "[Y/n]" if default else "[y/N]"
    prompt = f"  {question} {_c(default_hint, Colors.DIM)} "

    while True:
        try:
            answer = input(prompt).strip().lower()
            if not answer:
                return default
            if answer in ('y', 'yes'):
                return True
            if answer in ('n', 'no'):
                return False
            print(f"    {_c('Please enter Y or N', Colors.YELLOW)}")
        except (EOFError, KeyboardInterrupt):
            print()
            return default


def ask_choice(question: str, options: List[Tuple[str, str]], default: str = "") -> str:
    """Ask user to choose from multiple options.

    Args:
        question: The question to ask
        options: List of (key, description) tuples
        default: Default key if user just presses Enter

    Returns:
        The selected option key
    """
    print(f"  {question}")
    print()
    for key, desc in options:
        is_default = key == default
        marker = _c("(default)", Colors.DIM) if is_default else ""
        print(f"    [{_c(key, Colors.CYAN, Colors.BOLD)}] {desc} {marker}")
    print()

    valid_keys = [k for k, _ in options]
    prompt = f"  Choice [{'/'.join(valid_keys)}]: "

    while True:
        try:
            answer = input(prompt).strip().lower()
            if not answer and default:
                return default
            if answer in valid_keys:
                return answer
            print(f"    {_c('Invalid choice. Please try again.', Colors.YELLOW)}")
        except (EOFError, KeyboardInterrupt):
            print()
            return default if default else valid_keys[0]


def ask_input(prompt: str, default: str = "") -> str:
    """Ask for text input with optional default.

    Args:
        prompt: The prompt to display
        default: Default value if user just presses Enter

    Returns:
        User input or default
    """
    default_hint = f" [{_c(default, Colors.DIM)}]" if default else ""
    full_prompt = f"  {prompt}{default_hint}: "

    try:
        answer = input(full_prompt).strip()
        return answer if answer else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def wait_for_key(message: str = "Press Enter to continue...") -> None:
    """Wait for user to press Enter."""
    try:
        input(f"\n  {_c(message, Colors.DIM)}")
    except (EOFError, KeyboardInterrupt):
        print()


def _get_terminal_width() -> int:
    """Get terminal width."""
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 0:
        return "calculating..."

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def is_youtube_url(url: str) -> bool:
    """Check if a string is a valid YouTube URL.

    Args:
        url: String to check

    Returns:
        True if it's a valid YouTube URL
    """
    youtube_patterns = [
        r'^https?://(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://youtu\.be/[\w-]+',
        r'^https?://(www\.)?youtube\.com/shorts/[\w-]+',
        r'^https?://(www\.)?youtube\.com/embed/[\w-]+',
        r'^https?://m\.youtube\.com/watch\?v=[\w-]+',
    ]
    for pattern in youtube_patterns:
        if re.match(pattern, url.strip()):
            return True
    return False


def ask_folder(prompt: str, default: Optional[str] = None) -> Path:
    """Ask for a folder path with validation.

    Args:
        prompt: The prompt to display
        default: Default folder path

    Returns:
        Valid folder Path
    """
    default_hint = f" [{_c(default, Colors.DIM)}]" if default else ""
    full_prompt = f"  {prompt}{default_hint}: "

    while True:
        try:
            answer = input(full_prompt).strip()
            if not answer and default:
                folder = Path(default).expanduser()
            else:
                folder = Path(answer).expanduser()

            # Create folder if it doesn't exist
            if not folder.exists():
                create = ask_yes_no(f"Folder '{folder}' doesn't exist. Create it?", default=True)
                if create:
                    folder.mkdir(parents=True, exist_ok=True)
                    print_success(f"Created folder: {folder}")
                    return folder
                else:
                    continue
            elif not folder.is_dir():
                print_error(f"'{folder}' is not a directory")
                continue
            else:
                return folder

        except (EOFError, KeyboardInterrupt):
            print()
            if default:
                return Path(default).expanduser()
            raise


def download_youtube_video(
    url: str,
    output_folder: Path,
    quality: str = "best",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Optional[Path]:
    """Download a YouTube video to the specified folder.

    Args:
        url: YouTube URL
        output_folder: Folder to save the video
        quality: Quality preset (best, 1080p, 720p, etc.)
        progress_callback: Optional callback(progress, message)

    Returns:
        Path to downloaded video or None on failure
    """
    try:
        from ...utils.youtube import YouTubeDownloader, get_ytdlp_path

        if not get_ytdlp_path():
            print_error("yt-dlp not found. Please install it: pip install yt-dlp")
            return None

        downloader = YouTubeDownloader(output_dir=output_folder)

        # Get video info first
        if progress_callback:
            progress_callback(0.1, "Fetching video info...")

        info = downloader.get_video_info(url)
        if not info:
            print_error("Could not fetch video information")
            return None

        print()
        print(f"  {_c('Video Found:', Colors.BOLD)}")
        print_info("Title", info.title[:60] + "..." if len(info.title) > 60 else info.title)
        print_info("Channel", info.channel or "Unknown")
        print_info("Duration", format_time(info.duration or 0))
        print()

        # Download with progress tracking
        def yt_progress_hook(d: Dict[str, Any]) -> None:
            if d.get('status') == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                if total > 0 and progress_callback:
                    progress = downloaded / total
                    speed = d.get('speed', 0) or 0
                    speed_str = f"{speed / 1024 / 1024:.1f} MB/s" if speed else ""
                    progress_callback(0.1 + progress * 0.8, f"Downloading... {speed_str}")
            elif d.get('status') == 'finished':
                if progress_callback:
                    progress_callback(0.95, "Processing...")

        if progress_callback:
            progress_callback(0.15, "Starting download...")

        result = downloader.download(
            url,
            quality=quality,
            progress_callback=yt_progress_hook,
        )

        if result and result.exists():
            if progress_callback:
                progress_callback(1.0, "Download complete!")
            return result

        print_error("Download failed")
        return None

    except ImportError:
        print_error("YouTube downloader not available. Please install yt-dlp: pip install yt-dlp")
        return None
    except Exception as e:
        print_error(f"Download failed: {e}")
        return None


# =============================================================================
# Analysis Helpers
# =============================================================================

def _analyze_video(video_path: Path, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Analyze video file and return metadata and recommendations.

    Args:
        video_path: Path to video file
        progress_callback: Optional callback(progress, message)

    Returns:
        Dictionary with analysis results
    """
    result = {
        'path': video_path,
        'filename': video_path.name,
        'duration': 0.0,
        'resolution': (0, 0),
        'fps': 0.0,
        'codec': 'unknown',
        'bitrate': 0,
        'file_size': 0,
        'content_type': 'unknown',
        'quality_issues': [],
        'has_faces': False,
        'is_bw': False,
        'recommended_preset': 'balanced',
        'recommended_processors': [],
        'estimated_time': 0.0,
        'estimated_vram': 0,
    }

    if progress_callback:
        progress_callback(0.1, "Reading file info...")

    # Get file size
    if video_path.exists():
        result['file_size'] = video_path.stat().st_size

    # Try to get video metadata using ffprobe
    try:
        import subprocess
        import json

        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)

            # Find video stream
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                {}
            )

            # Resolution
            result['resolution'] = (
                int(video_stream.get('width', 0)),
                int(video_stream.get('height', 0)),
            )

            # FPS
            fps_str = video_stream.get('r_frame_rate', '24/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                result['fps'] = num / den if den else 24.0
            else:
                result['fps'] = float(fps_str) if fps_str else 24.0

            # Duration
            result['duration'] = float(data.get('format', {}).get('duration', 0))

            # Codec
            result['codec'] = video_stream.get('codec_name', 'unknown')

            # Bitrate
            bitrate = data.get('format', {}).get('bit_rate', 0)
            result['bitrate'] = int(bitrate) if bitrate else 0

    except Exception:
        pass

    if progress_callback:
        progress_callback(0.4, "Analyzing content...")

    # Try content analyzer if available
    try:
        from ...processors.analysis.content_analyzer import ContentAnalyzer, AnalyzerConfig

        config = AnalyzerConfig(
            sample_rate=200,
            max_samples=10,
            quick_sample_count=5,
        )
        analyzer = ContentAnalyzer(config)
        analysis = analyzer.quick_analyze(video_path)

        result['content_type'] = analysis.content_type.value
        result['has_faces'] = analysis.has_faces
        result['recommended_preset'] = analysis.recommended_preset
        result['recommended_processors'] = analysis.recommended_processors

        # Quality issues
        for deg in analysis.degradation_types:
            result['quality_issues'].append(deg.value)

        if progress_callback:
            progress_callback(0.7, "Detecting quality issues...")

    except ImportError:
        pass
    except Exception:
        pass

    if progress_callback:
        progress_callback(0.9, "Calculating estimates...")

    # Estimate processing time (rough calculation)
    frame_count = int(result['duration'] * result['fps']) if result['fps'] > 0 else 0
    # Assume ~0.5 seconds per frame for balanced preset
    result['estimated_time'] = frame_count * 0.5

    # Estimate VRAM based on resolution
    width, height = result['resolution']
    if width > 0:
        # Rough VRAM estimate: ~2GB base + ~0.5GB per megapixel
        megapixels = (width * height) / 1_000_000
        result['estimated_vram'] = int(2000 + megapixels * 500)

    if progress_callback:
        progress_callback(1.0, "Done")

    return result


def _get_hardware_info() -> Dict[str, Any]:
    """Get hardware information."""
    result = {
        'has_gpu': False,
        'gpu_name': 'None',
        'vram_mb': 0,
        'cuda_available': False,
        'cpu_cores': 4,
        'ram_gb': 8,
    }

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            result['has_gpu'] = True
            result['cuda_available'] = True
            result['gpu_name'] = torch.cuda.get_device_name(0)
            result['vram_mb'] = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    except ImportError:
        pass
    except Exception:
        pass

    # Get CPU/RAM info
    try:
        import psutil
        result['cpu_cores'] = psutil.cpu_count(logical=False) or 4
        result['ram_gb'] = int(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        pass

    return result


def _get_preset_info(preset_name: str) -> Dict[str, Any]:
    """Get information about a preset."""
    presets = {
        'fast': {
            'name': 'Fast',
            'description': 'Quick processing with good quality',
            'scale': '2x',
            'processors': ['Upscale', 'Temporal smoothing'],
            'time_factor': 0.5,
            'vram_factor': 0.6,
        },
        'balanced': {
            'name': 'Balanced',
            'description': 'Best balance of quality and speed',
            'scale': '2x',
            'processors': ['Denoise', 'Upscale', 'Face enhancement', 'Temporal smoothing'],
            'time_factor': 1.0,
            'vram_factor': 1.0,
        },
        'best': {
            'name': 'Best Quality',
            'description': 'Maximum quality, takes longer',
            'scale': '4x',
            'processors': ['Heavy denoise', 'Upscale', 'Face enhancement', 'Color correction', 'Temporal consistency'],
            'time_factor': 2.5,
            'vram_factor': 1.5,
        },
        'anime': {
            'name': 'Anime/Animation',
            'description': 'Optimized for animated content',
            'scale': '4x',
            'processors': ['Anime upscale', 'Line enhancement', 'Temporal smoothing'],
            'time_factor': 1.5,
            'vram_factor': 1.2,
        },
        'film': {
            'name': 'Film Restoration',
            'description': 'For old film footage with defects',
            'scale': '4x',
            'processors': ['Scratch removal', 'Dust removal', 'Upscale', 'Face enhancement', 'Grain reduction'],
            'time_factor': 2.0,
            'vram_factor': 1.3,
        },
        'vhs': {
            'name': 'VHS Restoration',
            'description': 'For VHS and analog tape footage',
            'scale': '2x',
            'processors': ['Tracking repair', 'Dropout repair', 'Chroma fix', 'Denoise', 'Upscale'],
            'time_factor': 1.8,
            'vram_factor': 1.1,
        },
    }

    return presets.get(preset_name, presets['balanced'])


# =============================================================================
# RestorationWizard
# =============================================================================

class RestorationWizard:
    """Interactive restoration wizard.

    Guides users through the video restoration process with a step-by-step
    interface, providing analysis, recommendations, and customization options.

    Example:
        >>> wizard = RestorationWizard()
        >>> config = wizard.run("my_video.mp4")
        >>> print(f"Output will be saved to: {config.get_output_path()}")
    """

    def __init__(self, config: Optional[WizardConfig] = None):
        """Initialize the wizard.

        Args:
            config: Wizard configuration options
        """
        self.config = config or WizardConfig()
        self._analysis: Optional[Dict[str, Any]] = None
        self._hardware: Optional[Dict[str, Any]] = None
        self._selected_preset: str = 'balanced'
        self._selected_quality: str = 'balanced'
        self._selected_scale: int = 2
        self._selected_format: str = 'mp4'
        self._custom_processors: List[str] = []
        self._output_path: Optional[Path] = None
        self._download_folder: Optional[Path] = None
        self._youtube_url: Optional[str] = None

        # Set up color support
        global _color_enabled
        _color_enabled = self.config.color_output and Colors.supports_color()
        if _color_enabled and os.name == 'nt':
            Colors.enable_windows_ansi()

    def run(self, input_path: Union[str, Path, None] = None) -> Any:
        """Run the wizard for the given input video.

        Args:
            input_path: Path to the input video file, or None to prompt for input

        Returns:
            Config object configured based on wizard selections
        """
        # Clear screen and show header
        clear_screen()
        print_header("FrameWright Restoration Wizard")

        # Step 1: Input selection (local file or YouTube)
        resolved_path = self._step_input_selection(input_path)
        if resolved_path is None:
            return None

        input_path = resolved_path

        # Step 2: Analyze
        if not self._step_analyze(input_path):
            return None

        # Step 3: Recommend
        self._step_recommend()

        # Step 4: Customize (optional)
        self._step_customize()

        # Step 5: Preview (optional)
        if self.config.show_preview:
            self._step_preview(input_path)

        # Step 6: Confirm
        result = self._step_confirm(input_path)

        return result

    def _step_input_selection(self, input_path: Union[str, Path, None]) -> Optional[Path]:
        """Step 1: Select input source (local file or YouTube).

        Args:
            input_path: Pre-provided path, or None to prompt

        Returns:
            Path to video file (downloaded if YouTube), or None on failure
        """
        print_step(1, "INPUT SOURCE", 6)

        # If a valid local path was provided, use it
        if input_path is not None:
            path = Path(input_path)
            if path.exists():
                print_info("Input", str(path))
                print_success("Local file found")
                wait_for_key()
                return path
            # Check if it's a YouTube URL
            if is_youtube_url(str(input_path)):
                return self._handle_youtube_input(str(input_path))
            print_error(f"File not found: {input_path}")
            print()

        # Ask user for input source
        print(f"  {_c('How would you like to provide your video?', Colors.CYAN)}")
        print()

        source_choice = ask_choice(
            "Select input source:",
            [
                ('1', 'Local file - Select a video file from your computer'),
                ('2', 'YouTube URL - Download a video from YouTube'),
            ],
            default='1'
        )

        print()

        if source_choice == '1':
            return self._select_local_file()
        else:
            return self._handle_youtube_input()

    def _select_local_file(self) -> Optional[Path]:
        """Prompt user to select a local video file.

        Returns:
            Path to the video file, or None if cancelled
        """
        print(f"  {_c('Enter the path to your video file:', Colors.BOLD)}")
        print(f"  {_c('(You can drag and drop the file here)', Colors.DIM)}")
        print()

        while True:
            try:
                path_str = ask_input("Video file path")
                if not path_str:
                    if ask_yes_no("Cancel wizard?", default=False):
                        return None
                    continue

                # Clean up the path (remove quotes if drag-dropped)
                path_str = path_str.strip().strip('"').strip("'")
                path = Path(path_str)

                if not path.exists():
                    print_error(f"File not found: {path}")
                    continue

                if not path.is_file():
                    print_error(f"Not a file: {path}")
                    continue

                # Check if it's a video file
                video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.ts', '.mts'}
                if path.suffix.lower() not in video_extensions:
                    print_warning(f"'{path.suffix}' may not be a video file")
                    if not ask_yes_no("Continue anyway?", default=False):
                        continue

                print()
                print_success(f"Selected: {path.name}")
                wait_for_key()
                return path

            except (EOFError, KeyboardInterrupt):
                print()
                return None

    def _handle_youtube_input(self, url: Optional[str] = None) -> Optional[Path]:
        """Handle YouTube URL input and download.

        Args:
            url: Pre-provided YouTube URL, or None to prompt

        Returns:
            Path to downloaded video, or None on failure
        """
        # Get YouTube URL if not provided
        if url is None:
            print(f"  {_c('Enter the YouTube video URL:', Colors.BOLD)}")
            print(f"  {_c('Supported: youtube.com/watch?v=..., youtu.be/..., shorts/', Colors.DIM)}")
            print()

            while True:
                url = ask_input("YouTube URL")
                if not url:
                    if ask_yes_no("Cancel wizard?", default=False):
                        return None
                    continue

                if not is_youtube_url(url):
                    print_error("Invalid YouTube URL. Please enter a valid URL.")
                    continue

                break

        self._youtube_url = url
        print()
        print_info("YouTube URL", url[:60] + "..." if len(url) > 60 else url)
        print()

        # Step 1b: Select download folder
        print(f"  {_c('Where would you like to save the downloaded video?', Colors.BOLD)}")
        print()

        # Determine default folder
        default_folder = str(Path.home() / "Videos" / "FrameWright")

        # Show folder selection options
        folder_choice = ask_choice(
            "Select download folder:",
            [
                ('1', f'Default - {default_folder}'),
                ('2', 'Current directory'),
                ('3', 'Custom folder - Choose your own location'),
            ],
            default='1'
        )

        print()

        if folder_choice == '1':
            download_folder = Path(default_folder)
            download_folder.mkdir(parents=True, exist_ok=True)
        elif folder_choice == '2':
            download_folder = Path.cwd()
        else:
            download_folder = ask_folder("Enter folder path", default_folder)

        self._download_folder = download_folder
        print()
        print_info("Download folder", str(download_folder))
        print()

        # Select download quality
        print(f"  {_c('Select download quality:', Colors.BOLD)}")
        print()

        quality_choice = ask_choice(
            "Quality:",
            [
                ('1', 'Best available - Highest quality (recommended for restoration)'),
                ('2', '1080p - Full HD'),
                ('3', '720p - HD'),
                ('4', '480p - Standard definition'),
            ],
            default='1'
        )

        quality_map = {'1': 'best', '2': '1080p', '3': '720p', '4': '480p'}
        quality = quality_map.get(quality_choice, 'best')

        print()

        # Download the video
        print(f"  {_c('Downloading video from YouTube...', Colors.CYAN)}")
        print()

        def progress_callback(progress: float, message: str) -> None:
            print_progress(progress, message)

        downloaded_path = download_youtube_video(
            url,
            download_folder,
            quality=quality,
            progress_callback=progress_callback,
        )

        print()  # New line after progress bar
        print()

        if downloaded_path is None:
            print_error("Failed to download video from YouTube")
            if ask_yes_no("Try again?", default=True):
                return self._handle_youtube_input()
            return None

        print_success(f"Downloaded: {downloaded_path.name}")
        print_info("Location", str(downloaded_path))
        print()
        wait_for_key()

        return downloaded_path

    def _step_analyze(self, input_path: Path) -> bool:
        """Step 2: Analyze the video."""
        clear_screen()
        print_header("FrameWright Restoration Wizard")
        print_step(2, "ANALYZE", 6)
        print(f"  {_c('Analyzing your video...', Colors.CYAN)}")
        print()

        # Show progress animation
        def progress_callback(progress: float, message: str) -> None:
            print_progress(progress, message)

        # Get hardware info
        print(f"  {_c('Detecting hardware...', Colors.DIM)}")
        self._hardware = _get_hardware_info()

        # Analyze video
        self._analysis = _analyze_video(input_path, progress_callback)
        print()  # New line after progress bar
        print()

        if self._analysis['duration'] <= 0:
            print_error("Could not read video file. Please check the file format.")
            return False

        # Display analysis results
        print(f"  {_c('Video Information:', Colors.BOLD)}")
        print()

        width, height = self._analysis['resolution']
        resolution = f"{width}x{height}" if width > 0 else "Unknown"
        duration = format_time(self._analysis['duration'])
        fps = f"{self._analysis['fps']:.2f}" if self._analysis['fps'] > 0 else "Unknown"

        print_info("File", self._analysis['filename'])
        print_info("Duration", duration)
        print_info("Resolution", resolution)
        print_info("Frame Rate", f"{fps} fps")
        print_info("Codec", self._analysis['codec'])
        print_info("Size", format_size(self._analysis['file_size']))

        print()

        # Show detected content type
        if self._analysis['content_type'] != 'unknown':
            print(f"  {_c('Content Analysis:', Colors.BOLD)}")
            print()
            print_info("Content Type", self._analysis['content_type'].replace('_', ' ').title())

            if self._analysis['has_faces']:
                print_info("Faces Detected", "Yes")

            if self._analysis['quality_issues']:
                issues = ', '.join(self._analysis['quality_issues'][:3])
                print_info("Quality Issues", issues)

        print()

        # Show hardware info if verbose
        if self.config.verbose:
            print(f"  {_c('Your Hardware:', Colors.BOLD)}")
            print()
            if self._hardware['has_gpu']:
                vram_gb = self._hardware['vram_mb'] / 1024
                print_info("GPU", f"{self._hardware['gpu_name']} ({vram_gb:.1f} GB)")
            else:
                print_info("GPU", "None (CPU mode)")
            print_info("CPU Cores", str(self._hardware['cpu_cores']))
            print_info("RAM", f"{self._hardware['ram_gb']} GB")
            print()

        wait_for_key()
        return True

    def _step_recommend(self) -> None:
        """Step 3: Show recommendations."""
        clear_screen()
        print_header("FrameWright Restoration Wizard")
        print_step(3, "RECOMMEND", 6)

        # Determine recommended preset
        self._selected_preset = self._analysis.get('recommended_preset', 'balanced')
        preset_info = _get_preset_info(self._selected_preset)

        print(f"  {_c('Based on analysis, we recommend:', Colors.CYAN)}")
        print()

        # Show preset recommendation
        print(f"  {_c('Recommended Preset:', Colors.BOLD)} {_c(preset_info['name'], Colors.GREEN, Colors.BOLD)}")
        print(f"  {_c(preset_info['description'], Colors.DIM)}")
        print()

        # Show what will be done
        print(f"  {_c('Processing Pipeline:', Colors.BOLD)}")
        for i, processor in enumerate(preset_info['processors'], 1):
            print(f"    {_c(str(i) + '.', Colors.DIM)} {processor}")
        print()

        # Show estimates
        base_time = self._analysis.get('estimated_time', 0)
        estimated_time = base_time * preset_info['time_factor']

        base_vram = self._analysis.get('estimated_vram', 2000)
        estimated_vram = int(base_vram * preset_info['vram_factor'])

        print(f"  {_c('Estimates:', Colors.BOLD)}")
        print_info("Processing Time", f"~{format_time(estimated_time)}")
        print_info("VRAM Required", f"~{estimated_vram} MB")
        print_info("Output Scale", preset_info['scale'])

        # Check VRAM warning
        if self._hardware and self._hardware['has_gpu']:
            available_vram = self._hardware['vram_mb']
            if estimated_vram > available_vram * 0.9:
                print()
                print_warning(f"This may require more VRAM than available ({available_vram} MB)")
                print_warning("Consider using the 'fast' preset or lower scale")
        elif not self._hardware or not self._hardware['has_gpu']:
            print()
            print_warning("No GPU detected. Processing will be slower.")

        print()
        wait_for_key()

    def _step_customize(self) -> None:
        """Step 4: Customize settings (optional)."""
        clear_screen()
        print_header("FrameWright Restoration Wizard")
        print_step(4, "CUSTOMIZE", 6)

        if not ask_yes_no("Would you like to customize the settings?", default=False):
            return

        print()

        # Quality selection
        quality_choice = ask_choice(
            "Select quality level:",
            [
                ('f', 'Fast - Quick results, good quality'),
                ('b', 'Balanced - Best balance of quality and speed'),
                ('t', 'Best - Maximum quality, takes longer'),
            ],
            default='b'
        )

        quality_map = {'f': 'fast', 'b': 'balanced', 't': 'best'}
        self._selected_quality = quality_map.get(quality_choice, 'balanced')

        # Update preset based on content type and quality
        if self._analysis['content_type'] == 'animation':
            self._selected_preset = 'anime'
        elif self._analysis['content_type'] == 'vhs':
            self._selected_preset = 'vhs'
        elif 'scratches' in self._analysis.get('quality_issues', []):
            self._selected_preset = 'film'
        else:
            self._selected_preset = self._selected_quality

        print()

        # Scale selection
        scale_choice = ask_choice(
            "Select upscale factor:",
            [
                ('2', '2x - Double resolution'),
                ('4', '4x - Quadruple resolution'),
            ],
            default='2'
        )
        self._selected_scale = int(scale_choice)

        print()

        # Output format
        format_choice = ask_choice(
            "Select output format:",
            [
                ('1', 'MP4 - Best compatibility'),
                ('2', 'MKV - Better quality, larger files'),
                ('3', 'MOV - Apple ecosystem'),
            ],
            default='1'
        )
        format_map = {'1': 'mp4', '2': 'mkv', '3': 'mov'}
        self._selected_format = format_map.get(format_choice, 'mp4')

        print()

        # Processor toggles
        if ask_yes_no("Would you like to enable/disable specific processors?", default=False):
            print()
            self._customize_processors()

        print()
        print_success("Settings customized")
        wait_for_key()

    def _customize_processors(self) -> None:
        """Show processor toggle options."""
        processors = [
            ('denoise', 'Noise Reduction', True),
            ('upscale', 'AI Upscaling', True),
            ('face', 'Face Enhancement', self._analysis.get('has_faces', False)),
            ('color', 'Color Correction', False),
            ('temporal', 'Temporal Consistency', True),
            ('interpolate', 'Frame Interpolation', False),
        ]

        print(f"  {_c('Toggle processors:', Colors.BOLD)}")
        print()

        enabled = []
        for key, name, default in processors:
            if ask_yes_no(f"Enable {name}?", default=default):
                enabled.append(key)

        self._custom_processors = enabled

    def _step_preview(self, input_path: Path) -> None:
        """Step 5: Show preview (optional)."""
        clear_screen()
        print_header("FrameWright Restoration Wizard")
        print_step(5, "PREVIEW", 6)

        if not ask_yes_no("Would you like to see a preview?", default=False):
            return

        print()
        print(f"  {_c('Generating preview...', Colors.CYAN)}")
        print(f"  {_c('This will process a 5-second segment to show the difference.', Colors.DIM)}")
        print()

        # Try to launch preview
        try:
            # Check if preview server is available
            from ..preview import RealTimePreview, PreviewConfig, PreviewBackend

            # Start web preview
            config = PreviewConfig(
                backend=PreviewBackend.WEB,
                web_port=8765,
            )

            preview = RealTimePreview(config)
            if preview.start():
                print()
                print_success("Preview server started")
                print_info("Open in browser", "http://localhost:8765")
                print()
                print(f"  {_c('The preview will show before/after comparison.', Colors.DIM)}")
                print(f"  {_c('Press Enter when ready to continue...', Colors.DIM)}")
                wait_for_key()
                preview.stop()
            else:
                print_warning("Could not start preview server")

        except ImportError:
            print_warning("Preview module not available")
        except Exception as e:
            print_warning(f"Preview failed: {e}")

        wait_for_key()

    def _step_confirm(self, input_path: Path) -> Any:
        """Step 6: Confirm and start processing."""
        clear_screen()
        print_header("FrameWright Restoration Wizard")
        print_step(6, "CONFIRM", 6)

        # Determine output path
        if self._output_path is None:
            self._output_path = input_path.parent / f"{input_path.stem}_restored.{self._selected_format}"

        # Show final settings summary
        preset_info = _get_preset_info(self._selected_preset)

        print(f"  {_c('Final Settings:', Colors.BOLD)}")
        print()
        print_info("Input", str(input_path))
        print_info("Output", str(self._output_path))
        print_info("Preset", preset_info['name'])
        print_info("Quality", self._selected_quality.title())
        print_info("Scale", f"{self._selected_scale}x")
        print_info("Format", self._selected_format.upper())

        if self._custom_processors:
            print_info("Processors", ', '.join(self._custom_processors))

        # Show estimates
        base_time = self._analysis.get('estimated_time', 0)
        time_factor = {'fast': 0.5, 'balanced': 1.0, 'best': 2.5}.get(self._selected_quality, 1.0)
        estimated_time = base_time * time_factor

        print()
        print_info("Estimated Time", f"~{format_time(estimated_time)}")

        print()
        print(f"  {_c('[Enter]', Colors.CYAN, Colors.BOLD)} Start processing")
        print(f"  {_c('[c]', Colors.CYAN, Colors.BOLD)}     Customize more")
        print(f"  {_c('[q]', Colors.CYAN, Colors.BOLD)}     Quit")
        print()

        while True:
            try:
                choice = input(f"  {_c('Ready?', Colors.BOLD)} ").strip().lower()

                if choice == '' or choice == 'y' or choice == 'yes':
                    # Create and return config
                    return self._create_config(input_path)

                elif choice == 'c':
                    # Go back to customize
                    self._step_customize()
                    return self._step_confirm(input_path)

                elif choice == 'q':
                    print()
                    print_warning("Wizard cancelled")
                    return None

                else:
                    print(f"    {_c('Press Enter to start, c to customize, or q to quit', Colors.DIM)}")

            except (EOFError, KeyboardInterrupt):
                print()
                print_warning("Wizard cancelled")
                return None

    def _create_config(self, input_path: Path) -> Any:
        """Create Config object from wizard selections."""
        try:
            from ...config import Config

            # Map preset to config values
            preset_map = {
                'fast': {'crf': 23, 'preset': 'fast', 'parallel_frames': 4},
                'balanced': {'crf': 18, 'preset': 'medium', 'parallel_frames': 2},
                'best': {'crf': 16, 'preset': 'slow', 'parallel_frames': 1},
                'anime': {'crf': 18, 'preset': 'medium', 'parallel_frames': 2, 'model_name': 'realesr-animevideov3'},
                'film': {'crf': 16, 'preset': 'slow', 'parallel_frames': 2, 'enable_auto_enhance': True, 'auto_defect_repair': True},
                'vhs': {'crf': 18, 'preset': 'medium', 'parallel_frames': 2, 'enable_vhs_restoration': True},
            }

            settings = preset_map.get(self._selected_preset, preset_map['balanced'])

            # Set model based on scale
            if self._selected_scale == 2:
                settings['scale_factor'] = 2
                if 'model_name' not in settings:
                    settings['model_name'] = 'realesrgan-x2plus'
            else:
                settings['scale_factor'] = 4
                if 'model_name' not in settings or settings['model_name'] == 'realesrgan-x2plus':
                    settings['model_name'] = 'realesrgan-x4plus'

            # Set output format
            settings['output_format'] = self._selected_format

            # Create project directory
            project_dir = input_path.parent / '.framewright_work'

            # Create config
            config = Config(
                project_dir=project_dir,
                output_dir=self._output_path.parent if self._output_path else None,
                **settings
            )

            return config

        except ImportError:
            print_error("Could not import Config module")
            return None


# =============================================================================
# Factory Functions
# =============================================================================

def run_wizard(input_path: Union[str, Path, None] = None, config: Optional[WizardConfig] = None) -> Any:
    """Run the restoration wizard for a video file.

    This is the main entry point for the wizard. It guides the user through
    selecting input (local file or YouTube URL), analyzing the video,
    reviewing recommendations, customizing settings, and confirming the restoration.

    Args:
        input_path: Path to the input video file, YouTube URL, or None to prompt
        config: Optional wizard configuration

    Returns:
        Config object configured based on user selections, or None if cancelled

    Example:
        >>> # Interactive mode - prompts for input
        >>> config = run_wizard()

        >>> # With local file
        >>> config = run_wizard("my_video.mp4")

        >>> # With YouTube URL
        >>> config = run_wizard("https://youtube.com/watch?v=...")

        >>> if config:
        ...     from framewright.restorer import VideoRestorer
        ...     restorer = VideoRestorer(config)
        ...     restorer.restore("my_video.mp4", "output.mp4")
    """
    wizard = RestorationWizard(config)
    return wizard.run(input_path)


def create_wizard(config: Optional[WizardConfig] = None) -> RestorationWizard:
    """Create a RestorationWizard instance.

    Args:
        config: Optional wizard configuration

    Returns:
        Configured RestorationWizard instance
    """
    return RestorationWizard(config)


# =============================================================================
# CLI Integration Helper
# =============================================================================

def integrate_wizard_to_cli() -> None:
    """Add wizard support to the main CLI.

    This function is called during CLI initialization to register
    the --wizard flag.
    """
    # This is a placeholder - actual integration happens in cli.py
    pass


__all__ = [
    'WizardConfig',
    'RestorationWizard',
    'run_wizard',
    'create_wizard',
    # YouTube helpers
    'is_youtube_url',
    'download_youtube_video',
    # UI helpers
    'clear_screen',
    'print_header',
    'print_step',
    'print_option',
    'print_progress',
    'print_comparison',
    'ask_yes_no',
    'ask_choice',
    'ask_folder',
]
