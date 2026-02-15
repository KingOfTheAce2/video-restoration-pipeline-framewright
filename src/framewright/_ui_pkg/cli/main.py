#!/usr/bin/env python3
"""FrameWright Simplified CLI - It Just Works.

The simplest possible interface for video restoration:

    framewright video.mp4

That's it. Everything else is automatic.

Optional quality preferences:
    framewright video.mp4 --quality fast|balanced|best

Optional style preferences:
    framewright video.mp4 --style film|animation|home-video|archive

Built on click for modern CLI experience with human-readable output.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
import logging

import click

# Configure logging to be quiet by default
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

QUALITY_PRESETS = {
    "fast": {
        "description": "Quick results, good quality",
        "scale_factor": 2,
        "model_name": "realesrgan-x2plus",
        "crf": 23,
        "preset": "fast",
        "parallel_frames": 4,
        "enable_checkpointing": False,
    },
    "balanced": {
        "description": "Best balance of quality and speed",
        "scale_factor": 2,
        "model_name": "realesrgan-x2plus",
        "crf": 18,
        "preset": "medium",
        "parallel_frames": 2,
        "enable_checkpointing": True,
    },
    "best": {
        "description": "Maximum quality, takes longer",
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 16,
        "preset": "slow",
        "parallel_frames": 1,
        "enable_checkpointing": True,
        "enable_validation": True,
    },
}

STYLE_PRESETS = {
    "film": {
        "description": "Classic film footage",
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "scratch_sensitivity": 0.7,
        "grain_reduction": 0.3,
        "auto_face_restore": True,
    },
    "animation": {
        "description": "Cartoons and anime",
        "model_name": "realesr-animevideov3",
        "grain_reduction": 0.0,
        "auto_face_restore": False,
    },
    "home-video": {
        "description": "Personal recordings, VHS, camcorder",
        "enable_auto_enhance": True,
        "grain_reduction": 0.5,
        "auto_face_restore": True,
        "enable_interpolation": True,
        "target_fps": 30,
    },
    "archive": {
        "description": "Historic/archival footage",
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "scratch_sensitivity": 0.85,
        "dust_sensitivity": 0.75,
        "grain_reduction": 0.2,
        "auto_face_restore": True,
        "enable_deduplication": True,
    },
}

# Human-readable stage names
STAGE_NAMES = {
    "download": "Downloading video",
    "extract_audio": "Extracting audio",
    "extract_frames": "Extracting frames",
    "analyze": "Analyzing content",
    "denoise": "Reducing noise",
    "enhance": "Enhancing frames",
    "face_restore": "Making faces look natural",
    "interpolate": "Smoothing motion",
    "temporal": "Ensuring consistency",
    "colorize": "Adding color",
    "reassemble": "Assembling final video",
    "validate": "Verifying quality",
}


# =============================================================================
# Terminal Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'

    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports color."""
        import os
        if os.name == 'nt':
            # Windows - check for modern terminal
            return os.environ.get('TERM_PROGRAM') == 'vscode' or \
                   os.environ.get('WT_SESSION') is not None or \
                   'ANSICON' in os.environ
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if Colors.supports_color():
        return f"{color}{text}{Colors.RESET}"
    return text


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 0:
        return "calculating..."

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    elif minutes > 0:
        return f"{minutes}m {int(seconds % 60):02d}s"
    else:
        return f"{int(seconds)}s"


def create_progress_bar(progress: float, width: int = 24) -> str:
    """Create a simple text progress bar."""
    filled = int(width * progress)
    empty = width - filled
    bar = colorize("█" * filled, Colors.GREEN) + colorize("░" * empty, Colors.DIM)
    return bar


# =============================================================================
# Hardware Detection
# =============================================================================

def detect_hardware() -> Dict[str, Any]:
    """Auto-detect hardware capabilities."""
    try:
        from ..auto_detect import SmartAnalyzer
        from ...hardware import get_gpu_capability, get_system_info

        gpu = get_gpu_capability()
        system = get_system_info()

        return {
            "has_gpu": gpu.has_gpu,
            "gpu_name": gpu.gpu_name,
            "vram_mb": gpu.vram_total_mb,
            "vram_free_mb": gpu.vram_free_mb,
            "recommended_tile_size": gpu.recommended_tile_size,
            "cpu_cores": system.cpu_cores,
            "ram_gb": system.ram_total_gb,
        }
    except ImportError:
        # Fallback if hardware module unavailable
        return {
            "has_gpu": False,
            "gpu_name": "Unknown",
            "vram_mb": 0,
            "vram_free_mb": 0,
            "recommended_tile_size": 256,
            "cpu_cores": 4,
            "ram_gb": 8,
        }


def get_optimal_settings(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Determine optimal settings based on hardware."""
    vram_gb = hardware.get("vram_mb", 0) / 1024
    settings = {}

    if not hardware.get("has_gpu"):
        # CPU mode - very conservative
        settings["tile_size"] = 128
        settings["parallel_frames"] = 1
        settings["scale_factor"] = 2
        settings["quality_tier"] = "cpu"
        return settings

    # GPU tiers based on VRAM
    if vram_gb >= 24:
        # RTX 4090 / 5090 tier
        settings["tile_size"] = None  # Full resolution
        settings["parallel_frames"] = 2
        settings["quality_tier"] = "high-end"
    elif vram_gb >= 12:
        # RTX 3080/4070 tier
        settings["tile_size"] = 512
        settings["parallel_frames"] = 2
        settings["quality_tier"] = "mid-high"
    elif vram_gb >= 8:
        # RTX 3070/4060 tier
        settings["tile_size"] = 384
        settings["parallel_frames"] = 2
        settings["quality_tier"] = "mid"
    elif vram_gb >= 6:
        # GTX 1060/RTX 3060 tier
        settings["tile_size"] = 256
        settings["parallel_frames"] = 1
        settings["quality_tier"] = "entry"
    else:
        # Low VRAM
        settings["tile_size"] = 192
        settings["parallel_frames"] = 1
        settings["quality_tier"] = "low"

    return settings


# =============================================================================
# Content Detection
# =============================================================================

def detect_content_type(video_path: Path) -> Dict[str, Any]:
    """Analyze video to detect content type and characteristics."""
    try:
        from ...ui.auto_detect import SmartAnalyzer, analyze_video_smart

        analyzer = SmartAnalyzer()
        analysis = analyzer.analyze(video_path)

        return {
            "content_type": analysis.content_profile.content_type.value,
            "is_bw": analysis.content_profile.is_black_and_white,
            "has_faces": analysis.content_profile.has_faces,
            "degradation_severity": analysis.degradation_profile.severity,
            "primary_issues": analysis.degradation_profile.primary_issues[:3],
            "duration_seconds": analysis.video_info.get("duration", 0),
            "resolution": analysis.video_info.get("resolution", "unknown"),
            "fps": analysis.video_info.get("fps", 30),
        }
    except Exception as e:
        logger.debug(f"Content detection failed: {e}")
        # Return minimal defaults
        return {
            "content_type": "unknown",
            "is_bw": False,
            "has_faces": True,
            "degradation_severity": "moderate",
            "primary_issues": [],
            "duration_seconds": 0,
            "resolution": "unknown",
            "fps": 30,
        }


# =============================================================================
# Progress Display
# =============================================================================

class SimpleProgressDisplay:
    """Human-readable progress display."""

    def __init__(self, video_name: str):
        self.video_name = video_name
        self.start_time = time.time()
        self.current_stage = ""
        self.current_stage_num = 0
        self.total_stages = 5
        self.progress = 0.0
        self.hardware_status = ""
        self._last_print_len = 0

    def set_hardware_status(self, gpu_name: str, gpu_utilization: float = 0):
        """Set hardware status for display."""
        if gpu_utilization > 0:
            status = "Working smoothly" if gpu_utilization > 50 else "Warming up"
            self.hardware_status = f"{status} at {gpu_utilization:.0f}%"
        else:
            self.hardware_status = gpu_name

    def update(self, stage: str, progress: float, stage_num: int, total_stages: int):
        """Update progress display."""
        self.current_stage = STAGE_NAMES.get(stage, stage.replace("_", " ").title())
        self.current_stage_num = stage_num
        self.total_stages = total_stages
        self.progress = progress
        self._print_progress()

    def _print_progress(self):
        """Print the progress to terminal."""
        # Calculate times
        elapsed = time.time() - self.start_time
        if self.progress > 0.05:
            eta = (elapsed / self.progress) * (1 - self.progress)
        else:
            eta = -1  # Unknown

        # Clear previous output
        if self._last_print_len > 0:
            # Move cursor up and clear lines
            sys.stdout.write(f"\033[{5}A")  # Move up 5 lines
            sys.stdout.write("\033[J")  # Clear from cursor to end

        # Build display
        pct = int(self.progress * 100)
        bar = create_progress_bar(self.progress)

        lines = [
            "",
            f"{bar} {colorize(f'{pct}%', Colors.BOLD)}",
            "",
            f"{colorize('Stage:', Colors.DIM)} {self.current_stage} ({self.current_stage_num} of {self.total_stages})",
            f"{colorize('Time:', Colors.DIM)} {format_time(elapsed)} done → ~{format_time(eta)} remaining",
        ]

        if self.hardware_status:
            lines.append(f"{colorize('GPU:', Colors.DIM)} {self.hardware_status}")

        output = "\n".join(lines)
        print(output)
        self._last_print_len = len(lines)

    def complete(self, output_path: Path):
        """Show completion message."""
        elapsed = time.time() - self.start_time

        print()
        print(colorize("Done!", Colors.GREEN + Colors.BOLD))
        print()
        print(f"  {colorize('Output:', Colors.DIM)} {output_path}")
        print(f"  {colorize('Time:', Colors.DIM)}   {format_time(elapsed)}")
        print()


# =============================================================================
# Error Handling
# =============================================================================

class HelpfulError:
    """Create helpful, human-readable error messages."""

    @staticmethod
    def gpu_memory(gpu_name: str, vram_gb: float, needed_gb: float) -> str:
        """GPU memory error with solutions."""
        return f"""
{colorize('Not enough GPU memory', Colors.RED + Colors.BOLD)}

Your {gpu_name} ({vram_gb:.0f}GB) needs more memory for this video.

{colorize('Quick fixes:', Colors.YELLOW)}
  framewright video.mp4 --quality fast

{colorize('Need help?', Colors.DIM)} framewright help gpu-memory
"""

    @staticmethod
    def file_not_found(path: str) -> str:
        """File not found error."""
        return f"""
{colorize('Video not found', Colors.RED + Colors.BOLD)}

Could not find: {path}

{colorize('Check that:', Colors.YELLOW)}
  - The file path is correct
  - The file exists
  - You have permission to read it
"""

    @staticmethod
    def ffmpeg_missing() -> str:
        """FFmpeg missing error."""
        return f"""
{colorize('FFmpeg not found', Colors.RED + Colors.BOLD)}

FrameWright needs FFmpeg to process videos.

{colorize('Install FFmpeg:', Colors.YELLOW)}
  Windows: winget install ffmpeg
  macOS:   brew install ffmpeg
  Linux:   sudo apt install ffmpeg

{colorize('Need help?', Colors.DIM)} framewright help install
"""

    @staticmethod
    def disk_space(needed_gb: float, available_gb: float) -> str:
        """Disk space error."""
        return f"""
{colorize('Not enough disk space', Colors.RED + Colors.BOLD)}

This video needs ~{needed_gb:.1f}GB but only {available_gb:.1f}GB available.

{colorize('Quick fixes:', Colors.YELLOW)}
  - Free up disk space
  - Use a different output location: framewright video.mp4 -o /other/drive/output.mp4
  - Use lower quality: framewright video.mp4 --quality fast
"""

    @staticmethod
    def generic(error: Exception) -> str:
        """Generic error with helpful context."""
        return f"""
{colorize('Something went wrong', Colors.RED + Colors.BOLD)}

{str(error)}

{colorize('Try:', Colors.YELLOW)}
  - Check the video file isn't corrupted
  - Run: framewright doctor
  - Get help: framewright help troubleshoot
"""


# =============================================================================
# Main Restoration Function
# =============================================================================

def run_restoration(
    input_path: Path,
    output_path: Optional[Path],
    quality: str,
    style: Optional[str],
    verbose: bool = False,
) -> int:
    """Run the video restoration pipeline."""

    # Validate input
    if not input_path.exists():
        click.echo(HelpfulError.file_not_found(str(input_path)))
        return 1

    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_restored.mp4"

    # Print header
    click.echo()
    click.echo(colorize("Enhancing your video...", Colors.CYAN + Colors.BOLD))
    click.echo()

    # Detect hardware
    hardware = detect_hardware()
    optimal = get_optimal_settings(hardware)

    if verbose:
        gpu_info = f"{hardware['gpu_name']} ({hardware['vram_mb']/1024:.0f}GB)" if hardware['has_gpu'] else "CPU mode"
        click.echo(f"{colorize('Hardware:', Colors.DIM)} {gpu_info}")

    # Detect content (quick analysis)
    content = detect_content_type(input_path)

    if verbose:
        click.echo(f"{colorize('Content:', Colors.DIM)} {content['content_type']}, {content['degradation_severity']} degradation")
        click.echo()

    # Build configuration
    config_dict = {}

    # Apply quality preset
    quality_preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    config_dict.update({k: v for k, v in quality_preset.items() if k != "description"})

    # Apply style preset if specified
    if style:
        style_preset = STYLE_PRESETS.get(style, {})
        config_dict.update({k: v for k, v in style_preset.items() if k != "description"})

    # Apply hardware-optimal settings
    if optimal.get("tile_size"):
        config_dict["tile_size"] = optimal["tile_size"]

    # Auto-detect style from content if not specified
    if not style:
        if content["content_type"] == "animation":
            config_dict["model_name"] = "realesr-animevideov3"
        if content["is_bw"]:
            config_dict["enable_colorization"] = False  # Don't auto-colorize

    # Setup progress display
    progress_display = SimpleProgressDisplay(input_path.name)
    if hardware.get("has_gpu"):
        progress_display.set_hardware_status(hardware["gpu_name"])

    try:
        # Import restoration components
        from ...config import Config
        from ...restorer import VideoRestorer, ProgressInfo

        # Create working directory
        work_dir = output_path.parent / ".framewright_work"
        config_dict["project_dir"] = work_dir
        config_dict["output_dir"] = output_path.parent

        # Create config
        config = Config(**config_dict)

        # Progress callback
        stage_mapping = {
            "download": (1, 5),
            "extract_audio": (1, 5),
            "extract_frames": (2, 5),
            "analyze": (2, 5),
            "enhance": (3, 5),
            "face_restore": (3, 5),
            "interpolate": (4, 5),
            "temporal": (4, 5),
            "reassemble": (5, 5),
            "validate": (5, 5),
        }

        def progress_callback(info):
            """Handle progress updates from restorer."""
            if isinstance(info, tuple):
                stage, progress = info
            else:
                stage = info.stage
                progress = info.progress

            stage_num, total = stage_mapping.get(stage, (1, 5))

            # Calculate overall progress
            stage_weight = 1.0 / total
            overall = ((stage_num - 1) / total) + (progress * stage_weight)

            progress_display.update(stage, overall, stage_num, total)

        # Create restorer and run
        restorer = VideoRestorer(config, progress_callback=progress_callback)
        result = restorer.restore(str(input_path), str(output_path))

        # Show completion
        progress_display.complete(output_path)

        return 0

    except ImportError as e:
        if "ffmpeg" in str(e).lower():
            click.echo(HelpfulError.ffmpeg_missing())
        else:
            click.echo(HelpfulError.generic(e))
        return 1

    except MemoryError:
        click.echo(HelpfulError.gpu_memory(
            hardware.get("gpu_name", "GPU"),
            hardware.get("vram_mb", 0) / 1024,
            8.0  # Estimated need
        ))
        return 1

    except Exception as e:
        if "VRAM" in str(e) or "memory" in str(e).lower():
            click.echo(HelpfulError.gpu_memory(
                hardware.get("gpu_name", "GPU"),
                hardware.get("vram_mb", 0) / 1024,
                8.0
            ))
        elif "disk" in str(e).lower() or "space" in str(e).lower():
            click.echo(HelpfulError.disk_space(10.0, 1.0))
        else:
            click.echo(HelpfulError.generic(e))
            if verbose:
                import traceback
                click.echo(traceback.format_exc())
        return 1


# =============================================================================
# CLI Commands
# =============================================================================

@click.group(invoke_without_command=True)
@click.argument('input_video', type=click.Path(exists=False), required=False)
@click.option(
    '--quality', '-q',
    type=click.Choice(['fast', 'balanced', 'best']),
    default='balanced',
    help='Quality preference: fast (quick), balanced (default), best (maximum)'
)
@click.option(
    '--style', '-s',
    type=click.Choice(['film', 'animation', 'home-video', 'archive']),
    default=None,
    help='Content style: film, animation, home-video, archive (auto-detected if not specified)'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default=None,
    help='Output file path (default: input_restored.mp4)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    default=False,
    help='Show detailed progress information'
)
@click.option(
    '--wizard', '-w',
    is_flag=True,
    default=False,
    help='Launch interactive wizard for step-by-step guidance'
)
@click.pass_context
def cli(ctx, input_video, quality, style, output, verbose, wizard):
    """FrameWright - AI Video Restoration

    \b
    Just run:
        framewright video.mp4

    Everything is automatic. Your GPU is detected, content is analyzed,
    and optimal settings are chosen.

    \b
    Optional quality preference:
        framewright video.mp4 --quality fast      # Quick results
        framewright video.mp4 --quality balanced  # Default
        framewright video.mp4 --quality best      # Maximum quality

    \b
    Optional style preference:
        framewright video.mp4 --style film        # Classic film
        framewright video.mp4 --style animation   # Anime/cartoons
        framewright video.mp4 --style home-video  # VHS/camcorder
        framewright video.mp4 --style archive     # Historic footage

    \b
    Interactive wizard mode:
        framewright video.mp4 --wizard            # Step-by-step guidance
    """
    # If a subcommand is invoked, let it handle everything
    if ctx.invoked_subcommand is not None:
        return

    # If no input video provided, show help
    if input_video is None:
        click.echo(ctx.get_help())
        return

    input_path = Path(input_video)

    # Run wizard mode if requested
    if wizard:
        from .wizard import run_wizard, WizardConfig

        wizard_config = WizardConfig(
            show_preview=True,
            auto_detect=True,
            verbose=verbose,
            color_output=True,
        )

        config = run_wizard(input_path, wizard_config)

        if config is None:
            sys.exit(0)  # User cancelled

        # After wizard, run restoration with the config
        output_path = Path(output) if output else None
        try:
            from ...config import Config
            from ...restorer import VideoRestorer

            # Create progress display
            progress_display = SimpleProgressDisplay(input_path.name)

            # Progress callback
            def progress_callback(info):
                if isinstance(info, tuple):
                    stage, progress = info
                else:
                    stage = info.stage
                    progress = info.progress
                progress_display.update(stage, progress, 1, 5)

            restorer = VideoRestorer(config, progress_callback=progress_callback)

            # Determine output path
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_restored.{config.output_format}"

            result = restorer.restore(str(input_path), str(output_path))
            progress_display.complete(output_path)
            sys.exit(0)

        except Exception as e:
            click.echo(HelpfulError.generic(e))
            if verbose:
                import traceback
                click.echo(traceback.format_exc())
            sys.exit(1)

    # Run standard restoration
    output_path = Path(output) if output else None
    sys.exit(run_restoration(input_path, output_path, quality, style, verbose))


@cli.command()
def doctor():
    """Check system readiness and diagnose issues."""
    click.echo()
    click.echo(colorize("FrameWright System Check", Colors.CYAN + Colors.BOLD))
    click.echo()

    issues = []

    # Check FFmpeg
    import shutil
    if shutil.which("ffmpeg"):
        click.echo(colorize("  [OK]", Colors.GREEN) + " FFmpeg installed")
    else:
        click.echo(colorize("  [X]", Colors.RED) + " FFmpeg not found")
        issues.append("ffmpeg")

    # Check GPU
    hardware = detect_hardware()
    if hardware["has_gpu"]:
        vram_gb = hardware["vram_mb"] / 1024
        click.echo(colorize("  [OK]", Colors.GREEN) + f" GPU: {hardware['gpu_name']} ({vram_gb:.0f}GB)")
    else:
        click.echo(colorize("  [!]", Colors.YELLOW) + " No GPU detected (will use CPU - slower)")

    # Check Python packages
    try:
        import torch
        click.echo(colorize("  [OK]", Colors.GREEN) + f" PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            click.echo(colorize("  [OK]", Colors.GREEN) + " CUDA available")
        else:
            click.echo(colorize("  [!]", Colors.YELLOW) + " CUDA not available")
    except ImportError:
        click.echo(colorize("  [!]", Colors.YELLOW) + " PyTorch not installed (optional)")

    click.echo()

    if issues:
        click.echo(colorize("Issues found:", Colors.YELLOW))
        for issue in issues:
            click.echo(f"  - {issue}")
        click.echo()
        click.echo("Run: framewright help install")
    else:
        click.echo(colorize("System is ready!", Colors.GREEN + Colors.BOLD))

    click.echo()


@cli.command()
@click.argument('topic', required=False)
def help(topic):
    """Get help on specific topics.

    \b
    Topics:
        install       - Installation guide
        gpu-memory    - GPU memory troubleshooting
        troubleshoot  - General troubleshooting
        quality       - Quality options explained
        styles        - Style options explained
    """
    topics = {
        "install": """
FFmpeg Installation
-------------------
Windows:  winget install ffmpeg
          or download from: https://ffmpeg.org/download.html

macOS:    brew install ffmpeg

Linux:    sudo apt install ffmpeg
          or: sudo dnf install ffmpeg
""",
        "gpu-memory": """
GPU Memory Troubleshooting
--------------------------
If you're running out of GPU memory:

1. Use --quality fast
   framewright video.mp4 --quality fast

2. Close other GPU-intensive applications

3. For very large videos, process in segments

4. Use CPU mode (slower but works):
   Set CUDA_VISIBLE_DEVICES="" before running
""",
        "troubleshoot": """
General Troubleshooting
-----------------------
1. Run system check:
   framewright doctor

2. Check your video file:
   - Is the file corrupted?
   - Is the format supported? (mp4, mkv, avi, mov, webm)

3. Check disk space:
   - Need ~10x video size for processing

4. Update GPU drivers

5. Try with a shorter video first
""",
        "quality": """
Quality Options
---------------
fast:      Quick results, 2x upscale
           Best for: previews, large batches

balanced:  Good quality, reasonable time (DEFAULT)
           Best for: most videos

best:      Maximum quality, 4x upscale
           Best for: important/archival footage
""",
        "styles": """
Style Options
-------------
film:       Classic cinema footage
            Repairs scratches, restores faces

animation:  Anime, cartoons
            Uses anime-optimized AI model

home-video: VHS, camcorder recordings
            Smooths motion, reduces noise

archive:    Historic/archival footage
            Maximum restoration, preserves authenticity
""",
    }

    if topic is None:
        click.echo("\nAvailable help topics:")
        for t in topics:
            click.echo(f"  framewright help {t}")
        click.echo()
        return

    if topic in topics:
        click.echo(topics[topic])
    else:
        click.echo(f"Unknown topic: {topic}")
        click.echo("\nAvailable topics:")
        for t in topics:
            click.echo(f"  {t}")


@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
def preview(input_video):
    """Quick preview - process a single frame to see results."""
    input_path = Path(input_video)

    click.echo()
    click.echo(colorize("Generating preview...", Colors.CYAN))
    click.echo()

    try:
        from ...cli_advanced import run_preview, extract_preview_frame
        from ..terminal import Console

        console = Console()
        result = run_preview(input_path, timestamp=-1, console=console)

        if result.comparison_image and result.comparison_image.exists():
            click.echo(f"Preview saved: {result.comparison_image}")
            click.echo(f"Processing time: {result.processing_time_seconds:.1f}s per frame")

            # Try to open the image
            import platform
            import subprocess
            try:
                if platform.system() == "Windows":
                    subprocess.run(["start", str(result.comparison_image)], shell=True)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", str(result.comparison_image)])
                else:
                    subprocess.run(["xdg-open", str(result.comparison_image)])
            except Exception:
                pass
        else:
            click.echo(colorize("Preview generation failed", Colors.RED))

    except Exception as e:
        click.echo(colorize(f"Error: {e}", Colors.RED))


@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
def analyze(input_video):
    """Analyze video and show what restoration would do."""
    input_path = Path(input_video)

    click.echo()
    click.echo(colorize(f"Analyzing: {input_path.name}", Colors.CYAN + Colors.BOLD))
    click.echo()

    # Detect hardware
    hardware = detect_hardware()
    optimal = get_optimal_settings(hardware)

    # Detect content
    content = detect_content_type(input_path)

    # Display results
    click.echo(colorize("Video Information", Colors.BOLD))
    click.echo(f"  Resolution:  {content.get('resolution', 'Unknown')}")
    click.echo(f"  Frame rate:  {content.get('fps', 'Unknown')} fps")
    click.echo(f"  Duration:    {format_time(content.get('duration_seconds', 0))}")
    click.echo()

    click.echo(colorize("Content Analysis", Colors.BOLD))
    click.echo(f"  Type:        {content.get('content_type', 'Unknown')}")
    click.echo(f"  Degradation: {content.get('degradation_severity', 'Unknown')}")
    if content.get('is_bw'):
        click.echo(f"  Color:       Black & White")
    if content.get('primary_issues'):
        click.echo(f"  Issues:      {', '.join(content['primary_issues'][:3])}")
    click.echo()

    click.echo(colorize("Your Hardware", Colors.BOLD))
    if hardware.get("has_gpu"):
        vram_gb = hardware.get("vram_mb", 0) / 1024
        click.echo(f"  GPU:         {hardware.get('gpu_name')} ({vram_gb:.0f}GB)")
        click.echo(f"  Quality:     {optimal.get('quality_tier', 'Unknown')} tier")
    else:
        click.echo(f"  GPU:         None (CPU mode)")
    click.echo()

    click.echo(colorize("Recommended Command", Colors.BOLD))

    # Suggest style
    suggested_style = ""
    if content.get("content_type") == "animation":
        suggested_style = " --style animation"
    elif content.get("content_type") in ["film", "documentary"]:
        suggested_style = " --style film"
    elif content.get("content_type") == "home_video":
        suggested_style = " --style home-video"

    click.echo(f"  framewright {input_path.name}{suggested_style}")
    click.echo()


@cli.command()
@click.argument('input_video', type=click.Path(exists=True), required=False)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    default=False,
    help='Show detailed analysis during wizard'
)
@click.option(
    '--no-preview',
    is_flag=True,
    default=False,
    help='Skip preview step in wizard'
)
@click.option(
    '--no-color',
    is_flag=True,
    default=False,
    help='Disable colored output'
)
def wizard(input_video, verbose, no_preview, no_color):
    """Interactive restoration wizard with step-by-step guidance.

    \b
    Guides you through the restoration process:
    - Analyzes your video
    - Recommends optimal settings
    - Allows customization
    - Shows preview (optional)
    - Starts processing

    \b
    Examples:
        framewright wizard video.mp4
        framewright wizard video.mp4 --verbose
        framewright wizard video.mp4 --no-preview
    """
    from .wizard import run_wizard, WizardConfig

    # If no input video provided, ask for it
    if input_video is None:
        click.echo()
        click.echo(colorize("FrameWright Interactive Wizard", Colors.CYAN + Colors.BOLD))
        click.echo()
        input_video = click.prompt("Enter path to video file")

    input_path = Path(input_video)

    if not input_path.exists():
        click.echo(HelpfulError.file_not_found(str(input_path)))
        sys.exit(1)

    # Configure wizard
    wizard_config = WizardConfig(
        show_preview=not no_preview,
        auto_detect=True,
        verbose=verbose,
        color_output=not no_color,
    )

    # Run wizard
    config = run_wizard(input_path, wizard_config)

    if config is None:
        sys.exit(0)  # User cancelled

    # After wizard, run restoration with the config
    try:
        from ...restorer import VideoRestorer

        # Create progress display
        progress_display = SimpleProgressDisplay(input_path.name)

        # Progress callback
        def progress_callback(info):
            if isinstance(info, tuple):
                stage, progress = info
            else:
                stage = info.stage
                progress = info.progress
            progress_display.update(stage, progress, 1, 5)

        restorer = VideoRestorer(config, progress_callback=progress_callback)

        # Determine output path
        output_path = input_path.parent / f"{input_path.stem}_restored.{config.output_format}"

        click.echo()
        click.echo(colorize("Starting restoration...", Colors.CYAN + Colors.BOLD))
        click.echo()

        result = restorer.restore(str(input_path), str(output_path))
        progress_display.complete(output_path)
        sys.exit(0)

    except Exception as e:
        click.echo(HelpfulError.generic(e))
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
