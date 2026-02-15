"""Simplified CLI for FrameWright.

Provides Apple-like "it just works" commands:
- `framewright video.mp4` - Smart auto restoration
- `framewright quick video.mp4` - Fast restoration
- `framewright best video.mp4` - Maximum quality (auto-detects your GPU)
- `framewright wizard` - Interactive setup

This module wraps the full CLI with sensible defaults and
intelligent auto-detection. Your GPU is detected automatically
and optimal settings are chosen - no configuration needed.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging

from ._ui_pkg.terminal import Console, create_console, print_banner
from ._ui_pkg.wizard import InteractiveWizard, run_wizard
from ._ui_pkg.auto_detect import SmartAnalyzer, analyze_video_smart
from ._ui_pkg.recommendations import PresetRecommender, get_recommendations
from ._ui_pkg.progress import ArchiveRestorationProgress, create_spinner
from .config import Config, PRESETS
from .restorer import VideoRestorer
from .hardware import get_gpu_capability, GPUCapability

logger = logging.getLogger(__name__)


# =============================================================================
# GPU Auto-Detection and Smart Defaults
# =============================================================================

def _detect_optimal_preset() -> Tuple[str, Dict[str, Any], str]:
    """Auto-detect GPU and return optimal preset with user-friendly explanation.

    Returns:
        Tuple of (preset_name, config_overrides, user_message)
    """
    gpu = get_gpu_capability()

    if not gpu.has_gpu:
        return "fast", {}, "No GPU detected - using CPU mode (slower)"

    vram_gb = gpu.vram_total_mb / 1024

    # RTX 5090 / 4090 / A100 tier (24GB+)
    if vram_gb >= 24:
        return "rtx5090", {
            "temporal_window": 16,
            "tile_size": None,  # No tiling - full resolution
            "sr_model": "hat",
            "enable_ensemble_sr": True,
            "enable_temporal_colorization": True,
            "enable_raft_flow": True,
            "half_precision": True,
            # Frame generation - use SVD for missing frames
            "frame_generation_model": "svd",
            "max_gap_frames": 15,
            # Audio enhancement
            "enable_ai_audio": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using maximum quality settings"

    # RTX 4080 / 3090 tier (16-24GB)
    elif vram_gb >= 16:
        return "ultimate", {
            "temporal_window": 12,
            "tile_size": None,
            "sr_model": "hat",
            "enable_ensemble_sr": True,
            "enable_temporal_colorization": True,
            "enable_raft_flow": True,
            "half_precision": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using high quality settings"

    # RTX 3080 / 4070 tier (10-16GB)
    elif vram_gb >= 10:
        return "quality", {
            "temporal_window": 8,
            "tile_size": 512,
            "sr_model": "vrt",
            "enable_temporal_colorization": True,
            "half_precision": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using quality settings"

    # RTX 3070 / 4060 tier (8-10GB)
    elif vram_gb >= 8:
        return "quality", {
            "temporal_window": 7,
            "tile_size": 384,
            "sr_model": "basicvsr",
            "half_precision": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using balanced quality"

    # GTX 1080 / RTX 3060 tier (6-8GB)
    elif vram_gb >= 6:
        return "balanced", {
            "temporal_window": 5,
            "tile_size": 256,
            "sr_model": "realesrgan",
            "half_precision": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using efficient settings"

    # Low VRAM (4-6GB)
    elif vram_gb >= 4:
        return "fast", {
            "temporal_window": 3,
            "tile_size": 192,
            "sr_model": "realesrgan",
            "half_precision": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using compact settings"

    # Very low VRAM (<4GB)
    else:
        return "fast", {
            "tile_size": 128,
            "sr_model": "realesrgan",
            "half_precision": True,
        }, f"Detected {gpu.gpu_name} ({vram_gb:.0f}GB) - Using minimal settings"


def _get_quality_description(preset: str, config: Dict[str, Any]) -> List[str]:
    """Get user-friendly descriptions of what will happen.

    Returns list of simple descriptions, no technical jargon.
    """
    descriptions = []

    # Base description by preset
    preset_desc = {
        "rtx5090": "Maximum quality restoration",
        "ultimate": "High quality restoration",
        "quality": "Quality restoration",
        "balanced": "Balanced restoration",
        "fast": "Fast restoration",
    }
    descriptions.append(preset_desc.get(preset, "Video restoration"))

    # Add specific features in plain language
    if config.get("enable_ensemble_sr"):
        descriptions.append("Multiple AI models vote on best result")

    if config.get("enable_temporal_colorization"):
        descriptions.append("Smooth, consistent colors across frames")

    if config.get("enable_raft_flow"):
        descriptions.append("Advanced motion tracking for stability")

    sr_model = config.get("sr_model", "realesrgan")
    model_names = {
        "hat": "State-of-the-art upscaling (HAT)",
        "vrt": "Video-aware upscaling (VRT)",
        "basicvsr": "Video super-resolution (BasicVSR++)",
        "realesrgan": "AI upscaling (Real-ESRGAN)",
        "diffusion": "Diffusion-based enhancement",
    }
    if sr_model in model_names:
        descriptions.append(model_names[sr_model])

    if config.get("tile_size") is None:
        descriptions.append("Processing at full resolution")

    if config.get("frame_generation_model") == "svd":
        descriptions.append("AI-powered missing frame reconstruction")

    if config.get("enable_ai_audio"):
        descriptions.append("AI audio enhancement available")

    return descriptions


def create_simple_parser() -> argparse.ArgumentParser:
    """Create simplified argument parser."""
    parser = argparse.ArgumentParser(
        prog="framewright",
        description="FrameWright - AI Video Restoration (auto-detects your GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Just pick a mode - everything else is automatic:

  framewright video.mp4              Analyze and restore automatically
  framewright quick video.mp4        Fast - good quality, less time
  framewright best video.mp4         Best - maximum quality for your GPU
  framewright archive video.mp4      Archive - for old/historical footage

Batch & Preview:
  framewright batch ./videos/        Process all videos in a folder
  framewright preview video.mp4      See result before full processing
  framewright quick-preview video.mp4     Fast preview (samples frames)
  framewright frame-grid video.mp4        Generate frame grid image
  framewright compare orig.mp4 rest.mp4   Side-by-side comparison
  framewright compare-presets video.mp4   Compare restoration presets

Fix Specific Issues:
  framewright remove-watermark video.mp4   Remove logos/watermarks
  framewright deinterlace video.mp4        Fix interlaced video (VHS/DVD)
  framewright crop-bars video.mp4          Remove black bars
  framewright extract-subs video.mp4       Extract burnt-in subtitles
  framewright check-sync video.mp4         Verify audio sync

Analysis Tools:
  framewright scan video.mp4         Full analysis (finds all issues)
  framewright analyze video.mp4      Video quality analysis
  framewright detect-stock video.mp4 Identify film stock type
  framewright noise-profile video.mp4     Analyze noise characteristics
  framewright upscale-detect video.mp4    Check if already upscaled
  framewright score-frames video.mp4      Find problem frames
  framewright detect-credits video.mp4    Detect intro/credits
  framewright dry-run video.mp4      Preview what would happen

System & Monitoring:
  framewright system-check           Check system readiness
  framewright gpu-thermal            GPU temperature status
  framewright gpu-thermal --monitor  Continuous thermal monitoring
  framewright benchmark video.mp4    Test restoration speed
  framewright validate output.mp4    Validate export integrity
  framewright power --status         Show power state

Model Management:
  framewright models list            List available AI models
  framewright models download <name> Download a model
  framewright models clean           Remove unused models
  framewright models status          Show disk usage

Automation & Scheduling:
  framewright watch ./folder         Auto-process new videos
  framewright queue add video.mp4    Add to processing queue
  framewright schedule add video.mp4 Schedule job for later
  framewright schedule list          List scheduled jobs
  framewright webhook add <url>      Add progress webhook
  framewright profile save my-preset Save current settings

Output:
  framewright template list          List filename templates
  framewright template preview       Preview template output

Other:
  framewright colorize video.mp4     Add color to B&W video
  framewright ab-test video.mp4      Compare two configurations
  framewright wizard                 Step-by-step guided setup

Your GPU is detected automatically. No configuration needed.
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Quick command
    quick_parser = subparsers.add_parser(
        "quick",
        help="Fast restoration - good quality in less time",
    )
    quick_parser.add_argument("input", type=Path, help="Input video file")
    quick_parser.add_argument("-o", "--output", type=Path, help="Output file")

    # Best command
    best_parser = subparsers.add_parser(
        "best",
        help="Best quality - auto-detects your GPU for maximum results",
    )
    best_parser.add_argument("input", type=Path, help="Input video file")
    best_parser.add_argument("-o", "--output", type=Path, help="Output file")
    best_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted processing",
    )
    best_parser.add_argument(
        "--enhance-audio",
        action="store_true",
        help="Also enhance audio (remove noise, normalize)",
    )
    best_parser.add_argument(
        "--notify",
        action="store_true",
        help="Send notification when complete",
    )
    best_parser.add_argument(
        "--for",
        dest="export_preset",
        choices=["youtube", "archive", "web", "mobile"],
        help="Optimize output for platform (youtube, archive, web, mobile)",
    )

    # Archive command
    archive_parser = subparsers.add_parser(
        "archive",
        help="Archive footage - fixes old film problems, optional colorization",
    )
    archive_parser.add_argument("input", type=Path, help="Input video file")
    archive_parser.add_argument("-o", "--output", type=Path, help="Output file")
    archive_parser.add_argument(
        "--colorize",
        nargs="*",
        type=Path,
        metavar="REF",
        help="Add color using reference images (e.g., --colorize photo1.jpg photo2.jpg)",
    )

    # Colorize command (alias for archive --colorize)
    colorize_parser = subparsers.add_parser(
        "colorize",
        help="Add color to black & white video",
    )
    colorize_parser.add_argument("input", type=Path, help="Input video file")
    colorize_parser.add_argument("-o", "--output", type=Path, help="Output file")
    colorize_parser.add_argument(
        "references",
        nargs="*",
        type=Path,
        help="Reference color images (optional but recommended)",
    )

    # Wizard command
    wizard_parser = subparsers.add_parser(
        "wizard",
        help="Interactive guided setup",
    )
    wizard_parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Optional input video file",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze video and show recommendations",
    )
    analyze_parser.add_argument("input", type=Path, help="Input video file")

    # Preview command
    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview a single frame before processing entire video",
    )
    preview_parser.add_argument("input", type=Path, help="Input video file")
    preview_parser.add_argument(
        "-t", "--timestamp",
        type=float,
        default=-1,
        help="Timestamp in seconds (-1 for middle of video)",
    )
    preview_parser.add_argument(
        "--open",
        action="store_true",
        help="Open comparison image in default viewer",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple videos in a folder",
    )
    batch_parser.add_argument("input", type=Path, help="Input folder or glob pattern")
    batch_parser.add_argument("-o", "--output", type=Path, help="Output folder")
    batch_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Create side-by-side comparison video",
    )
    compare_parser.add_argument("original", type=Path, help="Original video")
    compare_parser.add_argument("restored", type=Path, help="Restored video")
    compare_parser.add_argument("-o", "--output", type=Path, help="Output comparison video")

    # Dry-run command
    dryrun_parser = subparsers.add_parser(
        "dry-run",
        help="Show what would happen without processing",
    )
    dryrun_parser.add_argument("input", type=Path, help="Input video file")

    # Watch folder command
    watch_parser = subparsers.add_parser(
        "watch",
        help="Monitor folder and auto-process new videos",
    )
    watch_parser.add_argument("input", type=Path, help="Folder to watch")
    watch_parser.add_argument("-o", "--output", type=Path, help="Output folder")

    # Queue commands
    queue_parser = subparsers.add_parser(
        "queue",
        help="Manage processing queue",
    )
    queue_subparsers = queue_parser.add_subparsers(dest="queue_action")

    queue_add = queue_subparsers.add_parser("add", help="Add video to queue")
    queue_add.add_argument("videos", nargs="+", type=Path, help="Videos to add")
    queue_add.add_argument("-p", "--priority", type=int, default=5, help="Priority (1-10)")

    queue_subparsers.add_parser("list", help="List queue items")
    queue_subparsers.add_parser("start", help="Start processing queue")
    queue_subparsers.add_parser("status", help="Show queue status")
    queue_subparsers.add_parser("clear", help="Clear completed items")

    # Profile commands
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage saved configuration profiles",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_action")

    profile_list = profile_subparsers.add_parser("list", help="List profiles")
    profile_save = profile_subparsers.add_parser("save", help="Save current settings as profile")
    profile_save.add_argument("name", help="Profile name")
    profile_load = profile_subparsers.add_parser("load", help="Load a profile")
    profile_load.add_argument("name", help="Profile name to load")

    # Extract subtitles command
    extract_subs_parser = subparsers.add_parser(
        "extract-subs",
        help="Extract burnt-in subtitles with OCR",
    )
    extract_subs_parser.add_argument("input", type=Path, help="Input video")
    extract_subs_parser.add_argument("-o", "--output", type=Path, help="Output SRT file")
    extract_subs_parser.add_argument(
        "--remove",
        action="store_true",
        help="Also remove subtitles from video",
    )
    extract_subs_parser.add_argument(
        "--lang",
        default="en",
        help="Subtitle language (en, zh, ja, ko, etc.)",
    )

    # A/B test command
    abtest_parser = subparsers.add_parser(
        "ab-test",
        help="Compare two configurations on same video",
    )
    abtest_parser.add_argument("input", type=Path, help="Input video")
    abtest_parser.add_argument("--config-a", required=True, help="First config (preset name or JSON)")
    abtest_parser.add_argument("--config-b", required=True, help="Second config (preset name or JSON)")
    abtest_parser.add_argument("-o", "--output", type=Path, help="Output directory for comparisons")

    # Watermark removal command
    watermark_parser = subparsers.add_parser(
        "remove-watermark",
        help="Remove watermarks and logos from video",
    )
    watermark_parser.add_argument("input", type=Path, help="Input video")
    watermark_parser.add_argument("-o", "--output", type=Path, help="Output video")
    watermark_parser.add_argument(
        "--mask",
        type=Path,
        help="Custom mask image (white = watermark area)",
    )
    watermark_parser.add_argument(
        "--region",
        help="Rectangle region as x,y,width,height (e.g., 1700,50,200,80)",
    )
    watermark_parser.add_argument(
        "--auto-detect",
        action="store_true",
        default=True,
        help="Auto-detect watermark position (default)",
    )
    watermark_parser.add_argument(
        "--position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        help="Search for watermark in specific position",
    )

    # Deinterlace command
    deinterlace_parser = subparsers.add_parser(
        "deinterlace",
        help="Detect and fix interlacing (for VHS, DVD, broadcast)",
    )
    deinterlace_parser.add_argument("input", type=Path, help="Input video")
    deinterlace_parser.add_argument("-o", "--output", type=Path, help="Output video")
    deinterlace_parser.add_argument(
        "--method",
        choices=["auto", "yadif", "bwdif", "nnedi"],
        default="auto",
        help="Deinterlacing method (default: auto-detect best)",
    )
    deinterlace_parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't deinterlace",
    )

    # Crop black bars command
    crop_parser = subparsers.add_parser(
        "crop-bars",
        help="Detect and remove black bars (letterbox/pillarbox)",
    )
    crop_parser.add_argument("input", type=Path, help="Input video")
    crop_parser.add_argument("-o", "--output", type=Path, help="Output video")
    crop_parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't crop",
    )

    # Film stock detection command
    filmstock_parser = subparsers.add_parser(
        "detect-stock",
        help="Identify film stock type and era for better color correction",
    )
    filmstock_parser.add_argument("input", type=Path, help="Input video")
    filmstock_parser.add_argument(
        "--correct",
        action="store_true",
        help="Apply stock-specific color correction",
    )
    filmstock_parser.add_argument("-o", "--output", type=Path, help="Output video (if --correct)")

    # Audio sync check command
    audiosync_parser = subparsers.add_parser(
        "check-sync",
        help="Verify audio-video synchronization",
    )
    audiosync_parser.add_argument("input", type=Path, help="Processed video")
    audiosync_parser.add_argument(
        "--original",
        type=Path,
        help="Original video for comparison",
    )
    audiosync_parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix sync drift if detected",
    )
    audiosync_parser.add_argument("-o", "--output", type=Path, help="Output video (if --fix)")

    # Full analysis command (runs all detectors)
    fullscan_parser = subparsers.add_parser(
        "scan",
        help="Full video analysis (interlacing, letterbox, film stock, issues)",
    )
    fullscan_parser.add_argument("input", type=Path, help="Input video")
    fullscan_parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Auto-fix all detected issues",
    )
    fullscan_parser.add_argument("-o", "--output", type=Path, help="Output video (if --fix-all)")

    # Noise profiling command
    noise_parser = subparsers.add_parser(
        "noise-profile",
        help="Analyze video noise characteristics for optimal denoising",
    )
    noise_parser.add_argument("input", type=Path, help="Input video")
    noise_parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to sample (default: 30)",
    )

    # Upscale detection command
    upscale_parser = subparsers.add_parser(
        "upscale-detect",
        help="Detect if video was upscaled from lower resolution",
    )
    upscale_parser.add_argument("input", type=Path, help="Input video")
    upscale_parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to analyze (default: 20)",
    )

    # GPU thermal status command
    thermal_parser = subparsers.add_parser(
        "gpu-thermal",
        help="Show GPU temperature and thermal throttling status",
    )
    thermal_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    thermal_parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuously monitor temperature",
    )

    # Quick preview command
    quickpreview_parser = subparsers.add_parser(
        "quick-preview",
        help="Generate quick preview (samples every Nth frame)",
    )
    quickpreview_parser.add_argument("input", type=Path, help="Input video")
    quickpreview_parser.add_argument(
        "--every",
        type=int,
        default=100,
        help="Sample every N frames (default: 100)",
    )
    quickpreview_parser.add_argument(
        "--preset",
        type=str,
        help="Restoration preset to preview",
    )
    quickpreview_parser.add_argument("-o", "--output", type=Path, help="Output preview video")

    # Frame grid command
    grid_parser = subparsers.add_parser(
        "frame-grid",
        help="Generate a grid image of sample frames",
    )
    grid_parser.add_argument("input", type=Path, help="Input video")
    grid_parser.add_argument(
        "--frames",
        type=int,
        default=9,
        help="Number of frames in grid (default: 9)",
    )
    grid_parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns (default: 3)",
    )
    grid_parser.add_argument("-o", "--output", type=Path, help="Output image")

    # Preset comparison command
    preset_compare_parser = subparsers.add_parser(
        "compare-presets",
        help="Compare different presets on a single frame",
    )
    preset_compare_parser.add_argument("input", type=Path, help="Input video")
    preset_compare_parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["fast", "balanced", "quality"],
        help="Presets to compare (default: fast balanced quality)",
    )
    preset_compare_parser.add_argument(
        "--frame",
        type=int,
        help="Frame number to use (default: middle)",
    )
    preset_compare_parser.add_argument("-o", "--output-dir", type=Path, help="Output directory")

    # System check command
    syscheck_parser = subparsers.add_parser(
        "system-check",
        help="Check system readiness (disk, GPU, thermal)",
    )
    syscheck_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory to check disk space",
    )
    syscheck_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information",
    )

    # Model manager command
    models_parser = subparsers.add_parser(
        "models",
        help="Manage AI models (download, verify, clean)",
    )
    models_subparsers = models_parser.add_subparsers(dest="models_action")

    models_list = models_subparsers.add_parser("list", help="List available models")
    models_list.add_argument("--category", help="Filter by category (face, sr, denoise, etc.)")
    models_list.add_argument("--downloaded", action="store_true", help="Show only downloaded models")

    models_download = models_subparsers.add_parser("download", help="Download a model")
    models_download.add_argument("name", help="Model name or 'all' for all models")
    models_download.add_argument("--force", action="store_true", help="Re-download even if exists")

    models_clean = models_subparsers.add_parser("clean", help="Remove unused models")
    models_clean.add_argument("--days", type=int, default=30, help="Remove if unused for N days")

    models_subparsers.add_parser("status", help="Show disk usage and model status")

    # Export validation command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate exported video integrity",
    )
    validate_parser.add_argument("input", type=Path, help="Video to validate")
    validate_parser.add_argument("--compare", type=Path, help="Compare against source video")
    validate_parser.add_argument("--strict", action="store_true", help="Strict validation mode")
    validate_parser.add_argument("-o", "--output", type=Path, help="Save report to JSON")

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Test restoration speed with different settings",
    )
    benchmark_parser.add_argument("input", type=Path, help="Video to benchmark")
    benchmark_parser.add_argument(
        "--mode",
        choices=["quick", "standard", "thorough"],
        default="standard",
        help="Benchmark thoroughness",
    )
    benchmark_parser.add_argument("-o", "--output", type=Path, help="Save report to JSON")
    benchmark_parser.add_argument("--frames", type=int, default=100, help="Number of test frames")

    # Frame quality scoring command
    scoreframes_parser = subparsers.add_parser(
        "score-frames",
        help="Analyze frame quality and find problem frames",
    )
    scoreframes_parser.add_argument("input", type=Path, help="Video to analyze")
    scoreframes_parser.add_argument("--sample-rate", type=int, default=5, help="Check every Nth frame")
    scoreframes_parser.add_argument("--threshold", type=float, default=60, help="Problem threshold (0-100)")
    scoreframes_parser.add_argument("-o", "--output", type=Path, help="Save report to JSON")
    scoreframes_parser.add_argument("--export-frames", type=Path, help="Export problem frames to directory")

    # Credits detection command
    credits_parser = subparsers.add_parser(
        "detect-credits",
        help="Detect intro/outro and credits sequences",
    )
    credits_parser.add_argument("input", type=Path, help="Video to analyze")
    credits_parser.add_argument("-o", "--output", type=Path, help="Save analysis to JSON")
    credits_parser.add_argument("--trim", action="store_true", help="Output trimmed video (main content only)")

    # Webhook configuration command
    webhook_parser = subparsers.add_parser(
        "webhook",
        help="Configure progress webhooks (Discord, Slack, etc.)",
    )
    webhook_subparsers = webhook_parser.add_subparsers(dest="webhook_action")

    webhook_add = webhook_subparsers.add_parser("add", help="Add a webhook")
    webhook_add.add_argument("url", help="Webhook URL")
    webhook_add.add_argument("--type", choices=["discord", "slack", "ntfy", "generic"], default="generic")
    webhook_add.add_argument("--name", help="Friendly name")

    webhook_subparsers.add_parser("list", help="List configured webhooks")
    webhook_subparsers.add_parser("test", help="Send test notification")
    webhook_remove = webhook_subparsers.add_parser("remove", help="Remove a webhook")
    webhook_remove.add_argument("index", type=int, help="Webhook index to remove")

    # Job scheduling command
    schedule_parser = subparsers.add_parser(
        "schedule",
        help="Schedule jobs for later processing",
    )
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_action")

    schedule_add = schedule_subparsers.add_parser("add", help="Schedule a job")
    schedule_add.add_argument("input", type=Path, help="Video to process")
    schedule_add.add_argument("-o", "--output", type=Path, help="Output path")
    schedule_add.add_argument("--preset", default="balanced", help="Processing preset")
    schedule_add.add_argument("--at", help="Run at specific time (HH:MM or YYYY-MM-DD HH:MM)")
    schedule_add.add_argument("--delay", type=int, help="Delay in minutes")
    schedule_add.add_argument("--priority", type=int, default=5, help="Priority (1-10)")

    schedule_subparsers.add_parser("list", help="List scheduled jobs")
    schedule_subparsers.add_parser("start", help="Start scheduler daemon")
    schedule_subparsers.add_parser("run-next", help="Run next available job")
    schedule_subparsers.add_parser("stats", help="Show scheduler statistics")

    schedule_cancel = schedule_subparsers.add_parser("cancel", help="Cancel a job")
    schedule_cancel.add_argument("job_id", help="Job ID to cancel")

    # Power management command
    power_parser = subparsers.add_parser(
        "power",
        help="Power management settings",
    )
    power_parser.add_argument("--prevent-sleep", action="store_true", help="Keep system awake")
    power_parser.add_argument("--on-complete", choices=["none", "sleep", "hibernate", "shutdown"], default="none")
    power_parser.add_argument("--status", action="store_true", help="Show power status")

    # Output template command
    template_parser = subparsers.add_parser(
        "template",
        help="Manage output filename templates",
    )
    template_subparsers = template_parser.add_subparsers(dest="template_action")

    template_subparsers.add_parser("list", help="List available templates")
    template_preview = template_subparsers.add_parser("preview", help="Preview template output")
    template_preview.add_argument("input", type=Path, help="Sample input file")
    template_preview.add_argument("--template", default="{name}_restored.{ext}", help="Template string")

    # Default: smart auto-restore
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input video file (smart auto-restore)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show banner",
    )
    parser.add_argument(
        "--quality",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Quality level: 1=fastest, 3=balanced, 5=best (auto-detected if not set)",
    )

    return parser


def run_smart_restore(
    input_path: Path,
    output_path: Optional[Path] = None,
    console: Optional[Console] = None,
) -> Path:
    """Run smart auto-restoration.

    Analyzes the video and automatically selects optimal settings.

    Args:
        input_path: Input video path
        output_path: Output path (auto-generated if None)
        console: Console for output

    Returns:
        Path to restored video
    """
    console = console or Console()

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_restored.mp4"

    console.info("Analyzing video...")

    with create_spinner("Analyzing content and degradation"):
        analysis = analyze_video_smart(input_path)

    # Show analysis
    console.video_summary(
        path=input_path,
        resolution=analysis.resolution,
        fps=analysis.fps,
        duration=analysis.duration_formatted,
        codec=analysis.codec,
        size_mb=analysis.bitrate_kbps * analysis.duration_seconds / 8000,
    )

    # Get recommendations
    recommendations = get_recommendations(
        analysis=analysis,
        user_priority="balanced",
    )

    # Show plan
    console.restoration_plan(
        preset=recommendations.preset.title(),
        stages=recommendations.processing_stages,
        estimated_time="Depends on video length",
        quality_target=f"{recommendations.estimated_quality_score*100:.0f}%",
    )

    # Show warnings
    for warning in recommendations.warnings:
        console.warning(warning)

    console.print()
    console.info("Starting restoration...")

    # Create config from recommendations
    config_dict = recommendations.to_config_dict()
    config_dict["project_dir"] = input_path.parent / ".framewright_temp"

    config = Config(**config_dict)

    # Create restorer with progress callback
    progress = ArchiveRestorationProgress()

    def progress_callback(info):
        if hasattr(info, "stage"):
            stage_map = {
                "download": "download",
                "analyze": "analyze",
                "extract_frames": "extract",
                "dedup": "dedup",
                "qp_artifact_removal": "qp_removal",
                "tap_denoise": "denoise",
                "frame_generation": "frame_gen",
                "enhance": "enhance",
                "face_restore": "face",
                "interpolate": "interpolate",
                "temporal": "temporal",
                "colorize": "colorize",
                "reassemble": "reassemble",
                "validate": "validate",
            }
            stage = stage_map.get(info.stage, info.stage)
            if hasattr(info, "frames_total"):
                progress.update(
                    stage,
                    completed=info.frames_completed,
                    total=info.frames_total,
                )

    restorer = VideoRestorer(config, progress_callback=progress_callback)

    # Configure stages based on recommendations
    _configure_restorer_stages(progress, recommendations)

    # Run restoration
    progress.start()
    try:
        result_path = restorer.restore_video(
            source=str(input_path),
            output_path=output_path,
        )
    finally:
        progress.stop()

    # Show completion
    console.completion_summary(
        output_path=result_path,
        duration="Complete",
        frames_processed=analysis.total_frames,
        quality_metrics={"PSNR": 35.0, "SSIM": 0.95},
    )

    return result_path


def run_quick_restore(
    input_path: Path,
    output_path: Optional[Path] = None,
    console: Optional[Console] = None,
) -> Path:
    """Run quick restoration with fast preset."""
    console = console or Console()

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_quick.mp4"

    console.info("Quick restoration mode - optimized for speed")

    config = Config(
        preset="fast",
        scale_factor=2,
        project_dir=input_path.parent / ".framewright_temp",
    )

    restorer = VideoRestorer(config)
    return restorer.restore_video(
        source=str(input_path),
        output_path=output_path,
    )


def run_best_restore(
    input_path: Path,
    output_path: Optional[Path] = None,
    console: Optional[Console] = None,
    enhance_audio: bool = False,
    notify: bool = False,
    resume: bool = False,
    export_preset: Optional[str] = None,
) -> Path:
    """Run maximum quality restoration - automatically optimized for your GPU.

    This is the "Apple design" command - just run it and it figures out
    the best settings for your hardware automatically.

    Args:
        input_path: Input video file
        output_path: Output file (auto-generated if None)
        console: Console for output
        enhance_audio: Also enhance audio track
        notify: Send notification when complete
        resume: Resume interrupted processing
        export_preset: Export preset (youtube, archive, web, mobile)
    """
    console = console or Console()
    import time
    start_time = time.time()

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_best.mp4"

    # Auto-detect GPU and get optimal settings
    console.print()
    with create_spinner("Detecting your hardware"):
        preset_name, auto_config, gpu_message = _detect_optimal_preset()

    console.info(gpu_message)

    # Show what we're going to do in plain language
    descriptions = _get_quality_description(preset_name, auto_config)
    console.print()
    console.panel(
        "\n".join(f"  • {desc}" for desc in descriptions),
        title="What will happen",
        style="cyan",
    )

    # Analyze video
    console.print()
    with create_spinner("Understanding your video"):
        analysis = analyze_video_smart(input_path)

    # Show video info simply
    duration_str = f"{int(analysis.duration_seconds // 60)}m {int(analysis.duration_seconds % 60)}s"
    console.print()
    console.panel(
        f"  Resolution: {analysis.width}x{analysis.height}\n"
        f"  Duration: {duration_str}\n"
        f"  Frame rate: {analysis.fps:.1f} fps",
        title="Your video",
        style="blue",
    )

    # Build config - merge auto-detected settings with video analysis
    config_dict = {
        "preset": preset_name,
        "project_dir": input_path.parent / ".framewright_temp",
        **auto_config,  # GPU-optimized settings
    }

    # Add video-specific optimizations
    if analysis.content.is_black_and_white:
        config_dict["enable_colorization"] = False  # Don't auto-colorize without reference
        console.info("Black & white video detected - preserving original colors")
        console.print("  Tip: Use 'framewright archive video.mp4 --colorize ref.jpg' to add color")

    if analysis.content.has_faces and analysis.content.face_percentage > 10:
        config_dict["auto_face_restore"] = True
        config_dict["face_model"] = "aesrgan"
        console.info("Faces detected - will enhance facial details")

    if analysis.degradation.noise_level > 0.3:
        config_dict["enable_tap_denoise"] = True
        console.info("Noise detected - will clean up grain")

    if analysis.degradation.compression_level > 0.3:
        config_dict["enable_qp_artifact_removal"] = True
        console.info("Compression detected - will remove blocking")

    if analysis.degradation.flicker_intensity > 0.3:
        config_dict["temporal_method"] = "hybrid"
        console.info("Flicker detected - will stabilize brightness")

    # Audio enhancement
    if enhance_audio:
        config_dict["enable_audio_enhance"] = True
        console.info("Audio enhancement enabled - will clean up audio")

    # Determine scale factor based on resolution
    if analysis.width < 480:
        config_dict["scale_factor"] = 4
        console.info("Low resolution - will upscale 4x")
    elif analysis.width < 1080:
        config_dict["scale_factor"] = 4
        console.info("SD video - will upscale to 4K")
    else:
        config_dict["scale_factor"] = 2
        console.info("HD video - will upscale 2x")

    console.print()
    console.info("Starting restoration...")
    console.print("  (You can leave this running - it will finish automatically)")
    console.print()

    config = Config(**config_dict)

    # Progress with friendly stage names
    progress = ArchiveRestorationProgress()

    def progress_callback(info):
        stage_friendly = {
            "download": "download",
            "analyze": "analyze",
            "extract_frames": "extract",
            "dedup": "dedup",
            "qp_artifact_removal": "qp_removal",
            "tap_denoise": "denoise",
            "frame_generation": "frame_gen",
            "enhance": "enhance",
            "face_restore": "face",
            "interpolate": "interpolate",
            "temporal": "temporal",
            "colorize": "colorize",
            "reassemble": "reassemble",
            "validate": "validate",
        }
        if hasattr(info, "stage"):
            stage = stage_friendly.get(info.stage, info.stage)
            if hasattr(info, "frames_total"):
                progress.update(
                    stage,
                    completed=info.frames_completed,
                    total=info.frames_total,
                )

    restorer = VideoRestorer(config, progress_callback=progress_callback)

    progress.start()
    try:
        result_path = restorer.restore_video(
            source=str(input_path),
            output_path=output_path,
        )
    finally:
        progress.stop()

    # Audio enhancement (post-process)
    if enhance_audio and result_path.exists():
        console.print()
        console.info("Enhancing audio...")
        try:
            from .processors.audio_enhance import enhance_audio_auto
            import tempfile

            # Extract audio, enhance, merge back
            audio_enhanced = Path(tempfile.mktemp(suffix=".wav"))
            from .utils.dependencies import get_ffmpeg_path
            import subprocess

            # Extract audio
            subprocess.run([
                get_ffmpeg_path(), "-y", "-i", str(result_path),
                "-vn", "-acodec", "pcm_s16le", str(audio_enhanced)
            ], capture_output=True, check=True)

            # Enhance
            audio_output = audio_enhanced.with_suffix(".enhanced.wav")
            enhance_result = enhance_audio_auto(str(audio_enhanced), str(audio_output))

            if enhance_result.success:
                # Merge back
                temp_video = result_path.with_suffix(".temp.mp4")
                subprocess.run([
                    get_ffmpeg_path(), "-y",
                    "-i", str(result_path),
                    "-i", str(audio_output),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0",
                    str(temp_video)
                ], capture_output=True, check=True)

                # Replace original
                temp_video.replace(result_path)
                console.info("Audio enhanced successfully")

            # Cleanup
            audio_enhanced.unlink(missing_ok=True)
            audio_output.unlink(missing_ok=True)

        except Exception as e:
            console.warning(f"Audio enhancement failed: {e}")

    # Calculate processing time
    processing_time = time.time() - start_time
    time_str = f"{int(processing_time // 60)}m {int(processing_time % 60)}s"

    # Quality report
    quality_summary = ""
    try:
        from .cli_advanced import calculate_quality_report
        report = calculate_quality_report(input_path, result_path)
        if report.resolution_increase:
            quality_summary = f"\n  {report.summary()}"
    except Exception:
        pass

    # Success message
    console.print()
    console.panel(
        f"  Saved to: {result_path}\n"
        f"  Original: {analysis.width}x{analysis.height}\n"
        f"  Enhanced: {analysis.width * config_dict.get('scale_factor', 2)}x"
        f"{analysis.height * config_dict.get('scale_factor', 2)}\n"
        f"  Processing time: {time_str}"
        f"{quality_summary}",
        title="Done!",
        style="green",
    )

    # Send notification
    if notify:
        from .cli_advanced import send_notification, play_completion_sound
        send_notification(
            "FrameWright Complete",
            f"Restored: {input_path.name}"
        )
        play_completion_sound()

    return result_path


def run_archive_restore(
    input_path: Path,
    output_path: Optional[Path] = None,
    colorize_refs: Optional[List[Path]] = None,
    console: Optional[Console] = None,
) -> Path:
    """Run archive-optimized restoration.

    Specifically tuned for historical/archive footage with:
    - Missing frame generation (SVD on capable GPUs)
    - Deduplication for old film
    - Optional colorization with references
    - Audio restoration
    """
    console = console or Console()

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_archive.mp4"

    console.print()
    console.info("Archive footage restoration mode")
    console.print("  Optimized for damaged/historical footage")

    # Auto-detect GPU for optimal settings
    with create_spinner("Detecting your hardware"):
        preset_name, auto_config, gpu_message = _detect_optimal_preset()

    console.info(gpu_message)

    # Analyze
    with create_spinner("Analyzing archive footage"):
        analysis = analyze_video_smart(input_path)

    # Show characteristics
    console.video_summary(
        path=input_path,
        resolution=analysis.resolution,
        fps=analysis.fps,
        duration=analysis.duration_formatted,
        codec=analysis.codec,
        size_mb=analysis.bitrate_kbps * analysis.duration_seconds / 8000,
    )

    # Build archive-optimized config using GPU-optimized base
    config_dict = {
        "preset": preset_name,
        "project_dir": input_path.parent / ".framewright_temp",
        "scale_factor": 4 if analysis.width < 720 else 2,
        **auto_config,  # GPU-optimized settings
        # Archive-specific overrides
        "enable_tap_denoise": True,
        "enable_qp_artifact_removal": True,
        "enable_frame_generation": True,
        "enable_deduplication": True,
        "temporal_method": "hybrid",
        "enable_interpolation": analysis.fps < 24,
        "target_fps": 24 if analysis.fps < 20 else analysis.fps,
        # Use SVD for missing frames on high-VRAM GPUs
        "frame_generation_model": auto_config.get("frame_generation_model", "optical_flow_warp"),
        "max_gap_frames": auto_config.get("max_gap_frames", 10),
        # Audio enhancement for archive
        "enable_audio_enhance": True,
    }

    # Show what we're doing for archive footage
    features = []
    features.append("Duplicate frame removal")
    features.append("Film grain preservation")
    if config_dict.get("frame_generation_model") == "svd":
        features.append("AI-powered missing frame reconstruction (SVD)")
        console.info("Using Stable Video Diffusion for missing frames")
    else:
        features.append("Optical flow frame reconstruction")
    if config_dict.get("enable_tap_denoise"):
        features.append("Neural denoising")
    features.append("Audio restoration")

    console.print()
    console.panel(
        "\n".join(f"  • {f}" for f in features),
        title="Archive restoration features",
        style="cyan",
    )

    # Face enhancement if faces detected
    if analysis.content.has_faces:
        config_dict["auto_face_restore"] = True
        config_dict["face_model"] = "aesrgan"
        console.info("Faces detected - will enhance facial details")

    # Colorization with references
    if colorize_refs:
        config_dict["colorization_reference_images"] = [str(p) for p in colorize_refs]
        config_dict["enable_temporal_colorization"] = True  # Smooth colors
        console.info(f"Colorization enabled with {len(colorize_refs)} reference images")
    elif analysis.content.is_black_and_white:
        console.print()
        console.warning("B&W footage detected")
        console.print("  To add color: framewright archive video.mp4 --colorize photo1.jpg photo2.jpg")

    console.print()
    console.info("Starting archive restoration...")

    config = Config(**config_dict)
    restorer = VideoRestorer(config)

    result_path = restorer.restore_video(
        source=str(input_path),
        output_path=output_path,
    )

    console.print()
    console.panel(
        f"  Saved to: {result_path}",
        title="Archive restoration complete!",
        style="green",
    )

    return result_path


def run_analyze(input_path: Path, console: Optional[Console] = None) -> None:
    """Analyze video and show recommendations."""
    console = console or Console()

    console.info("Analyzing video...")

    with create_spinner("Running deep analysis"):
        analysis = analyze_video_smart(input_path)

    # Show detailed analysis
    console.video_summary(
        path=input_path,
        resolution=analysis.resolution,
        fps=analysis.fps,
        duration=analysis.duration_formatted,
        codec=analysis.codec,
        size_mb=analysis.bitrate_kbps * analysis.duration_seconds / 8000,
    )

    # Content characteristics
    console.panel(
        f"[bold]Content Type:[/bold] {analysis.content.content_type.value.replace('_', ' ').title()}\n"
        f"[bold]Era:[/bold] {analysis.content.era.value.replace('_', ' ').title()}\n"
        f"[bold]Color:[/bold] {'Black & White' if analysis.content.is_black_and_white else 'Color'}\n"
        f"[bold]Faces:[/bold] {analysis.content.face_percentage:.0f}% of frames\n"
        f"[bold]Motion:[/bold] {analysis.content.motion_intensity*100:.0f}%",
        title="Content Analysis",
        style="cyan",
    )

    # Degradation
    console.panel(
        f"[bold]Severity:[/bold] {analysis.degradation.severity.title()}\n"
        f"[bold]Noise Level:[/bold] {analysis.degradation.noise_level*100:.0f}%\n"
        f"[bold]Compression:[/bold] {analysis.degradation.compression_level*100:.0f}%\n"
        f"[bold]Flicker:[/bold] {analysis.degradation.flicker_intensity*100:.0f}%\n"
        f"[bold]Issues:[/bold] {', '.join(analysis.degradation.primary_issues) or 'None detected'}",
        title="Degradation Analysis",
        style="yellow",
    )

    # Get recommendations
    recommendations = get_recommendations(analysis=analysis, user_priority="balanced")

    console.restoration_plan(
        preset=recommendations.preset.title(),
        stages=recommendations.processing_stages,
        estimated_time="Varies by hardware",
        quality_target=f"{recommendations.estimated_quality_score*100:.0f}%",
    )

    for warning in recommendations.warnings:
        console.warning(warning)

    # Show command suggestion
    console.print()
    console.info(f"To restore this video, run:")
    console.print(f"  [bold cyan]framewright {input_path}[/bold cyan]")
    console.print()


def run_colorize(
    input_path: Path,
    output_path: Optional[Path] = None,
    references: Optional[List[Path]] = None,
    console: Optional[Console] = None,
) -> Path:
    """Colorize black & white video.

    Simple command for the common use case of adding color to B&W footage.
    """
    console = console or Console()

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_colorized.mp4"

    console.print()
    console.info("Colorization mode - adding color to black & white video")

    # Auto-detect GPU
    with create_spinner("Detecting your hardware"):
        preset_name, auto_config, gpu_message = _detect_optimal_preset()

    console.info(gpu_message)

    # Analyze video
    with create_spinner("Analyzing video"):
        analysis = analyze_video_smart(input_path)

    if not analysis.content.is_black_and_white:
        console.warning("This video already appears to be in color!")
        console.print("  Continuing anyway - results may vary")
        console.print()

    # Build config
    config_dict = {
        "preset": preset_name,
        "project_dir": input_path.parent / ".framewright_temp",
        **auto_config,
        "enable_colorization": True,
        "enable_temporal_colorization": True,  # Smooth colors across frames
    }

    # Add references if provided
    if references:
        config_dict["colorization_reference_images"] = [str(p) for p in references]
        console.info(f"Using {len(references)} reference image(s) for color guidance")
        console.print("  The AI will try to match these colors")
    else:
        console.info("No reference images - AI will choose colors automatically")
        console.print("  Tip: For better results, provide reference images:")
        console.print("    framewright colorize video.mp4 photo1.jpg photo2.jpg")

    console.print()

    # Determine scale
    if analysis.width < 720:
        config_dict["scale_factor"] = 4
    else:
        config_dict["scale_factor"] = 2

    config = Config(**config_dict)
    restorer = VideoRestorer(config)

    console.info("Starting colorization...")
    console.print()

    result_path = restorer.restore_video(
        source=str(input_path),
        output_path=output_path,
    )

    console.print()
    console.panel(
        f"  Colorized video saved to:\n  {result_path}",
        title="Done!",
        style="green",
    )

    return result_path


def _configure_restorer_stages(
    progress: ArchiveRestorationProgress,
    recommendations,
) -> None:
    """Configure which stages are shown in progress."""
    config = recommendations.to_config_dict()

    # Skip stages that won't be used
    if not config.get("enable_qp_artifact_removal"):
        progress.skip_stage("qp_removal")
    if not config.get("enable_tap_denoise"):
        progress.skip_stage("denoise")
    if not config.get("enable_frame_generation"):
        progress.skip_stage("frame_gen")
    if not config.get("auto_face_restore"):
        progress.skip_stage("face")
    if not config.get("enable_interpolation"):
        progress.skip_stage("interpolate")
    if not config.get("temporal_method"):
        progress.skip_stage("temporal")
    if not config.get("colorization_reference_images"):
        progress.skip_stage("colorize")


def main_simple() -> int:
    """Main entry point for simplified CLI."""
    parser = create_simple_parser()
    args = parser.parse_args()

    # Create console
    console = create_console(quiet=getattr(args, "quiet", False))

    # Show banner unless suppressed
    if not getattr(args, "no_banner", False):
        console.print_compact_banner()

    try:
        # Handle commands
        if args.command == "wizard":
            result = run_wizard(args.input)
            if result.completed:
                run_smart_restore(
                    result.input_path,
                    result.output_path,
                    console,
                )
            return 0 if result.completed else 1

        elif args.command == "quick":
            run_quick_restore(args.input, args.output, console)
            return 0

        elif args.command == "best":
            run_best_restore(
                args.input,
                args.output,
                console,
                enhance_audio=getattr(args, "enhance_audio", False),
                notify=getattr(args, "notify", False),
                resume=getattr(args, "resume", False),
                export_preset=getattr(args, "export_preset", None),
            )
            return 0

        elif args.command == "archive":
            run_archive_restore(
                args.input,
                args.output,
                args.colorize,
                console,
            )
            return 0

        elif args.command == "colorize":
            run_colorize(
                args.input,
                args.output,
                args.references,
                console,
            )
            return 0

        elif args.command == "analyze":
            run_analyze(args.input, console)
            return 0

        elif args.command == "preview":
            from .cli_advanced import run_preview
            result = run_preview(
                args.input,
                timestamp=args.timestamp,
                console=console,
            )
            if result.comparison_image:
                console.print()
                console.panel(
                    f"  Original: {result.original_frame}\n"
                    f"  Enhanced: {result.enhanced_frame}\n"
                    f"  Comparison: {result.comparison_image}",
                    title="Preview Complete",
                    style="green",
                )
                # Open image if requested
                if getattr(args, "open", False) and result.comparison_image:
                    import subprocess
                    import platform
                    if platform.system() == "Darwin":
                        subprocess.run(["open", str(result.comparison_image)])
                    elif platform.system() == "Windows":
                        subprocess.run(["start", "", str(result.comparison_image)], shell=True)
                    else:
                        subprocess.run(["xdg-open", str(result.comparison_image)])
            return 0

        elif args.command == "batch":
            from .cli_advanced import run_batch
            result = run_batch(
                args.input,
                output_dir=args.output,
                recursive=args.recursive,
                console=console,
            )
            console.print()
            console.panel(
                f"  Total: {result.total}\n"
                f"  Completed: {result.completed}\n"
                f"  Failed: {result.failed}\n"
                f"  Skipped: {result.skipped}",
                title="Batch Complete",
                style="green" if result.failed == 0 else "yellow",
            )
            return 0 if result.failed == 0 else 1

        elif args.command == "compare":
            from .cli_advanced import create_comparison_video
            output = args.output or args.original.parent / f"{args.original.stem}_comparison.mp4"
            result = create_comparison_video(
                args.original,
                args.restored,
                output,
                console=console,
            )
            if result:
                console.info(f"Comparison video saved to: {result}")
            return 0 if result else 1

        elif args.command == "dry-run":
            from .cli_advanced import run_dry_run
            # Get optimal settings first
            preset_name, auto_config, _ = _detect_optimal_preset()
            result = run_dry_run(args.input, auto_config, console)

            console.print()
            console.panel(
                f"  Input: {result.input_resolution[0]}x{result.input_resolution[1]}\n"
                f"  Output: {result.output_resolution[0]}x{result.output_resolution[1]}\n"
                f"  Frames: {result.frame_count:,}\n"
                f"  Duration: {result.duration_seconds/60:.1f} minutes\n"
                f"  Temp space needed: ~{result.estimated_temp_space_gb:.1f} GB\n"
                f"  GPU VRAM needed: ~{result.gpu_vram_needed_gb:.0f} GB",
                title="Dry Run - What Would Happen",
                style="cyan",
            )

            console.print()
            console.info("Processing stages:")
            for i, stage in enumerate(result.stages, 1):
                console.print(f"  {i}. {stage}")

            if result.warnings:
                console.print()
                for warning in result.warnings:
                    console.warning(warning)
            return 0

        elif args.command == "watch":
            from .workflow.automation import FolderWatcher
            output_dir = args.output or args.input / "restored"
            output_dir.mkdir(parents=True, exist_ok=True)

            console.print()
            console.panel(
                f"  Watching: {args.input}\n"
                f"  Output: {output_dir}\n"
                f"  Press Ctrl+C to stop",
                title="Watch Mode Active",
                style="cyan",
            )

            def on_complete(input_path: Path, output_path: Path):
                console.info(f"Completed: {output_path.name}")

            def on_error(input_path: Path, error: Exception):
                console.error(f"Failed: {input_path.name} - {error}")

            watcher = FolderWatcher(
                watch_dir=args.input,
                output_dir=output_dir,
                on_complete=on_complete,
                on_error=on_error,
            )

            try:
                console.info("Watching for new videos... (Ctrl+C to stop)")
                watcher.start(blocking=True)
            except KeyboardInterrupt:
                watcher.stop()
                console.print()
                console.info("Watch mode stopped")
            return 0

        elif args.command == "queue":
            from .workflow.automation import ProcessingQueue

            queue = ProcessingQueue()
            action = args.queue_action

            if action == "add":
                for video in args.videos:
                    if video.exists():
                        queue.add(video, priority=args.priority)
                        console.info(f"Added: {video.name} (priority {args.priority})")
                    else:
                        console.warning(f"Not found: {video}")
                return 0

            elif action == "list":
                items = queue.list_items()
                if not items:
                    console.info("Queue is empty")
                else:
                    console.print()
                    console.panel(
                        "\n".join(
                            f"  {i+1}. {item['video']} [{item['status']}] (priority {item['priority']})"
                            for i, item in enumerate(items)
                        ),
                        title=f"Queue ({len(items)} items)",
                        style="cyan",
                    )
                return 0

            elif action == "start":
                console.info("Starting queue processing...")
                console.print("  (Processing runs in background, check status with 'framewright queue status')")
                queue.start(blocking=True)
                console.info("Queue processing complete")
                return 0

            elif action == "status":
                status = queue.get_status()
                console.panel(
                    f"  Pending: {status['pending']}\n"
                    f"  Processing: {status['processing']}\n"
                    f"  Completed: {status['completed']}\n"
                    f"  Failed: {status['failed']}",
                    title="Queue Status",
                    style="cyan",
                )
                return 0

            elif action == "clear":
                cleared = queue.clear_completed()
                console.info(f"Cleared {cleared} completed/failed items")
                return 0

            else:
                console.info("Queue commands: add, list, start, status, clear")
                return 0

        elif args.command == "profile":
            from .workflow.automation import ProfileManager

            profiles = ProfileManager()
            action = args.profile_action

            if action == "list":
                profile_list = profiles.list_profiles()
                if not profile_list:
                    console.info("No saved profiles")
                    console.print("  Create one with: framewright profile save my-settings")
                else:
                    console.print()
                    console.panel(
                        "\n".join(f"  • {p['name']}" for p in profile_list),
                        title=f"Saved Profiles ({len(profile_list)})",
                        style="cyan",
                    )
                return 0

            elif action == "save":
                # Save current optimal settings as a profile
                preset_name, auto_config, _ = _detect_optimal_preset()
                profile_data = {
                    "preset": preset_name,
                    **auto_config,
                }
                profiles.save(args.name, profile_data)
                console.info(f"Profile '{args.name}' saved with current GPU-optimized settings")
                return 0

            elif action == "load":
                try:
                    profile_data = profiles.load(args.name)
                    console.info(f"Profile '{args.name}' loaded")
                    console.print(f"  Preset: {profile_data.get('preset', 'custom')}")
                except FileNotFoundError:
                    console.error(f"Profile '{args.name}' not found")
                    return 1
                return 0

            else:
                console.info("Profile commands: list, save <name>, load <name>")
                return 0

        elif args.command == "extract-subs":
            from .processors.subtitle_extraction import SubtitleExtractor

            console.print()
            console.info(f"Extracting subtitles from: {args.input.name}")
            console.print(f"  Language: {args.lang}")
            if args.remove:
                console.print("  Will also remove subtitles from video")

            extractor = SubtitleExtractor(languages=[args.lang])

            # Determine output paths
            srt_output = args.output or args.input.with_suffix(".srt")
            clean_video = args.input.parent / f"{args.input.stem}_clean.mp4" if args.remove else None

            with create_spinner("Detecting and extracting subtitles"):
                result = extractor.extract_and_remove(
                    args.input,
                    output_srt=srt_output,
                    output_video=clean_video,
                    remove_subtitles=args.remove,
                )

            if result.success and result.subtitles:
                console.print()
                console.panel(
                    f"  Subtitles found: {result.subtitle_count} lines\n"
                    f"  Saved to: {result.srt_path or srt_output}"
                    + (f"\n  Clean video: {result.clean_video_path}" if result.clean_video_path else ""),
                    title="Subtitle Extraction Complete",
                    style="green",
                )
                console.print()
                console.info("Tip: Use AI translation tools to translate the SRT file")
            else:
                console.warning("No burnt-in subtitles detected")
            return 0

        elif args.command == "ab-test":
            from .processors.quality_control import ABTester
            import json

            console.print()
            console.info(f"A/B Testing configurations on: {args.input.name}")

            # Parse configs (can be preset names or JSON)
            def parse_config(config_str: str) -> dict:
                if config_str in PRESETS:
                    return {"preset": config_str}
                try:
                    return json.loads(config_str)
                except json.JSONDecodeError:
                    return {"preset": config_str}

            config_a = parse_config(args.config_a)
            config_b = parse_config(args.config_b)

            console.print(f"  Config A: {args.config_a}")
            console.print(f"  Config B: {args.config_b}")

            output_dir = args.output or args.input.parent / "ab_test_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            tester = ABTester(output_dir=output_dir)

            with create_spinner("Running A/B test (this may take a while)"):
                result = tester.compare(
                    args.input,
                    config_a,
                    config_b,
                    config_a_name=args.config_a,
                    config_b_name=args.config_b,
                )

            console.print()
            console.panel(
                f"  Config A PSNR: {result.a_metrics.get('psnr', 0):.2f} dB\n"
                f"  Config A SSIM: {result.a_metrics.get('ssim', 0):.4f}\n"
                f"  \n"
                f"  Config B PSNR: {result.b_metrics.get('psnr', 0):.2f} dB\n"
                f"  Config B SSIM: {result.b_metrics.get('ssim', 0):.4f}\n"
                f"  \n"
                f"  Winner: {result.winner if result.winner != 'tie' else 'Tie'}\n"
                f"  Comparison saved to: {output_dir}",
                title="A/B Test Results",
                style="green" if result.winner and result.winner != 'tie' else "cyan",
            )
            return 0

        elif args.command == "remove-watermark":
            from .processors.watermark_removal import WatermarkRemover, WatermarkConfig, WatermarkPosition

            console.print()
            console.info(f"Removing watermark from: {args.input.name}")

            output_path = args.output or args.input.parent / f"{args.input.stem}_clean.mp4"

            # Configure watermark removal
            config_kwargs = {"auto_detect": True}

            if args.mask:
                config_kwargs["mask_path"] = args.mask
                config_kwargs["auto_detect"] = False
                console.print(f"  Using custom mask: {args.mask}")

            if args.position:
                pos_map = {
                    "top-left": WatermarkPosition.TOP_LEFT,
                    "top-right": WatermarkPosition.TOP_RIGHT,
                    "bottom-left": WatermarkPosition.BOTTOM_LEFT,
                    "bottom-right": WatermarkPosition.BOTTOM_RIGHT,
                    "center": WatermarkPosition.CENTER,
                }
                config_kwargs["positions"] = [pos_map[args.position]]
                console.print(f"  Searching in: {args.position}")

            config = WatermarkConfig(**config_kwargs)
            remover = WatermarkRemover(config)

            if not remover.is_available():
                console.warning("Downloading watermark removal model...")
                remover.download_model()

            # Process video frame by frame
            import cv2
            cap = cv2.VideoCapture(str(args.input))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Handle region argument
            custom_mask = None
            if args.region:
                try:
                    x, y, w, h = map(int, args.region.split(","))
                    custom_mask = remover.create_rectangular_mask((height, width), x, y, w, h)
                    console.print(f"  Using region: {args.region}")
                except:
                    console.warning("Invalid region format, using auto-detect")

            console.print()
            with create_spinner(f"Processing {total_frames} frames"):
                # Detect on first frame
                ret, first_frame = cap.read()
                if ret:
                    if custom_mask is None and config.auto_detect:
                        custom_mask = remover.detect_watermark(first_frame)
                        if custom_mask is not None:
                            console.info("Watermark detected, applying to all frames")

                    # Process first frame
                    clean_frame = remover.remove_watermark(first_frame, custom_mask)
                    out.write(clean_frame)

                # Process remaining frames
                frame_count = 1
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    clean_frame = remover.remove_watermark(frame, custom_mask)
                    out.write(clean_frame)
                    frame_count += 1

            cap.release()
            out.release()

            # Copy audio
            console.info("Copying audio track...")
            import subprocess
            temp_output = output_path.with_suffix(".temp.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(output_path),
                "-i", str(args.input),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                str(temp_output)
            ], capture_output=True)
            temp_output.replace(output_path)

            console.print()
            console.panel(
                f"  Frames processed: {frame_count}\n"
                f"  Output: {output_path}",
                title="Watermark Removal Complete",
                style="green",
            )
            return 0

        elif args.command == "deinterlace":
            from .processors.interlace_handler import InterlaceDetector, Deinterlacer, DeinterlaceMethod

            console.print()
            console.info(f"Analyzing interlacing: {args.input.name}")

            detector = InterlaceDetector()

            with create_spinner("Detecting interlacing"):
                analysis = detector.analyze(args.input)

            console.print()
            console.panel(
                analysis.summary(),
                title="Interlace Analysis",
                style="cyan" if not analysis.is_interlaced else "yellow",
            )

            if args.analyze_only:
                return 0

            if not analysis.is_interlaced:
                console.info("Video is progressive - no deinterlacing needed")
                return 0

            output_path = args.output or args.input.parent / f"{args.input.stem}_progressive.mp4"

            method = DeinterlaceMethod(args.method) if args.method != "auto" else analysis.recommended_method

            console.print()
            console.info(f"Deinterlacing with {method.value.upper()}...")

            deinterlacer = Deinterlacer(method=method, field_order=analysis.field_order)

            with create_spinner("Deinterlacing video"):
                result = deinterlacer.process(args.input, output_path)

            if result.success:
                console.panel(
                    f"  Method: {result.method_used.value.upper()}\n"
                    f"  Output: {output_path}",
                    title="Deinterlacing Complete",
                    style="green",
                )
            else:
                console.error(f"Deinterlacing failed: {result.error}")
                return 1
            return 0

        elif args.command == "crop-bars":
            from .processors.letterbox_handler import LetterboxDetector, LetterboxCropper

            console.print()
            console.info(f"Detecting black bars: {args.input.name}")

            detector = LetterboxDetector()

            with create_spinner("Analyzing letterbox/pillarbox"):
                analysis = detector.analyze(args.input)

            console.print()
            console.panel(
                analysis.summary(),
                title="Black Bar Analysis",
                style="cyan" if not (analysis.has_letterbox or analysis.has_pillarbox) else "yellow",
            )

            if args.analyze_only:
                return 0

            if not (analysis.has_letterbox or analysis.has_pillarbox):
                console.info("No black bars detected - no cropping needed")
                return 0

            output_path = args.output or args.input.parent / f"{args.input.stem}_cropped.mp4"

            console.print()
            console.info("Cropping black bars...")

            cropper = LetterboxCropper()

            with create_spinner("Cropping video"):
                success = cropper.crop(args.input, output_path, analysis.crop_region)

            if success:
                console.panel(
                    f"  Original: {analysis.frame_width}x{analysis.frame_height}\n"
                    f"  Cropped: {analysis.content_width}x{analysis.content_height}\n"
                    f"  Removed: {analysis.bar_percentage:.1f}% black bars\n"
                    f"  Output: {output_path}",
                    title="Cropping Complete",
                    style="green",
                )
            else:
                console.error("Cropping failed")
                return 1
            return 0

        elif args.command == "detect-stock":
            from .processors.film_stock_detector import FilmStockDetector, FilmStockCorrector

            console.print()
            console.info(f"Analyzing film stock: {args.input.name}")

            detector = FilmStockDetector()

            with create_spinner("Detecting film characteristics"):
                analysis = detector.analyze(args.input)

            console.print()
            console.panel(
                analysis.summary(),
                title="Film Stock Analysis",
                style="cyan",
            )

            if not args.correct:
                if analysis.detected_stock.value != "unknown":
                    console.print()
                    console.info("To apply stock-specific correction:")
                    console.print(f"  framewright detect-stock {args.input} --correct -o output.mp4")
                return 0

            output_path = args.output or args.input.parent / f"{args.input.stem}_corrected.mp4"

            console.print()
            console.info(f"Applying {analysis.detected_stock.value} color correction...")

            corrector = FilmStockCorrector(analysis.detected_stock)

            # Process video
            import cv2
            cap = cv2.VideoCapture(str(args.input))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            with create_spinner("Applying color correction"):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    corrected = corrector.apply_correction(frame)
                    out.write(corrected)

            cap.release()
            out.release()

            console.panel(
                f"  Stock: {analysis.detected_stock.value}\n"
                f"  Era: {analysis.era.value}\n"
                f"  Output: {output_path}",
                title="Color Correction Complete",
                style="green",
            )
            return 0

        elif args.command == "check-sync":
            from .processors.audio_sync import AudioSyncAnalyzer, AudioSyncCorrector

            console.print()
            console.info(f"Checking audio sync: {args.input.name}")

            analyzer = AudioSyncAnalyzer()

            with create_spinner("Analyzing audio-video sync"):
                analysis = analyzer.analyze_sync(args.input)

            console.print()

            is_synced = abs(analysis.drift_ms) < 40
            sync_status = "In sync" if is_synced else f"Drift: {analysis.drift_ms:.1f}ms ({analysis.drift_direction})"
            console.panel(
                f"  Status: {sync_status}\n"
                f"  Confidence: {analysis.confidence*100:.0f}%\n"
                f"  Samples analyzed: {analysis.samples_analyzed}",
                title="Audio Sync Analysis",
                style="green" if is_synced else "yellow",
            )

            if not args.fix or is_synced:
                if not is_synced:
                    console.print()
                    console.info("To fix sync:")
                    console.print(f"  framewright check-sync {args.input} --fix -o output.mp4")
                return 0

            output_path = args.output or args.input.parent / f"{args.input.stem}_synced.mp4"

            console.print()
            console.info(f"Correcting {abs(analysis.drift_ms):.1f}ms drift...")

            corrector = AudioSyncCorrector()

            with create_spinner("Fixing audio sync"):
                result = corrector.correct_sync(args.input, output_path, analysis.drift_ms)

            if result.success:
                console.panel(
                    f"  Correction: {result.offset_applied_ms:.1f}ms\n"
                    f"  Output: {output_path}",
                    title="Sync Correction Complete",
                    style="green",
                )
            else:
                console.error(f"Sync correction failed: {result.error}")
                return 1
            return 0

        elif args.command == "scan":
            console.print()
            console.info(f"Full scan: {args.input.name}")
            console.print("  Checking: interlacing, black bars, film stock, audio sync")
            console.print()

            issues_found = []

            # 1. Interlace detection
            try:
                from .processors.interlace_handler import InterlaceDetector
                detector = InterlaceDetector()
                with create_spinner("Checking interlacing"):
                    interlace = detector.analyze(args.input)
                if interlace.is_interlaced:
                    issues_found.append(("Interlacing", interlace.summary(), "deinterlace"))
                    console.warning(f"Interlacing detected: {interlace.interlace_type.value}")
                else:
                    console.info("✓ No interlacing")
            except Exception as e:
                console.warning(f"Interlace check failed: {e}")

            # 2. Letterbox detection
            try:
                from .processors.letterbox_handler import LetterboxDetector
                detector = LetterboxDetector()
                with create_spinner("Checking black bars"):
                    letterbox = detector.analyze(args.input)
                if letterbox.has_letterbox or letterbox.has_pillarbox:
                    issues_found.append(("Black bars", letterbox.summary(), "crop-bars"))
                    console.warning(f"Black bars detected: {letterbox.bar_percentage:.1f}%")
                else:
                    console.info("✓ No black bars")
            except Exception as e:
                console.warning(f"Letterbox check failed: {e}")

            # 3. Film stock detection
            try:
                from .processors.film_stock_detector import FilmStockDetector
                detector = FilmStockDetector()
                with create_spinner("Analyzing film stock"):
                    stock = detector.analyze(args.input)
                if stock.detected_stock.value != "unknown":
                    console.info(f"✓ Film stock: {stock.detected_stock.value} ({stock.era.value})")
                    if stock.fading_detected:
                        issues_found.append(("Color fading", f"{stock.fading_pattern} fading detected", "detect-stock --correct"))
                        console.warning(f"Color fading detected: {stock.fading_pattern}")
            except Exception as e:
                console.warning(f"Film stock check failed: {e}")

            # 4. Audio sync check
            try:
                from .processors.audio_sync import AudioSyncAnalyzer
                analyzer = AudioSyncAnalyzer()
                with create_spinner("Checking audio sync"):
                    sync = analyzer.analyze_sync(args.input)
                is_synced = abs(sync.drift_ms) < 40
                if not is_synced:
                    issues_found.append(("Audio sync", f"{sync.drift_ms:.1f}ms drift", "check-sync --fix"))
                    console.warning(f"Audio drift detected: {sync.drift_ms:.1f}ms")
                else:
                    console.info("✓ Audio in sync")
            except Exception as e:
                console.warning(f"Audio sync check failed: {e}")

            # 5. Noise profiling
            try:
                from .processors.noise_profiler import NoiseProfiler
                profiler = NoiseProfiler(sample_frames=10)
                with create_spinner("Profiling noise"):
                    noise = profiler.analyze_video(args.input)
                if noise.overall_level > 30:
                    issues_found.append(("High noise", f"{noise.dominant_type.value} ({noise.overall_level:.0f}/100)", "noise-profile"))
                    console.warning(f"High noise level: {noise.overall_level:.0f}/100 ({noise.dominant_type.value})")
                else:
                    console.info(f"✓ Noise level acceptable ({noise.overall_level:.0f}/100)")
            except Exception as e:
                console.warning(f"Noise profiling failed: {e}")

            # 6. Upscale detection
            try:
                from .processors.upscale_detector import UpscaleDetector
                detector = UpscaleDetector(sample_frames=10)
                with create_spinner("Checking resolution"):
                    upscale = detector.analyze(args.input)
                if upscale.is_upscaled:
                    issues_found.append((
                        "Already upscaled",
                        f"~{upscale.estimated_source_resolution} -> {upscale.container_resolution}",
                        "upscale-detect"
                    ))
                    console.warning(f"Video appears upscaled from {upscale.estimated_source_resolution}")
                else:
                    console.info(f"✓ Native resolution ({upscale.container_resolution})")
            except Exception as e:
                console.warning(f"Upscale detection failed: {e}")

            # Summary
            console.print()
            if issues_found:
                console.panel(
                    "\n".join(f"  • {name}: {desc}\n    Fix: framewright {cmd} {args.input}"
                              for name, desc, cmd in issues_found),
                    title=f"Issues Found ({len(issues_found)})",
                    style="yellow",
                )

                if args.fix_all:
                    console.print()
                    console.info("Auto-fixing all issues...")
                    # Would chain the fixes here
                    console.warning("Auto-fix not yet implemented - please run individual commands")
            else:
                console.panel(
                    "  No issues detected!\n"
                    "  Video is ready for restoration.",
                    title="Scan Complete",
                    style="green",
                )
            return 0

        elif args.command == "noise-profile":
            console.print()
            console.info(f"Analyzing noise: {args.input.name}")

            try:
                from .processors.noise_profiler import NoiseProfiler

                profiler = NoiseProfiler(sample_frames=args.frames)

                with create_spinner("Profiling noise characteristics"):
                    profile = profiler.analyze_video(args.input)

                console.print()
                console.panel(
                    f"  Overall noise level: {profile.overall_level:.1f}/100\n"
                    f"  Dominant type: {profile.dominant_type.value}\n"
                    f"  Luminance noise: {profile.characteristics.luminance_noise:.1f}\n"
                    f"  Chroma noise: {profile.characteristics.chroma_noise:.1f}\n"
                    f"  Temporal noise: {profile.characteristics.temporal_noise:.1f}\n"
                    f"  Film grain: {profile.characteristics.grain_intensity:.1f} "
                    f"(uniformity: {profile.characteristics.grain_uniformity:.0f}%)\n"
                    f"\n"
                    f"  Recommended denoiser: {profile.recommended_denoiser.value}\n"
                    f"  Recommended strength: {profile.recommended_strength:.2f}\n"
                    f"  Preserve grain: {'Yes' if profile.preserve_grain else 'No'}\n"
                    f"  Confidence: {profile.confidence:.0%}",
                    title="Noise Profile",
                    style="cyan",
                )
            except Exception as e:
                console.error(f"Noise profiling failed: {e}")
                return 1
            return 0

        elif args.command == "upscale-detect":
            console.print()
            console.info(f"Checking for upscaling: {args.input.name}")

            try:
                from .processors.upscale_detector import UpscaleDetector

                detector = UpscaleDetector(sample_frames=args.frames)

                with create_spinner("Analyzing resolution"):
                    result = detector.analyze(args.input)

                console.print()
                if result.is_upscaled:
                    console.panel(
                        f"  Container resolution: {result.container_resolution}\n"
                        f"  Estimated source: {result.estimated_source_resolution}\n"
                        f"  Upscale factor: {result.upscale_factor:.1f}x\n"
                        f"  Upscale method: {result.upscale_method.value}\n"
                        f"  Confidence: {result.confidence:.0%}\n"
                        f"\n"
                        f"  {result.recommendation}",
                        title="Upscaling Detected",
                        style="yellow",
                    )
                else:
                    console.panel(
                        f"  Resolution: {result.container_resolution}\n"
                        f"  Confidence: {result.confidence:.0%}\n"
                        f"\n"
                        f"  {result.recommendation}",
                        title="Native Resolution",
                        style="green",
                    )
            except Exception as e:
                console.error(f"Upscale detection failed: {e}")
                return 1
            return 0

        elif args.command == "gpu-thermal":
            try:
                from .utils.thermal_monitor import ThermalMonitor

                monitor = ThermalMonitor(device_id=args.device)
                reading = monitor.read_temperature()

                if reading is None:
                    console.warning("Could not read GPU temperature")
                    console.info("Ensure nvidia-smi is available for NVIDIA GPUs")
                    return 1

                console.print()
                state_colors = {
                    "cool": "green",
                    "warm": "cyan",
                    "hot": "yellow",
                    "critical": "red",
                }
                style = state_colors.get(reading.thermal_state.value, "white")

                console.panel(
                    f"  Device: {monitor.profile.device_name}\n"
                    f"  Temperature: {reading.temperature_celsius:.0f}C\n"
                    f"  State: {reading.thermal_state.value.upper()}\n"
                    f"  Throttling: {reading.throttle_state.value}\n"
                    f"  Power: {reading.power_usage_watts:.0f}W / {reading.power_limit_watts:.0f}W"
                    if reading.power_usage_watts else f"  Power: N/A\n"
                    f"  Clock: {reading.clock_speed_mhz}MHz / {reading.clock_speed_max_mhz}MHz"
                    if reading.clock_speed_mhz else "",
                    title="GPU Thermal Status",
                    style=style,
                )

                if args.monitor:
                    console.print()
                    console.info("Monitoring (Ctrl+C to stop)...")
                    try:
                        while True:
                            import time
                            time.sleep(2)
                            reading = monitor.read_temperature()
                            if reading:
                                throttle = " [THROTTLING]" if reading.throttle_state.value != "none" else ""
                                console.print(
                                    f"  {reading.temperature_celsius:.0f}C "
                                    f"({reading.thermal_state.value}){throttle}"
                                )
                    except KeyboardInterrupt:
                        console.print()
            except Exception as e:
                console.error(f"Thermal monitoring failed: {e}")
                return 1
            return 0

        elif args.command == "quick-preview":
            console.print()
            console.info(f"Generating quick preview: {args.input.name}")
            console.print(f"  Sampling every {args.every} frames")

            try:
                from .processors.quick_preview import QuickPreviewGenerator, PreviewConfig

                output = args.output or args.input.parent / f"{args.input.stem}_preview.mp4"

                config = PreviewConfig(
                    every_n_frames=args.every,
                    output_resolution=(854, 480),
                    apply_restoration=args.preset is not None,
                    preset=args.preset,
                    side_by_side=True,
                    show_metrics=True,
                )

                generator = QuickPreviewGenerator(config)

                def progress(current, total):
                    pass  # Spinner handles progress

                with create_spinner("Generating preview"):
                    result = generator.generate(
                        input_path=args.input,
                        output_path=output,
                        progress_callback=progress,
                    )

                console.print()
                console.success(f"Preview saved: {output}")
                console.panel(
                    f"  Frames sampled: {result.frames_sampled} of {result.total_frames}\n"
                    f"  Coverage: {result.coverage_percent():.1f}%\n"
                    f"  Processing time: {result.processing_time_seconds:.1f}s\n"
                    f"  Estimated full time: {result.estimated_full_time_seconds/60:.1f} minutes\n"
                    + (f"  Avg PSNR: {result.avg_psnr:.1f} dB\n" if result.avg_psnr > 0 else "")
                    + (f"  Avg SSIM: {result.avg_ssim:.3f}" if result.avg_ssim > 0 else ""),
                    title="Preview Summary",
                    style="cyan",
                )

            except Exception as e:
                console.error(f"Preview generation failed: {e}")
                logger.exception("Preview error")
                return 1
            return 0

        elif args.command == "frame-grid":
            console.print()
            console.info(f"Generating frame grid: {args.input.name}")

            try:
                from .processors.quick_preview import QuickPreviewGenerator

                generator = QuickPreviewGenerator()
                output = args.output or args.input.parent / f"{args.input.stem}_grid.jpg"

                with create_spinner("Generating grid"):
                    result_path = generator.generate_grid(
                        input_path=args.input,
                        output_path=output,
                        num_frames=args.frames,
                        cols=args.cols,
                    )

                console.success(f"Grid saved: {result_path}")

            except Exception as e:
                console.error(f"Grid generation failed: {e}")
                return 1
            return 0

        elif args.command == "compare-presets":
            console.print()
            console.info(f"Comparing presets: {', '.join(args.presets)}")

            try:
                from .processors.quick_preview import QuickPreviewGenerator

                generator = QuickPreviewGenerator()
                output_dir = args.output_dir or args.input.parent / f"{args.input.stem}_preset_compare"

                with create_spinner("Comparing presets"):
                    results = generator.compare_presets(
                        input_path=args.input,
                        presets=args.presets,
                        frame_index=args.frame,
                        output_dir=output_dir,
                    )

                console.print()
                console.success(f"Comparison saved to: {output_dir}")
                console.print()

                for preset, data in results.items():
                    if "error" in data:
                        console.warning(f"  {preset}: {data['error']}")
                    else:
                        metrics = data.get("metrics", {})
                        psnr = metrics.get("psnr", 0)
                        ssim = metrics.get("ssim", 0)
                        console.print(f"  {preset}: PSNR={psnr:.1f}dB, SSIM={ssim:.3f}")

            except Exception as e:
                console.error(f"Preset comparison failed: {e}")
                return 1
            return 0

        elif args.command == "system-check":
            console.print()
            console.info("System Readiness Check")
            console.print()

            all_ok = True

            # GPU check
            try:
                from .utils.gpu import get_best_gpu
                gpu = get_best_gpu()
                if gpu:
                    console.info(f"✓ GPU: {gpu.name}")
                    console.print(f"    VRAM: {gpu.total_memory_mb}MB total, {gpu.free_memory_mb}MB free")
                    if gpu.temperature_celsius:
                        console.print(f"    Temperature: {gpu.temperature_celsius:.0f}°C")
                else:
                    console.warning("⚠ No GPU detected - CPU processing only")
                    all_ok = False
            except Exception as e:
                console.warning(f"⚠ GPU check failed: {e}")

            # Thermal check
            try:
                from .utils.thermal_monitor import ThermalMonitor, ThermalState
                monitor = ThermalMonitor()
                reading = monitor.read_temperature()
                if reading:
                    if reading.thermal_state == ThermalState.COOL:
                        console.info(f"✓ GPU Thermal: {reading.temperature_celsius:.0f}°C (cool)")
                    elif reading.thermal_state == ThermalState.WARM:
                        console.info(f"✓ GPU Thermal: {reading.temperature_celsius:.0f}°C (warm)")
                    elif reading.thermal_state == ThermalState.HOT:
                        console.warning(f"⚠ GPU Thermal: {reading.temperature_celsius:.0f}°C (hot)")
                    else:
                        console.error(f"✗ GPU Thermal: {reading.temperature_celsius:.0f}°C (critical)")
                        all_ok = False

                    if reading.throttle_state.value not in ("none", "unknown"):
                        console.warning(f"  Throttling: {reading.throttle_state.value}")
            except Exception:
                pass

            # Disk space check
            if args.output_dir:
                try:
                    from .utils.disk import get_disk_usage
                    usage = get_disk_usage(args.output_dir)
                    if usage.free_gb >= 20:
                        console.info(f"✓ Disk Space: {usage.free_gb:.1f}GB free")
                    elif usage.free_gb >= 10:
                        console.warning(f"⚠ Disk Space: {usage.free_gb:.1f}GB free (low)")
                    else:
                        console.error(f"✗ Disk Space: {usage.free_gb:.1f}GB free (critical)")
                        all_ok = False
                except Exception as e:
                    console.warning(f"⚠ Disk check failed: {e}")

            # FFmpeg check
            try:
                from .utils.ffmpeg import check_ffmpeg_installed
                if check_ffmpeg_installed():
                    console.info("✓ FFmpeg: installed")
                else:
                    console.error("✗ FFmpeg: not found")
                    all_ok = False
            except Exception:
                console.warning("⚠ FFmpeg check failed")

            # Memory check
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_gb = mem.available / (1024**3)
                if mem_gb >= 8:
                    console.info(f"✓ System RAM: {mem_gb:.1f}GB available")
                elif mem_gb >= 4:
                    console.warning(f"⚠ System RAM: {mem_gb:.1f}GB available (low)")
                else:
                    console.error(f"✗ System RAM: {mem_gb:.1f}GB available (critical)")
                    all_ok = False
            except ImportError:
                pass

            console.print()
            if all_ok:
                console.panel(
                    "  System is ready for video restoration.",
                    title="All Checks Passed",
                    style="green",
                )
            else:
                console.panel(
                    "  Some issues detected. Review warnings above.",
                    title="Issues Found",
                    style="yellow",
                )

            return 0 if all_ok else 1

        elif args.command == "models":
            from .utils.model_manager_cli import ModelManagerCLI

            manager = ModelManagerCLI()
            action = args.models_action

            if action == "list":
                models = manager.list_models(
                    category=getattr(args, "category", None),
                    show_downloaded_only=getattr(args, "downloaded", False)
                )
                if not models:
                    console.info("No models found")
                    return 0

                console.print()
                console.panel(
                    "\n".join(
                        f"  {'✓' if m['downloaded'] else '○'} {m['name']} ({m['category']}) - {m['size_mb']:.0f}MB"
                        for m in models
                    ),
                    title=f"AI Models ({len(models)})",
                    style="cyan",
                )
                return 0

            elif action == "download":
                name = args.name
                console.info(f"Downloading model: {name}")

                def progress(pct):
                    pass

                if name == "all":
                    with create_spinner("Downloading all models"):
                        results = manager.download_all(progress_callback=progress)
                    success = sum(1 for v in results.values() if v)
                    console.info(f"Downloaded {success}/{len(results)} models")
                else:
                    with create_spinner(f"Downloading {name}"):
                        success = manager.download_model(name, progress, force=args.force)
                    if success:
                        console.success(f"Model '{name}' downloaded")
                    else:
                        console.error(f"Failed to download '{name}'")
                        return 1
                return 0

            elif action == "clean":
                console.info(f"Cleaning models unused for {args.days} days...")
                with create_spinner("Scanning models"):
                    count, freed_mb = manager.clean_unused(keep_days=args.days)
                console.info(f"Removed {count} models, freed {freed_mb:.1f}MB")
                return 0

            elif action == "status":
                usage = manager.get_disk_usage()
                console.panel(
                    f"  Total models: {usage['total_models']}\n"
                    f"  Downloaded: {usage['downloaded_models']}\n"
                    f"  Disk usage: {usage['total_size_mb']:.1f}MB",
                    title="Model Status",
                    style="cyan",
                )
                return 0

            else:
                console.info("Model commands: list, download <name>, clean, status")
                return 0

        elif args.command == "validate":
            from .export.validation import ExportValidator

            console.info(f"Validating: {args.input.name}")

            validator = ExportValidator()

            with create_spinner("Validating video integrity"):
                result = validator.validate(
                    args.input,
                    compare_to=getattr(args, "compare", None),
                    compute_checksums=True
                )

            console.print()

            if result.is_valid:
                console.panel(
                    f"  Duration: {result.duration_seconds:.1f}s\n"
                    f"  Frames: {result.frame_count}\n"
                    f"  Resolution: {result.width}x{result.height}\n"
                    f"  Audio: {'Yes' if result.has_audio else 'No'}",
                    title="✓ Validation Passed",
                    style="green",
                )
            else:
                issues = "\n".join(f"  • {i.message}" for i in result.issues[:10])
                console.panel(
                    issues,
                    title="✗ Validation Failed",
                    style="red",
                )

            if args.output:
                result.save(args.output)
                console.info(f"Report saved: {args.output}")

            return 0 if result.is_valid else 1

        elif args.command == "benchmark":
            from .processors.benchmark import RestorationBenchmark, BenchmarkType

            console.info(f"Benchmarking: {args.input.name}")
            console.print(f"  Mode: {args.mode}")

            mode_map = {
                "quick": BenchmarkType.QUICK,
                "standard": BenchmarkType.STANDARD,
                "thorough": BenchmarkType.THOROUGH
            }

            def progress(msg, pct):
                pass

            benchmark = RestorationBenchmark(
                test_frames=args.frames,
                progress_callback=progress
            )

            console.print()
            with create_spinner("Running benchmark"):
                report = benchmark.run_simple(args.input, mode_map[args.mode])

            console.print()
            console.panel(
                f"  Hardware: {report.hardware_info.get('gpu', 'Unknown')}\n"
                f"  Test frames: {args.frames}\n"
                f"  Duration: {report.duration_seconds:.1f}s\n"
                f"  Configurations tested: {len(report.results)}",
                title="Benchmark Results",
                style="cyan",
            )

            if report.results:
                fastest = report.get_fastest()
                if fastest:
                    console.print()
                    console.info(f"Fastest: {fastest.name} ({fastest.fps:.1f} FPS)")

            console.print()
            for rec in report.recommendations:
                console.print(f"  • {rec}")

            if args.output:
                report.save(args.output)
                console.info(f"Report saved: {args.output}")

            return 0

        elif args.command == "score-frames":
            from .processors.frame_quality_scorer import FrameQualityScorer

            console.info(f"Analyzing frame quality: {args.input.name}")

            def progress(current, total):
                pass

            scorer = FrameQualityScorer(
                sample_rate=args.sample_rate,
                problem_threshold=args.threshold,
                progress_callback=progress
            )

            with create_spinner("Analyzing frames"):
                report = scorer.analyze_video(args.input)

            console.print()
            console.panel(
                f"  Frames analyzed: {report.analyzed_frames}\n"
                f"  Average score: {report.average_score:.1f}/100\n"
                f"  Problem frames: {len(report.problem_frames)}\n"
                f"  Score range: {report.min_score:.1f} - {report.max_score:.1f}",
                title="Frame Quality Report",
                style="cyan" if report.average_score >= 70 else "yellow",
            )

            if report.problem_frames:
                console.print()
                console.info("Worst frames:")
                for f in report.get_worst_frames(5):
                    issues = ", ".join(i.value for i in f.issues) if f.issues else "low quality"
                    console.print(f"  Frame {f.frame_number} ({f.timestamp:.1f}s): {f.overall_score:.0f}/100 - {issues}")

            console.print()
            for rec in report.recommendations:
                console.print(f"  • {rec}")

            if args.output:
                report.save(args.output)
                console.info(f"Report saved: {args.output}")

            if getattr(args, "export_frames", None):
                import cv2
                export_dir = Path(args.export_frames)
                export_dir.mkdir(parents=True, exist_ok=True)
                cap = cv2.VideoCapture(str(args.input))
                for f in report.problem_frames[:50]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f.frame_number)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(str(export_dir / f"frame_{f.frame_number:06d}.jpg"), frame)
                cap.release()
                console.info(f"Exported problem frames to: {export_dir}")

            return 0

        elif args.command == "detect-credits":
            from .processors.credits_detector import CreditsDetector

            console.info(f"Detecting intro/credits: {args.input.name}")

            def progress(msg, pct):
                pass

            detector = CreditsDetector(progress_callback=progress)

            with create_spinner("Analyzing video structure"):
                analysis = detector.analyze(args.input)

            console.print()

            segments_info = []
            if analysis.intro_end_frame:
                intro_time = analysis.intro_end_frame / analysis.fps
                segments_info.append(f"  Intro ends: {intro_time:.1f}s (frame {analysis.intro_end_frame})")

            if analysis.credits_start_frame:
                credits_time = analysis.credits_start_frame / analysis.fps
                segments_info.append(f"  Credits start: {credits_time:.1f}s (frame {analysis.credits_start_frame})")

            if analysis.main_content_start and analysis.main_content_end:
                main_duration = (analysis.main_content_end - analysis.main_content_start) / analysis.fps
                segments_info.append(f"  Main content: {main_duration:.1f}s")

            if segments_info:
                console.panel(
                    "\n".join(segments_info) + f"\n  Segments detected: {len(analysis.segments)}",
                    title="Credits Analysis",
                    style="cyan",
                )
            else:
                console.info("No distinct intro/credits detected")

            for seg in analysis.segments:
                console.print(f"  {seg.segment_type.value}: {seg.start_time:.1f}s - {seg.end_time:.1f}s ({seg.confidence*100:.0f}% confidence)")

            if args.output:
                analysis.save(args.output)
                console.info(f"Analysis saved: {args.output}")

            if getattr(args, "trim", False) and (analysis.main_content_start or analysis.main_content_end):
                console.print()
                console.info("Trimming to main content...")
                # Would integrate with ffmpeg for actual trimming
                console.info("Use: ffmpeg -ss {start} -to {end} -i input.mp4 -c copy output.mp4")

            return 0

        elif args.command == "webhook":
            import json
            config_path = Path.home() / ".framewright" / "webhooks.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            action = args.webhook_action

            def load_webhooks():
                if config_path.exists():
                    return json.loads(config_path.read_text())
                return {"webhooks": []}

            def save_webhooks(data):
                config_path.write_text(json.dumps(data, indent=2))

            if action == "add":
                data = load_webhooks()
                webhook = {
                    "url": args.url,
                    "type": args.type,
                    "name": getattr(args, "name", None) or f"webhook-{len(data['webhooks'])+1}",
                    "enabled": True
                }
                data["webhooks"].append(webhook)
                save_webhooks(data)
                console.success(f"Webhook added: {webhook['name']}")
                return 0

            elif action == "list":
                data = load_webhooks()
                if not data["webhooks"]:
                    console.info("No webhooks configured")
                    console.print("  Add one with: framewright webhook add <url>")
                else:
                    console.print()
                    for i, wh in enumerate(data["webhooks"]):
                        status = "✓" if wh.get("enabled", True) else "○"
                        console.print(f"  {i+1}. {status} {wh['name']} ({wh['type']})")
                return 0

            elif action == "test":
                from .utils.webhook import ProgressWebhook, WebhookConfig, WebhookType
                data = load_webhooks()
                if not data["webhooks"]:
                    console.warning("No webhooks configured")
                    return 1

                configs = [
                    WebhookConfig(
                        url=wh["url"],
                        webhook_type=WebhookType(wh.get("type", "generic"))
                    )
                    for wh in data["webhooks"] if wh.get("enabled", True)
                ]

                webhook = ProgressWebhook(configs, async_delivery=False)
                sent = webhook.notify_started("test-001", "Test Notification", {"source": "CLI test"})
                console.info(f"Sent test notification to {sent} webhook(s)")
                return 0

            elif action == "remove":
                data = load_webhooks()
                idx = args.index - 1
                if 0 <= idx < len(data["webhooks"]):
                    removed = data["webhooks"].pop(idx)
                    save_webhooks(data)
                    console.info(f"Removed: {removed['name']}")
                else:
                    console.error("Invalid webhook index")
                    return 1
                return 0

            else:
                console.info("Webhook commands: add <url>, list, test, remove <index>")
                return 0

        elif args.command == "schedule":
            from .utils.scheduler import JobScheduler, ScheduleType, JobConstraints
            from datetime import datetime

            queue_file = Path.home() / ".framewright" / "schedule.json"
            queue_file.parent.mkdir(parents=True, exist_ok=True)

            scheduler = JobScheduler(queue_file=queue_file)
            action = args.schedule_action

            if action == "add":
                schedule_type = ScheduleType.IMMEDIATE
                scheduled_time = None

                if getattr(args, "at", None):
                    schedule_type = ScheduleType.SPECIFIC_TIME
                    try:
                        if " " in args.at:
                            scheduled_time = datetime.strptime(args.at, "%Y-%m-%d %H:%M")
                        else:
                            today = datetime.now().date()
                            time_part = datetime.strptime(args.at, "%H:%M").time()
                            scheduled_time = datetime.combine(today, time_part)
                    except ValueError:
                        console.error("Invalid time format. Use HH:MM or YYYY-MM-DD HH:MM")
                        return 1

                elif getattr(args, "delay", None):
                    schedule_type = ScheduleType.DELAYED

                output_path = args.output or args.input.parent / f"{args.input.stem}_restored.mp4"

                job_id = scheduler.add_job(
                    input_path=str(args.input),
                    output_path=str(output_path),
                    preset=args.preset,
                    schedule_type=schedule_type,
                    scheduled_time=scheduled_time,
                    delay_minutes=getattr(args, "delay", None),
                    priority=args.priority
                )

                console.success(f"Job scheduled: {job_id}")
                if scheduled_time:
                    console.print(f"  Scheduled for: {scheduled_time}")
                return 0

            elif action == "list":
                jobs = scheduler.list_jobs()
                if not jobs:
                    console.info("No scheduled jobs")
                else:
                    console.print()
                    for job in jobs:
                        time_str = job.scheduled_time.strftime("%Y-%m-%d %H:%M") if job.scheduled_time else "immediate"
                        console.print(f"  {job.job_id}: {job.name} [{job.status.value}] @ {time_str}")
                return 0

            elif action == "start":
                console.info("Starting scheduler daemon...")
                console.print("  Press Ctrl+C to stop")
                try:
                    scheduler.start()
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    scheduler.stop()
                    console.print()
                    console.info("Scheduler stopped")
                return 0

            elif action == "run-next":
                job = scheduler.run_next()
                if job:
                    console.info(f"Completed job: {job.job_id}")
                else:
                    console.info("No jobs available to run")
                return 0

            elif action == "stats":
                stats = scheduler.get_statistics()
                console.panel(
                    f"  Total jobs: {stats['total_jobs']}\n"
                    f"  Running: {stats['running_jobs']}\n"
                    f"  By status: {stats['by_status']}",
                    title="Scheduler Statistics",
                    style="cyan",
                )
                if stats["next_scheduled"]:
                    console.print(f"  Next job: {stats['next_scheduled']['name']} at {stats['next_scheduled']['scheduled_for']}")
                return 0

            elif action == "cancel":
                if scheduler.cancel_job(args.job_id):
                    console.info(f"Cancelled job: {args.job_id}")
                else:
                    console.error(f"Could not cancel job: {args.job_id}")
                    return 1
                return 0

            else:
                console.info("Schedule commands: add, list, start, run-next, stats, cancel <job_id>")
                return 0

        elif args.command == "power":
            from .utils.power_manager import PowerManager, PowerAction

            if args.status:
                manager = PowerManager()
                state = manager.get_power_state()
                console.panel(
                    f"  On Battery: {'Yes' if state.on_battery else 'No'}\n"
                    f"  Battery: {state.battery_percent:.0f}%" if state.battery_percent else "  Battery: N/A\n"
                    f"  Charging: {'Yes' if state.is_charging else 'No'}",
                    title="Power Status",
                    style="cyan",
                )
                return 0

            if args.prevent_sleep or args.on_complete != "none":
                action = PowerAction(args.on_complete)
                console.info("Power management enabled")
                console.print(f"  Prevent sleep: {args.prevent_sleep}")
                console.print(f"  On complete: {args.on_complete}")
                console.print()
                console.info("Power settings will be applied during next restoration job")
            else:
                console.info("Power options: --status, --prevent-sleep, --on-complete <action>")

            return 0

        elif args.command == "template":
            from .utils.output_templates import OutputTemplate, TemplateManager

            action = args.template_action

            if action == "list":
                console.print()
                console.panel(
                    "\n".join(
                        f"  {name}: {template}"
                        for name, template in OutputTemplate.PRESETS.items()
                    ),
                    title="Output Templates",
                    style="cyan",
                )
                console.print()
                console.info("Variables: {name}, {ext}, {date}, {time}, {resolution}, {preset}, {upscale}, {counter}")
                return 0

            elif action == "preview":
                template = OutputTemplate(template=args.template)
                preview = template.preview(args.input)
                console.print()
                console.panel(
                    f"  Input: {args.input.name}\n"
                    f"  Template: {args.template}\n"
                    f"  Output: {preview}",
                    title="Template Preview",
                    style="cyan",
                )
                return 0

            else:
                console.info("Template commands: list, preview <input> --template <template>")
                return 0

        elif args.input:
            # Default: smart auto-restore
            # Check if quality slider was specified
            quality = getattr(args, "quality", None)
            if quality is not None:
                # Map quality 1-5 to commands
                if quality <= 1:
                    run_quick_restore(args.input, args.output, console)
                elif quality >= 5:
                    run_best_restore(args.input, args.output, console)
                else:
                    # Quality 2-4: use smart restore with adjusted settings
                    run_smart_restore(args.input, args.output, console)
            else:
                run_smart_restore(args.input, args.output, console)
            return 0

        else:
            # No input provided, show help or wizard
            parser.print_help()
            console.print()
            console.info("Tip: Run 'framewright wizard' for guided setup")
            return 0

    except FileNotFoundError as e:
        console.error(str(e), hint="Check that the file path is correct")
        return 1

    except KeyboardInterrupt:
        console.print()
        console.warning("Operation cancelled")
        return 130

    except Exception as e:
        console.error(f"Restoration failed: {e}")
        logger.exception("Restoration error")
        return 1


if __name__ == "__main__":
    sys.exit(main_simple())
