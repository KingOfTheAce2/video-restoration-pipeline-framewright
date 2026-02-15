#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
FrameWright CLI - Video Restoration Pipeline
Command-line interface for video enhancement and restoration.
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable else range(total or 0)

try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    argcomplete = None  # type: ignore
    ARGCOMPLETE_AVAILABLE = False


# Supported output formats
SUPPORTED_FORMATS = ['mkv', 'mp4', 'webm', 'avi', 'mov']


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(message: str, color: str = Colors.OKBLUE):
    """Print colored message to console."""
    try:
        print(f"{color}{message}{Colors.ENDC}")
    except UnicodeEncodeError:
        # Fallback for terminals that can't handle unicode
        print(message.encode('ascii', errors='replace').decode('ascii'))


def print_header():
    """Print CLI header."""
    # Use ASCII-safe box drawing for Windows compatibility
    header = """
    +-------------------------------------------+
    |       FrameWright v1.0.0-dev              |
    |    Video Restoration Pipeline             |
    +-------------------------------------------+
    """
    print_colored(header, Colors.HEADER)


def validate_input(input_path: str) -> Path:
    """Validate input file or URL.

    For URLs (http/https), returns a PurePosixPath to preserve
    forward slashes in the URL on all platforms.
    For local files, validates existence and returns the Path.
    """
    if input_path.startswith(('http://', 'https://')):
        # Use PurePosixPath to preserve forward slashes on Windows
        from pathlib import PurePosixPath
        return PurePosixPath(input_path)

    path = Path(input_path)
    if not path.exists():
        print_colored(f"Error: Input file not found: {input_path}", Colors.FAIL)
        sys.exit(1)
    return path


def validate_scale(scale: int) -> int:
    """Validate upscaling factor."""
    if scale not in [2, 4]:
        print_colored("Error: Scale must be 2 or 4", Colors.FAIL)
        sys.exit(1)
    return scale


def get_output_format(args) -> str:
    """Get output format from args or infer from output path."""
    if hasattr(args, 'format') and args.format:
        return args.format
    if hasattr(args, 'output') and args.output:
        suffix = Path(args.output).suffix.lower().lstrip('.')
        if suffix in SUPPORTED_FORMATS:
            return suffix
    return 'mkv'  # Default


def get_output_path(args) -> Path:
    """Determine output path from --output or --output-dir arguments."""
    fmt = get_output_format(args)

    if hasattr(args, 'output') and args.output:
        output = Path(args.output)
        # Update extension if format specified
        if hasattr(args, 'format') and args.format:
            output = output.with_suffix(f'.{fmt}')
        return output
    elif hasattr(args, 'output_dir') and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"restored_video.{fmt}"
    else:
        return Path(f"./restored_video.{fmt}")


def get_output_dir(args) -> Path:
    """Get the output directory for saving additional files like improvements.md."""
    if hasattr(args, 'output_dir') and args.output_dir:
        return Path(args.output_dir)
    elif hasattr(args, 'output') and args.output:
        return Path(args.output).parent
    else:
        return Path(".")


def generate_improvements_report(output_dir: Path) -> Path:
    """Generate improvements.md report in the specified output directory."""
    from datetime import datetime

    report_path = output_dir / "improvements.md"

    report_content = f'''# FrameWright - Potential Improvements Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Output Directory:** {output_dir}

---

## High Priority

### 1. Parallel Frame Processing
- Use ThreadPoolExecutor for 2-4x speedup on multi-GPU systems

### 2. GPU Memory Pre-validation
- Add pre-flight VRAM check before processing starts

### 3. Batch Processing
- Add support for processing multiple videos in sequence

---

## Medium Priority

### 4. RIFE Model Auto-download
- Automatically download RIFE models on first use

### 5. Progress Improvements
- Add frame-level progress with ETA estimates

### 6. Configuration Presets
- Add "fast", "quality", "archive" presets

---

## Low Priority

### 7. Plugin Architecture
- Allow custom processors to be added without code changes

### 8. Async I/O Operations
- Use asyncio for I/O-bound operations

---

## Notes

This report was auto-generated based on codebase analysis.
For the full detailed analysis, see `docs/improvements.md` in the project repository.
'''

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content)

    return report_path


def _load_effective_config(args) -> Dict[str, Any]:
    """Load effective configuration by merging config file with CLI args."""
    try:
        from .utils.config_file import ConfigFileManager
        manager = ConfigFileManager()
        if manager.config_exists():
            manager.load()
            cli_args = {
                'scale': getattr(args, 'scale', None),
                'model': getattr(args, 'model', None),
                'format': getattr(args, 'format', None),
                'quality': getattr(args, 'quality', None),
                'model_dir': getattr(args, 'model_dir', None),
                'enable_rife': getattr(args, 'enable_rife', None),
                'target_fps': getattr(args, 'target_fps', None),
                'rife_model': getattr(args, 'rife_model', None),
                'auto_enhance': getattr(args, 'auto_enhance', None),
                'scratch_sensitivity': getattr(args, 'scratch_sensitivity', None),
                'grain_reduction': getattr(args, 'grain_reduction', None),
                'colorize': getattr(args, 'colorize', None),
                'colorize_model': getattr(args, 'colorize_model', None),
                'remove_watermark': getattr(args, 'remove_watermark', None),
                'profile': getattr(args, 'profile', None),
            }
            return manager.merge_with_cli_args(cli_args)
    except ImportError:
        pass
    return {}


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _format_duration_dry(seconds: float) -> str:
    """Format seconds to human readable duration."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_number(num: int) -> str:
    """Format number with comma separators."""
    return f"{num:,}"


def print_dry_run_output(args, source: str, output_path: Path, output_format: str):
    """Print detailed dry-run analysis with improved formatting."""
    from .dry_run import perform_dry_run
    from .config import PRESETS

    input_path = Path(source) if not source.startswith(('http://', 'https://')) else None

    if input_path is None:
        print_colored("\nDRY RUN - Cannot analyze URL sources in detail", Colors.WARNING)
        print_colored("Run without --dry-run to process the URL.", Colors.OKCYAN)
        return

    if not input_path.exists():
        print_colored(f"\nError: Input file not found: {source}", Colors.FAIL)
        sys.exit(1)

    try:
        # Perform dry-run analysis
        result = perform_dry_run(
            video_path=input_path,
            output_path=output_path,
            scale_factor=args.scale,
            model_name=args.model,
            crf=args.quality,
            output_format=output_format,
            enable_interpolation=args.enable_rife,
            target_fps=args.target_fps,
            enable_auto_enhance=args.auto_enhance,
            enable_face_restore=not getattr(args, 'no_face_restore', False),
        )

        # Print header
        print()
        print_colored("=" * 60, Colors.HEADER)
        print_colored("  DRY RUN - No changes will be made", Colors.HEADER)
        print_colored("=" * 60, Colors.HEADER)
        print()

        # Input info
        width, height = result.input_resolution
        file_size = input_path.stat().st_size
        print_colored("Input:", Colors.BOLD)
        print(f"  {input_path.name}")
        print(f"  {width}x{height}, {result.input_fps:.1f}fps, "
              f"{_format_duration_dry(result.input_duration_seconds)}, "
              f"{_format_size(file_size)}")
        print()

        # Output info
        out_w, out_h = result.output_resolution
        print_colored("Output:", Colors.BOLD)
        print(f"  {output_path}")
        print(f"  {out_w}x{out_h}, {result.output_fps:.1f}fps, {output_format.upper()}")
        print()

        # Processing plan
        print_colored("Processing Plan:", Colors.BOLD)
        print(f"  - Frames to extract: {_format_number(result.input_frame_count)}")
        print(f"  - Enhancement: {args.model} ({args.scale}x upscale)")

        if not getattr(args, 'no_face_restore', False) and result.detected_faces_estimate > 0:
            face_model = getattr(args, 'face_model', 'GFPGAN')
            print(f"  - Face restoration: {face_model.upper()} (~{_format_number(result.detected_faces_estimate)} faces)")
        else:
            print(f"  - Face restoration: OFF")

        if args.enable_rife:
            target = args.target_fps or result.output_fps
            print(f"  - Frame interpolation: RIFE -> {target:.0f}fps ({_format_number(result.output_frame_count)} frames)")
        else:
            print(f"  - Frame interpolation: OFF")

        if args.auto_enhance:
            print(f"  - Auto-enhance: ON (defect repair, color correction)")

        if getattr(args, 'colorize', False):
            colorize_model = getattr(args, 'colorize_model', 'ddcolor')
            print(f"  - Colorization: {colorize_model.upper()}")

        if getattr(args, 'tap_denoise', False):
            tap_model = getattr(args, 'tap_model', 'restormer')
            print(f"  - Neural denoising: {tap_model.upper()}")

        print()

        # Estimated resources
        print_colored("Estimated Resources:", Colors.BOLD)

        # Time estimate
        gpu_name = result.hardware.gpu_name if result.hardware.has_gpu else "CPU"
        time_str = result.time_estimate.time_range_str
        print(f"  - Processing time: ~{time_str} ({gpu_name})")

        # Disk space
        temp_gb = result.temp_disk_usage_bytes / (1024**3)
        output_gb = result.output_disk_usage_bytes / (1024**3)
        print(f"  - Temp disk space: ~{temp_gb:.1f} GB")
        print(f"  - Output size: ~{output_gb:.1f} GB")

        # VRAM
        if result.hardware.has_gpu:
            vram_needed = result.hardware.vram_required_mb / 1024
            vram_avail = result.hardware.vram_available_mb / 1024
            status = "OK" if result.hardware.vram_sufficient else "LOW"
            print(f"  - VRAM needed: ~{vram_needed:.1f} GB / {vram_avail:.1f} GB available [{status}]")
        print()

        # Warnings
        if result.warnings:
            print_colored("Warnings:", Colors.WARNING)
            for warning in result.warnings:
                print(f"  - {warning}")
            print()

        # Blocking issues
        if result.blocking_issues:
            print_colored("BLOCKING ISSUES:", Colors.FAIL)
            for issue in result.blocking_issues:
                print(f"  - {issue}")
            print()

        # Suggestions
        print_colored("Tips:", Colors.OKCYAN)
        preset = getattr(args, 'preset', None)
        if preset != 'fast' and result.time_estimate.total_hours > 2:
            print("  - Use --preset fast for quicker results (lower quality)")
        if not result.hardware.vram_sufficient and result.hardware.has_gpu:
            tile = result.hardware.recommended_tile_size or 256
            print(f"  - Use --tile-size {tile} if you run out of VRAM")
        if args.scale == 4 and result.time_estimate.total_hours > 4:
            print("  - Use --scale 2 to reduce processing time by ~75%")
        if not args.enable_rife and result.input_fps < 30:
            print(f"  - Use --enable-rife --target-fps 60 for smoother playback")
        if result.hardware.gpu_count > 1 and not getattr(args, 'multi_gpu', False):
            print(f"  - Use --multi-gpu to utilize all {result.hardware.gpu_count} GPUs")
        print()

        # Final instruction
        if result.can_proceed:
            print_colored("Run without --dry-run to start processing.", Colors.OKGREEN)
        else:
            print_colored("Cannot proceed due to blocking issues above.", Colors.FAIL)

    except Exception as e:
        print_colored(f"\nDry-run analysis failed: {e}", Colors.FAIL)
        print_colored("Run without --dry-run to attempt processing anyway.", Colors.WARNING)


def restore_video(args):
    """Full video restoration workflow with actual implementation."""
    from .config import Config
    from .restorer import VideoRestorer

    output_path = get_output_path(args)
    output_dir = get_output_dir(args)
    output_format = get_output_format(args)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print_colored(f"\n[Output] Will be saved to: {output_path}", Colors.OKCYAN)
    print_colored(f"[Format] {output_format.upper()}", Colors.OKCYAN)

    # Load user profile if specified
    user_profile_settings: Dict[str, Any] = {}
    user_profile_name = getattr(args, 'user_profile', None)
    if user_profile_name:
        try:
            from .utils.profiles import ProfileManager
            profile_manager = ProfileManager()
            profile_data = profile_manager.load_profile_raw(user_profile_name)
            user_profile_settings = profile_data.get("config", {})
            print_colored(f"[Profile] Using saved profile: {user_profile_name}", Colors.OKCYAN)
        except FileNotFoundError:
            print_colored(f"Error: User profile not found: {user_profile_name}", Colors.FAIL)
            print_colored("List available profiles with: framewright profile list", Colors.WARNING)
            sys.exit(1)
        except Exception as e:
            print_colored(f"Error loading profile: {e}", Colors.FAIL)
            sys.exit(1)

    # Determine input source
    if args.input:
        source = args.input
    elif args.url:
        source = args.url
    else:
        print_colored("Error: Please provide --input or --url", Colors.FAIL)
        sys.exit(1)

    # Handle dry-run mode - show analysis without processing
    if getattr(args, 'dry_run', False):
        print_dry_run_output(args, source, output_path, output_format)
        return

    # Validate input path for non-URL sources
    if args.input and not Path(source).exists():
        print_colored(f"Error: Input file not found: {source}", Colors.FAIL)
        sys.exit(1)

    # Helper to get value: CLI arg > user profile > default
    def get_setting(attr_name: str, profile_key: str, default: Any) -> Any:
        """Get setting with priority: explicit CLI arg > profile > default."""
        cli_value = getattr(args, attr_name, None)
        # Check if CLI arg was explicitly provided (not just default)
        # For boolean store_true args, default is False
        if cli_value is not None and cli_value != default:
            return cli_value
        if profile_key in user_profile_settings:
            return user_profile_settings[profile_key]
        return cli_value if cli_value is not None else default

    # Determine model based on scale
    scale = get_setting('scale', 'scale_factor', 2)
    model_name = get_setting('model', 'model_name', 'realesrgan-x4plus')
    if scale == 2 and 'x4' in model_name:
        model_name = 'realesrgan-x2plus'
        print_colored(f"[Info] Using {model_name} for 2x scale", Colors.WARNING)

    try:
        # Create configuration
        work_dir = output_dir / ".framewright_work"
        # Determine model directory
        model_dir = None
        if hasattr(args, 'model_dir') and args.model_dir:
            model_dir = Path(args.model_dir).expanduser()

        # Get settings with profile override support
        quality = get_setting('quality', 'crf', 18)
        enable_rife = get_setting('enable_rife', 'enable_interpolation', False)
        target_fps = get_setting('target_fps', 'target_fps', None)
        rife_model = get_setting('rife_model', 'rife_model', 'rife-v4.6')
        auto_enhance = get_setting('auto_enhance', 'enable_auto_enhance', False)
        scratch_sens = get_setting('scratch_sensitivity', 'scratch_sensitivity', 0.5)
        grain_red = get_setting('grain_reduction', 'grain_reduction', 0.3)
        colorize = get_setting('colorize', 'enable_colorization', False)
        colorize_model = get_setting('colorize_model', 'colorization_model', 'ddcolor')
        remove_wm = get_setting('remove_watermark', 'enable_watermark_removal', False)

        config = Config(
            project_dir=work_dir,
            output_dir=output_dir,
            scale_factor=scale,
            model_name=model_name,
            crf=quality,
            output_format=output_format,
            enable_checkpointing=True,
            enable_validation=True,
            enable_deduplication=getattr(args, 'deduplicate', False),
            deduplication_threshold=getattr(args, 'dedup_threshold', 0.98),
            enable_interpolation=enable_rife,
            target_fps=target_fps,
            rife_model=rife_model,
            enable_auto_enhance=auto_enhance,
            auto_face_restore=not args.no_face_restore if hasattr(args, 'no_face_restore') else True,
            auto_defect_repair=not args.no_defect_repair if hasattr(args, 'no_defect_repair') else True,
            scratch_sensitivity=scratch_sens,
            grain_reduction=grain_red,
            model_dir=model_dir if model_dir else Path.home() / ".framewright" / "models",
            enable_colorization=colorize,
            colorization_model=colorize_model,
            enable_watermark_removal=remove_wm,
            watermark_auto_detect=getattr(args, 'watermark_auto_detect', False),
            gpu_id=getattr(args, 'gpu', None),
            enable_multi_gpu=getattr(args, 'multi_gpu', False),
            # Ultimate preset features
            enable_tap_denoise=getattr(args, 'tap_denoise', False),
            tap_model=getattr(args, 'tap_model', 'restormer'),
            tap_strength=getattr(args, 'tap_strength', 1.0),
            tap_preserve_grain=getattr(args, 'tap_preserve_grain', False),
            sr_model='diffusion' if getattr(args, 'diffusion_sr', False) else 'realesrgan',
            diffusion_steps=getattr(args, 'diffusion_steps', 20),
            diffusion_guidance=getattr(args, 'diffusion_guidance', 7.5),
            face_model=getattr(args, 'face_model', 'gfpgan'),
            aesrgan_strength=getattr(args, 'aesrgan_strength', 0.8),
            enable_qp_artifact_removal=getattr(args, 'qp_artifact_removal', False),
            qp_strength=getattr(args, 'qp_strength', 1.0),
            colorization_reference_images=[Path(p) for p in (getattr(args, 'colorize_reference', None) or [])],
            colorization_temporal_fusion=getattr(args, 'colorize_temporal_fusion', True),
            enable_reference_enhance=getattr(args, 'reference_enhance', False),
            reference_images_dir=Path(args.reference_dir) if getattr(args, 'reference_dir', None) else None,
            reference_strength=getattr(args, 'reference_strength', 0.35),
            reference_guidance_scale=getattr(args, 'reference_guidance', 7.5),
            reference_ip_adapter_scale=getattr(args, 'reference_ip_scale', 0.6),
            enable_frame_generation=getattr(args, 'generate_frames', False),
            frame_gen_model=getattr(args, 'frame_gen_model', 'interpolate_blend'),
            max_gap_frames=getattr(args, 'max_gap_frames', 10),
            temporal_method=getattr(args, 'temporal_method', 'optical_flow'),
            cross_attention_window=getattr(args, 'cross_attention_window', 7),
            temporal_blend_strength=getattr(args, 'temporal_blend_strength', 0.8),
        )

        # Create restorer with progress callback
        def progress_callback(stage: str, progress: float):
            pass  # Progress handled by tqdm in restore_video

        restorer = VideoRestorer(config, progress_callback=progress_callback)

        # Preview callback if requested
        preview_callback = None
        if args.preview:
            def preview_callback(preview_info: Dict[str, Any]) -> bool:
                print_colored("\n" + "="*60, Colors.HEADER)
                print_colored("PREVIEW - Review before final assembly", Colors.HEADER)
                print_colored("="*60, Colors.HEADER)
                print(f"  Frames: {preview_info['total_frames']}")
                print(f"  Resolution: {preview_info['resolution']}")
                print(f"  Frame Rate: {preview_info['output_fps']}fps")
                print(f"  Duration: {preview_info['estimated_duration']}")
                print(f"  RIFE applied: {preview_info['interpolation_applied']}")
                print(f"\n  Sample frames saved in: {preview_info['frames_dir']}")
                print_colored("="*60, Colors.HEADER)

                response = input("\nProceed with video assembly? [Y/n]: ").strip().lower()
                return response != 'n'

        # Run the restoration pipeline
        print_colored("\n🎬 Starting video restoration pipeline...\n", Colors.OKBLUE)

        result_path = restorer.restore_video(
            source=source,
            output_path=output_path,
            cleanup=True,
            resume=True,
            enable_rife=args.enable_rife,
            target_fps=args.target_fps,
            enable_auto_enhance=args.auto_enhance,
            preview_callback=preview_callback,
        )

        print_colored(f"\n✓ Video restoration complete: {result_path}", Colors.OKGREEN)

        # Show statistics if available
        error_report = restorer.get_error_report()
        if error_report.total_operations > 0:
            print_colored(f"  {error_report.summary()}", Colors.OKCYAN)

        vram_stats = restorer.get_vram_statistics()
        if vram_stats:
            print_colored(f"  Peak VRAM: {vram_stats.get('peak_mb', 0):.0f}MB", Colors.OKCYAN)

    except Exception as e:
        print_colored(f"\n✗ Restoration failed: {e}", Colors.FAIL)

        # Try fallback with simpler settings
        print_colored("\nTrying fallback with simpler settings...", Colors.WARNING)
        try:
            # Use basic ffmpeg-based approach
            from .utils.ffmpeg import probe_video, extract_frames_to_dir, reassemble_from_frames

            temp_dir = Path(tempfile.mkdtemp(prefix="framewright_fallback_"))
            frames_dir = temp_dir / "frames"

            print_colored("[1/3] Extracting frames...", Colors.OKBLUE)
            input_path = Path(source) if not source.startswith('http') else None
            if input_path and input_path.exists():
                frame_count = extract_frames_to_dir(input_path, frames_dir)
                print_colored(f"      Extracted {frame_count} frames", Colors.OKCYAN)

                print_colored("[2/3] Processing (basic mode)...", Colors.OKBLUE)
                # In fallback, just copy frames without enhancement
                enhanced_dir = temp_dir / "enhanced"
                shutil.copytree(frames_dir, enhanced_dir)

                print_colored("[3/3] Reassembling video...", Colors.OKBLUE)
                metadata = probe_video(input_path)
                reassemble_from_frames(
                    frames_dir=enhanced_dir,
                    output_path=output_path,
                    framerate=metadata.get('framerate', 24.0),
                    crf=args.quality,
                    audio_source=input_path,
                )

                print_colored(f"\n✓ Video saved (basic mode): {output_path}", Colors.OKGREEN)
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                raise Exception("Cannot use fallback for URL sources")

        except Exception as fallback_error:
            print_colored(f"\n✗ Fallback also failed: {fallback_error}", Colors.FAIL)
            sys.exit(1)

    # Generate improvements report if requested
    if hasattr(args, 'generate_report') and args.generate_report:
        print_colored("\nGenerating improvements report...", Colors.OKBLUE)
        report_path = generate_improvements_report(output_dir)
        print_colored(f"✓ Report saved: {report_path}", Colors.OKGREEN)


def extract_frames(args):
    """Extract frames from video using FFmpeg."""
    from .utils.ffmpeg import extract_frames_to_dir, probe_video

    print_colored(f"\nExtracting frames from: {args.input}", Colors.OKBLUE)
    input_path = validate_input(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get video info
        metadata = probe_video(input_path)
        print_colored(f"  Resolution: {metadata.get('width')}x{metadata.get('height')}", Colors.OKCYAN)
        print_colored(f"  Frame Rate: {metadata.get('framerate')} fps", Colors.OKCYAN)
        print_colored(f"  Duration: {metadata.get('duration'):.1f} seconds", Colors.OKCYAN)

        # Extract frames
        frame_count = extract_frames_to_dir(input_path, output_dir)

        print_colored(f"\n✓ Extracted {frame_count} frames to: {output_dir}", Colors.OKGREEN)

    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def enhance_frames(args):
    """Enhance extracted frames — dispatches to the appropriate SR backend."""
    print_colored(f"\nEnhancing frames with {args.model} model (scale: {args.scale}x)", Colors.OKBLUE)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print_colored(f"Error: Input directory not found: {input_dir}", Colors.FAIL)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scale = validate_scale(args.scale)

    # Get frame list
    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        frames = sorted(input_dir.glob("*.jpg"))

    if not frames:
        print_colored("Error: No PNG or JPG frames found in input directory", Colors.FAIL)
        sys.exit(1)

    print_colored(f"  Found {len(frames)} frames to enhance", Colors.OKCYAN)

    model = args.model.lower()
    if model == "hat":
        _enhance_with_hat(args, input_dir, output_dir, scale, len(frames))
    elif model == "diffusion":
        _enhance_with_diffusion(args, input_dir, output_dir, scale, len(frames))
    elif model == "ensemble":
        _enhance_with_ensemble(args, input_dir, output_dir, scale, len(frames))
    else:
        _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)


def _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames):
    """Enhance frames using Real-ESRGAN."""
    import cv2

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model_dir = Path.home() / ".framewright" / "models"

        model_name = args.model
        if model_name in ("realesrgan-x4plus", "realesrgan"):
            model_path = model_dir / "RealESRGAN_x4plus.pth"
            model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif "anime" in model_name:
            model_path = model_dir / "realesr-animevideov3.pth"
            model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        else:
            model_path = model_dir / "realesr-general-x4v3.pth"
            model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4

        if not model_path.exists():
            for fallback in ["RealESRGAN_x4plus.pth", "realesr-general-x4v3.pth"]:
                fb_path = model_dir / fallback
                if fb_path.exists():
                    model_path = fb_path
                    print_colored(f"  Using fallback model: {fallback}", Colors.WARNING)
                    break

        if not model_path.exists():
            print_colored(f"Error: Model not found at {model_path}", Colors.FAIL)
            print_colored("Download models first via the dashboard or run:", Colors.WARNING)
            print_colored("  framewright restore --input video.mp4 --output out.mp4", Colors.WARNING)
            sys.exit(1)

        print_colored(f"  Loading model: {model_path.name}", Colors.OKCYAN)

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model_arch,
            tile=512,
            tile_pad=10,
            pre_pad=10,
            half=True,
        )
        print_colored(f"  Model loaded on GPU, tile size 512", Colors.OKCYAN)

    except ImportError as e:
        print_colored(f"Error: Required package not installed: {e}", Colors.FAIL)
        print_colored("Install with: pip install realesrgan basicsr", Colors.WARNING)
        sys.exit(1)

    failed_frames = []

    with tqdm(total=len(frames), desc="Enhancing frames", ncols=100) as pbar:
        for frame in frames:
            output_frame = output_dir / frame.name
            try:
                img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
                if img is None:
                    failed_frames.append(frame.name)
                    pbar.update(1)
                    continue
                output, _ = upsampler.enhance(img, outscale=scale)
                cv2.imwrite(str(output_frame), output)
            except Exception:
                failed_frames.append(frame.name)
            pbar.update(1)

    enhanced_count = len(frames) - len(failed_frames)
    print_colored(f"\n✓ Enhanced {enhanced_count}/{len(frames)} frames to: {output_dir}", Colors.OKGREEN)
    if failed_frames:
        print_colored(f"  Warning: {len(failed_frames)} frames failed to enhance", Colors.WARNING)


def _enhance_with_hat(args, input_dir, output_dir, scale, frame_count):
    """Enhance frames using HAT (Hybrid Attention Transformer)."""
    try:
        from .processors.hat_upscaler import HATUpscaler, HATConfig, HATModelSize

        size_map = {"small": HATModelSize.SMALL, "base": HATModelSize.BASE, "large": HATModelSize.LARGE}
        hat_size = size_map.get(getattr(args, 'hat_size', 'large'), HATModelSize.LARGE)

        config = HATConfig(scale=scale, model_size=hat_size)
        upscaler = HATUpscaler(config)

        if not upscaler.is_available():
            print_colored("  HAT model not found, attempting download...", Colors.WARNING)
            if not upscaler.download_model():
                print_colored("  HAT unavailable, falling back to Real-ESRGAN", Colors.WARNING)
                frames = sorted(input_dir.glob("*.png")) or sorted(input_dir.glob("*.jpg"))
                _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)
                return

        print_colored(f"  Using HAT-{hat_size.value} ({frame_count} frames)", Colors.OKCYAN)

        import time
        start = time.time()

        def progress_cb(pct):
            pass  # tqdm handles display via upscale_frames internals

        result = upscaler.upscale_frames(input_dir, output_dir, progress_callback=progress_cb)
        elapsed = time.time() - start
        fps = result.frames_processed / elapsed if elapsed > 0 else 0

        print_colored(
            f"\n✓ HAT enhanced {result.frames_processed}/{frame_count} frames "
            f"({fps:.1f} fps, {elapsed:.0f}s)", Colors.OKGREEN
        )
        if result.frames_failed:
            print_colored(f"  Warning: {result.frames_failed} frames failed", Colors.WARNING)

    except ImportError as e:
        print_colored(f"  HAT import failed ({e}), falling back to Real-ESRGAN", Colors.WARNING)
        frames = sorted(input_dir.glob("*.png")) or sorted(input_dir.glob("*.jpg"))
        _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)


def _enhance_with_diffusion(args, input_dir, output_dir, scale, frame_count):
    """Enhance frames using Diffusion SR."""
    try:
        from .processors.diffusion_sr import DiffusionSRProcessor, DiffusionSRConfig, DiffusionModel

        model_map = {
            "upscale_a_video": DiffusionModel.UPSCALE_A_VIDEO,
            "stable_sr": DiffusionModel.STABLE_SR,
            "resshift": DiffusionModel.RESSHIFT,
        }
        diff_model = model_map.get(
            getattr(args, 'diffusion_model', 'upscale_a_video'),
            DiffusionModel.UPSCALE_A_VIDEO,
        )
        steps = getattr(args, 'diffusion_steps', 20)

        config = DiffusionSRConfig(
            model=diff_model,
            scale_factor=scale,
            num_inference_steps=steps,
        )
        processor = DiffusionSRProcessor(config)

        if not processor.is_available():
            print_colored("  Diffusion SR unavailable, falling back to Real-ESRGAN", Colors.WARNING)
            frames = sorted(input_dir.glob("*.png")) or sorted(input_dir.glob("*.jpg"))
            _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)
            return

        print_colored(
            f"  Using Diffusion SR ({diff_model.value}, {steps} steps, {frame_count} frames)",
            Colors.OKCYAN,
        )

        import time
        start = time.time()
        result = processor.enhance_video(input_dir, output_dir)
        elapsed = time.time() - start
        fps = result.frames_processed / elapsed if elapsed > 0 else 0

        print_colored(
            f"\n✓ Diffusion SR enhanced {result.frames_processed}/{frame_count} frames "
            f"({fps:.2f} fps, {elapsed:.0f}s)", Colors.OKGREEN
        )
        if result.frames_failed:
            print_colored(f"  Warning: {result.frames_failed} frames failed", Colors.WARNING)

    except ImportError as e:
        print_colored(f"  Diffusion SR import failed ({e}), falling back to Real-ESRGAN", Colors.WARNING)
        frames = sorted(input_dir.glob("*.png")) or sorted(input_dir.glob("*.jpg"))
        _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)


def _enhance_with_ensemble(args, input_dir, output_dir, scale, frame_count):
    """Enhance frames using Ensemble SR (multiple models combined)."""
    try:
        from .processors.ensemble_sr import EnsembleSR, EnsembleConfig, VotingMethod

        method_map = {
            "weighted": VotingMethod.WEIGHTED,
            "max_quality": VotingMethod.MAX_QUALITY,
            "per_region": VotingMethod.PER_REGION,
            "adaptive": VotingMethod.ADAPTIVE,
            "median": VotingMethod.MEDIAN,
        }
        voting = method_map.get(
            getattr(args, 'ensemble_method', 'weighted'),
            VotingMethod.WEIGHTED,
        )
        model_list = getattr(args, 'ensemble_models', 'hat,realesrgan').split(',')

        config = EnsembleConfig(models=model_list, voting_method=voting)
        ensemble = EnsembleSR(config)

        if not ensemble.is_available():
            print_colored("  Ensemble SR unavailable (need >=2 models), falling back to Real-ESRGAN", Colors.WARNING)
            frames = sorted(input_dir.glob("*.png")) or sorted(input_dir.glob("*.jpg"))
            _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)
            return

        print_colored(
            f"  Using Ensemble SR ({', '.join(model_list)}, method={voting.value}, {frame_count} frames)",
            Colors.OKCYAN,
        )

        import time
        start = time.time()
        result = ensemble.upscale_frames(input_dir, output_dir)
        elapsed = time.time() - start
        fps = result.frames_processed / elapsed if elapsed > 0 else 0

        print_colored(
            f"\n✓ Ensemble SR enhanced {result.frames_processed}/{frame_count} frames "
            f"({fps:.2f} fps, {elapsed:.0f}s)", Colors.OKGREEN
        )
        if result.frames_failed:
            print_colored(f"  Warning: {result.frames_failed} frames failed", Colors.WARNING)

    except ImportError as e:
        print_colored(f"  Ensemble SR import failed ({e}), falling back to Real-ESRGAN", Colors.WARNING)
        frames = sorted(input_dir.glob("*.png")) or sorted(input_dir.glob("*.jpg"))
        _enhance_with_realesrgan(args, input_dir, output_dir, scale, frames)


def reassemble_video(args):
    """Reassemble video from enhanced frames."""
    from .utils.ffmpeg import reassemble_from_frames, probe_video

    print_colored("\nReassembling video from frames...", Colors.OKBLUE)
    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output)

    if not frames_dir.exists():
        print_colored(f"Error: Frames directory not found: {frames_dir}", Colors.FAIL)
        sys.exit(1)

    # Count frames
    frames = list(frames_dir.glob("*.png"))
    if not frames:
        print_colored("Error: No PNG frames found in directory", Colors.FAIL)
        sys.exit(1)

    print_colored(f"  Found {len(frames)} frames", Colors.OKCYAN)

    # Get framerate from audio source if provided
    framerate = 24.0  # Default
    audio_source = None

    if args.audio:
        audio_path = Path(args.audio)
        if audio_path.exists():
            audio_source = audio_path
            try:
                metadata = probe_video(audio_path)
                framerate = metadata.get('framerate', 24.0)
                print_colored(f"  Using framerate from audio source: {framerate} fps", Colors.OKCYAN)
            except Exception:
                pass

    # Check for framerate argument
    if hasattr(args, 'fps') and args.fps:
        framerate = args.fps
        print_colored(f"  Using specified framerate: {framerate} fps", Colors.OKCYAN)

    try:
        reassemble_from_frames(
            frames_dir=frames_dir,
            output_path=output_path,
            framerate=framerate,
            crf=args.quality,
            audio_source=audio_source,
        )

        print_colored(f"\n✓ Video reassembled: {output_path}", Colors.OKGREEN)

    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def enhance_audio(args):
    """Enhance audio track using audio processor."""
    print_colored("\nEnhancing audio track...", Colors.OKBLUE)
    input_path = validate_input(args.input)
    output_path = Path(args.output)

    try:
        from .processors.audio import AudioProcessor

        processor = AudioProcessor()

        # Extract audio first if input is a video file
        temp_audio = None
        if str(input_path).lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            import tempfile
            temp_audio = Path(tempfile.mktemp(suffix='.wav'))
            print_colored("  Extracting audio from video...", Colors.OKCYAN)
            processor.extract(str(input_path), str(temp_audio))
            audio_input = temp_audio
        else:
            audio_input = input_path

        with tqdm(total=100, desc="Processing audio", ncols=100) as pbar:
            pbar.update(30)
            processor.enhance(
                audio_path=str(audio_input),
                output_path=str(output_path),
            )
            pbar.update(70)

        # Cleanup temp file
        if temp_audio and temp_audio.exists():
            temp_audio.unlink()

        print_colored(f"\n✓ Audio enhanced: {output_path}", Colors.OKGREEN)

    except ImportError:
        # Fallback to basic ffmpeg audio normalization
        print_colored("  Using basic audio normalization...", Colors.WARNING)
        import subprocess

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
            '-y',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print_colored(f"\n✓ Audio normalized: {output_path}", Colors.OKGREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"Error: {e.stderr.decode() if e.stderr else e}", Colors.FAIL)
            sys.exit(1)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def interpolate_video(args):
    """Interpolate video frames using RIFE to increase frame rate."""
    print_colored(f"\nInterpolating video to {args.target_fps}fps using {args.model}...", Colors.OKBLUE)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print_colored(f"Error: Input not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .processors.interpolation import FrameInterpolator
        from .utils.ffmpeg import probe_video, extract_frames_to_dir, reassemble_from_frames

        interpolator = FrameInterpolator(model=args.model)

        # Determine if input is video or frame directory
        if input_path.is_dir():
            frames_dir = input_path
            source_fps = args.source_fps
            if source_fps is None:
                print_colored("Error: --source-fps required when input is a directory", Colors.FAIL)
                sys.exit(1)
        else:
            print_colored("[1/4] Analyzing video...", Colors.OKBLUE)
            metadata = probe_video(input_path)
            source_fps = args.source_fps or metadata.get('framerate', 24.0)
            print_colored(f"      Detected: {metadata.get('width')}x{metadata.get('height')} @ {source_fps}fps", Colors.OKCYAN)

            print_colored("[2/4] Extracting frames...", Colors.OKBLUE)
            frames_dir = Path(tempfile.mkdtemp(prefix="framewright_interp_"))
            extract_frames_to_dir(input_path, frames_dir)

        # Perform interpolation
        print_colored(f"[3/4] Interpolating {source_fps}fps -> {args.target_fps}fps...", Colors.OKBLUE)

        if args.frames_only:
            output_frames_dir = output_path
        else:
            output_frames_dir = Path(tempfile.mkdtemp(prefix="framewright_interp_out_"))

        output_frames_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=100, desc="RIFE interpolation", ncols=100) as pbar:
            def progress_cb(p):
                pbar.n = int(p * 100)
                pbar.refresh()

            result_dir, actual_fps = interpolator.interpolate_to_fps(
                input_dir=frames_dir,
                output_dir=output_frames_dir,
                source_fps=source_fps,
                target_fps=int(args.target_fps),
                progress_callback=progress_cb
            )

        frame_count = len(list(result_dir.glob("*.png")))
        print_colored(f"      Generated {frame_count} frames at {actual_fps}fps", Colors.OKCYAN)

        if args.frames_only:
            print_colored(f"\n✓ Interpolated frames saved to: {output_path}", Colors.OKGREEN)
        else:
            print_colored("[4/4] Reassembling video...", Colors.OKBLUE)
            reassemble_from_frames(
                frames_dir=result_dir,
                output_path=output_path,
                framerate=actual_fps,
                crf=args.quality,
                audio_source=input_path if not input_path.is_dir() else None
            )
            print_colored(f"\n✓ Interpolated video saved to: {output_path}", Colors.OKGREEN)

            # Cleanup temp directories
            if not input_path.is_dir():
                shutil.rmtree(frames_dir, ignore_errors=True)
            shutil.rmtree(output_frames_dir, ignore_errors=True)

    except ImportError as e:
        print_colored(f"Error: Missing dependency: {e}", Colors.FAIL)
        print_colored("Install with: pip install framewright[rife]", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def analyze_video(args):
    """Analyze video and show recommended restoration settings."""
    print_colored(f"\nAnalyzing video for restoration: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .processors.analyzer import FrameAnalyzer

        print_colored("[1/2] Extracting sample frames...", Colors.OKBLUE)

        analyzer = FrameAnalyzer(
            sample_rate=100,
            max_samples=50,
            enable_face_detection=True,
        )

        print_colored("[2/2] Analyzing content and degradation...", Colors.OKBLUE)

        with tqdm(total=100, desc="Analysis", ncols=100) as pbar:
            analysis = analyzer.analyze_video(input_path)
            pbar.update(100)

        # Display results
        print_colored("\n┌─────────────────────────────────────────────────────┐", Colors.HEADER)
        print_colored("│              Video Analysis Results                 │", Colors.HEADER)
        print_colored("└─────────────────────────────────────────────────────┘", Colors.HEADER)

        print_colored("\n  Video Info:", Colors.BOLD)
        print(f"     Resolution:    {analysis.resolution[0]}x{analysis.resolution[1]}")
        print(f"     Frame Rate:    {analysis.source_fps:.2f} fps")
        print(f"     Duration:      {analysis.duration:.1f} seconds")
        print(f"     Total Frames:  {analysis.total_frames}")

        print_colored("\n  Content Detection:", Colors.BOLD)
        print(f"     Primary Type:  {analysis.primary_content.name.replace('_', ' ').title()}")
        print(f"     Faces Found:   {analysis.face_frame_ratio * 100:.1f}% of frames")
        print(f"     Avg Brightness: {analysis.avg_brightness:.1f}")
        print(f"     Avg Noise:      {analysis.avg_noise:.2f}")

        print_colored("\n  Degradation Analysis:", Colors.BOLD)
        print(f"     Severity:      {analysis.degradation_severity.upper()}")
        degradation_names = [d.name.replace('_', ' ').title() for d in analysis.degradation_types]
        print(f"     Detected:      {', '.join(degradation_names) or 'None'}")

        print_colored("\n  Recommended Settings:", Colors.OKCYAN)
        print(f"     Scale Factor:  {analysis.recommended_scale}x")
        print(f"     Model:         {analysis.recommended_model}")
        print(f"     Denoise:       {analysis.recommended_denoise:.1f}")
        if analysis.enable_face_restoration:
            print(f"     Face Restore:  Yes (faces detected)")
        if analysis.enable_scratch_removal:
            print(f"     Defect Repair: Yes (degradation detected)")
        if analysis.recommended_target_fps:
            print(f"     Target FPS:    {analysis.recommended_target_fps} (for RIFE)")

        print_colored("\n  Suggested Command:", Colors.OKGREEN)
        cmd_parts = [
            "framewright restore",
            f"--input {args.input}",
            "--output restored.mp4",
            f"--scale {analysis.recommended_scale}",
            "--auto-enhance",
        ]
        if analysis.recommended_target_fps:
            cmd_parts.append(f"--enable-rife --target-fps {int(analysis.recommended_target_fps)}")

        print(f"     {' '.join(cmd_parts)}")
        print()

        # Save analysis to JSON if requested
        if args.output:
            import json
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(analysis.to_dict(), f, indent=2)
            print_colored(f"✓ Analysis saved to: {output_path}", Colors.OKGREEN)

    except ImportError as e:
        print_colored(f"Error: Missing dependency: {e}", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error analyzing video: {e}", Colors.FAIL)
        sys.exit(1)


def batch_process(args):
    """Process multiple videos in batch mode."""
    from .config import Config
    from .restorer import VideoRestorer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print_colored(f"Error: Input directory not found: {input_dir}", Colors.FAIL)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
    video_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print_colored(f"No video files found in {input_dir}", Colors.WARNING)
        sys.exit(0)

    print_colored(f"\n📁 Found {len(video_files)} videos to process", Colors.OKBLUE)

    # Load preset config if specified
    work_dir = output_dir / ".framewright_work"

    if args.preset:
        config = Config.from_preset(args.preset, project_dir=work_dir)
        print_colored(f"  Using preset: {args.preset}", Colors.OKCYAN)
    elif args.config:
        config = Config.load_preset_file(Path(args.config))
        print_colored(f"  Using config file: {args.config}", Colors.OKCYAN)
    else:
        config = Config(
            project_dir=work_dir,
            output_dir=output_dir,
            scale_factor=args.scale,
            crf=args.quality,
            output_format=args.format,
        )

    # Process each video
    results = {"success": [], "failed": []}

    for i, video_file in enumerate(video_files, 1):
        output_name = f"{video_file.stem}_restored.{args.format}"
        output_path = output_dir / output_name

        print_colored(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}", Colors.HEADER)

        try:
            restorer = VideoRestorer(config)
            result_path = restorer.restore_video(
                source=str(video_file),
                output_path=output_path,
                cleanup=True,
                resume=True,
            )
            results["success"].append(video_file.name)
            print_colored(f"  ✓ Saved: {output_path}", Colors.OKGREEN)

        except Exception as e:
            results["failed"].append((video_file.name, str(e)))
            print_colored(f"  ✗ Failed: {e}", Colors.FAIL)

            if not args.continue_on_error:
                print_colored("\nStopping batch processing due to error.", Colors.WARNING)
                break

    # Summary
    print_colored("\n" + "=" * 50, Colors.HEADER)
    print_colored("Batch Processing Complete", Colors.HEADER)
    print_colored("=" * 50, Colors.HEADER)
    print_colored(f"  Successful: {len(results['success'])}", Colors.OKGREEN)
    print_colored(f"  Failed: {len(results['failed'])}", Colors.FAIL if results['failed'] else Colors.OKGREEN)

    for name, error in results['failed']:
        print_colored(f"    - {name}: {error[:50]}...", Colors.WARNING)


def watch_folder(args):
    """Watch folder for new videos and process automatically."""
    import time
    from .config import Config
    from .restorer import VideoRestorer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    if not input_dir.exists():
        print_colored(f"Error: Input directory not found: {input_dir}", Colors.FAIL)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir:
        processed_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}
    processed_files = set()
    work_dir = output_dir / ".framewright_work"

    # Load config
    if args.preset:
        config = Config.from_preset(args.preset, project_dir=work_dir)
    else:
        config = Config(
            project_dir=work_dir,
            output_dir=output_dir,
            scale_factor=args.scale,
            crf=args.quality,
            output_format=args.format,
        )

    print_colored(f"\n👁 Watching folder: {input_dir}", Colors.OKBLUE)
    print_colored(f"   Output folder: {output_dir}", Colors.OKCYAN)
    print_colored(f"   Checking every {args.interval} seconds", Colors.OKCYAN)
    print_colored(f"\nPress Ctrl+C to stop.\n", Colors.WARNING)

    try:
        while True:
            # Scan for new videos
            for video_file in input_dir.iterdir():
                if not video_file.is_file():
                    continue
                if video_file.suffix.lower() not in video_extensions:
                    continue
                if video_file.name in processed_files:
                    continue

                # Skip files that are still being written (size changing)
                try:
                    size1 = video_file.stat().st_size
                    time.sleep(1)
                    size2 = video_file.stat().st_size
                    if size1 != size2:
                        continue
                except OSError:
                    continue

                print_colored(f"\n📹 New video detected: {video_file.name}", Colors.HEADER)

                output_name = f"{video_file.stem}_restored.{args.format}"
                output_path = output_dir / output_name

                try:
                    restorer = VideoRestorer(config)
                    result_path = restorer.restore_video(
                        source=str(video_file),
                        output_path=output_path,
                        cleanup=True,
                        resume=True,
                    )
                    print_colored(f"  ✓ Saved: {output_path}", Colors.OKGREEN)

                    # Move to processed folder if specified
                    if processed_dir:
                        dest = processed_dir / video_file.name
                        shutil.move(str(video_file), str(dest))
                        print_colored(f"  📦 Moved original to: {dest}", Colors.OKCYAN)

                except Exception as e:
                    print_colored(f"  ✗ Failed: {e}", Colors.FAIL)

                processed_files.add(video_file.name)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print_colored("\n\n✓ Watch mode stopped.", Colors.OKGREEN)


def watch_folder_enhanced(args):
    """Enhanced watch folder with webhooks, retry logic, and watchdog support."""
    from .watch import WatchMode, WatchConfig

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print_colored(f"Error: Input directory not found: {input_dir}", Colors.FAIL)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse file patterns
    file_patterns = None
    if args.file_patterns:
        file_patterns = [p.strip() for p in args.file_patterns.split(',')]

    # Use profile or preset (profile takes precedence)
    profile = args.profile if args.profile else (args.preset if args.preset else 'quality')

    # Build watch config
    config = WatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        profile=profile,
        file_patterns=file_patterns,
        on_complete_webhook=args.on_complete if args.on_complete and args.on_complete.startswith(('http://', 'https://')) else None,
        on_complete_command=args.on_complete if args.on_complete and not args.on_complete.startswith(('http://', 'https://')) else None,
        on_error_webhook=args.on_error if args.on_error and args.on_error.startswith(('http://', 'https://')) else None,
        on_error_command=args.on_error if args.on_error and not args.on_error.startswith(('http://', 'https://')) else None,
        retry_count=args.retry_count,
        poll_interval_seconds=float(args.interval),
        move_processed=args.processed_dir is not None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        delete_processed=args.delete_processed,
        scale_factor=args.scale,
        crf=args.quality,
        output_format=args.format,
    )

    print_colored(f"\n[ENHANCED WATCH MODE]", Colors.HEADER)

    watcher = WatchMode(config)

    try:
        watcher.start(blocking=True)
    except KeyboardInterrupt:
        watcher.stop()
        print_colored("\n\nWatch mode stopped.", Colors.OKGREEN)

        # Print summary
        status = watcher.get_status()
        if status['processed_files'] > 0:
            print_colored(f"\nSummary:", Colors.BOLD)
            print(f"  Processed: {status['processed_files']}")
            print(f"  Completed: {status['completed']}")
            print(f"  Failed: {status['failed']}")


def compare_videos(args):
    """Compare original and restored videos side by side."""
    import subprocess
    import json
    from datetime import datetime

    original = Path(args.original)
    restored = Path(args.restored)

    if not original.exists():
        print_colored(f"Error: Original video not found: {original}", Colors.FAIL)
        sys.exit(1)

    if not restored.exists():
        print_colored(f"Error: Restored video not found: {restored}", Colors.FAIL)
        sys.exit(1)

    print_colored(f"\n🔍 Comparing videos...", Colors.OKBLUE)
    print_colored(f"  Original: {original}", Colors.OKCYAN)
    print_colored(f"  Restored: {restored}", Colors.OKCYAN)

    try:
        from .utils.ffmpeg import probe_video

        # Get metadata for both videos
        orig_meta = probe_video(original)
        rest_meta = probe_video(restored)

        # Extract sample frames for comparison
        temp_dir = Path(tempfile.mkdtemp(prefix="framewright_compare_"))
        orig_frames_dir = temp_dir / "original"
        rest_frames_dir = temp_dir / "restored"
        orig_frames_dir.mkdir()
        rest_frames_dir.mkdir()

        # Extract 5 sample frames from each
        sample_count = args.samples
        duration = orig_meta.get('duration', 10)
        interval = duration / (sample_count + 1)

        print_colored(f"\n  Extracting {sample_count} sample frames...", Colors.OKBLUE)

        for i in range(sample_count):
            timestamp = interval * (i + 1)

            # Original
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', str(original),
                '-frames:v', '1', str(orig_frames_dir / f'frame_{i:02d}.png')
            ], capture_output=True)

            # Restored
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(timestamp), '-i', str(restored),
                '-frames:v', '1', str(rest_frames_dir / f'frame_{i:02d}.png')
            ], capture_output=True)

        # Calculate PSNR and SSIM if possible
        metrics = {"psnr": [], "ssim": []}

        for i in range(sample_count):
            orig_frame = orig_frames_dir / f'frame_{i:02d}.png'
            rest_frame = rest_frames_dir / f'frame_{i:02d}.png'

            if orig_frame.exists() and rest_frame.exists():
                # Use ffmpeg to calculate PSNR/SSIM
                cmd = [
                    'ffmpeg', '-i', str(orig_frame), '-i', str(rest_frame),
                    '-lavfi', 'psnr=stats_file=-;[0:v][1:v]ssim=stats_file=-',
                    '-f', 'null', '-'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                # Parse output for metrics (simplified)
                stderr = result.stderr
                if 'psnr' in stderr.lower():
                    import re
                    psnr_match = re.search(r'average:(\d+\.?\d*)', stderr)
                    if psnr_match:
                        metrics["psnr"].append(float(psnr_match.group(1)))
                    ssim_match = re.search(r'All:(\d+\.?\d*)', stderr)
                    if ssim_match:
                        metrics["ssim"].append(float(ssim_match.group(1)))

        # Display comparison results
        print_colored("\n┌─────────────────────────────────────────────────────┐", Colors.HEADER)
        print_colored("│              Video Comparison Results               │", Colors.HEADER)
        print_colored("└─────────────────────────────────────────────────────┘", Colors.HEADER)

        print_colored("\n  📊 Resolution:", Colors.BOLD)
        print(f"     Original:  {orig_meta.get('width')}x{orig_meta.get('height')}")
        print(f"     Restored:  {rest_meta.get('width')}x{rest_meta.get('height')}")

        if rest_meta.get('width') and orig_meta.get('width'):
            scale = rest_meta['width'] / orig_meta['width']
            print(f"     Scale:     {scale:.1f}x upscale")

        print_colored("\n  🎬 Frame Rate:", Colors.BOLD)
        print(f"     Original:  {orig_meta.get('framerate')} fps")
        print(f"     Restored:  {rest_meta.get('framerate')} fps")

        print_colored("\n  ⏱ Duration:", Colors.BOLD)
        print(f"     Original:  {orig_meta.get('duration', 0):.2f} seconds")
        print(f"     Restored:  {rest_meta.get('duration', 0):.2f} seconds")

        if metrics["psnr"]:
            avg_psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
            print_colored("\n  📈 Quality Metrics:", Colors.BOLD)
            print(f"     Avg PSNR:  {avg_psnr:.2f} dB (higher is better)")
        if metrics["ssim"]:
            avg_ssim = sum(metrics["ssim"]) / len(metrics["ssim"])
            print(f"     Avg SSIM:  {avg_ssim:.4f} (closer to 1 is better)")

        # Generate HTML report if requested
        if args.output:
            output_path = Path(args.output)
            report_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>FrameWright Video Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #4ecca3; }}
        .comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
        .frame {{ flex: 1; }}
        .frame img {{ max-width: 100%; border: 2px solid #4ecca3; }}
        .frame h3 {{ color: #4ecca3; }}
        .metrics {{ background: #16213e; padding: 20px; border-radius: 10px; }}
        .metrics h2 {{ color: #4ecca3; margin-top: 0; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #333; }}
    </style>
</head>
<body>
    <h1>🎬 FrameWright Video Comparison</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="metrics">
        <h2>📊 Comparison Summary</h2>
        <div class="metric"><span>Original Resolution</span><span>{orig_meta.get('width')}x{orig_meta.get('height')}</span></div>
        <div class="metric"><span>Restored Resolution</span><span>{rest_meta.get('width')}x{rest_meta.get('height')}</span></div>
        <div class="metric"><span>Original FPS</span><span>{orig_meta.get('framerate')}</span></div>
        <div class="metric"><span>Restored FPS</span><span>{rest_meta.get('framerate')}</span></div>
    </div>

    <h2>🖼 Frame Comparisons</h2>
'''
            for i in range(sample_count):
                orig_frame = orig_frames_dir / f'frame_{i:02d}.png'
                rest_frame = rest_frames_dir / f'frame_{i:02d}.png'
                if orig_frame.exists() and rest_frame.exists():
                    # Copy frames to output directory
                    report_dir = output_path.parent / f"{output_path.stem}_frames"
                    report_dir.mkdir(exist_ok=True)
                    shutil.copy(orig_frame, report_dir / f'original_{i:02d}.png')
                    shutil.copy(rest_frame, report_dir / f'restored_{i:02d}.png')

                    report_html += f'''
    <div class="comparison">
        <div class="frame"><h3>Original (Frame {i+1})</h3><img src="{output_path.stem}_frames/original_{i:02d}.png"></div>
        <div class="frame"><h3>Restored (Frame {i+1})</h3><img src="{output_path.stem}_frames/restored_{i:02d}.png"></div>
    </div>
'''

            report_html += '''
</body>
</html>'''

            output_path.write_text(report_html)
            print_colored(f"\n  ✓ Report saved: {output_path}", Colors.OKGREEN)
            print_colored(f"    Open in browser to view side-by-side comparisons", Colors.OKCYAN)

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print()

    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def manage_presets(args):
    """Manage configuration presets."""
    from .config import Config, PRESETS

    if args.action == 'list':
        print_colored("\n📋 Available Presets:", Colors.HEADER)
        print()
        for name, desc in Config.list_presets().items():
            print_colored(f"  {name}", Colors.OKGREEN)
            print(f"      {desc}")
            preset_config = PRESETS.get(name, {})
            print(f"      Scale: {preset_config.get('scale_factor', '?')}x, "
                  f"Model: {preset_config.get('model_name', '?')}, "
                  f"CRF: {preset_config.get('crf', '?')}")
            print()

    elif args.action == 'save':
        if not args.name or not args.file:
            print_colored("Error: --name and --file required for save", Colors.FAIL)
            sys.exit(1)

        # Build config from arguments
        config = Config(
            project_dir=Path("."),
            scale_factor=args.scale or 4,
            crf=args.quality or 18,
            output_format=args.format or 'mkv',
        )
        config.save_preset(Path(args.file), name=args.name)
        print_colored(f"✓ Preset saved: {args.file}", Colors.OKGREEN)

    elif args.action == 'show':
        if not args.name:
            print_colored("Error: --name required for show", Colors.FAIL)
            sys.exit(1)

        if args.name in PRESETS:
            print_colored(f"\n📋 Preset: {args.name}", Colors.HEADER)
            for key, value in PRESETS[args.name].items():
                print(f"  {key}: {value}")
        else:
            print_colored(f"Error: Unknown preset: {args.name}", Colors.FAIL)
            sys.exit(1)


def show_video_info(args):
    """Display video metadata."""
    print_colored(f"\nAnalyzing: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .utils.ffmpeg import probe_video

        metadata = probe_video(input_path)

        print_colored("\n┌─────────────────────────────────────┐", Colors.HEADER)
        print_colored("│         Video Information           │", Colors.HEADER)
        print_colored("└─────────────────────────────────────┘", Colors.HEADER)

        info_lines = [
            f"  Resolution:   {metadata.get('width', '?')}x{metadata.get('height', '?')}",
            f"  Frame Rate:   {metadata.get('framerate', '?')} fps",
            f"  Duration:     {metadata.get('duration', 0):.2f} seconds",
            f"  Video Codec:  {metadata.get('codec', '?')}",
            f"  Has Audio:    {'Yes' if metadata.get('has_audio') else 'No'}",
        ]

        if metadata.get('has_audio'):
            info_lines.append(f"  Audio Codec:  {metadata.get('audio_codec', '?')}")

        for line in info_lines:
            print(line)

        # Suggest RIFE settings
        fps = metadata.get('framerate', 24)
        print_colored("\n  Suggested RIFE targets:", Colors.OKCYAN)
        if fps <= 24:
            print(f"    --target-fps 48  (2x, smooth)")
            print(f"    --target-fps 60  (2.5x, very smooth)")
        elif fps <= 30:
            print(f"    --target-fps 60  (2x, smooth)")
        else:
            print(f"    --target-fps {int(fps * 2)}  (2x)")

        print()

    except Exception as e:
        print_colored(f"Error analyzing video: {e}", Colors.FAIL)
        sys.exit(1)


def analyze_profile_command(args):
    """Analyze a saved performance profile."""
    from .benchmarks import ProfileReport, analyze_profile

    profile_path = Path(args.profile_path)

    if not profile_path.exists():
        print_colored(f"Error: Profile file not found: {profile_path}", Colors.FAIL)
        sys.exit(1)

    print_colored(f"\nAnalyzing profile: {profile_path}", Colors.OKBLUE)

    try:
        # Load the main profile
        report = ProfileReport.load_json(profile_path)

        # Handle comparison if requested
        if args.compare:
            compare_reports = [report]
            labels = [profile_path.stem]

            for compare_path in args.compare:
                compare_file = Path(compare_path)
                if compare_file.exists():
                    compare_report = ProfileReport.load_json(compare_file)
                    compare_reports.append(compare_report)
                    labels.append(compare_file.stem)
                else:
                    print_colored(f"Warning: Comparison file not found: {compare_path}", Colors.WARNING)

            if len(compare_reports) > 1:
                comparison = ProfileReport.compare(compare_reports, labels)
                print(comparison)

                # Calculate improvement if comparing exactly 2 profiles
                if len(compare_reports) == 2:
                    improvement = ProfileReport.calculate_improvement(
                        compare_reports[0], compare_reports[1]
                    )
                    print_colored("\nImprovement Analysis:", Colors.HEADER)
                    if improvement.get("time_improvement_percent", 0) > 0:
                        print_colored(
                            f"  Time improved by {improvement['time_improvement_percent']:.1f}% "
                            f"({improvement.get('speedup_factor', 1):.2f}x speedup)",
                            Colors.OKGREEN
                        )
                    elif improvement.get("time_improvement_percent", 0) < 0:
                        print_colored(
                            f"  Time regressed by {abs(improvement['time_improvement_percent']):.1f}%",
                            Colors.WARNING
                        )

                    if improvement.get("memory_reduction_mb", 0) > 0:
                        print_colored(
                            f"  Memory reduced by {improvement['memory_reduction_mb']:.0f} MB",
                            Colors.OKGREEN
                        )
                    print()
            else:
                # Fall back to single profile analysis
                output = analyze_profile(profile_path)
                print(output)
        else:
            # Single profile analysis based on format
            if args.format == 'detailed':
                print(report.format_detailed())
            elif args.format == 'json':
                import json
                data = {
                    "session_name": report.session_name,
                    "timestamp": report.timestamp,
                    "total_time_seconds": report.total_time,
                    "summary": report.summary.to_dict(),
                    "stages": [s.to_dict() for s in report.stages],
                }
                print(json.dumps(data, indent=2))
            else:  # table format
                print(report.format_table())

        print_colored("Profile analysis complete.", Colors.OKGREEN)

    except Exception as e:
        print_colored(f"Error analyzing profile: {e}", Colors.FAIL)
        sys.exit(1)


def list_gpus_command(args):
    """List available GPUs for processing."""
    from .utils.multi_gpu import MultiGPUManager

    print_colored("\nDetecting available GPUs...\n", Colors.OKBLUE)

    manager = MultiGPUManager()
    table = manager.format_gpu_table()
    print(table)

    if hasattr(args, 'detailed') and args.detailed:
        gpus = manager.get_all_gpu_info()
        if gpus:
            print_colored("\n Detailed GPU Information:", Colors.HEADER)
            print()
            for gpu in gpus:
                print(f"  GPU {gpu.id}: {gpu.name}")
                print(f"    Total Memory:  {gpu.total_vram_mb} MB ({gpu.total_vram_mb / 1024:.1f} GB)")
                print(f"    Free Memory:   {gpu.free_vram_mb} MB ({gpu.free_vram_mb / 1024:.1f} GB)")
                print(f"    Used Memory:   {gpu.used_vram_mb} MB ({gpu.vram_usage_pct:.1f}% used)")
                print(f"    Utilization:   {gpu.utilization_pct:.1f}%")
                if gpu.temperature_c is not None:
                    temp_color = Colors.FAIL if gpu.temperature_c > 85 else Colors.OKGREEN
                    print(f"    Temperature:   {temp_color}{gpu.temperature_c:.0f}C{Colors.ENDC}")
                if gpu.compute_capability:
                    print(f"    Compute Cap:   {gpu.compute_capability}")
                print(f"    Health Status: {'Healthy' if gpu.is_healthy else 'Warning'}")
                print()

    # Usage hints
    print_colored("\nUsage:", Colors.OKCYAN)
    print("  --gpu N         Use specific GPU (e.g., framewright restore --gpu 0 ...)")
    print("  --multi-gpu     Use all available GPUs for faster processing")
    print()


def config_show(args):
    """Display current configuration."""
    from .utils.config_file import ConfigFileManager

    manager = ConfigFileManager()

    print_colored("\nLoading configuration...", Colors.OKBLUE)

    # Check which config files exist
    user_exists = manager.user_config_path.exists()
    project_exists = manager.project_config_path.exists()

    print_colored("\nConfiguration sources:", Colors.BOLD)
    print(f"  User config:    {manager.user_config_path}")
    print(f"                  {'[exists]' if user_exists else '[not found]'}")
    print(f"  Project config: {manager.project_config_path}")
    print(f"                  {'[exists]' if project_exists else '[not found]'}")

    if not user_exists and not project_exists:
        print_colored("\nNo configuration files found.", Colors.WARNING)
        print_colored("Create one with: framewright config init", Colors.OKCYAN)
        return

    try:
        manager.load()

        # Check for validation errors
        errors = manager.get_validation_errors()
        if errors:
            print_colored("\nValidation warnings:", Colors.WARNING)
            for error in errors:
                print(f"  {error.path}: {error.message}")

        print_colored("\nCurrent configuration:", Colors.HEADER)
        print()

        if args.format == 'yaml':
            print(manager.show_config(as_yaml=True))
        else:
            print(manager.show_config(as_yaml=False))

        # Show available profiles
        profiles = manager.list_profiles()
        if profiles:
            print_colored("\nAvailable profiles:", Colors.OKCYAN)
            for profile in profiles:
                print(f"  - {profile}")

    except ImportError:
        print_colored("Error: PyYAML is required for config file support.", Colors.FAIL)
        print_colored("Install with: pip install pyyaml", Colors.WARNING)
        sys.exit(1)


def config_get(args):
    """Get a specific configuration value."""
    from .utils.config_file import ConfigFileManager

    manager = ConfigFileManager()

    try:
        manager.load()
        value = manager.get(args.key)

        if value is None:
            print_colored(f"Key not found: {args.key}", Colors.WARNING)
            sys.exit(1)
        else:
            print(value)

    except ImportError:
        print_colored("Error: PyYAML is required for config file support.", Colors.FAIL)
        print_colored("Install with: pip install pyyaml", Colors.WARNING)
        sys.exit(1)


def config_set(args):
    """Set a configuration value."""
    from .utils.config_file import ConfigFileManager

    manager = ConfigFileManager()

    try:
        # Load existing config first
        if manager.user_config_path.exists() or manager.project_config_path.exists():
            manager.load()

        # Parse value (try to convert to appropriate type)
        value: Any = args.value
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        manager.set(args.key, value)
        print_colored(f"Set {args.key} = {value}", Colors.OKGREEN)

    except ImportError:
        print_colored("Error: PyYAML is required for config file support.", Colors.FAIL)
        print_colored("Install with: pip install pyyaml", Colors.WARNING)
        sys.exit(1)


def config_init(args):
    """Initialize a new configuration file."""
    from .utils.config_file import ConfigFileManager

    manager = ConfigFileManager()

    target = "project" if args.project else "user"

    try:
        config_path = manager.init_config(target=target)
        print_colored(f"Created configuration file: {config_path}", Colors.OKGREEN)
        print_colored("\nEdit this file to customize FrameWright settings.", Colors.OKCYAN)

    except ImportError:
        print_colored("Error: PyYAML is required for config file support.", Colors.FAIL)
        print_colored("Install with: pip install pyyaml", Colors.WARNING)
        sys.exit(1)
    except OSError as e:
        print_colored(f"Error creating config file: {e}", Colors.FAIL)
        sys.exit(1)


# =========================================================================
# Profile Commands - User preference profiles
# =========================================================================


def profile_save(args):
    """Save current settings as a named profile."""
    from .utils.profiles import ProfileManager
    from .config import PRESETS

    manager = ProfileManager()

    name = args.name
    description = getattr(args, 'description', None) or ""

    # Check if saving from a preset
    from_preset = getattr(args, 'from_preset', None)

    if from_preset:
        # Save from a built-in preset
        if from_preset not in PRESETS:
            available = ', '.join(PRESETS.keys())
            print_colored(f"Error: Unknown preset '{from_preset}'", Colors.FAIL)
            print_colored(f"Available presets: {available}", Colors.WARNING)
            sys.exit(1)

        config_dict = PRESETS[from_preset].copy()
        if not description:
            description = f"Based on '{from_preset}' preset"

        try:
            profile_path = manager.save_profile_from_dict(name, config_dict, description)
            print_colored(f"Profile saved: {profile_path}", Colors.OKGREEN)
            print_colored(f"  Source: {from_preset} preset", Colors.OKCYAN)
        except ValueError as e:
            print_colored(f"Error: {e}", Colors.FAIL)
            sys.exit(1)
        except OSError as e:
            print_colored(f"Error saving profile: {e}", Colors.FAIL)
            sys.exit(1)
    else:
        # Save current CLI-like settings as profile
        # Build config dict from common restoration settings
        config_dict = {
            "scale_factor": getattr(args, 'scale', 4),
            "model_name": getattr(args, 'model', 'realesrgan-x4plus'),
            "crf": getattr(args, 'quality', 18),
            "output_format": getattr(args, 'format', 'mkv'),
            "enable_interpolation": getattr(args, 'enable_rife', False),
            "target_fps": getattr(args, 'target_fps', None),
            "rife_model": getattr(args, 'rife_model', 'rife-v4.6'),
            "enable_auto_enhance": getattr(args, 'auto_enhance', False),
            "scratch_sensitivity": getattr(args, 'scratch_sensitivity', 0.5),
            "grain_reduction": getattr(args, 'grain_reduction', 0.3),
            "enable_colorization": getattr(args, 'colorize', False),
            "colorization_model": getattr(args, 'colorize_model', 'ddcolor'),
            "enable_watermark_removal": getattr(args, 'remove_watermark', False),
        }

        try:
            profile_path = manager.save_profile_from_dict(name, config_dict, description)
            print_colored(f"Profile saved: {profile_path}", Colors.OKGREEN)
        except ValueError as e:
            print_colored(f"Error: {e}", Colors.FAIL)
            sys.exit(1)
        except OSError as e:
            print_colored(f"Error saving profile: {e}", Colors.FAIL)
            sys.exit(1)


def profile_load(args):
    """Display settings from a named profile."""
    from .utils.profiles import ProfileManager

    manager = ProfileManager()
    name = args.name

    try:
        profile_data = manager.load_profile_raw(name)

        print_colored(f"\nProfile: {name}", Colors.HEADER)
        print_colored("=" * 50, Colors.HEADER)

        if profile_data.get("description"):
            print_colored(f"\nDescription: {profile_data['description']}", Colors.OKCYAN)

        if profile_data.get("created_at"):
            print(f"Created: {profile_data['created_at']}")
        if profile_data.get("updated_at"):
            print(f"Updated: {profile_data['updated_at']}")

        print_colored("\nSettings:", Colors.BOLD)
        config = profile_data.get("config", {})
        for key, value in sorted(config.items()):
            if value is not None:
                print(f"  {key}: {value}")

        print()
        print_colored(f"Use with: framewright restore --user-profile {name} ...", Colors.OKCYAN)

    except FileNotFoundError:
        print_colored(f"Error: Profile not found: {name}", Colors.FAIL)
        print_colored("\nAvailable profiles:", Colors.WARNING)
        for profile in manager.list_profiles():
            print(f"  - {profile}")
        sys.exit(1)
    except ValueError as e:
        print_colored(f"Error: Invalid profile: {e}", Colors.FAIL)
        sys.exit(1)


def profile_list(args):
    """List all saved profiles."""
    from .utils.profiles import ProfileManager

    manager = ProfileManager()

    profiles = manager.list_profiles_detailed()

    if not profiles:
        print_colored("\nNo profiles saved yet.", Colors.WARNING)
        print_colored("\nCreate a profile with:", Colors.OKCYAN)
        print("  framewright profile save <name> --from-preset quality")
        print("  framewright profile save <name> --from-preset anime")
        print()
        return

    print_colored("\nSaved Profiles:", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    print()

    for profile in profiles:
        name = profile["name"]
        desc = profile.get("description", "")
        created = profile.get("created_at", "")

        print_colored(f"  {name}", Colors.BOLD)
        if desc:
            print(f"    Description: {desc}")
        if created:
            # Format date nicely
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created)
                created_str = dt.strftime("%Y-%m-%d %H:%M")
                print(f"    Created: {created_str}")
            except (ValueError, TypeError):
                print(f"    Created: {created}")
        print()

    print_colored("Use a profile with:", Colors.OKCYAN)
    print("  framewright restore --user-profile <name> --input video.mp4 --output out.mkv")


def profile_delete(args):
    """Delete a saved profile."""
    from .utils.profiles import ProfileManager

    manager = ProfileManager()
    name = args.name

    if not manager.profile_exists(name):
        print_colored(f"Error: Profile not found: {name}", Colors.FAIL)
        sys.exit(1)

    # Confirm deletion unless --yes flag is provided
    if not getattr(args, 'yes', False):
        response = input(f"Delete profile '{name}'? [y/N]: ").strip().lower()
        if response != 'y':
            print_colored("Cancelled.", Colors.WARNING)
            return

    if manager.delete_profile(name):
        print_colored(f"Profile deleted: {name}", Colors.OKGREEN)
    else:
        print_colored(f"Error: Could not delete profile: {name}", Colors.FAIL)
        sys.exit(1)


def install_completion(args):
    """Install shell completion for the specified shell."""
    shell = args.shell

    if not ARGCOMPLETE_AVAILABLE:
        print_colored("Error: argcomplete is required for shell completion.", Colors.FAIL)
        print_colored("Install with: pip install argcomplete", Colors.WARNING)
        sys.exit(1)

    if shell == 'bash':
        _install_bash_completion()
    elif shell == 'zsh':
        _install_zsh_completion()
    elif shell == 'fish':
        _install_fish_completion()
    else:
        print_colored(f"Unsupported shell: {shell}", Colors.FAIL)
        sys.exit(1)


def _install_bash_completion():
    """Install bash completion."""
    # Check if argcomplete is properly set up
    activate_script = """
# FrameWright bash completion
eval "$(register-python-argcomplete framewright)"
"""
    bashrc_path = Path.home() / ".bashrc"
    bash_completion_d = Path("/etc/bash_completion.d")

    print_colored("Installing bash completion...", Colors.OKBLUE)

    # Try to add to .bashrc
    try:
        if bashrc_path.exists():
            content = bashrc_path.read_text()
            if 'register-python-argcomplete framewright' not in content:
                with open(bashrc_path, 'a') as f:
                    f.write(f"\n{activate_script}")
                print_colored(f"Added completion to: {bashrc_path}", Colors.OKGREEN)
            else:
                print_colored("Completion already installed in .bashrc", Colors.OKCYAN)
        else:
            with open(bashrc_path, 'w') as f:
                f.write(activate_script)
            print_colored(f"Created: {bashrc_path}", Colors.OKGREEN)

        print_colored("\nTo activate completion now, run:", Colors.OKCYAN)
        print('  eval "$(register-python-argcomplete framewright)"')
        print_colored("\nOr restart your shell.", Colors.OKCYAN)

    except OSError as e:
        print_colored(f"Error: Could not write to {bashrc_path}: {e}", Colors.FAIL)
        print_colored("\nManually add this line to your .bashrc:", Colors.WARNING)
        print(activate_script)


def _install_zsh_completion():
    """Install zsh completion."""
    zshrc_path = Path.home() / ".zshrc"
    activate_script = """
# FrameWright zsh completion
autoload -U bashcompinit
bashcompinit
eval "$(register-python-argcomplete framewright)"
"""

    print_colored("Installing zsh completion...", Colors.OKBLUE)

    try:
        if zshrc_path.exists():
            content = zshrc_path.read_text()
            if 'register-python-argcomplete framewright' not in content:
                with open(zshrc_path, 'a') as f:
                    f.write(f"\n{activate_script}")
                print_colored(f"Added completion to: {zshrc_path}", Colors.OKGREEN)
            else:
                print_colored("Completion already installed in .zshrc", Colors.OKCYAN)
        else:
            with open(zshrc_path, 'w') as f:
                f.write(activate_script)
            print_colored(f"Created: {zshrc_path}", Colors.OKGREEN)

        print_colored("\nRestart your shell or run:", Colors.OKCYAN)
        print("  source ~/.zshrc")

    except OSError as e:
        print_colored(f"Error: Could not write to {zshrc_path}: {e}", Colors.FAIL)
        print_colored("\nManually add this to your .zshrc:", Colors.WARNING)
        print(activate_script)


def _install_fish_completion():
    """Install fish completion."""
    fish_completions_dir = Path.home() / ".config" / "fish" / "completions"
    completion_file = fish_completions_dir / "framewright.fish"

    print_colored("Installing fish completion...", Colors.OKBLUE)

    try:
        fish_completions_dir.mkdir(parents=True, exist_ok=True)

        completion_content = """
# FrameWright fish completion
function __fish_framewright_complete
    set -lx _ARGCOMPLETE 1
    set -lx _ARGCOMPLETE_FISH 1
    set -lx _ARGCOMPLETE_SUPPRESS_SPACE 1
    set -lx COMP_LINE (commandline -p)
    set -lx COMP_POINT (commandline -pC)
    set -lx COMP_TYPE 'tab'
    set -lx COMP_TYPE_FISH 1
    set -l completions (framewright 2>/dev/null)
    for completion in $completions
        echo $completion
    end
end

complete -c framewright -f -a '(__fish_framewright_complete)'
"""

        with open(completion_file, 'w') as f:
            f.write(completion_content)

        print_colored(f"Created: {completion_file}", Colors.OKGREEN)
        print_colored("\nRestart fish or run: source ~/.config/fish/completions/framewright.fish", Colors.OKCYAN)

    except OSError as e:
        print_colored(f"Error: Could not write to {completion_file}: {e}", Colors.FAIL)
        sys.exit(1)


def _profile_completer(prefix, parsed_args, **kwargs):
    """Completer for profile names."""
    try:
        from .utils.config_file import ConfigFileManager
        manager = ConfigFileManager()
        manager.load()
        profiles = manager.list_profiles()
        return [p for p in profiles if p.startswith(prefix)]
    except Exception:
        # Return built-in profiles as fallback
        return [p for p in ['anime', 'film_restoration', 'fast', 'archive', 'smooth']
                if p.startswith(prefix)]


class SimplifiedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that shows grouped, simplified help by default."""

    def __init__(self, prog, indent_increment=2, max_help_position=30, width=None):
        super().__init__(prog, indent_increment, max_help_position, width)
        self._show_advanced = False

    def _format_action(self, action):
        # Skip advanced options unless explicitly requested
        if hasattr(action, 'advanced') and action.advanced and not self._show_advanced:
            return ''
        return super()._format_action(action)


def _print_simplified_restore_help():
    """Print simplified help for the restore command."""
    print()
    print_colored("FrameWright - Video Restoration", Colors.HEADER)
    print_colored("=" * 50, Colors.HEADER)
    print()
    print_colored("ESSENTIAL OPTIONS:", Colors.BOLD)
    print("  --input FILE         Input video file path")
    print("  --output FILE        Output video file path")
    print("  --output-dir DIR     Output directory (alternative to --output)")
    print("  --preset NAME        Use preset: fast, quality, archive, anime")
    print("  --scale {2,4}        Upscaling factor (default: 2)")
    print("  --dry-run            Show what would happen without processing")
    print()
    print_colored("COMMON OPTIONS:", Colors.BOLD)
    print("  --auto-enhance       Enable automatic face/defect repair")
    print("  --enable-rife        Enable frame interpolation for smoother video")
    print("  --target-fps N       Target FPS for interpolation (e.g., 60)")
    print("  --colorize           Colorize black & white footage")
    print("  --quality N          Output quality CRF (lower=better, default: 18)")
    print()
    print_colored("EXAMPLES:", Colors.OKCYAN)
    print("  # Quick restore with auto settings")
    print("  framewright restore --input old_film.mp4 --output restored.mp4 --auto-enhance")
    print()
    print("  # High quality 4x upscale with 60fps output")
    print("  framewright restore --input video.mp4 --output hd.mp4 --scale 4 --enable-rife --target-fps 60")
    print()
    print("  # Preview what will happen (dry run)")
    print("  framewright restore --input video.mp4 --output out.mp4 --dry-run")
    print()
    print_colored("Show all options with: framewright restore --help --advanced", Colors.WARNING)
    print()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description='FrameWright - AI-powered video restoration pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully automated restoration (recommended for old films)
  framewright restore --input old_film.mp4 --output restored.mp4 --auto-enhance

  # Restore to a specific folder with format selection
  framewright restore --input old_film.mp4 --output-dir /path/to/folder --format mp4

  # Choose output format explicitly
  framewright restore --input video.mp4 --output restored.webm --format webm

  # Analyze video first to get recommendations
  framewright analyze --input video.mp4

  # Restore a YouTube video
  framewright restore --url https://youtube.com/watch?v=xxx --output restored.mp4

  # Restore local video with 4x upscaling
  framewright restore --input video.mp4 --output enhanced.mp4 --scale 4

  # Restore with auto-enhancement AND RIFE interpolation
  framewright restore --input old_film.mp4 --output smooth.mp4 --scale 4 --auto-enhance --enable-rife --target-fps 60

  # Auto-enhance with custom sensitivity settings
  framewright restore --input video.mp4 --output enhanced.mp4 --auto-enhance --scratch-sensitivity 0.7 --grain-reduction 0.5

  # Preview frames before final assembly (inspect quality)
  framewright restore --input video.mp4 --output enhanced.mp4 --preview

  # Check video metadata and get RIFE suggestions
  framewright info --input video.mp4

  # Standalone RIFE interpolation (post-process existing video)
  framewright interpolate --input restored.mp4 --output smooth.mp4 --target-fps 60

  # RIFE on frame directory (must specify source fps)
  framewright interpolate --input frames/ --output interp_frames/ --source-fps 24 --target-fps 60 --frames-only

  # Extract frames only
  framewright extract-frames --input video.mp4 --output frames/

  # Enhance pre-extracted frames
  framewright enhance-frames --input frames/ --output enhanced/ --scale 2

  # Reassemble video from frames
  framewright reassemble --frames-dir enhanced/ --audio original.mp4 --output final.mp4 --fps 24
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Full video restoration workflow')
    restore_parser.add_argument('--url', type=str, help='YouTube or video URL')
    restore_parser.add_argument('--input', type=str, help='Input video file path')
    output_group = restore_parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--output', type=str, help='Output video file path')
    output_group.add_argument('--output-dir', type=str, help='Output directory (video saved as restored_video.<format>)')
    restore_parser.add_argument('--format', type=str, choices=SUPPORTED_FORMATS, default=None,
                               help=f'Output format ({", ".join(SUPPORTED_FORMATS)}). Default: mkv or inferred from output path')
    restore_parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help='Upscaling factor (default: 2)')
    restore_parser.add_argument('--model', type=str, default='realesrgan-x4plus', help='AI model to use')
    restore_parser.add_argument('--quality', type=int, default=18, help='CRF quality (lower=better, default: 18)')
    restore_parser.add_argument('--audio-enhance', action='store_true', help='Enable audio enhancement')
    # Frame deduplication options (for old film with padded frames)
    restore_parser.add_argument('--deduplicate', action='store_true',
                               help='Enable frame deduplication (removes duplicate frames from padded video)')
    restore_parser.add_argument('--dedup-threshold', type=float, default=0.98,
                               help='Similarity threshold for deduplication (0.9-1.0, default: 0.98)')
    # RIFE frame interpolation options
    restore_parser.add_argument('--enable-rife', action='store_true',
                               help='Enable RIFE frame interpolation (increases frame rate)')
    restore_parser.add_argument('--target-fps', type=float, default=None,
                               help='Target frame rate for RIFE (auto-detects source fps, e.g., --target-fps 60)')
    restore_parser.add_argument('--rife-model', type=str, default='rife-v4.6',
                               choices=['rife-v2.3', 'rife-v4.0', 'rife-v4.6'],
                               help='RIFE model version (default: rife-v4.6)')
    # Preview option
    restore_parser.add_argument('--preview', action='store_true',
                               help='Preview frames before final reassembly (pause for inspection)')
    # Auto-enhancement options
    restore_parser.add_argument('--auto-enhance', action='store_true',
                               help='Enable automatic enhancement (defect repair, face restore)')
    restore_parser.add_argument('--no-face-restore', action='store_true',
                               help='Disable automatic face restoration')
    restore_parser.add_argument('--no-defect-repair', action='store_true',
                               help='Disable automatic defect repair')
    restore_parser.add_argument('--scratch-sensitivity', type=float, default=0.5,
                               help='Scratch detection sensitivity (0-1, default: 0.5)')
    restore_parser.add_argument('--grain-reduction', type=float, default=0.3,
                               help='Film grain reduction strength (0-1, default: 0.3)')
    restore_parser.add_argument('--generate-report', action='store_true',
                               help='Generate improvements.md report in output folder')
    # Performance profiling options
    restore_parser.add_argument('--profile-performance', action='store_true',
                               help='Enable performance profiling during restoration')
    restore_parser.add_argument('--output-profile', type=str, default=None,
                               help='Save performance profile to JSON file (e.g., profile.json)')
    # Model directory configuration
    restore_parser.add_argument('--model-dir', type=str, default=None,
                               help='Custom model download directory (default: ~/.framewright/models)')
    # Colorization options
    restore_parser.add_argument('--colorize', action='store_true',
                               help='Enable AI colorization for black & white footage')
    restore_parser.add_argument('--colorize-model', type=str, default='ddcolor',
                               choices=['deoldify', 'ddcolor'],
                               help='Colorization model (default: ddcolor - better quality)')
    # Watermark removal options
    restore_parser.add_argument('--remove-watermark', action='store_true',
                               help='Enable watermark removal')
    restore_parser.add_argument('--watermark-mask', type=str, default=None,
                               help='Path to watermark mask image (white = watermark area)')
    restore_parser.add_argument('--watermark-region', type=str, action='append', default=None,
                               help='Watermark region as x,y,width,height (can specify multiple)')
    restore_parser.add_argument('--watermark-auto-detect', action='store_true',
                               help='Auto-detect watermark locations')
    # GPU selection options
    restore_parser.add_argument('--gpu', type=int, default=None, metavar='N',
                               help='Select specific GPU by index (e.g., --gpu 0)')
    restore_parser.add_argument('--multi-gpu', action='store_true',
                               help='Enable multi-GPU processing for faster restoration')
    # Burnt-in subtitle removal options
    restore_parser.add_argument('--remove-subtitles', action='store_true',
                               help='Remove burnt-in (hard-coded) subtitles from video')
    restore_parser.add_argument('--subtitle-region', type=str, default='bottom_third',
                               choices=['bottom_third', 'bottom_quarter', 'top_quarter', 'full_frame'],
                               help='Region to scan for subtitles (default: bottom_third)')
    restore_parser.add_argument('--subtitle-ocr', type=str, default='auto',
                               choices=['auto', 'easyocr', 'tesseract', 'paddleocr'],
                               help='OCR engine for subtitle detection (default: auto)')
    restore_parser.add_argument('--subtitle-languages', type=str, default='en',
                               help='Comma-separated language codes for OCR (e.g., "en,zh,ja")')

    # ===== Ultimate Preset Features (Advanced AI Restoration) =====

    # TAP Neural Denoising
    restore_parser.add_argument('--tap-denoise', action='store_true',
                               help='Enable TAP neural denoising (Restormer/NAFNet - 34-38 dB PSNR)')
    restore_parser.add_argument('--tap-model', type=str, default='restormer',
                               choices=['restormer', 'nafnet', 'tap'],
                               help='TAP denoising model (default: restormer)')
    restore_parser.add_argument('--tap-strength', type=float, default=1.0,
                               help='TAP denoising strength 0-1 (default: 1.0)')
    restore_parser.add_argument('--tap-preserve-grain', action='store_true',
                               help='Preserve film grain during TAP denoising')

    # Diffusion Super-Resolution
    restore_parser.add_argument('--diffusion-sr', action='store_true',
                               help='Use diffusion-based super-resolution (highest quality, slow)')
    restore_parser.add_argument('--diffusion-steps', type=int, default=20,
                               help='Number of diffusion steps (default: 20, more=better quality)')
    restore_parser.add_argument('--diffusion-guidance', type=float, default=7.5,
                               help='Diffusion guidance scale (default: 7.5)')

    # Face Enhancement Model Selection
    restore_parser.add_argument('--face-model', type=str, default='gfpgan',
                               choices=['gfpgan', 'codeformer', 'aesrgan'],
                               help='Face enhancement model (default: gfpgan, aesrgan=attention-enhanced)')
    restore_parser.add_argument('--aesrgan-strength', type=float, default=0.8,
                               help='AESRGAN face enhancement strength 0-1 (default: 0.8)')

    # QP-Aware Codec Artifact Removal
    restore_parser.add_argument('--qp-artifact-removal', action='store_true',
                               help='Enable QP-aware codec artifact removal (for compressed sources)')
    restore_parser.add_argument('--qp-strength', type=float, default=1.0,
                               help='Artifact removal strength multiplier (default: 1.0)')
    restore_parser.add_argument('--qp-manual', type=int, default=None,
                               help='Manual QP value override (auto-detected if not set)')

    # Exemplar-Based Colorization (SwinTExCo)
    restore_parser.add_argument('--colorize-reference', type=str, nargs='+', default=None,
                               help='Reference color images for exemplar-based colorization')
    restore_parser.add_argument('--colorize-temporal-fusion', action='store_true', default=True,
                               help='Enable temporal fusion for colorization consistency')

    # Reference-Guided Enhancement (IP-Adapter + ControlNet)
    restore_parser.add_argument('--reference-enhance', action='store_true',
                               help='Enable reference-guided enhancement (IP-Adapter + ControlNet)')
    restore_parser.add_argument('--reference-dir', type=str, default=None,
                               help='Directory with reference photos (3-10 high-quality images of same locations/subjects)')
    restore_parser.add_argument('--reference-strength', type=float, default=0.35,
                               help='Reference enhancement strength (0.0-1.0, default: 0.35)')
    restore_parser.add_argument('--reference-guidance', type=float, default=7.5,
                               help='Reference guidance scale (default: 7.5)')
    restore_parser.add_argument('--reference-ip-scale', type=float, default=0.6,
                               help='IP-Adapter influence scale (0.0-1.0, default: 0.6)')

    # Missing Frame Generation
    restore_parser.add_argument('--generate-frames', action='store_true',
                               help='Generate missing frames (for damaged film with gaps)')
    restore_parser.add_argument('--frame-gen-model', type=str, default='interpolate_blend',
                               choices=['svd', 'optical_flow_warp', 'interpolate_blend'],
                               help='Frame generation model (default: interpolate_blend)')
    restore_parser.add_argument('--max-gap-frames', type=int, default=10,
                               help='Maximum frames to generate in a gap (default: 10)')

    # Enhanced Temporal Consistency
    restore_parser.add_argument('--temporal-method', type=str, default='optical_flow',
                               choices=['optical_flow', 'cross_attention', 'hybrid'],
                               help='Temporal consistency method (default: optical_flow, hybrid=best)')
    restore_parser.add_argument('--cross-attention-window', type=int, default=7,
                               help='Cross-attention window size (default: 7)')
    restore_parser.add_argument('--temporal-blend-strength', type=float, default=0.8,
                               help='Temporal blending strength 0-1 (default: 0.8)')

    # Profile support from config file
    profile_arg = restore_parser.add_argument('--profile', type=str, default=None,
                               help='Use a named profile from config file (e.g., anime, film_restoration, ultimate)')
    # Add profile completer if argcomplete is available
    if ARGCOMPLETE_AVAILABLE:
        profile_arg.completer = _profile_completer  # type: ignore

    # User-saved profile support (from ~/.framewright/profiles/)
    restore_parser.add_argument('--user-profile', type=str, default=None, metavar='NAME',
                               help='Use a saved user profile (from framewright profile save)')

    # Dry-run mode
    restore_parser.add_argument('--dry-run', action='store_true',
                               help='Analyze video and show processing plan without executing')

    # ===== v2.1 Modular Features =====

    # JSON sidecar metadata export
    restore_parser.add_argument('--sidecar', action='store_true',
                               help='Export JSON sidecar metadata alongside output video')

    # Scene-aware processing
    restore_parser.add_argument('--scene-aware', action='store_true',
                               help='Enable per-scene intensity adjustment based on scene detection')

    # Motion-adaptive denoising
    restore_parser.add_argument('--motion-adaptive', action='store_true',
                               help='Enable motion-aware denoising (stronger on static, lighter on motion)')

    # Audio-video sync repair
    restore_parser.add_argument('--fix-sync', action='store_true',
                               help='Repair audio-video drift (analyze and correct A/V synchronization)')

    # HDR expansion
    restore_parser.add_argument('--expand-hdr', type=str, default=None,
                               choices=['hdr10', 'dolby-vision'],
                               help='Convert SDR to HDR (hdr10 or dolby-vision)')

    # Aspect ratio correction
    restore_parser.add_argument('--fix-aspect', type=str, default=None,
                               choices=['auto', '4:3', '16:9'],
                               help='Correct aspect ratio (auto-detect or specify 4:3/16:9)')

    # Inverse telecine
    restore_parser.add_argument('--ivtc', type=str, default=None,
                               choices=['auto', '3:2', '2:3'],
                               help='Inverse telecine to recover original film frames (auto, 3:2, or 2:3 pulldown)')

    # Perceptual balance
    restore_parser.add_argument('--perceptual', type=str, default=None,
                               metavar='MODE',
                               help='Perceptual balance mode: faithful, balanced, enhanced, or 0.0-1.0 value')

    # Advanced help flag - show all options when combined with --help
    restore_parser.add_argument('--advanced', action='store_true',
                               help='Show all advanced options (use with --help)')

    restore_parser.set_defaults(func=restore_video)

    # Extract frames command
    extract_parser = subparsers.add_parser('extract-frames', help='Extract frames from video')
    extract_parser.add_argument('--input', type=str, required=True, help='Input video file')
    extract_parser.add_argument('--output', type=str, required=True, help='Output directory for frames')
    extract_parser.set_defaults(func=extract_frames)

    # Enhance frames command
    enhance_parser = subparsers.add_parser('enhance-frames', help='Enhance frames with AI')
    enhance_parser.add_argument('--input', type=str, required=True, help='Input frames directory')
    enhance_parser.add_argument('--output', type=str, required=True, help='Output directory for enhanced frames')
    enhance_parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help='Upscaling factor (default: 2)')
    enhance_parser.add_argument('--model', type=str, default='realesrgan-x4plus',
                               help='AI model to use (realesrgan-x4plus, hat, diffusion, ensemble)')
    enhance_parser.add_argument('--hat-size', type=str, default='large',
                               choices=['small', 'base', 'large'],
                               help='HAT model size (default: large)')
    enhance_parser.add_argument('--diffusion-steps', type=int, default=20,
                               help='Number of diffusion inference steps (default: 20)')
    enhance_parser.add_argument('--diffusion-model', type=str, default='upscale_a_video',
                               choices=['upscale_a_video', 'stable_sr', 'resshift'],
                               help='Diffusion SR model (default: upscale_a_video)')
    enhance_parser.add_argument('--ensemble-models', type=str, default='hat,realesrgan',
                               help='Comma-separated models for ensemble (default: hat,realesrgan)')
    enhance_parser.add_argument('--ensemble-method', type=str, default='weighted',
                               choices=['weighted', 'max_quality', 'per_region', 'adaptive', 'median'],
                               help='Ensemble combination method (default: weighted)')
    enhance_parser.set_defaults(func=enhance_frames)

    # Reassemble command
    reassemble_parser = subparsers.add_parser('reassemble', help='Reassemble video from frames')
    reassemble_parser.add_argument('--frames-dir', type=str, required=True, help='Directory containing frames')
    reassemble_parser.add_argument('--audio', type=str, help='Source video for audio track')
    reassemble_parser.add_argument('--output', type=str, required=True, help='Output video file')
    reassemble_parser.add_argument('--fps', type=float, default=None, help='Frame rate (default: auto-detect from audio source)')
    reassemble_parser.add_argument('--quality', type=int, default=18, help='CRF quality (lower=better, default: 18)')
    reassemble_parser.add_argument('--format', type=str, choices=SUPPORTED_FORMATS, default=None,
                                  help='Output format (inferred from output path if not specified)')
    reassemble_parser.set_defaults(func=reassemble_video)

    # Audio enhance command
    audio_parser = subparsers.add_parser('audio-enhance', help='Enhance audio track')
    audio_parser.add_argument('--input', type=str, required=True, help='Input video/audio file')
    audio_parser.add_argument('--output', type=str, required=True, help='Output audio file')
    audio_parser.set_defaults(func=enhance_audio)

    # Interpolate command (standalone RIFE)
    interp_parser = subparsers.add_parser('interpolate',
                                          help='Interpolate frames using RIFE (increase frame rate)')
    interp_parser.add_argument('--input', type=str, required=True,
                              help='Input video file OR directory of frames')
    interp_parser.add_argument('--output', type=str, required=True,
                              help='Output video file OR directory for interpolated frames')
    interp_parser.add_argument('--target-fps', type=float, required=True,
                              help='Target frame rate (e.g., 60)')
    interp_parser.add_argument('--source-fps', type=float, default=None,
                              help='Source frame rate (auto-detected if input is video)')
    interp_parser.add_argument('--model', type=str, default='rife-v4.6',
                              choices=['rife-v2.3', 'rife-v4.0', 'rife-v4.6'],
                              help='RIFE model version (default: rife-v4.6)')
    interp_parser.add_argument('--frames-only', action='store_true',
                              help='Output interpolated frames only (no video reassembly)')
    interp_parser.add_argument('--quality', type=int, default=18,
                              help='CRF quality for output video (lower=better, default: 18)')
    interp_parser.set_defaults(func=interpolate_video)

    # Info command - show detected metadata
    info_parser = subparsers.add_parser('info', help='Show video metadata (fps, resolution, etc.)')
    info_parser.add_argument('--input', type=str, required=True, help='Input video file')
    info_parser.set_defaults(func=show_video_info)

    # Analyze command - full video analysis with recommendations
    analyze_parser = subparsers.add_parser('analyze',
                                           help='Analyze video and get restoration recommendations')
    analyze_parser.add_argument('--input', type=str, required=True, help='Input video file')
    analyze_parser.add_argument('--output', type=str, default=None,
                               help='Save analysis results to JSON file')
    analyze_parser.set_defaults(func=analyze_video)

    # Analyze profile command - analyze saved performance profile
    analyze_profile_parser = subparsers.add_parser('analyze-profile',
                                                    help='Analyze a saved performance profile')
    analyze_profile_parser.add_argument('profile_path', type=str,
                                        help='Path to profile JSON file')
    analyze_profile_parser.add_argument('--compare', type=str, nargs='+', default=None,
                                        help='Compare with other profile files')
    analyze_profile_parser.add_argument('--format', type=str, choices=['table', 'detailed', 'json'],
                                        default='table', help='Output format (default: table)')
    analyze_profile_parser.set_defaults(func=analyze_profile_command)

    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple videos in batch mode')
    batch_parser.add_argument('--input-dir', type=str, required=True,
                              help='Directory containing videos to process')
    batch_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for restored videos')
    batch_parser.add_argument('--preset', type=str, default=None,
                              choices=['fast', 'quality', 'archive', 'anime', 'film_restoration'],
                              help='Use a configuration preset')
    batch_parser.add_argument('--config', type=str, default=None,
                              help='Path to custom configuration JSON file')
    batch_parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                              help='Upscaling factor (default: 2)')
    batch_parser.add_argument('--quality', type=int, default=18,
                              help='CRF quality (default: 18)')
    batch_parser.add_argument('--format', type=str, choices=SUPPORTED_FORMATS, default='mkv',
                              help='Output format (default: mkv)')
    batch_parser.add_argument('--continue-on-error', action='store_true',
                              help='Continue processing if a video fails')
    batch_parser.set_defaults(func=batch_process)

    # Watch folder command (enhanced with webhooks and retry logic)
    watch_parser = subparsers.add_parser('watch', help='Watch folder for new videos and process automatically')
    watch_parser.add_argument('--input-dir', type=str, required=True,
                              help='Directory to watch for new videos')
    watch_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for restored videos')
    watch_parser.add_argument('--processed-dir', type=str, default=None,
                              help='Move processed originals to this directory')
    watch_parser.add_argument('--profile', type=str, default='quality',
                              choices=['fast', 'quality', 'archive', 'anime', 'film_restoration'],
                              help='Processing profile to use (default: quality)')
    watch_parser.add_argument('--preset', type=str, default=None,
                              choices=['fast', 'quality', 'archive', 'anime', 'film_restoration'],
                              help='Use a configuration preset')
    watch_parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                              help='Upscaling factor (default: 2)')
    watch_parser.add_argument('--quality', type=int, default=18,
                              help='CRF quality (default: 18)')
    watch_parser.add_argument('--format', type=str, choices=SUPPORTED_FORMATS, default='mkv',
                              help='Output format (default: mkv)')
    watch_parser.add_argument('--interval', type=int, default=30,
                              help='Check interval in seconds (default: 30)')
    # Webhook/notification options
    watch_parser.add_argument('--on-complete', type=str, default=None,
                              help='Webhook URL or command to execute on completion')
    watch_parser.add_argument('--on-error', type=str, default=None,
                              help='Webhook URL or command to execute on error')
    # Retry options
    watch_parser.add_argument('--retry-count', type=int, default=3,
                              help='Number of retries on failure (default: 3)')
    # File pattern options
    watch_parser.add_argument('--file-patterns', type=str, default=None,
                              help='Comma-separated glob patterns for video files (default: *.mp4,*.mkv,*.avi,*.mov,*.webm,*.flv,*.wmv,*.m4v)')
    # Delete processed option
    watch_parser.add_argument('--delete-processed', action='store_true',
                              help='Delete original file after successful processing (use with caution)')
    watch_parser.set_defaults(func=watch_folder)

    # Compare videos command
    compare_parser = subparsers.add_parser('compare', help='Compare original and restored videos')
    compare_parser.add_argument('--original', type=str, required=True,
                                help='Original video file')
    compare_parser.add_argument('--restored', type=str, required=True,
                                help='Restored video file')
    compare_parser.add_argument('--output', type=str, default=None,
                                help='Save comparison report to HTML file')
    compare_parser.add_argument('--samples', type=int, default=5,
                                help='Number of sample frames to compare (default: 5)')
    compare_parser.set_defaults(func=compare_videos)

    # Preset management command
    preset_parser = subparsers.add_parser('preset', help='Manage configuration presets')
    preset_parser.add_argument('action', type=str, choices=['list', 'save', 'show'],
                               help='Action: list, save, or show preset')
    preset_parser.add_argument('--name', type=str, default=None,
                               help='Preset name')
    preset_parser.add_argument('--file', type=str, default=None,
                               help='Preset file path (for save)')
    preset_parser.add_argument('--scale', type=int, default=None, choices=[2, 4],
                               help='Scale factor for new preset')
    preset_parser.add_argument('--quality', type=int, default=None,
                               help='CRF quality for new preset')
    preset_parser.add_argument('--format', type=str, default=None, choices=SUPPORTED_FORMATS,
                               help='Output format for new preset')
    preset_parser.set_defaults(func=manage_presets)

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run performance benchmarks'
    )
    benchmark_parser.add_argument(
        '--suite',
        type=str,
        choices=['standard', 'quick', 'gpu-stress', 'memory', 'io'],
        default=None,
        help='Run a predefined benchmark suite (standard: full tests, quick: fast validation, gpu-stress: GPU stress test, memory: memory efficiency, io: disk I/O performance)'
    )
    benchmark_parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Benchmark with a specific video file'
    )
    benchmark_parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        default=None,
        help='Compare performance across device types (gpu, cpu) or profile JSON files'
    )
    benchmark_parser.add_argument(
        '--compare-profiles',
        type=str,
        nargs='+',
        default=None,
        help='Compare multiple profile JSON files (e.g., --compare-profiles profile1.json profile2.json)'
    )
    benchmark_parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Save benchmark report to file (JSON or CSV based on extension)'
    )
    benchmark_parser.add_argument(
        '--scale',
        type=int,
        choices=[2, 4],
        default=2,
        help='Upscaling factor for custom benchmarks (default: 2)'
    )
    benchmark_parser.add_argument(
        '--frames',
        type=int,
        default=100,
        help='Number of frames to process (default: 100)'
    )
    benchmark_parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of benchmark iterations for averaging (default: 3)'
    )
    benchmark_parser.add_argument(
        '--keep-files',
        action='store_true',
        help='Keep temporary benchmark files after completion'
    )
    benchmark_parser.set_defaults(func=run_benchmark)

    # GPUs command - list available GPUs
    gpus_parser = subparsers.add_parser('gpus', help='List available GPUs for processing')
    gpus_parser.add_argument('--detailed', action='store_true',
                             help='Show detailed GPU information including utilization')
    gpus_parser.set_defaults(func=list_gpus_command)

    # Config command with subcommands
    config_parser = subparsers.add_parser('config', help='Manage configuration files')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Config actions')

    # config show
    config_show_parser = config_subparsers.add_parser('show', help='Display current configuration')
    config_show_parser.add_argument('--format', type=str, choices=['yaml', 'flat'], default='yaml',
                                    help='Output format (yaml or flat key=value)')
    config_show_parser.set_defaults(func=config_show)

    # config get
    config_get_parser = config_subparsers.add_parser('get', help='Get a configuration value')
    config_get_parser.add_argument('key', type=str, help='Configuration key (e.g., defaults.scale_factor)')
    config_get_parser.set_defaults(func=config_get)

    # config set
    config_set_parser = config_subparsers.add_parser('set', help='Set a configuration value')
    config_set_parser.add_argument('key', type=str, help='Configuration key (e.g., defaults.scale_factor)')
    config_set_parser.add_argument('value', type=str, help='Value to set')
    config_set_parser.set_defaults(func=config_set)

    # config init
    config_init_parser = config_subparsers.add_parser('init', help='Create default configuration file')
    config_init_parser.add_argument('--project', action='store_true',
                                    help='Create project-local config (.framewright.yaml) instead of user config')
    config_init_parser.set_defaults(func=config_init)

    # =========================================================================
    # Profile Commands - Save and load user preferences
    # =========================================================================
    profile_parser = subparsers.add_parser('profile', help='Manage saved user profiles')
    profile_subparsers = profile_parser.add_subparsers(dest='profile_action', help='Profile actions')

    # profile save
    profile_save_parser = profile_subparsers.add_parser('save', help='Save settings as a named profile')
    profile_save_parser.add_argument('name', type=str, help='Profile name (alphanumeric, hyphens, underscores)')
    profile_save_parser.add_argument('--from-preset', type=str, default=None, metavar='PRESET',
                                     help='Save a built-in preset as profile (fast, quality, archive, anime, etc.)')
    profile_save_parser.add_argument('--description', type=str, default=None,
                                     help='Description of this profile')
    # Optional settings to include in the profile
    profile_save_parser.add_argument('--scale', type=int, default=4, choices=[2, 4],
                                     help='Upscaling factor (default: 4)')
    profile_save_parser.add_argument('--model', type=str, default='realesrgan-x4plus',
                                     help='AI model to use')
    profile_save_parser.add_argument('--quality', type=int, default=18,
                                     help='CRF quality (default: 18)')
    profile_save_parser.add_argument('--format', type=str, default='mkv',
                                     choices=['mkv', 'mp4', 'webm', 'avi', 'mov'],
                                     help='Output format (default: mkv)')
    profile_save_parser.add_argument('--enable-rife', action='store_true',
                                     help='Enable RIFE interpolation')
    profile_save_parser.add_argument('--target-fps', type=float, default=None,
                                     help='Target frame rate for RIFE')
    profile_save_parser.add_argument('--rife-model', type=str, default='rife-v4.6',
                                     help='RIFE model version')
    profile_save_parser.add_argument('--auto-enhance', action='store_true',
                                     help='Enable auto-enhancement')
    profile_save_parser.add_argument('--scratch-sensitivity', type=float, default=0.5,
                                     help='Scratch detection sensitivity')
    profile_save_parser.add_argument('--grain-reduction', type=float, default=0.3,
                                     help='Film grain reduction')
    profile_save_parser.add_argument('--colorize', action='store_true',
                                     help='Enable colorization')
    profile_save_parser.add_argument('--colorize-model', type=str, default='ddcolor',
                                     help='Colorization model')
    profile_save_parser.add_argument('--remove-watermark', action='store_true',
                                     help='Enable watermark removal')
    profile_save_parser.set_defaults(func=profile_save)

    # profile load (show)
    profile_load_parser = profile_subparsers.add_parser('load', help='Display settings from a saved profile')
    profile_load_parser.add_argument('name', type=str, help='Profile name to load')
    profile_load_parser.set_defaults(func=profile_load)

    # profile list
    profile_list_parser = profile_subparsers.add_parser('list', help='List all saved profiles')
    profile_list_parser.set_defaults(func=profile_list)

    # profile delete
    profile_delete_parser = profile_subparsers.add_parser('delete', help='Delete a saved profile')
    profile_delete_parser.add_argument('name', type=str, help='Profile name to delete')
    profile_delete_parser.add_argument('--yes', '-y', action='store_true',
                                       help='Skip confirmation prompt')
    profile_delete_parser.set_defaults(func=profile_delete)

    # Shell completion installer
    completion_parser = subparsers.add_parser('completion', help='Install shell completion')
    completion_parser.add_argument('shell', type=str, choices=['bash', 'zsh', 'fish'],
                                   help='Shell to install completion for')
    completion_parser.set_defaults(func=install_completion)

    # =========================================================================
    # v2.1 Modular Feature Commands
    # =========================================================================

    # --- Notify Commands ---
    notify_parser = subparsers.add_parser('notify', help='Notification setup and management')
    notify_subparsers = notify_parser.add_subparsers(dest='notify_action', help='Notify actions')

    # notify setup
    notify_setup_parser = notify_subparsers.add_parser('setup', help='Configure notification channels')
    notify_setup_subparsers = notify_setup_parser.add_subparsers(dest='notify_setup_type', help='Setup type')

    # notify setup email
    notify_email_parser = notify_setup_subparsers.add_parser('email', help='Interactive SMTP email setup')
    notify_email_parser.set_defaults(func=notify_setup_email_command)

    # notify setup sms
    notify_sms_parser = notify_setup_subparsers.add_parser('sms', help='Interactive Twilio SMS setup')
    notify_sms_parser.set_defaults(func=notify_setup_sms_command)

    # --- Daemon Commands ---
    daemon_parser = subparsers.add_parser('daemon', help='Background daemon management')
    daemon_subparsers = daemon_parser.add_subparsers(dest='daemon_action', help='Daemon actions')

    # daemon start
    daemon_start_parser = daemon_subparsers.add_parser('start', help='Start background daemon')
    daemon_start_parser.add_argument('--port', type=int, default=8765,
                                     help='Port for daemon communication (default: 8765)')
    daemon_start_parser.set_defaults(func=daemon_start_command)

    # daemon stop
    daemon_stop_parser = daemon_subparsers.add_parser('stop', help='Stop background daemon')
    daemon_stop_parser.set_defaults(func=daemon_stop_command)

    # daemon status
    daemon_status_parser = daemon_subparsers.add_parser('status', help='Check daemon status')
    daemon_status_parser.set_defaults(func=daemon_status_command)

    # --- Schedule Commands ---
    schedule_parser = subparsers.add_parser('schedule', help='Job scheduling management')
    schedule_subparsers = schedule_parser.add_subparsers(dest='schedule_action', help='Schedule actions')

    # schedule add
    schedule_add_parser = schedule_subparsers.add_parser('add', help='Add a scheduled job')
    schedule_add_parser.add_argument('--input', type=str, required=True, help='Input video file or directory')
    schedule_add_parser.add_argument('--output', type=str, required=True, help='Output path')
    schedule_add_parser.add_argument('--cron', type=str, default=None,
                                     help='Cron expression for scheduling (e.g., "0 2 * * *" for 2 AM daily)')
    schedule_add_parser.add_argument('--at', type=str, default=None,
                                     help='Run at specific time (e.g., "2024-01-15 14:30" or "14:30")')
    schedule_add_parser.add_argument('--preset', type=str, default='quality',
                                     help='Processing preset (default: quality)')
    schedule_add_parser.set_defaults(func=schedule_add_command)

    # schedule list
    schedule_list_parser = schedule_subparsers.add_parser('list', help='List scheduled jobs')
    schedule_list_parser.add_argument('--all', action='store_true', help='Show completed jobs too')
    schedule_list_parser.set_defaults(func=schedule_list_command)

    # schedule remove
    schedule_remove_parser = schedule_subparsers.add_parser('remove', help='Remove a scheduled job')
    schedule_remove_parser.add_argument('job_id', type=str, help='Job ID to remove')
    schedule_remove_parser.set_defaults(func=schedule_remove_command)

    # --- Integrate Commands ---
    integrate_parser = subparsers.add_parser('integrate', help='Media server integration')
    integrate_subparsers = integrate_parser.add_subparsers(dest='integrate_action', help='Integration actions')

    # integrate plex
    integrate_plex_parser = integrate_subparsers.add_parser('plex', help='Setup Plex Media Server integration')
    integrate_plex_parser.add_argument('--url', type=str, help='Plex server URL')
    integrate_plex_parser.add_argument('--token', type=str, help='Plex auth token')
    integrate_plex_parser.set_defaults(func=integrate_plex_command)

    # integrate jellyfin
    integrate_jellyfin_parser = integrate_subparsers.add_parser('jellyfin', help='Setup Jellyfin integration')
    integrate_jellyfin_parser.add_argument('--url', type=str, help='Jellyfin server URL')
    integrate_jellyfin_parser.add_argument('--api-key', type=str, help='Jellyfin API key')
    integrate_jellyfin_parser.set_defaults(func=integrate_jellyfin_command)

    # --- Upload Commands ---
    upload_parser = subparsers.add_parser('upload', help='Upload restored videos to platforms')
    upload_subparsers = upload_parser.add_subparsers(dest='upload_action', help='Upload actions')

    # upload youtube
    upload_youtube_parser = upload_subparsers.add_parser('youtube', help='Upload to YouTube')
    upload_youtube_parser.add_argument('input', type=str, help='Video file to upload')
    upload_youtube_parser.add_argument('--title', type=str, required=True, help='Video title')
    upload_youtube_parser.add_argument('--description', type=str, default='', help='Video description')
    upload_youtube_parser.add_argument('--privacy', type=str, default='private',
                                       choices=['public', 'unlisted', 'private'],
                                       help='Privacy setting (default: private)')
    upload_youtube_parser.add_argument('--tags', type=str, nargs='+', help='Video tags')
    upload_youtube_parser.set_defaults(func=upload_youtube_command)

    # upload archive
    upload_archive_parser = upload_subparsers.add_parser('archive', help='Upload to Archive.org')
    upload_archive_parser.add_argument('input', type=str, help='Video file to upload')
    upload_archive_parser.add_argument('--identifier', type=str, required=True,
                                       help='Archive.org item identifier')
    upload_archive_parser.add_argument('--title', type=str, required=True, help='Item title')
    upload_archive_parser.add_argument('--description', type=str, default='', help='Item description')
    upload_archive_parser.add_argument('--collection', type=str, default='opensource_movies',
                                       help='Archive.org collection (default: opensource_movies)')
    upload_archive_parser.set_defaults(func=upload_archive_command)

    # --- Report Command ---
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_subparsers = report_parser.add_subparsers(dest='report_action', help='Report actions')

    # report trends
    report_trends_parser = report_subparsers.add_parser('trends', help='Generate quality trend report')
    report_trends_parser.add_argument('--input-dir', type=str, required=True,
                                      help='Directory containing processed videos')
    report_trends_parser.add_argument('--output', type=str, default=None,
                                      help='Output report file (HTML or JSON)')
    report_trends_parser.add_argument('--format', type=str, choices=['html', 'json', 'text'], default='text',
                                      help='Report format (default: text)')
    report_trends_parser.set_defaults(func=report_trends_command)

    # --- Estimate Command ---
    estimate_parser = subparsers.add_parser('estimate', help='Estimate processing cost and time')
    estimate_parser.add_argument('input', type=str, help='Input video file')
    estimate_parser.add_argument('--scale', type=int, default=2, choices=[2, 4],
                                 help='Upscaling factor (default: 2)')
    estimate_parser.add_argument('--preset', type=str, default='quality',
                                 help='Processing preset')
    estimate_parser.add_argument('--cloud', action='store_true',
                                 help='Include cloud GPU cost estimate')
    estimate_parser.set_defaults(func=estimate_command)

    # --- Proxy Commands ---
    proxy_parser = subparsers.add_parser('proxy', help='Proxy workflow management')
    proxy_subparsers = proxy_parser.add_subparsers(dest='proxy_action', help='Proxy actions')

    # proxy create
    proxy_create_parser = proxy_subparsers.add_parser('create', help='Create low-resolution proxy')
    proxy_create_parser.add_argument('input', type=str, help='Input video file')
    proxy_create_parser.add_argument('--output', type=str, default=None, help='Output proxy file')
    proxy_create_parser.add_argument('--scale', type=float, default=0.25,
                                     help='Scale factor for proxy (default: 0.25)')
    proxy_create_parser.add_argument('--quality', type=int, default=28,
                                     help='CRF quality for proxy (default: 28)')
    proxy_create_parser.set_defaults(func=proxy_create_command)

    # proxy apply
    proxy_apply_parser = proxy_subparsers.add_parser('apply', help='Apply proxy edits to original')
    proxy_apply_parser.add_argument('proxy', type=str, help='Edited proxy file')
    proxy_apply_parser.add_argument('original', type=str, help='Original high-res file')
    proxy_apply_parser.add_argument('--output', type=str, required=True, help='Output file')
    proxy_apply_parser.set_defaults(func=proxy_apply_command)

    # --- Project Command ---
    project_parser = subparsers.add_parser('project', help='Project management')
    project_subparsers = project_parser.add_subparsers(dest='project_action', help='Project actions')

    # project changelog
    project_changelog_parser = project_subparsers.add_parser('changelog', help='Show project history')
    project_changelog_parser.add_argument('--project-dir', type=str, default='.',
                                          help='Project directory (default: current)')
    project_changelog_parser.add_argument('--limit', type=int, default=20,
                                          help='Number of entries to show (default: 20)')
    project_changelog_parser.set_defaults(func=project_changelog_command)

    # --- Analyze Subcommands (extend existing analyze) ---
    # Note: The main analyze command exists, we add subparsers for scenes and sync

    # analyze scenes
    analyze_scenes_parser = subparsers.add_parser('analyze-scenes', help='Preview scene breakdown')
    analyze_scenes_parser.add_argument('input', type=str, help='Input video file')
    analyze_scenes_parser.add_argument('--threshold', type=float, default=0.3,
                                       help='Scene detection threshold (default: 0.3)')
    analyze_scenes_parser.add_argument('--output', type=str, default=None,
                                       help='Save scene list to JSON file')
    analyze_scenes_parser.set_defaults(func=analyze_scenes_command)

    # analyze sync
    analyze_sync_parser = subparsers.add_parser('analyze-sync', help='Show A/V drift report')
    analyze_sync_parser.add_argument('input', type=str, help='Input video file')
    analyze_sync_parser.add_argument('--detailed', action='store_true',
                                     help='Show detailed drift measurements')
    analyze_sync_parser.set_defaults(func=analyze_sync_command)

    # =========================================================================
    # Simplified Commands (Apple-like UX)
    # =========================================================================

    # Wizard - Interactive guided setup
    wizard_parser = subparsers.add_parser(
        'wizard',
        help='Interactive guided restoration setup',
        description='Launch the interactive wizard for guided video restoration setup.'
    )
    wizard_parser.add_argument('input', nargs='?', type=str, help='Optional input video file')
    wizard_parser.set_defaults(func=run_wizard_command)

    # Quick - Fast restoration
    quick_parser = subparsers.add_parser(
        'quick',
        help='Fast restoration with good quality',
        description='Quick restoration using optimized settings for speed.'
    )
    quick_parser.add_argument('input', type=str, help='Input video file')
    quick_parser.add_argument('-o', '--output', type=str, help='Output file path')
    quick_parser.set_defaults(func=run_quick_command)

    # Best - Maximum quality
    best_parser = subparsers.add_parser(
        'best',
        help='Maximum quality restoration (slower)',
        description='Maximum quality restoration using all available techniques.'
    )
    best_parser.add_argument('input', type=str, help='Input video file')
    best_parser.add_argument('-o', '--output', type=str, help='Output file path')
    best_parser.set_defaults(func=run_best_command)

    # Archive - Optimized for archive footage
    archive_parser = subparsers.add_parser(
        'archive',
        help='Optimized for archive/historical footage',
        description='Restoration optimized for archive footage with missing frame generation.'
    )
    archive_parser.add_argument('input', type=str, help='Input video file')
    archive_parser.add_argument('-o', '--output', type=str, help='Output file path')
    archive_parser.add_argument(
        '--colorize',
        nargs='*',
        type=str,
        metavar='REF',
        help='Reference images for colorization'
    )
    archive_parser.set_defaults(func=run_archive_command)

    # Auto - Smart auto-detection and restoration
    auto_parser = subparsers.add_parser(
        'auto',
        help='Smart auto-restore (analyzes and picks optimal settings)',
        description='Automatically analyze video and apply optimal restoration settings.'
    )
    auto_parser.add_argument('input', type=str, help='Input video file')
    auto_parser.add_argument('-o', '--output', type=str, help='Output file path')
    auto_parser.set_defaults(func=run_auto_command)

    # Cloud processing commands (Vast.ai + Google Drive)
    try:
        from .cloud.cli import setup_cloud_parser
        setup_cloud_parser(subparsers)
    except ImportError:
        pass  # Cloud module not available

    # =========================================================================
    # Model Management Commands
    # =========================================================================
    models_parser = subparsers.add_parser('models', help='Manage AI models (download, verify, list)')
    models_subparsers = models_parser.add_subparsers(dest='models_action', help='Model management actions')

    # models list
    models_list_parser = models_subparsers.add_parser('list', help='List available models with download status')
    models_list_parser.add_argument('--type', type=str, default=None,
                                    choices=['realesrgan', 'rife', 'deoldify', 'ddcolor', 'lama',
                                             'gfpgan', 'codeformer', 'tap_denoise', 'aesrgan',
                                             'diffusion_sr', 'qp_artifact', 'swintexco',
                                             'frame_generation', 'temporal_attention'],
                                    help='Filter models by type')
    models_list_parser.add_argument('--model-dir', type=str, default=None,
                                    help='Custom model directory (default: ~/.framewright/models)')
    models_list_parser.set_defaults(func=models_list_command)

    # models download
    models_download_parser = models_subparsers.add_parser('download', help='Download AI models')
    models_download_parser.add_argument('model_name', type=str, nargs='?', default=None,
                                        help='Name of the model to download')
    models_download_parser.add_argument('--all', action='store_true',
                                        help='Download all available models')
    models_download_parser.add_argument('--model-dir', type=str, default=None,
                                        help='Custom model directory (default: ~/.framewright/models)')
    models_download_parser.set_defaults(func=models_download_command)

    # models verify
    models_verify_parser = models_subparsers.add_parser('verify', help='Verify model integrity (checksum validation)')
    models_verify_parser.add_argument('--model-dir', type=str, default=None,
                                      help='Custom model directory (default: ~/.framewright/models)')
    models_verify_parser.set_defaults(func=models_verify_command)

    # models path
    models_path_parser = models_subparsers.add_parser('path', help='Show models directory path')
    models_path_parser.add_argument('--model-dir', type=str, default=None,
                                    help='Custom model directory (default: ~/.framewright/models)')
    models_path_parser.set_defaults(func=models_path_command)

    # Legacy --install-completion flag (for compatibility)
    parser.add_argument('--install-completion', type=str, choices=['bash', 'zsh', 'fish'],
                       metavar='SHELL', help='Install shell completion (bash, zsh, fish)')

    # Global logging options (apply to all commands)
    parser.add_argument('--log-level', type=str,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO',
                       help='Set logging level (default: INFO)')
    parser.add_argument('--log-format', type=str,
                       choices=['text', 'json'],
                       default='text',
                       help='Set logging format: text for human-readable, json for structured (default: text)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file (default: stderr only)')

    return parser


def run_benchmark(args):
    """Run performance benchmarks."""
    from .benchmarks import (
        BenchmarkRunner,
        BenchmarkConfig,
        BenchmarkReporter,
        StandardTestSuite,
        BenchmarkType,
        DeviceType,
    )

    print_colored("\n Running FrameWright Benchmarks\n", Colors.HEADER)

    # Initialize runner
    runner = BenchmarkRunner(cleanup=not args.keep_files if hasattr(args, 'keep_files') else True)

    try:
        results = {}

        if args.suite == "standard":
            print_colored("Running standard benchmark suite...", Colors.OKBLUE)
            print_colored("This includes: 720p->1080p, 1080p->4K, interpolation, and combined tests.\n", Colors.OKCYAN)

            def progress_callback(test_name, stage, progress):
                bar_length = 30
                filled = int(bar_length * progress)
                bar = "=" * filled + "-" * (bar_length - filled)
                print(f"\r  [{test_name}] [{bar}] {progress*100:.0f}%", end="", flush=True)

            results = StandardTestSuite.run_standard_suite(
                runner,
                progress_callback=progress_callback
            )
            print("\n")

        elif args.suite == "quick":
            print_colored("Running quick benchmark (validation mode)...\n", Colors.OKBLUE)
            result = StandardTestSuite.run_quick_benchmark(runner)
            results = {"quick_benchmark": result}

        elif args.video:
            print_colored(f"Benchmarking video: {args.video}\n", Colors.OKBLUE)

            video_path = Path(args.video)
            if not video_path.exists():
                print_colored(f"Error: Video file not found: {args.video}", Colors.FAIL)
                sys.exit(1)

            # Get video info for config
            from .utils.ffmpeg import probe_video
            try:
                metadata = probe_video(video_path)
                input_res = (metadata.get('width', 1920), metadata.get('height', 1080))
                output_res = (input_res[0] * args.scale, input_res[1] * args.scale)
            except Exception:
                input_res = (1920, 1080)
                output_res = (input_res[0] * args.scale, input_res[1] * args.scale)

            config = BenchmarkConfig(
                name=f"Custom: {video_path.name}",
                benchmark_type=BenchmarkType.CUSTOM,
                input_resolution=input_res,
                output_resolution=output_res,
                scale_factor=args.scale,
                frame_count=args.frames,
                iterations=args.iterations,
            )

            result = runner.run_benchmark(config)
            results = {"custom_benchmark": result}

        elif args.compare:
            print_colored("Running device comparison benchmark...\n", Colors.OKBLUE)

            # Use standard 720p test for comparison
            base_config = StandardTestSuite.STANDARD_TESTS["720p_to_1080p"]
            devices = [DeviceType.GPU if d == "gpu" else DeviceType.CPU for d in args.compare]

            results = runner.run_comparison(base_config, devices=devices)

        else:
            print_colored("No benchmark specified. Use --suite, --video, or --compare.", Colors.WARNING)
            print_colored("\nExamples:", Colors.OKCYAN)
            print("  framewright benchmark --suite standard")
            print("  framewright benchmark --suite quick")
            print("  framewright benchmark --video input.mp4 --frames 50")
            print("  framewright benchmark --compare gpu cpu")
            sys.exit(0)

        # Generate reports
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            if report_path.suffix.lower() == ".csv":
                BenchmarkReporter.generate_csv_report(results, report_path)
            else:
                BenchmarkReporter.generate_json_report(results, report_path)

            print_colored(f" Report saved to: {report_path}", Colors.OKGREEN)

        # Print summary
        print_colored("\n" + "=" * 60, Colors.HEADER)
        summary = BenchmarkReporter.generate_summary(results)
        print(summary)

        # Print comparison table if multiple results
        if len(results) > 1:
            table = BenchmarkReporter.generate_comparison_table(results)
            print(table)

        # Check for failures
        failed = [r for r in results.values() if not r.success]
        if failed:
            print_colored(f"\n {len(failed)} benchmark(s) failed.", Colors.FAIL)
            for result in failed:
                print_colored(f"   - {result.config.name}: {result.error_message or 'Unknown error'}", Colors.WARNING)
            sys.exit(1)

        print_colored("\n All benchmarks completed successfully.", Colors.OKGREEN)

    except Exception as e:
        print_colored(f"\n Benchmark error: {e}", Colors.FAIL)
        sys.exit(1)


# =============================================================================
# Simplified Command Handlers (Apple-like UX)
# =============================================================================

def run_wizard_command(args):
    """Run the interactive wizard."""
    from ._ui_pkg.wizard import InteractiveWizard
    from ._ui_pkg.terminal import Console

    console = Console()
    wizard = InteractiveWizard(console)

    input_path = Path(args.input) if args.input else None
    result = wizard.run(input_path)

    if result.completed:
        # Run restoration with wizard settings
        config_dict = result.to_config_dict()
        config_dict["project_dir"] = result.input_path.parent / ".framewright_temp"

        from .config import Config
        from .restorer import VideoRestorer

        config = Config(**config_dict)
        restorer = VideoRestorer(config)

        try:
            output = restorer.restore_video(
                source=str(result.input_path),
                output_path=result.output_path,
            )
            console.success(f"Restoration complete: {output}")
        except Exception as e:
            console.error(f"Restoration failed: {e}")
            sys.exit(1)
    elif result.cancelled:
        print_colored("Wizard cancelled", Colors.WARNING)
        sys.exit(0)


def run_quick_command(args):
    """Run quick restoration."""
    from ._ui_pkg.terminal import Console
    from .config import Config
    from .restorer import VideoRestorer

    console = Console()
    console.print_compact_banner()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_quick.mp4"

    console.info("Quick restoration mode - optimized for speed")

    config = Config(
        preset="fast",
        scale_factor=2,
        project_dir=input_path.parent / ".framewright_temp",
    )

    restorer = VideoRestorer(config)
    try:
        result = restorer.restore_video(
            source=str(input_path),
            output_path=output_path,
        )
        console.success(f"Quick restoration complete: {result}")
    except Exception as e:
        console.error(f"Restoration failed: {e}")
        sys.exit(1)


def run_best_command(args):
    """Run maximum quality restoration."""
    from ._ui_pkg.terminal import Console
    from ._ui_pkg.auto_detect import analyze_video_smart
    from ._ui_pkg.recommendations import get_recommendations
    from .config import Config
    from .restorer import VideoRestorer

    console = Console()
    console.print_compact_banner()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_best.mp4"

    console.info("Maximum quality mode - this will take longer")

    # Analyze video
    console.info("Analyzing video...")
    analysis = analyze_video_smart(input_path)

    # Get ultimate recommendations
    recommendations = get_recommendations(
        analysis=analysis,
        user_priority="maximum",
    )

    console.restoration_plan(
        preset="Ultimate",
        stages=recommendations.processing_stages,
        estimated_time="Significantly longer",
        quality_target="Maximum",
    )

    config_dict = recommendations.to_config_dict()
    config_dict["project_dir"] = input_path.parent / ".framewright_temp"
    config_dict["preset"] = "ultimate"

    config = Config(**config_dict)
    restorer = VideoRestorer(config)

    try:
        result = restorer.restore_video(
            source=str(input_path),
            output_path=output_path,
        )
        console.success(f"Maximum quality restoration complete: {result}")
    except Exception as e:
        console.error(f"Restoration failed: {e}")
        sys.exit(1)


def run_archive_command(args):
    """Run archive-optimized restoration."""
    from ._ui_pkg.terminal import Console
    from ._ui_pkg.auto_detect import analyze_video_smart
    from .config import Config
    from .restorer import VideoRestorer

    console = Console()
    console.print_compact_banner()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_archive.mp4"

    console.info("Archive footage restoration mode")

    # Analyze
    console.info("Analyzing archive footage...")
    analysis = analyze_video_smart(input_path)

    console.video_summary(
        path=input_path,
        resolution=analysis.resolution,
        fps=analysis.fps,
        duration=analysis.duration_formatted,
        codec=analysis.codec,
        size_mb=analysis.bitrate_kbps * analysis.duration_seconds / 8000,
    )

    # Build archive-optimized config
    config_dict = {
        "preset": "ultimate",
        "project_dir": input_path.parent / ".framewright_temp",
        "scale_factor": 4 if analysis.width < 720 else 2,
        "enable_tap_denoise": True,
        "enable_qp_artifact_removal": True,
        "enable_frame_generation": True,
        "enable_deduplication": True,
        "temporal_method": "hybrid",
        "enable_interpolation": analysis.fps < 24,
        "target_fps": 24 if analysis.fps < 20 else analysis.fps,
    }

    # Face enhancement if faces detected
    if analysis.content.has_faces:
        config_dict["auto_face_restore"] = True
        config_dict["face_model"] = "aesrgan"

    # Colorization with references
    if args.colorize:
        config_dict["colorization_reference_images"] = args.colorize
        console.info(f"Colorization enabled with {len(args.colorize)} reference images")
    elif analysis.content.is_black_and_white:
        console.warning("B&W footage detected. For colorization, use --colorize with reference images")

    config = Config(**config_dict)
    restorer = VideoRestorer(config)

    try:
        result = restorer.restore_video(
            source=str(input_path),
            output_path=output_path,
        )
        console.success(f"Archive restoration complete: {result}")
    except Exception as e:
        console.error(f"Restoration failed: {e}")
        sys.exit(1)


def run_auto_command(args):
    """Run smart auto-restoration."""
    from ._ui_pkg.terminal import Console
    from ._ui_pkg.auto_detect import analyze_video_smart
    from ._ui_pkg.recommendations import get_recommendations
    from .config import Config
    from .restorer import VideoRestorer

    console = Console()
    console.print_compact_banner()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_restored.mp4"

    console.info("Smart auto-restore mode")

    # Analyze video
    console.info("Analyzing video...")
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

    console.info("Starting restoration...")

    # Create config from recommendations
    config_dict = recommendations.to_config_dict()
    config_dict["project_dir"] = input_path.parent / ".framewright_temp"

    config = Config(**config_dict)
    restorer = VideoRestorer(config)

    try:
        result = restorer.restore_video(
            source=str(input_path),
            output_path=output_path,
        )
        console.success(f"Smart restoration complete: {result}")
    except Exception as e:
        console.error(f"Restoration failed: {e}")
        sys.exit(1)


# =============================================================================
# v2.1 Modular Feature Command Handlers
# =============================================================================

def notify_setup_email_command(args):
    """Interactive SMTP email setup."""
    print_colored("\nEmail Notification Setup", Colors.HEADER)
    print_colored("=" * 40, Colors.HEADER)

    try:
        smtp_server = input("SMTP Server (e.g., smtp.gmail.com): ").strip()
        smtp_port = input("SMTP Port (default: 587): ").strip() or "587"
        email_address = input("Email Address: ").strip()
        password = input("Password/App Password: ").strip()
        recipient = input("Notification Recipient Email: ").strip()

        # Save to config
        from .utils.config_file import ConfigFileManager
        manager = ConfigFileManager()
        if manager.config_exists():
            manager.load()

        manager.set("notifications.email.enabled", True)
        manager.set("notifications.email.smtp_server", smtp_server)
        manager.set("notifications.email.smtp_port", int(smtp_port))
        manager.set("notifications.email.sender", email_address)
        manager.set("notifications.email.recipient", recipient)
        # Note: Password should be stored securely, not in plain config
        print_colored("\nEmail configuration saved.", Colors.OKGREEN)
        print_colored("Note: Store password in FRAMEWRIGHT_SMTP_PASSWORD environment variable.", Colors.WARNING)

    except KeyboardInterrupt:
        print_colored("\nSetup cancelled.", Colors.WARNING)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def notify_setup_sms_command(args):
    """Interactive Twilio SMS setup."""
    print_colored("\nSMS Notification Setup (Twilio)", Colors.HEADER)
    print_colored("=" * 40, Colors.HEADER)

    try:
        account_sid = input("Twilio Account SID: ").strip()
        phone_from = input("Twilio Phone Number (e.g., +1234567890): ").strip()
        phone_to = input("Your Phone Number (e.g., +1234567890): ").strip()

        # Save to config
        from .utils.config_file import ConfigFileManager
        manager = ConfigFileManager()
        if manager.config_exists():
            manager.load()

        manager.set("notifications.sms.enabled", True)
        manager.set("notifications.sms.twilio_sid", account_sid)
        manager.set("notifications.sms.phone_from", phone_from)
        manager.set("notifications.sms.phone_to", phone_to)

        print_colored("\nSMS configuration saved.", Colors.OKGREEN)
        print_colored("Note: Store Twilio Auth Token in TWILIO_AUTH_TOKEN environment variable.", Colors.WARNING)

    except KeyboardInterrupt:
        print_colored("\nSetup cancelled.", Colors.WARNING)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def daemon_start_command(args):
    """Start background daemon."""
    print_colored(f"\nStarting FrameWright daemon on port {args.port}...", Colors.OKBLUE)

    try:
        from .daemon import FrameWrightDaemon
        daemon = FrameWrightDaemon(port=args.port)
        daemon.start()
        print_colored(f"Daemon started (PID: {daemon.pid})", Colors.OKGREEN)
        print_colored(f"Listening on port {args.port}", Colors.OKCYAN)
    except ImportError:
        print_colored("Daemon module not available. Install with: pip install framewright[daemon]", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Failed to start daemon: {e}", Colors.FAIL)
        sys.exit(1)


def daemon_stop_command(args):
    """Stop background daemon."""
    print_colored("\nStopping FrameWright daemon...", Colors.OKBLUE)

    try:
        from .daemon import FrameWrightDaemon
        daemon = FrameWrightDaemon()
        daemon.stop()
        print_colored("Daemon stopped.", Colors.OKGREEN)
    except ImportError:
        print_colored("Daemon module not available.", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Failed to stop daemon: {e}", Colors.FAIL)
        sys.exit(1)


def daemon_status_command(args):
    """Check daemon status."""
    try:
        from .daemon import FrameWrightDaemon
        daemon = FrameWrightDaemon()
        status = daemon.get_status()

        print_colored("\nFrameWright Daemon Status", Colors.HEADER)
        print_colored("=" * 40, Colors.HEADER)
        print(f"  Running:     {'Yes' if status.get('running') else 'No'}")
        if status.get('running'):
            print(f"  PID:         {status.get('pid')}")
            print(f"  Port:        {status.get('port')}")
            print(f"  Uptime:      {status.get('uptime', 'N/A')}")
            print(f"  Jobs Active: {status.get('active_jobs', 0)}")
            print(f"  Jobs Queue:  {status.get('queued_jobs', 0)}")
    except ImportError:
        print_colored("Daemon not running (module not available).", Colors.WARNING)
    except Exception as e:
        print_colored(f"Error checking status: {e}", Colors.FAIL)
        sys.exit(1)


def schedule_add_command(args):
    """Add a scheduled job."""
    print_colored("\nAdding scheduled job...", Colors.OKBLUE)

    if not args.cron and not args.at:
        print_colored("Error: Either --cron or --at must be specified.", Colors.FAIL)
        sys.exit(1)

    try:
        from .queue import JobScheduler
        scheduler = JobScheduler()

        job_id = scheduler.add_job(
            input_path=args.input,
            output_path=args.output,
            cron_expr=args.cron,
            run_at=args.at,
            preset=args.preset,
        )

        print_colored(f"Job scheduled successfully!", Colors.OKGREEN)
        print(f"  Job ID:   {job_id}")
        print(f"  Input:    {args.input}")
        print(f"  Output:   {args.output}")
        if args.cron:
            print(f"  Schedule: {args.cron} (cron)")
        else:
            print(f"  Run at:   {args.at}")

    except ImportError:
        print_colored("Scheduler module not available. Install with: pip install framewright[scheduler]", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Failed to schedule job: {e}", Colors.FAIL)
        sys.exit(1)


def schedule_list_command(args):
    """List scheduled jobs."""
    try:
        from .queue import JobScheduler
        scheduler = JobScheduler()
        jobs = scheduler.list_jobs(include_completed=args.all)

        print_colored("\nScheduled Jobs", Colors.HEADER)
        print_colored("=" * 60, Colors.HEADER)

        if not jobs:
            print_colored("No scheduled jobs found.", Colors.OKCYAN)
            return

        for job in jobs:
            status_color = Colors.OKGREEN if job['status'] == 'completed' else Colors.OKCYAN
            print_colored(f"\n  [{job['id']}]", Colors.BOLD)
            print(f"    Input:    {job['input']}")
            print(f"    Output:   {job['output']}")
            print(f"    Schedule: {job.get('cron') or job.get('run_at')}")
            print(f"    Status:   {status_color}{job['status']}{Colors.ENDC}")

    except ImportError:
        print_colored("Scheduler module not available.", Colors.WARNING)
    except Exception as e:
        print_colored(f"Error listing jobs: {e}", Colors.FAIL)
        sys.exit(1)


def schedule_remove_command(args):
    """Remove a scheduled job."""
    try:
        from .queue import JobScheduler
        scheduler = JobScheduler()
        scheduler.remove_job(args.job_id)
        print_colored(f"Job {args.job_id} removed.", Colors.OKGREEN)
    except ImportError:
        print_colored("Scheduler module not available.", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Failed to remove job: {e}", Colors.FAIL)
        sys.exit(1)


def integrate_plex_command(args):
    """Setup Plex integration."""
    print_colored("\nPlex Media Server Integration", Colors.HEADER)
    print_colored("=" * 40, Colors.HEADER)

    try:
        url = args.url or input("Plex Server URL (e.g., http://192.168.1.100:32400): ").strip()
        token = args.token or input("Plex Auth Token: ").strip()

        from .integration.plex import PlexIntegration
        plex = PlexIntegration(url=url, token=token)

        # Test connection
        if plex.test_connection():
            print_colored("Connection successful!", Colors.OKGREEN)

            # Save config
            from .utils.config_file import ConfigFileManager
            manager = ConfigFileManager()
            if manager.config_exists():
                manager.load()
            manager.set("integrations.plex.url", url)
            manager.set("integrations.plex.enabled", True)
            print_colored("Plex integration saved.", Colors.OKGREEN)
            print_colored("Note: Store token in PLEX_TOKEN environment variable.", Colors.WARNING)
        else:
            print_colored("Connection failed. Please check URL and token.", Colors.FAIL)

    except ImportError:
        print_colored("Plex integration not available. Install with: pip install framewright[plex]", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def integrate_jellyfin_command(args):
    """Setup Jellyfin integration."""
    print_colored("\nJellyfin Integration", Colors.HEADER)
    print_colored("=" * 40, Colors.HEADER)

    try:
        url = args.url or input("Jellyfin Server URL (e.g., http://192.168.1.100:8096): ").strip()
        api_key = args.api_key or input("Jellyfin API Key: ").strip()

        from .integration.jellyfin import JellyfinIntegration
        jellyfin = JellyfinIntegration(url=url, api_key=api_key)

        # Test connection
        if jellyfin.test_connection():
            print_colored("Connection successful!", Colors.OKGREEN)

            # Save config
            from .utils.config_file import ConfigFileManager
            manager = ConfigFileManager()
            if manager.config_exists():
                manager.load()
            manager.set("integrations.jellyfin.url", url)
            manager.set("integrations.jellyfin.enabled", True)
            print_colored("Jellyfin integration saved.", Colors.OKGREEN)
            print_colored("Note: Store API key in JELLYFIN_API_KEY environment variable.", Colors.WARNING)
        else:
            print_colored("Connection failed. Please check URL and API key.", Colors.FAIL)

    except ImportError:
        print_colored("Jellyfin integration not available. Install with: pip install framewright[jellyfin]", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        sys.exit(1)


def upload_youtube_command(args):
    """Upload video to YouTube."""
    print_colored(f"\nUploading to YouTube: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .export.youtube import YouTubeUploader
        uploader = YouTubeUploader()

        result = uploader.upload(
            video_path=input_path,
            title=args.title,
            description=args.description,
            privacy=args.privacy,
            tags=args.tags or [],
        )

        print_colored("Upload successful!", Colors.OKGREEN)
        print(f"  Video ID:  {result['video_id']}")
        print(f"  URL:       https://youtube.com/watch?v={result['video_id']}")

    except ImportError:
        print_colored("YouTube upload not available. Install with: pip install framewright[youtube]", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Upload failed: {e}", Colors.FAIL)
        sys.exit(1)


def upload_archive_command(args):
    """Upload video to Archive.org."""
    print_colored(f"\nUploading to Archive.org: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .export.archive_org import ArchiveOrgUploader
        uploader = ArchiveOrgUploader()

        result = uploader.upload(
            file_path=input_path,
            identifier=args.identifier,
            title=args.title,
            description=args.description,
            collection=args.collection,
        )

        print_colored("Upload successful!", Colors.OKGREEN)
        print(f"  Item URL: https://archive.org/details/{args.identifier}")

    except ImportError:
        print_colored("Archive.org upload not available. Install with: pip install framewright[archive]", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Upload failed: {e}", Colors.FAIL)
        sys.exit(1)


def report_trends_command(args):
    """Generate quality trend report."""
    print_colored(f"\nGenerating quality trend report for: {args.input_dir}", Colors.OKBLUE)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print_colored(f"Error: Directory not found: {input_dir}", Colors.FAIL)
        sys.exit(1)

    try:
        from .reports.trends import TrendAnalyzer
        analyzer = TrendAnalyzer()

        report = analyzer.analyze_directory(input_dir)

        if args.format == 'json':
            import json
            output = json.dumps(report.to_dict(), indent=2)
        elif args.format == 'html':
            output = report.to_html()
        else:
            output = report.to_text()

        if args.output:
            Path(args.output).write_text(output)
            print_colored(f"Report saved to: {args.output}", Colors.OKGREEN)
        else:
            print(output)

    except ImportError:
        print_colored("Trend analysis not available.", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error generating report: {e}", Colors.FAIL)
        sys.exit(1)


def estimate_command(args):
    """Estimate processing cost and time."""
    print_colored(f"\nEstimating processing for: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .utils.ffmpeg import probe_video

        metadata = probe_video(input_path)
        duration = metadata.get('duration', 0)
        resolution = (metadata.get('width', 0), metadata.get('height', 0))
        frame_count = int(duration * metadata.get('framerate', 24))

        # Estimation factors (rough estimates)
        output_pixels = resolution[0] * args.scale * resolution[1] * args.scale
        complexity_factor = 1.0 if args.preset == 'fast' else (1.5 if args.preset == 'quality' else 2.5)

        # Time estimate (very rough: ~0.5s per frame for quality preset at 1080p output)
        base_time_per_frame = 0.1 if args.preset == 'fast' else (0.5 if args.preset == 'quality' else 2.0)
        scale_factor = output_pixels / (1920 * 1080)
        estimated_seconds = frame_count * base_time_per_frame * scale_factor

        print_colored("\nProcessing Estimate", Colors.HEADER)
        print_colored("=" * 40, Colors.HEADER)
        print(f"  Input Resolution:  {resolution[0]}x{resolution[1]}")
        print(f"  Output Resolution: {resolution[0] * args.scale}x{resolution[1] * args.scale}")
        print(f"  Duration:          {duration:.1f} seconds")
        print(f"  Frame Count:       {frame_count}")
        print(f"  Preset:            {args.preset}")

        print_colored("\nTime Estimate (local GPU):", Colors.OKCYAN)
        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        print(f"  Estimated Time:    {hours}h {minutes}m")

        if args.cloud:
            # Cloud cost estimate
            gpu_price_per_hour = 0.50  # Rough Vast.ai estimate
            cloud_speedup = 2.0  # Cloud GPU typically faster
            cloud_hours = estimated_seconds / 3600 / cloud_speedup
            cloud_cost = cloud_hours * gpu_price_per_hour

            print_colored("\nCloud GPU Estimate (Vast.ai):", Colors.OKCYAN)
            print(f"  Estimated Time:    {int(cloud_hours * 60)}m")
            print(f"  Estimated Cost:    ${cloud_cost:.2f}")

    except Exception as e:
        print_colored(f"Error estimating: {e}", Colors.FAIL)
        sys.exit(1)


def proxy_create_command(args):
    """Create low-resolution proxy."""
    print_colored(f"\nCreating proxy for: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_proxy.mp4"

    try:
        import subprocess
        from .utils.ffmpeg import probe_video

        metadata = probe_video(input_path)
        new_width = int(metadata.get('width', 1920) * args.scale)
        new_height = int(metadata.get('height', 1080) * args.scale)

        # Ensure even dimensions
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264',
            '-crf', str(args.quality),
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '128k',
            str(output_path)
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        print_colored(f"Proxy created: {output_path}", Colors.OKGREEN)
        print(f"  Resolution: {new_width}x{new_height}")
        print(f"  Scale:      {args.scale * 100:.0f}% of original")

    except Exception as e:
        print_colored(f"Failed to create proxy: {e}", Colors.FAIL)
        sys.exit(1)


def proxy_apply_command(args):
    """Apply proxy edits to original."""
    print_colored(f"\nApplying proxy edits to original...", Colors.OKBLUE)

    proxy_path = Path(args.proxy)
    original_path = Path(args.original)
    output_path = Path(args.output)

    if not proxy_path.exists():
        print_colored(f"Error: Proxy file not found: {proxy_path}", Colors.FAIL)
        sys.exit(1)

    if not original_path.exists():
        print_colored(f"Error: Original file not found: {original_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .proxy import ProxyWorkflow
        workflow = ProxyWorkflow()

        result = workflow.apply_edits(
            proxy_path=proxy_path,
            original_path=original_path,
            output_path=output_path,
        )

        print_colored(f"Edits applied: {output_path}", Colors.OKGREEN)

    except ImportError:
        print_colored("Proxy workflow not available.", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Failed to apply proxy edits: {e}", Colors.FAIL)
        sys.exit(1)


def project_changelog_command(args):
    """Show project history."""
    print_colored("\nProject Changelog", Colors.HEADER)
    print_colored("=" * 40, Colors.HEADER)

    project_dir = Path(args.project_dir)
    log_file = project_dir / ".framewright" / "changelog.json"

    if not log_file.exists():
        print_colored("No project history found.", Colors.WARNING)
        return

    try:
        import json
        with open(log_file) as f:
            changelog = json.load(f)

        entries = changelog.get('entries', [])[-args.limit:]

        for entry in reversed(entries):
            timestamp = entry.get('timestamp', 'Unknown')
            action = entry.get('action', 'Unknown')
            details = entry.get('details', '')

            print_colored(f"\n  [{timestamp}]", Colors.OKCYAN)
            print(f"    Action:  {action}")
            if details:
                print(f"    Details: {details}")

    except Exception as e:
        print_colored(f"Error reading changelog: {e}", Colors.FAIL)
        sys.exit(1)


def analyze_scenes_command(args):
    """Analyze and preview scene breakdown."""
    print_colored(f"\nAnalyzing scenes in: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .diagnostics.scene_detection import SceneDetector

        detector = SceneDetector(threshold=args.threshold)
        scenes = detector.detect_scenes(input_path)

        print_colored("\nScene Breakdown", Colors.HEADER)
        print_colored("=" * 50, Colors.HEADER)
        print(f"  Total Scenes: {len(scenes)}")
        print()

        for i, scene in enumerate(scenes, 1):
            start = scene.get('start_time', 0)
            end = scene.get('end_time', 0)
            duration = end - start
            print(f"  Scene {i:3d}: {start:7.2f}s - {end:7.2f}s  (duration: {duration:.2f}s)")

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump({'scenes': scenes}, f, indent=2)
            print_colored(f"\nScene list saved to: {args.output}", Colors.OKGREEN)

    except ImportError:
        print_colored("Scene detection not available. Install with: pip install framewright[scenes]", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error analyzing scenes: {e}", Colors.FAIL)
        sys.exit(1)


def analyze_sync_command(args):
    """Analyze audio-video synchronization."""
    print_colored(f"\nAnalyzing A/V sync: {args.input}", Colors.OKBLUE)

    input_path = Path(args.input)
    if not input_path.exists():
        print_colored(f"Error: File not found: {input_path}", Colors.FAIL)
        sys.exit(1)

    try:
        from .diagnostics.av_sync import AVSyncAnalyzer

        analyzer = AVSyncAnalyzer()
        report = analyzer.analyze(input_path)

        print_colored("\nA/V Sync Analysis", Colors.HEADER)
        print_colored("=" * 40, Colors.HEADER)
        print(f"  Average Drift:  {report.get('avg_drift_ms', 0):.1f} ms")
        print(f"  Max Drift:      {report.get('max_drift_ms', 0):.1f} ms")
        print(f"  Drift Direction: {'Audio ahead' if report.get('avg_drift_ms', 0) > 0 else 'Video ahead'}")

        sync_status = "Good" if abs(report.get('avg_drift_ms', 0)) < 50 else "Needs correction"
        status_color = Colors.OKGREEN if sync_status == "Good" else Colors.WARNING
        print(f"  Sync Status:    {status_color}{sync_status}{Colors.ENDC}")

        if args.detailed and 'measurements' in report:
            print_colored("\nDetailed Measurements:", Colors.OKCYAN)
            for m in report['measurements'][:10]:
                print(f"    {m['timestamp']:.2f}s: {m['drift_ms']:.1f}ms")

        if sync_status != "Good":
            print_colored("\nTo fix sync issues, use: framewright restore --fix-sync", Colors.OKCYAN)

    except ImportError:
        print_colored("A/V sync analysis not available.", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"Error analyzing sync: {e}", Colors.FAIL)
        sys.exit(1)


# =============================================================================
# Model Management Commands
# =============================================================================

def models_list_command(args):
    """List available models with download status."""
    from .utils.model_manager import ModelManager, MODEL_REGISTRY, ModelType

    model_dir = Path(args.model_dir) if hasattr(args, 'model_dir') and args.model_dir else None
    manager = ModelManager(model_dir=model_dir)

    print_colored("\nAvailable Models", Colors.HEADER)
    print_colored("=" * 80, Colors.HEADER)

    # Group models by type
    models_by_type: Dict[str, List[str]] = {}
    for model_name, model_info in MODEL_REGISTRY.items():
        type_name = model_info.model_type.value if model_info.model_type else "other"
        if type_name not in models_by_type:
            models_by_type[type_name] = []
        models_by_type[type_name].append(model_name)

    # Filter by type if specified
    filter_type = getattr(args, 'type', None)

    for type_name in sorted(models_by_type.keys()):
        if filter_type and filter_type != type_name:
            continue

        print_colored(f"\n[{type_name.upper()}]", Colors.BOLD)

        for model_name in sorted(models_by_type[type_name]):
            model_info = MODEL_REGISTRY[model_name]
            is_downloaded = manager.is_model_available(model_name)

            # Status indicator
            status = f"{Colors.OKGREEN}[downloaded]{Colors.ENDC}" if is_downloaded else f"{Colors.WARNING}[not downloaded]{Colors.ENDC}"

            # Size formatting
            size_str = f"{model_info.size_mb:.1f} MB" if model_info.size_mb < 1000 else f"{model_info.size_mb / 1000:.1f} GB"

            print(f"  {model_name:<30} {size_str:>10}  {status}")
            if model_info.description:
                print(f"      {Colors.OKCYAN}{model_info.description}{Colors.ENDC}")

    # Summary
    downloaded = manager.list_downloaded_models()
    total = len(MODEL_REGISTRY)
    storage = manager.get_storage_usage()

    print_colored(f"\nSummary: {len(downloaded)}/{total} models downloaded", Colors.OKBLUE)
    print_colored(f"Storage used: {storage['total_size_mb']:.1f} MB", Colors.OKBLUE)
    print_colored(f"Model directory: {manager.model_dir}", Colors.OKCYAN)


def models_download_command(args):
    """Download models."""
    from .utils.model_manager import ModelManager, MODEL_REGISTRY, DownloadError

    model_dir = Path(args.model_dir) if hasattr(args, 'model_dir') and args.model_dir else None
    manager = ModelManager(model_dir=model_dir)

    if args.all:
        # Download all models
        models_to_download = list(MODEL_REGISTRY.keys())
        print_colored(f"\nDownloading all {len(models_to_download)} models...", Colors.HEADER)
    elif args.model_name:
        models_to_download = [args.model_name]
    else:
        print_colored("Error: Specify a model name or use --all to download all models", Colors.FAIL)
        sys.exit(1)

    success_count = 0
    fail_count = 0

    for model_name in models_to_download:
        if model_name not in MODEL_REGISTRY:
            print_colored(f"Error: Unknown model '{model_name}'", Colors.FAIL)
            print_colored("Use 'framewright models list' to see available models", Colors.OKCYAN)
            fail_count += 1
            continue

        model_info = MODEL_REGISTRY[model_name]

        # Check if already downloaded
        if manager.is_model_available(model_name):
            print_colored(f"Model '{model_name}' is already downloaded and verified", Colors.OKGREEN)
            success_count += 1
            continue

        print_colored(f"\nDownloading {model_name} ({model_info.size_mb:.1f} MB)...", Colors.OKBLUE)

        try:
            path = manager.download_model(model_name)
            print_colored(f"Successfully downloaded to: {path}", Colors.OKGREEN)
            success_count += 1
        except DownloadError as e:
            print_colored(f"Failed to download {model_name}: {e}", Colors.FAIL)
            fail_count += 1
        except Exception as e:
            print_colored(f"Error downloading {model_name}: {e}", Colors.FAIL)
            fail_count += 1

    # Summary
    if len(models_to_download) > 1:
        print_colored(f"\nDownload complete: {success_count} succeeded, {fail_count} failed", Colors.HEADER)


def models_verify_command(args):
    """Verify model integrity."""
    from .utils.model_manager import ModelManager, MODEL_REGISTRY, ModelVerificationError

    model_dir = Path(args.model_dir) if hasattr(args, 'model_dir') and args.model_dir else None
    manager = ModelManager(model_dir=model_dir)

    print_colored("\nVerifying Downloaded Models", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)

    downloaded_models = manager.list_downloaded_models()

    if not downloaded_models:
        print_colored("No models downloaded yet.", Colors.WARNING)
        print_colored("Use 'framewright models download <model_name>' to download models", Colors.OKCYAN)
        return

    verified_count = 0
    corrupted_count = 0
    no_checksum_count = 0

    for model_name in downloaded_models:
        model_info = MODEL_REGISTRY[model_name]
        model_path = manager.get_model_path(model_name)

        if not model_info.checksum:
            print(f"  {model_name}: {Colors.WARNING}[no checksum available]{Colors.ENDC}")
            no_checksum_count += 1
            continue

        try:
            manager.verify_model(model_name)
            print(f"  {model_name}: {Colors.OKGREEN}[verified]{Colors.ENDC}")
            verified_count += 1
        except ModelVerificationError as e:
            print(f"  {model_name}: {Colors.FAIL}[CORRUPTED]{Colors.ENDC}")
            print(f"      {Colors.FAIL}{e}{Colors.ENDC}")
            corrupted_count += 1
        except FileNotFoundError:
            print(f"  {model_name}: {Colors.FAIL}[MISSING]{Colors.ENDC}")
            corrupted_count += 1

    # Summary
    print_colored(f"\nVerification Summary:", Colors.BOLD)
    print(f"  Verified:    {Colors.OKGREEN}{verified_count}{Colors.ENDC}")
    print(f"  No checksum: {Colors.WARNING}{no_checksum_count}{Colors.ENDC}")
    print(f"  Corrupted:   {Colors.FAIL}{corrupted_count}{Colors.ENDC}")

    if corrupted_count > 0:
        print_colored("\nRe-download corrupted models with: framewright models download <model_name>", Colors.OKCYAN)
        sys.exit(1)


def models_path_command(args):
    """Show models directory path."""
    from .utils.model_manager import ModelManager

    model_dir = Path(args.model_dir) if hasattr(args, 'model_dir') and args.model_dir else None
    manager = ModelManager(model_dir=model_dir)

    print_colored("\nModel Storage Location", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    print(f"  Path: {manager.model_dir}")

    # Check if directory exists
    if manager.model_dir.exists():
        print(f"  Status: {Colors.OKGREEN}Directory exists{Colors.ENDC}")

        # Get storage stats
        storage = manager.get_storage_usage()
        print(f"  Models downloaded: {storage['model_count']}")
        print(f"  Total size: {storage['total_size_mb']:.1f} MB")
    else:
        print(f"  Status: {Colors.WARNING}Directory does not exist (will be created on first download){Colors.ENDC}")

    # Show how to customize
    print_colored("\nTo use a custom model directory:", Colors.OKCYAN)
    print("  framewright models list --model-dir /path/to/models")
    print("  framewright restore --model-dir /path/to/models ...")


def main():
    """Main CLI entry point."""
    # Check for simplified help on restore command (without --advanced)
    if len(sys.argv) >= 2 and sys.argv[1] == 'restore':
        if '--help' in sys.argv or '-h' in sys.argv:
            if '--advanced' not in sys.argv:
                # Show simplified help
                print_header()
                _print_simplified_restore_help()
                sys.exit(0)

    parser = create_parser()

    # Enable argcomplete if available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # Configure structured logging based on CLI arguments
    from .utils.logging import configure_from_cli
    configure_from_cli(
        log_level=getattr(args, 'log_level', 'INFO'),
        log_format=getattr(args, 'log_format', 'text'),
        log_file=getattr(args, 'log_file', None),
    )

    # Handle --install-completion flag
    if hasattr(args, 'install_completion') and args.install_completion:
        class CompletionArgs:
            shell = args.install_completion
        install_completion(CompletionArgs())
        sys.exit(0)

    if not args.command:
        print_header()
        parser.print_help()
        sys.exit(0)

    # Handle config command without action
    if args.command == 'config' and (not hasattr(args, 'config_action') or args.config_action is None):
        print_header()
        print_colored("Usage: framewright config <action>", Colors.OKBLUE)
        print_colored("\nAvailable actions:", Colors.BOLD)
        print("  show    Display current configuration")
        print("  get     Get a configuration value")
        print("  set     Set a configuration value")
        print("  init    Create default configuration file")
        print_colored("\nExamples:", Colors.OKCYAN)
        print("  framewright config show")
        print("  framewright config get defaults.scale_factor")
        print("  framewright config set defaults.scale_factor 2")
        print("  framewright config init")
        sys.exit(0)

    # Handle cloud command without action
    if args.command == 'cloud' and (not hasattr(args, 'cloud_action') or args.cloud_action is None):
        print_header()
        print_colored("Usage: framewright cloud <action>", Colors.OKBLUE)
        print_colored("\nCloud GPU processing with Vast.ai + Google Drive", Colors.OKCYAN)
        print_colored("\nAvailable actions:", Colors.BOLD)
        print("  submit    Submit video for cloud processing")
        print("  status    Check job status")
        print("  gpus      List available GPUs with pricing")
        print("  jobs      List all cloud jobs")
        print("  cancel    Cancel a running job")
        print("  balance   Check Vast.ai credit balance")
        print("  download  Download completed job result")
        print_colored("\nExamples:", Colors.OKCYAN)
        print("  framewright cloud submit --input video.mp4 --scale 4")
        print("  framewright cloud status fw_abc123")
        print("  framewright cloud gpus")
        print("  framewright cloud balance")
        sys.exit(0)

    # Handle models command without action
    if args.command == 'models' and (not hasattr(args, 'models_action') or args.models_action is None):
        print_header()
        print_colored("Usage: framewright models <action>", Colors.OKBLUE)
        print_colored("\nManage AI models for video restoration", Colors.OKCYAN)
        print_colored("\nAvailable actions:", Colors.BOLD)
        print("  list      List available models with download status")
        print("  download  Download AI models (by name or --all)")
        print("  verify    Verify model integrity (checksum validation)")
        print("  path      Show models directory path")
        print_colored("\nExamples:", Colors.OKCYAN)
        print("  framewright models list")
        print("  framewright models list --type realesrgan")
        print("  framewright models download realesrgan-x4plus")
        print("  framewright models download --all")
        print("  framewright models verify")
        print("  framewright models path")
        sys.exit(0)

    print_header()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print_colored("\n\nOperation cancelled by user", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nError: {str(e)}", Colors.FAIL)
        sys.exit(1)


def launch_ui():
    """Launch the web UI."""
    parser = argparse.ArgumentParser(
        description='Launch FrameWright Web UI',
    )
    parser.add_argument('--share', action='store_true', help='Create public shareable link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on (default: 7860)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')

    args = parser.parse_args()

    try:
        from .ui import launch_ui as _launch_ui, check_gradio_installed, install_gradio_instructions

        if not check_gradio_installed():
            print(install_gradio_instructions())
            sys.exit(1)

        _launch_ui(
            share=args.share,
            server_port=args.port,
            server_name=args.host,
        )
    except ImportError as e:
        print_colored(f"Error: Failed to import UI module: {e}", Colors.FAIL)
        print_colored("\nInstall UI dependencies with:", Colors.WARNING)
        print_colored("    pip install framewright[ui]", Colors.OKCYAN)
        sys.exit(1)


def check_hardware():
    """Run hardware compatibility check."""
    print_header()
    print_colored("\n Checking hardware compatibility...\n", Colors.OKBLUE)

    try:
        from .hardware import check_hardware as _check_hardware, print_hardware_report

        report = _check_hardware()
        print(print_hardware_report(report))

        # Exit code based on status
        if report.overall_status == "ready":
            sys.exit(0)
        elif report.overall_status == "limited":
            sys.exit(0)  # Still usable
        else:
            sys.exit(1)

    except ImportError as e:
        print_colored(f"Error: {e}", Colors.FAIL)
        print_colored("\nInstall full dependencies with:", Colors.WARNING)
        print_colored("    pip install framewright[full]", Colors.OKCYAN)
        sys.exit(1)


if __name__ == '__main__':
    main()
