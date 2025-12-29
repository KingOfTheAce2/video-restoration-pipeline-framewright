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
    print(f"{color}{message}{Colors.ENDC}")


def print_header():
    """Print CLI header."""
    header = """
    ╔═══════════════════════════════════════════╗
    ║       FrameWright v1.0.0-dev              ║
    ║    Video Restoration Pipeline             ║
    ╚═══════════════════════════════════════════╝
    """
    print_colored(header, Colors.HEADER)


def validate_input(input_path: str) -> Path:
    """Validate input file or URL."""
    if input_path.startswith(('http://', 'https://')):
        return Path(input_path)

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


def restore_video(args):
    """Full video restoration workflow with actual implementation."""
    from .config import Config
    from .restorer import VideoRestorer

    output_path = get_output_path(args)
    output_dir = get_output_dir(args)
    output_format = get_output_format(args)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print_colored(f"\n📁 Output will be saved to: {output_path}", Colors.OKCYAN)
    print_colored(f"📦 Output format: {output_format.upper()}", Colors.OKCYAN)

    # Determine input source
    if args.input:
        source = args.input
        if not Path(source).exists():
            print_colored(f"Error: Input file not found: {source}", Colors.FAIL)
            sys.exit(1)
    elif args.url:
        source = args.url
    else:
        print_colored("Error: Please provide --input or --url", Colors.FAIL)
        sys.exit(1)

    # Determine model based on scale
    model_name = args.model
    if args.scale == 2 and 'x4' in model_name:
        model_name = 'realesrgan-x2plus'
        print_colored(f"ℹ️  Using {model_name} for 2x scale", Colors.WARNING)

    try:
        # Create configuration
        work_dir = output_dir / ".framewright_work"
        # Determine model directory
        model_dir = None
        if hasattr(args, 'model_dir') and args.model_dir:
            model_dir = Path(args.model_dir).expanduser()

        config = Config(
            project_dir=work_dir,
            output_dir=output_dir,
            scale_factor=args.scale,
            model_name=model_name,
            crf=args.quality,
            output_format=output_format,
            enable_checkpointing=True,
            enable_validation=True,
            enable_interpolation=args.enable_rife,
            target_fps=args.target_fps,
            rife_model=args.rife_model,
            enable_auto_enhance=args.auto_enhance,
            auto_face_restore=not args.no_face_restore if hasattr(args, 'no_face_restore') else True,
            auto_defect_repair=not args.no_defect_repair if hasattr(args, 'no_defect_repair') else True,
            scratch_sensitivity=args.scratch_sensitivity,
            grain_reduction=args.grain_reduction,
            model_dir=model_dir if model_dir else Path.home() / ".framewright" / "models",
            enable_colorization=getattr(args, 'colorize', False),
            colorization_model=getattr(args, 'colorize_model', 'ddcolor'),
            enable_watermark_removal=getattr(args, 'remove_watermark', False),
            watermark_auto_detect=getattr(args, 'watermark_auto_detect', False),
            gpu_id=getattr(args, 'gpu', None),
            enable_multi_gpu=getattr(args, 'multi_gpu', False),
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
    """Enhance extracted frames using Real-ESRGAN."""
    import subprocess

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

    # Determine model name
    model_name = args.model
    if scale == 2 and 'x4' in model_name:
        model_name = 'realesrgan-x2plus'
        print_colored(f"  Using {model_name} for 2x scale", Colors.WARNING)

    # Process frames with Real-ESRGAN
    failed_frames = []

    with tqdm(total=len(frames), desc="Enhancing frames", ncols=100) as pbar:
        for frame in frames:
            output_frame = output_dir / frame.name

            cmd = [
                'realesrgan-ncnn-vulkan',
                '-i', str(frame),
                '-o', str(output_frame),
                '-n', model_name,
                '-s', str(scale),
                '-f', 'png'
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    failed_frames.append(frame.name)
            except subprocess.TimeoutExpired:
                failed_frames.append(frame.name)
            except FileNotFoundError:
                print_colored("\nError: realesrgan-ncnn-vulkan not found.", Colors.FAIL)
                print_colored("Install from: https://github.com/xinntao/Real-ESRGAN", Colors.WARNING)
                sys.exit(1)

            pbar.update(1)

    enhanced_count = len(frames) - len(failed_frames)
    print_colored(f"\n✓ Enhanced {enhanced_count}/{len(frames)} frames to: {output_dir}", Colors.OKGREEN)

    if failed_frames:
        print_colored(f"  Warning: {len(failed_frames)} frames failed to enhance", Colors.WARNING)


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
    # Profile support from config file
    profile_arg = restore_parser.add_argument('--profile', type=str, default=None,
                               help='Use a named profile from config file (e.g., anime, film_restoration, fast)')
    # Add profile completer if argcomplete is available
    if ARGCOMPLETE_AVAILABLE:
        profile_arg.completer = _profile_completer  # type: ignore
    # Dry-run mode
    restore_parser.add_argument('--dry-run', action='store_true',
                               help='Analyze video and show processing plan without executing')
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
    enhance_parser.add_argument('--model', type=str, default='realesrgan-x4plus', help='AI model to use')
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

    # Shell completion installer
    completion_parser = subparsers.add_parser('completion', help='Install shell completion')
    completion_parser.add_argument('shell', type=str, choices=['bash', 'zsh', 'fish'],
                                   help='Shell to install completion for')
    completion_parser.set_defaults(func=install_completion)

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


def main():
    """Main CLI entry point."""
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
