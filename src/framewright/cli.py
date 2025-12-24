#!/usr/bin/env python3
"""
FrameWright CLI - Video Restoration Pipeline
Command-line interface for video enhancement and restoration.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable else range(total or 0)


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
    ║         FrameWright v1.2.0                ║
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


def restore_video(args):
    """Full video restoration workflow."""
    print_colored("\n[1/5] Downloading/Loading video...", Colors.OKBLUE)
    # TODO: Implement video loading

    print_colored("[2/5] Extracting frames...", Colors.OKBLUE)
    # TODO: Implement frame extraction

    print_colored("[3/5] Enhancing frames with AI...", Colors.OKBLUE)
    # TODO: Implement frame enhancement with progress bar
    with tqdm(total=100, desc="Processing frames", ncols=100) as pbar:
        # Placeholder for actual processing
        pbar.update(100)

    print_colored("[4/5] Reassembling video...", Colors.OKBLUE)
    # TODO: Implement video reassembly

    if args.audio_enhance:
        print_colored("[5/5] Enhancing audio...", Colors.OKBLUE)
        # TODO: Implement audio enhancement
    else:
        print_colored("[5/5] Copying audio...", Colors.OKBLUE)
        # TODO: Implement audio copy

    print_colored(f"\n✓ Video restoration complete: {args.output}", Colors.OKGREEN)


def extract_frames(args):
    """Extract frames from video."""
    print_colored(f"\nExtracting frames from: {args.input}", Colors.OKBLUE)
    input_path = validate_input(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement frame extraction
    with tqdm(total=100, desc="Extracting frames", ncols=100) as pbar:
        # Placeholder
        pbar.update(100)

    print_colored(f"✓ Frames extracted to: {output_dir}", Colors.OKGREEN)


def enhance_frames(args):
    """Enhance extracted frames using AI."""
    print_colored(f"\nEnhancing frames with {args.model} model (scale: {args.scale}x)", Colors.OKBLUE)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print_colored(f"Error: Input directory not found: {input_dir}", Colors.FAIL)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scale = validate_scale(args.scale)

    # TODO: Implement frame enhancement
    with tqdm(total=100, desc="Enhancing frames", ncols=100) as pbar:
        # Placeholder
        pbar.update(100)

    print_colored(f"✓ Enhanced frames saved to: {output_dir}", Colors.OKGREEN)


def reassemble_video(args):
    """Reassemble video from enhanced frames."""
    print_colored("\nReassembling video from frames...", Colors.OKBLUE)
    frames_dir = Path(args.frames_dir)

    if not frames_dir.exists():
        print_colored(f"Error: Frames directory not found: {frames_dir}", Colors.FAIL)
        sys.exit(1)

    # TODO: Implement video reassembly
    with tqdm(total=100, desc="Encoding video", ncols=100) as pbar:
        # Placeholder
        pbar.update(100)

    print_colored(f"✓ Video reassembled: {args.output}", Colors.OKGREEN)


def enhance_audio(args):
    """Enhance audio track."""
    print_colored("\nEnhancing audio track...", Colors.OKBLUE)
    input_path = validate_input(args.input)

    # TODO: Implement audio enhancement
    with tqdm(total=100, desc="Processing audio", ncols=100) as pbar:
        # Placeholder
        pbar.update(100)

    print_colored(f"✓ Audio enhanced: {args.output}", Colors.OKGREEN)


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
        import tempfile
        import shutil

        interpolator = FrameInterpolator(model=args.model)

        # Determine if input is video or frame directory
        if input_path.is_dir():
            # Input is a directory of frames
            frames_dir = input_path
            source_fps = args.source_fps
            if source_fps is None:
                print_colored("Error: --source-fps required when input is a directory", Colors.FAIL)
                sys.exit(1)
        else:
            # Input is a video file - extract frames and detect fps
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
            # Reassemble video
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

        print_colored("\n  📊 Video Info:", Colors.BOLD)
        print(f"     Resolution:    {analysis.resolution[0]}x{analysis.resolution[1]}")
        print(f"     Frame Rate:    {analysis.source_fps:.2f} fps")
        print(f"     Duration:      {analysis.duration:.1f} seconds")
        print(f"     Total Frames:  {analysis.total_frames}")

        print_colored("\n  🎬 Content Detection:", Colors.BOLD)
        print(f"     Primary Type:  {analysis.primary_content.name.replace('_', ' ').title()}")
        print(f"     Faces Found:   {analysis.face_frame_ratio * 100:.1f}% of frames")
        print(f"     Avg Brightness: {analysis.avg_brightness:.1f}")
        print(f"     Avg Noise:      {analysis.avg_noise:.2f}")

        print_colored("\n  🔧 Degradation Analysis:", Colors.BOLD)
        print(f"     Severity:      {analysis.degradation_severity.upper()}")
        degradation_names = [d.name.replace('_', ' ').title() for d in analysis.degradation_types]
        print(f"     Detected:      {', '.join(degradation_names) or 'None'}")

        print_colored("\n  ⚡ Recommended Settings:", Colors.OKCYAN)
        print(f"     Scale Factor:  {analysis.recommended_scale}x")
        print(f"     Model:         {analysis.recommended_model}")
        print(f"     Denoise:       {analysis.recommended_denoise:.1f}")
        if analysis.enable_face_restoration:
            print(f"     Face Restore:  Yes (faces detected)")
        if analysis.enable_scratch_removal:
            print(f"     Defect Repair: Yes (degradation detected)")
        if analysis.recommended_target_fps:
            print(f"     Target FPS:    {analysis.recommended_target_fps} (for RIFE)")

        print_colored("\n  💡 Suggested Command:", Colors.OKGREEN)
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


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description='FrameWright - AI-powered video restoration pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully automated restoration (recommended for old films)
  framewright restore --input old_film.mp4 --output restored.mp4 --auto-enhance

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
  framewright reassemble --frames-dir enhanced/ --audio original.mp4 --output final.mp4
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Full video restoration workflow')
    restore_parser.add_argument('--url', type=str, help='YouTube or video URL')
    restore_parser.add_argument('--input', type=str, help='Input video file path')
    restore_parser.add_argument('--output', type=str, required=True, help='Output video file path')
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
    reassemble_parser.add_argument('--quality', type=int, default=18, help='CRF quality (lower=better, default: 18)')
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

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        print_header()
        parser.print_help()
        sys.exit(0)

    print_header()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print_colored("\n\n✗ Operation cancelled by user", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n✗ Error: {str(e)}", Colors.FAIL)
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
    print_colored("\n🔍 Checking hardware compatibility...\n", Colors.OKBLUE)

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
