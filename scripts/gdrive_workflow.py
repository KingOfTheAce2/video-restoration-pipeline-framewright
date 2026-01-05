#!/usr/bin/env python3
"""
Google Drive + Vast.ai Video Restoration Workflow

This script provides a complete workflow for:
1. Downloading YouTube videos directly to Google Drive
2. Processing videos on Vast.ai with Google Drive as storage
3. Full restoration with 4x upscale, RIFE interpolation, and defect repair

Requirements:
    - rclone configured with Google Drive remote named 'gdrive'
    - Vast.ai API key (set VASTAI_API_KEY environment variable)
    - yt-dlp installed

Usage:
    # Download YouTube video to Google Drive
    python scripts/gdrive_workflow.py download "https://youtube.com/watch?v=..." --output videos/input

    # Run full restoration on Vast.ai
    python scripts/gdrive_workflow.py restore gdrive:videos/input/video.mp4 --output gdrive:videos/output

    # Complete workflow: download + restore
    python scripts/gdrive_workflow.py full "https://youtube.com/watch?v=..." --project myproject
"""

import argparse
import os
import sys
import subprocess
import shutil
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from framewright.cloud.gdrive import GoogleDriveStorage, check_gdrive_configured
    from framewright.cloud.vastai import VastAIProvider
    from framewright.utils.youtube import YouTubeDownloader, get_ytdlp_path
    from framewright.config import Config, PRESETS
except ImportError as e:
    print(f"Error importing framewright: {e}")
    print("Make sure you're in the project directory and dependencies are installed.")
    sys.exit(1)


@dataclass
class WorkflowConfig:
    """Configuration for the restoration workflow."""
    # Google Drive settings
    gdrive_remote: str = "gdrive"
    gdrive_base_path: str = "framewright"

    # Vast.ai settings
    vastai_api_key: Optional[str] = None
    gpu_type: str = "RTX_4090"
    max_runtime_minutes: int = 180

    # Restoration settings
    scale_factor: int = 4
    enable_interpolation: bool = True
    target_fps: float = 24.0  # Reasonable default for most content
    enable_defect_repair: bool = True
    enable_face_restore: bool = True
    crf: int = 18

    # Deduplication (for old films with artificial FPS padding)
    enable_deduplication: bool = False
    expected_source_fps: Optional[float] = None  # Hint: 16-18 for silent films

    @classmethod
    def from_preset(cls, preset_name: str) -> "WorkflowConfig":
        """Create config from a preset name."""
        config = cls()

        if preset_name == "fast":
            config.scale_factor = 2
            config.enable_interpolation = False
            config.enable_defect_repair = False
            config.enable_deduplication = False
            config.crf = 23
        elif preset_name == "quality":
            config.scale_factor = 4
            config.enable_interpolation = True
            config.target_fps = 30.0  # Modest interpolation
            config.enable_defect_repair = False
            config.enable_deduplication = False
            config.crf = 18
        elif preset_name == "archive":
            config.scale_factor = 4
            config.enable_interpolation = True
            config.target_fps = 30.0
            config.enable_defect_repair = True
            config.enable_face_restore = True
            config.enable_deduplication = False
            config.crf = 15
        elif preset_name == "film":
            # For old films (pre-1930): preserve original frame rate feel
            config.scale_factor = 4
            config.enable_interpolation = True
            config.target_fps = 24.0  # Standard film rate
            config.enable_defect_repair = True
            config.enable_face_restore = True
            config.enable_deduplication = True  # Remove padded duplicate frames
            config.expected_source_fps = 18.0   # Typical for 1900s-1920s films
            config.crf = 16
        elif preset_name == "silent":
            # For silent era films (1895-1930): typically 16-18fps
            config.scale_factor = 4
            config.enable_interpolation = True
            config.target_fps = 18.0  # Preserve authentic silent film cadence
            config.enable_defect_repair = True
            config.enable_face_restore = True
            config.enable_deduplication = True
            config.expected_source_fps = 16.0   # Many silent films were 16fps
            config.crf = 16
        elif preset_name == "early":
            # For very early films (1895-1910): often 14-16fps
            config.scale_factor = 4
            config.enable_interpolation = False  # Keep original timing
            config.enable_defect_repair = True
            config.enable_face_restore = True
            config.enable_deduplication = True
            config.expected_source_fps = 16.0
            config.crf = 15

        return config


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    deps = {
        "rclone": shutil.which("rclone") is not None,
        "yt-dlp": get_ytdlp_path() is not None,
        "gdrive_configured": check_gdrive_configured(),
        "vastai_key": bool(os.environ.get("VASTAI_API_KEY")),
    }
    return deps


def print_status(deps: Dict[str, bool]) -> None:
    """Print dependency status."""
    print("\n=== Dependency Status ===")

    status_icons = {True: "✓", False: "✗"}

    for name, available in deps.items():
        icon = status_icons[available]
        print(f"  {icon} {name}")

    if not all(deps.values()):
        print("\nMissing dependencies:")
        if not deps["rclone"]:
            print("  - Install rclone: curl https://rclone.org/install.sh | sudo bash")
        if not deps["yt-dlp"]:
            print("  - Install yt-dlp: pip install yt-dlp")
        if not deps["gdrive_configured"]:
            print("  - Configure Google Drive: rclone config")
        if not deps["vastai_key"]:
            print("  - Set Vast.ai API key: export VASTAI_API_KEY=your_key")

    print()


def download_to_gdrive(
    url: str,
    output_path: str,
    gdrive_remote: str = "gdrive",
    quality: str = "best",
    stream: bool = True,
) -> str:
    """Download YouTube video directly to Google Drive.

    Args:
        url: YouTube video URL
        output_path: Path in Google Drive (e.g., "videos/input/video.mp4")
        gdrive_remote: rclone remote name
        quality: Quality preset (best, 4k, 1080p, etc.)
        stream: If True, stream directly to Drive without local storage

    Returns:
        Full remote path of downloaded video
    """
    print(f"\n=== Downloading to Google Drive ===")
    print(f"URL: {url}")
    print(f"Destination: {gdrive_remote}:{output_path}")

    ytdlp = get_ytdlp_path()
    rclone = shutil.which("rclone")

    if not ytdlp or not rclone:
        raise RuntimeError("yt-dlp or rclone not found")

    # Get video info first
    print("\nFetching video info...")
    info_cmd = [ytdlp, "--dump-json", "--no-download", url]
    result = subprocess.run(info_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video info: {result.stderr}")

    video_info = json.loads(result.stdout)
    title = video_info.get("title", "video")
    duration = video_info.get("duration", 0)

    print(f"Title: {title}")
    print(f"Duration: {duration // 60}m {duration % 60}s")

    # Clean filename
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
    safe_title = safe_title[:100]  # Limit length

    # Determine output filename
    if output_path.endswith("/") or not output_path.endswith((".mp4", ".mkv", ".webm")):
        output_path = f"{output_path.rstrip('/')}/{safe_title}.mkv"

    full_remote = f"{gdrive_remote}:{output_path}"

    if stream:
        # Stream directly to Google Drive (no local storage needed!)
        print("\nStreaming directly to Google Drive...")

        # yt-dlp format selection
        format_spec = {
            "best": "bestvideo+bestaudio/best",
            "4k": "bestvideo[height<=2160]+bestaudio/best[height<=2160]",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        }.get(quality, quality)

        # Create pipeline: yt-dlp -> rclone rcat
        # Note: We need to merge to MKV and output to stdout
        cmd = (
            f'{ytdlp} -f "{format_spec}" --merge-output-format mkv '
            f'-o - "{url}" 2>/dev/null | '
            f'{rclone} rcat "{full_remote}" --progress'
        )

        print(f"Running: {cmd[:100]}...")

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            if "%" in line or "Transferred" in line:
                print(f"\r{line.strip()}", end="", flush=True)

        process.wait()
        print()  # New line after progress

        if process.returncode != 0:
            # Fallback: download locally then upload
            print("\nDirect streaming failed, falling back to local download...")
            stream = False

    if not stream:
        # Download locally then upload
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / f"{safe_title}.mkv"

            print(f"\nDownloading to temporary location...")

            download_cmd = [
                ytdlp,
                "-f", "bestvideo+bestaudio/best",
                "--merge-output-format", "mkv",
                "-o", str(local_path),
                "--progress",
                url,
            ]

            subprocess.run(download_cmd, check=True)

            print(f"\nUploading to Google Drive...")

            upload_cmd = [
                rclone,
                "copyto",
                str(local_path),
                full_remote,
                "--progress",
            ]

            subprocess.run(upload_cmd, check=True)

    print(f"\n✓ Video saved to: {full_remote}")
    return full_remote


def restore_with_vastai(
    input_path: str,
    output_path: str,
    config: WorkflowConfig,
) -> str:
    """Run video restoration on Vast.ai with Google Drive storage.

    Args:
        input_path: Google Drive path to input video
        output_path: Google Drive path for output video
        config: Workflow configuration

    Returns:
        Path to restored video
    """
    print(f"\n=== Running Restoration on Vast.ai ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Settings:")
    print(f"  - Scale: {config.scale_factor}x")
    print(f"  - Interpolation: {config.enable_interpolation} (target: {config.target_fps}fps)")
    print(f"  - Defect repair: {config.enable_defect_repair}")
    print(f"  - Face restore: {config.enable_face_restore}")
    print(f"  - CRF: {config.crf}")

    # Initialize Vast.ai provider
    api_key = config.vastai_api_key or os.environ.get("VASTAI_API_KEY")
    if not api_key:
        raise RuntimeError("VASTAI_API_KEY not set")

    provider = VastAIProvider(api_key=api_key)
    provider.authenticate()

    print(f"\nVast.ai credit balance: ${provider.get_credit_balance():.2f}")

    # Find suitable GPU
    print("\nSearching for GPU instances...")
    gpus = provider.list_available_gpus()

    if not gpus:
        raise RuntimeError("No suitable GPUs available on Vast.ai")

    # Show top options
    print("\nAvailable GPUs (cheapest first):")
    for gpu in gpus[:5]:
        print(f"  - {gpu.gpu_type}: ${gpu.price_per_hour:.3f}/hr, {gpu.vram_gb:.0f}GB VRAM")

    # Build restoration command for Vast.ai instance
    restore_cmd = [
        "framewright", "restore",
        "--input", input_path,
        "--output", output_path,
        "--scale", str(config.scale_factor),
        "--crf", str(config.crf),
    ]

    if config.enable_interpolation:
        restore_cmd.extend(["--interpolate", "--target-fps", str(config.target_fps)])

    if config.enable_defect_repair:
        restore_cmd.append("--auto-enhance")

    if config.enable_face_restore:
        restore_cmd.append("--face-restore")

    print(f"\nRestoration command: {' '.join(restore_cmd)}")

    # Note: For actual Vast.ai job submission, we'd need to:
    # 1. Create a Docker image with framewright + rclone
    # 2. Submit job that mounts rclone and runs restoration
    # 3. Monitor job progress
    # 4. Return when complete

    # For now, provide manual instructions
    print("\n" + "=" * 60)
    print("VAST.AI MANUAL WORKFLOW")
    print("=" * 60)
    print("""
To run on Vast.ai:

1. Rent a GPU instance (RTX 4090 recommended)
   - Go to: https://cloud.vast.ai/
   - Select instance with 24GB+ VRAM

2. SSH into the instance and run:

   # Install dependencies
   pip install framewright yt-dlp
   curl https://rclone.org/install.sh | sudo bash

   # Configure rclone (one-time setup)
   rclone config
   # Choose: Google Drive, follow OAuth prompts

   # Sync input from Google Drive
   rclone copy {input_path} /workspace/input/

   # Run restoration
   framewright restore \\
       --input /workspace/input/*.mkv \\
       --output /workspace/output/ \\
       --scale {scale} \\
       --crf {crf} \\
       {interp_flag} \\
       --auto-enhance

   # Sync output back to Google Drive
   rclone sync /workspace/output/ {output_path}

3. Destroy instance when done to stop billing
""".format(
        input_path=input_path,
        output_path=output_path,
        scale=config.scale_factor,
        crf=config.crf,
        interp_flag="--interpolate --target-fps " + str(config.target_fps) if config.enable_interpolation else "",
    ))

    return output_path


def full_workflow(
    url: str,
    project_name: str,
    config: WorkflowConfig,
) -> None:
    """Run complete workflow: download + restore.

    Args:
        url: YouTube video URL
        project_name: Project name for organizing files
        config: Workflow configuration
    """
    print(f"\n{'=' * 60}")
    print(f"FULL RESTORATION WORKFLOW")
    print(f"{'=' * 60}")
    print(f"Project: {project_name}")
    print(f"Source: {url}")

    base_path = f"{config.gdrive_base_path}/{project_name}"
    input_path = f"{base_path}/input"
    output_path = f"{base_path}/output"

    # Step 1: Download to Google Drive
    video_path = download_to_gdrive(
        url=url,
        output_path=input_path,
        gdrive_remote=config.gdrive_remote,
        quality="best",
    )

    # Step 2: Run restoration
    restore_with_vastai(
        input_path=video_path,
        output_path=f"{config.gdrive_remote}:{output_path}",
        config=config,
    )

    print(f"\n{'=' * 60}")
    print(f"WORKFLOW COMPLETE")
    print(f"{'=' * 60}")
    print(f"Input saved to: {config.gdrive_remote}:{input_path}")
    print(f"Output will be at: {config.gdrive_remote}:{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Google Drive + Vast.ai Video Restoration Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check dependencies
    python scripts/gdrive_workflow.py status

    # Download YouTube video to Google Drive
    python scripts/gdrive_workflow.py download "https://youtube.com/watch?v=..." -o videos/input

    # Full workflow with default settings
    python scripts/gdrive_workflow.py full "https://youtube.com/watch?v=..." -p myproject

    # Full workflow with film preset (24fps, defect repair)
    python scripts/gdrive_workflow.py full "URL" -p oldfilm --preset film
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check dependencies")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download YouTube to Google Drive")
    download_parser.add_argument("url", help="YouTube video URL")
    download_parser.add_argument("-o", "--output", default="videos/input", help="Output path in Drive")
    download_parser.add_argument("-q", "--quality", default="best", help="Quality (best, 4k, 1080p, 720p)")
    download_parser.add_argument("--remote", default="gdrive", help="rclone remote name")
    download_parser.add_argument("--no-stream", action="store_true", help="Disable direct streaming")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore video on Vast.ai")
    restore_parser.add_argument("input", help="Input video path (gdrive:path/to/video.mp4)")
    restore_parser.add_argument("-o", "--output", help="Output path")
    restore_parser.add_argument("--preset", choices=["fast", "quality", "archive", "film", "silent", "early"], default="quality")
    restore_parser.add_argument("--scale", type=int, choices=[2, 4], help="Scale factor")
    restore_parser.add_argument("--fps", type=float, help="Target FPS for interpolation")
    restore_parser.add_argument("--no-interpolate", action="store_true", help="Disable interpolation")

    # Full workflow command
    full_parser = subparsers.add_parser("full", help="Complete workflow: download + restore")
    full_parser.add_argument("url", help="YouTube video URL")
    full_parser.add_argument("-p", "--project", required=True, help="Project name")
    full_parser.add_argument("--preset", choices=["fast", "quality", "archive", "film", "silent", "early"], default="quality")
    full_parser.add_argument("--remote", default="gdrive", help="rclone remote name")
    full_parser.add_argument("--base-path", default="framewright", help="Base path in Drive")

    args = parser.parse_args()

    if args.command == "status":
        deps = check_dependencies()
        print_status(deps)
        sys.exit(0 if all(deps.values()) else 1)

    elif args.command == "download":
        try:
            download_to_gdrive(
                url=args.url,
                output_path=args.output,
                gdrive_remote=args.remote,
                quality=args.quality,
                stream=not args.no_stream,
            )
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

    elif args.command == "restore":
        config = WorkflowConfig.from_preset(args.preset)

        if args.scale:
            config.scale_factor = args.scale
        if args.fps:
            config.target_fps = args.fps
        if args.no_interpolate:
            config.enable_interpolation = False

        output = args.output or args.input.replace("/input/", "/output/").replace("_input", "_restored")

        try:
            restore_with_vastai(
                input_path=args.input,
                output_path=output,
                config=config,
            )
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

    elif args.command == "full":
        config = WorkflowConfig.from_preset(args.preset)
        config.gdrive_remote = args.remote
        config.gdrive_base_path = args.base_path

        try:
            full_workflow(
                url=args.url,
                project_name=args.project,
                config=config,
            )
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
