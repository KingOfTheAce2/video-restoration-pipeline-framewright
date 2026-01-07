"""Cloud CLI commands for FrameWright.

Provides commands for submitting video restoration jobs to cloud GPU providers
(Vast.ai) with Google Drive storage integration via rclone.

Usage:
    framewright cloud submit --input video.mp4 --scale 4
    framewright cloud status <job_id>
    framewright cloud gpus
    framewright cloud cancel <job_id>
    framewright cloud balance
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Job state persistence file
JOBS_FILE = Path.home() / ".framewright" / "cloud_jobs.json"


def load_vastai_api_key() -> Optional[str]:
    """Load Vast.ai API key from environment or config file.

    Checks in order:
    1. VASTAI_API_KEY environment variable
    2. config/vastai.env file in current directory
    3. ~/.framewright/vastai.env

    Returns:
        API key string or None if not found
    """
    # Check environment variable first
    if os.environ.get("VASTAI_API_KEY"):
        return os.environ["VASTAI_API_KEY"]

    # Check config/vastai.env in current directory
    local_config = Path("config/vastai.env")
    if local_config.exists():
        return _parse_env_file(local_config)

    # Check ~/.framewright/vastai.env
    home_config = Path.home() / ".framewright" / "vastai.env"
    if home_config.exists():
        return _parse_env_file(home_config)

    return None


def _parse_env_file(path: Path) -> Optional[str]:
    """Parse API key from .env file."""
    try:
        content = path.read_text()
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("VASTAI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                if value and value != "your_api_key_here":
                    return value
    except Exception:
        pass
    return None


def save_job(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save job data to persistent storage."""
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

    jobs = {}
    if JOBS_FILE.exists():
        try:
            jobs = json.loads(JOBS_FILE.read_text())
        except Exception:
            pass

    jobs[job_id] = job_data
    JOBS_FILE.write_text(json.dumps(jobs, indent=2, default=str))


def load_jobs() -> Dict[str, Any]:
    """Load all saved jobs."""
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except Exception:
            pass
    return {}


def print_colored(message: str, color: str = "\033[94m"):
    """Print colored message."""
    print(f"{color}{message}\033[0m")


def print_error(message: str):
    """Print error message in red."""
    print_colored(f"Error: {message}", "\033[91m")


def print_success(message: str):
    """Print success message in green."""
    print_colored(message, "\033[92m")


def print_warning(message: str):
    """Print warning message in yellow."""
    print_colored(message, "\033[93m")


def cloud_submit(args) -> int:
    """Submit a video restoration job to Vast.ai cloud.

    Workflow:
    1. Download video from URL if provided
    2. Upload input video to Google Drive
    3. Submit job to Vast.ai with Google Drive URLs
    4. Save job ID for tracking
    """
    import subprocess
    import tempfile
    from .gdrive import GoogleDriveStorage, check_gdrive_configured
    from .vastai import VastAIProvider
    from .base import ProcessingConfig, CloudError

    print_colored("\n=== FrameWright Cloud Submit ===\n", "\033[95m")

    # Validate input - need either --input, --url, or --gdrive-input
    gdrive_input = getattr(args, 'gdrive_input', None)
    if not args.input and not args.url and not gdrive_input:
        print_error("Either --input, --url, or --gdrive-input is required")
        return 1

    # Check Google Drive is configured
    if not check_gdrive_configured(args.storage_remote):
        print_error(f"Google Drive remote '{args.storage_remote}' not configured.")
        print("Run: rclone config")
        return 1

    # Load Vast.ai API key
    api_key = load_vastai_api_key()
    if not api_key:
        print_error("Vast.ai API key not found.")
        print("Set it in config/vastai.env or VASTAI_API_KEY environment variable")
        return 1

    # Handle Google Drive input (skip download and upload)
    temp_dir = None
    skip_upload = False
    if gdrive_input:
        print(f"Using existing Google Drive file: {gdrive_input}")
        input_uri = f"{args.storage_remote}:{gdrive_input}"
        input_path = Path(gdrive_input)  # For job naming
        skip_upload = True

    # Handle URL download
    elif args.url:
        print(f"Downloading from URL: {args.url}")
        temp_dir = tempfile.mkdtemp(prefix="framewright_")
        temp_output = Path(temp_dir) / "downloaded_video.mp4"

        try:
            # Use yt-dlp to download
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(temp_output),
            ]
            # Add cookie authentication if provided
            if hasattr(args, 'cookies_from_browser') and args.cookies_from_browser:
                cmd.extend(["--cookies-from-browser", args.cookies_from_browser])
            if hasattr(args, 'cookies') and args.cookies:
                cmd.extend(["--cookies", args.cookies])
            cmd.append(args.url)
            print("  Running yt-dlp...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print_error(f"Failed to download video: {result.stderr}")
                return 1

            # Find the actual downloaded file (yt-dlp may change extension)
            downloaded_files = list(Path(temp_dir).glob("*"))
            if not downloaded_files:
                print_error("No file downloaded")
                return 1

            input_path = downloaded_files[0]
            print_success(f"  Downloaded: {input_path.name}")

        except FileNotFoundError:
            print_error("yt-dlp not found. Install with: pip install yt-dlp")
            return 1
    else:
        # Validate local input
        input_path = Path(args.input)
        if not input_path.exists():
            print_error(f"Input file not found: {args.input}")
            return 1

    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"fw_{input_path.stem}_{timestamp}"

    print(f"Input: {input_path}")
    print(f"Scale: {args.scale}x")
    print(f"Storage: {args.storage_remote}:")
    print()

    try:
        # Initialize Google Drive storage
        storage = GoogleDriveStorage(
            remote_name=args.storage_remote,
            base_path=args.storage_path,
        )

        if skip_upload:
            # Using existing Google Drive file - skip upload
            print_success(f"  Using: {input_uri}")
        else:
            # Upload input video to Google Drive
            print("Uploading to Google Drive...")
            remote_input = f"input/{job_name}/{input_path.name}"

            def upload_progress(p: float):
                bar = "=" * int(p * 30)
                print(f"\r  [{bar:<30}] {p*100:.0f}%", end="", flush=True)

            input_uri = storage.upload(input_path, remote_input, progress_callback=upload_progress)
            print()  # Newline after progress
            print_success(f"  Uploaded: {input_uri}")

        # Set output path in Google Drive
        output_filename = f"{input_path.stem}_restored_{args.scale}x.mkv"
        remote_output = f"output/{job_name}/{output_filename}"
        output_uri = storage._get_remote_path(remote_output)

        print(f"  Output will be: {output_uri}")
        print()

        # Initialize Vast.ai provider
        print("Connecting to Vast.ai...")
        provider = VastAIProvider(api_key=api_key)
        provider.authenticate()

        balance = provider.get_credit_balance()
        print(f"  Credit balance: ${balance:.2f}")

        if balance < 0.50:
            print_warning("  Warning: Low credit balance!")

        # Create processing config with all options
        config = ProcessingConfig(
            input_path=input_uri,
            output_path=output_uri,
            scale_factor=args.scale,
            model_name=args.model,
            crf=args.quality,
            output_format=getattr(args, 'format', 'mkv'),
            # Frame interpolation
            enable_interpolation=args.enable_rife,
            target_fps=args.target_fps,
            rife_model=getattr(args, 'rife_model', 'rife-v4.6'),
            # Colorization
            enable_colorization=getattr(args, 'colorize', False),
            colorize_model=getattr(args, 'colorize_model', 'deoldify'),
            # Auto enhancement
            enable_auto_enhance=args.auto_enhance,
            no_face_restore=getattr(args, 'no_face_restore', False),
            no_defect_repair=getattr(args, 'no_defect_repair', False),
            scratch_sensitivity=getattr(args, 'scratch_sensitivity', 0.5),
            grain_reduction=getattr(args, 'grain_reduction', 0.3),
            # Deduplication
            enable_deduplicate=args.deduplicate,
            dedup_threshold=args.dedup_threshold,
            # Audio
            enable_audio_enhance=args.audio_enhance,
            # Watermark removal
            enable_watermark_removal=getattr(args, 'remove_watermark', False),
            watermark_mask=getattr(args, 'watermark_mask', None),
            watermark_region=getattr(args, 'watermark_region', None),
            watermark_auto_detect=getattr(args, 'watermark_auto_detect', False),
            # Subtitle removal
            enable_subtitle_removal=getattr(args, 'remove_subtitles', False),
            subtitle_region=getattr(args, 'subtitle_region', 'bottom_third'),
            subtitle_ocr=getattr(args, 'subtitle_ocr', 'auto'),
            subtitle_languages=getattr(args, 'subtitle_languages', None),
            # Cloud settings
            gpu_type=args.gpu,
            max_runtime_minutes=args.timeout,
        )

        # Get cost estimate
        estimate = provider.get_estimated_cost(config)
        print(f"\n  Estimated cost: ${estimate['estimated_total']:.2f}")
        print(f"  GPU: {estimate['gpu_type']} @ ${estimate['cost_per_hour']:.2f}/hr")

        if not args.yes:
            response = input("\n  Proceed? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0

        # Submit job
        print("\nSubmitting job to Vast.ai...")
        job_id = provider.submit_job(config)
        instance_id = provider.get_instance_id(job_id)

        print_success(f"\n  Job submitted: {job_id}")

        # Save job data including Vast.ai instance_id
        save_job(job_id, {
            "job_id": job_id,
            "instance_id": instance_id,  # Vast.ai instance ID for status checks
            "input_file": str(input_path),
            "input_uri": input_uri,
            "output_uri": output_uri,
            "scale": args.scale,
            "status": "submitted",
            "created_at": datetime.now().isoformat(),
            "storage_remote": args.storage_remote,
        })

        print(f"\nTrack progress with:")
        print(f"  framewright cloud status {job_id}")

        if args.wait:
            print("\nWaiting for completion...")
            result = cloud_wait(job_id, provider, storage, args)
            # Cleanup temp directory
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            return result

        # Cleanup temp directory
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        return 0

    except CloudError as e:
        print_error(str(e))
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 1


def cloud_wait(job_id: str, provider, storage, args) -> int:
    """Wait for job completion and download result."""
    from .base import JobState

    poll_interval = 30
    last_status = None

    try:
        while True:
            status = provider.get_job_status(job_id)

            if status.state != last_status:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status.state.name}: {status.message}")
                last_status = status.state

            if status.is_terminal:
                if status.state == JobState.COMPLETED:
                    print_success("\nJob completed successfully!")

                    # Download result
                    if status.output_path:
                        output_local = Path(args.output_dir) / Path(status.output_path).name
                        print(f"\nDownloading result to {output_local}...")

                        storage.download(status.output_path, output_local)
                        print_success(f"  Downloaded: {output_local}")

                    return 0
                else:
                    print_error(f"\nJob failed: {status.error_message}")
                    return 1

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Job continues in cloud.")
        print(f"Check status with: framewright cloud status {job_id}")
        return 0


def cloud_status(args) -> int:
    """Check status of a cloud job."""
    from .vastai import VastAIProvider
    from .base import CloudError, JobState

    api_key = load_vastai_api_key()
    if not api_key:
        print_error("Vast.ai API key not found.")
        return 1

    job_id = args.job_id

    # Load saved job data
    jobs = load_jobs()
    job_data = jobs.get(job_id, {})

    if not job_data:
        print_error(f"Job not found: {job_id}")
        print("Use 'framewright cloud jobs' to list all jobs.")
        return 1

    # Get Vast.ai instance_id from saved data
    instance_id = job_data.get('instance_id')
    if not instance_id:
        print_error(f"No Vast.ai instance ID found for job: {job_id}")
        print("This job may have been created with an older version.")
        return 1

    print(f"\nJob: {job_id}")
    print(f"  Input: {job_data.get('input_file', 'N/A')}")
    print(f"  Scale: {job_data.get('scale', 'N/A')}x")
    print(f"  Created: {job_data.get('created_at', 'N/A')}")
    print()

    try:
        provider = VastAIProvider(api_key=api_key)
        provider.authenticate()

        # Register the job from saved data so provider can track it
        provider.register_job(job_id, instance_id)

        status = provider.get_job_status(job_id)

        # Status indicator
        state_colors = {
            JobState.PENDING: "\033[93m",
            JobState.QUEUED: "\033[93m",
            JobState.PROVISIONING: "\033[94m",
            JobState.RUNNING: "\033[94m",
            JobState.COMPLETED: "\033[92m",
            JobState.FAILED: "\033[91m",
            JobState.CANCELLED: "\033[91m",
        }
        color = state_colors.get(status.state, "\033[0m")
        print(f"Status: {color}{status.state.name}\033[0m")
        print(f"Message: {status.message}")

        if status.progress > 0:
            bar = "=" * int(status.progress * 30)
            print(f"Progress: [{bar:<30}] {status.progress*100:.0f}%")

        if status.metrics:
            print("\nMetrics:")
            for key, value in status.metrics.items():
                print(f"  {key}: {value}")

        if status.error_message:
            print_error(f"\nError: {status.error_message}")

        if status.output_path:
            print(f"\nOutput: {status.output_path}")

        # Update saved job status
        if job_data:
            job_data["status"] = status.state.name
            save_job(job_id, job_data)

        return 0

    except CloudError as e:
        print_error(str(e))
        return 1


def cloud_gpus(args) -> int:
    """List available GPUs on Vast.ai with pricing."""
    from .vastai import VastAIProvider
    from .base import CloudError

    api_key = load_vastai_api_key()
    if not api_key:
        print_error("Vast.ai API key not found.")
        return 1

    print_colored("\n=== Available GPUs on Vast.ai ===\n", "\033[95m")

    try:
        provider = VastAIProvider(api_key=api_key)
        provider.authenticate()

        gpus = provider.list_available_gpus()

        if not gpus:
            print_warning("No GPUs currently available.")
            return 0

        # Print table header
        print(f"{'GPU Type':<20} {'VRAM':<10} {'$/hr':<10} {'Available':<10} {'Location'}")
        print("-" * 70)

        for gpu in gpus[:20]:  # Show top 20
            vram = f"{gpu.vram_gb:.0f} GB"
            price = f"${gpu.price_per_hour:.3f}"
            avail = str(gpu.availability)
            location = gpu.location[:20] if gpu.location else "N/A"
            print(f"{gpu.gpu_type:<20} {vram:<10} {price:<10} {avail:<10} {location}")

        if len(gpus) > 20:
            print(f"\n... and {len(gpus) - 20} more")

        return 0

    except CloudError as e:
        print_error(str(e))
        return 1


def cloud_cancel(args) -> int:
    """Cancel a running cloud job."""
    from .vastai import VastAIProvider
    from .base import CloudError

    api_key = load_vastai_api_key()
    if not api_key:
        print_error("Vast.ai API key not found.")
        return 1

    job_id = args.job_id

    try:
        provider = VastAIProvider(api_key=api_key)
        provider.authenticate()

        print(f"Cancelling job {job_id}...")

        if provider.cancel_job(job_id):
            print_success("Job cancelled.")

            # Update saved job status
            jobs = load_jobs()
            if job_id in jobs:
                jobs[job_id]["status"] = "cancelled"
                save_job(job_id, jobs[job_id])

            return 0
        else:
            print_error("Failed to cancel job.")
            return 1

    except CloudError as e:
        print_error(str(e))
        return 1


def cloud_balance(args) -> int:
    """Check Vast.ai credit balance."""
    from .vastai import VastAIProvider
    from .base import CloudError

    api_key = load_vastai_api_key()
    if not api_key:
        print_error("Vast.ai API key not found.")
        return 1

    try:
        provider = VastAIProvider(api_key=api_key)
        provider.authenticate()

        balance = provider.get_credit_balance()

        print(f"\nVast.ai Credit Balance: ${balance:.2f}")

        if balance < 1.0:
            print_warning("\nLow balance! Add credits at https://vast.ai/billing")

        return 0

    except CloudError as e:
        print_error(str(e))
        return 1


def cloud_jobs(args) -> int:
    """List all saved cloud jobs."""
    jobs = load_jobs()

    if not jobs:
        print("No cloud jobs found.")
        return 0

    print_colored("\n=== Cloud Jobs ===\n", "\033[95m")

    print(f"{'Job ID':<20} {'Status':<12} {'Scale':<8} {'Created'}")
    print("-" * 60)

    for job_id, data in sorted(jobs.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        status = data.get("status", "unknown")
        scale = f"{data.get('scale', '?')}x"
        created = data.get("created_at", "N/A")[:19]

        # Color status
        if status == "completed":
            status = f"\033[92m{status}\033[0m"
        elif status in ("failed", "cancelled"):
            status = f"\033[91m{status}\033[0m"
        elif status in ("running", "submitted"):
            status = f"\033[94m{status}\033[0m"

        print(f"{job_id:<20} {status:<22} {scale:<8} {created}")

    return 0


def cloud_download(args) -> int:
    """Download completed job result from Google Drive."""
    from .gdrive import GoogleDriveStorage
    from .base import StorageError

    job_id = args.job_id
    jobs = load_jobs()

    if job_id not in jobs:
        print_error(f"Job not found: {job_id}")
        return 1

    job_data = jobs[job_id]
    output_uri = job_data.get("output_uri")

    if not output_uri:
        print_error("No output URI found for this job.")
        return 1

    try:
        storage = GoogleDriveStorage(
            remote_name=job_data.get("storage_remote", "gdrive"),
        )

        output_local = Path(args.output) if args.output else Path(output_uri).name

        print(f"Downloading from {output_uri}...")

        def progress(p: float):
            bar = "=" * int(p * 30)
            print(f"\r  [{bar:<30}] {p*100:.0f}%", end="", flush=True)

        storage.download(output_uri, output_local, progress_callback=progress)
        print()

        print_success(f"Downloaded: {output_local}")
        return 0

    except StorageError as e:
        print_error(str(e))
        return 1


def setup_cloud_parser(subparsers) -> None:
    """Set up cloud command parser with subcommands."""
    cloud_parser = subparsers.add_parser(
        'cloud',
        help='Cloud GPU processing (Vast.ai + Google Drive)'
    )

    cloud_subparsers = cloud_parser.add_subparsers(
        dest='cloud_action',
        help='Cloud actions'
    )

    # cloud submit
    submit_parser = cloud_subparsers.add_parser(
        'submit',
        help='Submit video to cloud for processing'
    )
    submit_parser.add_argument('--input', '-i', type=str,
                               help='Input video file (local path)')
    submit_parser.add_argument('--url', '-u', type=str,
                               help='YouTube or video URL to download and process')
    submit_parser.add_argument('--gdrive-input', type=str,
                               help='Path to video already on Google Drive (e.g., "framewright/input/video.mp4")')
    submit_parser.add_argument('--cookies-from-browser', type=str,
                               help='Browser to extract cookies from (chrome, firefox, edge, safari)')
    submit_parser.add_argument('--cookies', type=str,
                               help='Path to cookies.txt file for authentication')
    submit_parser.add_argument('--scale', '-s', type=int, default=4, choices=[2, 4],
                               help='Upscaling factor (default: 4 for archive quality)')
    submit_parser.add_argument('--model', type=str, default='realesrgan-x4plus',
                               help='AI model for enhancement')
    submit_parser.add_argument('--quality', '-q', type=int, default=15,
                               help='Output quality CRF (0-51, lower=better, default: 15 for archive)')
    submit_parser.add_argument('--gpu', type=str, default='RTX_4090',
                               help='Preferred GPU type (default: RTX_4090)')
    submit_parser.add_argument('--timeout', type=int, default=120,
                               help='Max runtime in minutes (default: 120)')
    submit_parser.add_argument('--enable-rife', action='store_true',
                               help='Enable RIFE frame interpolation')
    submit_parser.add_argument('--target-fps', type=float,
                               help='Target FPS for interpolation')
    submit_parser.add_argument('--auto-enhance', action='store_true',
                               help='Enable automatic enhancement features')
    submit_parser.add_argument('--deduplicate', action='store_true',
                               help='Enable frame deduplication (removes duplicate frames)')
    submit_parser.add_argument('--dedup-threshold', type=float, default=0.98,
                               help='Similarity threshold for deduplication (0.9-1.0, default: 0.98)')
    submit_parser.add_argument('--audio-enhance', action='store_true',
                               help='Enable audio enhancement')
    # Output format
    submit_parser.add_argument('--format', type=str, default='mkv',
                               choices=['mkv', 'mp4', 'webm', 'avi', 'mov'],
                               help='Output format (default: mkv)')
    # RIFE model selection
    submit_parser.add_argument('--rife-model', type=str, default='rife-v4.6',
                               choices=['rife-v2.3', 'rife-v4.0', 'rife-v4.6'],
                               help='RIFE model for interpolation (default: rife-v4.6)')
    # Colorization
    submit_parser.add_argument('--colorize', action='store_true',
                               help='Enable AI colorization for black & white footage')
    submit_parser.add_argument('--colorize-model', type=str, default='deoldify',
                               choices=['deoldify', 'ddcolor'],
                               help='Colorization model (default: deoldify)')
    # Defect repair tuning
    submit_parser.add_argument('--no-face-restore', action='store_true',
                               help='Disable automatic face restoration')
    submit_parser.add_argument('--no-defect-repair', action='store_true',
                               help='Disable automatic defect repair')
    submit_parser.add_argument('--scratch-sensitivity', type=float, default=0.5,
                               help='Scratch detection sensitivity 0.0-1.0 (default: 0.5)')
    submit_parser.add_argument('--grain-reduction', type=float, default=0.3,
                               help='Film grain reduction 0.0-1.0 (default: 0.3)')
    # Watermark removal
    submit_parser.add_argument('--remove-watermark', action='store_true',
                               help='Enable watermark removal')
    submit_parser.add_argument('--watermark-mask', type=str,
                               help='Path to watermark mask image')
    submit_parser.add_argument('--watermark-region', type=str,
                               help='Watermark region as x,y,w,h')
    submit_parser.add_argument('--watermark-auto-detect', action='store_true',
                               help='Auto-detect watermark location')
    # Subtitle removal
    submit_parser.add_argument('--remove-subtitles', action='store_true',
                               help='Remove burnt-in subtitles')
    submit_parser.add_argument('--subtitle-region', type=str, default='bottom_third',
                               choices=['bottom_third', 'bottom_quarter', 'top_quarter', 'full_frame'],
                               help='Region to scan for subtitles (default: bottom_third)')
    submit_parser.add_argument('--subtitle-ocr', type=str, default='auto',
                               choices=['auto', 'easyocr', 'tesseract', 'paddleocr'],
                               help='OCR engine for subtitle detection (default: auto)')
    submit_parser.add_argument('--subtitle-languages', type=str,
                               help='Languages for subtitle detection (comma-separated)')
    submit_parser.add_argument('--storage-remote', type=str, default='gdrive',
                               help='rclone remote name (default: gdrive)')
    submit_parser.add_argument('--storage-path', type=str, default='framewright',
                               help='Base path in cloud storage (default: framewright)')
    submit_parser.add_argument('--output-dir', type=str, default='.',
                               help='Local directory for downloaded results')
    submit_parser.add_argument('--wait', '-w', action='store_true',
                               help='Wait for job completion and download result')
    submit_parser.add_argument('--yes', '-y', action='store_true',
                               help='Skip confirmation prompt')
    submit_parser.set_defaults(func=cloud_submit)

    # cloud status
    status_parser = cloud_subparsers.add_parser(
        'status',
        help='Check job status'
    )
    status_parser.add_argument('job_id', type=str, help='Job ID to check')
    status_parser.set_defaults(func=cloud_status)

    # cloud gpus
    gpus_parser = cloud_subparsers.add_parser(
        'gpus',
        help='List available GPUs with pricing'
    )
    gpus_parser.set_defaults(func=cloud_gpus)

    # cloud cancel
    cancel_parser = cloud_subparsers.add_parser(
        'cancel',
        help='Cancel a running job'
    )
    cancel_parser.add_argument('job_id', type=str, help='Job ID to cancel')
    cancel_parser.set_defaults(func=cloud_cancel)

    # cloud balance
    balance_parser = cloud_subparsers.add_parser(
        'balance',
        help='Check Vast.ai credit balance'
    )
    balance_parser.set_defaults(func=cloud_balance)

    # cloud jobs
    jobs_parser = cloud_subparsers.add_parser(
        'jobs',
        help='List all cloud jobs'
    )
    jobs_parser.set_defaults(func=cloud_jobs)

    # cloud download
    download_parser = cloud_subparsers.add_parser(
        'download',
        help='Download completed job result'
    )
    download_parser.add_argument('job_id', type=str, help='Job ID')
    download_parser.add_argument('--output', '-o', type=str,
                                 help='Output file path')
    download_parser.set_defaults(func=cloud_download)
