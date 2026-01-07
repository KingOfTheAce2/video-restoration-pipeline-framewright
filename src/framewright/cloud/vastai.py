"""Vast.ai cloud provider integration for video processing."""

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import (
    AuthenticationError,
    CloudError,
    CloudProvider,
    GPUInfo,
    JobState,
    JobStatus,
    JobSubmissionError,
    ProcessingConfig,
    StorageError,
)


# Default Vast.ai Docker image for FrameWright
FRAMEWRIGHT_VASTAI_IMAGE = os.environ.get(
    "FRAMEWRIGHT_VASTAI_IMAGE", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
)

# GPU type mappings for cost optimization
VASTAI_GPU_RANKING = [
    # (search_term, min_vram, max_price_per_hour)
    ("RTX 4090", 24, 0.50),
    ("RTX 3090", 24, 0.30),
    ("A100", 40, 1.50),
    ("A6000", 48, 0.70),
    ("RTX 4080", 16, 0.35),
    ("RTX 3080", 10, 0.20),
]


@dataclass
class VastAIInstanceInfo:
    """Internal tracking for Vast.ai instances."""

    job_id: str
    instance_id: Optional[int] = None
    offer_id: Optional[int] = None
    config: Optional[ProcessingConfig] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None


class VastAIProvider(CloudProvider):
    """Vast.ai cloud provider for cost-optimized GPU video processing.

    Vast.ai provides access to GPU instances at competitive prices,
    often significantly cheaper than other cloud providers.

    Example:
        >>> provider = VastAIProvider(api_key="your_api_key")
        >>> provider.authenticate()
        >>> # List available GPUs sorted by cost
        >>> gpus = provider.list_available_gpus()
        >>> for gpu in gpus[:5]:
        ...     print(f"{gpu.gpu_type}: ${gpu.price_per_hour}/hr")
        >>> # Submit processing job
        >>> config = ProcessingConfig(
        ...     input_path="s3://bucket/input.mp4",
        ...     output_path="s3://bucket/output.mp4",
        ...     scale_factor=4,
        ... )
        >>> job_id = provider.submit_job(config)
    """

    API_BASE_URL = "https://cloud.vast.ai/api/v0"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        prefer_cheapest: bool = True,
    ):
        """Initialize Vast.ai provider.

        Args:
            api_key: Vast.ai API key. If not provided, reads from
                     VASTAI_API_KEY environment variable.
            api_base_url: Optional custom API base URL.
            docker_image: Docker image for processing (default: framewright/processor).
            prefer_cheapest: If True, automatically select cheapest suitable GPU.
        """
        api_key = api_key or os.environ.get("VASTAI_API_KEY")
        super().__init__(api_key=api_key, api_base_url=api_base_url or self.API_BASE_URL)
        self._docker_image = docker_image or FRAMEWRIGHT_VASTAI_IMAGE
        self._prefer_cheapest = prefer_cheapest
        self._jobs: Dict[str, VastAIInstanceInfo] = {}
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                self._session.headers.update(
                    {
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    }
                )
            except ImportError:
                raise CloudError(
                    "requests library required for Vast.ai. "
                    "Install with: pip install framewright[cloud]"
                )
        return self._session

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Make authenticated API request."""
        session = self._get_session()
        url = f"{self._api_base_url}{endpoint}"

        # Add API key to params
        params = params or {}
        params["api_key"] = self._api_key

        try:
            if method.upper() == "GET":
                response = session.get(url, params=params)
            elif method.upper() == "POST":
                response = session.post(url, json=data, params=params)
            elif method.upper() == "PUT":
                response = session.put(url, json=data, params=params)
            elif method.upper() == "DELETE":
                response = session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json() if response.text else {}

        except Exception as e:
            raise CloudError(f"Vast.ai API request failed: {e}")

    def authenticate(self) -> bool:
        """Authenticate with Vast.ai API.

        Returns:
            True if authentication successful.

        Raises:
            AuthenticationError: If API key is invalid or missing.
        """
        if not self._api_key:
            raise AuthenticationError(
                "Vast.ai API key not provided. Set VASTAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            # Verify API key by fetching user info
            data = self._make_request("GET", "/users/current")

            if not data.get("id"):
                raise AuthenticationError("Invalid Vast.ai API key")

            self._authenticated = True
            self._user_id = data.get("id")
            self._credit_balance = data.get("credit", 0)

            return True

        except CloudError as e:
            raise AuthenticationError(f"Vast.ai authentication failed: {e}")

    def upload_video(
        self,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload video for processing.

        Note: Vast.ai doesn't provide built-in storage. Videos should be
        uploaded to external storage (S3, etc.) and referenced by URL.

        Args:
            local_path: Path to local video file.
            progress_callback: Optional callback for upload progress.

        Returns:
            Remote path/URI for the uploaded file.

        Raises:
            StorageError: Since Vast.ai requires external storage.
        """
        raise StorageError(
            "Vast.ai requires videos to be uploaded to external storage "
            "(S3, GCS, etc.) first. Use a CloudStorageProvider to upload, "
            "then pass the URL to submit_job()."
        )

    def download_result(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download processed video.

        Note: Results should be downloaded from the external storage
        specified in the output_path config.

        Args:
            remote_path: Remote path/URI of the processed file.
            local_path: Local path to save the file.
            progress_callback: Optional callback for download progress.

        Raises:
            StorageError: Since Vast.ai outputs to external storage.
        """
        raise StorageError(
            "Vast.ai outputs to external storage. "
            "Use the appropriate CloudStorageProvider to download from the output_path."
        )

    def find_optimal_offer(
        self,
        gpu_type: str = "RTX_4090",
        min_vram_gb: int = 16,
        max_price_per_hour: float = 1.0,
        min_upload_speed: float = 100,
        min_download_speed: float = 100,
        disk_space_gb: int = 50,
    ) -> Optional[Dict]:
        """Find the optimal (cheapest suitable) GPU offer.

        Args:
            gpu_type: Preferred GPU type (used as search term).
            min_vram_gb: Minimum VRAM required.
            max_price_per_hour: Maximum acceptable price per hour.
            min_upload_speed: Minimum upload speed in Mbps.
            min_download_speed: Minimum download speed in Mbps.
            disk_space_gb: Required disk space.

        Returns:
            Best offer dict or None if no suitable offers.
        """
        try:
            # Build query for GPU search
            query = {
                "verified": {"eq": True},
                "external": {"eq": False},
                "rentable": {"eq": True},
                "disk_space": {"gte": disk_space_gb},
                "gpu_ram": {"gte": min_vram_gb * 1024},  # Convert to MB
                "inet_up": {"gte": min_upload_speed},
                "inet_down": {"gte": min_download_speed},
                "dph_total": {"lte": max_price_per_hour},
                # Normalize GPU name (RTX_4090 -> RTX 4090)
                "gpu_name": {"eq": gpu_type.replace("_", " ")},
            }

            data = self._make_request(
                "GET",
                "/bundles",
                params={
                    "q": json.dumps(query),
                    "order": "dph_total",  # Sort by price
                    "type": "on-demand",
                },
            )

            offers = data.get("offers", [])

            if not offers:
                # Try without GPU type filter
                del query["gpu_name"]
                data = self._make_request(
                    "GET",
                    "/bundles",
                    params={
                        "q": json.dumps(query),
                        "order": "dph_total",
                        "type": "on-demand",
                    },
                )
                offers = data.get("offers", [])

            if offers:
                return offers[0]  # Return cheapest suitable offer

            return None

        except Exception as e:
            raise CloudError(f"Failed to search offers: {e}")

    def submit_job(self, config: ProcessingConfig) -> str:
        """Submit a video processing job to Vast.ai.

        Args:
            config: Processing configuration.

        Returns:
            Job ID for tracking.

        Raises:
            JobSubmissionError: If job submission fails.
        """
        if not self._authenticated:
            self.authenticate()

        job_id = f"fw_{uuid.uuid4().hex[:12]}"

        try:
            # Find optimal offer
            min_vram = 16 if config.scale_factor == 2 else 24
            offer = self.find_optimal_offer(
                gpu_type=config.gpu_type,
                min_vram_gb=min_vram,
                max_price_per_hour=2.0,
                disk_space_gb=100,
            )

            if not offer:
                raise JobSubmissionError(
                    f"No suitable GPU offers found for {config.gpu_type} with {min_vram}GB VRAM"
                )

            offer_id = offer["id"]

            # Get rclone config for Google Drive access
            import base64
            rclone_config_path = Path.home() / ".config" / "rclone" / "rclone.conf"
            rclone_config_b64 = ""
            if rclone_config_path.exists():
                rclone_config_b64 = base64.b64encode(
                    rclone_config_path.read_bytes()
                ).decode()

            # Prepare environment variables for the container
            env_vars = {
                "FRAMEWRIGHT_JOB_ID": job_id,
                "FRAMEWRIGHT_CONFIG": json.dumps(config.to_dict()),
                "INPUT_PATH": config.input_path,
                "OUTPUT_PATH": config.output_path,
                "RCLONE_CONFIG_B64": rclone_config_b64,
                # Use PyTorch backend (CUDA) instead of ncnn-vulkan (Vulkan)
                # PyTorch works reliably in Docker; Vulkan often doesn't
                "FRAMEWRIGHT_BACKEND": "pytorch",
            }

            # Build restoration command with all options
            output_ext = config.output_format or "mkv"

            # Determine output mode based on frame options
            if config.frames_only or config.save_frames:
                # Use output-dir mode to preserve frames
                restore_cmd = (
                    f"framewright restore "
                    f"--input /workspace/input.mp4 "
                    f"--output-dir /workspace/project "
                    f"--format {output_ext} "
                    f"--scale {config.scale_factor} "
                    f"--model {config.model_name} "
                    f"--quality {config.crf}"
                )
            else:
                # Standard mode - output video directly
                restore_cmd = (
                    f"framewright restore "
                    f"--input /workspace/input.mp4 "
                    f"--output /workspace/output.{output_ext} "
                    f"--format {output_ext} "
                    f"--scale {config.scale_factor} "
                    f"--model {config.model_name} "
                    f"--quality {config.crf}"
                )
            # Frame interpolation
            if config.enable_interpolation:
                restore_cmd += f" --enable-rife --rife-model {config.rife_model}"
                if config.target_fps:
                    restore_cmd += f" --target-fps {config.target_fps}"
            # Colorization
            if config.enable_colorization:
                restore_cmd += f" --colorize --colorize-model {config.colorize_model}"
            # Auto enhancement
            if config.enable_auto_enhance:
                restore_cmd += " --auto-enhance"
            if config.no_face_restore:
                restore_cmd += " --no-face-restore"
            if config.no_defect_repair:
                restore_cmd += " --no-defect-repair"
            if config.scratch_sensitivity != 0.5:
                restore_cmd += f" --scratch-sensitivity {config.scratch_sensitivity}"
            if config.grain_reduction != 0.3:
                restore_cmd += f" --grain-reduction {config.grain_reduction}"
            # Deduplication
            if config.enable_deduplicate:
                restore_cmd += f" --deduplicate --dedup-threshold {config.dedup_threshold}"
            # Audio
            if config.enable_audio_enhance:
                restore_cmd += " --audio-enhance"
            # Watermark removal
            if config.enable_watermark_removal:
                restore_cmd += " --remove-watermark"
                if config.watermark_mask:
                    restore_cmd += f" --watermark-mask {config.watermark_mask}"
                if config.watermark_region:
                    restore_cmd += f" --watermark-region {config.watermark_region}"
                if config.watermark_auto_detect:
                    restore_cmd += " --watermark-auto-detect"
            # Subtitle removal
            if config.enable_subtitle_removal:
                restore_cmd += f" --remove-subtitles --subtitle-region {config.subtitle_region}"
                restore_cmd += f" --subtitle-ocr {config.subtitle_ocr}"
                if config.subtitle_languages:
                    restore_cmd += f" --subtitle-languages {config.subtitle_languages}"

            # Derive frames output path from video output path
            # e.g., "gdrive:framewright/output/video.mp4" -> "gdrive:framewright/output/video_frames/"
            output_base = config.output_path.rsplit(".", 1)[0] if "." in config.output_path else config.output_path
            frames_output_path = f"{output_base}_frames/"

            # Build upload commands based on mode
            if config.frames_only:
                # Only upload frames (unique frames if dedup enabled)
                upload_section = f'''
echo "=== Uploading enhanced frames to Google Drive ==="
# Find frames directory (enhanced or frames)
if [ -d "/workspace/project/enhanced" ]; then
    FRAMES_DIR="/workspace/project/enhanced"
elif [ -d "/workspace/project/frames" ]; then
    FRAMES_DIR="/workspace/project/frames"
else
    FRAMES_DIR="/workspace/project"
fi
echo "Uploading frames from $FRAMES_DIR"
rclone copy "$FRAMES_DIR" "{frames_output_path}" --progress
echo "Frames uploaded to: {frames_output_path}"
'''
            elif config.save_frames:
                # Upload both video AND frames
                upload_section = f'''
echo "=== Uploading enhanced frames to Google Drive ==="
if [ -d "/workspace/project/enhanced" ]; then
    FRAMES_DIR="/workspace/project/enhanced"
elif [ -d "/workspace/project/frames" ]; then
    FRAMES_DIR="/workspace/project/frames"
else
    FRAMES_DIR="/workspace/project"
fi
echo "Uploading frames from $FRAMES_DIR"
rclone copy "$FRAMES_DIR" "{frames_output_path}" --progress
echo "Frames uploaded to: {frames_output_path}"

echo "=== Uploading video to Google Drive ==="
# Find the output video in project dir
VIDEO_FILE=$(find /workspace/project -maxdepth 1 -name "*.{output_ext}" | head -1)
if [ -n "$VIDEO_FILE" ]; then
    rclone copyto "$VIDEO_FILE" "{config.output_path}" --progress
    echo "Video uploaded to: {config.output_path}"
else
    echo "Warning: No video file found to upload"
fi
'''
            else:
                # Standard mode - just upload video
                upload_section = f'''
echo "=== Uploading result to Google Drive ==="
rclone copyto /workspace/output.{output_ext} "{config.output_path}" --progress
'''

            # Create startup script
            onstart_script = f"""#!/bin/bash
set -e
cd /workspace

echo "=== Installing dependencies ==="
apt-get update && apt-get install -y ffmpeg rclone wget unzip

echo "=== Configuring rclone ==="
mkdir -p ~/.config/rclone
echo "$RCLONE_CONFIG_B64" | base64 -d > ~/.config/rclone/rclone.conf

echo "=== Installing Real-ESRGAN (PyTorch/CUDA) ==="
# Use PyTorch Real-ESRGAN which works with CUDA (already in PyTorch Docker)
# This avoids Vulkan issues that occur in Docker containers
pip install realesrgan basicsr

echo "=== Installing FrameWright ==="
pip install git+https://github.com/KingOfTheAce2/video-restoration-pipeline-framewright.git yt-dlp

# Force PyTorch backend (uses CUDA instead of Vulkan)
export FRAMEWRIGHT_BACKEND=pytorch

echo "=== Downloading input from Google Drive ==="
rclone copyto "{config.input_path}" /workspace/input.mp4 --progress

echo "=== Starting restoration ==="
{restore_cmd}
{upload_section}
echo "=== Done! Shutting down instance ==="
# Auto-destroy to avoid idle billing
vastai destroy instance $VAST_CONTAINERLABEL || true
"""

            # Create instance from offer
            create_data = {
                "client_id": "me",
                "image": self._docker_image,
                "env": env_vars,
                "disk": 100,
                "runtype": "ssh",
                "onstart": onstart_script,
            }

            response = self._make_request(
                "PUT",
                f"/asks/{offer_id}/",
                data=create_data,
            )

            instance_id = response.get("new_contract")
            if not instance_id:
                raise JobSubmissionError(f"No instance ID in response: {response}")

            self._jobs[job_id] = VastAIInstanceInfo(
                job_id=job_id,
                instance_id=instance_id,
                offer_id=offer_id,
                config=config,
            )

            # Return job_id with instance_id accessible
            self._last_instance_id = instance_id
            return job_id

        except Exception as e:
            if isinstance(e, JobSubmissionError):
                raise
            raise JobSubmissionError(f"Failed to submit Vast.ai job: {e}")

    def get_instance_id(self, job_id: str) -> Optional[str]:
        """Get the Vast.ai instance ID for a job."""
        if job_id in self._jobs:
            return self._jobs[job_id].instance_id
        return getattr(self, '_last_instance_id', None)

    def register_job(self, job_id: str, instance_id: str, config: Optional[ProcessingConfig] = None):
        """Register a job from saved data."""
        self._jobs[job_id] = VastAIInstanceInfo(
            job_id=job_id,
            instance_id=instance_id,
            offer_id="",
            config=config,
        )

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the current status of a job.

        Args:
            job_id: Job ID to check.

        Returns:
            Current job status.

        Raises:
            CloudError: If status check fails.
        """
        if job_id not in self._jobs:
            raise CloudError(f"Unknown job ID: {job_id}")

        job_info = self._jobs[job_id]

        try:
            # Get instance status
            data = self._make_request("GET", f"/instances/{job_info.instance_id}/")

            actual_status = data.get("actual_status", "unknown")
            intended_status = data.get("intended_status", "unknown")

            # Map Vast.ai status to our JobState
            state_mapping = {
                "running": JobState.RUNNING,
                "loading": JobState.PROVISIONING,
                "exited": JobState.COMPLETED,
                "created": JobState.QUEUED,
                "starting": JobState.PROVISIONING,
                "stopping": JobState.CANCELLED,
            }

            job_state = state_mapping.get(actual_status, JobState.PENDING)

            # Check if job completed based on exit code
            if actual_status == "exited":
                exit_code = data.get("exit_code", -1)
                if exit_code == 0:
                    job_state = JobState.COMPLETED
                else:
                    job_state = JobState.FAILED

            # Get SSH info for logs
            ssh_host = data.get("ssh_host")
            ssh_port = data.get("ssh_port")
            if ssh_host and ssh_port:
                job_info.ssh_host = ssh_host
                job_info.ssh_port = ssh_port

            # Calculate cost so far
            start_time = data.get("start_date")
            dph = data.get("dph_total", 0)

            metrics = {
                "instance_id": job_info.instance_id,
                "gpu": data.get("gpu_name", "Unknown"),
                "dph": dph,
                "ssh_host": ssh_host,
                "ssh_port": ssh_port,
            }

            return JobStatus(
                job_id=job_id,
                state=job_state,
                progress=0.0,  # Would need to SSH in to get actual progress
                message=f"Instance status: {actual_status}",
                output_path=job_info.config.output_path if job_state == JobState.COMPLETED else None,
                error_message=f"Exit code: {data.get('exit_code')}" if job_state == JobState.FAILED else None,
                metrics=metrics,
            )

        except Exception as e:
            raise CloudError(f"Failed to get job status: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job by destroying the instance.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        if job_id not in self._jobs:
            return False

        job_info = self._jobs[job_id]

        try:
            self._make_request("DELETE", f"/instances/{job_info.instance_id}/")
            return True
        except Exception:
            return False

    def list_available_gpus(self) -> List[GPUInfo]:
        """List available GPU instances sorted by price.

        Returns:
            List of available GPU configurations with pricing.
        """
        try:
            # Query for rentable GPUs
            query = {
                "verified": {"eq": True},
                "external": {"eq": False},
                "rentable": {"eq": True},
                "gpu_ram": {"gte": 8 * 1024},  # At least 8GB VRAM
            }

            data = self._make_request(
                "GET",
                "/bundles",
                params={
                    "q": json.dumps(query),
                    "order": "dph_total",
                    "type": "on-demand",
                    "limit": 100,
                },
            )

            gpus = []
            seen_gpus = set()

            for offer in data.get("offers", []):
                gpu_name = offer.get("gpu_name", "Unknown")
                key = (gpu_name, offer.get("num_gpus", 1))

                if key in seen_gpus:
                    continue
                seen_gpus.add(key)

                gpus.append(
                    GPUInfo(
                        gpu_type=gpu_name,
                        vram_gb=offer.get("gpu_ram", 0) / 1024,  # Convert MB to GB
                        price_per_hour=offer.get("dph_total", 0),
                        availability=offer.get("num_gpus", 1),
                        location=offer.get("geolocation", ""),
                        provider_id=str(offer.get("id", "")),
                    )
                )

            return gpus

        except Exception:
            return []

    def get_job_logs(self, job_id: str, tail: int = 100) -> str:
        """Get logs from a job.

        Note: Vast.ai requires SSH access to get container logs.

        Args:
            job_id: Job ID to get logs for.
            tail: Number of lines to return from the end.

        Returns:
            Log output as string or instructions for SSH access.
        """
        if job_id not in self._jobs:
            return f"Unknown job ID: {job_id}"

        job_info = self._jobs[job_id]

        if job_info.ssh_host and job_info.ssh_port:
            return (
                f"To view logs, SSH into the instance:\n"
                f"ssh -p {job_info.ssh_port} root@{job_info.ssh_host}\n\n"
                f"Then run: tail -n {tail} /var/log/framewright.log"
            )
        else:
            return "Instance SSH info not yet available. Job may still be provisioning."

    def get_estimated_cost(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Estimate cost for a processing job.

        Args:
            config: Processing configuration.

        Returns:
            Cost estimate with breakdown.
        """
        # Find a suitable offer to get actual pricing
        min_vram = 16 if config.scale_factor == 2 else 24
        offer = self.find_optimal_offer(
            gpu_type=config.gpu_type,
            min_vram_gb=min_vram,
            max_price_per_hour=5.0,
        )

        if offer:
            dph = offer.get("dph_total", 0.30)
            gpu_name = offer.get("gpu_name", config.gpu_type)
        else:
            dph = 0.30  # Default estimate
            gpu_name = config.gpu_type

        estimated_hours = config.max_runtime_minutes / 60

        return {
            "gpu_type": gpu_name,
            "cost_per_hour": dph,
            "estimated_hours": estimated_hours,
            "estimated_total": dph * estimated_hours,
            "currency": "USD",
            "note": "Based on current market prices. Actual cost may vary.",
        }

    def get_credit_balance(self) -> float:
        """Get current credit balance.

        Returns:
            Credit balance in USD.
        """
        if not self._authenticated:
            self.authenticate()

        try:
            data = self._make_request("GET", "/users/current")
            return data.get("credit", 0)
        except Exception:
            return self._credit_balance if hasattr(self, "_credit_balance") else 0
