"""Base classes and types for cloud processing backend."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime


class CloudError(Exception):
    """Base exception for cloud processing errors."""

    pass


class AuthenticationError(CloudError):
    """Raised when API authentication fails."""

    pass


class JobSubmissionError(CloudError):
    """Raised when job submission fails."""

    pass


class JobExecutionError(CloudError):
    """Raised when job execution fails."""

    pass


class StorageError(CloudError):
    """Raised when storage operations fail."""

    pass


class JobState(Enum):
    """Possible states for a cloud processing job."""

    PENDING = auto()
    QUEUED = auto()
    PROVISIONING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class JobStatus:
    """Status information for a cloud processing job.

    Attributes:
        job_id: Unique identifier for the job.
        state: Current state of the job.
        progress: Processing progress as percentage (0-100).
        message: Human-readable status message.
        created_at: Timestamp when job was created.
        started_at: Timestamp when job started running.
        completed_at: Timestamp when job completed.
        output_path: Path to output file when completed.
        error_message: Error message if job failed.
        metrics: Additional job metrics (frames processed, fps, etc.).
    """

    job_id: str
    state: JobState
    progress: float = 0.0
    message: str = ""
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
        )

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.state in (JobState.RUNNING, JobState.PROVISIONING)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds if available."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "job_id": self.job_id,
            "state": self.state.name,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


@dataclass
class ProcessingConfig:
    """Configuration for cloud processing jobs.

    Attributes:
        input_path: Path or URL to input video (local, s3://, gs://, etc.).
        output_path: Path for output video (local or cloud storage URI).
        scale_factor: Upscaling factor (2 or 4).
        model_name: AI model to use for enhancement.
        crf: Video encoding quality (0-51, lower is better).
        enable_interpolation: Enable RIFE frame interpolation.
        target_fps: Target FPS for interpolation.
        enable_colorization: Enable AI colorization.
        enable_auto_enhance: Enable automatic enhancement features.
        gpu_type: Preferred GPU type (e.g., RTX_4090, A100).
        max_runtime_minutes: Maximum runtime before timeout.
        priority: Job priority (low, medium, high).
        extra_args: Additional processor-specific arguments.
    """

    input_path: str
    output_path: str
    scale_factor: int = 4  # Archive quality default
    model_name: str = "realesrgan-x4plus"
    crf: int = 15  # Archive quality default (lower = better)
    output_format: str = "mkv"
    # Frame interpolation
    enable_interpolation: bool = False
    target_fps: Optional[float] = None
    rife_model: str = "rife-v4.6"
    # Colorization
    enable_colorization: bool = False
    colorize_model: str = "deoldify"
    # Auto enhancement
    enable_auto_enhance: bool = False
    no_face_restore: bool = False
    no_defect_repair: bool = False
    scratch_sensitivity: float = 0.5
    grain_reduction: float = 0.3
    # Deduplication
    enable_deduplicate: bool = False
    dedup_threshold: float = 0.98
    # Audio
    enable_audio_enhance: bool = False
    # Watermark removal
    enable_watermark_removal: bool = False
    watermark_mask: Optional[str] = None
    watermark_region: Optional[str] = None
    watermark_auto_detect: bool = False
    # Subtitle removal
    enable_subtitle_removal: bool = False
    subtitle_region: str = "bottom_third"
    subtitle_ocr: str = "auto"
    subtitle_languages: Optional[str] = None
    # Cloud settings
    gpu_type: str = "RTX_4090"
    max_runtime_minutes: int = 120
    priority: str = "medium"
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "scale_factor": self.scale_factor,
            "model_name": self.model_name,
            "crf": self.crf,
            "output_format": self.output_format,
            "enable_interpolation": self.enable_interpolation,
            "target_fps": self.target_fps,
            "rife_model": self.rife_model,
            "enable_colorization": self.enable_colorization,
            "colorize_model": self.colorize_model,
            "enable_auto_enhance": self.enable_auto_enhance,
            "no_face_restore": self.no_face_restore,
            "no_defect_repair": self.no_defect_repair,
            "scratch_sensitivity": self.scratch_sensitivity,
            "grain_reduction": self.grain_reduction,
            "enable_deduplicate": self.enable_deduplicate,
            "dedup_threshold": self.dedup_threshold,
            "enable_audio_enhance": self.enable_audio_enhance,
            "enable_watermark_removal": self.enable_watermark_removal,
            "watermark_mask": self.watermark_mask,
            "watermark_region": self.watermark_region,
            "watermark_auto_detect": self.watermark_auto_detect,
            "enable_subtitle_removal": self.enable_subtitle_removal,
            "subtitle_region": self.subtitle_region,
            "subtitle_ocr": self.subtitle_ocr,
            "subtitle_languages": self.subtitle_languages,
            "gpu_type": self.gpu_type,
            "max_runtime_minutes": self.max_runtime_minutes,
            "priority": self.priority,
            **self.extra_args,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Create from dictionary."""
        known_keys = {
            "input_path",
            "output_path",
            "scale_factor",
            "model_name",
            "crf",
            "output_format",
            "enable_interpolation",
            "target_fps",
            "rife_model",
            "enable_colorization",
            "colorize_model",
            "enable_auto_enhance",
            "no_face_restore",
            "no_defect_repair",
            "scratch_sensitivity",
            "grain_reduction",
            "enable_deduplicate",
            "dedup_threshold",
            "enable_audio_enhance",
            "enable_watermark_removal",
            "watermark_mask",
            "watermark_region",
            "watermark_auto_detect",
            "enable_subtitle_removal",
            "subtitle_region",
            "subtitle_ocr",
            "subtitle_languages",
            "gpu_type",
            "max_runtime_minutes",
            "priority",
        }
        main_args = {k: v for k, v in data.items() if k in known_keys}
        extra_args = {k: v for k, v in data.items() if k not in known_keys}
        return cls(**main_args, extra_args=extra_args)


@dataclass
class GPUInfo:
    """Information about an available GPU instance.

    Attributes:
        gpu_type: GPU model name (e.g., RTX_4090, A100).
        vram_gb: VRAM in gigabytes.
        price_per_hour: Price in USD per hour.
        availability: Number of available instances.
        location: Data center location.
        provider_id: Provider-specific instance ID.
    """

    gpu_type: str
    vram_gb: float
    price_per_hour: float
    availability: int
    location: str = ""
    provider_id: str = ""


class CloudProvider(ABC):
    """Abstract base class for cloud GPU providers.

    Implementations should handle:
    - API authentication
    - GPU pod/instance provisioning
    - Job submission and monitoring
    - Automatic cleanup on completion
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        """Initialize cloud provider.

        Args:
            api_key: API key for authentication. If not provided,
                     will attempt to load from environment or config.
            api_base_url: Optional base URL for API (for testing/custom endpoints).
        """
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._authenticated = False

    @property
    def name(self) -> str:
        """Get provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the cloud provider.

        Returns:
            True if authentication successful.

        Raises:
            AuthenticationError: If authentication fails.
        """
        pass

    @abstractmethod
    def upload_video(
        self,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload video to cloud storage for processing.

        Args:
            local_path: Path to local video file.
            progress_callback: Optional callback for upload progress (0-1).

        Returns:
            Remote path/URI for the uploaded file.

        Raises:
            StorageError: If upload fails.
        """
        pass

    @abstractmethod
    def download_result(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download processed video from cloud storage.

        Args:
            remote_path: Remote path/URI of the processed file.
            local_path: Local path to save the file.
            progress_callback: Optional callback for download progress (0-1).

        Raises:
            StorageError: If download fails.
        """
        pass

    @abstractmethod
    def submit_job(self, config: ProcessingConfig) -> str:
        """Submit a video processing job.

        Args:
            config: Processing configuration.

        Returns:
            Job ID for tracking.

        Raises:
            JobSubmissionError: If job submission fails.
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the current status of a job.

        Args:
            job_id: Job ID to check.

        Returns:
            Current job status.

        Raises:
            CloudError: If status check fails.
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancellation was successful.

        Raises:
            CloudError: If cancellation fails.
        """
        pass

    @abstractmethod
    def list_available_gpus(self) -> List[GPUInfo]:
        """List available GPU instances.

        Returns:
            List of available GPU configurations.
        """
        pass

    @abstractmethod
    def get_job_logs(self, job_id: str, tail: int = 100) -> str:
        """Get logs from a job.

        Args:
            job_id: Job ID to get logs for.
            tail: Number of lines to return from the end.

        Returns:
            Log output as string.
        """
        pass

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[JobStatus], None]] = None,
    ) -> JobStatus:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait (None for no timeout).
            progress_callback: Optional callback for status updates.

        Returns:
            Final job status.

        Raises:
            TimeoutError: If job doesn't complete within timeout.
            JobExecutionError: If job fails.
        """
        import time

        start_time = time.time()

        while True:
            status = self.get_job_status(job_id)

            if progress_callback:
                progress_callback(status)

            if status.is_terminal:
                if status.state == JobState.FAILED:
                    raise JobExecutionError(
                        f"Job {job_id} failed: {status.error_message}"
                    )
                return status

            if timeout and (time.time() - start_time) > timeout:
                self.cancel_job(job_id)
                raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")

            time.sleep(poll_interval)


class CloudStorageProvider(ABC):
    """Abstract base class for cloud storage providers."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
    ):
        """Initialize storage provider.

        Args:
            bucket: Default bucket/container name.
            credentials: Provider-specific credentials.
        """
        self._bucket = bucket
        self._credentials = credentials or {}

    @property
    @abstractmethod
    def scheme(self) -> str:
        """Get the URI scheme for this storage (e.g., 's3', 'gs', 'azure')."""
        pass

    @abstractmethod
    def upload(
        self,
        local_path: Path,
        remote_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload file to cloud storage.

        Args:
            local_path: Local file path.
            remote_path: Remote path within bucket.
            progress_callback: Optional progress callback (0-1).

        Returns:
            Full URI of uploaded file.

        Raises:
            StorageError: If upload fails.
        """
        pass

    @abstractmethod
    def download(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download file from cloud storage.

        Args:
            remote_path: Remote path or full URI.
            local_path: Local destination path.
            progress_callback: Optional progress callback (0-1).

        Raises:
            StorageError: If download fails.
        """
        pass

    @abstractmethod
    def delete(self, remote_path: str) -> bool:
        """Delete file from cloud storage.

        Args:
            remote_path: Remote path or full URI.

        Returns:
            True if deletion successful.
        """
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in cloud storage.

        Args:
            remote_path: Remote path or full URI.

        Returns:
            True if file exists.
        """
        pass

    @abstractmethod
    def list_files(
        self,
        prefix: str = "",
        max_results: int = 1000,
    ) -> List[str]:
        """List files in cloud storage.

        Args:
            prefix: Path prefix to filter by.
            max_results: Maximum number of results.

        Returns:
            List of file paths.
        """
        pass

    def get_uri(self, path: str) -> str:
        """Get full URI for a path.

        Args:
            path: Relative path within bucket.

        Returns:
            Full URI (e.g., s3://bucket/path).
        """
        if path.startswith(f"{self.scheme}://"):
            return path
        bucket = self._bucket or ""
        return f"{self.scheme}://{bucket}/{path.lstrip('/')}"

    def parse_uri(self, uri: str) -> tuple:
        """Parse a URI into bucket and path.

        Args:
            uri: Full URI or path.

        Returns:
            Tuple of (bucket, path).
        """
        if uri.startswith(f"{self.scheme}://"):
            uri = uri[len(f"{self.scheme}://") :]
        parts = uri.split("/", 1)
        bucket = parts[0] if parts else ""
        path = parts[1] if len(parts) > 1 else ""
        return bucket, path
