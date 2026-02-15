"""API client for FrameWright.

This module provides a Python client for interacting with the
FrameWright REST API, supporting both local and remote servers.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class JobStatus:
    """Represents the status of a restoration job."""

    job_id: str
    state: str
    input_path: str
    output_path: str
    total_frames: int
    frames_processed: int
    frames_failed: int
    progress_percent: float
    avg_frame_time_ms: float
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    error_message: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobStatus":
        """Create JobStatus from dictionary.

        Args:
            data: Dictionary with job data

        Returns:
            JobStatus instance
        """
        return cls(
            job_id=data.get("job_id", ""),
            state=data.get("state", "unknown"),
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            total_frames=data.get("total_frames", 0),
            frames_processed=data.get("frames_processed", 0),
            frames_failed=data.get("frames_failed", 0),
            progress_percent=data.get("progress_percent", 0.0),
            avg_frame_time_ms=data.get("avg_frame_time_ms", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            error_message=data.get("error_message"),
        )

    @property
    def is_complete(self) -> bool:
        """Check if job is complete (success or failure)."""
        return self.state in ("completed", "failed", "cancelled")

    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.state == "completed"

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.state == "processing"


@dataclass
class Model:
    """Represents an available model."""

    name: str
    type: str
    scale: int
    loaded: bool
    size_mb: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """Create Model from dictionary."""
        return cls(
            name=data.get("name", ""),
            type=data.get("type", ""),
            scale=data.get("scale", 4),
            loaded=data.get("loaded", False),
            size_mb=data.get("size_mb"),
        )


@dataclass
class Preset:
    """Represents a restoration preset."""

    name: str
    description: str
    recommended_for: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        """Create Preset from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            recommended_for=data.get("recommended_for", ""),
        )


@dataclass
class HardwareInfo:
    """Represents hardware information."""

    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    vram_used_gb: float
    vram_total_gb: float
    gpu_name: str
    gpu_temp: int
    platform: str = ""
    python_version: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareInfo":
        """Create HardwareInfo from dictionary."""
        return cls(
            cpu_percent=data.get("cpu_percent", 0.0),
            ram_used_gb=data.get("ram_used_gb", 0.0),
            ram_total_gb=data.get("ram_total_gb", 0.0),
            vram_used_gb=data.get("vram_used_gb", 0.0),
            vram_total_gb=data.get("vram_total_gb", 0.0),
            gpu_name=data.get("gpu_name", "N/A"),
            gpu_temp=data.get("gpu_temp", 0),
            platform=data.get("platform", ""),
            python_version=data.get("python_version", ""),
        )


@dataclass
class AnalysisResult:
    """Represents video analysis results."""

    content_type: str
    degradation: str
    resolution: str
    frame_count: int
    fps: float
    duration: float
    recommendations: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create AnalysisResult from dictionary."""
        analysis = data.get("analysis", data)
        return cls(
            content_type=analysis.get("content_type", "unknown"),
            degradation=analysis.get("degradation", "unknown"),
            resolution=analysis.get("resolution", "unknown"),
            frame_count=analysis.get("frame_count", 0),
            fps=analysis.get("fps", 0.0),
            duration=analysis.get("duration", 0.0),
            recommendations=analysis.get("recommendations", []),
        )


# =============================================================================
# Exceptions
# =============================================================================


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int = 0, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class NotFoundError(APIError):
    """Raised when a resource is not found."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, 429)
        self.retry_after = retry_after


class ConnectionError(APIError):
    """Raised when connection to server fails."""
    pass


# =============================================================================
# API Client
# =============================================================================


class FrameWrightClient:
    """Client for the FrameWright REST API.

    Provides methods for submitting restoration jobs, checking status,
    and managing the restoration pipeline.

    Example:
        >>> client = FrameWrightClient("http://localhost:8081")
        >>> job_id = client.restore("/path/to/video.mp4")
        >>> client.wait_for_completion(job_id)
        >>> status = client.get_status(job_id)
        >>> print(f"Completed: {status.output_path}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        # Normalize base URL
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/api/v1"):
            self.base_url = f"{self.base_url}/api/v1"

        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (e.g., "/jobs")
            data: Optional JSON data for POST requests
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            APIError: On API errors
            ConnectionError: On connection failures
        """
        # Build URL
        url = f"{self.base_url}{endpoint}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        # Prepare request
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        # Create request
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}

        except urllib.error.HTTPError as e:
            # Parse error response
            try:
                error_data = json.loads(e.read().decode("utf-8"))
                error_message = error_data.get("error", str(e))
            except (json.JSONDecodeError, Exception):
                error_message = str(e)

            # Raise appropriate exception
            if e.code == 401:
                raise AuthenticationError(error_message, e.code)
            elif e.code == 404:
                raise NotFoundError(error_message, e.code)
            elif e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 60))
                raise RateLimitError(error_message, retry_after)
            else:
                raise APIError(error_message, e.code)

        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {self.base_url}: {e}")

    # -------------------------------------------------------------------------
    # Job Operations
    # -------------------------------------------------------------------------

    def restore(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preset: str = "balanced",
        scale: int = 4,
        model: Optional[str] = None,
        **options,
    ) -> str:
        """Submit a video restoration job.

        Args:
            input_path: Path to input video file
            output_path: Optional path for output file
            preset: Restoration preset (fast, balanced, quality, ultra)
            scale: Upscaling factor (2 or 4)
            model: Specific model to use (optional)
            **options: Additional processing options

        Returns:
            Job ID for the submitted job

        Raises:
            APIError: On submission failure
        """
        data = {
            "input_path": str(input_path),
            "preset": preset,
            "scale": scale,
        }

        if output_path:
            data["output_path"] = str(output_path)

        if model:
            data["model"] = model

        if options:
            data["options"] = options

        response = self._request("POST", "/restore", data=data)

        if not response.get("success"):
            raise APIError(response.get("error", "Failed to submit job"))

        return response["job_id"]

    def get_status(self, job_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            job_id: ID of the job

        Returns:
            JobStatus object with current status

        Raises:
            NotFoundError: If job not found
        """
        response = self._request("GET", f"/jobs/{job_id}")
        return JobStatus.from_dict(response)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[JobStatus]:
        """List all jobs.

        Args:
            status: Filter by status (pending, processing, completed, failed, cancelled)
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of JobStatus objects
        """
        params = {"limit": str(limit), "offset": str(offset)}
        if status:
            params["status"] = status

        response = self._request("GET", "/jobs", params=params)
        return [JobStatus.from_dict(j) for j in response.get("jobs", [])]

    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            True if cancelled successfully

        Raises:
            NotFoundError: If job not found
            APIError: If job cannot be cancelled
        """
        response = self._request("DELETE", f"/jobs/{job_id}")
        return response.get("success", False)

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> JobStatus:
        """Wait for a job to complete.

        Args:
            job_id: ID of the job to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no limit)
            progress_callback: Optional callback(JobStatus) for progress updates

        Returns:
            Final JobStatus

        Raises:
            TimeoutError: If timeout exceeded
            APIError: On API errors
        """
        start_time = time.time()

        while True:
            status = self.get_status(job_id)

            if progress_callback:
                progress_callback(status)

            if status.is_complete:
                return status

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Job {job_id} did not complete within {timeout} seconds"
                    )

            time.sleep(poll_interval)

    def download_result(
        self,
        job_id: str,
        output_path: Union[str, Path],
    ) -> Path:
        """Download the result of a completed job.

        Note: This method assumes the output file is accessible locally
        or the server provides a download endpoint.

        Args:
            job_id: ID of the completed job
            output_path: Local path to save the result

        Returns:
            Path to the downloaded file

        Raises:
            APIError: If job not complete or download fails
        """
        status = self.get_status(job_id)

        if not status.is_successful:
            raise APIError(f"Job {job_id} is not complete (state: {status.state})")

        # If server and client are on the same machine, the output_path
        # from the job status can be used directly
        source_path = Path(status.output_path)
        dest_path = Path(output_path)

        if source_path.exists():
            # Local file - copy or link
            import shutil

            if dest_path != source_path:
                shutil.copy2(source_path, dest_path)
            return dest_path
        else:
            # For remote servers, you would need to implement a download
            # endpoint on the server side
            raise APIError(
                f"Output file not accessible locally: {status.output_path}. "
                "For remote servers, implement a download endpoint."
            )

    # -------------------------------------------------------------------------
    # Configuration Operations
    # -------------------------------------------------------------------------

    def get_presets(self) -> List[Preset]:
        """Get available presets.

        Returns:
            List of available presets
        """
        response = self._request("GET", "/presets")
        return [Preset.from_dict(p) for p in response.get("presets", [])]

    def get_models(self) -> List[Model]:
        """Get available models.

        Returns:
            List of available models
        """
        response = self._request("GET", "/models")
        return [Model.from_dict(m) for m in response.get("models", [])]

    def get_hardware(self) -> HardwareInfo:
        """Get hardware information.

        Returns:
            Hardware information from the server
        """
        response = self._request("GET", "/hardware")
        return HardwareInfo.from_dict(response)

    # -------------------------------------------------------------------------
    # Analysis Operations
    # -------------------------------------------------------------------------

    def analyze(self, input_path: str) -> AnalysisResult:
        """Analyze a video file.

        Args:
            input_path: Path to video file to analyze

        Returns:
            Analysis results with recommendations
        """
        data = {"input_path": str(input_path)}
        response = self._request("POST", "/analyze", data=data)
        return AnalysisResult.from_dict(response)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if the API server is healthy.

        Returns:
            True if server is responding
        """
        try:
            response = self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception:
            return False

    def get_api_info(self) -> Dict[str, Any]:
        """Get API information.

        Returns:
            API info including version and available endpoints
        """
        return self._request("GET", "/")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_client(
    url: str = "http://localhost:8081",
    api_key: Optional[str] = None,
) -> FrameWrightClient:
    """Create a FrameWright API client.

    Args:
        url: API server URL
        api_key: Optional API key

    Returns:
        Configured client instance
    """
    return FrameWrightClient(base_url=url, api_key=api_key)


def quick_restore(
    input_path: str,
    output_path: Optional[str] = None,
    server_url: str = "http://localhost:8081",
    api_key: Optional[str] = None,
    preset: str = "balanced",
    wait: bool = True,
) -> Union[str, JobStatus]:
    """Quick function to restore a video.

    Args:
        input_path: Path to input video
        output_path: Optional output path
        server_url: API server URL
        api_key: Optional API key
        preset: Restoration preset
        wait: If True, wait for completion

    Returns:
        Job ID if not waiting, JobStatus if waiting

    Example:
        >>> result = quick_restore("video.mp4", wait=True)
        >>> print(f"Output: {result.output_path}")
    """
    client = FrameWrightClient(base_url=server_url, api_key=api_key)

    job_id = client.restore(
        input_path=input_path,
        output_path=output_path,
        preset=preset,
    )

    if wait:
        return client.wait_for_completion(job_id)

    return job_id
