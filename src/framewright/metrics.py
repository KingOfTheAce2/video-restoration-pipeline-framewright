"""Processing metrics and progress reporting module for FrameWright pipeline.

Provides metrics collection, ETA calculation, and progress reporting.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


def _mean(values: List[float]) -> float:
    """Calculate mean, using numpy if available."""
    if not values:
        return 0.0
    if HAS_NUMPY:
        return float(np.mean(values))
    return sum(values) / len(values)


# =============================================================================
# Processing Metrics
# =============================================================================

@dataclass
class ProcessingMetrics:
    """Comprehensive processing metrics with export capability."""

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Frame counts
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    skipped_frames: int = 0

    # Performance
    avg_frame_time_ms: float = 0.0
    min_frame_time_ms: float = float("inf")
    max_frame_time_ms: float = 0.0
    total_processing_time_seconds: float = 0.0

    # Resource usage
    peak_vram_mb: int = 0
    avg_vram_mb: float = 0.0
    peak_ram_mb: int = 0
    disk_usage_gb: float = 0.0

    # Quality scores
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # Errors
    error_count: int = 0
    retry_count: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Checkpointing
    checkpoint_count: int = 0
    resume_count: int = 0

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Get processing success rate as percentage."""
        if self.processed_frames == 0:
            return 0.0
        return (self.processed_frames - self.failed_frames) / self.processed_frames * 100

    @property
    def frames_per_second(self) -> float:
        """Get processing speed in frames per second."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.processed_frames / elapsed

    def record_frame(self, frame_time_ms: float, success: bool = True) -> None:
        """Record metrics for a processed frame.

        Args:
            frame_time_ms: Processing time in milliseconds
            success: Whether the frame was processed successfully
        """
        self.processed_frames += 1
        self.total_processing_time_seconds += frame_time_ms / 1000

        if success:
            self.min_frame_time_ms = min(self.min_frame_time_ms, frame_time_ms)
            self.max_frame_time_ms = max(self.max_frame_time_ms, frame_time_ms)

            # Update rolling average
            if self.avg_frame_time_ms == 0:
                self.avg_frame_time_ms = frame_time_ms
            else:
                # Exponential moving average
                alpha = 0.1
                self.avg_frame_time_ms = alpha * frame_time_ms + (1 - alpha) * self.avg_frame_time_ms
        else:
            self.failed_frames += 1

    def record_vram(self, used_mb: int) -> None:
        """Record VRAM usage sample.

        Args:
            used_mb: Current VRAM usage in MB
        """
        self.peak_vram_mb = max(self.peak_vram_mb, used_mb)

        # Update rolling average
        if self.avg_vram_mb == 0:
            self.avg_vram_mb = float(used_mb)
        else:
            alpha = 0.05
            self.avg_vram_mb = alpha * used_mb + (1 - alpha) * self.avg_vram_mb

    def record_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Record an error occurrence.

        Args:
            error: The exception that occurred
            context: Optional context information
        """
        self.error_count += 1
        self.errors.append({
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        })

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_count += 1

    def record_checkpoint(self) -> None:
        """Record a checkpoint save."""
        self.checkpoint_count += 1

    def record_resume(self) -> None:
        """Record a resume from checkpoint."""
        self.resume_count += 1

    def finish(self) -> None:
        """Mark processing as finished."""
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)

        # Convert datetime objects to ISO strings
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()

        # Add computed properties
        data["elapsed_seconds"] = self.elapsed_seconds
        data["success_rate"] = self.success_rate
        data["frames_per_second"] = self.frames_per_second

        # Handle infinity
        if data["min_frame_time_ms"] == float("inf"):
            data["min_frame_time_ms"] = 0.0

        return data

    def export_json(self, path: Path) -> None:
        """Export metrics to JSON file.

        Args:
            path: Path to output JSON file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Metrics exported to {path}")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "Processing Metrics Summary",
            "=" * 50,
            f"Total Frames: {self.total_frames}",
            f"Processed: {self.processed_frames}",
            f"Failed: {self.failed_frames}",
            f"Success Rate: {self.success_rate:.1f}%",
            "",
            f"Elapsed Time: {self.elapsed_seconds:.1f}s",
            f"Avg Frame Time: {self.avg_frame_time_ms:.1f}ms",
            f"Processing Speed: {self.frames_per_second:.2f} fps",
            "",
            f"Peak VRAM: {self.peak_vram_mb}MB",
            f"Avg VRAM: {self.avg_vram_mb:.0f}MB",
            "",
            f"Errors: {self.error_count}",
            f"Retries: {self.retry_count}",
            f"Checkpoints: {self.checkpoint_count}",
            "=" * 50,
        ]
        return "\n".join(lines)


# =============================================================================
# Progress Reporter
# =============================================================================

@dataclass
class ProgressUpdate:
    """Progress update information."""
    current: int
    total: int
    percentage: float
    eta_seconds: float
    elapsed_seconds: float
    avg_frame_ms: float
    frames_per_second: float
    status: str = "processing"


class ProgressReporter:
    """Rich progress reporting with ETA calculation.

    Uses rolling average of frame times to estimate remaining time.
    """

    def __init__(
        self,
        total_frames: int,
        window_size: int = 100,
        callback: Optional[callable] = None,
    ):
        """Initialize progress reporter.

        Args:
            total_frames: Total number of frames to process
            window_size: Number of recent frame times to use for ETA
            callback: Optional callback function for progress updates
        """
        self.total = total_frames
        self.processed = 0
        self.start_time = time.time()
        self.frame_times: List[float] = []
        self.window_size = window_size
        self.callback = callback
        self._last_update_time = 0.0
        self._update_interval = 0.5  # Minimum seconds between updates

    def update(self, frame_num: int, frame_time_ms: float) -> ProgressUpdate:
        """Update progress with new frame completion.

        Args:
            frame_num: Current frame number
            frame_time_ms: Time taken to process this frame in milliseconds

        Returns:
            ProgressUpdate with current progress information
        """
        self.processed = frame_num
        self.frame_times.append(frame_time_ms)

        # Keep only recent frame times for rolling average
        if len(self.frame_times) > self.window_size:
            self.frame_times = self.frame_times[-self.window_size:]

        # Calculate rolling average
        avg_time = _mean(self.frame_times)

        # Calculate ETA
        remaining = self.total - self.processed
        eta_seconds = (remaining * avg_time) / 1000 if remaining > 0 else 0

        elapsed = time.time() - self.start_time
        fps = self.processed / elapsed if elapsed > 0 else 0

        update = ProgressUpdate(
            current=self.processed,
            total=self.total,
            percentage=(self.processed / self.total * 100) if self.total > 0 else 0,
            eta_seconds=eta_seconds,
            elapsed_seconds=elapsed,
            avg_frame_ms=avg_time,
            frames_per_second=fps,
        )

        # Call callback if enough time has passed
        current_time = time.time()
        if self.callback and (current_time - self._last_update_time >= self._update_interval):
            self._last_update_time = current_time
            try:
                self.callback(update)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        return update

    def get_eta_string(self, eta_seconds: float) -> str:
        """Format ETA as human-readable string.

        Args:
            eta_seconds: ETA in seconds

        Returns:
            Formatted string like "1h 23m 45s" or "5m 30s"
        """
        if eta_seconds <= 0:
            return "Complete"

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def format_progress(self, update: ProgressUpdate) -> str:
        """Format progress as a progress bar string.

        Args:
            update: Progress update to format

        Returns:
            Formatted progress string
        """
        bar_width = 30
        filled = int(bar_width * update.percentage / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        eta_str = self.get_eta_string(update.eta_seconds)

        return (
            f"[{bar}] {update.percentage:5.1f}% "
            f"({update.current}/{update.total}) "
            f"ETA: {eta_str} | {update.frames_per_second:.1f} fps"
        )

    def finish(self) -> ProgressUpdate:
        """Mark progress as complete.

        Returns:
            Final progress update
        """
        elapsed = time.time() - self.start_time
        avg_time = _mean(self.frame_times) if self.frame_times else 0
        fps = self.processed / elapsed if elapsed > 0 else 0

        return ProgressUpdate(
            current=self.total,
            total=self.total,
            percentage=100.0,
            eta_seconds=0,
            elapsed_seconds=elapsed,
            avg_frame_ms=avg_time,
            frames_per_second=fps,
            status="complete",
        )


# =============================================================================
# Console Progress Bar
# =============================================================================

class ConsoleProgressBar:
    """Simple console progress bar with ETA."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize console progress bar.

        Args:
            total: Total number of items
            description: Description to show
        """
        self.reporter = ProgressReporter(total)
        self.description = description
        self._last_line_length = 0

    def update(self, current: int, frame_time_ms: float) -> None:
        """Update progress bar.

        Args:
            current: Current item number
            frame_time_ms: Time for last item in milliseconds
        """
        progress = self.reporter.update(current, frame_time_ms)
        line = f"\r{self.description}: {self.reporter.format_progress(progress)}"

        # Clear previous line if new one is shorter
        if len(line) < self._last_line_length:
            line += " " * (self._last_line_length - len(line))

        print(line, end="", flush=True)
        self._last_line_length = len(line)

    def finish(self) -> None:
        """Complete the progress bar."""
        progress = self.reporter.finish()
        elapsed_str = self.reporter.get_eta_string(progress.elapsed_seconds)
        print(f"\n{self.description}: Complete in {elapsed_str}")


# =============================================================================
# Quality Metrics
# =============================================================================

def calculate_psnr(original: "np.ndarray", restored: "np.ndarray") -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images.

    Higher values indicate better quality (typically 30-50 dB is good).

    Args:
        original: Original image as numpy array
        restored: Restored image as numpy array

    Returns:
        PSNR value in decibels
    """
    if not HAS_NUMPY:
        return 0.0

    # Ensure same size
    if original.shape != restored.shape:
        return 0.0

    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(
    original: "np.ndarray",
    restored: "np.ndarray",
    window_size: int = 11,
) -> float:
    """Calculate Structural Similarity Index between two images.

    SSIM ranges from -1 to 1, where 1 means identical images.
    Values above 0.9 generally indicate good quality.

    Args:
        original: Original image as numpy array
        restored: Restored image as numpy array
        window_size: Size of the Gaussian window

    Returns:
        SSIM value (0-1 for typical images)
    """
    if not HAS_NUMPY:
        return 0.0

    # Ensure same size
    if original.shape != restored.shape:
        return 0.0

    try:
        # Try to use scikit-image if available
        from skimage.metrics import structural_similarity
        return structural_similarity(
            original, restored,
            channel_axis=-1 if len(original.shape) == 3 else None,
            data_range=255,
        )
    except ImportError:
        pass

    # Fallback: simplified SSIM calculation
    # Convert to grayscale if color
    if len(original.shape) == 3:
        orig_gray = np.mean(original, axis=2)
        rest_gray = np.mean(restored, axis=2)
    else:
        orig_gray = original.astype(float)
        rest_gray = restored.astype(float)

    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = np.mean(orig_gray)
    mu_y = np.mean(rest_gray)

    sigma_x = np.var(orig_gray)
    sigma_y = np.var(rest_gray)
    sigma_xy = np.mean((orig_gray - mu_x) * (rest_gray - mu_y))

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return float(ssim)


def calculate_sharpness(image: "np.ndarray") -> float:
    """Calculate sharpness of an image using Laplacian variance.

    Higher values indicate sharper images.

    Args:
        image: Image as numpy array

    Returns:
        Sharpness score (Laplacian variance)
    """
    if not HAS_NUMPY:
        return 0.0

    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        # Fallback without OpenCV
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)

        # Simple Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        from scipy.ndimage import convolve
        filtered = convolve(gray, laplacian)
        return float(np.var(filtered))


def estimate_noise_level(image: "np.ndarray") -> float:
    """Estimate noise level in an image.

    Uses median absolute deviation of high-frequency components.

    Args:
        image: Image as numpy array

    Returns:
        Estimated noise standard deviation
    """
    if not HAS_NUMPY:
        return 0.0

    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # High-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray.astype(float), (5, 5), 0)
        noise = gray.astype(float) - blurred

        # Robust noise estimation using MAD
        mad = np.median(np.abs(noise - np.median(noise)))
        sigma = 1.4826 * mad  # Convert MAD to standard deviation

        return float(sigma)
    except ImportError:
        return 0.0
