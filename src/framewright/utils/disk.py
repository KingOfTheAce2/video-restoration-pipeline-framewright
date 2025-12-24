"""Disk space validation and monitoring utilities for FrameWright.

Provides pre-flight disk space checks and monitoring during processing.
"""
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiskUsage:
    """Disk usage information."""
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def total_gb(self) -> float:
        """Total space in GB."""
        return self.total_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        """Used space in GB."""
        return self.used_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        """Free space in GB."""
        return self.free_bytes / (1024 ** 3)

    @property
    def usage_percent(self) -> float:
        """Usage percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100


def get_disk_usage(path: Path) -> DiskUsage:
    """Get disk usage for the filesystem containing path.

    Args:
        path: Path on the filesystem to check

    Returns:
        DiskUsage object with space information
    """
    usage = shutil.disk_usage(path)
    return DiskUsage(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )


def get_directory_size(path: Path) -> int:
    """Calculate total size of a directory and its contents.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total_size = 0

    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total_size += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError) as e:
        logger.warning(f"Error calculating directory size: {e}")

    return total_size


@dataclass
class SpaceEstimate:
    """Estimated disk space requirements."""
    frames_extraction_bytes: int
    enhanced_frames_bytes: int
    audio_bytes: int
    output_video_bytes: int
    temporary_bytes: int
    total_bytes: int

    @property
    def total_gb(self) -> float:
        """Total required space in GB."""
        return self.total_bytes / (1024 ** 3)


def estimate_required_space(
    video_path: Path,
    scale_factor: int = 4,
    frame_count: Optional[int] = None,
    video_duration_seconds: Optional[float] = None,
    fps: float = 30.0,
) -> SpaceEstimate:
    """Estimate disk space required for video restoration.

    Args:
        video_path: Path to source video
        scale_factor: Upscaling factor (2 or 4)
        frame_count: Number of frames (estimated from duration if not provided)
        video_duration_seconds: Video duration (extracted if not provided)
        fps: Frames per second (used if frame_count not provided)

    Returns:
        SpaceEstimate with breakdown of required space
    """
    video_size = video_path.stat().st_size if video_path.exists() else 0

    # Estimate frame count if not provided
    if frame_count is None:
        if video_duration_seconds:
            frame_count = int(video_duration_seconds * fps)
        else:
            # Rough estimate: typical 1080p video at 8mbps
            estimated_duration = video_size / (8_000_000 / 8)  # bytes per second
            frame_count = int(estimated_duration * fps)

    # PNG frame size estimates (based on typical 1080p frame)
    # PNG compression varies, but typically 2-5MB per 1080p frame
    avg_frame_size = 3 * 1024 * 1024  # 3MB baseline

    # Scale factor affects output frame size quadratically
    enhanced_frame_size = avg_frame_size * (scale_factor ** 2)

    # Calculate space estimates with safety margins
    frames_extraction = frame_count * avg_frame_size
    enhanced_frames = frame_count * enhanced_frame_size

    # Audio: ~10MB per minute for 48kHz 24-bit PCM
    audio_estimate = int((frame_count / fps / 60) * 10 * 1024 * 1024)

    # Output video: ~3x original size (high quality encoding)
    output_estimate = video_size * 3

    # Temporary processing overhead (~10% of total)
    subtotal = frames_extraction + enhanced_frames + audio_estimate + output_estimate
    temporary = int(subtotal * 0.1)

    total = subtotal + temporary

    return SpaceEstimate(
        frames_extraction_bytes=frames_extraction,
        enhanced_frames_bytes=enhanced_frames,
        audio_bytes=audio_estimate,
        output_video_bytes=output_estimate,
        temporary_bytes=temporary,
        total_bytes=total,
    )


def validate_disk_space(
    project_dir: Path,
    video_path: Path,
    scale_factor: int = 4,
    safety_margin: float = 1.2,
    frame_count: Optional[int] = None,
) -> Dict:
    """Validate sufficient disk space before processing.

    Args:
        project_dir: Directory for processing files
        video_path: Source video path
        scale_factor: Upscaling factor
        safety_margin: Multiplier for safety buffer (1.2 = 20% extra)
        frame_count: Optional known frame count

    Returns:
        Dictionary with validation results:
        - is_valid: bool
        - required_gb: float
        - available_gb: float
        - estimate: SpaceEstimate

    Raises:
        ValueError: If project_dir doesn't exist or isn't writable
    """
    # Ensure project directory exists or can be created
    if not project_dir.exists():
        try:
            project_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create project directory: {e}")

    # Check writability
    if not os.access(project_dir, os.W_OK):
        raise ValueError(f"Project directory is not writable: {project_dir}")

    # Get available space
    disk_usage = get_disk_usage(project_dir)
    available_bytes = disk_usage.free_bytes

    # Estimate required space
    estimate = estimate_required_space(
        video_path,
        scale_factor=scale_factor,
        frame_count=frame_count,
    )

    required_bytes = int(estimate.total_bytes * safety_margin)

    is_valid = available_bytes >= required_bytes

    result = {
        "is_valid": is_valid,
        "required_bytes": required_bytes,
        "available_bytes": available_bytes,
        "required_gb": required_bytes / (1024 ** 3),
        "available_gb": available_bytes / (1024 ** 3),
        "estimate": estimate,
        "disk_usage": disk_usage,
    }

    if not is_valid:
        logger.error(
            f"Insufficient disk space. "
            f"Required: {result['required_gb']:.1f}GB, "
            f"Available: {result['available_gb']:.1f}GB"
        )
    else:
        logger.info(
            f"Disk space OK. "
            f"Required: {result['required_gb']:.1f}GB, "
            f"Available: {result['available_gb']:.1f}GB"
        )

    return result


class DiskSpaceMonitor:
    """Monitor disk space during processing."""

    def __init__(
        self,
        project_dir: Path,
        warning_threshold_gb: float = 1.0,
        critical_threshold_gb: float = 0.5,
    ):
        """Initialize disk space monitor.

        Args:
            project_dir: Directory to monitor
            warning_threshold_gb: Warning when free space below this
            critical_threshold_gb: Error when free space below this
        """
        self.project_dir = Path(project_dir)
        self.warning_threshold_bytes = int(warning_threshold_gb * 1024 ** 3)
        self.critical_threshold_bytes = int(critical_threshold_gb * 1024 ** 3)
        self.initial_free_bytes: Optional[int] = None

    def initialize(self) -> DiskUsage:
        """Record initial disk state.

        Returns:
            Initial disk usage
        """
        usage = get_disk_usage(self.project_dir)
        self.initial_free_bytes = usage.free_bytes
        return usage

    def check(self) -> Dict:
        """Check current disk space status.

        Returns:
            Dictionary with status information:
            - status: 'ok', 'warning', or 'critical'
            - free_gb: Current free space
            - used_since_start_gb: Space used since monitoring started
        """
        usage = get_disk_usage(self.project_dir)

        if usage.free_bytes < self.critical_threshold_bytes:
            status = "critical"
            logger.error(f"Critical: Only {usage.free_gb:.2f}GB free!")
        elif usage.free_bytes < self.warning_threshold_bytes:
            status = "warning"
            logger.warning(f"Warning: Only {usage.free_gb:.2f}GB free")
        else:
            status = "ok"

        used_since_start = 0.0
        if self.initial_free_bytes is not None:
            used_since_start = (self.initial_free_bytes - usage.free_bytes) / (1024 ** 3)

        return {
            "status": status,
            "free_bytes": usage.free_bytes,
            "free_gb": usage.free_gb,
            "used_since_start_gb": used_since_start,
            "disk_usage": usage,
        }

    def is_critical(self) -> bool:
        """Check if disk space is critically low."""
        usage = get_disk_usage(self.project_dir)
        return usage.free_bytes < self.critical_threshold_bytes

    def has_space_for(self, required_bytes: int) -> bool:
        """Check if enough space for additional data.

        Args:
            required_bytes: Bytes needed

        Returns:
            True if sufficient space available
        """
        usage = get_disk_usage(self.project_dir)
        # Include critical threshold as buffer
        return usage.free_bytes - required_bytes > self.critical_threshold_bytes


def cleanup_old_temp_files(
    project_dir: Path,
    max_age_hours: float = 24.0,
) -> int:
    """Clean up old temporary files to reclaim space.

    Args:
        project_dir: Project directory
        max_age_hours: Delete files older than this

    Returns:
        Number of bytes reclaimed
    """
    import time

    temp_dir = project_dir / "temp"
    if not temp_dir.exists():
        return 0

    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    reclaimed = 0

    try:
        for entry in temp_dir.rglob("*"):
            if entry.is_file():
                try:
                    age = current_time - entry.stat().st_mtime
                    if age > max_age_seconds:
                        size = entry.stat().st_size
                        entry.unlink()
                        reclaimed += size
                        logger.debug(f"Deleted old temp file: {entry}")
                except (OSError, PermissionError):
                    pass
    except Exception as e:
        logger.warning(f"Error cleaning temp files: {e}")

    if reclaimed > 0:
        logger.info(f"Reclaimed {reclaimed / (1024**2):.1f}MB from old temp files")

    return reclaimed
