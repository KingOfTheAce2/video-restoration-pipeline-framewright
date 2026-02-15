"""Core type definitions for FrameWright video restoration pipeline.

This module provides fundamental data types used throughout the pipeline:
- Frame: numpy array wrapper with metadata
- FrameSequence: list of frames with video metadata
- VideoMetadata: fps, resolution, duration, codec info
- AudioMetadata: sample_rate, channels, duration
- BoundingBox: for face/object detection
- ProcessingResult: success/failure with metrics

These types provide a consistent interface for passing data between
processing stages and ensure type safety throughout the codebase.

Example usage:

    >>> from framewright.core.types import Frame, VideoMetadata, BoundingBox
    >>>
    >>> # Create a frame with metadata
    >>> frame = Frame(data=np.zeros((1080, 1920, 3), dtype=np.uint8), index=0)
    >>>
    >>> # Access video metadata
    >>> metadata = VideoMetadata(width=1920, height=1080, fps=24.0)
    >>> print(f"Resolution: {metadata.resolution}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Aliases
# =============================================================================

# Numpy array types
ImageArray: TypeAlias = NDArray[np.uint8]
FloatImageArray: TypeAlias = NDArray[np.float32]
GrayscaleArray: TypeAlias = NDArray[np.uint8]

# Coordinate types
Point: TypeAlias = Tuple[int, int]
PointFloat: TypeAlias = Tuple[float, float]
Size: TypeAlias = Tuple[int, int]
Region: TypeAlias = Tuple[int, int, int, int]  # x, y, width, height

# Path types
PathLike: TypeAlias = Union[str, Path]

# Color types
ColorRGB: TypeAlias = Tuple[int, int, int]
ColorRGBA: TypeAlias = Tuple[int, int, int, int]
ColorBGR: TypeAlias = Tuple[int, int, int]


# =============================================================================
# Enumerations
# =============================================================================


class ColorSpace(str, Enum):
    """Color space for image data."""
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    YUV = "yuv"
    LAB = "lab"
    HSV = "hsv"


class PixelFormat(str, Enum):
    """Pixel format for video data."""
    UINT8 = "uint8"
    UINT16 = "uint16"
    FLOAT32 = "float32"


class InterpolationMode(str, Enum):
    """Interpolation mode for resizing."""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS = "lanczos"
    AREA = "area"


class ProcessingStatus(str, Enum):
    """Status of a processing operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    """Quality assessment level."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


# =============================================================================
# Bounding Box
# =============================================================================


@dataclass
class BoundingBox:
    """Bounding box for object/face detection.

    Coordinates are in pixels, with (0, 0) at top-left corner.
    The box is defined by top-left corner and dimensions.

    Attributes:
        x: Left edge x-coordinate.
        y: Top edge y-coordinate.
        width: Box width.
        height: Box height.
        confidence: Detection confidence (0.0-1.0).
        label: Optional label for the detected object.
        track_id: Optional tracking ID for video tracking.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    label: Optional[str] = None
    track_id: Optional[int] = None

    @property
    def x1(self) -> int:
        """Left edge."""
        return self.x

    @property
    def y1(self) -> int:
        """Top edge."""
        return self.y

    @property
    def x2(self) -> int:
        """Right edge."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge."""
        return self.y + self.height

    @property
    def center(self) -> Point:
        """Center point of the box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def center_float(self) -> PointFloat:
        """Center point as floats."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        """Area of the box in pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x, self.y, self.x2, self.y2)

    def contains(self, point: Point) -> bool:
        """Check if point is inside the box.

        Args:
            point: (x, y) point to check.

        Returns:
            True if point is inside the box.
        """
        px, py = point
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects with another.

        Args:
            other: Another bounding box.

        Returns:
            True if boxes intersect.
        """
        return not (
            self.x2 <= other.x or
            other.x2 <= self.x or
            self.y2 <= other.y or
            other.y2 <= self.y
        )

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """Compute intersection with another box.

        Args:
            other: Another bounding box.

        Returns:
            Intersection box, or None if no intersection.
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Compute union with another box.

        Args:
            other: Another bounding box.

        Returns:
            Smallest box containing both.
        """
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)

        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box.

        Args:
            other: Another bounding box.

        Returns:
            IoU value (0.0-1.0).
        """
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0

        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def expand(self, pixels: int) -> "BoundingBox":
        """Expand box by given pixels in all directions.

        Args:
            pixels: Number of pixels to expand.

        Returns:
            New expanded bounding box.
        """
        return BoundingBox(
            x=self.x - pixels,
            y=self.y - pixels,
            width=self.width + 2 * pixels,
            height=self.height + 2 * pixels,
            confidence=self.confidence,
            label=self.label,
            track_id=self.track_id,
        )

    def scale(self, factor: float) -> "BoundingBox":
        """Scale box by given factor around center.

        Args:
            factor: Scale factor (1.0 = no change).

        Returns:
            New scaled bounding box.
        """
        cx, cy = self.center_float
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)

        return BoundingBox(
            x=int(cx - new_width / 2),
            y=int(cy - new_height / 2),
            width=new_width,
            height=new_height,
            confidence=self.confidence,
            label=self.label,
            track_id=self.track_id,
        )

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """Clamp box to image boundaries.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            New clamped bounding box.
        """
        x1 = max(0, min(self.x, width - 1))
        y1 = max(0, min(self.y, height - 1))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))

        return BoundingBox(
            x=x1,
            y=y1,
            width=max(0, x2 - x1),
            height=max(0, y2 - y1),
            confidence=self.confidence,
            label=self.label,
            track_id=self.track_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "label": self.label,
            "track_id": self.track_id,
        }

    @classmethod
    def from_xyxy(
        cls,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        **kwargs: Any,
    ) -> "BoundingBox":
        """Create from (x1, y1, x2, y2) coordinates.

        Args:
            x1: Left edge.
            y1: Top edge.
            x2: Right edge.
            y2: Bottom edge.
            **kwargs: Additional attributes.

        Returns:
            BoundingBox instance.
        """
        return cls(
            x=min(x1, x2),
            y=min(y1, y2),
            width=abs(x2 - x1),
            height=abs(y2 - y1),
            **kwargs,
        )


# =============================================================================
# Video Metadata
# =============================================================================


@dataclass
class VideoMetadata:
    """Metadata about a video file.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames.
        duration: Duration in seconds.
        codec: Video codec name.
        pixel_format: Pixel format string.
        bitrate: Bitrate in bits per second.
        has_audio: Whether video has audio track.
        rotation: Rotation in degrees (0, 90, 180, 270).
        color_space: Color space name.
        bit_depth: Bits per channel.
        is_interlaced: Whether video is interlaced.
        is_variable_fps: Whether FPS varies.
        container_format: Container format (e.g., 'mp4', 'mkv').
        file_path: Path to the video file.
        file_size_bytes: File size in bytes.
    """

    width: int
    height: int
    fps: float
    total_frames: int = 0
    duration: float = 0.0
    codec: str = ""
    pixel_format: str = ""
    bitrate: int = 0
    has_audio: bool = True
    rotation: int = 0
    color_space: str = ""
    bit_depth: int = 8
    is_interlaced: bool = False
    is_variable_fps: bool = False
    container_format: str = ""
    file_path: Optional[Path] = None
    file_size_bytes: int = 0

    @property
    def resolution(self) -> Size:
        """Resolution as (width, height) tuple."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def aspect_ratio_string(self) -> str:
        """Aspect ratio as string (e.g., '16:9')."""
        from math import gcd
        if self.width == 0 or self.height == 0:
            return "0:0"
        g = gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"

    @property
    def duration_timedelta(self) -> timedelta:
        """Duration as timedelta."""
        return timedelta(seconds=self.duration)

    @property
    def duration_string(self) -> str:
        """Duration as formatted string (HH:MM:SS.mmm)."""
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    @property
    def is_hd(self) -> bool:
        """Check if video is HD (720p or higher)."""
        return self.height >= 720

    @property
    def is_full_hd(self) -> bool:
        """Check if video is Full HD (1080p or higher)."""
        return self.height >= 1080

    @property
    def is_4k(self) -> bool:
        """Check if video is 4K (2160p or higher)."""
        return self.height >= 2160 or self.width >= 3840

    @property
    def megapixels(self) -> float:
        """Resolution in megapixels."""
        return (self.width * self.height) / 1_000_000

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    def get_frame_at_time(self, time_seconds: float) -> int:
        """Get frame number at given time.

        Args:
            time_seconds: Time in seconds.

        Returns:
            Frame number (0-indexed).
        """
        return int(time_seconds * self.fps)

    def get_time_at_frame(self, frame_number: int) -> float:
        """Get time at given frame.

        Args:
            frame_number: Frame number (0-indexed).

        Returns:
            Time in seconds.
        """
        if self.fps == 0:
            return 0.0
        return frame_number / self.fps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "bitrate": self.bitrate,
            "has_audio": self.has_audio,
            "rotation": self.rotation,
            "color_space": self.color_space,
            "bit_depth": self.bit_depth,
            "is_interlaced": self.is_interlaced,
            "is_variable_fps": self.is_variable_fps,
            "container_format": self.container_format,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_size_bytes": self.file_size_bytes,
        }


# =============================================================================
# Audio Metadata
# =============================================================================


@dataclass
class AudioMetadata:
    """Metadata about an audio track.

    Attributes:
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        duration: Duration in seconds.
        codec: Audio codec name.
        bitrate: Bitrate in bits per second.
        bit_depth: Bits per sample.
        channel_layout: Channel layout string (e.g., 'stereo').
        language: Language code (e.g., 'en').
        title: Track title.
    """

    sample_rate: int
    channels: int
    duration: float = 0.0
    codec: str = ""
    bitrate: int = 0
    bit_depth: int = 16
    channel_layout: str = ""
    language: str = ""
    title: str = ""

    @property
    def is_stereo(self) -> bool:
        """Check if audio is stereo."""
        return self.channels == 2

    @property
    def is_mono(self) -> bool:
        """Check if audio is mono."""
        return self.channels == 1

    @property
    def is_surround(self) -> bool:
        """Check if audio is surround (more than 2 channels)."""
        return self.channels > 2

    @property
    def duration_string(self) -> str:
        """Duration as formatted string."""
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": self.duration,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "bit_depth": self.bit_depth,
            "channel_layout": self.channel_layout,
            "language": self.language,
            "title": self.title,
        }


# =============================================================================
# Frame
# =============================================================================


@dataclass
class Frame:
    """A video frame with associated metadata.

    Wraps a numpy array with additional information about the frame's
    position in the video and processing state.

    Attributes:
        data: Frame pixel data as numpy array (H, W, C) or (H, W).
        index: Frame index in the video (0-indexed).
        timestamp: Frame timestamp in seconds.
        source_path: Path to the source file.
        color_space: Color space of the data.
        metadata: Additional frame-specific metadata.
        detections: Detected objects/faces in the frame.
        quality_score: Frame quality score (0.0-1.0).
        is_keyframe: Whether this is a keyframe.
        scene_id: ID of the scene this frame belongs to.
    """

    data: ImageArray
    index: int = 0
    timestamp: float = 0.0
    source_path: Optional[Path] = None
    color_space: ColorSpace = ColorSpace.BGR
    metadata: Dict[str, Any] = field(default_factory=dict)
    detections: List[BoundingBox] = field(default_factory=list)
    quality_score: Optional[float] = None
    is_keyframe: bool = False
    scene_id: Optional[int] = None

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """Number of color channels."""
        if len(self.data.shape) == 2:
            return 1
        return self.data.shape[2]

    @property
    def size(self) -> Size:
        """Frame size as (width, height)."""
        return (self.width, self.height)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Frame shape (H, W, C)."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of pixel values."""
        return self.data.dtype

    @property
    def is_grayscale(self) -> bool:
        """Check if frame is grayscale."""
        return self.channels == 1

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def copy(self) -> "Frame":
        """Create a deep copy of the frame.

        Returns:
            New Frame with copied data.
        """
        return Frame(
            data=self.data.copy(),
            index=self.index,
            timestamp=self.timestamp,
            source_path=self.source_path,
            color_space=self.color_space,
            metadata=self.metadata.copy(),
            detections=list(self.detections),
            quality_score=self.quality_score,
            is_keyframe=self.is_keyframe,
            scene_id=self.scene_id,
        )

    def crop(self, box: BoundingBox) -> "Frame":
        """Crop frame to bounding box.

        Args:
            box: Bounding box to crop to.

        Returns:
            New cropped frame.
        """
        clamped = box.clamp(self.width, self.height)
        cropped_data = self.data[
            clamped.y:clamped.y2,
            clamped.x:clamped.x2,
        ].copy()

        return Frame(
            data=cropped_data,
            index=self.index,
            timestamp=self.timestamp,
            source_path=self.source_path,
            color_space=self.color_space,
            metadata={**self.metadata, "cropped_from": box.to_dict()},
            quality_score=self.quality_score,
            is_keyframe=self.is_keyframe,
            scene_id=self.scene_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without pixel data).

        Returns:
            Dictionary with frame metadata.
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "color_space": self.color_space.value,
            "dtype": str(self.dtype),
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata,
            "detections": [d.to_dict() for d in self.detections],
            "quality_score": self.quality_score,
            "is_keyframe": self.is_keyframe,
            "scene_id": self.scene_id,
        }


# =============================================================================
# Frame Sequence
# =============================================================================


@dataclass
class FrameSequence:
    """A sequence of video frames with associated metadata.

    Provides iteration and indexing over frames with video metadata.

    Attributes:
        frames: List of frames in the sequence.
        video_metadata: Metadata about the source video.
        audio_metadata: Metadata about audio track (if any).
        start_index: Starting frame index in source video.
        end_index: Ending frame index in source video.
    """

    frames: List[Frame] = field(default_factory=list)
    video_metadata: Optional[VideoMetadata] = None
    audio_metadata: Optional[AudioMetadata] = None
    start_index: int = 0
    end_index: Optional[int] = None

    def __len__(self) -> int:
        """Number of frames in sequence."""
        return len(self.frames)

    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, "FrameSequence"]:
        """Get frame(s) by index.

        Args:
            index: Frame index or slice.

        Returns:
            Single frame or new FrameSequence.
        """
        if isinstance(index, slice):
            return FrameSequence(
                frames=self.frames[index],
                video_metadata=self.video_metadata,
                audio_metadata=self.audio_metadata,
            )
        return self.frames[index]

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames."""
        return iter(self.frames)

    def __bool__(self) -> bool:
        """Check if sequence has frames."""
        return len(self.frames) > 0

    @property
    def fps(self) -> float:
        """Frames per second from video metadata."""
        if self.video_metadata:
            return self.video_metadata.fps
        return 0.0

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self.fps == 0:
            return 0.0
        return len(self.frames) / self.fps

    @property
    def width(self) -> int:
        """Frame width (from first frame or metadata)."""
        if self.frames:
            return self.frames[0].width
        if self.video_metadata:
            return self.video_metadata.width
        return 0

    @property
    def height(self) -> int:
        """Frame height (from first frame or metadata)."""
        if self.frames:
            return self.frames[0].height
        if self.video_metadata:
            return self.video_metadata.height
        return 0

    def append(self, frame: Frame) -> None:
        """Append a frame to the sequence.

        Args:
            frame: Frame to append.
        """
        self.frames.append(frame)

    def extend(self, frames: Sequence[Frame]) -> None:
        """Extend sequence with multiple frames.

        Args:
            frames: Frames to add.
        """
        self.frames.extend(frames)

    def get_frame_at_time(self, time_seconds: float) -> Optional[Frame]:
        """Get frame at given time.

        Args:
            time_seconds: Time in seconds.

        Returns:
            Frame at that time, or None if out of range.
        """
        if self.fps == 0:
            return None

        index = int(time_seconds * self.fps) - self.start_index
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def to_numpy(self) -> np.ndarray:
        """Convert all frames to a single numpy array.

        Returns:
            Array of shape (N, H, W, C) where N is number of frames.
        """
        if not self.frames:
            return np.array([])
        return np.stack([f.data for f in self.frames])


# =============================================================================
# Processing Result
# =============================================================================


@dataclass
class ProcessingResult:
    """Result of a processing operation.

    Encapsulates success/failure status along with metrics
    and any output data from the operation.

    Attributes:
        success: Whether the operation succeeded.
        status: Processing status.
        message: Human-readable result message.
        error: Error message if failed.
        error_type: Type of error if failed.
        duration: Processing duration in seconds.
        input_path: Path to input file.
        output_path: Path to output file.
        frames_processed: Number of frames processed.
        metrics: Dictionary of quality/performance metrics.
        warnings: List of warning messages.
        data: Additional result data.
    """

    success: bool
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    message: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration: float = 0.0
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    frames_processed: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        message: str = "Processing completed successfully",
        **kwargs: Any,
    ) -> "ProcessingResult":
        """Create a success result.

        Args:
            message: Success message.
            **kwargs: Additional attributes.

        Returns:
            ProcessingResult indicating success.
        """
        return cls(
            success=True,
            status=ProcessingStatus.COMPLETED,
            message=message,
            **kwargs,
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        error_type: Optional[str] = None,
        **kwargs: Any,
    ) -> "ProcessingResult":
        """Create a failure result.

        Args:
            error: Error message.
            error_type: Type of error.
            **kwargs: Additional attributes.

        Returns:
            ProcessingResult indicating failure.
        """
        return cls(
            success=False,
            status=ProcessingStatus.FAILED,
            message=f"Processing failed: {error}",
            error=error,
            error_type=error_type,
            **kwargs,
        )

    @classmethod
    def skipped_result(
        cls,
        reason: str,
        **kwargs: Any,
    ) -> "ProcessingResult":
        """Create a skipped result.

        Args:
            reason: Reason for skipping.
            **kwargs: Additional attributes.

        Returns:
            ProcessingResult indicating skipped operation.
        """
        return cls(
            success=True,
            status=ProcessingStatus.SKIPPED,
            message=f"Processing skipped: {reason}",
            **kwargs,
        )

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the result.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self.metrics[name] = value

    def add_warning(self, warning: str) -> None:
        """Add a warning message.

        Args:
            warning: Warning message.
        """
        self.warnings.append(warning)

    def get_quality_level(self) -> QualityLevel:
        """Determine quality level from metrics.

        Returns:
            QualityLevel based on SSIM/PSNR metrics.
        """
        if not self.success:
            return QualityLevel.FAILED

        ssim = self.metrics.get("ssim", 1.0)
        psnr = self.metrics.get("psnr", 100.0)

        if ssim >= 0.95 and psnr >= 35:
            return QualityLevel.EXCELLENT
        elif ssim >= 0.90 and psnr >= 30:
            return QualityLevel.GOOD
        elif ssim >= 0.85 and psnr >= 25:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "error_type": self.error_type,
            "duration": self.duration,
            "input_path": str(self.input_path) if self.input_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "frames_processed": self.frames_processed,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "quality_level": self.get_quality_level().value,
        }


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Type aliases
    "ImageArray",
    "FloatImageArray",
    "GrayscaleArray",
    "Point",
    "PointFloat",
    "Size",
    "Region",
    "PathLike",
    "ColorRGB",
    "ColorRGBA",
    "ColorBGR",
    # Enums
    "ColorSpace",
    "PixelFormat",
    "InterpolationMode",
    "ProcessingStatus",
    "QualityLevel",
    # Data classes
    "BoundingBox",
    "VideoMetadata",
    "AudioMetadata",
    "Frame",
    "FrameSequence",
    "ProcessingResult",
]
