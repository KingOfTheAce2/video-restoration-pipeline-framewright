"""Letterbox and Pillarbox Detection and Handling.

Detects black bars (letterbox/pillarbox) in video and provides options
to crop them or preserve with proper aspect ratio handling.

Features:
- Auto-detect letterbox (horizontal bars) and pillarbox (vertical bars)
- Detect non-standard aspect ratios
- Smart cropping with content-aware detection
- Aspect ratio preservation for upscaling

Example:
    >>> detector = LetterboxDetector()
    >>> result = detector.analyze(video_path)
    >>> if result.has_letterbox:
    ...     cropper = LetterboxCropper()
    ...     cropper.crop(video_path, output_path, result.crop_region)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class CropRegion:
    """Region to crop from video."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def as_ffmpeg_filter(self) -> str:
        """Return as FFmpeg crop filter string."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass
class AspectRatio:
    """Aspect ratio representation."""
    width: int
    height: int

    @property
    def ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0

    @property
    def name(self) -> str:
        """Get common name for aspect ratio."""
        r = self.ratio
        if abs(r - 1.33) < 0.05:
            return "4:3 (Standard)"
        elif abs(r - 1.78) < 0.05:
            return "16:9 (Widescreen)"
        elif abs(r - 1.85) < 0.05:
            return "1.85:1 (Flat)"
        elif abs(r - 2.35) < 0.1:
            return "2.35:1 (CinemaScope)"
        elif abs(r - 2.39) < 0.1:
            return "2.39:1 (Anamorphic)"
        elif abs(r - 2.76) < 0.1:
            return "2.76:1 (Ultra Panavision)"
        elif abs(r - 1.0) < 0.05:
            return "1:1 (Square)"
        elif abs(r - 0.5625) < 0.05:
            return "9:16 (Vertical)"
        else:
            return f"{r:.2f}:1"

    def __str__(self) -> str:
        return f"{self.width}:{self.height} ({self.name})"


@dataclass
class LetterboxAnalysis:
    """Results of letterbox/pillarbox detection."""
    has_letterbox: bool = False  # Horizontal black bars (top/bottom)
    has_pillarbox: bool = False  # Vertical black bars (left/right)
    has_windowbox: bool = False  # Both (black on all sides)

    # Original frame dimensions
    frame_width: int = 0
    frame_height: int = 0

    # Detected content region
    content_x: int = 0
    content_y: int = 0
    content_width: int = 0
    content_height: int = 0

    # Bar sizes in pixels
    top_bar: int = 0
    bottom_bar: int = 0
    left_bar: int = 0
    right_bar: int = 0

    # Aspect ratios
    original_aspect: AspectRatio = field(default_factory=lambda: AspectRatio(16, 9))
    content_aspect: AspectRatio = field(default_factory=lambda: AspectRatio(16, 9))

    # Detection confidence
    confidence: float = 0.0

    # Is it variable (bars change size during video)?
    is_variable: bool = False

    @property
    def crop_region(self) -> CropRegion:
        """Get crop region to remove black bars."""
        return CropRegion(
            x=self.content_x,
            y=self.content_y,
            width=self.content_width,
            height=self.content_height,
        )

    @property
    def bar_percentage(self) -> float:
        """Percentage of frame that is black bars."""
        total_pixels = self.frame_width * self.frame_height
        content_pixels = self.content_width * self.content_height
        if total_pixels == 0:
            return 0
        return (1 - content_pixels / total_pixels) * 100

    def summary(self) -> str:
        """Get human-readable summary."""
        if not (self.has_letterbox or self.has_pillarbox):
            return "No black bars detected - full frame content"

        bar_type = []
        if self.has_letterbox:
            bar_type.append(f"Letterbox (top: {self.top_bar}px, bottom: {self.bottom_bar}px)")
        if self.has_pillarbox:
            bar_type.append(f"Pillarbox (left: {self.left_bar}px, right: {self.right_bar}px)")

        return (
            f"{' + '.join(bar_type)}\n"
            f"Original: {self.frame_width}x{self.frame_height} ({self.original_aspect.name})\n"
            f"Content: {self.content_width}x{self.content_height} ({self.content_aspect.name})\n"
            f"Black bars: {self.bar_percentage:.1f}% of frame"
        )


class LetterboxDetector:
    """Detects letterbox/pillarbox in video frames.

    Uses edge detection and luminance analysis to find black bars.
    """

    def __init__(
        self,
        black_threshold: int = 16,
        min_bar_size: int = 4,
        sample_count: int = 20,
        edge_tolerance: int = 2,
    ):
        """Initialize detector.

        Args:
            black_threshold: Maximum luminance to consider "black" (0-255)
            min_bar_size: Minimum bar size in pixels to detect
            sample_count: Number of frames to sample
            edge_tolerance: Pixels of tolerance for bar edge detection
        """
        self.black_threshold = black_threshold
        self.min_bar_size = min_bar_size
        self.sample_count = sample_count
        self.edge_tolerance = edge_tolerance

    def analyze(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> LetterboxAnalysis:
        """Analyze video for letterbox/pillarbox.

        Args:
            video_path: Path to video file
            progress_callback: Progress callback (0-1)

        Returns:
            LetterboxAnalysis with detection results
        """
        if not HAS_OPENCV:
            logger.error("OpenCV required for letterbox detection")
            return LetterboxAnalysis()

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return LetterboxAnalysis()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        result = LetterboxAnalysis(
            frame_width=frame_width,
            frame_height=frame_height,
            original_aspect=AspectRatio(frame_width, frame_height),
        )

        # Sample frames throughout video
        sample_interval = max(1, total_frames // self.sample_count)
        detections = []

        for i, frame_num in enumerate(range(0, total_frames, sample_interval)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            detection = self._detect_bars_in_frame(frame)
            detections.append(detection)

            if progress_callback:
                progress_callback((i + 1) / self.sample_count)

        cap.release()

        if not detections:
            return result

        # Aggregate results
        result = self._aggregate_detections(result, detections)

        return result

    def _detect_bars_in_frame(self, frame: "np.ndarray") -> Dict[str, int]:
        """Detect black bars in a single frame."""
        height, width = frame.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Detect top bar
        top_bar = 0
        for y in range(height // 2):
            row_mean = np.mean(gray[y, :])
            if row_mean > self.black_threshold:
                top_bar = y
                break

        # Detect bottom bar
        bottom_bar = 0
        for y in range(height - 1, height // 2, -1):
            row_mean = np.mean(gray[y, :])
            if row_mean > self.black_threshold:
                bottom_bar = height - 1 - y
                break

        # Detect left bar
        left_bar = 0
        for x in range(width // 2):
            col_mean = np.mean(gray[:, x])
            if col_mean > self.black_threshold:
                left_bar = x
                break

        # Detect right bar
        right_bar = 0
        for x in range(width - 1, width // 2, -1):
            col_mean = np.mean(gray[:, x])
            if col_mean > self.black_threshold:
                right_bar = width - 1 - x
                break

        return {
            "top": top_bar,
            "bottom": bottom_bar,
            "left": left_bar,
            "right": right_bar,
        }

    def _aggregate_detections(
        self,
        result: LetterboxAnalysis,
        detections: List[Dict[str, int]],
    ) -> LetterboxAnalysis:
        """Aggregate multiple frame detections."""
        # Use median values to handle outliers
        tops = [d["top"] for d in detections]
        bottoms = [d["bottom"] for d in detections]
        lefts = [d["left"] for d in detections]
        rights = [d["right"] for d in detections]

        result.top_bar = int(np.median(tops))
        result.bottom_bar = int(np.median(bottoms))
        result.left_bar = int(np.median(lefts))
        result.right_bar = int(np.median(rights))

        # Check for variability
        top_std = np.std(tops)
        bottom_std = np.std(bottoms)
        result.is_variable = (top_std > 5) or (bottom_std > 5)

        # Apply minimum bar size filter
        if result.top_bar < self.min_bar_size:
            result.top_bar = 0
        if result.bottom_bar < self.min_bar_size:
            result.bottom_bar = 0
        if result.left_bar < self.min_bar_size:
            result.left_bar = 0
        if result.right_bar < self.min_bar_size:
            result.right_bar = 0

        # Determine bar types
        result.has_letterbox = (result.top_bar > 0) or (result.bottom_bar > 0)
        result.has_pillarbox = (result.left_bar > 0) or (result.right_bar > 0)
        result.has_windowbox = result.has_letterbox and result.has_pillarbox

        # Calculate content region
        result.content_x = result.left_bar
        result.content_y = result.top_bar
        result.content_width = result.frame_width - result.left_bar - result.right_bar
        result.content_height = result.frame_height - result.top_bar - result.bottom_bar

        # Content aspect ratio
        if result.content_width > 0 and result.content_height > 0:
            result.content_aspect = AspectRatio(result.content_width, result.content_height)

        # Calculate confidence based on consistency
        if result.has_letterbox or result.has_pillarbox:
            consistency = 1 - (top_std + bottom_std) / (result.frame_height / 2)
            result.confidence = max(0, min(1, consistency))
        else:
            result.confidence = 1.0

        return result

    def analyze_frame(self, frame: "np.ndarray") -> LetterboxAnalysis:
        """Analyze a single frame for letterbox/pillarbox.

        Args:
            frame: Frame as numpy array (BGR)

        Returns:
            LetterboxAnalysis
        """
        if not HAS_OPENCV:
            return LetterboxAnalysis()

        height, width = frame.shape[:2]
        result = LetterboxAnalysis(
            frame_width=width,
            frame_height=height,
            original_aspect=AspectRatio(width, height),
        )

        detection = self._detect_bars_in_frame(frame)
        return self._aggregate_detections(result, [detection])


class LetterboxCropper:
    """Crop letterbox/pillarbox from video."""

    def __init__(
        self,
        align_to: int = 2,  # Align dimensions to this value (for codec compatibility)
    ):
        """Initialize cropper.

        Args:
            align_to: Align output dimensions to this value
        """
        self.align_to = align_to

    def crop(
        self,
        input_path: Path,
        output_path: Path,
        crop_region: CropRegion,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Crop video to remove black bars.

        Args:
            input_path: Input video path
            output_path: Output video path
            crop_region: Region to keep
            progress_callback: Progress callback

        Returns:
            True if successful
        """
        import subprocess

        # Align dimensions
        width = (crop_region.width // self.align_to) * self.align_to
        height = (crop_region.height // self.align_to) * self.align_to

        crop_filter = f"crop={width}:{height}:{crop_region.x}:{crop_region.y}"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", crop_filter,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "copy",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=3600)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Cropping failed: {e}")
            return False

    def auto_crop(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[bool, LetterboxAnalysis]:
        """Automatically detect and crop black bars.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Progress callback

        Returns:
            Tuple of (success, analysis)
        """
        detector = LetterboxDetector()
        analysis = detector.analyze(input_path, progress_callback)

        if not (analysis.has_letterbox or analysis.has_pillarbox):
            logger.info("No black bars detected, no cropping needed")
            return False, analysis

        success = self.crop(
            input_path, output_path,
            analysis.crop_region,
            progress_callback
        )

        return success, analysis


# Common aspect ratio presets
ASPECT_RATIOS = {
    "4:3": AspectRatio(4, 3),
    "16:9": AspectRatio(16, 9),
    "1.85:1": AspectRatio(185, 100),
    "2.35:1": AspectRatio(235, 100),
    "2.39:1": AspectRatio(239, 100),
    "21:9": AspectRatio(21, 9),
    "1:1": AspectRatio(1, 1),
    "9:16": AspectRatio(9, 16),
}


def detect_letterbox(video_path: Path) -> LetterboxAnalysis:
    """Convenience function to detect letterbox.

    Args:
        video_path: Path to video

    Returns:
        LetterboxAnalysis
    """
    detector = LetterboxDetector()
    return detector.analyze(video_path)


def crop_letterbox(
    input_path: Path,
    output_path: Path,
    auto_detect: bool = True,
    crop_region: Optional[CropRegion] = None,
) -> Tuple[bool, LetterboxAnalysis]:
    """Convenience function to crop letterbox.

    Args:
        input_path: Input video
        output_path: Output video
        auto_detect: Auto-detect crop region
        crop_region: Manual crop region (if auto_detect=False)

    Returns:
        Tuple of (success, analysis)
    """
    cropper = LetterboxCropper()

    if auto_detect:
        return cropper.auto_crop(input_path, output_path)
    elif crop_region:
        detector = LetterboxDetector()
        analysis = detector.analyze(input_path)
        success = cropper.crop(input_path, output_path, crop_region)
        return success, analysis
    else:
        return False, LetterboxAnalysis()
