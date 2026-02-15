"""Aspect ratio correction processor for video restoration.

This module handles aspect ratio detection and correction for various source formats,
including PAL squeeze, anamorphic video, and letterbox/pillarbox removal.

Features:
- Automatic aspect ratio detection
- PAL squeeze correction (720x576 to proper display ratio)
- Anamorphic (2x horizontal squeeze) handling
- Letterbox and pillarbox detection and removal
- Multiple correction methods (scale, crop, pad)

Example:
    >>> from pathlib import Path
    >>> from framewright.processors.aspect_correction import (
    ...     AspectCorrector,
    ...     AspectConfig,
    ...     AspectRatio,
    ... )
    >>> config = AspectConfig(target_ratio=AspectRatio.RATIO_16_9)
    >>> corrector = AspectCorrector(config)
    >>> analysis = corrector.analyze(Path("video.mp4"))
    >>> print(f"Detected: {analysis.detected_ratio:.3f}")
    >>> corrector.correct(Path("video.mp4"), Path("output.mp4"))

    >>> # Or use the convenience function
    >>> from framewright.processors.aspect_correction import fix_aspect_ratio
    >>> output = fix_aspect_ratio(Path("video.mp4"), target="16:9")
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from framewright.utils.dependencies import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger(__name__)


# Optional imports with fallback handling
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None


class AspectRatio(Enum):
    """Common aspect ratios for video content.

    Each value is a tuple of (width, height) representing the ratio.
    """
    # Standard television ratios
    RATIO_4_3 = (4, 3)  # Classic TV, early digital
    RATIO_16_9 = (16, 9)  # Modern HD/UHD standard
    RATIO_1_1 = (1, 1)  # Square (Instagram, etc.)

    # Widescreen cinema ratios
    RATIO_21_9 = (21, 9)  # Ultra-widescreen
    RATIO_1_85_1 = (185, 100)  # Academy Flat (US cinema standard)
    RATIO_2_35_1 = (235, 100)  # CinemaScope / anamorphic
    RATIO_2_39_1 = (239, 100)  # Modern anamorphic (SMPTE DCP)

    # Special formats
    ANAMORPHIC_2X = (32, 9)  # 2x horizontal anamorphic squeeze (16:9 * 2)

    @property
    def decimal(self) -> float:
        """Get the aspect ratio as a decimal value."""
        return self.value[0] / self.value[1]

    @property
    def display_name(self) -> str:
        """Get human-readable name for the aspect ratio."""
        names = {
            AspectRatio.RATIO_4_3: "4:3",
            AspectRatio.RATIO_16_9: "16:9",
            AspectRatio.RATIO_1_1: "1:1",
            AspectRatio.RATIO_21_9: "21:9",
            AspectRatio.RATIO_1_85_1: "1.85:1",
            AspectRatio.RATIO_2_35_1: "2.35:1",
            AspectRatio.RATIO_2_39_1: "2.39:1",
            AspectRatio.ANAMORPHIC_2X: "Anamorphic 2x",
        }
        return names.get(self, f"{self.value[0]}:{self.value[1]}")

    @classmethod
    def from_string(cls, ratio_str: str) -> Optional["AspectRatio"]:
        """Parse aspect ratio from string representation.

        Args:
            ratio_str: String like "16:9", "4:3", "2.35:1", etc.

        Returns:
            AspectRatio enum value or None if not recognized.
        """
        mapping = {
            "4:3": cls.RATIO_4_3,
            "16:9": cls.RATIO_16_9,
            "1:1": cls.RATIO_1_1,
            "21:9": cls.RATIO_21_9,
            "1.85:1": cls.RATIO_1_85_1,
            "2.35:1": cls.RATIO_2_35_1,
            "2.39:1": cls.RATIO_2_39_1,
            "anamorphic": cls.ANAMORPHIC_2X,
            "anamorphic_2x": cls.ANAMORPHIC_2X,
        }
        return mapping.get(ratio_str.lower().replace(" ", ""))

    @classmethod
    def closest_match(cls, ratio: float, tolerance: float = 0.05) -> Optional["AspectRatio"]:
        """Find the closest standard aspect ratio to a given decimal value.

        Args:
            ratio: Decimal aspect ratio (e.g., 1.778 for 16:9).
            tolerance: Maximum allowed deviation (default 5%).

        Returns:
            Closest AspectRatio or None if no match within tolerance.
        """
        best_match = None
        best_diff = float('inf')

        for ar in cls:
            diff = abs(ar.decimal - ratio)
            if diff < best_diff and diff / ratio < tolerance:
                best_diff = diff
                best_match = ar

        return best_match


class CorrectionMethod(Enum):
    """Methods for correcting aspect ratio."""
    SCALE = "scale"  # Scale to target ratio (may stretch)
    CROP = "crop"  # Crop to target ratio (lose content)
    PAD = "pad"  # Add letterbox/pillarbox (add black bars)
    FIT = "fit"  # Scale to fit within bounds (may add bars)


@dataclass
class AspectConfig:
    """Configuration for aspect ratio correction.

    Attributes:
        target_ratio: Target aspect ratio. None means auto-detect and correct.
        method: How to achieve target ratio (scale, crop, pad, fit).
        sample_aspect_ratio: Pixel aspect ratio as string (e.g., "32:27" for PAL).
            None means square pixels (1:1).
        detect_letterbox: Whether to detect existing letterbox/pillarbox.
        crop_letterbox: Whether to remove detected letterbox/pillarbox.
        letterbox_threshold: Black level threshold for letterbox detection (0-255).
        min_letterbox_height: Minimum height (px) to consider as letterbox.
        preserve_resolution: Keep original resolution when possible.
        upscale_allowed: Allow upscaling during correction.
    """
    target_ratio: Optional[AspectRatio] = None
    method: CorrectionMethod = CorrectionMethod.SCALE
    sample_aspect_ratio: Optional[str] = None  # e.g., "32:27" for PAL 16:9
    detect_letterbox: bool = True
    crop_letterbox: bool = True
    letterbox_threshold: int = 16  # Black level threshold
    min_letterbox_height: int = 8  # Minimum letterbox size in pixels
    preserve_resolution: bool = True
    upscale_allowed: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.letterbox_threshold < 0 or self.letterbox_threshold > 255:
            raise ValueError("letterbox_threshold must be between 0 and 255")
        if self.min_letterbox_height < 0:
            raise ValueError("min_letterbox_height must be non-negative")


@dataclass
class AspectAnalysis:
    """Analysis results for video aspect ratio.

    Attributes:
        detected_ratio: Detected display aspect ratio (width/height).
        display_ratio: Aspect ratio as it should be displayed (DAR).
        storage_ratio: Aspect ratio of stored pixels (SAR).
        pixel_aspect_ratio: Ratio of pixel width to height (PAR).
        has_letterbox: Whether letterboxing was detected.
        letterbox_top: Height of top black bar in pixels.
        letterbox_bottom: Height of bottom black bar in pixels.
        has_pillarbox: Whether pillarboxing was detected.
        pillarbox_left: Width of left black bar in pixels.
        pillarbox_right: Width of right black bar in pixels.
        is_anamorphic: Whether video appears to be anamorphic.
        anamorphic_factor: Estimated squeeze factor (e.g., 2.0 for 2x squeeze).
        is_pal_squeeze: Whether PAL 4:3/16:9 squeeze is detected.
        content_width: Width of actual content (excluding bars).
        content_height: Height of actual content (excluding bars).
        suggested_correction: Human-readable correction suggestion.
        confidence: Confidence in the analysis (0.0-1.0).
    """
    detected_ratio: float
    display_ratio: float
    storage_ratio: float
    pixel_aspect_ratio: float = 1.0
    has_letterbox: bool = False
    letterbox_top: int = 0
    letterbox_bottom: int = 0
    has_pillarbox: bool = False
    pillarbox_left: int = 0
    pillarbox_right: int = 0
    is_anamorphic: bool = False
    anamorphic_factor: float = 1.0
    is_pal_squeeze: bool = False
    content_width: int = 0
    content_height: int = 0
    suggested_correction: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "detected_ratio": self.detected_ratio,
            "display_ratio": self.display_ratio,
            "storage_ratio": self.storage_ratio,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "has_letterbox": self.has_letterbox,
            "letterbox_top": self.letterbox_top,
            "letterbox_bottom": self.letterbox_bottom,
            "has_pillarbox": self.has_pillarbox,
            "pillarbox_left": self.pillarbox_left,
            "pillarbox_right": self.pillarbox_right,
            "is_anamorphic": self.is_anamorphic,
            "anamorphic_factor": self.anamorphic_factor,
            "is_pal_squeeze": self.is_pal_squeeze,
            "content_width": self.content_width,
            "content_height": self.content_height,
            "suggested_correction": self.suggested_correction,
            "confidence": self.confidence,
        }


class AspectCorrectionError(Exception):
    """Exception raised for aspect correction errors."""
    pass


class AspectCorrector:
    """Aspect ratio analyzer and corrector for video files.

    This class provides comprehensive aspect ratio handling including:
    - Detection of display and storage aspect ratios
    - Letterbox/pillarbox detection and removal
    - PAL squeeze correction
    - Anamorphic video handling

    Example:
        >>> config = AspectConfig(target_ratio=AspectRatio.RATIO_16_9)
        >>> corrector = AspectCorrector(config)
        >>> analysis = corrector.analyze(Path("video.mp4"))
        >>> if analysis.has_letterbox:
        ...     corrector.correct(Path("video.mp4"), Path("clean.mp4"))
    """

    def __init__(self, config: Optional[AspectConfig] = None):
        """Initialize the aspect corrector.

        Args:
            config: Aspect correction configuration. Uses defaults if None.
        """
        self.config = config or AspectConfig()
        self._ffmpeg = get_ffmpeg_path()
        self._ffprobe = get_ffprobe_path()

    def analyze(self, video_path: Path) -> AspectAnalysis:
        """Analyze video aspect ratio and detect issues.

        Args:
            video_path: Path to input video file.

        Returns:
            AspectAnalysis with detected aspect ratio information.

        Raises:
            AspectCorrectionError: If video cannot be analyzed.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AspectCorrectionError(f"Video file not found: {video_path}")

        # Get video metadata
        metadata = self._get_video_metadata(video_path)

        width = metadata.get("width", 0)
        height = metadata.get("height", 0)

        if width == 0 or height == 0:
            raise AspectCorrectionError(f"Invalid video dimensions: {width}x{height}")

        storage_ratio = width / height

        # Parse sample aspect ratio (pixel aspect ratio)
        par = self._parse_aspect_ratio(
            metadata.get("sample_aspect_ratio", "1:1")
        )

        # Parse display aspect ratio
        dar = self._parse_aspect_ratio(
            metadata.get("display_aspect_ratio", f"{width}:{height}")
        )

        # If DAR is not set, calculate from SAR and PAR
        if dar == storage_ratio:
            dar = storage_ratio * par

        # Detect letterbox and pillarbox
        letterbox_top = 0
        letterbox_bottom = 0
        pillarbox_left = 0
        pillarbox_right = 0
        has_letterbox = False
        has_pillarbox = False

        if self.config.detect_letterbox and HAS_OPENCV:
            letterbox_top, letterbox_bottom = self._detect_letterbox(video_path, width, height)
            pillarbox_left, pillarbox_right = self._detect_pillarbox(video_path, width, height)

            has_letterbox = (letterbox_top + letterbox_bottom) >= self.config.min_letterbox_height
            has_pillarbox = (pillarbox_left + pillarbox_right) >= self.config.min_letterbox_height

        # Calculate content dimensions (excluding bars)
        content_width = width - pillarbox_left - pillarbox_right
        content_height = height - letterbox_top - letterbox_bottom

        # Calculate detected ratio from content
        if content_height > 0:
            detected_ratio = (content_width / content_height) * par
        else:
            detected_ratio = dar

        # Detect PAL squeeze
        is_pal_squeeze = self._detect_pal_squeeze(width, height, storage_ratio, par)

        # Detect anamorphic
        is_anamorphic, anamorphic_factor = self._detect_anamorphic(
            storage_ratio, dar, detected_ratio
        )

        # Generate correction suggestion
        suggestion = self._generate_suggestion(
            detected_ratio=detected_ratio,
            display_ratio=dar,
            has_letterbox=has_letterbox,
            has_pillarbox=has_pillarbox,
            is_anamorphic=is_anamorphic,
            is_pal_squeeze=is_pal_squeeze,
            letterbox_top=letterbox_top,
            letterbox_bottom=letterbox_bottom,
        )

        # Calculate confidence based on detection quality
        confidence = 1.0
        if has_letterbox or has_pillarbox:
            confidence = 0.9  # Letterbox detection adds some uncertainty
        if is_pal_squeeze:
            confidence = 0.85
        if is_anamorphic:
            confidence = 0.8

        return AspectAnalysis(
            detected_ratio=detected_ratio,
            display_ratio=dar,
            storage_ratio=storage_ratio,
            pixel_aspect_ratio=par,
            has_letterbox=has_letterbox,
            letterbox_top=letterbox_top,
            letterbox_bottom=letterbox_bottom,
            has_pillarbox=has_pillarbox,
            pillarbox_left=pillarbox_left,
            pillarbox_right=pillarbox_right,
            is_anamorphic=is_anamorphic,
            anamorphic_factor=anamorphic_factor,
            is_pal_squeeze=is_pal_squeeze,
            content_width=content_width,
            content_height=content_height,
            suggested_correction=suggestion,
            confidence=confidence,
        )

    def correct(
        self,
        input_path: Path,
        output_path: Path,
        target: Optional[AspectRatio] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Apply aspect ratio correction to video.

        Args:
            input_path: Path to input video file.
            output_path: Path for corrected output file.
            target: Target aspect ratio. Uses config value if None.
            progress_callback: Optional callback for progress updates (0.0-1.0).

        Returns:
            Path to the corrected video file.

        Raises:
            AspectCorrectionError: If correction fails.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Analyze input
        analysis = self.analyze(input_path)
        target = target or self.config.target_ratio

        # Build filter chain
        filters = []

        # Step 1: Remove letterbox/pillarbox if detected and configured
        if self.config.crop_letterbox and (analysis.has_letterbox or analysis.has_pillarbox):
            crop_filter = self._build_crop_filter(
                analysis.letterbox_top,
                analysis.letterbox_bottom,
                analysis.pillarbox_left,
                analysis.pillarbox_right,
                analysis.content_width,
                analysis.content_height,
            )
            if crop_filter:
                filters.append(crop_filter)

        # Step 2: Handle PAL squeeze
        if analysis.is_pal_squeeze:
            pal_filter = self._build_pal_correction_filter(analysis)
            if pal_filter:
                filters.append(pal_filter)

        # Step 3: Handle anamorphic
        if analysis.is_anamorphic:
            anamorphic_filter = self._build_anamorphic_correction_filter(
                analysis.anamorphic_factor
            )
            if anamorphic_filter:
                filters.append(anamorphic_filter)

        # Step 4: Apply target ratio correction if specified
        if target is not None:
            target_filter = self._build_target_ratio_filter(
                analysis, target, self.config.method
            )
            if target_filter:
                filters.append(target_filter)

        # Step 5: Set proper pixel aspect ratio
        filters.append("setsar=1:1")  # Ensure square pixels in output

        # Execute FFmpeg
        self._apply_filters(input_path, output_path, filters, progress_callback)

        logger.info(f"Aspect correction complete: {output_path}")
        return output_path

    def _detect_letterbox(self, video_path: Path, width: int, height: int) -> Tuple[int, int]:
        """Detect letterbox (horizontal black bars) in video.

        Args:
            video_path: Path to video file.
            width: Video width.
            height: Video height.

        Returns:
            Tuple of (top_bar_height, bottom_bar_height).
        """
        if not HAS_OPENCV:
            logger.warning("OpenCV not available for letterbox detection")
            return 0, 0

        # Sample multiple frames for robust detection
        frames = self._sample_frames(video_path, num_samples=5)
        if not frames:
            return 0, 0

        # Analyze each frame
        top_bars = []
        bottom_bars = []

        for frame in frames:
            top, bottom = self._analyze_frame_letterbox(frame)
            top_bars.append(top)
            bottom_bars.append(bottom)

        # Use median to filter outliers
        top = int(np.median(top_bars)) if top_bars else 0
        bottom = int(np.median(bottom_bars)) if bottom_bars else 0

        logger.debug(f"Detected letterbox: top={top}px, bottom={bottom}px")
        return top, bottom

    def _detect_pillarbox(self, video_path: Path, width: int, height: int) -> Tuple[int, int]:
        """Detect pillarbox (vertical black bars) in video.

        Args:
            video_path: Path to video file.
            width: Video width.
            height: Video height.

        Returns:
            Tuple of (left_bar_width, right_bar_width).
        """
        if not HAS_OPENCV:
            logger.warning("OpenCV not available for pillarbox detection")
            return 0, 0

        # Sample multiple frames for robust detection
        frames = self._sample_frames(video_path, num_samples=5)
        if not frames:
            return 0, 0

        # Analyze each frame
        left_bars = []
        right_bars = []

        for frame in frames:
            left, right = self._analyze_frame_pillarbox(frame)
            left_bars.append(left)
            right_bars.append(right)

        # Use median to filter outliers
        left = int(np.median(left_bars)) if left_bars else 0
        right = int(np.median(right_bars)) if right_bars else 0

        logger.debug(f"Detected pillarbox: left={left}px, right={right}px")
        return left, right

    def _analyze_frame_letterbox(self, frame: np.ndarray) -> Tuple[int, int]:
        """Analyze a single frame for letterbox bars.

        Args:
            frame: Frame as numpy array (BGR or grayscale).

        Returns:
            Tuple of (top_bar_height, bottom_bar_height).
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        height = gray.shape[0]
        threshold = self.config.letterbox_threshold
        min_size = self.config.min_letterbox_height

        # Detect top bar
        top = 0
        for y in range(height // 3):  # Only check top third
            row_mean = np.mean(gray[y, :])
            if row_mean > threshold:
                top = y
                break

        # Detect bottom bar
        bottom = 0
        for y in range(height - 1, 2 * height // 3, -1):  # Only check bottom third
            row_mean = np.mean(gray[y, :])
            if row_mean > threshold:
                bottom = height - 1 - y
                break

        # Apply minimum size filter
        if top < min_size:
            top = 0
        if bottom < min_size:
            bottom = 0

        return top, bottom

    def _analyze_frame_pillarbox(self, frame: np.ndarray) -> Tuple[int, int]:
        """Analyze a single frame for pillarbox bars.

        Args:
            frame: Frame as numpy array (BGR or grayscale).

        Returns:
            Tuple of (left_bar_width, right_bar_width).
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        width = gray.shape[1]
        threshold = self.config.letterbox_threshold
        min_size = self.config.min_letterbox_height  # Same threshold for pillarbox

        # Detect left bar
        left = 0
        for x in range(width // 3):  # Only check left third
            col_mean = np.mean(gray[:, x])
            if col_mean > threshold:
                left = x
                break

        # Detect right bar
        right = 0
        for x in range(width - 1, 2 * width // 3, -1):  # Only check right third
            col_mean = np.mean(gray[:, x])
            if col_mean > threshold:
                right = width - 1 - x
                break

        # Apply minimum size filter
        if left < min_size:
            left = 0
        if right < min_size:
            right = 0

        return left, right

    def _detect_aspect_ratio(self, video_path: Path) -> float:
        """Detect the display aspect ratio of a video.

        Args:
            video_path: Path to video file.

        Returns:
            Detected aspect ratio as float.
        """
        analysis = self.analyze(video_path)
        return analysis.detected_ratio

    def _correct_pal_squeeze(
        self,
        input_path: Path,
        output_path: Path,
        target_ratio: AspectRatio = AspectRatio.RATIO_16_9,
    ) -> Path:
        """Correct PAL squeeze distortion.

        PAL video at 720x576 is often stored with non-square pixels.
        This corrects to proper display aspect ratio.

        Args:
            input_path: Path to input video.
            output_path: Path for corrected output.
            target_ratio: Target aspect ratio (4:3 or 16:9).

        Returns:
            Path to corrected video.
        """
        # Determine correct SAR for target ratio
        if target_ratio == AspectRatio.RATIO_16_9:
            sar = "32:27"  # 16:9 PAL
        elif target_ratio == AspectRatio.RATIO_4_3:
            sar = "16:15"  # 4:3 PAL
        else:
            sar = "1:1"

        filters = [
            f"setsar={sar}",
            "scale=iw*sar:ih",  # Apply SAR to width
            "setsar=1:1",  # Reset to square pixels
        ]

        self._apply_filters(input_path, output_path, filters)
        return output_path

    def _remove_letterbox(
        self,
        input_path: Path,
        output_path: Path,
        top: int,
        bottom: int,
    ) -> Path:
        """Remove letterbox bars from video.

        Args:
            input_path: Path to input video.
            output_path: Path for cropped output.
            top: Height of top bar to remove.
            bottom: Height of bottom bar to remove.

        Returns:
            Path to cropped video.
        """
        metadata = self._get_video_metadata(input_path)
        width = metadata.get("width", 0)
        height = metadata.get("height", 0)

        crop_height = height - top - bottom
        crop_y = top

        filters = [f"crop={width}:{crop_height}:0:{crop_y}"]

        self._apply_filters(input_path, output_path, filters)
        return output_path

    def _get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using ffprobe.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary with video metadata.
        """
        cmd = [
            str(self._ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)

            if data.get("streams"):
                stream = data["streams"][0]
                return {
                    "width": stream.get("width", 0),
                    "height": stream.get("height", 0),
                    "sample_aspect_ratio": stream.get("sample_aspect_ratio", "1:1"),
                    "display_aspect_ratio": stream.get("display_aspect_ratio", ""),
                    "duration": float(stream.get("duration", 0)),
                    "fps": self._parse_fps(stream.get("r_frame_rate", "24/1")),
                    "codec": stream.get("codec_name", ""),
                }
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to get video metadata: {e}")

        return {}

    def _parse_aspect_ratio(self, ratio_str: str) -> float:
        """Parse aspect ratio string to float.

        Args:
            ratio_str: Ratio string like "16:9" or "1.778".

        Returns:
            Aspect ratio as float.
        """
        if not ratio_str or ratio_str == "N/A" or ratio_str == "0:1":
            return 1.0

        if ":" in ratio_str:
            parts = ratio_str.split(":")
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return 1.0
        else:
            try:
                return float(ratio_str)
            except ValueError:
                return 1.0

    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS string to float.

        Args:
            fps_str: FPS string like "30/1" or "29.97".

        Returns:
            FPS as float.
        """
        if "/" in fps_str:
            parts = fps_str.split("/")
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return 24.0
        else:
            try:
                return float(fps_str)
            except ValueError:
                return 24.0

    def _detect_pal_squeeze(
        self,
        width: int,
        height: int,
        storage_ratio: float,
        par: float,
    ) -> bool:
        """Detect if video has PAL squeeze distortion.

        Args:
            width: Video width.
            height: Video height.
            storage_ratio: Storage aspect ratio.
            par: Pixel aspect ratio.

        Returns:
            True if PAL squeeze is detected.
        """
        # Common PAL resolutions
        pal_resolutions = [
            (720, 576),  # PAL full
            (704, 576),  # PAL DV
            (720, 480),  # NTSC (for comparison)
            (704, 480),  # NTSC DV
        ]

        is_pal_resolution = (width, height) in pal_resolutions

        # Check for non-square pixels
        has_non_square_pixels = abs(par - 1.0) > 0.01

        return is_pal_resolution and has_non_square_pixels

    def _detect_anamorphic(
        self,
        storage_ratio: float,
        display_ratio: float,
        detected_ratio: float,
    ) -> Tuple[bool, float]:
        """Detect if video is anamorphic.

        Args:
            storage_ratio: Storage aspect ratio.
            display_ratio: Display aspect ratio.
            detected_ratio: Detected content aspect ratio.

        Returns:
            Tuple of (is_anamorphic, squeeze_factor).
        """
        # Calculate squeeze factor
        if storage_ratio > 0:
            squeeze_factor = display_ratio / storage_ratio
        else:
            squeeze_factor = 1.0

        # Anamorphic typically has 2x squeeze for 2.35:1 or 2.39:1 content
        # stored in 16:9 container
        is_anamorphic = squeeze_factor > 1.3  # More than 30% difference suggests anamorphic

        # Round to common factors
        if 1.9 < squeeze_factor < 2.1:
            squeeze_factor = 2.0
        elif 1.4 < squeeze_factor < 1.6:
            squeeze_factor = 1.5
        elif 1.2 < squeeze_factor < 1.4:
            squeeze_factor = 1.33

        return is_anamorphic, squeeze_factor

    def _generate_suggestion(
        self,
        detected_ratio: float,
        display_ratio: float,
        has_letterbox: bool,
        has_pillarbox: bool,
        is_anamorphic: bool,
        is_pal_squeeze: bool,
        letterbox_top: int,
        letterbox_bottom: int,
    ) -> str:
        """Generate human-readable correction suggestion.

        Returns:
            Suggestion string.
        """
        suggestions = []

        if has_letterbox:
            total = letterbox_top + letterbox_bottom
            suggestions.append(f"Remove {total}px letterbox (top: {letterbox_top}px, bottom: {letterbox_bottom}px)")

        if has_pillarbox:
            suggestions.append("Remove pillarbox (side bars)")

        if is_pal_squeeze:
            suggestions.append("Apply PAL squeeze correction")

        if is_anamorphic:
            suggestions.append("De-squeeze anamorphic video")

        # Match to standard ratio
        matched = AspectRatio.closest_match(detected_ratio)
        if matched:
            suggestions.append(f"Target ratio: {matched.display_name}")
        else:
            suggestions.append(f"Non-standard ratio: {detected_ratio:.3f}")

        return "; ".join(suggestions) if suggestions else "No correction needed"

    def _sample_frames(self, video_path: Path, num_samples: int = 5) -> List[np.ndarray]:
        """Sample frames from video for analysis.

        Args:
            video_path: Path to video file.
            num_samples: Number of frames to sample.

        Returns:
            List of frames as numpy arrays.
        """
        if not HAS_OPENCV:
            return []

        frames = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.warning(f"Could not open video for frame sampling: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Estimate from duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = 60  # Default assumption
            total_frames = int(fps * duration)

        # Sample frames evenly distributed through video
        # Skip first and last 5% to avoid credits/black frames
        start_frame = int(total_frames * 0.05)
        end_frame = int(total_frames * 0.95)

        if end_frame <= start_frame:
            start_frame = 0
            end_frame = total_frames - 1

        step = max(1, (end_frame - start_frame) // (num_samples + 1))

        for i in range(num_samples):
            frame_num = start_frame + (i + 1) * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def _build_crop_filter(
        self,
        top: int,
        bottom: int,
        left: int,
        right: int,
        content_width: int,
        content_height: int,
    ) -> Optional[str]:
        """Build FFmpeg crop filter string.

        Returns:
            Crop filter string or None if no cropping needed.
        """
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return None

        return f"crop={content_width}:{content_height}:{left}:{top}"

    def _build_pal_correction_filter(self, analysis: AspectAnalysis) -> Optional[str]:
        """Build FFmpeg filter for PAL squeeze correction.

        Returns:
            Filter string or None.
        """
        if not analysis.is_pal_squeeze:
            return None

        # Determine if source is 4:3 or 16:9 based on DAR
        if analysis.display_ratio > 1.5:
            # 16:9 content (DAR ~1.778)
            target_width = int(analysis.content_height * (16 / 9))
        else:
            # 4:3 content (DAR ~1.333)
            target_width = int(analysis.content_height * (4 / 3))

        return f"scale={target_width}:{analysis.content_height}"

    def _build_anamorphic_correction_filter(self, squeeze_factor: float) -> Optional[str]:
        """Build FFmpeg filter for anamorphic correction.

        Args:
            squeeze_factor: Horizontal squeeze factor (e.g., 2.0).

        Returns:
            Filter string or None.
        """
        if squeeze_factor <= 1.1:
            return None

        # Scale width by squeeze factor
        return f"scale=iw*{squeeze_factor}:ih"

    def _build_target_ratio_filter(
        self,
        analysis: AspectAnalysis,
        target: AspectRatio,
        method: CorrectionMethod,
    ) -> Optional[str]:
        """Build FFmpeg filter to achieve target aspect ratio.

        Returns:
            Filter string or None.
        """
        current_ratio = analysis.detected_ratio
        target_ratio = target.decimal

        if abs(current_ratio - target_ratio) < 0.01:
            return None  # Already correct

        if method == CorrectionMethod.SCALE:
            # Scale to target ratio (may distort)
            return f"scale=ih*{target_ratio}:ih"

        elif method == CorrectionMethod.CROP:
            # Crop to target ratio
            if current_ratio > target_ratio:
                # Too wide, crop width
                return f"crop=ih*{target_ratio}:ih"
            else:
                # Too tall, crop height
                return f"crop=iw:iw/{target_ratio}"

        elif method == CorrectionMethod.PAD:
            # Pad to target ratio (add bars)
            if current_ratio > target_ratio:
                # Too wide, add top/bottom bars
                return f"pad=iw:iw/{target_ratio}:(ow-iw)/2:(oh-ih)/2:black"
            else:
                # Too tall, add side bars
                return f"pad=ih*{target_ratio}:ih:(ow-iw)/2:(oh-ih)/2:black"

        elif method == CorrectionMethod.FIT:
            # Scale to fit, then pad
            return f"scale='min(iw,ih*{target_ratio})':'min(ih,iw/{target_ratio})',pad=ih*{target_ratio}:ih:(ow-iw)/2:(oh-ih)/2:black"

        return None

    def _apply_filters(
        self,
        input_path: Path,
        output_path: Path,
        filters: List[str],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Apply FFmpeg filter chain to video.

        Args:
            input_path: Path to input video.
            output_path: Path for output video.
            filters: List of FFmpeg filter strings.
            progress_callback: Optional progress callback.

        Raises:
            AspectCorrectionError: If FFmpeg fails.
        """
        if not filters:
            # No filters, just copy
            shutil.copy2(input_path, output_path)
            return

        filter_chain = ",".join(filters)

        cmd = [
            str(self._ffmpeg),
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-vf", filter_chain,
            "-c:a", "copy",  # Copy audio
            "-c:v", "libx264",  # Use H.264 for compatibility
            "-preset", "medium",
            "-crf", "18",  # High quality
            str(output_path),
        ]

        logger.debug(f"Running FFmpeg: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            if progress_callback:
                progress_callback(1.0)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise AspectCorrectionError(f"FFmpeg failed: {e.stderr}")


def fix_aspect_ratio(
    video_path: Path,
    target: str = "auto",
    output_path: Optional[Path] = None,
    method: str = "scale",
    crop_letterbox: bool = True,
) -> Path:
    """Convenience function to fix aspect ratio issues in a video.

    Args:
        video_path: Path to input video file.
        target: Target aspect ratio as string ("16:9", "4:3", "auto", etc.).
        output_path: Path for output file. Auto-generated if None.
        method: Correction method ("scale", "crop", "pad", "fit").
        crop_letterbox: Whether to remove letterbox/pillarbox.

    Returns:
        Path to the corrected video file.

    Example:
        >>> output = fix_aspect_ratio(Path("old_vhs.mp4"), target="16:9")
        >>> output = fix_aspect_ratio(Path("movie.mp4"), target="auto", crop_letterbox=True)
    """
    video_path = Path(video_path)

    # Generate output path if not provided
    if output_path is None:
        stem = video_path.stem
        suffix = video_path.suffix
        output_path = video_path.parent / f"{stem}_aspect_corrected{suffix}"

    # Parse target ratio
    target_ratio = None
    if target and target.lower() != "auto":
        target_ratio = AspectRatio.from_string(target)
        if target_ratio is None:
            logger.warning(f"Unknown target ratio '{target}', using auto detection")

    # Parse method
    try:
        correction_method = CorrectionMethod(method)
    except ValueError:
        correction_method = CorrectionMethod.SCALE

    # Create config
    config = AspectConfig(
        target_ratio=target_ratio,
        method=correction_method,
        crop_letterbox=crop_letterbox,
    )

    # Run correction
    corrector = AspectCorrector(config)

    # If auto mode, analyze first and find best target
    if target_ratio is None:
        analysis = corrector.analyze(video_path)
        matched = AspectRatio.closest_match(analysis.detected_ratio)
        if matched:
            target_ratio = matched
            logger.info(f"Auto-detected target ratio: {matched.display_name}")

    return corrector.correct(video_path, output_path, target_ratio)


def create_aspect_corrector(
    target: Optional[str] = None,
    method: str = "scale",
    crop_letterbox: bool = True,
) -> AspectCorrector:
    """Factory function to create an AspectCorrector with common settings.

    Args:
        target: Target aspect ratio string (e.g., "16:9", "4:3").
        method: Correction method ("scale", "crop", "pad", "fit").
        crop_letterbox: Whether to remove letterbox/pillarbox.

    Returns:
        Configured AspectCorrector instance.
    """
    target_ratio = AspectRatio.from_string(target) if target else None

    try:
        correction_method = CorrectionMethod(method)
    except ValueError:
        correction_method = CorrectionMethod.SCALE

    config = AspectConfig(
        target_ratio=target_ratio,
        method=correction_method,
        crop_letterbox=crop_letterbox,
    )

    return AspectCorrector(config)
