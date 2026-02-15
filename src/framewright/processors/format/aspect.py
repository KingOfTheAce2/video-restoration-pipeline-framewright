"""Aspect ratio format processor for FrameWright.

Comprehensive aspect ratio detection and correction:
- Detect actual content aspect ratio
- Detect and remove letterbox/pillarbox
- Convert between aspect ratios
- Add pillarbox/letterbox for target ratios
- Support for common ratios: 4:3, 16:9, 2.35:1, 1.85:1

Example:
    >>> from pathlib import Path
    >>> from framewright.processors.format.aspect import AspectHandler, AspectConfig
    >>> config = AspectConfig(target_ratio="16:9", method="crop", fill_mode="black")
    >>> handler = AspectHandler(config)
    >>> detected = handler.detect_aspect(frame)
    >>> if handler.detect_letterbox(frame):
    ...     frames = handler.crop_letterbox(frames)
"""

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None


class StandardAspectRatio(Enum):
    """Standard aspect ratios."""
    RATIO_4_3 = "4:3"           # Classic TV (1.333)
    RATIO_16_9 = "16:9"         # Modern HD (1.778)
    RATIO_1_85_1 = "1.85:1"     # Academy Flat (1.85)
    RATIO_2_35_1 = "2.35:1"     # CinemaScope (2.35)
    RATIO_2_39_1 = "2.39:1"     # Anamorphic (2.39)
    RATIO_21_9 = "21:9"         # Ultra-wide (2.333)
    RATIO_1_1 = "1:1"           # Square (1.0)
    RATIO_9_16 = "9:16"         # Vertical (0.5625)
    RATIO_IMAX = "1.43:1"       # IMAX (1.43)
    RATIO_70MM = "2.20:1"       # 70mm (2.20)

    @property
    def decimal(self) -> float:
        """Get ratio as decimal value."""
        ratios = {
            "4:3": 4/3,
            "16:9": 16/9,
            "1.85:1": 1.85,
            "2.35:1": 2.35,
            "2.39:1": 2.39,
            "21:9": 21/9,
            "1:1": 1.0,
            "9:16": 9/16,
            "1.43:1": 1.43,
            "2.20:1": 2.20,
        }
        return ratios.get(self.value, 16/9)

    @classmethod
    def from_decimal(cls, ratio: float, tolerance: float = 0.05) -> Optional["StandardAspectRatio"]:
        """Find closest standard ratio to decimal value."""
        for ar in cls:
            if abs(ar.decimal - ratio) < tolerance:
                return ar
        return None

    @classmethod
    def from_string(cls, ratio_str: str) -> Optional["StandardAspectRatio"]:
        """Parse ratio from string."""
        normalized = ratio_str.lower().strip()
        for ar in cls:
            if ar.value.lower() == normalized:
                return ar
        # Try parsing as decimal
        try:
            decimal = float(normalized.replace(":", "/").split("/")[0])
            return cls.from_decimal(decimal)
        except:
            return None


class ConversionMethod(Enum):
    """Methods for aspect ratio conversion."""
    CROP = "crop"           # Crop to fit (lose content)
    PAD = "pad"             # Add bars (letterbox/pillarbox)
    SCALE = "scale"         # Scale to fit (may distort)
    FIT = "fit"             # Scale and pad (no distortion, may add bars)
    STRETCH = "stretch"     # Stretch to fill (distorts)


class FillMode(Enum):
    """Fill mode for padding."""
    BLACK = "black"         # Black bars
    BLUR = "blur"           # Blurred edge content
    MIRROR = "mirror"       # Mirrored edge content
    COLOR = "color"         # Custom color


@dataclass
class AspectConfig:
    """Configuration for aspect ratio processing.

    Attributes:
        target_ratio: Target aspect ratio (string like "16:9" or decimal).
        method: Conversion method (crop, pad, scale, fit).
        fill_mode: Fill mode for padding (black, blur, mirror).
        fill_color: RGB tuple for COLOR fill mode.
        black_threshold: Threshold for black bar detection (0-255).
        min_bar_size: Minimum bar size in pixels to detect.
        sample_count: Number of frames to sample for detection.
        edge_tolerance: Pixels of tolerance for edge detection.
        preserve_center: When cropping, keep center of frame.
        align_to: Align output dimensions to this value (codec compatibility).
    """
    target_ratio: Optional[str] = None
    method: ConversionMethod = ConversionMethod.FIT
    fill_mode: FillMode = FillMode.BLACK
    fill_color: Tuple[int, int, int] = (0, 0, 0)
    black_threshold: int = 16
    min_bar_size: int = 4
    sample_count: int = 20
    edge_tolerance: int = 2
    preserve_center: bool = True
    align_to: int = 2

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.black_threshold <= 255:
            raise ValueError("black_threshold must be between 0 and 255")
        if self.min_bar_size < 0:
            raise ValueError("min_bar_size must be non-negative")


@dataclass
class CropRegion:
    """Region definition for cropping."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def as_ffmpeg_filter(self) -> str:
        """Return as FFmpeg crop filter."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass
class AspectAnalysis:
    """Results of aspect ratio analysis."""
    # Frame dimensions
    frame_width: int = 0
    frame_height: int = 0

    # Detected aspect ratio
    detected_ratio: float = 0.0
    detected_standard: Optional[StandardAspectRatio] = None

    # Letterbox/pillarbox detection
    has_letterbox: bool = False
    has_pillarbox: bool = False
    top_bar: int = 0
    bottom_bar: int = 0
    left_bar: int = 0
    right_bar: int = 0

    # Content region
    content_x: int = 0
    content_y: int = 0
    content_width: int = 0
    content_height: int = 0
    content_ratio: float = 0.0

    # Analysis metadata
    is_variable: bool = False
    confidence: float = 0.0

    @property
    def crop_region(self) -> CropRegion:
        """Get crop region to extract content."""
        return CropRegion(
            x=self.content_x,
            y=self.content_y,
            width=self.content_width,
            height=self.content_height,
        )

    @property
    def bar_percentage(self) -> float:
        """Percentage of frame that is black bars."""
        if self.frame_width == 0 or self.frame_height == 0:
            return 0.0
        total = self.frame_width * self.frame_height
        content = self.content_width * self.content_height
        return (1 - content / total) * 100 if total > 0 else 0.0

    def summary(self) -> str:
        """Get human-readable summary."""
        if not self.has_letterbox and not self.has_pillarbox:
            ratio_name = self.detected_standard.value if self.detected_standard else f"{self.detected_ratio:.2f}:1"
            return f"Full frame content at {ratio_name}"

        bar_info = []
        if self.has_letterbox:
            bar_info.append(f"Letterbox: top={self.top_bar}px, bottom={self.bottom_bar}px")
        if self.has_pillarbox:
            bar_info.append(f"Pillarbox: left={self.left_bar}px, right={self.right_bar}px")

        content_ratio_name = StandardAspectRatio.from_decimal(self.content_ratio)
        ratio_str = content_ratio_name.value if content_ratio_name else f"{self.content_ratio:.2f}:1"

        return (
            f"Frame: {self.frame_width}x{self.frame_height}\n"
            f"Content: {self.content_width}x{self.content_height} ({ratio_str})\n"
            f"{'; '.join(bar_info)}\n"
            f"Black bars: {self.bar_percentage:.1f}% of frame"
        )


class AspectHandler:
    """Main aspect ratio detection and conversion processor.

    Handles detection of actual content aspect ratio, letterbox/pillarbox
    detection and removal, and aspect ratio conversion.
    """

    def __init__(self, config: Optional[AspectConfig] = None):
        """Initialize aspect handler.

        Args:
            config: Aspect processing configuration.
        """
        self.config = config or AspectConfig()
        self._ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def detect_aspect(self, frame: Any) -> float:
        """Detect actual aspect ratio of frame content.

        Args:
            frame: Frame as numpy array.

        Returns:
            Detected aspect ratio as decimal.
        """
        if not HAS_OPENCV or frame is None:
            return 16/9  # Default to 16:9

        analysis = self.analyze_frame(frame)
        return analysis.content_ratio if analysis.content_ratio > 0 else analysis.detected_ratio

    def detect_letterbox(self, frame: Any) -> bool:
        """Check if frame has letterbox (horizontal black bars).

        Args:
            frame: Frame as numpy array.

        Returns:
            True if letterbox detected.
        """
        if not HAS_OPENCV or frame is None:
            return False

        analysis = self.analyze_frame(frame)
        return analysis.has_letterbox

    def analyze_frame(self, frame: Any) -> AspectAnalysis:
        """Analyze a single frame for aspect ratio and bars.

        Args:
            frame: Frame as numpy array.

        Returns:
            AspectAnalysis with detection results.
        """
        analysis = AspectAnalysis()

        if not HAS_OPENCV or frame is None:
            return analysis

        height, width = frame.shape[:2]
        analysis.frame_width = width
        analysis.frame_height = height
        analysis.detected_ratio = width / height if height > 0 else 1.0

        # Convert to grayscale for bar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Detect horizontal bars (letterbox)
        top_bar, bottom_bar = self._detect_horizontal_bars(gray)
        analysis.top_bar = top_bar
        analysis.bottom_bar = bottom_bar
        analysis.has_letterbox = (top_bar + bottom_bar) >= self.config.min_bar_size

        # Detect vertical bars (pillarbox)
        left_bar, right_bar = self._detect_vertical_bars(gray)
        analysis.left_bar = left_bar
        analysis.right_bar = right_bar
        analysis.has_pillarbox = (left_bar + right_bar) >= self.config.min_bar_size

        # Calculate content region
        analysis.content_x = left_bar
        analysis.content_y = top_bar
        analysis.content_width = width - left_bar - right_bar
        analysis.content_height = height - top_bar - bottom_bar

        # Calculate content aspect ratio
        if analysis.content_height > 0:
            analysis.content_ratio = analysis.content_width / analysis.content_height
        else:
            analysis.content_ratio = analysis.detected_ratio

        # Find closest standard ratio
        analysis.detected_standard = StandardAspectRatio.from_decimal(analysis.content_ratio)

        # Confidence based on how clean the detection was
        analysis.confidence = 1.0
        if analysis.has_letterbox or analysis.has_pillarbox:
            # Lower confidence if bars are asymmetric
            if analysis.has_letterbox and abs(top_bar - bottom_bar) > 5:
                analysis.confidence -= 0.1
            if analysis.has_pillarbox and abs(left_bar - right_bar) > 5:
                analysis.confidence -= 0.1

        return analysis

    def analyze(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> AspectAnalysis:
        """Analyze multiple frames for consistent aspect ratio.

        Args:
            frames: List of frames to analyze.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            AspectAnalysis with aggregated results.
        """
        if not frames or not HAS_OPENCV:
            return AspectAnalysis()

        # Sample frames
        sample_indices = np.linspace(
            0, len(frames) - 1,
            min(self.config.sample_count, len(frames)),
            dtype=int
        )

        analyses = []
        for i, idx in enumerate(sample_indices):
            analysis = self.analyze_frame(frames[idx])
            analyses.append(analysis)

            if progress_callback:
                progress_callback((i + 1) / len(sample_indices))

        # Aggregate results using median
        result = AspectAnalysis()
        result.frame_width = analyses[0].frame_width
        result.frame_height = analyses[0].frame_height
        result.detected_ratio = analyses[0].detected_ratio

        result.top_bar = int(np.median([a.top_bar for a in analyses]))
        result.bottom_bar = int(np.median([a.bottom_bar for a in analyses]))
        result.left_bar = int(np.median([a.left_bar for a in analyses]))
        result.right_bar = int(np.median([a.right_bar for a in analyses]))

        # Check for variability
        top_std = np.std([a.top_bar for a in analyses])
        bottom_std = np.std([a.bottom_bar for a in analyses])
        result.is_variable = top_std > 5 or bottom_std > 5

        # Apply minimum bar size filter
        if result.top_bar < self.config.min_bar_size:
            result.top_bar = 0
        if result.bottom_bar < self.config.min_bar_size:
            result.bottom_bar = 0
        if result.left_bar < self.config.min_bar_size:
            result.left_bar = 0
        if result.right_bar < self.config.min_bar_size:
            result.right_bar = 0

        result.has_letterbox = (result.top_bar + result.bottom_bar) > 0
        result.has_pillarbox = (result.left_bar + result.right_bar) > 0

        result.content_x = result.left_bar
        result.content_y = result.top_bar
        result.content_width = result.frame_width - result.left_bar - result.right_bar
        result.content_height = result.frame_height - result.top_bar - result.bottom_bar

        if result.content_height > 0:
            result.content_ratio = result.content_width / result.content_height

        result.detected_standard = StandardAspectRatio.from_decimal(result.content_ratio)

        # Calculate confidence
        consistency = 1.0 - min(1.0, (top_std + bottom_std) / 20)
        result.confidence = consistency

        return result

    def crop_letterbox(
        self,
        frames: List[Any],
        analysis: Optional[AspectAnalysis] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Remove black bars from frames.

        Args:
            frames: List of frames to process.
            analysis: Pre-computed analysis (auto-detect if None).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Cropped frames.
        """
        if not frames or not HAS_OPENCV:
            return frames

        # Auto-detect if analysis not provided
        if analysis is None:
            analysis = self.analyze(frames[:min(20, len(frames))])

        if not analysis.has_letterbox and not analysis.has_pillarbox:
            return frames

        result = []
        crop = analysis.crop_region

        for i, frame in enumerate(frames):
            # Align dimensions
            x = crop.x
            y = crop.y
            w = (crop.width // self.config.align_to) * self.config.align_to
            h = (crop.height // self.config.align_to) * self.config.align_to

            if len(frame.shape) == 3:
                cropped = frame[y:y+h, x:x+w, :]
            else:
                cropped = frame[y:y+h, x:x+w]

            result.append(cropped)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def convert_aspect(
        self,
        frames: List[Any],
        target: Union[str, StandardAspectRatio, float],
        method: Optional[ConversionMethod] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Convert frames to target aspect ratio.

        Args:
            frames: List of frames to process.
            target: Target aspect ratio.
            method: Conversion method (uses config if None).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Converted frames.
        """
        if not frames or not HAS_OPENCV:
            return frames

        # Parse target ratio
        if isinstance(target, StandardAspectRatio):
            target_ratio = target.decimal
        elif isinstance(target, str):
            std = StandardAspectRatio.from_string(target)
            target_ratio = std.decimal if std else 16/9
        else:
            target_ratio = float(target)

        method = method or self.config.method

        result = []
        for i, frame in enumerate(frames):
            converted = self._convert_frame(frame, target_ratio, method)
            result.append(converted)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def add_pillarbox(
        self,
        frames: List[Any],
        target: Union[str, StandardAspectRatio, float],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Add side bars to widen aspect ratio.

        Args:
            frames: List of frames to process.
            target: Target aspect ratio (must be wider than source).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Frames with pillarbox added.
        """
        return self.convert_aspect(
            frames, target,
            method=ConversionMethod.PAD,
            progress_callback=progress_callback
        )

    def add_letterbox(
        self,
        frames: List[Any],
        target: Union[str, StandardAspectRatio, float],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Add top/bottom bars to narrow aspect ratio.

        Args:
            frames: List of frames to process.
            target: Target aspect ratio (must be narrower than source).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Frames with letterbox added.
        """
        return self.convert_aspect(
            frames, target,
            method=ConversionMethod.PAD,
            progress_callback=progress_callback
        )

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        target: Optional[str] = None,
        crop_bars: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Process entire video file.

        Args:
            input_path: Input video path.
            output_path: Output video path.
            target: Target aspect ratio (None to just crop bars).
            crop_bars: Whether to crop existing black bars.
            progress_callback: Progress callback.

        Returns:
            Path to processed video.
        """
        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg not available")

        filters = []

        # Detect and crop existing bars
        if crop_bars:
            crop_filter = self._detect_ffmpeg_crop(input_path)
            if crop_filter:
                filters.append(crop_filter)

        # Convert to target ratio
        if target:
            std = StandardAspectRatio.from_string(target)
            target_ratio = std.decimal if std else 16/9

            if self.config.method == ConversionMethod.CROP:
                filters.append(f"crop=ih*{target_ratio}:ih")
            elif self.config.method == ConversionMethod.PAD:
                filters.append(f"pad=ih*{target_ratio}:ih:(ow-iw)/2:0:black")
            elif self.config.method == ConversionMethod.SCALE:
                filters.append(f"scale=ih*{target_ratio}:ih")
            elif self.config.method == ConversionMethod.FIT:
                filters.append(f"scale='min(iw,ih*{target_ratio})':'min(ih,iw/{target_ratio})'")
                filters.append(f"pad=ih*{target_ratio}:ih:(ow-iw)/2:(oh-ih)/2:black")

        # Ensure square pixels
        filters.append("setsar=1:1")

        if not filters:
            filters = ["null"]

        filter_chain = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=7200)
            logger.info(f"Aspect processing complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Aspect processing failed: {e}")
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")

    def _detect_horizontal_bars(self, gray: Any) -> Tuple[int, int]:
        """Detect top and bottom black bars."""
        height = gray.shape[0]
        threshold = self.config.black_threshold

        # Detect top bar
        top_bar = 0
        for y in range(height // 3):
            row_mean = np.mean(gray[y, :])
            if row_mean > threshold:
                top_bar = y
                break

        # Detect bottom bar
        bottom_bar = 0
        for y in range(height - 1, 2 * height // 3, -1):
            row_mean = np.mean(gray[y, :])
            if row_mean > threshold:
                bottom_bar = height - 1 - y
                break

        return top_bar, bottom_bar

    def _detect_vertical_bars(self, gray: Any) -> Tuple[int, int]:
        """Detect left and right black bars."""
        width = gray.shape[1]
        threshold = self.config.black_threshold

        # Detect left bar
        left_bar = 0
        for x in range(width // 3):
            col_mean = np.mean(gray[:, x])
            if col_mean > threshold:
                left_bar = x
                break

        # Detect right bar
        right_bar = 0
        for x in range(width - 1, 2 * width // 3, -1):
            col_mean = np.mean(gray[:, x])
            if col_mean > threshold:
                right_bar = width - 1 - x
                break

        return left_bar, right_bar

    def _convert_frame(
        self,
        frame: Any,
        target_ratio: float,
        method: ConversionMethod,
    ) -> Any:
        """Convert a single frame to target aspect ratio."""
        height, width = frame.shape[:2]
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.01:
            return frame  # Already correct

        if method == ConversionMethod.CROP:
            return self._crop_to_ratio(frame, target_ratio)
        elif method == ConversionMethod.PAD:
            return self._pad_to_ratio(frame, target_ratio)
        elif method == ConversionMethod.SCALE:
            return self._scale_to_ratio(frame, target_ratio)
        elif method == ConversionMethod.FIT:
            return self._fit_to_ratio(frame, target_ratio)
        elif method == ConversionMethod.STRETCH:
            return self._stretch_to_ratio(frame, target_ratio)
        else:
            return frame

    def _crop_to_ratio(self, frame: Any, target_ratio: float) -> Any:
        """Crop frame to target aspect ratio."""
        height, width = frame.shape[:2]
        current_ratio = width / height

        if current_ratio > target_ratio:
            # Too wide, crop width
            new_width = int(height * target_ratio)
            x_offset = (width - new_width) // 2
            if len(frame.shape) == 3:
                return frame[:, x_offset:x_offset + new_width, :]
            else:
                return frame[:, x_offset:x_offset + new_width]
        else:
            # Too tall, crop height
            new_height = int(width / target_ratio)
            y_offset = (height - new_height) // 2
            if len(frame.shape) == 3:
                return frame[y_offset:y_offset + new_height, :, :]
            else:
                return frame[y_offset:y_offset + new_height, :]

    def _pad_to_ratio(self, frame: Any, target_ratio: float) -> Any:
        """Pad frame to target aspect ratio."""
        height, width = frame.shape[:2]
        current_ratio = width / height

        # Determine fill color
        if self.config.fill_mode == FillMode.BLACK:
            fill_color = (0, 0, 0)
        elif self.config.fill_mode == FillMode.COLOR:
            fill_color = self.config.fill_color
        else:
            fill_color = (0, 0, 0)

        if current_ratio < target_ratio:
            # Too narrow, add pillarbox
            new_width = int(height * target_ratio)
            pad_left = (new_width - width) // 2
            pad_right = new_width - width - pad_left

            if len(frame.shape) == 3:
                result = np.full((height, new_width, 3), fill_color, dtype=np.uint8)
                result[:, pad_left:pad_left + width, :] = frame
            else:
                result = np.full((height, new_width), fill_color[0], dtype=np.uint8)
                result[:, pad_left:pad_left + width] = frame

            # Apply blur fill if configured
            if self.config.fill_mode == FillMode.BLUR:
                result = self._apply_blur_fill(result, frame, pad_left, 0, width, height)
            elif self.config.fill_mode == FillMode.MIRROR:
                result = self._apply_mirror_fill(result, frame, pad_left, 0, width, height)

        else:
            # Too wide, add letterbox
            new_height = int(width / target_ratio)
            pad_top = (new_height - height) // 2
            pad_bottom = new_height - height - pad_top

            if len(frame.shape) == 3:
                result = np.full((new_height, width, 3), fill_color, dtype=np.uint8)
                result[pad_top:pad_top + height, :, :] = frame
            else:
                result = np.full((new_height, width), fill_color[0], dtype=np.uint8)
                result[pad_top:pad_top + height, :] = frame

            # Apply blur fill if configured
            if self.config.fill_mode == FillMode.BLUR:
                result = self._apply_blur_fill(result, frame, 0, pad_top, width, height)
            elif self.config.fill_mode == FillMode.MIRROR:
                result = self._apply_mirror_fill(result, frame, 0, pad_top, width, height)

        return result

    def _scale_to_ratio(self, frame: Any, target_ratio: float) -> Any:
        """Scale frame to target aspect ratio (may distort)."""
        height, width = frame.shape[:2]
        new_width = int(height * target_ratio)
        return cv2.resize(frame, (new_width, height), interpolation=cv2.INTER_LANCZOS4)

    def _fit_to_ratio(self, frame: Any, target_ratio: float) -> Any:
        """Scale to fit then pad to target ratio."""
        height, width = frame.shape[:2]
        current_ratio = width / height

        if current_ratio > target_ratio:
            # Scale by height to fit width
            new_width = int(height * target_ratio)
            scaled = cv2.resize(frame, (new_width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Scale by width to fit height
            new_height = int(width / target_ratio)
            scaled = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return self._pad_to_ratio(scaled, target_ratio)

    def _stretch_to_ratio(self, frame: Any, target_ratio: float) -> Any:
        """Stretch frame to target ratio (distorts)."""
        height, width = frame.shape[:2]
        new_width = int(height * target_ratio)
        return cv2.resize(frame, (new_width, height), interpolation=cv2.INTER_LANCZOS4)

    def _apply_blur_fill(
        self,
        result: Any,
        source: Any,
        x_offset: int,
        y_offset: int,
        src_width: int,
        src_height: int,
    ) -> Any:
        """Apply blurred edge content to fill areas."""
        # Create heavily blurred version of source
        blurred = cv2.GaussianBlur(source, (99, 99), 0)
        blurred = cv2.resize(blurred, (result.shape[1], result.shape[0]))

        # Apply blurred background
        mask = np.zeros(result.shape[:2], dtype=np.uint8)
        mask[y_offset:y_offset + src_height, x_offset:x_offset + src_width] = 255

        if len(result.shape) == 3:
            result[mask == 0] = blurred[mask == 0]
        else:
            result[mask == 0] = blurred[mask == 0]

        return result

    def _apply_mirror_fill(
        self,
        result: Any,
        source: Any,
        x_offset: int,
        y_offset: int,
        src_width: int,
        src_height: int,
    ) -> Any:
        """Apply mirrored edge content to fill areas."""
        res_height, res_width = result.shape[:2]

        # Left pillarbox
        if x_offset > 0:
            left_strip = source[:, :min(x_offset, src_width)]
            left_mirror = cv2.flip(left_strip, 1)
            left_mirror = cv2.resize(left_mirror, (x_offset, src_height))
            if len(result.shape) == 3:
                result[y_offset:y_offset + src_height, :x_offset, :] = left_mirror
            else:
                result[y_offset:y_offset + src_height, :x_offset] = left_mirror

        # Right pillarbox
        right_offset = x_offset + src_width
        if right_offset < res_width:
            right_width = res_width - right_offset
            right_strip = source[:, max(0, src_width - right_width):]
            right_mirror = cv2.flip(right_strip, 1)
            right_mirror = cv2.resize(right_mirror, (right_width, src_height))
            if len(result.shape) == 3:
                result[y_offset:y_offset + src_height, right_offset:, :] = right_mirror
            else:
                result[y_offset:y_offset + src_height, right_offset:] = right_mirror

        # Top letterbox
        if y_offset > 0:
            top_strip = source[:min(y_offset, src_height), :]
            top_mirror = cv2.flip(top_strip, 0)
            top_mirror = cv2.resize(top_mirror, (src_width, y_offset))
            if len(result.shape) == 3:
                result[:y_offset, x_offset:x_offset + src_width, :] = top_mirror
            else:
                result[:y_offset, x_offset:x_offset + src_width] = top_mirror

        # Bottom letterbox
        bottom_offset = y_offset + src_height
        if bottom_offset < res_height:
            bottom_height = res_height - bottom_offset
            bottom_strip = source[max(0, src_height - bottom_height):, :]
            bottom_mirror = cv2.flip(bottom_strip, 0)
            bottom_mirror = cv2.resize(bottom_mirror, (src_width, bottom_height))
            if len(result.shape) == 3:
                result[bottom_offset:, x_offset:x_offset + src_width, :] = bottom_mirror
            else:
                result[bottom_offset:, x_offset:x_offset + src_width] = bottom_mirror

        return result

    def _detect_ffmpeg_crop(self, video_path: Path) -> Optional[str]:
        """Use FFmpeg to detect crop region."""
        try:
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", "cropdetect=24:16:0",
                "-frames:v", "100",
                "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            # Parse cropdetect output
            import re
            crop_pattern = r"crop=(\d+):(\d+):(\d+):(\d+)"
            matches = re.findall(crop_pattern, result.stderr)

            if matches:
                # Use most common crop value
                from collections import Counter
                counter = Counter(matches)
                most_common = counter.most_common(1)[0][0]
                w, h, x, y = most_common
                return f"crop={w}:{h}:{x}:{y}"

        except Exception as e:
            logger.debug(f"FFmpeg crop detection failed: {e}")

        return None


# Common aspect ratio constants
RATIO_4_3 = StandardAspectRatio.RATIO_4_3
RATIO_16_9 = StandardAspectRatio.RATIO_16_9
RATIO_2_35_1 = StandardAspectRatio.RATIO_2_35_1
RATIO_1_85_1 = StandardAspectRatio.RATIO_1_85_1


def create_aspect_handler(
    target_ratio: Optional[str] = None,
    method: str = "fit",
    fill_mode: str = "black",
) -> AspectHandler:
    """Factory function to create an AspectHandler.

    Args:
        target_ratio: Target aspect ratio string.
        method: Conversion method name.
        fill_mode: Fill mode for padding.

    Returns:
        Configured AspectHandler.
    """
    try:
        conv_method = ConversionMethod(method.lower())
    except ValueError:
        conv_method = ConversionMethod.FIT

    try:
        fill = FillMode(fill_mode.lower())
    except ValueError:
        fill = FillMode.BLACK

    config = AspectConfig(
        target_ratio=target_ratio,
        method=conv_method,
        fill_mode=fill,
    )

    return AspectHandler(config)
