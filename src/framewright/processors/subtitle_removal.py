"""Burnt-in subtitle detection and removal for video restoration.

This module provides functionality to detect and remove hard-coded (burnt-in)
subtitles from video frames. Unlike embedded subtitle streams which can be
toggled, burnt-in subtitles are permanently rendered into the video and
require OCR detection + inpainting to remove.

Features:
- Multi-engine OCR support (EasyOCR, Tesseract, PaddleOCR)
- Configurable subtitle region detection (bottom-third default)
- LaMA/OpenCV inpainting for seamless removal
- Batch processing with temporal consistency
- Auto-detection of subtitle presence
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Available OCR engines for subtitle detection."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    AUTO = "auto"  # Auto-select best available


class SubtitleRegion(Enum):
    """Predefined subtitle region presets."""
    BOTTOM_THIRD = "bottom_third"      # Bottom 33% of frame
    BOTTOM_QUARTER = "bottom_quarter"  # Bottom 25% of frame
    TOP_QUARTER = "top_quarter"        # Top 25% (for Chinese subs)
    FULL_FRAME = "full_frame"          # Scan entire frame
    CUSTOM = "custom"                   # User-defined region


@dataclass
class SubtitleBox:
    """Detected subtitle text box with coordinates.

    Attributes:
        x: Left coordinate
        y: Top coordinate
        width: Box width
        height: Box height
        text: Detected text content
        confidence: OCR confidence score (0-1)
        language: Detected language code
    """
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    confidence: float = 1.0
    language: str = "unknown"

    @property
    def x2(self) -> int:
        """Right coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom coordinate."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x, self.y, self.x2, self.y2)

    def expand(self, pixels: int) -> 'SubtitleBox':
        """Expand box by given pixels in all directions."""
        return SubtitleBox(
            x=max(0, self.x - pixels),
            y=max(0, self.y - pixels),
            width=self.width + 2 * pixels,
            height=self.height + 2 * pixels,
            text=self.text,
            confidence=self.confidence,
            language=self.language
        )


@dataclass
class SubtitleRemovalConfig:
    """Configuration for burnt-in subtitle removal.

    Attributes:
        ocr_engine: OCR engine to use for detection
        region: Subtitle region preset or custom
        custom_region: Custom region as (x, y, width, height) ratios (0-1)
        min_confidence: Minimum OCR confidence to consider as subtitle
        text_expansion: Pixels to expand around detected text
        min_text_height: Minimum text height in pixels
        max_text_height: Maximum text height in pixels
        inpainting_method: Method for inpainting ('lama', 'opencv', 'telea')
        languages: List of language codes to detect
        skip_frames_without_text: Skip processing frames with no detected text
        temporal_smoothing: Apply temporal smoothing for consistent masks
    """
    ocr_engine: OCREngine = OCREngine.AUTO
    region: SubtitleRegion = SubtitleRegion.BOTTOM_THIRD
    custom_region: Optional[Tuple[float, float, float, float]] = None
    min_confidence: float = 0.5
    text_expansion: int = 5
    min_text_height: int = 10
    max_text_height: int = 100
    inpainting_method: str = "lama"
    languages: List[str] = field(default_factory=lambda: ["en"])
    skip_frames_without_text: bool = True
    temporal_smoothing: bool = True


@dataclass
class SubtitleRemovalResult:
    """Result of subtitle removal processing."""
    frames_processed: int = 0
    frames_with_subtitles: int = 0
    frames_cleaned: int = 0
    frames_skipped: int = 0
    failed_frames: int = 0
    output_dir: Optional[Path] = None
    detected_texts: List[str] = field(default_factory=list)


class SubtitleDetector:
    """Detect burnt-in subtitles using OCR.

    Supports multiple OCR backends:
    - EasyOCR: Good accuracy, GPU-accelerated, multi-language
    - Tesseract: Fast, widely available, good for Latin scripts
    - PaddleOCR: Best for Asian languages (Chinese, Japanese, Korean)
    """

    def __init__(self, config: Optional[SubtitleRemovalConfig] = None):
        """Initialize subtitle detector.

        Args:
            config: SubtitleRemovalConfig with detection settings
        """
        self.config = config or SubtitleRemovalConfig()
        self._ocr_engine = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available OCR backend."""
        if self.config.ocr_engine == OCREngine.AUTO:
            # Try in order of preference
            for engine in [OCREngine.EASYOCR, OCREngine.PADDLEOCR, OCREngine.TESSERACT]:
                backend = self._check_engine(engine)
                if backend:
                    logger.info(f"Auto-selected OCR engine: {engine.value}")
                    return backend
            return None
        else:
            return self._check_engine(self.config.ocr_engine)

    def _check_engine(self, engine: OCREngine) -> Optional[str]:
        """Check if specific OCR engine is available."""
        if engine == OCREngine.EASYOCR:
            try:
                import easyocr
                return 'easyocr'
            except ImportError:
                pass

        elif engine == OCREngine.PADDLEOCR:
            try:
                from paddleocr import PaddleOCR
                return 'paddleocr'
            except ImportError:
                pass

        elif engine == OCREngine.TESSERACT:
            try:
                import pytesseract
                # Check if tesseract binary is available
                pytesseract.get_tesseract_version()
                return 'tesseract'
            except (ImportError, Exception):
                pass

        return None

    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._backend is not None

    def get_region_bounds(
        self,
        frame_height: int,
        frame_width: int
    ) -> Tuple[int, int, int, int]:
        """Get subtitle region bounds based on config.

        Args:
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels

        Returns:
            Tuple of (x, y, width, height) for region
        """
        if self.config.region == SubtitleRegion.CUSTOM and self.config.custom_region:
            rx, ry, rw, rh = self.config.custom_region
            return (
                int(frame_width * rx),
                int(frame_height * ry),
                int(frame_width * rw),
                int(frame_height * rh)
            )

        region_map = {
            SubtitleRegion.BOTTOM_THIRD: (0, 0.67, 1.0, 0.33),
            SubtitleRegion.BOTTOM_QUARTER: (0, 0.75, 1.0, 0.25),
            SubtitleRegion.TOP_QUARTER: (0, 0, 1.0, 0.25),
            SubtitleRegion.FULL_FRAME: (0, 0, 1.0, 1.0),
        }

        rx, ry, rw, rh = region_map.get(
            self.config.region,
            region_map[SubtitleRegion.BOTTOM_THIRD]
        )

        return (
            int(frame_width * rx),
            int(frame_height * ry),
            int(frame_width * rw),
            int(frame_height * rh)
        )

    def detect_subtitles(
        self,
        frame: np.ndarray,
        region_only: bool = True
    ) -> List[SubtitleBox]:
        """Detect subtitle text boxes in a frame.

        Args:
            frame: Input frame as numpy array (BGR format)
            region_only: Only search in configured subtitle region

        Returns:
            List of SubtitleBox objects with detected text
        """
        if not self._backend:
            logger.warning("No OCR backend available")
            return []

        height, width = frame.shape[:2]

        # Extract subtitle region if configured
        if region_only and self.config.region != SubtitleRegion.FULL_FRAME:
            rx, ry, rw, rh = self.get_region_bounds(height, width)
            region_frame = frame[ry:ry+rh, rx:rx+rw]
            offset = (rx, ry)
        else:
            region_frame = frame
            offset = (0, 0)

        # Detect using appropriate backend
        if self._backend == 'easyocr':
            boxes = self._detect_easyocr(region_frame, offset)
        elif self._backend == 'paddleocr':
            boxes = self._detect_paddleocr(region_frame, offset)
        elif self._backend == 'tesseract':
            boxes = self._detect_tesseract(region_frame, offset)
        else:
            boxes = []

        # Filter by confidence and size
        filtered_boxes = []
        for box in boxes:
            if box.confidence < self.config.min_confidence:
                continue
            if box.height < self.config.min_text_height:
                continue
            if box.height > self.config.max_text_height:
                continue
            filtered_boxes.append(box)

        return filtered_boxes

    def _detect_easyocr(
        self,
        frame: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[SubtitleBox]:
        """Detect using EasyOCR."""
        try:
            import easyocr

            if self._ocr_engine is None:
                self._ocr_engine = easyocr.Reader(
                    self.config.languages,
                    gpu=True,
                    verbose=False
                )

            # EasyOCR expects RGB
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self._ocr_engine.readtext(rgb_frame)

            boxes = []
            for bbox, text, conf in results:
                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x1 = int(min(p[0] for p in bbox)) + offset[0]
                y1 = int(min(p[1] for p in bbox)) + offset[1]
                x2 = int(max(p[0] for p in bbox)) + offset[0]
                y2 = int(max(p[1] for p in bbox)) + offset[1]

                boxes.append(SubtitleBox(
                    x=x1, y=y1,
                    width=x2-x1, height=y2-y1,
                    text=text,
                    confidence=conf
                ))

            return boxes

        except Exception as e:
            logger.error(f"EasyOCR detection failed: {e}")
            return []

    def _detect_paddleocr(
        self,
        frame: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[SubtitleBox]:
        """Detect using PaddleOCR."""
        try:
            from paddleocr import PaddleOCR

            if self._ocr_engine is None:
                # Determine language for PaddleOCR
                lang = self.config.languages[0] if self.config.languages else 'en'
                paddle_lang_map = {
                    'en': 'en', 'zh': 'ch', 'ja': 'japan',
                    'ko': 'korean', 'fr': 'fr', 'de': 'german'
                }
                paddle_lang = paddle_lang_map.get(lang, 'en')

                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    show_log=False
                )

            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._ocr_engine.ocr(rgb_frame, cls=True)

            boxes = []
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, conf) = line
                    x1 = int(min(p[0] for p in bbox)) + offset[0]
                    y1 = int(min(p[1] for p in bbox)) + offset[1]
                    x2 = int(max(p[0] for p in bbox)) + offset[0]
                    y2 = int(max(p[1] for p in bbox)) + offset[1]

                    boxes.append(SubtitleBox(
                        x=x1, y=y1,
                        width=x2-x1, height=y2-y1,
                        text=text,
                        confidence=conf
                    ))

            return boxes

        except Exception as e:
            logger.error(f"PaddleOCR detection failed: {e}")
            return []

    def _detect_tesseract(
        self,
        frame: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[SubtitleBox]:
        """Detect using Tesseract."""
        try:
            import pytesseract
            import cv2

            # Preprocess for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply threshold to enhance text
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Get bounding boxes with text
            data = pytesseract.image_to_data(
                binary,
                output_type=pytesseract.Output.DICT,
                lang='+'.join(self.config.languages)
            )

            boxes = []
            n_boxes = len(data['level'])

            for i in range(n_boxes):
                conf = int(data['conf'][i])
                if conf < 0:  # -1 means no text
                    continue

                text = data['text'][i].strip()
                if not text:
                    continue

                x = data['left'][i] + offset[0]
                y = data['top'][i] + offset[1]
                w = data['width'][i]
                h = data['height'][i]

                boxes.append(SubtitleBox(
                    x=x, y=y,
                    width=w, height=h,
                    text=text,
                    confidence=conf / 100.0
                ))

            return boxes

        except Exception as e:
            logger.error(f"Tesseract detection failed: {e}")
            return []


class SubtitleRemover:
    """Remove burnt-in subtitles from video frames.

    Uses OCR detection to find subtitle text regions, then inpainting
    to seamlessly fill in the removed areas.
    """

    def __init__(
        self,
        config: Optional[SubtitleRemovalConfig] = None,
        model_dir: Optional[Path] = None
    ):
        """Initialize subtitle remover.

        Args:
            config: SubtitleRemovalConfig with removal settings
            model_dir: Directory for model weights
        """
        self.config = config or SubtitleRemovalConfig()
        self.model_dir = model_dir or Path.home() / '.framewright' / 'models'
        self.detector = SubtitleDetector(config)
        self._inpainter = None

    def is_available(self) -> bool:
        """Check if subtitle removal is available."""
        return self.detector.is_available()

    def create_text_mask(
        self,
        frame: np.ndarray,
        boxes: List[SubtitleBox]
    ) -> np.ndarray:
        """Create a binary mask for detected text regions.

        Args:
            frame: Original frame
            boxes: List of detected subtitle boxes

        Returns:
            Binary mask (255 = text region, 0 = background)
        """
        import cv2

        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for box in boxes:
            # Expand box slightly for better coverage
            expanded = box.expand(self.config.text_expansion)

            # Clip to frame bounds
            x1 = max(0, expanded.x)
            y1 = max(0, expanded.y)
            x2 = min(width, expanded.x2)
            y2 = min(height, expanded.y2)

            mask[y1:y2, x1:x2] = 255

        # Optional: dilate mask for smoother edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def remove_subtitles(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[SubtitleBox]]:
        """Remove subtitles from a single frame.

        Args:
            frame: Input frame (BGR format)
            mask: Optional pre-computed mask (will detect if None)

        Returns:
            Tuple of (cleaned frame, detected boxes)
        """
        import cv2

        # Detect subtitles if no mask provided
        boxes = []
        if mask is None:
            boxes = self.detector.detect_subtitles(frame)
            if not boxes:
                return frame, boxes
            mask = self.create_text_mask(frame, boxes)

        # Check if mask is empty
        if not np.any(mask):
            return frame, boxes

        # Inpaint using configured method
        if self.config.inpainting_method == 'lama':
            result = self._inpaint_lama(frame, mask)
        elif self.config.inpainting_method == 'opencv':
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
        elif self.config.inpainting_method == 'telea':
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        else:
            # Default to OpenCV NS
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

        return result, boxes

    def _inpaint_lama(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using LaMA model (if available)."""
        try:
            # Try to use LaMA from watermark_removal module
            from .watermark_removal import WatermarkRemover, WatermarkRemovalConfig

            # Create a temporary remover with the mask
            config = WatermarkRemovalConfig(
                inpainting_method='lama',
                auto_detect=False
            )
            remover = WatermarkRemover(config, model_dir=self.model_dir)

            if remover.is_available():
                return remover.inpaint_region(frame, mask)
            else:
                # Fall back to OpenCV
                import cv2
                return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

        except (ImportError, Exception) as e:
            logger.debug(f"LaMA inpainting not available: {e}")
            import cv2
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SubtitleRemovalResult:
        """Remove subtitles from all frames in a directory.

        Args:
            input_dir: Directory with input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback (0.0-1.0)

        Returns:
            SubtitleRemovalResult with processing statistics
        """
        result = SubtitleRemovalResult(output_dir=output_dir)

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all frames
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        if not self.is_available():
            logger.warning("Subtitle detection not available, copying frames")
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            result.frames_processed = len(frames)
            result.frames_skipped = len(frames)
            return result

        logger.info(f"Processing {len(frames)} frames for subtitle removal")

        try:
            import cv2

            # For temporal smoothing, track previous masks
            prev_mask = None

            for i, frame_path in enumerate(frames):
                try:
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        logger.warning(f"Failed to read: {frame_path}")
                        result.failed_frames += 1
                        continue

                    # Detect and remove
                    boxes = self.detector.detect_subtitles(frame)

                    if boxes:
                        mask = self.create_text_mask(frame, boxes)

                        # Temporal smoothing: combine with previous mask
                        if self.config.temporal_smoothing and prev_mask is not None:
                            if prev_mask.shape == mask.shape:
                                # OR with previous to catch transitions
                                mask = cv2.bitwise_or(mask, prev_mask)

                        prev_mask = mask.copy()

                        # Remove subtitles
                        cleaned, _ = self.remove_subtitles(frame, mask)
                        result.frames_with_subtitles += 1
                        result.frames_cleaned += 1

                        # Collect detected text
                        for box in boxes:
                            if box.text:
                                result.detected_texts.append(box.text)

                    else:
                        # No subtitles detected
                        cleaned = frame
                        if self.config.skip_frames_without_text:
                            result.frames_skipped += 1
                        prev_mask = None

                    # Save output
                    cv2.imwrite(str(output_dir / frame_path.name), cleaned)
                    result.frames_processed += 1

                except Exception as e:
                    logger.error(f"Failed to process {frame_path.name}: {e}")
                    shutil.copy(frame_path, output_dir / frame_path.name)
                    result.failed_frames += 1
                    result.frames_processed += 1

                if progress_callback:
                    progress_callback((i + 1) / len(frames))

        except ImportError:
            logger.error("OpenCV not available")
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            result.frames_processed = len(frames)
            result.failed_frames = len(frames)

        logger.info(
            f"Subtitle removal complete: {result.frames_cleaned} cleaned, "
            f"{result.frames_skipped} skipped, {result.failed_frames} failed"
        )

        return result


class AutoSubtitleRemover:
    """Automatic subtitle detection and removal.

    Analyzes video to determine if burnt-in subtitles are present
    and removes them if detected.
    """

    def __init__(
        self,
        remover: Optional[SubtitleRemover] = None,
        sample_rate: int = 30,
        detection_threshold: float = 0.3
    ):
        """Initialize auto subtitle remover.

        Args:
            remover: Optional custom SubtitleRemover instance
            sample_rate: Sample every Nth frame for analysis
            detection_threshold: Fraction of frames with subs to trigger removal
        """
        self.remover = remover or SubtitleRemover()
        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold

    def analyze_for_subtitles(
        self,
        frames_dir: Path
    ) -> Tuple[bool, float]:
        """Analyze frames to determine if subtitles are present.

        Args:
            frames_dir: Directory containing frames

        Returns:
            Tuple of (has_subtitles, detection_ratio)
        """
        frames = sorted(Path(frames_dir).glob("*.png"))
        if not frames:
            frames = sorted(Path(frames_dir).glob("*.jpg"))

        if not frames:
            return False, 0.0

        if not self.remover.is_available():
            return False, 0.0

        sample_frames = frames[::self.sample_rate][:50]
        detected_count = 0

        try:
            import cv2

            for frame_path in sample_frames:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue

                boxes = self.remover.detector.detect_subtitles(frame)
                if boxes:
                    detected_count += 1

        except ImportError:
            logger.warning("OpenCV not available for analysis")
            return False, 0.0

        detection_ratio = detected_count / len(sample_frames) if sample_frames else 0.0
        has_subtitles = detection_ratio >= self.detection_threshold

        logger.info(
            f"Subtitle analysis: {detection_ratio:.1%} of frames have subtitles, "
            f"removal {'recommended' if has_subtitles else 'not needed'}"
        )

        return has_subtitles, detection_ratio

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        force: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[SubtitleRemovalResult, bool]:
        """Automatically analyze and remove subtitles if detected.

        Args:
            input_dir: Input frames directory
            output_dir: Output frames directory
            force: Force removal even if not auto-detected
            progress_callback: Progress callback

        Returns:
            Tuple of (removal result, was_processed)
        """
        if progress_callback:
            progress_callback(0.05)

        # Analyze content
        has_subtitles, ratio = self.analyze_for_subtitles(input_dir)

        if progress_callback:
            progress_callback(0.1)

        if not has_subtitles and not force:
            logger.info("No burnt-in subtitles detected, copying frames")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            frames = sorted(Path(input_dir).glob("*.png"))
            if not frames:
                frames = sorted(Path(input_dir).glob("*.jpg"))

            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)

            result = SubtitleRemovalResult(
                frames_processed=len(frames),
                frames_skipped=len(frames),
                output_dir=output_dir
            )

            if progress_callback:
                progress_callback(1.0)

            return result, False

        # Remove subtitles
        def scaled_progress(p):
            if progress_callback:
                progress_callback(0.1 + p * 0.9)

        result = self.remover.process_directory(
            input_dir,
            output_dir,
            scaled_progress
        )

        return result, True


# Convenience functions

def detect_burnt_subtitles(
    frame: np.ndarray,
    config: Optional[SubtitleRemovalConfig] = None
) -> List[SubtitleBox]:
    """Detect burnt-in subtitles in a frame.

    Args:
        frame: Input frame (BGR format)
        config: Optional configuration

    Returns:
        List of detected SubtitleBox objects
    """
    detector = SubtitleDetector(config)
    return detector.detect_subtitles(frame)


def remove_burnt_subtitles(
    frame: np.ndarray,
    config: Optional[SubtitleRemovalConfig] = None
) -> np.ndarray:
    """Remove burnt-in subtitles from a frame.

    Args:
        frame: Input frame (BGR format)
        config: Optional configuration

    Returns:
        Cleaned frame
    """
    remover = SubtitleRemover(config)
    cleaned, _ = remover.remove_subtitles(frame)
    return cleaned


def check_ocr_available() -> Dict[str, bool]:
    """Check which OCR engines are available.

    Returns:
        Dictionary mapping engine names to availability
    """
    results = {}

    # EasyOCR
    try:
        import easyocr
        results['easyocr'] = True
    except ImportError:
        results['easyocr'] = False

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        results['paddleocr'] = True
    except ImportError:
        results['paddleocr'] = False

    # Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        results['tesseract'] = True
    except (ImportError, Exception):
        results['tesseract'] = False

    return results
