"""Burnt-in Subtitle Extraction and Removal.

Extracts hardcoded/burnt-in subtitles from video using OCR, saves them as
separate subtitle files (SRT/VTT), then removes them from the video frames
using inpainting. This enables:
- Translation to other languages using AI tools
- Clean video without subtitles
- Preservation of original subtitle content

Supported OCR backends:
- Tesseract (default, open source)
- EasyOCR (better for non-Latin scripts)
- PaddleOCR (best for Asian languages)

Example:
    >>> extractor = SubtitleExtractor(ocr_backend="tesseract")
    >>> result = extractor.extract_and_remove(
    ...     video_path="movie.mp4",
    ...     output_video="movie_clean.mp4",
    ...     output_srt="movie.srt"
    ... )
    >>> print(f"Extracted {result.subtitle_count} subtitles")
"""

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    confidence: float = 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    language: Optional[str] = None


@dataclass
class SubtitleExtractionResult:
    """Result of subtitle extraction."""
    success: bool = False
    subtitle_count: int = 0
    subtitles: List[SubtitleEntry] = field(default_factory=list)
    srt_path: Optional[Path] = None
    vtt_path: Optional[Path] = None
    clean_video_path: Optional[Path] = None
    detected_language: Optional[str] = None
    processing_time_seconds: float = 0.0
    frames_with_subtitles: int = 0
    total_frames_analyzed: int = 0


@dataclass
class SubtitleRegion:
    """Detected subtitle region in frame."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str
    confidence: float
    frame_number: int


class SubtitleDetector:
    """Detects subtitle regions in video frames.

    Uses edge detection and text area analysis to find
    subtitle regions, typically at the bottom of the frame.
    """

    def __init__(
        self,
        search_region: str = "bottom",  # bottom, top, full
        region_height_ratio: float = 0.25,  # Search bottom 25%
        min_text_confidence: float = 0.5,
    ):
        """Initialize subtitle detector.

        Args:
            search_region: Where to search for subtitles
            region_height_ratio: Portion of frame to search
            min_text_confidence: Minimum OCR confidence
        """
        self.search_region = search_region
        self.region_height_ratio = region_height_ratio
        self.min_text_confidence = min_text_confidence

    def get_search_region(
        self,
        frame_height: int,
        frame_width: int
    ) -> Tuple[int, int, int, int]:
        """Get the region to search for subtitles.

        Returns:
            (y_start, y_end, x_start, x_end)
        """
        region_height = int(frame_height * self.region_height_ratio)

        if self.search_region == "bottom":
            return (frame_height - region_height, frame_height, 0, frame_width)
        elif self.search_region == "top":
            return (0, region_height, 0, frame_width)
        else:  # full
            return (0, frame_height, 0, frame_width)

    def detect_text_regions(
        self,
        frame: "np.ndarray",
    ) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions in frame.

        Args:
            frame: BGR frame

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if not HAS_OPENCV:
            return []

        h, w = frame.shape[:2]
        y_start, y_end, x_start, x_end = self.get_search_region(h, w)

        # Crop to search region
        roi = frame[y_start:y_end, x_start:x_end]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply threshold to find text (white text on dark background)
        # Try multiple methods
        regions = []

        # Method 1: Simple threshold for white text
        _, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        regions.extend(self._find_contour_boxes(thresh1, y_start))

        # Method 2: Adaptive threshold for outlined text
        thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        regions.extend(self._find_contour_boxes(thresh2, y_start))

        # Method 3: Edge detection for any text
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=2)
        regions.extend(self._find_contour_boxes(dilated, y_start))

        # Merge overlapping regions
        return self._merge_boxes(regions)

    def _find_contour_boxes(
        self,
        binary: "np.ndarray",
        y_offset: int = 0
    ) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes from binary image."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter by size - subtitles are typically wide and short
            if w > 50 and h > 10 and w > h:
                boxes.append((x, y + y_offset, w, h))

        return boxes

    def _merge_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        overlap_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []

        # Sort by y coordinate
        boxes = sorted(boxes, key=lambda b: b[1])

        merged = [boxes[0]]
        for box in boxes[1:]:
            last = merged[-1]

            # Check if boxes overlap significantly
            x1, y1, w1, h1 = last
            x2, y2, w2, h2 = box

            # Calculate overlap
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            box_area = min(w1 * h1, w2 * h2)

            if box_area > 0 and overlap_area / box_area > overlap_threshold:
                # Merge boxes
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged.append(box)

        return merged


class OCREngine:
    """OCR engine abstraction for multiple backends."""

    def __init__(
        self,
        backend: str = "tesseract",
        languages: List[str] = None,
        gpu: bool = True,
    ):
        """Initialize OCR engine.

        Args:
            backend: OCR backend ("tesseract", "easyocr", "paddleocr")
            languages: Languages to detect
            gpu: Use GPU acceleration if available
        """
        self.backend = backend
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the OCR engine."""
        if self.backend == "easyocr" and HAS_EASYOCR:
            self._engine = easyocr.Reader(
                self.languages,
                gpu=self.gpu
            )
        elif self.backend == "tesseract" and HAS_TESSERACT:
            # Tesseract doesn't need initialization
            self._engine = "tesseract"
        else:
            # Fallback to tesseract
            if HAS_TESSERACT:
                self._engine = "tesseract"
                self.backend = "tesseract"
            elif HAS_EASYOCR:
                self._engine = easyocr.Reader(["en"], gpu=self.gpu)
                self.backend = "easyocr"
            else:
                logger.warning("No OCR backend available")
                self._engine = None

    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._engine is not None

    def extract_text(
        self,
        image: "np.ndarray",
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[str, float]:
        """Extract text from image.

        Args:
            image: Image as numpy array
            region: Optional region (x, y, w, h) to extract from

        Returns:
            Tuple of (text, confidence)
        """
        if not self.is_available():
            return "", 0.0

        # Crop to region if specified
        if region:
            x, y, w, h = region
            image = image[y:y+h, x:x+w]

        if self.backend == "tesseract":
            return self._tesseract_ocr(image)
        elif self.backend == "easyocr":
            return self._easyocr_ocr(image)
        else:
            return "", 0.0

    def _tesseract_ocr(self, image: "np.ndarray") -> Tuple[str, float]:
        """Run Tesseract OCR."""
        try:
            # Preprocess for better OCR
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Upscale if too small
            if gray.shape[0] < 30:
                scale = 30 / gray.shape[0]
                gray = cv2.resize(gray, None, fx=scale, fy=scale)

            # Get detailed output
            data = pytesseract.image_to_data(
                gray,
                lang="+".join(self.languages),
                output_type=pytesseract.Output.DICT
            )

            # Combine text and calculate confidence
            texts = []
            confidences = []
            for i, text in enumerate(data['text']):
                conf = int(data['conf'][i])
                if conf > 0 and text.strip():
                    texts.append(text)
                    confidences.append(conf / 100.0)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return combined_text, avg_confidence

        except Exception as e:
            logger.debug(f"Tesseract OCR failed: {e}")
            return "", 0.0

    def _easyocr_ocr(self, image: "np.ndarray") -> Tuple[str, float]:
        """Run EasyOCR."""
        try:
            results = self._engine.readtext(image)

            texts = []
            confidences = []
            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return combined_text, avg_confidence

        except Exception as e:
            logger.debug(f"EasyOCR failed: {e}")
            return "", 0.0


class SubtitleInpainter:
    """Remove subtitles from frames using inpainting."""

    def __init__(
        self,
        method: str = "telea",  # telea, ns, or ai
        expand_mask: int = 5,
    ):
        """Initialize inpainter.

        Args:
            method: Inpainting method
            expand_mask: Pixels to expand mask by
        """
        self.method = method
        self.expand_mask = expand_mask

    def create_mask(
        self,
        frame: "np.ndarray",
        text_regions: List[Tuple[int, int, int, int]],
    ) -> "np.ndarray":
        """Create mask for text regions.

        Args:
            frame: Original frame
            text_regions: List of text region bboxes

        Returns:
            Binary mask (white = inpaint area)
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for x, y, w, h in text_regions:
            # Expand region slightly
            x = max(0, x - self.expand_mask)
            y = max(0, y - self.expand_mask)
            w = w + 2 * self.expand_mask
            h = h + 2 * self.expand_mask

            mask[y:y+h, x:x+w] = 255

        return mask

    def inpaint(
        self,
        frame: "np.ndarray",
        mask: "np.ndarray",
    ) -> "np.ndarray":
        """Inpaint masked regions.

        Args:
            frame: Original frame
            mask: Binary mask

        Returns:
            Inpainted frame
        """
        if not HAS_OPENCV:
            return frame

        if self.method == "telea":
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        elif self.method == "ns":
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
        else:
            # Default to Telea
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


class SubtitleExtractor:
    """Main class for extracting and removing burnt-in subtitles.

    Workflow:
    1. Detect frames with subtitles
    2. Extract text using OCR
    3. Group into subtitle entries with timing
    4. Remove subtitles from video using inpainting
    5. Export subtitle file (SRT/VTT)
    """

    def __init__(
        self,
        ocr_backend: str = "tesseract",
        languages: List[str] = None,
        search_region: str = "bottom",
        min_confidence: float = 0.5,
        sample_rate: int = 2,  # Check every N frames
        gpu: bool = True,
    ):
        """Initialize subtitle extractor.

        Args:
            ocr_backend: OCR backend to use
            languages: Languages to detect
            search_region: Where to search for subtitles
            min_confidence: Minimum OCR confidence
            sample_rate: Frame sampling rate
            gpu: Use GPU if available
        """
        self.languages = languages or ["en"]
        self.min_confidence = min_confidence
        self.sample_rate = sample_rate

        self.detector = SubtitleDetector(
            search_region=search_region,
            min_text_confidence=min_confidence,
        )
        self.ocr = OCREngine(
            backend=ocr_backend,
            languages=self.languages,
            gpu=gpu,
        )
        self.inpainter = SubtitleInpainter()

    def is_available(self) -> bool:
        """Check if subtitle extraction is available."""
        return HAS_OPENCV and self.ocr.is_available()

    def extract_and_remove(
        self,
        video_path: Path,
        output_video: Optional[Path] = None,
        output_srt: Optional[Path] = None,
        output_vtt: Optional[Path] = None,
        remove_subtitles: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SubtitleExtractionResult:
        """Extract subtitles and optionally remove them from video.

        Args:
            video_path: Input video path
            output_video: Output video path (without subtitles)
            output_srt: Output SRT file path
            output_vtt: Output VTT file path
            remove_subtitles: Whether to remove subtitles from video
            progress_callback: Progress callback (0-1)

        Returns:
            SubtitleExtractionResult
        """
        import time
        start_time = time.time()

        result = SubtitleExtractionResult()

        if not self.is_available():
            logger.error("Subtitle extraction not available (missing OpenCV or OCR)")
            return result

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return result

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result.total_frames_analyzed = total_frames // self.sample_rate

        logger.info(f"Extracting subtitles from {video_path.name} ({total_frames} frames)")

        # Phase 1: Extract subtitles
        subtitle_regions = []
        current_text = ""
        current_start = 0.0
        subtitle_index = 1

        for frame_num in range(0, total_frames, self.sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect text regions
            regions = self.detector.detect_text_regions(frame)

            if regions:
                # Get largest region (likely main subtitle)
                largest = max(regions, key=lambda r: r[2] * r[3])

                # OCR the region
                text, confidence = self.ocr.extract_text(frame, largest)
                text = self._clean_text(text)

                if text and confidence >= self.min_confidence:
                    timestamp = frame_num / fps

                    if text != current_text:
                        # End previous subtitle
                        if current_text:
                            result.subtitles.append(SubtitleEntry(
                                index=subtitle_index,
                                start_time=current_start,
                                end_time=timestamp,
                                text=current_text,
                                confidence=confidence,
                            ))
                            subtitle_index += 1

                        # Start new subtitle
                        current_text = text
                        current_start = timestamp

                    result.frames_with_subtitles += 1
                    subtitle_regions.append((frame_num, largest))

            if progress_callback:
                progress_callback(0.5 * frame_num / total_frames)

        # Add final subtitle
        if current_text:
            result.subtitles.append(SubtitleEntry(
                index=subtitle_index,
                start_time=current_start,
                end_time=total_frames / fps,
                text=current_text,
            ))

        result.subtitle_count = len(result.subtitles)
        logger.info(f"Extracted {result.subtitle_count} subtitles")

        # Phase 2: Export subtitle files
        if output_srt:
            output_srt = Path(output_srt)
            self._export_srt(result.subtitles, output_srt)
            result.srt_path = output_srt

        if output_vtt:
            output_vtt = Path(output_vtt)
            self._export_vtt(result.subtitles, output_vtt)
            result.vtt_path = output_vtt

        # Phase 3: Remove subtitles from video
        if remove_subtitles and output_video and subtitle_regions:
            output_video = Path(output_video)
            self._remove_subtitles_from_video(
                video_path,
                output_video,
                subtitle_regions,
                fps,
                width,
                height,
                total_frames,
                lambda p: progress_callback(0.5 + 0.5 * p) if progress_callback else None
            )
            result.clean_video_path = output_video

        cap.release()

        result.success = True
        result.processing_time_seconds = time.time() - start_time

        return result

    def _clean_text(self, text: str) -> str:
        """Clean OCR text."""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove common OCR errors
        text = text.replace("|", "I")
        text = text.replace("0", "O") if text.isupper() else text
        return text.strip()

    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for VTT."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _export_srt(self, subtitles: List[SubtitleEntry], output_path: Path):
        """Export subtitles as SRT file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(f"{sub.index}\n")
                f.write(f"{self._format_timestamp_srt(sub.start_time)} --> ")
                f.write(f"{self._format_timestamp_srt(sub.end_time)}\n")
                f.write(f"{sub.text}\n\n")

        logger.info(f"Exported SRT: {output_path}")

    def _export_vtt(self, subtitles: List[SubtitleEntry], output_path: Path):
        """Export subtitles as WebVTT file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for sub in subtitles:
                f.write(f"{sub.index}\n")
                f.write(f"{self._format_timestamp_vtt(sub.start_time)} --> ")
                f.write(f"{self._format_timestamp_vtt(sub.end_time)}\n")
                f.write(f"{sub.text}\n\n")

        logger.info(f"Exported VTT: {output_path}")

    def _remove_subtitles_from_video(
        self,
        input_path: Path,
        output_path: Path,
        subtitle_regions: List[Tuple[int, Tuple[int, int, int, int]]],
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        """Remove subtitles from video using inpainting."""
        # Create temporary frame directory
        with tempfile.TemporaryDirectory(prefix="framewright_sub_") as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / "frames"
            clean_dir = temp_path / "clean"
            frames_dir.mkdir()
            clean_dir.mkdir()

            # Extract all frames
            logger.info("Extracting frames for subtitle removal...")
            subprocess.run([
                'ffmpeg', '-y', '-i', str(input_path),
                '-q:v', '1',
                str(frames_dir / 'frame_%08d.png')
            ], capture_output=True, check=True)

            # Build frame->region mapping
            region_map = {}
            for frame_num, region in subtitle_regions:
                # Apply to nearby frames too (subtitles persist)
                for offset in range(-self.sample_rate, self.sample_rate + 1):
                    region_map[frame_num + offset] = region

            # Process frames
            frame_files = sorted(frames_dir.glob("*.png"))
            for i, frame_file in enumerate(frame_files):
                frame_num = i + 1  # 1-indexed from ffmpeg

                if frame_num in region_map:
                    # Inpaint this frame
                    frame = cv2.imread(str(frame_file))
                    region = region_map[frame_num]
                    mask = self.inpainter.create_mask(frame, [region])
                    clean_frame = self.inpainter.inpaint(frame, mask)
                    cv2.imwrite(str(clean_dir / frame_file.name), clean_frame)
                else:
                    # Copy unchanged
                    import shutil
                    shutil.copy(frame_file, clean_dir / frame_file.name)

                if progress_callback and i % 100 == 0:
                    progress_callback(i / len(frame_files))

            # Reassemble video
            logger.info("Reassembling video without subtitles...")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get audio from original
            subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(clean_dir / 'frame_%08d.png'),
                '-i', str(input_path),
                '-map', '0:v', '-map', '1:a?',
                '-c:v', 'libx264', '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ], capture_output=True, check=True)

            logger.info(f"Created clean video: {output_path}")


def extract_subtitles(
    video_path: Path,
    output_srt: Optional[Path] = None,
    output_video: Optional[Path] = None,
    languages: List[str] = None,
    remove_from_video: bool = True,
) -> SubtitleExtractionResult:
    """Convenience function to extract subtitles.

    Args:
        video_path: Input video
        output_srt: Output SRT path (auto-generated if None)
        output_video: Output video path (auto-generated if None)
        languages: Languages to detect
        remove_from_video: Remove subtitles from video

    Returns:
        SubtitleExtractionResult
    """
    video_path = Path(video_path)

    if output_srt is None:
        output_srt = video_path.with_suffix('.srt')

    if output_video is None and remove_from_video:
        output_video = video_path.parent / f"{video_path.stem}_no_subs.mp4"

    extractor = SubtitleExtractor(languages=languages)
    return extractor.extract_and_remove(
        video_path,
        output_video=output_video,
        output_srt=output_srt,
        remove_subtitles=remove_from_video,
    )
