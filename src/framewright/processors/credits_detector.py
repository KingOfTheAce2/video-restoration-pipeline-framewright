"""
Credits/Intro Detection - Auto-detect credits, intros, and outros.

Identifies common video segments like opening credits, end credits,
and title sequences for special processing or skipping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import json

import cv2
import numpy as np


class SegmentType(Enum):
    """Types of detected segments."""
    INTRO = "intro"
    CREDITS_OPENING = "credits_opening"
    CREDITS_CLOSING = "credits_closing"
    TITLE_CARD = "title_card"
    BLACK_SEGMENT = "black_segment"
    STATIC_LOGO = "static_logo"
    CONTENT = "content"


@dataclass
class DetectedSegment:
    """A detected segment in the video."""
    segment_type: SegmentType
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    characteristics: Dict[str, any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class CreditsAnalysis:
    """Complete credits/intro analysis results."""
    video_path: str
    total_frames: int
    total_duration: float
    fps: float
    segments: List[DetectedSegment]
    intro_end_frame: Optional[int] = None
    credits_start_frame: Optional[int] = None
    main_content_start: Optional[int] = None
    main_content_end: Optional[int] = None

    def get_segments_by_type(self, segment_type: SegmentType) -> List[DetectedSegment]:
        """Get all segments of a specific type."""
        return [s for s in self.segments if s.segment_type == segment_type]

    def get_main_content_range(self) -> Tuple[int, int]:
        """Get frame range of main content (excluding intro/credits)."""
        start = self.main_content_start or 0
        end = self.main_content_end or self.total_frames
        return start, end

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "total_duration": self.total_duration,
            "fps": self.fps,
            "intro_end_frame": self.intro_end_frame,
            "credits_start_frame": self.credits_start_frame,
            "main_content_start": self.main_content_start,
            "main_content_end": self.main_content_end,
            "segments": [
                {
                    "type": s.segment_type.value,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": s.duration,
                    "confidence": s.confidence,
                    "characteristics": s.characteristics
                }
                for s in self.segments
            ]
        }

    def save(self, path: Path) -> None:
        """Save analysis to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class CreditsDetector:
    """
    Detect intro sequences, credits, and title cards in videos.

    Uses multiple heuristics including:
    - Text density analysis (credits have lots of text)
    - Motion analysis (credits often scroll)
    - Color/contrast patterns (black backgrounds common)
    - Scene stability (credits tend to be stable)
    - Position in video (intro at start, credits at end)
    """

    # Detection parameters
    MIN_INTRO_DURATION = 3.0  # seconds
    MAX_INTRO_DURATION = 120.0  # seconds
    MIN_CREDITS_DURATION = 10.0  # seconds
    BLACK_THRESHOLD = 15  # Mean brightness below this = black
    TEXT_DENSITY_THRESHOLD = 0.02  # Fraction of frame with text-like edges
    SCROLL_DETECTION_THRESHOLD = 5.0  # Vertical motion threshold

    def __init__(
        self,
        sample_interval: float = 0.5,  # seconds between samples
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize detector.

        Args:
            sample_interval: Time between analysis samples
            progress_callback: Called with (message, progress_0_to_1)
        """
        self.sample_interval = sample_interval
        self.progress_callback = progress_callback

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def _analyze_text_density(self, gray: np.ndarray) -> float:
        """
        Estimate text density in frame.

        Returns fraction of frame likely containing text.
        """
        # Edge detection tuned for text
        edges = cv2.Canny(gray, 50, 150)

        # Text has lots of horizontal edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Calculate density
        text_pixels = np.sum(horizontal > 0)
        total_pixels = gray.size

        return text_pixels / total_pixels

    def _detect_scrolling(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Detect vertical scrolling motion (common in credits).

        Returns (is_scrolling, vertical_motion_magnitude)
        """
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Get vertical component
        vertical_flow = flow[:, :, 1]

        # Scrolling = consistent vertical motion
        mean_vertical = np.mean(vertical_flow)
        std_vertical = np.std(vertical_flow)

        # Credits scroll has consistent direction, low variance
        is_scrolling = (
            abs(mean_vertical) > self.SCROLL_DETECTION_THRESHOLD and
            std_vertical < abs(mean_vertical) * 0.5
        )

        return is_scrolling, mean_vertical

    def _is_black_frame(self, gray: np.ndarray) -> bool:
        """Check if frame is mostly black."""
        return np.mean(gray) < self.BLACK_THRESHOLD

    def _is_static_frame(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        threshold: float = 2.0
    ) -> bool:
        """Check if frame is nearly identical to previous."""
        diff = cv2.absdiff(prev_gray, curr_gray)
        return np.mean(diff) < threshold

    def _detect_high_contrast_text(self, gray: np.ndarray) -> float:
        """
        Detect high-contrast text (white on black or vice versa).

        Returns confidence score 0-1.
        """
        # Check for bimodal histogram (text + background)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # Find peaks
        dark_region = np.sum(hist[:50])
        bright_region = np.sum(hist[200:])
        mid_region = np.sum(hist[50:200])
        total = np.sum(hist)

        # High contrast text has strong peaks at extremes
        if total > 0:
            dark_ratio = dark_region / total
            bright_ratio = bright_region / total
            mid_ratio = mid_region / total

            # Credits often have >60% in one extreme, <20% in middle
            if (dark_ratio > 0.6 or bright_ratio > 0.6) and mid_ratio < 0.3:
                return 0.8
            elif (dark_ratio > 0.4 or bright_ratio > 0.4) and mid_ratio < 0.4:
                return 0.5

        return 0.0

    def _analyze_frame_batch(
        self,
        frames: List[np.ndarray],
        start_frame: int,
        fps: float
    ) -> Dict[str, any]:
        """Analyze a batch of frames for credits characteristics."""
        if not frames:
            return {}

        text_densities = []
        black_frames = 0
        static_count = 0
        scroll_detected = 0
        contrast_scores = []

        prev_gray = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Text density
            text_densities.append(self._analyze_text_density(gray))

            # Black frame check
            if self._is_black_frame(gray):
                black_frames += 1

            # Contrast analysis
            contrast_scores.append(self._detect_high_contrast_text(gray))

            if prev_gray is not None:
                # Static check
                if self._is_static_frame(prev_gray, gray):
                    static_count += 1

                # Scroll detection
                is_scrolling, _ = self._detect_scrolling(prev_gray, gray)
                if is_scrolling:
                    scroll_detected += 1

            prev_gray = gray

        n = len(frames)
        return {
            "avg_text_density": np.mean(text_densities),
            "max_text_density": max(text_densities),
            "black_frame_ratio": black_frames / n,
            "static_ratio": static_count / max(1, n - 1),
            "scroll_ratio": scroll_detected / max(1, n - 1),
            "avg_contrast_score": np.mean(contrast_scores),
            "frame_count": n
        }

    def _classify_segment(
        self,
        characteristics: Dict[str, any],
        position_ratio: float,  # 0 = start, 1 = end
        duration: float
    ) -> Tuple[SegmentType, float]:
        """
        Classify a segment based on its characteristics.

        Returns (segment_type, confidence)
        """
        text_density = characteristics.get("avg_text_density", 0)
        black_ratio = characteristics.get("black_frame_ratio", 0)
        static_ratio = characteristics.get("static_ratio", 0)
        scroll_ratio = characteristics.get("scroll_ratio", 0)
        contrast_score = characteristics.get("avg_contrast_score", 0)

        confidence = 0.0
        segment_type = SegmentType.CONTENT

        # Pure black segment
        if black_ratio > 0.8:
            return SegmentType.BLACK_SEGMENT, 0.9

        # Scrolling credits (most reliable indicator)
        if scroll_ratio > 0.5 and text_density > self.TEXT_DENSITY_THRESHOLD:
            confidence = 0.7 + scroll_ratio * 0.2
            if position_ratio > 0.7:
                return SegmentType.CREDITS_CLOSING, confidence
            elif position_ratio < 0.3:
                return SegmentType.CREDITS_OPENING, min(confidence, 0.7)

        # High contrast text with stability (title cards, static credits)
        if contrast_score > 0.5 and static_ratio > 0.7:
            confidence = 0.5 + contrast_score * 0.3
            if duration < 10:
                return SegmentType.TITLE_CARD, confidence
            elif position_ratio < 0.2:
                return SegmentType.INTRO, confidence
            elif position_ratio > 0.8:
                return SegmentType.CREDITS_CLOSING, confidence

        # Static logo (short, very static, high contrast)
        if static_ratio > 0.9 and duration < 5 and contrast_score > 0.3:
            return SegmentType.STATIC_LOGO, 0.6

        # Text-heavy sections
        if text_density > self.TEXT_DENSITY_THRESHOLD * 2:
            confidence = 0.4 + min(text_density * 10, 0.3)
            if position_ratio > 0.8 and duration > self.MIN_CREDITS_DURATION:
                return SegmentType.CREDITS_CLOSING, confidence
            elif position_ratio < 0.2:
                return SegmentType.INTRO, max(confidence - 0.2, 0.3)

        return segment_type, confidence

    def analyze(self, video_path: Path) -> CreditsAnalysis:
        """
        Analyze video for intros, credits, and title sequences.

        Args:
            video_path: Path to video file

        Returns:
            CreditsAnalysis with detected segments
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0

        sample_frames = int(fps * self.sample_interval)
        if sample_frames < 1:
            sample_frames = 1

        segments = []
        current_batch = []
        batch_start_frame = 0

        self._report_progress("Analyzing video structure...", 0.0)

        frame_num = 0
        prev_characteristics = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_frames == 0:
                current_batch.append(frame)

                # Analyze in chunks of ~5 seconds
                if len(current_batch) >= int(5 / self.sample_interval):
                    chars = self._analyze_frame_batch(
                        current_batch, batch_start_frame, fps
                    )

                    position_ratio = batch_start_frame / total_frames
                    duration = len(current_batch) * self.sample_interval

                    seg_type, confidence = self._classify_segment(
                        chars, position_ratio, duration
                    )

                    if seg_type != SegmentType.CONTENT and confidence > 0.4:
                        # Check if we should merge with previous segment
                        if (segments and
                            segments[-1].segment_type == seg_type and
                            segments[-1].end_frame == batch_start_frame):
                            # Extend previous segment
                            segments[-1].end_frame = frame_num
                            segments[-1].end_time = frame_num / fps
                            segments[-1].confidence = max(
                                segments[-1].confidence, confidence
                            )
                        else:
                            segments.append(DetectedSegment(
                                segment_type=seg_type,
                                start_frame=batch_start_frame,
                                end_frame=frame_num,
                                start_time=batch_start_frame / fps,
                                end_time=frame_num / fps,
                                confidence=confidence,
                                characteristics=chars
                            ))

                    prev_characteristics = chars
                    current_batch = []
                    batch_start_frame = frame_num

            frame_num += 1

            if frame_num % (total_frames // 20 + 1) == 0:
                self._report_progress(
                    "Analyzing...",
                    frame_num / total_frames
                )

        # Process final batch
        if current_batch:
            chars = self._analyze_frame_batch(current_batch, batch_start_frame, fps)
            position_ratio = batch_start_frame / total_frames
            duration = len(current_batch) * self.sample_interval

            seg_type, confidence = self._classify_segment(
                chars, position_ratio, duration
            )

            if seg_type != SegmentType.CONTENT and confidence > 0.4:
                segments.append(DetectedSegment(
                    segment_type=seg_type,
                    start_frame=batch_start_frame,
                    end_frame=frame_num,
                    start_time=batch_start_frame / fps,
                    end_time=frame_num / fps,
                    confidence=confidence,
                    characteristics=chars
                ))

        cap.release()

        # Determine main content boundaries
        intro_end = None
        credits_start = None

        for seg in segments:
            if seg.segment_type in [SegmentType.INTRO, SegmentType.CREDITS_OPENING]:
                if seg.start_frame < total_frames * 0.3:  # Must be in first 30%
                    intro_end = max(intro_end or 0, seg.end_frame)

            if seg.segment_type == SegmentType.CREDITS_CLOSING:
                if seg.start_frame > total_frames * 0.7:  # Must be in last 30%
                    if credits_start is None or seg.start_frame < credits_start:
                        credits_start = seg.start_frame

        main_start = intro_end if intro_end else 0
        main_end = credits_start if credits_start else total_frames

        self._report_progress("Analysis complete", 1.0)

        return CreditsAnalysis(
            video_path=str(video_path),
            total_frames=total_frames,
            total_duration=total_duration,
            fps=fps,
            segments=segments,
            intro_end_frame=intro_end,
            credits_start_frame=credits_start,
            main_content_start=main_start,
            main_content_end=main_end
        )
