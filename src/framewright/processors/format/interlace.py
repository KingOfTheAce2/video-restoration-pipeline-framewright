"""Interlace format processor for FrameWright.

Comprehensive interlacing detection and deinterlacing with multiple methods:
- BOB (field doubling) - Fast, doubles framerate
- WEAVE - Merges fields, best for static content
- YADIF - Yet Another DeInterlacing Filter, good quality
- BWDIF - Bob Weaver DeInterlacing Filter, better motion
- NEURAL - AI-based deinterlacing (FILM/RIFE style)

Example:
    >>> from pathlib import Path
    >>> from framewright.processors.format.interlace import Deinterlacer, InterlaceConfig
    >>> from framewright.processors.format.interlace import DeinterlaceMethod
    >>> config = InterlaceConfig(method=DeinterlaceMethod.YADIF, field_order="auto")
    >>> deinterlacer = Deinterlacer(config)
    >>> if deinterlacer.detect_interlacing(frames):
    ...     restored = deinterlacer.deinterlace(frames, DeinterlaceMethod.BWDIF)
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


class DeinterlaceMethod(Enum):
    """Available deinterlacing methods."""
    BOB = "bob"         # Field doubling (2x framerate)
    WEAVE = "weave"     # Field merging (best for static)
    YADIF = "yadif"     # Yet Another DeInterlacing Filter
    BWDIF = "bwdif"     # Bob Weaver DeInterlacing Filter
    NEURAL = "neural"   # AI-based (requires neural backend)
    NNEDI = "nnedi"     # Neural Network Edge Directed Interpolation

    @property
    def doubles_framerate(self) -> bool:
        """Whether this method doubles the output framerate."""
        return self in [DeinterlaceMethod.BOB]

    @property
    def requires_neural(self) -> bool:
        """Whether this method requires neural network models."""
        return self in [DeinterlaceMethod.NEURAL, DeinterlaceMethod.NNEDI]


class FieldOrder(Enum):
    """Interlaced field order."""
    TFF = "tff"         # Top Field First
    BFF = "bff"         # Bottom Field First
    AUTO = "auto"       # Auto-detect
    UNKNOWN = "unknown"

    @property
    def ffmpeg_value(self) -> str:
        """Get FFmpeg parity value."""
        if self == FieldOrder.TFF:
            return "0"
        elif self == FieldOrder.BFF:
            return "1"
        else:
            return "-1"


class TelecinePattern(Enum):
    """Telecine pulldown patterns."""
    PATTERN_3_2 = "3:2"   # Standard NTSC telecine
    PATTERN_2_3 = "2:3"   # Alternative ordering
    PATTERN_2_2 = "2:2"   # PAL telecine / hard telecine
    PATTERN_EURO = "euro" # European pulldown
    NONE = "none"


@dataclass
class InterlaceConfig:
    """Configuration for interlace processing.

    Attributes:
        method: Default deinterlacing method.
        field_order: Field order (TFF, BFF, or auto-detect).
        telecine: Enable telecine pattern detection.
        detection_threshold: Threshold for interlace detection (0.0-1.0).
        comb_threshold: Threshold for comb artifact detection (0.0-1.0).
        sample_count: Number of frames to sample for detection.
        preserve_framerate: Prevent framerate doubling methods.
        neural_model: Path to neural model for NEURAL method.
        ffmpeg_backend: Use FFmpeg for processing.
    """
    method: DeinterlaceMethod = DeinterlaceMethod.YADIF
    field_order: FieldOrder = FieldOrder.AUTO
    telecine: bool = True
    detection_threshold: float = 0.3
    comb_threshold: float = 0.15
    sample_count: int = 50
    preserve_framerate: bool = False
    neural_model: Optional[Path] = None
    ffmpeg_backend: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.comb_threshold <= 1.0:
            raise ValueError("comb_threshold must be between 0.0 and 1.0")


@dataclass
class InterlaceAnalysis:
    """Results of interlace detection."""
    is_interlaced: bool = False
    field_order: FieldOrder = FieldOrder.UNKNOWN
    confidence: float = 0.0
    combing_percentage: float = 0.0
    telecine_pattern: TelecinePattern = TelecinePattern.NONE
    recommended_method: DeinterlaceMethod = DeinterlaceMethod.YADIF
    progressive_percentage: float = 0.0
    tff_percentage: float = 0.0
    bff_percentage: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get human-readable summary."""
        if not self.is_interlaced:
            return "Progressive video (no deinterlacing needed)"

        pattern_str = ""
        if self.telecine_pattern != TelecinePattern.NONE:
            pattern_str = f" with {self.telecine_pattern.value} telecine"

        return (
            f"Interlaced ({self.field_order.value.upper()}{pattern_str})\n"
            f"Combing in {self.combing_percentage:.1f}% of frames\n"
            f"Confidence: {self.confidence*100:.0f}%\n"
            f"Recommended: {self.recommended_method.value.upper()}"
        )


@dataclass
class DeinterlaceResult:
    """Result of deinterlacing operation."""
    success: bool = False
    method_used: DeinterlaceMethod = DeinterlaceMethod.YADIF
    field_order_used: FieldOrder = FieldOrder.AUTO
    frames_processed: int = 0
    output_framerate: Optional[float] = None
    processing_time: float = 0.0
    error: Optional[str] = None


class Deinterlacer:
    """Main interlace detection and deinterlacing processor.

    Provides comprehensive interlacing detection and multiple
    deinterlacing algorithms with quality/speed tradeoffs.
    """

    def __init__(self, config: Optional[InterlaceConfig] = None):
        """Initialize deinterlacer.

        Args:
            config: Interlace processing configuration.
        """
        self.config = config or InterlaceConfig()
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

    def detect_interlacing(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Check if frames are interlaced.

        Args:
            frames: List of frames to analyze.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            True if interlacing is detected.
        """
        analysis = self.analyze(frames, progress_callback)
        return analysis.is_interlaced

    def detect_field_order(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> FieldOrder:
        """Detect field order (TFF or BFF).

        Args:
            frames: List of frames to analyze.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Detected FieldOrder.
        """
        analysis = self.analyze(frames, progress_callback)
        return analysis.field_order

    def analyze(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> InterlaceAnalysis:
        """Analyze frames for interlacing characteristics.

        Args:
            frames: List of frames as numpy arrays.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            InterlaceAnalysis with detection results.
        """
        analysis = InterlaceAnalysis()

        if not HAS_OPENCV or not frames:
            return analysis

        # Sample frames evenly
        sample_indices = np.linspace(
            0, len(frames) - 1, min(self.config.sample_count, len(frames)),
            dtype=int
        )

        combing_scores = []
        field_order_scores = {'tff': 0, 'bff': 0, 'prog': 0}
        frame_diffs = []

        for i, idx in enumerate(sample_indices):
            frame = frames[idx]

            # Detect combing in this frame
            has_combing, comb_score = self._detect_combing(frame)
            combing_scores.append(comb_score)

            # Detect field order preference
            order_hint = self._detect_field_order_single(frame)
            if order_hint == 'tff':
                field_order_scores['tff'] += 1
            elif order_hint == 'bff':
                field_order_scores['bff'] += 1
            else:
                field_order_scores['prog'] += 1

            # Calculate frame difference for telecine detection
            if idx < len(frames) - 1:
                diff = self._frame_difference(frame, frames[min(idx + 1, len(frames) - 1)])
                frame_diffs.append(diff)

            if progress_callback:
                progress_callback((i + 1) / len(sample_indices))

        # Aggregate results
        avg_combing = np.mean(combing_scores) if combing_scores else 0
        analysis.combing_percentage = avg_combing * 100

        total_samples = sum(field_order_scores.values())
        if total_samples > 0:
            analysis.tff_percentage = field_order_scores['tff'] / total_samples * 100
            analysis.bff_percentage = field_order_scores['bff'] / total_samples * 100
            analysis.progressive_percentage = field_order_scores['prog'] / total_samples * 100

        # Determine if interlaced
        interlaced_percentage = analysis.tff_percentage + analysis.bff_percentage
        analysis.is_interlaced = (
            interlaced_percentage > 30 or
            analysis.combing_percentage > self.config.detection_threshold * 100
        )

        # Determine field order
        if analysis.tff_percentage > analysis.bff_percentage + 10:
            analysis.field_order = FieldOrder.TFF
        elif analysis.bff_percentage > analysis.tff_percentage + 10:
            analysis.field_order = FieldOrder.BFF
        else:
            analysis.field_order = FieldOrder.UNKNOWN

        # Calculate confidence
        analysis.confidence = min(1.0, abs(interlaced_percentage - 50) / 50)

        # Detect telecine pattern
        if frame_diffs:
            analysis.telecine_pattern = self._detect_telecine_pattern(frame_diffs)
            analysis.details['telecine'] = {
                'pattern': analysis.telecine_pattern.value,
                'diff_variance': float(np.var(frame_diffs)),
            }

        # Recommend deinterlacing method
        analysis.recommended_method = self._recommend_method(analysis)

        return analysis

    def deinterlace(
        self,
        frames: List[Any],
        method: Optional[DeinterlaceMethod] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Remove interlacing from frames.

        Args:
            frames: List of frames to deinterlace.
            method: Deinterlacing method (uses config default if None).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Deinterlaced frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        method = method or self.config.method

        # Auto-detect field order if needed
        field_order = self.config.field_order
        if field_order == FieldOrder.AUTO:
            field_order = self.detect_field_order(frames[:min(20, len(frames))])
            if field_order == FieldOrder.UNKNOWN:
                field_order = FieldOrder.TFF  # Default to TFF

        # Apply deinterlacing method
        if method == DeinterlaceMethod.BOB:
            result = self._deinterlace_bob(frames, field_order, progress_callback)
        elif method == DeinterlaceMethod.WEAVE:
            result = self._deinterlace_weave(frames, field_order, progress_callback)
        elif method == DeinterlaceMethod.YADIF:
            result = self._deinterlace_yadif(frames, field_order, progress_callback)
        elif method == DeinterlaceMethod.BWDIF:
            result = self._deinterlace_bwdif(frames, field_order, progress_callback)
        elif method == DeinterlaceMethod.NEURAL:
            result = self._deinterlace_neural(frames, field_order, progress_callback)
        elif method == DeinterlaceMethod.NNEDI:
            result = self._deinterlace_nnedi(frames, field_order, progress_callback)
        else:
            result = self._deinterlace_yadif(frames, field_order, progress_callback)

        return result

    def detect_telecine(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TelecinePattern:
        """Detect 3:2 pulldown pattern.

        Args:
            frames: List of frames to analyze.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Detected TelecinePattern.
        """
        if not HAS_OPENCV or not frames or len(frames) < 10:
            return TelecinePattern.NONE

        # Calculate frame differences
        diffs = []
        for i in range(min(60, len(frames) - 1)):
            diff = self._frame_difference(frames[i], frames[i + 1])
            diffs.append(diff)

            if progress_callback:
                progress_callback((i + 1) / min(60, len(frames) - 1))

        return self._detect_telecine_pattern(diffs)

    def inverse_telecine(
        self,
        frames: List[Any],
        pattern: Optional[TelecinePattern] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Remove pulldown pattern and restore original framerate.

        Args:
            frames: List of frames with telecine.
            pattern: Telecine pattern (auto-detect if None).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Frames with telecine removed (fewer frames, original cadence).
        """
        if not HAS_OPENCV or not frames:
            return frames

        # Auto-detect pattern if not provided
        if pattern is None:
            pattern = self.detect_telecine(frames[:min(60, len(frames))])

        if pattern == TelecinePattern.NONE:
            return frames

        # Calculate frame differences
        diffs = []
        for i in range(len(frames) - 1):
            diff = self._frame_difference(frames[i], frames[i + 1])
            diffs.append(diff)

        # Find duplicate frame indices
        if not diffs:
            return frames

        mean_diff = np.mean(diffs)
        threshold = mean_diff * 0.3
        duplicate_mask = np.array(diffs) < threshold

        # Remove duplicate frames
        result = [frames[0]]
        for i in range(1, len(frames)):
            if i - 1 < len(duplicate_mask) and not duplicate_mask[i - 1]:
                result.append(frames[i])

            if progress_callback:
                progress_callback(i / len(frames))

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"IVTC: Reduced {len(frames)} frames to {len(result)} frames")
        return result

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        method: Optional[DeinterlaceMethod] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DeinterlaceResult:
        """Deinterlace entire video file using FFmpeg.

        Args:
            input_path: Input video path.
            output_path: Output video path.
            method: Deinterlacing method.
            progress_callback: Progress callback.

        Returns:
            DeinterlaceResult.
        """
        import time
        start_time = time.time()

        result = DeinterlaceResult()
        method = method or self.config.method
        result.method_used = method

        if not self._ffmpeg_available:
            result.error = "FFmpeg not available"
            return result

        # Build FFmpeg filter
        field_order = self.config.field_order
        parity = field_order.ffmpeg_value

        if method == DeinterlaceMethod.YADIF:
            vf_filter = f"yadif=parity={parity}:deint=1"
        elif method == DeinterlaceMethod.BWDIF:
            vf_filter = f"bwdif=parity={parity}:deint=1"
        elif method == DeinterlaceMethod.BOB:
            vf_filter = f"yadif=mode=1:parity={parity}:deint=1"
        elif method == DeinterlaceMethod.WEAVE:
            vf_filter = "tinterlace=mode=merge"
        elif method == DeinterlaceMethod.NNEDI:
            vf_filter = f"nnedi=weights=nnedi3_weights.bin:field={parity}"
        else:
            vf_filter = f"yadif=parity={parity}:deint=1"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            for line in process.stderr:
                if "frame=" in line:
                    try:
                        frame = int(line.split("frame=")[1].split()[0])
                        result.frames_processed = frame
                    except:
                        pass

            process.wait()

            if process.returncode == 0:
                result.success = True
            else:
                result.error = "FFmpeg processing failed"

        except Exception as e:
            result.error = str(e)
            logger.error(f"Deinterlacing failed: {e}")

        result.processing_time = time.time() - start_time
        return result

    def _detect_combing(self, frame: Any) -> Tuple[bool, float]:
        """Detect combing artifacts in a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Compare odd and even rows
        odd_rows = gray[1::2, :]
        even_rows = gray[::2, :]

        min_rows = min(odd_rows.shape[0], even_rows.shape[0])
        odd_rows = odd_rows[:min_rows]
        even_rows = even_rows[:min_rows]

        # Calculate difference
        diff = np.abs(odd_rows.astype(float) - even_rows.astype(float))

        # High difference in horizontal bands indicates combing
        row_means = np.mean(diff, axis=1)
        high_diff_ratio = np.sum(row_means > 30) / len(row_means)

        has_combing = high_diff_ratio > self.config.comb_threshold
        return has_combing, high_diff_ratio

    def _detect_field_order_single(self, frame: Any) -> str:
        """Detect field order preference in a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Analyze motion between fields
        odd_rows = gray[1::2, :].astype(np.float32)
        even_rows = gray[::2, :].astype(np.float32)

        min_rows = min(odd_rows.shape[0], even_rows.shape[0])
        odd_rows = odd_rows[:min_rows]
        even_rows = even_rows[:min_rows]

        # Calculate gradients
        odd_gradient = np.abs(np.diff(odd_rows, axis=0)).mean()
        even_gradient = np.abs(np.diff(even_rows, axis=0)).mean()

        diff = np.abs(odd_rows - even_rows).mean()

        # Low difference suggests progressive
        if diff < 5:
            return 'prog'

        # Higher odd gradient suggests TFF (odd field is newer)
        if odd_gradient > even_gradient * 1.1:
            return 'tff'
        elif even_gradient > odd_gradient * 1.1:
            return 'bff'

        return 'unknown'

    def _frame_difference(self, frame1: Any, frame2: Any) -> float:
        """Calculate difference between two frames."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
        return np.mean(diff)

    def _detect_telecine_pattern(self, diffs: List[float]) -> TelecinePattern:
        """Detect telecine pattern from frame differences."""
        if len(diffs) < 10:
            return TelecinePattern.NONE

        diffs_array = np.array(diffs)
        mean_diff = np.mean(diffs_array)

        # Find low-difference frames (potential duplicates)
        threshold = mean_diff * 0.3
        is_duplicate = diffs_array < threshold

        duplicate_ratio = np.sum(is_duplicate) / len(is_duplicate)

        # 3:2 pulldown has ~40% duplicates (2 out of 5)
        if 0.35 < duplicate_ratio < 0.45:
            # Verify 5-frame pattern
            for offset in range(5):
                pattern_matches = 0
                total_checks = 0
                for i in range(offset, len(is_duplicate) - 5, 5):
                    # In 3:2, positions 0,2 or 1,3 or 2,4 should be duplicates
                    if i + 2 < len(is_duplicate):
                        total_checks += 1
                        if is_duplicate[i] or is_duplicate[i + 2]:
                            pattern_matches += 1

                if total_checks > 0 and pattern_matches / total_checks > 0.6:
                    return TelecinePattern.PATTERN_3_2

        # 2:2 pulldown has ~50% duplicates
        if 0.45 < duplicate_ratio < 0.55:
            return TelecinePattern.PATTERN_2_2

        return TelecinePattern.NONE

    def _recommend_method(self, analysis: InterlaceAnalysis) -> DeinterlaceMethod:
        """Recommend best deinterlacing method based on analysis."""
        if not analysis.is_interlaced:
            return DeinterlaceMethod.WEAVE

        if analysis.telecine_pattern != TelecinePattern.NONE:
            # For telecined content, simple YADIF works well
            return DeinterlaceMethod.YADIF

        if analysis.combing_percentage > 50:
            # Heavy interlacing benefits from BWDIF
            return DeinterlaceMethod.BWDIF

        # Default to YADIF for most content
        return DeinterlaceMethod.YADIF

    def _deinterlace_bob(
        self,
        frames: List[Any],
        field_order: FieldOrder,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """BOB deinterlacing - double framerate by expanding fields."""
        result = []
        is_tff = field_order == FieldOrder.TFF

        for i, frame in enumerate(frames):
            height, width = frame.shape[:2]

            # Extract odd and even fields
            if len(frame.shape) == 3:
                even_field = frame[::2, :, :]
                odd_field = frame[1::2, :, :]
            else:
                even_field = frame[::2, :]
                odd_field = frame[1::2, :]

            # Resize fields to full height
            even_frame = cv2.resize(even_field, (width, height), interpolation=cv2.INTER_LINEAR)
            odd_frame = cv2.resize(odd_field, (width, height), interpolation=cv2.INTER_LINEAR)

            # Order based on field order
            if is_tff:
                result.extend([even_frame, odd_frame])
            else:
                result.extend([odd_frame, even_frame])

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def _deinterlace_weave(
        self,
        frames: List[Any],
        field_order: FieldOrder,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """WEAVE deinterlacing - merge fields (best for static content)."""
        # For weave, we just pass through since the frame already contains both fields
        # This is mainly useful when the content is actually progressive
        if progress_callback:
            progress_callback(1.0)
        return frames

    def _deinterlace_yadif(
        self,
        frames: List[Any],
        field_order: FieldOrder,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """YADIF-style deinterlacing using OpenCV."""
        result = []
        is_tff = field_order == FieldOrder.TFF

        for i, frame in enumerate(frames):
            height, width = frame.shape[:2]
            output = frame.copy()

            # Interpolate the lines that belong to the "other" field
            for y in range(1, height - 1):
                is_odd_line = y % 2 == 1

                # Determine if this line needs interpolation
                needs_interp = (is_tff and is_odd_line) or (not is_tff and not is_odd_line)

                if needs_interp:
                    # Temporal and spatial interpolation
                    if len(frame.shape) == 3:
                        # Use vertical neighbors for interpolation
                        output[y] = (
                            frame[y - 1].astype(np.float32) * 0.5 +
                            frame[y + 1].astype(np.float32) * 0.5
                        ).astype(np.uint8)
                    else:
                        output[y] = (
                            frame[y - 1].astype(np.float32) * 0.5 +
                            frame[y + 1].astype(np.float32) * 0.5
                        ).astype(np.uint8)

            result.append(output)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def _deinterlace_bwdif(
        self,
        frames: List[Any],
        field_order: FieldOrder,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """BWDIF-style deinterlacing with better motion handling."""
        result = []
        is_tff = field_order == FieldOrder.TFF

        for i, frame in enumerate(frames):
            height, width = frame.shape[:2]
            output = frame.copy()

            # Get temporal neighbors if available
            prev_frame = frames[i - 1] if i > 0 else frame
            next_frame = frames[i + 1] if i < len(frames) - 1 else frame

            for y in range(2, height - 2):
                is_odd_line = y % 2 == 1
                needs_interp = (is_tff and is_odd_line) or (not is_tff and not is_odd_line)

                if needs_interp:
                    if len(frame.shape) == 3:
                        # Weighted interpolation with more spatial context
                        spatial = (
                            frame[y - 2].astype(np.float32) * -0.0625 +
                            frame[y - 1].astype(np.float32) * 0.5625 +
                            frame[y + 1].astype(np.float32) * 0.5625 +
                            frame[y + 2].astype(np.float32) * -0.0625
                        )

                        # Temporal component
                        temporal = (
                            prev_frame[y].astype(np.float32) * 0.25 +
                            next_frame[y].astype(np.float32) * 0.25
                        )

                        # Blend spatial and temporal
                        output[y] = np.clip(spatial * 0.75 + temporal * 0.25, 0, 255).astype(np.uint8)
                    else:
                        spatial = (
                            frame[y - 2].astype(np.float32) * -0.0625 +
                            frame[y - 1].astype(np.float32) * 0.5625 +
                            frame[y + 1].astype(np.float32) * 0.5625 +
                            frame[y + 2].astype(np.float32) * -0.0625
                        )
                        temporal = (
                            prev_frame[y].astype(np.float32) * 0.25 +
                            next_frame[y].astype(np.float32) * 0.25
                        )
                        output[y] = np.clip(spatial * 0.75 + temporal * 0.25, 0, 255).astype(np.uint8)

            result.append(output)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def _deinterlace_neural(
        self,
        frames: List[Any],
        field_order: FieldOrder,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Neural network-based deinterlacing (placeholder - falls back to BWDIF)."""
        logger.warning("Neural deinterlacing not implemented, falling back to BWDIF")
        return self._deinterlace_bwdif(frames, field_order, progress_callback)

    def _deinterlace_nnedi(
        self,
        frames: List[Any],
        field_order: FieldOrder,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """NNEDI-style deinterlacing (placeholder - falls back to BWDIF)."""
        logger.warning("NNEDI deinterlacing not implemented, falling back to BWDIF")
        return self._deinterlace_bwdif(frames, field_order, progress_callback)


def create_deinterlacer(
    method: str = "yadif",
    field_order: str = "auto",
    telecine_detection: bool = True,
) -> Deinterlacer:
    """Factory function to create a Deinterlacer.

    Args:
        method: Deinterlacing method name.
        field_order: Field order ("tff", "bff", or "auto").
        telecine_detection: Enable telecine pattern detection.

    Returns:
        Configured Deinterlacer.
    """
    try:
        deint_method = DeinterlaceMethod(method.lower())
    except ValueError:
        deint_method = DeinterlaceMethod.YADIF

    try:
        order = FieldOrder(field_order.lower())
    except ValueError:
        order = FieldOrder.AUTO

    config = InterlaceConfig(
        method=deint_method,
        field_order=order,
        telecine=telecine_detection,
    )

    return Deinterlacer(config)
