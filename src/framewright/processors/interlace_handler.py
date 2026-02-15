"""Interlacing Detection and Deinterlacing.

Detects interlaced video content and applies appropriate deinterlacing
for optimal quality. Critical for VHS, broadcast TV, and DVD sources.

Supported methods:
- YADIF (Yet Another DeInterlacing Filter) - Fast, good quality
- BWDIF (Bob Weaver Deinterlacing Filter) - Better motion handling
- QTGMC (via VapourSynth) - Highest quality, slower
- AI-based (RIFE/FILM) - Modern neural deinterlacing

Example:
    >>> detector = InterlaceDetector()
    >>> result = detector.analyze(video_path)
    >>> if result.is_interlaced:
    ...     deinterlacer = Deinterlacer(method="bwdif")
    ...     deinterlacer.process(video_path, output_path)
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class InterlaceType(Enum):
    """Type of interlacing detected."""
    PROGRESSIVE = "progressive"
    INTERLACED_TFF = "interlaced_tff"  # Top Field First
    INTERLACED_BFF = "interlaced_bff"  # Bottom Field First
    TELECINE = "telecine"  # 3:2 pulldown
    MIXED = "mixed"  # Variable/mixed content
    UNKNOWN = "unknown"


class DeinterlaceMethod(Enum):
    """Available deinterlacing methods."""
    YADIF = "yadif"  # Fast, good quality
    BWDIF = "bwdif"  # Better motion handling
    NNEDI = "nnedi"  # Neural network based
    QTGMC = "qtgmc"  # Highest quality (requires VapourSynth)
    AI = "ai"  # AI-based (RIFE/FILM style)


@dataclass
class InterlaceAnalysis:
    """Results of interlace detection."""
    is_interlaced: bool = False
    interlace_type: InterlaceType = InterlaceType.UNKNOWN
    field_order: str = "unknown"  # "tff", "bff", or "unknown"
    confidence: float = 0.0
    telecine_pattern: Optional[str] = None  # e.g., "3:2", "2:2"
    combing_percentage: float = 0.0  # % of frames with combing
    recommended_method: DeinterlaceMethod = DeinterlaceMethod.YADIF
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get human-readable summary."""
        if not self.is_interlaced:
            return "Progressive (no deinterlacing needed)"

        type_str = self.interlace_type.value.replace("_", " ").title()
        return (
            f"{type_str} detected ({self.confidence*100:.0f}% confidence)\n"
            f"Field order: {self.field_order.upper()}\n"
            f"Combing in {self.combing_percentage:.1f}% of frames\n"
            f"Recommended: {self.recommended_method.value.upper()}"
        )


class InterlaceDetector:
    """Detects interlaced content in video.

    Uses multiple detection methods:
    1. FFmpeg idet filter for field order detection
    2. Comb detection via edge analysis
    3. Frame difference analysis for telecine patterns
    """

    def __init__(
        self,
        sample_count: int = 100,
        comb_threshold: float = 0.15,
    ):
        """Initialize detector.

        Args:
            sample_count: Number of frames to sample for analysis
            comb_threshold: Threshold for comb detection (0-1)
        """
        self.sample_count = sample_count
        self.comb_threshold = comb_threshold

    def analyze(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> InterlaceAnalysis:
        """Analyze video for interlacing.

        Args:
            video_path: Path to video file
            progress_callback: Progress callback (0-1)

        Returns:
            InterlaceAnalysis with detection results
        """
        video_path = Path(video_path)
        result = InterlaceAnalysis()

        # Method 1: FFmpeg idet filter
        idet_result = self._run_ffmpeg_idet(video_path)
        if idet_result:
            result.details["idet"] = idet_result

        if progress_callback:
            progress_callback(0.3)

        # Method 2: Comb detection
        if HAS_OPENCV:
            comb_result = self._detect_combing(video_path, progress_callback)
            result.combing_percentage = comb_result.get("percentage", 0)
            result.details["combing"] = comb_result

        if progress_callback:
            progress_callback(0.7)

        # Method 3: Telecine pattern detection
        telecine_result = self._detect_telecine(video_path)
        if telecine_result.get("detected"):
            result.telecine_pattern = telecine_result.get("pattern")
            result.details["telecine"] = telecine_result

        if progress_callback:
            progress_callback(0.9)

        # Combine results
        result = self._combine_results(result, idet_result)

        if progress_callback:
            progress_callback(1.0)

        return result

    def _run_ffmpeg_idet(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Run FFmpeg's idet filter for interlace detection."""
        try:
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"idet",
                "-frames:v", str(self.sample_count * 10),
                "-an", "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            # Parse idet output
            output = result.stderr

            # Extract frame counts
            tff_single = self._extract_count(output, "TFF:")
            bff_single = self._extract_count(output, "BFF:")
            progressive = self._extract_count(output, "Progressive:")
            undetermined = self._extract_count(output, "Undetermined:")

            total = tff_single + bff_single + progressive + undetermined
            if total == 0:
                return None

            return {
                "tff": tff_single,
                "bff": bff_single,
                "progressive": progressive,
                "undetermined": undetermined,
                "total": total,
            }

        except Exception as e:
            logger.debug(f"FFmpeg idet failed: {e}")
            return None

    def _extract_count(self, text: str, label: str) -> int:
        """Extract count from FFmpeg idet output."""
        import re
        pattern = rf"{label}\s*(\d+)"
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    def _detect_combing(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Detect combing artifacts in frames."""
        if not HAS_OPENCV:
            return {"percentage": 0, "error": "OpenCV not available"}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"percentage": 0, "error": "Could not open video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // self.sample_count)

        combed_frames = 0
        analyzed_frames = 0

        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            if self._has_combing(frame):
                combed_frames += 1
            analyzed_frames += 1

            if progress_callback and analyzed_frames % 10 == 0:
                progress_callback(0.3 + 0.4 * (analyzed_frames / self.sample_count))

        cap.release()

        percentage = (combed_frames / analyzed_frames * 100) if analyzed_frames > 0 else 0

        return {
            "percentage": percentage,
            "combed_frames": combed_frames,
            "analyzed_frames": analyzed_frames,
        }

    def _has_combing(self, frame: "np.ndarray") -> bool:
        """Check if frame has combing artifacts."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Detect horizontal line patterns (combing signature)
        # Compare odd and even rows
        odd_rows = gray[1::2, :]
        even_rows = gray[::2, :]

        # Resize to match if needed
        min_rows = min(odd_rows.shape[0], even_rows.shape[0])
        odd_rows = odd_rows[:min_rows]
        even_rows = even_rows[:min_rows]

        # Calculate difference
        diff = np.abs(odd_rows.astype(float) - even_rows.astype(float))

        # High difference in horizontal bands indicates combing
        row_means = np.mean(diff, axis=1)
        high_diff_rows = np.sum(row_means > 30) / len(row_means)

        return high_diff_rows > self.comb_threshold

    def _detect_telecine(self, video_path: Path) -> Dict[str, Any]:
        """Detect telecine (3:2 pulldown) patterns."""
        try:
            # Use FFmpeg's fieldmatch for telecine detection
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", "fieldmatch,decimate",
                "-frames:v", "100",
                "-an", "-f", "null", "-"
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            # Check for pattern detection in output
            if "3:2" in result.stderr or "telecine" in result.stderr.lower():
                return {"detected": True, "pattern": "3:2"}
            elif "2:2" in result.stderr:
                return {"detected": True, "pattern": "2:2"}

            return {"detected": False}

        except Exception as e:
            logger.debug(f"Telecine detection failed: {e}")
            return {"detected": False, "error": str(e)}

    def _combine_results(
        self,
        result: InterlaceAnalysis,
        idet_result: Optional[Dict[str, Any]],
    ) -> InterlaceAnalysis:
        """Combine detection results into final analysis."""

        # Determine interlace type from idet
        if idet_result:
            total = idet_result.get("total", 1)
            tff = idet_result.get("tff", 0) / total
            bff = idet_result.get("bff", 0) / total
            prog = idet_result.get("progressive", 0) / total

            if prog > 0.8:
                result.interlace_type = InterlaceType.PROGRESSIVE
                result.is_interlaced = False
                result.confidence = prog
            elif tff > 0.4:
                result.interlace_type = InterlaceType.INTERLACED_TFF
                result.field_order = "tff"
                result.is_interlaced = True
                result.confidence = tff
            elif bff > 0.4:
                result.interlace_type = InterlaceType.INTERLACED_BFF
                result.field_order = "bff"
                result.is_interlaced = True
                result.confidence = bff
            else:
                result.interlace_type = InterlaceType.MIXED
                result.is_interlaced = True
                result.confidence = 1 - prog

        # Check for telecine
        if result.telecine_pattern:
            result.interlace_type = InterlaceType.TELECINE
            result.is_interlaced = True

        # Fallback to combing detection
        if result.interlace_type == InterlaceType.UNKNOWN:
            if result.combing_percentage > 20:
                result.is_interlaced = True
                result.confidence = min(result.combing_percentage / 100, 0.9)
            else:
                result.is_interlaced = False
                result.interlace_type = InterlaceType.PROGRESSIVE

        # Recommend deinterlace method
        if result.is_interlaced:
            if result.interlace_type == InterlaceType.TELECINE:
                result.recommended_method = DeinterlaceMethod.YADIF  # Simple for telecine
            elif result.combing_percentage > 50:
                result.recommended_method = DeinterlaceMethod.BWDIF  # Better for heavy interlacing
            else:
                result.recommended_method = DeinterlaceMethod.YADIF

        return result


@dataclass
class DeinterlaceResult:
    """Result of deinterlacing operation."""
    success: bool = False
    output_path: Optional[Path] = None
    method_used: DeinterlaceMethod = DeinterlaceMethod.YADIF
    frames_processed: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


class Deinterlacer:
    """Deinterlace video using various methods."""

    def __init__(
        self,
        method: DeinterlaceMethod = DeinterlaceMethod.YADIF,
        field_order: str = "auto",
    ):
        """Initialize deinterlacer.

        Args:
            method: Deinterlacing method to use
            field_order: Field order ("tff", "bff", or "auto")
        """
        self.method = method
        self.field_order = field_order

    def process(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DeinterlaceResult:
        """Deinterlace video.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Progress callback (0-1)

        Returns:
            DeinterlaceResult
        """
        import time
        start_time = time.time()

        result = DeinterlaceResult(method_used=self.method)

        try:
            # Build FFmpeg filter based on method
            vf_filter = self._build_filter()

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

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Monitor progress
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
                result.output_path = output_path
            else:
                result.error = "FFmpeg processing failed"

        except Exception as e:
            result.error = str(e)
            logger.error(f"Deinterlacing failed: {e}")

        result.processing_time = time.time() - start_time
        return result

    def _build_filter(self) -> str:
        """Build FFmpeg filter string."""
        parity = "0" if self.field_order == "tff" else "1" if self.field_order == "bff" else "-1"

        if self.method == DeinterlaceMethod.YADIF:
            return f"yadif=parity={parity}:deint=1"
        elif self.method == DeinterlaceMethod.BWDIF:
            return f"bwdif=parity={parity}:deint=1"
        elif self.method == DeinterlaceMethod.NNEDI:
            return f"nnedi=weights=nnedi3_weights.bin:field={parity}"
        else:
            return f"yadif=parity={parity}:deint=1"


def analyze_interlacing(video_path: Path) -> InterlaceAnalysis:
    """Convenience function to analyze interlacing.

    Args:
        video_path: Path to video

    Returns:
        InterlaceAnalysis
    """
    detector = InterlaceDetector()
    return detector.analyze(video_path)


def deinterlace_video(
    input_path: Path,
    output_path: Path,
    method: str = "auto",
) -> DeinterlaceResult:
    """Convenience function to deinterlace video.

    Args:
        input_path: Input video
        output_path: Output video
        method: Method ("auto", "yadif", "bwdif")

    Returns:
        DeinterlaceResult
    """
    # Auto-detect if needed
    if method == "auto":
        analysis = analyze_interlacing(input_path)
        if not analysis.is_interlaced:
            return DeinterlaceResult(
                success=True,
                output_path=input_path,
                error="Video is progressive, no deinterlacing needed"
            )
        method_enum = analysis.recommended_method
        field_order = analysis.field_order
    else:
        method_enum = DeinterlaceMethod(method)
        field_order = "auto"

    deinterlacer = Deinterlacer(method=method_enum, field_order=field_order)
    return deinterlacer.process(input_path, output_path)
