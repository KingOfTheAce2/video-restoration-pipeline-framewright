"""VHS and analog media restoration processor.

Specialized restoration for VHS, Betamax, Hi8, and other analog formats
with authentic preservation of period characteristics.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import json
import tempfile
import shutil

logger = logging.getLogger(__name__)


class AnalogFormat(Enum):
    """Analog video format types."""
    VHS = "vhs"
    VHS_C = "vhs_c"
    SVHS = "svhs"
    BETAMAX = "betamax"
    BETACAM = "betacam"
    HI8 = "hi8"
    VIDEO8 = "video8"
    UMATIC = "umatic"
    LASERDISC = "laserdisc"
    V2000 = "v2000"
    CED = "ced"  # Capacitance Electronic Disc
    UNKNOWN = "unknown"


@dataclass
class AnalogFormatProfile:
    """Technical characteristics of analog format."""
    format_type: AnalogFormat
    horizontal_resolution: int  # Lines
    chroma_bandwidth: float  # MHz
    luma_bandwidth: float  # MHz
    snr_db: float  # Signal-to-noise ratio
    typical_artifacts: List[str] = field(default_factory=list)
    color_under_frequency: Optional[float] = None  # kHz


# Technical profiles for each format
ANALOG_PROFILES = {
    AnalogFormat.VHS: AnalogFormatProfile(
        format_type=AnalogFormat.VHS,
        horizontal_resolution=240,
        chroma_bandwidth=0.5,
        luma_bandwidth=3.0,
        snr_db=42,
        typical_artifacts=["head_switching", "tracking", "dropout", "chroma_bleed", "jitter"],
        color_under_frequency=629,
    ),
    AnalogFormat.SVHS: AnalogFormatProfile(
        format_type=AnalogFormat.SVHS,
        horizontal_resolution=400,
        chroma_bandwidth=0.5,
        luma_bandwidth=5.0,
        snr_db=45,
        typical_artifacts=["head_switching", "dropout", "minor_jitter"],
        color_under_frequency=629,
    ),
    AnalogFormat.BETAMAX: AnalogFormatProfile(
        format_type=AnalogFormat.BETAMAX,
        horizontal_resolution=250,
        chroma_bandwidth=0.5,
        luma_bandwidth=3.2,
        snr_db=43,
        typical_artifacts=["head_switching", "tracking", "dropout"],
        color_under_frequency=688,
    ),
    AnalogFormat.HI8: AnalogFormatProfile(
        format_type=AnalogFormat.HI8,
        horizontal_resolution=400,
        chroma_bandwidth=0.5,
        luma_bandwidth=5.0,
        snr_db=46,
        typical_artifacts=["dropout", "head_switching"],
        color_under_frequency=743,
    ),
    AnalogFormat.VIDEO8: AnalogFormatProfile(
        format_type=AnalogFormat.VIDEO8,
        horizontal_resolution=240,
        chroma_bandwidth=0.5,
        luma_bandwidth=3.0,
        snr_db=43,
        typical_artifacts=["dropout", "head_switching", "noise"],
        color_under_frequency=743,
    ),
    AnalogFormat.UMATIC: AnalogFormatProfile(
        format_type=AnalogFormat.UMATIC,
        horizontal_resolution=280,
        chroma_bandwidth=1.0,
        luma_bandwidth=4.2,
        snr_db=45,
        typical_artifacts=["dropout", "color_fringing"],
        color_under_frequency=None,  # Direct color
    ),
    AnalogFormat.LASERDISC: AnalogFormatProfile(
        format_type=AnalogFormat.LASERDISC,
        horizontal_resolution=425,
        chroma_bandwidth=1.5,
        luma_bandwidth=5.0,
        snr_db=48,
        typical_artifacts=["laser_rot", "crosstalk", "dropout"],
        color_under_frequency=None,
    ),
}


@dataclass
class VHSArtifactAnalysis:
    """Analysis of VHS-specific artifacts in a frame or video."""
    head_switching_detected: bool = False
    head_switching_position: Optional[int] = None  # Y position
    head_switching_severity: float = 0.0

    tracking_issues: bool = False
    tracking_severity: float = 0.0
    tracking_line_positions: List[int] = field(default_factory=list)

    dropout_detected: bool = False
    dropout_count: int = 0
    dropout_positions: List[Tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h

    chroma_bleed: bool = False
    chroma_bleed_severity: float = 0.0

    jitter_detected: bool = False
    jitter_severity: float = 0.0

    rainbow_effect: bool = False  # Composite video artifact
    dot_crawl: bool = False  # Composite video artifact

    overall_degradation: float = 0.0  # 0-1 scale


@dataclass
class VHSRestorationConfig:
    """Configuration for VHS restoration."""
    # Format detection
    auto_detect_format: bool = True
    assumed_format: AnalogFormat = AnalogFormat.VHS

    # Head switching removal
    remove_head_switching: bool = True
    head_switching_threshold: float = 0.3
    head_switching_blend_height: int = 4

    # Tracking line repair
    repair_tracking_lines: bool = True
    tracking_detection_threshold: float = 0.4
    tracking_interpolation_method: str = "temporal"  # temporal, spatial, hybrid

    # Dropout repair
    repair_dropouts: bool = True
    dropout_min_length: int = 3  # pixels
    dropout_interpolation: str = "temporal"  # temporal, spatial, inpaint
    dropout_search_range: int = 5  # frames

    # Chroma restoration
    fix_chroma_bleed: bool = True
    chroma_delay_compensation: int = 0  # samples
    chroma_noise_reduction: float = 0.3

    # Time base correction
    enable_tbc_simulation: bool = True
    tbc_dejitter_strength: float = 0.5
    tbc_line_sync: bool = True

    # Composite video cleanup
    remove_dot_crawl: bool = True
    remove_rainbow: bool = True
    comb_filter_strength: float = 0.7

    # Noise characteristics
    preserve_authentic_noise: bool = True
    noise_reduction_strength: float = 0.3
    temporal_noise_filter: bool = True

    # Luma/Chroma bandwidth
    restore_luma_bandwidth: bool = True
    restore_chroma_bandwidth: bool = True
    target_luma_bandwidth: Optional[float] = None  # Auto from format
    target_chroma_bandwidth: Optional[float] = None

    # Color correction
    fix_color_bleeding: bool = True
    correct_color_phase: bool = True
    color_phase_correction: float = 0.0  # degrees

    # Authenticity limits (from core system)
    max_enhancement: float = 0.6
    preserve_format_character: bool = True


class VHSArtifactDetector:
    """Detects VHS-specific artifacts in video frames."""

    def __init__(self, config: VHSRestorationConfig):
        self.config = config
        self._numpy_available = self._check_numpy()

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def analyze_frame(self, frame: Any, frame_number: int = 0) -> VHSArtifactAnalysis:
        """Analyze a single frame for VHS artifacts."""
        analysis = VHSArtifactAnalysis()

        if not self._numpy_available:
            return analysis

        import numpy as np

        if frame is None or not isinstance(frame, np.ndarray):
            return analysis

        height, width = frame.shape[:2]
        gray = frame if len(frame.shape) == 2 else np.mean(frame, axis=2).astype(np.uint8)

        # Detect head switching noise (typically in bottom 10-20 lines)
        analysis.head_switching_detected, analysis.head_switching_position, analysis.head_switching_severity = \
            self._detect_head_switching(gray, height)

        # Detect tracking lines (horizontal disturbances)
        analysis.tracking_issues, analysis.tracking_severity, analysis.tracking_line_positions = \
            self._detect_tracking_lines(gray, height)

        # Detect dropouts (sudden brightness changes in horizontal runs)
        analysis.dropout_detected, analysis.dropout_count, analysis.dropout_positions = \
            self._detect_dropouts(gray, width, height)

        # Detect chroma bleed (color channel misalignment)
        if len(frame.shape) == 3:
            analysis.chroma_bleed, analysis.chroma_bleed_severity = \
                self._detect_chroma_bleed(frame)

        # Detect jitter (horizontal line displacement)
        analysis.jitter_detected, analysis.jitter_severity = \
            self._detect_jitter(gray, width)

        # Detect composite artifacts
        if len(frame.shape) == 3:
            analysis.rainbow_effect = self._detect_rainbow_effect(frame)
            analysis.dot_crawl = self._detect_dot_crawl(frame)

        # Calculate overall degradation score
        analysis.overall_degradation = self._calculate_degradation_score(analysis)

        return analysis

    def _detect_head_switching(self, gray: Any, height: int) -> Tuple[bool, Optional[int], float]:
        """Detect head switching noise at bottom of frame."""
        import numpy as np

        # Check bottom 30 lines for characteristic noise
        bottom_region = gray[height - 30:, :]

        # Head switching shows as horizontal high-frequency noise
        row_variances = np.var(np.diff(bottom_region.astype(np.float32), axis=1), axis=1)

        # Find rows with abnormally high variance
        mean_var = np.mean(row_variances)
        threshold = mean_var * (2.0 + self.config.head_switching_threshold)

        noisy_rows = np.where(row_variances > threshold)[0]

        if len(noisy_rows) > 2:
            position = height - 30 + int(np.min(noisy_rows))
            severity = min(1.0, len(noisy_rows) / 15.0)
            return True, position, severity

        return False, None, 0.0

    def _detect_tracking_lines(self, gray: Any, height: int) -> Tuple[bool, float, List[int]]:
        """Detect horizontal tracking disturbances."""
        import numpy as np

        # Calculate horizontal gradient magnitude per row
        h_diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
        row_activity = np.mean(h_diff, axis=1)

        # Smooth to find local anomalies
        kernel_size = 5
        smoothed = np.convolve(row_activity, np.ones(kernel_size) / kernel_size, mode='same')

        # Find rows with activity much higher than neighbors
        local_deviation = np.abs(row_activity - smoothed)
        threshold = np.std(local_deviation) * (2.0 + self.config.tracking_detection_threshold)

        tracking_lines = np.where(local_deviation > threshold)[0].tolist()

        # Filter out head switching region
        tracking_lines = [y for y in tracking_lines if y < height - 30]

        if len(tracking_lines) > 0:
            severity = min(1.0, len(tracking_lines) / 20.0)
            return True, severity, tracking_lines

        return False, 0.0, []

    def _detect_dropouts(self, gray: Any, width: int, height: int) -> Tuple[bool, int, List[Tuple[int, int, int, int]]]:
        """Detect dropout artifacts (brief signal losses)."""
        import numpy as np

        dropouts = []

        # Scan for horizontal runs of very bright or dark pixels
        for y in range(height):
            row = gray[y, :]

            # Find runs of extreme values
            bright_mask = row > 250
            dark_mask = row < 5

            for mask in [bright_mask, dark_mask]:
                # Find contiguous runs
                changes = np.diff(mask.astype(int))
                starts = np.where(changes == 1)[0] + 1
                ends = np.where(changes == -1)[0] + 1

                if mask[0]:
                    starts = np.concatenate([[0], starts])
                if mask[-1]:
                    ends = np.concatenate([ends, [width]])

                for start, end in zip(starts, ends):
                    length = end - start
                    if length >= self.config.dropout_min_length:
                        dropouts.append((int(start), y, int(length), 1))

        # Merge adjacent dropouts vertically
        merged_dropouts = self._merge_dropouts(dropouts)

        if len(merged_dropouts) > 0:
            return True, len(merged_dropouts), merged_dropouts

        return False, 0, []

    def _merge_dropouts(self, dropouts: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge vertically adjacent dropout regions."""
        if not dropouts:
            return []

        # Sort by y, then x
        sorted_dropouts = sorted(dropouts, key=lambda d: (d[1], d[0]))

        merged = []
        current = list(sorted_dropouts[0])

        for d in sorted_dropouts[1:]:
            # Check if adjacent and overlapping in x
            if (d[1] <= current[1] + current[3] + 1 and
                d[0] < current[0] + current[2] and
                d[0] + d[2] > current[0]):
                # Merge
                new_x = min(current[0], d[0])
                new_w = max(current[0] + current[2], d[0] + d[2]) - new_x
                new_h = d[1] + d[3] - current[1]
                current = [new_x, current[1], new_w, new_h]
            else:
                merged.append(tuple(current))
                current = list(d)

        merged.append(tuple(current))
        return merged

    def _detect_chroma_bleed(self, frame: Any) -> Tuple[bool, float]:
        """Detect color channel bleeding/misalignment."""
        import numpy as np

        if frame.shape[2] < 3:
            return False, 0.0

        # Convert to YUV-like space
        y = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]

        # Check for horizontal color shifts at edges
        edges_y = np.abs(np.diff(y.astype(np.float32), axis=1))
        edge_mask = edges_y > 30

        if not np.any(edge_mask):
            return False, 0.0

        # At strong luma edges, check if color channels are offset
        r_diff = np.abs(np.diff(frame[:, :, 2].astype(np.float32), axis=1))
        g_diff = np.abs(np.diff(frame[:, :, 1].astype(np.float32), axis=1))
        b_diff = np.abs(np.diff(frame[:, :, 0].astype(np.float32), axis=1))

        # Calculate color edge offset from luma edge
        edge_positions = np.where(edge_mask)

        if len(edge_positions[0]) < 10:
            return False, 0.0

        # Sample some edges
        sample_indices = np.random.choice(len(edge_positions[0]),
                                         min(100, len(edge_positions[0])),
                                         replace=False)

        offsets = []
        for idx in sample_indices:
            y_pos, x_pos = edge_positions[0][idx], edge_positions[1][idx]

            # Check color channel edge positions nearby
            search_range = 5
            x_start = max(0, x_pos - search_range)
            x_end = min(frame.shape[1] - 2, x_pos + search_range)

            r_local = r_diff[y_pos, x_start:x_end]
            b_local = b_diff[y_pos, x_start:x_end]

            if len(r_local) > 0 and np.max(r_local) > 20:
                r_edge = x_start + np.argmax(r_local)
                offsets.append(abs(r_edge - x_pos))

            if len(b_local) > 0 and np.max(b_local) > 20:
                b_edge = x_start + np.argmax(b_local)
                offsets.append(abs(b_edge - x_pos))

        if offsets:
            mean_offset = np.mean(offsets)
            if mean_offset > 1.5:
                severity = min(1.0, mean_offset / 5.0)
                return True, severity

        return False, 0.0

    def _detect_jitter(self, gray: Any, width: int) -> Tuple[bool, float]:
        """Detect horizontal line displacement (jitter)."""
        import numpy as np

        # Calculate row-to-row horizontal shift using correlation
        shifts = []

        for y in range(1, gray.shape[0] - 1, 5):  # Sample every 5 rows
            row_curr = gray[y, :].astype(np.float32)
            row_prev = gray[y - 1, :].astype(np.float32)

            # Cross-correlation to find shift
            correlation = np.correlate(row_curr, row_prev, mode='same')
            shift = np.argmax(correlation) - width // 2
            shifts.append(shift)

        shifts = np.array(shifts)

        # High variance in shifts indicates jitter
        shift_variance = np.var(shifts)

        if shift_variance > 2.0:
            severity = min(1.0, shift_variance / 10.0)
            return True, severity

        return False, 0.0

    def _detect_rainbow_effect(self, frame: Any) -> bool:
        """Detect rainbow effect from composite video."""
        import numpy as np

        # Rainbow effect shows as periodic color patterns at ~3.58 MHz
        # This manifests as diagonal color stripes

        # Convert to HSV-like hue
        r, g, b = frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)

        # Check for periodic diagonal patterns in saturation
        # This is a simplified check
        sat_fft = np.fft.fft2(saturation)
        magnitude = np.abs(sat_fft)

        # Look for peaks at diagonal frequencies
        h, w = magnitude.shape
        diag_region = magnitude[h//4:h//2, w//4:w//2]

        mean_mag = np.mean(magnitude)
        diag_max = np.max(diag_region)

        return diag_max > mean_mag * 5

    def _detect_dot_crawl(self, frame: Any) -> bool:
        """Detect dot crawl artifact from composite video."""
        import numpy as np

        # Dot crawl shows as moving dots along color edges
        # Detect as periodic horizontal pattern in chroma

        r, g, b = frame[:, :, 2].astype(np.float32), frame[:, :, 1].astype(np.float32), frame[:, :, 0].astype(np.float32)

        # Simple chroma difference
        chroma = np.abs(r - b)

        # Check for high-frequency horizontal pattern
        h_diff = np.abs(np.diff(chroma, axis=1))

        # Calculate periodicity
        row_means = np.mean(h_diff, axis=0)

        if len(row_means) < 10:
            return False

        # FFT to find periodic components
        fft = np.fft.fft(row_means)
        magnitude = np.abs(fft[1:len(fft)//2])

        if len(magnitude) == 0:
            return False

        # Look for strong periodic component
        peak = np.max(magnitude)
        mean = np.mean(magnitude)

        return peak > mean * 8

    def _calculate_degradation_score(self, analysis: VHSArtifactAnalysis) -> float:
        """Calculate overall degradation score from artifact analysis."""
        score = 0.0

        # Weight each artifact type
        weights = {
            'head_switching': 0.15,
            'tracking': 0.25,
            'dropout': 0.20,
            'chroma_bleed': 0.15,
            'jitter': 0.10,
            'rainbow': 0.08,
            'dot_crawl': 0.07,
        }

        if analysis.head_switching_detected:
            score += weights['head_switching'] * analysis.head_switching_severity

        if analysis.tracking_issues:
            score += weights['tracking'] * analysis.tracking_severity

        if analysis.dropout_detected:
            dropout_severity = min(1.0, analysis.dropout_count / 50.0)
            score += weights['dropout'] * dropout_severity

        if analysis.chroma_bleed:
            score += weights['chroma_bleed'] * analysis.chroma_bleed_severity

        if analysis.jitter_detected:
            score += weights['jitter'] * analysis.jitter_severity

        if analysis.rainbow_effect:
            score += weights['rainbow']

        if analysis.dot_crawl:
            score += weights['dot_crawl']

        return min(1.0, score)


class VHSRestorer:
    """Restores VHS and analog video artifacts."""

    def __init__(self, config: Optional[VHSRestorationConfig] = None):
        self.config = config or VHSRestorationConfig()
        self.detector = VHSArtifactDetector(self.config)
        self._numpy_available = self._check_numpy()
        self._ffmpeg_path = self._find_ffmpeg()

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        ffmpeg = shutil.which("ffmpeg")
        return ffmpeg

    def detect_format(self, video_path: Path) -> AnalogFormat:
        """Auto-detect the analog format from video characteristics."""
        if not self._ffmpeg_path:
            return AnalogFormat.UNKNOWN

        # Get video info
        try:
            cmd = [
                self._ffmpeg_path, "-i", str(video_path),
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            info = result.stderr

            # Parse resolution
            import re
            res_match = re.search(r'(\d{3,4})x(\d{3,4})', info)
            if res_match:
                width, height = int(res_match.group(1)), int(res_match.group(2))
            else:
                width, height = 720, 480

            # Estimate format from resolution and characteristics
            if width >= 720:
                # Could be S-VHS, Hi8, or LaserDisc
                return AnalogFormat.SVHS
            elif width >= 640:
                # Standard VHS or Betamax
                return AnalogFormat.VHS
            else:
                # Low resolution, likely VHS
                return AnalogFormat.VHS

        except Exception as e:
            logger.warning(f"Format detection failed: {e}")
            return AnalogFormat.UNKNOWN

    def analyze_video(self, video_path: Path, sample_count: int = 10) -> Dict[str, Any]:
        """Analyze video for VHS artifacts."""
        if not self._numpy_available or not self._ffmpeg_path:
            return {"error": "Dependencies not available"}

        import numpy as np

        # Extract sample frames
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_pattern = Path(tmpdir) / "frame_%04d.png"

            # Get duration
            duration = self._get_duration(video_path)
            if duration <= 0:
                duration = 60  # Assume 1 minute

            # Extract frames at intervals
            interval = duration / (sample_count + 1)

            analyses = []

            for i in range(sample_count):
                timestamp = interval * (i + 1)
                frame_path = Path(tmpdir) / f"frame_{i:04d}.png"

                cmd = [
                    self._ffmpeg_path,
                    "-ss", str(timestamp),
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-y",
                    str(frame_path)
                ]

                try:
                    subprocess.run(cmd, capture_output=True, timeout=30)

                    if frame_path.exists():
                        # Load and analyze frame
                        import cv2
                        frame = cv2.imread(str(frame_path))
                        if frame is not None:
                            analysis = self.detector.analyze_frame(frame, i)
                            analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Frame analysis failed at {timestamp}s: {e}")

        # Aggregate results
        if not analyses:
            return {"error": "No frames analyzed"}

        return {
            "format_detected": self.detect_format(video_path).value,
            "sample_count": len(analyses),
            "head_switching_frequency": sum(1 for a in analyses if a.head_switching_detected) / len(analyses),
            "tracking_issues_frequency": sum(1 for a in analyses if a.tracking_issues) / len(analyses),
            "dropout_frequency": sum(1 for a in analyses if a.dropout_detected) / len(analyses),
            "chroma_bleed_frequency": sum(1 for a in analyses if a.chroma_bleed) / len(analyses),
            "jitter_frequency": sum(1 for a in analyses if a.jitter_detected) / len(analyses),
            "avg_degradation": np.mean([a.overall_degradation for a in analyses]),
            "max_degradation": max(a.overall_degradation for a in analyses),
            "dropout_total": sum(a.dropout_count for a in analyses),
            "rainbow_detected": any(a.rainbow_effect for a in analyses),
            "dot_crawl_detected": any(a.dot_crawl for a in analyses),
        }

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        try:
            cmd = [
                self._ffmpeg_path,
                "-i", str(video_path),
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            import re
            match = re.search(r'Duration: (\d+):(\d+):(\d+\.?\d*)', result.stderr)
            if match:
                hours, minutes, seconds = match.groups()
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except Exception:
            pass

        return 0

    def restore_frame(self, frame: Any, analysis: Optional[VHSArtifactAnalysis] = None) -> Any:
        """Restore a single frame with VHS artifact removal."""
        if not self._numpy_available:
            return frame

        import numpy as np

        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        # Analyze if not provided
        if analysis is None:
            analysis = self.detector.analyze_frame(frame)

        restored = frame.copy()

        # Apply restorations based on detected artifacts
        if self.config.remove_head_switching and analysis.head_switching_detected:
            restored = self._remove_head_switching(restored, analysis)

        if self.config.repair_tracking_lines and analysis.tracking_issues:
            restored = self._repair_tracking_lines(restored, analysis)

        if self.config.repair_dropouts and analysis.dropout_detected:
            restored = self._repair_dropouts(restored, analysis)

        if self.config.fix_chroma_bleed and analysis.chroma_bleed:
            restored = self._fix_chroma_bleed(restored, analysis)

        if self.config.enable_tbc_simulation and analysis.jitter_detected:
            restored = self._apply_tbc(restored, analysis)

        if self.config.remove_dot_crawl and analysis.dot_crawl:
            restored = self._remove_dot_crawl(restored)

        if self.config.remove_rainbow and analysis.rainbow_effect:
            restored = self._remove_rainbow(restored)

        return restored

    def _remove_head_switching(self, frame: Any, analysis: VHSArtifactAnalysis) -> Any:
        """Remove head switching noise from bottom of frame."""
        import numpy as np

        if analysis.head_switching_position is None:
            return frame

        height = frame.shape[0]
        switch_pos = analysis.head_switching_position
        blend_height = self.config.head_switching_blend_height

        # Replace affected region with interpolated content
        if switch_pos > blend_height:
            # Get clean region above
            clean_region = frame[switch_pos - blend_height:switch_pos, :]

            # Create gradient blend
            for y in range(switch_pos, min(height, switch_pos + 20)):
                blend_factor = 1.0 - (y - switch_pos) / 20.0
                if y < height and switch_pos - blend_height + (y - switch_pos) % blend_height < switch_pos:
                    source_y = switch_pos - blend_height + (y - switch_pos) % blend_height
                    frame[y] = (blend_factor * clean_region[(y - switch_pos) % blend_height] +
                               (1 - blend_factor) * frame[y]).astype(np.uint8)

        return frame

    def _repair_tracking_lines(self, frame: Any, analysis: VHSArtifactAnalysis) -> Any:
        """Repair horizontal tracking disturbances."""
        import numpy as np

        for y in analysis.tracking_line_positions:
            if y > 0 and y < frame.shape[0] - 1:
                # Interpolate from neighboring lines
                frame[y] = ((frame[y - 1].astype(np.float32) +
                            frame[y + 1].astype(np.float32)) / 2).astype(np.uint8)

        return frame

    def _repair_dropouts(self, frame: Any, analysis: VHSArtifactAnalysis) -> Any:
        """Repair dropout artifacts with spatial interpolation."""
        import numpy as np

        for x, y, w, h in analysis.dropout_positions:
            # Spatial interpolation for single-frame repair
            x_end = min(x + w, frame.shape[1])
            y_end = min(y + h, frame.shape[0])

            if x > 0 and x_end < frame.shape[1]:
                # Linear interpolation horizontally
                left = frame[y:y_end, x - 1:x]
                right = frame[y:y_end, x_end:x_end + 1]

                for xi in range(x, x_end):
                    t = (xi - x + 1) / (w + 1)
                    frame[y:y_end, xi:xi + 1] = (
                        (1 - t) * left + t * right
                    ).astype(np.uint8)

        return frame

    def _fix_chroma_bleed(self, frame: Any, analysis: VHSArtifactAnalysis) -> Any:
        """Fix color channel bleeding/misalignment."""
        import numpy as np

        if frame.shape[2] < 3:
            return frame

        # Apply slight horizontal shift to color channels
        shift_amount = int(analysis.chroma_bleed_severity * 2)
        if shift_amount > 0:
            # Shift red channel left, blue channel right (or vice versa)
            r = frame[:, :, 2]
            b = frame[:, :, 0]

            # Roll channels to align
            frame[:, shift_amount:, 2] = r[:, :-shift_amount]
            frame[:, :-shift_amount, 0] = b[:, shift_amount:]

        return frame

    def _apply_tbc(self, frame: Any, analysis: VHSArtifactAnalysis) -> Any:
        """Apply time base correction (dejitter)."""
        import numpy as np

        # Simple horizontal stabilization
        # In real TBC, this would use line sync detection

        height = frame.shape[0]
        width = frame.shape[1]

        # Calculate per-line horizontal offset
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame

        reference_line = gray[height // 2, :]

        for y in range(height):
            line = gray[y, :]

            # Cross-correlation to find shift
            correlation = np.correlate(line, reference_line, mode='same')
            shift = np.argmax(correlation) - width // 2

            # Apply correction with strength factor
            correction = int(shift * self.config.tbc_dejitter_strength)

            if correction != 0 and abs(correction) < width // 4:
                frame[y] = np.roll(frame[y], -correction, axis=0)

        return frame

    def _remove_dot_crawl(self, frame: Any) -> Any:
        """Remove dot crawl artifact using comb filtering."""
        import numpy as np

        # Simple notch filter at chroma subcarrier frequency
        # In NTSC, this is approximately 3.58 MHz

        # Convert to float
        frame_float = frame.astype(np.float32)

        # Apply horizontal low-pass to chroma while preserving luma
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size

        for c in range(frame.shape[2]):
            for y in range(frame.shape[0]):
                frame_float[y, :, c] = np.convolve(
                    frame_float[y, :, c], kernel, mode='same'
                )

        # Blend based on config strength
        result = (
            self.config.comb_filter_strength * frame_float +
            (1 - self.config.comb_filter_strength) * frame.astype(np.float32)
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _remove_rainbow(self, frame: Any) -> Any:
        """Remove rainbow effect artifact."""
        import numpy as np

        # Apply diagonal low-pass filter
        # Rainbow manifests as diagonal color stripes

        frame_float = frame.astype(np.float32)
        height, width = frame.shape[:2]

        # Simple diagonal smoothing
        result = frame_float.copy()

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Average with diagonal neighbors
                result[y, x] = (
                    0.5 * frame_float[y, x] +
                    0.125 * frame_float[y - 1, x - 1] +
                    0.125 * frame_float[y - 1, x + 1] +
                    0.125 * frame_float[y + 1, x - 1] +
                    0.125 * frame_float[y + 1, x + 1]
                )

        return np.clip(result, 0, 255).astype(np.uint8)

    def restore_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Restore entire video with VHS artifact removal."""
        if not self._ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

        # Build FFmpeg filter chain
        filters = self._build_ffmpeg_filters()

        # Construct command
        cmd = [
            self._ffmpeg_path,
            "-i", str(input_path),
            "-vf", filters,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "copy",
            "-y",
            str(output_path)
        ]

        logger.info(f"Running VHS restoration: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Monitor progress
            duration = self._get_duration(input_path)

            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break

                if "time=" in line and progress_callback and duration > 0:
                    import re
                    match = re.search(r'time=(\d+):(\d+):(\d+\.?\d*)', line)
                    if match:
                        h, m, s = match.groups()
                        current = int(h) * 3600 + int(m) * 60 + float(s)
                        progress_callback(current / duration)

            if process.returncode != 0:
                stderr = process.stderr.read()
                raise RuntimeError(f"FFmpeg failed: {stderr}")

            return output_path

        except Exception as e:
            logger.error(f"VHS restoration failed: {e}")
            raise

    def _build_ffmpeg_filters(self) -> str:
        """Build FFmpeg filter chain for VHS restoration."""
        filters = []

        # Time base correction (line stabilization)
        if self.config.enable_tbc_simulation:
            # Use temporal averaging for stabilization
            filters.append(f"tmix=frames=3:weights='1 2 1'")

        # Chroma noise reduction
        if self.config.chroma_noise_reduction > 0:
            strength = int(self.config.chroma_noise_reduction * 10)
            filters.append(f"hqdn3d=0:{strength}:{strength}:0")

        # Fix color bleeding with slight chroma delay
        if self.config.fix_color_bleeding:
            filters.append("colorchannelmixer=rr=1:rb=0.02:br=0.02:bb=1")

        # Remove head switching (crop bottom)
        if self.config.remove_head_switching:
            filters.append("crop=in_w:in_h-8:0:0")
            filters.append("pad=in_w:in_h+8:0:8")

        # Dot crawl removal (notch filter)
        if self.config.remove_dot_crawl:
            filters.append("pp=al")

        # Temporal noise reduction while preserving authentic character
        if self.config.temporal_noise_filter:
            strength = self.config.noise_reduction_strength
            if self.config.preserve_authentic_noise:
                strength *= 0.5  # Reduce strength to preserve character
            filters.append(f"hqdn3d={strength * 4}:{strength * 3}:{strength * 6}:{strength * 4.5}")

        # Luma sharpening (careful not to enhance noise)
        if self.config.restore_luma_bandwidth:
            filters.append("unsharp=3:3:0.5:3:3:0")

        # Color phase correction
        if self.config.correct_color_phase and self.config.color_phase_correction != 0:
            hue_shift = self.config.color_phase_correction
            filters.append(f"hue=h={hue_shift}")

        return ",".join(filters) if filters else "null"


def create_vhs_restorer(
    format_type: Optional[AnalogFormat] = None,
    aggressive: bool = False,
    preserve_character: bool = True
) -> VHSRestorer:
    """Factory function to create VHS restorer with common presets."""
    config = VHSRestorationConfig()

    if format_type:
        config.assumed_format = format_type
        config.auto_detect_format = False

        # Adjust settings based on format
        profile = ANALOG_PROFILES.get(format_type)
        if profile:
            # Formats with better quality need less aggressive restoration
            if profile.horizontal_resolution >= 400:
                config.noise_reduction_strength *= 0.7
                config.chroma_noise_reduction *= 0.7

    if aggressive:
        config.noise_reduction_strength = 0.6
        config.chroma_noise_reduction = 0.5
        config.preserve_authentic_noise = False
        config.max_enhancement = 0.8

    if preserve_character:
        config.preserve_authentic_noise = True
        config.preserve_format_character = True
        config.max_enhancement = 0.5
        config.noise_reduction_strength = min(config.noise_reduction_strength, 0.4)

    return VHSRestorer(config)
