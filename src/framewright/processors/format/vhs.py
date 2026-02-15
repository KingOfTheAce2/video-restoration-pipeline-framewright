"""VHS format-specific processor for FrameWright.

Specialized restoration for VHS and analog tape formats:
- Tracking error correction for horizontal distortion
- Head switching noise removal at bottom of frame
- Chroma bleed reduction for color bleeding
- Rainbow artifact removal from composite video
- Dropout repair for white/black line artifacts

Example:
    >>> from pathlib import Path
    >>> from framewright.processors.format.vhs import VHSProcessor, VHSConfig
    >>> config = VHSConfig(tracking=0.8, head_switching=0.9, chroma_bleed=0.6)
    >>> processor = VHSProcessor(config)
    >>> artifacts = processor.detect_vhs_artifacts(frame)
    >>> restored = processor.process(frames)
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


class VHSQuality(Enum):
    """VHS recording quality modes."""
    SP = "sp"      # Standard Play (best quality)
    LP = "lp"      # Long Play
    EP = "ep"      # Extended Play (SLP)
    UNKNOWN = "unknown"

    @property
    def horizontal_resolution(self) -> int:
        """Get typical horizontal resolution in lines."""
        resolutions = {
            VHSQuality.SP: 240,
            VHSQuality.LP: 220,
            VHSQuality.EP: 200,
            VHSQuality.UNKNOWN: 240,
        }
        return resolutions.get(self, 240)


class ArtifactType(Enum):
    """Types of VHS artifacts."""
    HEAD_SWITCHING = "head_switching"
    TRACKING_ERROR = "tracking_error"
    DROPOUT = "dropout"
    CHROMA_BLEED = "chroma_bleed"
    RAINBOW = "rainbow"
    DOT_CRAWL = "dot_crawl"
    JITTER = "jitter"
    COLOR_PHASE = "color_phase"


@dataclass
class VHSConfig:
    """Configuration for VHS format processing.

    Attributes:
        tracking: Tracking error correction strength (0.0-1.0).
        head_switching: Head switching noise removal strength (0.0-1.0).
        chroma_bleed: Chroma bleed reduction strength (0.0-1.0).
        rainbow_removal: Rainbow artifact removal strength (0.0-1.0).
        dropout_repair: Dropout repair strength (0.0-1.0).
        dot_crawl_removal: Dot crawl removal strength (0.0-1.0).
        jitter_correction: Horizontal jitter (TBC) correction strength (0.0-1.0).
        color_phase_correction: Color phase correction in degrees.
        head_switch_height: Height of head switching region in pixels.
        dropout_min_length: Minimum dropout length in pixels to detect.
        temporal_radius: Frames to use for temporal repairs.
        preserve_authentic: Preserve some authentic VHS character.
        quality_mode: Assumed recording quality mode.
    """
    tracking: float = 0.5
    head_switching: float = 0.7
    chroma_bleed: float = 0.5
    rainbow_removal: float = 0.5
    dropout_repair: float = 0.6
    dot_crawl_removal: float = 0.5
    jitter_correction: float = 0.5
    color_phase_correction: float = 0.0
    head_switch_height: int = 16
    dropout_min_length: int = 5
    temporal_radius: int = 3
    preserve_authentic: bool = True
    quality_mode: VHSQuality = VHSQuality.UNKNOWN

    def __post_init__(self):
        """Validate configuration values."""
        for attr in ['tracking', 'head_switching', 'chroma_bleed', 'rainbow_removal',
                     'dropout_repair', 'dot_crawl_removal', 'jitter_correction']:
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0")


@dataclass
class VHSArtifactInfo:
    """Information about detected VHS artifacts in a frame."""
    artifact_type: ArtifactType
    severity: float = 0.0
    location: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    confidence: float = 0.0


@dataclass
class VHSAnalysis:
    """Results of VHS artifact analysis."""
    # Head switching
    head_switching_detected: bool = False
    head_switching_position: Optional[int] = None
    head_switching_severity: float = 0.0

    # Tracking
    tracking_errors: bool = False
    tracking_severity: float = 0.0
    tracking_line_positions: List[int] = field(default_factory=list)

    # Dropouts
    dropout_detected: bool = False
    dropout_count: int = 0
    dropout_positions: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Chroma
    chroma_bleed: bool = False
    chroma_bleed_severity: float = 0.0

    # Composite artifacts
    rainbow_effect: bool = False
    dot_crawl: bool = False

    # Jitter
    jitter_detected: bool = False
    jitter_severity: float = 0.0

    # Overall
    overall_degradation: float = 0.0
    detected_quality: VHSQuality = VHSQuality.UNKNOWN
    all_artifacts: List[VHSArtifactInfo] = field(default_factory=list)

    def summary(self) -> str:
        """Get human-readable summary."""
        issues = []
        if self.head_switching_detected:
            issues.append(f"Head switching ({self.head_switching_severity*100:.0f}%)")
        if self.tracking_errors:
            issues.append(f"Tracking errors ({self.tracking_severity*100:.0f}%)")
        if self.dropout_detected:
            issues.append(f"{self.dropout_count} dropouts")
        if self.chroma_bleed:
            issues.append(f"Chroma bleed ({self.chroma_bleed_severity*100:.0f}%)")
        if self.rainbow_effect:
            issues.append("Rainbow effect")
        if self.dot_crawl:
            issues.append("Dot crawl")
        if self.jitter_detected:
            issues.append(f"Jitter ({self.jitter_severity*100:.0f}%)")

        issue_str = ", ".join(issues) if issues else "No significant issues"

        return (
            f"VHS Quality: {self.detected_quality.value.upper()}\n"
            f"Overall degradation: {self.overall_degradation*100:.0f}%\n"
            f"Issues: {issue_str}"
        )


class VHSProcessor:
    """Main VHS format processor.

    Handles detection and correction of VHS-specific artifacts including
    head switching noise, tracking errors, dropouts, and chroma issues.
    """

    def __init__(self, config: Optional[VHSConfig] = None):
        """Initialize VHS processor.

        Args:
            config: VHS processing configuration.
        """
        self.config = config or VHSConfig()
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

    def detect_vhs_artifacts(self, frame: Any) -> VHSAnalysis:
        """Auto-detect VHS artifacts in a single frame.

        Args:
            frame: Frame as numpy array (BGR).

        Returns:
            VHSAnalysis with detected artifacts.
        """
        analysis = VHSAnalysis()

        if not HAS_OPENCV or frame is None:
            return analysis

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Detect head switching
        hs_detected, hs_position, hs_severity = self._detect_head_switching(gray, height)
        analysis.head_switching_detected = hs_detected
        analysis.head_switching_position = hs_position
        analysis.head_switching_severity = hs_severity

        if hs_detected:
            analysis.all_artifacts.append(VHSArtifactInfo(
                artifact_type=ArtifactType.HEAD_SWITCHING,
                severity=hs_severity,
                location=(0, hs_position or 0, width, self.config.head_switch_height),
                confidence=0.9
            ))

        # Detect tracking errors
        tr_detected, tr_severity, tr_positions = self._detect_tracking_errors(gray, height)
        analysis.tracking_errors = tr_detected
        analysis.tracking_severity = tr_severity
        analysis.tracking_line_positions = tr_positions

        if tr_detected:
            for pos in tr_positions:
                analysis.all_artifacts.append(VHSArtifactInfo(
                    artifact_type=ArtifactType.TRACKING_ERROR,
                    severity=tr_severity,
                    location=(0, pos, width, 1),
                    confidence=0.8
                ))

        # Detect dropouts
        do_detected, do_count, do_positions = self._detect_dropouts(gray, width, height)
        analysis.dropout_detected = do_detected
        analysis.dropout_count = do_count
        analysis.dropout_positions = do_positions

        for pos in do_positions:
            analysis.all_artifacts.append(VHSArtifactInfo(
                artifact_type=ArtifactType.DROPOUT,
                severity=0.8,
                location=pos,
                confidence=0.85
            ))

        # Detect chroma bleed
        if len(frame.shape) == 3:
            cb_detected, cb_severity = self._detect_chroma_bleed(frame)
            analysis.chroma_bleed = cb_detected
            analysis.chroma_bleed_severity = cb_severity

            if cb_detected:
                analysis.all_artifacts.append(VHSArtifactInfo(
                    artifact_type=ArtifactType.CHROMA_BLEED,
                    severity=cb_severity,
                    confidence=0.7
                ))

        # Detect rainbow effect
        if len(frame.shape) == 3:
            analysis.rainbow_effect = self._detect_rainbow(frame)
            if analysis.rainbow_effect:
                analysis.all_artifacts.append(VHSArtifactInfo(
                    artifact_type=ArtifactType.RAINBOW,
                    severity=0.6,
                    confidence=0.6
                ))

        # Detect dot crawl
        if len(frame.shape) == 3:
            analysis.dot_crawl = self._detect_dot_crawl(frame)
            if analysis.dot_crawl:
                analysis.all_artifacts.append(VHSArtifactInfo(
                    artifact_type=ArtifactType.DOT_CRAWL,
                    severity=0.5,
                    confidence=0.6
                ))

        # Detect jitter
        jt_detected, jt_severity = self._detect_jitter(gray, width)
        analysis.jitter_detected = jt_detected
        analysis.jitter_severity = jt_severity

        if jt_detected:
            analysis.all_artifacts.append(VHSArtifactInfo(
                artifact_type=ArtifactType.JITTER,
                severity=jt_severity,
                confidence=0.75
            ))

        # Calculate overall degradation
        analysis.overall_degradation = self._calculate_degradation(analysis)

        # Estimate quality mode
        analysis.detected_quality = self._estimate_quality(analysis, width)

        return analysis

    def fix_tracking_errors(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Fix horizontal tracking distortion.

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Corrected frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.tracking
        if strength <= 0:
            return frames

        result = []
        for i, frame in enumerate(frames):
            corrected = self._fix_tracking_single(frame, strength)
            result.append(corrected)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def remove_head_switching(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Remove head switching noise from bottom of frame.

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Corrected frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.head_switching
        if strength <= 0:
            return frames

        result = []
        for i, frame in enumerate(frames):
            corrected = self._remove_head_switching_single(frame, strength)
            result.append(corrected)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def reduce_chroma_bleed(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Reduce color bleeding artifacts.

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Corrected frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.chroma_bleed
        if strength <= 0:
            return frames

        result = []
        for i, frame in enumerate(frames):
            corrected = self._reduce_chroma_bleed_single(frame, strength)
            result.append(corrected)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def remove_rainbow_artifacts(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Remove rainbow artifacts from composite video.

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Corrected frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.rainbow_removal
        if strength <= 0:
            return frames

        result = []
        for i, frame in enumerate(frames):
            corrected = self._remove_rainbow_single(frame, strength)
            result.append(corrected)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def fix_dropout(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Fix white/black line dropouts.

        Uses temporal interpolation from adjacent frames when possible.

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Corrected frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.dropout_repair
        if strength <= 0:
            return frames

        result = []
        for i, frame in enumerate(frames):
            # Get adjacent frames for temporal repair
            prev_frames = frames[max(0, i - self.config.temporal_radius):i]
            next_frames = frames[i + 1:min(len(frames), i + 1 + self.config.temporal_radius)]

            corrected = self._fix_dropout_single(frame, prev_frames, next_frames, strength)
            result.append(corrected)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def process(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Apply full VHS restoration pipeline.

        Args:
            frames: List of frames to process.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Restored frames.
        """
        if not frames:
            return frames

        total_steps = 5
        current_step = 0

        def step_callback(progress: float):
            if progress_callback:
                overall = (current_step + progress) / total_steps
                progress_callback(overall)

        # Step 1: Head switching removal
        if self.config.head_switching > 0:
            frames = self.remove_head_switching(frames, progress_callback=step_callback)
        current_step += 1

        # Step 2: Tracking error correction
        if self.config.tracking > 0:
            frames = self.fix_tracking_errors(frames, progress_callback=step_callback)
        current_step += 1

        # Step 3: Dropout repair
        if self.config.dropout_repair > 0:
            frames = self.fix_dropout(frames, progress_callback=step_callback)
        current_step += 1

        # Step 4: Chroma bleed reduction
        if self.config.chroma_bleed > 0:
            frames = self.reduce_chroma_bleed(frames, progress_callback=step_callback)
        current_step += 1

        # Step 5: Rainbow artifact removal
        if self.config.rainbow_removal > 0:
            frames = self.remove_rainbow_artifacts(frames, progress_callback=step_callback)
        current_step += 1

        if progress_callback:
            progress_callback(1.0)

        return frames

    def _detect_head_switching(
        self,
        gray: Any,
        height: int
    ) -> Tuple[bool, Optional[int], float]:
        """Detect head switching noise at bottom of frame."""
        # Check bottom region for characteristic high-frequency noise
        bottom_region = gray[height - 30:, :]

        row_variances = np.var(np.diff(bottom_region.astype(np.float32), axis=1), axis=1)

        mean_var = np.mean(row_variances)
        threshold = mean_var * 2.5

        noisy_rows = np.where(row_variances > threshold)[0]

        if len(noisy_rows) > 2:
            position = height - 30 + int(np.min(noisy_rows))
            severity = min(1.0, len(noisy_rows) / 15.0)
            return True, position, severity

        return False, None, 0.0

    def _detect_tracking_errors(
        self,
        gray: Any,
        height: int
    ) -> Tuple[bool, float, List[int]]:
        """Detect horizontal tracking disturbances."""
        h_diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
        row_activity = np.mean(h_diff, axis=1)

        kernel_size = 5
        smoothed = np.convolve(row_activity, np.ones(kernel_size) / kernel_size, mode='same')

        local_deviation = np.abs(row_activity - smoothed)
        threshold = np.std(local_deviation) * 2.5

        tracking_lines = np.where(local_deviation > threshold)[0].tolist()

        # Filter out head switching region
        tracking_lines = [y for y in tracking_lines if y < height - 30]

        if len(tracking_lines) > 0:
            severity = min(1.0, len(tracking_lines) / 20.0)
            return True, severity, tracking_lines

        return False, 0.0, []

    def _detect_dropouts(
        self,
        gray: Any,
        width: int,
        height: int
    ) -> Tuple[bool, int, List[Tuple[int, int, int, int]]]:
        """Detect dropout artifacts (brief signal losses)."""
        dropouts = []
        min_length = self.config.dropout_min_length

        for y in range(height):
            row = gray[y, :]

            # Find runs of extreme values
            bright_mask = row > 250
            dark_mask = row < 5

            for mask in [bright_mask, dark_mask]:
                changes = np.diff(mask.astype(int))
                starts = np.where(changes == 1)[0] + 1
                ends = np.where(changes == -1)[0] + 1

                if mask[0]:
                    starts = np.concatenate([[0], starts])
                if mask[-1]:
                    ends = np.concatenate([ends, [width]])

                for start, end in zip(starts, ends):
                    length = end - start
                    if length >= min_length:
                        dropouts.append((int(start), y, int(length), 1))

        # Merge adjacent dropouts
        merged = self._merge_dropouts(dropouts)

        return len(merged) > 0, len(merged), merged

    def _merge_dropouts(
        self,
        dropouts: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge vertically adjacent dropout regions."""
        if not dropouts:
            return []

        sorted_dropouts = sorted(dropouts, key=lambda d: (d[1], d[0]))

        merged = []
        current = list(sorted_dropouts[0])

        for d in sorted_dropouts[1:]:
            if (d[1] <= current[1] + current[3] + 1 and
                d[0] < current[0] + current[2] and
                d[0] + d[2] > current[0]):
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
        if frame.shape[2] < 3:
            return False, 0.0

        y = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]

        edges_y = np.abs(np.diff(y.astype(np.float32), axis=1))
        edge_mask = edges_y > 30

        if not np.any(edge_mask):
            return False, 0.0

        r_diff = np.abs(np.diff(frame[:, :, 2].astype(np.float32), axis=1))
        b_diff = np.abs(np.diff(frame[:, :, 0].astype(np.float32), axis=1))

        edge_positions = np.where(edge_mask)

        if len(edge_positions[0]) < 10:
            return False, 0.0

        sample_indices = np.random.choice(
            len(edge_positions[0]),
            min(100, len(edge_positions[0])),
            replace=False
        )

        offsets = []
        for idx in sample_indices:
            y_pos, x_pos = edge_positions[0][idx], edge_positions[1][idx]

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

    def _detect_rainbow(self, frame: Any) -> bool:
        """Detect rainbow effect from composite video."""
        r, g, b = frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)

        sat_fft = np.fft.fft2(saturation)
        magnitude = np.abs(sat_fft)

        h, w = magnitude.shape
        diag_region = magnitude[h//4:h//2, w//4:w//2]

        mean_mag = np.mean(magnitude)
        diag_max = np.max(diag_region)

        return diag_max > mean_mag * 5

    def _detect_dot_crawl(self, frame: Any) -> bool:
        """Detect dot crawl artifact from composite video."""
        r = frame[:, :, 2].astype(np.float32)
        b = frame[:, :, 0].astype(np.float32)

        chroma = np.abs(r - b)
        h_diff = np.abs(np.diff(chroma, axis=1))
        row_means = np.mean(h_diff, axis=0)

        if len(row_means) < 10:
            return False

        fft = np.fft.fft(row_means)
        magnitude = np.abs(fft[1:len(fft)//2])

        if len(magnitude) == 0:
            return False

        peak = np.max(magnitude)
        mean = np.mean(magnitude)

        return peak > mean * 8

    def _detect_jitter(self, gray: Any, width: int) -> Tuple[bool, float]:
        """Detect horizontal line displacement (jitter)."""
        shifts = []

        for y in range(1, gray.shape[0] - 1, 5):
            row_curr = gray[y, :].astype(np.float32)
            row_prev = gray[y - 1, :].astype(np.float32)

            correlation = np.correlate(row_curr, row_prev, mode='same')
            shift = np.argmax(correlation) - width // 2
            shifts.append(shift)

        shifts = np.array(shifts)
        shift_variance = np.var(shifts)

        if shift_variance > 2.0:
            severity = min(1.0, shift_variance / 10.0)
            return True, severity

        return False, 0.0

    def _calculate_degradation(self, analysis: VHSAnalysis) -> float:
        """Calculate overall degradation score."""
        score = 0.0

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

        if analysis.tracking_errors:
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

    def _estimate_quality(self, analysis: VHSAnalysis, width: int) -> VHSQuality:
        """Estimate VHS recording quality mode."""
        if analysis.overall_degradation > 0.6:
            return VHSQuality.EP
        elif analysis.overall_degradation > 0.3:
            return VHSQuality.LP
        elif width >= 720:
            return VHSQuality.SP
        else:
            return VHSQuality.UNKNOWN

    def _fix_tracking_single(self, frame: Any, strength: float) -> Any:
        """Fix tracking errors in a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        _, _, tracking_lines = self._detect_tracking_errors(gray, frame.shape[0])

        if not tracking_lines:
            return frame

        result = frame.copy()
        for y in tracking_lines:
            if y > 0 and y < frame.shape[0] - 1:
                if len(frame.shape) == 3:
                    interpolated = ((frame[y - 1].astype(np.float32) +
                                   frame[y + 1].astype(np.float32)) / 2)
                    result[y] = (strength * interpolated +
                               (1 - strength) * frame[y].astype(np.float32)).astype(np.uint8)
                else:
                    interpolated = (frame[y - 1].astype(np.float32) +
                                  frame[y + 1].astype(np.float32)) / 2
                    result[y] = (strength * interpolated +
                               (1 - strength) * frame[y].astype(np.float32)).astype(np.uint8)

        return result

    def _remove_head_switching_single(self, frame: Any, strength: float) -> Any:
        """Remove head switching noise from a single frame."""
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        detected, position, _ = self._detect_head_switching(gray, height)

        if not detected or position is None:
            return frame

        result = frame.copy()
        blend_height = self.config.head_switch_height

        if position > blend_height:
            for y in range(position, min(height, position + blend_height)):
                blend_factor = strength * (1.0 - (y - position) / blend_height)
                source_y = position - blend_height + (y - position) % blend_height

                if source_y >= 0 and source_y < height:
                    if len(frame.shape) == 3:
                        result[y] = (blend_factor * frame[source_y].astype(np.float32) +
                                   (1 - blend_factor) * frame[y].astype(np.float32)).astype(np.uint8)
                    else:
                        result[y] = (blend_factor * frame[source_y].astype(np.float32) +
                                   (1 - blend_factor) * frame[y].astype(np.float32)).astype(np.uint8)

        return result

    def _reduce_chroma_bleed_single(self, frame: Any, strength: float) -> Any:
        """Reduce chroma bleed in a single frame."""
        if len(frame.shape) != 3 or frame.shape[2] < 3:
            return frame

        detected, severity = self._detect_chroma_bleed(frame)

        if not detected:
            return frame

        result = frame.copy()
        shift_amount = int(severity * 2 * strength)

        if shift_amount > 0:
            r = result[:, :, 2].copy()
            b = result[:, :, 0].copy()

            result[:, shift_amount:, 2] = r[:, :-shift_amount]
            result[:, :-shift_amount, 0] = b[:, shift_amount:]

        return result

    def _remove_rainbow_single(self, frame: Any, strength: float) -> Any:
        """Remove rainbow effect from a single frame."""
        if len(frame.shape) != 3:
            return frame

        frame_float = frame.astype(np.float32)
        height, width = frame.shape[:2]

        result = frame_float.copy()

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                result[y, x] = (
                    0.5 * frame_float[y, x] +
                    0.125 * frame_float[y - 1, x - 1] +
                    0.125 * frame_float[y - 1, x + 1] +
                    0.125 * frame_float[y + 1, x - 1] +
                    0.125 * frame_float[y + 1, x + 1]
                )

        blended = strength * result + (1 - strength) * frame_float
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _fix_dropout_single(
        self,
        frame: Any,
        prev_frames: List[Any],
        next_frames: List[Any],
        strength: float
    ) -> Any:
        """Fix dropouts in a single frame using temporal interpolation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        detected, _, positions = self._detect_dropouts(gray, frame.shape[1], frame.shape[0])

        if not detected:
            return frame

        result = frame.copy()

        for x, y, w, h in positions:
            x_end = min(x + w, frame.shape[1])
            y_end = min(y + h, frame.shape[0])

            # Try temporal interpolation first
            if prev_frames or next_frames:
                # Find clean data from adjacent frames
                clean_data = None
                for adj_frame in prev_frames + next_frames:
                    adj_region = adj_frame[y:y_end, x:x_end]
                    adj_gray = cv2.cvtColor(adj_region, cv2.COLOR_BGR2GRAY) if len(adj_region.shape) == 3 else adj_region

                    # Check if region is clean (not extreme values)
                    if np.mean(adj_gray) > 10 and np.mean(adj_gray) < 245:
                        clean_data = adj_region
                        break

                if clean_data is not None:
                    result[y:y_end, x:x_end] = (
                        strength * clean_data.astype(np.float32) +
                        (1 - strength) * result[y:y_end, x:x_end].astype(np.float32)
                    ).astype(np.uint8)
                    continue

            # Fall back to spatial interpolation
            if x > 0 and x_end < frame.shape[1]:
                for xi in range(x, x_end):
                    t = (xi - x + 1) / (w + 1)
                    left = result[y:y_end, x - 1:x]
                    right = result[y:y_end, x_end:x_end + 1]
                    interpolated = ((1 - t) * left + t * right)
                    result[y:y_end, xi:xi + 1] = (
                        strength * interpolated +
                        (1 - strength) * result[y:y_end, xi:xi + 1].astype(np.float32)
                    ).astype(np.uint8)

        return result


def create_vhs_processor(
    tracking: float = 0.5,
    head_switching: float = 0.7,
    chroma_bleed: float = 0.5,
    dropout_repair: float = 0.6,
    preserve_authentic: bool = True,
) -> VHSProcessor:
    """Factory function to create a VHSProcessor.

    Args:
        tracking: Tracking error correction strength.
        head_switching: Head switching removal strength.
        chroma_bleed: Chroma bleed reduction strength.
        dropout_repair: Dropout repair strength.
        preserve_authentic: Preserve authentic VHS character.

    Returns:
        Configured VHSProcessor.
    """
    config = VHSConfig(
        tracking=tracking,
        head_switching=head_switching,
        chroma_bleed=chroma_bleed,
        dropout_repair=dropout_repair,
        preserve_authentic=preserve_authentic,
    )

    return VHSProcessor(config)
