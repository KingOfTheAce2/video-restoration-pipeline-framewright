"""Film-specific format processor for FrameWright.

Specialized restoration for film footage with format-aware processing:
- Gate weave stabilization for sprocket wobble
- Flicker reduction for brightness variations
- Color fade restoration
- Film type detection (8mm, 16mm, 35mm)
- Era-specific artifact handling

Example:
    >>> from pathlib import Path
    >>> from framewright.processors.format.film import FilmProcessor, FilmConfig
    >>> config = FilmConfig(gate_weave=0.8, flicker=0.6, color_fade=0.5)
    >>> processor = FilmProcessor(config)
    >>> film_type = processor.detect_film_type(frames)
    >>> restored = processor.process(frames)
"""

import logging
import subprocess
import tempfile
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


class FilmFormat(Enum):
    """Film gauge/format types."""
    SUPER_8 = "super8"
    STANDARD_8 = "8mm"
    MM_16 = "16mm"
    SUPER_16 = "super16"
    MM_35 = "35mm"
    MM_65 = "65mm"
    MM_70 = "70mm"
    IMAX = "imax"
    UNKNOWN = "unknown"

    @property
    def typical_resolution(self) -> Tuple[int, int]:
        """Get typical scan resolution for this format."""
        resolutions = {
            FilmFormat.SUPER_8: (1440, 1080),
            FilmFormat.STANDARD_8: (1440, 1080),
            FilmFormat.MM_16: (2048, 1556),
            FilmFormat.SUPER_16: (2048, 1556),
            FilmFormat.MM_35: (4096, 3112),
            FilmFormat.MM_65: (6144, 4608),
            FilmFormat.MM_70: (6144, 4608),
            FilmFormat.IMAX: (8192, 6144),
            FilmFormat.UNKNOWN: (1920, 1080),
        }
        return resolutions.get(self, (1920, 1080))

    @property
    def frame_rate(self) -> float:
        """Get typical frame rate for this format."""
        rates = {
            FilmFormat.SUPER_8: 18.0,
            FilmFormat.STANDARD_8: 16.0,
            FilmFormat.MM_16: 24.0,
            FilmFormat.SUPER_16: 24.0,
            FilmFormat.MM_35: 24.0,
            FilmFormat.MM_65: 24.0,
            FilmFormat.MM_70: 24.0,
            FilmFormat.IMAX: 24.0,
            FilmFormat.UNKNOWN: 24.0,
        }
        return rates.get(self, 24.0)


class FilmEra(Enum):
    """Historical era for film restoration presets."""
    SILENT = "silent"           # Pre-1930
    EARLY_SOUND = "early_sound" # 1930-1950
    GOLDEN_AGE = "golden_age"   # 1950-1965
    NEW_HOLLYWOOD = "new_hollywood"  # 1965-1980
    MODERN = "modern"           # 1980-2000
    DIGITAL_ERA = "digital"     # 2000+


@dataclass
class FilmConfig:
    """Configuration for film format processing.

    Attributes:
        gate_weave: Gate weave correction strength (0.0-1.0).
        flicker: Flicker reduction strength (0.0-1.0).
        color_fade: Color fade restoration strength (0.0-1.0).
        grain_preserve: Amount of film grain to preserve (0.0-1.0).
        scratch_removal: Scratch detection/removal strength (0.0-1.0).
        dust_removal: Dust spot removal strength (0.0-1.0).
        stabilization_smoothing: Temporal smoothing for stabilization.
        auto_detect_format: Automatically detect film format.
        assumed_format: Format to use if auto-detection fails.
        era: Historical era for preset selection.
        max_correction: Maximum pixel displacement for corrections.
        temporal_window: Number of frames for temporal analysis.
    """
    gate_weave: float = 0.5
    flicker: float = 0.5
    color_fade: float = 0.5
    grain_preserve: float = 0.7
    scratch_removal: float = 0.5
    dust_removal: float = 0.5
    stabilization_smoothing: float = 0.8
    auto_detect_format: bool = True
    assumed_format: FilmFormat = FilmFormat.UNKNOWN
    era: FilmEra = FilmEra.MODERN
    max_correction: int = 20
    temporal_window: int = 5

    def __post_init__(self):
        """Validate configuration values."""
        for attr in ['gate_weave', 'flicker', 'color_fade', 'grain_preserve',
                     'scratch_removal', 'dust_removal', 'stabilization_smoothing']:
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0")


@dataclass
class FilmAnalysis:
    """Results of film format analysis."""
    detected_format: FilmFormat = FilmFormat.UNKNOWN
    detected_era: FilmEra = FilmEra.MODERN
    gate_weave_detected: bool = False
    gate_weave_intensity: float = 0.0
    flicker_detected: bool = False
    flicker_intensity: float = 0.0
    color_fade_amount: float = 0.0
    grain_intensity: float = 0.0
    scratch_density: float = 0.0
    dust_density: float = 0.0
    is_color: bool = True
    aspect_ratio: str = "4:3"
    frame_rate: float = 24.0
    confidence: float = 0.0

    def summary(self) -> str:
        """Get human-readable summary."""
        issues = []
        if self.gate_weave_detected:
            issues.append(f"Gate weave ({self.gate_weave_intensity*100:.0f}%)")
        if self.flicker_detected:
            issues.append(f"Flicker ({self.flicker_intensity*100:.0f}%)")
        if self.color_fade_amount > 0.2:
            issues.append(f"Color fade ({self.color_fade_amount*100:.0f}%)")
        if self.scratch_density > 0.1:
            issues.append(f"Scratches ({self.scratch_density*100:.0f}%)")
        if self.dust_density > 0.1:
            issues.append(f"Dust ({self.dust_density*100:.0f}%)")

        issue_str = ", ".join(issues) if issues else "No significant issues"
        color_str = "Color" if self.is_color else "Black & White"

        return (
            f"Format: {self.detected_format.value} ({color_str})\n"
            f"Era: {self.detected_era.value}\n"
            f"Issues: {issue_str}\n"
            f"Confidence: {self.confidence*100:.0f}%"
        )


class FilmProcessor:
    """Main film format processor.

    Handles detection and correction of film-specific artifacts including
    gate weave, flicker, color fading, and format detection.
    """

    def __init__(self, config: Optional[FilmConfig] = None):
        """Initialize film processor.

        Args:
            config: Film processing configuration.
        """
        self.config = config or FilmConfig()
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

    def analyze(
        self,
        frames: List[Any],
        video_path: Optional[Path] = None,
    ) -> FilmAnalysis:
        """Analyze frames for film characteristics.

        Args:
            frames: List of frames as numpy arrays.
            video_path: Optional path to video for metadata extraction.

        Returns:
            FilmAnalysis with detected characteristics.
        """
        analysis = FilmAnalysis()

        if not HAS_OPENCV or not frames:
            return analysis

        # Detect format from resolution
        if frames:
            height, width = frames[0].shape[:2]
            analysis.detected_format = self._detect_format_from_resolution(width, height)

        # Detect color vs B&W
        analysis.is_color = self._detect_color(frames[:min(10, len(frames))])

        # Detect gate weave
        weave_detected, weave_intensity = self._detect_gate_weave(frames)
        analysis.gate_weave_detected = weave_detected
        analysis.gate_weave_intensity = weave_intensity

        # Detect flicker
        flicker_detected, flicker_intensity = self._detect_flicker(frames)
        analysis.flicker_detected = flicker_detected
        analysis.flicker_intensity = flicker_intensity

        # Detect color fade
        if analysis.is_color:
            analysis.color_fade_amount = self._detect_color_fade(frames[:min(10, len(frames))])

        # Detect grain
        analysis.grain_intensity = self._detect_grain_intensity(frames[:min(10, len(frames))])

        # Detect scratches and dust
        analysis.scratch_density = self._detect_scratches(frames[:min(10, len(frames))])
        analysis.dust_density = self._detect_dust(frames[:min(10, len(frames))])

        # Estimate era
        analysis.detected_era = self._estimate_era(analysis)

        # Calculate confidence
        analysis.confidence = self._calculate_confidence(analysis)

        return analysis

    def detect_film_type(self, frames: List[Any]) -> FilmFormat:
        """Detect film format from frames.

        Args:
            frames: List of frames as numpy arrays.

        Returns:
            Detected FilmFormat.
        """
        if not frames or not HAS_OPENCV:
            return FilmFormat.UNKNOWN

        height, width = frames[0].shape[:2]
        return self._detect_format_from_resolution(width, height)

    def remove_gate_weave(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Stabilize sprocket wobble (gate weave).

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Stabilized frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.gate_weave
        if strength <= 0:
            return frames

        result = []
        ref_frame = None
        cumulative_shift = np.array([0.0, 0.0])
        smoothed_shift = np.array([0.0, 0.0])

        for i, frame in enumerate(frames):
            if ref_frame is None:
                ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                result.append(frame)
                if progress_callback:
                    progress_callback((i + 1) / len(frames))
                continue

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Calculate shift using phase correlation
            shift, _ = cv2.phaseCorrelate(
                ref_frame.astype(np.float64),
                curr_gray.astype(np.float64)
            )

            # Accumulate and smooth shift
            cumulative_shift += np.array(shift)
            alpha = 1.0 - self.config.stabilization_smoothing
            smoothed_shift = alpha * cumulative_shift + (1 - alpha) * smoothed_shift

            # Calculate correction
            correction = cumulative_shift - smoothed_shift
            correction *= strength

            # Limit maximum correction
            correction = np.clip(correction, -self.config.max_correction, self.config.max_correction)

            # Apply correction
            if np.abs(correction).max() > 0.5:
                M = np.float32([[1, 0, -correction[0]], [0, 1, -correction[1]]])
                corrected = cv2.warpAffine(
                    frame, M, (frame.shape[1], frame.shape[0]),
                    borderMode=cv2.BORDER_REPLICATE
                )
                result.append(corrected)
            else:
                result.append(frame)

            ref_frame = curr_gray

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def reduce_flicker(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Fix brightness variations (flicker).

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Deflickered frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.flicker
        if strength <= 0:
            return frames

        # Calculate luminance for each frame
        luminances = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            luminances.append(np.mean(gray))

        # Calculate smoothed target luminance
        window = max(3, self.config.temporal_window)
        kernel = np.ones(window) / window
        smoothed = np.convolve(luminances, kernel, mode='same')

        # Handle edge effects
        half_window = window // 2
        smoothed[:half_window] = smoothed[half_window]
        smoothed[-half_window:] = smoothed[-half_window - 1]

        result = []
        for i, frame in enumerate(frames):
            if luminances[i] > 0:
                # Calculate correction factor
                correction = smoothed[i] / luminances[i]
                correction = 1.0 + (correction - 1.0) * strength

                # Limit correction to reasonable range
                correction = np.clip(correction, 0.5, 2.0)

                # Apply correction
                if len(frame.shape) == 3:
                    # Convert to HSV, adjust V, convert back
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 2] *= correction
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                    corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                else:
                    corrected = np.clip(frame.astype(np.float32) * correction, 0, 255).astype(np.uint8)

                result.append(corrected)
            else:
                result.append(frame)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def restore_color(
        self,
        frames: List[Any],
        strength: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Fix color fading in aged film.

        Args:
            frames: List of frames to process.
            strength: Override config strength (0.0-1.0).
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Color-restored frames.
        """
        if not HAS_OPENCV or not frames:
            return frames

        strength = strength if strength is not None else self.config.color_fade
        if strength <= 0:
            return frames

        result = []
        for i, frame in enumerate(frames):
            if len(frame.shape) != 3:
                result.append(frame)
                if progress_callback:
                    progress_callback((i + 1) / len(frames))
                continue

            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Normalize L channel (luminance)
            l_channel = lab[:, :, 0]
            l_mean = np.mean(l_channel)
            l_std = np.std(l_channel)

            if l_std > 0:
                # Stretch contrast
                target_mean = 128.0
                target_std = 50.0
                l_normalized = (l_channel - l_mean) / l_std * target_std + target_mean
                l_normalized = l_normalized * strength + l_channel * (1 - strength)
                lab[:, :, 0] = np.clip(l_normalized, 0, 255)

            # Boost color channels (a and b)
            for c in [1, 2]:
                channel = lab[:, :, c]
                c_mean = np.mean(channel)
                c_std = np.std(channel)

                if c_std > 0:
                    # Increase saturation
                    boost_factor = 1.0 + strength * 0.5
                    boosted = (channel - 128) * boost_factor + 128
                    lab[:, :, c] = np.clip(boosted, 0, 255)

            # Convert back to BGR
            corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            result.append(corrected)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return result

    def process(
        self,
        frames: List[Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Any]:
        """Apply full film restoration pipeline.

        Args:
            frames: List of frames to process.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Restored frames.
        """
        if not frames:
            return frames

        total_steps = 3
        current_step = 0

        def step_callback(progress: float):
            if progress_callback:
                overall = (current_step + progress) / total_steps
                progress_callback(overall)

        # Step 1: Gate weave correction
        if self.config.gate_weave > 0:
            frames = self.remove_gate_weave(frames, progress_callback=step_callback)
        current_step += 1

        # Step 2: Flicker reduction
        if self.config.flicker > 0:
            frames = self.reduce_flicker(frames, progress_callback=step_callback)
        current_step += 1

        # Step 3: Color restoration
        if self.config.color_fade > 0:
            frames = self.restore_color(frames, progress_callback=step_callback)
        current_step += 1

        if progress_callback:
            progress_callback(1.0)

        return frames

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Process entire video file.

        Args:
            input_path: Input video path.
            output_path: Output video path.
            progress_callback: Progress callback (0.0-1.0).

        Returns:
            Path to processed video.
        """
        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg not available for video processing")

        # Build FFmpeg filter chain
        filters = []

        # Gate weave correction via vidstab
        if self.config.gate_weave > 0:
            smoothing = int(self.config.stabilization_smoothing * 30)
            max_shift = self.config.max_correction
            filters.append(f"vidstabtransform=smoothing={smoothing}:maxshift={max_shift}")

        # Flicker reduction via deflicker
        if self.config.flicker > 0:
            window = self.config.temporal_window
            filters.append(f"deflicker=size={window}:mode=am")

        # Color restoration via colorbalance and eq
        if self.config.color_fade > 0:
            sat_boost = 1.0 + self.config.color_fade * 0.5
            filters.append(f"eq=saturation={sat_boost}")
            filters.append("normalize=blackpt=black:whitept=white:smoothing=50")

        if not filters:
            filters.append("null")

        filter_chain = ",".join(filters)

        # Two-pass for stabilization
        if self.config.gate_weave > 0:
            # Pass 1: Detect motion
            with tempfile.NamedTemporaryFile(suffix=".trf", delete=False) as tf:
                trf_path = tf.name

            detect_cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={trf_path}",
                "-f", "null", "-"
            ]

            try:
                subprocess.run(detect_cmd, capture_output=True, check=True, timeout=3600)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Stabilization detection failed: {e}")
                # Fall back to non-stabilized processing
                filters = [f for f in filters if not f.startswith("vidstab")]
                filter_chain = ",".join(filters) if filters else "null"

        # Final processing pass
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=7200)
            logger.info(f"Film processing complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Film processing failed: {e}")
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")

    def _detect_format_from_resolution(self, width: int, height: int) -> FilmFormat:
        """Detect film format from frame resolution."""
        # Calculate megapixels
        megapixels = (width * height) / 1_000_000

        if megapixels < 2:
            # Low resolution, likely 8mm or VHS digitization
            return FilmFormat.SUPER_8
        elif megapixels < 4:
            # Medium resolution, likely 16mm
            return FilmFormat.MM_16
        elif megapixels < 15:
            # High resolution, likely 35mm
            return FilmFormat.MM_35
        elif megapixels < 30:
            # Very high resolution, 65mm/70mm
            return FilmFormat.MM_70
        else:
            # Ultra high resolution, IMAX
            return FilmFormat.IMAX

    def _detect_color(self, frames: List[Any]) -> bool:
        """Detect if footage is color or B&W."""
        if not frames or not HAS_OPENCV:
            return True

        saturation_values = []
        for frame in frames[:10]:
            if len(frame.shape) == 3:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                saturation_values.append(np.mean(hsv[:, :, 1]))
            else:
                saturation_values.append(0)

        avg_saturation = np.mean(saturation_values) if saturation_values else 0
        return avg_saturation > 20  # Low saturation indicates B&W

    def _detect_gate_weave(self, frames: List[Any]) -> Tuple[bool, float]:
        """Detect gate weave (frame instability)."""
        if not frames or len(frames) < 3 or not HAS_OPENCV:
            return False, 0.0

        motions = []
        for i in range(min(len(frames) - 1, 30)):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) if len(frames[i].shape) == 3 else frames[i]
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY) if len(frames[i + 1].shape) == 3 else frames[i + 1]

            shift, _ = cv2.phaseCorrelate(
                gray1.astype(np.float64),
                gray2.astype(np.float64)
            )
            motions.append(np.sqrt(shift[0]**2 + shift[1]**2))

        avg_motion = np.mean(motions) if motions else 0
        intensity = min(1.0, avg_motion / 5.0)
        detected = avg_motion > 1.0

        return detected, intensity

    def _detect_flicker(self, frames: List[Any]) -> Tuple[bool, float]:
        """Detect luminance flicker."""
        if not frames or len(frames) < 3 or not HAS_OPENCV:
            return False, 0.0

        luminances = []
        for frame in frames[:min(50, len(frames))]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            luminances.append(np.mean(gray))

        if len(luminances) < 3:
            return False, 0.0

        # Calculate frame-to-frame variance
        lum_array = np.array(luminances)
        diff = np.abs(np.diff(lum_array))
        avg_diff = np.mean(diff)
        max_diff = np.max(diff)

        intensity = min(1.0, avg_diff / 20)
        detected = max_diff > 15 or avg_diff > 5

        return detected, intensity

    def _detect_color_fade(self, frames: List[Any]) -> float:
        """Detect amount of color fading."""
        if not frames or not HAS_OPENCV:
            return 0.0

        fade_scores = []
        for frame in frames:
            if len(frame.shape) != 3:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]

            avg_sat = np.mean(saturation)
            sat_uniformity = 1 - (np.std(saturation) / 128)

            fade = 1 - (avg_sat / 128) * sat_uniformity
            fade_scores.append(max(0, min(1, fade)))

        return np.mean(fade_scores) if fade_scores else 0.0

    def _detect_grain_intensity(self, frames: List[Any]) -> float:
        """Detect film grain intensity."""
        if not frames or not HAS_OPENCV:
            return 0.0

        noise_levels = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise = laplacian.std()
            noise_levels.append(noise)

        avg_noise = np.mean(noise_levels) if noise_levels else 0
        return min(1.0, max(0.0, (avg_noise - 5) / 25))

    def _detect_scratches(self, frames: List[Any]) -> float:
        """Detect vertical scratch density."""
        if not frames or not HAS_OPENCV:
            return 0.0

        scratch_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            threshold = np.percentile(np.abs(sobel_x), 99)
            vertical_lines = np.abs(sobel_x) > threshold

            col_sums = np.sum(vertical_lines, axis=0)
            potential_scratches = np.sum(col_sums > gray.shape[0] * 0.3)
            scratch_scores.append(potential_scratches)

        avg_scratches = np.mean(scratch_scores) if scratch_scores else 0
        return min(1.0, avg_scratches / 10)

    def _detect_dust(self, frames: List[Any]) -> float:
        """Detect dust and dirt spots."""
        if not frames or not HAS_OPENCV:
            return 0.0

        spot_counts = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

            _, spots_bright = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
            _, spots_dark = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

            spots = cv2.bitwise_or(spots_bright, spots_dark)
            spot_count = np.sum(spots > 0) / (gray.shape[0] * gray.shape[1])
            spot_counts.append(spot_count)

        avg_density = np.mean(spot_counts) if spot_counts else 0
        return min(1.0, avg_density * 100)

    def _estimate_era(self, analysis: FilmAnalysis) -> FilmEra:
        """Estimate film era from characteristics."""
        if not analysis.is_color:
            if analysis.grain_intensity > 0.7:
                return FilmEra.SILENT
            return FilmEra.EARLY_SOUND

        if analysis.color_fade_amount > 0.5:
            return FilmEra.GOLDEN_AGE

        if analysis.grain_intensity > 0.4:
            return FilmEra.NEW_HOLLYWOOD

        if analysis.grain_intensity > 0.2:
            return FilmEra.MODERN

        return FilmEra.DIGITAL_ERA

    def _calculate_confidence(self, analysis: FilmAnalysis) -> float:
        """Calculate analysis confidence score."""
        confidence = 0.5  # Base confidence

        # Higher confidence if clear format detection
        if analysis.detected_format != FilmFormat.UNKNOWN:
            confidence += 0.2

        # Higher confidence if characteristics are clear
        if analysis.gate_weave_detected or analysis.flicker_detected:
            confidence += 0.1

        if analysis.grain_intensity > 0.1:
            confidence += 0.1

        if not analysis.is_color or analysis.color_fade_amount > 0.2:
            confidence += 0.1

        return min(1.0, confidence)


def create_film_processor(
    gate_weave: float = 0.5,
    flicker: float = 0.5,
    color_fade: float = 0.5,
    grain_preserve: float = 0.7,
    era: Optional[str] = None,
) -> FilmProcessor:
    """Factory function to create a FilmProcessor.

    Args:
        gate_weave: Gate weave correction strength.
        flicker: Flicker reduction strength.
        color_fade: Color fade restoration strength.
        grain_preserve: Amount of grain to preserve.
        era: Film era preset name.

    Returns:
        Configured FilmProcessor.
    """
    film_era = FilmEra.MODERN
    if era:
        try:
            film_era = FilmEra(era.lower())
        except ValueError:
            pass

    config = FilmConfig(
        gate_weave=gate_weave,
        flicker=flicker,
        color_fade=color_fade,
        grain_preserve=grain_preserve,
        era=film_era,
    )

    return FilmProcessor(config)
