"""Film-specific restoration features for FrameWright.

Specialized processors for restoring film footage including:
- Film grain management (preserve/remove/match)
- Flicker reduction for old film
- Scratch and dust removal
- Sprocket hole stabilization
- Gate weave correction
- Color fade restoration
- Telecine pattern removal
"""

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FilmType(Enum):
    """Types of film stock."""
    SUPER_8 = "super8"
    STANDARD_8 = "8mm"
    MM_16 = "16mm"
    MM_35 = "35mm"
    MM_70 = "70mm"
    VHS = "vhs"
    BETAMAX = "betamax"
    UNKNOWN = "unknown"


class FilmEra(Enum):
    """Film era for restoration preset selection."""
    SILENT = "silent"  # Pre-1930
    EARLY_SOUND = "early_sound"  # 1930-1950
    GOLDEN_AGE = "golden_age"  # 1950-1970
    MODERN = "modern"  # 1970-2000
    DIGITAL_ERA = "digital"  # 2000+


@dataclass
class FilmCharacteristics:
    """Detected film characteristics."""
    film_type: FilmType = FilmType.UNKNOWN
    era: FilmEra = FilmEra.MODERN
    is_color: bool = True
    has_grain: bool = True
    grain_intensity: float = 0.0  # 0-1
    has_flicker: bool = False
    flicker_intensity: float = 0.0
    has_scratches: bool = False
    scratch_density: float = 0.0
    has_dust: bool = False
    dust_density: float = 0.0
    has_gate_weave: bool = False
    weave_intensity: float = 0.0
    color_fade: float = 0.0  # 0-1, amount of color fading
    frame_rate: float = 24.0
    aspect_ratio: str = "4:3"
    interlaced: bool = False
    telecine_pattern: Optional[str] = None  # "3:2", "2:2", etc.


class FilmAnalyzer:
    """Analyzes film footage to detect characteristics."""

    def __init__(self, sample_frames: int = 30):
        """Initialize analyzer.

        Args:
            sample_frames: Number of frames to sample for analysis
        """
        self.sample_frames = sample_frames
        self._cv2 = None
        self._np = None

    def _ensure_deps(self) -> bool:
        """Ensure OpenCV is available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            return False

    def analyze(self, video_path: Path) -> FilmCharacteristics:
        """Analyze video to detect film characteristics.

        Args:
            video_path: Path to video file

        Returns:
            Detected FilmCharacteristics
        """
        chars = FilmCharacteristics()

        if not self._ensure_deps():
            logger.warning("OpenCV not available, using basic analysis")
            return chars

        cv2 = self._cv2
        np = self._np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return chars

        # Get basic info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        chars.frame_rate = fps
        chars.aspect_ratio = self._detect_aspect_ratio(width, height)

        # Sample frames for analysis
        sample_indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)
        frames = []
        luminance_history = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luminance_history.append(np.mean(gray))

        cap.release()

        if not frames:
            return chars

        # Analyze characteristics
        chars.is_color = self._detect_color(frames)
        chars.has_grain, chars.grain_intensity = self._detect_grain(frames)
        chars.has_flicker, chars.flicker_intensity = self._detect_flicker(luminance_history)
        chars.has_scratches, chars.scratch_density = self._detect_scratches(frames)
        chars.has_dust, chars.dust_density = self._detect_dust(frames)
        chars.has_gate_weave, chars.weave_intensity = self._detect_gate_weave(frames)
        chars.color_fade = self._detect_color_fade(frames) if chars.is_color else 0.0
        chars.telecine_pattern = self._detect_telecine(frames, fps)
        chars.film_type = self._guess_film_type(width, height, chars)
        chars.era = self._guess_era(chars)

        logger.info(f"Film analysis complete: {chars.film_type.value}, era={chars.era.value}")
        return chars

    def _detect_aspect_ratio(self, width: int, height: int) -> str:
        """Detect aspect ratio from dimensions."""
        ratio = width / height if height > 0 else 1.33

        if abs(ratio - 1.33) < 0.05:
            return "4:3"
        elif abs(ratio - 1.37) < 0.05:
            return "Academy"
        elif abs(ratio - 1.66) < 0.05:
            return "5:3"
        elif abs(ratio - 1.78) < 0.05:
            return "16:9"
        elif abs(ratio - 1.85) < 0.05:
            return "1.85:1"
        elif abs(ratio - 2.35) < 0.1:
            return "Cinemascope"
        elif abs(ratio - 2.76) < 0.1:
            return "Ultra Panavision"
        else:
            return f"{ratio:.2f}:1"

    def _detect_color(self, frames: List) -> bool:
        """Detect if footage is color or B&W."""
        cv2 = self._cv2
        np = self._np

        saturation_values = []
        for frame in frames[:10]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation_values.append(np.mean(hsv[:, :, 1]))

        avg_saturation = np.mean(saturation_values)
        return avg_saturation > 20  # Low saturation = B&W

    def _detect_grain(self, frames: List) -> Tuple[bool, float]:
        """Detect film grain presence and intensity."""
        cv2 = self._cv2
        np = self._np

        noise_levels = []
        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Estimate noise using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise = laplacian.std()
            noise_levels.append(noise)

        avg_noise = np.mean(noise_levels)
        # Normalize to 0-1 range (typical grain is 5-30)
        intensity = min(1.0, max(0.0, (avg_noise - 5) / 25))
        has_grain = avg_noise > 8

        return has_grain, intensity

    def _detect_flicker(self, luminance_history: List[float]) -> Tuple[bool, float]:
        """Detect luminance flicker."""
        np = self._np

        if len(luminance_history) < 3:
            return False, 0.0

        # Calculate frame-to-frame variance
        lum = np.array(luminance_history)
        diff = np.abs(np.diff(lum))
        avg_diff = np.mean(diff)
        max_diff = np.max(diff)

        # Flicker is characterized by rapid luminance changes
        intensity = min(1.0, avg_diff / 20)
        has_flicker = max_diff > 15 or avg_diff > 5

        return has_flicker, intensity

    def _detect_scratches(self, frames: List) -> Tuple[bool, float]:
        """Detect vertical scratches on film."""
        cv2 = self._cv2
        np = self._np

        scratch_scores = []

        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect vertical lines using Sobel
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

            # Look for strong vertical gradients
            threshold = np.percentile(np.abs(sobel_x), 99)
            vertical_lines = np.abs(sobel_x) > threshold

            # Count continuous vertical structures
            col_sums = np.sum(vertical_lines, axis=0)
            potential_scratches = np.sum(col_sums > gray.shape[0] * 0.3)
            scratch_scores.append(potential_scratches)

        avg_scratches = np.mean(scratch_scores)
        density = min(1.0, avg_scratches / 10)
        has_scratches = avg_scratches > 2

        return has_scratches, density

    def _detect_dust(self, frames: List) -> Tuple[bool, float]:
        """Detect dust and dirt spots."""
        cv2 = self._cv2
        np = self._np

        spot_counts = []

        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply morphological operations to find spots
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

            # Threshold to find spots
            _, spots_bright = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
            _, spots_dark = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

            spots = cv2.bitwise_or(spots_bright, spots_dark)
            spot_count = np.sum(spots > 0) / (gray.shape[0] * gray.shape[1])
            spot_counts.append(spot_count)

        avg_density = np.mean(spot_counts)
        density = min(1.0, avg_density * 100)
        has_dust = avg_density > 0.001

        return has_dust, density

    def _detect_gate_weave(self, frames: List) -> Tuple[bool, float]:
        """Detect gate weave (frame instability)."""
        cv2 = self._cv2
        np = self._np

        if len(frames) < 3:
            return False, 0.0

        # Track motion between consecutive frames
        motions = []

        for i in range(min(len(frames) - 1, 10)):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            # Use phase correlation for sub-pixel motion estimation
            shift, _ = cv2.phaseCorrelate(
                gray1.astype(np.float64),
                gray2.astype(np.float64)
            )
            motions.append(np.sqrt(shift[0]**2 + shift[1]**2))

        avg_motion = np.mean(motions)
        intensity = min(1.0, avg_motion / 5)
        has_weave = avg_motion > 1.0

        return has_weave, intensity

    def _detect_color_fade(self, frames: List) -> float:
        """Detect amount of color fading."""
        cv2 = self._cv2
        np = self._np

        fade_scores = []

        for frame in frames[:10]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]

            # Color fade = low saturation + shifted hue towards yellow/magenta
            avg_sat = np.mean(saturation)
            sat_uniformity = 1 - (np.std(saturation) / 128)

            # Normalized fade score
            fade = 1 - (avg_sat / 128) * sat_uniformity
            fade_scores.append(fade)

        return np.mean(fade_scores)

    def _detect_telecine(self, frames: List, fps: float) -> Optional[str]:
        """Detect telecine pattern by analyzing frame differences."""
        cv2 = self._cv2
        np = self._np

        # Quick check based on FPS
        if abs(fps - 25.0) < 0.1 or abs(fps - 23.976) < 0.1:
            return None  # PAL or native film rate

        if len(frames) < 10:
            # Fall back to FPS-based detection
            if abs(fps - 29.97) < 0.1:
                return "3:2"
            return None

        # Analyze frame-to-frame differences to detect 3:2 pattern
        # In 3:2 pulldown, every 5 frames have pattern: A A B B C (or similar)
        # Duplicate frames have very low difference
        diffs = []

        for i in range(min(len(frames) - 1, 30)):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            # Calculate mean absolute difference
            diff = np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32)))
            diffs.append(diff)

        if not diffs:
            return None

        # Detect repeating pattern in differences
        diffs = np.array(diffs)
        mean_diff = np.mean(diffs)

        # Identify duplicate frames (very low difference)
        threshold = mean_diff * 0.3
        is_duplicate = diffs < threshold

        # Check for 3:2 pattern (2 duplicates per 5 frames = 40%)
        duplicate_ratio = np.sum(is_duplicate) / len(is_duplicate)

        if 0.35 < duplicate_ratio < 0.45:
            # Check for regular 5-frame pattern
            pattern_found = False
            for offset in range(5):
                # Check if duplicates occur at regular intervals
                expected_positions = list(range(offset, len(is_duplicate), 5))
                expected_positions += list(range((offset + 2) % 5, len(is_duplicate), 5))

                matches = sum(1 for p in expected_positions if p < len(is_duplicate) and is_duplicate[p])
                if matches > len(expected_positions) * 0.7:
                    pattern_found = True
                    break

            if pattern_found:
                return "3:2"

        # Check for 2:2 pattern (50% duplicates)
        if 0.45 < duplicate_ratio < 0.55:
            return "2:2"

        # Default based on FPS if no pattern detected
        if abs(fps - 29.97) < 0.1:
            return "3:2"

        return None

    def _guess_film_type(
        self,
        width: int,
        height: int,
        chars: FilmCharacteristics,
    ) -> FilmType:
        """Guess film type based on characteristics."""
        # Resolution-based guessing
        if width < 640:
            if chars.grain_intensity > 0.5:
                return FilmType.SUPER_8
            return FilmType.STANDARD_8
        elif width < 1280:
            return FilmType.MM_16
        elif width < 2048:
            return FilmType.MM_35
        else:
            return FilmType.MM_70

    def _guess_era(self, chars: FilmCharacteristics) -> FilmEra:
        """Guess film era based on characteristics."""
        if not chars.is_color:
            if chars.grain_intensity > 0.7:
                return FilmEra.SILENT
            return FilmEra.EARLY_SOUND

        if chars.color_fade > 0.5:
            return FilmEra.GOLDEN_AGE

        if chars.grain_intensity > 0.3:
            return FilmEra.MODERN

        return FilmEra.DIGITAL_ERA


class FilmGrainProcessor:
    """Handles film grain - preservation, removal, or matching."""

    class Mode(Enum):
        PRESERVE = "preserve"
        REMOVE = "remove"
        MATCH = "match"
        SYNTHESIZE = "synthesize"

    def __init__(
        self,
        mode: Mode = Mode.PRESERVE,
        grain_strength: float = 0.5,
        reference_frame: Optional[Path] = None,
    ):
        """Initialize grain processor.

        Args:
            mode: How to handle grain
            grain_strength: Strength for synthesis (0-1)
            reference_frame: Reference frame for grain matching
        """
        self.mode = mode
        self.grain_strength = grain_strength
        self.reference_frame = reference_frame

    def process(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Process film grain.

        Args:
            input_path: Input frame path
            output_path: Output frame path
            progress_callback: Progress callback

        Returns:
            Path to processed frame
        """
        if self.mode == self.Mode.REMOVE:
            return self._remove_grain(input_path, output_path)
        elif self.mode == self.Mode.PRESERVE:
            return self._preserve_grain(input_path, output_path)
        elif self.mode == self.Mode.MATCH:
            return self._match_grain(input_path, output_path)
        else:
            return self._synthesize_grain(input_path, output_path)

    def _remove_grain(self, input_path: Path, output_path: Path) -> Path:
        """Remove grain using temporal denoising."""
        # FFmpeg-based grain removal
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", "hqdn3d=4:3:6:4.5",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Grain removal failed: {e}")
            return input_path

        return output_path

    def _preserve_grain(self, input_path: Path, output_path: Path) -> Path:
        """Preserve original grain (minimal processing)."""
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

    def _match_grain(self, input_path: Path, output_path: Path) -> Path:
        """Match grain to reference frame by analyzing and replicating grain characteristics."""
        if self.reference_frame is None:
            logger.warning("No reference frame for grain matching, using synthesis")
            return self._synthesize_grain(input_path, output_path)

        try:
            import cv2
            import numpy as np

            # Load reference frame
            ref_frame = cv2.imread(str(self.reference_frame))
            if ref_frame is None:
                logger.warning("Could not load reference frame, using synthesis")
                return self._synthesize_grain(input_path, output_path)

            # Analyze grain in reference frame
            gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

            # Extract grain by subtracting blurred version
            blurred = cv2.GaussianBlur(gray_ref, (5, 5), 0)
            grain_pattern = cv2.subtract(gray_ref, blurred)

            # Calculate grain statistics
            grain_std = np.std(grain_pattern)
            grain_mean = np.mean(np.abs(grain_pattern))

            # Map grain intensity to FFmpeg noise parameter (0-100)
            # Typical grain std ranges from 2-15
            noise_strength = int(np.clip(grain_std * 2, 1, 30))

            # Analyze grain frequency characteristics using Laplacian
            laplacian = cv2.Laplacian(grain_pattern, cv2.CV_64F)
            high_freq_ratio = np.std(laplacian) / (grain_std + 0.001)

            # Determine grain type based on frequency analysis
            # High ratio = fine grain, low ratio = coarse grain
            if high_freq_ratio > 5:
                grain_type = "t"  # Temporal/fine grain
            else:
                grain_type = "u"  # Uniform/coarse grain

            # Build FFmpeg filter with matched parameters
            grain_filter = f"noise=alls={noise_strength}:allf={grain_type}"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", grain_filter,
                str(output_path)
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Grain matched: strength={noise_strength}, type={grain_type}")
            return output_path

        except ImportError:
            logger.warning("OpenCV not available for grain analysis, using synthesis")
            return self._synthesize_grain(input_path, output_path)
        except Exception as e:
            logger.error(f"Grain matching failed: {e}, falling back to synthesis")
            return self._synthesize_grain(input_path, output_path)

    def _synthesize_grain(self, input_path: Path, output_path: Path) -> Path:
        """Add synthetic film grain."""
        # FFmpeg grain synthesis
        grain_cmd = f"noise=alls={int(self.grain_strength * 20)}:allf=t+u"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", grain_cmd,
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Grain synthesis failed: {e}")
            return input_path

        return output_path


class FlickerRemover:
    """Removes luminance flicker from old film footage."""

    def __init__(
        self,
        strength: float = 1.0,
        temporal_window: int = 5,
        preserve_highlights: bool = True,
    ):
        """Initialize flicker remover.

        Args:
            strength: Correction strength (0-1)
            temporal_window: Number of frames for temporal analysis
            preserve_highlights: Preserve intentional lighting changes
        """
        self.strength = strength
        self.temporal_window = temporal_window
        self.preserve_highlights = preserve_highlights

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Remove flicker from video.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Progress callback

        Returns:
            Path to processed video
        """
        # FFmpeg deflicker filter
        deflicker_params = f"size={self.temporal_window}:mode=am"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"deflicker={deflicker_params}",
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Flicker removal complete: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Flicker removal failed: {e}")
            return input_path

        return output_path


class ScratchRemover:
    """Removes scratches and dust from film frames."""

    def __init__(
        self,
        sensitivity: float = 0.5,
        temporal_radius: int = 2,
        spatial_radius: int = 5,
    ):
        """Initialize scratch remover.

        Args:
            sensitivity: Detection sensitivity (0-1)
            temporal_radius: Frames before/after for reference
            spatial_radius: Inpainting radius
        """
        self.sensitivity = sensitivity
        self.temporal_radius = temporal_radius
        self.spatial_radius = spatial_radius
        self._cv2 = None
        self._np = None

    def _ensure_deps(self) -> bool:
        """Ensure OpenCV is available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            return False

    def detect_scratches(self, frame) -> Any:
        """Detect scratches in a frame.

        Args:
            frame: Input frame (numpy array)

        Returns:
            Binary mask of scratches
        """
        if not self._ensure_deps():
            return None

        cv2 = self._cv2
        np = self._np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect vertical lines (scratches)
        kernel = np.ones((15, 1), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Threshold based on sensitivity
        threshold = int(255 * (1 - self.sensitivity))
        _, mask = cv2.threshold(morph, threshold, 255, cv2.THRESH_BINARY)

        # Clean up mask
        kernel_clean = np.ones((3, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)

        return mask

    def remove_scratches(self, frame, mask) -> Any:
        """Remove scratches using inpainting.

        Args:
            frame: Input frame
            mask: Scratch mask

        Returns:
            Inpainted frame
        """
        cv2 = self._cv2

        # Use inpainting to fill scratches
        result = cv2.inpaint(frame, mask, self.spatial_radius, cv2.INPAINT_NS)
        return result

    def process_frame(self, frame) -> Any:
        """Process a single frame.

        Args:
            frame: Input frame

        Returns:
            Processed frame
        """
        mask = self.detect_scratches(frame)
        if mask is None:
            return frame

        return self.remove_scratches(frame, mask)


class GateWeaveCorrector:
    """Corrects gate weave (frame instability) in film footage."""

    def __init__(
        self,
        smoothing: float = 0.8,
        max_correction: int = 20,
        use_optical_flow: bool = True,
    ):
        """Initialize gate weave corrector.

        Args:
            smoothing: Smoothing factor for motion (0-1)
            max_correction: Maximum pixel correction
            use_optical_flow: Use optical flow for better tracking
        """
        self.smoothing = smoothing
        self.max_correction = max_correction
        self.use_optical_flow = use_optical_flow

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Stabilize video to correct gate weave.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Progress callback

        Returns:
            Path to stabilized video
        """
        # FFmpeg vidstab for stabilization
        stab_detect = input_path.parent / "transforms.trf"

        # Pass 1: Detect motion
        cmd1 = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={stab_detect}",
            "-f", "null", "-"
        ]

        # Pass 2: Apply stabilization
        smoothing_val = int(self.smoothing * 30)
        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"vidstabtransform=smoothing={smoothing_val}:maxshift={self.max_correction}:input={stab_detect}",
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd1, capture_output=True, check=True)
            subprocess.run(cmd2, capture_output=True, check=True)
            logger.info(f"Gate weave correction complete: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Gate weave correction failed: {e}")
            return input_path
        finally:
            if stab_detect.exists():
                stab_detect.unlink()

        return output_path


class ColorFadeRestorer:
    """Restores faded colors in aged film."""

    def __init__(
        self,
        auto_levels: bool = True,
        restore_saturation: float = 1.0,
        white_balance: bool = True,
        target_gamma: float = 1.0,
    ):
        """Initialize color fade restorer.

        Args:
            auto_levels: Automatically adjust levels
            restore_saturation: Saturation restoration amount (0-2)
            white_balance: Apply white balance correction
            target_gamma: Target gamma value
        """
        self.auto_levels = auto_levels
        self.restore_saturation = restore_saturation
        self.white_balance = white_balance
        self.target_gamma = target_gamma

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Restore color to faded video.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Progress callback

        Returns:
            Path to color-restored video
        """
        filters = []

        # Auto levels
        if self.auto_levels:
            filters.append("normalize=blackpt=black:whitept=white:smoothing=50")

        # White balance
        if self.white_balance:
            filters.append("colorbalance=rs=0:gs=0:bs=0.1")

        # Saturation restoration
        if self.restore_saturation != 1.0:
            filters.append(f"eq=saturation={self.restore_saturation}")

        # Gamma correction
        if self.target_gamma != 1.0:
            filters.append(f"eq=gamma={self.target_gamma}")

        filter_chain = ",".join(filters) if filters else "null"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_chain,
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Color restoration complete: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Color restoration failed: {e}")
            return input_path

        return output_path


class TelecineRemover:
    """Removes telecine patterns (inverse telecine / IVTC)."""

    def __init__(
        self,
        pattern: str = "auto",
        output_fps: float = 23.976,
    ):
        """Initialize telecine remover.

        Args:
            pattern: Telecine pattern (auto, 3:2, 2:2)
            output_fps: Target output framerate
        """
        self.pattern = pattern
        self.output_fps = output_fps

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Remove telecine and convert to progressive.

        Args:
            input_path: Input video path
            output_path: Output video path
            progress_callback: Progress callback

        Returns:
            Path to processed video
        """
        # FFmpeg inverse telecine
        if self.pattern == "auto":
            # Automatic field matching
            filter_str = "fieldmatch,yadif=deint=interlaced,decimate"
        elif self.pattern == "3:2":
            filter_str = "pullup,fps=24000/1001"
        else:
            filter_str = "yadif=mode=0,fps=24000/1001"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_str,
            "-r", str(self.output_fps),
            "-c:a", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Telecine removal complete: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Telecine removal failed: {e}")
            return input_path

        return output_path


@dataclass
class FilmRestorationConfig:
    """Configuration for film restoration."""
    # Film characteristics
    film_type: FilmType = FilmType.UNKNOWN
    era: FilmEra = FilmEra.MODERN

    # Grain handling
    grain_mode: str = "preserve"  # preserve, remove, match, synthesize
    grain_strength: float = 0.5

    # Flicker correction
    enable_deflicker: bool = True
    flicker_strength: float = 1.0

    # Scratch/dust removal
    enable_scratch_removal: bool = True
    scratch_sensitivity: float = 0.5

    # Stabilization
    enable_stabilization: bool = True
    stabilization_smoothing: float = 0.8

    # Color restoration
    enable_color_restoration: bool = True
    restore_saturation: float = 1.0

    # Telecine
    enable_ivtc: bool = False
    telecine_pattern: str = "auto"


class FilmRestorer:
    """Main film restoration orchestrator.

    Combines all film-specific processors into a unified pipeline.
    """

    def __init__(self, config: Optional[FilmRestorationConfig] = None):
        """Initialize film restorer.

        Args:
            config: Film restoration configuration
        """
        self.config = config or FilmRestorationConfig()
        self.analyzer = FilmAnalyzer()

    def restore(
        self,
        input_path: Path,
        output_path: Path,
        auto_detect: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Restore film footage.

        Args:
            input_path: Input video path
            output_path: Output video path
            auto_detect: Auto-detect film characteristics
            progress_callback: Progress callback

        Returns:
            Path to restored video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Starting film restoration: {input_path.name}")

        # Auto-detect characteristics
        if auto_detect:
            chars = self.analyzer.analyze(input_path)
            self._update_config_from_chars(chars)

        current_path = input_path
        temp_dir = output_path.parent / ".film_restore_temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # 1. Telecine removal (if applicable)
            if self.config.enable_ivtc:
                next_path = temp_dir / "step1_ivtc.mp4"
                remover = TelecineRemover(pattern=self.config.telecine_pattern)
                current_path = remover.process_video(current_path, next_path, progress_callback)

            # 2. Gate weave correction
            if self.config.enable_stabilization:
                next_path = temp_dir / "step2_stabilize.mp4"
                corrector = GateWeaveCorrector(smoothing=self.config.stabilization_smoothing)
                current_path = corrector.process_video(current_path, next_path, progress_callback)

            # 3. Flicker removal
            if self.config.enable_deflicker:
                next_path = temp_dir / "step3_deflicker.mp4"
                deflicker = FlickerRemover(strength=self.config.flicker_strength)
                current_path = deflicker.process_video(current_path, next_path, progress_callback)

            # 4. Color restoration
            if self.config.enable_color_restoration:
                next_path = temp_dir / "step4_color.mp4"
                color_restorer = ColorFadeRestorer(
                    restore_saturation=self.config.restore_saturation
                )
                current_path = color_restorer.process_video(current_path, next_path, progress_callback)

            # 5. Final copy to output
            import shutil
            shutil.copy(current_path, output_path)
            logger.info(f"Film restoration complete: {output_path}")

        finally:
            # Cleanup temp files
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return output_path

    def _update_config_from_chars(self, chars: FilmCharacteristics) -> None:
        """Update config based on detected characteristics."""
        self.config.film_type = chars.film_type
        self.config.era = chars.era

        # Enable deflicker if detected
        if chars.has_flicker:
            self.config.enable_deflicker = True
            self.config.flicker_strength = chars.flicker_intensity

        # Enable scratch removal if detected
        if chars.has_scratches or chars.has_dust:
            self.config.enable_scratch_removal = True
            self.config.scratch_sensitivity = max(chars.scratch_density, chars.dust_density)

        # Enable stabilization if gate weave detected
        if chars.has_gate_weave:
            self.config.enable_stabilization = True

        # Color restoration for faded film
        if chars.color_fade > 0.3:
            self.config.enable_color_restoration = True
            self.config.restore_saturation = 1.0 + chars.color_fade

        # IVTC for telecined footage
        if chars.telecine_pattern:
            self.config.enable_ivtc = True
            self.config.telecine_pattern = chars.telecine_pattern


def create_film_restorer(
    film_type: FilmType = FilmType.UNKNOWN,
    era: FilmEra = FilmEra.MODERN,
    grain_mode: str = "preserve",
    enable_deflicker: bool = True,
    enable_scratch_removal: bool = True,
    enable_stabilization: bool = True,
    enable_color_restoration: bool = True,
) -> FilmRestorer:
    """Factory function to create a FilmRestorer instance.

    Args:
        film_type: Type of film stock
        era: Film era for preset selection
        grain_mode: How to handle grain (preserve, remove, match)
        enable_deflicker: Enable flicker correction
        enable_scratch_removal: Enable scratch/dust removal
        enable_stabilization: Enable gate weave correction
        enable_color_restoration: Enable color fade restoration

    Returns:
        Configured FilmRestorer instance
    """
    config = FilmRestorationConfig(
        film_type=film_type,
        era=era,
        grain_mode=grain_mode,
        enable_deflicker=enable_deflicker,
        enable_scratch_removal=enable_scratch_removal,
        enable_stabilization=enable_stabilization,
        enable_color_restoration=enable_color_restoration,
    )
    return FilmRestorer(config)
