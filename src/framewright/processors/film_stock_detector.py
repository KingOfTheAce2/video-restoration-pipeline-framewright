"""Film Stock Detection and Era-Specific Color Correction.

Identifies the likely film stock type and era based on color characteristics,
grain patterns, and degradation signatures. Applies stock-specific color
correction for more accurate restoration.

Supported film stocks:
- Kodachrome (1935-2009) - Distinctive warm tones, stable archival
- Ektachrome (1946-present) - Cooler, more neutral tones
- Technicolor - Vibrant, saturated colors
- Agfacolor - European film stock, distinctive color palette
- Fujifilm - Various stocks with different characteristics

Example:
    >>> detector = FilmStockDetector()
    >>> result = detector.analyze(video_path)
    >>> if result.detected_stock:
    ...     corrector = FilmStockCorrector(result.detected_stock)
    ...     corrector.apply_correction(frame)
"""

import logging
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


class FilmStock(Enum):
    """Known film stock types."""
    UNKNOWN = "unknown"

    # Kodak stocks
    KODACHROME = "kodachrome"  # 1935-2009, warm, stable
    EKTACHROME = "ektachrome"  # 1946+, cooler, reversal
    KODACOLOR = "kodacolor"  # Consumer negative
    VISION = "vision"  # Modern Kodak cinema

    # Technicolor processes
    TECHNICOLOR_2 = "technicolor_2"  # 2-strip, 1922-1936
    TECHNICOLOR_3 = "technicolor_3"  # 3-strip, 1932-1955
    TECHNICOLOR_IB = "technicolor_ib"  # IB/dye transfer

    # Other manufacturers
    AGFACOLOR = "agfacolor"  # German stock
    FUJIFILM = "fujifilm"  # Japanese stock
    EASTMANCOLOR = "eastmancolor"  # 1950s+, prone to fading

    # Era-based (when specific stock unknown)
    EARLY_COLOR = "early_color"  # Pre-1950
    CLASSIC_COLOR = "classic_color"  # 1950-1970
    MODERN_COLOR = "modern_color"  # 1980+

    # Black & white
    BW_ORTHOCHROMATIC = "bw_orthochromatic"  # Pre-1930
    BW_PANCHROMATIC = "bw_panchromatic"  # 1930+


class FilmEra(Enum):
    """Approximate era of film."""
    SILENT = "silent"  # Pre-1930
    EARLY_SOUND = "early_sound"  # 1930-1940
    GOLDEN_AGE = "golden_age"  # 1940-1960
    NEW_HOLLYWOOD = "new_hollywood"  # 1960-1980
    MODERN = "modern"  # 1980-2000
    DIGITAL_ERA = "digital_era"  # 2000+
    UNKNOWN = "unknown"


@dataclass
class ColorProfile:
    """Color characteristics of a film stock."""
    # Color cast (RGB shifts from neutral)
    red_shift: float = 0.0  # -1 to 1
    green_shift: float = 0.0
    blue_shift: float = 0.0

    # Saturation characteristics
    saturation_factor: float = 1.0
    saturation_hue_bias: Optional[str] = None  # "warm", "cool", None

    # Contrast
    contrast_factor: float = 1.0
    gamma: float = 1.0

    # Color fading pattern (for restoration)
    cyan_fade: float = 0.0  # How much cyan has likely faded
    magenta_fade: float = 0.0
    yellow_fade: float = 0.0

    # Grain characteristics
    grain_intensity: float = 0.5  # 0-1
    grain_size: float = 0.5  # 0-1, larger = coarser

    def to_correction_matrix(self) -> "np.ndarray":
        """Generate color correction matrix."""
        if not HAS_OPENCV:
            return None

        # Simple correction based on shifts
        matrix = np.eye(3, dtype=np.float32)

        # Apply inverse of typical color cast
        matrix[0, 0] = 1.0 - self.red_shift * 0.3
        matrix[1, 1] = 1.0 - self.green_shift * 0.3
        matrix[2, 2] = 1.0 - self.blue_shift * 0.3

        # Compensate for fading
        matrix[0, 0] += self.cyan_fade * 0.2
        matrix[1, 1] += self.magenta_fade * 0.2
        matrix[2, 2] += self.yellow_fade * 0.2

        return matrix


# Known film stock color profiles
STOCK_PROFILES: Dict[FilmStock, ColorProfile] = {
    FilmStock.KODACHROME: ColorProfile(
        red_shift=0.1, green_shift=-0.05, blue_shift=-0.1,
        saturation_factor=1.2, saturation_hue_bias="warm",
        contrast_factor=1.1, gamma=1.0,
        grain_intensity=0.3, grain_size=0.3,
    ),
    FilmStock.EKTACHROME: ColorProfile(
        red_shift=-0.05, green_shift=0.0, blue_shift=0.1,
        saturation_factor=1.0, saturation_hue_bias="cool",
        contrast_factor=1.05, gamma=1.0,
        cyan_fade=0.1, magenta_fade=0.05, yellow_fade=0.0,
        grain_intensity=0.4, grain_size=0.4,
    ),
    FilmStock.TECHNICOLOR_3: ColorProfile(
        red_shift=0.15, green_shift=0.05, blue_shift=-0.05,
        saturation_factor=1.4, saturation_hue_bias="warm",
        contrast_factor=1.2, gamma=0.95,
        grain_intensity=0.2, grain_size=0.2,
    ),
    FilmStock.EASTMANCOLOR: ColorProfile(
        red_shift=0.2, green_shift=-0.1, blue_shift=-0.15,
        saturation_factor=0.8,  # Faded
        contrast_factor=0.9,
        cyan_fade=0.3, magenta_fade=0.4, yellow_fade=0.1,
        grain_intensity=0.5, grain_size=0.5,
    ),
    FilmStock.AGFACOLOR: ColorProfile(
        red_shift=0.05, green_shift=0.1, blue_shift=-0.05,
        saturation_factor=1.1,
        contrast_factor=1.0,
        grain_intensity=0.4, grain_size=0.35,
    ),
    FilmStock.FUJIFILM: ColorProfile(
        red_shift=-0.05, green_shift=0.1, blue_shift=0.0,
        saturation_factor=1.05, saturation_hue_bias="cool",
        contrast_factor=1.0, gamma=1.0,
        grain_intensity=0.35, grain_size=0.3,
    ),
}


@dataclass
class FilmStockAnalysis:
    """Results of film stock detection."""
    detected_stock: FilmStock = FilmStock.UNKNOWN
    confidence: float = 0.0
    era: FilmEra = FilmEra.UNKNOWN

    # Color analysis
    dominant_hue: str = "neutral"  # "warm", "cool", "neutral"
    color_cast: Tuple[float, float, float] = (0, 0, 0)  # RGB shifts
    saturation_level: float = 1.0
    contrast_level: float = 1.0

    # Degradation analysis
    fading_detected: bool = False
    fading_pattern: str = "none"  # "cyan", "magenta", "yellow", "overall"
    fading_severity: float = 0.0  # 0-1

    # Grain analysis
    grain_detected: bool = False
    grain_intensity: float = 0.0
    grain_uniformity: float = 0.0

    # Profile for correction
    color_profile: Optional[ColorProfile] = None

    # Alternative matches
    alternative_stocks: List[Tuple[FilmStock, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Get human-readable summary."""
        stock_name = self.detected_stock.value.replace("_", " ").title()

        lines = [
            f"Detected: {stock_name} ({self.confidence*100:.0f}% confidence)",
            f"Era: {self.era.value.replace('_', ' ').title()}",
            f"Color cast: {self.dominant_hue}",
        ]

        if self.fading_detected:
            lines.append(f"Fading: {self.fading_pattern} ({self.fading_severity*100:.0f}%)")

        if self.grain_detected:
            lines.append(f"Grain: {self.grain_intensity*100:.0f}% intensity")

        if self.alternative_stocks:
            alts = [f"{s.value} ({c*100:.0f}%)" for s, c in self.alternative_stocks[:2]]
            lines.append(f"Alternatives: {', '.join(alts)}")

        return "\n".join(lines)


class FilmStockDetector:
    """Detects film stock type from video characteristics."""

    def __init__(self, sample_count: int = 30):
        """Initialize detector.

        Args:
            sample_count: Number of frames to sample
        """
        self.sample_count = sample_count

    def analyze(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> FilmStockAnalysis:
        """Analyze video to detect film stock.

        Args:
            video_path: Path to video
            progress_callback: Progress callback (0-1)

        Returns:
            FilmStockAnalysis
        """
        if not HAS_OPENCV:
            logger.error("OpenCV required for film stock detection")
            return FilmStockAnalysis()

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return FilmStockAnalysis()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // self.sample_count)

        color_samples = []
        grain_samples = []

        for i, frame_num in enumerate(range(0, total_frames, sample_interval)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            color_samples.append(self._analyze_color(frame))
            grain_samples.append(self._analyze_grain(frame))

            if progress_callback:
                progress_callback((i + 1) / self.sample_count)

        cap.release()

        if not color_samples:
            return FilmStockAnalysis()

        # Aggregate and match
        result = self._aggregate_and_match(color_samples, grain_samples)

        return result

    def _analyze_color(self, frame: "np.ndarray") -> Dict[str, float]:
        """Analyze color characteristics of frame."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Calculate channel means
        b, g, r = cv2.split(frame)
        h, s, v = cv2.split(hsv)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # Color cast (deviation from neutral)
        rgb_mean = np.mean([r.mean(), g.mean(), b.mean()])
        r_shift = (r.mean() - rgb_mean) / 255
        g_shift = (g.mean() - rgb_mean) / 255
        b_shift = (b.mean() - rgb_mean) / 255

        # Saturation
        saturation = s.mean() / 255

        # Contrast (standard deviation of luminance)
        contrast = l_ch.std() / 255

        # Hue distribution
        hue_mean = h.mean()

        return {
            "r_shift": r_shift,
            "g_shift": g_shift,
            "b_shift": b_shift,
            "saturation": saturation,
            "contrast": contrast,
            "hue_mean": hue_mean,
            "luminance_mean": l_ch.mean() / 255,
        }

    def _analyze_grain(self, frame: "np.ndarray") -> Dict[str, float]:
        """Analyze grain characteristics of frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # High-pass filter to isolate grain
        blurred = cv2.GaussianBlur(gray.astype(float), (5, 5), 0)
        grain = gray.astype(float) - blurred

        # Grain intensity
        intensity = np.std(grain) / 255

        # Grain uniformity (how consistent across frame)
        # Split into quadrants and compare
        h, w = gray.shape
        quadrants = [
            grain[:h//2, :w//2],
            grain[:h//2, w//2:],
            grain[h//2:, :w//2],
            grain[h//2:, w//2:],
        ]
        quad_stds = [np.std(q) for q in quadrants]
        uniformity = 1 - (np.std(quad_stds) / (np.mean(quad_stds) + 1e-6))

        # Grain size (via frequency analysis)
        fft = np.fft.fft2(grain)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # High frequency content indicates fine grain
        center = (h // 2, w // 2)
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        outer_mask = (x - center[1])**2 + (y - center[0])**2 > radius**2
        high_freq_ratio = magnitude[outer_mask].mean() / (magnitude.mean() + 1e-6)

        return {
            "intensity": intensity,
            "uniformity": uniformity,
            "size": 1 - min(1, high_freq_ratio),  # Inverse: high freq = fine grain
        }

    def _aggregate_and_match(
        self,
        color_samples: List[Dict[str, float]],
        grain_samples: List[Dict[str, float]],
    ) -> FilmStockAnalysis:
        """Aggregate samples and match to film stock."""
        result = FilmStockAnalysis()

        # Aggregate color data
        r_shift = np.mean([s["r_shift"] for s in color_samples])
        g_shift = np.mean([s["g_shift"] for s in color_samples])
        b_shift = np.mean([s["b_shift"] for s in color_samples])
        saturation = np.mean([s["saturation"] for s in color_samples])
        contrast = np.mean([s["contrast"] for s in color_samples])

        result.color_cast = (r_shift, g_shift, b_shift)
        result.saturation_level = saturation
        result.contrast_level = contrast

        # Determine dominant hue
        if r_shift > 0.02 and b_shift < -0.02:
            result.dominant_hue = "warm"
        elif b_shift > 0.02 and r_shift < -0.02:
            result.dominant_hue = "cool"
        else:
            result.dominant_hue = "neutral"

        # Aggregate grain data
        grain_intensity = np.mean([s["intensity"] for s in grain_samples])
        grain_uniformity = np.mean([s["uniformity"] for s in grain_samples])

        result.grain_detected = grain_intensity > 0.02
        result.grain_intensity = grain_intensity
        result.grain_uniformity = grain_uniformity

        # Detect fading
        self._detect_fading(result, color_samples)

        # Match to known stocks
        matches = self._match_stocks(result)
        if matches:
            result.detected_stock = matches[0][0]
            result.confidence = matches[0][1]
            result.alternative_stocks = matches[1:4]

            # Get color profile
            if result.detected_stock in STOCK_PROFILES:
                result.color_profile = STOCK_PROFILES[result.detected_stock]

        # Estimate era
        result.era = self._estimate_era(result)

        return result

    def _detect_fading(
        self,
        result: FilmStockAnalysis,
        color_samples: List[Dict[str, float]],
    ):
        """Detect color fading patterns."""
        r_shift, g_shift, b_shift = result.color_cast

        # Common fading patterns
        # Eastmancolor: loses cyan (shifts red), then magenta (shifts green)
        if r_shift > 0.1 and g_shift > 0:
            result.fading_detected = True
            result.fading_pattern = "cyan-magenta"
            result.fading_severity = min(1, (r_shift + g_shift) / 0.4)

        # General fading (desaturation)
        elif result.saturation_level < 0.3:
            result.fading_detected = True
            result.fading_pattern = "overall"
            result.fading_severity = 1 - result.saturation_level / 0.5

    def _match_stocks(
        self,
        result: FilmStockAnalysis,
    ) -> List[Tuple[FilmStock, float]]:
        """Match analysis to known film stocks."""
        matches = []

        for stock, profile in STOCK_PROFILES.items():
            score = 0.0
            factors = 0

            # Color cast match
            r_diff = abs(result.color_cast[0] - profile.red_shift)
            g_diff = abs(result.color_cast[1] - profile.green_shift)
            b_diff = abs(result.color_cast[2] - profile.blue_shift)
            color_match = 1 - min(1, (r_diff + g_diff + b_diff) / 0.6)
            score += color_match * 2  # Weight color heavily
            factors += 2

            # Saturation match
            sat_diff = abs(result.saturation_level - profile.saturation_factor * 0.5)
            sat_match = 1 - min(1, sat_diff / 0.3)
            score += sat_match
            factors += 1

            # Grain match
            if result.grain_detected:
                grain_diff = abs(result.grain_intensity - profile.grain_intensity * 0.1)
                grain_match = 1 - min(1, grain_diff / 0.05)
                score += grain_match
                factors += 1

            confidence = score / factors if factors > 0 else 0
            matches.append((stock, confidence))

        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def _estimate_era(self, result: FilmStockAnalysis) -> FilmEra:
        """Estimate film era from characteristics."""
        stock = result.detected_stock

        # Direct stock-based era estimation
        era_map = {
            FilmStock.TECHNICOLOR_2: FilmEra.EARLY_SOUND,
            FilmStock.TECHNICOLOR_3: FilmEra.GOLDEN_AGE,
            FilmStock.KODACHROME: FilmEra.GOLDEN_AGE,
            FilmStock.EASTMANCOLOR: FilmEra.NEW_HOLLYWOOD,
            FilmStock.EKTACHROME: FilmEra.MODERN,
            FilmStock.VISION: FilmEra.DIGITAL_ERA,
        }

        if stock in era_map:
            return era_map[stock]

        # Heuristic-based era estimation
        if result.fading_severity > 0.5:
            return FilmEra.GOLDEN_AGE
        elif result.grain_intensity > 0.05:
            return FilmEra.NEW_HOLLYWOOD
        elif result.saturation_level > 0.5:
            return FilmEra.MODERN

        return FilmEra.UNKNOWN


class FilmStockCorrector:
    """Apply film stock-specific color correction."""

    def __init__(self, stock: FilmStock):
        """Initialize corrector.

        Args:
            stock: Film stock to correct for
        """
        self.stock = stock
        self.profile = STOCK_PROFILES.get(stock, ColorProfile())

    def apply_correction(
        self,
        frame: "np.ndarray",
        strength: float = 1.0,
    ) -> "np.ndarray":
        """Apply color correction to frame.

        Args:
            frame: Input frame (BGR)
            strength: Correction strength (0-1)

        Returns:
            Corrected frame
        """
        if not HAS_OPENCV:
            return frame

        result = frame.astype(np.float32) / 255

        # Apply color matrix correction
        matrix = self.profile.to_correction_matrix()
        if matrix is not None:
            # Blend with identity based on strength
            identity = np.eye(3, dtype=np.float32)
            blend_matrix = identity + (matrix - identity) * strength

            # Apply to each pixel
            result = result.reshape(-1, 3) @ blend_matrix.T
            result = result.reshape(frame.shape)

        # Adjust saturation
        hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(float)
        target_sat = 1 / self.profile.saturation_factor if self.profile.saturation_factor > 0 else 1
        sat_adjustment = 1 + (target_sat - 1) * strength
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_adjustment, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(float) / 255

        # Adjust contrast
        if self.profile.contrast_factor != 1.0:
            target_contrast = 1 / self.profile.contrast_factor
            contrast_adj = 1 + (target_contrast - 1) * strength
            mean = result.mean()
            result = (result - mean) * contrast_adj + mean

        # Clip and convert back
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return result


def detect_film_stock(video_path: Path) -> FilmStockAnalysis:
    """Convenience function to detect film stock.

    Args:
        video_path: Path to video

    Returns:
        FilmStockAnalysis
    """
    detector = FilmStockDetector()
    return detector.analyze(video_path)


def get_correction_for_stock(stock: FilmStock) -> FilmStockCorrector:
    """Get corrector for a film stock.

    Args:
        stock: Film stock

    Returns:
        FilmStockCorrector
    """
    return FilmStockCorrector(stock)
