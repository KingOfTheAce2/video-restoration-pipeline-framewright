"""Film Grain Manager for Extraction, Preservation, and Restoration.

This module provides comprehensive film grain management that:
- Extracts grain profile from original footage
- Separates grain from content using frequency domain analysis
- Removes grain while preserving fine detail
- Restores authentic grain after processing
- Synthesizes matching grain for processed frames

Grain Analysis Techniques:
- FFT-based frequency analysis for grain detection
- Wavelet decomposition for grain/detail separation
- Perlin noise generation for film grain synthesis
- Film stock LUTs for era-accurate grain patterns

VRAM Requirements:
- CPU fallback always available
- GPU acceleration: ~512MB VRAM for wavelet processing

Example:
    >>> from framewright.processors.restoration import GrainManager, GrainConfig
    >>>
    >>> config = GrainConfig(preserve_grain=True, grain_opacity=0.35)
    >>> manager = GrainManager(config)
    >>>
    >>> # Extract profile from original footage
    >>> profile = manager.extract_profile(original_frames)
    >>>
    >>> # Process with grain preservation
    >>> def my_processor(frames):
    ...     return [denoise(f) for f in frames]
    >>>
    >>> restored = manager.process_with_preservation(frames, my_processor)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available")

try:
    from scipy import ndimage, fft, signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("SciPy not available - grain analysis will be limited")

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    logger.debug("PyWavelets not available - using FFT fallback for grain separation")


class FilmStockGrainType(Enum):
    """Known film stock grain patterns."""
    UNKNOWN = "unknown"

    # Kodak stocks
    KODAK_5219 = "kodak_5219"      # Vision3 500T - medium grain
    KODAK_5213 = "kodak_5213"      # Vision3 200T - fine grain
    KODAK_5207 = "kodak_5207"      # Vision3 250D - fine grain daylight
    KODAK_5222 = "kodak_5222"      # Double-X B&W - distinctive grain

    # Fuji stocks
    FUJI_ETERNA = "fuji_eterna"    # Eterna 500 - fine grain
    FUJI_REALA = "fuji_reala"      # Consumer fine grain

    # Vintage stocks
    KODACHROME = "kodachrome"       # Very fine grain, archival
    EKTACHROME = "ektachrome"       # Medium grain, reversal
    AGFA = "agfa"                   # European stock, coarse grain

    # Era-based (when specific stock unknown)
    FINE = "fine"                   # Modern fine grain
    MEDIUM = "medium"               # Standard grain
    COARSE = "coarse"               # Vintage/high ISO grain


@dataclass
class GrainProfile:
    """Extracted grain characteristics from source footage.

    Attributes:
        grain_size: Average grain particle size (0-1 scale, 0=fine, 1=coarse)
        grain_intensity: Overall grain strength (0-1 scale)
        luminance_grain: Luma channel grain strength (0-1)
        chroma_grain: Chroma channel grain strength (0-1)
        frequency_profile: Frequency distribution array (power spectrum)
        temporal_variation: Frame-to-frame variation coefficient (0-1)
        film_stock_estimate: Estimated film stock type
        grain_uniformity: How uniform grain is across frame (0-1, 1=uniform)
        contrast_dependent: True if grain varies with brightness
        sample_grain_patches: Extracted grain texture samples
    """
    grain_size: float = 0.5
    grain_intensity: float = 0.3
    luminance_grain: float = 0.3
    chroma_grain: float = 0.15
    frequency_profile: np.ndarray = field(default_factory=lambda: np.zeros(64))
    temporal_variation: float = 0.1
    film_stock_estimate: FilmStockGrainType = FilmStockGrainType.UNKNOWN
    grain_uniformity: float = 0.8
    contrast_dependent: bool = True
    sample_grain_patches: List[np.ndarray] = field(default_factory=list)

    # Additional analysis data
    median_frequency: float = 0.0
    peak_frequency: float = 0.0
    frames_analyzed: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "grain_size": self.grain_size,
            "grain_intensity": self.grain_intensity,
            "luminance_grain": self.luminance_grain,
            "chroma_grain": self.chroma_grain,
            "temporal_variation": self.temporal_variation,
            "film_stock_estimate": self.film_stock_estimate.value,
            "grain_uniformity": self.grain_uniformity,
            "contrast_dependent": self.contrast_dependent,
            "median_frequency": self.median_frequency,
            "peak_frequency": self.peak_frequency,
            "frames_analyzed": self.frames_analyzed,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrainProfile":
        """Create from dictionary."""
        profile = cls()
        profile.grain_size = data.get("grain_size", 0.5)
        profile.grain_intensity = data.get("grain_intensity", 0.3)
        profile.luminance_grain = data.get("luminance_grain", 0.3)
        profile.chroma_grain = data.get("chroma_grain", 0.15)
        profile.temporal_variation = data.get("temporal_variation", 0.1)
        profile.film_stock_estimate = FilmStockGrainType(
            data.get("film_stock_estimate", "unknown")
        )
        profile.grain_uniformity = data.get("grain_uniformity", 0.8)
        profile.contrast_dependent = data.get("contrast_dependent", True)
        profile.median_frequency = data.get("median_frequency", 0.0)
        profile.peak_frequency = data.get("peak_frequency", 0.0)
        profile.frames_analyzed = data.get("frames_analyzed", 0)
        profile.confidence = data.get("confidence", 0.0)
        return profile


@dataclass
class GrainConfig:
    """Configuration for grain management.

    Attributes:
        preserve_grain: Whether to extract and re-apply grain after processing
        grain_opacity: Opacity for restored grain (0-1, default 0.35)
        extract_sample_count: Number of frames to sample for profile extraction
        wavelet_levels: Decomposition levels for wavelet analysis
        frequency_bands: Number of frequency bands for analysis
        temporal_coherence: Enable temporal grain coherence
        grain_seed: Random seed for reproducible grain synthesis
        match_luminance_response: Match grain to brightness levels
        chroma_grain_ratio: Ratio of chroma to luma grain (default 0.5)
        gpu_accelerate: Use GPU for processing if available
        film_stock_override: Force specific film stock grain pattern
    """
    preserve_grain: bool = True
    grain_opacity: float = 0.35
    extract_sample_count: int = 30
    wavelet_levels: int = 4
    frequency_bands: int = 64
    temporal_coherence: bool = True
    grain_seed: Optional[int] = None
    match_luminance_response: bool = True
    chroma_grain_ratio: float = 0.5
    gpu_accelerate: bool = True
    film_stock_override: Optional[FilmStockGrainType] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.grain_opacity <= 1.0:
            raise ValueError(f"grain_opacity must be 0-1, got {self.grain_opacity}")
        if not 0.0 <= self.chroma_grain_ratio <= 1.0:
            raise ValueError(f"chroma_grain_ratio must be 0-1, got {self.chroma_grain_ratio}")
        if self.wavelet_levels < 1 or self.wavelet_levels > 8:
            raise ValueError(f"wavelet_levels must be 1-8, got {self.wavelet_levels}")


@dataclass
class GrainRemovalResult:
    """Result of grain removal operation.

    Attributes:
        clean_frame: Frame with grain removed
        grain_layer: Extracted grain layer
        removal_strength: Effective removal strength applied
        detail_preserved: Estimated detail preservation (0-1)
    """
    clean_frame: np.ndarray
    grain_layer: np.ndarray
    removal_strength: float = 1.0
    detail_preserved: float = 0.95


# =============================================================================
# Film Stock Grain Patterns
# =============================================================================

# Predefined grain characteristics for known film stocks
FILM_STOCK_PROFILES: Dict[FilmStockGrainType, Dict[str, float]] = {
    FilmStockGrainType.KODAK_5219: {
        "size": 0.45,
        "intensity": 0.35,
        "luma_ratio": 0.7,
        "uniformity": 0.85,
        "temporal_var": 0.15,
    },
    FilmStockGrainType.KODAK_5213: {
        "size": 0.3,
        "intensity": 0.2,
        "luma_ratio": 0.8,
        "uniformity": 0.9,
        "temporal_var": 0.1,
    },
    FilmStockGrainType.KODAK_5222: {
        "size": 0.55,
        "intensity": 0.45,
        "luma_ratio": 1.0,  # B&W
        "uniformity": 0.75,
        "temporal_var": 0.2,
    },
    FilmStockGrainType.FUJI_ETERNA: {
        "size": 0.35,
        "intensity": 0.25,
        "luma_ratio": 0.75,
        "uniformity": 0.88,
        "temporal_var": 0.12,
    },
    FilmStockGrainType.KODACHROME: {
        "size": 0.25,
        "intensity": 0.15,
        "luma_ratio": 0.85,
        "uniformity": 0.92,
        "temporal_var": 0.08,
    },
    FilmStockGrainType.EKTACHROME: {
        "size": 0.4,
        "intensity": 0.3,
        "luma_ratio": 0.7,
        "uniformity": 0.82,
        "temporal_var": 0.18,
    },
    FilmStockGrainType.AGFA: {
        "size": 0.6,
        "intensity": 0.5,
        "luma_ratio": 0.65,
        "uniformity": 0.7,
        "temporal_var": 0.25,
    },
    FilmStockGrainType.FINE: {
        "size": 0.25,
        "intensity": 0.15,
        "luma_ratio": 0.8,
        "uniformity": 0.9,
        "temporal_var": 0.1,
    },
    FilmStockGrainType.MEDIUM: {
        "size": 0.45,
        "intensity": 0.35,
        "luma_ratio": 0.7,
        "uniformity": 0.8,
        "temporal_var": 0.15,
    },
    FilmStockGrainType.COARSE: {
        "size": 0.7,
        "intensity": 0.55,
        "luma_ratio": 0.6,
        "uniformity": 0.65,
        "temporal_var": 0.25,
    },
}


# =============================================================================
# Grain Manager Class
# =============================================================================

class GrainManager:
    """Film grain extraction, removal, and restoration manager.

    Provides a complete pipeline for handling film grain during video
    restoration, ensuring authentic grain is preserved or convincingly
    re-applied after processing.

    Key Features:
    - FFT-based frequency analysis for grain detection
    - Wavelet decomposition for grain/detail separation
    - Perlin noise synthesis for film grain generation
    - Temporal coherence for video grain
    - Film stock-specific grain patterns

    Example:
        >>> config = GrainConfig(preserve_grain=True, grain_opacity=0.35)
        >>> manager = GrainManager(config)
        >>>
        >>> # Extract grain profile
        >>> profile = manager.extract_profile(source_frames)
        >>>
        >>> # Remove grain for processing
        >>> clean, grain_layer = manager.remove_grain(frame, profile)
        >>>
        >>> # After processing, restore grain
        >>> restored = manager.restore_grain(processed_frame, profile, opacity=0.35)
    """

    def __init__(self, config: Optional[GrainConfig] = None):
        """Initialize grain manager.

        Args:
            config: Grain configuration (uses defaults if None)
        """
        self.config = config or GrainConfig()
        self._rng = np.random.RandomState(self.config.grain_seed)
        self._perlin_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._grain_texture_cache: List[np.ndarray] = []

        if not HAS_OPENCV:
            logger.warning("OpenCV not available - some features will be limited")
        if not HAS_SCIPY:
            logger.warning("SciPy not available - using simplified grain analysis")

    # =========================================================================
    # Profile Extraction
    # =========================================================================

    def extract_profile(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> GrainProfile:
        """Analyze frames and extract grain characteristics.

        Performs comprehensive grain analysis including:
        - Frequency domain analysis (FFT)
        - Spatial grain distribution
        - Luminance vs chroma grain separation
        - Temporal variation measurement
        - Film stock identification

        Args:
            frames: List of frames to analyze (BGR format)
            progress_callback: Optional progress callback (0-1)

        Returns:
            GrainProfile with extracted characteristics
        """
        profile = GrainProfile()

        if not frames:
            logger.warning("No frames provided for grain extraction")
            return profile

        # Sample frames if too many
        sample_indices = self._get_sample_indices(len(frames))
        sampled_frames = [frames[i] for i in sample_indices]

        logger.info(f"Extracting grain profile from {len(sampled_frames)} frames")

        # Collect per-frame measurements
        frame_measurements: List[Dict[str, float]] = []
        grain_patches: List[np.ndarray] = []
        frequency_profiles: List[np.ndarray] = []
        prev_frame = None

        for i, frame in enumerate(sampled_frames):
            try:
                # Analyze frame
                measurements = self._analyze_frame_grain(frame)
                frame_measurements.append(measurements)

                # Extract frequency profile
                freq_profile = self._extract_frequency_profile(frame)
                frequency_profiles.append(freq_profile)

                # Extract grain patches for texture synthesis
                patches = self._extract_grain_patches(frame)
                grain_patches.extend(patches)

                # Temporal variation (if previous frame available)
                if prev_frame is not None:
                    temporal_var = self._measure_temporal_variation(prev_frame, frame)
                    measurements["temporal_var"] = temporal_var

                prev_frame = frame

            except Exception as e:
                logger.warning(f"Failed to analyze frame {i}: {e}")
                continue

            if progress_callback:
                progress_callback((i + 1) / len(sampled_frames))

        if not frame_measurements:
            logger.error("No frames could be analyzed")
            return profile

        # Aggregate measurements
        profile = self._aggregate_measurements(
            frame_measurements,
            frequency_profiles,
            grain_patches,
        )

        # Identify film stock
        profile.film_stock_estimate = self._identify_film_stock(profile)

        # Override with config if specified
        if self.config.film_stock_override:
            profile.film_stock_estimate = self.config.film_stock_override
            self._apply_stock_profile(profile)

        profile.frames_analyzed = len(frame_measurements)
        profile.confidence = self._calculate_confidence(frame_measurements)

        logger.info(
            f"Grain profile extracted: size={profile.grain_size:.2f}, "
            f"intensity={profile.grain_intensity:.2f}, "
            f"stock={profile.film_stock_estimate.value}"
        )

        return profile

    def _get_sample_indices(self, total_frames: int) -> List[int]:
        """Get indices for frame sampling."""
        if total_frames <= self.config.extract_sample_count:
            return list(range(total_frames))

        step = total_frames / self.config.extract_sample_count
        return [int(i * step) for i in range(self.config.extract_sample_count)]

    def _analyze_frame_grain(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze grain characteristics of a single frame."""
        measurements: Dict[str, float] = {}

        # Convert to float for analysis
        frame_f = frame.astype(np.float32) / 255.0

        # Convert to YUV for luma/chroma separation
        if HAS_OPENCV:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32) / 255.0
            y_channel = yuv[:, :, 0]
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
        else:
            # Simple conversion without OpenCV
            y_channel = 0.299 * frame_f[:, :, 2] + 0.587 * frame_f[:, :, 1] + 0.114 * frame_f[:, :, 0]
            u_channel = 0.5 * (frame_f[:, :, 0] - y_channel)
            v_channel = 0.5 * (frame_f[:, :, 2] - y_channel)

        # Luminance grain analysis
        luma_grain = self._estimate_grain_level(y_channel)
        measurements["luminance_grain"] = luma_grain

        # Chroma grain analysis
        chroma_u = self._estimate_grain_level(u_channel)
        chroma_v = self._estimate_grain_level(v_channel)
        measurements["chroma_grain"] = (chroma_u + chroma_v) / 2

        # Overall grain intensity
        measurements["grain_intensity"] = (
            luma_grain * 0.7 + measurements["chroma_grain"] * 0.3
        )

        # Grain size estimation via frequency analysis
        measurements["grain_size"] = self._estimate_grain_size(y_channel)

        # Grain uniformity
        measurements["grain_uniformity"] = self._measure_grain_uniformity(y_channel)

        # Contrast-dependent grain
        measurements["contrast_dependent"] = self._detect_contrast_dependency(
            y_channel, frame_f
        )

        return measurements

    def _estimate_grain_level(self, channel: np.ndarray) -> float:
        """Estimate grain level using Laplacian variance method."""
        if HAS_OPENCV:
            # Laplacian of Gaussian for noise estimation
            blurred = cv2.GaussianBlur(channel, (0, 0), 1.5)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        else:
            # Simple gradient-based estimation
            if HAS_SCIPY:
                laplacian = ndimage.laplace(channel)
            else:
                # Manual Laplacian
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
                laplacian = self._convolve2d(channel.astype(np.float64), kernel)

        # Robust noise estimation using MAD
        sigma = np.median(np.abs(laplacian)) / 0.6745

        # Scale to 0-1 (typical sigma range 0-30)
        return min(1.0, sigma / 30.0)

    def _estimate_grain_size(self, channel: np.ndarray) -> float:
        """Estimate grain size using frequency analysis."""
        if not HAS_SCIPY:
            # Fallback: use local variance
            return self._estimate_grain_size_spatial(channel)

        # FFT analysis
        f_transform = fft.fft2(channel)
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Create radial frequency bins
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Normalize radius
        max_r = np.sqrt(crow ** 2 + ccol ** 2)
        r_norm = r / max_r

        # Find peak frequency (excluding DC)
        magnitude[crow, ccol] = 0  # Remove DC

        # Radial average
        bins = 64
        radial_profile = np.zeros(bins)
        for i in range(bins):
            r_min = i / bins
            r_max = (i + 1) / bins
            mask = (r_norm >= r_min) & (r_norm < r_max)
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude[mask])

        # Find weighted centroid (grain frequency)
        total_power = np.sum(radial_profile)
        if total_power > 0:
            centroid = np.sum(radial_profile * np.arange(bins)) / total_power
            # Normalize to 0-1 (low frequency = large grain)
            grain_size = 1.0 - (centroid / bins)
        else:
            grain_size = 0.5

        return np.clip(grain_size, 0.0, 1.0)

    def _estimate_grain_size_spatial(self, channel: np.ndarray) -> float:
        """Estimate grain size using spatial analysis (fallback)."""
        # Extract high-pass component
        kernel_size = 5
        if HAS_OPENCV:
            blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
        else:
            if HAS_SCIPY:
                blurred = ndimage.gaussian_filter(channel, sigma=kernel_size/6)
            else:
                blurred = channel  # No filtering available

        high_pass = np.abs(channel.astype(np.float64) - blurred.astype(np.float64))

        # Measure autocorrelation width as proxy for grain size
        # Larger autocorrelation width = larger grain
        if HAS_SCIPY:
            autocorr = signal.correlate2d(high_pass, high_pass, mode='same')
            center = autocorr[autocorr.shape[0]//2, autocorr.shape[1]//2]
            if center > 0:
                # Find half-power width
                threshold = center * 0.5
                above_threshold = autocorr > threshold
                width = np.sqrt(np.sum(above_threshold)) / autocorr.shape[0]
                return np.clip(width * 5, 0.0, 1.0)

        # Fallback: use variance ratio
        return 0.5

    def _measure_grain_uniformity(self, channel: np.ndarray) -> float:
        """Measure how uniform grain is across the frame."""
        # Divide into blocks
        block_size = 64
        h, w = channel.shape

        block_stds = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = channel[y:y+block_size, x:x+block_size]

                # Extract grain component
                if HAS_OPENCV:
                    blurred = cv2.GaussianBlur(block, (5, 5), 0)
                elif HAS_SCIPY:
                    blurred = ndimage.gaussian_filter(block, sigma=1)
                else:
                    blurred = block

                grain = np.abs(block.astype(np.float64) - blurred.astype(np.float64))
                block_stds.append(np.std(grain))

        if not block_stds:
            return 0.8

        # Uniformity = inverse of coefficient of variation
        mean_std = np.mean(block_stds)
        std_of_stds = np.std(block_stds)

        if mean_std > 0:
            cv = std_of_stds / mean_std
            uniformity = 1.0 / (1.0 + cv * 2)
        else:
            uniformity = 1.0

        return np.clip(uniformity, 0.0, 1.0)

    def _detect_contrast_dependency(
        self,
        y_channel: np.ndarray,
        frame: np.ndarray,
    ) -> float:
        """Detect if grain varies with brightness."""
        # Divide frame by brightness
        dark_mask = y_channel < 0.33
        mid_mask = (y_channel >= 0.33) & (y_channel < 0.67)
        bright_mask = y_channel >= 0.67

        # Measure grain in each region
        grain_levels = []
        for mask in [dark_mask, mid_mask, bright_mask]:
            if np.sum(mask) > 100:
                region = y_channel[mask]
                grain = self._estimate_grain_level(
                    region.reshape(int(np.sqrt(len(region))), -1)[:64, :64]
                ) if len(region) > 4096 else 0.3
                grain_levels.append(grain)

        if len(grain_levels) < 2:
            return 0.5

        # Correlation with brightness
        variation = np.std(grain_levels) / (np.mean(grain_levels) + 1e-6)
        return np.clip(variation * 3, 0.0, 1.0)

    def _extract_frequency_profile(self, frame: np.ndarray) -> np.ndarray:
        """Extract frequency profile of grain."""
        if not HAS_SCIPY:
            return np.zeros(self.config.frequency_bands)

        # Convert to grayscale
        if HAS_OPENCV:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.mean(frame, axis=2).astype(np.uint8)

        # Extract grain component
        if HAS_OPENCV:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        else:
            blurred = ndimage.gaussian_filter(gray, sigma=1)

        grain = gray.astype(np.float64) - blurred.astype(np.float64)

        # FFT
        f_transform = fft.fft2(grain)
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Radial profile
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        max_r = np.sqrt(crow ** 2 + ccol ** 2)

        # Bin into frequency bands
        profile = np.zeros(self.config.frequency_bands)
        for i in range(self.config.frequency_bands):
            r_min = i * max_r / self.config.frequency_bands
            r_max = (i + 1) * max_r / self.config.frequency_bands
            mask = (r >= r_min) & (r < r_max)
            if np.any(mask):
                profile[i] = np.mean(magnitude[mask])

        # Normalize
        if np.max(profile) > 0:
            profile = profile / np.max(profile)

        return profile

    def _extract_grain_patches(
        self,
        frame: np.ndarray,
        patch_count: int = 4,
        patch_size: int = 64,
    ) -> List[np.ndarray]:
        """Extract grain texture patches from frame."""
        patches = []

        # Convert to grayscale
        if HAS_OPENCV:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.mean(frame, axis=2).astype(np.uint8)

        h, w = gray.shape

        # Find relatively flat areas (low detail) for clean grain extraction
        if HAS_OPENCV:
            edges = cv2.Canny(gray, 30, 100)
        else:
            # Simple edge detection
            if HAS_SCIPY:
                edges = ndimage.sobel(gray)
                edges = (np.abs(edges) > 30).astype(np.uint8) * 255
            else:
                edges = np.zeros_like(gray)

        # Find patches with low edge content
        for _ in range(patch_count * 3):  # Try multiple times
            if len(patches) >= patch_count:
                break

            y = self._rng.randint(0, h - patch_size)
            x = self._rng.randint(0, w - patch_size)

            edge_patch = edges[y:y+patch_size, x:x+patch_size]
            if np.mean(edge_patch) < 10:  # Low edge content
                # Extract grain
                gray_patch = gray[y:y+patch_size, x:x+patch_size].astype(np.float64)

                if HAS_OPENCV:
                    blurred = cv2.GaussianBlur(gray_patch, (5, 5), 0)
                elif HAS_SCIPY:
                    blurred = ndimage.gaussian_filter(gray_patch, sigma=1)
                else:
                    blurred = gray_patch

                grain_patch = gray_patch - blurred
                patches.append(grain_patch.astype(np.float32))

        return patches

    def _measure_temporal_variation(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """Measure grain variation between consecutive frames."""
        # Extract grain from both frames
        if HAS_OPENCV:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float64)
            blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        else:
            gray1 = np.mean(frame1, axis=2).astype(np.float64)
            gray2 = np.mean(frame2, axis=2).astype(np.float64)
            if HAS_SCIPY:
                blurred1 = ndimage.gaussian_filter(gray1, sigma=1)
                blurred2 = ndimage.gaussian_filter(gray2, sigma=1)
            else:
                blurred1 = gray1
                blurred2 = gray2

        grain1 = gray1 - blurred1
        grain2 = gray2 - blurred2

        # Measure difference (should be high for real grain, low for fixed pattern)
        diff = np.abs(grain1 - grain2)

        # Normalize by grain magnitude
        grain_mag = (np.std(grain1) + np.std(grain2)) / 2
        if grain_mag > 0:
            variation = np.mean(diff) / grain_mag
        else:
            variation = 0.5

        return np.clip(variation, 0.0, 1.0)

    def _aggregate_measurements(
        self,
        measurements: List[Dict[str, float]],
        frequency_profiles: List[np.ndarray],
        grain_patches: List[np.ndarray],
    ) -> GrainProfile:
        """Aggregate per-frame measurements into profile."""
        profile = GrainProfile()

        # Average measurements
        profile.grain_size = np.mean([m.get("grain_size", 0.5) for m in measurements])
        profile.grain_intensity = np.mean([m.get("grain_intensity", 0.3) for m in measurements])
        profile.luminance_grain = np.mean([m.get("luminance_grain", 0.3) for m in measurements])
        profile.chroma_grain = np.mean([m.get("chroma_grain", 0.15) for m in measurements])
        profile.grain_uniformity = np.mean([m.get("grain_uniformity", 0.8) for m in measurements])
        profile.contrast_dependent = np.mean([m.get("contrast_dependent", 0.5) for m in measurements]) > 0.4

        # Temporal variation
        temporal_vars = [m.get("temporal_var", 0.15) for m in measurements if "temporal_var" in m]
        if temporal_vars:
            profile.temporal_variation = np.mean(temporal_vars)

        # Average frequency profile
        if frequency_profiles:
            profile.frequency_profile = np.mean(frequency_profiles, axis=0)

            # Calculate summary statistics
            total_power = np.sum(profile.frequency_profile)
            if total_power > 0:
                cumsum = np.cumsum(profile.frequency_profile)
                profile.median_frequency = np.searchsorted(cumsum, total_power / 2) / len(profile.frequency_profile)
                profile.peak_frequency = np.argmax(profile.frequency_profile) / len(profile.frequency_profile)

        # Store grain patches for texture synthesis
        profile.sample_grain_patches = grain_patches[:8]  # Keep up to 8 patches

        return profile

    def _identify_film_stock(self, profile: GrainProfile) -> FilmStockGrainType:
        """Identify likely film stock from grain profile."""
        best_match = FilmStockGrainType.UNKNOWN
        best_score = 0.0

        for stock, params in FILM_STOCK_PROFILES.items():
            score = 0.0

            # Size match
            size_diff = abs(profile.grain_size - params["size"])
            score += (1.0 - size_diff) * 0.3

            # Intensity match
            intensity_diff = abs(profile.grain_intensity - params["intensity"])
            score += (1.0 - intensity_diff) * 0.25

            # Uniformity match
            uniformity_diff = abs(profile.grain_uniformity - params["uniformity"])
            score += (1.0 - uniformity_diff) * 0.2

            # Temporal variation match
            temporal_diff = abs(profile.temporal_variation - params["temporal_var"])
            score += (1.0 - temporal_diff) * 0.15

            # Luma/chroma ratio match
            if profile.luminance_grain > 0:
                ratio = profile.luminance_grain / (profile.luminance_grain + profile.chroma_grain + 1e-6)
                ratio_diff = abs(ratio - params["luma_ratio"])
                score += (1.0 - ratio_diff) * 0.1

            if score > best_score:
                best_score = score
                best_match = stock

        logger.debug(f"Film stock match: {best_match.value} (score={best_score:.2f})")
        return best_match

    def _apply_stock_profile(self, profile: GrainProfile) -> None:
        """Apply film stock parameters to profile."""
        if profile.film_stock_estimate not in FILM_STOCK_PROFILES:
            return

        params = FILM_STOCK_PROFILES[profile.film_stock_estimate]

        # Blend measured values with stock parameters (70% measured, 30% stock)
        blend = 0.3
        profile.grain_size = profile.grain_size * (1 - blend) + params["size"] * blend
        profile.grain_intensity = profile.grain_intensity * (1 - blend) + params["intensity"] * blend
        profile.grain_uniformity = profile.grain_uniformity * (1 - blend) + params["uniformity"] * blend

    def _calculate_confidence(self, measurements: List[Dict[str, float]]) -> float:
        """Calculate confidence in grain profile."""
        if len(measurements) < 3:
            return 0.5

        # Based on consistency of measurements
        intensities = [m.get("grain_intensity", 0.3) for m in measurements]
        std = np.std(intensities)
        mean = np.mean(intensities)

        if mean > 0:
            cv = std / mean
            confidence = 1.0 / (1.0 + cv * 2)
        else:
            confidence = 0.5

        # Bonus for more frames
        frame_bonus = min(0.2, len(measurements) / 100)

        return np.clip(confidence + frame_bonus, 0.0, 1.0)

    # =========================================================================
    # Grain Removal
    # =========================================================================

    def remove_grain(
        self,
        frame: np.ndarray,
        profile: GrainProfile,
        strength: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove grain while preserving detail.

        Uses wavelet decomposition or frequency domain filtering to
        separate grain from image content.

        Args:
            frame: Input frame (BGR format)
            profile: Grain profile for adaptive removal
            strength: Removal strength (0-1, default 1.0)

        Returns:
            Tuple of (clean_frame, grain_layer)
        """
        if HAS_PYWT:
            return self._remove_grain_wavelet(frame, profile, strength)
        elif HAS_SCIPY:
            return self._remove_grain_frequency(frame, profile, strength)
        else:
            return self._remove_grain_simple(frame, profile, strength)

    def _remove_grain_wavelet(
        self,
        frame: np.ndarray,
        profile: GrainProfile,
        strength: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove grain using wavelet decomposition."""
        frame_f = frame.astype(np.float64)
        clean = np.zeros_like(frame_f)
        grain = np.zeros_like(frame_f)

        # Process each channel
        for c in range(frame.shape[2]):
            channel = frame_f[:, :, c]

            # Wavelet decomposition
            coeffs = pywt.wavedec2(channel, 'db4', level=self.config.wavelet_levels)

            # Estimate noise threshold based on profile
            # Higher grain intensity = higher threshold
            base_threshold = profile.grain_intensity * 30 * strength

            # Soft threshold detail coefficients
            denoised_coeffs = [coeffs[0]]  # Keep approximation
            grain_coeffs = [np.zeros_like(coeffs[0])]

            for i, (cH, cV, cD) in enumerate(coeffs[1:]):
                # Adaptive threshold per level
                # Fine grain (high frequency) in early levels
                level_weight = 1.0 - (i / self.config.wavelet_levels) * 0.5
                threshold = base_threshold * level_weight

                # Soft thresholding
                cH_clean, cH_grain = self._soft_threshold_separate(cH, threshold)
                cV_clean, cV_grain = self._soft_threshold_separate(cV, threshold)
                cD_clean, cD_grain = self._soft_threshold_separate(cD, threshold)

                denoised_coeffs.append((cH_clean, cV_clean, cD_clean))
                grain_coeffs.append((cH_grain, cV_grain, cD_grain))

            # Reconstruct
            clean[:, :, c] = pywt.waverec2(denoised_coeffs, 'db4')[:channel.shape[0], :channel.shape[1]]

            # Reconstruct grain layer
            grain_coeffs[0] = np.zeros_like(coeffs[0])  # No approximation in grain
            grain[:, :, c] = pywt.waverec2(grain_coeffs, 'db4')[:channel.shape[0], :channel.shape[1]]

        clean = np.clip(clean, 0, 255).astype(np.uint8)
        grain = grain.astype(np.float32)

        return clean, grain

    def _soft_threshold_separate(
        self,
        coeffs: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Soft threshold wavelet coefficients and separate grain."""
        # Soft thresholding
        sign = np.sign(coeffs)
        magnitude = np.abs(coeffs)

        # Values below threshold are grain
        grain = np.where(magnitude < threshold, coeffs, 0)

        # Shrink values above threshold
        clean = sign * np.maximum(magnitude - threshold, 0)

        return clean, grain

    def _remove_grain_frequency(
        self,
        frame: np.ndarray,
        profile: GrainProfile,
        strength: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove grain using frequency domain filtering."""
        frame_f = frame.astype(np.float64)
        clean = np.zeros_like(frame_f)
        grain = np.zeros_like(frame_f)

        for c in range(frame.shape[2]):
            channel = frame_f[:, :, c]

            # FFT
            f_transform = fft.fft2(channel)
            f_shift = fft.fftshift(f_transform)

            # Create frequency-based filter
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2

            y, x = np.ogrid[:rows, :cols]
            r = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
            max_r = np.sqrt(crow ** 2 + ccol ** 2)

            # Grain is typically in mid-to-high frequencies
            # Profile's grain_size indicates where grain starts
            grain_freq_start = (1.0 - profile.grain_size) * 0.3

            # Create smooth filter
            filter_strength = strength * profile.grain_intensity
            filter_mask = np.ones((rows, cols))

            # Attenuate frequencies where grain is expected
            grain_band = (r / max_r) > grain_freq_start
            filter_mask[grain_band] *= (1.0 - filter_strength * 0.7)

            # Very high frequencies (salt & pepper) get more attenuation
            very_high = (r / max_r) > 0.7
            filter_mask[very_high] *= (1.0 - filter_strength * 0.3)

            # Apply filter
            f_filtered = f_shift * filter_mask
            grain_spectrum = f_shift * (1 - filter_mask)

            # Inverse FFT
            f_ishift = fft.ifftshift(f_filtered)
            clean[:, :, c] = np.real(fft.ifft2(f_ishift))

            grain_ishift = fft.ifftshift(grain_spectrum)
            grain[:, :, c] = np.real(fft.ifft2(grain_ishift))

        clean = np.clip(clean, 0, 255).astype(np.uint8)
        grain = grain.astype(np.float32)

        return clean, grain

    def _remove_grain_simple(
        self,
        frame: np.ndarray,
        profile: GrainProfile,
        strength: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple grain removal using blur (fallback)."""
        # Kernel size based on grain size
        ksize = int(3 + profile.grain_size * 4) | 1  # Ensure odd
        sigma = profile.grain_intensity * 2 * strength

        if HAS_OPENCV:
            clean = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        else:
            clean = frame.copy()

        grain = (frame.astype(np.float32) - clean.astype(np.float32))

        return clean, grain

    # =========================================================================
    # Grain Restoration
    # =========================================================================

    def restore_grain(
        self,
        frame: np.ndarray,
        profile: GrainProfile,
        opacity: float = 0.35,
        grain_layer: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Re-apply authentic grain after processing.

        Args:
            frame: Processed frame (BGR format)
            profile: Grain profile to match
            opacity: Grain opacity (0-1, default 0.35)
            grain_layer: Optional pre-extracted grain to re-apply

        Returns:
            Frame with grain restored
        """
        if grain_layer is not None:
            # Re-apply extracted grain
            return self._apply_grain_layer(frame, grain_layer, opacity)
        else:
            # Synthesize matching grain
            grain = self.synthesize_grain(frame.shape[:2], profile)
            return self._apply_grain_layer(frame, grain, opacity)

    def _apply_grain_layer(
        self,
        frame: np.ndarray,
        grain: np.ndarray,
        opacity: float,
    ) -> np.ndarray:
        """Apply grain layer to frame."""
        frame_f = frame.astype(np.float64)

        # Handle single-channel grain
        if len(grain.shape) == 2:
            grain = np.stack([grain] * 3, axis=2)

        # Scale grain by opacity
        grain_scaled = grain * opacity

        # Add grain
        result = frame_f + grain_scaled

        return np.clip(result, 0, 255).astype(np.uint8)

    # =========================================================================
    # Grain Synthesis
    # =========================================================================

    def synthesize_grain(
        self,
        shape: Tuple[int, int],
        profile: GrainProfile,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Generate new grain matching the profile.

        Uses a combination of techniques:
        - Perlin noise for organic appearance
        - Frequency shaping to match profile
        - Temporal coherence for video

        Args:
            shape: Output shape (height, width)
            profile: Grain profile to match
            frame_index: Frame index for temporal coherence

        Returns:
            Synthesized grain layer (float32, centered around 0)
        """
        height, width = shape

        # Base noise
        if self.config.temporal_coherence:
            # Use frame index for temporal variation
            seed = (self.config.grain_seed or 0) + frame_index
            rng = np.random.RandomState(seed)
        else:
            rng = self._rng

        # Generate multi-octave Perlin-like noise for organic appearance
        grain = self._generate_perlin_noise(height, width, profile, rng)

        # Apply frequency shaping to match profile
        if HAS_SCIPY:
            grain = self._shape_grain_frequency(grain, profile)

        # Apply intensity
        grain = grain * profile.grain_intensity * 50  # Scale to typical range

        # Apply luminance-dependent response if enabled
        # (This would require the actual frame, so we return uniform grain here)

        return grain.astype(np.float32)

    def _generate_perlin_noise(
        self,
        height: int,
        width: int,
        profile: GrainProfile,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate Perlin-like noise for film grain."""
        # Octave frequencies based on grain size
        # Smaller grain = higher frequencies
        base_freq = 4 + int(profile.grain_size * 12)

        noise = np.zeros((height, width), dtype=np.float64)

        # Multiple octaves for organic appearance
        octaves = 3 if profile.grain_size < 0.5 else 2

        for octave in range(octaves):
            freq = base_freq * (2 ** octave)
            amplitude = 1.0 / (2 ** octave)

            # Generate noise at this frequency
            octave_noise = self._generate_smooth_noise(height, width, freq, rng)
            noise += octave_noise * amplitude

        # Normalize
        noise = noise / np.max(np.abs(noise) + 1e-6)

        return noise

    def _generate_smooth_noise(
        self,
        height: int,
        width: int,
        frequency: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate smooth noise at given frequency."""
        # Generate low-resolution noise
        small_h = max(2, height // frequency)
        small_w = max(2, width // frequency)

        small_noise = rng.randn(small_h, small_w)

        # Upscale with smooth interpolation
        if HAS_OPENCV:
            noise = cv2.resize(
                small_noise,
                (width, height),
                interpolation=cv2.INTER_CUBIC
            )
        elif HAS_SCIPY:
            from scipy.ndimage import zoom
            noise = zoom(small_noise, (height / small_h, width / small_w), order=3)
        else:
            # Simple nearest-neighbor upscale
            noise = np.repeat(np.repeat(small_noise, frequency, axis=0), frequency, axis=1)
            noise = noise[:height, :width]

        return noise

    def _shape_grain_frequency(
        self,
        grain: np.ndarray,
        profile: GrainProfile,
    ) -> np.ndarray:
        """Shape grain to match profile's frequency characteristics."""
        # FFT
        f_transform = fft.fft2(grain)
        f_shift = fft.fftshift(f_transform)

        rows, cols = grain.shape
        crow, ccol = rows // 2, cols // 2

        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        max_r = np.sqrt(crow ** 2 + ccol ** 2)

        # Create target frequency profile
        if len(profile.frequency_profile) > 0 and np.sum(profile.frequency_profile) > 0:
            # Map profile to 2D filter
            num_bands = len(profile.frequency_profile)
            target_filter = np.ones((rows, cols))

            for i, power in enumerate(profile.frequency_profile):
                r_min = i * max_r / num_bands
                r_max = (i + 1) * max_r / num_bands
                mask = (r >= r_min) & (r < r_max)
                target_filter[mask] = power + 0.1  # Avoid zero

            # Apply filter
            f_shift = f_shift * target_filter

        # Inverse FFT
        f_ishift = fft.ifftshift(f_shift)
        shaped = np.real(fft.ifft2(f_ishift))

        # Normalize
        shaped = shaped / (np.std(shaped) + 1e-6)

        return shaped

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def process_with_preservation(
        self,
        frames: List[np.ndarray],
        processor: Callable[[List[np.ndarray]], List[np.ndarray]],
        profile: Optional[GrainProfile] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Full pipeline: extract -> remove -> process -> restore grain.

        This is the main entry point for grain-preserving processing.

        Args:
            frames: Input frames (BGR format)
            processor: Processing function that takes and returns frames
            profile: Pre-computed grain profile (extracted if None)
            progress_callback: Progress callback (0-1)

        Returns:
            Processed frames with grain restored
        """
        if not frames:
            return frames

        logger.info(f"Processing {len(frames)} frames with grain preservation")

        # Phase 1: Extract profile if needed (10% progress)
        if profile is None:
            logger.info("Extracting grain profile...")
            profile = self.extract_profile(
                frames,
                progress_callback=lambda p: progress_callback(p * 0.1) if progress_callback else None
            )

        if progress_callback:
            progress_callback(0.1)

        # Phase 2: Remove grain from all frames (20% progress)
        logger.info("Removing grain for processing...")
        clean_frames = []
        grain_layers = []

        for i, frame in enumerate(frames):
            clean, grain = self.remove_grain(frame, profile)
            clean_frames.append(clean)
            grain_layers.append(grain)

            if progress_callback:
                progress_callback(0.1 + (i + 1) / len(frames) * 0.2)

        # Phase 3: Apply processor (50% progress)
        logger.info("Applying processing...")

        def proc_progress(p: float) -> None:
            if progress_callback:
                progress_callback(0.3 + p * 0.5)

        try:
            processed_frames = processor(clean_frames)
        except TypeError:
            # Processor might not accept progress callback
            processed_frames = processor(clean_frames)

        if progress_callback:
            progress_callback(0.8)

        # Phase 4: Restore grain (20% progress)
        logger.info("Restoring grain...")
        output_frames = []

        for i, (frame, grain_layer) in enumerate(zip(processed_frames, grain_layers)):
            # Use stored grain layer for perfect restoration
            restored = self.restore_grain(
                frame,
                profile,
                opacity=self.config.grain_opacity,
                grain_layer=grain_layer if self.config.preserve_grain else None
            )
            output_frames.append(restored)

            if progress_callback:
                progress_callback(0.8 + (i + 1) / len(frames) * 0.2)

        logger.info("Grain preservation complete")
        return output_frames

    # =========================================================================
    # Utilities
    # =========================================================================

    def _convolve2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
    ) -> np.ndarray:
        """Simple 2D convolution (fallback when scipy unavailable)."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        # Convolve
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(
                    padded[i:i+kh, j:j+kw] * kernel
                )

        return result

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._perlin_cache.clear()
        self._grain_texture_cache.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_grain_manager(
    preserve_grain: bool = True,
    grain_opacity: float = 0.35,
    film_stock: Optional[str] = None,
    **kwargs,
) -> GrainManager:
    """Factory function to create a grain manager.

    Args:
        preserve_grain: Whether to preserve grain through processing
        grain_opacity: Opacity for restored grain (0-1)
        film_stock: Override film stock type (e.g., "kodak_5219")
        **kwargs: Additional GrainConfig parameters

    Returns:
        Configured GrainManager instance
    """
    stock_override = None
    if film_stock:
        try:
            stock_override = FilmStockGrainType(film_stock)
        except ValueError:
            logger.warning(f"Unknown film stock: {film_stock}")

    config = GrainConfig(
        preserve_grain=preserve_grain,
        grain_opacity=grain_opacity,
        film_stock_override=stock_override,
        **kwargs,
    )

    return GrainManager(config)


def extract_grain_profile(
    frames: List[np.ndarray],
    sample_count: int = 30,
) -> GrainProfile:
    """Convenience function to extract grain profile.

    Args:
        frames: Source frames to analyze
        sample_count: Number of frames to sample

    Returns:
        Extracted GrainProfile
    """
    config = GrainConfig(extract_sample_count=sample_count)
    manager = GrainManager(config)
    return manager.extract_profile(frames)


def process_with_grain_preservation(
    frames: List[np.ndarray],
    processor: Callable[[List[np.ndarray]], List[np.ndarray]],
    opacity: float = 0.35,
) -> List[np.ndarray]:
    """Convenience function for grain-preserving processing.

    Args:
        frames: Input frames
        processor: Processing function
        opacity: Grain restoration opacity

    Returns:
        Processed frames with grain preserved
    """
    config = GrainConfig(preserve_grain=True, grain_opacity=opacity)
    manager = GrainManager(config)
    return manager.process_with_preservation(frames, processor)
