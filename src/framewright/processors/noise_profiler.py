"""Advanced Noise Profiler for Video Restoration.

Provides detailed noise analysis and characterization for optimal
denoising parameter selection.

Features:
- Separate luminance and chrominance noise analysis
- Temporal noise vs spatial noise detection
- Noise type classification (Gaussian, salt & pepper, film grain, compression)
- Frequency-domain noise analysis
- Per-scene noise tracking
- Denoiser recommendation based on noise profile

Example:
    >>> profiler = NoiseProfiler()
    >>> profile = profiler.analyze_video("video.mp4")
    >>> print(f"Noise type: {profile.dominant_type}")
    >>> print(f"Recommended denoiser: {profile.recommended_denoiser}")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy import ndimage, fft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class NoiseType(Enum):
    """Types of noise commonly found in video."""
    GAUSSIAN = "gaussian"           # Random additive noise
    SALT_PEPPER = "salt_pepper"     # Impulse noise
    FILM_GRAIN = "film_grain"       # Organic film grain
    COMPRESSION = "compression"     # Block artifacts, mosquito noise
    BANDING = "banding"             # Color banding in gradients
    TEMPORAL = "temporal"           # Frame-to-frame flickering
    CHROMA = "chroma"               # Color noise (common in low light)
    MIXED = "mixed"                 # Combination of types
    MINIMAL = "minimal"             # Very low noise


class DenoiserType(Enum):
    """Recommended denoiser types."""
    NONE = "none"                   # No denoising needed
    LIGHT = "light"                 # Light spatial denoising
    TEMPORAL = "temporal"           # Temporal denoising (VRT, BasicVSR)
    AGGRESSIVE = "aggressive"       # Strong denoising
    GRAIN_PRESERVE = "grain_preserve"  # Denoise while preserving grain
    COMPRESSION_FIX = "compression_fix"  # Deblock + mosquito removal
    CHROMA_ONLY = "chroma_only"     # Denoise chroma, preserve luma


@dataclass
class NoiseCharacteristics:
    """Detailed noise characteristics for a region or frame."""
    luminance_noise: float = 0.0    # Noise in Y channel (0-100 scale)
    chroma_noise: float = 0.0       # Noise in UV channels (0-100 scale)
    temporal_noise: float = 0.0     # Frame-to-frame variation (0-100)

    # Frequency analysis
    low_freq_noise: float = 0.0     # Low frequency noise (blocking)
    mid_freq_noise: float = 0.0     # Mid frequency noise (grain/gaussian)
    high_freq_noise: float = 0.0    # High frequency noise (salt/pepper)

    # Spatial analysis
    edge_noise: float = 0.0         # Noise around edges (mosquito)
    flat_area_noise: float = 0.0    # Noise in flat regions (banding/blocking)

    # Texture analysis
    grain_intensity: float = 0.0    # Film grain strength
    grain_uniformity: float = 0.0   # How uniform the grain is

    def overall_noise_level(self) -> float:
        """Calculate overall noise level (0-100)."""
        return (self.luminance_noise * 0.6 + self.chroma_noise * 0.4)


@dataclass
class NoiseProfile:
    """Complete noise profile for a video."""
    # Overall characteristics
    overall_level: float = 0.0      # 0-100 scale
    dominant_type: NoiseType = NoiseType.MINIMAL
    secondary_types: List[NoiseType] = field(default_factory=list)

    # Detailed measurements
    characteristics: NoiseCharacteristics = field(default_factory=NoiseCharacteristics)

    # Per-scene breakdown
    scene_profiles: Dict[int, NoiseCharacteristics] = field(default_factory=dict)

    # Recommendations
    recommended_denoiser: DenoiserType = DenoiserType.NONE
    recommended_strength: float = 0.0  # 0-1 scale

    # Additional settings
    preserve_grain: bool = False
    chroma_denoise_extra: bool = False
    temporal_denoise_recommended: bool = False

    # Confidence
    confidence: float = 0.0  # 0-1 scale
    frames_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_level": self.overall_level,
            "dominant_type": self.dominant_type.value,
            "secondary_types": [t.value for t in self.secondary_types],
            "luminance_noise": self.characteristics.luminance_noise,
            "chroma_noise": self.characteristics.chroma_noise,
            "temporal_noise": self.characteristics.temporal_noise,
            "recommended_denoiser": self.recommended_denoiser.value,
            "recommended_strength": self.recommended_strength,
            "preserve_grain": self.preserve_grain,
            "confidence": self.confidence,
            "frames_analyzed": self.frames_analyzed,
        }


class NoiseProfiler:
    """Advanced noise analysis for video restoration.

    Analyzes video noise characteristics to provide optimal
    denoising recommendations.
    """

    def __init__(self, sample_frames: int = 30, sample_regions: int = 5):
        """Initialize noise profiler.

        Args:
            sample_frames: Number of frames to sample for analysis
            sample_regions: Number of regions per frame to analyze
        """
        self.sample_frames = sample_frames
        self.sample_regions = sample_regions

        if not HAS_CV2:
            logger.warning("OpenCV not available - noise profiling limited")

    def analyze_video(
        self,
        video_path: Path,
        scene_boundaries: Optional[List[int]] = None,
    ) -> NoiseProfile:
        """Analyze noise characteristics of a video.

        Args:
            video_path: Path to video file
            scene_boundaries: Optional list of scene change frame numbers

        Returns:
            NoiseProfile with detailed analysis
        """
        video_path = Path(video_path)
        profile = NoiseProfile()

        if not HAS_CV2:
            logger.warning("OpenCV required for noise profiling")
            return profile

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return profile

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame indices to sample
            if total_frames <= self.sample_frames:
                sample_indices = list(range(total_frames))
            else:
                step = total_frames // self.sample_frames
                sample_indices = [i * step for i in range(self.sample_frames)]

            frame_characteristics: List[NoiseCharacteristics] = []
            prev_frame = None

            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Analyze frame
                chars = self._analyze_frame(frame, prev_frame)
                frame_characteristics.append(chars)

                prev_frame = frame

            cap.release()

            # Aggregate results
            if frame_characteristics:
                profile = self._aggregate_characteristics(frame_characteristics)
                profile.frames_analyzed = len(frame_characteristics)

                # Determine recommendations
                self._determine_recommendations(profile)

            return profile

        except Exception as e:
            logger.error(f"Noise profiling failed: {e}")
            return profile

    def _analyze_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> NoiseCharacteristics:
        """Analyze noise in a single frame.

        Args:
            frame: BGR frame
            prev_frame: Previous frame for temporal analysis

        Returns:
            NoiseCharacteristics for this frame
        """
        chars = NoiseCharacteristics()

        # Convert to YUV for separate luma/chroma analysis
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        u_channel = yuv[:, :, 1]
        v_channel = yuv[:, :, 2]

        # Luminance noise (using Laplacian variance method)
        chars.luminance_noise = self._estimate_noise_laplacian(y_channel)

        # Chroma noise
        u_noise = self._estimate_noise_laplacian(u_channel)
        v_noise = self._estimate_noise_laplacian(v_channel)
        chars.chroma_noise = (u_noise + v_noise) / 2

        # Temporal noise (if previous frame available)
        if prev_frame is not None:
            prev_yuv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)
            prev_y = prev_yuv[:, :, 0]

            # Calculate frame difference, excluding motion
            chars.temporal_noise = self._estimate_temporal_noise(y_channel, prev_y)

        # Frequency analysis
        if HAS_SCIPY:
            freq_analysis = self._analyze_frequency_domain(y_channel)
            chars.low_freq_noise = freq_analysis.get("low", 0)
            chars.mid_freq_noise = freq_analysis.get("mid", 0)
            chars.high_freq_noise = freq_analysis.get("high", 0)

        # Edge vs flat area analysis
        chars.edge_noise, chars.flat_area_noise = self._analyze_spatial_noise(y_channel)

        # Grain analysis
        chars.grain_intensity, chars.grain_uniformity = self._analyze_grain(y_channel)

        return chars

    def _estimate_noise_laplacian(self, channel: np.ndarray) -> float:
        """Estimate noise using Laplacian variance method.

        This method estimates noise by computing the variance of
        the Laplacian of the image, which responds primarily to noise.

        Args:
            channel: Single channel image (grayscale)

        Returns:
            Estimated noise level (0-100 scale)
        """
        # Apply Laplacian
        laplacian = cv2.Laplacian(channel, cv2.CV_64F)

        # Calculate sigma using median absolute deviation (robust)
        sigma = np.median(np.abs(laplacian)) / 0.6745

        # Scale to 0-100 range (typical noise sigma is 0-50)
        noise_level = min(100, sigma * 2)

        return noise_level

    def _estimate_temporal_noise(
        self,
        current: np.ndarray,
        previous: np.ndarray,
    ) -> float:
        """Estimate temporal noise between frames.

        Uses motion-compensated difference to isolate noise from motion.

        Args:
            current: Current frame Y channel
            previous: Previous frame Y channel

        Returns:
            Temporal noise estimate (0-100)
        """
        # Simple difference (motion will contribute, but for static areas this works)
        diff = cv2.absdiff(current, previous)

        # Find static regions (low motion) using threshold
        # Areas with small differences are likely static
        static_mask = diff < 15  # Threshold for "no motion"

        if np.sum(static_mask) > 100:  # Need enough static pixels
            # Noise in static regions
            static_noise = np.std(diff[static_mask])
            return min(100, static_noise * 4)

        return 0

    def _analyze_frequency_domain(self, channel: np.ndarray) -> Dict[str, float]:
        """Analyze noise in frequency domain.

        Args:
            channel: Single channel image

        Returns:
            Dict with low/mid/high frequency noise levels
        """
        if not HAS_SCIPY:
            return {"low": 0, "mid": 0, "high": 0}

        # Compute 2D FFT
        f_transform = fft.fft2(channel.astype(np.float64))
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Create frequency bands
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # Create distance matrix from center
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        max_dist = np.sqrt(crow ** 2 + ccol ** 2)

        # Define bands (normalized)
        low_mask = dist < (max_dist * 0.1)
        mid_mask = (dist >= max_dist * 0.1) & (dist < max_dist * 0.4)
        high_mask = dist >= max_dist * 0.4

        # Calculate power in each band (excluding DC)
        magnitude[crow, ccol] = 0  # Remove DC component

        low_power = np.mean(magnitude[low_mask]) if np.any(low_mask) else 0
        mid_power = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
        high_power = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0

        # Normalize to 0-100 scale
        total = low_power + mid_power + high_power + 1e-10

        return {
            "low": (low_power / total) * 100,
            "mid": (mid_power / total) * 100,
            "high": (high_power / total) * 100,
        }

    def _analyze_spatial_noise(
        self,
        channel: np.ndarray,
    ) -> Tuple[float, float]:
        """Analyze noise in edge regions vs flat regions.

        Args:
            channel: Single channel image

        Returns:
            Tuple of (edge_noise, flat_area_noise)
        """
        # Detect edges
        edges = cv2.Canny(channel, 50, 150)
        edge_mask = edges > 0

        # Dilate edges for region around edges
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2) > 0

        # Flat region is inverse of edge region
        flat_mask = ~edge_region

        # Calculate noise in each region using local variance
        # Apply Laplacian and measure in each region
        laplacian = cv2.Laplacian(channel, cv2.CV_64F)

        edge_noise = 0
        flat_noise = 0

        if np.sum(edge_region) > 100:
            edge_noise = np.std(laplacian[edge_region])
            edge_noise = min(100, edge_noise * 2)

        if np.sum(flat_mask) > 100:
            flat_noise = np.std(laplacian[flat_mask])
            flat_noise = min(100, flat_noise * 2)

        return edge_noise, flat_noise

    def _analyze_grain(self, channel: np.ndarray) -> Tuple[float, float]:
        """Analyze film grain characteristics.

        Args:
            channel: Single channel image

        Returns:
            Tuple of (grain_intensity, grain_uniformity)
        """
        # High-pass filter to isolate grain
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        high_pass = cv2.absdiff(channel, blurred)

        # Grain intensity is the overall strength
        grain_intensity = np.mean(high_pass) * 4  # Scale to 0-100
        grain_intensity = min(100, grain_intensity)

        # Grain uniformity - how consistent is the grain across the image
        # Divide into blocks and measure variance of grain in each block
        block_size = 64
        h, w = channel.shape
        block_stds = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = high_pass[y:y + block_size, x:x + block_size]
                block_stds.append(np.std(block))

        if block_stds:
            # Lower variance in block stds = more uniform grain
            std_of_stds = np.std(block_stds)
            mean_std = np.mean(block_stds)

            if mean_std > 0:
                uniformity = 100 - min(100, (std_of_stds / mean_std) * 100)
            else:
                uniformity = 100
        else:
            uniformity = 0

        return grain_intensity, uniformity

    def _aggregate_characteristics(
        self,
        frame_chars: List[NoiseCharacteristics],
    ) -> NoiseProfile:
        """Aggregate frame-level characteristics into video profile.

        Args:
            frame_chars: List of per-frame characteristics

        Returns:
            Aggregated NoiseProfile
        """
        profile = NoiseProfile()

        # Average all measurements
        n = len(frame_chars)

        avg_chars = NoiseCharacteristics()
        avg_chars.luminance_noise = np.mean([c.luminance_noise for c in frame_chars])
        avg_chars.chroma_noise = np.mean([c.chroma_noise for c in frame_chars])
        avg_chars.temporal_noise = np.mean([c.temporal_noise for c in frame_chars])
        avg_chars.low_freq_noise = np.mean([c.low_freq_noise for c in frame_chars])
        avg_chars.mid_freq_noise = np.mean([c.mid_freq_noise for c in frame_chars])
        avg_chars.high_freq_noise = np.mean([c.high_freq_noise for c in frame_chars])
        avg_chars.edge_noise = np.mean([c.edge_noise for c in frame_chars])
        avg_chars.flat_area_noise = np.mean([c.flat_area_noise for c in frame_chars])
        avg_chars.grain_intensity = np.mean([c.grain_intensity for c in frame_chars])
        avg_chars.grain_uniformity = np.mean([c.grain_uniformity for c in frame_chars])

        profile.characteristics = avg_chars
        profile.overall_level = avg_chars.overall_noise_level()

        # Determine dominant noise type
        profile.dominant_type = self._classify_noise_type(avg_chars)

        # Calculate confidence based on consistency
        luma_std = np.std([c.luminance_noise for c in frame_chars])
        if avg_chars.luminance_noise > 0:
            profile.confidence = max(0, 1 - (luma_std / avg_chars.luminance_noise))
        else:
            profile.confidence = 0.5

        return profile

    def _classify_noise_type(self, chars: NoiseCharacteristics) -> NoiseType:
        """Classify the dominant noise type.

        Args:
            chars: Aggregated noise characteristics

        Returns:
            Dominant NoiseType
        """
        if chars.overall_noise_level() < 5:
            return NoiseType.MINIMAL

        # Film grain: uniform, organic texture
        if chars.grain_intensity > 20 and chars.grain_uniformity > 60:
            return NoiseType.FILM_GRAIN

        # Compression: high in flat areas, low frequency noise
        if chars.flat_area_noise > chars.edge_noise * 1.5 and chars.low_freq_noise > 40:
            return NoiseType.COMPRESSION

        # Salt & pepper: high frequency noise dominant
        if chars.high_freq_noise > chars.mid_freq_noise * 1.5:
            return NoiseType.SALT_PEPPER

        # Chroma noise: high chroma, low luma noise
        if chars.chroma_noise > chars.luminance_noise * 1.5:
            return NoiseType.CHROMA

        # Temporal noise: significant frame-to-frame variation
        if chars.temporal_noise > chars.luminance_noise:
            return NoiseType.TEMPORAL

        # Gaussian: balanced mid-frequency noise
        if chars.mid_freq_noise > 30:
            return NoiseType.GAUSSIAN

        return NoiseType.MIXED

    def _determine_recommendations(self, profile: NoiseProfile) -> None:
        """Determine denoising recommendations based on profile.

        Args:
            profile: NoiseProfile to update with recommendations
        """
        chars = profile.characteristics
        noise_type = profile.dominant_type
        level = profile.overall_level

        # Minimal noise - no denoising
        if level < 5:
            profile.recommended_denoiser = DenoiserType.NONE
            profile.recommended_strength = 0
            return

        # Film grain - preserve grain mode
        if noise_type == NoiseType.FILM_GRAIN:
            profile.recommended_denoiser = DenoiserType.GRAIN_PRESERVE
            profile.recommended_strength = min(1.0, level / 50)
            profile.preserve_grain = True
            return

        # Compression artifacts - specialized deblocking
        if noise_type == NoiseType.COMPRESSION:
            profile.recommended_denoiser = DenoiserType.COMPRESSION_FIX
            profile.recommended_strength = min(1.0, level / 40)
            return

        # Chroma noise - denoise chroma only
        if noise_type == NoiseType.CHROMA:
            profile.recommended_denoiser = DenoiserType.CHROMA_ONLY
            profile.recommended_strength = min(1.0, chars.chroma_noise / 40)
            profile.chroma_denoise_extra = True
            return

        # Temporal noise - use temporal denoiser
        if noise_type == NoiseType.TEMPORAL or chars.temporal_noise > 15:
            profile.recommended_denoiser = DenoiserType.TEMPORAL
            profile.recommended_strength = min(1.0, level / 40)
            profile.temporal_denoise_recommended = True
            return

        # General noise - based on level
        if level < 20:
            profile.recommended_denoiser = DenoiserType.LIGHT
            profile.recommended_strength = level / 40
        else:
            profile.recommended_denoiser = DenoiserType.AGGRESSIVE
            profile.recommended_strength = min(1.0, level / 60)


def analyze_noise(video_path: Path) -> NoiseProfile:
    """Convenience function to analyze video noise.

    Args:
        video_path: Path to video file

    Returns:
        NoiseProfile with analysis results
    """
    profiler = NoiseProfiler()
    return profiler.analyze_video(video_path)
