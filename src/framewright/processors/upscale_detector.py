"""True Resolution Detection for Upscaled Content.

Detects if video has been upscaled from a lower resolution, identifying
the likely original resolution to avoid re-upscaling already processed content.

Features:
- Edge sharpness analysis for interpolation detection
- Block artifact detection for upscaling patterns
- Frequency analysis for resolution estimation
- Bicubic/bilinear/nearest-neighbor upscale detection
- AI upscaling detection (too-sharp edges)

Example:
    >>> detector = UpscaleDetector()
    >>> result = detector.analyze("video.mp4")
    >>> print(f"Container: {result.container_resolution}")
    >>> print(f"Estimated source: {result.estimated_source_resolution}")
    >>> print(f"Was upscaled: {result.is_upscaled}")
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
    from scipy import ndimage, signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class UpscaleMethod(Enum):
    """Detected upscaling methods."""
    NONE = "none"               # Not upscaled (native resolution)
    NEAREST = "nearest"         # Nearest neighbor (blocky)
    BILINEAR = "bilinear"       # Bilinear (soft edges)
    BICUBIC = "bicubic"         # Bicubic (ringing artifacts)
    LANCZOS = "lanczos"         # Lanczos (sharp with ringing)
    AI_UPSCALE = "ai_upscale"   # AI upscaling (too-sharp edges)
    UNKNOWN = "unknown"         # Upscaled but method unknown


@dataclass
class Resolution:
    """Video resolution."""
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0


# Common source resolutions to check against
COMMON_RESOLUTIONS = [
    Resolution(320, 240),    # QVGA
    Resolution(352, 288),    # CIF
    Resolution(480, 360),    # 360p
    Resolution(640, 480),    # VGA / 480p
    Resolution(720, 480),    # DVD NTSC
    Resolution(720, 576),    # DVD PAL
    Resolution(854, 480),    # 480p wide
    Resolution(960, 540),    # qHD
    Resolution(1024, 576),   # PAL widescreen
    Resolution(1280, 720),   # 720p HD
    Resolution(1440, 1080),  # HDV
    Resolution(1920, 1080),  # 1080p FHD
    Resolution(2560, 1440),  # 1440p QHD
    Resolution(3840, 2160),  # 4K UHD
]


@dataclass
class UpscaleAnalysis:
    """Complete upscale analysis results."""
    # Resolution info
    container_resolution: Resolution = field(default_factory=lambda: Resolution(0, 0))
    estimated_source_resolution: Resolution = field(default_factory=lambda: Resolution(0, 0))

    # Detection results
    is_upscaled: bool = False
    upscale_factor: float = 1.0
    upscale_method: UpscaleMethod = UpscaleMethod.NONE

    # Confidence and metrics
    confidence: float = 0.0
    edge_sharpness: float = 0.0       # 0-100, higher = sharper
    ringing_score: float = 0.0        # 0-100, higher = more ringing
    block_artifact_score: float = 0.0 # 0-100, higher = more blocking
    frequency_cutoff: float = 0.0     # Normalized frequency where power drops

    # Additional info
    frames_analyzed: int = 0
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "container_resolution": str(self.container_resolution),
            "estimated_source_resolution": str(self.estimated_source_resolution),
            "is_upscaled": self.is_upscaled,
            "upscale_factor": self.upscale_factor,
            "upscale_method": self.upscale_method.value,
            "confidence": self.confidence,
            "edge_sharpness": self.edge_sharpness,
            "ringing_score": self.ringing_score,
            "block_artifact_score": self.block_artifact_score,
            "frames_analyzed": self.frames_analyzed,
            "recommendation": self.recommendation,
        }


class UpscaleDetector:
    """Detects if video content has been upscaled.

    Uses multiple analysis techniques to determine if video
    is at its native resolution or has been upscaled.
    """

    def __init__(self, sample_frames: int = 20):
        """Initialize detector.

        Args:
            sample_frames: Number of frames to analyze
        """
        self.sample_frames = sample_frames

        if not HAS_CV2:
            logger.warning("OpenCV not available - upscale detection limited")

    def analyze(self, video_path: Path) -> UpscaleAnalysis:
        """Analyze video for upscaling.

        Args:
            video_path: Path to video file

        Returns:
            UpscaleAnalysis with detection results
        """
        video_path = Path(video_path)
        result = UpscaleAnalysis()

        if not HAS_CV2:
            logger.warning("OpenCV required for upscale detection")
            return result

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return result

            # Get container resolution
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result.container_resolution = Resolution(width, height)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames
            if total_frames <= self.sample_frames:
                sample_indices = list(range(total_frames))
            else:
                step = total_frames // self.sample_frames
                sample_indices = [i * step for i in range(self.sample_frames)]

            # Collect metrics from each frame
            edge_sharpness_list = []
            ringing_scores = []
            block_scores = []
            frequency_cutoffs = []

            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Analyze frame
                edge_sharpness_list.append(self._analyze_edge_sharpness(gray))
                ringing_scores.append(self._detect_ringing(gray))
                block_scores.append(self._detect_block_artifacts(gray))

                if HAS_SCIPY:
                    frequency_cutoffs.append(self._analyze_frequency_content(gray))

            cap.release()

            result.frames_analyzed = len(edge_sharpness_list)

            if result.frames_analyzed == 0:
                return result

            # Aggregate metrics
            result.edge_sharpness = np.median(edge_sharpness_list)
            result.ringing_score = np.median(ringing_scores)
            result.block_artifact_score = np.median(block_scores)

            if frequency_cutoffs:
                result.frequency_cutoff = np.median(frequency_cutoffs)

            # Determine if upscaled and estimate source resolution
            self._determine_upscale_status(result)

            return result

        except Exception as e:
            logger.error(f"Upscale detection failed: {e}")
            return result

    def _analyze_edge_sharpness(self, gray: np.ndarray) -> float:
        """Analyze edge sharpness to detect interpolation.

        Upscaled content often has softer edges (bilinear/bicubic)
        or unnaturally sharp edges (AI upscaling).

        Args:
            gray: Grayscale frame

        Returns:
            Edge sharpness score (0-100)
        """
        # Compute gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Edge sharpness is measured by the ratio of strong edges to weak edges
        strong_edges = np.sum(gradient_mag > 50)
        weak_edges = np.sum((gradient_mag > 10) & (gradient_mag <= 50))

        if weak_edges > 0:
            sharpness_ratio = strong_edges / weak_edges
        else:
            sharpness_ratio = 0

        # Scale to 0-100
        sharpness = min(100, sharpness_ratio * 20)

        return sharpness

    def _detect_ringing(self, gray: np.ndarray) -> float:
        """Detect ringing artifacts around edges.

        Ringing (Gibbs phenomenon) is common in Lanczos and
        some bicubic upscaling.

        Args:
            gray: Grayscale frame

        Returns:
            Ringing score (0-100)
        """
        # Find strong edges
        edges = cv2.Canny(gray, 100, 200)

        # Dilate edges to create region around edges
        kernel = np.ones((7, 7), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2)
        edge_region = edge_region > 0

        # Subtract original edges to get area near edges
        near_edge = edge_region & (edges == 0)

        if np.sum(near_edge) < 100:
            return 0

        # Check for oscillation pattern near edges (ringing)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Ringing appears as alternating light/dark bands
        near_edge_laplacian = laplacian[near_edge]

        # High variance in Laplacian near edges indicates ringing
        ringing_variance = np.std(near_edge_laplacian)

        # Scale to 0-100
        ringing_score = min(100, ringing_variance / 2)

        return ringing_score

    def _detect_block_artifacts(self, gray: np.ndarray) -> float:
        """Detect block artifacts from nearest-neighbor upscaling.

        Nearest-neighbor creates visible block patterns.

        Args:
            gray: Grayscale frame

        Returns:
            Block artifact score (0-100)
        """
        h, w = gray.shape

        # Look for periodic patterns at common upscale factors
        scores = []

        for factor in [2, 3, 4]:
            if w < factor * 8 or h < factor * 8:
                continue

            # Sample block boundaries
            block_diff_h = []
            block_diff_v = []

            # Check horizontal block boundaries
            for x in range(factor, w - factor, factor):
                col_left = gray[:, x - 1].astype(np.float64)
                col_right = gray[:, x].astype(np.float64)
                diff = np.mean(np.abs(col_left - col_right))
                block_diff_v.append(diff)

            # Check vertical block boundaries
            for y in range(factor, h - factor, factor):
                row_above = gray[y - 1, :].astype(np.float64)
                row_below = gray[y, :].astype(np.float64)
                diff = np.mean(np.abs(row_above - row_below))
                block_diff_h.append(diff)

            # Compare block boundary differences to average differences
            avg_h_diff = np.mean(block_diff_h) if block_diff_h else 0
            avg_v_diff = np.mean(block_diff_v) if block_diff_v else 0

            # Calculate random sample differences for comparison
            random_diffs = []
            for _ in range(20):
                rx = np.random.randint(1, w - 1)
                ry = np.random.randint(1, h - 1)
                diff = abs(float(gray[ry, rx]) - float(gray[ry, rx - 1]))
                random_diffs.append(diff)

            avg_random = np.mean(random_diffs) if random_diffs else 1

            # If block boundaries have higher differences than random, likely blocked
            if avg_random > 0:
                block_score = ((avg_h_diff + avg_v_diff) / 2) / avg_random
                scores.append(max(0, (block_score - 1) * 50))

        if scores:
            return min(100, max(scores))

        return 0

    def _analyze_frequency_content(self, gray: np.ndarray) -> float:
        """Analyze frequency content to estimate native resolution.

        Upscaled content has a frequency cutoff below the Nyquist
        frequency of the container resolution.

        Args:
            gray: Grayscale frame

        Returns:
            Normalized frequency cutoff (0-1, where 1 = full resolution)
        """
        if not HAS_SCIPY:
            return 1.0

        from scipy import fft

        # Compute 2D FFT
        f_transform = fft.fft2(gray.astype(np.float64))
        f_shift = fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Compute radial average of power spectrum
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create radial bins
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        max_dist = np.sqrt(crow ** 2 + ccol ** 2)

        # Compute radial power profile
        num_bins = min(rows, cols) // 2
        radial_power = np.zeros(num_bins)
        counts = np.zeros(num_bins)

        for i in range(rows):
            for j in range(cols):
                r = int(dist[i, j] * num_bins / max_dist)
                if r < num_bins:
                    radial_power[r] += magnitude[i, j]
                    counts[r] += 1

        # Normalize
        radial_power = radial_power / (counts + 1e-10)

        # Find where power drops significantly (80% drop from low freq)
        if len(radial_power) > 10:
            low_freq_power = np.mean(radial_power[1:5])  # Skip DC
            threshold = low_freq_power * 0.2  # 80% drop

            cutoff_bin = num_bins
            for i in range(5, num_bins):
                if radial_power[i] < threshold:
                    cutoff_bin = i
                    break

            return cutoff_bin / num_bins

        return 1.0

    def _determine_upscale_status(self, result: UpscaleAnalysis) -> None:
        """Determine if video is upscaled and estimate source resolution.

        Args:
            result: UpscaleAnalysis to update
        """
        container = result.container_resolution

        # Indicators of upscaling:
        # 1. Soft edges (low sharpness) + no ringing = bilinear
        # 2. Moderate sharpness + ringing = bicubic/lanczos
        # 3. Very high sharpness + some ringing = AI upscale
        # 4. Block artifacts = nearest neighbor
        # 5. Low frequency cutoff = any upscaling

        upscale_indicators = 0
        confidence_factors = []

        # Check edge sharpness
        if result.edge_sharpness < 30:
            # Very soft edges - likely bilinear upscale
            upscale_indicators += 2
            confidence_factors.append(0.7)
            result.upscale_method = UpscaleMethod.BILINEAR
        elif result.edge_sharpness > 80:
            # Unnaturally sharp - might be AI upscaled
            if result.ringing_score < 20:
                upscale_indicators += 1
                confidence_factors.append(0.5)
                result.upscale_method = UpscaleMethod.AI_UPSCALE

        # Check ringing
        if result.ringing_score > 40:
            upscale_indicators += 1
            confidence_factors.append(0.6)
            if result.upscale_method == UpscaleMethod.NONE:
                result.upscale_method = UpscaleMethod.LANCZOS

        # Check block artifacts
        if result.block_artifact_score > 30:
            upscale_indicators += 2
            confidence_factors.append(0.8)
            result.upscale_method = UpscaleMethod.NEAREST

        # Check frequency cutoff
        if result.frequency_cutoff < 0.7:
            upscale_indicators += 2
            confidence_factors.append(0.9 - result.frequency_cutoff)

            # Estimate source resolution from cutoff
            estimated_factor = 1.0 / result.frequency_cutoff if result.frequency_cutoff > 0 else 2
            result.upscale_factor = min(8, estimated_factor)

        # Determine if upscaled
        result.is_upscaled = upscale_indicators >= 2

        if result.is_upscaled:
            # Calculate confidence
            if confidence_factors:
                result.confidence = min(1.0, np.mean(confidence_factors))
            else:
                result.confidence = 0.5

            # Estimate source resolution
            if result.upscale_factor > 1.1:
                est_width = int(container.width / result.upscale_factor)
                est_height = int(container.height / result.upscale_factor)

                # Snap to common resolution
                result.estimated_source_resolution = self._snap_to_common_resolution(
                    est_width, est_height, container
                )
            else:
                # Small upscale or couldn't determine
                result.estimated_source_resolution = container
                result.upscale_factor = 1.0

            # Generate recommendation
            result.recommendation = self._generate_recommendation(result)
        else:
            result.confidence = 0.8
            result.upscale_factor = 1.0
            result.estimated_source_resolution = container
            result.upscale_method = UpscaleMethod.NONE
            result.recommendation = "Video appears to be at native resolution. Safe to upscale."

    def _snap_to_common_resolution(
        self,
        width: int,
        height: int,
        container: Resolution,
    ) -> Resolution:
        """Snap estimated resolution to nearest common resolution.

        Args:
            width: Estimated width
            height: Estimated height
            container: Container resolution

        Returns:
            Nearest common Resolution
        """
        estimated = Resolution(width, height)
        estimated_pixels = estimated.pixels

        # Find closest common resolution that's smaller than container
        best_match = estimated
        best_distance = float('inf')

        for common in COMMON_RESOLUTIONS:
            if common.pixels >= container.pixels:
                continue  # Skip resolutions >= container

            # Match by aspect ratio and total pixels
            aspect_diff = abs(common.aspect_ratio - container.aspect_ratio)
            if aspect_diff > 0.1:  # Aspect ratio too different
                continue

            pixel_diff = abs(common.pixels - estimated_pixels)
            if pixel_diff < best_distance:
                best_distance = pixel_diff
                best_match = common

        return best_match

    def _generate_recommendation(self, result: UpscaleAnalysis) -> str:
        """Generate recommendation based on analysis.

        Args:
            result: Analysis result

        Returns:
            Recommendation string
        """
        source = result.estimated_source_resolution
        container = result.container_resolution

        if result.upscale_method == UpscaleMethod.AI_UPSCALE:
            return (
                f"Video appears to be AI-upscaled from ~{source}. "
                f"Re-upscaling may produce artifacts. Consider processing at {source} "
                f"or using the original source if available."
            )

        if result.upscale_method == UpscaleMethod.NEAREST:
            return (
                f"Video was upscaled using nearest-neighbor from ~{source}. "
                f"Downscale to {source} before processing for best results."
            )

        if result.upscale_factor >= 2:
            return (
                f"Video was upscaled ~{result.upscale_factor:.1f}x from ~{source}. "
                f"Consider downscaling to {source} before restoration, "
                f"then upscale with AI for better quality."
            )

        return (
            f"Video may have been slightly upscaled from ~{source}. "
            f"Current resolution: {container}. Consider your target output resolution."
        )


def detect_upscaling(video_path: Path) -> UpscaleAnalysis:
    """Convenience function to detect upscaling.

    Args:
        video_path: Path to video file

    Returns:
        UpscaleAnalysis with results
    """
    detector = UpscaleDetector()
    return detector.analyze(video_path)
