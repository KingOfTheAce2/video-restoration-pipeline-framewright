"""Degradation Detector - Comprehensive video degradation analysis.

Detects and classifies various types of video degradation including
noise, blur, compression artifacts, film damage, and analog artifacts.

Features:
- Multiple degradation type detection
- Severity and confidence scoring
- Location mapping for spatial degradations
- Full video analysis with temporal tracking
- Processor recommendations based on detected issues

Example:
    >>> detector = DegradationDetector()
    >>> degradations = detector.detect(frame)
    >>> for d in degradations:
    ...     print(f"{d.type.name}: severity={d.severity:.2f}, confidence={d.confidence:.2f}")
    >>>
    >>> # Get recommended processors
    >>> recommendations = detector.suggest_processors(degradations)
    >>> print(f"Suggested pipeline: {recommendations}")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available - degradation detection limited")

try:
    from scipy import ndimage, signal, fft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Enums
# =============================================================================

class DegradationType(Enum):
    """Types of video degradation."""
    # Noise-related
    NOISE = auto()                    # General noise
    GAUSSIAN_NOISE = auto()           # Random additive noise
    SALT_PEPPER_NOISE = auto()        # Impulse noise
    FILM_GRAIN = auto()               # Organic film grain

    # Blur-related
    BLUR = auto()                     # General blur
    MOTION_BLUR = auto()              # Directional motion blur
    OUT_OF_FOCUS = auto()             # Focus blur

    # Compression-related
    COMPRESSION_ARTIFACTS = auto()    # General compression issues
    BLOCKING = auto()                 # Block/mosaic artifacts
    MOSQUITO_NOISE = auto()           # Ringing near edges
    BANDING = auto()                  # Color banding in gradients

    # Film damage
    SCRATCHES = auto()                # Vertical/horizontal scratches
    DUST = auto()                     # Dust and debris spots
    FILM_DAMAGE = auto()              # General film damage

    # Temporal issues
    INTERLACING = auto()              # Interlacing combing
    TELECINE = auto()                 # 3:2 pulldown patterns
    FLICKER = auto()                  # Brightness flickering

    # Color issues
    COLOR_FADE = auto()               # Faded colors
    COLOR_CAST = auto()               # Color tint/cast
    VIGNETTE = auto()                 # Dark corners

    # Analog artifacts (VHS, etc.)
    VHS_ARTIFACTS = auto()            # General VHS degradation
    TRACKING_ERRORS = auto()          # VHS tracking lines
    HEAD_SWITCHING = auto()           # VHS head switching noise
    CHROMA_BLEEDING = auto()          # Color bleeding/smearing
    DROPOUT = auto()                  # Signal dropout lines


class SeverityLevel(Enum):
    """Severity levels for degradation."""
    MINIMAL = auto()     # Barely noticeable
    LIGHT = auto()       # Minor but visible
    MODERATE = auto()    # Clearly visible
    HEAVY = auto()       # Significant degradation
    SEVERE = auto()      # Major quality impact


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DegradationRegion:
    """Region where degradation is detected."""
    x: int
    y: int
    width: int
    height: int
    intensity: float  # 0-1 intensity within this region

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def as_slice(self) -> Tuple[slice, slice]:
        """Return as numpy slice."""
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))


@dataclass
class DegradationInfo:
    """Information about a detected degradation.

    Attributes:
        type: Type of degradation detected
        severity: Severity level (0-1, higher is worse)
        location: Where the degradation occurs ("global", "local", or region list)
        confidence: Detection confidence (0-1)
        details: Additional details about the degradation
        regions: List of affected regions (for local degradations)
    """
    type: DegradationType
    severity: float  # 0-1
    location: str = "global"  # "global", "local", or description
    confidence: float = 1.0  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)
    regions: List[DegradationRegion] = field(default_factory=list)

    @property
    def severity_level(self) -> SeverityLevel:
        """Get severity as enum level."""
        if self.severity < 0.1:
            return SeverityLevel.MINIMAL
        elif self.severity < 0.3:
            return SeverityLevel.LIGHT
        elif self.severity < 0.5:
            return SeverityLevel.MODERATE
        elif self.severity < 0.7:
            return SeverityLevel.HEAVY
        return SeverityLevel.SEVERE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.name,
            "severity": round(self.severity, 3),
            "severity_level": self.severity_level.name,
            "location": self.location,
            "confidence": round(self.confidence, 3),
            "details": self.details,
            "region_count": len(self.regions),
        }


@dataclass
class VideoAnalysisResult:
    """Result of full video degradation analysis."""
    frame_count: int = 0
    analyzed_frames: int = 0
    fps: float = 0.0
    resolution: Tuple[int, int] = (0, 0)

    # Detected degradations with frame ranges
    degradations: List[DegradationInfo] = field(default_factory=list)
    temporal_issues: List[DegradationInfo] = field(default_factory=list)

    # Per-frame degradation map
    frame_degradations: Dict[int, List[DegradationInfo]] = field(default_factory=dict)

    # Summary
    primary_issues: List[DegradationType] = field(default_factory=list)
    overall_quality: str = "unknown"  # "good", "moderate", "poor"
    recommendations: List[str] = field(default_factory=list)

    def get_all_degradation_types(self) -> List[DegradationType]:
        """Get list of all detected degradation types."""
        types = set()
        for d in self.degradations:
            types.add(d.type)
        for d in self.temporal_issues:
            types.add(d.type)
        return list(types)


@dataclass
class ProcessorRecommendation:
    """Recommended processor for a degradation."""
    processor: str
    priority: int  # Lower = higher priority
    settings: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# =============================================================================
# Degradation Detector
# =============================================================================

class DegradationDetector:
    """Comprehensive video degradation detector.

    Analyzes frames for various types of degradation and provides
    recommendations for restoration processors.
    """

    # Thresholds for detection
    NOISE_THRESHOLD = 15.0
    BLUR_THRESHOLD = 100.0
    BLOCKING_THRESHOLD = 1.3
    SCRATCH_THRESHOLD = 0.02
    INTERLACE_THRESHOLD = 1.5

    def __init__(
        self,
        sensitivity: float = 1.0,
        enable_all_detectors: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize degradation detector.

        Args:
            sensitivity: Detection sensitivity multiplier (0.5-2.0)
            enable_all_detectors: Enable all detection methods
            progress_callback: Optional progress callback
        """
        self.sensitivity = max(0.5, min(2.0, sensitivity))
        self.enable_all_detectors = enable_all_detectors
        self.progress_callback = progress_callback

        # Adjust thresholds based on sensitivity
        self._adjust_thresholds()

        if not HAS_CV2:
            logger.warning("OpenCV required for degradation detection")

    def _adjust_thresholds(self) -> None:
        """Adjust detection thresholds based on sensitivity."""
        # Higher sensitivity = lower thresholds (detect more)
        factor = 2.0 - self.sensitivity
        self.noise_threshold = self.NOISE_THRESHOLD * factor
        self.blur_threshold = self.BLUR_THRESHOLD * factor
        self.blocking_threshold = self.BLOCKING_THRESHOLD * factor
        self.scratch_threshold = self.SCRATCH_THRESHOLD * factor
        self.interlace_threshold = self.INTERLACE_THRESHOLD * factor

    # =========================================================================
    # Main Detection Methods
    # =========================================================================

    def detect(self, frame: np.ndarray) -> List[DegradationInfo]:
        """Detect all degradations in a frame.

        Args:
            frame: BGR frame as numpy array

        Returns:
            List of DegradationInfo for detected issues
        """
        if not HAS_CV2:
            return []

        if frame is None or frame.size == 0:
            return []

        degradations = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Noise detection
        noise_info = self._detect_noise(gray)
        if noise_info:
            degradations.append(noise_info)

        # Blur detection
        blur_info = self._detect_blur(gray)
        if blur_info:
            degradations.append(blur_info)

        # Compression artifacts
        compression_infos = self._detect_compression_artifacts(gray)
        degradations.extend(compression_infos)

        # Film damage (scratches, dust)
        damage_infos = self._detect_film_damage(gray)
        degradations.extend(damage_infos)

        # Interlacing
        interlace_info = self._detect_interlacing(gray)
        if interlace_info:
            degradations.append(interlace_info)

        # Color issues
        color_infos = self._detect_color_issues(frame)
        degradations.extend(color_infos)

        # VHS/Analog artifacts
        if self.enable_all_detectors:
            analog_infos = self._detect_analog_artifacts(frame, gray)
            degradations.extend(analog_infos)

        return degradations

    def detect_type(
        self,
        frame: np.ndarray,
        degradation_type: DegradationType,
    ) -> Optional[DegradationInfo]:
        """Detect a specific type of degradation.

        Args:
            frame: BGR frame
            degradation_type: Type to detect

        Returns:
            DegradationInfo if detected, None otherwise
        """
        if not HAS_CV2:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector_map = {
            DegradationType.NOISE: lambda: self._detect_noise(gray),
            DegradationType.GAUSSIAN_NOISE: lambda: self._detect_noise(gray),
            DegradationType.BLUR: lambda: self._detect_blur(gray),
            DegradationType.MOTION_BLUR: lambda: self._detect_motion_blur(gray),
            DegradationType.COMPRESSION_ARTIFACTS: lambda: self._detect_blocking(gray),
            DegradationType.BLOCKING: lambda: self._detect_blocking(gray),
            DegradationType.SCRATCHES: lambda: self._detect_scratches(gray),
            DegradationType.DUST: lambda: self._detect_dust(gray),
            DegradationType.INTERLACING: lambda: self._detect_interlacing(gray),
            DegradationType.COLOR_FADE: lambda: self._detect_color_fade(frame),
            DegradationType.COLOR_CAST: lambda: self._detect_color_cast(frame),
            DegradationType.VIGNETTE: lambda: self._detect_vignette(gray),
            DegradationType.VHS_ARTIFACTS: lambda: self._detect_vhs_artifacts(frame, gray),
            DegradationType.TRACKING_ERRORS: lambda: self._detect_tracking_errors(gray),
        }

        detector = detector_map.get(degradation_type)
        if detector:
            return detector()

        return None

    def analyze_video(
        self,
        frames: Union[List[np.ndarray], Path],
        sample_rate: int = 10,
        max_frames: Optional[int] = None,
    ) -> VideoAnalysisResult:
        """Analyze an entire video for degradations.

        Args:
            frames: List of frames or path to video file
            sample_rate: Analyze every Nth frame
            max_frames: Maximum frames to analyze

        Returns:
            VideoAnalysisResult with full analysis
        """
        if isinstance(frames, Path):
            return self._analyze_video_file(frames, sample_rate, max_frames)
        return self._analyze_frame_list(frames, sample_rate, max_frames)

    def _analyze_video_file(
        self,
        video_path: Path,
        sample_rate: int,
        max_frames: Optional[int],
    ) -> VideoAnalysisResult:
        """Analyze video from file."""
        if not HAS_CV2:
            return VideoAnalysisResult()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return VideoAnalysisResult()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = VideoAnalysisResult(
            frame_count=total_frames,
            fps=fps,
            resolution=(width, height),
        )

        all_degradations: Dict[DegradationType, List[DegradationInfo]] = {}
        prev_frame = None
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                if max_frames and analyzed >= max_frames:
                    break

                # Detect degradations in frame
                frame_degs = self.detect(frame)
                result.frame_degradations[frame_idx] = frame_degs

                # Aggregate by type
                for deg in frame_degs:
                    if deg.type not in all_degradations:
                        all_degradations[deg.type] = []
                    all_degradations[deg.type].append(deg)

                # Temporal analysis
                if prev_frame is not None:
                    temporal = self._detect_temporal_issues(frame, prev_frame)
                    result.temporal_issues.extend(temporal)

                analyzed += 1
                prev_frame = frame.copy()

                if self.progress_callback:
                    self.progress_callback(analyzed, min(total_frames // sample_rate, max_frames or float('inf')))

            frame_idx += 1

        cap.release()
        result.analyzed_frames = analyzed

        # Aggregate results
        self._aggregate_video_analysis(result, all_degradations)

        return result

    def _analyze_frame_list(
        self,
        frames: List[np.ndarray],
        sample_rate: int,
        max_frames: Optional[int],
    ) -> VideoAnalysisResult:
        """Analyze video from frame list."""
        if not frames:
            return VideoAnalysisResult()

        h, w = frames[0].shape[:2]
        result = VideoAnalysisResult(
            frame_count=len(frames),
            resolution=(w, h),
        )

        all_degradations: Dict[DegradationType, List[DegradationInfo]] = {}
        prev_frame = None
        analyzed = 0

        for i, frame in enumerate(frames):
            if i % sample_rate != 0:
                continue

            if max_frames and analyzed >= max_frames:
                break

            frame_degs = self.detect(frame)
            result.frame_degradations[i] = frame_degs

            for deg in frame_degs:
                if deg.type not in all_degradations:
                    all_degradations[deg.type] = []
                all_degradations[deg.type].append(deg)

            if prev_frame is not None:
                temporal = self._detect_temporal_issues(frame, prev_frame)
                result.temporal_issues.extend(temporal)

            analyzed += 1
            prev_frame = frame

            if self.progress_callback:
                self.progress_callback(analyzed, min(len(frames) // sample_rate, max_frames or float('inf')))

        result.analyzed_frames = analyzed
        self._aggregate_video_analysis(result, all_degradations)

        return result

    def _aggregate_video_analysis(
        self,
        result: VideoAnalysisResult,
        all_degradations: Dict[DegradationType, List[DegradationInfo]],
    ) -> None:
        """Aggregate frame-level results into video-level summary."""
        # Average degradations by type
        for deg_type, instances in all_degradations.items():
            avg_severity = np.mean([d.severity for d in instances])
            avg_confidence = np.mean([d.confidence for d in instances])
            occurrence_rate = len(instances) / max(1, result.analyzed_frames)

            # Only include if occurs frequently enough
            if occurrence_rate > 0.1 or avg_severity > 0.5:
                result.degradations.append(DegradationInfo(
                    type=deg_type,
                    severity=avg_severity,
                    confidence=avg_confidence * occurrence_rate,
                    location="global",
                    details={
                        "occurrence_rate": round(occurrence_rate, 2),
                        "instance_count": len(instances),
                    },
                ))

        # Sort by severity
        result.degradations.sort(key=lambda d: d.severity, reverse=True)

        # Determine primary issues
        result.primary_issues = [
            d.type for d in result.degradations
            if d.severity > 0.3 and d.confidence > 0.5
        ][:5]

        # Overall quality assessment
        if not result.degradations:
            result.overall_quality = "good"
        else:
            max_severity = max(d.severity for d in result.degradations)
            if max_severity < 0.3:
                result.overall_quality = "good"
            elif max_severity < 0.6:
                result.overall_quality = "moderate"
            else:
                result.overall_quality = "poor"

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result.degradations)

    # =========================================================================
    # Individual Degradation Detectors
    # =========================================================================

    def _detect_noise(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect noise in image.

        Uses Laplacian variance method with flat region analysis.

        Args:
            gray: Grayscale image

        Returns:
            DegradationInfo if noise detected
        """
        # Estimate noise using median absolute deviation of Laplacian
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_map = cv2.absdiff(gray, blurred)

        # Use MAD for robust estimation
        median = np.median(noise_map)
        mad = np.median(np.abs(noise_map - median))
        sigma = mad / 0.6745

        if sigma < self.noise_threshold * 0.3:
            return None

        # Determine noise type
        noise_type = DegradationType.GAUSSIAN_NOISE

        # Check for salt and pepper (high local maxima/minima)
        local_max = ndimage.maximum_filter(gray, size=3) if HAS_SCIPY else gray
        local_min = ndimage.minimum_filter(gray, size=3) if HAS_SCIPY else gray
        extrema = np.sum((gray == local_max) | (gray == local_min)) / gray.size

        if extrema > 0.1:
            noise_type = DegradationType.SALT_PEPPER_NOISE

        # Check for film grain (uniform organic texture)
        if HAS_SCIPY and self._is_film_grain(noise_map):
            noise_type = DegradationType.FILM_GRAIN

        severity = min(1.0, sigma / (self.noise_threshold * 2))
        confidence = min(1.0, sigma / self.noise_threshold)

        return DegradationInfo(
            type=noise_type,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "estimated_sigma": round(sigma, 2),
                "mad": round(mad, 2),
            },
        )

    def _is_film_grain(self, noise_map: np.ndarray) -> bool:
        """Check if noise pattern resembles film grain."""
        # Film grain has uniform distribution and organic texture
        block_size = 64
        h, w = noise_map.shape
        block_stds = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_map[y:y + block_size, x:x + block_size]
                block_stds.append(np.std(block))

        if not block_stds:
            return False

        # Uniform grain has consistent std across blocks
        std_of_stds = np.std(block_stds)
        mean_std = np.mean(block_stds)

        if mean_std > 0:
            coefficient_of_variation = std_of_stds / mean_std
            return coefficient_of_variation < 0.3

        return False

    def _detect_blur(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect blur using Laplacian variance.

        Args:
            gray: Grayscale image

        Returns:
            DegradationInfo if blur detected
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        if variance > self.blur_threshold:
            return None

        severity = 1.0 - min(1.0, variance / self.blur_threshold)
        confidence = min(1.0, (self.blur_threshold - variance) / self.blur_threshold)

        # Try to determine blur type
        blur_type = DegradationType.BLUR

        # Check for motion blur (directional)
        if self._detect_motion_blur_direction(gray) is not None:
            blur_type = DegradationType.MOTION_BLUR

        return DegradationInfo(
            type=blur_type,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "laplacian_variance": round(variance, 2),
                "threshold": round(self.blur_threshold, 2),
            },
        )

    def _detect_motion_blur(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect motion blur specifically."""
        direction = self._detect_motion_blur_direction(gray)
        if direction is None:
            return None

        # Calculate blur severity from Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        severity = 1.0 - min(1.0, laplacian_var / self.blur_threshold)

        return DegradationInfo(
            type=DegradationType.MOTION_BLUR,
            severity=severity,
            confidence=0.7,
            location="global",
            details={
                "direction_degrees": round(direction, 1),
            },
        )

    def _detect_motion_blur_direction(self, gray: np.ndarray) -> Optional[float]:
        """Detect direction of motion blur using FFT.

        Returns:
            Blur direction in degrees, or None if not detected
        """
        if not HAS_SCIPY:
            return None

        # Compute FFT
        f_transform = fft.fft2(gray.astype(np.float64))
        f_shift = fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)

        # Motion blur creates a line in FFT perpendicular to motion direction
        # Use Radon-like approach: project along different angles

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Sample radius
        radius = min(h, w) // 4

        best_angle = None
        best_variance = 0

        for angle in range(0, 180, 5):
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)

            # Sample points along line
            samples = []
            for r in range(-radius, radius):
                x = int(cx + r * cos_a)
                y = int(cy + r * sin_a)
                if 0 <= x < w and 0 <= y < h:
                    samples.append(magnitude[y, x])

            if samples:
                var = np.var(samples)
                if var > best_variance:
                    best_variance = var
                    best_angle = angle

        # Motion blur direction is perpendicular to the detected line
        if best_angle is not None and best_variance > np.mean(magnitude) * 2:
            return (best_angle + 90) % 180

        return None

    def _detect_compression_artifacts(self, gray: np.ndarray) -> List[DegradationInfo]:
        """Detect compression artifacts."""
        results = []

        # Blocking artifacts
        blocking_info = self._detect_blocking(gray)
        if blocking_info:
            results.append(blocking_info)

        # Mosquito noise / ringing
        mosquito_info = self._detect_mosquito_noise(gray)
        if mosquito_info:
            results.append(mosquito_info)

        # Banding
        banding_info = self._detect_banding(gray)
        if banding_info:
            results.append(banding_info)

        return results

    def _detect_blocking(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect blocking artifacts."""
        h, w = gray.shape

        if h < 32 or w < 32:
            return None

        # Calculate differences at 8-pixel intervals
        h_block_diff = []
        v_block_diff = []

        for i in range(8, h - 8, 8):
            diff = np.mean(np.abs(gray[i, :].astype(float) - gray[i - 1, :].astype(float)))
            h_block_diff.append(diff)

        for j in range(8, w - 8, 8):
            diff = np.mean(np.abs(gray[:, j].astype(float) - gray[:, j - 1].astype(float)))
            v_block_diff.append(diff)

        if not h_block_diff or not v_block_diff:
            return None

        avg_block_diff = (np.mean(h_block_diff) + np.mean(v_block_diff)) / 2

        # Compare to non-boundary differences
        non_block_diffs = []
        for i in range(4, h - 4, 8):
            for offset in [1, 2, 3]:
                if i + offset < h - 1:
                    diff = np.mean(np.abs(
                        gray[i + offset, :].astype(float) - gray[i + offset - 1, :].astype(float)
                    ))
                    non_block_diffs.append(diff)

        if not non_block_diffs:
            return None

        avg_non_block = np.mean(non_block_diffs)

        if avg_non_block > 0:
            blocking_ratio = avg_block_diff / avg_non_block
        else:
            blocking_ratio = 1.0

        if blocking_ratio < self.blocking_threshold:
            return None

        severity = min(1.0, (blocking_ratio - 1.0) / 2.0)
        confidence = min(1.0, (blocking_ratio - self.blocking_threshold) / self.blocking_threshold)

        return DegradationInfo(
            type=DegradationType.BLOCKING,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "blocking_ratio": round(blocking_ratio, 2),
                "avg_block_diff": round(avg_block_diff, 2),
            },
        )

    def _detect_mosquito_noise(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect mosquito noise (ringing near edges)."""
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)

        # Dilate to get edge region
        kernel = np.ones((7, 7), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2) > 0

        # Compute Laplacian
        laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

        # Compare variance near edges vs away from edges
        if np.sum(edge_region) < 100 or np.sum(~edge_region) < 100:
            return None

        edge_var = np.var(laplacian[edge_region])
        non_edge_var = np.var(laplacian[~edge_region])

        if non_edge_var > 0:
            ratio = edge_var / non_edge_var
        else:
            ratio = 1.0

        if ratio < 1.5:
            return None

        severity = min(1.0, (ratio - 1.0) / 3.0)
        confidence = min(1.0, (ratio - 1.5) / 2.0)

        return DegradationInfo(
            type=DegradationType.MOSQUITO_NOISE,
            severity=severity,
            confidence=confidence,
            location="edges",
            details={
                "edge_variance_ratio": round(ratio, 2),
            },
        )

    def _detect_banding(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect color banding in gradients."""
        # Look for large flat regions with sharp transitions
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        # Strong edges should be sparse for banding
        strong_edges = np.sum(np.abs(gradient) > 30)
        weak_edges = np.sum((np.abs(gradient) > 5) & (np.abs(gradient) <= 30))

        if weak_edges == 0:
            return None

        banding_ratio = strong_edges / weak_edges

        if banding_ratio < 0.5:
            return None

        severity = min(1.0, banding_ratio / 4.0)
        confidence = min(1.0, banding_ratio / 2.0)

        return DegradationInfo(
            type=DegradationType.BANDING,
            severity=severity,
            confidence=confidence,
            location="gradients",
            details={
                "banding_ratio": round(banding_ratio, 2),
            },
        )

    def _detect_film_damage(self, gray: np.ndarray) -> List[DegradationInfo]:
        """Detect film damage (scratches, dust)."""
        results = []

        scratches = self._detect_scratches(gray)
        if scratches:
            results.append(scratches)

        dust = self._detect_dust(gray)
        if dust:
            results.append(dust)

        return results

    def _detect_scratches(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect vertical/horizontal scratches."""
        h, w = gray.shape

        # Detect vertical lines (scratches are often vertical)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        # Sum along columns
        col_variance = np.var(sobel_x, axis=0)

        # Scratches appear as columns with very high variance
        threshold = np.mean(col_variance) + 3 * np.std(col_variance)
        scratch_cols = np.sum(col_variance > threshold)

        scratch_ratio = scratch_cols / w

        if scratch_ratio < self.scratch_threshold:
            return None

        severity = min(1.0, scratch_ratio / (self.scratch_threshold * 5))
        confidence = min(1.0, scratch_ratio / self.scratch_threshold)

        # Find scratch locations
        regions = []
        for i, var in enumerate(col_variance):
            if var > threshold:
                regions.append(DegradationRegion(
                    x=i, y=0, width=1, height=h,
                    intensity=min(1.0, var / threshold),
                ))

        return DegradationInfo(
            type=DegradationType.SCRATCHES,
            severity=severity,
            confidence=confidence,
            location="local",
            regions=regions[:20],  # Limit to 20 regions
            details={
                "scratch_count": scratch_cols,
                "scratch_ratio": round(scratch_ratio, 4),
            },
        )

    def _detect_dust(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect dust and debris spots."""
        # Dust appears as small isolated bright or dark spots
        # Use morphological operations to detect

        # Detect bright spots (dust on lens)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Detect dark spots
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Combine
        spots = cv2.add(tophat, blackhat)

        # Threshold
        _, binary = cv2.threshold(spots, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by size (dust spots are small)
        dust_spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:  # Small spots only
                x, y, w, h = cv2.boundingRect(contour)
                dust_spots.append(DegradationRegion(
                    x=x, y=y, width=w, height=h,
                    intensity=1.0,
                ))

        if len(dust_spots) < 3:
            return None

        severity = min(1.0, len(dust_spots) / 50)
        confidence = min(1.0, len(dust_spots) / 10)

        return DegradationInfo(
            type=DegradationType.DUST,
            severity=severity,
            confidence=confidence,
            location="local",
            regions=dust_spots[:50],  # Limit regions
            details={
                "dust_spot_count": len(dust_spots),
            },
        )

    def _detect_interlacing(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect interlacing (combing) artifacts."""
        # Calculate difference between even and odd rows
        even_rows = gray[::2, :]
        odd_rows = gray[1::2, :]

        # Ensure same size
        min_h = min(even_rows.shape[0], odd_rows.shape[0])
        even_rows = even_rows[:min_h, :]
        odd_rows = odd_rows[:min_h, :]

        # Row difference
        row_diff = np.mean(np.abs(even_rows.astype(float) - odd_rows.astype(float)))

        # Compare to column difference (should be similar for non-interlaced)
        col_diff = np.mean(np.abs(gray[:, ::2].astype(float) - gray[:, 1::2].astype(float)))

        if col_diff > 0:
            interlace_ratio = row_diff / col_diff
        else:
            interlace_ratio = 1.0

        if interlace_ratio < self.interlace_threshold or row_diff < 10:
            return None

        severity = min(1.0, (interlace_ratio - 1.0) / 2.0)
        confidence = min(1.0, (interlace_ratio - self.interlace_threshold) / self.interlace_threshold)

        return DegradationInfo(
            type=DegradationType.INTERLACING,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "interlace_ratio": round(interlace_ratio, 2),
                "row_diff": round(row_diff, 2),
            },
        )

    def _detect_color_issues(self, frame: np.ndarray) -> List[DegradationInfo]:
        """Detect color-related issues."""
        results = []

        fade = self._detect_color_fade(frame)
        if fade:
            results.append(fade)

        cast = self._detect_color_cast(frame)
        if cast:
            results.append(cast)

        return results

    def _detect_color_fade(self, frame: np.ndarray) -> Optional[DegradationInfo]:
        """Detect faded colors."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        avg_saturation = np.mean(saturation)

        # Low saturation indicates faded colors
        if avg_saturation > 80:  # Normal saturation
            return None

        severity = 1.0 - (avg_saturation / 80)
        confidence = max(0, 1.0 - (avg_saturation / 60))

        return DegradationInfo(
            type=DegradationType.COLOR_FADE,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "avg_saturation": round(avg_saturation, 2),
            },
        )

    def _detect_color_cast(self, frame: np.ndarray) -> Optional[DegradationInfo]:
        """Detect color cast/tint."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # A and B channels should average around 128 for neutral
        a_mean = np.mean(lab[:, :, 1])
        b_mean = np.mean(lab[:, :, 2])

        # Distance from neutral
        cast_distance = np.sqrt((a_mean - 128) ** 2 + (b_mean - 128) ** 2)

        if cast_distance < 10:  # Acceptable range
            return None

        severity = min(1.0, cast_distance / 50)
        confidence = min(1.0, cast_distance / 20)

        # Determine cast color
        cast_color = "unknown"
        if a_mean > 135:
            cast_color = "magenta" if b_mean > 128 else "red"
        elif a_mean < 121:
            cast_color = "cyan" if b_mean > 128 else "green"
        elif b_mean > 135:
            cast_color = "yellow"
        elif b_mean < 121:
            cast_color = "blue"

        return DegradationInfo(
            type=DegradationType.COLOR_CAST,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "a_mean": round(a_mean, 2),
                "b_mean": round(b_mean, 2),
                "cast_distance": round(cast_distance, 2),
                "cast_color": cast_color,
            },
        )

    def _detect_vignette(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect vignetting (dark corners)."""
        h, w = gray.shape

        # Sample brightness in corners vs center
        corner_size = min(h, w) // 8

        corners = [
            gray[:corner_size, :corner_size],  # Top-left
            gray[:corner_size, -corner_size:],  # Top-right
            gray[-corner_size:, :corner_size],  # Bottom-left
            gray[-corner_size:, -corner_size:],  # Bottom-right
        ]

        center = gray[h // 3:2 * h // 3, w // 3:2 * w // 3]

        avg_corners = np.mean([np.mean(c) for c in corners])
        avg_center = np.mean(center)

        if avg_center == 0:
            return None

        vignette_ratio = avg_corners / avg_center

        if vignette_ratio > 0.85:  # Corners not significantly darker
            return None

        severity = 1.0 - vignette_ratio
        confidence = min(1.0, (0.85 - vignette_ratio) / 0.3)

        return DegradationInfo(
            type=DegradationType.VIGNETTE,
            severity=severity,
            confidence=confidence,
            location="corners",
            details={
                "corner_brightness": round(avg_corners, 2),
                "center_brightness": round(avg_center, 2),
                "vignette_ratio": round(vignette_ratio, 3),
            },
        )

    def _detect_analog_artifacts(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> List[DegradationInfo]:
        """Detect VHS and analog video artifacts."""
        results = []

        vhs = self._detect_vhs_artifacts(frame, gray)
        if vhs:
            results.append(vhs)

        tracking = self._detect_tracking_errors(gray)
        if tracking:
            results.append(tracking)

        chroma = self._detect_chroma_bleeding(frame)
        if chroma:
            results.append(chroma)

        return results

    def _detect_vhs_artifacts(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[DegradationInfo]:
        """Detect general VHS artifacts."""
        # VHS characteristics: low resolution, color bleeding, noise
        # Check for combination of factors

        indicators = 0
        total_checks = 4

        # 1. Check for horizontal noise bands
        row_stds = np.std(gray, axis=1)
        noise_bands = np.sum(np.abs(np.diff(row_stds)) > 10)
        if noise_bands > gray.shape[0] * 0.1:
            indicators += 1

        # 2. Check for color bleeding (high chroma spread)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_blur = cv2.GaussianBlur(hsv[:, :, 0], (15, 1), 0)
        h_diff = np.mean(np.abs(hsv[:, :, 0].astype(float) - h_blur.astype(float)))
        if h_diff > 10:
            indicators += 1

        # 3. Check for low sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 200:
            indicators += 1

        # 4. Check for limited color range
        unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
        max_colors = frame.shape[0] * frame.shape[1]
        if unique_colors / max_colors < 0.01:
            indicators += 1

        if indicators < 2:
            return None

        severity = indicators / total_checks
        confidence = min(1.0, indicators / 3)

        return DegradationInfo(
            type=DegradationType.VHS_ARTIFACTS,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "indicator_count": indicators,
            },
        )

    def _detect_tracking_errors(self, gray: np.ndarray) -> Optional[DegradationInfo]:
        """Detect VHS tracking error lines."""
        h, w = gray.shape

        # Tracking errors appear as horizontal bands of distortion
        # Usually near top or bottom of frame

        # Check top and bottom regions
        top_region = gray[:h // 8, :]
        bottom_region = gray[-h // 8:, :]
        middle_region = gray[h // 3:2 * h // 3, :]

        # Calculate row-wise variance
        top_variance = np.mean(np.var(top_region, axis=1))
        bottom_variance = np.mean(np.var(bottom_region, axis=1))
        middle_variance = np.mean(np.var(middle_region, axis=1))

        max_edge_var = max(top_variance, bottom_variance)

        if middle_variance > 0:
            tracking_ratio = max_edge_var / middle_variance
        else:
            tracking_ratio = 1.0

        if tracking_ratio < 2.0:
            return None

        severity = min(1.0, (tracking_ratio - 1.0) / 4.0)
        confidence = min(1.0, (tracking_ratio - 2.0) / 3.0)

        location = "top" if top_variance > bottom_variance else "bottom"

        return DegradationInfo(
            type=DegradationType.TRACKING_ERRORS,
            severity=severity,
            confidence=confidence,
            location=location,
            details={
                "tracking_ratio": round(tracking_ratio, 2),
            },
        )

    def _detect_chroma_bleeding(self, frame: np.ndarray) -> Optional[DegradationInfo]:
        """Detect chroma bleeding/smearing."""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]

        # Chroma bleeding: color extends beyond luminance edges
        y_edges = cv2.Canny(y_channel, 50, 150)
        cr_edges = cv2.Canny(cr_channel, 30, 100)
        cb_edges = cv2.Canny(cb_channel, 30, 100)

        # Chroma edges should align with luma edges
        y_edge_count = np.sum(y_edges > 0)
        cr_edge_count = np.sum(cr_edges > 0)
        cb_edge_count = np.sum(cb_edges > 0)

        if y_edge_count == 0:
            return None

        chroma_ratio = (cr_edge_count + cb_edge_count) / (2 * y_edge_count)

        # Low ratio means chroma is bleeding (fewer distinct chroma edges)
        if chroma_ratio > 0.3:
            return None

        severity = 1.0 - (chroma_ratio / 0.3)
        confidence = 1.0 - chroma_ratio

        return DegradationInfo(
            type=DegradationType.CHROMA_BLEEDING,
            severity=severity,
            confidence=confidence,
            location="global",
            details={
                "chroma_ratio": round(chroma_ratio, 3),
            },
        )

    def _detect_temporal_issues(
        self,
        current: np.ndarray,
        previous: np.ndarray,
    ) -> List[DegradationInfo]:
        """Detect temporal issues between frames."""
        results = []

        # Flicker detection
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

        curr_brightness = np.mean(curr_gray)
        prev_brightness = np.mean(prev_gray)

        brightness_change = abs(curr_brightness - prev_brightness)

        if brightness_change > 10:  # Significant brightness change
            severity = min(1.0, brightness_change / 50)
            results.append(DegradationInfo(
                type=DegradationType.FLICKER,
                severity=severity,
                confidence=min(1.0, brightness_change / 20),
                location="global",
                details={
                    "brightness_change": round(brightness_change, 2),
                },
            ))

        return results

    # =========================================================================
    # Processor Recommendations
    # =========================================================================

    def suggest_processors(
        self,
        degradations: List[DegradationInfo],
    ) -> List[ProcessorRecommendation]:
        """Suggest processors based on detected degradations.

        Args:
            degradations: List of detected degradations

        Returns:
            List of ProcessorRecommendation in suggested order
        """
        recommendations = []

        for deg in degradations:
            recs = self._get_recommendations_for_type(deg)
            recommendations.extend(recs)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        # Remove duplicates (keep highest priority)
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec.processor not in seen:
                seen.add(rec.processor)
                unique_recs.append(rec)

        return unique_recs

    def _get_recommendations_for_type(
        self,
        degradation: DegradationInfo,
    ) -> List[ProcessorRecommendation]:
        """Get processor recommendations for a degradation type."""
        recs = []
        deg_type = degradation.type
        severity = degradation.severity

        # Noise
        if deg_type in (DegradationType.NOISE, DegradationType.GAUSSIAN_NOISE):
            recs.append(ProcessorRecommendation(
                processor="temporal_denoise",
                priority=10,
                settings={"strength": severity},
                reason=f"Remove {deg_type.name.lower().replace('_', ' ')}",
            ))
            if severity > 0.5:
                recs.append(ProcessorRecommendation(
                    processor="tap_denoise",
                    priority=11,
                    settings={"strength": severity},
                    reason="Heavy noise requires advanced denoising",
                ))

        elif deg_type == DegradationType.FILM_GRAIN:
            recs.append(ProcessorRecommendation(
                processor="grain_manager",
                priority=15,
                settings={"preserve_grain": True, "strength": severity * 0.5},
                reason="Preserve film grain while reducing noise",
            ))

        # Blur
        elif deg_type in (DegradationType.BLUR, DegradationType.OUT_OF_FOCUS):
            recs.append(ProcessorRecommendation(
                processor="realesrgan",
                priority=20,
                settings={"denoise_strength": 0.3},
                reason="Upscaling can help restore sharpness",
            ))

        elif deg_type == DegradationType.MOTION_BLUR:
            recs.append(ProcessorRecommendation(
                processor="stabilization",
                priority=5,
                settings={},
                reason="Stabilize video to reduce motion blur",
            ))

        # Compression
        elif deg_type in (DegradationType.COMPRESSION_ARTIFACTS, DegradationType.BLOCKING):
            recs.append(ProcessorRecommendation(
                processor="qp_artifact_removal",
                priority=8,
                settings={"strength": severity},
                reason="Remove compression blocking artifacts",
            ))

        elif deg_type == DegradationType.MOSQUITO_NOISE:
            recs.append(ProcessorRecommendation(
                processor="qp_artifact_removal",
                priority=9,
                settings={"edge_filter": True},
                reason="Remove mosquito noise near edges",
            ))

        elif deg_type == DegradationType.BANDING:
            recs.append(ProcessorRecommendation(
                processor="perceptual_tuning",
                priority=25,
                settings={"debanding": True},
                reason="Remove color banding",
            ))

        # Film damage
        elif deg_type == DegradationType.SCRATCHES:
            recs.append(ProcessorRecommendation(
                processor="defect_repair",
                priority=3,
                settings={"scratch_removal": True},
                reason="Remove film scratches",
            ))

        elif deg_type == DegradationType.DUST:
            recs.append(ProcessorRecommendation(
                processor="defect_repair",
                priority=4,
                settings={"dust_removal": True},
                reason="Remove dust and debris",
            ))

        # Interlacing
        elif deg_type == DegradationType.INTERLACING:
            recs.append(ProcessorRecommendation(
                processor="interlace_handler",
                priority=1,
                settings={},
                reason="Deinterlace video",
            ))

        elif deg_type == DegradationType.TELECINE:
            recs.append(ProcessorRecommendation(
                processor="telecine",
                priority=2,
                settings={},
                reason="Apply inverse telecine",
            ))

        # Color issues
        elif deg_type == DegradationType.COLOR_FADE:
            recs.append(ProcessorRecommendation(
                processor="colorization",
                priority=30,
                settings={"saturation_boost": True},
                reason="Restore faded colors",
            ))

        elif deg_type == DegradationType.COLOR_CAST:
            recs.append(ProcessorRecommendation(
                processor="film_stock_corrector",
                priority=28,
                settings={"auto_white_balance": True},
                reason="Correct color cast",
            ))

        elif deg_type == DegradationType.VIGNETTE:
            recs.append(ProcessorRecommendation(
                processor="adaptive_enhance",
                priority=35,
                settings={"vignette_correction": True},
                reason="Correct vignetting",
            ))

        # VHS/Analog
        elif deg_type == DegradationType.VHS_ARTIFACTS:
            recs.append(ProcessorRecommendation(
                processor="vhs_restoration",
                priority=6,
                settings={"full_restoration": True},
                reason="Full VHS restoration",
            ))

        elif deg_type == DegradationType.TRACKING_ERRORS:
            recs.append(ProcessorRecommendation(
                processor="vhs_restoration",
                priority=7,
                settings={"tracking_fix": True},
                reason="Fix VHS tracking errors",
            ))

        elif deg_type == DegradationType.CHROMA_BLEEDING:
            recs.append(ProcessorRecommendation(
                processor="vhs_restoration",
                priority=12,
                settings={"chroma_fix": True},
                reason="Fix chroma bleeding",
            ))

        # Flicker
        elif deg_type == DegradationType.FLICKER:
            recs.append(ProcessorRecommendation(
                processor="temporal_denoise",
                priority=13,
                settings={"flicker_reduction": True},
                reason="Reduce brightness flicker",
            ))

        return recs

    def _generate_recommendations(
        self,
        degradations: List[DegradationInfo],
    ) -> List[str]:
        """Generate text recommendations based on degradations."""
        recommendations = []

        for deg in degradations:
            if deg.severity < 0.2:
                continue

            severity_text = deg.severity_level.name.lower()

            if deg.type == DegradationType.NOISE:
                recommendations.append(
                    f"{severity_text.title()} noise detected. Use temporal denoising for best results."
                )
            elif deg.type == DegradationType.BLUR:
                recommendations.append(
                    f"{severity_text.title()} blur detected. AI upscaling can help improve sharpness."
                )
            elif deg.type == DegradationType.BLOCKING:
                recommendations.append(
                    f"Compression blocking artifacts detected. Use QP artifact removal."
                )
            elif deg.type == DegradationType.INTERLACING:
                recommendations.append(
                    "Interlacing detected. Apply deinterlacing before other processing."
                )
            elif deg.type == DegradationType.SCRATCHES:
                recommendations.append(
                    f"Film scratches detected. Use defect repair processor."
                )
            elif deg.type == DegradationType.VHS_ARTIFACTS:
                recommendations.append(
                    "VHS artifacts detected. Use specialized VHS restoration."
                )
            elif deg.type == DegradationType.COLOR_FADE:
                recommendations.append(
                    "Faded colors detected. Consider colorization or saturation boost."
                )

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_degradations(frame: np.ndarray) -> List[DegradationInfo]:
    """Detect all degradations in a frame.

    Args:
        frame: BGR frame

    Returns:
        List of DegradationInfo
    """
    detector = DegradationDetector()
    return detector.detect(frame)


def analyze_video_degradations(
    video_path: Path,
    sample_rate: int = 10,
) -> VideoAnalysisResult:
    """Analyze video for degradations.

    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame

    Returns:
        VideoAnalysisResult
    """
    detector = DegradationDetector()
    return detector.analyze_video(video_path, sample_rate=sample_rate)


def suggest_restoration_pipeline(
    degradations: List[DegradationInfo],
) -> List[str]:
    """Suggest restoration processors for detected degradations.

    Args:
        degradations: List of detected degradations

    Returns:
        List of processor names in recommended order
    """
    detector = DegradationDetector()
    recommendations = detector.suggest_processors(degradations)
    return [r.processor for r in recommendations]
