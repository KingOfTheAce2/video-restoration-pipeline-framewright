"""Unified Video Stabilization Processor for FrameWright.

This module provides comprehensive video stabilization for restoration,
including motion analysis, trajectory smoothing, and multiple backend
support with specialized handling for different motion types.

Key Features:
- Motion analysis using optical flow and feature tracking
- Multiple stabilization backends (VidStab, OpenCV, custom)
- Rolling shutter correction
- Tripod mode for locked camera
- Handling for different motion types (handheld, panning, walking)
- Crop preview and border mode options

VRAM Requirements:
- CPU mode: No GPU required
- GPU optical flow: ~512MB VRAM

Example:
    >>> from framewright.processors.restoration import Stabilizer, StabilizationConfig
    >>>
    >>> config = StabilizationConfig(
    ...     smoothing_strength=0.8,
    ...     crop_ratio=0.1,
    ...     tripod_mode=False
    ... )
    >>> stabilizer = Stabilizer(config)
    >>>
    >>> # Stabilize frame sequence
    >>> result = stabilizer.stabilize(frames)
    >>> print(f"Stabilized {result.frames_processed} frames")
"""

import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_filter1d = None
    signal = None


# =============================================================================
# Enums and Constants
# =============================================================================

class SmoothingMode(Enum):
    """Trajectory smoothing modes."""
    GAUSSIAN = "gaussian"      # Gaussian kernel smoothing
    AVERAGE = "average"        # Moving average
    KALMAN = "kalman"          # Kalman filter
    NONE = "none"              # No smoothing


class MotionType(Enum):
    """Detected motion types."""
    STATIC = "static"          # Nearly static (tripod)
    HANDHELD = "handheld"      # Handheld shake
    PANNING = "panning"        # Deliberate pan/tilt
    TRACKING = "tracking"      # Following subject
    WALKING = "walking"        # Walking/running motion
    UNKNOWN = "unknown"


class StabilizationBackendType(Enum):
    """Available stabilization backends."""
    VIDSTAB = "vidstab"        # FFmpeg vidstab filter
    OPENCV = "opencv"          # OpenCV feature tracking
    HYBRID = "hybrid"          # Combination of methods
    AUTO = "auto"              # Auto-select best


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StabilizationConfig:
    """Configuration for video stabilization.

    Attributes:
        smoothing_strength: Smoothing strength (0-1, higher = smoother)
        crop_ratio: Ratio to crop for stabilization (0-0.5)
        rolling_shutter_correction: Enable rolling shutter correction
        tripod_mode: Lock camera position completely
        preserve_scale: Avoid zooming to fill after crop
        max_angle: Maximum rotation correction in degrees
        max_shift: Maximum translation in pixels
        smoothing_mode: Type of trajectory smoothing
        border_mode: Border handling ("replicate", "reflect", "black")
        backend: Stabilization backend to use
        motion_type_hint: Hint for expected motion type
        step_size: Analyze every N frames (1 = all)
        gpu_accelerate: Use GPU if available
    """
    smoothing_strength: float = 0.8
    crop_ratio: float = 0.1
    rolling_shutter_correction: bool = False
    tripod_mode: bool = False
    preserve_scale: bool = False
    max_angle: float = 5.0
    max_shift: float = 100.0
    smoothing_mode: SmoothingMode = SmoothingMode.GAUSSIAN
    border_mode: str = "replicate"
    backend: StabilizationBackendType = StabilizationBackendType.AUTO
    motion_type_hint: Optional[MotionType] = None
    step_size: int = 1
    gpu_accelerate: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.smoothing_strength <= 1.0:
            raise ValueError(f"smoothing_strength must be 0-1, got {self.smoothing_strength}")
        if not 0.0 <= self.crop_ratio <= 0.5:
            raise ValueError(f"crop_ratio must be 0-0.5, got {self.crop_ratio}")
        if self.max_angle < 0:
            raise ValueError(f"max_angle must be >= 0, got {self.max_angle}")
        if self.max_shift < 0:
            raise ValueError(f"max_shift must be >= 0, got {self.max_shift}")
        if self.step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {self.step_size}")

        # Convert string enums
        if isinstance(self.smoothing_mode, str):
            self.smoothing_mode = SmoothingMode(self.smoothing_mode.lower())
        if isinstance(self.backend, str):
            self.backend = StabilizationBackendType(self.backend.lower())


@dataclass
class CameraMotion:
    """Camera motion between two consecutive frames.

    Attributes:
        dx: Horizontal translation in pixels
        dy: Vertical translation in pixels
        rotation: Rotation angle in radians
        scale: Scale factor (1.0 = no change)
        timestamp: Frame timestamp
        frame_index: Frame number
        confidence: Motion estimate confidence (0-1)
    """
    dx: float
    dy: float
    rotation: float = 0.0
    scale: float = 1.0
    timestamp: float = 0.0
    frame_index: int = 0
    confidence: float = 1.0

    def magnitude(self) -> float:
        """Get motion magnitude."""
        return np.sqrt(self.dx**2 + self.dy**2)

    def to_transform_matrix(self) -> np.ndarray:
        """Convert to 2x3 affine transform matrix."""
        cos_r = np.cos(self.rotation) * self.scale
        sin_r = np.sin(self.rotation) * self.scale

        return np.array([
            [cos_r, -sin_r, self.dx],
            [sin_r, cos_r, self.dy]
        ], dtype=np.float32)

    def inverse(self) -> "CameraMotion":
        """Get inverse motion vector."""
        return CameraMotion(
            dx=-self.dx * self.scale * np.cos(self.rotation) -
               self.dy * self.scale * np.sin(self.rotation),
            dy=self.dx * self.scale * np.sin(self.rotation) -
               self.dy * self.scale * np.cos(self.rotation),
            rotation=-self.rotation,
            scale=1.0 / self.scale if self.scale != 0 else 1.0,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            confidence=self.confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dx": self.dx,
            "dy": self.dy,
            "rotation": self.rotation,
            "scale": self.scale,
            "frame_index": self.frame_index,
            "confidence": self.confidence,
        }


@dataclass
class MotionAnalysisResult:
    """Result of motion analysis.

    Attributes:
        motion_vectors: List of inter-frame motion vectors
        detected_motion_type: Detected type of motion
        shake_severity: Overall shake severity (0-1)
        problematic_segments: List of (start, end) frame indices with issues
        average_magnitude: Average motion magnitude
        max_magnitude: Maximum motion magnitude
    """
    motion_vectors: List[CameraMotion] = field(default_factory=list)
    detected_motion_type: MotionType = MotionType.UNKNOWN
    shake_severity: float = 0.0
    problematic_segments: List[Tuple[int, int]] = field(default_factory=list)
    average_magnitude: float = 0.0
    max_magnitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "motion_type": self.detected_motion_type.value,
            "shake_severity": self.shake_severity,
            "problematic_segments": self.problematic_segments,
            "average_magnitude": self.average_magnitude,
            "max_magnitude": self.max_magnitude,
            "vector_count": len(self.motion_vectors),
        }


@dataclass
class StabilizationResult:
    """Result of video stabilization.

    Attributes:
        frames_processed: Number of frames processed
        motion_vectors: Original motion vectors
        smoothed_vectors: Smoothed correction vectors
        crop_applied: Actual crop ratio used
        smoothing_applied: Actual smoothing strength used
        shake_severity: Detected shake severity
        motion_type: Detected motion type
        backend_used: Backend that was used
        success: Whether stabilization succeeded
        output_frames: Stabilized frames (if return_frames=True)
        errors: List of error messages
    """
    frames_processed: int = 0
    motion_vectors: List[CameraMotion] = field(default_factory=list)
    smoothed_vectors: List[CameraMotion] = field(default_factory=list)
    crop_applied: float = 0.0
    smoothing_applied: float = 0.0
    shake_severity: float = 0.0
    motion_type: MotionType = MotionType.UNKNOWN
    backend_used: str = ""
    success: bool = True
    output_frames: List[np.ndarray] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_processed": self.frames_processed,
            "crop_applied": self.crop_applied,
            "smoothing_applied": self.smoothing_applied,
            "shake_severity": self.shake_severity,
            "motion_type": self.motion_type.value,
            "backend_used": self.backend_used,
            "success": self.success,
            "errors": self.errors,
        }


# =============================================================================
# Motion Analyzer
# =============================================================================

class MotionAnalyzer:
    """Analyzes camera motion across video frames.

    Uses optical flow and feature tracking to detect inter-frame motion
    and classify motion types (handheld, panning, walking, etc.).

    Example:
        >>> analyzer = MotionAnalyzer()
        >>> result = analyzer.analyze(frames)
        >>> print(f"Motion type: {result.detected_motion_type}")
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 30,
        block_size: int = 3,
    ):
        """Initialize motion analyzer.

        Args:
            max_corners: Maximum corners for feature detection
            quality_level: Feature quality threshold
            min_distance: Minimum distance between features
            block_size: Block size for corner detection
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for motion analysis")

        self.feature_params = {
            "maxCorners": max_corners,
            "qualityLevel": quality_level,
            "minDistance": min_distance,
            "blockSize": block_size,
        }

        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    def analyze(
        self,
        frames: List[np.ndarray],
        step_size: int = 1,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> MotionAnalysisResult:
        """Analyze motion across frame sequence.

        Args:
            frames: List of frames to analyze
            step_size: Analyze every N frames
            progress_callback: Optional progress callback

        Returns:
            MotionAnalysisResult with analysis
        """
        result = MotionAnalysisResult()

        if len(frames) < 2:
            logger.warning("Not enough frames for motion analysis")
            return result

        prev_gray = None

        for i, frame in enumerate(frames):
            if i % step_size != 0 and i != len(frames) - 1:
                continue

            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if prev_gray is not None:
                vector = self._compute_motion(prev_gray, gray, i)
                result.motion_vectors.append(vector)

            prev_gray = gray.copy()

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        # Analyze collected vectors
        if result.motion_vectors:
            result.shake_severity = self._calculate_shake_severity(
                result.motion_vectors
            )
            result.detected_motion_type = self._classify_motion(
                result.motion_vectors
            )
            result.problematic_segments = self._find_problematic_segments(
                result.motion_vectors
            )

            magnitudes = [v.magnitude() for v in result.motion_vectors]
            result.average_magnitude = np.mean(magnitudes)
            result.max_magnitude = np.max(magnitudes)

        logger.info(
            f"Motion analysis: type={result.detected_motion_type.value}, "
            f"severity={result.shake_severity:.2f}"
        )

        return result

    def _compute_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        frame_index: int,
    ) -> CameraMotion:
        """Compute motion between two frames."""
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 10:
            return CameraMotion(dx=0, dy=0, frame_index=frame_index, confidence=0)

        # Track to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        # Filter by status
        valid = status.flatten() == 1
        if np.sum(valid) < 4:
            return CameraMotion(dx=0, dy=0, frame_index=frame_index, confidence=0)

        prev_good = prev_pts[valid]
        curr_good = curr_pts[valid]

        # Estimate transform
        try:
            transform, inliers = cv2.estimateAffinePartial2D(
                prev_good, curr_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )

            if transform is None:
                return CameraMotion(dx=0, dy=0, frame_index=frame_index, confidence=0)

            dx = transform[0, 2]
            dy = transform[1, 2]
            rotation = np.arctan2(transform[1, 0], transform[0, 0])
            scale = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)

            confidence = np.sum(inliers) / len(inliers) if inliers is not None else 0.5

            return CameraMotion(
                dx=dx,
                dy=dy,
                rotation=rotation,
                scale=scale,
                frame_index=frame_index,
                confidence=confidence,
            )

        except cv2.error as e:
            logger.debug(f"Transform estimation failed: {e}")
            return CameraMotion(dx=0, dy=0, frame_index=frame_index, confidence=0)

    def _calculate_shake_severity(self, vectors: List[CameraMotion]) -> float:
        """Calculate overall shake severity."""
        if not vectors:
            return 0.0

        dx_vals = np.array([v.dx for v in vectors])
        dy_vals = np.array([v.dy for v in vectors])
        rot_vals = np.array([v.rotation for v in vectors])

        # Standard deviations
        dx_std = np.std(dx_vals)
        dy_std = np.std(dy_vals)
        rot_std = np.std(rot_vals)

        # High-frequency component (frame-to-frame variation)
        dx_diff = np.diff(dx_vals)
        dy_diff = np.diff(dy_vals)
        hf_shake = (np.std(dx_diff) + np.std(dy_diff)) / 2

        # Combine metrics
        translation_score = min(1.0, (dx_std + dy_std) / 100)
        rotation_score = min(1.0, np.degrees(rot_std) / 5)
        hf_score = min(1.0, hf_shake / 30)

        severity = 0.4 * translation_score + 0.3 * rotation_score + 0.3 * hf_score

        return float(np.clip(severity, 0, 1))

    def _classify_motion(self, vectors: List[CameraMotion]) -> MotionType:
        """Classify the type of camera motion."""
        if not vectors:
            return MotionType.UNKNOWN

        dx_vals = np.array([v.dx for v in vectors])
        dy_vals = np.array([v.dy for v in vectors])

        avg_magnitude = np.mean([v.magnitude() for v in vectors])
        dx_mean = np.mean(dx_vals)
        dy_mean = np.mean(dy_vals)
        dx_std = np.std(dx_vals)
        dy_std = np.std(dy_vals)

        # Nearly static
        if avg_magnitude < 0.5:
            return MotionType.STATIC

        # Consistent direction = panning
        directional_consistency = np.abs(dx_mean) / (dx_std + 1) + np.abs(dy_mean) / (dy_std + 1)
        if directional_consistency > 2:
            return MotionType.PANNING

        # High-frequency low-amplitude = handheld
        dx_diff = np.diff(dx_vals)
        dy_diff = np.diff(dy_vals)
        hf_component = np.std(dx_diff) + np.std(dy_diff)

        if hf_component > 5 and avg_magnitude < 20:
            return MotionType.HANDHELD

        # Regular periodic motion = walking
        if len(vectors) > 20:
            # Check for periodicity using autocorrelation
            if HAS_SCIPY:
                autocorr = signal.correlate(dy_vals, dy_vals, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Look for peaks (periodic motion)
                peaks = signal.find_peaks(autocorr, distance=5)[0]
                if len(peaks) > 3:
                    return MotionType.WALKING

        return MotionType.HANDHELD

    def _find_problematic_segments(
        self,
        vectors: List[CameraMotion],
        threshold: float = 0.3,
    ) -> List[Tuple[int, int]]:
        """Find segments with excessive shake."""
        segments: List[Tuple[int, int]] = []
        segment_start: Optional[int] = None

        for vec in vectors:
            magnitude = vec.magnitude() + np.degrees(abs(vec.rotation)) * 5
            is_shaky = magnitude > threshold * 50

            if is_shaky and segment_start is None:
                segment_start = vec.frame_index
            elif not is_shaky and segment_start is not None:
                if vec.frame_index - segment_start >= 5:
                    segments.append((segment_start, vec.frame_index))
                segment_start = None

        # Handle segment at end
        if segment_start is not None and vectors:
            end = vectors[-1].frame_index
            if end - segment_start >= 5:
                segments.append((segment_start, end))

        return segments


# =============================================================================
# Stabilization Backends
# =============================================================================

class StabilizationBackend(ABC):
    """Abstract base class for stabilization backends."""

    @abstractmethod
    def stabilize(
        self,
        frames: List[np.ndarray],
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize a sequence of frames."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class VidStabBackend(StabilizationBackend):
    """FFmpeg vidstab-based stabilization backend.

    Uses FFmpeg's vidstab filter for two-pass stabilization.
    Requires FFmpeg with vidstab support.
    """

    def __init__(self):
        """Initialize VidStab backend."""
        self._ffmpeg_path: Optional[str] = None
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "vidstab"

    def is_available(self) -> bool:
        """Check if FFmpeg with vidstab is available."""
        if self._available is not None:
            return self._available

        try:
            # Try to find FFmpeg
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._available = "vidstab" in result.stdout
            if self._available:
                self._ffmpeg_path = "ffmpeg"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            self._available = False

        return self._available

    def stabilize(
        self,
        frames: List[np.ndarray],
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize using FFmpeg vidstab."""
        if not self.is_available():
            raise RuntimeError("FFmpeg vidstab not available")

        result = StabilizationResult()
        result.backend_used = self.name

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_video = temp_path / "input.mp4"
            output_video = temp_path / "output.mp4"
            transforms_file = temp_path / "transforms.trf"

            # Convert frames to video
            self._frames_to_video(frames, input_video)

            if progress_callback:
                progress_callback(0.1)

            # Calculate vidstab parameters
            shakiness = int(1 + config.smoothing_strength * 9)
            smoothing = int(5 + config.smoothing_strength * 25)
            zoom = 0 if config.preserve_scale else int(config.crop_ratio * 100)

            # Pass 1: Detect
            detect_cmd = [
                "ffmpeg", "-y", "-i", str(input_video),
                "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy=15:result={transforms_file}",
                "-f", "null", "-"
            ]

            try:
                subprocess.run(detect_cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                result.success = False
                result.errors.append(f"vidstabdetect failed: {e.stderr}")
                return result

            if progress_callback:
                progress_callback(0.5)

            # Pass 2: Transform
            transform_opts = [
                f"input={transforms_file}",
                f"smoothing={smoothing}",
                f"zoom={zoom}",
            ]

            if config.tripod_mode:
                transform_opts.append("tripod=1")

            transform_cmd = [
                "ffmpeg", "-y", "-i", str(input_video),
                "-vf", f"vidstabtransform={':'.join(transform_opts)}",
                str(output_video)
            ]

            try:
                subprocess.run(transform_cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                result.success = False
                result.errors.append(f"vidstabtransform failed: {e.stderr}")
                return result

            if progress_callback:
                progress_callback(0.9)

            # Extract frames
            result.output_frames = self._video_to_frames(output_video)
            result.frames_processed = len(result.output_frames)

        result.smoothing_applied = config.smoothing_strength
        result.crop_applied = config.crop_ratio if not config.preserve_scale else 0

        if progress_callback:
            progress_callback(1.0)

        return result

    def _frames_to_video(self, frames: List[np.ndarray], output_path: Path) -> None:
        """Convert frames to video."""
        if not frames:
            return

        h, w = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))

        for frame in frames:
            writer.write(frame)

        writer.release()

    def _video_to_frames(self, video_path: Path) -> List[np.ndarray]:
        """Extract frames from video."""
        frames = []

        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return frames


class OpenCVBackend(StabilizationBackend):
    """OpenCV-based stabilization using feature tracking."""

    def __init__(self):
        """Initialize OpenCV backend."""
        self._analyzer = MotionAnalyzer() if HAS_OPENCV else None

    @property
    def name(self) -> str:
        return "opencv"

    def is_available(self) -> bool:
        return HAS_OPENCV

    def stabilize(
        self,
        frames: List[np.ndarray],
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize using OpenCV."""
        if not self.is_available():
            raise RuntimeError("OpenCV not available")

        result = StabilizationResult()
        result.backend_used = self.name

        if len(frames) < 2:
            result.output_frames = frames
            result.frames_processed = len(frames)
            return result

        # Phase 1: Analyze motion (20%)
        analysis = self._analyzer.analyze(
            frames,
            step_size=config.step_size,
            progress_callback=lambda p: progress_callback(p * 0.2) if progress_callback else None,
        )

        result.motion_vectors = analysis.motion_vectors
        result.shake_severity = analysis.shake_severity
        result.motion_type = analysis.detected_motion_type

        if progress_callback:
            progress_callback(0.2)

        # Phase 2: Smooth trajectory (10%)
        smoothed = self._smooth_trajectory(
            analysis.motion_vectors, config
        )
        result.smoothed_vectors = smoothed
        result.smoothing_applied = config.smoothing_strength

        if progress_callback:
            progress_callback(0.3)

        # Phase 3: Apply stabilization (70%)
        h, w = frames[0].shape[:2]
        crop_x = int(w * config.crop_ratio)
        crop_y = int(h * config.crop_ratio)
        crop_w = w - 2 * crop_x
        crop_h = h - 2 * crop_y
        result.crop_applied = config.crop_ratio

        output_frames: List[np.ndarray] = []

        for i, frame in enumerate(frames):
            if i == 0 or i > len(smoothed):
                # First frame or no correction available
                stabilized = frame
            else:
                correction = smoothed[i - 1]
                transform = self._build_transform(correction, w, h)

                stabilized = cv2.warpAffine(
                    frame, transform, (w, h),
                    borderMode=self._get_border_mode(config.border_mode),
                )

            # Apply crop
            if config.crop_ratio > 0:
                cropped = stabilized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                if not config.preserve_scale:
                    cropped = cv2.resize(cropped, (w, h))
                output_frames.append(cropped)
            else:
                output_frames.append(stabilized)

            result.frames_processed += 1

            if progress_callback:
                progress_callback(0.3 + (i + 1) / len(frames) * 0.7)

        result.output_frames = output_frames

        return result

    def _smooth_trajectory(
        self,
        vectors: List[CameraMotion],
        config: StabilizationConfig,
    ) -> List[CameraMotion]:
        """Smooth motion trajectory."""
        if not vectors:
            return []

        dx_vals = np.array([v.dx for v in vectors])
        dy_vals = np.array([v.dy for v in vectors])
        rot_vals = np.array([v.rotation for v in vectors])
        scale_vals = np.array([v.scale for v in vectors])

        # Cumulative trajectory
        cum_dx = np.cumsum(dx_vals)
        cum_dy = np.cumsum(dy_vals)
        cum_rot = np.cumsum(rot_vals)
        cum_scale = np.cumprod(scale_vals)

        # Smooth
        if config.smoothing_mode == SmoothingMode.GAUSSIAN:
            kernel_size = int(30 * config.smoothing_strength) * 2 + 1
            if kernel_size > 1 and HAS_SCIPY:
                sigma = kernel_size / 6.0
                smooth_dx = gaussian_filter1d(cum_dx, sigma, mode="nearest")
                smooth_dy = gaussian_filter1d(cum_dy, sigma, mode="nearest")
                smooth_rot = gaussian_filter1d(cum_rot, sigma, mode="nearest")
                smooth_scale = gaussian_filter1d(cum_scale, sigma, mode="nearest")
            else:
                smooth_dx, smooth_dy = cum_dx, cum_dy
                smooth_rot, smooth_scale = cum_rot, cum_scale

        elif config.smoothing_mode == SmoothingMode.AVERAGE:
            window = int(30 * config.smoothing_strength)
            if window > 1:
                smooth_dx = self._moving_average(cum_dx, window)
                smooth_dy = self._moving_average(cum_dy, window)
                smooth_rot = self._moving_average(cum_rot, window)
                smooth_scale = self._moving_average(cum_scale, window)
            else:
                smooth_dx, smooth_dy = cum_dx, cum_dy
                smooth_rot, smooth_scale = cum_rot, cum_scale
        else:
            smooth_dx, smooth_dy = cum_dx, cum_dy
            smooth_rot, smooth_scale = cum_rot, cum_scale

        # Calculate corrections
        corr_dx = smooth_dx - cum_dx
        corr_dy = smooth_dy - cum_dy
        corr_rot = smooth_rot - cum_rot
        corr_scale = smooth_scale / cum_scale

        # Clamp
        max_shift = config.max_shift
        max_angle = np.radians(config.max_angle)

        corr_dx = np.clip(corr_dx, -max_shift, max_shift)
        corr_dy = np.clip(corr_dy, -max_shift, max_shift)
        corr_rot = np.clip(corr_rot, -max_angle, max_angle)

        # Build correction vectors
        smoothed: List[CameraMotion] = []
        for i, vec in enumerate(vectors):
            smoothed.append(CameraMotion(
                dx=corr_dx[i],
                dy=corr_dy[i],
                rotation=corr_rot[i],
                scale=corr_scale[i] if not config.preserve_scale else 1.0,
                frame_index=vec.frame_index,
                confidence=vec.confidence,
            ))

        return smoothed

    def _moving_average(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average."""
        if window < 1:
            return arr

        kernel = np.ones(window) / window
        pad_size = window // 2
        padded = np.pad(arr, pad_size, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

        return smoothed[:len(arr)]

    def _build_transform(
        self,
        correction: CameraMotion,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Build affine transform for correction."""
        cx, cy = width / 2, height / 2

        cos_r = np.cos(correction.rotation)
        sin_r = np.sin(correction.rotation)
        scale = correction.scale

        transform = np.array([
            [cos_r * scale, -sin_r * scale,
             (1 - cos_r * scale) * cx + sin_r * scale * cy + correction.dx],
            [sin_r * scale, cos_r * scale,
             -sin_r * scale * cx + (1 - cos_r * scale) * cy + correction.dy]
        ], dtype=np.float32)

        return transform

    def _get_border_mode(self, mode: str) -> int:
        """Convert border mode string to OpenCV constant."""
        modes = {
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT_101,
            "black": cv2.BORDER_CONSTANT,
            "constant": cv2.BORDER_CONSTANT,
        }
        return modes.get(mode.lower(), cv2.BORDER_REPLICATE)


# =============================================================================
# Main Stabilizer Class
# =============================================================================

class Stabilizer:
    """Unified video stabilization processor.

    Provides a single interface for video stabilization with support
    for multiple backends and automatic motion type detection.

    Example:
        >>> config = StabilizationConfig(smoothing_strength=0.8)
        >>> stabilizer = Stabilizer(config)
        >>> result = stabilizer.stabilize(frames)
    """

    def __init__(
        self,
        config: Optional[StabilizationConfig] = None,
    ):
        """Initialize stabilizer.

        Args:
            config: Stabilization configuration
        """
        self.config = config or StabilizationConfig()
        self._analyzer = MotionAnalyzer() if HAS_OPENCV else None

        # Initialize backends
        self._backends: Dict[StabilizationBackendType, StabilizationBackend] = {}

        if HAS_OPENCV:
            self._backends[StabilizationBackendType.OPENCV] = OpenCVBackend()

        vidstab = VidStabBackend()
        if vidstab.is_available():
            self._backends[StabilizationBackendType.VIDSTAB] = vidstab

    def _select_backend(self) -> StabilizationBackend:
        """Select appropriate backend."""
        if self.config.backend == StabilizationBackendType.AUTO:
            # Prefer vidstab if available
            if StabilizationBackendType.VIDSTAB in self._backends:
                return self._backends[StabilizationBackendType.VIDSTAB]
            elif StabilizationBackendType.OPENCV in self._backends:
                return self._backends[StabilizationBackendType.OPENCV]
            else:
                raise RuntimeError("No stabilization backend available")

        if self.config.backend in self._backends:
            return self._backends[self.config.backend]

        # Fallback
        if self._backends:
            return next(iter(self._backends.values()))

        raise RuntimeError("No stabilization backend available")

    def analyze(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> MotionAnalysisResult:
        """Analyze camera motion in frames.

        Args:
            frames: List of input frames
            progress_callback: Optional progress callback

        Returns:
            MotionAnalysisResult with analysis
        """
        if self._analyzer is None:
            raise RuntimeError("OpenCV required for motion analysis")

        return self._analyzer.analyze(
            frames,
            step_size=self.config.step_size,
            progress_callback=progress_callback,
        )

    def stabilize(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize a sequence of frames.

        Args:
            frames: List of input frames
            progress_callback: Optional progress callback

        Returns:
            StabilizationResult with stabilized frames
        """
        if not frames:
            return StabilizationResult()

        backend = self._select_backend()

        logger.info(
            f"Stabilizing {len(frames)} frames with {backend.name} backend"
        )

        return backend.stabilize(frames, self.config, progress_callback)

    def get_crop_preview(
        self,
        frames: List[np.ndarray],
        sample_count: int = 5,
    ) -> List[np.ndarray]:
        """Generate preview showing crop area.

        Args:
            frames: List of input frames
            sample_count: Number of preview frames

        Returns:
            List of preview frames with crop visualization
        """
        if not frames:
            return []

        h, w = frames[0].shape[:2]
        crop_x = int(w * self.config.crop_ratio)
        crop_y = int(h * self.config.crop_ratio)

        # Sample frames
        indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        previews = []

        for i in indices:
            preview = frames[i].copy()

            # Draw crop rectangle
            cv2.rectangle(
                preview,
                (crop_x, crop_y),
                (w - crop_x, h - crop_y),
                (0, 255, 0),
                2,
            )

            # Darken outside area
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[crop_y:h-crop_y, crop_x:w-crop_x] = 1

            preview_float = preview.astype(np.float32)
            preview_float[mask == 0] *= 0.5
            preview = preview_float.astype(np.uint8)

            previews.append(preview)

        return previews

    def is_available(self) -> bool:
        """Check if any stabilization backend is available."""
        return len(self._backends) > 0

    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return [b.name for b in self._backends.values()]


# =============================================================================
# Factory Functions
# =============================================================================

def create_stabilizer(
    smoothing_strength: float = 0.8,
    crop_ratio: float = 0.1,
    tripod_mode: bool = False,
    backend: str = "auto",
    **kwargs,
) -> Stabilizer:
    """Factory function to create a stabilizer.

    Args:
        smoothing_strength: Smoothing strength (0-1)
        crop_ratio: Crop ratio (0-0.5)
        tripod_mode: Enable tripod mode
        backend: Backend to use ("vidstab", "opencv", "auto")
        **kwargs: Additional config parameters

    Returns:
        Configured Stabilizer
    """
    config = StabilizationConfig(
        smoothing_strength=smoothing_strength,
        crop_ratio=crop_ratio,
        tripod_mode=tripod_mode,
        backend=StabilizationBackendType(backend.lower()),
        **kwargs,
    )
    return Stabilizer(config)


def detect_shake_severity(
    frames: List[np.ndarray],
    sample_count: int = 50,
) -> float:
    """Convenience function to detect shake severity.

    Args:
        frames: Input frames
        sample_count: Number of frames to sample

    Returns:
        Shake severity (0-1)
    """
    analyzer = MotionAnalyzer()

    # Sample frames
    if len(frames) > sample_count:
        indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        sampled = [frames[i] for i in indices]
    else:
        sampled = frames

    result = analyzer.analyze(sampled)
    return result.shake_severity


def stabilize_frames(
    frames: List[np.ndarray],
    smoothing_strength: float = 0.8,
    crop_ratio: float = 0.1,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[np.ndarray]:
    """Convenience function to stabilize frames.

    Args:
        frames: Input frames
        smoothing_strength: Smoothing strength (0-1)
        crop_ratio: Crop ratio (0-0.5)
        progress_callback: Optional progress callback

    Returns:
        Stabilized frames
    """
    stabilizer = create_stabilizer(
        smoothing_strength=smoothing_strength,
        crop_ratio=crop_ratio,
    )
    result = stabilizer.stabilize(frames, progress_callback)
    return result.output_frames
