"""Video stabilization processor using FFmpeg vidstab and OpenCV.

This module provides video stabilization capabilities to reduce camera shake
while preserving intentional motion. Supports both FFmpeg-based (vidstab filter)
and pure OpenCV-based approaches.

Features:
- Motion analysis using optical flow
- Trajectory smoothing with configurable strength
- Multiple stabilization algorithms (vidstab, opencv)
- Shake severity detection
- Crop/scale handling after stabilization
- Support for both video files and frame sequences

Example:
    >>> config = StabilizationConfig(smoothing_strength=0.8, algorithm="vidstab")
    >>> stabilizer = VideoStabilizer(config)
    >>> result = stabilizer.stabilize_video(
    ...     input_path=Path("shaky_video.mp4"),
    ...     output_path=Path("stable_video.mp4")
    ... )

    >>> # Or for frame sequences
    >>> result = stabilizer.stabilize_frames(
    ...     input_dir=Path("frames/input"),
    ...     output_dir=Path("frames/output"),
    ...     config=config
    ... )
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# Optional imports with fallback handling
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class StabilizationAlgorithm(Enum):
    """Available stabilization algorithms."""
    VIDSTAB = "vidstab"  # FFmpeg vidstab filter (2-pass)
    OPENCV = "opencv"  # OpenCV optical flow based
    AUTO = "auto"  # Automatically select best available


class SmoothingMode(Enum):
    """Trajectory smoothing modes."""
    GAUSS = "gauss"  # Gaussian smoothing
    AVERAGE = "average"  # Moving average
    NONE = "none"  # No smoothing (only transformation)


@dataclass
class MotionVector:
    """Represents motion between two consecutive frames.

    Attributes:
        dx: Horizontal translation in pixels.
        dy: Vertical translation in pixels.
        rotation: Rotation angle in radians.
        scale: Scale factor (1.0 = no scaling).
        timestamp: Frame timestamp in seconds (optional).
        frame_index: Frame number in sequence.
        confidence: Confidence score for the motion estimate (0-1).
    """
    dx: float
    dy: float
    rotation: float = 0.0
    scale: float = 1.0
    timestamp: float = 0.0
    frame_index: int = 0
    confidence: float = 1.0

    def to_transform_matrix(self) -> np.ndarray:
        """Convert motion vector to 2x3 affine transformation matrix.

        Returns:
            2x3 numpy array for use with cv2.warpAffine.
        """
        cos_r = np.cos(self.rotation) * self.scale
        sin_r = np.sin(self.rotation) * self.scale

        return np.array([
            [cos_r, -sin_r, self.dx],
            [sin_r, cos_r, self.dy]
        ], dtype=np.float32)

    def inverse(self) -> "MotionVector":
        """Get the inverse motion vector (for undoing this transform).

        Returns:
            New MotionVector that reverses this motion.
        """
        # For small angles and scales near 1, this approximation works well
        return MotionVector(
            dx=-self.dx * self.scale * np.cos(self.rotation) - self.dy * self.scale * np.sin(self.rotation),
            dy=self.dx * self.scale * np.sin(self.rotation) - self.dy * self.scale * np.cos(self.rotation),
            rotation=-self.rotation,
            scale=1.0 / self.scale if self.scale != 0 else 1.0,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            confidence=self.confidence,
        )


@dataclass
class StabilizationConfig:
    """Configuration for video stabilization.

    Attributes:
        smoothing_strength: Strength of trajectory smoothing (0.0-1.0).
            0.0 = no smoothing (raw stabilization)
            1.0 = maximum smoothing (very stable but may lose motion)
        crop_ratio: Ratio to crop after stabilization (0.0-0.5).
            Higher values allow more correction but reduce output size.
        algorithm: Stabilization algorithm to use.
        preserve_scale: If True, avoid zooming to fill after crop.
        max_angle: Maximum rotation correction in degrees.
        max_shift: Maximum translation correction in pixels.
        smoothing_mode: Type of trajectory smoothing.
        step_size: Analysis step size (analyze every N frames).
        border_mode: Border handling ('replicate', 'reflect', 'constant').
        tripod_mode: Enable tripod mode (completely static camera).
    """
    smoothing_strength: float = 0.8
    crop_ratio: float = 0.1
    algorithm: StabilizationAlgorithm = StabilizationAlgorithm.AUTO
    preserve_scale: bool = False
    max_angle: float = 3.0
    max_shift: float = 100.0
    smoothing_mode: SmoothingMode = SmoothingMode.GAUSS
    step_size: int = 1
    border_mode: str = "replicate"
    tripod_mode: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.smoothing_strength <= 1.0:
            raise ValueError("smoothing_strength must be between 0.0 and 1.0")
        if not 0.0 <= self.crop_ratio <= 0.5:
            raise ValueError("crop_ratio must be between 0.0 and 0.5")
        if self.max_angle < 0:
            raise ValueError("max_angle must be non-negative")
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        if self.step_size < 1:
            raise ValueError("step_size must be at least 1")

        # Convert string enums if needed
        if isinstance(self.algorithm, str):
            self.algorithm = StabilizationAlgorithm(self.algorithm.lower())
        if isinstance(self.smoothing_mode, str):
            self.smoothing_mode = SmoothingMode(self.smoothing_mode.lower())


@dataclass
class StabilizationResult:
    """Results from video stabilization.

    Attributes:
        motion_vectors: List of detected motion vectors.
        smoothed_vectors: List of smoothed motion vectors applied.
        crop_applied: Actual crop ratio applied.
        smoothing_applied: Actual smoothing strength applied.
        frames_processed: Number of frames processed.
        shake_severity: Detected shake severity (0-1).
        algorithm_used: Algorithm that was used.
        success: Whether stabilization completed successfully.
        errors: List of error messages encountered.
    """
    motion_vectors: List[MotionVector] = field(default_factory=list)
    smoothed_vectors: List[MotionVector] = field(default_factory=list)
    crop_applied: float = 0.0
    smoothing_applied: float = 0.0
    frames_processed: int = 0
    shake_severity: float = 0.0
    algorithm_used: str = ""
    success: bool = True
    errors: List[str] = field(default_factory=list)


class MotionAnalyzer:
    """Analyzes camera motion across video frames.

    Uses optical flow to detect global motion vectors, distinguishing
    camera shake from intentional motion.

    Attributes:
        feature_params: Parameters for feature detection.
        lk_params: Parameters for Lucas-Kanade optical flow.

    Example:
        >>> analyzer = MotionAnalyzer()
        >>> vectors = analyzer.analyze_motion(Path("frames/"))
        >>> severity = analyzer.detect_shake_severity(vectors)
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 30,
        block_size: int = 3,
    ) -> None:
        """Initialize the motion analyzer.

        Args:
            max_corners: Maximum number of corners for feature detection.
            quality_level: Quality level for corner detection.
            min_distance: Minimum distance between corners.
            block_size: Block size for corner detection.
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
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

    def analyze_motion(
        self,
        frames_dir: Path,
        step_size: int = 1,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[MotionVector]:
        """Analyze motion across a sequence of frames.

        Args:
            frames_dir: Directory containing frame images (PNG/JPG).
            step_size: Analyze every N frames (1 = all frames).
            progress_callback: Optional callback for progress updates.

        Returns:
            List of MotionVector objects describing inter-frame motion.
        """
        frames_dir = Path(frames_dir)
        frame_files = sorted(
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg"))
        )

        if len(frame_files) < 2:
            logger.warning(f"Not enough frames in {frames_dir} for motion analysis")
            return []

        motion_vectors: List[MotionVector] = []
        prev_gray = None

        for i, frame_path in enumerate(frame_files):
            if i % step_size != 0 and i != len(frame_files) - 1:
                continue

            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to load frame: {frame_path}")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute motion vector
                vector = self._compute_motion_vector(prev_gray, gray, i)
                motion_vectors.append(vector)

            prev_gray = gray.copy()

            if progress_callback:
                progress_callback(i / len(frame_files))

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Analyzed {len(motion_vectors)} motion vectors from {len(frame_files)} frames")
        return motion_vectors

    def _compute_motion_vector(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        frame_index: int,
    ) -> MotionVector:
        """Compute motion vector between two consecutive frames.

        Args:
            prev_gray: Previous frame (grayscale).
            curr_gray: Current frame (grayscale).
            frame_index: Current frame index.

        Returns:
            MotionVector describing the inter-frame motion.
        """
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 10:
            # Not enough features, return zero motion
            return MotionVector(
                dx=0.0, dy=0.0, rotation=0.0, scale=1.0,
                frame_index=frame_index, confidence=0.0
            )

        # Track features to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        # Filter by status
        valid_idx = status.flatten() == 1
        if np.sum(valid_idx) < 4:
            return MotionVector(
                dx=0.0, dy=0.0, rotation=0.0, scale=1.0,
                frame_index=frame_index, confidence=0.0
            )

        prev_good = prev_pts[valid_idx]
        curr_good = curr_pts[valid_idx]

        # Estimate affine transformation
        try:
            transform, inliers = cv2.estimateAffinePartial2D(
                prev_good, curr_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )

            if transform is None:
                return MotionVector(
                    dx=0.0, dy=0.0, rotation=0.0, scale=1.0,
                    frame_index=frame_index, confidence=0.0
                )

            # Extract motion parameters
            dx = transform[0, 2]
            dy = transform[1, 2]
            rotation = np.arctan2(transform[1, 0], transform[0, 0])
            scale = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)

            # Calculate confidence based on inlier ratio
            confidence = np.sum(inliers) / len(inliers) if inliers is not None else 0.5

            return MotionVector(
                dx=dx,
                dy=dy,
                rotation=rotation,
                scale=scale,
                frame_index=frame_index,
                confidence=confidence,
            )

        except cv2.error as e:
            logger.debug(f"Transform estimation failed: {e}")
            return MotionVector(
                dx=0.0, dy=0.0, rotation=0.0, scale=1.0,
                frame_index=frame_index, confidence=0.0
            )

    def detect_shake_severity(
        self,
        motion_vectors: List[MotionVector],
    ) -> float:
        """Calculate overall shake severity from motion vectors.

        Args:
            motion_vectors: List of motion vectors from analysis.

        Returns:
            Shake severity score (0.0-1.0).
            0.0 = very stable
            1.0 = severe shake
        """
        if not motion_vectors:
            return 0.0

        # Calculate statistics
        dx_vals = [v.dx for v in motion_vectors]
        dy_vals = [v.dy for v in motion_vectors]
        rot_vals = [v.rotation for v in motion_vectors]

        dx_std = np.std(dx_vals)
        dy_std = np.std(dy_vals)
        rot_std = np.std(rot_vals)

        # Compute high-frequency shake component
        dx_diff = np.diff(dx_vals)
        dy_diff = np.diff(dy_vals)
        rot_diff = np.diff(rot_vals)

        hf_shake = (np.std(dx_diff) + np.std(dy_diff)) / 2

        # Combine metrics into severity score
        translation_shake = (dx_std + dy_std) / 2
        rotation_shake = np.degrees(rot_std) * 10  # Weight rotation more

        # Normalize to 0-1 range
        # These thresholds are empirically determined
        translation_score = min(1.0, translation_shake / 50.0)
        rotation_score = min(1.0, rotation_shake / 10.0)
        hf_score = min(1.0, hf_shake / 20.0)

        # Weighted combination
        severity = 0.4 * translation_score + 0.3 * rotation_score + 0.3 * hf_score

        return float(np.clip(severity, 0.0, 1.0))

    def detect_shake_severity_from_video(
        self,
        video_path: Path,
        sample_frames: int = 100,
    ) -> float:
        """Detect shake severity directly from a video file.

        Args:
            video_path: Path to video file.
            sample_frames: Number of frames to sample for analysis.

        Returns:
            Shake severity score (0.0-1.0).
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for video analysis")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // sample_frames)

        motion_vectors: List[MotionVector] = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    vector = self._compute_motion_vector(prev_gray, gray, frame_idx)
                    motion_vectors.append(vector)

                prev_gray = gray.copy()

            frame_idx += 1

        cap.release()

        return self.detect_shake_severity(motion_vectors)

    def identify_problematic_segments(
        self,
        motion_vectors: List[MotionVector],
        threshold: float = 0.3,
        min_segment_length: int = 5,
    ) -> List[Tuple[int, int]]:
        """Identify segments with excessive camera shake.

        Args:
            motion_vectors: List of motion vectors from analysis.
            threshold: Threshold for identifying problematic motion.
            min_segment_length: Minimum frames for a segment.

        Returns:
            List of (start_frame, end_frame) tuples for problematic segments.
        """
        if not motion_vectors:
            return []

        problematic_segments: List[Tuple[int, int]] = []
        segment_start: Optional[int] = None

        for i, vec in enumerate(motion_vectors):
            # Calculate instantaneous shake magnitude
            magnitude = np.sqrt(vec.dx**2 + vec.dy**2) + np.degrees(abs(vec.rotation)) * 5

            # Threshold based on normalized magnitude
            is_shaky = magnitude > threshold * 50  # Scale threshold

            if is_shaky and segment_start is None:
                segment_start = vec.frame_index
            elif not is_shaky and segment_start is not None:
                segment_end = vec.frame_index
                if segment_end - segment_start >= min_segment_length:
                    problematic_segments.append((segment_start, segment_end))
                segment_start = None

        # Handle segment extending to end
        if segment_start is not None:
            segment_end = motion_vectors[-1].frame_index
            if segment_end - segment_start >= min_segment_length:
                problematic_segments.append((segment_start, segment_end))

        return problematic_segments


class VideoStabilizer:
    """Video stabilization processor.

    Provides video stabilization using either FFmpeg's vidstab filter
    or OpenCV-based motion compensation. Supports both video files
    and frame sequences.

    Attributes:
        config: StabilizationConfig with stabilization settings.
        analyzer: MotionAnalyzer instance for motion detection.

    Example:
        >>> config = StabilizationConfig(smoothing_strength=0.8)
        >>> stabilizer = VideoStabilizer(config)
        >>> result = stabilizer.stabilize_video(
        ...     input_path=Path("input.mp4"),
        ...     output_path=Path("output.mp4")
        ... )
    """

    def __init__(
        self,
        config: Optional[StabilizationConfig] = None,
    ) -> None:
        """Initialize the video stabilizer.

        Args:
            config: StabilizationConfig with stabilization settings.
        """
        self.config = config or StabilizationConfig()
        self._analyzer: Optional[MotionAnalyzer] = None
        self._ffmpeg_available: Optional[bool] = None
        self._vidstab_available: Optional[bool] = None

    @property
    def analyzer(self) -> MotionAnalyzer:
        """Get or create the motion analyzer instance."""
        if self._analyzer is None:
            self._analyzer = MotionAnalyzer()
        return self._analyzer

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        if self._ffmpeg_available is None:
            self._ffmpeg_available = shutil.which("ffmpeg") is not None
        return self._ffmpeg_available

    def _check_vidstab(self) -> bool:
        """Check if FFmpeg vidstab filter is available."""
        if self._vidstab_available is None:
            if not self._check_ffmpeg():
                self._vidstab_available = False
            else:
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-filters"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self._vidstab_available = "vidstab" in result.stdout
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    self._vidstab_available = False

        return self._vidstab_available

    def _get_algorithm(self) -> StabilizationAlgorithm:
        """Determine which algorithm to use."""
        if self.config.algorithm == StabilizationAlgorithm.AUTO:
            if self._check_vidstab():
                return StabilizationAlgorithm.VIDSTAB
            elif HAS_OPENCV:
                return StabilizationAlgorithm.OPENCV
            else:
                raise RuntimeError(
                    "No stabilization backend available. "
                    "Install FFmpeg with vidstab or OpenCV."
                )
        return self.config.algorithm

    def analyze_motion(
        self,
        frames_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[MotionVector]:
        """Analyze camera motion across frames.

        Args:
            frames_dir: Directory containing frame images.
            progress_callback: Optional progress callback.

        Returns:
            List of MotionVector objects.
        """
        return self.analyzer.analyze_motion(
            frames_dir,
            step_size=self.config.step_size,
            progress_callback=progress_callback,
        )

    def smooth_trajectory(
        self,
        vectors: List[MotionVector],
        strength: Optional[float] = None,
    ) -> List[MotionVector]:
        """Smooth the motion trajectory.

        Args:
            vectors: List of motion vectors from analysis.
            strength: Smoothing strength (0.0-1.0). Uses config if None.

        Returns:
            List of smoothed motion vectors.
        """
        if not vectors:
            return []

        strength = strength if strength is not None else self.config.smoothing_strength

        # Convert to arrays
        dx_vals = np.array([v.dx for v in vectors])
        dy_vals = np.array([v.dy for v in vectors])
        rot_vals = np.array([v.rotation for v in vectors])
        scale_vals = np.array([v.scale for v in vectors])

        # Compute cumulative trajectory
        cum_dx = np.cumsum(dx_vals)
        cum_dy = np.cumsum(dy_vals)
        cum_rot = np.cumsum(rot_vals)
        cum_scale = np.cumprod(scale_vals)

        # Smooth trajectory
        if self.config.smoothing_mode == SmoothingMode.GAUSS:
            # Gaussian smoothing with kernel size based on strength
            kernel_size = int(30 * strength) * 2 + 1  # Odd number
            if kernel_size > 1:
                from scipy.ndimage import gaussian_filter1d
                sigma = kernel_size / 6.0

                smooth_dx = gaussian_filter1d(cum_dx, sigma, mode="nearest")
                smooth_dy = gaussian_filter1d(cum_dy, sigma, mode="nearest")
                smooth_rot = gaussian_filter1d(cum_rot, sigma, mode="nearest")
                smooth_scale = gaussian_filter1d(cum_scale, sigma, mode="nearest")
            else:
                smooth_dx, smooth_dy = cum_dx, cum_dy
                smooth_rot, smooth_scale = cum_rot, cum_scale

        elif self.config.smoothing_mode == SmoothingMode.AVERAGE:
            # Moving average smoothing
            window = int(30 * strength)
            if window > 1:
                smooth_dx = self._moving_average(cum_dx, window)
                smooth_dy = self._moving_average(cum_dy, window)
                smooth_rot = self._moving_average(cum_rot, window)
                smooth_scale = self._moving_average(cum_scale, window)
            else:
                smooth_dx, smooth_dy = cum_dx, cum_dy
                smooth_rot, smooth_scale = cum_rot, cum_scale
        else:
            # No smoothing
            smooth_dx, smooth_dy = cum_dx, cum_dy
            smooth_rot, smooth_scale = cum_rot, cum_scale

        # Calculate correction (difference between smooth and original)
        corr_dx = smooth_dx - cum_dx
        corr_dy = smooth_dy - cum_dy
        corr_rot = smooth_rot - cum_rot
        corr_scale = smooth_scale / cum_scale

        # Clamp corrections to max values
        max_shift = self.config.max_shift
        max_angle = np.radians(self.config.max_angle)

        corr_dx = np.clip(corr_dx, -max_shift, max_shift)
        corr_dy = np.clip(corr_dy, -max_shift, max_shift)
        corr_rot = np.clip(corr_rot, -max_angle, max_angle)

        # Create smoothed motion vectors (corrections to apply)
        smoothed_vectors: List[MotionVector] = []
        for i, vec in enumerate(vectors):
            smoothed_vectors.append(MotionVector(
                dx=corr_dx[i],
                dy=corr_dy[i],
                rotation=corr_rot[i],
                scale=corr_scale[i] if not self.config.preserve_scale else 1.0,
                timestamp=vec.timestamp,
                frame_index=vec.frame_index,
                confidence=vec.confidence,
            ))

        return smoothed_vectors

    def _moving_average(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing.

        Args:
            arr: Input array.
            window: Window size.

        Returns:
            Smoothed array.
        """
        if window < 1:
            return arr

        kernel = np.ones(window) / window
        # Pad to handle edges
        pad_size = window // 2
        padded = np.pad(arr, pad_size, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

        return smoothed[:len(arr)]

    def stabilize_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[StabilizationConfig] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize a sequence of frames.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for stabilized output frames.
            config: Optional config override.
            progress_callback: Optional progress callback.

        Returns:
            StabilizationResult with processing details.
        """
        cfg = config or self.config
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        result = StabilizationResult()

        algorithm = self._get_algorithm()
        result.algorithm_used = algorithm.value

        try:
            if algorithm == StabilizationAlgorithm.OPENCV:
                return self._stabilize_frames_opencv(
                    input_dir, output_dir, cfg, progress_callback
                )
            else:
                # For vidstab, we need to work with video files
                # Convert frames to video, stabilize, then extract frames
                return self._stabilize_frames_via_video(
                    input_dir, output_dir, cfg, progress_callback
                )

        except Exception as e:
            logger.error(f"Stabilization failed: {e}")
            result.success = False
            result.errors.append(str(e))
            return result

    def _stabilize_frames_opencv(
        self,
        input_dir: Path,
        output_dir: Path,
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize frames using OpenCV.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            config: Stabilization configuration.
            progress_callback: Optional progress callback.

        Returns:
            StabilizationResult with processing details.
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for frame stabilization")

        result = StabilizationResult()
        result.algorithm_used = "opencv"

        # Get frame files
        frame_files = sorted(
            list(input_dir.glob("*.png")) +
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.jpeg"))
        )

        if len(frame_files) < 2:
            result.success = False
            result.errors.append("Not enough frames for stabilization")
            return result

        # Phase 1: Analyze motion (20% of progress)
        logger.info("Phase 1: Analyzing motion...")
        motion_vectors = self.analyze_motion(
            input_dir,
            progress_callback=lambda p: progress_callback(p * 0.2) if progress_callback else None,
        )

        result.motion_vectors = motion_vectors
        result.shake_severity = self.analyzer.detect_shake_severity(motion_vectors)

        if progress_callback:
            progress_callback(0.2)

        # Phase 2: Smooth trajectory (5% of progress)
        logger.info("Phase 2: Smoothing trajectory...")
        smoothed_vectors = self.smooth_trajectory(motion_vectors, config.smoothing_strength)
        result.smoothed_vectors = smoothed_vectors
        result.smoothing_applied = config.smoothing_strength

        if progress_callback:
            progress_callback(0.25)

        # Phase 3: Apply stabilization (75% of progress)
        logger.info("Phase 3: Applying stabilization...")

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]

        # Calculate crop dimensions
        crop_x = int(width * config.crop_ratio)
        crop_y = int(height * config.crop_ratio)
        crop_w = width - 2 * crop_x
        crop_h = height - 2 * crop_y
        result.crop_applied = config.crop_ratio

        # Process frames
        # First frame is reference (no transformation)
        output_path = output_dir / frame_files[0].name
        if config.crop_ratio > 0:
            cropped = first_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            if not config.preserve_scale:
                cropped = cv2.resize(cropped, (width, height))
            cv2.imwrite(str(output_path), cropped)
        else:
            cv2.imwrite(str(output_path), first_frame)

        result.frames_processed = 1

        # Process remaining frames with transformations
        for i, frame_path in enumerate(frame_files[1:], start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to load frame: {frame_path}")
                continue

            # Apply cumulative correction up to this frame
            if i - 1 < len(smoothed_vectors):
                correction = smoothed_vectors[i - 1]

                # Build transformation matrix
                transform = self._build_correction_transform(
                    correction, width, height
                )

                # Apply transformation
                stabilized = cv2.warpAffine(
                    frame,
                    transform,
                    (width, height),
                    borderMode=self._get_border_mode(config.border_mode),
                )
            else:
                stabilized = frame

            # Apply crop
            if config.crop_ratio > 0:
                cropped = stabilized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                if not config.preserve_scale:
                    cropped = cv2.resize(cropped, (width, height))
                output_frame = cropped
            else:
                output_frame = stabilized

            # Save frame
            output_path = output_dir / frame_path.name
            cv2.imwrite(str(output_path), output_frame)
            result.frames_processed += 1

            if progress_callback:
                progress = 0.25 + (i / len(frame_files)) * 0.75
                progress_callback(progress)

        if progress_callback:
            progress_callback(1.0)

        logger.info(
            f"Stabilization complete: {result.frames_processed} frames, "
            f"shake severity: {result.shake_severity:.2f}"
        )

        return result

    def _build_correction_transform(
        self,
        correction: MotionVector,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Build affine transformation matrix for correction.

        Args:
            correction: Motion correction to apply.
            width: Frame width.
            height: Frame height.

        Returns:
            2x3 affine transformation matrix.
        """
        cx, cy = width / 2, height / 2

        cos_r = np.cos(correction.rotation)
        sin_r = np.sin(correction.rotation)
        scale = correction.scale

        # Rotation around center + translation
        transform = np.array([
            [cos_r * scale, -sin_r * scale, (1 - cos_r * scale) * cx + sin_r * scale * cy + correction.dx],
            [sin_r * scale, cos_r * scale, -sin_r * scale * cx + (1 - cos_r * scale) * cy + correction.dy]
        ], dtype=np.float32)

        return transform

    def _get_border_mode(self, mode: str) -> int:
        """Convert border mode string to OpenCV constant.

        Args:
            mode: Border mode string.

        Returns:
            OpenCV border mode constant.
        """
        modes = {
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT_101,
            "constant": cv2.BORDER_CONSTANT,
            "wrap": cv2.BORDER_WRAP,
        }
        return modes.get(mode.lower(), cv2.BORDER_REPLICATE)

    def _stabilize_frames_via_video(
        self,
        input_dir: Path,
        output_dir: Path,
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize frames by converting to video, using vidstab, then extracting.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            config: Stabilization configuration.
            progress_callback: Optional progress callback.

        Returns:
            StabilizationResult with processing details.
        """
        result = StabilizationResult()
        result.algorithm_used = "vidstab"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_video = temp_path / "input.mp4"
            stable_video = temp_path / "stable.mp4"

            # Get frame files
            frame_files = sorted(
                list(input_dir.glob("*.png")) +
                list(input_dir.glob("*.jpg")) +
                list(input_dir.glob("*.jpeg"))
            )

            if len(frame_files) < 2:
                result.success = False
                result.errors.append("Not enough frames for stabilization")
                return result

            # Phase 1: Create temp video (10% of progress)
            logger.info("Phase 1: Creating temporary video...")
            self._frames_to_video(input_dir, temp_video, fps=30)

            if progress_callback:
                progress_callback(0.1)

            # Phase 2: Stabilize with vidstab (80% of progress)
            logger.info("Phase 2: Running vidstab stabilization...")
            result = self._stabilize_video_vidstab(
                temp_video, stable_video, config,
                progress_callback=lambda p: progress_callback(0.1 + p * 0.8) if progress_callback else None,
            )

            if not result.success:
                return result

            # Phase 3: Extract frames (10% of progress)
            logger.info("Phase 3: Extracting stabilized frames...")
            self._video_to_frames(stable_video, output_dir)

            if progress_callback:
                progress_callback(1.0)

        return result

    def stabilize_video(
        self,
        input_path: Path,
        output_path: Path,
        config: Optional[StabilizationConfig] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize a video file.

        Args:
            input_path: Path to input video.
            output_path: Path for stabilized output video.
            config: Optional config override.
            progress_callback: Optional progress callback.

        Returns:
            StabilizationResult with processing details.
        """
        cfg = config or self.config
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        algorithm = self._get_algorithm()

        if algorithm == StabilizationAlgorithm.VIDSTAB:
            return self._stabilize_video_vidstab(
                input_path, output_path, cfg, progress_callback
            )
        else:
            # OpenCV: extract frames, stabilize, reassemble
            return self._stabilize_video_opencv(
                input_path, output_path, cfg, progress_callback
            )

    def _stabilize_video_vidstab(
        self,
        input_path: Path,
        output_path: Path,
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize video using FFmpeg vidstab filter.

        Args:
            input_path: Input video path.
            output_path: Output video path.
            config: Stabilization configuration.
            progress_callback: Optional progress callback.

        Returns:
            StabilizationResult with processing details.
        """
        if not self._check_vidstab():
            raise RuntimeError("FFmpeg vidstab filter not available")

        result = StabilizationResult()
        result.algorithm_used = "vidstab"

        with tempfile.TemporaryDirectory() as temp_dir:
            transforms_file = Path(temp_dir) / "transforms.trf"

            # Calculate vidstab parameters from config
            # shakiness: 1-10, accuracy: 1-15
            shakiness = int(1 + config.smoothing_strength * 9)
            accuracy = 15  # Maximum accuracy

            # smoothing: number of frames for smoothing
            # Higher smoothing_strength = more frames = smoother
            smoothing = int(5 + config.smoothing_strength * 25)

            # zoom: negative = zoom out to show black borders, positive = zoom in
            zoom = 0 if config.preserve_scale else int(config.crop_ratio * 100)

            # Phase 1: Detect motion (vidstabdetect)
            logger.info("Phase 1: Running vidstabdetect...")
            detect_cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={transforms_file}",
                "-f", "null", "-"
            ]

            try:
                subprocess.run(
                    detect_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                result.success = False
                result.errors.append(f"vidstabdetect failed: {e.stderr}")
                return result

            if progress_callback:
                progress_callback(0.5)

            # Count frames for result
            if transforms_file.exists():
                with open(transforms_file) as f:
                    result.frames_processed = sum(1 for _ in f)

            # Phase 2: Apply stabilization (vidstabtransform)
            logger.info("Phase 2: Running vidstabtransform...")

            # Build transform filter
            transform_opts = [
                f"input={transforms_file}",
                f"smoothing={smoothing}",
                f"zoom={zoom}",
            ]

            if config.tripod_mode:
                transform_opts.append("tripod=1")

            # Add max angle limit
            if config.max_angle < 180:
                transform_opts.append(f"maxangle={np.radians(config.max_angle)}")

            # Add max shift limit
            if config.max_shift < 1000:
                transform_opts.append(f"maxshift={int(config.max_shift)}")

            transform_filter = f"vidstabtransform={':'.join(transform_opts)}"

            transform_cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vf", transform_filter,
                "-c:a", "copy",  # Copy audio
                str(output_path)
            ]

            try:
                subprocess.run(
                    transform_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                result.success = False
                result.errors.append(f"vidstabtransform failed: {e.stderr}")
                return result

            if progress_callback:
                progress_callback(1.0)

        result.smoothing_applied = config.smoothing_strength
        result.crop_applied = config.crop_ratio if not config.preserve_scale else 0

        logger.info(
            f"Vidstab stabilization complete: {result.frames_processed} frames"
        )

        return result

    def _stabilize_video_opencv(
        self,
        input_path: Path,
        output_path: Path,
        config: StabilizationConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> StabilizationResult:
        """Stabilize video using OpenCV (extract-stabilize-reassemble).

        Args:
            input_path: Input video path.
            output_path: Output video path.
            config: Stabilization configuration.
            progress_callback: Optional progress callback.

        Returns:
            StabilizationResult with processing details.
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for video stabilization")

        result = StabilizationResult()
        result.algorithm_used = "opencv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / "frames"
            stable_dir = temp_path / "stable"
            frames_dir.mkdir()
            stable_dir.mkdir()

            # Phase 1: Extract frames (20% of progress)
            logger.info("Phase 1: Extracting frames...")
            self._video_to_frames(input_path, frames_dir)

            if progress_callback:
                progress_callback(0.2)

            # Phase 2: Stabilize frames (60% of progress)
            logger.info("Phase 2: Stabilizing frames...")
            result = self._stabilize_frames_opencv(
                frames_dir, stable_dir, config,
                progress_callback=lambda p: progress_callback(0.2 + p * 0.6) if progress_callback else None,
            )

            if not result.success:
                return result

            # Phase 3: Reassemble video (20% of progress)
            logger.info("Phase 3: Reassembling video...")

            # Get FPS from original video
            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()

            self._frames_to_video(stable_dir, output_path, fps=fps)

            # Copy audio from original
            self._copy_audio(input_path, output_path)

            if progress_callback:
                progress_callback(1.0)

        return result

    def _frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float = 30,
    ) -> None:
        """Convert frame sequence to video using FFmpeg.

        Args:
            frames_dir: Directory containing frames.
            output_path: Output video path.
            fps: Frame rate for output video.
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is required for video creation")

        # Detect frame pattern
        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            raise ValueError(f"No frames found in {frames_dir}")

        # Determine pattern
        frame_name = frames[0].name
        if "_" in frame_name:
            # Format: frame_00000001.png
            pattern = frames_dir / "frame_%08d.png"
        else:
            pattern = frames_dir / "%08d.png"

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(pattern),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg frame assembly failed: {e.stderr}")

    def _video_to_frames(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> None:
        """Extract frames from video using FFmpeg.

        Args:
            video_path: Input video path.
            output_dir: Directory for output frames.
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is required for frame extraction")

        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = output_dir / "frame_%08d.png"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-qscale:v", "2",
            str(pattern)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg frame extraction failed: {e.stderr}")

    def _copy_audio(
        self,
        source_video: Path,
        target_video: Path,
    ) -> None:
        """Copy audio from source video to target video.

        Args:
            source_video: Video to copy audio from.
            target_video: Video to add audio to.
        """
        if not self._check_ffmpeg():
            return

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_output = Path(tmp.name)

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(target_video),
                "-i", str(source_video),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(temp_output)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                shutil.move(str(temp_output), str(target_video))
            else:
                # Audio copy failed (maybe no audio), keep video as-is
                temp_output.unlink(missing_ok=True)

        except Exception as e:
            logger.debug(f"Audio copy failed: {e}")
            temp_output.unlink(missing_ok=True)

    def detect_shake_severity(
        self,
        video_path: Path,
    ) -> float:
        """Detect shake severity in a video file.

        Args:
            video_path: Path to video file.

        Returns:
            Shake severity score (0.0-1.0).
        """
        return self.analyzer.detect_shake_severity_from_video(video_path)

    def is_available(self) -> bool:
        """Check if any stabilization backend is available.

        Returns:
            True if vidstab or OpenCV is available.
        """
        return self._check_vidstab() or HAS_OPENCV


# Convenience functions

def detect_shake_severity(video_path: Path) -> float:
    """Detect shake severity in a video file.

    Convenience function for quick shake analysis.

    Args:
        video_path: Path to video file.

    Returns:
        Shake severity score (0.0-1.0).
        0.0 = very stable
        1.0 = severe shake
    """
    analyzer = MotionAnalyzer()
    return analyzer.detect_shake_severity_from_video(video_path)


def stabilize_video(
    input_path: Path,
    output_path: Path,
    smoothing_strength: float = 0.8,
    crop_ratio: float = 0.1,
    algorithm: str = "auto",
) -> StabilizationResult:
    """Stabilize a video file with default settings.

    Convenience function for quick video stabilization.

    Args:
        input_path: Path to input video.
        output_path: Path for output video.
        smoothing_strength: Smoothing strength (0.0-1.0).
        crop_ratio: Crop ratio (0.0-0.5).
        algorithm: Algorithm to use ('vidstab', 'opencv', 'auto').

    Returns:
        StabilizationResult with processing details.
    """
    config = StabilizationConfig(
        smoothing_strength=smoothing_strength,
        crop_ratio=crop_ratio,
        algorithm=StabilizationAlgorithm(algorithm.lower()),
    )
    stabilizer = VideoStabilizer(config)
    return stabilizer.stabilize_video(input_path, output_path)


def stabilize_frames(
    input_dir: Path,
    output_dir: Path,
    smoothing_strength: float = 0.8,
    crop_ratio: float = 0.1,
) -> StabilizationResult:
    """Stabilize a frame sequence with default settings.

    Convenience function for quick frame sequence stabilization.

    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        smoothing_strength: Smoothing strength (0.0-1.0).
        crop_ratio: Crop ratio (0.0-0.5).

    Returns:
        StabilizationResult with processing details.
    """
    config = StabilizationConfig(
        smoothing_strength=smoothing_strength,
        crop_ratio=crop_ratio,
    )
    stabilizer = VideoStabilizer(config)
    return stabilizer.stabilize_frames(input_dir, output_dir)


def create_stabilizer(
    smoothing_strength: float = 0.8,
    crop_ratio: float = 0.1,
    algorithm: str = "auto",
    preserve_scale: bool = False,
    tripod_mode: bool = False,
) -> VideoStabilizer:
    """Create a VideoStabilizer with common settings.

    Args:
        smoothing_strength: Smoothing strength (0.0-1.0).
        crop_ratio: Crop ratio (0.0-0.5).
        algorithm: Algorithm to use ('vidstab', 'opencv', 'auto').
        preserve_scale: Avoid zooming to fill after crop.
        tripod_mode: Enable tripod mode (completely static camera).

    Returns:
        Configured VideoStabilizer instance.
    """
    config = StabilizationConfig(
        smoothing_strength=smoothing_strength,
        crop_ratio=crop_ratio,
        algorithm=StabilizationAlgorithm(algorithm.lower()),
        preserve_scale=preserve_scale,
        tripod_mode=tripod_mode,
    )
    return VideoStabilizer(config)
