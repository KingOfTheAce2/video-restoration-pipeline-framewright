"""Advanced Temporal Denoising for video restoration.

This module implements temporal-aware denoising to address frame-by-frame processing
inconsistencies. Key features:

- Multi-frame context for smoother, more consistent results
- Optical flow-guided temporal consistency for motion-aware denoising
- Flickering reduction algorithm for temporal artifact elimination

The temporal denoising pipeline leverages information from neighboring frames
to produce cleaner, more stable output while preserving motion and fine details.

Example:
    >>> config = TemporalDenoiseConfig(
    ...     temporal_radius=3,
    ...     noise_strength=0.5,
    ...     enable_optical_flow=True,
    ...     enable_flicker_reduction=True
    ... )
    >>> denoiser = TemporalDenoiser(config)
    >>> result = denoiser.denoise_frames(input_dir, output_dir)
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available - optical flow features limited")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.debug("PyTorch not available - neural denoising disabled")


class DenoiseMethod(Enum):
    """Available temporal denoising methods."""

    MULTI_FRAME_AVERAGE = auto()
    """Simple weighted average of neighboring frames."""

    OPTICAL_FLOW_WARP = auto()
    """Motion-compensated temporal filtering using optical flow."""

    NON_LOCAL_MEANS_TEMPORAL = auto()
    """Temporal extension of non-local means denoising."""

    BILATERAL_TEMPORAL = auto()
    """Temporal bilateral filtering for edge-preserving denoising."""

    VBM4D = auto()
    """Video Block-Matching and 4D filtering (highest quality)."""


class FlickerMode(Enum):
    """Flickering reduction modes."""

    LIGHT = "light"
    """Subtle smoothing for minor flickering."""

    MEDIUM = "medium"
    """Balanced smoothing for typical film flicker."""

    AGGRESSIVE = "aggressive"
    """Strong smoothing for severe flickering issues."""

    ADAPTIVE = "adaptive"
    """Automatically adjust strength based on detected flicker."""


class OpticalFlowMethod(Enum):
    """Optical flow estimation methods."""

    FARNEBACK = "farneback"
    """Gunnar Farneback's dense optical flow (fast, CPU)."""

    LUCAS_KANADE = "lucas_kanade"
    """Lucas-Kanade sparse optical flow (fast, sparse)."""

    DIS = "dis"
    """Dense Inverse Search optical flow (fast, GPU-accelerated)."""

    RAFT = "raft"
    """Recurrent All-Pairs Field Transforms (highest quality, needs PyTorch)."""

    RIFE = "rife"
    """Use RIFE's internal flow estimation (requires rife-ncnn-vulkan)."""


@dataclass
class TemporalDenoiseConfig:
    """Configuration for temporal denoising.

    Attributes:
        temporal_radius: Number of frames to consider on each side (total window = 2*radius + 1).
            Larger values provide better denoising but may cause motion blur.
            Recommended: 2-4 for standard video, 1-2 for fast motion.
        noise_strength: Denoising strength from 0 (none) to 1 (maximum).
            0.3-0.5 for light noise, 0.6-0.8 for moderate, 0.8-1.0 for heavy noise.
        method: Denoising algorithm to use.
        enable_optical_flow: Use motion compensation for better temporal alignment.
            Significantly improves quality but increases processing time.
        optical_flow_method: Algorithm for optical flow estimation.
        enable_flicker_reduction: Enable temporal flicker reduction.
        flicker_mode: Flicker reduction aggressiveness.
        preserve_edges: Apply edge-preserving filtering to maintain sharpness.
        edge_threshold: Threshold for edge detection (0-255).
        temporal_weight_decay: How quickly temporal weights decay with distance.
            Lower values = more weight to distant frames.
        scene_change_threshold: SSIM threshold for scene change detection.
            Denoising restarts at scene boundaries to avoid ghosting.
        gpu_id: GPU device ID for accelerated processing.
        chunk_size: Number of frames to process at once (memory management).
    """
    temporal_radius: int = 3
    noise_strength: float = 0.5
    method: DenoiseMethod = DenoiseMethod.OPTICAL_FLOW_WARP
    enable_optical_flow: bool = True
    optical_flow_method: OpticalFlowMethod = OpticalFlowMethod.FARNEBACK
    enable_flicker_reduction: bool = True
    flicker_mode: FlickerMode = FlickerMode.ADAPTIVE
    preserve_edges: bool = True
    edge_threshold: int = 30
    temporal_weight_decay: float = 0.5
    scene_change_threshold: float = 0.7
    gpu_id: int = 0
    chunk_size: int = 50

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.temporal_radius < 1:
            raise ValueError(f"temporal_radius must be >= 1, got {self.temporal_radius}")
        if not 0.0 <= self.noise_strength <= 1.0:
            raise ValueError(f"noise_strength must be 0-1, got {self.noise_strength}")
        if not 0.0 <= self.temporal_weight_decay <= 1.0:
            raise ValueError(f"temporal_weight_decay must be 0-1, got {self.temporal_weight_decay}")
        if not 0.0 <= self.scene_change_threshold <= 1.0:
            raise ValueError(f"scene_change_threshold must be 0-1, got {self.scene_change_threshold}")
        if self.chunk_size < 10:
            raise ValueError(f"chunk_size must be >= 10, got {self.chunk_size}")


@dataclass
class TemporalDenoiseResult:
    """Result of temporal denoising processing.

    Attributes:
        frames_processed: Number of frames successfully processed.
        frames_failed: Number of frames that failed processing.
        output_dir: Path to directory containing denoised frames.
        scene_changes_detected: List of frame indices where scenes changed.
        avg_noise_reduction: Average noise reduction achieved (0-1).
        flicker_reduction_applied: Whether flicker reduction was used.
        processing_time_seconds: Total processing time.
        peak_memory_mb: Peak memory usage during processing.
    """
    frames_processed: int = 0
    frames_failed: int = 0
    output_dir: Optional[Path] = None
    scene_changes_detected: List[int] = field(default_factory=list)
    avg_noise_reduction: float = 0.0
    flicker_reduction_applied: bool = False
    processing_time_seconds: float = 0.0
    peak_memory_mb: int = 0


@dataclass
class FlowField:
    """Optical flow field between two frames.

    Attributes:
        flow_x: Horizontal flow component (pixels).
        flow_y: Vertical flow component (pixels).
        magnitude: Flow magnitude at each pixel.
        confidence: Confidence/quality of flow estimation.
        frame_idx_from: Source frame index.
        frame_idx_to: Target frame index.
    """
    flow_x: np.ndarray
    flow_y: np.ndarray
    magnitude: np.ndarray
    confidence: np.ndarray
    frame_idx_from: int
    frame_idx_to: int


class OpticalFlowEstimator:
    """Estimate optical flow between frames for motion compensation.

    Optical flow provides motion vectors between consecutive frames,
    enabling motion-aware temporal filtering that preserves moving objects
    while reducing noise in static regions.

    Example:
        >>> estimator = OpticalFlowEstimator(method=OpticalFlowMethod.FARNEBACK)
        >>> flow = estimator.estimate(frame1, frame2)
        >>> warped = estimator.warp_frame(frame1, flow)
    """

    def __init__(
        self,
        method: OpticalFlowMethod = OpticalFlowMethod.FARNEBACK,
        gpu_id: int = 0,
    ):
        """Initialize optical flow estimator.

        Args:
            method: Flow estimation algorithm to use.
            gpu_id: GPU device for accelerated methods.
        """
        self.method = method
        self.gpu_id = gpu_id
        self._flow_calculator = None
        self._init_flow_calculator()

    def _init_flow_calculator(self) -> None:
        """Initialize the appropriate flow calculator."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available, optical flow disabled")
            return

        if self.method == OpticalFlowMethod.DIS:
            # Dense Inverse Search - fast and accurate
            self._flow_calculator = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            )
        elif self.method == OpticalFlowMethod.FARNEBACK:
            # Farneback - classic dense optical flow
            self._flow_calculator = None  # Use cv2.calcOpticalFlowFarneback directly
        elif self.method == OpticalFlowMethod.LUCAS_KANADE:
            # Lucas-Kanade - sparse feature tracking
            self._flow_calculator = None
        elif self.method == OpticalFlowMethod.RAFT and HAS_TORCH:
            # RAFT - neural network based (placeholder)
            logger.info("RAFT flow estimation requires model download")
            self._flow_calculator = None

    def estimate(
        self,
        frame1: Union[np.ndarray, Path],
        frame2: Union[np.ndarray, Path],
    ) -> FlowField:
        """Estimate optical flow from frame1 to frame2.

        Args:
            frame1: Source frame (numpy array or path to image).
            frame2: Target frame (numpy array or path to image).

        Returns:
            FlowField containing motion vectors and confidence.
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for optical flow estimation")

        # Load frames if paths
        if isinstance(frame1, (str, Path)):
            img1 = cv2.imread(str(frame1))
        else:
            img1 = frame1

        if isinstance(frame2, (str, Path)):
            img2 = cv2.imread(str(frame2))
        else:
            img2 = frame2

        # Convert to grayscale for flow estimation
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Estimate flow based on method
        if self.method == OpticalFlowMethod.FARNEBACK:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.1,
                flags=0
            )
        elif self.method == OpticalFlowMethod.DIS and self._flow_calculator:
            flow = self._flow_calculator.calc(gray1, gray2, None)
        elif self.method == OpticalFlowMethod.LUCAS_KANADE:
            # Sparse flow - interpolate to dense
            flow = self._estimate_lucas_kanade(gray1, gray2)
        else:
            # Fallback to Farneback
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.1, 0
            )

        # Compute magnitude and confidence
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Confidence based on local flow consistency
        confidence = self._compute_flow_confidence(flow)

        return FlowField(
            flow_x=flow_x,
            flow_y=flow_y,
            magnitude=magnitude,
            confidence=confidence,
            frame_idx_from=0,
            frame_idx_to=1
        )

    def _estimate_lucas_kanade(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
    ) -> np.ndarray:
        """Estimate sparse flow with Lucas-Kanade and interpolate.

        Args:
            gray1: First grayscale frame.
            gray2: Second grayscale frame.

        Returns:
            Dense flow field interpolated from sparse features.
        """
        # Detect features
        features = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=10
        )

        if features is None:
            # No features found, return zero flow
            h, w = gray1.shape
            return np.zeros((h, w, 2), dtype=np.float32)

        # Track features
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, features, None
        )

        # Filter good points
        good_old = features[status == 1]
        good_new = next_pts[status == 1]

        # Create sparse flow vectors
        sparse_flow = good_new - good_old

        # Interpolate to dense flow
        h, w = gray1.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)

        if len(good_old) > 3:
            # Use scipy for interpolation if available
            try:
                from scipy.interpolate import griddata

                points = good_old.reshape(-1, 2)
                values_x = sparse_flow[:, 0]
                values_y = sparse_flow[:, 1]

                grid_y, grid_x = np.mgrid[0:h, 0:w]

                flow[..., 0] = griddata(
                    points, values_x, (grid_y, grid_x),
                    method='linear', fill_value=0
                )
                flow[..., 1] = griddata(
                    points, values_y, (grid_y, grid_x),
                    method='linear', fill_value=0
                )
            except ImportError:
                # Simple nearest-neighbor fallback
                for i, (old, new) in enumerate(zip(good_old, good_new)):
                    x, y = int(old[0]), int(old[1])
                    if 0 <= x < w and 0 <= y < h:
                        dx, dy = new - old
                        flow[y, x] = [dx, dy]

        return flow

    def _compute_flow_confidence(self, flow: np.ndarray) -> np.ndarray:
        """Compute confidence map for optical flow.

        Args:
            flow: Dense optical flow field.

        Returns:
            Confidence map (0-1) indicating flow reliability.
        """
        # Compute local flow variance as confidence indicator
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # Use local variance - low variance = consistent flow = high confidence
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        # Local mean
        mean_x = cv2.filter2D(flow_x, -1, kernel)
        mean_y = cv2.filter2D(flow_y, -1, kernel)

        # Local variance
        var_x = cv2.filter2D((flow_x - mean_x) ** 2, -1, kernel)
        var_y = cv2.filter2D((flow_y - mean_y) ** 2, -1, kernel)

        variance = var_x + var_y

        # Convert variance to confidence (inverse relationship)
        # Normalize and invert
        max_var = np.percentile(variance, 95) + 1e-6
        confidence = 1.0 - np.clip(variance / max_var, 0, 1)

        return confidence.astype(np.float32)

    def warp_frame(
        self,
        frame: np.ndarray,
        flow: FlowField,
        inverse: bool = False,
    ) -> np.ndarray:
        """Warp a frame according to optical flow.

        Args:
            frame: Frame to warp.
            flow: Optical flow field.
            inverse: If True, warp in reverse direction.

        Returns:
            Warped frame aligned to the target.
        """
        h, w = frame.shape[:2]

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        if inverse:
            # Inverse warping (target to source)
            map_x = (x - flow.flow_x).astype(np.float32)
            map_y = (y - flow.flow_y).astype(np.float32)
        else:
            # Forward warping (source to target)
            map_x = (x + flow.flow_x).astype(np.float32)
            map_y = (y + flow.flow_y).astype(np.float32)

        # Remap using bilinear interpolation
        warped = cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return warped


class FlickerReducer:
    """Reduce temporal flickering artifacts in video.

    Flickering is a common artifact in old film where brightness
    varies between frames due to camera or film issues. This class
    implements adaptive temporal filtering to smooth these variations
    while preserving intentional brightness changes.

    Example:
        >>> reducer = FlickerReducer(mode=FlickerMode.ADAPTIVE)
        >>> metrics = reducer.analyze_flicker(frames_dir)
        >>> result = reducer.reduce_flicker(input_dir, output_dir)
    """

    # FFmpeg deflicker filter parameters by mode
    DEFLICKER_PARAMS = {
        FlickerMode.LIGHT: {"size": 3, "mode": "am"},
        FlickerMode.MEDIUM: {"size": 5, "mode": "am"},
        FlickerMode.AGGRESSIVE: {"size": 9, "mode": "gm"},
        FlickerMode.ADAPTIVE: {"size": 5, "mode": "am"},  # Will be adjusted
    }

    def __init__(
        self,
        mode: FlickerMode = FlickerMode.ADAPTIVE,
        preserve_brightness_changes: bool = True,
    ):
        """Initialize flicker reducer.

        Args:
            mode: Flicker reduction aggressiveness.
            preserve_brightness_changes: Whether to preserve intentional
                brightness changes (e.g., fades, lighting changes).
        """
        self.mode = mode
        self.preserve_brightness_changes = preserve_brightness_changes
        self._detected_severity: Optional[float] = None

    def analyze_flicker(
        self,
        frames_dir: Path,
        sample_rate: int = 1,
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """Analyze flickering severity in a frame sequence.

        Args:
            frames_dir: Directory containing frame images.
            sample_rate: Analyze every Nth frame.
            max_samples: Maximum number of frames to analyze.

        Returns:
            Dictionary with flicker metrics:
            - severity: Overall flicker severity (0-1)
            - temporal_variance: Brightness variance over time
            - frequency: Dominant flicker frequency if detected
            - recommended_mode: Suggested FlickerMode based on analysis
        """
        frames_dir = Path(frames_dir)
        frames = sorted(frames_dir.glob("*.png"))

        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        if len(frames) < 3:
            return {
                "severity": 0.0,
                "temporal_variance": 0.0,
                "frequency": 0.0,
                "recommended_mode": FlickerMode.LIGHT.value,
            }

        # Sample frames
        sample_frames = frames[::sample_rate][:max_samples]

        # Calculate brightness for each frame
        brightness_values = []

        for frame_path in sample_frames:
            if HAS_OPENCV:
                img = cv2.imread(str(frame_path))
                if img is not None:
                    # Calculate average brightness
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness_values.append(np.mean(gray))
            elif HAS_PIL:
                img = Image.open(frame_path).convert('L')
                brightness_values.append(np.mean(np.array(img)))

        if len(brightness_values) < 3:
            return {
                "severity": 0.0,
                "temporal_variance": 0.0,
                "frequency": 0.0,
                "recommended_mode": FlickerMode.LIGHT.value,
            }

        brightness = np.array(brightness_values)

        # Calculate metrics
        # 1. Temporal variance (normalized)
        temporal_variance = np.std(brightness) / (np.mean(brightness) + 1e-6)

        # 2. Frame-to-frame differences
        diffs = np.abs(np.diff(brightness))
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)

        # 3. Detect periodic flickering using FFT
        fft = np.fft.fft(brightness - np.mean(brightness))
        power = np.abs(fft[:len(fft)//2]) ** 2

        # Find dominant frequency (excluding DC component)
        if len(power) > 1:
            dominant_freq_idx = np.argmax(power[1:]) + 1
            dominant_power = power[dominant_freq_idx] / (np.sum(power) + 1e-6)
        else:
            dominant_freq_idx = 0
            dominant_power = 0.0

        # Calculate overall severity
        # Combine variance, frame differences, and periodicity
        severity = min(1.0, (
            temporal_variance * 2 +  # Weight variance
            (mean_diff / 255) * 3 +  # Weight frame differences
            dominant_power * 0.5     # Weight periodic component
        ))

        self._detected_severity = severity

        # Recommend mode based on severity
        if severity < 0.1:
            recommended = FlickerMode.LIGHT
        elif severity < 0.3:
            recommended = FlickerMode.MEDIUM
        else:
            recommended = FlickerMode.AGGRESSIVE

        return {
            "severity": float(severity),
            "temporal_variance": float(temporal_variance),
            "frequency": float(dominant_freq_idx),
            "mean_brightness_diff": float(mean_diff),
            "max_brightness_diff": float(max_diff),
            "recommended_mode": recommended.value,
        }

    def reduce_flicker(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, any]:
        """Apply flicker reduction to frame sequence.

        Uses a combination of temporal filtering approaches:
        1. FFmpeg deflicker filter for basic smoothing
        2. Adaptive histogram matching for brightness consistency
        3. Optional frame-by-frame brightness compensation

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary with processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found for flicker reduction")
            return {"frames_processed": 0, "mode_used": None}

        # Adaptive mode adjustment
        mode = self.mode
        if mode == FlickerMode.ADAPTIVE and self._detected_severity is not None:
            if self._detected_severity < 0.1:
                mode = FlickerMode.LIGHT
            elif self._detected_severity < 0.3:
                mode = FlickerMode.MEDIUM
            else:
                mode = FlickerMode.AGGRESSIVE

        # Get deflicker parameters
        params = self.DEFLICKER_PARAMS.get(mode, self.DEFLICKER_PARAMS[FlickerMode.MEDIUM])

        logger.info(f"Applying flicker reduction: mode={mode.value}, params={params}")

        if progress_callback:
            progress_callback(0.1)

        # Method 1: FFmpeg-based deflicker (process as video sequence)
        # Create temporary video, apply filter, extract frames
        result = self._apply_ffmpeg_deflicker(
            input_dir, output_dir, params, progress_callback
        )

        if not result["success"]:
            # Fallback: Python-based brightness normalization
            logger.info("FFmpeg deflicker failed, using Python fallback")
            result = self._apply_python_deflicker(
                input_dir, output_dir, progress_callback
            )

        result["mode_used"] = mode.value

        if progress_callback:
            progress_callback(1.0)

        return result

    def _apply_ffmpeg_deflicker(
        self,
        input_dir: Path,
        output_dir: Path,
        params: Dict,
        progress_callback: Optional[Callable[[float], None]],
    ) -> Dict[str, any]:
        """Apply FFmpeg deflicker filter.

        Args:
            input_dir: Input frames directory.
            output_dir: Output frames directory.
            params: Deflicker filter parameters.
            progress_callback: Progress callback.

        Returns:
            Processing result dictionary.
        """
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        # Build FFmpeg command
        filter_str = f"deflicker=size={params['size']}:mode={params['mode']}"

        # Process frame sequence
        input_pattern = str(input_dir / "frame_%08d.png")

        # Check if files match pattern
        if not (input_dir / "frame_00000001.png").exists():
            # Try to detect actual pattern
            sample = frames[0].name
            if "_" in sample:
                # Extract pattern like "frame_00000001.png"
                parts = sample.rsplit("_", 1)
                if len(parts) == 2:
                    num_part = parts[1].replace(".png", "").replace(".jpg", "")
                    if num_part.isdigit():
                        num_digits = len(num_part)
                        ext = frames[0].suffix
                        input_pattern = str(input_dir / f"{parts[0]}_%0{num_digits}d{ext}")

        output_pattern = str(output_dir / "frame_%08d.png")

        cmd = [
            'ffmpeg', '-y',
            '-i', input_pattern,
            '-vf', filter_str,
            '-q:v', '1',
            output_pattern,
            '-hide_banner', '-loglevel', 'error',
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)

            output_frames = list(output_dir.glob("*.png"))
            return {
                "success": True,
                "frames_processed": len(output_frames),
                "method": "ffmpeg_deflicker",
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.debug(f"FFmpeg deflicker failed: {e}")
            return {"success": False, "frames_processed": 0}

    def _apply_python_deflicker(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
    ) -> Dict[str, any]:
        """Apply Python-based brightness normalization.

        Uses adaptive histogram equalization and temporal smoothing.

        Args:
            input_dir: Input frames directory.
            output_dir: Output frames directory.
            progress_callback: Progress callback.

        Returns:
            Processing result dictionary.
        """
        if not HAS_OPENCV:
            # Just copy frames if OpenCV not available
            frames = sorted(input_dir.glob("*.png"))
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            return {"success": True, "frames_processed": len(frames), "method": "copy"}

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        # Calculate target brightness (median of all frames)
        brightness_values = []
        for frame_path in frames[::10][:50]:  # Sample frames
            img = cv2.imread(str(frame_path))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_values.append(np.mean(gray))

        target_brightness = np.median(brightness_values)

        # Process each frame with brightness compensation
        for i, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                shutil.copy(frame_path, output_dir / frame_path.name)
                continue

            # Convert to LAB for brightness adjustment
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            current_brightness = np.mean(l)

            # Gentle brightness adjustment
            adjustment = target_brightness - current_brightness
            # Limit adjustment to avoid drastic changes
            adjustment = np.clip(adjustment, -20, 20)

            l = np.clip(l.astype(np.float32) + adjustment * 0.5, 0, 255).astype(np.uint8)

            # Reconstruct image
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            cv2.imwrite(str(output_dir / frame_path.name), result)

            if progress_callback:
                progress_callback(0.1 + 0.9 * (i + 1) / len(frames))

        return {
            "success": True,
            "frames_processed": len(frames),
            "method": "python_brightness_normalization",
        }


class TemporalConsistencyFilter:
    """Apply temporal consistency filtering to maintain frame coherence.

    This filter ensures that processed frames maintain temporal consistency
    by enforcing smooth transitions and reducing temporal artifacts like
    jittering, popping, or color shifts between frames.

    Example:
        >>> filter = TemporalConsistencyFilter(strength=0.5)
        >>> result = filter.apply(input_dir, output_dir)
    """

    def __init__(
        self,
        strength: float = 0.5,
        temporal_radius: int = 2,
        use_optical_flow: bool = True,
        flow_estimator: Optional[OpticalFlowEstimator] = None,
    ):
        """Initialize temporal consistency filter.

        Args:
            strength: Filter strength (0-1). Higher = more temporal smoothing.
            temporal_radius: Number of frames to consider on each side.
            use_optical_flow: Use motion compensation for better alignment.
            flow_estimator: Optional pre-configured flow estimator.
        """
        self.strength = strength
        self.temporal_radius = temporal_radius
        self.use_optical_flow = use_optical_flow
        self.flow_estimator = flow_estimator or OpticalFlowEstimator()

    def apply(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, any]:
        """Apply temporal consistency filtering to frame sequence.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional progress callback.

        Returns:
            Processing statistics dictionary.
        """
        if not HAS_OPENCV:
            logger.warning("OpenCV required for temporal consistency filter")
            return {"success": False, "error": "OpenCV not available"}

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            return {"success": False, "error": "No frames found"}

        n_frames = len(frames)
        radius = self.temporal_radius

        # Load all frames into memory for temporal processing
        # For large videos, this should be done in chunks
        logger.info(f"Processing {n_frames} frames with temporal radius {radius}")

        processed = 0

        for i, frame_path in enumerate(frames):
            # Determine temporal window
            start_idx = max(0, i - radius)
            end_idx = min(n_frames, i + radius + 1)

            # Load frames in window
            window_frames = []
            for j in range(start_idx, end_idx):
                img = cv2.imread(str(frames[j]))
                if img is not None:
                    window_frames.append((j, img))

            if not window_frames:
                continue

            # Find center frame
            center_img = cv2.imread(str(frame_path))
            if center_img is None:
                continue

            # Apply temporal filtering
            if self.use_optical_flow and len(window_frames) > 1:
                result = self._apply_flow_guided_filter(
                    center_img, i, window_frames
                )
            else:
                result = self._apply_simple_temporal_filter(
                    center_img, i, window_frames
                )

            # Save result
            cv2.imwrite(str(output_dir / frame_path.name), result)
            processed += 1

            if progress_callback:
                progress_callback((i + 1) / n_frames)

        return {
            "success": True,
            "frames_processed": processed,
            "temporal_radius": radius,
            "flow_guided": self.use_optical_flow,
        }

    def _apply_flow_guided_filter(
        self,
        center: np.ndarray,
        center_idx: int,
        window: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Apply motion-compensated temporal filtering.

        Args:
            center: Center frame to process.
            center_idx: Index of center frame.
            window: List of (index, frame) tuples in temporal window.

        Returns:
            Filtered frame.
        """
        h, w = center.shape[:2]
        accumulated = np.zeros((h, w, 3), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

        for idx, frame in window:
            if idx == center_idx:
                # Center frame gets highest weight
                weight = 1.0
                aligned = frame
            else:
                # Estimate flow and warp
                try:
                    flow = self.flow_estimator.estimate(frame, center)
                    aligned = self.flow_estimator.warp_frame(frame, flow)

                    # Weight based on distance and flow confidence
                    distance = abs(idx - center_idx)
                    temporal_weight = np.exp(-distance * 0.5)

                    # Use flow confidence as spatial weight
                    weight = temporal_weight * (1 - self.strength) + \
                             temporal_weight * self.strength * flow.confidence
                except Exception as e:
                    logger.debug(f"Flow estimation failed: {e}")
                    aligned = frame
                    weight = np.exp(-abs(idx - center_idx) * 0.5)

            # Accumulate weighted contribution
            if isinstance(weight, np.ndarray):
                weight_3d = weight[:, :, np.newaxis]
                accumulated += aligned.astype(np.float64) * weight_3d
                weight_sum += weight
            else:
                accumulated += aligned.astype(np.float64) * weight
                weight_sum += weight

        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-6)
        if weight_sum.ndim == 2:
            weight_sum = weight_sum[:, :, np.newaxis]

        result = (accumulated / weight_sum).astype(np.uint8)

        # Blend with original to control strength
        blend_weight = self.strength
        result = cv2.addWeighted(
            center, 1 - blend_weight,
            result, blend_weight,
            0
        )

        return result

    def _apply_simple_temporal_filter(
        self,
        center: np.ndarray,
        center_idx: int,
        window: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Apply simple weighted temporal averaging.

        Args:
            center: Center frame.
            center_idx: Index of center frame.
            window: List of (index, frame) tuples.

        Returns:
            Filtered frame.
        """
        h, w = center.shape[:2]
        accumulated = np.zeros((h, w, 3), dtype=np.float64)
        weight_sum = 0.0

        for idx, frame in window:
            # Gaussian-like weight based on temporal distance
            distance = abs(idx - center_idx)
            weight = np.exp(-distance * 0.5)

            accumulated += frame.astype(np.float64) * weight
            weight_sum += weight

        result = (accumulated / weight_sum).astype(np.uint8)

        # Blend with original
        result = cv2.addWeighted(
            center, 1 - self.strength * 0.5,
            result, self.strength * 0.5,
            0
        )

        return result


class TemporalDenoiser:
    """Main temporal denoising processor.

    Combines multi-frame context, optical flow guidance, and flicker reduction
    for comprehensive temporal noise reduction.

    The denoiser works by:
    1. Analyzing frame sequence for noise characteristics and scene changes
    2. Estimating optical flow for motion compensation
    3. Applying motion-compensated temporal filtering
    4. Reducing flickering artifacts
    5. Ensuring temporal consistency in output

    Example:
        >>> config = TemporalDenoiseConfig(
        ...     temporal_radius=3,
        ...     noise_strength=0.5,
        ...     enable_optical_flow=True,
        ...     enable_flicker_reduction=True
        ... )
        >>> denoiser = TemporalDenoiser(config)
        >>> result = denoiser.denoise_frames(input_dir, output_dir)
    """

    def __init__(self, config: Optional[TemporalDenoiseConfig] = None):
        """Initialize temporal denoiser.

        Args:
            config: Denoising configuration. Uses defaults if not provided.
        """
        self.config = config or TemporalDenoiseConfig()
        self._flow_estimator = OpticalFlowEstimator(
            method=self.config.optical_flow_method,
            gpu_id=self.config.gpu_id
        )
        self._flicker_reducer = FlickerReducer(
            mode=self.config.flicker_mode
        )
        self._consistency_filter = TemporalConsistencyFilter(
            strength=self.config.noise_strength,
            temporal_radius=self.config.temporal_radius,
            use_optical_flow=self.config.enable_optical_flow,
            flow_estimator=self._flow_estimator
        )
        self._scene_changes: List[int] = []

    def analyze(
        self,
        frames_dir: Path,
        sample_rate: int = 5,
    ) -> Dict[str, any]:
        """Analyze frame sequence for noise and temporal artifacts.

        Args:
            frames_dir: Directory containing frames.
            sample_rate: Analyze every Nth frame.

        Returns:
            Analysis results including noise levels, flicker, scene changes.
        """
        frames_dir = Path(frames_dir)
        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            return {"error": "No frames found"}

        analysis = {
            "total_frames": len(frames),
            "noise_level": 0.0,
            "flicker_metrics": {},
            "scene_changes": [],
            "recommended_config": {},
        }

        # Analyze flicker
        if self.config.enable_flicker_reduction:
            analysis["flicker_metrics"] = self._flicker_reducer.analyze_flicker(
                frames_dir, sample_rate
            )

        # Detect scene changes
        analysis["scene_changes"] = self._detect_scene_changes(frames, sample_rate)
        self._scene_changes = analysis["scene_changes"]

        # Estimate noise level
        analysis["noise_level"] = self._estimate_noise_level(frames, sample_rate)

        # Generate recommendations
        analysis["recommended_config"] = self._generate_recommendations(analysis)

        return analysis

    def _detect_scene_changes(
        self,
        frames: List[Path],
        sample_rate: int,
    ) -> List[int]:
        """Detect scene changes in frame sequence.

        Args:
            frames: List of frame paths.
            sample_rate: Check every Nth pair.

        Returns:
            List of frame indices where scenes change.
        """
        if not HAS_OPENCV or len(frames) < 2:
            return []

        scene_changes = []
        threshold = self.config.scene_change_threshold

        prev_hist = None

        for i in range(0, len(frames) - 1, sample_rate):
            # Load frames
            img1 = cv2.imread(str(frames[i]))
            img2 = cv2.imread(str(frames[min(i + sample_rate, len(frames) - 1)]))

            if img1 is None or img2 is None:
                continue

            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate histograms
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

            # Normalize
            hist1 = hist1 / (hist1.sum() + 1e-6)
            hist2 = hist2 / (hist2.sum() + 1e-6)

            # Compare using correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            if correlation < threshold:
                scene_changes.append(i + sample_rate)
                logger.debug(f"Scene change detected at frame {i + sample_rate}, corr={correlation:.3f}")

        return scene_changes

    def _estimate_noise_level(
        self,
        frames: List[Path],
        sample_rate: int,
    ) -> float:
        """Estimate noise level in frame sequence.

        Uses Laplacian variance as a noise indicator.

        Args:
            frames: List of frame paths.
            sample_rate: Analyze every Nth frame.

        Returns:
            Estimated noise level (0-1).
        """
        if not HAS_OPENCV:
            return 0.0

        noise_estimates = []

        for frame_path in frames[::sample_rate][:50]:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Laplacian for high-frequency content (noise indicator)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            variance = laplacian.var()

            # Normalize to rough 0-1 scale
            # Higher variance = more high-frequency content (could be noise or detail)
            noise_estimates.append(variance)

        if not noise_estimates:
            return 0.0

        # Use percentile to be robust to outliers
        median_noise = np.median(noise_estimates)

        # Normalize (typical values range from 100-10000)
        normalized = np.clip(median_noise / 5000, 0, 1)

        return float(normalized)

    def _generate_recommendations(
        self,
        analysis: Dict[str, any],
    ) -> Dict[str, any]:
        """Generate configuration recommendations based on analysis.

        Args:
            analysis: Analysis results.

        Returns:
            Recommended configuration parameters.
        """
        recommendations = {
            "temporal_radius": self.config.temporal_radius,
            "noise_strength": self.config.noise_strength,
            "enable_flicker_reduction": self.config.enable_flicker_reduction,
            "flicker_mode": self.config.flicker_mode.value,
        }

        noise_level = analysis.get("noise_level", 0.0)
        flicker_severity = analysis.get("flicker_metrics", {}).get("severity", 0.0)

        # Adjust noise strength based on detected noise
        if noise_level < 0.2:
            recommendations["noise_strength"] = 0.3
        elif noise_level < 0.5:
            recommendations["noise_strength"] = 0.5
        else:
            recommendations["noise_strength"] = 0.7

        # Adjust temporal radius based on noise and scene changes
        n_scene_changes = len(analysis.get("scene_changes", []))
        if n_scene_changes > 10:
            # Many scene changes - use smaller radius to avoid cross-scene artifacts
            recommendations["temporal_radius"] = 2
        elif noise_level > 0.5:
            # Heavy noise - use larger radius for better averaging
            recommendations["temporal_radius"] = 4

        # Flicker reduction settings
        if flicker_severity > 0.3:
            recommendations["enable_flicker_reduction"] = True
            recommendations["flicker_mode"] = analysis.get(
                "flicker_metrics", {}
            ).get("recommended_mode", "medium")

        return recommendations

    def denoise_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TemporalDenoiseResult:
        """Apply temporal denoising to frame sequence.

        This is the main processing method that:
        1. Analyzes the input for optimal settings
        2. Applies flicker reduction if enabled
        3. Applies temporal denoising with motion compensation
        4. Ensures temporal consistency

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional progress callback.

        Returns:
            TemporalDenoiseResult with processing statistics.
        """
        import time
        start_time = time.time()

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = TemporalDenoiseResult()

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found for temporal denoising")
            return result

        logger.info(f"Starting temporal denoising on {len(frames)} frames")

        # Phase 1: Analysis (5%)
        if progress_callback:
            progress_callback(0.02)

        analysis = self.analyze(input_dir)
        result.scene_changes_detected = analysis.get("scene_changes", [])

        if progress_callback:
            progress_callback(0.05)

        # Use temp directory for intermediate results
        with tempfile.TemporaryDirectory(prefix="temporal_denoise_") as temp_dir:
            temp_dir = Path(temp_dir)

            current_input = input_dir

            # Phase 2: Flicker reduction (25%)
            if self.config.enable_flicker_reduction:
                logger.info("Applying flicker reduction...")
                flicker_output = temp_dir / "deflickered"
                flicker_output.mkdir()

                flicker_result = self._flicker_reducer.reduce_flicker(
                    current_input,
                    flicker_output,
                    lambda p: progress_callback(0.05 + p * 0.2) if progress_callback else None
                )

                result.flicker_reduction_applied = flicker_result.get("success", False)

                if result.flicker_reduction_applied:
                    current_input = flicker_output

            if progress_callback:
                progress_callback(0.25)

            # Phase 3: Temporal denoising (60%)
            logger.info("Applying temporal denoising...")
            denoise_output = temp_dir / "denoised"
            denoise_output.mkdir()

            denoise_success = self._apply_temporal_denoise(
                current_input,
                denoise_output,
                lambda p: progress_callback(0.25 + p * 0.6) if progress_callback else None
            )

            if denoise_success:
                current_input = denoise_output

            if progress_callback:
                progress_callback(0.85)

            # Phase 4: Final consistency pass (10%)
            logger.info("Applying temporal consistency...")

            consistency_result = self._consistency_filter.apply(
                current_input,
                output_dir,
                lambda p: progress_callback(0.85 + p * 0.1) if progress_callback else None
            )

            result.frames_processed = consistency_result.get("frames_processed", 0)
            result.frames_failed = len(frames) - result.frames_processed

        # Calculate metrics
        result.output_dir = output_dir
        result.processing_time_seconds = time.time() - start_time
        result.avg_noise_reduction = self._estimate_noise_reduction(
            input_dir, output_dir
        )

        if progress_callback:
            progress_callback(1.0)

        logger.info(
            f"Temporal denoising complete: {result.frames_processed} frames, "
            f"{result.processing_time_seconds:.1f}s, "
            f"noise reduction: {result.avg_noise_reduction:.1%}"
        )

        return result

    def _apply_temporal_denoise(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
    ) -> bool:
        """Apply core temporal denoising algorithm.

        Args:
            input_dir: Input frames directory.
            output_dir: Output frames directory.
            progress_callback: Progress callback.

        Returns:
            True if successful.
        """
        if not HAS_OPENCV:
            # Fallback: use FFmpeg temporal denoise
            return self._apply_ffmpeg_temporal_denoise(
                input_dir, output_dir, progress_callback
            )

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            return False

        n_frames = len(frames)
        radius = self.config.temporal_radius
        strength = self.config.noise_strength

        # Process in chunks for memory efficiency
        chunk_size = self.config.chunk_size

        for chunk_start in range(0, n_frames, chunk_size - 2 * radius):
            chunk_end = min(chunk_start + chunk_size, n_frames)

            # Load chunk with padding
            load_start = max(0, chunk_start - radius)
            load_end = min(n_frames, chunk_end + radius)

            # Load frames in chunk
            chunk_frames = []
            for i in range(load_start, load_end):
                img = cv2.imread(str(frames[i]))
                if img is not None:
                    chunk_frames.append((i, img))

            if not chunk_frames:
                continue

            # Process each frame in chunk
            for i in range(chunk_start, chunk_end):
                local_idx = i - load_start

                if i in self._scene_changes:
                    # Skip cross-scene filtering at scene boundaries
                    window_start = local_idx
                    window_end = local_idx + 1
                else:
                    window_start = max(0, local_idx - radius)
                    window_end = min(len(chunk_frames), local_idx + radius + 1)

                # Get window frames
                window = chunk_frames[window_start:window_end]

                if not window or local_idx >= len(chunk_frames):
                    continue

                center_idx, center_frame = chunk_frames[local_idx]

                # Apply temporal filtering
                if self.config.enable_optical_flow and len(window) > 1:
                    result = self._denoise_with_flow(center_frame, local_idx, window)
                else:
                    result = self._denoise_simple(center_frame, window)

                # Apply additional spatial denoising
                if strength > 0.3:
                    result = self._apply_spatial_denoise(result, strength)

                # Preserve edges if requested
                if self.config.preserve_edges:
                    result = self._preserve_edges(center_frame, result)

                # Save result
                cv2.imwrite(str(output_dir / frames[i].name), result)

            if progress_callback:
                progress_callback(chunk_end / n_frames)

        return True

    def _denoise_with_flow(
        self,
        center: np.ndarray,
        center_local_idx: int,
        window: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Apply motion-compensated temporal denoising.

        Args:
            center: Center frame.
            center_local_idx: Index in local window.
            window: List of (global_idx, frame) tuples.

        Returns:
            Denoised frame.
        """
        h, w = center.shape[:2]
        accumulated = np.zeros((h, w, 3), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

        decay = self.config.temporal_weight_decay

        for local_i, (global_idx, frame) in enumerate(window):
            distance = abs(local_i - center_local_idx)

            if distance == 0:
                # Center frame
                weight = np.ones((h, w), dtype=np.float64)
                aligned = frame
            else:
                # Estimate flow and warp
                try:
                    flow = self._flow_estimator.estimate(frame, center)
                    aligned = self._flow_estimator.warp_frame(frame, flow)

                    # Temporal weight with decay
                    temporal_weight = np.exp(-distance * decay)

                    # Spatial weight from flow confidence
                    weight = temporal_weight * flow.confidence

                    # Reduce weight for high-motion regions
                    motion_mask = flow.magnitude > np.percentile(flow.magnitude, 90)
                    weight[motion_mask] *= 0.5

                except Exception as e:
                    logger.debug(f"Flow failed, using unaligned: {e}")
                    aligned = frame
                    weight = np.exp(-distance * decay) * np.ones((h, w))

            # Accumulate
            weight_3d = weight[:, :, np.newaxis]
            accumulated += aligned.astype(np.float64) * weight_3d
            weight_sum += weight

        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-6)[:, :, np.newaxis]
        result = (accumulated / weight_sum).astype(np.uint8)

        return result

    def _denoise_simple(
        self,
        center: np.ndarray,
        window: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Apply simple weighted temporal averaging.

        Args:
            center: Center frame.
            window: List of (index, frame) tuples.

        Returns:
            Denoised frame.
        """
        accumulated = np.zeros_like(center, dtype=np.float64)
        weight_sum = 0.0

        center_idx = len(window) // 2
        decay = self.config.temporal_weight_decay

        for i, (_, frame) in enumerate(window):
            distance = abs(i - center_idx)
            weight = np.exp(-distance * decay)

            accumulated += frame.astype(np.float64) * weight
            weight_sum += weight

        return (accumulated / weight_sum).astype(np.uint8)

    def _apply_spatial_denoise(
        self,
        frame: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Apply additional spatial denoising.

        Args:
            frame: Input frame.
            strength: Denoising strength.

        Returns:
            Spatially denoised frame.
        """
        # Use non-local means for spatial denoising
        # Strength parameters scale with input strength
        h_value = int(3 + strength * 7)  # 3-10
        template_window = 7
        search_window = 21

        return cv2.fastNlMeansDenoisingColored(
            frame, None, h_value, h_value,
            template_window, search_window
        )

    def _preserve_edges(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
    ) -> np.ndarray:
        """Preserve edges from original frame in denoised output.

        Args:
            original: Original frame.
            denoised: Denoised frame.

        Returns:
            Frame with preserved edges.
        """
        # Detect edges in original
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.config.edge_threshold, self.config.edge_threshold * 3)

        # Dilate edges for smoother transition
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Create edge mask
        edge_mask = edges.astype(np.float32) / 255.0
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 0)
        edge_mask = edge_mask[:, :, np.newaxis]

        # Blend: use original at edges, denoised elsewhere
        result = original.astype(np.float32) * edge_mask + \
                 denoised.astype(np.float32) * (1 - edge_mask)

        return result.astype(np.uint8)

    def _apply_ffmpeg_temporal_denoise(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
    ) -> bool:
        """Apply FFmpeg-based temporal denoising as fallback.

        Args:
            input_dir: Input frames directory.
            output_dir: Output frames directory.
            progress_callback: Progress callback.

        Returns:
            True if successful.
        """
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            return False

        # Build hqdn3d filter (spatial + temporal)
        strength = self.config.noise_strength
        spatial_luma = int(2 + strength * 4)
        spatial_chroma = int(1 + strength * 3)
        temporal_luma = int(2 + strength * 4)
        temporal_chroma = int(1 + strength * 3)

        filter_str = (
            f"hqdn3d={spatial_luma}:{spatial_chroma}:{temporal_luma}:{temporal_chroma}"
        )

        # Detect input pattern
        sample = frames[0].name
        ext = frames[0].suffix

        # Try to build input pattern
        input_pattern = str(input_dir / f"frame_%08d{ext}")
        output_pattern = str(output_dir / f"frame_%08d{ext}")

        cmd = [
            'ffmpeg', '-y',
            '-i', input_pattern,
            '-vf', filter_str,
            '-q:v', '1',
            output_pattern,
            '-hide_banner', '-loglevel', 'error',
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1800)

            if progress_callback:
                progress_callback(1.0)

            return True
        except Exception as e:
            logger.debug(f"FFmpeg temporal denoise failed: {e}")
            # Fallback: just copy frames
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            return False

    def _estimate_noise_reduction(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> float:
        """Estimate noise reduction achieved.

        Args:
            input_dir: Original frames directory.
            output_dir: Denoised frames directory.

        Returns:
            Estimated noise reduction ratio (0-1).
        """
        if not HAS_OPENCV:
            return 0.0

        input_frames = sorted(input_dir.glob("*.png"))[:20]
        if not input_frames:
            input_frames = sorted(input_dir.glob("*.jpg"))[:20]

        if not input_frames:
            return 0.0

        input_noise = []
        output_noise = []

        for frame_path in input_frames:
            output_path = output_dir / frame_path.name

            if not output_path.exists():
                continue

            # Load and compute Laplacian variance (noise indicator)
            img_in = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            img_out = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

            if img_in is None or img_out is None:
                continue

            input_noise.append(cv2.Laplacian(img_in, cv2.CV_64F).var())
            output_noise.append(cv2.Laplacian(img_out, cv2.CV_64F).var())

        if not input_noise or not output_noise:
            return 0.0

        avg_input = np.mean(input_noise)
        avg_output = np.mean(output_noise)

        if avg_input <= 0:
            return 0.0

        # Reduction ratio (clamped to reasonable range)
        reduction = 1 - (avg_output / avg_input)
        return float(np.clip(reduction, 0, 1))


class AutoTemporalDenoiser:
    """Automated temporal denoising with optimal settings detection.

    This class automatically analyzes the input video and determines
    optimal denoising parameters, then applies the processing.

    Example:
        >>> denoiser = AutoTemporalDenoiser()
        >>> result = denoiser.process(input_dir, output_dir)
    """

    def __init__(
        self,
        gpu_id: int = 0,
        quality_preset: str = "balanced",
    ):
        """Initialize auto denoiser.

        Args:
            gpu_id: GPU device ID for processing.
            quality_preset: One of 'fast', 'balanced', 'quality'.
        """
        self.gpu_id = gpu_id
        self.quality_preset = quality_preset

        # Preset configurations
        self.presets = {
            "fast": TemporalDenoiseConfig(
                temporal_radius=2,
                enable_optical_flow=False,
                enable_flicker_reduction=True,
                flicker_mode=FlickerMode.LIGHT,
            ),
            "balanced": TemporalDenoiseConfig(
                temporal_radius=3,
                enable_optical_flow=True,
                optical_flow_method=OpticalFlowMethod.DIS,
                enable_flicker_reduction=True,
                flicker_mode=FlickerMode.ADAPTIVE,
            ),
            "quality": TemporalDenoiseConfig(
                temporal_radius=4,
                enable_optical_flow=True,
                optical_flow_method=OpticalFlowMethod.FARNEBACK,
                enable_flicker_reduction=True,
                flicker_mode=FlickerMode.ADAPTIVE,
                preserve_edges=True,
            ),
        }

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[TemporalDenoiseResult, Dict[str, any]]:
        """Process frame sequence with automatic parameter detection.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (processing result, analysis data).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Start with preset configuration
        config = self.presets.get(self.quality_preset, self.presets["balanced"])
        config.gpu_id = self.gpu_id

        # Create denoiser for analysis
        denoiser = TemporalDenoiser(config)

        # Analyze input
        if progress_callback:
            progress_callback(0.02)

        analysis = denoiser.analyze(input_dir)

        # Update config based on analysis recommendations
        recommendations = analysis.get("recommended_config", {})

        if recommendations:
            config.noise_strength = recommendations.get("noise_strength", config.noise_strength)
            config.temporal_radius = recommendations.get("temporal_radius", config.temporal_radius)

            if recommendations.get("flicker_mode"):
                config.flicker_mode = FlickerMode(recommendations["flicker_mode"])

        # Create updated denoiser
        denoiser = TemporalDenoiser(config)

        # Process
        result = denoiser.denoise_frames(input_dir, output_dir, progress_callback)

        return result, analysis


# Convenience functions

def create_temporal_denoiser(
    strength: float = 0.5,
    temporal_radius: int = 3,
    enable_optical_flow: bool = True,
    enable_flicker_reduction: bool = True,
    gpu_id: int = 0,
) -> TemporalDenoiser:
    """Create a temporal denoiser with common settings.

    Args:
        strength: Denoising strength (0-1).
        temporal_radius: Number of frames to consider on each side.
        enable_optical_flow: Use motion compensation.
        enable_flicker_reduction: Enable flicker reduction.
        gpu_id: GPU device ID.

    Returns:
        Configured TemporalDenoiser instance.
    """
    config = TemporalDenoiseConfig(
        noise_strength=strength,
        temporal_radius=temporal_radius,
        enable_optical_flow=enable_optical_flow,
        enable_flicker_reduction=enable_flicker_reduction,
        gpu_id=gpu_id,
    )
    return TemporalDenoiser(config)


def denoise_video_frames(
    input_dir: Path,
    output_dir: Path,
    strength: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> TemporalDenoiseResult:
    """Convenience function to denoise video frames.

    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        strength: Denoising strength (0-1).
        progress_callback: Optional progress callback.

    Returns:
        TemporalDenoiseResult with processing statistics.
    """
    denoiser = create_temporal_denoiser(strength=strength)
    return denoiser.denoise_frames(input_dir, output_dir, progress_callback)


def auto_denoise_video(
    input_dir: Path,
    output_dir: Path,
    quality: str = "balanced",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[TemporalDenoiseResult, Dict[str, any]]:
    """Automatically denoise video with optimal settings.

    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        quality: Quality preset ('fast', 'balanced', 'quality').
        progress_callback: Optional progress callback.

    Returns:
        Tuple of (result, analysis).
    """
    denoiser = AutoTemporalDenoiser(quality_preset=quality)
    return denoiser.process(input_dir, output_dir, progress_callback)
