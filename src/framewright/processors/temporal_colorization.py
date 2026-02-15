"""Temporal Colorization Consistency Processor.

This module provides temporal consistency for colorized video frames,
reducing color flickering that occurs when colorizing frame-by-frame.

Key features:
- Bidirectional temporal fusion for consistent colors across frames
- Optical flow-guided color propagation
- Scene-aware color consistency (respects scene boundaries)
- Integration with DDColor, DeOldify, and SwinTExCo outputs

The temporal colorization pipeline works by:
1. Colorizing frames using the base model (DDColor/DeOldify)
2. Estimating optical flow between consecutive frames
3. Propagating colors bidirectionally using flow-guided warping
4. Blending propagated colors with original colorization

Example:
    >>> config = TemporalColorizationConfig(
    ...     temporal_window=7,
    ...     propagation_mode="bidirectional",
    ...     blend_strength=0.6
    ... )
    >>> processor = TemporalColorizationProcessor(config)
    >>> result = processor.apply_temporal_consistency(
    ...     colorized_dir, output_dir, progress_callback
    ... )
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available - temporal colorization limited")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.debug("PyTorch not available - advanced features disabled")


class PropagationMode(Enum):
    """Color propagation direction modes."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


class BlendMethod(Enum):
    """Methods for blending propagated colors with original."""
    LINEAR = "linear"
    GAUSSIAN = "gaussian"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE = "adaptive"


@dataclass
class TemporalColorizationConfig:
    """Configuration for temporal colorization consistency.

    Attributes:
        temporal_window: Number of frames to consider on each side
        propagation_mode: Direction of color propagation
        blend_strength: Strength of temporal blending (0-1)
        blend_method: Method for blending colors
        optical_flow_method: Flow estimation method (farneback, dis, raft)
        scene_change_threshold: SSIM threshold for scene boundary detection
        preserve_saturation: Maintain original saturation levels
        color_smoothing_sigma: Sigma for color smoothing filter
        gpu_id: GPU device ID for accelerated processing
        half_precision: Use FP16 for reduced VRAM
    """
    temporal_window: int = 7
    propagation_mode: PropagationMode = PropagationMode.BIDIRECTIONAL
    blend_strength: float = 0.6
    blend_method: BlendMethod = BlendMethod.CONFIDENCE_WEIGHTED
    optical_flow_method: str = "farneback"  # farneback, dis, raft
    scene_change_threshold: float = 0.7
    preserve_saturation: bool = True
    color_smoothing_sigma: float = 1.5
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.propagation_mode, str):
            self.propagation_mode = PropagationMode(self.propagation_mode)
        if isinstance(self.blend_method, str):
            self.blend_method = BlendMethod(self.blend_method)
        if not 0.0 <= self.blend_strength <= 1.0:
            raise ValueError(f"blend_strength must be 0-1, got {self.blend_strength}")
        if self.temporal_window < 1:
            raise ValueError(f"temporal_window must be >= 1, got {self.temporal_window}")


@dataclass
class TemporalColorizationResult:
    """Result of temporal colorization processing.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        scene_changes_detected: Frame indices where scenes changed
        avg_color_consistency: Average color consistency score (0-1)
        processing_time_seconds: Total processing time
        output_dir: Path to output directory
    """
    frames_processed: int = 0
    frames_failed: int = 0
    scene_changes_detected: List[int] = field(default_factory=list)
    avg_color_consistency: float = 0.0
    processing_time_seconds: float = 0.0
    output_dir: Optional[Path] = None


class OpticalFlowColorPropagator:
    """Propagate colors using optical flow guidance.

    This class handles flow-based color propagation between frames,
    warping chrominance channels according to estimated motion.
    """

    def __init__(
        self,
        method: str = "farneback",
        gpu_id: int = 0,
    ):
        """Initialize flow propagator.

        Args:
            method: Flow estimation method (farneback, dis, raft)
            gpu_id: GPU device ID
        """
        self.method = method
        self.gpu_id = gpu_id
        self._flow_calculator = None
        self._raft_model = None
        self._init_flow_calculator()

    def _init_flow_calculator(self) -> None:
        """Initialize optical flow calculator."""
        if not HAS_OPENCV:
            return

        if self.method == "dis":
            self._flow_calculator = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            )
        elif self.method == "raft" and HAS_TORCH:
            self._init_raft_model()

    def _init_raft_model(self) -> None:
        """Initialize RAFT optical flow model."""
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

            self._raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
            self._raft_model = self._raft_model.to(device)
            self._raft_model.eval()
            logger.info("RAFT optical flow model initialized")
        except ImportError:
            logger.warning("RAFT not available, falling back to Farneback")
            self.method = "farneback"
        except Exception as e:
            logger.warning(f"Failed to load RAFT: {e}, falling back to Farneback")
            self.method = "farneback"

    def estimate_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate optical flow from frame1 to frame2.

        Args:
            frame1: Source frame (BGR)
            frame2: Target frame (BGR)

        Returns:
            Tuple of (flow_x, flow_y) motion vectors
        """
        if not HAS_OPENCV:
            h, w = frame1.shape[:2]
            return np.zeros((h, w)), np.zeros((h, w))

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if self.method == "raft" and self._raft_model is not None:
            return self._estimate_flow_raft(frame1, frame2)
        elif self.method == "dis" and self._flow_calculator is not None:
            flow = self._flow_calculator.calc(gray1, gray2, None)
        else:
            # Farneback
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )

        return flow[..., 0], flow[..., 1]

    def _estimate_flow_raft(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate flow using RAFT model."""
        device = next(self._raft_model.parameters()).device

        # Preprocess frames
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        img1_t = img1_t.to(device)
        img2_t = img2_t.to(device)

        with torch.no_grad():
            flow = self._raft_model(img1_t, img2_t)[-1]

        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow[..., 0], flow[..., 1]

    def compute_flow_confidence(
        self,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
    ) -> np.ndarray:
        """Compute confidence map for optical flow.

        Args:
            flow_x: Horizontal flow
            flow_y: Vertical flow

        Returns:
            Confidence map (0-1)
        """
        # Magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Local variance as consistency indicator
        kernel = np.ones((5, 5)) / 25
        mean_x = cv2.filter2D(flow_x, -1, kernel)
        mean_y = cv2.filter2D(flow_y, -1, kernel)
        var_x = cv2.filter2D((flow_x - mean_x)**2, -1, kernel)
        var_y = cv2.filter2D((flow_y - mean_y)**2, -1, kernel)
        variance = var_x + var_y

        # Low variance = consistent flow = high confidence
        max_var = np.percentile(variance, 95) + 1e-6
        confidence = 1.0 - np.clip(variance / max_var, 0, 1)

        # Reduce confidence for very large motions
        motion_penalty = np.clip(magnitude / 50.0, 0, 0.5)
        confidence = confidence * (1 - motion_penalty)

        return confidence.astype(np.float32)

    def warp_chrominance(
        self,
        frame: np.ndarray,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp chrominance channels using optical flow.

        Args:
            frame: Source frame (BGR)
            flow_x: Horizontal flow
            flow_y: Vertical flow
            confidence: Flow confidence map

        Returns:
            Tuple of (warped_a, warped_b) chrominance channels
        """
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        h, w = frame.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Inverse warping (target to source coordinates)
        map_x = (x - flow_x).astype(np.float32)
        map_y = (y - flow_y).astype(np.float32)

        # Warp chrominance channels
        warped_a = cv2.remap(lab[:, :, 1], map_x, map_y,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_b = cv2.remap(lab[:, :, 2], map_x, map_y,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return warped_a, warped_b


class TemporalColorizationProcessor:
    """Apply temporal consistency to colorized video frames.

    This processor takes colorized frames (from DDColor, DeOldify, etc.)
    and applies temporal consistency to reduce color flickering between
    frames while preserving intentional color changes.

    Example:
        >>> config = TemporalColorizationConfig(
        ...     temporal_window=7,
        ...     propagation_mode="bidirectional"
        ... )
        >>> processor = TemporalColorizationProcessor(config)
        >>> result = processor.apply_temporal_consistency(
        ...     colorized_dir, output_dir
        ... )
    """

    def __init__(self, config: Optional[TemporalColorizationConfig] = None):
        """Initialize temporal colorization processor.

        Args:
            config: Processing configuration
        """
        self.config = config or TemporalColorizationConfig()
        self._flow_propagator = OpticalFlowColorPropagator(
            method=self.config.optical_flow_method,
            gpu_id=self.config.gpu_id,
        )
        self._scene_boundaries: List[int] = []

    def detect_scene_changes(
        self,
        frames_dir: Path,
        sample_rate: int = 1,
    ) -> List[int]:
        """Detect scene boundaries in frame sequence.

        Args:
            frames_dir: Directory containing frames
            sample_rate: Check every Nth frame pair

        Returns:
            List of frame indices where scenes change
        """
        if not HAS_OPENCV:
            return []

        frames = sorted(Path(frames_dir).glob("*.png"))
        if not frames:
            frames = sorted(Path(frames_dir).glob("*.jpg"))

        if len(frames) < 2:
            return []

        scene_changes = []
        threshold = self.config.scene_change_threshold

        prev_hist = None
        for i, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-6)

            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < threshold:
                    scene_changes.append(i)
                    logger.debug(f"Scene change at frame {i}, correlation={correlation:.3f}")

            prev_hist = hist

        self._scene_boundaries = scene_changes
        return scene_changes

    def _is_in_same_scene(self, frame_idx1: int, frame_idx2: int) -> bool:
        """Check if two frames are in the same scene."""
        for boundary in self._scene_boundaries:
            if min(frame_idx1, frame_idx2) < boundary <= max(frame_idx1, frame_idx2):
                return False
        return True

    def _propagate_colors_forward(
        self,
        frames: List[Path],
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
    ) -> Dict[int, np.ndarray]:
        """Propagate colors forward through the sequence.

        Args:
            frames: List of frame paths
            output_dir: Output directory
            progress_callback: Progress callback

        Returns:
            Dictionary mapping frame index to chrominance data
        """
        propagated_colors = {}
        window = self.config.temporal_window

        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            current_a = lab[:, :, 1]
            current_b = lab[:, :, 2]

            # Accumulate propagated colors from previous frames
            if i > 0 and self.config.propagation_mode in [
                PropagationMode.FORWARD, PropagationMode.BIDIRECTIONAL
            ]:
                accumulated_a = np.zeros_like(current_a)
                accumulated_b = np.zeros_like(current_b)
                weight_sum = np.zeros_like(current_a)

                for j in range(max(0, i - window), i):
                    if not self._is_in_same_scene(j, i):
                        continue

                    prev_frame = cv2.imread(str(frames[j]))
                    if prev_frame is None:
                        continue

                    # Estimate flow from j to i
                    flow_x, flow_y = self._flow_propagator.estimate_flow(prev_frame, frame)
                    confidence = self._flow_propagator.compute_flow_confidence(flow_x, flow_y)

                    # Warp chrominance from previous frame
                    warped_a, warped_b = self._flow_propagator.warp_chrominance(
                        prev_frame, flow_x, flow_y, confidence
                    )

                    # Temporal weight decay
                    temporal_weight = np.exp(-(i - j) * 0.3)
                    weight = confidence * temporal_weight

                    accumulated_a += warped_a * weight
                    accumulated_b += warped_b * weight
                    weight_sum += weight

                # Normalize and blend
                if weight_sum.max() > 0:
                    weight_sum = np.maximum(weight_sum, 1e-6)
                    propagated_a = accumulated_a / weight_sum
                    propagated_b = accumulated_b / weight_sum

                    # Blend with current frame
                    blend = self.config.blend_strength
                    current_a = current_a * (1 - blend) + propagated_a * blend
                    current_b = current_b * (1 - blend) + propagated_b * blend

            propagated_colors[i] = np.stack([current_a, current_b], axis=-1)

            if progress_callback:
                progress_callback(0.5 * (i + 1) / len(frames))

        return propagated_colors

    def _propagate_colors_backward(
        self,
        frames: List[Path],
        forward_colors: Dict[int, np.ndarray],
        progress_callback: Optional[Callable[[float], None]],
    ) -> Dict[int, np.ndarray]:
        """Propagate colors backward and merge with forward pass.

        Args:
            frames: List of frame paths
            forward_colors: Forward propagated colors
            progress_callback: Progress callback

        Returns:
            Dictionary with merged bidirectional colors
        """
        if self.config.propagation_mode != PropagationMode.BIDIRECTIONAL:
            return forward_colors

        merged_colors = {}
        window = self.config.temporal_window
        n_frames = len(frames)

        for i in range(n_frames - 1, -1, -1):
            frame = cv2.imread(str(frames[i]))
            if frame is None:
                merged_colors[i] = forward_colors.get(i)
                continue

            forward_ab = forward_colors.get(i)
            if forward_ab is None:
                continue

            current_a = forward_ab[:, :, 0]
            current_b = forward_ab[:, :, 1]

            # Propagate from future frames
            accumulated_a = np.zeros_like(current_a)
            accumulated_b = np.zeros_like(current_b)
            weight_sum = np.zeros_like(current_a)

            for j in range(min(n_frames, i + window + 1), i, -1):
                j_idx = j - 1
                if j_idx >= n_frames:
                    continue
                if not self._is_in_same_scene(i, j_idx):
                    continue

                next_frame = cv2.imread(str(frames[j_idx]))
                if next_frame is None:
                    continue

                # Estimate flow from j to i
                flow_x, flow_y = self._flow_propagator.estimate_flow(next_frame, frame)
                confidence = self._flow_propagator.compute_flow_confidence(flow_x, flow_y)

                # Warp chrominance from future frame
                warped_a, warped_b = self._flow_propagator.warp_chrominance(
                    next_frame, flow_x, flow_y, confidence
                )

                temporal_weight = np.exp(-(j_idx - i) * 0.3)
                weight = confidence * temporal_weight

                accumulated_a += warped_a * weight
                accumulated_b += warped_b * weight
                weight_sum += weight

            # Merge backward with forward
            if weight_sum.max() > 0:
                weight_sum = np.maximum(weight_sum, 1e-6)
                backward_a = accumulated_a / weight_sum
                backward_b = accumulated_b / weight_sum

                # Equal blend of forward and backward
                merged_a = (current_a + backward_a) / 2
                merged_b = (current_b + backward_b) / 2
            else:
                merged_a = current_a
                merged_b = current_b

            merged_colors[i] = np.stack([merged_a, merged_b], axis=-1)

            if progress_callback:
                progress_callback(0.5 + 0.5 * (n_frames - i) / n_frames)

        return merged_colors

    def apply_temporal_consistency(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TemporalColorizationResult:
        """Apply temporal consistency to colorized frames.

        Args:
            input_dir: Directory containing colorized frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback

        Returns:
            TemporalColorizationResult with processing statistics
        """
        result = TemporalColorizationResult()
        start_time = time.time()

        if not HAS_OPENCV:
            logger.error("OpenCV required for temporal colorization")
            return result

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        # Get frames
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        n_frames = len(frames)
        logger.info(f"Applying temporal colorization consistency to {n_frames} frames")

        # Detect scene changes
        scene_changes = self.detect_scene_changes(input_dir)
        result.scene_changes_detected = scene_changes
        logger.info(f"Detected {len(scene_changes)} scene changes")

        # Forward propagation
        forward_colors = self._propagate_colors_forward(
            frames, output_dir,
            lambda p: progress_callback(p * 0.4) if progress_callback else None
        )

        # Backward propagation and merge (if bidirectional)
        merged_colors = self._propagate_colors_backward(
            frames, forward_colors,
            lambda p: progress_callback(0.4 + p * 0.4) if progress_callback else None
        )

        # Apply merged colors and save
        color_consistencies = []

        for i, frame_path in enumerate(frames):
            try:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    result.frames_failed += 1
                    continue

                # Get luminance from original
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                luminance = lab[:, :, 0]

                # Get merged chrominance
                merged_ab = merged_colors.get(i)
                if merged_ab is not None:
                    # Apply color smoothing
                    if self.config.color_smoothing_sigma > 0:
                        merged_a = cv2.GaussianBlur(
                            merged_ab[:, :, 0],
                            (0, 0),
                            self.config.color_smoothing_sigma
                        )
                        merged_b = cv2.GaussianBlur(
                            merged_ab[:, :, 1],
                            (0, 0),
                            self.config.color_smoothing_sigma
                        )
                    else:
                        merged_a = merged_ab[:, :, 0]
                        merged_b = merged_ab[:, :, 1]

                    # Preserve original saturation if requested
                    if self.config.preserve_saturation:
                        orig_a = lab[:, :, 1]
                        orig_b = lab[:, :, 2]
                        orig_sat = np.sqrt(orig_a**2 + orig_b**2) + 1e-6
                        new_sat = np.sqrt(merged_a**2 + merged_b**2) + 1e-6
                        scale = orig_sat / new_sat
                        merged_a = merged_a * scale
                        merged_b = merged_b * scale

                    # Reconstruct LAB
                    result_lab = np.stack([
                        luminance,
                        np.clip(merged_a, 0, 255),
                        np.clip(merged_b, 0, 255)
                    ], axis=-1).astype(np.uint8)
                else:
                    result_lab = lab.astype(np.uint8)

                # Convert back to BGR
                result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

                # Save output
                output_path = output_dir / frame_path.name
                cv2.imwrite(str(output_path), result_bgr)

                # Calculate color consistency with neighbors
                if i > 0:
                    prev_output = cv2.imread(str(output_dir / frames[i-1].name))
                    if prev_output is not None:
                        consistency = self._calculate_color_consistency(
                            prev_output, result_bgr
                        )
                        color_consistencies.append(consistency)

                result.frames_processed += 1

            except Exception as e:
                logger.error(f"Failed to process frame {frame_path}: {e}")
                result.frames_failed += 1
                # Copy original as fallback
                try:
                    shutil.copy2(frame_path, output_dir / frame_path.name)
                except Exception:
                    pass

            if progress_callback:
                progress_callback(0.8 + 0.2 * (i + 1) / n_frames)

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time
        if color_consistencies:
            result.avg_color_consistency = float(np.mean(color_consistencies))

        logger.info(
            f"Temporal colorization complete: {result.frames_processed}/{n_frames} frames, "
            f"consistency: {result.avg_color_consistency:.3f}, "
            f"time: {result.processing_time_seconds:.1f}s"
        )

        return result

    def _calculate_color_consistency(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """Calculate color consistency between two frames.

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)

        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        # Convert to LAB
        lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate chrominance difference
        a_diff = np.abs(lab1[:, :, 1] - lab2[:, :, 1]).mean()
        b_diff = np.abs(lab1[:, :, 2] - lab2[:, :, 2]).mean()

        # Convert to consistency score (lower diff = higher consistency)
        avg_diff = (a_diff + b_diff) / 2
        consistency = 1.0 - np.clip(avg_diff / 50.0, 0, 1)

        return float(consistency)


def create_temporal_colorization_processor(
    temporal_window: int = 7,
    propagation_mode: str = "bidirectional",
    blend_strength: float = 0.6,
    optical_flow_method: str = "farneback",
    gpu_id: int = 0,
) -> TemporalColorizationProcessor:
    """Factory function to create a temporal colorization processor.

    Args:
        temporal_window: Number of frames to consider
        propagation_mode: Color propagation direction
        blend_strength: Temporal blend strength
        optical_flow_method: Flow estimation method
        gpu_id: GPU device ID

    Returns:
        Configured TemporalColorizationProcessor
    """
    config = TemporalColorizationConfig(
        temporal_window=temporal_window,
        propagation_mode=PropagationMode(propagation_mode),
        blend_strength=blend_strength,
        optical_flow_method=optical_flow_method,
        gpu_id=gpu_id,
    )
    return TemporalColorizationProcessor(config)


def apply_temporal_colorization(
    input_dir: Path,
    output_dir: Path,
    temporal_window: int = 7,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> TemporalColorizationResult:
    """Convenience function to apply temporal colorization.

    Args:
        input_dir: Directory containing colorized frames
        output_dir: Directory for output frames
        temporal_window: Number of frames to consider
        progress_callback: Optional progress callback

    Returns:
        TemporalColorizationResult with processing statistics
    """
    processor = create_temporal_colorization_processor(
        temporal_window=temporal_window
    )
    return processor.apply_temporal_consistency(
        input_dir, output_dir, progress_callback
    )
