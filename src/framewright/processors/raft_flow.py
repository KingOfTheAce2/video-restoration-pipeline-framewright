"""RAFT Optical Flow Estimator for video restoration.

This module provides RAFT (Recurrent All-Pairs Field Transforms) optical flow
estimation, which offers significantly better accuracy than traditional methods
like Farneback or Lucas-Kanade.

RAFT advantages:
- 5-15% better accuracy than classical methods
- Handles large displacements well
- Better occlusion handling
- More robust to textureless regions

Requirements:
- PyTorch >= 1.9
- torchvision with optical_flow module (or standalone RAFT weights)

Model Sources:
- torchvision.models.optical_flow (recommended)
- https://github.com/princeton-vl/RAFT (original implementation)

Example:
    >>> estimator = RAFTFlowEstimator(gpu_id=0)
    >>> if estimator.is_available():
    ...     flow = estimator.estimate(frame1, frame2)
    ...     warped = estimator.warp_frame(frame1, flow)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class RAFTFlowField:
    """RAFT optical flow field with metadata.

    Attributes:
        flow_x: Horizontal flow component (pixels)
        flow_y: Vertical flow component (pixels)
        magnitude: Flow magnitude at each pixel
        confidence: Confidence/quality map
        iterations_used: Number of RAFT iterations used
    """
    flow_x: np.ndarray
    flow_y: np.ndarray
    magnitude: np.ndarray
    confidence: np.ndarray
    iterations_used: int = 12


class RAFTFlowEstimator:
    """RAFT-based optical flow estimation.

    Uses RAFT (Recurrent All-Pairs Field Transforms) for high-quality
    dense optical flow estimation between video frames.

    Example:
        >>> estimator = RAFTFlowEstimator(gpu_id=0, iterations=12)
        >>> if estimator.is_available():
        ...     flow = estimator.estimate(frame1, frame2)
        ...     print(f"Max displacement: {flow.magnitude.max():.1f} pixels")
    """

    # Default model directory
    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'raft'

    def __init__(
        self,
        gpu_id: int = 0,
        iterations: int = 12,
        model_dir: Optional[Path] = None,
        half_precision: bool = True,
    ):
        """Initialize RAFT flow estimator.

        Args:
            gpu_id: GPU device ID
            iterations: Number of RAFT recurrent iterations (more = better but slower)
            model_dir: Directory for model weights
            half_precision: Use FP16 for reduced VRAM
        """
        self.gpu_id = gpu_id
        self.iterations = iterations
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self.half_precision = half_precision
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available RAFT backend."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available - RAFT disabled")
            return None

        # Try torchvision RAFT first (recommended)
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            logger.info("Found torchvision RAFT backend")
            return 'torchvision'
        except ImportError:
            pass

        # Try torchvision small model
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            logger.info("Found torchvision RAFT-Small backend")
            return 'torchvision_small'
        except ImportError:
            pass

        # Check for standalone RAFT weights
        weights_path = self.model_dir / 'raft-things.pth'
        if weights_path.exists():
            logger.info(f"Found RAFT weights at {weights_path}")
            return 'weights_only'

        logger.warning(
            "RAFT not available. Install with: pip install torchvision>=0.13 "
            "or download weights to ~/.framewright/models/raft/"
        )
        return None

    def is_available(self) -> bool:
        """Check if RAFT flow estimation is available.

        Returns:
            True if RAFT can be used
        """
        return self._backend is not None

    def _load_model(self) -> None:
        """Load RAFT model."""
        if self._model is not None:
            return

        self._device = torch.device(
            f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu'
        )

        if self._backend == 'torchvision':
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self._model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        elif self._backend == 'torchvision_small':
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            raise RuntimeError(f"Cannot load model for backend: {self._backend}")

        self._model = self._model.to(self._device)
        self._model.eval()

        if self.half_precision and self._device.type == 'cuda':
            self._model = self._model.half()

        logger.info(f"RAFT model loaded on {self._device}")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for RAFT input.

        Args:
            frame: BGR frame as numpy array

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float() / 255.0

        # Reshape to [1, C, H, W]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        # Move to device
        tensor = tensor.to(self._device)

        if self.half_precision and self._device.type == 'cuda':
            tensor = tensor.half()

        return tensor

    def estimate(
        self,
        frame1: Union[np.ndarray, Path],
        frame2: Union[np.ndarray, Path],
    ) -> RAFTFlowField:
        """Estimate optical flow from frame1 to frame2.

        Args:
            frame1: Source frame (BGR numpy array or path)
            frame2: Target frame (BGR numpy array or path)

        Returns:
            RAFTFlowField with flow vectors and confidence
        """
        if not self.is_available():
            raise RuntimeError("RAFT is not available")

        # Load model if needed
        self._load_model()

        # Load frames if paths
        if isinstance(frame1, (str, Path)):
            frame1 = cv2.imread(str(frame1))
        if isinstance(frame2, (str, Path)):
            frame2 = cv2.imread(str(frame2))

        if frame1 is None or frame2 is None:
            raise ValueError("Failed to load input frames")

        # Preprocess
        img1 = self._preprocess_frame(frame1)
        img2 = self._preprocess_frame(frame2)

        # Run RAFT
        with torch.no_grad():
            # RAFT returns list of flow predictions at each iteration
            flow_predictions = self._model(img1, img2, num_flow_updates=self.iterations)

            # Take the final prediction
            flow = flow_predictions[-1]

        # Convert to numpy
        flow = flow.squeeze(0).permute(1, 2, 0)
        if self.half_precision:
            flow = flow.float()
        flow = flow.cpu().numpy()

        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        # Compute magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Compute confidence based on flow smoothness
        confidence = self._compute_confidence(flow_x, flow_y)

        return RAFTFlowField(
            flow_x=flow_x,
            flow_y=flow_y,
            magnitude=magnitude,
            confidence=confidence,
            iterations_used=self.iterations,
        )

    def _compute_confidence(
        self,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
    ) -> np.ndarray:
        """Compute confidence map for flow.

        Uses local flow consistency as confidence indicator.

        Args:
            flow_x: Horizontal flow
            flow_y: Vertical flow

        Returns:
            Confidence map (0-1)
        """
        # Compute flow gradients
        grad_x_x = np.abs(np.gradient(flow_x, axis=1))
        grad_x_y = np.abs(np.gradient(flow_x, axis=0))
        grad_y_x = np.abs(np.gradient(flow_y, axis=1))
        grad_y_y = np.abs(np.gradient(flow_y, axis=0))

        # Total gradient magnitude indicates inconsistency
        total_grad = grad_x_x + grad_x_y + grad_y_x + grad_y_y

        # Normalize and invert (low gradient = high confidence)
        max_grad = np.percentile(total_grad, 95) + 1e-6
        confidence = 1.0 - np.clip(total_grad / max_grad, 0, 1)

        # Smooth the confidence map
        confidence = cv2.GaussianBlur(confidence.astype(np.float32), (5, 5), 0)

        return confidence

    def estimate_bidirectional(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[RAFTFlowField, RAFTFlowField]:
        """Estimate bidirectional flow (forward and backward).

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)

        Returns:
            Tuple of (forward_flow, backward_flow)
        """
        forward_flow = self.estimate(frame1, frame2)
        backward_flow = self.estimate(frame2, frame1)
        return forward_flow, backward_flow

    def check_flow_consistency(
        self,
        forward_flow: RAFTFlowField,
        backward_flow: RAFTFlowField,
    ) -> np.ndarray:
        """Check forward-backward flow consistency.

        Computes occlusion mask based on cycle consistency.

        Args:
            forward_flow: Forward flow (1 -> 2)
            backward_flow: Backward flow (2 -> 1)

        Returns:
            Consistency mask (1 = consistent, 0 = occluded/inconsistent)
        """
        h, w = forward_flow.flow_x.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Forward then backward should return to original position
        # Warp backward flow using forward flow
        map_x = (x + forward_flow.flow_x).astype(np.float32)
        map_y = (y + forward_flow.flow_y).astype(np.float32)

        warped_back_x = cv2.remap(
            backward_flow.flow_x, map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        warped_back_y = cv2.remap(
            backward_flow.flow_y, map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Compute cycle error
        cycle_x = forward_flow.flow_x + warped_back_x
        cycle_y = forward_flow.flow_y + warped_back_y
        cycle_error = np.sqrt(cycle_x**2 + cycle_y**2)

        # Threshold for consistency
        forward_mag = forward_flow.magnitude
        threshold = 0.5 * (forward_mag + 1)  # Adaptive threshold

        consistency = (cycle_error < threshold).astype(np.float32)

        return consistency

    def warp_frame(
        self,
        frame: np.ndarray,
        flow: RAFTFlowField,
        inverse: bool = False,
    ) -> np.ndarray:
        """Warp a frame according to optical flow.

        Args:
            frame: Frame to warp (BGR)
            flow: Optical flow field
            inverse: If True, warp in reverse direction

        Returns:
            Warped frame
        """
        h, w = frame.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        if inverse:
            map_x = (x - flow.flow_x).astype(np.float32)
            map_y = (y - flow.flow_y).astype(np.float32)
        else:
            map_x = (x + flow.flow_x).astype(np.float32)
            map_y = (y + flow.flow_y).astype(np.float32)

        warped = cv2.remap(
            frame, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return warped

    def clear_cache(self) -> None:
        """Clear model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_raft_estimator(
    gpu_id: int = 0,
    iterations: int = 12,
    half_precision: bool = True,
) -> RAFTFlowEstimator:
    """Factory function to create a RAFT flow estimator.

    Args:
        gpu_id: GPU device ID
        iterations: Number of RAFT iterations
        half_precision: Use FP16 mode

    Returns:
        Configured RAFTFlowEstimator
    """
    return RAFTFlowEstimator(
        gpu_id=gpu_id,
        iterations=iterations,
        half_precision=half_precision,
    )


def estimate_flow_raft(
    frame1: np.ndarray,
    frame2: np.ndarray,
    gpu_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to estimate optical flow with RAFT.

    Args:
        frame1: Source frame (BGR)
        frame2: Target frame (BGR)
        gpu_id: GPU device ID

    Returns:
        Tuple of (flow_x, flow_y, confidence)
    """
    estimator = RAFTFlowEstimator(gpu_id=gpu_id)

    if not estimator.is_available():
        raise RuntimeError("RAFT is not available")

    flow = estimator.estimate(frame1, frame2)
    return flow.flow_x, flow.flow_y, flow.confidence
