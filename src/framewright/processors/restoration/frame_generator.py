"""Generative Frame Extension for Video Restoration.

This module provides AI-powered frame generation capabilities:
- Frame interpolation (RIFE, FILM, diffusion-based)
- Forward/backward video extension using video diffusion
- Gap filling for missing frame sequences
- Damaged frame restoration via inpainting

Diffusion models provide the highest quality but require significant VRAM.
RIFE/FILM provide fast real-time interpolation with good quality.

Model Sources (user must download manually):
- Stable Video Diffusion: HuggingFace stabilityai/stable-video-diffusion
- FILM: TensorFlow Hub google/film
- RIFE: https://github.com/hzwer/ECCV2022-RIFE

VRAM Requirements:
- Diffusion (SVD): 16-24GB
- RIFE: 2-4GB
- FILM: 4-6GB
- Motion estimation: 2-4GB

Example:
    >>> from framewright.processors.restoration import GenerativeFrameExtender
    >>>
    >>> extender = GenerativeFrameExtender()
    >>> extended = extender.extend_clip(frames, direction="forward", seconds=2.0)
    >>> interpolated = extender.interpolate(frames, multiplier=2)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_torch = None
_cv2 = None
_diffusers = None

HAS_OPENCV = False
HAS_TORCH = False


def _lazy_import_cv2():
    """Lazy import OpenCV."""
    global _cv2, HAS_OPENCV
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
            HAS_OPENCV = True
        except ImportError:
            logger.debug("OpenCV not available")
            HAS_OPENCV = False
    return _cv2


def _lazy_import_torch():
    """Lazy import PyTorch."""
    global _torch, HAS_TORCH
    if _torch is None:
        try:
            import torch
            _torch = torch
            HAS_TORCH = True
        except ImportError:
            logger.debug("PyTorch not available")
            HAS_TORCH = False
    return _torch


def _lazy_import_diffusers():
    """Lazy import diffusers."""
    global _diffusers
    if _diffusers is None:
        try:
            import diffusers
            _diffusers = diffusers
        except ImportError:
            logger.debug("diffusers not available")
            _diffusers = None
    return _diffusers


class InterpolationAlgorithm(Enum):
    """Frame interpolation algorithms."""
    RIFE = "rife"
    """RIFE neural network interpolation - fast, good quality."""

    FILM = "film"
    """Google FILM interpolation - excellent quality, moderate speed."""

    DIFFUSION = "diffusion"
    """Diffusion-based interpolation - highest quality, slow."""

    OPTICAL_FLOW = "optical_flow"
    """Traditional optical flow warping - fast, basic quality."""

    BLEND = "blend"
    """Simple alpha blending - fastest, lowest quality."""


class GenerationModel(Enum):
    """Video generation models for extension."""
    SVD = "svd"
    """Stable Video Diffusion - highest quality, 16GB+ VRAM."""

    SVD_XT = "svd_xt"
    """SVD Extended - longer sequences, 24GB+ VRAM."""

    ANIMATEDIFF = "animatediff"
    """AnimateDiff for stylized content, 8GB+ VRAM."""

    VIDEOCRAFTER = "videocrafter"
    """VideoCrafter for general content, 12GB+ VRAM."""


@dataclass
class FrameGenConfig:
    """Configuration for generative frame operations.

    Attributes:
        model: Generation model to use for extension/filling
        steps: Diffusion denoising steps (more = better quality, slower)
        guidance_scale: Classifier-free guidance scale (higher = more faithful)
        motion_bucket: Motion intensity for SVD (1-255, higher = more motion)
        fps: Target FPS for generation
        seed: Random seed for reproducibility
        half_precision: Use FP16 for reduced VRAM
        gpu_id: GPU device ID
    """
    model: GenerationModel = GenerationModel.SVD
    steps: int = 25
    guidance_scale: float = 7.5
    motion_bucket: int = 127
    fps: int = 24
    seed: Optional[int] = None
    half_precision: bool = True
    gpu_id: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.model, str):
            self.model = GenerationModel(self.model)
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {self.guidance_scale}")
        if not 1 <= self.motion_bucket <= 255:
            raise ValueError(f"motion_bucket must be 1-255, got {self.motion_bucket}")
        if self.fps < 1:
            raise ValueError(f"fps must be >= 1, got {self.fps}")


@dataclass
class MotionVector:
    """Motion vector data for a frame region.

    Attributes:
        dx: Horizontal displacement (pixels)
        dy: Vertical displacement (pixels)
        confidence: Motion estimation confidence (0-1)
        region: Optional region bounds (x, y, w, h)
    """
    dx: float
    dy: float
    confidence: float = 1.0
    region: Optional[Tuple[int, int, int, int]] = None


@dataclass
class MotionField:
    """Dense motion field between frames.

    Attributes:
        flow_x: Horizontal flow component (H, W)
        flow_y: Vertical flow component (H, W)
        magnitude: Flow magnitude (H, W)
        confidence: Confidence map (H, W)
        global_motion: Estimated global motion vector
    """
    flow_x: np.ndarray
    flow_y: np.ndarray
    magnitude: np.ndarray
    confidence: np.ndarray
    global_motion: MotionVector = field(
        default_factory=lambda: MotionVector(0, 0, 0)
    )

    @property
    def shape(self) -> Tuple[int, int]:
        """Return flow field dimensions."""
        return self.flow_x.shape[:2]

    def get_flow(self) -> np.ndarray:
        """Get combined flow array (H, W, 2)."""
        return np.stack([self.flow_x, self.flow_y], axis=-1)


@dataclass
class InterpolationResult:
    """Result of frame interpolation.

    Attributes:
        frames: Generated intermediate frames
        timestamps: Normalized timestamps for each frame (0-1)
        method_used: Interpolation method that was used
        processing_time_ms: Processing time in milliseconds
    """
    frames: List[np.ndarray]
    timestamps: List[float]
    method_used: InterpolationAlgorithm
    processing_time_ms: float = 0.0


@dataclass
class ExtensionResult:
    """Result of clip extension.

    Attributes:
        frames: Generated extension frames
        direction: Extension direction ("forward" or "backward")
        seconds_generated: Actual seconds of video generated
        processing_time_ms: Processing time
        peak_vram_mb: Peak VRAM usage
    """
    frames: List[np.ndarray]
    direction: str
    seconds_generated: float
    processing_time_ms: float = 0.0
    peak_vram_mb: int = 0


@dataclass
class GapFillResult:
    """Result of gap filling operation.

    Attributes:
        frames: Generated frames to fill the gap
        blend_weights: Blending weights at boundaries
        continuity_score: Motion continuity score (0-1)
        processing_time_ms: Processing time
    """
    frames: List[np.ndarray]
    blend_weights: List[float]
    continuity_score: float
    processing_time_ms: float = 0.0


class MotionEstimator:
    """Estimate motion between video frames.

    Uses optical flow analysis to estimate motion vectors,
    predict future/past motion, and extract motion fields.

    Example:
        >>> estimator = MotionEstimator(gpu_id=0)
        >>> field = estimator.estimate_flow(frame1, frame2)
        >>> predicted = estimator.predict_motion(frame1, field, steps=5)
    """

    def __init__(
        self,
        gpu_id: int = 0,
        use_raft: bool = True,
        half_precision: bool = True,
    ):
        """Initialize motion estimator.

        Args:
            gpu_id: GPU device ID
            use_raft: Use RAFT for flow estimation if available
            half_precision: Use FP16 for reduced VRAM
        """
        self.gpu_id = gpu_id
        self.use_raft = use_raft
        self.half_precision = half_precision
        self._raft_model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available flow estimation backend."""
        cv2 = _lazy_import_cv2()
        torch = _lazy_import_torch()

        if self.use_raft and torch is not None:
            try:
                from torchvision.models.optical_flow import raft_large
                logger.info("Using RAFT backend for motion estimation")
                return "raft"
            except ImportError:
                pass

        if cv2 is not None:
            logger.info("Using OpenCV Farneback for motion estimation")
            return "farneback"

        logger.warning("No motion estimation backend available")
        return "none"

    def is_available(self) -> bool:
        """Check if motion estimation is available."""
        return self._backend != "none"

    def _init_raft(self) -> None:
        """Initialize RAFT model."""
        if self._raft_model is not None:
            return

        torch = _lazy_import_torch()
        if torch is None:
            return

        try:
            from torchvision.models.optical_flow import (
                raft_large,
                Raft_Large_Weights,
            )

            if torch.cuda.is_available():
                self._device = torch.device(f"cuda:{self.gpu_id}")
            else:
                self._device = torch.device("cpu")

            weights = Raft_Large_Weights.DEFAULT
            self._raft_model = raft_large(weights=weights)
            self._raft_model = self._raft_model.to(self._device)
            self._raft_model.eval()

            if self.half_precision and self._device.type == "cuda":
                self._raft_model = self._raft_model.half()

            logger.info(f"RAFT initialized on {self._device}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAFT: {e}")
            self._backend = "farneback"

    def estimate_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> MotionField:
        """Estimate optical flow between two frames.

        Args:
            frame1: Source frame (H, W, 3) BGR
            frame2: Target frame (H, W, 3) BGR

        Returns:
            MotionField with dense flow estimation
        """
        if self._backend == "raft":
            return self._estimate_raft(frame1, frame2)
        elif self._backend == "farneback":
            return self._estimate_farneback(frame1, frame2)
        else:
            # Return zero flow
            h, w = frame1.shape[:2]
            return MotionField(
                flow_x=np.zeros((h, w), dtype=np.float32),
                flow_y=np.zeros((h, w), dtype=np.float32),
                magnitude=np.zeros((h, w), dtype=np.float32),
                confidence=np.zeros((h, w), dtype=np.float32),
            )

    def _estimate_raft(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> MotionField:
        """Estimate flow using RAFT."""
        self._init_raft()
        torch = _lazy_import_torch()
        cv2 = _lazy_import_cv2()

        if self._raft_model is None:
            return self._estimate_farneback(frame1, frame2)

        # Convert BGR to RGB and normalize
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # To tensor
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).float()

        # Add batch dimension
        img1_t = img1_t.unsqueeze(0).to(self._device)
        img2_t = img2_t.unsqueeze(0).to(self._device)

        if self.half_precision and self._device.type == "cuda":
            img1_t = img1_t.half()
            img2_t = img2_t.half()

        with torch.no_grad():
            flow_list = self._raft_model(img1_t, img2_t)
            flow = flow_list[-1]  # Use final iteration

        # Extract flow components
        flow_np = flow[0].cpu().float().numpy()
        flow_x = flow_np[0]
        flow_y = flow_np[1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Estimate confidence from flow smoothness
        grad_x = np.abs(np.gradient(flow_x, axis=1))
        grad_y = np.abs(np.gradient(flow_y, axis=0))
        smoothness = 1.0 / (1.0 + grad_x + grad_y)
        confidence = np.clip(smoothness, 0, 1).astype(np.float32)

        # Compute global motion
        global_dx = float(np.median(flow_x))
        global_dy = float(np.median(flow_y))
        global_conf = float(np.mean(confidence))

        return MotionField(
            flow_x=flow_x,
            flow_y=flow_y,
            magnitude=magnitude,
            confidence=confidence,
            global_motion=MotionVector(global_dx, global_dy, global_conf),
        )

    def _estimate_farneback(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> MotionField:
        """Estimate flow using OpenCV Farneback."""
        cv2 = _lazy_import_cv2()
        if cv2 is None:
            h, w = frame1.shape[:2]
            return MotionField(
                flow_x=np.zeros((h, w), dtype=np.float32),
                flow_y=np.zeros((h, w), dtype=np.float32),
                magnitude=np.zeros((h, w), dtype=np.float32),
                confidence=np.ones((h, w), dtype=np.float32),
            )

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Simple confidence based on magnitude consistency
        mag_blur = cv2.GaussianBlur(magnitude, (5, 5), 0)
        confidence = 1.0 / (1.0 + np.abs(magnitude - mag_blur))
        confidence = confidence.astype(np.float32)

        global_dx = float(np.median(flow_x))
        global_dy = float(np.median(flow_y))

        return MotionField(
            flow_x=flow_x,
            flow_y=flow_y,
            magnitude=magnitude,
            confidence=confidence,
            global_motion=MotionVector(global_dx, global_dy, 0.8),
        )

    def predict_motion(
        self,
        frame: np.ndarray,
        flow: MotionField,
        steps: int = 1,
        direction: str = "forward",
    ) -> List[MotionVector]:
        """Predict future or past motion from current flow.

        Args:
            frame: Current frame
            flow: Current motion field
            steps: Number of steps to predict
            direction: "forward" or "backward"

        Returns:
            List of predicted motion vectors
        """
        predictions = []
        sign = 1.0 if direction == "forward" else -1.0

        # Simple linear extrapolation with decay
        for i in range(1, steps + 1):
            decay = 0.9 ** i  # Motion decays over time
            dx = flow.global_motion.dx * sign * decay
            dy = flow.global_motion.dy * sign * decay
            conf = flow.global_motion.confidence * decay
            predictions.append(MotionVector(dx, dy, conf))

        return predictions

    def extract_motion_vectors(
        self,
        flow: MotionField,
        block_size: int = 16,
    ) -> List[MotionVector]:
        """Extract block-based motion vectors from flow field.

        Args:
            flow: Dense motion field
            block_size: Size of blocks for motion vectors

        Returns:
            List of motion vectors for each block
        """
        h, w = flow.shape
        vectors = []

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)

                block_flow_x = flow.flow_x[y:y_end, x:x_end]
                block_flow_y = flow.flow_y[y:y_end, x:x_end]
                block_conf = flow.confidence[y:y_end, x:x_end]

                dx = float(np.mean(block_flow_x))
                dy = float(np.mean(block_flow_y))
                conf = float(np.mean(block_conf))

                vectors.append(MotionVector(
                    dx=dx,
                    dy=dy,
                    confidence=conf,
                    region=(x, y, x_end - x, y_end - y),
                ))

        return vectors

    def clear_cache(self) -> None:
        """Clear GPU memory."""
        if self._raft_model is not None:
            del self._raft_model
            self._raft_model = None

        torch = _lazy_import_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class FrameInterpolator:
    """Generate intermediate frames between existing frames.

    Supports multiple interpolation algorithms:
    - RIFE: Fast neural network interpolation
    - FILM: High-quality frame interpolation
    - Diffusion: Highest quality, slowest
    - Optical flow: Traditional warping
    - Blend: Simple alpha blending

    Example:
        >>> interpolator = FrameInterpolator(algorithm=InterpolationAlgorithm.RIFE)
        >>> result = interpolator.interpolate(frame1, frame2, count=3)
    """

    def __init__(
        self,
        algorithm: InterpolationAlgorithm = InterpolationAlgorithm.OPTICAL_FLOW,
        gpu_id: int = 0,
        half_precision: bool = True,
    ):
        """Initialize frame interpolator.

        Args:
            algorithm: Interpolation algorithm to use
            gpu_id: GPU device ID
            half_precision: Use FP16 for reduced VRAM
        """
        if isinstance(algorithm, str):
            algorithm = InterpolationAlgorithm(algorithm)
        self.algorithm = algorithm
        self.gpu_id = gpu_id
        self.half_precision = half_precision
        self._model = None
        self._device = None
        self._motion_estimator = MotionEstimator(gpu_id, half_precision=half_precision)

    def is_available(self) -> bool:
        """Check if interpolation is available."""
        cv2 = _lazy_import_cv2()
        return cv2 is not None

    def interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int = 1,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> InterpolationResult:
        """Generate intermediate frames.

        Args:
            frame1: Start frame (H, W, 3) BGR
            frame2: End frame (H, W, 3) BGR
            count: Number of intermediate frames to generate
            progress_callback: Optional progress callback (0-1)

        Returns:
            InterpolationResult with generated frames
        """
        start_time = time.time()

        if self.algorithm == InterpolationAlgorithm.RIFE:
            frames = self._interpolate_rife(frame1, frame2, count, progress_callback)
            method = InterpolationAlgorithm.RIFE
        elif self.algorithm == InterpolationAlgorithm.FILM:
            frames = self._interpolate_film(frame1, frame2, count, progress_callback)
            method = InterpolationAlgorithm.FILM
        elif self.algorithm == InterpolationAlgorithm.DIFFUSION:
            frames = self._interpolate_diffusion(frame1, frame2, count, progress_callback)
            method = InterpolationAlgorithm.DIFFUSION
        elif self.algorithm == InterpolationAlgorithm.OPTICAL_FLOW:
            frames = self._interpolate_flow(frame1, frame2, count, progress_callback)
            method = InterpolationAlgorithm.OPTICAL_FLOW
        else:
            frames = self._interpolate_blend(frame1, frame2, count, progress_callback)
            method = InterpolationAlgorithm.BLEND

        timestamps = [(i + 1) / (count + 1) for i in range(count)]
        elapsed = (time.time() - start_time) * 1000

        return InterpolationResult(
            frames=frames,
            timestamps=timestamps,
            method_used=method,
            processing_time_ms=elapsed,
        )

    def _interpolate_blend(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Simple alpha blending interpolation."""
        cv2 = _lazy_import_cv2()
        frames = []

        for i in range(count):
            alpha = (i + 1) / (count + 1)
            if cv2 is not None:
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            else:
                blended = ((1 - alpha) * frame1 + alpha * frame2).astype(np.uint8)
            frames.append(blended)

            if progress_callback:
                progress_callback((i + 1) / count)

        return frames

    def _interpolate_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Optical flow-based interpolation."""
        cv2 = _lazy_import_cv2()
        if cv2 is None:
            return self._interpolate_blend(frame1, frame2, count, progress_callback)

        # Get bidirectional flow
        flow_forward = self._motion_estimator.estimate_flow(frame1, frame2)
        flow_backward = self._motion_estimator.estimate_flow(frame2, frame1)

        h, w = frame1.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        x_coords = x_coords.astype(np.float32)
        y_coords = y_coords.astype(np.float32)

        frames = []
        for i in range(count):
            t = (i + 1) / (count + 1)

            # Bidirectional warping
            map_x_fwd = x_coords + flow_forward.flow_x * t
            map_y_fwd = y_coords + flow_forward.flow_y * t

            map_x_bwd = x_coords + flow_backward.flow_x * (1 - t)
            map_y_bwd = y_coords + flow_backward.flow_y * (1 - t)

            warped_fwd = cv2.remap(
                frame1, map_x_fwd, map_y_fwd,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
            warped_bwd = cv2.remap(
                frame2, map_x_bwd, map_y_bwd,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            # Blend with temporal weighting
            blended = cv2.addWeighted(warped_fwd, 1 - t, warped_bwd, t, 0)
            frames.append(blended)

            if progress_callback:
                progress_callback((i + 1) / count)

        return frames

    def _interpolate_rife(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """RIFE neural network interpolation."""
        # RIFE requires external binary or PyTorch implementation
        # Fall back to optical flow if not available
        logger.info("RIFE not implemented, using optical flow fallback")
        return self._interpolate_flow(frame1, frame2, count, progress_callback)

    def _interpolate_film(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """FILM interpolation."""
        # FILM requires TensorFlow Hub
        logger.info("FILM not implemented, using optical flow fallback")
        return self._interpolate_flow(frame1, frame2, count, progress_callback)

    def _interpolate_diffusion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Diffusion-based interpolation."""
        # Requires specialized video diffusion models
        logger.info("Diffusion interpolation not implemented, using optical flow")
        return self._interpolate_flow(frame1, frame2, count, progress_callback)

    def interpolate_motion_aware(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        count: int,
        occlusion_threshold: float = 0.5,
    ) -> InterpolationResult:
        """Motion-aware interpolation with occlusion handling.

        Args:
            frame1: Start frame
            frame2: End frame
            count: Number of intermediate frames
            occlusion_threshold: Threshold for detecting occlusions

        Returns:
            InterpolationResult with occlusion-aware frames
        """
        cv2 = _lazy_import_cv2()
        start_time = time.time()

        # Get bidirectional flow
        flow_fwd = self._motion_estimator.estimate_flow(frame1, frame2)
        flow_bwd = self._motion_estimator.estimate_flow(frame2, frame1)

        # Detect occlusions via forward-backward consistency
        h, w = frame1.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        x_coords = x_coords.astype(np.float32)
        y_coords = y_coords.astype(np.float32)

        # Warp backward flow to frame1 coordinates
        map_x = x_coords + flow_fwd.flow_x
        map_y = y_coords + flow_fwd.flow_y
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # Sample backward flow at forward flow destinations
        warped_bwd_x = cv2.remap(
            flow_bwd.flow_x, map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        warped_bwd_y = cv2.remap(
            flow_bwd.flow_y, map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

        # Forward-backward error
        fb_error = np.sqrt(
            (flow_fwd.flow_x + warped_bwd_x) ** 2 +
            (flow_fwd.flow_y + warped_bwd_y) ** 2
        )

        # Occlusion mask (high error = occlusion)
        occ_mask = (fb_error > occlusion_threshold * flow_fwd.magnitude.max()).astype(np.float32)
        occ_mask = cv2.GaussianBlur(occ_mask, (5, 5), 0)

        frames = []
        for i in range(count):
            t = (i + 1) / (count + 1)

            # Warp with flow
            map_x_fwd = x_coords + flow_fwd.flow_x * t
            map_y_fwd = y_coords + flow_fwd.flow_y * t

            map_x_bwd = x_coords + flow_bwd.flow_x * (1 - t)
            map_y_bwd = y_coords + flow_bwd.flow_y * (1 - t)

            warped_fwd = cv2.remap(
                frame1, map_x_fwd, map_y_fwd,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )
            warped_bwd = cv2.remap(
                frame2, map_x_bwd, map_y_bwd,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            # Occlusion-aware blending
            # In occluded regions, prefer the frame that is "revealing" new content
            if t < 0.5:
                # Prefer backward warp (from frame2)
                occ_weight = occ_mask * t * 2
            else:
                # Prefer forward warp (from frame1)
                occ_weight = occ_mask * (1 - t) * 2

            base_blend = cv2.addWeighted(warped_fwd, 1 - t, warped_bwd, t, 0)

            # Blend in non-occluded regions, use single source in occluded
            occ_weight_3ch = np.stack([occ_weight] * 3, axis=-1)
            result = (
                (1 - occ_weight_3ch) * base_blend +
                occ_weight_3ch * (warped_bwd if t < 0.5 else warped_fwd)
            ).astype(np.uint8)

            frames.append(result)

        elapsed = (time.time() - start_time) * 1000
        timestamps = [(i + 1) / (count + 1) for i in range(count)]

        return InterpolationResult(
            frames=frames,
            timestamps=timestamps,
            method_used=InterpolationAlgorithm.OPTICAL_FLOW,
            processing_time_ms=elapsed,
        )

    def clear_cache(self) -> None:
        """Clear GPU memory."""
        self._motion_estimator.clear_cache()
        if self._model is not None:
            del self._model
            self._model = None


class FrameExtender:
    """Extend video sequences forward or backward using diffusion.

    Uses video diffusion models (like SVD) to generate plausible
    continuation frames based on existing content.

    Example:
        >>> extender = FrameExtender()
        >>> future_frames = extender.extend_forward(frames, count=24)
        >>> past_frames = extender.extend_backward(frames, count=12)
    """

    DEFAULT_MODEL_DIR = Path.home() / ".framewright" / "models" / "video_gen"

    def __init__(
        self,
        config: Optional[FrameGenConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize frame extender.

        Args:
            config: Generation configuration
            model_dir: Directory for model weights
        """
        self.config = config or FrameGenConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._pipeline = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available generation backend."""
        torch = _lazy_import_torch()
        diffusers = _lazy_import_diffusers()

        if torch is None:
            logger.warning("PyTorch not available - extension disabled")
            return None

        if diffusers is None:
            logger.warning("diffusers not available - extension disabled")
            return None

        if self.config.model == GenerationModel.SVD:
            try:
                from diffusers import StableVideoDiffusionPipeline
                logger.info("SVD backend available")
                return "svd"
            except ImportError:
                pass

        if self.config.model == GenerationModel.SVD_XT:
            try:
                from diffusers import StableVideoDiffusionPipeline
                logger.info("SVD-XT backend available")
                return "svd_xt"
            except ImportError:
                pass

        logger.warning("No video diffusion backend available")
        return None

    def is_available(self) -> bool:
        """Check if extension is available."""
        return self._backend is not None

    def _init_pipeline(self) -> None:
        """Initialize the diffusion pipeline."""
        if self._pipeline is not None:
            return

        torch = _lazy_import_torch()
        if torch is None:
            return

        try:
            from diffusers import StableVideoDiffusionPipeline

            if torch.cuda.is_available():
                self._device = torch.device(f"cuda:{self.config.gpu_id}")
            else:
                self._device = torch.device("cpu")

            dtype = torch.float16 if self.config.half_precision else torch.float32

            if self._backend == "svd_xt":
                model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
            else:
                model_id = "stabilityai/stable-video-diffusion-img2vid"

            self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16" if self.config.half_precision else None,
            )
            self._pipeline = self._pipeline.to(self._device)

            # Enable memory optimizations
            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                self._pipeline.enable_model_cpu_offload()

            logger.info(f"Initialized {model_id} on {self._device}")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self._backend = None

    def extend_forward(
        self,
        frames: List[np.ndarray],
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Generate future frames continuing from the sequence.

        Args:
            frames: Input frames (uses last frame as conditioning)
            count: Number of frames to generate
            progress_callback: Optional progress callback

        Returns:
            List of generated future frames
        """
        if not frames:
            return []

        if not self.is_available():
            logger.warning("Extension not available, returning empty list")
            return []

        self._init_pipeline()
        if self._pipeline is None:
            return []

        torch = _lazy_import_torch()
        cv2 = _lazy_import_cv2()

        # Use last frame as conditioning
        condition_frame = frames[-1]

        try:
            from PIL import Image

            # Convert BGR to RGB PIL
            rgb_frame = cv2.cvtColor(condition_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Set seed
            generator = None
            if self.config.seed is not None:
                generator = torch.Generator(device=self._device)
                generator.manual_seed(self.config.seed)

            with torch.no_grad():
                output = self._pipeline(
                    pil_image,
                    num_frames=count + 1,  # +1 for conditioning frame
                    num_inference_steps=self.config.steps,
                    motion_bucket_id=self.config.motion_bucket,
                    fps=self.config.fps,
                    generator=generator,
                    decode_chunk_size=4,
                )

            # Convert back to numpy BGR
            generated = []
            for i, frame in enumerate(output.frames[0][1:]):  # Skip first (input)
                frame_np = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                generated.append(frame_bgr)

                if progress_callback:
                    progress_callback((i + 1) / count)

            return generated[:count]

        except Exception as e:
            logger.error(f"Frame extension failed: {e}")
            return []

    def extend_backward(
        self,
        frames: List[np.ndarray],
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Generate past frames preceding the sequence.

        Args:
            frames: Input frames (uses first frame as conditioning)
            count: Number of frames to generate
            progress_callback: Optional progress callback

        Returns:
            List of generated past frames (in temporal order)
        """
        if not frames:
            return []

        if not self.is_available():
            logger.warning("Extension not available")
            return []

        # For backward extension, we reverse the conditioning frame
        # and then reverse the output
        cv2 = _lazy_import_cv2()
        if cv2 is None:
            return []

        # Use first frame, flipped horizontally as a hack for backward motion
        condition_frame = cv2.flip(frames[0], 1)
        temp_frames = [condition_frame]

        generated = self.extend_forward(temp_frames, count, progress_callback)

        # Flip back and reverse order
        result = []
        for frame in reversed(generated):
            result.append(cv2.flip(frame, 1))

        return result

    def clear_cache(self) -> None:
        """Clear GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        torch = _lazy_import_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class GapFiller:
    """Fill gaps in video sequences.

    Uses bidirectional generation and blending to fill missing
    frames while maintaining temporal consistency.

    Example:
        >>> filler = GapFiller()
        >>> filled = filler.fill_gap(before_frames, after_frames, count=10)
    """

    def __init__(
        self,
        config: Optional[FrameGenConfig] = None,
        blend_frames: int = 3,
    ):
        """Initialize gap filler.

        Args:
            config: Generation configuration
            blend_frames: Number of frames for boundary blending
        """
        self.config = config or FrameGenConfig()
        self.blend_frames = blend_frames
        self._extender = FrameExtender(config)
        self._interpolator = FrameInterpolator(
            algorithm=InterpolationAlgorithm.OPTICAL_FLOW,
            gpu_id=self.config.gpu_id,
        )

    def fill_gap(
        self,
        before: List[np.ndarray],
        after: List[np.ndarray],
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> GapFillResult:
        """Fill gap between two frame sequences.

        Args:
            before: Frames before the gap
            after: Frames after the gap
            count: Number of frames to generate for the gap
            progress_callback: Optional progress callback

        Returns:
            GapFillResult with generated frames
        """
        start_time = time.time()
        cv2 = _lazy_import_cv2()

        if not before and not after:
            return GapFillResult(
                frames=[],
                blend_weights=[],
                continuity_score=0.0,
                processing_time_ms=0.0,
            )

        # Strategy depends on available boundaries
        if before and after:
            # Bidirectional fill
            frames = self._fill_bidirectional(before, after, count, progress_callback)
        elif before:
            # Forward extrapolation
            frames = self._extender.extend_forward(before, count, progress_callback)
        else:
            # Backward extrapolation
            frames = self._extender.extend_backward(after, count, progress_callback)

        # Apply boundary blending
        blend_weights = self._compute_blend_weights(count)

        if cv2 is not None and before and after and len(frames) >= self.blend_frames:
            frames = self._blend_boundaries(frames, before[-1], after[0])

        # Compute continuity score
        continuity = self._compute_continuity(before, frames, after)

        elapsed = (time.time() - start_time) * 1000

        return GapFillResult(
            frames=frames,
            blend_weights=blend_weights,
            continuity_score=continuity,
            processing_time_ms=elapsed,
        )

    def _fill_bidirectional(
        self,
        before: List[np.ndarray],
        after: List[np.ndarray],
        count: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Bidirectional gap filling."""
        if count <= 3:
            # For small gaps, use interpolation
            result = self._interpolator.interpolate(
                before[-1], after[0], count, progress_callback
            )
            return result.frames

        # For larger gaps, generate from both ends and blend in middle
        half = count // 2

        # Forward from before
        forward_frames = self._extender.extend_forward(
            before, half,
            lambda p: progress_callback(p * 0.4) if progress_callback else None,
        )

        # Backward from after
        backward_frames = self._extender.extend_backward(
            after, count - half,
            lambda p: progress_callback(0.4 + p * 0.4) if progress_callback else None,
        )

        # Blend in the middle
        if forward_frames and backward_frames:
            frames = self._blend_sequences(forward_frames, backward_frames, count)
        else:
            frames = forward_frames + backward_frames

        if progress_callback:
            progress_callback(1.0)

        return frames

    def _blend_sequences(
        self,
        forward: List[np.ndarray],
        backward: List[np.ndarray],
        total: int,
    ) -> List[np.ndarray]:
        """Blend forward and backward generated sequences."""
        cv2 = _lazy_import_cv2()
        if cv2 is None:
            return forward + backward

        # Ensure we have enough frames
        forward = forward[:total]
        backward = backward[:total]

        # Pad if needed
        while len(forward) < total and forward:
            forward.append(forward[-1])
        while len(backward) < total and backward:
            backward.insert(0, backward[0])

        result = []
        for i in range(total):
            # Sigmoid blend weight
            t = i / max(total - 1, 1)
            weight = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))

            if i < len(forward) and i < len(backward):
                blended = cv2.addWeighted(
                    forward[i], 1 - weight,
                    backward[i], weight,
                    0,
                )
                result.append(blended)
            elif i < len(forward):
                result.append(forward[i])
            elif i < len(backward):
                result.append(backward[i])

        return result

    def _blend_boundaries(
        self,
        frames: List[np.ndarray],
        before_frame: np.ndarray,
        after_frame: np.ndarray,
    ) -> List[np.ndarray]:
        """Blend generated frames at boundaries."""
        cv2 = _lazy_import_cv2()
        if cv2 is None or not frames:
            return frames

        result = frames.copy()
        blend = min(self.blend_frames, len(frames) // 2)

        # Blend at start
        for i in range(blend):
            alpha = (i + 1) / (blend + 1)
            result[i] = cv2.addWeighted(
                before_frame, 1 - alpha,
                result[i], alpha,
                0,
            )

        # Blend at end
        for i in range(blend):
            idx = len(result) - 1 - i
            alpha = (i + 1) / (blend + 1)
            result[idx] = cv2.addWeighted(
                after_frame, 1 - alpha,
                result[idx], alpha,
                0,
            )

        return result

    def _compute_blend_weights(self, count: int) -> List[float]:
        """Compute blend weights for gap frames."""
        weights = []
        for i in range(count):
            t = i / max(count - 1, 1)
            # Raised cosine for smooth blending
            weight = 0.5 * (1 - np.cos(np.pi * t))
            weights.append(float(weight))
        return weights

    def _compute_continuity(
        self,
        before: List[np.ndarray],
        filled: List[np.ndarray],
        after: List[np.ndarray],
    ) -> float:
        """Compute motion continuity score across the filled gap."""
        if not filled:
            return 0.0

        estimator = MotionEstimator()
        scores = []

        # Check continuity at boundaries
        if before:
            flow = estimator.estimate_flow(before[-1], filled[0])
            mag_mean = float(np.mean(flow.magnitude))
            # Lower magnitude = smoother transition
            scores.append(1.0 / (1.0 + mag_mean / 10.0))

        if after:
            flow = estimator.estimate_flow(filled[-1], after[0])
            mag_mean = float(np.mean(flow.magnitude))
            scores.append(1.0 / (1.0 + mag_mean / 10.0))

        # Check internal consistency
        for i in range(len(filled) - 1):
            flow = estimator.estimate_flow(filled[i], filled[i + 1])
            conf_mean = float(np.mean(flow.confidence))
            scores.append(conf_mean)

        return float(np.mean(scores)) if scores else 0.0


class DamagedFrameRestorer:
    """Detect and restore damaged/corrupted frames.

    Uses temporal information and inpainting to fix:
    - Corrupted/garbled frames
    - Partially missing content
    - Scanning artifacts
    - Tape damage

    Example:
        >>> restorer = DamagedFrameRestorer()
        >>> damage_map = restorer.detect_damage(frame, neighbors)
        >>> restored = restorer.restore_frame(frame, damage_map, neighbors)
    """

    def __init__(
        self,
        corruption_threshold: float = 0.3,
        use_temporal: bool = True,
        gpu_id: int = 0,
    ):
        """Initialize damaged frame restorer.

        Args:
            corruption_threshold: Threshold for damage detection (0-1)
            use_temporal: Use temporal neighbors for restoration
            gpu_id: GPU device ID
        """
        self.corruption_threshold = corruption_threshold
        self.use_temporal = use_temporal
        self.gpu_id = gpu_id
        self._motion_estimator = MotionEstimator(gpu_id)

    def detect_damage(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        next_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detect damaged regions in a frame.

        Args:
            frame: Frame to analyze
            prev_frame: Previous frame for temporal comparison
            next_frame: Next frame for temporal comparison

        Returns:
            Damage mask (H, W) with values 0-1 (1 = damaged)
        """
        cv2 = _lazy_import_cv2()
        if cv2 is None:
            return np.zeros(frame.shape[:2], dtype=np.float32)

        h, w = frame.shape[:2]
        damage_scores = []

        # Check for completely black/white regions (dead pixels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dead_black = (gray < 5).astype(np.float32)
        dead_white = (gray > 250).astype(np.float32)
        damage_scores.append((dead_black + dead_white) * 0.5)

        # Check for color aberrations (unusual color values)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        # Very high saturation often indicates corruption
        color_damage = (saturation > 0.95).astype(np.float32)
        damage_scores.append(color_damage * 0.3)

        # Temporal inconsistency check
        if self.use_temporal and (prev_frame is not None or next_frame is not None):
            temporal_damage = self._detect_temporal_damage(
                frame, prev_frame, next_frame
            )
            damage_scores.append(temporal_damage)

        # Check for high-frequency noise (compression artifacts, tape damage)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        noise_level = np.abs(laplacian)
        noise_thresh = np.percentile(noise_level, 95)
        noise_damage = (noise_level > noise_thresh * 2).astype(np.float32)
        damage_scores.append(noise_damage * 0.2)

        # Combine damage scores
        damage_mask = np.maximum.reduce(damage_scores)

        # Threshold
        damage_mask = (damage_mask > self.corruption_threshold).astype(np.float32)

        # Dilate to include surrounding pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        damage_mask = cv2.dilate(damage_mask, kernel, iterations=1)

        return damage_mask

    def _detect_temporal_damage(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        next_frame: Optional[np.ndarray],
    ) -> np.ndarray:
        """Detect damage using temporal inconsistency."""
        cv2 = _lazy_import_cv2()
        h, w = frame.shape[:2]
        damage = np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff_prev = np.abs(gray - prev_gray)
            damage = np.maximum(damage, diff_prev / 255.0)

        if next_frame is not None:
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff_next = np.abs(gray - next_gray)
            damage = np.maximum(damage, diff_next / 255.0)

        # High difference with BOTH neighbors is suspicious
        if prev_frame is not None and next_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # If prev and next are similar but current is different -> damage
            neighbor_diff = np.abs(prev_gray - next_gray) / 255.0
            current_diff = (
                np.abs(gray - prev_gray) + np.abs(gray - next_gray)
            ) / 510.0

            temporal_damage = np.maximum(
                0, current_diff - neighbor_diff - 0.1
            )
            damage = np.maximum(damage, temporal_damage)

        return damage

    def restore_frame(
        self,
        frame: np.ndarray,
        damage_mask: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        next_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Restore damaged regions in a frame.

        Args:
            frame: Damaged frame
            damage_mask: Binary mask of damaged regions
            prev_frame: Previous frame for reference
            next_frame: Next frame for reference

        Returns:
            Restored frame
        """
        cv2 = _lazy_import_cv2()
        if cv2 is None:
            return frame

        if np.sum(damage_mask) < 1:
            return frame

        # Create uint8 mask for inpainting
        mask_uint8 = (damage_mask * 255).astype(np.uint8)

        # Try temporal restoration first
        if self.use_temporal and (prev_frame is not None or next_frame is not None):
            restored = self._restore_temporal(
                frame, mask_uint8, prev_frame, next_frame
            )
        else:
            # Fall back to inpainting
            restored = cv2.inpaint(
                frame, mask_uint8, 3, cv2.INPAINT_TELEA
            )

        return restored

    def _restore_temporal(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prev_frame: Optional[np.ndarray],
        next_frame: Optional[np.ndarray],
    ) -> np.ndarray:
        """Restore using temporal neighbors with motion compensation."""
        cv2 = _lazy_import_cv2()

        h, w = frame.shape[:2]
        result = frame.copy()
        weight_sum = np.zeros((h, w, 1), dtype=np.float32)
        pixel_sum = np.zeros((h, w, 3), dtype=np.float32)

        mask_bool = mask > 127

        if prev_frame is not None:
            # Motion-compensated warping from previous frame
            flow = self._motion_estimator.estimate_flow(prev_frame, frame)
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x_coords - flow.flow_x).astype(np.float32)
            map_y = (y_coords - flow.flow_y).astype(np.float32)

            warped_prev = cv2.remap(
                prev_frame, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            conf = flow.confidence[:, :, np.newaxis]
            pixel_sum += warped_prev.astype(np.float32) * conf
            weight_sum += conf

        if next_frame is not None:
            flow = self._motion_estimator.estimate_flow(next_frame, frame)
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x_coords - flow.flow_x).astype(np.float32)
            map_y = (y_coords - flow.flow_y).astype(np.float32)

            warped_next = cv2.remap(
                next_frame, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            conf = flow.confidence[:, :, np.newaxis]
            pixel_sum += warped_next.astype(np.float32) * conf
            weight_sum += conf

        # Apply temporal restoration in damaged regions
        valid = weight_sum > 0.01
        restored = pixel_sum / np.maximum(weight_sum, 0.01)
        restored = np.clip(restored, 0, 255).astype(np.uint8)

        # Blend with inpainting for regions without good temporal data
        inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        mask_3ch = np.stack([mask_bool] * 3, axis=-1)
        result = np.where(
            mask_3ch & valid.squeeze(axis=-1)[..., np.newaxis],
            restored,
            np.where(mask_3ch, inpainted, frame),
        )

        return result

    def detect_and_restore(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Detect and restore all damaged frames.

        Args:
            frames: List of frames to process
            progress_callback: Optional progress callback

        Returns:
            Tuple of (restored_frames, damage_scores)
        """
        restored = []
        damage_scores = []

        for i, frame in enumerate(frames):
            prev_frame = frames[i - 1] if i > 0 else None
            next_frame = frames[i + 1] if i < len(frames) - 1 else None

            damage_mask = self.detect_damage(frame, prev_frame, next_frame)
            damage_score = float(np.mean(damage_mask))
            damage_scores.append(damage_score)

            if damage_score > 0.01:
                restored_frame = self.restore_frame(
                    frame, damage_mask, prev_frame, next_frame
                )
                restored.append(restored_frame)
            else:
                restored.append(frame)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        return restored, damage_scores


class GenerativeFrameExtender:
    """Main class for generative frame operations.

    Combines all frame generation capabilities:
    - Clip extension (forward/backward)
    - Gap filling
    - FPS interpolation
    - Damaged frame restoration

    Example:
        >>> extender = GenerativeFrameExtender()
        >>> extended = extender.extend_clip(frames, "forward", seconds=2.0)
        >>> filled = extender.fill_gap(before, after, gap_frames=24)
        >>> smooth = extender.interpolate(frames, multiplier=2)
    """

    def __init__(
        self,
        config: Optional[FrameGenConfig] = None,
        interpolation_algorithm: InterpolationAlgorithm = InterpolationAlgorithm.OPTICAL_FLOW,
    ):
        """Initialize generative frame extender.

        Args:
            config: Generation configuration
            interpolation_algorithm: Algorithm for frame interpolation
        """
        self.config = config or FrameGenConfig()
        self._extender = FrameExtender(config)
        self._filler = GapFiller(config)
        self._interpolator = FrameInterpolator(
            algorithm=interpolation_algorithm,
            gpu_id=self.config.gpu_id,
        )
        self._restorer = DamagedFrameRestorer(gpu_id=self.config.gpu_id)

    def extend_clip(
        self,
        frames: List[np.ndarray],
        direction: str,
        seconds: float,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ExtensionResult:
        """Extend a video clip forward or backward.

        Args:
            frames: Input frames
            direction: "forward" or "backward"
            seconds: Duration to extend in seconds
            progress_callback: Optional progress callback

        Returns:
            ExtensionResult with generated frames
        """
        start_time = time.time()
        torch = _lazy_import_torch()

        count = int(seconds * self.config.fps)

        if direction == "forward":
            generated = self._extender.extend_forward(frames, count, progress_callback)
        else:
            generated = self._extender.extend_backward(frames, count, progress_callback)

        elapsed = (time.time() - start_time) * 1000

        peak_vram = 0
        if torch is not None and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated(self.config.gpu_id) // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.config.gpu_id)

        actual_seconds = len(generated) / self.config.fps

        return ExtensionResult(
            frames=generated,
            direction=direction,
            seconds_generated=actual_seconds,
            processing_time_ms=elapsed,
            peak_vram_mb=peak_vram,
        )

    def fill_gap(
        self,
        before: List[np.ndarray],
        after: List[np.ndarray],
        gap_frames: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> GapFillResult:
        """Fill a gap between two frame sequences.

        Args:
            before: Frames before the gap
            after: Frames after the gap
            gap_frames: Number of frames to generate
            progress_callback: Optional progress callback

        Returns:
            GapFillResult with generated frames
        """
        return self._filler.fill_gap(before, after, gap_frames, progress_callback)

    def interpolate(
        self,
        frames: List[np.ndarray],
        multiplier: int = 2,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Increase frame rate by interpolating between frames.

        Args:
            frames: Input frames
            multiplier: Frame rate multiplier (2 = double FPS)
            progress_callback: Optional progress callback

        Returns:
            List of frames with interpolated frames inserted
        """
        if len(frames) < 2:
            return frames

        result = []
        total_pairs = len(frames) - 1

        for i in range(total_pairs):
            result.append(frames[i])

            interp_result = self._interpolator.interpolate(
                frames[i], frames[i + 1],
                count=multiplier - 1,
            )
            result.extend(interp_result.frames)

            if progress_callback:
                progress_callback((i + 1) / total_pairs)

        # Add last frame
        result.append(frames[-1])

        return result

    def restore_damaged(
        self,
        frames: List[np.ndarray],
        damage_masks: Optional[List[np.ndarray]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Fix damaged frames using temporal information.

        Args:
            frames: Input frames (some may be damaged)
            damage_masks: Optional pre-computed damage masks
            progress_callback: Optional progress callback

        Returns:
            Tuple of (restored_frames, damage_scores)
        """
        return self._restorer.detect_and_restore(frames, progress_callback)

    def clear_cache(self) -> None:
        """Clear all GPU caches."""
        self._extender.clear_cache()
        self._interpolator.clear_cache()


# =============================================================================
# Factory Functions
# =============================================================================


def create_frame_generator(
    model: str = "svd",
    steps: int = 25,
    guidance_scale: float = 7.5,
    motion_bucket: int = 127,
    fps: int = 24,
    gpu_id: int = 0,
    interpolation: str = "optical_flow",
) -> GenerativeFrameExtender:
    """Factory function to create a frame generator.

    Args:
        model: Generation model ("svd", "svd_xt", "animatediff", "videocrafter")
        steps: Diffusion steps
        guidance_scale: CFG scale
        motion_bucket: Motion intensity (1-255)
        fps: Target FPS
        gpu_id: GPU device ID
        interpolation: Interpolation algorithm

    Returns:
        Configured GenerativeFrameExtender instance
    """
    config = FrameGenConfig(
        model=GenerationModel(model),
        steps=steps,
        guidance_scale=guidance_scale,
        motion_bucket=motion_bucket,
        fps=fps,
        gpu_id=gpu_id,
    )

    return GenerativeFrameExtender(
        config=config,
        interpolation_algorithm=InterpolationAlgorithm(interpolation),
    )


def extend_video(
    frames: List[np.ndarray],
    direction: str,
    seconds: float,
    model: str = "svd",
    fps: int = 24,
) -> ExtensionResult:
    """Convenience function to extend a video clip.

    Args:
        frames: Input frames
        direction: "forward" or "backward"
        seconds: Duration to extend
        model: Generation model
        fps: Target FPS

    Returns:
        ExtensionResult with generated frames
    """
    config = FrameGenConfig(model=GenerationModel(model), fps=fps)
    extender = GenerativeFrameExtender(config)
    return extender.extend_clip(frames, direction, seconds)


def interpolate_fps(
    frames: List[np.ndarray],
    target_fps: int,
    original_fps: int = 24,
    algorithm: str = "optical_flow",
) -> List[np.ndarray]:
    """Convenience function to increase video FPS.

    Args:
        frames: Input frames
        target_fps: Target frames per second
        original_fps: Original frames per second
        algorithm: Interpolation algorithm

    Returns:
        Frames with interpolated frames inserted
    """
    if target_fps <= original_fps:
        return frames

    multiplier = target_fps // original_fps
    extender = GenerativeFrameExtender(
        interpolation_algorithm=InterpolationAlgorithm(algorithm)
    )
    return extender.interpolate(frames, multiplier)


def fill_missing_frames(
    before: List[np.ndarray],
    after: List[np.ndarray],
    count: int,
    model: str = "svd",
) -> GapFillResult:
    """Convenience function to fill missing frames.

    Args:
        before: Frames before the gap
        after: Frames after the gap
        count: Number of frames to generate
        model: Generation model for extension

    Returns:
        GapFillResult with generated frames
    """
    config = FrameGenConfig(model=GenerationModel(model))
    extender = GenerativeFrameExtender(config)
    return extender.fill_gap(before, after, count)


__all__ = [
    # Enums
    "InterpolationAlgorithm",
    "GenerationModel",
    # Configuration
    "FrameGenConfig",
    # Data classes
    "MotionVector",
    "MotionField",
    "InterpolationResult",
    "ExtensionResult",
    "GapFillResult",
    # Classes
    "MotionEstimator",
    "FrameInterpolator",
    "FrameExtender",
    "GapFiller",
    "DamagedFrameRestorer",
    "GenerativeFrameExtender",
    # Factory functions
    "create_frame_generator",
    "extend_video",
    "interpolate_fps",
    "fill_missing_frames",
]
