"""Unified Super-Resolution Processor for video restoration.

This module provides a unified interface for multiple super-resolution backends,
automatically selecting the optimal backend based on hardware capabilities.

Supported backends:
- Low VRAM (2-4GB):
  * realesrgan_ncnn: NCNN-Vulkan backend (AMD/Intel GPUs)
  * realesrgan_x2: PyTorch Real-ESRGAN 2x (fast, lower VRAM)
  * realesrgan_x4: PyTorch Real-ESRGAN 4x
  * realesrgan_anime: Animation-optimized Real-ESRGAN

- Medium VRAM (8GB):
  * hat_small: HAT-S model (faster, good quality)
  * hat_base: HAT-B model (balanced)

- High VRAM (12GB+):
  * hat_large: HAT-L model (best HAT quality)
  * basicvsr_pp: BasicVSR++ with temporal consistency
  * vrt: Video Restoration Transformer

- Very High VRAM (16GB+):
  * diffusion: Diffusion-based SR (maximum quality)
  * ensemble: Multi-model ensemble voting

The module preserves ALL existing functionality from:
- pytorch_realesrgan.py: PyTorchESRGANConfig, enhance_frame_pytorch, etc.
- ncnn_vulkan.py: NcnnVulkanBackend, NcnnVulkanConfig, etc.
- hat_upscaler.py: HATUpscaler, HATConfig, HATResult, etc.
- diffusion_sr.py: DiffusionSRProcessor, DiffusionSRConfig, etc.
- ensemble_sr.py: EnsembleSR, EnsembleConfig, EnsembleResult, etc.
- advanced_models.py: BasicVSRPP, VRTProcessor, AdvancedModelConfig, etc.

Example:
    >>> config = SRConfig(scale=4, backend="auto")
    >>> hardware = detect_hardware()
    >>> sr = SuperResolution(config, hardware)
    >>> result = sr.upscale(input_dir, output_dir)

    # Or use the factory for automatic setup:
    >>> result = upscale_frames(input_dir, output_dir, scale=4)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

# Import HardwareInfo and related utilities from denoising module
from .denoising import (
    HardwareTier,
    HardwareInfo,
    GPUVendor,
    detect_hardware,
    get_hardware_tier,
)

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.debug("PyTorch not available")


# =============================================================================
# Backend Enumeration
# =============================================================================

class SRBackendType(Enum):
    """Super-resolution backend types.

    Backends are organized by VRAM requirements.
    """

    # Low VRAM (2-4GB)
    REALESRGAN_NCNN = "realesrgan_ncnn"
    """NCNN-Vulkan backend for AMD/Intel GPUs. Works on all Vulkan-capable GPUs."""

    REALESRGAN_X2 = "realesrgan_x2"
    """PyTorch Real-ESRGAN 2x scale. Fast, lower VRAM usage."""

    REALESRGAN_X4 = "realesrgan_x4"
    """PyTorch Real-ESRGAN 4x scale. Standard quality."""

    REALESRGAN_ANIME = "realesrgan_anime"
    """Real-ESRGAN optimized for animation content."""

    # Medium VRAM (8GB)
    HAT_SMALL = "hat_small"
    """HAT-S: Faster variant with good quality."""

    HAT_BASE = "hat_base"
    """HAT-B: Balanced speed and quality."""

    # High VRAM (12GB+)
    HAT_LARGE = "hat_large"
    """HAT-L: Best HAT quality, slower."""

    BASICVSR_PP = "basicvsr_pp"
    """BasicVSR++ with temporal consistency for video."""

    VRT = "vrt"
    """Video Restoration Transformer. Maximum temporal quality."""

    # Very High VRAM (16GB+)
    DIFFUSION = "diffusion"
    """Diffusion-based SR. Maximum quality, slowest."""

    ENSEMBLE = "ensemble"
    """Multi-model ensemble with voting. Combines multiple backends."""


# VRAM requirements in MB for each backend
BACKEND_VRAM_REQUIREMENTS: Dict[SRBackendType, int] = {
    SRBackendType.REALESRGAN_NCNN: 2000,
    SRBackendType.REALESRGAN_X2: 3000,
    SRBackendType.REALESRGAN_X4: 4000,
    SRBackendType.REALESRGAN_ANIME: 4000,
    SRBackendType.HAT_SMALL: 6000,
    SRBackendType.HAT_BASE: 8000,
    SRBackendType.HAT_LARGE: 12000,
    SRBackendType.BASICVSR_PP: 12000,
    SRBackendType.VRT: 12000,
    SRBackendType.DIFFUSION: 16000,
    SRBackendType.ENSEMBLE: 16000,
}

# Quality ranking (higher = better quality, not speed)
BACKEND_QUALITY_RANKING: Dict[SRBackendType, int] = {
    SRBackendType.REALESRGAN_NCNN: 50,
    SRBackendType.REALESRGAN_X2: 55,
    SRBackendType.REALESRGAN_X4: 60,
    SRBackendType.REALESRGAN_ANIME: 65,  # Higher for anime content
    SRBackendType.HAT_SMALL: 70,
    SRBackendType.HAT_BASE: 80,
    SRBackendType.HAT_LARGE: 85,
    SRBackendType.BASICVSR_PP: 82,  # Temporal consistency bonus
    SRBackendType.VRT: 88,  # Best temporal quality
    SRBackendType.DIFFUSION: 90,  # Maximum single-frame quality
    SRBackendType.ENSEMBLE: 95,  # Combined quality
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SRConfig:
    """Unified super-resolution configuration.

    Attributes:
        scale: Upscaling factor (2 or 4).
        backend: Backend to use ("auto" for automatic selection).
        half_precision: Use FP16 for reduced VRAM.
        tile_size: Tile size for large frames (0 = auto, None = no tiling).
        tile_overlap: Overlap between tiles.
        temporal_window: Frames for temporal consistency (video models).
        gpu_id: GPU device ID.
        quality_preset: Quality preset ("fast", "balanced", "quality", "maximum").
        fallback_chain: Ordered list of fallback backends.
    """

    scale: int = 4
    backend: str = "auto"  # Backend name or "auto"
    half_precision: bool = True
    tile_size: int = 0  # 0 = auto, None = no tiling
    tile_overlap: int = 32
    temporal_window: int = 7
    gpu_id: int = 0
    quality_preset: str = "balanced"  # fast, balanced, quality, maximum
    fallback_chain: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {self.scale}")
        if self.temporal_window < 1:
            raise ValueError(f"temporal_window must be >= 1, got {self.temporal_window}")

        valid_presets = ["fast", "balanced", "quality", "maximum"]
        if self.quality_preset not in valid_presets:
            raise ValueError(f"quality_preset must be one of {valid_presets}")


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class SRResult:
    """Result of super-resolution processing.

    Attributes:
        frames_processed: Number of frames successfully processed.
        frames_failed: Number of frames that failed.
        output_dir: Path to output directory.
        backend_used: Name of the backend that was used.
        processing_time_seconds: Total processing time.
        avg_fps: Average frames per second.
        peak_vram_mb: Peak VRAM usage.
        scale_factor: Actual scale factor used.
        warnings: Any warnings generated during processing.
    """

    frames_processed: int = 0
    frames_failed: int = 0
    output_dir: Optional[Path] = None
    backend_used: str = "unknown"
    processing_time_seconds: float = 0.0
    avg_fps: float = 0.0
    peak_vram_mb: int = 0
    scale_factor: int = 4
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Backend Protocol
# =============================================================================

class SRBackend(ABC):
    """Abstract base class for super-resolution backends.

    All backends must implement this interface for the unified SuperResolution
    class to work correctly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier name."""
        ...

    @property
    @abstractmethod
    def supported_scales(self) -> List[int]:
        """List of supported scale factors."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @abstractmethod
    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage in MB for given frame size.

        Args:
            width: Input frame width.
            height: Input frame height.
            scale: Upscaling factor.

        Returns:
            Estimated VRAM usage in MB.
        """
        ...

    @abstractmethod
    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame.

        Args:
            frame: Input frame (BGR numpy array).
            scale: Upscaling factor.

        Returns:
            Upscaled frame (BGR numpy array).
        """
        ...

    @abstractmethod
    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames in a directory.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            scale: Upscaling factor.
            progress_callback: Optional progress callback (0-1).

        Returns:
            SRResult with processing statistics.
        """
        ...

    def clear_cache(self) -> None:
        """Clear any cached models or GPU memory."""
        pass


# =============================================================================
# Real-ESRGAN NCNN Backend (Vulkan)
# =============================================================================

class NCNNVulkanSRBackend(SRBackend):
    """NCNN-Vulkan backend wrapping existing NcnnVulkanBackend.

    Works on AMD, Intel, and NVIDIA GPUs via Vulkan.
    """

    def __init__(self, config: SRConfig, hardware: HardwareInfo):
        """Initialize NCNN backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
        """
        self.config = config
        self.hardware = hardware
        self._backend = None
        self._ncnn_config = None

    @property
    def name(self) -> str:
        return "realesrgan_ncnn"

    @property
    def supported_scales(self) -> List[int]:
        return [2, 3, 4]

    def is_available(self) -> bool:
        """Check if ncnn-vulkan is available."""
        try:
            from ..ncnn_vulkan import NcnnVulkanBackend
            backend = NcnnVulkanBackend()
            return backend.is_available()
        except ImportError:
            return False

    def _ensure_backend(self) -> None:
        """Ensure backend is initialized."""
        if self._backend is None:
            from ..ncnn_vulkan import NcnnVulkanBackend, NcnnVulkanConfig

            self._backend = NcnnVulkanBackend()
            self._ncnn_config = NcnnVulkanConfig(
                model_name=self._get_model_name(),
                scale_factor=self.config.scale,
                tile_size=self.config.tile_size,
                gpu_id=self.config.gpu_id,
                require_gpu=True,
            )

    def _get_model_name(self) -> str:
        """Get appropriate ncnn model name."""
        if self.config.scale == 2:
            return "realesrgan-x2plus"
        return "realesrgan-x4plus"

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        # NCNN is very memory efficient
        base_vram = 500  # Base model size
        frame_vram = (width * height * 3 * 4 * scale * scale) // (1024 * 1024)
        return base_vram + frame_vram

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame using ncnn-vulkan."""
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for frame processing")

        self._ensure_backend()

        import tempfile
        with tempfile.TemporaryDirectory(prefix="ncnn_sr_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.png"
            output_path = tmpdir / "output.png"

            cv2.imwrite(str(input_path), frame)

            success, error = self._backend.enhance_frame(
                input_path, output_path, self._ncnn_config
            )

            if not success:
                raise RuntimeError(f"NCNN upscaling failed: {error}")

            return cv2.imread(str(output_path))

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames using ncnn-vulkan."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        self._ensure_backend()

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        success, fail, errors = self._backend.enhance_directory(
            input_dir, output_dir, self._ncnn_config, progress_callback
        )

        result.frames_processed = success
        result.frames_failed = fail
        result.processing_time_seconds = time.time() - start_time
        result.warnings = errors

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        return result


# =============================================================================
# Real-ESRGAN PyTorch Backend
# =============================================================================

class RealESRGANBackend(SRBackend):
    """PyTorch Real-ESRGAN backend wrapping existing implementation.

    Supports multiple model variants for different use cases.
    """

    def __init__(
        self,
        config: SRConfig,
        hardware: HardwareInfo,
        model_variant: str = "x4plus",
    ):
        """Initialize Real-ESRGAN backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
            model_variant: Model variant ("x2plus", "x4plus", "anime").
        """
        self.config = config
        self.hardware = hardware
        self.model_variant = model_variant
        self._esrgan_config = None

    @property
    def name(self) -> str:
        return f"realesrgan_{self.model_variant}"

    @property
    def supported_scales(self) -> List[int]:
        if "x2" in self.model_variant:
            return [2]
        return [4]

    def is_available(self) -> bool:
        """Check if PyTorch Real-ESRGAN is available."""
        try:
            from ..pytorch_realesrgan import is_pytorch_esrgan_available
            return is_pytorch_esrgan_available()
        except ImportError:
            return False

    def _get_model_name(self) -> str:
        """Get full model name."""
        models = {
            "x2plus": "RealESRGAN_x2plus",
            "x4plus": "RealESRGAN_x4plus",
            "anime": "RealESRGAN_x4plus_anime_6B",
            "animevideo": "realesr-animevideov3",
            "general": "realesr-general-x4v3",
        }
        return models.get(self.model_variant, "RealESRGAN_x4plus")

    def _ensure_config(self) -> None:
        """Ensure configuration is initialized."""
        if self._esrgan_config is None:
            from ..pytorch_realesrgan import PyTorchESRGANConfig

            self._esrgan_config = PyTorchESRGANConfig(
                model_name=self._get_model_name(),
                scale_factor=self.config.scale,
                tile_size=self.config.tile_size,
                half_precision=self.config.half_precision,
                gpu_id=self.config.gpu_id,
            )

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        base_vram = 1500 if "anime" in self.model_variant else 2000
        # Rough estimate: 4 bytes per pixel * 3 channels * input + output
        frame_vram = (width * height * 3 * 4 * (1 + scale * scale)) // (1024 * 1024)
        return base_vram + frame_vram

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame."""
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for frame processing")

        self._ensure_config()

        from ..pytorch_realesrgan import get_upsampler

        upsampler = get_upsampler(self._esrgan_config)
        output, _ = upsampler.enhance(frame, outscale=scale)
        return output

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        if not HAS_OPENCV:
            result.warnings.append("OpenCV not available")
            return result

        self._ensure_config()

        from ..pytorch_realesrgan import enhance_frame_pytorch

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            result.warnings.append("No frames found")
            return result

        total_frames = len(frames)
        peak_vram = 0

        for i, frame_path in enumerate(frames):
            try:
                output_path = output_dir / frame_path.name
                success, error = enhance_frame_pytorch(
                    frame_path, output_path, self._esrgan_config
                )

                if success:
                    result.frames_processed += 1
                else:
                    result.frames_failed += 1
                    result.warnings.append(f"Frame {frame_path.name}: {error}")

                # Track VRAM
                if HAS_TORCH and torch.cuda.is_available():
                    current = torch.cuda.max_memory_allocated(self.config.gpu_id)
                    peak_vram = max(peak_vram, current)

            except Exception as e:
                result.frames_failed += 1
                result.warnings.append(f"Frame {frame_path.name}: {str(e)}")

            if progress_callback:
                progress_callback((i + 1) / total_frames)

        result.processing_time_seconds = time.time() - start_time
        result.peak_vram_mb = peak_vram // (1024 * 1024)

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        return result

    def clear_cache(self) -> None:
        """Clear model cache."""
        try:
            from ..pytorch_realesrgan import clear_upsampler_cache
            clear_upsampler_cache()
        except ImportError:
            pass


# =============================================================================
# HAT Backend
# =============================================================================

class HATBackend(SRBackend):
    """HAT (Hybrid Attention Transformer) backend.

    Provides higher quality than Real-ESRGAN with better detail preservation.
    """

    def __init__(
        self,
        config: SRConfig,
        hardware: HardwareInfo,
        model_size: str = "large",
    ):
        """Initialize HAT backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
            model_size: Model size ("small", "base", "large").
        """
        self.config = config
        self.hardware = hardware
        self.model_size = model_size
        self._upscaler = None

    @property
    def name(self) -> str:
        return f"hat_{self.model_size}"

    @property
    def supported_scales(self) -> List[int]:
        return [2, 4]

    def is_available(self) -> bool:
        """Check if HAT is available."""
        try:
            from ..hat_upscaler import HATUpscaler, HATConfig, HATModelSize
            config = HATConfig(model_size=HATModelSize(self.model_size))
            upscaler = HATUpscaler(config)
            return upscaler.is_available()
        except (ImportError, Exception):
            return False

    def _ensure_upscaler(self) -> None:
        """Ensure upscaler is initialized."""
        if self._upscaler is None:
            from ..hat_upscaler import HATUpscaler, HATConfig, HATModelSize

            config = HATConfig(
                scale=self.config.scale,
                model_size=HATModelSize(self.model_size),
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
                tile_size=self.config.tile_size if self.config.tile_size else 0,
                tile_overlap=self.config.tile_overlap,
            )
            self._upscaler = HATUpscaler(config)

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        base_vram = {
            "small": 4000,
            "base": 6000,
            "large": 10000,
        }.get(self.model_size, 6000)

        frame_vram = (width * height * 3 * 4 * (1 + scale * scale)) // (1024 * 1024)
        return base_vram + frame_vram

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame."""
        self._ensure_upscaler()
        return self._upscaler.upscale_frame(frame)

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        self._ensure_upscaler()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        hat_result = self._upscaler.upscale_frames(
            input_dir, output_dir, progress_callback
        )

        result.frames_processed = hat_result.frames_processed
        result.frames_failed = hat_result.frames_failed
        result.processing_time_seconds = hat_result.processing_time_seconds
        result.avg_fps = hat_result.avg_fps
        result.peak_vram_mb = hat_result.peak_vram_mb

        return result

    def clear_cache(self) -> None:
        """Clear model cache."""
        if self._upscaler:
            self._upscaler.clear_cache()
            self._upscaler = None


# =============================================================================
# BasicVSR++ Backend
# =============================================================================

class BasicVSRPPBackend(SRBackend):
    """BasicVSR++ backend for temporal video super-resolution.

    Provides temporal consistency across frames for smoother video output.
    """

    def __init__(self, config: SRConfig, hardware: HardwareInfo):
        """Initialize BasicVSR++ backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
        """
        self.config = config
        self.hardware = hardware
        self._processor = None

    @property
    def name(self) -> str:
        return "basicvsr_pp"

    @property
    def supported_scales(self) -> List[int]:
        return [4]

    def is_available(self) -> bool:
        """Check if BasicVSR++ is available."""
        try:
            from ..advanced_models import BasicVSRPP, AdvancedModelConfig, AdvancedModel
            config = AdvancedModelConfig(model=AdvancedModel.BASICVSR_PP)
            processor = BasicVSRPP(config)
            return processor.is_available()
        except (ImportError, Exception):
            return False

    def _ensure_processor(self) -> None:
        """Ensure processor is initialized."""
        if self._processor is None:
            from ..advanced_models import BasicVSRPP, AdvancedModelConfig, AdvancedModel

            config = AdvancedModelConfig(
                model=AdvancedModel.BASICVSR_PP,
                scale_factor=self.config.scale,
                temporal_window=self.config.temporal_window,
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
                tile_size=self.config.tile_size if self.config.tile_size else 0,
            )
            self._processor = BasicVSRPP(config)

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        # BasicVSR++ needs memory for multiple frames
        base_vram = 8000
        frame_vram = (width * height * 3 * 4 * self.config.temporal_window) // (1024 * 1024)
        return base_vram + frame_vram * 2

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame.

        Note: BasicVSR++ is designed for video sequences. Single frame
        upscaling will work but won't benefit from temporal consistency.
        """
        logger.warning("BasicVSR++ is optimized for video sequences, not single frames")

        self._ensure_processor()

        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required")

        import tempfile
        with tempfile.TemporaryDirectory(prefix="basicvsr_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            cv2.imwrite(str(input_dir / "frame_000001.png"), frame)

            self._processor.enhance_frames(input_dir, output_dir)

            output_path = output_dir / "frame_000001.png"
            if output_path.exists():
                return cv2.imread(str(output_path))

            raise RuntimeError("BasicVSR++ processing failed")

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames with temporal consistency."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        self._ensure_processor()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        proc_result = self._processor.enhance_frames(
            input_dir, output_dir, progress_callback
        )

        result.frames_processed = proc_result.frames_processed
        result.frames_failed = proc_result.frames_failed
        result.processing_time_seconds = proc_result.processing_time_seconds
        result.avg_fps = proc_result.avg_fps
        result.peak_vram_mb = proc_result.peak_vram_mb

        return result


# =============================================================================
# VRT Backend
# =============================================================================

class VRTBackend(SRBackend):
    """Video Restoration Transformer backend.

    Provides maximum quality with excellent temporal consistency.
    """

    def __init__(self, config: SRConfig, hardware: HardwareInfo):
        """Initialize VRT backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
        """
        self.config = config
        self.hardware = hardware
        self._processor = None

    @property
    def name(self) -> str:
        return "vrt"

    @property
    def supported_scales(self) -> List[int]:
        return [4]

    def is_available(self) -> bool:
        """Check if VRT is available."""
        try:
            from ..advanced_models import VRTProcessor, AdvancedModelConfig, AdvancedModel
            config = AdvancedModelConfig(model=AdvancedModel.VRT)
            processor = VRTProcessor(config)
            return processor.is_available()
        except (ImportError, Exception):
            return False

    def _ensure_processor(self) -> None:
        """Ensure processor is initialized."""
        if self._processor is None:
            from ..advanced_models import VRTProcessor, AdvancedModelConfig, AdvancedModel

            config = AdvancedModelConfig(
                model=AdvancedModel.VRT,
                scale_factor=self.config.scale,
                temporal_window=min(self.config.temporal_window, 6),  # VRT max
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
            )
            self._processor = VRTProcessor(config)

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        # VRT is very VRAM hungry
        base_vram = 12000
        frame_vram = (width * height * 3 * 4 * 6) // (1024 * 1024)  # 6 frames
        return base_vram + frame_vram * 3

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame."""
        logger.warning("VRT is optimized for video sequences, not single frames")

        self._ensure_processor()

        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required")

        import tempfile
        with tempfile.TemporaryDirectory(prefix="vrt_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            cv2.imwrite(str(input_dir / "frame_000001.png"), frame)

            self._processor.enhance_frames(input_dir, output_dir)

            output_path = output_dir / "frame_000001.png"
            if output_path.exists():
                return cv2.imread(str(output_path))

            raise RuntimeError("VRT processing failed")

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        self._ensure_processor()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        proc_result = self._processor.enhance_frames(
            input_dir, output_dir, progress_callback
        )

        result.frames_processed = proc_result.frames_processed
        result.frames_failed = proc_result.frames_failed
        result.processing_time_seconds = proc_result.processing_time_seconds
        result.avg_fps = proc_result.avg_fps
        result.peak_vram_mb = proc_result.peak_vram_mb

        return result


# =============================================================================
# Diffusion SR Backend
# =============================================================================

class DiffusionSRBackend(SRBackend):
    """Diffusion-based super-resolution backend.

    Provides maximum quality with photorealistic detail generation.
    """

    def __init__(
        self,
        config: SRConfig,
        hardware: HardwareInfo,
        model_name: str = "upscale_a_video",
    ):
        """Initialize Diffusion SR backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
            model_name: Diffusion model name.
        """
        self.config = config
        self.hardware = hardware
        self.model_name = model_name
        self._processor = None

    @property
    def name(self) -> str:
        return "diffusion"

    @property
    def supported_scales(self) -> List[int]:
        return [2, 4]

    def is_available(self) -> bool:
        """Check if Diffusion SR is available."""
        try:
            from ..diffusion_sr import DiffusionSRProcessor, DiffusionSRConfig
            config = DiffusionSRConfig()
            processor = DiffusionSRProcessor(config)
            return processor.is_available()
        except (ImportError, Exception):
            return False

    def _ensure_processor(self) -> None:
        """Ensure processor is initialized."""
        if self._processor is None:
            from ..diffusion_sr import DiffusionSRProcessor, DiffusionSRConfig, DiffusionModel

            config = DiffusionSRConfig(
                model=DiffusionModel(self.model_name),
                scale_factor=self.config.scale,
                half_precision=self.config.half_precision,
                tile_size=self.config.tile_size if self.config.tile_size else 512,
                gpu_id=self.config.gpu_id,
            )
            self._processor = DiffusionSRProcessor(config)

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        # Diffusion models are VRAM hungry
        return 16000 + (width * height * 3 * 4) // (1024 * 1024)

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame."""
        self._ensure_processor()

        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required")

        import tempfile
        with tempfile.TemporaryDirectory(prefix="diffusion_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            cv2.imwrite(str(input_dir / "frame_000001.png"), frame)

            self._processor.enhance_video(input_dir, output_dir)

            output_path = output_dir / "frame_000001.png"
            if output_path.exists():
                return cv2.imread(str(output_path))

            raise RuntimeError("Diffusion SR processing failed")

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        self._ensure_processor()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        diff_result = self._processor.enhance_video(
            input_dir, output_dir, progress_callback=progress_callback
        )

        result.frames_processed = diff_result.frames_processed
        result.frames_failed = diff_result.frames_failed
        result.processing_time_seconds = diff_result.total_time_seconds
        result.peak_vram_mb = diff_result.peak_vram_mb

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        return result

    def clear_cache(self) -> None:
        """Clear model cache."""
        if self._processor:
            self._processor.clear_cache()
            self._processor = None


# =============================================================================
# Ensemble SR Backend
# =============================================================================

class EnsembleSRBackend(SRBackend):
    """Ensemble super-resolution backend.

    Combines multiple SR models for maximum quality through voting.
    """

    def __init__(
        self,
        config: SRConfig,
        hardware: HardwareInfo,
        models: Optional[List[str]] = None,
    ):
        """Initialize Ensemble backend.

        Args:
            config: SR configuration.
            hardware: Detected hardware info.
            models: List of models to ensemble (default: ["hat", "realesrgan"]).
        """
        self.config = config
        self.hardware = hardware
        self.models = models or ["hat", "realesrgan"]
        self._processor = None

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def supported_scales(self) -> List[int]:
        return [4]

    def is_available(self) -> bool:
        """Check if Ensemble SR is available."""
        try:
            from ..ensemble_sr import EnsembleSR, EnsembleConfig
            config = EnsembleConfig(models=self.models)
            processor = EnsembleSR(config)
            return processor.is_available()
        except (ImportError, Exception):
            return False

    def _ensure_processor(self) -> None:
        """Ensure processor is initialized."""
        if self._processor is None:
            from ..ensemble_sr import EnsembleSR, EnsembleConfig, VotingMethod

            config = EnsembleConfig(
                models=self.models,
                voting_method=VotingMethod.WEIGHTED,
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
            )
            self._processor = EnsembleSR(config)

    def estimate_vram_usage(self, width: int, height: int, scale: int) -> int:
        """Estimate VRAM usage."""
        # Need VRAM for all ensemble models
        return 20000 + len(self.models) * 4000

    def upscale_frame(self, frame: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale a single frame."""
        self._ensure_processor()
        return self._processor.upscale_frame(frame)

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        scale: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames."""
        result = SRResult(backend_used=self.name, scale_factor=scale)
        start_time = time.time()

        self._ensure_processor()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        ens_result = self._processor.upscale_frames(
            input_dir, output_dir, progress_callback
        )

        result.frames_processed = ens_result.frames_processed
        result.frames_failed = ens_result.frames_failed
        result.processing_time_seconds = ens_result.processing_time_seconds

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        return result

    def clear_cache(self) -> None:
        """Clear model cache."""
        if self._processor:
            self._processor.clear_cache()
            self._processor = None


# =============================================================================
# Unified SuperResolution Class
# =============================================================================

class SuperResolution:
    """Unified super-resolution processor with automatic backend selection.

    Automatically selects the optimal backend based on hardware capabilities
    and provides graceful fallback when backends are unavailable.

    Example:
        >>> config = SRConfig(scale=4, backend="auto")
        >>> hardware = detect_hardware()
        >>> sr = SuperResolution(config, hardware)
        >>> result = sr.upscale(input_dir, output_dir)
    """

    # Backend mapping for all supported backends
    BACKENDS: Dict[str, type] = {
        # Low VRAM (2-4GB)
        "realesrgan_ncnn": NCNNVulkanSRBackend,
        "realesrgan_x2": RealESRGANBackend,
        "realesrgan_x4": RealESRGANBackend,
        "realesrgan_anime": RealESRGANBackend,

        # Medium VRAM (8GB)
        "hat_small": HATBackend,
        "hat_base": HATBackend,

        # High VRAM (12GB+)
        "hat_large": HATBackend,
        "basicvsr_pp": BasicVSRPPBackend,
        "vrt": VRTBackend,

        # Very High VRAM (16GB+)
        "diffusion": DiffusionSRBackend,
        "ensemble": EnsembleSRBackend,
    }

    # Default fallback chains by quality preset
    FALLBACK_CHAINS: Dict[str, List[str]] = {
        "fast": [
            "realesrgan_ncnn", "realesrgan_x4", "realesrgan_x2",
        ],
        "balanced": [
            "hat_base", "realesrgan_x4", "hat_small", "realesrgan_ncnn",
        ],
        "quality": [
            "hat_large", "hat_base", "vrt", "basicvsr_pp", "realesrgan_x4",
        ],
        "maximum": [
            "ensemble", "diffusion", "hat_large", "vrt", "hat_base",
        ],
    }

    def __init__(
        self,
        config: Optional[SRConfig] = None,
        hardware: Optional[HardwareInfo] = None,
    ):
        """Initialize SuperResolution processor.

        Args:
            config: SR configuration (uses defaults if None).
            hardware: Detected hardware (auto-detects if None).
        """
        self.config = config or SRConfig()
        self.hardware = hardware or detect_hardware()
        self._backend: Optional[SRBackend] = None
        self._available_backends: Dict[str, bool] = {}

        # Select backend
        self.backend = self._select_backend()

    def _create_backend(self, backend_name: str) -> Optional[SRBackend]:
        """Create a backend instance by name.

        Args:
            backend_name: Name of the backend.

        Returns:
            Backend instance or None if creation fails.
        """
        backend_class = self.BACKENDS.get(backend_name)
        if not backend_class:
            logger.warning(f"Unknown backend: {backend_name}")
            return None

        try:
            # Create backend with appropriate parameters
            if backend_name == "realesrgan_ncnn":
                return NCNNVulkanSRBackend(self.config, self.hardware)

            elif backend_name.startswith("realesrgan_"):
                variant = backend_name.replace("realesrgan_", "")
                if variant == "anime":
                    variant = "anime"
                return RealESRGANBackend(self.config, self.hardware, variant)

            elif backend_name.startswith("hat_"):
                size = backend_name.replace("hat_", "")
                return HATBackend(self.config, self.hardware, size)

            elif backend_name == "basicvsr_pp":
                return BasicVSRPPBackend(self.config, self.hardware)

            elif backend_name == "vrt":
                return VRTBackend(self.config, self.hardware)

            elif backend_name == "diffusion":
                return DiffusionSRBackend(self.config, self.hardware)

            elif backend_name == "ensemble":
                return EnsembleSRBackend(self.config, self.hardware)

            else:
                logger.warning(f"Unhandled backend: {backend_name}")
                return None

        except Exception as e:
            logger.warning(f"Failed to create backend {backend_name}: {e}")
            return None

    def _check_backend_available(self, backend_name: str) -> bool:
        """Check if a backend is available.

        Args:
            backend_name: Name of the backend.

        Returns:
            True if available.
        """
        if backend_name in self._available_backends:
            return self._available_backends[backend_name]

        backend = self._create_backend(backend_name)
        available = backend is not None and backend.is_available()
        self._available_backends[backend_name] = available

        return available

    def _select_optimal_backend_for_hardware(self) -> str:
        """Select optimal backend based on hardware tier.

        Returns:
            Backend name.
        """
        tier = self.hardware.tier
        vram = self.hardware.vram_free_mb

        # Very high VRAM (16GB+)
        if vram >= 16000:
            if self._check_backend_available("ensemble"):
                return "ensemble"
            if self._check_backend_available("diffusion"):
                return "diffusion"

        # High VRAM (12GB+)
        if vram >= 12000:
            if self._check_backend_available("hat_large"):
                return "hat_large"
            if self._check_backend_available("vrt"):
                return "vrt"
            if self._check_backend_available("basicvsr_pp"):
                return "basicvsr_pp"

        # Medium VRAM (8GB)
        if vram >= 8000:
            if self._check_backend_available("hat_base"):
                return "hat_base"
            if self._check_backend_available("hat_small"):
                return "hat_small"

        # Low VRAM or NCNN (AMD/Intel)
        if tier == HardwareTier.NCNN or self.hardware.gpu_vendor in (GPUVendor.AMD, GPUVendor.INTEL):
            if self._check_backend_available("realesrgan_ncnn"):
                return "realesrgan_ncnn"

        # CUDA fallback
        if vram >= 4000:
            if self._check_backend_available("realesrgan_x4"):
                return "realesrgan_x4"

        if vram >= 3000:
            if self._check_backend_available("realesrgan_x2"):
                return "realesrgan_x2"

        # Last resort
        if self._check_backend_available("realesrgan_ncnn"):
            return "realesrgan_ncnn"

        if self._check_backend_available("realesrgan_x4"):
            return "realesrgan_x4"

        raise RuntimeError("No super-resolution backend available")

    def _select_backend(self) -> SRBackend:
        """Select and create the appropriate backend.

        Returns:
            Selected backend instance.
        """
        backend_name = self.config.backend

        # Auto-select based on hardware
        if backend_name == "auto":
            backend_name = self._select_optimal_backend_for_hardware()
            logger.info(f"Auto-selected backend: {backend_name}")

        # Try to create the requested backend
        backend = self._create_backend(backend_name)

        if backend and backend.is_available():
            logger.info(f"Using backend: {backend_name}")
            return backend

        # Fallback chain
        fallback_chain = self.config.fallback_chain or self.FALLBACK_CHAINS.get(
            self.config.quality_preset, self.FALLBACK_CHAINS["balanced"]
        )

        logger.warning(f"Backend {backend_name} not available, trying fallbacks")

        for fallback_name in fallback_chain:
            if fallback_name == backend_name:
                continue

            fallback = self._create_backend(fallback_name)
            if fallback and fallback.is_available():
                logger.info(f"Using fallback backend: {fallback_name}")
                return fallback

        raise RuntimeError("No super-resolution backend available after fallbacks")

    def get_available_backends(self) -> List[str]:
        """Get list of available backends on this system.

        Returns:
            List of available backend names.
        """
        available = []
        for name in self.BACKENDS.keys():
            if self._check_backend_available(name):
                available.append(name)
        return available

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend.

        Returns:
            Dictionary with backend information.
        """
        return {
            "name": self.backend.name,
            "supported_scales": self.backend.supported_scales,
            "hardware_tier": self.hardware.tier.value,
            "vram_total_mb": self.hardware.vram_total_mb,
            "vram_free_mb": self.hardware.vram_free_mb,
            "gpu_name": self.hardware.gpu_name,
        }

    def estimate_vram_usage(self, width: int, height: int) -> int:
        """Estimate VRAM usage for given frame size.

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            Estimated VRAM usage in MB.
        """
        return self.backend.estimate_vram_usage(width, height, self.config.scale)

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single frame.

        Args:
            frame: Input frame (BGR numpy array).

        Returns:
            Upscaled frame (BGR numpy array).
        """
        return self.backend.upscale_frame(frame, self.config.scale)

    def upscale(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SRResult:
        """Upscale all frames in a directory.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional progress callback (0-1).

        Returns:
            SRResult with processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        logger.info(
            f"Starting {self.config.scale}x upscale with {self.backend.name} backend"
        )

        return self.backend.upscale_frames(
            input_dir, output_dir, self.config.scale, progress_callback
        )

    # Alias for backward compatibility
    def process(
        self,
        frames: List[np.ndarray],
        scale: int = 4,
    ) -> List[np.ndarray]:
        """Process a list of frames.

        Args:
            frames: List of input frames.
            scale: Upscaling factor.

        Returns:
            List of upscaled frames.
        """
        results = []
        for frame in frames:
            results.append(self.backend.upscale_frame(frame, scale))
        return results

    def clear_cache(self) -> None:
        """Clear any cached models or GPU memory."""
        if self.backend:
            self.backend.clear_cache()

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Factory Functions
# =============================================================================

def create_super_resolution(
    scale: int = 4,
    backend: str = "auto",
    quality_preset: str = "balanced",
    gpu_id: int = 0,
    **kwargs,
) -> SuperResolution:
    """Factory function to create a SuperResolution processor.

    Args:
        scale: Upscaling factor (2 or 4).
        backend: Backend name or "auto".
        quality_preset: Quality preset ("fast", "balanced", "quality", "maximum").
        gpu_id: GPU device ID.
        **kwargs: Additional configuration options.

    Returns:
        Configured SuperResolution processor.
    """
    config = SRConfig(
        scale=scale,
        backend=backend,
        quality_preset=quality_preset,
        gpu_id=gpu_id,
        **kwargs,
    )

    hardware = detect_hardware()
    return SuperResolution(config, hardware)


def upscale_frames(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    scale: int = 4,
    backend: str = "auto",
    progress_callback: Optional[Callable[[float], None]] = None,
    **kwargs,
) -> SRResult:
    """Convenience function to upscale frames in one call.

    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        scale: Upscaling factor (2 or 4).
        backend: Backend name or "auto".
        progress_callback: Optional progress callback (0-1).
        **kwargs: Additional configuration options.

    Returns:
        SRResult with processing statistics.
    """
    sr = create_super_resolution(scale=scale, backend=backend, **kwargs)
    return sr.upscale(input_dir, output_dir, progress_callback)


def get_recommended_backend(hardware: Optional[HardwareInfo] = None) -> str:
    """Get the recommended backend for the current hardware.

    Args:
        hardware: Hardware info (auto-detected if None).

    Returns:
        Recommended backend name.
    """
    if hardware is None:
        hardware = detect_hardware()

    sr = SuperResolution(SRConfig(backend="auto"), hardware)
    return sr.backend.name


def list_available_backends() -> List[str]:
    """List all available backends on this system.

    Returns:
        List of available backend names.
    """
    sr = SuperResolution(SRConfig())
    return sr.get_available_backends()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "SRBackendType",
    # Configuration
    "SRConfig",
    # Result types
    "SRResult",
    # Backend protocol
    "SRBackend",
    # Backend implementations
    "NCNNVulkanSRBackend",
    "RealESRGANBackend",
    "HATBackend",
    "BasicVSRPPBackend",
    "VRTBackend",
    "DiffusionSRBackend",
    "EnsembleSRBackend",
    # Main class
    "SuperResolution",
    # Factory functions
    "create_super_resolution",
    "upscale_frames",
    "get_recommended_backend",
    "list_available_backends",
    # Re-exports from denoising for convenience
    "HardwareTier",
    "HardwareInfo",
    "GPUVendor",
    "detect_hardware",
    "get_hardware_tier",
]
