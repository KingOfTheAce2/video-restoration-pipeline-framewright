"""Unified Denoising Processor for video restoration.

This module provides a unified interface for multiple denoising backends,
automatically selecting the optimal backend based on hardware capabilities.

Supported backends:
- CPU (Traditional): FFmpeg-based, works everywhere
- CUDA 4GB (Temporal): OpenCV + temporal filtering
- CUDA 8GB+ (TAP): Neural network denoising (Restormer/NAFNet/TAP)
- NCNN (Vulkan): Vulkan-based for AMD/Intel GPUs

The module preserves ALL existing functionality from:
- temporal_denoise.py: TemporalDenoiser, FlickerReducer, etc.
- tap_denoise.py: TAPDenoiser, MotionAdaptiveTAPDenoiser, etc.

Example:
    >>> config = DenoiserConfig(strength=0.7)
    >>> hardware = detect_hardware()
    >>> denoiser = Denoiser(config, hardware)
    >>> result = denoiser.denoise(input_dir, output_dir)

    # Or use the factory for automatic setup:
    >>> result = denoise_frames(input_dir, output_dir, strength=0.7)
"""

import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

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
# Hardware Detection
# =============================================================================

class HardwareTier(Enum):
    """Hardware capability tiers for backend selection.

    Tiers determine which denoising backends are available and optimal.
    """

    CPU = "cpu"
    """CPU-only: FFmpeg-based traditional denoising (slowest, universal)."""

    CUDA_4GB = "cuda_4gb"
    """CUDA with 4GB VRAM: Temporal denoising with optical flow."""

    CUDA_8GB = "cuda_8gb"
    """CUDA with 8GB+ VRAM: Neural network denoising (TAP/Restormer)."""

    NCNN = "ncnn"
    """Vulkan-based (ncnn): For AMD/Intel GPUs with Vulkan support."""


class GPUVendor(Enum):
    """GPU vendor classification."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """Detected hardware information.

    Attributes:
        tier: Recommended hardware tier
        gpu_available: Whether a GPU is available
        gpu_name: Name of the detected GPU
        gpu_vendor: GPU vendor
        vram_total_mb: Total GPU VRAM in MB
        vram_free_mb: Available GPU VRAM in MB
        cuda_available: Whether CUDA is available
        vulkan_available: Whether Vulkan is available
        ncnn_available: Whether ncnn-vulkan is available
        cpu_cores: Number of CPU cores
        ram_total_mb: Total system RAM in MB
    """

    tier: HardwareTier = HardwareTier.CPU
    gpu_available: bool = False
    gpu_name: str = "None"
    gpu_vendor: GPUVendor = GPUVendor.UNKNOWN
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    cuda_available: bool = False
    vulkan_available: bool = False
    ncnn_available: bool = False
    cpu_cores: int = 1
    ram_total_mb: int = 0


def _check_ncnn_available() -> bool:
    """Check if ncnn-vulkan binary is available."""
    ncnn_names = [
        "realesrgan-ncnn-vulkan",
        "realesrgan-ncnn-vulkan.exe",
    ]

    for name in ncnn_names:
        if shutil.which(name):
            return True

    # Check common installation paths
    home = Path.home()
    paths = [
        home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan.exe",
        home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan",
    ]

    return any(p.exists() for p in paths)


def _get_nvidia_gpu_info() -> Optional[Tuple[str, int, int]]:
    """Get NVIDIA GPU info via nvidia-smi.

    Returns:
        Tuple of (name, total_vram_mb, free_vram_mb) or None.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total = int(parts[1])
                free = int(parts[2])
                return (name, total, free)
    except Exception:
        pass

    return None


def _get_torch_cuda_info() -> Optional[Tuple[str, int, int]]:
    """Get CUDA info via PyTorch.

    Returns:
        Tuple of (name, total_vram_mb, free_vram_mb) or None.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        name = props.name
        total = props.total_memory // (1024 * 1024)

        # Get free memory
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0] // (1024 * 1024)

        return (name, total, free)
    except Exception:
        pass

    return None


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and determine optimal tier.

    Returns:
        HardwareInfo with detected capabilities and recommended tier.
    """
    info = HardwareInfo()

    # Get CPU info
    try:
        import os
        info.cpu_cores = os.cpu_count() or 1
    except Exception:
        pass

    # Get RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.ram_total_mb = mem.total // (1024 * 1024)
    except ImportError:
        pass

    # Check NVIDIA GPU
    nvidia_info = _get_nvidia_gpu_info()
    if nvidia_info:
        info.gpu_available = True
        info.gpu_name = nvidia_info[0]
        info.gpu_vendor = GPUVendor.NVIDIA
        info.vram_total_mb = nvidia_info[1]
        info.vram_free_mb = nvidia_info[2]
        info.cuda_available = True
        info.vulkan_available = True

    # Fallback to PyTorch CUDA detection
    elif HAS_TORCH and torch.cuda.is_available():
        torch_info = _get_torch_cuda_info()
        if torch_info:
            info.gpu_available = True
            info.gpu_name = torch_info[0]
            info.gpu_vendor = GPUVendor.NVIDIA
            info.vram_total_mb = torch_info[1]
            info.vram_free_mb = torch_info[2]
            info.cuda_available = True
            info.vulkan_available = True

    # Check ncnn-vulkan availability (for AMD/Intel)
    info.ncnn_available = _check_ncnn_available()

    # Determine optimal tier
    info.tier = get_hardware_tier(info)

    return info


def get_hardware_tier(info: HardwareInfo) -> HardwareTier:
    """Determine the optimal hardware tier based on detected capabilities.

    Args:
        info: Detected hardware information.

    Returns:
        Optimal hardware tier for denoising.
    """
    # CUDA path for NVIDIA
    if info.cuda_available and info.gpu_vendor == GPUVendor.NVIDIA:
        if info.vram_free_mb >= 8000:
            return HardwareTier.CUDA_8GB
        elif info.vram_free_mb >= 4000:
            return HardwareTier.CUDA_4GB
        elif info.vram_free_mb >= 2000:
            return HardwareTier.CUDA_4GB  # Smaller tile size

    # NCNN path for AMD/Intel with Vulkan
    if info.ncnn_available and info.vulkan_available:
        return HardwareTier.NCNN

    # CPU fallback
    return HardwareTier.CPU


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DenoiserConfig:
    """Unified denoiser configuration.

    Attributes:
        strength: Denoising strength from 0.0 (none) to 1.0 (maximum).
        temporal_radius: Number of frames to consider for temporal filtering.
        enable_flicker_reduction: Apply flicker reduction for film sources.
        preserve_grain: Preserve film grain character.
        half_precision: Use FP16 for reduced VRAM (neural backends).
        tile_size: Tile size for large frame processing (0 = auto).
        gpu_id: GPU device ID.
        force_backend: Force specific backend (None = auto-select).
    """

    strength: float = 0.5
    temporal_radius: int = 3
    enable_flicker_reduction: bool = True
    preserve_grain: bool = False
    half_precision: bool = True
    tile_size: int = 0  # 0 = auto
    gpu_id: int = 0
    force_backend: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be 0-1, got {self.strength}")
        if self.temporal_radius < 1:
            raise ValueError(f"temporal_radius must be >= 1, got {self.temporal_radius}")


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class DenoiseResult:
    """Result of denoising processing.

    Attributes:
        frames_processed: Number of frames successfully processed.
        frames_failed: Number of frames that failed.
        output_dir: Path to output directory.
        backend_used: Name of the backend that was used.
        processing_time_seconds: Total processing time.
        avg_noise_reduction: Estimated noise reduction achieved (0-1).
        peak_vram_mb: Peak VRAM usage (if applicable).
        warnings: Any warnings generated during processing.
    """

    frames_processed: int = 0
    frames_failed: int = 0
    output_dir: Optional[Path] = None
    backend_used: str = "unknown"
    processing_time_seconds: float = 0.0
    avg_noise_reduction: float = 0.0
    peak_vram_mb: int = 0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Backend Protocol
# =============================================================================

class DenoiserBackend(ABC):
    """Abstract base class for denoiser backends.

    All backends must implement this interface for the unified Denoiser
    to work correctly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier name."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @abstractmethod
    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        config: DenoiserConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DenoiseResult:
        """Process frames through this backend.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            config: Denoiser configuration.
            progress_callback: Optional progress callback (0-1).

        Returns:
            DenoiseResult with processing statistics.
        """
        ...


# =============================================================================
# Traditional (CPU/FFmpeg) Backend
# =============================================================================

class TraditionalDenoiser(DenoiserBackend):
    """FFmpeg-based traditional denoiser.

    Works on all systems without GPU requirements.
    Uses hqdn3d filter for combined spatial+temporal denoising.
    """

    @property
    def name(self) -> str:
        return "traditional_ffmpeg"

    def is_available(self) -> bool:
        """Check if FFmpeg is available."""
        return shutil.which("ffmpeg") is not None

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        config: DenoiserConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DenoiseResult:
        """Apply FFmpeg-based denoising."""
        result = DenoiseResult(backend_used=self.name)
        start_time = time.time()

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        # Find frames
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found for denoising")
            return result

        # Build hqdn3d filter parameters based on strength
        strength = config.strength
        spatial_luma = int(2 + strength * 4)
        spatial_chroma = int(1 + strength * 3)
        temporal_luma = int(2 + strength * 4)
        temporal_chroma = int(1 + strength * 3)

        filters = [
            f"hqdn3d={spatial_luma}:{spatial_chroma}:{temporal_luma}:{temporal_chroma}"
        ]

        # Add flicker reduction if enabled
        if config.enable_flicker_reduction:
            filters.append("deflicker=size=5:mode=am")

        filter_str = ",".join(filters)

        # Detect frame pattern
        sample = frames[0]
        ext = sample.suffix

        # Try to detect pattern
        input_pattern = str(input_dir / f"frame_%08d{ext}")
        output_pattern = str(output_dir / f"frame_%08d{ext}")

        cmd = [
            "ffmpeg", "-y",
            "-i", input_pattern,
            "-vf", filter_str,
            "-q:v", "1",
            output_pattern,
            "-hide_banner", "-loglevel", "error",
        ]

        if progress_callback:
            progress_callback(0.1)

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)

            output_frames = list(output_dir.glob(f"*{ext}"))
            result.frames_processed = len(output_frames)
            result.frames_failed = len(frames) - len(output_frames)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"FFmpeg denoising failed: {e}, falling back to frame copy")

            # Fallback: copy frames
            for frame in frames:
                try:
                    shutil.copy(frame, output_dir / frame.name)
                    result.frames_processed += 1
                except Exception:
                    result.frames_failed += 1

            result.warnings.append("FFmpeg denoising failed, frames copied without processing")

        result.processing_time_seconds = time.time() - start_time

        if progress_callback:
            progress_callback(1.0)

        return result


# =============================================================================
# Temporal Denoiser Backend (wraps existing TemporalDenoiser)
# =============================================================================

class TemporalDenoiserBackend(DenoiserBackend):
    """Temporal denoising backend wrapping the existing TemporalDenoiser.

    Provides motion-compensated temporal filtering with optical flow.
    Requires OpenCV and moderate GPU VRAM (4GB+).
    """

    def __init__(self) -> None:
        self._denoiser = None
        self._temporal_module = None

    @property
    def name(self) -> str:
        return "temporal_opencv"

    def _import_temporal(self):
        """Import temporal denoise module with fallback paths."""
        if self._temporal_module is not None:
            return self._temporal_module

        # Try absolute import first (when used as part of framewright package)
        try:
            from framewright.processors.temporal_denoise import (
                TemporalDenoiser,
                TemporalDenoiseConfig,
                FlickerMode,
            )
            # Create a simple namespace object
            class _Module:
                pass
            self._temporal_module = _Module()
            self._temporal_module.TemporalDenoiser = TemporalDenoiser
            self._temporal_module.TemporalDenoiseConfig = TemporalDenoiseConfig
            self._temporal_module.FlickerMode = FlickerMode
            return self._temporal_module
        except ImportError:
            pass

        # Try direct file import as last resort
        try:
            import importlib.util
            from pathlib import Path

            # Find the module relative to this file
            current_file = Path(__file__)
            module_path = current_file.parent.parent / "temporal_denoise.py"

            if module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "temporal_denoise", module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                class _Module:
                    pass
                self._temporal_module = _Module()
                self._temporal_module.TemporalDenoiser = module.TemporalDenoiser
                self._temporal_module.TemporalDenoiseConfig = module.TemporalDenoiseConfig
                self._temporal_module.FlickerMode = module.FlickerMode
                return self._temporal_module
        except Exception as e:
            logger.debug(f"Direct import failed: {e}")

        return None

    def is_available(self) -> bool:
        """Check if temporal denoiser is available."""
        if not HAS_OPENCV:
            return False

        return self._import_temporal() is not None

    def _get_denoiser(self, config: DenoiserConfig):
        """Get or create temporal denoiser instance."""
        module = self._import_temporal()
        if module is None:
            raise ImportError("temporal_denoise module not available")

        TemporalDenoiser = module.TemporalDenoiser
        TemporalDenoiseConfig = module.TemporalDenoiseConfig
        FlickerMode = module.FlickerMode

        temporal_config = TemporalDenoiseConfig(
            temporal_radius=config.temporal_radius,
            noise_strength=config.strength,
            enable_optical_flow=True,
            enable_flicker_reduction=config.enable_flicker_reduction,
            flicker_mode=FlickerMode.ADAPTIVE,
            preserve_edges=not config.preserve_grain,
            gpu_id=config.gpu_id,
        )

        return TemporalDenoiser(temporal_config)

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        config: DenoiserConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DenoiseResult:
        """Apply temporal denoising."""
        result = DenoiseResult(backend_used=self.name)
        start_time = time.time()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        try:
            denoiser = self._get_denoiser(config)
            temporal_result = denoiser.denoise_frames(
                input_dir, output_dir, progress_callback
            )

            result.frames_processed = temporal_result.frames_processed
            result.frames_failed = temporal_result.frames_failed
            result.avg_noise_reduction = temporal_result.avg_noise_reduction
            result.peak_vram_mb = temporal_result.peak_memory_mb

        except Exception as e:
            logger.error(f"Temporal denoising failed: {e}")
            result.warnings.append(f"Temporal denoising error: {str(e)}")
            result.frames_failed = len(list(Path(input_dir).glob("*.png")))

        result.processing_time_seconds = time.time() - start_time
        return result


# =============================================================================
# TAP Neural Denoiser Backend (wraps existing TAPDenoiser)
# =============================================================================

class TAPDenoiserBackend(DenoiserBackend):
    """TAP neural denoising backend wrapping the existing TAPDenoiser.

    Uses Restormer/NAFNet/TAP neural networks for superior denoising.
    Requires PyTorch and 8GB+ GPU VRAM.
    """

    def __init__(self) -> None:
        self._denoiser = None
        self._tap_module = None

    @property
    def name(self) -> str:
        return "tap_neural"

    def _import_tap(self):
        """Import TAP denoise module with fallback paths."""
        if self._tap_module is not None:
            return self._tap_module

        # Try absolute import first (when used as part of framewright package)
        try:
            from framewright.processors.tap_denoise import (
                TAPDenoiser,
                TAPDenoiseConfig,
                TAPModel,
            )
            # Create a simple namespace object
            class _Module:
                pass
            self._tap_module = _Module()
            self._tap_module.TAPDenoiser = TAPDenoiser
            self._tap_module.TAPDenoiseConfig = TAPDenoiseConfig
            self._tap_module.TAPModel = TAPModel
            return self._tap_module
        except ImportError:
            pass

        # Try direct file import as last resort
        try:
            import importlib.util
            from pathlib import Path

            # Find the module relative to this file
            current_file = Path(__file__)
            module_path = current_file.parent.parent / "tap_denoise.py"

            if module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "tap_denoise", module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                class _Module:
                    pass
                self._tap_module = _Module()
                self._tap_module.TAPDenoiser = module.TAPDenoiser
                self._tap_module.TAPDenoiseConfig = module.TAPDenoiseConfig
                self._tap_module.TAPModel = module.TAPModel
                return self._tap_module
        except Exception as e:
            logger.debug(f"Direct TAP import failed: {e}")

        return None

    def is_available(self) -> bool:
        """Check if TAP denoiser is available."""
        if not HAS_TORCH:
            return False

        module = self._import_tap()
        if module is None:
            return False

        try:
            config = module.TAPDenoiseConfig()
            denoiser = module.TAPDenoiser(config)
            return denoiser.is_available()
        except Exception:
            return False

    def _get_denoiser(self, config: DenoiserConfig):
        """Get or create TAP denoiser instance."""
        module = self._import_tap()
        if module is None:
            raise ImportError("tap_denoise module not available")

        TAPDenoiser = module.TAPDenoiser
        TAPDenoiseConfig = module.TAPDenoiseConfig
        TAPModel = module.TAPModel

        tap_config = TAPDenoiseConfig(
            model=TAPModel.RESTORMER,
            temporal_window=config.temporal_radius * 2 + 1,
            strength=config.strength,
            preserve_grain=config.preserve_grain,
            half_precision=config.half_precision,
            tile_size=config.tile_size if config.tile_size > 0 else 512,
            gpu_id=config.gpu_id,
        )

        return TAPDenoiser(tap_config)

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        config: DenoiserConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DenoiseResult:
        """Apply TAP neural denoising."""
        result = DenoiseResult(backend_used=self.name)
        start_time = time.time()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        try:
            denoiser = self._get_denoiser(config)
            tap_result = denoiser.denoise_frames(
                input_dir, output_dir, progress_callback
            )

            result.frames_processed = tap_result.frames_processed
            result.frames_failed = tap_result.frames_failed
            result.avg_noise_reduction = tap_result.avg_psnr_improvement / 20.0  # Normalize
            result.peak_vram_mb = tap_result.peak_vram_mb

            if tap_result.model_used:
                result.backend_used = f"tap_{tap_result.model_used}"

        except Exception as e:
            logger.error(f"TAP denoising failed: {e}")
            result.warnings.append(f"TAP denoising error: {str(e)}")
            result.frames_failed = len(list(Path(input_dir).glob("*.png")))

        result.processing_time_seconds = time.time() - start_time
        return result


# =============================================================================
# NCNN Vulkan Backend
# =============================================================================

class NCNNDenoiser(DenoiserBackend):
    """NCNN Vulkan-based denoiser for AMD/Intel GPUs.

    Uses realesrgan-ncnn-vulkan binary for GPU-accelerated denoising.
    Works with any GPU supporting Vulkan.
    """

    @property
    def name(self) -> str:
        return "ncnn_vulkan"

    def is_available(self) -> bool:
        """Check if ncnn-vulkan denoiser is available."""
        return _check_ncnn_available()

    def _find_ncnn_binary(self) -> Optional[Path]:
        """Find the ncnn-vulkan binary."""
        names = [
            "realesrgan-ncnn-vulkan",
            "realesrgan-ncnn-vulkan.exe",
        ]

        for name in names:
            path = shutil.which(name)
            if path:
                return Path(path)

        home = Path.home()
        paths = [
            home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan.exe",
            home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan",
        ]

        for p in paths:
            if p.exists():
                return p

        return None

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        config: DenoiserConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DenoiseResult:
        """Apply NCNN-based denoising.

        Note: NCNN is primarily for upscaling, but the processing
        also applies some denoising. For pure denoising, this backend
        falls back to FFmpeg denoising combined with NCNN enhancement.
        """
        result = DenoiseResult(backend_used=self.name)
        start_time = time.time()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        ncnn_path = self._find_ncnn_binary()

        if not ncnn_path:
            logger.warning("NCNN binary not found, falling back to FFmpeg")
            traditional = TraditionalDenoiser()
            return traditional.process(input_dir, output_dir, config, progress_callback)

        # For denoising-focused workflow, use FFmpeg first then NCNN for cleanup
        # This provides better denoising than NCNN alone

        frames = sorted(Path(input_dir).glob("*.png"))
        if not frames:
            frames = sorted(Path(input_dir).glob("*.jpg"))

        if not frames:
            logger.warning("No frames found")
            return result

        # Since NCNN is primarily for upscaling, we apply FFmpeg denoising
        # and just copy frames if no upscaling is requested
        traditional = TraditionalDenoiser()
        ffmpeg_result = traditional.process(
            input_dir, output_dir, config, progress_callback
        )

        result.frames_processed = ffmpeg_result.frames_processed
        result.frames_failed = ffmpeg_result.frames_failed
        result.warnings.extend(ffmpeg_result.warnings)
        result.warnings.append("NCNN backend used FFmpeg for denoising (NCNN optimized for upscaling)")

        result.processing_time_seconds = time.time() - start_time
        return result


# =============================================================================
# Main Unified Denoiser
# =============================================================================

class Denoiser:
    """Unified denoiser with automatic backend selection.

    Automatically selects the optimal backend based on hardware capabilities,
    with graceful fallback if the preferred backend fails.

    Supported backends:
    - CPU: FFmpeg-based traditional denoising
    - CUDA_4GB: Temporal denoising with optical flow (TemporalDenoiser)
    - CUDA_8GB: Neural network denoising (TAPDenoiser)
    - NCNN: Vulkan-based for AMD/Intel GPUs

    Example:
        >>> config = DenoiserConfig(strength=0.7)
        >>> hardware = detect_hardware()
        >>> denoiser = Denoiser(config, hardware)
        >>> result = denoiser.denoise(input_dir, output_dir)
    """

    BACKENDS: Dict[HardwareTier, type] = {
        HardwareTier.CPU: TraditionalDenoiser,
        HardwareTier.CUDA_4GB: TemporalDenoiserBackend,
        HardwareTier.CUDA_8GB: TAPDenoiserBackend,
        HardwareTier.NCNN: NCNNDenoiser,
    }

    # Fallback chain for each tier
    FALLBACK_CHAIN: Dict[HardwareTier, List[HardwareTier]] = {
        HardwareTier.CUDA_8GB: [HardwareTier.CUDA_4GB, HardwareTier.CPU],
        HardwareTier.CUDA_4GB: [HardwareTier.CPU],
        HardwareTier.NCNN: [HardwareTier.CPU],
        HardwareTier.CPU: [],
    }

    def __init__(
        self,
        config: Optional[DenoiserConfig] = None,
        hardware: Optional[HardwareInfo] = None,
    ) -> None:
        """Initialize the unified denoiser.

        Args:
            config: Denoiser configuration. Uses defaults if None.
            hardware: Hardware information. Auto-detected if None.
        """
        self.config = config or DenoiserConfig()
        self.hardware = hardware or detect_hardware()
        self._backend: Optional[DenoiserBackend] = None
        self._selected_tier: HardwareTier = self.hardware.tier

        # Select optimal backend
        self._select_backend()

    def _select_backend(self) -> None:
        """Select the optimal available backend."""
        # Check for forced backend
        if self.config.force_backend:
            tier_map = {
                "cpu": HardwareTier.CPU,
                "traditional": HardwareTier.CPU,
                "temporal": HardwareTier.CUDA_4GB,
                "cuda_4gb": HardwareTier.CUDA_4GB,
                "tap": HardwareTier.CUDA_8GB,
                "neural": HardwareTier.CUDA_8GB,
                "cuda_8gb": HardwareTier.CUDA_8GB,
                "ncnn": HardwareTier.NCNN,
                "vulkan": HardwareTier.NCNN,
            }

            forced_tier = tier_map.get(self.config.force_backend.lower())
            if forced_tier:
                backend_class = self.BACKENDS.get(forced_tier)
                if backend_class:
                    backend = backend_class()
                    if backend.is_available():
                        self._backend = backend
                        self._selected_tier = forced_tier
                        logger.info(f"Forced backend: {backend.name}")
                        return

        # Try recommended tier first
        tier = self.hardware.tier
        backend_class = self.BACKENDS.get(tier)

        if backend_class:
            backend = backend_class()
            if backend.is_available():
                self._backend = backend
                self._selected_tier = tier
                logger.info(f"Selected backend: {backend.name} (tier: {tier.value})")
                return

        # Try fallback chain
        fallbacks = self.FALLBACK_CHAIN.get(tier, [])
        for fallback_tier in fallbacks:
            backend_class = self.BACKENDS.get(fallback_tier)
            if backend_class:
                backend = backend_class()
                if backend.is_available():
                    self._backend = backend
                    self._selected_tier = fallback_tier
                    logger.info(
                        f"Using fallback backend: {backend.name} "
                        f"(tier: {fallback_tier.value})"
                    )
                    return

        # Ultimate fallback to CPU
        self._backend = TraditionalDenoiser()
        self._selected_tier = HardwareTier.CPU
        logger.warning("Falling back to CPU backend")

    @property
    def backend(self) -> DenoiserBackend:
        """Get the selected backend."""
        if self._backend is None:
            self._select_backend()
        return self._backend

    @property
    def selected_tier(self) -> HardwareTier:
        """Get the selected hardware tier."""
        return self._selected_tier

    def denoise(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DenoiseResult:
        """Denoise video frames.

        Args:
            input_dir: Directory containing input frames (PNG or JPG).
            output_dir: Directory for denoised output frames.
            progress_callback: Optional callback for progress updates (0-1).

        Returns:
            DenoiseResult with processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        logger.info(
            f"Denoising frames: {input_dir} -> {output_dir}, "
            f"backend: {self.backend.name}, strength: {self.config.strength}"
        )

        result = self.backend.process(
            input_dir, output_dir, self.config, progress_callback
        )

        # Add hardware info to warnings if using fallback
        if self._selected_tier != self.hardware.tier:
            result.warnings.append(
                f"Using fallback backend ({self._selected_tier.value}) "
                f"instead of optimal ({self.hardware.tier.value})"
            )

        return result

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the selected backend.

        Returns:
            Dictionary with backend information.
        """
        return {
            "name": self.backend.name,
            "tier": self._selected_tier.value,
            "hardware_tier": self.hardware.tier.value,
            "is_fallback": self._selected_tier != self.hardware.tier,
            "available_backends": [
                tier.value for tier in HardwareTier
                if self.BACKENDS.get(tier, lambda: None)().is_available()
                if self.BACKENDS.get(tier) is not None
            ],
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_denoiser(
    strength: float = 0.5,
    temporal_radius: int = 3,
    enable_flicker_reduction: bool = True,
    preserve_grain: bool = False,
    force_backend: Optional[str] = None,
) -> Denoiser:
    """Factory function to create a configured denoiser.

    Args:
        strength: Denoising strength (0-1).
        temporal_radius: Number of frames for temporal filtering.
        enable_flicker_reduction: Apply flicker reduction.
        preserve_grain: Preserve film grain character.
        force_backend: Force specific backend (None = auto-select).

    Returns:
        Configured Denoiser instance.
    """
    config = DenoiserConfig(
        strength=strength,
        temporal_radius=temporal_radius,
        enable_flicker_reduction=enable_flicker_reduction,
        preserve_grain=preserve_grain,
        force_backend=force_backend,
    )

    hardware = detect_hardware()
    return Denoiser(config, hardware)


def denoise_frames(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    strength: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> DenoiseResult:
    """Convenience function to denoise frames with auto-detection.

    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        strength: Denoising strength (0-1).
        progress_callback: Optional progress callback.

    Returns:
        DenoiseResult with processing statistics.
    """
    denoiser = create_denoiser(strength=strength)
    return denoiser.denoise(input_dir, output_dir, progress_callback)
