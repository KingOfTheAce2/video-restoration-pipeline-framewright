"""Memory Management for FrameWright Infrastructure.

Provides tier-aware memory management with:
- Real-time VRAM monitoring
- Dynamic batch size adjustment
- Automatic model offloading
- Memory pressure detection and response
- Multi-GPU memory coordination

This module wraps and extends the existing gpu_memory_optimizer.py functionality.
"""

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detector import (
    BackendType,
    DeviceInfo,
    HardwareInfo,
    HardwareTier,
    get_hardware_info,
)

logger = logging.getLogger(__name__)

# Lazy imports
_torch = None
_torch_checked = False


def _get_torch():
    """Lazy load torch to avoid import overhead."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch


class MemoryPressure(Enum):
    """Memory pressure levels for adaptive processing."""
    LOW = "low"           # < 50% used - can increase batch size
    MODERATE = "moderate" # 50-75% used - normal operation
    HIGH = "high"         # 75-90% used - reduce batch size
    CRITICAL = "critical" # > 90% used - minimal batching, offload models


@dataclass
class MemoryStats:
    """Current memory statistics for a device."""
    device_id: int = 0
    device_name: str = "Unknown"
    total_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0
    reserved_mb: float = 0.0
    peak_mb: float = 0.0
    pressure: MemoryPressure = MemoryPressure.LOW

    @property
    def utilization_percent(self) -> float:
        """Calculate utilization percentage."""
        if self.total_mb == 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100

    @property
    def available_mb(self) -> float:
        """Available memory (free + reclaimable)."""
        return max(0, self.total_mb - self.used_mb)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    # Target memory utilization (leave headroom for spikes)
    target_utilization: float = 0.85

    # Batch size limits
    min_batch_size: int = 1
    max_batch_size: int = 32

    # Enable automatic garbage collection
    auto_gc: bool = True

    # Model memory estimates (MB per batch item at 1080p)
    model_memory_estimates: Dict[str, float] = field(default_factory=lambda: {
        # Upscalers
        "realesrgan": 800,
        "realesrgan-x4plus": 800,
        "realesrgan-x2plus": 500,
        "hat": 2000,
        "swinir": 1500,
        "esrgan": 600,

        # Video models
        "vrt": 1500,
        "basicvsr": 1200,
        "basicvsr++": 1400,
        "rife": 400,

        # Diffusion
        "diffusion": 3000,
        "stable-diffusion": 4000,
        "svd": 8000,  # Stable Video Diffusion

        # Restoration
        "face_restore": 400,
        "gfpgan": 500,
        "codeformer": 600,
        "colorization": 1000,
        "deoldify": 800,

        # Default
        "default": 1000,
    })

    # Tile sizes for different VRAM tiers (0 = no tiling)
    tier_tile_sizes: Dict[HardwareTier, int] = field(default_factory=lambda: {
        HardwareTier.VRAM_24GB_PLUS: 0,
        HardwareTier.VRAM_16GB_PLUS: 512,
        HardwareTier.VRAM_12GB: 384,
        HardwareTier.VRAM_8GB: 256,
        HardwareTier.VRAM_4GB: 192,
        HardwareTier.CPU_ONLY: 128,
        HardwareTier.APPLE_SILICON: 384,
    })

    # Batch sizes for different VRAM tiers
    tier_batch_sizes: Dict[HardwareTier, int] = field(default_factory=lambda: {
        HardwareTier.VRAM_24GB_PLUS: 16,
        HardwareTier.VRAM_16GB_PLUS: 8,
        HardwareTier.VRAM_12GB: 4,
        HardwareTier.VRAM_8GB: 2,
        HardwareTier.VRAM_4GB: 1,
        HardwareTier.CPU_ONLY: 1,
        HardwareTier.APPLE_SILICON: 4,
    })


class MemoryManager:
    """Unified memory manager for GPU/CPU processing.

    Provides tier-aware memory management that adapts to available
    hardware and current memory pressure.

    Example:
        >>> manager = MemoryManager()
        >>> batch_size = manager.get_optimal_batch_size(
        ...     frame_size=(1920, 1080),
        ...     model="realesrgan"
        ... )
        >>> with manager.managed_memory():
        ...     process_frames(batch_size=batch_size)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        hardware_info: Optional[HardwareInfo] = None,
    ):
        """Initialize memory manager.

        Args:
            config: Memory configuration (uses defaults if None)
            hardware_info: Hardware info (auto-detected if None)
        """
        self.config = config or MemoryConfig()
        self._hardware = hardware_info or get_hardware_info()
        self._stats_cache: Dict[int, MemoryStats] = {}
        self._peak_usage: Dict[int, float] = {}
        self._lock = threading.Lock()

        # Callbacks for memory pressure events
        self._pressure_callbacks: List[Callable[[MemoryPressure], None]] = []

        logger.debug(
            f"MemoryManager initialized: tier={self._hardware.tier.value}, "
            f"devices={self._hardware.device_count}"
        )

    @property
    def hardware_info(self) -> HardwareInfo:
        """Get hardware information."""
        return self._hardware

    @property
    def tier(self) -> HardwareTier:
        """Get hardware tier."""
        return self._hardware.tier

    def get_memory_stats(self, device_id: int = 0) -> MemoryStats:
        """Get current memory statistics for a device.

        Args:
            device_id: Device index (0 for primary)

        Returns:
            MemoryStats with current memory information
        """
        stats = MemoryStats(device_id=device_id)

        torch = _get_torch()
        if torch is None:
            return stats

        try:
            if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                # NVIDIA CUDA
                props = torch.cuda.get_device_properties(device_id)
                stats.device_name = props.name
                stats.total_mb = props.total_memory / (1024 * 1024)
                stats.used_mb = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
                stats.reserved_mb = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
                stats.free_mb = stats.total_mb - stats.reserved_mb

                # Track peak
                with self._lock:
                    peak = self._peak_usage.get(device_id, 0)
                    if stats.used_mb > peak:
                        self._peak_usage[device_id] = stats.used_mb
                    stats.peak_mb = self._peak_usage.get(device_id, stats.used_mb)

            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Apple Silicon MPS
                stats.device_name = "Apple Silicon"
                if self._hardware.primary_device:
                    stats.total_mb = self._hardware.primary_device.total_memory_mb
                    stats.free_mb = self._hardware.primary_device.free_memory_mb
                    stats.used_mb = stats.total_mb - stats.free_mb

            # Determine pressure level
            utilization = stats.utilization_percent
            if utilization < 50:
                stats.pressure = MemoryPressure.LOW
            elif utilization < 75:
                stats.pressure = MemoryPressure.MODERATE
            elif utilization < 90:
                stats.pressure = MemoryPressure.HIGH
            else:
                stats.pressure = MemoryPressure.CRITICAL

        except Exception as e:
            logger.debug(f"Failed to get memory stats: {e}")

        with self._lock:
            self._stats_cache[device_id] = stats

        return stats

    def get_optimal_batch_size(
        self,
        frame_size: Tuple[int, int],
        model: str = "realesrgan",
        scale_factor: int = 2,
        device_id: int = 0,
    ) -> int:
        """Calculate optimal batch size based on available memory.

        Args:
            frame_size: Frame dimensions (width, height)
            model: Model name for memory estimation
            scale_factor: Upscaling factor (affects output memory)
            device_id: Target device

        Returns:
            Optimal batch size (>= 1)
        """
        # Get tier-based default
        tier_batch = self.config.tier_batch_sizes.get(self.tier, 1)

        # No GPU or CPU-only tier
        if not self._hardware.has_gpu or self.tier == HardwareTier.CPU_ONLY:
            return min(tier_batch, self.config.min_batch_size)

        # Get current memory stats
        stats = self.get_memory_stats(device_id)

        # Get model memory estimate
        model_key = model.lower()
        base_memory = self.config.model_memory_estimates.get(
            model_key,
            self.config.model_memory_estimates["default"]
        )

        # Adjust for frame size (relative to 1080p baseline)
        width, height = frame_size
        size_factor = (width * height) / (1920 * 1080)

        # Adjust for scale factor (output uses more memory)
        scale_memory = base_memory * size_factor * (scale_factor ** 2)

        # Calculate available memory with safety margin
        available = stats.available_mb * self.config.target_utilization

        # Calculate batch size
        if scale_memory <= 0:
            batch_size = self.config.max_batch_size
        else:
            batch_size = int(available / scale_memory)

        # Clamp to valid range
        batch_size = max(self.config.min_batch_size, batch_size)
        batch_size = min(self.config.max_batch_size, batch_size)
        batch_size = min(tier_batch, batch_size)  # Don't exceed tier default

        # Reduce if under memory pressure
        if stats.pressure == MemoryPressure.HIGH:
            batch_size = max(1, batch_size // 2)
        elif stats.pressure == MemoryPressure.CRITICAL:
            batch_size = 1

        logger.debug(
            f"Optimal batch size for {model} at {width}x{height}: {batch_size} "
            f"(available: {available:.0f}MB, per-frame: {scale_memory:.0f}MB, "
            f"pressure: {stats.pressure.value})"
        )

        return batch_size

    def get_optimal_tile_size(
        self,
        frame_size: Tuple[int, int],
        device_id: int = 0,
    ) -> Optional[int]:
        """Get optimal tile size for processing large frames.

        Args:
            frame_size: Frame dimensions (width, height)
            device_id: Target device

        Returns:
            Tile size in pixels, or None/0 for no tiling
        """
        # Get tier-based default
        tile_size = self.config.tier_tile_sizes.get(self.tier, 256)

        if tile_size == 0:
            return None  # No tiling for high-end tiers

        # Adjust based on current pressure
        stats = self.get_memory_stats(device_id)

        if stats.pressure == MemoryPressure.HIGH:
            tile_size = max(128, tile_size - 64)
        elif stats.pressure == MemoryPressure.CRITICAL:
            tile_size = max(128, tile_size // 2)

        # Ensure tile is not larger than frame
        width, height = frame_size
        tile_size = min(tile_size, min(width, height))

        # Round to 32 for GPU alignment
        tile_size = (tile_size // 32) * 32
        tile_size = max(128, tile_size)

        return tile_size

    def should_use_fp16(self) -> bool:
        """Check if FP16 (half precision) should be used.

        Returns:
            True if FP16 is recommended for memory savings
        """
        # FP16 for limited VRAM or high pressure
        if self.tier in (HardwareTier.VRAM_4GB, HardwareTier.VRAM_8GB):
            return True

        if self._hardware.primary_device:
            return self._hardware.primary_device.supports_fp16

        return False

    def should_offload_model(
        self,
        model_name: str,
        device_id: int = 0,
    ) -> bool:
        """Check if a model should be offloaded to save memory.

        Args:
            model_name: Name of the model
            device_id: Device to check

        Returns:
            True if model should be offloaded to CPU
        """
        stats = self.get_memory_stats(device_id)
        return stats.pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL)

    def clear_memory(self, device_id: Optional[int] = None) -> None:
        """Clear GPU memory caches.

        Args:
            device_id: Specific device to clear (None for all)
        """
        if self.config.auto_gc:
            gc.collect()

        torch = _get_torch()
        if torch is None:
            return

        try:
            if torch.cuda.is_available():
                if device_id is not None:
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                logger.debug("Cleared CUDA memory cache")

            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have empty_cache, but gc helps
                gc.collect()
                logger.debug("Cleared MPS memory (via GC)")

        except Exception as e:
            logger.debug(f"Failed to clear memory: {e}")

    @contextmanager
    def managed_memory(self, device_id: int = 0):
        """Context manager for automatic memory management.

        Clears cache before and after processing, handles OOM errors.

        Args:
            device_id: Device to manage

        Yields:
            MemoryStats at context entry

        Raises:
            MemoryError: If GPU runs out of memory
        """
        # Clear before
        self.clear_memory(device_id)
        stats_before = self.get_memory_stats(device_id)

        try:
            yield stats_before
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "oom" in error_str:
                logger.warning("OOM detected, clearing memory...")
                self.clear_memory(device_id)
                raise MemoryError(f"GPU out of memory: {e}") from e
            raise
        finally:
            # Clear after
            self.clear_memory(device_id)

    def get_processing_config(
        self,
        frame_size: Tuple[int, int],
        models: List[str],
        scale_factor: int = 2,
        device_id: int = 0,
    ) -> Dict[str, Any]:
        """Get optimized processing configuration.

        Args:
            frame_size: Frame dimensions
            models: List of models to be used
            scale_factor: Upscaling factor
            device_id: Target device

        Returns:
            Configuration dict with optimal settings
        """
        stats = self.get_memory_stats(device_id)

        # Calculate combined memory requirement
        total_memory_per_frame = sum(
            self.config.model_memory_estimates.get(
                m.lower(),
                self.config.model_memory_estimates["default"]
            )
            for m in models
        )

        # Adjust for frame size
        width, height = frame_size
        size_factor = (width * height) / (1920 * 1080)
        total_memory_per_frame *= size_factor * (scale_factor ** 2)

        available = stats.available_mb * self.config.target_utilization

        config = {
            "batch_size": max(1, int(available / total_memory_per_frame)),
            "tile_size": self.get_optimal_tile_size(frame_size, device_id),
            "half_precision": self.should_use_fp16(),
            "sequential_models": stats.pressure != MemoryPressure.LOW,
            "offload_between_models": stats.pressure == MemoryPressure.CRITICAL,
            "device_id": device_id,
            "available_vram_mb": stats.available_mb,
            "total_vram_mb": stats.total_mb,
            "pressure": stats.pressure.value,
            "tier": self.tier.value,
        }

        # Clamp batch size
        config["batch_size"] = max(
            self.config.min_batch_size,
            min(self.config.max_batch_size, config["batch_size"])
        )

        return config

    def estimate_memory_usage(
        self,
        frame_size: Tuple[int, int],
        batch_size: int,
        models: List[str],
        scale_factor: int = 2,
    ) -> float:
        """Estimate memory usage for given configuration.

        Args:
            frame_size: Frame dimensions
            batch_size: Batch size
            models: Models to use
            scale_factor: Upscaling factor

        Returns:
            Estimated memory usage in MB
        """
        width, height = frame_size
        size_factor = (width * height) / (1920 * 1080)

        total = 0
        for model in models:
            base = self.config.model_memory_estimates.get(
                model.lower(),
                self.config.model_memory_estimates["default"]
            )
            total += base * size_factor * (scale_factor ** 2)

        return total * batch_size

    def register_pressure_callback(
        self,
        callback: Callable[[MemoryPressure], None]
    ) -> None:
        """Register callback for memory pressure changes.

        Args:
            callback: Function to call when pressure changes
        """
        self._pressure_callbacks.append(callback)

    def get_all_device_stats(self) -> List[MemoryStats]:
        """Get memory stats for all devices.

        Returns:
            List of MemoryStats for each device
        """
        stats = []
        for device in self._hardware.all_devices:
            stats.append(self.get_memory_stats(device.index))

        if not stats:
            stats.append(self.get_memory_stats(0))

        return stats


# =============================================================================
# Global instance and convenience functions
# =============================================================================

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance.

    Returns:
        Global MemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_optimal_batch_size(
    frame_size: Tuple[int, int],
    model: str = "realesrgan",
    scale_factor: int = 2,
) -> int:
    """Convenience function to get optimal batch size.

    Args:
        frame_size: Frame dimensions
        model: Model name
        scale_factor: Upscaling factor

    Returns:
        Optimal batch size
    """
    return get_memory_manager().get_optimal_batch_size(
        frame_size, model, scale_factor
    )


def get_optimal_tile_size(frame_size: Tuple[int, int]) -> Optional[int]:
    """Convenience function to get optimal tile size.

    Args:
        frame_size: Frame dimensions

    Returns:
        Tile size or None for no tiling
    """
    return get_memory_manager().get_optimal_tile_size(frame_size)


def clear_gpu_memory() -> None:
    """Convenience function to clear GPU memory."""
    get_memory_manager().clear_memory()


# =============================================================================
# Integration with existing gpu_memory_optimizer.py
# =============================================================================

def get_legacy_optimizer():
    """Get optimizer instance compatible with existing code.

    This provides backward compatibility with code using the
    GPUMemoryOptimizer from utils/gpu_memory_optimizer.py.
    """
    # Import existing optimizer
    from ...utils.gpu_memory_optimizer import GPUMemoryOptimizer, OptimizationConfig

    # Create with our config mapped
    manager = get_memory_manager()

    config = OptimizationConfig(
        target_utilization=manager.config.target_utilization,
        min_batch_size=manager.config.min_batch_size,
        max_batch_size=manager.config.max_batch_size,
        auto_gc=manager.config.auto_gc,
        model_memory_estimates=dict(manager.config.model_memory_estimates),
    )

    return GPUMemoryOptimizer(config)
