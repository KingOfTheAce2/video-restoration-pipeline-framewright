"""GPU Memory Optimizer for FrameWright.

Dynamically adjusts batch sizes, tile sizes, and model loading based on
available GPU VRAM. Prevents OOM errors and maximizes throughput.

Features:
- Real-time VRAM monitoring
- Dynamic batch size adjustment
- Automatic model offloading
- Tile size optimization for large frames
- Multi-GPU memory balancing

Example:
    >>> optimizer = GPUMemoryOptimizer()
    >>> batch_size = optimizer.get_optimal_batch_size(frame_size=(1920, 1080), model="hat")
    >>> with optimizer.managed_memory():
    ...     # Processing with automatic memory management
    ...     process_frames(batch_size=batch_size)
"""

import logging
import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MemoryPressure(Enum):
    """GPU memory pressure levels."""
    LOW = "low"  # < 50% used
    MODERATE = "moderate"  # 50-75% used
    HIGH = "high"  # 75-90% used
    CRITICAL = "critical"  # > 90% used


@dataclass
class GPUMemoryStats:
    """Current GPU memory statistics."""
    device_id: int = 0
    device_name: str = "Unknown"
    total_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0
    reserved_mb: float = 0.0
    utilization_percent: float = 0.0
    pressure: MemoryPressure = MemoryPressure.LOW

    @property
    def available_mb(self) -> float:
        """Available memory (free + reclaimable)."""
        return max(0, self.total_mb - self.used_mb)


@dataclass
class OptimizationConfig:
    """Configuration for memory optimization."""
    # Target memory usage (leave headroom)
    target_utilization: float = 0.85  # 85% max
    # Minimum batch size
    min_batch_size: int = 1
    # Maximum batch size
    max_batch_size: int = 32
    # Enable automatic garbage collection
    auto_gc: bool = True
    # Model memory estimates (MB per batch item at 1080p)
    model_memory_estimates: Dict[str, float] = field(default_factory=lambda: {
        "realesrgan": 800,
        "hat": 2000,
        "vrt": 1500,
        "basicvsr": 1200,
        "diffusion": 3000,
        "face_restore": 400,
        "colorization": 1000,
        "svd": 8000,  # Stable Video Diffusion
    })
    # Tile sizes for different VRAM tiers
    tile_sizes: Dict[str, int] = field(default_factory=lambda: {
        "32gb+": 0,  # No tiling
        "24gb": 0,  # No tiling
        "16gb": 512,
        "12gb": 384,
        "8gb": 256,
        "6gb": 192,
        "4gb": 128,
    })


class GPUMemoryOptimizer:
    """Optimizes GPU memory usage for video processing.

    Monitors VRAM usage and dynamically adjusts processing parameters
    to prevent out-of-memory errors while maximizing throughput.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self._device_count = 0
        self._stats_cache: Dict[int, GPUMemoryStats] = {}
        self._init_devices()

    def _init_devices(self) -> None:
        """Initialize GPU devices."""
        if not HAS_TORCH:
            return

        if torch.cuda.is_available():
            self._device_count = torch.cuda.device_count()
            logger.info(f"Found {self._device_count} CUDA device(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device_count = 1
            logger.info("Found MPS device (Apple Silicon)")

    def get_memory_stats(self, device_id: int = 0) -> GPUMemoryStats:
        """Get current memory statistics for a GPU.

        Args:
            device_id: GPU device ID

        Returns:
            GPUMemoryStats
        """
        stats = GPUMemoryStats(device_id=device_id)

        if not HAS_TORCH or not torch.cuda.is_available():
            return stats

        try:
            props = torch.cuda.get_device_properties(device_id)
            stats.device_name = props.name
            stats.total_mb = props.total_memory / (1024 * 1024)

            # Get current usage
            stats.used_mb = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
            stats.reserved_mb = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
            stats.free_mb = stats.total_mb - stats.reserved_mb

            stats.utilization_percent = (stats.used_mb / stats.total_mb) * 100

            # Determine pressure level
            if stats.utilization_percent < 50:
                stats.pressure = MemoryPressure.LOW
            elif stats.utilization_percent < 75:
                stats.pressure = MemoryPressure.MODERATE
            elif stats.utilization_percent < 90:
                stats.pressure = MemoryPressure.HIGH
            else:
                stats.pressure = MemoryPressure.CRITICAL

        except Exception as e:
            logger.debug(f"Failed to get GPU stats: {e}")

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
            model: Model name
            scale_factor: Upscaling factor
            device_id: GPU device ID

        Returns:
            Optimal batch size
        """
        stats = self.get_memory_stats(device_id)

        # Get base memory estimate for model
        base_memory = self.config.model_memory_estimates.get(model.lower(), 1000)

        # Adjust for frame size (relative to 1080p)
        width, height = frame_size
        size_factor = (width * height) / (1920 * 1080)

        # Adjust for scale factor
        scale_memory = base_memory * size_factor * (scale_factor ** 2)

        # Available memory (with safety margin)
        available = stats.available_mb * self.config.target_utilization

        # Calculate batch size
        if scale_memory <= 0:
            return self.config.max_batch_size

        batch_size = int(available / scale_memory)

        # Clamp to valid range
        batch_size = max(self.config.min_batch_size,
                        min(self.config.max_batch_size, batch_size))

        logger.debug(
            f"Optimal batch size for {model} at {width}x{height}: {batch_size} "
            f"(available: {available:.0f}MB, per-frame: {scale_memory:.0f}MB)"
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
            device_id: GPU device ID

        Returns:
            Tile size in pixels, or None for no tiling
        """
        stats = self.get_memory_stats(device_id)
        vram_gb = stats.total_mb / 1024

        # Select tile size based on VRAM tier
        if vram_gb >= 32:
            return None  # No tiling needed
        elif vram_gb >= 24:
            return None
        elif vram_gb >= 16:
            return self.config.tile_sizes["16gb"]
        elif vram_gb >= 12:
            return self.config.tile_sizes["12gb"]
        elif vram_gb >= 8:
            return self.config.tile_sizes["8gb"]
        elif vram_gb >= 6:
            return self.config.tile_sizes["6gb"]
        else:
            return self.config.tile_sizes["4gb"]

    def should_offload_model(
        self,
        model_name: str,
        device_id: int = 0,
    ) -> bool:
        """Check if a model should be offloaded to save memory.

        Args:
            model_name: Name of the model
            device_id: GPU device ID

        Returns:
            True if model should be offloaded
        """
        stats = self.get_memory_stats(device_id)
        return stats.pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL)

    def clear_memory(self, device_id: Optional[int] = None) -> None:
        """Clear GPU memory caches.

        Args:
            device_id: Specific device to clear, or None for all
        """
        if not HAS_TORCH:
            return

        if self.config.auto_gc:
            gc.collect()

        if torch.cuda.is_available():
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

            logger.debug("Cleared GPU memory cache")

    @contextmanager
    def managed_memory(self, device_id: int = 0):
        """Context manager for automatic memory management.

        Clears cache before and after processing, monitors for OOM.

        Args:
            device_id: GPU device ID

        Yields:
            Memory stats at start
        """
        # Clear before
        self.clear_memory(device_id)
        stats_before = self.get_memory_stats(device_id)

        try:
            yield stats_before
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM detected, clearing memory and retrying...")
                self.clear_memory(device_id)
                raise MemoryError(f"GPU out of memory: {e}")
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
            models: List of models to use
            scale_factor: Upscaling factor
            device_id: GPU device ID

        Returns:
            Configuration dict with optimal settings
        """
        stats = self.get_memory_stats(device_id)

        # Calculate requirements for all models
        total_memory_per_frame = sum(
            self.config.model_memory_estimates.get(m.lower(), 1000)
            for m in models
        )

        # Adjust for frame size
        width, height = frame_size
        size_factor = (width * height) / (1920 * 1080)
        total_memory_per_frame *= size_factor * (scale_factor ** 2)

        # Determine optimal settings
        available = stats.available_mb * self.config.target_utilization

        config = {
            "batch_size": max(1, int(available / total_memory_per_frame)),
            "tile_size": self.get_optimal_tile_size(frame_size, device_id),
            "half_precision": stats.total_mb < 16000,  # FP16 for < 16GB
            "sequential_models": stats.pressure != MemoryPressure.LOW,
            "offload_between_models": stats.pressure == MemoryPressure.CRITICAL,
            "device_id": device_id,
            "available_vram_mb": stats.available_mb,
            "total_vram_mb": stats.total_mb,
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
            base = self.config.model_memory_estimates.get(model.lower(), 1000)
            total += base * size_factor * (scale_factor ** 2)

        return total * batch_size


# Global optimizer instance
_optimizer: Optional[GPUMemoryOptimizer] = None


def get_optimizer() -> GPUMemoryOptimizer:
    """Get or create global optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = GPUMemoryOptimizer()
    return _optimizer


def get_optimal_batch_size(
    frame_size: Tuple[int, int],
    model: str = "realesrgan",
    scale_factor: int = 2,
) -> int:
    """Convenience function to get optimal batch size."""
    return get_optimizer().get_optimal_batch_size(frame_size, model, scale_factor)


def get_optimal_tile_size(frame_size: Tuple[int, int]) -> Optional[int]:
    """Convenience function to get optimal tile size."""
    return get_optimizer().get_optimal_tile_size(frame_size)


def clear_gpu_memory() -> None:
    """Convenience function to clear GPU memory."""
    get_optimizer().clear_memory()
