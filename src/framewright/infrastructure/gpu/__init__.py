"""Unified GPU Interface for FrameWright.

Provides hardware-agnostic GPU access across all supported platforms:
- NVIDIA (CUDA/TensorRT)
- AMD (ROCm/Vulkan/DirectML)
- Intel (oneAPI/OpenVINO/Vulkan/DirectML)
- Apple Silicon (Metal/CoreML)
- CPU fallback

The module automatically detects available hardware, selects the best backend,
and provides a consistent interface for memory management and compute operations.

Multi-GPU support is provided via the GPUDistributor and MultiGPUProcessor classes,
which handle work distribution across multiple GPUs of any vendor type.

Example:
    >>> from framewright.infrastructure.gpu import get_hardware_info
    >>> info = get_hardware_info()
    >>> print(f"Tier: {info.tier}, VRAM: {info.total_vram_mb}MB")
    >>> print(f"Available backends: {info.available_backends}")

Multi-GPU Example:
    >>> from framewright.infrastructure.gpu import MultiGPUProcessor
    >>> processor = MultiGPUProcessor()
    >>> processor.initialize()
    >>> results = processor.process_frames(frames, my_process_func)
"""

from .detector import (
    # Core types
    HardwareInfo,
    HardwareTier,
    GPUVendor,
    BackendType,
    DeviceInfo,
    # Detection functions
    detect_hardware,
    get_hardware_info,
    get_hardware_tier,
    # Convenience
    get_optimal_device,
    get_available_backends,
    is_gpu_available,
    get_vram_mb,
)

from .memory import (
    MemoryManager,
    MemoryPressure,
    MemoryStats,
    get_memory_manager,
)

from .backends.base import (
    Backend,
    get_backend,
)

from .distributor import (
    DistributionStrategy,
    GPUStats,
    DistributionPlan,
    ProcessingResult,
    GPUDistributor,
    MultiGPUProcessor,
    detect_multi_gpu_support,
    get_optimal_gpu,
    estimate_multi_gpu_speedup,
)

__all__ = [
    # Types from detector
    "HardwareInfo",
    "HardwareTier",
    "GPUVendor",
    "BackendType",
    "DeviceInfo",
    # Detection functions
    "detect_hardware",
    "get_hardware_info",
    "get_hardware_tier",
    "get_optimal_device",
    "get_available_backends",
    "is_gpu_available",
    "get_vram_mb",
    # Memory management
    "MemoryManager",
    "MemoryPressure",
    "MemoryStats",
    "get_memory_manager",
    # Backend
    "Backend",
    "get_backend",
    # Multi-GPU distribution
    "DistributionStrategy",
    "GPUStats",
    "DistributionPlan",
    "ProcessingResult",
    "GPUDistributor",
    "MultiGPUProcessor",
    "detect_multi_gpu_support",
    "get_optimal_gpu",
    "estimate_multi_gpu_speedup",
]
