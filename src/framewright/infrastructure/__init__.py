"""Unified Infrastructure Layer for FrameWright.

This module provides hardware abstraction for video restoration across all platforms:
- CPU only (no GPU)
- NVIDIA (CUDA)
- AMD (ROCm/Vulkan)
- Intel (oneAPI/Vulkan)
- Apple Silicon (Metal/CoreML)

Additionally, this module provides:
- Model management infrastructure with registry, downloading, and caching
- Memory leak detection and profiling for long-running jobs
- GPU memory management and eviction policies

The infrastructure layer automatically detects hardware, selects optimal backends,
and manages resources (memory, compute) transparently.

Example:
    >>> from framewright.infrastructure import get_hardware_info, get_backend
    >>> info = get_hardware_info()
    >>> print(f"Hardware tier: {info.tier}")
    >>> backend = get_backend()
    >>> backend.initialize()

Model Management Example:
    >>> from framewright.infrastructure.models import get_model_manager, ModelCategory
    >>> manager = get_model_manager()
    >>> path = manager.get_model("realesrgan-x4plus")

Memory Profiling Example:
    >>> from framewright.infrastructure.memory import MemoryProfiler, create_profiler
    >>> profiler = create_profiler(auto_gc=True, warning_threshold_mb=500)
    >>> profiler.start_session("restoration_job")
    >>> with profiler.profile_stage("upscale"):
    ...     upscale_frames(frames)
    >>> report = profiler.end_session()
    >>> print(report.format_summary())

Cloud Rendering:
    For cloud GPU offloading, use the framewright.cloud module instead:
    >>> from framewright.cloud import RunPodProvider, VastAIProvider
"""

from .gpu import (
    # Core types
    HardwareInfo,
    HardwareTier,
    GPUVendor,
    BackendType,
    # Detection functions
    get_hardware_info,
    detect_hardware,
    get_hardware_tier,
    # Backend management
    get_backend,
    get_memory_manager,
    # Convenience functions
    get_optimal_device,
    get_available_backends,
    is_gpu_available,
    get_vram_mb,
)

# Cloud infrastructure removed - use framewright.cloud instead
# from .cloud import ...

from .models import (
    # Registry
    Backend,
    ModelCategory,
    ModelInfo,
    ModelRegistry,
    get_builtin_registry,
    get_default_registry,
    # Downloader
    ChecksumError,
    DownloadError,
    DownloadProgress,
    DownloadStatus,
    ModelDownloader,
    ProgressCallback,
    create_progress_bar_callback,
    create_simple_progress_callback,
    # Manager
    ModelManager,
    ModelNotFoundError,
    ModelStatus,
    get_model_manager,
    reset_model_manager,
)

from .cache import (
    # Eviction policies
    EvictionPolicy,
    EvictionCandidate,
    LRUEviction,
    LFUEviction,
    FIFOEviction,
    SizeAwareEviction,
    CompositeEviction,
    create_eviction_policy,
    # Frame cache
    FrameCacheConfig,
    FrameCache,
    DiskFrameCache,
    # Model cache
    ModelCacheConfig,
    ModelCache,
    ModelPriority,
    get_model_cache,
    clear_model_cache,
)

from .memory import (
    # Core classes
    MemoryProfiler,
    LeakDetector,
    PipelineMemoryHooks,
    SessionReport,
    # Data classes
    MemorySnapshot,
    StageMemoryProfile,
    MemoryWarning,
    LeakReport,
    ProfilerConfig,
    # Enums
    MemoryWarningLevel,
    LeakSeverity,
    # Factory functions
    create_profiler,
    quick_memory_check,
    force_cleanup,
)

__all__ = [
    # GPU Types
    "HardwareInfo",
    "HardwareTier",
    "GPUVendor",
    "BackendType",
    # GPU Detection
    "get_hardware_info",
    "detect_hardware",
    "get_hardware_tier",
    # GPU Backend
    "get_backend",
    "get_memory_manager",
    # GPU Convenience
    "get_optimal_device",
    "get_available_backends",
    "is_gpu_available",
    "get_vram_mb",
    # Model Registry
    "Backend",
    "ModelCategory",
    "ModelInfo",
    "ModelRegistry",
    "get_builtin_registry",
    "get_default_registry",
    # Model Downloader
    "ChecksumError",
    "DownloadError",
    "DownloadProgress",
    "DownloadStatus",
    "ModelDownloader",
    "ProgressCallback",
    "create_progress_bar_callback",
    "create_simple_progress_callback",
    # Model Manager
    "ModelManager",
    "ModelNotFoundError",
    "ModelStatus",
    "get_model_manager",
    "reset_model_manager",
    # Cache - Eviction policies
    "EvictionPolicy",
    "EvictionCandidate",
    "LRUEviction",
    "LFUEviction",
    "FIFOEviction",
    "SizeAwareEviction",
    "CompositeEviction",
    "create_eviction_policy",
    # Cache - Frame cache
    "FrameCacheConfig",
    "FrameCache",
    "DiskFrameCache",
    # Cache - Model cache
    "ModelCacheConfig",
    "ModelCache",
    "ModelPriority",
    "get_model_cache",
    "clear_model_cache",
    # Memory - Core classes
    "MemoryProfiler",
    "LeakDetector",
    "PipelineMemoryHooks",
    "SessionReport",
    # Memory - Data classes
    "MemorySnapshot",
    "StageMemoryProfile",
    "MemoryWarning",
    "LeakReport",
    "ProfilerConfig",
    # Memory - Enums
    "MemoryWarningLevel",
    "LeakSeverity",
    # Memory - Factory functions
    "create_profiler",
    "quick_memory_check",
    "force_cleanup",
]
