"""Caching infrastructure for FrameWright video restoration.

This module provides high-performance caching solutions for both frames
and AI models used in video restoration pipelines.

Components:
    - FrameCache: Memory cache for video frames with LRU eviction
    - DiskFrameCache: Persistent disk-based frame cache
    - ModelCache: GPU model cache with VRAM management
    - Eviction policies: LRU, LFU, FIFO, Size-aware, Composite

Example:
    >>> from framewright.infrastructure.cache import (
    ...     FrameCache, FrameCacheConfig,
    ...     ModelCache, ModelCacheConfig,
    ... )
    >>>
    >>> # Set up frame caching
    >>> frame_config = FrameCacheConfig(max_memory_mb=2048)
    >>> frame_cache = FrameCache(frame_config)
    >>>
    >>> # Cache processed frames
    >>> frame_cache.put(0, processed_frame)
    >>> cached = frame_cache.get(0)
    >>>
    >>> # Set up model caching
    >>> model_config = ModelCacheConfig(max_models=5, max_vram_mb=8000)
    >>> model_cache = ModelCache(model_config)
    >>>
    >>> # Load models with automatic caching
    >>> model = model_cache.get_model("realesrgan-x4plus")
"""

from .eviction import (
    # Base classes
    EvictionPolicy,
    EvictionCandidate,
    EvictionResult,
    EvictionStats,
    # Policies
    LRUEviction,
    LFUEviction,
    FIFOEviction,
    SizeAwareEviction,
    TTLEviction,
    CompositeEviction,
    AdaptiveEviction,
    # Factory functions
    create_eviction_policy,
    create_composite_policy,
)

from .frame_cache import (
    # Configuration
    FrameCacheConfig,
    CachedFrame,
    FrameCacheStats,
    # Caches
    FrameCache,
    DiskFrameCache,
    MemoryMappedStorage,
)

from .model_cache import (
    # Configuration
    ModelCacheConfig,
    ModelPriority,
    CachedModel,
    ModelCacheStats,
    # Loaders
    ModelLoader,
    PyTorchModelLoader,
    ONNXModelLoader,
    SafetensorsModelLoader,
    # Cache
    ModelCache,
    # Global access
    get_model_cache,
    clear_model_cache,
)


__all__ = [
    # Eviction policies
    "EvictionPolicy",
    "EvictionCandidate",
    "EvictionResult",
    "EvictionStats",
    "LRUEviction",
    "LFUEviction",
    "FIFOEviction",
    "SizeAwareEviction",
    "TTLEviction",
    "CompositeEviction",
    "AdaptiveEviction",
    "create_eviction_policy",
    "create_composite_policy",
    # Frame cache
    "FrameCacheConfig",
    "CachedFrame",
    "FrameCacheStats",
    "FrameCache",
    "DiskFrameCache",
    "MemoryMappedStorage",
    # Model cache
    "ModelCacheConfig",
    "ModelPriority",
    "CachedModel",
    "ModelCacheStats",
    "ModelLoader",
    "PyTorchModelLoader",
    "ONNXModelLoader",
    "SafetensorsModelLoader",
    "ModelCache",
    "get_model_cache",
    "clear_model_cache",
]
