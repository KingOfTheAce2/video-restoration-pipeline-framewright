"""Model caching infrastructure for video restoration.

Provides intelligent model caching with automatic VRAM management,
LRU eviction, and integration with the model manager.
"""

import gc
import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from .eviction import (
        EvictionCandidate,
        EvictionPolicy,
        LRUEviction,
        create_eviction_policy,
    )
except ImportError:
    from eviction import (
        EvictionCandidate,
        EvictionPolicy,
        LRUEviction,
        create_eviction_policy,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ModelPriority(Enum):
    """Priority levels for model caching."""
    LOW = 0         # Evict first
    NORMAL = 1      # Standard priority
    HIGH = 2        # Keep loaded longer
    CRITICAL = 3    # Never evict automatically


@dataclass
class ModelCacheConfig:
    """Configuration for model cache.

    Attributes:
        max_models: Maximum number of models to keep loaded
        max_vram_mb: Maximum VRAM usage in megabytes (0 = auto-detect)
        preload_models: List of model names to preload
        auto_unload: Automatically unload models when VRAM pressure
        vram_headroom_mb: VRAM to keep free for processing
        eviction_policy: Eviction policy ("lru", "lfu", "size")
        model_dir: Directory for model storage
        enable_half_precision: Use FP16 to save VRAM
    """
    max_models: int = 10
    max_vram_mb: int = 0  # 0 = auto-detect
    preload_models: List[str] = field(default_factory=list)
    auto_unload: bool = True
    vram_headroom_mb: int = 512
    eviction_policy: str = "lru"
    model_dir: Optional[Path] = None
    enable_half_precision: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_models <= 0:
            raise ValueError("max_models must be positive")
        if self.model_dir is not None:
            self.model_dir = Path(self.model_dir)


@dataclass
class CachedModel:
    """Represents a cached model.

    Attributes:
        name: Model name/identifier
        model_type: Type of model (pytorch, onnx, etc.)
        model: The actual model object
        vram_mb: VRAM usage in megabytes
        ram_mb: System RAM usage in megabytes
        device: Device the model is on
        priority: Cache priority
        load_time_seconds: Time taken to load model
        created_at: When the model was loaded
        last_used: Last time model was used
        use_count: Number of times model was used
        path: Path to model file
    """
    name: str
    model_type: str = "pytorch"
    model: Any = None
    vram_mb: float = 0.0
    ram_mb: float = 0.0
    device: str = "cuda"
    priority: ModelPriority = ModelPriority.NORMAL
    load_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    path: Optional[Path] = None

    def touch(self) -> None:
        """Update usage metadata."""
        self.last_used = datetime.now()
        self.use_count += 1

    @property
    def total_memory_mb(self) -> float:
        """Get total memory usage (VRAM + RAM)."""
        return self.vram_mb + self.ram_mb


@dataclass
class ModelCacheStats:
    """Statistics for model cache.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        loads: Number of model loads
        evictions: Number of model evictions
        total_load_time: Total time spent loading models
        vram_used_mb: Current VRAM usage
        models_loaded: Number of models currently loaded
    """
    hits: int = 0
    misses: int = 0
    loads: int = 0
    evictions: int = 0
    total_load_time: float = 0.0
    vram_used_mb: float = 0.0
    models_loaded: int = 0

    @property
    def hit_ratio(self) -> float:
        """Calculate hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# Model Loaders
# =============================================================================

class ModelLoader:
    """Base class for model loaders."""

    def load(
        self,
        path: Path,
        device: str = "cuda",
        half_precision: bool = False,
    ) -> Tuple[Any, float, float]:
        """Load a model.

        Args:
            path: Path to model file
            device: Device to load model on
            half_precision: Whether to use FP16

        Returns:
            Tuple of (model, vram_mb, ram_mb)
        """
        raise NotImplementedError

    def unload(self, model: Any) -> None:
        """Unload a model and free memory.

        Args:
            model: Model to unload
        """
        del model
        self._cleanup_memory()

    def _cleanup_memory(self) -> None:
        """Force memory cleanup."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def get_memory_usage(self, model: Any) -> Tuple[float, float]:
        """Get memory usage of a model.

        Args:
            model: The model to measure

        Returns:
            Tuple of (vram_mb, ram_mb)
        """
        vram_mb = 0.0
        ram_mb = 0.0

        try:
            import torch
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    size_mb = param.numel() * param.element_size() / (1024 * 1024)
                    if param.device.type == "cuda":
                        vram_mb += size_mb
                    else:
                        ram_mb += size_mb
        except ImportError:
            pass

        return vram_mb, ram_mb


class PyTorchModelLoader(ModelLoader):
    """Loader for PyTorch models."""

    def load(
        self,
        path: Path,
        device: str = "cuda",
        half_precision: bool = False,
    ) -> Tuple[Any, float, float]:
        """Load a PyTorch model."""
        import torch

        # Record initial VRAM
        initial_vram = 0.0
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_vram = torch.cuda.memory_allocated() / (1024 * 1024)

        # Load model
        model = torch.load(path, map_location=device, weights_only=False)

        # Convert to half precision if requested
        if half_precision and hasattr(model, "half"):
            model = model.half()

        # Set to eval mode
        if hasattr(model, "eval"):
            model.eval()

        # Calculate VRAM usage
        vram_mb = 0.0
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) - initial_vram

        # Estimate RAM usage
        ram_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0

        return model, max(vram_mb, 0), ram_mb


class ONNXModelLoader(ModelLoader):
    """Loader for ONNX models."""

    def load(
        self,
        path: Path,
        device: str = "cuda",
        half_precision: bool = False,
    ) -> Tuple[Any, float, float]:
        """Load an ONNX model."""
        import onnxruntime as ort

        # Set up providers
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create inference session
        session = ort.InferenceSession(str(path), sess_options, providers=providers)

        # Estimate memory usage (rough estimate based on file size)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        vram_mb = file_size_mb * 1.5 if device.startswith("cuda") else 0
        ram_mb = file_size_mb

        return session, vram_mb, ram_mb


class SafetensorsModelLoader(ModelLoader):
    """Loader for Safetensors models."""

    def load(
        self,
        path: Path,
        device: str = "cuda",
        half_precision: bool = False,
    ) -> Tuple[Any, float, float]:
        """Load a Safetensors model."""
        from safetensors import safe_open
        import torch

        # Record initial VRAM
        initial_vram = 0.0
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_vram = torch.cuda.memory_allocated() / (1024 * 1024)

        # Load tensors
        tensors = {}
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if half_precision and tensor.dtype == torch.float32:
                    tensor = tensor.half()
                tensors[key] = tensor

        # Calculate VRAM usage
        vram_mb = 0.0
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) - initial_vram

        ram_mb = 0.0

        return tensors, max(vram_mb, 0), ram_mb


# =============================================================================
# Model Cache
# =============================================================================

class ModelCache:
    """High-performance model cache with VRAM management.

    Provides intelligent caching of loaded models with automatic
    eviction when VRAM pressure is detected.

    Example:
        >>> config = ModelCacheConfig(max_models=5, max_vram_mb=8000)
        >>> cache = ModelCache(config)
        >>>
        >>> # Get or load a model
        >>> model = cache.get_model("realesrgan-x4plus")
        >>>
        >>> # Check VRAM usage
        >>> usage = cache.get_vram_usage()
        >>> print(f"VRAM: {usage['used_mb']:.1f}/{usage['total_mb']:.1f} MB")
    """

    def __init__(self, config: Optional[ModelCacheConfig] = None):
        """Initialize model cache.

        Args:
            config: Cache configuration
        """
        self.config = config or ModelCacheConfig()
        self._models: OrderedDict[str, CachedModel] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = ModelCacheStats()

        # Initialize eviction policy
        self._eviction_policy = create_eviction_policy(self.config.eviction_policy)

        # Initialize model loaders
        self._loaders: Dict[str, ModelLoader] = {
            "pytorch": PyTorchModelLoader(),
            "onnx": ONNXModelLoader(),
            "safetensors": SafetensorsModelLoader(),
        }

        # Detect VRAM
        self._total_vram_mb = self._detect_vram()
        if self.config.max_vram_mb > 0:
            self._max_vram_mb = min(self.config.max_vram_mb, self._total_vram_mb)
        else:
            self._max_vram_mb = max(0, self._total_vram_mb - self.config.vram_headroom_mb)

        # Model manager integration (lazy loaded)
        self._model_manager: Optional[Any] = None

        logger.info(
            f"ModelCache initialized: max_models={self.config.max_models}, "
            f"max_vram={self._max_vram_mb:.0f}MB"
        )

    def _detect_vram(self) -> float:
        """Detect available VRAM.

        Returns:
            Total VRAM in megabytes
        """
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024 * 1024)
        except ImportError:
            pass
        return 8192  # Default 8GB

    def _get_model_manager(self) -> Any:
        """Get or create model manager instance.

        Returns:
            ModelManager instance
        """
        if self._model_manager is None:
            from ...utils.model_manager import ModelManager
            self._model_manager = ModelManager(self.config.model_dir)
        return self._model_manager

    def get_model(
        self,
        name: str,
        device: str = "cuda",
        priority: ModelPriority = ModelPriority.NORMAL,
    ) -> Any:
        """Get a model from cache, loading if necessary.

        Args:
            name: Model name (from model registry)
            device: Device to load model on
            priority: Cache priority for this model

        Returns:
            The loaded model
        """
        with self._lock:
            # Check cache
            if name in self._models:
                cached = self._models[name]
                cached.touch()
                self._eviction_policy.on_access(name)
                self._models.move_to_end(name)
                self._stats.hits += 1
                logger.debug(f"Model cache hit: {name}")
                return cached.model

            # Cache miss
            self._stats.misses += 1
            logger.debug(f"Model cache miss: {name}")

            # Ensure we have space
            if self.config.auto_unload:
                self._ensure_space()

            # Load model
            model, vram_mb, ram_mb, load_time = self._load_model(name, device)

            # Create cache entry
            cached = CachedModel(
                name=name,
                model=model,
                vram_mb=vram_mb,
                ram_mb=ram_mb,
                device=device,
                priority=priority,
                load_time_seconds=load_time,
            )

            self._models[name] = cached
            self._eviction_policy.on_insert(name, int(vram_mb * 1024 * 1024))

            self._stats.loads += 1
            self._stats.total_load_time += load_time
            self._update_stats()

            logger.info(f"Loaded model: {name} ({vram_mb:.1f}MB VRAM, {load_time:.2f}s)")

            return model

    def put_model(
        self,
        name: str,
        model: Any,
        vram_mb: float = 0.0,
        priority: ModelPriority = ModelPriority.NORMAL,
    ) -> None:
        """Put an already-loaded model into cache.

        Args:
            name: Model name
            model: The model object
            vram_mb: VRAM usage (estimated if 0)
            priority: Cache priority
        """
        with self._lock:
            # Estimate VRAM if not provided
            if vram_mb == 0:
                vram_mb, _ = PyTorchModelLoader().get_memory_usage(model)

            # Ensure space
            if self.config.auto_unload:
                self._ensure_space(additional_vram_mb=vram_mb)

            cached = CachedModel(
                name=name,
                model=model,
                vram_mb=vram_mb,
                priority=priority,
            )

            self._models[name] = cached
            self._eviction_policy.on_insert(name, int(vram_mb * 1024 * 1024))
            self._update_stats()

    def _load_model(
        self,
        name: str,
        device: str,
    ) -> Tuple[Any, float, float, float]:
        """Load a model from disk.

        Args:
            name: Model name
            device: Device to load on

        Returns:
            Tuple of (model, vram_mb, ram_mb, load_time_seconds)
        """
        start_time = time.time()

        # Get model path from model manager
        manager = self._get_model_manager()
        model_path = manager.get_model_path(name)

        # Ensure model is downloaded
        if not model_path.exists():
            model_path = manager.download_model(name)

        # Determine loader based on file extension
        suffix = model_path.suffix.lower()
        if suffix in (".pt", ".pth"):
            loader = self._loaders["pytorch"]
        elif suffix == ".onnx":
            loader = self._loaders["onnx"]
        elif suffix == ".safetensors":
            loader = self._loaders["safetensors"]
        else:
            loader = self._loaders["pytorch"]  # Default

        # Load model
        model, vram_mb, ram_mb = loader.load(
            model_path,
            device,
            self.config.enable_half_precision,
        )

        load_time = time.time() - start_time
        return model, vram_mb, ram_mb, load_time

    def _ensure_space(self, additional_vram_mb: float = 0.0) -> None:
        """Ensure there's space for a new model.

        Args:
            additional_vram_mb: Additional VRAM needed
        """
        # Check model count
        while len(self._models) >= self.config.max_models:
            if not self._evict_one():
                break

        # Check VRAM
        current_vram = sum(m.vram_mb for m in self._models.values())
        while current_vram + additional_vram_mb > self._max_vram_mb:
            evicted_vram = self._evict_one()
            if evicted_vram == 0:
                break
            current_vram -= evicted_vram

    def _evict_one(self) -> float:
        """Evict one model from cache.

        Returns:
            VRAM freed in MB
        """
        # Build candidates (exclude critical priority)
        candidates = [
            EvictionCandidate(
                key=name,
                score=0,
                size_bytes=int(cached.vram_mb * 1024 * 1024),
                metadata={"priority": cached.priority.value},
            )
            for name, cached in self._models.items()
            if cached.priority != ModelPriority.CRITICAL
        ]

        if not candidates:
            return 0.0

        victim = self._eviction_policy.select_victim(candidates)
        if victim is None:
            return 0.0

        vram_freed = self._unload_model(victim.key)
        self._stats.evictions += 1

        logger.info(f"Evicted model: {victim.key} (freed {vram_freed:.1f}MB VRAM)")
        return vram_freed

    def _unload_model(self, name: str) -> float:
        """Unload a specific model.

        Args:
            name: Model name

        Returns:
            VRAM freed in MB
        """
        if name not in self._models:
            return 0.0

        cached = self._models.pop(name)
        vram_freed = cached.vram_mb

        # Determine loader and unload
        if cached.model_type in self._loaders:
            self._loaders[cached.model_type].unload(cached.model)
        else:
            del cached.model

        self._eviction_policy.on_remove(name)
        self._update_stats()

        return vram_freed

    def unload_model(self, name: str) -> bool:
        """Explicitly unload a model.

        Args:
            name: Model name

        Returns:
            True if model was unloaded
        """
        with self._lock:
            if name in self._models:
                self._unload_model(name)
                return True
            return False

    def unload_all(self) -> None:
        """Unload all cached models."""
        with self._lock:
            for name in list(self._models.keys()):
                self._unload_model(name)

    def contains(self, name: str) -> bool:
        """Check if a model is cached.

        Args:
            name: Model name

        Returns:
            True if model is in cache
        """
        return name in self._models

    def get_model_info(self, name: str) -> Optional[CachedModel]:
        """Get information about a cached model.

        Args:
            name: Model name

        Returns:
            CachedModel info or None
        """
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """List all cached model names.

        Returns:
            List of model names
        """
        return list(self._models.keys())

    def _update_stats(self) -> None:
        """Update cache statistics."""
        self._stats.models_loaded = len(self._models)
        self._stats.vram_used_mb = sum(m.vram_mb for m in self._models.values())

    def get_stats(self) -> ModelCacheStats:
        """Get cache statistics.

        Returns:
            ModelCacheStats with current metrics
        """
        with self._lock:
            self._update_stats()
            return self._stats

    def get_vram_usage(self) -> Dict[str, float]:
        """Get VRAM usage information.

        Returns:
            Dictionary with VRAM usage details
        """
        with self._lock:
            used_mb = sum(m.vram_mb for m in self._models.values())
            return {
                "used_mb": used_mb,
                "max_mb": self._max_vram_mb,
                "total_mb": self._total_vram_mb,
                "available_mb": max(0, self._max_vram_mb - used_mb),
                "utilization": used_mb / self._max_vram_mb if self._max_vram_mb > 0 else 0,
            }

    def set_priority(self, name: str, priority: ModelPriority) -> bool:
        """Set the cache priority for a model.

        Args:
            name: Model name
            priority: New priority level

        Returns:
            True if priority was set
        """
        with self._lock:
            if name in self._models:
                self._models[name].priority = priority
                return True
            return False

    def preload_models(self, device: str = "cuda") -> int:
        """Preload configured models.

        Args:
            device: Device to load models on

        Returns:
            Number of models preloaded
        """
        count = 0
        for name in self.config.preload_models:
            try:
                self.get_model(name, device, ModelPriority.HIGH)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to preload model {name}: {e}")
        return count

    def warmup(
        self,
        models: List[Tuple[str, str]],
        device: str = "cuda",
    ) -> int:
        """Warm up cache by preloading models.

        Args:
            models: List of (name, model_type) tuples
            device: Device to load on

        Returns:
            Number of models loaded
        """
        logger.info(f"Warming up {len(models)} models...")
        count = 0
        for name, _ in models:
            try:
                self.get_model(name, device)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to warm up {name}: {e}")
        return count


# =============================================================================
# Global Model Cache
# =============================================================================

_global_model_cache: Optional[ModelCache] = None
_cache_lock = threading.Lock()


def get_model_cache(config: Optional[ModelCacheConfig] = None) -> ModelCache:
    """Get or create the global model cache.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ModelCache instance
    """
    global _global_model_cache

    with _cache_lock:
        if _global_model_cache is None:
            _global_model_cache = ModelCache(config)
        return _global_model_cache


def clear_model_cache() -> None:
    """Clear the global model cache."""
    global _global_model_cache

    with _cache_lock:
        if _global_model_cache is not None:
            _global_model_cache.unload_all()
            _global_model_cache = None
