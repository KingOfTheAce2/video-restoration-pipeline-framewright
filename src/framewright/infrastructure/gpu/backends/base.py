"""Abstract Backend Interface for FrameWright.

Defines the abstract interface that all compute backends must implement,
providing a consistent API for different hardware platforms.

The backend system uses a fallback chain:
    CUDA -> TensorRT -> ROCm -> Metal -> Vulkan -> DirectML -> CPU

Each backend handles:
- Device initialization and cleanup
- Memory allocation and management
- Model loading and inference
- Tensor operations
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..detector import BackendType, GPUVendor, HardwareInfo, get_hardware_info

logger = logging.getLogger(__name__)


@dataclass
class BackendCapabilities:
    """Capabilities and limits of a backend."""
    name: str
    backend_type: BackendType
    vendor: GPUVendor

    # Feature support
    supports_fp16: bool = True
    supports_fp32: bool = True
    supports_int8: bool = False
    supports_dynamic_shapes: bool = True
    supports_batching: bool = True

    # Memory limits
    max_memory_mb: int = 0
    recommended_memory_mb: int = 0

    # Processing limits
    max_batch_size: int = 32
    max_tile_size: int = 1024

    # Supported models/operations
    supported_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "backend_type": self.backend_type.value,
            "vendor": self.vendor.value,
            "supports_fp16": self.supports_fp16,
            "supports_int8": self.supports_int8,
            "max_memory_mb": self.max_memory_mb,
            "max_batch_size": self.max_batch_size,
        }


class Backend(ABC):
    """Abstract base class for compute backends.

    All compute backends (CUDA, Metal, Vulkan, etc.) must implement
    this interface to ensure consistent behavior across platforms.

    Example:
        >>> backend = get_backend()
        >>> backend.initialize()
        >>> result = backend.run_model("realesrgan", input_tensor)
        >>> backend.cleanup()
    """

    def __init__(self, device_id: int = 0):
        """Initialize backend.

        Args:
            device_id: Device index to use
        """
        self.device_id = device_id
        self._initialized = False
        self._capabilities: Optional[BackendCapabilities] = None

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get human-readable backend name."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend.

        Must be called before any processing. Handles device setup,
        library initialization, etc.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources.

        Should release all allocated resources, unload models, etc.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities.

        Returns:
            BackendCapabilities describing what this backend supports
        """
        pass

    @abstractmethod
    def allocate_memory(self, size_mb: float) -> bool:
        """Pre-allocate memory for processing.

        Args:
            size_mb: Amount of memory to allocate in MB

        Returns:
            True if allocation successful
        """
        pass

    @abstractmethod
    def free_memory(self) -> None:
        """Free allocated memory."""
        pass

    @abstractmethod
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information.

        Returns:
            Dict with 'total_mb', 'used_mb', 'free_mb' keys
        """
        pass

    @abstractmethod
    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        """Load a model for inference.

        Args:
            model_name: Name/identifier of the model
            model_path: Optional path to model weights
            **kwargs: Backend-specific options

        Returns:
            True if model loaded successfully
        """
        pass

    @abstractmethod
    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory.

        Args:
            model_name: Name of model to unload
        """
        pass

    @abstractmethod
    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        """Run inference with a loaded model.

        Args:
            model_name: Name of model to use
            inputs: Input data (format depends on backend)
            **kwargs: Additional options

        Returns:
            Model output (format depends on backend)
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


# =============================================================================
# Concrete Backend Implementations
# =============================================================================

class CUDABackend(Backend):
    """NVIDIA CUDA backend using PyTorch."""

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._torch = None
        self._device = None
        self._loaded_models: Dict[str, Any] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA

    @property
    def name(self) -> str:
        return "CUDA (PyTorch)"

    def initialize(self) -> bool:
        try:
            import torch
            self._torch = torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return False

            if self.device_id >= torch.cuda.device_count():
                logger.warning(f"Device {self.device_id} not available")
                return False

            self._device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self._device)

            # Warm up
            _ = torch.zeros(1, device=self._device)

            self._initialized = True
            logger.info(f"CUDA backend initialized on device {self.device_id}")
            return True

        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
            return False

    def cleanup(self) -> None:
        if self._torch is not None:
            # Unload all models
            for name in list(self._loaded_models.keys()):
                self.unload_model(name)

            # Clear cache
            self._torch.cuda.empty_cache()

        self._initialized = False

    def get_capabilities(self) -> BackendCapabilities:
        if self._capabilities is not None:
            return self._capabilities

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.CUDA,
            vendor=GPUVendor.NVIDIA,
        )

        if self._torch is not None and self._torch.cuda.is_available():
            props = self._torch.cuda.get_device_properties(self.device_id)
            caps.max_memory_mb = int(props.total_memory / (1024 * 1024))
            caps.supports_fp16 = props.major >= 6  # Pascal and later
            caps.supports_int8 = props.major >= 7  # Turing and later

        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        try:
            # PyTorch handles memory allocation automatically
            # We just check if enough is available
            info = self.get_memory_info()
            return info["free_mb"] >= size_mb
        except Exception:
            return False

    def free_memory(self) -> None:
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    def get_memory_info(self) -> Dict[str, float]:
        if self._torch is None or not self._torch.cuda.is_available():
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

        props = self._torch.cuda.get_device_properties(self.device_id)
        total = props.total_memory / (1024 * 1024)
        allocated = self._torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
        reserved = self._torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)

        return {
            "total_mb": total,
            "used_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": total - reserved,
        }

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        logger.debug(f"Loading model {model_name} on CUDA")
        # Actual model loading would be implemented by specific processors
        self._loaded_models[model_name] = {"path": model_path, "kwargs": kwargs}
        return True

    def unload_model(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            self.free_memory()

    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        # Actual inference would be implemented by specific processors
        return inputs


class MetalBackend(Backend):
    """Apple Metal backend using PyTorch MPS."""

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._torch = None
        self._device = None
        self._loaded_models: Dict[str, Any] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.METAL

    @property
    def name(self) -> str:
        return "Metal (MPS)"

    def initialize(self) -> bool:
        try:
            import torch
            self._torch = torch

            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                logger.warning("MPS not available")
                return False

            self._device = torch.device("mps")

            # Warm up
            _ = torch.zeros(1, device=self._device)

            self._initialized = True
            logger.info("Metal backend initialized")
            return True

        except Exception as e:
            logger.error(f"Metal initialization failed: {e}")
            return False

    def cleanup(self) -> None:
        for name in list(self._loaded_models.keys()):
            self.unload_model(name)
        self._initialized = False

    def get_capabilities(self) -> BackendCapabilities:
        if self._capabilities is not None:
            return self._capabilities

        info = get_hardware_info()

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.METAL,
            vendor=GPUVendor.APPLE,
            supports_fp16=True,
            supports_int8=True,  # Neural Engine
        )

        if info.primary_device:
            caps.max_memory_mb = info.primary_device.total_memory_mb

        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        # MPS handles memory automatically
        return True

    def free_memory(self) -> None:
        import gc
        gc.collect()

    def get_memory_info(self) -> Dict[str, float]:
        info = get_hardware_info()
        if info.primary_device:
            return {
                "total_mb": info.primary_device.total_memory_mb,
                "used_mb": info.primary_device.total_memory_mb - info.primary_device.free_memory_mb,
                "free_mb": info.primary_device.free_memory_mb,
            }
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        logger.debug(f"Loading model {model_name} on Metal")
        self._loaded_models[model_name] = {"path": model_path, "kwargs": kwargs}
        return True

    def unload_model(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            self.free_memory()

    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        return inputs


class VulkanBackend(Backend):
    """Vulkan backend using ncnn-vulkan for cross-platform GPU support."""

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._ncnn_path: Optional[Path] = None
        self._loaded_models: Dict[str, Any] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.VULKAN

    @property
    def name(self) -> str:
        return "Vulkan (ncnn)"

    def initialize(self) -> bool:
        import shutil

        # Find ncnn-vulkan binary
        ncnn_names = ["realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"]

        for name in ncnn_names:
            path = shutil.which(name)
            if path:
                self._ncnn_path = Path(path)
                break

        if self._ncnn_path is None:
            # Check ~/.framewright/bin
            home = Path.home()
            for name in ncnn_names:
                path = home / ".framewright" / "bin" / name
                if path.exists():
                    self._ncnn_path = path
                    break

        if self._ncnn_path is None:
            logger.warning("ncnn-vulkan not found")
            return False

        self._initialized = True
        logger.info(f"Vulkan backend initialized: {self._ncnn_path}")
        return True

    def cleanup(self) -> None:
        self._loaded_models.clear()
        self._initialized = False

    def get_capabilities(self) -> BackendCapabilities:
        if self._capabilities is not None:
            return self._capabilities

        info = get_hardware_info()

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.VULKAN,
            vendor=info.primary_device.vendor if info.primary_device else GPUVendor.UNKNOWN,
            supports_fp16=True,
            supports_int8=False,
            supported_models=["realesrgan", "waifu2x"],
        )

        if info.primary_device:
            caps.max_memory_mb = info.primary_device.total_memory_mb

        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        return True  # ncnn handles memory

    def free_memory(self) -> None:
        pass  # ncnn handles cleanup

    def get_memory_info(self) -> Dict[str, float]:
        info = get_hardware_info()
        if info.primary_device:
            return {
                "total_mb": info.primary_device.total_memory_mb,
                "used_mb": 0,  # Cannot query ncnn memory
                "free_mb": info.primary_device.free_memory_mb,
            }
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        logger.debug(f"Loading model {model_name} for Vulkan")
        self._loaded_models[model_name] = {"path": model_path, "kwargs": kwargs}
        return True

    def unload_model(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]

    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        # ncnn inference is handled via subprocess in the actual processors
        return inputs


class CPUBackend(Backend):
    """CPU fallback backend."""

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._loaded_models: Dict[str, Any] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CPU

    @property
    def name(self) -> str:
        return "CPU"

    def initialize(self) -> bool:
        self._initialized = True
        logger.info("CPU backend initialized")
        return True

    def cleanup(self) -> None:
        self._loaded_models.clear()
        self._initialized = False

    def get_capabilities(self) -> BackendCapabilities:
        if self._capabilities is not None:
            return self._capabilities

        import os
        info = get_hardware_info()

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.CPU,
            vendor=GPUVendor.UNKNOWN,
            supports_fp16=False,  # CPU FP16 is slow
            supports_int8=True,
            max_batch_size=1,  # CPU is slow, minimize batching
        )

        caps.max_memory_mb = info.ram_total_mb

        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        info = get_hardware_info()
        return info.ram_available_mb >= size_mb

    def free_memory(self) -> None:
        import gc
        gc.collect()

    def get_memory_info(self) -> Dict[str, float]:
        info = get_hardware_info()
        return {
            "total_mb": info.ram_total_mb,
            "used_mb": info.ram_total_mb - info.ram_available_mb,
            "free_mb": info.ram_available_mb,
        }

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        logger.debug(f"Loading model {model_name} on CPU")
        self._loaded_models[model_name] = {"path": model_path, "kwargs": kwargs}
        return True

    def unload_model(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            self.free_memory()

    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        return inputs


# =============================================================================
# Backend Registry and Factory
# =============================================================================

# Registry of backend classes by type
_backend_registry: Dict[BackendType, Type[Backend]] = {
    BackendType.CUDA: CUDABackend,
    BackendType.METAL: MetalBackend,
    BackendType.VULKAN: VulkanBackend,
    BackendType.CPU: CPUBackend,
}

# Fallback chain for automatic backend selection
_fallback_chain = [
    BackendType.CUDA,
    BackendType.TENSORRT,
    BackendType.ROCM,
    BackendType.METAL,
    BackendType.COREML,
    BackendType.ONEAPI,
    BackendType.VULKAN,
    BackendType.DIRECTML,
    BackendType.OPENVINO,
    BackendType.CPU,
]

# Cached backend instance
_active_backend: Optional[Backend] = None


def get_backend(
    backend_type: Optional[BackendType] = None,
    device_id: int = 0,
    force_new: bool = False,
) -> Backend:
    """Get a backend instance.

    If no backend_type is specified, automatically selects the best
    available backend based on hardware detection.

    Args:
        backend_type: Specific backend to use (None for auto)
        device_id: Device index
        force_new: Force creating a new instance (don't use cached)

    Returns:
        Backend instance (may be cached)

    Raises:
        RuntimeError: If no suitable backend is available
    """
    global _active_backend

    # Return cached if available and compatible
    if not force_new and _active_backend is not None:
        if backend_type is None or _active_backend.backend_type == backend_type:
            if _active_backend.device_id == device_id:
                return _active_backend

    # Auto-select backend
    if backend_type is None:
        info = get_hardware_info()
        backend_type = info.recommended_backend
        logger.debug(f"Auto-selected backend: {backend_type.value}")

    # Get backend class
    backend_class = _backend_registry.get(backend_type)

    if backend_class is None:
        # Try fallback chain
        for fallback_type in _fallback_chain:
            if fallback_type in _backend_registry:
                backend_class = _backend_registry[fallback_type]
                logger.warning(
                    f"Backend {backend_type.value} not available, "
                    f"falling back to {fallback_type.value}"
                )
                break

    if backend_class is None:
        raise RuntimeError("No suitable backend available")

    # Create instance
    backend = backend_class(device_id)

    # Cache it
    _active_backend = backend

    return backend


def get_backend_for_vendor(vendor: GPUVendor, device_id: int = 0) -> Backend:
    """Get the best backend for a specific GPU vendor.

    Args:
        vendor: GPU vendor
        device_id: Device index

    Returns:
        Backend instance optimized for the vendor
    """
    vendor_backends = {
        GPUVendor.NVIDIA: BackendType.CUDA,
        GPUVendor.AMD: BackendType.VULKAN,  # or ROCm on Linux
        GPUVendor.INTEL: BackendType.VULKAN,  # or oneAPI
        GPUVendor.APPLE: BackendType.METAL,
        GPUVendor.UNKNOWN: BackendType.CPU,
    }

    backend_type = vendor_backends.get(vendor, BackendType.CPU)
    return get_backend(backend_type, device_id)


def list_backends() -> List[Dict[str, Any]]:
    """List all available backends with their capabilities.

    Returns:
        List of backend info dictionaries
    """
    info = get_hardware_info()
    backends = []

    for backend_type in info.available_backends:
        backend_class = _backend_registry.get(backend_type)
        if backend_class:
            try:
                backend = backend_class()
                if backend.initialize():
                    caps = backend.get_capabilities()
                    backends.append({
                        "type": backend_type.value,
                        "name": backend.name,
                        "capabilities": caps.to_dict(),
                        "available": True,
                    })
                    backend.cleanup()
                else:
                    backends.append({
                        "type": backend_type.value,
                        "name": backend.name,
                        "available": False,
                    })
            except Exception as e:
                logger.debug(f"Backend {backend_type.value} check failed: {e}")

    return backends


def register_backend(backend_type: BackendType, backend_class: Type[Backend]) -> None:
    """Register a custom backend implementation.

    Args:
        backend_type: Backend type identifier
        backend_class: Backend class to register
    """
    _backend_registry[backend_type] = backend_class
    logger.debug(f"Registered backend: {backend_type.value}")
