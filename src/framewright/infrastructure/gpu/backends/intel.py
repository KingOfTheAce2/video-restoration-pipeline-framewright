"""Intel GPU Backend for FrameWright.

Provides GPU acceleration on Intel GPUs using Intel Extension for PyTorch (IPEX)
and OpenVINO, supporting both integrated and discrete (Arc) GPUs.

The backend supports:
- Intel Extension for PyTorch (IPEX) for XPU acceleration
- OpenVINO for optimized inference
- Arc discrete GPUs and Iris/UHD integrated graphics
- Automatic fallback to CPU when XPU is unavailable
"""

import gc
import logging
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..detector import BackendType, GPUVendor, get_hardware_info
from .base import Backend, BackendCapabilities, register_backend

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_torch = None
_torch_checked = False
_ipex = None
_ipex_checked = False
_openvino = None
_openvino_checked = False


def _get_torch():
    """Lazy import PyTorch."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch


def _get_ipex():
    """Lazy import Intel Extension for PyTorch."""
    global _ipex, _ipex_checked
    if not _ipex_checked:
        try:
            import intel_extension_for_pytorch as ipex
            _ipex = ipex
        except ImportError:
            _ipex = None
        _ipex_checked = True
    return _ipex


def _get_openvino():
    """Lazy import OpenVINO."""
    global _openvino, _openvino_checked
    if not _openvino_checked:
        try:
            from openvino.runtime import Core
            _openvino = Core()
        except ImportError:
            _openvino = None
        _openvino_checked = True
    return _openvino


class IntelGPUType:
    """Intel GPU type classification."""
    UNKNOWN = "unknown"
    ARC_A770 = "arc_a770"
    ARC_A750 = "arc_a750"
    ARC_A580 = "arc_a580"
    ARC_A380 = "arc_a380"
    ARC_A310 = "arc_a310"
    IRIS_XE = "iris_xe"
    IRIS_PLUS = "iris_plus"
    UHD_770 = "uhd_770"
    UHD_730 = "uhd_730"
    UHD_630 = "uhd_630"
    INTEGRATED = "integrated"


# VRAM estimates for Intel GPUs
INTEL_VRAM_ESTIMATES = {
    IntelGPUType.ARC_A770: 16384,
    IntelGPUType.ARC_A750: 8192,
    IntelGPUType.ARC_A580: 8192,
    IntelGPUType.ARC_A380: 6144,
    IntelGPUType.ARC_A310: 4096,
    IntelGPUType.IRIS_XE: 4096,  # Shares system RAM
    IntelGPUType.IRIS_PLUS: 3072,
    IntelGPUType.UHD_770: 2048,
    IntelGPUType.UHD_730: 2048,
    IntelGPUType.UHD_630: 1536,
    IntelGPUType.INTEGRATED: 2048,
    IntelGPUType.UNKNOWN: 2048,
}


@dataclass
class IntelConfig:
    """Configuration for Intel GPU backend.

    Attributes:
        device_id: GPU device index to use (default 0)
        use_xpu: Enable Intel XPU (GPU) acceleration via IPEX
        precision: Computation precision ("fp32", "fp16", "bf16")
        use_openvino: Use OpenVINO for inference
        openvino_device: OpenVINO device target ("GPU", "GPU.0", "GPU.1")
        cache_dir: Directory for caching compiled models
        fallback_to_cpu: Fall back to CPU if XPU unavailable
        enable_onednn: Enable oneDNN optimizations
        memory_fraction: Fraction of GPU memory to use
    """
    device_id: int = 0
    use_xpu: bool = True
    precision: str = "fp16"
    use_openvino: bool = True
    openvino_device: str = "GPU"
    cache_dir: Optional[Path] = None
    fallback_to_cpu: bool = True
    enable_onednn: bool = True
    memory_fraction: float = 0.9

    def __post_init__(self):
        """Validate and set up configuration."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".framewright" / "cache" / "intel"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"Invalid precision: {self.precision}")


@dataclass
class IntelDeviceInfo:
    """Information about an Intel GPU device.

    Attributes:
        device_id: Device index
        name: GPU name (e.g., "Intel Arc A770")
        gpu_type: GPU type classification
        total_memory_mb: Total VRAM in MB
        free_memory_mb: Available VRAM in MB
        driver_version: Driver version
        is_discrete: True if discrete GPU (Arc)
        supports_fp16: FP16 support
        supports_bf16: BF16 support
    """
    device_id: int
    name: str
    gpu_type: str = IntelGPUType.UNKNOWN
    total_memory_mb: int = 0
    free_memory_mb: int = 0
    driver_version: str = ""
    is_discrete: bool = False
    supports_fp16: bool = True
    supports_bf16: bool = True


def detect_intel_gpu_type(name: str) -> str:
    """Detect Intel GPU type from name string.

    Args:
        name: GPU device name

    Returns:
        IntelGPUType constant
    """
    name_lower = name.lower()

    # Arc series (discrete)
    if "a770" in name_lower:
        return IntelGPUType.ARC_A770
    if "a750" in name_lower:
        return IntelGPUType.ARC_A750
    if "a580" in name_lower:
        return IntelGPUType.ARC_A580
    if "a380" in name_lower:
        return IntelGPUType.ARC_A380
    if "a310" in name_lower:
        return IntelGPUType.ARC_A310

    # Iris series (integrated)
    if "iris xe" in name_lower or "iris(r) xe" in name_lower:
        return IntelGPUType.IRIS_XE
    if "iris plus" in name_lower:
        return IntelGPUType.IRIS_PLUS

    # UHD series (integrated)
    if "uhd" in name_lower:
        if "770" in name_lower:
            return IntelGPUType.UHD_770
        if "730" in name_lower:
            return IntelGPUType.UHD_730
        if "630" in name_lower:
            return IntelGPUType.UHD_630

    # Generic check for Arc
    if "arc" in name_lower:
        return IntelGPUType.ARC_A380  # Default Arc

    return IntelGPUType.INTEGRATED if "intel" in name_lower else IntelGPUType.UNKNOWN


def query_intel_gpus_ipex() -> List[IntelDeviceInfo]:
    """Query Intel GPU information using IPEX.

    Returns:
        List of IntelDeviceInfo for each detected Intel GPU
    """
    devices = []
    torch = _get_torch()
    ipex = _get_ipex()

    if torch is None or ipex is None:
        return devices

    try:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return devices

        device_count = torch.xpu.device_count()
        for i in range(device_count):
            props = torch.xpu.get_device_properties(i)
            name = props.name if hasattr(props, "name") else f"Intel GPU {i}"
            gpu_type = detect_intel_gpu_type(name)

            # Get memory info
            total_memory = props.total_memory if hasattr(props, "total_memory") else 0
            total_mb = total_memory // (1024 * 1024) if total_memory > 0 else INTEL_VRAM_ESTIMATES.get(gpu_type, 2048)

            devices.append(IntelDeviceInfo(
                device_id=i,
                name=name,
                gpu_type=gpu_type,
                total_memory_mb=total_mb,
                free_memory_mb=total_mb,
                is_discrete=gpu_type.startswith("arc_"),
                supports_fp16=True,
                supports_bf16=True,
            ))

    except Exception as e:
        logger.debug(f"IPEX GPU query failed: {e}")

    return devices


def query_intel_gpus_openvino() -> List[IntelDeviceInfo]:
    """Query Intel GPU information using OpenVINO.

    Returns:
        List of IntelDeviceInfo for each detected Intel GPU
    """
    devices = []
    core = _get_openvino()

    if core is None:
        return devices

    try:
        # Get available devices
        available = core.available_devices
        gpu_devices = [d for d in available if d.startswith("GPU")]

        for i, device_name in enumerate(gpu_devices):
            # Query device properties
            full_name = core.get_property(device_name, "FULL_DEVICE_NAME")
            gpu_type = detect_intel_gpu_type(full_name)

            devices.append(IntelDeviceInfo(
                device_id=i,
                name=full_name,
                gpu_type=gpu_type,
                total_memory_mb=INTEL_VRAM_ESTIMATES.get(gpu_type, 2048),
                free_memory_mb=INTEL_VRAM_ESTIMATES.get(gpu_type, 2048),
                is_discrete=gpu_type.startswith("arc_"),
            ))

    except Exception as e:
        logger.debug(f"OpenVINO GPU query failed: {e}")

    return devices


def query_intel_gpus_windows() -> List[IntelDeviceInfo]:
    """Query Intel GPU information on Windows.

    Returns:
        List of IntelDeviceInfo for each detected Intel GPU
    """
    devices = []

    if platform.system() != "Windows":
        return devices

    try:
        cmd = [
            "powershell", "-Command",
            "Get-WmiObject Win32_VideoController | "
            "Where-Object { $_.Name -like '*Intel*' } | "
            "Select-Object Name, AdapterRAM, DriverVersion | "
            "ConvertTo-Json"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        )

        if result.returncode != 0:
            return devices

        import json
        data = json.loads(result.stdout)

        if isinstance(data, dict):
            data = [data]

        for idx, gpu_data in enumerate(data):
            name = gpu_data.get("Name", "Intel GPU")
            gpu_type = detect_intel_gpu_type(name)

            # Get RAM (handle WMI overflow for > 4GB)
            ram_bytes = gpu_data.get("AdapterRAM", 0) or 0
            if ram_bytes < 0:
                ram_bytes = 4294967296 + ram_bytes
            total_mb = int(ram_bytes / (1024 * 1024))

            if total_mb == 0:
                total_mb = INTEL_VRAM_ESTIMATES.get(gpu_type, 2048)

            devices.append(IntelDeviceInfo(
                device_id=idx,
                name=name,
                gpu_type=gpu_type,
                total_memory_mb=total_mb,
                free_memory_mb=total_mb,
                driver_version=gpu_data.get("DriverVersion", ""),
                is_discrete=gpu_type.startswith("arc_"),
            ))

    except Exception as e:
        logger.debug(f"Windows Intel GPU query failed: {e}")

    return devices


class IntelModelWrapper:
    """Wrapper for models loaded on Intel GPU.

    Handles both PyTorch/IPEX and OpenVINO models with a unified interface.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        model_type: str,  # "pytorch" or "openvino"
        device: Any,
        config: IntelConfig,
    ):
        self.name = name
        self.model = model
        self.model_type = model_type
        self.device = device
        self.config = config
        self._lock = threading.Lock()
        self._compiled_model = None  # For OpenVINO
        self._infer_request = None

    def __call__(self, inputs: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Run inference on the model."""
        with self._lock:
            if self.model_type == "pytorch":
                return self._run_pytorch(inputs)
            elif self.model_type == "openvino":
                return self._run_openvino(inputs)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

    def _run_pytorch(self, inputs: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Run PyTorch/IPEX model inference."""
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available")

        # Convert input to tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)

        # Move to device
        inputs = inputs.to(self.device)

        # Convert precision
        if self.config.precision == "fp16":
            inputs = inputs.half()
        elif self.config.precision == "bf16":
            inputs = inputs.bfloat16()

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

        # Convert back to numpy
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().float().numpy()
        elif isinstance(outputs, (list, tuple)):
            return outputs[0].cpu().float().numpy()
        return outputs

    def _run_openvino(self, inputs: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Run OpenVINO model inference."""
        torch = _get_torch()

        # Convert to numpy if needed
        if torch is not None and isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()

        # Ensure contiguous
        inputs = np.ascontiguousarray(inputs)

        # Run inference
        if self._infer_request is None:
            self._infer_request = self._compiled_model.create_infer_request()

        self._infer_request.infer({0: inputs})
        return self._infer_request.get_output_tensor(0).data.copy()

    def cleanup(self):
        """Release model resources."""
        self.model = None
        self._compiled_model = None
        self._infer_request = None
        gc.collect()


class IntelBackend(Backend):
    """Intel GPU backend for acceleration.

    Supports Intel GPUs via Intel Extension for PyTorch (IPEX) and OpenVINO.
    Handles both Arc discrete GPUs and integrated graphics (Iris, UHD).

    Example:
        >>> config = IntelConfig(device_id=0, precision="fp16")
        >>> backend = IntelBackend(config)
        >>> if backend.initialize():
        ...     backend.load_model("upscaler", "/path/to/model.xml")
        ...     result = backend.run_inference("upscaler", input_tensor)
        ...     backend.cleanup()
    """

    def __init__(self, config: Optional[IntelConfig] = None, device_id: int = 0):
        """Initialize Intel backend.

        Args:
            config: Backend configuration (created with defaults if None)
            device_id: GPU device index (overridden by config if provided)
        """
        super().__init__(device_id if config is None else config.device_id)
        self.config = config or IntelConfig(device_id=device_id)
        self._torch = None
        self._ipex = None
        self._device = None
        self._openvino_core = None
        self._models: Dict[str, IntelModelWrapper] = {}
        self._device_info: Optional[IntelDeviceInfo] = None
        self._lock = threading.Lock()
        self._using_cpu_fallback = False
        self._using_openvino = False

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.ONEAPI

    @property
    def name(self) -> str:
        """Get human-readable backend name."""
        if self._using_cpu_fallback:
            return "Intel oneAPI (CPU Fallback)"
        if self._using_openvino:
            return "Intel OpenVINO"
        return "Intel oneAPI (XPU)"

    def is_available(self) -> bool:
        """Check if Intel XPU or OpenVINO is available.

        Checks for:
        1. Intel Extension for PyTorch (IPEX) with XPU support
        2. OpenVINO with GPU device

        Returns:
            True if Intel GPU backend can be used
        """
        # Check IPEX
        torch = _get_torch()
        ipex = _get_ipex()

        if torch is not None and ipex is not None:
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return True
            except Exception:
                pass

        # Check OpenVINO
        core = _get_openvino()
        if core is not None:
            try:
                available = core.available_devices
                if any(d.startswith("GPU") for d in available):
                    return True
            except Exception:
                pass

        return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the Intel GPU.

        Returns:
            Dictionary containing:
            - name: GPU name
            - gpu_type: GPU type classification
            - total_memory_mb: Total VRAM
            - free_memory_mb: Available VRAM
            - is_discrete: True if discrete GPU
            - driver_version: Driver version
        """
        if self._device_info is None:
            # Try IPEX first
            devices = query_intel_gpus_ipex()

            # Fall back to OpenVINO
            if not devices:
                devices = query_intel_gpus_openvino()

            # Fall back to Windows WMI
            if not devices:
                devices = query_intel_gpus_windows()

            if devices and self.device_id < len(devices):
                self._device_info = devices[self.device_id]
            else:
                # Create default info
                self._device_info = IntelDeviceInfo(
                    device_id=self.device_id,
                    name="Intel GPU",
                    gpu_type=IntelGPUType.UNKNOWN,
                    total_memory_mb=2048,
                    free_memory_mb=2048,
                )

        return {
            "name": self._device_info.name,
            "gpu_type": self._device_info.gpu_type,
            "total_memory_mb": self._device_info.total_memory_mb,
            "free_memory_mb": self._device_info.free_memory_mb,
            "is_discrete": self._device_info.is_discrete,
            "driver_version": self._device_info.driver_version,
            "supports_fp16": self._device_info.supports_fp16,
            "supports_bf16": self._device_info.supports_bf16,
        }

    def initialize(self) -> bool:
        """Initialize the Intel GPU backend.

        Attempts to initialize in order:
        1. Intel Extension for PyTorch (IPEX) with XPU
        2. OpenVINO with GPU
        3. CPU fallback (if configured)

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        with self._lock:
            try:
                # Get device info first
                _ = self.get_device_info()

                # Try IPEX XPU
                if self._init_ipex_xpu():
                    self._initialized = True
                    logger.info(
                        f"Intel XPU backend initialized on device {self.device_id} "
                        f"({self._device_info.name if self._device_info else 'Unknown'})"
                    )
                    return True

                # Try OpenVINO
                if self.config.use_openvino and self._init_openvino():
                    self._initialized = True
                    self._using_openvino = True
                    logger.info(
                        f"Intel OpenVINO backend initialized "
                        f"({self._device_info.name if self._device_info else 'Unknown'})"
                    )
                    return True

                # Fall back to CPU if configured
                if self.config.fallback_to_cpu:
                    logger.warning("Intel GPU not available, falling back to CPU")
                    self._using_cpu_fallback = True
                    torch = _get_torch()
                    if torch is not None:
                        self._torch = torch
                        self._device = torch.device("cpu")
                        self._initialized = True
                        return True

                logger.error("Intel backend initialization failed")
                return False

            except Exception as e:
                logger.error(f"Intel backend initialization error: {e}")
                return False

    def _init_ipex_xpu(self) -> bool:
        """Initialize Intel Extension for PyTorch with XPU."""
        torch = _get_torch()
        ipex = _get_ipex()

        if torch is None or ipex is None:
            return False

        try:
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                return False

            device_count = torch.xpu.device_count()
            if self.device_id >= device_count:
                logger.warning(f"Device {self.device_id} not available (found {device_count})")
                return False

            self._torch = torch
            self._ipex = ipex
            self._device = torch.device(f"xpu:{self.device_id}")
            torch.xpu.set_device(self._device)

            # Warm up
            _ = torch.zeros(1, device=self._device)

            return True

        except Exception as e:
            logger.debug(f"IPEX XPU init failed: {e}")
            return False

    def _init_openvino(self) -> bool:
        """Initialize OpenVINO with GPU device."""
        core = _get_openvino()

        if core is None:
            return False

        try:
            available = core.available_devices
            if not any(d.startswith("GPU") for d in available):
                return False

            self._openvino_core = core
            return True

        except Exception as e:
            logger.debug(f"OpenVINO init failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up backend resources."""
        with self._lock:
            # Unload all models
            for name in list(self._models.keys()):
                self.unload_model(name)

            # Clear GPU memory
            if self._torch is not None and self._device is not None:
                if hasattr(self._torch, "xpu") and str(self._device).startswith("xpu"):
                    self._torch.xpu.empty_cache()

            self._device = None
            self._openvino_core = None
            self._initialized = False
            gc.collect()

    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities.

        Returns:
            BackendCapabilities describing Intel GPU support
        """
        if self._capabilities is not None:
            return self._capabilities

        device_info = self.get_device_info()

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.ONEAPI,
            vendor=GPUVendor.INTEL,
            supports_fp16=device_info.get("supports_fp16", True),
            supports_fp32=True,
            supports_int8=True,  # OpenVINO supports INT8
            supports_dynamic_shapes=True,
            supports_batching=True,
            max_memory_mb=device_info.get("total_memory_mb", 2048),
            recommended_memory_mb=int(device_info.get("total_memory_mb", 2048) * 0.7),
            max_batch_size=8 if device_info.get("is_discrete") else 4,
            max_tile_size=384 if device_info.get("is_discrete") else 256,
            supported_models=["realesrgan", "esrgan", "swinir"],
        )

        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        """Check if memory can be allocated.

        Args:
            size_mb: Amount of memory to allocate in MB

        Returns:
            True if sufficient memory is available
        """
        mem_info = self.get_memory_info()
        return mem_info["free_mb"] >= size_mb

    def free_memory(self) -> None:
        """Free allocated GPU memory."""
        if self._torch is not None and hasattr(self._torch, "xpu"):
            try:
                self._torch.xpu.empty_cache()
            except Exception:
                pass
        gc.collect()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information.

        Returns:
            Dict with 'total_mb', 'used_mb', 'free_mb' keys
        """
        if self._using_cpu_fallback:
            info = get_hardware_info()
            return {
                "total_mb": info.ram_total_mb,
                "used_mb": info.ram_total_mb - info.ram_available_mb,
                "free_mb": info.ram_available_mb,
            }

        if self._torch is not None and hasattr(self._torch, "xpu"):
            try:
                props = self._torch.xpu.get_device_properties(self.device_id)
                total = props.total_memory if hasattr(props, "total_memory") else 0

                if total > 0:
                    allocated = self._torch.xpu.memory_allocated(self.device_id)
                    return {
                        "total_mb": total / (1024 * 1024),
                        "used_mb": allocated / (1024 * 1024),
                        "free_mb": (total - allocated) / (1024 * 1024),
                    }
            except Exception:
                pass

        # Fallback to device info
        device_info = self.get_device_info()
        return {
            "total_mb": device_info.get("total_memory_mb", 0),
            "used_mb": 0,
            "free_mb": device_info.get("free_memory_mb", 0),
        }

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        """Load a model for inference.

        Args:
            model_name: Name/identifier of the model
            model_path: Path to model weights (OpenVINO XML/ONNX or PyTorch)
            **kwargs: Additional options:
                - model: Pre-loaded PyTorch model
                - use_openvino: Force OpenVINO for this model

        Returns:
            True if model loaded successfully
        """
        with self._lock:
            try:
                # Check if model already loaded
                if model_name in self._models:
                    logger.debug(f"Model {model_name} already loaded")
                    return True

                # Handle pre-loaded PyTorch model
                if "model" in kwargs:
                    return self._load_pytorch_model(model_name, kwargs["model"])

                # Load from path
                if model_path is None:
                    logger.error(f"No model path provided for {model_name}")
                    return False

                model_path = Path(model_path)
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return False

                # Determine model type from extension
                suffix = model_path.suffix.lower()
                use_openvino = kwargs.get("use_openvino", self._using_openvino)

                if suffix in (".xml",) or (suffix == ".onnx" and use_openvino):
                    return self._load_openvino_model(model_name, model_path)
                elif suffix in (".pt", ".pth"):
                    return self._load_pytorch_file(model_name, model_path)
                elif suffix == ".onnx":
                    return self._load_openvino_model(model_name, model_path)
                else:
                    logger.error(f"Unsupported model format: {suffix}")
                    return False

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return False

    def _load_pytorch_model(self, model_name: str, model: Any) -> bool:
        """Load a pre-loaded PyTorch model."""
        if self._torch is None:
            logger.error("PyTorch not available")
            return False

        try:
            model = model.to(self._device)
            model.eval()

            # Apply IPEX optimization
            if self._ipex is not None and not self._using_cpu_fallback:
                try:
                    dtype = self._torch.float16 if self.config.precision == "fp16" else self._torch.float32
                    if self.config.precision == "bf16":
                        dtype = self._torch.bfloat16
                    model = self._ipex.optimize(model, dtype=dtype)
                except Exception as e:
                    logger.debug(f"IPEX optimization failed: {e}")

            self._models[model_name] = IntelModelWrapper(
                name=model_name,
                model=model,
                model_type="pytorch",
                device=self._device,
                config=self.config,
            )

            logger.info(f"Loaded PyTorch model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False

    def _load_pytorch_file(self, model_name: str, model_path: Path) -> bool:
        """Load a PyTorch model from file."""
        if self._torch is None:
            logger.error("PyTorch not available")
            return False

        try:
            model = self._torch.load(model_path, map_location=self._device)
            if hasattr(model, "eval"):
                model.eval()

            return self._load_pytorch_model(model_name, model)

        except Exception as e:
            logger.error(f"Failed to load PyTorch file: {e}")
            return False

    def _load_openvino_model(self, model_name: str, model_path: Path) -> bool:
        """Load an OpenVINO model."""
        if self._openvino_core is None:
            # Try to initialize OpenVINO
            self._openvino_core = _get_openvino()
            if self._openvino_core is None:
                logger.error("OpenVINO not available")
                return False

        try:
            # Read model
            model = self._openvino_core.read_model(str(model_path))

            # Determine device
            device = self.config.openvino_device
            if self._using_cpu_fallback:
                device = "CPU"

            # Compile model
            compiled_model = self._openvino_core.compile_model(model, device)

            wrapper = IntelModelWrapper(
                name=model_name,
                model=model,
                model_type="openvino",
                device=device,
                config=self.config,
            )
            wrapper._compiled_model = compiled_model

            self._models[model_name] = wrapper

            logger.info(f"Loaded OpenVINO model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            return False

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory.

        Args:
            model_name: Name of model to unload
        """
        with self._lock:
            if model_name in self._models:
                self._models[model_name].cleanup()
                del self._models[model_name]
                self.free_memory()
                logger.debug(f"Unloaded model: {model_name}")

    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        """Run inference with a loaded model.

        Args:
            model_name: Name of model to use
            inputs: Input data (numpy array or torch tensor)
            **kwargs: Additional options (unused)

        Returns:
            Model output as numpy array

        Raises:
            ValueError: If model not loaded
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not loaded")

        return self._models[model_name](inputs)

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._models.keys())

    def is_using_fallback(self) -> bool:
        """Check if backend is using CPU fallback."""
        return self._using_cpu_fallback

    def is_using_openvino(self) -> bool:
        """Check if backend is using OpenVINO."""
        return self._using_openvino


# =============================================================================
# Factory Functions
# =============================================================================

def create_intel_backend(
    device_id: int = 0,
    precision: str = "fp16",
    use_openvino: bool = True,
    fallback_to_cpu: bool = True,
    **kwargs,
) -> Optional[IntelBackend]:
    """Create and initialize an Intel GPU backend.

    Args:
        device_id: GPU device index
        precision: Computation precision ("fp32", "fp16", "bf16")
        use_openvino: Enable OpenVINO for inference
        fallback_to_cpu: Fall back to CPU if GPU unavailable
        **kwargs: Additional IntelConfig options

    Returns:
        Initialized IntelBackend or None if initialization failed
    """
    config = IntelConfig(
        device_id=device_id,
        precision=precision,
        use_openvino=use_openvino,
        fallback_to_cpu=fallback_to_cpu,
        **kwargs,
    )

    backend = IntelBackend(config)

    if backend.is_available() and backend.initialize():
        return backend
    elif fallback_to_cpu:
        # Force CPU fallback
        backend._using_cpu_fallback = True
        if backend.initialize():
            return backend

    return None


def is_intel_xpu_available() -> bool:
    """Check if Intel XPU (via IPEX) is available.

    Returns:
        True if Intel XPU can be used
    """
    torch = _get_torch()
    ipex = _get_ipex()

    if torch is None or ipex is None:
        return False

    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def is_openvino_gpu_available() -> bool:
    """Check if OpenVINO GPU device is available.

    Returns:
        True if OpenVINO GPU can be used
    """
    core = _get_openvino()
    if core is None:
        return False

    try:
        available = core.available_devices
        return any(d.startswith("GPU") for d in available)
    except Exception:
        return False


def list_intel_gpus() -> List[Dict[str, Any]]:
    """List all Intel GPUs detected.

    Returns:
        List of dictionaries with GPU information
    """
    # Try multiple detection methods
    devices = query_intel_gpus_ipex()
    if not devices:
        devices = query_intel_gpus_openvino()
    if not devices:
        devices = query_intel_gpus_windows()

    return [
        {
            "device_id": d.device_id,
            "name": d.name,
            "gpu_type": d.gpu_type,
            "total_memory_mb": d.total_memory_mb,
            "is_discrete": d.is_discrete,
            "driver_version": d.driver_version,
        }
        for d in devices
    ]


def get_ipex_version() -> Optional[str]:
    """Get Intel Extension for PyTorch version.

    Returns:
        Version string or None if not installed
    """
    ipex = _get_ipex()
    if ipex is None:
        return None
    return getattr(ipex, "__version__", "unknown")


def get_openvino_version() -> Optional[str]:
    """Get OpenVINO version.

    Returns:
        Version string or None if not installed
    """
    try:
        from openvino.runtime import get_version
        return get_version()
    except ImportError:
        return None


# =============================================================================
# Backend Registration
# =============================================================================

try:
    register_backend(BackendType.ONEAPI, IntelBackend)
except Exception:
    pass


__all__ = [
    "IntelConfig",
    "IntelDeviceInfo",
    "IntelGPUType",
    "IntelBackend",
    "IntelModelWrapper",
    "create_intel_backend",
    "is_intel_xpu_available",
    "is_openvino_gpu_available",
    "list_intel_gpus",
    "get_ipex_version",
    "get_openvino_version",
    "detect_intel_gpu_type",
    "query_intel_gpus_ipex",
    "query_intel_gpus_openvino",
    "query_intel_gpus_windows",
    "INTEL_VRAM_ESTIMATES",
]
