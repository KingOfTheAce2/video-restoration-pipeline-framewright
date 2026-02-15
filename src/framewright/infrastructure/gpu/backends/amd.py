"""AMD ROCm Backend for FrameWright.

Provides GPU acceleration on AMD GPUs using ROCm (Linux) and HIP,
with fallback support via ONNX Runtime ROCm provider.

The backend supports:
- ROCm-enabled PyTorch for native tensor operations
- ONNX Runtime with ROCm ExecutionProvider
- Automatic fallback to CPU when ROCm is unavailable
"""

import gc
import logging
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..detector import BackendType, GPUVendor, get_hardware_info
from .base import Backend, BackendCapabilities, register_backend

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_torch = None
_torch_checked = False
_onnxruntime = None
_onnxruntime_checked = False


def _get_torch():
    """Lazy import PyTorch with ROCm support."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch


def _get_onnxruntime():
    """Lazy import ONNX Runtime."""
    global _onnxruntime, _onnxruntime_checked
    if not _onnxruntime_checked:
        try:
            import onnxruntime as ort
            _onnxruntime = ort
        except ImportError:
            _onnxruntime = None
        _onnxruntime_checked = True
    return _onnxruntime


@dataclass
class AMDConfig:
    """Configuration for AMD ROCm backend.

    Attributes:
        device_id: GPU device index to use (default 0)
        use_rocm: Enable ROCm acceleration when available
        precision: Computation precision ("fp32", "fp16")
        memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        enable_hip_graphs: Use HIP graphs for optimization
        cache_dir: Directory for caching compiled models
        fallback_to_cpu: Fall back to CPU if ROCm unavailable
        onnx_optimization_level: ONNX Runtime optimization level
    """
    device_id: int = 0
    use_rocm: bool = True
    precision: str = "fp16"
    memory_fraction: float = 0.9
    enable_hip_graphs: bool = True
    cache_dir: Optional[Path] = None
    fallback_to_cpu: bool = True
    onnx_optimization_level: int = 99  # ORT_ENABLE_ALL

    def __post_init__(self):
        """Validate and set up configuration."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".framewright" / "cache" / "rocm"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.precision not in ("fp32", "fp16"):
            raise ValueError(f"Invalid precision: {self.precision}. Must be 'fp32' or 'fp16'")

        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError(f"memory_fraction must be in (0.0, 1.0], got {self.memory_fraction}")


@dataclass
class AMDDeviceInfo:
    """Information about an AMD GPU device.

    Attributes:
        device_id: Device index
        name: GPU name (e.g., "AMD Radeon RX 7900 XTX")
        total_memory_mb: Total VRAM in MB
        free_memory_mb: Available VRAM in MB
        compute_units: Number of compute units
        gfx_version: Graphics version (e.g., "gfx1100")
        driver_version: ROCm driver version
        hip_version: HIP runtime version
    """
    device_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int = 0
    compute_units: int = 0
    gfx_version: str = ""
    driver_version: str = ""
    hip_version: str = ""


def detect_rocm_installation() -> Dict[str, Any]:
    """Detect ROCm installation and version.

    Returns:
        Dictionary with ROCm installation info:
        - installed: bool
        - version: str or None
        - path: str or None
        - hip_version: str or None
    """
    result = {
        "installed": False,
        "version": None,
        "path": None,
        "hip_version": None,
    }

    # Check for rocm-smi
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        # Check common ROCm installation paths
        common_paths = [
            "/opt/rocm/bin/rocm-smi",
            "/opt/rocm-5.7.0/bin/rocm-smi",
            "/opt/rocm-5.6.0/bin/rocm-smi",
        ]
        for path in common_paths:
            if Path(path).exists():
                rocm_smi = path
                break

    if not rocm_smi:
        return result

    result["path"] = str(Path(rocm_smi).parent.parent)

    # Get ROCm version
    try:
        version_cmd = subprocess.run(
            [rocm_smi, "--showversion"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if version_cmd.returncode == 0:
            for line in version_cmd.stdout.split("\n"):
                if "ROCm" in line or "version" in line.lower():
                    # Parse version from output
                    parts = line.split()
                    for part in parts:
                        if part[0].isdigit():
                            result["version"] = part
                            break
            result["installed"] = True
    except Exception as e:
        logger.debug(f"Failed to get ROCm version: {e}")

    # Get HIP version
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            hip_cmd = subprocess.run(
                [hipconfig, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if hip_cmd.returncode == 0:
                result["hip_version"] = hip_cmd.stdout.strip()
        except Exception:
            pass

    return result


def query_amd_gpus_rocm_smi() -> List[AMDDeviceInfo]:
    """Query AMD GPU information using rocm-smi.

    Returns:
        List of AMDDeviceInfo for each detected AMD GPU
    """
    devices = []

    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return devices

    try:
        # Query GPU info
        cmd = subprocess.run(
            [rocm_smi, "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if cmd.returncode != 0:
            # Try without JSON
            cmd = subprocess.run(
                [rocm_smi, "-d", "0", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if cmd.returncode == 0:
                # Parse text output
                name = "AMD GPU"
                for line in cmd.stdout.split("\n"):
                    if "GPU" in line and ":" in line:
                        name = line.split(":")[-1].strip()
                        break

                devices.append(AMDDeviceInfo(
                    device_id=0,
                    name=name,
                    total_memory_mb=_estimate_amd_vram(name),
                    free_memory_mb=_estimate_amd_vram(name),
                ))
            return devices

        # Parse JSON output
        import json
        data = json.loads(cmd.stdout)

        for card_key, card_data in data.items():
            if not card_key.startswith("card"):
                continue

            device_id = int(card_key.replace("card", ""))
            name = card_data.get("Card series", "AMD GPU")

            # Get memory info
            total_mb = 0
            free_mb = 0
            if "VRAM Total Memory (B)" in card_data:
                total_mb = int(card_data["VRAM Total Memory (B)"]) // (1024 * 1024)
            if "VRAM Total Used Memory (B)" in card_data:
                used_mb = int(card_data["VRAM Total Used Memory (B)"]) // (1024 * 1024)
                free_mb = total_mb - used_mb

            devices.append(AMDDeviceInfo(
                device_id=device_id,
                name=name,
                total_memory_mb=total_mb if total_mb > 0 else _estimate_amd_vram(name),
                free_memory_mb=free_mb if free_mb > 0 else total_mb,
            ))

    except Exception as e:
        logger.debug(f"rocm-smi query failed: {e}")

    return devices


def _estimate_amd_vram(name: str) -> int:
    """Estimate VRAM for AMD GPU based on model name."""
    name_lower = name.lower() if name else ""

    # RX 7000 series
    if "7900" in name_lower:
        return 24576 if "xtx" in name_lower else 20480
    if "7800" in name_lower:
        return 16384
    if "7700" in name_lower:
        return 12288
    if "7600" in name_lower:
        return 8192

    # RX 6000 series
    if "6900" in name_lower or "6950" in name_lower:
        return 16384
    if "6800" in name_lower:
        return 16384
    if "6700" in name_lower:
        return 12288
    if "6650" in name_lower or "6600" in name_lower:
        return 8192
    if "6500" in name_lower:
        return 4096
    if "6400" in name_lower:
        return 4096

    # RX 5000 series
    if "5700" in name_lower:
        return 8192
    if "5600" in name_lower:
        return 6144
    if "5500" in name_lower:
        return 4096

    # Vega
    if "vega" in name_lower:
        if "64" in name_lower:
            return 8192
        if "56" in name_lower:
            return 8192
        return 8192

    # Default
    return 8192


class AMDModelWrapper:
    """Wrapper for models loaded on AMD GPU.

    Handles both PyTorch and ONNX models with a unified interface.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        model_type: str,  # "pytorch" or "onnx"
        device: Any,
        config: AMDConfig,
    ):
        self.name = name
        self.model = model
        self.model_type = model_type
        self.device = device
        self.config = config
        self._lock = threading.Lock()

    def __call__(self, inputs: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Run inference on the model."""
        with self._lock:
            if self.model_type == "pytorch":
                return self._run_pytorch(inputs)
            elif self.model_type == "onnx":
                return self._run_onnx(inputs)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

    def _run_pytorch(self, inputs: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Run PyTorch model inference."""
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

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

        # Convert back to numpy
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().float().numpy()
        elif isinstance(outputs, (list, tuple)):
            return outputs[0].cpu().float().numpy()
        return outputs

    def _run_onnx(self, inputs: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """Run ONNX model inference."""
        torch = _get_torch()

        # Convert to numpy if needed
        if torch is not None and isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()

        # Get input name
        input_name = self.model.get_inputs()[0].name

        # Run inference
        outputs = self.model.run(None, {input_name: inputs})
        return outputs[0]

    def cleanup(self):
        """Release model resources."""
        self.model = None
        gc.collect()


class AMDBackend(Backend):
    """AMD ROCm backend for GPU acceleration.

    Supports AMD GPUs via ROCm-enabled PyTorch and ONNX Runtime.
    Falls back to CPU when ROCm is not available.

    Example:
        >>> config = AMDConfig(device_id=0, precision="fp16")
        >>> backend = AMDBackend(config)
        >>> if backend.initialize():
        ...     backend.load_model("upscaler", "/path/to/model.onnx")
        ...     result = backend.run_inference("upscaler", input_tensor)
        ...     backend.cleanup()
    """

    def __init__(self, config: Optional[AMDConfig] = None, device_id: int = 0):
        """Initialize AMD backend.

        Args:
            config: Backend configuration (created with defaults if None)
            device_id: GPU device index (overridden by config if provided)
        """
        super().__init__(device_id if config is None else config.device_id)
        self.config = config or AMDConfig(device_id=device_id)
        self._torch = None
        self._device = None
        self._models: Dict[str, AMDModelWrapper] = {}
        self._device_info: Optional[AMDDeviceInfo] = None
        self._rocm_info: Dict[str, Any] = {}
        self._ort_session_options = None
        self._lock = threading.Lock()
        self._using_cpu_fallback = False

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.ROCM

    @property
    def name(self) -> str:
        """Get human-readable backend name."""
        if self._using_cpu_fallback:
            return "AMD ROCm (CPU Fallback)"
        return "AMD ROCm (HIP)"

    def is_available(self) -> bool:
        """Check if ROCm/HIP is available.

        Checks for:
        1. Linux operating system (ROCm is Linux-only)
        2. ROCm installation via rocm-smi
        3. PyTorch with ROCm support OR ONNX Runtime with ROCm provider

        Returns:
            True if ROCm backend can be used
        """
        # ROCm is currently Linux-only
        if platform.system() != "Linux":
            logger.debug("ROCm is only available on Linux")
            return False

        # Check ROCm installation
        self._rocm_info = detect_rocm_installation()
        if not self._rocm_info.get("installed"):
            logger.debug("ROCm not installed")
            return False

        # Check for ROCm-enabled PyTorch
        torch = _get_torch()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    # Check if it's actually ROCm (HIP)
                    if hasattr(torch.version, "hip") and torch.version.hip:
                        return True
                    # Some ROCm builds report CUDA but are actually HIP
                    if "rocm" in str(torch.__version__).lower():
                        return True
            except Exception:
                pass

        # Check for ONNX Runtime ROCm provider
        ort = _get_onnxruntime()
        if ort is not None:
            providers = ort.get_available_providers()
            if "ROCMExecutionProvider" in providers:
                return True

        return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the AMD GPU.

        Returns:
            Dictionary containing:
            - name: GPU name
            - total_memory_mb: Total VRAM
            - free_memory_mb: Available VRAM
            - compute_units: Number of CUs
            - gfx_version: Graphics version
            - driver_version: Driver version
            - hip_version: HIP version
            - rocm_version: ROCm version
        """
        if self._device_info is None:
            devices = query_amd_gpus_rocm_smi()
            if devices and self.device_id < len(devices):
                self._device_info = devices[self.device_id]
            else:
                # Create default info
                self._device_info = AMDDeviceInfo(
                    device_id=self.device_id,
                    name="AMD GPU",
                    total_memory_mb=8192,
                    free_memory_mb=8192,
                )

        return {
            "name": self._device_info.name,
            "total_memory_mb": self._device_info.total_memory_mb,
            "free_memory_mb": self._device_info.free_memory_mb,
            "compute_units": self._device_info.compute_units,
            "gfx_version": self._device_info.gfx_version,
            "driver_version": self._device_info.driver_version,
            "hip_version": self._rocm_info.get("hip_version", ""),
            "rocm_version": self._rocm_info.get("version", ""),
        }

    def initialize(self) -> bool:
        """Initialize the AMD ROCm backend.

        Sets up PyTorch with ROCm or falls back to CPU if configured.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        with self._lock:
            try:
                # Get device info first
                _ = self.get_device_info()

                # Try PyTorch ROCm
                torch = _get_torch()
                if torch is not None:
                    if self._init_pytorch_rocm(torch):
                        self._initialized = True
                        logger.info(
                            f"AMD ROCm backend initialized on device {self.device_id} "
                            f"({self._device_info.name if self._device_info else 'Unknown'})"
                        )
                        return True

                # Try ONNX Runtime ROCm
                ort = _get_onnxruntime()
                if ort is not None:
                    if self._init_onnx_rocm(ort):
                        self._initialized = True
                        logger.info("AMD backend initialized with ONNX Runtime ROCm")
                        return True

                # Fall back to CPU if configured
                if self.config.fallback_to_cpu:
                    logger.warning("ROCm not available, falling back to CPU")
                    self._using_cpu_fallback = True
                    if torch is not None:
                        self._torch = torch
                        self._device = torch.device("cpu")
                        self._initialized = True
                        return True

                logger.error("AMD ROCm backend initialization failed")
                return False

            except Exception as e:
                logger.error(f"AMD ROCm initialization error: {e}")
                return False

    def _init_pytorch_rocm(self, torch) -> bool:
        """Initialize PyTorch with ROCm/HIP support."""
        try:
            if not torch.cuda.is_available():
                return False

            # Verify ROCm/HIP
            if not (hasattr(torch.version, "hip") and torch.version.hip):
                return False

            device_count = torch.cuda.device_count()
            if self.device_id >= device_count:
                logger.warning(f"Device {self.device_id} not available (found {device_count})")
                return False

            self._torch = torch
            self._device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self._device)

            # Warm up
            _ = torch.zeros(1, device=self._device)

            # Set memory fraction
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction,
                    self.device_id
                )

            return True

        except Exception as e:
            logger.debug(f"PyTorch ROCm init failed: {e}")
            return False

    def _init_onnx_rocm(self, ort) -> bool:
        """Initialize ONNX Runtime with ROCm provider."""
        try:
            providers = ort.get_available_providers()
            if "ROCMExecutionProvider" not in providers:
                return False

            # Configure session options
            self._ort_session_options = ort.SessionOptions()
            self._ort_session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel(self.config.onnx_optimization_level)
            )

            return True

        except Exception as e:
            logger.debug(f"ONNX Runtime ROCm init failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up backend resources."""
        with self._lock:
            # Unload all models
            for name in list(self._models.keys()):
                self.unload_model(name)

            # Clear GPU memory
            if self._torch is not None and self._device is not None:
                if str(self._device).startswith("cuda"):
                    self._torch.cuda.empty_cache()

            self._device = None
            self._initialized = False
            gc.collect()

    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities.

        Returns:
            BackendCapabilities describing AMD ROCm support
        """
        if self._capabilities is not None:
            return self._capabilities

        device_info = self.get_device_info()

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.ROCM,
            vendor=GPUVendor.AMD,
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=False,  # Limited INT8 support on AMD
            supports_dynamic_shapes=True,
            supports_batching=True,
            max_memory_mb=device_info.get("total_memory_mb", 8192),
            recommended_memory_mb=int(device_info.get("total_memory_mb", 8192) * 0.7),
            max_batch_size=16,
            max_tile_size=512,
            supported_models=["realesrgan", "esrgan", "swinir", "restoreformer"],
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
        if self._torch is not None and not self._using_cpu_fallback:
            self._torch.cuda.empty_cache()
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

        if self._torch is not None and self._device is not None:
            try:
                total = self._torch.cuda.get_device_properties(self.device_id).total_memory
                allocated = self._torch.cuda.memory_allocated(self.device_id)
                reserved = self._torch.cuda.memory_reserved(self.device_id)

                return {
                    "total_mb": total / (1024 * 1024),
                    "used_mb": allocated / (1024 * 1024),
                    "reserved_mb": reserved / (1024 * 1024),
                    "free_mb": (total - reserved) / (1024 * 1024),
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
            model_path: Path to model weights (ONNX or PyTorch)
            **kwargs: Additional options:
                - model: Pre-loaded PyTorch model
                - input_shape: Expected input shape for tracing

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
                if suffix in (".onnx",):
                    return self._load_onnx_model(model_name, model_path)
                elif suffix in (".pt", ".pth"):
                    return self._load_pytorch_file(model_name, model_path)
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

            if self.config.precision == "fp16" and not self._using_cpu_fallback:
                model = model.half()

            self._models[model_name] = AMDModelWrapper(
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

    def _load_onnx_model(self, model_name: str, model_path: Path) -> bool:
        """Load an ONNX model."""
        ort = _get_onnxruntime()
        if ort is None:
            logger.error("ONNX Runtime not available")
            return False

        try:
            # Set up providers
            providers = []
            if not self._using_cpu_fallback:
                providers.append(("ROCMExecutionProvider", {
                    "device_id": self.device_id,
                }))
            providers.append("CPUExecutionProvider")

            # Create session
            session = ort.InferenceSession(
                str(model_path),
                sess_options=self._ort_session_options,
                providers=providers,
            )

            self._models[model_name] = AMDModelWrapper(
                name=model_name,
                model=session,
                model_type="onnx",
                device=None,
                config=self.config,
            )

            logger.info(f"Loaded ONNX model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
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


# =============================================================================
# Factory Functions
# =============================================================================

def create_amd_backend(
    device_id: int = 0,
    precision: str = "fp16",
    fallback_to_cpu: bool = True,
    **kwargs,
) -> Optional[AMDBackend]:
    """Create and initialize an AMD ROCm backend.

    Args:
        device_id: GPU device index
        precision: Computation precision ("fp32" or "fp16")
        fallback_to_cpu: Fall back to CPU if ROCm unavailable
        **kwargs: Additional AMDConfig options

    Returns:
        Initialized AMDBackend or None if initialization failed
    """
    config = AMDConfig(
        device_id=device_id,
        precision=precision,
        fallback_to_cpu=fallback_to_cpu,
        **kwargs,
    )

    backend = AMDBackend(config)

    if backend.is_available() and backend.initialize():
        return backend
    elif fallback_to_cpu:
        # Force CPU fallback
        backend._using_cpu_fallback = True
        if backend.initialize():
            return backend

    return None


def is_rocm_available() -> bool:
    """Check if ROCm is available on this system.

    Returns:
        True if ROCm can be used
    """
    if platform.system() != "Linux":
        return False

    rocm_info = detect_rocm_installation()
    return rocm_info.get("installed", False)


def get_rocm_version() -> Optional[str]:
    """Get the installed ROCm version.

    Returns:
        Version string or None if ROCm not installed
    """
    rocm_info = detect_rocm_installation()
    return rocm_info.get("version")


def list_amd_gpus() -> List[Dict[str, Any]]:
    """List all AMD GPUs detected via rocm-smi.

    Returns:
        List of dictionaries with GPU information
    """
    devices = query_amd_gpus_rocm_smi()
    return [
        {
            "device_id": d.device_id,
            "name": d.name,
            "total_memory_mb": d.total_memory_mb,
            "free_memory_mb": d.free_memory_mb,
            "compute_units": d.compute_units,
            "gfx_version": d.gfx_version,
        }
        for d in devices
    ]


# =============================================================================
# Backend Registration
# =============================================================================

try:
    register_backend(BackendType.ROCM, AMDBackend)
except Exception:
    pass


__all__ = [
    "AMDConfig",
    "AMDDeviceInfo",
    "AMDBackend",
    "AMDModelWrapper",
    "create_amd_backend",
    "is_rocm_available",
    "get_rocm_version",
    "list_amd_gpus",
    "detect_rocm_installation",
    "query_amd_gpus_rocm_smi",
]
