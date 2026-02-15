"""TensorRT Acceleration Backend for FrameWright.

Provides TensorRT-accelerated inference for NVIDIA GPUs, offering up to 2x
faster inference compared to PyTorch with optimized memory usage.

Features:
- Convert PyTorch models to TensorRT engines via ONNX
- Support FP32, FP16, and INT8 precision modes
- Dynamic batch sizes and input shapes
- Automatic caching of compiled engines
- Graceful fallback to PyTorch if TensorRT unavailable

Usage:
    >>> backend = create_tensorrt_backend(precision="fp16")
    >>> if backend and backend.is_available():
    ...     engine_path = backend.convert_model(model, input_shape, "my_model")
    ...     result = backend.infer("my_model", input_data)

Requirements:
    - NVIDIA GPU with compute capability >= 6.0
    - TensorRT 8.x or later
    - torch, onnx, onnxruntime (for conversion)
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..detector import BackendType, GPUVendor, get_hardware_info
from .base import Backend, BackendCapabilities, register_backend

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_torch = None
_torch_checked = False
_tensorrt = None
_tensorrt_checked = False
_onnx = None
_onnx_checked = False
_cuda = None
_cuda_checked = False


def _get_torch():
    """Lazy load torch."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch


def _get_tensorrt():
    """Lazy load tensorrt."""
    global _tensorrt, _tensorrt_checked
    if not _tensorrt_checked:
        try:
            import tensorrt as trt
            _tensorrt = trt
        except ImportError:
            _tensorrt = None
        _tensorrt_checked = True
    return _tensorrt


def _get_onnx():
    """Lazy load onnx."""
    global _onnx, _onnx_checked
    if not _onnx_checked:
        try:
            import onnx
            _onnx = onnx
        except ImportError:
            _onnx = None
        _onnx_checked = True
    return _onnx


def _get_cuda():
    """Lazy load pycuda for memory management."""
    global _cuda, _cuda_checked
    if not _cuda_checked:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            _cuda = cuda
        except ImportError:
            _cuda = None
        _cuda_checked = True
    return _cuda


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TensorRTConfig:
    """Configuration for TensorRT backend.

    Attributes:
        precision: Inference precision mode (fp32, fp16, int8)
        max_batch_size: Maximum batch size for engine optimization
        max_workspace_size_mb: Maximum GPU memory for TensorRT workspace
        cache_dir: Directory for caching compiled engines
        force_rebuild: Force rebuilding engines even if cached
        dynamic_shapes: Enable dynamic input shapes
        min_batch_size: Minimum batch size for dynamic shapes
        opt_batch_size: Optimal batch size for dynamic shapes
        calibration_data: Calibration data for INT8 quantization
        strict_types: Enforce strict type constraints
        dla_core: DLA core for Jetson devices (-1 to disable)
    """
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    max_batch_size: int = 1
    max_workspace_size_mb: int = 1024
    cache_dir: Optional[Path] = None
    force_rebuild: bool = False
    dynamic_shapes: bool = True
    min_batch_size: int = 1
    opt_batch_size: int = 1
    calibration_data: Optional[np.ndarray] = None
    strict_types: bool = False
    dla_core: int = -1

    def __post_init__(self):
        """Initialize default cache directory."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".framewright" / "cache" / "tensorrt"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate precision
        if self.precision not in ("fp32", "fp16", "int8"):
            raise ValueError(f"Invalid precision: {self.precision}")

        # Validate batch sizes
        if self.min_batch_size > self.max_batch_size:
            self.min_batch_size = self.max_batch_size
        if self.opt_batch_size > self.max_batch_size:
            self.opt_batch_size = self.max_batch_size
        if self.opt_batch_size < self.min_batch_size:
            self.opt_batch_size = self.min_batch_size


# =============================================================================
# TensorRT Engine Wrapper
# =============================================================================

class TRTEngine:
    """Wrapper for TensorRT engine with execution context.

    Manages engine lifecycle, memory allocation, and inference execution.
    """

    def __init__(self, engine_path: Path, device_id: int = 0):
        """Load a TensorRT engine from file.

        Args:
            engine_path: Path to serialized engine file
            device_id: CUDA device to use
        """
        self.engine_path = engine_path
        self.device_id = device_id
        self._engine = None
        self._context = None
        self._bindings = []
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}
        self._dtype_map: Dict[str, np.dtype] = {}
        self._stream = None
        self._lock = threading.Lock()

        self._load_engine()

    def _load_engine(self) -> None:
        """Load and initialize the TensorRT engine."""
        trt = _get_tensorrt()
        if trt is None:
            raise RuntimeError("TensorRT not available")

        torch = _get_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available")

        # Set device
        torch.cuda.set_device(self.device_id)

        # Load engine from file
        logger.debug(f"Loading TensorRT engine from {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create execution context")

        # Create CUDA stream
        self._stream = torch.cuda.Stream(device=self.device_id)

        # Analyze bindings
        self._analyze_bindings()

        logger.info(f"Loaded TensorRT engine: inputs={self._input_names}, outputs={self._output_names}")

    def _analyze_bindings(self) -> None:
        """Analyze engine bindings to determine I/O configuration."""
        trt = _get_tensorrt()

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            shape = self._engine.get_tensor_shape(name)
            dtype = self._engine.get_tensor_dtype(name)

            # Convert TRT dtype to numpy
            dtype_map = {
                trt.DataType.FLOAT: np.float32,
                trt.DataType.HALF: np.float16,
                trt.DataType.INT8: np.int8,
                trt.DataType.INT32: np.int32,
                trt.DataType.BOOL: np.bool_,
            }
            np_dtype = dtype_map.get(dtype, np.float32)
            self._dtype_map[name] = np_dtype

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._input_shapes[name] = tuple(shape)
            else:
                self._output_names.append(name)
                self._output_shapes[name] = tuple(shape)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference with the engine.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available")

        with self._lock:
            with torch.cuda.device(self.device_id):
                return self._infer_impl(inputs)

    def _infer_impl(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Internal inference implementation."""
        torch = _get_torch()
        trt = _get_tensorrt()

        # Prepare input tensors
        input_tensors = {}
        for name in self._input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

            data = inputs[name]

            # Convert to correct dtype
            expected_dtype = self._dtype_map.get(name, np.float32)
            if data.dtype != expected_dtype:
                data = data.astype(expected_dtype)

            # Ensure contiguous
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            # Create CUDA tensor
            tensor = torch.from_numpy(data).cuda(self.device_id)
            input_tensors[name] = tensor

            # Set tensor address
            self._context.set_tensor_address(name, tensor.data_ptr())

            # Update shape for dynamic inputs
            shape = list(data.shape)
            self._context.set_input_shape(name, shape)

        # Allocate output tensors
        output_tensors = {}
        for name in self._output_names:
            shape = self._context.get_tensor_shape(name)
            dtype = self._dtype_map.get(name, np.float32)

            # Create output tensor
            torch_dtype = torch.float32
            if dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            elif dtype == np.int8:
                torch_dtype = torch.int8

            tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=f"cuda:{self.device_id}")
            output_tensors[name] = tensor
            self._context.set_tensor_address(name, tensor.data_ptr())

        # Execute inference
        success = self._context.execute_async_v3(self._stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference execution failed")

        # Synchronize
        self._stream.synchronize()

        # Convert outputs to numpy
        outputs = {}
        for name, tensor in output_tensors.items():
            outputs[name] = tensor.cpu().numpy()

        return outputs

    def get_binding_shape(self, name: str) -> Tuple[int, ...]:
        """Get the shape of a binding."""
        if name in self._input_shapes:
            return self._input_shapes[name]
        elif name in self._output_shapes:
            return self._output_shapes[name]
        else:
            raise ValueError(f"Unknown binding: {name}")

    def cleanup(self) -> None:
        """Release engine resources."""
        with self._lock:
            self._context = None
            self._engine = None
            self._stream = None

            torch = _get_torch()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# INT8 Calibrator
# =============================================================================

class Int8Calibrator:
    """INT8 calibration for TensorRT quantization.

    Uses a calibration dataset to determine optimal quantization ranges.
    """

    def __init__(
        self,
        data: np.ndarray,
        cache_file: Path,
        batch_size: int = 1,
    ):
        """Initialize calibrator.

        Args:
            data: Calibration data array
            cache_file: Path to cache calibration data
            batch_size: Calibration batch size
        """
        trt = _get_tensorrt()
        if trt is None:
            raise RuntimeError("TensorRT not available")

        self.data = data
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = None

        torch = _get_torch()
        if torch is not None:
            # Pre-allocate device memory
            sample = data[:batch_size]
            self.device_input = torch.from_numpy(sample.copy()).cuda()

    def get_batch_size(self) -> int:
        """Return calibration batch size."""
        return self.batch_size

    def get_batch(self, names: List[str]) -> List[int]:
        """Get next calibration batch.

        Args:
            names: Input tensor names

        Returns:
            List of device pointers for each input
        """
        if self.current_index >= len(self.data):
            return None

        torch = _get_torch()
        if torch is None:
            return None

        # Get batch
        end_index = min(self.current_index + self.batch_size, len(self.data))
        batch = self.data[self.current_index:end_index]
        self.current_index = end_index

        # Pad batch if needed
        if len(batch) < self.batch_size:
            pad_size = self.batch_size - len(batch)
            batch = np.concatenate([batch, batch[:pad_size]])

        # Copy to device
        self.device_input = torch.from_numpy(batch.copy()).cuda()

        return [self.device_input.data_ptr()]

    def read_calibration_cache(self) -> bytes:
        """Read cached calibration data."""
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration data to cache."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# =============================================================================
# TensorRT Backend
# =============================================================================

class TensorRTBackend(Backend):
    """TensorRT acceleration backend for NVIDIA GPUs.

    Provides optimized inference through TensorRT engine compilation
    with support for FP16/INT8 precision and dynamic shapes.

    Example:
        >>> backend = TensorRTBackend(TensorRTConfig(precision="fp16"))
        >>> backend.initialize()
        >>> engine_path = backend.convert_model(model, (1, 3, 256, 256), "upscaler")
        >>> result = backend.infer("upscaler", input_array)
    """

    def __init__(self, config: Optional[TensorRTConfig] = None, device_id: int = 0):
        """Initialize TensorRT backend.

        Args:
            config: TensorRT configuration
            device_id: CUDA device ID
        """
        super().__init__(device_id)
        self.config = config or TensorRTConfig()
        self._engines: Dict[str, TRTEngine] = {}
        self._engine_paths: Dict[str, Path] = {}
        self._lock = threading.Lock()
        self._trt_logger = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.TENSORRT

    @property
    def name(self) -> str:
        return "TensorRT"

    def is_available(self) -> bool:
        """Check if TensorRT is available and GPU supports it."""
        # Check TensorRT import
        trt = _get_tensorrt()
        if trt is None:
            logger.debug("TensorRT not installed")
            return False

        # Check PyTorch CUDA
        torch = _get_torch()
        if torch is None or not torch.cuda.is_available():
            logger.debug("PyTorch CUDA not available")
            return False

        # Check device availability
        if self.device_id >= torch.cuda.device_count():
            logger.debug(f"Device {self.device_id} not available")
            return False

        # Check compute capability (need >= 6.0 for TensorRT)
        props = torch.cuda.get_device_properties(self.device_id)
        if props.major < 6:
            logger.debug(f"GPU compute capability {props.major}.{props.minor} < 6.0")
            return False

        return True

    def initialize(self) -> bool:
        """Initialize the TensorRT backend."""
        if not self.is_available():
            return False

        try:
            trt = _get_tensorrt()
            torch = _get_torch()

            # Set device
            torch.cuda.set_device(self.device_id)

            # Create TensorRT logger
            self._trt_logger = trt.Logger(trt.Logger.WARNING)

            # Warm up CUDA
            _ = torch.zeros(1, device=f"cuda:{self.device_id}")

            self._initialized = True
            logger.info(f"TensorRT backend initialized on device {self.device_id}")

            return True

        except Exception as e:
            logger.error(f"TensorRT initialization failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up TensorRT resources."""
        with self._lock:
            # Clean up all engines
            for name, engine in list(self._engines.items()):
                engine.cleanup()
            self._engines.clear()
            self._engine_paths.clear()

        # Clear CUDA cache
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.debug("TensorRT backend cleaned up")

    def get_capabilities(self) -> BackendCapabilities:
        """Get TensorRT backend capabilities."""
        if self._capabilities is not None:
            return self._capabilities

        caps = BackendCapabilities(
            name=self.name,
            backend_type=BackendType.TENSORRT,
            vendor=GPUVendor.NVIDIA,
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_dynamic_shapes=True,
            supports_batching=True,
        )

        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.device_id)
                caps.max_memory_mb = int(props.total_memory / (1024 * 1024))
                caps.recommended_memory_mb = int(caps.max_memory_mb * 0.85)

                # INT8 requires Turing or later
                caps.supports_int8 = props.major >= 7

            except Exception as e:
                logger.debug(f"Failed to get device properties: {e}")

        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        """Check if memory can be allocated."""
        info = self.get_memory_info()
        return info.get("free_mb", 0) >= size_mb

    def free_memory(self) -> None:
        """Free GPU memory."""
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        torch = _get_torch()
        if torch is None or not torch.cuda.is_available():
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

        try:
            props = torch.cuda.get_device_properties(self.device_id)
            total = props.total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)

            return {
                "total_mb": total,
                "used_mb": allocated,
                "reserved_mb": reserved,
                "free_mb": total - reserved,
            }
        except Exception:
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        **kwargs,
    ) -> bool:
        """Load a TensorRT engine.

        Args:
            model_name: Unique identifier for the model
            model_path: Path to TensorRT engine file
            **kwargs: Additional options

        Returns:
            True if model loaded successfully
        """
        if model_path is None:
            # Check cache
            engine_path = self._get_cache_path(model_name)
            if not engine_path.exists():
                logger.error(f"No engine found for {model_name}")
                return False
            model_path = engine_path

        try:
            with self._lock:
                engine = TRTEngine(model_path, self.device_id)
                self._engines[model_name] = engine
                self._engine_paths[model_name] = model_path

            logger.info(f"Loaded TensorRT engine: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load engine {model_name}: {e}")
            return False

    def unload_model(self, model_name: str) -> None:
        """Unload a TensorRT engine."""
        with self._lock:
            if model_name in self._engines:
                self._engines[model_name].cleanup()
                del self._engines[model_name]
                self._engine_paths.pop(model_name, None)
                logger.debug(f"Unloaded engine: {model_name}")

    def run_inference(
        self,
        model_name: str,
        inputs: Any,
        **kwargs,
    ) -> Any:
        """Run inference with a loaded engine.

        Args:
            model_name: Name of loaded model
            inputs: Input data (numpy array or dict of arrays)
            **kwargs: Additional options

        Returns:
            Inference result
        """
        if model_name not in self._engines:
            raise ValueError(f"Model {model_name} not loaded")

        engine = self._engines[model_name]

        # Handle different input formats
        if isinstance(inputs, np.ndarray):
            # Single input - use first input name
            input_name = engine._input_names[0]
            inputs = {input_name: inputs}
        elif not isinstance(inputs, dict):
            raise ValueError("Inputs must be numpy array or dict of arrays")

        return engine.infer(inputs)

    def convert_model(
        self,
        model: "torch.nn.Module",
        input_shape: Tuple[int, ...],
        model_name: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> Path:
        """Convert PyTorch model to TensorRT engine.

        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape (e.g., (1, 3, 256, 256))
            model_name: Unique name for the engine
            input_names: Names for input tensors
            output_names: Names for output tensors

        Returns:
            Path to compiled TensorRT engine
        """
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available")

        # Generate cache key based on model and config
        cache_key = self._compute_cache_key(model, input_shape, model_name)
        engine_path = self._get_cache_path(cache_key)

        # Check cache
        if engine_path.exists() and not self.config.force_rebuild:
            logger.info(f"Using cached TensorRT engine: {engine_path}")
            return engine_path

        logger.info(f"Converting {model_name} to TensorRT (precision={self.config.precision})")

        # Export to ONNX first
        onnx_path = self.config.cache_dir / f"{cache_key}.onnx"

        try:
            # Prepare model
            model = model.eval()
            model = model.cuda(self.device_id)

            if self.config.precision == "fp16":
                model = model.half()

            # Create dummy input
            dummy_input = torch.randn(input_shape, device=f"cuda:{self.device_id}")
            if self.config.precision == "fp16":
                dummy_input = dummy_input.half()

            # Export to ONNX
            input_names = input_names or ["input"]
            output_names = output_names or ["output"]

            dynamic_axes = None
            if self.config.dynamic_shapes:
                dynamic_axes = {
                    input_names[0]: {0: "batch_size"},
                }
                for name in output_names:
                    dynamic_axes[name] = {0: "batch_size"}

            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
            )

            logger.debug(f"Exported ONNX model to {onnx_path}")

            # Convert ONNX to TensorRT
            engine_path = self.convert_onnx(onnx_path, cache_key, input_shape)

            # Clean up ONNX file
            if onnx_path.exists():
                onnx_path.unlink()

            return engine_path

        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise

    def convert_onnx(
        self,
        onnx_path: Path,
        model_name: str,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Path:
        """Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            model_name: Unique name for the engine
            input_shape: Optional input shape for optimization

        Returns:
            Path to compiled TensorRT engine
        """
        trt = _get_tensorrt()
        onnx_module = _get_onnx()

        if trt is None:
            raise RuntimeError("TensorRT not available")
        if onnx_module is None:
            raise RuntimeError("ONNX not available")

        engine_path = self._get_cache_path(model_name)

        # Check cache
        if engine_path.exists() and not self.config.force_rebuild:
            logger.info(f"Using cached engine: {engine_path}")
            return engine_path

        logger.info(f"Building TensorRT engine from {onnx_path}")
        start_time = time.time()

        # Create builder
        builder = trt.Builder(self._trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self._trt_logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parsing failed: {errors}")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.max_workspace_size_mb * 1024 * 1024
        )

        # Set precision
        if self.config.precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.debug("Enabled FP16 precision")
            else:
                logger.warning("FP16 not supported, using FP32")

        elif self.config.precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)

                # Set calibrator if provided
                if self.config.calibration_data is not None:
                    cache_file = self.config.cache_dir / f"{model_name}_calib.cache"
                    calibrator = Int8Calibrator(
                        self.config.calibration_data,
                        cache_file,
                        self.config.max_batch_size,
                    )
                    config.int8_calibrator = calibrator
                    logger.debug("Enabled INT8 with calibration")
                else:
                    # Fall back to FP16 if no calibration data
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.warning("No INT8 calibration data, using FP16")
            else:
                logger.warning("INT8 not supported, using FP32")

        # Set optimization profiles for dynamic shapes
        if self.config.dynamic_shapes:
            profile = builder.create_optimization_profile()

            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                name = input_tensor.name
                shape = input_tensor.shape

                # Create dynamic shape ranges
                if input_shape is not None:
                    min_shape = list(input_shape)
                    opt_shape = list(input_shape)
                    max_shape = list(input_shape)

                    min_shape[0] = self.config.min_batch_size
                    opt_shape[0] = self.config.opt_batch_size
                    max_shape[0] = self.config.max_batch_size
                else:
                    # Use tensor shape with dynamic batch
                    min_shape = [self.config.min_batch_size] + list(shape[1:])
                    opt_shape = [self.config.opt_batch_size] + list(shape[1:])
                    max_shape = [self.config.max_batch_size] + list(shape[1:])

                # Replace -1 with concrete values
                for idx, dim in enumerate(min_shape):
                    if dim == -1:
                        min_shape[idx] = 1
                        opt_shape[idx] = 256
                        max_shape[idx] = 1024

                profile.set_shape(
                    name,
                    tuple(min_shape),
                    tuple(opt_shape),
                    tuple(max_shape),
                )

            config.add_optimization_profile(profile)

        # DLA configuration for Jetson
        if self.config.dla_core >= 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.config.dla_core
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Engine build failed")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        elapsed = time.time() - start_time
        engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Built TensorRT engine in {elapsed:.1f}s "
            f"(size: {engine_size_mb:.1f}MB): {engine_path}"
        )

        return engine_path

    def load_engine(self, engine_path: Path, model_name: Optional[str] = None) -> str:
        """Load a pre-compiled TensorRT engine.

        Args:
            engine_path: Path to engine file
            model_name: Optional name (uses filename if not provided)

        Returns:
            Model name used for inference
        """
        if model_name is None:
            model_name = engine_path.stem

        if not self.load_model(model_name, engine_path):
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        return model_name

    def infer(
        self,
        engine_name: str,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """Run inference using TensorRT engine.

        Args:
            engine_name: Name of loaded engine
            input_tensor: Input data as numpy array

        Returns:
            Output as numpy array
        """
        result = self.run_inference(engine_name, input_tensor)

        # Return first output if single output
        if isinstance(result, dict) and len(result) == 1:
            return list(result.values())[0]

        return result

    def infer_batch(
        self,
        engine_name: str,
        inputs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Batch inference for multiple inputs.

        Args:
            engine_name: Name of loaded engine
            inputs: List of input arrays

        Returns:
            List of output arrays
        """
        if not inputs:
            return []

        # Stack inputs into batch
        batch = np.stack(inputs, axis=0)

        # Run inference
        output = self.infer(engine_name, batch)

        # Split batch into list
        return [output[i] for i in range(len(inputs))]

    def _get_cache_path(self, name: str) -> Path:
        """Get cache path for an engine."""
        precision_suffix = f"_{self.config.precision}"
        return self.config.cache_dir / f"{name}{precision_suffix}.trt"

    def _compute_cache_key(
        self,
        model: "torch.nn.Module",
        input_shape: Tuple[int, ...],
        name: str,
    ) -> str:
        """Compute unique cache key for model configuration."""
        # Hash model parameters
        torch = _get_torch()

        param_data = b""
        for param in model.parameters():
            param_data += param.data.cpu().numpy().tobytes()[:100]  # Sample

        param_hash = hashlib.md5(param_data).hexdigest()[:8]
        shape_str = "x".join(map(str, input_shape))

        return f"{name}_{shape_str}_{param_hash}"


# =============================================================================
# Factory Functions
# =============================================================================

def create_tensorrt_backend(
    precision: str = "fp16",
    device_id: int = 0,
    **kwargs,
) -> Optional[TensorRTBackend]:
    """Create TensorRT backend if available, else return None.

    Args:
        precision: Inference precision (fp32, fp16, int8)
        device_id: CUDA device ID
        **kwargs: Additional TensorRTConfig options

    Returns:
        TensorRTBackend if available, None otherwise
    """
    config = TensorRTConfig(precision=precision, **kwargs)
    backend = TensorRTBackend(config, device_id)

    if backend.is_available():
        if backend.initialize():
            return backend
        else:
            logger.warning("TensorRT initialization failed")
    else:
        logger.debug("TensorRT not available")

    return None


def accelerate_model(
    model: "torch.nn.Module",
    input_shape: Tuple[int, ...],
    name: str,
    precision: str = "fp16",
    device_id: int = 0,
) -> Callable:
    """Wrap a PyTorch model with TensorRT acceleration.

    Creates a callable that transparently uses TensorRT if available,
    falling back to PyTorch otherwise.

    Args:
        model: PyTorch model to accelerate
        input_shape: Expected input shape
        name: Unique model name
        precision: TensorRT precision
        device_id: CUDA device ID

    Returns:
        Callable that runs accelerated inference

    Example:
        >>> model = MyModel()
        >>> fast_model = accelerate_model(model, (1, 3, 256, 256), "my_model")
        >>> output = fast_model(input_tensor)
    """
    torch = _get_torch()
    if torch is None:
        raise RuntimeError("PyTorch not available")

    # Try to create TensorRT backend
    backend = create_tensorrt_backend(precision=precision, device_id=device_id)

    if backend is not None:
        try:
            # Convert model
            engine_path = backend.convert_model(model, input_shape, name)
            backend.load_engine(engine_path, name)

            logger.info(f"Model '{name}' accelerated with TensorRT")

            def trt_inference(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
                """TensorRT inference wrapper."""
                # Convert torch tensor to numpy
                if hasattr(x, 'cpu'):
                    x = x.cpu().numpy()

                return backend.infer(name, x)

            return trt_inference

        except Exception as e:
            logger.warning(f"TensorRT acceleration failed, using PyTorch: {e}")
            backend.cleanup()

    # Fallback to PyTorch
    logger.info(f"Model '{name}' using PyTorch (TensorRT unavailable)")

    model = model.eval()
    model = model.cuda(device_id)

    def pytorch_inference(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """PyTorch inference wrapper."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x = x.cuda(device_id)

        with torch.no_grad():
            output = model(x)

        return output.cpu().numpy()

    return pytorch_inference


# =============================================================================
# Backend Registration
# =============================================================================

# Register TensorRT backend in the backend registry
try:
    register_backend(BackendType.TENSORRT, TensorRTBackend)
except Exception:
    pass  # Registry may not be available during import


# =============================================================================
# Convenience Functions
# =============================================================================

def is_tensorrt_available() -> bool:
    """Check if TensorRT is available.

    Returns:
        True if TensorRT can be used
    """
    trt = _get_tensorrt()
    torch = _get_torch()

    if trt is None:
        return False

    if torch is None or not torch.cuda.is_available():
        return False

    return True


def get_tensorrt_version() -> Optional[str]:
    """Get TensorRT version string.

    Returns:
        Version string or None if not available
    """
    trt = _get_tensorrt()
    if trt is None:
        return None

    return trt.__version__


def clear_engine_cache(cache_dir: Optional[Path] = None) -> int:
    """Clear cached TensorRT engines.

    Args:
        cache_dir: Cache directory (uses default if None)

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".framewright" / "cache" / "tensorrt"

    if not cache_dir.exists():
        return 0

    count = 0
    for engine_file in cache_dir.glob("*.trt"):
        engine_file.unlink()
        count += 1

    for onnx_file in cache_dir.glob("*.onnx"):
        onnx_file.unlink()
        count += 1

    for calib_file in cache_dir.glob("*.cache"):
        calib_file.unlink()
        count += 1

    logger.info(f"Cleared {count} cached files from {cache_dir}")
    return count
