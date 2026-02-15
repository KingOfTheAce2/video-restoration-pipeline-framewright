"""Mixed precision support for faster inference."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import functools

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Precision modes for inference."""
    FP32 = "fp32"  # Full precision
    FP16 = "fp16"  # Half precision
    BF16 = "bf16"  # Brain floating point
    INT8 = "int8"  # 8-bit quantization
    AUTO = "auto"  # Automatic selection


@dataclass
class PrecisionCapabilities:
    """Hardware precision capabilities."""
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_tensor_cores: bool = False
    compute_capability: Tuple[int, int] = (0, 0)
    device_name: str = ""


class MixedPrecisionManager:
    """Manages mixed precision for inference."""

    def __init__(
        self,
        mode: PrecisionMode = PrecisionMode.AUTO,
        enable_tensor_cores: bool = True,
    ):
        self.mode = mode
        self.enable_tensor_cores = enable_tensor_cores

        self._capabilities: Optional[PrecisionCapabilities] = None
        self._active_mode: PrecisionMode = PrecisionMode.FP32
        self._autocast_enabled: bool = False

        self._detect_capabilities()
        self._select_mode()

    def _detect_capabilities(self) -> None:
        """Detect hardware precision capabilities."""
        caps = PrecisionCapabilities()

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                caps.device_name = props.name
                caps.compute_capability = (props.major, props.minor)

                # FP16 supported on compute capability >= 5.3
                caps.supports_fp16 = props.major > 5 or (props.major == 5 and props.minor >= 3)

                # BF16 supported on Ampere+ (compute capability >= 8.0)
                caps.supports_bf16 = props.major >= 8

                # Tensor cores on Volta+ (7.0+) with Tensor Core support
                caps.supports_tensor_cores = props.major >= 7

                # INT8 generally available on modern GPUs
                caps.supports_int8 = props.major >= 6

                logger.info(f"GPU: {caps.device_name} (compute {props.major}.{props.minor})")
                logger.info(f"Precision support: FP16={caps.supports_fp16}, BF16={caps.supports_bf16}, "
                           f"INT8={caps.supports_int8}, TensorCores={caps.supports_tensor_cores}")

        except ImportError:
            logger.warning("PyTorch not available, using FP32 only")

        self._capabilities = caps

    def _select_mode(self) -> None:
        """Select optimal precision mode."""
        if self.mode != PrecisionMode.AUTO:
            self._active_mode = self.mode
            return

        caps = self._capabilities

        if caps is None:
            self._active_mode = PrecisionMode.FP32
            return

        # Prefer BF16 on Ampere+ for better numerical stability
        if caps.supports_bf16:
            self._active_mode = PrecisionMode.BF16
        elif caps.supports_fp16:
            self._active_mode = PrecisionMode.FP16
        else:
            self._active_mode = PrecisionMode.FP32

        logger.info(f"Selected precision mode: {self._active_mode.value}")

    @property
    def active_mode(self) -> PrecisionMode:
        """Get active precision mode."""
        return self._active_mode

    @property
    def capabilities(self) -> PrecisionCapabilities:
        """Get hardware capabilities."""
        return self._capabilities or PrecisionCapabilities()

    def get_dtype(self) -> Any:
        """Get PyTorch dtype for current mode."""
        try:
            import torch

            dtype_map = {
                PrecisionMode.FP32: torch.float32,
                PrecisionMode.FP16: torch.float16,
                PrecisionMode.BF16: torch.bfloat16,
            }
            return dtype_map.get(self._active_mode, torch.float32)

        except ImportError:
            return None

    @contextmanager
    def autocast_context(self) -> Iterator[None]:
        """Context manager for automatic mixed precision.

        Usage:
            with manager.autocast_context():
                output = model(input)
        """
        if self._active_mode == PrecisionMode.FP32:
            yield
            return

        try:
            import torch

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = self.get_dtype()

            with torch.autocast(device_type=device_type, dtype=dtype):
                yield

        except ImportError:
            yield

    def enable_autocast(self) -> None:
        """Enable automatic mixed precision globally."""
        self._autocast_enabled = True

    def disable_autocast(self) -> None:
        """Disable automatic mixed precision globally."""
        self._autocast_enabled = False

    def convert_model(self, model: Any) -> Any:
        """Convert model to appropriate precision.

        Args:
            model: PyTorch model

        Returns:
            Converted model
        """
        if self._active_mode == PrecisionMode.FP32:
            return model

        try:
            import torch

            dtype = self.get_dtype()

            if self._active_mode in (PrecisionMode.FP16, PrecisionMode.BF16):
                return model.to(dtype=dtype)

            elif self._active_mode == PrecisionMode.INT8:
                # Dynamic quantization for INT8
                return torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )

        except Exception as e:
            logger.warning(f"Failed to convert model: {e}")
            return model

        return model

    def prepare_input(self, tensor: Any) -> Any:
        """Prepare input tensor for current precision.

        Args:
            tensor: Input tensor

        Returns:
            Tensor in appropriate precision
        """
        if self._active_mode == PrecisionMode.FP32:
            return tensor

        try:
            import torch

            if not isinstance(tensor, torch.Tensor):
                return tensor

            dtype = self.get_dtype()
            return tensor.to(dtype=dtype)

        except:
            return tensor

    def inference_wrapper(
        self,
        func: Callable,
    ) -> Callable:
        """Decorator for inference functions with mixed precision.

        Usage:
            @manager.inference_wrapper
            def process(model, input):
                return model(input)
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self._active_mode == PrecisionMode.FP32:
                return func(*args, **kwargs)

            with self.autocast_context():
                return func(*args, **kwargs)

        return wrapper

    def get_memory_savings(self) -> Dict[str, float]:
        """Estimate memory savings from current precision mode.

        Returns:
            Dictionary with memory statistics
        """
        mode = self._active_mode

        # Relative to FP32
        savings = {
            PrecisionMode.FP32: 1.0,
            PrecisionMode.FP16: 0.5,
            PrecisionMode.BF16: 0.5,
            PrecisionMode.INT8: 0.25,
        }

        multiplier = savings.get(mode, 1.0)

        return {
            "precision_mode": mode.value,
            "memory_multiplier": multiplier,
            "estimated_savings_percent": (1 - multiplier) * 100,
        }

    def benchmark(
        self,
        model: Any,
        sample_input: Any,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark model at different precision levels.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            iterations: Number of iterations

        Returns:
            Benchmark results
        """
        results = {}

        try:
            import torch
            import time

            device = next(model.parameters()).device

            # Test each supported precision
            for mode in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16]:
                if mode == PrecisionMode.BF16 and not self._capabilities.supports_bf16:
                    continue
                if mode == PrecisionMode.FP16 and not self._capabilities.supports_fp16:
                    continue

                # Prepare model and input
                test_model = model.float()
                test_input = sample_input.to(device)

                dtype = {
                    PrecisionMode.FP32: torch.float32,
                    PrecisionMode.FP16: torch.float16,
                    PrecisionMode.BF16: torch.bfloat16,
                }.get(mode, torch.float32)

                # Warmup
                with torch.no_grad():
                    if mode != PrecisionMode.FP32:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            for _ in range(10):
                                _ = test_model(test_input)
                    else:
                        for _ in range(10):
                            _ = test_model(test_input)

                torch.cuda.synchronize()

                # Benchmark
                start = time.time()
                with torch.no_grad():
                    if mode != PrecisionMode.FP32:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            for _ in range(iterations):
                                _ = test_model(test_input)
                    else:
                        for _ in range(iterations):
                            _ = test_model(test_input)

                torch.cuda.synchronize()
                elapsed = time.time() - start

                results[mode.value] = {
                    "total_time_ms": elapsed * 1000,
                    "avg_time_ms": (elapsed / iterations) * 1000,
                    "throughput_fps": iterations / elapsed,
                    "memory_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                }

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

        return results


def get_optimal_precision(
    vram_gb: float = 0,
    model_size_gb: float = 0,
) -> PrecisionMode:
    """Get optimal precision mode based on hardware and model.

    Args:
        vram_gb: Available VRAM in GB (0 = auto-detect)
        model_size_gb: Model size in GB (0 = unknown)

    Returns:
        Recommended precision mode
    """
    manager = MixedPrecisionManager(mode=PrecisionMode.AUTO)
    caps = manager.capabilities

    # Auto-detect VRAM if not specified
    if vram_gb <= 0:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            vram_gb = 8  # Default assumption

    # If model is large relative to VRAM, prefer lower precision
    if model_size_gb > 0 and model_size_gb > vram_gb * 0.7:
        if caps.supports_fp16:
            return PrecisionMode.FP16

    # For RTX 30/40 series (Ampere+), prefer BF16
    if caps.supports_bf16:
        return PrecisionMode.BF16

    # For older cards, use FP16 if available
    if caps.supports_fp16:
        return PrecisionMode.FP16

    return PrecisionMode.FP32
