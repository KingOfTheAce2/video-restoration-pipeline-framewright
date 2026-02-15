"""Apple Silicon Neural Engine Backend for FrameWright.

Provides optimized inference on Apple Silicon (M1/M2/M3/M4) using CoreML and
the Apple Neural Engine (ANE) for maximum efficiency.
"""

import gc
import hashlib
import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..detector import BackendType, GPUVendor, get_hardware_info
from .base import Backend, BackendCapabilities, register_backend

logger = logging.getLogger(__name__)

# Lazy imports
_torch = None
_torch_checked = False
_coremltools = None
_coremltools_checked = False


def _get_torch():
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch


def _get_coremltools():
    global _coremltools, _coremltools_checked
    if not _coremltools_checked:
        try:
            import coremltools as ct
            _coremltools = ct
        except ImportError:
            _coremltools = None
        _coremltools_checked = True
    return _coremltools


class ComputeUnits(Enum):
    """CoreML compute unit configuration."""
    ALL = "all"
    CPU_AND_ANE = "cpu_and_ane"
    CPU_AND_GPU = "cpu_and_gpu"
    CPU_ONLY = "cpu_only"


class AppleChipVariant(Enum):
    """Apple Silicon chip variants."""
    M1 = "m1"
    M1_PRO = "m1_pro"
    M1_MAX = "m1_max"
    M1_ULTRA = "m1_ultra"
    M2 = "m2"
    M2_PRO = "m2_pro"
    M2_MAX = "m2_max"
    M2_ULTRA = "m2_ultra"
    M3 = "m3"
    M3_PRO = "m3_pro"
    M3_MAX = "m3_max"
    M4 = "m4"
    M4_PRO = "m4_pro"
    M4_MAX = "m4_max"
    UNKNOWN = "unknown"


# Neural Engine and GPU cores by chip variant
ANE_CORES = {
    AppleChipVariant.M1: 16, AppleChipVariant.M1_PRO: 16, AppleChipVariant.M1_MAX: 16,
    AppleChipVariant.M1_ULTRA: 32, AppleChipVariant.M2: 16, AppleChipVariant.M2_PRO: 16,
    AppleChipVariant.M2_MAX: 16, AppleChipVariant.M2_ULTRA: 32, AppleChipVariant.M3: 16,
    AppleChipVariant.M3_PRO: 16, AppleChipVariant.M3_MAX: 16, AppleChipVariant.M4: 16,
    AppleChipVariant.M4_PRO: 16, AppleChipVariant.M4_MAX: 16, AppleChipVariant.UNKNOWN: 16,
}

GPU_CORES = {
    AppleChipVariant.M1: 8, AppleChipVariant.M1_PRO: 16, AppleChipVariant.M1_MAX: 32,
    AppleChipVariant.M1_ULTRA: 64, AppleChipVariant.M2: 10, AppleChipVariant.M2_PRO: 19,
    AppleChipVariant.M2_MAX: 38, AppleChipVariant.M2_ULTRA: 76, AppleChipVariant.M3: 10,
    AppleChipVariant.M3_PRO: 18, AppleChipVariant.M3_MAX: 40, AppleChipVariant.M4: 10,
    AppleChipVariant.M4_PRO: 20, AppleChipVariant.M4_MAX: 40, AppleChipVariant.UNKNOWN: 8,
}


@dataclass
class AppleSiliconConfig:
    """Configuration for Apple Silicon backend."""
    use_ane: bool = True
    use_gpu: bool = True
    compute_units: Literal["all", "cpu_and_ane", "cpu_and_gpu", "cpu_only"] = "all"
    precision: Literal["fp32", "fp16"] = "fp16"
    cache_dir: Optional[Path] = None
    force_convert: bool = False
    optimize_for_ane: bool = True
    batch_size: int = 1

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".framewright" / "cache" / "coreml"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.precision not in ("fp32", "fp16"):
            raise ValueError(f"Invalid precision: {self.precision}")

    def get_compute_units(self):
        ct = _get_coremltools()
        if ct is None:
            return None
        mapping = {
            "all": ct.ComputeUnit.ALL, "cpu_and_ane": ct.ComputeUnit.CPU_AND_NE,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }
        return mapping.get(self.compute_units, ct.ComputeUnit.ALL)


class CoreMLConverter:
    """Convert PyTorch and ONNX models to CoreML format."""

    def __init__(self, config: AppleSiliconConfig):
        self.config = config
        self._lock = threading.Lock()

    def convert_pytorch(self, model: "torch.nn.Module", input_shape: Tuple[int, ...],
                        input_name: str = "input", output_name: str = "output",
                        model_name: Optional[str] = None) -> Any:
        torch = _get_torch()
        ct = _get_coremltools()
        if torch is None:
            raise RuntimeError("PyTorch not available")
        if ct is None:
            raise RuntimeError("coremltools not available")

        with self._lock:
            logger.info(f"Converting PyTorch model to CoreML (shape={input_shape})")
            model = model.eval().cpu()
            if hasattr(model, 'float'):
                model = model.float()

            with torch.no_grad():
                traced = torch.jit.trace(model, torch.randn(input_shape))

            mlmodel = ct.convert(
                traced, inputs=[ct.TensorType(name=input_name, shape=input_shape)],
                outputs=[ct.TensorType(name=output_name)], convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS13,
            )

            if self.config.precision == "fp16":
                mlmodel = self._apply_fp16(mlmodel)
            if model_name:
                mlmodel.short_description = f"FrameWright: {model_name}"
            return mlmodel

    def convert_onnx(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> Any:
        ct = _get_coremltools()
        if ct is None:
            raise RuntimeError("coremltools not available")
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        with self._lock:
            mlmodel = ct.converters.onnx.convert(str(onnx_path), minimum_deployment_target=ct.target.macOS13)
            if self.config.precision == "fp16":
                mlmodel = self._apply_fp16(mlmodel)
            if model_name:
                mlmodel.short_description = f"FrameWright: {model_name}"
            return mlmodel

    def optimize_for_ane(self, mlmodel: Any) -> Any:
        ct = _get_coremltools()
        if ct is None:
            return mlmodel
        try:
            from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
            config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=8))
            return palettize_weights(mlmodel, config)
        except (ImportError, Exception) as e:
            logger.debug(f"ANE optimization skipped: {e}")
            return mlmodel

    def quantize(self, mlmodel: Any, mode: Literal["float16", "int8"] = "float16") -> Any:
        ct = _get_coremltools()
        if ct is None:
            return mlmodel
        try:
            if mode == "float16":
                return self._apply_fp16(mlmodel)
            elif mode == "int8":
                from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
                config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
                return linear_quantize_weights(mlmodel, config)
        except (ImportError, Exception) as e:
            logger.warning(f"Quantization failed: {e}")
        return mlmodel

    def _apply_fp16(self, mlmodel: Any) -> Any:
        try:
            from coremltools.models.neural_network import quantization_utils
            return quantization_utils.quantize_weights(mlmodel, nbits=16)
        except Exception:
            return mlmodel


class CoreMLModel:
    """Wrapper for loaded CoreML models with inference support."""

    def __init__(self, mlmodel: Any, config: AppleSiliconConfig, name: str = "model"):
        self.mlmodel = mlmodel
        self.config = config
        self.name = name
        self._lock = threading.Lock()
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._analyze_model()

    def _analyze_model(self) -> None:
        try:
            spec = self.mlmodel.get_spec()
            for inp in spec.description.input:
                self._input_names.append(inp.name)
                if hasattr(inp.type, 'multiArrayType'):
                    self._input_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            for out in spec.description.output:
                self._output_names.append(out.name)
        except Exception as e:
            logger.debug(f"Failed to analyze model: {e}")

    def predict(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        with self._lock:
            if isinstance(inputs, np.ndarray):
                if not self._input_names:
                    raise ValueError("Model input names not determined")
                inputs = {self._input_names[0]: inputs}

            result = self.mlmodel.predict(inputs)
            outputs = {}
            for name in self._output_names:
                if name in result:
                    val = result[name]
                    outputs[name] = val.numpy() if hasattr(val, 'numpy') else np.array(val)
            return outputs

    @property
    def input_names(self) -> List[str]:
        return self._input_names.copy()

    @property
    def output_names(self) -> List[str]:
        return self._output_names.copy()

    def cleanup(self) -> None:
        self.mlmodel = None
        gc.collect()


class ANEOptimizer:
    """Optimize models specifically for Apple Neural Engine."""

    def __init__(self, chip: AppleChipVariant = AppleChipVariant.UNKNOWN):
        self.chip = chip
        self.ane_cores = ANE_CORES.get(chip, 16)
        self.gpu_cores = GPU_CORES.get(chip, 8)

    def optimize_model(self, mlmodel: Any) -> Any:
        ct = _get_coremltools()
        if ct is None:
            return mlmodel
        logger.info(f"Optimizing for ANE ({self.chip.value}, {self.ane_cores} cores)")
        # CoreML handles most optimizations internally during compilation
        return mlmodel

    def get_optimal_batch_size(self, input_shape: Tuple[int, ...], precision: str = "fp16") -> int:
        elements = 1
        for dim in input_shape:
            elements *= dim
        bytes_per_element = 2 if precision == "fp16" else 4
        memory_per_sample_mb = (elements * bytes_per_element) / (1024 * 1024)

        info = get_hardware_info()
        available_mb = info.ram_available_mb * 0.5
        max_batch = min(16, int(available_mb / memory_per_sample_mb))

        # Pro/Max/Ultra chips handle larger batches
        if self.chip in (AppleChipVariant.M1_PRO, AppleChipVariant.M1_MAX, AppleChipVariant.M1_ULTRA,
                         AppleChipVariant.M2_PRO, AppleChipVariant.M2_MAX, AppleChipVariant.M2_ULTRA,
                         AppleChipVariant.M3_PRO, AppleChipVariant.M3_MAX,
                         AppleChipVariant.M4_PRO, AppleChipVariant.M4_MAX):
            max_batch = min(32, max_batch * 2)
        return max(1, max_batch)


class PerformanceProfiler:
    """Profile CoreML model performance on different compute units."""

    def __init__(self, config: AppleSiliconConfig):
        self.config = config
        self._results: Dict[str, Dict[str, float]] = {}

    def profile_model(self, model_path: Path, input_shape: Tuple[int, ...],
                      num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, Dict[str, float]]:
        ct = _get_coremltools()
        if ct is None:
            raise RuntimeError("coremltools not available")

        results = {}
        compute_options = [
            ("all", ct.ComputeUnit.ALL), ("cpu_and_ane", ct.ComputeUnit.CPU_AND_NE),
            ("cpu_and_gpu", ct.ComputeUnit.CPU_AND_GPU), ("cpu_only", ct.ComputeUnit.CPU_ONLY),
        ]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for name, compute_unit in compute_options:
            try:
                mlmodel = ct.models.MLModel(str(model_path), compute_units=compute_unit)
                for _ in range(warmup_runs):
                    mlmodel.predict({"input": dummy_input})

                times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    mlmodel.predict({"input": dummy_input})
                    times.append((time.perf_counter() - start) * 1000)

                results[name] = {
                    "mean_ms": np.mean(times), "std_ms": np.std(times),
                    "min_ms": np.min(times), "max_ms": np.max(times),
                    "fps": 1000.0 / np.mean(times),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        self._results[str(model_path)] = results
        return results

    def find_optimal_config(self, model_path: Path, input_shape: Tuple[int, ...]) -> str:
        results = self.profile_model(model_path, input_shape)
        best_config, best_fps = "all", 0.0
        for config, metrics in results.items():
            if "error" not in metrics and metrics.get("fps", 0) > best_fps:
                best_fps = metrics["fps"]
                best_config = config
        return best_config

    def get_memory_usage(self) -> Dict[str, float]:
        try:
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return {}
            stats, page_size = {}, 16384
            for line in result.stdout.split("\n"):
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages free:" in line:
                    stats["free_mb"] = int(line.split()[2].replace(".", "")) * page_size / (1024 * 1024)
            return stats
        except Exception:
            return {}


class PreconvertedModelsRegistry:
    """Registry of pre-converted CoreML models."""

    KNOWN_MODELS = {
        "realesrgan-x4": "framewright/realesrgan-x4-coreml",
        "realesrgan-x2": "framewright/realesrgan-x2-coreml",
        "esrgan-general": "framewright/esrgan-general-coreml",
        "swinir-classical": "framewright/swinir-classical-coreml",
        "restoreformer": "framewright/restoreformer-coreml",
    }

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_models: Dict[str, Path] = {}
        self._scan_local_models()

    def _scan_local_models(self) -> None:
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir() and model_dir.suffix == ".mlpackage":
                self._local_models[model_dir.stem] = model_dir
        for model_file in self.cache_dir.glob("*.mlmodelc"):
            self._local_models[model_file.stem] = model_file

    def is_available(self, model_name: str) -> bool:
        return model_name in self._local_models

    def get_model_path(self, model_name: str) -> Optional[Path]:
        return self._local_models.get(model_name)

    def list_available(self) -> List[str]:
        return sorted(set(self._local_models.keys()) | set(self.KNOWN_MODELS.keys()))

    def list_local(self) -> List[str]:
        return sorted(self._local_models.keys())

    def download(self, model_name: str) -> Optional[Path]:
        if model_name not in self.KNOWN_MODELS:
            return None
        local_path = self.cache_dir / f"{model_name}.mlpackage"
        if local_path.exists():
            return local_path
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=self.KNOWN_MODELS[model_name], local_dir=str(local_path))
            self._local_models[model_name] = local_path
            return local_path
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return None

    def register_local(self, model_name: str, model_path: Path) -> None:
        if model_path.exists():
            self._local_models[model_name] = model_path

    def clear_cache(self) -> int:
        import shutil
        count = 0
        for path in list(self._local_models.values()):
            try:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                count += 1
            except Exception:
                pass
        self._local_models.clear()
        return count


class AppleSiliconBackend(Backend):
    """Apple Silicon backend with Neural Engine and Metal support."""

    def __init__(self, config: Optional[AppleSiliconConfig] = None, device_id: int = 0):
        super().__init__(device_id)
        self.config = config or AppleSiliconConfig()
        self._models: Dict[str, CoreMLModel] = {}
        self._model_paths: Dict[str, Path] = {}
        self._converter: Optional[CoreMLConverter] = None
        self._optimizer: Optional[ANEOptimizer] = None
        self._registry: Optional[PreconvertedModelsRegistry] = None
        self._chip: AppleChipVariant = AppleChipVariant.UNKNOWN
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> BackendType:
        return BackendType.COREML

    @property
    def name(self) -> str:
        return "Apple Silicon (CoreML)"

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=5)
            if "Apple" not in result.stdout:
                return False
        except Exception:
            return False
        return _get_coremltools() is not None

    def initialize(self) -> bool:
        if not self.is_available():
            return False
        try:
            self._chip = self._detect_chip()
            self._converter = CoreMLConverter(self.config)
            self._optimizer = ANEOptimizer(self._chip)
            self._registry = PreconvertedModelsRegistry(self.config.cache_dir)
            self._initialized = True
            logger.info(f"Apple Silicon backend initialized ({self._chip.value}, "
                        f"ANE: {self.get_ane_cores()} cores, GPU: {self.get_gpu_cores()} cores)")
            return True
        except Exception as e:
            logger.error(f"Apple Silicon initialization failed: {e}")
            return False

    def cleanup(self) -> None:
        with self._lock:
            for model in self._models.values():
                model.cleanup()
            self._models.clear()
            self._model_paths.clear()
        gc.collect()
        self._initialized = False

    def get_capabilities(self) -> BackendCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        caps = BackendCapabilities(
            name=self.name, backend_type=BackendType.COREML, vendor=GPUVendor.APPLE,
            supports_fp16=True, supports_fp32=True, supports_int8=True,
            supports_dynamic_shapes=True, supports_batching=True,
        )
        info = get_hardware_info()
        if info.primary_device:
            caps.max_memory_mb = info.primary_device.total_memory_mb
            caps.recommended_memory_mb = int(caps.max_memory_mb * 0.7)
        caps.supported_models = ["realesrgan", "esrgan", "swinir", "restoreformer"]
        self._capabilities = caps
        return caps

    def allocate_memory(self, size_mb: float) -> bool:
        return get_hardware_info().ram_available_mb >= size_mb

    def free_memory(self) -> None:
        gc.collect()

    def get_memory_info(self) -> Dict[str, float]:
        info = get_hardware_info()
        return {"total_mb": info.ram_total_mb, "used_mb": info.ram_total_mb - info.ram_available_mb,
                "free_mb": info.ram_available_mb}

    def load_model(self, model_name: str, model_path: Optional[Path] = None, **kwargs) -> bool:
        ct = _get_coremltools()
        if ct is None:
            return False
        with self._lock:
            try:
                if model_path is None and self._registry:
                    model_path = self._registry.get_model_path(model_name)
                if model_path is None or not Path(model_path).exists():
                    logger.error(f"Model not found: {model_name}")
                    return False
                model_path = Path(model_path)
                mlmodel = ct.models.MLModel(str(model_path), compute_units=self.config.get_compute_units())
                self._models[model_name] = CoreMLModel(mlmodel, self.config, model_name)
                self._model_paths[model_name] = model_path
                logger.info(f"Loaded CoreML model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return False

    def unload_model(self, model_name: str) -> None:
        with self._lock:
            if model_name in self._models:
                self._models[model_name].cleanup()
                del self._models[model_name]
                self._model_paths.pop(model_name, None)
        gc.collect()

    def run_inference(self, model_name: str, inputs: Any, **kwargs) -> Any:
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not loaded")
        result = self._models[model_name].predict(inputs)
        return list(result.values())[0] if len(result) == 1 else result

    def get_chip_info(self) -> Dict[str, Any]:
        return {"chip": self._chip.value, "ane_cores": self.get_ane_cores(),
                "gpu_cores": self.get_gpu_cores(), "unified_memory": True}

    def get_ane_cores(self) -> int:
        return ANE_CORES.get(self._chip, 16)

    def get_gpu_cores(self) -> int:
        return GPU_CORES.get(self._chip, 8)

    def get_memory_gb(self) -> float:
        return get_hardware_info().ram_total_mb / 1024.0

    def convert_model(self, model: "torch.nn.Module", model_name: str,
                      input_shape: Tuple[int, ...], optimize: bool = True) -> Path:
        if self._converter is None:
            raise RuntimeError("Backend not initialized")

        cache_key = self._compute_cache_key(model, input_shape, model_name)
        cache_path = self.config.cache_dir / f"{cache_key}.mlpackage"

        if cache_path.exists() and not self.config.force_convert:
            return cache_path

        mlmodel = self._converter.convert_pytorch(model, input_shape, model_name=model_name)
        if optimize and self.config.optimize_for_ane and self._optimizer:
            mlmodel = self._optimizer.optimize_model(mlmodel)
        mlmodel.save(str(cache_path))
        if self._registry:
            self._registry.register_local(model_name, cache_path)
        return cache_path

    def _detect_chip(self) -> AppleChipVariant:
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=5)
            cpu = result.stdout.strip().lower()
            for gen in ["m4", "m3", "m2", "m1"]:
                if gen in cpu:
                    if "ultra" in cpu:
                        return AppleChipVariant[f"{gen.upper()}_ULTRA"]
                    elif "max" in cpu:
                        return AppleChipVariant[f"{gen.upper()}_MAX"]
                    elif "pro" in cpu:
                        return AppleChipVariant[f"{gen.upper()}_PRO"]
                    return AppleChipVariant[gen.upper()]
        except Exception:
            pass
        return AppleChipVariant.UNKNOWN

    def _compute_cache_key(self, model: "torch.nn.Module", input_shape: Tuple[int, ...], name: str) -> str:
        torch = _get_torch()
        if torch is None:
            return f"{name}_{hash(str(input_shape))}"
        param_data = b""
        for param in model.parameters():
            param_data += param.data.cpu().numpy().tobytes()[:100]
        param_hash = hashlib.md5(param_data).hexdigest()[:8]
        return f"{name}_{'x'.join(map(str, input_shape))}_{self.config.precision}_{param_hash}"


# Factory functions
def create_apple_silicon_backend(use_ane: bool = True, use_gpu: bool = True,
                                  precision: str = "fp16", **kwargs) -> Optional[AppleSiliconBackend]:
    config = AppleSiliconConfig(use_ane=use_ane, use_gpu=use_gpu, precision=precision, **kwargs)
    backend = AppleSiliconBackend(config)
    if backend.is_available() and backend.initialize():
        return backend
    return None


def convert_to_coreml(model: "torch.nn.Module", model_name: str, input_shape: Tuple[int, ...],
                      optimize: bool = True, precision: str = "fp16") -> Optional[Path]:
    backend = create_apple_silicon_backend(precision=precision)
    if backend is None:
        return None
    try:
        return backend.convert_model(model, model_name, input_shape, optimize)
    finally:
        backend.cleanup()


def is_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, timeout=5)
        return "Apple" in result.stdout
    except Exception:
        return False


def get_apple_chip_info() -> Dict[str, Any]:
    if not is_apple_silicon():
        return {}
    backend = AppleSiliconBackend()
    if not backend.is_available():
        return {}
    chip = backend._detect_chip()
    return {"chip": chip.value, "ane_cores": ANE_CORES.get(chip, 16),
            "gpu_cores": GPU_CORES.get(chip, 8), "unified_memory": True, "neural_engine": True}


# Backend registration
try:
    register_backend(BackendType.COREML, AppleSiliconBackend)
except Exception:
    pass

__all__ = [
    "AppleSiliconConfig", "ComputeUnits", "AppleChipVariant",
    "CoreMLConverter", "CoreMLModel", "ANEOptimizer", "PerformanceProfiler",
    "PreconvertedModelsRegistry", "AppleSiliconBackend",
    "create_apple_silicon_backend", "convert_to_coreml", "is_apple_silicon", "get_apple_chip_info",
    "ANE_CORES", "GPU_CORES",
]
