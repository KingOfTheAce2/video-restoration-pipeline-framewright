"""Backend implementations for different compute platforms.

This package provides abstract and concrete backend implementations for:
- CUDA (NVIDIA)
- TensorRT (NVIDIA accelerated inference)
- ROCm (AMD GPUs)
- oneAPI (Intel GPUs - Arc, Iris, UHD)
- Metal (Apple Silicon MPS)
- CoreML (Apple Silicon Neural Engine)
- Vulkan (cross-platform via ncnn)
- DirectML (Windows AMD/Intel)
- CPU (fallback)

Each backend provides a consistent interface for:
- Device initialization
- Memory management
- Model loading/inference
- Tensor operations
"""

from .base import (
    Backend,
    BackendCapabilities,
    get_backend,
    get_backend_for_vendor,
    list_backends,
)

# TensorRT backend (optional - requires tensorrt package)
try:
    from .tensorrt import (
        TensorRTBackend,
        TensorRTConfig,
        create_tensorrt_backend,
        accelerate_model,
        is_tensorrt_available,
        get_tensorrt_version,
        clear_engine_cache,
    )
    _TENSORRT_AVAILABLE = True
except ImportError:
    _TENSORRT_AVAILABLE = False
    TensorRTBackend = None
    TensorRTConfig = None
    create_tensorrt_backend = None
    accelerate_model = None
    is_tensorrt_available = lambda: False
    get_tensorrt_version = lambda: None
    clear_engine_cache = lambda cache_dir=None: 0

# Apple Silicon backend (optional - requires coremltools on macOS)
try:
    from .apple_silicon import (
        AppleSiliconBackend,
        AppleSiliconConfig,
        CoreMLConverter,
        CoreMLModel,
        ANEOptimizer,
        PerformanceProfiler,
        PreconvertedModelsRegistry,
        ComputeUnits,
        AppleChipVariant,
        create_apple_silicon_backend,
        convert_to_coreml,
        is_apple_silicon,
        get_apple_chip_info,
        ANE_CORES,
        GPU_CORES,
    )
    _APPLE_SILICON_AVAILABLE = True
except ImportError:
    _APPLE_SILICON_AVAILABLE = False
    AppleSiliconBackend = None
    AppleSiliconConfig = None
    CoreMLConverter = None
    CoreMLModel = None
    ANEOptimizer = None
    PerformanceProfiler = None
    PreconvertedModelsRegistry = None
    ComputeUnits = None
    AppleChipVariant = None
    create_apple_silicon_backend = lambda **kwargs: None
    convert_to_coreml = lambda *args, **kwargs: None
    is_apple_silicon = lambda: False
    get_apple_chip_info = lambda: {}
    ANE_CORES = {}
    GPU_CORES = {}

# AMD ROCm backend (optional - requires ROCm on Linux)
try:
    from .amd import (
        AMDBackend,
        AMDConfig,
        AMDDeviceInfo,
        AMDModelWrapper,
        create_amd_backend,
        is_rocm_available,
        get_rocm_version,
        list_amd_gpus,
        detect_rocm_installation,
        query_amd_gpus_rocm_smi,
    )
    _AMD_AVAILABLE = True
except ImportError:
    _AMD_AVAILABLE = False
    AMDBackend = None
    AMDConfig = None
    AMDDeviceInfo = None
    AMDModelWrapper = None
    create_amd_backend = lambda **kwargs: None
    is_rocm_available = lambda: False
    get_rocm_version = lambda: None
    list_amd_gpus = lambda: []
    detect_rocm_installation = lambda: {"installed": False}
    query_amd_gpus_rocm_smi = lambda: []

# Intel oneAPI/OpenVINO backend (optional - requires IPEX or OpenVINO)
try:
    from .intel import (
        IntelBackend,
        IntelConfig,
        IntelDeviceInfo,
        IntelGPUType,
        IntelModelWrapper,
        create_intel_backend,
        is_intel_xpu_available,
        is_openvino_gpu_available,
        list_intel_gpus,
        get_ipex_version,
        get_openvino_version,
        detect_intel_gpu_type,
        query_intel_gpus_ipex,
        query_intel_gpus_openvino,
        query_intel_gpus_windows,
        INTEL_VRAM_ESTIMATES,
    )
    _INTEL_AVAILABLE = True
except ImportError:
    _INTEL_AVAILABLE = False
    IntelBackend = None
    IntelConfig = None
    IntelDeviceInfo = None
    IntelGPUType = None
    IntelModelWrapper = None
    create_intel_backend = lambda **kwargs: None
    is_intel_xpu_available = lambda: False
    is_openvino_gpu_available = lambda: False
    list_intel_gpus = lambda: []
    get_ipex_version = lambda: None
    get_openvino_version = lambda: None
    detect_intel_gpu_type = lambda name: "unknown"
    query_intel_gpus_ipex = lambda: []
    query_intel_gpus_openvino = lambda: []
    query_intel_gpus_windows = lambda: []
    INTEL_VRAM_ESTIMATES = {}

__all__ = [
    # Base backend
    "Backend",
    "BackendCapabilities",
    "get_backend",
    "get_backend_for_vendor",
    "list_backends",
    # TensorRT
    "TensorRTBackend",
    "TensorRTConfig",
    "create_tensorrt_backend",
    "accelerate_model",
    "is_tensorrt_available",
    "get_tensorrt_version",
    "clear_engine_cache",
    # Apple Silicon
    "AppleSiliconBackend",
    "AppleSiliconConfig",
    "CoreMLConverter",
    "CoreMLModel",
    "ANEOptimizer",
    "PerformanceProfiler",
    "PreconvertedModelsRegistry",
    "ComputeUnits",
    "AppleChipVariant",
    "create_apple_silicon_backend",
    "convert_to_coreml",
    "is_apple_silicon",
    "get_apple_chip_info",
    "ANE_CORES",
    "GPU_CORES",
    # AMD ROCm
    "AMDBackend",
    "AMDConfig",
    "AMDDeviceInfo",
    "AMDModelWrapper",
    "create_amd_backend",
    "is_rocm_available",
    "get_rocm_version",
    "list_amd_gpus",
    "detect_rocm_installation",
    "query_amd_gpus_rocm_smi",
    # Intel oneAPI/OpenVINO
    "IntelBackend",
    "IntelConfig",
    "IntelDeviceInfo",
    "IntelGPUType",
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
