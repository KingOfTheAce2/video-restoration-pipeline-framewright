"""Hardware Detection for FrameWright.

Unified hardware detection that works across ALL tiers:
- CPU only (no GPU)
- NVIDIA (CUDA)
- AMD (ROCm/Vulkan)
- Intel (oneAPI/Vulkan)
- Apple Silicon (Metal/CoreML)

Provides automatic hardware tier classification and capability detection.
"""

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendor identification."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class BackendType(Enum):
    """Available compute backend types."""
    CUDA = "cuda"           # NVIDIA CUDA
    TENSORRT = "tensorrt"   # NVIDIA TensorRT
    ROCM = "rocm"           # AMD ROCm
    VULKAN = "vulkan"       # Cross-platform Vulkan (ncnn)
    METAL = "metal"         # Apple Metal
    COREML = "coreml"       # Apple CoreML
    ONEAPI = "oneapi"       # Intel oneAPI
    DIRECTML = "directml"   # Windows DirectML (AMD/Intel/NVIDIA)
    OPENVINO = "openvino"   # Intel OpenVINO
    CPU = "cpu"             # CPU fallback


class HardwareTier(Enum):
    """Hardware capability tiers based on VRAM and features.

    Tiers determine optimal processing parameters like batch size,
    tile size, and model selection.
    """
    CPU_ONLY = "cpu_only"           # No GPU, CPU-only processing
    VRAM_4GB = "vram_4gb"           # 2-4GB VRAM (GTX 1050, RX 550)
    VRAM_8GB = "vram_8gb"           # 4-8GB VRAM (GTX 1070, RX 580)
    VRAM_12GB = "vram_12gb"         # 8-12GB VRAM (RTX 3060, RX 6700)
    VRAM_16GB_PLUS = "vram_16gb_plus"  # 12-16GB (RTX 3080, RX 6800)
    VRAM_24GB_PLUS = "vram_24gb_plus"  # 24GB+ (RTX 4090, A100)
    APPLE_SILICON = "apple_silicon"  # Apple M1/M2/M3 unified memory


@dataclass
class DeviceInfo:
    """Information about a single compute device."""
    index: int
    name: str
    vendor: GPUVendor
    total_memory_mb: int
    free_memory_mb: int = 0
    driver_version: str = ""
    compute_capability: str = ""  # CUDA compute capability
    is_dedicated: bool = True     # False for integrated GPUs
    supports_fp16: bool = True
    supports_int8: bool = False
    max_threads: int = 0

    @property
    def used_memory_mb(self) -> int:
        """Calculate used memory."""
        return self.total_memory_mb - self.free_memory_mb

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


@dataclass
class HardwareInfo:
    """Complete hardware information and capabilities.

    This dataclass contains all detected hardware information and
    computed capabilities for video processing.
    """
    # System info
    platform: str = ""          # win32, darwin, linux
    os_version: str = ""
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0

    # GPU info
    has_gpu: bool = False
    primary_device: Optional[DeviceInfo] = None
    all_devices: List[DeviceInfo] = field(default_factory=list)

    # Capability tier
    tier: HardwareTier = HardwareTier.CPU_ONLY

    # Available backends (in priority order)
    available_backends: List[BackendType] = field(default_factory=list)
    recommended_backend: BackendType = BackendType.CPU

    # Aggregate stats
    total_vram_mb: int = 0
    total_free_vram_mb: int = 0
    device_count: int = 0

    # Processing recommendations
    recommended_tile_size: int = 256
    recommended_batch_size: int = 1
    max_resolution: str = "720p"
    can_process_4k: bool = False
    use_fp16: bool = False

    # Feature flags
    has_cuda: bool = False
    has_vulkan: bool = False
    has_metal: bool = False
    has_rocm: bool = False
    has_directml: bool = False
    has_ncnn_vulkan: bool = False  # ncnn-vulkan binary available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "platform": self.platform,
            "os_version": self.os_version,
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "has_gpu": self.has_gpu,
            "tier": self.tier.value,
            "total_vram_mb": self.total_vram_mb,
            "device_count": self.device_count,
            "available_backends": [b.value for b in self.available_backends],
            "recommended_backend": self.recommended_backend.value,
            "recommended_tile_size": self.recommended_tile_size,
            "recommended_batch_size": self.recommended_batch_size,
            "max_resolution": self.max_resolution,
            "can_process_4k": self.can_process_4k,
            "devices": [
                {
                    "index": d.index,
                    "name": d.name,
                    "vendor": d.vendor.value,
                    "total_memory_mb": d.total_memory_mb,
                    "free_memory_mb": d.free_memory_mb,
                }
                for d in self.all_devices
            ],
        }


# =============================================================================
# Platform-specific detection
# =============================================================================

def _detect_nvidia_gpus() -> List[DeviceInfo]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        )

        if result.returncode != 0:
            logger.debug(f"nvidia-smi failed: {result.stderr}")
            return []

        devices = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            try:
                device = DeviceInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vendor=GPUVendor.NVIDIA,
                    total_memory_mb=int(parts[2]),
                    free_memory_mb=int(parts[3]),
                    driver_version=parts[4] if len(parts) > 4 else "",
                    compute_capability=parts[5] if len(parts) > 5 and parts[5] != "[N/A]" else "",
                    is_dedicated=True,
                    supports_fp16=True,
                    supports_int8=_check_nvidia_int8_support(parts[5] if len(parts) > 5 else ""),
                )
                devices.append(device)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse nvidia-smi line: {e}")
                continue

        return devices

    except subprocess.TimeoutExpired:
        logger.debug("nvidia-smi timed out")
        return []
    except Exception as e:
        logger.debug(f"NVIDIA detection failed: {e}")
        return []


def _check_nvidia_int8_support(compute_cap: str) -> bool:
    """Check if NVIDIA GPU supports INT8 inference."""
    if not compute_cap or compute_cap == "[N/A]":
        return False
    try:
        # INT8 requires compute capability >= 6.1 (Pascal and later)
        major, minor = compute_cap.split(".")
        return int(major) >= 7 or (int(major) == 6 and int(minor) >= 1)
    except (ValueError, AttributeError):
        return False


def _detect_windows_gpus() -> List[DeviceInfo]:
    """Detect GPUs on Windows via WMI (for AMD/Intel)."""
    if platform.system() != "Windows":
        return []

    try:
        cmd = [
            "powershell", "-Command",
            "Get-WmiObject Win32_VideoController | "
            "Select-Object Name, AdapterRAM, DriverVersion, PNPDeviceID | "
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
            return []

        import json
        data = json.loads(result.stdout)

        if isinstance(data, dict):
            data = [data]

        devices = []
        for idx, gpu_data in enumerate(data):
            name = gpu_data.get("Name", "Unknown GPU")

            # Skip virtual/basic display adapters
            if "basic display" in name.lower() or "microsoft" in name.lower():
                continue

            # Skip NVIDIA (detected via nvidia-smi)
            if _is_nvidia_name(name):
                continue

            # Get RAM (handle WMI overflow for > 4GB)
            ram_bytes = gpu_data.get("AdapterRAM", 0) or 0
            if ram_bytes < 0:
                ram_bytes = 4294967296 + ram_bytes
            total_mb = int(ram_bytes / (1024 * 1024))

            vendor = _detect_vendor_from_name(name)
            is_dedicated = _is_dedicated_gpu(name)

            device = DeviceInfo(
                index=idx,
                name=name,
                vendor=vendor,
                total_memory_mb=total_mb,
                free_memory_mb=int(total_mb * 0.9),  # Estimate
                driver_version=gpu_data.get("DriverVersion", ""),
                is_dedicated=is_dedicated,
                supports_fp16=True,
            )
            devices.append(device)

        return devices

    except Exception as e:
        logger.debug(f"Windows GPU detection failed: {e}")
        return []


def _detect_apple_silicon() -> Optional[DeviceInfo]:
    """Detect Apple Silicon (M1/M2/M3) GPUs."""
    if platform.system() != "Darwin":
        return None

    try:
        # Check for Apple Silicon
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        cpu_brand = result.stdout.strip()
        if "Apple" not in cpu_brand:
            return None

        # Get unified memory size
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        total_mem_bytes = int(result.stdout.strip())
        # Apple Silicon uses unified memory, GPU can use ~75% of RAM
        gpu_memory_mb = int((total_mem_bytes * 0.75) / (1024 * 1024))

        # Determine chip variant
        chip_name = "Apple Silicon"
        if "M1" in cpu_brand:
            chip_name = "Apple M1"
        elif "M2" in cpu_brand:
            chip_name = "Apple M2"
        elif "M3" in cpu_brand:
            chip_name = "Apple M3"

        return DeviceInfo(
            index=0,
            name=chip_name,
            vendor=GPUVendor.APPLE,
            total_memory_mb=gpu_memory_mb,
            free_memory_mb=int(gpu_memory_mb * 0.8),  # Estimate
            is_dedicated=True,  # Unified memory acts as dedicated
            supports_fp16=True,
            supports_int8=True,  # Neural Engine
        )

    except Exception as e:
        logger.debug(f"Apple Silicon detection failed: {e}")
        return None


def _detect_linux_gpus() -> List[DeviceInfo]:
    """Detect non-NVIDIA GPUs on Linux."""
    if platform.system() != "Linux":
        return []

    devices = []

    # Try lspci for basic detection
    try:
        result = subprocess.run(
            ["lspci", "-nn"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                line_lower = line.lower()

                # Skip NVIDIA (detected via nvidia-smi)
                if "nvidia" in line_lower:
                    continue

                if "vga" in line_lower or "display" in line_lower or "3d" in line_lower:
                    if "amd" in line_lower or "radeon" in line_lower:
                        # AMD GPU
                        name = _extract_gpu_name(line)
                        devices.append(DeviceInfo(
                            index=len(devices),
                            name=name or "AMD GPU",
                            vendor=GPUVendor.AMD,
                            total_memory_mb=_estimate_amd_vram(name),
                            free_memory_mb=_estimate_amd_vram(name),
                            is_dedicated=_is_dedicated_gpu(name),
                        ))
                    elif "intel" in line_lower:
                        # Intel GPU
                        name = _extract_gpu_name(line)
                        devices.append(DeviceInfo(
                            index=len(devices),
                            name=name or "Intel GPU",
                            vendor=GPUVendor.INTEL,
                            total_memory_mb=_estimate_intel_vram(name),
                            free_memory_mb=_estimate_intel_vram(name),
                            is_dedicated="arc" in line_lower,
                        ))

    except Exception as e:
        logger.debug(f"Linux GPU detection failed: {e}")

    return devices


# =============================================================================
# Helper functions
# =============================================================================

def _is_nvidia_name(name: str) -> bool:
    """Check if GPU name indicates NVIDIA."""
    keywords = ["nvidia", "geforce", "quadro", "tesla", "rtx", "gtx"]
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)


def _detect_vendor_from_name(name: str) -> GPUVendor:
    """Detect GPU vendor from device name."""
    name_lower = name.lower()

    if _is_nvidia_name(name):
        return GPUVendor.NVIDIA
    elif any(kw in name_lower for kw in ["amd", "radeon", "rx ", "vega", "navi"]):
        return GPUVendor.AMD
    elif any(kw in name_lower for kw in ["intel", "uhd", "iris", "arc", "xe"]):
        return GPUVendor.INTEL
    elif any(kw in name_lower for kw in ["apple", "m1", "m2", "m3"]):
        return GPUVendor.APPLE

    return GPUVendor.UNKNOWN


def _is_dedicated_gpu(name: str) -> bool:
    """Check if GPU is dedicated (not integrated)."""
    integrated_keywords = [
        "uhd", "iris", "integrated", "igpu", "vega 8", "vega 6",
        "vega 3", "graphics 630", "graphics 530", "hd graphics"
    ]
    name_lower = name.lower()
    return not any(kw in name_lower for kw in integrated_keywords)


def _extract_gpu_name(lspci_line: str) -> Optional[str]:
    """Extract GPU name from lspci output line."""
    try:
        # Format: "00:02.0 VGA compatible controller [0300]: Intel Corporation ..."
        parts = lspci_line.split(": ", 1)
        if len(parts) > 1:
            return parts[1].split(" [")[0].strip()
    except Exception:
        pass
    return None


def _estimate_amd_vram(name: Optional[str]) -> int:
    """Estimate AMD GPU VRAM from model name."""
    if not name:
        return 4096

    name_lower = name.lower()

    # RX 7000 series
    if "7900" in name_lower:
        return 24576 if "xtx" in name_lower else 20480
    if "7800" in name_lower:
        return 16384
    if "7600" in name_lower:
        return 8192

    # RX 6000 series
    if "6900" in name_lower or "6950" in name_lower:
        return 16384
    if "6800" in name_lower:
        return 16384
    if "6700" in name_lower:
        return 12288
    if "6600" in name_lower:
        return 8192
    if "6500" in name_lower:
        return 4096

    # RX 5000 series
    if "5700" in name_lower:
        return 8192
    if "5600" in name_lower:
        return 6144
    if "5500" in name_lower:
        return 4096

    return 4096  # Default


def _estimate_intel_vram(name: Optional[str]) -> int:
    """Estimate Intel GPU VRAM from model name."""
    if not name:
        return 2048

    name_lower = name.lower()

    # Arc series (dedicated)
    if "a770" in name_lower:
        return 16384
    if "a750" in name_lower:
        return 8192
    if "a380" in name_lower:
        return 6144

    # Integrated (share system RAM)
    if "iris xe" in name_lower or "iris plus" in name_lower:
        return 4096  # Can use more, but limited
    if "uhd" in name_lower:
        return 2048

    return 2048


def _check_backend_availability() -> Dict[BackendType, bool]:
    """Check which compute backends are available."""
    available = {bt: False for bt in BackendType}
    available[BackendType.CPU] = True  # Always available

    # CUDA check
    try:
        import torch
        available[BackendType.CUDA] = torch.cuda.is_available()
    except ImportError:
        pass

    # TensorRT check
    if available[BackendType.CUDA]:
        try:
            import tensorrt
            available[BackendType.TENSORRT] = True
        except ImportError:
            pass

    # Metal check (macOS)
    if platform.system() == "Darwin":
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available[BackendType.METAL] = True
        except ImportError:
            pass

        # CoreML check
        try:
            import coremltools
            available[BackendType.COREML] = True
        except ImportError:
            pass

    # ROCm check (AMD on Linux)
    if platform.system() == "Linux":
        try:
            import torch
            if torch.cuda.is_available() and "rocm" in torch.version.cuda.lower():
                available[BackendType.ROCM] = True
        except (ImportError, AttributeError):
            pass

    # DirectML check (Windows)
    if platform.system() == "Windows":
        try:
            import torch_directml
            available[BackendType.DIRECTML] = True
        except ImportError:
            pass

    # Vulkan check (ncnn-vulkan binary)
    ncnn_names = ["realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"]
    for name in ncnn_names:
        if shutil.which(name):
            available[BackendType.VULKAN] = True
            break

    # Check ~/.framewright/bin
    home = Path.home()
    ncnn_paths = [
        home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan.exe",
        home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan",
    ]
    for path in ncnn_paths:
        if path.exists():
            available[BackendType.VULKAN] = True
            break

    # OpenVINO check
    try:
        from openvino.runtime import Core
        available[BackendType.OPENVINO] = True
    except ImportError:
        pass

    # oneAPI check
    if platform.system() in ("Linux", "Windows"):
        try:
            import intel_extension_for_pytorch
            available[BackendType.ONEAPI] = True
        except ImportError:
            pass

    return available


def _determine_tier(vram_mb: int, vendor: GPUVendor) -> HardwareTier:
    """Determine hardware tier based on VRAM and vendor."""
    if vendor == GPUVendor.APPLE:
        return HardwareTier.APPLE_SILICON

    if vram_mb == 0:
        return HardwareTier.CPU_ONLY
    elif vram_mb < 4096:
        return HardwareTier.VRAM_4GB
    elif vram_mb < 8192:
        return HardwareTier.VRAM_8GB
    elif vram_mb < 12288:
        return HardwareTier.VRAM_12GB
    elif vram_mb < 20480:
        return HardwareTier.VRAM_16GB_PLUS
    else:
        return HardwareTier.VRAM_24GB_PLUS


def _get_processing_recommendations(tier: HardwareTier) -> Tuple[int, int, str, bool]:
    """Get processing recommendations for a hardware tier.

    Returns:
        Tuple of (tile_size, batch_size, max_resolution, can_4k)
    """
    recommendations = {
        HardwareTier.CPU_ONLY: (128, 1, "480p", False),
        HardwareTier.VRAM_4GB: (192, 1, "720p", False),
        HardwareTier.VRAM_8GB: (256, 2, "1080p", False),
        HardwareTier.VRAM_12GB: (384, 4, "1440p", True),
        HardwareTier.VRAM_16GB_PLUS: (512, 8, "4K", True),
        HardwareTier.VRAM_24GB_PLUS: (0, 16, "8K", True),  # 0 = no tiling
        HardwareTier.APPLE_SILICON: (384, 4, "4K", True),
    }
    return recommendations.get(tier, (256, 1, "1080p", False))


def _select_recommended_backend(
    vendor: GPUVendor,
    available: Dict[BackendType, bool],
) -> BackendType:
    """Select the best available backend for a vendor."""
    # Priority order by vendor
    priorities = {
        GPUVendor.NVIDIA: [
            BackendType.TENSORRT,
            BackendType.CUDA,
            BackendType.VULKAN,
            BackendType.CPU,
        ],
        GPUVendor.AMD: [
            BackendType.ROCM,
            BackendType.VULKAN,
            BackendType.DIRECTML,
            BackendType.CPU,
        ],
        GPUVendor.INTEL: [
            BackendType.ONEAPI,
            BackendType.OPENVINO,
            BackendType.VULKAN,
            BackendType.DIRECTML,
            BackendType.CPU,
        ],
        GPUVendor.APPLE: [
            BackendType.METAL,
            BackendType.COREML,
            BackendType.CPU,
        ],
        GPUVendor.UNKNOWN: [
            BackendType.VULKAN,
            BackendType.DIRECTML,
            BackendType.CPU,
        ],
    }

    for backend in priorities.get(vendor, [BackendType.CPU]):
        if available.get(backend, False):
            return backend

    return BackendType.CPU


def _get_system_info() -> Tuple[str, str, int, int, int]:
    """Get system information.

    Returns:
        Tuple of (cpu_name, os_version, cpu_cores, ram_total_mb, ram_available_mb)
    """
    cpu_name = "Unknown"
    os_version = platform.release()
    cpu_cores = 0
    ram_total_mb = 0
    ram_available_mb = 0

    try:
        import os as os_module
        cpu_cores = os_module.cpu_count() or 0
    except Exception:
        pass

    # CPU name
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cpu_name = result.stdout.strip()
    except Exception:
        pass

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_total_mb = int(mem.total / (1024 * 1024))
        ram_available_mb = int(mem.available / (1024 * 1024))
    except ImportError:
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            ram_total_mb = int(line.split()[1]) // 1024
                        elif "MemAvailable" in line:
                            ram_available_mb = int(line.split()[1]) // 1024
            except Exception:
                pass

    return cpu_name, os_version, cpu_cores, ram_total_mb, ram_available_mb


# =============================================================================
# Main detection functions
# =============================================================================

def detect_hardware(force_refresh: bool = False) -> HardwareInfo:
    """Detect all hardware and capabilities.

    This is the main entry point for hardware detection. It detects all
    GPUs, determines capabilities, and recommends optimal settings.

    Args:
        force_refresh: Force re-detection even if cached

    Returns:
        HardwareInfo with complete hardware information
    """
    if not force_refresh:
        cached = _get_cached_hardware_info()
        if cached is not None:
            return cached

    info = HardwareInfo(platform=platform.system())

    # System info
    cpu_name, os_version, cores, ram_total, ram_avail = _get_system_info()
    info.cpu_name = cpu_name
    info.os_version = os_version
    info.cpu_cores = cores
    info.ram_total_mb = ram_total
    info.ram_available_mb = ram_avail

    # Detect all GPUs
    all_devices: List[DeviceInfo] = []

    # NVIDIA GPUs (highest priority)
    nvidia_devices = _detect_nvidia_gpus()
    all_devices.extend(nvidia_devices)

    # Apple Silicon
    apple_device = _detect_apple_silicon()
    if apple_device:
        all_devices.append(apple_device)

    # Platform-specific detection for AMD/Intel
    if platform.system() == "Windows":
        windows_devices = _detect_windows_gpus()
        all_devices.extend(windows_devices)
    elif platform.system() == "Linux":
        linux_devices = _detect_linux_gpus()
        all_devices.extend(linux_devices)

    # Sort devices: dedicated first, then by VRAM
    all_devices.sort(key=lambda d: (not d.is_dedicated, -d.total_memory_mb))

    # Re-index after sorting
    for idx, device in enumerate(all_devices):
        device.index = idx

    info.all_devices = all_devices
    info.device_count = len(all_devices)

    # Select primary device
    if all_devices:
        info.has_gpu = True
        info.primary_device = all_devices[0]
        info.total_vram_mb = sum(d.total_memory_mb for d in all_devices)
        info.total_free_vram_mb = sum(d.free_memory_mb for d in all_devices)

    # Check backend availability
    backend_available = _check_backend_availability()
    info.has_cuda = backend_available.get(BackendType.CUDA, False)
    info.has_vulkan = backend_available.get(BackendType.VULKAN, False)
    info.has_metal = backend_available.get(BackendType.METAL, False)
    info.has_rocm = backend_available.get(BackendType.ROCM, False)
    info.has_directml = backend_available.get(BackendType.DIRECTML, False)
    info.has_ncnn_vulkan = backend_available.get(BackendType.VULKAN, False)

    # Build available backends list (in priority order)
    info.available_backends = [
        bt for bt in BackendType
        if backend_available.get(bt, False)
    ]

    # Determine tier and recommendations
    if info.primary_device:
        info.tier = _determine_tier(
            info.primary_device.total_memory_mb,
            info.primary_device.vendor
        )
        info.recommended_backend = _select_recommended_backend(
            info.primary_device.vendor,
            backend_available
        )
        info.use_fp16 = info.primary_device.supports_fp16
    else:
        info.tier = HardwareTier.CPU_ONLY
        info.recommended_backend = BackendType.CPU

    # Processing recommendations
    tile, batch, max_res, can_4k = _get_processing_recommendations(info.tier)
    info.recommended_tile_size = tile
    info.recommended_batch_size = batch
    info.max_resolution = max_res
    info.can_process_4k = can_4k

    # Cache the result
    _cache_hardware_info(info)

    return info


# Caching
_hardware_info_cache: Optional[HardwareInfo] = None


def _get_cached_hardware_info() -> Optional[HardwareInfo]:
    """Get cached hardware info."""
    global _hardware_info_cache
    return _hardware_info_cache


def _cache_hardware_info(info: HardwareInfo) -> None:
    """Cache hardware info."""
    global _hardware_info_cache
    _hardware_info_cache = info


# =============================================================================
# Convenience functions
# =============================================================================

def get_hardware_info() -> HardwareInfo:
    """Get hardware information (cached).

    This is the recommended entry point for getting hardware info.
    Results are cached for performance.

    Returns:
        HardwareInfo with all detected capabilities
    """
    return detect_hardware(force_refresh=False)


def get_hardware_tier() -> HardwareTier:
    """Get the hardware capability tier.

    Returns:
        HardwareTier enum value
    """
    return get_hardware_info().tier


def is_gpu_available() -> bool:
    """Check if any GPU is available.

    Returns:
        True if at least one GPU is detected
    """
    return get_hardware_info().has_gpu


def get_vram_mb() -> int:
    """Get total VRAM across all GPUs in MB.

    Returns:
        Total VRAM in megabytes, 0 if no GPU
    """
    return get_hardware_info().total_vram_mb


def get_optimal_device() -> int:
    """Get the optimal device index for processing.

    Returns:
        Device index (0 if no GPU available)
    """
    info = get_hardware_info()
    if info.primary_device:
        return info.primary_device.index
    return 0


def get_available_backends() -> List[BackendType]:
    """Get list of available compute backends.

    Returns:
        List of BackendType enums for available backends
    """
    return get_hardware_info().available_backends


# =============================================================================
# Integration with existing modules
# =============================================================================

def get_legacy_gpu_info():
    """Get GPU info in legacy format for compatibility.

    This function wraps the new detection to provide compatibility
    with existing code that uses the old gpu.py module.
    """
    # Import existing module to avoid circular imports
    from ...utils.gpu import GPUInfo as LegacyGPUInfo, GPUVendor as LegacyVendor

    info = get_hardware_info()
    legacy_devices = []

    vendor_map = {
        GPUVendor.NVIDIA: LegacyVendor.NVIDIA,
        GPUVendor.AMD: LegacyVendor.AMD,
        GPUVendor.INTEL: LegacyVendor.INTEL,
        GPUVendor.APPLE: LegacyVendor.UNKNOWN,
        GPUVendor.UNKNOWN: LegacyVendor.UNKNOWN,
    }

    for device in info.all_devices:
        legacy = LegacyGPUInfo(
            index=device.index,
            name=device.name,
            total_memory_mb=device.total_memory_mb,
            used_memory_mb=device.total_memory_mb - device.free_memory_mb,
            free_memory_mb=device.free_memory_mb,
            utilization_percent=0.0,
            vendor=vendor_map.get(device.vendor, LegacyVendor.UNKNOWN),
            driver_version=device.driver_version,
            vulkan_supported=device.vendor in (GPUVendor.NVIDIA, GPUVendor.AMD, GPUVendor.INTEL),
            cuda_supported=device.vendor == GPUVendor.NVIDIA,
        )
        legacy_devices.append(legacy)

    return legacy_devices
