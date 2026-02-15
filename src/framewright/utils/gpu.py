"""GPU and VRAM monitoring utilities for FrameWright.

Provides VRAM monitoring, adaptive tile sizing, and GPU capability detection.
Supports NVIDIA (CUDA), AMD (Vulkan/ROCm), and Intel (Vulkan) GPUs.
"""
import logging
import math
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendor identification."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """GPU device information."""
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: float
    temperature_celsius: Optional[float] = None
    vendor: GPUVendor = GPUVendor.UNKNOWN
    driver_version: str = ""
    vulkan_supported: bool = False
    cuda_supported: bool = False

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100

    @property
    def is_dedicated(self) -> bool:
        """Check if this is a dedicated GPU (not integrated)."""
        integrated_keywords = ["uhd", "iris", "integrated", "igpu", "vega 8", "vega 6"]
        name_lower = self.name.lower()
        return not any(kw in name_lower for kw in integrated_keywords)


def is_nvidia_gpu_available() -> bool:
    """Check if nvidia-smi is available."""
    return shutil.which("nvidia-smi") is not None


def is_gpu_available() -> bool:
    """Check if any GPU is available for processing.

    Returns True if NVIDIA GPU (via nvidia-smi) or any other GPU
    is detected on the system.
    """
    if is_nvidia_gpu_available():
        return True
    # Fallback: check for any GPU via platform-specific detection
    if platform.system() == "Windows":
        try:
            gpus = get_windows_gpu_info()
            return len(gpus) > 0
        except Exception:
            return False
    return False


def _detect_vendor(gpu_name: str) -> GPUVendor:
    """Detect GPU vendor from device name."""
    name_lower = gpu_name.lower()
    if any(kw in name_lower for kw in ["nvidia", "geforce", "quadro", "tesla", "rtx", "gtx"]):
        return GPUVendor.NVIDIA
    elif any(kw in name_lower for kw in ["amd", "radeon", "rx ", "vega", "navi"]):
        return GPUVendor.AMD
    elif any(kw in name_lower for kw in ["intel", "uhd", "iris", "arc"]):
        return GPUVendor.INTEL
    return GPUVendor.UNKNOWN


def get_windows_gpu_info() -> List[GPUInfo]:
    """Get GPU information on Windows via WMI.

    Detects AMD, Intel, and NVIDIA GPUs using Windows Management Instrumentation.

    Returns:
        List of GPUInfo for all detected GPUs
    """
    if platform.system() != "Windows":
        return []

    gpus = []

    try:
        # Use PowerShell to query WMI for video controllers
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
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )

        if result.returncode != 0:
            logger.warning(f"WMI query failed: {result.stderr}")
            return []

        import json
        data = json.loads(result.stdout)

        # Handle single GPU (returns dict) vs multiple (returns list)
        if isinstance(data, dict):
            data = [data]

        for idx, gpu_data in enumerate(data):
            name = gpu_data.get("Name", "Unknown GPU")

            # Skip Microsoft Basic Display Adapter (virtual)
            if "basic display" in name.lower() or "microsoft" in name.lower():
                continue

            # AdapterRAM is in bytes, convert to MB
            ram_bytes = gpu_data.get("AdapterRAM", 0)
            if ram_bytes is None:
                ram_bytes = 0
            # Handle potential overflow (WMI reports 4GB+ as negative or wrapped)
            if ram_bytes < 0:
                ram_bytes = 4294967296 + ram_bytes  # 2^32 + negative value
            total_mb = int(ram_bytes / (1024 * 1024))

            vendor = _detect_vendor(name)
            driver = gpu_data.get("DriverVersion", "")

            # For non-NVIDIA, we can't easily get used/free memory
            # Estimate free as 90% of total (conservative)
            estimated_free = int(total_mb * 0.9)

            gpu = GPUInfo(
                index=idx,
                name=name,
                total_memory_mb=total_mb,
                used_memory_mb=total_mb - estimated_free,
                free_memory_mb=estimated_free,
                utilization_percent=0.0,  # Can't easily get this for AMD/Intel
                vendor=vendor,
                driver_version=driver,
                vulkan_supported=(vendor in [GPUVendor.AMD, GPUVendor.INTEL, GPUVendor.NVIDIA]),
                cuda_supported=(vendor == GPUVendor.NVIDIA),
            )
            gpus.append(gpu)
            logger.debug(f"Detected GPU: {name} ({vendor.value}, {total_mb}MB)")

        # Sort: dedicated GPUs first, then by VRAM
        gpus.sort(key=lambda g: (not g.is_dedicated, -g.total_memory_mb))

        # Re-index after sorting
        for idx, gpu in enumerate(gpus):
            gpu.index = idx

        return gpus

    except subprocess.TimeoutExpired:
        logger.warning("WMI GPU query timed out")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse WMI GPU data: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to get Windows GPU info: {e}")
        return []


def get_all_gpus_multivendor() -> List[GPUInfo]:
    """Get all GPUs from all vendors (NVIDIA, AMD, Intel).

    Combines NVIDIA-specific detection (nvidia-smi) with generic
    Windows WMI detection for AMD/Intel GPUs.

    Returns:
        List of GPUInfo for all detected GPUs
    """
    all_gpus = []
    nvidia_names = set()

    # First, try NVIDIA-specific detection (more accurate for NVIDIA)
    if is_nvidia_gpu_available():
        nvidia_gpus = get_all_gpu_info()
        for gpu in nvidia_gpus:
            gpu.vendor = GPUVendor.NVIDIA
            gpu.cuda_supported = True
            gpu.vulkan_supported = True
            all_gpus.append(gpu)
            nvidia_names.add(gpu.name.lower())

    # Then get Windows WMI GPUs (for AMD/Intel, or if nvidia-smi failed)
    if platform.system() == "Windows":
        wmi_gpus = get_windows_gpu_info()
        for gpu in wmi_gpus:
            # Skip if we already have this GPU from nvidia-smi
            if gpu.name.lower() in nvidia_names:
                continue
            # Skip if it's NVIDIA and we already got NVIDIA GPUs
            if gpu.vendor == GPUVendor.NVIDIA and nvidia_names:
                continue
            all_gpus.append(gpu)

    # Re-index all GPUs
    for idx, gpu in enumerate(all_gpus):
        gpu.index = idx

    return all_gpus


def get_best_gpu() -> Optional[GPUInfo]:
    """Get the best available GPU for processing.

    Prioritizes: NVIDIA (CUDA) > AMD (Vulkan) > Intel (Vulkan)
    Within each vendor, prioritizes by VRAM.

    Returns:
        Best GPUInfo or None if no GPU detected
    """
    gpus = get_all_gpus_multivendor()
    if not gpus:
        return None

    # Priority: dedicated > integrated, then NVIDIA > AMD > Intel, then by VRAM
    def gpu_priority(g: GPUInfo) -> Tuple:
        vendor_priority = {
            GPUVendor.NVIDIA: 0,
            GPUVendor.AMD: 1,
            GPUVendor.INTEL: 2,
            GPUVendor.UNKNOWN: 3,
        }
        return (
            not g.is_dedicated,  # Dedicated first (False < True)
            vendor_priority.get(g.vendor, 3),
            -g.total_memory_mb,  # More VRAM first
        )

    gpus.sort(key=gpu_priority)
    return gpus[0]


def get_gpu_memory_info(device_id: Optional[int] = None) -> Optional[Dict[str, int]]:
    """Get current GPU memory status.

    Args:
        device_id: Optional specific GPU device ID

    Returns:
        Dictionary with total_mb, used_mb, free_mb or None if unavailable
    """
    if not is_nvidia_gpu_available():
        return None

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ]

        if device_id is not None:
            cmd.extend(["-i", str(device_id)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {result.stderr}")
            return None

        # Parse first GPU if multiple available
        line = result.stdout.strip().split("\n")[0]
        total, used, free = map(int, line.split(", "))

        return {
            "total_mb": total,
            "used_mb": used,
            "free_mb": free,
            "usage_percent": (used / total * 100) if total > 0 else 0
        }

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return None
    except (ValueError, IndexError, FileNotFoundError) as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def get_all_gpu_info() -> List[GPUInfo]:
    """Get information about all available GPUs.

    Returns:
        List of GPUInfo objects for each GPU
    """
    if not is_nvidia_gpu_available():
        return []

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(", ")]

            try:
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_memory_mb=int(parts[2]),
                    used_memory_mb=int(parts[3]),
                    free_memory_mb=int(parts[4]),
                    utilization_percent=float(parts[5]) if parts[5] != "[N/A]" else 0.0,
                    temperature_celsius=float(parts[6]) if parts[6] != "[N/A]" else None,
                )
                gpus.append(gpu)
            except (ValueError, IndexError):
                continue

        return gpus

    except Exception as e:
        logger.warning(f"Failed to enumerate GPUs: {e}")
        return []


def get_optimal_device() -> Optional[int]:
    """Get the optimal GPU device for processing.

    Selects the GPU with most free memory.

    Returns:
        GPU device index or None if no GPU available
    """
    gpus = get_all_gpu_info()
    if not gpus:
        return None

    # Sort by free memory, descending
    gpus.sort(key=lambda g: g.free_memory_mb, reverse=True)
    return gpus[0].index


def calculate_optimal_tile_size(
    frame_resolution: Tuple[int, int],
    scale_factor: int,
    available_vram_mb: Optional[int] = None,
    model_name: str = "realesrgan-x4plus",
    safety_factor: float = 0.7,
) -> int:
    """Calculate optimal tile size based on available VRAM and frame size.

    Args:
        frame_resolution: (width, height) of input frames
        scale_factor: Upscaling factor (2 or 4)
        available_vram_mb: Available VRAM in MB (auto-detected if None)
        model_name: Real-ESRGAN model name
        safety_factor: Fraction of available VRAM to use (0.0-1.0)

    Returns:
        Optimal tile size (0 means no tiling needed)
    """
    width, height = frame_resolution

    # Auto-detect VRAM if not provided
    if available_vram_mb is None:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            available_vram_mb = gpu_info["free_mb"]
        else:
            # Conservative default if can't detect
            available_vram_mb = 2048

    # Apply safety factor
    usable_vram_mb = int(available_vram_mb * safety_factor)

    # Model-specific memory coefficients (empirically determined)
    # These represent approximate MB per megapixel at output resolution
    model_coefficients = {
        "realesrgan-x4plus": 450,
        "realesrgan-x4plus-anime": 400,
        "realesrgan-x2plus": 250,
        "realesr-animevideov3": 350,
    }

    coeff = model_coefficients.get(model_name, 450)

    # Calculate output size
    output_width = width * scale_factor
    output_height = height * scale_factor
    output_megapixels = (output_width * output_height) / 1_000_000

    # Estimate VRAM needed for full frame
    estimated_vram_mb = output_megapixels * coeff

    # If we have enough VRAM, no tiling needed
    if estimated_vram_mb <= usable_vram_mb:
        logger.debug(f"No tiling needed: {estimated_vram_mb:.0f}MB estimated, {usable_vram_mb}MB available")
        return 0

    # Calculate tile size that fits in available VRAM
    # tile_vram = (tile_size * scale_factor)^2 / 1e6 * coeff
    max_output_tile_pixels = (usable_vram_mb / coeff) * 1_000_000
    max_output_tile_size = int(math.sqrt(max_output_tile_pixels))

    # Convert to input tile size
    max_input_tile_size = max_output_tile_size // scale_factor

    # Round down to nearest 32 (for GPU alignment)
    tile_size = (max_input_tile_size // 32) * 32

    # Ensure minimum tile size
    tile_size = max(128, tile_size)

    # Ensure tile size is not larger than input
    tile_size = min(tile_size, min(width, height))

    logger.info(
        f"Calculated tile size: {tile_size} "
        f"(frame: {width}x{height}, VRAM: {usable_vram_mb}MB available)"
    )

    return tile_size


def get_adaptive_tile_sequence(
    frame_resolution: Tuple[int, int],
    scale_factor: int,
    starting_tile_size: Optional[int] = None,
    min_tile_size: int = 128,
) -> List[int]:
    """Generate a sequence of tile sizes for progressive fallback.

    Useful for retrying with smaller tiles after VRAM errors.

    Args:
        frame_resolution: (width, height) of frames
        scale_factor: Upscaling factor
        starting_tile_size: Initial tile size (auto-calculated if None)
        min_tile_size: Minimum tile size to try

    Returns:
        List of tile sizes in decreasing order
    """
    if starting_tile_size is None:
        starting_tile_size = calculate_optimal_tile_size(
            frame_resolution, scale_factor
        )

    if starting_tile_size == 0:
        # No tiling needed initially, but provide fallback
        width, height = frame_resolution
        starting_tile_size = min(width, height)

    # Generate sequence: starting, 75%, 50%, 25%
    tile_sizes = []
    current = starting_tile_size

    while current >= min_tile_size:
        # Round to nearest 32
        rounded = (current // 32) * 32
        if rounded >= min_tile_size and rounded not in tile_sizes:
            tile_sizes.append(rounded)
        current = int(current * 0.75)

    # Ensure we have at least min_tile_size
    if min_tile_size not in tile_sizes:
        tile_sizes.append(min_tile_size)

    return tile_sizes


class VRAMMonitor:
    """Monitor VRAM usage during processing."""

    def __init__(self, device_id: int = 0, threshold_mb: int = 500):
        """Initialize VRAM monitor.

        Args:
            device_id: GPU device ID to monitor
            threshold_mb: Warning threshold in MB
        """
        self.device_id = device_id
        self.threshold_mb = threshold_mb
        self.peak_usage_mb = 0
        self.samples: List[Dict[str, int]] = []

    def sample(self) -> Optional[Dict[str, int]]:
        """Take a VRAM usage sample.

        Returns:
            Current memory info or None
        """
        info = get_gpu_memory_info(self.device_id)
        if info:
            self.samples.append(info)
            if info["used_mb"] > self.peak_usage_mb:
                self.peak_usage_mb = info["used_mb"]

            # Check threshold
            if info["free_mb"] < self.threshold_mb:
                logger.warning(
                    f"Low VRAM warning: {info['free_mb']}MB free "
                    f"(threshold: {self.threshold_mb}MB)"
                )

        return info

    def is_low_memory(self) -> bool:
        """Check if memory is below threshold."""
        info = get_gpu_memory_info(self.device_id)
        return info is not None and info["free_mb"] < self.threshold_mb

    def get_statistics(self) -> Dict[str, float]:
        """Get memory usage statistics.

        Returns:
            Dictionary with min, max, avg memory usage
        """
        if not self.samples:
            return {"min_mb": 0, "max_mb": 0, "avg_mb": 0}

        used_values = [s["used_mb"] for s in self.samples]

        return {
            "min_mb": min(used_values),
            "max_mb": max(used_values),
            "avg_mb": sum(used_values) / len(used_values),
            "peak_mb": self.peak_usage_mb,
            "samples": len(self.samples),
        }


def wait_for_vram(
    required_mb: int,
    device_id: int = 0,
    timeout_seconds: float = 30.0,
    check_interval: float = 1.0,
) -> bool:
    """Wait for sufficient VRAM to become available.

    Args:
        required_mb: Required free VRAM in MB
        device_id: GPU device ID
        timeout_seconds: Maximum time to wait
        check_interval: Time between checks

    Returns:
        True if sufficient VRAM available, False if timeout
    """
    import time

    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        info = get_gpu_memory_info(device_id)

        if info and info["free_mb"] >= required_mb:
            return True

        time.sleep(check_interval)

    return False


class MultiGPUManager:
    """Manager for distributing work across multiple GPUs.

    Supports load balancing and automatic GPU selection based on
    available memory and utilization.
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        load_balance_strategy: str = "memory",
    ):
        """Initialize multi-GPU manager.

        Args:
            gpu_ids: List of GPU IDs to use (None = auto-detect all)
            load_balance_strategy: How to distribute work:
                - "memory": Prioritize GPUs with most free memory
                - "round_robin": Distribute evenly across GPUs
                - "utilization": Prioritize least utilized GPUs
        """
        self.strategy = load_balance_strategy
        self._current_index = 0
        self._gpu_assignments: Dict[int, int] = {}

        # Auto-detect GPUs if not specified
        if gpu_ids is None:
            all_gpus = get_all_gpu_info()
            self.gpu_ids = [g.index for g in all_gpus]
        else:
            self.gpu_ids = gpu_ids

        if not self.gpu_ids:
            logger.warning("No GPUs detected for multi-GPU processing")

        logger.info(f"MultiGPUManager initialized with GPUs: {self.gpu_ids}")

    @property
    def gpu_count(self) -> int:
        """Get number of available GPUs."""
        return len(self.gpu_ids)

    def is_available(self) -> bool:
        """Check if multi-GPU processing is available."""
        return self.gpu_count > 1

    def get_next_gpu(self) -> int:
        """Get the next GPU to use based on load balancing strategy.

        Returns:
            GPU device ID
        """
        if not self.gpu_ids:
            return 0

        if self.strategy == "round_robin":
            gpu_id = self.gpu_ids[self._current_index % len(self.gpu_ids)]
            self._current_index += 1
            return gpu_id

        elif self.strategy == "utilization":
            gpus = get_all_gpu_info()
            valid_gpus = [g for g in gpus if g.index in self.gpu_ids]
            if not valid_gpus:
                return self.gpu_ids[0]
            # Sort by utilization (lowest first)
            valid_gpus.sort(key=lambda g: g.utilization_percent)
            return valid_gpus[0].index

        else:  # "memory" strategy (default)
            gpus = get_all_gpu_info()
            valid_gpus = [g for g in gpus if g.index in self.gpu_ids]
            if not valid_gpus:
                return self.gpu_ids[0]
            # Sort by free memory (highest first)
            valid_gpus.sort(key=lambda g: g.free_memory_mb, reverse=True)
            return valid_gpus[0].index

    def assign_frames(
        self,
        frame_paths: List[Path],
    ) -> Dict[int, List[Path]]:
        """Distribute frames across GPUs for parallel processing.

        Args:
            frame_paths: List of frame file paths

        Returns:
            Dictionary mapping GPU ID to list of frame paths
        """
        if not self.gpu_ids:
            return {0: frame_paths}

        if not self.is_available():
            return {self.gpu_ids[0]: frame_paths}

        # Distribute frames evenly across GPUs
        assignments: Dict[int, List[Path]] = {gpu: [] for gpu in self.gpu_ids}

        for i, frame in enumerate(frame_paths):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            assignments[gpu_id].append(frame)

        # Log distribution
        for gpu_id, frames in assignments.items():
            logger.debug(f"GPU {gpu_id}: {len(frames)} frames assigned")

        return assignments

    def get_gpu_status(self) -> List[Dict]:
        """Get current status of all managed GPUs.

        Returns:
            List of status dictionaries for each GPU
        """
        statuses = []
        all_gpus = get_all_gpu_info()

        for gpu in all_gpus:
            if gpu.index in self.gpu_ids:
                statuses.append({
                    "id": gpu.index,
                    "name": gpu.name,
                    "memory_total_mb": gpu.total_memory_mb,
                    "memory_used_mb": gpu.used_memory_mb,
                    "memory_free_mb": gpu.free_memory_mb,
                    "utilization_percent": gpu.utilization_percent,
                    "temperature_c": gpu.temperature_celsius,
                })

        return statuses

    def wait_for_all_ready(
        self,
        min_free_mb: int = 1000,
        timeout_seconds: float = 60.0,
    ) -> bool:
        """Wait for all GPUs to have sufficient free memory.

        Args:
            min_free_mb: Minimum free memory per GPU
            timeout_seconds: Maximum time to wait

        Returns:
            True if all GPUs ready, False on timeout
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            all_ready = True

            for gpu_id in self.gpu_ids:
                info = get_gpu_memory_info(gpu_id)
                if info is None or info["free_mb"] < min_free_mb:
                    all_ready = False
                    break

            if all_ready:
                return True

            time.sleep(1.0)

        return False


def distribute_frames_to_gpus(
    frame_paths: List[Path],
    gpu_ids: Optional[List[int]] = None,
) -> Dict[int, List[Path]]:
    """Convenience function to distribute frames across available GPUs.

    Args:
        frame_paths: List of frame file paths
        gpu_ids: Optional list of specific GPU IDs to use

    Returns:
        Dictionary mapping GPU ID to list of frames
    """
    manager = MultiGPUManager(gpu_ids=gpu_ids)
    return manager.assign_frames(frame_paths)
