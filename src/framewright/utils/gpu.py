"""GPU and VRAM monitoring utilities for FrameWright.

Provides VRAM monitoring, adaptive tile sizing, and GPU capability detection.
"""
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


def is_nvidia_gpu_available() -> bool:
    """Check if nvidia-smi is available."""
    return shutil.which("nvidia-smi") is not None


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
