"""Hardware compatibility checking for FrameWright pipeline.

Provides system analysis to determine if hardware meets requirements
for video restoration processing.
"""
import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils.gpu import (
    get_all_gpu_info,
    get_all_gpus_multivendor,
    get_best_gpu,
    get_gpu_memory_info,
    is_nvidia_gpu_available,
    GPUVendor,
)
from .utils.disk import get_disk_usage
from .utils.dependencies import (
    validate_all_dependencies,
    check_ffmpeg,
    check_realesrgan,
    check_rife,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SystemInfo:
    """System information summary."""
    os_name: str
    os_version: str
    python_version: str
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0


@dataclass
class GPUCapability:
    """GPU capability assessment."""
    has_gpu: bool = False
    gpu_name: str = "None detected"
    gpu_vendor: str = "unknown"  # nvidia, amd, intel, unknown
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    cuda_available: bool = False
    vulkan_available: bool = False
    ncnn_vulkan_available: bool = False  # For AMD/Intel GPU acceleration
    recommended_tile_size: int = 512
    max_resolution: str = "1080p"
    can_process_4k: bool = False
    driver_version: str = ""
    all_gpus: List[Dict[str, Any]] = field(default_factory=list)  # Info about all detected GPUs


@dataclass
class HardwareReport:
    """Complete hardware compatibility report."""
    system: SystemInfo
    gpu: GPUCapability
    disk_free_gb: float = 0.0
    dependencies_ok: bool = False
    missing_dependencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_status: str = "unknown"  # ready, limited, incompatible

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system": {
                "os": f"{self.system.os_name} {self.system.os_version}",
                "python": self.system.python_version,
                "cpu": self.system.cpu_name,
                "cores": self.system.cpu_cores,
                "ram_total_gb": round(self.system.ram_total_gb, 1),
                "ram_available_gb": round(self.system.ram_available_gb, 1),
            },
            "gpu": {
                "available": self.gpu.has_gpu,
                "name": self.gpu.gpu_name,
                "vram_total_mb": self.gpu.vram_total_mb,
                "vram_free_mb": self.gpu.vram_free_mb,
                "max_resolution": self.gpu.max_resolution,
                "can_process_4k": self.gpu.can_process_4k,
            },
            "disk_free_gb": round(self.disk_free_gb, 1),
            "dependencies_ok": self.dependencies_ok,
            "missing_dependencies": self.missing_dependencies,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "overall_status": self.overall_status,
        }


# =============================================================================
# System Information
# =============================================================================

def get_system_info() -> SystemInfo:
    """Gather system information."""
    info = SystemInfo(
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
    )

    # CPU info
    try:
        info.cpu_cores = __import__("os").cpu_count() or 0

        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info.cpu_name = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info.cpu_name = result.stdout.strip()
        elif platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            info.cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except Exception:
        pass

    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.ram_total_gb = mem.total / (1024 ** 3)
        info.ram_available_gb = mem.available / (1024 ** 3)
    except ImportError:
        # Fallback for Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[1])
                            info.ram_total_gb = kb / (1024 ** 2)
                        elif "MemAvailable" in line:
                            kb = int(line.split()[1])
                            info.ram_available_gb = kb / (1024 ** 2)
            except Exception:
                pass

    return info


def _check_ncnn_vulkan_available() -> bool:
    """Check if realesrgan-ncnn-vulkan binary is available."""
    # Check common locations
    ncnn_names = [
        "realesrgan-ncnn-vulkan",
        "realesrgan-ncnn-vulkan.exe",
    ]

    for name in ncnn_names:
        if shutil.which(name):
            return True

    # Check in ~/.framewright/bin
    home = Path.home()
    bin_paths = [
        home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan.exe",
        home / ".framewright" / "bin" / "realesrgan-ncnn-vulkan",
        Path.cwd() / "bin" / "realesrgan-ncnn-vulkan.exe",
    ]

    for path in bin_paths:
        if path.exists():
            return True

    return False


def get_gpu_capability() -> GPUCapability:
    """Assess GPU capabilities for video processing.

    Detects NVIDIA, AMD, and Intel GPUs and determines available
    acceleration options (CUDA, Vulkan, ncnn-vulkan).
    """
    cap = GPUCapability()

    # Check for ncnn-vulkan binary (works with AMD/Intel/NVIDIA via Vulkan)
    cap.ncnn_vulkan_available = _check_ncnn_vulkan_available()

    # Get all GPUs using multi-vendor detection
    all_gpus = get_all_gpus_multivendor()

    if all_gpus:
        # Store info about all detected GPUs
        cap.all_gpus = [
            {
                "name": g.name,
                "vendor": g.vendor.value,
                "vram_mb": g.total_memory_mb,
                "cuda": g.cuda_supported,
                "vulkan": g.vulkan_supported,
                "dedicated": g.is_dedicated,
            }
            for g in all_gpus
        ]

        # Use the best GPU (prioritizes dedicated, then NVIDIA > AMD > Intel)
        best = get_best_gpu()
        if best:
            cap.has_gpu = True
            cap.gpu_name = best.name
            cap.gpu_vendor = best.vendor.value
            cap.vram_total_mb = best.total_memory_mb
            cap.vram_free_mb = best.free_memory_mb
            cap.driver_version = best.driver_version
            cap.cuda_available = best.cuda_supported
            cap.vulkan_available = best.vulkan_supported

            # Determine capabilities based on VRAM
            if cap.vram_total_mb >= 8000:
                cap.max_resolution = "4K (3840x2160)"
                cap.can_process_4k = True
                cap.recommended_tile_size = 0  # No tiling needed
            elif cap.vram_total_mb >= 6000:
                cap.max_resolution = "4K with tiling"
                cap.can_process_4k = True
                cap.recommended_tile_size = 512
            elif cap.vram_total_mb >= 4000:
                cap.max_resolution = "1440p (2560x1440)"
                cap.can_process_4k = False
                cap.recommended_tile_size = 384
            elif cap.vram_total_mb >= 2000:
                cap.max_resolution = "1080p (1920x1080)"
                cap.can_process_4k = False
                cap.recommended_tile_size = 256
            else:
                cap.max_resolution = "720p (1280x720)"
                cap.can_process_4k = False
                cap.recommended_tile_size = 128

            return cap

    # Fallback: No GPUs detected via WMI, try legacy nvidia-smi check
    if is_nvidia_gpu_available():
        cap.has_gpu = True
        cap.cuda_available = True
        cap.vulkan_available = True
        cap.gpu_vendor = "nvidia"

        gpu_info = get_all_gpu_info()
        if gpu_info:
            gpu = gpu_info[0]
            cap.gpu_name = gpu.name
            cap.vram_total_mb = gpu.total_memory_mb
            cap.vram_free_mb = gpu.free_memory_mb

    return cap


# =============================================================================
# Hardware Check
# =============================================================================

def check_hardware(
    project_dir: Optional[Path] = None,
    video_resolution: Tuple[int, int] = (1920, 1080),
    scale_factor: int = 4,
) -> HardwareReport:
    """Perform comprehensive hardware compatibility check.

    Args:
        project_dir: Directory to check disk space (default: current dir)
        video_resolution: Target video resolution (width, height)
        scale_factor: Upscaling factor (2 or 4)

    Returns:
        HardwareReport with compatibility assessment
    """
    project_dir = project_dir or Path.cwd()

    # Gather information
    system = get_system_info()
    gpu = get_gpu_capability()

    # Check disk space
    disk = get_disk_usage(project_dir)
    disk_free_gb = disk.free_gb

    # Check dependencies
    dep_report = validate_all_dependencies()
    dependencies_ok = dep_report.is_ready()
    missing = dep_report.missing_required

    # Build report
    report = HardwareReport(
        system=system,
        gpu=gpu,
        disk_free_gb=disk_free_gb,
        dependencies_ok=dependencies_ok,
        missing_dependencies=missing,
    )

    # Generate warnings and recommendations
    _analyze_compatibility(report, video_resolution, scale_factor)

    return report


def _analyze_compatibility(
    report: HardwareReport,
    video_resolution: Tuple[int, int],
    scale_factor: int,
) -> None:
    """Analyze compatibility and generate recommendations."""
    warnings = []
    recommendations = []

    width, height = video_resolution
    output_width = width * scale_factor
    output_height = height * scale_factor

    # GPU checks
    if not report.gpu.has_gpu:
        warnings.append("No GPU detected - processing will be CPU-only (very slow)")
        recommendations.append("Install a GPU with 4GB+ VRAM for 10-50x faster processing")
    elif report.gpu.gpu_vendor in ["amd", "intel"] and not report.gpu.ncnn_vulkan_available:
        recommendations.append(
            f"Install realesrgan-ncnn-vulkan for {report.gpu.gpu_vendor.upper()} GPU acceleration: "
            "framewright install-ncnn-vulkan"
        )
    elif report.gpu.vram_total_mb < 2000:
        warnings.append(f"Low VRAM ({report.gpu.vram_total_mb}MB) - may cause out-of-memory errors")
        recommendations.append("Use smaller tile sizes (--tile 128) or process at lower resolution")

    # Resolution vs VRAM
    if output_width * output_height > 3840 * 2160 and not report.gpu.can_process_4k:
        warnings.append(f"Output resolution ({output_width}x{output_height}) exceeds GPU capability")
        recommendations.append(f"Reduce scale factor to 2x or use tile processing")

    # RAM checks
    if report.system.ram_total_gb < 8:
        warnings.append(f"Low RAM ({report.system.ram_total_gb:.1f}GB) - may limit parallel processing")
        recommendations.append("Close other applications while processing")
    elif report.system.ram_total_gb < 16:
        recommendations.append("Consider limiting parallel frames (--parallel-frames 2)")

    # Disk space checks
    # Estimate: 4K frame ~= 25MB uncompressed, need original + enhanced + temp
    estimated_gb_per_minute = (30 * 25 * 3) / 1024  # 30fps, 25MB/frame, 3 copies
    if report.disk_free_gb < 50:
        warnings.append(f"Low disk space ({report.disk_free_gb:.1f}GB free)")
        recommendations.append("Free up disk space - processing 1 minute of 4K video needs ~2GB")
    elif report.disk_free_gb < 100:
        recommendations.append("Monitor disk space during processing")

    # Dependency checks
    if not report.dependencies_ok:
        for dep in report.missing_dependencies:
            if dep == "ffmpeg":
                recommendations.append("Install FFmpeg: apt install ffmpeg (Linux) / brew install ffmpeg (Mac)")
            elif dep == "realesrgan":
                recommendations.append("Download Real-ESRGAN from: https://github.com/xinntao/Real-ESRGAN/releases")

    # CPU checks
    if report.system.cpu_cores < 4:
        warnings.append(f"Limited CPU cores ({report.system.cpu_cores}) - parallel processing limited")

    # Determine overall status
    if not report.dependencies_ok:
        report.overall_status = "incompatible"
    elif not report.gpu.has_gpu:
        report.overall_status = "incompatible"  # Changed: no GPU = incompatible (require_gpu=True by default)
        warnings.append(
            "CRITICAL: No GPU detected. Processing would use CPU only, "
            "which can freeze your system. GPU is required by default."
        )
    elif report.gpu.vram_total_mb < 1024:
        report.overall_status = "incompatible"  # Less than 1GB VRAM is not viable
        warnings.append(f"GPU VRAM ({report.gpu.vram_total_mb}MB) below minimum (1024MB)")
    elif report.gpu.vram_total_mb < 2000:
        report.overall_status = "limited"
    elif report.gpu.gpu_vendor in ["amd", "intel"] and not report.gpu.ncnn_vulkan_available:
        # AMD/Intel GPUs need ncnn-vulkan for acceleration
        report.overall_status = "limited"
        warnings.append("ncnn-vulkan required for GPU acceleration but not installed")
    elif len(warnings) > 2:
        report.overall_status = "limited"
    else:
        report.overall_status = "ready"

    report.warnings = warnings
    report.recommendations = recommendations


def print_hardware_report(report: HardwareReport) -> str:
    """Format hardware report for console display.

    Args:
        report: Hardware report to format

    Returns:
        Formatted string for display
    """
    status_icons = {
        "ready": "[OK]",
        "limited": "[!!]",
        "incompatible": "[X]",
        "unknown": "[?]",
    }

    lines = [
        "",
        "=" * 60,
        "  FrameWright Hardware Compatibility Report",
        "=" * 60,
        "",
        "SYSTEM INFORMATION",
        "-" * 40,
        f"  OS:        {report.system.os_name} {report.system.os_version}",
        f"  Python:    {report.system.python_version}",
        f"  CPU:       {report.system.cpu_name}",
        f"  Cores:     {report.system.cpu_cores}",
        f"  RAM:       {report.system.ram_total_gb:.1f} GB total, {report.system.ram_available_gb:.1f} GB available",
        "",
        "GPU INFORMATION",
        "-" * 40,
    ]

    if report.gpu.has_gpu:
        vendor_icons = {
            "nvidia": "NVIDIA",
            "amd": "AMD",
            "intel": "Intel",
            "unknown": "Unknown",
        }
        vendor_display = vendor_icons.get(report.gpu.gpu_vendor, report.gpu.gpu_vendor)

        lines.extend([
            f"  GPU:       {report.gpu.gpu_name}",
            f"  Vendor:    {vendor_display}",
            f"  VRAM:      {report.gpu.vram_total_mb} MB total, {report.gpu.vram_free_mb} MB free",
            f"  CUDA:      {'Yes' if report.gpu.cuda_available else 'No'}",
            f"  Vulkan:    {'Yes' if report.gpu.vulkan_available else 'No'}",
            f"  ncnn-vulkan: {'Yes' if report.gpu.ncnn_vulkan_available else 'No (install for AMD/Intel acceleration)'}",
            f"  Max Res:   {report.gpu.max_resolution}",
        ])

        # Show all detected GPUs if more than one
        if len(report.gpu.all_gpus) > 1:
            lines.append("")
            lines.append("  All GPUs detected:")
            for i, g in enumerate(report.gpu.all_gpus):
                gpu_type = "dedicated" if g.get("dedicated") else "integrated"
                lines.append(f"    [{i}] {g['name']} ({g['vendor']}, {g['vram_mb']}MB, {gpu_type})")
    else:
        lines.append("  No compatible GPU detected")

    lines.extend([
        "",
        "STORAGE",
        "-" * 40,
        f"  Free:      {report.disk_free_gb:.1f} GB",
        "",
        "DEPENDENCIES",
        "-" * 40,
    ])

    if report.dependencies_ok:
        lines.append("  [OK] All required dependencies installed")
    else:
        lines.append("  [X] Missing dependencies:")
        for dep in report.missing_dependencies:
            lines.append(f"      - {dep}")

    if report.warnings:
        lines.extend([
            "",
            "WARNINGS",
            "-" * 40,
        ])
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    if report.recommendations:
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for rec in report.recommendations:
            lines.append(f"  - {rec}")

    lines.extend([
        "",
        "-" * 60,
        f"  Overall Status: {status_icons.get(report.overall_status, '[?]')} {report.overall_status.upper()}",
        "-" * 60,
        "",
    ])

    return "\n".join(lines)


def quick_check() -> bool:
    """Quick check if system can run FrameWright.

    Returns:
        True if system meets minimum requirements
    """
    report = check_hardware()
    return report.overall_status in ("ready", "limited")
