"""System diagnostics and health monitoring."""

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemDiagnostics:
    """Complete system diagnostics report."""
    timestamp: datetime
    overall_status: HealthStatus

    # System info
    os_info: str = ""
    python_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""

    # Hardware
    cpu_info: str = ""
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpu_name: str = ""
    gpu_driver: str = ""
    vram_total_gb: float = 0.0
    vram_available_gb: float = 0.0
    gpu_temperature: int = 0

    # Disk
    disk_total_gb: float = 0.0
    disk_available_gb: float = 0.0

    # Dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)
    missing_dependencies: List[str] = field(default_factory=list)

    # FFmpeg
    ffmpeg_version: str = ""
    ffmpeg_codecs: List[str] = field(default_factory=list)

    # Health checks
    health_checks: List[HealthCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "system": {
                "os": self.os_info,
                "python": self.python_version,
                "cuda_available": self.cuda_available,
                "cuda_version": self.cuda_version,
            },
            "cpu": {
                "info": self.cpu_info,
                "cores": self.cpu_cores,
            },
            "memory": {
                "total_gb": self.ram_total_gb,
                "available_gb": self.ram_available_gb,
            },
            "gpu": {
                "name": self.gpu_name,
                "driver": self.gpu_driver,
                "vram_total_gb": self.vram_total_gb,
                "vram_available_gb": self.vram_available_gb,
                "temperature_c": self.gpu_temperature,
            },
            "disk": {
                "total_gb": self.disk_total_gb,
                "available_gb": self.disk_available_gb,
            },
            "ffmpeg": {
                "version": self.ffmpeg_version,
                "codecs": self.ffmpeg_codecs,
            },
            "dependencies": self.dependencies,
            "missing_dependencies": self.missing_dependencies,
            "health_checks": [
                {
                    "name": h.name,
                    "status": h.status.value,
                    "message": h.message,
                }
                for h in self.health_checks
            ],
        }


class DiagnosticsAnalyzer:
    """Analyzes system health and capabilities."""

    REQUIRED_DEPENDENCIES = [
        "torch",
        "numpy",
        "cv2",
        "PIL",
    ]

    OPTIONAL_DEPENDENCIES = [
        "torchvision",
        "basicsr",
        "facexlib",
        "realesrgan",
        "scipy",
        "skimage",
        "flask",
    ]

    def __init__(self):
        self._last_diagnostics: Optional[SystemDiagnostics] = None

    def run_diagnostics(self) -> SystemDiagnostics:
        """Run complete system diagnostics."""
        diag = SystemDiagnostics(
            timestamp=datetime.now(),
            overall_status=HealthStatus.HEALTHY,
        )

        # System info
        self._collect_system_info(diag)
        self._collect_cpu_info(diag)
        self._collect_memory_info(diag)
        self._collect_gpu_info(diag)
        self._collect_disk_info(diag)
        self._collect_ffmpeg_info(diag)
        self._collect_dependencies(diag)

        # Run health checks
        self._run_health_checks(diag)

        # Determine overall status
        self._determine_overall_status(diag)

        self._last_diagnostics = diag
        return diag

    def _collect_system_info(self, diag: SystemDiagnostics) -> None:
        """Collect system information."""
        diag.os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        diag.python_version = sys.version.split()[0]

        try:
            import torch
            diag.cuda_available = torch.cuda.is_available()
            if diag.cuda_available:
                diag.cuda_version = torch.version.cuda or ""
        except ImportError:
            pass

    def _collect_cpu_info(self, diag: SystemDiagnostics) -> None:
        """Collect CPU information."""
        diag.cpu_cores = os.cpu_count() or 0

        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            diag.cpu_info = info.get("brand_raw", platform.processor())
        except ImportError:
            diag.cpu_info = platform.processor()

    def _collect_memory_info(self, diag: SystemDiagnostics) -> None:
        """Collect memory information."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            diag.ram_total_gb = mem.total / (1024**3)
            diag.ram_available_gb = mem.available / (1024**3)
        except ImportError:
            pass

    def _collect_gpu_info(self, diag: SystemDiagnostics) -> None:
        """Collect GPU information."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                diag.gpu_name = props.name
                diag.vram_total_gb = props.total_memory / (1024**3)

                # Get available VRAM
                free, total = torch.cuda.mem_get_info(0)
                diag.vram_available_gb = free / (1024**3)
        except Exception:
            pass

        # Try nvidia-smi for more info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    diag.gpu_driver = parts[0].strip()
                    diag.gpu_temperature = int(parts[1].strip())
        except Exception:
            pass

    def _collect_disk_info(self, diag: SystemDiagnostics) -> None:
        """Collect disk information."""
        try:
            import psutil
            disk = psutil.disk_usage("/")
            diag.disk_total_gb = disk.total / (1024**3)
            diag.disk_available_gb = disk.free / (1024**3)
        except Exception:
            pass

    def _collect_ffmpeg_info(self, diag: SystemDiagnostics) -> None:
        """Collect FFmpeg information."""
        try:
            # Version
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                first_line = result.stdout.split("\n")[0]
                diag.ffmpeg_version = first_line

            # Codecs
            result = subprocess.run(
                ["ffmpeg", "-codecs"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                important_codecs = ["h264", "hevc", "h265", "av1", "vp9", "libx264", "libx265"]
                for codec in important_codecs:
                    if codec in result.stdout.lower():
                        diag.ffmpeg_codecs.append(codec)

        except Exception as e:
            logger.warning(f"FFmpeg check failed: {e}")

    def _collect_dependencies(self, diag: SystemDiagnostics) -> None:
        """Collect dependency information."""
        for dep in self.REQUIRED_DEPENDENCIES + self.OPTIONAL_DEPENDENCIES:
            try:
                module = __import__(dep)
                version = getattr(module, "__version__", "installed")
                diag.dependencies[dep] = version
            except ImportError:
                if dep in self.REQUIRED_DEPENDENCIES:
                    diag.missing_dependencies.append(dep)

    def _run_health_checks(self, diag: SystemDiagnostics) -> None:
        """Run health checks."""
        checks = [
            self._check_cuda,
            self._check_memory,
            self._check_disk,
            self._check_ffmpeg,
            self._check_dependencies,
            self._check_gpu_temperature,
        ]

        for check in checks:
            try:
                result = check(diag)
                diag.health_checks.append(result)
            except Exception as e:
                logger.warning(f"Health check failed: {e}")

    def _check_cuda(self, diag: SystemDiagnostics) -> HealthCheck:
        """Check CUDA availability."""
        if diag.cuda_available:
            return HealthCheck(
                name="CUDA",
                status=HealthStatus.HEALTHY,
                message=f"CUDA {diag.cuda_version} available with {diag.gpu_name}",
            )
        else:
            return HealthCheck(
                name="CUDA",
                status=HealthStatus.WARNING,
                message="CUDA not available, will use CPU (slower)",
            )

    def _check_memory(self, diag: SystemDiagnostics) -> HealthCheck:
        """Check available memory."""
        if diag.ram_available_gb <= 0:
            return HealthCheck(
                name="Memory",
                status=HealthStatus.UNKNOWN,
                message="Unable to detect memory",
            )

        if diag.ram_available_gb < 4:
            return HealthCheck(
                name="Memory",
                status=HealthStatus.CRITICAL,
                message=f"Low RAM: {diag.ram_available_gb:.1f}GB available",
            )
        elif diag.ram_available_gb < 8:
            return HealthCheck(
                name="Memory",
                status=HealthStatus.WARNING,
                message=f"Limited RAM: {diag.ram_available_gb:.1f}GB available",
            )
        else:
            return HealthCheck(
                name="Memory",
                status=HealthStatus.HEALTHY,
                message=f"{diag.ram_available_gb:.1f}GB RAM available",
            )

    def _check_disk(self, diag: SystemDiagnostics) -> HealthCheck:
        """Check available disk space."""
        if diag.disk_available_gb <= 0:
            return HealthCheck(
                name="Disk",
                status=HealthStatus.UNKNOWN,
                message="Unable to detect disk space",
            )

        if diag.disk_available_gb < 10:
            return HealthCheck(
                name="Disk",
                status=HealthStatus.CRITICAL,
                message=f"Low disk space: {diag.disk_available_gb:.1f}GB available",
            )
        elif diag.disk_available_gb < 50:
            return HealthCheck(
                name="Disk",
                status=HealthStatus.WARNING,
                message=f"Limited disk space: {diag.disk_available_gb:.1f}GB available",
            )
        else:
            return HealthCheck(
                name="Disk",
                status=HealthStatus.HEALTHY,
                message=f"{diag.disk_available_gb:.1f}GB disk space available",
            )

    def _check_ffmpeg(self, diag: SystemDiagnostics) -> HealthCheck:
        """Check FFmpeg installation."""
        if diag.ffmpeg_version:
            if "libx264" in diag.ffmpeg_codecs and "libx265" in diag.ffmpeg_codecs:
                return HealthCheck(
                    name="FFmpeg",
                    status=HealthStatus.HEALTHY,
                    message=f"FFmpeg installed with H.264 and H.265 support",
                )
            else:
                return HealthCheck(
                    name="FFmpeg",
                    status=HealthStatus.WARNING,
                    message="FFmpeg installed but may lack codec support",
                )
        else:
            return HealthCheck(
                name="FFmpeg",
                status=HealthStatus.CRITICAL,
                message="FFmpeg not found - video processing unavailable",
            )

    def _check_dependencies(self, diag: SystemDiagnostics) -> HealthCheck:
        """Check required dependencies."""
        if diag.missing_dependencies:
            return HealthCheck(
                name="Dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Missing: {', '.join(diag.missing_dependencies)}",
            )
        else:
            return HealthCheck(
                name="Dependencies",
                status=HealthStatus.HEALTHY,
                message=f"All {len(diag.dependencies)} dependencies installed",
            )

    def _check_gpu_temperature(self, diag: SystemDiagnostics) -> HealthCheck:
        """Check GPU temperature."""
        if diag.gpu_temperature <= 0:
            return HealthCheck(
                name="GPU Temperature",
                status=HealthStatus.UNKNOWN,
                message="Unable to read GPU temperature",
            )

        if diag.gpu_temperature > 85:
            return HealthCheck(
                name="GPU Temperature",
                status=HealthStatus.CRITICAL,
                message=f"GPU overheating: {diag.gpu_temperature}C",
            )
        elif diag.gpu_temperature > 75:
            return HealthCheck(
                name="GPU Temperature",
                status=HealthStatus.WARNING,
                message=f"GPU warm: {diag.gpu_temperature}C",
            )
        else:
            return HealthCheck(
                name="GPU Temperature",
                status=HealthStatus.HEALTHY,
                message=f"GPU temperature: {diag.gpu_temperature}C",
            )

    def _determine_overall_status(self, diag: SystemDiagnostics) -> None:
        """Determine overall system status."""
        statuses = [h.status for h in diag.health_checks]

        if HealthStatus.CRITICAL in statuses:
            diag.overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            diag.overall_status = HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            diag.overall_status = HealthStatus.HEALTHY
        else:
            diag.overall_status = HealthStatus.UNKNOWN

    def print_report(self, diag: Optional[SystemDiagnostics] = None) -> str:
        """Generate human-readable diagnostic report."""
        diag = diag or self._last_diagnostics
        if not diag:
            diag = self.run_diagnostics()

        lines = [
            "=" * 60,
            "FRAMEWRIGHT SYSTEM DIAGNOSTICS",
            "=" * 60,
            f"Timestamp: {diag.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {diag.overall_status.value.upper()}",
            "",
            "SYSTEM",
            "-" * 40,
            f"OS: {diag.os_info}",
            f"Python: {diag.python_version}",
            f"CUDA: {'Available' if diag.cuda_available else 'Not available'}",
            "",
            "HARDWARE",
            "-" * 40,
            f"CPU: {diag.cpu_info} ({diag.cpu_cores} cores)",
            f"RAM: {diag.ram_available_gb:.1f} / {diag.ram_total_gb:.1f} GB available",
            f"GPU: {diag.gpu_name or 'Not detected'}",
            f"VRAM: {diag.vram_available_gb:.1f} / {diag.vram_total_gb:.1f} GB available",
            f"Disk: {diag.disk_available_gb:.1f} / {diag.disk_total_gb:.1f} GB available",
            "",
            "HEALTH CHECKS",
            "-" * 40,
        ]

        for check in diag.health_checks:
            icon = {"healthy": "[OK]", "warning": "[!]", "critical": "[X]", "unknown": "[?]"}
            lines.append(f"{icon.get(check.status.value, '[ ]')} {check.name}: {check.message}")

        if diag.missing_dependencies:
            lines.extend([
                "",
                "MISSING DEPENDENCIES",
                "-" * 40,
            ])
            for dep in diag.missing_dependencies:
                lines.append(f"  - {dep}")

        lines.append("=" * 60)

        return "\n".join(lines)


def run_diagnostics() -> SystemDiagnostics:
    """Convenience function to run diagnostics."""
    analyzer = DiagnosticsAnalyzer()
    return analyzer.run_diagnostics()


def print_diagnostics() -> None:
    """Print diagnostic report to console."""
    analyzer = DiagnosticsAnalyzer()
    diag = analyzer.run_diagnostics()
    print(analyzer.print_report(diag))
