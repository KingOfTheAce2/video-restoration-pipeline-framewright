"""Dependency validation and management for FrameWright.

Provides comprehensive version validation and fallback strategies for external tools.
"""
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    command: str
    installed: bool = False
    version: Optional[str] = None
    path: Optional[str] = None
    meets_minimum: bool = True
    minimum_version: Optional[str] = None
    gpu_available: bool = False
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyReport:
    """Complete dependency validation report."""
    dependencies: Dict[str, DependencyInfo] = field(default_factory=dict)
    all_required_met: bool = True
    missing_required: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_ready(self) -> bool:
        """Check if all required dependencies are met."""
        return self.all_required_met and not self.missing_required

    def summary(self) -> str:
        """Generate summary string."""
        lines = ["Dependency Status:"]
        for name, info in self.dependencies.items():
            status = "OK" if info.installed and info.meets_minimum else "MISSING"
            version = f"v{info.version}" if info.version else "unknown"
            lines.append(f"  {name}: {status} ({version})")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


def compare_versions(version1: str, version2: str) -> int:
    """Compare two version strings.

    Args:
        version1: First version
        version2: Second version

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    def normalize(v: str) -> List[int]:
        # Remove common prefixes and extract numeric parts
        v = re.sub(r'^[vV]', '', v)
        v = re.sub(r'[-_].*$', '', v)  # Remove suffixes like -beta
        parts = re.findall(r'\d+', v)
        return [int(p) for p in parts] if parts else [0]

    v1_parts = normalize(version1)
    v2_parts = normalize(version2)

    # Pad to equal length
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))

    for p1, p2 in zip(v1_parts, v2_parts):
        if p1 < p2:
            return -1
        if p1 > p2:
            return 1

    return 0


def _find_ffmpeg_binary(name: str) -> Optional[str]:
    """Find ffmpeg/ffprobe binary in PATH or common locations."""
    import platform

    # Check PATH first
    path = shutil.which(name)
    if path:
        return path

    # Check common installation directories
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    search_paths = [
        # Common Windows locations
        Path("F:/ffmpeg-8.0.1-essentials_build/bin") / f"{name}{exe_suffix}",
        Path("C:/ffmpeg/bin") / f"{name}{exe_suffix}",
        Path.home() / "ffmpeg" / "bin" / f"{name}{exe_suffix}",
        Path.home() / ".framewright" / "bin" / f"{name}{exe_suffix}",
        # Common Unix locations
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
    ]

    for search_path in search_paths:
        if search_path.exists():
            logger.info(f"Found {name} at: {search_path}")
            return str(search_path)

    return None


# Cache paths for performance
_ffmpeg_path_cache: Optional[str] = None
_ffprobe_path_cache: Optional[str] = None


def get_ffmpeg_path() -> str:
    """Get ffmpeg binary path, raising error if not found.

    Returns:
        Path to ffmpeg executable

    Raises:
        FileNotFoundError: If ffmpeg is not found
    """
    global _ffmpeg_path_cache
    if _ffmpeg_path_cache is None:
        _ffmpeg_path_cache = _find_ffmpeg_binary("ffmpeg")
    if _ffmpeg_path_cache is None:
        raise FileNotFoundError(
            "ffmpeg not found in PATH or common locations. "
            "Please install FFmpeg or add it to PATH."
        )
    return _ffmpeg_path_cache


def get_ffprobe_path() -> str:
    """Get ffprobe binary path, raising error if not found.

    Returns:
        Path to ffprobe executable

    Raises:
        FileNotFoundError: If ffprobe is not found
    """
    global _ffprobe_path_cache
    if _ffprobe_path_cache is None:
        _ffprobe_path_cache = _find_ffmpeg_binary("ffprobe")
    if _ffprobe_path_cache is None:
        raise FileNotFoundError(
            "ffprobe not found in PATH or common locations. "
            "Please install FFmpeg or add it to PATH."
        )
    return _ffprobe_path_cache


def check_ffmpeg() -> DependencyInfo:
    """Check FFmpeg installation and version.

    Searches for ffmpeg in PATH and common installation directories.

    Returns:
        DependencyInfo for FFmpeg
    """
    info = DependencyInfo(
        name="FFmpeg",
        command="ffmpeg",
        minimum_version="4.0",
    )

    path = _find_ffmpeg_binary("ffmpeg")
    if not path:
        info.error_message = "ffmpeg not found in PATH or common locations"
        return info

    info.path = path
    info.command = path
    info.installed = True

    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse version from output
        match = re.search(r'ffmpeg version (\S+)', result.stdout)
        if match:
            info.version = match.group(1)
            info.meets_minimum = compare_versions(info.version, info.minimum_version) >= 0

        # Check for GPU encoders
        if "nvenc" in result.stdout.lower():
            info.gpu_available = True
            info.additional_info["nvidia_encoder"] = True

        if "vaapi" in result.stdout.lower():
            info.additional_info["vaapi"] = True

    except subprocess.TimeoutExpired:
        info.error_message = "ffmpeg version check timed out"
    except Exception as e:
        info.error_message = str(e)

    return info


def check_ffprobe() -> DependencyInfo:
    """Check ffprobe installation.

    Searches for ffprobe in PATH and common installation directories.

    Returns:
        DependencyInfo for ffprobe
    """
    info = DependencyInfo(
        name="ffprobe",
        command="ffprobe",
    )

    path = _find_ffmpeg_binary("ffprobe")
    if not path:
        info.error_message = "ffprobe not found in PATH or common locations"
        return info

    info.path = path
    info.command = path
    info.installed = True

    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        match = re.search(r'ffprobe version (\S+)', result.stdout)
        if match:
            info.version = match.group(1)

    except Exception as e:
        info.error_message = str(e)

    return info


def check_realesrgan() -> DependencyInfo:
    """Check Real-ESRGAN installation.

    Searches for:
    1. realesrgan-ncnn-vulkan binary in PATH or custom locations
    2. PyTorch realesrgan Python package (fallback)

    Returns:
        DependencyInfo for Real-ESRGAN
    """
    import platform

    info = DependencyInfo(
        name="Real-ESRGAN",
        command="realesrgan-ncnn-vulkan",
    )

    # Determine executable name based on platform
    exe_suffix = ".exe" if platform.system() == "Windows" else ""

    # Try different command names in PATH
    commands = [
        "realesrgan-ncnn-vulkan",
        "realesrgan",
        "realsr-ncnn-vulkan",
    ]

    for cmd in commands:
        path = shutil.which(cmd)
        if path:
            info.path = path
            info.command = cmd
            info.installed = True
            info.additional_info["backend"] = "ncnn"
            break

    # If not in PATH, check common installation directories
    if not info.installed:
        search_paths = [
            Path.home() / ".framewright" / "bin" / f"realesrgan-ncnn-vulkan{exe_suffix}",
            Path.cwd() / "bin" / f"realesrgan-ncnn-vulkan{exe_suffix}",
        ]

        for search_path in search_paths:
            if search_path.exists():
                info.path = str(search_path)
                info.command = str(search_path)
                info.installed = True
                info.additional_info["install_location"] = "framewright"
                info.additional_info["backend"] = "ncnn"
                logger.info(f"Found Real-ESRGAN at: {search_path}")
                break

    # Check for PyTorch version (fallback backend)
    # Must verify all required imports work, not just the package exists
    if not info.installed:
        try:
            import torch
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            info.installed = True
            info.command = "python -m realesrgan"
            info.path = "realesrgan (Python package)"
            info.additional_info["backend"] = "pytorch"
            # Try to get version
            try:
                import realesrgan
                info.version = getattr(realesrgan, "__version__", "unknown")
            except Exception:
                info.version = "installed"
            # Check if CUDA is available
            info.gpu_available = torch.cuda.is_available()
            if info.gpu_available:
                try:
                    info.additional_info["cuda_device"] = torch.cuda.get_device_name(0)
                except Exception:
                    pass
            logger.info("Found Real-ESRGAN PyTorch package (verified RealESRGANer and RRDBNet imports)")
            return info
        except ImportError as e:
            logger.warning(f"Real-ESRGAN PyTorch package check failed: {e}")
        except Exception as e:
            logger.debug(f"Real-ESRGAN PyTorch check error: {e}")

    if not info.installed:
        info.error_message = (
            "Real-ESRGAN not found. Install with: pip install realesrgan basicsr "
            "OR download ncnn-vulkan binary from https://github.com/xinntao/Real-ESRGAN/releases"
        )
        return info

    # For ncnn binary, try to get version info
    try:
        result = subprocess.run(
            [info.command, "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout + result.stderr

        # Check for GPU/Vulkan support
        if "vulkan" in output.lower() or "gpu" in output.lower():
            info.gpu_available = True

        # Try to extract version
        match = re.search(r'version[:\s]+(\S+)', output, re.IGNORECASE)
        if match:
            info.version = match.group(1)

        # Check available models
        models = []
        for model in ["realesrgan-x4plus", "realesrgan-x2plus", "realesr-animevideov3"]:
            if model in output:
                models.append(model)
        info.additional_info["models"] = models

    except Exception as e:
        info.error_message = str(e)

    return info


def _find_ytdlp_binary() -> Optional[str]:
    """Find yt-dlp binary in PATH or common locations."""
    import platform
    import os

    # Check PATH first
    path = shutil.which("yt-dlp")
    if path:
        return path

    # Check common installation directories
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    home = Path.home()

    search_paths = [
        # Python user Scripts (pip install --user)
        home / "AppData" / "Roaming" / "Python" / "Python313" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Roaming" / "Python" / "Python311" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "Scripts" / f"yt-dlp{exe_suffix}",
        # System Python
        Path("C:/Python313/Scripts") / f"yt-dlp{exe_suffix}",
        Path("C:/Python312/Scripts") / f"yt-dlp{exe_suffix}",
        # Unix locations
        home / ".local" / "bin" / "yt-dlp",
        Path("/usr/local/bin/yt-dlp"),
    ]

    for search_path in search_paths:
        if search_path.exists():
            logger.info(f"Found yt-dlp at: {search_path}")
            return str(search_path)

    return None


def check_ytdlp() -> DependencyInfo:
    """Check yt-dlp installation.

    Searches for yt-dlp in PATH and common installation directories.

    Returns:
        DependencyInfo for yt-dlp
    """
    info = DependencyInfo(
        name="yt-dlp",
        command="yt-dlp",
        minimum_version="2023.0.0",
    )

    path = _find_ytdlp_binary()
    if not path:
        info.error_message = "yt-dlp not found. Install with: pip install yt-dlp"
        return info

    info.path = path
    info.command = path
    info.installed = True

    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        info.version = result.stdout.strip()

        if info.version:
            info.meets_minimum = compare_versions(info.version, info.minimum_version) >= 0

    except Exception as e:
        info.error_message = str(e)

    return info


def check_rife() -> DependencyInfo:
    """Check RIFE installation (optional).

    Returns:
        DependencyInfo for RIFE
    """
    info = DependencyInfo(
        name="RIFE",
        command="rife-ncnn-vulkan",
    )

    commands = [
        "rife-ncnn-vulkan",
        "rife",
    ]

    for cmd in commands:
        path = shutil.which(cmd)
        if path:
            info.path = path
            info.command = cmd
            info.installed = True
            break

    if not info.installed:
        # RIFE is optional
        info.error_message = "RIFE not found (optional for frame interpolation)"
        return info

    try:
        result = subprocess.run(
            [info.command, "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout + result.stderr

        if "vulkan" in output.lower():
            info.gpu_available = True

        # Check for model versions
        models = []
        for model in ["rife-v2.3", "rife-v4.0", "rife-v4.6"]:
            if model in output:
                models.append(model)
        info.additional_info["models"] = models

    except Exception as e:
        info.error_message = str(e)

    return info


def check_python_package(package_name: str) -> DependencyInfo:
    """Check if a Python package is installed.

    Args:
        package_name: Name of the package

    Returns:
        DependencyInfo for the package
    """
    info = DependencyInfo(
        name=package_name,
        command=f"python -c \"import {package_name}\"",
    )

    try:
        import importlib.metadata

        try:
            version = importlib.metadata.version(package_name)
            info.installed = True
            info.version = version
        except importlib.metadata.PackageNotFoundError:
            info.error_message = f"Package {package_name} not installed"

    except Exception as e:
        info.error_message = str(e)

    return info


def validate_all_dependencies(
    required: Optional[List[str]] = None,
    optional: Optional[List[str]] = None,
) -> DependencyReport:
    """Validate all dependencies.

    Args:
        required: List of required dependency names
        optional: List of optional dependency names

    Returns:
        DependencyReport with validation results
    """
    if required is None:
        required = ["ffmpeg", "ffprobe", "realesrgan", "yt-dlp"]

    if optional is None:
        optional = ["rife"]

    report = DependencyReport()

    # Check required dependencies
    checkers = {
        "ffmpeg": check_ffmpeg,
        "ffprobe": check_ffprobe,
        "realesrgan": check_realesrgan,
        "yt-dlp": check_ytdlp,
        "rife": check_rife,
    }

    for dep_name in required:
        checker = checkers.get(dep_name.lower())
        if checker:
            info = checker()
            report.dependencies[dep_name] = info

            if not info.installed:
                report.all_required_met = False
                report.missing_required.append(dep_name)
            elif not info.meets_minimum:
                report.warnings.append(
                    f"{dep_name} version {info.version} is below minimum {info.minimum_version}"
                )

    for dep_name in optional:
        checker = checkers.get(dep_name.lower())
        if checker:
            info = checker()
            report.dependencies[dep_name] = info

            if not info.installed:
                report.warnings.append(f"Optional dependency {dep_name} not available")

    return report


class DependencyFallback:
    """Manage fallback strategies for dependencies."""

    def __init__(self):
        self._fallbacks: Dict[str, List[Callable[[], Optional[str]]]] = {}
        self._selected: Dict[str, str] = {}

    def register_fallback(
        self,
        dependency: str,
        check_func: Callable[[], Optional[str]],
    ) -> None:
        """Register a fallback option for a dependency.

        Args:
            dependency: Dependency name
            check_func: Function that returns command path if available
        """
        if dependency not in self._fallbacks:
            self._fallbacks[dependency] = []
        self._fallbacks[dependency].append(check_func)

    def get_available(self, dependency: str) -> Optional[str]:
        """Get the first available command for a dependency.

        Args:
            dependency: Dependency name

        Returns:
            Command path or None
        """
        if dependency in self._selected:
            return self._selected[dependency]

        for check_func in self._fallbacks.get(dependency, []):
            try:
                result = check_func()
                if result:
                    self._selected[dependency] = result
                    logger.info(f"Using {result} for {dependency}")
                    return result
            except Exception:
                continue

        return None


def get_enhancement_backend() -> Tuple[str, str]:
    """Get the best available enhancement backend.

    Returns:
        Tuple of (backend_type, command)

    Raises:
        RuntimeError: If no backend available
    """
    # Check ncnn-vulkan version (fastest)
    ncnn_path = shutil.which("realesrgan-ncnn-vulkan")
    if ncnn_path:
        return ("ncnn", ncnn_path)

    # Check for Python version
    try:
        import importlib.util
        if importlib.util.find_spec("realesrgan") is not None:
            return ("python", "python -m realesrgan.inference_realesrgan")
    except Exception:
        pass

    raise RuntimeError(
        "No Real-ESRGAN backend available. Install realesrgan-ncnn-vulkan or "
        "pip install realesrgan"
    )


def install_recommendations() -> str:
    """Generate installation recommendations based on current state.

    Returns:
        Markdown-formatted installation instructions
    """
    report = validate_all_dependencies()

    lines = ["# Installation Recommendations\n"]

    if report.is_ready():
        lines.append("All required dependencies are installed and ready!\n")
    else:
        lines.append("The following dependencies need attention:\n")

    for dep_name, info in report.dependencies.items():
        if not info.installed:
            lines.append(f"\n## {info.name}\n")
            lines.append(f"**Status:** Not installed\n")

            if dep_name == "ffmpeg":
                lines.append("Install with:\n")
                lines.append("```bash\n")
                lines.append("# Ubuntu/Debian\n")
                lines.append("sudo apt install ffmpeg\n")
                lines.append("\n# macOS\n")
                lines.append("brew install ffmpeg\n")
                lines.append("\n# Windows (chocolatey)\n")
                lines.append("choco install ffmpeg\n")
                lines.append("```\n")

            elif dep_name == "realesrgan":
                lines.append("Download pre-built binary from:\n")
                lines.append("https://github.com/xinntao/Real-ESRGAN/releases\n")
                lines.append("\nOr install Python version:\n")
                lines.append("```bash\n")
                lines.append("pip install realesrgan\n")
                lines.append("```\n")

            elif dep_name == "yt-dlp":
                lines.append("Install with:\n")
                lines.append("```bash\n")
                lines.append("pip install yt-dlp\n")
                lines.append("```\n")

    return "\n".join(lines)
