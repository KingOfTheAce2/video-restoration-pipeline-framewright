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


def check_ffmpeg() -> DependencyInfo:
    """Check FFmpeg installation and version.

    Returns:
        DependencyInfo for FFmpeg
    """
    info = DependencyInfo(
        name="FFmpeg",
        command="ffmpeg",
        minimum_version="4.0",
    )

    path = shutil.which("ffmpeg")
    if not path:
        info.error_message = "ffmpeg not found in PATH"
        return info

    info.path = path
    info.installed = True

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
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

    Returns:
        DependencyInfo for ffprobe
    """
    info = DependencyInfo(
        name="ffprobe",
        command="ffprobe",
    )

    path = shutil.which("ffprobe")
    if not path:
        info.error_message = "ffprobe not found in PATH"
        return info

    info.path = path
    info.installed = True

    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
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

    Returns:
        DependencyInfo for Real-ESRGAN
    """
    info = DependencyInfo(
        name="Real-ESRGAN",
        command="realesrgan-ncnn-vulkan",
    )

    # Try different command names
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
            break

    if not info.installed:
        info.error_message = "Real-ESRGAN binary not found"
        return info

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


def check_ytdlp() -> DependencyInfo:
    """Check yt-dlp installation.

    Returns:
        DependencyInfo for yt-dlp
    """
    info = DependencyInfo(
        name="yt-dlp",
        command="yt-dlp",
        minimum_version="2023.0.0",
    )

    path = shutil.which("yt-dlp")
    if not path:
        info.error_message = "yt-dlp not found in PATH"
        return info

    info.path = path
    info.installed = True

    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
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
