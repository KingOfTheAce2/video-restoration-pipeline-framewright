"""NCNN-Vulkan backend for Real-ESRGAN frame enhancement.

Provides GPU-accelerated frame enhancement using the ncnn-vulkan backend,
which supports AMD, Intel, and NVIDIA GPUs via the Vulkan API.

This is the recommended backend for non-NVIDIA GPUs.

IMPORTANT: This module includes CPU fallback detection to prevent
runaway CPU usage when GPU processing fails silently.
"""
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# CPU fallback indicators in ncnn-vulkan output
CPU_FALLBACK_INDICATORS = [
    "using cpu",
    "no vulkan device",
    "vulkan not found",
    "failed to create gpu instance",
    "cpu mode",
    "fallback to cpu",
]

# Download URLs for realesrgan-ncnn-vulkan
NCNN_VULKAN_RELEASES = {
    "windows": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
        "exe": "realesrgan-ncnn-vulkan.exe",
    },
    "linux": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        "exe": "realesrgan-ncnn-vulkan",
    },
    "darwin": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip",
        "exe": "realesrgan-ncnn-vulkan",
    },
}

# Supported models in ncnn-vulkan
NCNN_MODELS = [
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "realesr-animevideov3",
    "realesrnet-x4plus",
]


@dataclass
class NcnnVulkanConfig:
    """Configuration for ncnn-vulkan processing."""
    model_name: str = "realesrgan-x4plus"
    scale_factor: int = 4
    tile_size: int = 0  # 0 = auto
    gpu_id: int = -1  # -1 = auto
    tta_mode: bool = False  # Test-time augmentation
    output_format: str = "png"
    require_gpu: bool = True  # Fail if GPU not available (prevents CPU fallback)

    def validate(self) -> None:
        """Validate configuration."""
        if self.model_name not in NCNN_MODELS:
            raise ValueError(
                f"Invalid model: {self.model_name}. "
                f"Supported models: {', '.join(NCNN_MODELS)}"
            )
        if self.scale_factor not in [2, 3, 4]:
            raise ValueError(f"Scale factor must be 2, 3, or 4, got {self.scale_factor}")


class NcnnVulkanBackend:
    """Backend for Real-ESRGAN using ncnn-vulkan.

    This backend uses the Vulkan API for GPU acceleration,
    supporting AMD, Intel, and NVIDIA GPUs.
    """

    def __init__(
        self,
        binary_path: Optional[Path] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize the ncnn-vulkan backend.

        Args:
            binary_path: Path to realesrgan-ncnn-vulkan executable.
                        If None, will search common locations.
            model_dir: Path to model files. If None, uses bundled models.
        """
        self.binary_path = binary_path or self._find_binary()
        self.model_dir = model_dir
        self._available_gpus: Optional[List[dict]] = None

    @staticmethod
    def get_install_dir() -> Path:
        """Get the default installation directory."""
        return Path.home() / ".framewright" / "bin"

    @staticmethod
    def get_default_binary_path() -> Path:
        """Get the default path to the ncnn-vulkan binary."""
        install_dir = NcnnVulkanBackend.get_install_dir()
        if platform.system() == "Windows":
            return install_dir / "realesrgan-ncnn-vulkan.exe"
        return install_dir / "realesrgan-ncnn-vulkan"

    def _find_binary(self) -> Optional[Path]:
        """Find the realesrgan-ncnn-vulkan binary."""
        # Check PATH first
        exe_name = "realesrgan-ncnn-vulkan"
        if platform.system() == "Windows":
            exe_name += ".exe"

        path_binary = shutil.which(exe_name)
        if path_binary:
            return Path(path_binary)

        # Check common installation locations
        search_paths = [
            self.get_default_binary_path(),
            Path.home() / ".framewright" / "bin" / exe_name,
            Path.cwd() / "bin" / exe_name,
        ]

        # Add Windows-specific paths
        if platform.system() == "Windows":
            search_paths.extend([
                Path(os.environ.get("LOCALAPPDATA", "")) / "framewright" / "bin" / exe_name,
                Path(os.environ.get("PROGRAMFILES", "")) / "framewright" / "bin" / exe_name,
            ])

        for path in search_paths:
            if path.exists():
                logger.info(f"Found ncnn-vulkan binary at: {path}")
                return path

        return None

    def is_available(self) -> bool:
        """Check if the ncnn-vulkan backend is available."""
        return self.binary_path is not None and self.binary_path.exists()

    def get_version(self) -> Optional[str]:
        """Get the version of the ncnn-vulkan binary."""
        if not self.is_available():
            return None

        try:
            result = subprocess.run(
                [str(self.binary_path), "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Version is typically in the help output
            return "ncnn-vulkan (version from help)"
        except Exception:
            return None

    def list_gpus(self) -> List[dict]:
        """List available Vulkan GPUs.

        Returns:
            List of GPU info dictionaries with 'id' and 'name' keys.
        """
        if self._available_gpus is not None:
            return self._available_gpus

        if not self.is_available():
            return []

        # Try to get GPU list from vulkaninfo
        gpus = self._detect_vulkan_gpus()
        self._available_gpus = gpus
        return gpus

    def _detect_vulkan_gpus(self) -> List[dict]:
        """Detect Vulkan-capable GPUs using vulkaninfo.

        Returns:
            List of GPU dictionaries with 'id', 'name', and 'type' keys.
        """
        gpus = []

        # Try vulkaninfo first
        vulkaninfo = shutil.which("vulkaninfo")
        if vulkaninfo:
            try:
                result = subprocess.run(
                    [vulkaninfo, "--summary"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
                )

                if result.returncode == 0:
                    # Parse GPU info from vulkaninfo output
                    lines = result.stdout.split('\n')
                    gpu_id = 0
                    for line in lines:
                        # Look for device name lines
                        if 'deviceName' in line or 'GPU' in line.upper():
                            # Extract GPU name
                            match = re.search(r'=\s*(.+)$', line)
                            if match:
                                gpu_name = match.group(1).strip()
                                gpus.append({
                                    'id': gpu_id,
                                    'name': gpu_name,
                                    'type': 'discrete' if any(x in gpu_name.lower() for x in ['nvidia', 'amd', 'radeon', 'geforce', 'rtx', 'rx']) else 'integrated',
                                })
                                gpu_id += 1

                    if gpus:
                        logger.info(f"Detected {len(gpus)} Vulkan GPU(s): {[g['name'] for g in gpus]}")
                        return gpus

            except Exception as e:
                logger.debug(f"vulkaninfo failed: {e}")

        # Fallback: Try to detect via WMI on Windows
        if platform.system() == "Windows":
            try:
                from ..utils.gpu import get_all_gpus_multivendor
                all_gpus = get_all_gpus_multivendor()
                for i, gpu in enumerate(all_gpus):
                    if gpu.vulkan_supported:
                        gpus.append({
                            'id': i,
                            'name': gpu.name,
                            'type': 'discrete' if gpu.is_dedicated else 'integrated',
                        })
            except Exception as e:
                logger.debug(f"WMI GPU detection failed: {e}")

        return gpus

    def verify_gpu_available(self) -> Tuple[bool, str]:
        """Verify that GPU processing is actually available.

        Performs a quick test to ensure ncnn-vulkan can use the GPU
        and won't fall back to CPU.

        Returns:
            Tuple of (is_available, message)
        """
        gpus = self.list_gpus()

        if not gpus:
            return False, "No Vulkan-capable GPUs detected"

        # Check for discrete GPU (preferred)
        discrete_gpus = [g for g in gpus if g.get('type') == 'discrete']

        if discrete_gpus:
            return True, f"Discrete GPU available: {discrete_gpus[0]['name']}"
        elif gpus:
            return True, f"Integrated GPU available: {gpus[0]['name']}"

        return False, "No GPUs detected"

    def _check_cpu_fallback(self, stderr: str) -> bool:
        """Check if ncnn-vulkan output indicates CPU fallback.

        Args:
            stderr: The stderr output from ncnn-vulkan

        Returns:
            True if CPU fallback detected
        """
        if not stderr:
            return False

        stderr_lower = stderr.lower()
        for indicator in CPU_FALLBACK_INDICATORS:
            if indicator in stderr_lower:
                logger.warning(f"CPU fallback detected: '{indicator}' in output")
                return True

        return False

    def enhance_frame(
        self,
        input_path: Path,
        output_path: Path,
        config: NcnnVulkanConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Enhance a single frame using ncnn-vulkan.

        Args:
            input_path: Path to input image (PNG/JPG)
            output_path: Path for output image
            config: Processing configuration

        Returns:
            Tuple of (success, error_message)

        Note:
            If config.require_gpu=True, this will fail if CPU fallback is detected.
            This prevents runaway CPU usage that can freeze the system.
        """
        if not self.is_available():
            return False, "ncnn-vulkan binary not found"

        config.validate()

        # Pre-check GPU availability if required
        if config.require_gpu:
            gpu_ok, gpu_msg = self.verify_gpu_available()
            if not gpu_ok:
                return False, f"GPU required but not available: {gpu_msg}"

        # Build command
        cmd = [
            str(self.binary_path),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", config.model_name,
            "-s", str(config.scale_factor),
            "-f", config.output_format,
        ]

        if config.tile_size > 0:
            cmd.extend(["-t", str(config.tile_size)])

        # GPU selection: use explicit GPU ID if set, otherwise use first detected GPU
        if config.gpu_id >= 0:
            cmd.extend(["-g", str(config.gpu_id)])
        elif config.require_gpu:
            # When GPU is required, explicitly select GPU 0 to prevent CPU fallback
            gpus = self.list_gpus()
            if gpus:
                cmd.extend(["-g", str(gpus[0]['id'])])
                logger.debug(f"Explicitly selecting GPU {gpus[0]['id']}: {gpus[0]['name']}")

        if config.tta_mode:
            cmd.append("-x")

        if self.model_dir:
            cmd.extend(["-m", str(self.model_dir)])

        try:
            logger.debug(f"Running ncnn-vulkan: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )

            # Check for CPU fallback in output
            combined_output = f"{result.stdout or ''} {result.stderr or ''}"
            if config.require_gpu and self._check_cpu_fallback(combined_output):
                # Delete output file if created (we don't trust CPU-processed results)
                if output_path.exists():
                    output_path.unlink()
                return False, (
                    "CPU fallback detected! Processing used CPU instead of GPU. "
                    "This would consume excessive CPU resources. "
                    "Check GPU drivers, Vulkan installation, or set require_gpu=False."
                )

            if not output_path.exists():
                return False, "Output file was not created"

            return True, None

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)

            # Check for CPU fallback in error output
            if config.require_gpu and self._check_cpu_fallback(error_msg):
                return False, (
                    "CPU fallback detected in error output! "
                    "GPU processing failed and would fall back to CPU. "
                    "Check GPU drivers or Vulkan installation."
                )

            logger.error(f"ncnn-vulkan failed: {error_msg}")
            return False, error_msg
        except subprocess.TimeoutExpired:
            return False, "Processing timed out (>5 minutes) - possible CPU fallback causing slow processing"
        except Exception as e:
            return False, str(e)

    def enhance_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        config: NcnnVulkanConfig,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[int, int, List[str]]:
        """Enhance all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output images
            config: Processing configuration
            progress_callback: Optional callback(current, total)

        Returns:
            Tuple of (success_count, fail_count, error_messages)
        """
        if not self.is_available():
            return 0, 0, ["ncnn-vulkan binary not found"]

        config.validate()
        output_dir.mkdir(parents=True, exist_ok=True)

        # ncnn-vulkan can process directories directly
        cmd = [
            str(self.binary_path),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-n", config.model_name,
            "-s", str(config.scale_factor),
            "-f", config.output_format,
        ]

        if config.tile_size > 0:
            cmd.extend(["-t", str(config.tile_size)])

        if config.gpu_id >= 0:
            cmd.extend(["-g", str(config.gpu_id)])

        if config.tta_mode:
            cmd.append("-x")

        if self.model_dir:
            cmd.extend(["-m", str(self.model_dir)])

        try:
            logger.info(f"Processing directory with ncnn-vulkan: {input_dir}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for directory
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )

            # Count results
            input_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
            output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))

            success_count = len(output_files)
            fail_count = len(input_files) - success_count

            return success_count, fail_count, []

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)
            logger.error(f"ncnn-vulkan directory processing failed: {error_msg}")
            return 0, 0, [error_msg]
        except subprocess.TimeoutExpired:
            return 0, 0, ["Directory processing timed out (>1 hour)"]
        except Exception as e:
            return 0, 0, [str(e)]


def install_ncnn_vulkan(
    install_dir: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[bool, str]:
    """Download and install realesrgan-ncnn-vulkan.

    Args:
        install_dir: Installation directory (default: ~/.framewright/bin)
        progress_callback: Optional callback(bytes_downloaded, total_bytes)

    Returns:
        Tuple of (success, message)
    """
    system = platform.system().lower()
    if system not in NCNN_VULKAN_RELEASES:
        return False, f"Unsupported platform: {system}"

    release = NCNN_VULKAN_RELEASES[system]
    install_dir = install_dir or NcnnVulkanBackend.get_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    zip_path = install_dir / "realesrgan-ncnn-vulkan.zip"

    try:
        logger.info(f"Downloading ncnn-vulkan from {release['url']}")

        # Download with progress
        def report_hook(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                progress_callback(downloaded, total_size)

        urlretrieve(release["url"], zip_path, reporthook=report_hook)

        # Extract
        logger.info(f"Extracting to {install_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)

        # Make executable on Unix
        if system != "windows":
            exe_path = install_dir / release["exe"]
            if exe_path.exists():
                exe_path.chmod(exe_path.stat().st_mode | 0o755)

        # Clean up zip
        zip_path.unlink()

        # Verify installation
        backend = NcnnVulkanBackend()
        if backend.is_available():
            return True, f"Successfully installed ncnn-vulkan to {install_dir}"
        else:
            return False, "Installation completed but binary not found"

    except Exception as e:
        logger.error(f"Failed to install ncnn-vulkan: {e}")
        # Clean up on failure
        if zip_path.exists():
            zip_path.unlink()
        return False, str(e)


def get_ncnn_vulkan_path() -> Optional[Path]:
    """Get the path to the ncnn-vulkan binary if available.

    Returns:
        Path to binary or None if not found
    """
    backend = NcnnVulkanBackend()
    return backend.binary_path if backend.is_available() else None


def is_ncnn_vulkan_available() -> bool:
    """Check if ncnn-vulkan is available for use.

    Returns:
        True if ncnn-vulkan is installed and accessible
    """
    backend = NcnnVulkanBackend()
    return backend.is_available()
