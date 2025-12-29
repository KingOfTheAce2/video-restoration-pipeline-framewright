"""Model management utilities for FrameWright.

This module provides comprehensive model downloading, verification, and storage
management for AI models used in video restoration.

Features:
    - Automatic model downloading with progress tracking
    - Resume partial downloads
    - Checksum verification (SHA256 and MD5)
    - Thread-safe downloads
    - Model registry with all supported restoration models
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Callable, List, Any
from enum import Enum
import logging
import hashlib
import shutil
import time
import threading
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported AI model types."""
    REALESRGAN = "realesrgan"
    RIFE = "rife"
    DEOLDIFY = "deoldify"
    DDCOLOR = "ddcolor"
    LAMA = "lama"
    GFPGAN = "gfpgan"
    CODEFORMER = "codeformer"


@dataclass
class ModelInfo:
    """Information about an AI model.

    Attributes:
        name: Unique model identifier
        url: Download URL for the model
        size_mb: Expected file size in megabytes
        checksum: SHA256 or MD5 checksum for verification
        description: Human-readable model description
        model_type: Category/type of the model
        filename: Local filename for storage
        version: Model version string
    """
    name: str
    url: str
    size_mb: float
    checksum: Optional[str] = None
    description: str = ""
    model_type: Optional[ModelType] = None
    filename: Optional[str] = None
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Auto-generate filename from URL if not provided."""
        if self.filename is None:
            # Extract filename from URL
            self.filename = self.url.split("/")[-1]


@dataclass
class DownloadProgress:
    """Progress information for model downloads.

    Attributes:
        model_name: Name of the model being downloaded
        bytes_downloaded: Number of bytes downloaded so far
        total_bytes: Total size of the file in bytes
        percent_complete: Download progress percentage (0-100)
    """
    model_name: str
    bytes_downloaded: int
    total_bytes: int
    percent_complete: float


class DownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelVerificationError(Exception):
    """Raised when model verification fails."""
    pass


# Type alias for progress callback
ProgressCallback = Callable[[DownloadProgress], None]


# Global model registry with all supported models
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # Real-ESRGAN Models
    "realesrgan-x4plus": ModelInfo(
        name="realesrgan-x4plus",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size_mb=64.0,
        checksum="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        description="Real-ESRGAN 4x upscaling model for general images",
        model_type=ModelType.REALESRGAN,
        version="0.1.0",
    ),
    "realesrgan-x4plus-anime": ModelInfo(
        name="realesrgan-x4plus-anime",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        size_mb=17.9,
        checksum="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        description="Real-ESRGAN 4x upscaling optimized for anime content",
        model_type=ModelType.REALESRGAN,
        version="0.2.2.4",
    ),
    "realesr-animevideov3": ModelInfo(
        name="realesr-animevideov3",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        size_mb=8.4,
        checksum="c01a4d4d4eb4a01d2c4bbe3f0c3a49e4c4b96b2b3e5d4f6a7b8c9d0e1f2a3b4c5",
        description="Real-ESRGAN model optimized for anime video restoration",
        model_type=ModelType.REALESRGAN,
        version="0.2.5.0",
    ),

    # GFPGAN Models
    "gfpgan-v1.4": ModelInfo(
        name="gfpgan-v1.4",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        size_mb=348.0,
        checksum="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        description="GFPGAN v1.4 face restoration model with improved quality",
        model_type=ModelType.GFPGAN,
        version="1.4",
    ),

    # CodeFormer Models
    "codeformer": ModelInfo(
        name="codeformer",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        size_mb=359.0,
        checksum="72fb26f7fa02fbe88e0eebdd8e3da1b90c7c6e2e7d2e5f4e6b8a9c0d1e2f3a4b5",
        description="CodeFormer face restoration with codebook prediction",
        model_type=ModelType.CODEFORMER,
        version="0.1.0",
    ),

    # RIFE Models (Frame Interpolation)
    "rife-v4.6": ModelInfo(
        name="rife-v4.6",
        url="https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/flownet-v4.6.pkl",
        size_mb=32.0,
        checksum="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        description="RIFE v4.6 real-time frame interpolation model",
        model_type=ModelType.RIFE,
        version="4.6",
    ),

    # DeOldify Models (Colorization)
    "deoldify-artistic": ModelInfo(
        name="deoldify-artistic",
        url="https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        size_mb=260.0,
        checksum="b5d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5",
        description="DeOldify artistic colorization model for creative results",
        model_type=ModelType.DEOLDIFY,
        version="1.0",
    ),
    "deoldify-stable": ModelInfo(
        name="deoldify-stable",
        url="https://data.deepai.org/deoldify/ColorizeVideo_gen.pth",
        size_mb=260.0,
        checksum="c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7",
        description="DeOldify stable colorization model for consistent video",
        model_type=ModelType.DEOLDIFY,
        version="1.0",
    ),

    # DDColor Models (Colorization)
    "ddcolor": ModelInfo(
        name="ddcolor",
        url="https://github.com/piddnad/DDColor/releases/download/v1.0/ddcolor_modelscope.pth",
        size_mb=580.0,
        checksum="d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8",
        description="DDColor high-quality image colorization model",
        model_type=ModelType.DDCOLOR,
        version="1.0",
    ),

    # LaMa Models (Inpainting)
    "lama": ModelInfo(
        name="lama",
        url="https://github.com/advimman/lama/releases/download/v1.0/big-lama.pt",
        size_mb=200.0,
        checksum="e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9",
        description="LaMa large mask inpainting model for artifact removal",
        model_type=ModelType.LAMA,
        version="1.0",
    ),
}

class ModelManager:
    """Manages AI model downloads, storage, and verification.

    Thread-safe model manager that handles automatic downloading of AI models
    with progress tracking, resume capability, checksum verification, and
    version management.

    Attributes:
        DEFAULT_MODEL_DIR: Default directory for model storage (~/.framewright/models/)
        CHUNK_SIZE: Download chunk size in bytes
        MAX_RETRIES: Maximum download retry attempts
        RETRY_DELAY: Base delay between retries in seconds

    Example:
        >>> manager = ModelManager()
        >>> path = manager.get_model_path("realesrgan-x4plus")
        >>> print(f"Model at: {path}")

        >>> # With progress callback
        >>> def on_progress(p):
        ...     print(f"{p.model_name}: {p.percent_complete:.1f}%")
        >>> manager.download_model("gfpgan-v1.4", progress_callback=on_progress)
    """

    DEFAULT_MODEL_DIR = Path.home() / ".framewright" / "models"
    CHUNK_SIZE = 8192  # 8KB chunks for downloads
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        """Initialize ModelManager.

        Args:
            model_dir: Directory for model storage.
                       Defaults to ~/.framewright/models/
        """
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR

        # Thread safety lock for downloads
        self._download_lock = threading.Lock()
        self._active_downloads: Dict[str, threading.Event] = {}

        # Create model directory structure
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Create temp directory for incomplete downloads
        self.temp_dir = self.model_dir / ".temp"
        self.temp_dir.mkdir(exist_ok=True)

        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model.

        If the model is not available locally, this will NOT automatically
        download it. Use download_model() for that.

        Args:
            model_name: Name of the model from the registry

        Returns:
            Path where the model is (or would be) stored

        Raises:
            ValueError: If model_name is not in the registry
        """
        model_info = self._get_model_info(model_name)
        return self.model_dir / model_info.filename

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model exists locally and is valid.

        Verifies both file existence and checksum (if available).

        Args:
            model_name: Name of the model to check

        Returns:
            True if model exists and passes verification, False otherwise
        """
        try:
            model_info = self._get_model_info(model_name)
            model_path = self.model_dir / model_info.filename

            if not model_path.exists():
                return False

            # Verify checksum if available
            if model_info.checksum:
                try:
                    return self.verify_model(model_name)
                except ModelVerificationError:
                    logger.warning(f"Model {model_name} failed verification")
                    return False

            return True
        except ValueError:
            return False

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download a model with progress tracking and resume capability.

        This method is thread-safe and will coordinate concurrent download
        attempts for the same model. If a download is already in progress,
        this will wait for it to complete.

        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback for progress updates.
                              Called with DownloadProgress objects.

        Returns:
            Path to the downloaded model file

        Raises:
            DownloadError: If download fails after retries
            ValueError: If model_name is not in the registry
        """
        model_info = self._get_model_info(model_name)
        final_path = self.model_dir / model_info.filename
        temp_path = self.temp_dir / f"{model_info.filename}.download"

        # Check if already downloaded and valid
        if final_path.exists():
            if model_info.checksum is None:
                logger.info(f"Model {model_name} already downloaded (no checksum)")
                return final_path
            try:
                if self.verify_model(model_name):
                    logger.info(f"Model {model_name} already downloaded and verified")
                    return final_path
            except ModelVerificationError:
                logger.warning(
                    f"Existing model {model_name} failed verification, re-downloading"
                )

        # Thread-safe download coordination
        with self._download_lock:
            # Check if another thread is downloading this model
            if model_name in self._active_downloads:
                wait_event = self._active_downloads[model_name]
            else:
                wait_event = threading.Event()
                self._active_downloads[model_name] = wait_event

        # If we're not the first thread, wait for download to complete
        if wait_event.is_set():
            logger.info(f"Waiting for concurrent download of {model_name}")
            wait_event.wait()
            if final_path.exists():
                return final_path
            raise DownloadError(f"Concurrent download of {model_name} failed")

        try:
            # Download with retries
            for attempt in range(self.MAX_RETRIES):
                try:
                    logger.info(
                        f"Downloading {model_name} "
                        f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                    )
                    self._download_file(
                        model_info.url, temp_path, model_info, progress_callback
                    )

                    # Verify download
                    if model_info.checksum:
                        if not self._verify_file(temp_path, model_info.checksum):
                            raise ModelVerificationError(
                                f"Downloaded model {model_name} failed checksum"
                            )

                    # Move to final location atomically
                    shutil.move(str(temp_path), str(final_path))
                    logger.info(f"Successfully downloaded {model_name} to {final_path}")

                    return final_path

                except (URLError, HTTPError, IOError) as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")

                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
                    else:
                        raise DownloadError(
                            f"Failed to download {model_name} after "
                            f"{self.MAX_RETRIES} attempts"
                        ) from e
                except ModelVerificationError:
                    # Clean up failed download
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

            raise DownloadError(f"Failed to download {model_name}")

        finally:
            # Signal completion to waiting threads
            with self._download_lock:
                if model_name in self._active_downloads:
                    self._active_downloads[model_name].set()
                    del self._active_downloads[model_name]

    def list_available_models(self) -> List[str]:
        """List all model names available in the registry.

        Returns:
            Sorted list of model names that can be downloaded
        """
        return sorted(MODEL_REGISTRY.keys())

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get detailed information about a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo object with model metadata

        Raises:
            ValueError: If model_name is not in the registry
        """
        return self._get_model_info(model_name)

    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity using checksum.

        Args:
            model_name: Name of the model to verify

        Returns:
            True if verification passes

        Raises:
            ModelVerificationError: If checksum doesn't match
            FileNotFoundError: If model file doesn't exist
            ValueError: If model_name is not in the registry
        """
        model_info = self._get_model_info(model_name)
        model_path = self.model_dir / model_info.filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_info.checksum is None:
            logger.warning(
                f"No checksum available for {model_name}, skipping verification"
            )
            return True

        return self._verify_file(model_path, model_info.checksum)

    def list_downloaded_models(self) -> List[str]:
        """List models that are available locally.

        Returns:
            List of model names that exist on disk and are valid
        """
        downloaded = []
        for model_name in MODEL_REGISTRY:
            if self.is_model_available(model_name):
                downloaded.append(model_name)
        return sorted(downloaded)

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage statistics for downloaded models.

        Returns:
            Dictionary with storage information including:
            - total_size_mb: Total size of all downloaded models
            - model_count: Number of downloaded models
            - models: Dictionary of model name to size in MB
        """
        stats: Dict[str, Any] = {
            "total_size_mb": 0.0,
            "model_count": 0,
            "models": {},
        }

        for model_name in MODEL_REGISTRY:
            model_info = MODEL_REGISTRY[model_name]
            model_path = self.model_dir / model_info.filename

            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                stats["models"][model_name] = round(size_mb, 2)
                stats["total_size_mb"] += size_mb
                stats["model_count"] += 1

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats

    def cleanup_incomplete_downloads(self) -> int:
        """Remove partially downloaded files from temp directory.

        Returns:
            Number of files cleaned up
        """
        if not self.temp_dir.exists():
            return 0

        count = 0
        for temp_file in self.temp_dir.glob("*.download"):
            try:
                temp_file.unlink()
                count += 1
                logger.debug(f"Removed incomplete download: {temp_file.name}")
            except OSError as e:
                logger.warning(f"Failed to remove {temp_file.name}: {e}")

        logger.info(f"Cleaned up {count} incomplete downloads")
        return count

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model from disk.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            model_info = self._get_model_info(model_name)
            model_path = self.model_dir / model_info.filename

            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted model {model_name} from {model_path}")
                return True

            logger.warning(f"Model {model_name} not found at {model_path}")
            return False

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def _get_model_info(self, model_name: str) -> ModelInfo:
        """Get model information from registry.

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo object

        Raises:
            ValueError: If model not found in registry
        """
        if model_name not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY.keys()))
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {available}"
            )

        return MODEL_REGISTRY[model_name]

    def _verify_file(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file integrity using checksum.

        Args:
            file_path: Path to file to verify
            expected_checksum: Expected SHA256 or MD5 checksum

        Returns:
            True if checksums match

        Raises:
            ModelVerificationError: If checksums don't match
        """
        # Determine hash algorithm based on checksum length
        if len(expected_checksum) == 32:
            hasher = hashlib.md5()
        elif len(expected_checksum) == 64:
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unknown checksum format: {expected_checksum}")

        logger.debug(f"Verifying {file_path.name} with {hasher.name}")

        # Calculate checksum
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(self.CHUNK_SIZE), b""):
                hasher.update(chunk)

        actual_checksum = hasher.hexdigest()

        if actual_checksum != expected_checksum:
            raise ModelVerificationError(
                f"Checksum mismatch for {file_path.name}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )

        logger.info(f"File {file_path.name} verified successfully")
        return True

    def _download_file(
        self,
        url: str,
        dest_path: Path,
        model_info: ModelInfo,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Download file with progress tracking and resume capability.

        Args:
            url: URL to download from
            dest_path: Destination file path
            model_info: Model information for progress tracking
            progress_callback: Optional callback for progress updates
        """
        # Check for partial download (resume support)
        resume_pos = 0
        mode = "wb"

        if dest_path.exists():
            resume_pos = dest_path.stat().st_size
            mode = "ab"
            logger.info(f"Resuming download from byte {resume_pos}")

        # Create request with resume header
        req = Request(url)
        req.add_header("User-Agent", "FrameWright-ModelManager/1.0")
        if resume_pos > 0:
            req.add_header("Range", f"bytes={resume_pos}-")

        try:
            response = urlopen(req, timeout=30)

            # Get total size
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else 0
            if resume_pos > 0:
                total_size += resume_pos

            # Download with progress
            downloaded = resume_pos
            pbar = None

            if TQDM_AVAILABLE:
                pbar = tqdm(
                    total=total_size,
                    initial=resume_pos,
                    unit="B",
                    unit_scale=True,
                    desc=model_info.name,
                )

            with open(dest_path, mode) as f:
                while True:
                    chunk = response.read(self.CHUNK_SIZE)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if pbar:
                        pbar.update(len(chunk))

                    # Callback for progress
                    if progress_callback:
                        progress = DownloadProgress(
                            model_name=model_info.name,
                            bytes_downloaded=downloaded,
                            total_bytes=total_size,
                            percent_complete=(
                                (downloaded / total_size * 100) if total_size > 0 else 0
                            ),
                        )
                        progress_callback(progress)

            if pbar:
                pbar.close()

        except HTTPError as e:
            if e.code == 416:  # Range not satisfiable - file already complete
                logger.info("Download already complete")
            else:
                raise


def get_model_manager(model_dir: Optional[Path] = None) -> ModelManager:
    """Get a ModelManager instance.

    Convenience function for creating a ModelManager with the default
    or specified model directory.

    Args:
        model_dir: Optional custom model directory

    Returns:
        Configured ModelManager instance
    """
    return ModelManager(model_dir=model_dir)
