"""Model Manager CLI for FrameWright.

Pre-download and manage AI models before processing. Provides visibility
into model storage and allows bulk downloading.

Features:
- List all available models with sizes
- Pre-download specific or all models
- Clean unused/old models
- Show disk usage by model type
- Verify model integrity

Example:
    >>> manager = ModelManagerCLI()
    >>> manager.list_models()
    >>> manager.download_models(["realesrgan", "hat"])
    >>> manager.show_disk_usage()
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = Path.home() / ".framewright" / "models"


class ModelCategory(Enum):
    """Categories of AI models."""
    UPSCALING = "upscaling"
    FACE = "face"
    COLORIZATION = "colorization"
    INTERPOLATION = "interpolation"
    DENOISING = "denoising"
    INPAINTING = "inpainting"
    AUDIO = "audio"
    OTHER = "other"


@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    category: ModelCategory
    filename: str
    url: str
    size_mb: float
    description: str
    required: bool = False  # Required for basic functionality
    hash_md5: Optional[str] = None

    # Status
    is_downloaded: bool = False
    local_path: Optional[Path] = None
    is_valid: bool = False


@dataclass
class ModelRegistry:
    """Registry of all available models."""
    models: Dict[str, ModelInfo] = field(default_factory=dict)

    def get_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Get models by category."""
        return [m for m in self.models.values() if m.category == category]

    def get_required(self) -> List[ModelInfo]:
        """Get required models."""
        return [m for m in self.models.values() if m.required]


# Model registry with all available models
MODEL_REGISTRY = ModelRegistry(models={
    # Upscaling models
    "realesrgan-x4plus": ModelInfo(
        name="realesrgan-x4plus",
        category=ModelCategory.UPSCALING,
        filename="realesrgan-x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size_mb=64.0,
        description="Real-ESRGAN 4x upscaler for general content",
        required=True,
    ),
    "realesrgan-x4plus-anime": ModelInfo(
        name="realesrgan-x4plus-anime",
        category=ModelCategory.UPSCALING,
        filename="realesrgan-x4plus-anime.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        size_mb=17.0,
        description="Real-ESRGAN 4x upscaler optimized for anime",
    ),
    "realesrgan-x2plus": ModelInfo(
        name="realesrgan-x2plus",
        category=ModelCategory.UPSCALING,
        filename="realesrgan-x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        size_mb=64.0,
        description="Real-ESRGAN 2x upscaler",
    ),
    "hat-l": ModelInfo(
        name="hat-l",
        category=ModelCategory.UPSCALING,
        filename="HAT-L_SRx4_ImageNet-pretrain.pth",
        url="https://github.com/XPixelGroup/HAT/releases/download/v1.0.0/HAT-L_SRx4_ImageNet-pretrain.pth",
        size_mb=250.0,
        description="HAT Large - Hybrid Attention Transformer (highest quality)",
    ),
    "swinir": ModelInfo(
        name="swinir",
        category=ModelCategory.UPSCALING,
        filename="003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        url="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        size_mb=136.0,
        description="SwinIR - Swin Transformer for image restoration",
    ),

    # Face restoration models
    "gfpgan-v1.4": ModelInfo(
        name="gfpgan-v1.4",
        category=ModelCategory.FACE,
        filename="GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        size_mb=332.0,
        description="GFPGAN v1.4 face restoration",
        required=True,
    ),
    "codeformer": ModelInfo(
        name="codeformer",
        category=ModelCategory.FACE,
        filename="codeformer.pth",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        size_mb=376.0,
        description="CodeFormer face restoration (better quality)",
    ),
    "restoreformer": ModelInfo(
        name="restoreformer",
        category=ModelCategory.FACE,
        filename="RestoreFormer.pth",
        url="https://github.com/wzhouxiff/RestoreFormer/releases/download/v1.0/RestoreFormer.pth",
        size_mb=280.0,
        description="RestoreFormer face restoration",
    ),

    # Colorization models
    "ddcolor": ModelInfo(
        name="ddcolor",
        category=ModelCategory.COLORIZATION,
        filename="ddcolor_modelscope.pth",
        url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
        size_mb=520.0,
        description="DDColor automatic colorization",
    ),
    "deoldify-artistic": ModelInfo(
        name="deoldify-artistic",
        category=ModelCategory.COLORIZATION,
        filename="ColorizeArtistic_gen.pth",
        url="https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        size_mb=255.0,
        description="DeOldify artistic colorization",
    ),
    "deoldify-stable": ModelInfo(
        name="deoldify-stable",
        category=ModelCategory.COLORIZATION,
        filename="ColorizeVideo_gen.pth",
        url="https://data.deepai.org/deoldify/ColorizeVideo_gen.pth",
        size_mb=255.0,
        description="DeOldify stable video colorization",
    ),

    # Interpolation models
    "rife-v4.6": ModelInfo(
        name="rife-v4.6",
        category=ModelCategory.INTERPOLATION,
        filename="rife-v4.6.pth",
        url="https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/rife-v4.6.zip",
        size_mb=28.0,
        description="RIFE v4.6 frame interpolation",
        required=True,
    ),
    "rife-v4.15": ModelInfo(
        name="rife-v4.15",
        category=ModelCategory.INTERPOLATION,
        filename="rife-v4.15.pth",
        url="https://github.com/hzwer/Practical-RIFE/releases/download/v4.15/rife-v4.15.zip",
        size_mb=30.0,
        description="RIFE v4.15 frame interpolation (latest)",
    ),

    # Denoising models
    "nafnet": ModelInfo(
        name="nafnet",
        category=ModelCategory.DENOISING,
        filename="NAFNet-GoPro-width64.pth",
        url="https://github.com/megvii-research/NAFNet/releases/download/v0.0.1/NAFNet-GoPro-width64.pth",
        size_mb=67.0,
        description="NAFNet denoising/deblurring",
    ),
    "restormer": ModelInfo(
        name="restormer",
        category=ModelCategory.DENOISING,
        filename="restormer_deraining.pth",
        url="https://github.com/swz30/Restormer/releases/download/v1.0/restormer_deraining.pth",
        size_mb=99.0,
        description="Restormer for image restoration",
    ),

    # Inpainting models
    "lama": ModelInfo(
        name="lama",
        category=ModelCategory.INPAINTING,
        filename="big-lama.pt",
        url="https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
        size_mb=200.0,
        description="LaMa large mask inpainting (watermark removal)",
    ),
})


class ModelManagerCLI:
    """Command-line interface for model management."""

    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize model manager.

        Args:
            model_dir: Directory for model storage
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.registry = MODEL_REGISTRY
        self._update_model_status()

    def _update_model_status(self) -> None:
        """Update downloaded status of all models."""
        for model in self.registry.models.values():
            local_path = self._get_model_path(model)
            model.local_path = local_path
            model.is_downloaded = local_path.exists()
            if model.is_downloaded:
                model.is_valid = self._verify_model(model)

    def _get_model_path(self, model: ModelInfo) -> Path:
        """Get local path for a model."""
        category_dir = self.model_dir / model.category.value
        return category_dir / model.filename

    def _verify_model(self, model: ModelInfo) -> bool:
        """Verify model file integrity.

        Args:
            model: Model to verify

        Returns:
            True if valid
        """
        if not model.local_path or not model.local_path.exists():
            return False

        # Check file size (within 10% of expected)
        actual_size = model.local_path.stat().st_size / (1024 * 1024)
        if model.size_mb > 0:
            if abs(actual_size - model.size_mb) / model.size_mb > 0.1:
                return False

        # Check MD5 if available
        if model.hash_md5:
            try:
                hasher = hashlib.md5()
                with open(model.local_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hasher.update(chunk)
                if hasher.hexdigest() != model.hash_md5:
                    return False
            except Exception:
                pass

        return True

    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        show_downloaded_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List available models.

        Args:
            category: Filter by category
            show_downloaded_only: Only show downloaded models

        Returns:
            List of model info dicts
        """
        self._update_model_status()

        models = []
        for model in self.registry.models.values():
            if category and model.category != category:
                continue
            if show_downloaded_only and not model.is_downloaded:
                continue

            models.append({
                "name": model.name,
                "category": model.category.value,
                "size_mb": model.size_mb,
                "description": model.description,
                "required": model.required,
                "downloaded": model.is_downloaded,
                "valid": model.is_valid,
                "path": str(model.local_path) if model.local_path else None,
            })

        return models

    def download_model(
        self,
        name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force: bool = False,
    ) -> bool:
        """Download a specific model.

        Args:
            name: Model name
            progress_callback: Called with (bytes_downloaded, total_bytes)
            force: Force re-download even if exists

        Returns:
            True if successful
        """
        if name not in self.registry.models:
            logger.error(f"Unknown model: {name}")
            return False

        model = self.registry.models[name]

        if model.is_downloaded and model.is_valid and not force:
            logger.info(f"Model already downloaded: {name}")
            return True

        # Create category directory
        local_path = self._get_model_path(model)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import urllib.request

            logger.info(f"Downloading {name} ({model.size_mb:.0f}MB)...")

            # Download with progress
            def report_progress(block_num, block_size, total_size):
                if progress_callback:
                    progress_callback(block_num * block_size, total_size)

            temp_path = local_path.with_suffix('.tmp')
            urllib.request.urlretrieve(
                model.url,
                temp_path,
                reporthook=report_progress
            )

            # Handle zip files
            if model.url.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(temp_path, 'r') as zf:
                    zf.extractall(local_path.parent)
                temp_path.unlink()
            else:
                temp_path.rename(local_path)

            model.is_downloaded = True
            model.is_valid = self._verify_model(model)
            model.local_path = local_path

            logger.info(f"Downloaded: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            return False

    def download_models(
        self,
        names: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, bool]:
        """Download multiple models.

        Args:
            names: List of model names
            progress_callback: Called with (model_name, bytes, total)

        Returns:
            Dict mapping model names to success status
        """
        results = {}

        for name in names:
            def model_progress(downloaded, total):
                if progress_callback:
                    progress_callback(name, downloaded, total)

            results[name] = self.download_model(name, model_progress)

        return results

    def download_all(
        self,
        include_optional: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, bool]:
        """Download all models.

        Args:
            include_optional: Include non-required models
            progress_callback: Progress callback

        Returns:
            Dict mapping model names to success status
        """
        if include_optional:
            names = list(self.registry.models.keys())
        else:
            names = [m.name for m in self.registry.get_required()]

        return self.download_models(names, progress_callback)

    def download_required(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, bool]:
        """Download only required models.

        Args:
            progress_callback: Progress callback

        Returns:
            Dict mapping model names to success status
        """
        return self.download_all(include_optional=False, progress_callback=progress_callback)

    def clean_unused(self, keep_days: int = 30) -> Tuple[int, float]:
        """Remove models not used recently.

        Args:
            keep_days: Keep models used within this many days

        Returns:
            Tuple of (files_removed, mb_freed)
        """
        import os

        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        files_removed = 0
        bytes_freed = 0

        for model in self.registry.models.values():
            if not model.local_path or not model.local_path.exists():
                continue

            # Check last access time
            stat = model.local_path.stat()
            if stat.st_atime < cutoff_time:
                size = stat.st_size
                model.local_path.unlink()
                files_removed += 1
                bytes_freed += size

                model.is_downloaded = False
                model.is_valid = False

                logger.info(f"Removed unused model: {model.name}")

        return files_removed, bytes_freed / (1024 * 1024)

    def clean_invalid(self) -> Tuple[int, float]:
        """Remove invalid/corrupted models.

        Returns:
            Tuple of (files_removed, mb_freed)
        """
        files_removed = 0
        bytes_freed = 0

        self._update_model_status()

        for model in self.registry.models.values():
            if model.is_downloaded and not model.is_valid:
                if model.local_path and model.local_path.exists():
                    size = model.local_path.stat().st_size
                    model.local_path.unlink()
                    files_removed += 1
                    bytes_freed += size

                    model.is_downloaded = False
                    logger.info(f"Removed invalid model: {model.name}")

        return files_removed, bytes_freed / (1024 * 1024)

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage by model category.

        Returns:
            Dict with usage statistics
        """
        self._update_model_status()

        usage = {
            "total_mb": 0.0,
            "by_category": {},
            "models": [],
        }

        for category in ModelCategory:
            usage["by_category"][category.value] = 0.0

        for model in self.registry.models.values():
            if model.is_downloaded and model.local_path and model.local_path.exists():
                size_mb = model.local_path.stat().st_size / (1024 * 1024)
                usage["total_mb"] += size_mb
                usage["by_category"][model.category.value] += size_mb
                usage["models"].append({
                    "name": model.name,
                    "size_mb": size_mb,
                    "category": model.category.value,
                })

        # Sort models by size
        usage["models"].sort(key=lambda x: x["size_mb"], reverse=True)

        return usage

    def get_missing_required(self) -> List[ModelInfo]:
        """Get list of required models that are not downloaded.

        Returns:
            List of missing required models
        """
        self._update_model_status()
        return [
            m for m in self.registry.get_required()
            if not m.is_downloaded or not m.is_valid
        ]

    def verify_all(self) -> Dict[str, bool]:
        """Verify integrity of all downloaded models.

        Returns:
            Dict mapping model names to validity status
        """
        self._update_model_status()
        return {
            m.name: m.is_valid
            for m in self.registry.models.values()
            if m.is_downloaded
        }


def get_model_manager(model_dir: Optional[Path] = None) -> ModelManagerCLI:
    """Get model manager instance.

    Args:
        model_dir: Optional model directory

    Returns:
        ModelManagerCLI instance
    """
    return ModelManagerCLI(model_dir)
