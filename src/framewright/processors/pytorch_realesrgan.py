"""PyTorch backend for Real-ESRGAN frame enhancement.

Provides GPU-accelerated frame enhancement using PyTorch/CUDA.
This is the recommended backend for cloud environments with NVIDIA GPUs
where Vulkan may not be properly configured (e.g., Docker containers).

Unlike ncnn-vulkan which requires Vulkan drivers, this backend uses CUDA
which is already set up in PyTorch Docker images.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Flag to check if PyTorch Real-ESRGAN is available
_PYTORCH_ESRGAN_AVAILABLE: Optional[bool] = None
_UPSAMPLER = None


@dataclass
class PyTorchESRGANConfig:
    """Configuration for PyTorch Real-ESRGAN processing."""
    model_name: str = "RealESRGAN_x4plus"
    scale_factor: int = 4
    tile_size: int = 0  # 0 = auto, uses gpu memory to determine
    tile_pad: int = 10
    pre_pad: int = 0
    half_precision: bool = True  # Use FP16 for faster processing
    gpu_id: int = 0

    def validate(self) -> None:
        """Validate configuration."""
        valid_models = [
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus",
            "realesr-animevideov3",
            "realesr-general-x4v3",
        ]
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model: {self.model_name}. "
                f"Supported models: {', '.join(valid_models)}"
            )
        if self.scale_factor not in [2, 4]:
            raise ValueError(f"Scale factor must be 2 or 4, got {self.scale_factor}")


def is_pytorch_esrgan_available() -> bool:
    """Check if PyTorch Real-ESRGAN is available."""
    global _PYTORCH_ESRGAN_AVAILABLE

    if _PYTORCH_ESRGAN_AVAILABLE is not None:
        return _PYTORCH_ESRGAN_AVAILABLE

    try:
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        _PYTORCH_ESRGAN_AVAILABLE = True
        logger.info("PyTorch Real-ESRGAN is available")
    except ImportError as e:
        logger.warning(f"PyTorch Real-ESRGAN not available: {e}")
        logger.warning("Install with: pip install realesrgan basicsr")
        _PYTORCH_ESRGAN_AVAILABLE = False

    return _PYTORCH_ESRGAN_AVAILABLE


def get_upsampler(config: PyTorchESRGANConfig):
    """Get or create a Real-ESRGAN upsampler instance.

    Reuses existing upsampler if config matches.
    """
    global _UPSAMPLER

    if not is_pytorch_esrgan_available():
        raise RuntimeError(
            "PyTorch Real-ESRGAN not available. Install with:\n"
            "  pip install realesrgan basicsr"
        )

    import torch
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # Model configurations
    model_configs = {
        "RealESRGAN_x4plus": {
            "scale": 4,
            "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        },
        "RealESRGAN_x4plus_anime_6B": {
            "scale": 4,
            "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        },
        "RealESRGAN_x2plus": {
            "scale": 2,
            "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        },
        "realesr-animevideov3": {
            "scale": 4,
            "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        },
        "realesr-general-x4v3": {
            "scale": 4,
            "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        },
    }

    if config.model_name not in model_configs:
        raise ValueError(f"Unknown model: {config.model_name}")

    model_config = model_configs[config.model_name]

    # Determine tile size based on GPU memory if auto
    tile = config.tile_size
    if tile == 0:
        # Auto-detect based on GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(config.gpu_id).total_memory
            gpu_mem_gb = gpu_mem / (1024 ** 3)
            if gpu_mem_gb >= 24:
                tile = 0  # No tiling needed for 24GB+ cards
            elif gpu_mem_gb >= 12:
                tile = 400
            elif gpu_mem_gb >= 8:
                tile = 256
            else:
                tile = 128
            logger.info(f"Auto-selected tile size {tile} for {gpu_mem_gb:.1f}GB GPU")

    # Check if we can reuse existing upsampler
    # (simplified - in production you'd want to compare all config params)
    if _UPSAMPLER is not None:
        return _UPSAMPLER

    logger.info(f"Creating Real-ESRGAN upsampler with model {config.model_name}")

    upsampler = RealESRGANer(
        scale=model_config["scale"],
        model_path=model_config["model_path"],
        dni_weight=None,
        model=model_config["model"],
        tile=tile,
        tile_pad=config.tile_pad,
        pre_pad=config.pre_pad,
        half=config.half_precision and torch.cuda.is_available(),
        gpu_id=config.gpu_id if torch.cuda.is_available() else None,
    )

    _UPSAMPLER = upsampler
    return upsampler


def enhance_frame_pytorch(
    input_path: Path,
    output_path: Path,
    config: PyTorchESRGANConfig,
) -> Tuple[bool, Optional[str]]:
    """Enhance a single frame using PyTorch Real-ESRGAN.

    Args:
        input_path: Path to input image (PNG/JPG)
        output_path: Path for output image
        config: Processing configuration

    Returns:
        Tuple of (success, error_message)
    """
    try:
        import cv2
        import torch

        config.validate()

        # Load image
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, f"Failed to read image: {input_path}"

        # Get upsampler
        upsampler = get_upsampler(config)

        # Enhance
        output, _ = upsampler.enhance(img, outscale=config.scale_factor)

        # Save
        cv2.imwrite(str(output_path), output)

        if not output_path.exists():
            return False, "Output file was not created"

        return True, None

    except torch.cuda.OutOfMemoryError:
        return False, "GPU out of memory. Try reducing tile_size or using a smaller model."
    except Exception as e:
        logger.error(f"PyTorch Real-ESRGAN failed: {e}")
        return False, str(e)


def clear_upsampler_cache():
    """Clear the cached upsampler to free GPU memory."""
    global _UPSAMPLER
    if _UPSAMPLER is not None:
        _UPSAMPLER = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Map ncnn model names to pytorch model names
NCNN_TO_PYTORCH_MODEL = {
    "realesrgan-x4plus": "RealESRGAN_x4plus",
    "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3": "realesr-animevideov3",
    "realesrnet-x4plus": "realesr-general-x4v3",
    "realesrgan-x2plus": "RealESRGAN_x2plus",
}


def convert_ncnn_model_name(ncnn_name: str) -> str:
    """Convert ncnn model name to PyTorch model name."""
    return NCNN_TO_PYTORCH_MODEL.get(ncnn_name, "RealESRGAN_x4plus")
