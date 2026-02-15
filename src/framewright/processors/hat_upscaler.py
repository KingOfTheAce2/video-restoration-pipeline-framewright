"""HAT (Hybrid Attention Transformer) Upscaler for video restoration.

This module provides HAT-based super-resolution, which achieves state-of-the-art
quality by combining channel attention and self-attention mechanisms.

HAT advantages over other SR models:
- Better quality than Real-ESRGAN (less plastic look)
- More efficient than VRT with comparable quality
- Excellent detail preservation
- Good temporal consistency when used with overlapping windows

Model Sources:
- https://github.com/XPixelGroup/HAT (official)
- Weights: HAT-L_SRx4_ImageNet-pretrain.pth

VRAM Requirements (approximate):
- 720p: ~8 GB
- 1080p: ~14 GB
- 4K: ~28 GB (requires tiling on most GPUs)

Processing Speed (RTX 3080):
- 720p: ~2-3 fps
- 1080p: ~0.8-1.5 fps

Example:
    >>> config = HATConfig(scale=4, model_size="large")
    >>> upscaler = HATUpscaler(config)
    >>> if upscaler.is_available():
    ...     result = upscaler.upscale_video(input_path, output_path)
"""

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class HATModelSize(Enum):
    """HAT model size variants."""
    SMALL = "small"   # HAT-S: Faster, lower quality
    BASE = "base"     # HAT: Balanced
    LARGE = "large"   # HAT-L: Best quality, slower


# Model download URLs
HAT_MODEL_URLS = {
    HATModelSize.SMALL: "https://github.com/XPixelGroup/HAT/releases/download/v0.0.0/HAT-S_SRx4.pth",
    HATModelSize.BASE: "https://github.com/XPixelGroup/HAT/releases/download/v0.0.0/HAT_SRx4_ImageNet-pretrain.pth",
    HATModelSize.LARGE: "https://github.com/XPixelGroup/HAT/releases/download/v0.0.0/HAT-L_SRx4_ImageNet-pretrain.pth",
}


@dataclass
class HATConfig:
    """Configuration for HAT upscaler.

    Attributes:
        scale: Upscaling factor (2 or 4)
        model_size: HAT model variant (small, base, large)
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
        tile_size: Tile size for processing (0 = no tiling)
        tile_overlap: Overlap between tiles
        temporal_window: Frames to process together for consistency
    """
    scale: int = 4
    model_size: HATModelSize = HATModelSize.LARGE
    gpu_id: int = 0
    half_precision: bool = True
    tile_size: int = 0  # 0 = no tiling (for RTX 5090)
    tile_overlap: int = 32
    temporal_window: int = 1  # 1 = single frame, >1 = temporal batching


@dataclass
class HATResult:
    """Result of HAT upscaling.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        output_path: Path to output
        processing_time_seconds: Total processing time
        avg_fps: Average processing speed
        peak_vram_mb: Peak VRAM usage
    """
    frames_processed: int = 0
    frames_failed: int = 0
    output_path: Optional[Path] = None
    processing_time_seconds: float = 0.0
    avg_fps: float = 0.0
    peak_vram_mb: int = 0


class HATUpscaler:
    """HAT-based video/image super-resolution.

    Hybrid Attention Transformer achieves state-of-the-art SR quality
    by combining channel attention and self-attention mechanisms.

    Example:
        >>> config = HATConfig(scale=4, model_size=HATModelSize.LARGE)
        >>> upscaler = HATUpscaler(config)
        >>> if upscaler.is_available():
        ...     result = upscaler.upscale_frames(input_dir, output_dir)
    """

    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'hat'

    def __init__(
        self,
        config: Optional[HATConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize HAT upscaler.

        Args:
            config: HAT configuration
            model_dir: Directory for model weights
        """
        self.config = config or HATConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available HAT backend."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available - HAT disabled")
            return None

        if not HAS_OPENCV:
            logger.warning("OpenCV not available - HAT disabled")
            return None

        # Check for HAT model weights
        model_path = self._get_model_path()
        if model_path and model_path.exists():
            logger.info(f"Found HAT model weights at {model_path}")
            return 'hat_weights'

        # Check for basicsr with HAT
        try:
            from basicsr.archs.hat_arch import HAT
            logger.info("Found BasicSR HAT backend")
            return 'basicsr_hat'
        except ImportError:
            pass

        logger.info(
            "HAT model not found. Download with: "
            "upscaler.download_model() or manually from GitHub"
        )
        return None

    def _get_model_path(self) -> Optional[Path]:
        """Get path to HAT model weights."""
        model_files = {
            HATModelSize.SMALL: 'HAT-S_SRx4.pth',
            HATModelSize.BASE: 'HAT_SRx4_ImageNet-pretrain.pth',
            HATModelSize.LARGE: 'HAT-L_SRx4_ImageNet-pretrain.pth',
        }

        model_file = model_files.get(self.config.model_size)
        if model_file:
            path = self.model_dir / model_file
            if path.exists():
                return path

        # Check for any HAT model
        for name in model_files.values():
            path = self.model_dir / name
            if path.exists():
                return path

        return None

    def is_available(self) -> bool:
        """Check if HAT upscaling is available.

        Returns:
            True if HAT can be used
        """
        return self._backend is not None

    def download_model(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download HAT model weights.

        Downloads to ~/.framewright/models/hat/

        Args:
            progress_callback: Optional progress callback (0-1)

        Returns:
            True if model is available after download
        """
        url = HAT_MODEL_URLS.get(self.config.model_size)
        if not url:
            logger.error(f"No download URL for HAT {self.config.model_size.value}")
            return False

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_name = url.split('/')[-1]
        model_path = self.model_dir / model_name

        if model_path.exists():
            logger.info(f"HAT model already exists: {model_path}")
            self._backend = self._detect_backend()
            return self.is_available()

        logger.info(f"Downloading HAT {self.config.model_size.value} from {url}...")

        try:
            def reporthook(block_num, block_size, total_size):
                if progress_callback and total_size > 0:
                    progress = min(1.0, block_num * block_size / total_size)
                    progress_callback(progress)

            urlretrieve(url, model_path, reporthook=reporthook)
            logger.info(f"HAT model downloaded to {model_path}")

            self._backend = self._detect_backend()
            return self.is_available()

        except Exception as e:
            logger.error(f"Failed to download HAT model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def _load_model(self) -> None:
        """Load HAT model."""
        if self._model is not None:
            return

        self._device = torch.device(
            f'cuda:{self.config.gpu_id}' if torch.cuda.is_available() else 'cpu'
        )

        if self._backend == 'basicsr_hat':
            self._load_basicsr_model()
        elif self._backend == 'hat_weights':
            self._load_weights_only()
        else:
            raise RuntimeError(f"Cannot load model for backend: {self._backend}")

        logger.info(f"HAT model loaded on {self._device}")

    def _load_basicsr_model(self) -> None:
        """Load HAT using BasicSR library."""
        from basicsr.archs.hat_arch import HAT

        # HAT-L architecture parameters
        if self.config.model_size == HATModelSize.LARGE:
            self._model = HAT(
                upscale=self.config.scale,
                in_chans=3,
                img_size=64,
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv',
            )
        elif self.config.model_size == HATModelSize.BASE:
            self._model = HAT(
                upscale=self.config.scale,
                in_chans=3,
                img_size=64,
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv',
            )
        else:  # SMALL
            self._model = HAT(
                upscale=self.config.scale,
                in_chans=3,
                img_size=64,
                window_size=16,
                compress_ratio=24,
                squeeze_factor=24,
                conv_scale=0.01,
                overlap_ratio=0.5,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=144,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv',
            )

        # Load weights
        model_path = self._get_model_path()
        if model_path:
            checkpoint = torch.load(model_path, map_location=self._device)
            if 'params_ema' in checkpoint:
                self._model.load_state_dict(checkpoint['params_ema'], strict=False)
            elif 'params' in checkpoint:
                self._model.load_state_dict(checkpoint['params'], strict=False)
            else:
                self._model.load_state_dict(checkpoint, strict=False)

        self._model = self._model.to(self._device)
        self._model.eval()

        if self.config.half_precision and self._device.type == 'cuda':
            self._model = self._model.half()

    def _load_weights_only(self) -> None:
        """Load HAT with weights only (minimal deps)."""
        # This requires the HAT architecture to be available
        # For now, fall back to error
        logger.error(
            "Weights-only mode requires basicsr. "
            "Install with: pip install basicsr"
        )
        raise ImportError("basicsr required for HAT")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for HAT input.

        Args:
            frame: BGR frame as numpy array

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float() / 255.0

        # Reshape to [1, C, H, W]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        # Move to device
        tensor = tensor.to(self._device)

        if self.config.half_precision and self._device.type == 'cuda':
            tensor = tensor.half()

        return tensor

    def _postprocess_output(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess HAT output.

        Args:
            tensor: Output tensor [1, 3, H, W]

        Returns:
            BGR frame as numpy array
        """
        # Convert to float if half precision
        if tensor.dtype == torch.float16:
            tensor = tensor.float()

        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to numpy
        output = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Convert to [0, 255] uint8
        output = (output * 255.0).astype(np.uint8)

        # Convert RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output

    def _process_with_tiling(
        self,
        tensor: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """Process large image with tiling.

        Args:
            tensor: Input tensor [1, 3, H, W]
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            Upscaled tensor
        """
        _, _, h, w = tensor.shape
        scale = self.config.scale

        # Calculate output size
        out_h = h * scale
        out_w = w * scale

        # Initialize output
        output = torch.zeros(1, 3, out_h, out_w, device=self._device)
        if self.config.half_precision and self._device.type == 'cuda':
            output = output.half()

        # Process tiles
        step = tile_size - overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = tensor[:, :, y:y_end, x:x_end]

                # Process tile
                with torch.no_grad():
                    tile_out = self._model(tile)

                # Calculate output coordinates
                out_y = y * scale
                out_x = x * scale
                out_y_end = y_end * scale
                out_x_end = x_end * scale

                # Handle overlap blending
                if y > 0:
                    blend_h = overlap * scale // 2
                    out_y += blend_h
                    tile_out = tile_out[:, :, blend_h:, :]
                if x > 0:
                    blend_w = overlap * scale // 2
                    out_x += blend_w
                    tile_out = tile_out[:, :, :, blend_w:]

                # Place tile in output
                actual_h = tile_out.shape[2]
                actual_w = tile_out.shape[3]
                output[:, :, out_y:out_y+actual_h, out_x:out_x+actual_w] = tile_out

                # Clear cache
                if self._device.type == 'cuda':
                    torch.cuda.empty_cache()

        return output

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single frame.

        Args:
            frame: Input frame (BGR numpy array)

        Returns:
            Upscaled frame (BGR numpy array)
        """
        if not self.is_available():
            raise RuntimeError("HAT is not available")

        self._load_model()

        # Preprocess
        tensor = self._preprocess_frame(frame)

        # Process
        with torch.no_grad():
            if self.config.tile_size > 0:
                output = self._process_with_tiling(
                    tensor,
                    self.config.tile_size,
                    self.config.tile_overlap,
                )
            else:
                output = self._model(tensor)

        # Postprocess
        result = self._postprocess_output(output)

        return result

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> HATResult:
        """Upscale all frames in a directory.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback

        Returns:
            HATResult with processing statistics
        """
        result = HATResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("HAT is not available")
            return result

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_path = output_dir

        # Get frames
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found")
            return result

        n_frames = len(frames)
        logger.info(f"Upscaling {n_frames} frames with HAT-{self.config.model_size.value}")

        # Load model
        self._load_model()

        peak_vram = 0

        for i, frame_path in enumerate(frames):
            try:
                # Read frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    result.frames_failed += 1
                    continue

                # Upscale
                upscaled = self.upscale_frame(frame)

                # Track VRAM
                if HAS_TORCH and torch.cuda.is_available():
                    current_vram = torch.cuda.max_memory_allocated(self.config.gpu_id)
                    peak_vram = max(peak_vram, current_vram)

                # Save
                output_path = output_dir / frame_path.name
                cv2.imwrite(str(output_path), upscaled)

                result.frames_processed += 1

            except Exception as e:
                logger.error(f"Failed to upscale {frame_path}: {e}")
                result.frames_failed += 1

                # Copy original as fallback
                try:
                    shutil.copy2(frame_path, output_dir / frame_path.name)
                except Exception:
                    pass

            if progress_callback:
                progress_callback((i + 1) / n_frames)

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time
        result.peak_vram_mb = peak_vram // (1024 * 1024)

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        logger.info(
            f"HAT upscaling complete: {result.frames_processed}/{n_frames} frames, "
            f"{result.avg_fps:.2f} fps, peak VRAM: {result.peak_vram_mb} MB"
        )

        return result

    def upscale_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> HATResult:
        """Upscale a video file.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback

        Returns:
            HATResult with processing statistics
        """
        result = HATResult()
        start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self.is_available():
            logger.error("HAT is not available")
            return result

        with tempfile.TemporaryDirectory(prefix="hat_") as temp_dir:
            temp_dir = Path(temp_dir)
            input_frames = temp_dir / "input"
            output_frames = temp_dir / "output"
            input_frames.mkdir()
            output_frames.mkdir()

            # Extract frames
            if progress_callback:
                progress_callback(0.05)

            extract_cmd = [
                'ffmpeg', '-i', str(input_path),
                '-qscale:v', '2',
                str(input_frames / 'frame_%08d.png'),
                '-hide_banner', '-loglevel', 'error',
            ]
            subprocess.run(extract_cmd, check=True)

            # Upscale frames
            frame_result = self.upscale_frames(
                input_frames, output_frames,
                lambda p: progress_callback(0.05 + p * 0.85) if progress_callback else None
            )

            result.frames_processed = frame_result.frames_processed
            result.frames_failed = frame_result.frames_failed
            result.peak_vram_mb = frame_result.peak_vram_mb

            # Get original video info
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(input_path),
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            fps_str = probe_result.stdout.strip()

            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den else 30
            else:
                fps = float(fps_str) if fps_str else 30

            # Encode output
            if progress_callback:
                progress_callback(0.9)

            encode_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(output_frames / 'frame_%08d.png'),
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '16',
                '-pix_fmt', 'yuv420p',
                str(output_path),
                '-hide_banner', '-loglevel', 'error',
            ]
            subprocess.run(encode_cmd, check=True)

            result.output_path = output_path

        result.processing_time_seconds = time.time() - start_time

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        if progress_callback:
            progress_callback(1.0)

        return result

    def clear_cache(self) -> None:
        """Clear model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_hat_upscaler(
    scale: int = 4,
    model_size: str = "large",
    gpu_id: int = 0,
    tile_size: int = 0,
) -> HATUpscaler:
    """Factory function to create a HAT upscaler.

    Args:
        scale: Upscaling factor (2 or 4)
        model_size: Model size (small, base, large)
        gpu_id: GPU device ID
        tile_size: Tile size (0 = no tiling)

    Returns:
        Configured HATUpscaler
    """
    config = HATConfig(
        scale=scale,
        model_size=HATModelSize(model_size),
        gpu_id=gpu_id,
        tile_size=tile_size,
    )
    return HATUpscaler(config)
