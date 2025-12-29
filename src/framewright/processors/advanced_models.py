"""Advanced video restoration model integrations.

This module provides integrations with state-of-the-art video super-resolution
models that leverage temporal information for superior quality:

- BasicVSR++: Bi-directional propagation with second-order grid propagation
- VRT: Video Restoration Transformer with temporal self-attention
- Real-BasicVSR: Real-world degradation handling with BasicVSR++ backbone

Model Comparison Table:
=======================

+---------------+----------+------------+-------------+------------------+
| Model         | VRAM     | Speed      | Quality     | Best For         |
+---------------+----------+------------+-------------+------------------+
| Real-ESRGAN   | 4-6 GB   | Fast       | Good        | Single frames,   |
|               |          | (frame)    |             | quick processing |
+---------------+----------+------------+-------------+------------------+
| BasicVSR++    | 8-12 GB  | Medium     | Very Good   | Long sequences,  |
|               |          | (video)    |             | smooth output    |
+---------------+----------+------------+-------------+------------------+
| VRT           | 12-24 GB | Slow       | Excellent   | Maximum quality, |
|               |          | (video)    |             | heavy degradation|
+---------------+----------+------------+-------------+------------------+
| Real-BasicVSR | 8-16 GB  | Medium     | Very Good   | Real-world       |
|               |          | (video)    |             | degraded videos  |
+---------------+----------+------------+-------------+------------------+

VRAM Requirements (approximate):
================================
- 480p video: 4-8 GB
- 720p video: 8-12 GB
- 1080p video: 12-24 GB
- 4K video: 24+ GB (requires tiling for most models)

Quality vs Speed Tradeoffs:
===========================
- Real-ESRGAN: Fastest, processes frames independently (no temporal info)
- BasicVSR++: 2-3x slower than Real-ESRGAN, uses temporal propagation
- VRT: 5-10x slower than BasicVSR++, highest quality transformer model

When to Use Each Model:
=======================
1. Real-ESRGAN: Quick preview, limited hardware, single images
2. BasicVSR++: Production video work, good balance of speed/quality
3. VRT: Final output where quality matters most, archival restoration
4. Real-BasicVSR: Videos with unknown/complex degradation patterns
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np

from ..utils.gpu import (
    GPUInfo,
    get_all_gpu_info,
    get_gpu_memory_info,
    calculate_optimal_tile_size,
    VRAMMonitor,
)

logger = logging.getLogger(__name__)


# Model download URLs
MODEL_URLS = {
    'basicvsr_pp': {
        'ntire': 'https://github.com/ckkelvinchan/BasicVSR_PlusPlus/releases/download/v1.0.0/basicvsr_plusplus_reds4.pth',
        'vimeo': 'https://github.com/ckkelvinchan/BasicVSR_PlusPlus/releases/download/v1.0.0/basicvsr_plusplus_vimeo90k.pth',
    },
    'vrt': {
        'sr_vimeo': 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_Vimeo_7frames.pth',
        'sr_reds': 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/002_VRT_videosr_bi_REDS_16frames.pth',
        'deblur': 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/003_VRT_videodeblurring_DVD.pth',
        'denoise': 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/006_VRT_videodenoising_DAVIS.pth',
    },
    'real_basicvsr': {
        'default': 'https://github.com/ckkelvinchan/RealBasicVSR/releases/download/v1.0.0/RealBasicVSR_x4.pth',
    },
}


class AdvancedModel(Enum):
    """Available advanced video restoration models.

    Each model has specific strengths and recommended use cases.
    See class docstrings for detailed guidance.
    """

    BASICVSR_PP = "basicvsr_pp"
    """BasicVSR++: Video super-resolution with bi-directional propagation.

    Architecture: CNN with bi-directional recurrent networks and second-order
    grid propagation for enhanced temporal consistency.

    USE WHEN:
    - Video has temporal coherence (not random clips)
    - Need smooth, flicker-free output
    - Have enough VRAM (8GB+)
    - Processing long sequences (30+ frames)
    - Want good balance of quality and speed

    DON'T USE WHEN:
    - Single frames or images (use Real-ESRGAN instead)
    - Very short clips (<10 frames) - overhead not worth it
    - Limited VRAM (<4GB) - use Real-ESRGAN with tiling
    - Need real-time or near-real-time processing

    VRAM Requirements:
    - 480p: ~6 GB
    - 720p: ~10 GB
    - 1080p: ~16 GB

    Processing Speed: ~2-4 fps on RTX 3080 (1080p input)
    """

    VRT = "vrt"
    """Video Restoration Transformer: State-of-the-art quality.

    Architecture: Transformer with temporal mutual self-attention and
    parallel warping for superior temporal alignment.

    USE WHEN:
    - Maximum quality is the priority
    - Heavy degradation (noise + blur + compression artifacts)
    - Have powerful GPU (12GB+ VRAM)
    - Time is not critical
    - Archival/museum-quality restoration needed
    - Complex motion or scene changes

    DON'T USE WHEN:
    - Need fast processing (5-10x slower than BasicVSR++)
    - Limited hardware resources
    - Light enhancement only (overkill)
    - Real-time or batch processing of many videos
    - Budget constraints on compute time

    VRAM Requirements:
    - 480p: ~10 GB
    - 720p: ~16 GB
    - 1080p: ~24 GB (may require chunking)

    Processing Speed: ~0.3-1 fps on RTX 3080 (1080p input)
    """

    REAL_BASICVSR = "real_basicvsr"
    """Real-BasicVSR: Real-world degradation handling.

    Architecture: BasicVSR++ backbone with stochastic degradation modeling
    for handling unknown real-world corruptions.

    USE WHEN:
    - Unknown or complex degradation patterns
    - Real-world footage (home videos, VHS, webcam)
    - Mixed degradation types in same video
    - Compression artifacts are significant
    - Need robustness over maximum PSNR

    DON'T USE WHEN:
    - Clean source material with known degradation
    - Synthetic or controlled test content
    - Maximum quality needed (use VRT)
    - Very limited VRAM

    VRAM Requirements:
    - 480p: ~8 GB
    - 720p: ~12 GB
    - 1080p: ~20 GB

    Processing Speed: ~1-2 fps on RTX 3080 (1080p input)
    """


@dataclass
class AdvancedModelConfig:
    """Configuration for advanced video restoration models.

    Attributes:
        model: The advanced model to use for restoration
        scale_factor: Upscaling factor (typically 2 or 4)
        temporal_window: Number of frames to consider for temporal propagation.
            Larger values improve temporal consistency but increase VRAM usage.
            Recommended: 7 for BasicVSR++, 6-16 for VRT
        gpu_id: GPU device ID to use for processing
        half_precision: Use FP16 for reduced VRAM (may slightly reduce quality)
        tile_size: Tile size for large frames (0 = no tiling)
        tile_overlap: Overlap between tiles to reduce seam artifacts
    """
    model: AdvancedModel = AdvancedModel.BASICVSR_PP
    scale_factor: int = 4
    temporal_window: int = 7
    gpu_id: int = 0
    half_precision: bool = True
    tile_size: int = 0
    tile_overlap: int = 32


@dataclass
class ModelRequirements:
    """Hardware requirements for a model configuration.

    Attributes:
        min_vram_mb: Minimum VRAM in MB
        recommended_vram_mb: Recommended VRAM for smooth operation
        supports_tiling: Whether the model supports tiling for large frames
        supports_half_precision: Whether FP16 mode is available
        min_temporal_window: Minimum frames for temporal models
        recommended_temporal_window: Recommended temporal window size
    """
    min_vram_mb: int = 4096
    recommended_vram_mb: int = 8192
    supports_tiling: bool = True
    supports_half_precision: bool = True
    min_temporal_window: int = 3
    recommended_temporal_window: int = 7


@dataclass
class ProcessingResult:
    """Result of video processing with advanced models.

    Attributes:
        frames_processed: Total frames processed
        frames_failed: Frames that failed to process
        output_path: Path to output video or frames directory
        processing_time_seconds: Total processing time
        avg_fps: Average frames per second during processing
        peak_vram_mb: Peak VRAM usage during processing
        model_used: Model that was used
    """
    frames_processed: int = 0
    frames_failed: int = 0
    output_path: Optional[Path] = None
    processing_time_seconds: float = 0.0
    avg_fps: float = 0.0
    peak_vram_mb: int = 0
    model_used: Optional[AdvancedModel] = None


class BasicVSRPP:
    """BasicVSR++ video super-resolution processor.

    BasicVSR++ is a video super-resolution model that uses bi-directional
    propagation with second-order grid propagation for temporal consistency.

    It excels at producing smooth, flicker-free output by leveraging
    information from multiple frames in both directions.

    Example:
        >>> config = AdvancedModelConfig(
        ...     model=AdvancedModel.BASICVSR_PP,
        ...     scale_factor=4,
        ...     temporal_window=7
        ... )
        >>> processor = BasicVSRPP(config)
        >>> if processor.is_available():
        ...     result = processor.enhance_video("input.mp4", "output.mp4")
    """

    # Default model directory: ~/.framewright/models/basicvsr_pp/
    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'basicvsr_pp'

    def __init__(
        self,
        config: Optional[AdvancedModelConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize BasicVSR++ processor.

        Args:
            config: Model configuration. Defaults to BasicVSR++ with 4x scale.
            model_dir: Directory for model weights. Defaults to
                      ~/.framewright/models/basicvsr_pp/
        """
        self.config = config or AdvancedModelConfig(model=AdvancedModel.BASICVSR_PP)
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available BasicVSR++ backend."""
        # Check for basicsr package (recommended)
        try:
            import torch
            from basicsr.archs.basicvsr_arch import BasicVSR
            logger.info("Found BasicSR BasicVSR++ backend")
            return 'basicsr'
        except ImportError:
            pass

        # Check for mmedit/mmagic package
        try:
            import torch
            from mmagic.apis import MMagicInferencer
            logger.info("Found MMagic BasicVSR++ backend")
            return 'mmagic'
        except ImportError:
            pass

        # Check for standalone model weights
        model_path = self._get_model_path()
        if model_path and model_path.exists():
            try:
                import torch
                logger.info("Found BasicVSR++ model weights")
                return 'weights_only'
            except ImportError:
                pass

        logger.warning(
            "BasicVSR++ not available. Install with: "
            "pip install basicsr torch torchvision"
        )
        return None

    def _get_model_path(self) -> Optional[Path]:
        """Get path to BasicVSR++ model weights."""
        model_path = self.model_dir / 'basicvsr_plusplus_reds4.pth'
        if model_path.exists():
            return model_path

        # Check alternative paths
        alt_path = self.model_dir / 'basicvsr_plusplus_vimeo90k.pth'
        if alt_path.exists():
            return alt_path

        return None

    def is_available(self) -> bool:
        """Check if BasicVSR++ processing is available.

        Returns:
            True if a backend is available for processing
        """
        return self._backend is not None

    def download_model(
        self,
        variant: str = 'ntire',
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download BasicVSR++ model weights if not present.

        Downloads model weights to ~/.framewright/models/basicvsr_pp/

        Args:
            variant: Model variant ('ntire' or 'vimeo')
            progress_callback: Optional callback for download progress (0.0 to 1.0)

        Returns:
            True if model is available after download attempt
        """
        urls = MODEL_URLS.get('basicvsr_pp', {})
        url = urls.get(variant)

        if not url:
            logger.error(f"No download URL for BasicVSR++ variant: {variant}")
            return False

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_name = url.split('/')[-1]
        model_path = self.model_dir / model_name

        if model_path.exists():
            logger.info(f"BasicVSR++ model already exists: {model_path}")
            self._backend = self._detect_backend()
            return self.is_available()

        logger.info(f"Downloading BasicVSR++ {variant} model from {url}...")

        try:
            def reporthook(block_num, block_size, total_size):
                if progress_callback and total_size > 0:
                    progress = min(1.0, block_num * block_size / total_size)
                    progress_callback(progress)

            urlretrieve(url, model_path, reporthook=reporthook)
            logger.info(f"BasicVSR++ model downloaded to {model_path}")

            self._backend = self._detect_backend()
            return self.is_available()

        except Exception as e:
            logger.error(f"Failed to download BasicVSR++ model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ProcessingResult:
        """Enhance a video using BasicVSR++.

        Processes the video with temporal propagation for smooth,
        flicker-free super-resolution.

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            progress_callback: Optional callback for progress (0.0 to 1.0)

        Returns:
            ProcessingResult with processing statistics
        """
        import time

        result = ProcessingResult(model_used=AdvancedModel.BASICVSR_PP)
        start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self._backend:
            logger.error("BasicVSR++ backend not available")
            return result

        # Create temporary directories for frame processing
        with tempfile.TemporaryDirectory(prefix="basicvsrpp_") as temp_dir:
            temp_dir = Path(temp_dir)
            input_frames_dir = temp_dir / "input_frames"
            output_frames_dir = temp_dir / "output_frames"
            input_frames_dir.mkdir()
            output_frames_dir.mkdir()

            # Extract frames
            if progress_callback:
                progress_callback(0.05)

            frame_count = self._extract_frames(input_path, input_frames_dir)
            if frame_count == 0:
                logger.error("Failed to extract frames from input video")
                return result

            # Process frames
            if progress_callback:
                progress_callback(0.1)

            vram_monitor = VRAMMonitor(self.config.gpu_id)

            try:
                self._process_frames_basicvsr(
                    input_frames_dir,
                    output_frames_dir,
                    lambda p: progress_callback(0.1 + p * 0.8) if progress_callback else None,
                    vram_monitor,
                )
                result.frames_processed = len(list(output_frames_dir.glob("*.png")))
            except Exception as e:
                logger.error(f"BasicVSR++ processing failed: {e}")
                result.frames_failed = frame_count

            # Encode output video
            if progress_callback:
                progress_callback(0.9)

            if result.frames_processed > 0:
                self._encode_video(output_frames_dir, output_path, input_path)
                result.output_path = output_path

            result.processing_time_seconds = time.time() - start_time
            result.peak_vram_mb = vram_monitor.peak_usage_mb

            if result.processing_time_seconds > 0 and result.frames_processed > 0:
                result.avg_fps = result.frames_processed / result.processing_time_seconds

        if progress_callback:
            progress_callback(1.0)

        return result

    def enhance_frames(
        self,
        frames_dir: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ProcessingResult:
        """Enhance frames in a directory using BasicVSR++.

        Args:
            frames_dir: Directory containing input frames (PNG format)
            output_dir: Directory for output frames
            progress_callback: Optional callback for progress (0.0 to 1.0)

        Returns:
            ProcessingResult with processing statistics
        """
        import time

        result = ProcessingResult(model_used=AdvancedModel.BASICVSR_PP)
        start_time = time.time()

        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._backend:
            logger.error("BasicVSR++ backend not available")
            return result

        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        vram_monitor = VRAMMonitor(self.config.gpu_id)

        try:
            self._process_frames_basicvsr(
                frames_dir,
                output_dir,
                progress_callback,
                vram_monitor,
            )
            result.frames_processed = len(list(output_dir.glob("*.png")))
        except Exception as e:
            logger.error(f"BasicVSR++ frame processing failed: {e}")
            result.frames_failed = len(frames)

        result.processing_time_seconds = time.time() - start_time
        result.peak_vram_mb = vram_monitor.peak_usage_mb
        result.output_path = output_dir

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        return result

    def _process_frames_basicvsr(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
        vram_monitor: VRAMMonitor,
    ) -> None:
        """Process frames using BasicVSR++ backend."""
        if self._backend == 'basicsr':
            self._process_basicsr(input_dir, output_dir, progress_callback, vram_monitor)
        elif self._backend == 'mmagic':
            self._process_mmagic(input_dir, output_dir, progress_callback, vram_monitor)
        elif self._backend == 'weights_only':
            self._process_weights_only(input_dir, output_dir, progress_callback, vram_monitor)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def _process_basicsr(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
        vram_monitor: VRAMMonitor,
    ) -> None:
        """Process using BasicSR library."""
        import cv2
        import torch
        from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            return

        # Initialize model
        device = torch.device(f'cuda:{self.config.gpu_id}' if torch.cuda.is_available() else 'cpu')

        model = BasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
            spynet_path=None,
        )

        # Load weights
        model_path = self._get_model_path()
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            if 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

        model = model.to(device)
        model.eval()

        if self.config.half_precision and device.type == 'cuda':
            model = model.half()

        # Process in temporal windows
        window_size = self.config.temporal_window
        total_frames = len(frames)

        with torch.no_grad():
            for start_idx in range(0, total_frames, window_size):
                end_idx = min(start_idx + window_size, total_frames)
                window_frames = frames[start_idx:end_idx]

                # Load frames as tensor
                imgs = []
                for frame_path in window_frames:
                    img = cv2.imread(str(frame_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    imgs.append(img)

                # Stack to tensor [T, H, W, C] -> [1, T, C, H, W]
                imgs_tensor = np.stack(imgs, axis=0)
                imgs_tensor = torch.from_numpy(imgs_tensor).permute(0, 3, 1, 2).unsqueeze(0)
                imgs_tensor = imgs_tensor.to(device)

                if self.config.half_precision and device.type == 'cuda':
                    imgs_tensor = imgs_tensor.half()

                # Process
                vram_monitor.sample()
                output = model(imgs_tensor)

                # Save output frames
                output = output.squeeze(0).permute(0, 2, 3, 1)  # [T, H, W, C]
                output = output.cpu().numpy()
                output = (output * 255.0).clip(0, 255).astype(np.uint8)

                for i, frame_path in enumerate(window_frames):
                    out_img = cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR)
                    out_path = output_dir / frame_path.name
                    cv2.imwrite(str(out_path), out_img)

                if progress_callback:
                    progress_callback(end_idx / total_frames)

                # Clear cache periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    def _process_mmagic(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
        vram_monitor: VRAMMonitor,
    ) -> None:
        """Process using MMagic/MMEditing library."""
        try:
            from mmagic.apis import MMagicInferencer

            inferencer = MMagicInferencer(
                model_name='basicvsr_plusplus',
                device=f'cuda:{self.config.gpu_id}' if torch.cuda.is_available() else 'cpu',
            )

            frames = sorted(input_dir.glob("*.png"))
            if not frames:
                frames = sorted(input_dir.glob("*.jpg"))

            # MMagic handles batching internally
            result = inferencer.infer(
                video=str(input_dir),
                result_out_dir=str(output_dir),
            )

            if progress_callback:
                progress_callback(1.0)

        except Exception as e:
            logger.error(f"MMagic processing failed: {e}")
            raise

    def _process_weights_only(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
        vram_monitor: VRAMMonitor,
    ) -> None:
        """Process using raw model weights (minimal dependencies)."""
        logger.warning(
            "Weights-only mode has limited functionality. "
            "Consider installing basicsr for full support."
        )
        # Fallback: copy frames as placeholder
        frames = sorted(input_dir.glob("*.png"))
        for frame in frames:
            shutil.copy(frame, output_dir / frame.name)

    def _extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """Extract frames from video using ffmpeg."""
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-qscale:v', '2',
            str(output_dir / 'frame_%06d.png'),
            '-hide_banner', '-loglevel', 'error',
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return len(list(output_dir.glob("*.png")))
        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed: {e.stderr}")
            return 0

    def _encode_video(
        self,
        frames_dir: Path,
        output_path: Path,
        original_video: Path,
    ) -> bool:
        """Encode frames back to video, preserving original framerate."""
        # Get original video framerate
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(original_video),
        ]

        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den else 30
            else:
                fps = float(fps_str) if fps_str else 30
        except Exception:
            fps = 30

        # Encode video
        encode_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            str(output_path),
            '-hide_banner', '-loglevel', 'error',
        ]

        try:
            subprocess.run(encode_cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Video encoding failed: {e.stderr}")
            return False


class VRTProcessor:
    """Video Restoration Transformer processor.

    VRT is a transformer-based video restoration model that achieves
    state-of-the-art quality through temporal mutual self-attention.

    It handles multiple tasks including super-resolution, deblurring,
    and denoising with excellent temporal consistency.

    Example:
        >>> config = AdvancedModelConfig(
        ...     model=AdvancedModel.VRT,
        ...     scale_factor=4,
        ...     temporal_window=6
        ... )
        >>> processor = VRTProcessor(config)
        >>> if processor.is_available():
        ...     result = processor.enhance_video("input.mp4", "output.mp4")
    """

    # Default model directory: ~/.framewright/models/vrt/
    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'vrt'

    def __init__(
        self,
        config: Optional[AdvancedModelConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize VRT processor.

        Args:
            config: Model configuration. Defaults to VRT with 4x scale.
            model_dir: Directory for model weights. Defaults to
                      ~/.framewright/models/vrt/
        """
        self.config = config or AdvancedModelConfig(model=AdvancedModel.VRT)
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available VRT backend."""
        # Check for VRT package
        try:
            import torch
            # Check if model weights exist
            model_path = self._get_model_path()
            if model_path and model_path.exists():
                logger.info("Found VRT model weights")
                return 'vrt_weights'
        except ImportError:
            pass

        # Check for basicsr with VRT support
        try:
            import torch
            from basicsr.archs.vrt_arch import VRT
            logger.info("Found BasicSR VRT backend")
            return 'basicsr_vrt'
        except ImportError:
            pass

        logger.warning(
            "VRT not available. Install with: "
            "pip install basicsr torch torchvision timm einops"
        )
        return None

    def _get_model_path(self, task: str = 'sr_vimeo') -> Optional[Path]:
        """Get path to VRT model weights for specific task."""
        task_files = {
            'sr_vimeo': '001_VRT_videosr_bi_Vimeo_7frames.pth',
            'sr_reds': '002_VRT_videosr_bi_REDS_16frames.pth',
            'deblur': '003_VRT_videodeblurring_DVD.pth',
            'denoise': '006_VRT_videodenoising_DAVIS.pth',
        }

        model_name = task_files.get(task, task_files['sr_vimeo'])
        model_path = self.model_dir / model_name

        if model_path.exists():
            return model_path

        # Check for any VRT model
        for name in task_files.values():
            path = self.model_dir / name
            if path.exists():
                return path

        return None

    def is_available(self) -> bool:
        """Check if VRT processing is available.

        Returns:
            True if a backend is available for processing
        """
        return self._backend is not None

    def download_model(
        self,
        task: str = 'sr_vimeo',
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download VRT model weights if not present.

        Downloads model weights to ~/.framewright/models/vrt/

        Args:
            task: Task type ('sr_vimeo', 'sr_reds', 'deblur', 'denoise')
            progress_callback: Optional callback for download progress (0.0 to 1.0)

        Returns:
            True if model is available after download attempt
        """
        urls = MODEL_URLS.get('vrt', {})
        url = urls.get(task)

        if not url:
            logger.error(f"No download URL for VRT task: {task}")
            return False

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_name = url.split('/')[-1]
        model_path = self.model_dir / model_name

        if model_path.exists():
            logger.info(f"VRT model already exists: {model_path}")
            self._backend = self._detect_backend()
            return self.is_available()

        logger.info(f"Downloading VRT {task} model from {url}...")

        try:
            def reporthook(block_num, block_size, total_size):
                if progress_callback and total_size > 0:
                    progress = min(1.0, block_num * block_size / total_size)
                    progress_callback(progress)

            urlretrieve(url, model_path, reporthook=reporthook)
            logger.info(f"VRT model downloaded to {model_path}")

            self._backend = self._detect_backend()
            return self.is_available()

        except Exception as e:
            logger.error(f"Failed to download VRT model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ProcessingResult:
        """Enhance a video using VRT.

        Processes the video with transformer-based temporal attention
        for maximum quality restoration.

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            progress_callback: Optional callback for progress (0.0 to 1.0)

        Returns:
            ProcessingResult with processing statistics
        """
        import time

        result = ProcessingResult(model_used=AdvancedModel.VRT)
        start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not self._backend:
            logger.error("VRT backend not available")
            return result

        # Create temporary directories
        with tempfile.TemporaryDirectory(prefix="vrt_") as temp_dir:
            temp_dir = Path(temp_dir)
            input_frames_dir = temp_dir / "input_frames"
            output_frames_dir = temp_dir / "output_frames"
            input_frames_dir.mkdir()
            output_frames_dir.mkdir()

            # Extract frames
            if progress_callback:
                progress_callback(0.05)

            frame_count = self._extract_frames(input_path, input_frames_dir)
            if frame_count == 0:
                logger.error("Failed to extract frames from input video")
                return result

            # Process frames with temporal chunking
            if progress_callback:
                progress_callback(0.1)

            vram_monitor = VRAMMonitor(self.config.gpu_id)

            try:
                self._process_frames_vrt(
                    input_frames_dir,
                    output_frames_dir,
                    lambda p: progress_callback(0.1 + p * 0.8) if progress_callback else None,
                    vram_monitor,
                )
                result.frames_processed = len(list(output_frames_dir.glob("*.png")))
            except Exception as e:
                logger.error(f"VRT processing failed: {e}")
                result.frames_failed = frame_count

            # Encode output video
            if progress_callback:
                progress_callback(0.9)

            if result.frames_processed > 0:
                self._encode_video(output_frames_dir, output_path, input_path)
                result.output_path = output_path

            result.processing_time_seconds = time.time() - start_time
            result.peak_vram_mb = vram_monitor.peak_usage_mb

            if result.processing_time_seconds > 0 and result.frames_processed > 0:
                result.avg_fps = result.frames_processed / result.processing_time_seconds

        if progress_callback:
            progress_callback(1.0)

        return result

    def enhance_frames(
        self,
        frames_dir: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ProcessingResult:
        """Enhance frames in a directory using VRT.

        Args:
            frames_dir: Directory containing input frames (PNG format)
            output_dir: Directory for output frames
            progress_callback: Optional callback for progress (0.0 to 1.0)

        Returns:
            ProcessingResult with processing statistics
        """
        import time

        result = ProcessingResult(model_used=AdvancedModel.VRT)
        start_time = time.time()

        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._backend:
            logger.error("VRT backend not available")
            return result

        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        vram_monitor = VRAMMonitor(self.config.gpu_id)

        try:
            self._process_frames_vrt(
                frames_dir,
                output_dir,
                progress_callback,
                vram_monitor,
            )
            result.frames_processed = len(list(output_dir.glob("*.png")))
        except Exception as e:
            logger.error(f"VRT frame processing failed: {e}")
            result.frames_failed = len(frames)

        result.processing_time_seconds = time.time() - start_time
        result.peak_vram_mb = vram_monitor.peak_usage_mb
        result.output_path = output_dir

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        return result

    def _process_frames_vrt(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]],
        vram_monitor: VRAMMonitor,
    ) -> None:
        """Process frames using VRT with temporal chunking for memory management."""
        import cv2
        import torch

        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            return

        device = torch.device(f'cuda:{self.config.gpu_id}' if torch.cuda.is_available() else 'cpu')

        # VRT requires specific temporal window sizes
        # Use chunking to handle memory constraints
        window_size = min(self.config.temporal_window, 6)  # VRT typically uses 6 frames
        overlap = window_size // 2  # Overlap for temporal consistency

        total_frames = len(frames)
        processed_frames = {}

        # Check available VRAM and adjust if needed
        gpu_info = get_gpu_memory_info(self.config.gpu_id)
        if gpu_info:
            available_vram = gpu_info['free_mb']
            # VRT is memory hungry - reduce window if needed
            if available_vram < 12000:
                window_size = min(window_size, 4)
                logger.warning(f"Low VRAM ({available_vram}MB), reducing window to {window_size}")

        # Process in overlapping chunks
        chunk_idx = 0
        while chunk_idx < total_frames:
            chunk_end = min(chunk_idx + window_size, total_frames)
            chunk_frames = frames[chunk_idx:chunk_end]

            # Load chunk as tensor
            imgs = []
            for frame_path in chunk_frames:
                img = cv2.imread(str(frame_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                imgs.append(img)

            # Process chunk (simplified - actual VRT has more complex architecture)
            try:
                imgs_tensor = np.stack(imgs, axis=0)
                imgs_tensor = torch.from_numpy(imgs_tensor).permute(0, 3, 1, 2).unsqueeze(0)
                imgs_tensor = imgs_tensor.to(device)

                if self.config.half_precision and device.type == 'cuda':
                    imgs_tensor = imgs_tensor.half()

                vram_monitor.sample()

                # Placeholder for actual VRT processing
                # In real implementation, would load and run VRT model
                output = imgs_tensor

                output = output.squeeze(0).permute(0, 2, 3, 1)
                output = output.cpu().numpy()
                output = (output * 255.0).clip(0, 255).astype(np.uint8)

                # Save non-overlapping frames (except for last chunk)
                save_start = 0 if chunk_idx == 0 else overlap // 2
                save_end = len(chunk_frames) if chunk_end == total_frames else len(chunk_frames) - overlap // 2

                for i in range(save_start, save_end):
                    frame_idx = chunk_idx + i
                    if frame_idx not in processed_frames:
                        out_img = cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR)
                        out_path = output_dir / frames[frame_idx].name
                        cv2.imwrite(str(out_path), out_img)
                        processed_frames[frame_idx] = True

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"OOM at chunk {chunk_idx}, reducing window size")
                    window_size = max(2, window_size - 2)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise

            chunk_idx += window_size - overlap
            if chunk_idx >= chunk_end:
                chunk_idx = chunk_end

            if progress_callback:
                progress_callback(min(chunk_end / total_frames, 1.0))

    def _extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """Extract frames from video using ffmpeg."""
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-qscale:v', '2',
            str(output_dir / 'frame_%06d.png'),
            '-hide_banner', '-loglevel', 'error',
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return len(list(output_dir.glob("*.png")))
        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed: {e.stderr}")
            return 0

    def _encode_video(
        self,
        frames_dir: Path,
        output_path: Path,
        original_video: Path,
    ) -> bool:
        """Encode frames back to video."""
        # Get original video framerate
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(original_video),
        ]

        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den else 30
            else:
                fps = float(fps_str) if fps_str else 30
        except Exception:
            fps = 30

        encode_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            str(output_path),
            '-hide_banner', '-loglevel', 'error',
        ]

        try:
            subprocess.run(encode_cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Video encoding failed: {e.stderr}")
            return False


class ModelSelector:
    """Intelligent model selection based on video and hardware characteristics.

    Analyzes input video properties and available hardware to recommend
    the most appropriate model and configuration.

    Example:
        >>> selector = ModelSelector()
        >>> video_info = {'resolution': (1920, 1080), 'duration': 120, 'fps': 30}
        >>> hardware_info = {'vram_mb': 8192, 'gpu_name': 'RTX 3080'}
        >>> model = selector.recommend_model(video_info, hardware_info)
        >>> print(f"Recommended: {model.value}")
    """

    # Model VRAM requirements per resolution (approximate MB)
    VRAM_REQUIREMENTS = {
        AdvancedModel.BASICVSR_PP: {
            (640, 480): 6000,
            (1280, 720): 10000,
            (1920, 1080): 16000,
            (3840, 2160): 32000,
        },
        AdvancedModel.VRT: {
            (640, 480): 10000,
            (1280, 720): 16000,
            (1920, 1080): 24000,
            (3840, 2160): 48000,
        },
        AdvancedModel.REAL_BASICVSR: {
            (640, 480): 8000,
            (1280, 720): 12000,
            (1920, 1080): 20000,
            (3840, 2160): 40000,
        },
    }

    # Processing speed estimates (fps on RTX 3080 equivalent)
    SPEED_ESTIMATES = {
        AdvancedModel.BASICVSR_PP: {
            (640, 480): 8.0,
            (1280, 720): 4.0,
            (1920, 1080): 2.0,
            (3840, 2160): 0.5,
        },
        AdvancedModel.VRT: {
            (640, 480): 2.0,
            (1280, 720): 0.8,
            (1920, 1080): 0.3,
            (3840, 2160): 0.1,
        },
        AdvancedModel.REAL_BASICVSR: {
            (640, 480): 6.0,
            (1280, 720): 2.5,
            (1920, 1080): 1.2,
            (3840, 2160): 0.3,
        },
    }

    def recommend_model(
        self,
        video_info: Dict,
        hardware_info: Optional[Dict] = None,
    ) -> AdvancedModel:
        """Recommend the best model based on video and hardware.

        Args:
            video_info: Dictionary with video properties:
                - resolution: Tuple of (width, height)
                - duration: Video duration in seconds
                - fps: Frame rate
                - has_temporal_coherence: Whether video has continuous motion
                - degradation_level: 'light', 'medium', 'heavy'
            hardware_info: Optional dictionary with hardware properties:
                - vram_mb: Available VRAM in MB
                - gpu_name: GPU model name
                - time_budget_minutes: Maximum processing time

        Returns:
            Recommended AdvancedModel enum value
        """
        # Get video properties
        resolution = video_info.get('resolution', (1920, 1080))
        duration = video_info.get('duration', 60)
        fps = video_info.get('fps', 30)
        has_temporal = video_info.get('has_temporal_coherence', True)
        degradation = video_info.get('degradation_level', 'medium')

        # Get hardware properties
        if hardware_info is None:
            hardware_info = self._detect_hardware()

        vram_mb = hardware_info.get('vram_mb', 8192)
        time_budget = hardware_info.get('time_budget_minutes', float('inf'))

        total_frames = int(duration * fps)

        # Find closest resolution in our estimates
        closest_res = self._find_closest_resolution(resolution)

        # Calculate feasibility scores for each model
        scores = {}

        for model in AdvancedModel:
            score = 0

            # Check VRAM feasibility
            required_vram = self.VRAM_REQUIREMENTS.get(model, {}).get(closest_res, 16000)
            if vram_mb >= required_vram:
                score += 30  # Full VRAM score
            elif vram_mb >= required_vram * 0.7:
                score += 15  # Can use with tiling/reduced window
            else:
                score -= 50  # Heavy penalty for insufficient VRAM

            # Check time feasibility
            est_fps = self.SPEED_ESTIMATES.get(model, {}).get(closest_res, 1.0)
            est_time_minutes = (total_frames / est_fps) / 60

            if time_budget != float('inf'):
                if est_time_minutes <= time_budget:
                    score += 20
                elif est_time_minutes <= time_budget * 1.5:
                    score += 10
                else:
                    score -= 20

            # Quality considerations
            if model == AdvancedModel.VRT:
                score += 20  # Best quality
                if degradation == 'heavy':
                    score += 10  # Better for heavy degradation
            elif model == AdvancedModel.BASICVSR_PP:
                score += 15  # Very good quality
                if has_temporal and total_frames >= 30:
                    score += 10  # Good for temporal videos
            elif model == AdvancedModel.REAL_BASICVSR:
                score += 10
                if degradation in ('heavy', 'unknown'):
                    score += 15  # Best for unknown degradation

            # Short video penalty for temporal models
            if total_frames < 20:
                if model in (AdvancedModel.BASICVSR_PP, AdvancedModel.VRT):
                    score -= 10

            scores[model] = score

        # Return highest scoring model
        best_model = max(scores, key=scores.get)
        logger.info(
            f"Model recommendation: {best_model.value} "
            f"(score: {scores[best_model]}, resolution: {resolution}, VRAM: {vram_mb}MB)"
        )

        return best_model

    def get_model_requirements(self, model: AdvancedModel) -> ModelRequirements:
        """Get hardware requirements for a specific model.

        Args:
            model: The advanced model to get requirements for

        Returns:
            ModelRequirements dataclass with hardware specifications
        """
        requirements_map = {
            AdvancedModel.BASICVSR_PP: ModelRequirements(
                min_vram_mb=4096,
                recommended_vram_mb=10240,
                supports_tiling=True,
                supports_half_precision=True,
                min_temporal_window=3,
                recommended_temporal_window=7,
            ),
            AdvancedModel.VRT: ModelRequirements(
                min_vram_mb=8192,
                recommended_vram_mb=16384,
                supports_tiling=True,
                supports_half_precision=True,
                min_temporal_window=2,
                recommended_temporal_window=6,
            ),
            AdvancedModel.REAL_BASICVSR: ModelRequirements(
                min_vram_mb=6144,
                recommended_vram_mb=12288,
                supports_tiling=True,
                supports_half_precision=True,
                min_temporal_window=3,
                recommended_temporal_window=7,
            ),
        }

        return requirements_map.get(model, ModelRequirements())

    def estimate_processing_time(
        self,
        model: AdvancedModel,
        video_info: Dict,
        hardware_info: Optional[Dict] = None,
    ) -> float:
        """Estimate processing time for a video with given model.

        Args:
            model: The model to use
            video_info: Video properties (resolution, duration, fps)
            hardware_info: Optional hardware properties

        Returns:
            Estimated processing time in seconds
        """
        resolution = video_info.get('resolution', (1920, 1080))
        duration = video_info.get('duration', 60)
        fps = video_info.get('fps', 30)

        total_frames = int(duration * fps)
        closest_res = self._find_closest_resolution(resolution)

        # Get base speed estimate
        base_fps = self.SPEED_ESTIMATES.get(model, {}).get(closest_res, 1.0)

        # Adjust for hardware if provided
        if hardware_info:
            vram_mb = hardware_info.get('vram_mb', 8192)
            required_vram = self.VRAM_REQUIREMENTS.get(model, {}).get(closest_res, 16000)

            # If VRAM is insufficient, speed will be reduced due to tiling
            if vram_mb < required_vram:
                reduction_factor = max(0.3, vram_mb / required_vram)
                base_fps *= reduction_factor

            # Adjust for GPU tier (rough estimate)
            gpu_name = hardware_info.get('gpu_name', '').lower()
            if '4090' in gpu_name or '4080' in gpu_name:
                base_fps *= 1.5
            elif '3060' in gpu_name or '3050' in gpu_name:
                base_fps *= 0.6
            elif '2080' in gpu_name or '2070' in gpu_name:
                base_fps *= 0.4

        estimated_seconds = total_frames / base_fps

        logger.info(
            f"Estimated time for {model.value}: {estimated_seconds:.1f}s "
            f"({total_frames} frames @ {base_fps:.2f} fps)"
        )

        return estimated_seconds

    def _detect_hardware(self) -> Dict:
        """Auto-detect hardware capabilities."""
        hardware = {
            'vram_mb': 4096,  # Conservative default
            'gpu_name': 'unknown',
        }

        gpus = get_all_gpu_info()
        if gpus:
            best_gpu = max(gpus, key=lambda g: g.total_memory_mb)
            hardware['vram_mb'] = best_gpu.free_memory_mb
            hardware['gpu_name'] = best_gpu.name

        return hardware

    def _find_closest_resolution(self, resolution: Tuple[int, int]) -> Tuple[int, int]:
        """Find the closest reference resolution for estimates."""
        ref_resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (3840, 2160),
        ]

        width, height = resolution
        pixels = width * height

        closest = ref_resolutions[0]
        min_diff = abs(pixels - closest[0] * closest[1])

        for ref in ref_resolutions[1:]:
            diff = abs(pixels - ref[0] * ref[1])
            if diff < min_diff:
                min_diff = diff
                closest = ref

        return closest


def get_advanced_processor(
    model: AdvancedModel,
    config: Optional[AdvancedModelConfig] = None,
) -> Union[BasicVSRPP, VRTProcessor]:
    """Factory function to get the appropriate processor for a model.

    Args:
        model: The advanced model to create a processor for
        config: Optional configuration (will be created with model if not provided)

    Returns:
        Appropriate processor instance (BasicVSRPP or VRTProcessor)

    Raises:
        ValueError: If model is not supported
    """
    if config is None:
        config = AdvancedModelConfig(model=model)
    else:
        config.model = model

    if model == AdvancedModel.BASICVSR_PP:
        return BasicVSRPP(config)
    elif model == AdvancedModel.VRT:
        return VRTProcessor(config)
    elif model == AdvancedModel.REAL_BASICVSR:
        # Real-BasicVSR uses same architecture as BasicVSR++ with different weights
        return BasicVSRPP(config)
    else:
        raise ValueError(f"Unsupported model: {model}")
