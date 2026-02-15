"""Ensemble Super-Resolution for maximum quality video restoration.

This module provides ensemble-based super-resolution that combines outputs
from multiple SR models to achieve quality beyond any single model.

Ensemble strategies:
- Weighted average: Blend outputs based on model quality weights
- Per-region voting: Use different models for different image regions
- Quality-based selection: Pick best output per-patch using quality metrics
- Adaptive: Automatically select strategy based on content

Supported models for ensemble:
- HAT (Hybrid Attention Transformer) - highest quality
- VRT (Video Restoration Transformer) - excellent temporal consistency
- Real-ESRGAN - fast, good general quality
- BasicVSR++ - temporal-aware, smooth output
- Diffusion SR (Upscale-A-Video) - photorealistic details

Example:
    >>> config = EnsembleConfig(
    ...     models=["hat", "vrt", "realesrgan"],
    ...     voting_method="weighted",
    ... )
    >>> ensemble = EnsembleSR(config)
    >>> result = ensemble.upscale_frames(input_dir, output_dir)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VotingMethod(Enum):
    """Ensemble voting/combination methods."""
    WEIGHTED = "weighted"           # Weighted average based on model quality
    MAX_QUALITY = "max_quality"     # Select best output per-frame using SSIM/PSNR
    PER_REGION = "per_region"       # Different models for different regions
    ADAPTIVE = "adaptive"           # Auto-select based on content analysis
    MEDIAN = "median"               # Pixel-wise median (noise-robust)


class SRModel(Enum):
    """Available SR models for ensemble."""
    HAT = "hat"
    VRT = "vrt"
    REALESRGAN = "realesrgan"
    BASICVSR_PP = "basicvsr_pp"
    DIFFUSION = "diffusion"


# Default quality weights for models (higher = better quality)
DEFAULT_MODEL_WEIGHTS = {
    SRModel.HAT: 1.0,
    SRModel.VRT: 0.95,
    SRModel.DIFFUSION: 0.9,
    SRModel.BASICVSR_PP: 0.8,
    SRModel.REALESRGAN: 0.7,
}


@dataclass
class EnsembleConfig:
    """Configuration for ensemble super-resolution.

    Attributes:
        models: List of models to use in ensemble
        voting_method: Method for combining outputs
        model_weights: Custom weights for each model (optional)
        quality_metric: Metric for quality-based voting (ssim, psnr, lpips)
        patch_size: Patch size for per-region voting
        parallel_inference: Run models in parallel (uses more VRAM)
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
    """
    models: List[str] = field(default_factory=lambda: ["realesrgan"])  # HAT has download issues, using Real-ESRGAN
    voting_method: VotingMethod = VotingMethod.WEIGHTED
    model_weights: Optional[Dict[str, float]] = None
    quality_metric: str = "ssim"
    patch_size: int = 128
    parallel_inference: bool = False
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.voting_method, str):
            self.voting_method = VotingMethod(self.voting_method)
        if len(self.models) < 2:
            logger.warning("Ensemble with single model - using direct pass-through")


@dataclass
class EnsembleResult:
    """Result of ensemble upscaling.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        output_dir: Path to output directory
        processing_time_seconds: Total processing time
        models_used: List of models that were used
        model_contributions: Per-model contribution percentages
    """
    frames_processed: int = 0
    frames_failed: int = 0
    output_dir: Optional[Path] = None
    processing_time_seconds: float = 0.0
    models_used: List[str] = field(default_factory=list)
    model_contributions: Dict[str, float] = field(default_factory=dict)


class ModelProcessor:
    """Wrapper for individual SR model processing."""

    def __init__(self, model_name: str, gpu_id: int = 0):
        """Initialize model processor.

        Args:
            model_name: Name of the model
            gpu_id: GPU device ID
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self._processor = None
        self._loaded = False

    def load(self) -> bool:
        """Load the model.

        Returns:
            True if model loaded successfully
        """
        if self._loaded:
            return True

        try:
            if self.model_name == "hat":
                from .hat_upscaler import HATUpscaler, HATConfig
                config = HATConfig(gpu_id=self.gpu_id, tile_size=0)
                self._processor = HATUpscaler(config)
            elif self.model_name == "vrt":
                from .advanced_models import VRTProcessor, AdvancedModelConfig, AdvancedModel
                config = AdvancedModelConfig(model=AdvancedModel.VRT, gpu_id=self.gpu_id)
                self._processor = VRTProcessor(config)
            elif self.model_name == "realesrgan":
                from .pytorch_realesrgan import get_upsampler, PyTorchESRGANConfig, enhance_frame_pytorch
                config = PyTorchESRGANConfig(model_name="RealESRGAN_x4plus", gpu_id=self.gpu_id)
                # Store the config and functions for later use
                self._esrgan_config = config
                self._esrgan_upsampler = None  # Lazy load
                self._processor = self  # Use self as processor
            elif self.model_name == "basicvsr_pp":
                from .advanced_models import BasicVSRPP, AdvancedModelConfig, AdvancedModel
                config = AdvancedModelConfig(model=AdvancedModel.BASICVSR_PP, gpu_id=self.gpu_id)
                self._processor = BasicVSRPP(config)
            elif self.model_name == "diffusion":
                from .diffusion_sr import DiffusionSR
                self._processor = DiffusionSR(gpu_id=self.gpu_id)
            else:
                logger.error(f"Unknown model: {self.model_name}")
                return False

            self._loaded = self._processor.is_available() if hasattr(self._processor, 'is_available') else True
            return self._loaded

        except ImportError as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Upscaled frame or None if failed
        """
        if not self._loaded:
            if not self.load():
                return None

        try:
            # Special handling for Real-ESRGAN (functional API)
            if self.model_name == "realesrgan":
                from .pytorch_realesrgan import get_upsampler, enhance_frame_pytorch
                if not hasattr(self, '_esrgan_upsampler') or self._esrgan_upsampler is None:
                    self._esrgan_upsampler = get_upsampler(self._esrgan_config)
                return enhance_frame_pytorch(frame, self._esrgan_upsampler, self._esrgan_config)

            # Standard class-based processors
            if hasattr(self._processor, 'upscale_frame'):
                return self._processor.upscale_frame(frame)
            elif hasattr(self._processor, 'enhance_frame'):
                return self._processor.enhance_frame(frame)
            elif hasattr(self._processor, 'process'):
                return self._processor.process(frame)
            else:
                logger.error(f"No processing method found for {self.model_name}")
                return None
        except Exception as e:
            logger.error(f"Error processing with {self.model_name}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear model from GPU memory."""
        if self._processor and hasattr(self._processor, 'clear_cache'):
            self._processor.clear_cache()
        self._loaded = False


class QualityMetrics:
    """Image quality metrics for ensemble voting."""

    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            SSIM score (0-1)
        """
        if not HAS_OPENCV:
            return 0.5

        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2

        # Constants for SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)

        # Means
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variances
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            PSNR in dB
        """
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def estimate_sharpness(img: np.ndarray) -> float:
        """Estimate image sharpness using Laplacian variance.

        Args:
            img: Input image

        Returns:
            Sharpness score (higher = sharper)
        """
        if not HAS_OPENCV:
            return 0.0

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def estimate_noise(img: np.ndarray) -> float:
        """Estimate image noise level.

        Args:
            img: Input image

        Returns:
            Noise estimate (lower = cleaner)
        """
        if not HAS_OPENCV:
            return 0.0

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # High-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float64) - blurred.astype(np.float64)

        return float(np.std(noise))


class EnsembleSR:
    """Ensemble super-resolution processor.

    Combines multiple SR models to achieve quality beyond any single model.

    Example:
        >>> config = EnsembleConfig(
        ...     models=["hat", "vrt", "realesrgan"],
        ...     voting_method="weighted"
        ... )
        >>> ensemble = EnsembleSR(config)
        >>> result = ensemble.upscale_frames(input_dir, output_dir)
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble processor.

        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self._processors: Dict[str, ModelProcessor] = {}
        self._weights: Dict[str, float] = {}
        self._metrics = QualityMetrics()
        self._init_processors()

    def _init_processors(self) -> None:
        """Initialize model processors."""
        for model_name in self.config.models:
            self._processors[model_name] = ModelProcessor(
                model_name, self.config.gpu_id
            )

            # Set weights
            if self.config.model_weights and model_name in self.config.model_weights:
                self._weights[model_name] = self.config.model_weights[model_name]
            else:
                # Use default weights
                try:
                    model_enum = SRModel(model_name)
                    self._weights[model_name] = DEFAULT_MODEL_WEIGHTS.get(model_enum, 0.5)
                except ValueError:
                    self._weights[model_name] = 0.5

    def is_available(self) -> bool:
        """Check if ensemble processing is available.

        Returns:
            True if at least 2 models are available
        """
        available = sum(1 for p in self._processors.values() if p.load())
        return available >= 2

    def _combine_weighted(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine outputs using weighted average.

        Args:
            outputs: Dictionary of model name -> output frame

        Returns:
            Combined frame
        """
        # Normalize weights
        total_weight = sum(self._weights.get(m, 1.0) for m in outputs.keys())

        # Accumulate weighted sum
        result = np.zeros_like(list(outputs.values())[0], dtype=np.float64)

        for model_name, output in outputs.items():
            weight = self._weights.get(model_name, 1.0) / total_weight
            result += output.astype(np.float64) * weight

        return result.astype(np.uint8)

    def _combine_max_quality(
        self,
        outputs: Dict[str, np.ndarray],
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str]:
        """Select best output based on quality metrics.

        Args:
            outputs: Dictionary of model name -> output frame
            reference: Optional reference for comparison

        Returns:
            Tuple of (best frame, model name)
        """
        best_score = -float('inf')
        best_model = None
        best_output = None

        for model_name, output in outputs.items():
            # Calculate quality score
            sharpness = self._metrics.estimate_sharpness(output)
            noise = self._metrics.estimate_noise(output)

            # Higher sharpness, lower noise = better
            score = sharpness - noise * 0.5

            if reference is not None:
                # Use SSIM if reference available
                ssim = self._metrics.calculate_ssim(output, reference)
                score = ssim * 100 + sharpness * 0.01

            if score > best_score:
                best_score = score
                best_model = model_name
                best_output = output

        return best_output, best_model

    def _combine_per_region(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine outputs using per-region selection.

        Divides image into patches and selects best model per-patch.

        Args:
            outputs: Dictionary of model name -> output frame

        Returns:
            Combined frame
        """
        if not outputs:
            return None

        h, w = list(outputs.values())[0].shape[:2]
        patch_size = self.config.patch_size
        result = np.zeros((h, w, 3), dtype=np.uint8)

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)

                # Extract patches
                patches = {
                    m: out[y:y_end, x:x_end]
                    for m, out in outputs.items()
                }

                # Select best patch
                best_patch, _ = self._combine_max_quality(patches)
                result[y:y_end, x:x_end] = best_patch

        return result

    def _combine_median(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine outputs using pixel-wise median.

        Args:
            outputs: Dictionary of model name -> output frame

        Returns:
            Median-combined frame
        """
        # Stack all outputs
        stacked = np.stack(list(outputs.values()), axis=0)

        # Compute median
        result = np.median(stacked, axis=0).astype(np.uint8)

        return result

    def _combine_adaptive(
        self,
        outputs: Dict[str, np.ndarray],
        input_frame: np.ndarray,
    ) -> np.ndarray:
        """Adaptively combine outputs based on content.

        Uses different strategies for different image regions:
        - Face regions: Prefer smooth models (BasicVSR++)
        - Texture regions: Prefer sharp models (HAT)
        - Sky/smooth: Prefer denoising models

        Args:
            outputs: Dictionary of model name -> output frame
            input_frame: Original input frame

        Returns:
            Adaptively combined frame
        """
        # For now, fall back to weighted - full implementation would
        # do content analysis
        return self._combine_weighted(outputs)

    def combine_outputs(
        self,
        outputs: Dict[str, np.ndarray],
        input_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine model outputs using configured method.

        Args:
            outputs: Dictionary of model name -> output frame
            input_frame: Optional original input frame

        Returns:
            Combined frame
        """
        if len(outputs) == 0:
            raise ValueError("No outputs to combine")

        if len(outputs) == 1:
            return list(outputs.values())[0]

        if self.config.voting_method == VotingMethod.WEIGHTED:
            return self._combine_weighted(outputs)
        elif self.config.voting_method == VotingMethod.MAX_QUALITY:
            result, _ = self._combine_max_quality(outputs)
            return result
        elif self.config.voting_method == VotingMethod.PER_REGION:
            return self._combine_per_region(outputs)
        elif self.config.voting_method == VotingMethod.MEDIAN:
            return self._combine_median(outputs)
        elif self.config.voting_method == VotingMethod.ADAPTIVE:
            return self._combine_adaptive(outputs, input_frame)
        else:
            return self._combine_weighted(outputs)

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single frame using ensemble.

        Args:
            frame: Input frame (BGR)

        Returns:
            Upscaled frame
        """
        outputs = {}

        for model_name, processor in self._processors.items():
            result = processor.process_frame(frame)
            if result is not None:
                outputs[model_name] = result

        if not outputs:
            raise RuntimeError("All models failed to process frame")

        return self.combine_outputs(outputs, frame)

    def upscale_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> EnsembleResult:
        """Upscale all frames using ensemble.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback

        Returns:
            EnsembleResult with processing statistics
        """
        result = EnsembleResult()
        start_time = time.time()

        if not HAS_OPENCV:
            logger.error("OpenCV required for ensemble processing")
            return result

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        # Get frames
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found")
            return result

        n_frames = len(frames)
        logger.info(
            f"Ensemble upscaling {n_frames} frames with "
            f"{len(self.config.models)} models: {self.config.models}"
        )

        # Track model contributions
        model_usage = {m: 0 for m in self.config.models}

        for i, frame_path in enumerate(frames):
            try:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    result.frames_failed += 1
                    continue

                # Get outputs from all models
                outputs = {}
                for model_name, processor in self._processors.items():
                    model_result = processor.process_frame(frame)
                    if model_result is not None:
                        outputs[model_name] = model_result
                        model_usage[model_name] += 1

                if not outputs:
                    result.frames_failed += 1
                    continue

                # Combine outputs
                combined = self.combine_outputs(outputs, frame)

                # Save result
                output_path = output_dir / frame_path.name
                cv2.imwrite(str(output_path), combined)

                result.frames_processed += 1

            except Exception as e:
                logger.error(f"Failed to process {frame_path}: {e}")
                result.frames_failed += 1

            if progress_callback:
                progress_callback((i + 1) / n_frames)

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time
        result.models_used = [m for m, count in model_usage.items() if count > 0]

        total_usage = sum(model_usage.values())
        if total_usage > 0:
            result.model_contributions = {
                m: count / total_usage
                for m, count in model_usage.items()
            }

        logger.info(
            f"Ensemble upscaling complete: {result.frames_processed}/{n_frames} frames, "
            f"time: {result.processing_time_seconds:.1f}s, "
            f"models: {result.models_used}"
        )

        return result

    def clear_cache(self) -> None:
        """Clear all models from GPU memory."""
        for processor in self._processors.values():
            processor.clear_cache()

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_ensemble_sr(
    models: List[str] = ["hat", "realesrgan"],
    voting_method: str = "weighted",
    gpu_id: int = 0,
) -> EnsembleSR:
    """Factory function to create an ensemble SR processor.

    Args:
        models: List of model names
        voting_method: Voting/combination method
        gpu_id: GPU device ID

    Returns:
        Configured EnsembleSR processor
    """
    config = EnsembleConfig(
        models=models,
        voting_method=VotingMethod(voting_method),
        gpu_id=gpu_id,
    )
    return EnsembleSR(config)
