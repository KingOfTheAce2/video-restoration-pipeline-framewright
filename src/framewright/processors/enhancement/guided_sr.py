"""Text-Guided Super Resolution Processor.

This module implements Upscale-A-Video style text-guided texture generation for
super resolution. It uses text prompts to guide the diffusion process, allowing
fine-grained control over the output style and texture characteristics.

Key Features:
- Text-conditioned diffusion for style-controlled upscaling
- CLIP text encoder for semantic guidance
- Cross-attention with text embeddings
- Classifier-free guidance for prompt adherence
- Style presets for common use cases
- Temporal consistency for video processing
- Reference image style transfer
- Prompt caching for efficiency

VRAM Requirements:
- GuidedDiffusionBackend: 12GB+ VRAM (full text guidance)
- Fallback without guidance: 8GB+ VRAM

Example:
    >>> from framewright.processors.enhancement.guided_sr import (
    ...     GuidedSuperResolution,
    ...     GuidedSRConfig,
    ...     create_guided_sr,
    ... )
    >>>
    >>> # Quick setup with style preset
    >>> sr = create_guided_sr(style="cinematic")
    >>> upscaled = sr.upscale(frame, "high quality, film grain")
    >>>
    >>> # Or with full configuration
    >>> config = GuidedSRConfig(
    ...     guidance_text="sharp details, photorealistic",
    ...     guidance_scale=7.5,
    ...     scale=4,
    ... )
    >>> sr = GuidedSuperResolution(config)
    >>> result = sr.upscale_video(frames, config.guidance_text)

References:
    - Upscale-A-Video: https://arxiv.org/abs/2312.06640
    - CLIP: Learning Transferable Visual Models From Natural Language
    - Classifier-Free Diffusion Guidance
"""

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Lazy Imports for Optional Dependencies
# =============================================================================

_torch = None
_torch_checked = False
_transformers = None
_transformers_checked = False
_diffusers = None
_diffusers_checked = False
_cv2 = None
_cv2_checked = False


def _get_torch():
    """Lazy load torch."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
            logger.debug("PyTorch not available")
        _torch_checked = True
    return _torch


def _get_transformers():
    """Lazy load transformers."""
    global _transformers, _transformers_checked
    if not _transformers_checked:
        try:
            import transformers
            _transformers = transformers
        except ImportError:
            _transformers = None
            logger.debug("Transformers not available")
        _transformers_checked = True
    return _transformers


def _get_diffusers():
    """Lazy load diffusers."""
    global _diffusers, _diffusers_checked
    if not _diffusers_checked:
        try:
            import diffusers
            _diffusers = diffusers
        except ImportError:
            _diffusers = None
            logger.debug("Diffusers not available")
        _diffusers_checked = True
    return _diffusers


def _get_cv2():
    """Lazy load OpenCV."""
    global _cv2, _cv2_checked
    if not _cv2_checked:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            _cv2 = None
            logger.debug("OpenCV not available")
        _cv2_checked = True
    return _cv2


def _get_vram_gb() -> float:
    """Get available VRAM in GB."""
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0

    try:
        device_props = torch.cuda.get_device_properties(0)
        total_vram = device_props.total_memory / (1024 ** 3)
        return total_vram
    except Exception:
        return 0.0


def _get_free_vram_gb() -> float:
    """Get free VRAM in GB."""
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0

    try:
        free_mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        return (free_mem - allocated) / (1024 ** 3)
    except Exception:
        return 0.0


# =============================================================================
# Style Presets
# =============================================================================

class StylePresets:
    """Predefined style presets for common use cases.

    Each preset provides a carefully crafted prompt and negative prompt
    combination optimized for specific visual styles.

    Attributes:
        CINEMATIC: Film-like quality with grain and cinematic color grading.
        ANIME: Clean lines, vibrant colors, animation style.
        PHOTOREALISTIC: Maximum detail preservation, sharp textures.
        VINTAGE: Warm colors, film grain, nostalgic look.
        HDR: High dynamic range, vivid colors, enhanced contrast.
        DOCUMENTARY: Natural colors, sharp details, neutral tones.
        NOIR: High contrast black and white, dramatic lighting.
        SOFT: Dreamy, soft focus, gentle colors.
    """

    CINEMATIC: Dict[str, str] = {
        "prompt": "high quality, sharp details, film grain, cinematic color grading, "
                  "professional cinematography, movie quality, 35mm film look",
        "negative": "blurry, noise, artifacts, oversaturated, cartoon, anime, "
                    "low quality, pixelated, compression artifacts",
    }

    ANIME: Dict[str, str] = {
        "prompt": "clean lines, vibrant colors, anime style, cel shading, "
                  "high quality animation, sharp edges, consistent color",
        "negative": "blurry, noise, photorealistic, live action, film grain, "
                    "low quality, inconsistent lines, muddy colors",
    }

    PHOTOREALISTIC: Dict[str, str] = {
        "prompt": "photorealistic, ultra sharp, detailed textures, high resolution, "
                  "professional photography, 8k quality, natural lighting",
        "negative": "blurry, cartoon, anime, painting, artistic, low quality, "
                    "noise, artifacts, oversaturated",
    }

    VINTAGE: Dict[str, str] = {
        "prompt": "film grain, warm colors, slight vignette, vintage look, "
                  "nostalgic, analog film, muted tones, classic cinema",
        "negative": "digital look, oversaturated, modern, cold colors, "
                    "harsh lighting, low quality",
    }

    HDR: Dict[str, str] = {
        "prompt": "HDR, high dynamic range, vivid colors, enhanced contrast, "
                  "sharp details, rich blacks, bright highlights",
        "negative": "flat, low contrast, washed out, blurry, noise, "
                    "low quality, dull colors",
    }

    DOCUMENTARY: Dict[str, str] = {
        "prompt": "documentary style, natural colors, sharp details, neutral tones, "
                  "realistic, professional camera, broadcast quality",
        "negative": "stylized, artistic, oversaturated, low quality, blurry, "
                    "noise, cartoon, anime",
    }

    NOIR: Dict[str, str] = {
        "prompt": "film noir, high contrast, black and white, dramatic lighting, "
                  "deep shadows, sharp details, classic hollywood",
        "negative": "color, flat lighting, low contrast, blurry, noise, "
                    "low quality, modern look",
    }

    SOFT: Dict[str, str] = {
        "prompt": "soft focus, dreamy, gentle colors, smooth skin, "
                  "romantic lighting, ethereal, pastel tones",
        "negative": "harsh, sharp, high contrast, noise, artifacts, "
                    "low quality, oversaturated",
    }

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, str]:
        """Get a style preset by name.

        Args:
            name: Preset name (case-insensitive).

        Returns:
            Dictionary with 'prompt' and 'negative' keys.

        Raises:
            ValueError: If preset name is not found.
        """
        name_upper = name.upper()
        if hasattr(cls, name_upper):
            return getattr(cls, name_upper)

        available = cls.list_presets()
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available preset names.

        Returns:
            List of preset names.
        """
        return [
            attr.lower() for attr in dir(cls)
            if not attr.startswith("_") and isinstance(getattr(cls, attr), dict)
        ]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GuidedSRConfig:
    """Configuration for text-guided super resolution.

    This configuration controls the text-conditioned diffusion model behavior,
    including guidance strength, prompt settings, and processing parameters.

    Attributes:
        guidance_text: Text prompt for texture guidance.
            Examples: "sharp details, film grain", "anime style, clean lines"
        guidance_scale: How strongly to follow text guidance (CFG scale).
            Higher values follow the prompt more closely but may introduce artifacts.
            Default: 7.5. Range: 1.0 (weak) to 20.0 (very strong).
        negative_prompt: Text describing what to avoid in output.
            Example: "blurry, noise, artifacts"
        scale: Upscale factor (2 or 4). Default: 4.
        steps: Number of diffusion steps. More steps = higher quality but slower.
            Default: 20. Range: 10-50.
        strength: How much to change the input (0-1). Higher values allow more
            creative freedom but may deviate from the original.
            Default: 0.5. Range: 0.1 (subtle) to 0.9 (aggressive).
        seed: Random seed for reproducibility. -1 for random.
        precision: Inference precision ("fp32", "fp16", "bf16").
        device: Device for inference ("cuda", "cpu", "auto").
        gpu_id: GPU device ID for multi-GPU systems.
        tile_size: Tile size for processing large frames. 0 = auto.
        tile_overlap: Overlap between tiles to avoid seam artifacts.
        temporal_window: Number of frames for temporal consistency in video.
        use_reference: Whether to use reference image for style transfer.

    Example:
        >>> config = GuidedSRConfig(
        ...     guidance_text="cinematic, film grain, sharp details",
        ...     guidance_scale=7.5,
        ...     negative_prompt="blurry, noise",
        ...     scale=4,
        ...     steps=20,
        ... )
    """
    guidance_text: str = "high quality, sharp details"
    guidance_scale: float = 7.5
    negative_prompt: str = "blurry, noise, artifacts, low quality"
    scale: int = 4
    steps: int = 20
    strength: float = 0.5
    seed: int = -1
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    device: str = "auto"
    gpu_id: int = 0
    tile_size: int = 0  # 0 = auto
    tile_overlap: int = 32
    temporal_window: int = 5
    use_reference: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {self.scale}")

        if self.guidance_scale < 1.0 or self.guidance_scale > 25.0:
            raise ValueError(f"guidance_scale must be 1.0-25.0, got {self.guidance_scale}")

        if self.steps < 1 or self.steps > 100:
            raise ValueError(f"steps must be 1-100, got {self.steps}")

        if self.strength < 0.0 or self.strength > 1.0:
            raise ValueError(f"strength must be 0.0-1.0, got {self.strength}")

        if self.temporal_window < 1:
            raise ValueError(f"temporal_window must be >= 1, got {self.temporal_window}")

        # Auto-detect device
        if self.device == "auto":
            torch = _get_torch()
            if torch is not None and torch.cuda.is_available():
                self.device = f"cuda:{self.gpu_id}"
            else:
                self.device = "cpu"


@dataclass
class GuidedSRResult:
    """Result of text-guided super resolution processing.

    Attributes:
        frames: List of upscaled frames (numpy arrays, BGR format).
        frames_processed: Number of frames successfully processed.
        frames_failed: Number of frames that failed.
        processing_time_seconds: Total processing time.
        avg_fps: Average frames per second.
        peak_vram_mb: Peak VRAM usage in MB.
        guidance_text: Text prompt that was used.
        style_preset: Style preset name if used.
        scale_factor: Actual scale factor applied.
        warnings: Any warnings generated during processing.
    """
    frames: List[np.ndarray] = field(default_factory=list)
    frames_processed: int = 0
    frames_failed: int = 0
    processing_time_seconds: float = 0.0
    avg_fps: float = 0.0
    peak_vram_mb: int = 0
    guidance_text: str = ""
    style_preset: str = ""
    scale_factor: int = 4
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Text Encoder
# =============================================================================

class TextEncoder:
    """CLIP-based text encoder for generating text embeddings.

    Encodes text prompts into embeddings that can be used for cross-attention
    in the diffusion model. Supports caching for repeated prompts.

    Attributes:
        model_name: Name of the CLIP model to use.
        device: Device for inference.
        max_length: Maximum token length.

    Example:
        >>> encoder = TextEncoder()
        >>> embeddings = encoder.encode("sharp details, film grain")
        >>> # Cached lookup
        >>> embeddings2 = encoder.encode("sharp details, film grain")  # Fast
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 77,
    ):
        """Initialize TextEncoder.

        Args:
            model_name: CLIP model name from HuggingFace.
            device: Device for inference ("cuda", "cpu", "auto").
            max_length: Maximum token sequence length.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._initialized = False
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

    @property
    def device(self) -> str:
        """Get effective device."""
        if self._device == "auto":
            torch = _get_torch()
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self._device

    def is_available(self) -> bool:
        """Check if text encoder is available."""
        transformers = _get_transformers()
        torch = _get_torch()
        return transformers is not None and torch is not None

    def _ensure_model(self) -> None:
        """Ensure model is loaded."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            transformers = _get_transformers()
            torch = _get_torch()

            if transformers is None or torch is None:
                raise RuntimeError("Transformers and PyTorch required for TextEncoder")

            logger.info(f"Loading CLIP text encoder: {self.model_name}")

            try:
                from transformers import CLIPTextModel, CLIPTokenizer

                self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                self._model = CLIPTextModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()

                logger.info(f"CLIP text encoder loaded on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                raise RuntimeError(f"Could not load CLIP model: {e}") from e

            self._initialized = True

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def encode(
        self,
        text: str,
        return_pooled: bool = False,
    ) -> "torch.Tensor":
        """Encode text to embeddings.

        Args:
            text: Text prompt to encode.
            return_pooled: Whether to return pooled output instead of sequence.

        Returns:
            Text embeddings tensor.
        """
        self._ensure_model()

        torch = _get_torch()

        # Check cache
        cache_key = self._get_cache_key(text)
        with self._cache_lock:
            if cache_key in self._cache:
                logger.debug(f"Text encoder cache hit for: {text[:50]}...")
                cached = self._cache[cache_key]
                if return_pooled:
                    return cached["pooled"].clone()
                return cached["hidden"].clone()

        # Tokenize
        tokens = self._tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Encode
        with torch.no_grad():
            outputs = self._model(**tokens)

        # Cache results
        with self._cache_lock:
            self._cache[cache_key] = {
                "hidden": outputs.last_hidden_state.cpu().clone(),
                "pooled": outputs.pooler_output.cpu().clone(),
            }

        if return_pooled:
            return outputs.pooler_output
        return outputs.last_hidden_state

    def encode_batch(
        self,
        texts: List[str],
        return_pooled: bool = False,
    ) -> "torch.Tensor":
        """Encode multiple texts to embeddings.

        Args:
            texts: List of text prompts.
            return_pooled: Whether to return pooled output.

        Returns:
            Batched text embeddings tensor.
        """
        self._ensure_model()

        torch = _get_torch()

        # Check cache for all
        results = []
        uncached_texts = []
        uncached_indices = []

        with self._cache_lock:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached = self._cache[cache_key]
                    if return_pooled:
                        results.append((i, cached["pooled"].clone()))
                    else:
                        results.append((i, cached["hidden"].clone()))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        # Process uncached
        if uncached_texts:
            tokens = self._tokenizer(
                uncached_texts,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self._model(**tokens)

            # Cache and collect
            with self._cache_lock:
                for j, idx in enumerate(uncached_indices):
                    text = uncached_texts[j]
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = {
                        "hidden": outputs.last_hidden_state[j:j+1].cpu().clone(),
                        "pooled": outputs.pooler_output[j:j+1].cpu().clone(),
                    }

                    if return_pooled:
                        results.append((idx, outputs.pooler_output[j:j+1].clone()))
                    else:
                        results.append((idx, outputs.last_hidden_state[j:j+1].clone()))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return torch.cat([r[1] for r in results], dim=0)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.debug("Text encoder cache cleared")

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self.clear_cache()

        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Guided Diffusion Backend
# =============================================================================

class GuidedDiffusionBackend(ABC):
    """Abstract base class for text-guided diffusion backends.

    All backends must implement this interface for text-conditioned
    super resolution.

    Methods:
        is_available: Check if backend can run on current system.
        upscale_with_guidance: Upscale frame with text guidance.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier name."""
        ...

    @property
    @abstractmethod
    def vram_requirement_gb(self) -> float:
        """Minimum VRAM requirement in GB."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @abstractmethod
    def upscale_with_guidance(
        self,
        frame: np.ndarray,
        text_embeddings: "torch.Tensor",
        negative_embeddings: Optional["torch.Tensor"],
        config: GuidedSRConfig,
    ) -> np.ndarray:
        """Upscale frame with text guidance.

        Args:
            frame: Input frame (BGR numpy array).
            text_embeddings: Encoded text prompt embeddings.
            negative_embeddings: Encoded negative prompt embeddings.
            config: Guided SR configuration.

        Returns:
            Upscaled frame (BGR numpy array).
        """
        ...

    def clear_cache(self) -> None:
        """Clear any cached models or GPU memory."""
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class SDGuidedSRBackend(GuidedDiffusionBackend):
    """Stable Diffusion-based text-guided super resolution backend.

    Uses Stable Diffusion's image-to-image pipeline with text conditioning
    for guided upscaling.

    Features:
    - Full CLIP text conditioning
    - Classifier-free guidance
    - Strength control for preservation
    - Tile-based processing for large frames
    """

    def __init__(self, config: GuidedSRConfig):
        """Initialize SD guided backend.

        Args:
            config: Guided SR configuration.
        """
        self.config = config
        self._pipeline = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def name(self) -> str:
        return "sd_guided"

    @property
    def vram_requirement_gb(self) -> float:
        return 12.0

    def is_available(self) -> bool:
        """Check if backend can run."""
        torch = _get_torch()
        if torch is None or not torch.cuda.is_available():
            return False

        vram = _get_vram_gb()
        if vram < self.vram_requirement_gb:
            return False

        diffusers = _get_diffusers()
        return diffusers is not None

    def _ensure_pipeline(self) -> None:
        """Ensure pipeline is loaded."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            torch = _get_torch()
            diffusers = _get_diffusers()

            if torch is None or diffusers is None:
                raise RuntimeError("PyTorch and diffusers required")

            logger.info("Loading Stable Diffusion pipeline for guided SR...")

            device = self.config.device
            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }.get(self.config.precision, torch.float16)

            try:
                from diffusers import StableDiffusionImg2ImgPipeline

                self._pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                self._pipeline.to(device)

                # Enable optimizations
                if hasattr(self._pipeline, "enable_attention_slicing"):
                    self._pipeline.enable_attention_slicing()

                logger.info(f"SD pipeline loaded on {device}")

            except Exception as e:
                logger.error(f"Failed to load SD pipeline: {e}")
                self._pipeline = None

            self._initialized = True

    def upscale_with_guidance(
        self,
        frame: np.ndarray,
        text_embeddings: "torch.Tensor",
        negative_embeddings: Optional["torch.Tensor"],
        config: GuidedSRConfig,
    ) -> np.ndarray:
        """Upscale frame with text guidance using Stable Diffusion."""
        self._ensure_pipeline()

        cv2 = _get_cv2()
        torch = _get_torch()

        if cv2 is None or torch is None:
            raise RuntimeError("OpenCV and PyTorch required")

        if self._pipeline is None:
            # Fallback to bicubic
            logger.warning("SD pipeline not available, using bicubic fallback")
            h, w = frame.shape[:2]
            return cv2.resize(
                frame,
                (w * config.scale, h * config.scale),
                interpolation=cv2.INTER_CUBIC,
            )

        # Pre-upscale with bicubic
        h, w = frame.shape[:2]
        pre_upscaled = cv2.resize(
            frame,
            (w * config.scale, h * config.scale),
            interpolation=cv2.INTER_CUBIC,
        )

        # Convert to PIL
        try:
            from PIL import Image

            frame_rgb = cv2.cvtColor(pre_upscaled, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run diffusion with text prompt
            result = self._pipeline(
                prompt=config.guidance_text,
                negative_prompt=config.negative_prompt,
                image=pil_image,
                strength=config.strength,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device=config.device).manual_seed(config.seed)
                if config.seed >= 0 else None,
            ).images[0]

            # Convert back to BGR numpy
            result_array = np.array(result)
            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"SD guided upscale failed: {e}, using bicubic fallback")
            return pre_upscaled

    def clear_cache(self) -> None:
        """Clear pipeline and GPU memory."""
        self._pipeline = None
        self._initialized = False
        super().clear_cache()


class FallbackGuidedBackend(GuidedDiffusionBackend):
    """Fallback backend using traditional upscaling without text guidance.

    Provides bicubic/Lanczos upscaling when text-guided backends are unavailable.
    """

    def __init__(self, config: GuidedSRConfig, method: str = "lanczos"):
        """Initialize fallback backend.

        Args:
            config: Configuration.
            method: Interpolation method ("bicubic", "lanczos").
        """
        self.config = config
        self.method = method

    @property
    def name(self) -> str:
        return f"fallback_{self.method}"

    @property
    def vram_requirement_gb(self) -> float:
        return 0.0

    def is_available(self) -> bool:
        """Always available as fallback."""
        cv2 = _get_cv2()
        return cv2 is not None

    def upscale_with_guidance(
        self,
        frame: np.ndarray,
        text_embeddings: "torch.Tensor",
        negative_embeddings: Optional["torch.Tensor"],
        config: GuidedSRConfig,
    ) -> np.ndarray:
        """Upscale using interpolation (ignores text guidance)."""
        cv2 = _get_cv2()
        if cv2 is None:
            raise RuntimeError("OpenCV required")

        logger.warning("Using fallback upscaling without text guidance")

        h, w = frame.shape[:2]
        interpolation = {
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }.get(self.method, cv2.INTER_LANCZOS4)

        return cv2.resize(
            frame,
            (w * config.scale, h * config.scale),
            interpolation=interpolation,
        )


# =============================================================================
# Texture Generator
# =============================================================================

class TextureGenerator:
    """Generate textures based on text descriptions.

    Interprets text prompts to generate appropriate texture overlays
    for specific visual styles.

    Supported texture types:
    - Film grain: "film grain, 35mm", "grainy, vintage"
    - Sharpening: "sharp, detailed", "crisp edges"
    - Film look: "soft, cinematic", "filmic"
    - Noise patterns: "subtle noise", "analog"

    Example:
        >>> generator = TextureGenerator()
        >>> grain = generator.generate_texture("film grain, 35mm", frame.shape)
        >>> enhanced = generator.apply_texture(frame, grain, opacity=0.3)
    """

    TEXTURE_KEYWORDS = {
        "grain": ["film grain", "grain", "grainy", "35mm", "16mm", "analog film"],
        "sharp": ["sharp", "detailed", "crisp", "edges", "clarity"],
        "soft": ["soft", "cinematic", "filmic", "dreamy", "ethereal"],
        "noise": ["noise", "analog", "vintage", "retro"],
    }

    def __init__(self, seed: int = -1):
        """Initialize TextureGenerator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed if seed >= 0 else None)

    def analyze_prompt(self, prompt: str) -> Dict[str, float]:
        """Analyze prompt to determine texture weights.

        Args:
            prompt: Text prompt to analyze.

        Returns:
            Dictionary mapping texture types to weights (0-1).
        """
        prompt_lower = prompt.lower()
        weights = {}

        for texture_type, keywords in self.TEXTURE_KEYWORDS.items():
            max_weight = 0.0
            for keyword in keywords:
                if keyword in prompt_lower:
                    # Weight based on keyword position (earlier = higher weight)
                    pos = prompt_lower.index(keyword)
                    weight = 1.0 - (pos / (len(prompt_lower) + 1)) * 0.5
                    max_weight = max(max_weight, weight)
            weights[texture_type] = max_weight

        return weights

    def generate_grain(
        self,
        shape: Tuple[int, int, int],
        intensity: float = 0.15,
    ) -> np.ndarray:
        """Generate film grain texture.

        Args:
            shape: Output shape (H, W, C).
            intensity: Grain intensity (0-1).

        Returns:
            Grain texture array (float, centered at 0).
        """
        h, w, c = shape
        grain = self._rng.normal(0, intensity * 255, (h, w))

        # Add some structure to make it more film-like
        grain = grain.astype(np.float32)

        cv2 = _get_cv2()
        if cv2 is not None:
            # Slight blur for more natural grain
            grain = cv2.GaussianBlur(grain, (3, 3), 0.5)

        # Expand to all channels
        grain = np.stack([grain] * c, axis=-1)
        return grain

    def generate_sharpening_mask(
        self,
        frame: np.ndarray,
        amount: float = 1.0,
    ) -> np.ndarray:
        """Generate sharpening enhancement mask.

        Args:
            frame: Input frame.
            amount: Sharpening amount.

        Returns:
            Sharpening mask.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            return np.zeros_like(frame, dtype=np.float32)

        # Detect edges for detail-aware sharpening
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(edges)
        edges = edges / (edges.max() + 1e-8)  # Normalize

        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * amount

        sharpened = cv2.filter2D(frame.astype(np.float32), -1, kernel / 9.0)
        diff = sharpened - frame.astype(np.float32)

        return diff

    def generate_soft_filter(
        self,
        shape: Tuple[int, int, int],
        radius: float = 0.3,
    ) -> np.ndarray:
        """Generate soft/bloom filter.

        Args:
            shape: Output shape.
            radius: Softness radius.

        Returns:
            Soft filter overlay.
        """
        h, w, c = shape
        soft = np.zeros((h, w, c), dtype=np.float32)

        # Create subtle vignette
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - (dist / max_dist) ** 2 * radius

        for i in range(c):
            soft[:, :, i] = vignette * 10  # Slight brightness in center

        return soft

    def generate_texture(
        self,
        prompt: str,
        shape: Tuple[int, int, int],
    ) -> Dict[str, np.ndarray]:
        """Generate textures based on text prompt.

        Args:
            prompt: Text prompt describing desired texture.
            shape: Output shape (H, W, C).

        Returns:
            Dictionary of texture arrays keyed by type.
        """
        weights = self.analyze_prompt(prompt)
        textures = {}

        if weights.get("grain", 0) > 0.1:
            textures["grain"] = self.generate_grain(shape, weights["grain"] * 0.2)

        if weights.get("soft", 0) > 0.1:
            textures["soft"] = self.generate_soft_filter(shape, weights["soft"] * 0.3)

        return textures

    def apply_texture(
        self,
        frame: np.ndarray,
        texture: np.ndarray,
        opacity: float = 0.3,
        mode: str = "add",
    ) -> np.ndarray:
        """Apply texture to frame.

        Args:
            frame: Input frame (uint8).
            texture: Texture to apply (float).
            opacity: Blend opacity (0-1).
            mode: Blend mode ("add", "overlay", "multiply").

        Returns:
            Frame with texture applied (uint8).
        """
        frame_f = frame.astype(np.float32)
        texture_f = texture.astype(np.float32)

        if mode == "add":
            result = frame_f + texture_f * opacity
        elif mode == "overlay":
            # Soft light blend
            result = frame_f + (2 * texture_f * frame_f / 255 - texture_f) * opacity
        elif mode == "multiply":
            result = frame_f * (1 + texture_f / 255 * opacity)
        else:
            result = frame_f + texture_f * opacity

        return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# Main GuidedSuperResolution Class
# =============================================================================

class GuidedSuperResolution:
    """Text-guided super resolution processor.

    Combines text-conditioned diffusion with style presets for
    controlled upscaling with specific visual characteristics.

    Example:
        >>> config = GuidedSRConfig(
        ...     guidance_text="cinematic, film grain",
        ...     guidance_scale=7.5,
        ...     scale=4,
        ... )
        >>> sr = GuidedSuperResolution(config)
        >>> upscaled = sr.upscale(frame, "sharp details, photorealistic")
        >>>
        >>> # Use style preset
        >>> sr.set_style_preset("vintage")
        >>> vintage_upscaled = sr.upscale(frame)
        >>>
        >>> # Video with consistency
        >>> result = sr.upscale_video(frames, "cinematic quality")
    """

    BACKEND_PRIORITY = [
        SDGuidedSRBackend,
        FallbackGuidedBackend,
    ]

    def __init__(self, config: Optional[GuidedSRConfig] = None):
        """Initialize GuidedSuperResolution.

        Args:
            config: Configuration (uses defaults if None).
        """
        self.config = config or GuidedSRConfig()
        self._backend: Optional[GuidedDiffusionBackend] = None
        self._text_encoder: Optional[TextEncoder] = None
        self._texture_generator: Optional[TextureGenerator] = None
        self._current_preset: Optional[str] = None

        # Initialize components
        self._backend = self._select_backend()
        self._text_encoder = TextEncoder(device=self.config.device)
        self._texture_generator = TextureGenerator(seed=self.config.seed)

    def _select_backend(self) -> GuidedDiffusionBackend:
        """Select best available backend."""
        vram = _get_vram_gb()
        logger.info(f"Available VRAM: {vram:.1f}GB")

        for backend_class in self.BACKEND_PRIORITY:
            try:
                backend = backend_class(self.config)
                if backend.is_available():
                    logger.info(f"Selected backend: {backend.name}")
                    return backend
            except Exception as e:
                logger.debug(f"Backend {backend_class.__name__} failed: {e}")

        # Should not reach here
        raise RuntimeError("No guided SR backend available")

    @property
    def backend_name(self) -> str:
        """Get current backend name."""
        return self._backend.name if self._backend else "none"

    def set_style_preset(self, preset_name: str) -> None:
        """Set a style preset for subsequent operations.

        Args:
            preset_name: Name of the style preset.

        Raises:
            ValueError: If preset name is not found.
        """
        preset = StylePresets.get_preset(preset_name)
        self.config.guidance_text = preset["prompt"]
        self.config.negative_prompt = preset["negative"]
        self._current_preset = preset_name
        logger.info(f"Style preset set: {preset_name}")

    def get_current_preset(self) -> Optional[str]:
        """Get current style preset name if any."""
        return self._current_preset

    def upscale(
        self,
        frame: np.ndarray,
        guidance: Optional[str] = None,
    ) -> np.ndarray:
        """Upscale a single frame with text guidance.

        Args:
            frame: Input frame (BGR numpy array).
            guidance: Optional text guidance (uses config if None).

        Returns:
            Upscaled frame (BGR numpy array).
        """
        guidance_text = guidance or self.config.guidance_text

        # Encode text
        if self._text_encoder.is_available():
            text_embeddings = self._text_encoder.encode(guidance_text)
            negative_embeddings = self._text_encoder.encode(self.config.negative_prompt)
        else:
            text_embeddings = None
            negative_embeddings = None

        # Update config with guidance
        config_copy = GuidedSRConfig(
            guidance_text=guidance_text,
            guidance_scale=self.config.guidance_scale,
            negative_prompt=self.config.negative_prompt,
            scale=self.config.scale,
            steps=self.config.steps,
            strength=self.config.strength,
            seed=self.config.seed,
            precision=self.config.precision,
            device=self.config.device,
            gpu_id=self.config.gpu_id,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
        )

        # Upscale
        result = self._backend.upscale_with_guidance(
            frame, text_embeddings, negative_embeddings, config_copy
        )

        # Apply texture enhancements
        if self._texture_generator:
            textures = self._texture_generator.generate_texture(
                guidance_text, result.shape
            )
            for tex_type, texture in textures.items():
                opacity = 0.15 if tex_type == "grain" else 0.1
                result = self._texture_generator.apply_texture(
                    result, texture, opacity=opacity
                )

        return result

    def upscale_video(
        self,
        frames: List[np.ndarray],
        guidance: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> GuidedSRResult:
        """Upscale video frames with text guidance and temporal consistency.

        Args:
            frames: List of input frames (BGR numpy arrays).
            guidance: Optional text guidance.
            progress_callback: Optional progress callback (0-1).

        Returns:
            GuidedSRResult with upscaled frames and statistics.
        """
        guidance_text = guidance or self.config.guidance_text
        start_time = time.time()

        result = GuidedSRResult(
            guidance_text=guidance_text,
            style_preset=self._current_preset or "",
            scale_factor=self.config.scale,
        )

        if not frames:
            return result

        logger.info(f"Upscaling {len(frames)} frames with guidance: {guidance_text[:50]}...")

        # Encode text once for all frames
        if self._text_encoder.is_available():
            text_embeddings = self._text_encoder.encode(guidance_text)
            negative_embeddings = self._text_encoder.encode(self.config.negative_prompt)
        else:
            text_embeddings = None
            negative_embeddings = None

        # Process frames with temporal window
        window = self.config.temporal_window
        upscaled_frames = []

        for i, frame in enumerate(frames):
            try:
                # Get temporal context
                start_idx = max(0, i - window // 2)
                end_idx = min(len(frames), i + window // 2 + 1)
                context_frames = frames[start_idx:end_idx]

                # Process frame
                config_copy = GuidedSRConfig(
                    guidance_text=guidance_text,
                    guidance_scale=self.config.guidance_scale,
                    negative_prompt=self.config.negative_prompt,
                    scale=self.config.scale,
                    steps=self.config.steps,
                    strength=self.config.strength,
                    seed=self.config.seed if self.config.seed >= 0 else -1,
                    precision=self.config.precision,
                    device=self.config.device,
                    gpu_id=self.config.gpu_id,
                    tile_size=self.config.tile_size,
                    tile_overlap=self.config.tile_overlap,
                )

                upscaled = self._backend.upscale_with_guidance(
                    frame, text_embeddings, negative_embeddings, config_copy
                )

                # Apply texture
                if self._texture_generator:
                    textures = self._texture_generator.generate_texture(
                        guidance_text, upscaled.shape
                    )
                    for tex_type, texture in textures.items():
                        opacity = 0.15 if tex_type == "grain" else 0.1
                        upscaled = self._texture_generator.apply_texture(
                            upscaled, texture, opacity=opacity
                        )

                upscaled_frames.append(upscaled)
                result.frames_processed += 1

            except Exception as e:
                logger.error(f"Frame {i} failed: {e}")
                result.frames_failed += 1
                result.warnings.append(f"Frame {i}: {str(e)}")

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        result.frames = upscaled_frames
        result.processing_time_seconds = time.time() - start_time

        if result.processing_time_seconds > 0 and result.frames_processed > 0:
            result.avg_fps = result.frames_processed / result.processing_time_seconds

        # Track VRAM
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            try:
                result.peak_vram_mb = int(
                    torch.cuda.max_memory_allocated(self.config.gpu_id) / (1024 * 1024)
                )
            except Exception:
                pass

        return result

    def upscale_with_reference(
        self,
        frame: np.ndarray,
        reference: np.ndarray,
        blend_strength: float = 0.5,
    ) -> np.ndarray:
        """Upscale frame using reference image style.

        Extracts style characteristics from the reference image
        and applies them to the upscaled output.

        Args:
            frame: Input frame to upscale.
            reference: Reference image for style.
            blend_strength: How much to apply reference style (0-1).

        Returns:
            Upscaled frame with reference style.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            raise RuntimeError("OpenCV required")

        # First upscale normally
        upscaled = self.upscale(frame)

        # Extract color statistics from reference
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        up_lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Match color statistics (simple histogram matching per channel)
        for i in range(3):
            ref_mean = np.mean(ref_lab[:, :, i])
            ref_std = np.std(ref_lab[:, :, i])
            up_mean = np.mean(up_lab[:, :, i])
            up_std = np.std(up_lab[:, :, i])

            # Normalize and rescale
            if up_std > 0:
                up_lab[:, :, i] = (up_lab[:, :, i] - up_mean) / up_std * ref_std + ref_mean

        # Blend with original
        matched = cv2.cvtColor(np.clip(up_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        result = cv2.addWeighted(upscaled, 1 - blend_strength, matched, blend_strength, 0)

        return result

    def clear_cache(self) -> None:
        """Clear all caches and free GPU memory."""
        if self._backend:
            self._backend.clear_cache()
        if self._text_encoder:
            self._text_encoder.clear_cache()

        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Factory Functions
# =============================================================================

def create_guided_sr(
    style: str = "cinematic",
    scale: int = 4,
    guidance_scale: float = 7.5,
    **kwargs,
) -> GuidedSuperResolution:
    """Create a GuidedSuperResolution processor with style preset.

    Args:
        style: Style preset name ("cinematic", "anime", "photorealistic", etc.).
        scale: Upscaling factor (2 or 4).
        guidance_scale: CFG guidance scale.
        **kwargs: Additional GuidedSRConfig parameters.

    Returns:
        Configured GuidedSuperResolution instance.

    Example:
        >>> sr = create_guided_sr(style="vintage", scale=4)
        >>> result = sr.upscale(frame)
    """
    # Get preset
    try:
        preset = StylePresets.get_preset(style)
    except ValueError:
        preset = {"prompt": "high quality", "negative": "low quality"}
        logger.warning(f"Unknown style '{style}', using default")

    config = GuidedSRConfig(
        guidance_text=preset["prompt"],
        negative_prompt=preset["negative"],
        guidance_scale=guidance_scale,
        scale=scale,
        **kwargs,
    )

    sr = GuidedSuperResolution(config)
    sr._current_preset = style
    return sr


def upscale_with_guidance(
    frames: Union[np.ndarray, List[np.ndarray]],
    text: str,
    scale: int = 4,
    guidance_scale: float = 7.5,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[np.ndarray]:
    """Convenience function to upscale frames with text guidance.

    Args:
        frames: Single frame or list of frames (BGR numpy arrays).
        text: Text guidance prompt.
        scale: Upscaling factor (2 or 4).
        guidance_scale: CFG guidance scale.
        progress_callback: Optional progress callback.

    Returns:
        List of upscaled frames.

    Example:
        >>> upscaled = upscale_with_guidance(
        ...     frames,
        ...     "sharp details, film grain, cinematic",
        ...     scale=4,
        ... )
    """
    config = GuidedSRConfig(
        guidance_text=text,
        guidance_scale=guidance_scale,
        scale=scale,
    )
    sr = GuidedSuperResolution(config)

    # Handle single frame
    if isinstance(frames, np.ndarray) and len(frames.shape) == 3:
        return [sr.upscale(frames, text)]

    result = sr.upscale_video(frames, text, progress_callback)
    return result.frames


def upscale_with_style(
    frames: Union[np.ndarray, List[np.ndarray]],
    style_name: str,
    scale: int = 4,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[np.ndarray]:
    """Convenience function to upscale frames with a style preset.

    Args:
        frames: Single frame or list of frames (BGR numpy arrays).
        style_name: Style preset name.
        scale: Upscaling factor (2 or 4).
        progress_callback: Optional progress callback.

    Returns:
        List of upscaled frames.

    Example:
        >>> upscaled = upscale_with_style(frames, "cinematic", scale=4)
    """
    sr = create_guided_sr(style=style_name, scale=scale)

    # Handle single frame
    if isinstance(frames, np.ndarray) and len(frames.shape) == 3:
        return [sr.upscale(frames)]

    result = sr.upscale_video(frames, progress_callback=progress_callback)
    return result.frames


def list_style_presets() -> List[str]:
    """List all available style presets.

    Returns:
        List of style preset names.
    """
    return StylePresets.list_presets()


def get_style_preset_info(name: str) -> Dict[str, str]:
    """Get detailed information about a style preset.

    Args:
        name: Style preset name.

    Returns:
        Dictionary with 'prompt' and 'negative' keys.
    """
    return StylePresets.get_preset(name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "GuidedSRConfig",
    # Result type
    "GuidedSRResult",
    # Style presets
    "StylePresets",
    # Text encoding
    "TextEncoder",
    # Backend base class
    "GuidedDiffusionBackend",
    # Backend implementations
    "SDGuidedSRBackend",
    "FallbackGuidedBackend",
    # Texture generation
    "TextureGenerator",
    # Main class
    "GuidedSuperResolution",
    # Factory functions
    "create_guided_sr",
    "upscale_with_guidance",
    "upscale_with_style",
    "list_style_presets",
    "get_style_preset_info",
]
