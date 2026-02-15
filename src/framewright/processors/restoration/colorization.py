"""Unified Colorization Processor with Multiple Backends.

This module provides a unified interface for video colorization that:
- Auto-selects the optimal backend based on hardware tier
- Supports multiple colorization approaches (DeOldify, DDColor, SwinTExCo, Temporal)
- Preserves all existing functionality from individual processors
- Provides graceful fallback chain when backends are unavailable

Backend VRAM Requirements:
- DeOldify: ~2GB VRAM, fast, good for general colorization
- DDColor: ~4GB VRAM, higher quality dual-decoder architecture
- SwinTExCo: ~8GB+ VRAM, reference-based exemplar colorization
- Temporal: Post-processing for temporal consistency across frames

Example:
    >>> from framewright.processors.restoration import Colorizer, ColorizerConfig
    >>> from framewright.infrastructure.gpu import get_hardware_info
    >>>
    >>> hardware = get_hardware_info()
    >>> config = ColorizerConfig(backend="auto")
    >>> colorizer = Colorizer(config, hardware)
    >>>
    >>> if colorizer.is_available():
    ...     result = colorizer.colorize(frames, reference=ref_frame)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.debug("PyTorch not available")


class ColorBackend(Enum):
    """Available colorization backends."""
    DEOLDIFY = "deoldify"       # 2GB VRAM, fast, artistic
    DDCOLOR = "ddcolor"         # 4GB VRAM, better quality
    SWINTEXCO = "swintexco"     # 8GB+ VRAM, reference-based
    TEMPORAL = "temporal"       # Post-processing for consistency
    AUTO = "auto"               # Auto-select based on hardware


@dataclass
class ColorizerConfig:
    """Configuration for unified colorization.

    Attributes:
        backend: Colorization backend to use (auto, deoldify, ddcolor, swintexco, temporal)
        strength: Color saturation strength (0.0 to 1.0)
        temporal_consistency: Apply temporal consistency post-processing
        reference_images: List of reference images for SwinTExCo backend
        skip_colored: Skip frames that are already colored
        color_threshold: Threshold for grayscale detection
        artistic_style: DeOldify style (artistic, stable, video)
        render_factor: DeOldify render factor
        temporal_window: Window size for temporal consistency
        propagation_mode: Temporal propagation direction
        blend_strength: Temporal blend strength
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
    """
    backend: Union[ColorBackend, str] = ColorBackend.AUTO
    strength: float = 1.0
    temporal_consistency: bool = True
    reference_images: List[Path] = field(default_factory=list)
    skip_colored: bool = True
    color_threshold: float = 10.0
    artistic_style: str = "artistic"  # artistic, stable, video
    render_factor: int = 35
    temporal_window: int = 7
    propagation_mode: str = "bidirectional"
    blend_strength: float = 0.6
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        if isinstance(self.backend, str):
            self.backend = ColorBackend(self.backend.lower())
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be 0-1, got {self.strength}")
        if not 0.0 <= self.blend_strength <= 1.0:
            raise ValueError(f"blend_strength must be 0-1, got {self.blend_strength}")


@dataclass
class ColorizationResult:
    """Result of colorization processing.

    Attributes:
        frames_processed: Total frames processed
        frames_colorized: Frames that were colorized
        frames_skipped: Frames skipped (already colored or failed)
        backend_used: Backend that was used
        processing_time_seconds: Total processing time
        temporal_consistency_applied: Whether temporal post-processing was applied
        output_dir: Output directory if processing directory
        peak_vram_mb: Peak VRAM usage
    """
    frames_processed: int = 0
    frames_colorized: int = 0
    frames_skipped: int = 0
    backend_used: str = ""
    processing_time_seconds: float = 0.0
    temporal_consistency_applied: bool = False
    output_dir: Optional[Path] = None
    peak_vram_mb: int = 0


# =============================================================================
# Backend Abstract Base Class
# =============================================================================

class BaseColorBackend(ABC):
    """Abstract base class for colorization backends."""

    def __init__(self, config: ColorizerConfig):
        """Initialize backend.

        Args:
            config: Colorization configuration
        """
        self.config = config
        self._model = None
        self._device = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def min_vram_mb(self) -> int:
        """Minimum VRAM required in MB."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    @abstractmethod
    def colorize_frame(
        self,
        frame: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colorize a single frame.

        Args:
            frame: Input frame (BGR format)
            reference: Optional reference frame for exemplar-based colorization

        Returns:
            Colorized frame (BGR format)
        """
        pass

    def colorize_batch(
        self,
        frames: List[np.ndarray],
        reference: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Colorize a batch of frames.

        Default implementation processes frame by frame. Override for
        batch-optimized processing.

        Args:
            frames: List of input frames
            reference: Optional reference frame
            progress_callback: Progress callback

        Returns:
            List of colorized frames
        """
        results = []
        total = len(frames)
        for i, frame in enumerate(frames):
            colorized = self.colorize_frame(frame, reference)
            results.append(colorized)
            if progress_callback:
                progress_callback((i + 1) / total)
        return results

    def clear_cache(self) -> None:
        """Clear model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_grayscale(self, frame: np.ndarray) -> bool:
        """Check if frame is grayscale.

        Args:
            frame: Input frame (BGR format)

        Returns:
            True if frame appears to be grayscale
        """
        if len(frame.shape) == 2:
            return True
        if frame.shape[2] == 1:
            return True
        if frame.shape[2] < 3:
            return True

        # Compare color channels
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        diff_rg = np.abs(r.astype(np.float32) - g.astype(np.float32)).mean()
        diff_rb = np.abs(r.astype(np.float32) - b.astype(np.float32)).mean()
        diff_gb = np.abs(g.astype(np.float32) - b.astype(np.float32)).mean()
        avg_diff = (diff_rg + diff_rb + diff_gb) / 3.0

        return avg_diff < self.config.color_threshold


# =============================================================================
# DeOldify Backend
# =============================================================================

class DeOldifyBackend(BaseColorBackend):
    """DeOldify colorization backend.

    Wraps the existing Colorizer from colorization.py with DeOldify model.
    Fast colorization with artistic style options, requires ~2GB VRAM.
    """

    @property
    def name(self) -> str:
        return "deoldify"

    @property
    def min_vram_mb(self) -> int:
        return 2048  # 2GB

    def __init__(self, config: ColorizerConfig):
        super().__init__(config)
        self._colorizer = None

    def is_available(self) -> bool:
        """Check if DeOldify is available."""
        try:
            from ..colorization import Colorizer as LegacyColorizer
            from ..colorization import ColorizationConfig, ColorModel, ArtisticStyle

            # Map artistic style
            style_map = {
                "artistic": ArtisticStyle.ARTISTIC,
                "stable": ArtisticStyle.STABLE,
                "video": ArtisticStyle.VIDEO,
            }
            style = style_map.get(
                self.config.artistic_style.lower(),
                ArtisticStyle.ARTISTIC
            )

            legacy_config = ColorizationConfig(
                model=ColorModel.DEOLDIFY,
                strength=self.config.strength,
                artistic_style=style,
                render_factor=self.config.render_factor,
                skip_colored=self.config.skip_colored,
                color_threshold=self.config.color_threshold,
            )

            colorizer = LegacyColorizer(legacy_config)
            return colorizer.is_available()

        except ImportError as e:
            logger.debug(f"DeOldify not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"DeOldify check failed: {e}")
            return False

    def _get_colorizer(self):
        """Get or create the legacy colorizer."""
        if self._colorizer is None:
            from ..colorization import Colorizer as LegacyColorizer
            from ..colorization import ColorizationConfig, ColorModel, ArtisticStyle

            style_map = {
                "artistic": ArtisticStyle.ARTISTIC,
                "stable": ArtisticStyle.STABLE,
                "video": ArtisticStyle.VIDEO,
            }
            style = style_map.get(
                self.config.artistic_style.lower(),
                ArtisticStyle.ARTISTIC
            )

            legacy_config = ColorizationConfig(
                model=ColorModel.DEOLDIFY,
                strength=self.config.strength,
                artistic_style=style,
                render_factor=self.config.render_factor,
                skip_colored=self.config.skip_colored,
                color_threshold=self.config.color_threshold,
            )

            self._colorizer = LegacyColorizer(legacy_config)

        return self._colorizer

    def colorize_frame(
        self,
        frame: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colorize using DeOldify.

        Note: reference is ignored as DeOldify is not exemplar-based.
        """
        try:
            colorizer = self._get_colorizer()
            return colorizer.colorize_frame(frame)
        except Exception as e:
            logger.error(f"DeOldify colorization failed: {e}")
            return frame

    def clear_cache(self) -> None:
        """Clear DeOldify model."""
        if self._colorizer is not None:
            self._colorizer = None
        super().clear_cache()


# =============================================================================
# DDColor Backend
# =============================================================================

class DDColorBackend(BaseColorBackend):
    """DDColor colorization backend.

    Wraps the existing Colorizer from colorization.py with DDColor model.
    Higher quality dual-decoder architecture, requires ~4GB VRAM.
    """

    @property
    def name(self) -> str:
        return "ddcolor"

    @property
    def min_vram_mb(self) -> int:
        return 4096  # 4GB

    def __init__(self, config: ColorizerConfig):
        super().__init__(config)
        self._colorizer = None

    def is_available(self) -> bool:
        """Check if DDColor is available."""
        try:
            from ..colorization import Colorizer as LegacyColorizer
            from ..colorization import ColorizationConfig, ColorModel

            legacy_config = ColorizationConfig(
                model=ColorModel.DDCOLOR,
                strength=self.config.strength,
                skip_colored=self.config.skip_colored,
                color_threshold=self.config.color_threshold,
            )

            colorizer = LegacyColorizer(legacy_config)
            return colorizer.is_available()

        except ImportError as e:
            logger.debug(f"DDColor not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"DDColor check failed: {e}")
            return False

    def _get_colorizer(self):
        """Get or create the legacy colorizer."""
        if self._colorizer is None:
            from ..colorization import Colorizer as LegacyColorizer
            from ..colorization import ColorizationConfig, ColorModel

            legacy_config = ColorizationConfig(
                model=ColorModel.DDCOLOR,
                strength=self.config.strength,
                skip_colored=self.config.skip_colored,
                color_threshold=self.config.color_threshold,
            )

            self._colorizer = LegacyColorizer(legacy_config)

        return self._colorizer

    def colorize_frame(
        self,
        frame: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colorize using DDColor.

        Note: reference is ignored as DDColor is not exemplar-based.
        """
        try:
            colorizer = self._get_colorizer()
            return colorizer.colorize_frame(frame)
        except Exception as e:
            logger.error(f"DDColor colorization failed: {e}")
            return frame

    def clear_cache(self) -> None:
        """Clear DDColor model."""
        if self._colorizer is not None:
            self._colorizer = None
        super().clear_cache()


# =============================================================================
# SwinTExCo Backend
# =============================================================================

class SwinTExCoBackend(BaseColorBackend):
    """SwinTExCo exemplar-based colorization backend.

    Wraps the existing SwinTExCoColorizer for reference-based colorization.
    Uses Swin Transformer with temporal correspondence, requires ~8GB+ VRAM.
    """

    @property
    def name(self) -> str:
        return "swintexco"

    @property
    def min_vram_mb(self) -> int:
        return 8192  # 8GB

    def __init__(self, config: ColorizerConfig):
        super().__init__(config)
        self._colorizer = None

    def is_available(self) -> bool:
        """Check if SwinTExCo is available."""
        try:
            from ..swintexco_colorize import (
                SwinTExCoColorizer,
                ExemplarColorizeConfig,
                ColorPropagationMode,
            )

            prop_mode = ColorPropagationMode(self.config.propagation_mode)

            swintexco_config = ExemplarColorizeConfig(
                reference_images=list(self.config.reference_images),
                temporal_fusion=self.config.temporal_consistency,
                propagation_mode=prop_mode,
                style_strength=self.config.strength,
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
            )

            colorizer = SwinTExCoColorizer(swintexco_config)
            return colorizer.is_available()

        except ImportError as e:
            logger.debug(f"SwinTExCo not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"SwinTExCo check failed: {e}")
            return False

    def _get_colorizer(self):
        """Get or create the SwinTExCo colorizer."""
        if self._colorizer is None:
            from ..swintexco_colorize import (
                SwinTExCoColorizer,
                ExemplarColorizeConfig,
                ColorPropagationMode,
            )

            prop_mode = ColorPropagationMode(self.config.propagation_mode)

            swintexco_config = ExemplarColorizeConfig(
                reference_images=list(self.config.reference_images),
                temporal_fusion=self.config.temporal_consistency,
                propagation_mode=prop_mode,
                style_strength=self.config.strength,
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
            )

            self._colorizer = SwinTExCoColorizer(swintexco_config)

        return self._colorizer

    def colorize_frame(
        self,
        frame: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colorize using SwinTExCo with reference.

        Args:
            frame: Input frame (BGR)
            reference: Reference color image for exemplar matching
        """
        try:
            colorizer = self._get_colorizer()

            # SwinTExCo needs reference images
            if reference is None and not self.config.reference_images:
                logger.warning("SwinTExCo requires reference images, using simple transfer")

            # Use the internal colorization method
            references = []
            if reference is not None:
                references.append(reference)

            # Load config references if available
            for ref_path in self.config.reference_images:
                if HAS_OPENCV and Path(ref_path).exists():
                    ref_img = cv2.imread(str(ref_path))
                    if ref_img is not None:
                        references.append(ref_img)

            if not references:
                return frame

            return colorizer._colorize_frame(frame, references)

        except Exception as e:
            logger.error(f"SwinTExCo colorization failed: {e}")
            return frame

    def clear_cache(self) -> None:
        """Clear SwinTExCo model."""
        if self._colorizer is not None:
            self._colorizer.clear_cache()
            self._colorizer = None
        super().clear_cache()


# =============================================================================
# Temporal Consistency Backend
# =============================================================================

class TemporalColorBackend(BaseColorBackend):
    """Temporal colorization consistency backend.

    Wraps the existing TemporalColorizationProcessor for post-processing
    colorized frames to reduce flickering between frames.

    This is typically used as a post-processor after another colorization
    backend, not as a standalone colorizer.
    """

    @property
    def name(self) -> str:
        return "temporal"

    @property
    def min_vram_mb(self) -> int:
        return 2048  # 2GB for optical flow

    def __init__(self, config: ColorizerConfig):
        super().__init__(config)
        self._processor = None
        self._prev_colorized: Optional[np.ndarray] = None

    def is_available(self) -> bool:
        """Check if temporal processing is available."""
        try:
            from ..temporal_colorization import (
                TemporalColorizationProcessor,
                TemporalColorizationConfig,
                PropagationMode,
                BlendMethod,
            )
            return HAS_OPENCV
        except ImportError as e:
            logger.debug(f"Temporal colorization not available: {e}")
            return False

    def _get_processor(self):
        """Get or create the temporal processor."""
        if self._processor is None:
            from ..temporal_colorization import (
                TemporalColorizationProcessor,
                TemporalColorizationConfig,
                PropagationMode,
                BlendMethod,
            )

            prop_mode = PropagationMode(self.config.propagation_mode)

            temporal_config = TemporalColorizationConfig(
                temporal_window=self.config.temporal_window,
                propagation_mode=prop_mode,
                blend_strength=self.config.blend_strength,
                optical_flow_method="farneback",
                gpu_id=self.config.gpu_id,
                half_precision=self.config.half_precision,
            )

            self._processor = TemporalColorizationProcessor(temporal_config)

        return self._processor

    def colorize_frame(
        self,
        frame: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply temporal consistency to a single frame.

        Note: For best results, use colorize_batch which can consider
        neighboring frames for temporal consistency.

        Args:
            frame: Input colorized frame (BGR)
            reference: Previous colorized frame for consistency
        """
        # Temporal processing is more effective with batch processing
        # For single frames, return as-is or blend with previous
        if self._prev_colorized is not None and HAS_OPENCV:
            try:
                # Simple temporal blending
                blend = self.config.blend_strength
                result = cv2.addWeighted(
                    frame, 1.0 - blend * 0.3,
                    self._prev_colorized, blend * 0.3,
                    0
                )
                self._prev_colorized = frame
                return result
            except Exception:
                pass

        self._prev_colorized = frame
        return frame

    def colorize_batch(
        self,
        frames: List[np.ndarray],
        reference: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Apply temporal consistency to a batch of frames.

        This is the preferred method for temporal processing as it can
        consider multiple neighboring frames for better consistency.
        """
        try:
            if not HAS_OPENCV:
                return frames

            from ..temporal_colorization import (
                PropagationMode,
            )

            processor = self._get_processor()

            # Simple temporal blending implementation for batch
            results = []
            total = len(frames)

            for i, frame in enumerate(frames):
                result = frame.copy()

                # Forward blending
                if i > 0:
                    flow_x, flow_y = processor._flow_propagator.estimate_flow(
                        frames[i - 1], frame
                    )
                    confidence = processor._flow_propagator.compute_flow_confidence(
                        flow_x, flow_y
                    )
                    warped_a, warped_b = processor._flow_propagator.warp_chrominance(
                        results[i - 1], flow_x, flow_y, confidence
                    )

                    # Blend chrominance
                    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
                    blend = self.config.blend_strength * confidence.mean()
                    lab[:, :, 1] = lab[:, :, 1] * (1 - blend) + warped_a * blend
                    lab[:, :, 2] = lab[:, :, 2] * (1 - blend) + warped_b * blend
                    result = cv2.cvtColor(
                        np.clip(lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR
                    )

                results.append(result)

                if progress_callback:
                    progress_callback((i + 1) / total)

            return results

        except Exception as e:
            logger.error(f"Temporal consistency failed: {e}")
            return frames

    def clear_cache(self) -> None:
        """Clear temporal processor."""
        self._processor = None
        self._prev_colorized = None
        super().clear_cache()


# =============================================================================
# Unified Colorizer
# =============================================================================

class Colorizer:
    """Unified colorization processor with multiple backends.

    Automatically selects the optimal backend based on hardware capabilities
    and provides graceful fallback when backends are unavailable.

    Backend Selection (by VRAM):
    - 24GB+: SwinTExCo (if references available) or DDColor
    - 8-24GB: DDColor or SwinTExCo (if references available)
    - 4-8GB: DDColor
    - 2-4GB: DeOldify
    - <2GB or CPU: DeOldify (CPU mode)

    Example:
        >>> hardware = get_hardware_info()
        >>> config = ColorizerConfig(backend="auto")
        >>> colorizer = Colorizer(config, hardware)
        >>>
        >>> frames = [cv2.imread(f) for f in frame_paths]
        >>> colorized = colorizer.colorize(frames)
    """

    # Backend classes by name
    BACKENDS: Dict[str, type] = {
        "deoldify": DeOldifyBackend,
        "ddcolor": DDColorBackend,
        "swintexco": SwinTExCoBackend,
        "temporal": TemporalColorBackend,
    }

    # Fallback chain (highest quality first)
    FALLBACK_CHAIN = ["ddcolor", "deoldify", "swintexco"]

    def __init__(
        self,
        config: Optional[ColorizerConfig] = None,
        hardware: Optional[Any] = None,
    ):
        """Initialize unified colorizer.

        Args:
            config: Colorization configuration
            hardware: Hardware information for auto-selection
                     (from infrastructure.gpu.detector.HardwareInfo)
        """
        self.config = config or ColorizerConfig()
        self.hardware = hardware
        self._backend: Optional[BaseColorBackend] = None
        self._temporal_processor: Optional[TemporalColorBackend] = None

        # Initialize backend
        self._initialize_backend()

    def _get_vram_mb(self) -> int:
        """Get available VRAM from hardware info."""
        if self.hardware is None:
            return 0

        # Handle both dict and object
        if isinstance(self.hardware, dict):
            return self.hardware.get("total_vram_mb", 0)
        return getattr(self.hardware, "total_vram_mb", 0)

    def _select_optimal_backend(self) -> str:
        """Select optimal backend based on hardware tier.

        Returns:
            Backend name string
        """
        vram_mb = self._get_vram_mb()
        has_references = bool(self.config.reference_images)

        logger.debug(f"Selecting backend: VRAM={vram_mb}MB, references={has_references}")

        # High-end (24GB+): Prefer SwinTExCo if references available
        if vram_mb >= 24000:
            if has_references:
                return "swintexco"
            return "ddcolor"

        # Mid-high (8-24GB): DDColor or SwinTExCo
        if vram_mb >= 8000:
            if has_references:
                return "swintexco"
            return "ddcolor"

        # Mid (4-8GB): DDColor
        if vram_mb >= 4000:
            return "ddcolor"

        # Low (2-4GB): DeOldify
        if vram_mb >= 2000:
            return "deoldify"

        # CPU or very low VRAM: DeOldify (CPU mode)
        return "deoldify"

    def _initialize_backend(self) -> None:
        """Initialize the selected backend with fallback."""
        if self.config.backend == ColorBackend.AUTO:
            backend_name = self._select_optimal_backend()
        else:
            backend_name = self.config.backend.value

        # Try to initialize the selected backend
        if backend_name in self.BACKENDS:
            backend_class = self.BACKENDS[backend_name]
            backend = backend_class(self.config)

            if backend.is_available():
                self._backend = backend
                logger.info(f"Initialized colorization backend: {backend_name}")
                return

            logger.warning(f"Backend {backend_name} not available, trying fallback")

        # Fallback chain
        for fallback_name in self.FALLBACK_CHAIN:
            if fallback_name == backend_name:
                continue  # Already tried

            backend_class = self.BACKENDS.get(fallback_name)
            if backend_class is None:
                continue

            backend = backend_class(self.config)
            if backend.is_available():
                self._backend = backend
                logger.info(f"Using fallback backend: {fallback_name}")
                return

        logger.error("No colorization backend available")

        # Initialize temporal processor if enabled
        if self.config.temporal_consistency:
            self._temporal_processor = TemporalColorBackend(self.config)
            if not self._temporal_processor.is_available():
                self._temporal_processor = None

    def is_available(self) -> bool:
        """Check if colorization is available.

        Returns:
            True if at least one backend is available
        """
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        if self._backend is None:
            return "none"
        return self._backend.name

    def colorize(
        self,
        frames: List[np.ndarray],
        reference: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Colorize a list of frames.

        Args:
            frames: List of input frames (BGR format)
            reference: Optional reference frame for exemplar-based colorization
            progress_callback: Optional progress callback (0.0 to 1.0)

        Returns:
            List of colorized frames
        """
        if not self.is_available():
            logger.warning("No colorization backend available, returning original frames")
            return frames

        if not frames:
            return frames

        start_time = time.time()

        # Colorize using primary backend
        def colorize_progress(p: float) -> None:
            if progress_callback:
                if self.config.temporal_consistency and self._temporal_processor:
                    progress_callback(p * 0.7)  # 70% for colorization
                else:
                    progress_callback(p)

        colorized = self._backend.colorize_batch(
            frames,
            reference=reference,
            progress_callback=colorize_progress,
        )

        # Apply temporal consistency if enabled
        if (
            self.config.temporal_consistency
            and self._temporal_processor is not None
            and self._temporal_processor.is_available()
        ):
            def temporal_progress(p: float) -> None:
                if progress_callback:
                    progress_callback(0.7 + p * 0.3)  # 30% for temporal

            colorized = self._temporal_processor.colorize_batch(
                colorized,
                progress_callback=temporal_progress,
            )

        elapsed = time.time() - start_time
        logger.debug(
            f"Colorized {len(frames)} frames in {elapsed:.2f}s "
            f"using {self.backend_name}"
        )

        return colorized

    def colorize_frame(
        self,
        frame: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colorize a single frame.

        Args:
            frame: Input frame (BGR format)
            reference: Optional reference frame

        Returns:
            Colorized frame
        """
        if not self.is_available():
            return frame

        result = self._backend.colorize_frame(frame, reference)

        # Apply temporal consistency for single frame
        if (
            self.config.temporal_consistency
            and self._temporal_processor is not None
            and self._temporal_processor.is_available()
        ):
            result = self._temporal_processor.colorize_frame(result, reference)

        return result

    def colorize_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        reference: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ColorizationResult:
        """Colorize all frames in a directory.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            reference: Optional reference frame
            progress_callback: Progress callback

        Returns:
            ColorizationResult with processing statistics
        """
        result = ColorizationResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("No colorization backend available")
            return result

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir
        result.backend_used = self.backend_name

        # Find frames
        frame_files = sorted(input_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(input_dir.glob("*.jpg"))

        if not frame_files:
            logger.warning(f"No frames found in {input_dir}")
            return result

        total = len(frame_files)
        logger.info(f"Colorizing {total} frames using {self.backend_name}")

        # Process frames
        for i, frame_file in enumerate(frame_files):
            try:
                if not HAS_OPENCV:
                    import shutil
                    shutil.copy(frame_file, output_dir / frame_file.name)
                    result.frames_skipped += 1
                    result.frames_processed += 1
                    continue

                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Failed to read: {frame_file}")
                    result.frames_skipped += 1
                    result.frames_processed += 1
                    continue

                # Check if already colored
                if self.config.skip_colored and not self._backend.is_grayscale(frame):
                    cv2.imwrite(str(output_dir / frame_file.name), frame)
                    result.frames_skipped += 1
                    result.frames_processed += 1
                else:
                    # Colorize
                    colorized = self.colorize_frame(frame, reference)
                    cv2.imwrite(str(output_dir / frame_file.name), colorized)
                    result.frames_colorized += 1
                    result.frames_processed += 1

            except Exception as e:
                logger.error(f"Failed to colorize {frame_file}: {e}")
                # Copy original as fallback
                try:
                    import shutil
                    shutil.copy(frame_file, output_dir / frame_file.name)
                except Exception:
                    pass
                result.frames_skipped += 1
                result.frames_processed += 1

            if progress_callback:
                progress_callback((i + 1) / total)

        result.processing_time_seconds = time.time() - start_time
        result.temporal_consistency_applied = (
            self.config.temporal_consistency
            and self._temporal_processor is not None
        )

        # Get VRAM stats
        if HAS_TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"Colorization complete: {result.frames_colorized}/{total} colorized, "
            f"{result.frames_skipped} skipped, time: {result.processing_time_seconds:.1f}s"
        )

        return result

    def clear_cache(self) -> None:
        """Clear all models from memory."""
        if self._backend is not None:
            self._backend.clear_cache()
        if self._temporal_processor is not None:
            self._temporal_processor.clear_cache()


# =============================================================================
# Factory Functions
# =============================================================================

def create_colorizer(
    backend: str = "auto",
    strength: float = 1.0,
    temporal_consistency: bool = True,
    reference_images: Optional[List[Path]] = None,
    hardware: Optional[Any] = None,
    **kwargs,
) -> Colorizer:
    """Factory function to create a colorizer.

    Args:
        backend: Backend to use (auto, deoldify, ddcolor, swintexco, temporal)
        strength: Color strength (0-1)
        temporal_consistency: Apply temporal post-processing
        reference_images: Reference images for SwinTExCo
        hardware: Hardware info for auto-selection
        **kwargs: Additional config parameters

    Returns:
        Configured Colorizer instance
    """
    config = ColorizerConfig(
        backend=backend,
        strength=strength,
        temporal_consistency=temporal_consistency,
        reference_images=reference_images or [],
        **kwargs,
    )
    return Colorizer(config, hardware)


def colorize_auto(
    frames: List[np.ndarray],
    reference: Optional[np.ndarray] = None,
    hardware: Optional[Any] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> List[np.ndarray]:
    """Convenience function for automatic colorization.

    Args:
        frames: List of input frames (BGR format)
        reference: Optional reference frame
        hardware: Hardware info for backend selection
        progress_callback: Progress callback

    Returns:
        List of colorized frames
    """
    colorizer = create_colorizer(
        backend="auto",
        temporal_consistency=True,
        hardware=hardware,
    )
    return colorizer.colorize(frames, reference, progress_callback)
