"""Unified Face Restoration Processor for FrameWright.

This module provides a unified interface for face restoration that combines
multiple backends (GFPGAN, CodeFormer, RestoreFormer, AESRGAN) with automatic
hardware-based backend selection and graceful fallback chains.

Key features:
- Single FaceRestorer class with multiple backend support
- Auto-select optimal backend based on hardware tier
- Graceful fallback chain when backends are unavailable
- Preserves ALL existing functionality from individual backends
- Face detection, alignment, and background preservation

Example:
    >>> from framewright.processors.restoration.faces import UnifiedFaceRestorer, FaceConfig
    >>> from framewright.infrastructure.gpu import get_hardware_info
    >>>
    >>> config = FaceConfig(fidelity_weight=0.7)
    >>> hardware = get_hardware_info()
    >>> restorer = UnifiedFaceRestorer(config, hardware)
    >>>
    >>> result = restorer.restore_faces(input_dir, output_dir)
    >>> print(f"Restored {result.faces_restored} faces using {result.backend_used}")
"""

import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any, Type

import numpy as np

logger = logging.getLogger(__name__)

# Import existing implementations - wrap, don't rewrite
from ..face_restore import (
    FaceRestorer as GFPGANRestorer,
    FaceModel,
    FaceRestorationResult,
)
from ..aesrgan_face import (
    AESRGANFaceConfig,
    AESRGANFaceResult,
    AESRGANFaceRestorer,
    FaceDetectorType,
)

# Import hardware detection
from ...infrastructure.gpu import (
    HardwareInfo,
    HardwareTier,
    get_hardware_info,
)


class FaceBackendType(Enum):
    """Available face restoration backends."""
    GFPGAN_V1_3 = "gfpgan_v1.3"
    GFPGAN_V1_4 = "gfpgan_v1.4"
    CODEFORMER = "codeformer"
    RESTOREFORMER = "restoreformer"
    AESRGAN = "aesrgan"


@dataclass
class BackendInfo:
    """Information about a face restoration backend.

    Attributes:
        backend_type: The type of backend
        min_vram_mb: Minimum VRAM required in MB
        is_available: Whether the backend is currently available
        description: Human-readable description
        quality_rating: Quality rating 1-5 (5 is best)
        speed_rating: Speed rating 1-5 (5 is fastest)
    """
    backend_type: FaceBackendType
    min_vram_mb: int
    is_available: bool
    description: str
    quality_rating: int = 3
    speed_rating: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_type": self.backend_type.value,
            "min_vram_mb": self.min_vram_mb,
            "is_available": self.is_available,
            "description": self.description,
            "quality_rating": self.quality_rating,
            "speed_rating": self.speed_rating,
        }


@dataclass
class FaceConfig:
    """Unified configuration for face restoration.

    This configuration works across all backends, with backend-specific
    parameters applied when appropriate.

    Attributes:
        upscale: Output upscaling factor (1, 2, or 4)
        fidelity_weight: Balance between quality and fidelity (0=quality, 1=fidelity)
            - For CodeFormer: Maps directly to weight parameter
            - For GFPGAN: Affects enhancement strength
            - For AESRGAN: Maps to enhancement_strength
        only_center_face: Only process the center/largest face
        preserve_background: Preserve non-face regions from original
        detection_threshold: Face detection confidence threshold (0-1)
        bg_upsampler: Background upsampler (realesrgan, none)
        preferred_backend: Preferred backend (None = auto-select)
        gpu_id: GPU device ID for processing
        half_precision: Use FP16 for reduced VRAM usage
        face_detector: Face detection method (retinaface, mtcnn, dlib, opencv)
    """
    upscale: int = 2
    fidelity_weight: float = 0.5
    only_center_face: bool = False
    preserve_background: bool = True
    detection_threshold: float = 0.7
    bg_upsampler: str = "realesrgan"
    preferred_backend: Optional[FaceBackendType] = None
    gpu_id: int = 0
    half_precision: bool = True
    face_detector: str = "retinaface"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.upscale not in (1, 2, 4):
            raise ValueError(f"upscale must be 1, 2, or 4, got {self.upscale}")
        if not 0.0 <= self.fidelity_weight <= 1.0:
            raise ValueError(f"fidelity_weight must be 0-1, got {self.fidelity_weight}")
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError(f"detection_threshold must be 0-1, got {self.detection_threshold}")

        valid_detectors = ["retinaface", "mtcnn", "dlib", "opencv"]
        if self.face_detector not in valid_detectors:
            raise ValueError(
                f"Invalid face_detector '{self.face_detector}'. "
                f"Valid options: {valid_detectors}"
            )

    def to_gfpgan_config(self) -> Dict[str, Any]:
        """Convert to GFPGAN-compatible configuration."""
        # Map fidelity_weight to GFPGAN parameters
        # Higher fidelity = lower enhancement
        return {
            "upscale": self.upscale,
            "only_center_face": self.only_center_face,
            "bg_upsampler": self.bg_upsampler if self.preserve_background else None,
            "weight": self.fidelity_weight,  # CodeFormer uses this
        }

    def to_aesrgan_config(self) -> AESRGANFaceConfig:
        """Convert to AESRGAN-compatible configuration."""
        # Map fidelity_weight inversely to enhancement_strength
        # Higher fidelity = lower enhancement
        enhancement_strength = 1.0 - (self.fidelity_weight * 0.4)  # 0.6-1.0 range

        detector_map = {
            "retinaface": FaceDetectorType.RETINAFACE,
            "mtcnn": FaceDetectorType.MTCNN,
            "dlib": FaceDetectorType.DLIB,
            "opencv": FaceDetectorType.OPENCV,
        }

        return AESRGANFaceConfig(
            detection_threshold=self.detection_threshold,
            enhancement_strength=enhancement_strength,
            preserve_identity=self.fidelity_weight > 0.3,
            upscale_factor=self.upscale,
            paste_back=self.preserve_background,
            face_detector=detector_map.get(self.face_detector, FaceDetectorType.RETINAFACE),
            gpu_id=self.gpu_id,
            half_precision=self.half_precision,
        )


@dataclass
class UnifiedFaceResult:
    """Result of unified face restoration.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        faces_detected: Total number of faces detected
        faces_restored: Total number of faces restored
        backend_used: The backend that was used
        fallbacks_triggered: List of backends that failed before success
        output_dir: Path to output directory
        processing_time_seconds: Total processing time
        peak_vram_mb: Peak VRAM usage during processing
    """
    frames_processed: int = 0
    frames_failed: int = 0
    faces_detected: int = 0
    faces_restored: int = 0
    backend_used: Optional[FaceBackendType] = None
    fallbacks_triggered: List[FaceBackendType] = field(default_factory=list)
    output_dir: Optional[Path] = None
    processing_time_seconds: float = 0.0
    peak_vram_mb: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_processed": self.frames_processed,
            "frames_failed": self.frames_failed,
            "faces_detected": self.faces_detected,
            "faces_restored": self.faces_restored,
            "backend_used": self.backend_used.value if self.backend_used else None,
            "fallbacks_triggered": [fb.value for fb in self.fallbacks_triggered],
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "processing_time_seconds": self.processing_time_seconds,
            "peak_vram_mb": self.peak_vram_mb,
        }


class FaceBackend(ABC):
    """Abstract base class for face restoration backends."""

    @property
    @abstractmethod
    def backend_type(self) -> FaceBackendType:
        """Return the backend type."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> UnifiedFaceResult:
        """Restore faces in all frames in directory."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any cached models from memory."""
        pass


class GFPGANBackend(FaceBackend):
    """GFPGAN backend wrapper.

    Supports both GFPGAN v1.3 and v1.4 models.
    """

    def __init__(
        self,
        config: FaceConfig,
        model: FaceModel = FaceModel.GFPGAN_V1_4,
    ):
        self.config = config
        self.model = model
        self._restorer: Optional[GFPGANRestorer] = None

    @property
    def backend_type(self) -> FaceBackendType:
        if self.model == FaceModel.GFPGAN_V1_3:
            return FaceBackendType.GFPGAN_V1_3
        return FaceBackendType.GFPGAN_V1_4

    def is_available(self) -> bool:
        """Check if GFPGAN is available."""
        if self._restorer is None:
            gfpgan_config = self.config.to_gfpgan_config()
            self._restorer = GFPGANRestorer(
                model=self.model,
                upscale=gfpgan_config["upscale"],
                bg_upsampler=gfpgan_config["bg_upsampler"],
                only_center_face=gfpgan_config["only_center_face"],
                weight=gfpgan_config["weight"],
            )
        return self._restorer.is_available()

    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> UnifiedFaceResult:
        """Restore faces using GFPGAN."""
        start_time = time.time()

        if self._restorer is None:
            gfpgan_config = self.config.to_gfpgan_config()
            self._restorer = GFPGANRestorer(
                model=self.model,
                upscale=gfpgan_config["upscale"],
                bg_upsampler=gfpgan_config["bg_upsampler"],
                only_center_face=gfpgan_config["only_center_face"],
                weight=gfpgan_config["weight"],
            )

        # Call underlying restorer
        gfpgan_result = self._restorer.restore_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        # Convert to unified result
        return UnifiedFaceResult(
            frames_processed=gfpgan_result.frames_processed,
            frames_failed=gfpgan_result.failed_frames,
            faces_detected=gfpgan_result.faces_detected,
            faces_restored=gfpgan_result.faces_restored,
            backend_used=self.backend_type,
            output_dir=gfpgan_result.output_dir,
            processing_time_seconds=time.time() - start_time,
        )

    def clear_cache(self) -> None:
        """Clear GFPGAN model from memory."""
        self._restorer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class CodeFormerBackend(FaceBackend):
    """CodeFormer backend wrapper.

    Provides higher detail preservation compared to GFPGAN.
    """

    def __init__(self, config: FaceConfig):
        self.config = config
        self._restorer: Optional[GFPGANRestorer] = None

    @property
    def backend_type(self) -> FaceBackendType:
        return FaceBackendType.CODEFORMER

    def is_available(self) -> bool:
        """Check if CodeFormer is available."""
        if self._restorer is None:
            self._restorer = GFPGANRestorer(
                model=FaceModel.CODEFORMER,
                upscale=self.config.upscale,
                bg_upsampler=self.config.bg_upsampler if self.config.preserve_background else None,
                only_center_face=self.config.only_center_face,
                weight=self.config.fidelity_weight,
            )
        return self._restorer.is_available()

    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> UnifiedFaceResult:
        """Restore faces using CodeFormer."""
        start_time = time.time()

        if self._restorer is None:
            self._restorer = GFPGANRestorer(
                model=FaceModel.CODEFORMER,
                upscale=self.config.upscale,
                bg_upsampler=self.config.bg_upsampler if self.config.preserve_background else None,
                only_center_face=self.config.only_center_face,
                weight=self.config.fidelity_weight,
            )

        gfpgan_result = self._restorer.restore_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        return UnifiedFaceResult(
            frames_processed=gfpgan_result.frames_processed,
            frames_failed=gfpgan_result.failed_frames,
            faces_detected=gfpgan_result.faces_detected,
            faces_restored=gfpgan_result.faces_restored,
            backend_used=self.backend_type,
            output_dir=gfpgan_result.output_dir,
            processing_time_seconds=time.time() - start_time,
        )

    def clear_cache(self) -> None:
        """Clear CodeFormer model from memory."""
        self._restorer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class RestoreFormerBackend(FaceBackend):
    """RestoreFormer backend wrapper.

    Alternative architecture that may work better for some face types.
    Uses the GFPGAN infrastructure with RestoreFormer model.
    """

    def __init__(self, config: FaceConfig):
        self.config = config
        self._restorer: Optional[GFPGANRestorer] = None

    @property
    def backend_type(self) -> FaceBackendType:
        return FaceBackendType.RESTOREFORMER

    def is_available(self) -> bool:
        """Check if RestoreFormer is available."""
        if self._restorer is None:
            self._restorer = GFPGANRestorer(
                model=FaceModel.RESTOREFORMER,
                upscale=self.config.upscale,
                bg_upsampler=self.config.bg_upsampler if self.config.preserve_background else None,
                only_center_face=self.config.only_center_face,
            )
        return self._restorer.is_available()

    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> UnifiedFaceResult:
        """Restore faces using RestoreFormer."""
        start_time = time.time()

        if self._restorer is None:
            self._restorer = GFPGANRestorer(
                model=FaceModel.RESTOREFORMER,
                upscale=self.config.upscale,
                bg_upsampler=self.config.bg_upsampler if self.config.preserve_background else None,
                only_center_face=self.config.only_center_face,
            )

        gfpgan_result = self._restorer.restore_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        return UnifiedFaceResult(
            frames_processed=gfpgan_result.frames_processed,
            frames_failed=gfpgan_result.failed_frames,
            faces_detected=gfpgan_result.faces_detected,
            faces_restored=gfpgan_result.faces_restored,
            backend_used=self.backend_type,
            output_dir=gfpgan_result.output_dir,
            processing_time_seconds=time.time() - start_time,
        )

    def clear_cache(self) -> None:
        """Clear RestoreFormer model from memory."""
        self._restorer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class AESRGANBackend(FaceBackend):
    """AESRGAN (Attention-Enhanced ESRGAN) backend wrapper.

    Provides attention-enhanced face restoration with better preservation
    of subtle facial features. Requires more VRAM but produces higher
    quality results.
    """

    def __init__(self, config: FaceConfig, model_dir: Optional[Path] = None):
        self.config = config
        self.model_dir = model_dir
        self._restorer: Optional[AESRGANFaceRestorer] = None

    @property
    def backend_type(self) -> FaceBackendType:
        return FaceBackendType.AESRGAN

    def is_available(self) -> bool:
        """Check if AESRGAN is available."""
        if self._restorer is None:
            aesrgan_config = self.config.to_aesrgan_config()
            self._restorer = AESRGANFaceRestorer(
                config=aesrgan_config,
                model_dir=self.model_dir,
            )
        return self._restorer.is_available()

    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> UnifiedFaceResult:
        """Restore faces using AESRGAN."""
        if self._restorer is None:
            aesrgan_config = self.config.to_aesrgan_config()
            self._restorer = AESRGANFaceRestorer(
                config=aesrgan_config,
                model_dir=self.model_dir,
            )

        aesrgan_result = self._restorer.restore_faces(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        return UnifiedFaceResult(
            frames_processed=aesrgan_result.frames_processed,
            frames_failed=aesrgan_result.frames_failed,
            faces_detected=aesrgan_result.faces_enhanced,  # AESRGAN tracks enhanced
            faces_restored=aesrgan_result.faces_enhanced,
            backend_used=self.backend_type,
            output_dir=aesrgan_result.output_dir,
            processing_time_seconds=aesrgan_result.processing_time_seconds,
            peak_vram_mb=aesrgan_result.peak_vram_mb,
        )

    def clear_cache(self) -> None:
        """Clear AESRGAN model from memory."""
        if self._restorer is not None:
            self._restorer.clear_cache()
        self._restorer = None


class UnifiedFaceRestorer:
    """Unified face restoration with multiple backend support.

    This class provides a single interface for face restoration that:
    - Auto-selects the optimal backend based on hardware tier
    - Falls back gracefully when backends are unavailable
    - Preserves all functionality from individual backends

    Backend Selection by Hardware Tier:
        - CPU_ONLY: GFPGAN v1.3 (lightweight)
        - VRAM_4GB: GFPGAN v1.4 or v1.3
        - VRAM_8GB: CodeFormer or GFPGAN v1.4
        - VRAM_12GB+: CodeFormer or RestoreFormer
        - VRAM_16GB+: AESRGAN (attention-enhanced)
        - VRAM_24GB+: AESRGAN (full quality)

    Example:
        >>> config = FaceConfig(fidelity_weight=0.7)
        >>> hardware = get_hardware_info()
        >>> restorer = UnifiedFaceRestorer(config, hardware)
        >>>
        >>> # Process frames
        >>> result = restorer.restore_faces(input_dir, output_dir)
        >>> print(f"Used backend: {result.backend_used.value}")
    """

    # Backend requirements (VRAM in MB)
    BACKEND_REQUIREMENTS: Dict[FaceBackendType, int] = {
        FaceBackendType.GFPGAN_V1_3: 2000,   # 2GB VRAM, fast, lightweight
        FaceBackendType.GFPGAN_V1_4: 2500,   # 2.5GB VRAM, improved
        FaceBackendType.CODEFORMER: 3000,    # 3GB VRAM, higher detail
        FaceBackendType.RESTOREFORMER: 4000, # 4GB VRAM, alternative
        FaceBackendType.AESRGAN: 6000,       # 6GB+ VRAM, attention-based
    }

    # Fallback chains for when preferred backend is unavailable
    FALLBACK_CHAINS: Dict[FaceBackendType, List[FaceBackendType]] = {
        FaceBackendType.AESRGAN: [
            FaceBackendType.CODEFORMER,
            FaceBackendType.GFPGAN_V1_4,
            FaceBackendType.GFPGAN_V1_3,
        ],
        FaceBackendType.CODEFORMER: [
            FaceBackendType.GFPGAN_V1_4,
            FaceBackendType.GFPGAN_V1_3,
        ],
        FaceBackendType.RESTOREFORMER: [
            FaceBackendType.CODEFORMER,
            FaceBackendType.GFPGAN_V1_4,
            FaceBackendType.GFPGAN_V1_3,
        ],
        FaceBackendType.GFPGAN_V1_4: [
            FaceBackendType.GFPGAN_V1_3,
        ],
        FaceBackendType.GFPGAN_V1_3: [],
    }

    # Tier-based backend recommendations
    TIER_RECOMMENDATIONS: Dict[HardwareTier, FaceBackendType] = {
        HardwareTier.CPU_ONLY: FaceBackendType.GFPGAN_V1_3,
        HardwareTier.VRAM_4GB: FaceBackendType.GFPGAN_V1_4,
        HardwareTier.VRAM_8GB: FaceBackendType.CODEFORMER,
        HardwareTier.VRAM_12GB: FaceBackendType.CODEFORMER,
        HardwareTier.VRAM_16GB_PLUS: FaceBackendType.AESRGAN,
        HardwareTier.VRAM_24GB_PLUS: FaceBackendType.AESRGAN,
        HardwareTier.APPLE_SILICON: FaceBackendType.CODEFORMER,
    }

    def __init__(
        self,
        config: Optional[FaceConfig] = None,
        hardware: Optional[HardwareInfo] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize the unified face restorer.

        Args:
            config: Face restoration configuration (defaults to FaceConfig())
            hardware: Hardware information (auto-detected if None)
            model_dir: Directory for model weights (defaults to ~/.framewright/models)
        """
        self.config = config or FaceConfig()
        self.hardware = hardware or get_hardware_info()
        self.model_dir = model_dir

        # Initialize backend instances
        self._backends: Dict[FaceBackendType, FaceBackend] = {}
        self._active_backend: Optional[FaceBackend] = None

        # Select initial backend
        self._selected_backend = self._select_optimal_backend()
        logger.info(f"Selected face restoration backend: {self._selected_backend.value}")

    def _create_backend(self, backend_type: FaceBackendType) -> FaceBackend:
        """Create a backend instance for the given type."""
        if backend_type == FaceBackendType.GFPGAN_V1_3:
            return GFPGANBackend(self.config, FaceModel.GFPGAN_V1_3)
        elif backend_type == FaceBackendType.GFPGAN_V1_4:
            return GFPGANBackend(self.config, FaceModel.GFPGAN_V1_4)
        elif backend_type == FaceBackendType.CODEFORMER:
            return CodeFormerBackend(self.config)
        elif backend_type == FaceBackendType.RESTOREFORMER:
            return RestoreFormerBackend(self.config)
        elif backend_type == FaceBackendType.AESRGAN:
            return AESRGANBackend(self.config, self.model_dir)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def _get_backend(self, backend_type: FaceBackendType) -> FaceBackend:
        """Get or create a backend instance."""
        if backend_type not in self._backends:
            self._backends[backend_type] = self._create_backend(backend_type)
        return self._backends[backend_type]

    def _select_optimal_backend(self) -> FaceBackendType:
        """Select the optimal backend based on hardware and preferences.

        Returns:
            The selected backend type
        """
        # If user specified a preferred backend, try to use it
        if self.config.preferred_backend is not None:
            preferred = self.config.preferred_backend
            vram_required = self.BACKEND_REQUIREMENTS.get(preferred, 0)

            # Check if we have enough VRAM
            if self.hardware.total_vram_mb >= vram_required:
                backend = self._get_backend(preferred)
                if backend.is_available():
                    return preferred
                else:
                    logger.warning(
                        f"Preferred backend {preferred.value} not available, "
                        "falling back to auto-selection"
                    )

        # Auto-select based on hardware tier
        recommended = self.TIER_RECOMMENDATIONS.get(
            self.hardware.tier,
            FaceBackendType.GFPGAN_V1_4
        )

        # Verify we have enough VRAM for the recommended backend
        vram_available = self.hardware.total_vram_mb
        while recommended != FaceBackendType.GFPGAN_V1_3:
            vram_required = self.BACKEND_REQUIREMENTS.get(recommended, 0)
            if vram_available >= vram_required:
                backend = self._get_backend(recommended)
                if backend.is_available():
                    return recommended

            # Fall back to next option
            fallbacks = self.FALLBACK_CHAINS.get(recommended, [])
            if fallbacks:
                recommended = fallbacks[0]
            else:
                break

        # Final fallback to GFPGAN v1.3
        return FaceBackendType.GFPGAN_V1_3

    def get_available_backends(self) -> List[BackendInfo]:
        """Get list of all available backends with their info.

        Returns:
            List of BackendInfo for available backends
        """
        backend_infos = [
            BackendInfo(
                backend_type=FaceBackendType.GFPGAN_V1_3,
                min_vram_mb=2000,
                is_available=False,
                description="GFPGAN v1.3 - Fast, lightweight face restoration",
                quality_rating=3,
                speed_rating=5,
            ),
            BackendInfo(
                backend_type=FaceBackendType.GFPGAN_V1_4,
                min_vram_mb=2500,
                is_available=False,
                description="GFPGAN v1.4 - Improved quality over v1.3",
                quality_rating=4,
                speed_rating=4,
            ),
            BackendInfo(
                backend_type=FaceBackendType.CODEFORMER,
                min_vram_mb=3000,
                is_available=False,
                description="CodeFormer - Higher detail preservation with fidelity control",
                quality_rating=4,
                speed_rating=3,
            ),
            BackendInfo(
                backend_type=FaceBackendType.RESTOREFORMER,
                min_vram_mb=4000,
                is_available=False,
                description="RestoreFormer - Alternative architecture for varied faces",
                quality_rating=4,
                speed_rating=3,
            ),
            BackendInfo(
                backend_type=FaceBackendType.AESRGAN,
                min_vram_mb=6000,
                is_available=False,
                description="AESRGAN - Attention-enhanced, best for subtle details",
                quality_rating=5,
                speed_rating=2,
            ),
        ]

        # Check actual availability
        for info in backend_infos:
            try:
                backend = self._get_backend(info.backend_type)
                info.is_available = backend.is_available()
            except Exception:
                info.is_available = False

        return backend_infos

    def get_selected_backend(self) -> FaceBackendType:
        """Get the currently selected backend.

        Returns:
            The currently selected backend type
        """
        return self._selected_backend

    def set_backend(self, backend_type: FaceBackendType) -> bool:
        """Manually set the backend to use.

        Args:
            backend_type: The backend type to use

        Returns:
            True if backend was set successfully, False if unavailable
        """
        backend = self._get_backend(backend_type)
        if backend.is_available():
            self._selected_backend = backend_type
            logger.info(f"Manually set face restoration backend to: {backend_type.value}")
            return True

        logger.warning(f"Backend {backend_type.value} is not available")
        return False

    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> UnifiedFaceResult:
        """Restore faces in all frames in directory.

        This method attempts to use the selected backend, falling back
        through the fallback chain if needed.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback (0-1)

        Returns:
            UnifiedFaceResult with processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fallbacks_triggered: List[FaceBackendType] = []

        # Build list of backends to try
        backends_to_try = [self._selected_backend]
        backends_to_try.extend(
            self.FALLBACK_CHAINS.get(self._selected_backend, [])
        )

        # Try each backend in order
        for backend_type in backends_to_try:
            try:
                backend = self._get_backend(backend_type)

                if not backend.is_available():
                    logger.debug(f"Backend {backend_type.value} not available")
                    if backend_type != backends_to_try[0]:
                        fallbacks_triggered.append(backend_type)
                    continue

                logger.info(f"Using face restoration backend: {backend_type.value}")

                result = backend.restore_faces(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    progress_callback=progress_callback,
                )

                # Record fallbacks if we didn't use primary backend
                if fallbacks_triggered:
                    result.fallbacks_triggered = fallbacks_triggered

                self._active_backend = backend
                return result

            except Exception as e:
                logger.warning(f"Backend {backend_type.value} failed: {e}")
                fallbacks_triggered.append(backend_type)
                continue

        # All backends failed - copy originals as fallback
        logger.error("All face restoration backends failed, copying original frames")
        self._copy_frames(input_dir, output_dir)

        frame_count = len(list(input_dir.glob("*.png"))) + len(list(input_dir.glob("*.jpg")))

        return UnifiedFaceResult(
            frames_processed=frame_count,
            frames_failed=frame_count,
            faces_detected=0,
            faces_restored=0,
            backend_used=None,
            fallbacks_triggered=fallbacks_triggered,
            output_dir=output_dir,
        )

    def restore_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Restore faces in a single frame.

        Args:
            frame: Input BGR frame as numpy array

        Returns:
            Tuple of (enhanced frame, number of faces enhanced)
        """
        # Only AESRGAN backend supports single-frame processing
        if self._selected_backend == FaceBackendType.AESRGAN:
            backend = self._get_backend(FaceBackendType.AESRGAN)
            if isinstance(backend, AESRGANBackend) and backend.is_available():
                if backend._restorer is None:
                    aesrgan_config = self.config.to_aesrgan_config()
                    backend._restorer = AESRGANFaceRestorer(
                        config=aesrgan_config,
                        model_dir=self.model_dir,
                    )
                return backend._restorer.restore_frame(frame)

        # For other backends, we'd need to save/load through temp files
        # which is inefficient for single frames. Return original.
        logger.warning("Single-frame processing only supported for AESRGAN backend")
        return frame, 0

    def _copy_frames(self, input_dir: Path, output_dir: Path) -> None:
        """Copy frames when restoration fails."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame in input_dir.glob("*.png"):
            shutil.copy(frame, output_dir / frame.name)
        for frame in input_dir.glob("*.jpg"):
            shutil.copy(frame, output_dir / frame.name)

    def clear_cache(self) -> None:
        """Clear all backend models from memory."""
        for backend in self._backends.values():
            try:
                backend.clear_cache()
            except Exception as e:
                logger.debug(f"Error clearing backend cache: {e}")

        self._backends.clear()
        self._active_backend = None

        # Force GPU memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_face_restorer(
    fidelity_weight: float = 0.5,
    upscale: int = 2,
    preferred_backend: Optional[str] = None,
    gpu_id: int = 0,
) -> UnifiedFaceRestorer:
    """Factory function to create a unified face restorer.

    Args:
        fidelity_weight: Balance between quality and fidelity (0-1)
        upscale: Output upscaling factor (1, 2, or 4)
        preferred_backend: Preferred backend name (or None for auto)
        gpu_id: GPU device ID

    Returns:
        Configured UnifiedFaceRestorer instance
    """
    # Map string backend names to enum
    backend_map = {
        "gfpgan_v1.3": FaceBackendType.GFPGAN_V1_3,
        "gfpgan_v1.4": FaceBackendType.GFPGAN_V1_4,
        "gfpgan": FaceBackendType.GFPGAN_V1_4,
        "codeformer": FaceBackendType.CODEFORMER,
        "restoreformer": FaceBackendType.RESTOREFORMER,
        "aesrgan": FaceBackendType.AESRGAN,
    }

    backend_type = None
    if preferred_backend:
        backend_type = backend_map.get(preferred_backend.lower())
        if backend_type is None:
            logger.warning(
                f"Unknown backend '{preferred_backend}', using auto-selection"
            )

    config = FaceConfig(
        fidelity_weight=fidelity_weight,
        upscale=upscale,
        preferred_backend=backend_type,
        gpu_id=gpu_id,
    )

    return UnifiedFaceRestorer(config)


def restore_faces_auto(
    input_dir: Path,
    output_dir: Path,
    fidelity_weight: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> UnifiedFaceResult:
    """Convenience function for automatic face restoration.

    Automatically detects hardware, selects optimal backend, and
    processes all frames in the input directory.

    Args:
        input_dir: Directory containing input frames
        output_dir: Directory for output frames
        fidelity_weight: Balance between quality and fidelity (0-1)
        progress_callback: Optional progress callback (0-1)

    Returns:
        UnifiedFaceResult with processing statistics
    """
    restorer = create_face_restorer(fidelity_weight=fidelity_weight)

    try:
        result = restorer.restore_faces(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        return result
    finally:
        restorer.clear_cache()
