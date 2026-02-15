"""Format-specific processors for FrameWright.

This module provides specialized processors for different video format types:
- Film: Gate weave, flicker, color fade, grain handling
- VHS: Tracking errors, head switching, chroma bleed, dropouts
- Interlace: Deinterlacing with multiple methods, telecine removal
- Aspect: Letterbox/pillarbox detection and aspect ratio conversion

Example:
    >>> from framewright.processors.format import (
    ...     FilmProcessor, FilmConfig,
    ...     VHSProcessor, VHSConfig,
    ...     Deinterlacer, DeinterlaceMethod,
    ...     AspectHandler, StandardAspectRatio,
    ... )
    >>>
    >>> # Film restoration
    >>> film = FilmProcessor(FilmConfig(gate_weave=0.8, flicker=0.6))
    >>> restored = film.process(frames)
    >>>
    >>> # VHS restoration
    >>> vhs = VHSProcessor(VHSConfig(tracking=0.7, head_switching=0.9))
    >>> restored = vhs.process(frames)
    >>>
    >>> # Deinterlacing
    >>> deint = Deinterlacer()
    >>> if deint.detect_interlacing(frames):
    ...     progressive = deint.deinterlace(frames, DeinterlaceMethod.BWDIF)
    >>>
    >>> # Aspect ratio handling
    >>> aspect = AspectHandler()
    >>> if aspect.detect_letterbox(frame):
    ...     cropped = aspect.crop_letterbox(frames)
"""

# Film processor
from framewright.processors.format.film import (
    FilmConfig,
    FilmProcessor,
    FilmFormat,
    FilmEra,
    FilmAnalysis,
    create_film_processor,
)

# VHS processor
from framewright.processors.format.vhs import (
    VHSConfig,
    VHSProcessor,
    VHSQuality,
    VHSAnalysis,
    VHSArtifactInfo,
    ArtifactType,
    create_vhs_processor,
)

# Interlace processor
from framewright.processors.format.interlace import (
    InterlaceConfig,
    Deinterlacer,
    DeinterlaceMethod,
    FieldOrder,
    TelecinePattern,
    InterlaceAnalysis,
    DeinterlaceResult,
    create_deinterlacer,
)

# Aspect ratio processor
from framewright.processors.format.aspect import (
    AspectConfig,
    AspectHandler,
    AspectAnalysis,
    StandardAspectRatio,
    ConversionMethod,
    FillMode,
    CropRegion,
    create_aspect_handler,
    RATIO_4_3,
    RATIO_16_9,
    RATIO_2_35_1,
    RATIO_1_85_1,
)

__all__ = [
    # Film
    "FilmConfig",
    "FilmProcessor",
    "FilmFormat",
    "FilmEra",
    "FilmAnalysis",
    "create_film_processor",
    # VHS
    "VHSConfig",
    "VHSProcessor",
    "VHSQuality",
    "VHSAnalysis",
    "VHSArtifactInfo",
    "ArtifactType",
    "create_vhs_processor",
    # Interlace
    "InterlaceConfig",
    "Deinterlacer",
    "DeinterlaceMethod",
    "FieldOrder",
    "TelecinePattern",
    "InterlaceAnalysis",
    "DeinterlaceResult",
    "create_deinterlacer",
    # Aspect
    "AspectConfig",
    "AspectHandler",
    "AspectAnalysis",
    "StandardAspectRatio",
    "ConversionMethod",
    "FillMode",
    "CropRegion",
    "create_aspect_handler",
    "RATIO_4_3",
    "RATIO_16_9",
    "RATIO_2_35_1",
    "RATIO_1_85_1",
]
