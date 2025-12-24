"""Processors module for framewright video restoration pipeline.

This module contains various processors for audio and video manipulation:

- analyzer: Automated content and degradation detection
- defect_repair: Scratch, dust, and grain removal
- face_restore: GFPGAN/CodeFormer face enhancement
- adaptive_enhance: Content-aware adaptive processing
- interpolation: RIFE frame interpolation
- audio: Audio processing
"""

from .audio import AudioProcessor, AudioProcessorError
from .interpolation import FrameInterpolator, InterpolationError

from .analyzer import (
    FrameAnalyzer,
    VideoAnalysis,
    ContentType,
    DegradationType,
    FrameMetrics,
)

from .defect_repair import (
    DefectDetector,
    DefectRepairer,
    AutoDefectProcessor,
    DefectType,
    DefectMap,
    DefectRepairResult,
)

from .face_restore import (
    FaceRestorer,
    AutoFaceRestorer,
    FaceModel,
    FaceRestorationResult,
)

from .adaptive_enhance import (
    AdaptiveEnhancer,
    AutoEnhancePipeline,
    AdaptiveEnhanceResult,
)

__all__ = [
    # Audio
    'AudioProcessor',
    'AudioProcessorError',
    # Interpolation
    'FrameInterpolator',
    'InterpolationError',
    # Analyzer
    "FrameAnalyzer",
    "VideoAnalysis",
    "ContentType",
    "DegradationType",
    "FrameMetrics",
    # Defect repair
    "DefectDetector",
    "DefectRepairer",
    "AutoDefectProcessor",
    "DefectType",
    "DefectMap",
    "DefectRepairResult",
    # Face restoration
    "FaceRestorer",
    "AutoFaceRestorer",
    "FaceModel",
    "FaceRestorationResult",
    # Adaptive enhancement
    "AdaptiveEnhancer",
    "AutoEnhancePipeline",
    "AdaptiveEnhanceResult",
]
