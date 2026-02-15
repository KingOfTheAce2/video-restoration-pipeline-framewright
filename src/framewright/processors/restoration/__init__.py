"""Restoration processors module.

This module contains unified restoration processors that combine multiple
backends with automatic hardware-based selection.

Available processors:
- Colorizer: Unified colorization with DeOldify, DDColor, SwinTExCo, and Temporal backends
- UnifiedFaceRestorer: Unified face restoration with GFPGAN, CodeFormer, RestoreFormer, AESRGAN
- GrainManager: Film grain extraction, removal, and restoration for authentic film look
- GenerativeFrameExtender: Frame generation, interpolation, gap filling, and damage restoration
- DefectProcessor: Defect detection and repair (scratches, dust, damage)
- Stabilizer: Video stabilization with multiple backends
"""

from .colorization import (
    # Configuration
    ColorizerConfig,
    ColorBackend,
    # Result
    ColorizationResult,
    # Backend wrappers
    DeOldifyBackend,
    DDColorBackend,
    SwinTExCoBackend,
    TemporalColorBackend,
    # Main class
    Colorizer,
    # Factory functions
    create_colorizer,
    colorize_auto,
)

from .faces import (
    # Enums
    FaceBackendType,
    # Configuration
    FaceConfig,
    BackendInfo,
    # Result
    UnifiedFaceResult,
    # Backend wrappers
    GFPGANBackend,
    CodeFormerBackend,
    RestoreFormerBackend,
    AESRGANBackend,
    # Main class
    UnifiedFaceRestorer,
    # Factory functions
    create_face_restorer,
    restore_faces_auto,
)

from .grain_manager import (
    # Enums
    FilmStockGrainType,
    # Data classes
    GrainProfile,
    GrainConfig,
    GrainRemovalResult,
    # Constants
    FILM_STOCK_PROFILES,
    # Main class
    GrainManager,
    # Factory functions
    create_grain_manager,
    extract_grain_profile,
    process_with_grain_preservation,
)

from .frame_generator import (
    # Enums
    InterpolationAlgorithm,
    GenerationModel,
    # Configuration
    FrameGenConfig,
    # Data classes
    MotionVector,
    MotionField,
    InterpolationResult,
    ExtensionResult,
    GapFillResult,
    # Classes
    MotionEstimator,
    FrameInterpolator,
    FrameExtender,
    GapFiller,
    DamagedFrameRestorer,
    GenerativeFrameExtender,
    # Factory functions
    create_frame_generator,
    extend_video,
    interpolate_fps,
    fill_missing_frames,
)

from .defects import (
    # Enums
    DefectType,
    RepairBackend,
    DetectionMode,
    # Configuration
    DefectConfig,
    # Data classes
    DefectInfo,
    DetectionResult,
    RepairResult,
    # Classes
    DefectDetector,
    DefectRepairer,
    DefectProcessor,
    # Backend classes
    RepairBackendBase,
    OpenCVRepairBackend,
    # Factory functions
    create_defect_processor,
    detect_defects,
    repair_defects,
    process_frames_auto,
)

from .stabilization import (
    # Enums
    SmoothingMode,
    MotionType,
    StabilizationBackendType,
    # Configuration
    StabilizationConfig,
    # Data classes
    CameraMotion,
    MotionAnalysisResult,
    StabilizationResult,
    # Classes
    MotionAnalyzer,
    Stabilizer,
    # Backend classes
    StabilizationBackend,
    VidStabBackend,
    OpenCVBackend,
    # Factory functions
    create_stabilizer,
    detect_shake_severity,
    stabilize_frames,
)

__all__ = [
    # Colorization - Configuration
    "ColorizerConfig",
    "ColorBackend",
    # Colorization - Result
    "ColorizationResult",
    # Colorization - Backend wrappers
    "DeOldifyBackend",
    "DDColorBackend",
    "SwinTExCoBackend",
    "TemporalColorBackend",
    # Colorization - Main class
    "Colorizer",
    # Colorization - Factory functions
    "create_colorizer",
    "colorize_auto",
    # Face Restoration - Enums
    "FaceBackendType",
    # Face Restoration - Configuration
    "FaceConfig",
    "BackendInfo",
    # Face Restoration - Result
    "UnifiedFaceResult",
    # Face Restoration - Backend wrappers
    "GFPGANBackend",
    "CodeFormerBackend",
    "RestoreFormerBackend",
    "AESRGANBackend",
    # Face Restoration - Main class
    "UnifiedFaceRestorer",
    # Face Restoration - Factory functions
    "create_face_restorer",
    "restore_faces_auto",
    # Grain Management - Enums
    "FilmStockGrainType",
    # Grain Management - Data classes
    "GrainProfile",
    "GrainConfig",
    "GrainRemovalResult",
    # Grain Management - Constants
    "FILM_STOCK_PROFILES",
    # Grain Management - Main class
    "GrainManager",
    # Grain Management - Factory functions
    "create_grain_manager",
    "extract_grain_profile",
    "process_with_grain_preservation",
    # Frame Generation - Enums
    "InterpolationAlgorithm",
    "GenerationModel",
    # Frame Generation - Configuration
    "FrameGenConfig",
    # Frame Generation - Data classes
    "MotionVector",
    "MotionField",
    "InterpolationResult",
    "ExtensionResult",
    "GapFillResult",
    # Frame Generation - Classes
    "MotionEstimator",
    "FrameInterpolator",
    "FrameExtender",
    "GapFiller",
    "DamagedFrameRestorer",
    "GenerativeFrameExtender",
    # Frame Generation - Factory functions
    "create_frame_generator",
    "extend_video",
    "interpolate_fps",
    "fill_missing_frames",
    # Defect Detection and Repair - Enums
    "DefectType",
    "RepairBackend",
    "DetectionMode",
    # Defect Detection and Repair - Configuration
    "DefectConfig",
    # Defect Detection and Repair - Data classes
    "DefectInfo",
    "DetectionResult",
    "RepairResult",
    # Defect Detection and Repair - Classes
    "DefectDetector",
    "DefectRepairer",
    "DefectProcessor",
    # Defect Detection and Repair - Backend classes
    "RepairBackendBase",
    "OpenCVRepairBackend",
    # Defect Detection and Repair - Factory functions
    "create_defect_processor",
    "detect_defects",
    "repair_defects",
    "process_frames_auto",
    # Stabilization - Enums
    "SmoothingMode",
    "MotionType",
    "StabilizationBackendType",
    # Stabilization - Configuration
    "StabilizationConfig",
    # Stabilization - Data classes
    "CameraMotion",
    "MotionAnalysisResult",
    "StabilizationResult",
    # Stabilization - Classes
    "MotionAnalyzer",
    "Stabilizer",
    # Stabilization - Backend classes
    "StabilizationBackend",
    "VidStabBackend",
    "OpenCVBackend",
    # Stabilization - Factory functions
    "create_stabilizer",
    "detect_shake_severity",
    "stabilize_frames",
]
