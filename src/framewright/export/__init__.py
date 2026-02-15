"""Export module for FrameWright.

Provides video export with platform-optimized presets, metadata sidecar generation,
and export validation for verifying output integrity.
"""

from .presets import (
    ExportPreset,
    ExportPresetManager,
    VideoExporter,
    ExportPlatform,
    VideoCodec,
    AudioCodec,
    Container,
    ColorSpace,
    HDRMode,
    VideoSettings,
    AudioSettings,
    BUILTIN_PRESETS,
    export_video,
    list_presets,
)

from .sidecar import (
    SidecarMetadata,
    FileInfo,
    QualityMetricsData,
    ProcessingStats,
    RestorationSettings,
    generate_sidecar,
    load_sidecar,
    get_sidecar_path,
    sidecar_exists,
    quick_sidecar,
    verify_sidecar,
    calculate_md5,
    calculate_sha256,
    calculate_checksums,
)

from .validation import (
    IssueSeverity as ValidationSeverity,
    IssueType as ValidationType,
    ValidationIssue,
    ValidationResult,
    ExportValidator,
)

__all__ = [
    # Presets
    "ExportPreset",
    "ExportPresetManager",
    "VideoExporter",
    "ExportPlatform",
    "VideoCodec",
    "AudioCodec",
    "Container",
    "ColorSpace",
    "HDRMode",
    "VideoSettings",
    "AudioSettings",
    "BUILTIN_PRESETS",
    "export_video",
    "list_presets",
    # Sidecar
    "SidecarMetadata",
    "FileInfo",
    "QualityMetricsData",
    "ProcessingStats",
    "RestorationSettings",
    "generate_sidecar",
    "load_sidecar",
    "get_sidecar_path",
    "sidecar_exists",
    "quick_sidecar",
    "verify_sidecar",
    "calculate_md5",
    "calculate_sha256",
    "calculate_checksums",
    # Validation
    "ValidationSeverity",
    "ValidationType",
    "ValidationIssue",
    "ValidationResult",
    "ExportValidator",
]
