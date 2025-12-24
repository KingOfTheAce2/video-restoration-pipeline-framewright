"""FrameWright - Video Restoration Pipeline using Real-ESRGAN."""
__version__ = "1.2.0"

from .config import Config
from .restorer import VideoRestorer

# Checkpointing
from .checkpoint import (
    CheckpointManager,
    PipelineCheckpoint,
    FrameCheckpoint,
)

# Error handling
from .errors import (
    VideoRestorerError,
    TransientError,
    ResourceError,
    VRAMError,
    DiskSpaceError,
    NetworkError,
    FatalError,
    CorruptionError,
    DependencyError,
    ConfigurationError,
    ValidationError,
    DownloadError,
    MetadataError,
    AudioExtractionError,
    FrameExtractionError,
    EnhancementError,
    ReassemblyError,
    ErrorContext,
    RetryConfig,
    retry_with_backoff,
    RetryableOperation,
    ErrorReport,
)

# Validators
from .validators import (
    validate_frame_integrity,
    validate_frame_sequence,
    validate_frame_batch,
    compute_quality_metrics,
    validate_enhancement_quality,
    detect_artifacts,
    validate_temporal_consistency,
    validate_audio_stream,
    validate_av_sync,
    detect_audio_issues,
    analyze_audio_quality,
    QualityMetrics,
    FrameValidation,
    SequenceReport,
    ArtifactReport,
    TemporalReport,
    AudioValidation,
    AudioIssue,
    AudioQualityReport,
)

# Metrics
from .metrics import (
    ProcessingMetrics,
    ProgressReporter,
    ProgressUpdate,
    ConsoleProgressBar,
)

# Hardware
from .hardware import (
    check_hardware,
    print_hardware_report,
    quick_check,
    HardwareReport,
    SystemInfo,
    GPUCapability,
)

__all__ = [
    # Core
    "VideoRestorer",
    "Config",
    "__version__",
    # Checkpointing
    "CheckpointManager",
    "PipelineCheckpoint",
    "FrameCheckpoint",
    # Errors
    "VideoRestorerError",
    "TransientError",
    "ResourceError",
    "VRAMError",
    "DiskSpaceError",
    "NetworkError",
    "FatalError",
    "CorruptionError",
    "DependencyError",
    "ConfigurationError",
    "ValidationError",
    "DownloadError",
    "MetadataError",
    "AudioExtractionError",
    "FrameExtractionError",
    "EnhancementError",
    "ReassemblyError",
    "ErrorContext",
    "RetryConfig",
    "retry_with_backoff",
    "RetryableOperation",
    "ErrorReport",
    # Validators
    "validate_frame_integrity",
    "validate_frame_sequence",
    "validate_frame_batch",
    "compute_quality_metrics",
    "validate_enhancement_quality",
    "detect_artifacts",
    "validate_temporal_consistency",
    "validate_audio_stream",
    "validate_av_sync",
    "detect_audio_issues",
    "analyze_audio_quality",
    "QualityMetrics",
    "FrameValidation",
    "SequenceReport",
    "ArtifactReport",
    "TemporalReport",
    "AudioValidation",
    "AudioIssue",
    "AudioQualityReport",
    # Metrics
    "ProcessingMetrics",
    "ProgressReporter",
    "ProgressUpdate",
    "ConsoleProgressBar",
    # Hardware
    "check_hardware",
    "print_hardware_report",
    "quick_check",
    "HardwareReport",
    "SystemInfo",
    "GPUCapability",
]
