"""FrameWright - Video Restoration Pipeline using Real-ESRGAN."""
__version__ = "1.3.1"

from .config import Config
from .restorer import VideoRestorer, ProgressInfo

# Checkpointing
from .checkpoint import (
    CheckpointManager,
    PipelineCheckpoint,
    FrameCheckpoint,
)

# New standardized exceptions (v1.3.1+)
from .exceptions import (
    FramewrightError,
    ConfigurationError as FramewrightConfigurationError,
    ProcessingError,
    ModelError,
    HardwareError,
    GPUError,
    OutOfMemoryError,
    VideoError,
    DownloadError as FramewrightDownloadError,
    CheckpointError,
    InterpolationError as FramewrightInterpolationError,
    EnhancementError as FramewrightEnhancementError,
    ValidationError as FramewrightValidationError,
    DependencyError as FramewrightDependencyError,
    DiskSpaceError as FramewrightDiskSpaceError,
)

# Structured logging (v1.3.1+)
from .utils.logging import (
    LogConfig,
    FramewrightLogger,
    configure_logging,
    get_logger,
    set_level,
    add_file_handler,
    ProcessingMetricsLog,
    ErrorAggregator,
    configure_from_cli,
)

# Legacy error handling (for backward compatibility)
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
    "ProgressInfo",
    "__version__",
    # Checkpointing
    "CheckpointManager",
    "PipelineCheckpoint",
    "FrameCheckpoint",
    # New standardized exceptions (v1.3.1+)
    "FramewrightError",
    "FramewrightConfigurationError",
    "ProcessingError",
    "ModelError",
    "HardwareError",
    "GPUError",
    "OutOfMemoryError",
    "VideoError",
    "FramewrightDownloadError",
    "CheckpointError",
    "FramewrightInterpolationError",
    "FramewrightEnhancementError",
    "FramewrightValidationError",
    "FramewrightDependencyError",
    "FramewrightDiskSpaceError",
    # Structured logging (v1.3.1+)
    "LogConfig",
    "FramewrightLogger",
    "configure_logging",
    "get_logger",
    "set_level",
    "add_file_handler",
    "ProcessingMetricsLog",
    "ErrorAggregator",
    "configure_from_cli",
    # Legacy errors (backward compatibility)
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
