"""FrameWright - Video Restoration Pipeline using Real-ESRGAN."""
__version__ = "1.0.0"

# Core API - always available
from .config import Config
from .restorer import VideoRestorer, ProgressInfo

# Hardware detection
from .hardware import check_hardware, HardwareReport

# Key exception classes
from .core.errors import (
    FramewrightError,
    ProcessingError,
    ModelError,
    HardwareError,
    GPUError,
    VideoError,
    ConfigurationError,
    ValidationError,
)

__all__ = [
    # Core
    "VideoRestorer",
    "Config",
    "ProgressInfo",
    "__version__",
    # Hardware
    "check_hardware",
    "HardwareReport",
    # Exceptions
    "FramewrightError",
    "ProcessingError",
    "ModelError",
    "HardwareError",
    "GPUError",
    "VideoError",
    "ConfigurationError",
    "ValidationError",
]

# Lazy imports for backward compatibility.
# Modules that previously imported from framewright directly can still do so,
# but these are no longer eagerly loaded at package init time.


def __getattr__(name):
    """Lazy import for backward-compatible symbols not in __all__."""
    # Checkpointing
    if name in ("CheckpointManager", "PipelineCheckpoint", "FrameCheckpoint"):
        from . import checkpoint
        return getattr(checkpoint, name)

    # Legacy errors (errors.py / exceptions.py wrappers)
    _legacy_errors = {
        "VideoRestorerError", "TransientError", "ResourceError", "VRAMError",
        "DiskSpaceError", "NetworkError", "FatalError", "CorruptionError",
        "DependencyError", "DownloadError", "MetadataError",
        "AudioExtractionError", "FrameExtractionError", "EnhancementError",
        "ReassemblyError", "ErrorContext", "RetryConfig", "retry_with_backoff",
        "RetryableOperation", "ErrorReport", "OutOfMemoryError",
        "CheckpointError", "InterpolationError",
    }
    if name in _legacy_errors:
        from . import errors
        return getattr(errors, name)

    # Aliased exception names from old __init__
    _alias_map = {
        "FramewrightConfigurationError": "ConfigurationError",
        "FramewrightDownloadError": "DownloadError",
        "FramewrightInterpolationError": "InterpolationError",
        "FramewrightEnhancementError": "EnhancementError",
        "FramewrightValidationError": "ValidationError",
        "FramewrightDependencyError": "DependencyError",
        "FramewrightDiskSpaceError": "DiskSpaceError",
    }
    if name in _alias_map:
        from .core import errors as core_errors
        return getattr(core_errors, _alias_map[name])

    # Logging
    _logging_names = {
        "LogConfig", "FramewrightLogger", "configure_logging", "get_logger",
        "set_level", "add_file_handler", "ProcessingMetricsLog",
        "ErrorAggregator", "configure_from_cli",
    }
    if name in _logging_names:
        from .utils import logging as fw_logging
        return getattr(fw_logging, name)

    # Validators
    _validator_names = {
        "validate_frame_integrity", "validate_frame_sequence",
        "validate_frame_batch", "compute_quality_metrics",
        "validate_enhancement_quality", "detect_artifacts",
        "validate_temporal_consistency", "validate_audio_stream",
        "validate_av_sync", "detect_audio_issues", "analyze_audio_quality",
        "QualityMetrics", "FrameValidation", "SequenceReport",
        "ArtifactReport", "TemporalReport", "AudioValidation",
        "AudioIssue", "AudioQualityReport",
    }
    if name in _validator_names:
        from . import validators
        return getattr(validators, name)

    # Metrics
    _metrics_names = {
        "ProcessingMetrics", "ProgressReporter", "ProgressUpdate",
        "ConsoleProgressBar",
    }
    if name in _metrics_names:
        from . import metrics
        return getattr(metrics, name)

    # Hardware extras
    _hw_extras = {
        "print_hardware_report", "quick_check", "SystemInfo", "GPUCapability",
    }
    if name in _hw_extras:
        from . import hardware
        return getattr(hardware, name)

    raise AttributeError(f"module 'framewright' has no attribute {name!r}")
