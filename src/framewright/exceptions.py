"""Legacy exception hierarchy for FrameWright.

DEPRECATED: This module is maintained for backward compatibility.
Please import from `framewright.core.errors` instead.

All exceptions have been consolidated into the unified error handling
system at `framewright.core.errors`.
"""

import warnings

# Re-export everything from the new location
from .core.errors import (
    # Base classes
    FramewrightError,
    TransientError,
    FatalError,
    # Hardware errors
    HardwareError,
    GPUError,
    VRAMError,
    OutOfMemoryError,
    GPURequiredError,
    CPUFallbackError,
    # Configuration
    ConfigurationError,
    # Processing errors
    ProcessingError,
    EnhancementError,
    InterpolationError,
    FrameExtractionError,
    ReassemblyError,
    # Model errors
    ModelError,
    # Video errors
    VideoError,
    MetadataError,
    AudioExtractionError,
    # Network errors
    NetworkError,
    DownloadError,
    TimeoutError,
    # Storage errors
    StorageError,
    DiskSpaceError,
    CheckpointError,
    # Validation and dependency
    ValidationError,
    DependencyError,
    # Corruption
    CorruptionError,
    # Legacy compatibility
    ResourceError,
    VideoRestorerError,
    # Exception mapping
    EXCEPTION_MAP,
    get_exception_class,
)


def _emit_deprecation_warning() -> None:
    """Emit deprecation warning for this module."""
    warnings.warn(
        "framewright.exceptions is deprecated. "
        "Please import from framewright.core.errors instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Note: We don't emit the warning on import to avoid breaking existing code,
# but users should migrate to the new location.


__all__ = [
    # Base classes
    "FramewrightError",
    "TransientError",
    "FatalError",
    # Hardware errors
    "HardwareError",
    "GPUError",
    "VRAMError",
    "OutOfMemoryError",
    "GPURequiredError",
    "CPUFallbackError",
    # Configuration
    "ConfigurationError",
    # Processing errors
    "ProcessingError",
    "EnhancementError",
    "InterpolationError",
    "FrameExtractionError",
    "ReassemblyError",
    # Model errors
    "ModelError",
    # Video errors
    "VideoError",
    "MetadataError",
    "AudioExtractionError",
    # Network errors
    "NetworkError",
    "DownloadError",
    "TimeoutError",
    # Storage errors
    "StorageError",
    "DiskSpaceError",
    "CheckpointError",
    # Validation and dependency
    "ValidationError",
    "DependencyError",
    # Corruption
    "CorruptionError",
    # Legacy compatibility
    "ResourceError",
    "VideoRestorerError",
    # Exception mapping
    "EXCEPTION_MAP",
    "get_exception_class",
]
