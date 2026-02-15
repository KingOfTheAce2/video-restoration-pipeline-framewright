"""Core module for FrameWright.

Provides foundational systems including:
- Configuration: Main configuration classes and utilities
- Events: Pub/sub event system for processing notifications
- Types: Core data types (Frame, VideoMetadata, BoundingBox, etc.)
- Unified error handling with user-friendly messages
- Authenticity preservation for film restoration

Example usage:

    >>> from framewright.core import (
    ...     FrameWrightConfig,
    ...     EventBus,
    ...     EventType,
    ...     Frame,
    ...     VideoMetadata,
    ... )
    >>>
    >>> # Create configuration
    >>> config = FrameWrightConfig(project_dir="/path/to/project")
    >>>
    >>> # Use event bus
    >>> bus = EventBus()
    >>> bus.subscribe(EventType.PROGRESS, lambda e: print(e))
"""

# Configuration
from .config import (
    # Main classes
    FrameWrightConfig,
    HardwareConfig,
    OutputConfig,
    ProcessorConfig,
    BaseConfig,
    # Enums
    ScaleFactor,
    OutputFormat,
    VideoCodec,
    AudioCodec,
    PixelFormat as ConfigPixelFormat,  # Aliased to avoid conflict with types.PixelFormat
    EncodingPreset,
    GPULoadStrategy,
    TemporalMethod,
    # Exceptions
    ConfigError,
    ValidationError as ConfigValidationError,  # Aliased to avoid conflict with errors.ValidationError
    # Functions
    load_config,
    save_config,
    merge_configs,
    validate_config,
    create_config_from_preset,
)

# Events
from .events import (
    # Event types
    EventType,
    # Event classes
    Event,
    ProcessingStartedEvent,
    FrameProcessedEvent,
    StageCompletedEvent,
    ProcessingErrorEvent,
    ProgressEvent,
    QualityCheckEvent,
    # Event bus
    EventBus,
    EventCallback,
    # Filtering
    EventFilter,
    FilteredSubscriber,
    # Global bus
    get_event_bus,
    reset_event_bus,
    # Convenience functions
    subscribe,
    emit,
    emit_async,
)

# Types
from .types import (
    # Type aliases
    ImageArray,
    FloatImageArray,
    GrayscaleArray,
    Point,
    PointFloat,
    Size,
    Region,
    PathLike,
    ColorRGB,
    ColorRGBA,
    ColorBGR,
    # Enums
    ColorSpace,
    PixelFormat,
    InterpolationMode,
    ProcessingStatus,
    QualityLevel,
    # Data classes
    BoundingBox,
    VideoMetadata,
    AudioMetadata,
    Frame,
    FrameSequence,
    ProcessingResult,
)

# Authenticity
from .authenticity import (
    AuthenticityManager,
    AuthenticityGuard,
    AuthenticityProfile,
    Era,
    SourceMedium,
    RestorationPhilosophy,
    EraDetector,
    ERA_PROFILES,
    create_authenticity_manager,
)

# Errors
from .errors import (
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
    # Context and utilities
    ErrorContext,
    create_error_context,
    classify_error,
    # Retry utilities
    RetryConfig,
    retry_with_backoff,
    RetryableOperation,
    # Error reporting
    ErrorReport,
    # Exception mapping
    EXCEPTION_MAP,
    get_exception_class,
)

__all__ = [
    # ==========================================================================
    # Configuration
    # ==========================================================================
    # Main classes
    "FrameWrightConfig",
    "HardwareConfig",
    "OutputConfig",
    "ProcessorConfig",
    "BaseConfig",
    # Enums
    "ScaleFactor",
    "OutputFormat",
    "VideoCodec",
    "AudioCodec",
    "ConfigPixelFormat",
    "EncodingPreset",
    "GPULoadStrategy",
    "TemporalMethod",
    # Exceptions
    "ConfigError",
    "ConfigValidationError",
    # Functions
    "load_config",
    "save_config",
    "merge_configs",
    "validate_config",
    "create_config_from_preset",
    # ==========================================================================
    # Events
    # ==========================================================================
    # Event types
    "EventType",
    # Event classes
    "Event",
    "ProcessingStartedEvent",
    "FrameProcessedEvent",
    "StageCompletedEvent",
    "ProcessingErrorEvent",
    "ProgressEvent",
    "QualityCheckEvent",
    # Event bus
    "EventBus",
    "EventCallback",
    # Filtering
    "EventFilter",
    "FilteredSubscriber",
    # Global bus
    "get_event_bus",
    "reset_event_bus",
    # Convenience functions
    "subscribe",
    "emit",
    "emit_async",
    # ==========================================================================
    # Types
    # ==========================================================================
    # Type aliases
    "ImageArray",
    "FloatImageArray",
    "GrayscaleArray",
    "Point",
    "PointFloat",
    "Size",
    "Region",
    "PathLike",
    "ColorRGB",
    "ColorRGBA",
    "ColorBGR",
    # Enums
    "ColorSpace",
    "PixelFormat",
    "InterpolationMode",
    "ProcessingStatus",
    "QualityLevel",
    # Data classes
    "BoundingBox",
    "VideoMetadata",
    "AudioMetadata",
    "Frame",
    "FrameSequence",
    "ProcessingResult",
    # ==========================================================================
    # Authenticity
    # ==========================================================================
    "AuthenticityManager",
    "AuthenticityGuard",
    "AuthenticityProfile",
    "Era",
    "SourceMedium",
    "RestorationPhilosophy",
    "EraDetector",
    "ERA_PROFILES",
    "create_authenticity_manager",
    # ==========================================================================
    # Errors
    # ==========================================================================
    # Base exception classes
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
    # Context and utilities
    "ErrorContext",
    "create_error_context",
    "classify_error",
    # Retry utilities
    "RetryConfig",
    "retry_with_backoff",
    "RetryableOperation",
    # Error reporting
    "ErrorReport",
    # Exception mapping
    "EXCEPTION_MAP",
    "get_exception_class",
]
