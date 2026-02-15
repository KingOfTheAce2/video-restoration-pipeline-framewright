"""
FrameWright Utilities Package

Utility functions and helpers for video processing including:
- FFmpeg integration
- GPU memory management and optimization
- Disk space monitoring
- Dependency validation
- Async I/O operations
- Model management (download, verify, cleanup)
- Progress tracking and webhooks
- Job scheduling
- Power management
- Output filename templates
"""

from .ffmpeg import (
    check_ffmpeg_installed,
    get_video_info,
    extract_frames,
    reassemble_video,
    extract_audio,
    merge_audio_video,
    get_video_fps,
    get_video_duration,
    get_video_resolution,
    get_available_encoders,
    get_best_video_codec,
)

from .gpu import (
    get_gpu_memory_info,
    get_all_gpu_info,
    get_optimal_device,
    calculate_optimal_tile_size,
    get_adaptive_tile_sequence,
    VRAMMonitor,
    wait_for_vram,
    GPUInfo,
)

from .disk import (
    get_disk_usage,
    get_directory_size,
    validate_disk_space,
    estimate_required_space,
    DiskSpaceMonitor,
    cleanup_old_temp_files,
    DiskUsage,
    SpaceEstimate,
)

from .dependencies import (
    validate_all_dependencies,
    check_ffmpeg,
    check_ffprobe,
    check_realesrgan,
    check_ytdlp,
    check_rife,
    get_enhancement_backend,
    compare_versions,
    DependencyInfo,
    DependencyReport,
)

from .async_io import (
    AsyncFileOperations,
    AsyncSubprocess,
    AsyncDownloader,
    AsyncFrameProcessor,
    AsyncDownloadResult,
    AsyncReadResult,
    AsyncWriteResult,
    run_pipeline_async,
    run_async,
    get_io_executor,
    shutdown_executor,
)

from .model_manager import (
    ModelManager,
    ModelType,
    ModelInfo,
    DownloadProgress,
    DownloadError,
    ModelVerificationError,
    MODEL_REGISTRY,
    get_model_manager,
    ProgressCallback,
)

from .output_manager import (
    OutputManager,
    OutputPaths,
)

from .youtube import (
    YouTubeDownloader,
    VideoInfo,
    FormatInfo,
    ChapterInfo,
    DownloadProgress as YouTubeDownloadProgress,
    YouTubeDownloadError,
    YouTubeMetadataError,
    get_youtube_downloader,
    ProgressCallback as YouTubeProgressCallback,
)

from .cache import (
    FrameCache,
    CacheManager,
    CacheConfig,
    CacheEntry,
    CacheStats,
    PerceptualHasher,
    compute_frame_hash,
    get_cache_manager,
    get_global_cache,
    shutdown_cache,
)

from .multi_gpu import (
    GPUManager,
    MultiGPUDistributor,
    GPUInfo as MultiGPUInfo,
    DistributionResult,
    WorkItem,
    WorkStealingQueue,
    LoadBalanceStrategy,
    detect_gpus,
    get_optimal_gpu,
    distribute_frames,
)

from .config_file import (
    ConfigFileManager,
    ValidationError as ConfigValidationError,
    get_config_manager,
    DEFAULT_CONFIG_TEMPLATE,
    CONFIG_SCHEMA,
)

from .progress import (
    RichProgress,
    ProgressTracker,
    ProgressOutputMode,
    GPUMetrics,
    GPUMonitor,
    StageInfo,
    ProgressCallback as RichProgressCallback,
    LogFileCallback,
    FallbackProgress,
    create_progress_tracker,
    simple_progress,
    is_rich_available,
    is_pynvml_available,
)

# Structured logging (v1.3.1+)
from .logging import (
    LogConfig,
    FramewrightLogger,
    configure_logging,
    get_logger,
    set_level,
    add_file_handler,
    ProcessingMetricsLog,
    ErrorAggregator,
    configure_from_cli,
    get_cli_args_parser,
)

# GPU Memory Optimizer
from .gpu_memory_optimizer import (
    GPUMemoryOptimizer,
    GPUMemoryStats,
    OptimizationConfig,
    MemoryPressure,
    get_optimizer,
    get_optimal_batch_size,
    get_optimal_tile_size,
    clear_gpu_memory,
)

# Thermal Monitoring
from .thermal_monitor import (
    ThermalMonitor,
    ThermalState,
    ThrottleState,
    ThermalReading,
    ThermalProfile,
    get_gpu_temperature,
    is_gpu_throttling,
)

# Model Manager CLI
from .model_manager_cli import (
    ModelManagerCLI,
    ModelInfo as ModelManagerInfo,
    ModelCategory,
)

# Progress Webhook
from .webhook import (
    WebhookType,
    EventType,
    WebhookConfig,
    WebhookEvent,
    ProgressWebhook,
    WebhookManager,
)

# Job Scheduler
from .scheduler import (
    JobStatus,
    ScheduleType,
    JobConstraints,
    ScheduledJob,
    JobScheduler,
)

# Power Management
from .power_manager import (
    PowerAction,
    PowerState,
    PowerManager,
    KeepAwake,
    prevent_sleep_during,
)

# Output Templates
from .output_templates import (
    VideoMetadata,
    ProcessingContext,
    OutputTemplate,
    TemplateManager,
)

__all__ = [
    # FFmpeg utilities
    'check_ffmpeg_installed',
    'get_video_info',
    'extract_frames',
    'reassemble_video',
    'extract_audio',
    'merge_audio_video',
    'get_video_fps',
    'get_video_duration',
    'get_video_resolution',
    'get_available_encoders',
    'get_best_video_codec',
    # GPU utilities
    'get_gpu_memory_info',
    'get_all_gpu_info',
    'get_optimal_device',
    'calculate_optimal_tile_size',
    'get_adaptive_tile_sequence',
    'VRAMMonitor',
    'wait_for_vram',
    'GPUInfo',
    # Disk utilities
    'get_disk_usage',
    'get_directory_size',
    'validate_disk_space',
    'estimate_required_space',
    'DiskSpaceMonitor',
    'cleanup_old_temp_files',
    'DiskUsage',
    'SpaceEstimate',
    # Dependency utilities
    'validate_all_dependencies',
    'check_ffmpeg',
    'check_ffprobe',
    'check_realesrgan',
    'check_ytdlp',
    'check_rife',
    'get_enhancement_backend',
    'compare_versions',
    'DependencyInfo',
    'DependencyReport',
    # Async I/O utilities
    'AsyncFileOperations',
    'AsyncSubprocess',
    'AsyncDownloader',
    'AsyncFrameProcessor',
    'AsyncDownloadResult',
    'AsyncReadResult',
    'AsyncWriteResult',
    'run_pipeline_async',
    'run_async',
    'get_io_executor',
    'shutdown_executor',
    # Model management
    'ModelManager',
    'ModelType',
    'ModelInfo',
    'DownloadProgress',
    'DownloadError',
    'ModelVerificationError',
    'MODEL_REGISTRY',
    'get_model_manager',
    'ProgressCallback',
    # Output management
    'OutputManager',
    'OutputPaths',
    # YouTube utilities
    'YouTubeDownloader',
    'VideoInfo',
    'FormatInfo',
    'ChapterInfo',
    'YouTubeDownloadProgress',
    'YouTubeDownloadError',
    'YouTubeMetadataError',
    'get_youtube_downloader',
    'YouTubeProgressCallback',
    # Cache utilities
    'FrameCache',
    'CacheManager',
    'CacheConfig',
    'CacheEntry',
    'CacheStats',
    'PerceptualHasher',
    'compute_frame_hash',
    'get_cache_manager',
    'get_global_cache',
    'shutdown_cache',
    # Multi-GPU utilities
    'GPUManager',
    'MultiGPUDistributor',
    'MultiGPUInfo',
    'DistributionResult',
    'WorkItem',
    'WorkStealingQueue',
    'LoadBalanceStrategy',
    'detect_gpus',
    'get_optimal_gpu',
    'distribute_frames',
    # Config file utilities
    'ConfigFileManager',
    'ConfigValidationError',
    'get_config_manager',
    'DEFAULT_CONFIG_TEMPLATE',
    'CONFIG_SCHEMA',
    # Progress utilities
    'RichProgress',
    'ProgressTracker',
    'ProgressOutputMode',
    'GPUMetrics',
    'GPUMonitor',
    'StageInfo',
    'RichProgressCallback',
    'LogFileCallback',
    'FallbackProgress',
    'create_progress_tracker',
    'simple_progress',
    'is_rich_available',
    'is_pynvml_available',
    # Structured logging
    'LogConfig',
    'FramewrightLogger',
    'configure_logging',
    'get_logger',
    'set_level',
    'add_file_handler',
    'ProcessingMetricsLog',
    'ErrorAggregator',
    'configure_from_cli',
    'get_cli_args_parser',
    # GPU Memory Optimizer
    'GPUMemoryOptimizer',
    'GPUMemoryStats',
    'OptimizationConfig',
    'MemoryPressure',
    'get_optimizer',
    'get_optimal_batch_size',
    'get_optimal_tile_size',
    'clear_gpu_memory',
    # Thermal Monitoring
    'ThermalMonitor',
    'ThermalState',
    'ThrottleState',
    'ThermalReading',
    'ThermalProfile',
    'get_gpu_temperature',
    'is_gpu_throttling',
    # Model Manager CLI
    'ModelManagerCLI',
    'ModelManagerInfo',
    'ModelCategory',
    # Progress Webhook
    'WebhookType',
    'EventType',
    'WebhookConfig',
    'WebhookEvent',
    'ProgressWebhook',
    'WebhookManager',
    # Job Scheduler
    'JobStatus',
    'ScheduleType',
    'JobConstraints',
    'ScheduledJob',
    'JobScheduler',
    # Power Management
    'PowerAction',
    'PowerState',
    'PowerManager',
    'KeepAwake',
    'prevent_sleep_during',
    # Output Templates
    'VideoMetadata',
    'ProcessingContext',
    'OutputTemplate',
    'TemplateManager',
]
