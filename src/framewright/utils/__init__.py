"""
FrameWright Utilities Package
Utility functions and helpers for video processing.
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
]
