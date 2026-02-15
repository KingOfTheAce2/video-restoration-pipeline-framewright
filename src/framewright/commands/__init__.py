"""Command modules for FrameWright CLI.

Organized command handlers split from the monolithic cli.py:
- core: Main restoration workflow commands
- analysis: Video analysis and profiling
- batch: Batch processing and folder watching
- config: Configuration and profile management
- presets: Quick presets and wizards
- integration: External service integrations
- models: Model management
- utils: Utility commands
- hardware: Hardware detection and benchmarking
"""

# Core workflow commands
from .core import (
    restore_video,
    extract_frames,
    enhance_frames,
    reassemble_video,
    enhance_audio,
    interpolate_video,
)

# Analysis commands
from .analysis import (
    analyze_video,
    analyze_profile_command,
    analyze_scenes_command,
    analyze_sync_command,
    compare_videos,
)

# Batch processing
from .batch import (
    batch_process,
    watch_folder,
    watch_folder_enhanced,
)

# Configuration and profiles
from .config import (
    config_show,
    config_get,
    config_set,
    config_init,
    profile_save,
    profile_load,
    profile_list,
    profile_delete,
)

# Preset commands
from .presets import (
    manage_presets,
    run_wizard_command,
    run_quick_command,
    run_best_command,
    run_archive_command,
    run_auto_command,
)

# Integration commands
from .integration import (
    notify_setup_email_command,
    notify_setup_sms_command,
    daemon_start_command,
    daemon_stop_command,
    daemon_status_command,
    schedule_add_command,
    schedule_list_command,
    schedule_remove_command,
    integrate_plex_command,
    integrate_jellyfin_command,
    upload_youtube_command,
    upload_archive_command,
)

# Model management
from .models import (
    models_list_command,
    models_download_command,
    models_verify_command,
    models_path_command,
)

# Utility commands
from .utils import (
    show_video_info,
    estimate_command,
    proxy_create_command,
    proxy_apply_command,
    project_changelog_command,
    install_completion,
    report_trends_command,
)

# Hardware commands
from .hardware import (
    list_gpus_command,
    check_hardware,
    run_benchmark,
)

__all__ = [
    # Core
    'restore_video',
    'extract_frames',
    'enhance_frames',
    'reassemble_video',
    'enhance_audio',
    'interpolate_video',
    # Analysis
    'analyze_video',
    'analyze_profile_command',
    'analyze_scenes_command',
    'analyze_sync_command',
    'compare_videos',
    # Batch
    'batch_process',
    'watch_folder',
    'watch_folder_enhanced',
    # Config
    'config_show',
    'config_get',
    'config_set',
    'config_init',
    'profile_save',
    'profile_load',
    'profile_list',
    'profile_delete',
    # Presets
    'manage_presets',
    'run_wizard_command',
    'run_quick_command',
    'run_best_command',
    'run_archive_command',
    'run_auto_command',
    # Integration
    'notify_setup_email_command',
    'notify_setup_sms_command',
    'daemon_start_command',
    'daemon_stop_command',
    'daemon_status_command',
    'schedule_add_command',
    'schedule_list_command',
    'schedule_remove_command',
    'integrate_plex_command',
    'integrate_jellyfin_command',
    'upload_youtube_command',
    'upload_archive_command',
    # Models
    'models_list_command',
    'models_download_command',
    'models_verify_command',
    'models_path_command',
    # Utils
    'show_video_info',
    'estimate_command',
    'proxy_create_command',
    'proxy_apply_command',
    'project_changelog_command',
    'install_completion',
    'report_trends_command',
    # Hardware
    'list_gpus_command',
    'check_hardware',
    'run_benchmark',
]
