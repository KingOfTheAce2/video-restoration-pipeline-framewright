# FrameWright Command Modules

This directory contains modular command handlers extracted from the monolithic `cli.py` (originally 4,647 lines).

## Module Organization

| Module | Commands | Purpose |
|--------|----------|---------|
| `core.py` | 6 commands | Main video restoration workflow |
| `analysis.py` | 5 commands | Video analysis and comparison |
| `batch.py` | 3 commands | Batch processing and folder watching |
| `config.py` | 8 commands | Configuration and profile management |
| `presets.py` | 6 commands | Quick presets and workflow wizards |
| `integration.py` | 12 commands | External service integrations |
| `models.py` | 4 commands | AI model management |
| `utils.py` | 7 commands | Utility and helper commands |
| `hardware.py` | 3 commands | Hardware detection and benchmarking |

**Total: 54+ commands across 9 modules**

## Command Reference

### Core Workflow (`core.py`)
- `restore_video` - Full restoration pipeline
- `extract_frames` - Extract frames from video
- `enhance_frames` - Upscale frames with AI
- `reassemble_video` - Reassemble frames into video
- `enhance_audio` - Audio enhancement
- `interpolate_video` - Frame rate interpolation

### Analysis (`analysis.py`)
- `analyze_video` - Comprehensive video analysis
- `analyze_profile_command` - Profile performance
- `analyze_scenes_command` - Scene detection
- `analyze_sync_command` - Audio sync analysis
- `compare_videos` - Side-by-side comparison

### Batch Processing (`batch.py`)
- `batch_process` - Process multiple videos
- `watch_folder` - Watch folder for new videos
- `watch_folder_enhanced` - Enhanced folder watching

### Configuration (`config.py`)
- `config_show` - Show current configuration
- `config_get` - Get config value
- `config_set` - Set config value
- `config_init` - Initialize config
- `profile_save` - Save current profile
- `profile_load` - Load saved profile
- `profile_list` - List available profiles
- `profile_delete` - Delete profile

### Presets (`presets.py`)
- `manage_presets` - Manage preset configurations
- `run_wizard_command` - Interactive setup wizard
- `run_quick_command` - Quick processing
- `run_best_command` - Best quality processing
- `run_archive_command` - Archive-optimized processing
- `run_auto_command` - Automatic mode detection

### Integration (`integration.py`)
- `notify_setup_email_command` - Setup email notifications
- `notify_setup_sms_command` - Setup SMS notifications
- `daemon_start_command` - Start background daemon
- `daemon_stop_command` - Stop background daemon
- `daemon_status_command` - Check daemon status
- `schedule_add_command` - Add scheduled job
- `schedule_list_command` - List scheduled jobs
- `schedule_remove_command` - Remove scheduled job
- `integrate_plex_command` - Plex integration
- `integrate_jellyfin_command` - Jellyfin integration
- `upload_youtube_command` - Upload to YouTube
- `upload_archive_command` - Upload to archive.org

### Model Management (`models.py`)
- `models_list_command` - List available models
- `models_download_command` - Download model weights
- `models_verify_command` - Verify model integrity
- `models_path_command` - Show model paths

### Utilities (`utils.py`)
- `show_video_info` - Show video metadata
- `estimate_command` - Estimate processing time/cost
- `proxy_create_command` - Create proxy video
- `proxy_apply_command` - Apply proxy workflow
- `project_changelog_command` - Generate changelog
- `install_completion` - Install shell completion
- `report_trends_command` - Generate trend reports

### Hardware (`hardware.py`)
- `list_gpus_command` - List available GPUs
- `check_hardware` - Check hardware capabilities
- `run_benchmark` - Run performance benchmark

## Usage in cli.py

Commands are imported and used in the main CLI:

```python
from .commands import (
    restore_video,
    analyze_video,
    batch_process,
    # ... etc
)

# In main():
if args.command == 'restore':
    restore_video(args)
elif args.command == 'analyze':
    analyze_video(args)
# ... etc
```

## Design Principles

1. **Single Responsibility**: Each module handles related commands
2. **Minimal Dependencies**: Imports are localized within functions
3. **Shared Utilities**: Common helpers remain in cli.py or utils
4. **Backward Compatible**: No API changes, just reorganization
5. **Testable**: Each module can be tested independently
