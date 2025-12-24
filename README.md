# FrameWright - Video Restoration Pipeline

A modular, frame-accurate video restoration pipeline for recovering vintage and degraded footage. Optimized for **100-year-old film fragments**, combining AI upscaling, audio restoration, and optional frame interpolation with a craft-first, reproducible workflow.

## ✨ What's New in v1.3.0

- 🤖 **Auto-Enhancement** - Fully automated restoration with content detection
- 🔍 **Video Analysis** - Pre-scan for optimal settings recommendations
- 👤 **Face Restoration** - GFPGAN/CodeFormer integration for portrait enhancement
- 🩹 **Defect Repair** - Automatic scratch, dust, and grain removal
- 🎬 **RIFE Integration** - Frame interpolation with auto-FPS detection
- 📺 **Preview Mode** - Inspect results before final assembly

### Previous Updates (v1.2.0)
- 🖥️ **Web UI** - User-friendly browser interface for non-coders
- 🔧 **Hardware Check** - Verify your system can run the pipeline
- 📊 **Progress Tracking** - ETA calculation and detailed metrics
- 🎵 **Audio Analysis** - Silence detection and quality analysis
- 💾 **Checkpointing** - Resume interrupted processing
- 🛡️ **Error Recovery** - Automatic retry with backoff

---

## 🚀 Quick Start

### For Non-Coders: Web UI

The easiest way to use FrameWright is through the web interface:

```bash
# Install with UI support
pip install framewright[ui]

# Launch the web UI
framewright-ui

# Opens in your browser at http://localhost:7860
```

**Features of the Web UI:**
- 📁 Upload videos or paste YouTube URLs
- ⚙️ Simple settings with sensible defaults
- 📊 Real-time progress tracking
- 🔧 Hardware compatibility check built-in

### Check Your Hardware First

Before processing, verify your system is compatible:

```bash
# Quick hardware check
framewright-check
```

This shows:
- GPU detection and VRAM
- RAM availability
- Disk space
- Missing dependencies
- Recommendations for your setup

**Example output:**
```
============================================================
  FrameWright Hardware Compatibility Report
============================================================

📊 SYSTEM INFORMATION
────────────────────────────────────────
  OS:        Linux 5.15.0
  Python:    3.11.0
  CPU:       AMD Ryzen 9 5900X
  Cores:     24
  RAM:       64.0 GB total, 58.2 GB available

🎮 GPU INFORMATION
────────────────────────────────────────
  GPU:       NVIDIA GeForce RTX 3080
  VRAM:      10240 MB total, 9500 MB free
  CUDA:      Yes
  Max Res:   4K (3840x2160)

💡 RECOMMENDATIONS
────────────────────────────────────────
  • Your system is ready for 4K processing!

────────────────────────────────────────
  Overall Status: ✅ READY
────────────────────────────────────────
```

---

## How It Works

The pipeline processes video through discrete stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FRAMEWRIGHT PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ DOWNLOAD │ -> │ ANALYZE  │ -> │ EXTRACT  │ -> │ ENHANCE  │              │
│  │ (yt-dlp) │    │ (Auto)   │    │ (FFmpeg) │    │(Real-ESRGAN)            │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │              │                │               │                     │
│       v              v                v               v                     │
│   source.mkv    content type    frames/*.png    enhanced/*.png             │
│                 degradation     audio.wav                                   │
│                 recommendations                                             │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ DEFECT   │ -> │  FACE    │ -> │ UPSCALE  │ -> │REASSEMBLE│              │
│  │ REPAIR   │    │ RESTORE  │    │ (RIFE)   │    │ (FFmpeg) │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│  [Auto-Enhance]  [GFPGAN]       [Optional]       -> restored.mkv           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Basic Installation

```bash
pip install framewright
```

### With Web UI

```bash
pip install framewright[ui]
```

### Full Installation (Recommended)

```bash
pip install framewright[full]
```

### External Dependencies

FrameWright requires these external tools:

```bash
# FFmpeg (required)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html

# Real-ESRGAN (required for enhancement)
# Download from: https://github.com/xinntao/Real-ESRGAN/releases

# RIFE (optional, for frame interpolation)
pip install rife-ncnn-vulkan
```

---

## Usage

### Web UI (Recommended for Beginners)

```bash
# Launch the UI
framewright-ui

# With a public shareable link
framewright-ui --share

# On a different port
framewright-ui --port 8080
```

### Command Line

```bash
# Fully automated restoration (RECOMMENDED for old films)
framewright restore --input old_film.mp4 --output restored.mp4 --auto-enhance

# Analyze video first to see recommendations
framewright analyze --input video.mp4

# Restore from YouTube URL
framewright restore --url "https://youtube.com/watch?v=VIDEO_ID" --output restored.mp4

# Full options for archival quality
framewright restore \
    --url "https://youtube.com/watch?v=VIDEO_ID" \
    --scale 4 \
    --quality 15 \
    --auto-enhance \
    --enable-rife --target-fps 48 \
    --output ./restored/

# Auto-enhance with custom sensitivity
framewright restore --input video.mp4 --output enhanced.mp4 \
    --auto-enhance --scratch-sensitivity 0.7 --grain-reduction 0.5

# Preview frames before final assembly
framewright restore --input video.mp4 --output enhanced.mp4 --preview
```

### Python API

```python
from framewright import VideoRestorer, Config, check_hardware, print_hardware_report

# First, check hardware
report = check_hardware()
print(print_hardware_report(report))

if report.overall_status != "incompatible":
    # Configure
    config = Config(
        project_dir="./my_restoration",
        scale_factor=4,
        model_name="realesrgan-x4plus",
        crf=18,
        enable_checkpointing=True,  # Resume if interrupted
    )

    # Create restorer
    restorer = VideoRestorer(config)

    # Run full pipeline
    output = restorer.restore_video(
        source="https://youtube.com/watch?v=...",
        enhance_audio=True,
    )

    print(f"Restored video: {output}")
```

---

## Configuration

### Auto-Enhancement (NEW in v1.3.0)

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--auto-enhance` | flag | off | Enable fully automated enhancement pipeline |
| `--scratch-sensitivity` | `0-1` | `0.5` | Scratch detection sensitivity |
| `--dust-sensitivity` | `0-1` | `0.5` | Dust/debris detection sensitivity |
| `--grain-reduction` | `0-1` | `0.3` | Film grain reduction strength |
| `--no-face-restore` | flag | off | Disable automatic face restoration |
| `--no-defect-repair` | flag | off | Disable automatic defect repair |

**Auto-enhancement automatically:**
- Detects content type (faces, animation, landscapes, etc.)
- Detects degradation (noise, grain, scratches, blur)
- Applies targeted repairs (scratches, dust, grain removal)
- Restores faces using GFPGAN/CodeFormer (when faces detected)
- Adjusts parameters based on content analysis

### RIFE Frame Interpolation

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--enable-rife` | flag | off | Enable RIFE frame interpolation |
| `--target-fps` | number | auto | Target frame rate (e.g., 48, 60) |
| `--rife-model` | `rife-v2.3`, `rife-v4.0`, `rife-v4.6` | `rife-v4.6` | RIFE model version |

### Video Enhancement

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--scale` | `2`, `4` | `4` | Upscaling factor. 4x for heavily degraded footage |
| `--model` | See below | `realesrgan-x4plus` | AI model for enhancement |
| `--quality` | `0-51` | `18` | CRF quality (lower = better) |
| `--preview` | flag | off | Preview frames before final reassembly |

**Available Models:**
```
realesrgan-x4plus          # General purpose (recommended for film)
realesrgan-x4plus-anime    # Anime/animation
realesr-animevideov3       # Anime video optimized
realesrnet-x4plus          # Faster, slightly lower quality
```

### Audio Enhancement

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--highpass` | Hz | `80` | Remove rumble below this frequency |
| `--lowpass` | Hz | `12000` | Remove hiss above this frequency |
| `--noise-reduction` | dB | `20` | Noise reduction strength |

---

## Hardware Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8 GB
- **GPU**: None (CPU mode, very slow)
- **Disk**: 20 GB free

### Recommended
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **GPU**: NVIDIA with 4+ GB VRAM
- **Disk**: 100 GB free (SSD preferred)

### Optimal (4K Processing)
- **CPU**: 12+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3080 or better (8+ GB VRAM)
- **Disk**: 500 GB free SSD

### GPU VRAM Guidelines

| VRAM | Max Resolution | Notes |
|------|----------------|-------|
| 2 GB | 720p | Use tile size 128 |
| 4 GB | 1080p | Use tile size 256 |
| 6 GB | 1440p | Use tile size 384 |
| 8+ GB | 4K | No tiling needed |

---

## Tips for 100-Year-Old Film

### Recommended Settings (Fully Automated)

```bash
# Best for old film: use --auto-enhance for intelligent processing
framewright restore \
    --input old_film.mp4 \
    --scale 4 \
    --auto-enhance \
    --enable-rife --target-fps 48 \
    --quality 15 \
    --output restored_film.mkv
```

### Analyze First, Then Restore

```bash
# Step 1: Analyze to see what's detected
framewright analyze --input old_film.mp4

# Output shows:
#   Content: FACE_PORTRAIT (60% of frames have faces)
#   Degradation: MODERATE (film grain, light scratches)
#   Recommended: 4x scale, face restore, defect repair
#   Suggested command: framewright restore --input old_film.mp4 --output restored.mp4 --scale 4 --auto-enhance

# Step 2: Run with recommended settings
framewright restore --input old_film.mp4 --output restored.mp4 --auto-enhance
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Heavy grain/noise | Use `--auto-enhance` (auto-detects and adjusts) |
| Flickering brightness | Auto-enhance includes deflicker detection |
| Missing frames | `--enable-rife --target-fps 48` recreates frames |
| Scratches/dust | `--auto-enhance` applies targeted defect repair |
| Blurry faces | Auto-enhance uses GFPGAN when faces detected |
| Speed too fast/slow | Source FPS auto-detected; use `--target-fps` for RIFE |
| No audio | Pipeline handles silent films gracefully |

### Frame Rate Reference

| Era | Original FPS | Recommended Target |
|-----|--------------|-------------------|
| 1890s-1910s | 14-18 | 24 (subtle smoothing) |
| 1920s | 18-24 | 24-30 |
| 1930s+ (sound) | 24 | 48 or 60 |

---

## Robustness Features (v1.1+)

FrameWright includes enterprise-grade reliability features:

### Checkpointing & Resume
```python
config = Config(
    enable_checkpointing=True,
    checkpoint_interval=100,  # Save every 100 frames
)
# If interrupted, restart with same config to resume
```

### Error Recovery
- Automatic retry with exponential backoff
- VRAM overflow recovery (reduces tile size)
- Disk space monitoring

### Quality Validation
- PSNR/SSIM quality metrics
- Artifact detection (halos, tiling, banding)
- Temporal consistency checking
- Audio silence/clipping detection

### Metrics & Progress
```python
from framewright import ProcessingMetrics, ProgressReporter

metrics = ProcessingMetrics(total_frames=1000)
progress = ProgressReporter(total_frames=1000)

# During processing
progress.update(frame_num=100, frame_time_ms=50)
# Output: [████████░░░░░░] 10.0% (100/1000) ETA: 45s | 20.0 fps

# Export metrics
metrics.export_json(Path("metrics.json"))
```

---

## Project Structure

```
src/framewright/
├── __init__.py          # Package exports
├── config.py            # Configuration
├── restorer.py          # Core VideoRestorer class
├── cli.py               # Command-line interface
├── ui.py                # Web UI (Gradio)
├── hardware.py          # Hardware compatibility
├── checkpoint.py        # Checkpointing system
├── errors.py            # Error handling
├── validators.py        # Quality validation
├── metrics.py           # Progress & metrics
├── processors/          # Enhancement processors (NEW)
│   ├── analyzer.py      # Content & degradation detection
│   ├── adaptive_enhance.py  # Adaptive enhancement pipeline
│   ├── defect_repair.py # Scratch, dust, grain removal
│   ├── face_restore.py  # GFPGAN/CodeFormer integration
│   ├── interpolation.py # RIFE frame interpolation
│   └── audio.py         # Audio processing
└── utils/
    ├── gpu.py           # GPU/VRAM utilities
    ├── disk.py          # Disk space utilities
    ├── ffmpeg.py        # FFmpeg utilities
    └── dependencies.py  # Dependency checking

tests/                   # 270+ unit tests
```

---

## API Reference

### Core Classes

```python
from framewright import (
    # Core
    VideoRestorer,
    Config,

    # Hardware
    check_hardware,
    print_hardware_report,
    HardwareReport,

    # Metrics
    ProcessingMetrics,
    ProgressReporter,

    # Validation
    validate_frame_integrity,
    validate_audio_stream,
    analyze_audio_quality,
    detect_artifacts,

    # Errors
    VideoRestorerError,
    TransientError,
    VRAMError,
)

# Processors (NEW in v1.3.0)
from framewright.processors import (
    # Analysis
    FrameAnalyzer,
    VideoAnalysis,
    ContentType,
    DegradationType,

    # Enhancement
    AdaptiveEnhancer,
    AutoEnhancePipeline,

    # Defect Repair
    DefectDetector,
    DefectRepairer,
    AutoDefectProcessor,

    # Face Restoration
    FaceRestorer,
    FaceModel,

    # Interpolation
    FrameInterpolator,
)
```

### Auto-Enhancement API

```python
from framewright import VideoRestorer, Config

# Configure with auto-enhancement enabled
config = Config(
    project_dir="./restoration",
    scale_factor=4,
    enable_auto_enhance=True,      # Enable auto-enhancement
    auto_detect_content=True,       # Detect content type
    auto_defect_repair=True,        # Auto repair scratches/dust
    auto_face_restore=True,         # Auto restore faces
    scratch_sensitivity=0.5,        # Sensitivity tuning
    grain_reduction=0.3,
)

restorer = VideoRestorer(config)

# Run fully automated restoration
output = restorer.restore_video(
    source="old_film.mp4",
    enable_auto_enhance=True,       # Can also enable per-run
    enable_rife=True,
    target_fps=48,
)
```

---

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI upscaling
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) - Frame interpolation
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face restoration
- [CodeFormer](https://github.com/sczhou/CodeFormer) - Face restoration
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video downloading
- [FFmpeg](https://ffmpeg.org/) - Video/audio processing
- [Gradio](https://gradio.app/) - Web UI framework
