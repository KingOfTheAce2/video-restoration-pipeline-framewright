# FrameWright - Video Restoration Pipeline

A modular, frame-accurate video restoration pipeline for recovering vintage and degraded footage. Optimized for **100-year-old film fragments** and **YouTube source videos**, combining AI upscaling, colorization, watermark removal, and frame interpolation with an intuitive, reproducible workflow.

**Status:** v1.3.0 (Production Ready)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Features Overview

| Feature | Status | Description |
|---------|--------|-------------|
| AI Upscaling | ✅ Ready | Real-ESRGAN 2x/4x enhancement |
| Frame Interpolation | ✅ Ready | RIFE smooth motion interpolation |
| Face Restoration | ✅ Ready | GFPGAN/CodeFormer integration |
| Defect Repair | ✅ Ready | Scratch, dust, grain removal |
| Colorization | ✅ Ready | DeOldify/DDColor B&W colorization |
| Watermark Removal | ✅ Ready | LaMA inpainting |
| Subtitle Removal | ✅ Ready | OCR-based burnt-in subtitle detection & removal |
| Video Stabilization | ✅ Ready | FFmpeg vidstab & OpenCV integration |
| Scene Detection | ✅ Ready | Automatic scene boundary detection |
| Audio Enhancement | ✅ Ready | Noise reduction & clarity improvement |
| Audio Sync | ✅ Ready | AI-powered audio-video synchronization |
| HDR Conversion | ✅ Ready | SDR/HDR format conversion |
| Multi-GPU Support | ✅ Ready | Distribute processing across GPUs |
| Cloud Processing | ✅ Ready | RunPod & Vast.ai integration |
| Web UI | ✅ Ready | Gradio browser interface |
| YouTube Download | ✅ Ready | yt-dlp integration |
| Batch Processing | ✅ Ready | Process multiple videos with queue |
| Streaming Mode | ✅ Ready | Real-time progressive processing |

---

## Quick Start

### For Non-Coders: Web UI

The easiest way to use FrameWright is through the web interface:

```bash
# Install with UI support
pip install framewright[ui]

# Launch the web UI
framewright-ui

# Opens in your browser at http://localhost:7860
```

### Check Your Hardware First

```bash
framewright-check
```

---

## UI Settings Guide

The Web UI provides intuitive controls organized by category:

### Source Settings
| Setting | Description |
|---------|-------------|
| **Video File** | Upload a local video file |
| **YouTube URL** | Paste a YouTube URL to download and process |
| **Output Directory** | Where to save frames and final video (default: `./output/`) |

### Enhancement Options

| Setting | Default | Options |
|---------|---------|---------|
| **Scale Factor** | 4x | 2x, 4x |
| **Model** | Auto-detect | See [Model Selection](#model-selection-anime-vs-real-life) |
| **Enable Colorization** | ☐ Off | Colorize B&W footage with DeOldify/DDColor |
| **Enable Watermark Removal** | ☐ Off | Remove logos/watermarks with LaMA inpainting |
| **Enable Subtitle Removal** | ☐ Off | Remove burnt-in subtitles with OCR detection |
| **Auto-Enhance** | ☑ On | Automatic defect repair and face restoration |

### Frame Interpolation (Smoothing)

| Setting | Default | Description |
|---------|---------|-------------|
| **Enable RIFE** | ☐ Off | Enable AI frame interpolation |
| **Target FPS** | Auto | Target frame rate (24, 30, 48, 60) |
| **Smoothness** | Medium | Low/Medium/High quality |
| **RIFE Model** | rife-v4.6 | Model version for interpolation |

### Output Settings

| Setting | Default | Options |
|---------|---------|---------|
| **Format** | MKV | MKV, MP4, WebM, AVI, MOV |
| **Quality (CRF)** | 18 | 0-51 (lower = better quality) |
| **Generate Report** | ☐ Off | Create improvements.md with details |

---

## Colorization (B&W to Color)

FrameWright can automatically colorize black-and-white footage using AI models.

### Available Models

| Model | Best For | Quality | Speed |
|-------|----------|---------|-------|
| `ddcolor` | **General purpose** - Most footage (default) | High | Medium |
| `deoldify` | Artistic, vintage look | Medium | Fast |

### CLI Usage

```bash
# Enable colorization with DDColor (recommended)
framewright restore --input bw_film.mp4 --colorize --output color_film.mp4

# Use DeOldify for artistic look
framewright restore --input bw_film.mp4 --colorize --colorize-model deoldify --output color_film.mp4
```

---

## Watermark Removal

Remove logos, watermarks, and overlay graphics using AI inpainting.

### Features

- **Auto-detection**: Automatically find common watermark positions
- **Manual mask**: Provide a custom mask image for precise control
- **Region specification**: Define exact coordinates for watermark areas
- **LaMA inpainting**: State-of-the-art inpainting model for seamless removal

### CLI Usage

```bash
# Auto-detect and remove watermarks
framewright restore --input video.mp4 --remove-watermark --watermark-auto-detect --output clean.mp4

# Remove watermark using a mask image (white = watermark area)
framewright restore --input video.mp4 --remove-watermark --watermark-mask mask.png --output clean.mp4

# Specify watermark region manually (x,y,width,height)
framewright restore --input video.mp4 --remove-watermark --watermark-region 1700,50,200,80 --output clean.mp4
```

---

## Burnt-in Subtitle Removal

Remove hard-coded (burnt-in) subtitles that are permanently rendered into video frames.

### Features

- **Multi-engine OCR**: EasyOCR, Tesseract, and PaddleOCR support
- **Configurable regions**: Bottom-third (default), top-quarter, full-frame scan
- **Multi-language support**: English, Chinese, Japanese, Korean, and more
- **LaMA inpainting**: Seamless text removal without artifacts
- **Temporal smoothing**: Consistent removal across frames

### OCR Engines

| Engine | Languages | GPU | Quality |
|--------|-----------|-----|---------|
| `easyocr` | 80+ languages | ✅ GPU | High (default) |
| `paddleocr` | Asian languages | ✅ GPU | High |
| `tesseract` | Latin scripts | ❌ CPU | Medium |
| `auto` | Auto-detect best | - | - |

### CLI Usage

```bash
# Remove burnt-in subtitles (auto-detect OCR engine)
framewright restore --input movie.mp4 --remove-subtitles --output clean.mp4

# Specify OCR engine and languages
framewright restore --input movie.mp4 --remove-subtitles \
    --subtitle-ocr easyocr \
    --subtitle-languages "en,zh" \
    --output clean.mp4

# Scan top of frame (for Chinese/Japanese subtitles)
framewright restore --input anime.mp4 --remove-subtitles \
    --subtitle-region top_quarter \
    --subtitle-languages "ja,zh" \
    --output clean.mp4
```

### Subtitle Regions

| Region | Description | Use Case |
|--------|-------------|----------|
| `bottom_third` | Bottom 33% of frame (default) | Most Western subtitles |
| `bottom_quarter` | Bottom 25% of frame | Smaller subtitle areas |
| `top_quarter` | Top 25% of frame | Chinese/Japanese films |
| `full_frame` | Entire frame | Unknown subtitle position |

---

## Model Selection: Anime vs Real-Life

Choose the right model for your content:

### For Real-Life Footage (Recommended)

| Model | Best For | Speed |
|-------|----------|-------|
| `realesrgan-x4plus` | **General purpose** - old films, photographs | Medium |
| `realesrnet-x4plus` | Fast processing, slightly lower quality | Fast |
| `realesrgan-x4plus` + GFPGAN | Footage with faces | Medium |

### For Anime/Animation

| Model | Best For | Speed |
|-------|----------|-------|
| `realesrgan-x4plus-anime` | **Anime/cartoon upscaling** | Medium |
| `realesr-animevideov3` | **Anime video optimized** - best for animation | Medium |

### Model Download Location

Models are automatically downloaded to:
```
~/.framewright/models/
├── realesrgan/
│   ├── realesrgan-x4plus.pth
│   ├── realesrgan-x4plus-anime.pth
│   └── realesr-animevideov3.pth
├── gfpgan/
│   └── GFPGANv1.4.pth
├── codeformer/
│   └── codeformer.pth
└── rife/
    └── rife-v4.6/
```

**Custom location:**
```bash
framewright restore --input video.mp4 --model-dir /path/to/models/
```

---

## Output Directories

### Default Structure

```
./project_name/
├── source/              # Downloaded/copied source video
├── frames/              # Extracted frames (PNG)
├── enhanced/            # AI-enhanced frames
├── interpolated/        # RIFE interpolated frames (if enabled)
├── output/              # Final restored video
│   └── restored.mkv
└── logs/                # Processing logs
```

### Custom Output Location

```bash
# Specify output directory
framewright restore --input video.mp4 --output-dir /path/to/output/

# Or specific output file
framewright restore --input video.mp4 --output /path/to/restored.mp4
```

---

## Frame Interpolation (Motion Smoothing)

RIFE frame interpolation creates smooth motion by generating intermediate frames.

### When to Use

| Original FPS | Interpolation | Result | Use Case |
|--------------|---------------|--------|----------|
| 12-18 fps | → 24 fps | Subtle smoothing | Silent film era |
| 24 fps | → 48 fps | Cinematic smooth | Modern films |
| 24 fps | → 60 fps | Very smooth | Action footage |
| 30 fps | → 60 fps | Smooth gaming look | General content |

### Smoothness Levels

| Level | Description | Processing Time |
|-------|-------------|-----------------|
| **Low** | Fast, may have minor artifacts | 1x |
| **Medium** | Balanced quality/speed (default) | 2x |
| **High** | Maximum quality, slowest | 4x |

### CLI Usage

```bash
# Enable interpolation with target FPS
framewright restore --input old_film.mp4 \
    --enable-rife \
    --target-fps 48 \
    --rife-model rife-v4.6 \
    --output smooth_film.mp4
```

---

## YouTube Source Workflow

FrameWright is optimized for restoring YouTube videos:

### Basic Usage

```bash
# Web UI: Just paste the URL
# CLI:
framewright restore --url "https://youtube.com/watch?v=VIDEO_ID" --output restored.mp4
```

### Recommended Workflow

1. **Download in highest quality**
   ```bash
   framewright restore --url "https://youtube.com/watch?v=..." \
       --output-dir ./restoration/ \
       --format mkv
   ```

2. **Analyze before processing**
   ```bash
   framewright analyze --url "https://youtube.com/watch?v=..."
   ```

3. **Full restoration with all enhancements**
   ```bash
   framewright restore --url "https://youtube.com/watch?v=..." \
       --scale 4 \
       --auto-enhance \
       --enable-rife --target-fps 48 \
       --output-dir ./output/ \
       --format mkv
   ```

---

## Installation

### Basic Installation

```bash
pip install framewright
```

### With Web UI (Recommended)

```bash
pip install framewright[ui]
```

### Full Installation

```bash
pip install framewright[full]
```

### External Dependencies

```bash
# FFmpeg (required)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html

# Models are downloaded automatically on first use
```

---

## Command Line Reference

### Basic Commands

```bash
# Check hardware compatibility
framewright-check

# Launch Web UI
framewright-ui

# Analyze video (shows recommendations)
framewright analyze --input video.mp4

# Basic restoration
framewright restore --input video.mp4 --output restored.mp4
```

### Full Options

```bash
framewright restore \
    --input old_film.mp4 \          # Local file
    --url "https://youtube.com/..." \ # OR YouTube URL
    --output-dir ./output/ \         # Output directory
    --format mkv \                   # Output format
    --scale 4 \                      # Upscale factor (2 or 4)
    --model realesrgan-x4plus \      # Enhancement model
    --quality 18 \                   # CRF quality (0-51)
    --auto-enhance \                 # Enable auto-enhancement
    --enable-rife \                  # Enable frame interpolation
    --target-fps 48 \                # Target frame rate
    --rife-model rife-v4.6 \         # RIFE model version
    --colorize \                     # Enable B&W colorization
    --colorize-model ddcolor \       # Colorization model
    --remove-watermark \             # Enable watermark removal
    --watermark-auto-detect \        # Auto-detect watermark location
    --remove-subtitles \             # Remove burnt-in subtitles
    --subtitle-ocr auto \            # OCR engine (auto/easyocr/tesseract)
    --subtitle-region bottom_third \ # Subtitle region to scan
    --generate-report                # Create improvements.md
```

---

## Python API

```python
from framewright import VideoRestorer, Config, check_hardware

# Check hardware first
report = check_hardware()
if report.overall_status != "incompatible":

    # Configure
    config = Config(
        project_dir="./my_restoration",
        scale_factor=4,
        model_name="realesrgan-x4plus",  # or "realesrgan-x4plus-anime" for anime
        crf=18,
        enable_checkpointing=True,
        model_dir="~/.framewright/models/",  # Custom model directory
        output_dir="./output/",              # Custom output directory
    )

    # Create restorer
    restorer = VideoRestorer(config)

    # Restore from YouTube
    output = restorer.restore_video(
        source="https://youtube.com/watch?v=...",
        enable_auto_enhance=True,
        enable_rife=True,
        target_fps=48,
    )

    print(f"Restored video: {output}")
```

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

### VRAM Guidelines

| VRAM | Max Resolution | Tile Size |
|------|----------------|-----------|
| 2 GB | 720p | 128 |
| 4 GB | 1080p | 256 |
| 6 GB | 1440p | 384 |
| 8+ GB | 4K | No tiling needed |

---

## Open Source Models Used

All enhancement models are open source:

| Component | Model | License | Source |
|-----------|-------|---------|--------|
| Upscaling | Real-ESRGAN | BSD-3 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| Interpolation | RIFE | MIT | [megvii-research/ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE) |
| Face Restore | GFPGAN | Apache-2.0 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) |
| Face Restore | CodeFormer | S-Lab License | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) |
| Colorization | DeOldify | MIT | [jantic/DeOldify](https://github.com/jantic/DeOldify) |
| Colorization | DDColor | Apache-2.0 | [piddnad/DDColor](https://github.com/piddnad/DDColor) |
| Inpainting | LaMA | Apache-2.0 | [advimman/lama](https://github.com/advimman/lama) |
| OCR | EasyOCR | Apache-2.0 | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Download | yt-dlp | Unlicense | [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Video Processing | FFmpeg | LGPL/GPL | [ffmpeg.org](https://ffmpeg.org/) |
| Web UI | Gradio | Apache-2.0 | [gradio-app/gradio](https://github.com/gradio-app/gradio) |

---

## Project Structure

```
src/framewright/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── restorer.py          # Core VideoRestorer class
├── cli.py               # Command-line interface
├── ui.py                # Web UI (Gradio)
├── hardware.py          # Hardware compatibility checks
├── checkpoint.py        # Checkpointing & resume system
├── dry_run.py           # Dry run mode for testing
├── watch.py             # Watch mode for folder monitoring
├── exceptions.py        # Custom exception classes
├── processors/
│   ├── analyzer.py         # Content analysis & detection
│   ├── adaptive_enhance.py # Adaptive enhancement
│   ├── defect_repair.py    # Scratch/dust/grain removal
│   ├── face_restore.py     # GFPGAN/CodeFormer
│   ├── interpolation.py    # RIFE frame interpolation
│   ├── colorization.py     # DeOldify/DDColor
│   ├── watermark_removal.py # LaMA inpainting
│   ├── subtitle_removal.py # OCR + inpainting
│   ├── subtitles.py        # Subtitle extraction
│   ├── stabilization.py    # Video stabilization
│   ├── scene_detection.py  # Scene boundary detection
│   ├── hdr_conversion.py   # HDR/SDR conversion
│   ├── audio.py            # Audio processing
│   ├── audio_enhance.py    # Audio enhancement
│   ├── audio_sync.py       # Audio-video sync
│   ├── streaming.py        # Streaming mode
│   ├── preview.py          # Preview generation
│   └── advanced_models.py  # Extended model support
├── cloud/
│   ├── base.py             # Cloud provider base class
│   ├── runpod.py           # RunPod integration
│   ├── vastai.py           # Vast.ai integration
│   └── storage.py          # Cloud storage utilities
├── benchmarks/
│   ├── benchmark_suite.py  # Performance benchmarks
│   └── profiler.py         # Code profiling tools
└── utils/
    ├── gpu.py              # GPU utilities
    ├── multi_gpu.py        # Multi-GPU distribution
    ├── disk.py             # Disk management
    ├── ffmpeg.py           # FFmpeg wrapper
    ├── model_manager.py    # Model download & cache
    ├── output_manager.py   # Output directory config
    ├── cache.py            # Result caching
    ├── progress.py         # Progress tracking
    ├── logging.py          # Structured logging
    ├── security.py         # Input validation
    ├── youtube.py          # YouTube download
    ├── config_file.py      # Config file handling
    ├── async_io.py         # Async I/O utilities
    └── dependencies.py     # Dependency checks

tests/                   # Unit & integration tests
docs/                    # Documentation
docker/                  # Docker configurations
.github/                 # GitHub Actions CI/CD
```

---

## Technical Documentation

### For Developers

See the [Technical Guide](docs/technical-guide.md) for:
- API reference
- Plugin architecture
- Custom processor development
- Testing and benchmarking

### Configuration Options

All options can be set via:
1. CLI arguments
2. Config file (`~/.framewright/config.yaml`)
3. Environment variables (`FRAMEWRIGHT_*`)
4. Python API

---

## License

This project is licensed under the **MIT License** - see [LICENSE.md](LICENSE.md) for details.

This project builds upon many excellent open source projects. See [LICENSE.md](LICENSE.md) for full acknowledgements.

---

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

---

## Acknowledgments

Special thanks to the creators of the open source models and tools that make this project possible:

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) by Megvii Research
- [GFPGAN](https://github.com/TencentARC/GFPGAN) by Tencent ARC Lab
- [CodeFormer](https://github.com/sczhou/CodeFormer) by Shangchen Zhou
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) community
- [FFmpeg](https://ffmpeg.org/) project
- [Gradio](https://gradio.app/) by Hugging Face
