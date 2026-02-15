# FrameWright - Video Restoration Pipeline

A modular, frame-accurate video restoration pipeline for recovering vintage and degraded footage. Optimized for **100-year-old film fragments** and **YouTube source videos**, combining AI upscaling, colorization, watermark removal, and frame interpolation with an intuitive, reproducible workflow.

**Optimized for:** RTX 5090 (32GB VRAM) + 192GB RAM | Works on any hardware from CPU-only to data center GPUs

**Status:** v1.0.0 (Stable Release - Fully Local, No API Required)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic--2.0-blue.svg)](LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Key Highlights

- **100% Local** - No API keys, no cloud services, no internet required
- **Apple-like CLI** - Intuitive commands that just work (`framewright video.mp4`)
- **Auto GPU Detection** - Automatically optimizes settings for your hardware
- **True Resume** - Frame-level checkpointing survives crashes and restarts
- **Missing Frame Generation** - AI reconstructs lost footage (not just interpolation)
- **Film Stock Detection** - Identifies Kodachrome, Ektachrome, Technicolor and applies era-correct colors
- **Subtitle Preservation** - Extracts burnt-in subtitles via OCR before removal (for AI translation)
- **One-Command Scan** - `framewright scan video.mp4` detects all issues and suggests fixes

---

## Features Overview

| Feature | Status | Description |
|---------|--------|-------------|
| AI Upscaling | ✅ Ready | Real-ESRGAN 2x/4x enhancement |
| Frame Deduplication | ✅ Ready | Remove duplicate frames from old films (18fps→25fps) |
| Frame Interpolation | ✅ Ready | RIFE smooth motion interpolation |
| Face Restoration | ✅ Ready | GFPGAN/CodeFormer integration |
| Defect Repair | ✅ Ready | Scratch, dust, grain removal |
| Temporal Denoising | ✅ Ready | Multi-frame denoising with optical flow |
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
| GPU Requirement | ✅ Ready | Prevents CPU fallback to avoid system freeze |

### Smart Processing Features

| Feature | Status | Description |
|---------|--------|-------------|
| GPU Memory Optimizer | ✅ Ready | Dynamic batch sizing based on available VRAM |
| True Checkpoint Resume | ✅ Ready | Frame-level progress tracking with crash recovery |
| Preset Library | ✅ Ready | Community-shared presets (VHS, 8mm, 16mm, 35mm, etc.) |
| HTML Comparison Viewer | ✅ Ready | Interactive before/after slider in browser |
| Batch Templates | ✅ Ready | Reusable templates for multiple folders |
| Interlacing Detection | ✅ Ready | TFF/BFF field order and telecine detection |
| Letterbox Detection | ✅ Ready | Automatic black bar cropping |
| Film Stock Detection | ✅ Ready | Kodachrome, Ektachrome, Technicolor identification |
| Audio Sync Verification | ✅ Ready | AI-powered drift detection and correction |
| Subtitle OCR Extraction | ✅ Ready | Extract burnt-in subtitles before removal |
| Watch Folder Mode | ✅ Ready | Auto-process new files in monitored folders |
| A/B Testing | ✅ Ready | Compare different restoration settings |
| User Profiles | ✅ Ready | Save and load custom configurations |
| Video Scanning | ✅ Ready | Comprehensive issue detection in one command |

### AI Enhancement Features

| Feature | Status | Description |
|---------|--------|-------------|
| TAP Neural Denoising | ✅ Ready | Restormer/NAFNet temporal denoising (+4-6 dB PSNR) |
| AESRGAN Face Enhancement | ✅ Ready | Attention-enhanced face restoration |
| Diffusion Super-Resolution | ✅ Ready | Upscale-A-Video for state-of-art quality |
| QP Artifact Removal | ✅ Ready | Codec-aware compression damage repair |
| SwinTExCo Colorization | ✅ Ready | Reference-based exemplar colorization |
| Missing Frame Generation | ✅ Ready | AI reconstruction for damaged sections |
| Cross-Attention Temporal | ✅ Ready | Transformer-based flicker reduction |
| Film Grain Management | ✅ Ready | Preserve, remove, match, or synthesize grain |
| Film Flicker Removal | ✅ Ready | Luminance flicker correction for old film |
| Scratch & Dust Removal | ✅ Ready | Automatic inpainting of film defects |
| Gate Weave Correction | ✅ Ready | Frame stabilization for projector weave |
| Color Fade Restoration | ✅ Ready | Restore faded colors in aged film |
| Telecine Removal | ✅ Ready | Inverse telecine (IVTC) for 3:2 pulldown |
| Audio Restoration Suite | ✅ Ready | Comprehensive audio repair & enhancement |
| Real-Time Preview | ✅ Ready | Live before/after comparison |
| Batch Queue Processor | ✅ Ready | Priority-based job scheduling |
| QA Report Generator | ✅ Ready | Comprehensive quality reports (HTML/JSON) |
| Export Presets | ✅ Ready | Platform-optimized encoding profiles |
| Project Management | ✅ Ready | Save/load restoration projects |

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

#### Web UI Features

| Feature | Description |
|---------|-------------|
| **Video Analysis** | Auto-detects resolution, FPS, duration, codec, and audio |
| **Time Estimation** | Shows estimated processing time based on hardware and settings |
| **Hardware Detection** | Displays GPU/CPU mode, VRAM, and backend (CUDA/Vulkan) |
| **Checkpoint Resume** | Auto-detects and resumes interrupted restorations |
| **Live Progress Log** | Real-time processing log with detailed status |
| **Stop Button** | Cancel processing while preserving progress log |

#### GPU Support

| GPU Vendor | Backend | Status |
|------------|---------|--------|
| NVIDIA | CUDA / TensorRT / Vulkan (NCNN) | ✅ Full support (FP16, INT8 acceleration) |
| AMD | ROCm / HIP / Vulkan (NCNN) | ✅ Full support |
| Intel | oneAPI / OpenVINO / Vulkan (NCNN) | ✅ Full support |
| Apple Silicon | CoreML / Metal / ANE | ✅ Full support (M1-M4) |
| CPU | OpenCV / NumPy fallback | ⚠️ Slower (blocked by default) |

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

## Simple CLI (Apple-like Experience)

v1.0 introduces simplified commands that "just work" with intelligent auto-detection:

```bash
# Smart auto-restore (analyzes video and applies optimal settings)
framewright video.mp4

# Quick mode - fast processing with good quality
framewright quick video.mp4

# Best mode - maximum quality (slower)
framewright best video.mp4

# Archive mode - optimized for historical footage
framewright archive video.mp4

# Interactive wizard - guided step-by-step setup
framewright wizard

# Analyze video and show recommendations
framewright analyze video.mp4
```

### Archive Footage with Colorization

```bash
# Colorize B&W footage using reference images
framewright archive old_film.mp4 --colorize ref1.jpg ref2.jpg ref3.jpg
```

### Additional Commands

```bash
# Watch folder - auto-process new files
framewright watch ./input_folder --output ./restored

# Queue management
framewright queue add video1.mp4 video2.mp4 --preset archive
framewright queue list
framewright queue start
framewright queue clear

# User profiles
framewright profile save my_settings --from-config current.json
framewright profile load my_settings
framewright profile list

# Subtitle extraction (preserves before removal)
framewright extract-subs movie.mp4 --output subtitles.srt --languages en,es

# A/B testing (compare settings)
framewright ab-test video.mp4 --config-a fast.json --config-b quality.json

# Watermark removal
framewright remove-watermark video.mp4 --position top-right --output clean.mp4

# Deinterlacing
framewright deinterlace video.mp4 --method yadif --output progressive.mp4

# Black bar cropping
framewright crop-bars video.mp4 --output cropped.mp4

# Film stock detection
framewright detect-stock old_film.mp4

# Audio sync check
framewright check-sync video.mp4

# Comprehensive scan (all analyzers at once)
framewright scan video.mp4
# Shows: interlacing, letterbox, film stock, audio sync issues
# Suggests: fix commands for each detected issue
```

---

## Ultimate Preset (Maximum Quality)

The `ultimate` preset enables all cutting-edge AI features for the highest quality restoration:

```bash
# CLI
framewright restore input.mp4 --preset ultimate -o output.mp4

# With selective features
framewright restore input.mp4 \
  --tap-denoise \
  --diffusion-sr \
  --face-model aesrgan \
  --temporal-method hybrid \
  -o output.mp4
```

### Ultimate Preset Processing Pipeline

1. **QP Artifact Removal** - Clean compression damage first
2. **TAP Denoising** - Neural denoise with Restormer/NAFNet
3. **Missing Frame Generation** - Fill gaps before enhancement
4. **Diffusion Super-Resolution** - Maximum quality upscaling
5. **AESRGAN Face Enhancement** - Attention-enhanced face restoration
6. **RIFE Interpolation** - Increase framerate if needed
7. **Cross-Attention Temporal** - Final flicker reduction
8. **SwinTExCo Colorization** - Colorize if B&W + references provided

### Hardware Requirements for Ultimate Preset

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12 GB | 24+ GB |
| System RAM | 32 GB | 64+ GB |
| Storage | SSD 500GB | NVMe 1TB+ |

---

## GPU Memory Optimizer

FrameWright automatically adjusts processing parameters based on available VRAM to prevent out-of-memory errors while maximizing throughput.

### Features

| Feature | Description |
|---------|-------------|
| **Real-time VRAM Monitoring** | Tracks memory usage during processing |
| **Dynamic Batch Sizing** | Automatically adjusts batch size based on available memory |
| **Tile Size Optimization** | Selects optimal tile size for your GPU |
| **Model Offloading** | Offloads models when memory pressure is high |
| **Multi-GPU Balancing** | Distributes work across multiple GPUs |

### VRAM Tier Settings

| VRAM | Tile Size | Batch Size | Notes |
|------|-----------|------------|-------|
| 32GB+ | No tiling | 8-16 | Full resolution processing |
| 24GB | No tiling | 4-8 | Most models fit entirely |
| 16GB | 512px | 2-4 | Balanced performance |
| 12GB | 384px | 1-2 | Good for 4K output |
| 8GB | 256px | 1 | Suitable for 1080p |
| 6GB | 192px | 1 | Entry-level GPU |
| 4GB | 128px | 1 | Minimum viable |

### Python API

```python
from framewright.utils.gpu_memory_optimizer import GPUMemoryOptimizer

optimizer = GPUMemoryOptimizer()

# Get optimal settings for your hardware
config = optimizer.get_processing_config(
    frame_size=(1920, 1080),
    models=["realesrgan", "face_restore"],
    scale_factor=4
)
print(f"Batch size: {config['batch_size']}")
print(f"Tile size: {config['tile_size']}")
print(f"Use FP16: {config['half_precision']}")
```

---

## Checkpoint & Resume

True frame-level checkpointing allows interrupted jobs to resume exactly where they left off, even after crashes or power failures.

### Features

| Feature | Description |
|---------|-------------|
| **Frame-Level Tracking** | Saves progress for each processed frame |
| **Atomic Writes** | Crash-safe checkpoint storage |
| **Video Verification** | Hash-based verification to ensure same video |
| **Multi-Stage Support** | Tracks progress across processing stages |
| **Auto Cleanup** | Removes checkpoint files after completion |

### How It Works

```
1. Processing starts → checkpoint.json created
2. Every 100 frames → checkpoint updated atomically
3. Crash/interrupt → partial progress saved
4. Resume → reads checkpoint, skips completed frames
5. Complete → checkpoint files cleaned up
```

### CLI Usage

```bash
# Resume is automatic - just run the same command again
framewright restore video.mp4 --output restored/

# If interrupted, running the same command resumes from checkpoint
framewright restore video.mp4 --output restored/
# "Resuming from checkpoint: 45.2% complete (1356/3000 frames)"
```

### Python API

```python
from framewright.persistence.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    video_path="video.mp4",
    output_dir="./restored",
    total_frames=3000,
    checkpoint_interval=100  # Save every 100 frames
)

# Get remaining frames to process
for frame_idx in manager.get_remaining_frames():
    result = process_frame(frame_idx)
    manager.mark_frame_complete(frame_idx)

# Summary
print(manager.get_summary())
```

---

## Preset Library

Community-shared presets for common restoration scenarios. Includes built-in presets for VHS, 8mm, 16mm, 35mm film, and more.

### Built-in Presets

| Preset | Use Case | Description |
|--------|----------|-------------|
| `vhs_home_movie` | Home recordings | Light denoising, color correction |
| `vhs_commercial` | Commercial VHS | Stronger enhancement, sharpening |
| `film_8mm` | 8mm home movies | Grain management, stabilization |
| `film_16mm` | 16mm footage | Balanced restoration |
| `film_35mm_archive` | 35mm archival | Maximum quality preservation |
| `animation_cel` | Cel animation | Clean lines, vibrant colors |
| `broadcast_sd` | SD broadcast | Deinterlacing, upscaling |
| `youtube_compressed` | YouTube downloads | Artifact removal |
| `surveillance` | Security footage | Clarity enhancement |
| `vintage_photo` | Photo slideshows | Photo-optimized processing |

### CLI Usage

```bash
# List available presets
framewright preset list

# Apply preset
framewright restore video.mp4 --preset vhs_home_movie

# Create custom preset from config
framewright preset create my_preset --from-config settings.json

# Export preset for sharing
framewright preset export my_preset --output my_preset.json

# Import shared preset
framewright preset import shared_preset.json
```

### Python API

```python
from framewright.presets.preset_library import PresetLibrary

library = PresetLibrary()

# List presets
for preset in library.list_presets():
    print(f"{preset['name']}: {preset['description']}")

# Get preset config
config = library.get_preset("film_35mm_archive")

# Create custom preset
library.create_preset(
    name="my_vhs_preset",
    config={
        "denoise_strength": 0.4,
        "sharpen": 0.3,
        "color_correct": True
    },
    description="My custom VHS settings"
)
```

---

## HTML Comparison Viewer

Interactive before/after comparison viewer that opens in your browser with a draggable slider.

### Features

| Feature | Description |
|---------|-------------|
| **Slider Comparison** | Drag to compare before/after |
| **Quality Metrics** | Shows PSNR, SSIM, frame count |
| **Zoom Support** | Click to zoom on details |
| **Self-Contained** | Single HTML file, no dependencies |
| **Shareable** | Send HTML file to anyone |

### CLI Usage

```bash
# Generate comparison after restoration
framewright compare original.mp4 restored.mp4 --output comparison.html

# Open automatically in browser
framewright compare original.mp4 restored.mp4 --open
```

### Python API

```python
from framewright.export.comparison_viewer import ComparisonViewer

viewer = ComparisonViewer()

# Generate comparison
viewer.generate(
    original_path="original.mp4",
    restored_path="restored.mp4",
    output_html="comparison.html",
    frame_indices=[0, 100, 500, 1000],  # Specific frames
    metrics={"PSNR": 35.2, "SSIM": 0.95}
)

# Open in browser
viewer.open_in_browser("comparison.html")
```

---

## Batch Templates

Create reusable templates for processing multiple folders with consistent settings.

### Features

| Feature | Description |
|---------|-------------|
| **Folder Patterns** | Process multiple directories |
| **Priority Ordering** | Control processing order |
| **Per-Folder Overrides** | Custom settings per folder |
| **Progress Tracking** | Track progress across batches |
| **Template Import/Export** | Share templates with others |

### CLI Usage

```bash
# Create template
framewright batch-template create archive_project \
    --config preset=archive,scale=4 \
    --folder ./films/1950s --priority 1 \
    --folder ./films/1960s --priority 2 \
    --folder ./films/1970s --priority 3

# Run template
framewright batch-template run archive_project

# List templates
framewright batch-template list

# Export template
framewright batch-template export archive_project --output template.json
```

### Python API

```python
from framewright.batch.batch_templates import BatchTemplate, BatchRunner

# Create template
template = BatchTemplate("archive_restoration")
template.set_config({"preset": "archive", "scale_factor": 4})
template.add_folder("./films/1950s", priority=1)
template.add_folder("./films/1960s", priority=2)
template.add_folder("./films/1970s", priority=3, config_overrides={"colorize": True})

# Save for reuse
template.save("archive_template.json")

# Run batch
runner = BatchRunner(template)
progress = runner.run()
print(f"Processed {progress.completed_videos}/{progress.total_videos} videos")
```

---

## Interlacing Detection & Removal

Automatic detection and correction of interlaced video with support for various deinterlacing methods.

### Detection Features

| Feature | Description |
|---------|-------------|
| **Field Order Detection** | TFF (Top Field First) or BFF (Bottom Field First) |
| **Telecine Detection** | 3:2 pulldown pattern identification |
| **Progressive Check** | Verify if already progressive |
| **FFmpeg idet** | Uses FFmpeg's interlace detection filter |

### Deinterlacing Methods

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| `yadif` | Good | Fast | General purpose |
| `bwdif` | Better | Medium | Higher quality |
| `nnedi` | Best | Slow | Maximum quality |

### CLI Usage

```bash
# Detect interlacing
framewright detect-interlace video.mp4

# Deinterlace with auto-detection
framewright deinterlace video.mp4 --output progressive.mp4

# Specify method
framewright deinterlace video.mp4 --method bwdif --output progressive.mp4

# Inverse telecine for 3:2 pulldown
framewright deinterlace video.mp4 --ivtc --output film.mp4
```

---

## Letterbox Detection & Cropping

Automatic detection and removal of black bars (letterbox/pillarbox).

### Features

| Feature | Description |
|---------|-------------|
| **Luminance Analysis** | Detects bars via brightness thresholds |
| **Multi-Frame Sampling** | Avoids false detection from dark scenes |
| **Codec-Aligned Crop** | Ensures dimensions work with encoders |
| **Aspect Ratio Helpers** | Identifies common aspect ratios |

### CLI Usage

```bash
# Detect letterbox
framewright detect-bars video.mp4

# Auto-crop black bars
framewright crop-bars video.mp4 --output cropped.mp4

# Preview crop without applying
framewright crop-bars video.mp4 --dry-run
```

---

## Film Stock Detection

Identifies vintage film stock types and applies era-appropriate color correction.

### Supported Film Stocks

| Stock | Era | Characteristics |
|-------|-----|-----------------|
| Kodachrome | 1935-2009 | Warm yellows, rich reds |
| Ektachrome | 1946-2012 | Balanced, slight blue cast |
| Technicolor | 1916-2002 | Saturated, distinctive palette |
| Eastmancolor | 1950s+ | Prone to magenta fading |
| Agfacolor | 1936-1990s | European films, cool tones |
| Fujifilm | 1960s+ | Accurate skin tones |

### CLI Usage

```bash
# Detect film stock
framewright detect-stock old_film.mp4

# Apply stock-specific correction
framewright restore old_film.mp4 --correct-stock --output restored.mp4
```

---

## Audio Sync Verification

AI-powered detection and correction of audio-video synchronization issues.

### Features

| Feature | Description |
|---------|-------------|
| **Drift Detection** | Measures audio offset in milliseconds |
| **Confidence Score** | Indicates detection reliability |
| **Direction Detection** | Audio early or late |
| **Automatic Correction** | Shifts audio to match video |

### CLI Usage

```bash
# Check sync
framewright check-sync video.mp4
# Output: "Audio drift: +45ms (audio late), confidence: 92%"

# Fix sync issues
framewright fix-sync video.mp4 --output synced.mp4

# Manual offset
framewright fix-sync video.mp4 --offset -50 --output synced.mp4
```

---

## Subtitle Extraction & Removal

Extract burnt-in subtitles via OCR before removing them, preserving text for translation.

### Features

| Feature | Description |
|---------|-------------|
| **Multi-Engine OCR** | EasyOCR, Tesseract, PaddleOCR |
| **SRT Export** | Save extracted text as subtitle file |
| **Multi-Language** | 80+ languages supported |
| **Translation Ready** | SRT format works with AI translation tools |
| **Clean Removal** | LaMA inpainting for seamless removal |

### CLI Usage

```bash
# Extract subtitles to SRT (preserves text before removal)
framewright extract-subs movie.mp4 --output subtitles.srt --languages en,zh

# Extract and remove in one step
framewright extract-subs movie.mp4 \
    --output subtitles.srt \
    --remove \
    --output-video clean_movie.mp4

# Just remove (no extraction)
framewright restore movie.mp4 --remove-subtitles --output clean.mp4
```

### Translation Workflow

```bash
# 1. Extract subtitles
framewright extract-subs foreign_movie.mp4 --output original.srt

# 2. Translate with your preferred tool (AI translation, etc.)
# Creates: translated.srt

# 3. Remove burnt-in subtitles
framewright restore foreign_movie.mp4 --remove-subtitles --output clean.mp4

# 4. Add translated subtitles as soft subs
ffmpeg -i clean.mp4 -i translated.srt -c copy -c:s mov_text final.mp4
```

---

## Video Scanning

Comprehensive issue detection that runs all analyzers and suggests fixes.

### CLI Usage

```bash
framewright scan video.mp4
```

### Sample Output

```
FrameWright Video Scan Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Video: old_movie.mp4 (1920x1080, 29.97fps, 1:42:15)

Issues Detected:
  ⚠ Interlaced video (TFF, combed frames detected)
  ⚠ Black bars detected (letterbox 2.35:1)
  ⚠ Film stock: Eastmancolor (magenta fade detected)
  ⚠ Audio sync: +67ms drift (audio late)
  ⚠ High noise level: 45/100 (film_grain)
  ⚠ Video appears upscaled from 720x480

Suggested Fixes:
  framewright deinterlace video.mp4 --method bwdif -o step1.mp4
  framewright crop-bars step1.mp4 -o step2.mp4
  framewright restore step2.mp4 --correct-stock -o step3.mp4
  framewright fix-sync step3.mp4 -o final.mp4

Or fix all at once:
  framewright restore video.mp4 --auto-fix-all -o restored.mp4
```

---

## Noise Profiling

Analyze video noise characteristics for optimal denoiser selection.

### CLI Usage

```bash
# Analyze noise profile
framewright noise-profile video.mp4

# Sample more frames for accuracy
framewright noise-profile video.mp4 --frames 50
```

### Output Example

```
Noise Profile
━━━━━━━━━━━━━
  Overall noise level: 42.3/100
  Dominant type: film_grain
  Luminance noise: 35.2
  Chroma noise: 28.1
  Temporal noise: 15.4
  Film grain: 48.2 (uniformity: 78%)

  Recommended denoiser: grain_preserve
  Recommended strength: 0.65
  Preserve grain: Yes
  Confidence: 87%
```

---

## Upscale Detection

Detect if video has been upscaled from a lower resolution.

### Why It Matters

Many old videos are upscaled in their container resolution:
- DVD content in 1080p containers
- VHS captures stretched to 720p
- Web videos re-encoded at higher resolution

Re-upscaling already-upscaled content produces artifacts. FrameWright detects this and recommends the optimal approach.

### CLI Usage

```bash
# Check for upscaling
framewright upscale-detect video.mp4
```

### Output Example

```
Upscaling Detected
━━━━━━━━━━━━━━━━━
  Container resolution: 1920x1080
  Estimated source: 720x480
  Upscale factor: 2.3x
  Upscale method: bicubic
  Confidence: 82%

  Video was upscaled ~2.3x from ~720x480. Consider downscaling
  to 720x480 before restoration, then upscale with AI for
  better quality.
```

---

## GPU Thermal Monitoring

Monitor GPU temperature and detect thermal throttling for stable long-running jobs.

### CLI Usage

```bash
# Check current thermal status
framewright gpu-thermal

# Continuous monitoring
framewright gpu-thermal --monitor
```

### Features

| Feature | Description |
|---------|-------------|
| **Temperature Reading** | Current GPU temperature in Celsius |
| **Thermal State** | Cool (<60C), Warm (60-75C), Hot (75-85C), Critical (>85C) |
| **Throttle Detection** | Detects thermal, power limit, and reliability throttling |
| **Auto-Adaptation** | Automatically reduces batch size when GPU is hot |
| **Cool-Down Pause** | Pauses processing when critical temperature reached |

### Python API

```python
from framewright.utils.thermal_monitor import ThermalMonitor

monitor = ThermalMonitor()

# Use managed processing for automatic thermal management
with monitor.managed_processing():
    for batch in batches:
        # Batch size automatically reduced when hot
        safe_batch = monitor.get_safe_batch_size(max_batch_size=8)
        process_batch(batch[:safe_batch])

        if monitor.should_pause():
            monitor.cool_down_pause()
```

---

## Quick Preview

Generate a quick preview of restoration without processing the entire video.

### CLI Usage

```bash
# Generate preview (samples every 100 frames)
framewright quick-preview video.mp4 --every 100

# Preview with specific preset
framewright quick-preview video.mp4 --preset archive --output preview.mp4
```

This processes ~1% of frames in ~1% of the time, letting you validate settings before committing to a full restoration.

---

## Frame Grid

Generate a single image showing sample frames from the video.

```bash
# Generate 9-frame grid (3x3)
framewright frame-grid video.mp4

# Customize grid size
framewright frame-grid video.mp4 --frames 16 --cols 4 --output grid.jpg
```

---

## Preset Comparison

Compare multiple restoration presets on a single frame to choose the best settings.

```bash
# Compare default presets
framewright compare-presets video.mp4

# Compare specific presets
framewright compare-presets video.mp4 --presets fast balanced quality ultimate

# Use specific frame
framewright compare-presets video.mp4 --frame 500 --output-dir ./comparison
```

Generates separate images for each preset plus metrics comparison.

---

## System Check

Verify system readiness before starting long restoration jobs.

```bash
# Basic system check
framewright system-check

# Check with output directory for disk space
framewright system-check --output-dir /path/to/output
```

### Checks Performed

| Check | Description |
|-------|-------------|
| **GPU** | Detect GPU, VRAM, vendor |
| **Thermal** | Current temperature, throttle state |
| **Disk Space** | Available space in output directory |
| **FFmpeg** | Verify FFmpeg installation |
| **System RAM** | Available system memory |

---

## Processing Safeguards

FrameWright includes automatic safeguards for long-running jobs:

### Features

| Feature | Description |
|---------|-------------|
| **Pre-flight Checks** | Validates disk space, GPU, and memory before starting |
| **Real-time Monitoring** | Continuously monitors disk and thermal during processing |
| **Auto-Adaptation** | Reduces batch size when GPU is hot or memory is low |
| **Cool-down Pause** | Automatically pauses when GPU reaches critical temperature |
| **Graceful Degradation** | Continues at reduced performance rather than crashing |

### Python API

```python
from framewright.workflow import ProcessingSafeguards

safeguards = ProcessingSafeguards(
    video_path="video.mp4",
    output_dir="./output",
)

# Run with automatic safeguards
with safeguards.managed_processing() as context:
    for batch in batches:
        # Get safe batch size (auto-reduced when hot)
        batch_size = context.get_safe_batch_size(max_batch=8)

        # Check conditions (may pause for cool-down)
        context.check_conditions()

        # Process
        process_batch(batch[:batch_size])
        context.report_progress(batch_size)

# Get summary
print(safeguards.get_summary())
```

---

## Audio Restoration Suite

Comprehensive audio repair and enhancement for vintage footage:

### Features

| Feature | Description |
|---------|-------------|
| **Noise Reduction** | RNNoise/SoX/FFmpeg-based hiss removal |
| **Hum Removal** | 50/60Hz + harmonics elimination |
| **Click Removal** | Pop and crackle repair |
| **Declipping** | Repair clipped/distorted audio |
| **Dialog Enhancement** | Speech clarity improvement |
| **Source Separation** | Demucs/Spleeter vocal/music isolation |
| **Mono-to-Stereo** | Intelligent stereo upmixing |
| **DeReverb** | Reduce room reverb |
| **Loudness Normalization** | EBU R128 broadcast compliance |

### CLI Usage

```bash
# Enable audio restoration
framewright restore video.mp4 --audio-restore -o output.mp4

# Full audio suite
framewright restore video.mp4 \
  --audio-restore \
  --audio-denoise \
  --audio-dehum \
  --audio-declip \
  --dialog-enhance \
  -o output.mp4
```

### Python API

```python
from framewright.processors.audio_restoration import AudioRestorer

restorer = AudioRestorer()
restorer.restore(
    input_path="audio.wav",
    output_path="restored.wav",
    denoise=True,
    remove_hum=True,
    enhance_dialog=True,
    normalize=True,
)
```

---

## Film-Specific Restoration

Specialized features for film footage restoration:

### Film Analysis

```python
from framewright.processors.film_restoration import FilmAnalyzer

analyzer = FilmAnalyzer()
chars = analyzer.analyze("old_film.mp4")

print(f"Film Type: {chars.film_type.value}")      # super8, 16mm, 35mm
print(f"Era: {chars.era.value}")                   # silent, golden_age, modern
print(f"Has Grain: {chars.has_grain}")
print(f"Has Flicker: {chars.has_flicker}")
print(f"Has Scratches: {chars.has_scratches}")
print(f"Color Fade: {chars.color_fade:.0%}")
```

### Film Restoration Pipeline

```bash
# Auto-detect and restore film characteristics
framewright restore film.mp4 --preset archive -o restored.mp4

# Manual control
framewright restore film.mp4 \
  --deflicker \
  --scratch-repair \
  --stabilize \
  --color-restore \
  --grain-mode preserve \
  -o restored.mp4
```

### Grain Management Modes

| Mode | Description |
|------|-------------|
| `preserve` | Keep original grain intact |
| `remove` | Remove grain with temporal denoising |
| `match` | Match grain from reference frame |
| `synthesize` | Add synthetic film grain |

---

## Batch Processing

Process multiple videos with priority-based queue management:

### CLI Usage

```bash
# Add videos to queue
framewright batch add video1.mp4 video2.mp4 video3.mp4

# Add entire directory
framewright batch add-dir ./videos/ --pattern "*.mp4" --recursive

# Start processing
framewright batch start

# Monitor queue
framewright batch status

# Manage jobs
framewright batch pause
framewright batch resume
framewright batch cancel JOB_ID
```

### Python API

```python
from framewright.batch import BatchQueueProcessor, JobPriority

# Create processor
processor = BatchQueueProcessor(
    max_workers=1,
    persistence_path="~/.framewright/queue.json"
)

# Add jobs
processor.add_job("video1.mp4", preset="ultimate", priority=JobPriority.HIGH)
processor.add_job("video2.mp4", preset="balanced", priority=JobPriority.NORMAL)
processor.add_directory("./videos/", pattern="*.mp4", preset="fast")

# Start processing
processor.start()

# Monitor
print(processor.stats)
```

---

## QA Reports

Generate comprehensive quality assessment reports:

```bash
# Generate report after restoration
framewright qa-report original.mp4 restored.mp4 -o ./reports/

# Outputs:
#   ./reports/restored_qa_report.html  (visual report)
#   ./reports/restored_qa_report.json  (machine-readable)
#   ./reports/comparisons/             (frame comparisons)
```

### Report Contents

- **Quality Grade**: A+ to F rating
- **Metrics**: PSNR, SSIM, per-frame analysis
- **Problem Frames**: Identified quality issues
- **Segment Analysis**: Quality by video section
- **Recommendations**: Improvement suggestions
- **Visual Comparisons**: Before/after frame images

---

## Export Presets

Platform-optimized encoding profiles:

### Available Presets

| Preset | Platform | Description |
|--------|----------|-------------|
| `youtube_4k` | YouTube | 4K HDR optimized |
| `youtube_1080p` | YouTube | Standard HD |
| `vimeo_4k` | Vimeo | High quality 4K |
| `instagram_reel` | Instagram | Vertical 9:16 |
| `tiktok` | TikTok | Vertical optimized |
| `twitter` | Twitter/X | Fast-loading |
| `archive_master` | Archive | Maximum quality H.265 |
| `archive_prores` | Archive | ProRes 422 HQ |
| `broadcast_hd` | Broadcast | EBU R128 compliant |
| `web_optimized` | Web | Fast loading |
| `web_av1` | Web | Modern AV1 codec |

### CLI Usage

```bash
# Export with preset
framewright export restored.mp4 --preset youtube_4k -o final.mp4

# List available presets
framewright export --list-presets

# Create custom preset
framewright export restored.mp4 \
  --preset youtube_1080p \
  --video-bitrate 20M \
  --audio-bitrate 320k \
  -o final.mp4
```

---

## Project Management

Save and restore complete project state:

### Creating Projects

```bash
# Create new project
framewright project create "Moscow 1909 Restoration" --source video.mp4

# Open existing project
framewright project open ~/.framewright/projects/moscow_1909/

# List projects
framewright project list
```

### Python API

```python
from framewright.project import ProjectManager, create_project

# Create project
project = create_project(
    name="Historical Film Restoration",
    source_path="old_film.mp4",
    preset="archive",
    description="Restoring 1920s documentary",
    tags=["historical", "documentary", "b&w"]
)

# Later: resume project
manager = ProjectManager()
project = manager.load_project("~/.framewright/projects/historical_film/")

# Check progress
summary = manager.get_project_summary()
print(f"Progress: {summary['progress_percent']:.0f}%")
print(f"Current stage: {summary['current_stage']}")
```

### Project Features

- **Auto-save**: Progress saved automatically
- **Resume**: Continue interrupted restorations
- **Versioning**: Revert to previous project states
- **Export/Import**: Share projects as archives

---

## Real-Time Preview

Live preview of restoration quality during processing:

```python
from framewright.ui.preview import RealTimePreview, PreviewMode

# Create preview
preview = RealTimePreview(config={
    "backend": "opencv",
    "mode": PreviewMode.SLIDER,
    "show_metrics": True,
})

# Start preview
preview.start()

# Add frames during processing
preview.add_frame(
    frame_number=0,
    timestamp=0.0,
    original=original_frame,
    restored=restored_frame,
    metrics={"PSNR": 35.2, "SSIM": 0.95}
)
```

### Preview Modes

| Mode | Description |
|------|-------------|
| `side_by_side` | Original and restored side by side |
| `slider` | Interactive slider comparison |
| `toggle` | Alternate between original/restored |
| `diff` | Show difference visualization |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Pause/resume |
| `S` | Take screenshot |
| `1-4` | Switch preview mode |
| `Q/Esc` | Quit preview |

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
├── temp/
│   ├── frames/          # Extracted frames (PNG)
│   ├── unique_frames/   # Deduplicated frames (if deduplication enabled)
│   ├── enhanced/        # AI-enhanced frames
│   └── interpolated/    # RIFE interpolated frames (if enabled)
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

## Frame Deduplication (Historical Films)

Many historical films were shot at lower frame rates (16-18fps) but have been digitized or uploaded to YouTube at 25fps by duplicating frames. This causes:

- **Wasted GPU time** enhancing duplicate frames
- **Jerky motion** from artificial frame padding
- **Inaccurate frame rate** metadata

FrameWright can detect and remove these duplicate frames, enhance only unique frames, then use RIFE to create smooth motion.

### How It Works

```
Original: 18fps film → Uploaded as 25fps (frames duplicated)

Without Deduplication:
  Extract 25fps → Enhance all 25 frames/sec → Jerky 25fps output

With Deduplication + RIFE:
  Extract 25fps → Detect ~18fps unique → Enhance 18 frames/sec → RIFE interpolate → Smooth 25fps+ output
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **28-40% faster** | Enhance only unique frames (18fps vs 25fps) |
| **Smoother motion** | RIFE generates intermediate frames vs duplicating |
| **True frame rate** | Detects actual source FPS (16-18fps for 1900s film) |
| **Better quality** | No enhancement artifacts from processing duplicates |

### Workflow for 1909 Film Example

```python
from framewright import Config, VideoRestorer
from pathlib import Path

config = Config(
    project_dir=Path("F:/Video/Moscow Clad in Snow 1909"),

    # Enable deduplication for historical film
    enable_deduplication=True,
    deduplication_threshold=0.98,  # Similarity threshold
    expected_source_fps=18,        # Hint: 1909 film was ~16-18fps

    # Enable RIFE for smooth motion (instead of duplicating frames back)
    enable_interpolation=True,
    target_fps=25,  # Or 48 for ultra-smooth

    # Standard enhancement
    scale_factor=4,
    model_name="realesrgan-x4plus",
)

restorer = VideoRestorer(config)
output = restorer.restore_video("moscow_1909.mp4")
```

### CLI Usage

```bash
# Deduplicate + enhance + RIFE interpolation
framewright restore --input moscow_1909.mp4 \
    --deduplicate \
    --dedup-threshold 0.98 \
    --expected-fps 18 \
    --enable-rife \
    --target-fps 25 \
    --output restored.mp4

# Analyze frame duplication without processing
framewright analyze --input old_film.mp4 --detect-duplicates
```

### Pipeline Flow

1. **Extract frames** → `temp/frames/` (all 25fps frames)
2. **Deduplicate** → `temp/unique_frames/` (only ~18fps unique frames)
3. **Enhance** → `temp/enhanced/` (GPU processes fewer frames)
4. **RIFE interpolate** → `temp/interpolated/` (smooth 25fps+ from 18fps)
5. **Reassemble** → final video with smooth motion

### Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_deduplication` | `False` | Enable duplicate frame detection |
| `deduplication_threshold` | `0.98` | Similarity threshold (0.95-0.99) |
| `expected_source_fps` | `None` | Hint for original FPS (e.g., 18 for 1909 film) |

### Threshold Selection Guide

| Threshold | Use Case | Expected Result |
|-----------|----------|-----------------|
| **0.98-0.99** | Clean sources, minimal compression | Strict: only near-exact duplicates |
| **0.96-0.97** | YouTube videos (recommended) | Balanced: handles compression artifacts |
| **0.94-0.95** | Heavily compressed sources | Lenient: may catch similar (non-duplicate) frames |
| **0.90-0.93** | Very noisy/degraded sources | Very lenient: use with caution |

**Expected duplicate ratios for old films:**
- 16fps original → 25fps upload: ~36% duplicates (64% unique)
- 18fps original → 25fps upload: ~28% duplicates (72% unique)
- 18fps original → 30fps upload: ~40% duplicates (60% unique)

**If you see too few duplicates:** The video may be native frame rate (not padded) or threshold is too strict.

**If you see too many duplicates:** Threshold is too aggressive—try 0.96-0.97 instead.

> **Note:** Requires `imagehash` package for perceptual hashing. Install with: `pip install imagehash`

### Historical Film FPS Reference

| Era | Typical FPS | Notes |
|-----|-------------|-------|
| 1890s-1900s | 14-16 fps | Hand-cranked cameras |
| 1900s-1910s | 16-18 fps | Early motorized cameras |
| 1920s | 18-22 fps | Transition period |
| 1930s+ | 24 fps | Sound film standard |

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

# For frame deduplication (recommended for historical films)
pip install imagehash

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

### Cloud GPU Processing (Optional - Vast.ai)

> **Note:** FrameWright is designed to run 100% locally with no API or cloud services required. Cloud processing is entirely optional for users who want to offload heavy workloads.

Process videos on cloud GPUs without local hardware. Videos stay on Google Drive - zero local storage used.

**Setup (one-time):**
```bash
# Configure Google Drive access
rclone config  # Create remote named "gdrive"

# Set Vast.ai API key
echo "VASTAI_API_KEY=your_key_here" > ~/.framewright/vastai.env
```

**Basic Cloud Restore:**
```bash
# Upload video to Google Drive first, then:
framewright cloud submit \
    --gdrive-input "framewright/input/video.mp4" \
    --scale 4 \
    --quality 15 \
    --gpu RTX_4090 \
    --yes
```

**Archive Quality (1909 B&W Film):**
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/Moscow_clad_in_snow_1909.mp4" \
    --scale 4 \
    --quality 15 \
    --format mkv \
    --deduplicate \
    --dedup-threshold 0.98 \
    --enable-rife \
    --target-fps 25 \
    --rife-model rife-v4.6 \
    --auto-enhance \
    --scratch-sensitivity 0.6 \
    --grain-reduction 0.15 \
    --audio-enhance \
    --gpu RTX_4090 \
    --yes
```

**Full Archive with Colorization:**
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/old_bw_film.mp4" \
    --scale 4 \
    --quality 15 \
    --colorize \
    --colorize-model ddcolor \
    --deduplicate \
    --enable-rife \
    --target-fps 25 \
    --auto-enhance \
    --remove-watermark \
    --watermark-auto-detect \
    --audio-enhance \
    --gpu RTX_4090 \
    --yes
```

**Cloud Commands:**
```bash
framewright cloud gpus      # List available GPUs & pricing
framewright cloud balance   # Check Vast.ai credit balance
framewright cloud status <job_id>  # Check job progress
framewright cloud jobs      # List all jobs
framewright cloud cancel <job_id>  # Cancel a running job
```

**Features:**
- ✅ Auto-installs all dependencies on cloud instance
- ✅ Auto-destroys instance when done (no idle billing)
- ✅ Copies your rclone config for Google Drive access
- ✅ RTX 4090 @ ~$0.30/hr, H100 @ ~$1.60/hr

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
- **RAM**: 16 GB
- **GPU**: NVIDIA/AMD/Intel with 4+ GB VRAM
- **Disk**: 50 GB free (SSD recommended)

### Recommended
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3080+ with 10+ GB VRAM
- **Disk**: 500 GB free (NVMe SSD)

### Optimal (Designed For)
- **CPU**: 16+ cores
- **RAM**: 192 GB
- **GPU**: NVIDIA RTX 5090 with 32 GB VRAM
- **Disk**: 2 TB NVMe SSD

### VRAM Guidelines

| VRAM | Max Resolution | Tile Size | Batch Size | Best For |
|------|----------------|-----------|------------|----------|
| 4 GB | 720p | 128 | 1 | Testing, small clips |
| 6 GB | 1080p | 192 | 1 | Entry-level GPU |
| 8 GB | 1080p | 256 | 1-2 | GTX 1080, RTX 3060 |
| 12 GB | 1440p | 384 | 2-4 | RTX 3080, 4070 Ti |
| 16 GB | 4K | 512 | 4-8 | RTX 4080, A4000 |
| 24 GB | 4K+ | No tiling | 8-16 | RTX 3090, 4090 |
| 32 GB | 8K | No tiling | 16+ | RTX 5090, A6000 |

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
├── cli_simple.py        # Simplified Apple-like CLI
├── ui.py                # Web UI (Gradio)
├── hardware.py          # Hardware compatibility checks
├── checkpoint.py        # Checkpointing & resume system
├── dry_run.py           # Dry run mode for testing
├── watch.py             # Watch mode for folder monitoring
├── exceptions.py        # Custom exception classes
├── processors/
│   ├── analyzer.py            # Content analysis & detection
│   ├── adaptive_enhance.py    # Adaptive enhancement
│   ├── deduplication.py       # Frame deduplication for historical films
│   ├── defect_repair.py       # Scratch/dust/grain removal
│   ├── face_restore.py        # GFPGAN/CodeFormer
│   ├── aesrgan_face.py        # AESRGAN attention-enhanced faces
│   ├── interpolation.py       # RIFE frame interpolation
│   ├── ncnn_vulkan.py         # NCNN-Vulkan GPU backend
│   ├── colorization.py        # DeOldify/DDColor
│   ├── swintexco_colorize.py  # SwinTExCo exemplar colorization
│   ├── watermark_removal.py   # LaMA inpainting
│   ├── subtitle_removal.py    # OCR + inpainting
│   ├── subtitles.py           # Subtitle extraction
│   ├── stabilization.py       # Video stabilization
│   ├── scene_detection.py     # Scene boundary detection
│   ├── hdr_conversion.py      # HDR/SDR conversion
│   ├── audio.py               # Audio processing
│   ├── audio_enhance.py       # Audio enhancement
│   ├── audio_sync.py          # Audio-video sync
│   ├── audio_restoration.py   # Comprehensive audio restoration suite
│   ├── streaming.py           # Streaming mode
│   ├── preview.py             # Preview generation
│   ├── advanced_models.py     # Extended model support
│   ├── tap_denoise.py         # TAP neural denoising
│   ├── diffusion_sr.py        # Diffusion super-resolution
│   ├── qp_artifact_removal.py # QP-aware artifact removal
│   ├── frame_generation.py    # Missing frame generation (SVD)
│   ├── cross_attention_temporal.py # Temporal consistency
│   ├── film_restoration.py    # Film-specific restoration
│   ├── interlace_handler.py   # Interlacing detection & deinterlace
│   ├── letterbox_handler.py   # Letterbox detection & cropping
│   ├── film_stock_detector.py # Film stock identification & correction
│   ├── noise_profiler.py      # Advanced noise analysis
│   └── upscale_detector.py    # Upscale/native resolution detection
├── ui/
│   ├── __init__.py          # UI module exports
│   ├── terminal.py          # Rich terminal interface
│   ├── progress.py          # Multi-stage progress display
│   ├── auto_detect.py       # Smart video analyzer
│   ├── recommendations.py   # Intelligent preset recommender
│   ├── wizard.py            # Interactive guided setup
│   └── preview.py           # Real-time preview system
├── batch/
│   ├── __init__.py          # Batch module exports
│   ├── queue_processor.py   # Priority-based job queue
│   └── batch_templates.py   # Reusable batch processing templates
├── reports/
│   ├── __init__.py          # Reports module exports
│   └── qa_report.py         # QA report generator
├── export/
│   ├── __init__.py          # Export module exports
│   ├── presets.py           # Platform export presets
│   └── comparison_viewer.py # Interactive HTML comparison viewer
├── presets/
│   ├── __init__.py          # Presets module exports
│   └── preset_library.py    # Community preset library
├── persistence/
│   ├── __init__.py          # Persistence module exports
│   └── checkpoint_manager.py # Frame-level checkpoint/resume
├── workflow/
│   ├── __init__.py          # Workflow module exports
│   ├── automation.py        # Watch folder, queue processing
│   └── processing_safeguards.py # Thermal/disk monitoring integration
├── project/
│   ├── __init__.py          # Project module exports
│   └── project_manager.py   # Project file management
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
    ├── gpu_memory_optimizer.py # Dynamic VRAM management
    ├── thermal_monitor.py  # GPU thermal monitoring & throttle detection
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

This project is licensed under the **Elastic License 2.0 (ELv2)** - see [LICENSE.md](LICENSE.md) for details.

You are free to use FrameWright for any purpose, including commercial video production. You may not offer it as a hosted/managed service or sell modified versions as a competing product.

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

---

## Roadmap

### v1.0 Features (Current Release)

**Core Architecture (~43,000 lines)**

| Component | Description | Status |
|-----------|-------------|--------|
| **Unified Processor Interfaces** | Single API with multiple backends per processor | ✅ Complete |
| **Hardware Tier Auto-Detection** | CPU → NCNN → CUDA 4GB → 8GB → 12GB → 24GB+ fallback | ✅ Complete |
| **Long-Form Temporal Consistency** | Global anchors, chunked processing for 7,000+ frames | ✅ Complete |
| **TensorRT Acceleration** | FP16/INT8 with dynamic shapes and engine caching | ✅ Complete |
| **Apple Silicon Support** | CoreML conversion, M1-M4 detection, ANE acceleration | ✅ Complete |
| **AMD ROCm/HIP Backend** | Full GPU acceleration on AMD GPUs | ✅ Complete |
| **Intel oneAPI/OpenVINO** | Optimized inference on Intel hardware | ✅ Complete |

**Enhancement Pipeline**

| Enhancement | Description | Status |
|-------------|-------------|--------|
| **Unified Denoiser** | 8 backends: Restormer, NAFNet, VRT, SCUNet, DPIR, FFDNet, wavelet, bilateral | ✅ Complete |
| **Unified Super Resolution** | 11 backends: Real-ESRGAN, SwinIR, HAT, Diffusion, ESPCN, EDSR, etc. | ✅ Complete |
| **Diffusion-Based VSR** | FlashVSR one-step diffusion for superior textures | ✅ Complete |
| **Guided Super Resolution** | CLIP text-guided with 8 style presets | ✅ Complete |
| **TAP Denoising Framework** | Restormer/NAFNet temporal denoising (+4-6 dB PSNR) | ✅ Complete |
| **Temporal VAE** | 3D encoder/decoder with cross-frame attention | ✅ Complete |
| **HDR Export** | HDR10, HDR10+, Dolby Vision, HLG formats | ✅ Complete |

**Restoration Features**

| Feature | Description | Status |
|---------|-------------|--------|
| **Unified Colorizer** | 4 backends: DDColor, DeOldify, SwinTExCo, ECCV16 | ✅ Complete |
| **Unified Face Restorer** | 4 backends: CodeFormer, GFPGAN, RestoreFormer, DFDNet | ✅ Complete |
| **Film Grain Manager** | FFT analysis, wavelet extraction, Perlin synthesis | ✅ Complete |
| **Frame Generator** | RAFT flow, RIFE/FILM interpolation, SVD extension | ✅ Complete |
| **Defect Repair** | Scratch, dust, tear, and water damage removal | ✅ Complete |
| **Video Stabilization** | 3 backends: VidStab, OpenCV, DeepStab | ✅ Complete |

**Audio Enhancement**

| Feature | Description | Status |
|---------|-------------|--------|
| **Unified Audio Enhancer** | 4 backends: DeepFilterNet, SpeechBrain, RNNoise, Traditional | ✅ Complete |
| **DeepFilterNet Integration** | Real-time noise suppression (10ms latency) | ✅ Complete |
| **Audio-Video Sync** | AI-powered drift detection and correction | ✅ Complete |

**Analysis & Quality**

| Feature | Description | Status |
|---------|-------------|--------|
| **Unified Content Analyzer** | Scene detection, degradation analysis, face detection | ✅ Complete |
| **Quality Scorer** | PSNR, SSIM, VMAF, perceptual metrics | ✅ Complete |
| **Degradation Detector** | Noise, blur, compression, interlacing detection | ✅ Complete |

**Infrastructure**

| Component | Description | Status |
|-----------|-------------|--------|
| **Pipeline Engine** | Job scheduling with stage dependencies | ✅ Complete |
| **Checkpoint System** | Frame-level progress with crash recovery | ✅ Complete |
| **Model Registry** | Centralized model management and downloading | ✅ Complete |
| **Frame/Model Cache** | LRU eviction with memory pressure handling | ✅ Complete |
| **Cloud Coordinator** | RunPod, Vast.ai, Lambda Labs integration | ✅ Complete |

**User Experience**

| Feature | Description | Status |
|---------|-------------|--------|
| **Interactive Wizard** | 5-step guided setup for beginners | ✅ Complete |
| **Web Preview Server** | Real-time before/after at localhost:8080 | ✅ Complete |
| **Hardware-Aware Presets** | Auto-selects optimal settings per GPU tier | ✅ Complete |
| **Simplified CLI** | Apple-like commands that "just work" | ✅ Complete |
| **Dashboard** | Web-based monitoring and control | ✅ Complete |

### Future Enhancements (v1.1+)

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **Unified Multi-Task Model** | BCell-based RNN handling denoising, deblurring, and SR in one pass | Medium |
| **Plugin Architecture** | Third-party processor plugins via `~/.framewright/plugins/` | Medium |
| **REST API Server** | `framewright serve --port 8080` for integration workflows | Medium |
| **Real-Time Streaming** | Process live streams with low latency | Medium |
| **Mobile Companion** | iOS/Android app for remote monitoring and control | Low |
| **Kubernetes Deployment** | Helm charts for production cluster deployment | Low |
| **WebGPU Backend** | Browser-based GPU acceleration | Low |

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for architecture details.