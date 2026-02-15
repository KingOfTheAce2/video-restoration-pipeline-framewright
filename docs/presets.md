# Presets Guide

FrameWright v1.0 includes **hardware-aware presets** that automatically optimize for your GPU.

## Quick Start

```bash
# Auto-detect best preset for your hardware
framewright restore video.mp4

# Or pick one explicitly
framewright restore video.mp4 --preset quality
```

## Quality Levels

| Preset | Speed | VRAM | Best For |
|--------|-------|------|----------|
| `fast` | Fast | 2GB+ | Quick preview, testing |
| `quality` | Medium | 4GB+ | General use (default) |
| `archive` | Slow | 8GB+ | Historical footage |
| `ultimate` | Slowest | 12GB+ | Maximum quality |

```bash
framewright restore video.mp4 --preset fast
framewright restore video.mp4 --preset quality
framewright restore video.mp4 --preset archive
framewright restore video.mp4 --preset ultimate
```

## Content-Specific Presets

| Preset | Use When | Key Features |
|--------|----------|--------------|
| `film_restoration` | Old movies with scratches | Deflicker, scratch repair, grain preserve |
| `anime` | Anime and cartoons | Line preservation, color boost |
| `vhs` | VHS tapes | Tracking fix, color correction |
| `authentic` | Keep vintage look | Light touch, preserve character |
| `broadcast` | TV content | Deinterlace, noise reduction |

```bash
framewright restore anime.mp4 --preset anime
framewright restore tape.mp4 --preset vhs
framewright restore 1920s.mp4 --preset film_restoration
```

## Hardware-Aware Auto-Selection

FrameWright detects your GPU and auto-adjusts:

| Your VRAM | Default Preset | Tile Size | Batch |
|-----------|----------------|-----------|-------|
| 4GB | fast | 128px | 1 |
| 8GB | quality | 256px | 2 |
| 12GB | archive | 384px | 4 |
| 16GB | ultimate | 512px | 4 |
| 24GB+ | ultimate | No tiling | 8+ |

## Backend-Specific Presets

For specific hardware acceleration:

```bash
# TensorRT optimized (NVIDIA)
framewright restore video.mp4 --preset ultimate --backend tensorrt

# CoreML optimized (Apple Silicon)
framewright restore video.mp4 --preset quality --backend coreml

# OpenVINO optimized (Intel)
framewright restore video.mp4 --preset quality --backend openvino
```

## Preset Components

### What's in Each Preset

**fast**
- Basic Real-ESRGAN upscaling
- No face restoration
- No temporal consistency
- FP16 precision

**quality**
- Real-ESRGAN with face detection
- GFPGAN face restoration
- Basic temporal smoothing
- FP16 precision

**archive**
- Real-ESRGAN or SwinIR
- CodeFormer face restoration
- Full temporal consistency
- Grain preservation
- Deflicker enabled

**ultimate**
- Diffusion SR (FlashVSR)
- CodeFormer + GFPGAN ensemble
- Cross-attention temporal
- Grain management
- HDR expansion
- Full FP32 precision

## Override Settings

Combine preset with custom options:

```bash
# Fast preset but higher quality output
framewright restore video.mp4 --preset fast --crf 18

# Quality preset with lower VRAM usage
framewright restore video.mp4 --preset quality --tile 256

# Add frame interpolation to any preset
framewright restore video.mp4 --preset quality --enable-rife --target-fps 60

# Add colorization
framewright restore video.mp4 --preset archive --colorize

# Specify denoiser backend
framewright restore video.mp4 --preset quality --denoiser restormer
```

## Processor Backends by Preset

| Processor | fast | quality | archive | ultimate |
|-----------|------|---------|---------|----------|
| **Super Resolution** | ESPCN | Real-ESRGAN | SwinIR | FlashVSR |
| **Denoiser** | bilateral | FFDNet | NAFNet | Restormer |
| **Face Restorer** | none | GFPGAN | CodeFormer | Ensemble |
| **Colorizer** | DeOldify | DDColor | DDColor | SwinTExCo |
| **Temporal** | none | basic | full | cross-attn |

## Export Presets

For final output encoding:

```bash
framewright export restored.mp4 --preset youtube_4k
framewright export restored.mp4 --preset archive_master
```

| Export Preset | Target | Codec |
|--------------|--------|-------|
| `youtube_4k` | YouTube 4K | H.264 High |
| `youtube_1080p` | YouTube HD | H.264 High |
| `instagram_reel` | Instagram | H.264 9:16 |
| `archive_master` | Archival | H.265 (HEVC) |
| `archive_prores` | Professional | ProRes 422 HQ |
| `hdr10` | HDR Display | HEVC HDR10 |
| `dolby_vision` | Premium HDR | HEVC DV |

## List All Presets

```bash
# Restoration presets
framewright restore --list-presets

# Export presets
framewright export --list-presets

# Show preset details
framewright preset info ultimate
```

## Create Custom Presets

```bash
# Save current settings as preset
framewright preset create my_preset --from-config current.yaml

# Export for sharing
framewright preset export my_preset --output my_preset.json

# Import shared preset
framewright preset import shared_preset.json
```

## Preset Quick Reference

| I want... | Use this |
|-----------|----------|
| Quick results | `--preset fast` |
| Good quality | `--preset quality` |
| Best possible | `--preset ultimate` |
| Restore anime | `--preset anime` |
| VHS tapes | `--preset vhs` |
| Old film | `--preset film_restoration` |
| Keep vintage feel | `--preset authentic` |
| Archive preservation | `--preset archive` |
| TV broadcast content | `--preset broadcast` |
