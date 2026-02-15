# Quick Start

Get started with FrameWright v1.0 in 2 minutes.

## Install

```bash
pip install framewright
```

FFmpeg required:
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html

## Restore a Video

```bash
framewright restore video.mp4
```

Done. Your restored video is ready. FrameWright auto-detects your hardware and selects optimal settings.

## Common Commands

| Command | What it does |
|---------|--------------|
| `framewright restore video.mp4` | Auto-restore with best settings |
| `framewright video.mp4` | Same thing, simpler |
| `framewright quick video.mp4` | Fast preview |
| `framewright best video.mp4` | Maximum quality |
| `framewright archive video.mp4` | Old footage |
| `framewright wizard` | Interactive setup |

## Quick Examples

```bash
# Upscale to 4K
framewright restore video.mp4 --scale 4 -o video_4k.mp4

# Smooth motion (60fps)
framewright restore video.mp4 --enable-rife --target-fps 60

# Old film restoration
framewright restore old_film.mp4 --preset archive

# Colorize black & white
framewright restore bw_video.mp4 --colorize

# Remove watermark
framewright restore video.mp4 --remove-watermark

# Full analysis first
framewright analyze video.mp4
```

## Interactive Wizard

For guided setup:

```bash
framewright wizard
```

Walks you through:
1. Selecting your video
2. Choosing quality vs speed
3. Enabling features (colorization, interpolation, etc.)
4. Hardware optimization
5. Starting the restoration

## Check Your Setup

```bash
framewright-check
```

Shows your GPU, VRAM, available backends, and confirms everything works.

## Web UI (Optional)

```bash
pip install framewright[ui]
framewright-ui
```

Opens a drag-and-drop interface at http://localhost:7860

## Real-Time Preview

```bash
framewright preview video.mp4
```

Opens a browser at http://localhost:8080 with live before/after comparison.

## Hardware Auto-Detection

FrameWright automatically detects:

| Hardware | Backend Used |
|----------|--------------|
| NVIDIA GPU | CUDA + TensorRT |
| AMD GPU | ROCm or Vulkan |
| Intel GPU | OpenVINO or Vulkan |
| Apple Silicon | CoreML + Metal |
| CPU only | OpenCV fallback |

## Next Steps

- [Presets Guide](presets.md) - Choose quality vs speed
- [Hardware Guide](hardware.md) - Optimize for your GPU
- [Troubleshooting](troubleshooting.md) - Fix common issues
