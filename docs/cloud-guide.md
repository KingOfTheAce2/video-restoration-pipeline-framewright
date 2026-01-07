# FrameWright Cloud Processing Guide

Quick reference for cloud GPU processing on Vast.ai.

---

## Your Command (Copy & Paste)

```bash
framewright cloud submit \
    --input "gdrive:framewright/input/Moscow_clad_in_snow_1909.mp4" \
    --output-dir "gdrive:framewright/output/" \
    --scale 4 \
    --gpu RTX_4090
```

---

## Quick Start Examples

### Basic 4x Upscale (Archive Quality)
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/video.mp4" \
    --scale 4 \
    --quality 15 \
    --gpu RTX_4090
```

### Full Restoration (Old Film)
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/old_film.mp4" \
    --scale 4 \
    --auto-enhance \
    --deduplicate \
    --enable-rife \
    --target-fps 24 \
    --audio-enhance \
    --gpu RTX_4090
```

### Black & White to Color
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/bw_footage.mp4" \
    --scale 4 \
    --colorize \
    --colorize-model ddcolor \
    --auto-enhance \
    --gpu RTX_4090
```

### Remove Watermark & Subtitles
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/watermarked.mp4" \
    --scale 2 \
    --remove-watermark \
    --watermark-auto-detect \
    --remove-subtitles \
    --gpu RTX_4090
```

### YouTube Video Processing
```bash
framewright cloud submit \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --scale 4 \
    --auto-enhance \
    --gpu RTX_4090
```

### Wait for Result & Download
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/video.mp4" \
    --scale 4 \
    --wait \
    --output-dir ./restored/
```

---

## Command Reference

### Input Options

| Flag | Description | Example |
|------|-------------|---------|
| `--input`, `-i` | Local video file | `--input ./video.mp4` |
| `--gdrive-input` | Video on Google Drive | `--gdrive-input "framewright/input/video.mp4"` |
| `--url`, `-u` | YouTube/video URL | `--url "https://youtube.com/..."` |
| `--cookies-from-browser` | Browser for auth cookies | `--cookies-from-browser chrome` |
| `--cookies` | Path to cookies.txt | `--cookies ./cookies.txt` |

### Enhancement Options

| Flag | Description | Default |
|------|-------------|---------|
| `--scale`, `-s` | Upscaling factor (2 or 4) | `4` |
| `--model` | AI enhancement model | `realesrgan-x4plus` |
| `--quality`, `-q` | CRF quality (0-51, lower=better) | `15` |
| `--auto-enhance` | Enable all auto enhancements | off |
| `--no-face-restore` | Disable face restoration | enabled |
| `--no-defect-repair` | Disable scratch/dust repair | enabled |

### Frame Processing

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-rife` | Enable frame interpolation | off |
| `--target-fps` | Target FPS for RIFE | auto |
| `--rife-model` | RIFE model version | `rife-v4.6` |
| `--deduplicate` | Remove duplicate frames | off |
| `--dedup-threshold` | Dedup similarity (0.9-1.0) | `0.98` |

### Colorization

| Flag | Description | Default |
|------|-------------|---------|
| `--colorize` | Enable AI colorization | off |
| `--colorize-model` | Model: `deoldify` or `ddcolor` | `deoldify` |

### Defect Repair

| Flag | Description | Default |
|------|-------------|---------|
| `--scratch-sensitivity` | Scratch detection (0-1) | `0.5` |
| `--grain-reduction` | Film grain reduction (0-1) | `0.3` |

### Watermark Removal

| Flag | Description |
|------|-------------|
| `--remove-watermark` | Enable watermark removal |
| `--watermark-auto-detect` | Auto-detect watermark location |
| `--watermark-mask` | Path to mask image (white=watermark) |
| `--watermark-region` | Manual region: `x,y,width,height` |

### Subtitle Removal

| Flag | Description | Default |
|------|-------------|---------|
| `--remove-subtitles` | Remove burnt-in subtitles | off |
| `--subtitle-region` | Region to scan | `bottom_third` |
| `--subtitle-ocr` | OCR engine | `auto` |
| `--subtitle-languages` | Languages (comma-sep) | auto |

Subtitle regions: `bottom_third`, `bottom_quarter`, `top_quarter`, `full_frame`

### Audio

| Flag | Description |
|------|-------------|
| `--audio-enhance` | Enable audio noise reduction |

### Output Options

| Flag | Description | Default |
|------|-------------|---------|
| `--format` | Output format | `mkv` |
| `--storage-remote` | rclone remote name | `gdrive` |
| `--storage-path` | Base path in cloud | `framewright` |
| `--output-dir` | Local download directory | - |

Formats: `mkv`, `mp4`, `webm`, `avi`, `mov`

### Cloud Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gpu` | GPU type | `RTX_4090` |
| `--timeout` | Max runtime (minutes) | `120` |
| `--wait`, `-w` | Wait & download result | off |
| `--yes`, `-y` | Skip confirmation | off |

---

## GPU Recommendations

| GPU | Best For | Price/hr |
|-----|----------|----------|
| `RTX_4090` | Fastest, best quality | ~$0.40-0.70 |
| `RTX_3090` | Good balance | ~$0.25-0.45 |
| `A100` | Large videos, pro work | ~$1.00+ |
| `RTX_4080` | Budget option | ~$0.30-0.50 |

---

## Workflow Presets

### Archive Quality (Maximum)
```bash
--scale 4 --quality 12 --auto-enhance --enable-rife --target-fps 24 --deduplicate --audio-enhance
```

### Fast Preview
```bash
--scale 2 --quality 23 --no-face-restore --no-defect-repair
```

### Old Film Restoration
```bash
--scale 4 --auto-enhance --deduplicate --enable-rife --target-fps 24 --scratch-sensitivity 0.7 --grain-reduction 0.4
```

### Silent Film (B&W + No Audio)
```bash
--scale 4 --colorize --colorize-model ddcolor --auto-enhance --enable-rife --target-fps 18
```

---

## Other Cloud Commands

### Check GPU Prices
```bash
framewright cloud gpus
```

### Check Balance
```bash
framewright cloud balance
```

### List Jobs
```bash
framewright cloud status
```

### Cancel Job
```bash
framewright cloud cancel <job_id>
```

---

## Google Drive Setup

First time setup:
```bash
framewright cloud auth
```

This opens browser for Google OAuth. Your token is saved for future use.

### Folder Structure
```
Google Drive/
└── framewright/
    ├── input/      # Upload source videos here
    └── output/     # Restored videos appear here
```

---

## Troubleshooting

### "Is a directory" error
Fixed in latest version. Update with:
```bash
pip install --upgrade git+https://github.com/KingOfTheAce2/video-restoration-pipeline-framewright.git
```

### Job stuck or failed
Check logs on Vast.ai dashboard, then destroy instance manually to stop billing.

### Out of disk space
Default is 100GB. For very long videos, use smaller `--scale 2`.

---

*Last updated: 2026-01-07*
