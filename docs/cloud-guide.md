# FrameWright Cloud Processing Guide

Quick reference for cloud GPU processing on Vast.ai.

---

## Your Command (Copy & Paste)

```bash
framewright cloud submit \
    --input "gdrive:framewright/input/video.mp4" \
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

### B&W Archive (Keep Black & White)
Best for historical footage - maximum quality, keeps original B&W aesthetic.
```bash
framewright cloud submit \
    --gdrive-input "framewright/input/bw_footage.mp4" \
    --scale 4 \
    --quality 15 \
    --format mp4 \
    --deduplicate \
    --dedup-threshold 0.98 \
    --enable-rife \
    --target-fps 24 \
    --audio-enhance \
    --scratch-sensitivity 0.5 \
    --grain-reduction 0.15 \
    --timeout 120 \
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
| `--auto-enhance` | Enable defect repair + face restore | off |
| `--no-face-restore` | Disable face restoration | enabled |
| `--no-defect-repair` | Disable scratch/dust repair | enabled |

**CRF Quality Guide:**
- `0-12` = Visually lossless (huge files)
- `15` = Archive quality (recommended)
- `18` = High quality
- `23` = Good quality (smaller files)
- `28+` = Web/streaming

**What `--auto-enhance` does:**
- Enables defect repair (scratches, dust, dirt)
- Enables face restoration (GFPGAN)
- Uses default sensitivity values

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

**Defect repair removes:** scratches, dust spots, dirt, film grain artifacts

**Grain reduction guide:**
- `0.0` = Keep all grain (authentic look)
- `0.15` = Light reduction (recommended for archive)
- `0.3` = Moderate (default)
- `0.5+` = Heavy (can look too smooth)

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

### B&W Archive (Recommended for Historical)
```bash
--scale 4 --quality 15 --format mp4 --deduplicate --dedup-threshold 0.98 --enable-rife --target-fps 24 --audio-enhance --scratch-sensitivity 0.5 --grain-reduction 0.15
```

### Archive Quality (Maximum - Color)
```bash
--scale 4 --quality 12 --auto-enhance --enable-rife --target-fps 24 --deduplicate --audio-enhance
```

### Fast Preview
```bash
--scale 2 --quality 23 --no-face-restore --no-defect-repair
```

### Old Film Restoration (Heavy Damage)
```bash
--scale 4 --auto-enhance --deduplicate --enable-rife --target-fps 24 --scratch-sensitivity 0.7 --grain-reduction 0.4
```

### Silent Film (Colorize)
```bash
--scale 4 --colorize --colorize-model ddcolor --auto-enhance --enable-rife --target-fps 18
```

---

## Monitoring & Management

### Check Job Status
```bash
framewright cloud status fw_46cd66213217
```
Returns: job state, progress %, instance info, elapsed time

### List All Jobs
```bash
framewright cloud jobs
```
Shows all submitted jobs with their IDs and states

### Download Completed Result
```bash
framewright cloud download fw_46cd66213217
framewright cloud download fw_46cd66213217 --output ./my_restored_video.mp4
```

### Cancel Running Job
```bash
framewright cloud cancel fw_46cd66213217
```
Stops processing and destroys the instance (stops billing)

### Check GPU Prices
```bash
framewright cloud gpus
```

### Check Vast.ai Balance
```bash
framewright cloud balance
```

---

## Job States

| State | Meaning |
|-------|---------|
| `pending` | Waiting for instance to start |
| `starting` | Instance booting, installing dependencies |
| `running` | Processing video |
| `uploading` | Uploading result to Google Drive |
| `completed` | Done - download available |
| `failed` | Error occurred |
| `cancelled` | Manually cancelled |

---

## Vast.ai Dashboard Monitoring

For detailed logs, go to [cloud.vast.ai/instances](https://cloud.vast.ai/instances/):

1. Find your instance (matches job submission time)
2. Click **"Logs"** button to see real-time output
3. Watch for:
   - `=== Downloading input ===` - rclone pulling from GDrive
   - `=== Starting restoration ===` - framewright running
   - `=== Uploading result ===` - sending back to GDrive
   - Any red error messages

### What Good Progress Looks Like
```
=== Installing dependencies ===
=== Configuring rclone ===
=== Installing Real-ESRGAN ===
=== Installing FrameWright ===
=== Downloading input from Google Drive ===
Transferred: 26.450M / 26.450 MBytes, 100%
=== Starting restoration ===
[1/4] Extracting frames... 100%
[2/4] Enhancing frames... 45%    <-- This takes longest
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

### Instance stuck on "starting" (>5 min)
GPU might be unavailable. Cancel and retry:
```bash
framewright cloud cancel fw_xxxxx
framewright cloud submit ...  # try again
```

### Job failed without output
1. Check Vast.ai logs for error message
2. Destroy instance manually to stop billing
3. Common causes: out of disk, GPU memory, network timeout

### Out of disk space
Default is 100GB. For very long videos, use `--scale 2` instead of 4.

### Timeout reached (job killed)
Default 120 min may not be enough. Use longer timeout:
```bash
--timeout 180   # 3 hours
--timeout 240   # 4 hours
```

### Manual Instance Cleanup
If something goes wrong, destroy the instance on Vast.ai to stop charges:
1. Go to https://cloud.vast.ai/instances/
2. Find the instance
3. Click **"Destroy"**

Or via vastai CLI:
```bash
vastai destroy instance <instance_id>
```

---

*Last updated: 2026-01-07*
