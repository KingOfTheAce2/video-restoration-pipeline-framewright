# Troubleshooting

Problem → Solution. Copy-paste ready.

## Out of Memory

```bash
# Fix: Reduce tile size
framewright restore video.mp4 --tile 256

# Still failing? Go smaller
framewright restore video.mp4 --tile 128

# Use FP16 precision
framewright restore video.mp4 --tile 128 --precision fp16

# Last resort: CPU fallback
framewright restore video.mp4 --tile 64 --scale 2 --preset fast --allow-cpu
```

Also: Close Chrome, Discord, games - they use GPU memory.

## FFmpeg Not Found

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:** Download from https://ffmpeg.org/download.html, add to PATH.

## No GPU Detected

```bash
# Check GPU
framewright-check

# NVIDIA: Install drivers + CUDA
# https://www.nvidia.com/drivers

# AMD: Install ROCm (Linux) or use Vulkan
framewright restore video.mp4 --backend ncnn

# Intel: Install OpenVINO
pip install openvino
framewright restore video.mp4 --backend openvino

# Apple Silicon: Should auto-detect
framewright restore video.mp4 --backend coreml
```

## Backend Not Available

| Error | Fix |
|-------|-----|
| `TensorRTBackendError` | Install TensorRT: `pip install tensorrt` |
| `CoreMLBackendError` | macOS only, check Xcode tools |
| `ROCmBackendError` | Install ROCm toolkit (Linux AMD) |
| `OpenVINOBackendError` | `pip install openvino` |
| `NCNNBackendError` | `framewright install-ncnn-vulkan` |

## Slow Processing

| Problem | Fix |
|---------|-----|
| Using CPU instead of GPU | Check `framewright-check`, install GPU drivers |
| Using HDD | Move files to SSD |
| High quality preset | Use `--preset fast` for testing |
| TensorRT not enabled | `--backend tensorrt` (NVIDIA) |

```bash
# Speed up processing
framewright restore video.mp4 --preset fast --scale 2 --backend tensorrt
```

## Model Download Failed

```bash
# Check internet
ping huggingface.co

# Manual download location
~/.framewright/models/

# Or specify custom path
framewright restore video.mp4 --model-dir /path/to/models/

# Clear model cache
rm -rf ~/.framewright/models/
```

## Output Quality Issues

| Issue | Fix |
|-------|-----|
| Blurry | `--preset quality --crf 16` |
| Artifacts | `--enable-qp-artifact-removal` |
| Flickering | `--temporal-method cross-attention` |
| Colors wrong | Remove `--colorize` flag |
| Faces look weird | `--face-model codeformer` or `--no-face-restore` |
| Temporal inconsistency | `--temporal-consistency full` |
| Grain lost | `--grain-mode preserve` |

## Resume Failed Job

```bash
# Resume from checkpoint (automatic)
framewright restore video.mp4 --output ./my_project/

# Checkpoint corrupted? Clear it
rm -rf ./my_project/.framewright/checkpoint.json

# Force fresh start
framewright restore video.mp4 --no-resume
```

## Apple Silicon Issues

```bash
# Check CoreML availability
framewright-check --apple-silicon

# Force CoreML backend
framewright restore video.mp4 --backend coreml

# Disable ANE if unstable
framewright restore video.mp4 --backend coreml --no-ane

# Fall back to CPU
framewright restore video.mp4 --backend cpu
```

## AMD GPU Issues

```bash
# Check ROCm installation (Linux)
rocm-smi

# Use Vulkan instead
framewright restore video.mp4 --backend ncnn

# Install NCNN-Vulkan
framewright install-ncnn-vulkan
```

## Intel GPU Issues

```bash
# Check OpenVINO
python -c "import openvino; print(openvino.__version__)"

# Use OpenVINO backend
framewright restore video.mp4 --backend openvino

# Fall back to Vulkan
framewright restore video.mp4 --backend ncnn
```

## Installation Failed

```bash
# Upgrade pip first
pip install --upgrade pip

# Use virtual environment
python -m venv framewright_env
source framewright_env/bin/activate  # Linux/Mac
framewright_env\Scripts\activate     # Windows
pip install framewright

# Install with all optional dependencies
pip install framewright[full]
```

## Permission Denied

```bash
# Use a different output location
framewright restore video.mp4 -o ~/Videos/restored.mp4
```

## Cloud Processing Issues

```bash
# Check Vast.ai connection
framewright cloud status

# Check credentials
cat ~/.framewright/vastai.env

# Test Google Drive access
rclone ls gdrive:

# Debug mode
framewright cloud submit video.mp4 --debug
```

## Get Help

```bash
# Generate diagnostic info
framewright-check > report.txt

# Verbose logging
framewright restore video.mp4 --verbose

# Debug mode (very verbose)
framewright restore video.mp4 --debug

# Logs location
# Linux/Mac: ~/.framewright/logs/
# Windows: %USERPROFILE%\.framewright\logs\
```

## Error Quick Reference

| Error | Meaning | Fix |
|-------|---------|-----|
| `OutOfMemoryError` | Not enough VRAM | `--tile 128 --precision fp16` |
| `DependencyError` | FFmpeg missing | Install FFmpeg |
| `GPUError` | Driver issue | Update GPU drivers |
| `ModelError` | Download failed | Check internet, clear cache |
| `DiskSpaceError` | Full disk | Free up space |
| `BackendError` | Backend unavailable | Try different `--backend` |
| `CheckpointError` | Corrupted checkpoint | Delete checkpoint.json |
| `TemporalError` | Frame consistency failed | Reduce temporal window |
| `ConfigError` | Invalid settings | Check config syntax |
| `PipelineError` | Stage failed | Check logs for details |

## Still Stuck?

1. Run `framewright-check > diagnostic.txt`
2. Try `--verbose` mode
3. Check logs in `~/.framewright/logs/`
4. Open an issue with the diagnostic output
