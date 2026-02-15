# Hardware Guide

FrameWright v1.0 supports **all hardware tiers** from CPU-only to RTX 5090 and cloud GPUs.

## Supported Platforms

| Platform | Backend | Auto-Detected | Features |
|----------|---------|---------------|----------|
| **NVIDIA** | CUDA + TensorRT | ✅ Yes | FP16, INT8, dynamic batching |
| **AMD** | ROCm / HIP / Vulkan | ✅ Yes | Full GPU acceleration |
| **Intel** | oneAPI / OpenVINO / Vulkan | ✅ Yes | Optimized inference |
| **Apple Silicon** | CoreML / Metal / ANE | ✅ Yes | M1-M4 Neural Engine |
| **CPU** | OpenCV / NumPy | ✅ Yes | Fallback (slower) |

## GPU Tiers

| Your GPU | VRAM | What You Can Do |
|----------|------|-----------------|
| No GPU / Integrated | - | `--preset fast --allow-cpu` (slow) |
| Entry (GTX 1650) | 4GB | 1080p with `--preset fast` |
| Mid-range (RTX 3060) | 8-12GB | 4K with `--preset quality` |
| High-end (RTX 4080) | 16GB | `--preset ultimate`, all features |
| Enthusiast (RTX 4090/5090) | 24-32GB | Maximum settings, no tiling |
| Apple M1/M2 | 8-16GB unified | CoreML acceleration |
| Apple M3/M4 Pro/Max | 18-128GB unified | Full pipeline, ANE boost |

## Quick Settings by Hardware

```bash
# CPU-only (emergency fallback)
framewright restore video.mp4 --preset fast --allow-cpu --tile 64

# 4GB VRAM (GTX 1650, etc.)
framewright restore video.mp4 --preset fast --tile 128

# 8GB VRAM (RTX 3070, RX 6700 XT)
framewright restore video.mp4 --preset quality --tile 256

# 12GB VRAM (RTX 3060/4070)
framewright restore video.mp4 --preset archive --tile 384

# 16GB+ VRAM (RTX 4080)
framewright restore video.mp4 --preset ultimate --tile 512

# 24GB+ VRAM (RTX 4090/5090)
framewright restore video.mp4 --preset ultimate  # No tiling needed

# Apple Silicon (M1/M2/M3/M4)
framewright restore video.mp4 --backend coreml --preset quality
```

## Backend Selection

FrameWright auto-detects the best backend, but you can override:

```bash
# Force TensorRT (NVIDIA, fastest)
framewright restore video.mp4 --backend tensorrt

# Force CUDA (NVIDIA, most compatible)
framewright restore video.mp4 --backend cuda

# Force CoreML (Apple Silicon)
framewright restore video.mp4 --backend coreml

# Force ROCm (AMD)
framewright restore video.mp4 --backend rocm

# Force OpenVINO (Intel)
framewright restore video.mp4 --backend openvino

# Force NCNN-Vulkan (any GPU)
framewright restore video.mp4 --backend ncnn
```

## TensorRT Acceleration (NVIDIA)

For fastest processing on NVIDIA GPUs:

```bash
# Enable TensorRT with FP16
framewright restore video.mp4 --backend tensorrt --precision fp16

# Enable INT8 quantization (faster, slightly lower quality)
framewright restore video.mp4 --backend tensorrt --precision int8

# Engine caching (first run slower, subsequent runs fast)
framewright restore video.mp4 --backend tensorrt --cache-engines
```

## Apple Silicon Optimization

```bash
# Auto-detect M-series chip
framewright restore video.mp4  # CoreML auto-enabled on Mac

# Force ANE (Neural Engine) usage
framewright restore video.mp4 --backend coreml --use-ane

# Check Apple Silicon capabilities
framewright-check --apple-silicon
```

## AMD GPU Support

```bash
# ROCm backend (Linux)
framewright restore video.mp4 --backend rocm

# HIP backend
framewright restore video.mp4 --backend hip

# Vulkan fallback (Windows/Linux)
framewright restore video.mp4 --backend ncnn
```

## Intel GPU Support

```bash
# OpenVINO backend (recommended)
framewright restore video.mp4 --backend openvino

# oneAPI backend
framewright restore video.mp4 --backend oneapi

# Vulkan fallback
framewright restore video.mp4 --backend ncnn
```

## Out of Memory?

Try these in order:

```bash
# Step 1: Reduce tile size
framewright restore video.mp4 --tile 256

# Step 2: Smaller tiles
framewright restore video.mp4 --tile 128

# Step 3: Lower scale
framewright restore video.mp4 --tile 128 --scale 2

# Step 4: Fast preset with FP16
framewright restore video.mp4 --preset fast --tile 128 --precision fp16

# Step 5: CPU fallback (last resort)
framewright restore video.mp4 --preset fast --tile 64 --allow-cpu
```

## Storage Requirements

| Video Length | Temp Space Needed |
|--------------|-------------------|
| 1 minute | ~10GB |
| 10 minutes | ~100GB |
| 1 hour | ~500GB |
| Feature film (2h) | ~1TB |

Use NVMe SSD for 2-3x faster processing.

## Multi-GPU Support

```bash
# Auto-detect all GPUs
framewright restore video.mp4 --multi-gpu

# Specific GPUs
framewright restore video.mp4 --gpu-ids 0,1

# Distributed across machines (cloud)
framewright restore video.mp4 --distributed --workers 4
```

## Cloud GPU Options

No local GPU? Use cloud:

```bash
# Vast.ai (~$0.30/hr for RTX 4090)
framewright cloud submit --gdrive-input "video.mp4" --gpu RTX_4090

# RunPod
framewright cloud submit --provider runpod --gpu A100

# Lambda Labs
framewright cloud submit --provider lambda --gpu H100
```

## Check Your Hardware

```bash
framewright-check
```

Shows: GPU model, VRAM, detected backends, recommended settings.

## VRAM Usage by Feature

| Feature | VRAM Required |
|---------|---------------|
| Basic upscaling (2x) | 2GB |
| Quality upscaling (4x) | 4GB |
| Face restoration | +1GB |
| Colorization | +2GB |
| Frame interpolation | +2GB |
| Diffusion SR | +8GB |
| Ultimate preset | 12GB+ |
