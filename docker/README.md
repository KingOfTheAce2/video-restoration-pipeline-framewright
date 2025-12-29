# FrameWright Docker Configuration

Docker images and compose files for running FrameWright video restoration pipeline.

## Quick Start

### CPU Processing

```bash
# Build the CPU image
docker-compose build framewright-cpu

# Process a video
docker-compose run --rm framewright-cpu restore \
  --input /app/input/video.mp4 \
  --output /app/output/restored.mp4

# Start the web UI
docker-compose up framewright-ui
# Open http://localhost:7860 in your browser
```

### GPU Processing

```bash
# Build the GPU image
docker-compose build framewright-gpu

# Process a video with GPU acceleration
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml \
  run --rm framewright-gpu restore \
  --input /app/input/video.mp4 \
  --output /app/output/restored.mp4

# Start the GPU-accelerated web UI
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up framewright-ui-gpu
```

## Available Images

| Image | Description | Base | Size |
|-------|-------------|------|------|
| `framewright:cpu` | CPU-only, lightweight | Python 3.11 slim | ~1.5GB |
| `framewright:gpu` | Full GPU support | CUDA 12.1 runtime | ~5GB |

## Services

| Service | Description | Port |
|---------|-------------|------|
| `framewright-cpu` | CLI processing (CPU) | - |
| `framewright-gpu` | CLI processing (GPU) | - |
| `framewright-ui` | Web UI (CPU) | 7860 |
| `framewright-ui-gpu` | Web UI (GPU) | 7860 |
| `framewright-watch` | Batch processing | - |
| `framewright-watch-gpu` | Batch processing (GPU) | - |

## GPU Setup Guide

### Prerequisites

1. **NVIDIA GPU** with CUDA 12.1+ support
2. **NVIDIA Drivers** (version 525.60.13 or later)
3. **Docker** (version 19.03 or later)
4. **nvidia-container-toolkit**

### Installing nvidia-container-toolkit

#### Ubuntu/Debian

```bash
# Add NVIDIA container toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Verify Installation

```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `./input` | Input videos directory |
| `OUTPUT_DIR` | `./output` | Output videos directory |
| `MODEL_DIR` | `./models` | AI models directory |
| `UI_PORT` | `7860` | Web UI port |
| `PRESET` | `quality` | Processing preset |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection |

### Volume Mounts

```yaml
volumes:
  - ./input:/app/input:ro    # Input videos (read-only)
  - ./output:/app/output     # Output videos
  - ./models:/app/models     # AI models (cached)
```

## Common Configurations

### Use Specific GPU

```bash
# Use only GPU 0
NVIDIA_VISIBLE_DEVICES=0 docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up framewright-gpu

# Use GPUs 0 and 1
NVIDIA_VISIBLE_DEVICES=0,1 docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up framewright-gpu
```

### Limit Memory Usage

```bash
# Limit shared memory (for DataLoader workers)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml \
  run --rm --shm-size=4g framewright-gpu restore ...
```

### Custom Port for Web UI

```bash
UI_PORT=8080 docker-compose up framewright-ui
# Access at http://localhost:8080
```

### Batch Processing with Watch Mode

```bash
# Create directories
mkdir -p watch output processed

# Start watch service
docker-compose up -d framewright-watch

# Add videos to ./watch directory - they will be processed automatically
```

## Building Custom Images

### Build Both Images

```bash
docker-compose build
```

### Build Specific Image

```bash
# CPU only
docker build -t framewright:cpu -f docker/Dockerfile.cpu .

# GPU
docker build -t framewright:gpu -f docker/Dockerfile.gpu .
```

### Build with Custom Options

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t framewright:custom .
```

## Troubleshooting

### GPU Not Detected

```bash
# Check if GPU is visible
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# If not working, check nvidia-container-toolkit installation
nvidia-ctk --version

# Restart Docker
sudo systemctl restart docker
```

### Out of Memory Errors

1. **Reduce batch size** in processing options
2. **Increase shared memory**:
   ```bash
   docker-compose run --shm-size=16g framewright-gpu ...
   ```
3. **Use single GPU**:
   ```bash
   NVIDIA_VISIBLE_DEVICES=0 docker-compose ...
   ```

### Permission Denied on Output Directory

```bash
# Ensure output directory exists and is writable
mkdir -p output
chmod 777 output

# Or run container as current user
docker-compose run --rm --user $(id -u):$(id -g) framewright-cpu ...
```

### Container Exits Immediately

```bash
# Check logs
docker-compose logs framewright-cpu

# Run interactively to debug
docker-compose run --rm framewright-cpu /bin/bash
```

### Model Download Fails

```bash
# Pre-download models
docker-compose run --rm framewright-cpu python -c "from framewright.utils.model_manager import download_models; download_models()"

# Or mount local models directory
docker-compose run -v /path/to/models:/app/models framewright-cpu ...
```

### Health Check Fails

```bash
# Check container status
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' framewright-cpu

# Manually test
docker-compose exec framewright-cpu python -c "from framewright.config import Config; print('OK')"
```

## Performance Tips

1. **Use GPU** for 10-50x faster processing
2. **Mount models directory** to avoid re-downloading
3. **Use SSD storage** for input/output directories
4. **Enable TensorFloat-32** on Ampere GPUs (enabled by default)
5. **Use `quality` preset** for best results, `fast` for quick previews

## Docker Commands Reference

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d framewright-ui

# Stop services
docker-compose down

# View logs
docker-compose logs -f framewright-ui

# Clean up
docker-compose down -v --rmi local

# Process single video
docker-compose run --rm framewright-cpu restore \
  --input /app/input/video.mp4 \
  --output /app/output/restored.mp4 \
  --preset quality

# Interactive shell
docker-compose run --rm framewright-cpu /bin/bash
```

## Support

- Documentation: [GitHub README](https://github.com/framewright/framewright)
- Issues: [GitHub Issues](https://github.com/framewright/framewright/issues)
- Docker Hub: [framewright/framewright](https://hub.docker.com/r/framewright/framewright)
