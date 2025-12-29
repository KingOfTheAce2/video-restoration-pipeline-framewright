# FrameWright Video Restoration Pipeline - Default Image
# This is a compatibility wrapper that builds the GPU-enabled image.
# For specific variants, use:
#   - Dockerfile.cpu  - Lightweight CPU-only image
#   - Dockerfile.gpu  - Full GPU support with CUDA 12.1
#
# Build: docker build -t framewright -f docker/Dockerfile .
# Run:   docker run --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output framewright restore --input /app/input/video.mp4

# ============================================================
# Stage 1: Builder - Install dependencies and download models
# ============================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -e ".[all,ui]"

# Download Real-ESRGAN
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && unzip -q realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d /opt/bin \
    && chmod +x /opt/bin/realesrgan-ncnn-vulkan \
    && rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip

# Download RIFE
RUN wget -q https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip \
    && unzip -q rife-ncnn-vulkan-20221029-ubuntu.zip -d /tmp/rife \
    && mkdir -p /opt/bin \
    && cp /tmp/rife/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan /opt/bin/ \
    && cp /tmp/rife/rife-ncnn-vulkan-20221029-ubuntu/*.bin /opt/bin/ \
    && cp /tmp/rife/rife-ncnn-vulkan-20221029-ubuntu/*.param /opt/bin/ \
    && chmod +x /opt/bin/rife-ncnn-vulkan \
    && rm -rf /tmp/rife

# ============================================================
# Stage 2: Runtime - Optimized production image
# ============================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

LABEL maintainer="FrameWright Team"
LABEL version="1.3.1"
LABEL description="AI-powered video restoration and enhancement pipeline"
LABEL org.opencontainers.image.source="https://github.com/framewright/framewright"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:/opt/bin:/usr/local/cuda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libvulkan1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/bin /opt/bin

WORKDIR /app
COPY --from=builder /build/src ./src
COPY --from=builder /build/pyproject.toml ./

RUN mkdir -p /app/input /app/output /app/models /root/.cache/framewright

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "from framewright.config import Config; print('OK')" || exit 1

ENTRYPOINT ["framewright"]
CMD ["--help"]
