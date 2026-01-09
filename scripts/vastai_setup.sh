#!/bin/bash
# FrameWright Vast.ai Cloud Setup Script
# This script is downloaded and executed by the minimal onstart script
# All configuration comes from environment variables

set -e
cd /workspace

echo "=== Installing dependencies ==="
apt-get update && apt-get install -y rclone wget unzip xz-utils curl

echo "=== Installing FFmpeg with full codec support ==="
# Replace conda's FFmpeg (lacks libx265) with static build
if [ -f /opt/conda/bin/ffmpeg ]; then
    mv /opt/conda/bin/ffmpeg /opt/conda/bin/ffmpeg.bak 2>/dev/null || true
    mv /opt/conda/bin/ffprobe /opt/conda/bin/ffprobe.bak 2>/dev/null || true
fi
cd /tmp
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
rm -rf ffmpeg-*
cd /workspace
export PATH="/usr/local/bin:$PATH"
echo "FFmpeg: $(ffmpeg -version 2>&1 | head -1)"

echo "=== Configuring rclone ==="
mkdir -p ~/.config/rclone
echo "$RCLONE_CONFIG_B64" | base64 -d > ~/.config/rclone/rclone.conf

# Show PyTorch version (stock PyTorch 2.1.0 works with RTX 30xx/40xx)
# NOTE: RTX 50-series (Blackwell/sm_120) is NOT yet supported by PyTorch
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "Detected GPU: $GPU_NAME"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Check for unsupported GPU
if echo "$GPU_NAME" | grep -qE "RTX 50[0-9]{2}|Blackwell"; then
    echo "ERROR: RTX 50-series GPUs are not yet supported by PyTorch."
    echo "Please use RTX 4090 or RTX 3090 instead."
    exit 1
fi

echo "=== Installing Real-ESRGAN (PyTorch/CUDA) ==="
pip install "numpy<2.0" basicsr facexlib gfpgan realesrgan

echo "=== Installing FrameWright ==="
pip install git+https://github.com/KingOfTheAce2/video-restoration-pipeline-framewright.git yt-dlp imagehash
export FRAMEWRIGHT_BACKEND=pytorch

echo "=== Downloading input from Google Drive ==="
rclone copyto "$INPUT_PATH" /workspace/input.mp4 --progress

echo "=== Starting restoration ==="
# RESTORE_CMD is passed as env var
eval "$RESTORE_CMD"

echo "=== Uploading all intermediate results to Google Drive ==="
# This allows recovery from any stage if something fails

# 1. Upload deduplicated/unique frames (stage 1 output)
if [ -d "/workspace/project/unique_frames" ]; then
    echo "Uploading deduplicated frames..."
    rclone copy "/workspace/project/unique_frames" "$UNIQUE_FRAMES_PATH" --progress
    echo "Deduplicated frames: $UNIQUE_FRAMES_PATH"
fi

# 2. Upload enhanced frames (stage 2 output - after Real-ESRGAN)
if [ -d "/workspace/project/enhanced" ]; then
    echo "Uploading enhanced frames..."
    rclone copy "/workspace/project/enhanced" "$FRAMES_OUTPUT_PATH" --progress
    echo "Enhanced frames: $FRAMES_OUTPUT_PATH"
fi

# 3. Upload interpolated frames if RIFE was used (stage 3 output)
if [ -d "/workspace/project/interpolated" ]; then
    echo "Uploading interpolated frames..."
    INTERP_PATH="${FRAMES_OUTPUT_PATH%/}_interpolated/"
    rclone copy "/workspace/project/interpolated" "$INTERP_PATH" --progress
    echo "Interpolated frames: $INTERP_PATH"
fi

# 4. Upload final video
OUTPUT_EXT="${OUTPUT_PATH##*.}"
VIDEO_FILE=$(find /workspace/project -maxdepth 1 -name "*.$OUTPUT_EXT" 2>/dev/null | head -1)
if [ -n "$VIDEO_FILE" ]; then
    echo "Uploading final video..."
    rclone copyto "$VIDEO_FILE" "$OUTPUT_PATH" --progress
    echo "Final video: $OUTPUT_PATH"
else
    echo "Warning: No video file found"
fi

echo ""
echo "=== Output Summary ==="
echo "Deduplicated frames: $UNIQUE_FRAMES_PATH"
echo "Enhanced frames: $FRAMES_OUTPUT_PATH"
[ -d "/workspace/project/interpolated" ] && echo "Interpolated frames: ${FRAMES_OUTPUT_PATH%/}_interpolated/"
echo "Final video: $OUTPUT_PATH"

echo "=== Saving logs ==="
cp /var/log/onstart*.log /workspace/ 2>/dev/null || true
cat /workspace/*.log > /workspace/job_log.txt 2>/dev/null || true
rclone copy /workspace/job_log.txt "$LOGS_PATH" --quiet || true

echo "=== Done! Shutting down instance ==="
INSTANCE_ID=$(echo $VAST_CONTAINERLABEL | tr -d '[:space:]')
if [ -n "$INSTANCE_ID" ] && [ -n "$VASTAI_API_KEY" ]; then
    curl -s -X DELETE "https://cloud.vast.ai/api/v0/instances/$INSTANCE_ID/?api_key=$VASTAI_API_KEY" || true
fi
