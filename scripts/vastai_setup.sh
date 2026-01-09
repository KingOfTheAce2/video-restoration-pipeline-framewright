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

echo "=== Uploading results to Google Drive ==="
# Upload frames
if [ -d "/workspace/project/enhanced" ]; then
    rclone copy "/workspace/project/enhanced" "$FRAMES_OUTPUT_PATH" --progress
    echo "Enhanced frames uploaded to: $FRAMES_OUTPUT_PATH"
fi
if [ -d "/workspace/project/unique_frames" ]; then
    rclone copy "/workspace/project/unique_frames" "$UNIQUE_FRAMES_PATH" --progress
fi

# Upload video
OUTPUT_EXT="${OUTPUT_PATH##*.}"
VIDEO_FILE=$(find /workspace/project -maxdepth 1 -name "*.$OUTPUT_EXT" 2>/dev/null | head -1)
if [ -n "$VIDEO_FILE" ]; then
    rclone copyto "$VIDEO_FILE" "$OUTPUT_PATH" --progress
    echo "Video uploaded to: $OUTPUT_PATH"
else
    echo "Warning: No video file found"
fi

echo "=== Saving logs ==="
cp /var/log/onstart*.log /workspace/ 2>/dev/null || true
cat /workspace/*.log > /workspace/job_log.txt 2>/dev/null || true
rclone copy /workspace/job_log.txt "$LOGS_PATH" --quiet || true

echo "=== Done! Shutting down instance ==="
INSTANCE_ID=$(echo $VAST_CONTAINERLABEL | tr -d '[:space:]')
if [ -n "$INSTANCE_ID" ] && [ -n "$VASTAI_API_KEY" ]; then
    curl -s -X DELETE "https://cloud.vast.ai/api/v0/instances/$INSTANCE_ID/?api_key=$VASTAI_API_KEY" || true
fi
