#!/bin/bash
# FrameWright Vast.ai Cloud Setup Script
# This script is downloaded and executed by the minimal onstart script
# All configuration comes from environment variables

set -e
cd /workspace

# Configurable sync interval (default: 5 minutes)
SYNC_INTERVAL=${SYNC_INTERVAL:-300}

echo "=== Installing dependencies ==="
apt-get update && apt-get install -y rclone wget unzip xz-utils curl libgl1-mesa-glx libglib2.0-0

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

# =============================================================================
# Background Sync Function - uploads frames periodically during processing
# =============================================================================
background_sync() {
    echo "[SYNC] Starting background sync (interval: ${SYNC_INTERVAL}s)"

    WORK_DIR="/workspace/project/.framewright_work/temp"
    INTERP_PATH="${FRAMES_OUTPUT_PATH%/}_interpolated/"

    LAST_ENHANCED=0
    LAST_INTERPOLATED=0
    SYNC_COUNT=0

    while true; do
        sleep "$SYNC_INTERVAL"
        SYNC_COUNT=$((SYNC_COUNT + 1))

        echo ""
        echo "[SYNC #$SYNC_COUNT] $(date '+%Y-%m-%d %H:%M:%S') - Checking for new frames..."

        # Sync unique/deduplicated frames (only first time they appear)
        if [ -d "$WORK_DIR/unique_frames" ]; then
            UNIQUE_COUNT=$(find "$WORK_DIR/unique_frames" -name "*.png" 2>/dev/null | wc -l)
            if [ "$UNIQUE_COUNT" -gt 0 ]; then
                echo "[SYNC] Uploading $UNIQUE_COUNT unique frames..."
                rclone copy "$WORK_DIR/unique_frames" "$UNIQUE_FRAMES_PATH" --quiet --transfers 8
                echo "[SYNC] Unique frames synced: $UNIQUE_FRAMES_PATH"
            fi
        fi

        # Sync enhanced frames (incremental - only new files)
        if [ -d "$WORK_DIR/enhanced" ]; then
            ENHANCED_COUNT=$(find "$WORK_DIR/enhanced" -name "*.png" 2>/dev/null | wc -l)
            if [ "$ENHANCED_COUNT" -gt "$LAST_ENHANCED" ]; then
                NEW_ENHANCED=$((ENHANCED_COUNT - LAST_ENHANCED))
                echo "[SYNC] Uploading $NEW_ENHANCED new enhanced frames (total: $ENHANCED_COUNT)..."
                rclone copy "$WORK_DIR/enhanced" "$FRAMES_OUTPUT_PATH" --quiet --transfers 8
                LAST_ENHANCED=$ENHANCED_COUNT
                echo "[SYNC] Enhanced frames synced: $FRAMES_OUTPUT_PATH"
            fi
        fi

        # Also check project/enhanced (some configs output here)
        if [ -d "/workspace/project/enhanced" ]; then
            ENHANCED_COUNT=$(find "/workspace/project/enhanced" -name "*.png" 2>/dev/null | wc -l)
            if [ "$ENHANCED_COUNT" -gt "$LAST_ENHANCED" ]; then
                NEW_ENHANCED=$((ENHANCED_COUNT - LAST_ENHANCED))
                echo "[SYNC] Uploading $NEW_ENHANCED new enhanced frames (total: $ENHANCED_COUNT)..."
                rclone copy "/workspace/project/enhanced" "$FRAMES_OUTPUT_PATH" --quiet --transfers 8
                LAST_ENHANCED=$ENHANCED_COUNT
                echo "[SYNC] Enhanced frames synced: $FRAMES_OUTPUT_PATH"
            fi
        fi

        # Sync interpolated frames if RIFE is running
        if [ -d "$WORK_DIR/interpolated" ]; then
            INTERP_COUNT=$(find "$WORK_DIR/interpolated" -name "*.png" 2>/dev/null | wc -l)
            if [ "$INTERP_COUNT" -gt "$LAST_INTERPOLATED" ]; then
                NEW_INTERP=$((INTERP_COUNT - LAST_INTERPOLATED))
                echo "[SYNC] Uploading $NEW_INTERP new interpolated frames (total: $INTERP_COUNT)..."
                rclone copy "$WORK_DIR/interpolated" "$INTERP_PATH" --quiet --transfers 8
                LAST_INTERPOLATED=$INTERP_COUNT
                echo "[SYNC] Interpolated frames synced: $INTERP_PATH"
            fi
        fi

        # Also check project/interpolated
        if [ -d "/workspace/project/interpolated" ]; then
            INTERP_COUNT=$(find "/workspace/project/interpolated" -name "*.png" 2>/dev/null | wc -l)
            if [ "$INTERP_COUNT" -gt "$LAST_INTERPOLATED" ]; then
                NEW_INTERP=$((INTERP_COUNT - LAST_INTERPOLATED))
                echo "[SYNC] Uploading $NEW_INTERP new interpolated frames (total: $INTERP_COUNT)..."
                rclone copy "/workspace/project/interpolated" "$INTERP_PATH" --quiet --transfers 8
                LAST_INTERPOLATED=$INTERP_COUNT
                echo "[SYNC] Interpolated frames synced: $INTERP_PATH"
            fi
        fi

        echo "[SYNC #$SYNC_COUNT] Complete. Enhanced: $LAST_ENHANCED, Interpolated: $LAST_INTERPOLATED"
    done
}

# Start background sync in a subshell
echo "=== Starting background sync process ==="
background_sync &
SYNC_PID=$!
echo "Background sync PID: $SYNC_PID (syncing every ${SYNC_INTERVAL}s)"

# Trap to kill background sync on exit
cleanup() {
    echo "Stopping background sync..."
    kill $SYNC_PID 2>/dev/null || true
    wait $SYNC_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Starting restoration ==="
# RESTORE_CMD is passed as env var
eval "$RESTORE_CMD"
RESTORE_EXIT_CODE=$?

# Stop background sync
echo "=== Stopping background sync ==="
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

echo "=== Final upload of all results to Google Drive ==="
# Final sync to catch any remaining frames

WORK_DIR="/workspace/project/.framewright_work/temp"
INTERP_PATH="${FRAMES_OUTPUT_PATH%/}_interpolated/"

# 1. Upload deduplicated/unique frames
if [ -d "$WORK_DIR/unique_frames" ]; then
    echo "Final sync: unique frames..."
    rclone copy "$WORK_DIR/unique_frames" "$UNIQUE_FRAMES_PATH" --progress --transfers 8
    echo "Deduplicated frames: $UNIQUE_FRAMES_PATH"
elif [ -d "/workspace/project/unique_frames" ]; then
    echo "Final sync: unique frames..."
    rclone copy "/workspace/project/unique_frames" "$UNIQUE_FRAMES_PATH" --progress --transfers 8
    echo "Deduplicated frames: $UNIQUE_FRAMES_PATH"
fi

# 2. Upload enhanced frames
if [ -d "$WORK_DIR/enhanced" ]; then
    echo "Final sync: enhanced frames..."
    rclone copy "$WORK_DIR/enhanced" "$FRAMES_OUTPUT_PATH" --progress --transfers 8
    echo "Enhanced frames: $FRAMES_OUTPUT_PATH"
elif [ -d "/workspace/project/enhanced" ]; then
    echo "Final sync: enhanced frames..."
    rclone copy "/workspace/project/enhanced" "$FRAMES_OUTPUT_PATH" --progress --transfers 8
    echo "Enhanced frames: $FRAMES_OUTPUT_PATH"
fi

# 3. Upload interpolated frames if RIFE was used
if [ -d "$WORK_DIR/interpolated" ]; then
    echo "Final sync: interpolated frames..."
    rclone copy "$WORK_DIR/interpolated" "$INTERP_PATH" --progress --transfers 8
    echo "Interpolated frames: $INTERP_PATH"
elif [ -d "/workspace/project/interpolated" ]; then
    echo "Final sync: interpolated frames..."
    rclone copy "/workspace/project/interpolated" "$INTERP_PATH" --progress --transfers 8
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
    echo "Warning: No video file found (restoration may have failed)"
fi

echo ""
echo "=== Output Summary ==="
echo "Deduplicated frames: $UNIQUE_FRAMES_PATH"
echo "Enhanced frames: $FRAMES_OUTPUT_PATH"
[ -d "$WORK_DIR/interpolated" ] || [ -d "/workspace/project/interpolated" ] && echo "Interpolated frames: $INTERP_PATH"
echo "Final video: $OUTPUT_PATH"

echo "=== Saving logs ==="
cp /var/log/onstart*.log /workspace/ 2>/dev/null || true
cat /workspace/*.log > /workspace/job_log.txt 2>/dev/null || true
rclone copy /workspace/job_log.txt "$LOGS_PATH" --quiet || true

# Only shutdown if restoration succeeded
if [ "$RESTORE_EXIT_CODE" -eq 0 ]; then
    echo "=== Done! Shutting down instance ==="
    INSTANCE_ID=$(echo $VAST_CONTAINERLABEL | tr -d '[:space:]')
    if [ -n "$INSTANCE_ID" ] && [ -n "$VASTAI_API_KEY" ]; then
        curl -s -X DELETE "https://cloud.vast.ai/api/v0/instances/$INSTANCE_ID/?api_key=$VASTAI_API_KEY" || true
    fi
else
    echo "=== Restoration failed (exit code: $RESTORE_EXIT_CODE) - NOT shutting down ==="
    echo "Check logs and frames uploaded so far. Instance left running for debugging."
fi
