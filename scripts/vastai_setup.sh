#!/bin/bash
# FrameWright Vast.ai Cloud Setup Script
# This script is downloaded and executed by the minimal onstart script
# All configuration comes from environment variables
#
# RESUME SUPPORT:
#   Set RESUME_FROM to skip completed stages:
#   - "extract"      Start fresh (default)
#   - "dedupe"       Skip extraction, download raw frames from GDrive
#   - "enhance"      Skip extraction+dedupe, download unique frames
#   - "interpolate"  Skip to interpolation, download enhanced frames
#   - "encode"       Skip to encoding, download interpolated frames

set -e
cd /workspace

# Configurable sync interval (default: 5 minutes)
SYNC_INTERVAL=${SYNC_INTERVAL:-300}
RESUME_FROM=${RESUME_FROM:-extract}

echo "=============================================="
echo "  FrameWright Cloud Restoration Pipeline"
echo "=============================================="
echo "Resume from: $RESUME_FROM"
echo "Sync interval: ${SYNC_INTERVAL}s"
echo ""

echo "=== Installing dependencies ==="
apt-get update && apt-get install -y rclone wget unzip xz-utils curl libgl1-mesa-glx libglib2.0-0 bc

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

# Show PyTorch version
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

# =============================================================================
# Resume Support - Download existing frames from GDrive
# =============================================================================
WORK_DIR="/workspace/project/.framewright_work/temp"
INTERP_PATH="${FRAMES_OUTPUT_PATH%/}_interpolated/"

mkdir -p "$WORK_DIR/frames"
mkdir -p "$WORK_DIR/unique_frames"
mkdir -p "$WORK_DIR/enhanced"
mkdir -p "$WORK_DIR/interpolated"
mkdir -p /workspace/project

download_existing_frames() {
    local stage=$1
    echo ""
    echo "=== Resuming from stage: $stage ==="

    case $stage in
        dedupe)
            # Need raw extracted frames - check if they exist in GDrive
            echo "Checking for existing raw frames..."
            # Raw frames aren't typically uploaded, so we need the input video
            echo "Downloading input video..."
            rclone copyto "$INPUT_PATH" /workspace/input.mp4 --progress
            echo "Will re-extract frames from video"
            ;;
        enhance)
            # Download unique/deduplicated frames
            echo "Downloading deduplicated frames from GDrive..."
            if rclone lsf "$UNIQUE_FRAMES_PATH" 2>/dev/null | head -1 | grep -q .; then
                rclone copy "$UNIQUE_FRAMES_PATH" "$WORK_DIR/unique_frames" --progress --transfers 8
                FRAME_COUNT=$(find "$WORK_DIR/unique_frames" -name "*.png" | wc -l)
                echo "Downloaded $FRAME_COUNT unique frames"
            else
                echo "ERROR: No unique frames found at $UNIQUE_FRAMES_PATH"
                echo "Cannot resume from 'enhance' stage. Try 'extract' instead."
                exit 1
            fi
            ;;
        interpolate)
            # Download enhanced frames
            echo "Downloading enhanced frames from GDrive..."
            if rclone lsf "$FRAMES_OUTPUT_PATH" 2>/dev/null | head -1 | grep -q .; then
                rclone copy "$FRAMES_OUTPUT_PATH" "$WORK_DIR/enhanced" --progress --transfers 8
                FRAME_COUNT=$(find "$WORK_DIR/enhanced" -name "*.png" | wc -l)
                echo "Downloaded $FRAME_COUNT enhanced frames"
            else
                echo "ERROR: No enhanced frames found at $FRAMES_OUTPUT_PATH"
                echo "Cannot resume from 'interpolate' stage. Try 'enhance' instead."
                exit 1
            fi
            ;;
        encode)
            # Download interpolated frames (or enhanced if no interpolation)
            echo "Downloading frames for encoding..."
            if rclone lsf "$INTERP_PATH" 2>/dev/null | head -1 | grep -q .; then
                echo "Downloading interpolated frames..."
                rclone copy "$INTERP_PATH" "$WORK_DIR/interpolated" --progress --transfers 8
                FRAME_COUNT=$(find "$WORK_DIR/interpolated" -name "*.png" | wc -l)
                echo "Downloaded $FRAME_COUNT interpolated frames"
            elif rclone lsf "$FRAMES_OUTPUT_PATH" 2>/dev/null | head -1 | grep -q .; then
                echo "No interpolated frames found, downloading enhanced frames..."
                rclone copy "$FRAMES_OUTPUT_PATH" "$WORK_DIR/enhanced" --progress --transfers 8
                FRAME_COUNT=$(find "$WORK_DIR/enhanced" -name "*.png" | wc -l)
                echo "Downloaded $FRAME_COUNT enhanced frames"
            else
                echo "ERROR: No frames found for encoding"
                exit 1
            fi
            ;;
        extract|*)
            # Fresh start - download input video
            echo "=== Downloading input from Google Drive ==="
            rclone copyto "$INPUT_PATH" /workspace/input.mp4 --progress
            ;;
    esac
}

# Download based on resume stage
download_existing_frames "$RESUME_FROM"

# =============================================================================
# Run restoration based on resume stage
# Uses individual CLI commands for resume scenarios
# =============================================================================
run_restoration() {
    case $RESUME_FROM in
        extract|dedupe)
            # Fresh start or re-extraction - run full pipeline
            echo "Running full restoration pipeline..."
            eval "$RESTORE_CMD"
            ;;
        enhance)
            # Resume from enhancement - use downloaded unique frames
            echo "Resuming from ENHANCE stage..."
            FRAME_COUNT=$(find "$WORK_DIR/unique_frames" -name "*.png" | wc -l)
            echo "Found $FRAME_COUNT unique frames to enhance"

            # Enhance frames
            framewright enhance --input "$WORK_DIR/unique_frames" --output "$WORK_DIR/enhanced" --scale 4 --model realesrgan-x4plus

            # Check if interpolation was requested in original command
            if echo "$RESTORE_CMD" | grep -q -- "--enable-rife"; then
                TARGET_FPS=$(echo "$RESTORE_CMD" | grep -oP '(?<=--target-fps )[0-9.]+' || echo "24")
                echo "Interpolating to ${TARGET_FPS} fps..."
                framewright interpolate --input "$WORK_DIR/enhanced" --output "$WORK_DIR/interpolated" --target-fps "$TARGET_FPS" --frames-only
                FRAMES_FOR_ENCODE="$WORK_DIR/interpolated"
            else
                FRAMES_FOR_ENCODE="$WORK_DIR/enhanced"
            fi

            # Get FPS from original video or use target
            SOURCE_FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate /workspace/input.mp4 2>/dev/null | bc -l || echo "24")
            FINAL_FPS=${TARGET_FPS:-$SOURCE_FPS}

            # Reassemble
            framewright reassemble --frames-dir "$FRAMES_FOR_ENCODE" --audio /workspace/input.mp4 --output /workspace/project/restored_video.mkv --fps "$FINAL_FPS"
            ;;
        interpolate)
            # Resume from interpolation - use downloaded enhanced frames
            echo "Resuming from INTERPOLATE stage..."
            FRAME_COUNT=$(find "$WORK_DIR/enhanced" -name "*.png" | wc -l)
            echo "Found $FRAME_COUNT enhanced frames to interpolate"

            TARGET_FPS=$(echo "$RESTORE_CMD" | grep -oP '(?<=--target-fps )[0-9.]+' || echo "24")
            echo "Interpolating to ${TARGET_FPS} fps..."
            framewright interpolate --input "$WORK_DIR/enhanced" --output "$WORK_DIR/interpolated" --target-fps "$TARGET_FPS" --frames-only

            # Reassemble
            framewright reassemble --frames-dir "$WORK_DIR/interpolated" --audio /workspace/input.mp4 --output /workspace/project/restored_video.mkv --fps "$TARGET_FPS"
            ;;
        encode)
            # Resume from encoding - just reassemble downloaded frames
            echo "Resuming from ENCODE stage..."

            if [ -d "$WORK_DIR/interpolated" ] && [ "$(find "$WORK_DIR/interpolated" -name "*.png" | wc -l)" -gt 0 ]; then
                FRAMES_FOR_ENCODE="$WORK_DIR/interpolated"
                FRAME_COUNT=$(find "$FRAMES_FOR_ENCODE" -name "*.png" | wc -l)
                echo "Found $FRAME_COUNT interpolated frames"
                TARGET_FPS=$(echo "$RESTORE_CMD" | grep -oP '(?<=--target-fps )[0-9.]+' || echo "24")
            else
                FRAMES_FOR_ENCODE="$WORK_DIR/enhanced"
                FRAME_COUNT=$(find "$FRAMES_FOR_ENCODE" -name "*.png" | wc -l)
                echo "Found $FRAME_COUNT enhanced frames"
                TARGET_FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate /workspace/input.mp4 2>/dev/null | bc -l || echo "24")
            fi

            # Reassemble
            framewright reassemble --frames-dir "$FRAMES_FOR_ENCODE" --audio /workspace/input.mp4 --output /workspace/project/restored_video.mkv --fps "$TARGET_FPS"
            ;;
        *)
            echo "Unknown resume stage: $RESUME_FROM"
            echo "Valid stages: extract, dedupe, enhance, interpolate, encode"
            exit 1
            ;;
    esac
}

# =============================================================================
# Background Sync Function - uploads frames periodically during processing
# =============================================================================
background_sync() {
    echo "[SYNC] Starting background sync (interval: ${SYNC_INTERVAL}s)"

    LAST_UNIQUE=0
    LAST_ENHANCED=0
    LAST_INTERPOLATED=0
    SYNC_COUNT=0

    while true; do
        sleep "$SYNC_INTERVAL"
        SYNC_COUNT=$((SYNC_COUNT + 1))

        echo ""
        echo "[SYNC #$SYNC_COUNT] $(date '+%Y-%m-%d %H:%M:%S') - Checking for new frames..."

        # Sync unique/deduplicated frames
        for dir in "$WORK_DIR/unique_frames" "/workspace/project/unique_frames"; do
            if [ -d "$dir" ]; then
                UNIQUE_COUNT=$(find "$dir" -name "*.png" 2>/dev/null | wc -l)
                if [ "$UNIQUE_COUNT" -gt "$LAST_UNIQUE" ]; then
                    NEW_UNIQUE=$((UNIQUE_COUNT - LAST_UNIQUE))
                    echo "[SYNC] Uploading $NEW_UNIQUE new unique frames (total: $UNIQUE_COUNT)..."
                    rclone copy "$dir" "$UNIQUE_FRAMES_PATH" --quiet --transfers 8
                    LAST_UNIQUE=$UNIQUE_COUNT
                    echo "[SYNC] Unique frames synced: $UNIQUE_FRAMES_PATH"
                fi
                break
            fi
        done

        # Sync enhanced frames
        for dir in "$WORK_DIR/enhanced" "/workspace/project/enhanced"; do
            if [ -d "$dir" ]; then
                ENHANCED_COUNT=$(find "$dir" -name "*.png" 2>/dev/null | wc -l)
                if [ "$ENHANCED_COUNT" -gt "$LAST_ENHANCED" ]; then
                    NEW_ENHANCED=$((ENHANCED_COUNT - LAST_ENHANCED))
                    echo "[SYNC] Uploading $NEW_ENHANCED new enhanced frames (total: $ENHANCED_COUNT)..."
                    rclone copy "$dir" "$FRAMES_OUTPUT_PATH" --quiet --transfers 8
                    LAST_ENHANCED=$ENHANCED_COUNT
                    echo "[SYNC] Enhanced frames synced: $FRAMES_OUTPUT_PATH"
                fi
                break
            fi
        done

        # Sync interpolated frames
        for dir in "$WORK_DIR/interpolated" "/workspace/project/interpolated"; do
            if [ -d "$dir" ]; then
                INTERP_COUNT=$(find "$dir" -name "*.png" 2>/dev/null | wc -l)
                if [ "$INTERP_COUNT" -gt "$LAST_INTERPOLATED" ]; then
                    NEW_INTERP=$((INTERP_COUNT - LAST_INTERPOLATED))
                    echo "[SYNC] Uploading $NEW_INTERP new interpolated frames (total: $INTERP_COUNT)..."
                    rclone copy "$dir" "$INTERP_PATH" --quiet --transfers 8
                    LAST_INTERPOLATED=$INTERP_COUNT
                    echo "[SYNC] Interpolated frames synced: $INTERP_PATH"
                fi
                break
            fi
        done

        echo "[SYNC #$SYNC_COUNT] Complete. Unique: $LAST_UNIQUE, Enhanced: $LAST_ENHANCED, Interpolated: $LAST_INTERPOLATED"
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
echo "Resume stage: $RESUME_FROM"
echo "Original command: $RESTORE_CMD"
run_restoration
RESTORE_EXIT_CODE=$?

# Stop background sync
echo "=== Stopping background sync ==="
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

echo "=== Final upload of all results to Google Drive ==="

# 1. Upload deduplicated/unique frames
for dir in "$WORK_DIR/unique_frames" "/workspace/project/unique_frames"; do
    if [ -d "$dir" ] && [ "$(find "$dir" -name "*.png" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Final sync: unique frames..."
        rclone copy "$dir" "$UNIQUE_FRAMES_PATH" --progress --transfers 8
        echo "Deduplicated frames: $UNIQUE_FRAMES_PATH"
        break
    fi
done

# 2. Upload enhanced frames
for dir in "$WORK_DIR/enhanced" "/workspace/project/enhanced"; do
    if [ -d "$dir" ] && [ "$(find "$dir" -name "*.png" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Final sync: enhanced frames..."
        rclone copy "$dir" "$FRAMES_OUTPUT_PATH" --progress --transfers 8
        echo "Enhanced frames: $FRAMES_OUTPUT_PATH"
        break
    fi
done

# 3. Upload interpolated frames
for dir in "$WORK_DIR/interpolated" "/workspace/project/interpolated"; do
    if [ -d "$dir" ] && [ "$(find "$dir" -name "*.png" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Final sync: interpolated frames..."
        rclone copy "$dir" "$INTERP_PATH" --progress --transfers 8
        echo "Interpolated frames: $INTERP_PATH"
        break
    fi
done

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
echo "Interpolated frames: $INTERP_PATH"
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
    echo ""
    echo "To resume from a specific stage, set RESUME_FROM environment variable:"
    echo "  RESUME_FROM=enhance      Resume from enhancement (uses uploaded unique frames)"
    echo "  RESUME_FROM=interpolate  Resume from interpolation (uses uploaded enhanced frames)"
    echo "  RESUME_FROM=encode       Resume from encoding (uses uploaded frames)"
fi
