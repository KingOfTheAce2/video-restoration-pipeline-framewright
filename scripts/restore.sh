#!/bin/bash

#############################################################################
# Video Restoration Pipeline Script
# Handles downloading, frame extraction, enhancement, and reassembly
#############################################################################

set -euo pipefail

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default configuration
SCALE_FACTOR="${SCALE_FACTOR:-2}"
CRF="${CRF:-18}"
PRESET="${PRESET:-slow}"
WORK_DIR="${WORK_DIR:-./restoration_work}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-false}"

# Trap for cleanup on error
trap cleanup ERR INT TERM

#############################################################################
# Functions
#############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Script failed with exit code $exit_code"
    fi

    if [ "${KEEP_INTERMEDIATE}" = "false" ] && [ -d "${WORK_DIR}" ]; then
        log_info "Cleaning up intermediate files..."
        rm -rf "${WORK_DIR}/frames" "${WORK_DIR}/enhanced" 2>/dev/null || true
        log_success "Cleanup completed"
    fi
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [VIDEO_URL]

Video restoration pipeline using Real-ESRGAN upscaling.

OPTIONS:
    -s, --scale FACTOR      Scale factor for upscaling (default: 2)
    -c, --crf VALUE         CRF value for output encoding (default: 18)
    -p, --preset PRESET     FFmpeg preset (default: slow)
    -w, --workdir PATH      Working directory (default: ./restoration_work)
    -k, --keep              Keep intermediate files
    -h, --help              Show this help message

ARGUMENTS:
    VIDEO_URL               URL of video to restore (optional, will prompt if not provided)

EXAMPLES:
    $0 https://youtube.com/watch?v=example
    $0 -s 4 -c 16 -p medium https://youtube.com/watch?v=example
    $0 --keep --scale 2 https://youtube.com/watch?v=example

ENVIRONMENT VARIABLES:
    SCALE_FACTOR            Default scale factor
    CRF                     Default CRF value
    PRESET                  Default FFmpeg preset
    WORK_DIR                Default working directory
    KEEP_INTERMEDIATE       Keep intermediate files (true/false)

EOF
    exit 0
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--scale)
                SCALE_FACTOR="$2"
                shift 2
                ;;
            -c|--crf)
                CRF="$2"
                shift 2
                ;;
            -p|--preset)
                PRESET="$2"
                shift 2
                ;;
            -w|--workdir)
                WORK_DIR="$2"
                shift 2
                ;;
            -k|--keep)
                KEEP_INTERMEDIATE="true"
                shift
                ;;
            -h|--help)
                show_usage
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                ;;
            *)
                VIDEO_URL="$1"
                shift
                ;;
        esac
    done
}

check_dependencies() {
    log_info "Checking dependencies..."

    local deps=("yt-dlp" "ffmpeg" "ffprobe" "realesrgan-ncnn-vulkan")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Please install missing dependencies and try again"
        exit 1
    fi

    log_success "All dependencies found"
}

create_directory_structure() {
    log_info "Creating directory structure in ${WORK_DIR}..."

    mkdir -p "${WORK_DIR}"/{original,frames,enhanced,output}

    log_success "Directory structure created"
}

download_video() {
    local url="$1"
    local output_path="${WORK_DIR}/original/video.webm"

    log_info "Downloading video from ${url}..."
    log_info "Format: bestvideo[ext=webm]+bestaudio[ext=webm]"

    yt-dlp \
        -f 'bestvideo[ext=webm]+bestaudio[ext=webm]/best[ext=webm]/best' \
        --merge-output-format webm \
        -o "$output_path" \
        "$url"

    if [ ! -f "$output_path" ]; then
        log_error "Failed to download video"
        exit 1
    fi

    log_success "Video downloaded to ${output_path}"
    echo "$output_path"
}

extract_framerate() {
    local video_path="$1"

    log_info "Extracting framerate..."

    local fps=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate \
        -of default=noprint_wrappers=1:nokey=1 \
        "$video_path")

    # Convert fraction to decimal if needed
    if [[ $fps == *"/"* ]]; then
        fps=$(echo "$fps" | awk -F'/' '{printf "%.3f", $1/$2}')
    fi

    log_success "Framerate: ${fps} fps"
    echo "$fps"
}

extract_audio() {
    local video_path="$1"
    local audio_path="${WORK_DIR}/original/audio.wav"

    log_info "Extracting audio as WAV (pcm_s24le)..."

    ffmpeg -i "$video_path" \
        -vn \
        -acodec pcm_s24le \
        -ar 48000 \
        -y \
        "$audio_path" \
        2>&1 | grep -E '(Duration|time=)' || true

    if [ ! -f "$audio_path" ]; then
        log_error "Failed to extract audio"
        exit 1
    fi

    log_success "Audio extracted to ${audio_path}"
    echo "$audio_path"
}

extract_frames() {
    local video_path="$1"
    local frames_dir="${WORK_DIR}/frames"

    log_info "Extracting frames as high-quality PNG..."

    ffmpeg -i "$video_path" \
        -qscale:v 1 \
        -qmin 1 \
        -qmax 1 \
        -vsync 0 \
        "${frames_dir}/frame_%08d.png" \
        2>&1 | grep -E '(Duration|time=|frame=)' || true

    local frame_count=$(find "$frames_dir" -name "frame_*.png" | wc -l)

    if [ "$frame_count" -eq 0 ]; then
        log_error "Failed to extract frames"
        exit 1
    fi

    log_success "Extracted ${frame_count} frames"
    echo "$frame_count"
}

enhance_frames() {
    local scale="$1"
    local frames_dir="${WORK_DIR}/frames"
    local enhanced_dir="${WORK_DIR}/enhanced"

    log_info "Enhancing frames with Real-ESRGAN (scale: ${scale}x)..."
    log_warning "This may take a while depending on the number of frames..."

    # Check if realesrgan-ncnn-vulkan supports the scale factor
    local model_name
    case $scale in
        2)
            model_name="realesrgan-x4plus"
            ;;
        4)
            model_name="realesrgan-x4plus"
            ;;
        *)
            log_error "Unsupported scale factor: ${scale}. Use 2 or 4."
            exit 1
            ;;
    esac

    realesrgan-ncnn-vulkan \
        -i "$frames_dir" \
        -o "$enhanced_dir" \
        -n "$model_name" \
        -s "$scale" \
        -f png

    local enhanced_count=$(find "$enhanced_dir" -name "*.png" | wc -l)

    if [ "$enhanced_count" -eq 0 ]; then
        log_error "Failed to enhance frames"
        exit 1
    fi

    log_success "Enhanced ${enhanced_count} frames"
}

reassemble_video() {
    local fps="$1"
    local audio_path="$2"
    local enhanced_dir="${WORK_DIR}/enhanced"
    local output_path="${WORK_DIR}/output/restored_video.mkv"

    log_info "Reassembling video..."
    log_info "Codec: libx265, CRF: ${CRF}, Preset: ${PRESET}, Audio: FLAC"

    # First pass: create video from frames
    local temp_video="${WORK_DIR}/output/temp_video.mkv"

    ffmpeg -framerate "$fps" \
        -pattern_type glob \
        -i "${enhanced_dir}/*.png" \
        -c:v libx265 \
        -crf "$CRF" \
        -preset "$PRESET" \
        -pix_fmt yuv420p \
        -y \
        "$temp_video" \
        2>&1 | grep -E '(Duration|time=|frame=)' || true

    # Second pass: merge video with audio
    if [ -f "$audio_path" ]; then
        log_info "Merging enhanced audio..."
        ffmpeg -i "$temp_video" \
            -i "$audio_path" \
            -c:v copy \
            -c:a flac \
            -compression_level 12 \
            -y \
            "$output_path" \
            2>&1 | grep -E '(Duration|time=)' || true

        rm -f "$temp_video"
    else
        mv "$temp_video" "$output_path"
    fi

    if [ ! -f "$output_path" ]; then
        log_error "Failed to reassemble video"
        exit 1
    fi

    log_success "Video restored and saved to ${output_path}"

    # Show file size
    local file_size=$(du -h "$output_path" | cut -f1)
    log_info "Output file size: ${file_size}"

    echo "$output_path"
}

#############################################################################
# Main Script
#############################################################################

main() {
    log_info "Video Restoration Pipeline Starting..."
    log_info "Configuration: Scale=${SCALE_FACTOR}x, CRF=${CRF}, Preset=${PRESET}"

    # Parse arguments
    parse_arguments "$@"

    # Prompt for URL if not provided
    if [ -z "${VIDEO_URL:-}" ]; then
        read -p "Enter video URL: " VIDEO_URL
        if [ -z "$VIDEO_URL" ]; then
            log_error "No video URL provided"
            exit 1
        fi
    fi

    # Check dependencies
    check_dependencies

    # Create directory structure
    create_directory_structure

    # Download video
    local video_path=$(download_video "$VIDEO_URL")

    # Extract framerate
    local fps=$(extract_framerate "$video_path")

    # Extract audio
    local audio_path=$(extract_audio "$video_path")

    # Extract frames
    local frame_count=$(extract_frames "$video_path")

    # Enhance frames
    enhance_frames "$SCALE_FACTOR"

    # Reassemble video
    local output_path=$(reassemble_video "$fps" "$audio_path")

    # Final cleanup
    cleanup

    log_success "Restoration pipeline completed successfully!"
    log_info "Restored video: ${output_path}"
}

# Run main function
main "$@"
