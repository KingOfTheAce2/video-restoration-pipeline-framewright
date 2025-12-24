#!/bin/bash

#############################################################################
# Audio Enhancement Script
# Applies FFmpeg filters: highpass, lowpass, noise reduction, normalization
#############################################################################

set -euo pipefail

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default configuration
HIGHPASS_FREQ="${HIGHPASS_FREQ:-80}"
LOWPASS_FREQ="${LOWPASS_FREQ:-12000}"
NOISE_REDUCTION="${NOISE_REDUCTION:-0.02}"
TARGET_LOUDNESS="${TARGET_LOUDNESS:--16}"
SAMPLE_RATE="${SAMPLE_RATE:-48000}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-wav}"
OUTPUT_CODEC="${OUTPUT_CODEC:-pcm_s24le}"

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

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] INPUT_FILE [OUTPUT_FILE]

Audio enhancement pipeline using FFmpeg filters.

OPTIONS:
    -hp, --highpass FREQ    Highpass filter frequency in Hz (default: 80)
    -lp, --lowpass FREQ     Lowpass filter frequency in Hz (default: 12000)
    -nr, --noise-reduction  Noise reduction amount 0.0-1.0 (default: 0.02)
    -l, --loudness DB       Target loudness in LUFS (default: -16)
    -r, --rate RATE         Sample rate in Hz (default: 48000)
    -f, --format FORMAT     Output format: wav, flac, mp3 (default: wav)
    -c, --codec CODEC       Output codec (default: pcm_s24le for wav)
    -h, --help              Show this help message

ARGUMENTS:
    INPUT_FILE              Input audio file (required)
    OUTPUT_FILE             Output audio file (optional, default: INPUT_enhanced.EXT)

EXAMPLES:
    $0 audio.wav
    $0 audio.wav enhanced.wav
    $0 -hp 100 -lp 10000 -nr 0.05 audio.wav
    $0 --format flac --loudness -14 audio.wav audio_enhanced.flac

FILTER CHAIN:
    1. Highpass filter (removes low rumble)
    2. Lowpass filter (removes high frequency noise)
    3. FFT-based noise reduction (afftdn)
    4. Loudness normalization (loudnorm)

ENVIRONMENT VARIABLES:
    HIGHPASS_FREQ           Default highpass frequency
    LOWPASS_FREQ            Default lowpass frequency
    NOISE_REDUCTION         Default noise reduction amount
    TARGET_LOUDNESS         Default target loudness
    SAMPLE_RATE             Default sample rate
    OUTPUT_FORMAT           Default output format
    OUTPUT_CODEC            Default output codec

EOF
    exit 0
}

parse_arguments() {
    local input_set=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -hp|--highpass)
                HIGHPASS_FREQ="$2"
                shift 2
                ;;
            -lp|--lowpass)
                LOWPASS_FREQ="$2"
                shift 2
                ;;
            -nr|--noise-reduction)
                NOISE_REDUCTION="$2"
                shift 2
                ;;
            -l|--loudness)
                TARGET_LOUDNESS="$2"
                shift 2
                ;;
            -r|--rate)
                SAMPLE_RATE="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -c|--codec)
                OUTPUT_CODEC="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                ;;
            *)
                if [ "$input_set" = false ]; then
                    INPUT_FILE="$1"
                    input_set=true
                else
                    OUTPUT_FILE="$1"
                fi
                shift
                ;;
        esac
    done

    if [ -z "${INPUT_FILE:-}" ]; then
        log_error "No input file specified"
        show_usage
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v ffmpeg &> /dev/null; then
        log_error "ffmpeg is not installed"
        exit 1
    fi

    log_success "All dependencies found"
}

validate_input() {
    local input="$1"

    log_info "Validating input file..."

    if [ ! -f "$input" ]; then
        log_error "Input file does not exist: ${input}"
        exit 1
    fi

    # Check if file is a valid audio file
    if ! ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$input" &> /dev/null; then
        log_error "Input file is not a valid audio file: ${input}"
        exit 1
    fi

    log_success "Input file validated"
}

generate_output_filename() {
    local input="$1"
    local format="$2"

    # Extract directory, filename, and extension
    local dir=$(dirname "$input")
    local filename=$(basename "$input")
    local name="${filename%.*}"

    # Generate output filename
    echo "${dir}/${name}_enhanced.${format}"
}

set_codec_for_format() {
    local format="$1"

    case $format in
        wav)
            OUTPUT_CODEC="pcm_s24le"
            ;;
        flac)
            OUTPUT_CODEC="flac"
            ;;
        mp3)
            OUTPUT_CODEC="libmp3lame"
            ;;
        aac)
            OUTPUT_CODEC="aac"
            ;;
        opus)
            OUTPUT_CODEC="libopus"
            ;;
        *)
            log_warning "Unknown format: ${format}, using default codec"
            ;;
    esac
}

build_filter_chain() {
    local hp_freq="$1"
    local lp_freq="$2"
    local nr_amount="$3"
    local target_loud="$4"

    # Build comprehensive filter chain
    local filters=""

    # 1. Highpass filter (remove low rumble)
    filters+="highpass=f=${hp_freq},"

    # 2. Lowpass filter (remove high frequency noise)
    filters+="lowpass=f=${lp_freq},"

    # 3. FFT-based noise reduction
    filters+="afftdn=nr=${nr_amount}:nf=-25:tn=1,"

    # 4. Loudness normalization (EBU R128)
    filters+="loudnorm=I=${target_loud}:TP=-1.5:LRA=11:print_format=summary"

    echo "$filters"
}

enhance_audio() {
    local input="$1"
    local output="$2"

    log_info "Enhancing audio..."
    log_info "Input: ${input}"
    log_info "Output: ${output}"
    log_info "Filters: Highpass ${HIGHPASS_FREQ}Hz, Lowpass ${LOWPASS_FREQ}Hz, Noise Reduction ${NOISE_REDUCTION}, Loudness ${TARGET_LOUDNESS}LUFS"

    # Build filter chain
    local filter_chain=$(build_filter_chain "$HIGHPASS_FREQ" "$LOWPASS_FREQ" "$NOISE_REDUCTION" "$TARGET_LOUDNESS")

    # Run FFmpeg with filter chain
    ffmpeg -i "$input" \
        -af "$filter_chain" \
        -ar "$SAMPLE_RATE" \
        -c:a "$OUTPUT_CODEC" \
        -y \
        "$output" \
        2>&1 | tee /tmp/ffmpeg_audio_enhance.log | grep -E '(Duration|time=|Parsed_loudnorm)' || true

    if [ ! -f "$output" ]; then
        log_error "Failed to enhance audio"
        exit 1
    fi

    log_success "Audio enhancement completed"

    # Show file sizes
    local input_size=$(du -h "$input" | cut -f1)
    local output_size=$(du -h "$output" | cut -f1)
    log_info "Input size: ${input_size}, Output size: ${output_size}"

    # Extract loudness info from log if available
    if grep -q "Parsed_loudnorm" /tmp/ffmpeg_audio_enhance.log 2>/dev/null; then
        log_info "Loudness normalization details:"
        grep "Parsed_loudnorm" /tmp/ffmpeg_audio_enhance.log | tail -10
    fi
}

show_audio_info() {
    local file="$1"

    log_info "Audio file information:"

    ffprobe -v error \
        -show_entries format=duration,bit_rate \
        -show_entries stream=codec_name,sample_rate,channels \
        -of default=noprint_wrappers=1 \
        "$file"
}

#############################################################################
# Main Script
#############################################################################

main() {
    log_info "Audio Enhancement Pipeline Starting..."

    # Parse arguments
    parse_arguments "$@"

    # Check dependencies
    check_dependencies

    # Validate input
    validate_input "$INPUT_FILE"

    # Set codec for format if not manually specified
    if [ -z "${OUTPUT_CODEC:-}" ] || [ "${OUTPUT_CODEC}" = "pcm_s24le" ]; then
        set_codec_for_format "$OUTPUT_FORMAT"
    fi

    # Generate output filename if not provided
    if [ -z "${OUTPUT_FILE:-}" ]; then
        OUTPUT_FILE=$(generate_output_filename "$INPUT_FILE" "$OUTPUT_FORMAT")
    fi

    # Show input info
    show_audio_info "$INPUT_FILE"

    # Enhance audio
    enhance_audio "$INPUT_FILE" "$OUTPUT_FILE"

    # Show output info
    show_audio_info "$OUTPUT_FILE"

    log_success "Audio enhancement pipeline completed successfully!"
    log_info "Enhanced audio: ${OUTPUT_FILE}"
}

# Run main function
main "$@"
