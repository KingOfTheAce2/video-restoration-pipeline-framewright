"""Web-based UI for FrameWright video restoration pipeline.

Provides a user-friendly Gradio interface for non-technical users.
"""
import logging
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Presets Configuration
# =============================================================================

class QueueItemStatus(Enum):
    """Status for batch queue items."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class QueueItem:
    """Represents an item in the batch processing queue."""
    id: str
    source: str  # File path or YouTube URL
    source_type: str  # "file" or "youtube"
    status: QueueItemStatus = QueueItemStatus.PENDING
    progress: float = 0.0
    message: str = ""
    output_path: Optional[str] = None
    priority: int = 0  # Lower number = higher priority

    def to_display(self) -> List[str]:
        """Convert to display format for Gradio dataframe."""
        status_icons = {
            QueueItemStatus.PENDING: "⏳ Pending",
            QueueItemStatus.PROCESSING: "🔄 Processing",
            QueueItemStatus.COMPLETE: "✅ Complete",
            QueueItemStatus.ERROR: "❌ Error",
        }
        source_display = self.source if len(self.source) < 50 else f"...{self.source[-47:]}"
        return [
            self.id[:8],
            source_display,
            self.source_type.capitalize(),
            status_icons.get(self.status, "❓"),
            f"{self.progress:.0%}",
            self.message or "-",
        ]


# Quality preset definitions
QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "Quick Preview": {
        "scale_factor": 2,
        "crf": 23,
        "model": "realesrgan-x4plus",
        "enhance_audio": True,
        "interpolate": False,
        "enable_colorization": False,
        "enable_watermark_removal": False,
        "description": "Fast processing with lower quality. Good for previewing results.",
    },
    "Balanced": {
        "scale_factor": 4,
        "crf": 20,
        "model": "realesrgan-x4plus",
        "enhance_audio": True,
        "interpolate": False,
        "enable_colorization": False,
        "enable_watermark_removal": False,
        "description": "Recommended default. Good balance of quality and speed.",
    },
    "Maximum Quality": {
        "scale_factor": 4,
        "crf": 15,
        "model": "realesrgan-x4plus",
        "enhance_audio": True,
        "interpolate": True,
        "target_fps": 60,
        "enable_colorization": False,
        "enable_watermark_removal": False,
        "description": "Highest quality output. Best for archival purposes. Slower processing.",
    },
    "Custom": {
        "scale_factor": None,  # Keep current value
        "crf": None,
        "model": None,
        "enhance_audio": None,
        "interpolate": None,
        "enable_colorization": None,
        "enable_watermark_removal": None,
        "description": "Manual settings. Adjust all parameters yourself.",
    },
}

# Check for Gradio availability
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    gr = None


def check_gradio_installed() -> bool:
    """Check if Gradio is installed."""
    return HAS_GRADIO


def install_gradio_instructions() -> str:
    """Get instructions for installing Gradio."""
    return """
Gradio is required for the web UI but is not installed.

Install it with:
    pip install gradio

Or install FrameWright with UI support:
    pip install framewright[ui]
"""


# =============================================================================
# UI Components (only defined if Gradio is available)
# =============================================================================

if HAS_GRADIO:
    from .config import Config
    from .restorer import VideoRestorer
    from .hardware import check_hardware, print_hardware_report, HardwareReport

    # Global batch queue state
    _batch_queue: List[QueueItem] = []
    _batch_processing = False
    _batch_lock = threading.Lock()

    def get_queue_dataframe() -> List[List[str]]:
        """Get queue as dataframe for display."""
        with _batch_lock:
            if not _batch_queue:
                return []
            return [item.to_display() for item in sorted(_batch_queue, key=lambda x: x.priority)]

    def add_to_queue(
        source: str,
        source_type: str,
    ) -> Tuple[List[List[str]], str]:
        """Add an item to the batch queue."""
        if not source or not source.strip():
            return get_queue_dataframe(), "Please provide a video file or URL"

        with _batch_lock:
            item = QueueItem(
                id=str(uuid.uuid4()),
                source=source.strip(),
                source_type=source_type,
                priority=len(_batch_queue),
            )
            _batch_queue.append(item)

        return get_queue_dataframe(), f"Added to queue: {source[:50]}..."

    def remove_from_queue(selected_id: str) -> Tuple[List[List[str]], str]:
        """Remove an item from the queue by ID."""
        if not selected_id:
            return get_queue_dataframe(), "Select an item to remove"

        with _batch_lock:
            for i, item in enumerate(_batch_queue):
                if item.id.startswith(selected_id):
                    removed = _batch_queue.pop(i)
                    return get_queue_dataframe(), f"Removed: {removed.source[:30]}..."

        return get_queue_dataframe(), "Item not found"

    def move_queue_item(selected_id: str, direction: str) -> List[List[str]]:
        """Move an item up or down in the queue."""
        if not selected_id:
            return get_queue_dataframe()

        with _batch_lock:
            # Find item index
            idx = None
            for i, item in enumerate(_batch_queue):
                if item.id.startswith(selected_id):
                    idx = i
                    break

            if idx is None:
                return get_queue_dataframe()

            # Calculate new position
            if direction == "up" and idx > 0:
                _batch_queue[idx], _batch_queue[idx - 1] = _batch_queue[idx - 1], _batch_queue[idx]
                # Update priorities
                _batch_queue[idx].priority = idx
                _batch_queue[idx - 1].priority = idx - 1
            elif direction == "down" and idx < len(_batch_queue) - 1:
                _batch_queue[idx], _batch_queue[idx + 1] = _batch_queue[idx + 1], _batch_queue[idx]
                # Update priorities
                _batch_queue[idx].priority = idx
                _batch_queue[idx + 1].priority = idx + 1

        return get_queue_dataframe()

    def clear_queue() -> Tuple[List[List[str]], str]:
        """Clear all items from the queue."""
        with _batch_lock:
            _batch_queue.clear()
        return [], "Queue cleared"

    def create_ui() -> gr.Blocks:
        """Create the Gradio web interface.

        Returns:
            Gradio Blocks application
        """
        # State for tracking progress
        progress_state = {"current": 0, "total": 100, "stage": "idle"}

        def run_hardware_check() -> str:
            """Run hardware compatibility check."""
            try:
                report = check_hardware()
                return print_hardware_report(report)
            except Exception as e:
                return f"Error checking hardware: {e}"

        def get_hardware_summary() -> Tuple[str, str, str, str]:
            """Get hardware summary for display."""
            try:
                report = check_hardware()

                # Status badge
                status_map = {
                    "ready": "🟢 Ready",
                    "limited": "🟡 Limited",
                    "incompatible": "🔴 Incompatible",
                }
                status = status_map.get(report.overall_status, "❓ Unknown")

                # GPU info
                if report.gpu.has_gpu:
                    gpu = f"{report.gpu.gpu_name} ({report.gpu.vram_total_mb}MB VRAM)"
                else:
                    gpu = "No GPU detected (CPU mode)"

                # RAM
                ram = f"{report.system.ram_available_gb:.1f} GB available"

                # Disk
                disk = f"{report.disk_free_gb:.1f} GB free"

                return status, gpu, ram, disk
            except Exception as e:
                return "❓ Error", str(e), "Unknown", "Unknown"

        def analyze_video_for_settings(
            input_video: str,
            youtube_url: str,
        ) -> Tuple[str, int, str, str]:
            """Analyze video and return recommended settings.

            Returns:
                Tuple of (analysis_report, recommended_scale, recommended_model, content_type)
            """
            try:
                from .processors.analyzer import FrameAnalyzer, ContentType

                # Determine source
                if youtube_url and youtube_url.strip():
                    return ("⚠️ YouTube analysis requires download first.\n"
                            "Using default settings for real footage.",
                            4, "realesrgan-x4plus", "Unknown (YouTube)")
                elif not input_video:
                    return "❌ Please upload a video first.", 4, "realesrgan-x4plus", "None"

                analyzer = FrameAnalyzer(
                    sample_rate=50,
                    max_samples=20,
                    enable_face_detection=True,
                )

                analysis = analyzer.analyze_video(Path(input_video))

                # Build report
                content_name = analysis.primary_content.name.replace('_', ' ').title()

                # Model recommendation with explanation
                model_explanations = {
                    "realesrgan-x4plus": "Best for real footage (movies, photos, documentaries)",
                    "realesrgan-x4plus-anime": "Optimized for anime and animation",
                    "realesr-animevideov3": "Best quality for anime video",
                }
                model_explanation = model_explanations.get(
                    analysis.recommended_model,
                    "General purpose enhancement"
                )

                report = f"""## 🔍 Video Analysis Complete

**Content Detected:** {content_name}
**Resolution:** {analysis.resolution[0]}x{analysis.resolution[1]}
**Frame Rate:** {analysis.source_fps:.1f} fps
**Degradation:** {analysis.degradation_severity.title()}

### 🎯 Recommended Settings

| Setting | Recommended | Reason |
|---------|-------------|--------|
| **Scale** | {analysis.recommended_scale}x | {'Low resolution source' if analysis.recommended_scale == 4 else 'Good resolution source'} |
| **Model** | `{analysis.recommended_model}` | {model_explanation} |
| **Face Restore** | {'Yes' if analysis.enable_face_restoration else 'No'} | {'Faces detected in {:.0%} of frames'.format(analysis.face_frame_ratio) if analysis.face_frame_ratio > 0 else 'No faces detected'} |

*Click "Apply Recommendations" to use these settings.*
"""

                return (report,
                        analysis.recommended_scale,
                        analysis.recommended_model,
                        content_name)

            except Exception as e:
                return f"❌ Analysis failed: {e}", 4, "realesrgan-x4plus", "Error"

        def on_preset_change(preset_name: str) -> Tuple[Any, ...]:
            """Handle preset selection and update UI fields.

            Returns tuple of updated values for:
            (scale_factor, crf, model, enhance_audio, interpolate, target_fps_visible,
             enable_colorization, enable_watermark_removal, preset_description)
            """
            if preset_name == "Custom" or preset_name not in QUALITY_PRESETS:
                # Return gr.update() for each field to keep current values
                return (
                    gr.update(),  # scale_factor
                    gr.update(),  # crf
                    gr.update(),  # model
                    gr.update(),  # enhance_audio
                    gr.update(),  # interpolate
                    gr.update(),  # target_fps visibility
                    gr.update(),  # enable_colorization
                    gr.update(),  # enable_watermark_removal
                    "*Manual settings mode - adjust parameters as needed.*",  # description
                )

            preset = QUALITY_PRESETS[preset_name]
            interpolate_val = preset.get("interpolate", False)

            return (
                preset["scale_factor"],  # scale_factor
                preset["crf"],  # crf
                preset["model"],  # model
                preset["enhance_audio"],  # enhance_audio
                interpolate_val,  # interpolate
                gr.update(visible=interpolate_val),  # target_fps visibility
                preset["enable_colorization"],  # enable_colorization
                preset["enable_watermark_removal"],  # enable_watermark_removal
                f"*{preset['description']}*",  # description
            )

        def process_batch_queue(
            output_folder: str,
            output_format: str,
            preset_name: str,
            scale_factor: int,
            model: str,
            crf: int,
            enhance_audio: bool,
            model_download_dir: str,
            interpolate: bool,
            target_fps: int,
            enable_colorization: bool,
            colorization_model: str,
            enable_watermark_removal: bool,
            watermark_auto_detect: bool,
            progress: gr.Progress = gr.Progress(),
        ) -> Tuple[List[List[str]], str]:
            """Process all items in the batch queue."""
            global _batch_processing

            with _batch_lock:
                if _batch_processing:
                    return get_queue_dataframe(), "Batch processing already in progress"
                if not _batch_queue:
                    return [], "Queue is empty"
                _batch_processing = True

            logs = ["Starting batch processing...\n"]
            total_items = len(_batch_queue)

            try:
                for i, item in enumerate(_batch_queue):
                    with _batch_lock:
                        item.status = QueueItemStatus.PROCESSING
                        item.message = "Processing..."
                        item.progress = 0.0

                    try:
                        progress((i / total_items), desc=f"Processing {i+1}/{total_items}: {item.source[:30]}...")
                        logs.append(f"\n[{i+1}/{total_items}] Processing: {item.source}")

                        # Determine output directory
                        if output_folder and output_folder.strip():
                            output_dir = Path(output_folder.strip())
                            output_dir.mkdir(parents=True, exist_ok=True)
                        else:
                            output_dir = Path(tempfile.mkdtemp(prefix="framewright_batch_"))

                        # Determine model directory
                        model_dir = None
                        if model_download_dir and model_download_dir.strip():
                            model_dir = Path(model_download_dir.strip()).expanduser()

                        # Create config
                        config = Config(
                            project_dir=output_dir,
                            output_dir=output_dir,
                            scale_factor=scale_factor,
                            model_name=model,
                            crf=crf,
                            output_format=output_format if output_format else "mkv",
                            model_download_dir=model_dir,
                            enable_checkpointing=True,
                            enable_validation=True,
                            enable_colorization=enable_colorization,
                            colorization_model=colorization_model,
                            enable_watermark_removal=enable_watermark_removal,
                            watermark_auto_detect=watermark_auto_detect,
                        )

                        # Create restorer and process
                        restorer = VideoRestorer(config)

                        def on_item_progress(current: int, total: int, stage: str):
                            with _batch_lock:
                                item.progress = current / total if total > 0 else 0
                                item.message = stage

                        result = restorer.restore_video(
                            source=item.source,
                            cleanup=False,
                        )

                        with _batch_lock:
                            item.status = QueueItemStatus.COMPLETE
                            item.progress = 1.0
                            item.output_path = str(result)
                            item.message = f"Done: {result.name}"

                        logs.append(f"  -> Complete: {result}")

                    except Exception as e:
                        with _batch_lock:
                            item.status = QueueItemStatus.ERROR
                            item.message = str(e)[:50]
                        logs.append(f"  -> Error: {e}")
                        logger.exception(f"Batch item failed: {item.source}")

                logs.append(f"\n\nBatch processing complete!")
                return get_queue_dataframe(), "\n".join(logs)

            finally:
                with _batch_lock:
                    _batch_processing = False

        def restore_video(
            input_video: str,
            youtube_url: str,
            output_folder: str,
            output_format: str,
            scale_factor: int,
            model: str,
            crf: int,
            enhance_audio: bool,
            force_cpu: bool,
            model_download_dir: str,
            interpolate: bool,
            target_fps: int,
            enable_deduplication: bool,
            deduplication_threshold: float,
            enable_colorization: bool,
            colorization_model: str,
            enable_watermark_removal: bool,
            watermark_auto_detect: bool,
            progress: gr.Progress = gr.Progress(),
        ) -> Tuple[Optional[str], str]:
            """Run video restoration pipeline.

            Returns:
                Tuple of (output_video_path, log_messages)
            """
            import time
            logs = []
            start_time = time.time()

            def log(msg: str):
                elapsed = time.time() - start_time
                timestamp = f"[{elapsed:6.1f}s]"
                logs.append(f"{timestamp} {msg}")
                logger.info(msg)

            def log_section(title: str):
                logs.append(f"\n{'='*50}")
                logs.append(f"  {title}")
                logs.append(f"{'='*50}")

            try:
                log_section("FRAMEWRIGHT VIDEO RESTORATION")

                # Determine input source
                if youtube_url and youtube_url.strip():
                    source = youtube_url.strip()
                    log(f"Source: YouTube URL")
                    log(f"  URL: {source}")
                    video_duration = None
                    video_frames = None
                    video_resolution = None
                    video_fps = None
                elif input_video:
                    source = input_video
                    log(f"Source: Local file")
                    log(f"  Path: {source}")

                    # Analyze video file
                    try:
                        from .utils.ffmpeg import get_video_info
                        video_info = get_video_info(Path(input_video))

                        # Extract video stream info
                        video_stream = None
                        audio_stream = None
                        for stream in video_info.get('streams', []):
                            if stream.get('codec_type') == 'video' and not video_stream:
                                video_stream = stream
                            elif stream.get('codec_type') == 'audio' and not audio_stream:
                                audio_stream = stream

                        if video_stream:
                            # Resolution
                            width = video_stream.get('width', 0)
                            height = video_stream.get('height', 0)
                            video_resolution = (width, height)

                            # FPS
                            fps_str = video_stream.get('r_frame_rate', '0/1')
                            num, den = map(int, fps_str.split('/'))
                            video_fps = num / den if den != 0 else 0.0

                            # Duration and frame count
                            if 'duration' in video_stream:
                                video_duration = float(video_stream['duration'])
                            elif 'duration' in video_info.get('format', {}):
                                video_duration = float(video_info['format']['duration'])
                            else:
                                video_duration = None

                            if 'nb_frames' in video_stream:
                                video_frames = int(video_stream['nb_frames'])
                            elif video_duration and video_fps:
                                video_frames = int(video_duration * video_fps)
                            else:
                                video_frames = None

                            # File size
                            file_size_bytes = int(video_info.get('format', {}).get('size', 0))
                            file_size_mb = file_size_bytes / (1024 * 1024)

                            log_section("VIDEO ANALYSIS")
                            log(f"Resolution: {width}x{height}")
                            log(f"Frame rate: {video_fps:.2f} fps")
                            if video_duration:
                                mins, secs = divmod(int(video_duration), 60)
                                hrs, mins = divmod(mins, 60)
                                if hrs > 0:
                                    log(f"Duration: {hrs}h {mins}m {secs}s")
                                else:
                                    log(f"Duration: {mins}m {secs}s")
                            if video_frames:
                                log(f"Total frames: {video_frames:,}")
                            log(f"File size: {file_size_mb:.1f} MB")
                            log(f"Codec: {video_stream.get('codec_name', 'unknown')}")
                            if audio_stream:
                                log(f"Audio: {audio_stream.get('codec_name', 'unknown')} ({audio_stream.get('channels', '?')}ch)")
                            else:
                                log(f"Audio: None detected")
                        else:
                            video_duration = None
                            video_frames = None
                            video_resolution = None
                            video_fps = None
                    except Exception as e:
                        log(f"Note: Could not analyze video ({e})")
                        video_duration = None
                        video_frames = None
                        video_resolution = None
                        video_fps = None
                else:
                    return None, "Please provide a video file or YouTube URL"

                # Determine output directory
                checkpoint_found = False
                resumed_from_checkpoint = False

                if output_folder and output_folder.strip():
                    output_dir = Path(output_folder.strip())
                    output_dir.mkdir(parents=True, exist_ok=True)
                    log(f"Output: {output_dir}")
                else:
                    # Check for existing checkpoint in video's parent directory
                    if input_video:
                        video_parent = Path(input_video).parent
                        checkpoint_path = video_parent / ".framewright" / "checkpoint.json"
                        if checkpoint_path.exists():
                            try:
                                import json
                                with open(checkpoint_path, 'r') as f:
                                    checkpoint_data = json.load(f)
                                # Verify this checkpoint is for our video
                                checkpoint_source = Path(checkpoint_data.get('source_path', ''))
                                if checkpoint_source.name == Path(input_video).name:
                                    output_dir = video_parent
                                    checkpoint_found = True
                                    stage = checkpoint_data.get('stage', 'unknown')
                                    completed = checkpoint_data.get('last_completed_frame', 0)
                                    total = checkpoint_data.get('total_frames', 0)
                                    log_section("RESUME DETECTED")
                                    log(f"Found checkpoint in: {video_parent}")
                                    log(f"Stage: {stage}")
                                    log(f"Progress: {completed}/{total} frames ({100*completed/total:.1f}%)" if total > 0 else f"Progress: {completed} frames")
                                    resumed_from_checkpoint = True
                            except Exception as e:
                                logger.debug(f"Could not read checkpoint: {e}")

                    if not checkpoint_found:
                        output_dir = Path(tempfile.mkdtemp(prefix="framewright_"))
                        log(f"Working directory: {output_dir}")

                # Log hardware status
                log_section("HARDWARE")
                try:
                    hw_report = check_hardware()
                    if hw_report.gpu.has_gpu:
                        gpu_mode = "GPU"
                        gpu_info = hw_report.gpu.gpu_name
                        vram_info = f"{hw_report.gpu.vram_free_mb}MB free / {hw_report.gpu.vram_total_mb}MB total"
                        log(f"Mode: {gpu_mode} (accelerated)")
                        log(f"GPU: {gpu_info}")
                        log(f"VRAM: {vram_info}")
                        if hw_report.gpu.cuda_available:
                            log(f"Backend: CUDA")
                        elif hw_report.gpu.vulkan_available:
                            log(f"Backend: Vulkan (NCNN)")
                    else:
                        log(f"Mode: CPU (no GPU detected)")
                        log(f"CPU: {hw_report.system.cpu_name}")
                        log(f"Note: Processing will be slower without GPU")
                    log(f"RAM: {hw_report.system.ram_available_gb:.1f}GB available")
                    has_gpu = hw_report.gpu.has_gpu
                except Exception as e:
                    log(f"Mode: Unknown (hardware check failed: {e})")
                    has_gpu = False

                # Time estimation
                if video_frames:
                    log_section("TIME ESTIMATE")

                    # Base processing speed estimates (frames per second)
                    # These are rough estimates based on typical hardware
                    if has_gpu:
                        # GPU processing speeds vary by card
                        # Mid-range GPU (RTX 3060-level): ~2-5 fps for 4x upscale
                        # High-end GPU (RTX 3080+): ~5-10 fps for 4x upscale
                        if scale_factor == 4:
                            base_fps = 3.0  # Conservative estimate
                        else:
                            base_fps = 6.0  # 2x is faster
                    else:
                        # CPU is much slower
                        if scale_factor == 4:
                            base_fps = 0.1  # Very slow on CPU
                        else:
                            base_fps = 0.2

                    # Adjust for additional processing steps
                    multiplier = 1.0
                    if enable_deduplication:
                        multiplier *= 0.95  # Slight overhead for analysis, but saves frames
                    if interpolate:
                        multiplier *= 0.7  # RIFE adds significant time
                    if enable_colorization:
                        multiplier *= 0.5  # Colorization is expensive
                    if enable_watermark_removal:
                        multiplier *= 0.9  # Inpainting adds some time

                    effective_fps = base_fps * multiplier
                    estimated_seconds = video_frames / effective_fps

                    # Add overhead for extraction and encoding
                    estimated_seconds *= 1.2  # 20% overhead

                    # Format time estimate
                    est_mins, est_secs = divmod(int(estimated_seconds), 60)
                    est_hrs, est_mins = divmod(est_mins, 60)

                    if est_hrs > 0:
                        time_str = f"{est_hrs}h {est_mins}m"
                    elif est_mins > 0:
                        time_str = f"{est_mins}m {est_secs}s"
                    else:
                        time_str = f"{est_secs}s"

                    log(f"Estimated time: ~{time_str}")
                    log(f"Processing rate: ~{effective_fps:.1f} frames/sec")

                    # Show speed comparison
                    if video_duration:
                        speed_ratio = video_duration / estimated_seconds
                        if speed_ratio >= 1:
                            log(f"Speed: {speed_ratio:.1f}x realtime")
                        else:
                            log(f"Speed: {1/speed_ratio:.1f}x slower than realtime")

                    if not has_gpu:
                        log(f"Tip: GPU acceleration would be ~20-30x faster")

                # Log settings summary
                log_section("SETTINGS")
                fmt = output_format if output_format else "mkv"
                log(f"Upscale: {scale_factor}x")
                log(f"AI Model: {model}")
                log(f"Quality (CRF): {crf}")
                log(f"Output format: {fmt.upper()}")
                if force_cpu:
                    log(f"⚠️ Processing: CPU MODE (slow - GPU disabled by user)")

                # Configure
                model_dir = None
                if model_download_dir and model_download_dir.strip():
                    model_dir = Path(model_download_dir.strip()).expanduser()
                    log(f"Model location: {model_dir}")

                # Log feature settings
                features = []
                if enhance_audio:
                    features.append("Audio enhancement")
                if enable_deduplication:
                    features.append(f"Frame deduplication (threshold: {deduplication_threshold})")
                if interpolate:
                    features.append(f"Frame interpolation ({target_fps} fps)")
                if enable_colorization:
                    features.append(f"Colorization ({colorization_model})")
                if enable_watermark_removal:
                    detect_mode = "auto-detect" if watermark_auto_detect else "manual"
                    features.append(f"Watermark removal ({detect_mode})")

                if features:
                    log(f"Features: {', '.join(features)}")
                else:
                    log(f"Features: Standard restoration only")

                config = Config(
                    project_dir=output_dir,
                    output_dir=output_dir,
                    scale_factor=scale_factor,
                    model_name=model,
                    crf=crf,
                    output_format=fmt,
                    model_download_dir=model_dir,
                    enable_checkpointing=True,
                    enable_validation=True,
                    # GPU/CPU mode
                    require_gpu=not force_cpu,
                    # Frame deduplication
                    enable_deduplication=enable_deduplication,
                    deduplication_threshold=deduplication_threshold,
                    # Frame interpolation (RIFE)
                    enable_interpolation=interpolate,
                    target_fps=float(target_fps) if interpolate else None,
                    # Colorization
                    enable_colorization=enable_colorization,
                    colorization_model=colorization_model,
                    # Watermark removal
                    enable_watermark_removal=enable_watermark_removal,
                    watermark_auto_detect=watermark_auto_detect,
                )

                # Progress callback with detailed logging (no desc to avoid overlay)
                last_stage = [None]
                def on_progress(current: int, total: int, stage: str):
                    pct = (current / total * 100) if total > 0 else 0
                    progress((current / total) if total > 0 else 0)  # No desc - avoids overlay
                    if stage != last_stage[0]:
                        log_section(stage.upper())
                        last_stage[0] = stage
                    log(f"  Progress: {current}/{total} ({pct:.1f}%)")

                # Create restorer
                log_section("INITIALIZATION")
                log(f"Initializing video restorer...")
                restorer = VideoRestorer(config)
                log(f"Pipeline ready")

                # Run restoration
                log_section("PROCESSING")
                log(f"Starting video restoration...")
                progress(0.05)  # No desc - avoids overlay

                result = restorer.restore_video(
                    source=source,
                    cleanup=False,
                )

                # Final summary
                elapsed_total = time.time() - start_time
                log_section("COMPLETE")
                log(f"Restoration finished successfully!")
                log(f"Total time: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} min)")
                log(f"Output file: {result}")

                return str(result), "\n".join(logs)

            except Exception as e:
                import traceback
                error_msg = f"❌ Error: {e}"
                logs.append(error_msg)
                logs.append("")
                logs.append("─" * 50)
                logs.append("  DETAILED ERROR INFO")
                logs.append("─" * 50)
                logs.append(f"  Type: {type(e).__name__}")
                logs.append(f"  Message: {e}")
                # Add traceback for debugging
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines:
                    if line.strip():
                        logs.append(f"  {line}")
                logger.exception("Restoration failed")
                return None, "\n".join(logs)

        # Build the UI
        with gr.Blocks(
            title="FrameWright - Video Restoration",
        ) as app:
            gr.Markdown(
                """
                # 🎬 FrameWright Video Restoration

                Restore and enhance vintage or degraded video footage using AI.
                Perfect for 100-year-old film fragments, home videos, or any low-quality footage.

                ---
                """
            )

            with gr.Tabs():
                # Tab 1: Hardware Check
                with gr.TabItem("🔧 Hardware Check"):
                    gr.Markdown("### Check if your system can run FrameWright")

                    with gr.Row():
                        with gr.Column(scale=1):
                            status_display = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["status-badge"],
                            )
                        with gr.Column(scale=2):
                            gpu_display = gr.Textbox(label="GPU", interactive=False)
                        with gr.Column(scale=1):
                            ram_display = gr.Textbox(label="RAM", interactive=False)
                        with gr.Column(scale=1):
                            disk_display = gr.Textbox(label="Disk", interactive=False)

                    check_btn = gr.Button("🔍 Check Hardware", variant="primary", size="lg")
                    hardware_report = gr.Textbox(
                        label="Detailed Report",
                        lines=25,
                        interactive=False,
                    )

                    check_btn.click(
                        fn=run_hardware_check,
                        outputs=hardware_report,
                    ).then(
                        fn=get_hardware_summary,
                        outputs=[status_display, gpu_display, ram_display, disk_display],
                    )

                    # Auto-check on load
                    app.load(
                        fn=get_hardware_summary,
                        outputs=[status_display, gpu_display, ram_display, disk_display],
                    )

                # Tab 2: Restore Video
                with gr.TabItem("🎥 Restore Video"):
                    gr.Markdown("### Upload a video or provide a YouTube URL")

                    with gr.Row():
                        with gr.Column(scale=2):
                            input_video = gr.Video(
                                label="Upload Video",
                                sources=["upload"],
                            )
                            gr.Markdown("**OR**")
                            youtube_url = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                            )
                            gr.Markdown("---")
                            output_folder = gr.Textbox(
                                label="Output Folder (optional)",
                                placeholder="/path/to/save/folder",
                                info="Leave empty to use temporary folder",
                            )
                            output_format = gr.Dropdown(
                                choices=["mkv", "mp4", "webm", "avi", "mov"],
                                value="mkv",
                                label="Output Format",
                                info="MKV recommended for archival quality",
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### Settings")

                            # Quality Preset Section
                            with gr.Group():
                                gr.Markdown("#### Quality Preset")
                                quality_preset = gr.Dropdown(
                                    choices=list(QUALITY_PRESETS.keys()),
                                    value="Balanced",
                                    label="Quality Preset",
                                    info="Select a preset to auto-configure settings",
                                )
                                preset_description = gr.Markdown(
                                    f"*{QUALITY_PRESETS['Balanced']['description']}*"
                                )

                            gr.Markdown("---")

                            # Content-Aware Analysis Section
                            with gr.Group():
                                gr.Markdown("#### 🔍 Content-Aware Model Selection")
                                gr.Markdown(
                                    "*Analyze your video to auto-detect anime vs real footage*",
                                    elem_classes=["hint-text"]
                                )

                                with gr.Row():
                                    analyze_btn = gr.Button(
                                        "🔍 Analyze Video",
                                        variant="secondary",
                                        size="sm",
                                    )
                                    apply_btn = gr.Button(
                                        "✅ Apply",
                                        variant="primary",
                                        size="sm",
                                    )

                                analysis_output = gr.Markdown(
                                    value="*Upload a video, then click Analyze to get AI-recommended settings*",
                                    visible=True,
                                )

                                # Hidden state for recommendations
                                rec_scale = gr.State(value=4)
                                rec_model = gr.State(value="realesrgan-x4plus")

                            gr.Markdown("---")

                            scale_factor = gr.Radio(
                                choices=[2, 4],
                                value=4,
                                label="Upscale Factor",
                                info="4x: Old/degraded footage (480p->4K) | 2x: Newer video or mild enhancement",
                            )

                            gr.Markdown(
                                """
*Upscale guide: 4x turns 480p into ~4K, 2x turns 480p into ~1080p. Use 2x if your source is already decent quality.*
                                """,
                                elem_classes=["info-text"]
                            )

                            gr.Markdown(
                                """
                                **AI Model Guide:**
                                | Content Type | Best Model |
                                |--------------|------------|
                                | 🎬 Movies, Film, Real footage | `realesrgan-x4plus` |
                                | 🎌 Anime video (best) | `realesr-animevideov3` |
                                | 🖼️ Anime stills/mixed | `realesrgan-x4plus-anime` |
                                """
                            )

                            model = gr.Dropdown(
                                choices=[
                                    ("🎬 realesrgan-x4plus (Movies, Film, Photos)", "realesrgan-x4plus"),
                                    ("🎌 realesr-animevideov3 (Anime - Best Quality)", "realesr-animevideov3"),
                                    ("🖼️ realesrgan-x4plus-anime (Anime - Alternative)", "realesrgan-x4plus-anime"),
                                ],
                                value="realesrgan-x4plus",
                                label="AI Model",
                                info="Use 'Analyze Video' for auto-detection, or select manually",
                            )

                            crf = gr.Slider(
                                minimum=15,
                                maximum=28,
                                value=18,
                                step=1,
                                label="Quality (CRF)",
                                info="15-18: Archival | 19-22: High quality | 23: FFmpeg default | 24-28: Smaller files",
                            )

                            enhance_audio = gr.Checkbox(
                                value=True,
                                label="Enhance Audio",
                                info="Apply noise reduction and normalization",
                            )

                            force_cpu = gr.Checkbox(
                                value=False,
                                label="⚠️ Force CPU Mode",
                                info="Use CPU instead of GPU (VERY slow, use only if GPU fails)",
                            )

                            with gr.Accordion("Advanced Options", open=False):
                                model_download_dir = gr.Textbox(
                                    label="Model Download Location",
                                    placeholder="~/.framewright/models",
                                    info="Where AI models are downloaded (leave empty for default)",
                                )

                                gr.Markdown("### 🎞️ Frame Deduplication")
                                gr.Markdown(
                                    "*For old films digitized at higher FPS with duplicate frames (e.g., 18fps film uploaded as 25fps)*"
                                )

                                enable_deduplication = gr.Checkbox(
                                    value=False,
                                    label="Enable Frame Deduplication",
                                    info="Extract unique frames only - dramatically speeds up processing for padded footage",
                                )

                                deduplication_threshold = gr.Slider(
                                    minimum=0.90,
                                    maximum=0.99,
                                    value=0.98,
                                    step=0.01,
                                    label="Similarity Threshold",
                                    info="0.98 = strict (catches most duplicates), 0.95 = lenient (handles compression artifacts)",
                                    visible=False,
                                )

                                dedup_info = gr.Markdown(
                                    """
**How it works:**
1. Analyzes all frames to detect duplicates (e.g., 18fps film padded to 25fps)
2. Enhances only unique frames (saves 30-50% processing time)
3. RIFE interpolates back to smooth motion
4. Reconstructs final video at target FPS

**Recommended workflow for old films:**
- ✅ Enable Deduplication
- ✅ Enable Frame Interpolation (RIFE)
- Set Target FPS to 24 for natural motion
                                    """,
                                    visible=False,
                                )

                                enable_deduplication.change(
                                    fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
                                    inputs=enable_deduplication,
                                    outputs=[deduplication_threshold, dedup_info],
                                )

                                gr.Markdown("---")
                                gr.Markdown("### 🎬 Frame Interpolation (RIFE)")

                                interpolate = gr.Checkbox(
                                    value=False,
                                    label="Enable Frame Interpolation",
                                    info="Smoothly increase frame rate using AI",
                                )

                                target_fps = gr.Slider(
                                    minimum=18,
                                    maximum=60,
                                    value=24,
                                    step=6,
                                    label="Target FPS",
                                    visible=False,
                                )

                                fps_guide = gr.Markdown(
                                    """
**FPS Guide for Historical Footage:**
| Era | Original FPS | Recommended Target |
|-----|--------------|-------------------|
| 1890s-1920s (Silent films) | 12-18 fps | **24 fps** - natural smoothing |
| 1920s-1950s (Early talkies) | 18-24 fps | **24-30 fps** - subtle improvement |
| 1950s-1980s (TV/Film) | 24-30 fps | **30 fps** - standard |
| Modern footage | 24-60 fps | **30-60 fps** - if needed |

*Tip: For old films, 24 fps is usually ideal. 60 fps can look unnaturally smooth ("soap opera effect") for vintage footage.*
                                    """,
                                    visible=False,
                                )

                                interpolate.change(
                                    fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
                                    inputs=interpolate,
                                    outputs=[target_fps, fps_guide],
                                )

                                gr.Markdown("---")
                                gr.Markdown("### 🎨 Colorization")

                                enable_colorization = gr.Checkbox(
                                    value=False,
                                    label="Enable Colorization",
                                    info="Colorize black & white footage using AI",
                                )

                                colorization_model = gr.Dropdown(
                                    choices=[
                                        ("DDColor (Better quality)", "ddcolor"),
                                        ("DeOldify (Open source)", "deoldify"),
                                    ],
                                    value="ddcolor",
                                    label="Colorization Model",
                                    visible=False,
                                )

                                enable_colorization.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=enable_colorization,
                                    outputs=colorization_model,
                                )

                                gr.Markdown("---")
                                gr.Markdown("### 🚫 Watermark Removal")

                                enable_watermark_removal = gr.Checkbox(
                                    value=False,
                                    label="Enable Watermark Removal",
                                    info="Remove logos/watermarks using AI inpainting",
                                )

                                watermark_auto_detect = gr.Checkbox(
                                    value=True,
                                    label="Auto-detect Watermarks",
                                    info="Automatically detect watermark locations",
                                    visible=False,
                                )

                                enable_watermark_removal.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=enable_watermark_removal,
                                    outputs=watermark_auto_detect,
                                )

                    with gr.Row():
                        restore_btn = gr.Button(
                            "🚀 Start Restoration",
                            variant="primary",
                            size="lg",
                        )
                        stop_btn = gr.Button(
                            "⏹️ Stop",
                            variant="stop",
                            size="lg",
                        )

                    # Status bar - separate from log to avoid overlay issues
                    with gr.Row():
                        status_text = gr.Markdown(
                            value="*Ready to process*",
                            elem_id="status-bar",
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            output_video = gr.Video(label="Restored Video")
                        with gr.Column(scale=1):
                            log_output = gr.Textbox(
                                label="Processing Log",
                                lines=20,
                                max_lines=30,
                                interactive=False,
                                autoscroll=True,
                            )

                    restore_event = restore_btn.click(
                        fn=restore_video,
                        inputs=[
                            input_video,
                            youtube_url,
                            output_folder,
                            output_format,
                            scale_factor,
                            model,
                            crf,
                            enhance_audio,
                            force_cpu,
                            model_download_dir,
                            interpolate,
                            target_fps,
                            enable_deduplication,
                            deduplication_threshold,
                            enable_colorization,
                            colorization_model,
                            enable_watermark_removal,
                            watermark_auto_detect,
                        ],
                        outputs=[output_video, log_output],
                    )

                    # Stop button cancels the restoration (preserves existing log)
                    def cancel_with_log(current_log):
                        cancelled_msg = "\n\n" + "=" * 50 + "\n  ⏹️ CANCELLED BY USER\n" + "=" * 50
                        return None, (current_log or "") + cancelled_msg

                    stop_btn.click(
                        fn=cancel_with_log,
                        inputs=[log_output],
                        outputs=[output_video, log_output],
                        cancels=[restore_event],
                    )

                    # Analyze Video button - detects content type and recommends settings
                    analyze_btn.click(
                        fn=analyze_video_for_settings,
                        inputs=[input_video, youtube_url],
                        outputs=[analysis_output, rec_scale, rec_model],
                    )

                    # Apply Recommendations button - updates settings with recommendations
                    def apply_recommendations(recommended_scale, recommended_model):
                        """Apply the recommended settings to the UI controls."""
                        # Also set preset to Custom since we're overriding
                        return recommended_scale, recommended_model, "Custom"

                    apply_btn.click(
                        fn=apply_recommendations,
                        inputs=[rec_scale, rec_model],
                        outputs=[scale_factor, model, quality_preset],
                    )

                    # Preset change handler - updates all related fields
                    quality_preset.change(
                        fn=on_preset_change,
                        inputs=[quality_preset],
                        outputs=[
                            scale_factor,
                            crf,
                            model,
                            enhance_audio,
                            interpolate,
                            target_fps,
                            enable_colorization,
                            enable_watermark_removal,
                            preset_description,
                        ],
                    )

                    # When user manually changes settings, switch to Custom preset
                    def switch_to_custom():
                        return "Custom", "*Manual settings mode - adjust parameters as needed.*"

                    for component in [scale_factor, crf, model, enhance_audio, interpolate,
                                      enable_colorization, enable_watermark_removal]:
                        component.change(
                            fn=switch_to_custom,
                            outputs=[quality_preset, preset_description],
                        )

                # Tab 3: Batch Processing Queue
                with gr.TabItem("📋 Batch Queue"):
                    gr.Markdown(
                        """
                        ### Batch Processing Queue

                        Add multiple videos to process them sequentially with the same settings.
                        """
                    )

                    with gr.Row():
                        # Left column: Add to queue
                        with gr.Column(scale=1):
                            gr.Markdown("#### Add Videos to Queue")

                            batch_file_upload = gr.File(
                                label="Upload Video Files",
                                file_types=["video"],
                                file_count="multiple",
                            )

                            gr.Markdown("**OR**")

                            batch_youtube_url = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                            )

                            with gr.Row():
                                add_file_btn = gr.Button(
                                    "Add Files to Queue",
                                    variant="secondary",
                                )
                                add_url_btn = gr.Button(
                                    "Add URL to Queue",
                                    variant="secondary",
                                )

                            queue_status_msg = gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Queue is empty",
                            )

                        # Right column: Queue display and controls
                        with gr.Column(scale=2):
                            gr.Markdown("#### Queue")

                            queue_display = gr.Dataframe(
                                headers=["ID", "Source", "Type", "Status", "Progress", "Message"],
                                datatype=["str", "str", "str", "str", "str", "str"],
                                interactive=False,
                                wrap=True,
                                row_count=(5, "dynamic"),
                            )

                            with gr.Row():
                                selected_item_id = gr.Textbox(
                                    label="Selected Item ID",
                                    placeholder="Enter ID to select",
                                    scale=2,
                                )
                                move_up_btn = gr.Button("Move Up", size="sm")
                                move_down_btn = gr.Button("Move Down", size="sm")
                                remove_btn = gr.Button("Remove", variant="stop", size="sm")

                            with gr.Row():
                                clear_queue_btn = gr.Button(
                                    "Clear Queue",
                                    variant="stop",
                                )
                                refresh_queue_btn = gr.Button(
                                    "Refresh Queue",
                                    variant="secondary",
                                )

                    gr.Markdown("---")
                    gr.Markdown("#### Batch Settings")
                    gr.Markdown("*Uses the same settings from the Restore Video tab*")

                    with gr.Row():
                        batch_output_folder = gr.Textbox(
                            label="Output Folder",
                            placeholder="/path/to/output/folder",
                            info="All processed videos will be saved here",
                        )
                        batch_output_format = gr.Dropdown(
                            choices=["mkv", "mp4", "webm", "avi", "mov"],
                            value="mkv",
                            label="Output Format",
                        )

                    # Batch preset selection (mirrors restore tab)
                    batch_preset = gr.Dropdown(
                        choices=list(QUALITY_PRESETS.keys()),
                        value="Balanced",
                        label="Quality Preset",
                        info="Same presets as Restore Video tab",
                    )

                    start_batch_btn = gr.Button(
                        "Start Batch Processing",
                        variant="primary",
                        size="lg",
                    )

                    batch_log_output = gr.Textbox(
                        label="Batch Processing Log",
                        lines=10,
                        interactive=False,
                    )

                    # Queue event handlers
                    def add_files_to_queue(files) -> Tuple[List[List[str]], str]:
                        """Add uploaded files to the queue."""
                        if not files:
                            return get_queue_dataframe(), "No files selected"

                        added_count = 0
                        for file in files:
                            file_path = file.name if hasattr(file, 'name') else str(file)
                            add_to_queue(file_path, "file")
                            added_count += 1

                        return get_queue_dataframe(), f"Added {added_count} file(s) to queue"

                    def add_url_to_queue_handler(url: str) -> Tuple[List[List[str]], str]:
                        """Add YouTube URL to the queue."""
                        return add_to_queue(url, "youtube")

                    add_file_btn.click(
                        fn=add_files_to_queue,
                        inputs=[batch_file_upload],
                        outputs=[queue_display, queue_status_msg],
                    )

                    add_url_btn.click(
                        fn=add_url_to_queue_handler,
                        inputs=[batch_youtube_url],
                        outputs=[queue_display, queue_status_msg],
                    )

                    remove_btn.click(
                        fn=remove_from_queue,
                        inputs=[selected_item_id],
                        outputs=[queue_display, queue_status_msg],
                    )

                    move_up_btn.click(
                        fn=lambda x: move_queue_item(x, "up"),
                        inputs=[selected_item_id],
                        outputs=[queue_display],
                    )

                    move_down_btn.click(
                        fn=lambda x: move_queue_item(x, "down"),
                        inputs=[selected_item_id],
                        outputs=[queue_display],
                    )

                    clear_queue_btn.click(
                        fn=clear_queue,
                        outputs=[queue_display, queue_status_msg],
                    )

                    refresh_queue_btn.click(
                        fn=lambda: get_queue_dataframe(),
                        outputs=[queue_display],
                    )

                    # Start batch processing
                    # We need to use the settings from the restore tab
                    # For now, we use defaults matching the batch preset
                    def start_batch_with_preset(
                        output_folder: str,
                        output_format: str,
                        preset_name: str,
                        progress: gr.Progress = gr.Progress(),
                    ) -> Tuple[List[List[str]], str]:
                        """Start batch processing using preset settings."""
                        preset = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["Balanced"])

                        # Use preset values or defaults
                        return process_batch_queue(
                            output_folder=output_folder,
                            output_format=output_format,
                            preset_name=preset_name,
                            scale_factor=preset.get("scale_factor", 4) or 4,
                            model=preset.get("model", "realesrgan-x4plus") or "realesrgan-x4plus",
                            crf=preset.get("crf", 20) or 20,
                            enhance_audio=preset.get("enhance_audio", True) if preset.get("enhance_audio") is not None else True,
                            model_download_dir="",
                            interpolate=preset.get("interpolate", False) or False,
                            target_fps=preset.get("target_fps", 60) or 60,
                            enable_colorization=preset.get("enable_colorization", False) or False,
                            colorization_model="ddcolor",
                            enable_watermark_removal=preset.get("enable_watermark_removal", False) or False,
                            watermark_auto_detect=True,
                            progress=progress,
                        )

                    start_batch_btn.click(
                        fn=start_batch_with_preset,
                        inputs=[batch_output_folder, batch_output_format, batch_preset],
                        outputs=[queue_display, batch_log_output],
                    )

                # Tab 4: Help
                with gr.TabItem("❓ Help"):
                    gr.Markdown(
                        """
                        ## Quick Start Guide

                        ### 1. Check Your Hardware
                        - Go to the **Hardware Check** tab
                        - Click "Check Hardware" to see if your system is ready
                        - For best results, you need:
                          - **NVIDIA GPU** with 4GB+ VRAM (or be patient with CPU processing)
                          - **16GB RAM** recommended
                          - **50GB+ free disk space**

                        ### 2. Restore a Video
                        - Go to the **Restore Video** tab
                        - Either upload a video file or paste a YouTube URL
                        - **NEW: Click "🔍 Analyze Video"** to auto-detect content type (anime vs real footage)
                        - Click "✅ Apply" to use recommended settings
                        - Or manually adjust settings
                        - Click "Start Restoration" and wait

                        ### Quality Presets

                        | Preset | Best For | Settings |
                        |--------|----------|----------|
                        | **Quick Preview** | Fast preview of results | 2x scale, CRF 23, basic enhancements |
                        | **Balanced** | Default, good quality/speed | 4x scale, CRF 20, standard enhancements |
                        | **Maximum Quality** | Archival, best results | 4x scale, CRF 15, all enhancements + interpolation |
                        | **Custom** | Manual control | Set all parameters yourself |

                        ### Settings Explained

                        | Setting | What it does |
                        |---------|--------------|
                        | **Upscale Factor** | 4x doubles size twice (best for old film), 2x for newer video |
                        | **AI Model** | See model guide below |
                        | **Quality (CRF)** | 15-18 for archival, 20-23 for web sharing |
                        | **Enhance Audio** | Removes hiss, rumble, normalizes volume |
                        | **Frame Interpolation** | RIFE AI to increase frame rate smoothly |
                        | **Frame Deduplication** | Detect & skip duplicate frames (for old films padded to higher FPS) |
                        | **Model Download Location** | Custom folder for AI model files (Advanced Options) |
                        | **Colorization** | AI-powered colorization for B&W footage (DDColor/DeOldify) |
                        | **Watermark Removal** | Remove logos/watermarks using AI inpainting |

                        ### AI Model Guide

                        **💡 Tip:** Use the **"🔍 Analyze Video"** button to auto-detect content type!

                        | Content Type | Recommended Model | Auto-Detected As |
                        |--------------|-------------------|------------------|
                        | **Movies & Film** | `realesrgan-x4plus` | Landscape, Face Portrait |
                        | **Photos & Real Images** | `realesrgan-x4plus` | High Contrast, Architecture |
                        | **Anime (Best Quality)** | `realesr-animevideov3` | Animation |
                        | **Anime (Alternative)** | `realesrgan-x4plus-anime` | Animation |

                        The analyzer examines sample frames to detect:
                        - **Anime/Animation**: Low edge density, flat colors → Uses anime models
                        - **Real footage with faces**: Face detection → Enables face restoration
                        - **Degradation level**: Noise, grain, blur → Recommends 4x vs 2x scale

                        ### Tips for Old Film (1890s-1920s)

                        - Use **4x upscaling** - old film is usually low resolution
                        - Keep **enhance audio** on - old recordings are noisy
                        - For silent films, the pipeline handles missing audio gracefully
                        - Processing time: expect 2-10x the video length

                        ### Frame Deduplication Guide

                        **When to use:** Old films (1890s-1920s) were shot at 16-18 fps but are often
                        digitized/uploaded at 24-30 fps with duplicate frames inserted. This wastes
                        processing time enhancing the same frame multiple times.

                        **How it works:**
                        1. Analyzes all frames using perceptual hashing
                        2. Detects duplicates (e.g., 25fps video with only 18 unique frames per second)
                        3. Enhances only unique frames (30-50% faster!)
                        4. RIFE interpolates to create smooth motion at target FPS
                        5. Reconstructs final video

                        **Recommended settings for old films:**
                        | Setting | Value | Why |
                        |---------|-------|-----|
                        | **Enable Deduplication** | ✅ Yes | Saves 30-50% processing time |
                        | **Similarity Threshold** | 0.98 | Catches most duplicates |
                        | **Enable Interpolation** | ✅ Yes | Smooth motion from unique frames |
                        | **Target FPS** | 24 | Natural motion for vintage footage |

                        **Threshold guide:**
                        - **0.98-0.99**: Strict - only exact/near-exact duplicates
                        - **0.95-0.97**: Moderate - handles some compression artifacts
                        - **0.90-0.94**: Lenient - for heavily compressed sources

                        ### Colorization Guide

                        | Model | Best For | Notes |
                        |-------|----------|-------|
                        | **DDColor** | Higher quality results | Better color accuracy, slower |
                        | **DeOldify** | General colorization | Open source, faster |

                        **Tips:**
                        - Works best on high-contrast B&W footage
                        - May produce artifacts on already-colored videos
                        - GPU recommended for reasonable processing speed

                        ### Watermark Removal Guide

                        - **Auto-detect**: Automatically finds watermarks in corners
                        - **Manual regions**: Specify exact coordinates via CLI
                        - Uses LaMA (Large Mask Inpainting) for best quality
                        - Falls back to OpenCV if LaMA unavailable

                        ### Batch Processing Guide

                        The **Batch Queue** tab allows you to process multiple videos with the same settings:

                        1. **Adding Videos**:
                           - Upload multiple video files at once
                           - Or add YouTube URLs one at a time
                           - Each video is added to the queue

                        2. **Managing the Queue**:
                           - Enter an item's ID in the "Selected Item ID" field
                           - Use "Move Up" / "Move Down" to change processing order
                           - Use "Remove" to delete an item from the queue
                           - "Clear Queue" removes all items

                        3. **Processing**:
                           - Select a quality preset (applies to all videos)
                           - Set the output folder and format
                           - Click "Start Batch Processing"
                           - Monitor progress in the log and queue display

                        **Tips:**
                        - Process similar content types together for best results
                        - Use "Quick Preview" preset first to check results quickly
                        - The queue persists during your session but clears on refresh

                        ### Troubleshooting

                        **"Out of memory" errors:**
                        - Your GPU doesn't have enough VRAM
                        - Try using 2x upscaling instead of 4x
                        - Close other applications

                        **Processing is very slow:**
                        - You may be running on CPU (no GPU detected)
                        - Check the Hardware tab to confirm GPU status

                        **Video looks worse:**
                        - Try a different AI model
                        - Some heavily compressed videos don't improve much

                        ---

                        For more help, visit: [GitHub Issues](https://github.com/your-repo/framewright/issues)
                        """
                    )

        return app


def launch_ui(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1",
) -> None:
    """Launch the FrameWright web UI.

    Args:
        share: Create a public shareable link
        server_port: Port to run server on
        server_name: Server hostname (use "0.0.0.0" for network access)
    """
    if not HAS_GRADIO:
        print(install_gradio_instructions())
        return

    print("\n" + "=" * 50)
    print("  🎬 FrameWright Video Restoration UI")
    print("=" * 50)
    print(f"\n  Starting web interface...")
    print(f"  Open your browser to: http://{server_name}:{server_port}")
    if share:
        print("  (A public link will be generated...)")
    print("\n  Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")

    app = create_ui()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True,
        theme=gr.themes.Soft(),
    )
