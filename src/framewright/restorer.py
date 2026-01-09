"""Video restoration module using Real-ESRGAN for frame enhancement.

Includes robust error handling, checkpointing, and quality validation.
"""
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

from .config import Config, RestoreOptions


# =============================================================================
# Binary Path Helpers
# =============================================================================

def _find_ffmpeg_binary(name: str) -> Optional[str]:
    """Find ffmpeg/ffprobe binary in PATH or common locations."""
    # Check PATH first
    path = shutil.which(name)
    if path:
        return path

    # Check common installation directories
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    search_paths = [
        # Common Windows locations
        Path("F:/ffmpeg-8.0.1-essentials_build/bin") / f"{name}{exe_suffix}",
        Path("C:/ffmpeg/bin") / f"{name}{exe_suffix}",
        Path.home() / "ffmpeg" / "bin" / f"{name}{exe_suffix}",
        Path.home() / ".framewright" / "bin" / f"{name}{exe_suffix}",
        # Common Unix locations
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
    ]

    for search_path in search_paths:
        if search_path.exists():
            return str(search_path)

    return None


# Cache the binary paths
_ffmpeg_path: Optional[str] = None
_ffprobe_path: Optional[str] = None


def get_ffmpeg_path() -> str:
    """Get ffmpeg binary path, raising error if not found."""
    global _ffmpeg_path
    if _ffmpeg_path is None:
        _ffmpeg_path = _find_ffmpeg_binary("ffmpeg")
    if _ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg not found in PATH or common locations")
    return _ffmpeg_path


def get_ffprobe_path() -> str:
    """Get ffprobe binary path, raising error if not found."""
    global _ffprobe_path
    if _ffprobe_path is None:
        _ffprobe_path = _find_ffmpeg_binary("ffprobe")
    if _ffprobe_path is None:
        raise FileNotFoundError("ffprobe not found in PATH or common locations")
    return _ffprobe_path


def _find_ytdlp_binary() -> Optional[str]:
    """Find yt-dlp binary in PATH or common locations."""
    # Check PATH first
    path = shutil.which("yt-dlp")
    if path:
        return path

    # Check common installation directories
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    home = Path.home()
    search_paths = [
        # Python user scripts (pip install --user)
        home / "AppData" / "Roaming" / "Python" / "Python313" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Roaming" / "Python" / "Python311" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "Scripts" / f"yt-dlp{exe_suffix}",
        # FrameWright bin directory
        home / ".framewright" / "bin" / f"yt-dlp{exe_suffix}",
        # Common Unix locations
        Path("/usr/local/bin") / "yt-dlp",
        home / ".local" / "bin" / "yt-dlp",
    ]

    for search_path in search_paths:
        if search_path.exists():
            return str(search_path)

    return None


# Cache yt-dlp path
_ytdlp_path: Optional[str] = None


def get_ytdlp_path() -> str:
    """Get yt-dlp binary path, raising error if not found."""
    global _ytdlp_path
    if _ytdlp_path is None:
        _ytdlp_path = _find_ytdlp_binary()
    if _ytdlp_path is None:
        raise FileNotFoundError("yt-dlp not found in PATH or common locations")
    return _ytdlp_path


@dataclass
class ProgressInfo:
    """Detailed progress information for frame-level tracking.

    Attributes:
        stage: Current processing stage name
        progress: Progress value between 0.0 and 1.0
        eta_seconds: Estimated time remaining in seconds (None if unknown)
        frames_completed: Number of frames completed in current stage
        frames_total: Total frames to process in current stage
        stage_start_time: Unix timestamp when current stage started
        elapsed_seconds: Seconds elapsed since stage started
    """
    stage: str
    progress: float
    eta_seconds: Optional[float] = None
    frames_completed: int = 0
    frames_total: int = 0
    stage_start_time: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0

    def __post_init__(self):
        """Calculate elapsed time if not provided."""
        if self.elapsed_seconds == 0.0 and self.stage_start_time:
            self.elapsed_seconds = time.time() - self.stage_start_time

    @property
    def eta_formatted(self) -> str:
        """Return ETA as formatted string (HH:MM:SS or 'Unknown')."""
        if self.eta_seconds is None or self.eta_seconds < 0:
            return "Unknown"
        hours, remainder = divmod(int(self.eta_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def elapsed_formatted(self) -> str:
        """Return elapsed time as formatted string (HH:MM:SS)."""
        hours, remainder = divmod(int(self.elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def frames_per_second(self) -> float:
        """Calculate current processing speed in frames per second."""
        if self.elapsed_seconds > 0 and self.frames_completed > 0:
            return self.frames_completed / self.elapsed_seconds
        return 0.0


from .checkpoint import CheckpointManager, PipelineCheckpoint
from .errors import (
    VideoRestorerError,
    DownloadError,
    MetadataError,
    AudioExtractionError,
    FrameExtractionError,
    EnhancementError,
    ReassemblyError,
    TransientError,
    VRAMError,
    DiskSpaceError,
    FatalError,
    DependencyError,
    GPURequiredError,
    CPUFallbackError,
    ErrorContext,
    ErrorReport,
    RetryableOperation,
    classify_error,
    create_error_context,
)
from .validators import (
    validate_frame_integrity,
    validate_frame_sequence,
    validate_enhancement_quality,
    validate_temporal_consistency,
    SequenceReport,
)
from .utils.gpu import (
    get_gpu_memory_info,
    get_best_gpu,
    get_all_gpus_multivendor,
    calculate_optimal_tile_size,
    get_adaptive_tile_sequence,
    VRAMMonitor,
    GPUVendor,
)
from .utils.disk import (
    validate_disk_space,
    DiskSpaceMonitor,
)
from .utils.dependencies import validate_all_dependencies
from .utils.ffmpeg import get_best_video_codec
from .processors.interpolation import FrameInterpolator, InterpolationError
from .processors.analyzer import FrameAnalyzer, VideoAnalysis
from .processors.adaptive_enhance import AdaptiveEnhancer, AdaptiveEnhanceResult
from .processors.streaming import StreamingProcessor, StreamingConfig, ChunkInfo
from .processors.ncnn_vulkan import (
    NcnnVulkanBackend,
    NcnnVulkanConfig,
    get_ncnn_vulkan_path,
    is_ncnn_vulkan_available,
)
from .processors.pytorch_realesrgan import (
    is_pytorch_esrgan_available,
    enhance_frame_pytorch,
    PyTorchESRGANConfig,
    convert_ncnn_model_name,
    clear_upsampler_cache,
)
from .processors.deduplication import (
    FrameDeduplicator,
    DeduplicationConfig,
    DeduplicationResult,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoRestorer:
    """Video restoration pipeline using Real-ESRGAN for upscaling.

    This class handles the complete workflow with robustness features:
    1. Download video from URL or use local file
    2. Pre-flight validation (disk space, dependencies)
    3. Extract and analyze metadata
    4. Extract audio track
    5. Extract frames as PNG with checkpointing
    6. Enhance frames using Real-ESRGAN with retry logic
    7. Quality validation
    8. Reassemble video with enhanced frames and audio

    Attributes:
        config: Configuration object for the restoration pipeline
        metadata: Video metadata extracted from ffprobe
        progress_callback: Optional callback for progress updates
        checkpoint_manager: Manages checkpoint state for resume capability
    """

    def __init__(
        self,
        config: Config,
        progress_callback: Optional[Callable[[Union[ProgressInfo, Tuple[str, float]]], None]] = None
    ) -> None:
        """Initialize VideoRestorer with configuration.

        Args:
            config: Configuration object
            progress_callback: Optional callback function that accepts either:
                              - ProgressInfo object with detailed frame-level info
                              - (stage: str, progress: float) tuple for backward compatibility
                              The callback signature is auto-detected on first call.
        """
        self.config = config
        self.metadata: Dict[str, Any] = {}
        self.progress_callback = progress_callback
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self._vram_monitor: Optional[VRAMMonitor] = None
        self._disk_monitor: Optional[DiskSpaceMonitor] = None
        self._current_tile_size: Optional[int] = None
        self._error_report = ErrorReport()
        self._video_analysis: Optional[VideoAnalysis] = None
        self._enhance_result: Optional[AdaptiveEnhanceResult] = None
        self._dedup_result: Optional[DeduplicationResult] = None

        # Progress tracking state
        self._stage_start_times: Dict[str, float] = {}
        self._frame_processing_times: deque = deque(maxlen=100)  # Rolling window
        self._callback_accepts_progress_info: Optional[bool] = None  # Auto-detect

        # Verify required tools are installed
        self._verify_dependencies()

        # Initialize checkpoint manager if enabled
        if config.enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(
                project_dir=config.project_dir,
                checkpoint_interval=config.checkpoint_interval,
                config_hash=config.get_hash(),
            )

        # Initialize monitors if enabled
        if config.enable_vram_monitoring:
            self._vram_monitor = VRAMMonitor(threshold_mb=500)

        if config.enable_disk_validation:
            self._disk_monitor = DiskSpaceMonitor(
                project_dir=config.project_dir,
                warning_threshold_gb=1.0,
                critical_threshold_gb=0.5,
            )

    def _verify_dependencies(self) -> None:
        """Verify all required external tools are available with versions."""
        report = validate_all_dependencies(
            required=["ffmpeg", "ffprobe", "realesrgan", "yt-dlp"],
            optional=["rife"],
        )

        if not report.is_ready():
            missing = ", ".join(report.missing_required)
            raise DependencyError(
                f"Missing required tools: {missing}. "
                "Please install them before running the pipeline.\n"
                f"{report.summary()}"
            )

        # Log warnings for version issues
        for warning in report.warnings:
            logger.warning(warning)

        # Store dependency info for later use
        self._dependency_report = report

    def _validate_gpu_available(self) -> None:
        """Pre-flight GPU validation to prevent CPU fallback.

        When require_gpu=True, this ensures a GPU is available and working
        before processing begins. Prevents runaway CPU usage.

        Raises:
            GPURequiredError: If GPU is required but not available or not working
        """
        if not self.config.require_gpu:
            logger.info("GPU requirement disabled - CPU fallback allowed")
            return

        logger.info("Validating GPU availability (require_gpu=True)...")

        # Check if any GPU is available
        gpus = get_all_gpus_multivendor()

        if not gpus:
            raise GPURequiredError(
                "No GPU detected but require_gpu=True. "
                "Processing would fall back to CPU which can freeze your system. "
                "Options:\n"
                "  1. Install/enable a GPU with Vulkan support\n"
                "  2. Set require_gpu=False to allow CPU processing (not recommended)\n"
                "  3. Check GPU drivers are installed correctly"
            )

        # Get the best GPU
        best_gpu = get_best_gpu()
        if not best_gpu:
            raise GPURequiredError(
                "GPU detected but not accessible. Check drivers and permissions."
            )

        # Check backend preference - PyTorch doesn't need ncnn-vulkan
        env_backend = os.environ.get("FRAMEWRIGHT_BACKEND", "").lower()
        using_pytorch = (env_backend == "pytorch" and is_pytorch_esrgan_available())

        # Only require ncnn-vulkan if not using PyTorch backend
        if not using_pytorch and not is_ncnn_vulkan_available():
            # Check if PyTorch is available as fallback
            if is_pytorch_esrgan_available():
                logger.info(
                    f"ncnn-vulkan not installed, but PyTorch Real-ESRGAN is available. "
                    "Using PyTorch backend for GPU acceleration."
                )
                os.environ["FRAMEWRIGHT_BACKEND"] = "pytorch"
                using_pytorch = True
            else:
                raise GPURequiredError(
                    f"GPU found ({best_gpu.name}) but no Real-ESRGAN backend available. "
                    "Install either:\n"
                    "  1. PyTorch backend: pip install realesrgan\n"
                    "  2. ncnn-vulkan: python -c \"from framewright.processors.ncnn_vulkan import install_ncnn_vulkan; install_ncnn_vulkan()\""
                )

        # Check for minimum VRAM
        min_vram_mb = 1024  # Minimum 1GB VRAM for basic processing
        if best_gpu.total_memory_mb < min_vram_mb:
            raise GPURequiredError(
                f"GPU {best_gpu.name} has only {best_gpu.total_memory_mb}MB VRAM. "
                f"Minimum {min_vram_mb}MB required for GPU processing. "
                "Consider using a GPU with more VRAM or set require_gpu=False."
            )

        # Only check Vulkan support if using ncnn-vulkan backend
        if not using_pytorch and not best_gpu.vulkan_supported:
            raise GPURequiredError(
                f"GPU {best_gpu.name} does not support Vulkan. "
                "Either install Vulkan drivers or use PyTorch backend:\n"
                "  export FRAMEWRIGHT_BACKEND=pytorch"
            )

        logger.info(
            f"GPU validated: {best_gpu.name} "
            f"({best_gpu.vendor.value}, {best_gpu.total_memory_mb}MB VRAM, "
            f"Vulkan: {'Yes' if best_gpu.vulkan_supported else 'No'})"
        )

    def _update_progress(
        self,
        stage: str,
        progress: float,
        eta_seconds: Optional[float] = None,
        frames_completed: int = 0,
        frames_total: int = 0,
    ) -> None:
        """Update progress via callback if provided.

        Creates a ProgressInfo object with detailed timing information and
        invokes the callback. Maintains backward compatibility with callbacks
        that only accept (stage, progress) arguments.

        Args:
            stage: Current processing stage
            progress: Progress value between 0.0 and 1.0
            eta_seconds: Estimated time remaining in seconds (None if unknown)
            frames_completed: Number of frames completed in current stage
            frames_total: Total frames to process in current stage
        """
        # Track stage start time
        if stage not in self._stage_start_times or progress == 0.0:
            self._stage_start_times[stage] = time.time()

        stage_start_time = self._stage_start_times.get(stage, time.time())
        elapsed = time.time() - stage_start_time

        # Auto-calculate ETA if not provided and we have frame data
        if eta_seconds is None and frames_completed > 0 and frames_total > 0:
            eta_seconds = self._calculate_eta(frames_completed, frames_total, elapsed)

        # Create detailed progress info
        progress_info = ProgressInfo(
            stage=stage,
            progress=progress,
            eta_seconds=eta_seconds,
            frames_completed=frames_completed,
            frames_total=frames_total,
            stage_start_time=stage_start_time,
            elapsed_seconds=elapsed,
        )

        # Invoke callback with backward compatibility
        if self.progress_callback:
            self._invoke_progress_callback(progress_info)

        # Enhanced logging with ETA when available
        if eta_seconds is not None and frames_total > 0:
            logger.info(
                f"{stage}: {progress * 100:.1f}% complete "
                f"({frames_completed}/{frames_total} frames, "
                f"ETA: {progress_info.eta_formatted}, "
                f"{progress_info.frames_per_second:.1f} fps)"
            )
        else:
            logger.info(f"{stage}: {progress * 100:.1f}% complete")

    def _invoke_progress_callback(self, progress_info: ProgressInfo) -> None:
        """Invoke progress callback with auto-detection of signature.

        Tries to call with ProgressInfo first, falls back to (stage, progress)
        tuple for backward compatibility.

        Args:
            progress_info: Detailed progress information
        """
        if self.progress_callback is None:
            return

        # Check if we've already detected the callback type
        if self._callback_accepts_progress_info is True:
            self.progress_callback(progress_info)
            return
        elif self._callback_accepts_progress_info is False:
            self.progress_callback(progress_info.stage, progress_info.progress)
            return

        # Auto-detect callback signature on first call
        import inspect
        try:
            sig = inspect.signature(self.progress_callback)
            params = list(sig.parameters.values())

            # Check if callback accepts single ProgressInfo argument
            if len(params) == 1:
                param = params[0]
                # Check type hint if available
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation is ProgressInfo or \
                       (hasattr(param.annotation, '__origin__') and
                        param.annotation.__origin__ is Union and
                        ProgressInfo in param.annotation.__args__):
                        self._callback_accepts_progress_info = True
                        self.progress_callback(progress_info)
                        return

                # Try calling with ProgressInfo
                try:
                    self.progress_callback(progress_info)
                    self._callback_accepts_progress_info = True
                    return
                except TypeError:
                    pass

            # Fallback to legacy (stage, progress) signature
            self._callback_accepts_progress_info = False
            self.progress_callback(progress_info.stage, progress_info.progress)

        except (ValueError, TypeError):
            # If signature inspection fails, try legacy signature
            self._callback_accepts_progress_info = False
            self.progress_callback(progress_info.stage, progress_info.progress)

    def _calculate_eta(
        self,
        frames_completed: int,
        frames_total: int,
        elapsed_seconds: float,
    ) -> Optional[float]:
        """Calculate estimated time remaining using rolling average.

        Uses a rolling window of recent frame processing times to smooth
        out variations and provide more accurate estimates.

        Args:
            frames_completed: Number of frames completed
            frames_total: Total frames to process
            elapsed_seconds: Time elapsed since stage started

        Returns:
            Estimated seconds remaining, or None if cannot calculate
        """
        if frames_completed <= 0 or frames_total <= 0:
            return None

        frames_remaining = frames_total - frames_completed

        if frames_remaining <= 0:
            return 0.0

        # Calculate current average time per frame
        avg_time_per_frame = elapsed_seconds / frames_completed

        # Record this sample for rolling average
        self._frame_processing_times.append(avg_time_per_frame)

        # Use rolling average for smoother ETA
        if len(self._frame_processing_times) > 1:
            rolling_avg = sum(self._frame_processing_times) / len(self._frame_processing_times)
            # Blend current average with rolling average (70% rolling, 30% current)
            blended_avg = 0.7 * rolling_avg + 0.3 * avg_time_per_frame
            return frames_remaining * blended_avg
        else:
            return frames_remaining * avg_time_per_frame

    def _record_frame_time(self, frame_time: float) -> None:
        """Record individual frame processing time for ETA calculation.

        Args:
            frame_time: Time taken to process a single frame in seconds
        """
        self._frame_processing_times.append(frame_time)

    def _reset_stage_timing(self, stage: str) -> None:
        """Reset timing state for a new stage.

        Args:
            stage: Stage name to reset timing for
        """
        self._stage_start_times[stage] = time.time()
        self._frame_processing_times.clear()

    def _validate_disk_space(self, video_path: Path) -> None:
        """Validate sufficient disk space before processing.

        Args:
            video_path: Path to source video

        Raises:
            DiskSpaceError: If insufficient disk space
        """
        if not self.config.enable_disk_validation:
            return

        result = validate_disk_space(
            project_dir=self.config.project_dir,
            video_path=video_path,
            scale_factor=self.config.scale_factor,
            safety_margin=self.config.disk_safety_margin,
        )

        if not result["is_valid"]:
            raise DiskSpaceError(
                f"Insufficient disk space. "
                f"Required: {result['required_gb']:.1f}GB, "
                f"Available: {result['available_gb']:.1f}GB"
            )

        if self._disk_monitor:
            self._disk_monitor.initialize()

    def _validate_gpu_memory(self, frame_resolution: Optional[Tuple[int, int]] = None) -> None:
        """Pre-flight GPU memory validation before processing begins.

        Estimates VRAM requirements based on frame resolution and model,
        and validates sufficient VRAM is available.

        Args:
            frame_resolution: Optional (width, height) tuple. If not provided,
                              uses metadata or defaults to 1920x1080.

        Raises:
            VRAMError: If insufficient VRAM available
        """
        if not self.config.enable_vram_monitoring:
            return

        # Get frame resolution
        if frame_resolution is None:
            width = self.metadata.get('width', 1920)
            height = self.metadata.get('height', 1080)
        else:
            width, height = frame_resolution

        # Get current GPU memory info
        gpu_info = get_gpu_memory_info()
        if gpu_info is None:
            logger.warning("Could not detect GPU memory - skipping pre-check")
            return

        # Estimate VRAM requirements
        required_vram_mb = self._estimate_vram_requirements(width, height)
        available_vram_mb = gpu_info['free_mb']

        logger.info(
            f"GPU memory pre-check: {required_vram_mb}MB estimated required, "
            f"{available_vram_mb}MB available"
        )

        # Check if we have enough VRAM with a safety margin
        safety_margin = 1.2  # 20% extra
        required_with_margin = int(required_vram_mb * safety_margin)

        if available_vram_mb < required_with_margin:
            # Calculate optimal tile size that will fit
            optimal_tile = calculate_optimal_tile_size(
                frame_resolution=(width, height),
                scale_factor=self.config.scale_factor,
                available_vram_mb=available_vram_mb,
                model_name=self.config.model_name,
            )

            if optimal_tile > 0:
                logger.warning(
                    f"Insufficient VRAM for full-frame processing. "
                    f"Will use tile size: {optimal_tile}"
                )
                self._current_tile_size = optimal_tile
            else:
                raise VRAMError(
                    f"Insufficient VRAM for processing. "
                    f"Required: ~{required_vram_mb}MB, Available: {available_vram_mb}MB. "
                    f"Frame resolution: {width}x{height} with {self.config.scale_factor}x upscale. "
                    f"Try reducing resolution or using a smaller model."
                )

    def _estimate_vram_requirements(self, width: int, height: int) -> int:
        """Estimate VRAM requirements for processing a frame.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Estimated VRAM requirement in MB
        """
        # Model-specific VRAM coefficients (MB per output megapixel)
        model_coefficients = {
            "realesrgan-x4plus": 450,
            "realesrgan-x4plus-anime": 400,
            "realesrgan-x2plus": 250,
            "realesr-animevideov3": 350,
        }

        coeff = model_coefficients.get(self.config.model_name, 450)

        # Calculate output resolution
        output_width = width * self.config.scale_factor
        output_height = height * self.config.scale_factor
        output_megapixels = (output_width * output_height) / 1_000_000

        # Estimate VRAM needed
        estimated_vram = int(output_megapixels * coeff)

        # Add base overhead for model loading (~500MB)
        return estimated_vram + 500

    def _check_disk_space(self) -> None:
        """Check disk space during processing.

        Raises:
            DiskSpaceError: If disk space critically low
        """
        if self._disk_monitor and self._disk_monitor.is_critical():
            status = self._disk_monitor.check()
            raise DiskSpaceError(
                f"Critical disk space: only {status['free_gb']:.2f}GB remaining"
            )

    def _get_tile_size(self) -> int:
        """Get tile size for enhancement, with fallback sequence.

        Returns:
            Current tile size to use
        """
        if self._current_tile_size is not None:
            return self._current_tile_size

        width = self.metadata.get('width', 1920)
        height = self.metadata.get('height', 1080)

        return self.config.get_tile_size_for_resolution(width, height)

    def _reduce_tile_size(self) -> bool:
        """Reduce tile size after VRAM error.

        Returns:
            True if reduced successfully, False if at minimum
        """
        width = self.metadata.get('width', 1920)
        height = self.metadata.get('height', 1080)

        sequence = get_adaptive_tile_sequence(
            frame_resolution=(width, height),
            scale_factor=self.config.scale_factor,
            starting_tile_size=self._current_tile_size,
        )

        current = self._current_tile_size or sequence[0] if sequence else 0

        # Find next smaller tile size
        for tile_size in sequence:
            if tile_size < current:
                logger.info(f"Reducing tile size from {current} to {tile_size}")
                self._current_tile_size = tile_size
                return True

        logger.warning("Already at minimum tile size")
        return False

    def download_video(self, url: str, output_path: Optional[Path] = None) -> Path:
        """Download video from URL using yt-dlp.

        Args:
            url: Video URL to download
            output_path: Optional output path (defaults to config.project_dir/video.webm)

        Returns:
            Path to downloaded video file

        Raises:
            DownloadError: If download fails
        """
        if output_path is None:
            output_path = self.config.project_dir / "video.%(ext)s"

        self._update_progress("download", 0.0)
        logger.info(f"Downloading video from {url}")

        cmd = [
            get_ytdlp_path(),
            '--format', 'bestvideo[ext=webm]+bestaudio[ext=webm]/bestvideo[ext=mkv]+bestaudio[ext=mkv]/best',
            '--merge-output-format', 'mkv',
            '--output', str(output_path),
            '--no-playlist',
            url
        ]

        retry_op = RetryableOperation(
            operation_name="download",
            max_attempts=self.config.max_retries,
            retry_on=(TransientError,),
        )

        def do_download():
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )
                logger.debug(f"yt-dlp output: {result.stdout}")
                return result
            except subprocess.CalledProcessError as e:
                error_class = classify_error(e, e.stderr)
                if issubclass(error_class, TransientError):
                    raise error_class(f"Download failed (retryable): {e.stderr}")
                raise DownloadError(f"Failed to download video: {e.stderr}")
            except subprocess.TimeoutExpired:
                raise TransientError("Download timed out")

        try:
            retry_op.execute(do_download)

            # Find the actual downloaded file
            if "%(ext)s" in str(output_path):
                base_path = str(output_path).replace(".%(ext)s", "")
                for ext in ['.webm', '.mkv', '.mp4']:
                    actual_path = Path(base_path + ext)
                    if actual_path.exists():
                        output_path = actual_path
                        break

            if not output_path.exists():
                raise DownloadError("Downloaded file not found after yt-dlp execution")

            self._update_progress("download", 1.0)
            logger.info(f"Video downloaded to {output_path}")
            return output_path

        except Exception as e:
            context = create_error_context(
                stage="download",
                operation="yt-dlp download",
                command=cmd,
                stderr=str(e),
            )
            if not isinstance(e, VideoRestorerError):
                raise DownloadError(str(e), context=context)
            raise

    def analyze_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing video metadata (framerate, resolution, codec, etc.)

        Raises:
            MetadataError: If metadata extraction fails
        """
        self._update_progress("analyze_metadata", 0.0)
        logger.info(f"Analyzing metadata for {video_path}")

        cmd = [
            get_ffprobe_path(),
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            data = json.loads(result.stdout)

            # Extract video stream information
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                None
            )

            if not video_stream:
                raise MetadataError("No video stream found in file")

            # Extract audio stream information
            audio_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
                None
            )

            # Parse framerate (can be fraction like "30000/1001")
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, denom = map(int, fps_str.split('/'))
                framerate = num / denom if denom != 0 else 30.0
            else:
                framerate = float(fps_str)

            self.metadata = {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'framerate': framerate,
                'codec': video_stream.get('codec_name', 'unknown'),
                'duration': float(data.get('format', {}).get('duration', 0)),
                'bit_rate': int(data.get('format', {}).get('bit_rate', 0)),
                'has_audio': audio_stream is not None,
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'audio_sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
            }

            # Calculate tile size for this resolution
            self._current_tile_size = self._get_tile_size()

            self._update_progress("analyze_metadata", 1.0)
            logger.info(f"Metadata: {self.metadata}")
            return self.metadata

        except subprocess.CalledProcessError as e:
            logger.error(f"Metadata extraction failed: {e.stderr}")
            raise MetadataError(f"Failed to extract metadata: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise MetadataError("Metadata extraction timed out")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse metadata: {e}")
            raise MetadataError(f"Failed to parse metadata: {e}")

    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Extract audio track as WAV with PCM encoding.

        Args:
            video_path: Path to video file
            output_path: Optional output path (defaults to config.temp_dir/audio.wav)

        Returns:
            Path to extracted audio file, or None if no audio track exists

        Raises:
            AudioExtractionError: If audio extraction fails
        """
        if not self.metadata.get('has_audio', False):
            logger.info("No audio track found in video")
            return None

        if output_path is None:
            output_path = self.config.temp_dir / "audio.wav"

        self._update_progress("extract_audio", 0.0)
        logger.info(f"Extracting audio to {output_path}")

        cmd = [
            get_ffmpeg_path(),
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s24le',  # PCM 24-bit little-endian
            '-ar', '48000',  # 48kHz sample rate
            '-y',  # Overwrite output file
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            logger.debug(f"Audio extraction output: {result.stderr}")

            self._update_progress("extract_audio", 1.0)
            logger.info(f"Audio extracted to {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            raise AudioExtractionError(f"Failed to extract audio: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioExtractionError("Audio extraction timed out")

    def extract_frames(self, video_path: Path) -> int:
        """Extract all frames from video as PNG files.

        Args:
            video_path: Path to video file

        Returns:
            Number of frames extracted

        Raises:
            FrameExtractionError: If frame extraction fails
        """
        self._update_progress("extract_frames", 0.0)
        logger.info(f"Extracting frames to {self.config.frames_dir}")

        # Check for existing checkpoint
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint and checkpoint.stage in ("extract", "enhance"):
                # Frames already extracted, verify
                existing_frames = list(self.config.frames_dir.glob("frame_*.png"))
                if len(existing_frames) == checkpoint.total_frames:
                    logger.info(f"Resuming from checkpoint: {len(existing_frames)} frames exist")
                    self._update_progress("extract_frames", 1.0)
                    return len(existing_frames)

        # Clear existing frames for fresh extraction
        if self.config.frames_dir.exists():
            shutil.rmtree(self.config.frames_dir)
        self.config.frames_dir.mkdir(parents=True)

        output_pattern = self.config.frames_dir / "frame_%08d.png"

        cmd = [
            get_ffmpeg_path(),
            '-i', str(video_path),
            '-qscale:v', '1',  # Highest quality
            '-qmin', '1',
            '-qmax', '1',
            str(output_pattern)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            logger.debug(f"Frame extraction output: {result.stderr}")

            # Count extracted frames
            frame_count = len(list(self.config.frames_dir.glob("frame_*.png")))

            if frame_count == 0:
                raise FrameExtractionError("No frames were extracted")

            # Validate frame sequence
            seq_report = validate_frame_sequence(self.config.frames_dir)
            if seq_report.has_issues:
                logger.warning(
                    f"Frame sequence issues: {seq_report.missing_count} missing, "
                    f"{len(seq_report.duplicate_frames)} duplicates"
                )

            # Create checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.create_checkpoint(
                    stage="extract",
                    total_frames=frame_count,
                    source_path=str(video_path),
                    metadata=self.metadata,
                )

            self._update_progress("extract_frames", 1.0)
            logger.info(f"Extracted {frame_count} frames")
            return frame_count

        except subprocess.CalledProcessError as e:
            logger.error(f"Frame extraction failed: {e.stderr}")
            raise FrameExtractionError(f"Failed to extract frames: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FrameExtractionError("Frame extraction timed out")

    def deduplicate_frames(self) -> DeduplicationResult:
        """Detect and extract unique frames, removing duplicates.

        For historical films (e.g., 1909 film at 18fps uploaded as 25fps),
        this detects duplicate frames added during frame rate conversion
        and extracts only unique frames for enhancement.

        This can significantly reduce processing time by only enhancing
        unique frames, then reconstructing the full sequence afterward.

        Returns:
            DeduplicationResult with frame mapping and statistics

        Raises:
            ValueError: If no frames found or deduplication fails
        """
        self._update_progress("deduplicate", 0.0)
        logger.info("Analyzing frames for duplicates...")

        # Get video FPS for analysis
        target_fps = self.metadata.get('framerate', 25.0)

        # Configure deduplicator
        dedup_config = DeduplicationConfig(
            similarity_threshold=self.config.deduplication_threshold,
            use_perceptual_hash=True,
            expected_source_fps=self.config.expected_source_fps,
        )

        deduplicator = FrameDeduplicator(dedup_config)

        # Progress callback wrapper
        def progress_cb(progress: float) -> None:
            self._update_progress("deduplicate", progress * 0.5)  # First 50% for analysis

        # Analyze frames
        result = deduplicator.analyze_frames(
            self.config.frames_dir,
            target_fps=target_fps,
            progress_callback=progress_cb,
        )

        if result.unique_frames == 0:
            raise ValueError("No unique frames detected - check deduplication threshold")

        logger.info(result.summary())

        # Extract unique frames to separate directory
        def extract_progress_cb(progress: float) -> None:
            self._update_progress("deduplicate", 0.5 + progress * 0.5)  # Second 50%

        self.config.unique_frames_dir.mkdir(parents=True, exist_ok=True)

        deduplicator.extract_unique_frames(
            self.config.frames_dir,
            self.config.unique_frames_dir,
            result=result,
            target_fps=target_fps,
            progress_callback=extract_progress_cb,
        )

        # Store result for later reconstruction
        self._dedup_result = result

        self._update_progress("deduplicate", 1.0)
        logger.info(
            f"Extracted {result.unique_frames} unique frames "
            f"(estimated original FPS: {result.estimated_original_fps:.1f})"
        )

        return result

    def reconstruct_frames(self) -> int:
        """Reconstruct full frame sequence from enhanced unique frames.

        After enhancing only unique frames, this reconstructs the full
        sequence by copying enhanced frames to their duplicate positions.

        Must be called after enhance_frames() when deduplication was used.

        Returns:
            Total number of reconstructed frames

        Raises:
            ValueError: If no deduplication result available
        """
        if self._dedup_result is None:
            raise ValueError("No deduplication result - call deduplicate_frames() first")

        self._update_progress("reconstruct", 0.0)
        logger.info("Reconstructing full frame sequence from enhanced unique frames...")

        deduplicator = FrameDeduplicator()

        # Create reconstruction output directory
        reconstructed_dir = self.config.temp_dir / "reconstructed"
        reconstructed_dir.mkdir(parents=True, exist_ok=True)

        def progress_cb(progress: float) -> None:
            self._update_progress("reconstruct", progress)

        deduplicator.reconstruct_sequence(
            self.config.enhanced_dir,
            reconstructed_dir,
            self._dedup_result,
            progress_callback=progress_cb,
        )

        # Move reconstructed frames to enhanced_dir (replacing unique-only frames)
        # First, clear enhanced_dir
        for f in self.config.enhanced_dir.glob("frame_*.png"):
            f.unlink()

        # Move reconstructed frames
        for f in reconstructed_dir.glob("frame_*.png"):
            shutil.move(str(f), self.config.enhanced_dir / f.name)

        # Clean up
        shutil.rmtree(reconstructed_dir)

        self._update_progress("reconstruct", 1.0)
        logger.info(f"Reconstructed {self._dedup_result.total_frames} frames")

        return self._dedup_result.total_frames

    def _get_ncnn_vulkan_binary(self) -> Optional[Path]:
        """Get the path to the ncnn-vulkan binary.

        Searches in order:
        1. System PATH
        2. ~/.framewright/bin/
        3. Project bin/ directory

        Returns:
            Path to binary or None if not found
        """
        ncnn_path = get_ncnn_vulkan_path()
        if ncnn_path:
            return ncnn_path

        # Fallback: check if it's in PATH directly
        binary = shutil.which("realesrgan-ncnn-vulkan")
        if binary:
            return Path(binary)

        return None

    def _get_enhancement_backend(self) -> str:
        """Determine which enhancement backend to use.

        Order of preference:
        1. FRAMEWRIGHT_BACKEND environment variable (explicit override)
        2. PyTorch if available and ncnn-vulkan is not
        3. ncnn-vulkan (default if available)
        4. PyTorch as fallback

        Returns:
            'pytorch' or 'ncnn-vulkan'
        """
        # Check environment variable for explicit override
        env_backend = os.environ.get("FRAMEWRIGHT_BACKEND", "").lower()
        if env_backend == "pytorch":
            if is_pytorch_esrgan_available():
                logger.info("Using PyTorch backend (FRAMEWRIGHT_BACKEND=pytorch)")
                return "pytorch"
            else:
                logger.warning("FRAMEWRIGHT_BACKEND=pytorch but PyTorch Real-ESRGAN not installed")
        elif env_backend == "ncnn-vulkan" or env_backend == "ncnn":
            if is_ncnn_vulkan_available():
                return "ncnn-vulkan"
            else:
                logger.warning("FRAMEWRIGHT_BACKEND=ncnn-vulkan but ncnn-vulkan not installed")

        # Auto-detect: prefer ncnn-vulkan if available, otherwise PyTorch
        if is_ncnn_vulkan_available():
            return "ncnn-vulkan"
        elif is_pytorch_esrgan_available():
            logger.info("ncnn-vulkan not found, using PyTorch backend")
            return "pytorch"

        # Neither available - will fail later with helpful error
        return "ncnn-vulkan"

    def _enhance_single_frame(
        self,
        input_path: Path,
        output_dir: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame with Real-ESRGAN.

        Supports two backends:
        - PyTorch: Uses CUDA (best for cloud/Docker with NVIDIA GPUs)
        - ncnn-vulkan: Uses Vulkan API (supports AMD/Intel/NVIDIA)

        Backend is selected automatically or via FRAMEWRIGHT_BACKEND env var.

        Args:
            input_path: Path to input frame
            output_dir: Directory for output
            tile_size: Tile size for processing

        Returns:
            Tuple of (output_path, success, error_message)

        Note:
            Includes CPU fallback detection when require_gpu=True to prevent
            runaway CPU usage that can freeze the system.
        """
        output_path = output_dir / input_path.name
        backend = self._get_enhancement_backend()

        if backend == "pytorch":
            return self._enhance_single_frame_pytorch(input_path, output_path, tile_size)
        else:
            return self._enhance_single_frame_ncnn(input_path, output_dir, tile_size)

    def _enhance_single_frame_pytorch(
        self,
        input_path: Path,
        output_path: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame using PyTorch Real-ESRGAN (CUDA).

        Args:
            input_path: Path to input frame
            output_path: Path for output frame
            tile_size: Tile size for processing (0 = auto)

        Returns:
            Tuple of (output_path, success, error_message)
        """
        if not is_pytorch_esrgan_available():
            return output_path, False, (
                "PyTorch Real-ESRGAN not available. Install with:\n"
                "  pip install realesrgan basicsr"
            )

        # Convert ncnn model name to PyTorch model name
        pytorch_model = convert_ncnn_model_name(self.config.model_name)

        config = PyTorchESRGANConfig(
            model_name=pytorch_model,
            scale_factor=self.config.scale_factor,
            tile_size=tile_size,
            gpu_id=self.config.gpu_id if self.config.gpu_id is not None else 0,
        )

        success, error = enhance_frame_pytorch(input_path, output_path, config)

        if success:
            # Validate output (validate_frame_integrity imported from .validators at top)
            validation = validate_frame_integrity(output_path)
            if not validation.is_valid:
                return output_path, False, validation.error_message

        return output_path, success, error

    def _enhance_single_frame_ncnn(
        self,
        input_path: Path,
        output_dir: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame with Real-ESRGAN using ncnn-vulkan.

        Automatically finds the ncnn-vulkan binary in common locations,
        supporting AMD, Intel, and NVIDIA GPUs via Vulkan.

        Args:
            input_path: Path to input frame
            output_dir: Directory for output
            tile_size: Tile size for processing

        Returns:
            Tuple of (output_path, success, error_message)

        Note:
            Includes CPU fallback detection when require_gpu=True to prevent
            runaway CPU usage that can freeze the system.
        """
        output_path = output_dir / input_path.name

        # Find the ncnn-vulkan binary
        ncnn_binary = self._get_ncnn_vulkan_binary()
        if not ncnn_binary:
            return output_path, False, (
                "realesrgan-ncnn-vulkan not found. "
                "Install it with: python -c \"from framewright.processors.ncnn_vulkan import install_ncnn_vulkan; install_ncnn_vulkan()\""
            )

        # Get model directory (bundled with ncnn-vulkan)
        model_dir = ncnn_binary.parent / "models"

        cmd = [
            str(ncnn_binary),
            '-i', str(input_path),
            '-o', str(output_path),
            '-n', self.config.model_name,
            '-s', str(self.config.scale_factor),
            '-f', 'png'
        ]

        # Add model path if it exists
        if model_dir.exists():
            cmd.extend(['-m', str(model_dir)])

        if tile_size > 0:
            cmd.extend(['-t', str(tile_size)])

        # GPU/CPU mode selection
        if self.config.require_gpu:
            gpu_id = self.config.gpu_id if self.config.gpu_id is not None else 0
            cmd.extend(['-g', str(gpu_id)])
            logger.debug(f"Explicit GPU selection: GPU {gpu_id}")
        else:
            # Force CPU mode with -g -1
            cmd.extend(['-g', '-1'])
            logger.debug("Forcing CPU mode (-g -1)")

        # CPU fallback indicators to detect if ncnn-vulkan falls back to CPU
        cpu_fallback_indicators = [
            "using cpu", "no vulkan device", "vulkan not found",
            "failed to create gpu instance", "cpu mode", "fallback to cpu"
        ]

        try:
            # Use CREATE_NO_WINDOW on Windows to avoid console popups
            creationflags = 0
            if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                creationflags = subprocess.CREATE_NO_WINDOW

            # Set environment variables for Vulkan compatibility
            env = os.environ.copy()
            # Fix for AMD switchable graphics causing vkEnumeratePhysicalDevices to fail
            env['DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1'] = '1'
            # Disable problematic AMD switchable graphics Vulkan layer
            env['VK_LOADER_LAYERS_DISABLE'] = 'VK_LAYER_AMD_switchable_graphics'
            # Alternative: disable all implicit layers that might interfere
            env['VK_LOADER_LAYERS_ENABLE'] = ''
            # Ensure Vulkan finds the correct GPU
            env['VK_ICD_FILENAMES'] = env.get('VK_ICD_FILENAMES', '')

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per frame
                creationflags=creationflags,
                env=env,
            )

            # Check for CPU fallback in output when GPU is required
            if self.config.require_gpu:
                combined_output = f"{result.stdout or ''} {result.stderr or ''}".lower()
                for indicator in cpu_fallback_indicators:
                    if indicator in combined_output:
                        # Delete output if created (don't trust CPU-processed results)
                        if output_path.exists():
                            output_path.unlink()
                        return output_path, False, (
                            f"CPU fallback detected: '{indicator}'. "
                            "Processing would use CPU instead of GPU, which can freeze your system. "
                            "Check GPU drivers, Vulkan installation, or set require_gpu=False."
                        )

            # Validate output
            if not output_path.exists():
                return output_path, False, "Output file not created"

            validation = validate_frame_integrity(output_path)
            if not validation.is_valid:
                return output_path, False, validation.error_message

            return output_path, True, None

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or str(e)

            # Check for CPU fallback in error output
            if self.config.require_gpu:
                error_lower = error_msg.lower()
                for indicator in cpu_fallback_indicators:
                    if indicator in error_lower:
                        return output_path, False, (
                            f"CPU fallback detected in error: '{indicator}'. "
                            "GPU processing failed and would fall back to CPU."
                        )

            error_class = classify_error(e, e.stderr)
            return output_path, False, f"{error_class.__name__}: {error_msg}"
        except subprocess.TimeoutExpired:
            if self.config.require_gpu:
                return output_path, False, (
                    "Enhancement timed out (>5 min). "
                    "This may indicate CPU fallback causing extremely slow processing."
                )
            return output_path, False, "Enhancement timed out"

    def enhance_frames(self) -> int:
        """Enhance all extracted frames using Real-ESRGAN.

        Supports checkpointing, retry logic, and parallel processing.
        Uses ThreadPoolExecutor for concurrent frame enhancement when
        parallel_frames > 1.

        When deduplication is enabled, enhances only unique frames from
        unique_frames_dir instead of all frames from frames_dir.

        Returns:
            Number of frames enhanced

        Raises:
            EnhancementError: If frame enhancement fails
        """
        # Use unique_frames_dir if deduplication was performed
        if self._dedup_result is not None and self.config.unique_frames_dir.exists():
            source_dir = self.config.unique_frames_dir
            logger.info(
                f"Using deduplicated frames: {self._dedup_result.unique_frames} unique "
                f"(from {self._dedup_result.total_frames} total)"
            )
        else:
            source_dir = self.config.frames_dir

        frames = sorted(source_dir.glob("frame_*.png"))
        total_frames = len(frames)

        if total_frames == 0:
            raise EnhancementError("No frames found to enhance")

        logger.info(f"Enhancing {total_frames} frames using {self.config.model_name}")

        # Check for resume from checkpoint
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint and checkpoint.stage == "enhance":
                frames = self.checkpoint_manager.get_unprocessed_frames(frames)
                logger.info(f"Resuming enhancement: {len(frames)} frames remaining")

        if not frames:
            logger.info("All frames already enhanced")
            return total_frames

        # Prepare output directory
        if not self.config.enhanced_dir.exists():
            self.config.enhanced_dir.mkdir(parents=True)

        # Update checkpoint stage
        if self.checkpoint_manager:
            self.checkpoint_manager.update_stage("enhance")

        # Get initial tile size
        tile_size = self._get_tile_size()
        tile_sequence = get_adaptive_tile_sequence(
            frame_resolution=(self.metadata.get('width', 1920), self.metadata.get('height', 1080)),
            scale_factor=self.config.scale_factor,
        )

        self._update_progress(
            stage="enhance_frames",
            progress=0.0,
            frames_completed=0,
            frames_total=len(frames),
        )
        error_report = ErrorReport(total_operations=len(frames))

        # Determine processing mode
        num_workers = self.config.parallel_frames
        use_parallel = num_workers > 1 and len(frames) > 1

        if use_parallel:
            logger.info(f"Using parallel processing with {num_workers} workers")
            enhanced_count = self._enhance_frames_parallel(
                frames, tile_size, tile_sequence, error_report
            )
        else:
            logger.info("Using sequential processing")
            enhanced_count = self._enhance_frames_sequential(
                frames, tile_size, tile_sequence, error_report
            )

        # Final count from directory
        final_count = len(list(self.config.enhanced_dir.glob("*.png")))

        if final_count == 0:
            raise EnhancementError("No enhanced frames were created")

        self._update_progress(
            stage="enhance_frames",
            progress=1.0,
            frames_completed=final_count,
            frames_total=final_count,
            eta_seconds=0.0,
        )
        logger.info(f"Enhanced {final_count} frames ({error_report.summary()})")

        # Store error report
        self._error_report = error_report

        return final_count

    def _enhance_frames_sequential(
        self,
        frames: List[Path],
        tile_size: int,
        tile_sequence: List[int],
        error_report: ErrorReport,
    ) -> int:
        """Enhance frames sequentially (original behavior).

        Args:
            frames: List of frame paths to enhance
            tile_size: Initial tile size
            tile_sequence: Sequence of fallback tile sizes
            error_report: Error report to update

        Returns:
            Number of successfully enhanced frames
        """
        tile_index = 0
        enhanced_count = 0
        total_frames = len(frames)

        # Reset timing for this stage
        self._reset_stage_timing("enhance_frames")

        for i, frame_path in enumerate(frames):
            frame_start_time = time.time()
            retry_count = 0
            success = False

            while retry_count <= self.config.max_retries and not success:
                output_path, success, error_msg = self._enhance_single_frame(
                    frame_path,
                    self.config.enhanced_dir,
                    tile_size
                )

                if not success:
                    # Check if VRAM error
                    if error_msg and ("vram" in error_msg.lower() or "memory" in error_msg.lower()):
                        # Try smaller tile size
                        tile_index += 1
                        if tile_index < len(tile_sequence):
                            tile_size = tile_sequence[tile_index]
                            logger.info(f"VRAM error, reducing tile size to {tile_size}")
                            retry_count += 1
                            time.sleep(self.config.retry_delay)
                            continue
                        else:
                            logger.error("Exhausted tile size options")
                            break

                    retry_count += 1
                    if retry_count <= self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** retry_count)
                        logger.warning(f"Frame {frame_path.name} failed, retrying in {delay}s...")
                        time.sleep(delay)

            # Record frame processing time for ETA calculation
            frame_time = time.time() - frame_start_time
            self._record_frame_time(frame_time)

            if success:
                enhanced_count += 1
                error_report.add_success()

                # Update checkpoint
                if self.checkpoint_manager:
                    frame_num = int(frame_path.stem.split("_")[-1])
                    self.checkpoint_manager.update_frame(
                        frame_number=frame_num,
                        input_path=frame_path,
                        output_path=output_path,
                    )
            else:
                error_report.add_error(
                    frame_path.name,
                    EnhancementError(error_msg or "Unknown error"),
                )
                if not self.config.continue_on_error:
                    raise EnhancementError(
                        f"Failed to enhance frame {frame_path.name}: {error_msg}"
                    )
                else:
                    # Copy original frame to output when enhancement fails
                    # This ensures the video can still be assembled
                    try:
                        shutil.copy2(frame_path, output_path)
                        logger.warning(
                            f"Frame {frame_path.name} enhancement failed, using original. "
                            f"Error: {error_msg}"
                        )
                        enhanced_count += 1  # Count as processed (with original)
                    except Exception as copy_err:
                        logger.error(f"Could not copy original frame: {copy_err}")

            # Update progress with frame counts for ETA calculation
            frames_completed = i + 1
            progress = frames_completed / total_frames
            self._update_progress(
                stage="enhance_frames",
                progress=progress,
                frames_completed=frames_completed,
                frames_total=total_frames,
            )

            # Check disk space periodically
            if i % 100 == 0:
                self._check_disk_space()

            # Sample VRAM usage
            if self._vram_monitor and i % 10 == 0:
                self._vram_monitor.sample()

        return enhanced_count

    def _enhance_frames_parallel(
        self,
        frames: List[Path],
        tile_size: int,
        tile_sequence: List[int],
        error_report: ErrorReport,
    ) -> int:
        """Enhance frames in parallel using ThreadPoolExecutor.

        Provides 2-4x speedup on multi-GPU systems or when GPU is
        not fully utilized.

        Args:
            frames: List of frame paths to enhance
            tile_size: Initial tile size
            tile_sequence: Sequence of fallback tile sizes
            error_report: Error report to update

        Returns:
            Number of successfully enhanced frames
        """
        import threading

        num_workers = self.config.parallel_frames
        enhanced_count = 0
        completed = 0
        total_frames = len(frames)
        lock = threading.Lock()
        current_tile_size = tile_size

        # Reset timing for this stage
        self._reset_stage_timing("enhance_frames")

        def process_frame(frame_path: Path) -> Tuple[Path, bool, Optional[str], float]:
            """Process a single frame with retry logic.

            Returns:
                Tuple of (output_path, success, error_message, processing_time)
            """
            nonlocal current_tile_size
            frame_start = time.time()

            retry_count = 0
            while retry_count <= self.config.max_retries:
                output_path, success, error_msg = self._enhance_single_frame(
                    frame_path,
                    self.config.enhanced_dir,
                    current_tile_size
                )

                if success:
                    return output_path, True, None, time.time() - frame_start

                # Check if VRAM error - reduce tile size for all workers
                if error_msg and ("vram" in error_msg.lower() or "memory" in error_msg.lower()):
                    with lock:
                        for smaller_tile in tile_sequence:
                            if smaller_tile < current_tile_size:
                                logger.info(f"VRAM error, reducing tile size to {smaller_tile}")
                                current_tile_size = smaller_tile
                                break
                        else:
                            return output_path, False, "Exhausted tile size options", time.time() - frame_start

                retry_count += 1
                if retry_count <= self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry_count)
                    time.sleep(delay)

            return output_path, False, error_msg, time.time() - frame_start

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all frames for processing
            future_to_frame = {
                executor.submit(process_frame, frame): frame
                for frame in frames
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_path = future_to_frame[future]

                try:
                    output_path, success, error_msg, frame_time = future.result()

                    with lock:
                        completed += 1

                        # Record frame processing time for ETA
                        self._record_frame_time(frame_time)

                        progress = completed / total_frames
                        self._update_progress(
                            stage="enhance_frames",
                            progress=progress,
                            frames_completed=completed,
                            frames_total=total_frames,
                        )

                        if success:
                            enhanced_count += 1
                            error_report.add_success()

                            # Update checkpoint
                            if self.checkpoint_manager:
                                frame_num = int(frame_path.stem.split("_")[-1])
                                self.checkpoint_manager.update_frame(
                                    frame_number=frame_num,
                                    input_path=frame_path,
                                    output_path=output_path,
                                )
                        else:
                            error_report.add_error(
                                frame_path.name,
                                EnhancementError(error_msg or "Unknown error"),
                            )
                            if not self.config.continue_on_error:
                                # Cancel remaining futures
                                for f in future_to_frame:
                                    f.cancel()
                                raise EnhancementError(
                                    f"Failed to enhance frame {frame_path.name}: {error_msg}"
                                )
                            else:
                                # Copy original frame to output when enhancement fails
                                try:
                                    shutil.copy2(frame_path, output_path)
                                    logger.warning(
                                        f"Frame {frame_path.name} enhancement failed, using original. "
                                        f"Error: {error_msg}"
                                    )
                                    enhanced_count += 1  # Count as processed
                                except Exception as copy_err:
                                    logger.error(f"Could not copy original frame: {copy_err}")

                        # Check disk space periodically
                        if completed % 100 == 0:
                            self._check_disk_space()

                        # Sample VRAM usage
                        if self._vram_monitor and completed % 10 == 0:
                            self._vram_monitor.sample()

                except Exception as e:
                    if not isinstance(e, EnhancementError):
                        logger.error(f"Unexpected error processing {frame_path.name}: {e}")
                        error_report.add_error(frame_path.name, e)
                    else:
                        raise

        return enhanced_count

    def auto_enhance_frames(
        self,
        source_dir: Optional[Path] = None,
    ) -> AdaptiveEnhanceResult:
        """Apply automatic enhancements based on content analysis.

        Automatically detects and applies:
        - Defect repairs (scratches, dust, grain)
        - Face restoration (if faces detected)
        - Content-specific optimizations

        Must be called after enhance_frames() for best results.

        Args:
            source_dir: Directory with frames to enhance (default: enhanced_dir)

        Returns:
            AdaptiveEnhanceResult with processing details
        """
        if source_dir is None:
            source_dir = self.config.enhanced_dir

        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return AdaptiveEnhanceResult()

        self._update_progress("auto_enhance", 0.0)
        logger.info("Starting automatic enhancement pipeline...")

        # Create enhancer with config settings
        enhancer = AdaptiveEnhancer(
            enable_analysis=self.config.auto_detect_content,
            enable_defect_repair=self.config.auto_defect_repair,
            enable_face_restoration=self.config.auto_face_restore,
            scratch_sensitivity=self.config.scratch_sensitivity,
            dust_sensitivity=self.config.dust_sensitivity,
            grain_reduction=self.config.grain_reduction,
        )

        # Create output directory
        auto_enhanced_dir = self.config.temp_dir / "auto_enhanced"
        auto_enhanced_dir.mkdir(parents=True, exist_ok=True)

        def progress_cb(stage: str, progress: float):
            # Map sub-stages to overall progress
            stage_weights = {
                "analysis": 0.1,
                "defect_repair": 0.4,
                "face_restoration": 0.4,
                "complete": 0.1,
            }
            base = sum(stage_weights.get(s, 0) for s in ["analysis", "defect_repair", "face_restoration"]
                      if list(stage_weights.keys()).index(s) < list(stage_weights.keys()).index(stage))
            weight = stage_weights.get(stage, 0.1)
            overall = base + (progress * weight)
            self._update_progress("auto_enhance", overall)

        result = enhancer.process_frames(
            input_dir=source_dir,
            output_dir=auto_enhanced_dir,
            analysis=self._video_analysis,
            progress_callback=progress_cb,
        )

        # Move final frames back to enhanced_dir
        final_dir = auto_enhanced_dir / "final"
        if final_dir.exists():
            import shutil
            # Replace enhanced frames with auto-enhanced ones
            for frame in final_dir.glob("*.png"):
                shutil.copy(frame, self.config.enhanced_dir / frame.name)
            shutil.rmtree(auto_enhanced_dir, ignore_errors=True)

        self._enhance_result = result
        self._update_progress("auto_enhance", 1.0)

        logger.info(f"Auto-enhancement complete: {result.summary()}")
        return result

    def analyze_video(self, video_path: Path) -> VideoAnalysis:
        """Pre-analyze video for optimal restoration settings.

        Runs automatic analysis to detect:
        - Content type (faces, animation, landscapes, etc.)
        - Degradation type and severity
        - Recommended settings

        Args:
            video_path: Path to video file

        Returns:
            VideoAnalysis with detection results and recommendations
        """
        self._update_progress("analyze_video", 0.0)
        logger.info("Running pre-scan video analysis...")

        analyzer = FrameAnalyzer(
            sample_rate=100,
            max_samples=50,
            enable_face_detection=self.config.auto_face_restore,
        )

        analysis = analyzer.analyze_video(video_path)
        self._video_analysis = analysis

        self._update_progress("analyze_video", 1.0)

        logger.info(
            f"Analysis complete: content={analysis.primary_content.name}, "
            f"degradation={analysis.degradation_severity}, "
            f"recommended_scale={analysis.recommended_scale}x"
        )

        return analysis

    def apply_analysis_recommendations(self) -> None:
        """Apply recommendations from video analysis to config.

        Updates config settings based on auto-detected content and
        degradation. Must be called after analyze_video().
        """
        if not self._video_analysis:
            logger.warning("No video analysis available, skipping recommendations")
            return

        analysis = self._video_analysis

        # Apply scale and model recommendations
        logger.info(f"Applying analysis recommendations: scale={analysis.recommended_scale}x, "
                   f"model={analysis.recommended_model}")

        # Note: scale_factor and model_name are validated, so we need to
        # check if the recommended values are valid before applying
        if analysis.recommended_scale in (2, 4):
            # Would need to update config, but scale_factor is immutable after init
            # Log the recommendation instead
            if self.config.scale_factor != analysis.recommended_scale:
                logger.info(
                    f"Recommended scale: {analysis.recommended_scale}x "
                    f"(current: {self.config.scale_factor}x)"
                )

        # Update RIFE target if recommended
        if analysis.recommended_target_fps and not self.config.target_fps:
            logger.info(
                f"Recommended target FPS for RIFE: {analysis.recommended_target_fps}"
            )

    def interpolate_frames(
        self,
        source_dir: Optional[Path] = None,
        target_fps: Optional[float] = None,
        source_fps: Optional[float] = None,
    ) -> Tuple[Path, float]:
        """Interpolate frames using RIFE to increase frame rate.

        Must be called after enhance_frames(). Requires enable_interpolation=True
        in config or explicit target_fps parameter.

        Args:
            source_dir: Directory with frames to interpolate (default: enhanced_dir)
            target_fps: Target frame rate (default: from config or 2x source)
            source_fps: Source frame rate (default: from metadata). Override this
                       when interpolating deduplicated frames to specify the
                       detected original FPS (e.g., 18fps for 1909 film).

        Returns:
            Tuple of (output_directory, actual_fps)

        Raises:
            InterpolationError: If interpolation fails
        """
        if source_dir is None:
            source_dir = self.config.enhanced_dir

        if not source_dir.exists():
            raise InterpolationError(f"Source directory not found: {source_dir}")

        # Get source FPS - use provided value or fall back to metadata
        if source_fps is None:
            source_fps = self.metadata.get('framerate', 24.0)

        # Determine target FPS
        if target_fps is None:
            target_fps = self.config.target_fps
        if target_fps is None:
            # Default to 2x source fps
            target_fps = source_fps * 2
            logger.info(f"No target_fps specified, defaulting to 2x source: {target_fps}fps")

        if target_fps <= source_fps:
            logger.warning(
                f"Target FPS ({target_fps}) <= source FPS ({source_fps}), "
                "skipping interpolation"
            )
            return source_dir, source_fps

        self._update_progress("interpolate_frames", 0.0)
        logger.info(
            f"Interpolating frames: {source_fps}fps -> {target_fps}fps "
            f"using {self.config.rife_model}"
        )

        # Create interpolator
        interpolator = FrameInterpolator(
            model=self.config.rife_model,
            gpu_id=self.config.rife_gpu_id,
        )

        # Create output directory
        self.config.interpolated_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_dir, actual_fps = interpolator.interpolate_to_fps(
                input_dir=source_dir,
                output_dir=self.config.interpolated_dir,
                source_fps=source_fps,
                target_fps=int(target_fps),  # RIFE expects integer fps
                progress_callback=lambda p: self._update_progress("interpolate_frames", p),
            )

            frame_count = len(list(output_dir.glob("*.png")))
            logger.info(
                f"Interpolation complete: {frame_count} frames at {actual_fps}fps"
            )

            # Store actual fps for reassembly
            self.metadata['interpolated_fps'] = actual_fps

            self._update_progress("interpolate_frames", 1.0)
            return output_dir, actual_fps

        except Exception as e:
            logger.error(f"Frame interpolation failed: {e}")
            raise InterpolationError(f"Failed to interpolate frames: {e}")

    def preview_frames(
        self,
        frames_dir: Optional[Path] = None,
        sample_count: int = 5,
    ) -> Dict[str, Any]:
        """Generate preview information for user to inspect before reassembly.

        Args:
            frames_dir: Directory containing frames to preview
            sample_count: Number of sample frames to include

        Returns:
            Dictionary with preview information and sample frame paths
        """
        if frames_dir is None:
            # Use interpolated if available, otherwise enhanced
            if self.config.interpolated_dir.exists() and \
               list(self.config.interpolated_dir.glob("*.png")):
                frames_dir = self.config.interpolated_dir
            else:
                frames_dir = self.config.enhanced_dir

        frames = sorted(frames_dir.glob("*.png"))
        total_frames = len(frames)

        if total_frames == 0:
            return {
                "success": False,
                "error": "No frames found for preview",
                "frames_dir": str(frames_dir),
            }

        # Select evenly spaced sample frames
        sample_indices = [
            int(i * (total_frames - 1) / (sample_count - 1))
            for i in range(min(sample_count, total_frames))
        ]
        sample_frames = [frames[i] for i in sample_indices]

        # Calculate preview info
        source_fps = self.metadata.get('framerate', 24.0)
        output_fps = self.metadata.get('interpolated_fps', source_fps)
        duration = total_frames / output_fps if output_fps > 0 else 0

        preview_info = {
            "success": True,
            "frames_dir": str(frames_dir),
            "total_frames": total_frames,
            "sample_frames": [str(f) for f in sample_frames],
            "source_fps": source_fps,
            "output_fps": output_fps,
            "estimated_duration": f"{duration:.2f}s",
            "resolution": f"{self.metadata.get('width', 0) * self.config.scale_factor}x"
                         f"{self.metadata.get('height', 0) * self.config.scale_factor}",
            "interpolation_applied": self.config.interpolated_dir.exists() and
                                    bool(list(self.config.interpolated_dir.glob("*.png"))),
        }

        logger.info(
            f"Preview ready: {total_frames} frames, {output_fps}fps, "
            f"{preview_info['resolution']}"
        )

        return preview_info

    def reassemble_video(
        self,
        audio_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        frames_dir: Optional[Path] = None,
    ) -> Path:
        """Reassemble video from enhanced/interpolated frames with audio.

        Args:
            audio_path: Path to audio file (optional)
            output_path: Output video path (defaults to config.project_dir/output.mkv)
            frames_dir: Directory with frames (default: interpolated if exists, else enhanced)

        Returns:
            Path to output video file

        Raises:
            ReassemblyError: If video reassembly fails
        """
        if output_path is None:
            output_path = self.config.project_dir / f"output.{self.config.output_format}"

        # Determine which frames to use
        if frames_dir is None:
            # Prefer interpolated frames if they exist
            if self.config.interpolated_dir.exists() and \
               list(self.config.interpolated_dir.glob("*.png")):
                frames_dir = self.config.interpolated_dir
                logger.info("Using interpolated frames for reassembly")
            else:
                frames_dir = self.config.enhanced_dir
                logger.info("Using enhanced frames for reassembly")

        self._update_progress("reassemble_video", 0.0)
        logger.info(f"Reassembling video to {output_path}")

        # Verify we have frames
        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            raise ReassemblyError(f"No frames found in {frames_dir} for reassembly")

        # Validate frame sequence
        seq_report = validate_frame_sequence(frames_dir)
        if seq_report.missing_count > 0:
            logger.warning(f"Missing {seq_report.missing_count} frames in sequence")

        # Get framerate - use interpolated fps if available, otherwise source fps
        framerate = self.metadata.get('interpolated_fps') or self.metadata.get('framerate', 30)
        logger.info(f"Output framerate: {framerate}fps")

        # Base ffmpeg command for video encoding
        input_pattern = frames_dir / "frame_%08d.png"

        cmd = [
            get_ffmpeg_path(),
            '-framerate', str(framerate),
            '-i', str(input_pattern),
        ]

        # Add audio if available
        if audio_path and audio_path.exists():
            cmd.extend([
                '-i', str(audio_path),
                '-c:a', 'flac',  # FLAC audio codec
            ])

        # Video encoding settings - use best available codec with fallback
        codec, pix_fmt = get_best_video_codec('libx265')
        logger.info(f"Using video codec: {codec} with pixel format: {pix_fmt}")

        cmd.extend([
            '-c:v', codec,
            '-crf', str(self.config.crf),  # Quality
            '-preset', self.config.preset,  # Encoding preset
            '-pix_fmt', pix_fmt,
            '-y',  # Overwrite output
            str(output_path)
        ])

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            logger.debug(f"Video reassembly output: {result.stderr}")

            if not output_path.exists():
                raise ReassemblyError("Output video file was not created")

            # Update checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.update_stage("reassemble")

            self._update_progress("reassemble_video", 1.0)
            logger.info(f"Video reassembled to {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Video reassembly failed: {e.stderr}")
            raise ReassemblyError(f"Failed to reassemble video: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise ReassemblyError("Video reassembly timed out")

    def validate_output(self, output_path: Path) -> bool:
        """Validate the output video quality.

        Args:
            output_path: Path to output video

        Returns:
            True if validation passes
        """
        if not self.config.enable_validation:
            return True

        logger.info("Validating output quality...")

        # Check temporal consistency
        temporal_report = validate_temporal_consistency(
            self.config.enhanced_dir,
            sample_rate=10,
        )

        if temporal_report.flickering_detected:
            logger.warning(
                f"Flickering detected in {len(temporal_report.flicker_frames)} frames "
                f"(severity: {temporal_report.severity})"
            )

        return True

    def restore_video(
        self,
        source: str,
        output_path: Optional[Path] = None,
        cleanup: bool = True,
        resume: bool = True,
        enable_rife: Optional[bool] = None,
        target_fps: Optional[float] = None,
        enable_auto_enhance: Optional[bool] = None,
        preview_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Path:
        """Complete video restoration pipeline.

        Args:
            source: Video URL or local file path
            output_path: Optional output path for final video
            cleanup: Whether to remove temporary files after completion
            resume: Whether to resume from checkpoint if available
            enable_rife: Enable RIFE interpolation (None = use config setting)
            target_fps: Target frame rate for RIFE (None = use config or auto)
            enable_auto_enhance: Enable auto-enhancement (None = use config setting)
            preview_callback: Optional callback for preview approval.
                             Receives preview dict, returns True to proceed or False to abort.
                             If None and preview enabled, logs preview info and continues.

        Returns:
            Path to restored video file

        Raises:
            VideoRestorerError: If any step of the pipeline fails
        """
        try:
            # Create necessary directories
            self.config.create_directories()

            # Pre-flight GPU validation (prevents CPU fallback)
            self._validate_gpu_available()

            # Determine processing options
            use_rife = enable_rife if enable_rife is not None else self.config.enable_interpolation
            rife_target_fps = target_fps if target_fps is not None else self.config.target_fps
            use_auto_enhance = enable_auto_enhance if enable_auto_enhance is not None else self.config.enable_auto_enhance

            # Check for existing checkpoint
            checkpoint: Optional[PipelineCheckpoint] = None
            if resume and self.checkpoint_manager:
                checkpoint = self.checkpoint_manager.load_checkpoint()
                if checkpoint:
                    logger.info(
                        f"Found checkpoint at stage '{checkpoint.stage}' "
                        f"({checkpoint.last_completed_frame}/{checkpoint.total_frames} frames)"
                    )
                    # Restore metadata from checkpoint
                    if checkpoint.metadata:
                        self.metadata = checkpoint.metadata

            # Step 1: Download or copy video
            source_path = Path(source)
            if source_path.exists():
                logger.info(f"Using local video file: {source}")
                video_path = source_path
            else:
                # Skip download if checkpoint exists with source
                if checkpoint and checkpoint.source_path:
                    existing_video = Path(checkpoint.source_path)
                    if existing_video.exists():
                        video_path = existing_video
                        logger.info(f"Using video from checkpoint: {video_path}")
                    else:
                        logger.info(f"Downloading video from URL: {source}")
                        video_path = self.download_video(source)
                else:
                    logger.info(f"Downloading video from URL: {source}")
                    video_path = self.download_video(source)

            # Step 2: Analyze metadata (unless resuming with existing metadata)
            if not self.metadata:
                self.analyze_metadata(video_path)

            # Step 2b: Pre-scan analysis for auto-enhancement
            if use_auto_enhance and not self._video_analysis:
                self.analyze_video(video_path)
                self.apply_analysis_recommendations()

            # Log detected frame rate
            source_fps = self.metadata.get('framerate', 24.0)
            logger.info(f"Detected source frame rate: {source_fps}fps")

            # Step 3: Validate disk space
            self._validate_disk_space(video_path)

            # Step 4: Extract audio
            audio_path = None
            if not checkpoint or checkpoint.stage == "download":
                audio_path = self.extract_audio(video_path)
            else:
                # Check for existing audio
                existing_audio = self.config.temp_dir / "audio.wav"
                if existing_audio.exists():
                    audio_path = existing_audio

            # Step 5: Extract frames
            frame_count = self.extract_frames(video_path)
            logger.info(f"Processing {frame_count} frames")

            # Step 5b: Deduplicate frames (for old films with artificial FPS padding)
            if self.config.enable_deduplication:
                dedup_result = self.deduplicate_frames()
                logger.info(
                    f"Deduplication: {dedup_result.unique_frames}/{frame_count} unique frames "
                    f"(estimated original FPS: {dedup_result.estimated_original_fps:.1f})"
                )

            # Step 6: Enhance frames (Real-ESRGAN)
            self.enhance_frames()

            # Step 6b: Auto-enhancement (defect repair, face restore)
            if use_auto_enhance:
                logger.info("Applying auto-enhancement pipeline...")
                enhance_result = self.auto_enhance_frames()
                logger.info(f"Auto-enhancement stages: {', '.join(enhance_result.stages_applied)}")

            # Step 7: Frame interpolation or reconstruction
            frames_for_reassembly = self.config.enhanced_dir

            # When deduplication is used, prefer RIFE for smooth motion
            if self.config.enable_deduplication and self._dedup_result is not None:
                if use_rife:
                    # RIFE interpolation: unique enhanced frames → smooth target FPS
                    # This gives much better results than duplicating frames back
                    original_fps = self._dedup_result.estimated_original_fps
                    target = rife_target_fps or self.metadata.get('framerate', 25.0)

                    logger.info(
                        f"RIFE interpolation: {original_fps:.1f}fps → {target}fps "
                        f"(smooth motion from {self._dedup_result.unique_frames} unique frames)"
                    )

                    try:
                        # Set source FPS to the detected original FPS for proper interpolation
                        frames_for_reassembly, actual_fps = self.interpolate_frames(
                            target_fps=target,
                            source_fps=original_fps,
                        )
                        logger.info(f"RIFE interpolation complete: {actual_fps}fps with smooth motion")
                    except InterpolationError as e:
                        logger.warning(f"RIFE interpolation failed: {e}")
                        logger.warning("Falling back to frame reconstruction (duplicate-based)")
                        self.reconstruct_frames()
                        frames_for_reassembly = self.config.enhanced_dir
                else:
                    # No RIFE: reconstruct by duplicating frames back
                    logger.info("Reconstructing frames (no RIFE - consider enabling for smoother motion)")
                    self.reconstruct_frames()
                    frames_for_reassembly = self.config.enhanced_dir

            elif use_rife:
                # Standard RIFE interpolation (no deduplication)
                logger.info("RIFE interpolation enabled")
                try:
                    frames_for_reassembly, actual_fps = self.interpolate_frames(
                        target_fps=rife_target_fps
                    )
                    logger.info(f"Interpolation complete: output at {actual_fps}fps")
                except InterpolationError as e:
                    logger.warning(f"RIFE interpolation failed: {e}")
                    logger.warning("Continuing with enhanced frames (no interpolation)")
                    frames_for_reassembly = self.config.enhanced_dir

            # Step 8: Preview before reassembly (if callback provided)
            preview_info = self.preview_frames(frames_for_reassembly)
            if preview_info["success"]:
                logger.info(
                    f"\n{'='*60}\n"
                    f"PREVIEW: Ready to reassemble video\n"
                    f"  Frames: {preview_info['total_frames']}\n"
                    f"  Resolution: {preview_info['resolution']}\n"
                    f"  Frame Rate: {preview_info['output_fps']}fps\n"
                    f"  Duration: {preview_info['estimated_duration']}\n"
                    f"  RIFE applied: {preview_info['interpolation_applied']}\n"
                    f"  Sample frames: {preview_info['frames_dir']}\n"
                    f"{'='*60}"
                )

                if preview_callback:
                    proceed = preview_callback(preview_info)
                    if not proceed:
                        logger.info("User cancelled reassembly via preview callback")
                        raise FatalError("Reassembly cancelled by user")

            # Step 9: Reassemble video
            result_path = self.reassemble_video(
                audio_path=audio_path,
                output_path=output_path,
                frames_dir=frames_for_reassembly,
            )

            # Step 10: Validate output
            self.validate_output(result_path)

            # Step 11: Mark complete
            if self.checkpoint_manager:
                self.checkpoint_manager.complete()

            # Step 12: Cleanup if requested
            if cleanup:
                logger.info("Cleaning up temporary files")
                self.config.cleanup_temp()
                if self.checkpoint_manager:
                    self.checkpoint_manager.clear_checkpoint()

            logger.info(f"Video restoration complete: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"Video restoration failed: {e}")
            # Save checkpoint on failure for resume
            if self.checkpoint_manager:
                self.checkpoint_manager.force_save()
            raise

    def get_error_report(self) -> ErrorReport:
        """Get the error report from the last enhancement run.

        Returns:
            ErrorReport with details of any failures
        """
        return self._error_report

    def get_vram_statistics(self) -> Optional[Dict[str, float]]:
        """Get VRAM usage statistics from the monitoring.

        Returns:
            Dictionary with VRAM statistics or None if monitoring disabled
        """
        if self._vram_monitor:
            return self._vram_monitor.get_statistics()
        return None

    def restore_video_streaming(
        self,
        source: str,
        output_dir: Optional[Path] = None,
        chunk_duration: float = 300.0,
        on_chunk_ready: Optional[Callable[[ChunkInfo], None]] = None,
        cleanup: bool = True,
        enable_rife: Optional[bool] = None,
        target_fps: Optional[float] = None,
    ) -> Path:
        """Restore video with streaming output for very long videos.

        Processes and outputs video in chunks, allowing:
        - Preview chunks while rest of video processes
        - Recover from crashes with partial output
        - Handle videos larger than available disk space

        Args:
            source: Video URL or local file path
            output_dir: Output directory for chunks and final video
            chunk_duration: Duration of each chunk in seconds (default: 5 min)
            on_chunk_ready: Callback when a chunk is ready for preview
            cleanup: Whether to remove temporary files after completion
            enable_rife: Enable RIFE interpolation (None = use config)
            target_fps: Target frame rate for RIFE

        Returns:
            Path to final merged video file

        Example:
            >>> def on_chunk(chunk):
            ...     print(f"Chunk ready: {chunk.output_path}")
            ...     # User can watch this immediately!
            >>> result = restorer.restore_video_streaming(
            ...     "video.mp4",
            ...     chunk_duration=300,  # 5 min chunks
            ...     on_chunk_ready=on_chunk,
            ... )
        """
        try:
            # Create directories
            self.config.create_directories()
            if output_dir is None:
                output_dir = self.config.get_output_dir()
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine processing options
            use_rife = enable_rife if enable_rife is not None else self.config.enable_interpolation
            rife_target_fps = target_fps if target_fps is not None else self.config.target_fps

            # Step 1: Download or get video
            source_path = Path(source)
            if source_path.exists():
                video_path = source_path
            else:
                video_path = self.download_video(source)

            # Step 2: Analyze metadata
            self.analyze_metadata(video_path)

            # Step 3: Validate disk space
            self._validate_disk_space(video_path)

            # Step 4: Extract audio
            audio_path = self.extract_audio(video_path)

            # Step 5: Extract frames
            frame_count = self.extract_frames(video_path)
            logger.info(f"Processing {frame_count} frames in streaming mode")

            # Step 6: Enhance frames
            self.enhance_frames()

            # Step 7: Frame interpolation if enabled
            frames_dir = self.config.enhanced_dir
            output_fps = self.metadata.get('framerate', 30.0)

            if use_rife:
                try:
                    frames_dir, output_fps = self.interpolate_frames(
                        target_fps=rife_target_fps
                    )
                except InterpolationError as e:
                    logger.warning(f"RIFE failed, using enhanced frames: {e}")
                    frames_dir = self.config.enhanced_dir

            # Step 8: Stream processing - output chunks as they complete
            streaming_config = StreamingConfig(
                chunk_duration_seconds=chunk_duration,
                output_format=self.config.output_format,
                crf=self.config.crf,
                preset=self.config.preset,
            )

            processor = StreamingProcessor(
                streaming_config=streaming_config,
                framerate=output_fps,
                audio_path=audio_path,
            )

            if on_chunk_ready:
                processor.on_chunk_complete(on_chunk_ready)

            # Process all chunks
            chunks = list(processor.process_streaming(
                frames_dir=frames_dir,
                output_dir=output_dir,
                progress_callback=lambda p, m: self._update_progress("streaming", p),
            ))

            logger.info(f"Streaming complete: {len(chunks)} chunks ready")

            # Step 9: Merge chunks into final video
            final_path = output_dir / f"restored_video.{self.config.output_format}"
            result_path = processor.merge_chunks(final_path)

            # Step 10: Cleanup if requested
            if cleanup:
                logger.info("Cleaning up temporary files")
                self.config.cleanup_temp()

            logger.info(f"Streaming restoration complete: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"Streaming restoration failed: {e}")
            raise

    async def restore_video_async(
        self,
        source: str,
        output_path: Optional[Path] = None,
        cleanup: bool = True,
    ) -> Path:
        """Restore video with async I/O for better performance.

        Uses asyncio for non-blocking I/O operations:
        - Async downloads while other work continues
        - Concurrent file reads/writes
        - Better CPU/GPU utilization

        Args:
            source: Video URL or local file path
            output_path: Optional output path for final video
            cleanup: Whether to remove temporary files

        Returns:
            Path to restored video file

        Example:
            >>> result = await restorer.restore_video_async(
            ...     "https://example.com/video.mp4"
            ... )
        """
        import asyncio
        from .utils.async_io import AsyncDownloader, AsyncSubprocess

        try:
            # Create directories
            self.config.create_directories()

            # Step 1: Download asynchronously if URL
            source_path = Path(source)
            if source_path.exists():
                video_path = source_path
            else:
                self._update_progress("download", 0.0)
                async with AsyncDownloader() as dl:
                    result = await dl.download(
                        url=source,
                        output_dir=self.config.project_dir,
                        filename="video",
                    )
                    if not result.success:
                        raise DownloadError(f"Async download failed: {result.error}")
                    video_path = result.path
                self._update_progress("download", 1.0)

            # Step 2-onwards: Run standard pipeline
            # (These are CPU/GPU bound, not I/O bound, so sync is fine)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Default executor
                lambda: self.restore_video(
                    source=str(video_path),
                    output_path=output_path,
                    cleanup=cleanup,
                )
            )

            return result

        except Exception as e:
            logger.error(f"Async restoration failed: {e}")
            raise
