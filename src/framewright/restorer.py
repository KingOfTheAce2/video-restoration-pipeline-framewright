"""Video restoration module using Real-ESRGAN for frame enhancement.

Includes robust error handling, checkpointing, and quality validation.
"""
import json
import logging
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple

from .config import Config, RestoreOptions
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
    calculate_optimal_tile_size,
    get_adaptive_tile_sequence,
    VRAMMonitor,
)
from .utils.disk import (
    validate_disk_space,
    DiskSpaceMonitor,
)
from .utils.dependencies import validate_all_dependencies
from .processors.interpolation import FrameInterpolator, InterpolationError
from .processors.analyzer import FrameAnalyzer, VideoAnalysis
from .processors.adaptive_enhance import AdaptiveEnhancer, AdaptiveEnhanceResult


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
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """Initialize VideoRestorer with configuration.

        Args:
            config: Configuration object
            progress_callback: Optional callback function(stage: str, progress: float)
                              where progress is 0.0 to 1.0
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

    def _update_progress(self, stage: str, progress: float) -> None:
        """Update progress via callback if provided.

        Args:
            stage: Current processing stage
            progress: Progress value between 0.0 and 1.0
        """
        if self.progress_callback:
            self.progress_callback(stage, progress)
        logger.info(f"{stage}: {progress * 100:.1f}% complete")

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
            'yt-dlp',
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
            'ffprobe',
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
            'ffmpeg',
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
            'ffmpeg',
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

    def _enhance_single_frame(
        self,
        input_path: Path,
        output_dir: Path,
        tile_size: int
    ) -> Tuple[Path, bool, Optional[str]]:
        """Enhance a single frame with Real-ESRGAN.

        Args:
            input_path: Path to input frame
            output_dir: Directory for output
            tile_size: Tile size for processing

        Returns:
            Tuple of (output_path, success, error_message)
        """
        output_path = output_dir / input_path.name

        cmd = [
            'realesrgan-ncnn-vulkan',
            '-i', str(input_path),
            '-o', str(output_path),
            '-n', self.config.model_name,
            '-s', str(self.config.scale_factor),
            '-f', 'png'
        ]

        if tile_size > 0:
            cmd.extend(['-t', str(tile_size)])

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per frame
            )

            # Validate output
            if not output_path.exists():
                return output_path, False, "Output file not created"

            validation = validate_frame_integrity(output_path)
            if not validation.is_valid:
                return output_path, False, validation.error_message

            return output_path, True, None

        except subprocess.CalledProcessError as e:
            error_class = classify_error(e, e.stderr)
            return output_path, False, f"{error_class.__name__}: {e.stderr}"
        except subprocess.TimeoutExpired:
            return output_path, False, "Enhancement timed out"

    def enhance_frames(self) -> int:
        """Enhance all extracted frames using Real-ESRGAN.

        Supports checkpointing, retry logic, and parallel processing.

        Returns:
            Number of frames enhanced

        Raises:
            EnhancementError: If frame enhancement fails
        """
        frames = sorted(self.config.frames_dir.glob("frame_*.png"))
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
        tile_index = 0

        self._update_progress("enhance_frames", 0.0)
        enhanced_count = 0
        error_report = ErrorReport(total_operations=len(frames))

        # Process frames
        for i, frame_path in enumerate(frames):
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

            # Update progress
            progress = (i + 1) / len(frames)
            self._update_progress("enhance_frames", progress)

            # Check disk space periodically
            if i % 100 == 0:
                self._check_disk_space()

            # Sample VRAM usage
            if self._vram_monitor and i % 10 == 0:
                self._vram_monitor.sample()

        # Final count from directory
        final_count = len(list(self.config.enhanced_dir.glob("*.png")))

        if final_count == 0:
            raise EnhancementError("No enhanced frames were created")

        self._update_progress("enhance_frames", 1.0)
        logger.info(f"Enhanced {final_count} frames ({error_report.summary()})")

        # Store error report
        self._error_report = error_report

        return final_count

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
    ) -> Tuple[Path, float]:
        """Interpolate frames using RIFE to increase frame rate.

        Must be called after enhance_frames(). Requires enable_interpolation=True
        in config or explicit target_fps parameter.

        Args:
            source_dir: Directory with frames to interpolate (default: enhanced_dir)
            target_fps: Target frame rate (default: from config or 2x source)

        Returns:
            Tuple of (output_directory, actual_fps)

        Raises:
            InterpolationError: If interpolation fails
        """
        if source_dir is None:
            source_dir = self.config.enhanced_dir

        if not source_dir.exists():
            raise InterpolationError(f"Source directory not found: {source_dir}")

        # Get source FPS from metadata (auto-detected earlier)
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
            'ffmpeg',
            '-framerate', str(framerate),
            '-i', str(input_pattern),
        ]

        # Add audio if available
        if audio_path and audio_path.exists():
            cmd.extend([
                '-i', str(audio_path),
                '-c:a', 'flac',  # FLAC audio codec
            ])

        # Video encoding settings
        cmd.extend([
            '-c:v', 'libx265',  # x265 codec
            '-crf', str(self.config.crf),  # Quality
            '-preset', self.config.preset,  # Encoding preset
            '-pix_fmt', 'yuv420p10le',  # 10-bit color
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

            # Step 6: Enhance frames (Real-ESRGAN)
            self.enhance_frames()

            # Step 6b: Auto-enhancement (defect repair, face restore)
            if use_auto_enhance:
                logger.info("Applying auto-enhancement pipeline...")
                enhance_result = self.auto_enhance_frames()
                logger.info(f"Auto-enhancement stages: {', '.join(enhance_result.stages_applied)}")

            # Step 7: Frame interpolation (RIFE) - if enabled
            frames_for_reassembly = self.config.enhanced_dir
            if use_rife:
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
