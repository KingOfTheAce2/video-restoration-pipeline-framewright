"""Automated frame analysis for intelligent restoration parameter selection.

This module provides automatic detection of:
- Content types (faces, text, animation, landscapes)
- Degradation types (noise, blur, scratches, compression artifacts)
- Quality metrics (brightness, contrast, sharpness)
- Optimal restoration parameters
- Time estimation for processing
- Hardware-aware recommendations
- Pre-flight checks and batch analysis
"""

import logging
import os
import shutil
import subprocess
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Detected content type in frame."""
    UNKNOWN = auto()
    FACE_PORTRAIT = auto()
    FACE_GROUP = auto()
    LANDSCAPE = auto()
    ARCHITECTURE = auto()
    TEXT_DOCUMENT = auto()
    ANIMATION = auto()
    ACTION_SCENE = auto()
    LOW_LIGHT = auto()
    HIGH_CONTRAST = auto()


class DegradationType(Enum):
    """Detected degradation type."""
    NONE = auto()
    LIGHT_NOISE = auto()
    HEAVY_NOISE = auto()
    FILM_GRAIN = auto()
    BLUR = auto()
    COMPRESSION_ARTIFACTS = auto()
    SCRATCHES = auto()
    DUST_DEBRIS = auto()
    INTERLACING = auto()
    COLOR_FADE = auto()


class WarningSeverity(Enum):
    """Severity level for quality warnings."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


class BlockerType(Enum):
    """Types of blockers that prevent processing."""
    DISK_SPACE = auto()
    MEMORY = auto()
    GPU_UNAVAILABLE = auto()
    MODEL_UNAVAILABLE = auto()
    CORRUPT_VIDEO = auto()
    PERMISSION_DENIED = auto()


@dataclass
class FrameMetrics:
    """Metrics extracted from a single frame."""
    brightness: float = 0.0  # 0-255
    contrast: float = 0.0  # Standard deviation
    sharpness: float = 0.0  # Laplacian variance
    noise_level: float = 0.0  # Estimated noise
    edge_density: float = 0.0  # Edge pixel ratio
    color_variance: float = 0.0  # Color distribution
    entropy: float = 0.0  # Information content
    has_faces: bool = False
    face_count: int = 0
    face_area_ratio: float = 0.0  # Faces as % of frame


@dataclass
class TimeEstimate:
    """Time estimate for video processing.

    Attributes:
        total_seconds: Estimated total processing time in seconds
        per_frame_ms: Average time per frame in milliseconds
        factors: Dictionary of factors affecting the estimate
    """
    total_seconds: float
    per_frame_ms: float
    factors: Dict[str, float] = field(default_factory=dict)

    @property
    def formatted(self) -> str:
        """Return human-readable time estimate."""
        return format_time_estimate(self)


@dataclass
class ContentMix:
    """Content type breakdown for mixed-content videos.

    Attributes:
        anime_percent: Percentage of frames detected as anime/animation
        live_action_percent: Percentage of frames detected as live-action
        cgi_percent: Percentage of frames detected as CGI/3D rendered
        text_overlay_percent: Percentage of frames with significant text overlays
    """
    anime_percent: float = 0.0
    live_action_percent: float = 0.0
    cgi_percent: float = 0.0
    text_overlay_percent: float = 0.0

    def primary_content_type(self) -> str:
        """Return the dominant content type."""
        types = {
            "anime": self.anime_percent,
            "live_action": self.live_action_percent,
            "cgi": self.cgi_percent,
        }
        return max(types, key=types.get)


@dataclass
class ProcessingSegment:
    """A segment of video with specific processing recommendations.

    Attributes:
        start_frame: Starting frame number
        end_frame: Ending frame number
        content_type: Type of content in this segment
        recommended_model: Recommended model for this segment
        confidence: Confidence score for the detection (0.0-1.0)
    """
    start_frame: int
    end_frame: int
    content_type: str
    recommended_model: str
    confidence: float = 1.0


@dataclass
class QualityWarning:
    """A warning about source video quality.

    Attributes:
        type: Type of quality issue detected
        severity: How severe the issue is (INFO, WARNING, ERROR)
        message: Human-readable description of the issue
        recommendation: Suggested action to address the issue
    """
    type: str
    severity: WarningSeverity
    message: str
    recommendation: str


@dataclass
class HardwareInfo:
    """Hardware information for processing recommendations.

    Attributes:
        gpu_available: Whether a GPU is available
        gpu_name: Name of the GPU (if available)
        vram_total_mb: Total GPU VRAM in MB
        vram_free_mb: Free GPU VRAM in MB
        ram_total_mb: Total system RAM in MB
        ram_free_mb: Free system RAM in MB
        cpu_cores: Number of CPU cores
    """
    gpu_available: bool = False
    gpu_name: str = "None"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    ram_total_mb: int = 0
    ram_free_mb: int = 0
    cpu_cores: int = 1


@dataclass
class ProcessingPlan:
    """Hardware-aware processing plan.

    Attributes:
        model: Recommended model name
        scale: Recommended scale factor
        tile_size: Recommended tile size (0 = no tiling)
        batch_size: Recommended batch size for processing
        estimated_time: Estimated processing time
        estimated_vram: Estimated peak VRAM usage in MB
        use_cpu: Whether to use CPU fallback
        notes: Additional notes about the plan
    """
    model: str
    scale: int
    tile_size: int
    batch_size: int
    estimated_time: TimeEstimate
    estimated_vram: int
    use_cpu: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class PreflightBlocker:
    """A blocker preventing processing from starting.

    Attributes:
        type: Type of blocker
        message: Description of the blocking issue
        resolution: How to resolve the blocker
    """
    type: BlockerType
    message: str
    resolution: str


@dataclass
class PreflightResult:
    """Result of pre-flight check before processing.

    Attributes:
        can_proceed: Whether processing can proceed
        warnings: List of non-blocking warnings
        blockers: List of issues that prevent processing
        disk_space_required_mb: Estimated disk space needed in MB
        disk_space_available_mb: Available disk space in MB
        memory_required_mb: Estimated memory needed in MB
        memory_available_mb: Available memory in MB
    """
    can_proceed: bool
    warnings: List[QualityWarning] = field(default_factory=list)
    blockers: List[PreflightBlocker] = field(default_factory=list)
    disk_space_required_mb: int = 0
    disk_space_available_mb: int = 0
    memory_required_mb: int = 0
    memory_available_mb: int = 0


@dataclass
class BatchVideoInfo:
    """Information about a single video in a batch.

    Attributes:
        path: Path to the video file
        analysis: Video analysis results
        time_estimate: Processing time estimate
        disk_space_mb: Estimated disk space needed
        priority: Processing priority (lower = process first)
    """
    path: Path
    analysis: Optional["VideoAnalysis"] = None
    time_estimate: Optional[TimeEstimate] = None
    disk_space_mb: int = 0
    priority: int = 0


@dataclass
class BatchAnalysis:
    """Analysis results for a batch of videos.

    Attributes:
        videos: List of video information
        total_time_estimate: Total estimated processing time
        total_disk_space_mb: Total disk space required
        processing_order: Recommended order for processing (indices into videos list)
        warnings: Batch-level warnings
    """
    videos: List[BatchVideoInfo] = field(default_factory=list)
    total_time_estimate: Optional[TimeEstimate] = None
    total_disk_space_mb: int = 0
    processing_order: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def format_time_estimate(estimate: TimeEstimate) -> str:
    """Format a time estimate as a human-readable string.

    Args:
        estimate: TimeEstimate object to format

    Returns:
        Formatted string like "~2 hours 15 minutes" or "~45 minutes"
    """
    total_seconds = estimate.total_seconds

    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"

    total_minutes = total_seconds / 60

    if total_minutes < 60:
        return f"~{int(total_minutes)} minutes"

    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)

    if minutes == 0:
        return f"~{hours} hour{'s' if hours > 1 else ''}"

    return f"~{hours} hour{'s' if hours > 1 else ''} {minutes} minute{'s' if minutes > 1 else ''}"


def get_hardware_info() -> HardwareInfo:
    """Detect current hardware capabilities.

    Returns:
        HardwareInfo with detected hardware specifications
    """
    info = HardwareInfo()

    # Detect GPU
    try:
        from ..utils.gpu import get_all_gpu_info, is_nvidia_gpu_available
        if is_nvidia_gpu_available():
            gpus = get_all_gpu_info()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                info.gpu_available = True
                info.gpu_name = gpu.name
                info.vram_total_mb = gpu.total_memory_mb
                info.vram_free_mb = gpu.free_memory_mb
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    # Detect RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.ram_total_mb = int(mem.total / (1024 * 1024))
        info.ram_free_mb = int(mem.available / (1024 * 1024))
    except ImportError:
        # Fallback for systems without psutil
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('MemTotal:'):
                        info.ram_total_mb = int(line.split()[1]) // 1024
                    elif line.startswith('MemAvailable:'):
                        info.ram_free_mb = int(line.split()[1]) // 1024
        except Exception:
            pass

    # Detect CPU cores
    try:
        info.cpu_cores = os.cpu_count() or 1
    except Exception:
        info.cpu_cores = 1

    return info


@dataclass
class VideoAnalysis:
    """Complete analysis results for a video."""
    # Basic info
    total_frames: int = 0
    sample_count: int = 0
    resolution: Tuple[int, int] = (0, 0)
    source_fps: float = 0.0
    duration: float = 0.0

    # Aggregated metrics
    avg_brightness: float = 0.0
    avg_contrast: float = 0.0
    avg_sharpness: float = 0.0
    avg_noise: float = 0.0

    # Content detection
    primary_content: ContentType = ContentType.UNKNOWN
    content_breakdown: Dict[str, float] = field(default_factory=dict)
    face_frame_ratio: float = 0.0  # % of frames with faces

    # Degradation detection
    degradation_types: List[DegradationType] = field(default_factory=list)
    degradation_severity: str = "unknown"  # light, moderate, heavy

    # Recommended settings (auto-computed)
    recommended_scale: int = 2
    recommended_model: str = "realesrgan-x4plus"
    recommended_denoise: float = 0.0
    recommended_target_fps: Optional[float] = None
    enable_face_restoration: bool = False
    enable_scratch_removal: bool = False
    enable_deinterlace: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_frames": self.total_frames,
            "resolution": list(self.resolution),
            "source_fps": self.source_fps,
            "duration": self.duration,
            "avg_brightness": self.avg_brightness,
            "avg_noise": self.avg_noise,
            "primary_content": self.primary_content.name,
            "degradation_types": [d.name for d in self.degradation_types],
            "degradation_severity": self.degradation_severity,
            "recommendations": {
                "scale": self.recommended_scale,
                "model": self.recommended_model,
                "denoise": self.recommended_denoise,
                "target_fps": self.recommended_target_fps,
                "face_restoration": self.enable_face_restoration,
                "scratch_removal": self.enable_scratch_removal,
                "deinterlace": self.enable_deinterlace,
            }
        }


class FrameAnalyzer:
    """Automated frame analysis for restoration parameter optimization.

    This analyzer examines sample frames from a video to automatically
    determine optimal restoration settings without manual intervention.
    """

    def __init__(
        self,
        sample_rate: int = 100,  # Analyze every Nth frame
        max_samples: int = 50,   # Maximum frames to analyze
        enable_face_detection: bool = True,
    ):
        """Initialize the analyzer.

        Args:
            sample_rate: Analyze every Nth frame
            max_samples: Maximum number of frames to analyze
            enable_face_detection: Whether to detect faces (slower but more accurate)
        """
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.enable_face_detection = enable_face_detection
        self._face_detector = None

    def analyze_video(self, video_path: Path) -> VideoAnalysis:
        """Analyze video and return comprehensive analysis.

        Args:
            video_path: Path to video file

        Returns:
            VideoAnalysis with detected content, degradation, and recommendations
        """
        logger.info(f"Analyzing video: {video_path}")

        # Get video metadata
        metadata = self._get_video_metadata(video_path)

        analysis = VideoAnalysis(
            total_frames=int(metadata.get('frame_count', 0)),
            resolution=(metadata.get('width', 0), metadata.get('height', 0)),
            source_fps=metadata.get('framerate', 24.0),
            duration=metadata.get('duration', 0.0),
        )

        # Extract and analyze sample frames
        sample_frames = self._extract_sample_frames(video_path, analysis.total_frames)

        if not sample_frames:
            logger.warning("No sample frames extracted, using defaults")
            self._set_default_recommendations(analysis)
            return analysis

        analysis.sample_count = len(sample_frames)

        # Analyze each sample frame
        frame_metrics = []
        for frame_path in sample_frames:
            metrics = self._analyze_single_frame(frame_path)
            frame_metrics.append(metrics)

        # Aggregate metrics
        self._aggregate_metrics(analysis, frame_metrics)

        # Detect content type
        self._detect_content_type(analysis, frame_metrics)

        # Detect degradation
        self._detect_degradation(analysis, frame_metrics)

        # Generate recommendations
        self._generate_recommendations(analysis)

        # Cleanup sample frames
        self._cleanup_samples(sample_frames)

        logger.info(f"Analysis complete: {analysis.degradation_severity} degradation, "
                   f"recommended scale={analysis.recommended_scale}x")

        return analysis

    def analyze_frames_dir(self, frames_dir: Path) -> VideoAnalysis:
        """Analyze a directory of already-extracted frames.

        Args:
            frames_dir: Directory containing frame images

        Returns:
            VideoAnalysis with recommendations
        """
        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        analysis = VideoAnalysis(total_frames=len(frames))

        # Sample frames
        step = max(1, len(frames) // self.max_samples)
        sample_frames = frames[::step][:self.max_samples]
        analysis.sample_count = len(sample_frames)

        # Analyze samples
        frame_metrics = []
        for frame_path in sample_frames:
            metrics = self._analyze_single_frame(frame_path)
            frame_metrics.append(metrics)

        # Process results
        self._aggregate_metrics(analysis, frame_metrics)
        self._detect_content_type(analysis, frame_metrics)
        self._detect_degradation(analysis, frame_metrics)
        self._generate_recommendations(analysis)

        return analysis

    def _get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                {}
            )

            # Parse framerate
            fps_str = video_stream.get('r_frame_rate', '24/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                framerate = num / den if den else 24.0
            else:
                framerate = float(fps_str)

            # Estimate frame count
            duration = float(data.get('format', {}).get('duration', 0))
            frame_count = int(duration * framerate)

            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'framerate': framerate,
                'duration': duration,
                'frame_count': frame_count,
                'codec': video_stream.get('codec_name', 'unknown'),
            }
        except Exception as e:
            logger.warning(f"Failed to get metadata: {e}")
            return {}

    def _extract_sample_frames(self, video_path: Path, total_frames: int) -> List[Path]:
        """Extract sample frames for analysis."""
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="framewright_analyze_"))

        # Calculate which frames to extract
        if total_frames <= 0:
            # Estimate from duration
            total_frames = 1000

        num_samples = min(self.max_samples, total_frames // self.sample_rate)
        num_samples = max(5, num_samples)  # At least 5 samples

        # Extract frames at regular intervals
        # Use ffmpeg select filter for efficiency
        interval = max(1, total_frames // num_samples)

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{interval}))',
            '-vsync', 'vfr',
            '-frames:v', str(num_samples),
            '-q:v', '2',
            str(temp_dir / 'sample_%04d.png')
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            return sorted(temp_dir.glob('sample_*.png'))
        except Exception as e:
            logger.warning(f"Failed to extract samples: {e}")
            return []

    def _analyze_single_frame(self, frame_path: Path) -> FrameMetrics:
        """Analyze a single frame for various metrics."""
        metrics = FrameMetrics()

        try:
            # Read PNG file and extract basic stats
            # Using pure Python for portability (no PIL/CV2 required)
            stats = self._get_image_stats_ffmpeg(frame_path)

            metrics.brightness = stats.get('brightness', 128)
            metrics.contrast = stats.get('contrast', 50)
            metrics.entropy = stats.get('entropy', 5.0)

            # Estimate noise from high-frequency content
            metrics.noise_level = self._estimate_noise_level(stats)

            # Estimate sharpness
            metrics.sharpness = stats.get('sharpness', 50)

            # Edge density approximation
            metrics.edge_density = min(1.0, stats.get('edge_strength', 0.3))

            # Face detection (if enabled)
            if self.enable_face_detection:
                face_result = self._detect_faces_simple(frame_path)
                metrics.has_faces = face_result['has_faces']
                metrics.face_count = face_result['count']
                metrics.face_area_ratio = face_result['area_ratio']

        except Exception as e:
            logger.debug(f"Frame analysis error: {e}")

        return metrics

    def _get_image_stats_ffmpeg(self, frame_path: Path) -> Dict[str, float]:
        """Get image statistics using ffmpeg signalstats filter."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-f', 'lavfi',
            '-i', f'movie={frame_path},signalstats',
            '-show_entries', 'frame_tags',
            '-print_format', 'json'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            data = json.loads(result.stdout) if result.stdout.strip() else {}

            frames = data.get('frames', [{}])
            if frames:
                tags = frames[0].get('tags', {})
                return {
                    'brightness': float(tags.get('lavfi.signalstats.YAVG', 128)),
                    'contrast': float(tags.get('lavfi.signalstats.YMAX', 255)) -
                               float(tags.get('lavfi.signalstats.YMIN', 0)),
                    'entropy': float(tags.get('lavfi.signalstats.YHIGH', 200)) / 40,
                    'sharpness': 50,  # Default, would need edge detection
                    'edge_strength': 0.3,
                }
        except Exception:
            pass

        # Fallback: estimate from file size (higher entropy = larger file)
        try:
            file_size = frame_path.stat().st_size
            # Rough heuristic: larger PNG = more detail
            estimated_entropy = min(8.0, file_size / 100000)
            return {
                'brightness': 128,
                'contrast': 100,
                'entropy': estimated_entropy,
                'sharpness': 50,
                'edge_strength': 0.3,
            }
        except Exception:
            return {}

    def _estimate_noise_level(self, stats: Dict[str, float]) -> float:
        """Estimate noise level from image statistics."""
        # Higher entropy with lower edge strength suggests noise
        entropy = stats.get('entropy', 5.0)
        edge_strength = stats.get('edge_strength', 0.3)

        # Noise estimation heuristic
        if entropy > 6.5 and edge_strength < 0.2:
            return 0.8  # High noise
        elif entropy > 5.5 and edge_strength < 0.3:
            return 0.5  # Moderate noise
        elif entropy > 4.5:
            return 0.3  # Light noise
        else:
            return 0.1  # Minimal noise

    def _detect_faces_simple(self, frame_path: Path) -> Dict[str, Any]:
        """Simple face detection using ffmpeg drawbox heuristics.

        For better accuracy, GFPGAN or MTCNN would be used,
        but this provides a fast estimation without dependencies.
        """
        # Use ffmpeg's facedetect filter if available
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-f', 'lavfi',
            '-i', f'movie={frame_path},facedetect',
            '-show_entries', 'frame_tags',
            '-print_format', 'json'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout) if result.stdout.strip() else {}

            frames = data.get('frames', [{}])
            if frames:
                tags = frames[0].get('tags', {})
                # Count face detections
                face_count = sum(1 for k in tags.keys() if 'face' in k.lower())
                return {
                    'has_faces': face_count > 0,
                    'count': face_count,
                    'area_ratio': min(0.5, face_count * 0.1),
                }
        except Exception:
            pass

        # Fallback: assume no faces detected
        return {'has_faces': False, 'count': 0, 'area_ratio': 0.0}

    def _aggregate_metrics(self, analysis: VideoAnalysis, metrics: List[FrameMetrics]) -> None:
        """Aggregate frame metrics into video-level statistics."""
        if not metrics:
            return

        n = len(metrics)
        analysis.avg_brightness = sum(m.brightness for m in metrics) / n
        analysis.avg_contrast = sum(m.contrast for m in metrics) / n
        analysis.avg_sharpness = sum(m.sharpness for m in metrics) / n
        analysis.avg_noise = sum(m.noise_level for m in metrics) / n

        # Face statistics
        face_frames = sum(1 for m in metrics if m.has_faces)
        analysis.face_frame_ratio = face_frames / n

    def _detect_content_type(self, analysis: VideoAnalysis, metrics: List[FrameMetrics]) -> None:
        """Detect primary content type from metrics."""
        # Content type heuristics
        if analysis.face_frame_ratio > 0.5:
            avg_face_count = sum(m.face_count for m in metrics) / len(metrics)
            if avg_face_count > 2:
                analysis.primary_content = ContentType.FACE_GROUP
            else:
                analysis.primary_content = ContentType.FACE_PORTRAIT
        elif analysis.avg_brightness < 60:
            analysis.primary_content = ContentType.LOW_LIGHT
        elif analysis.avg_contrast > 180:
            analysis.primary_content = ContentType.HIGH_CONTRAST
        else:
            # Check edge density for animation vs live action
            avg_edge = sum(m.edge_density for m in metrics) / len(metrics)
            if avg_edge > 0.6:
                analysis.primary_content = ContentType.ARCHITECTURE
            elif avg_edge < 0.15:
                analysis.primary_content = ContentType.ANIMATION
            else:
                analysis.primary_content = ContentType.LANDSCAPE

        # Build content breakdown
        analysis.content_breakdown = {
            "faces": analysis.face_frame_ratio,
            "low_light": sum(1 for m in metrics if m.brightness < 60) / len(metrics),
            "high_detail": sum(1 for m in metrics if m.edge_density > 0.5) / len(metrics),
        }

    def _detect_degradation(self, analysis: VideoAnalysis, metrics: List[FrameMetrics]) -> None:
        """Detect degradation types from metrics."""
        degradations = []

        # Noise detection
        if analysis.avg_noise > 0.6:
            degradations.append(DegradationType.HEAVY_NOISE)
        elif analysis.avg_noise > 0.4:
            degradations.append(DegradationType.FILM_GRAIN)
        elif analysis.avg_noise > 0.2:
            degradations.append(DegradationType.LIGHT_NOISE)

        # Blur detection (low sharpness)
        if analysis.avg_sharpness < 30:
            degradations.append(DegradationType.BLUR)

        # Low resolution indicator
        width, height = analysis.resolution
        if width > 0 and width < 720:
            # Small source, likely needs significant upscaling
            if analysis.avg_noise < 0.3:
                degradations.append(DegradationType.COMPRESSION_ARTIFACTS)

        # Determine overall severity
        if DegradationType.HEAVY_NOISE in degradations or len(degradations) >= 3:
            analysis.degradation_severity = "heavy"
        elif len(degradations) >= 2 or DegradationType.FILM_GRAIN in degradations:
            analysis.degradation_severity = "moderate"
        elif degradations:
            analysis.degradation_severity = "light"
        else:
            analysis.degradation_severity = "minimal"

        analysis.degradation_types = degradations

    def _generate_recommendations(self, analysis: VideoAnalysis) -> None:
        """Generate optimal settings based on analysis."""

        # Scale factor
        width, height = analysis.resolution
        if width < 480 or analysis.degradation_severity == "heavy":
            analysis.recommended_scale = 4
        elif width < 720 or analysis.degradation_severity == "moderate":
            analysis.recommended_scale = 4
        else:
            analysis.recommended_scale = 2

        # Model selection
        if analysis.primary_content == ContentType.ANIMATION:
            analysis.recommended_model = "realesrgan-x4plus-anime"
        elif analysis.primary_content in (ContentType.FACE_PORTRAIT, ContentType.FACE_GROUP):
            analysis.recommended_model = "realesrgan-x4plus"
            analysis.enable_face_restoration = True
        else:
            analysis.recommended_model = "realesrgan-x4plus"

        # Denoise strength
        if DegradationType.HEAVY_NOISE in analysis.degradation_types:
            analysis.recommended_denoise = 0.8
        elif DegradationType.FILM_GRAIN in analysis.degradation_types:
            analysis.recommended_denoise = 0.5
        elif DegradationType.LIGHT_NOISE in analysis.degradation_types:
            analysis.recommended_denoise = 0.3
        else:
            analysis.recommended_denoise = 0.0

        # Face restoration
        if analysis.face_frame_ratio > 0.3:
            analysis.enable_face_restoration = True

        # Target FPS for RIFE
        if analysis.source_fps > 0:
            if analysis.source_fps <= 24:
                analysis.recommended_target_fps = 48.0  # 2x smooth
            elif analysis.source_fps <= 30:
                analysis.recommended_target_fps = 60.0
            else:
                analysis.recommended_target_fps = None  # Already smooth

        # Scratch removal (heavy degradation of old content)
        if analysis.degradation_severity == "heavy":
            analysis.enable_scratch_removal = True

    def _set_default_recommendations(self, analysis: VideoAnalysis) -> None:
        """Set safe default recommendations when analysis fails."""
        analysis.recommended_scale = 2
        analysis.recommended_model = "realesrgan-x4plus"
        analysis.recommended_denoise = 0.3
        analysis.degradation_severity = "unknown"

    def _cleanup_samples(self, sample_frames: List[Path]) -> None:
        """Clean up temporary sample frames."""
        if sample_frames:
            temp_dir = sample_frames[0].parent
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    # =========================================================================
    # Time Estimation
    # =========================================================================

    def estimate_processing_time(
        self,
        video_path: Path,
        config: Optional["Config"] = None,
    ) -> TimeEstimate:
        """Estimate processing time for a video.

        Args:
            video_path: Path to video file
            config: Optional configuration (uses defaults if not provided)

        Returns:
            TimeEstimate with total time and per-frame breakdown
        """
        # Get video metadata
        metadata = self._get_video_metadata(video_path)
        frame_count = metadata.get('frame_count', 0)
        width = metadata.get('width', 1920)
        height = metadata.get('height', 1080)

        if frame_count == 0:
            # Estimate from duration
            duration = metadata.get('duration', 60)
            fps = metadata.get('framerate', 24)
            frame_count = int(duration * fps)

        # Determine processing parameters
        if config is not None:
            scale = config.scale_factor
            model = config.model_name
        else:
            scale = 4
            model = "realesrgan-x4plus"

        # Calculate factors
        factors = self._calculate_time_factors(
            width, height, scale, model
        )

        # Base time per frame (milliseconds)
        # These are empirical values based on typical GPU performance
        base_ms_per_frame = 50.0  # Fast GPU baseline

        # Apply factors
        per_frame_ms = base_ms_per_frame
        per_frame_ms *= factors.get('resolution_factor', 1.0)
        per_frame_ms *= factors.get('scale_factor', 1.0)
        per_frame_ms *= factors.get('model_factor', 1.0)
        per_frame_ms *= factors.get('gpu_factor', 1.0)

        # Calculate total time
        total_ms = per_frame_ms * frame_count

        # Add overhead for frame extraction and reassembly (~10%)
        total_ms *= 1.1

        total_seconds = total_ms / 1000.0

        return TimeEstimate(
            total_seconds=total_seconds,
            per_frame_ms=per_frame_ms,
            factors=factors,
        )

    def _calculate_time_factors(
        self,
        width: int,
        height: int,
        scale: int,
        model: str,
    ) -> Dict[str, float]:
        """Calculate time multiplier factors.

        Args:
            width: Frame width
            height: Frame height
            scale: Upscaling factor
            model: Model name

        Returns:
            Dictionary of named factors
        """
        factors = {}

        # Resolution factor (relative to 1080p)
        pixels = width * height
        base_pixels = 1920 * 1080
        factors['resolution_factor'] = max(0.5, pixels / base_pixels)

        # Scale factor
        scale_multipliers = {2: 1.0, 4: 2.5}
        factors['scale_factor'] = scale_multipliers.get(scale, 2.5)

        # Model complexity factor
        model_multipliers = {
            'realesrgan-x2plus': 0.8,
            'realesrgan-x4plus': 1.0,
            'realesrgan-x4plus-anime': 0.9,
            'realesr-animevideov3': 0.85,
        }
        factors['model_factor'] = model_multipliers.get(model, 1.0)

        # GPU performance factor
        hw_info = get_hardware_info()
        if hw_info.gpu_available:
            # Estimate relative GPU speed based on VRAM
            # Higher VRAM usually means faster GPU
            if hw_info.vram_total_mb >= 12000:
                factors['gpu_factor'] = 0.5  # High-end GPU
            elif hw_info.vram_total_mb >= 8000:
                factors['gpu_factor'] = 0.75  # Mid-range GPU
            elif hw_info.vram_total_mb >= 4000:
                factors['gpu_factor'] = 1.0  # Entry GPU
            else:
                factors['gpu_factor'] = 2.0  # Low-end or integrated
        else:
            # CPU fallback is much slower
            factors['gpu_factor'] = 10.0

        return factors

    # =========================================================================
    # Enhanced Content Detection
    # =========================================================================

    def detect_content_mix(self, video_path: Path) -> ContentMix:
        """Detect the mix of content types in a video.

        Args:
            video_path: Path to video file

        Returns:
            ContentMix with percentages for each content type
        """
        logger.info(f"Detecting content mix for: {video_path}")

        # Analyze video
        analysis = self.analyze_video(video_path)

        # Initialize mix
        mix = ContentMix()

        # Use frame metrics to classify content
        if not hasattr(self, '_last_frame_metrics'):
            # Re-extract samples if needed
            metadata = self._get_video_metadata(video_path)
            samples = self._extract_sample_frames(video_path, metadata.get('frame_count', 1000))
            if not samples:
                return mix
            metrics = [self._analyze_single_frame(f) for f in samples]
            self._cleanup_samples(samples)
        else:
            metrics = self._last_frame_metrics

        if not metrics:
            return mix

        n = len(metrics)

        # Classify each frame
        anime_frames = 0
        live_action_frames = 0
        cgi_frames = 0
        text_frames = 0

        for m in metrics:
            # Animation detection heuristics
            # Low edge density + high contrast + flat colors = anime
            if m.edge_density < 0.2 and m.contrast > 150:
                anime_frames += 1
            # High edge density + natural brightness variation = live action
            elif m.edge_density > 0.25 and 80 < m.brightness < 200:
                live_action_frames += 1
            # Very smooth gradients + unnatural lighting = CGI
            elif m.edge_density < 0.15 and m.sharpness > 60:
                cgi_frames += 1

            # Text detection (high edge density in small regions)
            if m.edge_density > 0.7:
                text_frames += 1

        # Calculate percentages
        mix.anime_percent = (anime_frames / n) * 100 if n > 0 else 0
        mix.live_action_percent = (live_action_frames / n) * 100 if n > 0 else 0
        mix.cgi_percent = (cgi_frames / n) * 100 if n > 0 else 0
        mix.text_overlay_percent = (text_frames / n) * 100 if n > 0 else 0

        # Normalize if percentages don't add up
        total = mix.anime_percent + mix.live_action_percent + mix.cgi_percent
        if total > 0 and total < 100:
            # Assign remainder to live action as default
            mix.live_action_percent += (100 - total)

        return mix

    def recommend_multi_model_pipeline(
        self,
        mix: ContentMix,
    ) -> List[ProcessingSegment]:
        """Recommend a multi-model pipeline for mixed content.

        Args:
            mix: ContentMix from detect_content_mix

        Returns:
            List of ProcessingSegments with model recommendations
        """
        segments = []

        # If content is primarily one type (>80%), use single model
        if mix.anime_percent > 80:
            segments.append(ProcessingSegment(
                start_frame=0,
                end_frame=-1,  # -1 means end of video
                content_type="anime",
                recommended_model="realesrgan-x4plus-anime",
                confidence=mix.anime_percent / 100,
            ))
        elif mix.live_action_percent > 80:
            segments.append(ProcessingSegment(
                start_frame=0,
                end_frame=-1,
                content_type="live_action",
                recommended_model="realesrgan-x4plus",
                confidence=mix.live_action_percent / 100,
            ))
        elif mix.cgi_percent > 80:
            segments.append(ProcessingSegment(
                start_frame=0,
                end_frame=-1,
                content_type="cgi",
                recommended_model="realesrgan-x4plus",
                confidence=mix.cgi_percent / 100,
            ))
        else:
            # Mixed content - recommend hybrid approach
            # For simplicity, use the model for dominant content
            primary = mix.primary_content_type()
            model = {
                "anime": "realesrgan-x4plus-anime",
                "live_action": "realesrgan-x4plus",
                "cgi": "realesrgan-x4plus",
            }.get(primary, "realesrgan-x4plus")

            confidence = max(
                mix.anime_percent,
                mix.live_action_percent,
                mix.cgi_percent,
            ) / 100

            segments.append(ProcessingSegment(
                start_frame=0,
                end_frame=-1,
                content_type=primary,
                recommended_model=model,
                confidence=confidence,
            ))

            # Add note about mixed content
            logger.info(
                f"Mixed content detected: anime={mix.anime_percent:.1f}%, "
                f"live_action={mix.live_action_percent:.1f}%, "
                f"cgi={mix.cgi_percent:.1f}%"
            )

        return segments

    # =========================================================================
    # Quality Warnings
    # =========================================================================

    def analyze_source_quality(self, video_path: Path) -> List[QualityWarning]:
        """Analyze source video and generate quality warnings.

        Args:
            video_path: Path to video file

        Returns:
            List of QualityWarning objects
        """
        warnings = []

        # Get video metadata
        metadata = self._get_extended_metadata(video_path)

        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        codec = metadata.get('codec', 'unknown')
        bitrate = metadata.get('bitrate', 0)
        has_audio = metadata.get('has_audio', True)
        is_interlaced = metadata.get('is_interlaced', False)
        is_vfr = metadata.get('is_vfr', False)
        corrupt_frames = metadata.get('corrupt_frames', 0)

        # Very low resolution
        if height > 0 and height < 240:
            warnings.append(QualityWarning(
                type="very_low_resolution",
                severity=WarningSeverity.WARNING,
                message=f"Video resolution is very low ({width}x{height}). "
                        f"AI upscaling may produce limited improvement.",
                recommendation="Consider using 2x scale instead of 4x for better quality/speed ratio.",
            ))
        elif height > 0 and height < 360:
            warnings.append(QualityWarning(
                type="low_resolution",
                severity=WarningSeverity.INFO,
                message=f"Video resolution is low ({width}x{height}).",
                recommendation="4x upscaling is recommended for best results.",
            ))

        # Heavy compression artifacts
        if bitrate > 0:
            pixels = width * height if width > 0 and height > 0 else 1920 * 1080
            bits_per_pixel = (bitrate * 1000) / (pixels * 24)  # Assume 24fps
            if bits_per_pixel < 0.05:
                warnings.append(QualityWarning(
                    type="heavy_compression",
                    severity=WarningSeverity.WARNING,
                    message=f"Video appears heavily compressed (bitrate: {bitrate}kbps). "
                            f"AI enhancement may amplify compression artifacts.",
                    recommendation="Use moderate denoise settings (0.3-0.5) to reduce artifact amplification.",
                ))

        # Interlaced content
        if is_interlaced:
            warnings.append(QualityWarning(
                type="interlaced",
                severity=WarningSeverity.WARNING,
                message="Video appears to be interlaced, which may cause artifacts during processing.",
                recommendation="Apply deinterlacing before upscaling with: ffmpeg -vf yadif",
            ))

        # Variable frame rate
        if is_vfr:
            warnings.append(QualityWarning(
                type="variable_framerate",
                severity=WarningSeverity.WARNING,
                message="Video has variable frame rate, which may cause audio sync issues.",
                recommendation="Convert to constant frame rate first with: "
                              "ffmpeg -vsync cfr -r 24",
            ))

        # Missing audio
        if not has_audio:
            warnings.append(QualityWarning(
                type="no_audio",
                severity=WarningSeverity.INFO,
                message="No audio track detected in video.",
                recommendation="Output video will have no audio unless audio is added separately.",
            ))

        # Corrupt frames
        if corrupt_frames > 0:
            severity = WarningSeverity.ERROR if corrupt_frames > 10 else WarningSeverity.WARNING
            warnings.append(QualityWarning(
                type="corrupt_frames",
                severity=severity,
                message=f"Detected {corrupt_frames} potentially corrupt frame(s).",
                recommendation="Consider repairing video with: ffmpeg -fflags +genpts+discardcorrupt",
            ))

        return warnings

    def _get_extended_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Get extended video metadata including quality indicators.

        Args:
            video_path: Path to video file

        Returns:
            Extended metadata dictionary
        """
        basic_metadata = self._get_video_metadata(video_path)
        metadata = dict(basic_metadata)

        # Check for additional properties
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                {}
            )

            # Find audio stream
            audio_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
                None
            )

            metadata['has_audio'] = audio_stream is not None

            # Check for interlacing
            field_order = video_stream.get('field_order', 'progressive')
            metadata['is_interlaced'] = field_order not in ('progressive', 'unknown', '')

            # Check for VFR (variable frame rate)
            r_frame_rate = video_stream.get('r_frame_rate', '24/1')
            avg_frame_rate = video_stream.get('avg_frame_rate', '24/1')
            if r_frame_rate != avg_frame_rate:
                try:
                    r_num, r_den = map(float, r_frame_rate.split('/'))
                    a_num, a_den = map(float, avg_frame_rate.split('/'))
                    r_fps = r_num / r_den if r_den else 0
                    a_fps = a_num / a_den if a_den else 0
                    metadata['is_vfr'] = abs(r_fps - a_fps) > 1.0
                except Exception:
                    metadata['is_vfr'] = False
            else:
                metadata['is_vfr'] = False

            # Get bitrate
            format_data = data.get('format', {})
            bit_rate = format_data.get('bit_rate', 0)
            if bit_rate:
                metadata['bitrate'] = int(bit_rate) // 1000  # Convert to kbps

            # Check for corrupt frames (basic check)
            metadata['corrupt_frames'] = 0  # Would need deeper analysis

        except Exception as e:
            logger.debug(f"Extended metadata extraction failed: {e}")
            metadata['has_audio'] = True
            metadata['is_interlaced'] = False
            metadata['is_vfr'] = False
            metadata['bitrate'] = 0
            metadata['corrupt_frames'] = 0

        return metadata

    # =========================================================================
    # Hardware-aware Recommendations
    # =========================================================================

    def recommend_for_hardware(
        self,
        video_info: VideoAnalysis,
        hardware_info: Optional[HardwareInfo] = None,
    ) -> ProcessingPlan:
        """Generate hardware-aware processing recommendations.

        Args:
            video_info: VideoAnalysis from analyze_video
            hardware_info: HardwareInfo (auto-detected if None)

        Returns:
            ProcessingPlan with optimized settings
        """
        if hardware_info is None:
            hardware_info = get_hardware_info()

        notes = []
        use_cpu = False

        # Determine base settings from analysis
        model = video_info.recommended_model
        scale = video_info.recommended_scale

        # Adjust for GPU availability
        if not hardware_info.gpu_available:
            use_cpu = True
            notes.append("No GPU detected - using CPU processing (significantly slower)")
            # Reduce scale for CPU processing
            if scale == 4:
                scale = 2
                notes.append("Reduced scale to 2x for CPU processing")

        # Calculate tile size based on VRAM
        tile_size = 0
        if hardware_info.gpu_available:
            vram = hardware_info.vram_free_mb
            width, height = video_info.resolution

            if vram < 2000:
                tile_size = 256
                notes.append(f"Using tile size 256 due to limited VRAM ({vram}MB)")
            elif vram < 4000:
                tile_size = 384
                notes.append(f"Using tile size 384 for {vram}MB VRAM")
            elif vram < 8000:
                # Calculate if tiling is needed
                output_pixels = (width * scale * height * scale) / 1_000_000
                if output_pixels > 2.0:  # >2 megapixels output
                    tile_size = 512
            # else: no tiling needed

        # Determine batch size
        if use_cpu:
            batch_size = 1
        elif hardware_info.vram_total_mb >= 12000:
            batch_size = 4
        elif hardware_info.vram_total_mb >= 8000:
            batch_size = 2
        else:
            batch_size = 1

        # Estimate VRAM usage
        width, height = video_info.resolution
        output_pixels = width * scale * height * scale
        # Rough estimate: ~400 bytes per output pixel for Real-ESRGAN
        estimated_vram = int((output_pixels * 400) / (1024 * 1024))
        if tile_size > 0:
            # With tiling, VRAM scales with tile size
            tile_pixels = tile_size * scale * tile_size * scale
            estimated_vram = int((tile_pixels * 400) / (1024 * 1024))

        # Create time estimate
        from .analyzer import TimeEstimate
        factors = self._calculate_time_factors(width, height, scale, model)
        if use_cpu:
            factors['gpu_factor'] = 10.0

        per_frame_ms = 50.0
        for factor in factors.values():
            per_frame_ms *= factor

        total_seconds = (per_frame_ms * video_info.total_frames) / 1000.0 * 1.1

        time_estimate = TimeEstimate(
            total_seconds=total_seconds,
            per_frame_ms=per_frame_ms,
            factors=factors,
        )

        return ProcessingPlan(
            model=model,
            scale=scale,
            tile_size=tile_size,
            batch_size=batch_size,
            estimated_time=time_estimate,
            estimated_vram=estimated_vram,
            use_cpu=use_cpu,
            notes=notes,
        )

    # =========================================================================
    # Pre-flight Check
    # =========================================================================

    def preflight_check(
        self,
        video_path: Path,
        config: Optional["Config"] = None,
    ) -> PreflightResult:
        """Perform pre-flight check before processing.

        Args:
            video_path: Path to video file
            config: Optional configuration

        Returns:
            PreflightResult with warnings and blockers
        """
        warnings = []
        blockers = []

        # Get hardware info
        hw_info = get_hardware_info()

        # Get video metadata
        try:
            metadata = self._get_extended_metadata(video_path)
        except Exception as e:
            blockers.append(PreflightBlocker(
                type=BlockerType.CORRUPT_VIDEO,
                message=f"Cannot read video file: {e}",
                resolution="Verify the file is a valid video and not corrupted.",
            ))
            return PreflightResult(
                can_proceed=False,
                blockers=blockers,
            )

        # Check video file exists and is readable
        if not video_path.exists():
            blockers.append(PreflightBlocker(
                type=BlockerType.CORRUPT_VIDEO,
                message=f"Video file not found: {video_path}",
                resolution="Check the file path is correct.",
            ))
            return PreflightResult(can_proceed=False, blockers=blockers)

        # Analyze source quality warnings
        quality_warnings = self.analyze_source_quality(video_path)
        warnings.extend(quality_warnings)

        # Estimate disk space requirements
        frame_count = metadata.get('frame_count', 0)
        width = metadata.get('width', 1920)
        height = metadata.get('height', 1080)
        scale = config.scale_factor if config else 4

        # Estimate: input frames + output frames + temp files
        # PNG files ~= width * height * 3 bytes (uncompressed estimate)
        bytes_per_input_frame = width * height * 3
        bytes_per_output_frame = (width * scale) * (height * scale) * 3

        input_frames_mb = (frame_count * bytes_per_input_frame) / (1024 * 1024)
        output_frames_mb = (frame_count * bytes_per_output_frame) / (1024 * 1024)
        disk_space_required_mb = int((input_frames_mb + output_frames_mb) * 1.2)

        # Check available disk space
        output_dir = config.get_output_dir() if config else video_path.parent
        try:
            disk_usage = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent)
            disk_space_available_mb = int(disk_usage.free / (1024 * 1024))
        except Exception:
            disk_space_available_mb = 0

        if disk_space_available_mb < disk_space_required_mb:
            blockers.append(PreflightBlocker(
                type=BlockerType.DISK_SPACE,
                message=f"Insufficient disk space. Need ~{disk_space_required_mb}MB, "
                        f"only {disk_space_available_mb}MB available.",
                resolution=f"Free up at least {disk_space_required_mb - disk_space_available_mb}MB of disk space.",
            ))

        # Check memory requirements
        memory_required_mb = int((bytes_per_output_frame * 10) / (1024 * 1024))  # ~10 frames in memory
        memory_available_mb = hw_info.ram_free_mb

        if memory_available_mb > 0 and memory_available_mb < memory_required_mb:
            blockers.append(PreflightBlocker(
                type=BlockerType.MEMORY,
                message=f"Insufficient RAM. Need ~{memory_required_mb}MB, "
                        f"only {memory_available_mb}MB available.",
                resolution="Close other applications to free memory.",
            ))

        # Check GPU availability
        if not hw_info.gpu_available:
            warnings.append(QualityWarning(
                type="no_gpu",
                severity=WarningSeverity.WARNING,
                message="No NVIDIA GPU detected. Processing will use CPU and be much slower.",
                recommendation="For faster processing, use a system with an NVIDIA GPU.",
            ))

        # Check model availability
        if config and config.model_dir:
            model_name = config.model_name
            model_path = config.model_dir / f"{model_name}.pth"
            if not model_path.exists():
                # Check alternative extensions
                alt_path = config.model_dir / f"{model_name}.bin"
                if not alt_path.exists():
                    warnings.append(QualityWarning(
                        type="model_not_cached",
                        severity=WarningSeverity.INFO,
                        message=f"Model {model_name} not found locally.",
                        recommendation="Model will be downloaded on first run.",
                    ))

        # Determine if we can proceed
        can_proceed = len(blockers) == 0

        return PreflightResult(
            can_proceed=can_proceed,
            warnings=warnings,
            blockers=blockers,
            disk_space_required_mb=disk_space_required_mb,
            disk_space_available_mb=disk_space_available_mb,
            memory_required_mb=memory_required_mb,
            memory_available_mb=memory_available_mb,
        )

    # =========================================================================
    # Batch Analysis
    # =========================================================================

    def analyze_batch(self, video_paths: List[Path]) -> BatchAnalysis:
        """Analyze a batch of videos for processing.

        Args:
            video_paths: List of paths to video files

        Returns:
            BatchAnalysis with aggregate information
        """
        batch = BatchAnalysis()
        batch_warnings = []

        total_time = 0.0
        total_disk = 0

        for i, path in enumerate(video_paths):
            logger.info(f"Analyzing video {i+1}/{len(video_paths)}: {path.name}")

            video_info = BatchVideoInfo(path=path)

            try:
                # Analyze video
                analysis = self.analyze_video(path)
                video_info.analysis = analysis

                # Estimate processing time
                time_est = self.estimate_processing_time(path)
                video_info.time_estimate = time_est
                total_time += time_est.total_seconds

                # Estimate disk space
                width, height = analysis.resolution
                scale = analysis.recommended_scale
                frame_count = analysis.total_frames

                bytes_per_frame = (width * scale) * (height * scale) * 3
                disk_mb = int((frame_count * bytes_per_frame * 2) / (1024 * 1024))
                video_info.disk_space_mb = disk_mb
                total_disk += disk_mb

                # Calculate priority (smaller files first for quick wins)
                video_info.priority = frame_count

            except Exception as e:
                logger.warning(f"Failed to analyze {path}: {e}")
                batch_warnings.append(f"Could not analyze {path.name}: {e}")
                video_info.priority = 999999  # Process last if analysis failed

            batch.videos.append(video_info)

        # Calculate total time estimate
        batch.total_time_estimate = TimeEstimate(
            total_seconds=total_time,
            per_frame_ms=0,  # Not applicable for batch
            factors={"video_count": len(video_paths)},
        )

        batch.total_disk_space_mb = total_disk

        # Determine optimal processing order
        # Sort by priority (smaller files first for quick feedback)
        indexed = [(i, v.priority) for i, v in enumerate(batch.videos)]
        indexed.sort(key=lambda x: x[1])
        batch.processing_order = [i for i, _ in indexed]

        # Check if batch fits in available disk space
        try:
            if video_paths:
                disk_usage = shutil.disk_usage(video_paths[0].parent)
                available_mb = int(disk_usage.free / (1024 * 1024))
                if available_mb < total_disk:
                    batch_warnings.append(
                        f"Batch requires ~{total_disk}MB but only {available_mb}MB available. "
                        f"Consider processing videos individually."
                    )
        except Exception:
            pass

        batch.warnings = batch_warnings

        return batch
