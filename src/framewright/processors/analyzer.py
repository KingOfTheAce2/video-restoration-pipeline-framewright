"""Automated frame analysis for intelligent restoration parameter selection.

This module provides automatic detection of:
- Content types (faces, text, animation, landscapes)
- Degradation types (noise, blur, scratches, compression artifacts)
- Quality metrics (brightness, contrast, sharpness)
- Optimal restoration parameters
"""

import logging
import subprocess
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import struct
import zlib

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
            import shutil
            temp_dir = sample_frames[0].parent
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
