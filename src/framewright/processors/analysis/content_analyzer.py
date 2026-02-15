"""Unified Content Analyzer for FrameWright Video Restoration.

Consolidates ALL analysis capabilities into a single coherent interface:
- Frame analysis (brightness, contrast, sharpness, noise)
- Scene detection and classification
- Noise profiling and characterization
- Film stock detection and era estimation
- Frame quality scoring
- Content type detection (film, animation, live action, mixed)
- Degradation detection
- Processing recommendations

Uses lazy loading to defer initialization of heavy components until needed.

Example:
    >>> analyzer = ContentAnalyzer()
    >>> analysis = analyzer.analyze("my_video.mp4")
    >>> print(f"Type: {analysis.content_type.value}")
    >>> print(f"Era: {analysis.era}")
    >>> print(f"Preset: {analysis.recommended_preset}")
    >>> for scene in analysis.scenes[:5]:
    ...     print(f"  Scene {scene.start_frame}-{scene.end_frame}")
"""

import logging
import subprocess
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ContentType(Enum):
    """Primary content type classification."""
    UNKNOWN = "unknown"
    FILM = "film"                    # Cinema film footage
    ANIMATION = "animation"          # Anime, cartoons, CGI
    LIVE_ACTION = "live_action"      # Modern live-action video
    DOCUMENTARY = "documentary"      # Documentary-style footage
    MIXED = "mixed"                  # Multiple content types
    VHS = "vhs"                      # VHS/analog tape footage
    BROADCAST = "broadcast"          # TV broadcast recordings


class SourceFormat(Enum):
    """Detected source format/medium."""
    UNKNOWN = "unknown"
    VHS = "vhs"                      # VHS tape
    VHS_HI_FI = "vhs_hi_fi"          # VHS Hi-Fi
    BETAMAX = "betamax"              # Betamax tape
    LASERDISC = "laserdisc"          # LaserDisc
    FILM_8MM = "film_8mm"            # 8mm film
    SUPER_8 = "super_8"              # Super 8 film
    FILM_16MM = "film_16mm"          # 16mm film
    FILM_35MM = "film_35mm"          # 35mm film
    DIGITAL = "digital"              # Digital source
    DVD = "dvd"                      # DVD rip
    BROADCAST_SD = "broadcast_sd"    # SD TV broadcast
    BROADCAST_HD = "broadcast_hd"    # HD TV broadcast
    TELECINE = "telecine"            # Telecine transfer


class DegradationType(Enum):
    """Types of video degradation detected."""
    NONE = "none"
    LIGHT_NOISE = "light_noise"
    HEAVY_NOISE = "heavy_noise"
    FILM_GRAIN = "film_grain"
    BLUR = "blur"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    SCRATCHES = "scratches"
    DUST_DEBRIS = "dust_debris"
    INTERLACING = "interlacing"
    COLOR_FADE = "color_fade"
    VHS_ARTIFACTS = "vhs_artifacts"
    TELECINE_JUDDER = "telecine_judder"
    BANDING = "banding"
    BLOCKING = "blocking"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NoiseProfile:
    """Noise characteristics analysis result."""
    overall_level: float = 0.0      # 0-100 scale
    luminance_noise: float = 0.0    # Y channel noise
    chroma_noise: float = 0.0       # UV channel noise
    temporal_noise: float = 0.0     # Frame-to-frame variation
    noise_type: str = "minimal"     # gaussian, film_grain, compression, etc.
    recommended_denoiser: str = "none"
    recommended_strength: float = 0.0
    preserve_grain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_level": self.overall_level,
            "luminance_noise": self.luminance_noise,
            "chroma_noise": self.chroma_noise,
            "temporal_noise": self.temporal_noise,
            "noise_type": self.noise_type,
            "recommended_denoiser": self.recommended_denoiser,
            "recommended_strength": self.recommended_strength,
            "preserve_grain": self.preserve_grain,
        }


@dataclass
class Scene:
    """Detected scene in video."""
    start_frame: int
    end_frame: int
    duration_frames: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    scene_type: str = "unknown"     # static, action, dialog, transition
    avg_brightness: float = 128.0
    avg_motion: float = 0.0
    has_faces: bool = False
    face_count: int = 0
    quality_score: float = 0.5

    def __post_init__(self):
        if self.duration_frames <= 0:
            self.duration_frames = self.end_frame - self.start_frame + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "scene_type": self.scene_type,
            "has_faces": self.has_faces,
            "face_count": self.face_count,
            "quality_score": self.quality_score,
        }


@dataclass
class ContentAnalysis:
    """Complete content analysis result.

    Contains all information gathered from analyzing a video file,
    including basic metadata, content classification, quality assessment,
    scene information, and processing recommendations.
    """
    # Basic info
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    duration: float = 0.0
    frame_count: int = 0
    codec: str = "unknown"
    bitrate: int = 0

    # Content detection
    content_type: ContentType = ContentType.UNKNOWN
    era: Optional[str] = None       # "1950s", "1980s", "modern", etc.
    source_format: SourceFormat = SourceFormat.UNKNOWN

    # Quality assessment
    noise_profile: NoiseProfile = field(default_factory=NoiseProfile)
    degradation_types: List[DegradationType] = field(default_factory=list)
    quality_score: float = 0.5      # 0-1, higher is better
    avg_brightness: float = 128.0
    avg_contrast: float = 50.0
    avg_sharpness: float = 50.0

    # Scene info
    scenes: List[Scene] = field(default_factory=list)
    scene_count: int = 0
    has_faces: bool = False
    face_count_estimate: int = 0

    # Film stock info (for film content)
    film_stock: Optional[str] = None
    film_stock_confidence: float = 0.0
    color_profile: Optional[Dict[str, float]] = None

    # Recommendations
    recommended_preset: str = "balanced"
    recommended_processors: List[str] = field(default_factory=list)
    recommended_scale: int = 2
    recommended_model: str = "realesrgan-x4plus"
    recommended_denoise: float = 0.3

    # Processing flags
    enable_face_restoration: bool = False
    enable_scratch_removal: bool = False
    enable_deinterlace: bool = False
    enable_color_correction: bool = False

    # Analysis metadata
    confidence: float = 0.0
    frames_analyzed: int = 0
    analysis_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "basic_info": {
                "resolution": list(self.resolution),
                "fps": self.fps,
                "duration": self.duration,
                "frame_count": self.frame_count,
                "codec": self.codec,
                "bitrate": self.bitrate,
            },
            "content": {
                "content_type": self.content_type.value,
                "era": self.era,
                "source_format": self.source_format.value,
            },
            "quality": {
                "score": self.quality_score,
                "avg_brightness": self.avg_brightness,
                "avg_contrast": self.avg_contrast,
                "avg_sharpness": self.avg_sharpness,
                "degradation_types": [d.value for d in self.degradation_types],
                "noise_profile": self.noise_profile.to_dict(),
            },
            "scenes": {
                "count": self.scene_count,
                "has_faces": self.has_faces,
                "face_count_estimate": self.face_count_estimate,
                "scenes": [s.to_dict() for s in self.scenes[:20]],  # Limit for JSON
            },
            "film_stock": {
                "detected": self.film_stock,
                "confidence": self.film_stock_confidence,
            },
            "recommendations": {
                "preset": self.recommended_preset,
                "processors": self.recommended_processors,
                "scale": self.recommended_scale,
                "model": self.recommended_model,
                "denoise": self.recommended_denoise,
                "face_restoration": self.enable_face_restoration,
                "scratch_removal": self.enable_scratch_removal,
                "deinterlace": self.enable_deinterlace,
                "color_correction": self.enable_color_correction,
            },
            "metadata": {
                "confidence": self.confidence,
                "frames_analyzed": self.frames_analyzed,
                "analysis_time_ms": self.analysis_time_ms,
            },
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Content Analysis Summary",
            f"========================",
            f"Resolution: {self.resolution[0]}x{self.resolution[1]} @ {self.fps:.2f} fps",
            f"Duration: {self.duration:.1f}s ({self.frame_count} frames)",
            f"",
            f"Content Type: {self.content_type.value}",
            f"Source Format: {self.source_format.value}",
            f"Era: {self.era or 'Unknown'}",
            f"",
            f"Quality Score: {self.quality_score*100:.0f}%",
            f"Degradation: {', '.join(d.value for d in self.degradation_types) or 'None detected'}",
            f"Noise Level: {self.noise_profile.overall_level:.1f}%",
            f"",
            f"Scenes: {self.scene_count}",
            f"Faces: {'Yes' if self.has_faces else 'No'} ({self.face_count_estimate} estimated)",
            f"",
            f"Recommended Preset: {self.recommended_preset}",
            f"Recommended Scale: {self.recommended_scale}x",
            f"Recommended Model: {self.recommended_model}",
        ]
        return "\n".join(lines)


@dataclass
class AnalyzerConfig:
    """Configuration for ContentAnalyzer."""
    # Sampling
    sample_rate: int = 100          # Analyze every Nth frame
    max_samples: int = 50           # Maximum frames to analyze
    quick_sample_count: int = 10    # Frames for quick analysis

    # Feature toggles
    enable_face_detection: bool = True
    enable_scene_detection: bool = True
    enable_noise_profiling: bool = True
    enable_film_stock_detection: bool = True
    enable_quality_scoring: bool = True

    # Thresholds
    scene_threshold: float = 0.3    # Scene change sensitivity
    noise_threshold: float = 15.0   # Noise detection threshold
    quality_threshold: float = 0.6  # Minimum quality for "good"


# =============================================================================
# Main ContentAnalyzer Class
# =============================================================================

class ContentAnalyzer:
    """Unified content analyzer combining all analysis capabilities.

    Uses lazy loading to defer initialization of heavy components
    (OpenCV, neural networks, etc.) until actually needed.

    Example:
        >>> analyzer = ContentAnalyzer()
        >>>
        >>> # Full analysis
        >>> analysis = analyzer.analyze("video.mp4")
        >>> print(analysis.summary())
        >>>
        >>> # Quick analysis for auto-preset selection
        >>> quick = analyzer.quick_analyze("video.mp4")
        >>> print(f"Recommended: {quick.recommended_preset}")
        >>>
        >>> # Just scene detection
        >>> scenes = analyzer.detect_scenes("video.mp4")
        >>> print(f"Found {len(scenes)} scenes")
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initialize the content analyzer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AnalyzerConfig()

        # Lazy-loaded components (initialized on first use)
        self._frame_analyzer = None
        self._scene_detector = None
        self._scene_analyzer = None
        self._noise_profiler = None
        self._film_stock_detector = None
        self._quality_scorer = None
        self._scene_intelligence = None

        # Lazy-loaded dependencies
        self._cv2 = None
        self._np = None

    # =========================================================================
    # Lazy Loading Properties
    # =========================================================================

    def _ensure_cv2(self) -> bool:
        """Ensure OpenCV is available."""
        if self._cv2 is not None:
            return True
        try:
            import cv2
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            logger.warning("OpenCV not available - some analysis features disabled")
            return False

    @property
    def frame_analyzer(self):
        """Lazy-loaded FrameAnalyzer from analyzer module."""
        if self._frame_analyzer is None:
            try:
                from ..analyzer import FrameAnalyzer
                self._frame_analyzer = FrameAnalyzer(
                    sample_rate=self.config.sample_rate,
                    max_samples=self.config.max_samples,
                    enable_face_detection=self.config.enable_face_detection,
                )
            except ImportError as e:
                logger.warning(f"FrameAnalyzer not available: {e}")
        return self._frame_analyzer

    @property
    def scene_detector(self):
        """Lazy-loaded SceneDetector from scene_detection module."""
        if self._scene_detector is None:
            try:
                from ..scene_detection import SceneDetector
                self._scene_detector = SceneDetector(
                    histogram_threshold=self.config.scene_threshold,
                )
            except ImportError as e:
                logger.warning(f"SceneDetector not available: {e}")
        return self._scene_detector

    @property
    def scene_analyzer(self):
        """Lazy-loaded SceneAnalyzer from scene_detection module."""
        if self._scene_analyzer is None:
            try:
                from ..scene_detection import SceneAnalyzer
                self._scene_analyzer = SceneAnalyzer(
                    face_detection_enabled=self.config.enable_face_detection,
                )
            except ImportError as e:
                logger.warning(f"SceneAnalyzer not available: {e}")
        return self._scene_analyzer

    @property
    def noise_profiler(self):
        """Lazy-loaded NoiseProfiler from noise_profiler module."""
        if self._noise_profiler is None:
            try:
                from ..noise_profiler import NoiseProfiler
                self._noise_profiler = NoiseProfiler(
                    sample_frames=self.config.max_samples,
                )
            except ImportError as e:
                logger.warning(f"NoiseProfiler not available: {e}")
        return self._noise_profiler

    @property
    def film_stock_detector(self):
        """Lazy-loaded FilmStockDetector from film_stock_detector module."""
        if self._film_stock_detector is None:
            try:
                from ..film_stock_detector import FilmStockDetector
                self._film_stock_detector = FilmStockDetector(
                    sample_count=self.config.max_samples,
                )
            except ImportError as e:
                logger.warning(f"FilmStockDetector not available: {e}")
        return self._film_stock_detector

    @property
    def quality_scorer(self):
        """Lazy-loaded FrameQualityScorer from frame_quality_scorer module."""
        if self._quality_scorer is None:
            try:
                from ..frame_quality_scorer import FrameQualityScorer
                self._quality_scorer = FrameQualityScorer(
                    sample_rate=self.config.sample_rate,
                )
            except ImportError as e:
                logger.warning(f"FrameQualityScorer not available: {e}")
        return self._quality_scorer

    @property
    def scene_intelligence(self):
        """Lazy-loaded SceneIntelligence from scene_intelligence module."""
        if self._scene_intelligence is None:
            try:
                from ..scene_intelligence import SceneIntelligence
                self._scene_intelligence = SceneIntelligence(
                    enable_face_detection=self.config.enable_face_detection,
                    enable_text_detection=True,
                    enable_motion_analysis=True,
                    sample_rate=1.0 / self.config.sample_rate,
                )
            except ImportError as e:
                logger.warning(f"SceneIntelligence not available: {e}")
        return self._scene_intelligence

    # =========================================================================
    # Main Analysis Methods
    # =========================================================================

    def analyze(
        self,
        video_path: Union[str, Path],
        quick: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ContentAnalysis:
        """Full analysis of video content.

        Performs comprehensive analysis including:
        - Basic metadata extraction
        - Content type detection
        - Scene detection and analysis
        - Noise profiling
        - Film stock detection (if applicable)
        - Quality scoring
        - Processing recommendations

        Args:
            video_path: Path to video file
            quick: If True, perform faster analysis with fewer samples
            progress_callback: Optional callback(stage, progress) for updates

        Returns:
            ContentAnalysis with all detected information and recommendations
        """
        import time
        start_time = time.time()

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return ContentAnalysis()

        logger.info(f"Analyzing video: {video_path}")

        if quick:
            return self.quick_analyze(video_path, progress_callback)

        analysis = ContentAnalysis()

        # Stage 1: Basic metadata
        if progress_callback:
            progress_callback("metadata", 0.0)
        self._extract_metadata(video_path, analysis)
        if progress_callback:
            progress_callback("metadata", 1.0)

        # Stage 2: Frame analysis (content type, brightness, sharpness)
        if progress_callback:
            progress_callback("frame_analysis", 0.0)
        self._analyze_frames(video_path, analysis, progress_callback)
        if progress_callback:
            progress_callback("frame_analysis", 1.0)

        # Stage 3: Scene detection
        if self.config.enable_scene_detection:
            if progress_callback:
                progress_callback("scene_detection", 0.0)
            self._detect_and_analyze_scenes(video_path, analysis, progress_callback)
            if progress_callback:
                progress_callback("scene_detection", 1.0)

        # Stage 4: Noise profiling
        if self.config.enable_noise_profiling:
            if progress_callback:
                progress_callback("noise_profiling", 0.0)
            self._profile_noise(video_path, analysis)
            if progress_callback:
                progress_callback("noise_profiling", 1.0)

        # Stage 5: Film stock detection
        if self.config.enable_film_stock_detection:
            if progress_callback:
                progress_callback("film_stock", 0.0)
            self._detect_film_stock(video_path, analysis)
            if progress_callback:
                progress_callback("film_stock", 1.0)

        # Stage 6: Quality scoring
        if self.config.enable_quality_scoring:
            if progress_callback:
                progress_callback("quality_scoring", 0.0)
            self._score_quality(video_path, analysis)
            if progress_callback:
                progress_callback("quality_scoring", 1.0)

        # Stage 7: Generate recommendations
        if progress_callback:
            progress_callback("recommendations", 0.0)
        self._generate_recommendations(analysis)
        if progress_callback:
            progress_callback("recommendations", 1.0)

        # Finalize
        analysis.analysis_time_ms = (time.time() - start_time) * 1000
        analysis.confidence = self._calculate_confidence(analysis)

        logger.info(
            f"Analysis complete in {analysis.analysis_time_ms:.0f}ms: "
            f"{analysis.content_type.value}, {len(analysis.scenes)} scenes, "
            f"quality={analysis.quality_score*100:.0f}%"
        )

        return analysis

    def quick_analyze(
        self,
        video_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ContentAnalysis:
        """Fast analysis using sampling (for auto-preset selection).

        Performs a reduced analysis suitable for quickly determining
        appropriate processing presets without deep analysis.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(stage, progress)

        Returns:
            ContentAnalysis with basic info and recommendations
        """
        import time
        start_time = time.time()

        video_path = Path(video_path)
        analysis = ContentAnalysis()

        # Metadata (always needed)
        self._extract_metadata(video_path, analysis)

        # Quick frame analysis with reduced samples
        original_max = self.config.max_samples
        self.config.max_samples = self.config.quick_sample_count

        try:
            self._analyze_frames(video_path, analysis, progress_callback)
            self._quick_noise_estimate(video_path, analysis)
            self._generate_recommendations(analysis)
        finally:
            self.config.max_samples = original_max

        analysis.analysis_time_ms = (time.time() - start_time) * 1000
        analysis.confidence = 0.6  # Lower confidence for quick analysis

        logger.info(f"Quick analysis complete in {analysis.analysis_time_ms:.0f}ms")

        return analysis

    def detect_scenes(
        self,
        video_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[Scene]:
        """Detect scene boundaries in video.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(progress)

        Returns:
            List of detected Scene objects
        """
        video_path = Path(video_path)

        if self.scene_detector is None:
            logger.warning("Scene detector not available, using FFmpeg fallback")
            return self._detect_scenes_ffmpeg(video_path)

        try:
            # Use FFmpeg-based detection for efficiency
            raw_scenes = self.scene_detector.detect_scenes_ffmpeg(
                video_path,
                threshold=self.config.scene_threshold,
            )

            # Convert to our Scene type
            fps = self._get_fps(video_path)
            scenes = []
            for rs in raw_scenes:
                scene = Scene(
                    start_frame=rs.start_frame,
                    end_frame=rs.end_frame,
                    duration_frames=rs.duration_frames,
                    start_time=rs.start_frame / fps if fps > 0 else 0,
                    end_time=rs.end_frame / fps if fps > 0 else 0,
                    scene_type=rs.scene_type.name.lower() if hasattr(rs, 'scene_type') else "unknown",
                    avg_brightness=getattr(rs, 'avg_brightness', 128.0),
                    avg_motion=getattr(rs, 'avg_motion', 0.0),
                    has_faces=getattr(rs, 'has_faces', False),
                    quality_score=getattr(rs, 'quality_score', 0.5),
                )
                scenes.append(scene)

            return scenes

        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return self._detect_scenes_ffmpeg(video_path)

    def profile_noise(
        self,
        frames: List[Union[str, Path, np.ndarray]],
    ) -> NoiseProfile:
        """Analyze noise characteristics from frames.

        Args:
            frames: List of frame paths or numpy arrays

        Returns:
            NoiseProfile with noise analysis
        """
        profile = NoiseProfile()

        if not self._ensure_cv2():
            return profile

        cv2 = self._cv2

        noise_levels = []
        chroma_noises = []
        temporal_diffs = []
        prev_gray = None

        for frame in frames:
            # Load frame if path
            if isinstance(frame, (str, Path)):
                img = cv2.imread(str(frame))
            else:
                img = frame

            if img is None:
                continue

            # Convert to YUV
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]

            # Luminance noise (Laplacian variance)
            laplacian = cv2.Laplacian(y_channel, cv2.CV_64F)
            sigma = np.median(np.abs(laplacian)) / 0.6745
            noise_levels.append(min(100, sigma * 2))

            # Chroma noise
            u_lap = cv2.Laplacian(u_channel, cv2.CV_64F)
            v_lap = cv2.Laplacian(v_channel, cv2.CV_64F)
            u_sigma = np.median(np.abs(u_lap)) / 0.6745
            v_sigma = np.median(np.abs(v_lap)) / 0.6745
            chroma_noises.append(min(100, (u_sigma + v_sigma)))

            # Temporal noise
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                static_mask = diff < 15
                if np.sum(static_mask) > 100:
                    temporal_diffs.append(min(100, np.std(diff[static_mask]) * 4))
            prev_gray = gray

        # Aggregate
        if noise_levels:
            profile.luminance_noise = np.mean(noise_levels)
            profile.overall_level = profile.luminance_noise
        if chroma_noises:
            profile.chroma_noise = np.mean(chroma_noises)
        if temporal_diffs:
            profile.temporal_noise = np.mean(temporal_diffs)

        # Classify noise type
        profile.noise_type = self._classify_noise_type(profile)

        # Recommendations
        self._determine_noise_recommendations(profile)

        return profile

    # =========================================================================
    # Internal Analysis Methods
    # =========================================================================

    def _extract_metadata(self, video_path: Path, analysis: ContentAnalysis) -> None:
        """Extract basic video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
            )
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                {}
            )

            # Resolution
            analysis.resolution = (
                int(video_stream.get('width', 0)),
                int(video_stream.get('height', 0)),
            )

            # FPS
            fps_str = video_stream.get('r_frame_rate', '24/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                analysis.fps = num / den if den else 24.0
            else:
                analysis.fps = float(fps_str) if fps_str else 24.0

            # Duration and frame count
            analysis.duration = float(data.get('format', {}).get('duration', 0))
            analysis.frame_count = int(analysis.duration * analysis.fps)

            # Codec
            analysis.codec = video_stream.get('codec_name', 'unknown')

            # Bitrate
            bitrate = data.get('format', {}).get('bit_rate', 0)
            analysis.bitrate = int(bitrate) if bitrate else 0

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

    def _analyze_frames(
        self,
        video_path: Path,
        analysis: ContentAnalysis,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Analyze sample frames for content detection."""
        if self.frame_analyzer is None:
            logger.warning("FrameAnalyzer not available, using basic analysis")
            self._basic_frame_analysis(video_path, analysis)
            return

        try:
            video_analysis = self.frame_analyzer.analyze_video(video_path)

            # Transfer results
            analysis.avg_brightness = video_analysis.avg_brightness
            analysis.avg_contrast = video_analysis.avg_contrast
            analysis.avg_sharpness = video_analysis.avg_sharpness

            # Content type from primary content
            content_map = {
                'ANIMATION': ContentType.ANIMATION,
                'FACE_PORTRAIT': ContentType.LIVE_ACTION,
                'FACE_GROUP': ContentType.LIVE_ACTION,
                'LANDSCAPE': ContentType.DOCUMENTARY,
                'LOW_LIGHT': ContentType.LIVE_ACTION,
                'HIGH_CONTRAST': ContentType.FILM,
            }
            primary_name = video_analysis.primary_content.name
            analysis.content_type = content_map.get(primary_name, ContentType.UNKNOWN)

            # Degradation types
            for deg in video_analysis.degradation_types:
                deg_map = {
                    'LIGHT_NOISE': DegradationType.LIGHT_NOISE,
                    'HEAVY_NOISE': DegradationType.HEAVY_NOISE,
                    'FILM_GRAIN': DegradationType.FILM_GRAIN,
                    'BLUR': DegradationType.BLUR,
                    'COMPRESSION_ARTIFACTS': DegradationType.COMPRESSION_ARTIFACTS,
                    'SCRATCHES': DegradationType.SCRATCHES,
                    'INTERLACING': DegradationType.INTERLACING,
                    'COLOR_FADE': DegradationType.COLOR_FADE,
                }
                if deg.name in deg_map:
                    analysis.degradation_types.append(deg_map[deg.name])

            # Face info
            analysis.has_faces = video_analysis.face_frame_ratio > 0.1
            analysis.face_count_estimate = int(
                video_analysis.face_frame_ratio * video_analysis.sample_count * 2
            )

            analysis.frames_analyzed = video_analysis.sample_count

        except Exception as e:
            logger.warning(f"Frame analysis failed: {e}")
            self._basic_frame_analysis(video_path, analysis)

    def _basic_frame_analysis(self, video_path: Path, analysis: ContentAnalysis) -> None:
        """Basic frame analysis fallback using ffprobe."""
        # Use ffprobe signalstats for basic metrics
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-f', 'lavfi',
            '-i', f"movie='{video_path}',select='eq(n,0)+eq(n,100)',signalstats",
            '-show_entries', 'frame_tags',
            '-print_format', 'json'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                frames = data.get('frames', [])
                if frames:
                    tags = frames[0].get('tags', {})
                    analysis.avg_brightness = float(
                        tags.get('lavfi.signalstats.YAVG', 128)
                    )
        except Exception:
            pass

        analysis.frames_analyzed = 1

    def _detect_and_analyze_scenes(
        self,
        video_path: Path,
        analysis: ContentAnalysis,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Detect scenes and analyze them."""
        scenes = self.detect_scenes(video_path)
        analysis.scenes = scenes
        analysis.scene_count = len(scenes)

        # Aggregate face info from scenes
        if scenes:
            face_scenes = sum(1 for s in scenes if s.has_faces)
            if face_scenes > len(scenes) * 0.3:
                analysis.has_faces = True

    def _detect_scenes_ffmpeg(self, video_path: Path) -> List[Scene]:
        """FFmpeg-based scene detection fallback."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'frame=pts_time',
            '-of', 'json',
            '-f', 'lavfi',
            f"movie='{video_path}',select='gt(scene,{self.config.scene_threshold})'"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            fps = self._get_fps(video_path)
            total_frames = int(self._get_duration(video_path) * fps)

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                timestamps = [
                    float(f.get('pts_time', 0))
                    for f in data.get('frames', [])
                ]
                boundaries = [int(ts * fps) for ts in timestamps]
            else:
                boundaries = []

            # Ensure start and end
            if not boundaries or boundaries[0] != 0:
                boundaries.insert(0, 0)
            if boundaries[-1] != total_frames - 1:
                boundaries.append(total_frames - 1)

            # Build scenes
            scenes = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1] - 1 if i < len(boundaries) - 2 else boundaries[i + 1]
                scenes.append(Scene(
                    start_frame=start,
                    end_frame=end,
                    start_time=start / fps if fps > 0 else 0,
                    end_time=end / fps if fps > 0 else 0,
                ))

            return scenes

        except Exception as e:
            logger.warning(f"FFmpeg scene detection failed: {e}")
            return [Scene(start_frame=0, end_frame=max(0, int(self._get_duration(video_path) * 24)))]

    def _profile_noise(self, video_path: Path, analysis: ContentAnalysis) -> None:
        """Profile noise characteristics."""
        if self.noise_profiler is None:
            self._quick_noise_estimate(video_path, analysis)
            return

        try:
            noise_result = self.noise_profiler.analyze_video(video_path)

            analysis.noise_profile = NoiseProfile(
                overall_level=noise_result.overall_level,
                luminance_noise=noise_result.characteristics.luminance_noise,
                chroma_noise=noise_result.characteristics.chroma_noise,
                temporal_noise=noise_result.characteristics.temporal_noise,
                noise_type=noise_result.dominant_type.value,
                recommended_denoiser=noise_result.recommended_denoiser.value,
                recommended_strength=noise_result.recommended_strength,
                preserve_grain=noise_result.preserve_grain,
            )

        except Exception as e:
            logger.warning(f"Noise profiling failed: {e}")
            self._quick_noise_estimate(video_path, analysis)

    def _quick_noise_estimate(self, video_path: Path, analysis: ContentAnalysis) -> None:
        """Quick noise estimation from avg noise in frame analysis."""
        # Use avg_noise from frame analysis if available
        if self.frame_analyzer and hasattr(self.frame_analyzer, '_last_metrics'):
            avg_noise = getattr(self.frame_analyzer, 'avg_noise', 0.3)
            analysis.noise_profile.overall_level = avg_noise * 100
            analysis.noise_profile.luminance_noise = avg_noise * 100

        # Determine type from degradation
        if DegradationType.FILM_GRAIN in analysis.degradation_types:
            analysis.noise_profile.noise_type = "film_grain"
            analysis.noise_profile.preserve_grain = True
        elif DegradationType.HEAVY_NOISE in analysis.degradation_types:
            analysis.noise_profile.noise_type = "gaussian"
        elif DegradationType.COMPRESSION_ARTIFACTS in analysis.degradation_types:
            analysis.noise_profile.noise_type = "compression"

    def _detect_film_stock(self, video_path: Path, analysis: ContentAnalysis) -> None:
        """Detect film stock type if content appears to be film."""
        # Only analyze if content looks like film
        if analysis.content_type not in (ContentType.FILM, ContentType.UNKNOWN):
            # Check for film-like characteristics
            if DegradationType.FILM_GRAIN not in analysis.degradation_types:
                return

        if self.film_stock_detector is None:
            return

        try:
            film_result = self.film_stock_detector.analyze(video_path)

            if film_result.confidence > 0.3:
                analysis.film_stock = film_result.detected_stock.value
                analysis.film_stock_confidence = film_result.confidence
                analysis.era = film_result.era.value.replace("_", " ").title()
                analysis.content_type = ContentType.FILM

                if film_result.color_profile:
                    analysis.color_profile = {
                        "red_shift": film_result.color_profile.red_shift,
                        "green_shift": film_result.color_profile.green_shift,
                        "blue_shift": film_result.color_profile.blue_shift,
                        "saturation": film_result.color_profile.saturation_factor,
                    }
                    analysis.enable_color_correction = film_result.fading_detected

        except Exception as e:
            logger.warning(f"Film stock detection failed: {e}")

    def _score_quality(self, video_path: Path, analysis: ContentAnalysis) -> None:
        """Score overall video quality."""
        if self.quality_scorer is None:
            self._estimate_quality_score(analysis)
            return

        try:
            quality_report = self.quality_scorer.analyze_video(video_path)
            analysis.quality_score = quality_report.average_score / 100

            # Add degradation types from quality issues
            issue_map = {
                'blur': DegradationType.BLUR,
                'noise': DegradationType.HEAVY_NOISE,
                'blocking': DegradationType.BLOCKING,
                'banding': DegradationType.BANDING,
                'interlacing': DegradationType.INTERLACING,
            }

            for issue, count in quality_report.issue_counts.items():
                if count > 0 and issue in issue_map:
                    deg_type = issue_map[issue]
                    if deg_type not in analysis.degradation_types:
                        analysis.degradation_types.append(deg_type)

        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")
            self._estimate_quality_score(analysis)

    def _estimate_quality_score(self, analysis: ContentAnalysis) -> None:
        """Estimate quality score from available metrics."""
        score = 1.0

        # Penalize for degradation
        score -= len(analysis.degradation_types) * 0.1

        # Penalize for high noise
        score -= analysis.noise_profile.overall_level / 200

        # Penalize for low resolution
        width = analysis.resolution[0]
        if width < 480:
            score -= 0.2
        elif width < 720:
            score -= 0.1

        analysis.quality_score = max(0.0, min(1.0, score))

    def _generate_recommendations(self, analysis: ContentAnalysis) -> None:
        """Generate processing recommendations based on analysis."""
        processors = []

        # Determine preset
        if analysis.content_type == ContentType.ANIMATION:
            analysis.recommended_preset = "anime"
            analysis.recommended_model = "realesrgan-x4plus-anime"
        elif analysis.content_type == ContentType.FILM:
            analysis.recommended_preset = "film"
            analysis.recommended_model = "realesrgan-x4plus"
        elif analysis.source_format in (SourceFormat.VHS, SourceFormat.BETAMAX):
            analysis.recommended_preset = "vhs"
            processors.append("vhs_restoration")
        elif analysis.quality_score > 0.7:
            analysis.recommended_preset = "light"
        elif analysis.quality_score < 0.4:
            analysis.recommended_preset = "heavy"
        else:
            analysis.recommended_preset = "balanced"

        # Scale factor
        width = analysis.resolution[0]
        if width < 480:
            analysis.recommended_scale = 4
        elif width < 720:
            analysis.recommended_scale = 4
        elif width < 1080:
            analysis.recommended_scale = 2
        else:
            analysis.recommended_scale = 2

        # Denoise strength
        noise_level = analysis.noise_profile.overall_level
        if noise_level > 50:
            analysis.recommended_denoise = 0.8
            processors.append("temporal_denoise")
        elif noise_level > 30:
            analysis.recommended_denoise = 0.5
        elif noise_level > 15:
            analysis.recommended_denoise = 0.3
        else:
            analysis.recommended_denoise = 0.0

        # Face restoration
        if analysis.has_faces and analysis.face_count_estimate > 0:
            analysis.enable_face_restoration = True
            processors.append("face_restore")

        # Scratch removal for film
        if (DegradationType.SCRATCHES in analysis.degradation_types or
            DegradationType.DUST_DEBRIS in analysis.degradation_types):
            analysis.enable_scratch_removal = True
            processors.append("defect_repair")

        # Deinterlace
        if DegradationType.INTERLACING in analysis.degradation_types:
            analysis.enable_deinterlace = True
            processors.append("deinterlace")

        # Color correction for faded film
        if analysis.enable_color_correction or DegradationType.COLOR_FADE in analysis.degradation_types:
            analysis.enable_color_correction = True
            processors.append("color_correction")

        # Standard processors
        processors.extend(["upscale", "interpolation"])

        analysis.recommended_processors = processors

    def _calculate_confidence(self, analysis: ContentAnalysis) -> float:
        """Calculate overall analysis confidence."""
        confidence = 0.5

        # More frames analyzed = higher confidence
        if analysis.frames_analyzed >= self.config.max_samples:
            confidence += 0.2
        elif analysis.frames_analyzed >= self.config.max_samples // 2:
            confidence += 0.1

        # Film stock detection confidence
        if analysis.film_stock_confidence > 0.5:
            confidence += 0.1

        # Scene detection
        if analysis.scene_count > 1:
            confidence += 0.1

        return min(1.0, confidence)

    def _classify_noise_type(self, profile: NoiseProfile) -> str:
        """Classify noise type from profile."""
        if profile.overall_level < 5:
            return "minimal"
        if profile.temporal_noise > profile.luminance_noise:
            return "temporal"
        if profile.chroma_noise > profile.luminance_noise * 1.5:
            return "chroma"
        return "gaussian"

    def _determine_noise_recommendations(self, profile: NoiseProfile) -> None:
        """Determine denoising recommendations."""
        if profile.overall_level < 5:
            profile.recommended_denoiser = "none"
            profile.recommended_strength = 0.0
        elif profile.noise_type == "film_grain":
            profile.recommended_denoiser = "grain_preserve"
            profile.recommended_strength = min(1.0, profile.overall_level / 50)
            profile.preserve_grain = True
        elif profile.noise_type == "temporal":
            profile.recommended_denoiser = "temporal"
            profile.recommended_strength = min(1.0, profile.overall_level / 40)
        elif profile.overall_level > 30:
            profile.recommended_denoiser = "aggressive"
            profile.recommended_strength = min(1.0, profile.overall_level / 60)
        else:
            profile.recommended_denoiser = "light"
            profile.recommended_strength = profile.overall_level / 40

    def _get_fps(self, video_path: Path) -> float:
        """Get video FPS."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-print_format', 'json',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [{}])
                if streams:
                    fps_str = streams[0].get('r_frame_rate', '24/1')
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        return num / den if den else 24.0
                    return float(fps_str)
        except Exception:
            pass
        return 24.0

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-print_format', 'json',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
        except Exception:
            pass
        return 0.0


# =============================================================================
# Factory and Convenience Functions
# =============================================================================

def create_content_analyzer(
    sample_rate: int = 100,
    max_samples: int = 50,
    enable_face_detection: bool = True,
    enable_scene_detection: bool = True,
    enable_noise_profiling: bool = True,
    enable_film_stock_detection: bool = True,
) -> ContentAnalyzer:
    """Factory function to create a configured ContentAnalyzer.

    Args:
        sample_rate: Analyze every Nth frame
        max_samples: Maximum frames to analyze
        enable_face_detection: Enable face detection
        enable_scene_detection: Enable scene detection
        enable_noise_profiling: Enable noise profiling
        enable_film_stock_detection: Enable film stock detection

    Returns:
        Configured ContentAnalyzer instance
    """
    config = AnalyzerConfig(
        sample_rate=sample_rate,
        max_samples=max_samples,
        enable_face_detection=enable_face_detection,
        enable_scene_detection=enable_scene_detection,
        enable_noise_profiling=enable_noise_profiling,
        enable_film_stock_detection=enable_film_stock_detection,
    )
    return ContentAnalyzer(config)


def analyze_video(
    video_path: Union[str, Path],
    quick: bool = False,
) -> ContentAnalysis:
    """Convenience function to analyze a video.

    Args:
        video_path: Path to video file
        quick: If True, perform quick analysis

    Returns:
        ContentAnalysis with all detected information
    """
    analyzer = ContentAnalyzer()
    return analyzer.analyze(video_path, quick=quick)


def quick_analyze(video_path: Union[str, Path]) -> ContentAnalysis:
    """Convenience function for quick video analysis.

    Args:
        video_path: Path to video file

    Returns:
        ContentAnalysis with basic info and recommendations
    """
    analyzer = ContentAnalyzer()
    return analyzer.quick_analyze(video_path)
