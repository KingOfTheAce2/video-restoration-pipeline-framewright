"""Smart auto-detection engine for FrameWright.

Analyzes video content to automatically determine:
- Content type (film, animation, home video, etc.)
- Degradation profile (noise, compression, scratches, etc.)
- Optimal processing pipeline and settings
- Hardware requirements

This enables "it just works" mode where users only need to
provide the input video.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import subprocess
import json
import tempfile

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Detected content types."""
    FILM = "film"                    # Classic film footage
    ANIMATION = "animation"          # Cartoons, anime
    HOME_VIDEO = "home_video"        # Personal recordings
    DOCUMENTARY = "documentary"      # Documentary footage
    MUSIC_VIDEO = "music_video"      # Music videos
    NEWS = "news"                    # News/broadcast
    SPORTS = "sports"                # Sports footage
    SURVEILLANCE = "surveillance"    # Security camera
    UNKNOWN = "unknown"


class DegradationType(Enum):
    """Types of video degradation."""
    NOISE = "noise"
    COMPRESSION = "compression"
    SCRATCHES = "scratches"
    DUST = "dust"
    FLICKER = "flicker"
    COLOR_FADE = "color_fade"
    BLUR = "blur"
    INTERLACING = "interlacing"
    FRAME_DAMAGE = "frame_damage"
    LOW_RESOLUTION = "low_resolution"


class Era(Enum):
    """Estimated era of the footage."""
    SILENT_ERA = "silent_era"        # Pre-1930
    EARLY_SOUND = "early_sound"      # 1930-1950
    CLASSIC = "classic"              # 1950-1970
    MODERN_FILM = "modern_film"      # 1970-2000
    DIGITAL = "digital"              # 2000+
    UNKNOWN = "unknown"


@dataclass
class ContentProfile:
    """Profile describing the content characteristics."""
    content_type: ContentType = ContentType.UNKNOWN
    era: Era = Era.UNKNOWN
    has_faces: bool = False
    face_percentage: float = 0.0  # % of frames with faces
    is_black_and_white: bool = False
    is_sepia: bool = False
    has_subtitles: bool = False
    has_watermark: bool = False
    dominant_colors: List[str] = field(default_factory=list)
    scene_complexity: float = 0.5  # 0-1 scale
    motion_intensity: float = 0.5  # 0-1 scale

    def summary(self) -> str:
        """Get human-readable summary."""
        parts = [f"{self.content_type.value.replace('_', ' ').title()}"]
        if self.era != Era.UNKNOWN:
            parts.append(f"({self.era.value.replace('_', ' ')})")
        if self.is_black_and_white:
            parts.append("B&W")
        if self.has_faces:
            parts.append(f"{self.face_percentage:.0f}% faces")
        return " | ".join(parts)


@dataclass
class DegradationProfile:
    """Profile describing detected degradation."""
    degradations: List[DegradationType] = field(default_factory=list)
    noise_level: float = 0.0        # 0-1, higher = more noise
    compression_level: float = 0.0  # 0-1, higher = more compressed
    scratch_density: float = 0.0    # 0-1, higher = more scratches
    dust_density: float = 0.0       # 0-1, higher = more dust
    flicker_intensity: float = 0.0  # 0-1, higher = more flicker
    blur_amount: float = 0.0        # 0-1, higher = more blur
    color_accuracy: float = 1.0     # 0-1, lower = more faded
    frame_damage_ratio: float = 0.0 # % of frames with damage
    estimated_bitrate_kbps: int = 0
    estimated_qp: Optional[int] = None

    @property
    def severity(self) -> str:
        """Overall degradation severity."""
        score = (
            self.noise_level * 0.25 +
            self.compression_level * 0.2 +
            self.scratch_density * 0.15 +
            self.dust_density * 0.1 +
            self.flicker_intensity * 0.1 +
            self.blur_amount * 0.1 +
            (1 - self.color_accuracy) * 0.1
        )
        if score < 0.2:
            return "minimal"
        elif score < 0.4:
            return "light"
        elif score < 0.6:
            return "moderate"
        elif score < 0.8:
            return "heavy"
        else:
            return "severe"

    @property
    def primary_issues(self) -> List[str]:
        """Get top 3 issues to address."""
        issues = []
        if self.noise_level > 0.3:
            issues.append(("noise", self.noise_level))
        if self.compression_level > 0.3:
            issues.append(("compression", self.compression_level))
        if self.scratch_density > 0.2:
            issues.append(("scratches", self.scratch_density))
        if self.dust_density > 0.2:
            issues.append(("dust", self.dust_density))
        if self.flicker_intensity > 0.3:
            issues.append(("flicker", self.flicker_intensity))
        if self.blur_amount > 0.3:
            issues.append(("blur", self.blur_amount))
        if self.color_accuracy < 0.7:
            issues.append(("color fade", 1 - self.color_accuracy))

        # Sort by severity and return top 3
        issues.sort(key=lambda x: x[1], reverse=True)
        return [i[0] for i in issues[:3]]


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    video_path: Path
    # Video metadata
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    codec: str = ""
    bitrate_kbps: int = 0
    # Profiles
    content: ContentProfile = field(default_factory=ContentProfile)
    degradation: DegradationProfile = field(default_factory=DegradationProfile)
    # Recommendations
    recommended_preset: str = "balanced"
    recommended_stages: List[str] = field(default_factory=list)
    recommended_settings: Dict[str, Any] = field(default_factory=dict)
    # Warnings
    warnings: List[str] = field(default_factory=list)

    @property
    def resolution(self) -> str:
        """Get resolution string."""
        return f"{self.width}x{self.height}"

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration."""
        hours, remainder = divmod(int(self.duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def size_category(self) -> str:
        """Categorize video size."""
        if self.width >= 3840:
            return "4K"
        elif self.width >= 1920:
            return "1080p"
        elif self.width >= 1280:
            return "720p"
        elif self.width >= 854:
            return "480p"
        else:
            return "SD"


class SmartAnalyzer:
    """Smart video analyzer for automatic detection.

    Analyzes video content and degradation to determine
    optimal restoration settings automatically.

    Example:
        >>> analyzer = SmartAnalyzer()
        >>> result = analyzer.analyze("old_film.mp4")
        >>> print(f"Detected: {result.content.summary()}")
        >>> print(f"Issues: {result.degradation.primary_issues}")
        >>> print(f"Recommended: {result.recommended_preset}")
    """

    # Sample rates for different analysis types
    CONTENT_SAMPLE_RATE = 100  # Every 100th frame
    DEGRADATION_SAMPLE_RATE = 50  # Every 50th frame
    MAX_SAMPLES = 100

    def __init__(self, enable_gpu: bool = True):
        """Initialize analyzer.

        Args:
            enable_gpu: Use GPU for analysis if available
        """
        self.enable_gpu = enable_gpu
        self._face_cascade = None

    def analyze(
        self,
        video_path: Path,
        quick: bool = False,
    ) -> AnalysisResult:
        """Analyze video for content and degradation.

        Args:
            video_path: Path to video file
            quick: If True, use faster but less accurate analysis

        Returns:
            AnalysisResult with detected characteristics
        """
        video_path = Path(video_path)
        result = AnalysisResult(video_path=video_path)

        # Get metadata
        self._analyze_metadata(video_path, result)

        # Analyze content
        if CV2_AVAILABLE:
            sample_rate = self.CONTENT_SAMPLE_RATE * (3 if quick else 1)
            max_samples = self.MAX_SAMPLES // (3 if quick else 1)
            self._analyze_content(video_path, result, sample_rate, max_samples)
            self._analyze_degradation(video_path, result, sample_rate, max_samples)

        # Detect missing frames
        self._detect_missing_frames(video_path, result)

        # Generate recommendations
        self._generate_recommendations(result)

        return result

    def _analyze_metadata(self, video_path: Path, result: AnalysisResult) -> None:
        """Extract video metadata using FFprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            data = json.loads(output)

            # Find video stream
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    result.width = int(stream.get("width", 0))
                    result.height = int(stream.get("height", 0))
                    result.codec = stream.get("codec_name", "")

                    # Parse frame rate
                    fps_str = stream.get("r_frame_rate", "24/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        result.fps = float(num) / float(den) if float(den) > 0 else 24.0
                    else:
                        result.fps = float(fps_str)

                    # Get frame count
                    result.total_frames = int(stream.get("nb_frames", 0))
                    break

            # Get duration and bitrate from format
            fmt = data.get("format", {})
            result.duration_seconds = float(fmt.get("duration", 0))
            result.bitrate_kbps = int(fmt.get("bit_rate", 0)) // 1000

            # Calculate frames if not available
            if result.total_frames == 0 and result.duration_seconds > 0:
                result.total_frames = int(result.duration_seconds * result.fps)

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

    def _analyze_content(
        self,
        video_path: Path,
        result: AnalysisResult,
        sample_rate: int,
        max_samples: int,
    ) -> None:
        """Analyze content characteristics."""
        if not CV2_AVAILABLE:
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        frame_idx = 0
        samples_analyzed = 0
        face_frames = 0
        bw_frames = 0
        sepia_frames = 0
        motion_scores = []

        prev_frame = None

        try:
            while samples_analyzed < max_samples:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    # Check for faces
                    if self._detect_faces(frame):
                        face_frames += 1

                    # Check color characteristics
                    if self._is_grayscale(frame):
                        bw_frames += 1
                    elif self._is_sepia(frame):
                        sepia_frames += 1

                    # Calculate motion
                    if prev_frame is not None:
                        motion = self._calculate_motion(prev_frame, frame)
                        motion_scores.append(motion)

                    prev_frame = frame.copy()
                    samples_analyzed += 1

                frame_idx += 1

        finally:
            cap.release()

        if samples_analyzed > 0:
            result.content.has_faces = face_frames > samples_analyzed * 0.1
            result.content.face_percentage = (face_frames / samples_analyzed) * 100
            result.content.is_black_and_white = bw_frames > samples_analyzed * 0.8
            result.content.is_sepia = sepia_frames > samples_analyzed * 0.5

            if motion_scores:
                result.content.motion_intensity = np.mean(motion_scores)

        # Determine content type
        result.content.content_type = self._classify_content(result)

        # Estimate era
        result.content.era = self._estimate_era(result)

    def _analyze_degradation(
        self,
        video_path: Path,
        result: AnalysisResult,
        sample_rate: int,
        max_samples: int,
    ) -> None:
        """Analyze degradation characteristics."""
        if not CV2_AVAILABLE:
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        frame_idx = 0
        samples_analyzed = 0
        noise_scores = []
        blur_scores = []
        brightness_values = []
        prev_frame = None
        brightness_changes = []

        try:
            while samples_analyzed < max_samples:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Estimate noise
                    noise = self._estimate_noise(gray)
                    noise_scores.append(noise)

                    # Estimate blur
                    blur = self._estimate_blur(gray)
                    blur_scores.append(blur)

                    # Track brightness for flicker detection
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)

                    if prev_frame is not None:
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        brightness_change = abs(brightness - np.mean(prev_gray))
                        brightness_changes.append(brightness_change)

                    prev_frame = frame.copy()
                    samples_analyzed += 1

                frame_idx += 1

        finally:
            cap.release()

        if samples_analyzed > 0:
            # Normalize scores to 0-1
            result.degradation.noise_level = min(1.0, np.mean(noise_scores) / 50)
            result.degradation.blur_amount = 1 - min(1.0, np.mean(blur_scores) / 500)

            if brightness_changes:
                result.degradation.flicker_intensity = min(1.0, np.std(brightness_changes) / 30)

            # Estimate compression from bitrate
            if result.bitrate_kbps > 0:
                # Lower bitrate = more compression
                expected_bitrate = result.width * result.height * result.fps * 0.1 / 1000
                result.degradation.compression_level = max(0, 1 - (result.bitrate_kbps / expected_bitrate))
                result.degradation.estimated_bitrate_kbps = result.bitrate_kbps

        # Populate degradation types
        if result.degradation.noise_level > 0.3:
            result.degradation.degradations.append(DegradationType.NOISE)
        if result.degradation.compression_level > 0.3:
            result.degradation.degradations.append(DegradationType.COMPRESSION)
        if result.degradation.flicker_intensity > 0.3:
            result.degradation.degradations.append(DegradationType.FLICKER)
        if result.degradation.blur_amount > 0.3:
            result.degradation.degradations.append(DegradationType.BLUR)
        if result.width < 720:
            result.degradation.degradations.append(DegradationType.LOW_RESOLUTION)

    def _detect_missing_frames(self, video_path: Path, result: AnalysisResult) -> None:
        """Detect potential missing/damaged frames."""
        if not CV2_AVAILABLE:
            return

        # Quick scan for frame number gaps or duplicate frames
        # This is a simplified check
        if result.fps > 0 and result.duration_seconds > 0:
            expected_frames = int(result.fps * result.duration_seconds)
            if result.total_frames < expected_frames * 0.95:
                result.degradation.frame_damage_ratio = 1 - (result.total_frames / expected_frames)
                result.degradation.degradations.append(DegradationType.FRAME_DAMAGE)

    def _detect_faces(self, frame: np.ndarray) -> bool:
        """Detect if frame contains faces."""
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample for speed
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces = self._face_cascade.detectMultiScale(small, 1.1, 4, minSize=(30, 30))
        return len(faces) > 0

    def _is_grayscale(self, frame: np.ndarray) -> bool:
        """Check if frame is effectively grayscale."""
        if len(frame.shape) < 3:
            return True
        b, g, r = cv2.split(frame)
        diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
        diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))
        return diff_rg < 10 and diff_rb < 10

    def _is_sepia(self, frame: np.ndarray) -> bool:
        """Check if frame has sepia tones."""
        if len(frame.shape) < 3:
            return False
        b, g, r = cv2.split(frame)
        # Sepia typically has warm tones: R > G > B
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        return r_mean > g_mean > b_mean and (r_mean - b_mean) > 20

    def _calculate_motion(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """Calculate motion between frames (0-1 scale)."""
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        return np.mean(diff) / 255.0

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level in frame."""
        # Use Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # High frequency noise increases variance
        sigma = laplacian.var() ** 0.5
        return sigma

    def _estimate_blur(self, gray: np.ndarray) -> float:
        """Estimate blur amount (Laplacian variance method)."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def _classify_content(self, result: AnalysisResult) -> ContentType:
        """Classify content type based on characteristics."""
        # Animation detection (flat colors, high edge contrast)
        if result.content.motion_intensity > 0.7 and not result.content.has_faces:
            return ContentType.ANIMATION

        # High face percentage suggests narrative content
        if result.content.face_percentage > 50:
            if result.content.is_black_and_white:
                return ContentType.FILM
            return ContentType.HOME_VIDEO

        # Low resolution + old characteristics
        if result.width < 720 and result.content.is_black_and_white:
            return ContentType.FILM

        # High motion could be sports
        if result.content.motion_intensity > 0.6:
            return ContentType.SPORTS

        return ContentType.UNKNOWN

    def _estimate_era(self, result: AnalysisResult) -> Era:
        """Estimate era of the footage."""
        # B&W with low resolution likely older
        if result.content.is_black_and_white:
            if result.width < 400:
                return Era.SILENT_ERA
            elif result.width < 640:
                return Era.EARLY_SOUND
            else:
                return Era.CLASSIC

        # Color but SD resolution
        if result.width < 720:
            return Era.MODERN_FILM

        # HD or higher
        if result.width >= 1280:
            return Era.DIGITAL

        return Era.UNKNOWN

    def _generate_recommendations(self, result: AnalysisResult) -> None:
        """Generate processing recommendations based on analysis."""
        settings = {}
        stages = []
        warnings = []

        # Base preset selection
        severity = result.degradation.severity
        if severity in ("severe", "heavy"):
            result.recommended_preset = "ultimate"
        elif severity == "moderate":
            result.recommended_preset = "quality"
        elif severity == "light":
            result.recommended_preset = "balanced"
        else:
            result.recommended_preset = "fast"

        # Archive footage always benefits from ultimate
        if result.content.era in (Era.SILENT_ERA, Era.EARLY_SOUND, Era.CLASSIC):
            result.recommended_preset = "ultimate"

        # Build processing pipeline
        if DegradationType.COMPRESSION in result.degradation.degradations:
            stages.append("qp_artifact_removal")
            settings["enable_qp_artifact_removal"] = True

        if DegradationType.NOISE in result.degradation.degradations:
            stages.append("tap_denoise")
            settings["enable_tap_denoise"] = True
            settings["tap_strength"] = min(1.0, result.degradation.noise_level + 0.3)

        if DegradationType.FRAME_DAMAGE in result.degradation.degradations:
            stages.append("frame_generation")
            settings["enable_frame_generation"] = True
            warnings.append(f"Detected ~{result.degradation.frame_damage_ratio*100:.1f}% missing frames")

        # Super-resolution
        stages.append("super_resolution")
        if result.width < 720:
            settings["scale_factor"] = 4
        elif result.width < 1080:
            settings["scale_factor"] = 2
        else:
            settings["scale_factor"] = 2

        # Use diffusion SR for archive footage
        if result.content.era in (Era.SILENT_ERA, Era.EARLY_SOUND, Era.CLASSIC):
            settings["sr_model"] = "diffusion"

        # Face enhancement
        if result.content.has_faces and result.content.face_percentage > 20:
            stages.append("face_restoration")
            settings["auto_face_restore"] = True
            settings["face_model"] = "aesrgan"

        # Frame interpolation
        if result.fps < 25:
            stages.append("frame_interpolation")
            settings["enable_interpolation"] = True
            settings["target_fps"] = 30 if result.fps < 20 else 25

        # Temporal consistency
        if DegradationType.FLICKER in result.degradation.degradations:
            stages.append("temporal_consistency")
            settings["temporal_method"] = "hybrid"

        # Colorization
        if result.content.is_black_and_white:
            stages.append("colorization (optional)")
            settings["enable_colorization"] = False  # Requires user to provide references
            warnings.append("B&W footage detected - provide reference images for colorization")

        # Deduplication for old film
        if result.fps >= 24 and result.content.era in (Era.SILENT_ERA, Era.EARLY_SOUND):
            settings["enable_deduplication"] = True
            warnings.append("Old film detected - enabling deduplication for padded FPS")

        result.recommended_stages = stages
        result.recommended_settings = settings
        result.warnings = warnings


def analyze_video_smart(
    video_path: Path,
    quick: bool = False,
) -> AnalysisResult:
    """Convenience function for smart video analysis.

    Args:
        video_path: Path to video file
        quick: If True, use faster analysis

    Returns:
        AnalysisResult with detected characteristics and recommendations
    """
    analyzer = SmartAnalyzer()
    return analyzer.analyze(video_path, quick=quick)
