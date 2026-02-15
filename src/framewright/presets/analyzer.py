"""Video analyzer for detecting characteristics and suggesting presets."""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VideoEra(Enum):
    """Detected era of the video."""
    SILENT_FILM = "silent_film"  # Pre-1930
    EARLY_SOUND = "early_sound"  # 1930-1950
    GOLDEN_AGE = "golden_age"  # 1950-1970
    NEW_HOLLYWOOD = "new_hollywood"  # 1970-1985
    HOME_VIDEO = "home_video"  # 1985-2000
    DIGITAL_EARLY = "digital_early"  # 2000-2010
    MODERN = "modern"  # 2010+
    UNKNOWN = "unknown"


class VideoSource(Enum):
    """Detected source format."""
    FILM_35MM = "film_35mm"
    FILM_16MM = "film_16mm"
    FILM_8MM = "film_8mm"
    VHS = "vhs"
    BETAMAX = "betamax"
    HI8 = "hi8"
    LASERDISC = "laserdisc"
    DVD = "dvd"
    BROADCAST = "broadcast"
    DIGITAL = "digital"
    WEBCAM = "webcam"
    UNKNOWN = "unknown"


class DefectType(Enum):
    """Types of defects detected."""
    FILM_GRAIN = auto()
    SCRATCHES = auto()
    DUST = auto()
    FLICKERING = auto()
    INTERLACING = auto()
    COMPRESSION_ARTIFACTS = auto()
    COLOR_FADING = auto()
    VIGNETTING = auto()
    NOISE = auto()
    BLUR = auto()
    HEAD_SWITCHING = auto()
    DROPOUT = auto()
    TRACKING_ERRORS = auto()
    JITTER = auto()


@dataclass
class VideoCharacteristics:
    """Detected characteristics of a video."""
    # Basic info
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    codec: str = ""
    container: str = ""
    bitrate_kbps: int = 0

    # Color info
    is_color: bool = True
    is_grayscale: bool = False
    color_space: str = ""
    bit_depth: int = 8
    has_alpha: bool = False

    # Detected characteristics
    era: VideoEra = VideoEra.UNKNOWN
    source: VideoSource = VideoSource.UNKNOWN
    is_interlaced: bool = False
    field_order: str = ""  # "tff" or "bff"

    # Quality metrics (0-100)
    sharpness_score: float = 0.0
    noise_level: float = 0.0
    compression_score: float = 0.0
    stability_score: float = 0.0

    # Detected defects
    defects: List[DefectType] = field(default_factory=list)
    defect_severity: Dict[DefectType, float] = field(default_factory=dict)

    # Content analysis
    has_faces: bool = False
    face_count_avg: float = 0.0
    scene_count: int = 0
    avg_scene_duration: float = 0.0

    # Film-specific
    has_film_grain: bool = False
    grain_intensity: float = 0.0
    has_sprocket_damage: bool = False

    # Analog-specific
    has_vhs_artifacts: bool = False
    has_tracking_issues: bool = False

    # Audio info
    has_audio: bool = False
    audio_codec: str = ""
    audio_channels: int = 0
    audio_sample_rate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "codec": self.codec,
            "bitrate_kbps": self.bitrate_kbps,
            "is_color": self.is_color,
            "era": self.era.value,
            "source": self.source.value,
            "is_interlaced": self.is_interlaced,
            "quality_scores": {
                "sharpness": self.sharpness_score,
                "noise": self.noise_level,
                "compression": self.compression_score,
                "stability": self.stability_score,
            },
            "defects": [d.name for d in self.defects],
            "has_faces": self.has_faces,
            "has_audio": self.has_audio,
        }


class VideoAnalyzer:
    """Analyzes video to determine characteristics."""

    def __init__(
        self,
        sample_frames: int = 30,
        sample_interval: float = 0.0,  # 0 = auto
    ):
        self.sample_frames = sample_frames
        self.sample_interval = sample_interval

    def analyze(self, video_path: Path) -> VideoCharacteristics:
        """Analyze a video file."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        chars = VideoCharacteristics()

        # Get basic metadata from FFprobe
        self._analyze_metadata(video_path, chars)

        # Sample frames for visual analysis
        frames = self._sample_frames(video_path, chars)

        if frames:
            # Analyze visual characteristics
            self._analyze_color(frames, chars)
            self._analyze_quality(frames, chars)
            self._detect_defects(frames, chars)
            self._detect_source(frames, chars)
            self._detect_era(chars)
            self._detect_content(frames, chars)

        logger.info(f"Analyzed video: {video_path.name}")
        logger.info(f"  Era: {chars.era.value}, Source: {chars.source.value}")
        logger.info(f"  Defects: {[d.name for d in chars.defects]}")

        return chars

    def _analyze_metadata(self, path: Path, chars: VideoCharacteristics) -> None:
        """Extract metadata using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Find video stream
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        chars.width = int(stream.get("width", 0))
                        chars.height = int(stream.get("height", 0))
                        chars.codec = stream.get("codec_name", "")

                        # Parse fps
                        fps_str = stream.get("r_frame_rate", "0/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            chars.fps = float(num) / float(den) if float(den) > 0 else 0
                        else:
                            chars.fps = float(fps_str)

                        # Frame count
                        chars.total_frames = int(stream.get("nb_frames", 0))

                        # Interlacing
                        field_order = stream.get("field_order", "progressive")
                        chars.is_interlaced = field_order not in ("progressive", "unknown")
                        chars.field_order = field_order

                        # Color info
                        chars.color_space = stream.get("color_space", "")
                        chars.bit_depth = int(stream.get("bits_per_raw_sample", 8) or 8)

                    elif stream.get("codec_type") == "audio":
                        chars.has_audio = True
                        chars.audio_codec = stream.get("codec_name", "")
                        chars.audio_channels = int(stream.get("channels", 0))
                        chars.audio_sample_rate = int(stream.get("sample_rate", 0))

                # Format info
                fmt = data.get("format", {})
                chars.duration_seconds = float(fmt.get("duration", 0))
                chars.bitrate_kbps = int(fmt.get("bit_rate", 0)) // 1000
                chars.container = fmt.get("format_name", "")

                # Calculate frame count if not available
                if chars.total_frames == 0 and chars.fps > 0:
                    chars.total_frames = int(chars.duration_seconds * chars.fps)

        except Exception as e:
            logger.warning(f"FFprobe analysis failed: {e}")

    def _sample_frames(self, path: Path, chars: VideoCharacteristics) -> List[np.ndarray]:
        """Sample frames from video for analysis."""
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV required for frame analysis")
            return []

        frames = []
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            return frames

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = chars.total_frames or 1000

        # Calculate sample positions
        if self.sample_interval > 0:
            interval = int(self.sample_interval * chars.fps)
        else:
            interval = max(1, total // self.sample_frames)

        positions = list(range(0, total, interval))[:self.sample_frames]

        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def _analyze_color(self, frames: List[np.ndarray], chars: VideoCharacteristics) -> None:
        """Analyze color characteristics."""
        if not frames:
            return

        grayscale_count = 0

        for frame in frames:
            if len(frame.shape) == 2:
                grayscale_count += 1
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # Check if effectively grayscale (R ≈ G ≈ B)
                b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
                diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))

                if diff_rg < 5 and diff_rb < 5:
                    grayscale_count += 1

        chars.is_grayscale = grayscale_count > len(frames) * 0.8
        chars.is_color = not chars.is_grayscale

    def _analyze_quality(self, frames: List[np.ndarray], chars: VideoCharacteristics) -> None:
        """Analyze quality metrics."""
        try:
            import cv2
        except ImportError:
            return

        sharpness_scores = []
        noise_levels = []

        for frame in frames:
            # Sharpness via Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_scores.append(min(100, sharpness / 10))

            # Noise estimation via high-frequency content
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = np.abs(gray.astype(float) - blur.astype(float))
            noise = np.mean(diff)
            noise_levels.append(min(100, noise * 2))

        chars.sharpness_score = np.mean(sharpness_scores)
        chars.noise_level = np.mean(noise_levels)

        # Compression score based on blockiness
        chars.compression_score = self._estimate_compression_quality(frames)

        # Stability score would require temporal analysis
        chars.stability_score = 80.0  # Default

    def _estimate_compression_quality(self, frames: List[np.ndarray]) -> float:
        """Estimate compression quality (0=heavily compressed, 100=pristine)."""
        try:
            import cv2
        except ImportError:
            return 50.0

        blockiness_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Detect 8x8 block boundaries (JPEG/MPEG artifacts)
            h, w = gray.shape
            block_diffs = []

            for y in range(8, h - 8, 8):
                diff = np.abs(gray[y - 1, :].astype(float) - gray[y, :].astype(float))
                block_diffs.append(np.mean(diff))

            for x in range(8, w - 8, 8):
                diff = np.abs(gray[:, x - 1].astype(float) - gray[:, x].astype(float))
                block_diffs.append(np.mean(diff))

            avg_block_diff = np.mean(block_diffs) if block_diffs else 0
            blockiness_scores.append(avg_block_diff)

        avg_blockiness = np.mean(blockiness_scores) if blockiness_scores else 0
        # Higher blockiness = lower quality score
        return max(0, 100 - avg_blockiness * 5)

    def _detect_defects(self, frames: List[np.ndarray], chars: VideoCharacteristics) -> None:
        """Detect visual defects."""
        try:
            import cv2
        except ImportError:
            return

        # Noise detection
        if chars.noise_level > 30:
            chars.defects.append(DefectType.NOISE)
            chars.defect_severity[DefectType.NOISE] = chars.noise_level / 100

        # Film grain detection
        grain_score = self._detect_film_grain(frames)
        if grain_score > 0.3:
            chars.has_film_grain = True
            chars.grain_intensity = grain_score
            chars.defects.append(DefectType.FILM_GRAIN)
            chars.defect_severity[DefectType.FILM_GRAIN] = grain_score

        # Compression artifacts
        if chars.compression_score < 50:
            chars.defects.append(DefectType.COMPRESSION_ARTIFACTS)
            chars.defect_severity[DefectType.COMPRESSION_ARTIFACTS] = (100 - chars.compression_score) / 100

        # Blur detection
        if chars.sharpness_score < 20:
            chars.defects.append(DefectType.BLUR)
            chars.defect_severity[DefectType.BLUR] = (100 - chars.sharpness_score) / 100

        # Interlacing
        if chars.is_interlaced:
            chars.defects.append(DefectType.INTERLACING)
            chars.defect_severity[DefectType.INTERLACING] = 0.5

        # VHS artifacts detection
        vhs_score = self._detect_vhs_artifacts(frames)
        if vhs_score > 0.3:
            chars.has_vhs_artifacts = True
            chars.defects.append(DefectType.HEAD_SWITCHING)
            chars.defect_severity[DefectType.HEAD_SWITCHING] = vhs_score

        # Scratch detection for film
        scratch_score = self._detect_scratches(frames)
        if scratch_score > 0.2:
            chars.defects.append(DefectType.SCRATCHES)
            chars.defect_severity[DefectType.SCRATCHES] = scratch_score

    def _detect_film_grain(self, frames: List[np.ndarray]) -> float:
        """Detect film grain characteristics."""
        try:
            import cv2
        except ImportError:
            return 0.0

        grain_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Film grain has specific frequency characteristics
            # High-pass filter to isolate grain
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            diff = np.abs(gray.astype(float) - blur.astype(float))

            # Grain is typically uniform across the frame
            std_of_std = np.std([np.std(diff[i:i+32, j:j+32])
                                 for i in range(0, diff.shape[0]-32, 32)
                                 for j in range(0, diff.shape[1]-32, 32)])

            # Low variation in local noise = likely grain
            grain_score = 1.0 - min(1.0, std_of_std / 10)
            grain_scores.append(grain_score * (np.mean(diff) / 20))

        return min(1.0, np.mean(grain_scores))

    def _detect_vhs_artifacts(self, frames: List[np.ndarray]) -> float:
        """Detect VHS-specific artifacts."""
        try:
            import cv2
        except ImportError:
            return 0.0

        artifact_scores = []

        for frame in frames:
            h = frame.shape[0]

            # Check for head switching noise (bottom 10-20 lines)
            bottom = frame[int(h * 0.9):, :]
            rest = frame[:int(h * 0.9), :]

            bottom_noise = np.std(bottom)
            rest_noise = np.std(rest)

            if rest_noise > 0:
                ratio = bottom_noise / rest_noise
                if ratio > 1.5:
                    artifact_scores.append(min(1.0, (ratio - 1) / 2))
                else:
                    artifact_scores.append(0.0)
            else:
                artifact_scores.append(0.0)

        return np.mean(artifact_scores)

    def _detect_scratches(self, frames: List[np.ndarray]) -> float:
        """Detect film scratches."""
        try:
            import cv2
        except ImportError:
            return 0.0

        scratch_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Vertical edge detection for scratches
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_x = np.abs(sobel_x)

            # Look for strong vertical lines
            col_sums = np.sum(sobel_x, axis=0)
            threshold = np.mean(col_sums) + 2 * np.std(col_sums)
            scratch_count = np.sum(col_sums > threshold)

            scratch_scores.append(min(1.0, scratch_count / 10))

        return np.mean(scratch_scores)

    def _detect_source(self, frames: List[np.ndarray], chars: VideoCharacteristics) -> None:
        """Detect the likely source format."""
        # Use heuristics based on characteristics

        # VHS indicators
        if chars.has_vhs_artifacts or chars.has_tracking_issues:
            chars.source = VideoSource.VHS
            return

        # DVD indicators
        if chars.width == 720 and chars.height in (480, 576):
            if chars.compression_score > 70:
                chars.source = VideoSource.DVD
                return

        # Film indicators
        if chars.has_film_grain:
            if chars.width < 1000:
                chars.source = VideoSource.FILM_16MM
            elif chars.width < 2000:
                chars.source = VideoSource.FILM_35MM
            else:
                chars.source = VideoSource.FILM_35MM
            return

        # Digital indicators
        if chars.width >= 1920 and chars.compression_score > 80:
            chars.source = VideoSource.DIGITAL
            return

        # Broadcast indicators
        if chars.is_interlaced and chars.width in (720, 1920):
            chars.source = VideoSource.BROADCAST

    def _detect_era(self, chars: VideoCharacteristics) -> None:
        """Detect the likely era based on characteristics."""
        # Silent film
        if chars.is_grayscale and chars.fps < 20:
            chars.era = VideoEra.SILENT_FILM
            return

        # Early sound era
        if chars.is_grayscale and 20 <= chars.fps <= 25:
            chars.era = VideoEra.EARLY_SOUND
            return

        # VHS era
        if chars.source == VideoSource.VHS:
            chars.era = VideoEra.HOME_VIDEO
            return

        # Modern digital
        if chars.width >= 1920 and chars.bit_depth >= 8:
            if chars.compression_score > 90:
                chars.era = VideoEra.MODERN
            else:
                chars.era = VideoEra.DIGITAL_EARLY
            return

        # Default based on resolution
        if chars.width < 480:
            chars.era = VideoEra.GOLDEN_AGE
        elif chars.width < 720:
            chars.era = VideoEra.NEW_HOLLYWOOD
        else:
            chars.era = VideoEra.DIGITAL_EARLY

    def _detect_content(self, frames: List[np.ndarray], chars: VideoCharacteristics) -> None:
        """Detect content characteristics (faces, scenes)."""
        try:
            import cv2
        except ImportError:
            return

        # Face detection (simple cascade)
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            face_counts = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_counts.append(len(faces))

            chars.has_faces = sum(face_counts) > len(frames) * 0.1
            chars.face_count_avg = np.mean(face_counts) if face_counts else 0
        except Exception:
            pass
