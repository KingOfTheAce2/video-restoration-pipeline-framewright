"""AI Scene Intelligence for FrameWright.

Provides intelligent content-aware processing that adapts restoration
parameters based on what's actually in the frame.

Key principle: Different content needs different treatment.
- Faces need careful enhancement without plastic look
- Text/titles need sharpness preservation
- Landscapes can handle more aggressive processing
- Fast motion needs different temporal treatment
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
import json

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Primary content classification."""
    FACE_CLOSEUP = auto()
    FACE_MEDIUM = auto()
    GROUP_SHOT = auto()
    LANDSCAPE = auto()
    ARCHITECTURE = auto()
    TEXT_TITLE = auto()
    ACTION = auto()
    STATIC = auto()
    DOCUMENTARY = auto()
    ANIMATION = auto()
    UNKNOWN = auto()


class MotionLevel(Enum):
    """Motion intensity classification."""
    STATIC = "static"          # No motion (freeze frame, title card)
    MINIMAL = "minimal"        # Slight movement (talking head)
    MODERATE = "moderate"      # Normal movement
    HIGH = "high"              # Fast action
    EXTREME = "extreme"        # Very fast (sports, chase)


class LightingCondition(Enum):
    """Lighting condition classification."""
    BRIGHT = "bright"
    NORMAL = "normal"
    LOW_LIGHT = "low_light"
    HIGH_CONTRAST = "high_contrast"
    BACKLIT = "backlit"
    MIXED = "mixed"


@dataclass
class FaceRegion:
    """Detected face region in frame."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    is_profile: bool = False
    estimated_age: Optional[str] = None  # child, adult, elderly
    quality_score: float = 0.5


@dataclass
class TextRegion:
    """Detected text region in frame."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    text_type: str = "unknown"  # title, subtitle, sign, document
    is_period_font: bool = True  # Historical font style


@dataclass
class SceneAnalysis:
    """Complete analysis of a scene/shot."""
    frame_number: int
    timestamp: float

    # Content classification
    primary_content: ContentType = ContentType.UNKNOWN
    secondary_content: List[ContentType] = field(default_factory=list)

    # Motion analysis
    motion_level: MotionLevel = MotionLevel.MODERATE
    motion_direction: Optional[str] = None  # horizontal, vertical, zoom, etc.
    camera_movement: bool = False

    # Lighting
    lighting: LightingCondition = LightingCondition.NORMAL
    avg_brightness: float = 0.5
    contrast_ratio: float = 1.0

    # Detected regions
    faces: List[FaceRegion] = field(default_factory=list)
    text_regions: List[TextRegion] = field(default_factory=list)

    # Quality indicators
    blur_level: float = 0.0
    noise_level: float = 0.0
    has_artifacts: bool = False

    # Scene boundaries
    is_scene_start: bool = False
    is_scene_end: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "primary_content": self.primary_content.name,
            "motion_level": self.motion_level.value,
            "lighting": self.lighting.value,
            "face_count": len(self.faces),
            "text_regions": len(self.text_regions),
            "blur_level": self.blur_level,
            "noise_level": self.noise_level,
        }


@dataclass
class AdaptiveSettings:
    """Processing settings adapted for specific content."""
    # Enhancement
    sharpening: float = 0.3
    noise_reduction: float = 0.3
    detail_enhancement: float = 0.3

    # Face-specific
    face_enhancement: float = 0.5
    face_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Text-specific
    text_sharpening: float = 0.0  # Extra sharpening for text
    text_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Temporal
    temporal_smoothing: float = 0.5
    interpolation_quality: str = "medium"

    # Color
    color_correction: float = 0.3
    saturation_adjustment: float = 0.0

    # Region masks
    apply_regional: bool = False
    region_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpening": self.sharpening,
            "noise_reduction": self.noise_reduction,
            "detail_enhancement": self.detail_enhancement,
            "face_enhancement": self.face_enhancement,
            "temporal_smoothing": self.temporal_smoothing,
            "color_correction": self.color_correction,
        }


class SceneIntelligence:
    """AI-powered scene analysis and adaptive processing.

    Analyzes video content to provide intelligent, content-aware
    restoration that treats different elements appropriately.
    """

    def __init__(
        self,
        enable_face_detection: bool = True,
        enable_text_detection: bool = True,
        enable_motion_analysis: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize scene intelligence.

        Args:
            enable_face_detection: Detect faces for careful enhancement
            enable_text_detection: Detect text for sharpness preservation
            enable_motion_analysis: Analyze motion for temporal processing
            sample_rate: Fraction of frames to analyze (1.0 = all)
        """
        self.enable_face_detection = enable_face_detection
        self.enable_text_detection = enable_text_detection
        self.enable_motion_analysis = enable_motion_analysis
        self.sample_rate = sample_rate

        self._cv2 = None
        self._np = None
        self._face_cascade = None

    def _ensure_deps(self) -> bool:
        """Ensure OpenCV is available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            return False

    def _load_face_cascade(self) -> bool:
        """Load OpenCV face cascade."""
        if self._face_cascade is not None:
            return True

        if not self._ensure_deps():
            return False

        cv2 = self._cv2

        # Try to load Haar cascade
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        ]

        for path in cascade_paths:
            try:
                self._face_cascade = cv2.CascadeClassifier(path)
                if not self._face_cascade.empty():
                    return True
            except Exception:
                continue

        logger.warning("Face cascade not found, face detection disabled")
        return False

    def analyze_frame(self, frame, frame_number: int = 0, timestamp: float = 0.0) -> SceneAnalysis:
        """Analyze a single frame.

        Args:
            frame: Frame as numpy array or path
            frame_number: Frame index
            timestamp: Frame timestamp

        Returns:
            SceneAnalysis with detected content
        """
        if not self._ensure_deps():
            return SceneAnalysis(frame_number=frame_number, timestamp=timestamp)

        cv2 = self._cv2
        np = self._np

        # Load frame if path
        if isinstance(frame, (str, Path)):
            frame = cv2.imread(str(frame))
            if frame is None:
                return SceneAnalysis(frame_number=frame_number, timestamp=timestamp)

        analysis = SceneAnalysis(frame_number=frame_number, timestamp=timestamp)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Basic image statistics
        analysis.avg_brightness = np.mean(gray) / 255.0
        analysis.contrast_ratio = np.std(gray) / 128.0

        # Classify lighting
        analysis.lighting = self._classify_lighting(gray)

        # Detect noise level
        analysis.noise_level = self._estimate_noise(gray)

        # Detect blur level
        analysis.blur_level = self._estimate_blur(gray)

        # Face detection
        if self.enable_face_detection:
            analysis.faces = self._detect_faces(gray, frame)

        # Text detection
        if self.enable_text_detection:
            analysis.text_regions = self._detect_text_regions(gray)

        # Classify primary content
        analysis.primary_content = self._classify_content(analysis, w, h)

        return analysis

    def analyze_video(
        self,
        video_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> List[SceneAnalysis]:
        """Analyze entire video.

        Args:
            video_path: Path to video file
            progress_callback: Progress callback

        Returns:
            List of SceneAnalysis for sampled frames
        """
        if not self._ensure_deps():
            return []

        cv2 = self._cv2
        np = self._np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, int(1.0 / self.sample_rate))

        analyses = []
        prev_frame = None
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_interval == 0:
                timestamp = frame_num / fps if fps > 0 else 0

                analysis = self.analyze_frame(frame, frame_num, timestamp)

                # Motion analysis (compare to previous)
                if self.enable_motion_analysis and prev_frame is not None:
                    analysis.motion_level = self._analyze_motion(prev_frame, frame)

                analyses.append(analysis)

                if progress_callback:
                    progress_callback({
                        "stage": "scene_analysis",
                        "frame": frame_num,
                        "total": total_frames,
                    })

            prev_frame = frame.copy()
            frame_num += 1

        cap.release()

        # Detect scene boundaries
        self._detect_scene_boundaries(analyses)

        logger.info(f"Analyzed {len(analyses)} frames, found {sum(1 for a in analyses if a.is_scene_start)} scenes")
        return analyses

    def _classify_lighting(self, gray) -> LightingCondition:
        """Classify lighting conditions."""
        np = self._np

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Calculate histogram
        hist = np.histogram(gray, bins=256, range=(0, 256))[0]
        hist = np.asarray(hist, dtype=float)
        hist = hist / hist.sum()

        # Check for high contrast (bimodal distribution)
        dark_ratio = np.sum(hist[:64])
        bright_ratio = np.sum(hist[192:])

        if mean_brightness > 180:
            return LightingCondition.BRIGHT
        elif mean_brightness < 60:
            return LightingCondition.LOW_LIGHT
        elif dark_ratio > 0.3 and bright_ratio > 0.2:
            return LightingCondition.HIGH_CONTRAST
        elif bright_ratio > 0.4 and dark_ratio > 0.2:
            return LightingCondition.BACKLIT
        else:
            return LightingCondition.NORMAL

    def _estimate_noise(self, gray) -> float:
        """Estimate noise level in frame."""
        cv2 = self._cv2
        np = self._np

        # Use Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian)) / 0.6745

        # Normalize to 0-1 range
        return min(1.0, sigma / 30.0)

    def _estimate_blur(self, gray) -> float:
        """Estimate blur level in frame."""
        cv2 = self._cv2
        np = self._np

        # Variance of Laplacian (lower = more blur)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize (sharper images have higher variance)
        # Invert so higher = more blur
        sharpness = min(1.0, variance / 500.0)
        return 1.0 - sharpness

    def _detect_faces(self, gray, color_frame) -> List[FaceRegion]:
        """Detect faces in frame."""
        if not self._load_face_cascade():
            return []

        cv2 = self._cv2

        faces = []

        # Detect faces at multiple scales
        detected = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in detected:
            # Calculate confidence based on size and position
            frame_h, frame_w = gray.shape
            relative_size = (w * h) / (frame_w * frame_h)
            confidence = min(1.0, relative_size * 10 + 0.5)

            # Estimate if profile (aspect ratio)
            is_profile = w / h > 1.3 or w / h < 0.7

            # Quality score based on region brightness and contrast
            face_region = gray[y:y+h, x:x+w]
            brightness = self._np.mean(face_region) / 255.0
            contrast = self._np.std(face_region) / 128.0
            quality_score = (brightness * 0.3 + contrast * 0.7)

            faces.append(FaceRegion(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                is_profile=is_profile,
                quality_score=quality_score,
            ))

        return faces

    def _detect_text_regions(self, gray) -> List[TextRegion]:
        """Detect text regions in frame."""
        cv2 = self._cv2
        np = self._np

        text_regions = []
        h, w = gray.shape

        # Use morphological operations to find text-like regions
        # Text has high local contrast and horizontal structure

        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Text tends to have more horizontal than vertical gradients
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold
        _, binary = cv2.threshold(
            (gradient_mag / gradient_mag.max() * 255).astype(np.uint8),
            30, 255, cv2.THRESH_BINARY
        )

        # Morphological closing to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Filter by aspect ratio (text tends to be wide)
            aspect = cw / ch if ch > 0 else 0
            if aspect < 2 or cw < 50:
                continue

            # Filter by position (titles often at top or bottom)
            relative_y = y / h
            if 0.1 < relative_y < 0.85:
                text_type = "sign"
            elif relative_y <= 0.1:
                text_type = "title"
            else:
                text_type = "subtitle"

            # Calculate confidence
            area_ratio = (cw * ch) / (w * h)
            confidence = min(1.0, area_ratio * 20 + 0.3)

            text_regions.append(TextRegion(
                x=x, y=y, width=cw, height=ch,
                confidence=confidence,
                text_type=text_type,
            ))

        return text_regions

    def _classify_content(self, analysis: SceneAnalysis, w: int, h: int) -> ContentType:
        """Classify primary content type."""
        face_area = sum(f.width * f.height for f in analysis.faces)
        frame_area = w * h

        face_ratio = face_area / frame_area if frame_area > 0 else 0

        # Face-centric content
        if len(analysis.faces) == 1 and face_ratio > 0.1:
            if face_ratio > 0.25:
                return ContentType.FACE_CLOSEUP
            return ContentType.FACE_MEDIUM
        elif len(analysis.faces) > 2:
            return ContentType.GROUP_SHOT

        # Text-heavy content
        if len(analysis.text_regions) > 0:
            text_area = sum(t.width * t.height for t in analysis.text_regions)
            if text_area / frame_area > 0.15:
                return ContentType.TEXT_TITLE

        # Motion-based
        if analysis.motion_level == MotionLevel.EXTREME:
            return ContentType.ACTION
        elif analysis.motion_level == MotionLevel.STATIC:
            return ContentType.STATIC

        # Default based on lighting and composition
        if analysis.lighting in [LightingCondition.BRIGHT, LightingCondition.NORMAL]:
            if analysis.avg_brightness > 0.6:
                return ContentType.LANDSCAPE

        return ContentType.DOCUMENTARY

    def _analyze_motion(self, prev_frame, curr_frame) -> MotionLevel:
        """Analyze motion between frames."""
        cv2 = self._cv2
        np = self._np

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)

        # Classify motion level
        if mean_diff < 2:
            return MotionLevel.STATIC
        elif mean_diff < 8:
            return MotionLevel.MINIMAL
        elif mean_diff < 20:
            return MotionLevel.MODERATE
        elif mean_diff < 40:
            return MotionLevel.HIGH
        else:
            return MotionLevel.EXTREME

    def _detect_scene_boundaries(self, analyses: List[SceneAnalysis]) -> None:
        """Detect scene boundaries in analysis sequence."""
        if len(analyses) < 2:
            return

        # First frame is always scene start
        analyses[0].is_scene_start = True

        for i in range(1, len(analyses)):
            prev = analyses[i - 1]
            curr = analyses[i]

            # Detect scene change based on multiple factors
            brightness_change = abs(curr.avg_brightness - prev.avg_brightness)
            content_change = curr.primary_content != prev.primary_content
            motion_spike = (
                prev.motion_level == MotionLevel.STATIC and
                curr.motion_level in [MotionLevel.HIGH, MotionLevel.EXTREME]
            )

            # Scene boundary if significant change
            if brightness_change > 0.3 or content_change or motion_spike:
                curr.is_scene_start = True
                prev.is_scene_end = True

    def get_adaptive_settings(
        self,
        analysis: SceneAnalysis,
        base_settings: Optional[Dict[str, float]] = None,
    ) -> AdaptiveSettings:
        """Get adaptive processing settings for analyzed content.

        Args:
            analysis: Scene analysis results
            base_settings: Base settings to adapt from

        Returns:
            AdaptiveSettings tuned for content
        """
        settings = AdaptiveSettings()

        # Base settings
        if base_settings:
            settings.sharpening = base_settings.get("sharpening", 0.3)
            settings.noise_reduction = base_settings.get("noise_reduction", 0.3)

        # Adapt based on content type
        if analysis.primary_content == ContentType.FACE_CLOSEUP:
            # Careful face processing - avoid plastic look
            settings.face_enhancement = 0.4
            settings.sharpening = min(settings.sharpening, 0.25)
            settings.noise_reduction = min(settings.noise_reduction, 0.4)
            settings.detail_enhancement = 0.3
            settings.face_regions = [(f.x, f.y, f.width, f.height) for f in analysis.faces]
            settings.apply_regional = True

        elif analysis.primary_content == ContentType.TEXT_TITLE:
            # Preserve text sharpness
            settings.text_sharpening = 0.4
            settings.sharpening = 0.4
            settings.noise_reduction = 0.2  # Less NR to preserve text
            settings.text_regions = [(t.x, t.y, t.width, t.height) for t in analysis.text_regions]
            settings.apply_regional = True

        elif analysis.primary_content == ContentType.LANDSCAPE:
            # More aggressive enhancement for landscapes
            settings.sharpening = min(settings.sharpening * 1.2, 0.5)
            settings.detail_enhancement = 0.4
            settings.color_correction = 0.4

        elif analysis.primary_content == ContentType.ACTION:
            # Motion-aware processing
            settings.temporal_smoothing = 0.3  # Less temporal for fast motion
            settings.interpolation_quality = "high"
            settings.sharpening = 0.35

        # Adapt for motion level
        if analysis.motion_level == MotionLevel.STATIC:
            settings.temporal_smoothing = 0.8  # Heavy temporal for static
            settings.noise_reduction *= 1.2
        elif analysis.motion_level == MotionLevel.EXTREME:
            settings.temporal_smoothing = 0.2
            settings.interpolation_quality = "fast"

        # Adapt for lighting
        if analysis.lighting == LightingCondition.LOW_LIGHT:
            settings.noise_reduction *= 1.3  # More NR for low light
            settings.sharpening *= 0.8  # Less sharpening (amplifies noise)

        # Adapt for existing quality
        if analysis.blur_level > 0.5:
            settings.sharpening *= 1.2  # More sharpening for blurry
        if analysis.noise_level > 0.5:
            settings.noise_reduction *= 1.2  # More NR for noisy

        return settings

    def generate_processing_map(
        self,
        analyses: List[SceneAnalysis],
    ) -> Dict[int, AdaptiveSettings]:
        """Generate a processing map for entire video.

        Args:
            analyses: List of scene analyses

        Returns:
            Dict mapping frame numbers to adaptive settings
        """
        processing_map = {}

        for analysis in analyses:
            settings = self.get_adaptive_settings(analysis)
            processing_map[analysis.frame_number] = settings

        return processing_map

    def get_summary(self, analyses: List[SceneAnalysis]) -> Dict[str, Any]:
        """Get summary statistics for analyzed video.

        Args:
            analyses: List of scene analyses

        Returns:
            Summary statistics
        """
        if not analyses:
            return {}

        content_counts = {}
        for a in analyses:
            ct = a.primary_content.name
            content_counts[ct] = content_counts.get(ct, 0) + 1

        motion_counts = {}
        for a in analyses:
            ml = a.motion_level.value
            motion_counts[ml] = motion_counts.get(ml, 0) + 1

        total_faces = sum(len(a.faces) for a in analyses)
        total_text = sum(len(a.text_regions) for a in analyses)
        scene_count = sum(1 for a in analyses if a.is_scene_start)

        avg_brightness = sum(a.avg_brightness for a in analyses) / len(analyses)
        avg_noise = sum(a.noise_level for a in analyses) / len(analyses)
        avg_blur = sum(a.blur_level for a in analyses) / len(analyses)

        return {
            "frames_analyzed": len(analyses),
            "scene_count": scene_count,
            "content_distribution": content_counts,
            "motion_distribution": motion_counts,
            "total_faces_detected": total_faces,
            "total_text_regions": total_text,
            "average_brightness": avg_brightness,
            "average_noise_level": avg_noise,
            "average_blur_level": avg_blur,
            "has_faces": total_faces > 0,
            "has_text": total_text > 0,
            "is_static_content": motion_counts.get("static", 0) > len(analyses) * 0.5,
            "is_action_content": motion_counts.get("high", 0) + motion_counts.get("extreme", 0) > len(analyses) * 0.3,
        }


@dataclass
class SceneAdaptiveConfig:
    """Configuration for scene-adaptive processing.

    Controls how processing intensity is adjusted based on scene content.
    """
    intensity_scale: float = 1.0  # Global scaling factor for processing intensity
    preserve_faces: bool = True   # Apply lighter processing to face regions
    preserve_text: bool = True    # Preserve sharpness in text regions
    motion_sensitivity: float = 0.5  # How much motion affects processing (0.0-1.0)


class SceneAdaptiveProcessor:
    """Scene-aware video processor that adapts processing intensity per scene.

    Uses SceneAnalysis to determine optimal processing parameters based on
    content type, motion level, and detected regions (faces, text).

    Processing intensity guidelines:
    - Dialog scenes (FACE_CLOSEUP/FACE_MEDIUM): lighter processing, preserve detail
    - Action scenes (ACTION/HIGH motion): full processing
    - Static frames (STATIC): strongest denoise
    - Text scenes (TEXT_TITLE): preserve sharpness
    """

    # Intensity mappings for content types
    CONTENT_INTENSITY = {
        ContentType.FACE_CLOSEUP: 0.4,   # Light processing for close faces
        ContentType.FACE_MEDIUM: 0.5,    # Moderate for medium shots
        ContentType.GROUP_SHOT: 0.6,     # Slightly more for groups
        ContentType.LANDSCAPE: 0.8,      # More aggressive for landscapes
        ContentType.ARCHITECTURE: 0.7,   # Good detail for architecture
        ContentType.TEXT_TITLE: 0.3,     # Preserve text clarity
        ContentType.ACTION: 1.0,         # Full processing for action
        ContentType.STATIC: 0.9,         # Strong denoise for static
        ContentType.DOCUMENTARY: 0.6,    # Balanced for documentary
        ContentType.ANIMATION: 0.5,      # Careful with animation
        ContentType.UNKNOWN: 0.6,        # Default moderate
    }

    # Motion level intensity modifiers
    MOTION_INTENSITY = {
        MotionLevel.STATIC: 1.2,     # Increase for static (more denoise)
        MotionLevel.MINIMAL: 1.0,    # No change
        MotionLevel.MODERATE: 0.9,   # Slight reduction
        MotionLevel.HIGH: 1.0,       # Full for high motion
        MotionLevel.EXTREME: 1.0,    # Full for extreme motion
    }

    def __init__(self, config: Optional[SceneAdaptiveConfig] = None):
        """Initialize scene-adaptive processor.

        Args:
            config: Configuration for adaptive processing
        """
        self.config = config or SceneAdaptiveConfig()
        self._cv2 = None
        self._np = None

    def _ensure_deps(self) -> bool:
        """Ensure OpenCV and numpy are available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            return False

    def get_processing_intensity(self, analysis: SceneAnalysis) -> float:
        """Determine processing intensity for a scene analysis.

        Args:
            analysis: SceneAnalysis containing content classification and metrics

        Returns:
            Float between 0.0 and 1.0 representing processing intensity
            - 0.0: minimal processing (preserve original)
            - 1.0: maximum processing
        """
        # Base intensity from content type
        base_intensity = self.CONTENT_INTENSITY.get(
            analysis.primary_content,
            0.6
        )

        # Apply motion modifier
        motion_modifier = self.MOTION_INTENSITY.get(
            analysis.motion_level,
            1.0
        )

        # Blend motion influence based on sensitivity
        intensity = base_intensity * (
            1.0 + (motion_modifier - 1.0) * self.config.motion_sensitivity
        )

        # Reduce intensity for face-heavy content if preservation enabled
        if self.config.preserve_faces and analysis.faces:
            face_count = len(analysis.faces)
            # More faces = lighter processing
            face_reduction = min(0.3, face_count * 0.1)
            intensity -= face_reduction

        # Reduce intensity for text content if preservation enabled
        if self.config.preserve_text and analysis.text_regions:
            text_count = len(analysis.text_regions)
            text_reduction = min(0.3, text_count * 0.15)
            intensity -= text_reduction

        # Apply global scale
        intensity *= self.config.intensity_scale

        # Clamp to valid range
        return max(0.0, min(1.0, intensity))

    def adjust_settings_for_scene(
        self,
        base_settings: AdaptiveSettings,
        analysis: SceneAnalysis,
    ) -> AdaptiveSettings:
        """Adjust processing settings based on scene analysis.

        Args:
            base_settings: Base AdaptiveSettings to modify
            analysis: SceneAnalysis for the current scene

        Returns:
            Modified AdaptiveSettings tuned for the scene content
        """
        # Get processing intensity
        intensity = self.get_processing_intensity(analysis)

        # Create new settings based on base
        adjusted = AdaptiveSettings(
            sharpening=base_settings.sharpening,
            noise_reduction=base_settings.noise_reduction,
            detail_enhancement=base_settings.detail_enhancement,
            face_enhancement=base_settings.face_enhancement,
            face_regions=list(base_settings.face_regions),
            text_sharpening=base_settings.text_sharpening,
            text_regions=list(base_settings.text_regions),
            temporal_smoothing=base_settings.temporal_smoothing,
            interpolation_quality=base_settings.interpolation_quality,
            color_correction=base_settings.color_correction,
            saturation_adjustment=base_settings.saturation_adjustment,
            apply_regional=base_settings.apply_regional,
            region_weights=dict(base_settings.region_weights),
        )

        # Apply intensity-based adjustments
        adjusted.noise_reduction *= intensity
        adjusted.sharpening *= intensity
        adjusted.detail_enhancement *= intensity
        adjusted.color_correction *= intensity

        # Content-specific adjustments
        if analysis.primary_content in (ContentType.FACE_CLOSEUP, ContentType.FACE_MEDIUM):
            # Dialog/face scenes: preserve detail, lighter processing
            adjusted.face_enhancement = min(0.4, base_settings.face_enhancement)
            adjusted.sharpening = min(0.25, adjusted.sharpening)
            adjusted.noise_reduction = min(0.4, adjusted.noise_reduction)
            adjusted.temporal_smoothing = 0.6  # Moderate temporal for faces

            # Add face regions
            adjusted.face_regions = [
                (f.x, f.y, f.width, f.height) for f in analysis.faces
            ]
            adjusted.apply_regional = True
            adjusted.region_weights["faces"] = 0.5  # Lower weight for faces

        elif analysis.primary_content == ContentType.ACTION or analysis.motion_level == MotionLevel.HIGH:
            # Action scenes: full processing
            adjusted.temporal_smoothing = 0.3  # Less temporal for motion
            adjusted.interpolation_quality = "high"
            # Maintain higher processing levels
            adjusted.sharpening = max(adjusted.sharpening, 0.35)
            adjusted.noise_reduction = max(adjusted.noise_reduction, 0.4)

        elif analysis.primary_content == ContentType.STATIC or analysis.motion_level == MotionLevel.STATIC:
            # Static frames: strongest denoise
            adjusted.noise_reduction = min(1.0, base_settings.noise_reduction * 1.4)
            adjusted.temporal_smoothing = 0.9  # Heavy temporal averaging
            adjusted.detail_enhancement *= 0.8  # Slightly less detail enhancement

        elif analysis.primary_content == ContentType.TEXT_TITLE:
            # Text scenes: preserve sharpness
            adjusted.text_sharpening = max(0.4, base_settings.text_sharpening)
            adjusted.sharpening = max(0.4, adjusted.sharpening)
            adjusted.noise_reduction = min(0.2, adjusted.noise_reduction)  # Light NR

            # Add text regions
            adjusted.text_regions = [
                (t.x, t.y, t.width, t.height) for t in analysis.text_regions
            ]
            adjusted.apply_regional = True
            adjusted.region_weights["text"] = 0.3  # Very light processing on text

        # Adjust for noise level
        if analysis.noise_level > 0.5:
            adjusted.noise_reduction = min(1.0, adjusted.noise_reduction * 1.2)

        # Adjust for blur level
        if analysis.blur_level > 0.5:
            adjusted.sharpening = min(0.6, adjusted.sharpening * 1.2)

        # Adjust for lighting conditions
        if analysis.lighting == LightingCondition.LOW_LIGHT:
            adjusted.noise_reduction = min(1.0, adjusted.noise_reduction * 1.3)
            adjusted.sharpening *= 0.8  # Less sharpening (amplifies noise)

        return adjusted

    def process_video_scene_aware(
        self,
        frames_dir: Path,
        analyses: List[SceneAnalysis],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Process video frames with scene-aware adaptive settings.

        Args:
            frames_dir: Directory containing input frames
            analyses: List of SceneAnalysis for the frames
            output_dir: Directory to write processed frames

        Returns:
            Dict with processing results:
                - frames_processed: number of frames processed
                - frames_skipped: number of frames skipped
                - intensity_stats: min/max/avg intensity
                - content_breakdown: frames per content type
                - errors: list of any errors encountered
        """
        if not self._ensure_deps():
            return {
                "frames_processed": 0,
                "frames_skipped": 0,
                "errors": ["OpenCV/numpy not available"],
            }

        cv2 = self._cv2
        np = self._np

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build analysis lookup by frame number
        analysis_map = {a.frame_number: a for a in analyses}

        # Collect frame files
        frame_files = sorted(
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg"))
        )

        if not frame_files:
            return {
                "frames_processed": 0,
                "frames_skipped": 0,
                "errors": [f"No frames found in {frames_dir}"],
            }

        # Processing statistics
        frames_processed = 0
        frames_skipped = 0
        intensities = []
        content_breakdown = {}
        errors = []

        # Base settings to adapt from
        base_settings = AdaptiveSettings()

        for idx, frame_path in enumerate(frame_files):
            try:
                # Get analysis for this frame (or nearest)
                analysis = analysis_map.get(idx)

                if analysis is None:
                    # Find nearest analysis
                    nearest_idx = min(
                        analysis_map.keys(),
                        key=lambda x: abs(x - idx),
                        default=None
                    )
                    analysis = analysis_map.get(nearest_idx) if nearest_idx is not None else None

                if analysis is None:
                    # No analysis available, skip or use defaults
                    analysis = SceneAnalysis(frame_number=idx, timestamp=0.0)

                # Get intensity and adjusted settings
                intensity = self.get_processing_intensity(analysis)
                settings = self.adjust_settings_for_scene(base_settings, analysis)

                intensities.append(intensity)

                # Track content breakdown
                content_name = analysis.primary_content.name
                content_breakdown[content_name] = content_breakdown.get(content_name, 0) + 1

                # Read frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    errors.append(f"Could not read frame: {frame_path}")
                    frames_skipped += 1
                    continue

                # Apply processing based on settings
                processed = self._apply_adaptive_processing(frame, settings, analysis)

                # Write output frame
                output_path = output_dir / frame_path.name
                cv2.imwrite(str(output_path), processed)

                frames_processed += 1

            except Exception as e:
                errors.append(f"Error processing frame {idx}: {str(e)}")
                frames_skipped += 1

        # Calculate intensity statistics
        intensity_stats = {
            "min": min(intensities) if intensities else 0.0,
            "max": max(intensities) if intensities else 0.0,
            "avg": sum(intensities) / len(intensities) if intensities else 0.0,
        }

        logger.info(
            f"Scene-aware processing complete: {frames_processed} frames processed, "
            f"{frames_skipped} skipped, avg intensity: {intensity_stats['avg']:.2f}"
        )

        return {
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "intensity_stats": intensity_stats,
            "content_breakdown": content_breakdown,
            "errors": errors if errors else None,
        }

    def _apply_adaptive_processing(
        self,
        frame,
        settings: AdaptiveSettings,
        analysis: SceneAnalysis,
    ):
        """Apply adaptive processing to a frame.

        Args:
            frame: Input frame (numpy array)
            settings: AdaptiveSettings for this frame
            analysis: SceneAnalysis for context

        Returns:
            Processed frame
        """
        cv2 = self._cv2
        np = self._np

        processed = frame.copy()

        # Apply noise reduction
        if settings.noise_reduction > 0.01:
            # Use bilateral filter for edge-preserving denoising
            d = int(5 + settings.noise_reduction * 10)
            sigma_color = 50 + settings.noise_reduction * 100
            sigma_space = 50 + settings.noise_reduction * 100
            processed = cv2.bilateralFilter(
                processed, d, sigma_color, sigma_space
            )

        # Apply sharpening
        if settings.sharpening > 0.01:
            # Unsharp mask
            gaussian = cv2.GaussianBlur(processed, (0, 0), 3.0)
            sharpened = cv2.addWeighted(
                processed, 1.0 + settings.sharpening,
                gaussian, -settings.sharpening,
                0
            )
            processed = sharpened

        # Apply regional processing if enabled
        if settings.apply_regional:
            # Process face regions with care
            if settings.face_regions and self.config.preserve_faces:
                for (x, y, w, h) in settings.face_regions:
                    # Ensure bounds are valid
                    y_end = min(y + h, processed.shape[0])
                    x_end = min(x + w, processed.shape[1])
                    y = max(0, y)
                    x = max(0, x)

                    if y < y_end and x < x_end:
                        # Apply lighter processing to face region
                        face_region = frame[y:y_end, x:x_end].copy()
                        # Light bilateral filter only
                        face_processed = cv2.bilateralFilter(
                            face_region, 5, 40, 40
                        )
                        # Blend back
                        weight = settings.region_weights.get("faces", 0.5)
                        processed[y:y_end, x:x_end] = cv2.addWeighted(
                            processed[y:y_end, x:x_end], 1 - weight,
                            face_processed, weight,
                            0
                        )

            # Process text regions with sharpness preservation
            if settings.text_regions and self.config.preserve_text:
                for (x, y, w, h) in settings.text_regions:
                    # Ensure bounds are valid
                    y_end = min(y + h, processed.shape[0])
                    x_end = min(x + w, processed.shape[1])
                    y = max(0, y)
                    x = max(0, x)

                    if y < y_end and x < x_end:
                        # Extra sharpening for text
                        text_region = frame[y:y_end, x:x_end].copy()
                        gaussian = cv2.GaussianBlur(text_region, (0, 0), 2.0)
                        sharp_strength = settings.text_sharpening + 0.3
                        text_sharpened = cv2.addWeighted(
                            text_region, 1.0 + sharp_strength,
                            gaussian, -sharp_strength,
                            0
                        )
                        # Blend back
                        weight = settings.region_weights.get("text", 0.3)
                        processed[y:y_end, x:x_end] = cv2.addWeighted(
                            processed[y:y_end, x:x_end], weight,
                            text_sharpened, 1 - weight,
                            0
                        )

        return processed


def analyze_video_intelligence(
    video_path: Path,
    sample_rate: float = 0.5,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[SceneAnalysis], Dict[str, Any]]:
    """Analyze video with scene intelligence.

    Args:
        video_path: Path to video
        sample_rate: Fraction of frames to analyze
        progress_callback: Progress callback

    Returns:
        Tuple of (analyses, summary)
    """
    intelligence = SceneIntelligence(sample_rate=sample_rate)
    analyses = intelligence.analyze_video(video_path, progress_callback)
    summary = intelligence.get_summary(analyses)

    return analyses, summary


def create_scene_intelligence(
    sample_rate: float = 0.5,
    enable_face_detection: bool = True,
    enable_text_detection: bool = True,
    enable_motion_analysis: bool = True,
) -> SceneIntelligence:
    """Factory function to create a SceneIntelligence instance.

    Args:
        sample_rate: Fraction of frames to sample for analysis (0.0-1.0)
        enable_face_detection: Whether to detect faces
        enable_text_detection: Whether to detect text regions
        enable_motion_analysis: Whether to analyze motion

    Returns:
        Configured SceneIntelligence instance
    """
    return SceneIntelligence(
        sample_rate=sample_rate,
        enable_face_detection=enable_face_detection,
        enable_text_detection=enable_text_detection,
        enable_motion_analysis=enable_motion_analysis,
    )
