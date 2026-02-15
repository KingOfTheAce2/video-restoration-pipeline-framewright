"""Tests for the scene intelligence processor."""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from framewright.processors.scene_intelligence import (
    SceneIntelligence,
    SceneAnalysis,
    AdaptiveSettings,
    ContentType,
    MotionLevel,
    LightingCondition,
    FaceRegion,
    TextRegion,
    SceneAdaptiveConfig,
    SceneAdaptiveProcessor,
    analyze_video_intelligence,
)


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_type_members(self):
        """Test all expected content types exist."""
        assert ContentType.FACE_CLOSEUP is not None
        assert ContentType.FACE_MEDIUM is not None
        assert ContentType.GROUP_SHOT is not None
        assert ContentType.LANDSCAPE is not None
        assert ContentType.ARCHITECTURE is not None
        assert ContentType.TEXT_TITLE is not None
        assert ContentType.ACTION is not None
        assert ContentType.STATIC is not None
        assert ContentType.DOCUMENTARY is not None
        assert ContentType.ANIMATION is not None
        assert ContentType.UNKNOWN is not None


class TestMotionLevel:
    """Tests for MotionLevel enum."""

    def test_motion_level_values(self):
        """Test motion level enum values."""
        assert MotionLevel.STATIC.value == "static"
        assert MotionLevel.MINIMAL.value == "minimal"
        assert MotionLevel.MODERATE.value == "moderate"
        assert MotionLevel.HIGH.value == "high"
        assert MotionLevel.EXTREME.value == "extreme"


class TestLightingCondition:
    """Tests for LightingCondition enum."""

    def test_lighting_condition_values(self):
        """Test lighting condition enum values."""
        assert LightingCondition.BRIGHT.value == "bright"
        assert LightingCondition.NORMAL.value == "normal"
        assert LightingCondition.LOW_LIGHT.value == "low_light"
        assert LightingCondition.HIGH_CONTRAST.value == "high_contrast"
        assert LightingCondition.BACKLIT.value == "backlit"
        assert LightingCondition.MIXED.value == "mixed"


class TestFaceRegion:
    """Tests for FaceRegion dataclass."""

    def test_face_region_creation(self):
        """Test FaceRegion with required and optional fields."""
        region = FaceRegion(
            x=100, y=50, width=200, height=250,
            confidence=0.95,
        )

        assert region.x == 100
        assert region.y == 50
        assert region.width == 200
        assert region.height == 250
        assert region.confidence == 0.95
        assert region.is_profile is False  # Default
        assert region.estimated_age is None  # Default
        assert region.quality_score == 0.5  # Default

    def test_face_region_with_all_fields(self):
        """Test FaceRegion with all fields specified."""
        region = FaceRegion(
            x=100, y=50, width=200, height=250,
            confidence=0.95,
            is_profile=True,
            estimated_age="adult",
            quality_score=0.8,
        )

        assert region.is_profile is True
        assert region.estimated_age == "adult"
        assert region.quality_score == 0.8


class TestTextRegion:
    """Tests for TextRegion dataclass."""

    def test_text_region_creation(self):
        """Test TextRegion with required fields."""
        region = TextRegion(
            x=10, y=500, width=620, height=30,
            confidence=0.85,
        )

        assert region.x == 10
        assert region.y == 500
        assert region.width == 620
        assert region.height == 30
        assert region.confidence == 0.85
        assert region.text_type == "unknown"  # Default
        assert region.is_period_font is True  # Default

    def test_text_region_with_type(self):
        """Test TextRegion with text type specified."""
        region = TextRegion(
            x=10, y=10, width=300, height=40,
            confidence=0.9,
            text_type="title",
            is_period_font=False,
        )

        assert region.text_type == "title"
        assert region.is_period_font is False


class TestSceneAnalysis:
    """Tests for SceneAnalysis dataclass."""

    def test_scene_analysis_defaults(self):
        """Test SceneAnalysis with default values."""
        analysis = SceneAnalysis(frame_number=0, timestamp=0.0)

        assert analysis.frame_number == 0
        assert analysis.timestamp == 0.0
        assert analysis.primary_content == ContentType.UNKNOWN
        assert analysis.secondary_content == []
        assert analysis.motion_level == MotionLevel.MODERATE
        assert analysis.lighting == LightingCondition.NORMAL
        assert analysis.faces == []
        assert analysis.text_regions == []
        assert analysis.blur_level == 0.0
        assert analysis.noise_level == 0.0
        assert analysis.is_scene_start is False
        assert analysis.is_scene_end is False

    def test_scene_analysis_with_faces(self):
        """Test SceneAnalysis with face regions."""
        faces = [
            FaceRegion(x=100, y=100, width=150, height=200, confidence=0.9),
            FaceRegion(x=400, y=120, width=140, height=190, confidence=0.85),
        ]

        analysis = SceneAnalysis(
            frame_number=42,
            timestamp=1.75,
            primary_content=ContentType.GROUP_SHOT,
            faces=faces,
        )

        assert len(analysis.faces) == 2
        assert analysis.faces[0].confidence == 0.9

    def test_scene_analysis_to_dict(self):
        """Test SceneAnalysis to_dict conversion."""
        analysis = SceneAnalysis(
            frame_number=100,
            timestamp=4.16,
            primary_content=ContentType.FACE_CLOSEUP,
            motion_level=MotionLevel.MINIMAL,
            lighting=LightingCondition.BRIGHT,
            blur_level=0.2,
            noise_level=0.1,
        )

        data = analysis.to_dict()

        assert data["frame_number"] == 100
        assert data["timestamp"] == 4.16
        assert data["primary_content"] == "FACE_CLOSEUP"
        assert data["motion_level"] == "minimal"
        assert data["lighting"] == "bright"
        assert data["blur_level"] == 0.2
        assert data["noise_level"] == 0.1


class TestAdaptiveSettings:
    """Tests for AdaptiveSettings dataclass."""

    def test_adaptive_settings_defaults(self):
        """Test AdaptiveSettings with default values."""
        settings = AdaptiveSettings()

        assert settings.sharpening == 0.3
        assert settings.noise_reduction == 0.3
        assert settings.detail_enhancement == 0.3
        assert settings.face_enhancement == 0.5
        assert settings.face_regions == []
        assert settings.text_sharpening == 0.0
        assert settings.text_regions == []
        assert settings.temporal_smoothing == 0.5
        assert settings.interpolation_quality == "medium"
        assert settings.color_correction == 0.3
        assert settings.apply_regional is False

    def test_adaptive_settings_to_dict(self):
        """Test AdaptiveSettings to_dict conversion."""
        settings = AdaptiveSettings(
            sharpening=0.4,
            noise_reduction=0.5,
            face_enhancement=0.6,
        )

        data = settings.to_dict()

        assert data["sharpening"] == 0.4
        assert data["noise_reduction"] == 0.5
        assert data["face_enhancement"] == 0.6


class TestSceneIntelligenceInit:
    """Tests for SceneIntelligence initialization."""

    def test_default_initialization(self):
        """Test SceneIntelligence with defaults."""
        intel = SceneIntelligence()

        assert intel.enable_face_detection is True
        assert intel.enable_text_detection is True
        assert intel.enable_motion_analysis is True
        assert intel.sample_rate == 1.0

    def test_custom_initialization(self):
        """Test SceneIntelligence with custom parameters."""
        intel = SceneIntelligence(
            enable_face_detection=False,
            enable_text_detection=False,
            enable_motion_analysis=False,
            sample_rate=0.5,
        )

        assert intel.enable_face_detection is False
        assert intel.enable_text_detection is False
        assert intel.enable_motion_analysis is False
        assert intel.sample_rate == 0.5


class TestSceneIntelligenceAnalyzeFrame:
    """Tests for SceneIntelligence.analyze_frame method."""

    @pytest.fixture
    def mock_cv2(self):
        """Create mock cv2 module."""
        mock = MagicMock()
        mock.COLOR_BGR2GRAY = 6
        mock.CV_64F = 6
        mock.MORPH_RECT = 0
        mock.MORPH_CLOSE = 3
        mock.RETR_EXTERNAL = 0
        mock.CHAIN_APPROX_SIMPLE = 2
        mock.THRESH_BINARY = 0

        # Mock data
        mock.data.haarcascades = "/fake/path/"
        return mock

    @pytest.fixture
    def mock_numpy(self):
        """Create mock numpy module."""
        return MagicMock()

    def test_analyze_frame_returns_analysis(self):
        """Test that analyze_frame returns a SceneAnalysis."""
        intel = SceneIntelligence()

        # Test with no deps available
        with patch.object(intel, '_ensure_deps', return_value=False):
            result = intel.analyze_frame(None, frame_number=5, timestamp=0.21)

        assert isinstance(result, SceneAnalysis)
        assert result.frame_number == 5
        assert result.timestamp == 0.21

    def test_analyze_frame_with_path(self, tmp_path):
        """Test analyze_frame with a file path."""
        # Create a dummy image file
        frame_path = tmp_path / "frame.png"
        frame_path.write_bytes(b'fake png data')

        intel = SceneIntelligence()

        with patch.object(intel, '_ensure_deps', return_value=False):
            result = intel.analyze_frame(frame_path, frame_number=0, timestamp=0.0)

        assert isinstance(result, SceneAnalysis)


class TestSceneIntelligenceLighting:
    """Tests for lighting classification."""

    def test_classify_lighting_bright(self):
        """Test classification of bright lighting."""
        intel = SceneIntelligence()

        mock_np = MagicMock()
        mock_np.mean.return_value = 200  # High brightness
        mock_np.std.return_value = 30
        mock_np.histogram.return_value = ([1] * 256, None)
        mock_np.sum.return_value = 0.1

        intel._np = mock_np

        # Create a bright gray image mock
        gray = MagicMock()
        result = intel._classify_lighting(gray)

        assert result == LightingCondition.BRIGHT

    def test_classify_lighting_low_light(self):
        """Test classification of low light."""
        intel = SceneIntelligence()

        mock_np = MagicMock()
        mock_np.mean.return_value = 40  # Low brightness
        mock_np.std.return_value = 20
        mock_np.histogram.return_value = ([1] * 256, None)
        mock_np.sum.return_value = 0.1

        intel._np = mock_np

        gray = MagicMock()
        result = intel._classify_lighting(gray)

        assert result == LightingCondition.LOW_LIGHT


class TestSceneIntelligenceGetAdaptiveSettings:
    """Tests for get_adaptive_settings method."""

    def test_adaptive_settings_for_face_closeup(self):
        """Test adaptive settings for face closeup content."""
        intel = SceneIntelligence()

        analysis = SceneAnalysis(
            frame_number=0,
            timestamp=0.0,
            primary_content=ContentType.FACE_CLOSEUP,
            faces=[FaceRegion(x=100, y=100, width=200, height=250, confidence=0.95)],
        )

        settings = intel.get_adaptive_settings(analysis)

        # Face closeup should have lower sharpening
        assert settings.sharpening <= 0.25
        assert settings.face_enhancement == 0.4
        assert settings.apply_regional is True
        assert len(settings.face_regions) == 1

    def test_adaptive_settings_for_text_title(self):
        """Test adaptive settings for text/title content."""
        intel = SceneIntelligence()

        analysis = SceneAnalysis(
            frame_number=0,
            timestamp=0.0,
            primary_content=ContentType.TEXT_TITLE,
            text_regions=[TextRegion(x=10, y=10, width=300, height=50, confidence=0.9)],
        )

        settings = intel.get_adaptive_settings(analysis)

        # Text should have higher sharpening
        assert settings.text_sharpening >= 0.4
        assert settings.apply_regional is True
        assert len(settings.text_regions) == 1

    def test_adaptive_settings_for_landscape(self):
        """Test adaptive settings for landscape content."""
        intel = SceneIntelligence()

        analysis = SceneAnalysis(
            frame_number=0,
            timestamp=0.0,
            primary_content=ContentType.LANDSCAPE,
        )

        settings = intel.get_adaptive_settings(analysis)

        # Landscape can have more aggressive processing
        assert settings.detail_enhancement >= 0.4

    def test_adaptive_settings_for_action(self):
        """Test adaptive settings for action content."""
        intel = SceneIntelligence()

        analysis = SceneAnalysis(
            frame_number=0,
            timestamp=0.0,
            primary_content=ContentType.ACTION,
            motion_level=MotionLevel.HIGH,
        )

        settings = intel.get_adaptive_settings(analysis)

        # Action needs less temporal smoothing
        assert settings.temporal_smoothing <= 0.3
        assert settings.interpolation_quality == "high"

    def test_adaptive_settings_for_static_motion(self):
        """Test adaptive settings for static motion level."""
        intel = SceneIntelligence()

        analysis = SceneAnalysis(
            frame_number=0,
            timestamp=0.0,
            motion_level=MotionLevel.STATIC,
        )

        settings = intel.get_adaptive_settings(analysis)

        # Static should have heavy temporal smoothing
        assert settings.temporal_smoothing >= 0.8

    def test_adaptive_settings_for_low_light(self):
        """Test adaptive settings for low light conditions."""
        intel = SceneIntelligence()

        analysis = SceneAnalysis(
            frame_number=0,
            timestamp=0.0,
            lighting=LightingCondition.LOW_LIGHT,
        )

        settings = intel.get_adaptive_settings(analysis)

        # Low light needs more noise reduction, less sharpening
        # (Exact values depend on base settings)
        assert settings.noise_reduction > 0


class TestSceneIntelligenceSummary:
    """Tests for get_summary method."""

    def test_get_summary_empty_analyses(self):
        """Test get_summary with empty list."""
        intel = SceneIntelligence()
        summary = intel.get_summary([])

        assert summary == {}

    def test_get_summary_with_analyses(self):
        """Test get_summary with multiple analyses."""
        intel = SceneIntelligence()

        analyses = [
            SceneAnalysis(
                frame_number=0, timestamp=0.0,
                primary_content=ContentType.FACE_CLOSEUP,
                motion_level=MotionLevel.MINIMAL,
                avg_brightness=0.6, noise_level=0.1, blur_level=0.2,
                faces=[FaceRegion(x=0, y=0, width=100, height=100, confidence=0.9)],
                is_scene_start=True,
            ),
            SceneAnalysis(
                frame_number=1, timestamp=0.04,
                primary_content=ContentType.FACE_CLOSEUP,
                motion_level=MotionLevel.MINIMAL,
                avg_brightness=0.5, noise_level=0.15, blur_level=0.25,
                faces=[FaceRegion(x=0, y=0, width=100, height=100, confidence=0.9)],
            ),
            SceneAnalysis(
                frame_number=2, timestamp=0.08,
                primary_content=ContentType.ACTION,
                motion_level=MotionLevel.HIGH,
                avg_brightness=0.7, noise_level=0.2, blur_level=0.3,
                is_scene_start=True,
            ),
        ]

        summary = intel.get_summary(analyses)

        assert summary["frames_analyzed"] == 3
        assert summary["scene_count"] == 2
        assert "FACE_CLOSEUP" in summary["content_distribution"]
        assert summary["content_distribution"]["FACE_CLOSEUP"] == 2
        assert "ACTION" in summary["content_distribution"]
        assert summary["total_faces_detected"] == 2
        assert summary["has_faces"] is True
        assert summary["has_text"] is False


class TestSceneAdaptiveConfig:
    """Tests for SceneAdaptiveConfig dataclass."""

    def test_default_values(self):
        """Test SceneAdaptiveConfig default values."""
        config = SceneAdaptiveConfig()

        assert config.intensity_scale == 1.0
        assert config.preserve_faces is True
        assert config.preserve_text is True
        assert config.motion_sensitivity == 0.5

    def test_custom_values(self):
        """Test SceneAdaptiveConfig with custom values."""
        config = SceneAdaptiveConfig(
            intensity_scale=0.8,
            preserve_faces=False,
            preserve_text=False,
            motion_sensitivity=0.3,
        )

        assert config.intensity_scale == 0.8
        assert config.preserve_faces is False
        assert config.preserve_text is False
        assert config.motion_sensitivity == 0.3


class TestSceneAdaptiveProcessor:
    """Tests for SceneAdaptiveProcessor class."""

    def test_default_initialization(self):
        """Test SceneAdaptiveProcessor with defaults."""
        processor = SceneAdaptiveProcessor()

        assert processor.config is not None
        assert processor.config.intensity_scale == 1.0

    def test_custom_config_initialization(self):
        """Test SceneAdaptiveProcessor with custom config."""
        config = SceneAdaptiveConfig(intensity_scale=0.5)
        processor = SceneAdaptiveProcessor(config=config)

        assert processor.config.intensity_scale == 0.5

    def test_content_intensity_mapping(self):
        """Test that content intensity mapping is defined."""
        assert ContentType.FACE_CLOSEUP in SceneAdaptiveProcessor.CONTENT_INTENSITY
        assert ContentType.ACTION in SceneAdaptiveProcessor.CONTENT_INTENSITY
        assert ContentType.STATIC in SceneAdaptiveProcessor.CONTENT_INTENSITY

        # Face closeup should have lower intensity than action
        assert (
            SceneAdaptiveProcessor.CONTENT_INTENSITY[ContentType.FACE_CLOSEUP] <
            SceneAdaptiveProcessor.CONTENT_INTENSITY[ContentType.ACTION]
        )

    def test_motion_intensity_mapping(self):
        """Test that motion intensity mapping is defined."""
        assert MotionLevel.STATIC in SceneAdaptiveProcessor.MOTION_INTENSITY
        assert MotionLevel.HIGH in SceneAdaptiveProcessor.MOTION_INTENSITY


class TestSceneAdaptiveProcessorIntensity:
    """Tests for get_processing_intensity method."""

    def test_intensity_face_closeup(self):
        """Test processing intensity for face closeup."""
        processor = SceneAdaptiveProcessor()

        analysis = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.FACE_CLOSEUP,
        )

        intensity = processor.get_processing_intensity(analysis)

        assert 0.0 <= intensity <= 1.0
        assert intensity <= 0.5  # Face closeup should be light

    def test_intensity_action(self):
        """Test processing intensity for action content."""
        processor = SceneAdaptiveProcessor()

        analysis = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.ACTION,
            motion_level=MotionLevel.HIGH,
        )

        intensity = processor.get_processing_intensity(analysis)

        assert 0.0 <= intensity <= 1.0
        assert intensity >= 0.8  # Action should be high

    def test_intensity_reduced_for_faces(self):
        """Test that intensity is reduced when faces are present."""
        config = SceneAdaptiveConfig(preserve_faces=True)
        processor = SceneAdaptiveProcessor(config=config)

        analysis_no_faces = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.DOCUMENTARY,
        )

        analysis_with_faces = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.DOCUMENTARY,
            faces=[
                FaceRegion(x=0, y=0, width=100, height=100, confidence=0.9),
                FaceRegion(x=200, y=0, width=100, height=100, confidence=0.9),
            ],
        )

        intensity_no_faces = processor.get_processing_intensity(analysis_no_faces)
        intensity_with_faces = processor.get_processing_intensity(analysis_with_faces)

        assert intensity_with_faces < intensity_no_faces

    def test_intensity_reduced_for_text(self):
        """Test that intensity is reduced when text is present."""
        config = SceneAdaptiveConfig(preserve_text=True)
        processor = SceneAdaptiveProcessor(config=config)

        analysis_no_text = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.DOCUMENTARY,
        )

        analysis_with_text = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.DOCUMENTARY,
            text_regions=[
                TextRegion(x=0, y=500, width=640, height=30, confidence=0.9),
            ],
        )

        intensity_no_text = processor.get_processing_intensity(analysis_no_text)
        intensity_with_text = processor.get_processing_intensity(analysis_with_text)

        assert intensity_with_text < intensity_no_text

    def test_intensity_clamped_to_valid_range(self):
        """Test that intensity is always clamped to 0-1 range."""
        config = SceneAdaptiveConfig(intensity_scale=3.0)  # Extreme scale
        processor = SceneAdaptiveProcessor(config=config)

        analysis = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.ACTION,
            motion_level=MotionLevel.EXTREME,
        )

        intensity = processor.get_processing_intensity(analysis)

        assert intensity <= 1.0


class TestSceneAdaptiveProcessorAdjustSettings:
    """Tests for adjust_settings_for_scene method."""

    def test_adjust_settings_face_content(self):
        """Test settings adjustment for face content."""
        processor = SceneAdaptiveProcessor()

        base_settings = AdaptiveSettings(sharpening=0.5, noise_reduction=0.5)

        analysis = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.FACE_CLOSEUP,
            faces=[FaceRegion(x=100, y=100, width=200, height=250, confidence=0.9)],
        )

        adjusted = processor.adjust_settings_for_scene(base_settings, analysis)

        # Face content should have reduced processing
        assert adjusted.face_enhancement <= 0.4
        assert adjusted.sharpening <= 0.25
        assert len(adjusted.face_regions) == 1
        assert adjusted.apply_regional is True

    def test_adjust_settings_action_content(self):
        """Test settings adjustment for action content."""
        processor = SceneAdaptiveProcessor()

        base_settings = AdaptiveSettings()

        analysis = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.ACTION,
            motion_level=MotionLevel.HIGH,
        )

        adjusted = processor.adjust_settings_for_scene(base_settings, analysis)

        # Action should have less temporal smoothing
        assert adjusted.temporal_smoothing <= 0.3
        assert adjusted.interpolation_quality == "high"

    def test_adjust_settings_text_content(self):
        """Test settings adjustment for text content."""
        processor = SceneAdaptiveProcessor()

        base_settings = AdaptiveSettings()

        analysis = SceneAnalysis(
            frame_number=0, timestamp=0.0,
            primary_content=ContentType.TEXT_TITLE,
            text_regions=[TextRegion(x=10, y=10, width=300, height=50, confidence=0.9)],
        )

        adjusted = processor.adjust_settings_for_scene(base_settings, analysis)

        # Text should have preserved sharpness
        assert adjusted.text_sharpening >= 0.4
        assert adjusted.noise_reduction <= 0.2
        assert adjusted.apply_regional is True


class TestAnalyzeVideoIntelligence:
    """Tests for analyze_video_intelligence function."""

    def test_function_returns_tuple(self, tmp_path):
        """Test that function returns analyses and summary."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b'fake video data')

        with patch.object(
            SceneIntelligence,
            'analyze_video',
            return_value=[]
        ):
            with patch.object(
                SceneIntelligence,
                'get_summary',
                return_value={}
            ):
                analyses, summary = analyze_video_intelligence(video_path)

        assert isinstance(analyses, list)
        assert isinstance(summary, dict)

    def test_function_passes_sample_rate(self, tmp_path):
        """Test that sample_rate is passed correctly."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b'fake video data')

        with patch.object(
            SceneIntelligence,
            '__init__',
            return_value=None
        ) as mock_init:
            with patch.object(
                SceneIntelligence,
                'analyze_video',
                return_value=[]
            ):
                with patch.object(
                    SceneIntelligence,
                    'get_summary',
                    return_value={}
                ):
                    analyze_video_intelligence(video_path, sample_rate=0.25)

        mock_init.assert_called_once()
        # Verify sample_rate was passed (via kwargs)
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs.get('sample_rate') == 0.25
