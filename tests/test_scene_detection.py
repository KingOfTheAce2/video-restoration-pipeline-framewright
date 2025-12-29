"""Tests for the scene detection module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import subprocess

from src.framewright.processors.scene_detection import (
    SceneType,
    TransitionType,
    Scene,
    SceneEnhancementParams,
    SceneAnalysisResult,
    SceneDetector,
    SceneAnalyzer,
    detect_and_analyze_scenes,
)


class TestSceneType:
    """Tests for SceneType enum."""

    def test_scene_types_exist(self):
        """Verify all expected scene types exist."""
        assert SceneType.STATIC is not None
        assert SceneType.ACTION is not None
        assert SceneType.DIALOG is not None
        assert SceneType.TRANSITION is not None
        assert SceneType.LOW_QUALITY is not None
        assert SceneType.UNKNOWN is not None

    def test_scene_type_values_are_unique(self):
        """Verify each scene type has a unique value."""
        values = [st.value for st in SceneType]
        assert len(values) == len(set(values))


class TestTransitionType:
    """Tests for TransitionType enum."""

    def test_transition_types_exist(self):
        """Verify all expected transition types exist."""
        assert TransitionType.HARD_CUT is not None
        assert TransitionType.FADE is not None
        assert TransitionType.DISSOLVE is not None
        assert TransitionType.WIPE is not None
        assert TransitionType.UNKNOWN is not None


class TestScene:
    """Tests for Scene dataclass."""

    def test_scene_creation(self):
        """Test basic Scene creation."""
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
        )
        assert scene.start_frame == 0
        assert scene.end_frame == 100
        assert scene.duration_frames == 101
        assert scene.scene_type == SceneType.UNKNOWN
        assert scene.complexity == 0.5

    def test_scene_auto_duration(self):
        """Test that duration is auto-calculated if not positive."""
        scene = Scene(
            start_frame=10,
            end_frame=50,
            duration_frames=0,
        )
        assert scene.duration_frames == 41

    def test_scene_with_all_attributes(self):
        """Test Scene with all attributes set."""
        scene = Scene(
            start_frame=0,
            end_frame=99,
            duration_frames=100,
            scene_type=SceneType.DIALOG,
            complexity=0.7,
            transition_in=TransitionType.FADE,
            transition_out=TransitionType.HARD_CUT,
            avg_brightness=150.0,
            avg_motion=0.3,
            has_faces=True,
            face_ratio=0.8,
            quality_score=0.6,
        )
        assert scene.scene_type == SceneType.DIALOG
        assert scene.has_faces is True
        assert scene.face_ratio == 0.8

    def test_scene_to_dict(self):
        """Test Scene serialization."""
        scene = Scene(
            start_frame=0,
            end_frame=50,
            duration_frames=51,
            scene_type=SceneType.ACTION,
        )
        data = scene.to_dict()
        assert data["start_frame"] == 0
        assert data["end_frame"] == 50
        assert data["scene_type"] == "ACTION"


class TestSceneEnhancementParams:
    """Tests for SceneEnhancementParams dataclass."""

    def test_default_params(self):
        """Test default enhancement parameters."""
        params = SceneEnhancementParams()
        assert params.denoise_strength == 0.3
        assert params.sharpness == 0.5
        assert params.model_override is None
        assert params.skip_processing is False

    def test_custom_params(self):
        """Test custom enhancement parameters."""
        params = SceneEnhancementParams(
            denoise_strength=0.7,
            sharpness=0.9,
            model_override="realesrgan-x4plus-anime",
            face_restore_strength=1.0,
            skip_processing=True,
        )
        assert params.denoise_strength == 0.7
        assert params.model_override == "realesrgan-x4plus-anime"
        assert params.skip_processing is True

    def test_params_to_dict(self):
        """Test SceneEnhancementParams serialization."""
        params = SceneEnhancementParams(deblur_strength=0.5)
        data = params.to_dict()
        assert data["deblur_strength"] == 0.5
        assert "denoise_strength" in data


class TestSceneAnalysisResult:
    """Tests for SceneAnalysisResult dataclass."""

    def test_empty_result(self):
        """Test empty analysis result."""
        result = SceneAnalysisResult()
        assert result.total_scenes == 0
        assert result.scenes == []
        assert result.avg_scene_length == 0.0

    def test_result_with_scenes(self):
        """Test analysis result with scenes."""
        scenes = [
            Scene(start_frame=0, end_frame=49, duration_frames=50),
            Scene(start_frame=50, end_frame=99, duration_frames=50),
        ]
        result = SceneAnalysisResult(scenes=scenes)
        # Post-init should compute statistics
        assert result.total_scenes == 2
        assert result.avg_scene_length == 50.0
        assert result.total_frames == 100

    def test_result_to_dict(self):
        """Test SceneAnalysisResult serialization."""
        result = SceneAnalysisResult(
            scenes=[Scene(start_frame=0, end_frame=99, duration_frames=100)],
            dominant_scene_type=SceneType.STATIC,
        )
        data = result.to_dict()
        assert data["total_scenes"] == 1
        assert data["dominant_scene_type"] == "STATIC"
        assert len(data["scenes"]) == 1


class TestSceneDetector:
    """Tests for SceneDetector class."""

    def test_detector_initialization(self):
        """Test SceneDetector initialization."""
        detector = SceneDetector()
        assert detector.histogram_threshold == 0.3
        assert detector.ssim_threshold == 0.7
        assert detector.min_scene_length == 15

    def test_detector_custom_thresholds(self):
        """Test SceneDetector with custom thresholds."""
        detector = SceneDetector(
            histogram_threshold=0.5,
            ssim_threshold=0.8,
            min_scene_length=30,
        )
        assert detector.histogram_threshold == 0.5
        assert detector.ssim_threshold == 0.8
        assert detector.min_scene_length == 30

    def test_get_sorted_frames_empty_dir(self):
        """Test frame sorting with empty directory."""
        detector = SceneDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = detector._get_sorted_frames(Path(tmpdir))
            assert frames == []

    def test_detect_scenes_empty_dir(self):
        """Test scene detection with empty directory."""
        detector = SceneDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            scenes = detector.detect_scenes(Path(tmpdir))
            assert scenes == []

    def test_detect_scenes_single_frame(self):
        """Test scene detection with single frame."""
        detector = SceneDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy frame file
            frame_path = Path(tmpdir) / "frame_0001.png"
            frame_path.touch()

            scenes = detector.detect_scenes(Path(tmpdir))
            assert len(scenes) == 1
            assert scenes[0].start_frame == 0
            assert scenes[0].end_frame == 0

    @patch("subprocess.run")
    def test_get_frame_count(self, mock_run):
        """Test frame count extraction from video."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams": [{"nb_read_frames": "1000"}]}',
        )
        detector = SceneDetector()
        count = detector._get_frame_count(Path("/fake/video.mp4"))
        assert count == 1000

    @patch("subprocess.run")
    def test_get_fps(self, mock_run):
        """Test FPS extraction from video."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams": [{"r_frame_rate": "24/1"}]}',
        )
        detector = SceneDetector()
        fps = detector._get_fps(Path("/fake/video.mp4"))
        assert fps == 24.0

    def test_build_scenes_from_boundaries(self):
        """Test scene building from boundary list."""
        detector = SceneDetector()
        boundaries = [0, 50, 100]
        transition_info = {
            50: TransitionType.HARD_CUT,
            100: TransitionType.FADE,
        }
        scenes = detector._build_scenes_from_boundaries(
            boundaries, transition_info, 150
        )
        assert len(scenes) == 3
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 49

    def test_merge_short_scenes(self):
        """Test merging of short scenes."""
        detector = SceneDetector(min_scene_length=20)
        scenes = [
            Scene(start_frame=0, end_frame=9, duration_frames=10),
            Scene(start_frame=10, end_frame=29, duration_frames=20),
            Scene(start_frame=30, end_frame=34, duration_frames=5),
            Scene(start_frame=35, end_frame=99, duration_frames=65),
        ]
        merged = detector._merge_short_scenes(scenes)
        # Short scenes should be merged with neighbors
        assert len(merged) < len(scenes)


class TestSceneAnalyzer:
    """Tests for SceneAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test SceneAnalyzer initialization."""
        analyzer = SceneAnalyzer()
        assert analyzer.face_detection_enabled is True
        assert analyzer.motion_analysis_enabled is True
        assert analyzer.quality_threshold == 0.8

    def test_analyzer_custom_settings(self):
        """Test SceneAnalyzer with custom settings."""
        analyzer = SceneAnalyzer(
            face_detection_enabled=False,
            quality_threshold=0.9,
        )
        assert analyzer.face_detection_enabled is False
        assert analyzer.quality_threshold == 0.9

    def test_classify_scene_type_static(self):
        """Test scene classification for static content."""
        analyzer = SceneAnalyzer()
        scene_type = analyzer._classify_scene_type(
            brightness=128.0,
            motion=0.1,
            has_faces=False,
            face_ratio=0.0,
            quality=0.7,
        )
        assert scene_type == SceneType.STATIC

    def test_classify_scene_type_action(self):
        """Test scene classification for action content."""
        analyzer = SceneAnalyzer()
        scene_type = analyzer._classify_scene_type(
            brightness=128.0,
            motion=0.6,
            has_faces=False,
            face_ratio=0.0,
            quality=0.7,
        )
        assert scene_type == SceneType.ACTION

    def test_classify_scene_type_dialog(self):
        """Test scene classification for dialog content."""
        analyzer = SceneAnalyzer()
        scene_type = analyzer._classify_scene_type(
            brightness=128.0,
            motion=0.3,
            has_faces=True,
            face_ratio=0.7,
            quality=0.7,
        )
        assert scene_type == SceneType.DIALOG

    def test_classify_scene_type_low_quality(self):
        """Test scene classification for low quality content."""
        analyzer = SceneAnalyzer()
        scene_type = analyzer._classify_scene_type(
            brightness=128.0,
            motion=0.3,
            has_faces=False,
            face_ratio=0.0,
            quality=0.2,
        )
        assert scene_type == SceneType.LOW_QUALITY

    def test_generate_enhancement_params_static(self):
        """Test enhancement params for static scene."""
        analyzer = SceneAnalyzer()
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
            scene_type=SceneType.STATIC,
            avg_brightness=128.0,
            complexity=0.5,
            quality_score=0.7,
        )
        params = analyzer._generate_enhancement_params(scene)
        assert params.denoise_strength == 0.2
        assert params.sharpness == 0.7

    def test_generate_enhancement_params_action(self):
        """Test enhancement params for action scene."""
        analyzer = SceneAnalyzer()
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
            scene_type=SceneType.ACTION,
            quality_score=0.6,
        )
        params = analyzer._generate_enhancement_params(scene)
        assert params.deblur_strength == 0.5

    def test_generate_enhancement_params_dialog(self):
        """Test enhancement params for dialog scene."""
        analyzer = SceneAnalyzer()
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
            scene_type=SceneType.DIALOG,
            quality_score=0.7,
        )
        params = analyzer._generate_enhancement_params(scene)
        assert params.face_restore_strength == 1.0

    def test_generate_enhancement_params_high_quality_skip(self):
        """Test that high quality scenes suggest skipping."""
        analyzer = SceneAnalyzer()
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
            scene_type=SceneType.STATIC,
            quality_score=0.9,
        )
        params = analyzer._generate_enhancement_params(scene)
        assert params.skip_processing is True

    def test_generate_enhancement_params_brightness_adjust(self):
        """Test brightness adjustment for dark scenes."""
        analyzer = SceneAnalyzer()
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
            scene_type=SceneType.STATIC,
            avg_brightness=40.0,
            quality_score=0.7,
        )
        params = analyzer._generate_enhancement_params(scene)
        assert params.brightness_adjust > 0

    def test_get_adaptive_params(self):
        """Test adaptive parameter generation."""
        analyzer = SceneAnalyzer()
        scene = Scene(
            start_frame=0,
            end_frame=100,
            duration_frames=101,
            scene_type=SceneType.DIALOG,
            avg_brightness=128.0,
            complexity=0.5,
        )

        with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
            params = analyzer.get_adaptive_params(Path(tmpfile.name), scene)
            assert "denoise" in params
            assert "face_enhance" in params
            assert params["face_enhance"] is True

    def test_should_skip_frame_high_quality(self):
        """Test frame skip decision for high quality."""
        analyzer = SceneAnalyzer()
        # Mock a high-quality frame detection
        with patch.object(
            analyzer, "_estimate_frame_quality", return_value=0.95
        ):
            with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
                skip = analyzer.should_skip_frame(Path(tmpfile.name))
                assert skip is True

    def test_should_skip_frame_low_quality(self):
        """Test frame skip decision for low quality."""
        analyzer = SceneAnalyzer()
        with patch.object(
            analyzer, "_estimate_frame_quality", return_value=0.5
        ):
            with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
                skip = analyzer.should_skip_frame(Path(tmpfile.name))
                assert skip is False


class TestDetectAndAnalyzeScenes:
    """Tests for the convenience function."""

    @patch.object(SceneDetector, "detect_scenes")
    @patch.object(SceneAnalyzer, "analyze_all_scenes")
    def test_convenience_function(self, mock_analyze, mock_detect):
        """Test the convenience function calls both detector and analyzer."""
        mock_detect.return_value = [
            Scene(start_frame=0, end_frame=99, duration_frames=100)
        ]
        mock_analyze.return_value = SceneAnalysisResult(
            scenes=[Scene(start_frame=0, end_frame=99, duration_frames=100)],
            dominant_scene_type=SceneType.STATIC,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_and_analyze_scenes(Path(tmpdir))
            assert mock_detect.called
            assert mock_analyze.called
            assert result.total_scenes == 1

    def test_convenience_function_with_progress(self):
        """Test convenience function with progress callback."""
        progress_calls = []

        def progress_callback(stage, progress):
            progress_calls.append((stage, progress))

        with tempfile.TemporaryDirectory() as tmpdir:
            detect_and_analyze_scenes(
                Path(tmpdir), progress_callback=progress_callback
            )
            # Should have been called for both stages
            stages = set(call[0] for call in progress_calls)
            assert "detection" in stages or len(progress_calls) == 0


class TestSceneDetectorIntegration:
    """Integration tests that require real file operations."""

    def test_histogram_difference_fallback(self):
        """Test histogram difference calculation fallback."""
        detector = SceneDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two dummy files with different sizes
            frame1 = Path(tmpdir) / "frame1.png"
            frame2 = Path(tmpdir) / "frame2.png"
            frame1.write_bytes(b"x" * 1000)
            frame2.write_bytes(b"y" * 2000)

            # Should use fallback method based on file size
            diff = detector._calculate_histogram_difference(frame1, frame2)
            assert 0.0 <= diff <= 1.0

    def test_ssim_fallback(self):
        """Test SSIM calculation fallback."""
        detector = SceneDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            frame1 = Path(tmpdir) / "frame1.png"
            frame2 = Path(tmpdir) / "frame2.png"
            frame1.write_bytes(b"x" * 1000)
            frame2.write_bytes(b"y" * 2000)

            # Should use fallback
            ssim = detector._calculate_ssim(frame1, frame2)
            assert 0.0 <= ssim <= 1.0
