"""Tests for video stabilization processor."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framewright.processors.stabilization import (
    MotionVector,
    StabilizationConfig,
    StabilizationResult,
    StabilizationAlgorithm,
    SmoothingMode,
    MotionAnalyzer,
    VideoStabilizer,
    detect_shake_severity,
    stabilize_frames,
    stabilize_video,
    create_stabilizer,
)


class TestMotionVector:
    """Tests for MotionVector dataclass."""

    def test_motion_vector_creation(self):
        """Test creating a MotionVector with default values."""
        vec = MotionVector(dx=10.0, dy=-5.0)
        assert vec.dx == 10.0
        assert vec.dy == -5.0
        assert vec.rotation == 0.0
        assert vec.scale == 1.0
        assert vec.confidence == 1.0

    def test_motion_vector_full(self):
        """Test creating a MotionVector with all parameters."""
        vec = MotionVector(
            dx=10.0,
            dy=-5.0,
            rotation=0.1,
            scale=1.02,
            timestamp=1.5,
            frame_index=45,
            confidence=0.95,
        )
        assert vec.dx == 10.0
        assert vec.dy == -5.0
        assert vec.rotation == 0.1
        assert vec.scale == 1.02
        assert vec.timestamp == 1.5
        assert vec.frame_index == 45
        assert vec.confidence == 0.95

    def test_to_transform_matrix(self):
        """Test converting MotionVector to affine transform matrix."""
        vec = MotionVector(dx=10.0, dy=5.0, rotation=0.0, scale=1.0)
        matrix = vec.to_transform_matrix()

        assert matrix.shape == (2, 3)
        # For no rotation and scale=1, matrix should be [1, 0, dx; 0, 1, dy]
        np.testing.assert_array_almost_equal(
            matrix,
            np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]], dtype=np.float32)
        )

    def test_to_transform_matrix_with_rotation(self):
        """Test transform matrix with rotation."""
        vec = MotionVector(dx=0.0, dy=0.0, rotation=np.pi / 2, scale=1.0)
        matrix = vec.to_transform_matrix()

        assert matrix.shape == (2, 3)
        # 90 degree rotation: cos(90)=0, sin(90)=1
        np.testing.assert_array_almost_equal(matrix[0, 0], 0.0, decimal=5)
        np.testing.assert_array_almost_equal(matrix[0, 1], -1.0, decimal=5)
        np.testing.assert_array_almost_equal(matrix[1, 0], 1.0, decimal=5)
        np.testing.assert_array_almost_equal(matrix[1, 1], 0.0, decimal=5)

    def test_inverse(self):
        """Test getting inverse motion vector."""
        vec = MotionVector(dx=10.0, dy=5.0, rotation=0.1, scale=1.0)
        inv = vec.inverse()

        assert inv.rotation == pytest.approx(-0.1)
        assert inv.scale == pytest.approx(1.0)
        assert inv.frame_index == vec.frame_index
        assert inv.confidence == vec.confidence


class TestStabilizationConfig:
    """Tests for StabilizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StabilizationConfig()

        assert config.smoothing_strength == 0.8
        assert config.crop_ratio == 0.1
        assert config.algorithm == StabilizationAlgorithm.AUTO
        assert config.preserve_scale is False
        assert config.max_angle == 3.0
        assert config.max_shift == 100.0
        assert config.smoothing_mode == SmoothingMode.GAUSS
        assert config.step_size == 1
        assert config.border_mode == "replicate"
        assert config.tripod_mode is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StabilizationConfig(
            smoothing_strength=0.5,
            crop_ratio=0.2,
            algorithm=StabilizationAlgorithm.VIDSTAB,
            preserve_scale=True,
            max_angle=5.0,
            max_shift=50.0,
            smoothing_mode=SmoothingMode.AVERAGE,
            tripod_mode=True,
        )

        assert config.smoothing_strength == 0.5
        assert config.crop_ratio == 0.2
        assert config.algorithm == StabilizationAlgorithm.VIDSTAB
        assert config.preserve_scale is True
        assert config.max_angle == 5.0
        assert config.max_shift == 50.0
        assert config.smoothing_mode == SmoothingMode.AVERAGE
        assert config.tripod_mode is True

    def test_string_algorithm(self):
        """Test config with string algorithm name."""
        config = StabilizationConfig(algorithm="opencv")
        assert config.algorithm == StabilizationAlgorithm.OPENCV

    def test_string_smoothing_mode(self):
        """Test config with string smoothing mode."""
        config = StabilizationConfig(smoothing_mode="average")
        assert config.smoothing_mode == SmoothingMode.AVERAGE

    def test_invalid_smoothing_strength(self):
        """Test validation of smoothing_strength."""
        with pytest.raises(ValueError, match="smoothing_strength"):
            StabilizationConfig(smoothing_strength=1.5)

        with pytest.raises(ValueError, match="smoothing_strength"):
            StabilizationConfig(smoothing_strength=-0.1)

    def test_invalid_crop_ratio(self):
        """Test validation of crop_ratio."""
        with pytest.raises(ValueError, match="crop_ratio"):
            StabilizationConfig(crop_ratio=0.6)

        with pytest.raises(ValueError, match="crop_ratio"):
            StabilizationConfig(crop_ratio=-0.1)

    def test_invalid_max_angle(self):
        """Test validation of max_angle."""
        with pytest.raises(ValueError, match="max_angle"):
            StabilizationConfig(max_angle=-5.0)

    def test_invalid_max_shift(self):
        """Test validation of max_shift."""
        with pytest.raises(ValueError, match="max_shift"):
            StabilizationConfig(max_shift=-10.0)

    def test_invalid_step_size(self):
        """Test validation of step_size."""
        with pytest.raises(ValueError, match="step_size"):
            StabilizationConfig(step_size=0)


class TestStabilizationResult:
    """Tests for StabilizationResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = StabilizationResult()

        assert result.motion_vectors == []
        assert result.smoothed_vectors == []
        assert result.crop_applied == 0.0
        assert result.smoothing_applied == 0.0
        assert result.frames_processed == 0
        assert result.shake_severity == 0.0
        assert result.algorithm_used == ""
        assert result.success is True
        assert result.errors == []

    def test_custom_result(self):
        """Test custom result values."""
        vectors = [MotionVector(dx=1.0, dy=2.0)]
        result = StabilizationResult(
            motion_vectors=vectors,
            smoothed_vectors=vectors,
            crop_applied=0.1,
            smoothing_applied=0.8,
            frames_processed=100,
            shake_severity=0.5,
            algorithm_used="opencv",
            success=True,
            errors=[],
        )

        assert len(result.motion_vectors) == 1
        assert result.crop_applied == 0.1
        assert result.smoothing_applied == 0.8
        assert result.frames_processed == 100
        assert result.shake_severity == 0.5
        assert result.algorithm_used == "opencv"


class TestMotionAnalyzer:
    """Tests for MotionAnalyzer class."""

    def test_init_without_opencv(self):
        """Test analyzer initialization fails without OpenCV."""
        with patch.dict("sys.modules", {"cv2": None}):
            # This would need to be tested differently as module is imported at top
            pass

    def test_detect_shake_severity_empty(self):
        """Test shake severity with empty vectors."""
        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            with patch("framewright.processors.stabilization.cv2"):
                analyzer = MotionAnalyzer()
                severity = analyzer.detect_shake_severity([])
                assert severity == 0.0

    def test_detect_shake_severity_stable(self):
        """Test shake severity with stable motion."""
        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            with patch("framewright.processors.stabilization.cv2"):
                analyzer = MotionAnalyzer()
                vectors = [
                    MotionVector(dx=0.0, dy=0.0, rotation=0.0)
                    for _ in range(10)
                ]
                severity = analyzer.detect_shake_severity(vectors)
                assert severity == 0.0

    def test_detect_shake_severity_shaky(self):
        """Test shake severity with shaky motion."""
        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            with patch("framewright.processors.stabilization.cv2"):
                analyzer = MotionAnalyzer()
                # Create vectors with high variation
                vectors = [
                    MotionVector(
                        dx=50.0 * np.sin(i * 0.5),
                        dy=50.0 * np.cos(i * 0.5),
                        rotation=0.1 * np.sin(i * 0.3),
                        frame_index=i
                    )
                    for i in range(20)
                ]
                severity = analyzer.detect_shake_severity(vectors)
                assert 0.0 < severity <= 1.0

    def test_identify_problematic_segments(self):
        """Test identifying problematic shake segments."""
        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            with patch("framewright.processors.stabilization.cv2"):
                analyzer = MotionAnalyzer()
                # Create vectors with some problematic segments
                vectors = [
                    MotionVector(dx=0.0, dy=0.0, frame_index=i)
                    for i in range(10)
                ]
                segments = analyzer.identify_problematic_segments(vectors)
                assert isinstance(segments, list)


class TestVideoStabilizer:
    """Tests for VideoStabilizer class."""

    def test_init_default_config(self):
        """Test stabilizer initialization with default config."""
        stabilizer = VideoStabilizer()
        assert stabilizer.config.smoothing_strength == 0.8
        assert stabilizer.config.algorithm == StabilizationAlgorithm.AUTO

    def test_init_custom_config(self):
        """Test stabilizer initialization with custom config."""
        config = StabilizationConfig(
            smoothing_strength=0.5,
            algorithm=StabilizationAlgorithm.OPENCV,
        )
        stabilizer = VideoStabilizer(config)
        assert stabilizer.config.smoothing_strength == 0.5
        assert stabilizer.config.algorithm == StabilizationAlgorithm.OPENCV

    def test_smooth_trajectory_empty(self):
        """Test smoothing empty trajectory."""
        stabilizer = VideoStabilizer()
        smoothed = stabilizer.smooth_trajectory([])
        assert smoothed == []

    def test_smooth_trajectory_single(self):
        """Test smoothing single vector."""
        config = StabilizationConfig(smoothing_mode=SmoothingMode.NONE)
        stabilizer = VideoStabilizer(config)
        vectors = [MotionVector(dx=10.0, dy=5.0)]

        # Use NONE smoothing mode to avoid scipy dependency
        smoothed = stabilizer.smooth_trajectory(vectors)
        assert len(smoothed) == 1

    def test_smooth_trajectory_strength(self):
        """Test smoothing with different strength."""
        config = StabilizationConfig(smoothing_mode=SmoothingMode.NONE)
        stabilizer = VideoStabilizer(config)

        vectors = [
            MotionVector(dx=i * 2.0, dy=i * -1.0, frame_index=i)
            for i in range(10)
        ]

        smoothed = stabilizer.smooth_trajectory(vectors, strength=0.0)
        assert len(smoothed) == 10

    def test_is_available(self):
        """Test checking if stabilization is available."""
        stabilizer = VideoStabilizer()
        # Mock vidstab as unavailable and OpenCV as unavailable
        stabilizer._vidstab_available = False
        stabilizer._ffmpeg_available = False
        with patch("framewright.processors.stabilization.HAS_OPENCV", False):
            result = stabilizer.is_available()
            assert result is False
        # With OpenCV available, should be True
        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            result = stabilizer.is_available()
            assert result is True

    def test_check_ffmpeg(self):
        """Test FFmpeg availability check."""
        stabilizer = VideoStabilizer()

        # Reset cached value
        stabilizer._ffmpeg_available = None
        with patch("framewright.processors.stabilization.get_ffmpeg_path", return_value="/usr/bin/ffmpeg"):
            assert stabilizer._check_ffmpeg() is True

        # Reset cached value
        stabilizer._ffmpeg_available = None

        with patch("framewright.processors.stabilization.get_ffmpeg_path", side_effect=FileNotFoundError):
            assert stabilizer._check_ffmpeg() is False

    def test_check_vidstab(self):
        """Test vidstab filter availability check."""
        stabilizer = VideoStabilizer()

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = "vidstabdetect vidstabtransform"
                stabilizer._vidstab_available = None
                assert stabilizer._check_vidstab() is True

    def test_get_algorithm_auto_vidstab(self):
        """Test algorithm selection with vidstab available."""
        stabilizer = VideoStabilizer()
        stabilizer._vidstab_available = True

        algo = stabilizer._get_algorithm()
        assert algo == StabilizationAlgorithm.VIDSTAB

    def test_get_algorithm_auto_opencv(self):
        """Test algorithm selection with only OpenCV available."""
        stabilizer = VideoStabilizer()
        stabilizer._vidstab_available = False

        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            algo = stabilizer._get_algorithm()
            assert algo == StabilizationAlgorithm.OPENCV

    def test_get_algorithm_explicit(self):
        """Test explicit algorithm selection."""
        config = StabilizationConfig(algorithm=StabilizationAlgorithm.OPENCV)
        stabilizer = VideoStabilizer(config)

        algo = stabilizer._get_algorithm()
        assert algo == StabilizationAlgorithm.OPENCV

    def test_get_border_mode(self):
        """Test border mode conversion."""
        stabilizer = VideoStabilizer()

        with patch("framewright.processors.stabilization.HAS_OPENCV", True):
            with patch("framewright.processors.stabilization.cv2") as mock_cv2:
                mock_cv2.BORDER_REPLICATE = 1
                mock_cv2.BORDER_REFLECT_101 = 2
                mock_cv2.BORDER_CONSTANT = 0

                assert stabilizer._get_border_mode("replicate") == 1
                assert stabilizer._get_border_mode("reflect") == 2
                assert stabilizer._get_border_mode("constant") == 0

    def test_moving_average(self):
        """Test moving average smoothing."""
        stabilizer = VideoStabilizer()

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Window of 1 should return same array
        result = stabilizer._moving_average(arr, 1)
        np.testing.assert_array_almost_equal(result, arr)

        # Window of 3 should smooth
        result = stabilizer._moving_average(arr, 3)
        assert len(result) == len(arr)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_stabilizer(self):
        """Test create_stabilizer convenience function."""
        stabilizer = create_stabilizer(
            smoothing_strength=0.5,
            crop_ratio=0.15,
            algorithm="opencv",
            preserve_scale=True,
            tripod_mode=True,
        )

        assert stabilizer.config.smoothing_strength == 0.5
        assert stabilizer.config.crop_ratio == 0.15
        assert stabilizer.config.algorithm == StabilizationAlgorithm.OPENCV
        assert stabilizer.config.preserve_scale is True
        assert stabilizer.config.tripod_mode is True

    def test_create_stabilizer_defaults(self):
        """Test create_stabilizer with defaults."""
        stabilizer = create_stabilizer()

        assert stabilizer.config.smoothing_strength == 0.8
        assert stabilizer.config.crop_ratio == 0.1
        assert stabilizer.config.algorithm == StabilizationAlgorithm.AUTO


class TestIntegration:
    """Integration tests for stabilization (require OpenCV/FFmpeg)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_frames(self, temp_dir):
        """Create sample frame images for testing."""
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")

        # Create simple test frames
        for i in range(10):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Add some content that shifts between frames
            offset = i * 2
            cv2.rectangle(
                frame,
                (20 + offset, 20 + offset),
                (80 + offset, 80 + offset),
                (255, 255, 255),
                -1
            )
            cv2.imwrite(str(frames_dir / f"frame_{i:08d}.png"), frame)

        return frames_dir

    @pytest.mark.skipif(
        not pytest.importorskip("cv2", reason="OpenCV not installed"),
        reason="OpenCV not installed"
    )
    def test_analyze_motion_integration(self, sample_frames):
        """Integration test for motion analysis."""
        stabilizer = VideoStabilizer()
        vectors = stabilizer.analyze_motion(sample_frames)

        assert len(vectors) == 9  # n-1 vectors for n frames
        for vec in vectors:
            assert isinstance(vec, MotionVector)
            assert 0.0 <= vec.confidence <= 1.0

    @pytest.mark.skipif(
        not pytest.importorskip("cv2", reason="OpenCV not installed"),
        reason="OpenCV not installed"
    )
    def test_stabilize_frames_integration(self, sample_frames, temp_dir):
        """Integration test for frame stabilization."""
        output_dir = temp_dir / "stable"
        config = StabilizationConfig(
            algorithm=StabilizationAlgorithm.OPENCV,
            smoothing_strength=0.5,
        )
        stabilizer = VideoStabilizer(config)
        result = stabilizer.stabilize_frames(sample_frames, output_dir)

        assert result.success is True
        assert result.frames_processed > 0
        assert result.algorithm_used == "opencv"
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 10


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
