"""Tests for Frame Interpolation processor.

Tests cover smoothness levels, configuration validation, initialization,
and RIFE model settings.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestSmoothnessLevel:
    """Tests for SmoothnessLevel enum."""

    def test_smoothness_level_values(self):
        """Test SmoothnessLevel enum has expected values."""
        from framewright.processors.interpolation import SmoothnessLevel

        assert SmoothnessLevel.LOW.value == "low"
        assert SmoothnessLevel.MEDIUM.value == "medium"
        assert SmoothnessLevel.HIGH.value == "high"


class TestInterpolationConfig:
    """Tests for InterpolationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from framewright.processors.interpolation import (
            InterpolationConfig,
            SmoothnessLevel
        )

        config = InterpolationConfig()
        assert config.target_fps == 60
        assert config.smoothness == SmoothnessLevel.MEDIUM
        assert config.enable_scene_detection is True
        assert config.scene_threshold == 0.3
        assert config.enable_motion_blur_reduction is False
        assert config.rife_model == "rife-v4.6"
        assert config.gpu_id == 0

    def test_custom_config(self):
        """Test custom configuration values."""
        from framewright.processors.interpolation import (
            InterpolationConfig,
            SmoothnessLevel
        )

        config = InterpolationConfig(
            target_fps=120,
            smoothness=SmoothnessLevel.HIGH,
            enable_scene_detection=False,
            scene_threshold=0.5,
            enable_motion_blur_reduction=True,
            rife_model="rife-anime",
            gpu_id=1
        )
        assert config.target_fps == 120
        assert config.smoothness == SmoothnessLevel.HIGH
        assert config.enable_scene_detection is False
        assert config.scene_threshold == 0.5
        assert config.enable_motion_blur_reduction is True
        assert config.rife_model == "rife-anime"
        assert config.gpu_id == 1

    def test_config_validation_scene_threshold(self):
        """Test validation rejects invalid scene threshold."""
        from framewright.processors.interpolation import InterpolationConfig

        with pytest.raises(ValueError, match="scene_threshold must be between 0 and 1"):
            InterpolationConfig(scene_threshold=1.5)

        with pytest.raises(ValueError, match="scene_threshold must be between 0 and 1"):
            InterpolationConfig(scene_threshold=-0.1)

    def test_config_validation_target_fps(self):
        """Test validation rejects invalid target fps."""
        from framewright.processors.interpolation import InterpolationConfig

        with pytest.raises(ValueError, match="target_fps must be positive"):
            InterpolationConfig(target_fps=0)

        with pytest.raises(ValueError, match="target_fps must be positive"):
            InterpolationConfig(target_fps=-30)

    def test_config_string_to_enum_conversion(self):
        """Test string to enum conversion for smoothness."""
        from framewright.processors.interpolation import (
            InterpolationConfig,
            SmoothnessLevel
        )

        config = InterpolationConfig(smoothness="high")
        assert config.smoothness == SmoothnessLevel.HIGH

        config = InterpolationConfig(smoothness="low")
        assert config.smoothness == SmoothnessLevel.LOW


class TestInterpolationError:
    """Tests for InterpolationError exception."""

    def test_interpolation_error_inheritance(self):
        """Test InterpolationError inherits from Exception."""
        from framewright.processors.interpolation import InterpolationError

        assert issubclass(InterpolationError, Exception)

    def test_interpolation_error_message(self):
        """Test InterpolationError can carry message."""
        from framewright.processors.interpolation import InterpolationError

        error = InterpolationError("Test error message")
        assert str(error) == "Test error message"


class TestFrameInterpolatorInit:
    """Tests for FrameInterpolator initialization."""

    @patch('framewright.processors.interpolation.shutil.which')
    def test_default_init(self, mock_which):
        """Test FrameInterpolator default initialization."""
        mock_which.return_value = '/usr/bin/rife-ncnn-vulkan'

        from framewright.processors.interpolation import FrameInterpolator

        interpolator = FrameInterpolator()
        assert interpolator.model == "rife-v4.6"
        assert interpolator.gpu_id == 0

    @patch('framewright.processors.interpolation.shutil.which')
    def test_custom_model_init(self, mock_which):
        """Test FrameInterpolator with custom model."""
        mock_which.return_value = '/usr/bin/rife-ncnn-vulkan'

        from framewright.processors.interpolation import FrameInterpolator

        interpolator = FrameInterpolator(model="rife-anime", gpu_id=1)
        assert interpolator.model == "rife-anime"
        assert interpolator.gpu_id == 1

    @patch('framewright.processors.interpolation.shutil.which')
    def test_config_based_init(self, mock_which):
        """Test FrameInterpolator with InterpolationConfig."""
        mock_which.return_value = '/usr/bin/rife-ncnn-vulkan'

        from framewright.processors.interpolation import (
            FrameInterpolator,
            InterpolationConfig,
            SmoothnessLevel
        )

        config = InterpolationConfig(
            target_fps=120,
            smoothness=SmoothnessLevel.HIGH,
            rife_model="rife-v4.0"
        )

        interpolator = FrameInterpolator(config=config)
        assert interpolator.model == "rife-v4.0"
        assert interpolator.config.target_fps == 120
        assert interpolator.config.smoothness == SmoothnessLevel.HIGH


class TestRIFEModelSettings:
    """Tests for RIFE model settings constants."""

    def test_rife_model_settings_structure(self):
        """Test RIFE_MODEL_SETTINGS has correct structure."""
        from framewright.processors.interpolation import RIFE_MODEL_SETTINGS

        assert 'rife-v4.6' in RIFE_MODEL_SETTINGS
        assert 'rife-v4.0' in RIFE_MODEL_SETTINGS
        assert 'rife-anime' in RIFE_MODEL_SETTINGS

        # Check structure of one model
        v46 = RIFE_MODEL_SETTINGS['rife-v4.6']
        assert 'description' in v46
        assert 'strengths' in v46
        assert 'use_cases' in v46
        assert 'speed_factor' in v46

        assert isinstance(v46['strengths'], list)
        assert isinstance(v46['use_cases'], list)
        assert isinstance(v46['speed_factor'], (int, float))

    def test_supported_models_constant(self):
        """Test SUPPORTED_MODELS constant."""
        from framewright.processors.interpolation import FrameInterpolator

        assert 'rife-v2.3' in FrameInterpolator.SUPPORTED_MODELS
        assert 'rife-v4.0' in FrameInterpolator.SUPPORTED_MODELS
        assert 'rife-v4.6' in FrameInterpolator.SUPPORTED_MODELS
        assert 'rife-anime' in FrameInterpolator.SUPPORTED_MODELS

    def test_supported_target_fps_constant(self):
        """Test SUPPORTED_TARGET_FPS constant."""
        from framewright.processors.interpolation import FrameInterpolator

        assert 24 in FrameInterpolator.SUPPORTED_TARGET_FPS
        assert 30 in FrameInterpolator.SUPPORTED_TARGET_FPS
        assert 60 in FrameInterpolator.SUPPORTED_TARGET_FPS
        assert 120 in FrameInterpolator.SUPPORTED_TARGET_FPS
