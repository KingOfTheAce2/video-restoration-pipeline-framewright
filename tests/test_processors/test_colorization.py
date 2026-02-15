"""Tests for Colorizer processor.

Tests cover enum values, configuration, initialization, backend detection,
grayscale detection, and colorization processing.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestColorModel:
    """Tests for ColorModel enum."""

    def test_color_model_values(self):
        """Test ColorModel enum has expected values."""
        from framewright.processors.colorization import ColorModel

        assert ColorModel.DEOLDIFY.value == "deoldify"
        assert ColorModel.DDCOLOR.value == "ddcolor"


class TestArtisticStyle:
    """Tests for ArtisticStyle enum."""

    def test_artistic_style_values(self):
        """Test ArtisticStyle enum has expected values."""
        from framewright.processors.colorization import ArtisticStyle

        assert ArtisticStyle.ARTISTIC.value == "artistic"
        assert ArtisticStyle.STABLE.value == "stable"
        assert ArtisticStyle.VIDEO.value == "video"


class TestColorizationConfig:
    """Tests for ColorizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from framewright.processors.colorization import (
            ColorizationConfig,
            ColorModel,
            ArtisticStyle
        )

        config = ColorizationConfig()
        assert config.model == ColorModel.DEOLDIFY
        assert config.strength == 1.0
        assert config.artistic_style == ArtisticStyle.ARTISTIC
        assert config.render_factor == 35
        assert config.skip_colored is True
        assert config.color_threshold == 10.0

    def test_custom_config(self):
        """Test custom configuration values."""
        from framewright.processors.colorization import (
            ColorizationConfig,
            ColorModel,
            ArtisticStyle
        )

        config = ColorizationConfig(
            model=ColorModel.DDCOLOR,
            strength=0.7,
            artistic_style=ArtisticStyle.STABLE,
            render_factor=25,
            skip_colored=False,
            color_threshold=15.0
        )
        assert config.model == ColorModel.DDCOLOR
        assert config.strength == 0.7
        assert config.artistic_style == ArtisticStyle.STABLE
        assert config.render_factor == 25
        assert config.skip_colored is False
        assert config.color_threshold == 15.0


class TestColorizationResult:
    """Tests for ColorizationResult dataclass."""

    def test_default_result(self):
        """Test default ColorizationResult values."""
        from framewright.processors.colorization import ColorizationResult

        result = ColorizationResult()
        assert result.frames_processed == 0
        assert result.frames_colorized == 0
        assert result.frames_skipped == 0
        assert result.failed_frames == 0
        assert result.output_dir is None


class TestColorizerInit:
    """Tests for Colorizer initialization."""

    def test_default_init(self):
        """Test Colorizer default initialization."""
        from framewright.processors.colorization import Colorizer, ColorizationConfig

        colorizer = Colorizer()
        assert isinstance(colorizer.config, ColorizationConfig)
        assert colorizer.model_dir == Colorizer.DEFAULT_MODEL_DIR
        assert colorizer._model is None

    def test_custom_config_init(self, tmp_path):
        """Test Colorizer initialization with custom config."""
        from framewright.processors.colorization import (
            Colorizer,
            ColorizationConfig,
            ColorModel
        )

        config = ColorizationConfig(model=ColorModel.DDCOLOR)
        model_dir = tmp_path / "models"

        colorizer = Colorizer(config=config, model_dir=model_dir)
        assert colorizer.config.model == ColorModel.DDCOLOR
        assert colorizer.model_dir == model_dir


class TestBackendDetection:
    """Tests for backend detection."""

    def test_is_available_true(self):
        """Test is_available returns True when backend exists."""
        from framewright.processors.colorization import Colorizer

        colorizer = Colorizer()
        colorizer._backend = 'deoldify_module'

        assert colorizer.is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False when no backend."""
        from framewright.processors.colorization import Colorizer

        colorizer = Colorizer()
        colorizer._backend = None

        assert colorizer.is_available() is False

    def test_detect_deoldify_backend_module(self):
        """Test DeOldify module backend detection."""
        from framewright.processors.colorization import Colorizer

        with patch('framewright.processors.colorization.Colorizer._detect_deoldify_backend') as mock_detect:
            mock_detect.return_value = 'deoldify_module'

            colorizer = Colorizer()
            # Override backend detection result
            colorizer._backend = mock_detect.return_value

            assert colorizer._backend == 'deoldify_module'

    def test_detect_ddcolor_backend_modelscope(self):
        """Test DDColor modelscope backend detection."""
        from framewright.processors.colorization import Colorizer, ColorizationConfig, ColorModel

        config = ColorizationConfig(model=ColorModel.DDCOLOR)

        with patch('framewright.processors.colorization.Colorizer._detect_ddcolor_backend') as mock_detect:
            mock_detect.return_value = 'ddcolor_modelscope'

            colorizer = Colorizer(config=config)
            # Override backend detection result
            colorizer._backend = mock_detect.return_value

            assert colorizer._backend == 'ddcolor_modelscope'


class TestGrayscaleDetection:
    """Tests for grayscale detection."""

    def test_is_grayscale_single_channel(self):
        """Test grayscale detection with single channel image."""
        from framewright.processors.colorization import Colorizer

        colorizer = Colorizer()

        # Single channel grayscale
        frame = np.zeros((100, 100), dtype=np.uint8)
        assert colorizer.is_grayscale(frame) is True

    def test_is_grayscale_three_channel_gray(self):
        """Test grayscale detection with 3-channel grayscale image."""
        from framewright.processors.colorization import Colorizer

        colorizer = Colorizer()

        # 3-channel but all channels identical (grayscale)
        gray_value = 128
        frame = np.full((100, 100, 3), gray_value, dtype=np.uint8)

        assert colorizer.is_grayscale(frame) == True

    def test_is_grayscale_colored_image(self):
        """Test grayscale detection with colored image."""
        from framewright.processors.colorization import Colorizer

        colorizer = Colorizer()

        # Create a colored image (red channel different from others)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red channel

        assert colorizer.is_grayscale(frame) == False

    def test_is_grayscale_threshold(self):
        """Test grayscale detection with custom threshold."""
        from framewright.processors.colorization import Colorizer, ColorizationConfig

        # High threshold - more tolerant
        config = ColorizationConfig(color_threshold=50.0)
        colorizer = Colorizer(config=config)

        # Slightly tinted image (small color differences)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame[:, :, 2] += 5  # Slight red tint

        # Should be considered grayscale with high threshold
        assert colorizer.is_grayscale(frame) == True


class TestColorizeFrame:
    """Tests for colorize_frame method."""

    def test_colorize_frame_no_backend(self):
        """Test colorize_frame returns original when no backend."""
        from framewright.processors.colorization import Colorizer

        colorizer = Colorizer()
        colorizer._backend = None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = colorizer.colorize_frame(frame)

        # Should return original frame
        np.testing.assert_array_equal(result, frame)

    def test_colorize_frame_skip_colored(self):
        """Test colorize_frame skips already colored frames."""
        from framewright.processors.colorization import Colorizer, ColorizationConfig

        config = ColorizationConfig(skip_colored=True)
        colorizer = Colorizer(config=config)
        colorizer._backend = 'deoldify_module'

        # Create colored image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red

        result = colorizer.colorize_frame(frame)

        # Should return original since it's already colored
        np.testing.assert_array_equal(result, frame)


class TestStrengthBlending:
    """Tests for strength blending."""

    def test_apply_strength_blending_full(self):
        """Test strength blending with full strength."""
        from framewright.processors.colorization import Colorizer, ColorizationConfig

        config = ColorizationConfig(strength=1.0)
        colorizer = Colorizer(config=config)

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        colorized = np.full((100, 100, 3), 255, dtype=np.uint8)

        result = colorizer._apply_strength_blending(original, colorized)

        # With strength=1.0, should get full colorized
        np.testing.assert_array_equal(result, colorized)

    def test_apply_strength_blending_half(self):
        """Test strength blending with half strength."""
        from framewright.processors.colorization import Colorizer, ColorizationConfig

        config = ColorizationConfig(strength=0.5)
        colorizer = Colorizer(config=config)

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        colorized = np.full((100, 100, 3), 200, dtype=np.uint8)

        result = colorizer._apply_strength_blending(original, colorized)

        # Should be a blend (not all zeros, not all 200)
        assert result.mean() > 0
        assert result.mean() < 200


class TestModelPaths:
    """Tests for model path helpers."""

    def test_get_deoldify_model_path_artistic(self, tmp_path):
        """Test DeOldify artistic model path."""
        from framewright.processors.colorization import (
            Colorizer,
            ColorizationConfig,
            ArtisticStyle
        )

        config = ColorizationConfig(artistic_style=ArtisticStyle.ARTISTIC)
        colorizer = Colorizer(config=config, model_dir=tmp_path)

        # Create model file
        model_dir = tmp_path / "deoldify"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "ColorizeArtistic_gen.pth"
        model_path.touch()

        result = colorizer._get_deoldify_model_path()
        assert result == model_path

    def test_get_ddcolor_model_path(self, tmp_path):
        """Test DDColor model path."""
        from framewright.processors.colorization import (
            Colorizer,
            ColorizationConfig,
            ColorModel
        )

        config = ColorizationConfig(model=ColorModel.DDCOLOR)
        colorizer = Colorizer(config=config, model_dir=tmp_path)

        # Create model file
        model_dir = tmp_path / "ddcolor"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "ddcolor_modelscope.pth"
        model_path.touch()

        result = colorizer._get_ddcolor_model_path()
        assert result == model_path


class TestAutoColorizer:
    """Tests for AutoColorizer."""

    def test_auto_colorizer_init(self):
        """Test AutoColorizer initialization."""
        from framewright.processors.colorization import AutoColorizer

        auto = AutoColorizer()
        assert auto.sample_rate == 30
        assert auto.bw_threshold == 0.7

    def test_auto_colorizer_custom_params(self):
        """Test AutoColorizer with custom parameters."""
        from framewright.processors.colorization import AutoColorizer, Colorizer

        colorizer = Colorizer()
        auto = AutoColorizer(
            colorizer=colorizer,
            sample_rate=15,
            bw_threshold=0.5
        )
        assert auto.sample_rate == 15
        assert auto.bw_threshold == 0.5
        assert auto.colorizer is colorizer
