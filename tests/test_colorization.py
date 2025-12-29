"""Unit tests for colorization processor."""
import pytest
from pathlib import Path

from framewright.processors.colorization import (
    Colorizer,
    ColorModel,
    ColorizationResult,
    ColorizationConfig,
    ArtisticStyle,
)


class TestColorModel:
    """Tests for ColorModel enum."""

    def test_enum_values(self):
        """Test that expected model values exist."""
        assert ColorModel.DEOLDIFY.value == "deoldify"
        assert ColorModel.DDCOLOR.value == "ddcolor"

    def test_from_string(self):
        """Test creating enum from string."""
        assert ColorModel("deoldify") == ColorModel.DEOLDIFY
        assert ColorModel("ddcolor") == ColorModel.DDCOLOR

    def test_invalid_model(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError):
            ColorModel("invalid_model")


class TestArtisticStyle:
    """Tests for ArtisticStyle enum."""

    def test_enum_values(self):
        """Test that expected style values exist."""
        assert ArtisticStyle.ARTISTIC.value == "artistic"
        assert ArtisticStyle.STABLE.value == "stable"
        assert ArtisticStyle.VIDEO.value == "video"


class TestColorizationConfig:
    """Tests for ColorizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ColorizationConfig()
        assert config.model == ColorModel.DEOLDIFY
        assert config.strength == 1.0
        assert config.artistic_style == ArtisticStyle.ARTISTIC
        assert config.render_factor == 35
        assert config.skip_colored is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ColorizationConfig(
            model=ColorModel.DDCOLOR,
            strength=0.8,
            render_factor=50
        )
        assert config.model == ColorModel.DDCOLOR
        assert config.strength == 0.8
        assert config.render_factor == 50


class TestColorizationResult:
    """Tests for ColorizationResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = ColorizationResult()
        assert result.frames_processed == 0
        assert result.frames_colorized == 0
        assert result.frames_skipped == 0
        assert result.failed_frames == 0
        assert result.output_dir is None

    def test_custom_values(self, temp_dir):
        """Test custom result values."""
        result = ColorizationResult(
            frames_processed=100,
            frames_colorized=90,
            frames_skipped=8,
            failed_frames=2,
            output_dir=temp_dir
        )
        assert result.frames_processed == 100
        assert result.frames_colorized == 90
        assert result.frames_skipped == 8
        assert result.failed_frames == 2
        assert result.output_dir == temp_dir


class TestColorizer:
    """Tests for Colorizer class."""

    def test_init_default(self):
        """Test default initialization."""
        colorizer = Colorizer()
        assert colorizer.config.model == ColorModel.DEOLDIFY

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ColorizationConfig(model=ColorModel.DDCOLOR)
        colorizer = Colorizer(config=config)
        assert colorizer.config.model == ColorModel.DDCOLOR

    def test_init_with_model_dir(self, temp_dir):
        """Test initialization with custom model directory."""
        colorizer = Colorizer(model_dir=temp_dir)
        assert colorizer.model_dir == temp_dir

    def test_is_available(self):
        """Test availability check."""
        colorizer = Colorizer()
        # Returns bool indicating if colorization is available
        available = colorizer.is_available()
        assert isinstance(available, bool)


class TestColorizationIntegration:
    """Integration tests for colorization processor."""

    @pytest.mark.skip(reason="Requires actual image files and models")
    def test_full_colorization_pipeline(self, frames_dir, temp_dir):
        """Test full colorization pipeline."""
        output_dir = temp_dir / "colorized"
        colorizer = Colorizer()

        # Would need actual implementation
        pass
