"""Unit tests for watermark removal processor."""
import pytest
from pathlib import Path

from framewright.processors.watermark_removal import (
    WatermarkRemover,
    WatermarkConfig,
    WatermarkRemovalResult,
    WatermarkPosition,
)


class TestWatermarkPosition:
    """Tests for WatermarkPosition enum."""

    def test_enum_values(self):
        """Test that expected position values exist."""
        assert WatermarkPosition.TOP_LEFT is not None
        assert WatermarkPosition.TOP_RIGHT is not None
        assert WatermarkPosition.BOTTOM_LEFT is not None
        assert WatermarkPosition.BOTTOM_RIGHT is not None
        assert WatermarkPosition.CENTER is not None
        assert WatermarkPosition.CUSTOM is not None


class TestWatermarkConfig:
    """Tests for WatermarkConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = WatermarkConfig()
        assert config.mask_path is None
        assert config.auto_detect is False
        assert config.detection_threshold == 0.5
        assert config.margin_percent == 0.15
        assert config.dilate_mask == 5

    def test_custom_values(self, temp_dir):
        """Test custom config values."""
        mask_path = temp_dir / "mask.png"
        config = WatermarkConfig(
            mask_path=mask_path,
            auto_detect=True,
            detection_threshold=0.7
        )
        assert config.mask_path == mask_path
        assert config.auto_detect is True
        assert config.detection_threshold == 0.7

    def test_invalid_threshold_high(self):
        """Test that high threshold raises ValueError."""
        with pytest.raises(ValueError):
            WatermarkConfig(detection_threshold=1.5)

    def test_invalid_threshold_low(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError):
            WatermarkConfig(detection_threshold=-0.1)

    def test_invalid_margin_high(self):
        """Test that high margin raises ValueError."""
        with pytest.raises(ValueError):
            WatermarkConfig(margin_percent=0.6)

    def test_invalid_margin_low(self):
        """Test that negative margin raises ValueError."""
        with pytest.raises(ValueError):
            WatermarkConfig(margin_percent=-0.1)


class TestWatermarkRemovalResult:
    """Tests for WatermarkRemovalResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = WatermarkRemovalResult()
        assert result.frames_processed == 0
        assert result.frames_modified == 0
        assert result.watermarks_detected == 0
        assert result.errors == []

    def test_custom_values(self):
        """Test custom result values."""
        result = WatermarkRemovalResult(
            frames_processed=100,
            frames_modified=95,
            watermarks_detected=95,
            errors=["error1", "error2"]
        )
        assert result.frames_processed == 100
        assert result.frames_modified == 95
        assert result.watermarks_detected == 95
        assert len(result.errors) == 2


class TestWatermarkRemover:
    """Tests for WatermarkRemover class."""

    def test_init_default(self):
        """Test default initialization."""
        remover = WatermarkRemover()
        assert remover.config is not None
        assert remover.config.auto_detect is False

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = WatermarkConfig(auto_detect=True)
        remover = WatermarkRemover(config=config)
        assert remover.config.auto_detect is True

    def test_init_with_model_dir(self, temp_dir):
        """Test initialization with custom model directory."""
        remover = WatermarkRemover(model_dir=temp_dir)
        assert remover.model_dir == temp_dir

    def test_is_available(self):
        """Test availability check."""
        remover = WatermarkRemover()
        # Returns bool indicating if watermark removal is available
        available = remover.is_available()
        assert isinstance(available, bool)


class TestWatermarkDetection:
    """Tests for watermark detection functionality."""

    def test_detect_watermarks_empty_positions(self):
        """Test detection with empty position list."""
        config = WatermarkConfig(auto_detect=True, positions=[])
        remover = WatermarkRemover(config=config)

        # Should handle empty positions gracefully
        assert remover.config.positions == []

    def test_detect_watermarks_custom_positions(self):
        """Test detection with custom positions."""
        positions = [WatermarkPosition.TOP_LEFT, WatermarkPosition.BOTTOM_RIGHT]
        config = WatermarkConfig(auto_detect=True, positions=positions)
        remover = WatermarkRemover(config=config)

        assert len(remover.config.positions) == 2
        assert WatermarkPosition.TOP_LEFT in remover.config.positions
        assert WatermarkPosition.BOTTOM_RIGHT in remover.config.positions


class TestWatermarkRemovalIntegration:
    """Integration tests for watermark removal."""

    @pytest.mark.skip(reason="Requires LaMA model and test images")
    def test_full_removal_pipeline(self, frames_dir, temp_dir):
        """Test full watermark removal pipeline."""
        pass
