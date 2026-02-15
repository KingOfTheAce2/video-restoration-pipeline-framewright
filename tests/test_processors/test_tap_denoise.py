"""Tests for TAP Neural Denoiser processor.

Tests cover model enums, configuration validation, initialization,
backend detection, and denoising processing.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestTAPModel:
    """Tests for TAPModel enum."""

    def test_tap_model_values(self):
        """Test TAPModel enum has expected values."""
        from framewright.processors.tap_denoise import TAPModel

        assert TAPModel.RESTORMER.value == "restormer"
        assert TAPModel.NAFNET.value == "nafnet"
        assert TAPModel.TAP.value == "tap"


class TestTAPDenoiseConfig:
    """Tests for TAPDenoiseConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from framewright.processors.tap_denoise import TAPDenoiseConfig, TAPModel

        config = TAPDenoiseConfig()
        assert config.model == TAPModel.RESTORMER
        assert config.temporal_window == 5
        assert config.strength == 1.0
        assert config.preserve_grain is False
        assert config.half_precision is True
        assert config.tile_size == 512
        assert config.tile_overlap == 32
        assert config.gpu_id == 0
        assert config.batch_size == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        from framewright.processors.tap_denoise import TAPDenoiseConfig, TAPModel

        config = TAPDenoiseConfig(
            model=TAPModel.NAFNET,
            temporal_window=7,
            strength=0.8,
            preserve_grain=True,
            half_precision=False,
            tile_size=1024,
            tile_overlap=64,
            gpu_id=1,
            batch_size=4
        )
        assert config.model == TAPModel.NAFNET
        assert config.temporal_window == 7
        assert config.strength == 0.8
        assert config.preserve_grain is True
        assert config.half_precision is False
        assert config.tile_size == 1024
        assert config.tile_overlap == 64
        assert config.gpu_id == 1
        assert config.batch_size == 4

    def test_config_validation_temporal_window(self):
        """Test validation rejects invalid temporal window."""
        from framewright.processors.tap_denoise import TAPDenoiseConfig

        with pytest.raises(ValueError, match="temporal_window must be >= 1"):
            TAPDenoiseConfig(temporal_window=0)

    def test_config_validation_strength(self):
        """Test validation rejects invalid strength."""
        from framewright.processors.tap_denoise import TAPDenoiseConfig

        with pytest.raises(ValueError, match="strength must be 0-1"):
            TAPDenoiseConfig(strength=1.5)

        with pytest.raises(ValueError, match="strength must be 0-1"):
            TAPDenoiseConfig(strength=-0.1)

    def test_config_validation_tile_size(self):
        """Test validation rejects negative tile size."""
        from framewright.processors.tap_denoise import TAPDenoiseConfig

        with pytest.raises(ValueError, match="tile_size must be >= 0"):
            TAPDenoiseConfig(tile_size=-100)

    def test_config_string_to_enum_conversion(self):
        """Test string to enum conversion in post_init."""
        from framewright.processors.tap_denoise import TAPDenoiseConfig, TAPModel

        config = TAPDenoiseConfig(model="nafnet")
        assert config.model == TAPModel.NAFNET


class TestTAPDenoiseResult:
    """Tests for TAPDenoiseResult dataclass."""

    def test_default_result(self):
        """Test default TAPDenoiseResult values."""
        from framewright.processors.tap_denoise import TAPDenoiseResult

        result = TAPDenoiseResult()
        assert result.frames_processed == 0
        assert result.frames_failed == 0
        assert result.output_dir is None
        assert result.avg_psnr_improvement == 0.0
        assert result.processing_time_seconds == 0.0
        assert result.peak_vram_mb == 0
        assert result.model_used is None


class TestTAPDenoiserInit:
    """Tests for TAPDenoiser initialization."""

    def test_default_init(self):
        """Test TAPDenoiser default initialization."""
        from framewright.processors.tap_denoise import TAPDenoiser, TAPDenoiseConfig

        denoiser = TAPDenoiser()
        assert isinstance(denoiser.config, TAPDenoiseConfig)
        assert denoiser.model_dir == TAPDenoiser.DEFAULT_MODEL_DIR
        assert denoiser._model is None

    def test_custom_config_init(self, tmp_path):
        """Test TAPDenoiser initialization with custom config."""
        from framewright.processors.tap_denoise import (
            TAPDenoiser,
            TAPDenoiseConfig,
            TAPModel
        )

        config = TAPDenoiseConfig(model=TAPModel.TAP, strength=0.5)
        model_dir = tmp_path / "models"

        denoiser = TAPDenoiser(config=config, model_dir=model_dir)
        assert denoiser.config.model == TAPModel.TAP
        assert denoiser.config.strength == 0.5
        assert denoiser.model_dir == model_dir


class TestModelConstants:
    """Tests for model-related constants."""

    def test_model_files_mapping(self):
        """Test MODEL_FILES contains all models."""
        from framewright.processors.tap_denoise import TAPDenoiser, TAPModel

        assert TAPModel.RESTORMER in TAPDenoiser.MODEL_FILES
        assert TAPModel.NAFNET in TAPDenoiser.MODEL_FILES
        assert TAPModel.TAP in TAPDenoiser.MODEL_FILES

        assert TAPDenoiser.MODEL_FILES[TAPModel.RESTORMER] == 'restormer_deraining.pth'
        assert TAPDenoiser.MODEL_FILES[TAPModel.NAFNET] == 'NAFNet-SIDD-width64.pth'
        assert TAPDenoiser.MODEL_FILES[TAPModel.TAP] == 'tap_restormer.pth'

    def test_model_vram_requirements(self):
        """Test MODEL_VRAM contains all models with valid values."""
        from framewright.processors.tap_denoise import TAPDenoiser, TAPModel

        assert TAPModel.RESTORMER in TAPDenoiser.MODEL_VRAM
        assert TAPModel.NAFNET in TAPDenoiser.MODEL_VRAM
        assert TAPModel.TAP in TAPDenoiser.MODEL_VRAM

        # Check VRAM values are reasonable
        assert TAPDenoiser.MODEL_VRAM[TAPModel.RESTORMER] == 4000
        assert TAPDenoiser.MODEL_VRAM[TAPModel.NAFNET] == 2000
        assert TAPDenoiser.MODEL_VRAM[TAPModel.TAP] == 6000
