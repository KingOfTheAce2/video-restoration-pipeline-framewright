"""Tests for PyTorch Real-ESRGAN processor.

Tests cover configuration validation, availability checking, frame enhancement,
error handling, and GPU memory optimization integration.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np


def create_mock_torch():
    """Create a mock torch module."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = True
    mock.cuda.get_device_properties.return_value = Mock(total_memory=24 * 1024**3)
    mock.cuda.empty_cache = Mock()
    mock.cuda.OutOfMemoryError = RuntimeError
    return mock


def create_mock_cv2():
    """Create a mock cv2 module."""
    mock = MagicMock()
    mock.imread.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock.imwrite.return_value = True
    mock.IMREAD_UNCHANGED = 1
    return mock


@pytest.fixture
def mock_realesrgan():
    """Mock realesrgan module."""
    with patch('framewright.processors.pytorch_realesrgan.RealESRGANer') as mock_class:
        mock_instance = MagicMock()
        mock_instance.enhance.return_value = (np.zeros((4320, 7680, 3), dtype=np.uint8), None)
        mock_class.return_value = mock_instance
        yield mock_class


@pytest.fixture
def mock_rrdbnet():
    """Mock RRDBNet architecture."""
    with patch('framewright.processors.pytorch_realesrgan.RRDBNet') as mock:
        yield mock


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test_frame.png"
    # Create a simple test image with cv2
    import cv2
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


class TestPyTorchESRGANConfig:
    """Tests for PyTorchESRGANConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from framewright.processors.pytorch_realesrgan import PyTorchESRGANConfig

        config = PyTorchESRGANConfig()
        assert config.model_name == "RealESRGAN_x4plus"
        assert config.scale_factor == 4
        assert config.tile_size == 0
        assert config.half_precision is True

    def test_custom_config(self):
        """Test custom configuration values."""
        from framewright.processors.pytorch_realesrgan import PyTorchESRGANConfig

        config = PyTorchESRGANConfig(
            model_name="RealESRGAN_x2plus",
            scale_factor=2,
            tile_size=256,
            half_precision=False
        )
        assert config.model_name == "RealESRGAN_x2plus"
        assert config.scale_factor == 2
        assert config.tile_size == 256
        assert config.half_precision is False

    def test_config_validation_valid_model(self):
        """Test validation accepts valid models."""
        from framewright.processors.pytorch_realesrgan import PyTorchESRGANConfig

        valid_models = [
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus",
            "realesr-animevideov3",
            "realesr-general-x4v3",
        ]
        for model in valid_models:
            config = PyTorchESRGANConfig(model_name=model)
            config.validate()  # Should not raise

    def test_config_validation_invalid_model(self):
        """Test validation rejects invalid models."""
        from framewright.processors.pytorch_realesrgan import PyTorchESRGANConfig

        config = PyTorchESRGANConfig(model_name="invalid_model")
        with pytest.raises(ValueError, match="Invalid model"):
            config.validate()

    def test_config_validation_invalid_scale(self):
        """Test validation rejects invalid scale factors."""
        from framewright.processors.pytorch_realesrgan import PyTorchESRGANConfig

        config = PyTorchESRGANConfig(scale_factor=3)
        with pytest.raises(ValueError, match="Scale factor must be 2 or 4"):
            config.validate()


class TestAvailabilityCheck:
    """Tests for is_pytorch_esrgan_available() function."""

    def test_is_available_cached_true(self):
        """Test availability check returns True when cached."""
        import framewright.processors.pytorch_realesrgan as module

        # Set cached value directly
        module._PYTORCH_ESRGAN_AVAILABLE = True

        from framewright.processors.pytorch_realesrgan import is_pytorch_esrgan_available
        assert is_pytorch_esrgan_available() is True

    def test_is_available_cached_false(self):
        """Test availability check returns False when cached."""
        import framewright.processors.pytorch_realesrgan as module

        # Set cached value directly
        module._PYTORCH_ESRGAN_AVAILABLE = False

        from framewright.processors.pytorch_realesrgan import is_pytorch_esrgan_available
        assert is_pytorch_esrgan_available() is False


class TestEnhanceFrame:
    """Tests for enhance_frame_pytorch() function."""

    def test_enhance_frame_success(self, mock_torch, mock_cv2, mock_realesrgan, mock_rrdbnet, tmp_path):
        """Test successful frame enhancement."""
        from framewright.processors.pytorch_realesrgan import (
            enhance_frame_pytorch,
            PyTorchESRGANConfig
        )

        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        config = PyTorchESRGANConfig()

        success, error = enhance_frame_pytorch(input_path, output_path, config)

        assert success is True
        assert error is None
        mock_cv2.imread.assert_called_once()
        mock_cv2.imwrite.assert_called_once()

    def test_enhance_frame_invalid_input(self, mock_torch, mock_cv2, mock_realesrgan, mock_rrdbnet, tmp_path):
        """Test enhancement with invalid input file."""
        from framewright.processors.pytorch_realesrgan import (
            enhance_frame_pytorch,
            PyTorchESRGANConfig
        )

        mock_cv2.imread.return_value = None  # Simulate failed read

        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        config = PyTorchESRGANConfig()

        success, error = enhance_frame_pytorch(input_path, output_path, config)

        assert success is False
        assert "Failed to read image" in error

    def test_enhance_frame_oom_error(self, mock_torch, mock_cv2, mock_realesrgan, mock_rrdbnet, tmp_path):
        """Test OOM error handling with cache clearing."""
        from framewright.processors.pytorch_realesrgan import (
            enhance_frame_pytorch,
            PyTorchESRGANConfig
        )

        # Simulate OOM error
        mock_upsampler = MagicMock()
        mock_upsampler.enhance.side_effect = RuntimeError("CUDA out of memory")
        mock_realesrgan.return_value = mock_upsampler

        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        config = PyTorchESRGANConfig()

        success, error = enhance_frame_pytorch(input_path, output_path, config)

        assert success is False
        assert "GPU out of memory" in error

    def test_enhance_frame_with_gpu_optimizer(self, mock_torch, mock_cv2, mock_realesrgan, mock_rrdbnet, tmp_path):
        """Test frame enhancement with GPU memory optimizer."""
        from framewright.processors.pytorch_realesrgan import (
            enhance_frame_pytorch,
            PyTorchESRGANConfig
        )

        # Mock GPU optimizer
        with patch('framewright.processors.pytorch_realesrgan.GPU_OPTIMIZER_AVAILABLE', True):
            with patch('framewright.processors.pytorch_realesrgan._gpu_optimizer') as mock_optimizer:
                mock_optimizer.get_memory_stats.return_value = Mock(available_mb=10000)
                mock_optimizer.managed_memory.return_value.__enter__ = Mock()
                mock_optimizer.managed_memory.return_value.__exit__ = Mock()

                input_path = tmp_path / "input.png"
                output_path = tmp_path / "output.png"
                config = PyTorchESRGANConfig()

                success, error = enhance_frame_pytorch(input_path, output_path, config)

                assert success is True
                mock_optimizer.get_memory_stats.assert_called()


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_convert_ncnn_model_name(self):
        """Test NCNN to PyTorch model name conversion."""
        from framewright.processors.pytorch_realesrgan import convert_ncnn_model_name

        assert convert_ncnn_model_name("realesrgan-x4plus") == "RealESRGAN_x4plus"
        assert convert_ncnn_model_name("realesrgan-x4plus-anime") == "RealESRGAN_x4plus_anime_6B"
        assert convert_ncnn_model_name("realesr-animevideov3") == "realesr-animevideov3"
        assert convert_ncnn_model_name("unknown-model") == "RealESRGAN_x4plus"  # Default

    def test_clear_upsampler_cache(self, mock_torch):
        """Test upsampler cache clearing."""
        from framewright.processors.pytorch_realesrgan import clear_upsampler_cache
        import framewright.processors.pytorch_realesrgan as module

        # Set a cached upsampler
        module._UPSAMPLER = MagicMock()

        clear_upsampler_cache()

        assert module._UPSAMPLER is None
        mock_torch.cuda.empty_cache.assert_called_once()


class TestTileSizeSelection:
    """Tests for automatic tile size selection based on VRAM."""

    def test_auto_tile_size_24gb(self, mock_torch, mock_cv2, mock_realesrgan, mock_rrdbnet, tmp_path):
        """Test auto tile size with 24GB VRAM (no tiling)."""
        from framewright.processors.pytorch_realesrgan import (
            enhance_frame_pytorch,
            PyTorchESRGANConfig
        )

        # Mock 24GB GPU
        mock_torch.cuda.get_device_properties.return_value = Mock(total_memory=24 * 1024**3)
        mock_torch.cuda.is_available.return_value = True

        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        config = PyTorchESRGANConfig(tile_size=0)  # Auto mode

        with patch('framewright.processors.pytorch_realesrgan.GPU_OPTIMIZER_AVAILABLE', True):
            with patch('framewright.processors.pytorch_realesrgan._gpu_optimizer') as mock_opt:
                mock_opt.get_memory_stats.return_value = Mock(available_mb=20000)
                mock_opt.managed_memory.return_value.__enter__ = Mock()
                mock_opt.managed_memory.return_value.__exit__ = Mock()

                enhance_frame_pytorch(input_path, output_path, config)

                # In auto mode with high VRAM, tile_size should be 0 (no tiling)
                assert config.tile_size == 0

    def test_auto_tile_size_4gb(self, mock_torch, mock_cv2, mock_realesrgan, mock_rrdbnet, tmp_path):
        """Test auto tile size with 4GB VRAM (384px tiles)."""
        from framewright.processors.pytorch_realesrgan import (
            enhance_frame_pytorch,
            PyTorchESRGANConfig
        )

        # Mock 4GB GPU
        mock_torch.cuda.get_device_properties.return_value = Mock(total_memory=4 * 1024**3)
        mock_torch.cuda.is_available.return_value = True

        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        config = PyTorchESRGANConfig(tile_size=0)  # Auto mode

        with patch('framewright.processors.pytorch_realesrgan.GPU_OPTIMIZER_AVAILABLE', True):
            with patch('framewright.processors.pytorch_realesrgan._gpu_optimizer') as mock_opt:
                mock_opt.get_memory_stats.return_value = Mock(available_mb=3000)
                mock_opt.managed_memory.return_value.__enter__ = Mock()
                mock_opt.managed_memory.return_value.__exit__ = Mock()

                enhance_frame_pytorch(input_path, output_path, config)

                # With 3GB available, should select 384px tiles
                assert config.tile_size == 384
