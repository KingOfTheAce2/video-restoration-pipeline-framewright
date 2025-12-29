"""Unit tests for model download manager."""
import pytest
from pathlib import Path

from framewright.utils.model_manager import (
    ModelManager,
    ModelType,
    ModelInfo,
)


class TestModelType:
    """Tests for ModelType enum."""

    def test_enum_values(self):
        """Test that expected model type values exist."""
        assert ModelType.REALESRGAN.value == "realesrgan"
        assert ModelType.RIFE.value == "rife"
        assert ModelType.DEOLDIFY.value == "deoldify"
        assert ModelType.DDCOLOR.value == "ddcolor"
        assert ModelType.LAMA.value == "lama"
        assert ModelType.GFPGAN.value == "gfpgan"
        assert ModelType.CODEFORMER.value == "codeformer"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo instance."""
        info = ModelInfo(
            name="test-model",
            url="https://example.com/model.pth",
            size_mb=100.5,
            checksum="abc123",
            description="Test model"
        )
        assert info.name == "test-model"
        assert info.url == "https://example.com/model.pth"
        assert info.size_mb == 100.5
        assert info.checksum == "abc123"
        assert info.description == "Test model"

    def test_model_info_auto_filename(self):
        """Test auto-generated filename from URL."""
        info = ModelInfo(
            name="test",
            url="https://example.com/path/to/model.pth",
            size_mb=50.0
        )
        assert info.filename == "model.pth"

    def test_model_info_custom_filename(self):
        """Test custom filename."""
        info = ModelInfo(
            name="test",
            url="https://example.com/model.pth",
            size_mb=50.0,
            filename="custom_name.pth"
        )
        assert info.filename == "custom_name.pth"


class TestModelManager:
    """Tests for ModelManager."""

    def test_init_default_dir(self):
        """Test default model directory."""
        manager = ModelManager()
        expected_dir = Path.home() / ".framewright" / "models"
        assert manager.model_dir == expected_dir

    def test_init_custom_dir(self, temp_dir):
        """Test custom model directory."""
        custom_dir = temp_dir / "custom_models"
        manager = ModelManager(model_dir=custom_dir)
        assert manager.model_dir == custom_dir

    def test_model_dir_creation(self, temp_dir):
        """Test that model directory is created."""
        custom_dir = temp_dir / "new_models"
        manager = ModelManager(model_dir=custom_dir)
        assert custom_dir.exists()

    def test_get_model_path_known_model(self, temp_dir):
        """Test getting path for a known model."""
        manager = ModelManager(model_dir=temp_dir)
        # Use a model from the registry
        path = manager.get_model_path("realesrgan-x4plus")
        assert path.parent == temp_dir

    def test_get_model_path_unknown_model(self, temp_dir):
        """Test getting path for unknown model raises error."""
        manager = ModelManager(model_dir=temp_dir)
        with pytest.raises(ValueError):
            manager.get_model_path("nonexistent-model")


class TestModelDownload:
    """Tests for model download functionality."""

    def test_get_model_info_known(self, temp_dir):
        """Test getting info for known model."""
        manager = ModelManager(model_dir=temp_dir)
        # Should not raise for known model
        info = manager._get_model_info("realesrgan-x4plus")
        assert info.name == "realesrgan-x4plus"

    def test_get_model_info_unknown(self, temp_dir):
        """Test getting info for unknown model raises."""
        manager = ModelManager(model_dir=temp_dir)
        with pytest.raises(ValueError):
            manager._get_model_info("nonexistent")

    def test_list_available_models(self, temp_dir):
        """Test listing available models."""
        manager = ModelManager(model_dir=temp_dir)
        models = manager.list_available_models()
        assert "realesrgan-x4plus" in models
        assert "rife-v4.6" in models
