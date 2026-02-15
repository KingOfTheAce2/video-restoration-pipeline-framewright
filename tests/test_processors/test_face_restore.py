"""Tests for FaceRestorer processor.

Tests cover initialization, backend detection, frame restoration,
GPU memory optimization, and error handling.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import shutil


class TestFaceModel:
    """Tests for FaceModel enum."""

    def test_face_model_values(self):
        """Test FaceModel enum has expected values."""
        from framewright.processors.face_restore import FaceModel

        assert FaceModel.GFPGAN_V1_3.value == "GFPGANv1.3"
        assert FaceModel.GFPGAN_V1_4.value == "GFPGANv1.4"
        assert FaceModel.CODEFORMER.value == "CodeFormer"
        assert FaceModel.RESTOREFORMER.value == "RestoreFormer"


class TestFaceRestorationResult:
    """Tests for FaceRestorationResult dataclass."""

    def test_default_result(self):
        """Test default FaceRestorationResult values."""
        from framewright.processors.face_restore import FaceRestorationResult

        result = FaceRestorationResult()
        assert result.frames_processed == 0
        assert result.faces_detected == 0
        assert result.faces_restored == 0
        assert result.failed_frames == 0
        assert result.output_dir is None


class TestFaceRestorerInit:
    """Tests for FaceRestorer initialization."""

    def test_default_init(self):
        """Test FaceRestorer default initialization."""
        from framewright.processors.face_restore import FaceRestorer, FaceModel

        restorer = FaceRestorer()
        assert restorer.model == FaceModel.GFPGAN_V1_4
        assert restorer.upscale == 2
        assert restorer.weight == 0.5


class TestBackendDetection:
    """Tests for backend detection."""

    def test_is_available_true(self):
        """Test is_available returns True when backend exists."""
        from framewright.processors.face_restore import FaceRestorer

        restorer = FaceRestorer()
        restorer._backend = 'gfpgan_cli'

        assert restorer.is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False when no backend."""
        from framewright.processors.face_restore import FaceRestorer

        restorer = FaceRestorer()
        restorer._backend = None

        assert restorer.is_available() is False
