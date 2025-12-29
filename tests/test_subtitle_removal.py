"""Tests for burnt-in subtitle detection and removal.

These tests validate the OCR-based subtitle detection and inpainting-based
removal functionality for hard-coded subtitles in video frames.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framewright.processors.subtitle_removal import (
    AutoSubtitleRemover,
    OCREngine,
    SubtitleBox,
    SubtitleDetector,
    SubtitleRegion,
    SubtitleRemovalConfig,
    SubtitleRemovalResult,
    SubtitleRemover,
    check_ocr_available,
    detect_burnt_subtitles,
    remove_burnt_subtitles,
)


class TestSubtitleBox:
    """Test SubtitleBox dataclass."""

    def test_basic_creation(self):
        """Test creating a SubtitleBox."""
        box = SubtitleBox(x=10, y=20, width=100, height=30)
        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 30
        assert box.text == ""
        assert box.confidence == 1.0

    def test_with_text_and_confidence(self):
        """Test SubtitleBox with text and confidence."""
        box = SubtitleBox(
            x=10, y=20, width=100, height=30,
            text="Hello World", confidence=0.95
        )
        assert box.text == "Hello World"
        assert box.confidence == 0.95

    def test_x2_y2_properties(self):
        """Test computed x2 and y2 properties."""
        box = SubtitleBox(x=10, y=20, width=100, height=30)
        assert box.x2 == 110
        assert box.y2 == 50

    def test_area_property(self):
        """Test computed area property."""
        box = SubtitleBox(x=0, y=0, width=100, height=50)
        assert box.area == 5000

    def test_to_tuple(self):
        """Test conversion to tuple."""
        box = SubtitleBox(x=10, y=20, width=100, height=30)
        assert box.to_tuple() == (10, 20, 110, 50)

    def test_expand(self):
        """Test expanding box dimensions."""
        box = SubtitleBox(x=50, y=50, width=100, height=30)
        expanded = box.expand(10)
        assert expanded.x == 40
        assert expanded.y == 40
        assert expanded.width == 120
        assert expanded.height == 50

    def test_expand_clips_to_zero(self):
        """Test that expand clips negative coordinates to zero."""
        box = SubtitleBox(x=5, y=5, width=100, height=30)
        expanded = box.expand(10)
        assert expanded.x == 0
        assert expanded.y == 0


class TestSubtitleRemovalConfig:
    """Test SubtitleRemovalConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubtitleRemovalConfig()
        assert config.ocr_engine == OCREngine.AUTO
        assert config.region == SubtitleRegion.BOTTOM_THIRD
        assert config.min_confidence == 0.5
        assert config.text_expansion == 5
        assert config.inpainting_method == "lama"
        assert config.languages == ["en"]
        assert config.skip_frames_without_text is True
        assert config.temporal_smoothing is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SubtitleRemovalConfig(
            ocr_engine=OCREngine.EASYOCR,
            region=SubtitleRegion.TOP_QUARTER,
            min_confidence=0.7,
            languages=["en", "zh"]
        )
        assert config.ocr_engine == OCREngine.EASYOCR
        assert config.region == SubtitleRegion.TOP_QUARTER
        assert config.min_confidence == 0.7
        assert config.languages == ["en", "zh"]

    def test_custom_region(self):
        """Test custom region configuration."""
        config = SubtitleRemovalConfig(
            region=SubtitleRegion.CUSTOM,
            custom_region=(0.1, 0.8, 0.8, 0.15)
        )
        assert config.region == SubtitleRegion.CUSTOM
        assert config.custom_region == (0.1, 0.8, 0.8, 0.15)


class TestSubtitleRemovalResult:
    """Test SubtitleRemovalResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = SubtitleRemovalResult()
        assert result.frames_processed == 0
        assert result.frames_with_subtitles == 0
        assert result.frames_cleaned == 0
        assert result.frames_skipped == 0
        assert result.failed_frames == 0
        assert result.output_dir is None
        assert result.detected_texts == []


class TestSubtitleDetector:
    """Test SubtitleDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = SubtitleDetector()
        assert detector.config is not None

    def test_initialization_with_config(self):
        """Test detector initialization with custom config."""
        config = SubtitleRemovalConfig(
            ocr_engine=OCREngine.TESSERACT,
            languages=["en", "de"]
        )
        detector = SubtitleDetector(config)
        assert detector.config == config

    def test_get_region_bounds_bottom_third(self):
        """Test region bounds calculation for bottom_third."""
        config = SubtitleRemovalConfig(region=SubtitleRegion.BOTTOM_THIRD)
        detector = SubtitleDetector(config)

        x, y, w, h = detector.get_region_bounds(1080, 1920)
        # Bottom third: y starts at 67% of height
        assert x == 0
        assert y == int(1080 * 0.67)
        assert w == 1920
        assert h == int(1080 * 0.33)

    def test_get_region_bounds_bottom_quarter(self):
        """Test region bounds calculation for bottom_quarter."""
        config = SubtitleRemovalConfig(region=SubtitleRegion.BOTTOM_QUARTER)
        detector = SubtitleDetector(config)

        x, y, w, h = detector.get_region_bounds(1080, 1920)
        # Bottom quarter: y starts at 75% of height
        assert x == 0
        assert y == int(1080 * 0.75)
        assert w == 1920
        assert h == int(1080 * 0.25)

    def test_get_region_bounds_top_quarter(self):
        """Test region bounds calculation for top_quarter."""
        config = SubtitleRemovalConfig(region=SubtitleRegion.TOP_QUARTER)
        detector = SubtitleDetector(config)

        x, y, w, h = detector.get_region_bounds(1080, 1920)
        # Top quarter: y starts at 0
        assert x == 0
        assert y == 0
        assert w == 1920
        assert h == int(1080 * 0.25)

    def test_get_region_bounds_full_frame(self):
        """Test region bounds calculation for full_frame."""
        config = SubtitleRemovalConfig(region=SubtitleRegion.FULL_FRAME)
        detector = SubtitleDetector(config)

        x, y, w, h = detector.get_region_bounds(1080, 1920)
        assert x == 0
        assert y == 0
        assert w == 1920
        assert h == 1080

    def test_get_region_bounds_custom(self):
        """Test region bounds calculation for custom region."""
        config = SubtitleRemovalConfig(
            region=SubtitleRegion.CUSTOM,
            custom_region=(0.1, 0.8, 0.8, 0.15)
        )
        detector = SubtitleDetector(config)

        x, y, w, h = detector.get_region_bounds(1080, 1920)
        assert x == int(1920 * 0.1)
        assert y == int(1080 * 0.8)
        assert w == int(1920 * 0.8)
        assert h == int(1080 * 0.15)

    @patch('framewright.processors.subtitle_removal.SubtitleDetector._check_engine')
    def test_is_available_when_backend_exists(self, mock_check):
        """Test is_available returns True when backend is available."""
        mock_check.return_value = 'easyocr'
        detector = SubtitleDetector()
        assert detector.is_available() is True

    @patch('framewright.processors.subtitle_removal.SubtitleDetector._check_engine')
    def test_is_available_when_no_backend(self, mock_check):
        """Test is_available returns False when no backend."""
        mock_check.return_value = None
        detector = SubtitleDetector()
        assert detector.is_available() is False


class TestSubtitleRemover:
    """Test SubtitleRemover class."""

    def test_initialization(self):
        """Test remover initialization."""
        remover = SubtitleRemover()
        assert remover.config is not None
        assert remover.detector is not None

    def test_initialization_with_config(self):
        """Test remover initialization with custom config."""
        config = SubtitleRemovalConfig(
            inpainting_method='opencv',
            text_expansion=10
        )
        remover = SubtitleRemover(config)
        assert remover.config == config
        assert remover.config.inpainting_method == 'opencv'

    def test_create_text_mask_empty_boxes(self):
        """Test creating mask with no boxes."""
        remover = SubtitleRemover()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mask = remover.create_text_mask(frame, [])
        assert mask.shape == (1080, 1920)
        assert np.all(mask == 0)

    def test_create_text_mask_with_boxes(self):
        """Test creating mask with boxes."""
        remover = SubtitleRemover()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        boxes = [
            SubtitleBox(x=100, y=900, width=400, height=50),
            SubtitleBox(x=600, y=900, width=400, height=50),
        ]

        mask = remover.create_text_mask(frame, boxes)
        assert mask.shape == (1080, 1920)

        # Check that mask has non-zero values where boxes are
        # (accounting for text_expansion and dilation)
        assert np.any(mask[890:960, 90:520] == 255)
        assert np.any(mask[890:960, 590:1020] == 255)


class TestAutoSubtitleRemover:
    """Test AutoSubtitleRemover class."""

    def test_initialization(self):
        """Test auto remover initialization."""
        auto_remover = AutoSubtitleRemover()
        assert auto_remover.remover is not None
        assert auto_remover.sample_rate == 30
        assert auto_remover.detection_threshold == 0.3

    def test_initialization_custom_params(self):
        """Test auto remover with custom parameters."""
        auto_remover = AutoSubtitleRemover(
            sample_rate=50,
            detection_threshold=0.5
        )
        assert auto_remover.sample_rate == 50
        assert auto_remover.detection_threshold == 0.5


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_check_ocr_available(self):
        """Test check_ocr_available function."""
        results = check_ocr_available()
        assert 'easyocr' in results
        assert 'paddleocr' in results
        assert 'tesseract' in results
        # Values should be booleans
        assert all(isinstance(v, bool) for v in results.values())


class TestSubtitleRemovalIntegration:
    """Integration tests for subtitle removal."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_process_empty_directory(self, temp_dir):
        """Test processing an empty directory."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        remover = SubtitleRemover()
        result = remover.process_directory(input_dir, output_dir)

        assert result.frames_processed == 0
        assert result.output_dir == output_dir

    @pytest.mark.skipif(
        not any(check_ocr_available().values()),
        reason="No OCR engine available"
    )
    def test_process_frame_without_subtitles(self, temp_dir):
        """Test processing a frame without subtitles."""
        import cv2

        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        # Create a blank frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "frame_0001.png"), frame)

        remover = SubtitleRemover()
        result = remover.process_directory(input_dir, output_dir)

        assert result.frames_processed == 1
        assert result.frames_skipped == 1 or result.frames_cleaned == 0
        assert (output_dir / "frame_0001.png").exists()


class TestOCREngineEnum:
    """Test OCREngine enum."""

    def test_enum_values(self):
        """Test OCREngine enum values."""
        assert OCREngine.TESSERACT.value == "tesseract"
        assert OCREngine.EASYOCR.value == "easyocr"
        assert OCREngine.PADDLEOCR.value == "paddleocr"
        assert OCREngine.AUTO.value == "auto"


class TestSubtitleRegionEnum:
    """Test SubtitleRegion enum."""

    def test_enum_values(self):
        """Test SubtitleRegion enum values."""
        assert SubtitleRegion.BOTTOM_THIRD.value == "bottom_third"
        assert SubtitleRegion.BOTTOM_QUARTER.value == "bottom_quarter"
        assert SubtitleRegion.TOP_QUARTER.value == "top_quarter"
        assert SubtitleRegion.FULL_FRAME.value == "full_frame"
        assert SubtitleRegion.CUSTOM.value == "custom"
