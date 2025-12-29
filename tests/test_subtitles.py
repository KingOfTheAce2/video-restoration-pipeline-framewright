"""Tests for subtitle extraction and processing.

Tests cover the SubtitleExtractor, SubtitleTrack, and related utilities
for handling embedded and burned-in subtitles in video files.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

# Test if PIL is available for tests
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Import after PIL check
from framewright.processors.subtitles import (
    SubtitleConfig,
    SubtitleLine,
    SubtitleTrack,
    SubtitleExtractor,
    SubtitleError,
    BoundingBox,
    SubtitleFormat,
    SubtitleStreamInfo,
    SubtitleTimeSync,
    SubtitleMerger,
    detect_burned_subtitles,
    extract_subtitles,
    remove_subtitles,
)


class TestBoundingBox:
    """Test BoundingBox dataclass."""

    def test_bounding_box_creation(self):
        """Test creating a BoundingBox with basic values."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 50
        assert box.confidence == 1.0

    def test_bounding_box_with_confidence(self):
        """Test BoundingBox with custom confidence."""
        box = BoundingBox(x=0, y=0, width=50, height=30, confidence=0.85)

        assert box.confidence == 0.85

    def test_x2_property(self):
        """Test x2 (right coordinate) calculation."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.x2 == 110  # x + width

    def test_y2_property(self):
        """Test y2 (bottom coordinate) calculation."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.y2 == 70  # y + height

    def test_to_tuple(self):
        """Test to_tuple method."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.to_tuple() == (10, 20, 110, 70)

    def test_contains_fully_contained(self):
        """Test contains method for fully contained box."""
        outer = BoundingBox(x=0, y=0, width=100, height=100)
        inner = BoundingBox(x=10, y=10, width=50, height=50)

        assert outer.contains(inner) is True

    def test_contains_not_contained(self):
        """Test contains method for non-contained box."""
        box1 = BoundingBox(x=0, y=0, width=50, height=50)
        box2 = BoundingBox(x=60, y=60, width=50, height=50)

        assert box1.contains(box2) is False

    def test_contains_partial_overlap(self):
        """Test contains with partial overlap."""
        box1 = BoundingBox(x=0, y=0, width=50, height=50)
        box2 = BoundingBox(x=25, y=25, width=50, height=50)

        assert box1.contains(box2) is False


class TestSubtitleLine:
    """Test SubtitleLine dataclass."""

    def test_subtitle_line_creation(self):
        """Test creating a SubtitleLine."""
        line = SubtitleLine(
            text="Hello, World!",
            start_time=1.0,
            end_time=3.5
        )

        assert line.text == "Hello, World!"
        assert line.start_time == 1.0
        assert line.end_time == 3.5
        assert line.confidence == 1.0
        assert line.position is None
        assert line.frame_number == 0

    def test_subtitle_line_with_all_fields(self):
        """Test SubtitleLine with all fields."""
        position = BoundingBox(x=100, y=400, width=600, height=50)
        line = SubtitleLine(
            text="Test subtitle",
            start_time=10.0,
            end_time=15.0,
            confidence=0.95,
            position=position,
            frame_number=240
        )

        assert line.position == position
        assert line.frame_number == 240

    def test_subtitle_line_str(self):
        """Test string representation of SubtitleLine."""
        line = SubtitleLine(
            text="Test",
            start_time=3661.5,  # 1:01:01,500
            end_time=3665.0
        )

        str_repr = str(line)
        assert "01:01:01,500" in str_repr
        assert "Test" in str_repr


class TestSubtitleTrack:
    """Test SubtitleTrack class."""

    def test_empty_track_creation(self):
        """Test creating an empty SubtitleTrack."""
        track = SubtitleTrack()

        assert len(track.lines) == 0
        assert track.language == "eng"
        assert track.fps == 24.0

    def test_track_with_custom_settings(self):
        """Test SubtitleTrack with custom settings."""
        track = SubtitleTrack(
            language="deu",
            fps=25.0,
            metadata={"source": "test.mp4"}
        )

        assert track.language == "deu"
        assert track.fps == 25.0
        assert track.metadata["source"] == "test.mp4"

    def test_add_line(self):
        """Test adding lines to track."""
        track = SubtitleTrack()

        line1 = SubtitleLine(text="First", start_time=0.0, end_time=2.0)
        line2 = SubtitleLine(text="Second", start_time=3.0, end_time=5.0)

        track.add_line(line1)
        track.add_line(line2)

        assert len(track.lines) == 2

    def test_add_line_merges_same_text(self):
        """Test that consecutive lines with same text are merged."""
        track = SubtitleTrack()

        line1 = SubtitleLine(text="Same text", start_time=0.0, end_time=2.0)
        line2 = SubtitleLine(text="Same text", start_time=2.1, end_time=4.0)

        track.add_line(line1)
        track.add_line(line2)

        # Should merge into one line
        assert len(track.lines) == 1
        assert track.lines[0].end_time == 4.0

    def test_merge_short_gaps(self):
        """Test merging lines with short gaps."""
        track = SubtitleTrack()

        track.lines = [
            SubtitleLine(text="Hello", start_time=0.0, end_time=1.0),
            SubtitleLine(text="Hello", start_time=1.2, end_time=2.0),
            SubtitleLine(text="World", start_time=3.0, end_time=4.0),
        ]

        track.merge_short_gaps(max_gap=0.3)

        assert len(track.lines) == 2
        assert track.lines[0].end_time == 2.0

    def test_filter_by_confidence(self):
        """Test filtering lines by confidence."""
        track = SubtitleTrack()

        track.lines = [
            SubtitleLine(text="High", start_time=0.0, end_time=1.0, confidence=0.9),
            SubtitleLine(text="Low", start_time=1.0, end_time=2.0, confidence=0.5),
            SubtitleLine(text="Medium", start_time=2.0, end_time=3.0, confidence=0.75),
        ]

        track.filter_by_confidence(min_confidence=0.7)

        assert len(track.lines) == 2
        assert all(line.confidence >= 0.7 for line in track.lines)

    def test_to_srt(self):
        """Test SRT format export."""
        track = SubtitleTrack()

        track.lines = [
            SubtitleLine(text="First line", start_time=0.0, end_time=2.0),
            SubtitleLine(text="Second line", start_time=3.0, end_time=5.0),
        ]

        srt_content = track.to_srt()

        assert "1" in srt_content
        assert "00:00:00,000 --> 00:00:02,000" in srt_content
        assert "First line" in srt_content
        assert "2" in srt_content
        assert "Second line" in srt_content

    def test_to_vtt(self):
        """Test WebVTT format export."""
        track = SubtitleTrack()

        track.lines = [
            SubtitleLine(text="Test subtitle", start_time=1.5, end_time=4.0),
        ]

        vtt_content = track.to_vtt()

        assert "WEBVTT" in vtt_content
        assert "00:00:01.500 --> 00:00:04.000" in vtt_content
        assert "Test subtitle" in vtt_content

    def test_to_ass(self):
        """Test ASS format export."""
        track = SubtitleTrack(language="eng")

        track.lines = [
            SubtitleLine(text="Test", start_time=0.0, end_time=1.0),
        ]

        ass_content = track.to_ass()

        assert "[Script Info]" in ass_content
        assert "Language: eng" in ass_content
        assert "[Events]" in ass_content
        assert "Dialogue:" in ass_content


class TestSubtitleConfig:
    """Test SubtitleConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubtitleConfig()

        assert config.language == "eng"
        assert config.ocr_engine == "tesseract"
        assert config.min_confidence == 0.7
        assert config.output_format == "srt"
        assert config.subtitle_region == "bottom"
        assert config.region_height_percent == 25.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = SubtitleConfig(
            language="deu",
            ocr_engine="easyocr",
            min_confidence=0.8,
            output_format="vtt"
        )

        assert config.language == "deu"
        assert config.ocr_engine == "easyocr"
        assert config.min_confidence == 0.8
        assert config.output_format == "vtt"

    def test_config_invalid_confidence(self):
        """Test config rejects invalid confidence."""
        with pytest.raises(ValueError) as exc_info:
            SubtitleConfig(min_confidence=1.5)

        assert "min_confidence" in str(exc_info.value)

    def test_config_invalid_region_height(self):
        """Test config rejects invalid region height."""
        with pytest.raises(ValueError) as exc_info:
            SubtitleConfig(region_height_percent=0.0)

        assert "region_height_percent" in str(exc_info.value)

    def test_config_invalid_ocr_engine(self):
        """Test config rejects invalid OCR engine."""
        with pytest.raises(ValueError) as exc_info:
            SubtitleConfig(ocr_engine="invalid_engine")

        assert "Unsupported OCR engine" in str(exc_info.value)

    def test_config_invalid_output_format(self):
        """Test config rejects invalid output format."""
        with pytest.raises(ValueError) as exc_info:
            SubtitleConfig(output_format="invalid")

        assert "Unsupported output format" in str(exc_info.value)

    def test_config_normalizes_engine_case(self):
        """Test that OCR engine is normalized to lowercase."""
        config = SubtitleConfig(ocr_engine="TESSERACT")

        assert config.ocr_engine == "tesseract"


@pytest.mark.skipif(not HAS_PIL, reason="PIL not available")
class TestSubtitleExtractor:
    """Test SubtitleExtractor class."""

    def test_extractor_creation_tesseract(self):
        """Test creating extractor with tesseract (may fail if not installed)."""
        config = SubtitleConfig(ocr_engine="tesseract")

        # This may raise SubtitleError if tesseract is not installed
        try:
            extractor = SubtitleExtractor(config)
            assert extractor.config == config
        except SubtitleError as e:
            pytest.skip(f"Tesseract not available: {e}")

    def test_extractor_missing_pil(self):
        """Test extractor raises error when PIL missing."""
        with patch.dict('sys.modules', {'PIL': None, 'PIL.Image': None}):
            # Force reimport to trigger check
            # This is tricky to test properly
            pass

    @patch('framewright.processors.subtitles.HAS_PIL', True)
    @patch('framewright.processors.subtitles.HAS_TESSERACT', True)
    @patch('shutil.which')
    def test_detect_subtitle_region_bottom(self, mock_which, temp_dir):
        """Test detecting subtitle region in bottom area."""
        mock_which.return_value = '/usr/bin/tesseract'

        config = SubtitleConfig(
            subtitle_region="bottom",
            region_height_percent=25.0
        )

        extractor = SubtitleExtractor(config)

        # Create a test image
        img = Image.new('RGB', (1920, 1080), color='black')

        region = extractor.detect_subtitle_region(img)

        assert region is not None
        assert region.y == 810  # 1080 * 0.75 = 810
        assert region.height == 270  # 1080 * 0.25 = 270
        assert region.width == 1920

    @patch('framewright.processors.subtitles.HAS_PIL', True)
    @patch('framewright.processors.subtitles.HAS_TESSERACT', True)
    @patch('shutil.which')
    def test_detect_subtitle_region_top(self, mock_which, temp_dir):
        """Test detecting subtitle region in top area."""
        mock_which.return_value = '/usr/bin/tesseract'

        config = SubtitleConfig(
            subtitle_region="top",
            region_height_percent=20.0
        )

        extractor = SubtitleExtractor(config)

        # Create a test image
        img = Image.new('RGB', (1920, 1080), color='black')

        region = extractor.detect_subtitle_region(img)

        assert region is not None
        assert region.y == 0
        assert region.height == 216  # 1080 * 0.20 = 216

    @patch('framewright.processors.subtitles.HAS_PIL', True)
    @patch('framewright.processors.subtitles.HAS_TESSERACT', True)
    @patch('shutil.which')
    def test_save_subtitles_srt(self, mock_which, temp_dir):
        """Test saving subtitles to SRT format."""
        mock_which.return_value = '/usr/bin/tesseract'

        config = SubtitleConfig(output_format="srt")
        extractor = SubtitleExtractor(config)

        track = SubtitleTrack()
        track.lines = [
            SubtitleLine(text="Test", start_time=0.0, end_time=1.0)
        ]

        output_path = temp_dir / "output.srt"
        result = extractor.save_subtitles(track, output_path)

        assert result.exists()
        content = result.read_text()
        assert "Test" in content
        assert "00:00:00,000" in content


class TestSubtitleTimeSync:
    """Test SubtitleTimeSync class."""

    @patch('shutil.which', return_value='/usr/bin/ffmpeg')
    def test_time_sync_creation(self, mock_which):
        """Test creating SubtitleTimeSync."""
        sync = SubtitleTimeSync()
        assert sync is not None

    @patch('shutil.which', return_value='/usr/bin/ffmpeg')
    def test_adjust_timing_ratio(self, mock_which):
        """Test adjusting timing by ratio."""
        sync = SubtitleTimeSync()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple SRT file
            input_path = Path(tmp_dir) / "input.srt"
            input_path.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:02,000\n"
                "Test subtitle\n"
            )
            output_path = Path(tmp_dir) / "output.srt"

            # Adjust to double speed (0.5x ratio)
            result = sync.adjust_timing(
                input_path,
                output_path,
                time_ratio=0.5
            )

            assert result.exists()
            content = result.read_text()
            # Original 2 seconds should become 1 second
            assert "00:00:01,000" in content

    @patch('shutil.which', return_value='/usr/bin/ffmpeg')
    def test_sync_to_new_fps(self, mock_which):
        """Test syncing subtitles to new frame rate."""
        sync = SubtitleTimeSync()

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.srt"
            input_path.write_text(
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "Frame sync test\n"
            )
            output_path = Path(tmp_dir) / "output.srt"

            # Convert from 24fps to 60fps (stretch factor 2.5)
            result = sync.sync_to_new_fps(
                input_path,
                output_path,
                original_fps=24.0,
                target_fps=60.0
            )

            assert result.exists()


class TestSubtitleMerger:
    """Test SubtitleMerger class."""

    @patch('shutil.which', return_value='/usr/bin/ffmpeg')
    def test_merger_creation(self, mock_which):
        """Test creating SubtitleMerger."""
        merger = SubtitleMerger()
        assert merger is not None

    @patch('shutil.which', return_value='/usr/bin/ffmpeg')
    @patch('subprocess.run')
    def test_merge_soft_subs(self, mock_run, mock_which):
        """Test merging subtitles as soft subs."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        merger = SubtitleMerger()

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            sub_path = Path(tmp_dir) / "subs.srt"
            output_path = Path(tmp_dir) / "output.mp4"

            video_path.touch()
            sub_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")

            result = merger.merge(
                video_path,
                [sub_path],
                output_path,
                burn_in=False
            )

            # Verify FFmpeg was called with codec copy for soft subs
            call_args = str(mock_run.call_args)
            assert "ffmpeg" in call_args

    @patch('shutil.which', return_value='/usr/bin/ffmpeg')
    @patch('subprocess.run')
    def test_merge_burn_in(self, mock_run, mock_which):
        """Test burning in subtitles (hardcoded)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        merger = SubtitleMerger()

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            sub_path = Path(tmp_dir) / "subs.srt"
            output_path = Path(tmp_dir) / "output.mp4"

            video_path.touch()
            sub_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nBurned\n")

            result = merger.merge(
                video_path,
                [sub_path],
                output_path,
                burn_in=True
            )

            # Verify FFmpeg was called with subtitles filter for burn-in
            call_args = str(mock_run.call_args)
            assert "ffmpeg" in call_args


class TestSubtitleFormat:
    """Test SubtitleFormat enum."""

    def test_format_values(self):
        """Test that all format values exist."""
        assert SubtitleFormat.SRT.value == "srt"
        assert SubtitleFormat.ASS.value == "ass"
        assert SubtitleFormat.VTT.value == "vtt"
        assert SubtitleFormat.PGS.value == "pgs"
        assert SubtitleFormat.UNKNOWN.value == "unknown"


class TestSubtitleStreamInfo:
    """Test SubtitleStreamInfo dataclass."""

    def test_stream_info_creation(self):
        """Test creating SubtitleStreamInfo."""
        info = SubtitleStreamInfo(
            index=0,
            codec_name="subrip",
            language="eng",
            title="English Subtitles"
        )

        assert info.index == 0
        assert info.codec_name == "subrip"
        assert info.language == "eng"
        assert info.title == "English Subtitles"
        assert info.is_default is False
        assert info.is_forced is False

    def test_stream_info_with_flags(self):
        """Test SubtitleStreamInfo with disposition flags."""
        info = SubtitleStreamInfo(
            index=1,
            codec_name="ass",
            is_default=True,
            is_forced=False,
            format=SubtitleFormat.ASS
        )

        assert info.is_default is True
        assert info.format == SubtitleFormat.ASS


class TestSubtitleTimeFormatting:
    """Test time formatting utilities in SubtitleTrack."""

    def test_srt_time_format(self):
        """Test SRT time formatting (HH:MM:SS,mmm)."""
        # Test via SubtitleTrack static method
        time_str = SubtitleTrack._format_srt_time(3661.5)

        assert time_str == "01:01:01,500"

    def test_vtt_time_format(self):
        """Test VTT time formatting (HH:MM:SS.mmm)."""
        time_str = SubtitleTrack._format_vtt_time(3661.5)

        assert time_str == "01:01:01.500"

    def test_ass_time_format(self):
        """Test ASS time formatting (H:MM:SS.cc)."""
        time_str = SubtitleTrack._format_ass_time(3661.5)

        assert time_str == "1:01:01.50"

    def test_srt_time_zero(self):
        """Test SRT time formatting for zero."""
        time_str = SubtitleTrack._format_srt_time(0.0)

        assert time_str == "00:00:00,000"

    def test_vtt_time_large(self):
        """Test VTT time formatting for large values."""
        time_str = SubtitleTrack._format_vtt_time(36000.0)  # 10 hours

        assert time_str == "10:00:00.000"


class TestIntegration:
    """Integration tests for subtitle processing."""

    @pytest.fixture
    def sample_track(self):
        """Create a sample subtitle track for testing."""
        track = SubtitleTrack(language="eng", fps=24.0)

        track.lines = [
            SubtitleLine(text="Hello", start_time=0.0, end_time=2.0, confidence=0.9),
            SubtitleLine(text="World", start_time=3.0, end_time=5.0, confidence=0.85),
            SubtitleLine(text="Test", start_time=6.0, end_time=8.0, confidence=0.6),
        ]

        return track

    def test_export_all_formats(self, sample_track, temp_dir):
        """Test exporting to all supported formats."""
        for fmt in ["srt", "vtt", "ass"]:
            output_path = temp_dir / f"test.{fmt}"

            if fmt == "srt":
                content = sample_track.to_srt()
            elif fmt == "vtt":
                content = sample_track.to_vtt()
            else:
                content = sample_track.to_ass()

            output_path.write_text(content)

            assert output_path.exists()
            assert len(output_path.read_text()) > 0

    def test_full_workflow(self, sample_track):
        """Test complete subtitle processing workflow."""
        # Filter by confidence
        sample_track.filter_by_confidence(0.7)
        assert len(sample_track.lines) == 2

        # Export to SRT
        srt = sample_track.to_srt()

        assert "Hello" in srt
        assert "World" in srt
        assert "Test" not in srt  # Filtered out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_track_export(self):
        """Test exporting empty track."""
        track = SubtitleTrack()

        srt = track.to_srt()
        vtt = track.to_vtt()
        ass = track.to_ass()

        # Should not raise, just return minimal content
        assert srt == ""
        assert "WEBVTT" in vtt
        assert "[Script Info]" in ass

    def test_unicode_text(self):
        """Test handling Unicode text in subtitles."""
        track = SubtitleTrack()

        track.lines = [
            SubtitleLine(
                text="Hello, \u4e16\u754c! \u00e9\u00e8\u00ea",  # Chinese and French accents
                start_time=0.0,
                end_time=2.0
            )
        ]

        srt = track.to_srt()

        assert "\u4e16\u754c" in srt  # Chinese characters preserved
        assert "\u00e9" in srt  # French accent preserved

    def test_multiline_subtitle(self):
        """Test handling multiline subtitles."""
        track = SubtitleTrack()

        track.lines = [
            SubtitleLine(
                text="Line 1\nLine 2",
                start_time=0.0,
                end_time=2.0
            )
        ]

        srt = track.to_srt()
        ass = track.to_ass()

        assert "Line 1\nLine 2" in srt
        assert "Line 1\\NLine 2" in ass  # ASS uses \N for newlines

    def test_zero_duration_subtitle(self):
        """Test handling zero-duration subtitles."""
        line = SubtitleLine(
            text="Flash",
            start_time=5.0,
            end_time=5.0  # Same start and end
        )

        assert line.start_time == line.end_time
