"""Tests for subtitle extraction and processing.

Tests cover the SubtitleExtractor, SubtitleTrack, and related utilities
for handling embedded and burned-in subtitles in video files.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

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
    SubtitleEnhancer,
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
        assert box.x2 == 110

    def test_y2_property(self):
        """Test y2 (bottom coordinate) calculation."""
        box = BoundingBox(x=10, y=20, width=100, height=50)
        assert box.y2 == 70

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

    def test_scale(self):
        """Test scaling a bounding box."""
        box = BoundingBox(x=10, y=20, width=100, height=50)
        scaled = box.scale(2.0)
        assert scaled.x == 20
        assert scaled.y == 40
        assert scaled.width == 200
        assert scaled.height == 100
        assert scaled.confidence == box.confidence


class TestSubtitleLine:
    """Test SubtitleLine dataclass."""

    def test_subtitle_line_creation(self):
        """Test creating a SubtitleLine."""
        line = SubtitleLine(index=1, text="Hello, World!", start_time=1.0, end_time=3.5)
        assert line.text == "Hello, World!"
        assert line.start_time == 1.0
        assert line.end_time == 3.5
        assert line.index == 1
        assert line.position is None
        assert line.style is None
        assert line.layer == 0

    def test_subtitle_line_with_all_fields(self):
        """Test SubtitleLine with all fields."""
        position = BoundingBox(x=100, y=400, width=600, height=50)
        line = SubtitleLine(
            index=10, text="Test subtitle", start_time=10.0, end_time=15.0,
            position=position, style="Default", layer=1,
        )
        assert line.position == position
        assert line.style == "Default"
        assert line.layer == 1
        assert line.index == 10

    def test_subtitle_line_duration(self):
        """Test duration property of SubtitleLine."""
        line = SubtitleLine(index=1, text="Test", start_time=3661.5, end_time=3665.0)
        assert line.duration == pytest.approx(3.5)

    def test_subtitle_line_adjust_timing(self):
        """Test adjust_timing returns a new line with scaled times."""
        line = SubtitleLine(index=1, text="Hello", start_time=10.0, end_time=20.0)
        adjusted = line.adjust_timing(time_factor=2.0, offset=1.0)
        assert adjusted.start_time == pytest.approx(21.0)
        assert adjusted.end_time == pytest.approx(41.0)
        assert adjusted.text == "Hello"

    def test_subtitle_line_to_srt(self):
        """Test SRT export of a single line."""
        line = SubtitleLine(index=1, text="Test", start_time=0.0, end_time=2.0)
        srt = line.to_srt()
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "Test" in srt

    def test_subtitle_line_to_vtt(self):
        """Test VTT export of a single line."""
        line = SubtitleLine(index=1, text="Test", start_time=1.5, end_time=4.0)
        vtt = line.to_vtt()
        assert "00:00:01.500 --> 00:00:04.000" in vtt
        assert "Test" in vtt


class TestSubtitleTrack:
    """Test SubtitleTrack class."""

    def test_empty_track_creation(self):
        """Test creating an empty SubtitleTrack."""
        track = SubtitleTrack()
        assert len(track.lines) == 0
        assert track.language is None
        assert track.format == SubtitleFormat.SRT

    def test_track_with_custom_settings(self):
        """Test SubtitleTrack with custom settings."""
        track = SubtitleTrack(language="deu", metadata={"source": "test.mp4"})
        assert track.language == "deu"
        assert track.metadata["source"] == "test.mp4"

    def test_track_line_count(self):
        """Test line_count property."""
        track = SubtitleTrack()
        track.lines = [
            SubtitleLine(index=1, text="First", start_time=0.0, end_time=2.0),
            SubtitleLine(index=2, text="Second", start_time=3.0, end_time=5.0),
        ]
        assert track.line_count == 2

    def test_track_duration(self):
        """Test duration property."""
        track = SubtitleTrack()
        track.lines = [
            SubtitleLine(index=1, text="First", start_time=0.0, end_time=2.0),
            SubtitleLine(index=2, text="Second", start_time=3.0, end_time=5.0),
        ]
        assert track.duration == pytest.approx(5.0)

    def test_track_duration_empty(self):
        """Test duration on empty track."""
        track = SubtitleTrack()
        assert track.duration == 0.0

    def test_adjust_timing(self):
        """Test adjust_timing on a track."""
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Hello", start_time=10.0, end_time=20.0)]
        adjusted = track.adjust_timing(time_factor=0.5, offset=1.0)
        assert adjusted.lines[0].start_time == pytest.approx(6.0)
        assert adjusted.lines[0].end_time == pytest.approx(11.0)

    def test_to_srt(self):
        """Test SRT format export."""
        track = SubtitleTrack()
        track.lines = [
            SubtitleLine(index=1, text="First line", start_time=0.0, end_time=2.0),
            SubtitleLine(index=2, text="Second line", start_time=3.0, end_time=5.0),
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
        track.lines = [SubtitleLine(index=1, text="Test subtitle", start_time=1.5, end_time=4.0)]
        vtt_content = track.to_vtt()
        assert "WEBVTT" in vtt_content
        assert "00:00:01.500 --> 00:00:04.000" in vtt_content
        assert "Test subtitle" in vtt_content

    def test_to_ass(self):
        """Test ASS format export."""
        track = SubtitleTrack(language="eng")
        track.lines = [SubtitleLine(index=1, text="Test", start_time=0.0, end_time=1.0)]
        ass_content = track.to_ass()
        assert "[Script Info]" in ass_content
        assert "[Events]" in ass_content
        assert "Dialogue:" in ass_content

    def test_save_srt(self):
        """Test saving track to SRT file."""
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Test", start_time=0.0, end_time=1.0)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.srt"
            result = track.save(output_path, SubtitleFormat.SRT)
            assert result.exists()
            content = result.read_text()
            assert "Test" in content
            assert "00:00:00,000" in content


class TestSubtitleConfig:
    """Test SubtitleConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubtitleConfig()
        assert config.preserve_subtitles is True
        assert config.adjust_timing is True
        assert config.preferred_format == SubtitleFormat.SRT
        assert config.burn_in is False
        assert config.default_language == "und"
        assert config.extract_all is True
        assert config.font_scale == 1.0
        assert config.position_adjust is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SubtitleConfig(
            default_language="deu", preferred_format=SubtitleFormat.VTT,
            burn_in=True, font_scale=1.5,
        )
        assert config.default_language == "deu"
        assert config.preferred_format == SubtitleFormat.VTT
        assert config.burn_in is True
        assert config.font_scale == 1.5

    def test_config_format_string_normalization(self):
        """Test that preferred_format string is normalized to enum."""
        config = SubtitleConfig(preferred_format="srt")
        assert config.preferred_format == SubtitleFormat.SRT

    def test_config_format_string_vtt(self):
        """Test that vtt string is converted to enum."""
        config = SubtitleConfig(preferred_format="vtt")
        assert config.preferred_format == SubtitleFormat.VTT

    def test_config_format_string_ass(self):
        """Test that ass string is converted to enum."""
        config = SubtitleConfig(preferred_format="ass")
        assert config.preferred_format == SubtitleFormat.ASS

    def test_config_preserve_false(self):
        """Test config with preserve_subtitles disabled."""
        config = SubtitleConfig(preserve_subtitles=False)
        assert config.preserve_subtitles is False

    def test_config_extract_all_false(self):
        """Test config with extract_all disabled."""
        config = SubtitleConfig(extract_all=False)
        assert config.extract_all is False


class TestSubtitleExtractor:
    """Test SubtitleExtractor class."""

    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    def test_extractor_creation(self, mock_which):
        """Test creating extractor with default config."""
        extractor = SubtitleExtractor()
        assert extractor.config.preserve_subtitles is True

    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    def test_extractor_with_config(self, mock_which):
        """Test creating extractor with custom config."""
        config = SubtitleConfig(preferred_format=SubtitleFormat.VTT)
        extractor = SubtitleExtractor(config=config)
        assert extractor.config.preferred_format == SubtitleFormat.VTT

    @patch("shutil.which", return_value=None)
    def test_extractor_missing_ffmpeg(self, mock_which):
        """Test extractor raises error when ffmpeg missing."""
        with pytest.raises(SubtitleError):
            SubtitleExtractor()

    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    def test_detect_streams_missing_file(self, mock_which):
        """Test detect_streams raises error for missing file."""
        extractor = SubtitleExtractor()
        with pytest.raises(SubtitleError, match="not found"):
            extractor.detect_streams(Path("/nonexistent/video.mkv"))

    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    @patch("subprocess.run")
    def test_detect_streams_empty(self, mock_run, mock_which):
        """Test detect_streams with no subtitle streams."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"streams": []}', stderr="")
        extractor = SubtitleExtractor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mkv"
            video_path.touch()
            streams = extractor.detect_streams(video_path)
            assert streams == []

    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    @patch("subprocess.run")
    def test_detect_streams_found(self, mock_run, mock_which):
        """Test detect_streams finds subtitle streams."""
        import json
        stream_data = json.dumps({"streams": [{"index": 2, "codec_name": "subrip", "codec_type": "subtitle", "tags": {"language": "eng"}, "disposition": {"default": 1, "forced": 0}}]})
        mock_run.return_value = MagicMock(returncode=0, stdout=stream_data, stderr="")
        extractor = SubtitleExtractor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mkv"
            video_path.touch()
            streams = extractor.detect_streams(video_path)
            assert len(streams) == 1
            assert streams[0].index == 2
            assert streams[0].codec_name == "subrip"
            assert streams[0].language == "eng"
            assert streams[0].is_default is True


class TestSubtitleTimeSync:
    """Test SubtitleTimeSync class."""

    def test_time_sync_creation(self):
        """Test creating SubtitleTimeSync."""
        sync = SubtitleTimeSync()
        assert sync is not None
        assert sync.config.preserve_subtitles is True

    def test_time_sync_with_config(self):
        """Test creating SubtitleTimeSync with config."""
        config = SubtitleConfig(adjust_timing=False)
        sync = SubtitleTimeSync(config=config)
        assert sync.config.adjust_timing is False

    def test_adjust_for_framerate_change(self):
        """Test adjusting timing for frame rate change."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Test", start_time=0.0, end_time=1.0)]
        adjusted = sync.adjust_for_framerate_change(track, source_fps=24.0, target_fps=60.0)
        assert adjusted.lines[0].start_time == pytest.approx(0.0)
        assert adjusted.lines[0].end_time == pytest.approx(1.0)

    def test_adjust_for_duration_change(self):
        """Test adjusting timing for duration change."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Test", start_time=0.0, end_time=2.0)]
        adjusted = sync.adjust_for_duration_change(track, source_duration=10.0, target_duration=5.0)
        assert adjusted.lines[0].start_time == pytest.approx(0.0)
        assert adjusted.lines[0].end_time == pytest.approx(1.0)

    def test_adjust_for_speed_change(self):
        """Test adjusting timing for speed change."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Test", start_time=0.0, end_time=2.0)]
        adjusted = sync.adjust_for_speed_change(track, speed_factor=2.0)
        assert adjusted.lines[0].end_time == pytest.approx(1.0)

    def test_apply_offset(self):
        """Test applying a fixed offset to subtitles."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Test", start_time=5.0, end_time=10.0)]
        adjusted = sync.apply_offset(track, offset_seconds=2.0)
        assert adjusted.lines[0].start_time == pytest.approx(7.0)
        assert adjusted.lines[0].end_time == pytest.approx(12.0)

    def test_correct_drift(self):
        """Test correcting progressive timing drift."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Early", start_time=60.0, end_time=62.0)]
        adjusted = sync.correct_drift(track, drift_per_minute=1.0)
        assert adjusted.lines[0].start_time == pytest.approx(59.0)

    def test_adjust_for_duration_change_invalid(self):
        """Test that invalid durations raise errors."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        with pytest.raises(SubtitleError):
            sync.adjust_for_duration_change(track, source_duration=0.0, target_duration=5.0)

    def test_adjust_for_speed_change_invalid(self):
        """Test that invalid speed factor raises error."""
        sync = SubtitleTimeSync()
        track = SubtitleTrack()
        with pytest.raises(SubtitleError):
            sync.adjust_for_speed_change(track, speed_factor=-1.0)


class TestSubtitleMerger:
    """Test SubtitleMerger class."""

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_merger_creation(self, mock_which):
        """Test creating SubtitleMerger."""
        merger = SubtitleMerger()
        assert merger is not None

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_merge_soft_subs(self, mock_run, mock_which):
        """Test merging subtitles as soft subs."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        merger = SubtitleMerger()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            output_path = Path(tmp_dir) / "output.mp4"
            video_path.touch()
            track = SubtitleTrack(language="eng")
            track.lines = [SubtitleLine(index=1, text="Test", start_time=0.0, end_time=1.0)]
            result = merger.merge_soft(video_path, [track], output_path)
            assert mock_run.called
            call_args = str(mock_run.call_args)
            assert "ffmpeg" in call_args

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_merge_hard_burn(self, mock_run, mock_which):
        """Test burning in subtitles (hardcoded)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        merger = SubtitleMerger()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            output_path = Path(tmp_dir) / "output.mp4"
            video_path.touch()
            track = SubtitleTrack()
            track.lines = [SubtitleLine(index=1, text="Burned", start_time=0.0, end_time=1.0)]
            result = merger.merge_hard(video_path, track, output_path)
            assert mock_run.called
            call_args = str(mock_run.call_args)
            assert "ffmpeg" in call_args

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_merge_soft_empty_tracks(self, mock_which):
        """Test merge_soft with no tracks copies the video."""
        merger = SubtitleMerger()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            output_path = Path(tmp_dir) / "output.mp4"
            video_path.write_bytes(b"fake video data")
            result = merger.merge_soft(video_path, [], output_path)
            assert result == output_path
            assert output_path.exists()


class TestSubtitleFormat:
    """Test SubtitleFormat enum."""

    def test_format_values(self):
        """Test that all format values exist."""
        assert SubtitleFormat.SRT.value == "srt"
        assert SubtitleFormat.ASS.value == "ass"
        assert SubtitleFormat.VTT.value == "vtt"
        assert SubtitleFormat.PGS.value == "pgs"
        assert SubtitleFormat.UNKNOWN.value == "unknown"

    def test_format_ssa(self):
        """Test SSA format value."""
        assert SubtitleFormat.SSA.value == "ssa"

    def test_format_dvb(self):
        """Test DVB format value."""
        assert SubtitleFormat.DVB.value == "dvb"

    def test_format_sub(self):
        """Test SUB format value."""
        assert SubtitleFormat.SUB.value == "sub"


class TestSubtitleStreamInfo:
    """Test SubtitleStreamInfo dataclass."""

    def test_stream_info_creation(self):
        """Test creating SubtitleStreamInfo."""
        info = SubtitleStreamInfo(index=0, codec_name="subrip", language="eng", title="English Subtitles")
        assert info.index == 0
        assert info.codec_name == "subrip"
        assert info.language == "eng"
        assert info.title == "English Subtitles"
        assert info.is_default is False
        assert info.is_forced is False

    def test_stream_info_with_flags(self):
        """Test SubtitleStreamInfo with disposition flags."""
        info = SubtitleStreamInfo(index=1, codec_name="ass", is_default=True, is_forced=False, format=SubtitleFormat.ASS)
        assert info.is_default is True
        assert info.format == SubtitleFormat.ASS

    def test_stream_info_auto_format_detection(self):
        """Test automatic format detection from codec name."""
        info = SubtitleStreamInfo(index=0, codec_name="subrip")
        assert info.format == SubtitleFormat.SRT
        info2 = SubtitleStreamInfo(index=1, codec_name="webvtt")
        assert info2.format == SubtitleFormat.VTT
        info3 = SubtitleStreamInfo(index=2, codec_name="hdmv_pgs_subtitle")
        assert info3.format == SubtitleFormat.PGS


class TestSubtitleTimeFormatting:
    """Test time formatting utilities via SubtitleLine methods."""

    def test_srt_time_format(self):
        """Test SRT time formatting (HH:MM:SS,mmm)."""
        line = SubtitleLine(index=1, text="", start_time=0.0, end_time=0.0)
        assert line.to_srt_time(3661.5) == "01:01:01,500"

    def test_vtt_time_format(self):
        """Test VTT time formatting (HH:MM:SS.mmm)."""
        line = SubtitleLine(index=1, text="", start_time=0.0, end_time=0.0)
        assert line.to_vtt_time(3661.5) == "01:01:01.500"

    def test_ass_time_format(self):
        """Test ASS time formatting (H:MM:SS.cc)."""
        line = SubtitleLine(index=1, text="", start_time=0.0, end_time=0.0)
        assert line.to_ass_time(3661.5) == "1:01:01.50"

    def test_srt_time_zero(self):
        """Test SRT time formatting for zero."""
        line = SubtitleLine(index=1, text="", start_time=0.0, end_time=0.0)
        assert line.to_srt_time(0.0) == "00:00:00,000"

    def test_vtt_time_large(self):
        """Test VTT time formatting for large values."""
        line = SubtitleLine(index=1, text="", start_time=0.0, end_time=0.0)
        assert line.to_vtt_time(36000.0) == "10:00:00.000"


class TestSubtitleEnhancer:
    """Test SubtitleEnhancer class."""

    def test_enhancer_creation(self):
        """Test creating SubtitleEnhancer."""
        enhancer = SubtitleEnhancer()
        assert enhancer.config.preserve_subtitles is True

    def test_clean_ocr_artifacts(self):
        """Test cleaning OCR artifacts."""
        enhancer = SubtitleEnhancer()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Hello ,  World!  ", start_time=0.0, end_time=2.0)]
        cleaned = enhancer.clean_ocr_artifacts(track)
        assert cleaned.lines[0].text == "Hello, World!"

    def test_standardize_formatting(self):
        """Test standardizing subtitle formatting."""
        enhancer = SubtitleEnhancer()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="hello world", start_time=0.0, end_time=2.0)]
        standardized = enhancer.standardize_formatting(track)
        assert standardized.lines[0].text[0] == "H"

    def test_adjust_positions_for_scale(self):
        """Test adjusting positions for upscaling."""
        enhancer = SubtitleEnhancer()
        track = SubtitleTrack()
        track.lines = [SubtitleLine(
            index=1, text="Test", start_time=0.0, end_time=1.0,
            position=BoundingBox(x=10, y=20, width=100, height=50),
        )]
        scaled = enhancer.adjust_positions_for_scale(track, scale_factor=2.0)
        assert scaled.lines[0].position.x == 20
        assert scaled.lines[0].position.y == 40
        assert scaled.lines[0].position.width == 200
        assert scaled.lines[0].position.height == 100

class TestIntegration:
    """Integration tests for subtitle processing."""

    @pytest.fixture
    def sample_track(self):
        """Create a sample subtitle track for testing."""
        track = SubtitleTrack(language="eng")
        track.lines = [
            SubtitleLine(index=1, text="Hello", start_time=0.0, end_time=2.0),
            SubtitleLine(index=2, text="World", start_time=3.0, end_time=5.0),
            SubtitleLine(index=3, text="Test", start_time=6.0, end_time=8.0),
        ]
        return track

    def test_export_all_formats(self, sample_track):
        """Test exporting to all supported formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for fmt in ["srt", "vtt", "ass"]:
                output_path = tmp_path / f"test.{fmt}"
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
        sync = SubtitleTimeSync()
        adjusted = sync.adjust_for_speed_change(sample_track, speed_factor=2.0)
        assert adjusted.lines[0].end_time == pytest.approx(1.0)
        srt = adjusted.to_srt()
        assert "Hello" in srt
        assert "World" in srt
        assert "Test" in srt

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_track_export(self):
        """Test exporting empty track."""
        track = SubtitleTrack()
        srt = track.to_srt()
        vtt = track.to_vtt()
        ass = track.to_ass()
        assert srt == ""
        assert "WEBVTT" in vtt
        assert "[Script Info]" in ass

    def test_unicode_text(self):
        """Test handling Unicode text in subtitles."""
        track = SubtitleTrack()
        track.lines = [SubtitleLine(
            index=1, text="Hello, 世界! éèê",
            start_time=0.0, end_time=2.0,
        )]
        srt = track.to_srt()
        assert "世界" in srt
        assert "é" in srt

    def test_multiline_subtitle(self):
        """Test handling multiline subtitles."""
        track = SubtitleTrack()
        track.lines = [SubtitleLine(index=1, text="Line 1\nLine 2", start_time=0.0, end_time=2.0)]
        srt = track.to_srt()
        ass = track.to_ass()
        assert "Line 1\nLine 2" in srt
        assert "Line 1\\NLine 2" in ass

    def test_zero_duration_subtitle(self):
        """Test handling zero-duration subtitles."""
        line = SubtitleLine(index=1, text="Flash", start_time=5.0, end_time=5.0)
        assert line.start_time == line.end_time
        assert line.duration == 0.0

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_burned_subtitles(self):
        """Test detect_burned_subtitles returns expected structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            video_path.touch()
            result = detect_burned_subtitles(video_path)
            assert isinstance(result, dict)
            assert "has_subtitles" in result
            assert result["has_subtitles"] is False
    @patch("shutil.which", return_value="/usr/bin/ffprobe")
    @patch("subprocess.run")
    def test_extract_subtitles_empty(self, mock_run, mock_which):
        """Test extract_subtitles with no streams found."""
        import json as _json
        mock_run.return_value = MagicMock(returncode=0, stdout=_json.dumps({"streams": []}), stderr="")
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            video_path.touch()
            tracks = extract_subtitles(video_path)
            assert tracks == []
    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_remove_subtitles(self, mock_run, mock_which):
        """Test remove_subtitles."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as d:
            vp = Path(d) / "v.mp4"
            op = Path(d) / "o.mp4"
            vp.touch()
            remove_subtitles(vp, op)
            assert mock_run.called
