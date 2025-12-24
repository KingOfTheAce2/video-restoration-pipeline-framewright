"""Tests for the validators module."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from framewright.validators import (
    validate_frame_integrity,
    validate_frame_sequence,
    validate_frame_batch,
    compute_quality_metrics,
    validate_enhancement_quality,
    detect_artifacts,
    validate_temporal_consistency,
    validate_audio_stream,
    validate_av_sync,
    detect_audio_issues,
    analyze_audio_quality,
    QualityMetrics,
    FrameValidation,
    SequenceReport,
    ArtifactReport,
    TemporalReport,
    AudioValidation,
    AudioIssue,
    AudioQualityReport,
)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(psnr=35.0, ssim=0.95, vmaf=85.0)

        assert metrics.psnr == 35.0
        assert metrics.ssim == 0.95
        assert metrics.vmaf == 85.0

    def test_meets_threshold_pass(self):
        """Test metrics meeting threshold."""
        metrics = QualityMetrics(psnr=35.0, ssim=0.95)

        assert metrics.meets_threshold(min_psnr=25.0, min_ssim=0.85) is True

    def test_meets_threshold_fail_psnr(self):
        """Test metrics failing PSNR threshold."""
        metrics = QualityMetrics(psnr=20.0, ssim=0.95)

        assert metrics.meets_threshold(min_psnr=25.0, min_ssim=0.85) is False

    def test_meets_threshold_fail_ssim(self):
        """Test metrics failing SSIM threshold."""
        metrics = QualityMetrics(psnr=35.0, ssim=0.80)

        assert metrics.meets_threshold(min_psnr=25.0, min_ssim=0.85) is False


class TestFrameValidation:
    """Tests for FrameValidation dataclass."""

    def test_create_validation(self):
        """Test creating frame validation result."""
        validation = FrameValidation(
            frame_path=Path("/path/to/frame.png"),
            is_valid=True,
            width=1920,
            height=1080,
        )

        assert validation.is_valid is True
        assert validation.width == 1920

    def test_validation_with_error(self):
        """Test validation result with error."""
        validation = FrameValidation(
            frame_path=Path("/path/to/frame.png"),
            is_valid=False,
            error_message="File is corrupted",
        )

        assert validation.is_valid is False
        assert "corrupted" in validation.error_message


class TestSequenceReport:
    """Tests for SequenceReport dataclass."""

    def test_complete_sequence(self):
        """Test report for complete sequence."""
        report = SequenceReport(
            total_frames=100,
            expected_frames=100,
            is_complete=True,
        )

        assert report.is_complete is True
        assert report.missing_count == 0
        assert not report.has_issues

    def test_incomplete_sequence(self):
        """Test report for incomplete sequence."""
        report = SequenceReport(
            total_frames=95,
            expected_frames=100,
            missing_frames=[5, 10, 15, 20, 25],
            is_complete=False,
        )

        assert report.is_complete is False
        assert report.missing_count == 5
        assert report.has_issues


class TestValidateFrameIntegrity:
    """Tests for validate_frame_integrity function."""

    def test_nonexistent_file(self, tmp_path):
        """Test validation of nonexistent file."""
        fake_path = tmp_path / "nonexistent.png"

        result = validate_frame_integrity(fake_path)

        assert result.is_valid is False
        assert "does not exist" in result.error_message

    def test_empty_file(self, tmp_path):
        """Test validation of empty file."""
        empty_file = tmp_path / "empty.png"
        empty_file.touch()

        result = validate_frame_integrity(empty_file)

        assert result.is_valid is False
        assert "empty" in result.error_message

    @patch('subprocess.run')
    def test_valid_frame(self, mock_run, tmp_path):
        """Test validation of valid frame."""
        frame_file = tmp_path / "frame.png"
        frame_file.write_bytes(b"PNG data")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                }]
            })
        )

        result = validate_frame_integrity(frame_file)

        assert result.is_valid is True
        assert result.width == 1920
        assert result.height == 1080


class TestValidateFrameSequence:
    """Tests for validate_frame_sequence function."""

    def test_complete_sequence(self, tmp_path):
        """Test validation of complete frame sequence."""
        for i in range(1, 11):
            (tmp_path / f"frame_{i:08d}.png").write_bytes(b"PNG")

        report = validate_frame_sequence(tmp_path)

        assert report.total_frames == 10
        assert len(report.missing_frames) == 0

    def test_sequence_with_gaps(self, tmp_path):
        """Test validation of sequence with missing frames."""
        for i in [1, 2, 3, 5, 6, 8, 9, 10]:  # Missing 4 and 7
            (tmp_path / f"frame_{i:08d}.png").write_bytes(b"PNG")

        report = validate_frame_sequence(tmp_path)

        assert report.total_frames == 8
        assert 4 in report.missing_frames
        assert 7 in report.missing_frames

    def test_empty_directory(self, tmp_path):
        """Test validation of empty directory."""
        report = validate_frame_sequence(tmp_path)

        assert report.total_frames == 0
        assert report.is_complete is False


class TestArtifactReport:
    """Tests for ArtifactReport dataclass."""

    def test_no_artifacts(self):
        """Test report with no artifacts."""
        report = ArtifactReport(
            frame_path=Path("/path/to/frame.png"),
            has_artifacts=False,
        )

        assert report.has_artifacts is False
        assert report.severity == "none"

    def test_with_artifacts(self):
        """Test report with artifacts detected."""
        report = ArtifactReport(
            frame_path=Path("/path/to/frame.png"),
            has_artifacts=True,
            artifacts=["HALO_ARTIFACT", "COLOR_BANDING"],
            severity="medium",
        )

        assert report.has_artifacts is True
        assert len(report.artifacts) == 2
        assert report.severity == "medium"


class TestTemporalReport:
    """Tests for TemporalReport dataclass."""

    def test_stable_video(self):
        """Test report for stable video."""
        report = TemporalReport(
            frames_analyzed=100,
            brightness_variance=2.5,
            color_variance=1.0,
            flickering_detected=False,
        )

        assert report.flickering_detected is False
        assert report.severity == "none"

    def test_flickering_video(self):
        """Test report for video with flickering."""
        report = TemporalReport(
            frames_analyzed=100,
            brightness_variance=25.0,
            color_variance=15.0,
            flickering_detected=True,
            flicker_frames=[10, 25, 50, 75],
            severity="medium",
        )

        assert report.flickering_detected is True
        assert len(report.flicker_frames) == 4


class TestAudioValidation:
    """Tests for AudioValidation dataclass."""

    def test_valid_audio(self):
        """Test validation result for valid audio."""
        validation = AudioValidation(
            has_audio=True,
            codec="pcm_s24le",
            sample_rate=48000,
            channels=2,
            duration=120.5,
            is_valid=True,
        )

        assert validation.has_audio is True
        assert validation.sample_rate == 48000

    def test_no_audio(self):
        """Test validation result for file without audio."""
        validation = AudioValidation(
            has_audio=False,
            is_valid=True,
        )

        assert validation.has_audio is False


class TestValidateAudioStream:
    """Tests for validate_audio_stream function."""

    def test_nonexistent_file(self, tmp_path):
        """Test validation of nonexistent audio file."""
        fake_path = tmp_path / "nonexistent.wav"

        result = validate_audio_stream(fake_path)

        assert result.has_audio is False
        assert result.is_valid is False

    @patch('subprocess.run')
    def test_valid_audio_file(self, mock_run, tmp_path):
        """Test validation of valid audio file."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF data")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "codec_name": "pcm_s24le",
                    "sample_rate": "48000",
                    "channels": "2",
                    "duration": "120.5",
                }]
            })
        )

        result = validate_audio_stream(audio_file)

        assert result.has_audio is True
        assert result.codec == "pcm_s24le"
        assert result.sample_rate == 48000


class TestValidateAVSync:
    """Tests for validate_av_sync function."""

    @patch('subprocess.run')
    def test_synced_av(self, mock_run, tmp_path):
        """Test validation of synced audio/video."""
        video_file = tmp_path / "video.mp4"
        audio_file = tmp_path / "audio.wav"
        video_file.write_bytes(b"video")
        audio_file.write_bytes(b"audio")

        # First call for video, second for audio
        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "120.0"}})
            ),
            MagicMock(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "120.05"}})
            ),
        ]

        is_synced, diff_ms = validate_av_sync(video_file, audio_file)

        assert is_synced is True
        assert diff_ms < 100  # Within 100ms tolerance

    @patch('subprocess.run')
    def test_out_of_sync_av(self, mock_run, tmp_path):
        """Test validation of out-of-sync audio/video."""
        video_file = tmp_path / "video.mp4"
        audio_file = tmp_path / "audio.wav"
        video_file.write_bytes(b"video")
        audio_file.write_bytes(b"audio")

        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "120.0"}})
            ),
            MagicMock(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "121.0"}})  # 1 second diff
            ),
        ]

        is_synced, diff_ms = validate_av_sync(video_file, audio_file)

        assert is_synced is False
        assert diff_ms == 1000  # 1000ms difference


class TestAudioIssue:
    """Tests for AudioIssue dataclass."""

    def test_create_issue(self):
        """Test creating audio issue."""
        issue = AudioIssue(
            issue_type="SILENCE",
            start_time=10.0,
            end_time=15.0,
            severity="medium",
        )

        assert issue.issue_type == "SILENCE"
        assert issue.start_time == 10.0
        assert issue.end_time == 15.0
        assert issue.severity == "medium"

    def test_default_values(self):
        """Test default values."""
        issue = AudioIssue(issue_type="CLIPPING", start_time=0.0)

        assert issue.end_time is None
        assert issue.severity == "low"
        assert issue.details == {}


class TestAudioQualityReport:
    """Tests for AudioQualityReport dataclass."""

    def test_create_report(self, tmp_path):
        """Test creating audio quality report."""
        audio_path = tmp_path / "audio.wav"
        report = AudioQualityReport(
            audio_path=audio_path,
            max_volume_db=-3.0,
            mean_volume_db=-18.0,
            clipping_detected=False,
        )

        assert report.audio_path == audio_path
        assert report.max_volume_db == -3.0
        assert report.is_acceptable is True

    def test_has_issues_no_issues(self, tmp_path):
        """Test has_issues when no issues."""
        report = AudioQualityReport(
            audio_path=tmp_path / "audio.wav",
            clipping_detected=False,
        )

        assert report.has_issues is False

    def test_has_issues_with_clipping(self, tmp_path):
        """Test has_issues with clipping."""
        report = AudioQualityReport(
            audio_path=tmp_path / "audio.wav",
            clipping_detected=True,
        )

        assert report.has_issues is True

    def test_has_issues_with_issues_list(self, tmp_path):
        """Test has_issues with issues list."""
        report = AudioQualityReport(
            audio_path=tmp_path / "audio.wav",
            clipping_detected=False,
            issues=[AudioIssue(issue_type="SILENCE", start_time=0.0)],
        )

        assert report.has_issues is True


class TestDetectAudioIssues:
    """Tests for detect_audio_issues function."""

    def test_nonexistent_file(self, tmp_path):
        """Test with nonexistent file."""
        fake_path = tmp_path / "nonexistent.wav"

        issues = detect_audio_issues(fake_path)

        assert issues == []

    @patch('subprocess.run')
    def test_detect_silence(self, mock_run, tmp_path):
        """Test detecting silence segments."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        # Simulate ffmpeg output with silence detection
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr=(
                "[silencedetect @ 0x123] silence_start: 5.0\n"
                "[silencedetect @ 0x123] silence_end: 8.0 | silence_duration: 3.0\n"
                "[silencedetect @ 0x123] silence_start: 20.0\n"
                "[silencedetect @ 0x123] silence_end: 22.0 | silence_duration: 2.0\n"
            )
        )

        issues = detect_audio_issues(audio_file)

        assert len(issues) == 2
        assert issues[0].issue_type == "SILENCE"
        assert issues[0].start_time == 5.0
        assert issues[0].end_time == 8.0
        assert issues[1].start_time == 20.0

    @patch('subprocess.run')
    def test_no_silence_detected(self, mock_run, tmp_path):
        """Test when no silence is detected."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="Processing complete\n"
        )

        issues = detect_audio_issues(audio_file)

        assert len(issues) == 0

    @patch('subprocess.run')
    def test_severity_based_on_duration(self, mock_run, tmp_path):
        """Test severity is based on silence duration."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        mock_run.return_value = MagicMock(
            returncode=0,
            stderr=(
                "[silencedetect @ 0x123] silence_start: 0.0\n"
                "[silencedetect @ 0x123] silence_end: 2.0 | silence_duration: 2.0\n"  # low
                "[silencedetect @ 0x123] silence_start: 10.0\n"
                "[silencedetect @ 0x123] silence_end: 17.0 | silence_duration: 7.0\n"  # medium
                "[silencedetect @ 0x123] silence_start: 30.0\n"
                "[silencedetect @ 0x123] silence_end: 45.0 | silence_duration: 15.0\n"  # high
            )
        )

        issues = detect_audio_issues(audio_file)

        assert len(issues) == 3
        assert issues[0].severity == "low"
        assert issues[1].severity == "medium"
        assert issues[2].severity == "high"


class TestAnalyzeAudioQuality:
    """Tests for analyze_audio_quality function."""

    def test_nonexistent_file(self, tmp_path):
        """Test with nonexistent file."""
        fake_path = tmp_path / "nonexistent.wav"

        report = analyze_audio_quality(fake_path)

        assert report.is_acceptable is False

    @patch('framewright.validators.detect_audio_issues')
    @patch('subprocess.run')
    def test_analyze_normal_audio(self, mock_run, mock_detect, tmp_path):
        """Test analyzing normal audio."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        mock_detect.return_value = []

        # First call - volume detection
        # Second call - audio stats
        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stderr=(
                    "[Parsed_volumedetect_0] max_volume: -6.0 dB\n"
                    "[Parsed_volumedetect_0] mean_volume: -18.0 dB\n"
                )
            ),
            MagicMock(
                returncode=0,
                stderr="Audio stats output\n"
            ),
        ]

        report = analyze_audio_quality(audio_file)

        assert report.max_volume_db == -6.0
        assert report.mean_volume_db == -18.0
        assert report.clipping_detected is False
        assert report.is_acceptable is True

    @patch('framewright.validators.detect_audio_issues')
    @patch('subprocess.run')
    def test_detect_clipping(self, mock_run, mock_detect, tmp_path):
        """Test detecting audio clipping."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        mock_detect.return_value = []

        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stderr=(
                    "[Parsed_volumedetect_0] max_volume: 0.0 dB\n"
                    "[Parsed_volumedetect_0] mean_volume: -12.0 dB\n"
                )
            ),
            MagicMock(
                returncode=0,
                stderr="Audio stats\n"
            ),
        ]

        report = analyze_audio_quality(audio_file)

        assert report.clipping_detected is True
        assert report.has_issues is True

    @patch('framewright.validators.detect_audio_issues')
    @patch('subprocess.run')
    def test_detect_low_volume(self, mock_run, mock_detect, tmp_path):
        """Test detecting low volume audio."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        mock_detect.return_value = []

        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stderr=(
                    "[Parsed_volumedetect_0] max_volume: -30.0 dB\n"
                    "[Parsed_volumedetect_0] mean_volume: -50.0 dB\n"
                )
            ),
            MagicMock(
                returncode=0,
                stderr="Audio stats\n"
            ),
        ]

        report = analyze_audio_quality(audio_file)

        # Check for LOW_VOLUME issue
        low_vol_issues = [i for i in report.issues if i.issue_type == "LOW_VOLUME"]
        assert len(low_vol_issues) == 1

    @patch('framewright.validators.detect_audio_issues')
    @patch('subprocess.run')
    def test_includes_silence_segments(self, mock_run, mock_detect, tmp_path):
        """Test that silence segments are included."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio data")

        mock_detect.return_value = [
            AudioIssue(
                issue_type="SILENCE",
                start_time=5.0,
                end_time=10.0,
                severity="medium",
            )
        ]

        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stderr=(
                    "[Parsed_volumedetect_0] max_volume: -6.0 dB\n"
                    "[Parsed_volumedetect_0] mean_volume: -18.0 dB\n"
                )
            ),
            MagicMock(
                returncode=0,
                stderr="Audio stats\n"
            ),
        ]

        report = analyze_audio_quality(audio_file)

        assert len(report.silence_segments) == 1
        assert report.silence_segments[0] == (5.0, 10.0)
