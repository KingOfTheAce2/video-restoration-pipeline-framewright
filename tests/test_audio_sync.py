"""Tests for audio_sync module.

This module tests the AudioSyncAnalyzer and AudioSyncCorrector classes
for audio-video synchronization detection and correction.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framewright.processors.audio_sync import (
    AudioSyncAnalyzer,
    AudioSyncDetector,
    AudioSyncCorrector,
    AudioSyncError,
    AudioWaveformInfo,
    SyncAnalysis,
    SyncConfig,
    SyncCorrection,
    analyze_audio_sync,
    correct_audio_sync,
    sync_audio_to_video,
)


class TestAudioWaveformInfo:
    """Tests for AudioWaveformInfo dataclass."""

    def test_basic_initialization(self):
        """Test basic AudioWaveformInfo creation."""
        info = AudioWaveformInfo(
            sample_rate=48000,
            duration=120.5,
            peak_positions=[48000, 96000, 144000],
            rms_levels=[0.1, 0.2, 0.15],
            channels=2,
            bit_depth=16,
        )

        assert info.sample_rate == 48000
        assert info.duration == 120.5
        assert len(info.peak_positions) == 3
        assert info.channels == 2
        assert info.bit_depth == 16

    def test_peak_times(self):
        """Test conversion of peak positions to timestamps."""
        info = AudioWaveformInfo(
            sample_rate=48000,
            duration=10.0,
            peak_positions=[48000, 96000, 144000],
            rms_levels=[],
        )

        times = info.peak_times()
        assert len(times) == 3
        assert times[0] == pytest.approx(1.0, abs=0.001)
        assert times[1] == pytest.approx(2.0, abs=0.001)
        assert times[2] == pytest.approx(3.0, abs=0.001)

    def test_default_values(self):
        """Test default values for optional fields."""
        info = AudioWaveformInfo(
            sample_rate=44100,
            duration=60.0,
        )

        assert info.peak_positions == []
        assert info.rms_levels == []
        assert info.channels == 2
        assert info.bit_depth is None


class TestSyncAnalysis:
    """Tests for SyncAnalysis dataclass."""

    def test_basic_initialization(self):
        """Test basic SyncAnalysis creation."""
        analysis = SyncAnalysis(
            drift_ms=50.0,
            drift_direction="audio_ahead",
            confidence=0.85,
            samples_analyzed=25,
            is_progressive=False,
        )

        assert analysis.drift_ms == 50.0
        assert analysis.confidence == 0.85
        assert analysis.drift_direction == "audio_ahead"
        assert analysis.samples_analyzed == 25
        assert analysis.is_progressive is False

    def test_needs_correction_property(self):
        """Test needs_correction property returns True when drift is large enough."""
        analysis = SyncAnalysis(
            drift_ms=100.0,
            drift_direction="audio_ahead",
            confidence=0.85,
            samples_analyzed=25,
            is_progressive=False,
        )
        assert analysis.needs_correction is True

    def test_needs_correction_small_drift(self):
        """Test needs_correction returns False when drift is small."""
        analysis = SyncAnalysis(
            drift_ms=10.0,
            drift_direction="audio_ahead",
            confidence=0.85,
            samples_analyzed=25,
            is_progressive=False,
        )
        assert analysis.needs_correction is False

    def test_needs_correction_low_confidence(self):
        """Test needs_correction returns False when confidence is low."""
        analysis = SyncAnalysis(
            drift_ms=100.0,
            drift_direction="audio_ahead",
            confidence=0.3,
            samples_analyzed=25,
            is_progressive=False,
        )
        assert analysis.needs_correction is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = SyncAnalysis(
            drift_ms=-30.0,
            drift_direction="video_ahead",
            confidence=0.92,
            samples_analyzed=15,
            is_progressive=False,
        )

        data = analysis.to_dict()
        assert data["drift_ms"] == -30.0
        assert data["confidence"] == 0.92
        assert data["needs_correction"] is False
        assert data["drift_direction"] == "video_ahead"
        assert data["samples_analyzed"] == 15
        assert data["is_progressive"] is False

    def test_auto_determine_drift_direction(self):
        """Test drift_direction is auto-determined from drift_ms when invalid."""
        analysis = SyncAnalysis(
            drift_ms=50.0,
            drift_direction="invalid",
            confidence=0.85,
            samples_analyzed=10,
            is_progressive=False,
        )
        assert analysis.drift_direction == "audio_ahead"

    def test_confidence_clamped(self):
        """Test that confidence is clamped to [0.0, 1.0]."""
        analysis = SyncAnalysis(
            drift_ms=50.0,
            drift_direction="audio_ahead",
            confidence=1.5,
            samples_analyzed=10,
            is_progressive=False,
        )
        assert analysis.confidence == 1.0


class TestSyncCorrection:
    """Tests for SyncCorrection dataclass."""

    def test_basic_initialization(self):
        """Test basic SyncCorrection creation."""
        correction = SyncCorrection(
            original_duration=120.0,
            adjusted_duration=121.5,
            stretch_factor=1.0125,
            offset_applied_ms=50.0,
            output_path="/output/synced.wav",
            quality_preserved=True,
        )

        assert correction.original_duration == 120.0
        assert correction.adjusted_duration == 121.5
        assert correction.stretch_factor == pytest.approx(1.0125, abs=0.0001)
        assert correction.quality_preserved is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        correction = SyncCorrection(
            original_duration=60.0,
            adjusted_duration=60.5,
            stretch_factor=1.0083,
        )

        data = correction.to_dict()
        assert "original_duration" in data
        assert "adjusted_duration" in data
        assert "stretch_factor" in data


class TestSyncConfig:
    """Tests for SyncConfig dataclass."""

    def test_default_values(self):
        """Test default SyncConfig values."""
        config = SyncConfig()
        assert config.detection_method == "audio_fingerprint"
        assert config.max_drift_ms == 500.0
        assert config.correction_method == "resample"

    def test_invalid_detection_method(self):
        """Test that invalid detection_method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid detection_method"):
            SyncConfig(detection_method="invalid")

    def test_invalid_correction_method(self):
        """Test that invalid correction_method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid correction_method"):
            SyncConfig(correction_method="invalid")

    def test_invalid_max_drift(self):
        """Test that non-positive max_drift_ms raises ValueError."""
        with pytest.raises(ValueError, match="max_drift_ms must be positive"):
            SyncConfig(max_drift_ms=-1.0)


class TestAudioSyncAnalyzer:
    """Tests for AudioSyncAnalyzer (and its legacy alias AudioSyncDetector)."""

    @pytest.fixture
    def mock_ffmpeg(self):
        """Mock FFmpeg path lookup for testing."""
        with patch(
            "framewright.processors.audio_sync.get_ffmpeg_path",
            return_value="/usr/bin/ffmpeg",
        ):
            with patch(
                "framewright.processors.audio_sync.get_ffprobe_path",
                return_value="/usr/bin/ffprobe",
            ):
                yield

    @pytest.fixture
    def analyzer(self, mock_ffmpeg):
        """Create analyzer with mocked dependencies."""
        with patch.object(AudioSyncAnalyzer, "_check_librosa", return_value=False):
            return AudioSyncAnalyzer()

    def test_legacy_alias(self):
        """Test that AudioSyncDetector is an alias for AudioSyncAnalyzer."""
        assert AudioSyncDetector is AudioSyncAnalyzer

    def test_initialization(self, mock_ffmpeg):
        """Test AudioSyncAnalyzer initialization."""
        with patch.object(AudioSyncAnalyzer, "_check_librosa", return_value=True):
            analyzer = AudioSyncAnalyzer()
            assert analyzer._librosa_available is True

    def test_initialization_no_ffmpeg(self):
        """Test initialization fails without FFmpeg."""
        with patch(
            "framewright.processors.audio_sync.get_ffmpeg_path",
            side_effect=FileNotFoundError("ffmpeg not found"),
        ):
            with pytest.raises(AudioSyncError, match="FFmpeg"):
                AudioSyncAnalyzer()

    def test_calculate_drift_matching_events(self, analyzer):
        """Test drift calculation with matching events."""
        audio_events = [1.1, 2.1, 3.1, 4.1, 5.1]
        video_events = [1.0, 2.0, 3.0, 4.0, 5.0]

        drift_ms, confidence = analyzer._calculate_drift(audio_events, video_events)

        assert drift_ms == pytest.approx(-100.0, abs=60.0)
        assert confidence > 0.3

    def test_calculate_drift_no_events(self, analyzer):
        """Test drift calculation with no events."""
        drift_ms, confidence = analyzer._calculate_drift([], [])

        assert drift_ms == 0.0
        assert confidence == 0.0

    def test_calculate_drift_perfect_sync(self, analyzer):
        """Test drift calculation with perfectly synced events."""
        events = [1.0, 2.0, 3.0, 4.0, 5.0]

        drift_ms, confidence = analyzer._calculate_drift(events, events.copy())

        assert drift_ms == pytest.approx(0.0, abs=55.0)
        assert confidence > 0.5

    @patch("subprocess.run")
    def test_detect_peaks_ffmpeg(self, mock_run, analyzer):
        """Test audio peak detection using FFmpeg fallback."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr=(
                "[silencedetect @ 0x1234] silence_end: 1.5
"
                "[silencedetect @ 0x1234] silence_end: 3.2
"
                "[silencedetect @ 0x1234] silence_end: 5.8
"
            ),
            stdout=b"",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = Path(tmp_dir) / "test.wav"
            audio_file.touch()
            events = analyzer._detect_peaks_ffmpeg(audio_file)

        assert 1.5 in events
        assert 3.2 in events
        assert 5.8 in events

    @patch("subprocess.run")
    def test_detect_scene_changes(self, mock_run, analyzer):
        """Test video scene change detection."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr=(
                "[Parsed_showinfo] n:0 pts:1234 pts_time:2.5 fmt:yuv420p
"
                "[Parsed_showinfo] n:1 pts:5678 pts_time:7.3 fmt:yuv420p
"
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_file = Path(tmp_dir) / "test.mp4"
            video_file.touch()
            events = analyzer._detect_scene_changes(video_file)

        assert 2.5 in events
        assert 7.3 in events
