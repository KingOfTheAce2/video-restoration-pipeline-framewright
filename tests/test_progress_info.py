"""Tests for ProgressInfo dataclass.

Tests cover the ProgressInfo dataclass used for detailed progress tracking
including ETA calculations, elapsed time formatting, and frames per second.
"""
import time
from unittest.mock import patch
import pytest

from framewright.restorer import ProgressInfo


class TestProgressInfoCreation:
    """Test ProgressInfo dataclass creation and initialization."""

    def test_create_with_required_fields(self):
        """Test creating ProgressInfo with only required fields."""
        info = ProgressInfo(
            stage="extract_frames",
            progress=0.5
        )

        assert info.stage == "extract_frames"
        assert info.progress == 0.5
        assert info.eta_seconds is None
        assert info.frames_completed == 0
        assert info.frames_total == 0
        assert info.elapsed_seconds >= 0.0

    def test_create_with_all_fields(self):
        """Test creating ProgressInfo with all fields specified."""
        start_time = time.time() - 100  # 100 seconds ago

        info = ProgressInfo(
            stage="enhance_frames",
            progress=0.75,
            eta_seconds=150.0,
            frames_completed=750,
            frames_total=1000,
            stage_start_time=start_time,
            elapsed_seconds=100.0
        )

        assert info.stage == "enhance_frames"
        assert info.progress == 0.75
        assert info.eta_seconds == 150.0
        assert info.frames_completed == 750
        assert info.frames_total == 1000
        assert info.stage_start_time == start_time
        assert info.elapsed_seconds == 100.0

    def test_post_init_calculates_elapsed_when_zero(self):
        """Test that __post_init__ calculates elapsed_seconds when not provided."""
        start_time = time.time() - 5.0  # 5 seconds ago

        info = ProgressInfo(
            stage="test",
            progress=0.0,
            stage_start_time=start_time,
            elapsed_seconds=0.0  # Should trigger calculation
        )

        # Should have calculated approximately 5 seconds elapsed
        assert info.elapsed_seconds >= 4.5
        assert info.elapsed_seconds <= 6.0

    def test_post_init_preserves_explicit_elapsed(self):
        """Test that explicit elapsed_seconds is preserved."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            elapsed_seconds=42.0
        )

        assert info.elapsed_seconds == 42.0


class TestEtaFormatted:
    """Test eta_formatted property."""

    def test_eta_formatted_unknown_when_none(self):
        """Test eta_formatted returns 'Unknown' when eta_seconds is None."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=None
        )

        assert info.eta_formatted == "Unknown"

    def test_eta_formatted_unknown_when_negative(self):
        """Test eta_formatted returns 'Unknown' for negative values."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=-10.0
        )

        assert info.eta_formatted == "Unknown"

    def test_eta_formatted_seconds_only(self):
        """Test eta_formatted for durations under a minute."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=45.0
        )

        assert info.eta_formatted == "0:45"

    def test_eta_formatted_minutes_and_seconds(self):
        """Test eta_formatted for durations with minutes and seconds."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=125.0  # 2 minutes and 5 seconds
        )

        assert info.eta_formatted == "2:05"

    def test_eta_formatted_hours_minutes_seconds(self):
        """Test eta_formatted for durations with hours."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=3661.0  # 1 hour, 1 minute, 1 second
        )

        assert info.eta_formatted == "1:01:01"

    def test_eta_formatted_large_hours(self):
        """Test eta_formatted for large hour values."""
        info = ProgressInfo(
            stage="test",
            progress=0.1,
            eta_seconds=36000.0  # 10 hours
        )

        assert info.eta_formatted == "10:00:00"

    def test_eta_formatted_zero_seconds(self):
        """Test eta_formatted for zero seconds."""
        info = ProgressInfo(
            stage="test",
            progress=1.0,
            eta_seconds=0.0
        )

        assert info.eta_formatted == "0:00"

    def test_eta_formatted_fractional_seconds(self):
        """Test eta_formatted rounds fractional seconds correctly."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=65.7  # 1 minute and 5.7 seconds -> 1:05
        )

        assert info.eta_formatted == "1:05"

    def test_eta_formatted_zero_padded(self):
        """Test that seconds and minutes are zero-padded."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=3605.0  # 1 hour, 0 minutes, 5 seconds
        )

        assert info.eta_formatted == "1:00:05"


class TestElapsedFormatted:
    """Test elapsed_formatted property."""

    def test_elapsed_formatted_seconds_only(self):
        """Test elapsed_formatted for durations under a minute."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            elapsed_seconds=30.0
        )

        assert info.elapsed_formatted == "0:30"

    def test_elapsed_formatted_minutes_and_seconds(self):
        """Test elapsed_formatted for durations with minutes."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            elapsed_seconds=185.0  # 3 minutes and 5 seconds
        )

        assert info.elapsed_formatted == "3:05"

    def test_elapsed_formatted_hours_minutes_seconds(self):
        """Test elapsed_formatted for durations with hours."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            elapsed_seconds=7265.0  # 2 hours, 1 minute, 5 seconds
        )

        assert info.elapsed_formatted == "2:01:05"

    def test_elapsed_formatted_zero(self):
        """Test elapsed_formatted for zero elapsed time."""
        info = ProgressInfo(
            stage="test",
            progress=0.0,
            elapsed_seconds=0.0
        )

        assert info.elapsed_formatted == "0:00"

    def test_elapsed_formatted_zero_padded(self):
        """Test that minutes and seconds are properly zero-padded."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            elapsed_seconds=3601.0  # 1 hour, 0 minutes, 1 second
        )

        assert info.elapsed_formatted == "1:00:01"


class TestFramesPerSecond:
    """Test frames_per_second property calculation."""

    def test_frames_per_second_normal_calculation(self):
        """Test normal fps calculation."""
        info = ProgressInfo(
            stage="enhance",
            progress=0.5,
            frames_completed=100,
            frames_total=200,
            elapsed_seconds=10.0
        )

        assert info.frames_per_second == 10.0  # 100 frames / 10 seconds

    def test_frames_per_second_fractional(self):
        """Test fps with fractional result."""
        info = ProgressInfo(
            stage="enhance",
            progress=0.5,
            frames_completed=75,
            frames_total=150,
            elapsed_seconds=10.0
        )

        assert info.frames_per_second == 7.5

    def test_frames_per_second_zero_elapsed(self):
        """Test fps returns 0 when elapsed is zero."""
        info = ProgressInfo(
            stage="enhance",
            progress=0.5,
            frames_completed=100,
            frames_total=200,
            elapsed_seconds=0.0
        )

        assert info.frames_per_second == 0.0

    def test_frames_per_second_zero_frames(self):
        """Test fps returns 0 when no frames completed."""
        info = ProgressInfo(
            stage="enhance",
            progress=0.0,
            frames_completed=0,
            frames_total=200,
            elapsed_seconds=10.0
        )

        assert info.frames_per_second == 0.0

    def test_frames_per_second_high_rate(self):
        """Test fps for high frame processing rate."""
        info = ProgressInfo(
            stage="extract",
            progress=0.5,
            frames_completed=5000,
            frames_total=10000,
            elapsed_seconds=50.0
        )

        assert info.frames_per_second == 100.0  # 5000 / 50 = 100 fps

    def test_frames_per_second_low_rate(self):
        """Test fps for slow frame processing."""
        info = ProgressInfo(
            stage="enhance",
            progress=0.5,
            frames_completed=5,
            frames_total=100,
            elapsed_seconds=300.0  # 5 minutes for 5 frames
        )

        expected_fps = 5 / 300.0  # ~0.0167 fps
        assert abs(info.frames_per_second - expected_fps) < 0.001


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_progress_at_zero(self):
        """Test ProgressInfo at start of processing."""
        info = ProgressInfo(
            stage="extract_frames",
            progress=0.0,
            frames_completed=0,
            frames_total=1000
        )

        assert info.progress == 0.0
        assert info.frames_completed == 0
        assert info.frames_per_second == 0.0

    def test_progress_at_completion(self):
        """Test ProgressInfo at end of processing."""
        info = ProgressInfo(
            stage="enhance_frames",
            progress=1.0,
            eta_seconds=0.0,
            frames_completed=1000,
            frames_total=1000,
            elapsed_seconds=500.0
        )

        assert info.progress == 1.0
        assert info.eta_formatted == "0:00"
        assert info.frames_per_second == 2.0

    def test_progress_exceeds_one(self):
        """Test ProgressInfo handles progress > 1.0 (edge case)."""
        info = ProgressInfo(
            stage="test",
            progress=1.5  # Shouldn't happen but shouldn't crash
        )

        assert info.progress == 1.5

    def test_negative_progress(self):
        """Test ProgressInfo handles negative progress (edge case)."""
        info = ProgressInfo(
            stage="test",
            progress=-0.1  # Shouldn't happen but shouldn't crash
        )

        assert info.progress == -0.1

    def test_very_large_frame_count(self):
        """Test with very large frame counts."""
        info = ProgressInfo(
            stage="enhance",
            progress=0.5,
            frames_completed=1000000,
            frames_total=2000000,
            elapsed_seconds=86400.0  # 24 hours
        )

        expected_fps = 1000000 / 86400.0  # ~11.57 fps
        assert abs(info.frames_per_second - expected_fps) < 0.01

    def test_very_long_eta(self):
        """Test formatting very long ETA values."""
        info = ProgressInfo(
            stage="test",
            progress=0.01,
            eta_seconds=360000.0  # 100 hours
        )

        assert info.eta_formatted == "100:00:00"

    def test_stage_name_preservation(self):
        """Test various stage names are preserved correctly."""
        stages = [
            "download",
            "analyze_metadata",
            "extract_audio",
            "extract_frames",
            "enhance_frames",
            "interpolate_frames",
            "reassemble_video",
            "auto_enhance"
        ]

        for stage_name in stages:
            info = ProgressInfo(stage=stage_name, progress=0.5)
            assert info.stage == stage_name

    def test_immutability_of_calculated_properties(self):
        """Test that properties return consistent values."""
        info = ProgressInfo(
            stage="test",
            progress=0.5,
            eta_seconds=125.0,
            frames_completed=100,
            frames_total=200,
            elapsed_seconds=10.0
        )

        # Call properties multiple times
        assert info.eta_formatted == info.eta_formatted
        assert info.elapsed_formatted == info.elapsed_formatted
        assert info.frames_per_second == info.frames_per_second
