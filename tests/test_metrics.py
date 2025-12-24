"""Tests for the metrics module."""
import json
import pytest
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from framewright.metrics import (
    ProcessingMetrics,
    ProgressReporter,
    ProgressUpdate,
    ConsoleProgressBar,
)


class TestProcessingMetrics:
    """Tests for ProcessingMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating processing metrics."""
        metrics = ProcessingMetrics(total_frames=100)

        assert metrics.total_frames == 100
        assert metrics.processed_frames == 0
        assert metrics.failed_frames == 0
        assert metrics.start_time is not None

    def test_record_frame_success(self):
        """Test recording successful frame processing."""
        metrics = ProcessingMetrics()

        metrics.record_frame(50.0, success=True)
        metrics.record_frame(60.0, success=True)

        assert metrics.processed_frames == 2
        assert metrics.failed_frames == 0
        assert metrics.min_frame_time_ms == 50.0
        assert metrics.max_frame_time_ms == 60.0

    def test_record_frame_failure(self):
        """Test recording failed frame processing."""
        metrics = ProcessingMetrics()

        metrics.record_frame(50.0, success=True)
        metrics.record_frame(100.0, success=False)

        assert metrics.processed_frames == 2
        assert metrics.failed_frames == 1

    def test_record_vram(self):
        """Test recording VRAM usage."""
        metrics = ProcessingMetrics()

        metrics.record_vram(2000)
        metrics.record_vram(4000)
        metrics.record_vram(3000)

        assert metrics.peak_vram_mb == 4000
        assert metrics.avg_vram_mb > 0

    def test_record_error(self):
        """Test recording errors."""
        metrics = ProcessingMetrics()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            metrics.record_error(e, context={"frame": 10})

        assert metrics.error_count == 1
        assert len(metrics.errors) == 1
        assert metrics.errors[0]["type"] == "ValueError"
        assert metrics.errors[0]["context"]["frame"] == 10

    def test_record_retry(self):
        """Test recording retry attempts."""
        metrics = ProcessingMetrics()

        metrics.record_retry()
        metrics.record_retry()

        assert metrics.retry_count == 2

    def test_record_checkpoint(self):
        """Test recording checkpoints."""
        metrics = ProcessingMetrics()

        metrics.record_checkpoint()
        metrics.record_checkpoint()

        assert metrics.checkpoint_count == 2

    def test_record_resume(self):
        """Test recording resume events."""
        metrics = ProcessingMetrics()

        metrics.record_resume()

        assert metrics.resume_count == 1

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = ProcessingMetrics()

        metrics.record_frame(50.0, success=True)
        metrics.record_frame(50.0, success=True)
        metrics.record_frame(50.0, success=False)
        metrics.record_frame(50.0, success=True)

        # 3 successes out of 4 = 75%
        assert metrics.success_rate == 75.0

    def test_success_rate_no_frames(self):
        """Test success rate with no frames."""
        metrics = ProcessingMetrics()

        assert metrics.success_rate == 0.0

    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        metrics = ProcessingMetrics()

        # Small delay
        time.sleep(0.1)

        assert metrics.elapsed_seconds >= 0.1

    def test_frames_per_second(self):
        """Test FPS calculation."""
        metrics = ProcessingMetrics()
        metrics.start_time = datetime.now()

        # Simulate some processing
        for _ in range(10):
            metrics.record_frame(10.0, success=True)

        # Should have some positive FPS
        assert metrics.frames_per_second > 0

    def test_finish(self):
        """Test finishing metrics collection."""
        metrics = ProcessingMetrics()

        assert metrics.end_time is None

        metrics.finish()

        assert metrics.end_time is not None

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = ProcessingMetrics(total_frames=100)
        metrics.record_frame(50.0, success=True)
        metrics.finish()

        data = metrics.to_dict()

        assert "total_frames" in data
        assert "processed_frames" in data
        assert "elapsed_seconds" in data
        assert "success_rate" in data
        assert "frames_per_second" in data
        assert data["total_frames"] == 100
        assert data["processed_frames"] == 1

    def test_export_json(self, tmp_path):
        """Test exporting to JSON file."""
        metrics = ProcessingMetrics(total_frames=50)
        metrics.record_frame(100.0, success=True)
        metrics.finish()

        output_path = tmp_path / "metrics" / "output.json"
        metrics.export_json(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["total_frames"] == 50
        assert data["processed_frames"] == 1

    def test_summary(self):
        """Test generating summary."""
        metrics = ProcessingMetrics(total_frames=100)
        metrics.record_frame(50.0, success=True)
        metrics.record_frame(60.0, success=True)
        metrics.record_vram(2000)
        metrics.finish()

        summary = metrics.summary()

        assert "Processing Metrics Summary" in summary
        assert "Total Frames: 100" in summary
        assert "Processed: 2" in summary


class TestProgressUpdate:
    """Tests for ProgressUpdate dataclass."""

    def test_create_update(self):
        """Test creating progress update."""
        update = ProgressUpdate(
            current=50,
            total=100,
            percentage=50.0,
            eta_seconds=30.0,
            elapsed_seconds=30.0,
            avg_frame_ms=100.0,
            frames_per_second=10.0,
        )

        assert update.current == 50
        assert update.total == 100
        assert update.percentage == 50.0
        assert update.status == "processing"


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_init(self):
        """Test reporter initialization."""
        reporter = ProgressReporter(total_frames=100)

        assert reporter.total == 100
        assert reporter.processed == 0
        assert len(reporter.frame_times) == 0

    def test_update(self):
        """Test progress update."""
        reporter = ProgressReporter(total_frames=100)

        update = reporter.update(frame_num=10, frame_time_ms=50.0)

        assert update.current == 10
        assert update.percentage == 10.0
        assert update.eta_seconds > 0

    def test_update_rolling_average(self):
        """Test rolling average calculation."""
        reporter = ProgressReporter(total_frames=100, window_size=5)

        for i in range(10):
            reporter.update(frame_num=i + 1, frame_time_ms=100.0)

        # Should only keep last 5 frame times
        assert len(reporter.frame_times) == 5

    def test_update_with_callback(self):
        """Test callback is called on update."""
        callback_calls = []

        def callback(update):
            callback_calls.append(update)

        reporter = ProgressReporter(total_frames=100, callback=callback)
        reporter._update_interval = 0  # Disable throttling for test

        reporter.update(frame_num=1, frame_time_ms=50.0)

        assert len(callback_calls) == 1

    def test_get_eta_string(self):
        """Test ETA string formatting."""
        reporter = ProgressReporter(total_frames=100)

        assert reporter.get_eta_string(0) == "Complete"
        assert reporter.get_eta_string(45) == "45s"
        assert reporter.get_eta_string(90) == "1m 30s"
        assert reporter.get_eta_string(3665) == "1h 1m 5s"

    def test_format_progress(self):
        """Test progress bar formatting."""
        reporter = ProgressReporter(total_frames=100)

        update = ProgressUpdate(
            current=50,
            total=100,
            percentage=50.0,
            eta_seconds=30.0,
            elapsed_seconds=30.0,
            avg_frame_ms=100.0,
            frames_per_second=10.0,
        )

        formatted = reporter.format_progress(update)

        assert "50.0%" in formatted
        assert "(50/100)" in formatted
        assert "ETA:" in formatted
        assert "fps" in formatted

    def test_finish(self):
        """Test finishing progress."""
        reporter = ProgressReporter(total_frames=100)

        for i in range(100):
            reporter.update(frame_num=i + 1, frame_time_ms=10.0)

        final = reporter.finish()

        assert final.current == 100
        assert final.total == 100
        assert final.percentage == 100.0
        assert final.status == "complete"


class TestConsoleProgressBar:
    """Tests for ConsoleProgressBar class."""

    def test_init(self):
        """Test progress bar initialization."""
        bar = ConsoleProgressBar(total=100, description="Test")

        assert bar.description == "Test"
        assert bar.reporter.total == 100

    @patch('builtins.print')
    def test_update(self, mock_print):
        """Test progress bar update."""
        bar = ConsoleProgressBar(total=100)

        bar.update(current=10, frame_time_ms=50.0)

        mock_print.assert_called()
        call_args = mock_print.call_args
        assert "Processing:" in call_args[0][0]

    @patch('builtins.print')
    def test_finish(self, mock_print):
        """Test finishing progress bar."""
        bar = ConsoleProgressBar(total=100)

        bar.finish()

        # Last call should include "Complete"
        call_args = mock_print.call_args[0][0]
        assert "Complete" in call_args
