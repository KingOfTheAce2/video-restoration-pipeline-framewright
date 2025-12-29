"""Tests for the rich progress reporting system."""
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGPUMetrics:
    """Tests for GPUMetrics dataclass."""

    def test_memory_conversions(self):
        """Test memory unit conversions."""
        from framewright.utils.progress import GPUMetrics

        metrics = GPUMetrics(
            utilization_percent=85.0,
            memory_used_mb=4096.0,
            memory_total_mb=6144.0,
            available=True,
        )

        assert metrics.memory_used_gb == 4.0
        assert metrics.memory_total_gb == 6.0
        assert abs(metrics.memory_percent - 66.67) < 0.1

    def test_zero_memory(self):
        """Test handling of zero memory."""
        from framewright.utils.progress import GPUMetrics

        metrics = GPUMetrics(
            utilization_percent=0.0,
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            available=False,
        )

        assert metrics.memory_percent == 0.0


class TestStageInfo:
    """Tests for StageInfo dataclass."""

    def test_stage_progress(self):
        """Test stage progress calculation."""
        from framewright.utils.progress import StageInfo

        stage = StageInfo(
            name="Upscaling",
            total_frames=1000,
            completed_frames=500,
            stage_index=1,
            total_stages=5,
        )

        assert stage.progress_percent == 50.0
        assert not stage.is_complete

    def test_complete_stage(self):
        """Test complete stage detection."""
        from framewright.utils.progress import StageInfo

        stage = StageInfo(
            name="Upscaling",
            total_frames=1000,
            completed_frames=1000,
        )

        assert stage.is_complete
        assert stage.progress_percent == 100.0

    def test_fps_calculation(self):
        """Test FPS calculation."""
        from framewright.utils.progress import StageInfo

        start = time.time() - 10.0  # 10 seconds ago

        stage = StageInfo(
            name="Processing",
            total_frames=100,
            completed_frames=50,
            start_time=start,
        )

        # Should be approximately 5 fps (50 frames / 10 seconds)
        assert 4.5 <= stage.fps <= 5.5

    def test_eta_calculation(self):
        """Test ETA calculation."""
        from framewright.utils.progress import StageInfo

        start = time.time() - 10.0

        stage = StageInfo(
            name="Processing",
            total_frames=100,
            completed_frames=50,
            start_time=start,
        )

        # 50 remaining frames at ~5 fps = ~10 seconds
        eta = stage.eta_seconds
        assert eta is not None
        assert 8.0 <= eta <= 12.0

    def test_zero_frames(self):
        """Test handling of zero frames."""
        from framewright.utils.progress import StageInfo

        stage = StageInfo(
            name="Empty",
            total_frames=0,
            completed_frames=0,
        )

        assert stage.progress_percent == 100.0
        assert stage.is_complete


class TestGPUMonitor:
    """Tests for GPUMonitor class."""

    def test_fallback_no_gpu(self):
        """Test fallback when no GPU is available."""
        from framewright.utils.progress import GPUMonitor

        with patch("shutil.which", return_value=None):
            with patch.dict("sys.modules", {"pynvml": None}):
                monitor = GPUMonitor()
                metrics = monitor.get_metrics()

                assert not metrics.available

    def test_nvidia_smi_fallback(self):
        """Test nvidia-smi fallback metrics."""
        from framewright.utils.progress import GPUMonitor

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "85, 4096, 6144, 65"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                # Force nvidia-smi path by making pynvml unavailable
                with patch.object(GPUMonitor, "_init_backend"):
                    monitor = GPUMonitor()
                    monitor._initialized = True
                    monitor._handle = None

                    metrics = monitor._get_metrics_nvidia_smi()

                    assert metrics.available
                    assert metrics.utilization_percent == 85.0
                    assert metrics.memory_used_mb == 4096.0
                    assert metrics.memory_total_mb == 6144.0
                    assert metrics.temperature_celsius == 65.0

    def test_nvidia_smi_failure(self):
        """Test handling of nvidia-smi failure."""
        from framewright.utils.progress import GPUMonitor

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            monitor = GPUMonitor()
            monitor._initialized = True
            monitor._handle = None

            metrics = monitor._get_metrics_nvidia_smi()

            assert not metrics.available


class TestLogFileCallback:
    """Tests for LogFileCallback class."""

    def test_log_stage_start(self):
        """Test logging stage start."""
        from framewright.utils.progress import LogFileCallback, StageInfo

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            callback = LogFileCallback(log_path)

            stage = StageInfo(
                name="Upscaling",
                total_frames=1000,
                stage_index=1,
                total_stages=5,
            )

            callback.on_stage_start(stage)

            content = log_path.read_text()
            assert "STAGE_START" in content
            assert "Upscaling" in content
            assert "2/5" in content
            assert "1000 frames" in content

    def test_log_progress_update(self):
        """Test logging progress update."""
        from framewright.utils.progress import (
            GPUMetrics,
            LogFileCallback,
            StageInfo,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            callback = LogFileCallback(log_path, include_gpu_metrics=True)

            stage = StageInfo(
                name="Upscaling",
                total_frames=1000,
                completed_frames=500,
                start_time=time.time() - 10.0,
            )

            gpu_metrics = GPUMetrics(
                utilization_percent=85.0,
                memory_used_mb=4096.0,
                memory_total_mb=6144.0,
                available=True,
            )

            callback.on_progress_update(stage, gpu_metrics)

            content = log_path.read_text()
            assert "PROGRESS" in content
            assert "500/1000" in content
            assert "GPU: 85%" in content
            assert "VRAM:" in content

    def test_log_stage_complete(self):
        """Test logging stage completion."""
        from framewright.utils.progress import LogFileCallback, StageInfo

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            callback = LogFileCallback(log_path)

            stage = StageInfo(
                name="Upscaling",
                total_frames=1000,
                completed_frames=1000,
                start_time=time.time() - 100.0,
                end_time=time.time(),
            )

            callback.on_stage_complete(stage)

            content = log_path.read_text()
            assert "STAGE_COMPLETE" in content
            assert "Upscaling" in content
            assert "1000 frames" in content

    def test_log_pipeline_complete(self):
        """Test logging pipeline completion."""
        from framewright.utils.progress import LogFileCallback, StageInfo

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            callback = LogFileCallback(log_path)

            stages = [
                StageInfo(
                    name="Stage1",
                    total_frames=500,
                    start_time=time.time() - 20.0,
                    end_time=time.time() - 10.0,
                ),
                StageInfo(
                    name="Stage2",
                    total_frames=500,
                    start_time=time.time() - 10.0,
                    end_time=time.time(),
                ),
            ]

            callback.on_pipeline_complete(stages)

            content = log_path.read_text()
            assert "PIPELINE_COMPLETE" in content
            assert "2 stages" in content
            assert "1000 total frames" in content


class TestFallbackProgress:
    """Tests for FallbackProgress class (non-rich fallback)."""

    def test_start_stage(self):
        """Test starting a stage."""
        from framewright.utils.progress import FallbackProgress, ProgressOutputMode

        progress = FallbackProgress(
            video_name="test.mp4",
            output_mode=ProgressOutputMode.SILENT,
        )

        stage = progress.start_stage(
            name="Upscaling",
            total_frames=1000,
            stage_index=0,
            total_stages=3,
        )

        assert stage.name == "Upscaling"
        assert stage.total_frames == 1000
        assert stage.stage_index == 0
        assert stage.total_stages == 3
        assert stage.start_time is not None

    def test_update_progress(self):
        """Test updating progress."""
        from framewright.utils.progress import FallbackProgress, ProgressOutputMode

        progress = FallbackProgress(
            video_name="test.mp4",
            output_mode=ProgressOutputMode.SILENT,
        )

        progress.start_stage("Upscaling", 1000)
        progress.update(500)

        assert progress.current_stage is not None
        assert progress.current_stage.completed_frames == 500

    def test_complete_stage(self):
        """Test completing a stage."""
        from framewright.utils.progress import FallbackProgress, ProgressOutputMode

        progress = FallbackProgress(
            video_name="test.mp4",
            output_mode=ProgressOutputMode.SILENT,
        )

        progress.start_stage("Upscaling", 1000)
        progress.update(1000)
        progress.complete_stage()

        assert progress.current_stage is not None
        assert progress.current_stage.end_time is not None


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_create_tracker(self):
        """Test creating a progress tracker."""
        from framewright.utils.progress import (
            ProgressOutputMode,
            ProgressTracker,
        )

        tracker = ProgressTracker(
            video_name="test.mp4",
            total_stages=5,
            output_mode=ProgressOutputMode.SILENT,
        )

        assert tracker.video_name == "test.mp4"
        assert tracker.total_stages == 5

    def test_track_multiple_stages(self):
        """Test tracking multiple stages."""
        from framewright.utils.progress import (
            ProgressOutputMode,
            ProgressTracker,
        )

        tracker = ProgressTracker(
            video_name="test.mp4",
            total_stages=3,
            output_mode=ProgressOutputMode.SILENT,
        )

        # Stage 1
        tracker.start_stage("Upscaling", 100)
        for i in range(100):
            tracker.advance()
        tracker.complete_stage()

        # Stage 2
        tracker.start_stage("Face Restoration", 100)
        for i in range(100):
            tracker.advance()
        tracker.complete_stage()

        assert len(tracker.stages) == 2
        assert tracker.current_stage_index == 2

    def test_get_summary(self):
        """Test getting pipeline summary."""
        from framewright.utils.progress import (
            ProgressOutputMode,
            ProgressTracker,
        )

        tracker = ProgressTracker(
            video_name="test.mp4",
            total_stages=2,
            output_mode=ProgressOutputMode.SILENT,
        )

        tracker.start_stage("Stage1", 100)
        tracker.update(100)
        tracker.complete_stage()

        summary = tracker.get_summary()

        assert summary["video_name"] == "test.mp4"
        assert summary["total_stages"] == 2
        assert summary["completed_stages"] == 1
        assert summary["total_frames"] == 100

    def test_stage_context_manager(self):
        """Test stage context manager."""
        from framewright.utils.progress import (
            ProgressOutputMode,
            ProgressTracker,
        )

        tracker = ProgressTracker(
            video_name="test.mp4",
            total_stages=1,
            output_mode=ProgressOutputMode.SILENT,
        )

        with tracker.stage("Processing", 100) as stage:
            assert stage.name == "Processing"
            tracker.update(50)
            tracker.update(100)

        assert len(tracker.stages) == 1


class TestCreateProgressTracker:
    """Tests for create_progress_tracker factory function."""

    def test_create_with_defaults(self):
        """Test creating tracker with default options."""
        from framewright.utils.progress import create_progress_tracker

        tracker = create_progress_tracker(
            video_name="test.mp4",
            total_stages=3,
            output_mode="silent",
        )

        assert tracker.video_name == "test.mp4"
        assert tracker.total_stages == 3

    def test_create_with_log_output(self):
        """Test creating tracker with log output."""
        from framewright.utils.progress import create_progress_tracker

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "progress.log"

            tracker = create_progress_tracker(
                video_name="test.mp4",
                total_stages=1,
                output_mode="log",
                log_path=log_path,
            )

            tracker.start_stage("Test", 100)
            tracker.complete_stage()
            tracker.complete_pipeline()

            assert log_path.exists()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_rich_available(self):
        """Test rich availability check."""
        from framewright.utils.progress import is_rich_available

        # Just verify it returns a boolean
        result = is_rich_available()
        assert isinstance(result, bool)

    def test_is_pynvml_available(self):
        """Test pynvml availability check."""
        from framewright.utils.progress import is_pynvml_available

        result = is_pynvml_available()
        assert isinstance(result, bool)


class TestProgressOutputMode:
    """Tests for ProgressOutputMode enum."""

    def test_enum_values(self):
        """Test enum values."""
        from framewright.utils.progress import ProgressOutputMode

        assert ProgressOutputMode.TERMINAL.value == "terminal"
        assert ProgressOutputMode.LOG.value == "log"
        assert ProgressOutputMode.BOTH.value == "both"
        assert ProgressOutputMode.SILENT.value == "silent"


@pytest.mark.skipif(
    not pytest.importorskip("rich", reason="rich not installed"),
    reason="rich not installed",
)
class TestRichProgress:
    """Tests for RichProgress class (requires rich)."""

    def test_rich_progress_creation(self):
        """Test creating RichProgress instance."""
        from framewright.utils.progress import RichProgress

        progress = RichProgress(video_name="test.mp4")
        assert progress.video_name == "test.mp4"
        progress.gpu_monitor.shutdown()

    def test_rich_progress_stage(self):
        """Test RichProgress stage handling."""
        from framewright.utils.progress import RichProgress

        progress = RichProgress(video_name="test.mp4")

        try:
            stage = progress.start_stage(
                name="Upscaling",
                total_frames=100,
                stage_index=0,
                total_stages=2,
            )

            assert stage.name == "Upscaling"
            assert stage.total_frames == 100

            progress.update(50)
            assert progress.current_stage.completed_frames == 50

            progress.advance(10)
            assert progress.current_stage.completed_frames == 60

            progress.complete_stage()
            assert len(progress.stages) == 1
        finally:
            progress.gpu_monitor.shutdown()
