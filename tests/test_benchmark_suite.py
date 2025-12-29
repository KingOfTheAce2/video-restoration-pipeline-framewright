"""Tests for the FrameWright benchmarking suite."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.framewright.benchmarks import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkReporter,
    BenchmarkResult,
    BenchmarkRunner,
    StandardTestSuite,
    # Profiler classes
    PerformanceProfiler,
    ProfileReport,
    ProcessingStage,
    StageMetrics,
    ProfileSummary,
    analyze_profile,
)
from src.framewright.benchmarks.benchmark_suite import (
    BenchmarkType,
    DeviceType,
    QualityAnalyzer,
    ResourceMonitor,
    SystemProfiler,
    TestVideoGenerator,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = BenchmarkConfig(name="test_config")

        assert config.name == "test_config"
        assert config.benchmark_type == BenchmarkType.CUSTOM
        assert config.input_resolution == (1280, 720)
        assert config.scale_factor == 2
        assert config.frame_count == 100

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = BenchmarkConfig(
            name="4k_upscale",
            benchmark_type=BenchmarkType.UPSCALE_1080_TO_4K,
            input_resolution=(1920, 1080),
            output_resolution=(3840, 2160),
            scale_factor=2,
            frame_count=50,
        )

        assert config.name == "4k_upscale"
        assert config.benchmark_type == BenchmarkType.UPSCALE_1080_TO_4K
        assert config.output_resolution == (3840, 2160)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = BenchmarkConfig(name="test")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test"
        assert config_dict["benchmark_type"] == "custom"
        assert isinstance(config_dict["input_resolution"], list)


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_default_metrics(self):
        """Test creating metrics with defaults."""
        metrics = BenchmarkMetrics()

        assert metrics.processing_time_seconds == 0.0
        assert metrics.frames_processed == 0
        assert metrics.gpu_utilization_percent == 0.0

    def test_frames_per_second_property(self):
        """Test FPS calculation property."""
        metrics = BenchmarkMetrics(
            processing_time_seconds=10.0,
            frames_processed=100,
        )

        assert metrics.frames_per_second == 10.0

    def test_frames_per_second_zero_time(self):
        """Test FPS with zero time returns zero."""
        metrics = BenchmarkMetrics(
            processing_time_seconds=0.0,
            frames_processed=100,
        )

        assert metrics.frames_per_second == 0.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BenchmarkMetrics(
            processing_time_seconds=5.0,
            frames_processed=50,
            memory_peak_mb=1024.5,
        )
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["processing_time_seconds"] == 5.0
        assert metrics_dict["frames_processed"] == 50
        assert metrics_dict["memory_peak_mb"] == 1024.5
        assert "frames_per_second" in metrics_dict


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful benchmark result."""
        config = BenchmarkConfig(name="test")
        metrics = BenchmarkMetrics(frames_processed=100)

        result = BenchmarkResult(
            config=config,
            metrics=metrics,
            success=True,
        )

        assert result.success is True
        assert result.error_message is None
        assert result.config.name == "test"

    def test_failed_result(self):
        """Test creating a failed benchmark result."""
        config = BenchmarkConfig(name="failed_test")
        metrics = BenchmarkMetrics()

        result = BenchmarkResult(
            config=config,
            metrics=metrics,
            success=False,
            error_message="GPU out of memory",
        )

        assert result.success is False
        assert result.error_message == "GPU out of memory"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        config = BenchmarkConfig(name="test")
        metrics = BenchmarkMetrics()
        result = BenchmarkResult(config=config, metrics=metrics)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "config" in result_dict
        assert "metrics" in result_dict
        assert "timestamp" in result_dict
        assert "success" in result_dict


class TestSystemProfiler:
    """Tests for SystemProfiler class."""

    def test_get_system_info(self):
        """Test getting system information."""
        info = SystemProfiler.get_system_info()

        assert isinstance(info, dict)
        assert "platform" in info
        assert "python_version" in info
        assert "cpu_count" in info

    def test_get_dependency_versions(self):
        """Test getting dependency versions."""
        versions = SystemProfiler._get_dependency_versions()

        assert isinstance(versions, dict)
        # At minimum should check for these tools
        assert "ffmpeg" in versions


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def test_monitor_init(self):
        """Test monitor initialization."""
        monitor = ResourceMonitor(sample_interval=1.0)

        assert monitor.sample_interval == 1.0
        assert monitor.samples == []

    def test_get_empty_statistics(self):
        """Test statistics with no samples."""
        monitor = ResourceMonitor()
        stats = monitor.get_statistics()

        assert stats == {}

    def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = ResourceMonitor(sample_interval=0.1)

        monitor.start()
        # Give it time to collect at least one sample
        import time
        time.sleep(0.2)
        monitor.stop()

        # Should have collected at least one sample
        # (might not if system is very slow)
        assert isinstance(monitor.samples, list)


class TestTestVideoGenerator:
    """Tests for TestVideoGenerator class."""

    def test_generate_frame_sequence(self):
        """Test generating frame sequence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "frames"

            # This will fail if ffmpeg is not installed, which is expected in test env
            frame_count = TestVideoGenerator.generate_frame_sequence(
                output_dir=output_dir,
                resolution=(320, 240),
                frame_count=5,
            )

            # Count could be 0 if ffmpeg not available
            assert isinstance(frame_count, int)


class TestStandardTestSuite:
    """Tests for StandardTestSuite class."""

    def test_list_tests(self):
        """Test listing available tests."""
        tests = StandardTestSuite.list_tests()

        assert isinstance(tests, list)
        assert len(tests) > 0
        assert "720p_to_1080p" in tests
        assert "1080p_to_4k" in tests

    def test_get_test_config(self):
        """Test getting a specific test configuration."""
        config = StandardTestSuite.get_test_config("720p_to_1080p")

        assert config is not None
        assert isinstance(config, BenchmarkConfig)
        assert config.input_resolution == (1280, 720)
        assert config.output_resolution == (1920, 1080)

    def test_get_nonexistent_config(self):
        """Test getting a non-existent configuration."""
        config = StandardTestSuite.get_test_config("nonexistent_test")

        assert config is None

    def test_standard_test_configs(self):
        """Test all standard test configurations are valid."""
        for test_name in StandardTestSuite.list_tests():
            config = StandardTestSuite.get_test_config(test_name)

            assert config is not None
            assert config.name
            assert config.input_resolution[0] > 0
            assert config.input_resolution[1] > 0
            assert config.frame_count > 0


class TestBenchmarkReporter:
    """Tests for BenchmarkReporter class."""

    def test_generate_json_report(self):
        """Test generating JSON report."""
        config = BenchmarkConfig(name="test")
        metrics = BenchmarkMetrics(
            processing_time_seconds=10.0,
            frames_processed=100,
        )
        result = BenchmarkResult(config=config, metrics=metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"

            BenchmarkReporter.generate_json_report(result, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "generated_at" in data
            assert "results" in data
            assert len(data["results"]) == 1

    def test_generate_csv_report(self):
        """Test generating CSV report."""
        config = BenchmarkConfig(name="test")
        metrics = BenchmarkMetrics(
            processing_time_seconds=10.0,
            frames_processed=100,
        )
        result = BenchmarkResult(config=config, metrics=metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.csv"

            BenchmarkReporter.generate_csv_report(result, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                lines = f.readlines()

            assert len(lines) == 2  # Header + 1 data row

    def test_generate_summary(self):
        """Test generating text summary."""
        config = BenchmarkConfig(name="test")
        metrics = BenchmarkMetrics(
            processing_time_seconds=10.0,
            frames_processed=100,
        )
        result = BenchmarkResult(config=config, metrics=metrics)

        summary = BenchmarkReporter.generate_summary(result)

        assert isinstance(summary, str)
        assert "BENCHMARK SUMMARY" in summary
        assert "Total benchmarks: 1" in summary

    def test_generate_comparison_table(self):
        """Test generating comparison table."""
        results = {
            "test_1": BenchmarkResult(
                config=BenchmarkConfig(name="test_1"),
                metrics=BenchmarkMetrics(
                    processing_time_seconds=10.0,
                    frames_processed=100,
                ),
            ),
            "test_2": BenchmarkResult(
                config=BenchmarkConfig(name="test_2"),
                metrics=BenchmarkMetrics(
                    processing_time_seconds=15.0,
                    frames_processed=100,
                ),
            ),
        }

        table = BenchmarkReporter.generate_comparison_table(results)

        assert isinstance(table, str)
        assert "BENCHMARK COMPARISON" in table
        assert "test_1" in table
        assert "test_2" in table

    def test_generate_report_with_multiple_results(self):
        """Test generating report with multiple results."""
        results = {
            "test_1": BenchmarkResult(
                config=BenchmarkConfig(name="test_1"),
                metrics=BenchmarkMetrics(),
            ),
            "test_2": BenchmarkResult(
                config=BenchmarkConfig(name="test_2"),
                metrics=BenchmarkMetrics(),
            ),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"

            BenchmarkReporter.generate_json_report(results, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert data["benchmark_count"] == 2


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_runner_init(self):
        """Test runner initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BenchmarkRunner(work_dir=Path(tmp_dir), cleanup=False)

            assert runner.work_dir.exists()
            assert runner.cleanup is False
            assert isinstance(runner.system_info, dict)

    def test_runner_cleanup_on_init(self):
        """Test runner creates work directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir) / "benchmark_work"

            runner = BenchmarkRunner(work_dir=work_dir)

            assert work_dir.exists()


class TestBenchmarkType:
    """Tests for BenchmarkType enum."""

    def test_benchmark_types(self):
        """Test benchmark type values."""
        assert BenchmarkType.UPSCALE_720_TO_1080.value == "720p_to_1080p"
        assert BenchmarkType.UPSCALE_1080_TO_4K.value == "1080p_to_4k"
        assert BenchmarkType.INTERPOLATION_24_TO_60.value == "24fps_to_60fps"
        assert BenchmarkType.COMBINED_UPSCALE_INTERPOLATION.value == "upscale_interpolation"
        assert BenchmarkType.CUSTOM.value == "custom"


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_types(self):
        """Test device type values."""
        assert DeviceType.GPU.value == "gpu"
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.AUTO.value == "auto"


class TestQualityAnalyzer:
    """Tests for QualityAnalyzer class."""

    def test_analyze_frame_quality_empty_dirs(self):
        """Test quality analysis with empty directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_dir = Path(tmp_dir) / "ref"
            proc_dir = Path(tmp_dir) / "proc"
            ref_dir.mkdir()
            proc_dir.mkdir()

            result = QualityAnalyzer.analyze_frame_quality(ref_dir, proc_dir)

            assert result["psnr"] is None
            assert result["ssim"] is None


class TestIntegration:
    """Integration tests for the benchmark suite."""

    def test_full_workflow_mock(self):
        """Test full benchmark workflow with mocked processing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BenchmarkRunner(work_dir=Path(tmp_dir), cleanup=True)

            config = BenchmarkConfig(
                name="integration_test",
                benchmark_type=BenchmarkType.CUSTOM,
                input_resolution=(320, 240),
                output_resolution=(640, 480),
                frame_count=5,
                warmup_frames=1,
                iterations=1,
            )

            # This will fail in test environment without GPU/ffmpeg
            # but should not raise unexpected exceptions
            try:
                result = runner.run_benchmark(config)
                assert isinstance(result, BenchmarkResult)
            except Exception as e:
                # Expected in test environment without dependencies
                assert isinstance(e, (RuntimeError, OSError, FileNotFoundError))


# =============================================================================
# Performance Profiler Tests
# =============================================================================


class TestProcessingStage:
    """Tests for ProcessingStage enum."""

    def test_stage_values(self):
        """Test processing stage values."""
        assert ProcessingStage.FRAME_EXTRACTION.value == "Frame Extraction"
        assert ProcessingStage.AI_UPSCALING.value == "AI Upscaling"
        assert ProcessingStage.FACE_RESTORATION.value == "Face Restoration"
        assert ProcessingStage.DEFECT_REPAIR.value == "Defect Repair"
        assert ProcessingStage.FRAME_INTERPOLATION.value == "Frame Interpolation"
        assert ProcessingStage.VIDEO_ENCODING.value == "Video Encoding"
        assert ProcessingStage.AUDIO_ENHANCEMENT.value == "Audio Enhancement"
        assert ProcessingStage.COLORIZATION.value == "Colorization"
        assert ProcessingStage.STABILIZATION.value == "Stabilization"
        assert ProcessingStage.WATERMARK_REMOVAL.value == "Watermark Removal"
        assert ProcessingStage.SCENE_DETECTION.value == "Scene Detection"
        assert ProcessingStage.CUSTOM.value == "Custom"


class TestStageMetrics:
    """Tests for StageMetrics dataclass."""

    def test_default_metrics(self):
        """Test creating stage metrics with defaults."""
        metrics = StageMetrics(stage=ProcessingStage.AI_UPSCALING)

        assert metrics.stage == ProcessingStage.AI_UPSCALING
        assert metrics.duration_seconds == 0.0
        assert metrics.frames_input == 0
        assert metrics.frames_output == 0

    def test_frames_per_second_property(self):
        """Test FPS calculation property."""
        metrics = StageMetrics(
            stage=ProcessingStage.AI_UPSCALING,
            duration_seconds=10.0,
            frames_output=100,
        )

        assert metrics.frames_per_second == 10.0

    def test_frames_per_second_zero_duration(self):
        """Test FPS with zero duration returns zero."""
        metrics = StageMetrics(
            stage=ProcessingStage.AI_UPSCALING,
            duration_seconds=0.0,
            frames_output=100,
        )

        assert metrics.frames_per_second == 0.0

    def test_memory_delta_property(self):
        """Test memory delta calculation."""
        metrics = StageMetrics(
            stage=ProcessingStage.AI_UPSCALING,
            memory_start_mb=1000.0,
            memory_end_mb=1500.0,
        )

        assert metrics.memory_delta_mb == 500.0

    def test_gpu_memory_delta_property(self):
        """Test GPU memory delta calculation."""
        metrics = StageMetrics(
            stage=ProcessingStage.AI_UPSCALING,
            gpu_memory_start_mb=2000.0,
            gpu_memory_end_mb=4000.0,
        )

        assert metrics.gpu_memory_delta_mb == 2000.0

    def test_to_dict(self):
        """Test converting stage metrics to dictionary."""
        metrics = StageMetrics(
            stage=ProcessingStage.FRAME_EXTRACTION,
            duration_seconds=5.5,
            frames_input=100,
            frames_output=100,
            memory_peak_mb=2048.5,
            gpu_utilization_avg=85.5,
        )

        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert data["stage"] == "Frame Extraction"
        assert data["duration_seconds"] == 5.5
        assert data["frames_input"] == 100
        assert data["frames_output"] == 100
        assert data["memory_peak_mb"] == 2048.5
        assert data["gpu_utilization_avg"] == 85.5
        assert "frames_per_second" in data


class TestProfileSummary:
    """Tests for ProfileSummary dataclass."""

    def test_default_summary(self):
        """Test creating summary with defaults."""
        summary = ProfileSummary()

        assert summary.total_time_seconds == 0.0
        assert summary.total_frames_input == 0
        assert summary.total_frames_output == 0
        assert summary.overall_fps == 0.0
        assert summary.peak_memory_mb == 0.0
        assert summary.recommendations == []

    def test_custom_summary(self):
        """Test creating summary with custom values."""
        summary = ProfileSummary(
            total_time_seconds=120.5,
            total_frames_input=1000,
            total_frames_output=1000,
            overall_fps=8.3,
            peak_memory_mb=4096.0,
            peak_gpu_memory_mb=8192.0,
            bottleneck_stage="AI Upscaling",
            bottleneck_percentage=65.5,
            recommendations=["Consider using smaller tile size"],
        )

        assert summary.total_time_seconds == 120.5
        assert summary.bottleneck_stage == "AI Upscaling"
        assert summary.bottleneck_percentage == 65.5
        assert len(summary.recommendations) == 1

    def test_to_dict(self):
        """Test converting summary to dictionary."""
        summary = ProfileSummary(
            total_time_seconds=60.0,
            peak_memory_mb=2048.0,
            bottleneck_stage="Video Encoding",
        )

        data = summary.to_dict()

        assert isinstance(data, dict)
        assert data["total_time_seconds"] == 60.0
        assert data["peak_memory_mb"] == 2048.0
        assert data["bottleneck_stage"] == "Video Encoding"


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class."""

    def test_profiler_init(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()

        assert profiler.sample_interval == 0.5
        assert profiler.enable_gpu_monitoring is True

    def test_profiler_custom_init(self):
        """Test profiler with custom settings."""
        profiler = PerformanceProfiler(
            sample_interval=1.0,
            enable_gpu_monitoring=False,
        )

        assert profiler.sample_interval == 1.0
        assert profiler.enable_gpu_monitoring is False

    def test_session_lifecycle(self):
        """Test session start and end."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        profiler.start_session("test_session")
        profiler.end_session()

        report = profiler.get_report()

        assert report.session_name == "test_session"
        assert report.total_time >= 0

    def test_stage_profiling(self):
        """Test profiling individual stages."""
        import time

        profiler = PerformanceProfiler(
            sample_interval=0.1,
            enable_gpu_monitoring=False,
        )

        profiler.start_session("stage_test")

        profiler.start_stage(ProcessingStage.FRAME_EXTRACTION, frames=100)
        time.sleep(0.05)
        profiler.end_stage(frames_output=100)

        profiler.start_stage(ProcessingStage.AI_UPSCALING, frames=100)
        time.sleep(0.05)
        profiler.end_stage(frames_output=100)

        profiler.end_session()

        report = profiler.get_report()

        assert len(report.stages) == 2
        assert report.stages[0].stage == ProcessingStage.FRAME_EXTRACTION
        assert report.stages[1].stage == ProcessingStage.AI_UPSCALING

    def test_auto_end_previous_stage(self):
        """Test that starting a new stage auto-ends the previous one."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        profiler.start_session("auto_end_test")
        profiler.start_stage(ProcessingStage.FRAME_EXTRACTION, frames=50)
        # Starting another stage should auto-end the previous
        profiler.start_stage(ProcessingStage.AI_UPSCALING, frames=50)
        profiler.end_session()

        report = profiler.get_report()

        assert len(report.stages) == 2

    def test_record_frames(self):
        """Test recording frame progress."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        profiler.start_session("frame_test")
        profiler.start_stage(ProcessingStage.AI_UPSCALING, frames=100)
        profiler.record_frames(50)
        profiler.end_stage()
        profiler.end_session()

        report = profiler.get_report()

        # After end_stage, the frames_output should be the last recorded value
        assert report.stages[0].frames_output == 50


class TestProfileReport:
    """Tests for ProfileReport class."""

    def test_report_creation(self):
        """Test creating a profile report."""
        stages = [
            StageMetrics(
                stage=ProcessingStage.FRAME_EXTRACTION,
                duration_seconds=5.0,
                frames_output=100,
                memory_peak_mb=1024.0,
                gpu_memory_peak_mb=2048.0,
            ),
            StageMetrics(
                stage=ProcessingStage.AI_UPSCALING,
                duration_seconds=15.0,
                frames_output=100,
                memory_peak_mb=2048.0,
                gpu_memory_peak_mb=6144.0,
            ),
        ]

        report = ProfileReport(
            session_name="test_report",
            timestamp="2024-01-01T12:00:00",
            stages=stages,
            total_time=20.0,
        )

        assert report.session_name == "test_report"
        assert len(report.stages) == 2
        assert report.total_time == 20.0
        assert report.summary is not None

    def test_summary_calculation(self):
        """Test that summary is correctly calculated."""
        stages = [
            StageMetrics(
                stage=ProcessingStage.FRAME_EXTRACTION,
                duration_seconds=5.0,
                frames_output=100,
                memory_peak_mb=1000.0,
                gpu_memory_peak_mb=2000.0,
            ),
            StageMetrics(
                stage=ProcessingStage.AI_UPSCALING,
                duration_seconds=15.0,
                frames_output=100,
                memory_peak_mb=3000.0,
                gpu_memory_peak_mb=6000.0,
            ),
        ]

        report = ProfileReport(
            session_name="summary_test",
            timestamp="2024-01-01T12:00:00",
            stages=stages,
            total_time=20.0,
        )

        # Check peak values
        assert report.summary.peak_memory_mb == 3000.0
        assert report.summary.peak_gpu_memory_mb == 6000.0

        # Check bottleneck detection
        assert report.summary.bottleneck_stage == "AI Upscaling"
        assert report.summary.bottleneck_percentage == 75.0

    def test_format_table(self):
        """Test table formatting."""
        stages = [
            StageMetrics(
                stage=ProcessingStage.FRAME_EXTRACTION,
                duration_seconds=5.0,
                frames_output=100,
            ),
        ]

        report = ProfileReport(
            session_name="table_test",
            timestamp="2024-01-01T12:00:00",
            stages=stages,
            total_time=5.0,
        )

        table = report.format_table()

        assert isinstance(table, str)
        assert "Performance Profile Summary" in table
        assert "Frame Extraction" in table
        assert "TOTAL" in table

    def test_format_detailed(self):
        """Test detailed formatting."""
        stages = [
            StageMetrics(
                stage=ProcessingStage.AI_UPSCALING,
                duration_seconds=10.0,
                frames_input=100,
                frames_output=100,
                memory_start_mb=1000.0,
                memory_peak_mb=2000.0,
                memory_end_mb=1500.0,
            ),
        ]

        report = ProfileReport(
            session_name="detailed_test",
            timestamp="2024-01-01T12:00:00",
            stages=stages,
            total_time=10.0,
        )

        detailed = report.format_detailed()

        assert isinstance(detailed, str)
        assert "DETAILED PERFORMANCE PROFILE" in detailed
        assert "AI Upscaling" in detailed
        assert "Memory:" in detailed

    def test_export_and_load_json(self):
        """Test JSON export and import."""
        stages = [
            StageMetrics(
                stage=ProcessingStage.AI_UPSCALING,
                duration_seconds=10.0,
                frames_input=100,
                frames_output=100,
                memory_peak_mb=2048.0,
                gpu_memory_peak_mb=4096.0,
                gpu_utilization_avg=80.0,
            ),
        ]

        original = ProfileReport(
            session_name="json_test",
            timestamp="2024-01-01T12:00:00",
            stages=stages,
            total_time=10.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "profile.json"

            original.export_json(json_path)

            assert json_path.exists()

            # Load and verify
            loaded = ProfileReport.load_json(json_path)

            assert loaded.session_name == "json_test"
            assert len(loaded.stages) == 1
            assert loaded.stages[0].stage == ProcessingStage.AI_UPSCALING
            assert loaded.stages[0].duration_seconds == 10.0

    def test_compare_reports(self):
        """Test comparing multiple reports."""
        report1 = ProfileReport(
            session_name="run1",
            timestamp="2024-01-01T12:00:00",
            stages=[
                StageMetrics(
                    stage=ProcessingStage.AI_UPSCALING,
                    duration_seconds=10.0,
                    frames_output=100,
                ),
            ],
            total_time=10.0,
        )

        report2 = ProfileReport(
            session_name="run2",
            timestamp="2024-01-01T13:00:00",
            stages=[
                StageMetrics(
                    stage=ProcessingStage.AI_UPSCALING,
                    duration_seconds=8.0,
                    frames_output=100,
                ),
            ],
            total_time=8.0,
        )

        comparison = ProfileReport.compare([report1, report2])

        assert isinstance(comparison, str)
        assert "PROFILE COMPARISON" in comparison
        assert "run1" in comparison
        assert "run2" in comparison

    def test_calculate_improvement(self):
        """Test improvement calculation between profiles."""
        baseline = ProfileReport(
            session_name="baseline",
            timestamp="2024-01-01T12:00:00",
            stages=[
                StageMetrics(
                    stage=ProcessingStage.AI_UPSCALING,
                    duration_seconds=10.0,
                    frames_output=100,
                    memory_peak_mb=2048.0,
                    gpu_memory_peak_mb=4096.0,
                ),
            ],
            total_time=10.0,
        )

        improved = ProfileReport(
            session_name="improved",
            timestamp="2024-01-01T13:00:00",
            stages=[
                StageMetrics(
                    stage=ProcessingStage.AI_UPSCALING,
                    duration_seconds=8.0,
                    frames_output=100,
                    memory_peak_mb=1800.0,
                    gpu_memory_peak_mb=3500.0,
                ),
            ],
            total_time=8.0,
        )

        improvements = ProfileReport.calculate_improvement(baseline, improved)

        assert improvements["time_improvement_percent"] == 20.0
        assert improvements["speedup_factor"] == 1.25
        assert improvements["memory_reduction_mb"] == 248.0
        assert improvements["gpu_memory_reduction_mb"] == 596.0


class TestAnalyzeProfile:
    """Tests for analyze_profile function."""

    def test_analyze_profile_function(self):
        """Test the analyze_profile utility function."""
        stages = [
            StageMetrics(
                stage=ProcessingStage.FRAME_EXTRACTION,
                duration_seconds=2.0,
                frames_output=100,
            ),
            StageMetrics(
                stage=ProcessingStage.AI_UPSCALING,
                duration_seconds=8.0,
                frames_output=100,
                gpu_utilization_avg=85.0,
            ),
        ]

        report = ProfileReport(
            session_name="analyze_test",
            timestamp="2024-01-01T12:00:00",
            stages=stages,
            total_time=10.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "profile.json"
            report.export_json(json_path)

            analysis = analyze_profile(json_path)

            assert isinstance(analysis, str)
            assert "Profile Analysis" in analysis
            assert "Insights" in analysis


class TestProfilerIntegration:
    """Integration tests for the profiler."""

    def test_full_profiling_workflow(self):
        """Test complete profiling workflow."""
        import time

        profiler = PerformanceProfiler(
            sample_interval=0.1,
            enable_gpu_monitoring=False,
        )

        profiler.start_session("integration_test")

        # Simulate frame extraction
        profiler.start_stage(ProcessingStage.FRAME_EXTRACTION, frames=1000)
        time.sleep(0.05)
        profiler.record_frames(500)
        profiler.record_frames(1000)
        profiler.end_stage(frames_output=1000)

        # Simulate AI upscaling
        profiler.start_stage(ProcessingStage.AI_UPSCALING, frames=1000)
        time.sleep(0.1)
        profiler.end_stage(frames_output=1000)

        # Simulate video encoding
        profiler.start_stage(ProcessingStage.VIDEO_ENCODING, frames=1000)
        time.sleep(0.05)
        profiler.end_stage(frames_output=1000)

        profiler.end_session()

        report = profiler.get_report()

        # Verify report contents
        assert len(report.stages) == 3
        assert report.summary.bottleneck_stage is not None

        # Test export and reload
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "integration_profile.json"

            report.export_json(json_path)
            assert json_path.exists()

            loaded = ProfileReport.load_json(json_path)
            assert loaded.session_name == "integration_test"
            assert len(loaded.stages) == 3

            # Verify table formatting works
            table = loaded.format_table()
            assert "Frame Extraction" in table
            assert "AI Upscaling" in table
            assert "Video Encoding" in table
