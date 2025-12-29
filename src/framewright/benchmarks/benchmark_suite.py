"""FrameWright Benchmarking Suite.

Comprehensive benchmarking for video restoration pipeline performance testing,
including processing speed, memory usage, GPU utilization, and quality metrics.
"""

import csv
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks that can be run."""
    UPSCALE_720_TO_1080 = "720p_to_1080p"
    UPSCALE_1080_TO_4K = "1080p_to_4k"
    INTERPOLATION_24_TO_60 = "24fps_to_60fps"
    COMBINED_UPSCALE_INTERPOLATION = "upscale_interpolation"
    CUSTOM = "custom"


class DeviceType(Enum):
    """Device types for benchmark comparison."""
    GPU = "gpu"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        name: Human-readable name for this benchmark
        benchmark_type: Type of benchmark to run
        input_resolution: Input video resolution (width, height)
        output_resolution: Expected output resolution (width, height)
        input_fps: Input frame rate
        output_fps: Output frame rate (for interpolation tests)
        frame_count: Number of frames to process
        scale_factor: Upscaling factor (2 or 4)
        model_name: Real-ESRGAN model to use
        enable_interpolation: Whether to include RIFE interpolation
        device: Device to use for processing
        warmup_frames: Number of frames to process before timing
        iterations: Number of times to run the benchmark
    """
    name: str
    benchmark_type: BenchmarkType = BenchmarkType.CUSTOM
    input_resolution: Tuple[int, int] = (1280, 720)
    output_resolution: Tuple[int, int] = (1920, 1080)
    input_fps: float = 24.0
    output_fps: float = 24.0
    frame_count: int = 100
    scale_factor: int = 2
    model_name: str = "realesrgan-x2plus"
    enable_interpolation: bool = False
    device: DeviceType = DeviceType.AUTO
    warmup_frames: int = 5
    iterations: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "benchmark_type": self.benchmark_type.value,
            "input_resolution": list(self.input_resolution),
            "output_resolution": list(self.output_resolution),
            "input_fps": self.input_fps,
            "output_fps": self.output_fps,
            "frame_count": self.frame_count,
            "scale_factor": self.scale_factor,
            "model_name": self.model_name,
            "enable_interpolation": self.enable_interpolation,
            "device": self.device.value,
            "warmup_frames": self.warmup_frames,
            "iterations": self.iterations,
        }


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run.

    Attributes:
        processing_time_seconds: Total processing time in seconds
        frames_processed: Number of frames processed
        fps_input: Input frames processed per second
        fps_output: Output frames generated per second
        throughput_mb_per_second: Data throughput in MB/s
        memory_peak_mb: Peak memory usage in MB
        memory_average_mb: Average memory usage in MB
        gpu_utilization_percent: Average GPU utilization percentage
        gpu_memory_peak_mb: Peak GPU memory usage in MB
        gpu_memory_average_mb: Average GPU memory usage in MB
        psnr: Peak Signal-to-Noise Ratio (if reference available)
        ssim: Structural Similarity Index (if reference available)
        stage_timings: Per-stage timing breakdown
        cpu_utilization_percent: Average CPU utilization
        temperature_peak_celsius: Peak temperature during processing
    """
    processing_time_seconds: float = 0.0
    frames_processed: int = 0
    fps_input: float = 0.0
    fps_output: float = 0.0
    throughput_mb_per_second: float = 0.0
    memory_peak_mb: float = 0.0
    memory_average_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_average_mb: float = 0.0
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    cpu_utilization_percent: float = 0.0
    temperature_peak_celsius: Optional[float] = None

    @property
    def frames_per_second(self) -> float:
        """Calculate frames processed per second."""
        if self.processing_time_seconds > 0:
            return self.frames_processed / self.processing_time_seconds
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "frames_processed": self.frames_processed,
            "fps_input": round(self.fps_input, 2),
            "fps_output": round(self.fps_output, 2),
            "throughput_mb_per_second": round(self.throughput_mb_per_second, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 1),
            "memory_average_mb": round(self.memory_average_mb, 1),
            "gpu_utilization_percent": round(self.gpu_utilization_percent, 1),
            "gpu_memory_peak_mb": round(self.gpu_memory_peak_mb, 1),
            "gpu_memory_average_mb": round(self.gpu_memory_average_mb, 1),
            "psnr": round(self.psnr, 2) if self.psnr else None,
            "ssim": round(self.ssim, 4) if self.ssim else None,
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "cpu_utilization_percent": round(self.cpu_utilization_percent, 1),
            "temperature_peak_celsius": self.temperature_peak_celsius,
            "frames_per_second": round(self.frames_per_second, 2),
        }


@dataclass
class BenchmarkResult:
    """Complete result from a benchmark run.

    Attributes:
        config: Benchmark configuration used
        metrics: Metrics collected during the run
        timestamp: When the benchmark was run
        system_info: System information at time of benchmark
        success: Whether the benchmark completed successfully
        error_message: Error message if benchmark failed
        iteration_results: Results from each iteration
    """
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_info: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    iteration_results: List[BenchmarkMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "success": self.success,
            "error_message": self.error_message,
            "iteration_results": [m.to_dict() for m in self.iteration_results],
        }


class SystemProfiler:
    """Collects system information for benchmark context."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Collect comprehensive system information.

        Returns:
            Dictionary containing system specs
        """
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        }

        # CPU info
        try:
            import multiprocessing
            info["cpu_count"] = multiprocessing.cpu_count()
        except Exception:
            info["cpu_count"] = os.cpu_count() or 0

        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            info["memory_total_gb"] = None
            info["memory_available_gb"] = None

        # GPU info
        gpu_info = SystemProfiler._get_gpu_info()
        if gpu_info:
            info["gpu"] = gpu_info

        # FrameWright dependencies
        info["dependencies"] = SystemProfiler._get_dependency_versions()

        return info

    @staticmethod
    def _get_gpu_info() -> Optional[Dict[str, Any]]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,cuda_version",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(", ")]
                return {
                    "name": parts[0] if len(parts) > 0 else "Unknown",
                    "memory_total_mb": int(parts[1]) if len(parts) > 1 else 0,
                    "driver_version": parts[2] if len(parts) > 2 else "Unknown",
                    "cuda_version": parts[3] if len(parts) > 3 else "Unknown",
                }
        except Exception:
            pass
        return None

    @staticmethod
    def _get_dependency_versions() -> Dict[str, str]:
        """Get versions of key dependencies."""
        versions = {}

        # Check for key tools
        tools = {
            "ffmpeg": ["ffmpeg", "-version"],
            "realesrgan": ["realesrgan-ncnn-vulkan", "-h"],
            "rife": ["rife-ncnn-vulkan", "-h"],
        }

        for tool, cmd in tools.items():
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Try to extract version from output
                    first_line = result.stdout.split("\n")[0] if result.stdout else ""
                    versions[tool] = first_line[:100] if first_line else "available"
                else:
                    versions[tool] = "not found"
            except Exception:
                versions[tool] = "not found"

        return versions


class ResourceMonitor:
    """Monitors system resources during benchmark execution."""

    def __init__(self, sample_interval: float = 0.5):
        """Initialize resource monitor.

        Args:
            sample_interval: Seconds between resource samples
        """
        self.sample_interval = sample_interval
        self.samples: List[Dict[str, float]] = []
        self._monitoring = False
        self._monitor_thread = None

    def start(self) -> None:
        """Start resource monitoring in background thread."""
        import threading

        self._monitoring = True
        self.samples = []

        def monitor_loop():
            while self._monitoring:
                sample = self._collect_sample()
                if sample:
                    self.samples.append(sample)
                time.sleep(self.sample_interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _collect_sample(self) -> Optional[Dict[str, float]]:
        """Collect a single resource sample."""
        sample = {"timestamp": time.time()}

        # CPU and memory using psutil
        try:
            import psutil
            sample["cpu_percent"] = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            sample["memory_used_mb"] = mem.used / (1024**2)
            sample["memory_percent"] = mem.percent
        except ImportError:
            sample["cpu_percent"] = 0.0
            sample["memory_used_mb"] = 0.0
            sample["memory_percent"] = 0.0

        # GPU metrics
        gpu_sample = self._collect_gpu_sample()
        if gpu_sample:
            sample.update(gpu_sample)

        return sample

    def _collect_gpu_sample(self) -> Optional[Dict[str, float]]:
        """Collect GPU metrics sample."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(", ")]
                return {
                    "gpu_memory_used_mb": float(parts[0]) if len(parts) > 0 else 0,
                    "gpu_memory_total_mb": float(parts[1]) if len(parts) > 1 else 0,
                    "gpu_utilization_percent": float(parts[2]) if len(parts) > 2 and parts[2] != "[N/A]" else 0,
                    "gpu_temperature_c": float(parts[3]) if len(parts) > 3 and parts[3] != "[N/A]" else 0,
                }
        except Exception:
            pass
        return None

    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics from collected samples.

        Returns:
            Dictionary with min, max, avg for each metric
        """
        if not self.samples:
            return {}

        stats = {}

        # Define metrics to aggregate
        metrics = [
            "cpu_percent",
            "memory_used_mb",
            "gpu_memory_used_mb",
            "gpu_utilization_percent",
            "gpu_temperature_c",
        ]

        for metric in metrics:
            values = [s.get(metric, 0) for s in self.samples if metric in s]
            if values:
                stats[f"{metric}_min"] = min(values)
                stats[f"{metric}_max"] = max(values)
                stats[f"{metric}_avg"] = sum(values) / len(values)

        return stats


class TestVideoGenerator:
    """Generates synthetic test videos for benchmarking."""

    @staticmethod
    def generate_test_video(
        output_path: Path,
        resolution: Tuple[int, int],
        fps: float,
        duration_seconds: float,
        pattern: str = "testsrc2",
    ) -> bool:
        """Generate a synthetic test video using FFmpeg.

        Args:
            output_path: Path for output video
            resolution: Video resolution (width, height)
            fps: Frame rate
            duration_seconds: Video duration
            pattern: FFmpeg test pattern source

        Returns:
            True if generation succeeded
        """
        width, height = resolution

        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"{pattern}=size={width}x{height}:rate={fps}",
            "-t", str(duration_seconds),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-y",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0 and output_path.exists()
        except Exception as e:
            logger.error(f"Failed to generate test video: {e}")
            return False

    @staticmethod
    def generate_frame_sequence(
        output_dir: Path,
        resolution: Tuple[int, int],
        frame_count: int,
        pattern: str = "gradient",
    ) -> int:
        """Generate a sequence of test frames.

        Args:
            output_dir: Directory for output frames
            resolution: Frame resolution (width, height)
            frame_count: Number of frames to generate
            pattern: Pattern type ("gradient", "noise", "color_bars")

        Returns:
            Number of frames generated
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        width, height = resolution

        # Calculate duration to get desired frame count at 1 fps
        duration = frame_count

        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"testsrc2=size={width}x{height}:rate=1",
            "-frames:v", str(frame_count),
            "-y",
            str(output_dir / "frame_%08d.png")
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                return len(list(output_dir.glob("frame_*.png")))
        except Exception as e:
            logger.error(f"Failed to generate frame sequence: {e}")

        return 0


class QualityAnalyzer:
    """Analyzes output quality using PSNR and SSIM metrics."""

    @staticmethod
    def calculate_psnr(
        reference_path: Path,
        processed_path: Path,
    ) -> Optional[float]:
        """Calculate PSNR between reference and processed video/image.

        Args:
            reference_path: Path to reference file
            processed_path: Path to processed file

        Returns:
            PSNR value in dB, or None if calculation failed
        """
        cmd = [
            "ffmpeg",
            "-i", str(reference_path),
            "-i", str(processed_path),
            "-lavfi", "psnr=stats_file=-",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse PSNR from stderr
            import re
            match = re.search(r'average:(\d+\.?\d*)', result.stderr)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")

        return None

    @staticmethod
    def calculate_ssim(
        reference_path: Path,
        processed_path: Path,
    ) -> Optional[float]:
        """Calculate SSIM between reference and processed video/image.

        Args:
            reference_path: Path to reference file
            processed_path: Path to processed file

        Returns:
            SSIM value (0-1), or None if calculation failed
        """
        cmd = [
            "ffmpeg",
            "-i", str(reference_path),
            "-i", str(processed_path),
            "-lavfi", "ssim=stats_file=-",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse SSIM from stderr
            import re
            match = re.search(r'All:(\d+\.?\d*)', result.stderr)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")

        return None

    @staticmethod
    def analyze_frame_quality(
        reference_dir: Path,
        processed_dir: Path,
        sample_count: int = 10,
    ) -> Dict[str, float]:
        """Analyze quality across a sample of frames.

        Args:
            reference_dir: Directory with reference frames
            processed_dir: Directory with processed frames
            sample_count: Number of frames to sample

        Returns:
            Dictionary with average PSNR and SSIM
        """
        ref_frames = sorted(reference_dir.glob("*.png"))
        proc_frames = sorted(processed_dir.glob("*.png"))

        if not ref_frames or not proc_frames:
            return {"psnr": None, "ssim": None}

        # Select evenly spaced samples
        step = max(1, len(ref_frames) // sample_count)
        sample_indices = range(0, len(ref_frames), step)[:sample_count]

        psnr_values = []
        ssim_values = []

        for i in sample_indices:
            if i < len(ref_frames) and i < len(proc_frames):
                psnr = QualityAnalyzer.calculate_psnr(ref_frames[i], proc_frames[i])
                ssim = QualityAnalyzer.calculate_ssim(ref_frames[i], proc_frames[i])

                if psnr is not None:
                    psnr_values.append(psnr)
                if ssim is not None:
                    ssim_values.append(ssim)

        return {
            "psnr": sum(psnr_values) / len(psnr_values) if psnr_values else None,
            "ssim": sum(ssim_values) / len(ssim_values) if ssim_values else None,
        }


class BenchmarkRunner:
    """Runs standardized performance benchmarks.

    Example:
        >>> runner = BenchmarkRunner()
        >>> config = BenchmarkConfig(
        ...     name="720p_upscale_test",
        ...     input_resolution=(1280, 720),
        ...     scale_factor=2,
        ...     frame_count=100,
        ... )
        >>> result = runner.run_benchmark(config)
        >>> print(f"FPS: {result.metrics.frames_per_second:.2f}")
    """

    def __init__(
        self,
        work_dir: Optional[Path] = None,
        cleanup: bool = True,
    ):
        """Initialize benchmark runner.

        Args:
            work_dir: Working directory for benchmark files
            cleanup: Whether to clean up after benchmarks
        """
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="framewright_bench_"))

        self.work_dir = Path(work_dir)
        self.cleanup = cleanup
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.resource_monitor = ResourceMonitor()
        self.system_info = SystemProfiler.get_system_info()

        logger.info(f"BenchmarkRunner initialized with work_dir: {self.work_dir}")

    def run_benchmark(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark with the given configuration.

        Args:
            config: Benchmark configuration
            progress_callback: Optional callback for progress updates

        Returns:
            BenchmarkResult with metrics and status
        """
        logger.info(f"Starting benchmark: {config.name}")

        iteration_results = []

        try:
            # Setup test data
            if progress_callback:
                progress_callback("setup", 0.0)

            test_dir = self.work_dir / config.name.replace(" ", "_")
            test_dir.mkdir(parents=True, exist_ok=True)

            frames_dir = test_dir / "input_frames"
            output_dir = test_dir / "output_frames"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate test frames
            frame_count = TestVideoGenerator.generate_frame_sequence(
                output_dir=frames_dir,
                resolution=config.input_resolution,
                frame_count=config.frame_count + config.warmup_frames,
            )

            if frame_count < config.frame_count:
                raise RuntimeError(f"Failed to generate test frames: got {frame_count}")

            if progress_callback:
                progress_callback("setup", 1.0)

            # Run iterations
            for iteration in range(config.iterations):
                if progress_callback:
                    progress = iteration / config.iterations
                    progress_callback("benchmark", progress)

                logger.info(f"Running iteration {iteration + 1}/{config.iterations}")

                # Clear output for fresh run
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Run benchmark iteration
                metrics = self._run_iteration(
                    config=config,
                    frames_dir=frames_dir,
                    output_dir=output_dir,
                )

                iteration_results.append(metrics)

            # Calculate aggregate metrics
            aggregate_metrics = self._aggregate_metrics(iteration_results)

            if progress_callback:
                progress_callback("complete", 1.0)

            return BenchmarkResult(
                config=config,
                metrics=aggregate_metrics,
                system_info=self.system_info,
                success=True,
                iteration_results=iteration_results,
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                config=config,
                metrics=BenchmarkMetrics(),
                system_info=self.system_info,
                success=False,
                error_message=str(e),
            )

        finally:
            if self.cleanup:
                try:
                    shutil.rmtree(test_dir, ignore_errors=True)
                except Exception:
                    pass

    def _run_iteration(
        self,
        config: BenchmarkConfig,
        frames_dir: Path,
        output_dir: Path,
    ) -> BenchmarkMetrics:
        """Run a single benchmark iteration.

        Args:
            config: Benchmark configuration
            frames_dir: Directory containing input frames
            output_dir: Directory for output frames

        Returns:
            Metrics from this iteration
        """
        metrics = BenchmarkMetrics()
        stage_timings = {}

        # Start resource monitoring
        self.resource_monitor.start()

        try:
            frames = sorted(frames_dir.glob("frame_*.png"))

            # Warmup phase - process but don't count
            warmup_frames = frames[:config.warmup_frames]
            test_frames = frames[config.warmup_frames:config.warmup_frames + config.frame_count]

            # Warmup
            warmup_dir = output_dir / "warmup"
            warmup_dir.mkdir(parents=True, exist_ok=True)

            for frame in warmup_frames:
                self._process_frame(
                    input_path=frame,
                    output_path=warmup_dir / frame.name,
                    config=config,
                )

            shutil.rmtree(warmup_dir, ignore_errors=True)

            # Timed benchmark phase
            start_time = time.time()

            # Enhancement stage
            stage_start = time.time()
            for frame in test_frames:
                self._process_frame(
                    input_path=frame,
                    output_path=output_dir / frame.name,
                    config=config,
                )
            stage_timings["enhancement"] = time.time() - stage_start

            # Interpolation stage (if enabled)
            if config.enable_interpolation:
                stage_start = time.time()
                interpolated_dir = output_dir / "interpolated"
                self._run_interpolation(
                    input_dir=output_dir,
                    output_dir=interpolated_dir,
                    source_fps=config.input_fps,
                    target_fps=config.output_fps,
                )
                stage_timings["interpolation"] = time.time() - stage_start

            end_time = time.time()

            # Collect metrics
            metrics.processing_time_seconds = end_time - start_time
            metrics.frames_processed = len(test_frames)
            metrics.stage_timings = stage_timings

            # Calculate throughput
            if metrics.processing_time_seconds > 0:
                metrics.fps_input = metrics.frames_processed / metrics.processing_time_seconds

                # Calculate output FPS (accounting for interpolation)
                if config.enable_interpolation:
                    interpolation_factor = config.output_fps / config.input_fps
                    metrics.fps_output = metrics.fps_input * interpolation_factor
                else:
                    metrics.fps_output = metrics.fps_input

            # Calculate data throughput
            input_frame_size_mb = (config.input_resolution[0] * config.input_resolution[1] * 3) / (1024**2)
            output_frame_size_mb = (config.output_resolution[0] * config.output_resolution[1] * 3) / (1024**2)
            total_data_mb = (input_frame_size_mb + output_frame_size_mb) * metrics.frames_processed

            if metrics.processing_time_seconds > 0:
                metrics.throughput_mb_per_second = total_data_mb / metrics.processing_time_seconds

        finally:
            # Stop monitoring and collect statistics
            self.resource_monitor.stop()

        # Get resource statistics
        resource_stats = self.resource_monitor.get_statistics()

        metrics.memory_peak_mb = resource_stats.get("memory_used_mb_max", 0)
        metrics.memory_average_mb = resource_stats.get("memory_used_mb_avg", 0)
        metrics.gpu_memory_peak_mb = resource_stats.get("gpu_memory_used_mb_max", 0)
        metrics.gpu_memory_average_mb = resource_stats.get("gpu_memory_used_mb_avg", 0)
        metrics.gpu_utilization_percent = resource_stats.get("gpu_utilization_percent_avg", 0)
        metrics.cpu_utilization_percent = resource_stats.get("cpu_percent_avg", 0)
        metrics.temperature_peak_celsius = resource_stats.get("gpu_temperature_c_max")

        return metrics

    def _process_frame(
        self,
        input_path: Path,
        output_path: Path,
        config: BenchmarkConfig,
    ) -> bool:
        """Process a single frame with Real-ESRGAN.

        Args:
            input_path: Input frame path
            output_path: Output frame path
            config: Benchmark configuration

        Returns:
            True if processing succeeded
        """
        cmd = [
            "realesrgan-ncnn-vulkan",
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", config.model_name,
            "-s", str(config.scale_factor),
            "-f", "png"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0 and output_path.exists()
        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")
            return False

    def _run_interpolation(
        self,
        input_dir: Path,
        output_dir: Path,
        source_fps: float,
        target_fps: float,
    ) -> bool:
        """Run RIFE frame interpolation.

        Args:
            input_dir: Directory with input frames
            output_dir: Directory for interpolated frames
            source_fps: Source frame rate
            target_fps: Target frame rate

        Returns:
            True if interpolation succeeded
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate required interpolation factor
        factor = target_fps / source_fps

        # For RIFE, we typically need 2x interpolations
        # 24 -> 60 requires roughly 2.5x, achieved via multiple passes
        cmd = [
            "rife-ncnn-vulkan",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-m", "rife-v4.6",
            "-f", "frame_%08d.png",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return False

    def _aggregate_metrics(
        self,
        iteration_results: List[BenchmarkMetrics],
    ) -> BenchmarkMetrics:
        """Aggregate metrics from multiple iterations.

        Args:
            iteration_results: List of metrics from each iteration

        Returns:
            Aggregated metrics (averages)
        """
        if not iteration_results:
            return BenchmarkMetrics()

        # Calculate averages for numeric fields
        n = len(iteration_results)

        aggregated = BenchmarkMetrics(
            processing_time_seconds=sum(m.processing_time_seconds for m in iteration_results) / n,
            frames_processed=sum(m.frames_processed for m in iteration_results) // n,
            fps_input=sum(m.fps_input for m in iteration_results) / n,
            fps_output=sum(m.fps_output for m in iteration_results) / n,
            throughput_mb_per_second=sum(m.throughput_mb_per_second for m in iteration_results) / n,
            memory_peak_mb=max(m.memory_peak_mb for m in iteration_results),
            memory_average_mb=sum(m.memory_average_mb for m in iteration_results) / n,
            gpu_utilization_percent=sum(m.gpu_utilization_percent for m in iteration_results) / n,
            gpu_memory_peak_mb=max(m.gpu_memory_peak_mb for m in iteration_results),
            gpu_memory_average_mb=sum(m.gpu_memory_average_mb for m in iteration_results) / n,
            cpu_utilization_percent=sum(m.cpu_utilization_percent for m in iteration_results) / n,
        )

        # Aggregate stage timings
        stage_keys = set()
        for m in iteration_results:
            stage_keys.update(m.stage_timings.keys())

        for key in stage_keys:
            values = [m.stage_timings.get(key, 0) for m in iteration_results]
            aggregated.stage_timings[key] = sum(values) / len(values)

        # Peak temperature
        temps = [m.temperature_peak_celsius for m in iteration_results if m.temperature_peak_celsius is not None]
        if temps:
            aggregated.temperature_peak_celsius = max(temps)

        return aggregated

    def run_comparison(
        self,
        config: BenchmarkConfig,
        devices: List[DeviceType] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmark on multiple device configurations for comparison.

        Args:
            config: Base benchmark configuration
            devices: List of device types to compare

        Returns:
            Dictionary mapping device name to benchmark result
        """
        if devices is None:
            devices = [DeviceType.GPU, DeviceType.CPU]

        results = {}

        for device in devices:
            # Create config variant for this device
            device_config = BenchmarkConfig(
                name=f"{config.name}_{device.value}",
                benchmark_type=config.benchmark_type,
                input_resolution=config.input_resolution,
                output_resolution=config.output_resolution,
                input_fps=config.input_fps,
                output_fps=config.output_fps,
                frame_count=config.frame_count,
                scale_factor=config.scale_factor,
                model_name=config.model_name,
                enable_interpolation=config.enable_interpolation,
                device=device,
                warmup_frames=config.warmup_frames,
                iterations=config.iterations,
            )

            result = self.run_benchmark(device_config)
            results[device.value] = result

        return results

    def __del__(self):
        """Cleanup on destruction."""
        if self.cleanup and self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir, ignore_errors=True)
            except Exception:
                pass


class StandardTestSuite:
    """Predefined test cases for common benchmarking scenarios.

    Example:
        >>> suite = StandardTestSuite()
        >>> results = suite.run_standard_suite(runner)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.metrics.frames_per_second:.2f} FPS")
    """

    # Standard test configurations
    STANDARD_TESTS = {
        "720p_to_1080p": BenchmarkConfig(
            name="720p to 1080p Upscale",
            benchmark_type=BenchmarkType.UPSCALE_720_TO_1080,
            input_resolution=(1280, 720),
            output_resolution=(1920, 1080),
            scale_factor=2,
            model_name="realesrgan-x2plus",
            frame_count=100,
            iterations=3,
        ),
        "1080p_to_4k": BenchmarkConfig(
            name="1080p to 4K Upscale",
            benchmark_type=BenchmarkType.UPSCALE_1080_TO_4K,
            input_resolution=(1920, 1080),
            output_resolution=(3840, 2160),
            scale_factor=2,
            model_name="realesrgan-x2plus",
            frame_count=50,
            iterations=3,
        ),
        "interpolation_24_to_60": BenchmarkConfig(
            name="24fps to 60fps Interpolation",
            benchmark_type=BenchmarkType.INTERPOLATION_24_TO_60,
            input_resolution=(1920, 1080),
            output_resolution=(1920, 1080),
            input_fps=24.0,
            output_fps=60.0,
            scale_factor=1,
            enable_interpolation=True,
            frame_count=100,
            iterations=3,
        ),
        "combined_upscale_interpolation": BenchmarkConfig(
            name="Combined Upscale + Interpolation",
            benchmark_type=BenchmarkType.COMBINED_UPSCALE_INTERPOLATION,
            input_resolution=(1280, 720),
            output_resolution=(1920, 1080),
            input_fps=24.0,
            output_fps=60.0,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_interpolation=True,
            frame_count=50,
            iterations=3,
        ),
    }

    @classmethod
    def get_test_config(cls, test_name: str) -> Optional[BenchmarkConfig]:
        """Get a standard test configuration by name.

        Args:
            test_name: Name of the standard test

        Returns:
            BenchmarkConfig or None if not found
        """
        return cls.STANDARD_TESTS.get(test_name)

    @classmethod
    def list_tests(cls) -> List[str]:
        """List available standard tests.

        Returns:
            List of test names
        """
        return list(cls.STANDARD_TESTS.keys())

    @classmethod
    def run_standard_suite(
        cls,
        runner: BenchmarkRunner,
        tests: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run the standard benchmark suite.

        Args:
            runner: BenchmarkRunner instance
            tests: List of test names to run (None = all)
            progress_callback: Callback with (test_name, stage, progress)

        Returns:
            Dictionary mapping test name to result
        """
        if tests is None:
            tests = cls.list_tests()

        results = {}
        total_tests = len(tests)

        for i, test_name in enumerate(tests):
            config = cls.get_test_config(test_name)
            if config is None:
                logger.warning(f"Unknown test: {test_name}")
                continue

            logger.info(f"Running standard test [{i+1}/{total_tests}]: {test_name}")

            def test_progress(stage: str, progress: float):
                if progress_callback:
                    overall_progress = (i + progress) / total_tests
                    progress_callback(test_name, stage, overall_progress)

            result = runner.run_benchmark(config, progress_callback=test_progress)
            results[test_name] = result

        return results

    @classmethod
    def run_quick_benchmark(
        cls,
        runner: BenchmarkRunner,
    ) -> BenchmarkResult:
        """Run a quick benchmark for fast validation.

        Uses smaller frame count and single iteration.

        Args:
            runner: BenchmarkRunner instance

        Returns:
            BenchmarkResult from quick test
        """
        quick_config = BenchmarkConfig(
            name="Quick Benchmark",
            benchmark_type=BenchmarkType.UPSCALE_720_TO_1080,
            input_resolution=(640, 360),
            output_resolution=(1280, 720),
            scale_factor=2,
            model_name="realesrgan-x2plus",
            frame_count=20,
            warmup_frames=2,
            iterations=1,
        )

        return runner.run_benchmark(quick_config)


class BenchmarkReporter:
    """Generates benchmark reports in various formats.

    Example:
        >>> reporter = BenchmarkReporter()
        >>> reporter.generate_json_report(results, Path("benchmark_report.json"))
        >>> reporter.generate_csv_report(results, Path("benchmark_report.csv"))
    """

    @staticmethod
    def generate_json_report(
        results: Union[BenchmarkResult, Dict[str, BenchmarkResult], List[BenchmarkResult]],
        output_path: Path,
        pretty: bool = True,
    ) -> Path:
        """Generate a JSON benchmark report.

        Args:
            results: Single result, dict of results, or list of results
            output_path: Path for output file
            pretty: Whether to format JSON for readability

        Returns:
            Path to generated report
        """
        # Normalize results to list
        if isinstance(results, BenchmarkResult):
            results_list = [results]
        elif isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results

        report = {
            "generated_at": datetime.now().isoformat(),
            "benchmark_count": len(results_list),
            "results": [r.to_dict() for r in results_list],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if pretty:
                json.dump(report, f, indent=2)
            else:
                json.dump(report, f)

        logger.info(f"JSON report saved to: {output_path}")
        return output_path

    @staticmethod
    def generate_csv_report(
        results: Union[BenchmarkResult, Dict[str, BenchmarkResult], List[BenchmarkResult]],
        output_path: Path,
    ) -> Path:
        """Generate a CSV benchmark report.

        Args:
            results: Single result, dict of results, or list of results
            output_path: Path for output file

        Returns:
            Path to generated report
        """
        # Normalize results to list
        if isinstance(results, BenchmarkResult):
            results_list = [results]
        elif isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns
        fieldnames = [
            "name",
            "benchmark_type",
            "input_resolution",
            "output_resolution",
            "scale_factor",
            "frame_count",
            "processing_time_seconds",
            "frames_per_second",
            "throughput_mb_per_second",
            "memory_peak_mb",
            "gpu_memory_peak_mb",
            "gpu_utilization_percent",
            "success",
            "timestamp",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results_list:
                row = {
                    "name": result.config.name,
                    "benchmark_type": result.config.benchmark_type.value,
                    "input_resolution": f"{result.config.input_resolution[0]}x{result.config.input_resolution[1]}",
                    "output_resolution": f"{result.config.output_resolution[0]}x{result.config.output_resolution[1]}",
                    "scale_factor": result.config.scale_factor,
                    "frame_count": result.metrics.frames_processed,
                    "processing_time_seconds": round(result.metrics.processing_time_seconds, 3),
                    "frames_per_second": round(result.metrics.frames_per_second, 2),
                    "throughput_mb_per_second": round(result.metrics.throughput_mb_per_second, 2),
                    "memory_peak_mb": round(result.metrics.memory_peak_mb, 1),
                    "gpu_memory_peak_mb": round(result.metrics.gpu_memory_peak_mb, 1),
                    "gpu_utilization_percent": round(result.metrics.gpu_utilization_percent, 1),
                    "success": result.success,
                    "timestamp": result.timestamp,
                }
                writer.writerow(row)

        logger.info(f"CSV report saved to: {output_path}")
        return output_path

    @staticmethod
    def generate_comparison_table(
        results: Dict[str, BenchmarkResult],
    ) -> str:
        """Generate a text comparison table.

        Args:
            results: Dictionary of results to compare

        Returns:
            Formatted table as string
        """
        if not results:
            return "No results to compare."

        # Build header
        lines = [
            "=" * 80,
            "BENCHMARK COMPARISON",
            "=" * 80,
            "",
        ]

        # Build rows
        header_row = f"{'Benchmark':<30} {'FPS':>10} {'Time (s)':>10} {'GPU Mem':>10} {'Status':>10}"
        lines.append(header_row)
        lines.append("-" * 80)

        for name, result in results.items():
            fps = f"{result.metrics.frames_per_second:.2f}" if result.success else "N/A"
            time_s = f"{result.metrics.processing_time_seconds:.2f}" if result.success else "N/A"
            gpu_mem = f"{result.metrics.gpu_memory_peak_mb:.0f}MB" if result.success else "N/A"
            status = "PASS" if result.success else "FAIL"

            row = f"{name:<30} {fps:>10} {time_s:>10} {gpu_mem:>10} {status:>10}"
            lines.append(row)

        lines.append("-" * 80)
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_summary(
        results: Union[BenchmarkResult, Dict[str, BenchmarkResult], List[BenchmarkResult]],
    ) -> str:
        """Generate a human-readable summary.

        Args:
            results: Benchmark results

        Returns:
            Summary string
        """
        # Normalize results to list
        if isinstance(results, BenchmarkResult):
            results_list = [results]
        elif isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results

        successful = [r for r in results_list if r.success]
        failed = [r for r in results_list if not r.success]

        lines = [
            "=" * 60,
            "BENCHMARK SUMMARY",
            "=" * 60,
            "",
            f"Total benchmarks: {len(results_list)}",
            f"Successful: {len(successful)}",
            f"Failed: {len(failed)}",
            "",
        ]

        if successful:
            avg_fps = sum(r.metrics.frames_per_second for r in successful) / len(successful)
            max_fps = max(r.metrics.frames_per_second for r in successful)
            min_fps = min(r.metrics.frames_per_second for r in successful)

            lines.extend([
                "Performance Summary (successful runs):",
                f"  Average FPS: {avg_fps:.2f}",
                f"  Max FPS: {max_fps:.2f}",
                f"  Min FPS: {min_fps:.2f}",
                "",
            ])

            max_gpu_mem = max(r.metrics.gpu_memory_peak_mb for r in successful)
            avg_gpu_mem = sum(r.metrics.gpu_memory_peak_mb for r in successful) / len(successful)

            lines.extend([
                "Resource Usage:",
                f"  Peak GPU Memory: {max_gpu_mem:.0f} MB",
                f"  Average GPU Memory: {avg_gpu_mem:.0f} MB",
                "",
            ])

        if failed:
            lines.extend([
                "Failed benchmarks:",
            ])
            for r in failed:
                lines.append(f"  - {r.config.name}: {r.error_message or 'Unknown error'}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def load_historical_results(
        history_dir: Path,
    ) -> List[BenchmarkResult]:
        """Load historical benchmark results for comparison.

        Args:
            history_dir: Directory containing historical JSON reports

        Returns:
            List of historical BenchmarkResults
        """
        results = []

        for json_file in sorted(history_dir.glob("*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                for result_data in data.get("results", []):
                    # Reconstruct BenchmarkResult from dict
                    config = BenchmarkConfig(
                        name=result_data["config"]["name"],
                        benchmark_type=BenchmarkType(result_data["config"]["benchmark_type"]),
                        input_resolution=tuple(result_data["config"]["input_resolution"]),
                        output_resolution=tuple(result_data["config"]["output_resolution"]),
                        input_fps=result_data["config"]["input_fps"],
                        output_fps=result_data["config"]["output_fps"],
                        frame_count=result_data["config"]["frame_count"],
                        scale_factor=result_data["config"]["scale_factor"],
                        model_name=result_data["config"]["model_name"],
                        enable_interpolation=result_data["config"]["enable_interpolation"],
                        device=DeviceType(result_data["config"]["device"]),
                        warmup_frames=result_data["config"]["warmup_frames"],
                        iterations=result_data["config"]["iterations"],
                    )

                    metrics = BenchmarkMetrics(
                        processing_time_seconds=result_data["metrics"]["processing_time_seconds"],
                        frames_processed=result_data["metrics"]["frames_processed"],
                        fps_input=result_data["metrics"]["fps_input"],
                        fps_output=result_data["metrics"]["fps_output"],
                        throughput_mb_per_second=result_data["metrics"]["throughput_mb_per_second"],
                        memory_peak_mb=result_data["metrics"]["memory_peak_mb"],
                        memory_average_mb=result_data["metrics"]["memory_average_mb"],
                        gpu_utilization_percent=result_data["metrics"]["gpu_utilization_percent"],
                        gpu_memory_peak_mb=result_data["metrics"]["gpu_memory_peak_mb"],
                        gpu_memory_average_mb=result_data["metrics"]["gpu_memory_average_mb"],
                        psnr=result_data["metrics"].get("psnr"),
                        ssim=result_data["metrics"].get("ssim"),
                        stage_timings=result_data["metrics"].get("stage_timings", {}),
                        cpu_utilization_percent=result_data["metrics"].get("cpu_utilization_percent", 0),
                        temperature_peak_celsius=result_data["metrics"].get("temperature_peak_celsius"),
                    )

                    result = BenchmarkResult(
                        config=config,
                        metrics=metrics,
                        timestamp=result_data.get("timestamp", ""),
                        system_info=result_data.get("system_info", {}),
                        success=result_data.get("success", True),
                        error_message=result_data.get("error_message"),
                    )

                    results.append(result)

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return results

    @staticmethod
    def track_performance_over_time(
        current_result: BenchmarkResult,
        history_dir: Path,
    ) -> Dict[str, Any]:
        """Compare current result with historical performance.

        Args:
            current_result: Current benchmark result
            history_dir: Directory containing historical results

        Returns:
            Dictionary with trend analysis
        """
        historical = BenchmarkReporter.load_historical_results(history_dir)

        # Filter to matching benchmark types
        matching = [
            r for r in historical
            if r.config.benchmark_type == current_result.config.benchmark_type
            and r.success
        ]

        if not matching:
            return {
                "trend": "unknown",
                "historical_count": 0,
                "message": "No historical data available for comparison",
            }

        # Calculate statistics
        historical_fps = [r.metrics.frames_per_second for r in matching]
        avg_historical_fps = sum(historical_fps) / len(historical_fps)
        current_fps = current_result.metrics.frames_per_second

        # Determine trend
        if current_fps > avg_historical_fps * 1.05:
            trend = "improving"
            change_percent = ((current_fps - avg_historical_fps) / avg_historical_fps) * 100
        elif current_fps < avg_historical_fps * 0.95:
            trend = "degrading"
            change_percent = ((avg_historical_fps - current_fps) / avg_historical_fps) * 100
        else:
            trend = "stable"
            change_percent = 0

        return {
            "trend": trend,
            "historical_count": len(matching),
            "current_fps": round(current_fps, 2),
            "average_historical_fps": round(avg_historical_fps, 2),
            "change_percent": round(change_percent, 1),
            "message": f"Performance is {trend} ({change_percent:+.1f}% vs historical average)",
        }


def run_cli_benchmark(args) -> int:
    """Run benchmark from CLI arguments.

    Args:
        args: Parsed argparse arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    from ..cli import Colors, print_colored

    runner = BenchmarkRunner(cleanup=not args.keep_files if hasattr(args, 'keep_files') else True)

    try:
        if args.suite == "standard":
            print_colored("\nRunning standard benchmark suite...\n", Colors.OKBLUE)

            def progress(test_name, stage, progress):
                print(f"\r  [{test_name}] {stage}: {progress*100:.0f}%", end="", flush=True)

            results = StandardTestSuite.run_standard_suite(runner, progress_callback=progress)
            print("\n")

        elif args.suite == "quick":
            print_colored("\nRunning quick benchmark...\n", Colors.OKBLUE)
            result = StandardTestSuite.run_quick_benchmark(runner)
            results = {"quick": result}

        elif hasattr(args, 'video') and args.video:
            print_colored(f"\nBenchmarking video: {args.video}\n", Colors.OKBLUE)

            # Custom video benchmark
            config = BenchmarkConfig(
                name=f"Custom: {Path(args.video).name}",
                benchmark_type=BenchmarkType.CUSTOM,
                scale_factor=args.scale if hasattr(args, 'scale') else 2,
                frame_count=args.frames if hasattr(args, 'frames') else 100,
            )

            result = runner.run_benchmark(config)
            results = {"custom": result}

        else:
            print_colored("No benchmark specified. Use --suite or --video.", Colors.FAIL)
            return 1

        # Generate reports
        reporter = BenchmarkReporter

        if hasattr(args, 'report') and args.report:
            report_path = Path(args.report)
            if report_path.suffix == ".csv":
                reporter.generate_csv_report(results, report_path)
            else:
                reporter.generate_json_report(results, report_path)
            print_colored(f"\nReport saved to: {report_path}", Colors.OKGREEN)

        # Print summary
        summary = reporter.generate_summary(results)
        print(summary)

        if hasattr(args, 'compare') and args.compare:
            table = reporter.generate_comparison_table(results)
            print(table)

        # Check for failures
        failed = [r for r in results.values() if not r.success]
        if failed:
            print_colored(f"\n{len(failed)} benchmark(s) failed.", Colors.FAIL)
            return 1

        print_colored("\nAll benchmarks completed successfully.", Colors.OKGREEN)
        return 0

    except Exception as e:
        print_colored(f"\nBenchmark error: {e}", Colors.FAIL)
        logger.exception("Benchmark failed")
        return 1


def add_benchmark_parser(subparsers) -> None:
    """Add benchmark subcommand to CLI parser.

    Args:
        subparsers: argparse subparsers object
    """
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run performance benchmarks'
    )

    benchmark_parser.add_argument(
        '--suite',
        type=str,
        choices=['standard', 'quick'],
        default=None,
        help='Run a predefined benchmark suite'
    )

    benchmark_parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Benchmark with a specific video file'
    )

    benchmark_parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        choices=['gpu', 'cpu'],
        default=None,
        help='Compare performance across device types'
    )

    benchmark_parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Save benchmark report to file (JSON or CSV)'
    )

    benchmark_parser.add_argument(
        '--scale',
        type=int,
        choices=[2, 4],
        default=2,
        help='Upscaling factor for custom benchmarks'
    )

    benchmark_parser.add_argument(
        '--frames',
        type=int,
        default=100,
        help='Number of frames to process'
    )

    benchmark_parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of benchmark iterations'
    )

    benchmark_parser.add_argument(
        '--keep-files',
        action='store_true',
        help='Keep temporary benchmark files'
    )

    benchmark_parser.set_defaults(func=run_cli_benchmark)
