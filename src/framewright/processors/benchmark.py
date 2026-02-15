"""
Benchmark Mode - Test restoration speed with different settings.

Measures processing performance across various configurations to help
users find optimal settings for their hardware.
"""

import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import tempfile
import shutil

import cv2
import numpy as np


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    QUICK = "quick"  # ~30 seconds, basic metrics
    STANDARD = "standard"  # ~2 minutes, comprehensive
    THOROUGH = "thorough"  # ~5 minutes, all configurations


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    settings: Dict[str, Any]
    frames_processed: int
    total_time: float
    fps: float
    avg_frame_time_ms: float
    min_frame_time_ms: float
    max_frame_time_ms: float
    std_dev_ms: float
    vram_used_mb: float
    ram_used_mb: float
    gpu_temp_start: Optional[float] = None
    gpu_temp_end: Optional[float] = None
    thermal_throttled: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report with all runs."""
    hardware_info: Dict[str, Any]
    test_video_info: Dict[str, Any]
    results: List[BenchmarkResult]
    recommendations: List[str]
    timestamp: str
    duration_seconds: float

    def get_fastest(self) -> Optional[BenchmarkResult]:
        """Get the fastest configuration."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.fps)

    def get_most_stable(self) -> Optional[BenchmarkResult]:
        """Get the most consistent configuration (lowest std dev)."""
        valid = [r for r in self.results if r.std_dev_ms > 0]
        if not valid:
            return None
        return min(valid, key=lambda r: r.std_dev_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hardware_info": self.hardware_info,
            "test_video_info": self.test_video_info,
            "results": [
                {
                    "name": r.name,
                    "settings": r.settings,
                    "frames_processed": r.frames_processed,
                    "total_time": r.total_time,
                    "fps": r.fps,
                    "avg_frame_time_ms": r.avg_frame_time_ms,
                    "min_frame_time_ms": r.min_frame_time_ms,
                    "max_frame_time_ms": r.max_frame_time_ms,
                    "std_dev_ms": r.std_dev_ms,
                    "vram_used_mb": r.vram_used_mb,
                    "ram_used_mb": r.ram_used_mb,
                    "gpu_temp_start": r.gpu_temp_start,
                    "gpu_temp_end": r.gpu_temp_end,
                    "thermal_throttled": r.thermal_throttled,
                    "errors": r.errors
                }
                for r in self.results
            ],
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class RestorationBenchmark:
    """
    Benchmark restoration performance across different configurations.

    Tests various batch sizes, resolutions, and processing options
    to find optimal settings for the user's hardware.
    """

    # Default test configurations
    BATCH_SIZES = [1, 2, 4, 8, 16]
    TILE_SIZES = [256, 512, 768, 1024]

    def __init__(
        self,
        test_frames: int = 100,
        warmup_frames: int = 10,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize benchmark.

        Args:
            test_frames: Number of frames to process per test
            warmup_frames: Warmup frames before timing
            progress_callback: Called with (message, progress_0_to_1)
        """
        self.test_frames = test_frames
        self.warmup_frames = warmup_frames
        self.progress_callback = progress_callback

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Gather hardware information."""
        import platform

        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "gpu": "Unknown",
            "vram_total_mb": 0,
            "ram_total_mb": 0
        }

        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["vram_total_mb"] = props.total_memory / (1024 * 1024)
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass

        # Get RAM info
        try:
            import psutil
            info["ram_total_mb"] = psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            pass

        return info

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get test video information."""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "path": str(video_path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        cap.release()
        return info

    def _get_memory_usage(self) -> tuple:
        """Get current VRAM and RAM usage in MB."""
        vram_mb = 0
        ram_mb = 0

        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        try:
            import psutil
            ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            pass

        return vram_mb, ram_mb

    def _get_gpu_temp(self) -> Optional[float]:
        """Get current GPU temperature."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return float(temp)
        except Exception:
            return None

    def _run_single_benchmark(
        self,
        video_path: Path,
        name: str,
        settings: Dict[str, Any],
        processor_func: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
    ) -> BenchmarkResult:
        """
        Run a single benchmark configuration.

        Args:
            video_path: Path to test video
            name: Name for this benchmark
            settings: Settings dictionary
            processor_func: Function that processes a frame

        Returns:
            BenchmarkResult with timing data
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_times = []
        errors = []

        gpu_temp_start = self._get_gpu_temp()
        vram_start, ram_start = self._get_memory_usage()

        # Warmup
        for _ in range(self.warmup_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            try:
                processor_func(frame, settings)
            except Exception as e:
                errors.append(f"Warmup error: {e}")

        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Timed run
        total_start = time.perf_counter()
        frames_processed = 0

        for i in range(self.test_frames):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            frame_start = time.perf_counter()
            try:
                processor_func(frame, settings)
                frame_times.append((time.perf_counter() - frame_start) * 1000)
                frames_processed += 1
            except Exception as e:
                errors.append(f"Frame {i} error: {e}")

        total_time = time.perf_counter() - total_start

        cap.release()

        gpu_temp_end = self._get_gpu_temp()
        vram_end, ram_end = self._get_memory_usage()

        # Calculate statistics
        if frame_times:
            fps = frames_processed / total_time
            avg_time = statistics.mean(frame_times)
            min_time = min(frame_times)
            max_time = max(frame_times)
            std_dev = statistics.stdev(frame_times) if len(frame_times) > 1 else 0
        else:
            fps = avg_time = min_time = max_time = std_dev = 0

        # Check for thermal throttling
        thermal_throttled = False
        if gpu_temp_start and gpu_temp_end:
            if gpu_temp_end > 83 or (gpu_temp_end - gpu_temp_start) > 15:
                thermal_throttled = True

        return BenchmarkResult(
            name=name,
            settings=settings,
            frames_processed=frames_processed,
            total_time=total_time,
            fps=fps,
            avg_frame_time_ms=avg_time,
            min_frame_time_ms=min_time,
            max_frame_time_ms=max_time,
            std_dev_ms=std_dev,
            vram_used_mb=max(vram_end, vram_start),
            ram_used_mb=max(ram_end, ram_start),
            gpu_temp_start=gpu_temp_start,
            gpu_temp_end=gpu_temp_end,
            thermal_throttled=thermal_throttled,
            errors=errors
        )

    def _generate_recommendations(
        self,
        results: List[BenchmarkResult],
        hardware_info: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []

        if not results:
            recommendations.append("No benchmark results to analyze")
            return recommendations

        # Find best configurations
        fastest = max(results, key=lambda r: r.fps)
        most_stable = min(
            [r for r in results if r.std_dev_ms > 0],
            key=lambda r: r.std_dev_ms,
            default=None
        )

        recommendations.append(
            f"Fastest configuration: {fastest.name} at {fastest.fps:.2f} FPS"
        )

        if most_stable and most_stable != fastest:
            recommendations.append(
                f"Most stable configuration: {most_stable.name} "
                f"(±{most_stable.std_dev_ms:.1f}ms variance)"
            )

        # Check for thermal issues
        throttled = [r for r in results if r.thermal_throttled]
        if throttled:
            recommendations.append(
                f"Warning: {len(throttled)} configurations caused thermal throttling. "
                "Consider improving cooling or using smaller batch sizes."
            )

        # VRAM recommendations
        vram_total = hardware_info.get("vram_total_mb", 0)
        if vram_total > 0:
            high_vram = [r for r in results if r.vram_used_mb > vram_total * 0.8]
            if high_vram:
                recommendations.append(
                    f"{len(high_vram)} configurations use >80% VRAM. "
                    "Consider smaller tile sizes for stability."
                )

        # Batch size recommendations
        batch_results = {}
        for r in results:
            batch = r.settings.get("batch_size", 1)
            if batch not in batch_results:
                batch_results[batch] = []
            batch_results[batch].append(r.fps)

        if len(batch_results) > 1:
            avg_by_batch = {b: statistics.mean(fps_list) for b, fps_list in batch_results.items()}
            best_batch = max(avg_by_batch, key=avg_by_batch.get)
            recommendations.append(
                f"Optimal batch size: {best_batch} "
                f"(avg {avg_by_batch[best_batch]:.2f} FPS)"
            )

        return recommendations

    def run(
        self,
        video_path: Path,
        processor_func: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        benchmark_type: BenchmarkType = BenchmarkType.STANDARD,
        custom_configs: Optional[List[Dict[str, Any]]] = None
    ) -> BenchmarkReport:
        """
        Run benchmark suite.

        Args:
            video_path: Path to test video
            processor_func: Function that takes (frame, settings) and returns processed frame
            benchmark_type: Level of thoroughness
            custom_configs: Optional list of custom configurations to test

        Returns:
            BenchmarkReport with all results and recommendations
        """
        from datetime import datetime

        start_time = time.perf_counter()
        video_path = Path(video_path)

        self._report_progress("Gathering hardware info...", 0.0)
        hardware_info = self._get_hardware_info()
        video_info = self._get_video_info(video_path)

        # Generate test configurations based on benchmark type
        if custom_configs:
            configs = custom_configs
        else:
            configs = self._generate_configs(benchmark_type)

        results = []
        total_configs = len(configs)

        for i, config in enumerate(configs):
            name = config.get("name", f"Config {i+1}")
            progress = (i + 1) / total_configs
            self._report_progress(f"Testing: {name}", progress)

            result = self._run_single_benchmark(
                video_path,
                name,
                config,
                processor_func
            )
            results.append(result)

        self._report_progress("Generating recommendations...", 1.0)
        recommendations = self._generate_recommendations(results, hardware_info)

        duration = time.perf_counter() - start_time

        return BenchmarkReport(
            hardware_info=hardware_info,
            test_video_info=video_info,
            results=results,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration
        )

    def _generate_configs(self, benchmark_type: BenchmarkType) -> List[Dict[str, Any]]:
        """Generate test configurations based on benchmark type."""
        configs = []

        if benchmark_type == BenchmarkType.QUICK:
            # Quick: Just test a few batch sizes
            for batch in [1, 4, 8]:
                configs.append({
                    "name": f"Batch {batch}",
                    "batch_size": batch,
                    "tile_size": 512
                })

        elif benchmark_type == BenchmarkType.STANDARD:
            # Standard: Test batch sizes and tile sizes
            for batch in [1, 2, 4, 8]:
                for tile in [512, 768]:
                    configs.append({
                        "name": f"Batch {batch}, Tile {tile}",
                        "batch_size": batch,
                        "tile_size": tile
                    })

        else:  # THOROUGH
            # Thorough: Test all combinations
            for batch in self.BATCH_SIZES:
                for tile in self.TILE_SIZES:
                    configs.append({
                        "name": f"Batch {batch}, Tile {tile}",
                        "batch_size": batch,
                        "tile_size": tile
                    })

            # Add half-precision variants
            for batch in [4, 8, 16]:
                configs.append({
                    "name": f"Batch {batch}, FP16",
                    "batch_size": batch,
                    "tile_size": 512,
                    "half_precision": True
                })

        return configs

    def run_simple(
        self,
        video_path: Path,
        benchmark_type: BenchmarkType = BenchmarkType.QUICK
    ) -> BenchmarkReport:
        """
        Run simple benchmark with dummy processor (measures I/O overhead).

        Useful for establishing baseline performance.
        """
        def dummy_processor(frame: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
            # Simulate some processing with a simple operation
            batch_size = settings.get("batch_size", 1)
            tile_size = settings.get("tile_size", 512)

            # Do some actual work to simulate processing
            result = cv2.GaussianBlur(frame, (5, 5), 0)
            result = cv2.resize(result, (tile_size, tile_size))
            result = cv2.resize(result, (frame.shape[1], frame.shape[0]))

            return result

        return self.run(video_path, dummy_processor, benchmark_type)
