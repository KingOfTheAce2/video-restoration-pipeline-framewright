"""Performance Profiling for FrameWright Video Restoration Pipeline.

Provides detailed performance profiling with stage-by-stage timing,
memory tracking, GPU utilization monitoring, and bottleneck detection.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages in the video restoration pipeline."""
    FRAME_EXTRACTION = "Frame Extraction"
    AI_UPSCALING = "AI Upscaling"
    FACE_RESTORATION = "Face Restoration"
    DEFECT_REPAIR = "Defect Repair"
    FRAME_INTERPOLATION = "Frame Interpolation"
    VIDEO_ENCODING = "Video Encoding"
    AUDIO_ENHANCEMENT = "Audio Enhancement"
    COLORIZATION = "Colorization"
    STABILIZATION = "Stabilization"
    WATERMARK_REMOVAL = "Watermark Removal"
    SCENE_DETECTION = "Scene Detection"
    CUSTOM = "Custom"


@dataclass
class StageMetrics:
    """Metrics for a single processing stage.

    Attributes:
        stage: The processing stage this metrics belongs to
        start_time: Timestamp when stage started
        end_time: Timestamp when stage ended
        duration_seconds: Total time for this stage
        frames_input: Number of input frames
        frames_output: Number of output frames
        memory_start_mb: Memory usage at start
        memory_peak_mb: Peak memory usage during stage
        memory_end_mb: Memory usage at end
        gpu_memory_start_mb: GPU memory at start
        gpu_memory_peak_mb: Peak GPU memory during stage
        gpu_memory_end_mb: GPU memory at end
        gpu_utilization_avg: Average GPU utilization percentage
        cpu_utilization_avg: Average CPU utilization percentage
        samples: Resource samples collected during stage
    """
    stage: ProcessingStage
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    frames_input: int = 0
    frames_output: int = 0
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_end_mb: float = 0.0
    gpu_memory_start_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_end_mb: float = 0.0
    gpu_utilization_avg: float = 0.0
    cpu_utilization_avg: float = 0.0
    samples: List[Dict[str, float]] = field(default_factory=list)

    @property
    def frames_per_second(self) -> float:
        """Calculate processing speed in FPS."""
        if self.duration_seconds > 0:
            return self.frames_output / self.duration_seconds
        return 0.0

    @property
    def memory_delta_mb(self) -> float:
        """Memory change during this stage."""
        return self.memory_end_mb - self.memory_start_mb

    @property
    def gpu_memory_delta_mb(self) -> float:
        """GPU memory change during this stage."""
        return self.gpu_memory_end_mb - self.gpu_memory_start_mb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 3),
            "frames_input": self.frames_input,
            "frames_output": self.frames_output,
            "frames_per_second": round(self.frames_per_second, 2),
            "memory_start_mb": round(self.memory_start_mb, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 1),
            "memory_end_mb": round(self.memory_end_mb, 1),
            "memory_delta_mb": round(self.memory_delta_mb, 1),
            "gpu_memory_start_mb": round(self.gpu_memory_start_mb, 1),
            "gpu_memory_peak_mb": round(self.gpu_memory_peak_mb, 1),
            "gpu_memory_end_mb": round(self.gpu_memory_end_mb, 1),
            "gpu_memory_delta_mb": round(self.gpu_memory_delta_mb, 1),
            "gpu_utilization_avg": round(self.gpu_utilization_avg, 1),
            "cpu_utilization_avg": round(self.cpu_utilization_avg, 1),
        }


@dataclass
class ProfileSummary:
    """Summary of a complete profiling session.

    Attributes:
        total_time_seconds: Total processing time
        total_frames_input: Total input frames
        total_frames_output: Total output frames
        overall_fps: Overall processing speed
        peak_memory_mb: Peak system memory usage
        peak_gpu_memory_mb: Peak GPU memory usage
        bottleneck_stage: Stage that took the longest
        bottleneck_percentage: Percentage of time spent in bottleneck
        recommendations: List of optimization recommendations
    """
    total_time_seconds: float = 0.0
    total_frames_input: int = 0
    total_frames_output: int = 0
    overall_fps: float = 0.0
    peak_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_cpu_utilization: float = 0.0
    bottleneck_stage: Optional[str] = None
    bottleneck_percentage: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_time_seconds": round(self.total_time_seconds, 3),
            "total_frames_input": self.total_frames_input,
            "total_frames_output": self.total_frames_output,
            "overall_fps": round(self.overall_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 1),
            "avg_gpu_utilization": round(self.avg_gpu_utilization, 1),
            "avg_cpu_utilization": round(self.avg_cpu_utilization, 1),
            "bottleneck_stage": self.bottleneck_stage,
            "bottleneck_percentage": round(self.bottleneck_percentage, 1),
            "recommendations": self.recommendations,
        }


class PerformanceProfiler:
    """Track performance metrics for each processing stage.

    This profiler monitors time, memory, and GPU usage across all stages
    of the video restoration pipeline.

    Example:
        >>> profiler = PerformanceProfiler()
        >>> profiler.start_session("my_restoration")
        >>>
        >>> profiler.start_stage(ProcessingStage.FRAME_EXTRACTION, frames=1000)
        >>> # ... extraction code ...
        >>> profiler.end_stage(frames_output=1000)
        >>>
        >>> profiler.start_stage(ProcessingStage.AI_UPSCALING, frames=1000)
        >>> # ... upscaling code ...
        >>> profiler.end_stage(frames_output=1000)
        >>>
        >>> profiler.end_session()
        >>> report = profiler.get_report()
        >>> print(report.format_table())
    """

    def __init__(
        self,
        sample_interval: float = 0.5,
        enable_gpu_monitoring: bool = True,
    ):
        """Initialize the performance profiler.

        Args:
            sample_interval: Interval between resource samples in seconds
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.sample_interval = sample_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring

        self._session_name: Optional[str] = None
        self._session_start: float = 0.0
        self._session_end: float = 0.0
        self._stages: List[StageMetrics] = []
        self._current_stage: Optional[StageMetrics] = None
        self._monitoring_thread = None
        self._monitoring_active = False

        # Callbacks
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def start_session(
        self,
        session_name: str = "profile",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """Start a new profiling session.

        Args:
            session_name: Name to identify this session
            progress_callback: Optional callback for stage progress
        """
        self._session_name = session_name
        self._session_start = time.time()
        self._session_end = 0.0
        self._stages = []
        self._current_stage = None
        self._progress_callback = progress_callback

        logger.info(f"Started profiling session: {session_name}")

    def end_session(self) -> None:
        """End the current profiling session."""
        if self._current_stage:
            self.end_stage()

        self._session_end = time.time()
        logger.info(
            f"Ended profiling session: {self._session_name} "
            f"(total: {self._session_end - self._session_start:.2f}s)"
        )

    def start_stage(
        self,
        stage: ProcessingStage,
        frames: int = 0,
        custom_name: Optional[str] = None,
    ) -> None:
        """Start profiling a processing stage.

        Args:
            stage: The processing stage type
            frames: Expected number of frames to process
            custom_name: Optional custom name for CUSTOM stage type
        """
        if self._current_stage:
            # Auto-end previous stage
            self.end_stage()

        # Get current resource usage
        memory_info = self._get_memory_info()
        gpu_info = self._get_gpu_info() if self.enable_gpu_monitoring else {}

        self._current_stage = StageMetrics(
            stage=stage,
            start_time=time.time(),
            frames_input=frames,
            memory_start_mb=memory_info.get("used_mb", 0),
            gpu_memory_start_mb=gpu_info.get("memory_used_mb", 0),
        )

        # Start resource monitoring
        self._start_monitoring()

        stage_name = custom_name if custom_name else stage.value
        logger.debug(f"Started stage: {stage_name}")

    def end_stage(
        self,
        frames_output: Optional[int] = None,
    ) -> Optional[StageMetrics]:
        """End the current processing stage.

        Args:
            frames_output: Number of frames produced (defaults to input count)

        Returns:
            StageMetrics for the completed stage
        """
        if not self._current_stage:
            return None

        # Stop monitoring
        self._stop_monitoring()

        # Get final resource usage
        memory_info = self._get_memory_info()
        gpu_info = self._get_gpu_info() if self.enable_gpu_monitoring else {}

        self._current_stage.end_time = time.time()
        self._current_stage.duration_seconds = (
            self._current_stage.end_time - self._current_stage.start_time
        )

        if frames_output is not None:
            self._current_stage.frames_output = frames_output
        else:
            self._current_stage.frames_output = self._current_stage.frames_input

        self._current_stage.memory_end_mb = memory_info.get("used_mb", 0)
        self._current_stage.gpu_memory_end_mb = gpu_info.get("memory_used_mb", 0)

        # Calculate peaks and averages from samples
        self._calculate_stage_statistics()

        completed_stage = self._current_stage
        self._stages.append(completed_stage)
        self._current_stage = None

        logger.debug(
            f"Completed stage: {completed_stage.stage.value} "
            f"({completed_stage.duration_seconds:.2f}s, "
            f"{completed_stage.frames_output} frames)"
        )

        return completed_stage

    def record_frames(self, count: int) -> None:
        """Record frame count update for current stage.

        Args:
            count: Number of frames processed so far
        """
        if self._current_stage:
            self._current_stage.frames_output = count

            if self._progress_callback:
                progress = count / max(1, self._current_stage.frames_input)
                self._progress_callback(self._current_stage.stage.value, progress)

    def get_report(self) -> "ProfileReport":
        """Generate a complete profile report.

        Returns:
            ProfileReport with all collected metrics and analysis
        """
        return ProfileReport(
            session_name=self._session_name or "unnamed",
            timestamp=datetime.now().isoformat(),
            stages=list(self._stages),
            total_time=(self._session_end or time.time()) - self._session_start,
        )

    def _start_monitoring(self) -> None:
        """Start background resource monitoring."""
        import threading

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active and self._current_stage:
                sample = self._collect_sample()
                if sample and self._current_stage:
                    self._current_stage.samples.append(sample)
                time.sleep(self.sample_interval)

        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()

    def _stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None

    def _collect_sample(self) -> Dict[str, float]:
        """Collect a resource sample."""
        sample = {"timestamp": time.time()}

        memory_info = self._get_memory_info()
        sample["memory_mb"] = memory_info.get("used_mb", 0)
        sample["cpu_percent"] = memory_info.get("cpu_percent", 0)

        if self.enable_gpu_monitoring:
            gpu_info = self._get_gpu_info()
            sample["gpu_memory_mb"] = gpu_info.get("memory_used_mb", 0)
            sample["gpu_utilization"] = gpu_info.get("utilization_percent", 0)

        return sample

    def _calculate_stage_statistics(self) -> None:
        """Calculate statistics from collected samples."""
        if not self._current_stage or not self._current_stage.samples:
            return

        samples = self._current_stage.samples

        # Memory statistics
        memory_values = [s.get("memory_mb", 0) for s in samples]
        if memory_values:
            self._current_stage.memory_peak_mb = max(memory_values)

        # CPU statistics
        cpu_values = [s.get("cpu_percent", 0) for s in samples]
        if cpu_values:
            self._current_stage.cpu_utilization_avg = sum(cpu_values) / len(cpu_values)

        # GPU statistics
        gpu_memory_values = [s.get("gpu_memory_mb", 0) for s in samples]
        if gpu_memory_values:
            self._current_stage.gpu_memory_peak_mb = max(gpu_memory_values)

        gpu_util_values = [s.get("gpu_utilization", 0) for s in samples]
        if gpu_util_values:
            self._current_stage.gpu_utilization_avg = sum(gpu_util_values) / len(gpu_util_values)

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "used_mb": mem.used / (1024 ** 2),
                "available_mb": mem.available / (1024 ** 2),
                "percent": mem.percent,
                "cpu_percent": psutil.cpu_percent(),
            }
        except ImportError:
            return {}

    def _get_gpu_info(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
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
                    "memory_used_mb": float(parts[0]) if len(parts) > 0 else 0,
                    "memory_total_mb": float(parts[1]) if len(parts) > 1 else 0,
                    "utilization_percent": float(parts[2]) if len(parts) > 2 and parts[2] != "[N/A]" else 0,
                }
        except Exception:
            pass
        return {}


class ProfileReport:
    """Report containing complete profiling data and analysis.

    Provides methods to format, export, and compare profiling results.

    Example:
        >>> report = profiler.get_report()
        >>> print(report.format_table())
        >>> report.export_json(Path("profile.json"))
        >>>
        >>> # Compare multiple reports
        >>> diff = ProfileReport.compare([report1, report2])
    """

    def __init__(
        self,
        session_name: str,
        timestamp: str,
        stages: List[StageMetrics],
        total_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the profile report.

        Args:
            session_name: Name of the profiling session
            timestamp: ISO format timestamp
            stages: List of stage metrics
            total_time: Total session duration in seconds
            metadata: Optional additional metadata
        """
        self.session_name = session_name
        self.timestamp = timestamp
        self.stages = stages
        self.total_time = total_time
        self.metadata = metadata or {}

        # Calculate summary
        self.summary = self._calculate_summary()

    def _calculate_summary(self) -> ProfileSummary:
        """Calculate summary statistics from stage data."""
        if not self.stages:
            return ProfileSummary()

        # Calculate totals
        total_frames_input = sum(s.frames_input for s in self.stages)
        total_frames_output = sum(s.frames_output for s in self.stages)

        # Find peaks
        peak_memory = max(s.memory_peak_mb for s in self.stages)
        peak_gpu_memory = max(s.gpu_memory_peak_mb for s in self.stages)

        # Calculate averages
        total_weighted_gpu = sum(s.gpu_utilization_avg * s.duration_seconds for s in self.stages)
        total_weighted_cpu = sum(s.cpu_utilization_avg * s.duration_seconds for s in self.stages)
        total_duration = sum(s.duration_seconds for s in self.stages)

        avg_gpu = total_weighted_gpu / total_duration if total_duration > 0 else 0
        avg_cpu = total_weighted_cpu / total_duration if total_duration > 0 else 0

        # Find bottleneck
        bottleneck_stage = max(self.stages, key=lambda s: s.duration_seconds)
        bottleneck_percentage = (bottleneck_stage.duration_seconds / self.total_time * 100) if self.total_time > 0 else 0

        # Generate recommendations (pass computed values to avoid self.summary access)
        recommendations = self._generate_recommendations(
            bottleneck_stage,
            peak_memory_mb=peak_memory,
            avg_gpu_utilization=avg_gpu,
        )

        return ProfileSummary(
            total_time_seconds=self.total_time,
            total_frames_input=total_frames_input,
            total_frames_output=total_frames_output,
            overall_fps=total_frames_output / self.total_time if self.total_time > 0 else 0,
            peak_memory_mb=peak_memory,
            peak_gpu_memory_mb=peak_gpu_memory,
            avg_gpu_utilization=avg_gpu,
            avg_cpu_utilization=avg_cpu,
            bottleneck_stage=bottleneck_stage.stage.value,
            bottleneck_percentage=bottleneck_percentage,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        bottleneck: StageMetrics,
        peak_memory_mb: float = 0.0,
        avg_gpu_utilization: float = 0.0,
    ) -> List[str]:
        """Generate optimization recommendations based on profile data."""
        recommendations = []

        # Bottleneck-specific recommendations
        if bottleneck.stage == ProcessingStage.AI_UPSCALING:
            if bottleneck.gpu_utilization_avg < 70:
                recommendations.append(
                    "AI Upscaling: GPU underutilized. Consider using --tile-size 256 for better GPU saturation."
                )
            if bottleneck.gpu_memory_peak_mb > 6000:
                recommendations.append(
                    "AI Upscaling: High VRAM usage. Consider smaller --tile-size or use 2x upscaling."
                )
            else:
                recommendations.append(
                    "AI Upscaling is the bottleneck. Consider parallel processing or faster GPU."
                )

        elif bottleneck.stage == ProcessingStage.FRAME_EXTRACTION:
            recommendations.append(
                "Frame Extraction: Consider SSD storage for faster I/O."
            )
            if bottleneck.cpu_utilization_avg < 50:
                recommendations.append(
                    "Frame Extraction: CPU underutilized. Check disk I/O bottleneck."
                )

        elif bottleneck.stage == ProcessingStage.VIDEO_ENCODING:
            recommendations.append(
                "Video Encoding: Consider GPU-accelerated encoding (NVENC) or reduce CRF quality."
            )

        elif bottleneck.stage == ProcessingStage.FRAME_INTERPOLATION:
            recommendations.append(
                "Frame Interpolation: RIFE is the bottleneck. Reduce target FPS or use faster RIFE model."
            )

        # Memory recommendations
        if peak_memory_mb > 16000:
            recommendations.append(
                "High memory usage detected. Consider processing in smaller batches."
            )

        # GPU utilization recommendations
        if avg_gpu_utilization < 50:
            recommendations.append(
                "Low average GPU utilization. Consider batching frames or using larger tile sizes."
            )

        return recommendations

    def format_table(self, show_recommendations: bool = True) -> str:
        """Format profile data as a text table.

        Args:
            show_recommendations: Whether to include recommendations

        Returns:
            Formatted table string
        """
        lines = []

        # Header
        lines.append("")
        lines.append("Performance Profile Summary:")
        lines.append("-" + "-" * 70 + "-")

        # Table header with box drawing
        header = (
            f"| {'Stage':<21} | {'Time (s)':>8} | {'% Total':>7} | {'Frames':>7} | {'FPS':>7} |"
        )
        separator = (
            "|-" + "-" * 21 + "-|-" + "-" * 8 + "-|-" + "-" * 7 + "-|-" + "-" * 7 + "-|-" + "-" * 7 + "-|"
        )

        lines.append("-" + "-" * 70 + "-")
        lines.append(header)
        lines.append(separator)

        # Stage rows
        for stage in self.stages:
            pct = (stage.duration_seconds / self.total_time * 100) if self.total_time > 0 else 0
            row = (
                f"| {stage.stage.value:<21} | "
                f"{stage.duration_seconds:>8.1f} | "
                f"{pct:>6.1f}% | "
                f"{stage.frames_output:>7} | "
                f"{stage.frames_per_second:>7.1f} |"
            )
            lines.append(row)

        lines.append("-" + "-" * 70 + "-")

        # Summary row
        total_frames = sum(s.frames_output for s in self.stages)
        overall_fps = total_frames / self.total_time if self.total_time > 0 else 0
        summary_row = (
            f"| {'TOTAL':<21} | "
            f"{self.total_time:>8.1f} | "
            f"{'100.0%':>7} | "
            f"{total_frames:>7} | "
            f"{overall_fps:>7.1f} |"
        )
        lines.append(summary_row)
        lines.append("-" + "-" * 70 + "-")

        # Resource usage summary
        lines.append("")
        lines.append("Resource Usage:")
        lines.append(f"  Peak Memory:       {self.summary.peak_memory_mb:.0f} MB")
        lines.append(f"  Peak GPU Memory:   {self.summary.peak_gpu_memory_mb:.0f} MB")
        lines.append(f"  Avg GPU Utilization: {self.summary.avg_gpu_utilization:.1f}%")
        lines.append(f"  Avg CPU Utilization: {self.summary.avg_cpu_utilization:.1f}%")

        # Bottleneck indicator
        if self.summary.bottleneck_stage:
            lines.append("")
            lines.append(
                f"Bottleneck: {self.summary.bottleneck_stage} "
                f"({self.summary.bottleneck_percentage:.1f}% of total time)"
            )

        # Recommendations
        if show_recommendations and self.summary.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.summary.recommendations:
                lines.append(f"  - {rec}")

        lines.append("")

        return "\n".join(lines)

    def format_detailed(self) -> str:
        """Format detailed profile data with all metrics.

        Returns:
            Detailed formatted string
        """
        lines = []

        lines.append("=" * 80)
        lines.append(f"DETAILED PERFORMANCE PROFILE: {self.session_name}")
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append("=" * 80)
        lines.append("")

        for stage in self.stages:
            lines.append(f"Stage: {stage.stage.value}")
            lines.append("-" * 40)
            lines.append(f"  Duration:          {stage.duration_seconds:.3f}s")
            lines.append(f"  Frames In/Out:     {stage.frames_input} / {stage.frames_output}")
            lines.append(f"  FPS:               {stage.frames_per_second:.2f}")
            lines.append("")
            lines.append("  Memory:")
            lines.append(f"    Start:           {stage.memory_start_mb:.1f} MB")
            lines.append(f"    Peak:            {stage.memory_peak_mb:.1f} MB")
            lines.append(f"    End:             {stage.memory_end_mb:.1f} MB")
            lines.append(f"    Delta:           {stage.memory_delta_mb:+.1f} MB")
            lines.append("")
            lines.append("  GPU Memory:")
            lines.append(f"    Start:           {stage.gpu_memory_start_mb:.1f} MB")
            lines.append(f"    Peak:            {stage.gpu_memory_peak_mb:.1f} MB")
            lines.append(f"    End:             {stage.gpu_memory_end_mb:.1f} MB")
            lines.append(f"    Delta:           {stage.gpu_memory_delta_mb:+.1f} MB")
            lines.append("")
            lines.append(f"  GPU Utilization:   {stage.gpu_utilization_avg:.1f}%")
            lines.append(f"  CPU Utilization:   {stage.cpu_utilization_avg:.1f}%")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_json(self, output_path: Path) -> Path:
        """Export profile data to JSON.

        Args:
            output_path: Path for JSON output

        Returns:
            Path to exported file
        """
        data = {
            "session_name": self.session_name,
            "timestamp": self.timestamp,
            "total_time_seconds": round(self.total_time, 3),
            "summary": self.summary.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Profile exported to: {output_path}")
        return output_path

    @classmethod
    def load_json(cls, path: Path) -> "ProfileReport":
        """Load a profile report from JSON.

        Args:
            path: Path to JSON file

        Returns:
            ProfileReport instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        stages = []
        for stage_data in data.get("stages", []):
            stage = StageMetrics(
                stage=ProcessingStage(stage_data["stage"]),
                start_time=stage_data.get("start_time", 0),
                end_time=stage_data.get("end_time", 0),
                duration_seconds=stage_data.get("duration_seconds", 0),
                frames_input=stage_data.get("frames_input", 0),
                frames_output=stage_data.get("frames_output", 0),
                memory_start_mb=stage_data.get("memory_start_mb", 0),
                memory_peak_mb=stage_data.get("memory_peak_mb", 0),
                memory_end_mb=stage_data.get("memory_end_mb", 0),
                gpu_memory_start_mb=stage_data.get("gpu_memory_start_mb", 0),
                gpu_memory_peak_mb=stage_data.get("gpu_memory_peak_mb", 0),
                gpu_memory_end_mb=stage_data.get("gpu_memory_end_mb", 0),
                gpu_utilization_avg=stage_data.get("gpu_utilization_avg", 0),
                cpu_utilization_avg=stage_data.get("cpu_utilization_avg", 0),
            )
            stages.append(stage)

        return cls(
            session_name=data.get("session_name", "loaded"),
            timestamp=data.get("timestamp", ""),
            stages=stages,
            total_time=data.get("total_time_seconds", 0),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def compare(
        reports: List["ProfileReport"],
        labels: Optional[List[str]] = None,
    ) -> str:
        """Compare multiple profile reports.

        Args:
            reports: List of reports to compare
            labels: Optional labels for each report

        Returns:
            Formatted comparison string
        """
        if not reports:
            return "No reports to compare."

        if labels is None:
            labels = [r.session_name for r in reports]

        lines = []
        lines.append("=" * 80)
        lines.append("PROFILE COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        # Summary comparison
        header = f"{'Metric':<25}"
        for label in labels:
            header += f" | {label[:15]:>15}"
        lines.append(header)
        lines.append("-" * (25 + 18 * len(labels)))

        # Total time
        row = f"{'Total Time (s)':<25}"
        for report in reports:
            row += f" | {report.total_time:>15.2f}"
        lines.append(row)

        # Overall FPS
        row = f"{'Overall FPS':<25}"
        for report in reports:
            row += f" | {report.summary.overall_fps:>15.2f}"
        lines.append(row)

        # Peak memory
        row = f"{'Peak Memory (MB)':<25}"
        for report in reports:
            row += f" | {report.summary.peak_memory_mb:>15.0f}"
        lines.append(row)

        # Peak GPU memory
        row = f"{'Peak GPU Memory (MB)':<25}"
        for report in reports:
            row += f" | {report.summary.peak_gpu_memory_mb:>15.0f}"
        lines.append(row)

        # Bottleneck
        row = f"{'Bottleneck':<25}"
        for report in reports:
            stage = report.summary.bottleneck_stage or "N/A"
            if len(stage) > 15:
                stage = stage[:12] + "..."
            row += f" | {stage:>15}"
        lines.append(row)

        lines.append("")

        # Stage-by-stage comparison for common stages
        lines.append("Stage Timing Comparison:")
        lines.append("-" * 80)

        # Find all unique stages
        all_stages = set()
        for report in reports:
            for stage in report.stages:
                all_stages.add(stage.stage)

        for stage_type in sorted(all_stages, key=lambda s: s.value):
            header = f"  {stage_type.value:<20}"
            for report in reports:
                matching = [s for s in report.stages if s.stage == stage_type]
                if matching:
                    header += f" | {matching[0].duration_seconds:>10.2f}s"
                else:
                    header += f" | {'N/A':>10}"
            lines.append(header)

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def calculate_improvement(
        baseline: "ProfileReport",
        current: "ProfileReport",
    ) -> Dict[str, Any]:
        """Calculate improvement between two profiles.

        Args:
            baseline: The reference/older profile
            current: The current/newer profile

        Returns:
            Dictionary with improvement metrics
        """
        improvements = {}

        # Time improvement
        if baseline.total_time > 0:
            time_improvement = (baseline.total_time - current.total_time) / baseline.total_time * 100
            improvements["time_improvement_percent"] = round(time_improvement, 1)
            improvements["speedup_factor"] = round(baseline.total_time / max(current.total_time, 0.001), 2)

        # FPS improvement
        if baseline.summary.overall_fps > 0:
            fps_improvement = (current.summary.overall_fps - baseline.summary.overall_fps) / baseline.summary.overall_fps * 100
            improvements["fps_improvement_percent"] = round(fps_improvement, 1)

        # Memory improvement
        memory_change = baseline.summary.peak_memory_mb - current.summary.peak_memory_mb
        improvements["memory_reduction_mb"] = round(memory_change, 1)

        gpu_memory_change = baseline.summary.peak_gpu_memory_mb - current.summary.peak_gpu_memory_mb
        improvements["gpu_memory_reduction_mb"] = round(gpu_memory_change, 1)

        return improvements


def analyze_profile(profile_path: Path) -> str:
    """Analyze a saved profile and generate insights.

    Args:
        profile_path: Path to profile JSON file

    Returns:
        Analysis string with insights
    """
    report = ProfileReport.load_json(profile_path)

    lines = []
    lines.append(f"Profile Analysis: {report.session_name}")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(report.format_table())

    # Additional insights
    lines.append("")
    lines.append("Insights:")
    lines.append("-" * 40)

    # Efficiency analysis
    gpu_efficiency = report.summary.avg_gpu_utilization
    if gpu_efficiency < 30:
        lines.append(f"- GPU Utilization is LOW ({gpu_efficiency:.1f}%): Pipeline is likely CPU or I/O bound")
    elif gpu_efficiency < 70:
        lines.append(f"- GPU Utilization is MODERATE ({gpu_efficiency:.1f}%): Some optimization possible")
    else:
        lines.append(f"- GPU Utilization is GOOD ({gpu_efficiency:.1f}%): GPU is well utilized")

    # Stage balance
    if len(report.stages) > 1:
        durations = [s.duration_seconds for s in report.stages]
        max_dur = max(durations)
        min_dur = min(d for d in durations if d > 0) if any(d > 0 for d in durations) else 0

        if max_dur > 0 and min_dur > 0:
            imbalance = max_dur / min_dur
            if imbalance > 10:
                lines.append(f"- Stage imbalance detected: slowest stage is {imbalance:.1f}x slower than fastest")
                lines.append("  Consider parallelizing independent stages")

    lines.append("")

    return "\n".join(lines)
