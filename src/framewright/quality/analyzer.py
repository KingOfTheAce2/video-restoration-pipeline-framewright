"""Comprehensive quality analysis for video restoration."""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil

logger = logging.getLogger(__name__)


@dataclass
class FrameQualityMetrics:
    """Quality metrics for a single frame."""
    frame_number: int
    timestamp: float = 0.0

    # Standard metrics
    psnr: float = 0.0
    ssim: float = 0.0
    vmaf: float = 0.0

    # Noise analysis
    noise_level: float = 0.0
    snr: float = 0.0  # Signal-to-noise ratio

    # Sharpness
    sharpness: float = 0.0
    blur_score: float = 0.0

    # Color
    color_accuracy: float = 0.0
    saturation_level: float = 0.0
    contrast: float = 0.0

    # Artifact detection
    blocking_score: float = 0.0
    banding_score: float = 0.0
    ringing_score: float = 0.0

    # Temporal
    temporal_consistency: float = 0.0
    flicker_score: float = 0.0

    # Grain
    grain_level: float = 0.0
    grain_uniformity: float = 0.0

    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "ssim": 0.25,
            "psnr_norm": 0.15,
            "vmaf_norm": 0.20,
            "sharpness": 0.10,
            "blocking_inv": 0.10,
            "temporal": 0.10,
            "color": 0.10,
        }

        # Normalize PSNR (30-50 dB range to 0-1)
        psnr_norm = max(0, min(1, (self.psnr - 30) / 20))

        # Normalize VMAF (0-100 to 0-1)
        vmaf_norm = self.vmaf / 100

        # Invert blocking (low is good)
        blocking_inv = 1 - min(1, self.blocking_score)

        score = (
            weights["ssim"] * self.ssim +
            weights["psnr_norm"] * psnr_norm +
            weights["vmaf_norm"] * vmaf_norm +
            weights["sharpness"] * min(1, self.sharpness) +
            weights["blocking_inv"] * blocking_inv +
            weights["temporal"] * self.temporal_consistency +
            weights["color"] * self.color_accuracy
        )

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "vmaf": self.vmaf,
            "noise_level": self.noise_level,
            "snr": self.snr,
            "sharpness": self.sharpness,
            "blur_score": self.blur_score,
            "color_accuracy": self.color_accuracy,
            "saturation_level": self.saturation_level,
            "contrast": self.contrast,
            "blocking_score": self.blocking_score,
            "banding_score": self.banding_score,
            "ringing_score": self.ringing_score,
            "temporal_consistency": self.temporal_consistency,
            "flicker_score": self.flicker_score,
            "grain_level": self.grain_level,
            "grain_uniformity": self.grain_uniformity,
            "overall_score": self.overall_score(),
        }


@dataclass
class VideoQualityReport:
    """Complete quality report for a video."""
    # Source info
    source_path: Optional[Path] = None
    reference_path: Optional[Path] = None

    # Video properties
    duration: float = 0.0
    frame_count: int = 0
    fps: float = 0.0
    resolution: Tuple[int, int] = (0, 0)

    # Aggregate metrics
    mean_psnr: float = 0.0
    mean_ssim: float = 0.0
    mean_vmaf: float = 0.0

    min_psnr: float = 0.0
    min_ssim: float = 0.0
    min_vmaf: float = 0.0

    max_psnr: float = 0.0
    max_ssim: float = 0.0
    max_vmaf: float = 0.0

    std_psnr: float = 0.0
    std_ssim: float = 0.0
    std_vmaf: float = 0.0

    # Quality distribution
    percentile_5_vmaf: float = 0.0
    percentile_25_vmaf: float = 0.0
    percentile_50_vmaf: float = 0.0
    percentile_75_vmaf: float = 0.0
    percentile_95_vmaf: float = 0.0

    # Artifact analysis
    mean_blocking: float = 0.0
    mean_banding: float = 0.0
    mean_ringing: float = 0.0

    # Temporal analysis
    mean_temporal_consistency: float = 0.0
    flicker_count: int = 0

    # Noise analysis
    mean_noise_level: float = 0.0
    mean_grain_level: float = 0.0

    # Overall
    overall_quality_score: float = 0.0
    quality_grade: str = "Unknown"

    # Frame-by-frame data
    frame_metrics: List[FrameQualityMetrics] = field(default_factory=list)

    # Problem areas
    problem_frames: List[int] = field(default_factory=list)
    problem_regions: List[Dict[str, Any]] = field(default_factory=list)

    # Generation metadata
    generated_at: datetime = field(default_factory=datetime.now)
    analysis_time: float = 0.0

    def get_grade(self) -> str:
        """Get letter grade based on overall score."""
        if self.overall_quality_score >= 0.95:
            return "A+"
        elif self.overall_quality_score >= 0.90:
            return "A"
        elif self.overall_quality_score >= 0.85:
            return "A-"
        elif self.overall_quality_score >= 0.80:
            return "B+"
        elif self.overall_quality_score >= 0.75:
            return "B"
        elif self.overall_quality_score >= 0.70:
            return "B-"
        elif self.overall_quality_score >= 0.65:
            return "C+"
        elif self.overall_quality_score >= 0.60:
            return "C"
        elif self.overall_quality_score >= 0.55:
            return "C-"
        elif self.overall_quality_score >= 0.50:
            return "D"
        else:
            return "F"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": str(self.source_path) if self.source_path else None,
            "reference": str(self.reference_path) if self.reference_path else None,
            "video_properties": {
                "duration": self.duration,
                "frame_count": self.frame_count,
                "fps": self.fps,
                "resolution": list(self.resolution),
            },
            "aggregate_metrics": {
                "psnr": {"mean": self.mean_psnr, "min": self.min_psnr, "max": self.max_psnr, "std": self.std_psnr},
                "ssim": {"mean": self.mean_ssim, "min": self.min_ssim, "max": self.max_ssim, "std": self.std_ssim},
                "vmaf": {
                    "mean": self.mean_vmaf,
                    "min": self.min_vmaf,
                    "max": self.max_vmaf,
                    "std": self.std_vmaf,
                    "percentiles": {
                        "5": self.percentile_5_vmaf,
                        "25": self.percentile_25_vmaf,
                        "50": self.percentile_50_vmaf,
                        "75": self.percentile_75_vmaf,
                        "95": self.percentile_95_vmaf,
                    },
                },
            },
            "artifacts": {
                "blocking": self.mean_blocking,
                "banding": self.mean_banding,
                "ringing": self.mean_ringing,
            },
            "temporal": {
                "consistency": self.mean_temporal_consistency,
                "flicker_count": self.flicker_count,
            },
            "noise": {
                "noise_level": self.mean_noise_level,
                "grain_level": self.mean_grain_level,
            },
            "overall": {
                "score": self.overall_quality_score,
                "grade": self.quality_grade,
            },
            "problem_frames": self.problem_frames,
            "generated_at": self.generated_at.isoformat(),
            "analysis_time": self.analysis_time,
        }

    def save_json(self, path: Path) -> None:
        """Save report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_html(self, path: Path) -> None:
        """Save report as HTML."""
        html = self._generate_html()
        path.write_text(html)

    def _generate_html(self) -> str:
        """Generate HTML report."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Quality Report - {self.source_path.name if self.source_path else 'Video'}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f7; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1d1d1f; margin-bottom: 10px; }}
        h2 {{ color: #424245; border-bottom: 1px solid #e5e5e5; padding-bottom: 10px; margin-top: 30px; }}
        .grade {{ font-size: 72px; font-weight: bold; color: {self._grade_color()}; text-align: center; padding: 20px; }}
        .score {{ font-size: 24px; color: #6e6e73; text-align: center; }}
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px; }}
        .metric {{ background: #f5f5f7; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #1d1d1f; }}
        .metric-label {{ color: #6e6e73; margin-top: 5px; }}
        .problems {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        .problems.none {{ background: #d4edda; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        th {{ background: #f5f5f7; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Quality Report</h1>
        <p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="grade">{self.quality_grade}</div>
        <div class="score">Overall Score: {self.overall_quality_score:.1%}</div>

        <h2>Key Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{self.mean_vmaf:.1f}</div>
                <div class="metric-label">VMAF</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.mean_psnr:.1f}</div>
                <div class="metric-label">PSNR (dB)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.mean_ssim:.4f}</div>
                <div class="metric-label">SSIM</div>
            </div>
        </div>

        <h2>Video Properties</h2>
        <table>
            <tr><td>Resolution</td><td>{self.resolution[0]}x{self.resolution[1]}</td></tr>
            <tr><td>Duration</td><td>{self.duration:.2f}s</td></tr>
            <tr><td>Frame Count</td><td>{self.frame_count}</td></tr>
            <tr><td>Frame Rate</td><td>{self.fps:.2f} fps</td></tr>
        </table>

        <h2>Artifact Analysis</h2>
        <table>
            <tr><td>Blocking Artifacts</td><td>{self.mean_blocking:.3f}</td></tr>
            <tr><td>Banding Artifacts</td><td>{self.mean_banding:.3f}</td></tr>
            <tr><td>Ringing Artifacts</td><td>{self.mean_ringing:.3f}</td></tr>
            <tr><td>Temporal Consistency</td><td>{self.mean_temporal_consistency:.3f}</td></tr>
        </table>

        <h2>Problem Frames</h2>
        <div class="problems {'none' if not self.problem_frames else ''}">
            {f'Found {len(self.problem_frames)} problem frames: {self.problem_frames[:20]}{"..." if len(self.problem_frames) > 20 else ""}' if self.problem_frames else 'No significant problems detected.'}
        </div>
    </div>
</body>
</html>"""

    def _grade_color(self) -> str:
        """Get color for grade display."""
        if self.quality_grade.startswith("A"):
            return "#34c759"
        elif self.quality_grade.startswith("B"):
            return "#007aff"
        elif self.quality_grade.startswith("C"):
            return "#ff9500"
        elif self.quality_grade.startswith("D"):
            return "#ff6b6b"
        else:
            return "#ff3b30"


class QualityAnalyzer:
    """Comprehensive video quality analyzer."""

    def __init__(self):
        self._ffmpeg_path = shutil.which("ffmpeg")
        self._ffprobe_path = shutil.which("ffprobe")
        self._numpy_available = self._check_numpy()

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def analyze(
        self,
        video_path: Path,
        reference_path: Optional[Path] = None,
        sample_rate: int = 1,
        include_vmaf: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> VideoQualityReport:
        """Analyze video quality comprehensively."""
        import time
        start_time = time.time()

        report = VideoQualityReport(source_path=video_path, reference_path=reference_path)

        # Get video info
        info = self._get_video_info(video_path)
        report.duration = info.get("duration", 0)
        report.frame_count = info.get("frame_count", 0)
        report.fps = info.get("fps", 24)
        report.resolution = (info.get("width", 0), info.get("height", 0))

        # Calculate metrics
        if reference_path:
            # Full comparison analysis
            self._analyze_comparison(report, video_path, reference_path, sample_rate, include_vmaf, progress_callback)
        else:
            # Single video analysis (no-reference)
            self._analyze_single(report, video_path, sample_rate, progress_callback)

        # Calculate aggregates and overall score
        self._calculate_aggregates(report)
        report.quality_grade = report.get_grade()

        report.analysis_time = time.time() - start_time
        logger.info(f"Quality analysis completed in {report.analysis_time:.1f}s - Grade: {report.quality_grade}")

        return report

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information using ffprobe."""
        if not self._ffprobe_path:
            return {}

        try:
            cmd = [
                self._ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
                "-of", "json",
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                return {}

            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            format_data = data.get("format", {})

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "24/1")
            if "/" in fps_str:
                num, denom = fps_str.split("/")
                fps = float(num) / float(denom) if float(denom) != 0 else 24
            else:
                fps = float(fps_str)

            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "frame_count": int(stream.get("nb_frames", 0)),
                "duration": float(format_data.get("duration", 0)),
            }

        except Exception as e:
            logger.warning(f"Failed to get video info: {e}")
            return {}

    def _analyze_comparison(
        self,
        report: VideoQualityReport,
        video_path: Path,
        reference_path: Path,
        sample_rate: int,
        include_vmaf: bool,
        progress_callback: Optional[callable],
    ) -> None:
        """Analyze video quality against reference."""
        # Use VMAF calculator if available and requested
        if include_vmaf:
            from .vmaf import VMAFCalculator, VMAFConfig

            vmaf_calc = VMAFCalculator(VMAFConfig(
                include_psnr=True,
                include_ssim=True,
                subsample=sample_rate,
            ))

            vmaf_result = vmaf_calc.calculate(reference_path, video_path)

            # Copy VMAF metrics to report
            report.mean_vmaf = vmaf_result.vmaf_mean
            report.min_vmaf = vmaf_result.vmaf_min
            report.max_vmaf = vmaf_result.vmaf_max
            report.std_vmaf = vmaf_result.vmaf_std

            report.percentile_5_vmaf = vmaf_result.vmaf_percentile_5
            report.percentile_25_vmaf = vmaf_result.vmaf_percentile_25
            report.percentile_50_vmaf = vmaf_result.vmaf_percentile_50
            report.percentile_75_vmaf = vmaf_result.vmaf_percentile_75
            report.percentile_95_vmaf = vmaf_result.vmaf_percentile_95

            if vmaf_result.psnr_mean:
                report.mean_psnr = vmaf_result.psnr_mean

            if vmaf_result.ssim_mean:
                report.mean_ssim = vmaf_result.ssim_mean

            # Convert frame scores
            for vmaf_frame in vmaf_result.frame_scores:
                frame_metric = FrameQualityMetrics(
                    frame_number=vmaf_frame.frame_number,
                    vmaf=vmaf_frame.vmaf,
                    psnr=vmaf_frame.psnr_y or 0,
                    ssim=vmaf_frame.ssim or 0,
                )
                report.frame_metrics.append(frame_metric)

            # Identify problem frames
            for frame in vmaf_result.get_problem_frames(70):
                report.problem_frames.append(frame.frame_number)

    def _analyze_single(
        self,
        report: VideoQualityReport,
        video_path: Path,
        sample_rate: int,
        progress_callback: Optional[callable],
    ) -> None:
        """Analyze single video quality (no reference)."""
        if not self._numpy_available or not self._ffmpeg_path:
            return

        import numpy as np

        # Sample frames and analyze
        total_frames = report.frame_count or 1000
        sample_frames = list(range(0, total_frames, sample_rate * 10))[:100]

        frame_metrics = []

        for i, frame_num in enumerate(sample_frames):
            try:
                frame = self._extract_frame(video_path, frame_num)
                if frame is None:
                    continue

                metrics = self._analyze_frame(frame, frame_num)
                frame_metrics.append(metrics)
                report.frame_metrics.append(metrics)

                if progress_callback:
                    progress_callback((i + 1) / len(sample_frames))

            except Exception as e:
                logger.warning(f"Frame {frame_num} analysis failed: {e}")

        if not frame_metrics:
            return

        # Calculate aggregates from frame metrics
        report.mean_noise_level = np.mean([f.noise_level for f in frame_metrics])
        report.mean_grain_level = np.mean([f.grain_level for f in frame_metrics])
        report.mean_blocking = np.mean([f.blocking_score for f in frame_metrics])

        # Estimate SSIM/PSNR from sharpness and noise
        avg_sharpness = np.mean([f.sharpness for f in frame_metrics])
        report.mean_ssim = min(0.99, 0.85 + avg_sharpness * 0.1 - report.mean_noise_level * 0.1)
        report.mean_psnr = max(25, 35 + avg_sharpness * 5 - report.mean_noise_level * 10)

        # Estimate VMAF
        report.mean_vmaf = max(50, min(100, report.mean_ssim * 80 + 15))

    def _extract_frame(self, video_path: Path, frame_number: int) -> Optional[Any]:
        """Extract a frame from video."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cmd = [
                self._ffmpeg_path,
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{frame_number})",
                "-vframes", "1",
                "-y",
                str(temp_path)
            ]

            subprocess.run(cmd, capture_output=True, timeout=60)

            if temp_path.exists():
                import cv2
                return cv2.imread(str(temp_path))

        except Exception:
            pass
        finally:
            temp_path.unlink(missing_ok=True)

        return None

    def _analyze_frame(self, frame: Any, frame_number: int) -> FrameQualityMetrics:
        """Analyze single frame quality metrics."""
        import numpy as np

        metrics = FrameQualityMetrics(frame_number=frame_number)

        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame

        # Sharpness (Laplacian variance)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        padded = np.pad(gray, 1, mode='edge')
        lap_response = np.zeros_like(gray, dtype=np.float32)

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                window = padded[i:i + 3, j:j + 3]
                lap_response[i, j] = np.sum(window * laplacian)

        metrics.sharpness = np.var(lap_response) / 1000

        # Noise estimation
        kernel_size = 3
        smoothed = np.zeros_like(gray, dtype=np.float32)
        gray_float = gray.astype(np.float32)
        padded_smooth = np.pad(gray_float, kernel_size // 2, mode='edge')

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                window = padded_smooth[i:i + kernel_size, j:j + kernel_size]
                smoothed[i, j] = np.mean(window)

        metrics.noise_level = np.std(gray_float - smoothed) / 255

        # Blocking detection (8x8 grid analysis)
        block_size = 8
        h, w = gray.shape
        h_diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
        v_diff = np.abs(np.diff(gray.astype(np.float32), axis=0))

        # Check for grid-aligned edges
        h_grid_edges = h_diff[:, block_size - 1::block_size]
        h_other_edges = np.delete(h_diff, list(range(block_size - 1, h_diff.shape[1], block_size)), axis=1)

        if h_other_edges.size > 0:
            grid_strength = np.mean(h_grid_edges) if h_grid_edges.size > 0 else 0
            other_strength = np.mean(h_other_edges)
            metrics.blocking_score = max(0, (grid_strength - other_strength) / (other_strength + 1)) / 10

        # Contrast
        metrics.contrast = (np.max(gray) - np.min(gray)) / 255

        # Saturation
        if len(frame.shape) == 3:
            max_rgb = np.max(frame, axis=2)
            min_rgb = np.min(frame, axis=2)
            saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)
            metrics.saturation_level = np.mean(saturation)

        # Grain estimation (high-frequency content)
        metrics.grain_level = metrics.noise_level * 0.8

        return metrics

    def _calculate_aggregates(self, report: VideoQualityReport) -> None:
        """Calculate aggregate metrics from frame data."""
        if not report.frame_metrics:
            return

        import numpy as np

        psnr_values = [f.psnr for f in report.frame_metrics if f.psnr > 0]
        ssim_values = [f.ssim for f in report.frame_metrics if f.ssim > 0]
        vmaf_values = [f.vmaf for f in report.frame_metrics if f.vmaf > 0]

        if psnr_values:
            if not report.mean_psnr:
                report.mean_psnr = np.mean(psnr_values)
            report.min_psnr = np.min(psnr_values)
            report.max_psnr = np.max(psnr_values)
            report.std_psnr = np.std(psnr_values)

        if ssim_values:
            if not report.mean_ssim:
                report.mean_ssim = np.mean(ssim_values)
            report.min_ssim = np.min(ssim_values)
            report.max_ssim = np.max(ssim_values)
            report.std_ssim = np.std(ssim_values)

        if vmaf_values:
            if not report.mean_vmaf:
                report.mean_vmaf = np.mean(vmaf_values)
            report.min_vmaf = np.min(vmaf_values)
            report.max_vmaf = np.max(vmaf_values)
            report.std_vmaf = np.std(vmaf_values)

        # Calculate overall score
        scores = [f.overall_score() for f in report.frame_metrics]
        report.overall_quality_score = np.mean(scores) if scores else 0.5


def quick_analyze(video_path: Path, reference_path: Optional[Path] = None) -> Dict[str, Any]:
    """Quick quality analysis returning key metrics."""
    analyzer = QualityAnalyzer()
    report = analyzer.analyze(video_path, reference_path, sample_rate=10, include_vmaf=reference_path is not None)

    return {
        "grade": report.quality_grade,
        "score": report.overall_quality_score,
        "vmaf": report.mean_vmaf,
        "psnr": report.mean_psnr,
        "ssim": report.mean_ssim,
    }
