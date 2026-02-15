"""QA Report Generator for FrameWright.

Generates comprehensive quality assurance reports for restored videos
with visual comparisons, metrics analysis, and recommendations.
"""

import logging
import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QualityGrade(Enum):
    """Quality grade for restoration results."""
    EXCELLENT = "A+"
    VERY_GOOD = "A"
    GOOD = "B"
    ACCEPTABLE = "C"
    NEEDS_IMPROVEMENT = "D"
    POOR = "F"


@dataclass
class FrameMetrics:
    """Quality metrics for a single frame."""
    frame_number: int
    timestamp: float
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0  # Lower is better
    vmaf: float = 0.0
    niqe: float = 0.0  # Lower is better (no-reference)
    brisque: float = 0.0  # Lower is better
    sharpness: float = 0.0
    noise_level: float = 0.0
    has_artifacts: bool = False
    artifact_types: List[str] = field(default_factory=list)


@dataclass
class SegmentAnalysis:
    """Analysis of a video segment."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    avg_psnr: float = 0.0
    avg_ssim: float = 0.0
    avg_vmaf: float = 0.0
    min_quality_frame: int = 0
    max_quality_frame: int = 0
    issues: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class FaceQualityReport:
    """Quality report for face regions."""
    total_faces_detected: int = 0
    faces_restored: int = 0
    avg_face_quality: float = 0.0
    identity_preservation: float = 0.0
    detail_recovery: float = 0.0
    problem_frames: List[int] = field(default_factory=list)


@dataclass
class AudioQualityReport:
    """Quality report for audio restoration."""
    original_snr: float = 0.0
    restored_snr: float = 0.0
    hum_removed: bool = False
    clicks_removed: int = 0
    dialog_clarity: float = 0.0
    loudness_lufs: float = 0.0
    peak_dbfs: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class ProcessingStats:
    """Processing statistics."""
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    interpolated_frames: int = 0
    generated_frames: int = 0
    processing_time_seconds: float = 0.0
    avg_frame_time: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    stages_completed: List[str] = field(default_factory=list)


@dataclass
class QAReport:
    """Comprehensive QA report for a restoration job."""
    # Identification
    report_id: str = ""
    created_at: str = ""
    input_file: str = ""
    output_file: str = ""
    preset_used: str = ""

    # Video info
    resolution_original: str = ""
    resolution_output: str = ""
    duration_seconds: float = 0.0
    fps_original: float = 0.0
    fps_output: float = 0.0
    codec_original: str = ""
    codec_output: str = ""

    # Overall quality
    overall_grade: QualityGrade = QualityGrade.GOOD
    overall_score: float = 0.0
    quality_improvement: float = 0.0  # Percentage

    # Detailed metrics
    avg_psnr: float = 0.0
    avg_ssim: float = 0.0
    avg_vmaf: float = 0.0
    min_psnr: float = 0.0
    max_psnr: float = 0.0
    psnr_std_dev: float = 0.0

    # Frame-level data
    frame_metrics: List[FrameMetrics] = field(default_factory=list)
    problem_frames: List[int] = field(default_factory=list)
    segments: List[SegmentAnalysis] = field(default_factory=list)

    # Component reports
    face_report: Optional[FaceQualityReport] = None
    audio_report: Optional[AudioQualityReport] = None
    processing_stats: ProcessingStats = field(default_factory=ProcessingStats)

    # Issues and recommendations
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["overall_grade"] = self.overall_grade.value
        return data


class QAReportGenerator:
    """Generates comprehensive QA reports for restored videos.

    Features:
    - Full-reference metrics (PSNR, SSIM, VMAF, LPIPS)
    - No-reference quality assessment (NIQE, BRISQUE)
    - Problem frame detection
    - Segment-by-segment analysis
    - Visual comparison generation
    - HTML/JSON/PDF export
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        enable_vmaf: bool = True,
        enable_lpips: bool = True,
        enable_no_reference: bool = True,
        comparison_frames: int = 10,
    ):
        """Initialize QA report generator.

        Args:
            sample_rate: Fraction of frames to analyze (1.0 = all)
            enable_vmaf: Calculate VMAF scores
            enable_lpips: Calculate LPIPS scores
            enable_no_reference: Calculate no-reference metrics
            comparison_frames: Number of comparison frames to generate
        """
        self.sample_rate = sample_rate
        self.enable_vmaf = enable_vmaf
        self.enable_lpips = enable_lpips
        self.enable_no_reference = enable_no_reference
        self.comparison_frames = comparison_frames

        self._cv2 = None
        self._np = None

    def _ensure_deps(self) -> bool:
        """Ensure dependencies are available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            logger.warning("OpenCV/NumPy not available for QA analysis")
            return False

    def generate_report(
        self,
        original_path: Path,
        restored_path: Path,
        output_dir: Optional[Path] = None,
        config_used: Optional[Dict[str, Any]] = None,
    ) -> QAReport:
        """Generate comprehensive QA report.

        Args:
            original_path: Path to original video
            restored_path: Path to restored video
            output_dir: Directory for report outputs
            config_used: Configuration used for restoration

        Returns:
            QAReport with all analysis results
        """
        import uuid
        from datetime import datetime

        original_path = Path(original_path)
        restored_path = Path(restored_path)

        if output_dir is None:
            output_dir = restored_path.parent / "qa_reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize report
        report = QAReport(
            report_id=str(uuid.uuid4())[:8],
            created_at=datetime.now().isoformat(),
            input_file=str(original_path),
            output_file=str(restored_path),
            preset_used=config_used.get("preset", "unknown") if config_used else "unknown",
        )

        logger.info(f"Generating QA report for {restored_path.name}")

        # Get video info
        self._analyze_video_info(original_path, restored_path, report)

        # Analyze frames
        if self._ensure_deps():
            self._analyze_frames(original_path, restored_path, report)

        # Calculate overall metrics
        self._calculate_overall_metrics(report)

        # Detect problem areas
        self._detect_problems(report)

        # Analyze segments
        self._analyze_segments(report)

        # Generate grade
        self._calculate_grade(report)

        # Generate recommendations
        self._generate_recommendations(report)

        # Generate comparison images
        self._generate_comparisons(original_path, restored_path, output_dir, report)

        logger.info(f"QA report generated: Grade {report.overall_grade.value}")
        return report

    def _analyze_video_info(
        self,
        original: Path,
        restored: Path,
        report: QAReport,
    ) -> None:
        """Extract video information."""
        import subprocess
        import json

        def get_video_info(path: Path) -> Dict[str, Any]:
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-print_format", "json",
                        "-show_format", "-show_streams",
                        str(path)
                    ],
                    capture_output=True,
                    text=True,
                )
                return json.loads(result.stdout)
            except Exception:
                return {}

        orig_info = get_video_info(original)
        rest_info = get_video_info(restored)

        # Extract original info
        if orig_info:
            for stream in orig_info.get("streams", []):
                if stream.get("codec_type") == "video":
                    report.resolution_original = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        report.fps_original = float(num) / float(den) if float(den) > 0 else 0
                    report.codec_original = stream.get("codec_name", "unknown")
                    break

            fmt = orig_info.get("format", {})
            report.duration_seconds = float(fmt.get("duration", 0))

        # Extract restored info
        if rest_info:
            for stream in rest_info.get("streams", []):
                if stream.get("codec_type") == "video":
                    report.resolution_output = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        report.fps_output = float(num) / float(den) if float(den) > 0 else 0
                    report.codec_output = stream.get("codec_name", "unknown")
                    break

    def _analyze_frames(
        self,
        original: Path,
        restored: Path,
        report: QAReport,
    ) -> None:
        """Analyze frame-by-frame quality."""
        cv2 = self._cv2
        np = self._np

        cap_orig = cv2.VideoCapture(str(original))
        cap_rest = cv2.VideoCapture(str(restored))

        if not cap_orig.isOpened() or not cap_rest.isOpened():
            logger.error("Could not open video files for analysis")
            return

        total_frames = int(cap_rest.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_rest.get(cv2.CAP_PROP_FPS)

        # Calculate which frames to sample
        sample_interval = max(1, int(1.0 / self.sample_rate))

        frame_num = 0
        psnr_values = []
        ssim_values = []

        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_rest, frame_rest = cap_rest.read()

            if not ret_orig or not ret_rest:
                break

            if frame_num % sample_interval == 0:
                # Calculate metrics
                psnr = self._calculate_psnr(frame_orig, frame_rest)
                ssim = self._calculate_ssim(frame_orig, frame_rest)

                metrics = FrameMetrics(
                    frame_number=frame_num,
                    timestamp=frame_num / fps if fps > 0 else 0,
                    psnr=psnr,
                    ssim=ssim,
                )

                # Additional metrics if enabled
                if self.enable_no_reference:
                    metrics.sharpness = self._calculate_sharpness(frame_rest)
                    metrics.noise_level = self._estimate_noise(frame_rest)

                # Detect artifacts
                artifacts = self._detect_artifacts(frame_rest)
                if artifacts:
                    metrics.has_artifacts = True
                    metrics.artifact_types = artifacts

                report.frame_metrics.append(metrics)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

            frame_num += 1

        cap_orig.release()
        cap_rest.release()

        # Update processing stats
        report.processing_stats.total_frames = total_frames
        report.processing_stats.processed_frames = len(report.frame_metrics)

    def _calculate_psnr(self, original: Any, restored: Any) -> float:
        """Calculate PSNR between two frames."""
        cv2 = self._cv2
        np = self._np

        # Resize if needed
        if original.shape != restored.shape:
            original = cv2.resize(original, (restored.shape[1], restored.shape[0]))

        mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0  # Perfect match

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return float(psnr)

    def _calculate_ssim(self, original: Any, restored: Any) -> float:
        """Calculate SSIM between two frames."""
        cv2 = self._cv2
        np = self._np

        # Resize if needed
        if original.shape != restored.shape:
            original = cv2.resize(original, (restored.shape[1], restored.shape[0]))

        # Convert to grayscale
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_rest = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)

        # Constants for stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Calculate means
        mu1 = cv2.GaussianBlur(gray_orig.astype(np.float64), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray_rest.astype(np.float64), (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Calculate variances
        sigma1_sq = cv2.GaussianBlur(gray_orig.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray_rest.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(
            gray_orig.astype(np.float64) * gray_rest.astype(np.float64),
            (11, 11), 1.5
        ) - mu1_mu2

        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return float(np.mean(ssim_map))

    def _calculate_sharpness(self, frame: Any) -> float:
        """Calculate sharpness using Laplacian variance."""
        cv2 = self._cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _estimate_noise(self, frame: Any) -> float:
        """Estimate noise level in frame."""
        cv2 = self._cv2
        np = self._np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use median absolute deviation of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian)) / 0.6745

        return float(sigma)

    def _detect_artifacts(self, frame: Any) -> List[str]:
        """Detect visual artifacts in frame."""
        cv2 = self._cv2
        np = self._np
        artifacts = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect blocking artifacts (8x8 or 16x16 block boundaries)
        # Look for regular grid patterns in gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Check for vertical blocking (every 8 or 16 pixels)
        for block_size in [8, 16]:
            if w > block_size * 4:
                col_edges = np.abs(sobel_x[:, block_size::block_size])
                avg_block_edge = np.mean(col_edges)
                avg_other = np.mean(np.abs(sobel_x))
                if avg_block_edge > avg_other * 1.5 and avg_block_edge > 10:
                    artifacts.append("blocking")
                    break

        # Detect ringing artifacts (oscillating patterns near edges)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        edge_region = cv2.bitwise_and(gray, gray, mask=dilated_edges)

        if np.sum(dilated_edges) > 0:
            # Calculate local variance in edge regions
            edge_pixels = gray[dilated_edges > 0]
            if len(edge_pixels) > 100:
                local_std = np.std(edge_pixels)
                if local_std > 40:  # High variance near edges suggests ringing
                    artifacts.append("ringing")

        # Detect banding (color quantization artifacts)
        # Look for flat regions with sharp transitions
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        flat_regions = diff < 5

        # Count unique values in supposedly flat regions
        if np.sum(flat_regions) > h * w * 0.3:
            flat_values = gray[flat_regions]
            unique_count = len(np.unique(flat_values))
            expected_unique = min(256, len(flat_values) // 100)
            if unique_count < expected_unique * 0.3:
                artifacts.append("banding")

        # Detect mosquito noise (high frequency noise around edges)
        high_freq = cv2.Laplacian(gray, cv2.CV_64F)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        if np.sum(edge_mask) > 0:
            noise_near_edges = np.abs(high_freq[edge_mask > 0])
            if len(noise_near_edges) > 0 and np.mean(noise_near_edges) > 30:
                artifacts.append("mosquito_noise")

        # Detect color bleeding (check for color shifts at edges)
        if len(frame.shape) == 3:
            b, g, r = cv2.split(frame)
            edge_mask_small = cv2.erode(edges, kernel, iterations=1)
            if np.sum(edge_mask_small) > 100:
                # Check color channel alignment at edges
                r_at_edges = r[edge_mask_small > 0]
                g_at_edges = g[edge_mask_small > 0]
                b_at_edges = b[edge_mask_small > 0]

                if len(r_at_edges) > 0:
                    color_variance = np.std([np.mean(r_at_edges), np.mean(g_at_edges), np.mean(b_at_edges)])
                    if color_variance > 50:
                        artifacts.append("color_bleeding")

        return artifacts

    def _calculate_overall_metrics(self, report: QAReport) -> None:
        """Calculate overall quality metrics."""
        if not report.frame_metrics:
            return

        psnr_values = [m.psnr for m in report.frame_metrics]
        ssim_values = [m.ssim for m in report.frame_metrics]

        report.avg_psnr = statistics.mean(psnr_values)
        report.avg_ssim = statistics.mean(ssim_values)
        report.min_psnr = min(psnr_values)
        report.max_psnr = max(psnr_values)
        report.psnr_std_dev = statistics.stdev(psnr_values) if len(psnr_values) > 1 else 0.0

    def _detect_problems(self, report: QAReport) -> None:
        """Detect problem frames and issues."""
        if not report.frame_metrics:
            return

        avg_psnr = report.avg_psnr
        avg_ssim = report.avg_ssim

        for metrics in report.frame_metrics:
            # Flag frames significantly below average
            if metrics.psnr < avg_psnr - 5 or metrics.ssim < avg_ssim - 0.1:
                report.problem_frames.append(metrics.frame_number)

            if metrics.has_artifacts:
                for artifact in metrics.artifact_types:
                    issue = f"Artifact ({artifact}) at frame {metrics.frame_number}"
                    if issue not in report.issues_found:
                        report.issues_found.append(issue)

        # General issues
        if report.avg_psnr < 30:
            report.issues_found.append("Low overall PSNR indicates significant quality loss")

        if report.psnr_std_dev > 5:
            report.issues_found.append("High PSNR variance indicates inconsistent quality")

        if len(report.problem_frames) > len(report.frame_metrics) * 0.1:
            report.issues_found.append("More than 10% of frames have quality issues")

    def _analyze_segments(self, report: QAReport) -> None:
        """Analyze video in segments."""
        if not report.frame_metrics or len(report.frame_metrics) < 10:
            return

        # Divide into 10 segments
        segment_size = len(report.frame_metrics) // 10

        for i in range(10):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < 9 else len(report.frame_metrics)

            segment_metrics = report.frame_metrics[start_idx:end_idx]

            if not segment_metrics:
                continue

            psnr_values = [m.psnr for m in segment_metrics]
            ssim_values = [m.ssim for m in segment_metrics]

            segment = SegmentAnalysis(
                start_frame=segment_metrics[0].frame_number,
                end_frame=segment_metrics[-1].frame_number,
                start_time=segment_metrics[0].timestamp,
                end_time=segment_metrics[-1].timestamp,
                avg_psnr=statistics.mean(psnr_values),
                avg_ssim=statistics.mean(ssim_values),
                min_quality_frame=min(segment_metrics, key=lambda m: m.psnr).frame_number,
                max_quality_frame=max(segment_metrics, key=lambda m: m.psnr).frame_number,
            )

            report.segments.append(segment)

    def _calculate_grade(self, report: QAReport) -> None:
        """Calculate overall quality grade."""
        # Score based on multiple factors
        scores = []

        # PSNR score (target: 35+)
        if report.avg_psnr >= 40:
            scores.append(100)
        elif report.avg_psnr >= 35:
            scores.append(90)
        elif report.avg_psnr >= 30:
            scores.append(70)
        elif report.avg_psnr >= 25:
            scores.append(50)
        else:
            scores.append(30)

        # SSIM score (target: 0.95+)
        if report.avg_ssim >= 0.98:
            scores.append(100)
        elif report.avg_ssim >= 0.95:
            scores.append(90)
        elif report.avg_ssim >= 0.90:
            scores.append(75)
        elif report.avg_ssim >= 0.85:
            scores.append(60)
        else:
            scores.append(40)

        # Consistency score
        if report.psnr_std_dev < 2:
            scores.append(100)
        elif report.psnr_std_dev < 4:
            scores.append(80)
        elif report.psnr_std_dev < 6:
            scores.append(60)
        else:
            scores.append(40)

        # Problem frames penalty
        if report.frame_metrics:
            problem_ratio = len(report.problem_frames) / len(report.frame_metrics)
            scores.append(100 * (1 - problem_ratio))

        report.overall_score = statistics.mean(scores) if scores else 0.0

        # Assign grade
        if report.overall_score >= 95:
            report.overall_grade = QualityGrade.EXCELLENT
        elif report.overall_score >= 85:
            report.overall_grade = QualityGrade.VERY_GOOD
        elif report.overall_score >= 75:
            report.overall_grade = QualityGrade.GOOD
        elif report.overall_score >= 60:
            report.overall_grade = QualityGrade.ACCEPTABLE
        elif report.overall_score >= 40:
            report.overall_grade = QualityGrade.NEEDS_IMPROVEMENT
        else:
            report.overall_grade = QualityGrade.POOR

    def _generate_recommendations(self, report: QAReport) -> None:
        """Generate improvement recommendations."""
        if report.avg_psnr < 30:
            report.recommendations.append(
                "Consider using 'ultimate' preset for higher quality upscaling"
            )

        if report.psnr_std_dev > 5:
            report.recommendations.append(
                "Enable temporal consistency (--temporal-method hybrid) for smoother results"
            )

        if len(report.problem_frames) > 5:
            report.recommendations.append(
                f"Review frames {report.problem_frames[:5]} for quality issues"
            )

        if report.avg_ssim < 0.9:
            report.recommendations.append(
                "Try enabling TAP denoising for better detail preservation"
            )

    def _generate_comparisons(
        self,
        original: Path,
        restored: Path,
        output_dir: Path,
        report: QAReport,
    ) -> None:
        """Generate visual comparison images."""
        if not self._ensure_deps():
            return

        cv2 = self._cv2
        np = self._np

        cap_orig = cv2.VideoCapture(str(original))
        cap_rest = cv2.VideoCapture(str(restored))

        if not cap_orig.isOpened() or not cap_rest.isOpened():
            return

        total_frames = int(cap_rest.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select frames for comparison
        comparison_indices = [
            int(i * total_frames / (self.comparison_frames + 1))
            for i in range(1, self.comparison_frames + 1)
        ]

        comparisons_dir = output_dir / "comparisons"
        comparisons_dir.mkdir(exist_ok=True)

        for idx in comparison_indices:
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_rest.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret_orig, frame_orig = cap_orig.read()
            ret_rest, frame_rest = cap_rest.read()

            if not ret_orig or not ret_rest:
                continue

            # Resize original to match restored if needed
            if frame_orig.shape != frame_rest.shape:
                frame_orig = cv2.resize(frame_orig, (frame_rest.shape[1], frame_rest.shape[0]))

            # Create side-by-side comparison
            comparison = np.hstack([frame_orig, frame_rest])

            # Add labels
            h = comparison.shape[0]
            cv2.putText(comparison, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Restored", (frame_rest.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Save
            output_path = comparisons_dir / f"comparison_frame_{idx:06d}.png"
            cv2.imwrite(str(output_path), comparison)

        cap_orig.release()
        cap_rest.release()

    def export_json(self, report: QAReport, output_path: Path) -> None:
        """Export report as JSON.

        Args:
            report: QA report to export
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"JSON report saved: {output_path}")

    def export_html(self, report: QAReport, output_path: Path) -> None:
        """Export report as HTML.

        Args:
            report: QA report to export
            output_path: Output file path
        """
        html = self._generate_html_report(report)

        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved: {output_path}")

    def _generate_html_report(self, report: QAReport) -> str:
        """Generate HTML report content."""
        grade_colors = {
            QualityGrade.EXCELLENT: "#22c55e",
            QualityGrade.VERY_GOOD: "#84cc16",
            QualityGrade.GOOD: "#eab308",
            QualityGrade.ACCEPTABLE: "#f97316",
            QualityGrade.NEEDS_IMPROVEMENT: "#ef4444",
            QualityGrade.POOR: "#dc2626",
        }

        grade_color = grade_colors.get(report.overall_grade, "#888")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright QA Report - {Path(report.output_file).name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: rgba(255,255,255,0.05);
            border-radius: 1rem;
        }}
        .header h1 {{ color: #c084fc; margin-bottom: 0.5rem; }}
        .grade {{
            font-size: 4rem;
            font-weight: bold;
            color: {grade_color};
            text-shadow: 0 0 20px {grade_color}40;
        }}
        .score {{ font-size: 1.5rem; color: #a1a1aa; }}
        .section {{
            background: rgba(255,255,255,0.03);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .section h2 {{
            color: #c084fc;
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .metric {{
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #c084fc;
        }}
        .metric-label {{ color: #a1a1aa; font-size: 0.875rem; }}
        .issue {{ color: #ef4444; padding: 0.5rem 0; }}
        .recommendation {{ color: #22c55e; padding: 0.5rem 0; }}
        .warning {{ color: #f97316; padding: 0.5rem 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ color: #c084fc; }}
        .footer {{
            text-align: center;
            color: #71717a;
            margin-top: 2rem;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FrameWright Quality Report</h1>
            <div class="grade">{report.overall_grade.value}</div>
            <div class="score">Overall Score: {report.overall_score:.1f}/100</div>
        </div>

        <div class="section">
            <h2>Video Information</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{report.resolution_original}</div>
                    <div class="metric-label">Original Resolution</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.resolution_output}</div>
                    <div class="metric-label">Output Resolution</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.duration_seconds:.1f}s</div>
                    <div class="metric-label">Duration</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.fps_output:.1f}</div>
                    <div class="metric-label">Output FPS</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Quality Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{report.avg_psnr:.2f} dB</div>
                    <div class="metric-label">Average PSNR</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.avg_ssim:.4f}</div>
                    <div class="metric-label">Average SSIM</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.min_psnr:.2f} dB</div>
                    <div class="metric-label">Minimum PSNR</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.psnr_std_dev:.2f}</div>
                    <div class="metric-label">PSNR Std Dev</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Issues Found</h2>
            {"".join(f'<div class="issue">• {issue}</div>' for issue in report.issues_found) or '<div>No significant issues detected</div>'}
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            {"".join(f'<div class="recommendation">• {rec}</div>' for rec in report.recommendations) or '<div>No additional recommendations</div>'}
        </div>

        <div class="section">
            <h2>Processing Statistics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{report.processing_stats.total_frames}</div>
                    <div class="metric-label">Total Frames</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report.processing_stats.processed_frames}</div>
                    <div class="metric-label">Analyzed Frames</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(report.problem_frames)}</div>
                    <div class="metric-label">Problem Frames</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Generated by FrameWright Video Restoration Pipeline</p>
            <p>Report ID: {report.report_id} | {report.created_at}</p>
        </div>
    </div>
</body>
</html>"""

        return html


def generate_qa_report(
    original_path: Path,
    restored_path: Path,
    output_dir: Optional[Path] = None,
    export_formats: List[str] = None,
    **kwargs,
) -> QAReport:
    """Generate QA report with exports.

    Args:
        original_path: Original video path
        restored_path: Restored video path
        output_dir: Output directory
        export_formats: List of export formats (json, html)
        **kwargs: Additional options

    Returns:
        Generated QAReport
    """
    if export_formats is None:
        export_formats = ["json", "html"]

    generator = QAReportGenerator(**kwargs)
    report = generator.generate_report(original_path, restored_path, output_dir)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(restored_path).stem

        if "json" in export_formats:
            generator.export_json(report, output_dir / f"{base_name}_qa_report.json")

        if "html" in export_formats:
            generator.export_html(report, output_dir / f"{base_name}_qa_report.html")

    return report
