"""Quality Scorer - Comprehensive video quality assessment.

Provides detailed quality metrics for frames and videos with support for
both reference-based and no-reference quality assessment methods.

Features:
- Single frame quality scoring
- Full video quality metrics with temporal consistency
- Before/after comparison for restoration evaluation
- Multiple quality metrics (sharpness, noise, BRISQUE, etc.)
- HTML and JSON report generation
- Reference vs no-reference assessment modes

Example:
    >>> scorer = QualityScorer()
    >>> metrics = scorer.score_frame(frame)
    >>> print(f"Overall quality: {metrics.overall_score:.2f}")
    >>>
    >>> # Video analysis
    >>> video_metrics = scorer.score_video(frames)
    >>> print(f"Temporal consistency: {video_metrics.temporal_consistency:.2f}")
    >>>
    >>> # Before/after comparison
    >>> comparison = scorer.compare(original, restored)
    >>> print(f"Improvement: {comparison['improvement']:.2f}%")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available - quality scoring limited")

try:
    from scipy import ndimage, signal
    from scipy.special import gamma
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a frame or video.

    Attributes:
        sharpness: Sharpness score (0-100, higher is sharper)
        noise_level: Noise level (0-100, lower is cleaner)
        contrast: Contrast score (0-100, optimal around 50)
        color_accuracy: Color accuracy score (0-100, higher is better)
        temporal_consistency: Frame-to-frame consistency (0-100, video only)
        face_quality: Face region quality (0-100, if faces detected)
        overall_score: Weighted overall quality score (0-100)
    """
    sharpness: float = 0.0
    noise_level: float = 0.0
    contrast: float = 0.0
    color_accuracy: float = 0.0
    temporal_consistency: float = 100.0
    face_quality: Optional[float] = None
    overall_score: float = 0.0

    # Additional detailed metrics
    brisque_score: Optional[float] = None
    niqe_score: Optional[float] = None
    laplacian_variance: float = 0.0
    brightness: float = 0.0
    saturation: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    blocking_artifact: float = 0.0
    ringing_artifact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sharpness": round(self.sharpness, 2),
            "noise_level": round(self.noise_level, 2),
            "contrast": round(self.contrast, 2),
            "color_accuracy": round(self.color_accuracy, 2),
            "temporal_consistency": round(self.temporal_consistency, 2),
            "face_quality": round(self.face_quality, 2) if self.face_quality else None,
            "overall_score": round(self.overall_score, 2),
            "brisque_score": round(self.brisque_score, 2) if self.brisque_score else None,
            "niqe_score": round(self.niqe_score, 2) if self.niqe_score else None,
            "laplacian_variance": round(self.laplacian_variance, 2),
            "brightness": round(self.brightness, 2),
            "saturation": round(self.saturation, 2),
            "entropy": round(self.entropy, 2),
            "edge_density": round(self.edge_density, 4),
            "blocking_artifact": round(self.blocking_artifact, 2),
            "ringing_artifact": round(self.ringing_artifact, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetrics":
        """Create from dictionary."""
        return cls(
            sharpness=data.get("sharpness", 0.0),
            noise_level=data.get("noise_level", 0.0),
            contrast=data.get("contrast", 0.0),
            color_accuracy=data.get("color_accuracy", 0.0),
            temporal_consistency=data.get("temporal_consistency", 100.0),
            face_quality=data.get("face_quality"),
            overall_score=data.get("overall_score", 0.0),
            brisque_score=data.get("brisque_score"),
            niqe_score=data.get("niqe_score"),
            laplacian_variance=data.get("laplacian_variance", 0.0),
            brightness=data.get("brightness", 0.0),
            saturation=data.get("saturation", 0.0),
            entropy=data.get("entropy", 0.0),
            edge_density=data.get("edge_density", 0.0),
            blocking_artifact=data.get("blocking_artifact", 0.0),
            ringing_artifact=data.get("ringing_artifact", 0.0),
        )


@dataclass
class VideoQualityMetrics:
    """Quality metrics aggregated over a video."""
    frame_count: int = 0
    fps: float = 0.0
    duration: float = 0.0
    resolution: Tuple[int, int] = (0, 0)

    # Aggregated metrics
    average_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    min_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    max_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    std_metrics: Dict[str, float] = field(default_factory=dict)

    # Per-frame data
    frame_scores: List[float] = field(default_factory=list)
    problem_frames: List[int] = field(default_factory=list)  # Frame indices
    scene_quality: Dict[int, float] = field(default_factory=dict)  # Scene -> avg quality

    def get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of frame quality scores."""
        distribution = {
            "excellent (90-100)": 0,
            "good (70-89)": 0,
            "fair (50-69)": 0,
            "poor (30-49)": 0,
            "bad (0-29)": 0,
        }
        for score in self.frame_scores:
            if score >= 90:
                distribution["excellent (90-100)"] += 1
            elif score >= 70:
                distribution["good (70-89)"] += 1
            elif score >= 50:
                distribution["fair (50-69)"] += 1
            elif score >= 30:
                distribution["poor (30-49)"] += 1
            else:
                distribution["bad (0-29)"] += 1
        return distribution


@dataclass
class ComparisonResult:
    """Result of comparing original and restored quality."""
    original_metrics: QualityMetrics
    restored_metrics: QualityMetrics
    improvement_percent: float
    metric_improvements: Dict[str, float]
    is_improved: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original_metrics.to_dict(),
            "restored": self.restored_metrics.to_dict(),
            "improvement_percent": round(self.improvement_percent, 2),
            "metric_improvements": {
                k: round(v, 2) for k, v in self.metric_improvements.items()
            },
            "is_improved": self.is_improved,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Quality Report Generator
# =============================================================================

class QualityReport:
    """Generate quality assessment reports in HTML and JSON formats."""

    def __init__(
        self,
        video_path: Optional[Path] = None,
        video_metrics: Optional[VideoQualityMetrics] = None,
        comparison: Optional[ComparisonResult] = None,
    ):
        """Initialize report generator.

        Args:
            video_path: Path to analyzed video
            video_metrics: Video quality metrics
            comparison: Optional before/after comparison
        """
        self.video_path = video_path
        self.video_metrics = video_metrics
        self.comparison = comparison
        self.timestamp = datetime.now()

    def to_json(self, pretty: bool = True) -> str:
        """Generate JSON report.

        Args:
            pretty: Whether to format with indentation

        Returns:
            JSON string
        """
        report_data = {
            "timestamp": self.timestamp.isoformat(),
            "video_path": str(self.video_path) if self.video_path else None,
        }

        if self.video_metrics:
            report_data["video_info"] = {
                "frame_count": self.video_metrics.frame_count,
                "fps": self.video_metrics.fps,
                "duration": self.video_metrics.duration,
                "resolution": list(self.video_metrics.resolution),
            }
            report_data["average_metrics"] = self.video_metrics.average_metrics.to_dict()
            report_data["quality_distribution"] = self.video_metrics.get_quality_distribution()
            report_data["problem_frame_count"] = len(self.video_metrics.problem_frames)

        if self.comparison:
            report_data["comparison"] = self.comparison.to_dict()

        return json.dumps(report_data, indent=2 if pretty else None)

    def save_json(self, output_path: Path) -> None:
        """Save JSON report to file.

        Args:
            output_path: Path for output file
        """
        with open(output_path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved JSON report to {output_path}")

    def to_html(self) -> str:
        """Generate HTML report.

        Returns:
            HTML string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>Video Quality Report</title>",
            "<style>",
            self._get_css_styles(),
            "</style>",
            "</head><body>",
            "<div class='container'>",
            f"<h1>Video Quality Report</h1>",
            f"<p class='timestamp'>Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        if self.video_path:
            html_parts.append(f"<p class='video-path'>Video: {self.video_path.name}</p>")

        if self.video_metrics:
            html_parts.append(self._generate_video_section())

        if self.comparison:
            html_parts.append(self._generate_comparison_section())

        html_parts.extend([
            "</div>",
            "</body></html>",
        ])

        return "\n".join(html_parts)

    def save_html(self, output_path: Path) -> None:
        """Save HTML report to file.

        Args:
            output_path: Path for output file
        """
        with open(output_path, 'w') as f:
            f.write(self.to_html())
        logger.info(f"Saved HTML report to {output_path}")

    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white;
                         padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; margin-bottom: 5px; }
            h2 { color: #555; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .timestamp { color: #888; font-size: 0.9em; }
            .video-path { color: #666; font-size: 0.95em; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                            gap: 15px; margin: 20px 0; }
            .metric-card { background: #f9f9f9; padding: 15px; border-radius: 6px; }
            .metric-label { font-size: 0.85em; color: #666; margin-bottom: 5px; }
            .metric-value { font-size: 1.5em; font-weight: 600; color: #333; }
            .metric-good { color: #28a745; }
            .metric-warning { color: #ffc107; }
            .metric-bad { color: #dc3545; }
            .bar-container { background: #e0e0e0; border-radius: 4px; height: 8px; margin-top: 8px; }
            .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
            .improvement { font-size: 1.2em; padding: 15px; background: #e8f5e9;
                           border-radius: 6px; margin: 20px 0; }
            .improvement.negative { background: #ffebee; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f5f5f5; font-weight: 600; }
            .warning { color: #856404; background: #fff3cd; padding: 10px;
                       border-radius: 4px; margin: 10px 0; }
        """

    def _generate_video_section(self) -> str:
        """Generate video metrics HTML section."""
        metrics = self.video_metrics
        avg = metrics.average_metrics

        html = [
            "<section>",
            "<h2>Video Information</h2>",
            "<div class='metrics-grid'>",
            f"<div class='metric-card'><div class='metric-label'>Resolution</div>"
            f"<div class='metric-value'>{metrics.resolution[0]}x{metrics.resolution[1]}</div></div>",
            f"<div class='metric-card'><div class='metric-label'>Duration</div>"
            f"<div class='metric-value'>{metrics.duration:.1f}s</div></div>",
            f"<div class='metric-card'><div class='metric-label'>Frame Count</div>"
            f"<div class='metric-value'>{metrics.frame_count}</div></div>",
            f"<div class='metric-card'><div class='metric-label'>FPS</div>"
            f"<div class='metric-value'>{metrics.fps:.2f}</div></div>",
            "</div>",
            "<h2>Quality Metrics</h2>",
            "<div class='metrics-grid'>",
        ]

        # Add metric cards
        metrics_to_show = [
            ("Overall Score", avg.overall_score),
            ("Sharpness", avg.sharpness),
            ("Noise Level", 100 - avg.noise_level),  # Invert for display
            ("Contrast", avg.contrast),
            ("Color Accuracy", avg.color_accuracy),
            ("Temporal Consistency", avg.temporal_consistency),
        ]

        for label, value in metrics_to_show:
            color_class = self._get_color_class(value)
            html.append(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value {color_class}'>{value:.1f}</div>"
                f"<div class='bar-container'>"
                f"<div class='bar-fill {color_class}' style='width: {value}%'></div>"
                f"</div></div>"
            )

        html.append("</div>")

        # Quality distribution
        dist = metrics.get_quality_distribution()
        html.extend([
            "<h2>Quality Distribution</h2>",
            "<table>",
            "<tr><th>Range</th><th>Frame Count</th><th>Percentage</th></tr>",
        ])
        total = max(1, sum(dist.values()))
        for label, count in dist.items():
            pct = (count / total) * 100
            html.append(f"<tr><td>{label}</td><td>{count}</td><td>{pct:.1f}%</td></tr>")
        html.append("</table>")

        # Problem frames
        if metrics.problem_frames:
            html.extend([
                "<h2>Problem Frames</h2>",
                f"<p>Found {len(metrics.problem_frames)} frames with quality issues.</p>",
            ])

        html.append("</section>")
        return "\n".join(html)

    def _generate_comparison_section(self) -> str:
        """Generate comparison HTML section."""
        comp = self.comparison

        improvement_class = "" if comp.is_improved else "negative"
        improvement_text = "improved" if comp.is_improved else "degraded"

        html = [
            "<section>",
            "<h2>Before/After Comparison</h2>",
            f"<div class='improvement {improvement_class}'>",
            f"Overall quality {improvement_text} by <strong>{abs(comp.improvement_percent):.1f}%</strong>",
            "</div>",
            "<table>",
            "<tr><th>Metric</th><th>Original</th><th>Restored</th><th>Change</th></tr>",
        ]

        metrics_to_compare = [
            ("Overall Score", "overall_score"),
            ("Sharpness", "sharpness"),
            ("Noise Level", "noise_level"),
            ("Contrast", "contrast"),
            ("Color Accuracy", "color_accuracy"),
        ]

        for label, attr in metrics_to_compare:
            orig = getattr(comp.original_metrics, attr)
            rest = getattr(comp.restored_metrics, attr)
            change = comp.metric_improvements.get(attr, 0)
            change_str = f"+{change:.1f}" if change > 0 else f"{change:.1f}"
            change_class = "metric-good" if change > 0 else "metric-bad" if change < 0 else ""
            html.append(
                f"<tr><td>{label}</td><td>{orig:.1f}</td><td>{rest:.1f}</td>"
                f"<td class='{change_class}'>{change_str}</td></tr>"
            )

        html.append("</table>")

        # Warnings
        if comp.warnings:
            html.append("<h3>Warnings</h3>")
            for warning in comp.warnings:
                html.append(f"<div class='warning'>{warning}</div>")

        # Recommendations
        if comp.recommendations:
            html.append("<h3>Recommendations</h3>")
            html.append("<ul>")
            for rec in comp.recommendations:
                html.append(f"<li>{rec}</li>")
            html.append("</ul>")

        html.append("</section>")
        return "\n".join(html)

    def _get_color_class(self, value: float) -> str:
        """Get CSS color class based on value."""
        if value >= 70:
            return "metric-good"
        elif value >= 40:
            return "metric-warning"
        return "metric-bad"


# =============================================================================
# Quality Scorer Class
# =============================================================================

class QualityScorer:
    """Comprehensive video quality scorer with multiple metrics.

    Supports both reference-based (comparing to original) and no-reference
    (blind) quality assessment methods.
    """

    # Weights for overall score calculation
    DEFAULT_WEIGHTS = {
        "sharpness": 0.25,
        "noise_level": 0.20,
        "contrast": 0.15,
        "color_accuracy": 0.15,
        "temporal_consistency": 0.15,
        "face_quality": 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        enable_brisque: bool = True,
        enable_face_detection: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize quality scorer.

        Args:
            weights: Custom weights for overall score calculation
            enable_brisque: Enable BRISQUE no-reference quality metric
            enable_face_detection: Enable face detection for face quality scoring
            progress_callback: Optional callback for progress reporting
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.enable_brisque = enable_brisque and HAS_SCIPY
        self.enable_face_detection = enable_face_detection
        self.progress_callback = progress_callback

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Face detector (lazy initialization)
        self._face_cascade = None

        if not HAS_CV2:
            logger.warning("OpenCV required for quality scoring")

    # =========================================================================
    # Single Frame Scoring
    # =========================================================================

    def score_frame(self, frame: np.ndarray) -> QualityMetrics:
        """Score quality of a single frame.

        Args:
            frame: BGR frame as numpy array

        Returns:
            QualityMetrics with all calculated metrics
        """
        if not HAS_CV2:
            return QualityMetrics()

        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided")
            return QualityMetrics()

        metrics = QualityMetrics()

        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate individual metrics
        metrics.sharpness = self._calculate_sharpness(gray)
        metrics.laplacian_variance = self._calculate_laplacian_variance(gray)
        metrics.noise_level = self._calculate_noise(gray)
        metrics.contrast = self._calculate_contrast(gray)
        metrics.brightness = self._calculate_brightness(gray)
        metrics.color_accuracy = self._calculate_color_accuracy(frame)
        metrics.saturation = self._calculate_saturation(frame)
        metrics.entropy = self._calculate_entropy(gray)
        metrics.edge_density = self._calculate_edge_density(gray)
        metrics.blocking_artifact = self._calculate_blocking(gray)
        metrics.ringing_artifact = self._calculate_ringing(gray)

        # BRISQUE score (no-reference quality)
        if self.enable_brisque:
            metrics.brisque_score = self._calculate_brisque(gray)

        # Face quality
        if self.enable_face_detection:
            metrics.face_quality = self._calculate_face_quality(frame, gray)

        # Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics)

        return metrics

    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness score using Laplacian variance.

        Higher variance indicates sharper images.

        Args:
            gray: Grayscale image

        Returns:
            Sharpness score (0-100)
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to 0-100 (typical range 0-2000 for sharp images)
        score = min(100, (variance / 500) * 100)
        return float(score)

    def _calculate_laplacian_variance(self, gray: np.ndarray) -> float:
        """Calculate raw Laplacian variance.

        Args:
            gray: Grayscale image

        Returns:
            Laplacian variance value
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _calculate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level using high-frequency analysis.

        Args:
            gray: Grayscale image

        Returns:
            Noise level (0-100, lower is cleaner)
        """
        # Use median absolute deviation for robust noise estimation
        # Apply Laplacian and measure noise in flat regions
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_map = cv2.absdiff(gray, blurred)

        # Use median absolute deviation (robust to outliers)
        median = np.median(noise_map)
        mad = np.median(np.abs(noise_map - median))

        # Estimate sigma and scale to 0-100
        sigma = mad / 0.6745
        noise_level = min(100, sigma * 4)

        return float(noise_level)

    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate contrast score.

        Args:
            gray: Grayscale image

        Returns:
            Contrast score (0-100, optimal around 50-70)
        """
        # Calculate contrast as standard deviation of pixel values
        std = np.std(gray.astype(np.float64))

        # Also check histogram spread
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Find 5th and 95th percentile
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        p5 = np.searchsorted(cumsum, total * 0.05)
        p95 = np.searchsorted(cumsum, total * 0.95)
        dynamic_range = p95 - p5

        # Combine std and dynamic range
        contrast_score = (std / 64) * 50 + (dynamic_range / 255) * 50
        contrast_score = min(100, contrast_score)

        return float(contrast_score)

    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate average brightness.

        Args:
            gray: Grayscale image

        Returns:
            Brightness value (0-255)
        """
        return float(np.mean(gray))

    def _calculate_color_accuracy(self, frame: np.ndarray) -> float:
        """Calculate color accuracy score.

        Analyzes color distribution, saturation, and white balance.

        Args:
            frame: BGR frame

        Returns:
            Color accuracy score (0-100)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Check for color cast (a and b channels should average around 128)
        a_mean = np.mean(lab[:, :, 1])
        b_mean = np.mean(lab[:, :, 2])

        # Distance from neutral
        color_cast = np.sqrt((a_mean - 128) ** 2 + (b_mean - 128) ** 2)

        # Penalize strong color casts
        color_cast_penalty = min(50, color_cast)

        # Check color variety
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])

        # Score based on color variety and balance
        variety_score = min(50, h_std)
        saturation_score = min(30, s_mean / 8)

        # Combine scores
        score = 100 - color_cast_penalty + variety_score / 2
        score = max(0, min(100, score))

        return float(score)

    def _calculate_saturation(self, frame: np.ndarray) -> float:
        """Calculate average saturation.

        Args:
            frame: BGR frame

        Returns:
            Saturation value (0-255)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))

    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate image entropy (information content).

        Args:
            gray: Grayscale image

        Returns:
            Entropy value (0-8)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Normalize
        hist = hist / hist.sum()

        # Calculate entropy
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density (ratio of edge pixels).

        Args:
            gray: Grayscale image

        Returns:
            Edge density (0-1)
        """
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = gray.size

        return float(edge_pixels / total_pixels)

    def _calculate_blocking(self, gray: np.ndarray) -> float:
        """Detect blocking artifacts (compression artifacts).

        Args:
            gray: Grayscale image

        Returns:
            Blocking artifact level (0-100, lower is better)
        """
        h, w = gray.shape

        if h < 16 or w < 16:
            return 0.0

        # Calculate differences at 8-pixel boundaries (JPEG block size)
        block_diff = 0.0
        count = 0

        # Horizontal boundaries
        for i in range(8, h - 8, 8):
            diff = np.abs(gray[i, :].astype(float) - gray[i - 1, :].astype(float))
            block_diff += np.mean(diff)
            count += 1

        # Vertical boundaries
        for j in range(8, w - 8, 8):
            diff = np.abs(gray[:, j].astype(float) - gray[:, j - 1].astype(float))
            block_diff += np.mean(diff)
            count += 1

        if count > 0:
            block_diff /= count

        # Compare to overall edge strength
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        overall_edges = np.mean(np.abs(edges))

        if overall_edges > 0:
            blocking_ratio = block_diff / overall_edges
        else:
            blocking_ratio = 0

        # Convert to 0-100 scale (higher = more blocking)
        blocking_score = min(100, max(0, (blocking_ratio - 0.8) * 100))

        return float(blocking_score)

    def _calculate_ringing(self, gray: np.ndarray) -> float:
        """Detect ringing/Gibbs artifacts near edges.

        Args:
            gray: Grayscale image

        Returns:
            Ringing artifact level (0-100, lower is better)
        """
        # Detect strong edges
        edges = cv2.Canny(gray, 100, 200)

        # Dilate edges to get surrounding regions
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2)
        edge_region = edge_region > 0

        # Look for oscillations near edges
        laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

        if np.any(edge_region):
            edge_laplacian_var = np.var(laplacian[edge_region])
            non_edge_mask = ~edge_region
            if np.any(non_edge_mask):
                non_edge_laplacian_var = np.var(laplacian[non_edge_mask])
            else:
                non_edge_laplacian_var = 1

            if non_edge_laplacian_var > 0:
                ringing_ratio = edge_laplacian_var / non_edge_laplacian_var
            else:
                ringing_ratio = 1
        else:
            ringing_ratio = 1

        # Score (higher ratio = more ringing)
        ringing_score = min(100, max(0, (ringing_ratio - 1) * 20))

        return float(ringing_score)

    def _calculate_brisque(self, gray: np.ndarray) -> Optional[float]:
        """Calculate BRISQUE no-reference quality score.

        Blind/Referenceless Image Spatial Quality Evaluator.

        Args:
            gray: Grayscale image

        Returns:
            BRISQUE score (lower is better, typically 0-100)
        """
        if not HAS_SCIPY:
            return None

        try:
            # Normalize image
            gray_f = gray.astype(np.float64)

            # Apply local mean normalization
            kernel_size = 7
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

            local_mean = ndimage.convolve(gray_f, kernel)
            local_var = ndimage.convolve(gray_f ** 2, kernel) - local_mean ** 2
            local_var = np.maximum(local_var, 0)
            local_std = np.sqrt(local_var) + 1e-7

            # MSCN coefficients
            mscn = (gray_f - local_mean) / local_std

            # Calculate generalized Gaussian parameters
            alpha, sigma_sq = self._estimate_ggd_params(mscn.flatten())

            # Feature: alpha and sigma
            # Lower alpha and higher sigma typically indicate poorer quality
            quality_estimate = alpha / (sigma_sq + 0.01)

            # Normalize to 0-100 scale (inverted: lower BRISQUE = better quality)
            brisque_score = max(0, min(100, 100 - quality_estimate * 10))

            return float(brisque_score)

        except Exception as e:
            logger.debug(f"BRISQUE calculation failed: {e}")
            return None

    def _estimate_ggd_params(self, data: np.ndarray) -> Tuple[float, float]:
        """Estimate generalized Gaussian distribution parameters.

        Args:
            data: Flattened image data

        Returns:
            Tuple of (alpha, sigma_squared)
        """
        # Remove outliers
        data = data[np.abs(data) < 3]

        if len(data) < 100:
            return 1.0, 1.0

        # Estimate parameters using method of moments
        abs_data = np.abs(data)
        m1 = np.mean(abs_data)
        m2 = np.mean(data ** 2)

        if m1 == 0:
            return 1.0, 1.0

        rho = m2 / (m1 ** 2)

        # Estimate alpha (shape parameter)
        # Using approximation: rho = gamma(3/alpha) * gamma(1/alpha) / gamma(2/alpha)^2
        alpha = 1.0
        if rho > 0.5 and rho < 3:
            alpha = 1 / np.log(rho + 0.5)
            alpha = max(0.1, min(10, alpha))

        sigma_sq = m2

        return alpha, sigma_sq

    def _calculate_face_quality(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[float]:
        """Calculate quality of face regions.

        Args:
            frame: BGR frame
            gray: Grayscale image

        Returns:
            Face quality score (0-100) or None if no faces
        """
        # Initialize face detector
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)

        # Detect faces
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        face_scores = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_gray = gray[y:y + h, x:x + w]

            # Calculate sharpness of face region
            sharpness = self._calculate_sharpness(face_gray)

            # Calculate noise in face region
            noise = self._calculate_noise(face_gray)

            # Calculate brightness
            brightness = np.mean(face_gray)
            brightness_score = 100 - abs(brightness - 128) / 1.28

            # Combine scores
            face_score = (sharpness * 0.5 + (100 - noise) * 0.3 + brightness_score * 0.2)
            face_scores.append(face_score)

        return float(np.mean(face_scores)) if face_scores else None

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score.

        Args:
            metrics: Individual quality metrics

        Returns:
            Overall score (0-100)
        """
        score = 0.0

        # Sharpness (higher is better)
        if "sharpness" in self.weights:
            score += metrics.sharpness * self.weights["sharpness"]

        # Noise (lower is better, so invert)
        if "noise_level" in self.weights:
            score += (100 - metrics.noise_level) * self.weights["noise_level"]

        # Contrast
        if "contrast" in self.weights:
            score += metrics.contrast * self.weights["contrast"]

        # Color accuracy
        if "color_accuracy" in self.weights:
            score += metrics.color_accuracy * self.weights["color_accuracy"]

        # Temporal consistency (only for video)
        if "temporal_consistency" in self.weights:
            score += metrics.temporal_consistency * self.weights["temporal_consistency"]

        # Face quality (if detected)
        if "face_quality" in self.weights and metrics.face_quality is not None:
            score += metrics.face_quality * self.weights["face_quality"]
        elif "face_quality" in self.weights:
            # Redistribute face weight if no faces
            remaining_weight = self.weights["face_quality"]
            other_weights = sum(
                w for k, w in self.weights.items() if k != "face_quality"
            )
            if other_weights > 0:
                redistribution = remaining_weight / other_weights
                score *= (1 + redistribution)

        return float(min(100, max(0, score)))

    # =========================================================================
    # Video Scoring
    # =========================================================================

    def score_video(
        self,
        frames: Union[List[np.ndarray], Path],
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
    ) -> VideoQualityMetrics:
        """Score quality of an entire video.

        Args:
            frames: List of BGR frames or path to video file
            sample_rate: Analyze every Nth frame (1 = all frames)
            max_frames: Maximum frames to analyze (None = all)

        Returns:
            VideoQualityMetrics with aggregated results
        """
        if not HAS_CV2:
            return VideoQualityMetrics()

        # Handle video file path
        if isinstance(frames, Path):
            return self._score_video_file(frames, sample_rate, max_frames)

        return self._score_frame_list(frames, sample_rate, max_frames)

    def _score_video_file(
        self,
        video_path: Path,
        sample_rate: int,
        max_frames: Optional[int],
    ) -> VideoQualityMetrics:
        """Score video from file.

        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame
            max_frames: Maximum frames to analyze

        Returns:
            VideoQualityMetrics
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return VideoQualityMetrics()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = VideoQualityMetrics(
            frame_count=total_frames,
            fps=fps,
            duration=total_frames / fps if fps > 0 else 0,
            resolution=(width, height),
        )

        frame_metrics: List[QualityMetrics] = []
        prev_frame = None
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                if max_frames and analyzed >= max_frames:
                    break

                metrics = self.score_frame(frame)

                # Calculate temporal consistency
                if prev_frame is not None:
                    metrics.temporal_consistency = self._calculate_temporal_consistency(
                        frame, prev_frame
                    )

                frame_metrics.append(metrics)
                result.frame_scores.append(metrics.overall_score)

                # Track problem frames
                if metrics.overall_score < 50:
                    result.problem_frames.append(frame_idx)

                analyzed += 1
                prev_frame = frame.copy()

                if self.progress_callback:
                    self.progress_callback(analyzed, total_frames // sample_rate)

            frame_idx += 1

        cap.release()

        # Aggregate metrics
        if frame_metrics:
            result.average_metrics = self._aggregate_metrics(frame_metrics)
            result.min_metrics, result.max_metrics = self._get_min_max_metrics(frame_metrics)
            result.std_metrics = self._calculate_std_metrics(frame_metrics)

        return result

    def _score_frame_list(
        self,
        frames: List[np.ndarray],
        sample_rate: int,
        max_frames: Optional[int],
    ) -> VideoQualityMetrics:
        """Score video from frame list.

        Args:
            frames: List of BGR frames
            sample_rate: Analyze every Nth frame
            max_frames: Maximum frames to analyze

        Returns:
            VideoQualityMetrics
        """
        if not frames:
            return VideoQualityMetrics()

        h, w = frames[0].shape[:2]

        result = VideoQualityMetrics(
            frame_count=len(frames),
            resolution=(w, h),
        )

        frame_metrics: List[QualityMetrics] = []
        prev_frame = None

        for i, frame in enumerate(frames):
            if i % sample_rate != 0:
                continue

            if max_frames and len(frame_metrics) >= max_frames:
                break

            metrics = self.score_frame(frame)

            if prev_frame is not None:
                metrics.temporal_consistency = self._calculate_temporal_consistency(
                    frame, prev_frame
                )

            frame_metrics.append(metrics)
            result.frame_scores.append(metrics.overall_score)

            if metrics.overall_score < 50:
                result.problem_frames.append(i)

            prev_frame = frame

            if self.progress_callback:
                self.progress_callback(len(frame_metrics), len(frames) // sample_rate)

        if frame_metrics:
            result.average_metrics = self._aggregate_metrics(frame_metrics)
            result.min_metrics, result.max_metrics = self._get_min_max_metrics(frame_metrics)
            result.std_metrics = self._calculate_std_metrics(frame_metrics)

        return result

    def _calculate_temporal_consistency(
        self,
        current: np.ndarray,
        previous: np.ndarray,
    ) -> float:
        """Calculate temporal consistency between frames.

        Args:
            current: Current frame
            previous: Previous frame

        Returns:
            Consistency score (0-100, higher is more consistent)
        """
        # Convert to grayscale
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM-like metric
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        curr_f = curr_gray.astype(np.float64)
        prev_f = prev_gray.astype(np.float64)

        mu1 = cv2.GaussianBlur(curr_f, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(prev_f, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(curr_f ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(prev_f ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(curr_f * prev_f, (11, 11), 1.5) - mu1_mu2

        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        # Convert to 0-100 scale
        consistency = float(np.mean(ssim) * 100)
        return max(0, min(100, consistency))

    def _aggregate_metrics(self, metrics: List[QualityMetrics]) -> QualityMetrics:
        """Aggregate frame metrics into average.

        Args:
            metrics: List of frame metrics

        Returns:
            Averaged QualityMetrics
        """
        if not metrics:
            return QualityMetrics()

        n = len(metrics)
        return QualityMetrics(
            sharpness=sum(m.sharpness for m in metrics) / n,
            noise_level=sum(m.noise_level for m in metrics) / n,
            contrast=sum(m.contrast for m in metrics) / n,
            color_accuracy=sum(m.color_accuracy for m in metrics) / n,
            temporal_consistency=sum(m.temporal_consistency for m in metrics) / n,
            face_quality=self._avg_optional([m.face_quality for m in metrics]),
            overall_score=sum(m.overall_score for m in metrics) / n,
            brisque_score=self._avg_optional([m.brisque_score for m in metrics]),
            laplacian_variance=sum(m.laplacian_variance for m in metrics) / n,
            brightness=sum(m.brightness for m in metrics) / n,
            saturation=sum(m.saturation for m in metrics) / n,
            entropy=sum(m.entropy for m in metrics) / n,
            edge_density=sum(m.edge_density for m in metrics) / n,
            blocking_artifact=sum(m.blocking_artifact for m in metrics) / n,
            ringing_artifact=sum(m.ringing_artifact for m in metrics) / n,
        )

    def _avg_optional(self, values: List[Optional[float]]) -> Optional[float]:
        """Average optional values, ignoring None."""
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    def _get_min_max_metrics(
        self,
        metrics: List[QualityMetrics],
    ) -> Tuple[QualityMetrics, QualityMetrics]:
        """Get minimum and maximum metrics.

        Args:
            metrics: List of frame metrics

        Returns:
            Tuple of (min_metrics, max_metrics)
        """
        if not metrics:
            return QualityMetrics(), QualityMetrics()

        min_m = QualityMetrics(
            sharpness=min(m.sharpness for m in metrics),
            noise_level=min(m.noise_level for m in metrics),
            contrast=min(m.contrast for m in metrics),
            color_accuracy=min(m.color_accuracy for m in metrics),
            temporal_consistency=min(m.temporal_consistency for m in metrics),
            overall_score=min(m.overall_score for m in metrics),
        )

        max_m = QualityMetrics(
            sharpness=max(m.sharpness for m in metrics),
            noise_level=max(m.noise_level for m in metrics),
            contrast=max(m.contrast for m in metrics),
            color_accuracy=max(m.color_accuracy for m in metrics),
            temporal_consistency=max(m.temporal_consistency for m in metrics),
            overall_score=max(m.overall_score for m in metrics),
        )

        return min_m, max_m

    def _calculate_std_metrics(self, metrics: List[QualityMetrics]) -> Dict[str, float]:
        """Calculate standard deviation of metrics.

        Args:
            metrics: List of frame metrics

        Returns:
            Dictionary of metric name -> std dev
        """
        if not metrics:
            return {}

        return {
            "sharpness": float(np.std([m.sharpness for m in metrics])),
            "noise_level": float(np.std([m.noise_level for m in metrics])),
            "contrast": float(np.std([m.contrast for m in metrics])),
            "color_accuracy": float(np.std([m.color_accuracy for m in metrics])),
            "temporal_consistency": float(np.std([m.temporal_consistency for m in metrics])),
            "overall_score": float(np.std([m.overall_score for m in metrics])),
        }

    # =========================================================================
    # Comparison
    # =========================================================================

    def compare(
        self,
        original: Union[np.ndarray, List[np.ndarray]],
        restored: Union[np.ndarray, List[np.ndarray]],
    ) -> ComparisonResult:
        """Compare original and restored quality.

        Args:
            original: Original frame(s)
            restored: Restored frame(s)

        Returns:
            ComparisonResult with improvement analysis
        """
        # Score both versions
        if isinstance(original, np.ndarray):
            orig_metrics = self.score_frame(original)
            rest_metrics = self.score_frame(restored)
        else:
            orig_video = self.score_video(original)
            rest_video = self.score_video(restored)
            orig_metrics = orig_video.average_metrics
            rest_metrics = rest_video.average_metrics

        # Calculate improvements
        metric_improvements = {}
        for attr in ["sharpness", "contrast", "color_accuracy", "overall_score"]:
            orig_val = getattr(orig_metrics, attr)
            rest_val = getattr(rest_metrics, attr)
            metric_improvements[attr] = rest_val - orig_val

        # Noise level (improvement is reduction)
        metric_improvements["noise_level"] = orig_metrics.noise_level - rest_metrics.noise_level

        # Overall improvement
        overall_improvement = rest_metrics.overall_score - orig_metrics.overall_score
        is_improved = overall_improvement > 0

        # Generate warnings and recommendations
        warnings = []
        recommendations = []

        # Check for over-sharpening
        if metric_improvements["sharpness"] > 30:
            warnings.append("Sharpening may be too aggressive, causing halos")
            recommendations.append("Consider reducing sharpening strength")

        # Check for over-denoising
        if metric_improvements["noise_level"] > 30 and rest_metrics.sharpness < orig_metrics.sharpness - 10:
            warnings.append("Denoising may have removed too much detail")
            recommendations.append("Consider reducing denoise strength or using temporal denoising")

        # Check for color shift
        if abs(rest_metrics.color_accuracy - orig_metrics.color_accuracy) > 15:
            warnings.append("Noticeable color shift detected")
            recommendations.append("Review colorization/color correction settings")

        # Check for artifacts
        if rest_metrics.blocking_artifact > orig_metrics.blocking_artifact + 10:
            warnings.append("Compression artifacts increased")
            recommendations.append("Use higher quality output codec settings")

        if not is_improved:
            warnings.append("Overall quality did not improve")
            recommendations.append("Review processing settings and consider alternative approaches")

        return ComparisonResult(
            original_metrics=orig_metrics,
            restored_metrics=rest_metrics,
            improvement_percent=overall_improvement,
            metric_improvements=metric_improvements,
            is_improved=is_improved,
            warnings=warnings,
            recommendations=recommendations,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def score_frame(frame: np.ndarray) -> QualityMetrics:
    """Score a single frame quality.

    Args:
        frame: BGR frame

    Returns:
        QualityMetrics
    """
    scorer = QualityScorer()
    return scorer.score_frame(frame)


def score_video(video_path: Path, sample_rate: int = 1) -> VideoQualityMetrics:
    """Score video quality.

    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame

    Returns:
        VideoQualityMetrics
    """
    scorer = QualityScorer()
    return scorer.score_video(video_path, sample_rate=sample_rate)


def compare_quality(
    original: np.ndarray,
    restored: np.ndarray,
) -> ComparisonResult:
    """Compare original and restored frame quality.

    Args:
        original: Original frame
        restored: Restored frame

    Returns:
        ComparisonResult
    """
    scorer = QualityScorer()
    return scorer.compare(original, restored)


def generate_quality_report(
    video_path: Path,
    output_path: Optional[Path] = None,
    format: str = "html",
) -> str:
    """Generate quality report for a video.

    Args:
        video_path: Path to video file
        output_path: Optional path to save report
        format: Report format ("html" or "json")

    Returns:
        Report content as string
    """
    scorer = QualityScorer()
    metrics = scorer.score_video(video_path)

    report = QualityReport(
        video_path=video_path,
        video_metrics=metrics,
    )

    if format == "html":
        content = report.to_html()
        if output_path:
            report.save_html(output_path)
    else:
        content = report.to_json()
        if output_path:
            report.save_json(output_path)

    return content
