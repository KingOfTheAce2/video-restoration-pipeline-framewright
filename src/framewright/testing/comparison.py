"""Comparison engine for A/B testing restoration approaches."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricsDiff:
    """Difference in quality metrics between two variants."""
    psnr_diff: float = 0.0
    ssim_diff: float = 0.0
    vmaf_diff: Optional[float] = None

    # Per-frame analysis
    psnr_improvements: int = 0  # Frames where variant B is better
    psnr_regressions: int = 0  # Frames where variant A is better
    ssim_improvements: int = 0
    ssim_regressions: int = 0

    # Statistical significance
    psnr_significant: bool = False
    ssim_significant: bool = False
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "psnr_diff": self.psnr_diff,
            "ssim_diff": self.ssim_diff,
            "vmaf_diff": self.vmaf_diff,
            "psnr_improvements": self.psnr_improvements,
            "psnr_regressions": self.psnr_regressions,
            "ssim_improvements": self.ssim_improvements,
            "ssim_regressions": self.ssim_regressions,
            "psnr_significant": self.psnr_significant,
            "ssim_significant": self.ssim_significant,
            "confidence_level": self.confidence_level,
        }


@dataclass
class VisualDiff:
    """Visual difference analysis between two variants."""
    diff_map: Optional[np.ndarray] = None
    diff_magnitude: float = 0.0

    # Region analysis
    regions_improved: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    regions_degraded: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Perceptual analysis
    perceptual_diff: float = 0.0
    edge_preservation_diff: float = 0.0
    texture_preservation_diff: float = 0.0
    color_accuracy_diff: float = 0.0

    # Face-specific (if applicable)
    face_quality_diff: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "diff_magnitude": self.diff_magnitude,
            "regions_improved_count": len(self.regions_improved),
            "regions_degraded_count": len(self.regions_degraded),
            "perceptual_diff": self.perceptual_diff,
            "edge_preservation_diff": self.edge_preservation_diff,
            "texture_preservation_diff": self.texture_preservation_diff,
            "color_accuracy_diff": self.color_accuracy_diff,
            "face_quality_diff": self.face_quality_diff,
        }


@dataclass
class ComparisonResult:
    """Complete comparison result between two variants."""
    variant_a_name: str
    variant_b_name: str

    # Overall winner
    winner: str = ""  # "A", "B", or "tie"
    confidence: float = 0.0

    # Component analysis
    metrics_diff: MetricsDiff = field(default_factory=MetricsDiff)
    visual_diff: VisualDiff = field(default_factory=VisualDiff)

    # Per-frame results
    frame_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    # Processing comparison
    processing_time_diff: float = 0.0  # Seconds (B - A)
    vram_usage_diff: float = 0.0  # MB (B - A)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_a": self.variant_a_name,
            "variant_b": self.variant_b_name,
            "winner": self.winner,
            "confidence": self.confidence,
            "metrics_diff": self.metrics_diff.to_dict(),
            "visual_diff": self.visual_diff.to_dict(),
            "frame_comparisons_count": len(self.frame_comparisons),
            "processing_time_diff_seconds": self.processing_time_diff,
            "vram_usage_diff_mb": self.vram_usage_diff,
            "recommendations": self.recommendations,
        }


class ComparisonEngine:
    """Engine for comparing restoration results."""

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_improvement_db: float = 0.5,
        min_ssim_improvement: float = 0.01,
    ):
        self.significance_threshold = significance_threshold
        self.min_improvement_db = min_improvement_db
        self.min_ssim_improvement = min_ssim_improvement

    def compare_frames(
        self,
        original: np.ndarray,
        variant_a: np.ndarray,
        variant_b: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare two restored frames against original."""
        result = {
            "a_psnr": self._calculate_psnr(original, variant_a),
            "b_psnr": self._calculate_psnr(original, variant_b),
            "a_ssim": self._calculate_ssim(original, variant_a),
            "b_ssim": self._calculate_ssim(original, variant_b),
        }

        result["psnr_winner"] = "B" if result["b_psnr"] > result["a_psnr"] else "A"
        result["ssim_winner"] = "B" if result["b_ssim"] > result["a_ssim"] else "A"

        # Calculate difference map
        diff_ab = np.abs(variant_a.astype(float) - variant_b.astype(float))
        result["diff_magnitude"] = float(np.mean(diff_ab))

        return result

    def compare_variants(
        self,
        original_frames: List[np.ndarray],
        variant_a_frames: List[np.ndarray],
        variant_b_frames: List[np.ndarray],
        variant_a_name: str = "Variant A",
        variant_b_name: str = "Variant B",
    ) -> ComparisonResult:
        """Compare two complete restoration variants."""
        result = ComparisonResult(
            variant_a_name=variant_a_name,
            variant_b_name=variant_b_name,
        )

        # Per-frame comparison
        a_psnrs = []
        b_psnrs = []
        a_ssims = []
        b_ssims = []

        for i, (orig, a, b) in enumerate(zip(original_frames, variant_a_frames, variant_b_frames)):
            frame_result = self.compare_frames(orig, a, b)
            frame_result["frame_number"] = i
            result.frame_comparisons.append(frame_result)

            a_psnrs.append(frame_result["a_psnr"])
            b_psnrs.append(frame_result["b_psnr"])
            a_ssims.append(frame_result["a_ssim"])
            b_ssims.append(frame_result["b_ssim"])

        # Calculate overall metrics diff
        result.metrics_diff.psnr_diff = np.mean(b_psnrs) - np.mean(a_psnrs)
        result.metrics_diff.ssim_diff = np.mean(b_ssims) - np.mean(a_ssims)

        # Count improvements/regressions
        for a_psnr, b_psnr in zip(a_psnrs, b_psnrs):
            if b_psnr > a_psnr + self.min_improvement_db:
                result.metrics_diff.psnr_improvements += 1
            elif a_psnr > b_psnr + self.min_improvement_db:
                result.metrics_diff.psnr_regressions += 1

        for a_ssim, b_ssim in zip(a_ssims, b_ssims):
            if b_ssim > a_ssim + self.min_ssim_improvement:
                result.metrics_diff.ssim_improvements += 1
            elif a_ssim > b_ssim + self.min_ssim_improvement:
                result.metrics_diff.ssim_regressions += 1

        # Statistical significance
        result.metrics_diff.psnr_significant = self._is_significant(a_psnrs, b_psnrs)
        result.metrics_diff.ssim_significant = self._is_significant(a_ssims, b_ssims)

        # Determine winner
        result.winner, result.confidence = self._determine_winner(result.metrics_diff)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def compare_from_paths(
        self,
        original_dir: Path,
        variant_a_dir: Path,
        variant_b_dir: Path,
        variant_a_name: str = "Variant A",
        variant_b_name: str = "Variant B",
        max_frames: Optional[int] = None,
    ) -> ComparisonResult:
        """Compare variants from frame directories."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV required for frame comparison")
            return ComparisonResult(variant_a_name, variant_b_name)

        # Find matching frames
        original_frames = sorted(Path(original_dir).glob("*.png"))
        if max_frames:
            original_frames = original_frames[:max_frames]

        orig_list = []
        a_list = []
        b_list = []

        for orig_path in original_frames:
            a_path = variant_a_dir / orig_path.name
            b_path = variant_b_dir / orig_path.name

            if a_path.exists() and b_path.exists():
                orig_list.append(cv2.imread(str(orig_path)))
                a_list.append(cv2.imread(str(a_path)))
                b_list.append(cv2.imread(str(b_path)))

        return self.compare_variants(
            orig_list, a_list, b_list,
            variant_a_name, variant_b_name
        )

    def generate_visual_diff(
        self,
        variant_a: np.ndarray,
        variant_b: np.ndarray,
    ) -> VisualDiff:
        """Generate visual difference analysis."""
        visual = VisualDiff()

        # Calculate difference map
        diff = np.abs(variant_a.astype(float) - variant_b.astype(float))
        visual.diff_map = diff.astype(np.uint8)
        visual.diff_magnitude = float(np.mean(diff))

        # Find regions with significant differences
        threshold = 30  # Pixel difference threshold
        diff_gray = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff

        try:
            import cv2
            # Find contours of different regions
            binary = (diff_gray > threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 100:  # Minimum region size
                    # Determine if improvement or degradation based on local quality
                    visual.regions_improved.append((x, y, w, h))
        except ImportError:
            pass

        # Calculate perceptual metrics
        visual.perceptual_diff = self._calculate_perceptual_diff(variant_a, variant_b)
        visual.edge_preservation_diff = self._calculate_edge_diff(variant_a, variant_b)

        return visual

    def _calculate_psnr(self, original: np.ndarray, restored: np.ndarray) -> float:
        """Calculate PSNR between original and restored."""
        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def _calculate_ssim(self, original: np.ndarray, restored: np.ndarray) -> float:
        """Calculate SSIM between original and restored."""
        try:
            from skimage.metrics import structural_similarity
            # Convert to grayscale if color
            if len(original.shape) == 3:
                import cv2
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                rest_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                rest_gray = restored

            return structural_similarity(orig_gray, rest_gray)
        except ImportError:
            # Simplified SSIM approximation
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2

            orig = original.astype(float)
            rest = restored.astype(float)

            mu_x = np.mean(orig)
            mu_y = np.mean(rest)
            sigma_x = np.std(orig)
            sigma_y = np.std(rest)
            sigma_xy = np.mean((orig - mu_x) * (rest - mu_y))

            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))

            return float(ssim)

    def _is_significant(self, a_values: List[float], b_values: List[float]) -> bool:
        """Check if difference is statistically significant using t-test."""
        try:
            from scipy import stats
            _, p_value = stats.ttest_rel(a_values, b_values)
            return p_value < self.significance_threshold
        except ImportError:
            # Simple heuristic if scipy not available
            diff = np.mean(b_values) - np.mean(a_values)
            std = np.std([b - a for a, b in zip(a_values, b_values)])
            return abs(diff) > 2 * std if std > 0 else diff != 0

    def _determine_winner(self, metrics: MetricsDiff) -> Tuple[str, float]:
        """Determine overall winner and confidence."""
        score_b = 0.0
        total_weight = 0.0

        # PSNR (weight: 0.4)
        if metrics.psnr_significant:
            if metrics.psnr_diff > self.min_improvement_db:
                score_b += 0.4
            elif metrics.psnr_diff < -self.min_improvement_db:
                score_b -= 0.4
            total_weight += 0.4

        # SSIM (weight: 0.4)
        if metrics.ssim_significant:
            if metrics.ssim_diff > self.min_ssim_improvement:
                score_b += 0.4
            elif metrics.ssim_diff < -self.min_ssim_improvement:
                score_b -= 0.4
            total_weight += 0.4

        # VMAF if available (weight: 0.2)
        if metrics.vmaf_diff is not None:
            if metrics.vmaf_diff > 2:
                score_b += 0.2
            elif metrics.vmaf_diff < -2:
                score_b -= 0.2
            total_weight += 0.2

        if total_weight == 0:
            return "tie", 0.0

        confidence = abs(score_b) / total_weight

        if score_b > 0.1:
            return "B", confidence
        elif score_b < -0.1:
            return "A", confidence
        else:
            return "tie", confidence

    def _generate_recommendations(self, result: ComparisonResult) -> List[str]:
        """Generate recommendations based on comparison."""
        recs = []

        md = result.metrics_diff

        if result.winner == "B":
            recs.append(f"Use {result.variant_b_name} for better overall quality")
        elif result.winner == "A":
            recs.append(f"Use {result.variant_a_name} for better overall quality")
        else:
            recs.append("Both variants produce similar quality - choose based on speed/resources")

        if md.psnr_significant and abs(md.psnr_diff) > 1:
            if md.psnr_diff > 0:
                recs.append(f"{result.variant_b_name} shows {md.psnr_diff:.1f}dB PSNR improvement")
            else:
                recs.append(f"{result.variant_a_name} shows {-md.psnr_diff:.1f}dB PSNR improvement")

        if md.ssim_significant and abs(md.ssim_diff) > 0.02:
            better = result.variant_b_name if md.ssim_diff > 0 else result.variant_a_name
            recs.append(f"{better} preserves structural details better")

        # Performance recommendations
        if result.processing_time_diff > 10:
            recs.append(f"{result.variant_a_name} is {result.processing_time_diff:.1f}s faster per batch")
        elif result.processing_time_diff < -10:
            recs.append(f"{result.variant_b_name} is {-result.processing_time_diff:.1f}s faster per batch")

        if result.vram_usage_diff > 1000:
            recs.append(f"{result.variant_a_name} uses {result.vram_usage_diff/1000:.1f}GB less VRAM")
        elif result.vram_usage_diff < -1000:
            recs.append(f"{result.variant_b_name} uses {-result.vram_usage_diff/1000:.1f}GB less VRAM")

        return recs

    def _calculate_perceptual_diff(
        self,
        variant_a: np.ndarray,
        variant_b: np.ndarray,
    ) -> float:
        """Calculate perceptual difference (simplified LPIPS-like)."""
        # Simple perceptual difference based on local variance
        try:
            import cv2
            a_gray = cv2.cvtColor(variant_a, cv2.COLOR_BGR2GRAY) if len(variant_a.shape) == 3 else variant_a
            b_gray = cv2.cvtColor(variant_b, cv2.COLOR_BGR2GRAY) if len(variant_b.shape) == 3 else variant_b

            # Local variance comparison
            kernel_size = 7
            a_blur = cv2.GaussianBlur(a_gray.astype(float), (kernel_size, kernel_size), 0)
            b_blur = cv2.GaussianBlur(b_gray.astype(float), (kernel_size, kernel_size), 0)

            a_var = cv2.GaussianBlur((a_gray.astype(float) - a_blur) ** 2, (kernel_size, kernel_size), 0)
            b_var = cv2.GaussianBlur((b_gray.astype(float) - b_blur) ** 2, (kernel_size, kernel_size), 0)

            return float(np.mean(np.abs(a_var - b_var)))
        except:
            return 0.0

    def _calculate_edge_diff(
        self,
        variant_a: np.ndarray,
        variant_b: np.ndarray,
    ) -> float:
        """Calculate difference in edge preservation."""
        try:
            import cv2
            a_gray = cv2.cvtColor(variant_a, cv2.COLOR_BGR2GRAY) if len(variant_a.shape) == 3 else variant_a
            b_gray = cv2.cvtColor(variant_b, cv2.COLOR_BGR2GRAY) if len(variant_b.shape) == 3 else variant_b

            # Sobel edge detection
            a_edges = cv2.Sobel(a_gray, cv2.CV_64F, 1, 1)
            b_edges = cv2.Sobel(b_gray, cv2.CV_64F, 1, 1)

            # Compare edge magnitudes
            a_mag = np.sqrt(a_edges ** 2)
            b_mag = np.sqrt(b_edges ** 2)

            return float(np.mean(b_mag) - np.mean(a_mag))
        except:
            return 0.0
