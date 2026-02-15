"""A/B Testing Framework for comparing restoration approaches."""

import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    name: str
    description: str = ""

    # Variants to test (name -> config overrides)
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Test parameters
    sample_frames: List[int] = field(default_factory=list)  # Specific frames to test
    sample_count: int = 10  # Number of frames if not specified
    sample_method: str = "uniform"  # uniform, random, keyframes

    # Metrics to compare
    compare_psnr: bool = True
    compare_ssim: bool = True
    compare_vmaf: bool = True
    compare_visual: bool = True

    # Output
    output_dir: Optional[Path] = None
    save_frames: bool = True
    generate_report: bool = True


@dataclass
class VariantResult:
    """Result for a single variant."""
    name: str
    config: Dict[str, Any]

    # Metrics (averaged across frames)
    avg_psnr: float = 0.0
    avg_ssim: float = 0.0
    avg_vmaf: float = 0.0

    # Per-frame metrics
    frame_metrics: List[Dict[str, float]] = field(default_factory=list)

    # Processing info
    processing_time_seconds: float = 0.0
    vram_peak_mb: float = 0.0

    # Output paths
    output_frames: List[Path] = field(default_factory=list)

    # User votes (for subjective comparison)
    votes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "config": self.config,
            "avg_psnr": self.avg_psnr,
            "avg_ssim": self.avg_ssim,
            "avg_vmaf": self.avg_vmaf,
            "processing_time_seconds": self.processing_time_seconds,
            "vram_peak_mb": self.vram_peak_mb,
            "votes": self.votes,
            "frame_count": len(self.frame_metrics),
        }


@dataclass
class ABTestResult:
    """Complete A/B test result."""
    test_id: str
    config: ABTestConfig

    # Input info
    input_path: str = ""
    total_frames: int = 0
    frames_tested: List[int] = field(default_factory=list)

    # Variant results
    variants: Dict[str, VariantResult] = field(default_factory=dict)

    # Winner determination
    winner_by_psnr: Optional[str] = None
    winner_by_ssim: Optional[str] = None
    winner_by_vmaf: Optional[str] = None
    winner_by_votes: Optional[str] = None
    overall_winner: Optional[str] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def determine_winners(self) -> None:
        """Determine winners by each metric."""
        if not self.variants:
            return

        # By PSNR
        by_psnr = sorted(self.variants.items(), key=lambda x: x[1].avg_psnr, reverse=True)
        self.winner_by_psnr = by_psnr[0][0] if by_psnr else None

        # By SSIM
        by_ssim = sorted(self.variants.items(), key=lambda x: x[1].avg_ssim, reverse=True)
        self.winner_by_ssim = by_ssim[0][0] if by_ssim else None

        # By VMAF
        by_vmaf = sorted(self.variants.items(), key=lambda x: x[1].avg_vmaf, reverse=True)
        self.winner_by_vmaf = by_vmaf[0][0] if by_vmaf else None

        # By votes
        by_votes = sorted(self.variants.items(), key=lambda x: x[1].votes, reverse=True)
        self.winner_by_votes = by_votes[0][0] if by_votes and by_votes[0][1].votes > 0 else None

        # Overall (weighted score)
        scores = {}
        for name, result in self.variants.items():
            # Normalize and weight metrics
            score = (
                result.avg_psnr / 50 * 0.2 +  # PSNR typically 20-50
                result.avg_ssim * 0.3 +        # SSIM 0-1
                result.avg_vmaf / 100 * 0.3 +  # VMAF 0-100
                (result.votes / 10) * 0.2      # Votes weighted
            )
            scores[name] = score

        if scores:
            self.overall_winner = max(scores.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.config.name,
            "description": self.config.description,
            "input_path": self.input_path,
            "total_frames": self.total_frames,
            "frames_tested": self.frames_tested,
            "variants": {name: v.to_dict() for name, v in self.variants.items()},
            "winners": {
                "psnr": self.winner_by_psnr,
                "ssim": self.winner_by_ssim,
                "vmaf": self.winner_by_vmaf,
                "votes": self.winner_by_votes,
                "overall": self.overall_winner,
            },
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def save_report(self, path: Path) -> None:
        """Save test report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def generate_html_report(self, path: Path) -> None:
        """Generate HTML comparison report."""
        html = self._generate_html()
        path.write_text(html)

    def _generate_html(self) -> str:
        """Generate HTML report content."""
        variant_rows = ""
        for name, result in self.variants.items():
            is_winner = name == self.overall_winner
            winner_class = "winner" if is_winner else ""
            variant_rows += f"""
            <tr class="{winner_class}">
                <td>{name}{'  🏆' if is_winner else ''}</td>
                <td>{result.avg_psnr:.2f} dB</td>
                <td>{result.avg_ssim:.4f}</td>
                <td>{result.avg_vmaf:.1f}</td>
                <td>{result.processing_time_seconds:.1f}s</td>
                <td>{result.votes}</td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>A/B Test Report - {self.config.name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f7; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1d1d1f; }}
        h2 {{ color: #424245; border-bottom: 1px solid #e5e5e5; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        th {{ background: #f5f5f7; font-weight: 600; }}
        .winner {{ background: #d4edda; }}
        .winner td:first-child {{ font-weight: bold; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f5f5f7; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1d1d1f; }}
        .stat-label {{ color: #6e6e73; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>A/B Test Report</h1>
        <p><strong>{self.config.name}</strong></p>
        <p>{self.config.description}</p>

        <div class="summary">
            <div class="stat">
                <div class="stat-value">{len(self.variants)}</div>
                <div class="stat-label">Variants Tested</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.frames_tested)}</div>
                <div class="stat-label">Frames Compared</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.overall_winner or 'N/A'}</div>
                <div class="stat-label">Overall Winner</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(v.votes for v in self.variants.values())}</div>
                <div class="stat-label">Total Votes</div>
            </div>
        </div>

        <h2>Results</h2>
        <table>
            <tr>
                <th>Variant</th>
                <th>PSNR</th>
                <th>SSIM</th>
                <th>VMAF</th>
                <th>Time</th>
                <th>Votes</th>
            </tr>
            {variant_rows}
        </table>

        <h2>Winners by Metric</h2>
        <ul>
            <li><strong>PSNR:</strong> {self.winner_by_psnr or 'N/A'}</li>
            <li><strong>SSIM:</strong> {self.winner_by_ssim or 'N/A'}</li>
            <li><strong>VMAF:</strong> {self.winner_by_vmaf or 'N/A'}</li>
            <li><strong>User Votes:</strong> {self.winner_by_votes or 'N/A'}</li>
        </ul>

        <p style="color: #6e6e73; margin-top: 40px;">
            Generated by FrameWright A/B Testing Framework
        </p>
    </div>
</body>
</html>"""


@dataclass
class ABTest:
    """A/B Test definition."""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: ABTestConfig = field(default_factory=ABTestConfig)
    result: Optional[ABTestResult] = None

    def add_variant(self, name: str, config_overrides: Dict[str, Any]) -> None:
        """Add a variant to test."""
        self.config.variants[name] = config_overrides

    def set_sample_frames(self, frames: List[int]) -> None:
        """Set specific frames to test."""
        self.config.sample_frames = frames


class ABTestRunner:
    """Runs A/B tests and collects results."""

    def __init__(self):
        self._ffmpeg_path = shutil.which("ffmpeg")
        self._numpy_available = self._check_numpy()

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def run(
        self,
        test: ABTest,
        input_path: Path,
        processor_factory: Callable[[Dict[str, Any]], Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ABTestResult:
        """Run an A/B test.

        Args:
            test: Test configuration
            input_path: Path to input video
            processor_factory: Function that creates a processor from config
            progress_callback: Optional progress callback (progress, message)
        """
        result = ABTestResult(
            test_id=test.test_id,
            config=test.config,
            input_path=str(input_path),
            started_at=datetime.now(),
        )

        # Get video info
        total_frames = self._get_frame_count(input_path)
        result.total_frames = total_frames

        # Determine which frames to test
        if test.config.sample_frames:
            frames_to_test = test.config.sample_frames
        else:
            frames_to_test = self._select_sample_frames(
                total_frames,
                test.config.sample_count,
                test.config.sample_method,
            )
        result.frames_tested = frames_to_test

        # Create output directory
        output_dir = test.config.output_dir or Path(tempfile.mkdtemp(prefix="abtest_"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract reference frames
        if progress_callback:
            progress_callback(0.0, "Extracting reference frames...")

        reference_frames = self._extract_frames(input_path, frames_to_test, output_dir / "reference")

        # Test each variant
        total_variants = len(test.config.variants)
        for i, (name, config) in enumerate(test.config.variants.items()):
            if progress_callback:
                progress_callback(
                    (i / total_variants) * 0.9,
                    f"Testing variant: {name}"
                )

            variant_result = self._run_variant(
                name,
                config,
                input_path,
                frames_to_test,
                reference_frames,
                output_dir / name,
                processor_factory,
                test.config,
            )
            result.variants[name] = variant_result

        # Determine winners
        result.determine_winners()
        result.completed_at = datetime.now()

        # Generate report
        if test.config.generate_report:
            result.save_report(output_dir / "report.json")
            result.generate_html_report(output_dir / "report.html")

        if progress_callback:
            progress_callback(1.0, f"Test complete. Winner: {result.overall_winner}")

        test.result = result
        return result

    def _get_frame_count(self, video_path: Path) -> int:
        """Get total frame count."""
        if not self._ffmpeg_path:
            return 100

        try:
            ffprobe = self._ffmpeg_path.replace("ffmpeg", "ffprobe")
            cmd = [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 100

    def _select_sample_frames(
        self,
        total_frames: int,
        count: int,
        method: str,
    ) -> List[int]:
        """Select sample frames for testing."""
        if method == "uniform":
            step = total_frames // (count + 1)
            return [step * (i + 1) for i in range(count)]
        elif method == "random":
            import random
            return sorted(random.sample(range(total_frames), min(count, total_frames)))
        else:  # keyframes - simplified
            step = total_frames // (count + 1)
            return [step * (i + 1) for i in range(count)]

    def _extract_frames(
        self,
        video_path: Path,
        frames: List[int],
        output_dir: Path,
    ) -> Dict[int, Path]:
        """Extract specific frames from video."""
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {}

        for frame_num in frames:
            output_path = output_dir / f"frame_{frame_num:08d}.png"

            if self._ffmpeg_path:
                cmd = [
                    self._ffmpeg_path,
                    "-i", str(video_path),
                    "-vf", f"select=eq(n\\,{frame_num})",
                    "-vframes", "1",
                    "-y",
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)

            if output_path.exists():
                result[frame_num] = output_path

        return result

    def _run_variant(
        self,
        name: str,
        config: Dict[str, Any],
        input_path: Path,
        frames: List[int],
        reference_frames: Dict[int, Path],
        output_dir: Path,
        processor_factory: Callable,
        test_config: ABTestConfig,
    ) -> VariantResult:
        """Run a single variant and collect metrics."""
        import time

        output_dir.mkdir(parents=True, exist_ok=True)

        result = VariantResult(name=name, config=config)
        start_time = time.time()

        try:
            # Create processor with variant config
            processor = processor_factory(config)

            # Process each frame
            psnr_values = []
            ssim_values = []
            vmaf_values = []

            for frame_num in frames:
                if frame_num not in reference_frames:
                    continue

                ref_path = reference_frames[frame_num]
                output_path = output_dir / f"frame_{frame_num:08d}.png"

                # Process frame (simplified - would use actual processor)
                # For now, just copy and measure against reference
                if hasattr(processor, 'process_frame'):
                    processor.process_frame(ref_path, output_path)
                else:
                    shutil.copy(ref_path, output_path)

                if output_path.exists():
                    result.output_frames.append(output_path)

                    # Calculate metrics
                    metrics = self._calculate_metrics(ref_path, output_path, test_config)
                    result.frame_metrics.append(metrics)

                    if metrics.get("psnr"):
                        psnr_values.append(metrics["psnr"])
                    if metrics.get("ssim"):
                        ssim_values.append(metrics["ssim"])
                    if metrics.get("vmaf"):
                        vmaf_values.append(metrics["vmaf"])

            # Calculate averages
            if psnr_values:
                result.avg_psnr = sum(psnr_values) / len(psnr_values)
            if ssim_values:
                result.avg_ssim = sum(ssim_values) / len(ssim_values)
            if vmaf_values:
                result.avg_vmaf = sum(vmaf_values) / len(vmaf_values)

        except Exception as e:
            logger.error(f"Variant {name} failed: {e}")

        result.processing_time_seconds = time.time() - start_time
        return result

    def _calculate_metrics(
        self,
        reference: Path,
        distorted: Path,
        config: ABTestConfig,
    ) -> Dict[str, float]:
        """Calculate quality metrics between two frames."""
        metrics = {}

        if not self._numpy_available:
            return metrics

        import numpy as np

        try:
            import cv2
            ref = cv2.imread(str(reference))
            dist = cv2.imread(str(distorted))

            if ref is None or dist is None:
                return metrics

            # PSNR
            if config.compare_psnr:
                mse = np.mean((ref.astype(np.float64) - dist.astype(np.float64)) ** 2)
                if mse > 0:
                    metrics["psnr"] = 10 * np.log10(255 ** 2 / mse)
                else:
                    metrics["psnr"] = 100.0

            # SSIM (simplified)
            if config.compare_ssim:
                ref_gray = np.mean(ref, axis=2)
                dist_gray = np.mean(dist, axis=2)

                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2

                mu_ref = np.mean(ref_gray)
                mu_dist = np.mean(dist_gray)
                sigma_ref = np.var(ref_gray)
                sigma_dist = np.var(dist_gray)
                sigma_ref_dist = np.mean((ref_gray - mu_ref) * (dist_gray - mu_dist))

                num = (2 * mu_ref * mu_dist + C1) * (2 * sigma_ref_dist + C2)
                den = (mu_ref ** 2 + mu_dist ** 2 + C1) * (sigma_ref + sigma_dist + C2)
                metrics["ssim"] = num / den if den != 0 else 0

        except ImportError:
            pass

        return metrics


def create_ab_test(
    name: str,
    variants: Dict[str, Dict[str, Any]],
    sample_count: int = 10,
) -> ABTest:
    """Create an A/B test with the given variants."""
    config = ABTestConfig(
        name=name,
        variants=variants,
        sample_count=sample_count,
    )
    return ABTest(config=config)
