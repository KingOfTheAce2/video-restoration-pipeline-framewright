"""VMAF (Video Multi-method Assessment Fusion) quality calculation.

Netflix's perceptual video quality metric.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil

logger = logging.getLogger(__name__)


@dataclass
class VMAFConfig:
    """Configuration for VMAF calculation."""
    # Model selection
    model: str = "vmaf_v0.6.1"  # vmaf_v0.6.1, vmaf_4k_v0.6.1, vmaf_b_v0.6.3
    phone_model: bool = False  # Use phone model for mobile viewing

    # Calculation settings
    subsample: int = 1  # Frame subsampling (1 = every frame, 5 = every 5th)
    threads: int = 0  # 0 = auto

    # Additional metrics
    include_psnr: bool = True
    include_ssim: bool = True
    include_ms_ssim: bool = False

    # Output
    per_frame_scores: bool = True
    output_json: bool = True

    # Scaling
    scale_to_reference: bool = True  # Scale distorted to reference size


@dataclass
class VMAFFrameScore:
    """VMAF score for a single frame."""
    frame_number: int
    vmaf: float
    psnr_y: Optional[float] = None
    psnr_cb: Optional[float] = None
    psnr_cr: Optional[float] = None
    ssim: Optional[float] = None
    ms_ssim: Optional[float] = None

    @property
    def psnr(self) -> Optional[float]:
        """Get overall PSNR (Y channel)."""
        return self.psnr_y


@dataclass
class VMAFResult:
    """Complete VMAF calculation result."""
    # Aggregate scores
    vmaf_mean: float = 0.0
    vmaf_min: float = 0.0
    vmaf_max: float = 0.0
    vmaf_harmonic_mean: float = 0.0
    vmaf_percentile_5: float = 0.0
    vmaf_percentile_25: float = 0.0
    vmaf_percentile_50: float = 0.0
    vmaf_percentile_75: float = 0.0
    vmaf_percentile_95: float = 0.0
    vmaf_std: float = 0.0

    # Additional metrics
    psnr_mean: Optional[float] = None
    ssim_mean: Optional[float] = None
    ms_ssim_mean: Optional[float] = None

    # Per-frame scores
    frame_scores: List[VMAFFrameScore] = field(default_factory=list)

    # Metadata
    model_version: str = ""
    frame_count: int = 0
    calculation_time: float = 0.0

    def get_quality_grade(self) -> str:
        """Get quality grade based on VMAF score."""
        if self.vmaf_mean >= 95:
            return "Excellent"
        elif self.vmaf_mean >= 90:
            return "Great"
        elif self.vmaf_mean >= 80:
            return "Good"
        elif self.vmaf_mean >= 70:
            return "Fair"
        elif self.vmaf_mean >= 60:
            return "Poor"
        else:
            return "Bad"

    def get_problem_frames(self, threshold: float = 70.0) -> List[VMAFFrameScore]:
        """Get frames with VMAF below threshold."""
        return [f for f in self.frame_scores if f.vmaf < threshold]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vmaf": {
                "mean": self.vmaf_mean,
                "min": self.vmaf_min,
                "max": self.vmaf_max,
                "harmonic_mean": self.vmaf_harmonic_mean,
                "std": self.vmaf_std,
                "percentiles": {
                    "5": self.vmaf_percentile_5,
                    "25": self.vmaf_percentile_25,
                    "50": self.vmaf_percentile_50,
                    "75": self.vmaf_percentile_75,
                    "95": self.vmaf_percentile_95,
                },
            },
            "psnr_mean": self.psnr_mean,
            "ssim_mean": self.ssim_mean,
            "ms_ssim_mean": self.ms_ssim_mean,
            "quality_grade": self.get_quality_grade(),
            "model_version": self.model_version,
            "frame_count": self.frame_count,
            "calculation_time": self.calculation_time,
        }


class VMAFCalculator:
    """Calculate VMAF scores between reference and distorted videos."""

    def __init__(self, config: Optional[VMAFConfig] = None):
        self.config = config or VMAFConfig()
        self._ffmpeg_path = shutil.which("ffmpeg")

    def calculate(
        self,
        reference: Path,
        distorted: Path,
        progress_callback: Optional[callable] = None,
    ) -> VMAFResult:
        """Calculate VMAF between reference and distorted videos."""
        if not self._ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

        import time
        start_time = time.time()

        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            # Build filter chain
            filters = self._build_filter(reference, distorted, json_path)

            # Run FFmpeg with libvmaf
            cmd = [
                self._ffmpeg_path,
                "-i", str(distorted),
                "-i", str(reference),
                "-lavfi", filters,
                "-f", "null", "-"
            ]

            if self.config.threads > 0:
                cmd.insert(1, "-threads")
                cmd.insert(2, str(self.config.threads))

            logger.info(f"Running VMAF calculation...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"VMAF calculation failed: {result.stderr}")
                raise RuntimeError(f"VMAF calculation failed: {result.stderr}")

            # Parse results
            vmaf_result = self._parse_results(json_path)
            vmaf_result.calculation_time = time.time() - start_time

            return vmaf_result

        finally:
            json_path.unlink(missing_ok=True)

    def _build_filter(self, reference: Path, distorted: Path, output_path: Path) -> str:
        """Build FFmpeg filter string for VMAF."""
        # Build feature list
        features = []
        if self.config.include_psnr:
            features.append("psnr=1")
        if self.config.include_ssim:
            features.append("ssim=1")
        if self.config.include_ms_ssim:
            features.append("ms_ssim=1")

        # Model path
        model = self.config.model
        if self.config.phone_model:
            model = "vmaf_v0.6.1neg"  # Phone model

        # Build filter
        vmaf_filter = f"libvmaf="
        vmaf_filter += f"model='version={model}'"
        vmaf_filter += f":n_subsample={self.config.subsample}"
        vmaf_filter += f":log_path='{output_path}'"
        vmaf_filter += ":log_fmt=json"

        if features:
            vmaf_filter += f":feature='{':'.join(features)}'"

        # Scale distorted to reference if needed
        if self.config.scale_to_reference:
            return f"[0:v]scale=flags=bicubic[distorted];[distorted][1:v]{vmaf_filter}"
        else:
            return f"[0:v][1:v]{vmaf_filter}"

    def _parse_results(self, json_path: Path) -> VMAFResult:
        """Parse VMAF JSON output."""
        with open(json_path) as f:
            data = json.load(f)

        result = VMAFResult()

        # Parse pooled metrics
        pooled = data.get("pooled_metrics", {})

        vmaf_data = pooled.get("vmaf", {})
        result.vmaf_mean = vmaf_data.get("mean", 0.0)
        result.vmaf_min = vmaf_data.get("min", 0.0)
        result.vmaf_max = vmaf_data.get("max", 0.0)
        result.vmaf_harmonic_mean = vmaf_data.get("harmonic_mean", result.vmaf_mean)
        result.vmaf_std = vmaf_data.get("stddev", 0.0)

        # Percentiles
        percentiles = vmaf_data.get("percentile", {})
        result.vmaf_percentile_5 = percentiles.get("5", result.vmaf_min)
        result.vmaf_percentile_25 = percentiles.get("25", result.vmaf_mean)
        result.vmaf_percentile_50 = percentiles.get("50", result.vmaf_mean)
        result.vmaf_percentile_75 = percentiles.get("75", result.vmaf_mean)
        result.vmaf_percentile_95 = percentiles.get("95", result.vmaf_max)

        # Additional metrics
        if "psnr_y" in pooled:
            result.psnr_mean = pooled["psnr_y"].get("mean")

        if "ssim" in pooled:
            result.ssim_mean = pooled["ssim"].get("mean")

        if "ms_ssim" in pooled:
            result.ms_ssim_mean = pooled["ms_ssim"].get("mean")

        # Model version
        result.model_version = self.config.model

        # Per-frame scores
        if self.config.per_frame_scores:
            frames = data.get("frames", [])
            result.frame_count = len(frames)

            for frame_data in frames:
                metrics = frame_data.get("metrics", {})
                score = VMAFFrameScore(
                    frame_number=frame_data.get("frameNum", 0),
                    vmaf=metrics.get("vmaf", 0.0),
                    psnr_y=metrics.get("psnr_y"),
                    psnr_cb=metrics.get("psnr_cb"),
                    psnr_cr=metrics.get("psnr_cr"),
                    ssim=metrics.get("ssim"),
                    ms_ssim=metrics.get("ms_ssim"),
                )
                result.frame_scores.append(score)

        return result

    def calculate_batch(
        self,
        pairs: List[Tuple[Path, Path]],
        progress_callback: Optional[callable] = None,
    ) -> List[VMAFResult]:
        """Calculate VMAF for multiple reference/distorted pairs."""
        results = []

        for i, (ref, dist) in enumerate(pairs):
            try:
                result = self.calculate(ref, dist)
                results.append(result)
            except Exception as e:
                logger.error(f"VMAF calculation failed for pair {i}: {e}")
                results.append(VMAFResult())

            if progress_callback:
                progress_callback((i + 1) / len(pairs))

        return results


def quick_vmaf(reference: Path, distorted: Path) -> float:
    """Quick VMAF calculation returning just the mean score."""
    calc = VMAFCalculator(VMAFConfig(
        per_frame_scores=False,
        include_psnr=False,
        include_ssim=False,
        subsample=5,  # Every 5th frame for speed
    ))

    result = calc.calculate(reference, distorted)
    return result.vmaf_mean
