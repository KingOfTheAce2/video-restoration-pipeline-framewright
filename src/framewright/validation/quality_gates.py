"""Quality gates for ensuring output meets standards."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GateDecision(Enum):
    """Decision from quality gate."""
    PASS = auto()
    WARN = auto()
    FAIL = auto()


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    # PSNR thresholds
    min_psnr: float = 28.0
    warn_psnr: float = 32.0

    # SSIM thresholds
    min_ssim: float = 0.85
    warn_ssim: float = 0.92

    # Frame-level requirements
    max_failed_frames_percent: float = 1.0
    max_black_frames_percent: float = 5.0
    max_blurry_frames_percent: float = 10.0

    # Output requirements
    min_bitrate_kbps: int = 1000
    max_file_size_gb: float = 50.0

    # Enable/disable checks
    check_psnr: bool = True
    check_ssim: bool = True
    check_frame_quality: bool = True
    check_output_specs: bool = True


@dataclass
class QualityGateResult:
    """Result from quality gate evaluation."""
    decision: GateDecision
    gate_name: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class QualityGate:
    """Evaluates quality metrics against thresholds."""

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()

    def evaluate(
        self,
        psnr_values: Optional[List[float]] = None,
        ssim_values: Optional[List[float]] = None,
        frame_issues: Optional[Dict[str, int]] = None,
        output_info: Optional[Dict[str, Any]] = None,
    ) -> List[QualityGateResult]:
        """Evaluate quality metrics against gates.

        Args:
            psnr_values: List of per-frame PSNR values
            ssim_values: List of per-frame SSIM values
            frame_issues: Dict of issue type -> count
            output_info: Output file information

        Returns:
            List of gate results
        """
        results = []

        if psnr_values and self.config.check_psnr:
            results.append(self._check_psnr(psnr_values))

        if ssim_values and self.config.check_ssim:
            results.append(self._check_ssim(ssim_values))

        if frame_issues and self.config.check_frame_quality:
            results.extend(self._check_frame_quality(frame_issues))

        if output_info and self.config.check_output_specs:
            results.extend(self._check_output_specs(output_info))

        return results

    def evaluate_all(
        self,
        psnr_values: Optional[List[float]] = None,
        ssim_values: Optional[List[float]] = None,
        frame_issues: Optional[Dict[str, int]] = None,
        output_info: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Evaluate and return overall decision.

        Returns:
            (overall_decision, results)
        """
        results = self.evaluate(psnr_values, ssim_values, frame_issues, output_info)

        if any(r.decision == GateDecision.FAIL for r in results):
            return GateDecision.FAIL, results
        elif any(r.decision == GateDecision.WARN for r in results):
            return GateDecision.WARN, results
        else:
            return GateDecision.PASS, results

    def _check_psnr(self, values: List[float]) -> QualityGateResult:
        """Check PSNR values."""
        avg_psnr = np.mean(values)
        min_psnr = np.min(values)

        if avg_psnr < self.config.min_psnr:
            return QualityGateResult(
                decision=GateDecision.FAIL,
                gate_name="PSNR",
                message=f"Average PSNR too low: {avg_psnr:.2f}dB < {self.config.min_psnr}dB",
                value=avg_psnr,
                threshold=self.config.min_psnr,
                details={"min_psnr": min_psnr, "avg_psnr": avg_psnr},
            )
        elif avg_psnr < self.config.warn_psnr:
            return QualityGateResult(
                decision=GateDecision.WARN,
                gate_name="PSNR",
                message=f"Average PSNR below ideal: {avg_psnr:.2f}dB",
                value=avg_psnr,
                threshold=self.config.warn_psnr,
            )
        else:
            return QualityGateResult(
                decision=GateDecision.PASS,
                gate_name="PSNR",
                message=f"PSNR OK: {avg_psnr:.2f}dB",
                value=avg_psnr,
            )

    def _check_ssim(self, values: List[float]) -> QualityGateResult:
        """Check SSIM values."""
        avg_ssim = np.mean(values)
        min_ssim = np.min(values)

        if avg_ssim < self.config.min_ssim:
            return QualityGateResult(
                decision=GateDecision.FAIL,
                gate_name="SSIM",
                message=f"Average SSIM too low: {avg_ssim:.4f} < {self.config.min_ssim}",
                value=avg_ssim,
                threshold=self.config.min_ssim,
                details={"min_ssim": min_ssim, "avg_ssim": avg_ssim},
            )
        elif avg_ssim < self.config.warn_ssim:
            return QualityGateResult(
                decision=GateDecision.WARN,
                gate_name="SSIM",
                message=f"Average SSIM below ideal: {avg_ssim:.4f}",
                value=avg_ssim,
                threshold=self.config.warn_ssim,
            )
        else:
            return QualityGateResult(
                decision=GateDecision.PASS,
                gate_name="SSIM",
                message=f"SSIM OK: {avg_ssim:.4f}",
                value=avg_ssim,
            )

    def _check_frame_quality(
        self,
        issues: Dict[str, int],
    ) -> List[QualityGateResult]:
        """Check frame-level quality issues."""
        results = []
        total_frames = issues.get("total_frames", 1)

        # Failed frames
        failed = issues.get("failed", 0)
        failed_pct = (failed / total_frames) * 100
        if failed_pct > self.config.max_failed_frames_percent:
            results.append(QualityGateResult(
                decision=GateDecision.FAIL,
                gate_name="Failed Frames",
                message=f"Too many failed frames: {failed_pct:.1f}%",
                value=failed_pct,
                threshold=self.config.max_failed_frames_percent,
            ))

        # Black frames
        black = issues.get("black_frames", 0)
        black_pct = (black / total_frames) * 100
        if black_pct > self.config.max_black_frames_percent:
            results.append(QualityGateResult(
                decision=GateDecision.WARN,
                gate_name="Black Frames",
                message=f"Many black frames: {black_pct:.1f}%",
                value=black_pct,
                threshold=self.config.max_black_frames_percent,
            ))

        # Blurry frames
        blurry = issues.get("blurry_frames", 0)
        blurry_pct = (blurry / total_frames) * 100
        if blurry_pct > self.config.max_blurry_frames_percent:
            results.append(QualityGateResult(
                decision=GateDecision.WARN,
                gate_name="Blurry Frames",
                message=f"Many blurry frames: {blurry_pct:.1f}%",
                value=blurry_pct,
                threshold=self.config.max_blurry_frames_percent,
            ))

        if not results:
            results.append(QualityGateResult(
                decision=GateDecision.PASS,
                gate_name="Frame Quality",
                message="Frame quality checks passed",
            ))

        return results

    def _check_output_specs(
        self,
        info: Dict[str, Any],
    ) -> List[QualityGateResult]:
        """Check output file specifications."""
        results = []

        # Bitrate
        bitrate = info.get("bitrate", 0) // 1000  # Convert to kbps
        if bitrate > 0 and bitrate < self.config.min_bitrate_kbps:
            results.append(QualityGateResult(
                decision=GateDecision.WARN,
                gate_name="Bitrate",
                message=f"Low bitrate: {bitrate}kbps",
                value=float(bitrate),
                threshold=float(self.config.min_bitrate_kbps),
            ))

        # File size
        file_size_gb = info.get("file_size_mb", 0) / 1024
        if file_size_gb > self.config.max_file_size_gb:
            results.append(QualityGateResult(
                decision=GateDecision.WARN,
                gate_name="File Size",
                message=f"Large file: {file_size_gb:.1f}GB",
                value=file_size_gb,
                threshold=self.config.max_file_size_gb,
            ))

        if not results:
            results.append(QualityGateResult(
                decision=GateDecision.PASS,
                gate_name="Output Specs",
                message="Output specifications OK",
            ))

        return results

    def generate_report(self, results: List[QualityGateResult]) -> str:
        """Generate human-readable quality report."""
        lines = [
            "=" * 50,
            "QUALITY GATE REPORT",
            "=" * 50,
        ]

        # Group by decision
        passed = [r for r in results if r.decision == GateDecision.PASS]
        warned = [r for r in results if r.decision == GateDecision.WARN]
        failed = [r for r in results if r.decision == GateDecision.FAIL]

        if failed:
            lines.append("\nFAILED:")
            lines.append("-" * 40)
            for r in failed:
                lines.append(f"  [X] {r.gate_name}: {r.message}")

        if warned:
            lines.append("\nWARNINGS:")
            lines.append("-" * 40)
            for r in warned:
                lines.append(f"  [!] {r.gate_name}: {r.message}")

        if passed:
            lines.append("\nPASSED:")
            lines.append("-" * 40)
            for r in passed:
                lines.append(f"  [OK] {r.gate_name}: {r.message}")

        # Overall verdict
        if failed:
            lines.append("\n" + "=" * 50)
            lines.append("VERDICT: FAILED - Output does not meet quality standards")
        elif warned:
            lines.append("\n" + "=" * 50)
            lines.append("VERDICT: PASSED WITH WARNINGS")
        else:
            lines.append("\n" + "=" * 50)
            lines.append("VERDICT: PASSED - All quality gates met")

        lines.append("=" * 50)

        return "\n".join(lines)
