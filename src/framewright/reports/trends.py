"""Quality Trend Reports for FrameWright.

Tracks quality metrics over time across multiple restoration jobs
and provides trend analysis and reporting capabilities.
"""

import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class QualityDataPoint:
    """A single quality measurement from a restoration job."""

    timestamp: datetime
    job_id: str
    job_name: str
    psnr: Optional[float]
    ssim: Optional[float]
    vmaf: Optional[float]
    processing_time_seconds: float
    frames_processed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "job_id": self.job_id,
            "job_name": self.job_name,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "vmaf": self.vmaf,
            "processing_time_seconds": self.processing_time_seconds,
            "frames_processed": self.frames_processed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityDataPoint":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            job_id=data["job_id"],
            job_name=data["job_name"],
            psnr=data.get("psnr"),
            ssim=data.get("ssim"),
            vmaf=data.get("vmaf"),
            processing_time_seconds=data["processing_time_seconds"],
            frames_processed=data["frames_processed"],
        )


@dataclass
class TrendAnalysis:
    """Analysis of quality trends over multiple jobs."""

    avg_psnr: float
    avg_ssim: float
    avg_vmaf: Optional[float]
    psnr_trend: str  # "improving", "stable", "declining"
    ssim_trend: str  # "improving", "stable", "declining"
    total_jobs: int
    total_frames: int
    avg_processing_time: float = 0.0
    vmaf_trend: Optional[str] = None  # "improving", "stable", "declining"
    psnr_std_dev: float = 0.0
    ssim_std_dev: float = 0.0
    vmaf_std_dev: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Quality Trends Manager
# =============================================================================


class QualityTrends:
    """Manages quality trend data storage, analysis, and reporting.

    Features:
    - Persistent storage of quality data points as JSON
    - Trend analysis using linear regression
    - Multiple report formats (text, HTML, CSV)
    - ASCII chart generation for terminal display
    """

    DATA_FILE = "quality_data.json"

    def __init__(self, data_dir: Path):
        """Initialize the quality trends manager.

        Args:
            data_dir: Directory to store/load quality data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = self.data_dir / self.DATA_FILE
        self._data_points: List[QualityDataPoint] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load existing data from disk."""
        if self._data_file.exists():
            try:
                with open(self._data_file, "r") as f:
                    data = json.load(f)
                self._data_points = [
                    QualityDataPoint.from_dict(d) for d in data.get("data_points", [])
                ]
                logger.debug(f"Loaded {len(self._data_points)} quality data points")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading quality data: {e}")
                self._data_points = []
        else:
            self._data_points = []

    def _save_data(self) -> None:
        """Save data to disk."""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "data_points": [dp.to_dict() for dp in self._data_points],
        }
        with open(self._data_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self._data_points)} quality data points")

    def add_data_point(self, point: QualityDataPoint) -> None:
        """Store a new quality metric data point.

        Args:
            point: The quality data point to store
        """
        self._data_points.append(point)
        # Sort by timestamp
        self._data_points.sort(key=lambda p: p.timestamp)
        self._save_data()
        logger.info(f"Added quality data point for job: {point.job_name}")

    def get_data_points(self, count: int = 10) -> List[QualityDataPoint]:
        """Get the most recent quality data points.

        Args:
            count: Number of recent points to return

        Returns:
            List of most recent QualityDataPoint objects
        """
        return self._data_points[-count:] if self._data_points else []

    def get_all_data_points(self) -> List[QualityDataPoint]:
        """Get all stored data points.

        Returns:
            List of all QualityDataPoint objects
        """
        return list(self._data_points)

    def analyze_trends(self) -> TrendAnalysis:
        """Calculate trends from stored quality data.

        Analyzes PSNR, SSIM, and VMAF values over time to determine
        if quality metrics are improving, stable, or declining.

        Returns:
            TrendAnalysis with calculated averages and trends
        """
        if not self._data_points:
            return TrendAnalysis(
                avg_psnr=0.0,
                avg_ssim=0.0,
                avg_vmaf=None,
                psnr_trend="stable",
                ssim_trend="stable",
                total_jobs=0,
                total_frames=0,
            )

        # Extract values
        psnr_values = [p.psnr for p in self._data_points if p.psnr is not None]
        ssim_values = [p.ssim for p in self._data_points if p.ssim is not None]
        vmaf_values = [p.vmaf for p in self._data_points if p.vmaf is not None]
        processing_times = [p.processing_time_seconds for p in self._data_points]

        # Calculate averages
        avg_psnr = statistics.mean(psnr_values) if psnr_values else 0.0
        avg_ssim = statistics.mean(ssim_values) if ssim_values else 0.0
        avg_vmaf = statistics.mean(vmaf_values) if vmaf_values else None
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0

        # Calculate standard deviations
        psnr_std = statistics.stdev(psnr_values) if len(psnr_values) > 1 else 0.0
        ssim_std = statistics.stdev(ssim_values) if len(ssim_values) > 1 else 0.0
        vmaf_std = statistics.stdev(vmaf_values) if len(vmaf_values) > 1 else None

        # Calculate trends using linear regression
        psnr_trend = self._calculate_trend(psnr_values)
        ssim_trend = self._calculate_trend(ssim_values)
        vmaf_trend = self._calculate_trend(vmaf_values) if vmaf_values else None

        # Total frames
        total_frames = sum(p.frames_processed for p in self._data_points)

        return TrendAnalysis(
            avg_psnr=avg_psnr,
            avg_ssim=avg_ssim,
            avg_vmaf=avg_vmaf,
            psnr_trend=psnr_trend,
            ssim_trend=ssim_trend,
            total_jobs=len(self._data_points),
            total_frames=total_frames,
            avg_processing_time=avg_processing_time,
            vmaf_trend=vmaf_trend,
            psnr_std_dev=psnr_std,
            ssim_std_dev=ssim_std,
            vmaf_std_dev=vmaf_std,
        )

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction using linear regression.

        Args:
            values: List of values in chronological order

        Returns:
            "improving", "stable", or "declining"
        """
        if len(values) < 2:
            return "stable"

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        # Calculate slope
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Normalize slope by the range of values
        value_range = max(values) - min(values) if max(values) != min(values) else 1.0
        normalized_slope = slope / value_range

        # Determine trend based on normalized slope
        if normalized_slope > 0.02:  # More than 2% improvement per data point
            return "improving"
        elif normalized_slope < -0.02:  # More than 2% decline per data point
            return "declining"
        else:
            return "stable"

    def generate_report(self, format: str = "text") -> str:
        """Generate a quality trends report.

        Args:
            format: Report format ("text" or "html")

        Returns:
            Formatted report string
        """
        if format.lower() == "html":
            return self._generate_html_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate a text format report."""
        analysis = self.analyze_trends()
        recent = self.get_data_points(10)

        lines = [
            "=" * 60,
            "FrameWright Quality Trends Report",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary",
            "-" * 40,
            f"Total Jobs Analyzed: {analysis.total_jobs}",
            f"Total Frames Processed: {analysis.total_frames:,}",
            f"Average Processing Time: {analysis.avg_processing_time:.1f}s",
            "",
            "Quality Metrics",
            "-" * 40,
            f"Average PSNR: {analysis.avg_psnr:.2f} dB (Trend: {analysis.psnr_trend})",
            f"Average SSIM: {analysis.avg_ssim:.4f} (Trend: {analysis.ssim_trend})",
        ]

        if analysis.avg_vmaf is not None:
            lines.append(
                f"Average VMAF: {analysis.avg_vmaf:.2f} (Trend: {analysis.vmaf_trend})"
            )

        lines.extend([
            "",
            "Standard Deviations",
            "-" * 40,
            f"PSNR Std Dev: {analysis.psnr_std_dev:.2f} dB",
            f"SSIM Std Dev: {analysis.ssim_std_dev:.4f}",
        ])

        if analysis.vmaf_std_dev is not None:
            lines.append(f"VMAF Std Dev: {analysis.vmaf_std_dev:.2f}")

        if recent:
            lines.extend([
                "",
                "Recent Jobs",
                "-" * 40,
            ])
            for point in recent[-5:]:
                psnr_str = f"{point.psnr:.2f} dB" if point.psnr else "N/A"
                ssim_str = f"{point.ssim:.4f}" if point.ssim else "N/A"
                lines.append(
                    f"  {point.job_name[:30]:<30} PSNR: {psnr_str:<12} SSIM: {ssim_str}"
                )

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _generate_html_report(self) -> str:
        """Generate an HTML format report."""
        analysis = self.analyze_trends()
        recent = self.get_data_points(10)

        trend_colors = {
            "improving": "#22c55e",
            "stable": "#eab308",
            "declining": "#ef4444",
        }

        psnr_color = trend_colors.get(analysis.psnr_trend, "#888")
        ssim_color = trend_colors.get(analysis.ssim_trend, "#888")
        vmaf_color = trend_colors.get(analysis.vmaf_trend, "#888") if analysis.vmaf_trend else "#888"

        # Build recent jobs table
        jobs_rows = ""
        for point in recent:
            psnr_str = f"{point.psnr:.2f}" if point.psnr else "N/A"
            ssim_str = f"{point.ssim:.4f}" if point.ssim else "N/A"
            vmaf_str = f"{point.vmaf:.2f}" if point.vmaf else "N/A"
            jobs_rows += f"""
            <tr>
                <td>{point.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{point.job_name}</td>
                <td>{psnr_str}</td>
                <td>{ssim_str}</td>
                <td>{vmaf_str}</td>
                <td>{point.frames_processed:,}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright Quality Trends Report</title>
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
        .trend {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .trend-improving {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; }}
        .trend-stable {{ background: rgba(234, 179, 8, 0.2); color: #eab308; }}
        .trend-declining {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ color: #c084fc; font-weight: 600; }}
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
            <h1>Quality Trends Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{analysis.total_jobs}</div>
                    <div class="metric-label">Total Jobs</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis.total_frames:,}</div>
                    <div class="metric-label">Total Frames</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis.avg_processing_time:.1f}s</div>
                    <div class="metric-label">Avg Processing Time</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Quality Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{analysis.avg_psnr:.2f} dB</div>
                    <div class="metric-label">Average PSNR</div>
                    <span class="trend trend-{analysis.psnr_trend}">{analysis.psnr_trend}</span>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis.avg_ssim:.4f}</div>
                    <div class="metric-label">Average SSIM</div>
                    <span class="trend trend-{analysis.ssim_trend}">{analysis.ssim_trend}</span>
                </div>
                <div class="metric">
                    <div class="metric-value">{f'{analysis.avg_vmaf:.2f}' if analysis.avg_vmaf else 'N/A'}</div>
                    <div class="metric-label">Average VMAF</div>
                    {f'<span class="trend trend-{analysis.vmaf_trend}">{analysis.vmaf_trend}</span>' if analysis.vmaf_trend else ''}
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Recent Jobs</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Job Name</th>
                        <th>PSNR</th>
                        <th>SSIM</th>
                        <th>VMAF</th>
                        <th>Frames</th>
                    </tr>
                </thead>
                <tbody>
                    {jobs_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by FrameWright Video Restoration Pipeline</p>
        </div>
    </div>
</body>
</html>"""

        return html

    def export_csv(self, output_path: Path) -> None:
        """Export all quality data as CSV.

        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "timestamp,job_id,job_name,psnr,ssim,vmaf,processing_time_seconds,frames_processed"
        ]

        for point in self._data_points:
            psnr = f"{point.psnr:.4f}" if point.psnr is not None else ""
            ssim = f"{point.ssim:.6f}" if point.ssim is not None else ""
            vmaf = f"{point.vmaf:.4f}" if point.vmaf is not None else ""

            # Escape commas in job name
            job_name = f'"{point.job_name}"' if "," in point.job_name else point.job_name

            lines.append(
                f"{point.timestamp.isoformat()},{point.job_id},{job_name},"
                f"{psnr},{ssim},{vmaf},{point.processing_time_seconds:.2f},{point.frames_processed}"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported {len(self._data_points)} data points to {output_path}")

    def plot_trends(self, output_path: Path) -> None:
        """Generate a trends visualization.

        Attempts to use matplotlib if available, falls back to ASCII chart.
        For matplotlib, supports png, pdf, svg, jpg formats based on extension.
        For ASCII output, use .txt extension.

        Args:
            output_path: Path to save the plot
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if output is for ASCII chart
        if output_path.suffix.lower() in (".txt", ".asc", ""):
            ascii_chart = self._generate_ascii_chart()
            with open(output_path, "w") as f:
                f.write(ascii_chart)
            logger.info(f"ASCII chart saved to {output_path}")
            return

        # Try matplotlib for image formats
        try:
            self._plot_matplotlib(output_path)
            return
        except ImportError:
            logger.debug("matplotlib not available, using ASCII chart")
            # Fall back to ASCII chart with .txt extension
            txt_path = output_path.with_suffix(".txt")
            ascii_chart = self._generate_ascii_chart()
            with open(txt_path, "w") as f:
                f.write(ascii_chart)
            logger.info(f"ASCII chart saved to {txt_path} (matplotlib not available)")

    def _plot_matplotlib(self, output_path: Path) -> None:
        """Generate matplotlib plot."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not self._data_points:
            raise ValueError("No data points to plot")

        # Extract data
        timestamps = [p.timestamp for p in self._data_points]
        psnr_values = [p.psnr for p in self._data_points]
        ssim_values = [p.ssim for p in self._data_points]
        vmaf_values = [p.vmaf for p in self._data_points]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("FrameWright Quality Trends", fontsize=14, fontweight="bold")

        # Style
        plt.style.use("dark_background")

        # PSNR plot
        ax1 = axes[0]
        valid_psnr = [(t, v) for t, v in zip(timestamps, psnr_values) if v is not None]
        if valid_psnr:
            t, v = zip(*valid_psnr)
            ax1.plot(t, v, "o-", color="#c084fc", linewidth=2, markersize=6)
            ax1.fill_between(t, v, alpha=0.3, color="#c084fc")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title("PSNR Over Time")
        ax1.grid(True, alpha=0.3)

        # SSIM plot
        ax2 = axes[1]
        valid_ssim = [(t, v) for t, v in zip(timestamps, ssim_values) if v is not None]
        if valid_ssim:
            t, v = zip(*valid_ssim)
            ax2.plot(t, v, "o-", color="#22c55e", linewidth=2, markersize=6)
            ax2.fill_between(t, v, alpha=0.3, color="#22c55e")
        ax2.set_ylabel("SSIM")
        ax2.set_title("SSIM Over Time")
        ax2.grid(True, alpha=0.3)

        # VMAF plot
        ax3 = axes[2]
        valid_vmaf = [(t, v) for t, v in zip(timestamps, vmaf_values) if v is not None]
        if valid_vmaf:
            t, v = zip(*valid_vmaf)
            ax3.plot(t, v, "o-", color="#eab308", linewidth=2, markersize=6)
            ax3.fill_between(t, v, alpha=0.3, color="#eab308")
        ax3.set_ylabel("VMAF")
        ax3.set_title("VMAF Over Time")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("Date")

        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

        logger.info(f"Plot saved to {output_path}")

    def _generate_ascii_chart(self) -> str:
        """Generate an ASCII chart of trends."""
        if not self._data_points:
            return "No data points available for charting."

        # Chart dimensions
        width = 60
        height = 15

        lines = [
            "=" * (width + 10),
            "FrameWright Quality Trends - ASCII Chart",
            "=" * (width + 10),
            "",
        ]

        # Generate PSNR chart
        psnr_values = [p.psnr for p in self._data_points if p.psnr is not None]
        if psnr_values:
            lines.append("PSNR (dB)")
            lines.append("-" * (width + 10))
            lines.extend(self._make_ascii_chart(psnr_values, width, height))
            lines.append("")

        # Generate SSIM chart
        ssim_values = [p.ssim for p in self._data_points if p.ssim is not None]
        if ssim_values:
            lines.append("SSIM")
            lines.append("-" * (width + 10))
            lines.extend(self._make_ascii_chart(ssim_values, width, height))
            lines.append("")

        # Generate VMAF chart
        vmaf_values = [p.vmaf for p in self._data_points if p.vmaf is not None]
        if vmaf_values:
            lines.append("VMAF")
            lines.append("-" * (width + 10))
            lines.extend(self._make_ascii_chart(vmaf_values, width, height))

        lines.append("")
        lines.append("=" * (width + 10))

        return "\n".join(lines)

    def _make_ascii_chart(
        self, values: List[float], width: int, height: int
    ) -> List[str]:
        """Create an ASCII line chart."""
        if not values:
            return ["No data"]

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1.0

        # Normalize values to chart height
        normalized = [(v - min_val) / val_range for v in values]

        # Create chart grid
        chart = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, norm_val in enumerate(normalized):
            x = int(i * (width - 1) / max(len(values) - 1, 1))
            y = height - 1 - int(norm_val * (height - 1))
            y = max(0, min(height - 1, y))
            chart[y][x] = "*"

        # Build output lines with Y-axis labels
        lines = []
        for i, row in enumerate(chart):
            if i == 0:
                label = f"{max_val:7.2f} |"
            elif i == height - 1:
                label = f"{min_val:7.2f} |"
            elif i == height // 2:
                mid_val = (max_val + min_val) / 2
                label = f"{mid_val:7.2f} |"
            else:
                label = "        |"
            lines.append(label + "".join(row))

        # Add X-axis
        lines.append("        +" + "-" * width)
        lines.append("        First" + " " * (width - 9) + "Latest")

        return lines

    def clear_data(self) -> None:
        """Clear all stored data points."""
        self._data_points = []
        self._save_data()
        logger.info("Cleared all quality trend data")


# =============================================================================
# Factory Function
# =============================================================================


def create_quality_tracker(data_dir: Optional[Path] = None) -> QualityTrends:
    """Create a QualityTrends tracker instance.

    Args:
        data_dir: Optional directory for data storage.
                  Defaults to ~/.framewright/quality_data

    Returns:
        Configured QualityTrends instance
    """
    if data_dir is None:
        data_dir = Path.home() / ".framewright" / "quality_data"

    return QualityTrends(data_dir)
