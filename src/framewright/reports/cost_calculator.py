"""Processing Cost Calculator for FrameWright.

Provides accurate cost estimation for video restoration processing
including time, disk usage, electricity, and cloud compute costs.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

logger = logging.getLogger(__name__)


# Default GPU power consumption in watts
DEFAULT_GPU_POWER_WATTS: Dict[str, int] = {
    # NVIDIA Consumer GPUs
    "RTX_4090": 450,
    "RTX_4080": 320,
    "RTX_4070_Ti": 285,
    "RTX_4070": 200,
    "RTX_4060_Ti": 165,
    "RTX_4060": 115,
    "RTX_3090": 350,
    "RTX_3090_Ti": 450,
    "RTX_3080": 320,
    "RTX_3080_Ti": 350,
    "RTX_3070": 220,
    "RTX_3070_Ti": 290,
    "RTX_3060": 170,
    "RTX_3060_Ti": 200,
    "RTX_2080_Ti": 260,
    "RTX_2080": 215,
    "RTX_2070": 175,
    "RTX_2060": 160,
    # NVIDIA Data Center GPUs
    "A100_80GB": 400,
    "A100_40GB": 400,
    "A6000": 300,
    "A5000": 230,
    "A4000": 140,
    "H100": 700,
    "H100_SXM": 700,
    "H100_PCIe": 350,
    "L40": 300,
    "L4": 72,
    "T4": 70,
    "V100": 300,
    "V100S": 250,
    # AMD GPUs
    "RX_7900_XTX": 355,
    "RX_7900_XT": 315,
    "RX_6900_XT": 300,
    "MI300X": 750,
    "MI250X": 560,
    "MI100": 300,
    # Default fallback
    "unknown": 250,
}

# Default cloud rates in USD per hour (approximate market rates)
DEFAULT_CLOUD_RATES: Dict[str, Dict[str, float]] = {
    "runpod": {
        "RTX_4090": 0.69,
        "RTX_4080": 0.59,
        "RTX_3090": 0.44,
        "RTX_3080": 0.34,
        "A100_80GB": 1.99,
        "A100_40GB": 1.49,
        "A6000": 0.79,
        "H100": 4.49,
        "L40": 1.19,
    },
    "vastai": {
        "RTX_4090": 0.45,
        "RTX_4080": 0.35,
        "RTX_3090": 0.30,
        "RTX_3080": 0.22,
        "A100_80GB": 1.50,
        "A100_40GB": 1.10,
        "A6000": 0.55,
        "H100": 3.50,
        "L40": 0.85,
    },
    "lambda_labs": {
        "A100_80GB": 1.29,
        "A100_40GB": 1.10,
        "H100": 2.49,
        "RTX_4090": 0.75,
    },
    "paperspace": {
        "RTX_4000": 0.45,
        "RTX_5000": 0.78,
        "A4000": 0.56,
        "A5000": 0.87,
        "A6000": 1.12,
        "A100_80GB": 3.09,
    },
}

# Processing time multipliers by preset (relative to "fast")
PRESET_TIME_MULTIPLIERS: Dict[str, float] = {
    "fast": 1.0,
    "quality": 2.5,
    "archive": 4.0,
    "anime": 2.0,
    "film_restoration": 3.5,
    "ultimate": 6.0,
    "authentic": 2.0,
    "vhs": 2.5,
}

# Base processing time per frame in seconds (for 1080p, fast preset)
BASE_FRAME_TIME_SECONDS = 0.15

# Disk usage per frame in MB (for different resolutions)
DISK_USAGE_PER_FRAME_MB: Dict[str, float] = {
    "480p": 0.5,
    "720p": 1.0,
    "1080p": 2.5,
    "1440p": 4.0,
    "2160p": 8.0,  # 4K
    "4320p": 32.0,  # 8K
}


@dataclass
class CostEstimate:
    """Comprehensive cost estimate for video processing."""

    estimated_time_seconds: float
    estimated_disk_gb: float
    estimated_electricity_kwh: float
    estimated_electricity_cost_usd: float
    estimated_cloud_cost_usd: Optional[float] = None
    gpu_type: str = "unknown"
    confidence: str = "medium"  # high, medium, low

    # Additional breakdown
    frame_count: int = 0
    resolution: Tuple[int, int] = (0, 0)
    preset: str = "quality"
    cloud_provider: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimated_time_seconds": self.estimated_time_seconds,
            "estimated_time_formatted": self._format_time(self.estimated_time_seconds),
            "estimated_disk_gb": round(self.estimated_disk_gb, 2),
            "estimated_electricity_kwh": round(self.estimated_electricity_kwh, 4),
            "estimated_electricity_cost_usd": round(self.estimated_electricity_cost_usd, 2),
            "estimated_cloud_cost_usd": (
                round(self.estimated_cloud_cost_usd, 2)
                if self.estimated_cloud_cost_usd is not None
                else None
            ),
            "gpu_type": self.gpu_type,
            "confidence": self.confidence,
            "frame_count": self.frame_count,
            "resolution": list(self.resolution),
            "preset": self.preset,
            "cloud_provider": self.cloud_provider,
            "notes": self.notes,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"


@dataclass
class CostConfig:
    """Configuration for cost estimation."""

    electricity_rate_per_kwh: float = 0.12  # USD per kWh (US average)
    gpu_power_watts: Dict[str, int] = field(default_factory=dict)
    cloud_rates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default_gpu_type: str = "RTX_3080"
    default_cloud_provider: str = "runpod"

    def __post_init__(self) -> None:
        """Initialize with defaults if not provided."""
        if not self.gpu_power_watts:
            self.gpu_power_watts = DEFAULT_GPU_POWER_WATTS.copy()
        if not self.cloud_rates:
            self.cloud_rates = DEFAULT_CLOUD_RATES.copy()


class CostCalculator:
    """Calculate processing costs for video restoration."""

    def __init__(self, config: Optional[CostConfig] = None) -> None:
        """Initialize the cost calculator.

        Args:
            config: Cost configuration. Uses defaults if not provided.
        """
        self.config = config or CostConfig()

    def estimate_from_video(
        self,
        video_path: Path,
        preset: str = "quality",
        gpu_type: Optional[str] = None,
        cloud_provider: Optional[str] = None,
    ) -> CostEstimate:
        """Estimate processing costs from a video file.

        Args:
            video_path: Path to the input video file.
            preset: Processing preset name.
            gpu_type: GPU type for estimation (uses default if not provided).
            cloud_provider: Cloud provider for cloud cost estimation.

        Returns:
            CostEstimate with all cost projections.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video cannot be probed.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            from ..utils.ffmpeg import probe_video

            video_info = probe_video(video_path)
            frame_count = int(video_info["duration"] * video_info["framerate"])
            resolution = (video_info["width"], video_info["height"])

            return self.estimate_from_params(
                frame_count=frame_count,
                resolution=resolution,
                preset=preset,
                gpu_type=gpu_type,
                cloud_provider=cloud_provider,
            )

        except ImportError:
            logger.warning("FFmpeg utils not available, using file size estimation")
            # Fallback: estimate from file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            # Rough estimate: assume 30fps, 2.5MB per second of 1080p video
            estimated_duration = file_size_mb / 2.5
            frame_count = int(estimated_duration * 30)

            return self.estimate_from_params(
                frame_count=frame_count,
                resolution=(1920, 1080),  # Assume 1080p
                preset=preset,
                gpu_type=gpu_type,
                cloud_provider=cloud_provider,
            )

    def estimate_from_params(
        self,
        frame_count: int,
        resolution: Tuple[int, int],
        preset: str = "quality",
        gpu_type: Optional[str] = None,
        cloud_provider: Optional[str] = None,
    ) -> CostEstimate:
        """Estimate processing costs from video parameters.

        Args:
            frame_count: Total number of frames to process.
            resolution: Video resolution as (width, height).
            preset: Processing preset name.
            gpu_type: GPU type for estimation.
            cloud_provider: Cloud provider for cloud cost estimation.

        Returns:
            CostEstimate with all cost projections.
        """
        gpu_type = gpu_type or self.config.default_gpu_type
        cloud_provider = cloud_provider or self.config.default_cloud_provider

        # Calculate processing time
        processing_time = self._get_processing_time(
            frames=frame_count,
            resolution=resolution,
            preset=preset,
        )

        # Calculate disk usage
        disk_usage = self._get_disk_usage(
            frames=frame_count,
            resolution=resolution,
        )

        # Calculate electricity costs
        electricity_kwh, electricity_cost = self._calculate_electricity(
            time_seconds=processing_time,
            gpu_type=gpu_type,
        )

        # Calculate cloud costs
        cloud_cost = self._calculate_cloud_cost(
            time_seconds=processing_time,
            gpu_type=gpu_type,
            provider=cloud_provider,
        )

        # Determine confidence level
        confidence = self._determine_confidence(
            frame_count=frame_count,
            resolution=resolution,
            gpu_type=gpu_type,
        )

        # Build notes
        notes = self._build_notes(
            preset=preset,
            gpu_type=gpu_type,
            resolution=resolution,
        )

        return CostEstimate(
            estimated_time_seconds=processing_time,
            estimated_disk_gb=disk_usage,
            estimated_electricity_kwh=electricity_kwh,
            estimated_electricity_cost_usd=electricity_cost,
            estimated_cloud_cost_usd=cloud_cost,
            gpu_type=gpu_type,
            confidence=confidence,
            frame_count=frame_count,
            resolution=resolution,
            preset=preset,
            cloud_provider=cloud_provider,
            notes=notes,
        )

    def _get_processing_time(
        self,
        frames: int,
        resolution: Tuple[int, int],
        preset: str,
    ) -> float:
        """Calculate estimated processing time in seconds.

        Args:
            frames: Number of frames to process.
            resolution: Video resolution (width, height).
            preset: Processing preset name.

        Returns:
            Estimated processing time in seconds.
        """
        # Get preset multiplier
        preset_multiplier = PRESET_TIME_MULTIPLIERS.get(preset, 2.5)

        # Calculate resolution multiplier (relative to 1080p)
        pixels = resolution[0] * resolution[1]
        reference_pixels = 1920 * 1080
        resolution_multiplier = (pixels / reference_pixels) ** 0.7  # Sub-linear scaling

        # Calculate base time
        base_time = frames * BASE_FRAME_TIME_SECONDS

        # Apply multipliers
        total_time = base_time * preset_multiplier * resolution_multiplier

        return total_time

    def _get_disk_usage(
        self,
        frames: int,
        resolution: Tuple[int, int],
    ) -> float:
        """Calculate estimated disk usage in GB.

        Args:
            frames: Number of frames.
            resolution: Video resolution (width, height).

        Returns:
            Estimated disk usage in GB.
        """
        height = resolution[1]

        # Find closest resolution tier
        if height <= 480:
            usage_per_frame = DISK_USAGE_PER_FRAME_MB["480p"]
        elif height <= 720:
            usage_per_frame = DISK_USAGE_PER_FRAME_MB["720p"]
        elif height <= 1080:
            usage_per_frame = DISK_USAGE_PER_FRAME_MB["1080p"]
        elif height <= 1440:
            usage_per_frame = DISK_USAGE_PER_FRAME_MB["1440p"]
        elif height <= 2160:
            usage_per_frame = DISK_USAGE_PER_FRAME_MB["2160p"]
        else:
            usage_per_frame = DISK_USAGE_PER_FRAME_MB["4320p"]

        # Frames directory + enhanced directory + output video + overhead
        # Factor in 3x for original frames, enhanced frames, and intermediate
        total_mb = frames * usage_per_frame * 3.0

        # Add 10% overhead for miscellaneous temp files
        total_mb *= 1.1

        return total_mb / 1024  # Convert to GB

    def _calculate_electricity(
        self,
        time_seconds: float,
        gpu_type: str,
    ) -> Tuple[float, float]:
        """Calculate electricity consumption and cost.

        Args:
            time_seconds: Processing time in seconds.
            gpu_type: GPU type for power consumption lookup.

        Returns:
            Tuple of (kWh consumed, cost in USD).
        """
        # Get GPU power consumption
        power_watts = self.config.gpu_power_watts.get(
            gpu_type,
            self.config.gpu_power_watts.get("unknown", 250),
        )

        # Add system overhead (CPU, RAM, cooling, PSU efficiency ~85%)
        total_power = power_watts * 1.3 / 0.85

        # Calculate energy in kWh
        time_hours = time_seconds / 3600
        energy_kwh = (total_power / 1000) * time_hours

        # Calculate cost
        cost_usd = energy_kwh * self.config.electricity_rate_per_kwh

        return (energy_kwh, cost_usd)

    def _calculate_cloud_cost(
        self,
        time_seconds: float,
        gpu_type: str,
        provider: str,
    ) -> Optional[float]:
        """Calculate cloud compute cost.

        Args:
            time_seconds: Processing time in seconds.
            gpu_type: GPU type for pricing lookup.
            provider: Cloud provider name.

        Returns:
            Estimated cloud cost in USD, or None if provider/GPU not found.
        """
        provider_rates = self.config.cloud_rates.get(provider)
        if not provider_rates:
            return None

        hourly_rate = provider_rates.get(gpu_type)
        if hourly_rate is None:
            # Try to find a similar GPU
            for rate_gpu, rate in provider_rates.items():
                if gpu_type in rate_gpu or rate_gpu in gpu_type:
                    hourly_rate = rate
                    break

        if hourly_rate is None:
            return None

        time_hours = time_seconds / 3600
        # Add 10% buffer for startup/shutdown time
        billable_hours = time_hours * 1.1

        return billable_hours * hourly_rate

    def _determine_confidence(
        self,
        frame_count: int,
        resolution: Tuple[int, int],
        gpu_type: str,
    ) -> str:
        """Determine estimation confidence level.

        Args:
            frame_count: Number of frames.
            resolution: Video resolution.
            gpu_type: GPU type.

        Returns:
            Confidence level: "high", "medium", or "low".
        """
        # Check if we have data for this GPU
        has_gpu_data = gpu_type in self.config.gpu_power_watts

        # Check resolution is standard
        height = resolution[1]
        is_standard_resolution = height in [480, 720, 1080, 1440, 2160, 4320]

        # Very long videos have more uncertainty
        is_reasonable_length = frame_count < 500000  # ~5 hours at 30fps

        if has_gpu_data and is_standard_resolution and is_reasonable_length:
            return "high"
        elif has_gpu_data or is_standard_resolution:
            return "medium"
        else:
            return "low"

    def _build_notes(
        self,
        preset: str,
        gpu_type: str,
        resolution: Tuple[int, int],
    ) -> str:
        """Build informational notes about the estimate.

        Args:
            preset: Processing preset.
            gpu_type: GPU type.
            resolution: Video resolution.

        Returns:
            Notes string.
        """
        notes_parts = []

        if preset == "ultimate":
            notes_parts.append("Ultimate preset uses maximum quality settings")
        elif preset == "fast":
            notes_parts.append("Fast preset prioritizes speed over quality")

        if resolution[1] >= 2160:
            notes_parts.append("4K+ resolution significantly increases processing time")

        if gpu_type not in self.config.gpu_power_watts:
            notes_parts.append(f"GPU '{gpu_type}' not in database, using estimates")

        return "; ".join(notes_parts) if notes_parts else ""

    def format_estimate(self, estimate: CostEstimate) -> str:
        """Format cost estimate as human-readable text.

        Args:
            estimate: CostEstimate to format.

        Returns:
            Formatted multi-line string.
        """
        lines = [
            "=" * 60,
            "FrameWright Processing Cost Estimate",
            "=" * 60,
            "",
            "Input Parameters:",
            f"  Frames:     {estimate.frame_count:,}",
            f"  Resolution: {estimate.resolution[0]}x{estimate.resolution[1]}",
            f"  Preset:     {estimate.preset}",
            f"  GPU:        {estimate.gpu_type}",
            "",
            "Time Estimate:",
            f"  Processing: {self._format_time_detailed(estimate.estimated_time_seconds)}",
            "",
            "Storage Requirement:",
            f"  Disk Space: {estimate.estimated_disk_gb:.1f} GB",
            "",
            "Energy Consumption:",
            f"  Electricity: {estimate.estimated_electricity_kwh:.3f} kWh",
            f"  Cost:        ${estimate.estimated_electricity_cost_usd:.2f} "
            f"(at ${self.config.electricity_rate_per_kwh}/kWh)",
            "",
        ]

        if estimate.estimated_cloud_cost_usd is not None:
            lines.extend([
                "Cloud Processing Cost:",
                f"  Provider:   {estimate.cloud_provider}",
                f"  Cost:       ${estimate.estimated_cloud_cost_usd:.2f}",
                "",
            ])

        lines.extend([
            f"Confidence:   {estimate.confidence.upper()}",
        ])

        if estimate.notes:
            lines.extend([
                "",
                "Notes:",
                f"  {estimate.notes}",
            ])

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    @staticmethod
    def _format_time_detailed(seconds: float) -> str:
        """Format seconds into detailed time string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes} min {secs} sec ({seconds:.0f}s total)"
        else:
            hours = int(seconds // 3600)
            remaining = seconds % 3600
            minutes = int(remaining // 60)
            return f"{hours}h {minutes}m ({seconds/3600:.1f} hours)"


def estimate_processing_cost(
    video_path: Path,
    preset: str = "quality",
    gpu_type: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    config: Optional[CostConfig] = None,
) -> CostEstimate:
    """Factory function to estimate processing costs for a video.

    This is a convenience function that creates a CostCalculator
    and estimates costs for the given video file.

    Args:
        video_path: Path to the input video file.
        preset: Processing preset name (fast, quality, archive, etc.).
        gpu_type: GPU type for estimation (e.g., "RTX_4090", "A100_80GB").
        cloud_provider: Cloud provider for cost estimation (runpod, vastai).
        config: Custom cost configuration.

    Returns:
        CostEstimate with all cost projections.

    Example:
        >>> estimate = estimate_processing_cost(
        ...     Path("my_video.mp4"),
        ...     preset="quality",
        ...     gpu_type="RTX_4090",
        ... )
        >>> print(f"Time: {estimate.estimated_time_seconds / 60:.1f} minutes")
        >>> print(f"Disk: {estimate.estimated_disk_gb:.1f} GB")
        >>> print(f"Cost: ${estimate.estimated_electricity_cost_usd:.2f}")
    """
    calculator = CostCalculator(config=config)
    return calculator.estimate_from_video(
        video_path=video_path,
        preset=preset,
        gpu_type=gpu_type,
        cloud_provider=cloud_provider,
    )
