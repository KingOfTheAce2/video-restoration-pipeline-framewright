"""Proxy Workflow Module for FrameWright.

This module provides a proxy-based workflow for efficient video restoration tuning.
The workflow allows users to:
1. Create a low-resolution proxy of their video for fast iteration
2. Tune restoration settings on the proxy with quick feedback
3. Save the tuned settings
4. Apply the settings to the full-resolution original

This dramatically speeds up the workflow for long or high-resolution videos.
"""

import json
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.ffmpeg import (
    check_ffmpeg_installed,
    get_video_info,
    get_video_resolution,
    check_video_has_audio,
    FFmpegError,
)


@dataclass
class ProxyConfig:
    """Configuration for proxy video creation.

    Attributes:
        scale: Scale factor for proxy resolution (0.25 = 1/4 resolution).
        quality: CRF value for proxy encoding (higher = smaller file, lower quality).
        format: Output container format for the proxy.
        preserve_audio: Whether to include audio in the proxy.
    """

    scale: float = 0.25  # 1/4 resolution by default
    quality: int = 28  # CRF for proxy (higher = faster iteration)
    format: str = "mp4"
    preserve_audio: bool = True

    def __post_init__(self) -> None:
        """Validate proxy configuration."""
        if not 0.05 <= self.scale <= 1.0:
            raise ValueError("scale must be between 0.05 and 1.0")

        if not 0 <= self.quality <= 51:
            raise ValueError("quality (CRF) must be between 0 and 51")

        valid_formats = {"mp4", "mkv", "webm", "mov"}
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of: {valid_formats}")


@dataclass
class ProxySettings:
    """Settings saved from proxy tuning session.

    Stores the relationship between a proxy and its original video,
    along with the restoration settings that were tuned on the proxy.

    Attributes:
        original_path: Path to the original high-resolution video.
        proxy_path: Path to the proxy video that was used for tuning.
        scale_factor: Scale factor that was used to create the proxy.
        settings: Dictionary of restoration settings tuned on the proxy.
        created_at: ISO timestamp of when settings were saved.
        original_resolution: Original video resolution (width, height).
        proxy_resolution: Proxy video resolution (width, height).
        notes: Optional user notes about the tuning session.
    """

    original_path: str
    proxy_path: str
    scale_factor: float
    settings: Dict[str, Any]
    created_at: str
    original_resolution: tuple = field(default_factory=lambda: (0, 0))
    proxy_resolution: tuple = field(default_factory=lambda: (0, 0))
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_path": self.original_path,
            "proxy_path": self.proxy_path,
            "scale_factor": self.scale_factor,
            "settings": self.settings,
            "created_at": self.created_at,
            "original_resolution": list(self.original_resolution),
            "proxy_resolution": list(self.proxy_resolution),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProxySettings":
        """Create ProxySettings from dictionary."""
        return cls(
            original_path=data["original_path"],
            proxy_path=data["proxy_path"],
            scale_factor=data["scale_factor"],
            settings=data["settings"],
            created_at=data["created_at"],
            original_resolution=tuple(data.get("original_resolution", (0, 0))),
            proxy_resolution=tuple(data.get("proxy_resolution", (0, 0))),
            notes=data.get("notes", ""),
        )


class ProxyWorkflow:
    """Manages the proxy-based restoration workflow.

    This class provides methods to create proxy videos, save and load
    restoration settings, and apply tuned settings to original videos.

    Example workflow:
        >>> workflow = ProxyWorkflow(Path("./work"))
        >>> proxy_path = workflow.create_proxy(Path("input.mp4"), scale=0.25)
        >>> # User tunes settings on proxy using framewright restore...
        >>> settings_path = workflow.save_settings(proxy_path, {"crf": 18, "scale_factor": 4})
        >>> output = workflow.apply_to_original(
        ...     Path("input.mp4"),
        ...     settings_path,
        ...     Path("output.mp4")
        ... )
    """

    def __init__(self, work_dir: Path) -> None:
        """Initialize the proxy workflow.

        Args:
            work_dir: Working directory for proxy files and settings.
        """
        self.work_dir = Path(work_dir)
        self.proxy_dir = self.work_dir / "proxies"
        self.settings_dir = self.work_dir / "settings"

        # Ensure directories exist
        self.proxy_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)

    def create_proxy(
        self,
        input_path: Path,
        scale: float = 0.25,
        config: Optional[ProxyConfig] = None,
    ) -> Path:
        """Create a low-resolution proxy video for fast tuning.

        Args:
            input_path: Path to the original video file.
            scale: Scale factor for proxy resolution (0.25 = 1/4 resolution).
            config: Optional ProxyConfig for advanced settings.

        Returns:
            Path to the created proxy video.

        Raises:
            FileNotFoundError: If input video doesn't exist.
            FFmpegError: If proxy creation fails.
        """
        check_ffmpeg_installed()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Use provided config or create default
        if config is None:
            config = ProxyConfig(scale=scale)

        # Generate proxy filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        proxy_name = f"{input_path.stem}_proxy_{timestamp}.{config.format}"
        proxy_path = self.proxy_dir / proxy_name

        # Get original resolution
        orig_width, orig_height = get_video_resolution(input_path)

        # Calculate proxy resolution (ensure even dimensions)
        proxy_width = max(2, int(orig_width * config.scale) // 2 * 2)
        proxy_height = max(2, int(orig_height * config.scale) // 2 * 2)

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", f"scale={proxy_width}:{proxy_height}",
            "-c:v", "libx264",
            "-crf", str(config.quality),
            "-preset", "fast",  # Fast encoding for proxy
        ]

        # Handle audio
        if config.preserve_audio and check_video_has_audio(input_path):
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.extend(["-an"])  # No audio

        cmd.extend(["-y", str(proxy_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Failed to create proxy: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Proxy creation timed out (1 hour limit)")

        # Save proxy metadata
        metadata = {
            "original_path": str(input_path.absolute()),
            "original_resolution": [orig_width, orig_height],
            "proxy_resolution": [proxy_width, proxy_height],
            "scale_factor": config.scale,
            "created_at": datetime.now().isoformat(),
        }
        metadata_path = proxy_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return proxy_path

    def save_settings(
        self,
        proxy_path: Path,
        settings: Dict[str, Any],
        output_path: Optional[Path] = None,
        notes: str = "",
    ) -> Path:
        """Save restoration settings tuned on a proxy.

        Args:
            proxy_path: Path to the proxy video that was used for tuning.
            settings: Dictionary of restoration settings.
            output_path: Optional custom path for settings file.
            notes: Optional notes about the tuning session.

        Returns:
            Path to the saved settings JSON file.

        Raises:
            FileNotFoundError: If proxy video or its metadata doesn't exist.
            ValueError: If settings dictionary is empty.
        """
        proxy_path = Path(proxy_path)
        if not proxy_path.exists():
            raise FileNotFoundError(f"Proxy video not found: {proxy_path}")

        if not settings:
            raise ValueError("Settings dictionary cannot be empty")

        # Load proxy metadata
        metadata_path = proxy_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            # Try to reconstruct minimal metadata
            metadata = {
                "original_path": "",
                "original_resolution": [0, 0],
                "proxy_resolution": list(get_video_resolution(proxy_path)),
                "scale_factor": 0.25,  # Assume default
            }

        # Create ProxySettings
        proxy_settings = ProxySettings(
            original_path=metadata.get("original_path", ""),
            proxy_path=str(proxy_path.absolute()),
            scale_factor=metadata.get("scale_factor", 0.25),
            settings=settings,
            created_at=datetime.now().isoformat(),
            original_resolution=tuple(metadata.get("original_resolution", [0, 0])),
            proxy_resolution=tuple(metadata.get("proxy_resolution", [0, 0])),
            notes=notes,
        )

        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            settings_name = f"{proxy_path.stem}_settings_{timestamp}.json"
            output_path = self.settings_dir / settings_name
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save settings
        with open(output_path, "w") as f:
            json.dump(proxy_settings.to_dict(), f, indent=2)

        return output_path

    def load_settings(self, settings_path: Path) -> ProxySettings:
        """Load restoration settings from a file.

        Args:
            settings_path: Path to the settings JSON file.

        Returns:
            ProxySettings instance with loaded settings.

        Raises:
            FileNotFoundError: If settings file doesn't exist.
            ValueError: If settings file is invalid.
        """
        settings_path = Path(settings_path)
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")

        try:
            with open(settings_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid settings file: {e}")

        required_keys = {"original_path", "proxy_path", "scale_factor", "settings", "created_at"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Settings file missing required keys: {missing}")

        return ProxySettings.from_dict(data)

    def apply_to_original(
        self,
        original_path: Path,
        settings_path: Path,
        output_path: Path,
    ) -> Path:
        """Apply proxy-tuned settings to the original video.

        This method loads the settings tuned on a proxy, scales them
        appropriately for the original resolution, and runs the
        restoration pipeline on the full-resolution video.

        Args:
            original_path: Path to the original high-resolution video.
            settings_path: Path to the settings JSON file.
            output_path: Path for the restored output video.

        Returns:
            Path to the restored video.

        Raises:
            FileNotFoundError: If original video or settings don't exist.
            ValueError: If settings are incompatible with the video.
        """
        original_path = Path(original_path)
        settings_path = Path(settings_path)
        output_path = Path(output_path)

        if not original_path.exists():
            raise FileNotFoundError(f"Original video not found: {original_path}")

        # Load settings
        proxy_settings = self.load_settings(settings_path)

        # Scale settings from proxy to original resolution
        scaled_settings = self._scale_settings(
            proxy_settings.settings,
            from_scale=proxy_settings.scale_factor,
            to_scale=1.0,
        )

        # Import here to avoid circular dependency
        from ..config import Config
        from ..restorer import VideoRestorer

        # Create config from scaled settings
        config_dict = {
            "project_dir": self.work_dir / "restore_jobs" / original_path.stem,
            **scaled_settings,
        }

        # Filter to valid config keys
        valid_config_keys = {
            "project_dir", "output_dir", "scale_factor", "model_name", "crf",
            "preset", "output_format", "enable_checkpointing", "checkpoint_interval",
            "enable_validation", "min_ssim_threshold", "min_psnr_threshold",
            "enable_disk_validation", "disk_safety_margin", "enable_vram_monitoring",
            "tile_size", "max_retries", "retry_delay", "parallel_frames",
            "continue_on_error", "require_gpu", "gpu_id", "enable_multi_gpu",
            "gpu_ids", "gpu_load_balance_strategy", "workers_per_gpu",
            "enable_work_stealing", "enable_interpolation", "target_fps",
            "rife_model", "rife_gpu_id", "enable_deduplication",
            "deduplication_threshold", "expected_source_fps", "enable_auto_enhance",
            "auto_detect_content", "auto_defect_repair", "auto_face_restore",
            "scratch_sensitivity", "dust_sensitivity", "grain_reduction",
            "model_download_dir", "model_dir", "enable_colorization",
            "colorization_model", "colorization_strength", "enable_watermark_removal",
            "watermark_mask_path", "watermark_auto_detect", "enable_subtitle_removal",
            "subtitle_region", "subtitle_ocr_engine", "subtitle_languages",
            "enable_tap_denoise", "tap_model", "tap_strength", "tap_preserve_grain",
            "sr_model", "diffusion_steps", "diffusion_guidance", "face_model",
            "aesrgan_strength", "enable_qp_artifact_removal", "qp_auto_detect",
            "qp_strength", "enable_frame_generation", "frame_gen_model",
            "max_gap_frames", "temporal_method", "cross_attention_window",
            "temporal_blend_strength",
        }

        filtered_config = {
            k: v for k, v in config_dict.items()
            if k in valid_config_keys
        }

        config = Config(**filtered_config)
        config.create_directories()

        # Run restoration
        restorer = VideoRestorer(config)
        result = restorer.restore(str(original_path))

        # Copy to final output path
        if result and Path(result).exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result, output_path)
            return output_path

        raise RuntimeError("Restoration failed to produce output")

    def _scale_settings(
        self,
        settings: Dict[str, Any],
        from_scale: float,
        to_scale: float,
    ) -> Dict[str, Any]:
        """Scale settings from one resolution to another.

        Some settings need to be adjusted when moving from proxy
        resolution to original resolution. This method handles
        those adjustments.

        Args:
            settings: Original settings dictionary.
            from_scale: Scale factor of the source resolution (e.g., 0.25 for proxy).
            to_scale: Scale factor of the target resolution (e.g., 1.0 for original).

        Returns:
            Adjusted settings dictionary.
        """
        scaled = settings.copy()
        scale_ratio = to_scale / from_scale

        # Settings that need resolution-based scaling
        resolution_dependent = {
            "tile_size": lambda v: max(0, int(v * scale_ratio)) if v and v > 0 else v,
        }

        # Settings that might need adjustment for full resolution
        # (but values stay the same, just need validation)
        # Most restoration settings are resolution-independent

        for key, scale_func in resolution_dependent.items():
            if key in scaled and scaled[key] is not None:
                scaled[key] = scale_func(scaled[key])

        # Ensure CRF is appropriate for full resolution
        # (might want slightly better quality for full res)
        if "crf" in scaled:
            # Optionally reduce CRF for full resolution (better quality)
            # scaled["crf"] = max(0, scaled["crf"] - 2)
            pass  # Keep same CRF as tuned

        return scaled

    def cleanup_proxy(self, proxy_path: Path) -> None:
        """Remove a proxy video and its metadata.

        Args:
            proxy_path: Path to the proxy video to remove.
        """
        proxy_path = Path(proxy_path)

        # Remove proxy video
        if proxy_path.exists():
            proxy_path.unlink()

        # Remove metadata
        metadata_path = proxy_path.with_suffix(".json")
        if metadata_path.exists():
            metadata_path.unlink()

    def list_proxies(self) -> list:
        """List all proxy videos in the work directory.

        Returns:
            List of dictionaries with proxy info.
        """
        proxies = []
        for proxy_file in self.proxy_dir.glob("*_proxy_*.mp4"):
            metadata_path = proxy_file.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                proxies.append({
                    "path": str(proxy_file),
                    "original": metadata.get("original_path", ""),
                    "scale_factor": metadata.get("scale_factor", 0.25),
                    "created_at": metadata.get("created_at", ""),
                })
            else:
                proxies.append({
                    "path": str(proxy_file),
                    "original": "",
                    "scale_factor": 0.25,
                    "created_at": "",
                })
        return proxies

    def list_settings(self) -> list:
        """List all saved settings files.

        Returns:
            List of dictionaries with settings info.
        """
        settings_list = []
        for settings_file in self.settings_dir.glob("*_settings_*.json"):
            try:
                with open(settings_file) as f:
                    data = json.load(f)
                settings_list.append({
                    "path": str(settings_file),
                    "original": data.get("original_path", ""),
                    "proxy": data.get("proxy_path", ""),
                    "created_at": data.get("created_at", ""),
                    "notes": data.get("notes", ""),
                })
            except (json.JSONDecodeError, KeyError):
                settings_list.append({
                    "path": str(settings_file),
                    "original": "",
                    "proxy": "",
                    "created_at": "",
                    "notes": "",
                })
        return settings_list


# Factory functions for convenience

def create_proxy(
    video_path: Path,
    scale: float = 0.25,
    work_dir: Optional[Path] = None,
) -> Path:
    """Create a proxy video for fast restoration tuning.

    This is a convenience function that creates a ProxyWorkflow
    instance and generates a proxy video.

    Args:
        video_path: Path to the original video.
        scale: Scale factor for proxy resolution (0.25 = 1/4 resolution).
        work_dir: Optional working directory (defaults to video directory).

    Returns:
        Path to the created proxy video.

    Example:
        >>> proxy = create_proxy(Path("my_video.mp4"), scale=0.25)
        >>> print(f"Proxy created: {proxy}")
    """
    video_path = Path(video_path)

    if work_dir is None:
        work_dir = video_path.parent / ".framewright_proxy"

    workflow = ProxyWorkflow(work_dir)
    return workflow.create_proxy(video_path, scale=scale)


def apply_proxy_settings(
    original: Path,
    settings: Path,
    output: Path,
    work_dir: Optional[Path] = None,
) -> Path:
    """Apply proxy-tuned settings to the original video.

    This is a convenience function that creates a ProxyWorkflow
    instance and applies saved settings to the full-resolution video.

    Args:
        original: Path to the original high-resolution video.
        settings: Path to the settings JSON file saved from proxy tuning.
        output: Path for the restored output video.
        work_dir: Optional working directory (defaults to video directory).

    Returns:
        Path to the restored video.

    Example:
        >>> result = apply_proxy_settings(
        ...     Path("my_video.mp4"),
        ...     Path("settings.json"),
        ...     Path("restored.mp4")
        ... )
        >>> print(f"Restored: {result}")
    """
    original = Path(original)
    settings = Path(settings)
    output = Path(output)

    if work_dir is None:
        work_dir = original.parent / ".framewright_proxy"

    workflow = ProxyWorkflow(work_dir)
    return workflow.apply_to_original(original, settings, output)
