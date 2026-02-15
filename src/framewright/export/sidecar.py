"""Metadata sidecar export for FrameWright.

Generates JSON sidecar files alongside video output containing:
- All restoration settings used
- Processing metrics (PSNR, SSIM, VMAF)
- File checksums (MD5, SHA256)
- Timestamps
- Source/output file info
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FileInfo:
    """Information about a source or output file."""

    path: str
    filename: str
    file_size_bytes: int
    md5_checksum: Optional[str] = None
    sha256_checksum: Optional[str] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileInfo":
        """Create from dictionary."""
        valid_keys = {
            "path", "filename", "file_size_bytes", "md5_checksum",
            "sha256_checksum", "duration_seconds", "width", "height",
            "fps", "codec", "audio_codec", "audio_channels", "audio_sample_rate"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class QualityMetricsData:
    """Quality metrics from restoration processing."""

    psnr_mean: Optional[float] = None
    psnr_min: Optional[float] = None
    psnr_max: Optional[float] = None
    ssim_mean: Optional[float] = None
    ssim_min: Optional[float] = None
    ssim_max: Optional[float] = None
    vmaf_mean: Optional[float] = None
    vmaf_min: Optional[float] = None
    vmaf_max: Optional[float] = None
    vmaf_percentile_5: Optional[float] = None
    vmaf_percentile_95: Optional[float] = None
    quality_grade: Optional[str] = None
    frames_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetricsData":
        """Create from dictionary."""
        valid_keys = {
            "psnr_mean", "psnr_min", "psnr_max",
            "ssim_mean", "ssim_min", "ssim_max",
            "vmaf_mean", "vmaf_min", "vmaf_max",
            "vmaf_percentile_5", "vmaf_percentile_95",
            "quality_grade", "frames_analyzed"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class ProcessingStats:
    """Processing statistics and performance metrics."""

    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    skipped_frames: int = 0
    processing_time_seconds: float = 0.0
    frames_per_second: float = 0.0
    avg_frame_time_ms: float = 0.0
    peak_vram_mb: Optional[int] = None
    avg_vram_mb: Optional[float] = None
    checkpoint_count: int = 0
    retry_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingStats":
        """Create from dictionary."""
        valid_keys = {
            "total_frames", "processed_frames", "failed_frames",
            "skipped_frames", "processing_time_seconds", "frames_per_second",
            "avg_frame_time_ms", "peak_vram_mb", "avg_vram_mb",
            "checkpoint_count", "retry_count", "error_count"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class RestorationSettings:
    """Restoration pipeline settings used for processing."""

    # Core settings
    scale_factor: int = 4
    model_name: str = "realesrgan-x4plus"
    crf: int = 18
    preset: str = "medium"
    output_format: str = "mkv"

    # Enhancement features
    enable_checkpointing: bool = True
    enable_validation: bool = True
    enable_interpolation: bool = False
    enable_deduplication: bool = False
    enable_auto_enhance: bool = False
    enable_colorization: bool = False
    enable_watermark_removal: bool = False
    enable_subtitle_removal: bool = False

    # Advanced features
    enable_tap_denoise: bool = False
    tap_model: Optional[str] = None
    tap_strength: Optional[float] = None
    sr_model: str = "realesrgan"
    face_model: str = "gfpgan"
    enable_qp_artifact_removal: bool = False
    temporal_method: str = "optical_flow"

    # GPU settings
    gpu_id: Optional[int] = None
    enable_multi_gpu: bool = False
    tile_size: Optional[int] = None

    # Quality thresholds
    min_ssim_threshold: float = 0.85
    min_psnr_threshold: float = 25.0

    # Additional settings as flexible dict
    extra_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Remove None values and empty extra_settings
        data = {k: v for k, v in data.items() if v is not None}
        if not data.get("extra_settings"):
            data.pop("extra_settings", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RestorationSettings":
        """Create from dictionary."""
        valid_keys = {
            "scale_factor", "model_name", "crf", "preset", "output_format",
            "enable_checkpointing", "enable_validation", "enable_interpolation",
            "enable_deduplication", "enable_auto_enhance", "enable_colorization",
            "enable_watermark_removal", "enable_subtitle_removal",
            "enable_tap_denoise", "tap_model", "tap_strength",
            "sr_model", "face_model", "enable_qp_artifact_removal",
            "temporal_method", "gpu_id", "enable_multi_gpu", "tile_size",
            "min_ssim_threshold", "min_psnr_threshold", "extra_settings"
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_config(cls, config) -> "RestorationSettings":
        """Create from a Config object.

        Args:
            config: framewright.config.Config instance

        Returns:
            RestorationSettings populated from config
        """
        extra = {}

        # Collect additional settings not in core fields
        optional_fields = [
            "colorization_model", "colorization_strength",
            "rife_model", "target_fps", "deduplication_threshold",
            "scratch_sensitivity", "dust_sensitivity", "grain_reduction",
            "diffusion_steps", "diffusion_guidance", "aesrgan_strength",
            "cross_attention_window", "temporal_blend_strength",
            "enable_vhs_restoration", "enable_authenticity_guard",
            "enable_scene_intelligence", "enable_vmaf_analysis",
        ]

        for field_name in optional_fields:
            if hasattr(config, field_name):
                val = getattr(config, field_name)
                if val is not None:
                    extra[field_name] = val

        return cls(
            scale_factor=getattr(config, "scale_factor", 4),
            model_name=getattr(config, "model_name", "realesrgan-x4plus"),
            crf=getattr(config, "crf", 18),
            preset=getattr(config, "preset", "medium"),
            output_format=getattr(config, "output_format", "mkv"),
            enable_checkpointing=getattr(config, "enable_checkpointing", True),
            enable_validation=getattr(config, "enable_validation", True),
            enable_interpolation=getattr(config, "enable_interpolation", False),
            enable_deduplication=getattr(config, "enable_deduplication", False),
            enable_auto_enhance=getattr(config, "enable_auto_enhance", False),
            enable_colorization=getattr(config, "enable_colorization", False),
            enable_watermark_removal=getattr(config, "enable_watermark_removal", False),
            enable_subtitle_removal=getattr(config, "enable_subtitle_removal", False),
            enable_tap_denoise=getattr(config, "enable_tap_denoise", False),
            tap_model=getattr(config, "tap_model", None) if getattr(config, "enable_tap_denoise", False) else None,
            tap_strength=getattr(config, "tap_strength", None) if getattr(config, "enable_tap_denoise", False) else None,
            sr_model=getattr(config, "sr_model", "realesrgan"),
            face_model=getattr(config, "face_model", "gfpgan"),
            enable_qp_artifact_removal=getattr(config, "enable_qp_artifact_removal", False),
            temporal_method=getattr(config, "temporal_method", "optical_flow"),
            gpu_id=getattr(config, "gpu_id", None),
            enable_multi_gpu=getattr(config, "enable_multi_gpu", False),
            tile_size=getattr(config, "tile_size", None),
            min_ssim_threshold=getattr(config, "min_ssim_threshold", 0.85),
            min_psnr_threshold=getattr(config, "min_psnr_threshold", 25.0),
            extra_settings=extra if extra else {},
        )


@dataclass
class SidecarMetadata:
    """Complete sidecar metadata for a restored video.

    Contains all information about source, output, settings, and metrics.
    """

    # Version for format compatibility
    schema_version: str = "1.0.0"
    generator: str = "FrameWright"
    generator_version: str = "2.0.0"

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # File information
    source_file: Optional[FileInfo] = None
    output_file: Optional[FileInfo] = None

    # Restoration settings
    settings: Optional[RestorationSettings] = None

    # Quality metrics
    quality_metrics: Optional[QualityMetricsData] = None

    # Processing statistics
    processing_stats: Optional[ProcessingStats] = None

    # Processing notes and warnings
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "schema_version": self.schema_version,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "timestamps": {
                "created_at": self.created_at,
            },
        }

        if self.started_at:
            data["timestamps"]["started_at"] = self.started_at
        if self.completed_at:
            data["timestamps"]["completed_at"] = self.completed_at

        if self.source_file:
            data["source_file"] = self.source_file.to_dict()

        if self.output_file:
            data["output_file"] = self.output_file.to_dict()

        if self.settings:
            data["restoration_settings"] = self.settings.to_dict()

        if self.quality_metrics:
            data["quality_metrics"] = self.quality_metrics.to_dict()

        if self.processing_stats:
            data["processing_stats"] = self.processing_stats.to_dict()

        if self.notes:
            data["notes"] = self.notes

        if self.warnings:
            data["warnings"] = self.warnings

        if self.errors:
            data["errors"] = self.errors

        if self.custom:
            data["custom"] = self.custom

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SidecarMetadata":
        """Create from dictionary."""
        timestamps = data.get("timestamps", {})

        source_file = None
        if "source_file" in data:
            source_file = FileInfo.from_dict(data["source_file"])

        output_file = None
        if "output_file" in data:
            output_file = FileInfo.from_dict(data["output_file"])

        settings = None
        if "restoration_settings" in data:
            settings = RestorationSettings.from_dict(data["restoration_settings"])

        quality_metrics = None
        if "quality_metrics" in data:
            quality_metrics = QualityMetricsData.from_dict(data["quality_metrics"])

        processing_stats = None
        if "processing_stats" in data:
            processing_stats = ProcessingStats.from_dict(data["processing_stats"])

        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            generator=data.get("generator", "FrameWright"),
            generator_version=data.get("generator_version", "2.0.0"),
            created_at=timestamps.get("created_at", datetime.now().isoformat()),
            started_at=timestamps.get("started_at"),
            completed_at=timestamps.get("completed_at"),
            source_file=source_file,
            output_file=output_file,
            settings=settings,
            quality_metrics=quality_metrics,
            processing_stats=processing_stats,
            notes=data.get("notes", []),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            custom=data.get("custom", {}),
        )

    def add_note(self, note: str) -> None:
        """Add a note to the metadata."""
        self.notes.append(note)

    def add_warning(self, warning: str) -> None:
        """Add a warning to the metadata."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error to the metadata."""
        self.errors.append(error)


# =============================================================================
# Checksum Calculation
# =============================================================================


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 checksum for a file.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read at a time

    Returns:
        MD5 hex digest string
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 checksum for a file.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read at a time

    Returns:
        SHA256 hex digest string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def calculate_checksums(
    file_path: Path,
    algorithms: List[str] = None,
) -> Dict[str, str]:
    """Calculate multiple checksums for a file.

    Args:
        file_path: Path to the file
        algorithms: List of algorithms ('md5', 'sha256'). Defaults to both.

    Returns:
        Dictionary mapping algorithm name to hex digest
    """
    if algorithms is None:
        algorithms = ["md5", "sha256"]

    checksums = {}

    if "md5" in algorithms:
        checksums["md5"] = calculate_md5(file_path)

    if "sha256" in algorithms:
        checksums["sha256"] = calculate_sha256(file_path)

    return checksums


# =============================================================================
# File Info Extraction
# =============================================================================


def get_video_info(file_path: Path) -> Dict[str, Any]:
    """Extract video file information using ffprobe.

    Args:
        file_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    import json
    import subprocess

    info = {}

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-print_format", "json",
            str(file_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)

            # Format info
            fmt = data.get("format", {})
            info["duration_seconds"] = float(fmt.get("duration", 0))

            # Stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = stream.get("width")
                    info["height"] = stream.get("height")
                    info["codec"] = stream.get("codec_name")

                    # Calculate FPS from r_frame_rate
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        if int(den) > 0:
                            info["fps"] = round(int(num) / int(den), 3)

                elif stream.get("codec_type") == "audio":
                    info["audio_codec"] = stream.get("codec_name")
                    info["audio_channels"] = stream.get("channels")
                    info["audio_sample_rate"] = int(stream.get("sample_rate", 0))

    except Exception as e:
        logger.warning(f"Failed to extract video info: {e}")

    return info


def create_file_info(
    file_path: Path,
    include_checksums: bool = True,
    include_video_info: bool = True,
) -> FileInfo:
    """Create FileInfo from a file path.

    Args:
        file_path: Path to the file
        include_checksums: Whether to calculate checksums
        include_video_info: Whether to extract video properties

    Returns:
        FileInfo instance
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    info = FileInfo(
        path=str(file_path.absolute()),
        filename=file_path.name,
        file_size_bytes=file_path.stat().st_size,
    )

    if include_checksums:
        checksums = calculate_checksums(file_path)
        info.md5_checksum = checksums.get("md5")
        info.sha256_checksum = checksums.get("sha256")

    if include_video_info:
        video_info = get_video_info(file_path)
        info.duration_seconds = video_info.get("duration_seconds")
        info.width = video_info.get("width")
        info.height = video_info.get("height")
        info.fps = video_info.get("fps")
        info.codec = video_info.get("codec")
        info.audio_codec = video_info.get("audio_codec")
        info.audio_channels = video_info.get("audio_channels")
        info.audio_sample_rate = video_info.get("audio_sample_rate")

    return info


# =============================================================================
# Sidecar Generation
# =============================================================================


def generate_sidecar(
    output_video_path: Path,
    source_video_path: Optional[Path] = None,
    config=None,
    metrics=None,
    quality_result=None,
    include_checksums: bool = True,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a JSON sidecar file for a restored video.

    The sidecar file is named the same as the output video with .json extension added.

    Args:
        output_video_path: Path to the output video file
        source_video_path: Path to the source video file (optional)
        config: framewright.config.Config instance (optional)
        metrics: ProcessingMetrics instance from restoration (optional)
        quality_result: VMAFResult or quality metrics dict (optional)
        include_checksums: Whether to calculate file checksums
        custom_metadata: Additional custom metadata to include

    Returns:
        Path to the generated sidecar JSON file
    """
    output_video_path = Path(output_video_path)

    # Create sidecar metadata
    sidecar = SidecarMetadata()

    # Output file info
    if output_video_path.exists():
        try:
            sidecar.output_file = create_file_info(
                output_video_path,
                include_checksums=include_checksums,
            )
        except Exception as e:
            logger.warning(f"Failed to get output file info: {e}")
            sidecar.add_warning(f"Failed to extract output file info: {e}")

    # Source file info
    if source_video_path and Path(source_video_path).exists():
        try:
            sidecar.source_file = create_file_info(
                Path(source_video_path),
                include_checksums=include_checksums,
            )
        except Exception as e:
            logger.warning(f"Failed to get source file info: {e}")
            sidecar.add_warning(f"Failed to extract source file info: {e}")

    # Restoration settings from config
    if config is not None:
        try:
            sidecar.settings = RestorationSettings.from_config(config)
        except Exception as e:
            logger.warning(f"Failed to extract settings from config: {e}")
            sidecar.add_warning(f"Failed to extract restoration settings: {e}")

    # Processing statistics from metrics
    if metrics is not None:
        try:
            sidecar.processing_stats = ProcessingStats(
                total_frames=getattr(metrics, "total_frames", 0),
                processed_frames=getattr(metrics, "processed_frames", 0),
                failed_frames=getattr(metrics, "failed_frames", 0),
                skipped_frames=getattr(metrics, "skipped_frames", 0),
                processing_time_seconds=getattr(metrics, "total_processing_time_seconds", 0.0),
                frames_per_second=getattr(metrics, "frames_per_second", 0.0),
                avg_frame_time_ms=getattr(metrics, "avg_frame_time_ms", 0.0),
                peak_vram_mb=getattr(metrics, "peak_vram_mb", None),
                avg_vram_mb=getattr(metrics, "avg_vram_mb", None),
                checkpoint_count=getattr(metrics, "checkpoint_count", 0),
                retry_count=getattr(metrics, "retry_count", 0),
                error_count=getattr(metrics, "error_count", 0),
            )

            # Extract timestamps
            start_time = getattr(metrics, "start_time", None)
            end_time = getattr(metrics, "end_time", None)

            if start_time:
                sidecar.started_at = start_time.isoformat() if hasattr(start_time, "isoformat") else str(start_time)
            if end_time:
                sidecar.completed_at = end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time)

        except Exception as e:
            logger.warning(f"Failed to extract processing metrics: {e}")
            sidecar.add_warning(f"Failed to extract processing metrics: {e}")

    # Quality metrics
    if quality_result is not None:
        try:
            if hasattr(quality_result, "vmaf_mean"):
                # VMAFResult object
                sidecar.quality_metrics = QualityMetricsData(
                    vmaf_mean=getattr(quality_result, "vmaf_mean", None),
                    vmaf_min=getattr(quality_result, "vmaf_min", None),
                    vmaf_max=getattr(quality_result, "vmaf_max", None),
                    vmaf_percentile_5=getattr(quality_result, "vmaf_percentile_5", None),
                    vmaf_percentile_95=getattr(quality_result, "vmaf_percentile_95", None),
                    psnr_mean=getattr(quality_result, "psnr_mean", None),
                    ssim_mean=getattr(quality_result, "ssim_mean", None),
                    quality_grade=getattr(quality_result, "get_quality_grade", lambda: None)(),
                    frames_analyzed=getattr(quality_result, "frame_count", 0),
                )
            elif isinstance(quality_result, dict):
                # Dictionary format
                sidecar.quality_metrics = QualityMetricsData.from_dict(quality_result)

        except Exception as e:
            logger.warning(f"Failed to extract quality metrics: {e}")
            sidecar.add_warning(f"Failed to extract quality metrics: {e}")

    # Custom metadata
    if custom_metadata:
        sidecar.custom = custom_metadata

    # Write sidecar file
    sidecar_path = Path(str(output_video_path) + ".json")

    try:
        sidecar_data = sidecar.to_dict()
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated sidecar metadata: {sidecar_path}")

    except Exception as e:
        logger.error(f"Failed to write sidecar file: {e}")
        raise

    return sidecar_path


def load_sidecar(sidecar_path: Path) -> SidecarMetadata:
    """Load sidecar metadata from a JSON file.

    Args:
        sidecar_path: Path to the sidecar JSON file

    Returns:
        SidecarMetadata instance

    Raises:
        FileNotFoundError: If sidecar file doesn't exist
        ValueError: If sidecar file is invalid
    """
    sidecar_path = Path(sidecar_path)

    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar file not found: {sidecar_path}")

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SidecarMetadata.from_dict(data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in sidecar file: {e}")


def get_sidecar_path(video_path: Path) -> Path:
    """Get the expected sidecar path for a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Path where the sidecar JSON would be located
    """
    return Path(str(video_path) + ".json")


def sidecar_exists(video_path: Path) -> bool:
    """Check if a sidecar file exists for a video.

    Args:
        video_path: Path to the video file

    Returns:
        True if sidecar exists
    """
    return get_sidecar_path(video_path).exists()


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_sidecar(
    output_path: Path,
    source_path: Optional[Path] = None,
) -> Path:
    """Generate a minimal sidecar with just file info and checksums.

    Args:
        output_path: Path to output video
        source_path: Optional path to source video

    Returns:
        Path to generated sidecar
    """
    return generate_sidecar(
        output_video_path=output_path,
        source_video_path=source_path,
        include_checksums=True,
    )


def verify_sidecar(
    video_path: Path,
    check_checksums: bool = True,
) -> Dict[str, Any]:
    """Verify a video against its sidecar metadata.

    Args:
        video_path: Path to the video file
        check_checksums: Whether to verify checksums

    Returns:
        Dictionary with verification results
    """
    video_path = Path(video_path)
    sidecar_path = get_sidecar_path(video_path)

    result = {
        "valid": True,
        "video_exists": video_path.exists(),
        "sidecar_exists": sidecar_path.exists(),
        "issues": [],
    }

    if not result["video_exists"]:
        result["valid"] = False
        result["issues"].append("Video file not found")
        return result

    if not result["sidecar_exists"]:
        result["valid"] = False
        result["issues"].append("Sidecar file not found")
        return result

    try:
        sidecar = load_sidecar(sidecar_path)

        # Check file size
        current_size = video_path.stat().st_size
        if sidecar.output_file and sidecar.output_file.file_size_bytes != current_size:
            result["valid"] = False
            result["issues"].append(
                f"File size mismatch: expected {sidecar.output_file.file_size_bytes}, "
                f"got {current_size}"
            )

        # Check checksums if requested
        if check_checksums and sidecar.output_file:
            if sidecar.output_file.md5_checksum:
                current_md5 = calculate_md5(video_path)
                if current_md5 != sidecar.output_file.md5_checksum:
                    result["valid"] = False
                    result["issues"].append("MD5 checksum mismatch")
                else:
                    result["md5_verified"] = True

            if sidecar.output_file.sha256_checksum:
                current_sha256 = calculate_sha256(video_path)
                if current_sha256 != sidecar.output_file.sha256_checksum:
                    result["valid"] = False
                    result["issues"].append("SHA256 checksum mismatch")
                else:
                    result["sha256_verified"] = True

        result["sidecar"] = sidecar

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading sidecar: {e}")

    return result
