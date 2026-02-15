"""Core configuration module for FrameWright video restoration pipeline.

This module provides the foundational configuration classes for FrameWright:
- FrameWrightConfig: Main configuration dataclass
- ProcessorConfig: Per-processor settings
- HardwareConfig: Hardware preferences
- OutputConfig: Output settings (format, codec, quality)

Configuration can be loaded from YAML/JSON files and supports progressive
disclosure - simple defaults for beginners with advanced options available.

Example usage:

    >>> from framewright.core.config import (
    ...     FrameWrightConfig,
    ...     load_config,
    ...     save_config,
    ... )
    >>>
    >>> # Create with simple defaults
    >>> config = FrameWrightConfig(project_dir="/path/to/project")
    >>>
    >>> # Load from file
    >>> config = load_config("config.yaml")
    >>>
    >>> # Merge configurations
    >>> merged = merge_configs(base_config, override_config)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for generic config operations
T = TypeVar("T", bound="BaseConfig")


class ConfigError(Exception):
    """Error in configuration."""
    pass


class ValidationError(ConfigError):
    """Configuration validation error."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        valid_values: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.field_value = field_value
        self.valid_values = valid_values


# =============================================================================
# Enums for Configuration Options
# =============================================================================


class ScaleFactor(int, Enum):
    """Valid scale factors for upscaling."""
    X2 = 2
    X4 = 4


class OutputFormat(str, Enum):
    """Supported output video formats."""
    MKV = "mkv"
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


class VideoCodec(str, Enum):
    """Supported video codecs."""
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    PRORES = "prores_ks"
    DNXHD = "dnxhd"


class AudioCodec(str, Enum):
    """Supported audio codecs."""
    AAC = "aac"
    OPUS = "libopus"
    FLAC = "flac"
    PCM = "pcm_s16le"
    COPY = "copy"


class PixelFormat(str, Enum):
    """Supported pixel formats."""
    YUV420P = "yuv420p"
    YUV422P = "yuv422p"
    YUV444P = "yuv444p"
    YUV420P10LE = "yuv420p10le"
    YUV422P10LE = "yuv422p10le"


class EncodingPreset(str, Enum):
    """FFmpeg encoding speed presets."""
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"


class GPULoadStrategy(str, Enum):
    """Multi-GPU load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    VRAM_AWARE = "vram_aware"
    WEIGHTED = "weighted"


class TemporalMethod(str, Enum):
    """Temporal consistency methods."""
    OPTICAL_FLOW = "optical_flow"
    CROSS_ATTENTION = "cross_attention"
    HYBRID = "hybrid"
    RAFT = "raft"


# =============================================================================
# Base Configuration Class
# =============================================================================


@dataclass
class BaseConfig:
    """Base class for all configuration dataclasses.

    Provides common serialization and validation methods.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, BaseConfig):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, BaseConfig)
                    else str(v) if isinstance(v, Path)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            Configuration instance.
        """
        # Get field types for conversion
        import dataclasses
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}

        converted = {}
        for key, value in data.items():
            if key not in field_types:
                continue

            field_type = field_types[key]

            # Handle Path conversion
            if field_type == Path or field_type == Optional[Path]:
                if value is not None:
                    converted[key] = Path(value)
                else:
                    converted[key] = None
            else:
                converted[key] = value

        return cls(**converted)

    def validate(self) -> List[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages (empty if valid).
        """
        return []

    def merge_with(self: T, override: T) -> T:
        """Merge this configuration with an override.

        Non-None values in override take precedence.

        Args:
            override: Configuration with override values.

        Returns:
            New merged configuration.
        """
        base_dict = self.to_dict()
        override_dict = override.to_dict()

        for key, value in override_dict.items():
            if value is not None:
                base_dict[key] = value

        return self.__class__.from_dict(base_dict)


# =============================================================================
# Hardware Configuration
# =============================================================================


@dataclass
class HardwareConfig(BaseConfig):
    """Hardware preferences and constraints.

    Attributes:
        require_gpu: Require GPU processing (blocks CPU fallback).
        gpu_id: Specific GPU index to use (None for auto-select).
        enable_multi_gpu: Enable multi-GPU distribution.
        gpu_ids: List of GPU IDs for multi-GPU mode.
        load_balance_strategy: Multi-GPU load balancing strategy.
        workers_per_gpu: Number of worker threads per GPU.
        enable_work_stealing: Allow idle workers to steal work.
        tile_size: Processing tile size (0 for auto, None for no tiling).
        max_vram_usage: Maximum VRAM usage fraction (0.0-1.0).
        enable_vram_monitoring: Monitor VRAM usage during processing.
        enable_thermal_throttling: Reduce load on thermal issues.
        thermal_limit_celsius: Temperature limit for throttling.
        max_cpu_threads: Maximum CPU threads (None for auto).
        enable_memory_optimization: Enable aggressive memory optimization.
    """

    # GPU settings
    require_gpu: bool = True
    gpu_id: Optional[int] = None
    enable_multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None
    load_balance_strategy: str = "vram_aware"
    workers_per_gpu: int = 2
    enable_work_stealing: bool = True

    # Memory settings
    tile_size: Optional[int] = 0  # 0 = auto, None = no tiling
    max_vram_usage: float = 0.9
    enable_vram_monitoring: bool = True

    # Thermal settings
    enable_thermal_throttling: bool = True
    thermal_limit_celsius: int = 85

    # CPU settings
    max_cpu_threads: Optional[int] = None

    # Optimization
    enable_memory_optimization: bool = True

    def validate(self) -> List[str]:
        """Validate hardware configuration."""
        errors = []

        if self.gpu_id is not None and self.gpu_id < 0:
            errors.append(f"gpu_id must be non-negative, got {self.gpu_id}")

        if self.gpu_ids is not None:
            for gid in self.gpu_ids:
                if gid < 0:
                    errors.append(f"GPU IDs must be non-negative, got {gid}")

        valid_strategies = [s.value for s in GPULoadStrategy]
        if self.load_balance_strategy not in valid_strategies:
            errors.append(
                f"Invalid load_balance_strategy '{self.load_balance_strategy}'. "
                f"Valid: {valid_strategies}"
            )

        if self.workers_per_gpu < 1:
            errors.append(f"workers_per_gpu must be >= 1, got {self.workers_per_gpu}")

        if self.tile_size is not None and self.tile_size < 0:
            errors.append(f"tile_size must be non-negative, got {self.tile_size}")

        if not 0.0 <= self.max_vram_usage <= 1.0:
            errors.append(f"max_vram_usage must be 0.0-1.0, got {self.max_vram_usage}")

        if self.thermal_limit_celsius < 50 or self.thermal_limit_celsius > 100:
            errors.append(
                f"thermal_limit_celsius should be 50-100, got {self.thermal_limit_celsius}"
            )

        if self.max_cpu_threads is not None and self.max_cpu_threads < 1:
            errors.append(f"max_cpu_threads must be >= 1, got {self.max_cpu_threads}")

        return errors


# =============================================================================
# Output Configuration
# =============================================================================


@dataclass
class OutputConfig(BaseConfig):
    """Output video settings.

    Attributes:
        format: Output container format (mkv, mp4, etc.).
        video_codec: Video codec to use.
        audio_codec: Audio codec to use.
        pixel_format: Pixel format for output.
        crf: Constant Rate Factor (0-51, lower is better quality).
        preset: Encoding speed preset.
        bitrate: Target bitrate (None for CRF mode).
        max_bitrate: Maximum bitrate for VBR.
        audio_bitrate: Audio bitrate in kbps.
        sample_rate: Audio sample rate in Hz.
        include_metadata: Include source metadata in output.
        copy_chapters: Copy chapter markers.
        copy_subtitles: Copy subtitle streams.
        output_template: Template for output filename.
        create_preview: Create low-res preview alongside output.
        preview_scale: Preview scale factor (e.g., 0.25 for quarter res).
    """

    # Container and codecs
    format: str = "mkv"
    video_codec: str = "libx265"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p10le"

    # Quality settings
    crf: int = 18
    preset: str = "medium"
    bitrate: Optional[str] = None  # e.g., "10M"
    max_bitrate: Optional[str] = None  # e.g., "15M"

    # Audio settings
    audio_bitrate: int = 192
    sample_rate: int = 48000

    # Metadata and streams
    include_metadata: bool = True
    copy_chapters: bool = True
    copy_subtitles: bool = True

    # Output naming
    output_template: str = "{input_stem}_restored.{format}"

    # Preview
    create_preview: bool = False
    preview_scale: float = 0.25

    def validate(self) -> List[str]:
        """Validate output configuration."""
        errors = []

        valid_formats = [f.value for f in OutputFormat]
        if self.format not in valid_formats:
            errors.append(
                f"Invalid format '{self.format}'. Valid: {valid_formats}"
            )

        valid_video_codecs = [c.value for c in VideoCodec]
        if self.video_codec not in valid_video_codecs:
            errors.append(
                f"Invalid video_codec '{self.video_codec}'. Valid: {valid_video_codecs}"
            )

        valid_audio_codecs = [c.value for c in AudioCodec]
        if self.audio_codec not in valid_audio_codecs:
            errors.append(
                f"Invalid audio_codec '{self.audio_codec}'. Valid: {valid_audio_codecs}"
            )

        valid_presets = [p.value for p in EncodingPreset]
        if self.preset not in valid_presets:
            errors.append(
                f"Invalid preset '{self.preset}'. Valid: {valid_presets}"
            )

        if not 0 <= self.crf <= 51:
            errors.append(f"crf must be 0-51, got {self.crf}")

        if self.audio_bitrate < 32 or self.audio_bitrate > 512:
            errors.append(f"audio_bitrate should be 32-512 kbps, got {self.audio_bitrate}")

        if self.sample_rate < 8000 or self.sample_rate > 192000:
            errors.append(
                f"sample_rate should be 8000-192000 Hz, got {self.sample_rate}"
            )

        if not 0.0 < self.preview_scale <= 1.0:
            errors.append(f"preview_scale must be 0.0-1.0, got {self.preview_scale}")

        return errors

    def get_ffmpeg_args(self) -> List[str]:
        """Get FFmpeg arguments for this output configuration.

        Returns:
            List of FFmpeg command-line arguments.
        """
        args = [
            "-c:v", self.video_codec,
            "-preset", self.preset,
            "-pix_fmt", self.pixel_format,
        ]

        if self.bitrate:
            args.extend(["-b:v", self.bitrate])
            if self.max_bitrate:
                args.extend(["-maxrate", self.max_bitrate, "-bufsize", self.max_bitrate])
        else:
            args.extend(["-crf", str(self.crf)])

        if self.audio_codec != "copy":
            args.extend([
                "-c:a", self.audio_codec,
                "-b:a", f"{self.audio_bitrate}k",
                "-ar", str(self.sample_rate),
            ])
        else:
            args.extend(["-c:a", "copy"])

        if self.copy_subtitles:
            args.extend(["-c:s", "copy"])

        return args


# =============================================================================
# Processor Configuration
# =============================================================================


@dataclass
class ProcessorConfig(BaseConfig):
    """Configuration for a specific processor/enhancement stage.

    Attributes:
        enabled: Whether this processor is enabled.
        strength: Processing strength/intensity (0.0-1.0).
        model_name: Model to use (if applicable).
        model_path: Custom model path (overrides model_name).
        priority: Processing order priority (lower runs first).
        condition: Condition expression for when to run.
        parameters: Additional processor-specific parameters.
        fallback_processor: Processor to use if this one fails.
    """

    enabled: bool = True
    strength: float = 1.0
    model_name: Optional[str] = None
    model_path: Optional[Path] = None
    priority: int = 100
    condition: Optional[str] = None  # e.g., "scene.has_faces"
    parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_processor: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate processor configuration."""
        errors = []

        if not 0.0 <= self.strength <= 1.0:
            errors.append(f"strength must be 0.0-1.0, got {self.strength}")

        if self.model_path is not None and not self.model_path.exists():
            errors.append(f"Model path does not exist: {self.model_path}")

        return errors


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class FrameWrightConfig(BaseConfig):
    """Main configuration for FrameWright video restoration.

    This is the primary configuration class that aggregates all settings
    needed for video restoration. It supports progressive disclosure -
    simple defaults for basic usage with advanced options available.

    Attributes:
        project_dir: Root directory for processing files.
        output_dir: Directory for output files (defaults to project_dir/output).

        scale_factor: Upscaling factor (2 or 4).
        model_name: Default model for super-resolution.

        hardware: Hardware configuration.
        output: Output settings.
        processors: Per-processor configurations.

        enable_checkpointing: Enable checkpoint/resume.
        checkpoint_interval: Frames between checkpoints.

        enable_validation: Enable quality validation.
        min_ssim_threshold: Minimum SSIM for validation.
        min_psnr_threshold: Minimum PSNR for validation.

        parallel_frames: Frames to process in parallel.
        max_retries: Retry attempts for transient errors.
        retry_delay: Initial delay between retries.
        continue_on_error: Continue if some frames fail.

        preset_name: Name of preset this config was derived from.
        config_version: Configuration format version.
    """

    # Required paths
    project_dir: Path = field(default_factory=lambda: Path.cwd())
    output_dir: Optional[Path] = None

    # Core processing settings
    scale_factor: int = 4
    model_name: str = "realesrgan-x4plus"

    # Sub-configurations
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processors: Dict[str, ProcessorConfig] = field(default_factory=dict)

    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100

    # Validation
    enable_validation: bool = True
    min_ssim_threshold: float = 0.85
    min_psnr_threshold: float = 25.0

    # Processing control
    parallel_frames: int = 1
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True

    # Metadata
    preset_name: Optional[str] = None
    config_version: str = "1.0"

    # Derived paths (computed in __post_init__)
    temp_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    enhanced_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    model_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived paths and convert types."""
        # Convert paths
        if not isinstance(self.project_dir, Path):
            self.project_dir = Path(self.project_dir)

        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Convert sub-configs from dicts
        if isinstance(self.hardware, dict):
            self.hardware = HardwareConfig.from_dict(self.hardware)

        if isinstance(self.output, dict):
            self.output = OutputConfig.from_dict(self.output)

        if self.processors:
            for name, proc_config in list(self.processors.items()):
                if isinstance(proc_config, dict):
                    self.processors[name] = ProcessorConfig.from_dict(proc_config)

        # Compute derived paths
        self.temp_dir = self.project_dir / "temp"
        self.frames_dir = self.temp_dir / "frames"
        self.enhanced_dir = self.temp_dir / "enhanced"
        self.checkpoint_dir = self.project_dir / ".framewright"
        self.model_dir = Path.home() / ".framewright" / "models"

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []

        # Validate scale factor
        if self.scale_factor not in (2, 4):
            errors.append(f"scale_factor must be 2 or 4, got {self.scale_factor}")

        # Validate thresholds
        if not 0.0 <= self.min_ssim_threshold <= 1.0:
            errors.append(
                f"min_ssim_threshold must be 0.0-1.0, got {self.min_ssim_threshold}"
            )

        if self.min_psnr_threshold < 0:
            errors.append(f"min_psnr_threshold must be >= 0, got {self.min_psnr_threshold}")

        # Validate processing control
        if self.parallel_frames < 1:
            errors.append(f"parallel_frames must be >= 1, got {self.parallel_frames}")

        if self.max_retries < 0:
            errors.append(f"max_retries must be >= 0, got {self.max_retries}")

        if self.retry_delay < 0:
            errors.append(f"retry_delay must be >= 0, got {self.retry_delay}")

        if self.checkpoint_interval < 1:
            errors.append(f"checkpoint_interval must be >= 1, got {self.checkpoint_interval}")

        # Validate sub-configs
        errors.extend(self.hardware.validate())
        errors.extend(self.output.validate())

        for name, proc_config in self.processors.items():
            proc_errors = proc_config.validate()
            errors.extend([f"processors.{name}: {e}" for e in proc_errors])

        return errors

    def get_output_dir(self) -> Path:
        """Get the effective output directory.

        Returns:
            Output directory path.
        """
        if self.output_dir is not None:
            return self.output_dir
        return self.project_dir / "output"

    def get_output_path(self, input_path: Path) -> Path:
        """Get output path for a given input file.

        Args:
            input_path: Path to input video file.

        Returns:
            Path for output file.
        """
        template = self.output.output_template
        output_name = template.format(
            input_stem=input_path.stem,
            input_name=input_path.name,
            format=self.output.format,
        )
        return self.get_output_dir() / output_name

    def get_processor(self, name: str) -> ProcessorConfig:
        """Get processor configuration by name.

        Args:
            name: Processor name.

        Returns:
            ProcessorConfig for the processor.
        """
        return self.processors.get(name, ProcessorConfig())

    def set_processor(self, name: str, config: ProcessorConfig) -> None:
        """Set processor configuration.

        Args:
            name: Processor name.
            config: Processor configuration.
        """
        self.processors[name] = config

    def enable_processor(self, name: str, **kwargs: Any) -> None:
        """Enable a processor with optional settings.

        Args:
            name: Processor name.
            **kwargs: Additional processor settings.
        """
        if name not in self.processors:
            self.processors[name] = ProcessorConfig()

        self.processors[name].enabled = True
        for key, value in kwargs.items():
            if hasattr(self.processors[name], key):
                setattr(self.processors[name], key, value)
            else:
                self.processors[name].parameters[key] = value

    def disable_processor(self, name: str) -> None:
        """Disable a processor.

        Args:
            name: Processor name.
        """
        if name in self.processors:
            self.processors[name].enabled = False

    def create_directories(self) -> None:
        """Create all required directories."""
        directories = [
            self.project_dir,
            self.get_output_dir(),
            self.temp_dir,
            self.frames_dir,
            self.enhanced_dir,
            self.model_dir,
        ]

        if self.enable_checkpointing:
            directories.append(self.checkpoint_dir)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def cleanup_temp(self) -> None:
        """Remove temporary directories."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "project_dir": str(self.project_dir),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "scale_factor": self.scale_factor,
            "model_name": self.model_name,
            "hardware": self.hardware.to_dict(),
            "output": self.output.to_dict(),
            "processors": {
                name: proc.to_dict()
                for name, proc in self.processors.items()
            },
            "enable_checkpointing": self.enable_checkpointing,
            "checkpoint_interval": self.checkpoint_interval,
            "enable_validation": self.enable_validation,
            "min_ssim_threshold": self.min_ssim_threshold,
            "min_psnr_threshold": self.min_psnr_threshold,
            "parallel_frames": self.parallel_frames,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "continue_on_error": self.continue_on_error,
            "preset_name": self.preset_name,
            "config_version": self.config_version,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameWrightConfig":
        """Create configuration from dictionary."""
        # Convert nested configs
        if "hardware" in data and isinstance(data["hardware"], dict):
            data["hardware"] = HardwareConfig.from_dict(data["hardware"])

        if "output" in data and isinstance(data["output"], dict):
            data["output"] = OutputConfig.from_dict(data["output"])

        if "processors" in data:
            processors = {}
            for name, proc_data in data["processors"].items():
                if isinstance(proc_data, dict):
                    processors[name] = ProcessorConfig.from_dict(proc_data)
                else:
                    processors[name] = proc_data
            data["processors"] = processors

        # Convert paths
        if "project_dir" in data:
            data["project_dir"] = Path(data["project_dir"])

        if "output_dir" in data and data["output_dir"]:
            data["output_dir"] = Path(data["output_dir"])

        # Filter to valid keys
        valid_keys = {
            "project_dir", "output_dir", "scale_factor", "model_name",
            "hardware", "output", "processors",
            "enable_checkpointing", "checkpoint_interval",
            "enable_validation", "min_ssim_threshold", "min_psnr_threshold",
            "parallel_frames", "max_retries", "retry_delay", "continue_on_error",
            "preset_name", "config_version",
        }

        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# Configuration I/O Functions
# =============================================================================


def load_config(path: Union[str, Path]) -> FrameWrightConfig:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file.

    Returns:
        FrameWrightConfig instance.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ConfigError: If file format is unsupported or invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ConfigError(
                    "PyYAML is required for YAML configuration files. "
                    "Install with: pip install pyyaml"
                )

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        else:
            raise ConfigError(
                f"Unsupported configuration format: {suffix}. "
                "Use .yaml, .yml, or .json"
            )

        if data is None:
            data = {}

        return FrameWrightConfig.from_dict(data)

    except (json.JSONDecodeError, Exception) as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(f"Failed to parse configuration file: {e}")


def save_config(config: FrameWrightConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file.

    Args:
        config: Configuration to save.
        path: Output file path.

    Raises:
        ConfigError: If file format is unsupported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ConfigError(
                "PyYAML is required for YAML configuration files. "
                "Install with: pip install pyyaml"
            )

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    else:
        raise ConfigError(
            f"Unsupported configuration format: {suffix}. "
            "Use .yaml, .yml, or .json"
        )


def merge_configs(
    base: FrameWrightConfig,
    override: FrameWrightConfig,
) -> FrameWrightConfig:
    """Merge two configurations.

    Values from override take precedence when not None.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        New merged configuration.
    """
    base_dict = base.to_dict()
    override_dict = override.to_dict()

    def deep_merge(base_d: Dict, override_d: Dict) -> Dict:
        """Recursively merge dictionaries."""
        result = base_d.copy()

        for key, value in override_d.items():
            if value is None:
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    merged = deep_merge(base_dict, override_dict)
    return FrameWrightConfig.from_dict(merged)


def validate_config(config: FrameWrightConfig) -> List[str]:
    """Validate a configuration.

    Args:
        config: Configuration to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    return config.validate()


def create_config_from_preset(
    preset_name: str,
    project_dir: Path,
    **overrides: Any,
) -> FrameWrightConfig:
    """Create configuration from a preset.

    Args:
        preset_name: Name of preset to use.
        project_dir: Project directory.
        **overrides: Settings to override from preset.

    Returns:
        FrameWrightConfig instance.
    """
    try:
        from ..presets import PresetRegistry

        preset = PresetRegistry.get_for_hardware(preset_name)
        config_dict = preset.to_config_kwargs(project_dir)
        config_dict.update(overrides)

        return FrameWrightConfig.from_dict(config_dict)

    except ImportError:
        # Fall back to basic preset
        logger.warning("Preset system not available, using basic config")
        return FrameWrightConfig(project_dir=project_dir, **overrides)


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Main classes
    "FrameWrightConfig",
    "HardwareConfig",
    "OutputConfig",
    "ProcessorConfig",
    "BaseConfig",
    # Enums
    "ScaleFactor",
    "OutputFormat",
    "VideoCodec",
    "AudioCodec",
    "PixelFormat",
    "EncodingPreset",
    "GPULoadStrategy",
    "TemporalMethod",
    # Exceptions
    "ConfigError",
    "ValidationError",
    # Functions
    "load_config",
    "save_config",
    "merge_configs",
    "validate_config",
    "create_config_from_preset",
]
