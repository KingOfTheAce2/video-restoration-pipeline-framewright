"""Configuration module for FrameWright video restoration pipeline."""
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Dict, Any


@dataclass
class Config:
    """Configuration for video restoration pipeline.

    Attributes:
        project_dir: Root directory for all processing files
        scale_factor: Upscaling factor (2x or 4x)
        model_name: Real-ESRGAN model to use
        crf: Constant Rate Factor for x265 encoding (0-51, lower is better quality)
        preset: x265 encoding preset (ultrafast to veryslow)
        output_format: Output video container format
        temp_dir: Temporary directory for intermediate files (auto-created)
        frames_dir: Directory for extracted frames (auto-created)
        enhanced_dir: Directory for enhanced frames (auto-created)

        # Robustness options
        enable_checkpointing: Enable checkpoint/resume functionality
        checkpoint_interval: Save checkpoint every N frames
        enable_validation: Enable quality validation
        min_ssim_threshold: Minimum SSIM for quality validation
        min_psnr_threshold: Minimum PSNR for quality validation
        enable_disk_validation: Pre-check disk space
        disk_safety_margin: Extra disk space buffer (1.2 = 20% extra)
        enable_vram_monitoring: Monitor GPU VRAM usage
        tile_size: Tile size for Real-ESRGAN (0 = auto, None = no tiling)
        max_retries: Maximum retry attempts for transient errors
        retry_delay: Initial delay between retries in seconds
        parallel_frames: Number of frames to process in parallel (1 = sequential)
        continue_on_error: Continue processing even if some frames fail
    """

    project_dir: Path
    scale_factor: Literal[2, 4] = 4
    model_name: str = "realesrgan-x4plus"
    crf: int = 18
    preset: str = "medium"
    output_format: str = "mkv"

    # Robustness options
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    enable_validation: bool = True
    min_ssim_threshold: float = 0.85
    min_psnr_threshold: float = 25.0
    enable_disk_validation: bool = True
    disk_safety_margin: float = 1.2
    enable_vram_monitoring: bool = True
    tile_size: Optional[int] = 0  # 0 = auto, None = no tiling
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_frames: int = 1
    continue_on_error: bool = False

    # RIFE frame interpolation options (optional)
    enable_interpolation: bool = False  # Must explicitly enable RIFE
    target_fps: Optional[float] = None  # Target frame rate (None = auto from source)
    rife_model: str = "rife-v4.6"  # RIFE model version
    rife_gpu_id: int = 0  # GPU for RIFE processing

    # Auto-enhancement options (fully automated processing)
    enable_auto_enhance: bool = False  # Enable automatic enhancement pipeline
    auto_detect_content: bool = True  # Auto-detect content type (faces, animation, etc.)
    auto_defect_repair: bool = True  # Auto-detect and repair defects
    auto_face_restore: bool = True  # Auto face restoration when faces detected
    scratch_sensitivity: float = 0.5  # Sensitivity for scratch detection (0-1)
    dust_sensitivity: float = 0.5  # Sensitivity for dust detection (0-1)
    grain_reduction: float = 0.3  # Film grain reduction strength (0-1)

    # Auto-generated paths
    temp_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    enhanced_dir: Path = field(init=False)
    interpolated_dir: Path = field(init=False)  # For RIFE output
    checkpoint_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived paths and validate configuration."""
        # Ensure project_dir is a Path object
        if not isinstance(self.project_dir, Path):
            self.project_dir = Path(self.project_dir)

        # Create derived directories
        self.temp_dir = self.project_dir / "temp"
        self.frames_dir = self.temp_dir / "frames"
        self.enhanced_dir = self.temp_dir / "enhanced"
        self.interpolated_dir = self.temp_dir / "interpolated"
        self.checkpoint_dir = self.project_dir / ".framewright"

        # Validate scale factor
        if self.scale_factor not in (2, 4):
            raise ValueError("scale_factor must be 2 or 4")

        # Validate CRF range
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")

        # Validate model name based on scale factor
        valid_models = {
            2: ["realesrgan-x2plus"],
            4: ["realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3"]
        }

        if self.model_name not in valid_models.get(self.scale_factor, []):
            raise ValueError(
                f"Invalid model '{self.model_name}' for scale factor {self.scale_factor}. "
                f"Valid models: {valid_models.get(self.scale_factor, [])}"
            )

        # Validate threshold ranges
        if not 0.0 <= self.min_ssim_threshold <= 1.0:
            raise ValueError("min_ssim_threshold must be between 0.0 and 1.0")

        if self.min_psnr_threshold < 0:
            raise ValueError("min_psnr_threshold must be non-negative")

        # Validate retry settings
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        # Validate parallel frames
        if self.parallel_frames < 1:
            raise ValueError("parallel_frames must be at least 1")

        # Validate tile size
        if self.tile_size is not None and self.tile_size < 0:
            raise ValueError("tile_size must be non-negative or None")

        # Validate RIFE options
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

        valid_rife_models = ['rife-v2.3', 'rife-v4.0', 'rife-v4.6']
        if self.rife_model not in valid_rife_models:
            raise ValueError(
                f"Invalid RIFE model '{self.rife_model}'. "
                f"Valid models: {valid_rife_models}"
            )

        # Validate auto-enhancement options
        if not 0.0 <= self.scratch_sensitivity <= 1.0:
            raise ValueError("scratch_sensitivity must be between 0.0 and 1.0")
        if not 0.0 <= self.dust_sensitivity <= 1.0:
            raise ValueError("dust_sensitivity must be between 0.0 and 1.0")
        if not 0.0 <= self.grain_reduction <= 1.0:
            raise ValueError("grain_reduction must be between 0.0 and 1.0")

    def create_directories(self) -> None:
        """Create all required directories for the pipeline."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_interpolation:
            self.interpolated_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_checkpointing:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_temp(self) -> None:
        """Remove temporary directories and their contents."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Useful for serialization and checkpointing.
        """
        data = {}
        for key in [
            'project_dir', 'scale_factor', 'model_name', 'crf', 'preset',
            'output_format', 'enable_checkpointing', 'checkpoint_interval',
            'enable_validation', 'min_ssim_threshold', 'min_psnr_threshold',
            'enable_disk_validation', 'disk_safety_margin', 'enable_vram_monitoring',
            'tile_size', 'max_retries', 'retry_delay', 'parallel_frames',
            'continue_on_error', 'enable_interpolation', 'target_fps',
            'rife_model', 'rife_gpu_id', 'enable_auto_enhance', 'auto_detect_content',
            'auto_defect_repair', 'auto_face_restore', 'scratch_sensitivity',
            'dust_sensitivity', 'grain_reduction'
        ]:
            val = getattr(self, key)
            if isinstance(val, Path):
                data[key] = str(val)
            else:
                data[key] = val
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        # Convert project_dir to Path
        if 'project_dir' in data and isinstance(data['project_dir'], str):
            data['project_dir'] = Path(data['project_dir'])

        # Filter to only valid init parameters
        valid_keys = {
            'project_dir', 'scale_factor', 'model_name', 'crf', 'preset',
            'output_format', 'enable_checkpointing', 'checkpoint_interval',
            'enable_validation', 'min_ssim_threshold', 'min_psnr_threshold',
            'enable_disk_validation', 'disk_safety_margin', 'enable_vram_monitoring',
            'tile_size', 'max_retries', 'retry_delay', 'parallel_frames',
            'continue_on_error', 'enable_interpolation', 'target_fps',
            'rife_model', 'rife_gpu_id', 'enable_auto_enhance', 'auto_detect_content',
            'auto_defect_repair', 'auto_face_restore', 'scratch_sensitivity',
            'dust_sensitivity', 'grain_reduction'
        }

        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def get_hash(self) -> str:
        """Generate hash of configuration for change detection.

        Returns:
            SHA256 hash (first 16 characters) of configuration
        """
        # Only hash settings that affect output
        hash_data = {
            'scale_factor': self.scale_factor,
            'model_name': self.model_name,
            'crf': self.crf,
            'preset': self.preset,
            'tile_size': self.tile_size,
        }
        config_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_tile_size_for_resolution(
        self,
        width: int,
        height: int,
        available_vram_mb: Optional[int] = None
    ) -> int:
        """Calculate appropriate tile size for given resolution.

        Args:
            width: Frame width
            height: Frame height
            available_vram_mb: Available VRAM (auto-detected if None)

        Returns:
            Tile size (0 if no tiling needed)
        """
        if self.tile_size is None:
            return 0  # No tiling

        if self.tile_size > 0:
            return self.tile_size  # Use configured size

        # Auto-calculate tile size
        from .utils.gpu import calculate_optimal_tile_size

        return calculate_optimal_tile_size(
            frame_resolution=(width, height),
            scale_factor=self.scale_factor,
            available_vram_mb=available_vram_mb,
            model_name=self.model_name,
        )


@dataclass
class RestoreOptions:
    """Additional options for video restoration.

    Separate from Config to allow per-restoration customization.
    """
    source: str  # URL or file path
    output_path: Optional[Path] = None
    cleanup: bool = True
    resume: bool = True  # Resume from checkpoint if available
    validate_output: bool = True
    skip_audio: bool = False
    dry_run: bool = False  # Validate only, don't process

    # Preview/approval options
    preview_before_reassembly: bool = False  # Pause for user to inspect frames
    preview_frame_count: int = 5  # Number of sample frames to show

    # RIFE interpolation overrides (can override Config settings per-run)
    enable_rife: Optional[bool] = None  # None = use Config setting
    target_fps: Optional[float] = None  # None = use Config setting or auto-detect

    def __post_init__(self) -> None:
        """Validate options."""
        if self.output_path is not None and not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

        if self.preview_frame_count < 1:
            raise ValueError("preview_frame_count must be at least 1")
