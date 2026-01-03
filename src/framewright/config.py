"""Configuration module for FrameWright video restoration pipeline."""
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Dict, Any, ClassVar, List


# Configuration presets for common use cases
PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "scale_factor": 2,
        "model_name": "realesrgan-x2plus",
        "crf": 23,
        "preset": "fast",
        "parallel_frames": 4,
        "enable_checkpointing": False,
        "enable_validation": False,
    },
    "quality": {
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 18,
        "preset": "slow",
        "parallel_frames": 2,
        "enable_checkpointing": True,
        "enable_validation": True,
    },
    "archive": {
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 15,
        "preset": "veryslow",
        "parallel_frames": 1,
        "enable_checkpointing": True,
        "enable_validation": True,
        "min_ssim_threshold": 0.9,
        "min_psnr_threshold": 30.0,
    },
    "anime": {
        "scale_factor": 4,
        "model_name": "realesr-animevideov3",
        "crf": 18,
        "preset": "medium",
        "parallel_frames": 2,
        "enable_checkpointing": True,
    },
    "film_restoration": {
        "scale_factor": 4,
        "model_name": "realesrgan-x4plus",
        "crf": 16,
        "preset": "slow",
        "parallel_frames": 2,
        "enable_checkpointing": True,
        "enable_validation": True,
        "enable_auto_enhance": True,
        "auto_defect_repair": True,
        "auto_face_restore": True,
        "scratch_sensitivity": 0.7,
        "grain_reduction": 0.4,
    },
}


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

        # Multi-GPU distribution options
        enable_multi_gpu: Enable multi-GPU frame distribution for faster processing
        gpu_ids: Specific GPU IDs to use (None = auto-detect all available GPUs)
        gpu_load_balance_strategy: Load balancing strategy (round_robin, least_loaded, vram_aware, weighted)
        workers_per_gpu: Number of worker threads per GPU for parallel processing
        enable_work_stealing: Allow idle workers to steal work from busy workers

        # Directory configuration
        model_dir: Directory for storing model files (default: ~/.framewright/models)
        _output_dir_override: Override for output directory (if None, uses project_dir/output/)
        _frames_dir_override: Override for frames directory (if None, uses project_dir/frames/)
        _enhanced_dir_override: Override for enhanced directory (if None, uses project_dir/enhanced/)

        # Colorization options
        enable_colorization: Enable automatic colorization of black-and-white footage
        colorization_model: Colorization model to use ('ddcolor' or 'deoldify')
        colorization_strength: Strength of colorization effect (0.0-1.0)

        # Watermark removal options
        enable_watermark_removal: Enable watermark removal processing
        watermark_mask_path: Path to a mask image defining watermark region
        watermark_auto_detect: Automatically detect watermark location

        # Burnt-in subtitle removal options
        enable_subtitle_removal: Enable burnt-in subtitle detection and removal
        subtitle_region: Region to scan for subtitles (bottom_third, bottom_quarter, top_quarter, full_frame)
        subtitle_ocr_engine: OCR engine for detection (auto, easyocr, tesseract, paddleocr)
        subtitle_languages: List of language codes for OCR detection
    """

    project_dir: Path
    output_dir: Optional[Path] = None  # Explicit output directory (defaults to project_dir)
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
    continue_on_error: bool = True  # Continue processing even if some frames fail

    # GPU selection and multi-GPU distribution options
    require_gpu: bool = True  # Require GPU for processing - blocks CPU fallback to prevent runaway CPU usage
    gpu_id: Optional[int] = None  # Select specific GPU by index (--gpu N), None = auto-select
    enable_multi_gpu: bool = False  # Enable multi-GPU frame distribution (--multi-gpu)
    gpu_ids: Optional[List[int]] = None  # Specific GPU IDs to use (None = auto-detect all)
    gpu_load_balance_strategy: str = "vram_aware"  # round_robin, least_loaded, vram_aware, weighted
    workers_per_gpu: int = 2  # Worker threads per GPU for parallel processing
    enable_work_stealing: bool = True  # Allow idle workers to steal work from busy ones

    # RIFE frame interpolation options (optional)
    enable_interpolation: bool = False  # Must explicitly enable RIFE
    target_fps: Optional[float] = None  # Target frame rate (None = auto from source)
    rife_model: str = "rife-v4.6"  # RIFE model version
    rife_gpu_id: int = 0  # GPU for RIFE processing

    # Frame deduplication options (for old films with artificial FPS padding)
    enable_deduplication: bool = False  # Detect and remove duplicate frames
    deduplication_threshold: float = 0.98  # Similarity threshold (0.98 = very similar)
    expected_source_fps: Optional[float] = None  # Hint for original FPS (e.g., 18 for 1909 film)

    # Auto-enhancement options (fully automated processing)
    enable_auto_enhance: bool = False  # Enable automatic enhancement pipeline
    auto_detect_content: bool = True  # Auto-detect content type (faces, animation, etc.)
    auto_defect_repair: bool = True  # Auto-detect and repair defects
    auto_face_restore: bool = True  # Auto face restoration when faces detected
    scratch_sensitivity: float = 0.5  # Sensitivity for scratch detection (0-1)
    dust_sensitivity: float = 0.5  # Sensitivity for dust detection (0-1)
    grain_reduction: float = 0.3  # Film grain reduction strength (0-1)

    # Model download location
    model_download_dir: Optional[Path] = None  # Custom model download location

    # Directory configuration (new flexible paths)
    model_dir: Path = field(default_factory=lambda: Path.home() / ".framewright" / "models")
    _output_dir_override: Optional[Path] = None  # Internal: explicit output directory
    _frames_dir_override: Optional[Path] = None  # Internal: explicit frames directory
    _enhanced_dir_override: Optional[Path] = None  # Internal: explicit enhanced directory

    # Colorization options
    enable_colorization: bool = False
    colorization_model: str = "ddcolor"  # "ddcolor" or "deoldify"
    colorization_strength: float = 1.0  # Strength of colorization (0.0-1.0)

    # Watermark removal options
    enable_watermark_removal: bool = False
    watermark_mask_path: Optional[Path] = None  # Path to watermark mask image
    watermark_auto_detect: bool = True  # Auto-detect watermark location

    # Burnt-in subtitle removal options
    enable_subtitle_removal: bool = False
    subtitle_region: str = "bottom_third"  # bottom_third, bottom_quarter, top_quarter, full_frame
    subtitle_ocr_engine: str = "auto"  # auto, easyocr, tesseract, paddleocr
    subtitle_languages: List[str] = field(default_factory=lambda: ["en"])

    # Auto-generated paths
    temp_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    unique_frames_dir: Path = field(init=False)  # For deduplicated frames
    enhanced_dir: Path = field(init=False)
    interpolated_dir: Path = field(init=False)  # For RIFE output
    checkpoint_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived paths and validate configuration."""
        # Ensure project_dir is a Path object
        if not isinstance(self.project_dir, Path):
            self.project_dir = Path(self.project_dir)

        # Convert model_dir to Path if needed
        if not isinstance(self.model_dir, Path):
            self.model_dir = Path(self.model_dir)

        # Convert override paths to Path if provided as strings
        if self._output_dir_override is not None and not isinstance(self._output_dir_override, Path):
            self._output_dir_override = Path(self._output_dir_override)
        if self._frames_dir_override is not None and not isinstance(self._frames_dir_override, Path):
            self._frames_dir_override = Path(self._frames_dir_override)
        if self._enhanced_dir_override is not None and not isinstance(self._enhanced_dir_override, Path):
            self._enhanced_dir_override = Path(self._enhanced_dir_override)

        # Convert watermark_mask_path to Path if provided as string
        if self.watermark_mask_path is not None and not isinstance(self.watermark_mask_path, Path):
            self.watermark_mask_path = Path(self.watermark_mask_path)

        # Create derived directories (using overrides if provided)
        self.temp_dir = self.project_dir / "temp"
        self.frames_dir = self._frames_dir_override if self._frames_dir_override is not None else self.temp_dir / "frames"
        self.unique_frames_dir = self.temp_dir / "unique_frames"  # Deduplicated frames
        self.enhanced_dir = self._enhanced_dir_override if self._enhanced_dir_override is not None else self.temp_dir / "enhanced"
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

        # Validate deduplication options
        if not 0.0 <= self.deduplication_threshold <= 1.0:
            raise ValueError("deduplication_threshold must be between 0.0 and 1.0")
        if self.expected_source_fps is not None and self.expected_source_fps <= 0:
            raise ValueError("expected_source_fps must be positive")

        # Validate GPU options
        if self.gpu_id is not None:
            if not isinstance(self.gpu_id, int) or self.gpu_id < 0:
                raise ValueError(f"Invalid gpu_id: {self.gpu_id}. Must be non-negative integer or None.")

        # Validate multi-GPU options
        valid_strategies = ["round_robin", "least_loaded", "vram_aware", "weighted"]
        if self.gpu_load_balance_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid GPU load balance strategy '{self.gpu_load_balance_strategy}'. "
                f"Valid strategies: {valid_strategies}"
            )

        if self.workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be at least 1")

        if self.gpu_ids is not None:
            if not isinstance(self.gpu_ids, list):
                raise ValueError("gpu_ids must be a list of integers or None")
            for gid in self.gpu_ids:
                if not isinstance(gid, int) or gid < 0:
                    raise ValueError(f"Invalid GPU ID: {gid}. Must be non-negative integer.")

        # Validate auto-enhancement options
        if not 0.0 <= self.scratch_sensitivity <= 1.0:
            raise ValueError("scratch_sensitivity must be between 0.0 and 1.0")
        if not 0.0 <= self.dust_sensitivity <= 1.0:
            raise ValueError("dust_sensitivity must be between 0.0 and 1.0")
        if not 0.0 <= self.grain_reduction <= 1.0:
            raise ValueError("grain_reduction must be between 0.0 and 1.0")

        # Validate colorization options
        valid_colorization_models = ["ddcolor", "deoldify"]
        if self.colorization_model not in valid_colorization_models:
            raise ValueError(
                f"Invalid colorization model '{self.colorization_model}'. "
                f"Valid models: {valid_colorization_models}"
            )
        if not 0.0 <= self.colorization_strength <= 1.0:
            raise ValueError("colorization_strength must be between 0.0 and 1.0")

        # Validate watermark options
        if self.watermark_mask_path is not None:
            if not self.watermark_mask_path.exists():
                raise ValueError(f"Watermark mask file not found: {self.watermark_mask_path}")
            # Validate file extension for common image formats
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            if self.watermark_mask_path.suffix.lower() not in valid_extensions:
                raise ValueError(
                    f"Invalid watermark mask format '{self.watermark_mask_path.suffix}'. "
                    f"Supported formats: {valid_extensions}"
                )

        # Validate subtitle removal options
        valid_subtitle_regions = ["bottom_third", "bottom_quarter", "top_quarter", "full_frame"]
        if self.subtitle_region not in valid_subtitle_regions:
            raise ValueError(
                f"Invalid subtitle region '{self.subtitle_region}'. "
                f"Valid regions: {valid_subtitle_regions}"
            )

        valid_ocr_engines = ["auto", "easyocr", "tesseract", "paddleocr"]
        if self.subtitle_ocr_engine not in valid_ocr_engines:
            raise ValueError(
                f"Invalid OCR engine '{self.subtitle_ocr_engine}'. "
                f"Valid engines: {valid_ocr_engines}"
            )

        # Convert output_dir to Path if provided
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Convert model_download_dir to Path if provided
        if self.model_download_dir is not None and not isinstance(self.model_download_dir, Path):
            self.model_download_dir = Path(self.model_download_dir)

    def get_output_dir(self) -> Path:
        """Get the effective output directory.

        Returns:
            output_dir if set, otherwise _output_dir_override if set,
            otherwise project_dir/output/
        """
        if self.output_dir is not None:
            return self.output_dir
        if self._output_dir_override is not None:
            return self._output_dir_override
        return self.project_dir / "output"

    def get_frames_dir(self) -> Path:
        """Get the effective frames directory.

        Returns:
            _frames_dir_override if set, otherwise the computed frames_dir
        """
        if self._frames_dir_override is not None:
            return self._frames_dir_override
        return self.frames_dir

    def get_enhanced_dir(self) -> Path:
        """Get the effective enhanced frames directory.

        Returns:
            _enhanced_dir_override if set, otherwise the computed enhanced_dir
        """
        if self._enhanced_dir_override is not None:
            return self._enhanced_dir_override
        return self.enhanced_dir

    def get_model_path(self, model_name: str) -> Path:
        """Get the full path for a model file.

        Args:
            model_name: Name of the model (with or without extension)

        Returns:
            Full path to the model file in the model directory
        """
        return self.model_dir / model_name

    def ensure_directories(self) -> None:
        """Create all required directories for the pipeline.

        Creates:
            - project_dir
            - model_dir
            - output_dir (from get_output_dir())
            - frames_dir (from get_frames_dir())
            - enhanced_dir (from get_enhanced_dir())
            - temp_dir
            - checkpoint_dir (if checkpointing enabled)
            - interpolated_dir (if interpolation enabled)
        """
        directories: List[Path] = [
            self.project_dir,
            self.model_dir,
            self.get_output_dir(),
            self.get_frames_dir(),
            self.get_enhanced_dir(),
            self.temp_dir,
        ]

        if self.enable_checkpointing:
            directories.append(self.checkpoint_dir)

        if self.enable_interpolation:
            directories.append(self.interpolated_dir)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> List[str]:
        """Validate all configured paths.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors: List[str] = []

        # Check if project_dir parent exists or can be created
        if not self.project_dir.parent.exists():
            try:
                self.project_dir.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                errors.append(f"Cannot create project directory parent: {e}")

        # Check model_dir is writable (or its parent)
        model_dir_to_check = self.model_dir if self.model_dir.exists() else self.model_dir.parent
        if model_dir_to_check.exists() and not os.access(model_dir_to_check, os.W_OK):
            errors.append(f"Model directory is not writable: {self.model_dir}")

        # Check watermark mask if specified
        if self.watermark_mask_path is not None and not self.watermark_mask_path.exists():
            errors.append(f"Watermark mask file not found: {self.watermark_mask_path}")

        # Validate override directories if specified
        for name, path in [
            ("output_dir_override", self._output_dir_override),
            ("frames_dir_override", self._frames_dir_override),
            ("enhanced_dir_override", self._enhanced_dir_override),
        ]:
            if path is not None:
                parent = path.parent
                if parent.exists() and not os.access(parent, os.W_OK):
                    errors.append(f"{name} parent directory is not writable: {parent}")

        return errors

    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """Get the full output file path.

        Args:
            filename: Optional filename (defaults to 'restored_video.{format}')

        Returns:
            Full path to output file
        """
        if filename is None:
            filename = f"restored_video.{self.output_format}"
        return self.get_output_dir() / filename

    def create_directories(self) -> None:
        """Create all required directories for the pipeline."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_deduplication:
            self.unique_frames_dir.mkdir(parents=True, exist_ok=True)
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
            'project_dir', 'output_dir', 'scale_factor', 'model_name', 'crf', 'preset',
            'output_format', 'enable_checkpointing', 'checkpoint_interval',
            'enable_validation', 'min_ssim_threshold', 'min_psnr_threshold',
            'enable_disk_validation', 'disk_safety_margin', 'enable_vram_monitoring',
            'tile_size', 'max_retries', 'retry_delay', 'parallel_frames',
            'continue_on_error', 'require_gpu', 'gpu_id', 'enable_multi_gpu', 'gpu_ids', 'gpu_load_balance_strategy',
            'workers_per_gpu', 'enable_work_stealing', 'enable_interpolation', 'target_fps',
            'rife_model', 'rife_gpu_id', 'enable_deduplication', 'deduplication_threshold',
            'expected_source_fps', 'enable_auto_enhance', 'auto_detect_content',
            'auto_defect_repair', 'auto_face_restore', 'scratch_sensitivity',
            'dust_sensitivity', 'grain_reduction', 'model_download_dir', 'model_dir',
            'enable_colorization', 'colorization_model', 'colorization_strength',
            'enable_watermark_removal', 'watermark_mask_path', 'watermark_auto_detect',
            'enable_subtitle_removal', 'subtitle_region', 'subtitle_ocr_engine', 'subtitle_languages'
        ]:
            val = getattr(self, key)
            if isinstance(val, Path):
                data[key] = str(val)
            elif val is None:
                data[key] = None
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

        # Convert output_dir to Path if provided
        if 'output_dir' in data and isinstance(data['output_dir'], str):
            data['output_dir'] = Path(data['output_dir'])

        # Convert model_download_dir to Path if provided
        if 'model_download_dir' in data and isinstance(data['model_download_dir'], str):
            data['model_download_dir'] = Path(data['model_download_dir'])

        # Convert model_dir to Path if provided
        if 'model_dir' in data and isinstance(data['model_dir'], str):
            data['model_dir'] = Path(data['model_dir'])

        # Convert watermark_mask_path to Path if provided
        if 'watermark_mask_path' in data and isinstance(data['watermark_mask_path'], str):
            data['watermark_mask_path'] = Path(data['watermark_mask_path'])

        # Filter to only valid init parameters
        valid_keys = {
            'project_dir', 'output_dir', 'scale_factor', 'model_name', 'crf', 'preset',
            'output_format', 'enable_checkpointing', 'checkpoint_interval',
            'enable_validation', 'min_ssim_threshold', 'min_psnr_threshold',
            'enable_disk_validation', 'disk_safety_margin', 'enable_vram_monitoring',
            'tile_size', 'max_retries', 'retry_delay', 'parallel_frames',
            'continue_on_error', 'require_gpu', 'gpu_id', 'enable_multi_gpu', 'gpu_ids', 'gpu_load_balance_strategy',
            'workers_per_gpu', 'enable_work_stealing', 'enable_interpolation', 'target_fps',
            'rife_model', 'rife_gpu_id', 'enable_deduplication', 'deduplication_threshold',
            'expected_source_fps', 'enable_auto_enhance', 'auto_detect_content',
            'auto_defect_repair', 'auto_face_restore', 'scratch_sensitivity',
            'dust_sensitivity', 'grain_reduction', 'model_download_dir', 'model_dir',
            'enable_colorization', 'colorization_model', 'colorization_strength',
            'enable_watermark_removal', 'watermark_mask_path', 'watermark_auto_detect',
            'enable_subtitle_removal', 'subtitle_region', 'subtitle_ocr_engine', 'subtitle_languages',
            '_output_dir_override', '_frames_dir_override', '_enhanced_dir_override'
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

    @classmethod
    def from_preset(cls, preset_name: str, project_dir: Path, **overrides) -> "Config":
        """Create a configuration from a preset name.

        Args:
            preset_name: Name of the preset ('fast', 'quality', 'archive', 'anime', 'film_restoration')
            project_dir: Root directory for processing files
            **overrides: Additional parameters to override preset defaults

        Returns:
            Config instance with preset settings

        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in PRESETS:
            available = ', '.join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

        preset_config = PRESETS[preset_name].copy()
        preset_config['project_dir'] = project_dir
        preset_config.update(overrides)

        return cls(**preset_config)

    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """Get list of available presets with descriptions.

        Returns:
            Dictionary mapping preset names to descriptions
        """
        descriptions = {
            "fast": "Quick processing with 2x upscale, minimal quality checks",
            "quality": "High quality 4x upscale with validation enabled",
            "archive": "Archival quality with maximum quality settings",
            "anime": "Optimized for anime/animation content",
            "film_restoration": "Full restoration for old films with defect repair",
        }
        return descriptions

    def save_preset(self, filepath: Path, name: str = "custom") -> None:
        """Save current configuration as a preset file.

        Args:
            filepath: Path to save the preset JSON file
            name: Name to identify this preset
        """
        preset_data = {
            "name": name,
            "version": "1.3.1",
            "config": self.to_dict(),
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)

    @classmethod
    def load_preset_file(cls, filepath: Path) -> "Config":
        """Load a configuration from a preset file.

        Args:
            filepath: Path to the preset JSON file

        Returns:
            Config instance from the preset file

        Raises:
            FileNotFoundError: If preset file doesn't exist
            ValueError: If preset file is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Preset file not found: {filepath}")

        with open(filepath, 'r') as f:
            preset_data = json.load(f)

        if 'config' not in preset_data:
            raise ValueError("Invalid preset file: missing 'config' key")

        return cls.from_dict(preset_data['config'])

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


@dataclass
class CloudConfig:
    """Configuration for cloud processing.

    Attributes:
        provider: Cloud GPU provider ('runpod', 'vastai').
        api_key: Provider API key (can also be set via environment variable).
        gpu_type: Preferred GPU type (e.g., 'RTX_4090', 'A100_80GB').
        storage_backend: Cloud storage backend ('s3', 'gcs', 'azure').
        storage_bucket: Storage bucket/container name.
        storage_credentials: Storage-specific credentials dict.
        max_runtime_minutes: Maximum job runtime before timeout.
        auto_cleanup: Automatically delete remote files after completion.
        use_serverless: Use serverless endpoints if available.
        endpoint_id: Provider-specific endpoint ID for serverless.
    """

    provider: str = "runpod"  # runpod, vastai
    api_key: Optional[str] = None
    gpu_type: str = "RTX_4090"
    storage_backend: str = "s3"  # s3, gcs, azure
    storage_bucket: Optional[str] = None
    storage_credentials: Dict[str, Any] = field(default_factory=dict)
    max_runtime_minutes: int = 120
    auto_cleanup: bool = True
    use_serverless: bool = True
    endpoint_id: Optional[str] = None

    # Valid providers and GPU types
    VALID_PROVIDERS: ClassVar[List[str]] = ["runpod", "vastai"]
    VALID_STORAGE_BACKENDS: ClassVar[List[str]] = ["s3", "gcs", "azure"]
    VALID_GPU_TYPES: ClassVar[List[str]] = [
        "RTX_4090",
        "RTX_3090",
        "RTX_4080",
        "RTX_3080",
        "A100_80GB",
        "A100_40GB",
        "A6000",
        "H100",
        "L40",
    ]

    def __post_init__(self) -> None:
        """Validate cloud configuration."""
        if self.provider not in self.VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{self.provider}'. "
                f"Valid providers: {self.VALID_PROVIDERS}"
            )

        if self.storage_backend not in self.VALID_STORAGE_BACKENDS:
            raise ValueError(
                f"Invalid storage backend '{self.storage_backend}'. "
                f"Valid backends: {self.VALID_STORAGE_BACKENDS}"
            )

        if self.gpu_type not in self.VALID_GPU_TYPES:
            raise ValueError(
                f"Invalid GPU type '{self.gpu_type}'. "
                f"Valid types: {self.VALID_GPU_TYPES}"
            )

        if self.max_runtime_minutes < 1:
            raise ValueError("max_runtime_minutes must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "gpu_type": self.gpu_type,
            "storage_backend": self.storage_backend,
            "storage_bucket": self.storage_bucket,
            "storage_credentials": self.storage_credentials,
            "max_runtime_minutes": self.max_runtime_minutes,
            "auto_cleanup": self.auto_cleanup,
            "use_serverless": self.use_serverless,
            "endpoint_id": self.endpoint_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudConfig":
        """Create from dictionary."""
        valid_keys = {
            "provider",
            "api_key",
            "gpu_type",
            "storage_backend",
            "storage_bucket",
            "storage_credentials",
            "max_runtime_minutes",
            "auto_cleanup",
            "use_serverless",
            "endpoint_id",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load_from_file(cls, filepath: Path) -> "CloudConfig":
        """Load cloud configuration from JSON file.

        Args:
            filepath: Path to configuration file.

        Returns:
            CloudConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file contains invalid configuration.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Cloud config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_to_file(self, filepath: Path) -> None:
        """Save cloud configuration to JSON file.

        Args:
            filepath: Path to save configuration.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Don't save API key to file for security
        data = self.to_dict()
        if data.get("api_key"):
            data["api_key"] = "***REDACTED***"
        if data.get("storage_credentials"):
            data["storage_credentials"] = {"***": "REDACTED"}

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get default path for cloud configuration file."""
        return Path.home() / ".framewright" / "cloud_config.json"

    def get_provider_instance(self):
        """Get configured cloud provider instance.

        Returns:
            CloudProvider instance configured with this config.

        Raises:
            ImportError: If cloud dependencies not installed.
        """
        try:
            from .cloud import RunPodProvider, VastAIProvider

            if self.provider == "runpod":
                return RunPodProvider(
                    api_key=self.api_key,
                    endpoint_id=self.endpoint_id,
                    use_serverless=self.use_serverless,
                )
            elif self.provider == "vastai":
                return VastAIProvider(
                    api_key=self.api_key,
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except ImportError:
            raise ImportError(
                "Cloud dependencies not installed. "
                "Install with: pip install framewright[cloud]"
            )

    def get_storage_instance(self):
        """Get configured storage provider instance.

        Returns:
            CloudStorageProvider instance configured with this config.

        Raises:
            ImportError: If cloud dependencies not installed.
        """
        try:
            from .cloud.storage import get_storage_provider

            return get_storage_provider(
                self.storage_backend,
                bucket=self.storage_bucket,
                credentials=self.storage_credentials,
            )

        except ImportError:
            raise ImportError(
                "Cloud dependencies not installed. "
                "Install with: pip install framewright[cloud]"
            )
