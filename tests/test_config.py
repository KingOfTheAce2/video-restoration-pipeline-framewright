"""Tests for the Config dataclass validation."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from framewright.config import Config, PRESETS, CloudConfig, RestoreOptions


class TestConfigCreation:
    """Test Config dataclass creation with defaults."""

    def test_valid_config_with_defaults(self, tmp_path):
        """Test creating a config with minimal required params uses defaults."""
        config = Config(project_dir=tmp_path)

        # Check defaults
        assert config.scale_factor == 4
        assert config.model_name == "realesrgan-x4plus"
        assert config.crf == 18
        assert config.preset == "medium"
        assert config.output_format == "mkv"
        assert config.enable_checkpointing is True
        assert config.checkpoint_interval == 100
        assert config.enable_validation is True
        assert config.parallel_frames == 1

    def test_valid_config_with_scale_2(self, tmp_path):
        """Test creating a config with 2x scale factor."""
        config = Config(
            project_dir=tmp_path,
            scale_factor=2,
            model_name="realesrgan-x2plus"
        )

        assert config.scale_factor == 2
        assert config.model_name == "realesrgan-x2plus"

    def test_valid_config_with_all_options(self, tmp_path):
        """Test creating a config with many explicit options."""
        config = Config(
            project_dir=tmp_path,
            scale_factor=4,
            model_name="realesrgan-x4plus",
            crf=15,
            preset="slow",
            output_format="mp4",
            enable_checkpointing=True,
            checkpoint_interval=50,
            enable_validation=True,
            min_ssim_threshold=0.9,
            min_psnr_threshold=30.0,
            enable_disk_validation=True,
            max_retries=5,
            retry_delay=2.0,
            parallel_frames=4,
            continue_on_error=False,
        )

        assert config.crf == 15
        assert config.preset == "slow"
        assert config.output_format == "mp4"
        assert config.checkpoint_interval == 50
        assert config.min_ssim_threshold == 0.9
        assert config.min_psnr_threshold == 30.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.parallel_frames == 4
        assert config.continue_on_error is False

    def test_derived_directories_created(self, tmp_path):
        """Test that derived directories are computed correctly."""
        config = Config(project_dir=tmp_path)

        assert config.temp_dir == tmp_path / "temp"
        assert config.frames_dir == tmp_path / "temp" / "frames"
        assert config.enhanced_dir == tmp_path / "temp" / "enhanced"
        assert config.checkpoint_dir == tmp_path / ".framewright"

    def test_project_dir_string_conversion(self, tmp_path):
        """Test that string project_dir is converted to Path."""
        config = Config(project_dir=str(tmp_path))

        assert isinstance(config.project_dir, Path)
        assert config.project_dir == tmp_path


class TestConfigValidation:
    """Test Config validation errors."""

    def test_invalid_scale_factor_raises_error(self, tmp_path):
        """Test that invalid scale_factor raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, scale_factor=3)

        assert "scale_factor must be 2 or 4" in str(exc_info.value)

    def test_invalid_crf_too_low_raises_error(self, tmp_path):
        """Test that CRF below 0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, crf=-1)

        assert "crf must be between 0 and 51" in str(exc_info.value)

    def test_invalid_crf_too_high_raises_error(self, tmp_path):
        """Test that CRF above 51 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, crf=52)

        assert "crf must be between 0 and 51" in str(exc_info.value)

    def test_invalid_model_for_scale_factor_raises_error(self, tmp_path):
        """Test that mismatched model and scale_factor raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(
                project_dir=tmp_path,
                scale_factor=2,
                model_name="realesrgan-x4plus"
            )

        assert "Invalid model" in str(exc_info.value)
        assert "realesrgan-x4plus" in str(exc_info.value)

    def test_invalid_ssim_threshold_too_high_raises_error(self, tmp_path):
        """Test that SSIM threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, min_ssim_threshold=1.5)

        assert "min_ssim_threshold must be between 0.0 and 1.0" in str(exc_info.value)

    def test_invalid_ssim_threshold_negative_raises_error(self, tmp_path):
        """Test that negative SSIM threshold raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, min_ssim_threshold=-0.1)

        assert "min_ssim_threshold must be between 0.0 and 1.0" in str(exc_info.value)

    def test_invalid_psnr_threshold_negative_raises_error(self, tmp_path):
        """Test that negative PSNR threshold raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, min_psnr_threshold=-5.0)

        assert "min_psnr_threshold must be non-negative" in str(exc_info.value)

    def test_invalid_max_retries_negative_raises_error(self, tmp_path):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, max_retries=-1)

        assert "max_retries must be non-negative" in str(exc_info.value)

    def test_invalid_retry_delay_negative_raises_error(self, tmp_path):
        """Test that negative retry_delay raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, retry_delay=-1.0)

        assert "retry_delay must be non-negative" in str(exc_info.value)

    def test_invalid_parallel_frames_zero_raises_error(self, tmp_path):
        """Test that parallel_frames < 1 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, parallel_frames=0)

        assert "parallel_frames must be at least 1" in str(exc_info.value)

    def test_invalid_tile_size_negative_raises_error(self, tmp_path):
        """Test that negative tile_size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, tile_size=-100)

        assert "tile_size must be non-negative or None" in str(exc_info.value)

    def test_invalid_rife_model_raises_error(self, tmp_path):
        """Test that invalid RIFE model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, rife_model="invalid-model")

        assert "Invalid RIFE model" in str(exc_info.value)

    def test_invalid_colorization_model_raises_error(self, tmp_path):
        """Test that invalid colorization model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, colorization_model="invalid")

        assert "Invalid colorization model" in str(exc_info.value)

    def test_invalid_colorization_strength_raises_error(self, tmp_path):
        """Test that colorization_strength out of range raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, colorization_strength=1.5)

        assert "colorization_strength must be between 0.0 and 1.0" in str(exc_info.value)

    def test_invalid_tap_model_raises_error(self, tmp_path):
        """Test that invalid TAP model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, tap_model="invalid")

        assert "Invalid TAP model" in str(exc_info.value)

    def test_invalid_sr_model_raises_error(self, tmp_path):
        """Test that invalid SR model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, sr_model="invalid")

        assert "Invalid SR model" in str(exc_info.value)

    def test_invalid_face_model_raises_error(self, tmp_path):
        """Test that invalid face model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, face_model="invalid")

        assert "Invalid face model" in str(exc_info.value)

    def test_invalid_gpu_id_negative_raises_error(self, tmp_path):
        """Test that negative GPU ID raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, gpu_id=-1)

        assert "Invalid gpu_id" in str(exc_info.value)

    def test_invalid_gpu_load_balance_strategy_raises_error(self, tmp_path):
        """Test that invalid GPU load balance strategy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(project_dir=tmp_path, gpu_load_balance_strategy="invalid")

        assert "Invalid GPU load balance strategy" in str(exc_info.value)


class TestConfigSerialization:
    """Test Config to_dict() and from_dict() roundtrip."""

    def test_to_dict_returns_dict(self, tmp_path):
        """Test that to_dict returns a dictionary."""
        config = Config(project_dir=tmp_path)
        data = config.to_dict()

        assert isinstance(data, dict)
        assert "project_dir" in data
        assert "scale_factor" in data
        assert "model_name" in data

    def test_to_dict_converts_paths_to_strings(self, tmp_path):
        """Test that to_dict converts Path objects to strings."""
        config = Config(project_dir=tmp_path)
        data = config.to_dict()

        assert isinstance(data["project_dir"], str)

    def test_from_dict_creates_config(self, tmp_path):
        """Test that from_dict creates a valid Config."""
        data = {
            "project_dir": str(tmp_path),
            "scale_factor": 4,
            "model_name": "realesrgan-x4plus",
            "crf": 18,
        }

        config = Config.from_dict(data)

        assert config.project_dir == tmp_path
        assert config.scale_factor == 4
        assert config.model_name == "realesrgan-x4plus"
        assert config.crf == 18

    def test_roundtrip_preserves_values(self, tmp_path):
        """Test that to_dict/from_dict roundtrip preserves all values."""
        original = Config(
            project_dir=tmp_path,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            crf=15,
            preset="slow",
            enable_checkpointing=True,
            checkpoint_interval=50,
            min_ssim_threshold=0.9,
            parallel_frames=4,
        )

        data = original.to_dict()
        restored = Config.from_dict(data)

        assert restored.project_dir == original.project_dir
        assert restored.scale_factor == original.scale_factor
        assert restored.model_name == original.model_name
        assert restored.crf == original.crf
        assert restored.preset == original.preset
        assert restored.enable_checkpointing == original.enable_checkpointing
        assert restored.checkpoint_interval == original.checkpoint_interval
        assert restored.min_ssim_threshold == original.min_ssim_threshold
        assert restored.parallel_frames == original.parallel_frames

    def test_from_dict_ignores_unknown_keys(self, tmp_path):
        """Test that from_dict ignores unknown keys."""
        data = {
            "project_dir": str(tmp_path),
            "unknown_key": "should be ignored",
            "another_unknown": 123,
        }

        config = Config.from_dict(data)

        assert config.project_dir == tmp_path
        assert not hasattr(config, "unknown_key")


class TestConfigPresets:
    """Test Config.from_preset() for each preset."""

    def test_fast_preset(self, tmp_path):
        """Test 'fast' preset configuration."""
        config = Config.from_preset("fast", tmp_path)

        assert config.scale_factor == 2
        assert config.model_name == "realesrgan-x2plus"
        assert config.crf == 23
        assert config.preset == "fast"
        assert config.parallel_frames == 4
        assert config.enable_checkpointing is False

    def test_quality_preset(self, tmp_path):
        """Test 'quality' preset configuration."""
        config = Config.from_preset("quality", tmp_path)

        assert config.scale_factor == 4
        assert config.model_name == "realesrgan-x4plus"
        assert config.crf == 18
        assert config.preset == "slow"
        assert config.enable_checkpointing is True
        assert config.enable_validation is True

    def test_archive_preset(self, tmp_path):
        """Test 'archive' preset configuration."""
        config = Config.from_preset("archive", tmp_path)

        assert config.scale_factor == 4
        assert config.model_name == "realesrgan-x4plus"
        assert config.enable_checkpointing is True

    def test_anime_preset(self, tmp_path):
        """Test 'anime' preset configuration."""
        config = Config.from_preset("anime", tmp_path)

        assert config.scale_factor == 4
        assert config.model_name == "realesr-animevideov3"
        assert config.enable_checkpointing is True

    def test_film_restoration_preset(self, tmp_path):
        """Test 'film_restoration' preset configuration."""
        config = Config.from_preset("film_restoration", tmp_path)

        assert config.scale_factor == 4
        assert config.enable_auto_enhance is True
        assert config.auto_defect_repair is True
        assert config.auto_face_restore is True

    def test_ultimate_preset(self, tmp_path):
        """Test 'ultimate' preset configuration."""
        config = Config.from_preset("ultimate", tmp_path)

        assert config.scale_factor == 4
        assert config.crf == 14
        assert config.preset == "veryslow"
        assert config.enable_tap_denoise is True
        assert config.enable_qp_artifact_removal is True

    def test_authentic_preset(self, tmp_path):
        """Test 'authentic' preset configuration."""
        config = Config.from_preset("authentic", tmp_path)

        assert config.scale_factor == 2
        assert config.enable_authenticity_guard is True
        assert config.preserve_era_character is True
        assert config.preserve_grain is True

    def test_vhs_preset(self, tmp_path):
        """Test 'vhs' preset configuration."""
        config = Config.from_preset("vhs", tmp_path)

        assert config.enable_vhs_restoration is True
        assert config.vhs_remove_tracking is True
        assert config.vhs_fix_chroma is True

    def test_invalid_preset_raises_error(self, tmp_path):
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config.from_preset("nonexistent_preset", tmp_path)

        assert "Unknown preset" in str(exc_info.value)

    def test_preset_with_overrides(self, tmp_path):
        """Test that preset can be created with overrides."""
        config = Config.from_preset("fast", tmp_path, crf=15, parallel_frames=8)

        assert config.scale_factor == 2  # From preset
        assert config.crf == 15  # Overridden
        assert config.parallel_frames == 8  # Overridden

    def test_list_presets_returns_descriptions(self):
        """Test that list_presets returns preset descriptions."""
        presets = Config.list_presets()

        assert isinstance(presets, dict)
        assert "fast" in presets
        assert "quality" in presets
        assert "archive" in presets
        assert isinstance(presets["fast"], str)


class TestConfigDirectories:
    """Test Config directory creation and management."""

    def test_create_directories_creates_all_dirs(self, tmp_path):
        """Test that create_directories creates all required directories."""
        config = Config(project_dir=tmp_path)
        config.create_directories()

        assert config.project_dir.exists()
        assert config.temp_dir.exists()
        assert config.frames_dir.exists()
        assert config.enhanced_dir.exists()

    def test_create_directories_with_checkpointing(self, tmp_path):
        """Test that checkpoint_dir is created when checkpointing enabled."""
        config = Config(project_dir=tmp_path, enable_checkpointing=True)
        config.create_directories()

        assert config.checkpoint_dir.exists()

    def test_create_directories_with_interpolation(self, tmp_path):
        """Test that interpolated_dir is created when interpolation enabled."""
        config = Config(project_dir=tmp_path, enable_interpolation=True)
        config.create_directories()

        assert config.interpolated_dir.exists()

    def test_get_output_dir_default(self, tmp_path):
        """Test get_output_dir returns project_dir/output by default."""
        config = Config(project_dir=tmp_path)

        assert config.get_output_dir() == tmp_path / "output"

    def test_get_output_dir_with_override(self, tmp_path):
        """Test get_output_dir returns override when set."""
        custom_output = tmp_path / "custom_output"
        config = Config(project_dir=tmp_path, output_dir=custom_output)

        assert config.get_output_dir() == custom_output

    def test_cleanup_temp_removes_directory(self, tmp_path):
        """Test that cleanup_temp removes temp directory."""
        config = Config(project_dir=tmp_path)
        config.create_directories()

        # Verify temp dir exists
        assert config.temp_dir.exists()

        config.cleanup_temp()

        assert not config.temp_dir.exists()


class TestConfigHash:
    """Test Config hash generation."""

    def test_get_hash_returns_string(self, tmp_path):
        """Test that get_hash returns a hex string."""
        config = Config(project_dir=tmp_path)
        hash_value = config.get_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16  # SHA256 first 16 chars

    def test_get_hash_same_config_same_hash(self, tmp_path):
        """Test that same config produces same hash."""
        config1 = Config(project_dir=tmp_path, scale_factor=4, crf=18)
        config2 = Config(project_dir=tmp_path, scale_factor=4, crf=18)

        assert config1.get_hash() == config2.get_hash()

    def test_get_hash_different_config_different_hash(self, tmp_path):
        """Test that different config produces different hash."""
        config1 = Config(project_dir=tmp_path, scale_factor=4, crf=18)
        config2 = Config(
            project_dir=tmp_path,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            crf=23
        )

        assert config1.get_hash() != config2.get_hash()


class TestCloudConfig:
    """Test CloudConfig dataclass."""

    def test_cloud_config_defaults(self):
        """Test CloudConfig with defaults."""
        config = CloudConfig()

        assert config.provider == "runpod"
        assert config.gpu_type == "RTX_4090"
        assert config.storage_backend == "s3"
        assert config.max_runtime_minutes == 120
        assert config.auto_cleanup is True

    def test_cloud_config_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CloudConfig(provider="invalid_provider")

        assert "Invalid provider" in str(exc_info.value)

    def test_cloud_config_invalid_storage_backend(self):
        """Test that invalid storage backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CloudConfig(storage_backend="invalid")

        assert "Invalid storage backend" in str(exc_info.value)

    def test_cloud_config_invalid_gpu_type(self):
        """Test that invalid GPU type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CloudConfig(gpu_type="InvalidGPU")

        assert "Invalid GPU type" in str(exc_info.value)

    def test_cloud_config_to_dict(self):
        """Test CloudConfig to_dict."""
        config = CloudConfig(provider="vastai", gpu_type="A100_80GB")
        data = config.to_dict()

        assert data["provider"] == "vastai"
        assert data["gpu_type"] == "A100_80GB"

    def test_cloud_config_from_dict(self):
        """Test CloudConfig from_dict."""
        data = {
            "provider": "runpod",
            "gpu_type": "RTX_4090",
            "max_runtime_minutes": 60,
        }
        config = CloudConfig.from_dict(data)

        assert config.provider == "runpod"
        assert config.gpu_type == "RTX_4090"
        assert config.max_runtime_minutes == 60


class TestRestoreOptions:
    """Test RestoreOptions dataclass."""

    def test_restore_options_defaults(self):
        """Test RestoreOptions with defaults."""
        options = RestoreOptions(source="/path/to/video.mp4")

        assert options.source == "/path/to/video.mp4"
        assert options.cleanup is True
        assert options.resume is True
        assert options.validate_output is True
        assert options.dry_run is False

    def test_restore_options_with_custom_values(self, tmp_path):
        """Test RestoreOptions with custom values."""
        output = tmp_path / "output.mkv"
        options = RestoreOptions(
            source="https://example.com/video.mp4",
            output_path=output,
            cleanup=False,
            resume=False,
            dry_run=True,
        )

        assert options.source == "https://example.com/video.mp4"
        assert options.output_path == output
        assert options.cleanup is False
        assert options.resume is False
        assert options.dry_run is True

    def test_restore_options_invalid_target_fps(self):
        """Test that invalid target_fps raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            RestoreOptions(source="/path/to/video.mp4", target_fps=-1)

        assert "target_fps must be positive" in str(exc_info.value)

    def test_restore_options_invalid_preview_frame_count(self):
        """Test that invalid preview_frame_count raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            RestoreOptions(source="/path/to/video.mp4", preview_frame_count=0)

        assert "preview_frame_count must be at least 1" in str(exc_info.value)
