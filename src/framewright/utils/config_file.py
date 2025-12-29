"""Configuration file management for FrameWright.

Supports loading configuration from:
1. User config: ~/.framewright/config.yaml
2. Project config: .framewright.yaml (in current directory)
3. CLI arguments (highest precedence)

Config files are merged with CLI taking precedence over project over user.
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# Default configuration schema
CONFIG_SCHEMA = {
    "defaults": {
        "scale_factor": {"type": int, "choices": [2, 4], "default": 4},
        "model": {"type": str, "default": "realesrgan-x4plus"},
        "output_format": {"type": str, "choices": ["mkv", "mp4", "webm", "avi", "mov"], "default": "mkv"},
        "quality": {"type": int, "range": (0, 51), "default": 18},
        "model_dir": {"type": str, "default": "~/.framewright/models/"},
        "tile_size": {"type": int, "range": (0, 2048), "default": 0},
        "enable_rife": {"type": bool, "default": False},
        "target_fps": {"type": float, "range": (1, 240), "default": None},
        "rife_model": {"type": str, "choices": ["rife-v2.3", "rife-v4.0", "rife-v4.6"], "default": "rife-v4.6"},
        "auto_enhance": {"type": bool, "default": False},
        "scratch_sensitivity": {"type": float, "range": (0, 1), "default": 0.5},
        "grain_reduction": {"type": float, "range": (0, 1), "default": 0.3},
    },
    "profiles": {
        "_is_dict": True,
        "_inherits": "defaults",
    },
}

# Default config file template
DEFAULT_CONFIG_TEMPLATE = """\
# FrameWright Configuration File
# Location: ~/.framewright/config.yaml or .framewright.yaml (project-local)
#
# CLI arguments take precedence over config file values.
# Project-local config (.framewright.yaml) overrides user config (~/.framewright/config.yaml).

# Default settings applied to all operations
defaults:
  # Upscaling factor (2 or 4)
  scale_factor: 4

  # AI model for upscaling
  # Options: realesrgan-x4plus, realesrgan-x4plus-anime, realesrgan-x2plus
  model: realesrgan-x4plus

  # Output video format
  # Options: mkv, mp4, webm, avi, mov
  output_format: mkv

  # Video quality (CRF value, lower = better quality, higher file size)
  # Range: 0-51, recommended: 15-23
  quality: 18

  # Directory for storing AI models
  model_dir: ~/.framewright/models/

  # Tile size for processing (0 = auto)
  # tile_size: 0

  # Frame interpolation settings
  # enable_rife: false
  # target_fps: 60
  # rife_model: rife-v4.6

  # Auto-enhancement settings
  # auto_enhance: false
  # scratch_sensitivity: 0.5
  # grain_reduction: 0.3

# Named profiles for specific use cases
# Use with: framewright restore --profile <name>
profiles:
  # Optimized for anime content
  anime:
    model: realesrgan-x4plus-anime
    enable_rife: true
    target_fps: 24
    quality: 18

  # Film restoration preset
  film_restoration:
    colorize: true
    colorize_model: ddcolor
    remove_watermark: true
    auto_enhance: true
    scratch_sensitivity: 0.7
    grain_reduction: 0.4

  # Fast processing for previews
  fast:
    scale_factor: 2
    tile_size: 256
    quality: 23
    model: realesrgan-x2plus

  # High quality archival
  archive:
    scale_factor: 4
    quality: 15
    auto_enhance: true
    scratch_sensitivity: 0.5
    grain_reduction: 0.2

  # Smooth video (high FPS)
  smooth:
    enable_rife: true
    target_fps: 60
    rife_model: rife-v4.6
"""


@dataclass
class ValidationError:
    """Represents a config validation error."""
    path: str
    message: str
    value: Any = None


@dataclass
class ConfigFileManager:
    """Manages configuration file loading, saving, and merging.

    Attributes:
        user_config_path: Path to user-level config file
        project_config_path: Path to project-local config file
        loaded_config: The merged configuration dictionary
    """

    user_config_path: Path = field(default_factory=lambda: Path.home() / ".framewright" / "config.yaml")
    project_config_path: Path = field(default_factory=lambda: Path.cwd() / ".framewright.yaml")
    loaded_config: Dict[str, Any] = field(default_factory=dict)
    _validation_errors: List[ValidationError] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if not isinstance(self.user_config_path, Path):
            self.user_config_path = Path(self.user_config_path)
        if not isinstance(self.project_config_path, Path):
            self.project_config_path = Path(self.project_config_path)

    @staticmethod
    def _check_yaml_available() -> None:
        """Check if PyYAML is installed."""
        if yaml is None:
            raise ImportError(
                "PyYAML is required for config file support. "
                "Install with: pip install pyyaml"
            )

    def load(self) -> Dict[str, Any]:
        """Load and merge configuration from all sources.

        Order of precedence (later overrides earlier):
        1. Built-in defaults
        2. User config (~/.framewright/config.yaml)
        3. Project config (.framewright.yaml)

        Returns:
            Merged configuration dictionary
        """
        self._check_yaml_available()
        self._validation_errors = []

        # Start with built-in defaults
        config: Dict[str, Any] = self._get_builtin_defaults()

        # Load user config
        if self.user_config_path.exists():
            user_config = self._load_yaml_file(self.user_config_path)
            if user_config:
                config = self._deep_merge(config, user_config)

        # Load project config (overrides user config)
        if self.project_config_path.exists():
            project_config = self._load_yaml_file(self.project_config_path)
            if project_config:
                config = self._deep_merge(config, project_config)

        # Validate merged config
        self._validate_config(config)

        self.loaded_config = config
        return config

    def _get_builtin_defaults(self) -> Dict[str, Any]:
        """Get built-in default configuration."""
        defaults: Dict[str, Any] = {"defaults": {}, "profiles": {}}

        for key, schema in CONFIG_SCHEMA["defaults"].items():
            if isinstance(schema, dict) and "default" in schema:
                defaults["defaults"][key] = schema["default"]

        return defaults

    def _load_yaml_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load a YAML configuration file.

        Args:
            path: Path to the YAML file

        Returns:
            Parsed configuration dictionary, or None if loading fails
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            self._validation_errors.append(
                ValidationError(path=str(path), message=f"YAML parsing error: {e}")
            )
            return None
        except OSError as e:
            self._validation_errors.append(
                ValidationError(path=str(path), message=f"Failed to read file: {e}")
            )
            return None

    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, overlay takes precedence.

        Args:
            base: Base dictionary
            overlay: Dictionary to overlay on base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate
        """
        defaults = config.get("defaults", {})

        for key, schema in CONFIG_SCHEMA["defaults"].items():
            if not isinstance(schema, dict):
                continue

            if key not in defaults:
                continue

            value = defaults[key]
            if value is None:
                continue

            # Type validation
            expected_type = schema.get("type")
            if expected_type and not isinstance(value, expected_type):
                # Allow int for float fields
                if expected_type is float and isinstance(value, int):
                    pass
                else:
                    self._validation_errors.append(
                        ValidationError(
                            path=f"defaults.{key}",
                            message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                            value=value,
                        )
                    )

            # Choices validation
            choices = schema.get("choices")
            if choices and value not in choices:
                self._validation_errors.append(
                    ValidationError(
                        path=f"defaults.{key}",
                        message=f"Invalid value. Must be one of: {choices}",
                        value=value,
                    )
                )

            # Range validation
            value_range = schema.get("range")
            if value_range and isinstance(value, (int, float)):
                min_val, max_val = value_range
                if not (min_val <= value <= max_val):
                    self._validation_errors.append(
                        ValidationError(
                            path=f"defaults.{key}",
                            message=f"Value must be between {min_val} and {max_val}",
                            value=value,
                        )
                    )

    def get_validation_errors(self) -> List[ValidationError]:
        """Get list of validation errors from last load."""
        return self._validation_errors

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation path.

        Args:
            key_path: Dot-separated path (e.g., "defaults.scale_factor")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value: Any = self.loaded_config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any, persist: bool = True) -> None:
        """Set a configuration value by dot-notation path.

        Args:
            key_path: Dot-separated path (e.g., "defaults.scale_factor")
            value: Value to set
            persist: If True, save to user config file
        """
        keys = key_path.split(".")
        config = self.loaded_config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

        if persist:
            self.save_user_config()

    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a named profile configuration.

        Profiles inherit from defaults and override specific settings.

        Args:
            profile_name: Name of the profile

        Returns:
            Merged profile configuration, or None if profile not found
        """
        profiles = self.loaded_config.get("profiles", {})
        if profile_name not in profiles:
            return None

        # Start with defaults
        defaults = self.loaded_config.get("defaults", {}).copy()

        # Merge profile settings
        profile = profiles[profile_name]
        if isinstance(profile, dict):
            defaults.update(profile)

        return defaults

    def list_profiles(self) -> List[str]:
        """Get list of available profile names.

        Returns:
            List of profile names
        """
        profiles = self.loaded_config.get("profiles", {})
        return list(profiles.keys()) if isinstance(profiles, dict) else []

    def save_user_config(self) -> None:
        """Save current configuration to user config file."""
        self._check_yaml_available()

        # Ensure directory exists
        self.user_config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.user_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.loaded_config, f, default_flow_style=False, sort_keys=False)

    def init_config(self, target: str = "user") -> Path:
        """Initialize a new configuration file with defaults.

        Args:
            target: "user" for ~/.framewright/config.yaml,
                   "project" for .framewright.yaml

        Returns:
            Path to created config file
        """
        if target == "user":
            config_path = self.user_config_path
        else:
            config_path = self.project_config_path

        # Create parent directory
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write default template
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TEMPLATE)

        return config_path

    def show_config(self, as_yaml: bool = True) -> str:
        """Get string representation of current configuration.

        Args:
            as_yaml: If True, format as YAML. Otherwise, format as key=value pairs.

        Returns:
            Configuration string
        """
        if as_yaml:
            self._check_yaml_available()
            return yaml.dump(self.loaded_config, default_flow_style=False, sort_keys=False)
        else:
            lines = []
            self._flatten_config(self.loaded_config, "", lines)
            return "\n".join(lines)

    def _flatten_config(self, config: Dict[str, Any], prefix: str, lines: List[str]) -> None:
        """Flatten nested config to dot-notation lines."""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_config(value, full_key, lines)
            else:
                lines.append(f"{full_key}={value}")

    def merge_with_cli_args(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded config with CLI arguments.

        CLI arguments take precedence over config file values.

        Args:
            cli_args: Dictionary of CLI arguments

        Returns:
            Merged configuration with CLI values taking precedence
        """
        # Start with loaded config defaults
        result = self.loaded_config.get("defaults", {}).copy()

        # Map CLI arg names to config keys
        cli_to_config_map = {
            "scale": "scale_factor",
            "model": "model",
            "format": "output_format",
            "quality": "quality",
            "model_dir": "model_dir",
            "enable_rife": "enable_rife",
            "target_fps": "target_fps",
            "rife_model": "rife_model",
            "auto_enhance": "auto_enhance",
            "scratch_sensitivity": "scratch_sensitivity",
            "grain_reduction": "grain_reduction",
            "colorize": "colorize",
            "colorize_model": "colorize_model",
            "remove_watermark": "remove_watermark",
        }

        # Override with CLI args (only non-None values)
        for cli_key, config_key in cli_to_config_map.items():
            if cli_key in cli_args and cli_args[cli_key] is not None:
                result[config_key] = cli_args[cli_key]

        # Handle profile
        if "profile" in cli_args and cli_args["profile"]:
            profile_config = self.get_profile(cli_args["profile"])
            if profile_config:
                # Merge profile, then override with explicit CLI args
                merged = profile_config.copy()
                for cli_key, config_key in cli_to_config_map.items():
                    if cli_key in cli_args and cli_args[cli_key] is not None:
                        merged[config_key] = cli_args[cli_key]
                result = merged

        return result

    def config_exists(self) -> bool:
        """Check if any configuration file exists.

        Returns:
            True if user or project config exists
        """
        return self.user_config_path.exists() or self.project_config_path.exists()


def get_config_manager() -> ConfigFileManager:
    """Get a ConfigFileManager instance with loaded configuration.

    Returns:
        ConfigFileManager with configuration loaded
    """
    manager = ConfigFileManager()
    try:
        manager.load()
    except ImportError:
        # PyYAML not installed, return empty manager
        pass
    return manager
