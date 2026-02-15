"""Hardware-aware preset registry for FrameWright.

Loads preset configurations from YAML and provides methods to
retrieve and merge presets based on hardware capabilities.

Example:
    >>> from framewright.presets.registry import PresetRegistry
    >>> from framewright.utils.gpu import get_hardware_info
    >>>
    >>> # Get a base preset
    >>> config = PresetRegistry.get("balanced")
    >>>
    >>> # Get hardware-optimized preset
    >>> hardware = get_hardware_info()
    >>> config = PresetRegistry.get_for_hardware("best", hardware)
    >>>
    >>> # Combine with style
    >>> config = PresetRegistry.with_style("balanced", "film")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union
import copy

logger = logging.getLogger(__name__)

# Try to import YAML parser
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not installed. Using fallback preset loading.")


@dataclass
class HardwareInfo:
    """Information about available hardware.

    Attributes:
        gpu_name: Name of the GPU (e.g., "NVIDIA GeForce RTX 4090")
        vram_gb: Available VRAM in gigabytes
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version string (e.g., "12.1")
        compute_capability: CUDA compute capability (e.g., (8, 9))
        ram_gb: System RAM in gigabytes
        cpu_cores: Number of CPU cores
        architecture: GPU architecture name (e.g., "ada_lovelace")
    """
    gpu_name: str = "Unknown"
    vram_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: str = ""
    compute_capability: tuple = (0, 0)
    ram_gb: float = 16.0
    cpu_cores: int = 4
    architecture: str = "unknown"

    @classmethod
    def detect(cls) -> "HardwareInfo":
        """Detect current hardware capabilities.

        Returns:
            HardwareInfo instance with detected capabilities.
        """
        info = cls()

        # Try to detect GPU via torch
        try:
            import torch
            if torch.cuda.is_available():
                info.cuda_available = True
                info.gpu_name = torch.cuda.get_device_name(0)
                info.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info.cuda_version = torch.version.cuda or ""
                info.compute_capability = torch.cuda.get_device_capability(0)

                # Detect architecture from compute capability
                cc_major = info.compute_capability[0]
                if cc_major >= 10:
                    info.architecture = "blackwell"  # RTX 50xx
                elif cc_major >= 9:
                    info.architecture = "hopper"  # H100
                elif cc_major >= 8 and info.compute_capability[1] >= 9:
                    info.architecture = "ada_lovelace"  # RTX 40xx
                elif cc_major >= 8:
                    info.architecture = "ampere"  # RTX 30xx
                elif cc_major >= 7:
                    info.architecture = "turing"  # RTX 20xx
                else:
                    info.architecture = "older"

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to detect GPU: {e}")

        # Detect system RAM
        try:
            import psutil
            info.ram_gb = psutil.virtual_memory().total / (1024**3)
            info.cpu_cores = psutil.cpu_count(logical=False) or 4
        except ImportError:
            pass

        return info

    def get_effective_vram(self, architecture_bonuses: Dict[str, float]) -> float:
        """Get effective VRAM considering architecture efficiency.

        Args:
            architecture_bonuses: Mapping of architecture to bonus multiplier.

        Returns:
            Effective VRAM in GB.
        """
        bonus = architecture_bonuses.get(self.architecture, 1.0)
        return self.vram_gb * bonus


@dataclass
class PresetConfig:
    """A fully resolved preset configuration.

    Contains all settings needed for video restoration,
    merged from base preset, hardware tier, and style.

    Attributes:
        name: Name of the preset combination
        description: Human-readable description
        settings: Dictionary of all configuration settings
        source_presets: List of preset names that were merged
        hardware_tier: Name of the hardware tier applied
        style: Name of the style applied (if any)
        warnings: List of warning messages about adjustments made
    """
    name: str
    description: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    source_presets: List[str] = field(default_factory=list)
    hardware_tier: Optional[str] = None
    style: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            key: Setting name.
            default: Default value if not found.

        Returns:
            Setting value or default.
        """
        return self.settings.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a setting value."""
        return self.settings[key]

    def __contains__(self, key: str) -> bool:
        """Check if setting exists."""
        return key in self.settings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary.

        Returns:
            Dictionary suitable for creating a Config instance.
        """
        return copy.deepcopy(self.settings)

    def to_config_kwargs(self, project_dir: Path) -> Dict[str, Any]:
        """Convert to kwargs for Config constructor.

        Args:
            project_dir: Project directory for the Config.

        Returns:
            Dictionary of kwargs for Config.__init__
        """
        kwargs = self.to_dict()
        kwargs["project_dir"] = project_dir
        return kwargs


class PresetRegistry:
    """Registry for hardware-aware presets.

    Loads preset definitions from YAML and provides methods
    to retrieve and merge presets based on hardware.

    Class Attributes:
        _presets_data: Cached preset data from YAML
        _yaml_path: Path to the presets YAML file
    """

    _presets_data: ClassVar[Optional[Dict[str, Any]]] = None
    _yaml_path: ClassVar[Path] = Path(__file__).parent / "presets.yaml"

    @classmethod
    def _load_presets(cls) -> Dict[str, Any]:
        """Load presets from YAML file.

        Returns:
            Dictionary of preset data.

        Raises:
            RuntimeError: If YAML file cannot be loaded.
        """
        if cls._presets_data is not None:
            return cls._presets_data

        if not HAS_YAML:
            # Return minimal fallback presets
            cls._presets_data = cls._get_fallback_presets()
            return cls._presets_data

        if not cls._yaml_path.exists():
            logger.warning(f"Presets file not found: {cls._yaml_path}")
            cls._presets_data = cls._get_fallback_presets()
            return cls._presets_data

        try:
            with open(cls._yaml_path, "r", encoding="utf-8") as f:
                cls._presets_data = yaml.safe_load(f)
            logger.debug(f"Loaded presets from {cls._yaml_path}")
        except Exception as e:
            logger.error(f"Failed to load presets YAML: {e}")
            cls._presets_data = cls._get_fallback_presets()

        return cls._presets_data

    @classmethod
    def _get_fallback_presets(cls) -> Dict[str, Any]:
        """Get minimal fallback presets when YAML is unavailable.

        Returns:
            Dictionary with basic preset definitions.
        """
        return {
            "primary": {
                "fast": {
                    "description": "Quick processing",
                    "scale_factor": 2,
                    "model_name": "realesrgan-x2plus",
                    "crf": 23,
                    "preset": "fast",
                    "parallel_frames": 4,
                },
                "balanced": {
                    "description": "Balanced quality/speed",
                    "scale_factor": 2,
                    "model_name": "realesrgan-x2plus",
                    "crf": 18,
                    "preset": "medium",
                    "parallel_frames": 2,
                },
                "best": {
                    "description": "Maximum quality",
                    "scale_factor": 4,
                    "model_name": "realesrgan-x4plus",
                    "crf": 16,
                    "preset": "slow",
                    "parallel_frames": 1,
                },
            },
            "hardware_tiers": {
                "cpu_only": {"max_scale_factor": 2, "tile_size": 128},
                "vram_4gb": {"max_scale_factor": 2, "tile_size": 256},
                "vram_8gb": {"max_scale_factor": 4, "tile_size": 512},
            },
            "styles": {},
            "long_form": {
                "chunk_size": 50,
                "chunk_overlap": 4,
            },
            "hardware_detection": {
                "vram_thresholds": [
                    {"max_vram": 0, "tier": "cpu_only"},
                    {"max_vram": 4, "tier": "vram_4gb"},
                    {"max_vram": 8, "tier": "vram_8gb"},
                ],
                "architecture_bonuses": {},
                "ram_requirements": {},
            },
            "merge_rules": {
                "min_merge": ["max_scale_factor", "tile_size"],
                "max_merge": ["min_ssim_threshold"],
                "hardware_override": ["tile_size"],
                "style_override": ["model_name"],
            },
        }

    @classmethod
    def reload(cls) -> None:
        """Force reload of preset data from YAML."""
        cls._presets_data = None
        cls._load_presets()

    @classmethod
    def get(cls, name: str) -> PresetConfig:
        """Get a primary preset by name.

        Args:
            name: Preset name (e.g., "fast", "balanced", "best").

        Returns:
            PresetConfig with preset settings.

        Raises:
            ValueError: If preset name is not found.
        """
        data = cls._load_presets()
        primary = data.get("primary", {})

        if name not in primary:
            available = ", ".join(primary.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")

        preset_data = copy.deepcopy(primary[name])
        description = preset_data.pop("description", "")

        return PresetConfig(
            name=name,
            description=description,
            settings=preset_data,
            source_presets=[name],
        )

    @classmethod
    def list_presets(cls) -> List[str]:
        """List available primary preset names.

        Returns:
            List of preset names.
        """
        data = cls._load_presets()
        return list(data.get("primary", {}).keys())

    @classmethod
    def list_hardware_tiers(cls) -> List[str]:
        """List available hardware tier names.

        Returns:
            List of hardware tier names.
        """
        data = cls._load_presets()
        return list(data.get("hardware_tiers", {}).keys())

    @classmethod
    def list_styles(cls) -> List[str]:
        """List available style names.

        Returns:
            List of style names.
        """
        data = cls._load_presets()
        return list(data.get("styles", {}).keys())

    @classmethod
    def get_hardware_tier(cls, hardware: HardwareInfo) -> str:
        """Determine appropriate hardware tier for given hardware.

        Args:
            hardware: Hardware information.

        Returns:
            Name of the hardware tier.
        """
        data = cls._load_presets()
        detection = data.get("hardware_detection", {})

        # Get architecture bonuses
        arch_bonuses = detection.get("architecture_bonuses", {})
        effective_vram = hardware.get_effective_vram(arch_bonuses)

        # Check VRAM thresholds
        vram_thresholds = detection.get("vram_thresholds", [])
        selected_tier = "cpu_only"

        for threshold in vram_thresholds:
            max_vram = threshold.get("max_vram", 0)
            tier = threshold.get("tier", "cpu_only")

            if effective_vram >= max_vram:
                selected_tier = tier

        # Check RAM requirements
        ram_requirements = detection.get("ram_requirements", {})
        required_ram = ram_requirements.get(selected_tier, 0)

        if hardware.ram_gb < required_ram * 0.8:  # Allow 80% of required
            # Downgrade tier if RAM is insufficient
            logger.warning(
                f"Insufficient RAM ({hardware.ram_gb:.1f}GB) for tier {selected_tier} "
                f"(requires {required_ram}GB). Downgrading."
            )
            # Find a lower tier that fits RAM
            for threshold in reversed(vram_thresholds):
                tier = threshold.get("tier", "cpu_only")
                tier_ram = ram_requirements.get(tier, 0)
                if hardware.ram_gb >= tier_ram * 0.8:
                    selected_tier = tier
                    break

        return selected_tier

    @classmethod
    def get_for_hardware(
        cls,
        name: str,
        hardware: Union[HardwareInfo, None] = None,
    ) -> PresetConfig:
        """Get a preset optimized for specific hardware.

        Args:
            name: Base preset name (e.g., "balanced").
            hardware: Hardware information. Auto-detected if None.

        Returns:
            PresetConfig merged with hardware-specific settings.
        """
        # Auto-detect hardware if not provided
        if hardware is None:
            hardware = HardwareInfo.detect()

        # Get base preset
        preset = cls.get(name)

        # Determine hardware tier
        tier_name = cls.get_hardware_tier(hardware)

        # Get hardware tier settings
        data = cls._load_presets()
        hardware_tiers = data.get("hardware_tiers", {})

        if tier_name not in hardware_tiers:
            logger.warning(f"Hardware tier '{tier_name}' not found")
            return preset

        tier_data = copy.deepcopy(hardware_tiers[tier_name])
        tier_description = tier_data.pop("description", "")
        force_adjustments = tier_data.pop("force_adjustments", {})

        # Merge settings
        merged_settings = cls._merge_settings(
            preset.settings,
            tier_data,
            force_adjustments,
            data.get("merge_rules", {}),
        )

        # Apply force adjustments
        for key, value in force_adjustments.items():
            if value is None:
                merged_settings.pop(key, None)
            else:
                merged_settings[key] = value

        # Build warnings
        warnings = []
        if tier_name == "cpu_only":
            warnings.append("No GPU detected. Processing will be slow.")
        elif hardware.vram_gb < 6:
            warnings.append(f"Limited VRAM ({hardware.vram_gb:.1f}GB). Some features disabled.")

        return PresetConfig(
            name=f"{name}_{tier_name}",
            description=f"{preset.description} ({tier_description})",
            settings=merged_settings,
            source_presets=[name, tier_name],
            hardware_tier=tier_name,
            warnings=warnings,
        )

    @classmethod
    def with_style(
        cls,
        base: str,
        style: str,
        hardware: Union[HardwareInfo, None] = None,
    ) -> PresetConfig:
        """Get a preset with a content style applied.

        Args:
            base: Base preset name (e.g., "balanced").
            style: Style name (e.g., "film", "animation").
            hardware: Hardware information. Auto-detected if None.

        Returns:
            PresetConfig with base, hardware, and style merged.

        Raises:
            ValueError: If style name is not found.
        """
        # Start with hardware-optimized preset
        preset = cls.get_for_hardware(base, hardware)

        # Get style data
        data = cls._load_presets()
        styles = data.get("styles", {})

        if style not in styles:
            available = ", ".join(styles.keys())
            raise ValueError(f"Unknown style '{style}'. Available: {available}")

        style_data = copy.deepcopy(styles[style])
        style_description = style_data.pop("description", "")
        modifiers = style_data.pop("modifiers", {})

        # Merge style settings
        merge_rules = data.get("merge_rules", {})
        merged_settings = cls._merge_settings(
            preset.settings,
            style_data,
            {},
            merge_rules,
        )

        # Apply modifiers
        cls._apply_modifiers(merged_settings, modifiers)

        # Apply style overrides
        for key in merge_rules.get("style_override", []):
            if key in style_data:
                merged_settings[key] = style_data[key]

        return PresetConfig(
            name=f"{preset.name}_{style}",
            description=f"{preset.description} Style: {style_description}",
            settings=merged_settings,
            source_presets=preset.source_presets + [f"style:{style}"],
            hardware_tier=preset.hardware_tier,
            style=style,
            warnings=preset.warnings,
        )

    @classmethod
    def _merge_settings(
        cls,
        base: Dict[str, Any],
        overlay: Dict[str, Any],
        force: Dict[str, Any],
        rules: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Merge two settings dictionaries following merge rules.

        Args:
            base: Base settings.
            overlay: Settings to merge in.
            force: Forced settings that override everything.
            rules: Merge rules dictionary.

        Returns:
            Merged settings dictionary.
        """
        result = copy.deepcopy(base)

        min_merge_keys = set(rules.get("min_merge", []))
        max_merge_keys = set(rules.get("max_merge", []))
        hardware_override_keys = set(rules.get("hardware_override", []))

        for key, value in overlay.items():
            if key in force:
                # Skip, will be applied later
                continue

            if key in hardware_override_keys:
                # Hardware tier overrides base
                result[key] = value
            elif key in min_merge_keys:
                # Take minimum
                if key in result and result[key] is not None:
                    if value is not None:
                        result[key] = min(result[key], value)
                else:
                    result[key] = value
            elif key in max_merge_keys:
                # Take maximum
                if key in result and result[key] is not None:
                    if value is not None:
                        result[key] = max(result[key], value)
                else:
                    result[key] = value
            else:
                # Default: overlay wins if present
                result[key] = value

        return result

    @classmethod
    def _apply_modifiers(
        cls,
        settings: Dict[str, Any],
        modifiers: Dict[str, Any],
    ) -> None:
        """Apply modifier rules to settings in-place.

        Modifiers can include:
        - Direct values to set
        - Multipliers (keys ending in _multiplier)
        - Caps (keys ending in _cap)

        Args:
            settings: Settings dictionary to modify.
            modifiers: Modifier rules.
        """
        for key, value in modifiers.items():
            if key.endswith("_multiplier"):
                # Apply multiplier to existing setting
                base_key = key[:-11]  # Remove "_multiplier"
                if base_key in settings:
                    settings[base_key] = settings[base_key] * value
            elif key.endswith("_cap"):
                # Cap the value
                base_key = key[:-4]  # Remove "_cap"
                if base_key in settings:
                    settings[base_key] = min(settings[base_key], value)
            else:
                # Direct value
                settings[key] = value

    @classmethod
    def get_long_form_settings(cls) -> Dict[str, Any]:
        """Get settings for long-form video processing.

        Returns:
            Dictionary of long-form processing settings.
        """
        data = cls._load_presets()
        return copy.deepcopy(data.get("long_form", {}))

    @classmethod
    def describe_preset(cls, name: str) -> str:
        """Get a human-readable description of a preset.

        Args:
            name: Preset name.

        Returns:
            Multi-line description string.
        """
        try:
            preset = cls.get(name)
        except ValueError as e:
            return str(e)

        lines = [
            f"Preset: {preset.name}",
            f"Description: {preset.description}",
            "",
            "Settings:",
        ]

        for key, value in sorted(preset.settings.items()):
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    @classmethod
    def describe_hardware_tier(cls, tier: str) -> str:
        """Get a human-readable description of a hardware tier.

        Args:
            tier: Hardware tier name.

        Returns:
            Multi-line description string.
        """
        data = cls._load_presets()
        tiers = data.get("hardware_tiers", {})

        if tier not in tiers:
            return f"Unknown hardware tier: {tier}"

        tier_data = tiers[tier]
        lines = [
            f"Hardware Tier: {tier}",
            f"Description: {tier_data.get('description', 'N/A')}",
            "",
            "Settings:",
        ]

        for key, value in sorted(tier_data.items()):
            if key not in ("description", "force_adjustments"):
                lines.append(f"  {key}: {value}")

        if "force_adjustments" in tier_data:
            lines.extend(["", "Forced Adjustments:"])
            for key, value in tier_data["force_adjustments"].items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


def get_preset(name: str, hardware: Optional[HardwareInfo] = None) -> PresetConfig:
    """Convenience function to get a hardware-aware preset.

    Args:
        name: Preset name.
        hardware: Hardware info (auto-detected if None).

    Returns:
        PresetConfig with merged settings.
    """
    return PresetRegistry.get_for_hardware(name, hardware)


def get_preset_with_style(
    name: str,
    style: str,
    hardware: Optional[HardwareInfo] = None,
) -> PresetConfig:
    """Convenience function to get a preset with style.

    Args:
        name: Preset name.
        style: Style name.
        hardware: Hardware info (auto-detected if None).

    Returns:
        PresetConfig with merged settings.
    """
    return PresetRegistry.with_style(name, style, hardware)
