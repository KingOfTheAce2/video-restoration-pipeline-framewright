"""Preset library with curated restoration presets."""

import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PresetCategory(Enum):
    """Categories for organizing presets."""
    FILM = "film"
    ANALOG_VIDEO = "analog_video"
    DIGITAL = "digital"
    SPECIAL = "special"
    USER = "user"


@dataclass
class LibraryPreset:
    """A preset in the library."""
    name: str
    display_name: str
    description: str
    category: PresetCategory
    config: Dict[str, Any]
    tags: List[str]
    author: str = "FrameWright"
    version: str = "1.0"


# Built-in presets
BUILTIN_PRESETS: Dict[str, LibraryPreset] = {
    # Film presets
    "film_silent": LibraryPreset(
        name="film_silent",
        display_name="Silent Film (Pre-1930)",
        description="Careful restoration for silent era films. Preserves grain and period characteristics.",
        category=PresetCategory.FILM,
        config={
            "enable_denoise": True,
            "denoise_method": "tap",
            "tap_preserve_grain": True,
            "denoise_strength": 0.3,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "enable_face_restore": True,
            "face_model": "codeformer",
            "face_restore_strength": 0.3,
            "enable_stabilization": True,
            "stabilization_strength": 0.5,
            "enable_scratch_removal": True,
            "preserve_authenticity": True,
            "era_mode": "silent",
            "crf": 16,
        },
        tags=["silent", "film", "grain", "vintage", "authentic"],
    ),

    "film_classic": LibraryPreset(
        name="film_classic",
        display_name="Classic Film (1930-1960)",
        description="Restoration for golden age cinema. Respects period aesthetics while improving clarity.",
        category=PresetCategory.FILM,
        config={
            "enable_denoise": True,
            "denoise_method": "restormer",
            "denoise_strength": 0.5,
            "enable_upscale": True,
            "scale_factor": 4,
            "sr_model": "realesrgan",
            "enable_face_restore": True,
            "face_model": "codeformer",
            "codeformer_fidelity": 0.7,
            "enable_scratch_removal": True,
            "enable_color_correction": True,
            "preserve_authenticity": True,
            "era_mode": "classic",
            "crf": 16,
        },
        tags=["classic", "film", "hollywood", "golden_age"],
    ),

    "film_technicolor": LibraryPreset(
        name="film_technicolor",
        display_name="Technicolor Film",
        description="Optimized for Technicolor film sources with color calibration.",
        category=PresetCategory.FILM,
        config={
            "enable_denoise": True,
            "denoise_method": "restormer",
            "denoise_strength": 0.4,
            "enable_upscale": True,
            "scale_factor": 4,
            "sr_model": "realesrgan",
            "enable_face_restore": True,
            "face_model": "codeformer",
            "enable_color_correction": True,
            "color_mode": "technicolor",
            "preserve_authenticity": True,
            "crf": 14,
        },
        tags=["technicolor", "film", "color", "classic"],
    ),

    # Analog video presets
    "vhs_standard": LibraryPreset(
        name="vhs_standard",
        display_name="VHS Standard",
        description="Standard VHS restoration with artifact removal and quality enhancement.",
        category=PresetCategory.ANALOG_VIDEO,
        config={
            "enable_vhs_restoration": True,
            "vhs_head_switch_repair": True,
            "vhs_tracking_repair": True,
            "vhs_color_bleed_fix": True,
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.7,
            "enable_upscale": True,
            "scale_factor": 4,
            "sr_model": "realesrgan",
            "enable_face_restore": True,
            "face_model": "gfpgan",
            "enable_deinterlace": True,
            "deinterlace_method": "yadif",
            "crf": 18,
        },
        tags=["vhs", "analog", "tape", "home_video"],
    ),

    "vhs_nostalgia": LibraryPreset(
        name="vhs_nostalgia",
        display_name="VHS Nostalgia",
        description="Light VHS restoration that keeps the nostalgic analog feel.",
        category=PresetCategory.ANALOG_VIDEO,
        config={
            "enable_vhs_restoration": True,
            "vhs_head_switch_repair": True,
            "vhs_tracking_repair": False,
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.3,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "enable_face_restore": False,
            "preserve_authenticity": True,
            "era_mode": "vhs",
            "crf": 20,
        },
        tags=["vhs", "analog", "nostalgic", "authentic"],
    ),

    "betamax": LibraryPreset(
        name="betamax",
        display_name="Betamax",
        description="Restoration optimized for Betamax sources.",
        category=PresetCategory.ANALOG_VIDEO,
        config={
            "enable_vhs_restoration": True,
            "vhs_format": "betamax",
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.6,
            "enable_upscale": True,
            "scale_factor": 4,
            "sr_model": "realesrgan",
            "enable_deinterlace": True,
            "crf": 18,
        },
        tags=["betamax", "analog", "tape"],
    ),

    "broadcast": LibraryPreset(
        name="broadcast",
        display_name="Broadcast TV",
        description="Restoration for broadcast TV recordings with deinterlacing.",
        category=PresetCategory.ANALOG_VIDEO,
        config={
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.5,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "enable_deinterlace": True,
            "deinterlace_method": "yadif",
            "enable_face_restore": True,
            "face_model": "gfpgan",
            "crf": 18,
        },
        tags=["broadcast", "tv", "interlaced"],
    ),

    # Digital presets
    "dvd_upscale": LibraryPreset(
        name="dvd_upscale",
        display_name="DVD Upscale",
        description="Upscale DVD quality video to HD with artifact removal.",
        category=PresetCategory.DIGITAL,
        config={
            "enable_qp_artifact_removal": True,
            "qp_strength": 0.5,
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.3,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "enable_face_restore": True,
            "face_model": "codeformer",
            "crf": 18,
        },
        tags=["dvd", "upscale", "compression"],
    ),

    "youtube_cleanup": LibraryPreset(
        name="youtube_cleanup",
        display_name="YouTube Cleanup",
        description="Clean up YouTube-quality videos with compression artifact removal.",
        category=PresetCategory.DIGITAL,
        config={
            "enable_qp_artifact_removal": True,
            "qp_auto_detect": True,
            "qp_strength": 0.7,
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.4,
            "enable_upscale": False,
            "crf": 18,
        },
        tags=["youtube", "compression", "web"],
    ),

    "webcam_enhance": LibraryPreset(
        name="webcam_enhance",
        display_name="Webcam Enhancement",
        description="Enhance webcam quality with noise reduction and upscaling.",
        category=PresetCategory.DIGITAL,
        config={
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.6,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "enable_face_restore": True,
            "face_model": "gfpgan",
            "crf": 20,
        },
        tags=["webcam", "low_quality", "faces"],
    ),

    # Special presets
    "ultimate": LibraryPreset(
        name="ultimate",
        display_name="Ultimate Quality",
        description="Maximum quality restoration using all available techniques. Requires high VRAM.",
        category=PresetCategory.SPECIAL,
        config={
            "enable_tap_denoise": True,
            "tap_model": "restormer",
            "enable_qp_artifact_removal": True,
            "enable_upscale": True,
            "scale_factor": 4,
            "sr_model": "diffusion",
            "diffusion_steps": 20,
            "enable_face_restore": True,
            "face_model": "aesrgan",
            "enable_interpolation": True,
            "target_fps": 60,
            "temporal_method": "cross_attention",
            "enable_auto_enhance": True,
            "auto_defect_repair": True,
            "crf": 14,
            "encoder_preset": "veryslow",
        },
        tags=["ultimate", "quality", "best", "slow"],
    ),

    "quick": LibraryPreset(
        name="quick",
        display_name="Quick Enhancement",
        description="Fast enhancement with minimal processing for quick previews.",
        category=PresetCategory.SPECIAL,
        config={
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.5,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "sr_model_variant": "fast",
            "enable_face_restore": False,
            "crf": 23,
            "encoder_preset": "medium",
        },
        tags=["quick", "fast", "preview"],
    ),

    "colorize": LibraryPreset(
        name="colorize",
        display_name="Colorize B&W",
        description="Colorize black and white footage with optional reference images.",
        category=PresetCategory.SPECIAL,
        config={
            "enable_colorize": True,
            "colorization_method": "deoldify",
            "colorization_strength": 1.0,
            "enable_denoise": True,
            "denoise_method": "nafnet",
            "denoise_strength": 0.4,
            "enable_upscale": True,
            "scale_factor": 2,
            "crf": 18,
        },
        tags=["colorize", "bw", "color"],
    ),

    "authentic": LibraryPreset(
        name="authentic",
        display_name="Authentic Restoration",
        description="Minimal processing to preserve original character. Only fixes major issues.",
        category=PresetCategory.SPECIAL,
        config={
            "preserve_authenticity": True,
            "authenticity_mode": "strict",
            "enable_denoise": True,
            "denoise_method": "tap",
            "tap_preserve_grain": True,
            "denoise_strength": 0.2,
            "enable_upscale": True,
            "scale_factor": 2,
            "sr_model": "realesrgan",
            "enable_face_restore": False,
            "enable_scratch_removal": True,
            "scratch_detection_sensitivity": 0.9,
            "crf": 16,
        },
        tags=["authentic", "minimal", "preserve"],
    ),
}


class PresetLibrary:
    """Library of restoration presets."""

    def __init__(self, user_presets_path: Optional[Path] = None):
        self._presets: Dict[str, LibraryPreset] = dict(BUILTIN_PRESETS)
        self._user_presets_path = user_presets_path or Path.home() / ".framewright" / "presets.json"

        self._load_user_presets()

    def _load_user_presets(self) -> None:
        """Load user-defined presets from file."""
        if not self._user_presets_path.exists():
            return

        try:
            with open(self._user_presets_path, "r") as f:
                data = json.load(f)

            for name, preset_data in data.items():
                preset = LibraryPreset(
                    name=name,
                    display_name=preset_data.get("display_name", name),
                    description=preset_data.get("description", ""),
                    category=PresetCategory.USER,
                    config=preset_data.get("config", {}),
                    tags=preset_data.get("tags", []),
                    author=preset_data.get("author", "User"),
                    version=preset_data.get("version", "1.0"),
                )
                self._presets[name] = preset

            logger.info(f"Loaded {len(data)} user presets")
        except Exception as e:
            logger.warning(f"Failed to load user presets: {e}")

    def _save_user_presets(self) -> None:
        """Save user-defined presets to file."""
        user_presets = {
            name: {
                "display_name": preset.display_name,
                "description": preset.description,
                "config": preset.config,
                "tags": preset.tags,
                "author": preset.author,
                "version": preset.version,
            }
            for name, preset in self._presets.items()
            if preset.category == PresetCategory.USER
        }

        self._user_presets_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._user_presets_path, "w") as f:
            json.dump(user_presets, f, indent=2)

    def get(self, name: str) -> Optional[LibraryPreset]:
        """Get a preset by name."""
        return self._presets.get(name)

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get just the config for a preset."""
        preset = self._presets.get(name)
        return preset.config if preset else None

    def list_all(self) -> List[LibraryPreset]:
        """List all presets."""
        return list(self._presets.values())

    def list_by_category(self, category: PresetCategory) -> List[LibraryPreset]:
        """List presets in a category."""
        return [p for p in self._presets.values() if p.category == category]

    def search(self, query: str) -> List[LibraryPreset]:
        """Search presets by name, description, or tags."""
        query = query.lower()
        results = []

        for preset in self._presets.values():
            if (
                query in preset.name.lower()
                or query in preset.display_name.lower()
                or query in preset.description.lower()
                or any(query in tag.lower() for tag in preset.tags)
            ):
                results.append(preset)

        return results

    def add_user_preset(
        self,
        name: str,
        display_name: str,
        description: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> LibraryPreset:
        """Add a user-defined preset."""
        preset = LibraryPreset(
            name=name,
            display_name=display_name,
            description=description,
            category=PresetCategory.USER,
            config=config,
            tags=tags or [],
            author="User",
        )

        self._presets[name] = preset
        self._save_user_presets()

        logger.info(f"Added user preset: {name}")
        return preset

    def remove_user_preset(self, name: str) -> bool:
        """Remove a user-defined preset."""
        if name not in self._presets:
            return False

        preset = self._presets[name]
        if preset.category != PresetCategory.USER:
            logger.warning(f"Cannot remove built-in preset: {name}")
            return False

        del self._presets[name]
        self._save_user_presets()

        logger.info(f"Removed user preset: {name}")
        return True

    def export_preset(self, name: str, path: Path) -> bool:
        """Export a preset to a file."""
        preset = self._presets.get(name)
        if not preset:
            return False

        data = {
            "name": preset.name,
            "display_name": preset.display_name,
            "description": preset.description,
            "config": preset.config,
            "tags": preset.tags,
            "author": preset.author,
            "version": preset.version,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return True

    def import_preset(self, path: Path) -> Optional[LibraryPreset]:
        """Import a preset from a file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            return self.add_user_preset(
                name=data["name"],
                display_name=data.get("display_name", data["name"]),
                description=data.get("description", ""),
                config=data.get("config", {}),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.error(f"Failed to import preset: {e}")
            return None
