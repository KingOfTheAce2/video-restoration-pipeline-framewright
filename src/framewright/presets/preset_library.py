"""Preset Library for Community-Shared Restoration Presets.

Provides a system for creating, saving, sharing, and importing
restoration presets with metadata and ratings.

Features:
- Built-in presets for common scenarios
- User-created custom presets
- Import/export for sharing
- Preset validation
- Content-type specific presets

Example:
    >>> library = PresetLibrary()
    >>> library.list_presets(category="vhs")
    >>> preset = library.get_preset("vhs_home_movie")
    >>> library.save_preset("my_preset", config, description="My custom settings")
"""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PresetCategory(Enum):
    """Preset categories."""
    GENERAL = "general"
    VHS = "vhs"
    FILM = "film"
    ANIMATION = "animation"
    DOCUMENTARY = "documentary"
    HOME_VIDEO = "home_video"
    BROADCAST = "broadcast"
    GAMING = "gaming"
    CUSTOM = "custom"


class ContentEra(Enum):
    """Content era for presets."""
    SILENT = "silent"  # Pre-1930
    EARLY_SOUND = "early_sound"  # 1930-1950
    CLASSIC = "classic"  # 1950-1970
    MODERN = "modern"  # 1970-2000
    DIGITAL = "digital"  # 2000+
    ANY = "any"


@dataclass
class PresetMetadata:
    """Metadata for a preset."""
    name: str
    description: str
    category: PresetCategory = PresetCategory.GENERAL
    era: ContentEra = ContentEra.ANY
    author: str = "FrameWright"
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    rating: float = 0.0  # 0-5 stars
    usage_count: int = 0

    # Processing hints
    recommended_for: List[str] = field(default_factory=list)
    not_recommended_for: List[str] = field(default_factory=list)
    estimated_quality_boost: str = "medium"  # low, medium, high, maximum
    estimated_processing_time: str = "medium"  # fast, medium, slow, very_slow


@dataclass
class Preset:
    """A restoration preset."""
    metadata: PresetMetadata
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": asdict(self.metadata),
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        """Create from dictionary."""
        metadata_data = data.get("metadata", {})

        # Handle enum conversion
        if "category" in metadata_data:
            metadata_data["category"] = PresetCategory(metadata_data["category"])
        if "era" in metadata_data:
            metadata_data["era"] = ContentEra(metadata_data["era"])

        metadata = PresetMetadata(**metadata_data)
        return cls(metadata=metadata, config=data.get("config", {}))


# Built-in presets
BUILTIN_PRESETS: Dict[str, Preset] = {
    # VHS presets
    "vhs_home_movie": Preset(
        metadata=PresetMetadata(
            name="VHS Home Movie",
            description="Optimized for home-recorded VHS tapes with typical artifacts",
            category=PresetCategory.VHS,
            era=ContentEra.MODERN,
            tags=["vhs", "home", "family", "analog"],
            recommended_for=["Home recordings", "Family videos", "Birthday parties"],
            estimated_quality_boost="high",
            estimated_processing_time="medium",
        ),
        config={
            "preset": "quality",
            "scale_factor": 2,
            "enable_tap_denoise": True,
            "temporal_method": "hybrid",
            "enable_vhs_restoration": True,
            "auto_face_restore": True,
            "enable_audio_enhance": True,
        },
    ),
    "vhs_commercial": Preset(
        metadata=PresetMetadata(
            name="VHS Commercial Recording",
            description="For commercial VHS releases with better source quality",
            category=PresetCategory.VHS,
            era=ContentEra.MODERN,
            tags=["vhs", "commercial", "movie", "rental"],
            recommended_for=["Movie rentals", "Commercial releases"],
            estimated_quality_boost="medium",
            estimated_processing_time="medium",
        ),
        config={
            "preset": "quality",
            "scale_factor": 4,
            "enable_tap_denoise": True,
            "enable_qp_artifact_removal": True,
            "temporal_method": "cross_attention",
        },
    ),

    # Film presets
    "film_8mm": Preset(
        metadata=PresetMetadata(
            name="8mm Film",
            description="For digitized 8mm home movies",
            category=PresetCategory.FILM,
            era=ContentEra.CLASSIC,
            tags=["8mm", "film", "home", "vintage"],
            recommended_for=["8mm film scans", "Home movies"],
            estimated_quality_boost="high",
            estimated_processing_time="slow",
        ),
        config={
            "preset": "ultimate",
            "scale_factor": 4,
            "enable_tap_denoise": True,
            "enable_frame_generation": True,
            "enable_film_restoration": True,
            "temporal_method": "hybrid",
            "auto_face_restore": True,
            "enable_stabilization": True,
        },
    ),
    "film_16mm": Preset(
        metadata=PresetMetadata(
            name="16mm Film",
            description="For 16mm film scans with higher quality source",
            category=PresetCategory.FILM,
            era=ContentEra.CLASSIC,
            tags=["16mm", "film", "professional"],
            recommended_for=["16mm film scans", "News footage", "Documentaries"],
            estimated_quality_boost="high",
            estimated_processing_time="slow",
        ),
        config={
            "preset": "ultimate",
            "scale_factor": 2,
            "enable_tap_denoise": True,
            "enable_film_restoration": True,
            "enable_temporal_colorization": True,
            "grain_preservation": 0.3,
        },
    ),
    "film_35mm_archive": Preset(
        metadata=PresetMetadata(
            name="35mm Archive",
            description="For 35mm film scans from archives, maximum quality",
            category=PresetCategory.FILM,
            era=ContentEra.ANY,
            tags=["35mm", "film", "archive", "cinema"],
            recommended_for=["Archive restoration", "Cinema classics"],
            estimated_quality_boost="maximum",
            estimated_processing_time="very_slow",
        ),
        config={
            "preset": "rtx5090",
            "scale_factor": 2,
            "enable_ensemble_sr": True,
            "enable_temporal_colorization": True,
            "enable_raft_flow": True,
            "enable_film_restoration": True,
            "sr_model": "hat",
            "grain_preservation": 0.5,
        },
    ),

    # Animation presets
    "animation_cel": Preset(
        metadata=PresetMetadata(
            name="Cel Animation",
            description="For hand-drawn cel animation",
            category=PresetCategory.ANIMATION,
            era=ContentEra.ANY,
            tags=["animation", "cel", "cartoon", "hand-drawn"],
            recommended_for=["Classic cartoons", "Anime", "Hand-drawn animation"],
            estimated_quality_boost="high",
            estimated_processing_time="medium",
        ),
        config={
            "preset": "quality",
            "scale_factor": 4,
            "enable_tap_denoise": True,
            "animation_mode": True,
            "line_preservation": True,
            "temporal_method": "cross_attention",
        },
    ),

    # Documentary presets
    "documentary_archive": Preset(
        metadata=PresetMetadata(
            name="Documentary Archive",
            description="For historical documentary footage",
            category=PresetCategory.DOCUMENTARY,
            era=ContentEra.ANY,
            tags=["documentary", "archive", "historical", "news"],
            recommended_for=["Historical footage", "News archives", "Documentaries"],
            estimated_quality_boost="high",
            estimated_processing_time="slow",
        ),
        config={
            "preset": "quality",
            "scale_factor": 2,
            "enable_tap_denoise": True,
            "enable_frame_generation": True,
            "preserve_grain": True,
            "enable_audio_enhance": True,
        },
    ),

    # Broadcast presets
    "broadcast_sd": Preset(
        metadata=PresetMetadata(
            name="SD Broadcast",
            description="For SD television recordings",
            category=PresetCategory.BROADCAST,
            era=ContentEra.MODERN,
            tags=["broadcast", "tv", "sd", "ntsc", "pal"],
            recommended_for=["TV recordings", "DVR captures", "Broadcast archives"],
            estimated_quality_boost="high",
            estimated_processing_time="medium",
        ),
        config={
            "preset": "quality",
            "scale_factor": 4,
            "enable_qp_artifact_removal": True,
            "enable_tap_denoise": True,
            "enable_deinterlace": True,
            "deinterlace_method": "bwdif",
        },
    ),

    # Gaming presets
    "gaming_retro": Preset(
        metadata=PresetMetadata(
            name="Retro Gaming",
            description="For retro game footage (preserve pixel art)",
            category=PresetCategory.GAMING,
            era=ContentEra.MODERN,
            tags=["gaming", "retro", "pixel", "8bit", "16bit"],
            recommended_for=["Retro game footage", "Pixel art games"],
            not_recommended_for=["Modern 3D games"],
            estimated_quality_boost="medium",
            estimated_processing_time="fast",
        ),
        config={
            "preset": "fast",
            "scale_factor": 4,
            "pixel_art_mode": True,
            "nearest_neighbor_scale": True,
            "enable_scanlines": False,
        },
    ),

    # Quick presets
    "quick_cleanup": Preset(
        metadata=PresetMetadata(
            name="Quick Cleanup",
            description="Fast cleanup for modern videos with minor issues",
            category=PresetCategory.GENERAL,
            era=ContentEra.DIGITAL,
            tags=["quick", "fast", "cleanup", "modern"],
            recommended_for=["Modern videos", "Minor cleanup", "Quick fixes"],
            estimated_quality_boost="low",
            estimated_processing_time="fast",
        ),
        config={
            "preset": "fast",
            "scale_factor": 1,
            "enable_tap_denoise": True,
        },
    ),
    "youtube_ready": Preset(
        metadata=PresetMetadata(
            name="YouTube Ready",
            description="Optimized for YouTube upload",
            category=PresetCategory.GENERAL,
            era=ContentEra.ANY,
            tags=["youtube", "upload", "web", "streaming"],
            recommended_for=["YouTube uploads", "Social media"],
            estimated_quality_boost="medium",
            estimated_processing_time="medium",
        ),
        config={
            "preset": "balanced",
            "scale_factor": 2,
            "export_preset": "youtube",
            "max_resolution": 2160,
            "target_bitrate": "high",
        },
    ),
}


class PresetLibrary:
    """Library for managing restoration presets."""

    PRESETS_DIR = "presets"
    USER_PRESETS_FILE = "user_presets.json"

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize preset library.

        Args:
            storage_dir: Directory for storing user presets
        """
        self.storage_dir = storage_dir or (Path.home() / ".framewright" / self.PRESETS_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._user_presets: Dict[str, Preset] = {}
        self._load_user_presets()

    def _load_user_presets(self) -> None:
        """Load user presets from storage."""
        presets_file = self.storage_dir / self.USER_PRESETS_FILE
        if not presets_file.exists():
            return

        try:
            with open(presets_file) as f:
                data = json.load(f)

            for name, preset_data in data.items():
                try:
                    self._user_presets[name] = Preset.from_dict(preset_data)
                except Exception as e:
                    logger.warning(f"Failed to load preset '{name}': {e}")

            logger.info(f"Loaded {len(self._user_presets)} user presets")

        except Exception as e:
            logger.warning(f"Failed to load user presets: {e}")

    def _save_user_presets(self) -> None:
        """Save user presets to storage."""
        presets_file = self.storage_dir / self.USER_PRESETS_FILE

        try:
            data = {
                name: preset.to_dict()
                for name, preset in self._user_presets.items()
            }

            with open(presets_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save user presets: {e}")

    def list_presets(
        self,
        category: Optional[str] = None,
        era: Optional[str] = None,
        include_builtin: bool = True,
        include_user: bool = True,
    ) -> List[Dict[str, Any]]:
        """List available presets.

        Args:
            category: Filter by category
            era: Filter by era
            include_builtin: Include built-in presets
            include_user: Include user presets

        Returns:
            List of preset info dictionaries
        """
        presets = []

        if include_builtin:
            for name, preset in BUILTIN_PRESETS.items():
                presets.append({
                    "name": name,
                    "is_builtin": True,
                    **asdict(preset.metadata),
                })

        if include_user:
            for name, preset in self._user_presets.items():
                presets.append({
                    "name": name,
                    "is_builtin": False,
                    **asdict(preset.metadata),
                })

        # Filter
        if category:
            cat = PresetCategory(category) if isinstance(category, str) else category
            presets = [p for p in presets if p.get("category") == cat]

        if era:
            era_enum = ContentEra(era) if isinstance(era, str) else era
            presets = [p for p in presets if p.get("era") in (era_enum, ContentEra.ANY)]

        return presets

    def get_preset(self, name: str) -> Optional[Preset]:
        """Get a preset by name.

        Args:
            name: Preset name

        Returns:
            Preset or None if not found
        """
        # Check user presets first (allow overriding builtin)
        if name in self._user_presets:
            preset = self._user_presets[name]
            preset.metadata.usage_count += 1
            self._save_user_presets()
            return preset

        if name in BUILTIN_PRESETS:
            return BUILTIN_PRESETS[name]

        return None

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get just the config from a preset.

        Args:
            name: Preset name

        Returns:
            Config dictionary or None
        """
        preset = self.get_preset(name)
        return preset.config if preset else None

    def save_preset(
        self,
        name: str,
        config: Dict[str, Any],
        description: str = "",
        category: str = "custom",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> bool:
        """Save a user preset.

        Args:
            name: Preset name
            config: Configuration dictionary
            description: Preset description
            category: Category name
            tags: List of tags
            overwrite: Overwrite existing preset

        Returns:
            True if saved successfully
        """
        if name in self._user_presets and not overwrite:
            logger.warning(f"Preset '{name}' already exists")
            return False

        if name in BUILTIN_PRESETS and not overwrite:
            logger.warning(f"Cannot overwrite builtin preset '{name}'")
            return False

        metadata = PresetMetadata(
            name=name,
            description=description,
            category=PresetCategory(category),
            author="User",
            tags=tags or [],
        )

        self._user_presets[name] = Preset(metadata=metadata, config=config)
        self._save_user_presets()

        logger.info(f"Saved preset: {name}")
        return True

    def delete_preset(self, name: str) -> bool:
        """Delete a user preset.

        Args:
            name: Preset name

        Returns:
            True if deleted
        """
        if name in BUILTIN_PRESETS:
            logger.warning(f"Cannot delete builtin preset '{name}'")
            return False

        if name not in self._user_presets:
            return False

        del self._user_presets[name]
        self._save_user_presets()

        logger.info(f"Deleted preset: {name}")
        return True

    def export_preset(self, name: str, output_path: Path) -> bool:
        """Export a preset to a file for sharing.

        Args:
            name: Preset name
            output_path: Output file path

        Returns:
            True if exported
        """
        preset = self.get_preset(name)
        if not preset:
            return False

        try:
            with open(output_path, 'w') as f:
                json.dump(preset.to_dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to export preset: {e}")
            return False

    def import_preset(self, input_path: Path, name: Optional[str] = None) -> bool:
        """Import a preset from a file.

        Args:
            input_path: Input file path
            name: Override preset name

        Returns:
            True if imported
        """
        try:
            with open(input_path) as f:
                data = json.load(f)

            preset = Preset.from_dict(data)

            if name:
                preset.metadata.name = name

            self._user_presets[preset.metadata.name] = preset
            self._save_user_presets()

            logger.info(f"Imported preset: {preset.metadata.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to import preset: {e}")
            return False

    def recommend_preset(
        self,
        content_type: Optional[str] = None,
        era: Optional[str] = None,
        source_format: Optional[str] = None,
    ) -> Optional[str]:
        """Recommend a preset based on content characteristics.

        Args:
            content_type: Type of content (film, animation, etc.)
            era: Content era
            source_format: Source format (vhs, 8mm, etc.)

        Returns:
            Recommended preset name or None
        """
        # Simple recommendation logic
        if source_format:
            format_lower = source_format.lower()
            if "vhs" in format_lower:
                return "vhs_home_movie"
            elif "8mm" in format_lower:
                return "film_8mm"
            elif "16mm" in format_lower:
                return "film_16mm"
            elif "35mm" in format_lower:
                return "film_35mm_archive"

        if content_type:
            type_lower = content_type.lower()
            if "animation" in type_lower or "cartoon" in type_lower:
                return "animation_cel"
            elif "documentary" in type_lower:
                return "documentary_archive"
            elif "game" in type_lower or "gaming" in type_lower:
                return "gaming_retro"

        # Default
        return "quick_cleanup"


# Global library instance
_library: Optional[PresetLibrary] = None


def get_library() -> PresetLibrary:
    """Get or create global library instance."""
    global _library
    if _library is None:
        _library = PresetLibrary()
    return _library


def list_presets(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to list presets."""
    return get_library().list_presets(category=category)


def get_preset_config(name: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get preset config."""
    return get_library().get_config(name)
