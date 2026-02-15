"""Profile management for FrameWright user preferences.

Provides a system for saving and loading named configuration profiles,
allowing users to create reusable restoration settings.

Profiles are stored as JSON files in ~/.framewright/profiles/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProfileManager:
    """Manages user configuration profiles for FrameWright.

    Profiles allow users to save and restore their preferred settings
    for different restoration scenarios.

    Attributes:
        profiles_dir: Directory where profiles are stored
    """

    def __init__(self, profiles_dir: Optional[Path] = None) -> None:
        """Initialize the profile manager.

        Args:
            profiles_dir: Custom profiles directory path. If None, uses
                         ~/.framewright/profiles/
        """
        if profiles_dir is None:
            self.profiles_dir = Path.home() / ".framewright" / "profiles"
        else:
            self.profiles_dir = Path(profiles_dir)

    def _ensure_profiles_dir(self) -> None:
        """Create profiles directory if it doesn't exist."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def get_profile_path(self, name: str) -> Path:
        """Get the file path for a named profile.

        Args:
            name: Profile name (alphanumeric, hyphens, underscores)

        Returns:
            Path to the profile JSON file
        """
        # Sanitize name to prevent path traversal
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
        if not safe_name:
            raise ValueError(f"Invalid profile name: {name}")
        return self.profiles_dir / f"{safe_name}.json"

    def save_profile(
        self,
        name: str,
        config: "Config",  # type: ignore  # noqa: F821
        description: Optional[str] = None,
    ) -> Path:
        """Save a configuration as a named profile.

        Args:
            name: Profile name
            config: Config instance to save
            description: Optional description of the profile

        Returns:
            Path to the saved profile file

        Raises:
            ValueError: If profile name is invalid
        """
        self._ensure_profiles_dir()
        profile_path = self.get_profile_path(name)

        # Get config as dict
        config_dict = config.to_dict()

        # Remove path-specific settings that shouldn't be in profiles
        keys_to_remove = [
            "project_dir",
            "output_dir",
            "temp_dir",
            "frames_dir",
            "enhanced_dir",
            "checkpoint_dir",
            "_output_dir_override",
            "_frames_dir_override",
            "_enhanced_dir_override",
        ]
        for key in keys_to_remove:
            config_dict.pop(key, None)

        # Create profile data structure
        profile_data = {
            "name": name,
            "description": description or "",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": config_dict,
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, default=str)

        return profile_path

    def save_profile_from_dict(
        self,
        name: str,
        config_dict: Dict[str, Any],
        description: Optional[str] = None,
    ) -> Path:
        """Save a configuration dictionary as a named profile.

        Args:
            name: Profile name
            config_dict: Configuration dictionary to save
            description: Optional description of the profile

        Returns:
            Path to the saved profile file

        Raises:
            ValueError: If profile name is invalid
        """
        self._ensure_profiles_dir()
        profile_path = self.get_profile_path(name)

        # Remove path-specific settings that shouldn't be in profiles
        config_copy = config_dict.copy()
        keys_to_remove = [
            "project_dir",
            "output_dir",
            "temp_dir",
            "frames_dir",
            "enhanced_dir",
            "checkpoint_dir",
            "_output_dir_override",
            "_frames_dir_override",
            "_enhanced_dir_override",
        ]
        for key in keys_to_remove:
            config_copy.pop(key, None)

        # Create profile data structure
        profile_data = {
            "name": name,
            "description": description or "",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": config_copy,
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, default=str)

        return profile_path

    def load_profile(self, name: str) -> "Config":  # type: ignore  # noqa: F821
        """Load a named profile as a Config instance.

        Args:
            name: Profile name to load

        Returns:
            Config instance with profile settings

        Raises:
            FileNotFoundError: If profile doesn't exist
            ValueError: If profile is invalid
        """
        from ..config import Config

        profile_path = self.get_profile_path(name)

        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {name}")

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        if "config" not in profile_data:
            raise ValueError(f"Invalid profile format: {name}")

        config_dict = profile_data["config"]

        # Profiles need a project_dir to create a Config, use a placeholder
        # that will be overridden when the profile is applied
        if "project_dir" not in config_dict:
            config_dict["project_dir"] = Path.cwd()

        return Config.from_dict(config_dict)

    def load_profile_raw(self, name: str) -> Dict[str, Any]:
        """Load a profile as a raw dictionary.

        Args:
            name: Profile name to load

        Returns:
            Full profile data dictionary including metadata

        Raises:
            FileNotFoundError: If profile doesn't exist
        """
        profile_path = self.get_profile_path(name)

        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {name}")

        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_profiles(self) -> List[str]:
        """List all available profile names.

        Returns:
            Sorted list of profile names
        """
        if not self.profiles_dir.exists():
            return []

        profiles = []
        for profile_file in self.profiles_dir.glob("*.json"):
            profiles.append(profile_file.stem)

        return sorted(profiles)

    def list_profiles_detailed(self) -> List[Dict[str, Any]]:
        """List all profiles with their metadata.

        Returns:
            List of dictionaries containing profile name, description,
            and timestamps
        """
        if not self.profiles_dir.exists():
            return []

        profiles = []
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profiles.append({
                    "name": profile_file.stem,
                    "description": data.get("description", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, IOError):
                # Skip invalid profile files
                profiles.append({
                    "name": profile_file.stem,
                    "description": "(invalid profile)",
                    "created_at": "",
                    "updated_at": "",
                })

        return sorted(profiles, key=lambda x: x["name"])

    def delete_profile(self, name: str) -> bool:
        """Delete a named profile.

        Args:
            name: Profile name to delete

        Returns:
            True if profile was deleted, False if it didn't exist
        """
        profile_path = self.get_profile_path(name)

        if not profile_path.exists():
            return False

        profile_path.unlink()
        return True

    def profile_exists(self, name: str) -> bool:
        """Check if a profile exists.

        Args:
            name: Profile name to check

        Returns:
            True if profile exists
        """
        profile_path = self.get_profile_path(name)
        return profile_path.exists()

    def export_profile(self, name: str, output_path: Path) -> Path:
        """Export a profile to a custom location.

        Args:
            name: Profile name to export
            output_path: Destination path for the exported profile

        Returns:
            Path to the exported file

        Raises:
            FileNotFoundError: If profile doesn't exist
        """
        profile_path = self.get_profile_path(name)

        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {name}")

        import shutil
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile_path, output_path)
        return output_path

    def import_profile(
        self,
        source_path: Path,
        name: Optional[str] = None,
    ) -> str:
        """Import a profile from an external file.

        Args:
            source_path: Path to the profile JSON file to import
            name: Optional name override (uses file name if not provided)

        Returns:
            Name of the imported profile

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If profile format is invalid
        """
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        with open(source_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        if "config" not in profile_data:
            raise ValueError(f"Invalid profile format: {source_path}")

        if name is None:
            name = profile_data.get("name", source_path.stem)

        self._ensure_profiles_dir()
        profile_path = self.get_profile_path(name)

        # Update metadata
        profile_data["name"] = name
        profile_data["updated_at"] = datetime.now().isoformat()

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, default=str)

        return name


def get_profile_manager() -> ProfileManager:
    """Get a ProfileManager instance with default configuration.

    Returns:
        ProfileManager instance
    """
    return ProfileManager()
