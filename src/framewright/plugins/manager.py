"""Plugin manager for loading and managing plugins."""

import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .base import (
    PluginBase,
    ProcessorPlugin,
    AnalyzerPlugin,
    FilterPlugin,
    PluginMetadata,
    PluginCapability,
)

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    metadata: PluginMetadata
    plugin_class: Type[PluginBase]
    source_path: Optional[Path] = None
    is_builtin: bool = False
    is_enabled: bool = True
    load_error: Optional[str] = None


class PluginRegistry:
    """Registry of available plugins."""

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._by_capability: Dict[PluginCapability, List[str]] = {}

    def register(
        self,
        plugin_class: Type[PluginBase],
        source_path: Optional[Path] = None,
        is_builtin: bool = False,
    ) -> None:
        """Register a plugin class."""
        try:
            metadata = plugin_class.get_metadata()
            name = metadata.name

            if name in self._plugins:
                existing = self._plugins[name]
                if existing.is_builtin and not is_builtin:
                    logger.warning(f"Cannot override built-in plugin: {name}")
                    return
                logger.info(f"Replacing plugin: {name}")

            info = PluginInfo(
                metadata=metadata,
                plugin_class=plugin_class,
                source_path=source_path,
                is_builtin=is_builtin,
            )
            self._plugins[name] = info

            # Index by capability
            for cap in metadata.capabilities:
                if cap not in self._by_capability:
                    self._by_capability[cap] = []
                if name not in self._by_capability[cap]:
                    self._by_capability[cap].append(name)

            logger.info(f"Registered plugin: {name} v{metadata.version}")

        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name not in self._plugins:
            return False

        info = self._plugins[name]
        if info.is_builtin:
            logger.warning(f"Cannot unregister built-in plugin: {name}")
            return False

        # Remove from capability index
        for cap in info.metadata.capabilities:
            if cap in self._by_capability:
                self._by_capability[cap] = [
                    n for n in self._by_capability[cap] if n != name
                ]

        del self._plugins[name]
        logger.info(f"Unregistered plugin: {name}")
        return True

    def get(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        return self._plugins.get(name)

    def get_by_capability(self, capability: PluginCapability) -> List[PluginInfo]:
        """Get plugins providing a capability."""
        names = self._by_capability.get(capability, [])
        return [self._plugins[n] for n in names if n in self._plugins]

    def list_all(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return list(self._plugins.values())

    def list_enabled(self) -> List[PluginInfo]:
        """List enabled plugins."""
        return [p for p in self._plugins.values() if p.is_enabled]

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self._plugins:
            self._plugins[name].is_enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        if name in self._plugins:
            self._plugins[name].is_enabled = False
            return True
        return False


class PluginLoader:
    """Loads plugins from various sources."""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self._plugin_dirs: List[Path] = []

    def add_plugin_directory(self, path: Path) -> None:
        """Add a directory to search for plugins."""
        path = Path(path)
        if path.exists() and path.is_dir():
            self._plugin_dirs.append(path)
            logger.info(f"Added plugin directory: {path}")

    def load_builtin_plugins(self) -> int:
        """Load built-in plugins."""
        count = 0
        # Built-in plugins would be registered here
        # For now, return 0 as they're defined elsewhere
        return count

    def load_from_directory(self, directory: Path) -> int:
        """Load plugins from a directory."""
        directory = Path(directory)
        if not directory.exists():
            return 0

        count = 0
        for path in directory.iterdir():
            if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
                if self.load_from_file(path):
                    count += 1
            elif path.is_dir() and (path / "__init__.py").exists():
                if self.load_from_package(path):
                    count += 1

        return count

    def load_from_file(self, path: Path) -> bool:
        """Load a plugin from a Python file."""
        try:
            path = Path(path)
            module_name = f"framewright_plugin_{path.stem}"

            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find plugin classes
            found = False
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, PluginBase)
                    and obj not in (PluginBase, ProcessorPlugin, AnalyzerPlugin, FilterPlugin)
                ):
                    try:
                        self.registry.register(obj, source_path=path)
                        found = True
                    except Exception as e:
                        logger.error(f"Failed to register {name}: {e}")

            return found

        except Exception as e:
            logger.error(f"Failed to load plugin from {path}: {e}")
            return False

    def load_from_package(self, path: Path) -> bool:
        """Load a plugin from a package directory."""
        try:
            path = Path(path)
            module_name = f"framewright_plugin_{path.name}"

            # Add to path temporarily
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

            try:
                module = importlib.import_module(path.name)

                # Find plugin classes
                found = False
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, PluginBase)
                        and obj not in (PluginBase, ProcessorPlugin, AnalyzerPlugin, FilterPlugin)
                    ):
                        try:
                            self.registry.register(obj, source_path=path)
                            found = True
                        except Exception as e:
                            logger.error(f"Failed to register {name}: {e}")

                return found
            finally:
                if parent in sys.path:
                    sys.path.remove(parent)

        except Exception as e:
            logger.error(f"Failed to load plugin from {path}: {e}")
            return False

    def load_all(self) -> int:
        """Load plugins from all configured directories."""
        count = self.load_builtin_plugins()

        for directory in self._plugin_dirs:
            count += self.load_from_directory(directory)

        logger.info(f"Loaded {count} plugins total")
        return count


class PluginManager:
    """Main plugin manager for the application."""

    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        auto_load: bool = True,
    ):
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)

        # Active plugin instances
        self._instances: Dict[str, PluginBase] = {}
        self._device: str = "cpu"

        # Default plugin directories
        default_dirs = [
            Path.home() / ".framewright" / "plugins",
            Path(__file__).parent.parent / "plugins_contrib",
        ]

        for d in default_dirs:
            self.loader.add_plugin_directory(d)

        if plugin_dirs:
            for d in plugin_dirs:
                self.loader.add_plugin_directory(d)

        if auto_load:
            self.loader.load_all()

    def set_device(self, device: str) -> None:
        """Set default device for plugins."""
        self._device = device

    def get_plugin(
        self,
        name: str,
        settings: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> Optional[PluginBase]:
        """Get or create a plugin instance."""
        # Check for existing instance
        if name in self._instances:
            instance = self._instances[name]
            if settings:
                instance.update_settings(settings)
            return instance

        # Create new instance
        info = self.registry.get(name)
        if info is None or not info.is_enabled:
            logger.warning(f"Plugin not available: {name}")
            return None

        try:
            instance = info.plugin_class()
            use_device = device or self._device
            instance.initialize(device=use_device, settings=settings)

            self._instances[name] = instance
            return instance

        except Exception as e:
            logger.error(f"Failed to initialize plugin {name}: {e}")
            return None

    def get_processor(
        self,
        name: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProcessorPlugin]:
        """Get a processor plugin."""
        plugin = self.get_plugin(name, settings)
        if plugin and isinstance(plugin, ProcessorPlugin):
            return plugin
        return None

    def get_analyzer(
        self,
        name: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnalyzerPlugin]:
        """Get an analyzer plugin."""
        plugin = self.get_plugin(name, settings)
        if plugin and isinstance(plugin, AnalyzerPlugin):
            return plugin
        return None

    def get_filter(
        self,
        name: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[FilterPlugin]:
        """Get a filter plugin."""
        plugin = self.get_plugin(name, settings)
        if plugin and isinstance(plugin, FilterPlugin):
            return plugin
        return None

    def find_plugins_for_capability(
        self,
        capability: PluginCapability,
    ) -> List[PluginInfo]:
        """Find plugins providing a capability."""
        return self.registry.get_by_capability(capability)

    def release_plugin(self, name: str) -> None:
        """Release a plugin instance."""
        if name in self._instances:
            self._instances[name].cleanup()
            del self._instances[name]

    def release_all(self) -> None:
        """Release all plugin instances."""
        for name in list(self._instances.keys()):
            self.release_plugin(name)

    def list_plugins(
        self,
        capability: Optional[PluginCapability] = None,
        plugin_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available plugins."""
        if capability:
            infos = self.registry.get_by_capability(capability)
        else:
            infos = self.registry.list_all()

        # Filter by type
        if plugin_type:
            type_map = {
                "processor": ProcessorPlugin,
                "analyzer": AnalyzerPlugin,
                "filter": FilterPlugin,
            }
            target_type = type_map.get(plugin_type.lower())
            if target_type:
                infos = [
                    i for i in infos
                    if issubclass(i.plugin_class, target_type)
                ]

        return [
            {
                **info.metadata.to_dict(),
                "is_enabled": info.is_enabled,
                "is_builtin": info.is_builtin,
                "is_loaded": info.metadata.name in self._instances,
            }
            for info in infos
        ]

    def export_config(self, path: Path) -> None:
        """Export plugin configuration."""
        config = {
            "enabled": [
                name for name, info in self.registry._plugins.items()
                if info.is_enabled
            ],
            "disabled": [
                name for name, info in self.registry._plugins.items()
                if not info.is_enabled
            ],
            "settings": {
                name: instance.settings
                for name, instance in self._instances.items()
            },
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def import_config(self, path: Path) -> None:
        """Import plugin configuration."""
        with open(path, "r") as f:
            config = json.load(f)

        for name in config.get("enabled", []):
            self.registry.enable(name)

        for name in config.get("disabled", []):
            self.registry.disable(name)

        # Apply settings to loaded plugins
        for name, settings in config.get("settings", {}).items():
            if name in self._instances:
                self._instances[name].update_settings(settings)
