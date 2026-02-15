"""Plugin architecture for extensible restoration processors."""

from .base import (
    PluginBase,
    ProcessorPlugin,
    AnalyzerPlugin,
    FilterPlugin,
    PluginMetadata,
    PluginCapability,
)
from .manager import (
    PluginManager,
    PluginRegistry,
    PluginLoader,
)
from .hooks import (
    HookManager,
    HookPoint,
    HookCallback,
)

__all__ = [
    "PluginBase",
    "ProcessorPlugin",
    "AnalyzerPlugin",
    "FilterPlugin",
    "PluginMetadata",
    "PluginCapability",
    "PluginManager",
    "PluginRegistry",
    "PluginLoader",
    "HookManager",
    "HookPoint",
    "HookCallback",
]
