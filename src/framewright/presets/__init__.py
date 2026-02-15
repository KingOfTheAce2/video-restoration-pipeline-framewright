"""Smart presets module for automatic configuration generation.

This module provides:
- Video analysis for detecting characteristics
- Preset generation based on detected features
- Hardware-aware preset registry
- Smart preset selection combining video analysis and hardware detection

Example usage:

    >>> from framewright.presets import SmartPresetSelector, PresetRegistry, HardwareInfo
    >>>
    >>> # Auto-select best preset for a video
    >>> selector = SmartPresetSelector()
    >>> config = selector.select("my_video.mp4")
    >>>
    >>> # Get hardware-optimized preset
    >>> hardware = HardwareInfo.detect()
    >>> config = PresetRegistry.get_for_hardware("balanced", hardware)
    >>>
    >>> # Apply a content style
    >>> config = PresetRegistry.with_style("balanced", "film", hardware)
"""

from .analyzer import VideoAnalyzer, VideoCharacteristics
from .generator import PresetGenerator, GeneratedPreset
from .library import PresetLibrary, PresetCategory
from .registry import (
    HardwareInfo,
    PresetConfig,
    PresetRegistry,
    get_preset,
    get_preset_with_style,
)
from .smart_selector import (
    SmartPresetSelector,
    SelectionReasoning,
    auto_select_preset,
)

__all__ = [
    # Analyzer
    "VideoAnalyzer",
    "VideoCharacteristics",
    # Generator
    "PresetGenerator",
    "GeneratedPreset",
    # Library
    "PresetLibrary",
    "PresetCategory",
    # Registry (new)
    "HardwareInfo",
    "PresetConfig",
    "PresetRegistry",
    "get_preset",
    "get_preset_with_style",
    # Smart Selector (new)
    "SmartPresetSelector",
    "SelectionReasoning",
    "auto_select_preset",
]
