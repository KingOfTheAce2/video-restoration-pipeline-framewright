"""Base classes for plugin architecture."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type
import numpy as np

logger = logging.getLogger(__name__)


class PluginCapability(Enum):
    """Capabilities a plugin can provide."""
    # Processing capabilities
    DENOISE = auto()
    UPSCALE = auto()
    INTERPOLATE = auto()
    COLORIZE = auto()
    FACE_RESTORE = auto()
    ARTIFACT_REMOVAL = auto()
    STABILIZE = auto()
    DEINTERLACE = auto()

    # Analysis capabilities
    QUALITY_ANALYSIS = auto()
    SCENE_DETECTION = auto()
    FACE_DETECTION = auto()
    DEFECT_DETECTION = auto()
    MOTION_ANALYSIS = auto()

    # Filter capabilities
    COLOR_CORRECTION = auto()
    GRAIN_SYNTHESIS = auto()
    SHARPENING = auto()
    TEMPORAL_FILTER = auto()

    # Format capabilities
    VHS_RESTORATION = auto()
    FILM_RESTORATION = auto()
    BROADCAST_RESTORATION = auto()

    # I/O capabilities
    CUSTOM_INPUT = auto()
    CUSTOM_OUTPUT = auto()
    METADATA_HANDLER = auto()


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    name: str
    version: str
    description: str
    author: str = ""
    website: str = ""
    license: str = ""

    # Requirements
    capabilities: Set[PluginCapability] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)  # Other plugin names
    python_packages: List[str] = field(default_factory=list)

    # Resource requirements
    min_vram_mb: int = 0
    recommended_vram_mb: int = 0
    supports_cpu: bool = True
    supports_cuda: bool = True
    supports_mps: bool = False

    # Compatibility
    min_framewright_version: str = "1.0.0"
    max_framewright_version: Optional[str] = None

    # Settings schema
    settings_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "website": self.website,
            "license": self.license,
            "capabilities": [c.name for c in self.capabilities],
            "dependencies": self.dependencies,
            "python_packages": self.python_packages,
            "min_vram_mb": self.min_vram_mb,
            "recommended_vram_mb": self.recommended_vram_mb,
            "supports_cpu": self.supports_cpu,
            "supports_cuda": self.supports_cuda,
            "supports_mps": self.supports_mps,
        }


class PluginBase(ABC):
    """Base class for all plugins."""

    def __init__(self):
        self._initialized = False
        self._settings: Dict[str, Any] = {}
        self._device: str = "cpu"

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    def initialize(self, device: str = "cpu", settings: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin."""
        self._device = device
        if settings:
            self._settings.update(settings)
        self._on_initialize()
        self._initialized = True
        logger.info(f"Plugin initialized: {self.get_metadata().name}")

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._on_cleanup()
        self._initialized = False
        logger.info(f"Plugin cleaned up: {self.get_metadata().name}")

    def _on_initialize(self) -> None:
        """Override for custom initialization."""
        pass

    def _on_cleanup(self) -> None:
        """Override for custom cleanup."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def settings(self) -> Dict[str, Any]:
        """Get plugin settings."""
        return self._settings

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update plugin settings."""
        self._settings.update(settings)
        self._on_settings_changed(settings)

    def _on_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Override to handle settings changes."""
        pass

    def validate_requirements(self) -> List[str]:
        """Validate plugin requirements. Returns list of issues."""
        issues = []
        meta = self.get_metadata()

        # Check Python packages
        for package in meta.python_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing Python package: {package}")

        # Check VRAM if CUDA
        if self._device.startswith("cuda") and meta.min_vram_mb > 0:
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    if vram < meta.min_vram_mb:
                        issues.append(f"Insufficient VRAM: {vram:.0f}MB < {meta.min_vram_mb}MB required")
            except:
                pass

        return issues


class ProcessorPlugin(PluginBase):
    """Base class for frame processing plugins."""

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Process a single frame.

        Args:
            frame: Input frame (BGR, uint8)
            frame_number: Frame index in video
            context: Optional context (previous frames, metadata, etc.)

        Returns:
            Processed frame (BGR, uint8)
        """
        pass

    def process_batch(
        self,
        frames: List[np.ndarray],
        start_frame: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        """Process a batch of frames.

        Override for batch-optimized processing.
        """
        return [
            self.process_frame(frame, start_frame + i, context)
            for i, frame in enumerate(frames)
        ]

    def get_temporal_radius(self) -> int:
        """Get number of frames needed before/after current frame.

        Override if plugin needs temporal context.
        """
        return 0

    def supports_batch(self) -> bool:
        """Check if batch processing is optimized."""
        return False

    def estimate_output_size(self, input_size: tuple) -> tuple:
        """Estimate output frame size.

        Args:
            input_size: (height, width) of input

        Returns:
            (height, width) of output
        """
        return input_size

    def get_progress_weight(self) -> float:
        """Get relative processing weight for progress estimation.

        Higher = more expensive. Default 1.0.
        """
        return 1.0


class AnalyzerPlugin(PluginBase):
    """Base class for analysis plugins."""

    @abstractmethod
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> Dict[str, Any]:
        """Analyze a single frame.

        Returns analysis results as dictionary.
        """
        pass

    def analyze_video(
        self,
        frames: List[np.ndarray],
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Analyze complete video.

        Override for video-level analysis.
        """
        results = []
        for i, frame in enumerate(frames):
            result = self.analyze_frame(frame, i)
            results.append(result)
            if callback:
                callback(i, result)

        return {"frames": results}

    def get_analysis_schema(self) -> Dict[str, Any]:
        """Get JSON schema for analysis results."""
        return {}


class FilterPlugin(PluginBase):
    """Base class for simple filter plugins."""

    @abstractmethod
    def apply(
        self,
        frame: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Apply filter to frame.

        Args:
            frame: Input frame
            strength: Filter strength (0.0-1.0)

        Returns:
            Filtered frame
        """
        pass

    def preview(
        self,
        frame: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Generate preview (may be lower quality for speed)."""
        return self.apply(frame, strength)

    def get_strength_range(self) -> tuple:
        """Get valid strength range (min, max, default)."""
        return (0.0, 1.0, 0.5)


# Type alias for plugin factories
PluginFactory = Callable[[], PluginBase]


def plugin(
    name: str,
    version: str,
    description: str,
    capabilities: Optional[Set[PluginCapability]] = None,
    **metadata_kwargs
) -> Callable[[Type[PluginBase]], Type[PluginBase]]:
    """Decorator for registering plugins with metadata."""

    def decorator(cls: Type[PluginBase]) -> Type[PluginBase]:
        # Create metadata
        meta = PluginMetadata(
            name=name,
            version=version,
            description=description,
            capabilities=capabilities or set(),
            **metadata_kwargs
        )

        # Override get_metadata
        @classmethod
        def get_metadata(cls) -> PluginMetadata:
            return meta

        cls.get_metadata = get_metadata
        return cls

    return decorator
