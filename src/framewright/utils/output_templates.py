"""
Output Templates - Filename templates with variables.

Provides flexible output naming with support for variables like
date, resolution, preset, and custom tokens.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from string import Template


@dataclass
class VideoMetadata:
    """Metadata extracted from video for template variables."""
    filename: str
    stem: str  # filename without extension
    extension: str
    width: int = 0
    height: int = 0
    fps: float = 0
    duration_seconds: float = 0
    codec: str = ""
    bitrate_kbps: int = 0

    @classmethod
    def from_path(cls, path: Path) -> "VideoMetadata":
        """Extract metadata from video file."""
        import cv2

        path = Path(path)
        meta = cls(
            filename=path.name,
            stem=path.stem,
            extension=path.suffix.lstrip('.')
        )

        try:
            cap = cv2.VideoCapture(str(path))
            meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta.fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if meta.fps > 0:
                meta.duration_seconds = frame_count / meta.fps
            meta.codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
        except Exception:
            pass

        return meta


@dataclass
class ProcessingContext:
    """Context information for template variables."""
    preset: str = "balanced"
    upscale_factor: int = 1
    denoise_strength: float = 0.0
    models_used: List[str] = field(default_factory=list)
    custom_vars: Dict[str, str] = field(default_factory=dict)


class OutputTemplate:
    """
    Template engine for output filenames.

    Supports variables like:
    - {name} - Original filename without extension
    - {ext} - Original extension
    - {date} - Current date (YYYY-MM-DD)
    - {time} - Current time (HH-MM-SS)
    - {datetime} - Full datetime
    - {width}, {height} - Video dimensions
    - {resolution} - Resolution label (480p, 720p, 1080p, 4K, etc.)
    - {fps} - Frame rate
    - {preset} - Processing preset name
    - {upscale} - Upscale factor (e.g., "2x")
    - {counter} - Auto-incrementing counter
    - {hash} - Short hash of input file
    - Custom variables via context
    """

    # Resolution labels based on height
    RESOLUTION_LABELS = {
        480: "480p",
        576: "576p",
        720: "720p",
        1080: "1080p",
        1440: "1440p",
        2160: "4K",
        4320: "8K"
    }

    # Default templates for common use cases
    PRESETS = {
        "simple": "{name}_restored.{ext}",
        "dated": "{name}_{date}_restored.{ext}",
        "detailed": "{name}_{preset}_{resolution}_{date}.{ext}",
        "archive": "{date}/{name}_{preset}_{upscale}.{ext}",
        "versioned": "{name}_v{counter:03d}.{ext}",
        "organized": "{resolution}/{preset}/{name}_restored.{ext}"
    }

    def __init__(
        self,
        template: str = "{name}_restored.{ext}",
        output_dir: Optional[Path] = None,
        counter_start: int = 1
    ):
        """
        Initialize template.

        Args:
            template: Template string with {variables}
            output_dir: Base output directory (template can add subdirs)
            counter_start: Starting value for {counter}
        """
        self.template = template
        self.output_dir = Path(output_dir) if output_dir else None
        self._counter = counter_start

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        output_dir: Optional[Path] = None
    ) -> "OutputTemplate":
        """Create template from preset name."""
        template = cls.PRESETS.get(preset_name, cls.PRESETS["simple"])
        return cls(template=template, output_dir=output_dir)

    def _get_resolution_label(self, height: int) -> str:
        """Get resolution label for video height."""
        # Find closest match
        for res_height, label in sorted(self.RESOLUTION_LABELS.items()):
            if height <= res_height:
                return label
        return f"{height}p"

    def _compute_hash(self, path: Path, length: int = 8) -> str:
        """Compute short hash of file."""
        import hashlib

        try:
            with open(path, 'rb') as f:
                # Read first and last 64KB for speed
                data = f.read(65536)
                f.seek(-65536, 2)
                data += f.read(65536)
            return hashlib.md5(data).hexdigest()[:length]
        except Exception:
            return "00000000"[:length]

    def _build_variables(
        self,
        input_path: Path,
        video_meta: Optional[VideoMetadata] = None,
        context: Optional[ProcessingContext] = None
    ) -> Dict[str, str]:
        """Build all template variables."""
        input_path = Path(input_path)

        if video_meta is None:
            video_meta = VideoMetadata.from_path(input_path)

        if context is None:
            context = ProcessingContext()

        now = datetime.now()

        # Calculate output resolution (after upscaling)
        out_height = video_meta.height * context.upscale_factor
        out_width = video_meta.width * context.upscale_factor

        variables = {
            # File info
            "name": video_meta.stem,
            "filename": video_meta.filename,
            "ext": video_meta.extension or "mp4",
            "hash": self._compute_hash(input_path),

            # Date/time
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H-%M-%S"),
            "datetime": now.strftime("%Y-%m-%d_%H-%M-%S"),
            "year": now.strftime("%Y"),
            "month": now.strftime("%m"),
            "day": now.strftime("%d"),

            # Video properties (original)
            "width": str(video_meta.width),
            "height": str(video_meta.height),
            "resolution": self._get_resolution_label(video_meta.height),
            "fps": f"{video_meta.fps:.2f}".rstrip('0').rstrip('.'),
            "duration": str(int(video_meta.duration_seconds)),

            # Output properties
            "out_width": str(out_width),
            "out_height": str(out_height),
            "out_resolution": self._get_resolution_label(out_height),

            # Processing info
            "preset": context.preset,
            "upscale": f"{context.upscale_factor}x" if context.upscale_factor > 1 else "1x",
            "upscale_factor": str(context.upscale_factor),
            "denoise": f"{context.denoise_strength:.1f}",
            "models": "_".join(context.models_used) if context.models_used else "none",

            # Counter
            "counter": str(self._counter),
            "counter:02d": f"{self._counter:02d}",
            "counter:03d": f"{self._counter:03d}",
            "counter:04d": f"{self._counter:04d}",
        }

        # Add custom variables
        variables.update(context.custom_vars)

        return variables

    def format(
        self,
        input_path: Path,
        video_meta: Optional[VideoMetadata] = None,
        context: Optional[ProcessingContext] = None,
        increment_counter: bool = True
    ) -> Path:
        """
        Generate output path from template.

        Args:
            input_path: Path to input video
            video_meta: Pre-extracted video metadata
            context: Processing context
            increment_counter: Whether to increment counter after

        Returns:
            Complete output path
        """
        variables = self._build_variables(input_path, video_meta, context)

        # Handle counter formatting
        template = self.template
        for fmt in ["counter:04d", "counter:03d", "counter:02d"]:
            template = template.replace("{" + fmt + "}", variables[fmt])

        # Format the template
        try:
            output_name = template.format(**variables)
        except KeyError as e:
            # Unknown variable - leave as-is
            output_name = template
            for var, value in variables.items():
                output_name = output_name.replace("{" + var + "}", value)

        # Clean up filename
        output_name = self._sanitize_filename(output_name)

        # Build full path
        if self.output_dir:
            output_path = self.output_dir / output_name
        else:
            output_path = Path(input_path).parent / output_name

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if increment_counter:
            self._counter += 1

        return output_path

    def _sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename."""
        # Replace invalid characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Handle path separators in subdirectories
        parts = filename.split('/')
        parts = [self._sanitize_part(p) for p in parts]

        return '/'.join(parts)

    def _sanitize_part(self, part: str) -> str:
        """Sanitize a single path component."""
        # Remove leading/trailing whitespace and dots
        part = part.strip().strip('.')

        # Replace multiple underscores/dashes
        part = re.sub(r'[_\-]{2,}', '_', part)

        return part or "unnamed"

    def preview(
        self,
        input_path: Path,
        video_meta: Optional[VideoMetadata] = None,
        context: Optional[ProcessingContext] = None
    ) -> str:
        """
        Preview what the output path would be without incrementing counter.

        Returns the formatted path as a string.
        """
        return str(self.format(
            input_path,
            video_meta,
            context,
            increment_counter=False
        ))

    def get_unique_path(
        self,
        input_path: Path,
        video_meta: Optional[VideoMetadata] = None,
        context: Optional[ProcessingContext] = None
    ) -> Path:
        """
        Generate a unique output path, avoiding overwrites.

        If the formatted path exists, appends _1, _2, etc.
        """
        base_path = self.format(input_path, video_meta, context, increment_counter=False)

        if not base_path.exists():
            self._counter += 1
            return base_path

        # Find unique name
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                self._counter += 1
                return new_path
            counter += 1
            if counter > 9999:
                raise ValueError("Could not find unique filename")


class TemplateManager:
    """
    Manage multiple output templates and presets.
    """

    def __init__(self):
        self._templates: Dict[str, OutputTemplate] = {}
        self._default_template = "simple"

        # Register built-in presets
        for name in OutputTemplate.PRESETS:
            self._templates[name] = OutputTemplate.from_preset(name)

    def register(
        self,
        name: str,
        template: str,
        output_dir: Optional[Path] = None
    ) -> None:
        """Register a custom template."""
        self._templates[name] = OutputTemplate(
            template=template,
            output_dir=output_dir
        )

    def get(self, name: str) -> OutputTemplate:
        """Get a template by name."""
        if name not in self._templates:
            # Try to create from string
            return OutputTemplate(template=name)
        return self._templates[name]

    def set_default(self, name: str) -> None:
        """Set the default template."""
        if name in self._templates or name in OutputTemplate.PRESETS:
            self._default_template = name

    def format(
        self,
        input_path: Path,
        template_name: Optional[str] = None,
        **kwargs
    ) -> Path:
        """Format using a named template or the default."""
        name = template_name or self._default_template
        template = self.get(name)
        return template.format(input_path, **kwargs)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(set(list(self._templates.keys()) + list(OutputTemplate.PRESETS.keys())))

    def get_template_string(self, name: str) -> str:
        """Get the template string for a preset."""
        if name in OutputTemplate.PRESETS:
            return OutputTemplate.PRESETS[name]
        if name in self._templates:
            return self._templates[name].template
        return name  # Assume it's a raw template string
