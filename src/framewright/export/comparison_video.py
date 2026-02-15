"""Before/after comparison video export."""

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ComparisonLayout(Enum):
    """Layout styles for comparison videos."""
    SIDE_BY_SIDE = "side_by_side"  # Original | Restored
    TOP_BOTTOM = "top_bottom"  # Original on top
    SPLIT_DIAGONAL = "split_diagonal"  # Diagonal split
    SPLIT_VERTICAL = "split_vertical"  # Vertical split line
    SPLIT_HORIZONTAL = "split_horizontal"  # Horizontal split line
    WIPE = "wipe"  # Animated wipe transition
    SLIDER = "slider"  # Interactive slider (for web export)
    QUAD = "quad"  # 2x2 grid for multiple variants
    FLICKER = "flicker"  # Alternating frames


@dataclass
class ComparisonConfig:
    """Configuration for comparison video export."""
    layout: ComparisonLayout = ComparisonLayout.SIDE_BY_SIDE
    output_path: Optional[Path] = None

    # Labels
    show_labels: bool = True
    original_label: str = "Original"
    restored_label: str = "Restored"
    label_font_size: int = 32
    label_position: str = "top"  # top, bottom, corner

    # Split settings
    split_position: float = 0.5  # 0.0-1.0
    split_line_width: int = 2
    split_line_color: str = "white"

    # Wipe settings
    wipe_duration: float = 3.0  # Seconds per wipe cycle
    wipe_direction: str = "left_to_right"

    # Output settings
    width: Optional[int] = None  # None = auto
    height: Optional[int] = None
    fps: float = 30.0
    crf: int = 18
    codec: str = "libx264"

    # Quality metrics overlay
    show_metrics: bool = False
    metrics_position: str = "bottom_right"

    # Border between videos
    border_width: int = 4
    border_color: str = "black"


class ComparisonExporter:
    """Export before/after comparison videos."""

    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()

    def export(
        self,
        original_path: Path,
        restored_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Export a comparison video.

        Args:
            original_path: Path to original video
            restored_path: Path to restored video
            output_path: Output path (optional, auto-generated if not provided)

        Returns:
            Path to output comparison video
        """
        output_path = output_path or self.config.output_path
        if output_path is None:
            output_path = restored_path.parent / f"{restored_path.stem}_comparison.mp4"

        layout = self.config.layout

        if layout == ComparisonLayout.SIDE_BY_SIDE:
            return self._export_side_by_side(original_path, restored_path, output_path)
        elif layout == ComparisonLayout.TOP_BOTTOM:
            return self._export_top_bottom(original_path, restored_path, output_path)
        elif layout == ComparisonLayout.SPLIT_VERTICAL:
            return self._export_split(original_path, restored_path, output_path, "vertical")
        elif layout == ComparisonLayout.SPLIT_HORIZONTAL:
            return self._export_split(original_path, restored_path, output_path, "horizontal")
        elif layout == ComparisonLayout.WIPE:
            return self._export_wipe(original_path, restored_path, output_path)
        elif layout == ComparisonLayout.FLICKER:
            return self._export_flicker(original_path, restored_path, output_path)
        else:
            return self._export_side_by_side(original_path, restored_path, output_path)

    def _export_side_by_side(
        self,
        original: Path,
        restored: Path,
        output: Path,
    ) -> Path:
        """Export side-by-side comparison."""
        filter_complex = []

        # Scale both videos to same height
        filter_complex.append("[0:v]scale=-1:720[left]")
        filter_complex.append("[1:v]scale=-1:720[right]")

        # Add labels if configured
        if self.config.show_labels:
            filter_complex.append(
                f"[left]drawtext=text='{self.config.original_label}':"
                f"fontsize={self.config.label_font_size}:fontcolor=white:"
                f"x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[left_labeled]"
            )
            filter_complex.append(
                f"[right]drawtext=text='{self.config.restored_label}':"
                f"fontsize={self.config.label_font_size}:fontcolor=white:"
                f"x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=5[right_labeled]"
            )
            filter_complex.append("[left_labeled][right_labeled]hstack=inputs=2[out]")
        else:
            filter_complex.append("[left][right]hstack=inputs=2[out]")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(restored),
            "-filter_complex", ";".join(filter_complex),
            "-map", "[out]",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            "-preset", "medium",
            str(output)
        ]

        self._run_ffmpeg(cmd)
        logger.info(f"Created side-by-side comparison: {output}")
        return output

    def _export_top_bottom(
        self,
        original: Path,
        restored: Path,
        output: Path,
    ) -> Path:
        """Export top-bottom comparison."""
        filter_complex = []

        # Scale both videos to same width
        filter_complex.append("[0:v]scale=1280:-1[top]")
        filter_complex.append("[1:v]scale=1280:-1[bottom]")

        if self.config.show_labels:
            filter_complex.append(
                f"[top]drawtext=text='{self.config.original_label}':"
                f"fontsize={self.config.label_font_size}:fontcolor=white:"
                f"x=10:y=10:box=1:boxcolor=black@0.5[top_labeled]"
            )
            filter_complex.append(
                f"[bottom]drawtext=text='{self.config.restored_label}':"
                f"fontsize={self.config.label_font_size}:fontcolor=white:"
                f"x=10:y=10:box=1:boxcolor=black@0.5[bottom_labeled]"
            )
            filter_complex.append("[top_labeled][bottom_labeled]vstack=inputs=2[out]")
        else:
            filter_complex.append("[top][bottom]vstack=inputs=2[out]")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(restored),
            "-filter_complex", ";".join(filter_complex),
            "-map", "[out]",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            str(output)
        ]

        self._run_ffmpeg(cmd)
        logger.info(f"Created top-bottom comparison: {output}")
        return output

    def _export_split(
        self,
        original: Path,
        restored: Path,
        output: Path,
        direction: str,
    ) -> Path:
        """Export split comparison with visible divider line."""
        split_pos = self.config.split_position

        if direction == "vertical":
            # Vertical split - left is original, right is restored
            filter_complex = (
                f"[0:v]scale=1920:1080[orig];"
                f"[1:v]scale=1920:1080[rest];"
                f"[orig]crop=iw*{split_pos}:ih:0:0[left];"
                f"[rest]crop=iw*{1-split_pos}:ih:iw*{split_pos}:0[right];"
                f"[left][right]hstack=inputs=2,"
                f"drawbox=x={int(1920*split_pos)-1}:y=0:w=2:h=ih:c={self.config.split_line_color}[out]"
            )
        else:
            # Horizontal split
            filter_complex = (
                f"[0:v]scale=1920:1080[orig];"
                f"[1:v]scale=1920:1080[rest];"
                f"[orig]crop=iw:ih*{split_pos}:0:0[top];"
                f"[rest]crop=iw:ih*{1-split_pos}:0:ih*{split_pos}[bottom];"
                f"[top][bottom]vstack=inputs=2,"
                f"drawbox=x=0:y={int(1080*split_pos)-1}:w=iw:h=2:c={self.config.split_line_color}[out]"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(restored),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            str(output)
        ]

        self._run_ffmpeg(cmd)
        logger.info(f"Created split comparison: {output}")
        return output

    def _export_wipe(
        self,
        original: Path,
        restored: Path,
        output: Path,
    ) -> Path:
        """Export animated wipe comparison."""
        # Get video duration
        duration = self._get_duration(original)
        wipe_duration = self.config.wipe_duration

        # Create expression for animated split position
        # Oscillates between 0 and 1
        expr = f"abs(sin(t*PI/{wipe_duration}))"

        filter_complex = (
            f"[0:v]scale=1920:1080[orig];"
            f"[1:v]scale=1920:1080[rest];"
            f"[orig][rest]blend=all_expr='if(X/W<{expr},A,B)'[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(restored),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            str(output)
        ]

        self._run_ffmpeg(cmd)
        logger.info(f"Created wipe comparison: {output}")
        return output

    def _export_flicker(
        self,
        original: Path,
        restored: Path,
        output: Path,
    ) -> Path:
        """Export flicker comparison (alternating frames)."""
        filter_complex = (
            "[0:v]scale=1920:1080[orig];"
            "[1:v]scale=1920:1080[rest];"
            "[orig][rest]interleave=2[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(restored),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            "-r", str(self.config.fps),
            str(output)
        ]

        self._run_ffmpeg(cmd)
        logger.info(f"Created flicker comparison: {output}")
        return output

    def export_multi_variant(
        self,
        original_path: Path,
        variant_paths: List[Tuple[str, Path]],
        output_path: Path,
    ) -> Path:
        """Export comparison with multiple restoration variants.

        Args:
            original_path: Path to original video
            variant_paths: List of (label, path) tuples
            output_path: Output path

        Returns:
            Path to output video
        """
        n_variants = len(variant_paths) + 1  # +1 for original

        # Determine grid size
        if n_variants <= 2:
            cols, rows = 2, 1
        elif n_variants <= 4:
            cols, rows = 2, 2
        elif n_variants <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 3, 3

        # Build FFmpeg command
        inputs = ["-i", str(original_path)]
        for _, path in variant_paths:
            inputs.extend(["-i", str(path)])

        # Build filter for grid layout
        filter_parts = []
        labeled = []

        # Scale and label each input
        for i, (label, _) in enumerate([("Original", original_path)] + list(variant_paths)):
            filter_parts.append(f"[{i}:v]scale=640:360[s{i}]")
            if self.config.show_labels:
                filter_parts.append(
                    f"[s{i}]drawtext=text='{label}':"
                    f"fontsize=24:fontcolor=white:x=10:y=10:"
                    f"box=1:boxcolor=black@0.5[l{i}]"
                )
                labeled.append(f"[l{i}]")
            else:
                labeled.append(f"[s{i}]")

        # Pad to fill grid
        while len(labeled) < cols * rows:
            idx = len(labeled)
            filter_parts.append(f"color=black:s=640x360[pad{idx}]")
            labeled.append(f"[pad{idx}]")

        # Build grid
        row_outputs = []
        for r in range(rows):
            row_inputs = "".join(labeled[r*cols:(r+1)*cols])
            filter_parts.append(f"{row_inputs}hstack=inputs={cols}[row{r}]")
            row_outputs.append(f"[row{r}]")

        filter_parts.append(f"{''.join(row_outputs)}vstack=inputs={rows}[out]")

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]",
            "-c:v", self.config.codec,
            "-crf", str(self.config.crf),
            str(output_path)
        ]

        self._run_ffmpeg(cmd)
        logger.info(f"Created multi-variant comparison: {output_path}")
        return output_path

    def _get_duration(self, path: Path) -> float:
        """Get video duration in seconds."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except:
            return 30.0  # Default

    def _run_ffmpeg(self, cmd: List[str]) -> None:
        """Run FFmpeg command."""
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")


# Convenience functions
def create_side_by_side(
    original: Path,
    restored: Path,
    output: Optional[Path] = None,
    labels: bool = True,
) -> Path:
    """Create a side-by-side comparison video."""
    config = ComparisonConfig(
        layout=ComparisonLayout.SIDE_BY_SIDE,
        show_labels=labels,
    )
    exporter = ComparisonExporter(config)
    return exporter.export(original, restored, output)


def create_split_wipe(
    original: Path,
    restored: Path,
    output: Optional[Path] = None,
    wipe_duration: float = 3.0,
) -> Path:
    """Create an animated wipe comparison video."""
    config = ComparisonConfig(
        layout=ComparisonLayout.WIPE,
        wipe_duration=wipe_duration,
    )
    exporter = ComparisonExporter(config)
    return exporter.export(original, restored, output)


def create_slider_comparison(
    original: Path,
    restored: Path,
    output: Optional[Path] = None,
    split_position: float = 0.5,
) -> Path:
    """Create a vertical split comparison (static slider position)."""
    config = ComparisonConfig(
        layout=ComparisonLayout.SPLIT_VERTICAL,
        split_position=split_position,
        show_labels=True,
    )
    exporter = ComparisonExporter(config)
    return exporter.export(original, restored, output)
