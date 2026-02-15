"""Thumbnail grid generator for visual comparison."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for thumbnail grid."""
    columns: int = 4
    rows: int = 4
    thumb_width: int = 320
    thumb_height: int = 180
    padding: int = 4
    background_color: str = "black"
    show_timestamps: bool = True
    timestamp_format: str = "%H:%M:%S"
    font_size: int = 14
    font_color: str = "white"


class ThumbnailGridGenerator:
    """Generate thumbnail grid images from videos."""

    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()

    def generate(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Path:
        """Generate thumbnail grid from video.

        Args:
            video_path: Path to video file
            output_path: Output image path (default: video_thumbnails.jpg)
            start_time: Start time in seconds
            end_time: End time in seconds (default: video duration)

        Returns:
            Path to generated grid image
        """
        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_thumbnails.jpg"
        output_path = Path(output_path)

        # Get video duration
        duration = self._get_duration(video_path)
        if end_time is None:
            end_time = duration

        # Calculate number of thumbnails
        total_thumbs = self.config.columns * self.config.rows
        interval = (end_time - start_time) / total_thumbs

        # Generate grid using FFmpeg
        cfg = self.config
        filter_complex = (
            f"select='not(mod(n,{int(interval * 30)}))',"  # Sample frames
            f"scale={cfg.thumb_width}:{cfg.thumb_height},"
            f"tile={cfg.columns}x{cfg.rows}:padding={cfg.padding}:color={cfg.background_color}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(end_time - start_time),
            "-vf", filter_complex,
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning(f"FFmpeg grid failed, trying alternative method")
                return self._generate_manual(video_path, output_path, start_time, end_time)

            logger.info(f"Generated thumbnail grid: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate grid: {e}")
            raise

    def generate_comparison_grid(
        self,
        original_path: Path,
        restored_path: Path,
        output_path: Optional[Path] = None,
        num_samples: int = 8,
    ) -> Path:
        """Generate side-by-side comparison thumbnail grid.

        Args:
            original_path: Path to original video
            restored_path: Path to restored video
            output_path: Output image path
            num_samples: Number of comparison pairs

        Returns:
            Path to generated comparison image
        """
        original_path = Path(original_path)
        restored_path = Path(restored_path)

        if output_path is None:
            output_path = restored_path.parent / f"{restored_path.stem}_comparison_grid.jpg"
        output_path = Path(output_path)

        duration = self._get_duration(original_path)
        interval = duration / num_samples

        cfg = self.config
        thumb_w = cfg.thumb_width
        thumb_h = cfg.thumb_height

        # Extract thumbnails from both videos and create side-by-side pairs
        filter_complex = []

        # Scale both inputs
        filter_complex.append(f"[0:v]fps=1/{interval},scale={thumb_w}:{thumb_h}[orig]")
        filter_complex.append(f"[1:v]fps=1/{interval},scale={thumb_w}:{thumb_h}[rest]")

        # Stack horizontally for each pair
        filter_complex.append("[orig][rest]hstack=inputs=2[pairs]")

        # Tile into grid
        rows = math.ceil(num_samples / 2)
        filter_complex.append(f"[pairs]tile=2x{rows}:padding={cfg.padding}[out]")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(original_path),
            "-i", str(restored_path),
            "-filter_complex", ";".join(filter_complex),
            "-map", "[out]",
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError("Failed to generate comparison grid")

            logger.info(f"Generated comparison grid: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate comparison grid: {e}")
            raise

    def _generate_manual(
        self,
        video_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        """Generate grid manually by extracting individual frames."""
        try:
            import cv2
            from PIL import Image
        except ImportError:
            raise RuntimeError("OpenCV and Pillow required for manual grid generation")

        cfg = self.config
        total_thumbs = cfg.columns * cfg.rows
        interval = (end_time - start_time) / total_thumbs

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        thumbnails = []
        for i in range(total_thumbs):
            time_pos = start_time + (i * interval)
            frame_pos = int(time_pos * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if ret:
                # Resize
                frame = cv2.resize(frame, (cfg.thumb_width, cfg.thumb_height))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thumbnails.append(Image.fromarray(frame))
            else:
                # Create placeholder
                thumbnails.append(
                    Image.new('RGB', (cfg.thumb_width, cfg.thumb_height), color='gray')
                )

        cap.release()

        # Create grid image
        grid_width = cfg.columns * (cfg.thumb_width + cfg.padding) - cfg.padding
        grid_height = cfg.rows * (cfg.thumb_height + cfg.padding) - cfg.padding
        grid = Image.new('RGB', (grid_width, grid_height), color=cfg.background_color)

        for i, thumb in enumerate(thumbnails):
            row = i // cfg.columns
            col = i % cfg.columns
            x = col * (cfg.thumb_width + cfg.padding)
            y = row * (cfg.thumb_height + cfg.padding)
            grid.paste(thumb, (x, y))

        grid.save(output_path, quality=95)
        logger.info(f"Generated thumbnail grid (manual): {output_path}")
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
            return 60.0  # Default
