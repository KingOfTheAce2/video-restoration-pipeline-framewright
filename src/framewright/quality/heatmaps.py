"""Quality heatmap generation for visual analysis.

Creates visual representations of quality metrics across frames and regions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)


class HeatmapType(Enum):
    """Types of quality heatmaps."""
    SSIM = "ssim"           # Structural similarity
    PSNR = "psnr"           # Peak signal-to-noise ratio
    DIFFERENCE = "diff"     # Absolute difference
    MOTION = "motion"       # Motion magnitude
    BLOCKING = "blocking"   # Block artifact detection
    NOISE = "noise"         # Noise level
    SHARPNESS = "sharpness" # Local sharpness
    TEMPORAL = "temporal"   # Frame-to-frame consistency


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    # General
    heatmap_type: HeatmapType = HeatmapType.SSIM
    colormap: str = "jet"  # jet, viridis, plasma, magma, inferno, hot

    # Block settings
    block_size: int = 16  # Size of analysis blocks

    # Output
    overlay_original: bool = True
    overlay_alpha: float = 0.5
    output_resolution: Optional[Tuple[int, int]] = None

    # Thresholds
    min_value: float = 0.0
    max_value: float = 1.0
    auto_scale: bool = True

    # Temporal
    temporal_window: int = 5  # Frames to consider for temporal metrics


@dataclass
class RegionQuality:
    """Quality metrics for a region of a frame."""
    x: int
    y: int
    width: int
    height: int
    ssim: float = 0.0
    psnr: float = 0.0
    difference: float = 0.0
    sharpness: float = 0.0
    noise: float = 0.0


@dataclass
class FrameHeatmap:
    """Heatmap data for a single frame."""
    frame_number: int
    heatmap_type: HeatmapType
    image_path: Optional[Path] = None
    regions: List[RegionQuality] = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 1.0
    mean_value: float = 0.0


class QualityHeatmapGenerator:
    """Generates visual quality heatmaps."""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        self.config = config or HeatmapConfig()
        self._ffmpeg_path = shutil.which("ffmpeg")
        self._numpy_available = self._check_numpy()

    def _check_numpy(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def generate_comparison_heatmap(
        self,
        reference: Path,
        distorted: Path,
        output_path: Path,
        frame_number: int = 0,
    ) -> FrameHeatmap:
        """Generate heatmap comparing reference and distorted frames."""
        if not self._numpy_available:
            return self._generate_ffmpeg_heatmap(reference, distorted, output_path, frame_number)

        import numpy as np

        # Extract frames
        ref_frame = self._extract_frame(reference, frame_number)
        dist_frame = self._extract_frame(distorted, frame_number)

        if ref_frame is None or dist_frame is None:
            raise RuntimeError("Failed to extract frames")

        # Generate heatmap based on type
        if self.config.heatmap_type == HeatmapType.SSIM:
            heatmap, regions = self._calculate_ssim_heatmap(ref_frame, dist_frame)
        elif self.config.heatmap_type == HeatmapType.PSNR:
            heatmap, regions = self._calculate_psnr_heatmap(ref_frame, dist_frame)
        elif self.config.heatmap_type == HeatmapType.DIFFERENCE:
            heatmap, regions = self._calculate_difference_heatmap(ref_frame, dist_frame)
        elif self.config.heatmap_type == HeatmapType.SHARPNESS:
            heatmap, regions = self._calculate_sharpness_heatmap(dist_frame)
        elif self.config.heatmap_type == HeatmapType.NOISE:
            heatmap, regions = self._calculate_noise_heatmap(dist_frame)
        else:
            heatmap, regions = self._calculate_difference_heatmap(ref_frame, dist_frame)

        # Apply colormap
        colored_heatmap = self._apply_colormap(heatmap)

        # Overlay on original if configured
        if self.config.overlay_original:
            output_image = self._overlay_heatmap(dist_frame, colored_heatmap)
        else:
            output_image = colored_heatmap

        # Save output
        self._save_image(output_image, output_path)

        return FrameHeatmap(
            frame_number=frame_number,
            heatmap_type=self.config.heatmap_type,
            image_path=output_path,
            regions=regions,
            min_value=float(np.min(heatmap)),
            max_value=float(np.max(heatmap)),
            mean_value=float(np.mean(heatmap)),
        )

    def _extract_frame(self, video_path: Path, frame_number: int) -> Optional[Any]:
        """Extract a frame from video."""
        if not self._ffmpeg_path:
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cmd = [
                self._ffmpeg_path,
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{frame_number})",
                "-vframes", "1",
                "-y",
                str(temp_path)
            ]

            subprocess.run(cmd, capture_output=True, timeout=60)

            if temp_path.exists():
                import cv2
                frame = cv2.imread(str(temp_path))
                return frame

        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")

        finally:
            temp_path.unlink(missing_ok=True)

        return None

    def _calculate_ssim_heatmap(self, ref: Any, dist: Any) -> Tuple[Any, List[RegionQuality]]:
        """Calculate SSIM heatmap."""
        import numpy as np

        # Convert to grayscale
        ref_gray = np.mean(ref, axis=2) if len(ref.shape) == 3 else ref
        dist_gray = np.mean(dist, axis=2) if len(dist.shape) == 3 else dist

        height, width = ref_gray.shape
        block_size = self.config.block_size

        heatmap = np.zeros((height, width), dtype=np.float32)
        regions = []

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                ref_block = ref_gray[y:y + block_size, x:x + block_size]
                dist_block = dist_gray[y:y + block_size, x:x + block_size]

                ssim = self._compute_ssim_block(ref_block, dist_block)
                heatmap[y:y + block_size, x:x + block_size] = ssim

                regions.append(RegionQuality(
                    x=x, y=y, width=block_size, height=block_size, ssim=ssim
                ))

        return heatmap, regions

    def _compute_ssim_block(self, ref: Any, dist: Any) -> float:
        """Compute SSIM for a block."""
        import numpy as np

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ref = ref.astype(np.float64)
        dist = dist.astype(np.float64)

        mu_ref = np.mean(ref)
        mu_dist = np.mean(dist)
        sigma_ref = np.var(ref)
        sigma_dist = np.var(dist)
        sigma_ref_dist = np.mean((ref - mu_ref) * (dist - mu_dist))

        numerator = (2 * mu_ref * mu_dist + C1) * (2 * sigma_ref_dist + C2)
        denominator = (mu_ref ** 2 + mu_dist ** 2 + C1) * (sigma_ref + sigma_dist + C2)

        return numerator / denominator if denominator != 0 else 0

    def _calculate_psnr_heatmap(self, ref: Any, dist: Any) -> Tuple[Any, List[RegionQuality]]:
        """Calculate PSNR heatmap."""
        import numpy as np

        # Convert to grayscale
        ref_gray = np.mean(ref, axis=2) if len(ref.shape) == 3 else ref
        dist_gray = np.mean(dist, axis=2) if len(dist.shape) == 3 else dist

        height, width = ref_gray.shape
        block_size = self.config.block_size

        heatmap = np.zeros((height, width), dtype=np.float32)
        regions = []

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                ref_block = ref_gray[y:y + block_size, x:x + block_size]
                dist_block = dist_gray[y:y + block_size, x:x + block_size]

                mse = np.mean((ref_block - dist_block) ** 2)
                psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 100
                psnr_normalized = min(1.0, psnr / 50)  # Normalize to 0-1

                heatmap[y:y + block_size, x:x + block_size] = psnr_normalized

                regions.append(RegionQuality(
                    x=x, y=y, width=block_size, height=block_size, psnr=psnr
                ))

        return heatmap, regions

    def _calculate_difference_heatmap(self, ref: Any, dist: Any) -> Tuple[Any, List[RegionQuality]]:
        """Calculate absolute difference heatmap."""
        import numpy as np

        # Convert to grayscale
        ref_gray = np.mean(ref, axis=2) if len(ref.shape) == 3 else ref
        dist_gray = np.mean(dist, axis=2) if len(dist.shape) == 3 else dist

        # Calculate absolute difference
        diff = np.abs(ref_gray.astype(np.float32) - dist_gray.astype(np.float32))
        heatmap = diff / 255.0  # Normalize to 0-1

        # Invert so high values = good quality
        heatmap = 1.0 - heatmap

        height, width = ref_gray.shape
        block_size = self.config.block_size
        regions = []

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block_diff = np.mean(diff[y:y + block_size, x:x + block_size])
                regions.append(RegionQuality(
                    x=x, y=y, width=block_size, height=block_size,
                    difference=float(block_diff)
                ))

        return heatmap, regions

    def _calculate_sharpness_heatmap(self, frame: Any) -> Tuple[Any, List[RegionQuality]]:
        """Calculate local sharpness heatmap."""
        import numpy as np

        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        height, width = gray.shape
        block_size = self.config.block_size

        heatmap = np.zeros((height, width), dtype=np.float32)
        regions = []

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = gray[y:y + block_size, x:x + block_size]

                # Calculate Laplacian variance (measure of sharpness)
                laplacian = np.array([
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                ], dtype=np.float32)

                # Simple convolution for Laplacian
                padded = np.pad(block, 1, mode='edge')
                lap_response = np.zeros_like(block, dtype=np.float32)

                for i in range(block_size):
                    for j in range(block_size):
                        window = padded[i:i + 3, j:j + 3]
                        lap_response[i, j] = np.sum(window * laplacian)

                sharpness = np.var(lap_response)
                normalized = min(1.0, sharpness / 1000)  # Normalize

                heatmap[y:y + block_size, x:x + block_size] = normalized

                regions.append(RegionQuality(
                    x=x, y=y, width=block_size, height=block_size,
                    sharpness=float(sharpness)
                ))

        return heatmap, regions

    def _calculate_noise_heatmap(self, frame: Any) -> Tuple[Any, List[RegionQuality]]:
        """Calculate noise level heatmap."""
        import numpy as np

        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        height, width = gray.shape
        block_size = self.config.block_size

        heatmap = np.zeros((height, width), dtype=np.float32)
        regions = []

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = gray[y:y + block_size, x:x + block_size].astype(np.float32)

                # Estimate noise as high-frequency content
                # Simple: variance of differences from smoothed version
                kernel_size = 3
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

                # Simple averaging
                smoothed = np.zeros_like(block)
                padded = np.pad(block, kernel_size // 2, mode='edge')

                for i in range(block_size):
                    for j in range(block_size):
                        window = padded[i:i + kernel_size, j:j + kernel_size]
                        smoothed[i, j] = np.mean(window)

                noise = np.std(block - smoothed)
                normalized = min(1.0, noise / 30)  # Normalize

                # Invert so low noise = high value (good)
                heatmap[y:y + block_size, x:x + block_size] = 1.0 - normalized

                regions.append(RegionQuality(
                    x=x, y=y, width=block_size, height=block_size,
                    noise=float(noise)
                ))

        return heatmap, regions

    def _apply_colormap(self, heatmap: Any) -> Any:
        """Apply colormap to heatmap."""
        import numpy as np

        # Normalize to 0-255
        if self.config.auto_scale:
            min_val = np.min(heatmap)
            max_val = np.max(heatmap)
            if max_val > min_val:
                normalized = (heatmap - min_val) / (max_val - min_val)
            else:
                normalized = heatmap
        else:
            normalized = np.clip(
                (heatmap - self.config.min_value) / (self.config.max_value - self.config.min_value),
                0, 1
            )

        # Apply colormap
        colored = self._jet_colormap(normalized)

        return colored

    def _jet_colormap(self, values: Any) -> Any:
        """Apply jet colormap."""
        import numpy as np

        # Jet colormap approximation
        r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)

        colored = np.stack([
            (b * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (r * 255).astype(np.uint8)
        ], axis=-1)

        return colored

    def _overlay_heatmap(self, original: Any, heatmap: Any) -> Any:
        """Overlay heatmap on original image."""
        import numpy as np

        alpha = self.config.overlay_alpha

        # Resize heatmap if needed
        if heatmap.shape[:2] != original.shape[:2]:
            # Simple resize
            heatmap = self._resize_image(heatmap, original.shape[:2])

        # Blend
        blended = (
            alpha * heatmap.astype(np.float32) +
            (1 - alpha) * original.astype(np.float32)
        )

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _resize_image(self, image: Any, target_size: Tuple[int, int]) -> Any:
        """Simple image resize."""
        import numpy as np

        src_h, src_w = image.shape[:2]
        dst_h, dst_w = target_size

        # Create coordinate mapping
        y_ratio = src_h / dst_h
        x_ratio = src_w / dst_w

        result = np.zeros((dst_h, dst_w, image.shape[2]), dtype=image.dtype)

        for y in range(dst_h):
            for x in range(dst_w):
                src_y = min(int(y * y_ratio), src_h - 1)
                src_x = min(int(x * x_ratio), src_w - 1)
                result[y, x] = image[src_y, src_x]

        return result

    def _save_image(self, image: Any, path: Path) -> None:
        """Save image to file."""
        try:
            import cv2
            cv2.imwrite(str(path), image)
        except ImportError:
            # Fallback: save as raw PPM
            import numpy as np
            height, width = image.shape[:2]
            with open(path, 'wb') as f:
                f.write(f"P6\n{width} {height}\n255\n".encode())
                # Convert BGR to RGB
                rgb = image[:, :, ::-1]
                f.write(rgb.tobytes())

    def _generate_ffmpeg_heatmap(
        self,
        reference: Path,
        distorted: Path,
        output_path: Path,
        frame_number: int,
    ) -> FrameHeatmap:
        """Generate heatmap using FFmpeg (fallback)."""
        if not self._ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

        # Use FFmpeg blend filter for difference
        cmd = [
            self._ffmpeg_path,
            "-i", str(distorted),
            "-i", str(reference),
            "-filter_complex",
            f"[0:v]select=eq(n\\,{frame_number})[a];"
            f"[1:v]select=eq(n\\,{frame_number})[b];"
            f"[a][b]blend=difference,eq=contrast=2:brightness=0.5,"
            f"colorize=hue=240:saturation=1",
            "-frames:v", "1",
            "-y",
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, timeout=120)

        return FrameHeatmap(
            frame_number=frame_number,
            heatmap_type=self.config.heatmap_type,
            image_path=output_path,
        )

    def generate_video_heatmap(
        self,
        reference: Path,
        distorted: Path,
        output_path: Path,
        frame_step: int = 1,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Generate heatmap video showing quality changes over time."""
        if not self._ffmpeg_path:
            raise RuntimeError("FFmpeg not found")

        # Build filter for continuous heatmap
        filter_str = (
            f"[0:v][1:v]blend=difference,"
            f"colorchannelmixer=rr=0:rg=1:rb=0:gr=0:gg=1:gb=0:br=0:bg=1:bb=0"
        )

        cmd = [
            self._ffmpeg_path,
            "-i", str(distorted),
            "-i", str(reference),
            "-filter_complex", filter_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-y",
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, timeout=7200)

        return output_path
