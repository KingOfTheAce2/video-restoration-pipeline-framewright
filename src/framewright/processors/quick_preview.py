"""Quick Preview Generator for Fast Restoration Validation.

Generates a fast preview by sampling every Nth frame, allowing users
to validate restoration settings before processing the full video.

Features:
- Sample every Nth frame for speed
- Apply restoration pipeline to samples only
- Generate preview video or image grid
- Show estimated quality metrics
- Compare different presets quickly

Example:
    >>> preview = QuickPreviewGenerator()
    >>> result = preview.generate(
    ...     input_path="video.mp4",
    ...     every_n=100,  # Sample every 100 frames
    ...     preset="archive"
    ... )
    >>> print(f"Preview saved: {result.output_path}")
    >>> print(f"Processed {result.frames_sampled} of {result.total_frames} frames")
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class PreviewConfig:
    """Configuration for quick preview generation."""
    # Sampling
    every_n_frames: int = 100        # Sample every N frames
    max_frames: int = 50             # Maximum frames to sample

    # Output
    output_resolution: Tuple[int, int] = (854, 480)  # 480p preview
    output_fps: float = 2.0          # Slow playback for inspection

    # Processing
    preset: Optional[str] = None     # Restoration preset to apply
    apply_restoration: bool = True   # Actually apply restoration

    # Comparison
    side_by_side: bool = True        # Show original vs restored
    show_metrics: bool = True        # Overlay quality metrics


@dataclass
class PreviewFrame:
    """A single preview frame."""
    frame_number: int
    timestamp: float
    original: "np.ndarray"
    restored: Optional["np.ndarray"] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class QuickPreviewResult:
    """Result of quick preview generation."""
    output_path: Path
    total_frames: int
    frames_sampled: int
    sample_interval: int

    # Quality metrics (averaged)
    avg_psnr: float = 0.0
    avg_ssim: float = 0.0

    # Timing
    processing_time_seconds: float = 0.0
    estimated_full_time_seconds: float = 0.0

    # Preview frames (for grid generation)
    preview_frames: List[PreviewFrame] = field(default_factory=list)

    def coverage_percent(self) -> float:
        """Percentage of video covered by preview."""
        if self.total_frames == 0:
            return 0.0
        return (self.frames_sampled / self.total_frames) * 100


class QuickPreviewGenerator:
    """Generates quick previews for restoration validation.

    Samples frames at regular intervals and applies restoration
    to provide a fast preview of the final result.
    """

    def __init__(self, config: Optional[PreviewConfig] = None):
        """Initialize preview generator.

        Args:
            config: Preview configuration
        """
        self.config = config or PreviewConfig()

        if not HAS_CV2:
            logger.warning("OpenCV not available - preview generation limited")

    def generate(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        every_n: Optional[int] = None,
        preset: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> QuickPreviewResult:
        """Generate quick preview video.

        Args:
            input_path: Input video path
            output_path: Output preview path (auto-generated if None)
            every_n: Override sample interval
            preset: Restoration preset to apply
            progress_callback: Called with (current, total) frames

        Returns:
            QuickPreviewResult with preview info
        """
        import time

        input_path = Path(input_path)
        start_time = time.time()

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_preview.mp4"

        every_n = every_n or self.config.every_n_frames
        preset = preset or self.config.preset

        result = QuickPreviewResult(
            output_path=output_path,
            total_frames=0,
            frames_sampled=0,
            sample_interval=every_n,
        )

        if not HAS_CV2:
            logger.error("OpenCV required for preview generation")
            return result

        try:
            # Open video
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {input_path}")
                return result

            result.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate sample indices
            sample_indices = list(range(0, result.total_frames, every_n))
            if len(sample_indices) > self.config.max_frames:
                # Reduce to max_frames by increasing interval
                step = len(sample_indices) // self.config.max_frames
                sample_indices = sample_indices[::step]

            # Setup output video writer
            out_width, out_height = self.config.output_resolution
            if self.config.side_by_side:
                out_width *= 2  # Side by side doubles width

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.config.output_fps,
                (out_width, out_height)
            )

            # Load restoration pipeline if needed
            restorer = None
            if self.config.apply_restoration and preset:
                restorer = self._load_restorer(preset)

            # Process sample frames
            psnr_values = []
            ssim_values = []

            for idx, frame_num in enumerate(sample_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                timestamp = frame_num / original_fps if original_fps > 0 else 0

                # Resize original for preview
                original_resized = cv2.resize(
                    frame,
                    self.config.output_resolution,
                    interpolation=cv2.INTER_AREA
                )

                # Apply restoration if available
                if restorer is not None:
                    restored = self._apply_restoration(frame, restorer)
                    restored_resized = cv2.resize(
                        restored,
                        self.config.output_resolution,
                        interpolation=cv2.INTER_AREA
                    )

                    # Calculate metrics
                    metrics = self._calculate_metrics(original_resized, restored_resized)
                    if "psnr" in metrics:
                        psnr_values.append(metrics["psnr"])
                    if "ssim" in metrics:
                        ssim_values.append(metrics["ssim"])
                else:
                    restored_resized = original_resized
                    metrics = {}

                # Create output frame
                if self.config.side_by_side:
                    output_frame = np.hstack([original_resized, restored_resized])
                else:
                    output_frame = restored_resized

                # Add overlay text
                if self.config.show_metrics:
                    output_frame = self._add_overlay(
                        output_frame,
                        frame_num,
                        timestamp,
                        metrics,
                        self.config.side_by_side
                    )

                out.write(output_frame)
                result.frames_sampled += 1

                # Store preview frame
                preview_frame = PreviewFrame(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    original=original_resized,
                    restored=restored_resized if restorer else None,
                    metrics=metrics,
                )
                result.preview_frames.append(preview_frame)

                if progress_callback:
                    progress_callback(idx + 1, len(sample_indices))

            cap.release()
            out.release()

            # Calculate average metrics
            if psnr_values:
                result.avg_psnr = sum(psnr_values) / len(psnr_values)
            if ssim_values:
                result.avg_ssim = sum(ssim_values) / len(ssim_values)

            result.processing_time_seconds = time.time() - start_time

            # Estimate full processing time
            if result.frames_sampled > 0:
                time_per_frame = result.processing_time_seconds / result.frames_sampled
                result.estimated_full_time_seconds = time_per_frame * result.total_frames

            logger.info(
                f"Preview generated: {result.frames_sampled} frames sampled "
                f"({result.coverage_percent():.1f}% coverage)"
            )

            return result

        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            return result

    def generate_grid(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        num_frames: int = 9,
        cols: int = 3,
    ) -> Path:
        """Generate a grid image showing sample frames.

        Args:
            input_path: Input video path
            output_path: Output image path
            num_frames: Number of frames in grid
            cols: Number of columns

        Returns:
            Path to grid image
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_grid.jpg"

        if not HAS_CV2:
            logger.error("OpenCV required for grid generation")
            return output_path

        try:
            cap = cv2.VideoCapture(str(input_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame indices
            step = total_frames // (num_frames + 1)
            indices = [step * (i + 1) for i in range(num_frames)]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize for grid
                    frame = cv2.resize(frame, (320, 180))
                    frames.append(frame)

            cap.release()

            if not frames:
                return output_path

            # Build grid
            rows = (len(frames) + cols - 1) // cols

            # Pad with black frames if needed
            while len(frames) < rows * cols:
                frames.append(np.zeros_like(frames[0]))

            # Create grid
            grid_rows = []
            for r in range(rows):
                row_frames = frames[r * cols:(r + 1) * cols]
                grid_rows.append(np.hstack(row_frames))

            grid = np.vstack(grid_rows)

            cv2.imwrite(str(output_path), grid)
            logger.info(f"Grid saved: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Grid generation failed: {e}")
            return output_path

    def compare_presets(
        self,
        input_path: Path,
        presets: List[str],
        frame_index: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple presets on a single frame.

        Args:
            input_path: Input video path
            presets: List of preset names to compare
            frame_index: Frame to use (middle of video if None)
            output_dir: Directory for comparison images

        Returns:
            Dict mapping preset names to their results
        """
        input_path = Path(input_path)

        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_preset_compare"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_CV2:
            logger.error("OpenCV required for preset comparison")
            return {}

        try:
            # Get middle frame if not specified
            cap = cv2.VideoCapture(str(input_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_index is None:
                frame_index = total_frames // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, original = cap.read()
            cap.release()

            if not ret:
                return {}

            # Save original
            cv2.imwrite(str(output_dir / "original.jpg"), original)

            results = {}

            for preset in presets:
                try:
                    restorer = self._load_restorer(preset)
                    if restorer is None:
                        continue

                    restored = self._apply_restoration(original, restorer)
                    metrics = self._calculate_metrics(original, restored)

                    # Save result
                    output_path = output_dir / f"{preset}.jpg"
                    cv2.imwrite(str(output_path), restored)

                    results[preset] = {
                        "output_path": output_path,
                        "metrics": metrics,
                    }

                except Exception as e:
                    logger.warning(f"Preset {preset} failed: {e}")
                    results[preset] = {"error": str(e)}

            return results

        except Exception as e:
            logger.error(f"Preset comparison failed: {e}")
            return {}

    def _load_restorer(self, preset: str) -> Optional[Any]:
        """Load restoration pipeline for preset.

        Args:
            preset: Preset name

        Returns:
            Restorer object or None
        """
        try:
            from ..config import Config, PRESETS
            from ..restorer import VideoRestorer

            if preset in PRESETS:
                config = Config.from_preset(preset)
            else:
                config = Config()

            # Use lightweight settings for preview
            config.tile_size = 256  # Smaller tiles for speed

            return VideoRestorer(config)

        except Exception as e:
            logger.warning(f"Could not load restorer: {e}")
            return None

    def _apply_restoration(
        self,
        frame: "np.ndarray",
        restorer: Any,
    ) -> "np.ndarray":
        """Apply restoration to a single frame.

        Args:
            frame: Input frame (BGR)
            restorer: Restorer object

        Returns:
            Restored frame
        """
        try:
            # Try to use restorer's frame processing
            if hasattr(restorer, 'process_frame'):
                return restorer.process_frame(frame)
            elif hasattr(restorer, 'enhance_frame'):
                return restorer.enhance_frame(frame)
            else:
                # Fallback: return original
                return frame
        except Exception as e:
            logger.debug(f"Frame restoration failed: {e}")
            return frame

    def _calculate_metrics(
        self,
        original: "np.ndarray",
        restored: "np.ndarray",
    ) -> Dict[str, float]:
        """Calculate quality metrics between frames.

        Args:
            original: Original frame
            restored: Restored frame

        Returns:
            Dict with metric values
        """
        metrics = {}

        try:
            # Ensure same size
            if original.shape != restored.shape:
                restored = cv2.resize(restored, (original.shape[1], original.shape[0]))

            # PSNR
            mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
            if mse > 0:
                metrics["psnr"] = 10 * np.log10((255 ** 2) / mse)
            else:
                metrics["psnr"] = float('inf')

            # SSIM (simplified)
            try:
                from skimage.metrics import structural_similarity
                gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                gray_rest = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
                metrics["ssim"] = structural_similarity(gray_orig, gray_rest)
            except ImportError:
                pass

        except Exception as e:
            logger.debug(f"Metrics calculation failed: {e}")

        return metrics

    def _add_overlay(
        self,
        frame: "np.ndarray",
        frame_num: int,
        timestamp: float,
        metrics: Dict[str, float],
        side_by_side: bool,
    ) -> "np.ndarray":
        """Add text overlay to frame.

        Args:
            frame: Frame to annotate
            frame_num: Frame number
            timestamp: Timestamp in seconds
            metrics: Quality metrics
            side_by_side: Whether frame is side-by-side

        Returns:
            Annotated frame
        """
        frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1

        # Frame info
        text = f"Frame {frame_num} ({timestamp:.1f}s)"
        cv2.putText(frame, text, (10, 20), font, font_scale, color, thickness)

        # Labels for side-by-side
        if side_by_side:
            mid = frame.shape[1] // 2
            cv2.putText(frame, "Original", (10, 40), font, font_scale, color, thickness)
            cv2.putText(frame, "Restored", (mid + 10, 40), font, font_scale, color, thickness)

            # Draw divider line
            cv2.line(frame, (mid, 0), (mid, frame.shape[0]), (128, 128, 128), 1)

        # Metrics
        y_offset = 60
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key.upper()}: {value:.2f}"
                cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 20

        return frame


def generate_quick_preview(
    input_path: Path,
    output_path: Optional[Path] = None,
    every_n: int = 100,
    preset: Optional[str] = None,
) -> QuickPreviewResult:
    """Convenience function to generate quick preview.

    Args:
        input_path: Input video path
        output_path: Output preview path
        every_n: Sample every N frames
        preset: Restoration preset

    Returns:
        QuickPreviewResult
    """
    generator = QuickPreviewGenerator()
    return generator.generate(
        input_path=input_path,
        output_path=output_path,
        every_n=every_n,
        preset=preset,
    )
