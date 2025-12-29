"""Preview system for video restoration comparison.

Provides various preview modes for comparing original and enhanced video:
- Split view (vertical/horizontal)
- Side-by-side comparison
- Interactive slider (for UI)
- Difference visualization

Also includes quality metrics comparison (PSNR, SSIM) and
live preview capabilities for UI integration.
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class PreviewMode(Enum):
    """Available preview modes for comparing original and enhanced frames."""
    SPLIT_VERTICAL = "split_vertical"      # Left/right split with vertical line
    SPLIT_HORIZONTAL = "split_horizontal"  # Top/bottom split with horizontal line
    SIDE_BY_SIDE = "side_by_side"          # Both frames next to each other
    SLIDER = "slider"                       # Interactive slider (for UI)
    DIFF = "diff"                           # Difference visualization (highlights changes)


@dataclass
class PreviewConfig:
    """Configuration for preview generation.

    Attributes:
        mode: Preview mode (SPLIT_VERTICAL, SPLIT_HORIZONTAL, SIDE_BY_SIDE, SLIDER, DIFF)
        split_position: Position of split line (0-1, only for split/slider modes)
        show_labels: Whether to show "Original" and "Enhanced" labels
        label_original: Label text for original frame
        label_enhanced: Label text for enhanced frame
        output_size: Target output size (width, height) or None to auto-calculate
        label_font_scale: Font scale for labels
        label_thickness: Font thickness for labels
        label_color: Label color as BGR tuple
        label_bg_color: Label background color as BGR tuple
        diff_amplification: Amplification factor for difference visualization
        diff_colormap: OpenCV colormap for difference visualization
    """
    mode: PreviewMode = PreviewMode.SPLIT_VERTICAL
    split_position: float = 0.5
    show_labels: bool = True
    label_original: str = "Original"
    label_enhanced: str = "Enhanced"
    output_size: Optional[Tuple[int, int]] = None
    label_font_scale: float = 1.0
    label_thickness: int = 2
    label_color: Tuple[int, int, int] = (255, 255, 255)
    label_bg_color: Tuple[int, int, int] = (0, 0, 0)
    diff_amplification: float = 3.0
    diff_colormap: int = 2  # cv2.COLORMAP_JET

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.split_position <= 1.0:
            raise ValueError("split_position must be between 0.0 and 1.0")

        if self.diff_amplification <= 0:
            raise ValueError("diff_amplification must be positive")

        if self.label_font_scale <= 0:
            raise ValueError("label_font_scale must be positive")

        if self.label_thickness < 1:
            raise ValueError("label_thickness must be at least 1")


@dataclass
class ComparisonMetrics:
    """Quality comparison metrics between original and enhanced frames.

    Attributes:
        psnr: Peak Signal-to-Noise Ratio (dB) - higher is better
        ssim: Structural Similarity Index (0-1) - higher is better
        mse: Mean Squared Error - lower is better
        original_size: Original file/data size in bytes
        enhanced_size: Enhanced file/data size in bytes
        size_change_percent: Percentage change in size
        processing_time: Time taken for enhancement in seconds
        resolution_original: Original frame resolution (width, height)
        resolution_enhanced: Enhanced frame resolution (width, height)
    """
    psnr: float = 0.0
    ssim: float = 0.0
    mse: float = 0.0
    original_size: int = 0
    enhanced_size: int = 0
    size_change_percent: float = 0.0
    processing_time: float = 0.0
    resolution_original: Tuple[int, int] = (0, 0)
    resolution_enhanced: Tuple[int, int] = (0, 0)


class PreviewGenerator:
    """Generate various preview comparisons between original and enhanced frames.

    Supports multiple preview modes for visual comparison of video
    restoration results.

    Example:
        >>> config = PreviewConfig(mode=PreviewMode.SPLIT_VERTICAL)
        >>> generator = PreviewGenerator(config)
        >>> preview = generator.create_split_preview(original_frame, enhanced_frame)
    """

    def __init__(self, config: Optional[PreviewConfig] = None):
        """Initialize preview generator.

        Args:
            config: Preview configuration, uses defaults if not provided
        """
        self.config = config or PreviewConfig()
        self._cv2_available = self._check_cv2()

    def _check_cv2(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available, preview generation limited")
            return False

    def _resize_to_match(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize frames to match dimensions.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Tuple of resized frames with matching dimensions
        """
        if not self._cv2_available:
            return frame1, frame2

        import cv2

        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        if h1 == h2 and w1 == w2:
            return frame1, frame2

        # Use the larger dimensions
        target_h = max(h1, h2)
        target_w = max(w1, w2)

        if (h1, w1) != (target_h, target_w):
            frame1 = cv2.resize(frame1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if (h2, w2) != (target_h, target_w):
            frame2 = cv2.resize(frame2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return frame1, frame2

    def _add_label(
        self,
        frame: np.ndarray,
        text: str,
        position: str = "top-left",
    ) -> np.ndarray:
        """Add text label to frame.

        Args:
            frame: Input frame
            text: Label text
            position: Label position ("top-left", "top-right", "bottom-left", "bottom-right")

        Returns:
            Frame with label added
        """
        if not self._cv2_available or not self.config.show_labels:
            return frame

        import cv2

        frame = frame.copy()
        h, w = frame.shape[:2]

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, self.config.label_font_scale, self.config.label_thickness
        )

        padding = 10

        # Calculate position
        if position == "top-left":
            x, y = padding, text_h + padding
        elif position == "top-right":
            x, y = w - text_w - padding, text_h + padding
        elif position == "bottom-left":
            x, y = padding, h - padding
        elif position == "bottom-right":
            x, y = w - text_w - padding, h - padding
        else:
            x, y = padding, text_h + padding

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            self.config.label_bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            frame, text, (x, y), font,
            self.config.label_font_scale,
            self.config.label_color,
            self.config.label_thickness
        )

        return frame

    def _apply_output_size(self, frame: np.ndarray) -> np.ndarray:
        """Apply output size configuration.

        Args:
            frame: Input frame

        Returns:
            Resized frame if output_size is configured, otherwise original
        """
        if self.config.output_size is None or not self._cv2_available:
            return frame

        import cv2

        target_w, target_h = self.config.output_size
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def create_split_preview(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> np.ndarray:
        """Create split preview (vertical or horizontal).

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            Preview frame with split comparison
        """
        if not self._cv2_available:
            logger.error("OpenCV required for split preview")
            return original

        import cv2

        # Resize frames to match
        original, enhanced = self._resize_to_match(original, enhanced)
        h, w = original.shape[:2]

        # Create output frame
        result = np.zeros_like(original)

        if self.config.mode == PreviewMode.SPLIT_VERTICAL:
            # Vertical split (left/right)
            split_x = int(w * self.config.split_position)
            result[:, :split_x] = original[:, :split_x]
            result[:, split_x:] = enhanced[:, split_x:]

            # Draw split line
            cv2.line(result, (split_x, 0), (split_x, h), (255, 255, 255), 2)

            # Add labels
            if self.config.show_labels:
                result = self._add_label(result, self.config.label_original, "top-left")
                result = self._add_label(result, self.config.label_enhanced, "top-right")

        else:
            # Horizontal split (top/bottom)
            split_y = int(h * self.config.split_position)
            result[:split_y, :] = original[:split_y, :]
            result[split_y:, :] = enhanced[split_y:, :]

            # Draw split line
            cv2.line(result, (0, split_y), (w, split_y), (255, 255, 255), 2)

            # Add labels
            if self.config.show_labels:
                result = self._add_label(result, self.config.label_original, "top-left")
                result = self._add_label(result, self.config.label_enhanced, "bottom-left")

        return self._apply_output_size(result)

    def create_side_by_side(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side comparison.

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            Preview frame with side-by-side comparison
        """
        if not self._cv2_available:
            logger.error("OpenCV required for side-by-side preview")
            return original

        import cv2

        # Resize frames to match
        original, enhanced = self._resize_to_match(original, enhanced)

        # Add labels
        if self.config.show_labels:
            original = self._add_label(original, self.config.label_original, "top-left")
            enhanced = self._add_label(enhanced, self.config.label_enhanced, "top-left")

        # Concatenate horizontally
        result = np.hstack([original, enhanced])

        # Add separator line
        h, w = result.shape[:2]
        cv2.line(result, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        return self._apply_output_size(result)

    def create_diff_view(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> np.ndarray:
        """Create difference visualization.

        Highlights the differences between original and enhanced frames
        using a colormap to make changes visible.

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            Difference visualization frame
        """
        if not self._cv2_available:
            logger.error("OpenCV required for diff view")
            return original

        import cv2

        # Resize frames to match
        original, enhanced = self._resize_to_match(original, enhanced)

        # Convert to grayscale for difference calculation
        if len(original.shape) == 3:
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray_orig = original

        if len(enhanced.shape) == 3:
            gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray_enh = enhanced

        # Calculate absolute difference
        diff = cv2.absdiff(gray_orig.astype(np.float32), gray_enh.astype(np.float32))

        # Amplify differences
        diff = np.clip(diff * self.config.diff_amplification, 0, 255).astype(np.uint8)

        # Apply colormap
        diff_colored = cv2.applyColorMap(diff, self.config.diff_colormap)

        # Add label
        if self.config.show_labels:
            diff_colored = self._add_label(diff_colored, "Difference", "top-left")

        return self._apply_output_size(diff_colored)

    def create_slider_preview(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        position: float,
    ) -> np.ndarray:
        """Create slider preview at specific position.

        This is similar to split preview but designed for
        interactive slider usage in UI.

        Args:
            original: Original frame
            enhanced: Enhanced frame
            position: Slider position (0-1)

        Returns:
            Preview frame at given slider position
        """
        # Store original split position
        orig_position = self.config.split_position
        orig_mode = self.config.mode

        # Temporarily set position and mode
        self.config.split_position = position
        self.config.mode = PreviewMode.SPLIT_VERTICAL

        try:
            result = self.create_split_preview(original, enhanced)
        finally:
            # Restore original settings
            self.config.split_position = orig_position
            self.config.mode = orig_mode

        return result

    def create_preview(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> np.ndarray:
        """Create preview using configured mode.

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            Preview frame using configured mode
        """
        if self.config.mode == PreviewMode.SPLIT_VERTICAL:
            return self.create_split_preview(original, enhanced)
        elif self.config.mode == PreviewMode.SPLIT_HORIZONTAL:
            return self.create_split_preview(original, enhanced)
        elif self.config.mode == PreviewMode.SIDE_BY_SIDE:
            return self.create_side_by_side(original, enhanced)
        elif self.config.mode == PreviewMode.SLIDER:
            return self.create_slider_preview(original, enhanced, self.config.split_position)
        elif self.config.mode == PreviewMode.DIFF:
            return self.create_diff_view(original, enhanced)
        else:
            logger.warning(f"Unknown preview mode: {self.config.mode}")
            return self.create_split_preview(original, enhanced)

    def create_preview_video(
        self,
        original_video: Path,
        enhanced_video: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Create preview video comparing original and enhanced.

        Args:
            original_video: Path to original video
            enhanced_video: Path to enhanced video
            output_path: Path for output preview video
            progress_callback: Optional progress callback (0.0 to 1.0)

        Returns:
            True if successful, False otherwise
        """
        if not self._cv2_available:
            logger.error("OpenCV required for preview video generation")
            return False

        import cv2

        try:
            # Open both videos
            cap_orig = cv2.VideoCapture(str(original_video))
            cap_enh = cv2.VideoCapture(str(enhanced_video))

            if not cap_orig.isOpened() or not cap_enh.isOpened():
                logger.error("Failed to open video files")
                return False

            # Get video properties
            fps = cap_orig.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

            # Read first frames to get dimensions
            ret_orig, frame_orig = cap_orig.read()
            ret_enh, frame_enh = cap_enh.read()

            if not ret_orig or not ret_enh:
                logger.error("Failed to read first frames")
                return False

            # Calculate output size based on mode
            frame_orig, frame_enh = self._resize_to_match(frame_orig, frame_enh)
            preview_frame = self.create_preview(frame_orig, frame_enh)
            h, w = preview_frame.shape[:2]

            # Reset video positions
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap_enh.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            if not writer.isOpened():
                logger.error("Failed to create output video writer")
                return False

            # Process frames
            frame_count = 0
            while True:
                ret_orig, frame_orig = cap_orig.read()
                ret_enh, frame_enh = cap_enh.read()

                if not ret_orig or not ret_enh:
                    break

                # Resize and create preview
                frame_orig, frame_enh = self._resize_to_match(frame_orig, frame_enh)
                preview = self.create_preview(frame_orig, frame_enh)

                writer.write(preview)
                frame_count += 1

                if progress_callback and total_frames > 0:
                    progress_callback(frame_count / total_frames)

            # Cleanup
            cap_orig.release()
            cap_enh.release()
            writer.release()

            logger.info(f"Preview video created: {output_path} ({frame_count} frames)")
            return True

        except Exception as e:
            logger.error(f"Failed to create preview video: {e}")
            return False

    def create_animated_comparison(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        output_gif: Path,
        duration_ms: int = 1500,
        steps: int = 30,
    ) -> bool:
        """Create animated GIF comparison with sliding effect.

        Args:
            original: Original frame
            enhanced: Enhanced frame
            output_gif: Output GIF path
            duration_ms: Total animation duration in milliseconds
            steps: Number of steps in the animation

        Returns:
            True if successful, False otherwise
        """
        if not self._cv2_available:
            logger.error("OpenCV required for animated comparison")
            return False

        try:
            from PIL import Image
        except ImportError:
            logger.error("Pillow required for GIF creation: pip install Pillow")
            return False

        import cv2

        try:
            # Resize frames to match
            original, enhanced = self._resize_to_match(original, enhanced)

            frames = []
            frame_duration = duration_ms // (steps * 2)  # Back and forth

            # Generate forward sweep frames
            for i in range(steps):
                position = i / (steps - 1)
                preview = self.create_slider_preview(original, enhanced, position)

                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                frames.append(pil_frame)

            # Generate reverse sweep frames
            for i in range(steps - 2, 0, -1):
                position = i / (steps - 1)
                preview = self.create_slider_preview(original, enhanced, position)

                rgb_frame = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                frames.append(pil_frame)

            # Save as GIF
            output_gif = Path(output_gif)
            output_gif.parent.mkdir(parents=True, exist_ok=True)

            frames[0].save(
                str(output_gif),
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,
            )

            logger.info(f"Animated comparison created: {output_gif}")
            return True

        except Exception as e:
            logger.error(f"Failed to create animated comparison: {e}")
            return False


class LivePreview:
    """Live preview for UI integration.

    Provides methods for quickly generating preview frames
    before full video processing.

    Example:
        >>> preview = LivePreview()
        >>> sample_frames = preview.sample_frames("video.mp4", count=5)
        >>> for frame in sample_frames:
        ...     enhanced = preview.quick_enhance(frame, config)
    """

    def __init__(self):
        """Initialize live preview."""
        self._cv2_available = self._check_cv2()

    def _check_cv2(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available")
            return False

    def sample_frames(
        self,
        video_path: Path,
        count: int = 5,
        start_percent: float = 0.1,
        end_percent: float = 0.9,
    ) -> List[np.ndarray]:
        """Sample frames from a video.

        Samples frames evenly distributed throughout the video
        for preview purposes.

        Args:
            video_path: Path to video file
            count: Number of frames to sample
            start_percent: Start position as percentage (0-1)
            end_percent: End position as percentage (0-1)

        Returns:
            List of sampled frames as numpy arrays
        """
        if not self._cv2_available:
            logger.error("OpenCV required for frame sampling")
            return []

        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error("Video has no frames")
                return []

            # Calculate sample positions
            start_frame = int(total_frames * start_percent)
            end_frame = int(total_frames * end_percent)
            frame_range = end_frame - start_frame

            if count == 1:
                sample_positions = [start_frame + frame_range // 2]
            else:
                step = frame_range // (count - 1)
                sample_positions = [start_frame + i * step for i in range(count)]

            frames = []
            for pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read frame at position {pos}")

            cap.release()

            logger.info(f"Sampled {len(frames)} frames from {video_path.name}")
            return frames

        except Exception as e:
            logger.error(f"Failed to sample frames: {e}")
            return []

    def quick_enhance(
        self,
        frame: np.ndarray,
        config: Any,
        timeout: float = 10.0,
    ) -> np.ndarray:
        """Quick enhancement of a single frame for preview.

        Applies a simplified enhancement for quick preview.
        Does NOT use full restoration pipeline for speed.

        Args:
            frame: Input frame
            config: Config object with enhancement settings
            timeout: Maximum time for enhancement in seconds

        Returns:
            Enhanced frame (or original if enhancement fails/times out)
        """
        if not self._cv2_available:
            return frame

        import cv2

        start_time = time.time()

        try:
            # Apply simple enhancements for quick preview
            enhanced = frame.copy()

            # Basic sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            # Auto contrast adjustment
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)

            enhanced = cv2.merge([l_channel, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning("Quick enhance timeout, returning partial result")
                return enhanced

            # Simple upscale if scale factor > 1
            if hasattr(config, 'scale_factor') and config.scale_factor > 1:
                h, w = frame.shape[:2]
                new_size = (w * config.scale_factor, h * config.scale_factor)
                enhanced = cv2.resize(enhanced, new_size, interpolation=cv2.INTER_LANCZOS4)

            return enhanced

        except Exception as e:
            logger.error(f"Quick enhance failed: {e}")
            return frame

    def preview_settings(
        self,
        video_path: Path,
        config: Any,
        sample_count: int = 3,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Preview enhancement settings on sample frames.

        Args:
            video_path: Path to video file
            config: Enhancement configuration
            sample_count: Number of frames to sample

        Returns:
            List of (original, enhanced) frame pairs
        """
        frames = self.sample_frames(video_path, count=sample_count)

        results = []
        for frame in frames:
            enhanced = self.quick_enhance(frame, config)
            results.append((frame, enhanced))

        return results

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get basic video information.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info (width, height, fps, frames, duration)
        """
        if not self._cv2_available:
            return {}

        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            return {}

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {}

            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': 0.0,
            }

            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']

            cap.release()
            return info

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}


class QualityComparison:
    """Compare quality metrics between original and enhanced frames.

    Calculates PSNR, SSIM, and other metrics for quality assessment.

    Example:
        >>> comparison = QualityComparison()
        >>> metrics = comparison.compare_metrics(original, enhanced)
        >>> print(f"PSNR: {metrics.psnr:.2f} dB, SSIM: {metrics.ssim:.4f}")
    """

    def __init__(self):
        """Initialize quality comparison."""
        self._cv2_available = self._check_cv2()
        self._skimage_available = self._check_skimage()

    def _check_cv2(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            return False

    def _check_skimage(self) -> bool:
        """Check if scikit-image is available."""
        try:
            from skimage.metrics import structural_similarity, peak_signal_noise_ratio
            return True
        except ImportError:
            return False

    def calculate_psnr(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> float:
        """Calculate Peak Signal-to-Noise Ratio.

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            PSNR value in dB (higher is better)
        """
        if self._skimage_available:
            try:
                from skimage.metrics import peak_signal_noise_ratio

                # Ensure same size
                if original.shape != enhanced.shape:
                    if self._cv2_available:
                        import cv2
                        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
                    else:
                        return 0.0

                return peak_signal_noise_ratio(original, enhanced)
            except Exception as e:
                logger.debug(f"PSNR calculation error: {e}")

        # Fallback to manual calculation
        try:
            if original.shape != enhanced.shape:
                return 0.0

            mse = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
            if mse == 0:
                return float('inf')

            max_pixel = 255.0
            return 20 * np.log10(max_pixel / np.sqrt(mse))
        except Exception as e:
            logger.error(f"PSNR calculation failed: {e}")
            return 0.0

    def calculate_ssim(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> float:
        """Calculate Structural Similarity Index.

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            SSIM value (0-1, higher is better)
        """
        if not self._skimage_available:
            logger.warning("scikit-image required for SSIM: pip install scikit-image")
            return 0.0

        try:
            from skimage.metrics import structural_similarity

            # Ensure same size
            if original.shape != enhanced.shape:
                if self._cv2_available:
                    import cv2
                    enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
                else:
                    return 0.0

            # Convert to grayscale if color
            if len(original.shape) == 3:
                if self._cv2_available:
                    import cv2
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                else:
                    original = np.mean(original, axis=2).astype(np.uint8)
                    enhanced = np.mean(enhanced, axis=2).astype(np.uint8)

            return structural_similarity(original, enhanced)

        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0

    def calculate_mse(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> float:
        """Calculate Mean Squared Error.

        Args:
            original: Original frame
            enhanced: Enhanced frame

        Returns:
            MSE value (lower is better)
        """
        try:
            # Ensure same size
            if original.shape != enhanced.shape:
                if self._cv2_available:
                    import cv2
                    enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
                else:
                    return 0.0

            return np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)

        except Exception as e:
            logger.error(f"MSE calculation failed: {e}")
            return 0.0

    def compare_metrics(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        processing_time: float = 0.0,
    ) -> ComparisonMetrics:
        """Compare all quality metrics between frames.

        Args:
            original: Original frame
            enhanced: Enhanced frame
            processing_time: Time taken for enhancement in seconds

        Returns:
            ComparisonMetrics with all calculated metrics
        """
        metrics = ComparisonMetrics()

        # Calculate quality metrics
        metrics.psnr = self.calculate_psnr(original, enhanced)
        metrics.ssim = self.calculate_ssim(original, enhanced)
        metrics.mse = self.calculate_mse(original, enhanced)

        # Get resolutions
        metrics.resolution_original = (original.shape[1], original.shape[0])
        metrics.resolution_enhanced = (enhanced.shape[1], enhanced.shape[0])

        # Estimate data sizes (raw pixel data)
        metrics.original_size = original.nbytes
        metrics.enhanced_size = enhanced.nbytes

        if metrics.original_size > 0:
            metrics.size_change_percent = (
                (metrics.enhanced_size - metrics.original_size) /
                metrics.original_size * 100
            )

        metrics.processing_time = processing_time

        return metrics

    def compare_video_metrics(
        self,
        original_video: Path,
        enhanced_video: Path,
        sample_count: int = 10,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ComparisonMetrics:
        """Compare metrics across sampled video frames.

        Args:
            original_video: Path to original video
            enhanced_video: Path to enhanced video
            sample_count: Number of frames to sample
            progress_callback: Optional progress callback

        Returns:
            Average ComparisonMetrics across sampled frames
        """
        if not self._cv2_available:
            logger.error("OpenCV required for video comparison")
            return ComparisonMetrics()

        import cv2

        try:
            cap_orig = cv2.VideoCapture(str(original_video))
            cap_enh = cv2.VideoCapture(str(enhanced_video))

            if not cap_orig.isOpened() or not cap_enh.isOpened():
                logger.error("Failed to open video files")
                return ComparisonMetrics()

            total_frames_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_enh = int(cap_enh.get(cv2.CAP_PROP_FRAME_COUNT))

            # Use the shorter video length
            total_frames = min(total_frames_orig, total_frames_enh)

            if total_frames == 0:
                return ComparisonMetrics()

            # Calculate sample positions
            step = total_frames // sample_count
            sample_positions = [i * step for i in range(sample_count)]

            # Collect metrics
            psnr_values = []
            ssim_values = []
            mse_values = []

            for i, pos in enumerate(sample_positions):
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, pos)
                cap_enh.set(cv2.CAP_PROP_POS_FRAMES, pos)

                ret_orig, frame_orig = cap_orig.read()
                ret_enh, frame_enh = cap_enh.read()

                if ret_orig and ret_enh:
                    psnr_values.append(self.calculate_psnr(frame_orig, frame_enh))
                    ssim_values.append(self.calculate_ssim(frame_orig, frame_enh))
                    mse_values.append(self.calculate_mse(frame_orig, frame_enh))

                if progress_callback:
                    progress_callback((i + 1) / len(sample_positions))

            cap_orig.release()
            cap_enh.release()

            # Calculate averages
            metrics = ComparisonMetrics()
            if psnr_values:
                metrics.psnr = np.mean(psnr_values)
            if ssim_values:
                metrics.ssim = np.mean(ssim_values)
            if mse_values:
                metrics.mse = np.mean(mse_values)

            # Get file sizes
            metrics.original_size = original_video.stat().st_size
            metrics.enhanced_size = enhanced_video.stat().st_size

            if metrics.original_size > 0:
                metrics.size_change_percent = (
                    (metrics.enhanced_size - metrics.original_size) /
                    metrics.original_size * 100
                )

            return metrics

        except Exception as e:
            logger.error(f"Video comparison failed: {e}")
            return ComparisonMetrics()

    def generate_comparison_report(self, metrics: ComparisonMetrics) -> str:
        """Generate human-readable comparison report.

        Args:
            metrics: ComparisonMetrics to report

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 50,
            "Quality Comparison Report",
            "=" * 50,
            "",
            "Quality Metrics:",
            f"  PSNR: {metrics.psnr:.2f} dB" + (
                " (Excellent)" if metrics.psnr > 40 else
                " (Good)" if metrics.psnr > 30 else
                " (Fair)" if metrics.psnr > 20 else
                " (Poor)"
            ),
            f"  SSIM: {metrics.ssim:.4f}" + (
                " (Excellent)" if metrics.ssim > 0.95 else
                " (Good)" if metrics.ssim > 0.90 else
                " (Fair)" if metrics.ssim > 0.80 else
                " (Poor)"
            ),
            f"  MSE:  {metrics.mse:.2f}",
            "",
            "Resolution:",
            f"  Original: {metrics.resolution_original[0]}x{metrics.resolution_original[1]}",
            f"  Enhanced: {metrics.resolution_enhanced[0]}x{metrics.resolution_enhanced[1]}",
            "",
            "File Size:",
            f"  Original: {metrics.original_size / 1024:.1f} KB",
            f"  Enhanced: {metrics.enhanced_size / 1024:.1f} KB",
            f"  Change:   {metrics.size_change_percent:+.1f}%",
            "",
        ]

        if metrics.processing_time > 0:
            lines.extend([
                "Processing Time:",
                f"  {metrics.processing_time:.2f} seconds",
                "",
            ])

        lines.append("=" * 50)

        return "\n".join(lines)


def create_gradio_slider_preview(
    original: np.ndarray,
    enhanced: np.ndarray,
    config: Optional[PreviewConfig] = None,
) -> Dict[str, Any]:
    """Create data for Gradio slider preview component.

    This helper function prepares frames for use with
    Gradio's image comparison slider component.

    Args:
        original: Original frame (BGR format)
        enhanced: Enhanced frame (BGR format)
        config: Optional preview configuration

    Returns:
        Dictionary with 'original' and 'enhanced' RGB frames
        suitable for Gradio components
    """
    config = config or PreviewConfig()

    try:
        import cv2

        # Resize to match if needed
        generator = PreviewGenerator(config)
        original, enhanced = generator._resize_to_match(original, enhanced)

        # Convert BGR to RGB for Gradio
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        # Add labels if configured
        if config.show_labels:
            original_rgb = generator._add_label(
                cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR),
                config.label_original,
                "top-left"
            )
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)

            enhanced_rgb = generator._add_label(
                cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR),
                config.label_enhanced,
                "top-left"
            )
            enhanced_rgb = cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2RGB)

        return {
            'original': original_rgb,
            'enhanced': enhanced_rgb,
            'width': original_rgb.shape[1],
            'height': original_rgb.shape[0],
        }

    except ImportError:
        logger.error("OpenCV required for Gradio preview")
        return {
            'original': original,
            'enhanced': enhanced,
            'width': original.shape[1] if len(original.shape) > 1 else 0,
            'height': original.shape[0] if len(original.shape) > 0 else 0,
        }
