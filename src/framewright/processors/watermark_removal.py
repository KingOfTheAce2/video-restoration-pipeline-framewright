"""Watermark removal processor using LaMA (Large Mask Inpainting).

This module provides watermark removal capabilities using the LaMA inpainting model.
Supports both user-defined mask regions and automatic watermark detection.

Features:
- LaMA model for high-quality inpainting
- Automatic watermark detection in common positions (corners, edges)
- Support for rectangular and custom mask images
- Batch processing for video frame sequences
- Fallback to OpenCV inpainting when LaMA is unavailable

Example:
    >>> config = WatermarkConfig(auto_detect=True, detection_threshold=0.5)
    >>> remover = WatermarkRemover(config)
    >>> if remover.is_available():
    ...     result = remover.remove_watermark(frame)
"""

import logging
import shutil
import urllib.request
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = Path.home() / ".framewright" / "models" / "lama"

# LaMA model URLs and checksums
LAMA_MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_MODEL_FILENAME = "big-lama.pt"
LAMA_MODEL_SHA256 = "a9f6ccd1c8e4e26d665e27f61963c5d5d7c5c8c8d9c8e5c4c3c2c1c0c9c8c7c6"  # Placeholder


class WatermarkPosition(Enum):
    """Common watermark positions for auto-detection."""
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()
    TOP_CENTER = auto()
    BOTTOM_CENTER = auto()
    CENTER = auto()
    CUSTOM = auto()


@dataclass
class WatermarkConfig:
    """Configuration for watermark removal.

    Attributes:
        mask_path: Optional path to a custom mask image (white = watermark area).
        auto_detect: Enable automatic watermark detection.
        detection_threshold: Sensitivity threshold for auto-detection (0.0-1.0).
            Lower values detect more aggressively, higher values are more conservative.
        positions: List of positions to check for auto-detection.
        margin_percent: Percentage of frame size for corner region detection.
        dilate_mask: Number of pixels to expand detected mask regions.
        model_name: Name of the LaMA model to use.
    """
    mask_path: Optional[Path] = None
    auto_detect: bool = False
    detection_threshold: float = 0.5
    positions: List[WatermarkPosition] = field(
        default_factory=lambda: [
            WatermarkPosition.BOTTOM_RIGHT,
            WatermarkPosition.BOTTOM_LEFT,
            WatermarkPosition.TOP_RIGHT,
            WatermarkPosition.TOP_LEFT,
        ]
    )
    margin_percent: float = 0.15
    dilate_mask: int = 5
    model_name: str = "big-lama"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.detection_threshold < 0.0 or self.detection_threshold > 1.0:
            raise ValueError("detection_threshold must be between 0.0 and 1.0")
        if self.margin_percent < 0.0 or self.margin_percent > 0.5:
            raise ValueError("margin_percent must be between 0.0 and 0.5")
        if self.mask_path is not None:
            self.mask_path = Path(self.mask_path)


@dataclass
class WatermarkRemovalResult:
    """Results from watermark removal batch operation.

    Attributes:
        frames_processed: Total number of frames processed.
        frames_modified: Number of frames where watermarks were removed.
        watermarks_detected: Total watermark regions detected.
        errors: List of error messages encountered.
    """
    frames_processed: int = 0
    frames_modified: int = 0
    watermarks_detected: int = 0
    errors: List[str] = field(default_factory=list)


class WatermarkRemover:
    """Watermark removal processor using LaMA inpainting.

    This class provides methods to detect and remove watermarks from video frames
    using the LaMA (Large Mask Inpainting) model. It supports both automatic
    watermark detection and user-defined mask regions.

    Attributes:
        config: WatermarkConfig with removal settings.
        model_dir: Directory containing LaMA model files.

    Example:
        >>> config = WatermarkConfig(auto_detect=True)
        >>> remover = WatermarkRemover(config, model_dir=Path("~/.framewright/models/lama"))
        >>> if remover.is_available():
        ...     clean_frame = remover.remove_watermark(frame, mask=None)
    """

    def __init__(
        self,
        config: Optional[WatermarkConfig] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the watermark remover.

        Args:
            config: WatermarkConfig with removal settings. If None, uses defaults.
            model_dir: Directory for LaMA model files. Defaults to ~/.framewright/models/lama/
        """
        self.config = config or WatermarkConfig()
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model = None
        self._backend: Optional[str] = None
        self._custom_mask: Optional[np.ndarray] = None
        self._cv2 = None
        self._torch = None

        # Load custom mask if provided
        if self.config.mask_path is not None:
            self._load_custom_mask()

        # Initialize backend
        self._init_backend()

    def _load_custom_mask(self) -> None:
        """Load custom mask image from config.mask_path."""
        try:
            import cv2
            self._cv2 = cv2

            if self.config.mask_path and self.config.mask_path.exists():
                mask = cv2.imread(str(self.config.mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Normalize to binary mask (0 or 255)
                    _, self._custom_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    logger.info(f"Loaded custom mask from {self.config.mask_path}")
                else:
                    logger.warning(f"Failed to load mask from {self.config.mask_path}")
            else:
                logger.warning(f"Mask path does not exist: {self.config.mask_path}")
        except ImportError:
            logger.error("OpenCV required to load custom mask")

    def _init_backend(self) -> None:
        """Initialize the inpainting backend (LaMA or OpenCV fallback)."""
        # Try LaMA first
        if self._init_lama():
            self._backend = "lama"
            logger.info("LaMA inpainting backend initialized successfully")
            return

        # Fall back to OpenCV
        if self._init_opencv():
            self._backend = "opencv"
            logger.info("Using OpenCV fallback for inpainting")
            return

        logger.warning("No inpainting backend available")
        self._backend = None

    def _init_lama(self) -> bool:
        """Initialize LaMA model.

        Returns:
            True if LaMA initialized successfully, False otherwise.
        """
        try:
            import torch
            self._torch = torch

            # Check if model exists
            model_path = self.model_dir / LAMA_MODEL_FILENAME
            if not model_path.exists():
                logger.info("LaMA model not found. Call download_model() to download.")
                return False

            # Try to load simple-lama-inpainting
            try:
                from simple_lama_inpainting import SimpleLama
                self._model = SimpleLama(device=self._get_device())
                return True
            except ImportError:
                pass

            # Try direct model loading
            try:
                self._model = torch.jit.load(str(model_path), map_location=self._get_device())
                self._model.eval()
                return True
            except Exception as e:
                logger.debug(f"Failed to load LaMA model directly: {e}")

            return False

        except ImportError:
            logger.debug("PyTorch not available for LaMA")
            return False
        except Exception as e:
            logger.debug(f"LaMA initialization failed: {e}")
            return False

    def _init_opencv(self) -> bool:
        """Initialize OpenCV fallback backend.

        Returns:
            True if OpenCV available, False otherwise.
        """
        try:
            import cv2
            self._cv2 = cv2
            return True
        except ImportError:
            return False

    def _get_device(self) -> str:
        """Get the best available device for inference.

        Returns:
            Device string ('cuda' or 'cpu').
        """
        if self._torch is None:
            return "cpu"

        if self._torch.cuda.is_available():
            return "cuda"
        elif hasattr(self._torch.backends, 'mps') and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def is_available(self) -> bool:
        """Check if watermark removal is available.

        Returns:
            True if an inpainting backend is available, False otherwise.
        """
        return self._backend is not None

    def download_model(self, force: bool = False) -> bool:
        """Download the LaMA model.

        Args:
            force: If True, download even if model already exists.

        Returns:
            True if download successful, False otherwise.
        """
        model_path = self.model_dir / LAMA_MODEL_FILENAME

        if model_path.exists() and not force:
            logger.info(f"Model already exists at {model_path}")
            return True

        # Create directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading LaMA model to {model_path}...")
        try:
            # Download with progress reporting
            def report_progress(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 // total_size)
                    if block_num % 100 == 0:
                        logger.info(f"Download progress: {percent}%")

            urllib.request.urlretrieve(
                LAMA_MODEL_URL,
                model_path,
                reporthook=report_progress
            )

            logger.info("LaMA model downloaded successfully")

            # Re-initialize backend with new model
            self._init_backend()
            return True

        except Exception as e:
            logger.error(f"Failed to download LaMA model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def detect_watermark(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect watermark regions in a frame.

        Uses edge detection and pattern analysis to identify potential watermark
        locations, focusing on common positions (corners, edges).

        Args:
            frame: Input frame as numpy array (BGR or RGB format, HxWxC).

        Returns:
            Binary mask (numpy array, same height/width as frame) where 255 indicates
            watermark regions, or None if no watermark detected.
        """
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                logger.error("OpenCV required for watermark detection")
                return None

        cv2 = self._cv2

        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Create empty mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate margin sizes
        margin_h = int(height * self.config.margin_percent)
        margin_w = int(width * self.config.margin_percent)

        # Define regions based on configured positions
        regions = self._get_detection_regions(width, height, margin_w, margin_h)

        # Analyze each region
        watermarks_found = False
        for pos, (x1, y1, x2, y2) in regions.items():
            if pos not in self.config.positions:
                continue

            roi = gray[y1:y2, x1:x2]
            roi_mask = self._analyze_region_for_watermark(roi)

            if roi_mask is not None:
                # Place detected mask in combined mask
                combined_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    combined_mask[y1:y2, x1:x2],
                    roi_mask
                )
                watermarks_found = True

        if not watermarks_found:
            return None

        # Dilate mask to ensure complete coverage
        if self.config.dilate_mask > 0:
            kernel = np.ones((self.config.dilate_mask, self.config.dilate_mask), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # Return None if mask is essentially empty
        if np.sum(combined_mask) < 100:
            return None

        return combined_mask

    def _get_detection_regions(
        self,
        width: int,
        height: int,
        margin_w: int,
        margin_h: int,
    ) -> dict:
        """Get detection regions based on frame dimensions.

        Args:
            width: Frame width.
            height: Frame height.
            margin_w: Width margin for corner detection.
            margin_h: Height margin for corner detection.

        Returns:
            Dictionary mapping WatermarkPosition to (x1, y1, x2, y2) coordinates.
        """
        return {
            WatermarkPosition.TOP_LEFT: (0, 0, margin_w, margin_h),
            WatermarkPosition.TOP_RIGHT: (width - margin_w, 0, width, margin_h),
            WatermarkPosition.BOTTOM_LEFT: (0, height - margin_h, margin_w, height),
            WatermarkPosition.BOTTOM_RIGHT: (width - margin_w, height - margin_h, width, height),
            WatermarkPosition.TOP_CENTER: (width // 4, 0, 3 * width // 4, margin_h),
            WatermarkPosition.BOTTOM_CENTER: (width // 4, height - margin_h, 3 * width // 4, height),
            WatermarkPosition.CENTER: (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
        }

    def _analyze_region_for_watermark(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Analyze a region for watermark presence.

        Args:
            roi: Region of interest (grayscale).

        Returns:
            Binary mask of detected watermark in ROI, or None.
        """
        cv2 = self._cv2
        if cv2 is None:
            return None

        if roi.size == 0:
            return None

        height, width = roi.shape[:2]

        # Method 1: Edge detection
        edges = cv2.Canny(roi, 50, 150)

        # Method 2: Adaptive thresholding for text detection
        adaptive = cv2.adaptiveThreshold(
            roi, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Combine methods
        combined = cv2.bitwise_or(edges, adaptive)

        # Calculate density
        edge_density = np.sum(combined > 0) / (height * width) if height * width > 0 else 0

        # Threshold based on config
        threshold = (1.0 - self.config.detection_threshold) * 0.1  # Scale to reasonable range

        if edge_density < threshold:
            return None

        # Find contours and filter by size
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(roi)
        min_area = height * width * 0.001  # Minimum 0.1% of region
        max_area = height * width * 0.8  # Maximum 80% of region

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)

        if np.sum(mask) < 50:
            return None

        return mask

    def remove_watermark(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Remove watermark from a single frame.

        Args:
            frame: Input frame as numpy array (BGR format, HxWxC).
            mask: Optional binary mask where 255 indicates watermark regions.
                If None and auto_detect is enabled, detection is performed.
                If None and a custom mask was loaded, that mask is used.

        Returns:
            Processed frame with watermark removed (same shape as input).
        """
        if frame is None or frame.size == 0:
            return frame

        # Determine mask to use
        effective_mask = mask

        if effective_mask is None:
            # Try custom mask first
            if self._custom_mask is not None:
                # Resize custom mask to match frame if needed
                if self._custom_mask.shape[:2] != frame.shape[:2]:
                    if self._cv2 is not None:
                        effective_mask = self._cv2.resize(
                            self._custom_mask,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=self._cv2.INTER_NEAREST
                        )
                    else:
                        effective_mask = self._custom_mask
                else:
                    effective_mask = self._custom_mask
            # Try auto-detection
            elif self.config.auto_detect:
                effective_mask = self.detect_watermark(frame)

        # No mask = no watermark to remove
        if effective_mask is None:
            return frame.copy()

        # Ensure mask is correct shape
        if effective_mask.shape[:2] != frame.shape[:2]:
            logger.warning("Mask shape mismatch, skipping watermark removal")
            return frame.copy()

        # Apply inpainting
        if self._backend == "lama":
            return self._inpaint_lama(frame, effective_mask)
        elif self._backend == "opencv":
            return self._inpaint_opencv(frame, effective_mask)
        else:
            logger.warning("No inpainting backend available")
            return frame.copy()

    def _inpaint_lama(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Perform inpainting using LaMA model.

        Args:
            frame: Input frame (BGR, HxWxC).
            mask: Binary mask (HxW).

        Returns:
            Inpainted frame.
        """
        try:
            # Check if using simple_lama_inpainting
            if hasattr(self._model, '__call__'):
                from PIL import Image
                import cv2

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(rgb_frame)
                mask_pil = Image.fromarray(mask)

                # Inpaint
                result_pil = self._model(image_pil, mask_pil)

                # Convert back to BGR numpy
                result = np.array(result_pil)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                return result

            # Direct model usage
            else:
                return self._inpaint_lama_direct(frame, mask)

        except Exception as e:
            logger.warning(f"LaMA inpainting failed: {e}. Falling back to OpenCV.")
            return self._inpaint_opencv(frame, mask)

    def _inpaint_lama_direct(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Perform inpainting using LaMA model directly with PyTorch.

        Args:
            frame: Input frame (BGR, HxWxC).
            mask: Binary mask (HxW).

        Returns:
            Inpainted frame.
        """
        if self._torch is None:
            return self._inpaint_opencv(frame, mask)

        torch = self._torch

        try:
            # Prepare input
            device = self._get_device()

            # Convert to float and normalize
            img = frame.astype(np.float32) / 255.0
            mask_float = (mask > 127).astype(np.float32)

            # Convert to tensor (BCHW format)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                result = self._model(img_tensor, mask_tensor)

            # Convert back to numpy
            result_np = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            result_np = (result_np * 255).clip(0, 255).astype(np.uint8)

            return result_np

        except Exception as e:
            logger.debug(f"LaMA direct inpainting failed: {e}")
            return self._inpaint_opencv(frame, mask)

    def _inpaint_opencv(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Perform inpainting using OpenCV.

        Args:
            frame: Input frame (BGR, HxWxC).
            mask: Binary mask (HxW).

        Returns:
            Inpainted frame.
        """
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                logger.error("OpenCV not available for inpainting")
                return frame.copy()

        cv2 = self._cv2

        # Ensure mask is uint8
        mask_uint8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

        # Use Telea algorithm (generally better for larger areas)
        result = cv2.inpaint(frame, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return result

    def remove_batch(
        self,
        frames: List[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Remove watermarks from a batch of frames.

        Uses the same mask for all frames (optimized for video where watermark
        position is static).

        Args:
            frames: List of input frames as numpy arrays.
            mask: Optional binary mask. If None, auto-detection is performed
                on the first frame and used for all subsequent frames.

        Returns:
            List of processed frames with watermarks removed.
        """
        if not frames:
            return []

        results = []
        effective_mask = mask

        # Auto-detect on first frame if needed
        if effective_mask is None and self.config.auto_detect:
            effective_mask = self.detect_watermark(frames[0])
            if effective_mask is not None:
                logger.info("Watermark detected on first frame, applying to batch")

        # Use custom mask if available
        if effective_mask is None and self._custom_mask is not None:
            effective_mask = self._custom_mask

        # Process all frames with the same mask
        for i, frame in enumerate(frames):
            try:
                result = self.remove_watermark(frame, effective_mask)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process frame {i}: {e}")
                results.append(frame.copy())

        return results

    def create_rectangular_mask(
        self,
        frame_shape: Tuple[int, int],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Create a rectangular mask for a specific region.

        Args:
            frame_shape: Shape of the frame as (height, width).
            x: X coordinate of top-left corner.
            y: Y coordinate of top-left corner.
            width: Width of the rectangle.
            height: Height of the rectangle.

        Returns:
            Binary mask with the specified region filled.
        """
        mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)

        # Clamp coordinates to valid range
        x1 = max(0, min(x, frame_shape[1]))
        y1 = max(0, min(y, frame_shape[0]))
        x2 = max(0, min(x + width, frame_shape[1]))
        y2 = max(0, min(y + height, frame_shape[0]))

        mask[y1:y2, x1:x2] = 255

        return mask

    def set_custom_mask(self, mask: np.ndarray) -> None:
        """Set a custom mask for watermark removal.

        Args:
            mask: Binary mask where 255 indicates watermark regions.
        """
        if len(mask.shape) == 3:
            # Convert to grayscale
            if self._cv2 is not None:
                mask = self._cv2.cvtColor(mask, self._cv2.COLOR_BGR2GRAY)
            else:
                mask = mask[:, :, 0]

        # Ensure binary
        self._custom_mask = (mask > 127).astype(np.uint8) * 255

    def clear_custom_mask(self) -> None:
        """Clear the custom mask."""
        self._custom_mask = None

    @property
    def backend(self) -> Optional[str]:
        """Get the current inpainting backend name.

        Returns:
            'lama', 'opencv', or None if no backend available.
        """
        return self._backend


# Convenience function for single-frame processing
def remove_watermark_from_image(
    image_path: Path,
    output_path: Path,
    mask_path: Optional[Path] = None,
    auto_detect: bool = True,
) -> bool:
    """Remove watermark from a single image file.

    Convenience function for one-off watermark removal.

    Args:
        image_path: Path to input image.
        output_path: Path for output image.
        mask_path: Optional path to mask image.
        auto_detect: Enable auto-detection if no mask provided.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import cv2

        config = WatermarkConfig(
            mask_path=mask_path,
            auto_detect=auto_detect,
        )
        remover = WatermarkRemover(config)

        if not remover.is_available():
            logger.error("No inpainting backend available")
            return False

        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return False

        # Process
        result = remover.remove_watermark(frame)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        return True

    except Exception as e:
        logger.error(f"Watermark removal failed: {e}")
        return False
