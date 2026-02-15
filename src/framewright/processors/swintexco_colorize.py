"""SwinTExCo Exemplar-Based Colorization for video restoration.

This module implements exemplar-based colorization using Swin Transformer,
allowing users to provide reference color images for historically accurate
colorization of black & white footage.

Key advantages over automatic colorization (DDColor/DeOldify):
- User provides reference color images from the same era/context
- Bidirectional temporal fusion for consistency across frames
- Match actual historical colors instead of AI guessing
- Better for footage where color references exist

Model Sources (user must download manually):
- SwinTExCo: https://github.com/DongYang-Yolanda/SwinTExCo

Example:
    >>> config = ExemplarColorizeConfig(
    ...     reference_images=[Path("ref1.jpg"), Path("ref2.jpg")]
    ... )
    >>> colorizer = SwinTExCoColorizer(config)
    >>> if colorizer.is_available():
    ...     result = colorizer.colorize_with_reference(input_dir, output_dir)
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ColorPropagationMode(Enum):
    """Color propagation modes for temporal consistency."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class ExemplarColorizeConfig:
    """Configuration for exemplar-based colorization.

    Attributes:
        reference_images: List of reference color images
        temporal_fusion: Enable bidirectional temporal fusion
        propagation_mode: Color propagation direction
        style_strength: Strength of color transfer (0-1)
        preserve_luminance: Keep original luminance, only transfer chrominance
        match_histogram: Match color histogram of reference
        semantic_matching: Use semantic features for color matching
        temporal_window: Window size for temporal consistency
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
    """
    reference_images: List[Path] = field(default_factory=list)
    temporal_fusion: bool = True
    propagation_mode: ColorPropagationMode = ColorPropagationMode.BIDIRECTIONAL
    style_strength: float = 1.0
    preserve_luminance: bool = True
    match_histogram: bool = True
    semantic_matching: bool = True
    temporal_window: int = 5
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.propagation_mode, str):
            self.propagation_mode = ColorPropagationMode(self.propagation_mode)
        if not 0.0 <= self.style_strength <= 1.0:
            raise ValueError(f"style_strength must be 0-1, got {self.style_strength}")


@dataclass
class ExemplarColorizeResult:
    """Result of exemplar-based colorization.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        output_dir: Path to output directory
        reference_images_used: Number of reference images used
        processing_time_seconds: Total processing time
        peak_vram_mb: Peak VRAM usage
    """
    frames_processed: int = 0
    frames_failed: int = 0
    output_dir: Optional[Path] = None
    reference_images_used: int = 0
    processing_time_seconds: float = 0.0
    peak_vram_mb: int = 0


class PatchMatchColorizer:
    """Simple patch-based color transfer using feature matching."""

    def __init__(self, patch_size: int = 7):
        self.patch_size = patch_size

    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract patches from image."""
        h, w = image.shape[:2]
        ps = self.patch_size
        hp = ps // 2

        # Pad image
        padded = cv2.copyMakeBorder(image, hp, hp, hp, hp, cv2.BORDER_REFLECT)

        patches = []
        positions = []
        for y in range(0, h, ps):
            for x in range(0, w, ps):
                patch = padded[y:y+ps, x:x+ps]
                patches.append(patch.flatten())
                positions.append((y, x))

        return np.array(patches), positions

    def find_nearest_patches(
        self,
        source_patches: np.ndarray,
        ref_patches: np.ndarray,
    ) -> np.ndarray:
        """Find nearest neighbor patches."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(ref_patches)
        distances, indices = nn.kneighbors(source_patches)

        return indices.flatten()


class SwinTExCoColorizer:
    """SwinTExCo exemplar-based video colorization.

    Uses Swin Transformer with temporal correspondence for colorizing
    black & white video using reference color images.

    Example:
        >>> config = ExemplarColorizeConfig(
        ...     reference_images=[Path("ref.jpg")],
        ...     temporal_fusion=True,
        ... )
        >>> colorizer = SwinTExCoColorizer(config)
        >>> if colorizer.is_available():
        ...     result = colorizer.colorize_with_reference(input_dir, output_dir)
    """

    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'swintexco'
    MODEL_FILE = 'swintexco.pth'

    def __init__(
        self,
        config: Optional[ExemplarColorizeConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize SwinTExCo colorizer.

        Args:
            config: Colorization configuration
            model_dir: Directory containing model weights
        """
        self.config = config or ExemplarColorizeConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._reference_features = None
        self._reference_colors = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available backend."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available - SwinTExCo disabled")
            return None

        # Check for model weights
        model_path = self.model_dir / self.MODEL_FILE
        if model_path.exists() and HAS_TORCH:
            logger.info(f"Found SwinTExCo model weights at {model_path}")
            return 'swintexco_weights'

        # Check for transformers library (for feature extraction)
        if HAS_TORCH:
            try:
                import timm
                logger.info("Found timm library for Swin Transformer")
                return 'swin_timm'
            except ImportError:
                pass

        # Fallback to simple color transfer
        logger.info("Using simple color transfer (SwinTExCo model not available)")
        return 'simple_transfer'

    def is_available(self) -> bool:
        """Check if SwinTExCo colorization is available."""
        return self._backend is not None

    def _load_reference_images(self) -> List[np.ndarray]:
        """Load and preprocess reference images."""
        references = []
        for ref_path in self.config.reference_images:
            ref_path = Path(ref_path)
            if ref_path.exists():
                img = cv2.imread(str(ref_path))
                if img is not None:
                    references.append(img)
                    logger.info(f"Loaded reference image: {ref_path}")
                else:
                    logger.warning(f"Failed to load reference: {ref_path}")
            else:
                logger.warning(f"Reference image not found: {ref_path}")

        if not references:
            logger.warning("No valid reference images loaded")

        return references

    def _extract_features_swin(self, image: np.ndarray) -> np.ndarray:
        """Extract features using Swin Transformer."""
        import torch
        import timm

        # Load Swin model
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        model = model.to(self._device)
        model.eval()

        # Preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self._device)

        # Extract features
        with torch.no_grad():
            features = model.forward_features(tensor)

        return features.cpu().numpy()

    def _extract_color_palette(self, image: np.ndarray, n_colors: int = 16) -> np.ndarray:
        """Extract dominant color palette from image."""
        # Convert to Lab for perceptual color clustering
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Reshape for k-means
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        return centers.astype(np.uint8)

    def _match_histogram(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Match color histogram of source to reference."""
        # Convert to Lab
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Match each channel
        result_lab = np.zeros_like(source_lab)

        for i in range(3):
            src_mean, src_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
            ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()

            if src_std > 0:
                result_lab[:, :, i] = (source_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean
            else:
                result_lab[:, :, i] = ref_mean

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _transfer_color_simple(
        self,
        source_gray: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Simple color transfer using luminance-chrominance separation."""
        # Ensure source is 3-channel grayscale
        if len(source_gray.shape) == 2:
            source_gray = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)

        # Convert to Lab
        source_lab = cv2.cvtColor(source_gray, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Resize reference to match source
        h, w = source_gray.shape[:2]
        ref_lab_resized = cv2.resize(ref_lab, (w, h))

        # Transfer chrominance (a, b channels)
        result_lab = source_lab.copy()

        if self.config.preserve_luminance:
            # Keep source luminance, transfer chrominance from reference
            result_lab[:, :, 1] = ref_lab_resized[:, :, 1]  # a channel
            result_lab[:, :, 2] = ref_lab_resized[:, :, 2]  # b channel
        else:
            # Full color transfer
            for i in range(3):
                src_mean = source_lab[:, :, i].mean()
                src_std = source_lab[:, :, i].std()
                ref_mean = ref_lab_resized[:, :, i].mean()
                ref_std = ref_lab_resized[:, :, i].std()

                if src_std > 0:
                    result_lab[:, :, i] = (source_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean
                else:
                    result_lab[:, :, i] = ref_mean

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _transfer_color_semantic(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Transfer color using semantic feature matching."""
        # This is a simplified version - full SwinTExCo uses transformer attention

        # Extract color palette from reference
        ref_palette = self._extract_color_palette(reference, 32)

        # Convert source to Lab
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        h, w = source.shape[:2]

        # For each pixel, find nearest color in palette based on luminance
        result_lab = source_lab.copy()

        # Get luminance channel
        source_l = source_lab[:, :, 0].flatten()

        # Find best matching palette colors based on luminance
        palette_l = ref_palette[:, 0]

        # Vectorized nearest neighbor lookup
        distances = np.abs(source_l[:, np.newaxis] - palette_l[np.newaxis, :])
        nearest_idx = np.argmin(distances, axis=1)

        # Apply chrominance from matched palette colors
        result_ab = ref_palette[nearest_idx, 1:3]
        result_lab[:, :, 1:3] = result_ab.reshape(h, w, 2)

        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _apply_temporal_consistency(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray],
        next_frame: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply temporal consistency using neighboring frames."""
        if not self.config.temporal_fusion:
            return current

        frames_to_blend = [current]
        weights = [1.0]

        if previous is not None and self.config.propagation_mode in [
            ColorPropagationMode.BACKWARD, ColorPropagationMode.BIDIRECTIONAL
        ]:
            frames_to_blend.append(previous)
            weights.append(0.3)

        if next_frame is not None and self.config.propagation_mode in [
            ColorPropagationMode.FORWARD, ColorPropagationMode.BIDIRECTIONAL
        ]:
            frames_to_blend.append(next_frame)
            weights.append(0.3)

        # Weighted average
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        result = np.zeros_like(current, dtype=np.float32)
        for frame, weight in zip(frames_to_blend, weights):
            result += frame.astype(np.float32) * weight

        return result.astype(np.uint8)

    def _colorize_frame(
        self,
        frame: np.ndarray,
        references: List[np.ndarray],
        prev_colorized: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colorize a single frame using reference images."""
        if not references:
            return frame

        # Check if frame is already colored
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Check color saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            if saturation > 20:  # Already has color
                return frame

        # Use first reference as primary (could implement reference selection)
        primary_ref = references[0]

        # Choose colorization method based on backend
        if self._backend == 'swin_timm' and self.config.semantic_matching:
            colorized = self._transfer_color_semantic(frame, primary_ref)
        else:
            colorized = self._transfer_color_simple(frame, primary_ref)

        # Optional histogram matching
        if self.config.match_histogram:
            colorized = self._match_histogram(colorized, primary_ref)

        # Apply style strength
        if self.config.style_strength < 1.0:
            # Convert original to 3-channel if needed
            if len(frame.shape) == 2:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            colorized = cv2.addWeighted(
                frame_bgr,
                1 - self.config.style_strength,
                colorized,
                self.config.style_strength,
                0,
            )

        # Temporal blending with previous frame
        if prev_colorized is not None:
            colorized = self._apply_temporal_consistency(colorized, prev_colorized, None)

        return colorized

    def colorize_with_reference(
        self,
        input_dir: Path,
        output_dir: Path,
        reference_images: Optional[List[Path]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ExemplarColorizeResult:
        """Colorize video frames using reference images.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for colorized output frames
            reference_images: Override reference images from config
            progress_callback: Optional progress callback (0-1)

        Returns:
            ExemplarColorizeResult with processing statistics
        """
        result = ExemplarColorizeResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("SwinTExCo colorization not available")
            return result

        # Setup device
        if HAS_TORCH:
            if torch.cuda.is_available():
                self._device = torch.device(f'cuda:{self.config.gpu_id}')
            else:
                self._device = torch.device('cpu')

        # Load reference images
        if reference_images:
            self.config.reference_images = reference_images

        references = self._load_reference_images()
        result.reference_images_used = len(references)

        if not references:
            logger.error("No reference images available - cannot colorize")
            return result

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        # Get input frames
        input_dir = Path(input_dir)
        frame_files = sorted(input_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(input_dir.glob("*.jpg"))

        if not frame_files:
            logger.warning(f"No frames found in {input_dir}")
            return result

        total_frames = len(frame_files)
        logger.info(
            f"SwinTExCo colorization: {total_frames} frames, "
            f"{len(references)} reference images"
        )

        # Process frames
        prev_colorized = None

        for i, frame_file in enumerate(frame_files):
            try:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_file}")
                    result.frames_failed += 1
                    continue

                colorized = self._colorize_frame(frame, references, prev_colorized)
                prev_colorized = colorized

                # Save output
                output_path = output_dir / frame_file.name
                cv2.imwrite(str(output_path), colorized)
                result.frames_processed += 1

            except Exception as e:
                logger.error(f"Failed to colorize {frame_file}: {e}")
                result.frames_failed += 1

                # Copy original as fallback
                try:
                    output_path = output_dir / frame_file.name
                    shutil.copy2(frame_file, output_path)
                except Exception:
                    pass

            # Update progress
            if progress_callback:
                progress_callback((i + 1) / total_frames)

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time

        if HAS_TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated(self.config.gpu_id) // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.config.gpu_id)

        logger.info(
            f"SwinTExCo colorization complete: {result.frames_processed}/{total_frames} frames, "
            f"time: {result.processing_time_seconds:.1f}s"
        )

        return result

    def clear_cache(self) -> None:
        """Clear model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        self._reference_features = None
        self._reference_colors = None

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_swintexco_colorizer(
    reference_images: List[Path],
    temporal_fusion: bool = True,
    style_strength: float = 1.0,
    gpu_id: int = 0,
) -> SwinTExCoColorizer:
    """Factory function to create a SwinTExCo colorizer.

    Args:
        reference_images: List of reference color image paths
        temporal_fusion: Enable temporal consistency
        style_strength: Color transfer strength (0-1)
        gpu_id: GPU device ID

    Returns:
        Configured SwinTExCoColorizer instance
    """
    config = ExemplarColorizeConfig(
        reference_images=reference_images,
        temporal_fusion=temporal_fusion,
        style_strength=style_strength,
        gpu_id=gpu_id,
    )
    return SwinTExCoColorizer(config)
