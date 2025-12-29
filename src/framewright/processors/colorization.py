"""Colorization processor for black & white video restoration.

Supports multiple colorization models:
- DeOldify: Deep learning colorization with artistic style options
- DDColor: High-quality dual decoder colorization

Automatically detects grayscale frames and skips already colored content.
"""

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np

logger = logging.getLogger(__name__)

# Model download URLs
DEOLDIFY_MODEL_URLS = {
    'artistic': 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth',
    'stable': 'https://data.deepai.org/deoldify/ColorizeStable_gen.pth',
    'video': 'https://data.deepai.org/deoldify/ColorizeVideo_gen.pth',
}

DDCOLOR_MODEL_URLS = {
    'modelscope': 'https://modelscope.cn/api/v1/models/damo/cv_ddcolor_image-colorization/repo?Revision=master&FilePath=pytorch_model.pt',
    'huggingface': 'https://huggingface.co/piddnad/DDColor/resolve/main/ddcolor_modelscope.pth',
}


class ColorModel(Enum):
    """Available colorization models."""
    DEOLDIFY = "deoldify"
    DDCOLOR = "ddcolor"


class ArtisticStyle(Enum):
    """DeOldify artistic style presets."""
    ARTISTIC = "artistic"  # More vibrant, artistic interpretation
    STABLE = "stable"      # More conservative, stable colors
    VIDEO = "video"        # Optimized for video with temporal consistency


@dataclass
class ColorizationConfig:
    """Configuration for colorization processing.

    Attributes:
        model: Colorization model to use (DEOLDIFY or DDCOLOR)
        strength: Color saturation strength (0.0 to 1.0)
        artistic_style: DeOldify artistic style preset (ignored for DDColor)
        render_factor: Render resolution factor (higher = better quality, slower)
        skip_colored: Whether to skip frames that are already colored
        color_threshold: Threshold for detecting colored vs grayscale frames
    """
    model: ColorModel = ColorModel.DEOLDIFY
    strength: float = 1.0
    artistic_style: ArtisticStyle = ArtisticStyle.ARTISTIC
    render_factor: int = 35
    skip_colored: bool = True
    color_threshold: float = 10.0


@dataclass
class ColorizationResult:
    """Result of colorization processing."""
    frames_processed: int = 0
    frames_colorized: int = 0
    frames_skipped: int = 0
    failed_frames: int = 0
    output_dir: Optional[Path] = None


class Colorizer:
    """Video frame colorization using DeOldify or DDColor.

    This processor colorizes black & white video frames using
    state-of-the-art deep learning models.

    Supports multiple backends:
    - DeOldify (PyTorch implementation)
    - DDColor (Dual Decoder architecture)

    Falls back gracefully if colorization tools aren't installed.

    Example:
        >>> config = ColorizationConfig(model=ColorModel.DEOLDIFY, strength=0.8)
        >>> colorizer = Colorizer(config)
        >>> if colorizer.is_available():
        ...     result = colorizer.colorize_frame(frame)
    """

    # Default model directory: ~/.framewright/models/
    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models'

    def __init__(
        self,
        config: Optional[ColorizationConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize colorizer.

        Args:
            config: Colorization configuration
            model_dir: Directory for storing model weights
                      Defaults to ~/.framewright/models/
        """
        self.config = config or ColorizationConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available colorization backend."""
        if self.config.model == ColorModel.DEOLDIFY:
            return self._detect_deoldify_backend()
        elif self.config.model == ColorModel.DDCOLOR:
            return self._detect_ddcolor_backend()
        return None

    def _detect_deoldify_backend(self) -> Optional[str]:
        """Detect DeOldify backend availability."""
        # Check for DeOldify Python module
        try:
            from deoldify import device as deoldify_device
            from deoldify.visualize import get_image_colorizer
            logger.info("Found DeOldify Python module")
            return 'deoldify_module'
        except ImportError:
            pass

        # Check for FastAI with manual DeOldify setup
        try:
            import torch
            import fastai
            model_path = self._get_deoldify_model_path()
            if model_path and model_path.exists():
                logger.info("Found DeOldify model weights with FastAI")
                return 'deoldify_weights'
        except ImportError:
            pass

        logger.warning(
            "DeOldify not available. Install with: "
            "pip install deoldify fastai"
        )
        return None

    def _detect_ddcolor_backend(self) -> Optional[str]:
        """Detect DDColor backend availability."""
        try:
            import torch
            # Check if modelscope is available
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                logger.info("Found ModelScope DDColor backend")
                return 'ddcolor_modelscope'
            except ImportError:
                pass

            # Check for DDColor model weights
            model_path = self._get_ddcolor_model_path()
            if model_path and model_path.exists():
                logger.info("Found DDColor model weights")
                return 'ddcolor_weights'

        except ImportError:
            pass

        logger.warning(
            "DDColor not available. Install with: "
            "pip install torch torchvision modelscope"
        )
        return None

    def _get_deoldify_model_path(self) -> Optional[Path]:
        """Get path to DeOldify model weights.

        Model path: ~/.framewright/models/deoldify/
        """
        style_map = {
            ArtisticStyle.ARTISTIC: 'ColorizeArtistic_gen.pth',
            ArtisticStyle.STABLE: 'ColorizeStable_gen.pth',
            ArtisticStyle.VIDEO: 'ColorizeVideo_gen.pth',
        }
        model_name = style_map.get(self.config.artistic_style, 'ColorizeArtistic_gen.pth')
        model_path = self.model_dir / 'deoldify' / model_name

        if model_path.exists():
            return model_path
        return None

    def _get_ddcolor_model_path(self) -> Optional[Path]:
        """Get path to DDColor model weights.

        Model path: ~/.framewright/models/ddcolor/
        """
        model_path = self.model_dir / 'ddcolor' / 'ddcolor_modelscope.pth'
        if model_path.exists():
            return model_path

        # Also check for alternative naming
        alt_path = self.model_dir / 'ddcolor' / 'pytorch_model.pt'
        if alt_path.exists():
            return alt_path

        return None

    def is_available(self) -> bool:
        """Check if colorization is available.

        Returns:
            True if a colorization backend is available
        """
        return self._backend is not None

    def download_model(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download model weights if not present.

        Downloads model weights to:
        - DeOldify: ~/.framewright/models/deoldify/
        - DDColor: ~/.framewright/models/ddcolor/

        Args:
            progress_callback: Optional callback for download progress (0.0 to 1.0)

        Returns:
            True if model is available after download attempt
        """
        if self.config.model == ColorModel.DEOLDIFY:
            return self._download_deoldify_model(progress_callback)
        elif self.config.model == ColorModel.DDCOLOR:
            return self._download_ddcolor_model(progress_callback)
        return False

    def _download_deoldify_model(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download DeOldify model weights."""
        style_map = {
            ArtisticStyle.ARTISTIC: 'artistic',
            ArtisticStyle.STABLE: 'stable',
            ArtisticStyle.VIDEO: 'video',
        }
        style_key = style_map.get(self.config.artistic_style, 'artistic')
        url = DEOLDIFY_MODEL_URLS.get(style_key)

        if not url:
            logger.error(f"No download URL for DeOldify style: {style_key}")
            return False

        model_dir = self.model_dir / 'deoldify'
        model_dir.mkdir(parents=True, exist_ok=True)

        model_name = f'Colorize{style_key.capitalize()}_gen.pth'
        model_path = model_dir / model_name

        if model_path.exists():
            logger.info(f"DeOldify model already exists: {model_path}")
            self._backend = self._detect_backend()
            return self.is_available()

        logger.info(f"Downloading DeOldify {style_key} model from {url}...")

        try:
            def reporthook(block_num, block_size, total_size):
                if progress_callback and total_size > 0:
                    progress = min(1.0, block_num * block_size / total_size)
                    progress_callback(progress)

            urlretrieve(url, model_path, reporthook=reporthook)
            logger.info(f"DeOldify model downloaded to {model_path}")

            self._backend = self._detect_backend()
            return self.is_available()

        except Exception as e:
            logger.error(f"Failed to download DeOldify model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def _download_ddcolor_model(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download DDColor model weights."""
        url = DDCOLOR_MODEL_URLS.get('huggingface')

        if not url:
            logger.error("No download URL for DDColor model")
            return False

        model_dir = self.model_dir / 'ddcolor'
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / 'ddcolor_modelscope.pth'

        if model_path.exists():
            logger.info(f"DDColor model already exists: {model_path}")
            self._backend = self._detect_backend()
            return self.is_available()

        logger.info(f"Downloading DDColor model from {url}...")

        try:
            def reporthook(block_num, block_size, total_size):
                if progress_callback and total_size > 0:
                    progress = min(1.0, block_num * block_size / total_size)
                    progress_callback(progress)

            urlretrieve(url, model_path, reporthook=reporthook)
            logger.info(f"DDColor model downloaded to {model_path}")

            self._backend = self._detect_backend()
            return self.is_available()

        except Exception as e:
            logger.error(f"Failed to download DDColor model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def is_grayscale(self, frame: np.ndarray) -> bool:
        """Detect if a frame is grayscale or already colored.

        Args:
            frame: Input frame as numpy array (BGR or RGB format)

        Returns:
            True if the frame appears to be grayscale
        """
        if len(frame.shape) == 2:
            # Already single channel
            return True

        if frame.shape[2] == 1:
            # Single channel with explicit dimension
            return True

        if frame.shape[2] < 3:
            return True

        # Compare color channels
        # For grayscale images, R, G, B values should be very similar
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

        # Calculate average difference between channels
        diff_rg = np.abs(r.astype(np.float32) - g.astype(np.float32)).mean()
        diff_rb = np.abs(r.astype(np.float32) - b.astype(np.float32)).mean()
        diff_gb = np.abs(g.astype(np.float32) - b.astype(np.float32)).mean()

        avg_diff = (diff_rg + diff_rb + diff_gb) / 3.0

        return avg_diff < self.config.color_threshold

    def colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Colorize a single frame.

        Args:
            frame: Input frame as numpy array (BGR format, as from cv2.imread)

        Returns:
            Colorized frame as numpy array (BGR format)
        """
        if not self._backend:
            logger.warning("No colorization backend available, returning original")
            return frame

        # Check if already colored
        if self.config.skip_colored and not self.is_grayscale(frame):
            logger.debug("Frame already colored, skipping")
            return frame

        if self._backend == 'deoldify_module':
            return self._colorize_deoldify_module(frame)
        elif self._backend == 'deoldify_weights':
            return self._colorize_deoldify_weights(frame)
        elif self._backend == 'ddcolor_modelscope':
            return self._colorize_ddcolor_modelscope(frame)
        elif self._backend == 'ddcolor_weights':
            return self._colorize_ddcolor_weights(frame)
        else:
            logger.warning(f"Unknown backend: {self._backend}")
            return frame

    def _colorize_deoldify_module(self, frame: np.ndarray) -> np.ndarray:
        """Colorize using DeOldify Python module."""
        try:
            import tempfile
            import cv2
            from deoldify import device as deoldify_device
            from deoldify.visualize import get_image_colorizer

            # Initialize colorizer on first use
            if self._model is None:
                deoldify_device.set(device=0)  # GPU if available
                if self.config.artistic_style == ArtisticStyle.ARTISTIC:
                    self._model = get_image_colorizer(artistic=True)
                else:
                    self._model = get_image_colorizer(artistic=False)

            # DeOldify works with file paths, so save temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), frame)

            # Colorize
            result_path = self._model.plot_transformed_image(
                str(tmp_path),
                render_factor=self.config.render_factor,
                compare=False,
            )

            # Read result
            result = cv2.imread(str(result_path))

            # Cleanup
            tmp_path.unlink(missing_ok=True)
            if result_path and Path(result_path).exists():
                Path(result_path).unlink(missing_ok=True)

            # Apply strength blending
            if self.config.strength < 1.0:
                result = self._apply_strength_blending(frame, result)

            return result

        except Exception as e:
            logger.error(f"DeOldify colorization failed: {e}")
            return frame

    def _colorize_deoldify_weights(self, frame: np.ndarray) -> np.ndarray:
        """Colorize using DeOldify weights with FastAI."""
        try:
            import cv2
            import torch

            # This is a simplified implementation
            # Full implementation would require loading the actual DeOldify architecture
            logger.warning("DeOldify weights-only mode not fully implemented")
            return frame

        except Exception as e:
            logger.error(f"DeOldify weights colorization failed: {e}")
            return frame

    def _colorize_ddcolor_modelscope(self, frame: np.ndarray) -> np.ndarray:
        """Colorize using DDColor via ModelScope."""
        try:
            import cv2
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            # Initialize model on first use
            if self._model is None:
                self._model = pipeline(
                    Tasks.image_colorization,
                    model='damo/cv_ddcolor_image-colorization'
                )

            # DDColor expects RGB input
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run colorization
            result = self._model(rgb_frame)
            colorized = result['output_img']

            # Convert back to BGR
            bgr_result = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)

            # Apply strength blending
            if self.config.strength < 1.0:
                bgr_result = self._apply_strength_blending(frame, bgr_result)

            return bgr_result

        except Exception as e:
            logger.error(f"DDColor ModelScope colorization failed: {e}")
            return frame

    def _colorize_ddcolor_weights(self, frame: np.ndarray) -> np.ndarray:
        """Colorize using DDColor weights directly."""
        try:
            import cv2
            import torch

            # Initialize model on first use
            if self._model is None:
                model_path = self._get_ddcolor_model_path()
                if not model_path:
                    logger.error("DDColor model not found")
                    return frame

                # Detect device
                self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )

                # Note: This is a simplified loader - real implementation
                # would need the actual DDColor model architecture
                logger.warning("DDColor weights-only mode not fully implemented")
                return frame

            return frame

        except Exception as e:
            logger.error(f"DDColor weights colorization failed: {e}")
            return frame

    def _apply_strength_blending(
        self,
        original: np.ndarray,
        colorized: np.ndarray
    ) -> np.ndarray:
        """Apply strength blending between original and colorized frame.

        Args:
            original: Original grayscale frame (BGR format)
            colorized: Colorized frame (BGR format)

        Returns:
            Blended frame based on config.strength
        """
        try:
            import cv2

            # Convert grayscale to 3-channel for blending
            if self.is_grayscale(original):
                gray_3ch = cv2.cvtColor(
                    cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                    cv2.COLOR_GRAY2BGR
                )
            else:
                gray_3ch = original

            # Ensure same size
            if colorized.shape != gray_3ch.shape:
                colorized = cv2.resize(colorized, (gray_3ch.shape[1], gray_3ch.shape[0]))

            return cv2.addWeighted(
                colorized, self.config.strength,
                gray_3ch, 1.0 - self.config.strength,
                0
            )
        except Exception as e:
            logger.error(f"Strength blending failed: {e}")
            return colorized

    def colorize_batch(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Colorize a batch of frames.

        Args:
            frames: List of input frames as numpy arrays (BGR format)
            progress_callback: Optional callback for progress (0.0 to 1.0)

        Returns:
            List of colorized frames
        """
        results = []
        total = len(frames)

        for i, frame in enumerate(frames):
            colorized = self.colorize_frame(frame)
            results.append(colorized)

            if progress_callback:
                progress_callback((i + 1) / total)

        return results

    def colorize_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ColorizationResult:
        """Colorize all frames in a directory.

        Args:
            input_dir: Directory containing input frames (PNG/JPG)
            output_dir: Directory for output frames
            progress_callback: Optional callback for progress (0.0 to 1.0)

        Returns:
            ColorizationResult with processing statistics
        """
        result = ColorizationResult(output_dir=output_dir)

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all frames
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        if not self._backend:
            logger.warning("Colorization not available, copying frames")
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            result.frames_processed = len(frames)
            result.frames_skipped = len(frames)
            return result

        logger.info(
            f"Colorizing {len(frames)} frames using {self.config.model.value}"
        )

        if progress_callback:
            progress_callback(0.0)

        try:
            import cv2

            for i, frame_path in enumerate(frames):
                try:
                    # Read frame
                    frame = cv2.imread(str(frame_path))

                    if frame is None:
                        logger.warning(f"Failed to read frame: {frame_path}")
                        result.failed_frames += 1
                        continue

                    # Check if already colored
                    if self.config.skip_colored and not self.is_grayscale(frame):
                        shutil.copy(frame_path, output_dir / frame_path.name)
                        result.frames_skipped += 1
                        result.frames_processed += 1
                    else:
                        # Colorize
                        colorized = self.colorize_frame(frame)

                        # Save output
                        output_path = output_dir / frame_path.name
                        cv2.imwrite(str(output_path), colorized)

                        result.frames_colorized += 1
                        result.frames_processed += 1

                except Exception as e:
                    logger.debug(f"Failed to colorize {frame_path.name}: {e}")
                    shutil.copy(frame_path, output_dir / frame_path.name)
                    result.failed_frames += 1
                    result.frames_processed += 1

                if progress_callback:
                    progress_callback((i + 1) / len(frames))

        except ImportError:
            logger.error("OpenCV not available for frame processing")
            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)
            result.frames_processed = len(frames)
            result.failed_frames = len(frames)

        return result


class AutoColorizer:
    """Automatic colorization that detects and colorizes B&W content.

    Analyzes video to determine if colorization is needed and
    applies appropriate settings.
    """

    def __init__(
        self,
        colorizer: Optional[Colorizer] = None,
        sample_rate: int = 30,
        bw_threshold: float = 0.7,
    ):
        """Initialize auto colorizer.

        Args:
            colorizer: Optional custom colorizer instance
            sample_rate: Sample every Nth frame for analysis
            bw_threshold: Fraction of B&W frames to trigger colorization
        """
        self.colorizer = colorizer or Colorizer()
        self.sample_rate = sample_rate
        self.bw_threshold = bw_threshold

    def analyze_content(
        self,
        frames_dir: Path,
    ) -> Tuple[bool, float]:
        """Analyze frames to determine if colorization is needed.

        Args:
            frames_dir: Directory containing frames

        Returns:
            Tuple of (needs_colorization, bw_ratio)
        """
        frames = sorted(Path(frames_dir).glob("*.png"))
        if not frames:
            frames = sorted(Path(frames_dir).glob("*.jpg"))

        if not frames:
            return False, 0.0

        sample_frames = frames[::self.sample_rate][:50]  # Max 50 samples
        bw_count = 0

        try:
            import cv2

            for frame_path in sample_frames:
                frame = cv2.imread(str(frame_path))
                if frame is not None and self.colorizer.is_grayscale(frame):
                    bw_count += 1

        except ImportError:
            logger.warning("OpenCV not available for content analysis")
            return False, 0.0

        bw_ratio = bw_count / len(sample_frames) if sample_frames else 0.0
        needs_colorization = bw_ratio >= self.bw_threshold

        logger.info(
            f"Content analysis: {bw_ratio:.1%} B&W frames, "
            f"colorization {'recommended' if needs_colorization else 'not needed'}"
        )

        return needs_colorization, bw_ratio

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        force: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[ColorizationResult, bool]:
        """Automatically analyze and colorize if needed.

        Args:
            input_dir: Input frames directory
            output_dir: Output frames directory
            force: Force colorization even if not detected as B&W
            progress_callback: Progress callback

        Returns:
            Tuple of (colorization result, was_colorized)
        """
        if progress_callback:
            progress_callback(0.05)

        # Analyze content
        needs_colorization, bw_ratio = self.analyze_content(input_dir)

        if progress_callback:
            progress_callback(0.1)

        if not needs_colorization and not force:
            logger.info("Content appears colored, skipping colorization")
            # Copy frames
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            frames = sorted(Path(input_dir).glob("*.png"))
            if not frames:
                frames = sorted(Path(input_dir).glob("*.jpg"))

            for frame in frames:
                shutil.copy(frame, output_dir / frame.name)

            result = ColorizationResult(
                frames_processed=len(frames),
                frames_skipped=len(frames),
                output_dir=output_dir,
            )
            if progress_callback:
                progress_callback(1.0)
            return result, False

        # Colorize
        def scaled_progress(p):
            if progress_callback:
                progress_callback(0.1 + p * 0.9)

        result = self.colorizer.colorize_directory(
            input_dir,
            output_dir,
            scaled_progress,
        )

        return result, True
