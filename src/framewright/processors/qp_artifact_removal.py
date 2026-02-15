"""QP-Aware Codec Artifact Removal for video restoration.

This module implements compression artifact removal that adapts to the
quantization parameter (QP) of the source video. Critical for restoring
heavily compressed sources like YouTube videos.

Key features:
- Auto-detection of QP from video metadata
- Targeted deblocking based on compression level
- Deringing for high-frequency artifact removal
- Edge-preserving smoothing

Model Sources (user must download manually):
- ARCNN: https://github.com/jianzhnie/ARCNN
- DnCNN compression: Pretrained denoising network

Example:
    >>> config = QPArtifactConfig(auto_detect_qp=True)
    >>> remover = QPArtifactRemover(config)
    >>> if remover.is_available():
    ...     result = remover.remove_artifacts(input_dir, output_dir)
"""

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.debug("PyTorch not available")


class ArtifactType(Enum):
    """Types of compression artifacts."""
    BLOCKING = "blocking"
    RINGING = "ringing"
    BANDING = "banding"
    MOSQUITO = "mosquito"
    ALL = "all"


class CompressionLevel(Enum):
    """Compression severity levels based on QP."""
    LIGHT = "light"      # QP < 23
    MODERATE = "moderate"  # QP 23-32
    HEAVY = "heavy"      # QP 32-40
    SEVERE = "severe"    # QP > 40


@dataclass
class QPArtifactConfig:
    """Configuration for QP-aware artifact removal.

    Attributes:
        auto_detect_qp: Automatically detect QP from video metadata
        manual_qp: Manual QP override (if auto_detect fails or disabled)
        strength_multiplier: Scale the removal strength (0.5-2.0)
        artifact_types: Types of artifacts to remove
        block_size: Codec block size (8 for H.264/H.265)
        preserve_edges: Enable edge-preserving processing
        preserve_texture: Reduce processing in textured areas
        gpu_id: GPU device ID for neural methods
        use_neural: Use neural network-based removal when available
    """
    auto_detect_qp: bool = True
    manual_qp: Optional[int] = None
    strength_multiplier: float = 1.0
    artifact_types: List[ArtifactType] = field(default_factory=lambda: [ArtifactType.ALL])
    block_size: int = 8
    preserve_edges: bool = True
    preserve_texture: bool = True
    gpu_id: int = 0
    use_neural: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.1 <= self.strength_multiplier <= 3.0:
            raise ValueError(f"strength_multiplier must be 0.1-3.0, got {self.strength_multiplier}")
        if self.block_size not in [4, 8, 16, 32]:
            raise ValueError(f"block_size must be 4, 8, 16, or 32, got {self.block_size}")
        if self.manual_qp is not None and not 0 <= self.manual_qp <= 51:
            raise ValueError(f"manual_qp must be 0-51, got {self.manual_qp}")


@dataclass
class QPArtifactResult:
    """Result of QP artifact removal.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        output_dir: Path to directory containing processed frames
        detected_qp: QP value that was detected/used
        compression_level: Detected compression severity
        processing_time_seconds: Total processing time
        artifacts_removed: Types of artifacts that were processed
    """
    frames_processed: int = 0
    frames_failed: int = 0
    output_dir: Optional[Path] = None
    detected_qp: Optional[int] = None
    compression_level: Optional[CompressionLevel] = None
    processing_time_seconds: float = 0.0
    artifacts_removed: List[str] = field(default_factory=list)


# Conditionally define PyTorch modules only when PyTorch is available
if HAS_TORCH:
    class ARCNN(nn.Module):
        """Artifact Reduction CNN for compression artifact removal.

        Simplified ARCNN architecture for removing JPEG/video compression artifacts.
        """

        def __init__(self):
            super().__init__()
            # Feature extraction
            self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=7, padding=3)
            self.conv3 = nn.Conv2d(32, 16, kernel_size=1, padding=0)
            self.conv4 = nn.Conv2d(16, 3, kernel_size=5, padding=2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv4(x)
            return x + residual

    class DnCNNDeblock(nn.Module):
        """DnCNN-based deblocking network."""

        def __init__(self, num_layers: int = 17, num_features: int = 64):
            super().__init__()
            layers = []

            # First layer
            layers.append(nn.Conv2d(3, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))

            # Middle layers
            for _ in range(num_layers - 2):
                layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(num_features))
                layers.append(nn.ReLU(inplace=True))

            # Last layer
            layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1, bias=False))

            self.dncnn = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            noise = self.dncnn(x)
            return x - noise


class QPArtifactRemover:
    """QP-aware codec artifact remover.

    Removes compression artifacts (blocking, ringing, banding) from video
    frames, with strength adapted to the detected quantization parameter.

    Example:
        >>> config = QPArtifactConfig(auto_detect_qp=True)
        >>> remover = QPArtifactRemover(config)
        >>> qp = remover.detect_qp_from_video(video_path)
        >>> result = remover.remove_artifacts(input_dir, output_dir, qp)
    """

    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'qp_artifact'

    MODEL_FILES = {
        'arcnn': 'arcnn.pth',
        'dncnn_deblock': 'dncnn_deblock.pth',
    }

    def __init__(
        self,
        config: Optional[QPArtifactConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize QP artifact remover.

        Args:
            config: Processing configuration
            model_dir: Directory containing model weights
        """
        self.config = config or QPArtifactConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available backend."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available - QP artifact removal disabled")
            return None

        # Check for neural backend
        if self.config.use_neural and HAS_TORCH:
            # Check for model weights
            arcnn_path = self.model_dir / self.MODEL_FILES['arcnn']
            dncnn_path = self.model_dir / self.MODEL_FILES['dncnn_deblock']

            if arcnn_path.exists():
                logger.info("Found ARCNN model weights")
                return 'arcnn'
            elif dncnn_path.exists():
                logger.info("Found DnCNN deblock model weights")
                return 'dncnn'
            else:
                logger.info("Neural models not found, will use architecture without pretrained weights")
                return 'neural_untrained'

        # Fallback to OpenCV-based methods
        logger.info("Using OpenCV-based artifact removal")
        return 'opencv'

    def is_available(self) -> bool:
        """Check if artifact removal is available."""
        return self._backend is not None

    def _load_model(self) -> None:
        """Load neural network model."""
        if self._model is not None:
            return

        if not HAS_TORCH or self._backend == 'opencv':
            return

        import torch

        # Set device
        if torch.cuda.is_available():
            self._device = torch.device(f'cuda:{self.config.gpu_id}')
        else:
            self._device = torch.device('cpu')

        # Load appropriate model
        if self._backend in ['arcnn', 'neural_untrained']:
            self._model = ARCNN()
            model_path = self.model_dir / self.MODEL_FILES['arcnn']
        else:
            self._model = DnCNNDeblock()
            model_path = self.model_dir / self.MODEL_FILES['dncnn_deblock']

        # Load weights if available
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self._model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self._model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded artifact removal model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}")

        self._model = self._model.to(self._device)
        self._model.eval()

    def detect_qp_from_video(self, video_path: Path) -> Optional[int]:
        """Detect average QP from video file using FFprobe.

        Args:
            video_path: Path to video file

        Returns:
            Estimated average QP, or None if detection fails
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return None

        try:
            # Try to get QP from FFprobe
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'frame=pict_type,qp',
                '-of', 'json',
                '-read_intervals', '%+#50',  # Analyze first 50 frames
                str(video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                frames = data.get('frames', [])

                qp_values = []
                for frame in frames:
                    qp = frame.get('qp')
                    if qp is not None:
                        qp_values.append(int(qp))

                if qp_values:
                    avg_qp = int(np.mean(qp_values))
                    logger.info(f"Detected average QP: {avg_qp} (from {len(qp_values)} frames)")
                    return avg_qp

            # Fallback: estimate from bitrate
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=bit_rate,width,height',
                '-of', 'json',
                str(video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [])

                if streams:
                    stream = streams[0]
                    bitrate = int(stream.get('bit_rate', 0))
                    width = int(stream.get('width', 1920))
                    height = int(stream.get('height', 1080))

                    if bitrate > 0:
                        # Estimate QP from bitrate per pixel
                        pixels = width * height
                        bpp = bitrate / (pixels * 30)  # Assume 30fps

                        # Rough QP estimation based on bits per pixel
                        if bpp > 0.5:
                            estimated_qp = 18
                        elif bpp > 0.2:
                            estimated_qp = 23
                        elif bpp > 0.1:
                            estimated_qp = 28
                        elif bpp > 0.05:
                            estimated_qp = 34
                        else:
                            estimated_qp = 40

                        logger.info(f"Estimated QP from bitrate: {estimated_qp} (bpp: {bpp:.3f})")
                        return estimated_qp

        except subprocess.TimeoutExpired:
            logger.warning("FFprobe timed out during QP detection")
        except Exception as e:
            logger.warning(f"Failed to detect QP: {e}")

        return None

    def _get_compression_level(self, qp: int) -> CompressionLevel:
        """Determine compression level from QP."""
        if qp < 23:
            return CompressionLevel.LIGHT
        elif qp < 32:
            return CompressionLevel.MODERATE
        elif qp < 40:
            return CompressionLevel.HEAVY
        else:
            return CompressionLevel.SEVERE

    def _get_strength_for_qp(self, qp: int) -> float:
        """Calculate processing strength based on QP."""
        # Higher QP = more artifacts = more aggressive processing
        base_strength = (qp - 15) / 35.0  # Normalize to ~0-1 range
        base_strength = np.clip(base_strength, 0.1, 1.0)
        return base_strength * self.config.strength_multiplier

    def _remove_blocking_opencv(
        self,
        frame: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Remove blocking artifacts using OpenCV."""
        # Bilateral filter for edge-preserving smoothing
        d = int(5 + strength * 10)  # Filter diameter
        sigma_color = 30 + strength * 70
        sigma_space = 30 + strength * 70

        result = cv2.bilateralFilter(
            frame,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        )

        # Additional deblocking at block boundaries
        if self.config.block_size > 0:
            h, w = frame.shape[:2]
            block_size = self.config.block_size

            # Create block boundary mask
            mask = np.zeros((h, w), dtype=np.float32)
            for y in range(0, h, block_size):
                mask[max(0, y-1):min(h, y+2), :] = 1.0
            for x in range(0, w, block_size):
                mask[:, max(0, x-1):min(w, x+2)] = 1.0

            # Apply stronger smoothing at block boundaries
            boundary_smooth = cv2.GaussianBlur(frame, (5, 5), 0)
            mask_3ch = mask[:, :, np.newaxis]
            result = (result * (1 - mask_3ch * strength * 0.5) +
                     boundary_smooth * mask_3ch * strength * 0.5).astype(np.uint8)

        return result

    def _remove_ringing_opencv(
        self,
        frame: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Remove ringing artifacts using OpenCV."""
        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to get ringing region
        kernel = np.ones((3, 3), np.uint8)
        ring_region = cv2.dilate(edges, kernel, iterations=2)
        ring_region = cv2.subtract(ring_region, edges)

        # Apply localized smoothing in ringing regions
        smoothed = cv2.GaussianBlur(frame, (5, 5), 0)

        ring_mask = (ring_region > 0).astype(np.float32)
        ring_mask = cv2.GaussianBlur(ring_mask, (5, 5), 0)
        ring_mask = ring_mask[:, :, np.newaxis] * strength

        result = (frame * (1 - ring_mask) + smoothed * ring_mask).astype(np.uint8)

        return result

    def _remove_banding_opencv(
        self,
        frame: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Remove color banding artifacts using OpenCV."""
        # Add subtle dithering to break up banding
        noise = np.random.normal(0, 1 + strength * 2, frame.shape).astype(np.float32)

        # Apply in smooth gradient areas (low variance regions)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        variance = cv2.Laplacian(gray, cv2.CV_32F)
        variance = np.abs(variance)

        # Normalize and invert (high variance = low mask)
        mask = 1.0 - np.clip(variance / 30.0, 0, 1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask[:, :, np.newaxis]

        result = frame.astype(np.float32) + noise * mask * strength
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _remove_artifacts_neural(
        self,
        frame: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Remove artifacts using neural network."""
        import torch

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self._device)

        # Process
        with torch.no_grad():
            output = self._model(tensor)

        # Postprocess
        output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Blend based on strength
        if strength < 1.0:
            result = cv2.addWeighted(frame, 1 - strength, result, strength, 0)

        return result

    def _process_frame(
        self,
        frame: np.ndarray,
        qp: int,
    ) -> np.ndarray:
        """Process a single frame to remove artifacts."""
        strength = self._get_strength_for_qp(qp)

        # Use neural method if available
        if self._backend in ['arcnn', 'dncnn', 'neural_untrained'] and self._model is not None:
            return self._remove_artifacts_neural(frame, strength)

        # OpenCV fallback
        result = frame.copy()

        artifact_types = self.config.artifact_types
        if ArtifactType.ALL in artifact_types:
            artifact_types = [ArtifactType.BLOCKING, ArtifactType.RINGING, ArtifactType.BANDING]

        if ArtifactType.BLOCKING in artifact_types:
            result = self._remove_blocking_opencv(result, strength)

        if ArtifactType.RINGING in artifact_types:
            result = self._remove_ringing_opencv(result, strength)

        if ArtifactType.BANDING in artifact_types:
            result = self._remove_banding_opencv(result, strength)

        return result

    def remove_artifacts(
        self,
        input_dir: Path,
        output_dir: Path,
        detected_qp: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> QPArtifactResult:
        """Remove compression artifacts from video frames.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for processed output frames
            detected_qp: Pre-detected QP (or None to use config)
            progress_callback: Optional progress callback (0-1)

        Returns:
            QPArtifactResult with processing statistics
        """
        result = QPArtifactResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("QP artifact removal not available")
            return result

        # Determine QP to use
        qp = detected_qp
        if qp is None:
            qp = self.config.manual_qp
        if qp is None:
            qp = 28  # Default assumption for web video
            logger.warning(f"No QP detected, using default: {qp}")

        result.detected_qp = qp
        result.compression_level = self._get_compression_level(qp)

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
            f"QP artifact removal: {total_frames} frames, "
            f"QP={qp} ({result.compression_level.value}), "
            f"strength={self._get_strength_for_qp(qp):.2f}"
        )

        # Load neural model if using
        if self._backend in ['arcnn', 'dncnn', 'neural_untrained']:
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Failed to load neural model, using OpenCV: {e}")
                self._backend = 'opencv'

        # Process frames
        for i, frame_file in enumerate(frame_files):
            try:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_file}")
                    result.frames_failed += 1
                    continue

                processed = self._process_frame(frame, qp)

                # Save output
                output_path = output_dir / frame_file.name
                cv2.imwrite(str(output_path), processed)
                result.frames_processed += 1

            except Exception as e:
                logger.error(f"Failed to process {frame_file}: {e}")
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

        # Record results
        result.processing_time_seconds = time.time() - start_time
        result.artifacts_removed = [at.value for at in self.config.artifact_types]

        logger.info(
            f"QP artifact removal complete: {result.frames_processed}/{total_frames} frames, "
            f"time: {result.processing_time_seconds:.1f}s"
        )

        return result

    def clear_cache(self) -> None:
        """Clear model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_qp_artifact_remover(
    auto_detect: bool = True,
    strength: float = 1.0,
    use_neural: bool = True,
    gpu_id: int = 0,
) -> QPArtifactRemover:
    """Factory function to create a QP artifact remover.

    Args:
        auto_detect: Enable automatic QP detection
        strength: Processing strength multiplier
        use_neural: Use neural network methods when available
        gpu_id: GPU device ID

    Returns:
        Configured QPArtifactRemover instance
    """
    config = QPArtifactConfig(
        auto_detect_qp=auto_detect,
        strength_multiplier=strength,
        use_neural=use_neural,
        gpu_id=gpu_id,
    )
    return QPArtifactRemover(config)
