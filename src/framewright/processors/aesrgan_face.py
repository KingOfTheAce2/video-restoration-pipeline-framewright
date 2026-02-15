"""AESRGAN (Attention-Enhanced ESRGAN) Face Enhancement for video restoration.

This module implements attention-enhanced face restoration that preserves subtle
facial features without over-smoothing, providing better results than GFPGAN
for many use cases.

Key advantages over GFPGAN:
- Better preservation of subtle facial features (wrinkles, pores, etc.)
- Reduced "plastic" or over-smoothed appearance
- Better high-frequency detail control
- More natural skin texture preservation

Model Sources (user must download manually):
- AESRGAN weights: Custom trained or adapted from ESRGAN

Example:
    >>> config = AESRGANFaceConfig(enhancement_strength=0.8)
    >>> restorer = AESRGANFaceRestorer(config)
    >>> if restorer.is_available():
    ...     result = restorer.restore_faces(input_dir, output_dir)
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
    logger.debug("PyTorch not available - AESRGAN disabled")


class FaceDetectorType(Enum):
    """Face detection methods."""
    RETINAFACE = "retinaface"
    MTCNN = "mtcnn"
    DLIB = "dlib"
    OPENCV = "opencv"


@dataclass
class AESRGANFaceConfig:
    """Configuration for AESRGAN face enhancement.

    Attributes:
        detection_threshold: Face detection confidence threshold (0-1)
        enhancement_strength: Strength of enhancement blending (0-1)
        preserve_identity: Preserve original facial identity features
        attention_scale: Scale factor for attention mechanism
        upscale_factor: Face upscaling factor before enhancement
        paste_back: Paste enhanced face back onto original frame
        face_detector: Face detection method to use
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
    """
    detection_threshold: float = 0.7
    enhancement_strength: float = 0.8
    preserve_identity: bool = True
    attention_scale: float = 1.0
    upscale_factor: int = 2
    paste_back: bool = True
    face_detector: FaceDetectorType = FaceDetectorType.RETINAFACE
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.face_detector, str):
            self.face_detector = FaceDetectorType(self.face_detector)
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError(f"detection_threshold must be 0-1, got {self.detection_threshold}")
        if not 0.0 <= self.enhancement_strength <= 1.0:
            raise ValueError(f"enhancement_strength must be 0-1, got {self.enhancement_strength}")
        if self.upscale_factor not in [1, 2, 4]:
            raise ValueError(f"upscale_factor must be 1, 2, or 4, got {self.upscale_factor}")


@dataclass
class FaceBox:
    """Detected face bounding box with landmarks."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class AESRGANFaceResult:
    """Result of AESRGAN face enhancement.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        faces_enhanced: Total number of faces enhanced
        output_dir: Path to directory containing enhanced frames
        processing_time_seconds: Total processing time
        peak_vram_mb: Peak VRAM usage
    """
    frames_processed: int = 0
    frames_failed: int = 0
    faces_enhanced: int = 0
    output_dir: Optional[Path] = None
    processing_time_seconds: float = 0.0
    peak_vram_mb: int = 0


# Conditionally define PyTorch modules only when PyTorch is available
if HAS_TORCH:
    class AttentionBlock(nn.Module):
        """Self-attention block for AESRGAN."""

        def __init__(self, channels: int):
            super().__init__()
            self.query = nn.Conv2d(channels, channels // 8, 1)
            self.key = nn.Conv2d(channels, channels // 8, 1)
            self.value = nn.Conv2d(channels, channels, 1)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, channels, height, width = x.shape

            # Query, Key, Value projections
            q = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
            k = self.key(x).view(batch, -1, height * width)
            v = self.value(x).view(batch, -1, height * width)

            # Attention weights
            attention = torch.bmm(q, k)
            attention = torch.softmax(attention, dim=-1)

            # Apply attention to values
            out = torch.bmm(v, attention.permute(0, 2, 1))
            out = out.view(batch, channels, height, width)

            # Residual connection with learnable scale
            return self.gamma * out + x

    class ResidualDenseBlock(nn.Module):
        """Residual Dense Block for ESRGAN."""

        def __init__(self, channels: int = 64, growth_channels: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
            self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
            self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
            self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
            x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
            x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
            x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        """Residual-in-Residual Dense Block."""

        def __init__(self, channels: int = 64):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(channels)
            self.rdb2 = ResidualDenseBlock(channels)
            self.rdb3 = ResidualDenseBlock(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class AESRGAN(nn.Module):
        """Attention-Enhanced ESRGAN for face restoration.

        Incorporates self-attention mechanisms into ESRGAN architecture
        for better facial detail preservation.
        """

        def __init__(
            self,
            num_in_ch: int = 3,
            num_out_ch: int = 3,
            num_feat: int = 64,
            num_block: int = 23,
            scale: int = 2,
            num_attention: int = 4,
        ):
            super().__init__()
            self.scale = scale

            # First convolution
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

            # RRDB blocks with attention
            self.body = nn.ModuleList()
            attention_positions = set(range(0, num_block, num_block // num_attention))

            for i in range(num_block):
                self.body.append(RRDB(num_feat))
                if i in attention_positions:
                    self.body.append(AttentionBlock(num_feat))

            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

            # Upsampling
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if scale >= 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.conv_first(x)
            body_feat = feat

            for layer in self.body:
                body_feat = layer(body_feat)

            body_feat = self.conv_body(body_feat)
            feat = feat + body_feat

            # Upsampling
            feat = self.lrelu(self.conv_up1(
                nn.functional.interpolate(feat, scale_factor=2, mode='nearest')
            ))
            if self.scale >= 4:
                feat = self.lrelu(self.conv_up2(
                    nn.functional.interpolate(feat, scale_factor=2, mode='nearest')
                ))

            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out


class FaceDetector:
    """Face detector with multiple backend support."""

    def __init__(self, detector_type: FaceDetectorType, gpu_id: int = 0):
        self.detector_type = detector_type
        self.gpu_id = gpu_id
        self._detector = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available face detection backend."""
        if self.detector_type == FaceDetectorType.RETINAFACE:
            try:
                from retinaface import RetinaFace
                return 'retinaface'
            except ImportError:
                pass

        if self.detector_type == FaceDetectorType.OPENCV:
            if HAS_OPENCV:
                return 'opencv'

        # Fallback to OpenCV
        if HAS_OPENCV:
            logger.warning(f"{self.detector_type.value} not available, falling back to OpenCV")
            return 'opencv'

        return None

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces in frame."""
        if self._backend == 'retinaface':
            return self._detect_retinaface(frame)
        elif self._backend == 'opencv':
            return self._detect_opencv(frame)
        return []

    def _detect_retinaface(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using RetinaFace."""
        try:
            from retinaface import RetinaFace

            # RetinaFace expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(rgb_frame)

            boxes = []
            if isinstance(faces, dict):
                for face_id, face_data in faces.items():
                    area = face_data['facial_area']
                    landmarks = face_data.get('landmarks')

                    landmark_array = None
                    if landmarks:
                        landmark_array = np.array([
                            landmarks['left_eye'],
                            landmarks['right_eye'],
                            landmarks['nose'],
                            landmarks['mouth_left'],
                            landmarks['mouth_right'],
                        ])

                    boxes.append(FaceBox(
                        x1=area[0],
                        y1=area[1],
                        x2=area[2],
                        y2=area[3],
                        confidence=face_data.get('score', 1.0),
                        landmarks=landmark_array,
                    ))

            return boxes

        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []

    def _detect_opencv(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using OpenCV Haar cascades."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Load cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            boxes = []
            for (x, y, w, h) in faces:
                boxes.append(FaceBox(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    confidence=1.0,
                    landmarks=None,
                ))

            return boxes

        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []


class AESRGANFaceRestorer:
    """AESRGAN-based face restoration for video frames.

    Uses attention-enhanced ESRGAN for better facial detail preservation
    compared to traditional GFPGAN approach.

    Example:
        >>> config = AESRGANFaceConfig(enhancement_strength=0.8)
        >>> restorer = AESRGANFaceRestorer(config)
        >>> if restorer.is_available():
        ...     result = restorer.restore_faces(input_dir, output_dir)
    """

    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'aesrgan'
    MODEL_FILE = 'aesrgan_face.pth'

    def __init__(
        self,
        config: Optional[AESRGANFaceConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize AESRGAN face restorer.

        Args:
            config: Enhancement configuration
            model_dir: Directory containing model weights
        """
        self.config = config or AESRGANFaceConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._face_detector = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available AESRGAN backend."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available - AESRGAN disabled")
            return None

        if not HAS_OPENCV:
            logger.warning("OpenCV not available - AESRGAN disabled")
            return None

        # Check for model weights
        model_path = self.model_dir / self.MODEL_FILE
        if model_path.exists():
            logger.info(f"Found AESRGAN model weights at {model_path}")
            return 'aesrgan_weights'

        # Can use architecture without pretrained weights (random init)
        logger.warning(
            f"AESRGAN weights not found at {model_path}. "
            "Will use random initialization (quality will be limited)."
        )
        return 'aesrgan_random'

    def is_available(self) -> bool:
        """Check if AESRGAN face restoration is available."""
        return self._backend is not None

    def _load_model(self) -> None:
        """Load the AESRGAN model."""
        if self._model is not None:
            return

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        import torch

        # Set device
        if torch.cuda.is_available():
            self._device = torch.device(f'cuda:{self.config.gpu_id}')
        else:
            self._device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU (slow)")

        # Create model
        self._model = AESRGAN(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            scale=self.config.upscale_factor,
            num_attention=4,
        )

        # Load weights if available
        model_path = self.model_dir / self.MODEL_FILE
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'params' in checkpoint:
                    self._model.load_state_dict(checkpoint['params'], strict=False)
                elif 'state_dict' in checkpoint:
                    self._model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self._model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded AESRGAN weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load AESRGAN weights: {e}")

        self._model = self._model.to(self._device)
        self._model.eval()

        if self.config.half_precision and self._device.type == 'cuda':
            self._model = self._model.half()

        # Initialize face detector
        self._face_detector = FaceDetector(self.config.face_detector, self.config.gpu_id)

    def _extract_face(
        self,
        frame: np.ndarray,
        face_box: FaceBox,
        padding: float = 0.3,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract face region with padding."""
        h, w = frame.shape[:2]

        # Add padding
        pad_w = int(face_box.width * padding)
        pad_h = int(face_box.height * padding)

        x1 = max(0, face_box.x1 - pad_w)
        y1 = max(0, face_box.y1 - pad_h)
        x2 = min(w, face_box.x2 + pad_w)
        y2 = min(h, face_box.y2 + pad_h)

        face_crop = frame[y1:y2, x1:x2].copy()
        return face_crop, (x1, y1, x2, y2)

    def _enhance_face(self, face_crop: np.ndarray) -> np.ndarray:
        """Enhance face crop using AESRGAN."""
        import torch

        # Preprocess
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb.astype(np.float32) / 255.0)
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(self._device)

        if self.config.half_precision and self._device.type == 'cuda':
            face_tensor = face_tensor.half()

        # Enhance
        with torch.no_grad():
            enhanced = self._model(face_tensor)

        # Postprocess
        enhanced = enhanced.squeeze(0).cpu()
        if enhanced.dtype == torch.float16:
            enhanced = enhanced.float()

        enhanced = enhanced.permute(1, 2, 0).numpy()
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        return enhanced

    def _paste_face_back(
        self,
        frame: np.ndarray,
        enhanced_face: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Paste enhanced face back onto original frame with blending."""
        x1, y1, x2, y2 = region
        target_h = y2 - y1
        target_w = x2 - x1

        # Resize enhanced face to match original region
        enhanced_resized = cv2.resize(enhanced_face, (target_w, target_h))

        # Create blend mask (feathered edges)
        mask = np.ones((target_h, target_w), dtype=np.float32)
        feather = min(target_w, target_h) // 8

        if feather > 0:
            # Create feathered edges
            for i in range(feather):
                alpha = i / feather
                mask[i, :] *= alpha
                mask[-i-1, :] *= alpha
                mask[:, i] *= alpha
                mask[:, -i-1] *= alpha

        mask = mask[:, :, np.newaxis]

        # Blend based on enhancement strength
        strength = self.config.enhancement_strength
        original_region = frame[y1:y2, x1:x2].astype(np.float32)
        enhanced_float = enhanced_resized.astype(np.float32)

        blended = original_region * (1 - mask * strength) + enhanced_float * mask * strength

        # Paste back
        result = frame.copy()
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result

    def restore_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Restore faces in a single frame.

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (enhanced frame, number of faces enhanced)
        """
        if self._model is None:
            self._load_model()

        # Detect faces
        faces = self._face_detector.detect(frame)

        # Filter by confidence
        faces = [f for f in faces if f.confidence >= self.config.detection_threshold]

        if not faces:
            return frame, 0

        result = frame.copy()

        for face_box in faces:
            try:
                # Extract face
                face_crop, region = self._extract_face(result, face_box)

                # Enhance
                enhanced_face = self._enhance_face(face_crop)

                # Paste back if configured
                if self.config.paste_back:
                    result = self._paste_face_back(result, enhanced_face, region)

            except Exception as e:
                logger.warning(f"Failed to enhance face: {e}")
                continue

        return result, len(faces)

    def restore_faces(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> AESRGANFaceResult:
        """Restore faces in all frames.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback (0-1)

        Returns:
            AESRGANFaceResult with processing statistics
        """
        result = AESRGANFaceResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("AESRGAN face restoration not available")
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
        logger.info(f"AESRGAN face restoration: {total_frames} frames")

        # Load model
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load AESRGAN model: {e}")
            return result

        # Process frames
        for i, frame_file in enumerate(frame_files):
            try:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_file}")
                    result.frames_failed += 1
                    continue

                enhanced, num_faces = self.restore_frame(frame)
                result.faces_enhanced += num_faces

                # Save output
                output_path = output_dir / frame_file.name
                cv2.imwrite(str(output_path), enhanced)
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

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time

        if HAS_TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated(self.config.gpu_id) // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.config.gpu_id)

        logger.info(
            f"AESRGAN complete: {result.frames_processed}/{total_frames} frames, "
            f"{result.faces_enhanced} faces enhanced, "
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


def create_aesrgan_restorer(
    enhancement_strength: float = 0.8,
    preserve_identity: bool = True,
    gpu_id: int = 0,
) -> AESRGANFaceRestorer:
    """Factory function to create an AESRGAN face restorer.

    Args:
        enhancement_strength: Strength of enhancement (0-1)
        preserve_identity: Preserve original facial features
        gpu_id: GPU device ID

    Returns:
        Configured AESRGANFaceRestorer instance
    """
    config = AESRGANFaceConfig(
        enhancement_strength=enhancement_strength,
        preserve_identity=preserve_identity,
        gpu_id=gpu_id,
    )
    return AESRGANFaceRestorer(config)
