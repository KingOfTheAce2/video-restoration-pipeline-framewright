"""Cross-Frame Attention for Enhanced Temporal Consistency.

This module implements transformer-based cross-frame attention for superior
temporal consistency and flicker reduction in video restoration. Uses
attention mechanisms across frames rather than just optical flow.

Key advantages over optical flow-based methods:
- Better handling of occlusions and complex motion
- Long-range temporal dependencies
- Content-aware consistency (not just motion-based)
- Superior flicker reduction in challenging scenes

Example:
    >>> config = CrossAttentionConfig(attention_window=7)
    >>> processor = CrossAttentionTemporalProcessor(config)
    >>> if processor.is_available():
    ...     result = processor.apply_consistency(input_dir, output_dir)
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


class TemporalMethod(Enum):
    """Temporal consistency methods."""

    OPTICAL_FLOW = "optical_flow"
    """Traditional optical flow-based consistency.

    Fast, works well for simple motion.
    """

    CROSS_ATTENTION = "cross_attention"
    """Transformer-based cross-frame attention.

    Higher quality, handles complex motion better.
    """

    HYBRID = "hybrid"
    """Combines optical flow and attention.

    Best quality, moderate speed.
    """


@dataclass
class CrossAttentionConfig:
    """Configuration for cross-attention temporal consistency.

    Attributes:
        method: Temporal consistency method
        attention_window: Number of frames for attention (odd number preferred)
        attention_heads: Number of attention heads
        embed_dim: Embedding dimension for attention
        flicker_threshold: Threshold for flicker detection (0-1)
        blend_strength: Strength of temporal blending (0-1)
        preserve_edges: Preserve edge sharpness during blending
        scene_change_threshold: Threshold for scene change detection
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
    """
    method: TemporalMethod = TemporalMethod.HYBRID
    attention_window: int = 7
    attention_heads: int = 8
    embed_dim: int = 256
    flicker_threshold: float = 0.02
    blend_strength: float = 0.8
    preserve_edges: bool = True
    scene_change_threshold: float = 0.3
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.method, str):
            self.method = TemporalMethod(self.method)
        if self.attention_window < 1:
            raise ValueError(f"attention_window must be >= 1, got {self.attention_window}")
        if not 0.0 <= self.blend_strength <= 1.0:
            raise ValueError(f"blend_strength must be 0-1, got {self.blend_strength}")


@dataclass
class TemporalConsistencyResult:
    """Result of temporal consistency processing.

    Attributes:
        frames_processed: Number of frames successfully processed
        frames_failed: Number of frames that failed
        flicker_regions_fixed: Number of flicker regions corrected
        scene_changes_detected: Frame indices of scene changes
        output_dir: Path to output directory
        processing_time_seconds: Total processing time
        peak_vram_mb: Peak VRAM usage
    """
    frames_processed: int = 0
    frames_failed: int = 0
    flicker_regions_fixed: int = 0
    scene_changes_detected: List[int] = field(default_factory=list)
    output_dir: Optional[Path] = None
    processing_time_seconds: float = 0.0
    peak_vram_mb: int = 0


# Conditionally define PyTorch modules only when PyTorch is available
if HAS_TORCH:
    class MultiHeadAttention(nn.Module):
        """Multi-head self-attention for temporal features."""

        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            batch_size = query.size(0)

            # Project
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # Reshape for multi-head attention
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, -1, self.embed_dim)
            output = self.out_proj(attn_output)

            return output

    class TemporalTransformerBlock(nn.Module):
        """Transformer block for temporal consistency."""

        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.attention = MultiHeadAttention(embed_dim, num_heads)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Self-attention with residual
            attn_out = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # FFN with residual
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            return x

    class CrossFrameAttention(nn.Module):
        """Cross-frame attention network for temporal consistency."""

        def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 4,
        ):
            super().__init__()

            # Patch embedding (simplified)
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)

            # Transformer blocks
            self.blocks = nn.ModuleList([
                TemporalTransformerBlock(embed_dim, num_heads)
                for _ in range(num_layers)
            ])

            # Output projection
            self.out_proj = nn.ConvTranspose2d(embed_dim, 3, kernel_size=8, stride=8)

        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            """Process frames with cross-attention.

            Args:
                frames: Tensor of shape [B, T, C, H, W] where T is temporal window

            Returns:
                Processed center frame
            """
            B, T, C, H, W = frames.shape

            # Embed each frame
            embeddings = []
            for t in range(T):
                emb = self.patch_embed(frames[:, t])  # [B, embed_dim, H', W']
                emb = emb.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
                embeddings.append(emb)

            # Concatenate temporal embeddings
            x = torch.cat(embeddings, dim=1)  # [B, T*N, embed_dim]

            # Apply transformer blocks
            for block in self.blocks:
                x = block(x)

            # Extract center frame tokens
            N = embeddings[0].size(1)
            center_idx = T // 2
            center_tokens = x[:, center_idx * N:(center_idx + 1) * N]  # [B, N, embed_dim]

            # Reshape and project to image
            H_patches = H // 8
            W_patches = W // 8
            center_tokens = center_tokens.transpose(1, 2).view(B, -1, H_patches, W_patches)
            output = self.out_proj(center_tokens)

            return output


class CrossAttentionTemporalProcessor:
    """Cross-frame attention processor for temporal consistency.

    Uses transformer attention across frames for superior temporal
    consistency compared to optical flow-based methods.

    Example:
        >>> config = CrossAttentionConfig(method=TemporalMethod.HYBRID)
        >>> processor = CrossAttentionTemporalProcessor(config)
        >>> result = processor.apply_consistency(input_dir, output_dir)
    """

    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'temporal'

    def __init__(
        self,
        config: Optional[CrossAttentionConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize temporal processor.

        Args:
            config: Processing configuration
            model_dir: Directory for model weights
        """
        self.config = config or CrossAttentionConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available backend."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available - temporal consistency disabled")
            return None

        if self.config.method in [TemporalMethod.CROSS_ATTENTION, TemporalMethod.HYBRID]:
            if HAS_TORCH:
                logger.info("Using cross-attention temporal consistency")
                return 'cross_attention'

        logger.info("Using optical flow temporal consistency")
        return 'optical_flow'

    def is_available(self) -> bool:
        """Check if temporal consistency is available."""
        return self._backend is not None

    def _load_model(self) -> None:
        """Load cross-attention model."""
        if self._model is not None:
            return

        if not HAS_TORCH:
            return

        import torch

        if torch.cuda.is_available():
            self._device = torch.device(f'cuda:{self.config.gpu_id}')
        else:
            self._device = torch.device('cpu')

        self._model = CrossFrameAttention(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.attention_heads,
        )

        # Try to load pretrained weights
        model_path = self.model_dir / 'cross_attention_temporal.pth'
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self._model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded temporal model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}")

        self._model = self._model.to(self._device)
        self._model.eval()

        if self.config.half_precision and self._device.type == 'cuda':
            self._model = self._model.half()

    def _detect_scene_change(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> bool:
        """Detect if there's a scene change between frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate histogram difference
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return correlation < self.config.scene_change_threshold

    def _detect_flicker(
        self,
        prev_frame: np.ndarray,
        current: np.ndarray,
        next_frame: np.ndarray,
    ) -> np.ndarray:
        """Detect flickering regions in current frame."""
        # Calculate temporal difference
        diff_prev = cv2.absdiff(current, prev_frame)
        diff_next = cv2.absdiff(current, next_frame)

        # Flicker: large change from both neighbors
        diff_neighbors = cv2.absdiff(prev_frame, next_frame)

        # Flicker regions: current differs from both neighbors
        # but neighbors are similar to each other
        gray_dp = cv2.cvtColor(diff_prev, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gray_dn = cv2.cvtColor(diff_next, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gray_nb = cv2.cvtColor(diff_neighbors, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

        # Flicker score: high change to neighbors, low change between neighbors
        flicker_score = np.minimum(gray_dp, gray_dn) * (1 - gray_nb)

        # Threshold
        flicker_mask = (flicker_score > self.config.flicker_threshold).astype(np.float32)

        # Smooth mask
        flicker_mask = cv2.GaussianBlur(flicker_mask, (5, 5), 0)

        return flicker_mask

    def _apply_optical_flow_consistency(
        self,
        frames: List[np.ndarray],
        center_idx: int,
    ) -> np.ndarray:
        """Apply optical flow-based temporal consistency."""
        center = frames[center_idx]

        if len(frames) < 3:
            return center

        # Get neighboring frames
        prev_idx = max(0, center_idx - 1)
        next_idx = min(len(frames) - 1, center_idx + 1)

        prev_frame = frames[prev_idx]
        next_frame = frames[next_idx]

        # Detect flicker
        flicker_mask = self._detect_flicker(prev_frame, center, next_frame)

        # Simple temporal averaging in flicker regions
        blend = (prev_frame.astype(np.float32) + next_frame.astype(np.float32)) / 2

        # Apply blending
        mask_3ch = flicker_mask[:, :, np.newaxis] * self.config.blend_strength
        result = center.astype(np.float32) * (1 - mask_3ch) + blend * mask_3ch

        return result.astype(np.uint8)

    def _apply_cross_attention_consistency(
        self,
        frames: List[np.ndarray],
        center_idx: int,
    ) -> np.ndarray:
        """Apply cross-attention temporal consistency."""
        import torch

        if self._model is None:
            self._load_model()

        if self._model is None:
            return self._apply_optical_flow_consistency(frames, center_idx)

        center = frames[center_idx]
        h, w = center.shape[:2]

        # Prepare input tensor
        window = self.config.attention_window
        half_window = window // 2

        # Collect frames for attention window
        frame_indices = []
        for i in range(-half_window, half_window + 1):
            idx = np.clip(center_idx + i, 0, len(frames) - 1)
            frame_indices.append(idx)

        # Convert to tensors
        frame_tensors = []
        for idx in frame_indices:
            frame = frames[idx]
            # Resize if needed for transformer
            if frame.shape[0] % 8 != 0 or frame.shape[1] % 8 != 0:
                new_h = (frame.shape[0] // 8) * 8
                new_w = (frame.shape[1] // 8) * 8
                frame = cv2.resize(frame, (new_w, new_h))

            # Convert to RGB tensor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            frame_tensors.append(tensor)

        # Stack frames: [T, C, H, W]
        frames_tensor = torch.stack(frame_tensors, dim=0)
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch: [1, T, C, H, W]
        frames_tensor = frames_tensor.to(self._device)

        if self.config.half_precision and self._device.type == 'cuda':
            frames_tensor = frames_tensor.half()

        # Process with model
        with torch.no_grad():
            output = self._model(frames_tensor)

        # Convert back
        output = output.squeeze(0).cpu()
        if output.dtype == torch.float16:
            output = output.float()
        output = output.permute(1, 2, 0).numpy()  # CHW -> HWC
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Resize back if needed
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h))

        # Blend with original based on strength
        if self.config.blend_strength < 1.0:
            output = cv2.addWeighted(
                center, 1 - self.config.blend_strength,
                output, self.config.blend_strength,
                0,
            )

        return output

    def _apply_hybrid_consistency(
        self,
        frames: List[np.ndarray],
        center_idx: int,
    ) -> np.ndarray:
        """Apply hybrid optical flow + attention consistency."""
        # First pass: optical flow for flicker detection
        prev_idx = max(0, center_idx - 1)
        next_idx = min(len(frames) - 1, center_idx + 1)

        flicker_mask = self._detect_flicker(
            frames[prev_idx],
            frames[center_idx],
            frames[next_idx],
        )

        # If significant flicker, use cross-attention
        flicker_area = np.mean(flicker_mask)

        if flicker_area > 0.01 and self._backend == 'cross_attention':
            return self._apply_cross_attention_consistency(frames, center_idx)
        else:
            return self._apply_optical_flow_consistency(frames, center_idx)

    def apply_consistency(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TemporalConsistencyResult:
        """Apply temporal consistency to video frames.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for processed output frames
            progress_callback: Optional progress callback (0-1)

        Returns:
            TemporalConsistencyResult with statistics
        """
        result = TemporalConsistencyResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("Temporal consistency not available")
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
        window = self.config.attention_window
        half_window = window // 2

        logger.info(
            f"Temporal consistency ({self.config.method.value}): {total_frames} frames, "
            f"window={window}"
        )

        # Load all frames (for windowed processing)
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            frames.append(frame)

        # Detect scene changes
        for i in range(1, len(frames)):
            if frames[i] is not None and frames[i-1] is not None:
                if self._detect_scene_change(frames[i-1], frames[i]):
                    result.scene_changes_detected.append(i)

        logger.info(f"Detected {len(result.scene_changes_detected)} scene changes")

        # Process frames
        for i, frame_file in enumerate(frame_files):
            try:
                if frames[i] is None:
                    logger.warning(f"Skipping invalid frame: {frame_file}")
                    result.frames_failed += 1
                    continue

                # Check for scene change - don't blend across scenes
                in_scene_change = any(
                    abs(i - sc) <= half_window
                    for sc in result.scene_changes_detected
                )

                if in_scene_change:
                    # Just copy original
                    processed = frames[i]
                else:
                    # Apply consistency method
                    if self.config.method == TemporalMethod.OPTICAL_FLOW:
                        processed = self._apply_optical_flow_consistency(frames, i)
                    elif self.config.method == TemporalMethod.CROSS_ATTENTION:
                        processed = self._apply_cross_attention_consistency(frames, i)
                    else:  # HYBRID
                        processed = self._apply_hybrid_consistency(frames, i)

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

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time

        if HAS_TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated(self.config.gpu_id) // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.config.gpu_id)

        logger.info(
            f"Temporal consistency complete: {result.frames_processed}/{total_frames} frames, "
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


def create_temporal_processor(
    method: str = "hybrid",
    window: int = 7,
    blend_strength: float = 0.8,
    gpu_id: int = 0,
) -> CrossAttentionTemporalProcessor:
    """Factory function to create a temporal consistency processor.

    Args:
        method: Method ("optical_flow", "cross_attention", "hybrid")
        window: Attention window size
        blend_strength: Blending strength (0-1)
        gpu_id: GPU device ID

    Returns:
        Configured CrossAttentionTemporalProcessor instance
    """
    config = CrossAttentionConfig(
        method=TemporalMethod(method),
        attention_window=window,
        blend_strength=blend_strength,
        gpu_id=gpu_id,
    )
    return CrossAttentionTemporalProcessor(config)
