"""Missing Frame Generation for damaged video restoration.

This module implements generative AI-based frame reconstruction for videos
with damaged or missing frames. Goes beyond RIFE interpolation by generating
content when there are no adjacent valid frames to interpolate from.

Key use cases:
- Film reels with consecutive damaged frames
- Videos with dropped frames or gaps
- Corrupted sequences where data is lost

Model Sources (user must download manually):
- Stable Video Diffusion: HuggingFace stabilityai/stable-video-diffusion
- VideoGPT: https://github.com/wilson1yan/VideoGPT

Example:
    >>> config = FrameGenerationConfig(max_gap_frames=10)
    >>> generator = MissingFrameGenerator(config)
    >>> gaps = generator.detect_gaps(frames_dir, expected_count=1000)
    >>> result = generator.generate_missing(frames_dir, output_dir, gaps)
"""

import logging
import re
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GenerationModel(Enum):
    """Available frame generation models."""

    SVD = "svd"
    """Stable Video Diffusion for high-quality frame generation.

    VRAM: ~16GB
    Quality: Excellent
    Speed: Slow (5-15 sec per frame)
    Best for: High-quality reconstruction, short gaps
    """

    INTERPOLATE_BLEND = "interpolate_blend"
    """Simple interpolation with blending for gap filling.

    VRAM: Minimal
    Quality: Basic
    Speed: Fast
    Best for: Short gaps with similar adjacent frames
    """

    OPTICAL_FLOW_WARP = "optical_flow_warp"
    """Optical flow-based frame warping and blending.

    VRAM: ~2GB
    Quality: Good for small motion
    Speed: Medium
    Best for: Gaps with predictable motion
    """


@dataclass
class FrameGenerationConfig:
    """Configuration for missing frame generation.

    Attributes:
        model: Generation model to use
        max_gap_frames: Maximum number of frames to generate in a single gap
        match_grain: Match film grain of adjacent frames
        blend_edges: Number of frames to blend at gap edges
        motion_extrapolation: Enable motion extrapolation for generation
        temporal_consistency: Ensure temporal consistency across generated frames
        seed: Random seed for reproducibility
        gpu_id: GPU device ID
        half_precision: Use FP16 for reduced VRAM
    """
    model: GenerationModel = GenerationModel.INTERPOLATE_BLEND
    max_gap_frames: int = 10
    match_grain: bool = True
    blend_edges: int = 3
    motion_extrapolation: bool = True
    temporal_consistency: bool = True
    seed: Optional[int] = None
    gpu_id: int = 0
    half_precision: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.model, str):
            self.model = GenerationModel(self.model)
        if self.max_gap_frames < 1:
            raise ValueError(f"max_gap_frames must be >= 1, got {self.max_gap_frames}")
        if self.blend_edges < 0:
            raise ValueError(f"blend_edges must be >= 0, got {self.blend_edges}")


@dataclass
class GapInfo:
    """Information about a gap in the frame sequence."""
    start_frame: int  # Last valid frame before gap
    end_frame: int    # First valid frame after gap
    gap_size: int     # Number of missing frames
    before_path: Optional[Path] = None
    after_path: Optional[Path] = None


@dataclass
class FrameGenerationResult:
    """Result of frame generation.

    Attributes:
        frames_generated: Number of frames successfully generated
        frames_failed: Number of frames that failed
        gaps_processed: Number of gaps that were processed
        output_dir: Path to output directory
        processing_time_seconds: Total processing time
        peak_vram_mb: Peak VRAM usage
    """
    frames_generated: int = 0
    frames_failed: int = 0
    gaps_processed: int = 0
    output_dir: Optional[Path] = None
    processing_time_seconds: float = 0.0
    peak_vram_mb: int = 0


class MissingFrameGenerator:
    """Generator for missing video frames.

    Detects gaps in frame sequences and generates missing frames
    using various methods from simple interpolation to diffusion models.

    Example:
        >>> config = FrameGenerationConfig(model=GenerationModel.INTERPOLATE_BLEND)
        >>> generator = MissingFrameGenerator(config)
        >>> gaps = generator.detect_gaps(frames_dir, expected_count=1000)
        >>> result = generator.generate_missing(frames_dir, output_dir, gaps)
    """

    DEFAULT_MODEL_DIR = Path.home() / '.framewright' / 'models' / 'frame_gen'

    def __init__(
        self,
        config: Optional[FrameGenerationConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        """Initialize frame generator.

        Args:
            config: Generation configuration
            model_dir: Directory containing model weights
        """
        self.config = config or FrameGenerationConfig()
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model = None
        self._device = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> Optional[str]:
        """Detect available backend."""
        if not HAS_OPENCV:
            logger.warning("OpenCV not available - frame generation disabled")
            return None

        if self.config.model == GenerationModel.SVD:
            if HAS_TORCH:
                try:
                    import diffusers
                    logger.info("Found diffusers library for SVD")
                    return 'svd'
                except ImportError:
                    pass
            logger.warning("SVD not available, falling back to interpolation")
            return 'interpolate'

        if self.config.model == GenerationModel.OPTICAL_FLOW_WARP:
            if HAS_OPENCV:
                return 'optical_flow'

        return 'interpolate'

    def is_available(self) -> bool:
        """Check if frame generation is available."""
        return self._backend is not None

    def _parse_frame_number(self, filename: str) -> Optional[int]:
        """Extract frame number from filename."""
        # Common patterns: frame_00001.png, 00001.png, frame00001.png
        patterns = [
            r'frame[_-]?(\d+)',
            r'^(\d+)\.',
            r'_(\d+)\.',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def detect_gaps(
        self,
        frames_dir: Path,
        expected_count: Optional[int] = None,
    ) -> List[GapInfo]:
        """Detect gaps in frame sequence.

        Args:
            frames_dir: Directory containing frames
            expected_count: Expected total number of frames (for detecting end gaps)

        Returns:
            List of GapInfo objects describing each gap
        """
        frames_dir = Path(frames_dir)

        # Get all frame files
        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.jpg"))

        if not frame_files:
            logger.warning(f"No frames found in {frames_dir}")
            return []

        # Parse frame numbers
        frame_numbers = []
        path_by_number = {}

        for f in frame_files:
            num = self._parse_frame_number(f.name)
            if num is not None:
                frame_numbers.append(num)
                path_by_number[num] = f

        frame_numbers.sort()

        if not frame_numbers:
            logger.warning("Could not parse frame numbers")
            return []

        # Detect gaps
        gaps = []

        for i in range(len(frame_numbers) - 1):
            current = frame_numbers[i]
            next_num = frame_numbers[i + 1]

            if next_num - current > 1:
                gap_size = next_num - current - 1

                if gap_size <= self.config.max_gap_frames:
                    gaps.append(GapInfo(
                        start_frame=current,
                        end_frame=next_num,
                        gap_size=gap_size,
                        before_path=path_by_number.get(current),
                        after_path=path_by_number.get(next_num),
                    ))
                else:
                    logger.warning(
                        f"Gap too large to fill: frames {current}-{next_num} "
                        f"({gap_size} missing, max {self.config.max_gap_frames})"
                    )

        # Check for end gap
        if expected_count and frame_numbers:
            last_frame = frame_numbers[-1]
            if last_frame < expected_count - 1:
                gap_size = expected_count - 1 - last_frame
                if gap_size <= self.config.max_gap_frames:
                    gaps.append(GapInfo(
                        start_frame=last_frame,
                        end_frame=expected_count - 1,
                        gap_size=gap_size,
                        before_path=path_by_number.get(last_frame),
                        after_path=None,  # No frame after
                    ))

        logger.info(f"Detected {len(gaps)} gaps in frame sequence")
        for gap in gaps:
            logger.debug(f"  Gap: frames {gap.start_frame}-{gap.end_frame} ({gap.gap_size} missing)")

        return gaps

    def _generate_interpolated(
        self,
        before: np.ndarray,
        after: np.ndarray,
        num_frames: int,
    ) -> List[np.ndarray]:
        """Generate frames using simple interpolation."""
        generated = []

        for i in range(num_frames):
            alpha = (i + 1) / (num_frames + 1)

            # Linear interpolation
            frame = cv2.addWeighted(
                before, 1 - alpha,
                after, alpha,
                0,
            )

            generated.append(frame)

        return generated

    def _generate_optical_flow(
        self,
        before: np.ndarray,
        after: np.ndarray,
        num_frames: int,
    ) -> List[np.ndarray]:
        """Generate frames using optical flow warping."""
        # Calculate forward and backward flow
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Forward flow (before -> after)
        flow_forward = cv2.calcOpticalFlowFarneback(
            gray_before, gray_after,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )

        # Backward flow (after -> before)
        flow_backward = cv2.calcOpticalFlowFarneback(
            gray_after, gray_before,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )

        h, w = before.shape[:2]
        generated = []

        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)

            # Create interpolated flow
            flow_t = flow_forward * t

            # Create coordinate grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x = x.astype(np.float32)
            y = y.astype(np.float32)

            # Warp forward frame
            map_x = x + flow_t[:, :, 0]
            map_y = y + flow_t[:, :, 1]

            warped_before = cv2.remap(
                before, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            # Warp backward frame
            flow_t_back = flow_backward * (1 - t)
            map_x_back = x + flow_t_back[:, :, 0]
            map_y_back = y + flow_t_back[:, :, 1]

            warped_after = cv2.remap(
                after, map_x_back, map_y_back,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
            )

            # Blend warped frames
            frame = cv2.addWeighted(
                warped_before, 1 - t,
                warped_after, t,
                0,
            )

            generated.append(frame)

        return generated

    def _generate_svd(
        self,
        before: np.ndarray,
        after: Optional[np.ndarray],
        num_frames: int,
    ) -> List[np.ndarray]:
        """Generate frames using Stable Video Diffusion."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            import torch

            if self._device is None:
                if torch.cuda.is_available():
                    self._device = torch.device(f'cuda:{self.config.gpu_id}')
                else:
                    self._device = torch.device('cpu')

            # Load pipeline (cached)
            if self._model is None:
                self._model = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid",
                    torch_dtype=torch.float16 if self.config.half_precision else torch.float32,
                    variant="fp16" if self.config.half_precision else None,
                )
                self._model = self._model.to(self._device)

            # Convert to PIL
            from PIL import Image
            img_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            # Set seed
            generator = None
            if self.config.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(self.config.seed)

            # Generate frames
            with torch.no_grad():
                output = self._model(
                    pil_image,
                    num_frames=num_frames + 1,  # +1 because first frame is input
                    generator=generator,
                    decode_chunk_size=4,
                )

            # Convert back to numpy
            generated = []
            for frame in output.frames[0][1:]:  # Skip first frame (input)
                frame_np = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                generated.append(frame_bgr)

            return generated

        except Exception as e:
            logger.warning(f"SVD generation failed, falling back to interpolation: {e}")
            if after is not None:
                return self._generate_interpolated(before, after, num_frames)
            else:
                # Extrapolate by repeating with slight variation
                return [before.copy() for _ in range(num_frames)]

    def _extrapolate_frames(
        self,
        reference: np.ndarray,
        num_frames: int,
        direction: str = "forward",
    ) -> List[np.ndarray]:
        """Extrapolate frames when only one boundary is available."""
        # Simple extrapolation - slight variations of reference frame
        generated = []

        for i in range(num_frames):
            # Add slight noise for variation
            noise = np.random.normal(0, 2, reference.shape).astype(np.float32)
            frame = reference.astype(np.float32) + noise
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            generated.append(frame)

        return generated

    def _match_film_grain(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Match film grain characteristics of reference frame."""
        if not self.config.match_grain:
            return generated

        # Extract high-frequency content (grain) from reference
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref_blur = cv2.GaussianBlur(ref_gray, (0, 0), 2)
        ref_grain = ref_gray - ref_blur

        # Analyze grain statistics
        grain_std = np.std(ref_grain)

        # Add matching grain to generated frame
        gen_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gen_blur = cv2.GaussianBlur(gen_gray, (0, 0), 2)
        gen_grain = gen_gray - gen_blur
        current_std = np.std(gen_grain)

        if current_std > 0 and grain_std > current_std:
            # Add more grain
            additional_grain = np.random.normal(0, grain_std - current_std, ref_grain.shape)
            grain_3ch = np.stack([additional_grain] * 3, axis=-1)

            result = generated.astype(np.float32) + grain_3ch
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result

        return generated

    def _blend_edges(
        self,
        generated: List[np.ndarray],
        before: np.ndarray,
        after: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Blend generated frames with boundary frames for smooth transition."""
        if self.config.blend_edges <= 0 or not generated:
            return generated

        blend_frames = min(self.config.blend_edges, len(generated))

        for i in range(blend_frames):
            # Blend with before frame
            alpha = (i + 1) / (blend_frames + 1)
            generated[i] = cv2.addWeighted(
                before, 1 - alpha,
                generated[i], alpha,
                0,
            )

        if after is not None:
            for i in range(blend_frames):
                idx = len(generated) - 1 - i
                if idx >= 0:
                    alpha = (i + 1) / (blend_frames + 1)
                    generated[idx] = cv2.addWeighted(
                        after, 1 - alpha,
                        generated[idx], alpha,
                        0,
                    )

        return generated

    def generate_missing(
        self,
        input_dir: Path,
        output_dir: Path,
        gaps: List[GapInfo],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> FrameGenerationResult:
        """Generate missing frames to fill detected gaps.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output (original + generated frames)
            gaps: List of gaps to fill
            progress_callback: Optional progress callback (0-1)

        Returns:
            FrameGenerationResult with statistics
        """
        result = FrameGenerationResult()
        start_time = time.time()

        if not self.is_available():
            logger.error("Frame generation not available")
            return result

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = output_dir

        input_dir = Path(input_dir)

        # Copy all existing frames first
        for frame_file in input_dir.glob("*.png"):
            shutil.copy2(frame_file, output_dir / frame_file.name)
        for frame_file in input_dir.glob("*.jpg"):
            shutil.copy2(frame_file, output_dir / frame_file.name)

        if not gaps:
            logger.info("No gaps to fill")
            return result

        total_work = sum(gap.gap_size for gap in gaps)
        work_done = 0

        logger.info(f"Generating {total_work} frames to fill {len(gaps)} gaps")

        for gap in gaps:
            try:
                # Load boundary frames
                before = None
                after = None

                if gap.before_path and gap.before_path.exists():
                    before = cv2.imread(str(gap.before_path))

                if gap.after_path and gap.after_path.exists():
                    after = cv2.imread(str(gap.after_path))

                if before is None and after is None:
                    logger.warning(f"No boundary frames for gap at {gap.start_frame}")
                    result.frames_failed += gap.gap_size
                    continue

                # Generate frames based on available boundaries
                if before is not None and after is not None:
                    # Full interpolation
                    if self._backend == 'optical_flow':
                        generated = self._generate_optical_flow(before, after, gap.gap_size)
                    elif self._backend == 'svd':
                        generated = self._generate_svd(before, after, gap.gap_size)
                    else:
                        generated = self._generate_interpolated(before, after, gap.gap_size)

                elif before is not None:
                    # Forward extrapolation
                    if self._backend == 'svd':
                        generated = self._generate_svd(before, None, gap.gap_size)
                    else:
                        generated = self._extrapolate_frames(before, gap.gap_size, "forward")

                else:  # after is not None
                    # Backward extrapolation
                    generated = self._extrapolate_frames(after, gap.gap_size, "backward")
                    generated = generated[::-1]  # Reverse order

                # Post-process generated frames
                reference = before if before is not None else after

                for i, frame in enumerate(generated):
                    # Match grain
                    frame = self._match_film_grain(frame, reference)
                    generated[i] = frame

                # Blend edges
                generated = self._blend_edges(generated, before or after, after)

                # Save generated frames
                for i, frame in enumerate(generated):
                    frame_num = gap.start_frame + i + 1
                    # Match filename format of existing frames
                    filename = f"frame_{frame_num:08d}.png"

                    output_path = output_dir / filename
                    cv2.imwrite(str(output_path), frame)
                    result.frames_generated += 1

                result.gaps_processed += 1

            except Exception as e:
                logger.error(f"Failed to fill gap at {gap.start_frame}: {e}")
                result.frames_failed += gap.gap_size

            work_done += gap.gap_size
            if progress_callback:
                progress_callback(work_done / total_work)

        # Calculate statistics
        result.processing_time_seconds = time.time() - start_time

        if HAS_TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated(self.config.gpu_id) // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.config.gpu_id)

        logger.info(
            f"Frame generation complete: {result.frames_generated} generated, "
            f"{result.gaps_processed}/{len(gaps)} gaps filled, "
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


def create_frame_generator(
    model: str = "interpolate_blend",
    max_gap: int = 10,
    match_grain: bool = True,
    gpu_id: int = 0,
) -> MissingFrameGenerator:
    """Factory function to create a frame generator.

    Args:
        model: Generation model ("svd", "optical_flow_warp", "interpolate_blend")
        max_gap: Maximum gap size to fill
        match_grain: Match film grain of adjacent frames
        gpu_id: GPU device ID

    Returns:
        Configured MissingFrameGenerator instance
    """
    config = FrameGenerationConfig(
        model=GenerationModel(model),
        max_gap_frames=max_gap,
        match_grain=match_grain,
        gpu_id=gpu_id,
    )
    return MissingFrameGenerator(config)
