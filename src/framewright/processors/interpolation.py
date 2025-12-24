"""Frame interpolation using RIFE (Real-Time Intermediate Flow Estimation)."""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Optional, Literal

logger = logging.getLogger(__name__)


class InterpolationError(Exception):
    """Exception raised for interpolation errors."""
    pass


class FrameInterpolator:
    """Frame interpolation using RIFE for increasing video frame rate.

    Supports interpolation from original frame rate (typically 24fps for old films)
    to higher frame rates (30, 48, 50, 60fps) using AI motion estimation.
    """

    SUPPORTED_MODELS = ['rife-v2.3', 'rife-v4.0', 'rife-v4.6']
    SUPPORTED_TARGET_FPS = [24, 30, 48, 50, 60, 120]

    def __init__(
        self,
        model: str = 'rife-v4.6',
        gpu_id: int = 0
    ):
        """Initialize the frame interpolator.

        Args:
            model: RIFE model version to use
            gpu_id: GPU device ID for processing
        """
        self.model = model
        self.gpu_id = gpu_id
        self._verify_dependencies()

    def _verify_dependencies(self) -> None:
        """Verify that RIFE is installed."""
        if not shutil.which('rife-ncnn-vulkan'):
            logger.warning(
                "rife-ncnn-vulkan not found. Install with: pip install rife-ncnn-vulkan "
                "or download from https://github.com/nihui/rife-ncnn-vulkan/releases"
            )

    def interpolate(
        self,
        input_dir: Path,
        output_dir: Path,
        source_fps: float = 24.0,
        target_fps: float = 60.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Interpolate frames to increase frame rate.

        Args:
            input_dir: Directory containing input frames (PNG sequence)
            output_dir: Directory for output frames
            source_fps: Original frame rate
            target_fps: Target frame rate
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            Path to output directory

        Raises:
            InterpolationError: If interpolation fails
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise InterpolationError(f"Input directory does not exist: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate interpolation factor
        factor = target_fps / source_fps

        # Determine number of intermediate frames
        # RIFE works in powers of 2 for interpolation
        if factor <= 2:
            n_interp = 1  # 2x
        elif factor <= 4:
            n_interp = 2  # 4x
        elif factor <= 8:
            n_interp = 3  # 8x
        else:
            n_interp = 4  # 16x max

        logger.info(
            f"Interpolating from {source_fps}fps to {target_fps}fps "
            f"(factor: {factor:.2f}x, intermediate frames: {2**n_interp - 1})"
        )

        cmd = [
            'rife-ncnn-vulkan',
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-m', self.model,
            '-g', str(self.gpu_id),
            '-n', str(n_interp),
            '-f', 'frame_%08d.png'
        ]

        if progress_callback:
            progress_callback(0.1)

        try:
            logger.info(f"Running RIFE: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"RIFE stdout: {result.stdout}")

            if progress_callback:
                progress_callback(0.9)

        except subprocess.CalledProcessError as e:
            raise InterpolationError(
                f"RIFE interpolation failed: {e.stderr}"
            ) from e
        except FileNotFoundError:
            raise InterpolationError(
                "rife-ncnn-vulkan not found. Please install RIFE."
            )

        # Verify output
        output_frames = list(output_dir.glob('*.png'))
        if not output_frames:
            raise InterpolationError("No output frames generated")

        logger.info(f"Generated {len(output_frames)} interpolated frames")

        if progress_callback:
            progress_callback(1.0)

        return output_dir

    def interpolate_to_fps(
        self,
        input_dir: Path,
        output_dir: Path,
        source_fps: float,
        target_fps: Literal[24, 30, 48, 50, 60],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> tuple[Path, float]:
        """Interpolate to a specific target FPS with frame decimation if needed.

        For non-power-of-2 multipliers (e.g., 24->30fps = 1.25x), this uses
        a higher interpolation factor and then decimates frames.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            source_fps: Original frame rate
            target_fps: Target frame rate (24, 30, 48, 50, or 60)
            progress_callback: Optional progress callback

        Returns:
            Tuple of (output_path, actual_fps)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        factor = target_fps / source_fps

        # Common scenarios for old film (24fps source):
        # 24 -> 30: interpolate 2x (48), decimate to 30
        # 24 -> 48: interpolate 2x (48), no decimation
        # 24 -> 50: interpolate 4x (96), decimate to 50
        # 24 -> 60: interpolate 4x (96), decimate to 60

        if factor <= 1.0:
            logger.warning(f"Target FPS ({target_fps}) <= source FPS ({source_fps}), no interpolation needed")
            # Just copy frames
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(sorted(input_dir.glob('*.png'))):
                shutil.copy(frame, output_dir / f'frame_{i:08d}.png')
            return output_dir, source_fps

        # Interpolate to power-of-2 multiple
        if factor <= 2:
            interp_fps = source_fps * 2
        elif factor <= 4:
            interp_fps = source_fps * 4
        else:
            interp_fps = source_fps * 8

        # First pass: interpolate
        temp_dir = output_dir.parent / f"{output_dir.name}_temp"
        self.interpolate(
            input_dir,
            temp_dir,
            source_fps,
            interp_fps,
            lambda p: progress_callback(p * 0.7) if progress_callback else None
        )

        # Second pass: decimate to target FPS if needed
        if abs(interp_fps - target_fps) > 0.5:
            logger.info(f"Decimating from {interp_fps}fps to {target_fps}fps")
            output_dir.mkdir(parents=True, exist_ok=True)

            interp_frames = sorted(temp_dir.glob('*.png'))
            frame_ratio = interp_fps / target_fps

            out_idx = 0
            for i, frame in enumerate(interp_frames):
                # Select frames at regular intervals
                if i >= out_idx * frame_ratio:
                    shutil.copy(frame, output_dir / f'frame_{out_idx:08d}.png')
                    out_idx += 1

            # Cleanup temp directory
            shutil.rmtree(temp_dir)

            if progress_callback:
                progress_callback(1.0)

            return output_dir, target_fps
        else:
            # No decimation needed, rename temp to output
            if output_dir.exists():
                shutil.rmtree(output_dir)
            temp_dir.rename(output_dir)

            if progress_callback:
                progress_callback(1.0)

            return output_dir, interp_fps

    @staticmethod
    def calculate_interpolation_factor(source_fps: float, target_fps: float) -> dict:
        """Calculate the interpolation strategy for given FPS conversion.

        Args:
            source_fps: Original frame rate
            target_fps: Desired frame rate

        Returns:
            Dictionary with interpolation strategy details
        """
        factor = target_fps / source_fps

        # Determine power-of-2 interpolation needed
        if factor <= 2:
            interp_multiplier = 2
        elif factor <= 4:
            interp_multiplier = 4
        elif factor <= 8:
            interp_multiplier = 8
        else:
            interp_multiplier = 16

        interp_fps = source_fps * interp_multiplier
        needs_decimation = abs(interp_fps - target_fps) > 0.5

        return {
            'source_fps': source_fps,
            'target_fps': target_fps,
            'factor': factor,
            'interpolation_multiplier': interp_multiplier,
            'intermediate_fps': interp_fps,
            'needs_decimation': needs_decimation,
            'final_fps': target_fps if needs_decimation else interp_fps
        }
