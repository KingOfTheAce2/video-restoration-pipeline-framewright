"""Frame interpolation using RIFE (Real-Time Intermediate Flow Estimation)."""

import logging
import os
import subprocess
import shutil
import sys
import zipfile
import tarfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Literal, Union

import numpy as np

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class SmoothnessLevel(Enum):
    """Smoothness level affecting interpolation quality and processing speed.

    LOW: Fast processing, may have artifacts (single pass)
    MEDIUM: Balanced quality/speed (default, double pass)
    HIGH: Maximum quality, slow (triple pass with refinement)
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class InterpolationConfig:
    """Configuration for frame interpolation.

    Attributes:
        target_fps: Target frames per second for output
        smoothness: Processing quality level (LOW, MEDIUM, HIGH)
        enable_scene_detection: Skip interpolation between scene cuts
        scene_threshold: Sensitivity for scene detection (0-1, higher = more sensitive)
        enable_motion_blur_reduction: Apply sharpening to reduce motion blur
        rife_model: RIFE model to use (rife-v4.6, rife-v4.0, rife-anime)
        gpu_id: GPU device ID for processing
    """
    target_fps: int = 60
    smoothness: SmoothnessLevel = SmoothnessLevel.MEDIUM
    enable_scene_detection: bool = True
    scene_threshold: float = 0.3
    enable_motion_blur_reduction: bool = False
    rife_model: str = "rife-v4.6"
    gpu_id: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.scene_threshold <= 1.0:
            raise ValueError(f"scene_threshold must be between 0 and 1, got {self.scene_threshold}")
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if isinstance(self.smoothness, str):
            self.smoothness = SmoothnessLevel(self.smoothness.lower())


# Model-specific settings for different RIFE variants
RIFE_MODEL_SETTINGS = {
    'rife-v4.6': {
        'description': 'Best quality, recommended for most content',
        'strengths': ['high quality', 'good motion handling', 'artifact-free'],
        'use_cases': ['live action', 'general purpose'],
        'speed_factor': 1.0,
    },
    'rife-v4.0': {
        'description': 'Faster processing with good quality',
        'strengths': ['faster processing', 'good quality', 'lower memory'],
        'use_cases': ['quick processing', 'lower-end GPUs'],
        'speed_factor': 1.3,
    },
    'rife-anime': {
        'description': 'Optimized for animation (flat colors, clean lines)',
        'strengths': ['anime optimization', 'flat color handling', 'clean line preservation'],
        'use_cases': ['anime', 'cartoons', 'animated content'],
        'speed_factor': 1.1,
    },
    'rife-v2.3': {
        'description': 'Legacy model for compatibility',
        'strengths': ['compatibility', 'stable'],
        'use_cases': ['legacy workflows'],
        'speed_factor': 1.2,
    },
}

# RIFE model download URLs
RIFE_MODEL_URLS = {
    'rife-v4.6': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip',
    'rife-v4.0': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20220728/rife-ncnn-vulkan-20220728-ubuntu.zip',
    'rife-v2.3': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20210728/rife-ncnn-vulkan-20210728-ubuntu.zip',
    'rife-anime': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip',
}

# Platform-specific download URLs
RIFE_PLATFORM_URLS = {
    'linux': {
        'rife-v4.6': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip',
    },
    'darwin': {
        'rife-v4.6': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip',
    },
    'win32': {
        'rife-v4.6': 'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip',
    },
}


class InterpolationError(Exception):
    """Exception raised for interpolation errors."""
    pass


class FrameInterpolator:
    """Frame interpolation using RIFE for increasing video frame rate.

    Supports interpolation from original frame rate (typically 24fps for old films)
    to higher frame rates (30, 48, 50, 60fps) using AI motion estimation.

    Enhanced features:
    - Multiple smoothness levels (LOW, MEDIUM, HIGH)
    - Scene change detection to avoid cross-scene interpolation
    - Motion blur reduction with sharpening filters
    - Multiple RIFE model options including anime-optimized
    """

    SUPPORTED_MODELS = ['rife-v2.3', 'rife-v4.0', 'rife-v4.6', 'rife-anime']
    SUPPORTED_TARGET_FPS = [24, 30, 48, 50, 60, 120]

    def __init__(
        self,
        model: str = 'rife-v4.6',
        gpu_id: int = 0,
        config: Optional[InterpolationConfig] = None
    ):
        """Initialize the frame interpolator.

        Args:
            model: RIFE model version to use (overridden by config if provided)
            gpu_id: GPU device ID for processing (overridden by config if provided)
            config: Optional InterpolationConfig for advanced settings
        """
        self.config = config or InterpolationConfig(
            rife_model=model,
            gpu_id=gpu_id
        )
        self.model = self.config.rife_model
        self.gpu_id = self.config.gpu_id
        self._scene_boundaries: list[int] = []
        self._verify_dependencies()

    def _verify_dependencies(self) -> None:
        """Verify that RIFE is installed, attempt auto-download if not found."""
        if not shutil.which('rife-ncnn-vulkan'):
            logger.warning(
                "rife-ncnn-vulkan not found. Attempting automatic download..."
            )
            try:
                self._auto_download_rife()
            except Exception as e:
                logger.warning(
                    f"Auto-download failed: {e}. "
                    "Install manually with: pip install rife-ncnn-vulkan "
                    "or download from https://github.com/nihui/rife-ncnn-vulkan/releases"
                )

    def _auto_download_rife(self) -> None:
        """Automatically download and install RIFE binaries."""
        import platform

        # Determine platform
        system = sys.platform
        if system.startswith('linux'):
            platform_key = 'linux'
        elif system == 'darwin':
            platform_key = 'darwin'
        elif system == 'win32':
            platform_key = 'win32'
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        # Get download URL
        urls = RIFE_PLATFORM_URLS.get(platform_key, {})
        url = urls.get(self.model)

        if not url:
            url = RIFE_MODEL_URLS.get(self.model)

        if not url:
            raise RuntimeError(f"No download URL available for {self.model}")

        # Download to user's local bin directory
        local_bin = Path.home() / '.local' / 'bin'
        local_bin.mkdir(parents=True, exist_ok=True)

        download_dir = Path.home() / '.cache' / 'framewright' / 'rife'
        download_dir.mkdir(parents=True, exist_ok=True)

        zip_path = download_dir / 'rife-ncnn-vulkan.zip'

        logger.info(f"Downloading RIFE from {url}...")

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                logger.debug(f"Download progress: {percent}%")

        urlretrieve(url, zip_path, reporthook=progress_hook)

        logger.info("Extracting RIFE...")

        # Extract the archive
        if str(zip_path).endswith('.zip'):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(download_dir)
        elif str(zip_path).endswith('.tar.gz'):
            with tarfile.open(zip_path, 'r:gz') as tf:
                tf.extractall(download_dir)

        # Find the executable
        for root, dirs, files in (download_dir).walk():
            for f in files:
                if f == 'rife-ncnn-vulkan' or f == 'rife-ncnn-vulkan.exe':
                    src = root / f
                    dst = local_bin / f

                    shutil.copy2(src, dst)
                    if platform_key != 'win32':
                        dst.chmod(0o755)

                    logger.info(f"RIFE installed to {dst}")

                    # Copy model files if present
                    for model_file in root.glob('*.bin'):
                        shutil.copy2(model_file, local_bin)
                    for model_file in root.glob('*.param'):
                        shutil.copy2(model_file, local_bin)

                    # Add to PATH hint
                    if str(local_bin) not in os.environ.get('PATH', ''):
                        logger.info(
                            f"Add {local_bin} to your PATH to use rife-ncnn-vulkan globally"
                        )
                    return

        raise RuntimeError("Could not find RIFE executable in downloaded archive")

    def detect_scene_change(
        self,
        frame1: Union[Path, np.ndarray],
        frame2: Union[Path, np.ndarray]
    ) -> bool:
        """Detect if there is a scene change between two frames.

        Uses histogram difference or SSIM (if available) to detect scene cuts.
        Scene changes should skip interpolation to avoid ghosting artifacts.

        Args:
            frame1: First frame (path or numpy array)
            frame2: Second frame (path or numpy array)

        Returns:
            True if scene change detected, False otherwise
        """
        # Load frames if paths provided
        if isinstance(frame1, Path) or isinstance(frame1, str):
            if not HAS_PIL:
                logger.warning("PIL not available, scene detection disabled")
                return False
            img1 = np.array(Image.open(frame1).convert('RGB'))
        else:
            img1 = frame1

        if isinstance(frame2, Path) or isinstance(frame2, str):
            if not HAS_PIL:
                return False
            img2 = np.array(Image.open(frame2).convert('RGB'))
        else:
            img2 = frame2

        # Use SSIM if available (more accurate)
        if HAS_SKIMAGE:
            try:
                # Convert to grayscale for SSIM
                gray1 = np.mean(img1, axis=2).astype(np.uint8)
                gray2 = np.mean(img2, axis=2).astype(np.uint8)

                # Calculate SSIM
                similarity = ssim(gray1, gray2, data_range=255)

                # Invert threshold: high similarity = low scene change probability
                # threshold of 0.3 means SSIM below 0.7 is considered scene change
                is_scene_change = similarity < (1.0 - self.config.scene_threshold)

                logger.debug(
                    f"SSIM: {similarity:.3f}, threshold: {1.0 - self.config.scene_threshold:.3f}, "
                    f"scene_change: {is_scene_change}"
                )
                return is_scene_change
            except Exception as e:
                logger.debug(f"SSIM calculation failed: {e}, falling back to histogram")

        # Fallback to histogram comparison
        return self._detect_scene_by_histogram(img1, img2)

    def _detect_scene_by_histogram(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> bool:
        """Detect scene change using histogram comparison.

        Args:
            img1: First frame as numpy array
            img2: Second frame as numpy array

        Returns:
            True if scene change detected
        """
        # Calculate histograms for each channel
        hist1 = np.concatenate([
            np.histogram(img1[:, :, c], bins=64, range=(0, 256))[0]
            for c in range(3)
        ])
        hist2 = np.concatenate([
            np.histogram(img2[:, :, c], bins=64, range=(0, 256))[0]
            for c in range(3)
        ])

        # Normalize histograms
        hist1 = hist1.astype(float) / hist1.sum()
        hist2 = hist2.astype(float) / hist2.sum()

        # Calculate histogram intersection (similarity measure)
        intersection = np.minimum(hist1, hist2).sum()

        # Higher threshold value = more sensitive to changes
        # With scene_threshold=0.3, we detect scene change when similarity < 0.7
        is_scene_change = intersection < (1.0 - self.config.scene_threshold)

        logger.debug(
            f"Histogram intersection: {intersection:.3f}, "
            f"threshold: {1.0 - self.config.scene_threshold:.3f}, "
            f"scene_change: {is_scene_change}"
        )

        return is_scene_change

    def detect_all_scene_changes(
        self,
        frame_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> list[int]:
        """Detect all scene changes in a frame sequence.

        Args:
            frame_dir: Directory containing frame images
            progress_callback: Optional progress callback

        Returns:
            List of frame indices where scene changes occur
        """
        frame_dir = Path(frame_dir)
        frames = sorted(frame_dir.glob('*.png'))

        if len(frames) < 2:
            return []

        scene_boundaries = []

        for i in range(len(frames) - 1):
            if self.detect_scene_change(frames[i], frames[i + 1]):
                scene_boundaries.append(i + 1)
                logger.info(f"Scene change detected at frame {i + 1}")

            if progress_callback:
                progress_callback((i + 1) / (len(frames) - 1))

        self._scene_boundaries = scene_boundaries
        logger.info(f"Detected {len(scene_boundaries)} scene changes")

        return scene_boundaries

    def apply_motion_blur_reduction(
        self,
        frame: Union[Path, np.ndarray],
        output_path: Optional[Path] = None,
        strength: float = 1.0
    ) -> np.ndarray:
        """Apply motion blur reduction (sharpening) to a frame.

        Args:
            frame: Input frame (path or numpy array)
            output_path: Optional path to save the result
            strength: Sharpening strength (0.5-2.0, default 1.0)

        Returns:
            Sharpened frame as numpy array
        """
        if not HAS_PIL:
            logger.warning("PIL not available, motion blur reduction disabled")
            if isinstance(frame, np.ndarray):
                return frame
            return np.array(Image.open(frame))

        # Load frame
        if isinstance(frame, Path) or isinstance(frame, str):
            img = Image.open(frame)
        else:
            img = Image.fromarray(frame.astype(np.uint8))

        # Apply unsharp mask for motion blur reduction
        # This enhances edges and reduces perceived blur
        sharpened = img.filter(ImageFilter.UnsharpMask(
            radius=2,
            percent=int(100 * strength),
            threshold=3
        ))

        # Optionally apply a second pass for fast motion sequences
        if strength > 1.5:
            sharpened = sharpened.filter(ImageFilter.UnsharpMask(
                radius=1,
                percent=int(50 * strength),
                threshold=2
            ))

        result = np.array(sharpened)

        if output_path:
            sharpened.save(output_path)

        return result

    def apply_motion_blur_reduction_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        strength: float = 1.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Apply motion blur reduction to all frames in a directory.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            strength: Sharpening strength (0.5-2.0)
            progress_callback: Optional progress callback

        Returns:
            Path to output directory
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(input_dir.glob('*.png'))

        for i, frame_path in enumerate(frames):
            output_path = output_dir / frame_path.name
            self.apply_motion_blur_reduction(frame_path, output_path, strength)

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        logger.info(f"Applied motion blur reduction to {len(frames)} frames")
        return output_dir

    def _get_pass_count(self) -> int:
        """Get the number of interpolation passes based on smoothness level.

        Returns:
            Number of passes (1 for LOW, 2 for MEDIUM, 3 for HIGH)
        """
        if self.config.smoothness == SmoothnessLevel.LOW:
            return 1
        elif self.config.smoothness == SmoothnessLevel.MEDIUM:
            return 2
        else:  # HIGH
            return 3

    @staticmethod
    def get_model_info(model: str) -> dict:
        """Get information about a RIFE model.

        Args:
            model: Model name (e.g., 'rife-v4.6', 'rife-anime')

        Returns:
            Dictionary with model information
        """
        return RIFE_MODEL_SETTINGS.get(model, {
            'description': 'Unknown model',
            'strengths': [],
            'use_cases': [],
            'speed_factor': 1.0,
        })

    @classmethod
    def list_available_models(cls) -> list[dict]:
        """List all available RIFE models with their descriptions.

        Returns:
            List of dictionaries with model information
        """
        return [
            {'name': model, **RIFE_MODEL_SETTINGS.get(model, {})}
            for model in cls.SUPPORTED_MODELS
        ]

    def interpolate(
        self,
        input_dir: Path,
        output_dir: Path,
        source_fps: float = 24.0,
        target_fps: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        config: Optional[InterpolationConfig] = None
    ) -> Path:
        """Interpolate frames to increase frame rate.

        Uses the configured smoothness level to determine processing quality:
        - LOW: Single pass, fast but may have artifacts
        - MEDIUM: Double pass for balanced quality/speed
        - HIGH: Triple pass with refinement for maximum quality

        Optionally detects scene changes and skips interpolation between scenes
        to avoid ghosting artifacts. Can also apply motion blur reduction.

        Args:
            input_dir: Directory containing input frames (PNG sequence)
            output_dir: Directory for output frames
            source_fps: Original frame rate
            target_fps: Target frame rate (uses config.target_fps if None)
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            config: Optional config override (uses self.config if None)

        Returns:
            Path to output directory

        Raises:
            InterpolationError: If interpolation fails
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Use provided config or instance config
        cfg = config or self.config
        target_fps = target_fps or cfg.target_fps

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

        # Get pass count based on smoothness
        pass_count = self._get_pass_count()
        model_info = self.get_model_info(self.model)

        logger.info(
            f"Interpolating from {source_fps}fps to {target_fps}fps "
            f"(factor: {factor:.2f}x, intermediate frames: {2**n_interp - 1}, "
            f"passes: {pass_count}, model: {self.model})"
        )

        # Detect scene changes if enabled
        scene_boundaries = []
        if cfg.enable_scene_detection:
            if progress_callback:
                progress_callback(0.02)
            logger.info("Detecting scene changes...")
            scene_boundaries = self.detect_all_scene_changes(
                input_dir,
                lambda p: progress_callback(0.02 + p * 0.08) if progress_callback else None
            )

        if progress_callback:
            progress_callback(0.1)

        # Process based on smoothness level
        current_input_dir = input_dir
        current_output_dir = output_dir

        for pass_num in range(pass_count):
            is_final_pass = (pass_num == pass_count - 1)

            # For multi-pass, use temp directories for intermediate results
            if not is_final_pass:
                current_output_dir = output_dir.parent / f"{output_dir.name}_pass{pass_num}"
                current_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                current_output_dir = output_dir

            # Build RIFE command
            cmd = [
                'rife-ncnn-vulkan',
                '-i', str(current_input_dir),
                '-o', str(current_output_dir),
                '-m', self.model,
                '-g', str(self.gpu_id),
                '-n', str(n_interp if pass_num == 0 else 1),  # Only first pass does full interpolation
                '-f', 'frame_%08d.png'
            ]

            # For HIGH smoothness, add refinement on later passes
            if cfg.smoothness == SmoothnessLevel.HIGH and pass_num > 0:
                # Additional passes refine the result
                cmd.extend(['-x'])  # Enable UHD mode for finer quality

            try:
                logger.info(f"Running RIFE pass {pass_num + 1}/{pass_count}: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.debug(f"RIFE stdout: {result.stdout}")

                # Progress update based on pass
                if progress_callback:
                    base_progress = 0.1 + (0.7 * (pass_num + 1) / pass_count)
                    progress_callback(base_progress)

            except subprocess.CalledProcessError as e:
                raise InterpolationError(
                    f"RIFE interpolation failed on pass {pass_num + 1}: {e.stderr}"
                ) from e
            except FileNotFoundError:
                raise InterpolationError(
                    "rife-ncnn-vulkan not found. Please install RIFE."
                )

            # Cleanup intermediate directories and prepare for next pass
            if not is_final_pass:
                # Use output of this pass as input for next
                if pass_num > 0:
                    # Remove previous intermediate dir
                    prev_dir = output_dir.parent / f"{output_dir.name}_pass{pass_num - 1}"
                    if prev_dir.exists():
                        shutil.rmtree(prev_dir)
                current_input_dir = current_output_dir

        # Cleanup last intermediate directory if exists
        if pass_count > 1:
            for i in range(pass_count - 1):
                temp_dir = output_dir.parent / f"{output_dir.name}_pass{i}"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        # Apply motion blur reduction if enabled
        if cfg.enable_motion_blur_reduction:
            logger.info("Applying motion blur reduction...")
            if progress_callback:
                progress_callback(0.85)

            # Apply to output frames in place
            temp_blur_dir = output_dir.parent / f"{output_dir.name}_deblur"
            self.apply_motion_blur_reduction_batch(
                output_dir,
                temp_blur_dir,
                strength=1.0,
                progress_callback=lambda p: progress_callback(0.85 + p * 0.1) if progress_callback else None
            )

            # Replace original output with deblurred frames
            shutil.rmtree(output_dir)
            temp_blur_dir.rename(output_dir)

        # Verify output
        output_frames = list(output_dir.glob('*.png'))
        if not output_frames:
            raise InterpolationError("No output frames generated")

        logger.info(
            f"Generated {len(output_frames)} interpolated frames "
            f"(smoothness: {cfg.smoothness.value}, scene_changes: {len(scene_boundaries)})"
        )

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

    def interpolate_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        source_fps: float = 24.0,
        config: Optional[InterpolationConfig] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> dict:
        """Enhanced frame interpolation using configuration object.

        This is the recommended high-level API for frame interpolation. It uses
        InterpolationConfig for all settings and returns detailed results.

        Args:
            input_dir: Directory containing input frames (PNG sequence)
            output_dir: Directory for output frames
            source_fps: Original frame rate
            config: InterpolationConfig (uses self.config if None)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with interpolation results:
            - output_dir: Path to output directory
            - input_frames: Number of input frames
            - output_frames: Number of output frames
            - source_fps: Original FPS
            - target_fps: Target FPS
            - actual_fps: Achieved FPS
            - scene_changes: List of detected scene change frame indices
            - model: RIFE model used
            - smoothness: Smoothness level used
            - motion_blur_reduced: Whether motion blur reduction was applied

        Example:
            >>> config = InterpolationConfig(
            ...     target_fps=60,
            ...     smoothness=SmoothnessLevel.HIGH,
            ...     enable_scene_detection=True,
            ...     enable_motion_blur_reduction=True,
            ...     rife_model='rife-v4.6'
            ... )
            >>> interpolator = FrameInterpolator(config=config)
            >>> result = interpolator.interpolate_frames(
            ...     input_dir=Path('/frames/input'),
            ...     output_dir=Path('/frames/output'),
            ...     source_fps=24.0
            ... )
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Use provided config or instance config
        cfg = config or self.config

        # Count input frames
        input_frames = sorted(input_dir.glob('*.png'))
        input_count = len(input_frames)

        if input_count == 0:
            raise InterpolationError(f"No PNG frames found in {input_dir}")

        logger.info(
            f"Starting enhanced interpolation: {input_count} frames, "
            f"{source_fps}fps -> {cfg.target_fps}fps, "
            f"model={cfg.rife_model}, smoothness={cfg.smoothness.value}"
        )

        # Perform interpolation
        result_path = self.interpolate(
            input_dir=input_dir,
            output_dir=output_dir,
            source_fps=source_fps,
            target_fps=cfg.target_fps,
            progress_callback=progress_callback,
            config=cfg
        )

        # Count output frames
        output_frames = list(result_path.glob('*.png'))
        output_count = len(output_frames)

        # Calculate actual achieved FPS
        if input_count > 0:
            actual_fps = source_fps * (output_count / input_count)
        else:
            actual_fps = cfg.target_fps

        return {
            'output_dir': result_path,
            'input_frames': input_count,
            'output_frames': output_count,
            'source_fps': source_fps,
            'target_fps': cfg.target_fps,
            'actual_fps': actual_fps,
            'scene_changes': self._scene_boundaries.copy(),
            'model': cfg.rife_model,
            'smoothness': cfg.smoothness.value,
            'motion_blur_reduced': cfg.enable_motion_blur_reduction
        }


# Convenience functions for common use cases

def create_interpolator(
    target_fps: int = 60,
    smoothness: str = "medium",
    enable_scene_detection: bool = True,
    rife_model: str = "rife-v4.6",
    gpu_id: int = 0
) -> FrameInterpolator:
    """Create a FrameInterpolator with common settings.

    Args:
        target_fps: Target frames per second
        smoothness: Processing quality ("low", "medium", "high")
        enable_scene_detection: Detect and handle scene changes
        rife_model: RIFE model to use
        gpu_id: GPU device ID

    Returns:
        Configured FrameInterpolator instance
    """
    config = InterpolationConfig(
        target_fps=target_fps,
        smoothness=SmoothnessLevel(smoothness.lower()),
        enable_scene_detection=enable_scene_detection,
        rife_model=rife_model,
        gpu_id=gpu_id
    )
    return FrameInterpolator(config=config)


def interpolate_for_anime(
    input_dir: Path,
    output_dir: Path,
    source_fps: float = 24.0,
    target_fps: int = 60,
    progress_callback: Optional[Callable[[float], None]] = None
) -> dict:
    """Convenience function for anime/animation interpolation.

    Uses anime-optimized settings for flat colors and clean lines.

    Args:
        input_dir: Directory containing input frames
        output_dir: Directory for output frames
        source_fps: Original frame rate
        target_fps: Target frame rate
        progress_callback: Optional progress callback

    Returns:
        Dictionary with interpolation results
    """
    config = InterpolationConfig(
        target_fps=target_fps,
        smoothness=SmoothnessLevel.MEDIUM,
        enable_scene_detection=True,
        scene_threshold=0.4,  # Slightly more sensitive for anime scene cuts
        enable_motion_blur_reduction=False,  # Anime typically doesn't have motion blur
        rife_model='rife-anime'
    )

    interpolator = FrameInterpolator(config=config)
    return interpolator.interpolate_frames(
        input_dir=input_dir,
        output_dir=output_dir,
        source_fps=source_fps,
        progress_callback=progress_callback
    )


def interpolate_high_quality(
    input_dir: Path,
    output_dir: Path,
    source_fps: float = 24.0,
    target_fps: int = 60,
    progress_callback: Optional[Callable[[float], None]] = None
) -> dict:
    """Convenience function for maximum quality interpolation.

    Uses triple-pass processing with scene detection and motion blur reduction.

    Args:
        input_dir: Directory containing input frames
        output_dir: Directory for output frames
        source_fps: Original frame rate
        target_fps: Target frame rate
        progress_callback: Optional progress callback

    Returns:
        Dictionary with interpolation results
    """
    config = InterpolationConfig(
        target_fps=target_fps,
        smoothness=SmoothnessLevel.HIGH,
        enable_scene_detection=True,
        scene_threshold=0.3,
        enable_motion_blur_reduction=True,
        rife_model='rife-v4.6'
    )

    interpolator = FrameInterpolator(config=config)
    return interpolator.interpolate_frames(
        input_dir=input_dir,
        output_dir=output_dir,
        source_fps=source_fps,
        progress_callback=progress_callback
    )


def interpolate_fast(
    input_dir: Path,
    output_dir: Path,
    source_fps: float = 24.0,
    target_fps: int = 60,
    progress_callback: Optional[Callable[[float], None]] = None
) -> dict:
    """Convenience function for fast interpolation.

    Uses single-pass processing with faster model for quick results.

    Args:
        input_dir: Directory containing input frames
        output_dir: Directory for output frames
        source_fps: Original frame rate
        target_fps: Target frame rate
        progress_callback: Optional progress callback

    Returns:
        Dictionary with interpolation results
    """
    config = InterpolationConfig(
        target_fps=target_fps,
        smoothness=SmoothnessLevel.LOW,
        enable_scene_detection=False,  # Skip for speed
        enable_motion_blur_reduction=False,
        rife_model='rife-v4.0'  # Faster model
    )

    interpolator = FrameInterpolator(config=config)
    return interpolator.interpolate_frames(
        input_dir=input_dir,
        output_dir=output_dir,
        source_fps=source_fps,
        progress_callback=progress_callback
    )
