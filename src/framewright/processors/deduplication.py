"""Frame deduplication for historical film restoration.

Detects and removes duplicate frames caused by frame rate conversion
(e.g., 18fps film padded to 25fps for YouTube). This significantly
reduces processing time by only enhancing unique frames.

Typical use case:
- 1909 film shot at 16-18fps
- Digitized/uploaded at 25fps with duplicate frames
- We detect unique frames, enhance only those, then reconstruct
"""
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import image comparison libraries
try:
    from PIL import Image
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.debug("imagehash not available, using pixel-based comparison")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class DeduplicationResult:
    """Result of frame deduplication analysis."""
    total_frames: int = 0
    unique_frames: int = 0
    duplicate_frames: int = 0
    detected_source_fps: float = 0.0
    target_fps: float = 25.0
    frame_mapping: Dict[int, int] = field(default_factory=dict)  # original_idx -> unique_idx
    unique_indices: List[int] = field(default_factory=list)  # indices of unique frames

    @property
    def duplication_ratio(self) -> float:
        """Ratio of duplicates to total frames."""
        if self.total_frames == 0:
            return 0.0
        return self.duplicate_frames / self.total_frames

    @property
    def estimated_original_fps(self) -> float:
        """Estimate original FPS based on unique frame count."""
        if self.unique_frames == 0 or self.total_frames == 0:
            return self.target_fps
        ratio = self.unique_frames / self.total_frames
        return self.target_fps * ratio

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Frames: {self.unique_frames}/{self.total_frames} unique "
            f"({self.duplicate_frames} duplicates, {self.duplication_ratio:.1%} reduction)\n"
            f"Estimated original FPS: {self.estimated_original_fps:.1f} "
            f"(target: {self.target_fps}fps)"
        )


@dataclass
class DeduplicationConfig:
    """Configuration for frame deduplication."""
    # Similarity threshold (0.0 = exact match only, higher = more lenient)
    similarity_threshold: float = 0.98
    # Use perceptual hashing (faster, handles compression artifacts)
    use_perceptual_hash: bool = True
    # Hash size for perceptual hashing (higher = more precise)
    hash_size: int = 16
    # Compare every Nth pixel for fast mode (1 = all pixels)
    pixel_sample_rate: int = 4
    # Minimum unique frames (abort if too few detected)
    min_unique_ratio: float = 0.3
    # Expected source FPS hints (helps validation)
    expected_source_fps: Optional[float] = None


class FrameDeduplicator:
    """Detects and removes duplicate frames from video sequences.

    Uses perceptual hashing (pHash) or pixel comparison to identify
    frames that are duplicates or near-duplicates.
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        """Initialize the deduplicator.

        Args:
            config: Deduplication configuration
        """
        self.config = config or DeduplicationConfig()
        self._hash_cache: Dict[Path, str] = {}

    def _compute_hash(self, image_path: Path) -> str:
        """Compute perceptual hash for an image.

        Args:
            image_path: Path to image file

        Returns:
            Hash string for comparison
        """
        if image_path in self._hash_cache:
            return self._hash_cache[image_path]

        if IMAGEHASH_AVAILABLE and self.config.use_perceptual_hash:
            try:
                img = Image.open(image_path)
                # Use difference hash (dHash) - fast and effective
                phash = imagehash.dhash(img, hash_size=self.config.hash_size)
                hash_str = str(phash)
                self._hash_cache[image_path] = hash_str
                return hash_str
            except Exception as e:
                logger.debug(f"pHash failed for {image_path}: {e}, falling back to pixel hash")

        # Fallback: MD5 of downsampled pixels
        return self._compute_pixel_hash(image_path)

    def _compute_pixel_hash(self, image_path: Path) -> str:
        """Compute hash based on pixel values (fallback method).

        Args:
            image_path: Path to image file

        Returns:
            MD5 hash of sampled pixels
        """
        try:
            img = Image.open(image_path)
            # Downsample for speed
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img = img.convert('L')  # Grayscale

            # Get pixel data
            pixels = list(img.getdata())

            # Sample pixels
            sample_rate = self.config.pixel_sample_rate
            sampled = pixels[::sample_rate]

            # Hash the pixel values
            pixel_bytes = bytes(sampled)
            hash_str = hashlib.md5(pixel_bytes).hexdigest()

            self._hash_cache[image_path] = hash_str
            return hash_str

        except Exception as e:
            logger.warning(f"Could not hash {image_path}: {e}")
            # Return unique hash on error
            return hashlib.md5(str(image_path).encode()).hexdigest()

    def _compare_frames(self, hash1: str, hash2: str) -> float:
        """Compare two frame hashes and return similarity score.

        Args:
            hash1: First frame hash
            hash2: Second frame hash

        Returns:
            Similarity score (1.0 = identical, 0.0 = completely different)
        """
        if hash1 == hash2:
            return 1.0

        if IMAGEHASH_AVAILABLE and self.config.use_perceptual_hash:
            try:
                # For perceptual hashes, compute Hamming distance
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
                # Normalize distance to similarity (0-1)
                max_dist = self.config.hash_size * self.config.hash_size
                distance = h1 - h2
                similarity = 1.0 - (distance / max_dist)
                return similarity
            except Exception:
                pass

        # For MD5 hashes, it's binary (match or not)
        return 1.0 if hash1 == hash2 else 0.0

    def analyze_frames(
        self,
        frames_dir: Path,
        target_fps: float = 25.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DeduplicationResult:
        """Analyze frames to detect duplicates.

        Args:
            frames_dir: Directory containing extracted frames
            target_fps: The FPS of the source video (e.g., 25 for YouTube)
            progress_callback: Optional callback for progress updates

        Returns:
            DeduplicationResult with analysis
        """
        frames = sorted(frames_dir.glob("frame_*.png"))
        total = len(frames)

        if total == 0:
            logger.warning(f"No frames found in {frames_dir}")
            return DeduplicationResult()

        logger.info(f"Analyzing {total} frames for duplicates...")

        result = DeduplicationResult(
            total_frames=total,
            target_fps=target_fps,
        )

        unique_indices: List[int] = []
        frame_mapping: Dict[int, int] = {}

        # First frame is always unique
        unique_indices.append(0)
        frame_mapping[0] = 0
        last_unique_hash = self._compute_hash(frames[0])
        last_unique_idx = 0

        for i, frame_path in enumerate(frames[1:], start=1):
            if progress_callback and i % 100 == 0:
                progress_callback(i / total)

            current_hash = self._compute_hash(frame_path)
            similarity = self._compare_frames(last_unique_hash, current_hash)

            if similarity >= self.config.similarity_threshold:
                # This is a duplicate of the last unique frame
                frame_mapping[i] = last_unique_idx
            else:
                # This is a new unique frame
                unique_indices.append(i)
                frame_mapping[i] = i
                last_unique_hash = current_hash
                last_unique_idx = i

        result.unique_frames = len(unique_indices)
        result.duplicate_frames = total - result.unique_frames
        result.unique_indices = unique_indices
        result.frame_mapping = frame_mapping
        result.detected_source_fps = result.estimated_original_fps

        if progress_callback:
            progress_callback(1.0)

        logger.info(result.summary())

        # Validate result
        if result.unique_frames / total < self.config.min_unique_ratio:
            logger.warning(
                f"Very few unique frames detected ({result.unique_frames}/{total}). "
                f"This may indicate incorrect threshold or non-duplicated content."
            )

        return result

    def extract_unique_frames(
        self,
        frames_dir: Path,
        output_dir: Path,
        result: Optional[DeduplicationResult] = None,
        target_fps: float = 25.0,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[Path, DeduplicationResult]:
        """Extract unique frames to a separate directory.

        Args:
            frames_dir: Source directory with all frames
            output_dir: Destination directory for unique frames
            result: Pre-computed deduplication result (will analyze if None)
            target_fps: Target FPS for analysis
            progress_callback: Optional progress callback

        Returns:
            Tuple of (output_dir, DeduplicationResult)
        """
        # Analyze if not provided
        if result is None:
            result = self.analyze_frames(frames_dir, target_fps, progress_callback)

        if result.unique_frames == 0:
            raise ValueError("No unique frames detected")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frames_dir.glob("frame_*.png"))

        logger.info(f"Copying {result.unique_frames} unique frames to {output_dir}")

        # Copy unique frames, maintaining frame numbering for reconstruction
        for i, idx in enumerate(result.unique_indices):
            if progress_callback and i % 50 == 0:
                progress_callback(i / len(result.unique_indices))

            src = frames[idx]
            # Keep original frame number for proper reconstruction
            dst = output_dir / src.name
            shutil.copy2(src, dst)

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Extracted {result.unique_frames} unique frames")

        return output_dir, result

    def reconstruct_sequence(
        self,
        enhanced_dir: Path,
        output_dir: Path,
        result: DeduplicationResult,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Reconstruct full frame sequence from enhanced unique frames.

        After enhancing only unique frames, this reconstructs the full
        sequence by copying enhanced frames to their duplicate positions.

        Args:
            enhanced_dir: Directory with enhanced unique frames
            output_dir: Output directory for full reconstructed sequence
            result: Deduplication result with frame mapping
            progress_callback: Optional progress callback

        Returns:
            Path to output directory with full sequence
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build reverse mapping: unique_idx -> enhanced_frame_path
        enhanced_frames = {
            int(f.stem.split('_')[-1]): f
            for f in enhanced_dir.glob("frame_*.png")
        }

        logger.info(
            f"Reconstructing {result.total_frames} frames from "
            f"{len(enhanced_frames)} enhanced unique frames"
        )

        # Reconstruct full sequence
        for orig_idx in range(result.total_frames):
            if progress_callback and orig_idx % 100 == 0:
                progress_callback(orig_idx / result.total_frames)

            # Find which unique frame this maps to
            unique_idx = result.frame_mapping.get(orig_idx, orig_idx)

            # Get the enhanced version
            if unique_idx in enhanced_frames:
                src = enhanced_frames[unique_idx]
            else:
                # Fallback: find nearest unique frame
                nearest = min(enhanced_frames.keys(), key=lambda x: abs(x - unique_idx))
                src = enhanced_frames[nearest]
                logger.debug(f"Frame {orig_idx} mapped to nearest {nearest}")

            # Output with original frame number
            dst = output_dir / f"frame_{orig_idx:08d}.png"
            shutil.copy2(src, dst)

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Reconstructed {result.total_frames} frames to {output_dir}")

        return output_dir


def detect_duplicate_frames(
    frames_dir: Path,
    target_fps: float = 25.0,
    similarity_threshold: float = 0.98,
) -> DeduplicationResult:
    """Convenience function to detect duplicate frames.

    Args:
        frames_dir: Directory containing frames
        target_fps: FPS of the video
        similarity_threshold: How similar frames must be to be duplicates

    Returns:
        DeduplicationResult with analysis
    """
    config = DeduplicationConfig(similarity_threshold=similarity_threshold)
    deduplicator = FrameDeduplicator(config)
    return deduplicator.analyze_frames(frames_dir, target_fps)


def deduplicate_and_enhance(
    frames_dir: Path,
    unique_dir: Path,
    target_fps: float = 25.0,
    similarity_threshold: float = 0.98,
) -> Tuple[Path, DeduplicationResult]:
    """Extract unique frames for enhancement.

    Args:
        frames_dir: Source frames directory
        unique_dir: Output directory for unique frames
        target_fps: FPS of the video
        similarity_threshold: Similarity threshold for duplicates

    Returns:
        Tuple of (unique_frames_dir, result)
    """
    config = DeduplicationConfig(similarity_threshold=similarity_threshold)
    deduplicator = FrameDeduplicator(config)
    return deduplicator.extract_unique_frames(
        frames_dir, unique_dir, target_fps=target_fps
    )
