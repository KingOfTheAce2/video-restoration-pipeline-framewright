"""Long-form temporal consistency system for videos with 7000+ frames.

This module provides advanced temporal consistency management designed for
processing very long videos while maintaining visual coherence across the
entire duration. Key features:

- GlobalAnchors: Extract and maintain reference data from key frames
- LongFormConsistencyManager: Coordinate consistency across thousands of frames
- ColorConsistencyEnforcer: Detect and correct color drift over time
- ChunkedProcessor: Memory-efficient processing with boundary blending

The system uses overlapping chunk processing (default 50 frames with 4 overlap)
to ensure smooth transitions while keeping memory usage bounded.

Example:
    >>> config = TemporalConsistencyConfig(
    ...     chunk_size=50,
    ...     overlap_frames=4,
    ...     anchor_interval=500,
    ... )
    >>> manager = LongFormConsistencyManager(config)
    >>> manager.initialize_from_video(video_path)
    >>> for result in manager.process_streaming(input_dir, output_dir):
    ...     print(f"Processed chunk {result.chunk_id}")
"""

import gc
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.debug("OpenCV not available - some features limited")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available - using fallback clustering")


class AnchorType(Enum):
    """Types of global anchors extracted from key frames."""

    COLOR_PALETTE = auto()
    """Dominant colors and white balance reference."""

    GRAIN_PROFILE = auto()
    """Film grain / noise characteristics."""

    FACE_EMBEDDINGS = auto()
    """Face identity embeddings for tracking."""

    BRIGHTNESS_HISTOGRAM = auto()
    """Brightness distribution reference."""

    CONTRAST_PROFILE = auto()
    """Contrast and dynamic range profile."""


class DriftCorrectionMode(Enum):
    """Modes for correcting color drift."""

    NONE = "none"
    """No drift correction."""

    SUBTLE = "subtle"
    """Light correction to maintain natural variation."""

    BALANCED = "balanced"
    """Balanced correction for typical videos."""

    AGGRESSIVE = "aggressive"
    """Strong correction for severely drifting content."""


@dataclass
class TemporalConsistencyConfig:
    """Configuration for long-form temporal consistency processing.

    Attributes:
        chunk_size: Number of frames per processing chunk.
        overlap_frames: Number of frames to overlap between chunks for blending.
        anchor_interval: Extract anchors every N frames.
        anchor_sample_count: Number of anchor frames to sample for initial analysis.
        max_color_drift: Maximum allowed color drift (0-1) before correction.
        drift_correction_mode: How aggressively to correct color drift.
        face_tracking_enabled: Enable face identity tracking across frames.
        grain_preservation: Preserve film grain characteristics (0-1).
        buffer_size: Maximum frames to keep in memory buffer.
        enable_gpu: Use GPU acceleration if available.
        gpu_id: GPU device ID for processing.
    """

    chunk_size: int = 50
    overlap_frames: int = 4
    anchor_interval: int = 500
    anchor_sample_count: int = 10
    max_color_drift: float = 0.15
    drift_correction_mode: DriftCorrectionMode = DriftCorrectionMode.BALANCED
    face_tracking_enabled: bool = True
    grain_preservation: float = 0.5
    buffer_size: int = 100
    enable_gpu: bool = True
    gpu_id: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.chunk_size < 10:
            raise ValueError(f"chunk_size must be >= 10, got {self.chunk_size}")
        if self.overlap_frames < 0:
            raise ValueError(f"overlap_frames must be >= 0, got {self.overlap_frames}")
        if self.overlap_frames >= self.chunk_size // 2:
            raise ValueError(
                f"overlap_frames ({self.overlap_frames}) must be less than "
                f"half of chunk_size ({self.chunk_size})"
            )
        if not 0.0 <= self.max_color_drift <= 1.0:
            raise ValueError(f"max_color_drift must be 0-1, got {self.max_color_drift}")
        if not 0.0 <= self.grain_preservation <= 1.0:
            raise ValueError(
                f"grain_preservation must be 0-1, got {self.grain_preservation}"
            )


@dataclass
class ColorPalette:
    """Color palette extracted from reference frames.

    Attributes:
        dominant_colors: Array of dominant colors (N x 3, RGB).
        white_point: Estimated white point (R, G, B).
        black_point: Estimated black point (R, G, B).
        color_temperature: Estimated color temperature in Kelvin.
        saturation_mean: Mean saturation level.
        hue_histogram: Histogram of hue values (360 bins).
    """

    dominant_colors: np.ndarray
    white_point: np.ndarray
    black_point: np.ndarray
    color_temperature: float
    saturation_mean: float
    hue_histogram: np.ndarray

    def distance_to(self, other: "ColorPalette") -> float:
        """Calculate distance to another color palette.

        Uses weighted combination of dominant color distance,
        white/black point shifts, and saturation change.

        Args:
            other: Another ColorPalette to compare to.

        Returns:
            Distance metric (0 = identical, higher = more different).
        """
        # Dominant color distance (Earth Mover's Distance approximation)
        color_dist = np.mean(
            np.min(
                np.linalg.norm(
                    self.dominant_colors[:, np.newaxis, :]
                    - other.dominant_colors[np.newaxis, :, :],
                    axis=2,
                ),
                axis=1,
            )
        ) / 255.0

        # White/black point shift
        wp_dist = np.linalg.norm(self.white_point - other.white_point) / (
            255.0 * np.sqrt(3)
        )
        bp_dist = np.linalg.norm(self.black_point - other.black_point) / (
            255.0 * np.sqrt(3)
        )

        # Saturation difference
        sat_dist = abs(self.saturation_mean - other.saturation_mean)

        # Color temperature difference (normalized)
        temp_dist = abs(self.color_temperature - other.color_temperature) / 10000.0

        # Weighted combination
        return (
            0.4 * color_dist
            + 0.2 * wp_dist
            + 0.1 * bp_dist
            + 0.2 * sat_dist
            + 0.1 * temp_dist
        )


@dataclass
class GrainProfile:
    """Film grain / noise profile characteristics.

    Attributes:
        noise_mean: Mean noise level per channel.
        noise_std: Standard deviation of noise per channel.
        frequency_profile: Noise power spectrum (spatial frequencies).
        grain_size: Estimated grain size in pixels.
        color_noise_ratio: Ratio of chroma to luma noise.
    """

    noise_mean: np.ndarray
    noise_std: np.ndarray
    frequency_profile: np.ndarray
    grain_size: float
    color_noise_ratio: float

    def similarity_to(self, other: "GrainProfile") -> float:
        """Calculate similarity to another grain profile.

        Args:
            other: Another GrainProfile to compare to.

        Returns:
            Similarity score (0-1, 1 = identical).
        """
        # Compare noise statistics
        mean_sim = 1.0 - np.mean(np.abs(self.noise_mean - other.noise_mean)) / 50.0
        std_sim = 1.0 - np.mean(np.abs(self.noise_std - other.noise_std)) / 30.0

        # Compare grain size
        size_sim = 1.0 - abs(self.grain_size - other.grain_size) / 5.0

        # Compare color noise ratio
        ratio_sim = 1.0 - abs(self.color_noise_ratio - other.color_noise_ratio)

        return np.clip(0.3 * mean_sim + 0.3 * std_sim + 0.2 * size_sim + 0.2 * ratio_sim, 0, 1)


@dataclass
class FaceEmbedding:
    """Face embedding for identity tracking.

    Attributes:
        embedding: Feature vector for the face.
        bbox: Bounding box (x, y, width, height).
        confidence: Detection confidence.
        frame_index: Frame where this embedding was extracted.
    """

    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_index: int

    def similarity_to(self, other: "FaceEmbedding") -> float:
        """Calculate cosine similarity to another face embedding.

        Args:
            other: Another FaceEmbedding to compare to.

        Returns:
            Cosine similarity (0-1, 1 = same person).
        """
        if self.embedding.shape != other.embedding.shape:
            return 0.0

        norm1 = np.linalg.norm(self.embedding)
        norm2 = np.linalg.norm(other.embedding)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(self.embedding, other.embedding) / (norm1 * norm2))


@dataclass
class GlobalAnchors:
    """Global anchors extracted from key frames before processing.

    This class holds reference data extracted from representative frames
    throughout the video. These anchors are used to maintain consistency
    during chunk-by-chunk processing.

    Attributes:
        color_palettes: List of color palettes from anchor frames.
        grain_profiles: List of grain profiles from anchor frames.
        face_embeddings: List of face embeddings for identity tracking.
        anchor_frame_indices: Frame indices where anchors were extracted.
        reference_palette: The primary reference color palette.
        reference_grain: The primary reference grain profile.
        video_duration_frames: Total number of frames in the video.
    """

    color_palettes: List[ColorPalette] = field(default_factory=list)
    grain_profiles: List[GrainProfile] = field(default_factory=list)
    face_embeddings: List[FaceEmbedding] = field(default_factory=list)
    anchor_frame_indices: List[int] = field(default_factory=list)
    reference_palette: Optional[ColorPalette] = None
    reference_grain: Optional[GrainProfile] = None
    video_duration_frames: int = 0

    @classmethod
    def extract_from_video(
        cls,
        video_path: Path,
        config: TemporalConsistencyConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> "GlobalAnchors":
        """Extract global anchors from a video file.

        Args:
            video_path: Path to the video file.
            config: Configuration for anchor extraction.
            progress_callback: Optional progress callback.

        Returns:
            GlobalAnchors instance with extracted data.
        """
        if not HAS_OPENCV:
            logger.warning("OpenCV not available, returning empty anchors")
            return cls()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        anchors = cls(video_duration_frames=total_frames)

        # Calculate anchor frame positions
        if total_frames <= config.anchor_sample_count:
            anchor_positions = list(range(total_frames))
        else:
            step = total_frames // config.anchor_sample_count
            anchor_positions = [i * step for i in range(config.anchor_sample_count)]
            # Always include first and last
            if 0 not in anchor_positions:
                anchor_positions.insert(0, 0)
            if total_frames - 1 not in anchor_positions:
                anchor_positions.append(total_frames - 1)

        logger.info(
            f"Extracting anchors from {len(anchor_positions)} frames "
            f"(total: {total_frames})"
        )

        for i, frame_idx in enumerate(anchor_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {frame_idx}")
                continue

            # Extract color palette
            palette = _extract_color_palette(frame)
            anchors.color_palettes.append(palette)

            # Extract grain profile
            grain = _extract_grain_profile(frame)
            anchors.grain_profiles.append(grain)

            # Extract face embeddings if enabled
            if config.face_tracking_enabled:
                faces = _extract_face_embeddings(frame, frame_idx)
                anchors.face_embeddings.extend(faces)

            anchors.anchor_frame_indices.append(frame_idx)

            if progress_callback:
                progress_callback((i + 1) / len(anchor_positions))

        cap.release()

        # Set reference palette (median of all palettes)
        if anchors.color_palettes:
            anchors.reference_palette = _compute_median_palette(anchors.color_palettes)

        # Set reference grain (median of all profiles)
        if anchors.grain_profiles:
            anchors.reference_grain = _compute_median_grain(anchors.grain_profiles)

        logger.info(
            f"Extracted {len(anchors.color_palettes)} palettes, "
            f"{len(anchors.grain_profiles)} grain profiles, "
            f"{len(anchors.face_embeddings)} face embeddings"
        )

        return anchors

    @classmethod
    def extract_from_frames(
        cls,
        frames_dir: Path,
        config: TemporalConsistencyConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> "GlobalAnchors":
        """Extract global anchors from a directory of frames.

        Args:
            frames_dir: Directory containing frame images.
            config: Configuration for anchor extraction.
            progress_callback: Optional progress callback.

        Returns:
            GlobalAnchors instance with extracted data.
        """
        if not HAS_OPENCV:
            logger.warning("OpenCV not available, returning empty anchors")
            return cls()

        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            raise ValueError(f"No frames found in {frames_dir}")

        total_frames = len(frames)
        anchors = cls(video_duration_frames=total_frames)

        # Calculate anchor frame positions
        if total_frames <= config.anchor_sample_count:
            anchor_indices = list(range(total_frames))
        else:
            step = total_frames // config.anchor_sample_count
            anchor_indices = [i * step for i in range(config.anchor_sample_count)]
            if 0 not in anchor_indices:
                anchor_indices.insert(0, 0)
            if total_frames - 1 not in anchor_indices:
                anchor_indices.append(total_frames - 1)

        logger.info(
            f"Extracting anchors from {len(anchor_indices)} frames "
            f"(total: {total_frames})"
        )

        for i, frame_idx in enumerate(anchor_indices):
            frame = cv2.imread(str(frames[frame_idx]))
            if frame is None:
                logger.warning(f"Could not read frame {frames[frame_idx]}")
                continue

            # Extract color palette
            palette = _extract_color_palette(frame)
            anchors.color_palettes.append(palette)

            # Extract grain profile
            grain = _extract_grain_profile(frame)
            anchors.grain_profiles.append(grain)

            # Extract face embeddings if enabled
            if config.face_tracking_enabled:
                faces = _extract_face_embeddings(frame, frame_idx)
                anchors.face_embeddings.extend(faces)

            anchors.anchor_frame_indices.append(frame_idx)

            if progress_callback:
                progress_callback((i + 1) / len(anchor_indices))

        # Set reference palette and grain
        if anchors.color_palettes:
            anchors.reference_palette = _compute_median_palette(anchors.color_palettes)

        if anchors.grain_profiles:
            anchors.reference_grain = _compute_median_grain(anchors.grain_profiles)

        return anchors

    def get_nearest_palette(self, frame_index: int) -> Optional[ColorPalette]:
        """Get the nearest color palette to a given frame index.

        Args:
            frame_index: Frame index to find nearest palette for.

        Returns:
            Nearest ColorPalette or None if no palettes available.
        """
        if not self.color_palettes or not self.anchor_frame_indices:
            return None

        # Find nearest anchor frame
        nearest_idx = min(
            range(len(self.anchor_frame_indices)),
            key=lambda i: abs(self.anchor_frame_indices[i] - frame_index),
        )

        return self.color_palettes[nearest_idx]

    def get_interpolated_palette(
        self, frame_index: int
    ) -> Optional[ColorPalette]:
        """Get interpolated color palette for a given frame index.

        Linearly interpolates between surrounding anchor palettes.

        Args:
            frame_index: Frame index to interpolate palette for.

        Returns:
            Interpolated ColorPalette or None if not enough data.
        """
        if len(self.color_palettes) < 2 or not self.anchor_frame_indices:
            return self.get_nearest_palette(frame_index)

        # Find surrounding anchor frames
        prev_idx = None
        next_idx = None

        for i, anchor_frame in enumerate(self.anchor_frame_indices):
            if anchor_frame <= frame_index:
                prev_idx = i
            if anchor_frame >= frame_index and next_idx is None:
                next_idx = i

        if prev_idx is None:
            return self.color_palettes[0]
        if next_idx is None:
            return self.color_palettes[-1]
        if prev_idx == next_idx:
            return self.color_palettes[prev_idx]

        # Interpolate
        prev_frame = self.anchor_frame_indices[prev_idx]
        next_frame = self.anchor_frame_indices[next_idx]
        t = (frame_index - prev_frame) / max(1, next_frame - prev_frame)

        return _interpolate_palettes(
            self.color_palettes[prev_idx], self.color_palettes[next_idx], t
        )


@dataclass
class ConsistencyResult:
    """Result of consistency processing for a chunk.

    Attributes:
        chunk_id: Chunk identifier.
        start_frame: Starting frame index.
        end_frame: Ending frame index (exclusive).
        frames_processed: Number of frames processed.
        color_drift_detected: Whether color drift was detected.
        drift_correction_applied: Whether drift correction was applied.
        drift_amount: Amount of drift detected (0-1).
        processing_time_seconds: Time taken to process chunk.
        output_paths: List of output frame paths.
    """

    chunk_id: int
    start_frame: int
    end_frame: int
    frames_processed: int = 0
    color_drift_detected: bool = False
    drift_correction_applied: bool = False
    drift_amount: float = 0.0
    processing_time_seconds: float = 0.0
    output_paths: List[Path] = field(default_factory=list)


class ColorConsistencyEnforcer:
    """Detect and correct color drift over time.

    This class monitors color characteristics across frames and applies
    smoothed corrections when drift from the reference palette exceeds
    configured thresholds.

    Example:
        >>> enforcer = ColorConsistencyEnforcer(reference_palette, config)
        >>> for frame in frames:
        ...     corrected = enforcer.process_frame(frame, frame_idx)
    """

    def __init__(
        self,
        reference_palette: ColorPalette,
        config: TemporalConsistencyConfig,
    ):
        """Initialize color consistency enforcer.

        Args:
            reference_palette: Reference color palette to maintain.
            config: Consistency configuration.
        """
        self.reference = reference_palette
        self.config = config

        # Drift tracking
        self._drift_history: Deque[float] = deque(maxlen=100)
        self._correction_history: Deque[np.ndarray] = deque(maxlen=20)
        self._last_palette: Optional[ColorPalette] = None

        # Correction strength mapping
        self._correction_strength = {
            DriftCorrectionMode.NONE: 0.0,
            DriftCorrectionMode.SUBTLE: 0.3,
            DriftCorrectionMode.BALANCED: 0.6,
            DriftCorrectionMode.AGGRESSIVE: 0.9,
        }

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[np.ndarray, float]:
        """Process a frame with color consistency enforcement.

        Args:
            frame: Input frame (BGR format).
            frame_index: Frame index for tracking.

        Returns:
            Tuple of (corrected frame, drift amount).
        """
        if not HAS_OPENCV or self.config.drift_correction_mode == DriftCorrectionMode.NONE:
            return frame, 0.0

        # Extract current palette
        current_palette = _extract_color_palette(frame)
        self._last_palette = current_palette

        # Calculate drift from reference
        drift = self.reference.distance_to(current_palette)
        self._drift_history.append(drift)

        # Check if correction needed
        if drift <= self.config.max_color_drift:
            return frame, drift

        # Apply correction
        strength = self._correction_strength[self.config.drift_correction_mode]
        corrected = self._apply_correction(frame, current_palette, strength)

        return corrected, drift

    def _apply_correction(
        self,
        frame: np.ndarray,
        current_palette: ColorPalette,
        strength: float,
    ) -> np.ndarray:
        """Apply color correction to bring frame closer to reference.

        Args:
            frame: Input frame.
            current_palette: Current frame's color palette.
            strength: Correction strength (0-1).

        Returns:
            Color-corrected frame.
        """
        # Convert to LAB for perceptually uniform correction
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # White balance correction
        wp_shift = (self.reference.white_point - current_palette.white_point) * strength
        lab[:, :, 0] += wp_shift[0] * 0.1  # L channel
        lab[:, :, 1] += wp_shift[1] * 0.5  # a channel
        lab[:, :, 2] += wp_shift[2] * 0.5  # b channel

        # Saturation correction
        sat_diff = self.reference.saturation_mean - current_palette.saturation_mean
        if abs(sat_diff) > 0.05:
            # Adjust a and b channels (color)
            lab[:, :, 1] *= 1 + sat_diff * strength
            lab[:, :, 2] *= 1 + sat_diff * strength

        # Clip and convert back
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Smooth correction by blending with history
        if self._correction_history:
            # Running average of corrections
            avg_correction = np.mean(
                [corrected.astype(np.float32) - frame.astype(np.float32)
                 for _ in range(min(5, len(self._correction_history)))],
                axis=0,
            )
            current_correction = corrected.astype(np.float32) - frame.astype(np.float32)
            smoothed_correction = 0.7 * current_correction + 0.3 * avg_correction
            corrected = np.clip(
                frame.astype(np.float32) + smoothed_correction, 0, 255
            ).astype(np.uint8)

        return corrected

    def get_drift_statistics(self) -> Dict[str, float]:
        """Get drift statistics from processing history.

        Returns:
            Dictionary with mean, max, and recent drift values.
        """
        if not self._drift_history:
            return {"mean": 0.0, "max": 0.0, "recent": 0.0}

        history = list(self._drift_history)
        return {
            "mean": float(np.mean(history)),
            "max": float(np.max(history)),
            "recent": float(history[-1]) if history else 0.0,
            "samples": len(history),
        }


class ChunkedProcessor:
    """Stream process entire video with chunk boundary blending.

    This class handles memory-efficient processing of long videos by
    processing in overlapping chunks and blending at boundaries.

    Example:
        >>> processor = ChunkedProcessor(config)
        >>> processor.set_process_fn(my_enhance_function)
        >>> for result in processor.process(input_dir, output_dir):
        ...     print(f"Chunk {result.chunk_id} complete")
    """

    def __init__(
        self,
        config: TemporalConsistencyConfig,
        anchors: Optional[GlobalAnchors] = None,
    ):
        """Initialize chunked processor.

        Args:
            config: Processing configuration.
            anchors: Optional pre-extracted global anchors.
        """
        self.config = config
        self.anchors = anchors
        self._process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

        # Processing state
        self._buffer: Deque[Tuple[int, np.ndarray]] = deque(
            maxlen=config.buffer_size
        )
        self._processed_chunks: List[ConsistencyResult] = []
        self._lock = threading.Lock()

        # Color enforcer
        self._color_enforcer: Optional[ColorConsistencyEnforcer] = None

    def set_process_fn(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Set the frame processing function.

        Args:
            fn: Function that takes a frame and returns processed frame.
        """
        self._process_fn = fn

    def set_anchors(self, anchors: GlobalAnchors) -> None:
        """Set or update global anchors.

        Args:
            anchors: GlobalAnchors instance.
        """
        self.anchors = anchors
        if anchors.reference_palette:
            self._color_enforcer = ColorConsistencyEnforcer(
                anchors.reference_palette, self.config
            )

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[ConsistencyResult, None, None]:
        """Process frames in overlapping chunks with blending.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional callback(progress, message).

        Yields:
            ConsistencyResult for each completed chunk.
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for chunked processing")

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get sorted frame list
        frames = sorted(input_dir.glob("*.png"))
        if not frames:
            frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            logger.warning(f"No frames found in {input_dir}")
            return

        total_frames = len(frames)
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap_frames

        # Calculate number of chunks
        effective_chunk_size = chunk_size - overlap
        num_chunks = (total_frames + effective_chunk_size - 1) // effective_chunk_size

        logger.info(
            f"Processing {total_frames} frames in {num_chunks} chunks "
            f"(chunk_size={chunk_size}, overlap={overlap})"
        )

        # Previous chunk's overlap frames for blending
        prev_overlap_frames: List[np.ndarray] = []
        prev_overlap_indices: List[int] = []

        for chunk_id in range(num_chunks):
            start_time = time.time()

            # Calculate frame range for this chunk
            start_idx = chunk_id * effective_chunk_size
            end_idx = min(start_idx + chunk_size, total_frames)

            # Load chunk frames
            chunk_frames: List[Tuple[int, np.ndarray]] = []
            for idx in range(start_idx, end_idx):
                frame = cv2.imread(str(frames[idx]))
                if frame is not None:
                    chunk_frames.append((idx, frame))

            if not chunk_frames:
                continue

            # Process chunk
            processed_frames = self._process_chunk(
                chunk_frames, prev_overlap_frames, prev_overlap_indices
            )

            # Save processed frames
            output_paths = []
            for idx, processed in processed_frames:
                output_path = output_dir / frames[idx].name
                cv2.imwrite(str(output_path), processed)
                output_paths.append(output_path)

            # Store overlap frames for next chunk
            if overlap > 0 and end_idx < total_frames:
                prev_overlap_frames = [
                    frame for _, frame in processed_frames[-overlap:]
                ]
                prev_overlap_indices = [idx for idx, _ in processed_frames[-overlap:]]
            else:
                prev_overlap_frames = []
                prev_overlap_indices = []

            # Calculate drift statistics
            drift_stats = {"mean": 0.0, "max": 0.0}
            if self._color_enforcer:
                drift_stats = self._color_enforcer.get_drift_statistics()

            # Create result
            result = ConsistencyResult(
                chunk_id=chunk_id,
                start_frame=start_idx,
                end_frame=end_idx,
                frames_processed=len(processed_frames),
                color_drift_detected=drift_stats["max"] > self.config.max_color_drift,
                drift_correction_applied=drift_stats["max"] > self.config.max_color_drift,
                drift_amount=drift_stats["mean"],
                processing_time_seconds=time.time() - start_time,
                output_paths=output_paths,
            )

            self._processed_chunks.append(result)

            if progress_callback:
                progress = (chunk_id + 1) / num_chunks
                progress_callback(
                    progress,
                    f"Chunk {chunk_id + 1}/{num_chunks}: {len(processed_frames)} frames"
                )

            # Cleanup for memory
            if self.config.enable_gpu:
                gc.collect()

            yield result

    def _process_chunk(
        self,
        chunk_frames: List[Tuple[int, np.ndarray]],
        prev_overlap_frames: List[np.ndarray],
        prev_overlap_indices: List[int],
    ) -> List[Tuple[int, np.ndarray]]:
        """Process a chunk of frames with optional blending.

        Args:
            chunk_frames: List of (index, frame) tuples for this chunk.
            prev_overlap_frames: Overlap frames from previous chunk.
            prev_overlap_indices: Indices of overlap frames from previous chunk.

        Returns:
            List of (index, processed_frame) tuples.
        """
        processed: List[Tuple[int, np.ndarray]] = []
        overlap = self.config.overlap_frames

        for i, (idx, frame) in enumerate(chunk_frames):
            # Apply main processing function
            if self._process_fn:
                frame = self._process_fn(frame)

            # Apply color consistency
            if self._color_enforcer:
                frame, _ = self._color_enforcer.process_frame(frame, idx)

            # Blend with previous chunk's overlap if in overlap region
            if prev_overlap_frames and i < overlap:
                blend_weight = i / max(1, overlap)
                prev_frame = prev_overlap_frames[i]

                # Linear interpolation blend
                frame = _blend_frames(prev_frame, frame, blend_weight)

            processed.append((idx, frame))

        return processed

    def get_results(self) -> List[ConsistencyResult]:
        """Get all processing results.

        Returns:
            List of ConsistencyResult for all processed chunks.
        """
        return self._processed_chunks.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with aggregate statistics.
        """
        if not self._processed_chunks:
            return {
                "chunks": 0,
                "frames": 0,
                "avg_time_per_chunk": 0.0,
                "total_time": 0.0,
            }

        total_frames = sum(r.frames_processed for r in self._processed_chunks)
        total_time = sum(r.processing_time_seconds for r in self._processed_chunks)
        drift_amounts = [r.drift_amount for r in self._processed_chunks]

        return {
            "chunks": len(self._processed_chunks),
            "frames": total_frames,
            "avg_time_per_chunk": total_time / len(self._processed_chunks),
            "total_time": total_time,
            "avg_drift": float(np.mean(drift_amounts)) if drift_amounts else 0.0,
            "max_drift": float(np.max(drift_amounts)) if drift_amounts else 0.0,
            "chunks_with_correction": sum(
                1 for r in self._processed_chunks if r.drift_correction_applied
            ),
        }


class LongFormConsistencyManager:
    """Coordinate consistency across thousands of frames.

    This is the main entry point for long-form video consistency processing.
    It combines anchor extraction, color enforcement, and chunked processing
    into a unified pipeline.

    Example:
        >>> manager = LongFormConsistencyManager(config)
        >>> manager.initialize_from_video(video_path)
        >>> for result in manager.process_streaming(input_dir, output_dir):
        ...     print(f"Chunk {result.chunk_id}: drift={result.drift_amount:.3f}")
    """

    def __init__(
        self,
        config: Optional[TemporalConsistencyConfig] = None,
    ):
        """Initialize long-form consistency manager.

        Args:
            config: Configuration for consistency processing.
        """
        self.config = config or TemporalConsistencyConfig()
        self.anchors: Optional[GlobalAnchors] = None
        self._processor: Optional[ChunkedProcessor] = None
        self._initialized = False

    def initialize_from_video(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> GlobalAnchors:
        """Pre-analyze video to extract global anchors.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional progress callback.

        Returns:
            Extracted GlobalAnchors.
        """
        logger.info(f"Initializing from video: {video_path}")

        self.anchors = GlobalAnchors.extract_from_video(
            video_path, self.config, progress_callback
        )

        self._processor = ChunkedProcessor(self.config, self.anchors)
        self._initialized = True

        logger.info(
            f"Initialization complete: {len(self.anchors.color_palettes)} anchor frames"
        )

        return self.anchors

    def initialize_from_frames(
        self,
        frames_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> GlobalAnchors:
        """Pre-analyze frames directory to extract global anchors.

        Args:
            frames_dir: Directory containing frame images.
            progress_callback: Optional progress callback.

        Returns:
            Extracted GlobalAnchors.
        """
        logger.info(f"Initializing from frames: {frames_dir}")

        self.anchors = GlobalAnchors.extract_from_frames(
            frames_dir, self.config, progress_callback
        )

        self._processor = ChunkedProcessor(self.config, self.anchors)
        self._initialized = True

        logger.info(
            f"Initialization complete: {len(self.anchors.color_palettes)} anchor frames"
        )

        return self.anchors

    def set_process_fn(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Set the frame processing function.

        Args:
            fn: Function that takes a frame and returns processed frame.
        """
        if self._processor:
            self._processor.set_process_fn(fn)

    def process_chunk_with_consistency(
        self,
        chunk_frames: List[np.ndarray],
        chunk_start_index: int,
        process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Process a chunk of frames while maintaining consistency.

        This method is useful for integrating with existing processing pipelines.

        Args:
            chunk_frames: List of frames in the chunk.
            chunk_start_index: Starting frame index for this chunk.
            process_fn: Optional processing function to apply.

        Returns:
            List of processed frames with consistency applied.
        """
        if not self._initialized or not self.anchors:
            logger.warning("Manager not initialized, returning unprocessed frames")
            return chunk_frames

        # Create color enforcer if needed
        if not self.anchors.reference_palette:
            logger.warning("No reference palette, returning unprocessed frames")
            return chunk_frames

        enforcer = ColorConsistencyEnforcer(
            self.anchors.reference_palette, self.config
        )

        processed = []
        for i, frame in enumerate(chunk_frames):
            frame_idx = chunk_start_index + i

            # Apply main processing
            if process_fn:
                frame = process_fn(frame)

            # Apply color consistency
            frame, _ = enforcer.process_frame(frame, frame_idx)

            processed.append(frame)

        return processed

    def process_streaming(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[ConsistencyResult, None, None]:
        """Process frames with streaming output.

        Args:
            input_dir: Directory containing input frames.
            output_dir: Directory for output frames.
            progress_callback: Optional callback(progress, message).

        Yields:
            ConsistencyResult for each completed chunk.
        """
        if not self._initialized:
            logger.info("Auto-initializing from input frames")
            self.initialize_from_frames(input_dir)

        if not self._processor:
            raise RuntimeError("Processor not initialized")

        yield from self._processor.process(input_dir, output_dir, progress_callback)

    def get_anchors(self) -> Optional[GlobalAnchors]:
        """Get extracted global anchors.

        Returns:
            GlobalAnchors or None if not initialized.
        """
        return self.anchors

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with aggregate statistics.
        """
        stats: Dict[str, Any] = {
            "initialized": self._initialized,
            "anchor_frames": 0,
            "total_frames": 0,
        }

        if self.anchors:
            stats["anchor_frames"] = len(self.anchors.anchor_frame_indices)
            stats["total_frames"] = self.anchors.video_duration_frames
            stats["face_embeddings"] = len(self.anchors.face_embeddings)

        if self._processor:
            stats["processing"] = self._processor.get_statistics()

        return stats


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_color_palette(frame: np.ndarray, n_colors: int = 5) -> ColorPalette:
    """Extract color palette from a frame.

    Args:
        frame: BGR frame.
        n_colors: Number of dominant colors to extract.

    Returns:
        ColorPalette with extracted data.
    """
    # Convert to RGB for processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Resize for faster processing
    scale = min(1.0, 200.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(rgb, None, fx=scale, fy=scale)
    else:
        small = rgb

    # Flatten pixels
    pixels = small.reshape(-1, 3).astype(np.float32)

    # Extract dominant colors using K-means
    if HAS_SKLEARN and len(pixels) > n_colors:
        kmeans = KMeans(n_clusters=n_colors, n_init=3, max_iter=50, random_state=42)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
    else:
        # Fallback: simple quantile-based extraction
        dominant_colors = np.array([
            np.percentile(pixels, p, axis=0)
            for p in np.linspace(10, 90, n_colors)
        ])

    # White and black points (brightest and darkest pixels)
    brightness = np.sum(pixels, axis=1)
    white_point = pixels[np.argmax(brightness)]
    black_point = pixels[np.argmin(brightness)]

    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    saturation_mean = np.mean(hsv[:, :, 1]) / 255.0

    # Hue histogram
    hue_histogram = np.histogram(hsv[:, :, 0], bins=180, range=(0, 180))[0]
    hue_histogram = hue_histogram.astype(np.float32) / max(1, hue_histogram.sum())

    # Estimate color temperature (simplified)
    avg_color = np.mean(pixels, axis=0)
    r_b_ratio = (avg_color[0] + 1) / (avg_color[2] + 1)
    # Approximate mapping: ratio 1.0 = 6500K, higher = warmer, lower = cooler
    color_temperature = 6500 * r_b_ratio

    return ColorPalette(
        dominant_colors=dominant_colors.astype(np.uint8),
        white_point=white_point.astype(np.uint8),
        black_point=black_point.astype(np.uint8),
        color_temperature=float(np.clip(color_temperature, 2000, 15000)),
        saturation_mean=float(saturation_mean),
        hue_histogram=hue_histogram,
    )


def _extract_grain_profile(frame: np.ndarray) -> GrainProfile:
    """Extract film grain / noise profile from a frame.

    Args:
        frame: BGR frame.

    Returns:
        GrainProfile with extracted characteristics.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply high-pass filter to isolate noise
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)

    # Noise statistics
    noise_mean = np.mean(np.abs(noise))
    noise_std = np.std(noise)

    # Per-channel noise
    channels = cv2.split(frame)
    channel_noise_mean = []
    channel_noise_std = []

    for ch in channels:
        ch_blur = cv2.GaussianBlur(ch, (21, 21), 0)
        ch_noise = ch.astype(np.float32) - ch_blur.astype(np.float32)
        channel_noise_mean.append(np.mean(np.abs(ch_noise)))
        channel_noise_std.append(np.std(ch_noise))

    # Frequency profile using FFT
    f_transform = np.fft.fft2(noise)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Radial average of frequency spectrum
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    frequency_profile = np.zeros(max_radius)

    for r in range(max_radius):
        mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = (dist >= r) & (dist < r + 1)
        if mask.any():
            frequency_profile[r] = np.mean(magnitude[mask])

    # Normalize frequency profile
    if frequency_profile.max() > 0:
        frequency_profile = frequency_profile / frequency_profile.max()

    # Estimate grain size from frequency rolloff
    rolloff_idx = np.argmax(frequency_profile < 0.5) if (frequency_profile < 0.5).any() else len(frequency_profile) // 2
    grain_size = max_radius / max(1, rolloff_idx)

    # Color noise ratio (chroma noise / luma noise)
    luma_noise = channel_noise_std[0]  # Assuming BGR, use B as proxy
    chroma_noise = (channel_noise_std[1] + channel_noise_std[2]) / 2
    color_noise_ratio = chroma_noise / max(1, luma_noise)

    return GrainProfile(
        noise_mean=np.array(channel_noise_mean),
        noise_std=np.array(channel_noise_std),
        frequency_profile=frequency_profile,
        grain_size=float(grain_size),
        color_noise_ratio=float(np.clip(color_noise_ratio, 0, 2)),
    )


def _extract_face_embeddings(
    frame: np.ndarray,
    frame_index: int,
) -> List[FaceEmbedding]:
    """Extract face embeddings from a frame.

    Args:
        frame: BGR frame.
        frame_index: Frame index for tracking.

    Returns:
        List of FaceEmbedding objects found in the frame.
    """
    embeddings: List[FaceEmbedding] = []

    if not HAS_OPENCV:
        return embeddings

    # Use OpenCV's face detector (Haar cascade as fallback)
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except Exception:
        return embeddings

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        # Extract face region
        face_roi = frame[y : y + h, x : x + w]

        # Create simple embedding (histogram-based for simplicity)
        # In production, use a proper face embedding model
        face_resized = cv2.resize(face_roi, (64, 64))
        face_hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)

        # Concatenate color histograms as simple embedding
        h_hist = cv2.calcHist([face_hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([face_hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([face_hsv], [2], None, [32], [0, 256]).flatten()

        embedding = np.concatenate([h_hist, s_hist, v_hist])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        embeddings.append(
            FaceEmbedding(
                embedding=embedding,
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=0.8,  # Placeholder confidence
                frame_index=frame_index,
            )
        )

    return embeddings


def _compute_median_palette(palettes: List[ColorPalette]) -> ColorPalette:
    """Compute median color palette from a list of palettes.

    Args:
        palettes: List of ColorPalette objects.

    Returns:
        Median ColorPalette.
    """
    if not palettes:
        raise ValueError("No palettes to compute median from")

    if len(palettes) == 1:
        return palettes[0]

    # Stack arrays and compute median
    dominant_colors = np.median(
        [p.dominant_colors for p in palettes], axis=0
    ).astype(np.uint8)
    white_point = np.median(
        [p.white_point for p in palettes], axis=0
    ).astype(np.uint8)
    black_point = np.median(
        [p.black_point for p in palettes], axis=0
    ).astype(np.uint8)
    color_temperature = float(np.median([p.color_temperature for p in palettes]))
    saturation_mean = float(np.median([p.saturation_mean for p in palettes]))
    hue_histogram = np.median([p.hue_histogram for p in palettes], axis=0)

    return ColorPalette(
        dominant_colors=dominant_colors,
        white_point=white_point,
        black_point=black_point,
        color_temperature=color_temperature,
        saturation_mean=saturation_mean,
        hue_histogram=hue_histogram,
    )


def _compute_median_grain(profiles: List[GrainProfile]) -> GrainProfile:
    """Compute median grain profile from a list of profiles.

    Args:
        profiles: List of GrainProfile objects.

    Returns:
        Median GrainProfile.
    """
    if not profiles:
        raise ValueError("No profiles to compute median from")

    if len(profiles) == 1:
        return profiles[0]

    noise_mean = np.median([p.noise_mean for p in profiles], axis=0)
    noise_std = np.median([p.noise_std for p in profiles], axis=0)

    # Handle potentially different frequency profile lengths
    min_len = min(len(p.frequency_profile) for p in profiles)
    frequency_profiles = [p.frequency_profile[:min_len] for p in profiles]
    frequency_profile = np.median(frequency_profiles, axis=0)

    grain_size = float(np.median([p.grain_size for p in profiles]))
    color_noise_ratio = float(np.median([p.color_noise_ratio for p in profiles]))

    return GrainProfile(
        noise_mean=noise_mean,
        noise_std=noise_std,
        frequency_profile=frequency_profile,
        grain_size=grain_size,
        color_noise_ratio=color_noise_ratio,
    )


def _interpolate_palettes(
    palette1: ColorPalette,
    palette2: ColorPalette,
    t: float,
) -> ColorPalette:
    """Linearly interpolate between two color palettes.

    Args:
        palette1: First palette (t=0).
        palette2: Second palette (t=1).
        t: Interpolation factor (0-1).

    Returns:
        Interpolated ColorPalette.
    """
    t = np.clip(t, 0, 1)

    return ColorPalette(
        dominant_colors=(
            (1 - t) * palette1.dominant_colors + t * palette2.dominant_colors
        ).astype(np.uint8),
        white_point=(
            (1 - t) * palette1.white_point + t * palette2.white_point
        ).astype(np.uint8),
        black_point=(
            (1 - t) * palette1.black_point + t * palette2.black_point
        ).astype(np.uint8),
        color_temperature=(
            (1 - t) * palette1.color_temperature + t * palette2.color_temperature
        ),
        saturation_mean=(
            (1 - t) * palette1.saturation_mean + t * palette2.saturation_mean
        ),
        hue_histogram=(
            (1 - t) * palette1.hue_histogram + t * palette2.hue_histogram
        ),
    )


def _blend_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend two frames with linear interpolation.

    Args:
        frame1: First frame.
        frame2: Second frame.
        weight: Weight for frame2 (0 = frame1, 1 = frame2).

    Returns:
        Blended frame.
    """
    weight = np.clip(weight, 0, 1)

    if frame1.shape != frame2.shape:
        # Resize frame2 to match frame1 if needed
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    blended = (
        (1 - weight) * frame1.astype(np.float32)
        + weight * frame2.astype(np.float32)
    )

    return np.clip(blended, 0, 255).astype(np.uint8)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_consistency_manager(
    chunk_size: int = 50,
    overlap_frames: int = 4,
    max_color_drift: float = 0.15,
    correction_mode: str = "balanced",
) -> LongFormConsistencyManager:
    """Create a consistency manager with common settings.

    Args:
        chunk_size: Frames per processing chunk.
        overlap_frames: Overlap for chunk blending.
        max_color_drift: Maximum allowed color drift before correction.
        correction_mode: One of "none", "subtle", "balanced", "aggressive".

    Returns:
        Configured LongFormConsistencyManager.
    """
    config = TemporalConsistencyConfig(
        chunk_size=chunk_size,
        overlap_frames=overlap_frames,
        max_color_drift=max_color_drift,
        drift_correction_mode=DriftCorrectionMode(correction_mode),
    )
    return LongFormConsistencyManager(config)


def process_long_video_with_consistency(
    input_dir: Path,
    output_dir: Path,
    process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    chunk_size: int = 50,
    overlap_frames: int = 4,
) -> List[ConsistencyResult]:
    """Process a long video with temporal consistency.

    This is a convenience function that handles the full pipeline:
    1. Extracts global anchors
    2. Processes frames in chunks with blending
    3. Applies color drift correction

    Args:
        input_dir: Directory containing input frames.
        output_dir: Directory for output frames.
        process_fn: Optional frame processing function.
        progress_callback: Optional callback(progress, message).
        chunk_size: Frames per chunk.
        overlap_frames: Overlap for blending.

    Returns:
        List of ConsistencyResult for all processed chunks.
    """
    manager = create_consistency_manager(
        chunk_size=chunk_size,
        overlap_frames=overlap_frames,
    )

    # Initialize from input frames
    manager.initialize_from_frames(input_dir)

    # Set processing function
    if process_fn:
        manager.set_process_fn(process_fn)

    # Process and collect results
    results = []
    for result in manager.process_streaming(input_dir, output_dir, progress_callback):
        results.append(result)

    return results
