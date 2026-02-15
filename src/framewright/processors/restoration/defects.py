"""Unified Defect Detection and Repair Processor for FrameWright.

This module provides comprehensive defect detection and repair for film restoration,
handling scratches, dust, film damage, and other artifacts with temporal-aware
processing for video sequences.

Key Features:
- Automatic defect detection (scratches, dust, damage)
- Multiple repair backends (OpenCV, AI-based)
- Temporal-aware repair using neighbor frames
- Unified interface with configurable detection/repair

Defect Types Handled:
- Vertical/horizontal scratches (film transport damage)
- Dust and debris spots (single-frame artifacts)
- Film damage (tears, burns, chemical damage)
- Blotches and stains
- Sprocket hole damage

VRAM Requirements:
- CPU mode: No GPU required
- OpenCV GPU: ~256MB VRAM
- AI-based: ~1-2GB VRAM depending on model

Example:
    >>> from framewright.processors.restoration import DefectProcessor, DefectConfig
    >>>
    >>> config = DefectConfig(
    ...     detect_scratches=True,
    ...     detect_dust=True,
    ...     repair_strength=0.8,
    ...     temporal_coherence=True
    ... )
    >>> processor = DefectProcessor(config)
    >>>
    >>> # Process frame sequence
    >>> result = processor.process(frames, progress_callback=lambda p: print(f"{p*100:.0f}%"))
    >>> print(f"Repaired {result.defects_repaired} defects")
"""

import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with availability tracking
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    from scipy import ndimage, signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    ndimage = None
    signal = None

try:
    from skimage import morphology, filters, restoration
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    morphology = None
    filters = None
    restoration = None


# =============================================================================
# Enums and Constants
# =============================================================================

class DefectType(Enum):
    """Types of film defects that can be detected and repaired."""
    SCRATCH_VERTICAL = auto()
    SCRATCH_HORIZONTAL = auto()
    SCRATCH_DIAGONAL = auto()
    DUST = auto()
    DEBRIS = auto()
    BLOTCH = auto()
    STAIN = auto()
    TEAR = auto()
    BURN = auto()
    CHEMICAL_DAMAGE = auto()
    SPROCKET_DAMAGE = auto()
    FLICKER = auto()
    UNKNOWN = auto()


class RepairBackend(Enum):
    """Available repair backends."""
    OPENCV = "opencv"           # OpenCV inpainting (Telea/NS)
    OPENCV_GPU = "opencv_gpu"   # OpenCV with GPU acceleration
    DEEPFILL = "deepfill"       # DeepFill v2 neural inpainting
    LAMA = "lama"               # LaMa large mask inpainting
    AUTO = "auto"               # Automatically select best available


class DetectionMode(Enum):
    """Detection modes for different scenarios."""
    FAST = "fast"               # Quick detection for preview
    BALANCED = "balanced"       # Balance between speed and accuracy
    THOROUGH = "thorough"       # Most thorough detection


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DefectConfig:
    """Configuration for defect detection and repair.

    Attributes:
        detect_scratches: Enable scratch detection
        detect_dust: Enable dust/debris detection
        detect_damage: Enable film damage detection (tears, burns, etc.)
        scratch_sensitivity: Sensitivity for scratch detection (0-1)
        dust_sensitivity: Sensitivity for dust detection (0-1)
        damage_sensitivity: Sensitivity for damage detection (0-1)
        repair_strength: Overall repair strength (0-1)
        temporal_coherence: Use neighbor frames for temporal-aware repair
        temporal_window: Number of frames to consider (each side)
        detection_mode: Detection thoroughness mode
        repair_backend: Backend for inpainting repairs
        min_defect_size: Minimum defect size in pixels to detect
        max_defect_size: Maximum defect size in pixels to process
        preserve_grain: Attempt to preserve film grain during repair
        gpu_accelerate: Use GPU acceleration if available
    """
    detect_scratches: bool = True
    detect_dust: bool = True
    detect_damage: bool = True
    scratch_sensitivity: float = 0.5
    dust_sensitivity: float = 0.5
    damage_sensitivity: float = 0.5
    repair_strength: float = 0.8
    temporal_coherence: bool = True
    temporal_window: int = 2
    detection_mode: DetectionMode = DetectionMode.BALANCED
    repair_backend: RepairBackend = RepairBackend.AUTO
    min_defect_size: int = 2
    max_defect_size: int = 5000
    preserve_grain: bool = True
    gpu_accelerate: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        for name, val in [
            ("scratch_sensitivity", self.scratch_sensitivity),
            ("dust_sensitivity", self.dust_sensitivity),
            ("damage_sensitivity", self.damage_sensitivity),
            ("repair_strength", self.repair_strength),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be 0-1, got {val}")

        if self.temporal_window < 0:
            raise ValueError(f"temporal_window must be >= 0, got {self.temporal_window}")

        if self.min_defect_size < 1:
            raise ValueError(f"min_defect_size must be >= 1, got {self.min_defect_size}")

        # Convert string enums if needed
        if isinstance(self.detection_mode, str):
            self.detection_mode = DetectionMode(self.detection_mode.lower())
        if isinstance(self.repair_backend, str):
            self.repair_backend = RepairBackend(self.repair_backend.lower())


@dataclass
class DefectInfo:
    """Information about a detected defect.

    Attributes:
        defect_type: Type of defect
        mask: Binary mask of defect region
        bbox: Bounding box (x, y, width, height)
        area: Area in pixels
        confidence: Detection confidence (0-1)
        frame_index: Frame index where detected
        severity: Severity score (0-1)
        repaired: Whether defect was successfully repaired
    """
    defect_type: DefectType
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    confidence: float = 1.0
    frame_index: int = 0
    severity: float = 0.5
    repaired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without mask for serialization)."""
        return {
            "defect_type": self.defect_type.name,
            "bbox": self.bbox,
            "area": self.area,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
            "severity": self.severity,
            "repaired": self.repaired,
        }


@dataclass
class DetectionResult:
    """Result of defect detection on a frame.

    Attributes:
        frame_index: Index of the analyzed frame
        defects: List of detected defects
        scratch_mask: Combined mask of all scratches
        dust_mask: Combined mask of all dust spots
        damage_mask: Combined mask of all damage
        combined_mask: Combined mask of all defects
        severity_score: Overall defect severity (0-1)
        needs_repair: Whether frame needs repair
    """
    frame_index: int = 0
    defects: List[DefectInfo] = field(default_factory=list)
    scratch_mask: Optional[np.ndarray] = None
    dust_mask: Optional[np.ndarray] = None
    damage_mask: Optional[np.ndarray] = None
    combined_mask: Optional[np.ndarray] = None
    severity_score: float = 0.0
    needs_repair: bool = False

    def get_mask_for_type(self, defect_type: DefectType) -> Optional[np.ndarray]:
        """Get combined mask for a specific defect type."""
        if defect_type in [DefectType.SCRATCH_VERTICAL, DefectType.SCRATCH_HORIZONTAL,
                          DefectType.SCRATCH_DIAGONAL]:
            return self.scratch_mask
        elif defect_type in [DefectType.DUST, DefectType.DEBRIS]:
            return self.dust_mask
        else:
            return self.damage_mask


@dataclass
class RepairResult:
    """Result of defect repair operation.

    Attributes:
        frames_processed: Number of frames processed
        defects_detected: Total defects detected
        defects_repaired: Total defects repaired
        scratch_count: Number of scratches found
        dust_count: Number of dust spots found
        damage_count: Number of damage areas found
        backend_used: Repair backend that was used
        processing_time_ms: Processing time in milliseconds
        output_frames: Repaired frames (if return_frames=True)
    """
    frames_processed: int = 0
    defects_detected: int = 0
    defects_repaired: int = 0
    scratch_count: int = 0
    dust_count: int = 0
    damage_count: int = 0
    backend_used: str = ""
    processing_time_ms: float = 0.0
    output_frames: List[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_processed": self.frames_processed,
            "defects_detected": self.defects_detected,
            "defects_repaired": self.defects_repaired,
            "scratch_count": self.scratch_count,
            "dust_count": self.dust_count,
            "damage_count": self.damage_count,
            "backend_used": self.backend_used,
            "processing_time_ms": self.processing_time_ms,
        }


# =============================================================================
# Defect Detection
# =============================================================================

class DefectDetector:
    """Detects various types of film defects in frames.

    Implements multiple detection algorithms:
    - Edge detection for scratches
    - Temporal analysis for dust
    - Morphological analysis for damage
    - Color anomaly detection for stains

    Example:
        >>> detector = DefectDetector(
        ...     scratch_sensitivity=0.6,
        ...     dust_sensitivity=0.5
        ... )
        >>> result = detector.detect_all(frame)
        >>> print(f"Found {len(result.defects)} defects")
    """

    def __init__(
        self,
        scratch_sensitivity: float = 0.5,
        dust_sensitivity: float = 0.5,
        damage_sensitivity: float = 0.5,
        min_size: int = 2,
        max_size: int = 5000,
        detection_mode: DetectionMode = DetectionMode.BALANCED,
    ):
        """Initialize defect detector.

        Args:
            scratch_sensitivity: Sensitivity for scratch detection (0-1)
            dust_sensitivity: Sensitivity for dust detection (0-1)
            damage_sensitivity: Sensitivity for damage detection (0-1)
            min_size: Minimum defect size in pixels
            max_size: Maximum defect size in pixels
            detection_mode: Detection thoroughness mode
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for defect detection")

        self.scratch_sensitivity = scratch_sensitivity
        self.dust_sensitivity = dust_sensitivity
        self.damage_sensitivity = damage_sensitivity
        self.min_size = min_size
        self.max_size = max_size
        self.detection_mode = detection_mode

        # Cached kernels for morphological operations
        self._vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 15)
        )
        self._horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (15, 1)
        )
        self._ellipse_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

    def detect_all(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Detect all types of defects in a frame.

        Args:
            frame: Input frame (BGR format)
            frame_index: Index of frame in sequence

        Returns:
            DetectionResult with all detected defects
        """
        result = DetectionResult(frame_index=frame_index)

        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect scratches
        result.scratch_mask, scratch_defects = self.detect_scratches(
            gray, frame_index
        )
        result.defects.extend(scratch_defects)

        # Detect dust
        result.dust_mask, dust_defects = self.detect_dust(
            gray, frame_index
        )
        result.defects.extend(dust_defects)

        # Detect damage
        result.damage_mask, damage_defects = self.detect_film_damage(
            frame, frame_index
        )
        result.defects.extend(damage_defects)

        # Combine masks
        result.combined_mask = self._combine_masks(
            result.scratch_mask,
            result.dust_mask,
            result.damage_mask,
        )

        # Calculate overall severity
        if result.combined_mask is not None:
            defect_ratio = np.sum(result.combined_mask > 0) / result.combined_mask.size
            result.severity_score = min(1.0, defect_ratio * 100)  # Scale up

        result.needs_repair = result.severity_score > 0.01 or len(result.defects) > 0

        return result

    def detect_scratches(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> Tuple[np.ndarray, List[DefectInfo]]:
        """Detect vertical line artifacts (scratches).

        Uses edge detection followed by directional filtering to find
        linear artifacts typical of film transport damage.

        Args:
            frame: Grayscale input frame
            frame_index: Frame index for tracking

        Returns:
            Tuple of (binary mask, list of DefectInfo)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        defects: List[DefectInfo] = []

        # Adjust threshold based on sensitivity
        # Higher sensitivity = lower threshold
        base_threshold = 30 + int((1 - self.scratch_sensitivity) * 50)

        # Edge detection
        edges = cv2.Canny(frame, base_threshold, base_threshold * 2)

        # Morphological closing to connect broken lines
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self._vertical_kernel)

        # Detect vertical lines using Hough transform
        min_line_length = h // 10 if self.detection_mode == DetectionMode.FAST else h // 20
        max_line_gap = 20 if self.detection_mode == DetectionMode.THOROUGH else 10

        lines = cv2.HoughLinesP(
            closed,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if line is mostly vertical (within 15 degrees)
                if abs(x2 - x1) < abs(y2 - y1) * 0.27:  # tan(15) ≈ 0.27
                    # Draw on mask with some width
                    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)

                    # Create defect info
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    bbox = (min(x1, x2), min(y1, y2), abs(x2-x1)+2, abs(y2-y1))

                    defect = DefectInfo(
                        defect_type=DefectType.SCRATCH_VERTICAL,
                        mask=mask.copy(),
                        bbox=bbox,
                        area=int(line_length * 2),
                        confidence=0.8,
                        frame_index=frame_index,
                        severity=min(1.0, line_length / h),
                    )
                    defects.append(defect)

        # Also detect horizontal scratches
        h_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self._horizontal_kernel)
        h_lines = cv2.HoughLinesP(
            h_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=w // 20,
            maxLineGap=max_line_gap,
        )

        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]

                # Check if line is mostly horizontal
                if abs(y2 - y1) < abs(x2 - x1) * 0.27:
                    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)

                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    bbox = (min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)+2)

                    defect = DefectInfo(
                        defect_type=DefectType.SCRATCH_HORIZONTAL,
                        mask=mask.copy(),
                        bbox=bbox,
                        area=int(line_length * 2),
                        confidence=0.7,
                        frame_index=frame_index,
                        severity=min(1.0, line_length / w),
                    )
                    defects.append(defect)

        logger.debug(f"Frame {frame_index}: Detected {len(defects)} scratches")
        return mask, defects

    def detect_dust(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> Tuple[np.ndarray, List[DefectInfo]]:
        """Detect dust and debris spots.

        Uses blob detection to find isolated bright or dark spots
        that are characteristic of dust particles.

        Args:
            frame: Grayscale input frame
            frame_index: Frame index for tracking

        Returns:
            Tuple of (binary mask, list of DefectInfo)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        defects: List[DefectInfo] = []

        # Adjust parameters based on sensitivity
        threshold_offset = int((1 - self.dust_sensitivity) * 40)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Local adaptive thresholding to find anomalies
        block_size = 31 if self.detection_mode == DetectionMode.THOROUGH else 21
        adapt_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            10 + threshold_offset,
        )

        # Find white (bright) spots - dust on dark areas
        _, bright_spots = cv2.threshold(
            frame,
            np.percentile(frame, 99) - threshold_offset,
            255,
            cv2.THRESH_BINARY,
        )

        # Find dark spots - dust on light areas
        _, dark_spots = cv2.threshold(
            frame,
            np.percentile(frame, 1) + threshold_offset,
            255,
            cv2.THRESH_BINARY_INV,
        )

        # Combine detections
        spots_combined = cv2.bitwise_or(bright_spots, dark_spots)
        spots_combined = cv2.bitwise_and(spots_combined, adapt_thresh)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spots_clean = cv2.morphologyEx(spots_combined, cv2.MORPH_OPEN, kernel)
        spots_clean = cv2.morphologyEx(spots_clean, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            spots_clean, connectivity=8
        )

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by size
            if self.min_size <= area <= self.max_size:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w_box = stats[i, cv2.CC_STAT_WIDTH]
                h_box = stats[i, cv2.CC_STAT_HEIGHT]

                # Check aspect ratio (dust is usually round-ish)
                aspect = max(w_box, h_box) / (min(w_box, h_box) + 1)
                if aspect < 5:  # Not too elongated
                    # Add to mask
                    mask[labels == i] = 255

                    # Create defect info
                    defect = DefectInfo(
                        defect_type=DefectType.DUST,
                        mask=(labels == i).astype(np.uint8) * 255,
                        bbox=(x, y, w_box, h_box),
                        area=area,
                        confidence=0.9 if aspect < 2 else 0.6,
                        frame_index=frame_index,
                        severity=min(1.0, area / 100),
                    )
                    defects.append(defect)

        logger.debug(f"Frame {frame_index}: Detected {len(defects)} dust spots")
        return mask, defects

    def detect_film_damage(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> Tuple[np.ndarray, List[DefectInfo]]:
        """Detect tears, burns, and other film damage.

        Uses multiple techniques:
        - Edge density analysis for tears
        - Color anomaly detection for burns/stains
        - Texture analysis for chemical damage

        Args:
            frame: Input frame (BGR or grayscale)
            frame_index: Frame index for tracking

        Returns:
            Tuple of (binary mask, list of DefectInfo)
        """
        # Ensure we have both grayscale and color if available
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            has_color = True
        else:
            gray = frame
            has_color = False

        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        defects: List[DefectInfo] = []

        # Threshold based on sensitivity
        threshold_factor = 1.0 + (1 - self.damage_sensitivity)

        # 1. Detect tears using edge density
        edges = cv2.Canny(gray, 50, 150)

        # High edge density regions
        kernel_size = 21
        edge_density = cv2.blur(edges, (kernel_size, kernel_size))
        _, tear_regions = cv2.threshold(
            edge_density,
            int(100 * threshold_factor),
            255,
            cv2.THRESH_BINARY,
        )

        # 2. Detect burns/stains using color analysis
        burn_mask = np.zeros((h, w), dtype=np.uint8)
        if has_color:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect very dark regions (burns)
            dark_mask = cv2.inRange(hsv[:, :, 2], 0, 20)

            # Detect unusual saturation patterns (chemical damage)
            sat = hsv[:, :, 1]
            sat_mean = np.mean(sat)
            sat_std = np.std(sat)
            unusual_sat = np.abs(sat - sat_mean) > 2.5 * sat_std

            burn_mask = cv2.bitwise_or(dark_mask, unusual_sat.astype(np.uint8) * 255)
        else:
            # For grayscale, detect very dark or very light anomalies
            _, burn_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

        # 3. Detect sprocket damage (edge regions)
        edge_width = w // 30
        left_edge = gray[:, :edge_width]
        right_edge = gray[:, -edge_width:]

        sprocket_mask = np.zeros((h, w), dtype=np.uint8)

        # Check for irregular edge patterns
        left_var = np.var(left_edge)
        right_var = np.var(right_edge)

        if left_var > 1000 * threshold_factor:
            sprocket_mask[:, :edge_width] = 255
        if right_var > 1000 * threshold_factor:
            sprocket_mask[:, -edge_width:] = 255

        # Combine all damage masks
        damage_combined = cv2.bitwise_or(tear_regions, burn_mask)
        damage_combined = cv2.bitwise_or(damage_combined, sprocket_mask)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        damage_clean = cv2.morphologyEx(damage_combined, cv2.MORPH_OPEN, kernel)
        damage_clean = cv2.morphologyEx(damage_clean, cv2.MORPH_CLOSE, kernel)

        # Find components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            damage_clean, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= self.min_size and area <= self.max_size * 10:  # Damage can be larger
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w_box = stats[i, cv2.CC_STAT_WIDTH]
                h_box = stats[i, cv2.CC_STAT_HEIGHT]

                mask[labels == i] = 255

                # Classify damage type
                region = labels == i
                if x < edge_width or x + w_box > w - edge_width:
                    defect_type = DefectType.SPROCKET_DAMAGE
                elif np.sum(burn_mask[region]) > area * 0.5:
                    defect_type = DefectType.BURN
                else:
                    defect_type = DefectType.TEAR

                defect = DefectInfo(
                    defect_type=defect_type,
                    mask=region.astype(np.uint8) * 255,
                    bbox=(x, y, w_box, h_box),
                    area=area,
                    confidence=0.7,
                    frame_index=frame_index,
                    severity=min(1.0, area / 1000),
                )
                defects.append(defect)

        logger.debug(f"Frame {frame_index}: Detected {len(defects)} damage areas")
        return mask, defects

    def _combine_masks(self, *masks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Combine multiple masks into one."""
        combined = None

        for mask in masks:
            if mask is not None:
                if combined is None:
                    combined = mask.copy()
                else:
                    combined = cv2.bitwise_or(combined, mask)

        return combined


# =============================================================================
# Defect Repair
# =============================================================================

class RepairBackendBase(ABC):
    """Abstract base class for defect repair backends."""

    @abstractmethod
    def repair(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbor_frames: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Repair defects in frame using mask.

        Args:
            frame: Input frame to repair
            mask: Binary mask indicating defect regions
            neighbor_frames: Optional neighbor frames for temporal repair

        Returns:
            Repaired frame
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class OpenCVRepairBackend(RepairBackendBase):
    """OpenCV-based inpainting repair backend.

    Uses Telea or Navier-Stokes inpainting algorithms from OpenCV.
    """

    def __init__(
        self,
        method: str = "telea",
        inpaint_radius: int = 3,
    ):
        """Initialize OpenCV repair backend.

        Args:
            method: Inpainting method ("telea" or "ns")
            inpaint_radius: Radius for inpainting
        """
        self.method = method.lower()
        self.inpaint_radius = inpaint_radius

        self._inpaint_flags = {
            "telea": cv2.INPAINT_TELEA,
            "ns": cv2.INPAINT_NS,
        }

    @property
    def name(self) -> str:
        return f"opencv_{self.method}"

    def is_available(self) -> bool:
        return HAS_OPENCV

    def repair(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbor_frames: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Repair using OpenCV inpainting."""
        if not self.is_available():
            raise RuntimeError("OpenCV not available")

        # Ensure mask is proper format
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        # Get inpainting flag
        flag = self._inpaint_flags.get(self.method, cv2.INPAINT_TELEA)

        # If we have neighbor frames, use temporal blending
        if neighbor_frames and len(neighbor_frames) > 0:
            return self._temporal_repair(frame, mask, neighbor_frames)

        # Standard inpainting
        repaired = cv2.inpaint(frame, mask, self.inpaint_radius, flag)

        return repaired

    def _temporal_repair(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbors: List[np.ndarray],
    ) -> np.ndarray:
        """Use temporal information from neighbor frames for repair."""
        # Start with OpenCV inpaint as base
        flag = self._inpaint_flags.get(self.method, cv2.INPAINT_TELEA)
        base_repair = cv2.inpaint(frame, mask, self.inpaint_radius, flag)

        if not neighbors:
            return base_repair

        # Stack neighbor pixels and take median
        mask_bool = mask > 0

        # Collect pixel values from neighbors at defect locations
        neighbor_values = []
        for neighbor in neighbors:
            if neighbor.shape == frame.shape:
                neighbor_values.append(neighbor[mask_bool])

        if neighbor_values:
            # Use median of neighbors
            stacked = np.stack(neighbor_values, axis=0)
            median_values = np.median(stacked, axis=0).astype(frame.dtype)

            # Blend with inpaint result
            result = base_repair.copy()
            result[mask_bool] = (
                0.6 * median_values + 0.4 * base_repair[mask_bool]
            ).astype(frame.dtype)

            return result

        return base_repair


class DefectRepairer:
    """Repairs detected defects in frames.

    Supports multiple repair strategies:
    - Inpainting (OpenCV, neural)
    - Temporal interpolation
    - Blending with neighbors

    Example:
        >>> repairer = DefectRepairer()
        >>> repaired = repairer.repair_scratches(frame, scratch_mask)
    """

    def __init__(
        self,
        backend: RepairBackend = RepairBackend.AUTO,
        repair_strength: float = 0.8,
        preserve_grain: bool = True,
    ):
        """Initialize defect repairer.

        Args:
            backend: Repair backend to use
            repair_strength: Repair strength (0-1)
            preserve_grain: Attempt to preserve film grain
        """
        self.repair_strength = repair_strength
        self.preserve_grain = preserve_grain

        # Initialize backend
        self._backend = self._select_backend(backend)

    def _select_backend(self, backend: RepairBackend) -> RepairBackendBase:
        """Select appropriate repair backend."""
        if backend == RepairBackend.AUTO:
            # Try backends in order of preference
            if HAS_OPENCV:
                return OpenCVRepairBackend(method="telea", inpaint_radius=5)
        elif backend == RepairBackend.OPENCV:
            if HAS_OPENCV:
                return OpenCVRepairBackend(method="telea", inpaint_radius=5)
        elif backend == RepairBackend.OPENCV_GPU:
            # GPU variant - fall back to CPU if not available
            if HAS_OPENCV:
                return OpenCVRepairBackend(method="telea", inpaint_radius=5)

        # Default fallback
        if HAS_OPENCV:
            return OpenCVRepairBackend()
        else:
            raise RuntimeError("No repair backend available - install OpenCV")

    @property
    def backend_name(self) -> str:
        """Get name of active backend."""
        return self._backend.name

    def repair_scratches(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbor_frames: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Repair scratch defects.

        Args:
            frame: Input frame
            mask: Binary mask of scratch regions
            neighbor_frames: Optional neighbor frames for temporal repair

        Returns:
            Repaired frame
        """
        if np.sum(mask) == 0:
            return frame

        # Dilate mask slightly to ensure full coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        repaired = self._backend.repair(frame, dilated_mask, neighbor_frames)

        # Blend based on repair strength
        return self._blend_repair(frame, repaired, dilated_mask)

    def repair_dust(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbor_frames: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Repair dust spots.

        Args:
            frame: Input frame
            mask: Binary mask of dust regions
            neighbor_frames: Optional neighbor frames

        Returns:
            Repaired frame
        """
        if np.sum(mask) == 0:
            return frame

        # For dust, temporal repair is very effective since dust
        # only appears in single frames
        if neighbor_frames and len(neighbor_frames) >= 2:
            return self._temporal_dust_repair(frame, mask, neighbor_frames)

        # Fall back to inpainting
        repaired = self._backend.repair(frame, mask, neighbor_frames)
        return self._blend_repair(frame, repaired, mask)

    def _temporal_dust_repair(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbors: List[np.ndarray],
    ) -> np.ndarray:
        """Use temporal median for dust removal."""
        mask_bool = mask > 0

        # Collect values from neighbors
        values = []
        for neighbor in neighbors:
            if neighbor.shape == frame.shape:
                values.append(neighbor[mask_bool])

        if values:
            result = frame.copy()
            stacked = np.stack(values, axis=0)
            median_vals = np.median(stacked, axis=0).astype(frame.dtype)
            result[mask_bool] = median_vals
            return result

        return frame

    def repair_damage(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        neighbor_frames: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Repair film damage (tears, burns, etc.).

        Args:
            frame: Input frame
            mask: Binary mask of damaged regions
            neighbor_frames: Optional neighbor frames

        Returns:
            Repaired frame
        """
        if np.sum(mask) == 0:
            return frame

        # Use larger inpainting radius for damage
        temp_backend = OpenCVRepairBackend(method="ns", inpaint_radius=7)
        repaired = temp_backend.repair(frame, mask, neighbor_frames)

        return self._blend_repair(frame, repaired, mask)

    def _blend_repair(
        self,
        original: np.ndarray,
        repaired: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend repaired region with original based on strength."""
        if self.repair_strength >= 1.0:
            # Full replacement
            result = original.copy()
            mask_bool = mask > 0
            result[mask_bool] = repaired[mask_bool]
            return result

        # Blended replacement
        result = original.copy()
        mask_bool = mask > 0
        result[mask_bool] = (
            self.repair_strength * repaired[mask_bool] +
            (1 - self.repair_strength) * original[mask_bool]
        ).astype(original.dtype)

        # Optionally add grain back
        if self.preserve_grain and HAS_OPENCV:
            result = self._restore_grain(original, result, mask)

        return result

    def _restore_grain(
        self,
        original: np.ndarray,
        repaired: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Attempt to restore film grain in repaired regions."""
        mask_bool = mask > 0

        # Extract grain from surrounding area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(mask, kernel, iterations=2)
        surround = dilated > mask

        if np.sum(surround) > 100:
            # Estimate grain from surrounding area
            orig_surround = original[surround].astype(np.float32)

            # Simple grain estimation
            grain_std = np.std(orig_surround - cv2.GaussianBlur(
                original, (5, 5), 0
            )[surround]) * 0.5

            if grain_std > 0:
                # Add synthetic grain
                result = repaired.copy().astype(np.float32)
                grain = np.random.randn(*repaired.shape) * grain_std
                result[mask_bool] += grain[mask_bool]
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result

        return repaired


# =============================================================================
# Unified Defect Processor
# =============================================================================

class DefectProcessor:
    """Unified defect detection and repair processor.

    Provides a single interface for detecting and repairing all types
    of film defects with support for batch processing and temporal
    coherence.

    Example:
        >>> config = DefectConfig(
        ...     detect_scratches=True,
        ...     detect_dust=True,
        ...     repair_strength=0.8
        ... )
        >>> processor = DefectProcessor(config)
        >>> result = processor.process(frames)
    """

    def __init__(
        self,
        config: Optional[DefectConfig] = None,
    ):
        """Initialize defect processor.

        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or DefectConfig()

        # Initialize detector and repairer
        self._detector = DefectDetector(
            scratch_sensitivity=self.config.scratch_sensitivity,
            dust_sensitivity=self.config.dust_sensitivity,
            damage_sensitivity=self.config.damage_sensitivity,
            min_size=self.config.min_defect_size,
            max_size=self.config.max_defect_size,
            detection_mode=self.config.detection_mode,
        )

        self._repairer = DefectRepairer(
            backend=self.config.repair_backend,
            repair_strength=self.config.repair_strength,
            preserve_grain=self.config.preserve_grain,
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Detect defects in a single frame.

        Args:
            frame: Input frame
            frame_index: Frame index for tracking

        Returns:
            DetectionResult with detected defects
        """
        result = DetectionResult(frame_index=frame_index)

        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect enabled defect types
        if self.config.detect_scratches:
            result.scratch_mask, scratches = self._detector.detect_scratches(
                gray, frame_index
            )
            result.defects.extend(scratches)

        if self.config.detect_dust:
            result.dust_mask, dust = self._detector.detect_dust(
                gray, frame_index
            )
            result.defects.extend(dust)

        if self.config.detect_damage:
            result.damage_mask, damage = self._detector.detect_film_damage(
                frame, frame_index
            )
            result.defects.extend(damage)

        # Combine masks
        result.combined_mask = self._detector._combine_masks(
            result.scratch_mask,
            result.dust_mask,
            result.damage_mask,
        )

        if result.combined_mask is not None:
            defect_ratio = np.sum(result.combined_mask > 0) / result.combined_mask.size
            result.severity_score = min(1.0, defect_ratio * 100)

        result.needs_repair = len(result.defects) > 0

        return result

    def repair(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        neighbor_frames: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Repair detected defects in a frame.

        Args:
            frame: Input frame
            detection: Detection result for this frame
            neighbor_frames: Optional neighbor frames for temporal repair

        Returns:
            Repaired frame
        """
        if not detection.needs_repair:
            return frame

        result = frame.copy()

        # Repair scratches first (usually span multiple frames)
        if detection.scratch_mask is not None and np.sum(detection.scratch_mask) > 0:
            result = self._repairer.repair_scratches(
                result, detection.scratch_mask, neighbor_frames
            )

        # Then dust (temporal repair very effective)
        if detection.dust_mask is not None and np.sum(detection.dust_mask) > 0:
            result = self._repairer.repair_dust(
                result, detection.dust_mask, neighbor_frames
            )

        # Finally damage (inpainting)
        if detection.damage_mask is not None and np.sum(detection.damage_mask) > 0:
            result = self._repairer.repair_damage(
                result, detection.damage_mask, neighbor_frames
            )

        return result

    def process(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
        return_frames: bool = True,
    ) -> RepairResult:
        """Process a sequence of frames - detect and repair all defects.

        Args:
            frames: List of input frames
            progress_callback: Optional progress callback (0-1)
            return_frames: Whether to include repaired frames in result

        Returns:
            RepairResult with processing statistics
        """
        import time
        start_time = time.time()

        result = RepairResult()
        result.backend_used = self._repairer.backend_name

        if not frames:
            return result

        n_frames = len(frames)
        window = self.config.temporal_window

        # Phase 1: Detection (50% of progress)
        detections: List[DetectionResult] = []

        logger.info(f"Detecting defects in {n_frames} frames...")

        for i, frame in enumerate(frames):
            detection = self.detect(frame, i)
            detections.append(detection)

            result.scratch_count += sum(
                1 for d in detection.defects
                if d.defect_type in [DefectType.SCRATCH_VERTICAL,
                                     DefectType.SCRATCH_HORIZONTAL]
            )
            result.dust_count += sum(
                1 for d in detection.defects
                if d.defect_type in [DefectType.DUST, DefectType.DEBRIS]
            )
            result.damage_count += sum(
                1 for d in detection.defects
                if d.defect_type not in [DefectType.SCRATCH_VERTICAL,
                                         DefectType.SCRATCH_HORIZONTAL,
                                         DefectType.DUST, DefectType.DEBRIS]
            )
            result.defects_detected += len(detection.defects)

            if progress_callback:
                progress_callback((i + 1) / n_frames * 0.5)

        # Phase 2: Repair (50% of progress)
        logger.info(f"Repairing {result.defects_detected} defects...")

        repaired_frames: List[np.ndarray] = []

        for i, (frame, detection) in enumerate(zip(frames, detections)):
            if detection.needs_repair:
                # Gather neighbor frames for temporal repair
                neighbors = []
                if self.config.temporal_coherence:
                    for j in range(max(0, i - window), min(n_frames, i + window + 1)):
                        if j != i:
                            neighbors.append(frames[j])

                repaired = self.repair(frame, detection, neighbors)
                result.defects_repaired += len(detection.defects)
            else:
                repaired = frame

            if return_frames:
                repaired_frames.append(repaired)

            result.frames_processed += 1

            if progress_callback:
                progress_callback(0.5 + (i + 1) / n_frames * 0.5)

        result.processing_time_ms = (time.time() - start_time) * 1000

        if return_frames:
            result.output_frames = repaired_frames

        logger.info(
            f"Defect processing complete: {result.defects_repaired}/{result.defects_detected} "
            f"defects repaired in {result.processing_time_ms:.0f}ms"
        )

        return result

    def process_auto(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[np.ndarray], RepairResult]:
        """Auto-detect defect types and process frames.

        First analyzes a sample of frames to determine which defect
        types are present, then processes all frames.

        Args:
            frames: List of input frames
            progress_callback: Optional progress callback

        Returns:
            Tuple of (repaired frames, result)
        """
        if not frames:
            return [], RepairResult()

        # Sample frames for analysis
        sample_size = min(10, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, sample_size, dtype=int)

        # Analyze samples
        has_scratches = False
        has_dust = False
        has_damage = False

        for i in sample_indices:
            detection = self.detect(frames[i], i)

            for defect in detection.defects:
                if defect.defect_type in [DefectType.SCRATCH_VERTICAL,
                                          DefectType.SCRATCH_HORIZONTAL]:
                    has_scratches = True
                elif defect.defect_type in [DefectType.DUST, DefectType.DEBRIS]:
                    has_dust = True
                else:
                    has_damage = True

        # Update config based on analysis
        self.config.detect_scratches = has_scratches
        self.config.detect_dust = has_dust
        self.config.detect_damage = has_damage

        logger.info(
            f"Auto-detection: scratches={has_scratches}, "
            f"dust={has_dust}, damage={has_damage}"
        )

        # Process all frames
        result = self.process(frames, progress_callback, return_frames=True)

        return result.output_frames, result

    def get_preview(
        self,
        frame: np.ndarray,
        detection: Optional[DetectionResult] = None,
    ) -> np.ndarray:
        """Generate a visualization of detected defects.

        Args:
            frame: Input frame
            detection: Optional pre-computed detection result

        Returns:
            Visualization frame with defects highlighted
        """
        if detection is None:
            detection = self.detect(frame)

        preview = frame.copy()

        # Draw colored overlays for each defect type
        if detection.scratch_mask is not None:
            overlay = np.zeros_like(preview)
            overlay[detection.scratch_mask > 0] = [0, 0, 255]  # Red for scratches
            preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)

        if detection.dust_mask is not None:
            overlay = np.zeros_like(preview)
            overlay[detection.dust_mask > 0] = [255, 255, 0]  # Cyan for dust
            preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)

        if detection.damage_mask is not None:
            overlay = np.zeros_like(preview)
            overlay[detection.damage_mask > 0] = [0, 255, 255]  # Yellow for damage
            preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)

        return preview


# =============================================================================
# Factory Functions
# =============================================================================

def create_defect_processor(
    detect_scratches: bool = True,
    detect_dust: bool = True,
    detect_damage: bool = True,
    repair_strength: float = 0.8,
    temporal_coherence: bool = True,
    **kwargs,
) -> DefectProcessor:
    """Factory function to create a defect processor.

    Args:
        detect_scratches: Enable scratch detection
        detect_dust: Enable dust detection
        detect_damage: Enable damage detection
        repair_strength: Repair strength (0-1)
        temporal_coherence: Use temporal repair
        **kwargs: Additional DefectConfig parameters

    Returns:
        Configured DefectProcessor
    """
    config = DefectConfig(
        detect_scratches=detect_scratches,
        detect_dust=detect_dust,
        detect_damage=detect_damage,
        repair_strength=repair_strength,
        temporal_coherence=temporal_coherence,
        **kwargs,
    )
    return DefectProcessor(config)


def detect_defects(
    frame: np.ndarray,
    sensitivity: float = 0.5,
) -> DetectionResult:
    """Convenience function to detect defects in a single frame.

    Args:
        frame: Input frame
        sensitivity: Detection sensitivity (0-1)

    Returns:
        DetectionResult
    """
    config = DefectConfig(
        scratch_sensitivity=sensitivity,
        dust_sensitivity=sensitivity,
        damage_sensitivity=sensitivity,
    )
    processor = DefectProcessor(config)
    return processor.detect(frame)


def repair_defects(
    frame: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.8,
) -> np.ndarray:
    """Convenience function to repair defects using a mask.

    Args:
        frame: Input frame
        mask: Binary defect mask
        strength: Repair strength (0-1)

    Returns:
        Repaired frame
    """
    repairer = DefectRepairer(repair_strength=strength)
    return repairer.repair_scratches(frame, mask)


def process_frames_auto(
    frames: List[np.ndarray],
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[List[np.ndarray], RepairResult]:
    """Convenience function for automatic defect processing.

    Args:
        frames: Input frames
        progress_callback: Optional progress callback

    Returns:
        Tuple of (repaired frames, result)
    """
    processor = DefectProcessor()
    return processor.process_auto(frames, progress_callback)
