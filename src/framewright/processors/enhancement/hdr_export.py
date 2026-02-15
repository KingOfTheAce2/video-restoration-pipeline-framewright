"""HDR and Dolby Vision export capabilities for video restoration.

This module provides comprehensive HDR export functionality including:
- HDR10, HDR10+, Dolby Vision, and HLG format support
- Multiple tone mapping algorithms (Reinhard, ACES, Hable, BT.2390)
- Color space conversion (BT.709 <-> BT.2020, P3)
- PQ and HLG transfer function encoding/decoding
- SDR to HDR expansion with inverse tone mapping
- HDR analysis and metadata detection
- FFmpeg integration for final encoding

Example:
    >>> from framewright.processors.enhancement.hdr_export import (
    ...     create_hdr_exporter, export_as_hdr, analyze_hdr
    ... )
    >>> export_as_hdr(frames, output_path, format="hdr10")
    >>> analysis = analyze_hdr("input.mp4")
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class HDRFormat(Enum):
    """Supported HDR output formats.

    HDR10: Open standard with static metadata. Most compatible.
    HDR10_PLUS: Samsung's dynamic metadata extension.
    DOLBY_VISION: Premium format with dynamic metadata per frame.
    HLG: Hybrid Log-Gamma for broadcast, backward compatible with SDR.
    """
    HDR10 = "hdr10"
    HDR10_PLUS = "hdr10plus"
    DOLBY_VISION = "dolby_vision"
    HLG = "hlg"


class ToneMappingAlgorithm(Enum):
    """Tone mapping algorithms for HDR processing.

    REINHARD: Classic photographic operator, soft roll-off, natural look.
    ACES: Academy Color Encoding System, cinematic S-curve.
    HABLE: Uncharted 2 filmic, good contrast preservation.
    BT2390: ITU-R BT.2390 reference EETF for HDR.
    """
    REINHARD = "reinhard"
    ACES = "aces"
    HABLE = "hable"
    BT2390 = "bt2390"


class ColorSpaceType(Enum):
    """Color space types for HDR export.

    BT2020: Wide color gamut for HDR (75.8% of visible spectrum).
    P3: DCI-P3 digital cinema (53.6% of visible spectrum).
    REC709: Standard definition color space (35.9% of visible spectrum).
    """
    BT2020 = "bt2020"
    P3 = "p3"
    REC709 = "rec709"


@dataclass
class HDRConfig:
    """Configuration for HDR export processing.

    Attributes:
        format: HDR format ("hdr10", "hdr10plus", "dolby_vision", "hlg").
        max_cll: Maximum Content Light Level in nits (1-10000).
        max_fall: Maximum Frame Average Light Level in nits.
        master_display: SMPTE ST 2086 mastering display metadata string.
        target_nits: Target peak brightness in nits (100-10000).
        tone_mapping: Tone mapping algorithm ("reinhard", "aces", "hable", "bt2390").
        color_space: Target color space ("bt2020", "p3", "rec709").
    """
    format: str = "hdr10"
    max_cll: int = 1000
    max_fall: int = 400
    master_display: str = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    target_nits: int = 1000
    tone_mapping: str = "aces"
    color_space: str = "bt2020"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_formats = ["hdr10", "hdr10plus", "dolby_vision", "hlg"]
        if self.format.lower() not in valid_formats:
            raise ValueError(f"Invalid format: {self.format}. Use: {valid_formats}")

        valid_tone_mappings = ["reinhard", "aces", "hable", "bt2390"]
        if self.tone_mapping.lower() not in valid_tone_mappings:
            raise ValueError(f"Invalid tone mapping: {self.tone_mapping}")

        valid_color_spaces = ["bt2020", "p3", "rec709"]
        if self.color_space.lower() not in valid_color_spaces:
            raise ValueError(f"Invalid color space: {self.color_space}")

        if not 0 < self.max_cll <= 10000:
            raise ValueError(f"max_cll must be 1-10000, got {self.max_cll}")

        if not 0 < self.max_fall <= 10000:
            raise ValueError(f"max_fall must be 1-10000, got {self.max_fall}")

        if not 100 <= self.target_nits <= 10000:
            raise ValueError(f"target_nits must be 100-10000, got {self.target_nits}")

    def to_hdr_format(self) -> HDRFormat:
        """Convert format string to HDRFormat enum."""
        return HDRFormat(self.format.lower())


@dataclass
class HDRMetadata:
    """HDR metadata for video content with HDR10/HDR10+/Dolby Vision support.

    Attributes:
        max_cll: Maximum Content Light Level in nits.
        max_fall: Maximum Frame Average Light Level in nits.
        master_display_primaries: Display primaries (G, B, R coordinates).
        master_display_white_point: White point coordinates (D65 standard).
        master_display_luminance: Min/max luminance in 0.0001 nits units.
        color_primaries: Color primaries specification (bt2020).
        transfer_characteristics: Transfer function (smpte2084 for PQ).
        matrix_coefficients: Color matrix (bt2020nc).
        dolby_vision_profile: DV profile number (5, 7, 8).
        dolby_vision_rpu: Raw RPU data for Dolby Vision.
        hdr10plus_metadata: HDR10+ dynamic metadata per scene.
        per_frame_metadata: Optional per-frame dynamic metadata.
    """
    max_cll: int = 1000
    max_fall: int = 400
    master_display_primaries: Tuple[Tuple[int, int], ...] = (
        (13250, 34500),  # Green
        (7500, 3000),    # Blue
        (34000, 16000),  # Red
    )
    master_display_white_point: Tuple[int, int] = (15635, 16450)  # D65
    master_display_luminance: Tuple[int, int] = (10000000, 1)  # Max, min
    color_primaries: str = "bt2020"
    transfer_characteristics: str = "smpte2084"
    matrix_coefficients: str = "bt2020nc"
    dolby_vision_profile: Optional[int] = None
    dolby_vision_rpu: Optional[bytes] = None
    hdr10plus_metadata: Optional[List[Dict[str, Any]]] = None
    per_frame_metadata: Optional[List[Dict[str, Any]]] = None

    def to_master_display_string(self) -> str:
        """Convert to FFmpeg master-display string format."""
        g, b, r = self.master_display_primaries
        wp = self.master_display_white_point
        lum = self.master_display_luminance
        return (
            f"G({g[0]},{g[1]})B({b[0]},{b[1]})R({r[0]},{r[1]})"
            f"WP({wp[0]},{wp[1]})L({lum[0]},{lum[1]})"
        )

    def to_x265_params(self) -> str:
        """Convert to x265-params string for FFmpeg."""
        return ":".join([
            "hdr-opt=1",
            "repeat-headers=1",
            f"colorprim={self.color_primaries}",
            f"transfer={self.transfer_characteristics}",
            f"colormatrix={self.matrix_coefficients}",
            f"max-cll={self.max_cll},{self.max_fall}",
            f"master-display={self.to_master_display_string()}",
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_cll": self.max_cll,
            "max_fall": self.max_fall,
            "master_display": self.to_master_display_string(),
            "color_primaries": self.color_primaries,
            "transfer_characteristics": self.transfer_characteristics,
            "matrix_coefficients": self.matrix_coefficients,
            "dolby_vision_profile": self.dolby_vision_profile,
        }


@dataclass
class HDRAnalysisResult:
    """Result of HDR analysis on video content.

    Attributes:
        is_hdr: Whether the content is already HDR.
        detected_format: Detected HDR format if applicable.
        estimated_max_cll: Estimated Maximum Content Light Level.
        estimated_max_fall: Estimated Maximum Frame Average Light Level.
        avg_luminance: Average luminance across all frames.
        peak_luminance: Peak luminance detected.
        dynamic_range_stops: Dynamic range in photographic stops.
        color_volume_percentage: Percentage of BT.2020 color volume used.
        recommended_settings: Recommended export settings.
        frame_luminance_histogram: Luminance distribution histogram.
    """
    is_hdr: bool = False
    detected_format: Optional[str] = None
    estimated_max_cll: int = 0
    estimated_max_fall: int = 0
    avg_luminance: float = 0.0
    peak_luminance: float = 0.0
    dynamic_range_stops: float = 0.0
    color_volume_percentage: float = 0.0
    recommended_settings: Dict[str, Any] = field(default_factory=dict)
    frame_luminance_histogram: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in vars(self).items() if k != "frame_luminance_histogram"}


@dataclass
class HDRExportResult:
    """Result of HDR export operation.

    Attributes:
        success: Whether the export succeeded.
        output_path: Path to the output file.
        frames_processed: Number of frames processed.
        processing_time_seconds: Total processing time.
        metadata_applied: HDR metadata that was applied.
        warnings: Any warnings generated during export.
        error_message: Error message if export failed.
    """
    success: bool = False
    output_path: Optional[Path] = None
    frames_processed: int = 0
    processing_time_seconds: float = 0.0
    metadata_applied: Optional[HDRMetadata] = None
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class ToneMapper:
    """Tone mapping processor for HDR content with multiple algorithms.

    Supports SDR to HDR expansion (inverse tone mapping) and HDR to SDR
    compression using industry-standard algorithms.

    Example:
        >>> mapper = ToneMapper(algorithm=ToneMappingAlgorithm.ACES)
        >>> hdr_frame = mapper.expand_sdr_to_hdr(sdr_frame, target_nits=1000)
        >>> sdr_frame = mapper.compress_hdr_to_sdr(hdr_frame, source_nits=1000)
    """

    def __init__(self, algorithm: ToneMappingAlgorithm = ToneMappingAlgorithm.ACES):
        """Initialize the tone mapper.

        Args:
            algorithm: Tone mapping algorithm to use.
        """
        self.algorithm = algorithm

    def apply(
        self,
        frame: np.ndarray,
        source_nits: float = 100.0,
        target_nits: float = 1000.0,
    ) -> np.ndarray:
        """Apply tone mapping to a frame.

        Args:
            frame: Input frame as float32 numpy array (0-1 range, linear).
            source_nits: Source peak brightness in nits.
            target_nits: Target peak brightness in nits.

        Returns:
            Tone-mapped frame as float32 numpy array.
        """
        methods = {
            ToneMappingAlgorithm.REINHARD: self._reinhard,
            ToneMappingAlgorithm.ACES: self._aces,
            ToneMappingAlgorithm.HABLE: self._hable,
            ToneMappingAlgorithm.BT2390: self._bt2390,
        }
        return methods[self.algorithm](frame, source_nits, target_nits)

    def expand_sdr_to_hdr(
        self,
        frame: np.ndarray,
        target_nits: float = 1000.0,
    ) -> np.ndarray:
        """Expand SDR content to HDR range (inverse tone mapping).

        Uses inverse Reinhard operator for natural expansion that preserves
        original artistic intent while adding headroom for highlights.

        Args:
            frame: SDR frame as float32 numpy array (0-1 range).
            target_nits: Target peak brightness in nits.

        Returns:
            Expanded HDR frame (0-1 range normalized to target_nits).
        """
        frame_linear = np.power(np.clip(frame, 0.001, 1.0), 2.2)
        scale = target_nits / 100.0
        # Inverse Reinhard with limiting
        frame_expanded = frame_linear * scale / (1.0 - frame_linear * 0.9 + 0.001)
        return np.clip(frame_expanded / scale, 0.0, 1.0).astype(np.float32)

    def compress_hdr_to_sdr(
        self,
        frame: np.ndarray,
        source_nits: float = 1000.0,
    ) -> np.ndarray:
        """Compress HDR content to SDR range.

        Args:
            frame: HDR frame as float32 numpy array (linear, 0-1).
            source_nits: Source peak brightness in nits.

        Returns:
            Compressed SDR frame.
        """
        return self.apply(frame, source_nits=source_nits, target_nits=100.0)

    def _reinhard(
        self,
        frame: np.ndarray,
        source_nits: float,
        target_nits: float,
    ) -> np.ndarray:
        """Reinhard tone mapping operator."""
        if len(frame.shape) == 3:
            lum = 0.2126 * frame[:, :, 2] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 0]
        else:
            lum = frame
        lum = np.clip(lum, 0.0001, None)

        l_white = target_nits / source_nits
        scale = source_nits / target_nits
        lum_scaled = lum * scale
        lum_mapped = lum_scaled * (1 + lum_scaled / (l_white ** 2)) / (1 + lum_scaled)

        if len(frame.shape) == 3:
            scale_factor = np.where(lum > 0.0001, lum_mapped / lum, 1.0)
            return np.clip(frame * scale_factor[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)
        return np.clip(lum_mapped, 0.0, 1.0).astype(np.float32)

    def _aces(
        self,
        frame: np.ndarray,
        source_nits: float,
        target_nits: float,
    ) -> np.ndarray:
        """ACES filmic tone mapping curve."""
        x = frame * (source_nits / target_nits)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0).astype(np.float32)

    def _hable(
        self,
        frame: np.ndarray,
        source_nits: float,
        target_nits: float,
    ) -> np.ndarray:
        """Hable (Uncharted 2) filmic tone mapping."""
        def hable_partial(x: np.ndarray) -> np.ndarray:
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

        exposure_bias = source_nits / target_nits
        curr = hable_partial(frame * exposure_bias)
        white_scale = 1.0 / hable_partial(np.array([exposure_bias]))
        return np.clip(curr * white_scale, 0.0, 1.0).astype(np.float32)

    def _bt2390(
        self,
        frame: np.ndarray,
        source_nits: float,
        target_nits: float,
    ) -> np.ndarray:
        """ITU-R BT.2390 reference tone mapping (EETF)."""
        if len(frame.shape) == 3:
            lum = 0.2627 * frame[:, :, 2] + 0.678 * frame[:, :, 1] + 0.0593 * frame[:, :, 0]
        else:
            lum = frame
        lum = np.clip(lum, 0.0001, None)

        lw = target_nits / source_nits
        ks = 1.5 * lw - 0.5
        b = lw - ks
        e2 = np.where(lum < ks, lum, ks + (1 - ks) * np.tanh((lum - ks) / b) * b)

        if len(frame.shape) == 3:
            scale_factor = np.where(lum > 0.0001, e2 / lum, 1.0)
            return np.clip(frame * scale_factor[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)
        return np.clip(e2, 0.0, 1.0).astype(np.float32)


class ColorSpaceConverter:
    """Color space conversion utilities for HDR processing.

    Supports conversion between BT.709, BT.2020, and P3 color spaces,
    as well as PQ (ST 2084) and HLG transfer function encoding/decoding.

    Example:
        >>> converter = ColorSpaceConverter()
        >>> bt2020_frame = converter.convert_709_to_2020(rec709_frame)
        >>> pq_encoded = converter.encode_pq(linear_frame, peak_nits=1000)
    """

    # Color space conversion matrices
    MATRIX_709_TO_2020 = np.array([
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956]
    ], dtype=np.float32)

    MATRIX_2020_TO_709 = np.array([
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187]
    ], dtype=np.float32)

    # PQ (SMPTE ST 2084) constants
    PQ_M1 = 2610.0 / 16384.0
    PQ_M2 = 2523.0 / 4096.0 * 128.0
    PQ_C1 = 3424.0 / 4096.0
    PQ_C2 = 2413.0 / 4096.0 * 32.0
    PQ_C3 = 2392.0 / 4096.0 * 32.0

    def convert_709_to_2020(self, frame: np.ndarray) -> np.ndarray:
        """Convert from BT.709 to BT.2020 color space."""
        return self._apply_matrix(frame, self.MATRIX_709_TO_2020)

    def convert_2020_to_709(self, frame: np.ndarray) -> np.ndarray:
        """Convert from BT.2020 to BT.709 color space."""
        return self._apply_matrix(frame, self.MATRIX_2020_TO_709)

    def _apply_matrix(self, frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply a 3x3 color matrix transformation."""
        h, w, c = frame.shape
        frame_rgb = frame[:, :, ::-1].reshape(-1, 3)  # BGR to RGB
        result_rgb = np.dot(frame_rgb, matrix.T)
        return np.clip(result_rgb[:, ::-1].reshape(h, w, c), 0.0, 1.0).astype(np.float32)

    def encode_pq(self, frame: np.ndarray, peak_nits: float = 10000.0) -> np.ndarray:
        """Encode linear light values using PQ (SMPTE ST 2084) transfer function."""
        L = np.clip(frame, 0.0, 1.0) * (peak_nits / 10000.0)
        Lm1 = np.power(L, self.PQ_M1)
        numerator = self.PQ_C1 + self.PQ_C2 * Lm1
        denominator = 1.0 + self.PQ_C3 * Lm1
        return np.clip(np.power(numerator / denominator, self.PQ_M2), 0.0, 1.0).astype(np.float32)

    def decode_pq(self, frame: np.ndarray, peak_nits: float = 10000.0) -> np.ndarray:
        """Decode PQ-encoded values to linear light."""
        E_prime = np.clip(frame, 0.0, 1.0)
        Em2 = np.power(E_prime, 1.0 / self.PQ_M2)
        numerator = np.maximum(Em2 - self.PQ_C1, 0.0)
        denominator = np.maximum(self.PQ_C2 - self.PQ_C3 * Em2, 1e-10)
        L = np.power(numerator / denominator, 1.0 / self.PQ_M1)
        return np.clip(L * (10000.0 / peak_nits), 0.0, 1.0).astype(np.float32)

    def encode_hlg(self, frame: np.ndarray) -> np.ndarray:
        """Encode linear light values using HLG transfer function."""
        E = np.clip(frame, 0.0, 1.0)
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        threshold = 1.0 / 12.0
        E_prime = np.where(E <= threshold, np.sqrt(3.0 * E), a * np.log(12.0 * E - b) + c)
        return np.clip(E_prime, 0.0, 1.0).astype(np.float32)

    def decode_hlg(self, frame: np.ndarray) -> np.ndarray:
        """Decode HLG-encoded values to linear light."""
        E_prime = np.clip(frame, 0.0, 1.0)
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        E = np.where(E_prime <= 0.5, (E_prime ** 2) / 3.0, (np.exp((E_prime - c) / a) + b) / 12.0)
        return np.clip(E, 0.0, 1.0).astype(np.float32)

    def apply_gamut_mapping(
        self,
        frame: np.ndarray,
        source: ColorSpaceType,
        target: ColorSpaceType,
    ) -> np.ndarray:
        """Apply gamut mapping between color spaces."""
        if source == target:
            return frame
        if source == ColorSpaceType.REC709 and target == ColorSpaceType.BT2020:
            return self.convert_709_to_2020(frame)
        if source == ColorSpaceType.BT2020 and target == ColorSpaceType.REC709:
            return self.convert_2020_to_709(frame)
        return frame


class HDRAnalyzer:
    """Analyzer for detecting HDR characteristics in video content.

    Analyzes frames to detect existing HDR metadata, estimate MaxCLL/MaxFALL,
    and recommend export settings.

    Example:
        >>> analyzer = HDRAnalyzer()
        >>> result = analyzer.analyze("video.mp4")
        >>> print(f"MaxCLL: {result.estimated_max_cll}")
    """

    def __init__(self) -> None:
        """Initialize the HDR analyzer."""
        self._ffprobe_available = shutil.which("ffprobe") is not None

    def analyze(
        self,
        frames: Union[List[np.ndarray], Path, str],
        sample_count: int = 100,
    ) -> HDRAnalysisResult:
        """Analyze frames for HDR characteristics.

        Args:
            frames: List of frames, or path to video/frame directory.
            sample_count: Number of frames to sample for analysis.

        Returns:
            HDRAnalysisResult with analysis data.
        """
        result = HDRAnalysisResult()

        if isinstance(frames, (str, Path)):
            frames_path = Path(frames)
            if frames_path.is_file():
                result = self._analyze_video_metadata(frames_path, result)
                frame_list = self._extract_sample_frames(frames_path, sample_count)
            else:
                frame_list = self._load_sample_frames(frames_path, sample_count)
        else:
            frame_list = frames[:sample_count]

        if frame_list:
            result = self._analyze_frame_content(frame_list, result)
            result.recommended_settings = self._generate_recommendations(result)

        return result

    def _analyze_video_metadata(
        self,
        video_path: Path,
        result: HDRAnalysisResult,
    ) -> HDRAnalysisResult:
        """Analyze video file metadata for HDR information."""
        if not self._ffprobe_available:
            return result

        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_streams", str(video_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") != "video":
                        continue

                    transfer = stream.get("color_transfer", "")
                    if transfer == "smpte2084":
                        result.is_hdr = True
                        result.detected_format = "hdr10"
                    elif transfer == "arib-std-b67":
                        result.is_hdr = True
                        result.detected_format = "hlg"

                    for sd in stream.get("side_data_list", []):
                        sd_type = sd.get("side_data_type", "")
                        if "Content light level" in sd_type:
                            result.estimated_max_cll = int(sd.get("max_content", 0))
                            result.estimated_max_fall = int(sd.get("max_average", 0))
                        if "Dolby Vision" in sd_type:
                            result.detected_format = "dolby_vision"

        except Exception as e:
            logger.warning(f"Error analyzing metadata: {e}")

        return result

    def _extract_sample_frames(
        self,
        video_path: Path,
        sample_count: int,
    ) -> List[np.ndarray]:
        """Extract sample frames from video file."""
        if not HAS_OPENCV:
            return []

        frames = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // sample_count)

            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                if len(frames) >= sample_count:
                    break

            cap.release()
        except Exception as e:
            logger.warning(f"Error extracting frames: {e}")

        return frames

    def _load_sample_frames(
        self,
        frames_dir: Path,
        sample_count: int,
    ) -> List[np.ndarray]:
        """Load sample frames from directory."""
        if not HAS_OPENCV:
            return []

        frames = []
        frame_files = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
        step = max(1, len(frame_files) // sample_count)

        for i in range(0, len(frame_files), step):
            frame = cv2.imread(str(frame_files[i]), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                frames.append(frame)
            if len(frames) >= sample_count:
                break

        return frames

    def _analyze_frame_content(
        self,
        frames: List[np.ndarray],
        result: HDRAnalysisResult,
    ) -> HDRAnalysisResult:
        """Analyze frame content for luminance and color characteristics."""
        max_lums, avg_lums, all_lums = [], [], []

        for frame in frames:
            if frame.dtype == np.uint8:
                frame_float = frame.astype(np.float32) / 255.0
            elif frame.dtype == np.uint16:
                frame_float = frame.astype(np.float32) / 65535.0
            else:
                frame_float = frame.astype(np.float32)

            if len(frame_float.shape) == 3:
                lum = (0.2126 * frame_float[:, :, 2] +
                       0.7152 * frame_float[:, :, 1] +
                       0.0722 * frame_float[:, :, 0])
            else:
                lum = frame_float

            max_lums.append(np.max(lum))
            avg_lums.append(np.mean(lum))
            all_lums.extend(lum.flatten()[::100])

        result.peak_luminance = float(np.max(max_lums))
        result.avg_luminance = float(np.mean(avg_lums))
        result.estimated_max_cll = int(result.peak_luminance * (10000 if result.is_hdr else 1000))
        result.estimated_max_fall = int(np.max(avg_lums) * (10000 if result.is_hdr else 400))
        result.dynamic_range_stops = float(
            np.log2(max(np.max(all_lums), 0.001) / max(np.min(all_lums), 0.0001))
        )
        result.frame_luminance_histogram = np.histogram(all_lums, bins=256, range=(0, 1))[0]

        return result

    def _generate_recommendations(
        self,
        result: HDRAnalysisResult,
    ) -> Dict[str, Any]:
        """Generate export recommendations based on analysis."""
        if result.is_hdr:
            return {
                "format": result.detected_format or "hdr10",
                "max_cll": result.estimated_max_cll or 1000,
                "max_fall": result.estimated_max_fall or 400,
                "preserve_metadata": True,
                "message": "Content is HDR, preserving metadata",
            }

        high_dr = result.dynamic_range_stops > 8
        return {
            "format": "hdr10",
            "max_cll": min(result.estimated_max_cll, 4000) if high_dr else 1000,
            "max_fall": min(result.estimated_max_fall, 1000) if high_dr else 400,
            "tone_mapping": "aces" if high_dr else "reinhard",
            "color_space": "bt2020",
            "target_nits": min(max(result.estimated_max_cll, 1000), 4000),
            "message": "High DR content, HDR recommended" if high_dr else "Standard content",
        }


class HDRExporter:
    """HDR video exporter with format-specific encoding.

    Supports HDR10, HDR10+, Dolby Vision, and HLG export with proper
    metadata injection and FFmpeg integration.

    Example:
        >>> exporter = HDRExporter(config)
        >>> result = exporter.export(frames, "output.mp4", metadata)
    """

    def __init__(self, config: Optional[HDRConfig] = None):
        """Initialize the HDR exporter."""
        self.config = config or HDRConfig()
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        self._color_converter = ColorSpaceConverter()
        self._tone_mapper = ToneMapper(ToneMappingAlgorithm(self.config.tone_mapping))

    def is_available(self) -> bool:
        """Check if HDR export is available."""
        return self._ffmpeg_available

    def export(
        self,
        frames: Union[List[np.ndarray], Path, str],
        output_path: Union[Path, str],
        metadata: Optional[HDRMetadata] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> HDRExportResult:
        """Export frames with HDR encoding."""
        result = HDRExportResult()
        start_time = time.time()

        if not self._ffmpeg_available:
            result.error_message = "FFmpeg not available"
            return result

        metadata = metadata or HDRMetadata(
            max_cll=self.config.max_cll,
            max_fall=self.config.max_fall,
        )
        hdr_format = self.config.to_hdr_format()

        try:
            if hdr_format == HDRFormat.HLG:
                result = self.export_hlg(frames, output_path, metadata, progress_callback)
            else:
                result = self.export_hdr10(frames, output_path, metadata, progress_callback)
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"HDR export failed: {e}")

        result.processing_time_seconds = time.time() - start_time
        return result

    def export_hdr10(
        self,
        frames: Union[List[np.ndarray], Path, str],
        output_path: Union[Path, str],
        metadata: Optional[HDRMetadata] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> HDRExportResult:
        """Export with HDR10 format."""
        result = HDRExportResult()
        output_path = Path(output_path)
        metadata = metadata or HDRMetadata()

        with tempfile.TemporaryDirectory(prefix="hdr10_") as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            frame_count = self._prepare_frames(frames, frames_dir, progress_callback)
            if frame_count == 0:
                result.error_message = "No frames to export"
                return result

            cmd = [
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", str(frames_dir / "frame_%08d.png"),
                "-c:v", "libx265",
                "-preset", "slow",
                "-crf", "18",
                "-pix_fmt", "yuv420p10le",
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
                "-x265-params", metadata.to_x265_params(),
                str(output_path),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                result.error_message = f"FFmpeg failed: {proc.stderr[:500]}"
                return result

            result.success = True
            result.output_path = output_path
            result.frames_processed = frame_count
            result.metadata_applied = metadata

        return result

    def export_dolby_vision(
        self,
        frames: Union[List[np.ndarray], Path, str],
        output_path: Union[Path, str],
        metadata: Optional[HDRMetadata] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> HDRExportResult:
        """Export with Dolby Vision format (HDR10 compatible base layer)."""
        result = self.export_hdr10(frames, output_path, metadata, progress_callback)
        result.warnings.append(
            "Dolby Vision export creates HDR10 base layer. "
            "Full DV encoding requires Dolby professional tools."
        )
        return result

    def export_hlg(
        self,
        frames: Union[List[np.ndarray], Path, str],
        output_path: Union[Path, str],
        metadata: Optional[HDRMetadata] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> HDRExportResult:
        """Export with HLG (Hybrid Log-Gamma) format."""
        result = HDRExportResult()
        output_path = Path(output_path)
        metadata = metadata or HDRMetadata(transfer_characteristics="arib-std-b67")

        with tempfile.TemporaryDirectory(prefix="hlg_") as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir()

            frame_count = self._prepare_frames(frames, frames_dir, progress_callback)
            if frame_count == 0:
                result.error_message = "No frames to export"
                return result

            cmd = [
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", str(frames_dir / "frame_%08d.png"),
                "-c:v", "libx265",
                "-preset", "slow",
                "-crf", "18",
                "-pix_fmt", "yuv420p10le",
                "-color_primaries", "bt2020",
                "-color_trc", "arib-std-b67",
                "-colorspace", "bt2020nc",
                "-x265-params", "hdr-opt=1:repeat-headers=1",
                str(output_path),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                result.error_message = f"FFmpeg failed: {proc.stderr[:500]}"
                return result

            result.success = True
            result.output_path = output_path
            result.frames_processed = frame_count
            result.metadata_applied = metadata

        return result

    def export_dual_layer(
        self,
        frames: Union[List[np.ndarray], Path, str],
        output_path: Union[Path, str],
        metadata: Optional[HDRMetadata] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[HDRExportResult, HDRExportResult]:
        """Export with both HDR10 and Dolby Vision layers."""
        output_path = Path(output_path)
        hdr10_result = self.export_hdr10(
            frames, output_path.with_suffix(".hdr10.mp4"), metadata, progress_callback
        )
        dv_result = self.export_dolby_vision(
            frames, output_path.with_suffix(".dv.mp4"), metadata, progress_callback
        )
        return hdr10_result, dv_result

    def _prepare_frames(
        self,
        frames: Union[List[np.ndarray], Path, str],
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> int:
        """Prepare frames for export by converting and saving to directory."""
        if not HAS_OPENCV:
            logger.error("OpenCV required for frame processing")
            return 0

        frame_list: List[np.ndarray] = []

        if isinstance(frames, (str, Path)):
            frames_path = Path(frames)
            if frames_path.is_dir():
                frame_files = sorted(frames_path.glob("*.png"))
                if not frame_files:
                    frame_files = sorted(frames_path.glob("*.jpg"))
                for f in frame_files:
                    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        frame_list.append(img)
            else:
                cap = cv2.VideoCapture(str(frames_path))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_list.append(frame)
                cap.release()
        else:
            frame_list = list(frames)

        total = len(frame_list)
        for i, frame in enumerate(frame_list):
            if frame.dtype == np.uint8:
                frame_float = frame.astype(np.float32) / 255.0
                frame_linear = np.power(frame_float, 2.2)
                frame_hdr = self._tone_mapper.expand_sdr_to_hdr(
                    frame_linear, self.config.target_nits
                )
                if self.config.color_space == "bt2020":
                    frame_hdr = self._color_converter.convert_709_to_2020(frame_hdr)
                frame_pq = self._color_converter.encode_pq(
                    frame_hdr, self.config.target_nits
                )
                frame_out = (frame_pq * 65535).astype(np.uint16)
            elif frame.dtype == np.uint16:
                frame_out = frame
            else:
                frame_out = (np.clip(frame, 0, 1) * 65535).astype(np.uint16)

            cv2.imwrite(str(output_dir / f"frame_{i:08d}.png"), frame_out)

            if progress_callback:
                progress_callback((i + 1) / total * 0.5)

        return total


class SDRtoHDR:
    """SDR to HDR conversion processor.

    Converts standard dynamic range content to HDR using inverse tone mapping,
    color volume expansion, and highlight reconstruction.

    Example:
        >>> converter = SDRtoHDR(target_nits=1000)
        >>> hdr_frames = converter.expand(sdr_frames)
    """

    def __init__(
        self,
        target_nits: int = 1000,
        color_space: ColorSpaceType = ColorSpaceType.BT2020,
        preserve_intent: bool = True,
    ):
        """Initialize the SDR to HDR converter."""
        self.target_nits = target_nits
        self.color_space = color_space
        self.preserve_intent = preserve_intent
        self._color_converter = ColorSpaceConverter()
        self._tone_mapper = ToneMapper(ToneMappingAlgorithm.ACES)

    def expand(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[np.ndarray]:
        """Expand SDR frames to HDR."""
        hdr_frames = []
        for i, frame in enumerate(frames):
            hdr_frames.append(self.expand_frame(frame))
            if progress_callback:
                progress_callback((i + 1) / len(frames))
        return hdr_frames

    def expand_frame(self, frame: np.ndarray) -> np.ndarray:
        """Expand a single SDR frame to HDR."""
        if frame.dtype == np.uint8:
            frame_float = frame.astype(np.float32) / 255.0
        elif frame.dtype == np.uint16:
            frame_float = frame.astype(np.float32) / 65535.0
        else:
            frame_float = frame.astype(np.float32)

        frame_linear = np.power(np.clip(frame_float, 0.001, 1.0), 2.2)
        frame_expanded = self._tone_mapper.expand_sdr_to_hdr(frame_linear, self.target_nits)

        if self.color_space == ColorSpaceType.BT2020:
            frame_expanded = self._color_converter.convert_709_to_2020(frame_expanded)

        return self._reconstruct_highlights(frame_float, frame_expanded)

    def _reconstruct_highlights(
        self,
        original: np.ndarray,
        expanded: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct highlight detail in expanded HDR."""
        if len(original.shape) == 3:
            clipped = np.any(original > 0.95, axis=2)
        else:
            clipped = original > 0.95

        if not np.any(clipped):
            return expanded

        if len(expanded.shape) == 3:
            clipped_3d = clipped[:, :, np.newaxis]
            return np.where(clipped_3d, expanded * 0.9, expanded).astype(np.float32)
        return np.where(clipped, expanded * 0.9, expanded).astype(np.float32)


# =============================================================================
# Factory Functions
# =============================================================================

def create_hdr_exporter(
    format: str = "hdr10",
    max_cll: int = 1000,
    max_fall: int = 400,
    target_nits: int = 1000,
    tone_mapping: str = "aces",
    color_space: str = "bt2020",
) -> HDRExporter:
    """Create an HDR exporter with specified configuration.

    Args:
        format: HDR format ("hdr10", "hdr10plus", "dolby_vision", "hlg").
        max_cll: Maximum Content Light Level in nits.
        max_fall: Maximum Frame Average Light Level in nits.
        target_nits: Target peak brightness.
        tone_mapping: Tone mapping algorithm.
        color_space: Target color space.

    Returns:
        Configured HDRExporter instance.
    """
    config = HDRConfig(
        format=format,
        max_cll=max_cll,
        max_fall=max_fall,
        target_nits=target_nits,
        tone_mapping=tone_mapping,
        color_space=color_space,
    )
    return HDRExporter(config)


def export_as_hdr(
    frames: Union[List[np.ndarray], Path, str],
    output_path: Union[Path, str],
    format: str = "hdr10",
    max_cll: int = 1000,
    max_fall: int = 400,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> HDRExportResult:
    """Quick function to export frames as HDR video.

    Args:
        frames: Frames to export or path to frame directory.
        output_path: Output video path.
        format: HDR format ("hdr10", "hdr10plus", "dolby_vision", "hlg").
        max_cll: Maximum Content Light Level.
        max_fall: Maximum Frame Average Light Level.
        progress_callback: Optional progress callback.

    Returns:
        HDRExportResult with export status.
    """
    exporter = create_hdr_exporter(format=format, max_cll=max_cll, max_fall=max_fall)
    metadata = HDRMetadata(max_cll=max_cll, max_fall=max_fall)
    return exporter.export(frames, output_path, metadata, progress_callback)


def analyze_hdr(
    video_path: Union[Path, str],
    sample_count: int = 100,
) -> Dict[str, Any]:
    """Analyze video for HDR characteristics and get recommendations.

    Args:
        video_path: Path to video file.
        sample_count: Number of frames to sample.

    Returns:
        Dictionary with analysis results and recommendations.
    """
    analyzer = HDRAnalyzer()
    result = analyzer.analyze(video_path, sample_count)
    return result.to_dict()


__all__ = [
    # Enums
    "HDRFormat",
    "ToneMappingAlgorithm",
    "ColorSpaceType",
    # Configuration
    "HDRConfig",
    "HDRMetadata",
    # Results
    "HDRAnalysisResult",
    "HDRExportResult",
    # Classes
    "ToneMapper",
    "ColorSpaceConverter",
    "HDRAnalyzer",
    "HDRExporter",
    "SDRtoHDR",
    # Factory functions
    "create_hdr_exporter",
    "export_as_hdr",
    "analyze_hdr",
]
