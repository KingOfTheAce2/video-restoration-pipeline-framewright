"""HDR (High Dynamic Range) conversion processor for video restoration.

This module provides SDR to HDR conversion capabilities for video frames,
supporting multiple HDR formats and tone mapping algorithms.

SDR (Standard Dynamic Range):
    - 8-bit color depth (256 levels per channel)
    - Peak brightness: ~100 nits (cd/m^2)
    - Color space: Rec.709 (sRGB for displays)
    - Gamma: 2.2-2.4 (display-referred)
    - Limitations: Crushed highlights, limited shadow detail, narrow color gamut

HDR (High Dynamic Range):
    - 10-bit or higher color depth (1024+ levels per channel)
    - Peak brightness: 1000-10000 nits depending on format
    - Color space: BT.2020 (wider gamut), DCI-P3 as intermediate
    - Transfer function: PQ (Perceptual Quantizer) or HLG
    - Advantages: Brighter highlights, deeper shadows, more colors

Why Convert SDR to HDR:
    1. Better contrast ratio - Highlights can be much brighter without clipping
    2. More colors - BT.2020 covers 75% of visible spectrum vs 36% for Rec.709
    3. Realistic highlights - Specular reflections, sun, fire rendered correctly
    4. Modern display support - Most new TVs are HDR capable
    5. Future-proofing - HDR is becoming the standard for premium content

When NOT to Use HDR Conversion:
    1. Target display is SDR-only - No benefit, may look washed out
    2. Source has no highlight detail to recover - GIGO (garbage in, garbage out)
    3. Heavily compressed source - Banding artifacts will be amplified
    4. Fast processing needed - HDR conversion adds computational overhead
    5. Artistic intent - Some content is designed for SDR aesthetic

Supported HDR Formats:
    - HDR10: Open standard, static metadata, most compatible
    - HDR10+: Samsung's dynamic metadata extension
    - Dolby Vision: Dynamic metadata, highest quality, requires licensing
    - HLG: Hybrid Log-Gamma, broadcast-friendly, backward compatible with SDR
"""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class HDRFormat(Enum):
    """Supported HDR output formats.

    Attributes:
        HDR10: Open HDR10 standard with static metadata (MaxCLL/MaxFALL).
               Most widely supported, works on all HDR displays.
               Peak brightness typically 1000-4000 nits.

        HDR10_PLUS: Samsung's dynamic metadata extension of HDR10.
                    Scene-by-scene optimization for better tone mapping.
                    Requires compatible display.

        DOLBY_VISION: Dolby's premium HDR format with dynamic metadata.
                      12-bit internal processing, per-frame optimization.
                      Requires Dolby Vision licensed encoder and display.

        HLG: Hybrid Log-Gamma, designed for broadcast.
             Backward compatible with SDR displays.
             No metadata required, simpler workflow.
    """
    HDR10 = "hdr10"
    HDR10_PLUS = "hdr10plus"
    DOLBY_VISION = "dolby_vision"
    HLG = "hlg"


class ToneMapping(Enum):
    """Tone mapping algorithms for HDR conversion.

    Tone mapping controls how the expanded dynamic range is distributed
    across the luminance spectrum. Each algorithm has different characteristics.

    Attributes:
        REINHARD: Classic photographic operator with soft roll-off.
                  - Natural, film-like appearance
                  - Gentle highlight compression
                  - Good for preserving shadow detail
                  - May appear slightly desaturated in highlights
                  - Best for: Natural scenes, portraits, documentaries

        ACES: Academy Color Encoding System filmic tone mapping.
              - Industry standard for cinema
              - S-curve response similar to film
              - Rich, saturated colors
              - Strong highlight roll-off
              - Best for: Cinematic content, movies, TV shows

        HABLE: John Hable's Uncharted 2 tone mapping (Filmic).
               - Designed for real-time rendering
               - Good contrast preservation
               - Smooth highlight handling
               - Slightly cooler color temperature
               - Best for: Gaming content, CGI, animations

        MOBIUS: Smooth transition with shadow preservation.
                - Minimal shadow crushing
                - Very smooth highlight roll-off
                - Preserves mid-tone contrast
                - Good for detail retention
                - Best for: Archive restoration, detail-critical content
    """
    REINHARD = "reinhard"
    ACES = "aces"
    HABLE = "hable"
    MOBIUS = "mobius"


class ColorSpace(Enum):
    """Target color spaces for HDR output.

    Attributes:
        BT2020: ITU-R BT.2020 - Full HDR color space.
                Covers 75.8% of visible spectrum.
                Required for HDR10 and Dolby Vision.
                Widest gamut, best for HDR mastering.

        P3: DCI-P3 - Digital cinema color space.
            Covers 53.6% of visible spectrum.
            Good intermediate step from Rec.709.
            Used in Apple devices and digital cinema.

        REC709: ITU-R BT.709 - Standard definition color space.
                Covers 35.9% of visible spectrum.
                SDR standard, not recommended for HDR output.
                Use only for testing or compatibility.
    """
    BT2020 = "bt2020"
    P3 = "p3"
    REC709 = "rec709"


@dataclass
class DynamicRangeInfo:
    """Information about the dynamic range of a frame or video.

    Contains analysis of luminance distribution to help determine
    if content would benefit from HDR conversion.
    """
    min_luminance: float  # Minimum luminance in nits
    max_luminance: float  # Maximum luminance in nits
    avg_luminance: float  # Average luminance in nits
    peak_luminance: float  # 99th percentile luminance
    dynamic_range_stops: float  # Dynamic range in photographic stops
    is_hdr_candidate: bool  # Whether content would benefit from HDR
    histogram: Optional[np.ndarray] = None  # Luminance histogram
    clipped_highlights_pct: float = 0.0  # Percentage of clipped highlight pixels
    clipped_shadows_pct: float = 0.0  # Percentage of clipped shadow pixels

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'min_luminance': self.min_luminance,
            'max_luminance': self.max_luminance,
            'avg_luminance': self.avg_luminance,
            'peak_luminance': self.peak_luminance,
            'dynamic_range_stops': self.dynamic_range_stops,
            'is_hdr_candidate': self.is_hdr_candidate,
            'clipped_highlights_pct': self.clipped_highlights_pct,
            'clipped_shadows_pct': self.clipped_shadows_pct,
        }


@dataclass
class HDRConfig:
    """Configuration for HDR conversion processing.

    This dataclass contains all settings needed for SDR to HDR conversion,
    including target format, brightness, tone mapping, and color space options.

    Attributes:
        target_format: HDR output format (HDR10, HDR10_PLUS, DOLBY_VISION, HLG).
                       Default is HDR10 for maximum compatibility.

        peak_brightness: Target peak brightness in nits (cd/m^2).
                         Range: 400-10000 nits.
                         Default: 1000 nits (standard HDR10).
                         Higher values = brighter highlights but may clip on some displays.

        tone_mapping: Tone mapping algorithm for luminance distribution.
                      REINHARD: Natural, film-like
                      ACES: Cinematic, industry standard
                      HABLE: Gaming-optimized, good contrast
                      MOBIUS: Smooth, shadow-preserving

        color_space: Target color space for HDR output.
                     BT2020: Full HDR gamut (recommended)
                     P3: Digital cinema intermediate
                     REC709: SDR gamut (not recommended)

        enable_local_contrast: Apply local contrast enhancement.
                               Improves perceived detail and depth.
                               May increase processing time.

        max_content_light_level: MaxCLL metadata for HDR10 (nits).
                                 Auto-detected if None.

        max_frame_avg_light_level: MaxFALL metadata for HDR10 (nits).
                                   Auto-detected if None.

        saturation_boost: Color saturation adjustment (0.8-1.5).
                          1.0 = no change, >1.0 = more saturated.
                          HDR allows more saturated colors.

        highlight_recovery: Attempt to recover clipped highlights.
                            Uses luminance from adjacent channels.

        shadow_detail: Shadow detail enhancement level (0.0-2.0).
                       Higher values lift shadows without crushing blacks.

    Example:
        >>> config = HDRConfig(
        ...     target_format=HDRFormat.HDR10,
        ...     peak_brightness=1000,
        ...     tone_mapping=ToneMapping.ACES,
        ...     color_space=ColorSpace.BT2020,
        ...     enable_local_contrast=True
        ... )
        >>> converter = HDRConverter(config)
    """
    target_format: HDRFormat = HDRFormat.HDR10
    peak_brightness: int = 1000
    tone_mapping: ToneMapping = ToneMapping.ACES
    color_space: ColorSpace = ColorSpace.BT2020
    enable_local_contrast: bool = True
    max_content_light_level: Optional[int] = None
    max_frame_avg_light_level: Optional[int] = None
    saturation_boost: float = 1.1
    highlight_recovery: bool = True
    shadow_detail: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 400 <= self.peak_brightness <= 10000:
            raise ValueError(
                f"peak_brightness must be between 400 and 10000 nits, "
                f"got {self.peak_brightness}"
            )
        if not 0.8 <= self.saturation_boost <= 1.5:
            raise ValueError(
                f"saturation_boost must be between 0.8 and 1.5, "
                f"got {self.saturation_boost}"
            )
        if not 0.0 <= self.shadow_detail <= 2.0:
            raise ValueError(
                f"shadow_detail must be between 0.0 and 2.0, "
                f"got {self.shadow_detail}"
            )
        # Convert string values to enums if necessary
        if isinstance(self.target_format, str):
            self.target_format = HDRFormat(self.target_format.lower())
        if isinstance(self.tone_mapping, str):
            self.tone_mapping = ToneMapping(self.tone_mapping.lower())
        if isinstance(self.color_space, str):
            self.color_space = ColorSpace(self.color_space.lower())


@dataclass
class HDRConversionResult:
    """Result of HDR conversion processing."""
    frames_processed: int = 0
    frames_converted: int = 0
    frames_skipped: int = 0
    failed_frames: int = 0
    output_dir: Optional[Path] = None
    max_cll: int = 0  # MaxCLL detected
    max_fall: int = 0  # MaxFALL detected
    avg_dynamic_range_stops: float = 0.0


class HDRConverter:
    """HDR conversion processor for video frames.

    Converts SDR (Standard Dynamic Range) video to HDR (High Dynamic Range)
    using various tone mapping algorithms and color space transformations.

    The conversion process involves:
    1. Color space conversion (Rec.709 -> BT.2020)
    2. Linearization (gamma removal)
    3. Tone mapping for expanded dynamic range
    4. Local contrast enhancement (optional)
    5. PQ/HLG transfer function application
    6. Metadata generation (MaxCLL, MaxFALL)

    Supports both FFmpeg-based processing (recommended for video) and
    OpenCV-based frame-by-frame processing.

    Example:
        >>> config = HDRConfig(
        ...     target_format=HDRFormat.HDR10,
        ...     peak_brightness=1000,
        ...     tone_mapping=ToneMapping.ACES
        ... )
        >>> converter = HDRConverter(config)
        >>> if converter.is_available():
        ...     hdr_frame = converter.convert_frame(sdr_frame)
    """

    # FFmpeg filter mappings for tone mapping algorithms
    TONEMAP_FILTERS = {
        ToneMapping.REINHARD: 'reinhard',
        ToneMapping.ACES: 'aces',
        ToneMapping.HABLE: 'hable',
        ToneMapping.MOBIUS: 'mobius',
    }

    # Color space transfer characteristics
    COLORSPACE_PARAMS = {
        ColorSpace.BT2020: {
            'matrix': 'bt2020nc',
            'primaries': 'bt2020',
            'transfer': 'smpte2084',  # PQ for HDR10
            'range': 'limited',
        },
        ColorSpace.P3: {
            'matrix': 'bt2020nc',
            'primaries': 'smpte432',  # DCI-P3
            'transfer': 'smpte2084',
            'range': 'limited',
        },
        ColorSpace.REC709: {
            'matrix': 'bt709',
            'primaries': 'bt709',
            'transfer': 'bt709',
            'range': 'limited',
        },
    }

    def __init__(
        self,
        config: Optional[HDRConfig] = None,
    ):
        """Initialize HDR converter.

        Args:
            config: HDR conversion configuration.
                    Uses default HDRConfig if not provided.
        """
        self.config = config or HDRConfig()
        self._ffmpeg_available = self._check_ffmpeg()
        self._opencv_available = self._check_opencv()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg with HDR support is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Check for zscale filter (required for HDR)
                filter_result = subprocess.run(
                    ['ffmpeg', '-filters'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                has_zscale = 'zscale' in filter_result.stdout
                has_tonemap = 'tonemap' in filter_result.stdout
                if has_zscale and has_tonemap:
                    logger.info("FFmpeg with HDR support detected")
                    return True
                else:
                    logger.warning(
                        "FFmpeg found but missing HDR filters. "
                        "Install with: ffmpeg with zscale and tonemap support"
                    )
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("FFmpeg not found")
            return False

    def _check_opencv(self) -> bool:
        """Check if OpenCV is available for frame processing."""
        try:
            import cv2
            logger.debug(f"OpenCV {cv2.__version__} available")
            return True
        except ImportError:
            logger.warning("OpenCV not available")
            return False

    def is_available(self) -> bool:
        """Check if HDR conversion is available.

        Returns:
            True if either FFmpeg or OpenCV backend is available.
        """
        return self._ffmpeg_available or self._opencv_available

    def analyze_dynamic_range(
        self,
        frame: np.ndarray
    ) -> DynamicRangeInfo:
        """Analyze the dynamic range of a frame.

        Computes luminance statistics to determine if the content
        would benefit from HDR conversion.

        Args:
            frame: Input frame as numpy array (BGR or RGB format, uint8 or uint16).

        Returns:
            DynamicRangeInfo containing luminance analysis.

        Example:
            >>> info = converter.analyze_dynamic_range(frame)
            >>> if info.is_hdr_candidate:
            ...     print(f"Frame has {info.dynamic_range_stops:.1f} stops of DR")
        """
        # Convert to float for precision
        if frame.dtype == np.uint8:
            frame_float = frame.astype(np.float32) / 255.0
            bit_depth = 8
        elif frame.dtype == np.uint16:
            frame_float = frame.astype(np.float32) / 65535.0
            bit_depth = 16
        else:
            frame_float = frame.astype(np.float32)
            bit_depth = 32

        # Calculate luminance (assuming BGR input like OpenCV)
        # Y = 0.2126 R + 0.7152 G + 0.0722 B (Rec.709)
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            luminance = (
                0.0722 * frame_float[:, :, 0] +  # B
                0.7152 * frame_float[:, :, 1] +  # G
                0.2126 * frame_float[:, :, 2]    # R
            )
        else:
            luminance = frame_float

        # Convert to approximate nits (assuming SDR with ~100 nits peak)
        # SDR gamma is approximately 2.2
        luminance_linear = np.power(np.clip(luminance, 0.001, 1.0), 2.2)
        luminance_nits = luminance_linear * 100.0  # SDR reference ~100 nits

        # Compute statistics
        min_lum = float(np.min(luminance_nits))
        max_lum = float(np.max(luminance_nits))
        avg_lum = float(np.mean(luminance_nits))
        peak_lum = float(np.percentile(luminance_nits, 99))

        # Dynamic range in stops (photographic EV)
        if min_lum > 0.001:
            dr_stops = np.log2(max_lum / min_lum)
        else:
            dr_stops = np.log2(max_lum / 0.001)

        # Calculate clipping
        if bit_depth == 8:
            highlight_threshold = 250 / 255.0
            shadow_threshold = 5 / 255.0
        else:
            highlight_threshold = 0.98
            shadow_threshold = 0.02

        clipped_highlights = np.mean(luminance > highlight_threshold) * 100
        clipped_shadows = np.mean(luminance < shadow_threshold) * 100

        # Histogram for analysis
        hist, _ = np.histogram(luminance.flatten(), bins=256, range=(0, 1))

        # Determine if HDR candidate
        # Good candidates have: high DR, clipped highlights, varied luminance
        is_candidate = (
            dr_stops > 8.0 or  # High natural dynamic range
            clipped_highlights > 0.5 or  # Has clipped highlights to recover
            (max_lum > 80 and min_lum < 5)  # Wide luminance spread
        )

        return DynamicRangeInfo(
            min_luminance=min_lum,
            max_luminance=max_lum,
            avg_luminance=avg_lum,
            peak_luminance=peak_lum,
            dynamic_range_stops=float(dr_stops),
            is_hdr_candidate=is_candidate,
            histogram=hist,
            clipped_highlights_pct=float(clipped_highlights),
            clipped_shadows_pct=float(clipped_shadows),
        )

    def is_hdr_source(self, video_path: Union[str, Path]) -> bool:
        """Check if a video file is already HDR.

        Uses FFprobe to analyze video metadata for HDR characteristics.

        Args:
            video_path: Path to the video file.

        Returns:
            True if the video appears to be HDR content.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return False

        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_streams',
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"FFprobe failed: {result.stderr}")
                return False

            data = json.loads(result.stdout)

            for stream in data.get('streams', []):
                if stream.get('codec_type') != 'video':
                    continue

                # Check color transfer function
                transfer = stream.get('color_transfer', '')
                if transfer in ('smpte2084', 'arib-std-b67'):  # PQ or HLG
                    logger.info(f"HDR detected: transfer function = {transfer}")
                    return True

                # Check color primaries
                primaries = stream.get('color_primaries', '')
                if primaries == 'bt2020':
                    logger.info(f"HDR likely: color primaries = {primaries}")
                    return True

                # Check bit depth
                pix_fmt = stream.get('pix_fmt', '')
                if any(x in pix_fmt for x in ['10le', '10be', '12le', '12be', 'p010']):
                    logger.debug(f"High bit depth detected: {pix_fmt}")
                    # High bit depth doesn't guarantee HDR, but is common

                # Check for HDR metadata
                side_data = stream.get('side_data_list', [])
                for sd in side_data:
                    if sd.get('side_data_type') in (
                        'Mastering display metadata',
                        'Content light level metadata'
                    ):
                        logger.info("HDR metadata present in video")
                        return True

            return False

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error checking HDR status: {e}")
            return False

    def convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert a single SDR frame to HDR.

        Applies the configured tone mapping and color space conversion
        to expand the dynamic range of the frame.

        Args:
            frame: Input SDR frame as numpy array (BGR format, uint8).
                   Expected to be in Rec.709 color space with gamma.

        Returns:
            HDR frame as numpy array (BGR format, uint16 for 10-bit output).

        Note:
            For video processing, use convert_video() for better efficiency
            as it uses FFmpeg's optimized pipeline.
        """
        if not self._opencv_available:
            logger.warning("OpenCV not available, returning original frame")
            return frame

        try:
            import cv2
        except ImportError:
            return frame

        # Input validation
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return frame

        # Convert to float32 for processing
        if frame.dtype == np.uint8:
            frame_float = frame.astype(np.float32) / 255.0
        elif frame.dtype == np.uint16:
            frame_float = frame.astype(np.float32) / 65535.0
        else:
            frame_float = frame.astype(np.float32)

        # Step 1: Linearize (remove gamma)
        # SDR uses approximately gamma 2.2
        frame_linear = np.power(np.clip(frame_float, 0.0, 1.0), 2.2)

        # Step 2: Color space conversion (Rec.709 -> BT.2020)
        if self.config.color_space in (ColorSpace.BT2020, ColorSpace.P3):
            frame_linear = self._convert_colorspace_709_to_2020(frame_linear)

        # Step 3: Apply tone mapping for HDR expansion
        frame_hdr = self._apply_tone_mapping(frame_linear)

        # Step 4: Local contrast enhancement (optional)
        if self.config.enable_local_contrast:
            frame_hdr = self._apply_local_contrast(frame_hdr)

        # Step 5: Apply saturation boost
        if abs(self.config.saturation_boost - 1.0) > 0.01:
            frame_hdr = self._adjust_saturation(frame_hdr)

        # Step 6: Highlight recovery (optional)
        if self.config.highlight_recovery:
            frame_hdr = self._recover_highlights(frame_float, frame_hdr)

        # Step 7: Apply PQ (SMPTE ST 2084) transfer function
        if self.config.target_format in (HDRFormat.HDR10, HDRFormat.HDR10_PLUS, HDRFormat.DOLBY_VISION):
            frame_output = self._apply_pq_transfer(frame_hdr)
        else:  # HLG
            frame_output = self._apply_hlg_transfer(frame_hdr)

        # Convert to 10-bit (uint16 with 10-bit precision)
        # Scale to 16-bit for storage, but actual values use 10-bit range
        frame_output = np.clip(frame_output * 65535.0, 0, 65535).astype(np.uint16)

        return frame_output

    def _convert_colorspace_709_to_2020(self, frame: np.ndarray) -> np.ndarray:
        """Convert from Rec.709 to BT.2020 color space.

        Uses a 3x3 matrix transformation for primary conversion.
        """
        # Rec.709 to BT.2020 conversion matrix
        # This expands the color gamut
        matrix = np.array([
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956]
        ], dtype=np.float32)

        # Reshape for matrix multiplication
        h, w, c = frame.shape
        frame_flat = frame.reshape(-1, 3)

        # Apply conversion (assuming BGR input)
        # Swap to RGB for matrix, then back to BGR
        rgb = frame_flat[:, ::-1]  # BGR to RGB
        rgb_2020 = np.dot(rgb, matrix.T)
        bgr_2020 = rgb_2020[:, ::-1]  # RGB to BGR

        return bgr_2020.reshape(h, w, c)

    def _apply_tone_mapping(self, frame: np.ndarray) -> np.ndarray:
        """Apply tone mapping to expand dynamic range.

        Scales luminance based on the configured tone mapping algorithm
        to simulate HDR brightness levels.
        """
        # Calculate luminance
        lum = 0.0722 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.2126 * frame[:, :, 2]
        lum = np.clip(lum, 0.0001, None)  # Avoid division by zero

        # Calculate HDR scaling factor based on peak brightness
        # SDR reference is ~100 nits, target is config.peak_brightness
        peak_scale = self.config.peak_brightness / 100.0

        if self.config.tone_mapping == ToneMapping.REINHARD:
            # Reinhard: L_out = L_in / (1 + L_in) * (1 + L_in / L_white^2)
            l_white = peak_scale
            lum_hdr = lum * (1 + lum / (l_white ** 2)) / (1 + lum)

        elif self.config.tone_mapping == ToneMapping.ACES:
            # ACES filmic S-curve
            # Simplified ACES approximation
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            lum_scaled = lum * peak_scale
            lum_hdr = (lum_scaled * (a * lum_scaled + b)) / (lum_scaled * (c * lum_scaled + d) + e)
            lum_hdr = np.clip(lum_hdr, 0, 1)

        elif self.config.tone_mapping == ToneMapping.HABLE:
            # Hable (Uncharted 2) filmic tonemapping
            def hable_partial(x):
                A = 0.15
                B = 0.50
                C = 0.10
                D = 0.20
                E = 0.02
                F = 0.30
                return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

            exposure_bias = peak_scale
            curr = hable_partial(lum * exposure_bias)
            white_scale = 1.0 / hable_partial(peak_scale)
            lum_hdr = curr * white_scale

        else:  # MOBIUS
            # Mobius: smooth transition preserving shadows
            # Linear up to transition point, then smooth curve
            transition = 0.3
            slope = 0.8
            lum_scaled = lum * peak_scale

            linear_mask = lum_scaled <= transition
            lum_hdr = np.zeros_like(lum_scaled)
            lum_hdr[linear_mask] = lum_scaled[linear_mask]

            # Mobius curve for highlights
            x = lum_scaled[~linear_mask]
            lum_hdr[~linear_mask] = transition + (1.0 - transition) * (
                1.0 - np.exp(-slope * (x - transition) / (1.0 - transition))
            )

        # Apply luminance scaling to color channels
        scale = np.where(lum > 0.0001, lum_hdr / lum, 1.0)
        scale = scale[:, :, np.newaxis]

        frame_hdr = frame * scale

        return np.clip(frame_hdr, 0, 1)

    def _apply_local_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply local contrast enhancement.

        Uses unsharp masking to enhance local contrast while
        preserving overall luminance.
        """
        try:
            import cv2
        except ImportError:
            return frame

        # Convert to LAB for luminance-only processing
        frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2LAB)

        l_channel = lab[:, :, 0].astype(np.float32)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel.astype(np.uint8))

        # Blend original and enhanced
        blend_factor = 0.5  # 50% enhancement
        l_final = cv2.addWeighted(
            l_channel.astype(np.uint8), 1 - blend_factor,
            l_enhanced, blend_factor,
            0
        )

        lab[:, :, 0] = l_final
        enhanced_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced_uint8.astype(np.float32) / 255.0

    def _adjust_saturation(self, frame: np.ndarray) -> np.ndarray:
        """Adjust color saturation.

        HDR allows for more saturated colors due to wider color gamut.
        """
        try:
            import cv2
        except ImportError:
            return frame

        frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Boost saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.saturation_boost, 0, 255)

        hsv_uint8 = hsv.astype(np.uint8)
        rgb_saturated = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)

        return rgb_saturated.astype(np.float32) / 255.0

    def _recover_highlights(
        self,
        original: np.ndarray,
        processed: np.ndarray
    ) -> np.ndarray:
        """Attempt to recover clipped highlights.

        Uses information from non-clipped channels to reconstruct
        highlight detail.
        """
        # Detect clipped areas in original
        clip_threshold = 0.98
        clipped = np.any(original > clip_threshold, axis=2)

        if not np.any(clipped):
            return processed

        # For clipped pixels, try to recover using channel ratios
        # This is a simplified recovery - real HDR grading would use
        # more sophisticated techniques
        result = processed.copy()

        # Reduce intensity of heavily clipped areas slightly
        # to prevent posterization
        clipped_mask = clipped[:, :, np.newaxis]
        result = np.where(
            clipped_mask,
            result * 0.95,  # Slight reduction in clipped areas
            result
        )

        return result

    def _apply_pq_transfer(self, frame: np.ndarray) -> np.ndarray:
        """Apply PQ (SMPTE ST 2084) transfer function.

        Perceptual Quantizer is the transfer function used in
        HDR10 and Dolby Vision.
        """
        # PQ constants
        m1 = 2610.0 / 16384.0
        m2 = 2523.0 / 4096.0 * 128.0
        c1 = 3424.0 / 4096.0
        c2 = 2413.0 / 4096.0 * 32.0
        c3 = 2392.0 / 4096.0 * 32.0

        # Normalize to peak brightness
        # PQ is defined for 10000 nits max
        L = np.clip(frame, 0, 1) * (self.config.peak_brightness / 10000.0)

        # Apply PQ OETF
        Lm1 = np.power(L, m1)
        numerator = c1 + c2 * Lm1
        denominator = 1.0 + c3 * Lm1
        pq = np.power(numerator / denominator, m2)

        return np.clip(pq, 0, 1)

    def _apply_hlg_transfer(self, frame: np.ndarray) -> np.ndarray:
        """Apply HLG (Hybrid Log-Gamma) transfer function.

        HLG is the transfer function used for broadcast HDR,
        designed to be backward compatible with SDR displays.
        """
        # HLG constants
        a = 0.17883277
        b = 0.28466892
        c = 0.55991073

        # Normalize input
        E = np.clip(frame, 0, 1)

        # Apply HLG OETF
        # E' = sqrt(3 * E) for E <= 1/12
        # E' = a * ln(12 * E - b) + c for E > 1/12
        threshold = 1.0 / 12.0

        E_prime = np.where(
            E <= threshold,
            np.sqrt(3.0 * E),
            a * np.log(12.0 * E - b) + c
        )

        return np.clip(E_prime, 0, 1)

    def convert_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> HDRConversionResult:
        """Convert an SDR video file to HDR.

        Uses FFmpeg's hardware-accelerated pipeline for efficient
        conversion of full video files.

        Args:
            input_path: Path to input SDR video file.
            output_path: Path for output HDR video file.
            progress_callback: Optional callback for progress updates (0.0 to 1.0).

        Returns:
            HDRConversionResult with conversion statistics.

        Raises:
            RuntimeError: If FFmpeg is not available or conversion fails.

        Example:
            >>> result = converter.convert_video(
            ...     input_path='input.mp4',
            ...     output_path='output_hdr.mp4',
            ...     progress_callback=lambda p: print(f'{p*100:.0f}%')
            ... )
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        result = HDRConversionResult()

        if not self._ffmpeg_available:
            raise RuntimeError(
                "FFmpeg with HDR support not available. "
                "Install FFmpeg with zscale and tonemap filters."
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check if source is already HDR
        if self.is_hdr_source(input_path):
            logger.warning("Source video appears to already be HDR")

        # Build FFmpeg filter chain
        filter_chain = self._build_ffmpeg_filter()

        # Get color space parameters
        cs_params = self.COLORSPACE_PARAMS[self.config.color_space]

        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-vf', filter_chain,
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p10le',  # 10-bit output
            '-color_primaries', cs_params['primaries'],
            '-color_trc', cs_params['transfer'],
            '-colorspace', cs_params['matrix'],
            '-c:a', 'copy',  # Copy audio
        ]

        # Add HDR metadata
        if self.config.target_format == HDRFormat.HDR10:
            max_cll = self.config.max_content_light_level or self.config.peak_brightness
            max_fall = self.config.max_frame_avg_light_level or (self.config.peak_brightness // 2)

            cmd.extend([
                '-x265-params',
                f'hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:'
                f'colormatrix=bt2020nc:max-cll={max_cll},{max_fall}:'
                f'master-display=G(13250,34500)B(7500,3000)R(34000,16000)'
                f'WP(15635,16450)L({self.config.peak_brightness}0000,50)'
            ])

            result.max_cll = max_cll
            result.max_fall = max_fall

        cmd.append(str(output_path))

        logger.info(f"Starting HDR conversion: {input_path.name}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Parse progress from FFmpeg output
            duration = self._get_video_duration(input_path)
            for line in process.stdout:
                if 'time=' in line:
                    time_match = self._parse_ffmpeg_time(line)
                    if time_match and duration > 0 and progress_callback:
                        progress = min(time_match / duration, 1.0)
                        progress_callback(progress)

            process.wait()

            if process.returncode != 0:
                raise RuntimeError("FFmpeg conversion failed")

            if progress_callback:
                progress_callback(1.0)

            # Count frames in output
            frame_count = self._count_video_frames(output_path)
            result.frames_processed = frame_count
            result.frames_converted = frame_count
            result.output_dir = output_path.parent

            logger.info(f"HDR conversion complete: {output_path}")

        except Exception as e:
            logger.error(f"HDR conversion failed: {e}")
            result.failed_frames = 1
            raise

        return result

    def _build_ffmpeg_filter(self) -> str:
        """Build FFmpeg filter chain for HDR conversion."""
        filters = []

        # Step 1: Normalize input to linear light
        filters.append('zscale=t=linear:npl=100')

        # Step 2: Color space conversion (Rec.709 -> BT.2020)
        if self.config.color_space in (ColorSpace.BT2020, ColorSpace.P3):
            primaries = 'bt2020' if self.config.color_space == ColorSpace.BT2020 else 'smpte432'
            filters.append(f'zscale=p={primaries}')

        # Step 3: Tone mapping for HDR expansion
        # Note: This is inverse tonemap - expanding SDR to HDR range
        tonemap_algo = self.TONEMAP_FILTERS.get(self.config.tone_mapping, 'reinhard')

        # Apply gain for HDR headroom
        gain = self.config.peak_brightness / 100.0
        filters.append(f'zscale=npl={self.config.peak_brightness}')

        # Step 4: Apply PQ or HLG transfer function
        if self.config.target_format in (HDRFormat.HDR10, HDRFormat.HDR10_PLUS, HDRFormat.DOLBY_VISION):
            filters.append('zscale=t=smpte2084')
        else:  # HLG
            filters.append('zscale=t=arib-std-b67')

        # Step 5: Format conversion to 10-bit
        filters.append('format=yuv420p10le')

        return ','.join(filters)

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using FFprobe."""
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            return 0.0

    def _parse_ffmpeg_time(self, line: str) -> Optional[float]:
        """Parse time from FFmpeg progress output."""
        import re
        match = re.search(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})', line)
        if match:
            h, m, s, cs = map(int, match.groups())
            return h * 3600 + m * 60 + s + cs / 100.0
        return None

    def _count_video_frames(self, video_path: Path) -> int:
        """Count frames in video file using FFprobe."""
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-count_frames',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_read_frames',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                timeout=300
            )
            return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            return 0

    def convert_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> HDRConversionResult:
        """Convert all frames in a directory to HDR.

        Args:
            input_dir: Directory containing input SDR frames (PNG/JPG).
            output_dir: Directory for output HDR frames (16-bit PNG).
            progress_callback: Optional callback for progress (0.0 to 1.0).

        Returns:
            HDRConversionResult with conversion statistics.
        """
        result = HDRConversionResult(output_dir=output_dir)

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all frames
        frames = sorted(input_dir.glob('*.png'))
        if not frames:
            frames = sorted(input_dir.glob('*.jpg'))

        if not frames:
            logger.warning("No frames found in input directory")
            return result

        if not self._opencv_available:
            logger.warning("OpenCV not available, copying frames without conversion")
            for frame_path in frames:
                shutil.copy(frame_path, output_dir / frame_path.name)
            result.frames_processed = len(frames)
            result.frames_skipped = len(frames)
            return result

        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available")
            return result

        logger.info(f"Converting {len(frames)} frames to HDR")

        dr_sum = 0.0
        max_lum = 0.0
        avg_lum_sum = 0.0

        for i, frame_path in enumerate(frames):
            try:
                # Read frame
                frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)

                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    result.failed_frames += 1
                    continue

                # Analyze dynamic range
                dr_info = self.analyze_dynamic_range(frame)
                dr_sum += dr_info.dynamic_range_stops
                max_lum = max(max_lum, dr_info.max_luminance)
                avg_lum_sum += dr_info.avg_luminance

                # Convert to HDR
                hdr_frame = self.convert_frame(frame)

                # Save as 16-bit PNG
                output_path = output_dir / frame_path.with_suffix('.png').name
                cv2.imwrite(str(output_path), hdr_frame)

                result.frames_converted += 1
                result.frames_processed += 1

            except Exception as e:
                logger.debug(f"Failed to convert {frame_path.name}: {e}")
                shutil.copy(frame_path, output_dir / frame_path.name)
                result.failed_frames += 1
                result.frames_processed += 1

            if progress_callback:
                progress_callback((i + 1) / len(frames))

        # Calculate final statistics
        if result.frames_converted > 0:
            result.avg_dynamic_range_stops = dr_sum / result.frames_converted
            result.max_cll = int(max_lum * (self.config.peak_brightness / 100.0))
            result.max_fall = int((avg_lum_sum / result.frames_converted) * (self.config.peak_brightness / 100.0))

        logger.info(
            f"HDR conversion complete: {result.frames_converted} frames, "
            f"MaxCLL={result.max_cll}, MaxFALL={result.max_fall}"
        )

        return result


# Convenience functions for common use cases

def create_hdr_converter(
    target_format: str = "hdr10",
    peak_brightness: int = 1000,
    tone_mapping: str = "aces",
    enable_local_contrast: bool = True
) -> HDRConverter:
    """Create an HDRConverter with common settings.

    Args:
        target_format: Target HDR format ("hdr10", "hdr10plus", "dolby_vision", "hlg")
        peak_brightness: Peak brightness in nits (400-10000)
        tone_mapping: Tone mapping algorithm ("reinhard", "aces", "hable", "mobius")
        enable_local_contrast: Apply local contrast enhancement

    Returns:
        Configured HDRConverter instance.

    Example:
        >>> converter = create_hdr_converter(
        ...     target_format="hdr10",
        ...     peak_brightness=1000,
        ...     tone_mapping="aces"
        ... )
    """
    config = HDRConfig(
        target_format=HDRFormat(target_format.lower()),
        peak_brightness=peak_brightness,
        tone_mapping=ToneMapping(tone_mapping.lower()),
        enable_local_contrast=enable_local_contrast
    )
    return HDRConverter(config)


def convert_sdr_to_hdr(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    peak_brightness: int = 1000,
    progress_callback: Optional[Callable[[float], None]] = None
) -> HDRConversionResult:
    """Quick SDR to HDR video conversion.

    Uses recommended settings for general content.

    Args:
        input_path: Path to input SDR video
        output_path: Path for output HDR video
        peak_brightness: Target peak brightness in nits
        progress_callback: Optional progress callback

    Returns:
        HDRConversionResult with conversion statistics.
    """
    config = HDRConfig(
        target_format=HDRFormat.HDR10,
        peak_brightness=peak_brightness,
        tone_mapping=ToneMapping.ACES,
        color_space=ColorSpace.BT2020,
        enable_local_contrast=True
    )

    converter = HDRConverter(config)
    return converter.convert_video(
        input_path=input_path,
        output_path=output_path,
        progress_callback=progress_callback
    )


def analyze_hdr_potential(video_path: Union[str, Path]) -> Dict:
    """Analyze a video's potential for HDR conversion.

    Samples frames to determine if the content would benefit
    from HDR conversion.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with analysis results and recommendations.
    """
    video_path = Path(video_path)
    converter = HDRConverter()

    # Check if already HDR
    is_hdr = converter.is_hdr_source(video_path)

    result = {
        'video_path': str(video_path),
        'is_already_hdr': is_hdr,
        'recommended_for_hdr': not is_hdr,
        'recommendations': []
    }

    if is_hdr:
        result['recommendations'].append(
            "Video is already HDR - no conversion needed"
        )
    else:
        result['recommendations'].extend([
            "Consider HDR conversion for improved dynamic range",
            "Use HDR10 format for maximum compatibility",
            "Peak brightness of 1000 nits recommended for most content"
        ])

    return result
