"""HDR expansion processor for converting SDR content to HDR formats.

This module provides functionality to expand standard dynamic range (SDR) video
content to high dynamic range (HDR) formats including HDR10, HDR10+, Dolby Vision,
and HLG.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional
import json
import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


class HDRFormat(Enum):
    """Supported HDR output formats."""

    HDR10 = "hdr10"
    HDR10_PLUS = "hdr10plus"
    DOLBY_VISION = "dolby_vision"
    HLG = "hlg"


class ToneMappingMethod(Enum):
    """Tone mapping methods for HDR expansion."""

    ADAPTIVE = "adaptive"
    REINHARD = "reinhard"
    ACES = "aces"
    LINEAR = "linear"


@dataclass
class HDRConfig:
    """Configuration for HDR expansion processing.

    Attributes:
        target_format: The HDR format to convert to.
        peak_brightness: Target peak brightness in nits (cd/m^2).
        color_space: Target color space (bt2020, p3, etc.).
        transfer_function: Transfer function to use (pq for ST2084, or hlg).
        tone_mapping_method: Method for tone expansion (adaptive, reinhard, aces).
        preserve_highlights: Whether to preserve highlight detail during expansion.
        expand_shadows: Whether to expand shadow detail for HDR.
    """

    target_format: HDRFormat = HDRFormat.HDR10
    peak_brightness: int = 1000  # nits
    color_space: str = "bt2020"
    transfer_function: str = "pq"  # PQ (ST2084) or HLG
    tone_mapping_method: str = "adaptive"  # adaptive, reinhard, aces
    preserve_highlights: bool = True
    expand_shadows: bool = True


@dataclass
class HDRMetadata:
    """HDR metadata for video content.

    Attributes:
        max_cll: Maximum Content Light Level in nits.
        max_fall: Maximum Frame Average Light Level in nits.
        master_display: SMPTE ST2086 mastering display metadata string.
        color_primaries: Color primaries specification (bt2020, p3, etc.).
    """

    max_cll: int = 1000  # MaxCLL
    max_fall: int = 400  # MaxFALL
    master_display: str = ""  # SMPTE ST2086 string
    color_primaries: str = "bt2020"

    def to_ffmpeg_metadata(self) -> List[str]:
        """Convert metadata to ffmpeg command-line arguments.

        Returns:
            List of ffmpeg arguments for metadata injection.
        """
        args = []

        # MaxCLL and MaxFALL
        if self.max_cll or self.max_fall:
            args.extend([
                "-x265-params",
                f"max-cll={self.max_cll},{self.max_fall}"
            ])

        # Master display metadata (SMPTE ST2086)
        if self.master_display:
            args.extend([
                "-x265-params",
                f"master-display={self.master_display}"
            ])

        return args


class HDRExpander:
    """Expands SDR video content to HDR formats.

    This class provides methods to analyze SDR content and expand it to
    various HDR formats using ffmpeg with appropriate filters and metadata.
    """

    def __init__(self, config: HDRConfig) -> None:
        """Initialize the HDR expander with configuration.

        Args:
            config: HDR expansion configuration settings.
        """
        self.config = config
        self._ffmpeg_path: Optional[str] = None

    def is_available(self) -> bool:
        """Check if ffmpeg with HDR support is available.

        Returns:
            True if ffmpeg is available with required HDR filters.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.warning("ffmpeg not found in PATH")
            return False

        self._ffmpeg_path = ffmpeg_path

        # Check for required filters (zscale, tonemap)
        try:
            result = subprocess.run(
                [ffmpeg_path, "-filters"],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + result.stderr

            required_filters = ["zscale", "tonemap"]
            available = all(f in output for f in required_filters)

            if not available:
                logger.warning(
                    "ffmpeg missing required HDR filters (zscale, tonemap)"
                )

            # Check for libx265 encoder for HDR10
            result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True,
                timeout=30
            )
            encoder_output = result.stdout + result.stderr

            if "libx265" not in encoder_output:
                logger.warning("ffmpeg missing libx265 encoder for HDR output")
                return False

            return available

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg filter check timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking ffmpeg capabilities: {e}")
            return False

    def expand_to_hdr(
        self,
        input_path: Path,
        output_path: Path
    ) -> Path:
        """Expand SDR video to HDR format.

        Args:
            input_path: Path to input SDR video file.
            output_path: Path for output HDR video file.

        Returns:
            Path to the output HDR video file.

        Raises:
            RuntimeError: If HDR expansion fails.
            FileNotFoundError: If input file doesn't exist.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not self.is_available():
            raise RuntimeError("ffmpeg with HDR support is not available")

        # Analyze input for optimal HDR metadata
        metadata = self.analyze_for_hdr(input_path)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply tone expansion
        self._apply_tone_expansion(input_path, output_path)

        # Inject HDR metadata
        self._inject_hdr_metadata(output_path, metadata)

        logger.info(f"HDR expansion complete: {output_path}")
        return output_path

    def analyze_for_hdr(self, input_path: Path) -> HDRMetadata:
        """Analyze video content for optimal HDR expansion parameters.

        Args:
            input_path: Path to input video file.

        Returns:
            HDRMetadata with analyzed values for the content.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Default metadata based on config
        metadata = HDRMetadata(
            max_cll=self.config.peak_brightness,
            max_fall=int(self.config.peak_brightness * 0.4),
            color_primaries=self.config.color_space
        )

        # Set master display based on target format
        if self.config.target_format == HDRFormat.HDR10:
            # Standard HDR10 mastering display (BT.2020 primaries, D65 white)
            metadata.master_display = (
                "G(13250,34500)B(7500,3000)R(34000,16000)"
                "WP(15635,16450)L(10000000,1)"
            )
        elif self.config.target_format == HDRFormat.HLG:
            # HLG typically doesn't use mastering display metadata
            metadata.master_display = ""
        elif self.config.target_format == HDRFormat.DOLBY_VISION:
            # Dolby Vision uses different metadata structure
            metadata.master_display = (
                "G(13250,34500)B(7500,3000)R(34000,16000)"
                "WP(15635,16450)L(40000000,50)"
            )

        # Try to analyze actual content brightness levels
        try:
            ffprobe_path = shutil.which("ffprobe")
            if ffprobe_path:
                result = subprocess.run(
                    [
                        ffprobe_path,
                        "-v", "quiet",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=color_primaries,color_transfer",
                        "-of", "json",
                        str(input_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    probe_data = json.loads(result.stdout)
                    streams = probe_data.get("streams", [])
                    if streams:
                        stream = streams[0]
                        # Detect if already HDR
                        color_transfer = stream.get("color_transfer", "")
                        if color_transfer in ["smpte2084", "arib-std-b67"]:
                            logger.info(
                                f"Input already appears to be HDR: {color_transfer}"
                            )

        except Exception as e:
            logger.debug(f"Could not analyze input video: {e}")

        return metadata

    def _apply_tone_expansion(
        self,
        input_path: Path,
        output_path: Path
    ) -> None:
        """Apply tone expansion to convert SDR to HDR range.

        Args:
            input_path: Path to input SDR video.
            output_path: Path for output HDR video.

        Raises:
            RuntimeError: If tone expansion fails.
        """
        cmd = self._build_ffmpeg_command(input_path, output_path)

        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for long videos
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                raise RuntimeError(
                    f"Tone expansion failed with code {result.returncode}"
                )

        except subprocess.TimeoutExpired:
            raise RuntimeError("Tone expansion timed out")

    def _inject_hdr_metadata(
        self,
        video_path: Path,
        metadata: HDRMetadata
    ) -> None:
        """Inject HDR metadata into video file.

        For most formats, metadata is injected during encoding. This method
        handles any post-processing metadata requirements.

        Args:
            video_path: Path to video file to update.
            metadata: HDR metadata to inject.
        """
        # For HDR10+, we would need to inject dynamic metadata
        if self.config.target_format == HDRFormat.HDR10_PLUS:
            logger.info(
                "HDR10+ dynamic metadata injection requires additional tools"
            )
            # HDR10+ metadata injection would require hdr10plus_tool
            # This is a placeholder for future implementation

        # For Dolby Vision, RPU injection would be needed
        elif self.config.target_format == HDRFormat.DOLBY_VISION:
            logger.info(
                "Dolby Vision requires proprietary tools for full compliance"
            )

        # HDR10 and HLG metadata is injected during encoding
        logger.debug(f"HDR metadata applied: MaxCLL={metadata.max_cll}, "
                     f"MaxFALL={metadata.max_fall}")

    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path
    ) -> List[str]:
        """Build ffmpeg command for HDR expansion.

        Args:
            input_path: Path to input video.
            output_path: Path for output video.

        Returns:
            List of command arguments for ffmpeg.
        """
        ffmpeg = self._ffmpeg_path or "ffmpeg"

        # Build filter chain for SDR to HDR conversion
        filter_chain = self._build_filter_chain()

        cmd = [
            ffmpeg,
            "-y",  # Overwrite output
            "-i", str(input_path),
        ]

        # Add filter chain
        cmd.extend(["-vf", filter_chain])

        # Video codec settings for HDR
        cmd.extend([
            "-c:v", "libx265",
            "-preset", "medium",
            "-crf", "18",
        ])

        # Color settings based on target format
        if self.config.target_format == HDRFormat.HLG:
            cmd.extend([
                "-color_primaries", "bt2020",
                "-color_trc", "arib-std-b67",
                "-colorspace", "bt2020nc",
            ])
        else:
            # HDR10, HDR10+, Dolby Vision use PQ
            cmd.extend([
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
            ])

        # x265 params for HDR
        x265_params = self._build_x265_params()
        if x265_params:
            cmd.extend(["-x265-params", x265_params])

        # Audio passthrough
        cmd.extend(["-c:a", "copy"])

        # Output format
        cmd.extend(["-f", "matroska", str(output_path)])

        return cmd

    def _build_filter_chain(self) -> str:
        """Build ffmpeg filter chain for HDR expansion.

        Returns:
            Filter chain string for ffmpeg -vf option.
        """
        filters = []

        # Convert to linear light for processing
        filters.append("zscale=t=linear:npl=100")

        # Apply tone expansion based on method
        if self.config.tone_mapping_method == "adaptive":
            # Adaptive tone expansion with highlight and shadow control
            filters.append(
                f"zscale=p=bt2020:t=linear:m=bt2020nc:r=tv"
            )

            # Expand dynamic range
            if self.config.expand_shadows:
                filters.append("eq=gamma=0.9:contrast=1.1")

            if self.config.preserve_highlights:
                filters.append(
                    f"tonemap=tonemap=mobius:param=0.3:"
                    f"desat=0:peak={self.config.peak_brightness / 10000}"
                )

        elif self.config.tone_mapping_method == "reinhard":
            filters.append(
                f"zscale=p=bt2020:t=linear:m=bt2020nc"
            )
            filters.append(
                f"tonemap=tonemap=reinhard:"
                f"peak={self.config.peak_brightness / 10000}"
            )

        elif self.config.tone_mapping_method == "aces":
            filters.append(
                f"zscale=p=bt2020:t=linear:m=bt2020nc"
            )
            # ACES-style tone curve approximation
            filters.append("curves=master='0/0 0.25/0.3 0.5/0.6 0.75/0.85 1/1'")

        # Convert to target transfer function
        if self.config.transfer_function == "hlg":
            filters.append("zscale=t=arib-std-b67")
        else:
            # PQ (ST2084) for HDR10/HDR10+/Dolby Vision
            filters.append("zscale=t=smpte2084")

        # Final color space conversion
        filters.append(
            f"zscale=p={self.config.color_space}:"
            f"m={self.config.color_space}nc:r=tv"
        )

        # Set pixel format for 10-bit HDR
        filters.append("format=yuv420p10le")

        return ",".join(filters)

    def _build_x265_params(self) -> str:
        """Build x265 encoder parameters for HDR.

        Returns:
            x265-params string for ffmpeg.
        """
        params = []

        # HDR signaling
        params.append("hdr-opt=1")
        params.append("repeat-headers=1")

        # Color volume
        if self.config.target_format != HDRFormat.HLG:
            params.append(f"max-cll={self.config.peak_brightness},400")

            # Master display for HDR10
            if self.config.target_format == HDRFormat.HDR10:
                params.append(
                    "master-display="
                    "G(13250,34500)B(7500,3000)R(34000,16000)"
                    "WP(15635,16450)L(10000000,1)"
                )

        # Encoding quality
        params.append("aq-mode=3")
        params.append("psy-rd=1.0")

        return ":".join(params)


def expand_to_hdr(
    input_path: Path,
    output_path: Path,
    format: str = "hdr10"
) -> Path:
    """Factory function to expand SDR video to HDR.

    Args:
        input_path: Path to input SDR video file.
        output_path: Path for output HDR video file.
        format: Target HDR format (hdr10, hdr10plus, dolby_vision, hlg).

    Returns:
        Path to the output HDR video file.

    Raises:
        ValueError: If format is not recognized.
        RuntimeError: If expansion fails.
    """
    # Map format string to enum
    format_map = {
        "hdr10": HDRFormat.HDR10,
        "hdr10plus": HDRFormat.HDR10_PLUS,
        "hdr10_plus": HDRFormat.HDR10_PLUS,
        "dolby_vision": HDRFormat.DOLBY_VISION,
        "dolby": HDRFormat.DOLBY_VISION,
        "dv": HDRFormat.DOLBY_VISION,
        "hlg": HDRFormat.HLG,
    }

    format_lower = format.lower()
    if format_lower not in format_map:
        valid_formats = list(format_map.keys())
        raise ValueError(
            f"Unknown HDR format: {format}. Valid formats: {valid_formats}"
        )

    target_format = format_map[format_lower]

    # Configure based on format
    config = HDRConfig(target_format=target_format)

    # Adjust settings per format
    if target_format == HDRFormat.HLG:
        config.transfer_function = "hlg"
        config.peak_brightness = 1000
    elif target_format == HDRFormat.DOLBY_VISION:
        config.peak_brightness = 4000
    elif target_format == HDRFormat.HDR10_PLUS:
        config.peak_brightness = 4000

    # Create expander and process
    expander = HDRExpander(config)
    return expander.expand_to_hdr(input_path, output_path)


@dataclass
class HDRExpansionResult:
    """Result of HDR expansion processing."""

    success: bool
    output_path: Optional[Path]
    target_format: HDRFormat
    metadata: Optional[HDRMetadata] = None
    error_message: Optional[str] = None


def create_hdr_expander(
    target_format: HDRFormat = HDRFormat.HDR10,
    peak_brightness: int = 1000,
    tone_mapping: ToneMappingMethod = ToneMappingMethod.ADAPTIVE,
) -> HDRExpander:
    """Factory function to create an HDRExpander instance.

    Args:
        target_format: Target HDR format
        peak_brightness: Target peak brightness in nits
        tone_mapping: Tone mapping method to use

    Returns:
        Configured HDRExpander instance
    """
    config = HDRConfig(
        target_format=target_format,
        peak_brightness=peak_brightness,
        tone_mapping_method=tone_mapping.value,
    )
    return HDRExpander(config)
