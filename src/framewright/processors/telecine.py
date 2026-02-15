"""Inverse Telecine (IVTC) processor for removing pulldown patterns.

Inverse telecine removes the effects of the telecine process used to convert
film (24fps) to NTSC video (29.97fps) by detecting and removing duplicate fields.

Common telecine patterns:
- 3:2 pulldown (NTSC): Film to 29.97fps with 3-2-3-2 field pattern
- 2:3 pulldown (reverse): Alternative field ordering
- 2:2 pulldown (PAL): Film to 25fps, simpler field duplication

This module provides:
- Pattern detection (3:2, 2:3, 2:2, mixed)
- Field order detection (TFF, BFF, progressive)
- Automatic IVTC processing using ffmpeg filters
- Cadence break detection for hybrid content
"""

import logging
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TelecinePattern(Enum):
    """Telecine pulldown patterns.

    PATTERN_3_2: Standard NTSC 3:2 pulldown (24fps to 29.97fps)
    PATTERN_2_3: Reverse pulldown (alternative field ordering)
    PATTERN_2_2: PAL/film telecine (24fps to 25fps)
    PATTERN_MIXED: Variable pattern (hybrid content)
    PATTERN_UNKNOWN: Could not determine pattern
    """
    PATTERN_3_2 = "3:2"  # Standard NTSC pulldown
    PATTERN_2_3 = "2:3"  # Reverse pulldown
    PATTERN_2_2 = "2:2"  # PAL/film
    PATTERN_MIXED = "mixed"  # Variable pattern
    PATTERN_UNKNOWN = "unknown"


class FieldOrder(Enum):
    """Interlaced field order.

    TFF: Top field first (common in PAL, professional formats)
    BFF: Bottom field first (common in consumer NTSC)
    PROGRESSIVE: Non-interlaced content
    """
    TFF = "tff"  # Top field first
    BFF = "bff"  # Bottom field first
    PROGRESSIVE = "progressive"


@dataclass
class TelecineAnalysis:
    """Results of telecine pattern analysis.

    Attributes:
        pattern: Detected telecine pattern
        field_order: Detected field order (TFF, BFF, or progressive)
        confidence: Confidence score of detection (0.0 to 1.0)
        is_interlaced: Whether the content is interlaced
        cadence_breaks: Number of pattern interruptions detected
        source_fps: Estimated original frame rate before telecine
    """
    pattern: TelecinePattern
    field_order: FieldOrder
    confidence: float
    is_interlaced: bool
    cadence_breaks: int  # Pattern interruptions
    source_fps: float  # Estimated original FPS

    def summary(self) -> str:
        """Generate human-readable summary."""
        interlace_str = "interlaced" if self.is_interlaced else "progressive"
        return (
            f"Pattern: {self.pattern.value} ({interlace_str})\n"
            f"Field order: {self.field_order.value}\n"
            f"Confidence: {self.confidence:.1%}\n"
            f"Cadence breaks: {self.cadence_breaks}\n"
            f"Estimated source FPS: {self.source_fps:.3f}"
        )


@dataclass
class IVTCConfig:
    """Configuration for inverse telecine processing.

    Attributes:
        pattern: Telecine pattern to remove (PATTERN_UNKNOWN for auto-detect)
        field_order: Field order to use (TFF is common default)
        mode: Processing mode ('auto' or 'force_pattern')
        preserve_orphan_fields: Keep unpaired fields at scene changes
        output_fps: Target output FPS (None for automatic based on pattern)
    """
    pattern: TelecinePattern = TelecinePattern.PATTERN_UNKNOWN  # Auto-detect
    field_order: FieldOrder = FieldOrder.TFF
    mode: str = "auto"  # auto, force_pattern
    preserve_orphan_fields: bool = True
    output_fps: Optional[float] = None  # None = auto

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mode not in ("auto", "force_pattern"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'auto' or 'force_pattern'")
        if self.output_fps is not None and self.output_fps <= 0:
            raise ValueError(f"output_fps must be positive, got {self.output_fps}")


class IVTCError(Exception):
    """Exception raised for inverse telecine errors."""
    pass


class InverseTelecine:
    """Inverse telecine processor for removing pulldown patterns.

    Uses ffmpeg filters (fieldmatch, decimate, yadif, w3fdif) to detect
    and remove telecine patterns, converting interlaced telecined video
    back to its original progressive frame rate.

    Example:
        >>> config = IVTCConfig(pattern=TelecinePattern.PATTERN_UNKNOWN)
        >>> ivtc = InverseTelecine(config)
        >>> analysis = ivtc.analyze(Path("telecined_video.mp4"))
        >>> print(analysis.summary())
        >>> output = ivtc.process(
        ...     Path("telecined_video.mp4"),
        ...     Path("progressive_output.mp4")
        ... )
    """

    def __init__(self, config: Optional[IVTCConfig] = None):
        """Initialize the inverse telecine processor.

        Args:
            config: IVTC configuration (uses defaults if None)
        """
        self.config = config or IVTCConfig()
        self._verify_dependencies()

    def _verify_dependencies(self) -> None:
        """Verify that ffmpeg is installed with required filters."""
        if not shutil.which('ffmpeg'):
            raise IVTCError(
                "ffmpeg not found. Please install ffmpeg with filter support."
            )

        if not shutil.which('ffprobe'):
            raise IVTCError(
                "ffprobe not found. Please install ffmpeg (includes ffprobe)."
            )

        # Verify required filters are available
        try:
            result = subprocess.run(
                ['ffmpeg', '-filters'],
                capture_output=True,
                text=True,
                check=True
            )
            filters_output = result.stdout

            required_filters = ['fieldmatch', 'decimate', 'yadif']
            missing = [f for f in required_filters if f not in filters_output]

            if missing:
                logger.warning(
                    f"Some ffmpeg filters may not be available: {missing}. "
                    "IVTC may have limited functionality."
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not verify ffmpeg filters: {e}")

    def analyze(
        self,
        video_path: Path,
        sample_duration: float = 30.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> TelecineAnalysis:
        """Analyze video for telecine patterns.

        Examines the video to detect:
        - Whether content is interlaced
        - Telecine pattern (3:2, 2:3, 2:2, mixed)
        - Field order (TFF, BFF)
        - Number of cadence breaks
        - Estimated original frame rate

        Args:
            video_path: Path to input video
            sample_duration: Duration in seconds to analyze (longer = more accurate)
            progress_callback: Optional callback for progress updates

        Returns:
            TelecineAnalysis with detection results

        Raises:
            IVTCError: If analysis fails
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise IVTCError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing video for telecine patterns: {video_path}")

        if progress_callback:
            progress_callback(0.1)

        # Get video info
        video_info = self._get_video_info(video_path)
        current_fps = video_info.get('fps', 29.97)

        if progress_callback:
            progress_callback(0.2)

        # Detect if interlaced
        is_interlaced = self._detect_interlacing(video_path, sample_duration)

        if progress_callback:
            progress_callback(0.4)

        # Detect field order
        field_order = self._detect_field_order(video_path)

        if progress_callback:
            progress_callback(0.6)

        # Detect pattern
        pattern, confidence, cadence_breaks = self._detect_pattern(
            video_path, sample_duration
        )

        if progress_callback:
            progress_callback(0.9)

        # Estimate source FPS based on pattern
        source_fps = self._estimate_source_fps(current_fps, pattern)

        analysis = TelecineAnalysis(
            pattern=pattern,
            field_order=field_order,
            confidence=confidence,
            is_interlaced=is_interlaced,
            cadence_breaks=cadence_breaks,
            source_fps=source_fps
        )

        logger.info(f"Analysis complete:\n{analysis.summary()}")

        if progress_callback:
            progress_callback(1.0)

        return analysis

    def process(
        self,
        input_path: Path,
        output_path: Path,
        analysis: Optional[TelecineAnalysis] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """Apply inverse telecine to convert video back to original frame rate.

        Removes telecine pulldown patterns to restore progressive frames
        at the original film frame rate.

        Args:
            input_path: Path to input telecined video
            output_path: Path for output progressive video
            analysis: Pre-computed analysis (will analyze if None)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to output video

        Raises:
            IVTCError: If processing fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise IVTCError(f"Input file not found: {input_path}")

        # Analyze if not provided
        if analysis is None:
            analysis = self.analyze(
                input_path,
                progress_callback=lambda p: progress_callback(p * 0.3) if progress_callback else None
            )

        if progress_callback:
            progress_callback(0.3)

        # Build filter string
        filter_str = self._build_ffmpeg_filter(analysis)

        logger.info(f"Applying IVTC with filter: {filter_str}")

        # Apply IVTC
        self._apply_ivtc(
            input_path,
            output_path,
            filter_str,
            analysis,
            progress_callback=lambda p: progress_callback(0.3 + p * 0.7) if progress_callback else None
        )

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"IVTC complete: {output_path}")

        return output_path

    def _get_video_info(self, video_path: Path) -> Dict:
        """Get video information using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 'v:0',
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            import json
            data = json.loads(result.stdout)

            if not data.get('streams'):
                return {}

            stream = data['streams'][0]

            # Parse frame rate
            fps = 29.97
            if 'r_frame_rate' in stream:
                try:
                    num, den = map(int, stream['r_frame_rate'].split('/'))
                    if den > 0:
                        fps = num / den
                except (ValueError, ZeroDivisionError):
                    pass

            return {
                'fps': fps,
                'width': stream.get('width', 0),
                'height': stream.get('height', 0),
                'codec': stream.get('codec_name', ''),
                'field_order': stream.get('field_order', 'unknown'),
            }

        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return {}

    def _detect_interlacing(
        self,
        video_path: Path,
        sample_duration: float = 30.0
    ) -> bool:
        """Detect if video is interlaced using idet filter.

        Args:
            video_path: Path to video
            sample_duration: Duration to analyze

        Returns:
            True if interlaced content detected
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-t', str(sample_duration),
                '-vf', 'idet',
                '-f', 'null',
                '-'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            output = result.stderr

            # Parse idet output
            # Look for patterns like "TFF:123 BFF:456 Progressive:789"
            tff_match = re.search(r'TFF:\s*(\d+)', output)
            bff_match = re.search(r'BFF:\s*(\d+)', output)
            prog_match = re.search(r'Progressive:\s*(\d+)', output)

            tff = int(tff_match.group(1)) if tff_match else 0
            bff = int(bff_match.group(1)) if bff_match else 0
            prog = int(prog_match.group(1)) if prog_match else 0

            total = tff + bff + prog
            if total == 0:
                return False

            interlaced_ratio = (tff + bff) / total

            logger.debug(
                f"Interlace detection: TFF={tff}, BFF={bff}, "
                f"Progressive={prog}, ratio={interlaced_ratio:.2%}"
            )

            # Consider interlaced if more than 30% of frames are interlaced
            return interlaced_ratio > 0.3

        except Exception as e:
            logger.warning(f"Interlace detection failed: {e}")
            return False

    def _detect_field_order(self, video_path: Path) -> FieldOrder:
        """Detect field order (TFF or BFF) of interlaced content.

        Args:
            video_path: Path to video

        Returns:
            Detected FieldOrder
        """
        # First check metadata
        video_info = self._get_video_info(video_path)
        metadata_order = video_info.get('field_order', '').lower()

        if 'tt' in metadata_order or 'tff' in metadata_order:
            logger.debug("Field order from metadata: TFF")
            return FieldOrder.TFF
        elif 'bb' in metadata_order or 'bff' in metadata_order:
            logger.debug("Field order from metadata: BFF")
            return FieldOrder.BFF
        elif 'progressive' in metadata_order:
            logger.debug("Field order from metadata: Progressive")
            return FieldOrder.PROGRESSIVE

        # Analyze with idet filter
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-t', '10',
                '-vf', 'idet',
                '-f', 'null',
                '-'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stderr

            # Parse TFF/BFF counts
            tff_match = re.search(r'TFF:\s*(\d+)', output)
            bff_match = re.search(r'BFF:\s*(\d+)', output)

            tff = int(tff_match.group(1)) if tff_match else 0
            bff = int(bff_match.group(1)) if bff_match else 0

            if tff > bff:
                logger.debug(f"Field order detected: TFF (TFF={tff}, BFF={bff})")
                return FieldOrder.TFF
            elif bff > tff:
                logger.debug(f"Field order detected: BFF (TFF={tff}, BFF={bff})")
                return FieldOrder.BFF
            else:
                logger.debug("Field order unclear, defaulting to TFF")
                return FieldOrder.TFF

        except Exception as e:
            logger.warning(f"Field order detection failed: {e}, defaulting to TFF")
            return FieldOrder.TFF

    def _detect_pattern(
        self,
        video_path: Path,
        sample_duration: float = 30.0
    ) -> Tuple[TelecinePattern, float, int]:
        """Detect telecine pattern using fieldmatch analysis.

        Args:
            video_path: Path to video
            sample_duration: Duration to analyze

        Returns:
            Tuple of (pattern, confidence, cadence_breaks)
        """
        try:
            # Use fieldmatch in analysis mode
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-t', str(sample_duration),
                '-vf', 'fieldmatch=mode=pc_n:combmatch=full,decimate=cycle=5',
                '-f', 'null',
                '-'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stderr

            # Count decimate decisions
            # Pattern 3:2 should show consistent 4:1 decimation (24fps from 30fps)
            # Pattern 2:2 shows 2:1 (24fps from 48fps or 25fps from 50fps)

            # Look for frame metrics in output
            decimate_count = output.count('drop_count')

            # Get video info to determine likely pattern
            video_info = self._get_video_info(video_path)
            fps = video_info.get('fps', 29.97)

            # Heuristic pattern detection based on frame rate
            if 29.0 <= fps <= 30.0:
                # NTSC territory - likely 3:2 or 2:3
                pattern = TelecinePattern.PATTERN_3_2
                confidence = 0.8
            elif 23.5 <= fps <= 24.5:
                # Already progressive 24fps
                pattern = TelecinePattern.PATTERN_UNKNOWN
                confidence = 0.5
            elif 24.5 < fps <= 26.0:
                # PAL territory - likely 2:2 or already progressive 25fps
                pattern = TelecinePattern.PATTERN_2_2
                confidence = 0.7
            elif 59.0 <= fps <= 60.0:
                # 60fps from 24fps - double 3:2
                pattern = TelecinePattern.PATTERN_3_2
                confidence = 0.7
            else:
                pattern = TelecinePattern.PATTERN_UNKNOWN
                confidence = 0.4

            # Detect cadence breaks by looking for comb detection variations
            comb_matches = re.findall(r'comb=(\d+)', output)
            cadence_breaks = 0

            if comb_matches:
                comb_values = [int(c) for c in comb_matches]
                # High variance in comb values indicates cadence breaks
                if len(comb_values) > 10:
                    avg_comb = sum(comb_values) / len(comb_values)
                    variance = sum((c - avg_comb) ** 2 for c in comb_values) / len(comb_values)
                    if variance > avg_comb * 0.5:
                        cadence_breaks = int(variance / avg_comb)
                        if cadence_breaks > 5:
                            pattern = TelecinePattern.PATTERN_MIXED
                            confidence = max(0.5, confidence - 0.2)

            logger.debug(
                f"Pattern detection: {pattern.value}, "
                f"confidence={confidence:.2f}, breaks={cadence_breaks}"
            )

            return pattern, confidence, cadence_breaks

        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return TelecinePattern.PATTERN_UNKNOWN, 0.3, 0

    def _estimate_source_fps(
        self,
        current_fps: float,
        pattern: TelecinePattern
    ) -> float:
        """Estimate original source FPS based on pattern.

        Args:
            current_fps: Current video frame rate
            pattern: Detected telecine pattern

        Returns:
            Estimated original frame rate
        """
        if pattern == TelecinePattern.PATTERN_3_2:
            # 3:2 pulldown: 24fps -> 29.97fps
            # Inverse: 29.97 * (4/5) = 23.976fps
            return current_fps * (4 / 5)

        elif pattern == TelecinePattern.PATTERN_2_3:
            # Same ratio as 3:2 but different field order
            return current_fps * (4 / 5)

        elif pattern == TelecinePattern.PATTERN_2_2:
            # 2:2 pulldown: 24fps -> 48fps or similar
            # For PAL: 24fps -> 25fps (slight speedup, not true telecine)
            if current_fps > 40:
                return current_fps / 2
            else:
                # PAL speed-up case
                return 24.0

        elif pattern == TelecinePattern.PATTERN_MIXED:
            # Best guess for mixed content
            return 23.976

        else:
            # Unknown - assume it might be 24fps source
            if 29.0 <= current_fps <= 30.0:
                return 23.976
            elif 24.5 <= current_fps <= 26.0:
                return 24.0
            else:
                return current_fps

    def _build_ffmpeg_filter(self, analysis: TelecineAnalysis) -> str:
        """Build ffmpeg filter string for IVTC based on analysis.

        Args:
            analysis: Telecine analysis results

        Returns:
            FFmpeg filter string
        """
        filters = []

        # Determine field order parameter
        if analysis.field_order == FieldOrder.TFF:
            field_param = "tff"
        elif analysis.field_order == FieldOrder.BFF:
            field_param = "bff"
        else:
            field_param = "auto"

        if not analysis.is_interlaced:
            # Progressive content - may still have duplicate frames
            logger.debug("Content is progressive, using frame decimation only")

            if analysis.pattern == TelecinePattern.PATTERN_3_2:
                # Remove 1 in 5 duplicate frames
                filters.append("decimate=cycle=5")
            elif analysis.pattern == TelecinePattern.PATTERN_2_2:
                # Remove 1 in 2 duplicate frames
                filters.append("decimate=cycle=2")

        else:
            # Interlaced content - full IVTC pipeline
            if analysis.pattern in (TelecinePattern.PATTERN_3_2, TelecinePattern.PATTERN_2_3):
                # Standard 3:2 IVTC
                # fieldmatch: Match fields to reconstruct progressive frames
                # decimate: Remove the duplicate frame created by 3:2 pulldown

                fieldmatch_opts = [
                    f"order={field_param}",
                    "combmatch=full",
                    "mode=pc_n" if self.config.preserve_orphan_fields else "mode=pc"
                ]

                filters.append(f"fieldmatch={':'.join(fieldmatch_opts)}")
                filters.append("decimate=cycle=5")

            elif analysis.pattern == TelecinePattern.PATTERN_2_2:
                # 2:2 pulldown - simpler deinterlacing
                # Use yadif for bob deinterlacing
                filters.append(f"yadif=mode=1:parity={'0' if field_param == 'tff' else '1'}")
                filters.append("decimate=cycle=2")

            elif analysis.pattern == TelecinePattern.PATTERN_MIXED:
                # Mixed content - use adaptive approach
                # w3fdif handles varied interlacing better
                filters.append(f"w3fdif=filter=complex:deint=all")

                # Use pullup for mixed pattern detection
                filters.append("pullup")

            else:
                # Unknown pattern - try smart bob deinterlacing
                # yadif with frame-doubling, then let user decimate
                filters.append(f"yadif=mode=0:parity={'0' if field_param == 'tff' else '1'}")

        filter_str = ','.join(filters) if filters else "null"

        return filter_str

    def _apply_ivtc(
        self,
        input_path: Path,
        output_path: Path,
        filter_str: str,
        analysis: TelecineAnalysis,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Apply IVTC using ffmpeg.

        Args:
            input_path: Input video path
            output_path: Output video path
            filter_str: FFmpeg filter string
            analysis: Telecine analysis for FPS calculation
            progress_callback: Optional progress callback

        Raises:
            IVTCError: If ffmpeg processing fails
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine output FPS
        if self.config.output_fps is not None:
            output_fps = self.config.output_fps
        else:
            output_fps = analysis.source_fps

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', str(input_path),
            '-vf', filter_str,
            '-r', f'{output_fps:.3f}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-c:a', 'copy',  # Copy audio
            str(output_path)
        ]

        logger.info(f"Running IVTC: {' '.join(cmd)}")

        try:
            # Get duration for progress tracking
            video_info = self._get_video_info(input_path)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Read stderr for progress
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break

                # Parse progress from ffmpeg output
                if progress_callback and 'time=' in line:
                    time_match = re.search(r'time=(\d+):(\d+):(\d+)', line)
                    if time_match:
                        h, m, s = map(int, time_match.groups())
                        current_time = h * 3600 + m * 60 + s
                        # Estimate progress (rough)
                        progress_callback(min(0.99, current_time / 60))

            # Wait for completion
            returncode = process.wait()

            if returncode != 0:
                stderr = process.stderr.read()
                raise IVTCError(f"ffmpeg failed with code {returncode}: {stderr}")

            # Verify output
            if not output_path.exists():
                raise IVTCError("Output file was not created")

            if output_path.stat().st_size == 0:
                output_path.unlink()
                raise IVTCError("Output file is empty")

            logger.info(
                f"IVTC complete: {input_path.name} -> {output_path.name} "
                f"at {output_fps:.3f}fps"
            )

        except subprocess.SubprocessError as e:
            raise IVTCError(f"ffmpeg process error: {e}") from e


# Factory functions

def analyze_telecine(
    video_path: Path,
    sample_duration: float = 30.0
) -> TelecineAnalysis:
    """Analyze video for telecine patterns.

    Convenience function that creates an InverseTelecine processor
    and analyzes the video.

    Args:
        video_path: Path to video file
        sample_duration: Duration to analyze in seconds

    Returns:
        TelecineAnalysis with detection results

    Example:
        >>> analysis = analyze_telecine(Path("old_movie.avi"))
        >>> print(f"Pattern: {analysis.pattern.value}")
        >>> print(f"Source FPS: {analysis.source_fps:.3f}")
    """
    ivtc = InverseTelecine()
    return ivtc.analyze(video_path, sample_duration)


def apply_ivtc(
    input_path: Path,
    output_path: Path,
    pattern: str = "auto"
) -> Path:
    """Apply inverse telecine to a video.

    Convenience function that handles the complete IVTC workflow:
    1. Analyze video for telecine patterns
    2. Build appropriate filter chain
    3. Process video to remove pulldown

    Args:
        input_path: Path to input telecined video
        output_path: Path for output progressive video
        pattern: Pattern to use ('auto', '3:2', '2:3', '2:2', 'mixed')

    Returns:
        Path to output video

    Example:
        >>> output = apply_ivtc(
        ...     Path("telecined.mpg"),
        ...     Path("progressive.mp4"),
        ...     pattern="auto"
        ... )
    """
    # Map pattern string to enum
    pattern_map = {
        'auto': TelecinePattern.PATTERN_UNKNOWN,
        '3:2': TelecinePattern.PATTERN_3_2,
        '2:3': TelecinePattern.PATTERN_2_3,
        '2:2': TelecinePattern.PATTERN_2_2,
        'mixed': TelecinePattern.PATTERN_MIXED,
    }

    pattern_enum = pattern_map.get(pattern, TelecinePattern.PATTERN_UNKNOWN)

    config = IVTCConfig(pattern=pattern_enum)
    ivtc = InverseTelecine(config)

    return ivtc.process(input_path, output_path)


def create_ivtc_processor(
    pattern: str = "auto",
    field_order: str = "tff",
    preserve_orphan_fields: bool = True,
    output_fps: Optional[float] = None
) -> InverseTelecine:
    """Create an InverseTelecine processor with specified settings.

    Args:
        pattern: Pattern to detect/use ('auto', '3:2', '2:3', '2:2', 'mixed')
        field_order: Field order ('tff', 'bff', 'progressive')
        preserve_orphan_fields: Keep unpaired fields at scene changes
        output_fps: Target output FPS (None for automatic)

    Returns:
        Configured InverseTelecine processor

    Example:
        >>> ivtc = create_ivtc_processor(pattern='3:2', field_order='tff')
        >>> analysis = ivtc.analyze(Path("video.mpg"))
        >>> ivtc.process(Path("video.mpg"), Path("output.mp4"))
    """
    # Map pattern string to enum
    pattern_map = {
        'auto': TelecinePattern.PATTERN_UNKNOWN,
        '3:2': TelecinePattern.PATTERN_3_2,
        '2:3': TelecinePattern.PATTERN_2_3,
        '2:2': TelecinePattern.PATTERN_2_2,
        'mixed': TelecinePattern.PATTERN_MIXED,
    }

    # Map field order string to enum
    field_order_map = {
        'tff': FieldOrder.TFF,
        'bff': FieldOrder.BFF,
        'progressive': FieldOrder.PROGRESSIVE,
    }

    config = IVTCConfig(
        pattern=pattern_map.get(pattern, TelecinePattern.PATTERN_UNKNOWN),
        field_order=field_order_map.get(field_order, FieldOrder.TFF),
        preserve_orphan_fields=preserve_orphan_fields,
        output_fps=output_fps
    )

    return InverseTelecine(config)
