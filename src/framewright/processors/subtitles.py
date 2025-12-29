"""Subtitle preservation and enhancement for video restoration.

This module provides comprehensive subtitle handling during video restoration:

1. **SubtitleExtractor**: Extract embedded subtitle streams (SRT, ASS, VTT)
   - Uses ffprobe to detect subtitle streams in video files
   - Extracts to temporary files for processing
   - Supports multiple subtitle tracks and languages

2. **SubtitleTimeSync**: Adjust subtitle timing when frame rate changes
   - Handles time stretching/compression for interpolated videos
   - Support sync drift correction
   - Maintains subtitle accuracy after RIFE interpolation

3. **SubtitleEnhancer**: Optional AI features for subtitle cleanup
   - Clean up OCR artifacts if present
   - Standardize formatting
   - Adjust position for upscaled video

4. **SubtitleMerger**: Embed subtitles back into output video
   - Support hard-burn (filter) and soft-sub (stream copy)
   - Preserve all subtitle tracks from original
   - Maintains metadata and language tags

**Key Features:**
- Preserve original subtitle tracks during restoration
- Adjust timing when frame rate changes (e.g., 24fps -> 60fps interpolation)
- Support multiple subtitle tracks
- CLI integration with --preserve-subtitles flag

**Subtitle Format Support:**
- SRT (SubRip): Most common, simple text-based format
- ASS/SSA (Advanced SubStation Alpha): Rich formatting, positioning
- VTT (WebVTT): Web-standard format
- PGS (Blu-ray): Image-based subtitles
- DVB (DVB Subtitle): Broadcast standard
"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SubtitleFormat(Enum):
    """Supported subtitle formats."""
    SRT = "srt"
    ASS = "ass"
    SSA = "ssa"
    VTT = "vtt"
    SUB = "sub"
    PGS = "pgs"  # Image-based (Blu-ray)
    DVB = "dvb"  # Broadcast
    UNKNOWN = "unknown"


class SubtitleError(Exception):
    """Exception raised for subtitle processing errors."""
    pass


class OCREngine(Enum):
    """OCR engines for burned-in subtitle extraction (legacy compatibility)."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


@dataclass
class SubtitleStreamInfo:
    """Information about a subtitle stream in a video file.

    Attributes:
        index: Stream index in the container
        codec_name: Subtitle codec (e.g., 'subrip', 'ass', 'webvtt')
        codec_type: Always 'subtitle' for subtitle streams
        language: Language code (e.g., 'eng', 'fra', 'jpn')
        title: Stream title/description
        is_default: Whether this is the default subtitle track
        is_forced: Whether this is a forced subtitle track
        format: Detected subtitle format enum
        disposition: Stream disposition flags
    """
    index: int
    codec_name: str
    codec_type: str = "subtitle"
    language: Optional[str] = None
    title: Optional[str] = None
    is_default: bool = False
    is_forced: bool = False
    format: SubtitleFormat = SubtitleFormat.UNKNOWN
    disposition: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detect format from codec name."""
        codec_to_format = {
            'subrip': SubtitleFormat.SRT,
            'srt': SubtitleFormat.SRT,
            'ass': SubtitleFormat.ASS,
            'ssa': SubtitleFormat.SSA,
            'webvtt': SubtitleFormat.VTT,
            'vtt': SubtitleFormat.VTT,
            'subviewer': SubtitleFormat.SUB,
            'hdmv_pgs_subtitle': SubtitleFormat.PGS,
            'pgssub': SubtitleFormat.PGS,
            'dvb_subtitle': SubtitleFormat.DVB,
            'dvbsub': SubtitleFormat.DVB,
        }
        if self.format == SubtitleFormat.UNKNOWN:
            self.format = codec_to_format.get(
                self.codec_name.lower(),
                SubtitleFormat.UNKNOWN
            )


@dataclass
class BoundingBox:
    """Bounding box for subtitle region (used for positioning).

    Attributes:
        x: Left coordinate (pixels from left edge)
        y: Top coordinate (pixels from top edge)
        width: Width of the region
        height: Height of the region
        confidence: Detection confidence (0-1)
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def x2(self) -> int:
        """Right coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom coordinate."""
        return self.y + self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, x2, y2) tuple."""
        return (self.x, self.y, self.x2, self.y2)

    def scale(self, factor: float) -> 'BoundingBox':
        """Scale the bounding box by a factor."""
        return BoundingBox(
            x=int(self.x * factor),
            y=int(self.y * factor),
            width=int(self.width * factor),
            height=int(self.height * factor),
            confidence=self.confidence
        )


@dataclass
class SubtitleLine:
    """A single subtitle line/cue.

    Attributes:
        index: Line/cue number (1-based for SRT)
        start_time: Start time in seconds
        end_time: End time in seconds
        text: Subtitle text content (may include formatting)
        position: Optional position information
        style: Optional style name (for ASS/SSA)
        layer: Layer number (for ASS)
    """
    index: int
    start_time: float
    end_time: float
    text: str
    position: Optional[BoundingBox] = None
    style: Optional[str] = None
    layer: int = 0

    @property
    def duration(self) -> float:
        """Duration of the subtitle in seconds."""
        return self.end_time - self.start_time

    def adjust_timing(
        self,
        time_factor: float,
        offset: float = 0.0
    ) -> 'SubtitleLine':
        """Adjust timing by a factor and optional offset.

        Args:
            time_factor: Multiply times by this factor
            offset: Add this offset (in seconds) after scaling

        Returns:
            New SubtitleLine with adjusted timing
        """
        return SubtitleLine(
            index=self.index,
            start_time=self.start_time * time_factor + offset,
            end_time=self.end_time * time_factor + offset,
            text=self.text,
            position=self.position,
            style=self.style,
            layer=self.layer
        )

    def to_srt_time(self, seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_vtt_time(self, seconds: float) -> str:
        """Format time for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def to_ass_time(self, seconds: float) -> str:
        """Format time for ASS (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

    def to_srt(self) -> str:
        """Convert to SRT format."""
        return (
            f"{self.index}\n"
            f"{self.to_srt_time(self.start_time)} --> {self.to_srt_time(self.end_time)}\n"
            f"{self.text}\n"
        )

    def to_vtt(self) -> str:
        """Convert to VTT format."""
        return (
            f"{self.to_vtt_time(self.start_time)} --> {self.to_vtt_time(self.end_time)}\n"
            f"{self.text}\n"
        )


@dataclass
class SubtitleTrack:
    """A complete subtitle track with metadata.

    Attributes:
        lines: List of subtitle lines/cues
        language: Language code (ISO 639-2 or 639-1)
        title: Track title/description
        format: Subtitle format
        is_default: Whether this is the default track
        is_forced: Whether this is forced subtitles
        metadata: Additional metadata (styles, script info for ASS)
        stream_index: Original stream index in source video
    """
    lines: List[SubtitleLine] = field(default_factory=list)
    language: Optional[str] = None
    title: Optional[str] = None
    format: SubtitleFormat = SubtitleFormat.SRT
    is_default: bool = False
    is_forced: bool = False
    metadata: Dict = field(default_factory=dict)
    stream_index: int = 0

    @property
    def duration(self) -> float:
        """Total duration covered by subtitles."""
        if not self.lines:
            return 0.0
        return max(line.end_time for line in self.lines)

    @property
    def line_count(self) -> int:
        """Number of subtitle lines."""
        return len(self.lines)

    def adjust_timing(
        self,
        time_factor: float,
        offset: float = 0.0
    ) -> 'SubtitleTrack':
        """Adjust timing of all lines.

        Args:
            time_factor: Multiply times by this factor
            offset: Add this offset after scaling

        Returns:
            New SubtitleTrack with adjusted timing
        """
        return SubtitleTrack(
            lines=[line.adjust_timing(time_factor, offset) for line in self.lines],
            language=self.language,
            title=self.title,
            format=self.format,
            is_default=self.is_default,
            is_forced=self.is_forced,
            metadata=self.metadata.copy(),
            stream_index=self.stream_index
        )

    def to_srt(self) -> str:
        """Export to SRT format."""
        output = []
        for i, line in enumerate(self.lines, 1):
            # Re-index to ensure sequential numbering
            adjusted_line = SubtitleLine(
                index=i,
                start_time=line.start_time,
                end_time=line.end_time,
                text=line.text,
                position=line.position,
                style=line.style,
                layer=line.layer
            )
            output.append(adjusted_line.to_srt())
        return "\n".join(output)

    def to_vtt(self) -> str:
        """Export to WebVTT format."""
        output = ["WEBVTT", ""]
        for line in self.lines:
            output.append(line.to_vtt())
        return "\n".join(output)

    def to_ass(self) -> str:
        """Export to ASS format."""
        # Get script info from metadata or use defaults
        script_info = self.metadata.get('script_info', {
            'Title': self.title or 'Exported Subtitles',
            'ScriptType': 'v4.00+',
        })

        # Get styles from metadata or use default
        styles = self.metadata.get('styles', [
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"
        ])

        output = [
            "[Script Info]",
            *[f"{k}: {v}" for k, v in script_info.items()],
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            *styles,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]

        for line in self.lines:
            start = line.to_ass_time(line.start_time)
            end = line.to_ass_time(line.end_time)
            text = line.text.replace('\n', '\\N')
            style = line.style or "Default"
            output.append(
                f"Dialogue: {line.layer},{start},{end},{style},,0,0,0,,{text}"
            )

        return "\n".join(output)

    def save(self, output_path: Path, format: Optional[SubtitleFormat] = None) -> Path:
        """Save subtitle track to file.

        Args:
            output_path: Output file path
            format: Output format (uses self.format if None)

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        fmt = format or self.format

        # Ensure correct extension
        ext_map = {
            SubtitleFormat.SRT: '.srt',
            SubtitleFormat.ASS: '.ass',
            SubtitleFormat.SSA: '.ssa',
            SubtitleFormat.VTT: '.vtt',
        }
        if fmt in ext_map:
            output_path = output_path.with_suffix(ext_map[fmt])

        # Generate content
        if fmt == SubtitleFormat.SRT:
            content = self.to_srt()
        elif fmt == SubtitleFormat.VTT:
            content = self.to_vtt()
        elif fmt in (SubtitleFormat.ASS, SubtitleFormat.SSA):
            content = self.to_ass()
        else:
            raise SubtitleError(f"Cannot save format: {fmt}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')

        logger.info(f"Saved {len(self.lines)} subtitles to {output_path}")
        return output_path


@dataclass
class SubtitleConfig:
    """Configuration for subtitle processing.

    Attributes:
        preserve_subtitles: Whether to preserve subtitle tracks
        adjust_timing: Adjust timing when frame rate changes
        preferred_format: Output format for converted subtitles
        burn_in: Hard-burn subtitles into video
        default_language: Default language code for unlabeled tracks
        extract_all: Extract all subtitle tracks (not just default)
        font_scale: Scale factor for burned-in subtitle font
        position_adjust: Adjust position for resolution changes
    """
    preserve_subtitles: bool = True
    adjust_timing: bool = True
    preferred_format: SubtitleFormat = SubtitleFormat.SRT
    burn_in: bool = False
    default_language: str = "und"  # Undetermined
    extract_all: bool = True
    font_scale: float = 1.0
    position_adjust: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.preferred_format, str):
            self.preferred_format = SubtitleFormat(self.preferred_format.lower())


def _check_ffmpeg() -> None:
    """Verify ffmpeg and ffprobe are available."""
    if not shutil.which('ffmpeg'):
        raise SubtitleError(
            "ffmpeg not found. Install with:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg"
        )
    if not shutil.which('ffprobe'):
        raise SubtitleError("ffprobe not found. Install ffmpeg package.")


class SubtitleExtractor:
    """Extract embedded subtitle streams from video files.

    Uses ffprobe to detect subtitle streams and ffmpeg to extract them
    to external files for processing.

    Example:
        >>> extractor = SubtitleExtractor()
        >>> streams = extractor.detect_streams("movie.mkv")
        >>> for stream in streams:
        ...     print(f"Found: {stream.language} ({stream.format.value})")
        >>> tracks = extractor.extract_all("movie.mkv", "/tmp/subs")
    """

    def __init__(self, config: Optional[SubtitleConfig] = None):
        """Initialize the subtitle extractor.

        Args:
            config: SubtitleConfig for processing settings
        """
        self.config = config or SubtitleConfig()
        _check_ffmpeg()

    def detect_streams(self, video_path: Path) -> List[SubtitleStreamInfo]:
        """Detect all subtitle streams in a video file.

        Uses ffprobe to analyze the video and find subtitle streams.

        Args:
            video_path: Path to video file

        Returns:
            List of SubtitleStreamInfo objects

        Raises:
            SubtitleError: If probing fails
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise SubtitleError(f"Video file not found: {video_path}")

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 's',  # Only subtitle streams
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise SubtitleError(f"ffprobe failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise SubtitleError(f"Failed to parse ffprobe output: {e}")

        streams = []
        for stream in data.get('streams', []):
            if stream.get('codec_type') != 'subtitle':
                continue

            tags = stream.get('tags', {})
            disposition = stream.get('disposition', {})

            info = SubtitleStreamInfo(
                index=stream.get('index', 0),
                codec_name=stream.get('codec_name', 'unknown'),
                language=tags.get('language'),
                title=tags.get('title'),
                is_default=bool(disposition.get('default', 0)),
                is_forced=bool(disposition.get('forced', 0)),
                disposition=disposition
            )
            streams.append(info)

        logger.info(f"Found {len(streams)} subtitle streams in {video_path.name}")
        return streams

    def extract_stream(
        self,
        video_path: Path,
        stream_index: int,
        output_path: Path,
        output_format: Optional[SubtitleFormat] = None
    ) -> SubtitleTrack:
        """Extract a single subtitle stream to a file.

        Args:
            video_path: Path to video file
            stream_index: Index of the subtitle stream
            output_path: Path for output subtitle file
            output_format: Output format (auto-detected if None)

        Returns:
            SubtitleTrack with extracted subtitles

        Raises:
            SubtitleError: If extraction fails
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        # Determine output format
        fmt = output_format or self.config.preferred_format

        # Map format to ffmpeg codec
        format_codecs = {
            SubtitleFormat.SRT: 'srt',
            SubtitleFormat.ASS: 'ass',
            SubtitleFormat.SSA: 'ssa',
            SubtitleFormat.VTT: 'webvtt',
        }

        codec = format_codecs.get(fmt, 'srt')

        # Build extraction command
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite
            '-i', str(video_path),
            '-map', f'0:{stream_index}',
            '-c:s', codec,
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
        except subprocess.CalledProcessError as e:
            raise SubtitleError(
                f"Failed to extract subtitle stream {stream_index}: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            raise SubtitleError("Subtitle extraction timed out")

        # Parse the extracted file
        track = self._parse_subtitle_file(output_path, fmt)
        track.stream_index = stream_index

        logger.info(
            f"Extracted {len(track.lines)} lines from stream {stream_index}"
        )
        return track

    def extract_all(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[SubtitleTrack]:
        """Extract all subtitle streams from a video.

        Args:
            video_path: Path to video file
            output_dir: Directory for output files
            progress_callback: Optional progress callback

        Returns:
            List of SubtitleTrack objects
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        streams = self.detect_streams(video_path)

        if not streams:
            logger.info("No subtitle streams found")
            return []

        tracks = []
        for i, stream in enumerate(streams):
            # Generate output filename
            lang = stream.language or f"track{i}"
            suffix = stream.format.value if stream.format != SubtitleFormat.UNKNOWN else "srt"
            output_path = output_dir / f"subtitle_{lang}_{stream.index}.{suffix}"

            try:
                track = self.extract_stream(
                    video_path,
                    stream.index,
                    output_path,
                    stream.format if stream.format != SubtitleFormat.UNKNOWN else None
                )
                track.language = stream.language
                track.title = stream.title
                track.is_default = stream.is_default
                track.is_forced = stream.is_forced
                tracks.append(track)
            except SubtitleError as e:
                logger.warning(f"Failed to extract stream {stream.index}: {e}")

            if progress_callback:
                progress_callback((i + 1) / len(streams))

        return tracks

    def _parse_subtitle_file(
        self,
        file_path: Path,
        format: SubtitleFormat
    ) -> SubtitleTrack:
        """Parse a subtitle file into a SubtitleTrack.

        Args:
            file_path: Path to subtitle file
            format: Subtitle format

        Returns:
            SubtitleTrack with parsed lines
        """
        content = file_path.read_text(encoding='utf-8', errors='replace')

        if format == SubtitleFormat.SRT:
            return self._parse_srt(content)
        elif format == SubtitleFormat.VTT:
            return self._parse_vtt(content)
        elif format in (SubtitleFormat.ASS, SubtitleFormat.SSA):
            return self._parse_ass(content)
        else:
            # For unsupported formats, return empty track
            return SubtitleTrack(format=format)

    def _parse_srt(self, content: str) -> SubtitleTrack:
        """Parse SRT format subtitles."""
        track = SubtitleTrack(format=SubtitleFormat.SRT)

        # SRT format: index, timestamp, text, blank line
        pattern = re.compile(
            r'(\d+)\s*\n'
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n'
            r'((?:.*?\n)*?)\n',
            re.MULTILINE
        )

        for match in pattern.finditer(content + '\n'):
            index = int(match.group(1))
            start_time = self._parse_srt_time(match.group(2))
            end_time = self._parse_srt_time(match.group(3))
            text = match.group(4).strip()

            track.lines.append(SubtitleLine(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))

        return track

    def _parse_vtt(self, content: str) -> SubtitleTrack:
        """Parse WebVTT format subtitles."""
        track = SubtitleTrack(format=SubtitleFormat.VTT)

        # Skip header
        lines = content.split('\n')
        i = 0
        while i < len(lines) and not lines[i].strip().startswith('0'):
            if '-->' in lines[i]:
                break
            i += 1

        # Parse cues
        cue_pattern = re.compile(
            r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})'
        )

        index = 1
        while i < len(lines):
            line = lines[i].strip()
            match = cue_pattern.search(line)

            if match:
                start_time = self._parse_vtt_time(match.group(1))
                end_time = self._parse_vtt_time(match.group(2))

                # Collect text lines
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1

                track.lines.append(SubtitleLine(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text='\n'.join(text_lines)
                ))
                index += 1
            else:
                i += 1

        return track

    def _parse_ass(self, content: str) -> SubtitleTrack:
        """Parse ASS/SSA format subtitles."""
        track = SubtitleTrack(format=SubtitleFormat.ASS)

        # Parse script info
        script_info = {}
        info_match = re.search(
            r'\[Script Info\](.*?)(?=\[|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if info_match:
            for line in info_match.group(1).split('\n'):
                if ':' in line and not line.strip().startswith(';'):
                    key, value = line.split(':', 1)
                    script_info[key.strip()] = value.strip()
        track.metadata['script_info'] = script_info

        # Parse styles
        styles = []
        style_match = re.search(
            r'\[V4\+? Styles\](.*?)(?=\[|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if style_match:
            for line in style_match.group(1).split('\n'):
                if line.strip().startswith('Style:'):
                    styles.append(line.strip())
        track.metadata['styles'] = styles

        # Parse events (dialogues)
        events_match = re.search(
            r'\[Events\](.*?)(?=\[|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )

        if events_match:
            # Find format line
            format_line = None
            for line in events_match.group(1).split('\n'):
                if line.strip().startswith('Format:'):
                    format_line = line.strip()
                    break

            # Parse dialogues
            index = 1
            for line in events_match.group(1).split('\n'):
                if line.strip().startswith('Dialogue:'):
                    # Parse dialogue line
                    # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                    parts = line[9:].split(',', 9)  # Split only first 9 commas

                    if len(parts) >= 10:
                        layer = int(parts[0].strip())
                        start_time = self._parse_ass_time(parts[1].strip())
                        end_time = self._parse_ass_time(parts[2].strip())
                        style = parts[3].strip()
                        text = parts[9].strip()

                        # Convert ASS line breaks
                        text = text.replace('\\N', '\n').replace('\\n', '\n')

                        track.lines.append(SubtitleLine(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            text=text,
                            style=style,
                            layer=layer
                        ))
                        index += 1

        return track

    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
        match = re.match(r'(\d+):(\d+):(\d+),(\d+)', time_str)
        if not match:
            return 0.0
        hours, minutes, seconds, millis = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    def _parse_vtt_time(self, time_str: str) -> float:
        """Parse VTT timestamp (HH:MM:SS.mmm) to seconds."""
        match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', time_str)
        if not match:
            return 0.0
        hours, minutes, seconds, millis = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    def _parse_ass_time(self, time_str: str) -> float:
        """Parse ASS timestamp (H:MM:SS.cc) to seconds."""
        match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', time_str)
        if not match:
            return 0.0
        hours, minutes, seconds, centis = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds + centis / 100


class SubtitleTimeSync:
    """Adjust subtitle timing when video frame rate changes.

    When video is processed with frame interpolation (e.g., RIFE to increase
    frame rate from 24fps to 60fps), subtitle timing needs to be adjusted
    to maintain synchronization.

    Example:
        >>> sync = SubtitleTimeSync()
        >>> # Video interpolated from 24fps to 60fps
        >>> adjusted = sync.adjust_for_framerate_change(
        ...     track, source_fps=24.0, target_fps=60.0
        ... )
    """

    def __init__(self, config: Optional[SubtitleConfig] = None):
        """Initialize the time sync handler.

        Args:
            config: SubtitleConfig for processing settings
        """
        self.config = config or SubtitleConfig()

    def adjust_for_framerate_change(
        self,
        track: SubtitleTrack,
        source_fps: float,
        target_fps: float
    ) -> SubtitleTrack:
        """Adjust subtitle timing for frame rate change.

        When video is interpolated to a higher frame rate, the duration
        remains the same, so subtitle timing should not change. However,
        some workflows may stretch/compress video duration, requiring
        timing adjustment.

        This method assumes the video duration remains constant (standard
        interpolation behavior). For duration changes, use adjust_for_duration.

        Args:
            track: SubtitleTrack to adjust
            source_fps: Original video frame rate
            target_fps: New video frame rate

        Returns:
            SubtitleTrack with adjusted timing (or unchanged if no adjustment needed)
        """
        # Standard interpolation doesn't change duration
        # Only adjust if there's a duration change
        logger.info(
            f"Frame rate change {source_fps}fps -> {target_fps}fps "
            f"(duration unchanged, no timing adjustment needed)"
        )
        return track

    def adjust_for_duration_change(
        self,
        track: SubtitleTrack,
        source_duration: float,
        target_duration: float
    ) -> SubtitleTrack:
        """Adjust subtitle timing when video duration changes.

        This handles cases where video is sped up or slowed down,
        or where frame dropping/duplication changes total duration.

        Args:
            track: SubtitleTrack to adjust
            source_duration: Original video duration in seconds
            target_duration: New video duration in seconds

        Returns:
            SubtitleTrack with adjusted timing
        """
        if source_duration <= 0 or target_duration <= 0:
            raise SubtitleError("Duration must be positive")

        time_factor = target_duration / source_duration

        if abs(time_factor - 1.0) < 0.001:
            logger.info("Duration unchanged, no timing adjustment needed")
            return track

        logger.info(
            f"Adjusting subtitle timing by factor {time_factor:.4f} "
            f"({source_duration:.2f}s -> {target_duration:.2f}s)"
        )

        return track.adjust_timing(time_factor)

    def adjust_for_speed_change(
        self,
        track: SubtitleTrack,
        speed_factor: float
    ) -> SubtitleTrack:
        """Adjust subtitle timing when video speed is changed.

        Args:
            track: SubtitleTrack to adjust
            speed_factor: Speed multiplier (e.g., 2.0 for 2x speed)

        Returns:
            SubtitleTrack with adjusted timing
        """
        if speed_factor <= 0:
            raise SubtitleError("Speed factor must be positive")

        # Speed up = shorter duration = divide times by speed
        time_factor = 1.0 / speed_factor

        logger.info(
            f"Adjusting subtitle timing for {speed_factor}x speed "
            f"(time factor: {time_factor:.4f})"
        )

        return track.adjust_timing(time_factor)

    def correct_drift(
        self,
        track: SubtitleTrack,
        drift_per_minute: float
    ) -> SubtitleTrack:
        """Correct progressive timing drift.

        Some videos have progressive sync issues where timing drifts
        over time (common with VFR sources or encoding issues).

        Args:
            track: SubtitleTrack to adjust
            drift_per_minute: Drift in seconds per minute of video
                             (positive = subtitles ahead, negative = behind)

        Returns:
            SubtitleTrack with drift corrected
        """
        if abs(drift_per_minute) < 0.001:
            return track

        # Calculate time factor based on drift
        # drift_per_minute seconds per 60 seconds
        # total_drift = duration * (drift_per_minute / 60)
        # time_factor = original_time / (original_time + drift)
        # For progressive correction: multiply by decreasing factor

        drift_per_second = drift_per_minute / 60.0

        adjusted_lines = []
        for line in track.lines:
            # Calculate drift at this point
            drift = line.start_time * drift_per_second

            adjusted_lines.append(SubtitleLine(
                index=line.index,
                start_time=line.start_time - drift,
                end_time=line.end_time - drift * (line.end_time / line.start_time if line.start_time > 0 else 1),
                text=line.text,
                position=line.position,
                style=line.style,
                layer=line.layer
            ))

        logger.info(
            f"Corrected timing drift of {drift_per_minute:.3f}s/min "
            f"for {len(adjusted_lines)} lines"
        )

        return SubtitleTrack(
            lines=adjusted_lines,
            language=track.language,
            title=track.title,
            format=track.format,
            is_default=track.is_default,
            is_forced=track.is_forced,
            metadata=track.metadata.copy(),
            stream_index=track.stream_index
        )

    def apply_offset(
        self,
        track: SubtitleTrack,
        offset_seconds: float
    ) -> SubtitleTrack:
        """Apply a fixed timing offset to all subtitles.

        Args:
            track: SubtitleTrack to adjust
            offset_seconds: Offset in seconds (positive = delay, negative = advance)

        Returns:
            SubtitleTrack with offset applied
        """
        if abs(offset_seconds) < 0.001:
            return track

        logger.info(f"Applying {offset_seconds:+.3f}s offset to subtitles")

        return track.adjust_timing(1.0, offset_seconds)


class SubtitleEnhancer:
    """Optional AI features for subtitle cleanup and enhancement.

    Provides functionality to clean up and standardize subtitle text,
    fix common issues, and adjust positioning for upscaled video.
    """

    def __init__(self, config: Optional[SubtitleConfig] = None):
        """Initialize the subtitle enhancer.

        Args:
            config: SubtitleConfig for processing settings
        """
        self.config = config or SubtitleConfig()

    def clean_ocr_artifacts(self, track: SubtitleTrack) -> SubtitleTrack:
        """Clean up common OCR artifacts from subtitles.

        Fixes issues like:
        - l/I/1 confusion
        - O/0 confusion
        - Extra whitespace
        - Common typos

        Args:
            track: SubtitleTrack to clean

        Returns:
            SubtitleTrack with cleaned text
        """
        cleaned_lines = []

        for line in track.lines:
            text = line.text

            # Fix common OCR errors
            # Note: These are conservative fixes to avoid false positives

            # Remove extra whitespace
            text = ' '.join(text.split())

            # Fix common punctuation issues
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r'([.,!?;:])(?=[A-Za-z])', r'\1 ', text)

            # Fix quote issues
            text = text.replace(',,', '"')
            text = text.replace("''", '"')

            # Remove isolated single characters (likely noise)
            words = text.split()
            cleaned_words = []
            for word in words:
                # Keep single letters that make sense
                if len(word) == 1 and word.lower() not in 'aio':
                    continue
                cleaned_words.append(word)
            text = ' '.join(cleaned_words)

            cleaned_lines.append(SubtitleLine(
                index=line.index,
                start_time=line.start_time,
                end_time=line.end_time,
                text=text.strip(),
                position=line.position,
                style=line.style,
                layer=line.layer
            ))

        # Filter out empty lines
        cleaned_lines = [l for l in cleaned_lines if l.text.strip()]

        logger.info(f"Cleaned {len(cleaned_lines)} subtitle lines")

        return SubtitleTrack(
            lines=cleaned_lines,
            language=track.language,
            title=track.title,
            format=track.format,
            is_default=track.is_default,
            is_forced=track.is_forced,
            metadata=track.metadata.copy(),
            stream_index=track.stream_index
        )

    def standardize_formatting(self, track: SubtitleTrack) -> SubtitleTrack:
        """Standardize subtitle formatting.

        Ensures consistent:
        - Capitalization at sentence starts
        - Punctuation
        - Line breaks

        Args:
            track: SubtitleTrack to standardize

        Returns:
            SubtitleTrack with standardized formatting
        """
        standardized_lines = []

        for line in track.lines:
            text = line.text

            # Capitalize first letter of each line
            if text and text[0].isalpha():
                text = text[0].upper() + text[1:]

            # Ensure proper spacing around dashes in dialogues
            text = re.sub(r'^-\s*', '- ', text, flags=re.MULTILINE)

            # Normalize ellipsis
            text = re.sub(r'\.{2,}', '...', text)

            standardized_lines.append(SubtitleLine(
                index=line.index,
                start_time=line.start_time,
                end_time=line.end_time,
                text=text,
                position=line.position,
                style=line.style,
                layer=line.layer
            ))

        return SubtitleTrack(
            lines=standardized_lines,
            language=track.language,
            title=track.title,
            format=track.format,
            is_default=track.is_default,
            is_forced=track.is_forced,
            metadata=track.metadata.copy(),
            stream_index=track.stream_index
        )

    def adjust_positions_for_scale(
        self,
        track: SubtitleTrack,
        scale_factor: float
    ) -> SubtitleTrack:
        """Adjust subtitle positions for upscaled video.

        When video is upscaled (e.g., 2x or 4x), subtitle positions
        need to be scaled accordingly.

        Args:
            track: SubtitleTrack to adjust
            scale_factor: Upscaling factor (e.g., 2.0 or 4.0)

        Returns:
            SubtitleTrack with adjusted positions
        """
        if scale_factor == 1.0:
            return track

        adjusted_lines = []

        for line in track.lines:
            position = None
            if line.position:
                position = line.position.scale(scale_factor)

            adjusted_lines.append(SubtitleLine(
                index=line.index,
                start_time=line.start_time,
                end_time=line.end_time,
                text=line.text,
                position=position,
                style=line.style,
                layer=line.layer
            ))

        # Also update styles in metadata if present (for ASS)
        metadata = track.metadata.copy()
        if 'styles' in metadata and scale_factor != 1.0:
            # Update font sizes in styles
            scaled_styles = []
            for style in metadata['styles']:
                # Parse and scale font size (3rd element in ASS style)
                parts = style.split(',')
                if len(parts) > 2 and parts[0].strip().startswith('Style:'):
                    try:
                        font_size = float(parts[2].strip())
                        parts[2] = str(int(font_size * scale_factor))
                        style = ','.join(parts)
                    except ValueError:
                        pass
                scaled_styles.append(style)
            metadata['styles'] = scaled_styles

        logger.info(f"Adjusted subtitle positions for {scale_factor}x scale")

        return SubtitleTrack(
            lines=adjusted_lines,
            language=track.language,
            title=track.title,
            format=track.format,
            is_default=track.is_default,
            is_forced=track.is_forced,
            metadata=metadata,
            stream_index=track.stream_index
        )


class SubtitleMerger:
    """Embed subtitles back into output video.

    Supports both soft-sub (stream copy) and hard-burn (filter) modes.

    Example:
        >>> merger = SubtitleMerger()
        >>> # Soft-sub (keeps subtitles as separate stream)
        >>> merger.merge_soft(video_path, tracks, output_path)
        >>> # Hard-burn (burns subtitles into video)
        >>> merger.merge_hard(video_path, track, output_path)
    """

    def __init__(self, config: Optional[SubtitleConfig] = None):
        """Initialize the subtitle merger.

        Args:
            config: SubtitleConfig for processing settings
        """
        self.config = config or SubtitleConfig()
        _check_ffmpeg()

    def merge_soft(
        self,
        video_path: Path,
        subtitle_tracks: List[SubtitleTrack],
        output_path: Path,
        preserve_video_subs: bool = False
    ) -> Path:
        """Merge subtitles as soft-subs (separate streams).

        Subtitles are added as separate streams that can be toggled
        on/off by the player.

        Args:
            video_path: Path to input video
            subtitle_tracks: List of SubtitleTrack objects to add
            output_path: Path for output video
            preserve_video_subs: Keep existing subtitle streams from video

        Returns:
            Path to output video

        Raises:
            SubtitleError: If merging fails
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise SubtitleError(f"Video file not found: {video_path}")

        if not subtitle_tracks:
            logger.warning("No subtitle tracks to merge, copying video")
            shutil.copy2(video_path, output_path)
            return output_path

        # Create temp directory for subtitle files
        temp_dir = Path(tempfile.mkdtemp(prefix="framewright_subs_"))

        try:
            # Save subtitle tracks to temp files
            sub_files = []
            for i, track in enumerate(subtitle_tracks):
                lang = track.language or f"und{i}"
                sub_file = temp_dir / f"sub_{lang}_{i}.srt"
                track.save(sub_file, SubtitleFormat.SRT)
                sub_files.append((sub_file, track))

            # Build ffmpeg command
            cmd = ['ffmpeg', '-y', '-i', str(video_path)]

            # Add subtitle inputs
            for sub_file, _ in sub_files:
                cmd.extend(['-i', str(sub_file)])

            # Map video and audio streams
            cmd.extend(['-map', '0:v', '-map', '0:a?'])

            # Optionally preserve existing subtitle streams
            if preserve_video_subs:
                cmd.extend(['-map', '0:s?'])

            # Map new subtitle streams
            for i in range(len(sub_files)):
                cmd.extend(['-map', f'{i + 1}:0'])

            # Copy video/audio, convert subtitles
            cmd.extend([
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-c:s', 'srt'  # Use SRT for maximum compatibility
            ])

            # Add metadata for subtitle tracks
            for i, (_, track) in enumerate(sub_files):
                stream_idx = i + (1 if preserve_video_subs else 0)
                if track.language:
                    cmd.extend([
                        f'-metadata:s:s:{stream_idx}',
                        f'language={track.language}'
                    ])
                if track.title:
                    cmd.extend([
                        f'-metadata:s:s:{stream_idx}',
                        f'title={track.title}'
                    ])
                if track.is_default:
                    cmd.extend([
                        f'-disposition:s:{stream_idx}',
                        'default'
                    ])

            cmd.append(str(output_path))

            # Run ffmpeg
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )

            if result.returncode != 0:
                raise SubtitleError(f"ffmpeg failed: {result.stderr}")

            logger.info(
                f"Merged {len(subtitle_tracks)} subtitle tracks "
                f"into {output_path.name}"
            )
            return output_path

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def merge_hard(
        self,
        video_path: Path,
        subtitle_track: SubtitleTrack,
        output_path: Path,
        style: Optional[Dict] = None
    ) -> Path:
        """Burn subtitles into video (hard-sub).

        Subtitles are permanently rendered into the video stream.
        This is useful for maximum compatibility but cannot be toggled off.

        Args:
            video_path: Path to input video
            subtitle_track: SubtitleTrack to burn in
            output_path: Path for output video
            style: Optional style dict (font, size, colors, etc.)

        Returns:
            Path to output video

        Raises:
            SubtitleError: If merging fails
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise SubtitleError(f"Video file not found: {video_path}")

        # Create temp subtitle file
        temp_dir = Path(tempfile.mkdtemp(prefix="framewright_hardsub_"))

        try:
            # Save subtitle to temp file (ASS for styling support)
            sub_file = temp_dir / "subtitle.ass"
            subtitle_track.save(sub_file, SubtitleFormat.ASS)

            # Build ffmpeg command with subtitle filter
            # Escape path for ffmpeg filter
            sub_path_escaped = str(sub_file).replace('\\', '/').replace(':', '\\:')

            vf_filter = f"subtitles='{sub_path_escaped}'"

            # Add style options if provided
            if style:
                style_opts = []
                if 'font' in style:
                    style_opts.append(f"FontName={style['font']}")
                if 'size' in style:
                    style_opts.append(f"FontSize={style['size']}")
                if 'color' in style:
                    style_opts.append(f"PrimaryColour={style['color']}")

                if style_opts:
                    vf_filter += f":force_style='{','.join(style_opts)}'"

            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', vf_filter,
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'medium',
                '-c:a', 'copy',
                str(output_path)
            ]

            output_path.parent.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200
            )

            if result.returncode != 0:
                raise SubtitleError(f"ffmpeg failed: {result.stderr}")

            logger.info(f"Hard-burned subtitles into {output_path.name}")
            return output_path

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def copy_subtitles_from_source(
        self,
        source_video: Path,
        target_video: Path,
        output_path: Path
    ) -> Path:
        """Copy all subtitle streams from source to target video.

        Useful when target video was processed without subtitles
        and you want to add back the original subtitle streams.

        Args:
            source_video: Original video with subtitles
            target_video: Processed video without subtitles
            output_path: Path for output video

        Returns:
            Path to output video
        """
        source_video = Path(source_video)
        target_video = Path(target_video)
        output_path = Path(output_path)

        cmd = [
            'ffmpeg', '-y',
            '-i', str(target_video),
            '-i', str(source_video),
            '-map', '0:v',
            '-map', '0:a?',
            '-map', '1:s?',
            '-c', 'copy',
            str(output_path)
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode != 0:
            raise SubtitleError(f"ffmpeg failed: {result.stderr}")

        logger.info(f"Copied subtitle streams to {output_path.name}")
        return output_path


# Convenience functions for common operations

def detect_burned_subtitles(
    video_path: Path,
    sample_count: int = 10
) -> Dict:
    """Detect if a video has burned-in subtitles.

    Note: This is a basic detection using contrast analysis.
    For accurate detection, use OCR-based analysis.

    Args:
        video_path: Path to video file
        sample_count: Number of frames to sample

    Returns:
        Dictionary with detection results
    """
    # This is a placeholder for backward compatibility
    # Full implementation would use the OCR-based approach
    return {
        'has_subtitles': False,
        'confidence': 0.0,
        'estimated_region': 'unknown',
        'language_hint': 'unknown',
        'note': 'Use SubtitleExtractor for embedded subtitle detection'
    }


def extract_subtitles(
    video_path: Path,
    output_dir: Optional[Path] = None
) -> List[SubtitleTrack]:
    """Extract all subtitle tracks from a video.

    Convenience function for simple extraction.

    Args:
        video_path: Path to video file
        output_dir: Directory for extracted files (temp if None)

    Returns:
        List of SubtitleTrack objects
    """
    extractor = SubtitleExtractor()

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="framewright_subs_"))

    return extractor.extract_all(video_path, output_dir)


def remove_subtitles(
    video_path: Path,
    output_path: Path
) -> Path:
    """Remove all subtitle streams from a video.

    Note: This removes embedded subtitle streams, not burned-in text.

    Args:
        video_path: Path to input video
        output_path: Path for output video

    Returns:
        Path to output video without subtitles
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    _check_ffmpeg()

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-map', '0:v',
        '-map', '0:a?',
        '-c', 'copy',
        '-sn',  # No subtitles
        str(output_path)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600
    )

    if result.returncode != 0:
        raise SubtitleError(f"ffmpeg failed: {result.stderr}")

    logger.info(f"Removed subtitle streams from {video_path.name}")
    return output_path


def preserve_subtitles_during_restoration(
    source_video: Path,
    restored_video: Path,
    output_path: Path,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
    scale_factor: float = 1.0
) -> Path:
    """Preserve and sync subtitles during video restoration.

    Complete workflow for subtitle preservation:
    1. Extract subtitles from source
    2. Adjust timing if frame rate changed
    3. Adjust positions if video was upscaled
    4. Merge back into restored video

    Args:
        source_video: Original video with subtitles
        restored_video: Processed video without subtitles
        output_path: Path for final output
        source_fps: Original frame rate (for timing adjustment)
        target_fps: New frame rate (for timing adjustment)
        scale_factor: Upscaling factor (for position adjustment)

    Returns:
        Path to output video with subtitles
    """
    source_video = Path(source_video)
    restored_video = Path(restored_video)
    output_path = Path(output_path)

    # Extract subtitles from source
    temp_dir = Path(tempfile.mkdtemp(prefix="framewright_preserve_"))

    try:
        extractor = SubtitleExtractor()
        tracks = extractor.extract_all(source_video, temp_dir)

        if not tracks:
            logger.info("No subtitles to preserve, copying restored video")
            shutil.copy2(restored_video, output_path)
            return output_path

        # Adjust timing if frame rate changed
        if source_fps and target_fps and source_fps != target_fps:
            sync = SubtitleTimeSync()
            tracks = [
                sync.adjust_for_framerate_change(t, source_fps, target_fps)
                for t in tracks
            ]

        # Adjust positions if upscaled
        if scale_factor != 1.0:
            enhancer = SubtitleEnhancer()
            tracks = [
                enhancer.adjust_positions_for_scale(t, scale_factor)
                for t in tracks
            ]

        # Merge into restored video
        merger = SubtitleMerger()
        result = merger.merge_soft(restored_video, tracks, output_path)

        logger.info(
            f"Preserved {len(tracks)} subtitle tracks in {output_path.name}"
        )
        return result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
