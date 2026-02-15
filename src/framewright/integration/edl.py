"""EDL (Edit Decision List) support for professional workflows.

Supports CMX 3600, Final Cut Pro XML, and DaVinci Resolve EDL formats.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class EDLFormat(Enum):
    """Supported EDL formats."""
    CMX3600 = "cmx3600"
    FCPXML = "fcpxml"
    RESOLVE = "resolve"


class TransitionType(Enum):
    """Edit transition types."""
    CUT = "C"
    DISSOLVE = "D"
    WIPE = "W"
    KEY = "K"


@dataclass
class Timecode:
    """SMPTE timecode representation."""
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    frames: int = 0
    fps: float = 24.0
    drop_frame: bool = False

    @classmethod
    def from_string(cls, tc_str: str, fps: float = 24.0) -> "Timecode":
        """Parse timecode from string (HH:MM:SS:FF or HH;MM;SS;FF)."""
        drop_frame = ";" in tc_str
        parts = re.split(r"[:;]", tc_str)

        if len(parts) != 4:
            raise ValueError(f"Invalid timecode: {tc_str}")

        return cls(
            hours=int(parts[0]),
            minutes=int(parts[1]),
            seconds=int(parts[2]),
            frames=int(parts[3]),
            fps=fps,
            drop_frame=drop_frame,
        )

    @classmethod
    def from_frames(cls, total_frames: int, fps: float = 24.0, drop_frame: bool = False) -> "Timecode":
        """Create timecode from total frame count."""
        if drop_frame and fps in (29.97, 59.94):
            # Drop frame calculation
            frames_per_min = int(fps * 60)
            frames_per_10min = int(fps * 60 * 10) - 18  # 18 frames dropped per 10 min at 29.97

            ten_min_chunks = total_frames // frames_per_10min
            remaining = total_frames % frames_per_10min

            # Add back dropped frames for calculation
            if remaining > 2:
                remaining += 2 * ((remaining - 2) // (frames_per_min - 2))

            total_frames = ten_min_chunks * int(fps * 60 * 10) + remaining

        frames = int(total_frames % fps)
        total_seconds = int(total_frames // fps)
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60

        return cls(hours, minutes, seconds, frames, fps, drop_frame)

    @classmethod
    def from_seconds(cls, seconds: float, fps: float = 24.0) -> "Timecode":
        """Create timecode from seconds."""
        total_frames = int(seconds * fps)
        return cls.from_frames(total_frames, fps)

    def to_frames(self) -> int:
        """Convert to total frame count."""
        base_frames = (
            self.frames +
            self.seconds * int(self.fps) +
            self.minutes * int(self.fps) * 60 +
            self.hours * int(self.fps) * 3600
        )

        if self.drop_frame and self.fps in (29.97, 59.94):
            # Subtract dropped frames
            total_minutes = self.hours * 60 + self.minutes
            drop_count = 2 * (total_minutes - total_minutes // 10)
            base_frames -= drop_count

        return base_frames

    def to_seconds(self) -> float:
        """Convert to seconds."""
        return self.to_frames() / self.fps

    def __str__(self) -> str:
        """Format as string."""
        sep = ";" if self.drop_frame else ":"
        return f"{self.hours:02d}{sep}{self.minutes:02d}{sep}{self.seconds:02d}{sep}{self.frames:02d}"

    def __add__(self, other: "Timecode") -> "Timecode":
        """Add two timecodes."""
        return Timecode.from_frames(
            self.to_frames() + other.to_frames(),
            self.fps,
            self.drop_frame
        )

    def __sub__(self, other: "Timecode") -> "Timecode":
        """Subtract two timecodes."""
        return Timecode.from_frames(
            max(0, self.to_frames() - other.to_frames()),
            self.fps,
            self.drop_frame
        )


@dataclass
class EDLEvent:
    """Single event in an EDL."""
    event_number: int
    reel: str
    track_type: str  # V, A, A2, etc.
    transition: TransitionType
    transition_duration: int = 0  # frames

    source_in: Timecode = field(default_factory=Timecode)
    source_out: Timecode = field(default_factory=Timecode)
    record_in: Timecode = field(default_factory=Timecode)
    record_out: Timecode = field(default_factory=Timecode)

    clip_name: str = ""
    source_file: str = ""
    comments: List[str] = field(default_factory=list)
    motion_effect: Optional[float] = None  # Speed percentage

    # Restoration metadata
    restoration_settings: Dict[str, Any] = field(default_factory=dict)

    def duration_frames(self) -> int:
        """Get duration in frames."""
        return self.record_out.to_frames() - self.record_in.to_frames()

    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_frames() / self.record_in.fps


@dataclass
class EDL:
    """Edit Decision List container."""
    title: str = "Untitled"
    fps: float = 24.0
    drop_frame: bool = False
    format: EDLFormat = EDLFormat.CMX3600
    events: List[EDLEvent] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def total_duration(self) -> Timecode:
        """Get total duration of the EDL."""
        if not self.events:
            return Timecode(fps=self.fps)

        last_event = max(self.events, key=lambda e: e.record_out.to_frames())
        return last_event.record_out

    def get_events_at(self, timecode: Timecode) -> List[EDLEvent]:
        """Get all events active at a given timecode."""
        frame = timecode.to_frames()
        return [
            e for e in self.events
            if e.record_in.to_frames() <= frame < e.record_out.to_frames()
        ]

    def add_event(self, event: EDLEvent) -> None:
        """Add an event to the EDL."""
        if not event.event_number:
            event.event_number = len(self.events) + 1
        self.events.append(event)

    def get_clip_list(self) -> List[str]:
        """Get list of unique clip names/reels."""
        reels = set()
        for event in self.events:
            if event.reel and event.reel not in ("BL", "AX"):
                reels.add(event.reel)
        return sorted(reels)


class EDLParser:
    """Parse various EDL formats."""

    def __init__(self, fps: float = 24.0):
        self.fps = fps

    def parse(self, path: Path) -> EDL:
        """Parse an EDL file, auto-detecting format."""
        content = path.read_text(encoding="utf-8", errors="replace")

        # Detect format
        if content.strip().startswith("<?xml") or content.strip().startswith("<fcpxml"):
            return self._parse_fcpxml(content)
        elif "TITLE:" in content or re.match(r"^\d{3}\s+", content, re.MULTILINE):
            return self._parse_cmx3600(content)
        else:
            raise ValueError(f"Unknown EDL format in {path}")

    def _parse_cmx3600(self, content: str) -> EDL:
        """Parse CMX 3600 format EDL."""
        edl = EDL(format=EDLFormat.CMX3600, fps=self.fps)

        lines = content.split("\n")
        current_event: Optional[EDLEvent] = None

        for line in lines:
            line = line.strip()

            # Title
            if line.startswith("TITLE:"):
                edl.title = line[6:].strip()
                continue

            # FCM (Frame Code Mode)
            if line.startswith("FCM:"):
                fcm = line[4:].strip().upper()
                edl.drop_frame = "DROP" in fcm
                continue

            # Event line pattern: 001  REEL     V     C        00:00:00:00 00:00:10:00 01:00:00:00 01:00:10:00
            event_match = re.match(
                r"^(\d{3})\s+(\S+)\s+([VAB]\d?)\s+([CDWK])(\d*)?\s+"
                r"(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})\s+"
                r"(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})\s+"
                r"(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})\s+"
                r"(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})",
                line
            )

            if event_match:
                if current_event:
                    edl.add_event(current_event)

                groups = event_match.groups()
                transition_duration = int(groups[4]) if groups[4] else 0

                current_event = EDLEvent(
                    event_number=int(groups[0]),
                    reel=groups[1],
                    track_type=groups[2],
                    transition=TransitionType(groups[3]),
                    transition_duration=transition_duration,
                    source_in=Timecode.from_string(groups[5], self.fps),
                    source_out=Timecode.from_string(groups[6], self.fps),
                    record_in=Timecode.from_string(groups[7], self.fps),
                    record_out=Timecode.from_string(groups[8], self.fps),
                )
                continue

            # Comment lines
            if line.startswith("*") and current_event:
                comment = line[1:].strip()

                # Parse special comments
                if comment.startswith("FROM CLIP NAME:"):
                    current_event.clip_name = comment[15:].strip()
                elif comment.startswith("SOURCE FILE:"):
                    current_event.source_file = comment[12:].strip()
                elif comment.startswith("RESTORATION:"):
                    # Parse restoration settings JSON
                    try:
                        import json
                        settings_str = comment[12:].strip()
                        current_event.restoration_settings = json.loads(settings_str)
                    except Exception:
                        pass
                else:
                    current_event.comments.append(comment)
                continue

            # Motion effect
            if line.startswith("M2") and current_event:
                m2_match = re.match(r"M2\s+\S+\s+([\d.]+)", line)
                if m2_match:
                    current_event.motion_effect = float(m2_match.group(1))

        # Add last event
        if current_event:
            edl.add_event(current_event)

        return edl

    def _parse_fcpxml(self, content: str) -> EDL:
        """Parse Final Cut Pro XML format."""
        edl = EDL(format=EDLFormat.FCPXML, fps=self.fps)

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid FCPXML: {e}")

        # Find project/sequence
        for sequence in root.iter("sequence"):
            name = sequence.get("name", "Untitled")
            edl.title = name

            # Parse format
            format_elem = sequence.find(".//format")
            if format_elem is not None:
                fps_str = format_elem.get("frameDuration", "1/24s")
                # Parse "1001/30000s" or "1/24s" format
                if "/" in fps_str:
                    parts = fps_str.rstrip("s").split("/")
                    if len(parts) == 2:
                        num, denom = int(parts[0]), int(parts[1])
                        self.fps = denom / num
                        edl.fps = self.fps

            # Parse spine/clips
            event_num = 0
            for clip in sequence.iter("asset-clip"):
                event_num += 1

                # Parse timing
                offset = self._parse_fcpxml_time(clip.get("offset", "0s"))
                duration = self._parse_fcpxml_time(clip.get("duration", "0s"))
                start = self._parse_fcpxml_time(clip.get("start", "0s"))

                event = EDLEvent(
                    event_number=event_num,
                    reel=clip.get("ref", "AX"),
                    track_type="V",
                    transition=TransitionType.CUT,
                    source_in=Timecode.from_seconds(start, self.fps),
                    source_out=Timecode.from_seconds(start + duration, self.fps),
                    record_in=Timecode.from_seconds(offset, self.fps),
                    record_out=Timecode.from_seconds(offset + duration, self.fps),
                    clip_name=clip.get("name", ""),
                )

                edl.add_event(event)

        return edl

    def _parse_fcpxml_time(self, time_str: str) -> float:
        """Parse FCPXML time format (e.g., '3600/2400s' or '10s')."""
        if not time_str:
            return 0.0

        time_str = time_str.rstrip("s")

        if "/" in time_str:
            parts = time_str.split("/")
            if len(parts) == 2:
                return int(parts[0]) / int(parts[1])
        else:
            return float(time_str)

        return 0.0


class EDLWriter:
    """Write EDL files in various formats."""

    def __init__(self, fps: float = 24.0):
        self.fps = fps

    def write(self, edl: EDL, path: Path, format: Optional[EDLFormat] = None) -> None:
        """Write EDL to file."""
        format = format or edl.format

        if format == EDLFormat.CMX3600:
            content = self._write_cmx3600(edl)
        elif format == EDLFormat.FCPXML:
            content = self._write_fcpxml(edl)
        else:
            raise ValueError(f"Unsupported format: {format}")

        path.write_text(content, encoding="utf-8")
        logger.info(f"Written EDL to {path}")

    def _write_cmx3600(self, edl: EDL) -> str:
        """Write CMX 3600 format."""
        lines = []

        lines.append(f"TITLE: {edl.title}")
        lines.append(f"FCM: {'DROP FRAME' if edl.drop_frame else 'NON-DROP FRAME'}")
        lines.append("")

        for event in edl.events:
            # Main event line
            trans = event.transition.value
            trans_dur = f"{event.transition_duration:03d}" if event.transition_duration else ""

            line = (
                f"{event.event_number:03d}  "
                f"{event.reel:<8} "
                f"{event.track_type:<6} "
                f"{trans}{trans_dur:<4} "
                f"{event.source_in} {event.source_out} "
                f"{event.record_in} {event.record_out}"
            )
            lines.append(line)

            # Comments
            if event.clip_name:
                lines.append(f"* FROM CLIP NAME: {event.clip_name}")

            if event.source_file:
                lines.append(f"* SOURCE FILE: {event.source_file}")

            if event.restoration_settings:
                import json
                settings_json = json.dumps(event.restoration_settings)
                lines.append(f"* RESTORATION: {settings_json}")

            for comment in event.comments:
                lines.append(f"* {comment}")

            # Motion effect
            if event.motion_effect:
                lines.append(f"M2   {event.reel}       {event.motion_effect:.1f}")

            lines.append("")

        return "\n".join(lines)

    def _write_fcpxml(self, edl: EDL) -> str:
        """Write Final Cut Pro XML format."""
        root = ET.Element("fcpxml", version="1.10")

        # Resources
        resources = ET.SubElement(root, "resources")

        # Format
        fps_num = 1
        fps_denom = int(edl.fps)
        if edl.fps == 29.97:
            fps_num, fps_denom = 1001, 30000
        elif edl.fps == 23.976:
            fps_num, fps_denom = 1001, 24000

        format_elem = ET.SubElement(
            resources, "format",
            id="r1",
            frameDuration=f"{fps_num}/{fps_denom}s",
            width="1920",
            height="1080"
        )

        # Assets
        reels = set()
        for event in edl.events:
            if event.reel not in reels and event.reel not in ("BL", "AX"):
                reels.add(event.reel)
                ET.SubElement(
                    resources, "asset",
                    id=event.reel,
                    name=event.clip_name or event.reel,
                    src=event.source_file or ""
                )

        # Library/Event/Project structure
        library = ET.SubElement(root, "library")
        fcp_event = ET.SubElement(library, "event", name=edl.title)
        project = ET.SubElement(fcp_event, "project", name=edl.title)
        sequence = ET.SubElement(project, "sequence", format="r1")
        spine = ET.SubElement(sequence, "spine")

        # Clips
        for event in edl.events:
            offset_seconds = event.record_in.to_seconds()
            duration_seconds = event.duration_seconds()
            start_seconds = event.source_in.to_seconds()

            clip = ET.SubElement(
                spine, "asset-clip",
                ref=event.reel,
                name=event.clip_name or event.reel,
                offset=f"{offset_seconds:.6f}s",
                duration=f"{duration_seconds:.6f}s",
                start=f"{start_seconds:.6f}s"
            )

        return ET.tostring(root, encoding="unicode")


def create_restoration_edl(
    video_path: Path,
    segments: List[Dict[str, Any]],
    fps: float = 24.0
) -> EDL:
    """Create an EDL with restoration settings for each segment."""
    edl = EDL(
        title=f"Restoration - {video_path.stem}",
        fps=fps,
    )

    for i, segment in enumerate(segments):
        event = EDLEvent(
            event_number=i + 1,
            reel=video_path.stem[:8].upper(),
            track_type="V",
            transition=TransitionType.CUT,
            source_in=Timecode.from_seconds(segment.get("start", 0), fps),
            source_out=Timecode.from_seconds(segment.get("end", 0), fps),
            record_in=Timecode.from_seconds(segment.get("start", 0), fps),
            record_out=Timecode.from_seconds(segment.get("end", 0), fps),
            clip_name=segment.get("name", f"Segment {i + 1}"),
            restoration_settings=segment.get("settings", {}),
        )
        edl.add_event(event)

    return edl
