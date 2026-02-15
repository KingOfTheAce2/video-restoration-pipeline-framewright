"""Export presets system for FrameWright.

Provides predefined and customizable export profiles for various
platforms and use cases with optimized encoding settings.
"""

import logging
import json
import subprocess
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ExportPlatform(Enum):
    """Target platforms for export."""
    YOUTUBE = "youtube"
    VIMEO = "vimeo"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    ARCHIVE = "archive"
    BROADCAST = "broadcast"
    CINEMA = "cinema"
    WEB = "web"
    MOBILE = "mobile"
    CUSTOM = "custom"


class VideoCodec(Enum):
    """Video codecs."""
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"
    PRORES = "prores"
    DNXHD = "dnxhd"
    CINEFORM = "cineform"


class AudioCodec(Enum):
    """Audio codecs."""
    AAC = "aac"
    AC3 = "ac3"
    EAC3 = "eac3"
    OPUS = "opus"
    FLAC = "flac"
    PCM = "pcm_s24le"
    MP3 = "mp3"


class Container(Enum):
    """Container formats."""
    MP4 = "mp4"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    AVI = "avi"
    MXF = "mxf"


class ColorSpace(Enum):
    """Color spaces."""
    REC709 = "bt709"
    REC2020 = "bt2020nc"
    DCI_P3 = "p3"
    SRGB = "srgb"


class HDRMode(Enum):
    """HDR modes."""
    SDR = "sdr"
    HDR10 = "hdr10"
    HDR10_PLUS = "hdr10plus"
    DOLBY_VISION = "dolbyvision"
    HLG = "hlg"


@dataclass
class VideoSettings:
    """Video encoding settings."""
    codec: VideoCodec = VideoCodec.H264
    width: Optional[int] = None  # None = preserve
    height: Optional[int] = None
    fps: Optional[float] = None  # None = preserve
    bitrate: Optional[str] = None  # e.g., "20M"
    crf: Optional[int] = None
    preset: str = "slow"  # encoding speed preset
    profile: str = "high"
    level: Optional[str] = None
    pixel_format: str = "yuv420p"
    color_space: ColorSpace = ColorSpace.REC709
    hdr_mode: HDRMode = HDRMode.SDR
    max_bitrate: Optional[str] = None
    bufsize: Optional[str] = None
    keyint: int = 48
    bframes: int = 3
    ref_frames: int = 4
    tune: Optional[str] = None  # film, animation, grain, etc.
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSettings:
    """Audio encoding settings."""
    codec: AudioCodec = AudioCodec.AAC
    bitrate: str = "192k"
    sample_rate: int = 48000
    channels: int = 2
    normalize: bool = True
    loudness_target: float = -14.0  # LUFS
    true_peak: float = -1.0  # dBTP
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportPreset:
    """Complete export preset configuration."""
    name: str
    description: str = ""
    platform: ExportPlatform = ExportPlatform.CUSTOM
    container: Container = Container.MP4
    video: VideoSettings = field(default_factory=VideoSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    two_pass: bool = False
    faststart: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["platform"] = self.platform.value
        data["container"] = self.container.value
        data["video"]["codec"] = self.video.codec.value
        data["video"]["color_space"] = self.video.color_space.value
        data["video"]["hdr_mode"] = self.video.hdr_mode.value
        data["audio"]["codec"] = self.audio.codec.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportPreset":
        """Create from dictionary."""
        data = data.copy()

        # Convert enums
        data["platform"] = ExportPlatform(data.get("platform", "custom"))
        data["container"] = Container(data.get("container", "mp4"))

        video_data = data.get("video", {})
        video_data["codec"] = VideoCodec(video_data.get("codec", "h264"))
        video_data["color_space"] = ColorSpace(video_data.get("color_space", "bt709"))
        video_data["hdr_mode"] = HDRMode(video_data.get("hdr_mode", "sdr"))
        data["video"] = VideoSettings(**video_data)

        audio_data = data.get("audio", {})
        audio_data["codec"] = AudioCodec(audio_data.get("codec", "aac"))
        data["audio"] = AudioSettings(**audio_data)

        return cls(**data)


# Built-in presets
BUILTIN_PRESETS: Dict[str, ExportPreset] = {
    # YouTube presets
    "youtube_4k": ExportPreset(
        name="YouTube 4K",
        description="Optimized for YouTube 4K HDR uploads",
        platform=ExportPlatform.YOUTUBE,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=3840,
            height=2160,
            crf=18,
            preset="slow",
            profile="high",
            level="5.1",
            keyint=48,
            tune="film",
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="320k",
            sample_rate=48000,
        ),
        two_pass=True,
        faststart=True,
        tags=["youtube", "4k", "social"],
    ),

    "youtube_1080p": ExportPreset(
        name="YouTube 1080p",
        description="Standard YouTube HD upload",
        platform=ExportPlatform.YOUTUBE,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=1920,
            height=1080,
            crf=20,
            preset="slow",
            profile="high",
            level="4.1",
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="256k",
        ),
        faststart=True,
        tags=["youtube", "1080p", "social"],
    ),

    # Vimeo presets
    "vimeo_4k": ExportPreset(
        name="Vimeo 4K",
        description="High quality Vimeo 4K",
        platform=ExportPlatform.VIMEO,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=3840,
            height=2160,
            crf=16,
            preset="slow",
            profile="high",
            level="5.2",
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="320k",
        ),
        two_pass=True,
        tags=["vimeo", "4k", "professional"],
    ),

    # Social media presets
    "instagram_reel": ExportPreset(
        name="Instagram Reel",
        description="Vertical video for Instagram Reels",
        platform=ExportPlatform.INSTAGRAM,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=1080,
            height=1920,
            crf=20,
            preset="medium",
            fps=30,
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="192k",
        ),
        faststart=True,
        tags=["instagram", "vertical", "social"],
    ),

    "tiktok": ExportPreset(
        name="TikTok",
        description="Optimized for TikTok",
        platform=ExportPlatform.TIKTOK,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=1080,
            height=1920,
            crf=20,
            preset="medium",
            fps=30,
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="192k",
        ),
        faststart=True,
        tags=["tiktok", "vertical", "social"],
    ),

    "twitter": ExportPreset(
        name="Twitter/X",
        description="Optimized for Twitter video",
        platform=ExportPlatform.TWITTER,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=1280,
            height=720,
            bitrate="5M",
            max_bitrate="8M",
            preset="medium",
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="128k",
        ),
        faststart=True,
        tags=["twitter", "social"],
    ),

    # Archive presets
    "archive_master": ExportPreset(
        name="Archive Master",
        description="Maximum quality archive preservation",
        platform=ExportPlatform.ARCHIVE,
        container=Container.MKV,
        video=VideoSettings(
            codec=VideoCodec.H265,
            crf=14,
            preset="slow",
            profile="main10",
            pixel_format="yuv420p10le",
        ),
        audio=AudioSettings(
            codec=AudioCodec.FLAC,
            bitrate="0",  # Lossless
            sample_rate=48000,
        ),
        tags=["archive", "master", "lossless"],
    ),

    "archive_prores": ExportPreset(
        name="ProRes Archive",
        description="ProRes 422 HQ for professional archive",
        platform=ExportPlatform.ARCHIVE,
        container=Container.MOV,
        video=VideoSettings(
            codec=VideoCodec.PRORES,
            pixel_format="yuv422p10le",
            extra_params={"profile": 3},  # ProRes 422 HQ
        ),
        audio=AudioSettings(
            codec=AudioCodec.PCM,
            sample_rate=48000,
        ),
        tags=["archive", "prores", "professional"],
    ),

    # Broadcast presets
    "broadcast_hd": ExportPreset(
        name="Broadcast HD",
        description="EBU broadcast compliant HD",
        platform=ExportPlatform.BROADCAST,
        container=Container.MXF,
        video=VideoSettings(
            codec=VideoCodec.DNXHD,
            width=1920,
            height=1080,
            fps=25.0,
            pixel_format="yuv422p",
        ),
        audio=AudioSettings(
            codec=AudioCodec.PCM,
            sample_rate=48000,
            loudness_target=-23.0,  # EBU R128
        ),
        tags=["broadcast", "ebu", "professional"],
    ),

    # Cinema presets
    "cinema_dcp": ExportPreset(
        name="Cinema 2K",
        description="Digital Cinema Package preparation",
        platform=ExportPlatform.CINEMA,
        container=Container.MXF,
        video=VideoSettings(
            codec=VideoCodec.H264,  # Would be JPEG2000 for real DCP
            width=2048,
            height=1080,
            fps=24.0,
            color_space=ColorSpace.DCI_P3,
            pixel_format="xyz12le",
        ),
        audio=AudioSettings(
            codec=AudioCodec.PCM,
            sample_rate=48000,
            channels=6,  # 5.1
        ),
        tags=["cinema", "dcp", "theatrical"],
    ),

    # Web optimized
    "web_optimized": ExportPreset(
        name="Web Optimized",
        description="Fast-loading web video",
        platform=ExportPlatform.WEB,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=1280,
            height=720,
            crf=23,
            preset="faster",
            profile="main",
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="128k",
        ),
        faststart=True,
        tags=["web", "streaming"],
    ),

    "web_av1": ExportPreset(
        name="Web AV1",
        description="Modern AV1 codec for web",
        platform=ExportPlatform.WEB,
        container=Container.WEBM,
        video=VideoSettings(
            codec=VideoCodec.AV1,
            crf=30,
            preset="5",  # AV1 speed
        ),
        audio=AudioSettings(
            codec=AudioCodec.OPUS,
            bitrate="128k",
        ),
        tags=["web", "av1", "modern"],
    ),

    # Mobile
    "mobile_optimized": ExportPreset(
        name="Mobile Optimized",
        description="Optimized for mobile playback",
        platform=ExportPlatform.MOBILE,
        container=Container.MP4,
        video=VideoSettings(
            codec=VideoCodec.H264,
            width=1280,
            height=720,
            crf=23,
            preset="fast",
            profile="baseline",
            level="3.1",
        ),
        audio=AudioSettings(
            codec=AudioCodec.AAC,
            bitrate="128k",
            sample_rate=44100,
        ),
        faststart=True,
        tags=["mobile", "compatibility"],
    ),
}


class ExportPresetManager:
    """Manages export presets including custom user presets."""

    def __init__(self, user_presets_path: Optional[Path] = None):
        """Initialize preset manager.

        Args:
            user_presets_path: Path to user presets JSON file
        """
        self.user_presets_path = user_presets_path
        self._user_presets: Dict[str, ExportPreset] = {}

        if user_presets_path and user_presets_path.exists():
            self._load_user_presets()

    def _load_user_presets(self) -> None:
        """Load user presets from file."""
        try:
            with open(self.user_presets_path) as f:
                data = json.load(f)

            for name, preset_data in data.items():
                self._user_presets[name] = ExportPreset.from_dict(preset_data)

            logger.info(f"Loaded {len(self._user_presets)} user presets")

        except Exception as e:
            logger.error(f"Failed to load user presets: {e}")

    def _save_user_presets(self) -> None:
        """Save user presets to file."""
        if not self.user_presets_path:
            return

        try:
            self.user_presets_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                name: preset.to_dict()
                for name, preset in self._user_presets.items()
            }

            with open(self.user_presets_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save user presets: {e}")

    def get_preset(self, name: str) -> Optional[ExportPreset]:
        """Get preset by name.

        Args:
            name: Preset name

        Returns:
            ExportPreset or None
        """
        # Check user presets first
        if name in self._user_presets:
            return self._user_presets[name]

        return BUILTIN_PRESETS.get(name)

    def list_presets(
        self,
        platform: Optional[ExportPlatform] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ExportPreset]:
        """List available presets.

        Args:
            platform: Filter by platform
            tags: Filter by tags (any match)

        Returns:
            List of matching presets
        """
        all_presets = {**BUILTIN_PRESETS, **self._user_presets}
        presets = list(all_presets.values())

        if platform:
            presets = [p for p in presets if p.platform == platform]

        if tags:
            presets = [
                p for p in presets
                if any(t in p.tags for t in tags)
            ]

        return presets

    def create_custom_preset(
        self,
        name: str,
        base_preset: Optional[str] = None,
        **overrides,
    ) -> ExportPreset:
        """Create a custom preset.

        Args:
            name: Name for new preset
            base_preset: Optional preset to base on
            **overrides: Settings to override

        Returns:
            Created ExportPreset
        """
        if base_preset:
            base = self.get_preset(base_preset)
            if base:
                preset_dict = base.to_dict()
            else:
                preset_dict = {}
        else:
            preset_dict = {}

        preset_dict["name"] = name
        preset_dict["platform"] = ExportPlatform.CUSTOM.value

        # Apply overrides
        for key, value in overrides.items():
            if key.startswith("video_"):
                preset_dict.setdefault("video", {})[key[6:]] = value
            elif key.startswith("audio_"):
                preset_dict.setdefault("audio", {})[key[6:]] = value
            else:
                preset_dict[key] = value

        preset = ExportPreset.from_dict(preset_dict)
        self._user_presets[name] = preset
        self._save_user_presets()

        return preset

    def delete_custom_preset(self, name: str) -> bool:
        """Delete a custom preset.

        Args:
            name: Preset name

        Returns:
            True if deleted
        """
        if name in self._user_presets:
            del self._user_presets[name]
            self._save_user_presets()
            return True
        return False


class VideoExporter:
    """Exports video using specified presets."""

    def __init__(
        self,
        preset: Optional[ExportPreset] = None,
        preset_name: Optional[str] = None,
    ):
        """Initialize exporter.

        Args:
            preset: ExportPreset to use
            preset_name: Name of builtin preset
        """
        if preset:
            self.preset = preset
        elif preset_name:
            self.preset = BUILTIN_PRESETS.get(preset_name)
            if not self.preset:
                raise ValueError(f"Unknown preset: {preset_name}")
        else:
            self.preset = BUILTIN_PRESETS["youtube_1080p"]

    def export(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Export video using preset settings.

        Args:
            input_path: Input video path
            output_path: Output path (auto-generated if None)
            progress_callback: Progress callback

        Returns:
            Path to exported file
        """
        input_path = Path(input_path)

        if output_path is None:
            ext = self._get_extension()
            output_path = input_path.parent / f"{input_path.stem}_{self.preset.name.lower().replace(' ', '_')}{ext}"
        output_path = Path(output_path)

        logger.info(f"Exporting with preset: {self.preset.name}")

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(input_path, output_path)

        if self.preset.two_pass:
            self._run_two_pass(cmd, input_path, output_path, progress_callback)
        else:
            self._run_single_pass(cmd, progress_callback)

        logger.info(f"Export complete: {output_path}")
        return output_path

    def _get_extension(self) -> str:
        """Get file extension for container."""
        ext_map = {
            Container.MP4: ".mp4",
            Container.MOV: ".mov",
            Container.MKV: ".mkv",
            Container.WEBM: ".webm",
            Container.AVI: ".avi",
            Container.MXF: ".mxf",
        }
        return ext_map.get(self.preset.container, ".mp4")

    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
    ) -> List[str]:
        """Build FFmpeg command from preset."""
        cmd = ["ffmpeg", "-y", "-i", str(input_path)]

        # Video settings
        video = self.preset.video
        cmd.extend(self._build_video_options(video))

        # Audio settings
        audio = self.preset.audio
        cmd.extend(self._build_audio_options(audio))

        # Container options
        if self.preset.faststart and self.preset.container == Container.MP4:
            cmd.extend(["-movflags", "+faststart"])

        # Metadata
        for key, value in self.preset.metadata.items():
            cmd.extend(["-metadata", f"{key}={value}"])

        cmd.append(str(output_path))
        return cmd

    def _build_video_options(self, video: VideoSettings) -> List[str]:
        """Build video encoding options."""
        opts = []

        # Codec
        codec_map = {
            VideoCodec.H264: "libx264",
            VideoCodec.H265: "libx265",
            VideoCodec.VP9: "libvpx-vp9",
            VideoCodec.AV1: "libaom-av1",
            VideoCodec.PRORES: "prores_ks",
            VideoCodec.DNXHD: "dnxhd",
        }
        opts.extend(["-c:v", codec_map.get(video.codec, "libx264")])

        # Resolution
        if video.width and video.height:
            opts.extend(["-s", f"{video.width}x{video.height}"])
        elif video.width:
            opts.extend(["-vf", f"scale={video.width}:-2"])
        elif video.height:
            opts.extend(["-vf", f"scale=-2:{video.height}"])

        # Frame rate
        if video.fps:
            opts.extend(["-r", str(video.fps)])

        # Quality
        if video.crf is not None:
            opts.extend(["-crf", str(video.crf)])
        elif video.bitrate:
            opts.extend(["-b:v", video.bitrate])

        # Preset
        if video.preset:
            opts.extend(["-preset", video.preset])

        # Profile and level
        if video.profile:
            opts.extend(["-profile:v", video.profile])
        if video.level:
            opts.extend(["-level", video.level])

        # Pixel format
        if video.pixel_format:
            opts.extend(["-pix_fmt", video.pixel_format])

        # GOP settings
        opts.extend(["-g", str(video.keyint)])
        if video.bframes:
            opts.extend(["-bf", str(video.bframes)])

        # Tune
        if video.tune:
            opts.extend(["-tune", video.tune])

        # Rate control
        if video.max_bitrate:
            opts.extend(["-maxrate", video.max_bitrate])
        if video.bufsize:
            opts.extend(["-bufsize", video.bufsize])

        return opts

    def _build_audio_options(self, audio: AudioSettings) -> List[str]:
        """Build audio encoding options."""
        opts = []

        # Codec
        codec_map = {
            AudioCodec.AAC: "aac",
            AudioCodec.AC3: "ac3",
            AudioCodec.EAC3: "eac3",
            AudioCodec.OPUS: "libopus",
            AudioCodec.FLAC: "flac",
            AudioCodec.PCM: "pcm_s24le",
            AudioCodec.MP3: "libmp3lame",
        }
        opts.extend(["-c:a", codec_map.get(audio.codec, "aac")])

        # Bitrate (not for lossless)
        if audio.codec not in [AudioCodec.FLAC, AudioCodec.PCM] and audio.bitrate:
            opts.extend(["-b:a", audio.bitrate])

        # Sample rate
        opts.extend(["-ar", str(audio.sample_rate)])

        # Channels
        opts.extend(["-ac", str(audio.channels)])

        # Loudness normalization
        if audio.normalize:
            loudnorm = f"loudnorm=I={audio.loudness_target}:TP={audio.true_peak}:LRA=11"
            opts.extend(["-af", loudnorm])

        return opts

    def _run_single_pass(
        self,
        cmd: List[str],
        progress_callback: Optional[callable],
    ) -> None:
        """Run single-pass encode."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Export failed: {e.stderr}")
            raise RuntimeError(f"Export failed: {e.stderr}")

    def _run_two_pass(
        self,
        cmd: List[str],
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[callable],
    ) -> None:
        """Run two-pass encode."""
        # Pass 1
        pass1_cmd = cmd.copy()
        pass1_cmd.extend(["-pass", "1", "-f", "null"])
        # Remove output path, use null
        pass1_cmd = pass1_cmd[:-1] + ["-"]

        logger.info("Running pass 1/2...")
        subprocess.run(pass1_cmd, capture_output=True, check=True)

        # Pass 2
        pass2_cmd = cmd.copy()
        pass2_cmd.extend(["-pass", "2"])

        logger.info("Running pass 2/2...")
        subprocess.run(pass2_cmd, capture_output=True, check=True)


def export_video(
    input_path: Path,
    preset: str = "youtube_1080p",
    output_path: Optional[Path] = None,
    **overrides,
) -> Path:
    """Quick export function.

    Args:
        input_path: Input video path
        preset: Preset name
        output_path: Output path
        **overrides: Settings to override

    Returns:
        Path to exported file
    """
    manager = ExportPresetManager()

    if overrides:
        export_preset = manager.create_custom_preset(
            f"temp_{preset}",
            base_preset=preset,
            **overrides,
        )
    else:
        export_preset = manager.get_preset(preset)

    if not export_preset:
        raise ValueError(f"Unknown preset: {preset}")

    exporter = VideoExporter(preset=export_preset)
    return exporter.export(input_path, output_path)


def list_presets(platform: Optional[str] = None) -> List[str]:
    """List available preset names.

    Args:
        platform: Filter by platform name

    Returns:
        List of preset names
    """
    manager = ExportPresetManager()

    if platform:
        try:
            plat = ExportPlatform(platform)
            presets = manager.list_presets(platform=plat)
        except ValueError:
            presets = manager.list_presets()
    else:
        presets = manager.list_presets()

    return [p.name for p in presets]
