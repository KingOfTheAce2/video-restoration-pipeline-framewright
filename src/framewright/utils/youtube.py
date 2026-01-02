"""YouTube video downloading utilities for FrameWright.

This module provides comprehensive YouTube video downloading and metadata extraction
using yt-dlp as the backend. Features include quality selection, progress tracking,
playlist support, and metadata preservation.

Features:
    - Download videos with quality selection (4K, 1080p, 720p, etc.)
    - Extract video metadata without downloading
    - Support for playlists with batch downloading
    - Progress tracking with speed and ETA
    - Metadata preservation (title, description, chapters)
    - Handle age-restricted content with cookies
    - Codec preference (vp9/av1 > h264)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict
from datetime import datetime, date
import json
import logging
import platform
import subprocess
import shutil
import re

logger = logging.getLogger(__name__)


def _find_ytdlp_binary() -> Optional[str]:
    """Find yt-dlp binary in PATH or common locations.

    Returns:
        Path to yt-dlp binary or None if not found
    """
    # Check PATH first
    path = shutil.which("yt-dlp")
    if path:
        return path

    # Check common installation directories
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    home = Path.home()

    search_paths = [
        # Python user Scripts (pip install --user)
        home / "AppData" / "Roaming" / "Python" / "Python313" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Roaming" / "Python" / "Python311" / "Scripts" / f"yt-dlp{exe_suffix}",
        home / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "Scripts" / f"yt-dlp{exe_suffix}",
        # System Python
        Path("C:/Python313/Scripts") / f"yt-dlp{exe_suffix}",
        Path("C:/Python312/Scripts") / f"yt-dlp{exe_suffix}",
        # Unix locations
        home / ".local" / "bin" / "yt-dlp",
        Path("/usr/local/bin/yt-dlp"),
    ]

    for search_path in search_paths:
        if search_path.exists():
            logger.info(f"Found yt-dlp at: {search_path}")
            return str(search_path)

    return None


# Cache the yt-dlp path
_ytdlp_path: Optional[str] = None


def get_ytdlp_path() -> Optional[str]:
    """Get cached yt-dlp path."""
    global _ytdlp_path
    if _ytdlp_path is None:
        _ytdlp_path = _find_ytdlp_binary()
    return _ytdlp_path


@dataclass
class FormatInfo:
    """Information about a video format/quality option.

    Attributes:
        format_id: yt-dlp format identifier
        ext: File extension (webm, mp4, mkv, etc.)
        resolution: Video resolution (e.g., "1920x1080", "3840x2160")
        fps: Frames per second
        vcodec: Video codec (vp9, av1, h264, etc.)
        acodec: Audio codec (opus, aac, etc.)
        filesize: Estimated file size in bytes (may be None)
        tbr: Total bitrate in kbps
        vbr: Video bitrate in kbps
        abr: Audio bitrate in kbps
        width: Video width in pixels
        height: Video height in pixels
    """
    format_id: str
    ext: str
    resolution: Optional[str] = None
    fps: Optional[float] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    filesize: Optional[int] = None
    tbr: Optional[float] = None
    vbr: Optional[float] = None
    abr: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def quality_score(self) -> float:
        """Calculate quality score for format comparison.

        Higher score = better quality. Considers resolution, codec, and fps.

        Returns:
            Quality score as a float
        """
        score = 0.0

        # Resolution score (height-based)
        if self.height:
            score += self.height * 10

        # Codec preference: av1 > vp9 > h264 > others
        if self.vcodec:
            codec = self.vcodec.lower()
            if "av01" in codec or "av1" in codec:
                score += 3000
            elif "vp9" in codec or "vp09" in codec:
                score += 2000
            elif "avc" in codec or "h264" in codec:
                score += 1000

        # FPS bonus
        if self.fps:
            if self.fps >= 60:
                score += 500
            elif self.fps >= 30:
                score += 250

        # Bitrate bonus
        if self.tbr:
            score += self.tbr / 100

        return score

    def __str__(self) -> str:
        """Human-readable format description."""
        parts = [self.format_id]
        if self.resolution:
            parts.append(self.resolution)
        if self.fps:
            parts.append(f"{self.fps:.0f}fps")
        if self.vcodec:
            parts.append(self.vcodec)
        if self.filesize:
            size_mb = self.filesize / (1024 * 1024)
            parts.append(f"{size_mb:.1f}MB")
        return " | ".join(parts)


@dataclass
class ChapterInfo:
    """Information about a video chapter.

    Attributes:
        title: Chapter title
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    title: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Chapter duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class VideoInfo:
    """Complete metadata for a YouTube video.

    Attributes:
        title: Video title
        description: Full video description
        duration: Video duration in seconds
        upload_date: Upload date (YYYYMMDD format from yt-dlp)
        thumbnail_url: URL to the video thumbnail
        channel: Channel name
        channel_id: Channel ID
        view_count: Number of views
        like_count: Number of likes
        available_formats: List of available format options
        chapters: List of chapter markers
        url: Original video URL
        video_id: YouTube video ID
        is_live: Whether this is a live stream
        age_restricted: Whether video is age-restricted
        categories: Video categories
        tags: Video tags
    """
    title: str
    description: str
    duration: float
    upload_date: Optional[str] = None
    thumbnail_url: Optional[str] = None
    channel: Optional[str] = None
    channel_id: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    available_formats: List[FormatInfo] = field(default_factory=list)
    chapters: List[ChapterInfo] = field(default_factory=list)
    url: Optional[str] = None
    video_id: Optional[str] = None
    is_live: bool = False
    age_restricted: bool = False
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def upload_datetime(self) -> Optional[date]:
        """Convert upload_date string to date object."""
        if not self.upload_date:
            return None
        try:
            return datetime.strptime(self.upload_date, "%Y%m%d").date()
        except ValueError:
            return None

    @property
    def duration_formatted(self) -> str:
        """Get duration as HH:MM:SS string."""
        hours, remainder = divmod(int(self.duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @property
    def best_format(self) -> Optional[FormatInfo]:
        """Get the highest quality format available."""
        if not self.available_formats:
            return None
        return max(self.available_formats, key=lambda f: f.quality_score)


@dataclass
class DownloadProgress:
    """Progress information for video downloads.

    Attributes:
        status: Current status (downloading, finished, error)
        downloaded_bytes: Bytes downloaded so far
        total_bytes: Total file size in bytes
        speed: Download speed in bytes/second
        eta: Estimated time remaining in seconds
        percent: Download progress percentage (0-100)
        filename: Current filename being downloaded
    """
    status: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed: float = 0.0
    eta: float = 0.0
    percent: float = 0.0
    filename: Optional[str] = None

    @property
    def speed_formatted(self) -> str:
        """Get speed as human-readable string."""
        if self.speed < 1024:
            return f"{self.speed:.1f} B/s"
        elif self.speed < 1024 * 1024:
            return f"{self.speed / 1024:.1f} KB/s"
        else:
            return f"{self.speed / (1024 * 1024):.1f} MB/s"

    @property
    def eta_formatted(self) -> str:
        """Get ETA as human-readable string."""
        if self.eta <= 0:
            return "Unknown"
        minutes, seconds = divmod(int(self.eta), 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


# Type alias for progress callback
ProgressCallback = Callable[[DownloadProgress], None]


class YouTubeDownloadError(Exception):
    """Raised when YouTube download fails."""
    pass


class YouTubeMetadataError(Exception):
    """Raised when metadata extraction fails."""
    pass


class YouTubeDownloader:
    """YouTube video downloader with quality selection and progress tracking.

    Uses yt-dlp as the backend for robust downloading with support for
    quality selection, playlist handling, and metadata preservation.

    Attributes:
        QUALITY_PRESETS: Predefined quality selection presets
        CODEC_PRIORITY: Preferred codec order

    Example:
        >>> downloader = YouTubeDownloader(output_dir=Path("./videos"))
        >>>
        >>> # Get video information
        >>> info = downloader.get_video_info("https://youtube.com/watch?v=xxx")
        >>> print(f"Title: {info.title}, Duration: {info.duration_formatted}")
        >>>
        >>> # Download with progress callback
        >>> def on_progress(p):
        ...     print(f"Downloaded: {p.percent:.1f}% at {p.speed_formatted}")
        >>> path = downloader.download(url, progress_callback=on_progress)
    """

    QUALITY_PRESETS = {
        "best": "bestvideo[ext=webm]+bestaudio[ext=webm]/bestvideo+bestaudio/best",
        "4k": "bestvideo[height<=2160][ext=webm]+bestaudio[ext=webm]/"
              "bestvideo[height<=2160]+bestaudio/best[height<=2160]",
        "1080p": "bestvideo[height<=1080][ext=webm]+bestaudio[ext=webm]/"
                 "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "720p": "bestvideo[height<=720][ext=webm]+bestaudio[ext=webm]/"
                "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "audio": "bestaudio[ext=m4a]/bestaudio",
    }

    CODEC_PRIORITY = ["av01", "vp9", "vp09", "avc1", "h264"]

    def __init__(
        self,
        output_dir: Path,
        prefer_quality: str = "best",
        cookies_file: Optional[Path] = None,
    ) -> None:
        """Initialize YouTubeDownloader.

        Args:
            output_dir: Directory for downloaded videos
            prefer_quality: Default quality preset (best, 4k, 1080p, 720p, etc.)
            cookies_file: Optional path to cookies.txt for age-restricted content

        Raises:
            FileNotFoundError: If yt-dlp is not installed
        """
        self.output_dir = Path(output_dir)
        self.prefer_quality = prefer_quality
        self.cookies_file = cookies_file

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify yt-dlp is available
        if not self._check_ytdlp():
            raise FileNotFoundError(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )

        logger.info(
            f"YouTubeDownloader initialized: output_dir={output_dir}, "
            f"quality={prefer_quality}"
        )

    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed and accessible.

        Returns:
            True if yt-dlp is available, False otherwise
        """
        return get_ytdlp_path() is not None

    def _get_base_command(self) -> List[str]:
        """Get base yt-dlp command with common options.

        Returns:
            Base command list with common options
        """
        ytdlp_path = get_ytdlp_path()
        if not ytdlp_path:
            raise FileNotFoundError("yt-dlp not found")

        cmd = [ytdlp_path, "--no-warnings", "--no-check-certificates"]

        if self.cookies_file and self.cookies_file.exists():
            cmd.extend(["--cookies", str(self.cookies_file)])

        return cmd

    def get_video_info(self, url: str) -> VideoInfo:
        """Get video metadata without downloading.

        Extracts comprehensive metadata including available formats,
        chapters, and video details.

        Args:
            url: YouTube video URL

        Returns:
            VideoInfo object with complete metadata

        Raises:
            YouTubeMetadataError: If metadata extraction fails
        """
        cmd = self._get_base_command() + [
            "--dump-json",
            "--skip-download",
            url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown error"
                raise YouTubeMetadataError(f"Failed to get video info: {error_msg}")

            data = json.loads(result.stdout)
            return self._parse_video_info(data, url)

        except subprocess.TimeoutExpired:
            raise YouTubeMetadataError("Metadata extraction timed out")
        except json.JSONDecodeError as e:
            raise YouTubeMetadataError(f"Failed to parse video info: {e}")
        except Exception as e:
            raise YouTubeMetadataError(f"Unexpected error: {e}")

    def _parse_video_info(self, data: Dict[str, Any], url: str) -> VideoInfo:
        """Parse yt-dlp JSON output into VideoInfo object.

        Args:
            data: Parsed JSON from yt-dlp
            url: Original video URL

        Returns:
            VideoInfo object
        """
        # Parse formats
        formats = []
        for fmt in data.get("formats", []):
            # Skip audio-only or video-only format fragments
            format_info = FormatInfo(
                format_id=fmt.get("format_id", "unknown"),
                ext=fmt.get("ext", "unknown"),
                resolution=fmt.get("resolution"),
                fps=fmt.get("fps"),
                vcodec=fmt.get("vcodec"),
                acodec=fmt.get("acodec"),
                filesize=fmt.get("filesize") or fmt.get("filesize_approx"),
                tbr=fmt.get("tbr"),
                vbr=fmt.get("vbr"),
                abr=fmt.get("abr"),
                width=fmt.get("width"),
                height=fmt.get("height"),
            )
            formats.append(format_info)

        # Parse chapters
        chapters = []
        for chap in data.get("chapters", []) or []:
            chapter_info = ChapterInfo(
                title=chap.get("title", ""),
                start_time=chap.get("start_time", 0),
                end_time=chap.get("end_time", 0),
            )
            chapters.append(chapter_info)

        return VideoInfo(
            title=data.get("title", "Unknown"),
            description=data.get("description", ""),
            duration=data.get("duration", 0),
            upload_date=data.get("upload_date"),
            thumbnail_url=data.get("thumbnail"),
            channel=data.get("channel") or data.get("uploader"),
            channel_id=data.get("channel_id") or data.get("uploader_id"),
            view_count=data.get("view_count"),
            like_count=data.get("like_count"),
            available_formats=formats,
            chapters=chapters,
            url=url,
            video_id=data.get("id"),
            is_live=data.get("is_live", False),
            age_restricted=data.get("age_limit", 0) > 0,
            categories=data.get("categories", []) or [],
            tags=data.get("tags", []) or [],
        )

    def get_available_formats(self, url: str) -> List[FormatInfo]:
        """Get list of all available video formats/qualities.

        Args:
            url: YouTube video URL

        Returns:
            List of FormatInfo objects sorted by quality (best first)

        Raises:
            YouTubeMetadataError: If format extraction fails
        """
        info = self.get_video_info(url)

        # Filter and sort formats
        video_formats = [
            f for f in info.available_formats
            if f.vcodec and f.vcodec != "none"
        ]

        # Sort by quality score (descending)
        video_formats.sort(key=lambda f: f.quality_score, reverse=True)

        return video_formats

    def download(
        self,
        url: str,
        quality: str = "best",
        progress_callback: Optional[ProgressCallback] = None,
        filename: Optional[str] = None,
        write_info_json: bool = True,
    ) -> Path:
        """Download a YouTube video.

        Downloads the video with the specified quality, optionally saving
        metadata to a sidecar .info.json file.

        Args:
            url: YouTube video URL
            quality: Quality preset or yt-dlp format spec
            progress_callback: Optional callback for progress updates
            filename: Optional custom filename (without extension)
            write_info_json: Whether to save metadata as .info.json

        Returns:
            Path to the downloaded video file

        Raises:
            YouTubeDownloadError: If download fails
        """
        # Get format spec
        if quality in self.QUALITY_PRESETS:
            format_spec = self.QUALITY_PRESETS[quality]
        else:
            format_spec = quality

        # Build output template
        if filename:
            # Sanitize filename
            safe_filename = re.sub(r'[<>:"/\\|?*]', "", filename)
            output_template = str(self.output_dir / f"{safe_filename}.%(ext)s")
        else:
            output_template = str(self.output_dir / "%(title)s.%(ext)s")

        # Build command
        cmd = self._get_base_command() + [
            "--format", format_spec,
            "--merge-output-format", "mkv",
            "--output", output_template,
            "--no-playlist",
            "--newline",  # For progress parsing
            "--progress-template", "%(progress._percent_str)s %(progress._speed_str)s %(progress._eta_str)s",
        ]

        if write_info_json:
            cmd.append("--write-info-json")

        # Add embed options for metadata preservation
        cmd.extend([
            "--embed-chapters",
            "--embed-metadata",
            "--embed-thumbnail",
        ])

        cmd.append(url)

        logger.info(f"Starting download: {url} with quality={quality}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_file = None

            for line in process.stdout:
                line = line.strip()

                # Parse progress
                if progress_callback:
                    progress = self._parse_progress_line(line)
                    if progress:
                        progress_callback(progress)

                # Look for download destination
                if "[download] Destination:" in line:
                    output_file = Path(line.split("Destination:")[-1].strip())
                elif "Merging formats into" in line:
                    # Extract the merged output file
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        output_file = Path(match.group(1))
                elif "[Merger]" in line and "Merging formats into" in line:
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        output_file = Path(match.group(1))

            process.wait()

            if process.returncode != 0:
                raise YouTubeDownloadError(
                    f"Download failed with exit code {process.returncode}"
                )

            # Find the downloaded file if we don't have it
            if not output_file or not output_file.exists():
                output_file = self._find_downloaded_file(url, filename)

            if not output_file:
                raise YouTubeDownloadError("Could not find downloaded file")

            logger.info(f"Download complete: {output_file}")

            # Call progress callback with finished status
            if progress_callback:
                progress_callback(DownloadProgress(
                    status="finished",
                    percent=100.0,
                    filename=str(output_file),
                ))

            return output_file

        except subprocess.TimeoutExpired:
            raise YouTubeDownloadError("Download timed out")
        except Exception as e:
            if isinstance(e, YouTubeDownloadError):
                raise
            raise YouTubeDownloadError(f"Download failed: {e}")

    def _parse_progress_line(self, line: str) -> Optional[DownloadProgress]:
        """Parse yt-dlp progress output line.

        Args:
            line: Output line from yt-dlp

        Returns:
            DownloadProgress or None if line is not progress info
        """
        if not line or "[download]" not in line:
            return None

        try:
            # Try to parse percentage
            percent_match = re.search(r"(\d+\.?\d*)%", line)
            speed_match = re.search(r"at\s+([\d.]+\s*[KMG]?i?B/s)", line)
            eta_match = re.search(r"ETA\s+(\d+:\d+(?::\d+)?)", line)
            size_match = re.search(r"of\s+~?([\d.]+\s*[KMG]?i?B)", line)

            progress = DownloadProgress(status="downloading")

            if percent_match:
                progress.percent = float(percent_match.group(1))

            if speed_match:
                speed_str = speed_match.group(1)
                progress.speed = self._parse_size(speed_str)

            if eta_match:
                progress.eta = self._parse_time(eta_match.group(1))

            if size_match:
                progress.total_bytes = int(self._parse_size(size_match.group(1)))
                progress.downloaded_bytes = int(progress.total_bytes * progress.percent / 100)

            return progress

        except Exception:
            return None

    def _parse_size(self, size_str: str) -> float:
        """Parse size string like '10.5MiB' to bytes.

        Args:
            size_str: Size string with unit

        Returns:
            Size in bytes
        """
        size_str = size_str.strip().upper()
        multipliers = {
            "B": 1,
            "KB": 1024, "KIB": 1024, "K": 1024,
            "MB": 1024**2, "MIB": 1024**2, "M": 1024**2,
            "GB": 1024**3, "GIB": 1024**3, "G": 1024**3,
        }

        for unit, mult in multipliers.items():
            if unit in size_str:
                num = float(re.sub(r"[^\d.]", "", size_str.replace(unit, "")))
                return num * mult

        try:
            return float(re.sub(r"[^\d.]", "", size_str))
        except ValueError:
            return 0.0

    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '1:30' or '1:30:00' to seconds.

        Args:
            time_str: Time string in HH:MM:SS or MM:SS format

        Returns:
            Time in seconds
        """
        parts = time_str.split(":")
        parts = [int(p) for p in parts]

        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return 0.0

    def _find_downloaded_file(
        self,
        url: str,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """Find the most recently downloaded file.

        Args:
            url: Video URL (used to extract video ID)
            filename: Expected filename if specified

        Returns:
            Path to downloaded file or None
        """
        extensions = [".mkv", ".webm", ".mp4", ".avi", ".mov"]

        if filename:
            safe_filename = re.sub(r'[<>:"/\\|?*]', "", filename)
            for ext in extensions:
                candidate = self.output_dir / f"{safe_filename}{ext}"
                if candidate.exists():
                    return candidate

        # Get the most recently modified video file
        video_files = []
        for ext in extensions:
            video_files.extend(self.output_dir.glob(f"*{ext}"))

        if video_files:
            return max(video_files, key=lambda p: p.stat().st_mtime)

        return None

    def download_best_quality(
        self,
        url: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download video in the best available quality.

        Automatically selects the highest quality format with preferred codecs.

        Args:
            url: YouTube video URL
            progress_callback: Optional progress callback

        Returns:
            Path to downloaded video file
        """
        return self.download(
            url,
            quality="best",
            progress_callback=progress_callback,
        )

    def download_playlist(
        self,
        url: str,
        max_videos: Optional[int] = None,
        quality: str = "best",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Path]:
        """Download videos from a YouTube playlist.

        Args:
            url: YouTube playlist URL
            max_videos: Maximum number of videos to download (None for all)
            quality: Quality preset for all videos
            progress_callback: Optional progress callback

        Returns:
            List of paths to downloaded video files

        Raises:
            YouTubeDownloadError: If playlist download fails
        """
        # Get format spec
        if quality in self.QUALITY_PRESETS:
            format_spec = self.QUALITY_PRESETS[quality]
        else:
            format_spec = quality

        # Build command
        cmd = self._get_base_command() + [
            "--format", format_spec,
            "--merge-output-format", "mkv",
            "--output", str(self.output_dir / "%(playlist_index)s - %(title)s.%(ext)s"),
            "--yes-playlist",
            "--write-info-json",
            "--embed-chapters",
            "--embed-metadata",
            "--newline",
        ]

        if max_videos:
            cmd.extend(["--playlist-end", str(max_videos)])

        cmd.append(url)

        logger.info(f"Starting playlist download: {url}")

        try:
            # Track files before download
            existing_files = set(self.output_dir.glob("*.mkv"))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            current_video = None

            for line in process.stdout:
                line = line.strip()

                # Parse progress
                if progress_callback:
                    progress = self._parse_progress_line(line)
                    if progress:
                        if current_video:
                            progress.filename = current_video
                        progress_callback(progress)

                # Track current video
                if "[download] Downloading video" in line:
                    match = re.search(r"(\d+)\s+of\s+(\d+)", line)
                    if match:
                        current_video = f"Video {match.group(1)}/{match.group(2)}"

            process.wait()

            if process.returncode != 0:
                raise YouTubeDownloadError(
                    f"Playlist download failed with exit code {process.returncode}"
                )

            # Find newly downloaded files
            new_files = set(self.output_dir.glob("*.mkv")) - existing_files
            downloaded = sorted(list(new_files), key=lambda p: p.name)

            logger.info(f"Playlist download complete: {len(downloaded)} videos")

            return downloaded

        except Exception as e:
            if isinstance(e, YouTubeDownloadError):
                raise
            raise YouTubeDownloadError(f"Playlist download failed: {e}")

    def save_metadata(
        self,
        video_info: VideoInfo,
        output_path: Path,
    ) -> Path:
        """Save video metadata to a sidecar JSON file.

        Creates a .info.json file with title, description, chapters, etc.

        Args:
            video_info: VideoInfo object with metadata
            output_path: Path to the video file

        Returns:
            Path to the created metadata file
        """
        metadata_path = output_path.with_suffix(".info.json")

        metadata = {
            "title": video_info.title,
            "description": video_info.description,
            "duration": video_info.duration,
            "duration_formatted": video_info.duration_formatted,
            "upload_date": video_info.upload_date,
            "channel": video_info.channel,
            "channel_id": video_info.channel_id,
            "view_count": video_info.view_count,
            "like_count": video_info.like_count,
            "thumbnail_url": video_info.thumbnail_url,
            "url": video_info.url,
            "video_id": video_info.video_id,
            "categories": video_info.categories,
            "tags": video_info.tags,
            "chapters": [
                {
                    "title": ch.title,
                    "start_time": ch.start_time,
                    "end_time": ch.end_time,
                }
                for ch in video_info.chapters
            ],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata to: {metadata_path}")
        return metadata_path

    def extract_chapters(self, video_info: VideoInfo) -> List[ChapterInfo]:
        """Extract chapter markers from video info.

        Args:
            video_info: VideoInfo with chapter data

        Returns:
            List of ChapterInfo objects
        """
        return video_info.chapters


def get_youtube_downloader(
    output_dir: Path,
    prefer_quality: str = "best",
    cookies_file: Optional[Path] = None,
) -> YouTubeDownloader:
    """Get a YouTubeDownloader instance.

    Convenience function for creating a YouTubeDownloader.

    Args:
        output_dir: Output directory for downloads
        prefer_quality: Default quality preset
        cookies_file: Optional cookies file for age-restricted content

    Returns:
        Configured YouTubeDownloader instance
    """
    return YouTubeDownloader(
        output_dir=output_dir,
        prefer_quality=prefer_quality,
        cookies_file=cookies_file,
    )
