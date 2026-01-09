"""
FFmpeg Helper Functions
Utilities for video processing using FFmpeg and ffprobe.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any


class FFmpegError(Exception):
    """Custom exception for FFmpeg-related errors."""
    pass


def check_ffmpeg_installed() -> bool:
    """
    Check if FFmpeg and ffprobe are installed and available.

    Returns:
        bool: True if both FFmpeg and ffprobe are installed

    Raises:
        FFmpegError: If FFmpeg or ffprobe is not found
    """
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    if not ffmpeg_path:
        raise FFmpegError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )

    if not ffprobe_path:
        raise FFmpegError(
            "ffprobe not found. Please install FFmpeg (includes ffprobe)."
        )

    return True


# Cache for available encoders
_available_encoders: Optional[set] = None


def get_available_encoders() -> set:
    """
    Get the set of available FFmpeg video encoders.

    Returns:
        Set of encoder names (e.g., {'libx264', 'libx265', 'libvpx'})
    """
    global _available_encoders
    if _available_encoders is not None:
        return _available_encoders

    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Parse encoder list - each line starts with flags then encoder name
        encoders = set()
        for line in result.stdout.split('\n'):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith('V'):  # Video encoders start with V
                encoders.add(parts[1])
        _available_encoders = encoders
        return encoders
    except Exception:
        return set()


def get_best_video_codec(preferred: str = 'libx265') -> Tuple[str, str]:
    """
    Get the best available video codec with fallback options.

    This function checks if the preferred codec is available and falls back
    to alternatives if not. This is particularly important for cloud environments
    where FFmpeg may be compiled without certain codecs.

    Args:
        preferred: Preferred codec (default: libx265 for high quality archival)

    Returns:
        Tuple of (codec_name, pixel_format) for use with FFmpeg
    """
    encoders = get_available_encoders()

    # Codec priority with their optimal pixel formats
    codec_options = [
        ('libx265', 'yuv420p10le'),  # Best quality, 10-bit
        ('libx264', 'yuv420p'),       # Widely available
        ('libvpx-vp9', 'yuv420p'),    # WebM format
        ('mpeg4', 'yuv420p'),         # Last resort
    ]

    # If preferred codec is available, use it
    for codec, pix_fmt in codec_options:
        if codec == preferred and codec in encoders:
            return (codec, pix_fmt)

    # Otherwise, find best available
    for codec, pix_fmt in codec_options:
        if codec in encoders:
            return (codec, pix_fmt)

    # Fallback to libx264 even if not detected (usually available)
    return ('libx264', 'yuv420p')


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Get detailed video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video metadata

    Raises:
        FFmpegError: If ffprobe command fails
    """
    check_ffmpeg_installed()

    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to get video info: {e.stderr}")
    except json.JSONDecodeError as e:
        raise FFmpegError(f"Failed to parse ffprobe output: {e}")


def get_video_fps(video_path: Path) -> float:
    """
    Get video frames per second.

    Args:
        video_path: Path to video file

    Returns:
        Frame rate as float
    """
    info = get_video_info(video_path)

    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            fps_str = stream.get('r_frame_rate', '0/1')
            num, den = map(int, fps_str.split('/'))
            return num / den if den != 0 else 0.0

    return 0.0


def get_video_duration(video_path: Path) -> float:
    """
    Get video duration in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds as float
    """
    info = get_video_info(video_path)
    return float(info.get('format', {}).get('duration', 0.0))


def get_video_resolution(video_path: Path) -> Tuple[int, int]:
    """
    Get video resolution (width, height).

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)
    """
    info = get_video_info(video_path)

    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            return (width, height)

    return (0, 0)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    quality: int = 2
) -> int:
    """
    Extract frames from video using FFmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Target frame rate (None = use source fps)
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = until end)
        quality: JPEG quality (1-31, lower is better, default: 2)

    Returns:
        Number of frames extracted

    Raises:
        FFmpegError: If frame extraction fails
    """
    check_ffmpeg_installed()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['ffmpeg', '-i', str(video_path)]

    if start_time is not None:
        cmd.extend(['-ss', str(start_time)])

    if end_time is not None:
        cmd.extend(['-to', str(end_time)])

    if fps is not None:
        cmd.extend(['-vf', f'fps={fps}'])

    cmd.extend([
        '-qscale:v', str(quality),
        str(output_dir / 'frame_%06d.png')
    ])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count extracted frames
        frame_count = len(list(output_dir.glob('frame_*.png')))
        return frame_count
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract frames: {e.stderr}")


def reassemble_video(
    frames_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    crf: int = 18,
    preset: str = 'medium',
    codec: str = 'libx264'
) -> None:
    """
    Reassemble video from frames using FFmpeg.

    Args:
        frames_dir: Directory containing frames
        output_path: Path for output video
        fps: Frame rate for output video
        crf: Constant Rate Factor (0-51, lower is better quality, default: 18)
        preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        codec: Video codec to use (libx264, libx265, etc.)

    Raises:
        FFmpegError: If video reassembly fails
    """
    check_ffmpeg_installed()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', str(frames_dir / 'frame_*.png'),
        '-c:v', codec,
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to reassemble video: {e.stderr}")


def extract_audio(video_path: Path, output_path: Path, codec: str = 'aac') -> None:
    """
    Extract audio track from video.

    Args:
        video_path: Path to input video
        output_path: Path for output audio file
        codec: Audio codec (aac, mp3, flac, etc.)

    Raises:
        FFmpegError: If audio extraction fails
    """
    check_ffmpeg_installed()

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', codec,
        '-y',  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract audio: {e.stderr}")


def merge_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_codec: str = 'aac'
) -> None:
    """
    Merge audio and video files.

    Args:
        video_path: Path to video file (no audio)
        audio_path: Path to audio file
        output_path: Path for output video with audio
        audio_codec: Audio codec to use (copy, aac, mp3, etc.)

    Raises:
        FFmpegError: If merging fails
    """
    check_ffmpeg_installed()

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy',  # Copy video stream without re-encoding
        '-c:a', audio_codec,
        '-shortest',  # Finish when shortest input ends
        '-y',  # Overwrite output file
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to merge audio and video: {e.stderr}")


def check_video_has_audio(video_path: Path) -> bool:
    """
    Check if video file has an audio stream.

    Args:
        video_path: Path to video file

    Returns:
        True if video has audio stream, False otherwise
    """
    info = get_video_info(video_path)

    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'audio':
            return True

    return False


def probe_video(video_path: Path) -> Dict[str, Any]:
    """
    Get simplified video metadata in a standard format.

    This is a convenience wrapper that returns metadata in the same format
    as VideoRestorer.analyze_metadata().

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with keys: width, height, framerate, duration, codec,
                             has_audio, audio_codec
    """
    info = get_video_info(video_path)

    video_stream = None
    audio_stream = None

    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video' and video_stream is None:
            video_stream = stream
        elif stream.get('codec_type') == 'audio' and audio_stream is None:
            audio_stream = stream

    # Parse framerate
    framerate = 24.0
    if video_stream:
        fps_str = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            framerate = num / den if den != 0 else 24.0
        else:
            framerate = float(fps_str)

    return {
        'width': video_stream.get('width', 0) if video_stream else 0,
        'height': video_stream.get('height', 0) if video_stream else 0,
        'framerate': framerate,
        'duration': float(info.get('format', {}).get('duration', 0)),
        'codec': video_stream.get('codec_name', 'unknown') if video_stream else 'unknown',
        'has_audio': audio_stream is not None,
        'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
    }


def extract_frames_to_dir(
    video_path: Path,
    output_dir: Path,
    frame_pattern: str = "frame_%08d.png"
) -> int:
    """
    Extract all frames from video as PNG files.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_pattern: Filename pattern for frames

    Returns:
        Number of frames extracted

    Raises:
        FFmpegError: If extraction fails
    """
    check_ffmpeg_installed()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-qscale:v', '1',  # Highest quality
        '-qmin', '1',
        '-qmax', '1',
        str(output_dir / frame_pattern)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
        frame_count = len(list(output_dir.glob('*.png')))
        return frame_count
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract frames: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise FFmpegError("Frame extraction timed out (1 hour limit)")


def reassemble_from_frames(
    frames_dir: Path,
    output_path: Path,
    framerate: float,
    crf: int = 18,
    preset: str = 'medium',
    audio_source: Optional[Path] = None,
    frame_pattern: str = "frame_%08d.png"
) -> None:
    """
    Reassemble video from frames with optional audio.

    Args:
        frames_dir: Directory containing frames
        output_path: Path for output video
        framerate: Frame rate for output video
        crf: Constant Rate Factor (0-51, lower = better quality)
        preset: Encoding preset
        audio_source: Optional video file to copy audio from

    Raises:
        FFmpegError: If reassembly fails
    """
    check_ffmpeg_installed()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_pattern = frames_dir / frame_pattern

    cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', str(input_pattern),
    ]

    # Add audio from source if available
    if audio_source and audio_source.exists() and check_video_has_audio(audio_source):
        cmd.extend([
            '-i', str(audio_source),
            '-c:a', 'aac',
            '-b:a', '192k',
        ])

    # Use best available codec with automatic fallback
    codec, pix_fmt = get_best_video_codec('libx265')

    cmd.extend([
        '-c:v', codec,
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', pix_fmt,
        '-y',
        str(output_path)
    ])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)
    except subprocess.CalledProcessError as e:
        # If codec fails, try fallback to libx264
        if 'libx265' in codec or 'Unknown encoder' in str(e.stderr):
            cmd_fallback = cmd.copy()
            # Replace codec in command
            for i, arg in enumerate(cmd_fallback):
                if arg == codec:
                    cmd_fallback[i] = 'libx264'
                elif arg == pix_fmt:
                    cmd_fallback[i] = 'yuv420p'
            try:
                subprocess.run(cmd_fallback, capture_output=True, text=True, check=True, timeout=7200)
                return  # Success with fallback
            except subprocess.CalledProcessError:
                pass  # Fall through to original error
        raise FFmpegError(f"Failed to reassemble video: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise FFmpegError("Video reassembly timed out (2 hour limit)")
