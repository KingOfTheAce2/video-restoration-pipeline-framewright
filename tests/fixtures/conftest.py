"""Fixtures for generating synthetic test videos and frames.

This module provides utilities to generate synthetic test videos and images
for integration testing without requiring external video files.
"""
import struct
import tempfile
import shutil
import zlib
from pathlib import Path
from typing import List, Tuple, Optional
import pytest


def generate_png_image(
    width: int = 320,
    height: int = 240,
    color: Tuple[int, int, int] = (128, 64, 192),
    frame_number: int = 0
) -> bytes:
    """Generate a minimal valid PNG image.

    Creates a synthetic PNG image with a solid color and an embedded
    pattern for frame identification.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple (0-255 each)
        frame_number: Frame number to embed in the pattern

    Returns:
        Raw PNG file bytes
    """
    def create_chunk(chunk_type: bytes, data: bytes) -> bytes:
        """Create a PNG chunk with CRC."""
        chunk_len = struct.pack(">I", len(data))
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        crc_bytes = struct.pack(">I", crc)
        return chunk_len + chunk_type + data + crc_bytes

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = create_chunk(b'IHDR', ihdr_data)

    # Generate raw pixel data (RGB)
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # Filter byte
        for x in range(width):
            # Create a pattern based on position and frame number
            r, g, b = color
            # Add some variation for frame identification
            if (x + y + frame_number) % 32 < 4:
                r = min(255, r + 64)
            if x < 16 and y < 16:
                # Embed frame number pattern in top-left corner
                bit = (frame_number >> ((y % 4) * 4 + (x % 4))) & 1
                if bit:
                    r, g, b = 255, 255, 255
            raw_data += bytes([r, g, b])

    # Compress the data
    compressed = zlib.compress(raw_data, 6)
    idat = create_chunk(b'IDAT', compressed)

    # IEND chunk
    iend = create_chunk(b'IEND', b'')

    return signature + ihdr + idat + iend


def generate_minimal_mp4(
    duration_seconds: float = 5.0,
    width: int = 320,
    height: int = 240,
    fps: float = 24.0
) -> bytes:
    """Generate a minimal valid MP4 file structure.

    Creates a minimal MP4 file that can be parsed by ffprobe.
    Note: This creates a structurally valid MP4 but the video data
    is minimal - suitable for metadata parsing tests only.

    For full video tests, use the ffmpeg-based generator.

    Args:
        duration_seconds: Video duration
        width: Video width
        height: Video height
        fps: Frames per second

    Returns:
        Raw MP4 file bytes
    """
    # This is a minimal MP4 structure
    # For real testing, we use ffmpeg to generate proper test videos

    def box(box_type: bytes, data: bytes) -> bytes:
        """Create an MP4 box."""
        size = len(data) + 8
        return struct.pack(">I", size) + box_type + data

    # ftyp box
    ftyp = box(b'ftyp', b'isom\x00\x00\x00\x01' + b'isom' + b'avc1')

    # Minimal moov box structure
    mvhd_data = (
        b'\x00\x00\x00\x00' +  # version/flags
        b'\x00\x00\x00\x00' +  # creation time
        b'\x00\x00\x00\x00' +  # modification time
        struct.pack(">I", 1000) +  # timescale
        struct.pack(">I", int(duration_seconds * 1000)) +  # duration
        b'\x00\x01\x00\x00' +  # rate
        b'\x01\x00' +  # volume
        b'\x00\x00' +  # reserved
        b'\x00' * 8 +  # reserved
        b'\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' +  # matrix (36 bytes)
        b'\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00' +
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00' +
        b'\x00' * 24 +  # pre-defined
        struct.pack(">I", 2)  # next track id
    )
    mvhd = box(b'mvhd', mvhd_data)

    moov = box(b'moov', mvhd)

    return ftyp + moov


def generate_test_video_ffmpeg(
    output_path: Path,
    duration_seconds: float = 5.0,
    width: int = 480,
    height: int = 360,
    fps: float = 24.0,
    color: str = "blue"
) -> bool:
    """Generate a test video using ffmpeg.

    Creates a proper test video using ffmpeg's lavfi source.

    Args:
        output_path: Path to save the video
        duration_seconds: Video duration
        width: Video width
        height: Video height
        fps: Frames per second
        color: Color name or hex code

    Returns:
        True if successful, False otherwise
    """
    import subprocess

    # Use testsrc2 for a pattern that includes timecode
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'testsrc2=duration={duration_seconds}:size={width}x{height}:rate={fps}',
        '-f', 'lavfi',
        '-i', f'sine=frequency=1000:duration={duration_seconds}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-crf', '28',
        '-c:a', 'aac',
        '-b:a', '64k',
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60
        )
        return result.returncode == 0 and output_path.exists()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available for video generation."""
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


class SyntheticVideoGenerator:
    """Generate synthetic test videos and frames for testing."""

    def __init__(self, base_dir: Path):
        """Initialize generator with output directory.

        Args:
            base_dir: Base directory for generated files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._has_ffmpeg = check_ffmpeg_available()

    @property
    def has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        return self._has_ffmpeg

    def generate_video(
        self,
        name: str = "test_video.mp4",
        duration: float = 5.0,
        width: int = 480,
        height: int = 360,
        fps: float = 24.0
    ) -> Optional[Path]:
        """Generate a synthetic test video.

        Args:
            name: Output filename
            duration: Video duration in seconds
            width: Video width
            height: Video height
            fps: Frames per second

        Returns:
            Path to generated video, or None if generation failed
        """
        output_path = self.base_dir / name

        if self._has_ffmpeg:
            if generate_test_video_ffmpeg(output_path, duration, width, height, fps):
                return output_path

        # Fallback to minimal MP4 structure
        mp4_data = generate_minimal_mp4(duration, width, height, fps)
        output_path.write_bytes(mp4_data)
        return output_path

    def generate_frames(
        self,
        count: int = 10,
        width: int = 320,
        height: int = 240,
        pattern: str = "frame_{:08d}.png"
    ) -> Path:
        """Generate a sequence of PNG frames.

        Args:
            count: Number of frames to generate
            width: Frame width
            height: Frame height
            pattern: Filename pattern with frame number placeholder

        Returns:
            Path to directory containing frames
        """
        frames_dir = self.base_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        colors = [
            (128, 64, 192),
            (64, 128, 192),
            (192, 128, 64),
            (128, 192, 64),
            (64, 192, 128),
        ]

        for i in range(1, count + 1):
            color = colors[i % len(colors)]
            png_data = generate_png_image(width, height, color, i)
            frame_path = frames_dir / pattern.format(i)
            frame_path.write_bytes(png_data)

        return frames_dir

    def generate_test_frame(
        self,
        name: str = "test_frame.png",
        width: int = 320,
        height: int = 240
    ) -> Path:
        """Generate a single test frame.

        Args:
            name: Output filename
            width: Frame width
            height: Frame height

        Returns:
            Path to generated frame
        """
        output_path = self.base_dir / name
        png_data = generate_png_image(width, height)
        output_path.write_bytes(png_data)
        return output_path

    def generate_corrupt_video(self, name: str = "corrupt_video.mp4") -> Path:
        """Generate a corrupt/invalid video file.

        Args:
            name: Output filename

        Returns:
            Path to generated corrupt file
        """
        output_path = self.base_dir / name
        # Write random-ish bytes that look like MP4 but aren't valid
        corrupt_data = b'CORRUPT' + b'\x00' * 100 + b'ftypisom'
        output_path.write_bytes(corrupt_data)
        return output_path

    def cleanup(self):
        """Remove all generated files."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)


# Pytest fixtures
@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    """Create a session-scoped fixtures directory."""
    fixtures_path = tmp_path_factory.mktemp("framewright_fixtures")
    yield fixtures_path
    # Cleanup is automatic with tmp_path_factory


@pytest.fixture(scope="session")
def video_generator(fixtures_dir):
    """Create a video generator for the test session."""
    generator = SyntheticVideoGenerator(fixtures_dir)
    yield generator


@pytest.fixture(scope="session")
def test_video_small(video_generator) -> Optional[Path]:
    """Generate a small 5-second 480p test video.

    Returns:
        Path to test video, or None if generation failed
    """
    return video_generator.generate_video(
        name="test_video_small.mp4",
        duration=5.0,
        width=480,
        height=360,
        fps=24.0
    )


@pytest.fixture(scope="session")
def test_frame(video_generator) -> Path:
    """Generate a single test frame."""
    return video_generator.generate_test_frame(
        name="test_frame.png",
        width=320,
        height=240
    )


@pytest.fixture(scope="session")
def test_frames_sequence(video_generator) -> Path:
    """Generate a sequence of test frames."""
    return video_generator.generate_frames(
        count=24,  # 1 second at 24fps
        width=320,
        height=240
    )


@pytest.fixture
def corrupt_video(tmp_path) -> Path:
    """Generate a corrupt video file."""
    generator = SyntheticVideoGenerator(tmp_path / "corrupt")
    return generator.generate_corrupt_video()


@pytest.fixture
def ffmpeg_available(video_generator) -> bool:
    """Check if ffmpeg is available for full video tests."""
    return video_generator.has_ffmpeg
