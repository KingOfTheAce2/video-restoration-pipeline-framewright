"""Shared pytest fixtures for FrameWright tests."""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def project_dir(temp_dir):
    """Create a project directory structure."""
    project = temp_dir / "test_project"
    project.mkdir(parents=True, exist_ok=True)
    return project


@pytest.fixture
def video_file(temp_dir):
    """Create a mock video file."""
    video_path = temp_dir / "test_video.mp4"
    video_path.write_text("mock video content")
    return video_path


@pytest.fixture
def audio_file(temp_dir):
    """Create a mock audio file."""
    audio_path = temp_dir / "test_audio.wav"
    audio_path.write_text("mock audio content")
    return audio_path


@pytest.fixture
def frames_dir(temp_dir):
    """Create a directory with mock frame files."""
    frames_path = temp_dir / "frames"
    frames_path.mkdir(parents=True, exist_ok=True)

    # Create mock frame files
    for i in range(1, 11):
        frame_file = frames_path / f"frame_{i:08d}.png"
        frame_file.write_text(f"mock frame {i}")

    return frames_path


@pytest.fixture
def mock_subprocess_run():
    """Create a mock for subprocess.run."""
    mock = Mock()
    mock.return_value.returncode = 0
    mock.return_value.stdout = ""
    mock.return_value.stderr = ""
    return mock


@pytest.fixture
def mock_subprocess_popen():
    """Create a mock for subprocess.Popen."""
    mock = MagicMock()
    process_mock = MagicMock()
    process_mock.stdout = iter(["Processing...", "Done"])
    process_mock.wait.return_value = 0
    process_mock.communicate.return_value = ("", "")
    mock.return_value = process_mock
    return mock


@pytest.fixture
def mock_ffprobe_output():
    """Mock ffprobe JSON output."""
    return '''{
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001"
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000"
            }
        ],
        "format": {
            "duration": "120.5",
            "bit_rate": "5000000"
        }
    }'''


@pytest.fixture
def expected_metadata():
    """Expected metadata after parsing."""
    return {
        'width': 1920,
        'height': 1080,
        'framerate': 29.97002997002997,
        'codec': 'h264',
        'duration': 120.5,
        'bit_rate': 5000000,
        'has_audio': True,
        'audio_codec': 'aac',
        'audio_sample_rate': 48000
    }


@pytest.fixture
def mock_progress_callback():
    """Create a mock progress callback function."""
    return Mock()


@pytest.fixture
def mock_shutil_which():
    """Mock shutil.which to return tool paths."""
    def which_mock(command):
        tools = {
            'ffmpeg': '/usr/bin/ffmpeg',
            'ffprobe': '/usr/bin/ffprobe',
            'yt-dlp': '/usr/bin/yt-dlp',
            'realesrgan-ncnn-vulkan': '/usr/bin/realesrgan-ncnn-vulkan'
        }
        return tools.get(command)
    return which_mock
