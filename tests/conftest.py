"""Shared pytest fixtures for FrameWright tests."""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, MagicMock, patch
import pytest

# Import fixtures from fixtures package
from tests.fixtures.conftest import (
    SyntheticVideoGenerator,
    generate_png_image,
    check_ffmpeg_available,
    generate_test_video_ffmpeg,
)


# ============================================================================
# Session-scoped fixtures for integration tests
# ============================================================================

@pytest.fixture(scope="session")
def integration_fixtures_dir(tmp_path_factory) -> Path:
    """Create a session-scoped fixtures directory for integration tests."""
    return tmp_path_factory.mktemp("integration_fixtures")


@pytest.fixture(scope="session")
def synthetic_video_generator(integration_fixtures_dir) -> SyntheticVideoGenerator:
    """Create a synthetic video generator for the session."""
    return SyntheticVideoGenerator(integration_fixtures_dir)


@pytest.fixture(scope="session")
def test_video_5s(synthetic_video_generator) -> Optional[Path]:
    """Generate a 5-second 480p test video for integration tests.

    Returns None if ffmpeg is not available.
    """
    if not synthetic_video_generator.has_ffmpeg:
        return None
    return synthetic_video_generator.generate_video(
        name="test_video_5s.mp4",
        duration=5.0,
        width=480,
        height=360,
        fps=24.0
    )


@pytest.fixture(scope="session")
def test_video_1s(synthetic_video_generator) -> Optional[Path]:
    """Generate a 1-second test video for quick tests."""
    if not synthetic_video_generator.has_ffmpeg:
        return None
    return synthetic_video_generator.generate_video(
        name="test_video_1s.mp4",
        duration=1.0,
        width=320,
        height=240,
        fps=24.0
    )


@pytest.fixture(scope="session")
def test_frames_dir(synthetic_video_generator) -> Path:
    """Generate a directory with test frames."""
    return synthetic_video_generator.generate_frames(
        count=24,
        width=320,
        height=240
    )


@pytest.fixture(scope="session")
def test_single_frame(synthetic_video_generator) -> Path:
    """Generate a single test frame."""
    return synthetic_video_generator.generate_test_frame(
        name="single_frame.png",
        width=320,
        height=240
    )


# ============================================================================
# Mock fixtures for GPU and system resources
# ============================================================================

@pytest.fixture
def mock_gpu():
    """Mock GPU availability and VRAM for testing without hardware."""
    mock_info = {
        'device_count': 1,
        'device_name': 'Mock GPU',
        'total_mb': 8192,
        'free_mb': 6144,
        'used_mb': 2048,
    }

    with patch('framewright.utils.gpu.get_gpu_memory_info', return_value=mock_info):
        with patch('framewright.utils.gpu.is_gpu_available', return_value=True):
            yield mock_info


@pytest.fixture
def mock_no_gpu():
    """Mock GPU unavailability for CPU fallback testing."""
    with patch('framewright.utils.gpu.get_gpu_memory_info', return_value=None):
        with patch('framewright.utils.gpu.is_gpu_available', return_value=False):
            yield


@pytest.fixture
def mock_low_vram():
    """Mock low VRAM conditions for tile size testing."""
    mock_info = {
        'device_count': 1,
        'device_name': 'Mock GPU (Low VRAM)',
        'total_mb': 2048,
        'free_mb': 512,
        'used_mb': 1536,
    }

    with patch('framewright.utils.gpu.get_gpu_memory_info', return_value=mock_info):
        with patch('framewright.utils.gpu.is_gpu_available', return_value=True):
            yield mock_info


# ============================================================================
# Configuration fixtures for testing
# ============================================================================

@pytest.fixture
def restore_config(temp_dir) -> Dict[str, Any]:
    """Default test configuration for video restoration."""
    return {
        'project_dir': temp_dir / "project",
        'output_dir': temp_dir / "output",
        'scale_factor': 2,
        'model_name': 'realesrgan-x2plus',
        'crf': 23,
        'preset': 'fast',
        'enable_checkpointing': True,
        'checkpoint_interval': 10,
        'enable_validation': False,
        'enable_disk_validation': False,
        'enable_vram_monitoring': False,
        'max_retries': 1,
        'retry_delay': 0.1,
        'parallel_frames': 1,
        'continue_on_error': True,
    }


@pytest.fixture
def fast_config(temp_dir) -> Dict[str, Any]:
    """Fast configuration for quick integration tests."""
    return {
        'project_dir': temp_dir / "project",
        'output_dir': temp_dir / "output",
        'scale_factor': 2,
        'model_name': 'realesrgan-x2plus',
        'crf': 28,
        'preset': 'ultrafast',
        'enable_checkpointing': False,
        'enable_validation': False,
        'enable_disk_validation': False,
        'enable_vram_monitoring': False,
        'max_retries': 0,
        'retry_delay': 0.0,
        'parallel_frames': 1,
        'continue_on_error': True,
    }


# ============================================================================
# Subprocess mocking fixtures
# ============================================================================

@pytest.fixture
def mock_realesrgan():
    """Mock Real-ESRGAN subprocess calls."""
    def side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        if isinstance(cmd, list) and 'realesrgan-ncnn-vulkan' in cmd[0]:
            # Get input and output paths
            try:
                input_idx = cmd.index('-i') + 1
                output_idx = cmd.index('-o') + 1
                input_path = Path(cmd[input_idx])
                output_path = Path(cmd[output_idx])

                # Copy input to output (simulate enhancement)
                if input_path.exists():
                    shutil.copy(input_path, output_path)

                result = Mock()
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""
                return result
            except (ValueError, IndexError):
                pass

        # Default mock result
        result = Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch('subprocess.run', side_effect=side_effect):
        yield


@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg subprocess calls."""
    def side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        if isinstance(cmd, list):
            cmd_str = ' '.join(cmd)

            # Mock ffprobe output
            if 'ffprobe' in cmd_str:
                result = Mock()
                result.returncode = 0
                result.stdout = '''{
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 480,
                            "height": 360,
                            "r_frame_rate": "24/1"
                        },
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "48000"
                        }
                    ],
                    "format": {
                        "duration": "5.0",
                        "bit_rate": "2000000"
                    }
                }'''
                result.stderr = ""
                return result

            # Mock ffmpeg output
            if 'ffmpeg' in cmd_str:
                result = Mock()
                result.returncode = 0
                result.stdout = ""
                result.stderr = "frame=  120 fps=30.0 q=28.0 size=   500kB"
                return result

        # Default mock result
        result = Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch('subprocess.run', side_effect=side_effect):
        yield


# ============================================================================
# Utility fixtures
# ============================================================================

@pytest.fixture
def ffmpeg_available() -> bool:
    """Check if ffmpeg is available for real video tests."""
    return check_ffmpeg_available()


@pytest.fixture
def skip_if_no_ffmpeg(ffmpeg_available):
    """Skip test if ffmpeg is not available."""
    if not ffmpeg_available:
        pytest.skip("FFmpeg not available")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    try:
        from framewright.utils.gpu import is_gpu_available
        if not is_gpu_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("GPU utilities not available")


# ============================================================================
# Original fixtures (kept for backward compatibility)
# ============================================================================


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
