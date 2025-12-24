"""Tests for VideoRestorer class."""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pytest

from framewright.config import Config
from framewright.restorer import (
    VideoRestorer,
    VideoRestorerError,
    DownloadError,
    MetadataError,
    AudioExtractionError,
    FrameExtractionError,
    EnhancementError,
    ReassemblyError,
)


class TestVideoRestorerInitialization:
    """Test VideoRestorer initialization."""

    def test_init_with_valid_config(self, project_dir, mock_shutil_which):
        """Test successful initialization with valid config."""
        config = Config(project_dir=project_dir, scale_factor=4)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        assert restorer.config == config
        assert restorer.metadata == {}
        assert restorer.progress_callback is None

    def test_init_with_progress_callback(self, project_dir, mock_shutil_which):
        """Test initialization with progress callback."""
        config = Config(project_dir=project_dir)
        callback = Mock()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config, progress_callback=callback)

        assert restorer.progress_callback == callback

    def test_init_missing_dependencies(self, project_dir):
        """Test initialization fails with missing dependencies."""
        config = Config(project_dir=project_dir)

        with patch('shutil.which', return_value=None):
            with pytest.raises(VideoRestorerError) as exc_info:
                VideoRestorer(config)

            assert "Missing required tools" in str(exc_info.value)
            assert "yt-dlp" in str(exc_info.value)

    def test_init_partial_dependencies(self, project_dir):
        """Test initialization with only some tools available."""
        config = Config(project_dir=project_dir)

        def partial_which(command):
            return '/usr/bin/ffmpeg' if command == 'ffmpeg' else None

        with patch('shutil.which', side_effect=partial_which):
            with pytest.raises(VideoRestorerError) as exc_info:
                VideoRestorer(config)

            assert "Missing required tools" in str(exc_info.value)


class TestVideoRestorerDirectoryCreation:
    """Test directory creation and management."""

    def test_config_creates_directories(self, project_dir, mock_shutil_which):
        """Test that config creates necessary directories."""
        config = Config(project_dir=project_dir, scale_factor=2, model_name='realesrgan-x2plus')

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)
            restorer.config.create_directories()

        assert config.project_dir.exists()
        assert config.temp_dir.exists()
        assert config.frames_dir.exists()
        assert config.enhanced_dir.exists()


class TestMetadataExtraction:
    """Test metadata extraction with mocked ffprobe."""

    def test_analyze_metadata_success(
        self, project_dir, video_file, mock_shutil_which,
        mock_ffprobe_output, expected_metadata
    ):
        """Test successful metadata extraction."""
        config = Config(project_dir=project_dir)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        mock_result = Mock()
        mock_result.stdout = mock_ffprobe_output
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            metadata = restorer.analyze_metadata(video_file)

        assert metadata == expected_metadata
        assert restorer.metadata == expected_metadata

    def test_analyze_metadata_no_video_stream(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test metadata extraction fails with no video stream."""
        config = Config(project_dir=project_dir)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        mock_result = Mock()
        mock_result.stdout = '{"streams": [], "format": {}}'

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(MetadataError) as exc_info:
                restorer.analyze_metadata(video_file)

            assert "No video stream found" in str(exc_info.value)

    def test_analyze_metadata_ffprobe_failure(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test metadata extraction handles ffprobe failure."""
        config = Config(project_dir=project_dir)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(
            1, 'ffprobe', stderr='Error'
        )):
            with pytest.raises(MetadataError) as exc_info:
                restorer.analyze_metadata(video_file)

            assert "Failed to extract metadata" in str(exc_info.value)

    def test_analyze_metadata_invalid_json(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test metadata extraction handles invalid JSON."""
        config = Config(project_dir=project_dir)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        mock_result = Mock()
        mock_result.stdout = 'invalid json'

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(MetadataError) as exc_info:
                restorer.analyze_metadata(video_file)

            assert "Failed to parse metadata" in str(exc_info.value)

    def test_analyze_metadata_no_audio(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test metadata extraction with no audio stream."""
        config = Config(project_dir=project_dir)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        mock_output = '''{
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "30/1"
                }
            ],
            "format": {
                "duration": "60.0",
                "bit_rate": "3000000"
            }
        }'''

        mock_result = Mock()
        mock_result.stdout = mock_output

        with patch('subprocess.run', return_value=mock_result):
            metadata = restorer.analyze_metadata(video_file)

        assert metadata['has_audio'] is False
        assert metadata['audio_codec'] is None
        assert metadata['audio_sample_rate'] is None


class TestFrameExtraction:
    """Test frame extraction command building."""

    def test_extract_frames_command_building(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test that extract_frames builds correct ffmpeg command."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        # Create mock frames after "extraction"
        def create_mock_frames(*args, **kwargs):
            for i in range(1, 6):
                frame_file = config.frames_dir / f"frame_{i:08d}.png"
                frame_file.write_text(f"frame {i}")

            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        with patch('subprocess.run', side_effect=create_mock_frames) as mock_run:
            frame_count = restorer.extract_frames(video_file)

        assert frame_count == 5

        # Verify ffmpeg command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == 'ffmpeg'
        assert '-i' in call_args
        assert str(video_file) in call_args
        assert '-qscale:v' in call_args
        assert '1' in call_args

    def test_extract_frames_no_frames_created(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test extraction fails when no frames are created."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        mock_result = Mock()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(FrameExtractionError) as exc_info:
                restorer.extract_frames(video_file)

            assert "No frames were extracted" in str(exc_info.value)

    def test_extract_frames_ffmpeg_failure(
        self, project_dir, video_file, mock_shutil_which
    ):
        """Test extraction handles ffmpeg failure."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(
            1, 'ffmpeg', stderr='Error'
        )):
            with pytest.raises(FrameExtractionError) as exc_info:
                restorer.extract_frames(video_file)

            assert "Failed to extract frames" in str(exc_info.value)


class TestEnhancementCommandBuilding:
    """Test enhancement command building."""

    def test_enhance_frames_command_building(
        self, project_dir, frames_dir, mock_shutil_which
    ):
        """Test that enhance_frames builds correct realesrgan command."""
        config = Config(project_dir=project_dir, scale_factor=4)
        config.create_directories()

        # Copy frames to config frames_dir
        import shutil
        shutil.copytree(frames_dir, config.frames_dir, dirs_exist_ok=True)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        # Create mock enhanced frames
        def create_enhanced_frames(*args, **kwargs):
            process_mock = MagicMock()
            process_mock.stdout = iter(["done"] * 10)
            process_mock.wait.return_value = 0

            # Create enhanced frames
            for i in range(1, 11):
                enhanced_file = config.enhanced_dir / f"frame_{i:08d}.png"
                enhanced_file.write_text(f"enhanced {i}")

            return process_mock

        with patch('subprocess.Popen', side_effect=create_enhanced_frames) as mock_popen:
            enhanced_count = restorer.enhance_frames()

        assert enhanced_count == 10

        # Verify realesrgan command
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == 'realesrgan-ncnn-vulkan'
        assert '-i' in call_args
        assert '-o' in call_args
        assert '-n' in call_args
        assert 'realesrgan-x4plus' in call_args
        assert '-s' in call_args
        assert '4' in call_args

    def test_enhance_frames_no_input_frames(
        self, project_dir, mock_shutil_which
    ):
        """Test enhancement fails with no input frames."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        with pytest.raises(EnhancementError) as exc_info:
            restorer.enhance_frames()

        assert "No frames found to enhance" in str(exc_info.value)

    def test_enhance_frames_process_failure(
        self, project_dir, frames_dir, mock_shutil_which
    ):
        """Test enhancement handles process failure."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        import shutil
        shutil.copytree(frames_dir, config.frames_dir, dirs_exist_ok=True)

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        process_mock = MagicMock()
        process_mock.stdout = iter([])
        process_mock.wait.return_value = 1

        with patch('subprocess.Popen', return_value=process_mock):
            with pytest.raises(EnhancementError) as exc_info:
                restorer.enhance_frames()

            assert "exited with code 1" in str(exc_info.value)


class TestReassemblyCommandBuilding:
    """Test video reassembly command building."""

    def test_reassemble_video_with_audio(
        self, project_dir, audio_file, mock_shutil_which
    ):
        """Test reassembly command with audio."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        # Create mock enhanced frames
        for i in range(1, 6):
            frame_file = config.enhanced_dir / f"frame_{i:08d}.png"
            frame_file.write_text(f"enhanced {i}")

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        restorer.metadata = {'framerate': 30.0}

        output_path = project_dir / "output.mkv"

        def create_output(*args, **kwargs):
            output_path.write_text("output video")
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        with patch('subprocess.run', side_effect=create_output) as mock_run:
            result = restorer.reassemble_video(audio_path=audio_file, output_path=output_path)

        assert result == output_path

        # Verify ffmpeg command includes audio
        call_args = mock_run.call_args[0][0]
        assert 'ffmpeg' in call_args
        assert str(audio_file) in call_args
        assert '-c:a' in call_args
        assert 'flac' in call_args

    def test_reassemble_video_without_audio(
        self, project_dir, mock_shutil_which
    ):
        """Test reassembly command without audio."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        # Create mock enhanced frames
        for i in range(1, 4):
            frame_file = config.enhanced_dir / f"frame_{i:08d}.png"
            frame_file.write_text(f"enhanced {i}")

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        restorer.metadata = {'framerate': 24.0}

        output_path = project_dir / "output.mkv"

        def create_output(*args, **kwargs):
            output_path.write_text("output video")
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        with patch('subprocess.run', side_effect=create_output) as mock_run:
            result = restorer.reassemble_video(output_path=output_path)

        assert result == output_path

        # Verify ffmpeg command excludes audio
        call_args = mock_run.call_args[0][0]
        assert 'ffmpeg' in call_args
        assert '-c:v' in call_args
        assert 'libx265' in call_args

    def test_reassemble_video_no_enhanced_frames(
        self, project_dir, mock_shutil_which
    ):
        """Test reassembly fails with no enhanced frames."""
        config = Config(project_dir=project_dir)
        config.create_directories()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        with pytest.raises(ReassemblyError) as exc_info:
            restorer.reassemble_video()

        assert "No enhanced frames found" in str(exc_info.value)


class TestFullPipeline:
    """Test full restoration pipeline with mocks."""

    def test_full_pipeline_local_file(
        self, project_dir, video_file, mock_shutil_which, mock_ffprobe_output
    ):
        """Test complete pipeline with local video file."""
        config = Config(project_dir=project_dir, scale_factor=2, model_name='realesrgan-x2plus')

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        # Mock all subprocess calls
        def mock_subprocess_side_effect(*args, **kwargs):
            cmd = args[0]

            # ffprobe call
            if 'ffprobe' in cmd:
                mock_result = Mock()
                mock_result.stdout = mock_ffprobe_output
                mock_result.returncode = 0
                return mock_result

            # ffmpeg frame extraction
            if '-qscale:v' in cmd:
                for i in range(1, 4):
                    frame_file = config.frames_dir / f"frame_{i:08d}.png"
                    frame_file.parent.mkdir(parents=True, exist_ok=True)
                    frame_file.write_text(f"frame {i}")
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stderr = ""
                return mock_result

            # ffmpeg reassembly
            if 'libx265' in cmd:
                output_idx = cmd.index('-y') + 1
                output_path = Path(cmd[output_idx])
                output_path.write_text("output video")
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stderr = ""
                return mock_result

            # Audio extraction
            if '-acodec' in cmd and 'pcm_s24le' in cmd:
                output_idx = cmd.index('-y') + 1
                audio_path = Path(cmd[output_idx])
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_text("audio")
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stderr = ""
                return mock_result

            mock_result = Mock()
            mock_result.returncode = 0
            return mock_result

        # Mock Popen for realesrgan
        def mock_popen_side_effect(*args, **kwargs):
            process_mock = MagicMock()
            process_mock.stdout = iter(["done"] * 3)
            process_mock.wait.return_value = 0

            # Create enhanced frames
            for i in range(1, 4):
                enhanced_file = config.enhanced_dir / f"frame_{i:08d}.png"
                enhanced_file.parent.mkdir(parents=True, exist_ok=True)
                enhanced_file.write_text(f"enhanced {i}")

            return process_mock

        with patch('subprocess.run', side_effect=mock_subprocess_side_effect):
            with patch('subprocess.Popen', side_effect=mock_popen_side_effect):
                result = restorer.restore_video(str(video_file), cleanup=False)

        assert result.exists()
        assert result.suffix == '.mkv'

    def test_full_pipeline_with_cleanup(
        self, project_dir, video_file, mock_shutil_which, mock_ffprobe_output
    ):
        """Test pipeline cleans up temp files when requested."""
        config = Config(project_dir=project_dir, scale_factor=2, model_name='realesrgan-x2plus')

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config)

        # Use same mocks as previous test
        def mock_subprocess_side_effect(*args, **kwargs):
            cmd = args[0]

            if 'ffprobe' in cmd:
                mock_result = Mock()
                mock_result.stdout = mock_ffprobe_output
                return mock_result

            if '-qscale:v' in cmd:
                for i in range(1, 3):
                    frame_file = config.frames_dir / f"frame_{i:08d}.png"
                    frame_file.parent.mkdir(parents=True, exist_ok=True)
                    frame_file.write_text(f"frame {i}")
                mock_result = Mock()
                mock_result.stderr = ""
                return mock_result

            if 'libx265' in cmd:
                output_idx = cmd.index('-y') + 1
                output_path = Path(cmd[output_idx])
                output_path.write_text("output")
                mock_result = Mock()
                mock_result.stderr = ""
                return mock_result

            if '-acodec' in cmd:
                output_idx = cmd.index('-y') + 1
                audio_path = Path(cmd[output_idx])
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_text("audio")
                mock_result = Mock()
                mock_result.stderr = ""
                return mock_result

            return Mock()

        def mock_popen_side_effect(*args, **kwargs):
            process_mock = MagicMock()
            process_mock.stdout = iter(["done"] * 2)
            process_mock.wait.return_value = 0

            for i in range(1, 3):
                enhanced_file = config.enhanced_dir / f"frame_{i:08d}.png"
                enhanced_file.parent.mkdir(parents=True, exist_ok=True)
                enhanced_file.write_text(f"enhanced {i}")

            return process_mock

        with patch('subprocess.run', side_effect=mock_subprocess_side_effect):
            with patch('subprocess.Popen', side_effect=mock_popen_side_effect):
                result = restorer.restore_video(str(video_file), cleanup=True)

        # Verify temp directory was cleaned up
        assert not config.temp_dir.exists()


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_called(
        self, project_dir, video_file, mock_shutil_which, mock_ffprobe_output
    ):
        """Test that progress callback is called during operations."""
        config = Config(project_dir=project_dir)
        callback = Mock()

        with patch('shutil.which', side_effect=mock_shutil_which):
            restorer = VideoRestorer(config, progress_callback=callback)

        mock_result = Mock()
        mock_result.stdout = mock_ffprobe_output

        with patch('subprocess.run', return_value=mock_result):
            restorer.analyze_metadata(video_file)

        # Verify callback was called with stage and progress
        assert callback.call_count >= 2  # Start and end

        # Check that it was called with proper arguments
        calls = callback.call_args_list
        assert calls[0][0][0] == "analyze_metadata"
        assert calls[0][0][1] == 0.0
        assert calls[-1][0][1] == 1.0
