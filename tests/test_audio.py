"""Tests for AudioProcessor class."""
import subprocess
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pytest

from framewright.processors.audio import AudioProcessor, AudioProcessorError


class TestAudioProcessorInitialization:
    """Test AudioProcessor initialization."""

    def test_init_with_ffmpeg_available(self):
        """Test successful initialization when FFmpeg is available."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        assert processor is not None

    def test_init_ffmpeg_not_found(self):
        """Test initialization fails when FFmpeg is not found."""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            with pytest.raises(AudioProcessorError) as exc_info:
                AudioProcessor()

            assert "FFmpeg is not installed" in str(exc_info.value)

    def test_init_ffmpeg_error(self):
        """Test initialization fails when FFmpeg command fails."""
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(
            1, 'ffmpeg', stderr='Error'
        )):
            with pytest.raises(AudioProcessorError) as exc_info:
                AudioProcessor()

            assert "FFmpeg is not installed" in str(exc_info.value)


class TestAudioProcessorExtract:
    """Test audio extraction functionality."""

    def test_extract_command_building(self, video_file, temp_dir):
        """Test that extract builds correct FFmpeg command."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            processor = AudioProcessor()

        output_path = temp_dir / "audio.wav"

        # Reset mock for extract call
        mock_run.reset_mock()

        # Mock Popen for extraction
        process_mock = MagicMock()
        process_mock.stderr = iter(["Processing...", "Done"])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            # Create output file in mock
            output_path.write_text("audio data")
            processor.extract(str(video_file), str(output_path))

        # Verify FFmpeg command
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == 'ffmpeg'
        assert '-i' in call_args
        assert str(video_file) in call_args
        assert '-vn' in call_args
        assert '-acodec' in call_args
        assert 'pcm_s16le' in call_args
        assert '-ar' in call_args
        assert '48000' in call_args
        assert '-ac' in call_args
        assert '2' in call_args
        assert str(output_path) in call_args

    def test_extract_creates_output_dir(self, video_file, temp_dir):
        """Test that extract creates output directory if it doesn't exist."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "subdir" / "nested" / "audio.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("audio")
            processor.extract(str(video_file), str(output_path))

        assert output_path.parent.exists()

    def test_extract_invalid_input(self, temp_dir):
        """Test extract fails with invalid input file."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        invalid_path = temp_dir / "nonexistent.mp4"
        output_path = temp_dir / "audio.wav"

        with pytest.raises(AudioProcessorError) as exc_info:
            processor.extract(str(invalid_path), str(output_path))

        assert "does not exist" in str(exc_info.value)

    def test_extract_ffmpeg_failure(self, video_file, temp_dir):
        """Test extract handles FFmpeg failure."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "audio.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter(["Error occurred"])
        process_mock.wait.return_value = 1
        process_mock.communicate.return_value = ("", "FFmpeg error")

        with patch('subprocess.Popen', return_value=process_mock):
            with pytest.raises(AudioProcessorError) as exc_info:
                processor.extract(str(video_file), str(output_path))

            assert "failed with return code 1" in str(exc_info.value)

    def test_extract_with_progress_callback(self, video_file, temp_dir):
        """Test extract calls progress callback."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "audio.wav"
        callback = Mock()

        process_mock = MagicMock()
        process_mock.stderr = iter(["Frame 1", "Frame 2", "Done"])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock):
            output_path.write_text("audio")
            processor.extract(str(video_file), str(output_path), progress_callback=callback)

        # Verify callback was called
        assert callback.call_count == 3


class TestAudioProcessorEnhance:
    """Test audio enhancement functionality."""

    def test_enhance_filters(self, audio_file, temp_dir):
        """Test enhance applies correct filters."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "enhanced.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("enhanced audio")
            processor.enhance(str(audio_file), str(output_path))

        # Verify FFmpeg command includes filters
        call_args = mock_popen.call_args[0][0]
        assert 'ffmpeg' in call_args
        assert '-af' in call_args

        # Find the filter argument
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert 'highpass' in filters
        assert 'lowpass' in filters
        assert 'loudnorm' in filters


class TestAudioProcessorNormalize:
    """Test audio normalization functionality."""

    def test_normalize_settings(self, audio_file, temp_dir):
        """Test normalize applies correct settings."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "normalized.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("normalized audio")
            processor.normalize(str(audio_file), str(output_path), target_loudness=-18.0)

        # Verify loudnorm filter with target
        call_args = mock_popen.call_args[0][0]
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert 'loudnorm' in filters
        assert 'I=-18.0' in filters

    @pytest.mark.parametrize("target_loudness", [-16.0, -18.0, -23.0])
    def test_normalize_different_targets(self, audio_file, temp_dir, target_loudness):
        """Test normalize with different target loudness values."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "normalized.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("normalized audio")
            processor.normalize(str(audio_file), str(output_path), target_loudness=target_loudness)

        call_args = mock_popen.call_args[0][0]
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert f'I={target_loudness}' in filters


class TestAudioProcessorDenoise:
    """Test audio denoising functionality."""

    def test_denoise_with_default_noise_floor(self, audio_file, temp_dir):
        """Test denoise with default noise floor."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "denoised.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("denoised audio")
            processor.denoise(str(audio_file), str(output_path))

        call_args = mock_popen.call_args[0][0]
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert 'afftdn' in filters
        assert 'nr=' in filters
        assert 'nf=' in filters

    @pytest.mark.parametrize("noise_floor", [-10.0, -20.0, -30.0])
    def test_denoise_different_noise_floors(self, audio_file, temp_dir, noise_floor):
        """Test denoise with different noise floor values."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "denoised.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("denoised audio")
            processor.denoise(str(audio_file), str(output_path), noise_floor=noise_floor)

        call_args = mock_popen.call_args[0][0]
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert f'nf={noise_floor}' in filters


class TestAudioProcessorApplyFilters:
    """Test frequency filtering functionality."""

    def test_apply_filters_default(self, audio_file, temp_dir):
        """Test apply_filters with default values."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "filtered.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("filtered audio")
            processor.apply_filters(str(audio_file), str(output_path))

        call_args = mock_popen.call_args[0][0]
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert 'highpass=f=80' in filters
        assert 'lowpass=f=12000' in filters

    @pytest.mark.parametrize("highpass,lowpass", [
        (100, 15000),
        (50, 10000),
        (200, 8000),
    ])
    def test_apply_filters_custom_frequencies(self, audio_file, temp_dir, highpass, lowpass):
        """Test apply_filters with custom frequency values."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "filtered.wav"

        process_mock = MagicMock()
        process_mock.stderr = iter([])
        process_mock.wait.return_value = 0

        with patch('subprocess.Popen', return_value=process_mock) as mock_popen:
            output_path.write_text("filtered audio")
            processor.apply_filters(str(audio_file), str(output_path), highpass=highpass, lowpass=lowpass)

        call_args = mock_popen.call_args[0][0]
        af_index = call_args.index('-af')
        filters = call_args[af_index + 1]

        assert f'highpass=f={highpass}' in filters
        assert f'lowpass=f={lowpass}' in filters

    def test_apply_filters_invalid_range(self, audio_file, temp_dir):
        """Test apply_filters fails when highpass >= lowpass."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "filtered.wav"

        with pytest.raises(AudioProcessorError) as exc_info:
            processor.apply_filters(str(audio_file), str(output_path), highpass=15000, lowpass=100)

        assert "must be less than" in str(exc_info.value)


class TestAudioProcessorErrorHandling:
    """Test error handling across all methods."""

    def test_validate_input_file_not_exists(self):
        """Test validation fails for nonexistent file."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        with pytest.raises(AudioProcessorError) as exc_info:
            processor._validate_input_file("/nonexistent/file.wav")

        assert "does not exist" in str(exc_info.value)

    def test_validate_input_file_is_directory(self, temp_dir):
        """Test validation fails for directory."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        dir_path = temp_dir / "testdir"
        dir_path.mkdir()

        with pytest.raises(AudioProcessorError) as exc_info:
            processor._validate_input_file(str(dir_path))

        assert "is not a file" in str(exc_info.value)

    def test_subprocess_error_handling(self, audio_file, temp_dir):
        """Test handling of subprocess errors."""
        mock_result = Mock()
        mock_result.stdout = "ffmpeg version 4.4.2"

        with patch('subprocess.run', return_value=mock_result):
            processor = AudioProcessor()

        output_path = temp_dir / "output.wav"

        with patch('subprocess.Popen', side_effect=subprocess.SubprocessError("Error")):
            with pytest.raises(AudioProcessorError) as exc_info:
                processor.enhance(str(audio_file), str(output_path))

            assert "Failed to execute FFmpeg" in str(exc_info.value)
