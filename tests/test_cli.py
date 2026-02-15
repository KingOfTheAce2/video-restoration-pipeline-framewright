"""Tests for CLI interface."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from framewright.cli import (
    create_parser,
    validate_input,
    validate_scale,
    restore_video,
    extract_frames,
    enhance_frames,
    _enhance_with_realesrgan,
    _enhance_with_hat,
    _enhance_with_diffusion,
    _enhance_with_ensemble,
    reassemble_video,
    enhance_audio,
    main,
)


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_create_parser_restore_command(self):
        """Test parser handles restore command arguments."""
        parser = create_parser()

        args = parser.parse_args([
            'restore',
            '--input', 'video.mp4',
            '--output', 'output.mp4',
            '--scale', '4',
            '--model', 'realesrgan-x4plus',
        ])

        assert args.command == 'restore'
        assert args.input == 'video.mp4'
        assert args.output == 'output.mp4'
        assert args.scale == 4
        assert args.model == 'realesrgan-x4plus'
        assert args.audio_enhance is False

    def test_create_parser_restore_with_url(self):
        """Test parser handles restore with URL."""
        parser = create_parser()

        args = parser.parse_args([
            'restore',
            '--url', 'https://youtube.com/watch?v=xxx',
            '--output', 'output.mp4',
        ])

        assert args.command == 'restore'
        assert args.url == 'https://youtube.com/watch?v=xxx'
        assert args.output == 'output.mp4'

    def test_create_parser_restore_with_audio_enhance(self):
        """Test parser handles audio enhancement flag."""
        parser = create_parser()

        args = parser.parse_args([
            'restore',
            '--input', 'video.mp4',
            '--output', 'output.mp4',
            '--audio-enhance',
        ])

        assert args.audio_enhance is True

    def test_create_parser_extract_frames_command(self):
        """Test parser handles extract-frames command."""
        parser = create_parser()

        args = parser.parse_args([
            'extract-frames',
            '--input', 'video.mp4',
            '--output', 'frames/',
        ])

        assert args.command == 'extract-frames'
        assert args.input == 'video.mp4'
        assert args.output == 'frames/'

    def test_create_parser_enhance_frames_command(self):
        """Test parser handles enhance-frames command."""
        parser = create_parser()

        args = parser.parse_args([
            'enhance-frames',
            '--input', 'frames/',
            '--output', 'enhanced/',
            '--scale', '2',
            '--model', 'realesrgan-x2plus',
        ])

        assert args.command == 'enhance-frames'
        assert args.input == 'frames/'
        assert args.output == 'enhanced/'
        assert args.scale == 2
        assert args.model == 'realesrgan-x2plus'

    def test_create_parser_reassemble_command(self):
        """Test parser handles reassemble command."""
        parser = create_parser()

        args = parser.parse_args([
            'reassemble',
            '--frames-dir', 'enhanced/',
            '--audio', 'original.mp4',
            '--output', 'final.mp4',
            '--quality', '20',
        ])

        assert args.command == 'reassemble'
        assert args.frames_dir == 'enhanced/'
        assert args.audio == 'original.mp4'
        assert args.output == 'final.mp4'
        assert args.quality == 20

    def test_create_parser_audio_enhance_command(self):
        """Test parser handles audio-enhance command."""
        parser = create_parser()

        args = parser.parse_args([
            'audio-enhance',
            '--input', 'video.mp4',
            '--output', 'audio.wav',
        ])

        assert args.command == 'audio-enhance'
        assert args.input == 'video.mp4'
        assert args.output == 'audio.wav'

    def test_create_parser_no_command(self):
        """Test parser with no command."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.command is None

    @pytest.mark.parametrize("scale", [2, 4])
    def test_create_parser_valid_scale_choices(self, scale):
        """Test parser accepts valid scale values."""
        parser = create_parser()

        args = parser.parse_args([
            'restore',
            '--input', 'video.mp4',
            '--output', 'output.mp4',
            '--scale', str(scale),
        ])

        assert args.scale == scale

    def test_create_parser_invalid_scale_choice(self):
        """Test parser rejects invalid scale values."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([
                'restore',
                '--input', 'video.mp4',
                '--output', 'output.mp4',
                '--scale', '3',
            ])


class TestInputValidation:
    """Test input validation functions."""

    def test_validate_input_local_file_exists(self, video_file):
        """Test validate_input with existing local file."""
        result = validate_input(str(video_file))

        assert result == video_file
        assert isinstance(result, Path)

    def test_validate_input_local_file_not_exists(self):
        """Test validate_input with nonexistent file exits."""
        with pytest.raises(SystemExit) as exc_info:
            validate_input('/nonexistent/file.mp4')

        assert exc_info.value.code == 1

    def test_validate_input_http_url(self):
        """Test validate_input with HTTP URL."""
        url = 'http://example.com/video.mp4'
        result = validate_input(url)

        # Path() removes extra slashes, so compare without them
        assert 'example.com/video.mp4' in str(result)

    def test_validate_input_https_url(self):
        """Test validate_input with HTTPS URL."""
        url = 'https://youtube.com/watch?v=xxx'
        result = validate_input(url)

        # Path() removes extra slashes, so compare without them
        assert 'youtube.com/watch?v=xxx' in str(result)


class TestScaleValidation:
    """Test scale validation function."""

    @pytest.mark.parametrize("scale", [2, 4])
    def test_validate_scale_valid(self, scale):
        """Test validate_scale with valid values."""
        result = validate_scale(scale)
        assert result == scale

    @pytest.mark.parametrize("scale", [1, 3, 8, 0, -1])
    def test_validate_scale_invalid(self, scale):
        """Test validate_scale with invalid values exits."""
        with pytest.raises(SystemExit) as exc_info:
            validate_scale(scale)

        assert exc_info.value.code == 1


class TestCommandDispatch:
    """Test command dispatch functions."""

    def test_restore_video_command(self, temp_dir):
        """Test restore_video command function."""
        mock_args = Mock()
        mock_args.input = None
        mock_args.url = 'https://example.com/video.mp4'
        mock_args.output = str(temp_dir / 'output.mp4')
        mock_args.output_dir = str(temp_dir)
        mock_args.scale = 2
        mock_args.model = 'realesrgan-x2plus'
        mock_args.audio_enhance = False
        mock_args.dry_run = True
        mock_args.format = None
        mock_args.user_profile = None
        mock_args.quality = 18
        mock_args.enable_rife = False
        mock_args.target_fps = None
        mock_args.auto_enhance = False

        # dry_run mode with URL avoids actual processing
        restore_video(mock_args)

    def test_extract_frames_command_valid_input(self, video_file, temp_dir):
        """Test extract_frames command with valid input."""
        mock_args = Mock()
        mock_args.input = str(video_file)
        mock_args.output = str(temp_dir / "frames")

        with patch('framewright.utils.ffmpeg.extract_frames_to_dir', return_value=10):
            with patch('framewright.utils.ffmpeg.probe_video', return_value={
                'width': 480, 'height': 360, 'framerate': 24.0, 'duration': 5.0
            }):
                extract_frames(mock_args)

        # Verify output directory was created
        assert Path(mock_args.output).exists()

    def test_extract_frames_command_invalid_input(self):
        """Test extract_frames command with invalid input."""
        mock_args = Mock()
        mock_args.input = '/nonexistent/video.mp4'
        mock_args.output = '/tmp/frames'

        with pytest.raises(SystemExit):
            extract_frames(mock_args)

    def test_enhance_frames_command_valid_dir(self, frames_dir, temp_dir):
        """Test enhance_frames command with valid directory."""
        import numpy as np
        mock_args = Mock()
        mock_args.input = str(frames_dir)
        mock_args.output = str(temp_dir / "enhanced")
        mock_args.scale = 2
        mock_args.model = 'realesrgan-x2plus'

        # Mock Real-ESRGAN model loading and cv2 operations
        mock_output = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_upsampler = Mock()
        mock_upsampler.enhance.return_value = (mock_output, None)
        mock_realesrgan_cls = Mock(return_value=mock_upsampler)
        mock_rrdbnet = Mock()

        with patch.dict('sys.modules', {
            'realesrgan': type('mod', (), {'RealESRGANer': mock_realesrgan_cls})(),
            'basicsr': Mock(),
            'basicsr.archs': Mock(),
            'basicsr.archs.rrdbnet_arch': type('mod', (), {'RRDBNet': mock_rrdbnet})(),
        }):
            with patch('cv2.imread', return_value=np.zeros((240, 320, 3), dtype=np.uint8)):
                with patch('cv2.imwrite', return_value=True):
                    enhance_frames(mock_args)

        # Verify output directory was created
        assert Path(mock_args.output).exists()

    def test_enhance_frames_command_invalid_dir(self, temp_dir):
        """Test enhance_frames command with invalid directory."""
        mock_args = Mock()
        mock_args.input = str(temp_dir / "nonexistent")
        mock_args.output = str(temp_dir / "enhanced")
        mock_args.scale = 2
        mock_args.model = 'realesrgan-x2plus'

        with pytest.raises(SystemExit):
            enhance_frames(mock_args)

    def test_enhance_frames_command_invalid_scale(self, frames_dir, temp_dir):
        """Test enhance_frames command validates scale."""
        mock_args = Mock()
        mock_args.input = str(frames_dir)
        mock_args.output = str(temp_dir / "enhanced")
        mock_args.scale = 3  # Invalid scale
        mock_args.model = 'realesrgan-x4plus'

        with pytest.raises(SystemExit):
            enhance_frames(mock_args)

    def test_reassemble_video_command_valid(self, frames_dir, temp_dir):
        """Test reassemble_video command with valid frames directory."""
        mock_args = Mock()
        mock_args.frames_dir = str(frames_dir)
        mock_args.audio = None
        mock_args.output = str(temp_dir / "output.mp4")
        mock_args.quality = 23
        mock_args.fps = None

        with patch('framewright.utils.ffmpeg.reassemble_from_frames'):
            reassemble_video(mock_args)

    def test_reassemble_video_command_invalid_frames_dir(self, temp_dir):
        """Test reassemble_video command with invalid frames directory."""
        mock_args = Mock()
        mock_args.frames_dir = str(temp_dir / "nonexistent")
        mock_args.audio = None
        mock_args.output = str(temp_dir / "output.mp4")

        with pytest.raises(SystemExit):
            reassemble_video(mock_args)

    def test_enhance_audio_command_valid(self, video_file, temp_dir):
        """Test enhance_audio command with valid input."""
        mock_args = Mock()
        mock_args.input = str(video_file)
        mock_args.output = str(temp_dir / "audio.wav")

        # Mock audio processor to avoid import and processing issues
        mock_processor = Mock()
        mock_audio_mod = type("mod", (), {"AudioProcessor": Mock(return_value=mock_processor)})()
        with patch.dict("sys.modules", {"framewright.processors.audio": mock_audio_mod}):
            enhance_audio(mock_args)

    def test_enhance_audio_command_invalid(self):
        """Test enhance_audio command with invalid input."""
        mock_args = Mock()
        mock_args.input = '/nonexistent/video.mp4'
        mock_args.output = '/tmp/audio.wav'

        with pytest.raises(SystemExit):
            enhance_audio(mock_args)


class TestMainFunction:
    """Test main CLI entry point."""

    def test_main_no_args_shows_help(self):
        """Test main with no arguments shows help."""
        with patch('sys.argv', ['framewright']):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_valid_command(self, video_file, temp_dir):
        """Test main with valid command."""
        with patch('sys.argv', [
            'framewright',
            'extract-frames',
            '--input', str(video_file),
            '--output', str(temp_dir / "frames"),
        ]):
            with patch('framewright.utils.ffmpeg.probe_video', return_value={
                'width': 480, 'height': 360, 'framerate': 24.0, 'duration': 5.0
            }):
                with patch('framewright.utils.ffmpeg.extract_frames_to_dir', return_value=10):
                    main()

    def test_main_keyboard_interrupt(self):
        """Test main handles keyboard interrupt."""
        with patch('sys.argv', [
            'framewright',
            'restore',
            '--input', 'video.mp4',
            '--output', 'output.mp4',
        ]):
            with patch('framewright.cli.restore_video', side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    def test_main_exception_handling(self):
        """Test main handles general exceptions."""
        with patch('sys.argv', [
            'framewright',
            'restore',
            '--input', 'video.mp4',
            '--output', 'output.mp4',
        ]):
            with patch('framewright.cli.restore_video', side_effect=Exception("Test error")):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1


class TestErrorMessages:
    """Test error message formatting."""

    def test_input_file_not_found_message(self, capsys):
        """Test error message for missing input file."""
        with pytest.raises(SystemExit):
            validate_input('/nonexistent/file.mp4')

        captured = capsys.readouterr()
        assert "Error: Input file not found" in captured.out

    def test_invalid_scale_message(self, capsys):
        """Test error message for invalid scale."""
        with pytest.raises(SystemExit):
            validate_scale(3)

        captured = capsys.readouterr()
        assert "Error: Scale must be 2 or 4" in captured.out

    def test_directory_not_found_message(self, temp_dir, capsys):
        """Test error message for missing directory."""
        mock_args = Mock()
        mock_args.input = str(temp_dir / "nonexistent")
        mock_args.output = str(temp_dir / "output")
        mock_args.scale = 2
        mock_args.model = 'realesrgan-x2plus'

        with pytest.raises(SystemExit):
            enhance_frames(mock_args)

        captured = capsys.readouterr()
        assert "Error: Input directory not found" in captured.out


class TestColoredOutput:
    """Test colored output functions."""

    def test_print_colored_output(self, capsys):
        """Test print_colored produces output."""
        from framewright.cli import print_colored, Colors

        print_colored("Test message", Colors.OKGREEN)
        captured = capsys.readouterr()

        assert "Test message" in captured.out

    def test_print_header_output(self, capsys):
        """Test print_header produces output."""
        from framewright.cli import print_header

        print_header()
        captured = capsys.readouterr()

        assert "FrameWright" in captured.out
        assert "v1.0.0" in captured.out


class TestEnhanceFramesDispatch:
    """Test enhance_frames backend dispatch and new CLI arguments."""

    # -- Argument parsing --

    def test_parse_hat_args(self):
        """Test parser accepts HAT-specific arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'enhance-frames', '--input', 'f/', '--output', 'o/',
            '--model', 'hat', '--hat-size', 'large',
        ])
        assert args.model == 'hat'
        assert args.hat_size == 'large'

    def test_parse_diffusion_args(self):
        """Test parser accepts diffusion-specific arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'enhance-frames', '--input', 'f/', '--output', 'o/',
            '--model', 'diffusion', '--diffusion-steps', '15',
            '--diffusion-model', 'resshift',
        ])
        assert args.model == 'diffusion'
        assert args.diffusion_steps == 15
        assert args.diffusion_model == 'resshift'

    def test_parse_ensemble_args(self):
        """Test parser accepts ensemble-specific arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'enhance-frames', '--input', 'f/', '--output', 'o/',
            '--model', 'ensemble', '--ensemble-models', 'hat,realesrgan',
            '--ensemble-method', 'adaptive',
        ])
        assert args.model == 'ensemble'
        assert args.ensemble_models == 'hat,realesrgan'
        assert args.ensemble_method == 'adaptive'

    def test_parse_defaults(self):
        """Test new args have correct defaults."""
        parser = create_parser()
        args = parser.parse_args([
            'enhance-frames', '--input', 'f/', '--output', 'o/',
        ])
        assert args.hat_size == 'large'
        assert args.diffusion_steps == 20
        assert args.diffusion_model == 'upscale_a_video'
        assert args.ensemble_models == 'hat,realesrgan'
        assert args.ensemble_method == 'weighted'

    # -- Dispatch routing --

    @pytest.fixture
    def frames_dir(self, tmp_path):
        d = tmp_path / "frames"
        d.mkdir()
        for i in range(3):
            (d / f"frame_{i:04d}.png").write_bytes(b'\x89PNG' + b'\0' * 100)
        return d

    @patch('framewright.cli._enhance_with_hat')
    def test_dispatches_to_hat(self, mock_hat, frames_dir, tmp_path):
        """Model 'hat' dispatches to _enhance_with_hat."""
        args = Mock(model='hat', input=str(frames_dir),
                    output=str(tmp_path / 'out'), scale=4)
        enhance_frames(args)
        mock_hat.assert_called_once()

    @patch('framewright.cli._enhance_with_diffusion')
    def test_dispatches_to_diffusion(self, mock_diff, frames_dir, tmp_path):
        """Model 'diffusion' dispatches to _enhance_with_diffusion."""
        args = Mock(model='diffusion', input=str(frames_dir),
                    output=str(tmp_path / 'out'), scale=4)
        enhance_frames(args)
        mock_diff.assert_called_once()

    @patch('framewright.cli._enhance_with_ensemble')
    def test_dispatches_to_ensemble(self, mock_ens, frames_dir, tmp_path):
        """Model 'ensemble' dispatches to _enhance_with_ensemble."""
        args = Mock(model='ensemble', input=str(frames_dir),
                    output=str(tmp_path / 'out'), scale=4)
        enhance_frames(args)
        mock_ens.assert_called_once()

    @patch('framewright.cli._enhance_with_realesrgan')
    def test_defaults_to_realesrgan(self, mock_rgan, frames_dir, tmp_path):
        """Model 'realesrgan-x4plus' dispatches to _enhance_with_realesrgan."""
        args = Mock(model='realesrgan-x4plus', input=str(frames_dir),
                    output=str(tmp_path / 'out'), scale=4)
        enhance_frames(args)
        mock_rgan.assert_called_once()

    # -- Fallback behavior --

    @patch('framewright.cli._enhance_with_realesrgan')
    def test_hat_fallback_on_import_error(self, mock_rgan, frames_dir, tmp_path):
        """HAT falls back to Real-ESRGAN when processor import fails."""
        args = Mock(model='hat', hat_size='large',
                    input=str(frames_dir), output=str(tmp_path / 'out'), scale=4)
        with patch('framewright.cli._enhance_with_hat.__module__', 'framewright.cli'):
            with patch.dict('sys.modules', {'framewright.processors.hat_upscaler': None}):
                _enhance_with_hat(args, frames_dir, tmp_path / 'out', 4, 3)
        mock_rgan.assert_called_once()

    @patch('framewright.cli._enhance_with_realesrgan')
    def test_diffusion_fallback_when_unavailable(self, mock_rgan, frames_dir, tmp_path):
        """Diffusion falls back to Real-ESRGAN when is_available() is False."""
        mock_proc = Mock()
        mock_proc.is_available.return_value = False
        mock_config = Mock()
        with patch('framewright.processors.diffusion_sr.DiffusionSRProcessor', return_value=mock_proc):
            with patch('framewright.processors.diffusion_sr.DiffusionSRConfig', return_value=mock_config):
                _enhance_with_diffusion(args=Mock(diffusion_model='upscale_a_video', diffusion_steps=20),
                                        input_dir=frames_dir, output_dir=tmp_path / 'out',
                                        scale=4, frame_count=3)
        mock_rgan.assert_called_once()

    @patch('framewright.cli._enhance_with_realesrgan')
    def test_ensemble_fallback_when_unavailable(self, mock_rgan, frames_dir, tmp_path):
        """Ensemble falls back to Real-ESRGAN when < 2 models available."""
        mock_ens = Mock()
        mock_ens.is_available.return_value = False
        mock_config = Mock()
        with patch('framewright.processors.ensemble_sr.EnsembleSR', return_value=mock_ens):
            with patch('framewright.processors.ensemble_sr.EnsembleConfig', return_value=mock_config):
                _enhance_with_ensemble(args=Mock(ensemble_models='hat,realesrgan', ensemble_method='weighted'),
                                       input_dir=frames_dir, output_dir=tmp_path / 'out',
                                       scale=4, frame_count=3)
        mock_rgan.assert_called_once()
