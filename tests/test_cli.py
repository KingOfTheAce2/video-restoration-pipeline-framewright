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

    def test_restore_video_command(self):
        """Test restore_video command function."""
        mock_args = Mock()
        mock_args.input = 'video.mp4'
        mock_args.output = 'output.mp4'
        mock_args.scale = 2
        mock_args.audio_enhance = False

        # restore_video is a placeholder, just verify it doesn't crash
        restore_video(mock_args)

    def test_extract_frames_command_valid_input(self, video_file, temp_dir):
        """Test extract_frames command with valid input."""
        mock_args = Mock()
        mock_args.input = str(video_file)
        mock_args.output = str(temp_dir / "frames")

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
        mock_args = Mock()
        mock_args.input = str(frames_dir)
        mock_args.output = str(temp_dir / "enhanced")
        mock_args.scale = 2
        mock_args.model = 'realesrgan-x2plus'

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
