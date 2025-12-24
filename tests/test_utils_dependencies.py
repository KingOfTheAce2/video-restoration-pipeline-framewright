"""Tests for the dependencies utilities module."""
import pytest
from unittest.mock import MagicMock, patch

from framewright.utils.dependencies import (
    compare_versions,
    check_ffmpeg,
    check_ffprobe,
    check_realesrgan,
    check_ytdlp,
    check_rife,
    validate_all_dependencies,
    get_enhancement_backend,
    DependencyInfo,
    DependencyReport,
)


class TestCompareVersions:
    """Tests for compare_versions function."""

    def test_equal_versions(self):
        """Test comparing equal versions."""
        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("2.5.3", "2.5.3") == 0

    def test_less_than(self):
        """Test version less than comparison."""
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("1.5.0", "1.6.0") == -1
        assert compare_versions("1.5.3", "1.5.4") == -1

    def test_greater_than(self):
        """Test version greater than comparison."""
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.6.0", "1.5.0") == 1
        assert compare_versions("1.5.4", "1.5.3") == 1

    def test_version_with_prefix(self):
        """Test versions with v prefix."""
        assert compare_versions("v1.0.0", "1.0.0") == 0
        assert compare_versions("V2.0.0", "v2.0.0") == 0

    def test_version_with_suffix(self):
        """Test versions with suffixes."""
        assert compare_versions("1.0.0-beta", "1.0.0") == 0
        assert compare_versions("2.0.0_rc1", "2.0.0") == 0

    def test_different_length_versions(self):
        """Test versions with different lengths."""
        assert compare_versions("1.0", "1.0.0") == 0
        assert compare_versions("1.0.0.0", "1.0.0") == 0


class TestDependencyInfo:
    """Tests for DependencyInfo dataclass."""

    def test_create_info(self):
        """Test creating dependency info."""
        info = DependencyInfo(
            name="FFmpeg",
            command="ffmpeg",
            installed=True,
            version="5.1.2",
        )

        assert info.name == "FFmpeg"
        assert info.installed is True
        assert info.version == "5.1.2"

    def test_default_values(self):
        """Test default values."""
        info = DependencyInfo(name="Test", command="test")

        assert info.installed is False
        assert info.version is None
        assert info.meets_minimum is True


class TestDependencyReport:
    """Tests for DependencyReport class."""

    def test_is_ready_all_met(self):
        """Test is_ready when all requirements met."""
        report = DependencyReport(all_required_met=True)

        assert report.is_ready() is True

    def test_is_ready_missing_required(self):
        """Test is_ready when requirements missing."""
        report = DependencyReport(
            all_required_met=False,
            missing_required=["ffmpeg"],
        )

        assert report.is_ready() is False

    def test_summary(self):
        """Test summary generation."""
        report = DependencyReport()
        report.dependencies["ffmpeg"] = DependencyInfo(
            name="FFmpeg",
            command="ffmpeg",
            installed=True,
            version="5.1.2",
        )

        summary = report.summary()

        assert "ffmpeg" in summary.lower()
        assert "OK" in summary


class TestCheckFFmpeg:
    """Tests for check_ffmpeg function."""

    @patch('shutil.which')
    def test_ffmpeg_not_found(self, mock_which):
        """Test when ffmpeg is not found."""
        mock_which.return_value = None

        info = check_ffmpeg()

        assert info.installed is False
        assert info.error_message is not None

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_ffmpeg_found(self, mock_which, mock_run):
        """Test when ffmpeg is found."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 5.1.2 Copyright..."
        )

        info = check_ffmpeg()

        assert info.installed is True
        assert info.version == "5.1.2"
        assert info.path == "/usr/bin/ffmpeg"


class TestCheckFFprobe:
    """Tests for check_ffprobe function."""

    @patch('shutil.which')
    def test_ffprobe_not_found(self, mock_which):
        """Test when ffprobe is not found."""
        mock_which.return_value = None

        info = check_ffprobe()

        assert info.installed is False

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_ffprobe_found(self, mock_which, mock_run):
        """Test when ffprobe is found."""
        mock_which.return_value = "/usr/bin/ffprobe"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ffprobe version 5.1.2"
        )

        info = check_ffprobe()

        assert info.installed is True


class TestCheckRealesrgan:
    """Tests for check_realesrgan function."""

    @patch('shutil.which')
    def test_realesrgan_not_found(self, mock_which):
        """Test when Real-ESRGAN is not found."""
        mock_which.return_value = None

        info = check_realesrgan()

        assert info.installed is False

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_realesrgan_found(self, mock_which, mock_run):
        """Test when Real-ESRGAN is found."""
        mock_which.return_value = "/usr/local/bin/realesrgan-ncnn-vulkan"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="realesrgan-ncnn-vulkan version 0.2.0\n-n model_name"
        )

        info = check_realesrgan()

        assert info.installed is True


class TestCheckYtdlp:
    """Tests for check_ytdlp function."""

    @patch('shutil.which')
    def test_ytdlp_not_found(self, mock_which):
        """Test when yt-dlp is not found."""
        mock_which.return_value = None

        info = check_ytdlp()

        assert info.installed is False

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_ytdlp_found(self, mock_which, mock_run):
        """Test when yt-dlp is found."""
        mock_which.return_value = "/usr/local/bin/yt-dlp"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="2023.12.30"
        )

        info = check_ytdlp()

        assert info.installed is True
        assert info.version == "2023.12.30"


class TestCheckRife:
    """Tests for check_rife function."""

    @patch('shutil.which')
    def test_rife_not_found(self, mock_which):
        """Test when RIFE is not found."""
        mock_which.return_value = None

        info = check_rife()

        assert info.installed is False
        # RIFE is optional, so this is not an error

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_rife_found(self, mock_which, mock_run):
        """Test when RIFE is found."""
        mock_which.return_value = "/usr/local/bin/rife-ncnn-vulkan"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="rife-ncnn-vulkan rife-v4.6"
        )

        info = check_rife()

        assert info.installed is True


class TestValidateAllDependencies:
    """Tests for validate_all_dependencies function."""

    @patch('framewright.utils.dependencies.check_ffmpeg')
    @patch('framewright.utils.dependencies.check_ffprobe')
    @patch('framewright.utils.dependencies.check_realesrgan')
    @patch('framewright.utils.dependencies.check_ytdlp')
    @patch('framewright.utils.dependencies.check_rife')
    def test_all_installed(self, mock_rife, mock_ytdlp, mock_realesrgan, mock_ffprobe, mock_ffmpeg):
        """Test when all dependencies are installed."""
        mock_ffmpeg.return_value = DependencyInfo(
            name="FFmpeg", command="ffmpeg", installed=True, version="5.0"
        )
        mock_ffprobe.return_value = DependencyInfo(
            name="ffprobe", command="ffprobe", installed=True
        )
        mock_realesrgan.return_value = DependencyInfo(
            name="Real-ESRGAN", command="realesrgan", installed=True
        )
        mock_ytdlp.return_value = DependencyInfo(
            name="yt-dlp", command="yt-dlp", installed=True, version="2023.12.30"
        )
        mock_rife.return_value = DependencyInfo(
            name="RIFE", command="rife", installed=True
        )

        report = validate_all_dependencies()

        assert report.is_ready() is True
        assert len(report.missing_required) == 0

    @patch('framewright.utils.dependencies.check_ffmpeg')
    @patch('framewright.utils.dependencies.check_ffprobe')
    @patch('framewright.utils.dependencies.check_realesrgan')
    @patch('framewright.utils.dependencies.check_ytdlp')
    def test_missing_required(self, mock_ytdlp, mock_realesrgan, mock_ffprobe, mock_ffmpeg):
        """Test when required dependency is missing."""
        mock_ffmpeg.return_value = DependencyInfo(
            name="FFmpeg", command="ffmpeg", installed=False
        )
        mock_ffprobe.return_value = DependencyInfo(
            name="ffprobe", command="ffprobe", installed=True
        )
        mock_realesrgan.return_value = DependencyInfo(
            name="Real-ESRGAN", command="realesrgan", installed=True
        )
        mock_ytdlp.return_value = DependencyInfo(
            name="yt-dlp", command="yt-dlp", installed=True
        )

        report = validate_all_dependencies(
            required=["ffmpeg", "ffprobe", "realesrgan", "yt-dlp"]
        )

        assert report.is_ready() is False
        assert "ffmpeg" in report.missing_required


class TestGetEnhancementBackend:
    """Tests for get_enhancement_backend function."""

    @patch('shutil.which')
    def test_ncnn_backend(self, mock_which):
        """Test when ncnn backend is available."""
        mock_which.return_value = "/usr/local/bin/realesrgan-ncnn-vulkan"

        backend_type, command = get_enhancement_backend()

        assert backend_type == "ncnn"
        assert "realesrgan-ncnn-vulkan" in command

    @patch('importlib.util.find_spec')
    @patch('shutil.which')
    def test_python_backend_fallback(self, mock_which, mock_find_spec):
        """Test fallback to Python backend."""
        mock_which.return_value = None  # No binary
        mock_find_spec.return_value = MagicMock()  # Python package exists

        backend_type, command = get_enhancement_backend()

        assert backend_type == "python"

    @patch('importlib.util.find_spec')
    @patch('shutil.which')
    def test_no_backend_available(self, mock_which, mock_find_spec):
        """Test when no backend is available."""
        mock_which.return_value = None
        mock_find_spec.return_value = None

        with pytest.raises(RuntimeError) as excinfo:
            get_enhancement_backend()

        assert "No Real-ESRGAN backend available" in str(excinfo.value)
