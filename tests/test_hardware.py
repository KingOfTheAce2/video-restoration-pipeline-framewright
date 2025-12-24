"""Tests for hardware compatibility checking."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from framewright.hardware import (
    check_hardware,
    print_hardware_report,
    quick_check,
    HardwareReport,
    SystemInfo,
    GPUCapability,
    get_system_info,
    get_gpu_capability,
)


class TestSystemInfo:
    """Tests for system information gathering."""

    def test_get_system_info_returns_dataclass(self):
        """Test that system info returns correct dataclass."""
        info = get_system_info()
        assert isinstance(info, SystemInfo)
        assert info.os_name is not None
        assert info.python_version is not None
        assert info.cpu_cores > 0
        assert info.ram_total_gb > 0

    def test_system_info_has_valid_ram(self):
        """Test that RAM values are sensible."""
        info = get_system_info()
        assert info.ram_total_gb > 0
        assert info.ram_available_gb >= 0
        assert info.ram_available_gb <= info.ram_total_gb


class TestGPUInfo:
    """Tests for GPU information gathering."""

    def test_get_gpu_info_returns_dataclass(self):
        """Test that GPU info returns correct dataclass."""
        info = get_gpu_capability()
        assert isinstance(info, GPUCapability)
        # has_gpu might be True or False depending on system
        assert isinstance(info.has_gpu, bool)

    def test_gpu_no_nvidia_smi(self):
        """Test handling when nvidia-smi is not available."""
        with patch("framewright.hardware.subprocess.run", side_effect=FileNotFoundError):
            info = get_gpu_capability()
            assert info.has_gpu is False
            assert info.cuda_available is False

    def test_gpu_nvidia_smi_error(self):
        """Test handling when nvidia-smi returns error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("framewright.hardware.subprocess.run", return_value=mock_result):
            info = get_gpu_capability()
            assert info.has_gpu is False


class TestHardwareReport:
    """Tests for full hardware report generation."""

    def test_check_hardware_returns_report(self):
        """Test that check_hardware returns HardwareReport."""
        report = check_hardware()
        assert isinstance(report, HardwareReport)
        assert isinstance(report.system, SystemInfo)
        assert isinstance(report.gpu, GPUCapability)

    def test_report_has_overall_status(self):
        """Test that report has an overall status."""
        report = check_hardware()
        assert report.overall_status in ["ready", "limited", "incompatible"]

    def test_report_has_recommendations(self):
        """Test that report generates recommendations."""
        report = check_hardware()
        assert isinstance(report.recommendations, list)

    def test_report_has_warnings(self):
        """Test that report tracks warnings."""
        report = check_hardware()
        assert isinstance(report.warnings, list)

    def test_report_disk_free_positive(self):
        """Test that disk free space is reported."""
        report = check_hardware()
        assert report.disk_free_gb >= 0

    def test_report_to_dict(self):
        """Test report can be converted to dictionary."""
        report = check_hardware()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "system" in d
        assert "gpu" in d
        assert "overall_status" in d


class TestPrintReport:
    """Tests for report formatting."""

    def test_print_hardware_report_returns_string(self):
        """Test that print function returns formatted string."""
        report = check_hardware()
        output = print_hardware_report(report)
        assert isinstance(output, str)
        assert len(output) > 100  # Should be substantial

    def test_print_report_contains_sections(self):
        """Test that printed report contains expected sections."""
        report = check_hardware()
        output = print_hardware_report(report)
        assert "SYSTEM INFORMATION" in output
        assert "GPU INFORMATION" in output
        assert "STORAGE" in output
        assert "DEPENDENCIES" in output

    def test_print_report_shows_status(self):
        """Test that report shows overall status."""
        report = check_hardware()
        output = print_hardware_report(report)
        assert "Overall Status:" in output


class TestQuickCheck:
    """Tests for quick check function."""

    def test_quick_check_returns_bool(self):
        """Test that quick_check returns boolean."""
        result = quick_check()
        assert isinstance(result, bool)


class TestGPUCapability:
    """Tests for GPUCapability dataclass."""

    def test_default_values(self):
        """Test default values for GPU capability."""
        gpu = GPUCapability()
        assert gpu.has_gpu is False
        assert gpu.gpu_name == "None detected"
        assert gpu.vram_total_mb == 0
        assert gpu.vram_free_mb == 0
        assert gpu.cuda_available is False
        assert gpu.vulkan_available is False
        assert gpu.recommended_tile_size == 512
        assert gpu.max_resolution == "1080p"
        assert gpu.can_process_4k is False

    def test_custom_values(self):
        """Test custom values for GPU capability."""
        gpu = GPUCapability(
            has_gpu=True,
            gpu_name="Test GPU",
            vram_total_mb=8192,
            vram_free_mb=7000,
            cuda_available=True,
            max_resolution="4K",
            can_process_4k=True,
        )
        assert gpu.has_gpu is True
        assert gpu.gpu_name == "Test GPU"
        assert gpu.vram_total_mb == 8192
        assert gpu.vram_free_mb == 7000
        assert gpu.cuda_available is True
        assert gpu.max_resolution == "4K"
        assert gpu.can_process_4k is True


class TestSystemInfoDataclass:
    """Tests for SystemInfo dataclass."""

    def test_dataclass_creation(self):
        """Test creating SystemInfo with all fields."""
        info = SystemInfo(
            os_name="Linux",
            os_version="5.0.0",
            python_version="3.11.0",
            cpu_name="Test CPU",
            cpu_cores=8,
            ram_total_gb=32.0,
            ram_available_gb=16.0,
        )
        assert info.os_name == "Linux"
        assert info.cpu_cores == 8
        assert info.ram_total_gb == 32.0

    def test_dataclass_defaults(self):
        """Test dataclass with default values."""
        info = SystemInfo(
            os_name="Linux",
            os_version="5.0.0",
            python_version="3.11.0",
        )
        assert info.cpu_name == "Unknown"
        assert info.cpu_cores == 0


class TestHardwareReportDataclass:
    """Tests for HardwareReport dataclass."""

    def test_report_creation(self):
        """Test creating HardwareReport with required fields."""
        system = SystemInfo(
            os_name="Linux",
            os_version="5.0.0",
            python_version="3.11.0",
        )
        gpu = GPUCapability()
        report = HardwareReport(system=system, gpu=gpu)
        assert report.system == system
        assert report.gpu == gpu
        assert report.overall_status == "unknown"
        assert report.disk_free_gb == 0.0

    def test_report_defaults(self):
        """Test HardwareReport default values."""
        system = SystemInfo(
            os_name="Test",
            os_version="1.0",
            python_version="3.11.0",
        )
        gpu = GPUCapability()
        report = HardwareReport(system=system, gpu=gpu)
        assert report.dependencies_ok is False
        assert report.missing_dependencies == []
        assert report.warnings == []
        assert report.recommendations == []
