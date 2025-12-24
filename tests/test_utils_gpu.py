"""Tests for the GPU utilities module."""
import pytest
from unittest.mock import MagicMock, patch

from framewright.utils.gpu import (
    is_nvidia_gpu_available,
    get_gpu_memory_info,
    get_all_gpu_info,
    get_optimal_device,
    calculate_optimal_tile_size,
    get_adaptive_tile_sequence,
    VRAMMonitor,
    wait_for_vram,
    GPUInfo,
)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_create_gpu_info(self):
        """Test creating GPU info."""
        info = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 3080",
            total_memory_mb=10240,
            used_memory_mb=2048,
            free_memory_mb=8192,
            utilization_percent=25.0,
        )

        assert info.index == 0
        assert info.total_memory_mb == 10240

    def test_memory_usage_percent(self):
        """Test memory usage percentage calculation."""
        info = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_mb=10000,
            used_memory_mb=2500,
            free_memory_mb=7500,
            utilization_percent=0.0,
        )

        assert info.memory_usage_percent == 25.0


class TestIsNvidiaGpuAvailable:
    """Tests for is_nvidia_gpu_available function."""

    @patch('shutil.which')
    def test_nvidia_smi_available(self, mock_which):
        """Test when nvidia-smi is available."""
        mock_which.return_value = "/usr/bin/nvidia-smi"

        assert is_nvidia_gpu_available() is True

    @patch('shutil.which')
    def test_nvidia_smi_not_available(self, mock_which):
        """Test when nvidia-smi is not available."""
        mock_which.return_value = None

        assert is_nvidia_gpu_available() is False


class TestGetGpuMemoryInfo:
    """Tests for get_gpu_memory_info function."""

    @patch('framewright.utils.gpu.is_nvidia_gpu_available')
    def test_no_gpu_available(self, mock_available):
        """Test when no GPU is available."""
        mock_available.return_value = False

        result = get_gpu_memory_info()

        assert result is None

    @patch('subprocess.run')
    @patch('framewright.utils.gpu.is_nvidia_gpu_available')
    def test_get_memory_info(self, mock_available, mock_run):
        """Test getting GPU memory info."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="10240, 2048, 8192\n"
        )

        result = get_gpu_memory_info()

        assert result is not None
        assert result["total_mb"] == 10240
        assert result["used_mb"] == 2048
        assert result["free_mb"] == 8192


class TestGetAllGpuInfo:
    """Tests for get_all_gpu_info function."""

    @patch('framewright.utils.gpu.is_nvidia_gpu_available')
    def test_no_gpu_available(self, mock_available):
        """Test when no GPU is available."""
        mock_available.return_value = False

        result = get_all_gpu_info()

        assert result == []

    @patch('subprocess.run')
    @patch('framewright.utils.gpu.is_nvidia_gpu_available')
    def test_get_all_info(self, mock_available, mock_run):
        """Test getting info for all GPUs."""
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX 3080, 10240, 2048, 8192, 25, 65\n"
        )

        result = get_all_gpu_info()

        assert len(result) == 1
        assert result[0].name == "NVIDIA RTX 3080"
        assert result[0].total_memory_mb == 10240


class TestCalculateOptimalTileSize:
    """Tests for calculate_optimal_tile_size function."""

    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_no_tiling_needed(self, mock_memory):
        """Test when no tiling is needed (enough VRAM)."""
        mock_memory.return_value = {"free_mb": 16000}

        tile_size = calculate_optimal_tile_size(
            frame_resolution=(1920, 1080),
            scale_factor=4,
        )

        # With 16GB VRAM, should not need tiling for 1080p
        assert tile_size == 0  # 0 means no tiling

    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_tiling_needed_low_vram(self, mock_memory):
        """Test when tiling is needed due to low VRAM."""
        mock_memory.return_value = {"free_mb": 2000}

        tile_size = calculate_optimal_tile_size(
            frame_resolution=(3840, 2160),  # 4K
            scale_factor=4,
        )

        # With only 2GB VRAM for 4K upscaling, should need tiling
        assert tile_size > 0
        assert tile_size % 32 == 0  # Should be aligned to 32

    def test_with_explicit_vram(self):
        """Test with explicitly provided VRAM."""
        tile_size = calculate_optimal_tile_size(
            frame_resolution=(3840, 2160),
            scale_factor=4,
            available_vram_mb=4000,
        )

        # Should calculate tile size based on provided VRAM
        assert tile_size > 0

    def test_minimum_tile_size(self):
        """Test that tile size doesn't go below minimum."""
        tile_size = calculate_optimal_tile_size(
            frame_resolution=(7680, 4320),  # 8K
            scale_factor=4,
            available_vram_mb=1000,  # Very limited VRAM
        )

        # Should be at least 128
        assert tile_size >= 128


class TestGetAdaptiveTileSequence:
    """Tests for get_adaptive_tile_sequence function."""

    def test_generate_sequence(self):
        """Test generating adaptive tile sequence."""
        sequence = get_adaptive_tile_sequence(
            frame_resolution=(1920, 1080),
            scale_factor=4,
            starting_tile_size=512,
        )

        # Should be descending
        for i in range(len(sequence) - 1):
            assert sequence[i] > sequence[i + 1]

        # All should be multiples of 32
        for tile in sequence:
            assert tile % 32 == 0

    def test_minimum_in_sequence(self):
        """Test that minimum tile size is in sequence."""
        sequence = get_adaptive_tile_sequence(
            frame_resolution=(1920, 1080),
            scale_factor=4,
            min_tile_size=128,
        )

        assert 128 in sequence


class TestVRAMMonitor:
    """Tests for VRAMMonitor class."""

    def test_init(self):
        """Test monitor initialization."""
        monitor = VRAMMonitor(device_id=0, threshold_mb=500)

        assert monitor.device_id == 0
        assert monitor.threshold_mb == 500
        assert monitor.peak_usage_mb == 0

    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_sample(self, mock_memory):
        """Test taking VRAM sample."""
        mock_memory.return_value = {
            "total_mb": 10000,
            "used_mb": 3000,
            "free_mb": 7000,
        }

        monitor = VRAMMonitor()
        sample = monitor.sample()

        assert sample is not None
        assert monitor.peak_usage_mb == 3000
        assert len(monitor.samples) == 1

    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_is_low_memory(self, mock_memory):
        """Test low memory detection."""
        mock_memory.return_value = {
            "total_mb": 10000,
            "used_mb": 9600,
            "free_mb": 400,
        }

        monitor = VRAMMonitor(threshold_mb=500)

        assert monitor.is_low_memory() is True

    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_get_statistics(self, mock_memory):
        """Test getting VRAM statistics."""
        mock_memory.side_effect = [
            {"total_mb": 10000, "used_mb": 2000, "free_mb": 8000},
            {"total_mb": 10000, "used_mb": 3000, "free_mb": 7000},
            {"total_mb": 10000, "used_mb": 2500, "free_mb": 7500},
        ]

        monitor = VRAMMonitor()
        for _ in range(3):
            monitor.sample()

        stats = monitor.get_statistics()

        assert stats["min_mb"] == 2000
        assert stats["max_mb"] == 3000
        assert stats["samples"] == 3


class TestWaitForVram:
    """Tests for wait_for_vram function."""

    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_vram_available_immediately(self, mock_memory):
        """Test when VRAM is available immediately."""
        mock_memory.return_value = {
            "total_mb": 10000,
            "used_mb": 2000,
            "free_mb": 8000,
        }

        result = wait_for_vram(required_mb=4000, timeout_seconds=1.0)

        assert result is True

    @patch('time.time')
    @patch('time.sleep')
    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_vram_becomes_available(self, mock_memory, mock_sleep, mock_time):
        """Test when VRAM becomes available after waiting."""
        # Simulate time passing
        mock_time.side_effect = [0, 0.5, 1.0, 1.5]

        mock_memory.side_effect = [
            {"free_mb": 2000},  # Not enough
            {"free_mb": 3000},  # Not enough
            {"free_mb": 5000},  # Enough
        ]

        result = wait_for_vram(
            required_mb=4000,
            timeout_seconds=5.0,
            check_interval=0.5,
        )

        assert result is True

    @patch('time.time')
    @patch('time.sleep')
    @patch('framewright.utils.gpu.get_gpu_memory_info')
    def test_vram_timeout(self, mock_memory, mock_sleep, mock_time):
        """Test timeout when VRAM never becomes available."""
        mock_time.side_effect = [0, 1, 2, 3]  # Time advancing
        mock_memory.return_value = {"free_mb": 1000}  # Always low

        result = wait_for_vram(
            required_mb=4000,
            timeout_seconds=2.0,
            check_interval=1.0,
        )

        assert result is False
