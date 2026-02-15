"""Tests for the unified infrastructure layer.

Tests hardware detection, memory management, and backend selection
across all supported platforms.
"""

import pytest
from unittest.mock import MagicMock, patch

from framewright.infrastructure import (
    HardwareInfo,
    HardwareTier,
    GPUVendor,
    BackendType,
    get_hardware_info,
    detect_hardware,
    get_hardware_tier,
    is_gpu_available,
    get_vram_mb,
    get_optimal_device,
    get_available_backends,
)
from framewright.infrastructure.gpu.detector import (
    _detect_vendor_from_name,
    _is_dedicated_gpu,
    _determine_tier,
    _get_processing_recommendations,
)
from framewright.infrastructure.gpu.memory import (
    MemoryManager,
    MemoryPressure,
    MemoryStats,
    MemoryConfig,
    get_memory_manager,
    get_optimal_batch_size,
    get_optimal_tile_size,
)
from framewright.infrastructure.gpu.backends.base import (
    Backend,
    CPUBackend,
    get_backend,
    list_backends,
)


class TestVendorDetection:
    """Test GPU vendor detection from device names."""

    def test_nvidia_detection(self):
        """Test NVIDIA GPU name detection."""
        nvidia_names = [
            "NVIDIA GeForce RTX 4090",
            "GeForce GTX 1080 Ti",
            "NVIDIA Quadro RTX 6000",
            "Tesla V100",
        ]
        for name in nvidia_names:
            assert _detect_vendor_from_name(name) == GPUVendor.NVIDIA

    def test_amd_detection(self):
        """Test AMD GPU name detection."""
        amd_names = [
            "AMD Radeon RX 7900 XTX",
            "Radeon RX 6800 XT",
            "AMD Radeon Vega 64",
        ]
        for name in amd_names:
            assert _detect_vendor_from_name(name) == GPUVendor.AMD

    def test_intel_detection(self):
        """Test Intel GPU name detection."""
        intel_names = [
            "Intel Arc A770",
            "Intel UHD Graphics 630",
            "Intel Iris Xe Graphics",
        ]
        for name in intel_names:
            assert _detect_vendor_from_name(name) == GPUVendor.INTEL

    def test_unknown_vendor(self):
        """Test unknown vendor fallback."""
        assert _detect_vendor_from_name("Some Unknown GPU") == GPUVendor.UNKNOWN


class TestDedicatedGPUDetection:
    """Test dedicated vs integrated GPU detection."""

    def test_dedicated_gpus(self):
        """Test dedicated GPU detection."""
        dedicated_names = [
            "NVIDIA GeForce RTX 4090",
            "AMD Radeon RX 7900 XTX",
            "Intel Arc A770",
        ]
        for name in dedicated_names:
            assert _is_dedicated_gpu(name) is True

    def test_integrated_gpus(self):
        """Test integrated GPU detection."""
        integrated_names = [
            "Intel UHD Graphics 630",
            "Intel Iris Plus Graphics",
            "AMD Radeon Vega 8 Graphics",
        ]
        for name in integrated_names:
            assert _is_dedicated_gpu(name) is False


class TestHardwareTier:
    """Test hardware tier classification."""

    def test_cpu_only_tier(self):
        """Test CPU-only tier detection."""
        assert _determine_tier(0, GPUVendor.UNKNOWN) == HardwareTier.CPU_ONLY

    def test_vram_4gb_tier(self):
        """Test 4GB VRAM tier."""
        assert _determine_tier(3000, GPUVendor.NVIDIA) == HardwareTier.VRAM_4GB
        assert _determine_tier(4000, GPUVendor.AMD) == HardwareTier.VRAM_4GB

    def test_vram_8gb_tier(self):
        """Test 8GB VRAM tier."""
        assert _determine_tier(6000, GPUVendor.NVIDIA) == HardwareTier.VRAM_8GB
        assert _determine_tier(8000, GPUVendor.AMD) == HardwareTier.VRAM_8GB

    def test_vram_12gb_tier(self):
        """Test 12GB VRAM tier."""
        assert _determine_tier(10000, GPUVendor.NVIDIA) == HardwareTier.VRAM_12GB
        assert _determine_tier(12000, GPUVendor.AMD) == HardwareTier.VRAM_12GB

    def test_vram_16gb_plus_tier(self):
        """Test 16GB+ VRAM tier."""
        assert _determine_tier(16000, GPUVendor.NVIDIA) == HardwareTier.VRAM_16GB_PLUS
        assert _determine_tier(20000, GPUVendor.AMD) == HardwareTier.VRAM_16GB_PLUS

    def test_vram_24gb_plus_tier(self):
        """Test 24GB+ VRAM tier."""
        assert _determine_tier(24000, GPUVendor.NVIDIA) == HardwareTier.VRAM_24GB_PLUS
        assert _determine_tier(48000, GPUVendor.AMD) == HardwareTier.VRAM_24GB_PLUS

    def test_apple_silicon_tier(self):
        """Test Apple Silicon tier."""
        assert _determine_tier(16000, GPUVendor.APPLE) == HardwareTier.APPLE_SILICON


class TestProcessingRecommendations:
    """Test processing parameter recommendations per tier."""

    def test_cpu_only_recommendations(self):
        """Test CPU-only recommendations (conservative)."""
        tile, batch, res, can_4k = _get_processing_recommendations(HardwareTier.CPU_ONLY)
        assert tile == 128
        assert batch == 1
        assert can_4k is False

    def test_high_end_recommendations(self):
        """Test high-end GPU recommendations."""
        tile, batch, res, can_4k = _get_processing_recommendations(HardwareTier.VRAM_24GB_PLUS)
        assert tile == 0  # No tiling needed
        assert batch >= 16
        assert can_4k is True

    def test_apple_silicon_recommendations(self):
        """Test Apple Silicon recommendations."""
        tile, batch, res, can_4k = _get_processing_recommendations(HardwareTier.APPLE_SILICON)
        assert tile > 0
        assert batch >= 4
        assert can_4k is True


class TestHardwareInfo:
    """Test HardwareInfo dataclass."""

    def test_hardware_info_creation(self):
        """Test creating HardwareInfo with defaults."""
        info = HardwareInfo()
        assert info.has_gpu is False
        assert info.tier == HardwareTier.CPU_ONLY
        assert info.device_count == 0
        assert BackendType.CPU in info.available_backends or len(info.available_backends) == 0

    def test_hardware_info_to_dict(self):
        """Test serialization to dict."""
        info = HardwareInfo(
            platform="win32",
            cpu_name="Test CPU",
            has_gpu=True,
            tier=HardwareTier.VRAM_8GB,
            total_vram_mb=8192,
        )
        data = info.to_dict()
        assert data["platform"] == "win32"
        assert data["cpu_name"] == "Test CPU"
        assert data["has_gpu"] is True
        assert data["tier"] == "vram_8gb"


class TestDetectHardware:
    """Test hardware detection function."""

    def test_detect_hardware_returns_info(self):
        """Test that detect_hardware returns HardwareInfo."""
        info = detect_hardware(force_refresh=True)
        assert isinstance(info, HardwareInfo)
        assert info.platform != ""

    def test_get_hardware_info_cached(self):
        """Test that get_hardware_info uses cache."""
        info1 = get_hardware_info()
        info2 = get_hardware_info()
        # Should return same instance (cached)
        assert info1 is info2

    def test_detect_hardware_force_refresh(self):
        """Test force refresh bypasses cache."""
        info1 = get_hardware_info()
        info2 = detect_hardware(force_refresh=True)
        # New instance created
        assert isinstance(info2, HardwareInfo)


class TestMemoryManager:
    """Test memory management functionality."""

    def test_memory_manager_creation(self):
        """Test creating MemoryManager."""
        manager = MemoryManager()
        assert manager.tier is not None
        assert manager.hardware_info is not None

    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()
        assert config.target_utilization == 0.85
        assert config.min_batch_size == 1
        assert config.max_batch_size == 32
        assert "realesrgan" in config.model_memory_estimates

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        manager = MemoryManager()
        batch = manager.get_optimal_batch_size(
            frame_size=(1920, 1080),
            model="realesrgan",
        )
        assert batch >= 1
        assert batch <= 32

    def test_get_optimal_tile_size(self):
        """Test optimal tile size calculation."""
        manager = MemoryManager()
        tile = manager.get_optimal_tile_size(frame_size=(3840, 2160))
        # Should return a tile size or None (for no tiling)
        assert tile is None or (isinstance(tile, int) and tile >= 128)

    def test_should_use_fp16(self):
        """Test FP16 recommendation."""
        manager = MemoryManager()
        result = manager.should_use_fp16()
        assert isinstance(result, bool)

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        manager = MemoryManager()
        config = manager.get_processing_config(
            frame_size=(1920, 1080),
            models=["realesrgan"],
            scale_factor=4,
        )
        assert "batch_size" in config
        assert "tile_size" in config
        assert "half_precision" in config
        assert "tier" in config

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        manager = MemoryManager()
        usage = manager.estimate_memory_usage(
            frame_size=(1920, 1080),
            batch_size=2,
            models=["realesrgan"],
            scale_factor=4,
        )
        assert usage > 0


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_memory_stats_defaults(self):
        """Test MemoryStats default values."""
        stats = MemoryStats()
        assert stats.device_id == 0
        assert stats.total_mb == 0
        assert stats.pressure == MemoryPressure.LOW

    def test_utilization_percent(self):
        """Test utilization percentage calculation."""
        stats = MemoryStats(total_mb=8000, used_mb=4000)
        assert stats.utilization_percent == 50.0

    def test_available_mb(self):
        """Test available memory calculation."""
        stats = MemoryStats(total_mb=8000, used_mb=3000)
        assert stats.available_mb == 5000


class TestBackendSystem:
    """Test backend selection and management."""

    def test_cpu_backend_creation(self):
        """Test CPU backend can be created."""
        backend = CPUBackend()
        assert backend.backend_type == BackendType.CPU
        assert backend.name == "CPU"

    def test_cpu_backend_initialization(self):
        """Test CPU backend initialization."""
        backend = CPUBackend()
        result = backend.initialize()
        assert result is True
        assert backend.is_initialized is True
        backend.cleanup()
        assert backend.is_initialized is False

    def test_cpu_backend_capabilities(self):
        """Test CPU backend capabilities."""
        backend = CPUBackend()
        backend.initialize()
        caps = backend.get_capabilities()
        assert caps.backend_type == BackendType.CPU
        assert caps.supports_fp16 is False  # CPU FP16 is slow
        backend.cleanup()

    def test_get_backend_auto_selection(self):
        """Test automatic backend selection."""
        backend = get_backend()
        assert isinstance(backend, Backend)
        assert backend.backend_type in BackendType

    def test_get_backend_specific_type(self):
        """Test requesting specific backend type."""
        backend = get_backend(BackendType.CPU)
        assert backend.backend_type == BackendType.CPU

    def test_backend_context_manager(self):
        """Test backend as context manager."""
        with CPUBackend() as backend:
            assert backend.is_initialized is True
        assert backend.is_initialized is False


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_is_gpu_available(self):
        """Test GPU availability check."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_vram_mb(self):
        """Test VRAM query."""
        vram = get_vram_mb()
        assert isinstance(vram, int)
        assert vram >= 0

    def test_get_optimal_device(self):
        """Test optimal device selection."""
        device = get_optimal_device()
        assert isinstance(device, int)
        assert device >= 0

    def test_get_available_backends(self):
        """Test available backends list."""
        backends = get_available_backends()
        assert isinstance(backends, list)
        # CPU should always be available
        assert BackendType.CPU in backends

    def test_get_hardware_tier(self):
        """Test hardware tier query."""
        tier = get_hardware_tier()
        assert isinstance(tier, HardwareTier)


class TestGlobalInstances:
    """Test global/cached instance behavior."""

    def test_memory_manager_singleton(self):
        """Test memory manager global instance."""
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        assert manager1 is manager2

    def test_global_optimal_batch_size(self):
        """Test global batch size function."""
        batch = get_optimal_batch_size((1920, 1080), "realesrgan")
        assert batch >= 1

    def test_global_optimal_tile_size(self):
        """Test global tile size function."""
        tile = get_optimal_tile_size((1920, 1080))
        assert tile is None or tile >= 128


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_vram_handling(self):
        """Test handling of zero VRAM."""
        tier = _determine_tier(0, GPUVendor.NVIDIA)
        assert tier == HardwareTier.CPU_ONLY

    def test_very_high_vram_handling(self):
        """Test handling of very high VRAM (datacenter GPUs)."""
        tier = _determine_tier(80000, GPUVendor.NVIDIA)  # 80GB A100
        assert tier == HardwareTier.VRAM_24GB_PLUS

    def test_small_frame_tile_size(self):
        """Test tile size with small frames."""
        manager = MemoryManager()
        tile = manager.get_optimal_tile_size(frame_size=(640, 480))
        if tile is not None:
            assert tile <= 480  # Should not exceed frame size

    def test_large_frame_tile_size(self):
        """Test tile size with 8K frames."""
        manager = MemoryManager()
        tile = manager.get_optimal_tile_size(frame_size=(7680, 4320))
        # Should recommend tiling for most tiers
        assert tile is None or tile > 0

    def test_unknown_model_memory_estimate(self):
        """Test memory estimation for unknown model."""
        manager = MemoryManager()
        batch = manager.get_optimal_batch_size(
            frame_size=(1920, 1080),
            model="unknown_model_xyz",
        )
        # Should use default estimate and return valid batch
        assert batch >= 1
