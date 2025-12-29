"""Tests for the multi-GPU distribution module."""
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from framewright.utils.multi_gpu import (
    DistributionResult,
    GPUInfo,
    GPUManager,
    GPUSelector,
    LoadBalanceStrategy,
    MultiGPUDistributor,
    MultiGPUManager,
    WorkItem,
    WorkStealingQueue,
    detect_gpus,
    distribute_frames,
    get_optimal_gpu,
    list_gpus,
    select_gpu,
)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test basic GPUInfo creation."""
        gpu = GPUInfo(
            id=0,
            name="NVIDIA GeForce RTX 3080",
            total_vram_mb=10240,
            free_vram_mb=8192,
            utilization_pct=25.0,
            temperature_c=65.0,
        )

        assert gpu.id == 0
        assert gpu.name == "NVIDIA GeForce RTX 3080"
        assert gpu.total_vram_mb == 10240
        assert gpu.free_vram_mb == 8192
        assert gpu.utilization_pct == 25.0
        assert gpu.temperature_c == 65.0

    def test_used_vram_calculation(self):
        """Test used VRAM calculation."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=8000,
            free_vram_mb=3000,
            utilization_pct=50.0,
        )

        assert gpu.used_vram_mb == 5000

    def test_vram_usage_percentage(self):
        """Test VRAM usage percentage calculation."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=10000,
            free_vram_mb=2500,
            utilization_pct=50.0,
        )

        assert gpu.vram_usage_pct == 75.0

    def test_vram_usage_percentage_zero_total(self):
        """Test VRAM usage percentage with zero total VRAM."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=0,
            free_vram_mb=0,
            utilization_pct=0.0,
        )

        assert gpu.vram_usage_pct == 0.0

    def test_is_healthy_normal_temp(self):
        """Test health check with normal temperature."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=8000,
            free_vram_mb=4000,
            utilization_pct=50.0,
            temperature_c=70.0,
        )

        assert gpu.is_healthy is True

    def test_is_healthy_high_temp(self):
        """Test health check with high temperature."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=8000,
            free_vram_mb=4000,
            utilization_pct=50.0,
            temperature_c=95.0,
        )

        assert gpu.is_healthy is False

    def test_is_healthy_no_temp(self):
        """Test health check without temperature data."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=8000,
            free_vram_mb=4000,
            utilization_pct=50.0,
        )

        assert gpu.is_healthy is True

    def test_effective_capacity(self):
        """Test effective capacity calculation."""
        gpu = GPUInfo(
            id=0,
            name="Test GPU",
            total_vram_mb=10000,
            free_vram_mb=8000,
            utilization_pct=20.0,
        )

        # 80% VRAM free * 0.7 + 80% idle * 0.3 = 0.56 + 0.24 = 0.80
        expected = (0.8 * 0.7) + (0.8 * 0.3)
        assert abs(gpu.effective_capacity - expected) < 0.01


class TestDistributionResult:
    """Tests for DistributionResult dataclass."""

    def test_empty_result(self):
        """Test empty distribution result."""
        result = DistributionResult()

        assert result.total_frames == 0
        assert result.success_rate == 100.0
        assert len(result.errors) == 0

    def test_total_frames_calculation(self):
        """Test total frames calculation."""
        result = DistributionResult(
            frames_per_gpu={
                0: [Path("frame1.png"), Path("frame2.png")],
                1: [Path("frame3.png")],
            }
        )

        assert result.total_frames == 3

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = DistributionResult(
            frames_per_gpu={
                0: [Path("frame1.png"), Path("frame2.png")],
                1: [Path("frame3.png")],
            },
            errors={"frame4.png": "Error"},
        )

        # 3 successful, 1 error = 75%
        assert result.success_rate == 75.0

    def test_summary_generation(self):
        """Test summary string generation."""
        result = DistributionResult(
            frames_per_gpu={
                0: [Path("frame1.png")],
                1: [Path("frame2.png"), Path("frame3.png")],
            },
            total_time=10.5,
            speedup_factor=1.85,
        )

        summary = result.summary()
        assert "3 frames" in summary
        assert "2 GPUs" in summary
        assert "10.5s" in summary
        assert "1.85x" in summary


class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_work_item_creation(self):
        """Test basic work item creation."""
        item = WorkItem(
            frame_path=Path("frame_001.png"),
            output_dir=Path("/output"),
        )

        assert item.frame_path == Path("frame_001.png")
        assert item.output_dir == Path("/output")
        assert item.priority == 0
        assert item.assigned_gpu is None
        assert item.attempts == 0
        assert len(item.failed_gpus) == 0

    def test_can_retry_initial(self):
        """Test can_retry on fresh work item."""
        item = WorkItem(
            frame_path=Path("frame_001.png"),
            output_dir=Path("/output"),
        )

        assert item.can_retry is True

    def test_can_retry_max_attempts(self):
        """Test can_retry after max attempts."""
        item = WorkItem(
            frame_path=Path("frame_001.png"),
            output_dir=Path("/output"),
            attempts=3,
        )

        assert item.can_retry is False


class TestWorkStealingQueue:
    """Tests for WorkStealingQueue."""

    def test_add_and_get_work(self):
        """Test adding and getting work from queue."""
        wsq = WorkStealingQueue(num_workers=2)

        item = WorkItem(frame_path=Path("frame.png"), output_dir=Path("/out"))
        wsq.add_work(item, worker_id=0)

        retrieved = wsq.get_work(worker_id=0, timeout=1.0)

        assert retrieved is not None
        assert retrieved.frame_path == Path("frame.png")

    def test_work_stealing(self):
        """Test work stealing from other worker's queue."""
        wsq = WorkStealingQueue(num_workers=2)

        # Add multiple items to worker 0's queue
        for i in range(3):
            item = WorkItem(frame_path=Path(f"frame_{i}.png"), output_dir=Path("/out"))
            wsq.add_work(item, worker_id=0)

        # Worker 1 should be able to steal work
        stolen = wsq.get_work(worker_id=1, timeout=0.5)

        assert stolen is not None

    def test_progress_tracking(self):
        """Test progress tracking."""
        wsq = WorkStealingQueue(num_workers=2)

        for i in range(4):
            item = WorkItem(frame_path=Path(f"frame_{i}.png"), output_dir=Path("/out"))
            wsq.add_work(item, worker_id=i % 2)

        assert wsq.progress == 0.0

        # Complete 2 items
        wsq.mark_complete()
        wsq.mark_complete()

        assert wsq.progress == 0.5

    def test_is_complete(self):
        """Test completion detection."""
        wsq = WorkStealingQueue(num_workers=1)

        item = WorkItem(frame_path=Path("frame.png"), output_dir=Path("/out"))
        wsq.add_work(item, worker_id=0)

        assert wsq.is_complete is False

        wsq.get_work(worker_id=0, timeout=0.1)
        wsq.mark_complete()

        assert wsq.is_complete is True


class TestGPUManager:
    """Tests for GPUManager."""

    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_no_nvidia_smi(self, mock_which):
        """Test behavior when nvidia-smi is not available."""
        mock_which.return_value = None

        manager = GPUManager()
        gpus = manager.detect_gpus()

        assert len(gpus) == 0

    @patch("framewright.utils.multi_gpu.subprocess.run")
    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_detect_gpus(self, mock_which, mock_run):
        """Test GPU detection with mocked nvidia-smi."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 3080, 10240, 8192, 25, 65, 4, 16\n"
            "1, NVIDIA GeForce RTX 3070, 8192, 6144, 30, 70, 4, 16\n",
        )

        manager = GPUManager()
        gpus = manager.detect_gpus()

        assert len(gpus) == 2
        assert gpus[0].id == 0
        assert gpus[0].name == "NVIDIA GeForce RTX 3080"
        assert gpus[0].total_vram_mb == 10240
        assert gpus[0].free_vram_mb == 8192
        assert gpus[1].id == 1
        assert gpus[1].name == "NVIDIA GeForce RTX 3070"

    @patch("framewright.utils.multi_gpu.subprocess.run")
    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_gpu_count(self, mock_which, mock_run):
        """Test GPU count property."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, GPU 0, 8000, 4000, 50, 70, 4, 16\n"
            "1, GPU 1, 8000, 4000, 50, 70, 4, 16\n",
        )

        manager = GPUManager()

        assert manager.gpu_count == 2
        assert manager.is_multi_gpu is True

    @patch("framewright.utils.multi_gpu.subprocess.run")
    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_get_optimal_gpu_vram_aware(self, mock_which, mock_run):
        """Test optimal GPU selection with VRAM-aware strategy."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, GPU 0, 8000, 2000, 50, 70, 4, 16\n"
            "1, GPU 1, 8000, 6000, 30, 65, 4, 16\n",
        )

        manager = GPUManager()
        optimal = manager.get_optimal_gpu(LoadBalanceStrategy.VRAM_AWARE)

        # GPU 1 has more free VRAM
        assert optimal == 1

    @patch("framewright.utils.multi_gpu.subprocess.run")
    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_get_optimal_gpu_least_loaded(self, mock_which, mock_run):
        """Test optimal GPU selection with least-loaded strategy."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, GPU 0, 8000, 4000, 80, 70, 4, 16\n"
            "1, GPU 1, 8000, 4000, 20, 65, 4, 16\n",
        )

        manager = GPUManager()
        optimal = manager.get_optimal_gpu(LoadBalanceStrategy.LEAST_LOADED)

        # GPU 1 has lower utilization
        assert optimal == 1

    def test_specific_gpu_ids(self):
        """Test manager with specific GPU IDs."""
        manager = GPUManager(gpu_ids=[0, 2])

        assert manager.gpu_ids == [0, 2]

    @patch("framewright.utils.multi_gpu.subprocess.run")
    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_get_healthy_gpus(self, mock_which, mock_run):
        """Test filtering healthy GPUs."""
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, GPU 0, 8000, 4000, 50, 70, 4, 16\n"
            "1, GPU 1, 8000, 4000, 50, 95, 4, 16\n",  # High temp
        )

        manager = GPUManager()
        healthy = manager.get_healthy_gpus()

        # Only GPU 0 should be healthy (GPU 1 temp too high)
        assert len(healthy) == 1
        assert healthy[0].id == 0


class TestMultiGPUDistributor:
    """Tests for MultiGPUDistributor."""

    def test_distributor_creation(self):
        """Test distributor creation with default settings."""
        distributor = MultiGPUDistributor()

        assert distributor.strategy == LoadBalanceStrategy.VRAM_AWARE
        assert distributor.workers_per_gpu == 2
        assert distributor.max_retries == 2
        assert distributor.enable_work_stealing is True

    def test_distributor_custom_settings(self):
        """Test distributor creation with custom settings."""
        distributor = MultiGPUDistributor(
            strategy=LoadBalanceStrategy.ROUND_ROBIN,
            workers_per_gpu=4,
            max_retries=5,
            enable_work_stealing=False,
        )

        assert distributor.strategy == LoadBalanceStrategy.ROUND_ROBIN
        assert distributor.workers_per_gpu == 4
        assert distributor.max_retries == 5
        assert distributor.enable_work_stealing is False

    def test_stop_distributor(self):
        """Test stopping the distributor."""
        distributor = MultiGPUDistributor()
        distributor.stop()

        # Should set stop event
        assert distributor._stop_event.is_set()

    @patch.object(GPUManager, "get_healthy_gpus")
    def test_distribute_empty_frames(self, mock_healthy):
        """Test distribution with empty frame list."""
        mock_healthy.return_value = []

        distributor = MultiGPUDistributor()
        result = distributor.distribute_frames(
            frames=[],
            process_fn=lambda x, y, z: (x, True, None),
            output_dir=Path("/output"),
        )

        assert result.total_frames == 0

    @patch.object(GPUManager, "get_healthy_gpus")
    def test_distribute_no_gpus(self, mock_healthy):
        """Test distribution with no available GPUs."""
        mock_healthy.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [Path(tmpdir) / f"frame_{i}.png" for i in range(3)]
            for f in frames:
                f.touch()

            distributor = MultiGPUDistributor()
            result = distributor.distribute_frames(
                frames=frames,
                process_fn=lambda x, y, z: (x, True, None),
                output_dir=Path(tmpdir) / "output",
            )

            # All frames should be errors
            assert len(result.errors) == 3


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("framewright.utils.multi_gpu.GPUManager")
    def test_detect_gpus_convenience(self, mock_manager_cls):
        """Test detect_gpus convenience function."""
        mock_manager = MagicMock()
        mock_manager.detect_gpus.return_value = [
            GPUInfo(id=0, name="GPU 0", total_vram_mb=8000, free_vram_mb=4000, utilization_pct=50.0)
        ]
        mock_manager_cls.return_value = mock_manager

        gpus = detect_gpus()

        assert len(gpus) == 1
        mock_manager.detect_gpus.assert_called_once()

    @patch("framewright.utils.multi_gpu.GPUManager")
    def test_get_optimal_gpu_convenience(self, mock_manager_cls):
        """Test get_optimal_gpu convenience function."""
        mock_manager = MagicMock()
        mock_manager.get_optimal_gpu.return_value = 1
        mock_manager_cls.return_value = mock_manager

        optimal = get_optimal_gpu(LoadBalanceStrategy.LEAST_LOADED)

        assert optimal == 1
        mock_manager.get_optimal_gpu.assert_called_once_with(LoadBalanceStrategy.LEAST_LOADED)


class TestLoadBalanceStrategy:
    """Tests for LoadBalanceStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all expected strategies are defined."""
        assert LoadBalanceStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalanceStrategy.LEAST_LOADED.value == "least_loaded"
        assert LoadBalanceStrategy.VRAM_AWARE.value == "vram_aware"
        assert LoadBalanceStrategy.WEIGHTED.value == "weighted"


class TestFrameAssignment:
    """Tests for frame assignment logic."""

    @patch.object(GPUManager, "get_healthy_gpus")
    @patch.object(GPUManager, "detect_gpus")
    def test_round_robin_assignment(self, mock_detect, mock_healthy):
        """Test round-robin frame assignment."""
        gpus = [
            GPUInfo(id=0, name="GPU 0", total_vram_mb=8000, free_vram_mb=4000, utilization_pct=50.0),
            GPUInfo(id=1, name="GPU 1", total_vram_mb=8000, free_vram_mb=4000, utilization_pct=50.0),
        ]
        mock_detect.return_value = gpus
        mock_healthy.return_value = gpus

        distributor = MultiGPUDistributor(strategy=LoadBalanceStrategy.ROUND_ROBIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [Path(tmpdir) / f"frame_{i}.png" for i in range(4)]

            assignments = distributor._assign_frames(frames, gpus)

            # Should be evenly distributed
            assert len(assignments[0]) == 2
            assert len(assignments[1]) == 2

    @patch.object(GPUManager, "get_healthy_gpus")
    @patch.object(GPUManager, "detect_gpus")
    def test_vram_aware_assignment(self, mock_detect, mock_healthy):
        """Test VRAM-aware frame assignment."""
        gpus = [
            GPUInfo(id=0, name="GPU 0", total_vram_mb=8000, free_vram_mb=2000, utilization_pct=50.0),
            GPUInfo(id=1, name="GPU 1", total_vram_mb=8000, free_vram_mb=6000, utilization_pct=50.0),
        ]
        mock_detect.return_value = gpus
        mock_healthy.return_value = gpus

        distributor = MultiGPUDistributor(strategy=LoadBalanceStrategy.VRAM_AWARE)

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [Path(tmpdir) / f"frame_{i}.png" for i in range(8)]

            assignments = distributor._assign_frames(frames, gpus)

            # GPU 1 has 3x the free VRAM, should get more frames
            assert len(assignments[1]) > len(assignments[0])


class TestGPUSelector:
    """Tests for GPUSelector class."""

    @patch.object(GPUManager, "get_gpu_info")
    @patch.object(GPUManager, "get_optimal_gpu")
    def test_select_by_index_valid(self, mock_optimal, mock_get_info):
        """Test selecting a valid GPU by index."""
        from framewright.utils.multi_gpu import GPUSelector

        mock_gpu = GPUInfo(
            id=0, name="Test GPU", total_vram_mb=8000,
            free_vram_mb=4000, utilization_pct=50.0
        )
        mock_get_info.return_value = mock_gpu

        selector = GPUSelector()
        result = selector.select_by_index(0)

        assert result is not None
        assert result.id == 0
        mock_get_info.assert_called_with(0, refresh=True)

    def test_select_by_index_negative(self):
        """Test selecting with negative index raises error."""
        from framewright.utils.multi_gpu import GPUSelector

        selector = GPUSelector()
        with pytest.raises(ValueError, match="Invalid GPU index"):
            selector.select_by_index(-1)

    @patch.object(GPUManager, "get_gpu_info")
    @patch.object(GPUManager, "get_optimal_gpu")
    def test_validate_gpu_healthy(self, mock_optimal, mock_get_info):
        """Test GPU validation for healthy GPU."""
        from framewright.utils.multi_gpu import GPUSelector

        mock_gpu = GPUInfo(
            id=0, name="Test GPU", total_vram_mb=8000,
            free_vram_mb=4000, utilization_pct=50.0,
            temperature_c=65.0
        )
        mock_get_info.return_value = mock_gpu

        selector = GPUSelector()
        assert selector.validate_gpu(0) is True

    @patch.object(GPUManager, "get_gpu_info")
    @patch.object(GPUManager, "get_optimal_gpu")
    def test_get_gpu_for_task_specific(self, mock_optimal, mock_get_info):
        """Test getting GPU for task with specific GPU ID."""
        from framewright.utils.multi_gpu import GPUSelector

        mock_gpu = GPUInfo(
            id=1, name="Test GPU", total_vram_mb=8000,
            free_vram_mb=4000, utilization_pct=50.0,
            temperature_c=65.0
        )
        mock_get_info.return_value = mock_gpu
        mock_optimal.return_value = 0

        selector = GPUSelector()
        gpu_id, use_multi = selector.get_gpu_for_task(gpu_id=1, multi_gpu=False)

        assert gpu_id == 1
        assert use_multi is False

    @patch.object(GPUManager, "get_optimal_gpu")
    def test_get_gpu_for_task_multi_gpu(self, mock_optimal):
        """Test getting GPU for task with multi-GPU mode."""
        from framewright.utils.multi_gpu import GPUSelector

        mock_optimal.return_value = 0

        selector = GPUSelector()
        gpu_id, use_multi = selector.get_gpu_for_task(gpu_id=None, multi_gpu=True)

        assert gpu_id == 0
        assert use_multi is True


class TestMultiGPUManagerEnhanced:
    """Tests for enhanced MultiGPUManager class."""

    @patch("framewright.utils.multi_gpu.subprocess.run")
    @patch("framewright.utils.multi_gpu.shutil.which")
    def test_format_gpu_table(self, mock_which, mock_run):
        """Test GPU table formatting."""
        from framewright.utils.multi_gpu import MultiGPUManager

        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4090, 24576, 20000, 25, 65, 4, 8.9\n"
        )

        manager = MultiGPUManager()
        table = manager.format_gpu_table()

        assert "Available GPUs" in table
        assert "ID" in table
        assert "Name" in table
        assert "Memory" in table
        assert "Compute" in table
        assert "Status" in table

    def test_update_processing_speed(self):
        """Test processing speed update for dynamic rebalancing."""
        from framewright.utils.multi_gpu import MultiGPUManager

        manager = MultiGPUManager(gpu_ids=[0, 1])
        manager.update_processing_speed(0, 10.0)
        manager.update_processing_speed(1, 5.0)

        assert manager._processing_speeds[0] == 10.0
        assert manager._processing_speeds[1] == 5.0


class TestConvenienceFunctionsEnhanced:
    """Tests for enhanced convenience functions."""

    @patch("framewright.utils.multi_gpu.MultiGPUManager")
    def test_list_gpus(self, mock_manager_cls):
        """Test list_gpus convenience function."""
        from framewright.utils.multi_gpu import list_gpus

        mock_manager = MagicMock()
        mock_manager.format_gpu_table.return_value = "Available GPUs:\n..."
        mock_manager_cls.return_value = mock_manager

        result = list_gpus()

        assert "Available GPUs" in result
        mock_manager.format_gpu_table.assert_called_once()

    @patch("framewright.utils.multi_gpu.GPUSelector")
    def test_select_gpu(self, mock_selector_cls):
        """Test select_gpu convenience function."""
        from framewright.utils.multi_gpu import select_gpu

        mock_selector = MagicMock()
        mock_selector.get_gpu_for_task.return_value = (0, True)
        mock_selector_cls.return_value = mock_selector

        gpu_id, use_multi = select_gpu(gpu_id=0, multi_gpu=True)

        assert gpu_id == 0
        assert use_multi is True
        mock_selector.get_gpu_for_task.assert_called_once_with(0, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
