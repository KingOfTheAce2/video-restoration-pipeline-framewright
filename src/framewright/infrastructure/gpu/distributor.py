"""Multi-GPU Distributor for FrameWright.

Provides work distribution across multiple GPUs of any vendor (NVIDIA, AMD, Intel),
with support for mixed GPU configurations and automatic load balancing.

Key features:
- Detect and enumerate all available GPUs
- Distribute frame processing across multiple GPUs
- Support multiple distribution strategies
- Handle GPU failures gracefully
- Support mixed GPU types (e.g., NVIDIA + AMD)
"""

import gc
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from .detector import (
    BackendType,
    DeviceInfo,
    GPUVendor,
    HardwareInfo,
    get_hardware_info,
)
from .backends.base import Backend, get_backend

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DistributionStrategy(Enum):
    """Strategy for distributing work across GPUs.

    Attributes:
        ROUND_ROBIN: Distribute frames evenly in order
        LOAD_BALANCED: Balance based on current GPU load
        MEMORY_AWARE: Balance based on available memory
        SPEED_AWARE: Balance based on GPU processing speed
        PRIORITY: Prefer faster/larger GPUs
    """
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    MEMORY_AWARE = "memory_aware"
    SPEED_AWARE = "speed_aware"
    PRIORITY = "priority"


@dataclass
class GPUStats:
    """Runtime statistics for a GPU.

    Attributes:
        device_id: GPU device index
        vendor: GPU vendor
        name: GPU name
        total_memory_mb: Total VRAM
        used_memory_mb: Currently used VRAM
        frames_processed: Number of frames processed
        total_time_seconds: Total processing time
        avg_time_per_frame: Average time per frame
        errors: Number of errors encountered
        is_healthy: GPU is functioning correctly
    """
    device_id: int
    vendor: GPUVendor
    name: str
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    frames_processed: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_frame: float = 0.0
    errors: int = 0
    is_healthy: bool = True

    def update_timing(self, elapsed: float):
        """Update timing statistics."""
        self.frames_processed += 1
        self.total_time_seconds += elapsed
        self.avg_time_per_frame = self.total_time_seconds / self.frames_processed


@dataclass
class DistributionPlan:
    """Plan for distributing work across GPUs.

    Attributes:
        frame_assignments: Dict mapping frame index to GPU device_id
        gpu_workloads: Dict mapping GPU device_id to list of frame indices
        total_frames: Total number of frames
        estimated_time_seconds: Estimated total processing time
    """
    frame_assignments: Dict[int, int] = field(default_factory=dict)
    gpu_workloads: Dict[int, List[int]] = field(default_factory=dict)
    total_frames: int = 0
    estimated_time_seconds: float = 0.0


@dataclass
class ProcessingResult:
    """Result of processing a frame on a GPU.

    Attributes:
        frame_index: Index of the processed frame
        device_id: GPU that processed the frame
        success: Whether processing succeeded
        output: Processing output (if successful)
        error: Error message (if failed)
        elapsed_seconds: Processing time
    """
    frame_index: int
    device_id: int
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class GPUDistributor:
    """Detect and manage distribution of work across multiple GPUs.

    This class handles GPU detection, work distribution planning, and
    coordination of multi-GPU processing.

    Example:
        >>> distributor = GPUDistributor()
        >>> gpus = distributor.detect_all_gpus()
        >>> plan = distributor.get_optimal_distribution(1000)
        >>> print(f"Distributing 1000 frames across {len(gpus)} GPUs")
    """

    def __init__(
        self,
        strategy: DistributionStrategy = DistributionStrategy.LOAD_BALANCED,
        excluded_devices: Optional[List[int]] = None,
    ):
        """Initialize the GPU distributor.

        Args:
            strategy: Distribution strategy to use
            excluded_devices: List of device IDs to exclude
        """
        self.strategy = strategy
        self.excluded_devices = set(excluded_devices or [])
        self._devices: List[DeviceInfo] = []
        self._stats: Dict[int, GPUStats] = {}
        self._backends: Dict[int, Backend] = {}
        self._lock = threading.Lock()
        self._initialized = False

    def detect_all_gpus(self, force_refresh: bool = False) -> List[DeviceInfo]:
        """Detect all available GPUs across all vendors.

        Args:
            force_refresh: Force re-detection even if cached

        Returns:
            List of DeviceInfo for all detected GPUs
        """
        if self._devices and not force_refresh:
            return self._devices

        with self._lock:
            hw_info = get_hardware_info()
            self._devices = [
                d for d in hw_info.all_devices
                if d.index not in self.excluded_devices
            ]

            # Initialize stats for each device
            for device in self._devices:
                if device.index not in self._stats:
                    self._stats[device.index] = GPUStats(
                        device_id=device.index,
                        vendor=device.vendor,
                        name=device.name,
                        total_memory_mb=device.total_memory_mb,
                        used_memory_mb=device.total_memory_mb - device.free_memory_mb,
                    )

            logger.info(f"Detected {len(self._devices)} GPUs: "
                        f"{[d.name for d in self._devices]}")

            return self._devices

    def get_device_count(self) -> int:
        """Get the number of available GPUs.

        Returns:
            Number of detected GPUs
        """
        if not self._devices:
            self.detect_all_gpus()
        return len(self._devices)

    def get_device_info(self, device_id: int) -> Optional[DeviceInfo]:
        """Get information about a specific GPU.

        Args:
            device_id: Device index

        Returns:
            DeviceInfo or None if not found
        """
        if not self._devices:
            self.detect_all_gpus()

        for device in self._devices:
            if device.index == device_id:
                return device
        return None

    def get_gpu_stats(self, device_id: int) -> Optional[GPUStats]:
        """Get runtime statistics for a GPU.

        Args:
            device_id: Device index

        Returns:
            GPUStats or None if not found
        """
        return self._stats.get(device_id)

    def get_all_stats(self) -> Dict[int, GPUStats]:
        """Get statistics for all GPUs.

        Returns:
            Dict mapping device_id to GPUStats
        """
        return self._stats.copy()

    def distribute_frames(
        self,
        frame_count: int,
        strategy: Optional[DistributionStrategy] = None,
    ) -> DistributionPlan:
        """Create a distribution plan for frames across GPUs.

        Args:
            frame_count: Number of frames to distribute
            strategy: Override default strategy

        Returns:
            DistributionPlan with frame assignments
        """
        if not self._devices:
            self.detect_all_gpus()

        strategy = strategy or self.strategy
        plan = DistributionPlan(total_frames=frame_count)

        if not self._devices:
            logger.warning("No GPUs available for distribution")
            return plan

        # Get healthy devices only
        healthy_devices = [
            d for d in self._devices
            if self._stats.get(d.index, GPUStats(0, GPUVendor.UNKNOWN, "")).is_healthy
        ]

        if not healthy_devices:
            healthy_devices = self._devices  # Use all if none marked healthy

        if strategy == DistributionStrategy.ROUND_ROBIN:
            plan = self._distribute_round_robin(frame_count, healthy_devices)
        elif strategy == DistributionStrategy.LOAD_BALANCED:
            plan = self._distribute_load_balanced(frame_count, healthy_devices)
        elif strategy == DistributionStrategy.MEMORY_AWARE:
            plan = self._distribute_memory_aware(frame_count, healthy_devices)
        elif strategy == DistributionStrategy.SPEED_AWARE:
            plan = self._distribute_speed_aware(frame_count, healthy_devices)
        elif strategy == DistributionStrategy.PRIORITY:
            plan = self._distribute_priority(frame_count, healthy_devices)

        return plan

    def _distribute_round_robin(
        self,
        frame_count: int,
        devices: List[DeviceInfo],
    ) -> DistributionPlan:
        """Distribute frames in round-robin fashion."""
        plan = DistributionPlan(total_frames=frame_count)

        for device in devices:
            plan.gpu_workloads[device.index] = []

        for frame_idx in range(frame_count):
            device_idx = frame_idx % len(devices)
            device_id = devices[device_idx].index
            plan.frame_assignments[frame_idx] = device_id
            plan.gpu_workloads[device_id].append(frame_idx)

        return plan

    def _distribute_load_balanced(
        self,
        frame_count: int,
        devices: List[DeviceInfo],
    ) -> DistributionPlan:
        """Distribute frames based on current GPU load."""
        plan = DistributionPlan(total_frames=frame_count)

        for device in devices:
            plan.gpu_workloads[device.index] = []

        # Calculate load scores (lower is better)
        load_scores = {}
        for device in devices:
            stats = self._stats.get(device.index)
            if stats:
                # Consider processing speed and memory
                speed_factor = 1.0 / (stats.avg_time_per_frame + 0.001)
                memory_factor = device.free_memory_mb / max(device.total_memory_mb, 1)
                load_scores[device.index] = speed_factor * memory_factor
            else:
                load_scores[device.index] = 1.0

        # Normalize scores to get weights
        total_score = sum(load_scores.values())
        weights = {
            d_id: score / total_score
            for d_id, score in load_scores.items()
        }

        # Distribute frames according to weights
        assigned = 0
        for device_id, weight in weights.items():
            count = int(frame_count * weight)
            if device_id == list(weights.keys())[-1]:
                count = frame_count - assigned  # Assign remaining to last

            for _ in range(count):
                if assigned < frame_count:
                    plan.frame_assignments[assigned] = device_id
                    plan.gpu_workloads[device_id].append(assigned)
                    assigned += 1

        return plan

    def _distribute_memory_aware(
        self,
        frame_count: int,
        devices: List[DeviceInfo],
    ) -> DistributionPlan:
        """Distribute frames based on available memory."""
        plan = DistributionPlan(total_frames=frame_count)

        for device in devices:
            plan.gpu_workloads[device.index] = []

        # Calculate memory-based weights
        total_free_memory = sum(d.free_memory_mb for d in devices)
        if total_free_memory == 0:
            return self._distribute_round_robin(frame_count, devices)

        weights = {
            d.index: d.free_memory_mb / total_free_memory
            for d in devices
        }

        # Distribute frames according to weights
        assigned = 0
        for device_id, weight in weights.items():
            count = int(frame_count * weight)
            if device_id == list(weights.keys())[-1]:
                count = frame_count - assigned

            for _ in range(count):
                if assigned < frame_count:
                    plan.frame_assignments[assigned] = device_id
                    plan.gpu_workloads[device_id].append(assigned)
                    assigned += 1

        return plan

    def _distribute_speed_aware(
        self,
        frame_count: int,
        devices: List[DeviceInfo],
    ) -> DistributionPlan:
        """Distribute frames based on GPU processing speed."""
        plan = DistributionPlan(total_frames=frame_count)

        for device in devices:
            plan.gpu_workloads[device.index] = []

        # Calculate speed-based weights
        speeds = {}
        for device in devices:
            stats = self._stats.get(device.index)
            if stats and stats.avg_time_per_frame > 0:
                speeds[device.index] = 1.0 / stats.avg_time_per_frame
            else:
                # Default speed based on device tier
                if device.total_memory_mb >= 16384:
                    speeds[device.index] = 100.0
                elif device.total_memory_mb >= 8192:
                    speeds[device.index] = 60.0
                else:
                    speeds[device.index] = 30.0

        total_speed = sum(speeds.values())
        weights = {d_id: s / total_speed for d_id, s in speeds.items()}

        # Distribute frames according to weights
        assigned = 0
        for device_id, weight in weights.items():
            count = int(frame_count * weight)
            if device_id == list(weights.keys())[-1]:
                count = frame_count - assigned

            for _ in range(count):
                if assigned < frame_count:
                    plan.frame_assignments[assigned] = device_id
                    plan.gpu_workloads[device_id].append(assigned)
                    assigned += 1

        return plan

    def _distribute_priority(
        self,
        frame_count: int,
        devices: List[DeviceInfo],
    ) -> DistributionPlan:
        """Distribute frames preferring faster/larger GPUs."""
        # Sort devices by VRAM (larger first)
        sorted_devices = sorted(
            devices,
            key=lambda d: d.total_memory_mb,
            reverse=True,
        )

        # Weight heavily toward first devices
        weights = []
        remaining = 1.0
        for i, device in enumerate(sorted_devices):
            if i == len(sorted_devices) - 1:
                weights.append(remaining)
            else:
                weight = remaining * 0.6  # 60% of remaining to next priority
                weights.append(weight)
                remaining -= weight

        plan = DistributionPlan(total_frames=frame_count)
        for device in sorted_devices:
            plan.gpu_workloads[device.index] = []

        assigned = 0
        for device, weight in zip(sorted_devices, weights):
            count = int(frame_count * weight)
            if device == sorted_devices[-1]:
                count = frame_count - assigned

            for _ in range(count):
                if assigned < frame_count:
                    plan.frame_assignments[assigned] = device.index
                    plan.gpu_workloads[device.index].append(assigned)
                    assigned += 1

        return plan

    def get_optimal_distribution(
        self,
        frame_count: int,
        frame_memory_mb: float = 100.0,
    ) -> DistributionPlan:
        """Get optimal distribution considering all factors.

        Args:
            frame_count: Number of frames to process
            frame_memory_mb: Estimated memory per frame in MB

        Returns:
            Optimized DistributionPlan
        """
        if not self._devices:
            self.detect_all_gpus()

        # Use speed-aware if we have timing data
        has_timing_data = any(
            stats.frames_processed > 10
            for stats in self._stats.values()
        )

        if has_timing_data:
            return self.distribute_frames(frame_count, DistributionStrategy.SPEED_AWARE)

        # Check memory constraints
        memory_constrained = any(
            d.free_memory_mb < frame_memory_mb * 10
            for d in self._devices
        )

        if memory_constrained:
            return self.distribute_frames(frame_count, DistributionStrategy.MEMORY_AWARE)

        # Default to load balanced
        return self.distribute_frames(frame_count, DistributionStrategy.LOAD_BALANCED)

    def collect_results(
        self,
        futures: List[Future],
        timeout: Optional[float] = None,
    ) -> List[ProcessingResult]:
        """Collect results from processing futures.

        Args:
            futures: List of Future objects from processing
            timeout: Optional timeout in seconds

        Returns:
            List of ProcessingResult objects
        """
        results = []

        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)

                # Update stats
                if result.device_id in self._stats:
                    stats = self._stats[result.device_id]
                    if result.success:
                        stats.update_timing(result.elapsed_seconds)
                    else:
                        stats.errors += 1
                        if stats.errors > 5:
                            stats.is_healthy = False

            except Exception as e:
                logger.error(f"Error collecting result: {e}")
                results.append(ProcessingResult(
                    frame_index=-1,
                    device_id=-1,
                    success=False,
                    error=str(e),
                ))

        return results

    def mark_device_unhealthy(self, device_id: int) -> None:
        """Mark a device as unhealthy (exclude from future work).

        Args:
            device_id: Device to mark
        """
        if device_id in self._stats:
            self._stats[device_id].is_healthy = False
            logger.warning(f"GPU {device_id} marked as unhealthy")

    def mark_device_healthy(self, device_id: int) -> None:
        """Mark a device as healthy (include in future work).

        Args:
            device_id: Device to mark
        """
        if device_id in self._stats:
            self._stats[device_id].is_healthy = True
            self._stats[device_id].errors = 0

    def reset_stats(self) -> None:
        """Reset all GPU statistics."""
        for stats in self._stats.values():
            stats.frames_processed = 0
            stats.total_time_seconds = 0.0
            stats.avg_time_per_frame = 0.0
            stats.errors = 0
            stats.is_healthy = True


class MultiGPUProcessor:
    """Process frames across multiple GPUs with automatic distribution.

    This class provides high-level frame processing across multiple GPUs
    with automatic work distribution, error handling, and result collection.

    Example:
        >>> processor = MultiGPUProcessor()
        >>> processor.initialize()
        >>> results = processor.process_frames(
        ...     frames=[np.zeros((480, 640, 3)) for _ in range(100)],
        ...     process_func=my_upscale_function,
        ... )
        >>> processor.cleanup()
    """

    def __init__(
        self,
        strategy: DistributionStrategy = DistributionStrategy.LOAD_BALANCED,
        max_workers_per_gpu: int = 1,
        excluded_devices: Optional[List[int]] = None,
    ):
        """Initialize the multi-GPU processor.

        Args:
            strategy: Work distribution strategy
            max_workers_per_gpu: Max concurrent workers per GPU
            excluded_devices: Devices to exclude from processing
        """
        self.distributor = GPUDistributor(strategy, excluded_devices)
        self.max_workers_per_gpu = max_workers_per_gpu
        self._backends: Dict[int, Backend] = {}
        self._executors: Dict[int, ThreadPoolExecutor] = {}
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize backends for all available GPUs.

        Returns:
            True if at least one GPU was initialized
        """
        if self._initialized:
            return True

        with self._lock:
            devices = self.distributor.detect_all_gpus()

            if not devices:
                logger.warning("No GPUs detected")
                return False

            for device in devices:
                try:
                    # Get appropriate backend for device
                    backend = get_backend(
                        backend_type=self._get_backend_type(device.vendor),
                        device_id=device.index,
                        force_new=True,
                    )

                    if backend.initialize():
                        self._backends[device.index] = backend
                        self._executors[device.index] = ThreadPoolExecutor(
                            max_workers=self.max_workers_per_gpu,
                            thread_name_prefix=f"gpu_{device.index}_",
                        )
                        logger.info(f"Initialized GPU {device.index}: {device.name}")
                    else:
                        logger.warning(f"Failed to initialize GPU {device.index}")

                except Exception as e:
                    logger.error(f"Error initializing GPU {device.index}: {e}")

            self._initialized = len(self._backends) > 0
            return self._initialized

    def _get_backend_type(self, vendor: GPUVendor) -> BackendType:
        """Get the appropriate backend type for a vendor."""
        vendor_backends = {
            GPUVendor.NVIDIA: BackendType.CUDA,
            GPUVendor.AMD: BackendType.ROCM,
            GPUVendor.INTEL: BackendType.ONEAPI,
            GPUVendor.APPLE: BackendType.METAL,
            GPUVendor.UNKNOWN: BackendType.CPU,
        }
        return vendor_backends.get(vendor, BackendType.CPU)

    def cleanup(self) -> None:
        """Clean up all GPU resources."""
        with self._lock:
            # Shutdown executors
            for executor in self._executors.values():
                executor.shutdown(wait=True)
            self._executors.clear()

            # Cleanup backends
            for backend in self._backends.values():
                backend.cleanup()
            self._backends.clear()

            self._initialized = False
            gc.collect()

    def process_frames(
        self,
        frames: List[np.ndarray],
        process_func: Callable[[np.ndarray, int], np.ndarray],
        callback: Optional[Callable[[ProcessingResult], None]] = None,
        timeout_per_frame: float = 60.0,
    ) -> List[ProcessingResult]:
        """Process frames across all available GPUs.

        Args:
            frames: List of frames to process
            process_func: Function to process each frame
                          Signature: (frame, device_id) -> processed_frame
            callback: Optional callback for each completed frame
            timeout_per_frame: Timeout per frame in seconds

        Returns:
            List of ProcessingResult objects
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize multi-GPU processor")

        # Get distribution plan
        plan = self.distributor.get_optimal_distribution(len(frames))

        # Submit all tasks
        futures = []
        for frame_idx, device_id in plan.frame_assignments.items():
            if device_id not in self._executors:
                continue

            future = self._executors[device_id].submit(
                self._process_single_frame,
                frames[frame_idx],
                frame_idx,
                device_id,
                process_func,
            )
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures, timeout=timeout_per_frame * len(frames)):
            try:
                result = future.result(timeout=timeout_per_frame)
                results.append(result)

                # Update distributor stats
                if result.device_id in self.distributor._stats:
                    stats = self.distributor._stats[result.device_id]
                    if result.success:
                        stats.update_timing(result.elapsed_seconds)
                    else:
                        stats.errors += 1

                # Call callback
                if callback:
                    callback(result)

            except Exception as e:
                logger.error(f"Frame processing error: {e}")

        # Sort results by frame index
        results.sort(key=lambda r: r.frame_index)
        return results

    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        device_id: int,
        process_func: Callable,
    ) -> ProcessingResult:
        """Process a single frame on a specific GPU."""
        start_time = time.time()

        try:
            output = process_func(frame, device_id)
            elapsed = time.time() - start_time

            return ProcessingResult(
                frame_index=frame_idx,
                device_id=device_id,
                success=True,
                output=output,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error processing frame {frame_idx} on GPU {device_id}: {e}")

            return ProcessingResult(
                frame_index=frame_idx,
                device_id=device_id,
                success=False,
                error=str(e),
                elapsed_seconds=elapsed,
            )

    def process_batch(
        self,
        batch: np.ndarray,
        process_func: Callable[[np.ndarray, int], np.ndarray],
        device_id: Optional[int] = None,
    ) -> np.ndarray:
        """Process a batch on a single GPU.

        Args:
            batch: Batch of frames (N, H, W, C)
            process_func: Function to process the batch
            device_id: GPU to use (auto-selected if None)

        Returns:
            Processed batch
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize")

        # Auto-select device if not specified
        if device_id is None:
            # Find device with most free memory
            best_device = None
            best_memory = 0
            for d_id, backend in self._backends.items():
                mem_info = backend.get_memory_info()
                if mem_info["free_mb"] > best_memory:
                    best_memory = mem_info["free_mb"]
                    best_device = d_id
            device_id = best_device or list(self._backends.keys())[0]

        return process_func(batch, device_id)

    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU device IDs."""
        return list(self._backends.keys())

    def get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        return len(self._backends)

    def get_backend(self, device_id: int) -> Optional[Backend]:
        """Get the backend for a specific device."""
        return self._backends.get(device_id)

    def get_stats(self) -> Dict[int, GPUStats]:
        """Get processing statistics for all GPUs."""
        return self.distributor.get_all_stats()


# =============================================================================
# Utility Functions
# =============================================================================

def detect_multi_gpu_support() -> Dict[str, Any]:
    """Detect multi-GPU support and capabilities.

    Returns:
        Dictionary with:
        - gpu_count: Number of GPUs
        - vendors: List of GPU vendors
        - total_memory_mb: Total VRAM across all GPUs
        - mixed_vendors: True if multiple vendor types
        - gpus: List of GPU info dicts
    """
    distributor = GPUDistributor()
    devices = distributor.detect_all_gpus()

    vendors = list(set(d.vendor for d in devices))
    total_memory = sum(d.total_memory_mb for d in devices)

    return {
        "gpu_count": len(devices),
        "vendors": [v.value for v in vendors],
        "total_memory_mb": total_memory,
        "mixed_vendors": len(vendors) > 1,
        "gpus": [
            {
                "device_id": d.index,
                "name": d.name,
                "vendor": d.vendor.value,
                "memory_mb": d.total_memory_mb,
            }
            for d in devices
        ],
    }


def get_optimal_gpu() -> int:
    """Get the device ID of the optimal GPU for processing.

    Returns:
        Device ID of the best GPU (0 if no GPU available)
    """
    hw_info = get_hardware_info()
    if hw_info.primary_device:
        return hw_info.primary_device.index
    return 0


def estimate_multi_gpu_speedup(gpu_count: int) -> float:
    """Estimate theoretical speedup from multi-GPU processing.

    Args:
        gpu_count: Number of GPUs

    Returns:
        Estimated speedup factor (e.g., 3.5x for 4 GPUs)
    """
    if gpu_count <= 1:
        return 1.0

    # Account for overhead (communication, synchronization)
    # Efficiency decreases as GPU count increases
    efficiency = 0.95 - (0.05 * (gpu_count - 2))
    efficiency = max(efficiency, 0.7)  # Minimum 70% efficiency

    return gpu_count * efficiency


__all__ = [
    "DistributionStrategy",
    "GPUStats",
    "DistributionPlan",
    "ProcessingResult",
    "GPUDistributor",
    "MultiGPUProcessor",
    "detect_multi_gpu_support",
    "get_optimal_gpu",
    "estimate_multi_gpu_speedup",
]
