"""Multi-GPU Distribution System for FrameWright.

Provides advanced GPU distribution capabilities including:
- Automatic GPU detection and VRAM tracking
- Multiple load balancing strategies (round-robin, least-loaded, VRAM-aware)
- Work-stealing for dynamic load balancing
- Graceful failure handling with retry on alternate GPUs
- Support for heterogeneous GPU configurations
"""
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)


# Type variable for generic process function
T = TypeVar("T")


class LoadBalanceStrategy(Enum):
    """Load balancing strategies for GPU distribution."""

    ROUND_ROBIN = "round_robin"  # Simple round-robin distribution
    LEAST_LOADED = "least_loaded"  # Assign to GPU with lowest utilization
    VRAM_AWARE = "vram_aware"  # Prioritize GPUs with most free VRAM
    WEIGHTED = "weighted"  # Weight by VRAM capacity and utilization


@dataclass
class GPUInfo:
    """Detailed GPU device information.

    Attributes:
        id: GPU device index
        name: GPU model name
        total_vram_mb: Total VRAM in megabytes
        free_vram_mb: Currently free VRAM in megabytes
        utilization_pct: Current GPU utilization percentage (0-100)
        temperature_c: Current temperature in Celsius (optional)
        pcie_bandwidth_gbps: PCIe bandwidth in GB/s (optional)
        compute_capability: CUDA compute capability (optional)
    """

    id: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    utilization_pct: float
    temperature_c: Optional[float] = None
    pcie_bandwidth_gbps: Optional[float] = None
    compute_capability: Optional[str] = None

    @property
    def used_vram_mb(self) -> int:
        """Calculate used VRAM."""
        return self.total_vram_mb - self.free_vram_mb

    @property
    def vram_usage_pct(self) -> float:
        """Calculate VRAM usage percentage."""
        if self.total_vram_mb == 0:
            return 0.0
        return (self.used_vram_mb / self.total_vram_mb) * 100

    @property
    def is_healthy(self) -> bool:
        """Check if GPU is in healthy operating state."""
        # Consider unhealthy if utilization stuck at 100% or temperature too high
        if self.temperature_c is not None and self.temperature_c > 90:
            return False
        return True

    @property
    def effective_capacity(self) -> float:
        """Calculate effective processing capacity score (0-1).

        Higher score means more available capacity.
        """
        # Weight VRAM availability more heavily than utilization
        vram_score = self.free_vram_mb / max(self.total_vram_mb, 1)
        util_score = 1.0 - (self.utilization_pct / 100.0)
        return (vram_score * 0.7) + (util_score * 0.3)


@dataclass
class DistributionResult:
    """Result of frame distribution across GPUs.

    Attributes:
        frames_per_gpu: Dictionary mapping GPU ID to list of processed frame paths
        total_time: Total processing time in seconds
        speedup_factor: Speedup compared to single GPU (estimated)
        gpu_utilization: Average utilization per GPU during processing
        errors: Dictionary of frame path to error message for failed frames
        retried_frames: Frames that were retried on different GPUs
    """

    frames_per_gpu: Dict[int, List[Path]] = field(default_factory=dict)
    total_time: float = 0.0
    speedup_factor: float = 1.0
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    retried_frames: List[Path] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        """Total number of successfully processed frames."""
        return sum(len(frames) for frames in self.frames_per_gpu.values())

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.total_frames + len(self.errors)
        if total == 0:
            return 100.0
        return (self.total_frames / total) * 100

    def summary(self) -> str:
        """Generate human-readable summary."""
        gpu_counts = ", ".join(
            f"GPU{gid}: {len(frames)}" for gid, frames in self.frames_per_gpu.items()
        )
        return (
            f"Processed {self.total_frames} frames across {len(self.frames_per_gpu)} GPUs "
            f"({gpu_counts}) in {self.total_time:.1f}s "
            f"(speedup: {self.speedup_factor:.2f}x, success: {self.success_rate:.1f}%)"
        )


@dataclass
class WorkItem:
    """A unit of work for GPU processing.

    Attributes:
        frame_path: Path to the input frame
        output_dir: Directory for output
        priority: Processing priority (lower = higher priority)
        assigned_gpu: GPU assigned to process this item (None if unassigned)
        attempts: Number of processing attempts
        failed_gpus: GPUs that failed to process this item
    """

    frame_path: Path
    output_dir: Path
    priority: int = 0
    assigned_gpu: Optional[int] = None
    attempts: int = 0
    failed_gpus: List[int] = field(default_factory=list)

    @property
    def can_retry(self) -> bool:
        """Check if this work item can be retried."""
        return self.attempts < 3


class GPUManager:
    """Manager for detecting and monitoring GPUs.

    Provides GPU detection, VRAM monitoring, and health tracking
    for multi-GPU processing scenarios.
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        refresh_interval: float = 5.0,
    ):
        """Initialize GPU manager.

        Args:
            gpu_ids: Specific GPU IDs to manage (None = auto-detect all)
            refresh_interval: Interval between automatic GPU info refreshes
        """
        self._gpu_ids = gpu_ids
        self._refresh_interval = refresh_interval
        self._gpu_cache: Dict[int, GPUInfo] = {}
        self._last_refresh: float = 0
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Initial detection
        self._refresh_gpu_info()

    @property
    def gpu_ids(self) -> List[int]:
        """Get list of managed GPU IDs."""
        if self._gpu_ids is not None:
            return self._gpu_ids
        return list(self._gpu_cache.keys())

    @property
    def gpu_count(self) -> int:
        """Get number of managed GPUs."""
        return len(self.gpu_ids)

    @property
    def is_multi_gpu(self) -> bool:
        """Check if multiple GPUs are available."""
        return self.gpu_count > 1

    def detect_gpus(self) -> List[GPUInfo]:
        """Detect all available GPUs using nvidia-smi.

        Returns:
            List of GPUInfo objects for each detected GPU
        """
        if not self._is_nvidia_smi_available():
            logger.warning("nvidia-smi not available, no GPUs detected")
            return []

        try:
            # Query detailed GPU information
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,"
                "temperature.gpu,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]

                try:
                    gpu = GPUInfo(
                        id=int(parts[0]),
                        name=parts[1],
                        total_vram_mb=int(parts[2]),
                        free_vram_mb=int(parts[3]),
                        utilization_pct=float(parts[4]) if parts[4] != "[N/A]" else 0.0,
                        temperature_c=float(parts[5]) if parts[5] != "[N/A]" else None,
                    )

                    # Filter to specified GPUs if provided
                    if self._gpu_ids is None or gpu.id in self._gpu_ids:
                        gpus.append(gpu)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse GPU info: {e}")
                    continue

            return gpus

        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return []

    def get_gpu_info(self, gpu_id: int, refresh: bool = False) -> Optional[GPUInfo]:
        """Get information for a specific GPU.

        Args:
            gpu_id: GPU device ID
            refresh: Force refresh of GPU info

        Returns:
            GPUInfo for the specified GPU or None if not found
        """
        if refresh or self._cache_stale():
            self._refresh_gpu_info()

        with self._lock:
            return self._gpu_cache.get(gpu_id)

    def get_all_gpu_info(self, refresh: bool = False) -> List[GPUInfo]:
        """Get information for all managed GPUs.

        Args:
            refresh: Force refresh of GPU info

        Returns:
            List of GPUInfo objects
        """
        if refresh or self._cache_stale():
            self._refresh_gpu_info()

        with self._lock:
            return [self._gpu_cache[gid] for gid in self.gpu_ids if gid in self._gpu_cache]

    def get_optimal_gpu(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.VRAM_AWARE) -> int:
        """Get the optimal GPU for processing based on strategy.

        Args:
            strategy: Load balancing strategy to use

        Returns:
            GPU device ID of the optimal GPU
        """
        gpus = self.get_all_gpu_info(refresh=True)

        if not gpus:
            return 0  # Fallback to GPU 0

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            # Simple round-robin (use first available)
            return gpus[0].id

        elif strategy == LoadBalanceStrategy.LEAST_LOADED:
            # Sort by utilization (lowest first)
            gpus.sort(key=lambda g: g.utilization_pct)
            return gpus[0].id

        elif strategy == LoadBalanceStrategy.VRAM_AWARE:
            # Sort by free VRAM (highest first)
            gpus.sort(key=lambda g: g.free_vram_mb, reverse=True)
            return gpus[0].id

        elif strategy == LoadBalanceStrategy.WEIGHTED:
            # Sort by effective capacity score (highest first)
            gpus.sort(key=lambda g: g.effective_capacity, reverse=True)
            return gpus[0].id

        # Default fallback
        return gpus[0].id

    def get_healthy_gpus(self) -> List[GPUInfo]:
        """Get list of healthy GPUs ready for processing.

        Returns:
            List of GPUInfo for healthy GPUs
        """
        return [gpu for gpu in self.get_all_gpu_info(refresh=True) if gpu.is_healthy]

    def wait_for_vram(
        self,
        required_mb: int,
        gpu_id: Optional[int] = None,
        timeout: float = 60.0,
    ) -> bool:
        """Wait for sufficient VRAM to become available.

        Args:
            required_mb: Required free VRAM in MB
            gpu_id: Specific GPU to wait for (None = any GPU)
            timeout: Maximum time to wait in seconds

        Returns:
            True if sufficient VRAM available, False on timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            gpus = self.get_all_gpu_info(refresh=True)

            if gpu_id is not None:
                # Check specific GPU
                gpu = next((g for g in gpus if g.id == gpu_id), None)
                if gpu and gpu.free_vram_mb >= required_mb:
                    return True
            else:
                # Check any GPU
                if any(g.free_vram_mb >= required_mb for g in gpus):
                    return True

            time.sleep(1.0)

        return False

    def start_monitoring(self) -> None:
        """Start background GPU monitoring thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="GPUMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started GPU monitoring")

    def stop_monitoring(self) -> None:
        """Stop background GPU monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Stopped GPU monitoring")

    def _is_nvidia_smi_available(self) -> bool:
        """Check if nvidia-smi is available."""
        return shutil.which("nvidia-smi") is not None

    def _refresh_gpu_info(self) -> None:
        """Refresh cached GPU information."""
        gpus = self.detect_gpus()
        with self._lock:
            self._gpu_cache = {g.id: g for g in gpus}
            self._last_refresh = time.time()

    def _cache_stale(self) -> bool:
        """Check if GPU cache needs refresh."""
        return time.time() - self._last_refresh > self._refresh_interval

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._refresh_gpu_info()
                time.sleep(self._refresh_interval)
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
                time.sleep(1.0)


class WorkStealingQueue:
    """Thread-safe work queue with work-stealing support.

    Allows idle workers to steal work from busy workers for
    better load balancing.
    """

    def __init__(self, num_workers: int):
        """Initialize work-stealing queue.

        Args:
            num_workers: Number of worker threads
        """
        self._queues: Dict[int, queue.Queue] = {
            i: queue.Queue() for i in range(num_workers)
        }
        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def add_work(self, item: WorkItem, worker_id: int) -> None:
        """Add work item to specific worker's queue.

        Args:
            item: Work item to add
            worker_id: Target worker ID
        """
        with self._lock:
            self._queues[worker_id].put(item)
            self._total += 1

    def get_work(self, worker_id: int, timeout: float = 0.1) -> Optional[WorkItem]:
        """Get work for a worker, with stealing from others if empty.

        Args:
            worker_id: Worker requesting work
            timeout: Timeout for blocking get

        Returns:
            WorkItem or None if no work available
        """
        # Try own queue first
        try:
            return self._queues[worker_id].get(timeout=timeout)
        except queue.Empty:
            pass

        # Try stealing from other queues
        with self._lock:
            for other_id, q in self._queues.items():
                if other_id != worker_id and q.qsize() > 1:
                    try:
                        item = q.get_nowait()
                        logger.debug(f"Worker {worker_id} stole work from {other_id}")
                        return item
                    except queue.Empty:
                        continue

        return None

    def mark_complete(self) -> None:
        """Mark one work item as complete."""
        with self._lock:
            self._completed += 1

    @property
    def progress(self) -> float:
        """Get progress as fraction (0-1)."""
        with self._lock:
            if self._total == 0:
                return 0.0
            return self._completed / self._total

    @property
    def is_complete(self) -> bool:
        """Check if all work is complete."""
        with self._lock:
            return self._completed >= self._total and all(
                q.empty() for q in self._queues.values()
            )


class MultiGPUDistributor:
    """Distributes frame processing across multiple GPUs.

    Features:
    - Multiple load balancing strategies
    - Work-stealing for dynamic load balancing
    - Graceful GPU failure handling
    - Retry on alternate GPUs
    - Progress tracking and statistics
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUManager] = None,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.VRAM_AWARE,
        workers_per_gpu: int = 2,
        max_retries: int = 2,
        enable_work_stealing: bool = True,
    ):
        """Initialize multi-GPU distributor.

        Args:
            gpu_manager: GPU manager instance (creates new if None)
            strategy: Load balancing strategy
            workers_per_gpu: Number of worker threads per GPU
            max_retries: Maximum retry attempts per frame
            enable_work_stealing: Enable work stealing between workers
        """
        self.gpu_manager = gpu_manager or GPUManager()
        self.strategy = strategy
        self.workers_per_gpu = workers_per_gpu
        self.max_retries = max_retries
        self.enable_work_stealing = enable_work_stealing

        self._result: Optional[DistributionResult] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def distribute_frames(
        self,
        frames: List[Path],
        process_fn: Callable[[Path, Path, int], Tuple[Path, bool, Optional[str]]],
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> DistributionResult:
        """Distribute frame processing across GPUs.

        Args:
            frames: List of input frame paths
            process_fn: Processing function with signature:
                       (input_path, output_dir, gpu_id) -> (output_path, success, error_msg)
            output_dir: Output directory for processed frames
            progress_callback: Optional callback(progress: float, message: str)

        Returns:
            DistributionResult with processing statistics
        """
        if not frames:
            return DistributionResult()

        start_time = time.time()
        self._stop_event.clear()

        # Get available GPUs
        gpus = self.gpu_manager.get_healthy_gpus()
        if not gpus:
            logger.error("No healthy GPUs available")
            return DistributionResult(
                errors={str(f): "No GPUs available" for f in frames}
            )

        num_gpus = len(gpus)
        gpu_ids = [g.id for g in gpus]
        total_workers = num_gpus * self.workers_per_gpu

        logger.info(
            f"Distributing {len(frames)} frames across {num_gpus} GPUs "
            f"({total_workers} workers, strategy={self.strategy.value})"
        )

        # Initialize result tracking
        result = DistributionResult()
        result.frames_per_gpu = {gid: [] for gid in gpu_ids}

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Distribute frames based on strategy
        frame_assignments = self._assign_frames(frames, gpus)

        # Create work queues
        if self.enable_work_stealing:
            work_queue = WorkStealingQueue(total_workers)
        else:
            work_queue = None

        # Per-worker queues for simple distribution
        worker_queues: Dict[int, queue.Queue] = {}
        for worker_id in range(total_workers):
            worker_queues[worker_id] = queue.Queue()

        # Distribute work items to workers
        worker_id = 0
        for gpu_id, assigned_frames in frame_assignments.items():
            for frame_path in assigned_frames:
                work_item = WorkItem(
                    frame_path=frame_path,
                    output_dir=output_dir,
                    assigned_gpu=gpu_id,
                )

                if work_queue:
                    work_queue.add_work(work_item, worker_id % total_workers)
                else:
                    worker_queues[worker_id % total_workers].put(work_item)

                worker_id += 1

        # Track completion
        completed = 0
        errors: Dict[str, str] = {}
        retried: List[Path] = []
        lock = threading.Lock()

        def worker_fn(wid: int, gpu_id: int) -> None:
            """Worker function for processing frames."""
            nonlocal completed

            while not self._stop_event.is_set():
                # Get work item
                if work_queue:
                    work_item = work_queue.get_work(wid, timeout=0.5)
                else:
                    try:
                        work_item = worker_queues[wid].get(timeout=0.5)
                    except queue.Empty:
                        work_item = None

                if work_item is None:
                    # Check if all work is done
                    if work_queue and work_queue.is_complete:
                        break
                    elif not work_queue:
                        if worker_queues[wid].empty():
                            break
                    continue

                # Process frame
                current_gpu = work_item.assigned_gpu if work_item.assigned_gpu is not None else gpu_id

                try:
                    output_path, success, error_msg = process_fn(
                        work_item.frame_path,
                        work_item.output_dir,
                        current_gpu,
                    )

                    with lock:
                        if success:
                            result.frames_per_gpu[current_gpu].append(output_path)
                            completed += 1

                            if work_queue:
                                work_queue.mark_complete()

                            if progress_callback:
                                progress = completed / len(frames)
                                progress_callback(
                                    progress,
                                    f"Processed {completed}/{len(frames)} frames"
                                )
                        else:
                            # Handle failure
                            work_item.attempts += 1
                            work_item.failed_gpus.append(current_gpu)

                            if work_item.can_retry and len(work_item.failed_gpus) < num_gpus:
                                # Retry on different GPU
                                available_gpus = [
                                    g for g in gpu_ids if g not in work_item.failed_gpus
                                ]
                                if available_gpus:
                                    work_item.assigned_gpu = available_gpus[0]
                                    retried.append(work_item.frame_path)

                                    # Re-queue work item
                                    if work_queue:
                                        next_worker = wid
                                        work_queue.add_work(work_item, next_worker)
                                    else:
                                        worker_queues[wid].put(work_item)

                                    logger.warning(
                                        f"Retrying {work_item.frame_path.name} on GPU {work_item.assigned_gpu}"
                                    )
                                    continue

                            # Max retries exceeded
                            errors[str(work_item.frame_path)] = error_msg or "Unknown error"
                            completed += 1

                            if work_queue:
                                work_queue.mark_complete()

                            if progress_callback:
                                progress = completed / len(frames)
                                progress_callback(progress, f"Error: {work_item.frame_path.name}")

                except Exception as e:
                    with lock:
                        errors[str(work_item.frame_path)] = str(e)
                        completed += 1

                        if work_queue:
                            work_queue.mark_complete()

                        logger.error(f"Worker error processing {work_item.frame_path}: {e}")

        # Start workers
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = []

            for i in range(total_workers):
                gpu_id = gpu_ids[i % num_gpus]
                future = executor.submit(worker_fn, i, gpu_id)
                futures.append(future)

            # Wait for completion
            for future in futures:
                try:
                    future.result(timeout=3600)  # 1 hour timeout
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        # Calculate result statistics
        end_time = time.time()
        result.total_time = end_time - start_time
        result.errors = errors
        result.retried_frames = retried

        # Estimate speedup factor
        if num_gpus > 1 and result.total_frames > 0:
            # Theoretical linear speedup, adjusted by actual distribution
            actual_distribution = [len(f) for f in result.frames_per_gpu.values()]
            if actual_distribution:
                max_per_gpu = max(actual_distribution)
                if max_per_gpu > 0:
                    ideal_per_gpu = result.total_frames / num_gpus
                    distribution_efficiency = ideal_per_gpu / max_per_gpu
                    result.speedup_factor = num_gpus * distribution_efficiency

        # Get final GPU utilization
        for gpu in self.gpu_manager.get_all_gpu_info():
            if gpu.id in gpu_ids:
                result.gpu_utilization[gpu.id] = gpu.utilization_pct

        logger.info(result.summary())
        self._result = result

        return result

    def stop(self) -> None:
        """Stop ongoing distribution processing."""
        self._stop_event.set()

    def get_result(self) -> Optional[DistributionResult]:
        """Get the result of the last distribution operation."""
        return self._result

    def _assign_frames(
        self,
        frames: List[Path],
        gpus: List[GPUInfo],
    ) -> Dict[int, List[Path]]:
        """Assign frames to GPUs based on strategy.

        Args:
            frames: List of frame paths
            gpus: List of available GPUs

        Returns:
            Dictionary mapping GPU ID to assigned frames
        """
        assignments: Dict[int, List[Path]] = {g.id: [] for g in gpus}

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            # Simple round-robin
            for i, frame in enumerate(frames):
                gpu_id = gpus[i % len(gpus)].id
                assignments[gpu_id].append(frame)

        elif self.strategy == LoadBalanceStrategy.LEAST_LOADED:
            # Sort by utilization and distribute evenly
            sorted_gpus = sorted(gpus, key=lambda g: g.utilization_pct)
            for i, frame in enumerate(frames):
                gpu_id = sorted_gpus[i % len(sorted_gpus)].id
                assignments[gpu_id].append(frame)

        elif self.strategy == LoadBalanceStrategy.VRAM_AWARE:
            # Weight assignment by free VRAM
            total_free = sum(g.free_vram_mb for g in gpus)
            if total_free == 0:
                # Fallback to round-robin
                for i, frame in enumerate(frames):
                    gpu_id = gpus[i % len(gpus)].id
                    assignments[gpu_id].append(frame)
            else:
                # Calculate target frames per GPU based on VRAM ratio
                targets = {}
                for gpu in gpus:
                    ratio = gpu.free_vram_mb / total_free
                    targets[gpu.id] = int(len(frames) * ratio)

                # Distribute frames
                frame_idx = 0
                for gpu_id, target in targets.items():
                    for _ in range(target):
                        if frame_idx < len(frames):
                            assignments[gpu_id].append(frames[frame_idx])
                            frame_idx += 1

                # Handle any remaining frames
                while frame_idx < len(frames):
                    gpu_id = gpus[frame_idx % len(gpus)].id
                    assignments[gpu_id].append(frames[frame_idx])
                    frame_idx += 1

        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            # Weight by effective capacity
            capacities = {g.id: g.effective_capacity for g in gpus}
            total_capacity = sum(capacities.values())

            if total_capacity == 0:
                # Fallback to round-robin
                for i, frame in enumerate(frames):
                    gpu_id = gpus[i % len(gpus)].id
                    assignments[gpu_id].append(frame)
            else:
                # Distribute based on capacity weights
                frame_idx = 0
                for gpu_id, capacity in capacities.items():
                    target = int(len(frames) * (capacity / total_capacity))
                    for _ in range(target):
                        if frame_idx < len(frames):
                            assignments[gpu_id].append(frames[frame_idx])
                            frame_idx += 1

                # Handle remaining
                while frame_idx < len(frames):
                    # Assign to GPU with highest capacity
                    best_gpu = max(capacities.keys(), key=lambda x: capacities[x])
                    assignments[best_gpu].append(frames[frame_idx])
                    frame_idx += 1

        # Log distribution
        for gpu_id, assigned in assignments.items():
            logger.debug(f"GPU {gpu_id}: {len(assigned)} frames assigned")

        return assignments


def detect_gpus() -> List[GPUInfo]:
    """Convenience function to detect all available GPUs.

    Returns:
        List of GPUInfo objects
    """
    manager = GPUManager()
    return manager.detect_gpus()


def get_optimal_gpu(strategy: LoadBalanceStrategy = LoadBalanceStrategy.VRAM_AWARE) -> int:
    """Convenience function to get optimal GPU for processing.

    Args:
        strategy: Load balancing strategy

    Returns:
        Optimal GPU device ID
    """
    manager = GPUManager()
    return manager.get_optimal_gpu(strategy)


def distribute_frames(
    frames: List[Path],
    process_fn: Callable[[Path, Path, int], Tuple[Path, bool, Optional[str]]],
    output_dir: Path,
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.VRAM_AWARE,
    workers_per_gpu: int = 2,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> DistributionResult:
    """Convenience function to distribute frames across GPUs.

    Args:
        frames: List of input frame paths
        process_fn: Processing function
        output_dir: Output directory
        strategy: Load balancing strategy
        workers_per_gpu: Workers per GPU
        progress_callback: Progress callback

    Returns:
        DistributionResult with statistics
    """
    distributor = MultiGPUDistributor(
        strategy=strategy,
        workers_per_gpu=workers_per_gpu,
    )
    return distributor.distribute_frames(
        frames=frames,
        process_fn=process_fn,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )


# Integration helpers for Config
def add_multi_gpu_config_fields() -> Dict[str, Any]:
    """Get default multi-GPU configuration fields.

    Returns:
        Dictionary of field names and default values for Config integration
    """
    return {
        "enable_multi_gpu": False,
        "gpu_id": None,  # Optional[int] - Single GPU selection
        "gpu_ids": None,  # Optional[List[int]]
        "gpu_load_balance_strategy": "vram_aware",
        "workers_per_gpu": 2,
        "enable_work_stealing": True,
    }


class GPUSelector:
    """GPU selection utility for choosing optimal or specific GPUs.

    Provides methods for:
    - Selecting a specific GPU by index
    - Auto-selecting the best GPU based on available memory
    - Validating GPU availability
    """

    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """Initialize GPU selector.

        Args:
            gpu_manager: GPU manager instance (creates new if None)
        """
        self.gpu_manager = gpu_manager or GPUManager()

    def select_by_index(self, gpu_id: int) -> Optional[GPUInfo]:
        """Select a specific GPU by index.

        Args:
            gpu_id: GPU device index to select

        Returns:
            GPUInfo for the selected GPU or None if not available

        Raises:
            ValueError: If GPU index is invalid or GPU not found
        """
        if gpu_id < 0:
            raise ValueError(f"Invalid GPU index: {gpu_id}. Must be non-negative.")

        gpu = self.gpu_manager.get_gpu_info(gpu_id, refresh=True)
        if gpu is None:
            available = self.gpu_manager.gpu_ids
            raise ValueError(
                f"GPU {gpu_id} not found. Available GPUs: {available}"
            )
        return gpu

    def select_best(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.VRAM_AWARE) -> int:
        """Auto-select the best GPU based on available memory.

        Args:
            strategy: Strategy for selecting the best GPU

        Returns:
            GPU device ID of the best available GPU
        """
        return self.gpu_manager.get_optimal_gpu(strategy)

    def validate_gpu(self, gpu_id: int) -> bool:
        """Check if a GPU is valid and available.

        Args:
            gpu_id: GPU device index to validate

        Returns:
            True if GPU is available and healthy
        """
        gpu = self.gpu_manager.get_gpu_info(gpu_id, refresh=True)
        return gpu is not None and gpu.is_healthy

    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of all available GPUs.

        Returns:
            List of GPUInfo for all available GPUs
        """
        return self.gpu_manager.get_all_gpu_info(refresh=True)

    def get_gpu_for_task(
        self,
        gpu_id: Optional[int] = None,
        multi_gpu: bool = False,
    ) -> Tuple[int, bool]:
        """Get the appropriate GPU for a processing task.

        Args:
            gpu_id: Specific GPU index (--gpu N), or None for auto-select
            multi_gpu: Whether multi-GPU mode is enabled (--multi-gpu)

        Returns:
            Tuple of (primary_gpu_id, use_multi_gpu)
        """
        if gpu_id is not None:
            # User specified a specific GPU
            if not self.validate_gpu(gpu_id):
                logger.warning(f"Specified GPU {gpu_id} not available, falling back to auto-select")
                return self.select_best(), multi_gpu
            return gpu_id, multi_gpu

        if multi_gpu:
            # Multi-GPU mode, return optimal GPU as primary
            return self.select_best(), True

        # Auto-select best single GPU
        return self.select_best(), False


class MultiGPUManager:
    """Enhanced manager for multi-GPU detection and distribution.

    Extends GPUManager with additional features:
    - Compute capability detection
    - Formatted GPU listing
    - Dynamic rebalancing support
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        refresh_interval: float = 5.0,
    ):
        """Initialize multi-GPU manager.

        Args:
            gpu_ids: Specific GPU IDs to manage (None = auto-detect all)
            refresh_interval: Interval between automatic GPU info refreshes
        """
        self._base_manager = GPUManager(gpu_ids=gpu_ids, refresh_interval=refresh_interval)
        self._processing_speeds: Dict[int, float] = {}  # GPU ID -> frames/sec
        self._lock = threading.Lock()

    @property
    def gpu_ids(self) -> List[int]:
        """Get list of managed GPU IDs."""
        return self._base_manager.gpu_ids

    @property
    def gpu_count(self) -> int:
        """Get number of managed GPUs."""
        return self._base_manager.gpu_count

    @property
    def is_multi_gpu(self) -> bool:
        """Check if multiple GPUs are available."""
        return self._base_manager.is_multi_gpu

    def detect_gpus_with_compute(self) -> List[GPUInfo]:
        """Detect all available GPUs including compute capability.

        Returns:
            List of GPUInfo objects with compute capability populated
        """
        if not self._base_manager._is_nvidia_smi_available():
            logger.warning("nvidia-smi not available, no GPUs detected")
            return []

        try:
            # Query detailed GPU information including compute capability
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,"
                "temperature.gpu,pcie.link.gen.current,compute_cap",
                "--format=csv,noheader,nounits",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]

                try:
                    compute_cap = parts[7] if len(parts) > 7 and parts[7] != "[N/A]" else None

                    gpu = GPUInfo(
                        id=int(parts[0]),
                        name=parts[1],
                        total_vram_mb=int(parts[2]),
                        free_vram_mb=int(parts[3]),
                        utilization_pct=float(parts[4]) if parts[4] != "[N/A]" else 0.0,
                        temperature_c=float(parts[5]) if parts[5] != "[N/A]" else None,
                        compute_capability=compute_cap,
                    )

                    if self._base_manager._gpu_ids is None or gpu.id in self._base_manager._gpu_ids:
                        gpus.append(gpu)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse GPU info: {e}")
                    continue

            return gpus

        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return []

    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """Get information for a specific GPU.

        Args:
            gpu_id: GPU device ID

        Returns:
            GPUInfo for the specified GPU or None if not found
        """
        return self._base_manager.get_gpu_info(gpu_id, refresh=True)

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information for all managed GPUs.

        Returns:
            List of GPUInfo objects
        """
        return self._base_manager.get_all_gpu_info(refresh=True)

    def get_healthy_gpus(self) -> List[GPUInfo]:
        """Get list of healthy GPUs ready for processing.

        Returns:
            List of GPUInfo for healthy GPUs
        """
        return self._base_manager.get_healthy_gpus()

    def update_processing_speed(self, gpu_id: int, frames_per_second: float) -> None:
        """Update the processing speed for a GPU.

        Used for dynamic rebalancing based on actual performance.

        Args:
            gpu_id: GPU device ID
            frames_per_second: Measured processing speed
        """
        with self._lock:
            self._processing_speeds[gpu_id] = frames_per_second

    def get_dynamic_distribution(self, total_frames: int) -> Dict[int, int]:
        """Get frame distribution based on measured processing speeds.

        Falls back to VRAM-aware distribution if no speed data available.

        Args:
            total_frames: Total number of frames to distribute

        Returns:
            Dictionary mapping GPU ID to number of frames
        """
        gpus = self.get_healthy_gpus()
        if not gpus:
            return {}

        with self._lock:
            speeds = self._processing_speeds.copy()

        # If we have speed data for all GPUs, use it
        gpu_ids = [g.id for g in gpus]
        if speeds and all(gid in speeds for gid in gpu_ids):
            total_speed = sum(speeds.values())
            if total_speed > 0:
                distribution = {}
                assigned = 0
                for gpu in gpus[:-1]:  # All but last
                    ratio = speeds[gpu.id] / total_speed
                    count = int(total_frames * ratio)
                    distribution[gpu.id] = count
                    assigned += count
                # Last GPU gets remainder
                distribution[gpus[-1].id] = total_frames - assigned
                return distribution

        # Fall back to VRAM-aware distribution
        total_free = sum(g.free_vram_mb for g in gpus)
        if total_free == 0:
            # Even distribution as last resort
            per_gpu = total_frames // len(gpus)
            distribution = {g.id: per_gpu for g in gpus[:-1]}
            distribution[gpus[-1].id] = total_frames - (per_gpu * (len(gpus) - 1))
            return distribution

        distribution = {}
        assigned = 0
        for gpu in gpus[:-1]:
            ratio = gpu.free_vram_mb / total_free
            count = int(total_frames * ratio)
            distribution[gpu.id] = count
            assigned += count
        distribution[gpus[-1].id] = total_frames - assigned

        return distribution

    def format_gpu_table(self) -> str:
        """Generate a formatted table of available GPUs.

        Returns:
            Formatted table string for display
        """
        gpus = self.detect_gpus_with_compute()

        if not gpus:
            return "No GPUs detected. Ensure NVIDIA drivers are installed."

        # Table header
        lines = [
            "Available GPUs:",
            "+------+---------------------------+---------+------------+--------+",
            "| ID   | Name                      | Memory  | Compute    | Status |",
            "+------+---------------------------+---------+------------+--------+",
        ]

        for gpu in gpus:
            # Format memory in GB
            memory_gb = gpu.total_vram_mb / 1024
            memory_str = f"{memory_gb:.0f} GB"

            # Format compute capability
            compute_str = gpu.compute_capability if gpu.compute_capability else "N/A"

            # Determine status
            if not gpu.is_healthy:
                status = "Hot"
            elif gpu.utilization_pct > 90:
                status = "Busy"
            else:
                status = "Ready"

            line = f"| {gpu.id:<4} | {gpu.name:<25} | {memory_str:<7} | {compute_str:<10} | {status:<6} |"
            lines.append(line)

        lines.append("+------+---------------------------+---------+------------+--------+")

        return "\n".join(lines)


def list_gpus() -> str:
    """Convenience function to list all available GPUs.

    Returns:
        Formatted table string of GPUs
    """
    manager = MultiGPUManager()
    return manager.format_gpu_table()


def select_gpu(
    gpu_id: Optional[int] = None,
    multi_gpu: bool = False,
) -> Tuple[int, bool]:
    """Convenience function to select GPU for processing.

    Args:
        gpu_id: Specific GPU index or None for auto-select
        multi_gpu: Enable multi-GPU mode

    Returns:
        Tuple of (gpu_id, use_multi_gpu)
    """
    selector = GPUSelector()
    return selector.get_gpu_for_task(gpu_id, multi_gpu)
