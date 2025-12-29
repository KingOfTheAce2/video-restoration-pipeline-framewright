"""Streaming output support for processing very long videos in chunks.

Enables:
- Output video chunks as they're processed
- Preview early results while still processing
- Recover partial output if something fails mid-way
- Handle videos larger than available disk space
- Multi-GPU distribution for parallel processing
- Memory-optimized batch frame processing
- Streaming pipeline with configurable buffers
"""
import gc
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

from framewright.utils.gpu import (
    GPUInfo,
    get_all_gpu_info,
    get_gpu_memory_info,
    is_nvidia_gpu_available,
    VRAMMonitor,
)

logger = logging.getLogger(__name__)

# Type variable for generic frame data
FrameT = TypeVar("FrameT")


@dataclass
class ChunkInfo:
    """Information about a processed video chunk."""
    chunk_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    output_path: Path
    duration_seconds: float
    is_final: bool = False


class DistributionStrategy(Enum):
    """Strategy for distributing frames across GPUs."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    MEMORY_BASED = "memory_based"


@dataclass
class StreamingConfig:
    """Configuration for streaming video processing.

    This enhanced configuration supports:
    - Chunk-based processing for memory efficiency
    - Multi-GPU distribution
    - Batch processing with auto-detection
    - Memory limits and buffer management
    """
    # Chunk settings
    chunk_duration_seconds: float = 300.0  # 5 minutes per chunk
    min_chunk_frames: int = 100  # Minimum frames per chunk
    max_chunk_frames: int = 10000  # Maximum frames per chunk
    chunk_size: int = 100  # Frames per processing chunk

    # Buffer settings for streaming pipeline
    max_buffer_size: int = 50  # Maximum frames in memory buffer

    # Output settings
    output_format: str = "mkv"
    crf: int = 18
    preset: str = "medium"

    # Processing settings
    cleanup_chunks_on_merge: bool = True
    keep_intermediate_chunks: bool = False

    # Callback settings
    emit_on_chunk_complete: bool = True

    # Multi-GPU settings
    enable_multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None
    distribution_strategy: DistributionStrategy = DistributionStrategy.LOAD_BALANCED

    # Batch processing settings
    batch_size: int = 1  # Frames per batch (0 = auto-detect based on VRAM)
    max_batch_size: int = 8  # Maximum batch size when auto-detecting

    # Memory settings
    memory_limit_mb: Optional[int] = None  # Maximum VRAM usage in MB
    memory_safety_factor: float = 0.8  # Fraction of available VRAM to use
    cleanup_between_chunks: bool = True  # Force garbage collection between chunks

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative (0 for auto)")
        if self.memory_safety_factor <= 0 or self.memory_safety_factor > 1:
            raise ValueError("memory_safety_factor must be between 0 and 1")


@dataclass
class StreamingState:
    """State tracker for streaming processing."""
    total_frames: int = 0
    processed_frames: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    chunks_completed: List[ChunkInfo] = field(default_factory=list)
    is_complete: bool = False
    error: Optional[str] = None

    @property
    def progress(self) -> float:
        """Get overall progress as 0.0-1.0."""
        if self.total_frames == 0:
            return 0.0
        return self.processed_frames / self.total_frames

    @property
    def chunks_ready(self) -> int:
        """Number of chunks ready for viewing."""
        return len(self.chunks_completed)


@dataclass
class BatchResult:
    """Result of processing a batch of frames."""
    batch_id: int
    frame_indices: List[int]
    output_paths: List[Path]
    processing_time_seconds: float
    gpu_id: Optional[int] = None
    memory_used_mb: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state."""
    timestamp: float
    gpu_id: int
    total_mb: int
    used_mb: int
    free_mb: int

    @property
    def usage_percent(self) -> float:
        """Memory usage as percentage."""
        if self.total_mb == 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100


# =============================================================================
# Memory Management
# =============================================================================


class MemoryManager:
    """Manages VRAM usage and provides adaptive tile/batch sizing.

    This class tracks GPU memory usage and automatically adjusts processing
    parameters to maximize throughput while staying within memory limits.

    Example:
        >>> manager = MemoryManager(gpu_id=0, memory_limit_mb=4000)
        >>> tile_size = manager.get_optimal_tile_size(frame_size=(1920, 1080))
        >>> batch_size = manager.get_optimal_batch_size(frame_size=(1920, 1080))
    """

    # Empirically determined memory coefficients per model type
    MODEL_MEMORY_COEFFICIENTS = {
        "realesrgan-x4plus": 450,
        "realesrgan-x4plus-anime": 400,
        "realesrgan-x2plus": 250,
        "realesr-animevideov3": 350,
        "default": 450,
    }

    # Base memory overhead for model loading (MB)
    MODEL_BASE_OVERHEAD_MB = 500

    def __init__(
        self,
        gpu_id: int = 0,
        memory_limit_mb: Optional[int] = None,
        safety_factor: float = 0.8,
        model_name: str = "default",
    ):
        """Initialize memory manager.

        Args:
            gpu_id: GPU device ID to manage
            memory_limit_mb: Maximum VRAM to use (None = auto-detect)
            safety_factor: Fraction of available VRAM to use (0.0-1.0)
            model_name: Model name for memory coefficient lookup
        """
        self.gpu_id = gpu_id
        self.safety_factor = safety_factor
        self.model_name = model_name
        self._snapshots: List[MemorySnapshot] = []
        self._peak_usage_mb = 0
        self._lock = threading.Lock()

        # Set memory limit
        if memory_limit_mb is not None:
            self._memory_limit_mb = memory_limit_mb
        else:
            self._memory_limit_mb = self._detect_available_memory()

        logger.info(
            f"MemoryManager initialized: GPU {gpu_id}, "
            f"limit={self._memory_limit_mb}MB, safety={safety_factor}"
        )

    def _detect_available_memory(self) -> int:
        """Detect available GPU memory."""
        info = get_gpu_memory_info(self.gpu_id)
        if info:
            return int(info["total_mb"] * self.safety_factor)
        # Conservative default if detection fails
        return 4096

    @property
    def memory_limit_mb(self) -> int:
        """Current memory limit in MB."""
        return self._memory_limit_mb

    @property
    def usable_memory_mb(self) -> int:
        """Usable memory after applying safety factor."""
        return int(self._memory_limit_mb * self.safety_factor)

    def get_current_usage(self) -> Optional[MemorySnapshot]:
        """Get current GPU memory usage.

        Returns:
            MemorySnapshot or None if unavailable
        """
        info = get_gpu_memory_info(self.gpu_id)
        if not info:
            return None

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            gpu_id=self.gpu_id,
            total_mb=info["total_mb"],
            used_mb=info["used_mb"],
            free_mb=info["free_mb"],
        )

        with self._lock:
            self._snapshots.append(snapshot)
            if snapshot.used_mb > self._peak_usage_mb:
                self._peak_usage_mb = snapshot.used_mb

        return snapshot

    def get_free_memory_mb(self) -> int:
        """Get current free memory in MB."""
        snapshot = self.get_current_usage()
        if snapshot:
            return snapshot.free_mb
        return self.usable_memory_mb

    def get_memory_coefficient(self) -> int:
        """Get memory coefficient for current model."""
        return self.MODEL_MEMORY_COEFFICIENTS.get(
            self.model_name,
            self.MODEL_MEMORY_COEFFICIENTS["default"]
        )

    def get_optimal_tile_size(
        self,
        frame_size: Tuple[int, int],
        scale_factor: int = 4,
        min_tile_size: int = 128,
    ) -> int:
        """Calculate optimal tile size based on available memory.

        Args:
            frame_size: (width, height) of input frame
            scale_factor: Upscaling factor
            min_tile_size: Minimum allowed tile size

        Returns:
            Optimal tile size, or 0 if no tiling needed
        """
        width, height = frame_size
        coeff = self.get_memory_coefficient()
        usable_mb = self.get_free_memory_mb() - self.MODEL_BASE_OVERHEAD_MB

        # Calculate if full frame fits
        output_megapixels = (width * scale_factor * height * scale_factor) / 1_000_000
        estimated_mb = output_megapixels * coeff

        if estimated_mb <= usable_mb:
            logger.debug(f"No tiling needed: {estimated_mb:.0f}MB < {usable_mb}MB")
            return 0

        # Calculate maximum tile size that fits
        max_output_tile_pixels = (usable_mb / coeff) * 1_000_000
        max_output_tile_size = int(max_output_tile_pixels ** 0.5)
        max_input_tile_size = max_output_tile_size // scale_factor

        # Align to 32 for GPU efficiency
        tile_size = (max_input_tile_size // 32) * 32
        tile_size = max(min_tile_size, tile_size)
        tile_size = min(tile_size, min(width, height))

        logger.info(
            f"Optimal tile size: {tile_size} "
            f"(frame: {width}x{height}, usable: {usable_mb}MB)"
        )

        return tile_size

    def get_optimal_batch_size(
        self,
        frame_size: Tuple[int, int],
        scale_factor: int = 4,
        max_batch_size: int = 8,
    ) -> int:
        """Calculate optimal batch size based on available memory.

        Args:
            frame_size: (width, height) of input frame
            scale_factor: Upscaling factor
            max_batch_size: Maximum allowed batch size

        Returns:
            Optimal batch size (at least 1)
        """
        width, height = frame_size
        coeff = self.get_memory_coefficient()
        usable_mb = self.get_free_memory_mb() - self.MODEL_BASE_OVERHEAD_MB

        # Memory per frame at output resolution
        output_megapixels = (width * scale_factor * height * scale_factor) / 1_000_000
        mb_per_frame = output_megapixels * coeff

        if mb_per_frame <= 0:
            return 1

        # Calculate how many frames fit
        optimal_batch = int(usable_mb / mb_per_frame)
        optimal_batch = max(1, min(optimal_batch, max_batch_size))

        logger.debug(
            f"Optimal batch size: {optimal_batch} "
            f"({mb_per_frame:.0f}MB/frame, {usable_mb}MB available)"
        )

        return optimal_batch

    def adjust_for_memory_pressure(
        self,
        current_tile_size: int,
        current_batch_size: int,
    ) -> Tuple[int, int]:
        """Adjust tile and batch sizes when memory is under pressure.

        Args:
            current_tile_size: Current tile size
            current_batch_size: Current batch size

        Returns:
            Tuple of (new_tile_size, new_batch_size)
        """
        snapshot = self.get_current_usage()

        if snapshot and snapshot.usage_percent > 90:
            # Critical memory pressure
            new_batch = max(1, current_batch_size // 2)
            new_tile = max(128, (current_tile_size * 3) // 4)
            new_tile = (new_tile // 32) * 32

            logger.warning(
                f"Memory pressure ({snapshot.usage_percent:.1f}%): "
                f"batch {current_batch_size} -> {new_batch}, "
                f"tile {current_tile_size} -> {new_tile}"
            )

            return new_tile, new_batch

        return current_tile_size, current_batch_size

    def cleanup(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        logger.debug("Memory cleanup completed")

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary with min, max, avg, peak usage
        """
        with self._lock:
            if not self._snapshots:
                return {
                    "samples": 0,
                    "peak_mb": 0,
                    "min_mb": 0,
                    "max_mb": 0,
                    "avg_mb": 0,
                }

            used_values = [s.used_mb for s in self._snapshots]

            return {
                "samples": len(self._snapshots),
                "peak_mb": self._peak_usage_mb,
                "min_mb": min(used_values),
                "max_mb": max(used_values),
                "avg_mb": sum(used_values) / len(used_values),
            }


# =============================================================================
# GPU Distribution
# =============================================================================


def get_available_gpus() -> List[GPUInfo]:
    """Get list of all available GPUs.

    Returns:
        List of GPUInfo objects for each available GPU
    """
    return get_all_gpu_info()


class GPUDistributor:
    """Distributes frame processing work across multiple GPUs.

    Supports round-robin and load-balanced distribution strategies
    to maximize throughput on multi-GPU systems.

    Example:
        >>> distributor = GPUDistributor(gpu_ids=[0, 1], strategy="load_balanced")
        >>> assignments = distributor.distribute_frames(frames, len(frames))
        >>> for gpu_id, frame_list in assignments.items():
        ...     process_on_gpu(gpu_id, frame_list)
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        strategy: Union[str, DistributionStrategy] = DistributionStrategy.LOAD_BALANCED,
    ):
        """Initialize GPU distributor.

        Args:
            gpu_ids: List of GPU IDs to use (None = auto-detect)
            strategy: Distribution strategy ("round_robin", "load_balanced", "memory_based")
        """
        if isinstance(strategy, str):
            strategy = DistributionStrategy(strategy)
        self.strategy = strategy
        self._round_robin_index = 0
        self._lock = threading.Lock()

        # Auto-detect GPUs if not specified
        if gpu_ids is None:
            available = get_available_gpus()
            self.gpu_ids = [g.index for g in available]
        else:
            self.gpu_ids = gpu_ids

        if not self.gpu_ids:
            logger.warning("No GPUs available, using CPU fallback")
            self.gpu_ids = [0]  # Fallback to default device

        # Create memory managers for each GPU
        self._memory_managers: Dict[int, MemoryManager] = {
            gpu_id: MemoryManager(gpu_id=gpu_id)
            for gpu_id in self.gpu_ids
        }

        logger.info(
            f"GPUDistributor initialized: {len(self.gpu_ids)} GPUs, "
            f"strategy={strategy.value}"
        )

    @property
    def gpu_count(self) -> int:
        """Number of available GPUs."""
        return len(self.gpu_ids)

    def is_multi_gpu(self) -> bool:
        """Check if multiple GPUs are available."""
        return self.gpu_count > 1

    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """Get info for a specific GPU."""
        for gpu in get_available_gpus():
            if gpu.index == gpu_id:
                return gpu
        return None

    def get_next_gpu(self) -> int:
        """Get next GPU based on distribution strategy.

        Returns:
            GPU device ID
        """
        if not self.gpu_ids:
            return 0

        with self._lock:
            if self.strategy == DistributionStrategy.ROUND_ROBIN:
                gpu_id = self.gpu_ids[self._round_robin_index % len(self.gpu_ids)]
                self._round_robin_index += 1
                return gpu_id

            elif self.strategy == DistributionStrategy.MEMORY_BASED:
                # Select GPU with most free memory
                best_gpu = self.gpu_ids[0]
                best_free = 0

                for gpu_id in self.gpu_ids:
                    manager = self._memory_managers.get(gpu_id)
                    if manager:
                        free = manager.get_free_memory_mb()
                        if free > best_free:
                            best_free = free
                            best_gpu = gpu_id

                return best_gpu

            else:  # LOAD_BALANCED
                # Balance by current utilization and memory
                gpus = get_available_gpus()
                valid_gpus = [g for g in gpus if g.index in self.gpu_ids]

                if not valid_gpus:
                    return self.gpu_ids[0]

                # Score by free memory and low utilization
                def score(gpu: GPUInfo) -> float:
                    return gpu.free_memory_mb * (100 - gpu.utilization_percent)

                best = max(valid_gpus, key=score)
                return best.index

    def distribute_frames(
        self,
        frames: List[Path],
        total_count: Optional[int] = None,
    ) -> Dict[int, List[Path]]:
        """Distribute frames across available GPUs.

        Args:
            frames: List of frame paths to distribute
            total_count: Total frame count (for progress calculation)

        Returns:
            Dictionary mapping GPU ID to list of frame paths
        """
        if not frames:
            return {gpu: [] for gpu in self.gpu_ids}

        if not self.is_multi_gpu():
            return {self.gpu_ids[0]: frames}

        assignments: Dict[int, List[Path]] = {gpu: [] for gpu in self.gpu_ids}

        if self.strategy == DistributionStrategy.ROUND_ROBIN:
            # Simple round-robin distribution
            for i, frame in enumerate(frames):
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                assignments[gpu_id].append(frame)

        elif self.strategy == DistributionStrategy.MEMORY_BASED:
            # Distribute proportionally based on free memory
            memory_totals = {}
            for gpu_id in self.gpu_ids:
                manager = self._memory_managers.get(gpu_id)
                memory_totals[gpu_id] = manager.get_free_memory_mb() if manager else 1000

            total_memory = sum(memory_totals.values())

            # Calculate proportional allocation
            allocations = {
                gpu_id: int(len(frames) * mem / total_memory)
                for gpu_id, mem in memory_totals.items()
            }

            # Distribute frames according to allocation
            idx = 0
            for gpu_id, count in allocations.items():
                assignments[gpu_id] = frames[idx:idx + count]
                idx += count

            # Handle remainder
            while idx < len(frames):
                gpu_id = self.get_next_gpu()
                assignments[gpu_id].append(frames[idx])
                idx += 1

        else:  # LOAD_BALANCED
            # Distribute based on current load
            for frame in frames:
                gpu_id = self.get_next_gpu()
                assignments[gpu_id].append(frame)

        # Log distribution
        for gpu_id, gpu_frames in assignments.items():
            logger.debug(f"GPU {gpu_id}: {len(gpu_frames)} frames assigned")

        return assignments

    def get_memory_manager(self, gpu_id: int) -> Optional[MemoryManager]:
        """Get memory manager for a specific GPU."""
        return self._memory_managers.get(gpu_id)

    def get_all_memory_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get memory statistics for all GPUs."""
        return {
            gpu_id: manager.get_statistics()
            for gpu_id, manager in self._memory_managers.items()
        }

    def cleanup_all(self) -> None:
        """Run cleanup on all GPU memory managers."""
        for manager in self._memory_managers.values():
            manager.cleanup()


# =============================================================================
# Frame Buffer
# =============================================================================


class FrameBuffer(Generic[FrameT]):
    """Thread-safe frame buffer with configurable maximum size.

    Provides a bounded buffer for managing frames in memory during
    streaming pipeline processing. Blocks when full to prevent
    memory exhaustion.

    Example:
        >>> buffer = FrameBuffer(max_size=50)
        >>> buffer.put(frame_data)
        >>> frame = buffer.get()
    """

    def __init__(
        self,
        max_size: int = 50,
        timeout: float = 30.0,
    ):
        """Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to buffer
            timeout: Timeout for blocking operations (seconds)
        """
        self.max_size = max_size
        self.timeout = timeout
        self._queue: Queue[Optional[FrameT]] = Queue(maxsize=max_size)
        self._closed = False
        self._total_in = 0
        self._total_out = 0
        self._lock = threading.Lock()

    def put(self, frame: FrameT, timeout: Optional[float] = None) -> bool:
        """Add frame to buffer.

        Args:
            frame: Frame data to add
            timeout: Optional timeout override

        Returns:
            True if successful, False if buffer is closed or timeout

        Raises:
            RuntimeError: If buffer is closed
        """
        if self._closed:
            raise RuntimeError("Cannot put to closed buffer")

        try:
            self._queue.put(frame, timeout=timeout or self.timeout)
            with self._lock:
                self._total_in += 1
            return True
        except Exception:
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[FrameT]:
        """Get frame from buffer.

        Args:
            timeout: Optional timeout override

        Returns:
            Frame data or None if timeout/closed
        """
        try:
            frame = self._queue.get(timeout=timeout or self.timeout)
            if frame is not None:
                with self._lock:
                    self._total_out += 1
            return frame
        except Empty:
            return None

    def close(self) -> None:
        """Close buffer and signal consumers."""
        self._closed = True
        # Put sentinel to unblock consumers
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

    @property
    def is_closed(self) -> bool:
        """Check if buffer is closed."""
        return self._closed

    @property
    def size(self) -> int:
        """Current number of frames in buffer."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._queue.empty()

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "max_size": self.max_size,
                "current_size": self.size,
                "total_in": self._total_in,
                "total_out": self._total_out,
            }


# =============================================================================
# Streaming Pipeline
# =============================================================================


class PipelineStage(Protocol):
    """Protocol for pipeline stage processors."""

    def process(self, input_data: Any) -> Any:
        """Process input and return output."""
        ...


@dataclass
class PipelineFrame:
    """Frame data flowing through the pipeline."""
    index: int
    path: Path
    data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    error: Optional[str] = None


class StreamingPipeline:
    """Multi-stage streaming pipeline for frame processing.

    Processes frames as they're extracted without waiting for
    full video extraction. Uses configurable buffers between
    stages to optimize memory usage.

    Pipeline stages:
    1. Extract: Read frames from source
    2. Enhance: Apply enhancement (upscaling, restoration)
    3. Write: Write enhanced frames to output

    Example:
        >>> pipeline = StreamingPipeline(
        ...     config=StreamingConfig(chunk_size=100, max_buffer_size=50)
        ... )
        >>> pipeline.set_enhancer(my_enhance_function)
        >>> results = pipeline.process(input_dir, output_dir)
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        gpu_distributor: Optional[GPUDistributor] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """Initialize streaming pipeline.

        Args:
            config: Pipeline configuration
            gpu_distributor: Optional multi-GPU distributor
            memory_manager: Optional memory manager
        """
        self.config = config or StreamingConfig()
        self.gpu_distributor = gpu_distributor
        self.memory_manager = memory_manager or MemoryManager()

        # Pipeline buffers
        self._extract_buffer: Optional[FrameBuffer[PipelineFrame]] = None
        self._enhance_buffer: Optional[FrameBuffer[PipelineFrame]] = None

        # Processing state
        self._running = False
        self._error: Optional[str] = None
        self._processed_count = 0
        self._total_count = 0
        self._lock = threading.Lock()

        # Callbacks
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._batch_callback: Optional[Callable[[BatchResult], None]] = None

        # Enhancement function
        self._enhance_fn: Optional[Callable[[List[PipelineFrame]], List[PipelineFrame]]] = None

        logger.info(
            f"StreamingPipeline initialized: "
            f"chunk_size={self.config.chunk_size}, "
            f"buffer_size={self.config.max_buffer_size}"
        )

    def set_enhancer(
        self,
        enhance_fn: Callable[[List[PipelineFrame]], List[PipelineFrame]],
    ) -> None:
        """Set the enhancement function.

        Args:
            enhance_fn: Function that takes list of frames and returns enhanced frames
        """
        self._enhance_fn = enhance_fn

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None],
    ) -> None:
        """Set progress callback.

        Args:
            callback: Function called with (progress, message)
        """
        self._progress_callback = callback

    def set_batch_callback(
        self,
        callback: Callable[[BatchResult], None],
    ) -> None:
        """Set batch completion callback.

        Args:
            callback: Function called with BatchResult
        """
        self._batch_callback = callback

    def _create_buffers(self) -> None:
        """Create pipeline buffers."""
        self._extract_buffer = FrameBuffer(max_size=self.config.max_buffer_size)
        self._enhance_buffer = FrameBuffer(max_size=self.config.max_buffer_size)

    def _report_progress(self, message: str) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            with self._lock:
                progress = self._processed_count / max(1, self._total_count)
            self._progress_callback(progress, message)

    def _extract_stage(
        self,
        frames: List[Path],
    ) -> None:
        """Extract stage: read frames and push to buffer.

        Args:
            frames: List of frame paths to process
        """
        logger.debug(f"Extract stage started: {len(frames)} frames")

        for i, frame_path in enumerate(frames):
            if not self._running:
                break

            pipeline_frame = PipelineFrame(
                index=i,
                path=frame_path,
                metadata={"source_path": str(frame_path)},
            )

            try:
                if self._extract_buffer:
                    self._extract_buffer.put(pipeline_frame)
            except Exception as e:
                logger.error(f"Extract buffer error: {e}")
                pipeline_frame.error = str(e)

        # Signal end of extraction
        if self._extract_buffer:
            self._extract_buffer.close()

        logger.debug("Extract stage completed")

    def _enhance_stage(self) -> None:
        """Enhance stage: process frames in batches."""
        logger.debug("Enhance stage started")

        batch: List[PipelineFrame] = []
        batch_size = self._get_optimal_batch_size()
        batch_id = 0

        while self._running:
            # Get frame from extract buffer
            if self._extract_buffer:
                frame = self._extract_buffer.get(timeout=1.0)
            else:
                break

            if frame is None:
                # Check if buffer is closed
                if self._extract_buffer and self._extract_buffer.is_closed:
                    break
                continue

            batch.append(frame)

            # Process batch when full
            if len(batch) >= batch_size:
                self._process_batch(batch, batch_id)
                batch = []
                batch_id += 1

                # Adjust batch size based on memory
                batch_size = self._get_optimal_batch_size()

        # Process remaining frames
        if batch:
            self._process_batch(batch, batch_id)

        # Signal end of enhancement
        if self._enhance_buffer:
            self._enhance_buffer.close()

        logger.debug("Enhance stage completed")

    def _process_batch(
        self,
        batch: List[PipelineFrame],
        batch_id: int,
    ) -> None:
        """Process a batch of frames.

        Args:
            batch: List of frames to process
            batch_id: Batch identifier
        """
        start_time = time.time()
        gpu_id = None

        if self.gpu_distributor:
            gpu_id = self.gpu_distributor.get_next_gpu()

        try:
            # Apply enhancement
            if self._enhance_fn:
                enhanced = self._enhance_fn(batch)
            else:
                enhanced = batch
                for frame in enhanced:
                    frame.processed = True

            # Push to output buffer
            for frame in enhanced:
                if self._enhance_buffer:
                    self._enhance_buffer.put(frame)

                with self._lock:
                    self._processed_count += 1

            # Create batch result
            elapsed = time.time() - start_time
            memory_used = None
            if self.memory_manager:
                snapshot = self.memory_manager.get_current_usage()
                if snapshot:
                    memory_used = snapshot.used_mb

            result = BatchResult(
                batch_id=batch_id,
                frame_indices=[f.index for f in enhanced],
                output_paths=[f.path for f in enhanced],
                processing_time_seconds=elapsed,
                gpu_id=gpu_id,
                memory_used_mb=memory_used,
            )

            if self._batch_callback:
                self._batch_callback(result)

            self._report_progress(f"Batch {batch_id}: {len(batch)} frames")

        except Exception as e:
            logger.error(f"Batch {batch_id} processing error: {e}")
            for frame in batch:
                frame.error = str(e)
                if self._enhance_buffer:
                    self._enhance_buffer.put(frame)

    def _write_stage(
        self,
        output_dir: Path,
    ) -> List[Path]:
        """Write stage: write enhanced frames to output.

        Args:
            output_dir: Output directory

        Returns:
            List of output paths
        """
        logger.debug("Write stage started")
        output_paths: List[Path] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        while self._running:
            if self._enhance_buffer:
                frame = self._enhance_buffer.get(timeout=1.0)
            else:
                break

            if frame is None:
                if self._enhance_buffer and self._enhance_buffer.is_closed:
                    break
                continue

            if frame.error:
                logger.warning(f"Skipping frame {frame.index}: {frame.error}")
                continue

            # Write frame (in real implementation, this would save enhanced data)
            output_path = output_dir / f"frame_{frame.index:08d}.png"
            output_paths.append(output_path)

        logger.debug(f"Write stage completed: {len(output_paths)} frames")
        return output_paths

    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on configuration and memory."""
        if self.config.batch_size > 0:
            return self.config.batch_size

        # Auto-detect based on memory
        if self.memory_manager:
            # Assume 1080p frames for estimation
            return self.memory_manager.get_optimal_batch_size(
                frame_size=(1920, 1080),
                max_batch_size=self.config.max_batch_size,
            )

        return 1

    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Path]:
        """Process frames through the streaming pipeline.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback

        Returns:
            List of output frame paths
        """
        if progress_callback:
            self.set_progress_callback(progress_callback)

        # Get input frames
        frames = sorted(input_dir.glob("frame_*.png"))
        if not frames:
            logger.warning(f"No frames found in {input_dir}")
            return []

        with self._lock:
            self._total_count = len(frames)
            self._processed_count = 0

        logger.info(f"Starting pipeline: {len(frames)} frames")

        # Create buffers
        self._create_buffers()
        self._running = True
        output_paths: List[Path] = []

        try:
            # Start pipeline stages in threads
            with ThreadPoolExecutor(max_workers=3) as executor:
                extract_future = executor.submit(self._extract_stage, frames)
                enhance_future = executor.submit(self._enhance_stage)
                write_future = executor.submit(self._write_stage, output_dir)

                # Wait for completion
                extract_future.result()
                enhance_future.result()
                output_paths = write_future.result()

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._error = str(e)
            raise
        finally:
            self._running = False

            # Cleanup
            if self.config.cleanup_between_chunks:
                if self.memory_manager:
                    self.memory_manager.cleanup()
                if self.gpu_distributor:
                    self.gpu_distributor.cleanup_all()

        return output_paths

    def process_chunks(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[List[Path], None, None]:
        """Process frames in chunks, yielding results as they complete.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback

        Yields:
            List of output paths for each completed chunk
        """
        if progress_callback:
            self.set_progress_callback(progress_callback)

        frames = sorted(input_dir.glob("frame_*.png"))
        if not frames:
            return

        chunk_size = self.config.chunk_size
        total_chunks = (len(frames) + chunk_size - 1) // chunk_size

        with self._lock:
            self._total_count = len(frames)
            self._processed_count = 0

        for chunk_id in range(total_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(frames))
            chunk_frames = frames[start_idx:end_idx]

            chunk_output_dir = output_dir / f"chunk_{chunk_id:04d}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Processing chunk {chunk_id + 1}/{total_chunks}: "
                f"frames {start_idx}-{end_idx}"
            )

            # Process chunk through pipeline
            self._create_buffers()
            self._running = True

            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    extract_future = executor.submit(self._extract_stage, chunk_frames)
                    enhance_future = executor.submit(self._enhance_stage)
                    write_future = executor.submit(self._write_stage, chunk_output_dir)

                    extract_future.result()
                    enhance_future.result()
                    chunk_paths = write_future.result()

                yield chunk_paths

            finally:
                self._running = False

                if self.config.cleanup_between_chunks:
                    if self.memory_manager:
                        self.memory_manager.cleanup()

        logger.info("All chunks processed")

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._running = False
        if self._extract_buffer:
            self._extract_buffer.close()
        if self._enhance_buffer:
            self._enhance_buffer.close()

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    @property
    def progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        with self._lock:
            if self._total_count == 0:
                return 0.0
            return self._processed_count / self._total_count

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats: Dict[str, Any] = {
            "processed": self._processed_count,
            "total": self._total_count,
            "progress": self.progress,
            "running": self._running,
            "error": self._error,
        }

        if self._extract_buffer:
            stats["extract_buffer"] = self._extract_buffer.get_stats()
        if self._enhance_buffer:
            stats["enhance_buffer"] = self._enhance_buffer.get_stats()
        if self.memory_manager:
            stats["memory"] = self.memory_manager.get_statistics()

        return stats


# =============================================================================
# Batch Frame Processor
# =============================================================================


class BatchFrameProcessor:
    """Process multiple frames in parallel batches with VRAM optimization.

    Automatically detects optimal batch size based on available VRAM
    and provides progress callbacks per batch.

    Example:
        >>> processor = BatchFrameProcessor(batch_size=0)  # Auto-detect
        >>> processor.set_process_fn(enhance_frames)
        >>> results = processor.process(frames, progress_callback=on_progress)
    """

    def __init__(
        self,
        batch_size: int = 0,
        max_batch_size: int = 8,
        gpu_id: int = 0,
        memory_limit_mb: Optional[int] = None,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Frames per batch (0 = auto-detect)
            max_batch_size: Maximum batch size when auto-detecting
            gpu_id: GPU device ID
            memory_limit_mb: Optional memory limit
        """
        self.configured_batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.gpu_id = gpu_id

        self.memory_manager = MemoryManager(
            gpu_id=gpu_id,
            memory_limit_mb=memory_limit_mb,
        )

        self._process_fn: Optional[Callable[[List[Path]], List[Path]]] = None
        self._batch_results: List[BatchResult] = []

    def set_process_fn(
        self,
        fn: Callable[[List[Path]], List[Path]],
    ) -> None:
        """Set the frame processing function.

        Args:
            fn: Function that takes list of input paths and returns output paths
        """
        self._process_fn = fn

    def get_optimal_batch_size(
        self,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> int:
        """Get optimal batch size based on VRAM.

        Args:
            frame_size: Optional frame size for estimation

        Returns:
            Optimal batch size
        """
        if self.configured_batch_size > 0:
            return self.configured_batch_size

        if frame_size:
            return self.memory_manager.get_optimal_batch_size(
                frame_size=frame_size,
                max_batch_size=self.max_batch_size,
            )

        # Default estimation for 1080p
        return self.memory_manager.get_optimal_batch_size(
            frame_size=(1920, 1080),
            max_batch_size=self.max_batch_size,
        )

    def process(
        self,
        frames: List[Path],
        progress_callback: Optional[Callable[[int, int, BatchResult], None]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> List[Path]:
        """Process frames in optimized batches.

        Args:
            frames: List of frame paths to process
            progress_callback: Called with (batch_num, total_batches, result)
            frame_size: Optional frame size for batch optimization

        Returns:
            List of output paths
        """
        if not frames:
            return []

        if not self._process_fn:
            raise ValueError("Process function not set. Call set_process_fn first.")

        batch_size = self.get_optimal_batch_size(frame_size)
        total_batches = (len(frames) + batch_size - 1) // batch_size
        output_paths: List[Path] = []
        self._batch_results = []

        logger.info(
            f"Processing {len(frames)} frames in {total_batches} batches "
            f"(batch_size={batch_size})"
        )

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]

            start_time = time.time()

            try:
                batch_outputs = self._process_fn(batch_frames)
                output_paths.extend(batch_outputs)
                success = True
                error = None
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                success = False
                error = str(e)
                batch_outputs = []

            elapsed = time.time() - start_time
            snapshot = self.memory_manager.get_current_usage()

            result = BatchResult(
                batch_id=batch_num,
                frame_indices=list(range(start_idx, end_idx)),
                output_paths=batch_outputs,
                processing_time_seconds=elapsed,
                gpu_id=self.gpu_id,
                memory_used_mb=snapshot.used_mb if snapshot else None,
                success=success,
                error=error,
            )

            self._batch_results.append(result)

            if progress_callback:
                progress_callback(batch_num + 1, total_batches, result)

            # Adjust batch size based on memory pressure
            if self.configured_batch_size == 0:
                new_tile, new_batch = self.memory_manager.adjust_for_memory_pressure(
                    current_tile_size=0,
                    current_batch_size=batch_size,
                )
                if new_batch != batch_size:
                    batch_size = new_batch
                    total_batches = (len(frames) - end_idx + batch_size - 1) // batch_size + batch_num + 1

        return output_paths

    def get_results(self) -> List[BatchResult]:
        """Get all batch results."""
        return self._batch_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self._batch_results:
            return {"batches": 0}

        times = [r.processing_time_seconds for r in self._batch_results]
        memories = [r.memory_used_mb for r in self._batch_results if r.memory_used_mb]

        return {
            "batches": len(self._batch_results),
            "total_frames": sum(len(r.frame_indices) for r in self._batch_results),
            "successful_batches": sum(1 for r in self._batch_results if r.success),
            "total_time_seconds": sum(times),
            "avg_time_per_batch": sum(times) / len(times),
            "avg_memory_mb": sum(memories) / len(memories) if memories else 0,
            "peak_memory_mb": max(memories) if memories else 0,
        }


class StreamingProcessor:
    """Process video in chunks for streaming output.

    This enables:
    1. Early preview - watch processed chunks while rest processes
    2. Fault tolerance - recover from crashes with partial output
    3. Memory efficiency - process huge videos without disk overflow
    4. Progress visibility - see actual output, not just percentages

    Example:
        >>> processor = StreamingProcessor(config, streaming_config)
        >>> for chunk in processor.process_streaming(frames_dir, output_dir):
        ...     print(f"Chunk {chunk.chunk_id} ready: {chunk.output_path}")
        ...     # User can watch this chunk immediately!
    """

    def __init__(
        self,
        streaming_config: Optional[StreamingConfig] = None,
        framerate: float = 30.0,
        audio_path: Optional[Path] = None,
    ):
        """Initialize streaming processor.

        Args:
            streaming_config: Streaming configuration (uses defaults if None)
            framerate: Video frame rate for chunk duration calculation
            audio_path: Optional audio file to include in chunks
        """
        self.config = streaming_config or StreamingConfig()
        self.framerate = framerate
        self.audio_path = audio_path
        self.state = StreamingState()
        self._chunk_callbacks: List[Callable[[ChunkInfo], None]] = []

    def on_chunk_complete(self, callback: Callable[[ChunkInfo], None]) -> None:
        """Register callback for when a chunk is completed.

        Args:
            callback: Function called with ChunkInfo when chunk is ready
        """
        self._chunk_callbacks.append(callback)

    def _notify_chunk_complete(self, chunk: ChunkInfo) -> None:
        """Notify all registered callbacks of chunk completion."""
        for callback in self._chunk_callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.warning(f"Chunk callback error: {e}")

    def calculate_chunks(
        self,
        total_frames: int,
        framerate: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """Calculate frame ranges for each chunk.

        Args:
            total_frames: Total number of frames to process
            framerate: Override frame rate (uses instance default if None)

        Returns:
            List of (start_frame, end_frame) tuples for each chunk
        """
        fps = framerate or self.framerate

        # Calculate frames per chunk based on duration
        frames_per_chunk = int(self.config.chunk_duration_seconds * fps)

        # Apply limits
        frames_per_chunk = max(frames_per_chunk, self.config.min_chunk_frames)
        frames_per_chunk = min(frames_per_chunk, self.config.max_chunk_frames)

        chunks = []
        start = 0

        while start < total_frames:
            end = min(start + frames_per_chunk, total_frames)
            chunks.append((start, end))
            start = end

        return chunks

    def process_streaming(
        self,
        frames_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Generator[ChunkInfo, None, None]:
        """Process frames and yield chunks as they complete.

        This is the main streaming interface. Each yielded chunk
        represents a playable video segment.

        Args:
            frames_dir: Directory containing enhanced frames
            output_dir: Directory for output chunk files
            progress_callback: Optional callback(progress, message)

        Yields:
            ChunkInfo for each completed chunk

        Example:
            >>> for chunk in processor.process_streaming(frames, output):
            ...     # This chunk is immediately playable!
            ...     player.queue(chunk.output_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all frames sorted
        frames = sorted(frames_dir.glob("frame_*.png"))
        total_frames = len(frames)

        if total_frames == 0:
            logger.error("No frames found for streaming processing")
            self.state.error = "No frames found"
            return

        # Calculate chunk boundaries
        chunk_ranges = self.calculate_chunks(total_frames)

        self.state.total_frames = total_frames
        self.state.total_chunks = len(chunk_ranges)

        logger.info(
            f"Streaming processing: {total_frames} frames in "
            f"{len(chunk_ranges)} chunks of ~{self.config.chunk_duration_seconds}s each"
        )

        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for chunk_id, (start_frame, end_frame) in enumerate(chunk_ranges):
            self.state.current_chunk = chunk_id

            try:
                chunk_info = self._process_chunk(
                    chunk_id=chunk_id,
                    frames=frames[start_frame:end_frame],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    output_dir=chunks_dir,
                    is_final=(chunk_id == len(chunk_ranges) - 1),
                )

                self.state.processed_frames = end_frame
                self.state.chunks_completed.append(chunk_info)

                if progress_callback:
                    progress_callback(
                        self.state.progress,
                        f"Chunk {chunk_id + 1}/{len(chunk_ranges)} complete"
                    )

                # Notify callbacks
                if self.config.emit_on_chunk_complete:
                    self._notify_chunk_complete(chunk_info)

                yield chunk_info

            except Exception as e:
                logger.error(f"Chunk {chunk_id} failed: {e}")
                self.state.error = str(e)
                raise

        self.state.is_complete = True
        logger.info(f"Streaming processing complete: {len(self.state.chunks_completed)} chunks")

    def _process_chunk(
        self,
        chunk_id: int,
        frames: List[Path],
        start_frame: int,
        end_frame: int,
        output_dir: Path,
        is_final: bool = False,
    ) -> ChunkInfo:
        """Process a single chunk of frames into a video segment.

        Args:
            chunk_id: Chunk identifier
            frames: List of frame paths for this chunk
            start_frame: Starting frame number
            end_frame: Ending frame number (exclusive)
            output_dir: Directory for output
            is_final: Whether this is the last chunk

        Returns:
            ChunkInfo with details about the processed chunk
        """
        chunk_frames_dir = output_dir / f"chunk_{chunk_id:04d}_frames"
        chunk_frames_dir.mkdir(parents=True, exist_ok=True)

        # Create sequential frame links/copies for ffmpeg
        for i, frame in enumerate(frames):
            dest = chunk_frames_dir / f"frame_{i:08d}.png"
            # Use symlink if possible, else copy
            try:
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(frame.resolve())
            except OSError:
                shutil.copy(frame, dest)

        # Output path
        output_path = output_dir / f"chunk_{chunk_id:04d}.{self.config.output_format}"

        # Calculate duration
        duration = len(frames) / self.framerate

        # Build ffmpeg command
        input_pattern = chunk_frames_dir / "frame_%08d.png"

        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(self.framerate),
            '-i', str(input_pattern),
        ]

        # Add audio segment if available (for first chunk or with offset)
        if self.audio_path and self.audio_path.exists():
            audio_start = start_frame / self.framerate
            cmd.extend([
                '-ss', str(audio_start),
                '-i', str(self.audio_path),
                '-t', str(duration),
                '-c:a', 'aac',
                '-b:a', '192k',
            ])

        # Video encoding
        cmd.extend([
            '-c:v', 'libx265',
            '-crf', str(self.config.crf),
            '-preset', self.config.preset,
            '-pix_fmt', 'yuv420p10le',
            '-t', str(duration),
            str(output_path),
        ])

        logger.info(f"Processing chunk {chunk_id}: frames {start_frame}-{end_frame}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min per chunk
            )
            logger.debug(f"Chunk {chunk_id} ffmpeg output: {result.stderr[-500:]}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Chunk {chunk_id} encoding failed: {e.stderr}")
            raise RuntimeError(f"Failed to encode chunk {chunk_id}: {e.stderr}")
        finally:
            # Cleanup chunk frames directory
            if not self.config.keep_intermediate_chunks:
                shutil.rmtree(chunk_frames_dir, ignore_errors=True)

        return ChunkInfo(
            chunk_id=chunk_id,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_count=len(frames),
            output_path=output_path,
            duration_seconds=duration,
            is_final=is_final,
        )

    def merge_chunks(
        self,
        output_path: Path,
        chunks: Optional[List[ChunkInfo]] = None,
    ) -> Path:
        """Merge all completed chunks into a single video file.

        Args:
            output_path: Final output video path
            chunks: Specific chunks to merge (uses all completed if None)

        Returns:
            Path to merged video file
        """
        chunks = chunks or self.state.chunks_completed

        if not chunks:
            raise ValueError("No chunks to merge")

        # Sort chunks by ID
        chunks = sorted(chunks, key=lambda c: c.chunk_id)

        logger.info(f"Merging {len(chunks)} chunks into {output_path}")

        # Create concat file for ffmpeg
        concat_file = output_path.parent / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for chunk in chunks:
                # Use absolute paths with proper escaping
                escaped_path = str(chunk.output_path.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # No re-encoding needed
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            logger.info(f"Merge complete: {output_path}")

            # Cleanup
            concat_file.unlink()
            if self.config.cleanup_chunks_on_merge:
                for chunk in chunks:
                    if chunk.output_path.exists():
                        chunk.output_path.unlink()

            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Merge failed: {e.stderr}")
            raise RuntimeError(f"Failed to merge chunks: {e.stderr}")

    def get_playable_chunks(self) -> List[Path]:
        """Get list of chunk files that are ready to play.

        Returns:
            List of paths to completed chunk videos
        """
        return [c.output_path for c in self.state.chunks_completed if c.output_path.exists()]

    def get_state(self) -> StreamingState:
        """Get current streaming state.

        Returns:
            StreamingState with progress information
        """
        return self.state

    def estimate_remaining_time(self, elapsed_seconds: float) -> float:
        """Estimate remaining processing time.

        Args:
            elapsed_seconds: Time elapsed since processing started

        Returns:
            Estimated seconds remaining
        """
        if self.state.progress <= 0:
            return 0.0

        total_estimated = elapsed_seconds / self.state.progress
        return total_estimated - elapsed_seconds


def create_streaming_restorer(
    frames_dir: Path,
    output_dir: Path,
    framerate: float = 30.0,
    chunk_duration: float = 300.0,
    audio_path: Optional[Path] = None,
    on_chunk_ready: Optional[Callable[[ChunkInfo], None]] = None,
) -> Generator[ChunkInfo, None, Path]:
    """Convenience function for streaming video restoration.

    Args:
        frames_dir: Directory with enhanced frames
        output_dir: Output directory for chunks and final video
        framerate: Video frame rate
        chunk_duration: Seconds per chunk
        audio_path: Optional audio file
        on_chunk_ready: Optional callback when chunk is ready

    Yields:
        ChunkInfo for each completed chunk

    Returns:
        Path to final merged video

    Example:
        >>> gen = create_streaming_restorer(frames, output)
        >>> for chunk in gen:
        ...     print(f"Preview ready: {chunk.output_path}")
        >>> final_video = gen.value  # After iteration completes
    """
    config = StreamingConfig(chunk_duration_seconds=chunk_duration)
    processor = StreamingProcessor(
        streaming_config=config,
        framerate=framerate,
        audio_path=audio_path,
    )

    if on_chunk_ready:
        processor.on_chunk_complete(on_chunk_ready)

    # Process all chunks
    for chunk in processor.process_streaming(frames_dir, output_dir):
        yield chunk

    # Merge into final video
    final_path = output_dir / f"restored_video.{config.output_format}"
    return processor.merge_chunks(final_path)
