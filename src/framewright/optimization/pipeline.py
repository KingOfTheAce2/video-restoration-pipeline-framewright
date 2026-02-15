"""Pipeline optimization for high-throughput frame processing."""

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages in the processing pipeline."""
    READ = auto()
    DENOISE = auto()
    UPSCALE = auto()
    FACE_RESTORE = auto()
    COLOR = auto()
    TEMPORAL = auto()
    WRITE = auto()


@dataclass
class FrameData:
    """Data for a single frame in the pipeline."""
    frame_number: int
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: ProcessingStage = ProcessingStage.READ
    error: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for pipeline optimization."""
    # Batching
    batch_size: int = 4
    max_batch_size: int = 16
    adaptive_batching: bool = True

    # Prefetching
    prefetch_frames: int = 8
    prefetch_workers: int = 2

    # Parallelism
    parallel_stages: bool = True
    stage_queue_size: int = 16

    # Memory management
    max_memory_mb: int = 8000
    frame_memory_limit_mb: int = 500

    # Performance
    enable_cuda_streams: bool = True
    enable_tensor_cores: bool = True
    pin_memory: bool = True


class FramePrefetcher:
    """Prefetch frames from disk for faster processing."""

    def __init__(
        self,
        frame_paths: List[Path],
        prefetch_count: int = 8,
        num_workers: int = 2,
    ):
        self.frame_paths = frame_paths
        self.prefetch_count = prefetch_count
        self.num_workers = num_workers

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
        self._stop_event = threading.Event()
        self._workers: List[threading.Thread] = []
        self._current_index = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start prefetching."""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"Prefetcher-{i}",
            )
            worker.start()
            self._workers.append(worker)

        logger.debug(f"Started {self.num_workers} prefetch workers")

    def stop(self) -> None:
        """Stop prefetching."""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._workers.clear()

    def _worker_loop(self) -> None:
        """Worker loop for prefetching frames."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV required for prefetching")
            return

        while not self._stop_event.is_set():
            # Get next index to fetch
            with self._lock:
                if self._current_index >= len(self.frame_paths):
                    break
                idx = self._current_index
                self._current_index += 1

            # Load frame
            path = self.frame_paths[idx]
            try:
                frame = cv2.imread(str(path))
                if frame is not None:
                    frame_data = FrameData(
                        frame_number=idx,
                        data=frame,
                        metadata={"source_path": str(path)},
                    )
                    self._queue.put(frame_data, timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to prefetch frame {idx}: {e}")

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate over prefetched frames."""
        fetched = 0
        while fetched < len(self.frame_paths):
            try:
                frame = self._queue.get(timeout=10.0)
                yield frame
                fetched += 1
            except queue.Empty:
                if self._stop_event.is_set():
                    break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class BatchProcessor:
    """Process frames in batches for GPU efficiency."""

    def __init__(
        self,
        batch_size: int = 4,
        adaptive: bool = True,
        max_batch_size: int = 16,
    ):
        self.batch_size = batch_size
        self.adaptive = adaptive
        self.max_batch_size = max_batch_size

        # Timing for adaptive batching
        self._batch_times: deque = deque(maxlen=10)
        self._optimal_batch: int = batch_size

    def process_batch(
        self,
        frames: List[FrameData],
        processor: Callable[[List[np.ndarray]], List[np.ndarray]],
    ) -> List[FrameData]:
        """Process a batch of frames.

        Args:
            frames: List of frame data
            processor: Function that processes list of frame arrays

        Returns:
            Processed frame data
        """
        start_time = time.time()

        # Extract arrays
        arrays = [f.data for f in frames]

        # Process
        try:
            results = processor(arrays)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Return original frames with error
            for f in frames:
                f.error = str(e)
            return frames

        # Update frame data
        for i, (frame, result) in enumerate(zip(frames, results)):
            frame.data = result

        # Track timing for adaptive batching
        elapsed = time.time() - start_time
        self._batch_times.append((len(frames), elapsed))

        if self.adaptive:
            self._update_batch_size()

        return frames

    def _update_batch_size(self) -> None:
        """Update batch size based on performance metrics."""
        if len(self._batch_times) < 3:
            return

        # Calculate throughput for recent batches
        throughputs = []
        for batch_size, elapsed in self._batch_times:
            if elapsed > 0:
                throughputs.append(batch_size / elapsed)

        avg_throughput = sum(throughputs) / len(throughputs)

        # Try to optimize batch size
        current_batch = self.batch_size
        current_throughput = throughputs[-1] if throughputs else 0

        if current_throughput < avg_throughput * 0.8 and current_batch > 1:
            # Reduce batch size
            self.batch_size = max(1, current_batch - 1)
        elif current_throughput >= avg_throughput and current_batch < self.max_batch_size:
            # Try larger batch
            self.batch_size = min(self.max_batch_size, current_batch + 1)

    def get_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.batch_size

    def create_batches(
        self,
        frames: Iterator[FrameData],
    ) -> Iterator[List[FrameData]]:
        """Create batches from frame iterator."""
        batch = []

        for frame in frames:
            batch.append(frame)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        # Yield remaining frames
        if batch:
            yield batch


class PipelineOptimizer:
    """Optimized pipeline for frame processing."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self._stage_queues: Dict[ProcessingStage, queue.Queue] = {}
        self._stage_threads: Dict[ProcessingStage, threading.Thread] = {}
        self._stop_event = threading.Event()

        # Performance tracking
        self._stage_times: Dict[ProcessingStage, deque] = {
            stage: deque(maxlen=100) for stage in ProcessingStage
        }

        # CUDA streams if available
        self._cuda_streams: Dict[ProcessingStage, Any] = {}
        self._setup_cuda()

    def _setup_cuda(self) -> None:
        """Setup CUDA streams for parallel execution."""
        if not self.config.enable_cuda_streams:
            return

        try:
            import torch
            if torch.cuda.is_available():
                for stage in [ProcessingStage.DENOISE, ProcessingStage.UPSCALE,
                              ProcessingStage.FACE_RESTORE]:
                    self._cuda_streams[stage] = torch.cuda.Stream()
                logger.info("Created CUDA streams for parallel processing")
        except ImportError:
            pass

    def process_pipeline(
        self,
        frame_paths: List[Path],
        stages: List[Tuple[ProcessingStage, Callable]],
        output_callback: Callable[[FrameData], None],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Process frames through optimized pipeline.

        Args:
            frame_paths: List of frame paths
            stages: List of (stage, processor_function) tuples
            output_callback: Called with each processed frame
            progress_callback: Called with (current, total)

        Returns:
            Processing statistics
        """
        stats = {
            "total_frames": len(frame_paths),
            "processed_frames": 0,
            "failed_frames": 0,
            "stage_times": {},
            "throughput_fps": 0,
        }

        start_time = time.time()

        # Setup prefetcher
        with FramePrefetcher(
            frame_paths,
            prefetch_count=self.config.prefetch_frames,
            num_workers=self.config.prefetch_workers,
        ) as prefetcher:

            # Setup batch processor
            batch_processor = BatchProcessor(
                batch_size=self.config.batch_size,
                adaptive=self.config.adaptive_batching,
                max_batch_size=self.config.max_batch_size,
            )

            # Process in batches
            processed = 0
            for batch in batch_processor.create_batches(prefetcher):
                # Process through each stage
                for stage, processor in stages:
                    stage_start = time.time()

                    # Use CUDA stream if available
                    stream = self._cuda_streams.get(stage)
                    if stream:
                        try:
                            import torch
                            with torch.cuda.stream(stream):
                                batch = batch_processor.process_batch(batch, processor)
                        except:
                            batch = batch_processor.process_batch(batch, processor)
                    else:
                        batch = batch_processor.process_batch(batch, processor)

                    # Track timing
                    stage_time = time.time() - stage_start
                    self._stage_times[stage].append(stage_time)

                # Output processed frames
                for frame in batch:
                    if frame.error:
                        stats["failed_frames"] += 1
                    else:
                        stats["processed_frames"] += 1

                    output_callback(frame)
                    processed += 1

                    if progress_callback:
                        progress_callback(processed, len(frame_paths))

        # Calculate statistics
        elapsed = time.time() - start_time
        stats["throughput_fps"] = stats["processed_frames"] / elapsed if elapsed > 0 else 0

        for stage in ProcessingStage:
            times = list(self._stage_times[stage])
            if times:
                stats["stage_times"][stage.name] = {
                    "avg_ms": sum(times) / len(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }

        return stats

    def estimate_memory(
        self,
        frame_size: Tuple[int, int],
        batch_size: int,
        stages: List[ProcessingStage],
    ) -> Dict[str, float]:
        """Estimate memory requirements.

        Args:
            frame_size: (width, height)
            batch_size: Batch size
            stages: Processing stages

        Returns:
            Memory estimates in MB
        """
        w, h = frame_size
        frame_mb = (w * h * 3) / (1024 * 1024)  # RGB uint8

        estimates = {
            "frame_batch_mb": frame_mb * batch_size,
            "prefetch_mb": frame_mb * self.config.prefetch_frames,
        }

        # Stage-specific estimates
        stage_vram = {
            ProcessingStage.DENOISE: 2000,  # NAFNet ~2GB
            ProcessingStage.UPSCALE: 3000,  # Real-ESRGAN ~3GB
            ProcessingStage.FACE_RESTORE: 1500,  # GFPGAN ~1.5GB
            ProcessingStage.TEMPORAL: 2000,  # RIFE ~2GB
        }

        total_vram = 500  # Base overhead
        for stage in stages:
            total_vram += stage_vram.get(stage, 100)

        estimates["estimated_vram_mb"] = total_vram
        estimates["recommended_batch_size"] = self._recommend_batch_size(total_vram)

        return estimates

    def _recommend_batch_size(self, estimated_vram_mb: float) -> int:
        """Recommend batch size based on available VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                available = total_vram - estimated_vram_mb - 1000  # 1GB safety margin

                if available > 4000:
                    return 8
                elif available > 2000:
                    return 4
                elif available > 1000:
                    return 2
                else:
                    return 1
        except:
            pass

        return 4  # Default

    def profile_stage(
        self,
        stage: ProcessingStage,
        processor: Callable,
        sample_frames: List[np.ndarray],
    ) -> Dict[str, float]:
        """Profile a processing stage.

        Args:
            stage: Stage to profile
            processor: Processor function
            sample_frames: Sample frames for profiling

        Returns:
            Profile results
        """
        # Warmup
        for _ in range(2):
            processor(sample_frames[:1])

        # Profile
        times = []
        memory_before = self._get_memory_usage()

        for batch_size in [1, 2, 4, 8]:
            if batch_size > len(sample_frames):
                break

            batch = sample_frames[:batch_size]

            # Time multiple iterations
            start = time.time()
            for _ in range(3):
                processor(batch)
            elapsed = (time.time() - start) / 3

            times.append({
                "batch_size": batch_size,
                "time_ms": elapsed * 1000,
                "fps": batch_size / elapsed,
            })

        memory_after = self._get_memory_usage()

        return {
            "stage": stage.name,
            "batch_profiles": times,
            "memory_delta_mb": memory_after - memory_before,
            "recommended_batch": self._find_optimal_batch(times),
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            pass
        return 0

    def _find_optimal_batch(self, profiles: List[Dict]) -> int:
        """Find optimal batch size from profiles."""
        if not profiles:
            return 4

        # Find batch size with best throughput
        best = max(profiles, key=lambda p: p["fps"])
        return best["batch_size"]
