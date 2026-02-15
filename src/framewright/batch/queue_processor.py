"""Batch queue processor for FrameWright.

Handles multiple video restoration jobs with priority queuing,
resource management, and automatic scheduling.
"""

import logging
import threading
import queue
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Priority levels for batch jobs."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class JobProgress:
    """Progress information for a job."""
    stage: str = ""
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    frames_processed: int = 0
    frames_total: int = 0
    current_fps: float = 0.0
    eta_seconds: float = 0.0
    started_at: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class BatchJob:
    """Represents a single batch restoration job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_path: str = ""
    output_path: str = ""
    preset: str = "balanced"
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchJob":
        """Create from dictionary."""
        data = data.copy()
        data["priority"] = JobPriority(data.get("priority", 2))
        data["status"] = JobStatus(data.get("status", "pending"))
        if "progress" in data and isinstance(data["progress"], dict):
            data["progress"] = JobProgress(**data["progress"])
        return cls(**data)


class PriorityQueue:
    """Thread-safe priority queue for batch jobs."""

    def __init__(self):
        self._queue: List[BatchJob] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def put(self, job: BatchJob) -> None:
        """Add job to queue with priority ordering."""
        with self._lock:
            self._queue.append(job)
            # Sort by priority (higher first), then by creation time
            self._queue.sort(
                key=lambda j: (-j.priority.value, j.created_at)
            )
            job.status = JobStatus.QUEUED
            self._condition.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[BatchJob]:
        """Get highest priority job from queue."""
        with self._condition:
            while not self._queue:
                if not self._condition.wait(timeout=timeout):
                    return None

            job = self._queue.pop(0)
            return job

    def peek(self) -> Optional[BatchJob]:
        """Peek at highest priority job without removing."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def remove(self, job_id: str) -> bool:
        """Remove job from queue by ID."""
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    del self._queue[i]
                    return True
            return False

    def reorder(self, job_id: str, new_priority: JobPriority) -> bool:
        """Change priority of a queued job."""
        with self._lock:
            for job in self._queue:
                if job.id == job_id:
                    job.priority = new_priority
                    self._queue.sort(
                        key=lambda j: (-j.priority.value, j.created_at)
                    )
                    return True
            return False

    @property
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def list_jobs(self) -> List[BatchJob]:
        """Get list of all queued jobs."""
        with self._lock:
            return list(self._queue)

    def clear(self) -> int:
        """Clear all jobs from queue."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


class BatchQueueProcessor:
    """Manages batch processing of multiple restoration jobs.

    Features:
    - Priority-based job scheduling
    - Concurrent processing with configurable workers
    - Pause/resume individual jobs or entire queue
    - Progress tracking and callbacks
    - Automatic retry on failure
    - Job persistence for recovery
    """

    def __init__(
        self,
        max_workers: int = 1,
        persistence_path: Optional[Path] = None,
        on_job_start: Optional[Callable[[BatchJob], None]] = None,
        on_job_complete: Optional[Callable[[BatchJob], None]] = None,
        on_job_failed: Optional[Callable[[BatchJob, Exception], None]] = None,
        on_progress: Optional[Callable[[BatchJob, JobProgress], None]] = None,
    ):
        """Initialize batch processor.

        Args:
            max_workers: Maximum concurrent jobs (default 1 for GPU)
            persistence_path: Path to save queue state for recovery
            on_job_start: Callback when job starts
            on_job_complete: Callback when job completes
            on_job_failed: Callback when job fails
            on_progress: Callback for progress updates
        """
        self.max_workers = max_workers
        self.persistence_path = persistence_path

        # Callbacks
        self.on_job_start = on_job_start
        self.on_job_complete = on_job_complete
        self.on_job_failed = on_job_failed
        self.on_progress = on_progress

        # State
        self._queue = PriorityQueue()
        self._running_jobs: Dict[str, BatchJob] = {}
        self._completed_jobs: Dict[str, BatchJob] = {}
        self._failed_jobs: Dict[str, BatchJob] = {}

        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}

        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "jobs_completed": 0,
            "jobs_failed": 0,
            "total_processing_time": 0.0,
            "total_frames_processed": 0,
        }

        # Load persisted state if available
        if persistence_path and persistence_path.exists():
            self._load_state()

    def add_job(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        preset: str = "balanced",
        priority: JobPriority = JobPriority.NORMAL,
        config_overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Add a new job to the queue.

        Args:
            input_path: Input video path
            output_path: Output path (auto-generated if None)
            preset: Restoration preset name
            priority: Job priority
            config_overrides: Additional config options
            metadata: Custom metadata

        Returns:
            Created BatchJob instance
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_restored.mp4"

        job = BatchJob(
            input_path=str(input_path),
            output_path=str(output_path),
            preset=preset,
            priority=priority,
            config_overrides=config_overrides or {},
            metadata=metadata or {},
        )

        self._queue.put(job)
        self._save_state()

        logger.info(f"Job {job.id} added to queue: {input_path.name}")
        return job

    def add_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*.mp4",
        preset: str = "balanced",
        priority: JobPriority = JobPriority.NORMAL,
        recursive: bool = False,
    ) -> List[BatchJob]:
        """Add all matching videos from a directory.

        Args:
            input_dir: Directory containing videos
            output_dir: Output directory (uses input_dir if None)
            pattern: Glob pattern for video files
            preset: Restoration preset
            priority: Priority for all jobs
            recursive: Search subdirectories

        Returns:
            List of created jobs
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir / "restored"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find matching files
        if recursive:
            files = list(input_dir.rglob(pattern))
        else:
            files = list(input_dir.glob(pattern))

        jobs = []
        for file_path in sorted(files):
            # Maintain directory structure in output
            relative = file_path.relative_to(input_dir)
            out_path = output_dir / relative.parent / f"{file_path.stem}_restored.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            job = self.add_job(
                input_path=file_path,
                output_path=out_path,
                preset=preset,
                priority=priority,
            )
            jobs.append(job)

        logger.info(f"Added {len(jobs)} jobs from {input_dir}")
        return jobs

    def start(self) -> None:
        """Start processing the queue."""
        if self._running:
            logger.warning("Queue processor already running")
            return

        self._running = True
        self._paused = False

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()

        logger.info(f"Batch processor started with {self.max_workers} workers")

    def stop(self, wait: bool = True) -> None:
        """Stop processing the queue.

        Args:
            wait: Wait for current jobs to complete
        """
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            if wait:
                self._worker_thread.join(timeout=30)

        if self._executor:
            self._executor.shutdown(wait=wait)

        self._save_state()
        logger.info("Batch processor stopped")

    def pause(self) -> None:
        """Pause queue processing (current jobs continue)."""
        self._paused = True
        logger.info("Queue processing paused")

    def resume(self) -> None:
        """Resume queue processing."""
        self._paused = False
        logger.info("Queue processing resumed")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        # Check if queued
        if self._queue.remove(job_id):
            logger.info(f"Job {job_id} removed from queue")
            return True

        # Check if running
        with self._lock:
            if job_id in self._running_jobs:
                job = self._running_jobs[job_id]
                job.status = JobStatus.CANCELLED
                # Future cancellation is limited, but we mark it
                if job_id in self._futures:
                    self._futures[job_id].cancel()
                logger.info(f"Job {job_id} cancelled")
                return True

        return False

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        # Check queue
        for job in self._queue.list_jobs():
            if job.id == job_id:
                return job

        # Check running
        with self._lock:
            if job_id in self._running_jobs:
                return self._running_jobs[job_id]

        # Check completed/failed
        if job_id in self._completed_jobs:
            return self._completed_jobs[job_id]
        if job_id in self._failed_jobs:
            return self._failed_jobs[job_id]

        return None

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
    ) -> List[BatchJob]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Filter by status (None for all)

        Returns:
            List of matching jobs
        """
        jobs = []

        # Queued jobs
        jobs.extend(self._queue.list_jobs())

        # Running jobs
        with self._lock:
            jobs.extend(self._running_jobs.values())

        # Completed/failed
        jobs.extend(self._completed_jobs.values())
        jobs.extend(self._failed_jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at)

    def reprioritize(self, job_id: str, new_priority: JobPriority) -> bool:
        """Change priority of a queued job.

        Args:
            job_id: Job ID
            new_priority: New priority level

        Returns:
            True if successful
        """
        return self._queue.reorder(job_id, new_priority)

    def retry_failed(self, job_id: Optional[str] = None) -> List[BatchJob]:
        """Retry failed jobs.

        Args:
            job_id: Specific job ID or None for all failed

        Returns:
            List of requeued jobs
        """
        requeued = []

        if job_id:
            if job_id in self._failed_jobs:
                job = self._failed_jobs.pop(job_id)
                job.status = JobStatus.PENDING
                job.error_message = None
                self._queue.put(job)
                requeued.append(job)
        else:
            for jid in list(self._failed_jobs.keys()):
                job = self._failed_jobs.pop(jid)
                job.status = JobStatus.PENDING
                job.error_message = None
                self._queue.put(job)
                requeued.append(job)

        logger.info(f"Requeued {len(requeued)} failed jobs")
        return requeued

    def clear_completed(self) -> int:
        """Clear completed jobs from history.

        Returns:
            Number of jobs cleared
        """
        count = len(self._completed_jobs)
        self._completed_jobs.clear()
        self._save_state()
        return count

    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue

            # Check if we can start more jobs
            with self._lock:
                if len(self._running_jobs) >= self.max_workers:
                    time.sleep(0.1)
                    continue

            # Get next job
            job = self._queue.get(timeout=0.5)
            if job is None:
                continue

            # Start job
            self._start_job(job)

    def _start_job(self, job: BatchJob) -> None:
        """Start processing a job."""
        with self._lock:
            self._running_jobs[job.id] = job

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        job.progress.started_at = job.started_at

        if self.on_job_start:
            self.on_job_start(job)

        logger.info(f"Starting job {job.id}: {Path(job.input_path).name}")

        # Submit to executor
        future = self._executor.submit(self._execute_job, job)
        self._futures[job.id] = future

        # Add completion callback
        future.add_done_callback(
            lambda f, j=job: self._on_job_done(j, f)
        )

    def _execute_job(self, job: BatchJob) -> Path:
        """Execute a restoration job.

        Args:
            job: Job to execute

        Returns:
            Path to output file
        """
        from ..config import Config
        from ..restorer import VideoRestorer

        # Build config
        config_dict = {
            "preset": job.preset,
            "project_dir": Path(job.input_path).parent / ".framewright_temp",
            **job.config_overrides,
        }
        config = Config(**config_dict)

        # Create progress callback
        def progress_callback(info: Any) -> None:
            if hasattr(info, "stage"):
                job.progress.stage = info.stage
            if hasattr(info, "frames_total"):
                job.progress.frames_total = info.frames_total
            if hasattr(info, "frames_completed"):
                job.progress.frames_processed = info.frames_completed
                if job.progress.frames_total > 0:
                    job.progress.stage_progress = (
                        info.frames_completed / job.progress.frames_total
                    )
            if hasattr(info, "fps"):
                job.progress.current_fps = info.fps
            if hasattr(info, "eta"):
                job.progress.eta_seconds = info.eta

            # Calculate elapsed time
            if job.progress.started_at:
                start = datetime.fromisoformat(job.progress.started_at)
                job.progress.elapsed_seconds = (datetime.now() - start).total_seconds()

            if self.on_progress:
                self.on_progress(job, job.progress)

        restorer = VideoRestorer(config, progress_callback=progress_callback)

        return restorer.restore_video(
            source=job.input_path,
            output_path=Path(job.output_path),
        )

    def _on_job_done(self, job: BatchJob, future: Future) -> None:
        """Handle job completion."""
        with self._lock:
            if job.id in self._running_jobs:
                del self._running_jobs[job.id]
            if job.id in self._futures:
                del self._futures[job.id]

        try:
            result_path = future.result()
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            job.result_path = str(result_path)
            job.progress.overall_progress = 1.0

            self._completed_jobs[job.id] = job
            self._stats["jobs_completed"] += 1
            self._stats["total_frames_processed"] += job.progress.frames_processed

            if job.progress.elapsed_seconds:
                self._stats["total_processing_time"] += job.progress.elapsed_seconds

            if self.on_job_complete:
                self.on_job_complete(job)

            logger.info(f"Job {job.id} completed: {result_path}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now().isoformat()
            job.error_message = str(e)

            self._failed_jobs[job.id] = job
            self._stats["jobs_failed"] += 1

            if self.on_job_failed:
                self.on_job_failed(job, e)

            logger.error(f"Job {job.id} failed: {e}")

        self._save_state()

    def _save_state(self) -> None:
        """Save queue state for recovery."""
        if not self.persistence_path:
            return

        try:
            state = {
                "queued": [j.to_dict() for j in self._queue.list_jobs()],
                "completed": [j.to_dict() for j in self._completed_jobs.values()],
                "failed": [j.to_dict() for j in self._failed_jobs.values()],
                "stats": self._stats,
            }

            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")

    def _load_state(self) -> None:
        """Load queue state from persistence file."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path) as f:
                state = json.load(f)

            # Restore queued jobs
            for job_data in state.get("queued", []):
                job = BatchJob.from_dict(job_data)
                self._queue.put(job)

            # Restore completed
            for job_data in state.get("completed", []):
                job = BatchJob.from_dict(job_data)
                self._completed_jobs[job.id] = job

            # Restore failed
            for job_data in state.get("failed", []):
                job = BatchJob.from_dict(job_data)
                self._failed_jobs[job.id] = job

            # Restore stats
            self._stats.update(state.get("stats", {}))

            logger.info(f"Restored {self._queue.size} queued jobs from {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "queued_jobs": self._queue.size,
            "running_jobs": len(self._running_jobs),
            "completed_jobs": len(self._completed_jobs),
            "failed_jobs": len(self._failed_jobs),
            "is_running": self._running,
            "is_paused": self._paused,
        }

    @property
    def is_idle(self) -> bool:
        """Check if processor is idle (no queued or running jobs)."""
        return self._queue.size == 0 and len(self._running_jobs) == 0


def create_batch_processor(
    max_workers: int = 1,
    persistence_dir: Optional[Path] = None,
) -> BatchQueueProcessor:
    """Create a batch queue processor.

    Args:
        max_workers: Maximum concurrent jobs
        persistence_dir: Directory for queue state persistence

    Returns:
        Configured BatchQueueProcessor
    """
    persistence_path = None
    if persistence_dir:
        persistence_path = Path(persistence_dir) / "batch_queue.json"

    return BatchQueueProcessor(
        max_workers=max_workers,
        persistence_path=persistence_path,
    )
