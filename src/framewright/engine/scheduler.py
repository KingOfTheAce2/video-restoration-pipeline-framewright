"""Job scheduler for video restoration processing.

This module provides job scheduling and queue management for video
restoration tasks. It extends the basic scheduling in utils/scheduler.py
with features specific to the engine module:

- JobScheduler: Main scheduler with priority queue and persistence
- Job: Dataclass representing a restoration job
- JobStatus: Job lifecycle states
- Background worker threads
- Job persistence for restart recovery

Example:
    >>> scheduler = JobScheduler(max_concurrent=2, persist_dir=Path("./jobs"))
    >>> job_id = scheduler.submit(
    ...     Job(input="video.mp4", output="restored.mp4", config={"preset": "quality"})
    ... )
    >>> scheduler.get_status(job_id)
    JobStatus.RUNNING
    >>> scheduler.cancel(job_id)
    True
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class JobStatus(Enum):
    """Status of a restoration job."""

    QUEUED = "queued"
    """Job is waiting in the queue."""

    SCHEDULED = "scheduled"
    """Job is scheduled for a specific time."""

    RUNNING = "running"
    """Job is currently being processed."""

    COMPLETED = "completed"
    """Job completed successfully."""

    FAILED = "failed"
    """Job failed with an error."""

    CANCELLED = "cancelled"
    """Job was cancelled by user."""

    PAUSED = "paused"
    """Job is paused and can be resumed."""


class JobPriority(Enum):
    """Priority levels for jobs."""

    CRITICAL = 1
    """Highest priority - process immediately."""

    HIGH = 2
    """High priority - process before normal jobs."""

    NORMAL = 3
    """Normal priority - default level."""

    LOW = 4
    """Low priority - process when resources available."""

    BACKGROUND = 5
    """Lowest priority - process during idle time."""


# =============================================================================
# Job Data Classes
# =============================================================================


@dataclass
class JobConfig:
    """Configuration for a restoration job.

    Attributes:
        preset: Processing preset name.
        scale: Upscaling factor.
        denoise_strength: Denoising strength (0.0-1.0).
        face_restore: Enable face restoration.
        interpolate: Enable frame interpolation.
        target_fps: Target FPS for interpolation.
        audio_enhance: Enable audio enhancement.
        extra_params: Additional processor parameters.
    """

    preset: str = "balanced"
    scale: int = 4
    denoise_strength: float = 0.3
    face_restore: bool = False
    interpolate: bool = False
    target_fps: float = 60.0
    audio_enhance: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class JobProgress:
    """Progress information for a running job.

    Attributes:
        stage: Current processing stage name.
        stage_progress: Progress within current stage (0.0-1.0).
        overall_progress: Overall job progress (0.0-1.0).
        frames_processed: Number of frames processed.
        total_frames: Total frames to process.
        eta_seconds: Estimated time remaining.
        current_fps: Current processing speed.
    """

    stage: str = ""
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    eta_seconds: Optional[float] = None
    current_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Job:
    """A restoration job to be processed.

    Attributes:
        id: Unique job identifier.
        input: Input video path.
        output: Output video path.
        config: Job configuration.
        status: Current job status.
        priority: Job priority level.
        progress: Current progress information.
        created_at: When the job was created.
        started_at: When processing started.
        completed_at: When processing completed.
        scheduled_for: Scheduled start time (optional).
        error_message: Error message if failed.
        retry_count: Number of retry attempts.
        max_retries: Maximum retry attempts allowed.
        tags: User-defined tags for categorization.
        metadata: Additional job metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input: str = ""
    output: str = ""
    config: JobConfig = field(default_factory=JobConfig)
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL
    progress: JobProgress = field(default_factory=JobProgress)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Job") -> bool:
        """Compare jobs for priority queue ordering.

        Jobs are ordered by:
        1. Priority (lower value = higher priority)
        2. Scheduled time (earlier = higher priority)
        3. Creation time (earlier = higher priority)
        """
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        self_time = self.scheduled_for or self.created_at
        other_time = other.scheduled_for or other.created_at
        return self_time < other_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create job from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            input=data.get("input", ""),
            output=data.get("output", ""),
            config=JobConfig.from_dict(data.get("config", {})),
            status=JobStatus(data.get("status", "queued")),
            priority=JobPriority(data.get("priority", 3)),
            progress=JobProgress(**data.get("progress", {})),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]) if data.get("scheduled_for") else None,
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def can_start(self) -> bool:
        """Check if job can be started."""
        if self.status not in (JobStatus.QUEUED, JobStatus.SCHEDULED):
            return False
        if self.scheduled_for and self.scheduled_for > datetime.now():
            return False
        return True

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


@dataclass
class JobFilter:
    """Filter criteria for listing jobs.

    Attributes:
        status: Filter by status (or list of statuses).
        priority: Filter by priority (or list of priorities).
        tags: Filter by tags (job must have all tags).
        created_after: Filter by creation date.
        created_before: Filter by creation date.
        input_contains: Filter by input path substring.
    """

    status: Optional[Union[JobStatus, List[JobStatus]]] = None
    priority: Optional[Union[JobPriority, List[JobPriority]]] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    input_contains: Optional[str] = None

    def matches(self, job: Job) -> bool:
        """Check if a job matches the filter criteria."""
        if self.status is not None:
            statuses = [self.status] if isinstance(self.status, JobStatus) else self.status
            if job.status not in statuses:
                return False

        if self.priority is not None:
            priorities = [self.priority] if isinstance(self.priority, JobPriority) else self.priority
            if job.priority not in priorities:
                return False

        if self.tags:
            if not all(tag in job.tags for tag in self.tags):
                return False

        if self.created_after and job.created_at < self.created_after:
            return False

        if self.created_before and job.created_at > self.created_before:
            return False

        if self.input_contains and self.input_contains not in job.input:
            return False

        return True


# =============================================================================
# Event System
# =============================================================================


class JobEventType(Enum):
    """Types of job events."""

    JOB_SUBMITTED = auto()
    JOB_STARTED = auto()
    JOB_COMPLETED = auto()
    JOB_FAILED = auto()
    JOB_CANCELLED = auto()
    JOB_PAUSED = auto()
    JOB_RESUMED = auto()
    JOB_PROGRESS = auto()
    JOB_RETRYING = auto()
    SCHEDULER_STARTED = auto()
    SCHEDULER_STOPPED = auto()
    WORKER_IDLE = auto()


@dataclass
class JobEvent:
    """Event emitted by the scheduler.

    Attributes:
        event_type: Type of event.
        job_id: Associated job ID (if applicable).
        timestamp: When the event occurred.
        data: Additional event data.
    """

    event_type: JobEventType
    job_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


JobEventCallback = Callable[[JobEvent], None]


# =============================================================================
# Job Scheduler
# =============================================================================


class JobScheduler:
    """Scheduler for restoration jobs with priority queue support.

    The JobScheduler manages a queue of restoration jobs, handling:
    - Priority-based scheduling
    - Concurrent job execution
    - Job persistence for restart recovery
    - Event notification
    - Graceful shutdown

    Example:
        >>> scheduler = JobScheduler(
        ...     max_concurrent=2,
        ...     persist_dir=Path("./jobs"),
        ...     auto_start=True,
        ... )
        >>> job = Job(input="video.mp4", output="restored.mp4")
        >>> job_id = scheduler.submit(job)
        >>> print(scheduler.get_status(job_id))
        JobStatus.QUEUED
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        persist_dir: Optional[Path] = None,
        auto_start: bool = False,
        processor_callback: Optional[Callable[[Job], bool]] = None,
    ) -> None:
        """Initialize the job scheduler.

        Args:
            max_concurrent: Maximum concurrent jobs.
            persist_dir: Directory for job persistence (optional).
            auto_start: Start worker threads automatically.
            processor_callback: Function to process jobs.
        """
        self.max_concurrent = max_concurrent
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.processor_callback = processor_callback

        # Job storage
        self._jobs: Dict[str, Job] = {}
        self._job_queue: List[Job] = []  # Priority queue (heapq)
        self._running_jobs: Set[str] = set()

        # Thread synchronization
        self._lock = threading.Lock()
        self._queue_condition = threading.Condition(self._lock)
        self._shutdown_event = threading.Event()

        # Workers
        self._workers: List[threading.Thread] = []
        self._running = False

        # Event callbacks
        self._event_callbacks: List[JobEventCallback] = []

        # Load persisted jobs
        if self.persist_dir:
            self._load_persisted_jobs()

        # Auto-start if requested
        if auto_start:
            self.start()

    # -------------------------------------------------------------------------
    # Job Submission and Management
    # -------------------------------------------------------------------------

    def submit(self, job: Job) -> str:
        """Submit a job to the queue.

        Args:
            job: Job to submit.

        Returns:
            Job ID.

        Raises:
            ValueError: If job with same ID already exists.
        """
        with self._lock:
            if job.id in self._jobs:
                raise ValueError(f"Job with ID '{job.id}' already exists")

            # Ensure job is in queued state
            if job.status == JobStatus.QUEUED and job.scheduled_for:
                job.status = JobStatus.SCHEDULED

            self._jobs[job.id] = job
            heapq.heappush(self._job_queue, job)

            # Persist job
            self._persist_job(job)

            # Notify workers
            self._queue_condition.notify()

        logger.info(f"Job submitted: {job.id} ({job.input})")

        self._emit_event(JobEventType.JOB_SUBMITTED, job.id, {
            "input": job.input,
            "output": job.output,
            "priority": job.priority.name,
        })

        return job.id

    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: ID of job to cancel.

        Returns:
            True if job was cancelled.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.is_terminal:
                return False

            # Can cancel queued/scheduled/paused jobs immediately
            if job.status in (JobStatus.QUEUED, JobStatus.SCHEDULED, JobStatus.PAUSED):
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self._persist_job(job)
                self._emit_event(JobEventType.JOB_CANCELLED, job_id)
                return True

            # For running jobs, set cancellation flag
            if job.status == JobStatus.RUNNING:
                job.metadata["cancel_requested"] = True
                return True

        return False

    def pause(self, job_id: str) -> bool:
        """Pause a queued or scheduled job.

        Args:
            job_id: ID of job to pause.

        Returns:
            True if job was paused.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status not in (JobStatus.QUEUED, JobStatus.SCHEDULED):
                return False

            job.status = JobStatus.PAUSED
            self._persist_job(job)
            self._emit_event(JobEventType.JOB_PAUSED, job_id)
            return True

    def resume(self, job_id: str) -> bool:
        """Resume a paused job.

        Args:
            job_id: ID of job to resume.

        Returns:
            True if job was resumed.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status != JobStatus.PAUSED:
                return False

            job.status = JobStatus.QUEUED
            heapq.heappush(self._job_queue, job)
            self._persist_job(job)
            self._queue_condition.notify()
            self._emit_event(JobEventType.JOB_RESUMED, job_id)
            return True

    def retry(self, job_id: str) -> bool:
        """Retry a failed job.

        Args:
            job_id: ID of job to retry.

        Returns:
            True if job was requeued for retry.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status != JobStatus.FAILED:
                return False

            if job.retry_count >= job.max_retries:
                return False

            job.status = JobStatus.QUEUED
            job.retry_count += 1
            job.error_message = None
            job.started_at = None
            job.completed_at = None
            heapq.heappush(self._job_queue, job)
            self._persist_job(job)
            self._queue_condition.notify()
            self._emit_event(JobEventType.JOB_RETRYING, job_id, {
                "retry_count": job.retry_count,
            })
            return True

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a job.

        Args:
            job_id: Job ID.

        Returns:
            JobStatus or None if job not found.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return job.status if job else None

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID.

        Args:
            job_id: Job ID.

        Returns:
            Job or None if not found.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get progress information for a job.

        Args:
            job_id: Job ID.

        Returns:
            JobProgress or None if job not found.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return job.progress if job else None

    def update_progress(
        self,
        job_id: str,
        stage: Optional[str] = None,
        stage_progress: Optional[float] = None,
        overall_progress: Optional[float] = None,
        frames_processed: Optional[int] = None,
        total_frames: Optional[int] = None,
        eta_seconds: Optional[float] = None,
        current_fps: Optional[float] = None,
    ) -> bool:
        """Update progress for a running job.

        Args:
            job_id: Job ID.
            stage: Current stage name.
            stage_progress: Progress within stage (0.0-1.0).
            overall_progress: Overall progress (0.0-1.0).
            frames_processed: Frames processed so far.
            total_frames: Total frames to process.
            eta_seconds: Estimated time remaining.
            current_fps: Current processing speed.

        Returns:
            True if progress was updated.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.RUNNING:
                return False

            if stage is not None:
                job.progress.stage = stage
            if stage_progress is not None:
                job.progress.stage_progress = stage_progress
            if overall_progress is not None:
                job.progress.overall_progress = overall_progress
            if frames_processed is not None:
                job.progress.frames_processed = frames_processed
            if total_frames is not None:
                job.progress.total_frames = total_frames
            if eta_seconds is not None:
                job.progress.eta_seconds = eta_seconds
            if current_fps is not None:
                job.progress.current_fps = current_fps

            self._emit_event(JobEventType.JOB_PROGRESS, job_id, {
                "stage": job.progress.stage,
                "overall_progress": job.progress.overall_progress,
            })
            return True

    def list_jobs(
        self,
        filter: Optional[JobFilter] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional filtering.

        Args:
            filter: Optional filter criteria.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.

        Returns:
            List of matching jobs.
        """
        with self._lock:
            jobs = list(self._jobs.values())

        # Apply filter
        if filter:
            jobs = [j for j in jobs if filter.matches(j)]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply pagination
        if offset:
            jobs = jobs[offset:]
        if limit:
            jobs = jobs[:limit]

        return jobs

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler.

        Only completed, failed, or cancelled jobs can be removed.

        Args:
            job_id: Job ID.

        Returns:
            True if job was removed.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if not job.is_terminal:
                return False

            del self._jobs[job_id]
            self._remove_persisted_job(job_id)
            return True

    def clear_completed(self) -> int:
        """Remove all completed jobs.

        Returns:
            Number of jobs removed.
        """
        count = 0
        with self._lock:
            to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.status == JobStatus.COMPLETED
            ]
            for job_id in to_remove:
                del self._jobs[job_id]
                self._remove_persisted_job(job_id)
                count += 1
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            stats = {
                "total_jobs": len(self._jobs),
                "queued": 0,
                "scheduled": 0,
                "running": len(self._running_jobs),
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "paused": 0,
                "workers": len(self._workers),
                "max_concurrent": self.max_concurrent,
                "is_running": self._running,
            }

            for job in self._jobs.values():
                status_key = job.status.value
                if status_key in stats:
                    stats[status_key] += 1

            return stats

    # -------------------------------------------------------------------------
    # Worker Management
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler workers."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start worker threads
        for i in range(self.max_concurrent):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"JobScheduler-Worker-{i}",
                daemon=True,
            )
            self._workers.append(worker)
            worker.start()

        logger.info(f"JobScheduler started with {self.max_concurrent} workers")
        self._emit_event(JobEventType.SCHEDULER_STARTED, data={
            "workers": self.max_concurrent,
        })

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Stop the scheduler.

        Args:
            wait: Wait for running jobs to complete.
            timeout: Maximum time to wait for jobs.
        """
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        # Wake up waiting workers
        with self._queue_condition:
            self._queue_condition.notify_all()

        if wait:
            deadline = time.time() + timeout
            for worker in self._workers:
                remaining = deadline - time.time()
                if remaining > 0:
                    worker.join(timeout=remaining)

        self._workers.clear()
        logger.info("JobScheduler stopped")
        self._emit_event(JobEventType.SCHEDULER_STOPPED)

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            job = self._get_next_job()

            if job is None:
                # Wait for new jobs or shutdown
                with self._queue_condition:
                    self._queue_condition.wait(timeout=1.0)
                continue

            self._process_job(job)

    def _get_next_job(self) -> Optional[Job]:
        """Get the next job to process.

        Returns:
            Job to process or None if no jobs available.
        """
        with self._lock:
            # Check if we can run more jobs
            if len(self._running_jobs) >= self.max_concurrent:
                return None

            # Find next job that can start
            now = datetime.now()
            temp_queue = []

            while self._job_queue:
                job = heapq.heappop(self._job_queue)

                # Skip jobs that can't start
                if job.status not in (JobStatus.QUEUED, JobStatus.SCHEDULED):
                    continue

                # Check scheduled time
                if job.scheduled_for and job.scheduled_for > now:
                    temp_queue.append(job)
                    continue

                # Check for cancellation request
                if job.metadata.get("cancel_requested"):
                    job.status = JobStatus.CANCELLED
                    job.completed_at = now
                    self._persist_job(job)
                    self._emit_event(JobEventType.JOB_CANCELLED, job.id)
                    continue

                # Found a job to process
                job.status = JobStatus.RUNNING
                job.started_at = now
                self._running_jobs.add(job.id)

                # Restore skipped jobs
                for temp_job in temp_queue:
                    heapq.heappush(self._job_queue, temp_job)

                return job

            # Restore all skipped jobs
            for temp_job in temp_queue:
                heapq.heappush(self._job_queue, temp_job)

            return None

    def _process_job(self, job: Job) -> None:
        """Process a job.

        Args:
            job: Job to process.
        """
        logger.info(f"Starting job: {job.id} ({job.input})")
        self._emit_event(JobEventType.JOB_STARTED, job.id, {
            "input": job.input,
            "output": job.output,
        })

        try:
            if self.processor_callback:
                success = self.processor_callback(job)
            else:
                # No processor - simulate processing
                logger.warning(f"No processor callback for job {job.id}")
                success = True

            with self._lock:
                if success:
                    job.status = JobStatus.COMPLETED
                    job.progress.overall_progress = 1.0
                    logger.info(f"Job completed: {job.id}")
                    self._emit_event(JobEventType.JOB_COMPLETED, job.id, {
                        "duration_seconds": job.duration_seconds,
                    })
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = "Processor returned failure"
                    logger.error(f"Job failed: {job.id}")
                    self._emit_event(JobEventType.JOB_FAILED, job.id, {
                        "error": job.error_message,
                    })

        except Exception as e:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            logger.exception(f"Job {job.id} failed with exception")
            self._emit_event(JobEventType.JOB_FAILED, job.id, {
                "error": str(e),
            })

        finally:
            with self._lock:
                job.completed_at = datetime.now()
                self._running_jobs.discard(job.id)
                self._persist_job(job)

                # Auto-retry if configured
                if (
                    job.status == JobStatus.FAILED
                    and job.retry_count < job.max_retries
                ):
                    job.status = JobStatus.QUEUED
                    job.retry_count += 1
                    job.error_message = None
                    job.started_at = None
                    job.completed_at = None
                    heapq.heappush(self._job_queue, job)
                    self._emit_event(JobEventType.JOB_RETRYING, job.id, {
                        "retry_count": job.retry_count,
                    })

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def on_event(self, callback: JobEventCallback) -> Callable[[], None]:
        """Subscribe to scheduler events.

        Args:
            callback: Function to call when event occurs.

        Returns:
            Unsubscribe function.
        """
        self._event_callbacks.append(callback)
        return lambda: self._event_callbacks.remove(callback)

    def _emit_event(
        self,
        event_type: JobEventType,
        job_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an event to all subscribers.

        Args:
            event_type: Type of event.
            job_id: Associated job ID.
            data: Additional event data.
        """
        event = JobEvent(
            event_type=event_type,
            job_id=job_id,
            data=data or {},
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _get_job_file(self, job_id: str) -> Optional[Path]:
        """Get the persistence file path for a job."""
        if not self.persist_dir:
            return None
        return self.persist_dir / f"{job_id}.json"

    def _persist_job(self, job: Job) -> None:
        """Save a job to disk."""
        job_file = self._get_job_file(job.id)
        if not job_file:
            return

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            temp_file = job_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(job.to_dict(), f, indent=2)
            temp_file.replace(job_file)
        except Exception as e:
            logger.error(f"Failed to persist job {job.id}: {e}")

    def _remove_persisted_job(self, job_id: str) -> None:
        """Remove a persisted job file."""
        job_file = self._get_job_file(job_id)
        if job_file and job_file.exists():
            try:
                job_file.unlink()
            except Exception as e:
                logger.error(f"Failed to remove job file {job_id}: {e}")

    def _load_persisted_jobs(self) -> None:
        """Load jobs from persistence directory."""
        if not self.persist_dir or not self.persist_dir.exists():
            return

        loaded = 0
        for job_file in self.persist_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    data = json.load(f)
                job = Job.from_dict(data)

                # Re-queue non-terminal jobs
                if not job.is_terminal:
                    if job.status == JobStatus.RUNNING:
                        # Job was interrupted - re-queue
                        job.status = JobStatus.QUEUED
                        job.started_at = None

                    self._jobs[job.id] = job
                    heapq.heappush(self._job_queue, job)
                    loaded += 1
                else:
                    # Keep terminal jobs for history
                    self._jobs[job.id] = job

            except Exception as e:
                logger.error(f"Failed to load job from {job_file}: {e}")

        if loaded:
            logger.info(f"Loaded {loaded} pending jobs from persistence")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_scheduler(
    max_concurrent: int = 1,
    persist_dir: Optional[Union[str, Path]] = None,
    auto_start: bool = True,
) -> JobScheduler:
    """Create a job scheduler with common settings.

    Args:
        max_concurrent: Maximum concurrent jobs.
        persist_dir: Directory for job persistence.
        auto_start: Start scheduler automatically.

    Returns:
        Configured JobScheduler.
    """
    return JobScheduler(
        max_concurrent=max_concurrent,
        persist_dir=Path(persist_dir) if persist_dir else None,
        auto_start=auto_start,
    )


def create_job(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    preset: str = "balanced",
    priority: JobPriority = JobPriority.NORMAL,
    scheduled_for: Optional[datetime] = None,
    **kwargs: Any,
) -> Job:
    """Create a restoration job.

    Args:
        input_path: Input video path.
        output_path: Output video path.
        preset: Processing preset.
        priority: Job priority.
        scheduled_for: Optional scheduled start time.
        **kwargs: Additional config parameters.

    Returns:
        Configured Job.
    """
    config = JobConfig(preset=preset, **{
        k: v for k, v in kwargs.items()
        if k in JobConfig.__dataclass_fields__
    })

    return Job(
        input=str(input_path),
        output=str(output_path),
        config=config,
        priority=priority,
        scheduled_for=scheduled_for,
    )
