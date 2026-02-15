"""
Scheduled Processing - Queue jobs for specific times.

.. deprecated::
    DEPRECATED: Legacy scheduler. Use `framewright.engine.scheduler` for new code.

Allows scheduling restoration jobs to run at specific times,
during off-peak hours, or with resource constraints.
"""

import json
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import heapq


class JobStatus(Enum):
    """Status of a scheduled job."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ScheduleType(Enum):
    """Type of schedule."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    SPECIFIC_TIME = "specific_time"
    RECURRING = "recurring"
    RESOURCE_BASED = "resource_based"  # Run when resources available


@dataclass
class JobConstraints:
    """Resource constraints for job execution."""
    min_free_disk_gb: float = 50.0
    max_gpu_temp: float = 80.0
    min_battery_percent: Optional[float] = None  # For laptops
    require_ac_power: bool = False
    max_concurrent_jobs: int = 1
    allowed_hours: Optional[tuple] = None  # (start_hour, end_hour)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_free_disk_gb": self.min_free_disk_gb,
            "max_gpu_temp": self.max_gpu_temp,
            "min_battery_percent": self.min_battery_percent,
            "require_ac_power": self.require_ac_power,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "allowed_hours": self.allowed_hours
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "JobConstraints":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ScheduledJob:
    """A job scheduled for processing."""
    job_id: str
    name: str
    input_path: str
    output_path: str
    preset: str = "balanced"
    schedule_type: ScheduleType = ScheduleType.IMMEDIATE
    scheduled_time: Optional[datetime] = None
    constraints: JobConstraints = field(default_factory=JobConstraints)
    status: JobStatus = JobStatus.PENDING
    priority: int = 5  # 1-10, lower = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "ScheduledJob") -> bool:
        """For heap ordering - by scheduled time then priority."""
        self_time = self.scheduled_time or datetime.min
        other_time = other.scheduled_time or datetime.min
        if self_time == other_time:
            return self.priority < other.priority
        return self_time < other_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "preset": self.preset,
            "schedule_type": self.schedule_type.value,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "constraints": self.constraints.to_dict(),
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress": self.progress,
            "extra_args": self.extra_args
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ScheduledJob":
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            input_path=data["input_path"],
            output_path=data["output_path"],
            preset=data.get("preset", "balanced"),
            schedule_type=ScheduleType(data.get("schedule_type", "immediate")),
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]) if data.get("scheduled_time") else None,
            constraints=JobConstraints.from_dict(data.get("constraints", {})),
            status=JobStatus(data.get("status", "pending")),
            priority=data.get("priority", 5),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            progress=data.get("progress", 0.0),
            extra_args=data.get("extra_args", {})
        )


class JobScheduler:
    """
    Schedule and manage restoration jobs.

    Supports immediate execution, delayed start, specific times,
    and resource-based scheduling.
    """

    def __init__(
        self,
        queue_file: Optional[Path] = None,
        processor_callback: Optional[Callable[[ScheduledJob], bool]] = None
    ):
        """
        Initialize scheduler.

        Args:
            queue_file: Path to persist queue (optional)
            processor_callback: Function to call when job should run
        """
        self.queue_file = queue_file
        self.processor_callback = processor_callback

        self._jobs: Dict[str, ScheduledJob] = {}
        self._job_heap: List[ScheduledJob] = []
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._current_jobs: List[str] = []

        if queue_file and queue_file.exists():
            self._load_queue()

    def _load_queue(self) -> None:
        """Load queue from file."""
        try:
            with open(self.queue_file) as f:
                data = json.load(f)

            for job_data in data.get("jobs", []):
                job = ScheduledJob.from_dict(job_data)
                if job.status in [JobStatus.PENDING, JobStatus.SCHEDULED]:
                    self._jobs[job.job_id] = job
                    heapq.heappush(self._job_heap, job)
        except Exception as e:
            print(f"Error loading queue: {e}")

    def _save_queue(self) -> None:
        """Save queue to file."""
        if not self.queue_file:
            return

        try:
            data = {
                "jobs": [job.to_dict() for job in self._jobs.values()],
                "saved_at": datetime.now().isoformat()
            }

            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving queue: {e}")

    def add_job(
        self,
        input_path: str,
        output_path: str,
        name: Optional[str] = None,
        preset: str = "balanced",
        schedule_type: ScheduleType = ScheduleType.IMMEDIATE,
        scheduled_time: Optional[datetime] = None,
        delay_minutes: Optional[int] = None,
        constraints: Optional[JobConstraints] = None,
        priority: int = 5,
        **extra_args
    ) -> str:
        """
        Add a job to the queue.

        Args:
            input_path: Path to input video
            output_path: Path for output
            name: Job name (defaults to filename)
            preset: Processing preset
            schedule_type: When to run
            scheduled_time: Specific time (for SPECIFIC_TIME type)
            delay_minutes: Delay in minutes (for DELAYED type)
            constraints: Resource constraints
            priority: 1-10, lower = higher priority
            **extra_args: Additional processor arguments

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())[:8]

        if name is None:
            name = Path(input_path).stem

        # Calculate scheduled time
        if schedule_type == ScheduleType.DELAYED and delay_minutes:
            scheduled_time = datetime.now() + timedelta(minutes=delay_minutes)
        elif schedule_type == ScheduleType.IMMEDIATE:
            scheduled_time = datetime.now()

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            input_path=str(input_path),
            output_path=str(output_path),
            preset=preset,
            schedule_type=schedule_type,
            scheduled_time=scheduled_time,
            constraints=constraints or JobConstraints(),
            status=JobStatus.SCHEDULED,
            priority=priority,
            extra_args=extra_args
        )

        with self._lock:
            self._jobs[job_id] = job
            heapq.heappush(self._job_heap, job)
            self._save_queue()

        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status in [JobStatus.PENDING, JobStatus.SCHEDULED]:
                    job.status = JobStatus.CANCELLED
                    self._save_queue()
                    return True
        return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status == JobStatus.SCHEDULED:
                    job.status = JobStatus.PAUSED
                    self._save_queue()
                    return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status == JobStatus.PAUSED:
                    job.status = JobStatus.SCHEDULED
                    heapq.heappush(self._job_heap, job)
                    self._save_queue()
                    return True
        return False

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None
    ) -> List[ScheduledJob]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: (j.scheduled_time or datetime.max, j.priority))

    def get_next_job(self) -> Optional[ScheduledJob]:
        """Get the next job that should run."""
        now = datetime.now()

        with self._lock:
            while self._job_heap:
                job = self._job_heap[0]

                # Skip cancelled/completed jobs
                if job.status not in [JobStatus.SCHEDULED, JobStatus.PENDING]:
                    heapq.heappop(self._job_heap)
                    continue

                # Check if it's time
                if job.scheduled_time and job.scheduled_time > now:
                    return None  # Not time yet

                # Check constraints
                if not self._check_constraints(job):
                    return None  # Constraints not met

                return job

        return None

    def _check_constraints(self, job: ScheduledJob) -> bool:
        """Check if job constraints are satisfied."""
        constraints = job.constraints

        # Check concurrent job limit
        if len(self._current_jobs) >= constraints.max_concurrent_jobs:
            return False

        # Check allowed hours
        if constraints.allowed_hours:
            current_hour = datetime.now().hour
            start_hour, end_hour = constraints.allowed_hours
            if not (start_hour <= current_hour < end_hour):
                return False

        # Check disk space
        try:
            import shutil
            output_dir = Path(job.output_path).parent
            if output_dir.exists():
                free_gb = shutil.disk_usage(output_dir).free / (1024**3)
                if free_gb < constraints.min_free_disk_gb:
                    return False
        except Exception:
            pass

        # Check GPU temperature
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            if temp > constraints.max_gpu_temp:
                return False
        except Exception:
            pass

        # Check battery (if applicable)
        if constraints.min_battery_percent or constraints.require_ac_power:
            try:
                import psutil
                battery = psutil.sensors_battery()
                if battery:
                    if constraints.require_ac_power and not battery.power_plugged:
                        return False
                    if constraints.min_battery_percent and battery.percent < constraints.min_battery_percent:
                        return False
            except Exception:
                pass

        return True

    def _run_job(self, job: ScheduledJob) -> bool:
        """Execute a job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self._current_jobs.append(job.job_id)
        self._save_queue()

        try:
            if self.processor_callback:
                success = self.processor_callback(job)
            else:
                # No processor - just mark complete
                success = True

            if success:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
            else:
                job.status = JobStatus.FAILED
                job.error_message = "Processor returned failure"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            success = False

        finally:
            job.completed_at = datetime.now()
            self._current_jobs.remove(job.job_id)
            self._save_queue()

        return success

    def start(self) -> None:
        """Start the scheduler background thread."""
        if self._running:
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            job = self.get_next_job()

            if job:
                # Remove from heap
                with self._lock:
                    if self._job_heap and self._job_heap[0] == job:
                        heapq.heappop(self._job_heap)

                self._run_job(job)
            else:
                # Wait before checking again
                time.sleep(10)

    def run_next(self) -> Optional[ScheduledJob]:
        """
        Run the next available job immediately (synchronous).

        Returns the job if one was run, None if no jobs available.
        """
        job = self.get_next_job()

        if job:
            with self._lock:
                if self._job_heap and self._job_heap[0] == job:
                    heapq.heappop(self._job_heap)

            self._run_job(job)
            return job

        return None

    def clear_completed(self) -> int:
        """Remove completed and failed jobs from the queue."""
        removed = 0
        with self._lock:
            to_remove = [
                jid for jid, job in self._jobs.items()
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            ]
            for jid in to_remove:
                del self._jobs[jid]
                removed += 1
            self._save_queue()
        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            "total_jobs": len(self._jobs),
            "by_status": {},
            "running_jobs": len(self._current_jobs),
            "next_scheduled": None
        }

        for status in JobStatus:
            count = sum(1 for j in self._jobs.values() if j.status == status)
            if count > 0:
                stats["by_status"][status.value] = count

        # Find next scheduled
        for job in sorted(self._jobs.values(), key=lambda j: j.scheduled_time or datetime.max):
            if job.status == JobStatus.SCHEDULED and job.scheduled_time:
                stats["next_scheduled"] = {
                    "job_id": job.job_id,
                    "name": job.name,
                    "scheduled_for": job.scheduled_time.isoformat()
                }
                break

        return stats
