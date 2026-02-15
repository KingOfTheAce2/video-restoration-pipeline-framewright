"""Cron-like scheduling for FrameWright batch processing.

.. deprecated::
    DEPRECATED: Legacy batch scheduler. Use `framewright.engine.scheduler` for new code.

Provides scheduled job execution based on cron expressions with persistence
and automatic execution at specified times.
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Represents a scheduled restoration job.

    Attributes:
        id: Unique job identifier (UUID)
        cron_expression: Cron expression in "minute hour day month weekday" format
        video_path: Path to the video file to process
        config: Configuration dictionary for the restoration
        enabled: Whether the job is active
        last_run: ISO timestamp of last execution
        next_run: ISO timestamp of next scheduled execution
        created_at: ISO timestamp of job creation
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cron_expression: str = "0 2 * * *"
    video_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledJob":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ScheduleConfig:
    """Configuration for the job scheduler.

    Attributes:
        jobs_file: Path to the JSON file storing scheduled jobs
        check_interval: Seconds between schedule checks (default 60)
        max_concurrent: Maximum concurrent jobs to run (default 1)
    """

    jobs_file: Path = field(default_factory=lambda: Path.home() / ".framewright" / "scheduled_jobs.json")
    check_interval: int = 60
    max_concurrent: int = 1

    def __post_init__(self):
        """Ensure jobs_file is a Path object."""
        if isinstance(self.jobs_file, str):
            self.jobs_file = Path(self.jobs_file)


class CronParser:
    """Simple cron expression parser supporting a subset of cron syntax.

    Supported format: "minute hour day month weekday"
    - minute: 0-59
    - hour: 0-23
    - day: 1-31
    - month: 1-12
    - weekday: 0-6 (0 = Sunday)

    Special characters:
    - *: matches any value

    Examples:
    - "0 2 * * *": 2:00 AM daily
    - "30 4 * * 0": 4:30 AM every Sunday
    - "0 0 1 * *": Midnight on the 1st of every month
    """

    FIELD_NAMES = ["minute", "hour", "day", "month", "weekday"]
    FIELD_RANGES = {
        "minute": (0, 59),
        "hour": (0, 23),
        "day": (1, 31),
        "month": (1, 12),
        "weekday": (0, 6),
    }

    @classmethod
    def parse(cls, expression: str) -> Dict[str, Any]:
        """Parse a cron expression into its components.

        Args:
            expression: Cron expression string (e.g., "0 2 * * *")

        Returns:
            Dictionary with keys: minute, hour, day, month, weekday
            Values are either integers or "*" for wildcards

        Raises:
            ValueError: If the expression is invalid
        """
        parts = expression.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: expected 5 fields, got {len(parts)}. "
                f"Format: 'minute hour day month weekday'"
            )

        result = {}
        for i, (name, value) in enumerate(zip(cls.FIELD_NAMES, parts)):
            if value == "*":
                result[name] = "*"
            else:
                try:
                    num = int(value)
                    min_val, max_val = cls.FIELD_RANGES[name]
                    if num < min_val or num > max_val:
                        raise ValueError(
                            f"Invalid {name} value: {num}. "
                            f"Must be between {min_val} and {max_val}"
                        )
                    result[name] = num
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(
                            f"Invalid {name} value: '{value}'. Must be an integer or '*'"
                        )
                    raise

        return result

    @classmethod
    def get_next_run(cls, expression: str, after: datetime) -> datetime:
        """Calculate the next run time after a given datetime.

        Args:
            expression: Cron expression string
            after: Datetime to calculate next run from

        Returns:
            Next datetime when the cron expression matches
        """
        parsed = cls.parse(expression)

        # Start from the next minute
        current = after.replace(second=0, microsecond=0)
        current = current.replace(minute=current.minute + 1) if current.minute < 59 else \
            current.replace(hour=current.hour + 1, minute=0)

        # Search for the next matching time (max 2 years to prevent infinite loop)
        max_iterations = 525600 * 2  # 2 years of minutes
        for _ in range(max_iterations):
            if cls._matches(parsed, current):
                return current

            # Increment by one minute
            if current.minute < 59:
                current = current.replace(minute=current.minute + 1)
            else:
                current = current.replace(minute=0)
                if current.hour < 23:
                    current = current.replace(hour=current.hour + 1)
                else:
                    current = current.replace(hour=0)
                    # Move to next day
                    from datetime import timedelta
                    current = current + timedelta(days=1)

        raise ValueError(f"Could not find next run time for expression: {expression}")

    @classmethod
    def matches_now(cls, expression: str) -> bool:
        """Check if a cron expression matches the current time.

        Args:
            expression: Cron expression string

        Returns:
            True if the expression matches the current minute
        """
        now = datetime.now().replace(second=0, microsecond=0)
        parsed = cls.parse(expression)
        return cls._matches(parsed, now)

    @classmethod
    def _matches(cls, parsed: Dict[str, Any], dt: datetime) -> bool:
        """Check if a parsed cron expression matches a datetime.

        Args:
            parsed: Parsed cron expression dictionary
            dt: Datetime to check

        Returns:
            True if all fields match
        """
        # Get datetime components
        # Note: weekday() returns 0=Monday, but cron uses 0=Sunday
        cron_weekday = (dt.weekday() + 1) % 7

        checks = [
            (parsed["minute"], dt.minute),
            (parsed["hour"], dt.hour),
            (parsed["day"], dt.day),
            (parsed["month"], dt.month),
            (parsed["weekday"], cron_weekday),
        ]

        for cron_val, dt_val in checks:
            if cron_val != "*" and cron_val != dt_val:
                return False

        return True


class JobScheduler:
    """Manages scheduled restoration jobs with cron-like scheduling.

    Features:
    - Add, remove, enable/disable scheduled jobs
    - Persistent storage of job schedules
    - Background checking for due jobs
    - Integration with BatchQueueProcessor for execution
    """

    def __init__(self, config: ScheduleConfig):
        """Initialize the job scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self._jobs: Dict[str, ScheduledJob] = {}
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._on_job_due: Optional[callable] = None

        # Load existing jobs
        self._load_jobs()

    def add_job(
        self,
        cron: str,
        video_path: Path,
        preset: str = "quality",
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> ScheduledJob:
        """Add a new scheduled job.

        Args:
            cron: Cron expression (e.g., "0 2 * * *" for 2 AM daily)
            video_path: Path to the video file
            preset: Restoration preset name
            config_overrides: Additional configuration options

        Returns:
            The created ScheduledJob instance

        Raises:
            ValueError: If the cron expression is invalid
        """
        # Validate cron expression
        CronParser.parse(cron)

        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning(f"Video path does not exist: {video_path}")

        job_config = {
            "preset": preset,
            **(config_overrides or {}),
        }

        job = ScheduledJob(
            cron_expression=cron,
            video_path=str(video_path),
            config=job_config,
            enabled=True,
        )

        # Calculate next run time
        job.next_run = CronParser.get_next_run(cron, datetime.now()).isoformat()

        with self._lock:
            self._jobs[job.id] = job

        self._save_jobs()
        logger.info(f"Added scheduled job {job.id}: {video_path.name} at '{cron}'")

        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job.

        Args:
            job_id: ID of the job to remove

        Returns:
            True if the job was removed, False if not found
        """
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                self._save_jobs()
                logger.info(f"Removed scheduled job {job_id}")
                return True
            return False

    def list_jobs(self) -> List[ScheduledJob]:
        """Get all scheduled jobs.

        Returns:
            List of all scheduled jobs sorted by next run time
        """
        with self._lock:
            jobs = list(self._jobs.values())

        # Sort by next_run (None values at the end)
        return sorted(
            jobs,
            key=lambda j: j.next_run if j.next_run else "9999",
        )

    def enable_job(self, job_id: str) -> bool:
        """Enable a scheduled job.

        Args:
            job_id: ID of the job to enable

        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.enabled = True
                # Recalculate next run
                job.next_run = CronParser.get_next_run(
                    job.cron_expression, datetime.now()
                ).isoformat()
                self._save_jobs()
                logger.info(f"Enabled scheduled job {job_id}")
                return True
            return False

    def disable_job(self, job_id: str) -> bool:
        """Disable a scheduled job.

        Args:
            job_id: ID of the job to disable

        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = False
                self._save_jobs()
                logger.info(f"Disabled scheduled job {job_id}")
                return True
            return False

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID.

        Args:
            job_id: Job ID to look up

        Returns:
            ScheduledJob or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)

    def get_next_jobs(self, count: int = 5) -> List[ScheduledJob]:
        """Get the next jobs scheduled to run.

        Args:
            count: Maximum number of jobs to return

        Returns:
            List of jobs sorted by next run time, limited to count
        """
        jobs = [j for j in self.list_jobs() if j.enabled and j.next_run]
        return jobs[:count]

    def _check_schedule(self) -> List[ScheduledJob]:
        """Check for jobs that are due to run now.

        Returns:
            List of jobs that should be executed
        """
        due_jobs = []
        now = datetime.now()

        with self._lock:
            for job in self._jobs.values():
                if not job.enabled:
                    continue

                if job.next_run:
                    next_run = datetime.fromisoformat(job.next_run)
                    if next_run <= now:
                        due_jobs.append(job)

        return due_jobs

    def _update_job_after_run(self, job: ScheduledJob) -> None:
        """Update job state after execution.

        Args:
            job: The job that was executed
        """
        with self._lock:
            if job.id in self._jobs:
                now = datetime.now()
                self._jobs[job.id].last_run = now.isoformat()
                self._jobs[job.id].next_run = CronParser.get_next_run(
                    job.cron_expression, now
                ).isoformat()

        self._save_jobs()

    def start(self, on_job_due: Optional[callable] = None) -> None:
        """Start the background scheduler.

        Args:
            on_job_due: Callback function called with ScheduledJob when a job is due
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._on_job_due = on_job_due
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="FramewrightScheduler",
        )
        self._scheduler_thread.start()
        logger.info("Job scheduler started")

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=self.config.check_interval + 5)
        logger.info("Job scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Background loop that checks for due jobs."""
        while self._running:
            try:
                due_jobs = self._check_schedule()

                for job in due_jobs:
                    logger.info(f"Job {job.id} is due for execution")

                    if self._on_job_due:
                        try:
                            self._on_job_due(job)
                        except Exception as e:
                            logger.error(f"Error in job due callback for {job.id}: {e}")

                    # Update job state
                    self._update_job_after_run(job)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

            # Sleep in small increments to allow for clean shutdown
            for _ in range(self.config.check_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _save_jobs(self) -> None:
        """Save jobs to persistent storage."""
        try:
            self.config.jobs_file.parent.mkdir(parents=True, exist_ok=True)

            jobs_data = {
                job_id: job.to_dict()
                for job_id, job in self._jobs.items()
            }

            with open(self.config.jobs_file, "w") as f:
                json.dump(jobs_data, f, indent=2)

            logger.debug(f"Saved {len(jobs_data)} scheduled jobs")

        except Exception as e:
            logger.error(f"Failed to save scheduled jobs: {e}")

    def _load_jobs(self) -> None:
        """Load jobs from persistent storage."""
        if not self.config.jobs_file.exists():
            logger.debug("No existing scheduled jobs file found")
            return

        try:
            with open(self.config.jobs_file) as f:
                jobs_data = json.load(f)

            for job_id, job_dict in jobs_data.items():
                self._jobs[job_id] = ScheduledJob.from_dict(job_dict)

            # Update next_run times for any jobs where next_run has passed
            now = datetime.now()
            for job in self._jobs.values():
                if job.enabled and job.next_run:
                    next_run = datetime.fromisoformat(job.next_run)
                    if next_run < now:
                        job.next_run = CronParser.get_next_run(
                            job.cron_expression, now
                        ).isoformat()

            self._save_jobs()
            logger.info(f"Loaded {len(self._jobs)} scheduled jobs")

        except Exception as e:
            logger.error(f"Failed to load scheduled jobs: {e}")


# Default scheduler instance
_default_scheduler: Optional[JobScheduler] = None
_default_lock = threading.Lock()


def create_scheduler(
    jobs_file: Optional[Path] = None,
    check_interval: int = 60,
    max_concurrent: int = 1,
) -> JobScheduler:
    """Create a new job scheduler instance.

    Args:
        jobs_file: Path for storing job schedules
        check_interval: Seconds between schedule checks
        max_concurrent: Maximum concurrent jobs

    Returns:
        Configured JobScheduler instance
    """
    config = ScheduleConfig(
        jobs_file=jobs_file or Path.home() / ".framewright" / "scheduled_jobs.json",
        check_interval=check_interval,
        max_concurrent=max_concurrent,
    )
    return JobScheduler(config)


def get_default_scheduler() -> JobScheduler:
    """Get or create the default scheduler instance.

    Returns:
        The default JobScheduler instance
    """
    global _default_scheduler

    with _default_lock:
        if _default_scheduler is None:
            _default_scheduler = create_scheduler()
        return _default_scheduler


def schedule_job(
    cron: str,
    video: Path,
    preset: str = "quality",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Schedule a video restoration job using the default scheduler.

    This is a convenience function for quick scheduling without
    manually managing a scheduler instance.

    Args:
        cron: Cron expression (e.g., "0 2 * * *" for 2 AM daily)
        video: Path to the video file
        preset: Restoration preset name
        config_overrides: Additional configuration options

    Returns:
        The ID of the created scheduled job

    Examples:
        # Schedule daily at 2 AM
        job_id = schedule_job("0 2 * * *", Path("/videos/old_movie.mp4"))

        # Schedule every Sunday at 4:30 AM with high quality preset
        job_id = schedule_job("30 4 * * 0", Path("/videos/archive.mp4"), preset="quality")
    """
    scheduler = get_default_scheduler()
    job = scheduler.add_job(
        cron=cron,
        video_path=video,
        preset=preset,
        config_overrides=config_overrides,
    )
    return job.id


__all__ = [
    "ScheduledJob",
    "ScheduleConfig",
    "CronParser",
    "JobScheduler",
    "create_scheduler",
    "get_default_scheduler",
    "schedule_job",
]
