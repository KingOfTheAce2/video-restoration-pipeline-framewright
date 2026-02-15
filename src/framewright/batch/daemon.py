"""Daemon mode with auto-resume for FrameWright.

Provides a background daemon service that manages video restoration jobs
with automatic crash recovery, job persistence, and graceful shutdown.
"""
import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DaemonStatus(Enum):
    """Status of the daemon process."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class JobState(Enum):
    """State of an individual job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CRASHED = "crashed"  # Job was interrupted by crash


@dataclass
class DaemonConfig:
    """Configuration for the FrameWright daemon.

    Attributes:
        pid_file: Path for storing the daemon's process ID.
        work_dir: Working directory for daemon operations.
        log_file: Optional path for daemon-specific logs.
        auto_resume: Whether to automatically resume crashed jobs on startup.
        poll_interval: Seconds between job queue checks.
        max_concurrent_jobs: Maximum number of jobs to process simultaneously.
    """

    pid_file: Path
    work_dir: Path
    log_file: Optional[Path] = None
    auto_resume: bool = True
    poll_interval: float = 5.0
    max_concurrent_jobs: int = 1

    def __post_init__(self) -> None:
        """Convert paths and validate configuration."""
        if not isinstance(self.pid_file, Path):
            self.pid_file = Path(self.pid_file)
        if not isinstance(self.work_dir, Path):
            self.work_dir = Path(self.work_dir)
        if self.log_file is not None and not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)

        # Validate configuration
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "pid_file": str(self.pid_file),
            "work_dir": str(self.work_dir),
            "log_file": str(self.log_file) if self.log_file else None,
            "auto_resume": self.auto_resume,
            "poll_interval": self.poll_interval,
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DaemonConfig":
        """Create configuration from dictionary."""
        return cls(
            pid_file=Path(data["pid_file"]),
            work_dir=Path(data["work_dir"]),
            log_file=Path(data["log_file"]) if data.get("log_file") else None,
            auto_resume=data.get("auto_resume", True),
            poll_interval=data.get("poll_interval", 5.0),
            max_concurrent_jobs=data.get("max_concurrent_jobs", 1),
        )


@dataclass
class DaemonJob:
    """Represents a job managed by the daemon.

    Attributes:
        id: Unique job identifier.
        video_path: Path to the input video file.
        config: Job-specific configuration dictionary.
        state: Current state of the job.
        created_at: Timestamp when the job was created.
        started_at: Timestamp when processing started (if applicable).
        completed_at: Timestamp when processing completed (if applicable).
        output_path: Path to the output file (if completed).
        error_message: Error message (if failed).
        progress: Processing progress (0.0 to 1.0).
        retry_count: Number of retry attempts.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    video_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DaemonJob":
        """Create job from dictionary."""
        data = data.copy()
        data["state"] = JobState(data.get("state", "pending"))
        return cls(**data)


@dataclass
class DaemonState:
    """Tracks the overall state of the daemon.

    Attributes:
        status: Current daemon status.
        pid: Process ID of the running daemon.
        started_at: Timestamp when daemon started.
        jobs: Dictionary of all jobs (job_id -> DaemonJob).
        current_jobs: List of currently running job IDs.
        stats: Processing statistics.
    """

    status: DaemonStatus = DaemonStatus.STOPPED
    pid: Optional[int] = None
    started_at: Optional[str] = None
    jobs: Dict[str, DaemonJob] = field(default_factory=dict)
    current_jobs: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "jobs_completed": 0,
        "jobs_failed": 0,
        "total_processing_time": 0.0,
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for persistence."""
        return {
            "status": self.status.value,
            "pid": self.pid,
            "started_at": self.started_at,
            "jobs": {jid: job.to_dict() for jid, job in self.jobs.items()},
            "current_jobs": self.current_jobs,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DaemonState":
        """Create state from dictionary."""
        state = cls(
            status=DaemonStatus(data.get("status", "stopped")),
            pid=data.get("pid"),
            started_at=data.get("started_at"),
            current_jobs=data.get("current_jobs", []),
            stats=data.get("stats", {}),
        )
        # Restore jobs
        for jid, job_data in data.get("jobs", {}).items():
            state.jobs[jid] = DaemonJob.from_dict(job_data)
        return state


class FramewrightDaemon:
    """Background daemon for FrameWright video restoration.

    Provides a long-running service that manages video restoration jobs
    with automatic crash recovery, job persistence, and graceful shutdown.

    Features:
    - PID file management for single-instance enforcement
    - Signal handling for graceful shutdown (SIGTERM, SIGINT)
    - Automatic job recovery after crashes
    - Persistent job queue across restarts
    - Configurable concurrency

    Example:
        >>> config = DaemonConfig(
        ...     pid_file=Path("/var/run/framewright.pid"),
        ...     work_dir=Path("/var/lib/framewright"),
        ... )
        >>> daemon = FramewrightDaemon(config)
        >>> daemon.start()  # Starts the daemon
    """

    def __init__(
        self,
        config: DaemonConfig,
        on_job_start: Optional[Callable[[DaemonJob], None]] = None,
        on_job_complete: Optional[Callable[[DaemonJob], None]] = None,
        on_job_failed: Optional[Callable[[DaemonJob, Exception], None]] = None,
    ) -> None:
        """Initialize the daemon.

        Args:
            config: Daemon configuration.
            on_job_start: Callback when a job starts processing.
            on_job_complete: Callback when a job completes successfully.
            on_job_failed: Callback when a job fails.
        """
        self.config = config
        self.on_job_start = on_job_start
        self.on_job_complete = on_job_complete
        self.on_job_failed = on_job_failed

        self._state = DaemonState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        self._job_queue: List[str] = []  # Queue of pending job IDs
        self._queue_condition = threading.Condition(self._lock)

        # Paths for state persistence
        self._state_file = config.work_dir / "daemon_state.json"
        self._jobs_dir = config.work_dir / "jobs"

        # Original signal handlers for restoration
        self._original_sigterm: Optional[signal.Handlers] = None
        self._original_sigint: Optional[signal.Handlers] = None

    def _ensure_directories(self) -> None:
        """Create required directories."""
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_pid_file(self) -> None:
        """Write current process ID to PID file."""
        pid = os.getpid()
        self.config.pid_file.write_text(str(pid))
        logger.debug(f"Wrote PID {pid} to {self.config.pid_file}")

    def _remove_pid_file(self) -> None:
        """Remove the PID file."""
        try:
            if self.config.pid_file.exists():
                self.config.pid_file.unlink()
                logger.debug(f"Removed PID file {self.config.pid_file}")
        except OSError as e:
            logger.warning(f"Failed to remove PID file: {e}")

    def _read_pid_file(self) -> Optional[int]:
        """Read PID from file if it exists.

        Returns:
            Process ID or None if file doesn't exist.
        """
        try:
            if self.config.pid_file.exists():
                return int(self.config.pid_file.read_text().strip())
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read PID file: {e}")
        return None

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running.

        Args:
            pid: Process ID to check.

        Returns:
            True if process is running.
        """
        if sys.platform == "win32":
            # Windows process check
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        else:
            # Unix process check
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def _save_state(self) -> None:
        """Persist daemon state to disk."""
        try:
            state_data = self._state.to_dict()
            temp_file = self._state_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(state_data, indent=2))
            temp_file.replace(self._state_file)
            logger.debug("Saved daemon state")
        except Exception as e:
            logger.error(f"Failed to save daemon state: {e}")

    def _load_state(self) -> None:
        """Load daemon state from disk."""
        try:
            if self._state_file.exists():
                state_data = json.loads(self._state_file.read_text())
                self._state = DaemonState.from_dict(state_data)
                # Rebuild job queue from pending jobs
                self._job_queue = [
                    jid for jid, job in self._state.jobs.items()
                    if job.state in (JobState.PENDING, JobState.CRASHED)
                ]
                logger.info(f"Loaded daemon state: {len(self._state.jobs)} jobs")
        except Exception as e:
            logger.error(f"Failed to load daemon state: {e}")
            self._state = DaemonState()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        # Store original handlers
        if sys.platform != "win32":
            self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if sys.platform != "win32" and self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
        except Exception as e:
            logger.warning(f"Failed to restore signal handlers: {e}")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"Received signal {signal_name}, initiating graceful shutdown")
        self.stop()

    def _cleanup(self) -> None:
        """Cleanup resources on exit."""
        self._remove_pid_file()
        self._restore_signal_handlers()
        atexit.unregister(self._cleanup)

    def _recover_crashed_jobs(self) -> int:
        """Resume incomplete jobs from previous run.

        Jobs that were in RUNNING state when the daemon crashed are
        marked as CRASHED and re-queued if auto_resume is enabled.

        Returns:
            Number of jobs recovered.
        """
        recovered = 0
        with self._lock:
            # Find jobs that were running when daemon stopped
            for jid in list(self._state.current_jobs):
                if jid in self._state.jobs:
                    job = self._state.jobs[jid]
                    if job.state == JobState.RUNNING:
                        job.state = JobState.CRASHED
                        job.error_message = "Job interrupted by daemon crash"
                        if self.config.auto_resume:
                            job.state = JobState.PENDING
                            job.retry_count += 1
                            if jid not in self._job_queue:
                                self._job_queue.append(jid)
                            recovered += 1
                            logger.info(f"Recovered crashed job {jid}: {job.video_path}")

            # Clear current jobs list
            self._state.current_jobs.clear()

        if recovered > 0:
            self._save_state()
            logger.info(f"Recovered {recovered} crashed jobs")

        return recovered

    def start(self) -> None:
        """Start the daemon.

        Sets up signal handlers, writes PID file, recovers crashed jobs,
        and starts the main processing loop.

        Raises:
            RuntimeError: If daemon is already running.
        """
        # Check if already running
        existing_pid = self._read_pid_file()
        if existing_pid is not None and self._is_process_running(existing_pid):
            raise RuntimeError(
                f"Daemon already running with PID {existing_pid}. "
                f"Use stop() first or remove {self.config.pid_file}"
            )

        logger.info("Starting FrameWright daemon")

        # Initialize
        self._ensure_directories()
        self._load_state()
        self._setup_signal_handlers()
        self._write_pid_file()

        # Update state
        self._state.status = DaemonStatus.RUNNING
        self._state.pid = os.getpid()
        self._state.started_at = datetime.now().isoformat()
        self._save_state()

        # Recover crashed jobs
        if self.config.auto_resume:
            self._recover_crashed_jobs()

        # Start worker threads
        self._stop_event.clear()
        for i in range(self.config.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"framewright-daemon-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._worker_threads.append(worker)

        logger.info(
            f"Daemon started with PID {os.getpid()}, "
            f"{self.config.max_concurrent_jobs} workers"
        )

        # Run main loop
        self._run_loop()

    def stop(self) -> None:
        """Stop the daemon gracefully.

        Signals workers to stop, waits for current jobs to complete,
        and saves final state.
        """
        logger.info("Stopping FrameWright daemon")

        # Signal stop
        self._stop_event.set()

        # Wake up waiting workers
        with self._queue_condition:
            self._queue_condition.notify_all()

        # Wait for workers to finish
        for worker in self._worker_threads:
            if worker.is_alive():
                worker.join(timeout=30)

        # Update state
        with self._lock:
            self._state.status = DaemonStatus.STOPPED
            self._save_state()

        # Cleanup
        self._cleanup()

        logger.info("Daemon stopped")

    def status(self) -> DaemonStatus:
        """Get the current daemon status.

        Returns:
            Current DaemonStatus.
        """
        # Check if actually running
        pid = self._read_pid_file()
        if pid is None:
            return DaemonStatus.STOPPED
        if not self._is_process_running(pid):
            return DaemonStatus.STOPPED
        return self._state.status

    def add_job(self, video_path: Path, config: Dict[str, Any]) -> str:
        """Add a new job to the queue.

        Args:
            video_path: Path to the input video file.
            config: Job-specific configuration dictionary.

        Returns:
            Unique job ID.

        Raises:
            ValueError: If video file doesn't exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        job = DaemonJob(
            video_path=str(video_path.resolve()),
            config=config,
        )

        with self._queue_condition:
            self._state.jobs[job.id] = job
            self._job_queue.append(job.id)
            self._queue_condition.notify()

        self._save_state()
        logger.info(f"Added job {job.id}: {video_path.name}")

        return job.id

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get list of all jobs.

        Returns:
            List of job dictionaries with status information.
        """
        with self._lock:
            return [job.to_dict() for job in self._state.jobs.values()]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Job dictionary or None if not found.
        """
        with self._lock:
            job = self._state.jobs.get(job_id)
            return job.to_dict() if job else None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job.

        Args:
            job_id: Job identifier to cancel.

        Returns:
            True if job was cancelled.
        """
        with self._lock:
            if job_id not in self._state.jobs:
                return False

            job = self._state.jobs[job_id]
            if job.state != JobState.PENDING:
                return False

            job.state = JobState.FAILED
            job.error_message = "Cancelled by user"
            job.completed_at = datetime.now().isoformat()

            if job_id in self._job_queue:
                self._job_queue.remove(job_id)

        self._save_state()
        logger.info(f"Cancelled job {job_id}")
        return True

    def _run_loop(self) -> None:
        """Main daemon processing loop."""
        while not self._stop_event.is_set():
            try:
                # Periodic state save
                self._save_state()

                # Wait for poll interval or stop signal
                self._stop_event.wait(self.config.poll_interval)

            except Exception as e:
                logger.error(f"Error in daemon main loop: {e}")

    def _worker_loop(self) -> None:
        """Worker thread loop for processing jobs."""
        while not self._stop_event.is_set():
            job_id = None
            try:
                # Wait for a job
                with self._queue_condition:
                    while not self._job_queue and not self._stop_event.is_set():
                        self._queue_condition.wait(timeout=1.0)

                    if self._stop_event.is_set():
                        break

                    if self._job_queue:
                        job_id = self._job_queue.pop(0)

                if job_id:
                    self._process_job(job_id)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                if job_id:
                    with self._lock:
                        if job_id in self._state.jobs:
                            job = self._state.jobs[job_id]
                            job.state = JobState.FAILED
                            job.error_message = str(e)
                            job.completed_at = datetime.now().isoformat()
                    self._save_state()

    def _process_job(self, job_id: str) -> None:
        """Process a single job.

        Args:
            job_id: ID of the job to process.
        """
        with self._lock:
            if job_id not in self._state.jobs:
                logger.warning(f"Job {job_id} not found")
                return

            job = self._state.jobs[job_id]
            job.state = JobState.RUNNING
            job.started_at = datetime.now().isoformat()
            self._state.current_jobs.append(job_id)

        self._save_state()
        logger.info(f"Processing job {job_id}: {Path(job.video_path).name}")

        if self.on_job_start:
            try:
                self.on_job_start(job)
            except Exception as e:
                logger.warning(f"Error in on_job_start callback: {e}")

        try:
            # Import here to avoid circular imports
            from ..config import Config
            from ..restorer import VideoRestorer

            # Build configuration
            video_path = Path(job.video_path)
            project_dir = self._jobs_dir / job_id

            config_dict = {
                "project_dir": project_dir,
                **job.config,
            }

            config = Config(**config_dict)
            config.create_directories()

            # Set up progress callback
            def progress_callback(progress_info: Any) -> None:
                if hasattr(progress_info, "progress"):
                    with self._lock:
                        job.progress = progress_info.progress
                elif hasattr(progress_info, "frames_completed") and hasattr(progress_info, "frames_total"):
                    if progress_info.frames_total > 0:
                        with self._lock:
                            job.progress = progress_info.frames_completed / progress_info.frames_total

            # Run restoration
            restorer = VideoRestorer(config, progress_callback=progress_callback)
            output_path = restorer.restore_video(
                source=job.video_path,
                cleanup=True,
                resume=True,
            )

            # Update job state on success
            with self._lock:
                job.state = JobState.COMPLETED
                job.completed_at = datetime.now().isoformat()
                job.output_path = str(output_path)
                job.progress = 1.0
                if job_id in self._state.current_jobs:
                    self._state.current_jobs.remove(job_id)
                self._state.stats["jobs_completed"] = self._state.stats.get("jobs_completed", 0) + 1

            self._save_state()
            logger.info(f"Completed job {job_id}: {output_path}")

            if self.on_job_complete:
                try:
                    self.on_job_complete(job)
                except Exception as e:
                    logger.warning(f"Error in on_job_complete callback: {e}")

        except Exception as e:
            # Handle job failure
            logger.error(f"Job {job_id} failed: {e}")

            with self._lock:
                job.state = JobState.FAILED
                job.completed_at = datetime.now().isoformat()
                job.error_message = str(e)
                if job_id in self._state.current_jobs:
                    self._state.current_jobs.remove(job_id)
                self._state.stats["jobs_failed"] = self._state.stats.get("jobs_failed", 0) + 1

            self._save_state()

            if self.on_job_failed:
                try:
                    self.on_job_failed(job, e)
                except Exception as callback_error:
                    logger.warning(f"Error in on_job_failed callback: {callback_error}")


# Factory functions

def get_default_daemon_config() -> DaemonConfig:
    """Get default daemon configuration.

    Returns:
        DaemonConfig with default paths based on platform.
    """
    if sys.platform == "win32":
        base_dir = Path.home() / ".framewright"
    else:
        base_dir = Path("/var/lib/framewright")

    return DaemonConfig(
        pid_file=base_dir / "daemon.pid",
        work_dir=base_dir / "daemon",
        log_file=base_dir / "logs" / "daemon.log",
    )


def start_daemon(
    config: Optional[DaemonConfig] = None,
    **kwargs: Any,
) -> FramewrightDaemon:
    """Start the FrameWright daemon.

    Args:
        config: Optional daemon configuration. Uses defaults if not provided.
        **kwargs: Additional arguments passed to DaemonConfig.

    Returns:
        Running FramewrightDaemon instance.
    """
    if config is None:
        config = get_default_daemon_config()

    # Apply any overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = DaemonConfig.from_dict(config_dict)

    daemon = FramewrightDaemon(config)
    daemon.start()
    return daemon


def stop_daemon(config: Optional[DaemonConfig] = None) -> bool:
    """Stop a running daemon.

    Sends termination signal to the daemon process.

    Args:
        config: Optional daemon configuration to locate PID file.

    Returns:
        True if daemon was stopped successfully.
    """
    if config is None:
        config = get_default_daemon_config()

    pid_file = config.pid_file
    if not pid_file.exists():
        logger.warning("No PID file found, daemon may not be running")
        return False

    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError) as e:
        logger.error(f"Failed to read PID file: {e}")
        return False

    # Send termination signal
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_TERMINATE = 0x0001
            handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
                logger.info(f"Sent termination signal to daemon (PID {pid})")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop daemon: {e}")
            return False
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to daemon (PID {pid})")
            return True
        except OSError as e:
            logger.error(f"Failed to stop daemon: {e}")
            return False


def get_daemon_status(config: Optional[DaemonConfig] = None) -> Dict[str, Any]:
    """Get the status of the daemon.

    Args:
        config: Optional daemon configuration.

    Returns:
        Dictionary with daemon status information.
    """
    if config is None:
        config = get_default_daemon_config()

    result: Dict[str, Any] = {
        "status": DaemonStatus.STOPPED.value,
        "pid": None,
        "uptime_seconds": None,
        "jobs_pending": 0,
        "jobs_running": 0,
        "jobs_completed": 0,
        "jobs_failed": 0,
    }

    # Check PID file
    pid_file = config.pid_file
    if not pid_file.exists():
        return result

    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return result

    # Check if process is running
    def is_running(p: int) -> bool:
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, p)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        else:
            try:
                os.kill(p, 0)
                return True
            except OSError:
                return False

    if not is_running(pid):
        return result

    result["pid"] = pid
    result["status"] = DaemonStatus.RUNNING.value

    # Load state file for detailed info
    state_file = config.work_dir / "daemon_state.json"
    if state_file.exists():
        try:
            state_data = json.loads(state_file.read_text())
            state = DaemonState.from_dict(state_data)

            result["status"] = state.status.value

            # Calculate uptime
            if state.started_at:
                started = datetime.fromisoformat(state.started_at)
                result["uptime_seconds"] = (datetime.now() - started).total_seconds()

            # Count jobs by state
            for job in state.jobs.values():
                if job.state == JobState.PENDING:
                    result["jobs_pending"] += 1
                elif job.state == JobState.RUNNING:
                    result["jobs_running"] += 1
                elif job.state == JobState.COMPLETED:
                    result["jobs_completed"] += 1
                elif job.state in (JobState.FAILED, JobState.CRASHED):
                    result["jobs_failed"] += 1

            # Add stats
            result["stats"] = state.stats

        except Exception as e:
            logger.warning(f"Failed to load daemon state: {e}")

    return result
