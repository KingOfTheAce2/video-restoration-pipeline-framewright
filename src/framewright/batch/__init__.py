"""Batch processing module for FrameWright.

Provides job queue management, batch processing, cron-like scheduling, and daemon mode.
"""

from .queue_processor import (
    BatchQueueProcessor,
    BatchJob,
    JobStatus,
    JobPriority,
    JobProgress,
    PriorityQueue,
    create_batch_processor,
)
from .scheduler import (
    ScheduledJob,
    ScheduleConfig,
    CronParser,
    JobScheduler,
    create_scheduler,
    get_default_scheduler,
    schedule_job,
)
from .daemon import (
    DaemonConfig,
    DaemonStatus,
    DaemonState,
    DaemonJob,
    JobState,
    FramewrightDaemon,
    start_daemon,
    stop_daemon,
    get_daemon_status,
    get_default_daemon_config,
)

__all__ = [
    # Queue processor
    "BatchQueueProcessor",
    "BatchJob",
    "JobStatus",
    "JobPriority",
    "JobProgress",
    "PriorityQueue",
    "create_batch_processor",
    # Scheduler
    "ScheduledJob",
    "ScheduleConfig",
    "CronParser",
    "JobScheduler",
    "create_scheduler",
    "get_default_scheduler",
    "schedule_job",
    # Daemon mode
    "DaemonConfig",
    "DaemonStatus",
    "DaemonState",
    "DaemonJob",
    "JobState",
    "FramewrightDaemon",
    "start_daemon",
    "stop_daemon",
    "get_daemon_status",
    "get_default_daemon_config",
]
