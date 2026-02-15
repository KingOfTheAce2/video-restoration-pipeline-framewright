"""Persistence module for FrameWright.

Provides SQLite-based job state, progress tracking, and crash recovery.
"""

from .job_store import (
    JobStore,
    JobRecord,
    FrameRecord,
    JobState,
    FrameState,
)
from .progress_tracker import (
    ProgressTracker,
    ProgressSnapshot,
    CheckpointManager,
)

__all__ = [
    "JobStore",
    "JobRecord",
    "FrameRecord",
    "JobState",
    "FrameState",
    "ProgressTracker",
    "ProgressSnapshot",
    "CheckpointManager",
]
