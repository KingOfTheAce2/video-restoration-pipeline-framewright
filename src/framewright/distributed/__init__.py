"""Distributed render farm module for FrameWright.

Provides multi-node processing for large-scale video restoration.
"""

from .coordinator import RenderCoordinator, CoordinatorConfig
from .worker import RenderWorker, WorkerConfig
from .job import RenderJob, JobStatus, JobPriority, FrameRange
from .discovery import NodeDiscovery, DiscoveryMethod

__all__ = [
    "RenderCoordinator",
    "CoordinatorConfig",
    "RenderWorker",
    "WorkerConfig",
    "RenderJob",
    "JobStatus",
    "JobPriority",
    "FrameRange",
    "NodeDiscovery",
    "DiscoveryMethod",
]
