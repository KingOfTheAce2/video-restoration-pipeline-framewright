"""Web dashboard for monitoring restoration jobs."""

from .app import create_app, DashboardConfig
from .server import DashboardServer

__all__ = [
    "create_app",
    "DashboardConfig",
    "DashboardServer",
]
