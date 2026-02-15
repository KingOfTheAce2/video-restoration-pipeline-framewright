"""Web dashboard for FrameWright video restoration.

This module provides a web-based dashboard for monitoring and managing
video restoration jobs. It uses only Python standard library (http.server)
with no external dependencies required.

Features:
- Real-time job monitoring with WebSocket updates
- System resource visualization (CPU, RAM, GPU)
- Job queue management (submit, cancel, view details)
- Model management interface
- Responsive design for desktop and mobile
- Optional API key authentication

Example usage:

    >>> from framewright.ui.dashboard import DashboardServer, DashboardConfig
    >>>
    >>> # Start with default settings
    >>> server = DashboardServer()
    >>> server.start()  # Blocks until stopped
    >>>
    >>> # Start with custom configuration
    >>> config = DashboardConfig(
    ...     host="0.0.0.0",
    ...     port=8080,
    ...     require_auth=True,
    ...     api_key="your-secret-key",
    ... )
    >>> server = DashboardServer(config=config)
    >>> server.start(blocking=False)  # Run in background
    >>> print(f"Dashboard at {server.url}")
    >>>
    >>> # Stop the server
    >>> server.stop()

Quick start function:

    >>> from framewright.ui.dashboard import start_dashboard
    >>>
    >>> # Start dashboard with defaults
    >>> start_dashboard(open_browser=True)
    >>>
    >>> # Start in background
    >>> server = start_dashboard(blocking=False)
"""

from .server import (
    DashboardConfig,
    DashboardServer,
    DashboardHandler,
    WebSocketConnection,
    WebSocketManager,
    start_dashboard,
)

from .templates import (
    render_dashboard_page,
    render_error_page,
    render_login_page,
    render_job_card,
    render_model_card,
    DASHBOARD_CSS,
    DASHBOARD_JS,
    ICONS,
    icon,
)

__all__ = [
    # Server
    "DashboardConfig",
    "DashboardServer",
    "DashboardHandler",
    "WebSocketConnection",
    "WebSocketManager",
    "start_dashboard",
    # Templates
    "render_dashboard_page",
    "render_error_page",
    "render_login_page",
    "render_job_card",
    "render_model_card",
    "DASHBOARD_CSS",
    "DASHBOARD_JS",
    "ICONS",
    "icon",
]
