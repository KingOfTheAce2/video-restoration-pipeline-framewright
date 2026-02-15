"""Dashboard server for running the web interface."""

import logging
import threading
import webbrowser
from pathlib import Path
from typing import Any, Optional

from .app import create_app, DashboardConfig

logger = logging.getLogger(__name__)


class DashboardServer:
    """Server for the FrameWright dashboard."""

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        job_store: Optional[Any] = None,
        progress_tracker: Optional[Any] = None,
    ):
        self.config = config or DashboardConfig()
        self.job_store = job_store
        self.progress_tracker = progress_tracker

        self._app = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def create_app(self) -> Any:
        """Create the Flask application."""
        self._app = create_app(
            config=self.config,
            job_store=self.job_store,
            progress_tracker=self.progress_tracker,
        )
        return self._app

    def run(
        self,
        open_browser: bool = True,
        blocking: bool = True,
    ) -> None:
        """Run the dashboard server.

        Args:
            open_browser: Open browser automatically
            blocking: If True, blocks until server stops
        """
        if self._app is None:
            self.create_app()

        if self._app is None:
            logger.error("Failed to create Flask app. Is Flask installed?")
            return

        url = f"http://{self.config.host}:{self.config.port}"

        if blocking:
            logger.info(f"Starting dashboard at {url}")
            if open_browser:
                webbrowser.open(url)

            self._running = True
            try:
                self._app.run(
                    host=self.config.host,
                    port=self.config.port,
                    debug=self.config.debug,
                    use_reloader=False,
                )
            except KeyboardInterrupt:
                pass
            finally:
                self._running = False
        else:
            # Run in background thread
            self._server_thread = threading.Thread(
                target=self._run_in_thread,
                daemon=True,
            )
            self._server_thread.start()

            logger.info(f"Dashboard started in background at {url}")
            if open_browser:
                import time
                time.sleep(0.5)  # Give server time to start
                webbrowser.open(url)

    def _run_in_thread(self) -> None:
        """Run server in a thread."""
        self._running = True
        try:
            # Use werkzeug server directly for thread safety
            from werkzeug.serving import make_server
            server = make_server(
                self.config.host,
                self.config.port,
                self._app,
                threaded=True,
            )
            self._server = server
            server.serve_forever()
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the dashboard server."""
        if hasattr(self, "_server"):
            self._server.shutdown()
        self._running = False
        logger.info("Dashboard server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"

    def set_job_store(self, job_store: Any) -> None:
        """Set the job store (for lazy initialization)."""
        self.job_store = job_store
        if self._app:
            self._app.job_store = job_store

    def set_progress_tracker(self, progress_tracker: Any) -> None:
        """Set the progress tracker (for lazy initialization)."""
        self.progress_tracker = progress_tracker
        if self._app:
            self._app.progress_tracker = progress_tracker


def start_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    job_store: Optional[Any] = None,
    progress_tracker: Optional[Any] = None,
    open_browser: bool = True,
    blocking: bool = True,
) -> Optional[DashboardServer]:
    """Convenience function to start the dashboard.

    Args:
        host: Host to bind to
        port: Port to listen on
        job_store: Optional job store instance
        progress_tracker: Optional progress tracker instance
        open_browser: Open browser automatically
        blocking: If True, blocks until stopped

    Returns:
        DashboardServer instance if non-blocking, None if blocking
    """
    config = DashboardConfig(host=host, port=port)
    server = DashboardServer(
        config=config,
        job_store=job_store,
        progress_tracker=progress_tracker,
    )
    server.run(open_browser=open_browser, blocking=blocking)

    return server if not blocking else None
