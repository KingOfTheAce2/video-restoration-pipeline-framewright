"""Watch mode module for FrameWright.

Monitors directories for new video files and automatically processes them
using the restoration pipeline.
"""
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)


# Default video file patterns
DEFAULT_VIDEO_PATTERNS = [
    "*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm", "*.flv", "*.wmv", "*.m4v"
]


class FileStatus(Enum):
    """Status of a file in the watch queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WatchedFile:
    """Represents a file being watched/processed."""
    path: Path
    status: FileStatus = FileStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    output_path: Optional[Path] = None
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None

    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.processing_start and self.processing_end:
            return self.processing_end - self.processing_start
        return None


@dataclass
class WatchConfig:
    """Configuration for watch mode."""
    input_dir: Path
    output_dir: Path
    profile: str = "quality"
    file_patterns: List[str] = field(default_factory=lambda: DEFAULT_VIDEO_PATTERNS.copy())
    on_complete_webhook: Optional[str] = None
    on_complete_command: Optional[str] = None
    on_error_webhook: Optional[str] = None
    on_error_command: Optional[str] = None
    retry_count: int = 3
    retry_delay_seconds: float = 60.0
    poll_interval_seconds: float = 5.0
    min_file_age_seconds: float = 5.0  # Wait for file to be fully written
    move_processed: bool = False
    processed_dir: Optional[Path] = None
    delete_processed: bool = False
    scale_factor: int = 4
    model_name: str = "realesrgan-x4plus"
    crf: int = 18
    output_format: str = "mkv"
    enable_interpolation: bool = False
    target_fps: Optional[float] = None
    enable_auto_enhance: bool = False

    def __post_init__(self):
        """Convert paths and validate configuration."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.processed_dir:
            self.processed_dir = Path(self.processed_dir)


class WebhookNotifier:
    """Handles webhook notifications for watch events."""

    @staticmethod
    def send_webhook(url: str, payload: Dict[str, Any], timeout: int = 30) -> bool:
        """Send a webhook notification.

        Args:
            url: Webhook URL
            payload: JSON payload to send
            timeout: Request timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            data = json.dumps(payload).encode("utf-8")
            request = Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                return response.status == 200
        except URLError as e:
            logger.error(f"Webhook failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False

    @staticmethod
    def execute_command(
        command: str,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ) -> bool:
        """Execute a command as notification.

        Args:
            command: Shell command to execute
            env_vars: Additional environment variables
            timeout: Command timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)

            result = subprocess.run(
                command,
                shell=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return False
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False


class WatchMode:
    """Watch mode for automatic video processing.

    Monitors a directory for new video files and automatically processes
    them using the FrameWright restoration pipeline.

    Features:
    - File system monitoring with configurable patterns
    - Processing queue with retry logic
    - Webhook/command notifications on completion/error
    - Graceful shutdown handling
    - Thread-safe operation

    Example:
        >>> config = WatchConfig(
        ...     input_dir=Path("./incoming"),
        ...     output_dir=Path("./processed"),
        ...     profile="quality",
        ...     on_complete_webhook="https://example.com/webhook",
        ... )
        >>> watcher = WatchMode(config)
        >>> watcher.start()  # Blocks until stopped
    """

    def __init__(self, config: WatchConfig):
        """Initialize watch mode.

        Args:
            config: Watch configuration
        """
        self.config = config
        self._queue: queue.Queue[WatchedFile] = queue.Queue()
        self._processed: Dict[str, WatchedFile] = {}
        self._known_files: Set[str] = set()
        self._running = False
        self._stop_event = threading.Event()
        self._processor_thread: Optional[threading.Thread] = None
        self._use_watchdog = False
        self._observer = None

        # Try to use watchdog if available
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            self._use_watchdog = True
            logger.info("Using watchdog for file system monitoring")
        except ImportError:
            logger.info("Watchdog not available, using polling mode")

    def _setup_watchdog(self):
        """Set up watchdog observer if available."""
        if not self._use_watchdog:
            return

        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

        watcher = self

        class VideoFileHandler(FileSystemEventHandler):
            """Handle file system events for video files."""

            def on_created(self, event):
                if not event.is_directory:
                    watcher._handle_new_file(Path(event.src_path))

            def on_moved(self, event):
                if not event.is_directory:
                    watcher._handle_new_file(Path(event.dest_path))

        self._observer = Observer()
        self._observer.schedule(
            VideoFileHandler(),
            str(self.config.input_dir),
            recursive=False,
        )

    def _matches_pattern(self, path: Path) -> bool:
        """Check if a file matches configured patterns.

        Args:
            path: File path to check

        Returns:
            True if file matches any pattern
        """
        for pattern in self.config.file_patterns:
            if path.match(pattern):
                return True
        return False

    def _is_file_ready(self, path: Path) -> bool:
        """Check if a file is ready for processing (fully written).

        Args:
            path: File path to check

        Returns:
            True if file is ready
        """
        try:
            # Check file age
            mtime = path.stat().st_mtime
            age = time.time() - mtime
            if age < self.config.min_file_age_seconds:
                return False

            # Check file size stability
            size1 = path.stat().st_size
            time.sleep(0.5)
            size2 = path.stat().st_size
            return size1 == size2 and size1 > 0

        except (OSError, FileNotFoundError):
            return False

    def _handle_new_file(self, path: Path):
        """Handle a newly detected file.

        Args:
            path: Path to the new file
        """
        if not path.exists():
            return

        if not self._matches_pattern(path):
            return

        file_key = str(path.resolve())
        if file_key in self._known_files:
            return

        self._known_files.add(file_key)
        logger.info(f"New video detected: {path.name}")

        watched = WatchedFile(path=path)
        self._queue.put(watched)

    def _scan_directory(self):
        """Scan input directory for existing video files."""
        for pattern in self.config.file_patterns:
            for path in self.config.input_dir.glob(pattern):
                if path.is_file():
                    self._handle_new_file(path)

    def _poll_directory(self):
        """Poll directory for new files (fallback when watchdog unavailable)."""
        while not self._stop_event.is_set():
            try:
                self._scan_directory()
            except Exception as e:
                logger.error(f"Error scanning directory: {e}")

            # Wait for poll interval
            self._stop_event.wait(self.config.poll_interval_seconds)

    def _process_file(self, watched: WatchedFile) -> bool:
        """Process a single video file.

        Args:
            watched: Watched file to process

        Returns:
            True if processing succeeded
        """
        from .config import Config, PRESETS
        from .restorer import VideoRestorer

        watched.status = FileStatus.PROCESSING
        watched.processing_start = time.time()
        watched.attempts += 1

        logger.info(f"Processing: {watched.path.name} (attempt {watched.attempts})")

        try:
            # Wait for file to be ready
            if not self._is_file_ready(watched.path):
                logger.debug(f"File not ready, requeueing: {watched.path.name}")
                watched.status = FileStatus.PENDING
                self._queue.put(watched)
                return False

            # Build output path
            output_name = f"{watched.path.stem}_restored.{self.config.output_format}"
            output_path = self.config.output_dir / output_name

            # Create work directory
            work_dir = self.config.output_dir / ".framewright_work" / watched.path.stem
            work_dir.mkdir(parents=True, exist_ok=True)

            # Build config from profile or settings
            if self.config.profile in PRESETS:
                config = Config.from_preset(
                    self.config.profile,
                    project_dir=work_dir,
                    output_dir=self.config.output_dir,
                    output_format=self.config.output_format,
                )
            else:
                config = Config(
                    project_dir=work_dir,
                    output_dir=self.config.output_dir,
                    scale_factor=self.config.scale_factor,
                    model_name=self.config.model_name,
                    crf=self.config.crf,
                    output_format=self.config.output_format,
                    enable_interpolation=self.config.enable_interpolation,
                    target_fps=self.config.target_fps,
                    enable_auto_enhance=self.config.enable_auto_enhance,
                )

            # Run restoration
            restorer = VideoRestorer(config)
            result_path = restorer.restore_video(
                source=str(watched.path),
                output_path=output_path,
                cleanup=True,
                resume=False,
            )

            watched.output_path = result_path
            watched.status = FileStatus.COMPLETED
            watched.processing_end = time.time()
            watched.last_error = None

            logger.info(
                f"Completed: {watched.path.name} -> {result_path.name} "
                f"({watched.processing_time:.1f}s)"
            )

            # Handle post-processing
            self._post_process_success(watched)

            return True

        except Exception as e:
            watched.last_error = str(e)
            watched.processing_end = time.time()
            logger.error(f"Failed: {watched.path.name} - {e}")

            # Check if should retry
            if watched.attempts < self.config.retry_count:
                watched.status = FileStatus.PENDING
                logger.info(
                    f"Will retry {watched.path.name} in {self.config.retry_delay_seconds}s "
                    f"(attempt {watched.attempts}/{self.config.retry_count})"
                )
                # Delay before retry
                time.sleep(self.config.retry_delay_seconds)
                self._queue.put(watched)
            else:
                watched.status = FileStatus.FAILED
                self._post_process_failure(watched)

            return False

    def _post_process_success(self, watched: WatchedFile):
        """Handle successful processing.

        Args:
            watched: Completed watched file
        """
        self._processed[str(watched.path)] = watched

        # Send completion notification
        payload = {
            "event": "complete",
            "input_file": str(watched.path),
            "output_file": str(watched.output_path) if watched.output_path else None,
            "processing_time_seconds": watched.processing_time,
            "timestamp": time.time(),
        }

        if self.config.on_complete_webhook:
            WebhookNotifier.send_webhook(self.config.on_complete_webhook, payload)

        if self.config.on_complete_command:
            env_vars = {
                "FRAMEWRIGHT_INPUT": str(watched.path),
                "FRAMEWRIGHT_OUTPUT": str(watched.output_path) if watched.output_path else "",
                "FRAMEWRIGHT_STATUS": "complete",
            }
            WebhookNotifier.execute_command(self.config.on_complete_command, env_vars)

        # Move or delete processed file
        if self.config.move_processed and self.config.processed_dir:
            self.config.processed_dir.mkdir(parents=True, exist_ok=True)
            dest = self.config.processed_dir / watched.path.name
            try:
                shutil.move(str(watched.path), str(dest))
                logger.info(f"Moved original to: {dest}")
            except Exception as e:
                logger.error(f"Failed to move file: {e}")

        elif self.config.delete_processed:
            try:
                watched.path.unlink()
                logger.info(f"Deleted original: {watched.path.name}")
            except Exception as e:
                logger.error(f"Failed to delete file: {e}")

    def _post_process_failure(self, watched: WatchedFile):
        """Handle failed processing.

        Args:
            watched: Failed watched file
        """
        self._processed[str(watched.path)] = watched

        # Send failure notification
        payload = {
            "event": "error",
            "input_file": str(watched.path),
            "error": watched.last_error,
            "attempts": watched.attempts,
            "timestamp": time.time(),
        }

        if self.config.on_error_webhook:
            WebhookNotifier.send_webhook(self.config.on_error_webhook, payload)

        if self.config.on_error_command:
            env_vars = {
                "FRAMEWRIGHT_INPUT": str(watched.path),
                "FRAMEWRIGHT_ERROR": watched.last_error or "Unknown error",
                "FRAMEWRIGHT_STATUS": "error",
                "FRAMEWRIGHT_ATTEMPTS": str(watched.attempts),
            }
            WebhookNotifier.execute_command(self.config.on_error_command, env_vars)

    def _processor_loop(self):
        """Main processing loop."""
        while not self._stop_event.is_set():
            try:
                # Get next file from queue with timeout
                watched = self._queue.get(timeout=1.0)
                self._process_file(watched)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processor error: {e}")

    def start(self, blocking: bool = True):
        """Start watch mode.

        Args:
            blocking: If True, block until stopped. If False, return immediately.
        """
        if self._running:
            logger.warning("Watch mode already running")
            return

        self._running = True
        self._stop_event.clear()

        # Ensure directories exist
        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting watch mode")
        logger.info(f"  Input directory: {self.config.input_dir}")
        logger.info(f"  Output directory: {self.config.output_dir}")
        logger.info(f"  Profile: {self.config.profile}")
        logger.info(f"  File patterns: {', '.join(self.config.file_patterns)}")

        # Initial scan
        self._scan_directory()

        # Start processor thread
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            daemon=True,
            name="framewright-processor",
        )
        self._processor_thread.start()

        # Start file monitoring
        if self._use_watchdog:
            self._setup_watchdog()
            self._observer.start()
            logger.info("File system observer started")
        else:
            # Start polling thread
            poll_thread = threading.Thread(
                target=self._poll_directory,
                daemon=True,
                name="framewright-poller",
            )
            poll_thread.start()
            logger.info("Polling mode started")

        if blocking:
            try:
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()

    def stop(self):
        """Stop watch mode gracefully."""
        logger.info("Stopping watch mode...")
        self._running = False
        self._stop_event.set()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

        if self._processor_thread:
            self._processor_thread.join(timeout=10)

        logger.info("Watch mode stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current watch mode status.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "known_files": len(self._known_files),
            "processed_files": len(self._processed),
            "completed": sum(
                1 for f in self._processed.values()
                if f.status == FileStatus.COMPLETED
            ),
            "failed": sum(
                1 for f in self._processed.values()
                if f.status == FileStatus.FAILED
            ),
        }

    def get_processed_files(self) -> List[Dict[str, Any]]:
        """Get list of processed files.

        Returns:
            List of processed file information
        """
        return [
            {
                "input": str(f.path),
                "output": str(f.output_path) if f.output_path else None,
                "status": f.status.value,
                "attempts": f.attempts,
                "error": f.last_error,
                "processing_time": f.processing_time,
            }
            for f in self._processed.values()
        ]


def start_watch_mode(
    input_dir: Path,
    output_dir: Path,
    profile: str = "quality",
    on_complete: Optional[str] = None,
    on_error: Optional[str] = None,
    retry_count: int = 3,
    file_patterns: Optional[List[str]] = None,
    **kwargs,
) -> WatchMode:
    """Convenience function to start watch mode.

    Args:
        input_dir: Directory to watch for new videos
        output_dir: Directory for processed videos
        profile: Processing profile name
        on_complete: Webhook URL or command for completion
        on_error: Webhook URL or command for errors
        retry_count: Number of retries on failure
        file_patterns: List of glob patterns for video files
        **kwargs: Additional config options

    Returns:
        WatchMode instance (already started)
    """
    # Determine if on_complete/on_error are webhooks or commands
    on_complete_webhook = None
    on_complete_command = None
    if on_complete:
        if on_complete.startswith(("http://", "https://")):
            on_complete_webhook = on_complete
        else:
            on_complete_command = on_complete

    on_error_webhook = None
    on_error_command = None
    if on_error:
        if on_error.startswith(("http://", "https://")):
            on_error_webhook = on_error
        else:
            on_error_command = on_error

    config = WatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        profile=profile,
        file_patterns=file_patterns or DEFAULT_VIDEO_PATTERNS.copy(),
        on_complete_webhook=on_complete_webhook,
        on_complete_command=on_complete_command,
        on_error_webhook=on_error_webhook,
        on_error_command=on_error_command,
        retry_count=retry_count,
        **kwargs,
    )

    watcher = WatchMode(config)
    watcher.start(blocking=False)
    return watcher
