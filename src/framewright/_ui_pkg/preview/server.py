"""Real-time preview server for live restoration preview.

Provides a web-based preview interface for video restoration with:
- Before/after comparison slider
- Segment caching for instant replay
- Background rendering queue
- Real-time progress updates via WebSocket
- Side-by-side comparison mode

Example:
    >>> # Quick start
    >>> preview_video("input.mp4")

    >>> # With configuration
    >>> config = PreviewConfig(port=8080, quality="high")
    >>> server = PreviewServer()
    >>> server.start("input.mp4", config)
    >>> # ... browse to http://localhost:8080
    >>> server.stop()
"""

import base64
import hashlib
import http.server
import io
import json
import logging
import os
import queue
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class PreviewQuality(Enum):
    """Preview quality levels."""
    LOW = "low"       # Fast, 360p max
    MEDIUM = "medium" # Balanced, 540p max
    HIGH = "high"     # High quality, 720p max


@dataclass
class PreviewConfig:
    """Configuration for the preview server.

    Attributes:
        host: Server host address (default "localhost")
        port: Server port (default 8080)
        quality: Preview quality level (low/medium/high)
        max_resolution: Maximum preview resolution in pixels (height)
        cache_segments: Whether to cache rendered segments
        auto_open_browser: Open browser automatically on start
        cache_dir: Directory for segment cache (auto-created if None)
        segment_duration: Default segment duration in seconds
        max_cache_size_mb: Maximum cache size in megabytes
        processor_timeout: Timeout for segment rendering in seconds
    """
    host: str = "localhost"
    port: int = 8080
    quality: Union[str, PreviewQuality] = PreviewQuality.MEDIUM
    max_resolution: int = 720
    cache_segments: bool = True
    auto_open_browser: bool = True
    cache_dir: Optional[Path] = None
    segment_duration: float = 5.0
    max_cache_size_mb: int = 500
    processor_timeout: float = 120.0

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        if isinstance(self.quality, str):
            try:
                self.quality = PreviewQuality(self.quality.lower())
            except ValueError:
                logger.warning(f"Invalid quality '{self.quality}', using medium")
                self.quality = PreviewQuality.MEDIUM

        # Set resolution based on quality if not explicitly set high
        quality_resolutions = {
            PreviewQuality.LOW: 360,
            PreviewQuality.MEDIUM: 540,
            PreviewQuality.HIGH: 720,
        }

        # Cap resolution based on quality
        max_for_quality = quality_resolutions.get(self.quality, 720)
        if self.max_resolution > max_for_quality:
            self.max_resolution = max_for_quality

        # Create cache directory
        if self.cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "framewright_preview_cache"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class VideoInfo:
    """Information about the loaded video."""
    path: Path
    width: int = 0
    height: int = 0
    fps: float = 30.0
    duration: float = 0.0
    frame_count: int = 0
    codec: str = ""

    @classmethod
    def from_path(cls, path: Path) -> "VideoInfo":
        """Extract video information using ffprobe."""
        info = cls(path=path)

        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    str(path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Find video stream
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        info.width = int(stream.get("width", 0))
                        info.height = int(stream.get("height", 0))
                        info.codec = stream.get("codec_name", "")
                        info.frame_count = int(stream.get("nb_frames", 0))

                        # Parse frame rate
                        fps_str = stream.get("r_frame_rate", "30/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            info.fps = float(num) / float(den) if float(den) > 0 else 30.0
                        else:
                            info.fps = float(fps_str)
                        break

                # Duration from format
                fmt = data.get("format", {})
                info.duration = float(fmt.get("duration", 0))

                if info.frame_count == 0 and info.duration > 0:
                    info.frame_count = int(info.duration * info.fps)

        except Exception as e:
            logger.warning(f"Could not get video info: {e}")

        return info


@dataclass
class RenderTask:
    """A segment rendering task."""
    id: str
    start: float
    duration: float
    config_hash: str
    priority: int = 0
    status: str = "pending"  # pending, rendering, completed, failed
    output_path: Optional[Path] = None
    error: Optional[str] = None
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class ServerStatus:
    """Server status information."""
    running: bool = False
    video_loaded: bool = False
    video_path: Optional[str] = None
    video_info: Optional[Dict[str, Any]] = None
    active_renders: int = 0
    cached_segments: int = 0
    cache_size_mb: float = 0.0
    connected_clients: int = 0
    uptime_seconds: float = 0.0


class SegmentCache:
    """Cache for rendered preview segments."""

    def __init__(self, cache_dir: Path, max_size_mb: int = 500):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._entries: Dict[str, Tuple[Path, float]] = {}  # key -> (path, access_time)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing cache entries."""
        try:
            for f in self.cache_dir.glob("*.mp4"):
                key = f.stem
                self._entries[key] = (f, f.stat().st_mtime)
        except Exception as e:
            logger.debug(f"Error loading cache: {e}")

    def _make_key(self, start: float, duration: float, config_hash: str) -> str:
        """Generate cache key for a segment."""
        data = f"{start:.3f}_{duration:.3f}_{config_hash}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def get(self, start: float, duration: float, config_hash: str) -> Optional[Path]:
        """Get cached segment if available."""
        key = self._make_key(start, duration, config_hash)

        with self._lock:
            if key in self._entries:
                path, _ = self._entries[key]
                if path.exists():
                    self._entries[key] = (path, time.time())
                    return path
                else:
                    del self._entries[key]
        return None

    def put(self, start: float, duration: float, config_hash: str, data: bytes) -> Path:
        """Store segment in cache."""
        key = self._make_key(start, duration, config_hash)
        path = self.cache_dir / f"{key}.mp4"

        with self._lock:
            # Evict old entries if needed
            self._evict_if_needed(len(data))

            # Write data
            path.write_bytes(data)
            self._entries[key] = (path, time.time())

        return path

    def put_file(self, start: float, duration: float, config_hash: str, source: Path) -> Path:
        """Move file into cache."""
        key = self._make_key(start, duration, config_hash)
        dest = self.cache_dir / f"{key}.mp4"

        with self._lock:
            size = source.stat().st_size
            self._evict_if_needed(size)

            shutil.copy2(source, dest)
            self._entries[key] = (dest, time.time())

        return dest

    def _evict_if_needed(self, needed_bytes: int) -> None:
        """Evict old entries to make room."""
        max_bytes = self.max_size_mb * 1024 * 1024
        current_size = self.get_size_bytes()

        if current_size + needed_bytes <= max_bytes:
            return

        # Sort by access time (oldest first)
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1][1]
        )

        for key, (path, _) in sorted_entries:
            if current_size + needed_bytes <= max_bytes * 0.8:
                break

            try:
                size = path.stat().st_size
                path.unlink()
                del self._entries[key]
                current_size -= size
            except Exception:
                pass

    def get_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for key, (path, _) in list(self._entries.items()):
            try:
                total += path.stat().st_size
            except Exception:
                pass
        return total

    def clear(self) -> None:
        """Clear all cached segments."""
        with self._lock:
            for key, (path, _) in list(self._entries.items()):
                try:
                    path.unlink()
                except Exception:
                    pass
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class RenderQueue:
    """Background rendering queue for preview segments."""

    def __init__(
        self,
        video_path: Path,
        config: PreviewConfig,
        cache: SegmentCache,
        processor: Optional[Any] = None
    ):
        self.video_path = video_path
        self.config = config
        self.cache = cache
        self.processor = processor

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._tasks: Dict[str, RenderTask] = {}
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._current_task: Optional[RenderTask] = None
        self._progress_callbacks: List[Callable[[RenderTask], None]] = []

    def start(self) -> None:
        """Start the render queue worker."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the render queue."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put((float('inf'), None))  # Sentinel to wake up worker
            self._worker_thread.join(timeout=5.0)

    def submit(self, start: float, duration: float, config_hash: str, priority: int = 0) -> RenderTask:
        """Submit a rendering task."""
        task_id = f"{start:.3f}_{duration:.3f}_{config_hash}"

        with self._lock:
            # Check if task already exists
            if task_id in self._tasks:
                return self._tasks[task_id]

            # Check cache first
            cached = self.cache.get(start, duration, config_hash)
            if cached:
                task = RenderTask(
                    id=task_id,
                    start=start,
                    duration=duration,
                    config_hash=config_hash,
                    status="completed",
                    output_path=cached,
                    progress=1.0,
                    completed_at=time.time()
                )
                self._tasks[task_id] = task
                return task

            # Create new task
            task = RenderTask(
                id=task_id,
                start=start,
                duration=duration,
                config_hash=config_hash,
                priority=priority
            )
            self._tasks[task_id] = task
            self._queue.put((priority, task_id))

        return task

    def get_task(self, task_id: str) -> Optional[RenderTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == "pending":
                task.status = "cancelled"
                return True
        return False

    def on_progress(self, callback: Callable[[RenderTask], None]) -> None:
        """Register progress callback."""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, task: RenderTask) -> None:
        """Notify all progress callbacks."""
        for cb in self._progress_callbacks:
            try:
                cb(task)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    def _worker(self) -> None:
        """Background worker that processes render tasks."""
        while self._running:
            try:
                priority, task_id = self._queue.get(timeout=1.0)

                if task_id is None:  # Sentinel
                    continue

                with self._lock:
                    task = self._tasks.get(task_id)
                    if not task or task.status != "pending":
                        continue
                    task.status = "rendering"
                    self._current_task = task

                self._notify_progress(task)

                try:
                    output = self._render_segment(task)

                    with self._lock:
                        if output:
                            # Cache the result
                            cached_path = self.cache.put_file(
                                task.start, task.duration, task.config_hash, output
                            )
                            task.output_path = cached_path
                            task.status = "completed"
                            task.progress = 1.0
                            task.completed_at = time.time()

                            # Clean up temp file
                            try:
                                output.unlink()
                            except Exception:
                                pass
                        else:
                            task.status = "failed"
                            task.error = "Render returned no output"

                except Exception as e:
                    logger.error(f"Render task failed: {e}")
                    with self._lock:
                        task.status = "failed"
                        task.error = str(e)

                finally:
                    with self._lock:
                        self._current_task = None
                    self._notify_progress(task)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Render worker error: {e}")

    def _render_segment(self, task: RenderTask) -> Optional[Path]:
        """Render a single segment."""
        output_path = Path(tempfile.mktemp(suffix=".mp4"))

        # Calculate scale for preview
        scale_height = self.config.max_resolution

        # Use ffmpeg to extract and scale segment
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(task.start),
                "-i", str(self.video_path),
                "-t", str(task.duration),
                "-vf", f"scale=-2:{scale_height}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",  # No audio for preview
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.config.processor_timeout
            )

            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                logger.error(f"FFmpeg error: {result.stderr.decode()}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Segment render timeout: {task.start}-{task.duration}")
            return None
        except Exception as e:
            logger.error(f"Segment render error: {e}")
            return None

    @property
    def active_count(self) -> int:
        """Number of active (pending + rendering) tasks."""
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.status in ("pending", "rendering")
            )


class PreviewServer:
    """Real-time preview server for video restoration.

    Provides a web interface for previewing video restoration with
    before/after comparison and real-time progress updates.

    Example:
        >>> server = PreviewServer()
        >>> server.start("input.mp4")
        >>> # Browse to http://localhost:8080
        >>> server.stop()
    """

    def __init__(self, processor: Optional[Any] = None):
        """Initialize the preview server.

        Args:
            processor: Optional restoration processor for rendering
        """
        self.processor = processor

        self._server: Optional[socketserver.TCPServer] = None
        self._config: Optional[PreviewConfig] = None
        self._video_info: Optional[VideoInfo] = None
        self._cache: Optional[SegmentCache] = None
        self._render_queue: Optional[RenderQueue] = None

        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._ws_clients: List[Any] = []
        self._lock = threading.RLock()

        # Current settings for comparison
        self._current_settings: Dict[str, Any] = {}
        self._settings_hash: str = ""

    def start(
        self,
        input_path: Union[str, Path],
        config: Optional[PreviewConfig] = None
    ) -> bool:
        """Start the preview server.

        Args:
            input_path: Path to the input video
            config: Server configuration

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Server already running")
            return False

        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Video file not found: {input_path}")
            return False

        self._config = config or PreviewConfig()
        self._video_info = VideoInfo.from_path(input_path)

        if self._video_info.duration <= 0:
            logger.error("Could not determine video duration")
            return False

        # Initialize cache
        self._cache = SegmentCache(
            self._config.cache_dir,
            self._config.max_cache_size_mb
        )

        # Initialize render queue
        self._render_queue = RenderQueue(
            input_path,
            self._config,
            self._cache,
            self.processor
        )
        self._render_queue.start()

        # Update settings hash
        self._update_settings_hash()

        # Create and start HTTP server
        try:
            handler = self._create_handler()
            self._server = socketserver.TCPServer(
                (self._config.host, self._config.port),
                handler
            )
            self._server.socket.settimeout(1.0)

            self._running = True
            self._start_time = time.time()

            self._server_thread = threading.Thread(
                target=self._serve_forever,
                daemon=True
            )
            self._server_thread.start()

            url = f"http://{self._config.host}:{self._config.port}"
            logger.info(f"Preview server started at {url}")

            if self._config.auto_open_browser:
                webbrowser.open(url)

            return True

        except OSError as e:
            logger.error(f"Could not start server: {e}")
            return False

    def stop(self) -> None:
        """Stop the preview server."""
        self._running = False

        if self._render_queue:
            self._render_queue.stop()

        if self._server:
            self._server.shutdown()

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)

        logger.info("Preview server stopped")

    def render_segment(
        self,
        start: float,
        duration: Optional[float] = None,
        priority: int = 0
    ) -> RenderTask:
        """Request rendering of a preview segment.

        Args:
            start: Start time in seconds
            duration: Duration in seconds (default from config)
            priority: Render priority (lower = higher priority)

        Returns:
            RenderTask tracking the render progress
        """
        if not self._render_queue:
            raise RuntimeError("Server not started")

        duration = duration or self._config.segment_duration
        return self._render_queue.submit(start, duration, self._settings_hash, priority)

    def compare_settings(self, configs: List[Dict[str, Any]]) -> Dict[str, RenderTask]:
        """Render segments with different settings for comparison.

        Args:
            configs: List of settings dictionaries to compare

        Returns:
            Dict mapping config description to render task
        """
        if not self._render_queue:
            raise RuntimeError("Server not started")

        results = {}
        mid_point = self._video_info.duration / 2 if self._video_info else 0

        for i, cfg in enumerate(configs):
            # Create hash for this config
            cfg_str = json.dumps(cfg, sort_keys=True)
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:16]

            task = self._render_queue.submit(
                mid_point,
                self._config.segment_duration,
                cfg_hash,
                priority=i
            )
            results[cfg.get("name", f"config_{i}")] = task

        return results

    def get_status(self) -> ServerStatus:
        """Get current server status."""
        status = ServerStatus(
            running=self._running,
            video_loaded=self._video_info is not None,
        )

        if self._video_info:
            status.video_path = str(self._video_info.path)
            status.video_info = {
                "width": self._video_info.width,
                "height": self._video_info.height,
                "fps": self._video_info.fps,
                "duration": self._video_info.duration,
                "frame_count": self._video_info.frame_count,
            }

        if self._render_queue:
            status.active_renders = self._render_queue.active_count

        if self._cache:
            status.cached_segments = len(self._cache)
            status.cache_size_mb = self._cache.get_size_bytes() / (1024 * 1024)

        if self._running:
            status.uptime_seconds = time.time() - self._start_time

        return status

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update preview settings.

        Args:
            settings: New settings dictionary
        """
        with self._lock:
            self._current_settings.update(settings)
            self._update_settings_hash()

    def _update_settings_hash(self) -> None:
        """Update the settings hash."""
        settings_str = json.dumps(self._current_settings, sort_keys=True)
        self._settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:16]

    def _serve_forever(self) -> None:
        """Server main loop."""
        while self._running:
            try:
                self._server.handle_request()
            except Exception as e:
                if self._running:
                    logger.error(f"Server error: {e}")

    def _create_handler(self) -> type:
        """Create HTTP request handler class."""
        server = self

        class PreviewHandler(http.server.BaseHTTPRequestHandler):
            """HTTP request handler for preview server."""

            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:
                """Suppress default logging."""
                pass

            def do_GET(self) -> None:
                """Handle GET requests."""
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/" or path == "/index.html":
                    self._serve_html()
                elif path == "/status":
                    self._serve_status()
                elif path.startswith("/preview/"):
                    self._serve_preview(path)
                elif path.startswith("/thumbnail/"):
                    self._serve_thumbnail(path)
                elif path == "/compare":
                    self._serve_compare()
                elif path == "/video-info":
                    self._serve_video_info()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                """Handle POST requests."""
                if self.path == "/settings":
                    self._handle_settings()
                elif self.path == "/render":
                    self._handle_render()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def _serve_html(self) -> None:
                """Serve the main HTML page."""
                html = _get_web_ui_html()
                self._send_response(html, "text/html")

            def _serve_status(self) -> None:
                """Serve server status as JSON."""
                status = server.get_status()
                data = {
                    "running": status.running,
                    "videoLoaded": status.video_loaded,
                    "videoPath": status.video_path,
                    "videoInfo": status.video_info,
                    "activeRenders": status.active_renders,
                    "cachedSegments": status.cached_segments,
                    "cacheSizeMb": round(status.cache_size_mb, 2),
                    "uptimeSeconds": round(status.uptime_seconds, 1),
                }
                self._send_json(data)

            def _serve_video_info(self) -> None:
                """Serve video information."""
                if server._video_info:
                    data = {
                        "width": server._video_info.width,
                        "height": server._video_info.height,
                        "fps": server._video_info.fps,
                        "duration": server._video_info.duration,
                        "frameCount": server._video_info.frame_count,
                        "codec": server._video_info.codec,
                    }
                    self._send_json(data)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "No video loaded")

            def _serve_preview(self, path: str) -> None:
                """Serve a preview segment."""
                try:
                    parts = path.split("/")
                    # /preview/<start>/<duration>
                    if len(parts) >= 4:
                        start = float(parts[2])
                        duration = float(parts[3])
                    else:
                        start = float(parts[2]) if len(parts) > 2 else 0
                        duration = server._config.segment_duration

                    # Request render
                    task = server.render_segment(start, duration, priority=0)

                    # Wait for completion with timeout
                    timeout = 60.0
                    poll_interval = 0.1
                    elapsed = 0.0

                    while elapsed < timeout:
                        if task.status == "completed" and task.output_path:
                            # Serve the video file
                            self._serve_file(task.output_path, "video/mp4")
                            return
                        elif task.status == "failed":
                            self.send_error(
                                HTTPStatus.INTERNAL_SERVER_ERROR,
                                task.error or "Render failed"
                            )
                            return

                        time.sleep(poll_interval)
                        elapsed += poll_interval
                        task = server._render_queue.get_task(task.id)

                    self.send_error(HTTPStatus.GATEWAY_TIMEOUT, "Render timeout")

                except (ValueError, IndexError) as e:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(e))

            def _serve_thumbnail(self, path: str) -> None:
                """Serve a frame thumbnail."""
                try:
                    parts = path.split("/")
                    time_sec = float(parts[2]) if len(parts) > 2 else 0

                    # Extract frame using ffmpeg
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-ss", str(time_sec),
                            "-i", str(server._video_info.path),
                            "-vframes", "1",
                            "-f", "image2pipe",
                            "-vcodec", "mjpeg",
                            "-"
                        ],
                        capture_output=True,
                        timeout=10
                    )

                    if result.returncode == 0:
                        self._send_response(result.stdout, "image/jpeg")
                    else:
                        self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)

                except Exception as e:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(e))

            def _serve_compare(self) -> None:
                """Serve comparison view."""
                # For now, just serve a simple comparison page
                html = _get_compare_html()
                self._send_response(html, "text/html")

            def _handle_settings(self) -> None:
                """Handle settings update."""
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    settings = json.loads(body.decode())

                    server.update_settings(settings)

                    self._send_json({"success": True})

                except Exception as e:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(e))

            def _handle_render(self) -> None:
                """Handle render request."""
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode())

                    start = data.get("start", 0)
                    duration = data.get("duration", server._config.segment_duration)

                    task = server.render_segment(start, duration)

                    self._send_json({
                        "taskId": task.id,
                        "status": task.status,
                        "progress": task.progress,
                    })

                except Exception as e:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(e))

            def _send_response(self, content: Union[str, bytes], content_type: str) -> None:
                """Send HTTP response."""
                if isinstance(content, str):
                    content = content.encode("utf-8")

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)

            def _send_json(self, data: Any) -> None:
                """Send JSON response."""
                self._send_response(
                    json.dumps(data),
                    "application/json"
                )

            def _serve_file(self, path: Path, content_type: str) -> None:
                """Serve a file."""
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                    self._send_response(content, content_type)
                except Exception as e:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

        return PreviewHandler

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def url(self) -> Optional[str]:
        """Get server URL."""
        if self._running and self._config:
            return f"http://{self._config.host}:{self._config.port}"
        return None


def _get_web_ui_html() -> str:
    """Get the main web UI HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright Preview</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }

        header h1 {
            font-size: 24px;
            font-weight: 600;
            color: #c084fc;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid rgba(34, 197, 94, 0.4);
            border-radius: 20px;
            font-size: 13px;
            color: #22c55e;
        }

        .status-badge.offline {
            background: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.4);
            color: #ef4444;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }

        @media (max-width: 1000px) {
            .main-content { grid-template-columns: 1fr; }
        }

        .preview-section {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            overflow: hidden;
        }

        .video-container {
            position: relative;
            background: #000;
            aspect-ratio: 16/9;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-container video {
            max-width: 100%;
            max-height: 100%;
        }

        .loading-overlay {
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.7);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .loading-overlay.hidden { display: none; }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top-color: #c084fc;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .comparison-slider {
            position: absolute;
            inset: 0;
            overflow: hidden;
        }

        .comparison-slider .before,
        .comparison-slider .after {
            position: absolute;
            inset: 0;
        }

        .comparison-slider .before video,
        .comparison-slider .after video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .comparison-slider .after {
            clip-path: inset(0 50% 0 0);
        }

        .slider-handle {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 4px;
            left: 50%;
            transform: translateX(-50%);
            background: #c084fc;
            cursor: ew-resize;
            z-index: 10;
        }

        .slider-handle::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            background: #c084fc;
            border-radius: 50%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .slider-handle::after {
            content: '\\2194';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 20px;
        }

        .timeline {
            padding: 15px;
            background: rgba(0,0,0,0.3);
        }

        .timeline-track {
            height: 50px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            position: relative;
            cursor: pointer;
            overflow: hidden;
        }

        .timeline-progress {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            background: linear-gradient(90deg, #c084fc 0%, #a855f7 100%);
            width: 0%;
            border-radius: 8px 0 0 8px;
        }

        .timeline-markers {
            position: absolute;
            inset: 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 10px;
        }

        .timeline-marker {
            font-size: 11px;
            color: rgba(255,255,255,0.6);
        }

        .timeline-playhead {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 3px;
            background: white;
            left: 0%;
            cursor: grab;
        }

        .timeline-playhead::before {
            content: '';
            position: absolute;
            top: -5px;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: white;
        }

        .controls {
            display: flex;
            gap: 10px;
            padding: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: #c084fc;
            color: white;
        }

        .btn-primary:hover { background: #a855f7; }

        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: white;
        }

        .btn-secondary:hover { background: rgba(255,255,255,0.2); }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
        }

        .panel-header {
            padding: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-weight: 600;
            font-size: 14px;
        }

        .panel-content {
            padding: 15px;
        }

        .settings-group {
            margin-bottom: 15px;
        }

        .settings-group:last-child { margin-bottom: 0; }

        .settings-group label {
            display: block;
            font-size: 12px;
            color: rgba(255,255,255,0.6);
            margin-bottom: 6px;
        }

        .settings-group select,
        .settings-group input[type="range"] {
            width: 100%;
        }

        select {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            padding: 8px 12px;
            color: white;
            font-size: 13px;
        }

        input[type="range"] {
            -webkit-appearance: none;
            height: 6px;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #c084fc;
            border-radius: 50%;
            cursor: pointer;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .stat-item {
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: 600;
            color: #c084fc;
        }

        .stat-label {
            font-size: 11px;
            color: rgba(255,255,255,0.5);
            margin-top: 4px;
        }

        .progress-bar {
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-bar-fill {
            height: 100%;
            background: #c084fc;
            width: 0%;
            transition: width 0.3s;
        }

        .keyboard-shortcuts {
            font-size: 12px;
            color: rgba(255,255,255,0.5);
        }

        .keyboard-shortcuts kbd {
            display: inline-block;
            padding: 2px 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            font-family: monospace;
            margin-right: 4px;
        }

        .keyboard-shortcuts li {
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>FrameWright Preview</h1>
            <div class="status-badge" id="status-badge">
                <span class="status-dot"></span>
                <span id="status-text">Connecting...</span>
            </div>
        </header>

        <div class="main-content">
            <div class="preview-section">
                <div class="video-container" id="video-container">
                    <div class="comparison-slider" id="comparison-slider">
                        <div class="before">
                            <video id="video-before" muted loop></video>
                        </div>
                        <div class="after" id="after-container">
                            <video id="video-after" muted loop></video>
                        </div>
                        <div class="slider-handle" id="slider-handle"></div>
                    </div>

                    <div class="loading-overlay hidden" id="loading-overlay">
                        <div class="spinner"></div>
                        <span id="loading-text">Loading preview...</span>
                        <div class="progress-bar" style="width: 200px;">
                            <div class="progress-bar-fill" id="loading-progress"></div>
                        </div>
                    </div>
                </div>

                <div class="timeline">
                    <div class="timeline-track" id="timeline-track">
                        <div class="timeline-progress" id="timeline-progress"></div>
                        <div class="timeline-playhead" id="timeline-playhead"></div>
                        <div class="timeline-markers" id="timeline-markers"></div>
                    </div>
                </div>

                <div class="controls">
                    <button class="btn btn-primary" id="btn-play">
                        <span id="play-icon">&#9654;</span> Play
                    </button>
                    <button class="btn btn-secondary" id="btn-compare">
                        &#8596; Compare
                    </button>
                    <button class="btn btn-secondary" id="btn-refresh">
                        &#8635; Refresh
                    </button>
                    <span style="flex: 1;"></span>
                    <span id="time-display" style="font-size: 14px; color: rgba(255,255,255,0.6);">
                        0:00 / 0:00
                    </span>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <div class="panel-header">Settings</div>
                    <div class="panel-content">
                        <div class="settings-group">
                            <label>Quality</label>
                            <select id="quality-select">
                                <option value="low">Low (Fast)</option>
                                <option value="medium" selected>Medium</option>
                                <option value="high">High (Slow)</option>
                            </select>
                        </div>

                        <div class="settings-group">
                            <label>Preview Mode</label>
                            <select id="mode-select">
                                <option value="slider" selected>Slider</option>
                                <option value="sidebyside">Side by Side</option>
                                <option value="toggle">Toggle</option>
                            </select>
                        </div>

                        <div class="settings-group">
                            <label>Enhancement Strength</label>
                            <input type="range" id="strength-slider" min="0" max="100" value="50">
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">Video Info</div>
                    <div class="panel-content">
                        <div class="stat-grid">
                            <div class="stat-item">
                                <div class="stat-value" id="stat-resolution">--</div>
                                <div class="stat-label">Resolution</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" id="stat-fps">--</div>
                                <div class="stat-label">FPS</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" id="stat-duration">--</div>
                                <div class="stat-label">Duration</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value" id="stat-cached">0</div>
                                <div class="stat-label">Cached</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">Keyboard Shortcuts</div>
                    <div class="panel-content">
                        <ul class="keyboard-shortcuts">
                            <li><kbd>Space</kbd> <span>Play/Pause</span></li>
                            <li><kbd>&#8592;</kbd> <span>Seek -5s</span></li>
                            <li><kbd>&#8594;</kbd> <span>Seek +5s</span></li>
                            <li><kbd>C</kbd> <span>Toggle Compare</span></li>
                            <li><kbd>R</kbd> <span>Refresh</span></li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let videoInfo = null;
        let isPlaying = false;
        let currentTime = 0;
        let sliderPosition = 50;
        let isDraggingSlider = false;
        let isDraggingPlayhead = false;

        // Elements
        const videoBefore = document.getElementById('video-before');
        const videoAfter = document.getElementById('video-after');
        const afterContainer = document.getElementById('after-container');
        const sliderHandle = document.getElementById('slider-handle');
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');
        const loadingProgress = document.getElementById('loading-progress');
        const timelineTrack = document.getElementById('timeline-track');
        const timelinePlayhead = document.getElementById('timeline-playhead');
        const timelineProgress = document.getElementById('timeline-progress');
        const timelineMarkers = document.getElementById('timeline-markers');
        const timeDisplay = document.getElementById('time-display');
        const statusBadge = document.getElementById('status-badge');
        const statusText = document.getElementById('status-text');
        const playIcon = document.getElementById('play-icon');

        // Format time as M:SS
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        // Update slider position
        function updateSlider(percent) {
            sliderPosition = Math.max(0, Math.min(100, percent));
            sliderHandle.style.left = sliderPosition + '%';
            afterContainer.style.clipPath = `inset(0 ${100 - sliderPosition}% 0 0)`;
        }

        // Update timeline markers
        function updateTimelineMarkers() {
            if (!videoInfo) return;

            const duration = videoInfo.duration || 0;
            const numMarkers = 5;
            let html = '';

            for (let i = 0; i <= numMarkers; i++) {
                const time = (duration / numMarkers) * i;
                html += `<span class="timeline-marker">${formatTime(time)}</span>`;
            }

            timelineMarkers.innerHTML = html;
        }

        // Load video info
        async function loadVideoInfo() {
            try {
                const response = await fetch('/video-info');
                if (response.ok) {
                    videoInfo = await response.json();

                    // Update stats
                    document.getElementById('stat-resolution').textContent =
                        `${videoInfo.width}x${videoInfo.height}`;
                    document.getElementById('stat-fps').textContent =
                        Math.round(videoInfo.fps);
                    document.getElementById('stat-duration').textContent =
                        formatTime(videoInfo.duration);

                    updateTimelineMarkers();
                    updateTimeDisplay();

                    // Load initial preview segment
                    loadPreviewSegment(0);
                }
            } catch (e) {
                console.error('Failed to load video info:', e);
            }
        }

        // Load preview segment
        async function loadPreviewSegment(startTime) {
            loadingOverlay.classList.remove('hidden');
            loadingText.textContent = 'Rendering preview...';
            loadingProgress.style.width = '0%';

            try {
                const duration = 5; // 5 second segments
                const url = `/preview/${startTime}/${duration}`;

                // Show progress
                loadingProgress.style.width = '30%';

                const response = await fetch(url);
                if (response.ok) {
                    loadingProgress.style.width = '70%';
                    const blob = await response.blob();
                    const videoUrl = URL.createObjectURL(blob);

                    // Load into both videos (for now using same source)
                    videoBefore.src = videoUrl;
                    videoAfter.src = videoUrl;

                    await Promise.all([
                        new Promise(r => videoBefore.onloadeddata = r),
                        new Promise(r => videoAfter.onloadeddata = r)
                    ]);

                    loadingProgress.style.width = '100%';
                    loadingOverlay.classList.add('hidden');

                    if (isPlaying) {
                        videoBefore.play();
                        videoAfter.play();
                    }
                } else {
                    loadingText.textContent = 'Failed to load preview';
                }
            } catch (e) {
                console.error('Failed to load preview:', e);
                loadingText.textContent = 'Error loading preview';
            }
        }

        // Update time display
        function updateTimeDisplay() {
            const current = formatTime(currentTime);
            const total = videoInfo ? formatTime(videoInfo.duration) : '0:00';
            timeDisplay.textContent = `${current} / ${total}`;

            if (videoInfo && videoInfo.duration > 0) {
                const percent = (currentTime / videoInfo.duration) * 100;
                timelinePlayhead.style.left = percent + '%';
                timelineProgress.style.width = percent + '%';
            }
        }

        // Poll server status
        async function pollStatus() {
            try {
                const response = await fetch('/status');
                if (response.ok) {
                    const status = await response.json();

                    statusBadge.classList.remove('offline');
                    statusText.textContent = status.running ? 'Connected' : 'Offline';

                    document.getElementById('stat-cached').textContent = status.cachedSegments;
                }
            } catch (e) {
                statusBadge.classList.add('offline');
                statusText.textContent = 'Disconnected';
            }
        }

        // Toggle play/pause
        function togglePlay() {
            isPlaying = !isPlaying;
            playIcon.textContent = isPlaying ? '\\u275A\\u275A' : '\\u25B6';

            if (isPlaying) {
                videoBefore.play();
                videoAfter.play();
            } else {
                videoBefore.pause();
                videoAfter.pause();
            }
        }

        // Seek video
        function seek(delta) {
            if (!videoInfo) return;

            currentTime = Math.max(0, Math.min(videoInfo.duration, currentTime + delta));
            loadPreviewSegment(currentTime);
            updateTimeDisplay();
        }

        // Event Listeners

        // Slider dragging
        sliderHandle.addEventListener('mousedown', (e) => {
            isDraggingSlider = true;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (isDraggingSlider) {
                const container = document.getElementById('video-container');
                const rect = container.getBoundingClientRect();
                const percent = ((e.clientX - rect.left) / rect.width) * 100;
                updateSlider(percent);
            }

            if (isDraggingPlayhead && videoInfo) {
                const rect = timelineTrack.getBoundingClientRect();
                const percent = ((e.clientX - rect.left) / rect.width) * 100;
                const clampedPercent = Math.max(0, Math.min(100, percent));
                currentTime = (clampedPercent / 100) * videoInfo.duration;
                updateTimeDisplay();
            }
        });

        document.addEventListener('mouseup', () => {
            if (isDraggingPlayhead && videoInfo) {
                loadPreviewSegment(currentTime);
            }
            isDraggingSlider = false;
            isDraggingPlayhead = false;
        });

        // Timeline clicking
        timelineTrack.addEventListener('mousedown', (e) => {
            if (!videoInfo) return;

            isDraggingPlayhead = true;
            const rect = timelineTrack.getBoundingClientRect();
            const percent = ((e.clientX - rect.left) / rect.width) * 100;
            currentTime = (Math.max(0, Math.min(100, percent)) / 100) * videoInfo.duration;
            updateTimeDisplay();
        });

        // Button clicks
        document.getElementById('btn-play').addEventListener('click', togglePlay);
        document.getElementById('btn-refresh').addEventListener('click', () => loadPreviewSegment(currentTime));
        document.getElementById('btn-compare').addEventListener('click', () => {
            window.location.href = '/compare';
        });

        // Video time update
        videoBefore.addEventListener('timeupdate', () => {
            if (videoInfo && !isDraggingPlayhead) {
                // Update based on video position within segment
                const segmentProgress = videoBefore.currentTime;
                // currentTime is segment start, add video progress
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlay();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    seek(-5);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    seek(5);
                    break;
                case 'c':
                case 'C':
                    window.location.href = '/compare';
                    break;
                case 'r':
                case 'R':
                    loadPreviewSegment(currentTime);
                    break;
            }
        });

        // Settings changes
        document.getElementById('quality-select').addEventListener('change', async (e) => {
            await fetch('/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ quality: e.target.value })
            });
            loadPreviewSegment(currentTime);
        });

        // Initialize
        updateSlider(50);
        loadVideoInfo();
        pollStatus();
        setInterval(pollStatus, 5000);
    </script>
</body>
</html>'''


def _get_compare_html() -> str:
    """Get the comparison view HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FrameWright - Compare Settings</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        header h1 { font-size: 24px; color: #c084fc; }

        .back-link {
            color: #c084fc;
            text-decoration: none;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .comparison-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
        }

        .comparison-panel video {
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: contain;
            background: #000;
        }

        .comparison-panel .info {
            padding: 15px;
        }

        .comparison-panel h3 {
            font-size: 16px;
            margin-bottom: 10px;
        }

        .comparison-panel .settings {
            font-size: 12px;
            color: rgba(255,255,255,0.6);
        }

        .settings-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Compare Settings</h1>
            <a href="/" class="back-link">&larr; Back to Preview</a>
        </header>

        <div class="comparison-grid">
            <div class="comparison-panel">
                <video id="video-original" muted loop autoplay></video>
                <div class="info">
                    <h3>Original</h3>
                    <div class="settings">
                        <div class="settings-item">
                            <span>Enhancement</span>
                            <span>None</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="comparison-panel">
                <video id="video-low" muted loop autoplay></video>
                <div class="info">
                    <h3>Low Quality</h3>
                    <div class="settings">
                        <div class="settings-item">
                            <span>Quality</span>
                            <span>Low</span>
                        </div>
                        <div class="settings-item">
                            <span>Speed</span>
                            <span>Fast</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="comparison-panel">
                <video id="video-medium" muted loop autoplay></video>
                <div class="info">
                    <h3>Medium Quality</h3>
                    <div class="settings">
                        <div class="settings-item">
                            <span>Quality</span>
                            <span>Medium</span>
                        </div>
                        <div class="settings-item">
                            <span>Speed</span>
                            <span>Balanced</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="comparison-panel">
                <video id="video-high" muted loop autoplay></video>
                <div class="info">
                    <h3>High Quality</h3>
                    <div class="settings">
                        <div class="settings-item">
                            <span>Quality</span>
                            <span>High</span>
                        </div>
                        <div class="settings-item">
                            <span>Speed</span>
                            <span>Slow</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Load previews for comparison
        async function loadComparisons() {
            const videoInfo = await (await fetch('/video-info')).json();
            const midPoint = videoInfo.duration / 2;

            // Load same segment for all panels
            const url = `/preview/${midPoint}/5`;

            for (const video of document.querySelectorAll('video')) {
                try {
                    const response = await fetch(url);
                    if (response.ok) {
                        const blob = await response.blob();
                        video.src = URL.createObjectURL(blob);
                    }
                } catch (e) {
                    console.error('Failed to load comparison:', e);
                }
            }
        }

        loadComparisons();
    </script>
</body>
</html>'''


# =============================================================================
# Factory Functions
# =============================================================================

def start_preview_server(
    video_path: Union[str, Path],
    port: int = 8080,
    quality: str = "medium",
    auto_open: bool = True
) -> PreviewServer:
    """Start a preview server for the given video.

    This is a convenience function for quickly starting a preview server.

    Args:
        video_path: Path to the video file
        port: Server port (default 8080)
        quality: Preview quality (low/medium/high)
        auto_open: Open browser automatically

    Returns:
        Running PreviewServer instance

    Example:
        >>> server = start_preview_server("input.mp4")
        >>> # ... preview in browser ...
        >>> server.stop()
    """
    config = PreviewConfig(
        port=port,
        quality=quality,
        auto_open_browser=auto_open
    )

    server = PreviewServer()
    server.start(video_path, config)

    return server


def preview_video(
    video_path: Union[str, Path],
    port: int = 8080,
    quality: str = "medium"
) -> None:
    """Preview a video with restoration comparison.

    One-liner that starts the server and opens the browser.
    Blocks until the user presses Ctrl+C.

    Args:
        video_path: Path to the video file
        port: Server port (default 8080)
        quality: Preview quality (low/medium/high)

    Example:
        >>> preview_video("input.mp4")
        # Opens browser at http://localhost:8080
        # Press Ctrl+C to stop
    """
    server = start_preview_server(video_path, port, quality, auto_open=True)

    try:
        print(f"\nPreview server running at {server.url}")
        print("Press Ctrl+C to stop...\n")

        while server.is_running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()


__all__ = [
    "PreviewConfig",
    "PreviewQuality",
    "PreviewServer",
    "VideoInfo",
    "RenderTask",
    "ServerStatus",
    "SegmentCache",
    "RenderQueue",
    "start_preview_server",
    "preview_video",
]
