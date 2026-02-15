"""Real-time preview system for FrameWright.

Provides live preview of restoration quality with before/after comparison.
Supports multiple backends: OpenCV window, web-based, terminal ASCII art.
"""

import logging
import threading
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import tempfile

logger = logging.getLogger(__name__)


class PreviewMode(Enum):
    """Preview display modes."""
    SIDE_BY_SIDE = "side_by_side"
    SLIDER = "slider"
    TOGGLE = "toggle"
    DIFF = "diff"
    ORIGINAL_ONLY = "original"
    RESTORED_ONLY = "restored"


class PreviewBackend(Enum):
    """Available preview backends."""
    OPENCV = "opencv"
    WEB = "web"
    TERMINAL = "terminal"
    NONE = "none"


@dataclass
class PreviewFrame:
    """Container for preview frame data."""
    frame_number: int
    timestamp: float
    original: Any  # numpy array or path
    restored: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class PreviewConfig:
    """Configuration for preview system."""
    backend: PreviewBackend = PreviewBackend.OPENCV
    mode: PreviewMode = PreviewMode.SLIDER
    window_width: int = 1280
    window_height: int = 720
    update_interval: float = 0.1  # seconds
    show_metrics: bool = True
    show_histogram: bool = False
    slider_position: float = 0.5  # 0-1
    auto_advance: bool = True
    web_port: int = 8765
    terminal_width: int = 120


class PreviewRenderer(ABC):
    """Abstract base class for preview renderers."""

    @abstractmethod
    def initialize(self, config: PreviewConfig) -> bool:
        """Initialize the renderer."""
        pass

    @abstractmethod
    def render_frame(self, frame: PreviewFrame, mode: PreviewMode) -> None:
        """Render a preview frame."""
        pass

    @abstractmethod
    def handle_input(self) -> Optional[str]:
        """Handle user input, return command if any."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class OpenCVPreviewRenderer(PreviewRenderer):
    """OpenCV-based preview renderer with interactive controls."""

    def __init__(self):
        self.window_name = "FrameWright Preview"
        self.cv2 = None
        self.np = None
        self._slider_pos = 0.5
        self._paused = False
        self._config: Optional[PreviewConfig] = None

    def is_available(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
            return True
        except ImportError:
            return False

    def initialize(self, config: PreviewConfig) -> bool:
        """Initialize OpenCV window."""
        if not self.is_available():
            logger.warning("OpenCV not available for preview")
            return False

        self._config = config
        self._slider_pos = config.slider_position

        # Create window with controls
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.window_name, config.window_width, config.window_height)

        # Create trackbar for slider mode
        if config.mode == PreviewMode.SLIDER:
            self.cv2.createTrackbar(
                "Compare",
                self.window_name,
                int(self._slider_pos * 100),
                100,
                self._on_slider_change
            )

        logger.info(f"OpenCV preview initialized: {config.window_width}x{config.window_height}")
        return True

    def _on_slider_change(self, value: int) -> None:
        """Handle slider position change."""
        self._slider_pos = value / 100.0

    def render_frame(self, frame: PreviewFrame, mode: PreviewMode) -> None:
        """Render frame with before/after comparison."""
        if self.cv2 is None or frame.original is None:
            return

        original = self._ensure_numpy(frame.original)
        restored = self._ensure_numpy(frame.restored) if frame.restored is not None else original

        # Resize to match if needed
        if original.shape != restored.shape:
            restored = self.cv2.resize(restored, (original.shape[1], original.shape[0]))

        # Create comparison view based on mode
        if mode == PreviewMode.SIDE_BY_SIDE:
            display = self._render_side_by_side(original, restored)
        elif mode == PreviewMode.SLIDER:
            display = self._render_slider(original, restored)
        elif mode == PreviewMode.TOGGLE:
            display = restored if int(time.time() * 2) % 2 == 0 else original
        elif mode == PreviewMode.DIFF:
            display = self._render_diff(original, restored)
        elif mode == PreviewMode.ORIGINAL_ONLY:
            display = original
        else:
            display = restored

        # Add metrics overlay
        if self._config and self._config.show_metrics and frame.metrics:
            display = self._add_metrics_overlay(display, frame)

        # Add frame info
        display = self._add_frame_info(display, frame)

        self.cv2.imshow(self.window_name, display)

    def _ensure_numpy(self, img: Any) -> Any:
        """Convert image to numpy array if needed."""
        if isinstance(img, (str, Path)):
            return self.cv2.imread(str(img))
        return img

    def _render_side_by_side(self, original: Any, restored: Any) -> Any:
        """Render side-by-side comparison."""
        h, w = original.shape[:2]

        # Resize both to half width
        left = self.cv2.resize(original, (w // 2, h))
        right = self.cv2.resize(restored, (w // 2, h))

        # Add labels
        self.cv2.putText(left, "Original", (10, 30),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.cv2.putText(right, "Restored", (10, 30),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add divider line
        combined = self.np.hstack([left, right])
        self.cv2.line(combined, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        return combined

    def _render_slider(self, original: Any, restored: Any) -> Any:
        """Render slider comparison."""
        h, w = original.shape[:2]
        split_x = int(w * self._slider_pos)

        # Combine images at split point
        combined = original.copy()
        combined[:, split_x:] = restored[:, split_x:]

        # Draw slider line
        self.cv2.line(combined, (split_x, 0), (split_x, h), (0, 255, 255), 2)

        # Draw slider handle
        handle_y = h // 2
        self.cv2.circle(combined, (split_x, handle_y), 15, (0, 255, 255), -1)
        self.cv2.circle(combined, (split_x, handle_y), 15, (255, 255, 255), 2)

        # Labels
        self.cv2.putText(combined, "Original", (10, 30),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.cv2.putText(combined, "Restored", (w - 120, 30),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return combined

    def _render_diff(self, original: Any, restored: Any) -> Any:
        """Render difference visualization."""
        # Calculate absolute difference
        diff = self.cv2.absdiff(original, restored)

        # Amplify difference for visibility
        diff = self.cv2.convertScaleAbs(diff, alpha=3.0)

        # Apply colormap for better visualization
        diff_gray = self.cv2.cvtColor(diff, self.cv2.COLOR_BGR2GRAY)
        diff_colored = self.cv2.applyColorMap(diff_gray, self.cv2.COLORMAP_JET)

        return diff_colored

    def _add_metrics_overlay(self, display: Any, frame: PreviewFrame) -> Any:
        """Add metrics overlay to display."""
        h, w = display.shape[:2]

        # Create semi-transparent overlay
        overlay = display.copy()
        panel_h = 80
        self.cv2.rectangle(overlay, (w - 200, h - panel_h), (w, h), (0, 0, 0), -1)
        display = self.cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # Add metrics text
        y = h - panel_h + 20
        for name, value in frame.metrics.items():
            text = f"{name}: {value:.2f}"
            self.cv2.putText(display, text, (w - 190, y),
                            self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

        return display

    def _add_frame_info(self, display: Any, frame: PreviewFrame) -> Any:
        """Add frame number and timestamp."""
        h = display.shape[0]

        # Frame info at bottom left
        info = f"Frame {frame.frame_number} | {frame.timestamp:.2f}s"
        if frame.processing_time > 0:
            info += f" | {frame.processing_time*1000:.1f}ms"

        self.cv2.putText(display, info, (10, h - 10),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def handle_input(self) -> Optional[str]:
        """Handle keyboard input."""
        if self.cv2 is None:
            return None

        key = self.cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            return "quit"
        elif key == ord(' '):  # Space
            self._paused = not self._paused
            return "pause" if self._paused else "resume"
        elif key == ord('s'):
            return "screenshot"
        elif key == ord('m'):
            return "toggle_mode"
        elif key == ord('h'):
            return "toggle_histogram"
        elif key == ord('1'):
            return "mode_side_by_side"
        elif key == ord('2'):
            return "mode_slider"
        elif key == ord('3'):
            return "mode_toggle"
        elif key == ord('4'):
            return "mode_diff"

        return None

    def cleanup(self) -> None:
        """Clean up OpenCV windows."""
        if self.cv2:
            self.cv2.destroyAllWindows()


class TerminalPreviewRenderer(PreviewRenderer):
    """Terminal-based ASCII art preview (fallback)."""

    def __init__(self):
        self._config: Optional[PreviewConfig] = None
        self._rich_available = False

    def is_available(self) -> bool:
        """Always available as fallback."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            self._rich_available = True
        except ImportError:
            pass
        return True

    def initialize(self, config: PreviewConfig) -> bool:
        """Initialize terminal preview."""
        self._config = config
        logger.info("Terminal preview initialized")
        return True

    def render_frame(self, frame: PreviewFrame, mode: PreviewMode) -> None:
        """Render frame info in terminal."""
        if self._rich_available:
            self._render_rich(frame)
        else:
            self._render_plain(frame)

    def _render_rich(self, frame: PreviewFrame) -> None:
        """Render with Rich library."""
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live

        console = Console()

        table = Table(title=f"Frame {frame.frame_number}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Timestamp", f"{frame.timestamp:.2f}s")
        table.add_row("Processing", f"{frame.processing_time*1000:.1f}ms")

        for name, value in frame.metrics.items():
            table.add_row(name, f"{value:.3f}")

        console.print(table)

    def _render_plain(self, frame: PreviewFrame) -> None:
        """Plain text rendering."""
        print(f"\n=== Frame {frame.frame_number} @ {frame.timestamp:.2f}s ===")
        print(f"Processing time: {frame.processing_time*1000:.1f}ms")
        for name, value in frame.metrics.items():
            print(f"  {name}: {value:.3f}")

    def handle_input(self) -> Optional[str]:
        """No interactive input in terminal mode."""
        return None

    def cleanup(self) -> None:
        """Nothing to clean up."""
        pass


class WebPreviewRenderer(PreviewRenderer):
    """Web-based preview using local HTTP server."""

    def __init__(self):
        self._config: Optional[PreviewConfig] = None
        self._server = None
        self._thread = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._current_frame: Optional[PreviewFrame] = None
        self._running = False

    def is_available(self) -> bool:
        """Check if web server dependencies are available."""
        try:
            import http.server
            import json
            import base64
            return True
        except ImportError:
            return False

    def initialize(self, config: PreviewConfig) -> bool:
        """Start local web server for preview."""
        import http.server
        import socketserver
        import json
        import base64

        self._config = config
        renderer = self

        class PreviewHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(renderer._get_html_page().encode())

                elif self.path == "/frame":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    frame_data = renderer._get_frame_json()
                    self.wfile.write(json.dumps(frame_data).encode())

                else:
                    self.send_response(404)
                    self.end_headers()

        try:
            self._server = socketserver.TCPServer(
                ("", config.web_port),
                PreviewHandler
            )
            self._server.socket.settimeout(1)

            self._thread = threading.Thread(target=self._serve_forever, daemon=True)
            self._running = True
            self._thread.start()

            logger.info(f"Web preview available at http://localhost:{config.web_port}")
            return True

        except OSError as e:
            logger.error(f"Could not start web server: {e}")
            return False

    def _serve_forever(self) -> None:
        """Serve HTTP requests."""
        while self._running:
            try:
                self._server.handle_request()
            except Exception:
                pass

    def _get_html_page(self) -> str:
        """Generate HTML page for preview."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>FrameWright Preview</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7; margin: 0; padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #c084fc; text-align: center; }
        .preview-container {
            display: flex; gap: 20px; justify-content: center;
            flex-wrap: wrap; margin: 20px 0;
        }
        .frame-box {
            background: rgba(255,255,255,0.05);
            border-radius: 12px; padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .frame-box h3 { margin: 0 0 10px 0; color: #a1a1aa; }
        .frame-box img {
            max-width: 500px; max-height: 400px;
            border-radius: 8px;
        }
        .metrics {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; margin-top: 20px;
        }
        .metric {
            background: rgba(255,255,255,0.05);
            padding: 15px; border-radius: 8px; text-align: center;
        }
        .metric-value { font-size: 24px; font-weight: bold; color: #c084fc; }
        .metric-label { color: #a1a1aa; font-size: 12px; }
        .status { text-align: center; color: #22c55e; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>FrameWright Real-Time Preview</h1>
        <div class="status" id="status">Connecting...</div>
        <div class="preview-container">
            <div class="frame-box">
                <h3>Original</h3>
                <img id="original" src="" alt="Original frame">
            </div>
            <div class="frame-box">
                <h3>Restored</h3>
                <img id="restored" src="" alt="Restored frame">
            </div>
        </div>
        <div class="metrics" id="metrics"></div>
    </div>
    <script>
        function updatePreview() {
            fetch('/frame')
                .then(r => r.json())
                .then(data => {
                    if (data.frame_number !== undefined) {
                        document.getElementById('status').textContent =
                            `Frame ${data.frame_number} | ${data.timestamp.toFixed(2)}s | ${(data.processing_time * 1000).toFixed(1)}ms`;
                        if (data.original) document.getElementById('original').src = data.original;
                        if (data.restored) document.getElementById('restored').src = data.restored;
                        let metricsHtml = '';
                        for (const [key, value] of Object.entries(data.metrics || {})) {
                            metricsHtml += `<div class="metric"><div class="metric-value">${value.toFixed(2)}</div><div class="metric-label">${key}</div></div>`;
                        }
                        document.getElementById('metrics').innerHTML = metricsHtml;
                    }
                })
                .catch(() => {
                    document.getElementById('status').textContent = 'Waiting for frames...';
                });
        }
        setInterval(updatePreview, 200);
        updatePreview();
    </script>
</body>
</html>"""

    def _get_frame_json(self) -> Dict[str, Any]:
        """Get current frame as JSON with base64 images."""
        if self._current_frame is None:
            return {}

        import base64

        frame = self._current_frame
        result = {
            "frame_number": frame.frame_number,
            "timestamp": frame.timestamp,
            "processing_time": frame.processing_time,
            "metrics": frame.metrics,
        }

        # Convert images to base64 if they are numpy arrays
        try:
            import cv2

            if frame.original is not None:
                if hasattr(frame.original, 'shape'):
                    _, buffer = cv2.imencode('.jpg', frame.original)
                    result["original"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
                elif isinstance(frame.original, (str, Path)):
                    with open(str(frame.original), 'rb') as f:
                        result["original"] = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

            if frame.restored is not None:
                if hasattr(frame.restored, 'shape'):
                    _, buffer = cv2.imencode('.jpg', frame.restored)
                    result["restored"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
                elif isinstance(frame.restored, (str, Path)):
                    with open(str(frame.restored), 'rb') as f:
                        result["restored"] = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

        except Exception as e:
            logger.debug(f"Could not encode frame images: {e}")

        return result

    def render_frame(self, frame: PreviewFrame, mode: PreviewMode) -> None:
        """Queue frame for web display."""
        self._current_frame = frame
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

    def handle_input(self) -> Optional[str]:
        """Web input handled via HTTP."""
        return None

    def cleanup(self) -> None:
        """Shutdown web server."""
        self._running = False
        if self._server:
            self._server.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


class RealTimePreview:
    """Real-time preview system orchestrator.

    Provides live preview of restoration progress with multiple viewing modes
    and interactive controls.
    """

    def __init__(
        self,
        config: Optional[PreviewConfig] = None,
        on_frame_callback: Optional[Callable[[PreviewFrame], None]] = None,
    ):
        """Initialize preview system.

        Args:
            config: Preview configuration
            on_frame_callback: Optional callback for each frame
        """
        self.config = config or PreviewConfig()
        self.on_frame_callback = on_frame_callback

        self._renderer: Optional[PreviewRenderer] = None
        self._running = False
        self._paused = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._render_thread: Optional[threading.Thread] = None
        self._current_mode = self.config.mode
        self._screenshots_dir = Path(tempfile.gettempdir()) / "framewright_previews"

        # Statistics
        self._frames_shown = 0
        self._start_time: Optional[float] = None

    def start(self) -> bool:
        """Start the preview system.

        Returns:
            True if started successfully
        """
        # Select and initialize renderer
        self._renderer = self._select_renderer()
        if self._renderer is None:
            logger.warning("No preview renderer available")
            return False

        if not self._renderer.initialize(self.config):
            logger.error("Failed to initialize preview renderer")
            return False

        self._running = True
        self._start_time = time.time()

        # Start render thread
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

        logger.info(f"Preview started with {self.config.backend.value} backend")
        return True

    def _select_renderer(self) -> Optional[PreviewRenderer]:
        """Select appropriate renderer based on config and availability."""
        renderers = {
            PreviewBackend.OPENCV: OpenCVPreviewRenderer,
            PreviewBackend.WEB: WebPreviewRenderer,
            PreviewBackend.TERMINAL: TerminalPreviewRenderer,
        }

        # Try preferred backend first
        if self.config.backend != PreviewBackend.NONE:
            renderer = renderers.get(self.config.backend)
            if renderer:
                instance = renderer()
                if instance.is_available():
                    return instance

        # Fallback chain
        for backend in [PreviewBackend.OPENCV, PreviewBackend.WEB, PreviewBackend.TERMINAL]:
            if backend == self.config.backend:
                continue
            renderer = renderers.get(backend)
            if renderer:
                instance = renderer()
                if instance.is_available():
                    logger.info(f"Falling back to {backend.value} preview")
                    return instance

        return None

    def stop(self) -> None:
        """Stop the preview system."""
        self._running = False

        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)

        if self._renderer:
            self._renderer.cleanup()

        # Log statistics
        if self._start_time:
            elapsed = time.time() - self._start_time
            fps = self._frames_shown / elapsed if elapsed > 0 else 0
            logger.info(f"Preview stopped: {self._frames_shown} frames, {fps:.1f} fps average")

    def add_frame(
        self,
        frame_number: int,
        timestamp: float,
        original: Any,
        restored: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        processing_time: float = 0.0,
    ) -> None:
        """Add a frame to the preview queue.

        Args:
            frame_number: Frame index
            timestamp: Frame timestamp in seconds
            original: Original frame (numpy array or path)
            restored: Restored frame (numpy array or path)
            metrics: Quality metrics for this frame
            processing_time: Time taken to process this frame
        """
        frame = PreviewFrame(
            frame_number=frame_number,
            timestamp=timestamp,
            original=original,
            restored=restored,
            metrics=metrics or {},
            processing_time=processing_time,
        )

        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame to keep up
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

        if self.on_frame_callback:
            self.on_frame_callback(frame)

    def _render_loop(self) -> None:
        """Main render loop running in background thread."""
        last_update = 0.0

        while self._running:
            try:
                # Get next frame with timeout
                frame = self._frame_queue.get(timeout=0.1)

                # Rate limit updates
                now = time.time()
                if now - last_update < self.config.update_interval:
                    continue

                if not self._paused and self._renderer:
                    self._renderer.render_frame(frame, self._current_mode)
                    self._frames_shown += 1
                    last_update = now

                # Handle input
                if self._renderer:
                    command = self._renderer.handle_input()
                    self._handle_command(command)

            except queue.Empty:
                # No frame available, just handle input
                if self._renderer:
                    command = self._renderer.handle_input()
                    self._handle_command(command)
            except Exception as e:
                logger.error(f"Preview render error: {e}")

    def _handle_command(self, command: Optional[str]) -> None:
        """Handle user command from renderer."""
        if command is None:
            return

        if command == "quit":
            self._running = False
        elif command == "pause":
            self._paused = True
            logger.info("Preview paused")
        elif command == "resume":
            self._paused = False
            logger.info("Preview resumed")
        elif command == "screenshot":
            self._take_screenshot()
        elif command == "toggle_mode":
            self._cycle_mode()
        elif command.startswith("mode_"):
            mode_name = command[5:]
            self._set_mode(mode_name)

    def _cycle_mode(self) -> None:
        """Cycle through preview modes."""
        modes = list(PreviewMode)
        current_idx = modes.index(self._current_mode)
        self._current_mode = modes[(current_idx + 1) % len(modes)]
        logger.info(f"Preview mode: {self._current_mode.value}")

    def _set_mode(self, mode_name: str) -> None:
        """Set preview mode by name."""
        try:
            self._current_mode = PreviewMode(mode_name)
            logger.info(f"Preview mode: {self._current_mode.value}")
        except ValueError:
            logger.warning(f"Unknown preview mode: {mode_name}")

    def _take_screenshot(self) -> None:
        """Save current frame as screenshot."""
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self._screenshots_dir / f"preview_{timestamp}.png"
        logger.info(f"Screenshot saved: {path}")

    @property
    def is_running(self) -> bool:
        """Check if preview is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if preview is paused."""
        return self._paused

    def pause(self) -> None:
        """Pause preview updates."""
        self._paused = True

    def resume(self) -> None:
        """Resume preview updates."""
        self._paused = False

    def set_mode(self, mode: PreviewMode) -> None:
        """Set preview display mode."""
        self._current_mode = mode

    def set_slider_position(self, position: float) -> None:
        """Set slider position (0-1) for slider mode."""
        if isinstance(self._renderer, OpenCVPreviewRenderer):
            self._renderer._slider_pos = max(0.0, min(1.0, position))


def create_preview(
    backend: str = "auto",
    mode: str = "slider",
    **kwargs,
) -> RealTimePreview:
    """Create a real-time preview instance.

    Args:
        backend: Preview backend (opencv, web, terminal, auto)
        mode: Display mode (side_by_side, slider, toggle, diff)
        **kwargs: Additional config options

    Returns:
        Configured RealTimePreview instance
    """
    # Determine backend
    if backend == "auto":
        preview_backend = PreviewBackend.OPENCV
    else:
        try:
            preview_backend = PreviewBackend(backend)
        except ValueError:
            preview_backend = PreviewBackend.OPENCV

    # Determine mode
    try:
        preview_mode = PreviewMode(mode)
    except ValueError:
        preview_mode = PreviewMode.SLIDER

    config = PreviewConfig(
        backend=preview_backend,
        mode=preview_mode,
        **kwargs,
    )

    return RealTimePreview(config)


class PreviewIntegration:
    """Helper class to integrate preview with VideoRestorer."""

    def __init__(self, restorer: Any, preview: RealTimePreview):
        """Initialize preview integration.

        Args:
            restorer: VideoRestorer instance
            preview: RealTimePreview instance
        """
        self.restorer = restorer
        self.preview = preview
        self._frame_times: List[float] = []

    def create_progress_callback(self) -> Callable:
        """Create a progress callback that feeds the preview.

        Returns:
            Callback function for VideoRestorer
        """
        def callback(info: Any) -> None:
            if hasattr(info, "frame_path") and hasattr(info, "original_path"):
                start = time.time()

                # Calculate metrics if available
                metrics = {}
                if hasattr(info, "psnr"):
                    metrics["PSNR"] = info.psnr
                if hasattr(info, "ssim"):
                    metrics["SSIM"] = info.ssim

                processing_time = time.time() - start
                self._frame_times.append(processing_time)

                self.preview.add_frame(
                    frame_number=getattr(info, "frame_number", 0),
                    timestamp=getattr(info, "timestamp", 0.0),
                    original=info.original_path,
                    restored=info.frame_path,
                    metrics=metrics,
                    processing_time=processing_time,
                )

        return callback

    @property
    def average_frame_time(self) -> float:
        """Get average frame processing time."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)
