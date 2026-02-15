"""Dashboard HTTP server using Python standard library.

This module provides a web dashboard server built on http.server
without requiring any external dependencies like Flask.
"""

import base64
import hashlib
import http.server
import json
import logging
import os
import platform
import queue
import re
import secrets
import socketserver
import struct
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .templates import (
    render_dashboard_page,
    render_error_page,
    render_login_page,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DashboardConfig:
    """Configuration for the dashboard server."""

    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False

    # Authentication
    require_auth: bool = False
    api_key: str = ""
    session_timeout: int = 3600  # 1 hour

    # CORS settings
    allow_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # WebSocket settings
    enable_websocket: bool = True
    ws_ping_interval: int = 30

    # Refresh settings
    auto_refresh_seconds: int = 5

    # SSL settings (optional)
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # Job store reference
    job_store: Optional[Any] = None

    # Model manager reference
    model_manager: Optional[Any] = None

    def __post_init__(self):
        """Generate API key if not provided and auth is required."""
        if self.require_auth and not self.api_key:
            self.api_key = secrets.token_urlsafe(32)
            logger.info(f"Generated API key: {self.api_key}")


# =============================================================================
# WebSocket Support
# =============================================================================


class WebSocketConnection:
    """Represents a WebSocket connection."""

    def __init__(self, socket, address: Tuple[str, int]):
        self.socket = socket
        self.address = address
        self.connected = True
        self.id = secrets.token_hex(8)
        self.created_at = datetime.now()
        self.last_ping = datetime.now()

    def send(self, message: Union[str, dict]) -> bool:
        """Send a message over the WebSocket.

        Args:
            message: String or dict to send

        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            if isinstance(message, dict):
                message = json.dumps(message)

            data = message.encode("utf-8")

            # WebSocket frame format
            header = bytearray()

            # FIN bit + text opcode
            header.append(0x81)

            # Payload length
            if len(data) < 126:
                header.append(len(data))
            elif len(data) < 65536:
                header.append(126)
                header.extend(struct.pack(">H", len(data)))
            else:
                header.append(127)
                header.extend(struct.pack(">Q", len(data)))

            self.socket.sendall(bytes(header) + data)
            return True

        except Exception as e:
            logger.debug(f"WebSocket send error: {e}")
            self.connected = False
            return False

    def receive(self) -> Optional[str]:
        """Receive a message from the WebSocket.

        Returns:
            Decoded message or None if connection closed
        """
        try:
            # Read frame header
            header = self.socket.recv(2)
            if not header or len(header) < 2:
                self.connected = False
                return None

            fin = header[0] & 0x80
            opcode = header[0] & 0x0F
            masked = header[1] & 0x80
            payload_len = header[1] & 0x7F

            # Handle close frame
            if opcode == 0x08:
                self.connected = False
                return None

            # Handle ping
            if opcode == 0x09:
                self._send_pong()
                return None

            # Extended payload length
            if payload_len == 126:
                ext = self.socket.recv(2)
                payload_len = struct.unpack(">H", ext)[0]
            elif payload_len == 127:
                ext = self.socket.recv(8)
                payload_len = struct.unpack(">Q", ext)[0]

            # Read mask key if present
            mask_key = None
            if masked:
                mask_key = self.socket.recv(4)

            # Read payload
            payload = bytearray()
            while len(payload) < payload_len:
                chunk = self.socket.recv(min(4096, payload_len - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)

            # Unmask if needed
            if mask_key:
                payload = bytearray(
                    b ^ mask_key[i % 4] for i, b in enumerate(payload)
                )

            return payload.decode("utf-8")

        except Exception as e:
            logger.debug(f"WebSocket receive error: {e}")
            self.connected = False
            return None

    def _send_pong(self):
        """Send a pong frame."""
        try:
            self.socket.sendall(bytes([0x8A, 0]))
        except Exception:
            pass

    def close(self):
        """Close the WebSocket connection."""
        if self.connected:
            try:
                # Send close frame
                self.socket.sendall(bytes([0x88, 0]))
            except Exception:
                pass
            self.connected = False


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self._lock = threading.Lock()
        self._broadcast_queue: queue.Queue = queue.Queue()
        self._running = False
        self._broadcast_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the broadcast thread."""
        self._running = True
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self._broadcast_thread.start()

    def stop(self):
        """Stop the broadcast thread and close all connections."""
        self._running = False
        with self._lock:
            for conn in self.connections.values():
                conn.close()
            self.connections.clear()

    def add(self, conn: WebSocketConnection):
        """Add a new connection."""
        with self._lock:
            self.connections[conn.id] = conn
        logger.debug(f"WebSocket connected: {conn.id}")

    def remove(self, conn_id: str):
        """Remove a connection."""
        with self._lock:
            if conn_id in self.connections:
                self.connections[conn_id].close()
                del self.connections[conn_id]
        logger.debug(f"WebSocket disconnected: {conn_id}")

    def broadcast(self, message: Union[str, dict]):
        """Queue a message for broadcast to all connections."""
        self._broadcast_queue.put(message)

    def _broadcast_worker(self):
        """Worker thread for broadcasting messages."""
        while self._running:
            try:
                message = self._broadcast_queue.get(timeout=1.0)
                self._do_broadcast(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    def _do_broadcast(self, message: Union[str, dict]):
        """Actually broadcast to all connections."""
        with self._lock:
            dead_connections = []
            for conn_id, conn in self.connections.items():
                if not conn.send(message):
                    dead_connections.append(conn_id)

            # Clean up dead connections
            for conn_id in dead_connections:
                del self.connections[conn_id]


# =============================================================================
# Request Handler
# =============================================================================


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    # Class-level references (set by server)
    config: DashboardConfig = None
    job_store: Any = None
    model_manager: Any = None
    ws_manager: WebSocketManager = None
    sessions: Dict[str, datetime] = {}
    _sessions_lock = threading.Lock()

    def log_message(self, format: str, *args):
        """Override to use Python logging."""
        if self.config and self.config.debug:
            logger.debug(f"{self.address_string()} - {format % args}")

    def send_json(self, data: Any, status: int = 200):
        """Send a JSON response.

        Args:
            data: Data to serialize as JSON
            status: HTTP status code
        """
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, status: int = 200):
        """Send an HTML response.

        Args:
            html: HTML content
            status: HTTP status code
        """
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, code: int, message: str):
        """Send an error response.

        Args:
            code: HTTP status code
            message: Error message
        """
        # Check if client expects JSON
        accept = self.headers.get("Accept", "")
        if "application/json" in accept:
            self.send_json({"error": message}, code)
        else:
            self.send_html(render_error_page(code, message), code)

    def _add_cors_headers(self):
        """Add CORS headers if enabled."""
        if self.config and self.config.allow_cors:
            origin = self.headers.get("Origin", "*")
            if "*" in self.config.cors_origins or origin in self.config.cors_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")

    def _check_auth(self) -> bool:
        """Check if request is authenticated.

        Returns:
            True if authenticated or auth not required
        """
        if not self.config or not self.config.require_auth:
            return True

        # Check API key header
        api_key = self.headers.get("X-API-Key")
        if api_key and api_key == self.config.api_key:
            return True

        # Check Authorization header
        auth = self.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth[7:]
            if token == self.config.api_key:
                return True

        # Check session cookie
        cookies = self.headers.get("Cookie", "")
        session_id = None
        for cookie in cookies.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("session="):
                session_id = cookie[8:]
                break

        if session_id:
            with self._sessions_lock:
                if session_id in self.sessions:
                    created = self.sessions[session_id]
                    if (datetime.now() - created).seconds < self.config.session_timeout:
                        return True
                    else:
                        del self.sessions[session_id]

        return False

    def _create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        with self._sessions_lock:
            self.sessions[session_id] = datetime.now()
        return session_id

    def do_OPTIONS(self):
        """Handle OPTIONS requests (CORS preflight)."""
        self.send_response(204)
        self._add_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        # Parse URL
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # WebSocket upgrade
        if path == "/ws" and self._is_websocket_upgrade():
            self._handle_websocket_upgrade()
            return

        # Check authentication for protected routes
        if path != "/login" and not self._check_auth():
            if "application/json" in self.headers.get("Accept", ""):
                self.send_json({"error": "Unauthorized"}, 401)
            else:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
            return

        # Route handling
        try:
            if path == "/":
                self._handle_dashboard()
            elif path == "/login":
                self._handle_login_page()
            elif path == "/jobs":
                self._handle_list_jobs()
            elif path.startswith("/jobs/"):
                job_id = path[6:]
                self._handle_get_job(job_id)
            elif path == "/system":
                self._handle_system_info()
            elif path == "/models":
                self._handle_list_models()
            elif path == "/presets":
                self._handle_list_presets()
            elif path == "/hardware":
                self._handle_hardware_info()
            else:
                self.send_error_response(404, "Not Found")

        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self.send_error_response(500, str(e))

    def do_POST(self):
        """Handle POST requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Login doesn't require auth
        if path == "/login":
            self._handle_login()
            return

        # Check authentication
        if not self._check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        try:
            # Read body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""

            # Parse JSON body
            data = {}
            if body:
                try:
                    data = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    self.send_json({"error": "Invalid JSON"}, 400)
                    return

            # Route handling
            if path == "/jobs":
                self._handle_submit_job(data)
            elif path == "/analyze":
                self._handle_analyze_video(data)
            else:
                self.send_error_response(404, "Not Found")

        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self.send_error_response(500, str(e))

    def do_DELETE(self):
        """Handle DELETE requests."""
        if not self._check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path.startswith("/jobs/"):
                job_id = path[6:]
                self._handle_cancel_job(job_id)
            else:
                self.send_error_response(404, "Not Found")

        except Exception as e:
            logger.error(f"Error handling DELETE {path}: {e}")
            self.send_error_response(500, str(e))

    # -------------------------------------------------------------------------
    # Route Handlers
    # -------------------------------------------------------------------------

    def _handle_dashboard(self):
        """Render the main dashboard page."""
        config_dict = {
            "title": "FrameWright Dashboard",
            "auto_refresh": self.config.auto_refresh_seconds if self.config else 5,
        }
        html = render_dashboard_page(config_dict)
        self.send_html(html)

    def _handle_login_page(self):
        """Render the login page."""
        html = render_login_page()
        self.send_html(html)

    def _handle_login(self):
        """Handle login POST request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Parse form data
        data = urllib.parse.parse_qs(body.decode("utf-8"))
        api_key = data.get("api_key", [""])[0]

        if api_key == self.config.api_key:
            session_id = self._create_session()
            self.send_response(302)
            self.send_header("Set-Cookie", f"session={session_id}; HttpOnly; Path=/")
            self.send_header("Location", "/")
            self.end_headers()
        else:
            self.send_html(render_login_page() + "<p style='color: red;'>Invalid API key</p>")

    def _handle_list_jobs(self):
        """List all jobs."""
        jobs = []

        if self.job_store:
            try:
                job_list = self.job_store.list_jobs()
                jobs = [self._job_to_dict(j) for j in job_list]
            except Exception as e:
                logger.error(f"Error listing jobs: {e}")

        self.send_json({"jobs": jobs, "count": len(jobs)})

    def _handle_get_job(self, job_id: str):
        """Get details for a specific job."""
        if not self.job_store:
            self.send_json({"error": "No job store configured"}, 500)
            return

        try:
            job = self.job_store.get_job(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
                return

            result = self._job_to_dict(job)

            # Include frame details if available
            try:
                frames = self.job_store.get_frames(job_id)
                result["frames"] = [self._frame_to_dict(f) for f in frames[:100]]
                result["frame_count"] = len(frames)
            except Exception:
                result["frames"] = []
                result["frame_count"] = 0

            self.send_json(result)

        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_submit_job(self, data: Dict[str, Any]):
        """Submit a new restoration job."""
        input_path = data.get("input_path")
        if not input_path:
            self.send_json({"error": "input_path is required"}, 400)
            return

        if not self.job_store:
            self.send_json({"error": "No job store configured"}, 500)
            return

        try:
            # Create job with provided options
            job_id = self.job_store.create_job(
                input_path=input_path,
                output_path=data.get("output_path"),
                preset=data.get("preset", "balanced"),
                scale=data.get("scale", 4),
                options=data.get("options", {}),
            )

            self.send_json({
                "success": True,
                "job_id": job_id,
                "message": "Job submitted successfully",
            })

            # Broadcast update
            if self.ws_manager:
                self.ws_manager.broadcast({
                    "type": "job_created",
                    "data": {"job_id": job_id},
                })

        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_cancel_job(self, job_id: str):
        """Cancel a job."""
        if not self.job_store:
            self.send_json({"error": "No job store configured"}, 500)
            return

        try:
            success = self.job_store.cancel_job(job_id)
            if success:
                self.send_json({"success": True, "message": "Job cancelled"})

                # Broadcast update
                if self.ws_manager:
                    self.ws_manager.broadcast({
                        "type": "job_cancelled",
                        "data": {"job_id": job_id},
                    })
            else:
                self.send_json({"error": "Failed to cancel job"}, 400)

        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_system_info(self):
        """Get system resource information."""
        info = self._get_system_status()
        self.send_json(info)

    def _handle_list_models(self):
        """List available models."""
        models = []

        if self.model_manager:
            try:
                model_list = self.model_manager.list_models()
                models = [
                    {
                        "name": m.name,
                        "type": m.type,
                        "scale": m.scale,
                        "loaded": m.loaded,
                        "size_mb": m.size_mb,
                    }
                    for m in model_list
                ]
            except Exception as e:
                logger.error(f"Error listing models: {e}")

        # Fallback: return common models
        if not models:
            models = [
                {"name": "realesr-general-x4v3", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
                {"name": "realesr-animevideov3", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
                {"name": "RealESRGAN_x4plus", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
                {"name": "RealESRGAN_x4plus_anime_6B", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
            ]

        self.send_json({"models": models, "count": len(models)})

    def _handle_list_presets(self):
        """List available presets."""
        presets = [
            {
                "name": "fast",
                "description": "Fastest processing with basic quality",
                "recommended_for": "Quick previews and testing",
            },
            {
                "name": "balanced",
                "description": "Good balance of speed and quality",
                "recommended_for": "General use",
            },
            {
                "name": "quality",
                "description": "High quality with longer processing time",
                "recommended_for": "Final output",
            },
            {
                "name": "ultra",
                "description": "Maximum quality, slowest processing",
                "recommended_for": "Professional work",
            },
        ]

        self.send_json({"presets": presets})

    def _handle_hardware_info(self):
        """Get detailed hardware information."""
        try:
            from ...hardware import check_hardware

            report = check_hardware()
            self.send_json(report.to_dict())
        except ImportError:
            # Fallback if hardware module not available
            self.send_json(self._get_system_status())

    def _handle_analyze_video(self, data: Dict[str, Any]):
        """Analyze a video file."""
        input_path = data.get("input_path")
        if not input_path:
            self.send_json({"error": "input_path is required"}, 400)
            return

        try:
            # Try to use the analyzer if available
            from ...ui.auto_detect import analyze_video_smart

            result = analyze_video_smart(input_path)
            self.send_json({
                "success": True,
                "analysis": {
                    "content_type": result.content_profile.value if result.content_profile else "unknown",
                    "degradation": result.degradation_profile.value if result.degradation_profile else "unknown",
                    "resolution": f"{result.width}x{result.height}",
                    "frame_count": result.frame_count,
                    "fps": result.fps,
                    "duration": result.duration,
                    "recommendations": result.recommendations,
                },
            })
        except ImportError:
            # Fallback basic analysis
            self.send_json({
                "success": True,
                "analysis": {
                    "input_path": input_path,
                    "message": "Detailed analysis not available",
                },
            })
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    # -------------------------------------------------------------------------
    # WebSocket Support
    # -------------------------------------------------------------------------

    def _is_websocket_upgrade(self) -> bool:
        """Check if this is a WebSocket upgrade request."""
        return (
            self.headers.get("Upgrade", "").lower() == "websocket"
            and "websocket" in self.headers.get("Connection", "").lower()
        )

    def _handle_websocket_upgrade(self):
        """Handle WebSocket upgrade handshake."""
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error_response(400, "Missing WebSocket key")
            return

        # Generate accept key
        magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        accept = base64.b64encode(
            hashlib.sha1((key + magic).encode()).digest()
        ).decode()

        # Send upgrade response
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        # Create WebSocket connection
        conn = WebSocketConnection(self.request, self.client_address)

        if self.ws_manager:
            self.ws_manager.add(conn)

        # Handle WebSocket messages
        try:
            while conn.connected:
                message = conn.receive()
                if message is None:
                    break

                # Handle ping/pong
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        conn.send({"type": "pong"})
                    elif data.get("type") == "subscribe":
                        # Client subscribing to updates
                        pass
                except json.JSONDecodeError:
                    pass

        finally:
            if self.ws_manager:
                self.ws_manager.remove(conn.id)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _job_to_dict(self, job: Any) -> Dict[str, Any]:
        """Convert a job object to a dictionary."""
        return {
            "job_id": getattr(job, "job_id", str(job)),
            "input_path": getattr(job, "input_path", ""),
            "output_path": getattr(job, "output_path", ""),
            "state": getattr(job, "state", {}).value if hasattr(getattr(job, "state", {}), "value") else str(getattr(job, "state", "unknown")),
            "total_frames": getattr(job, "total_frames", 0),
            "frames_processed": getattr(job, "frames_processed", 0),
            "frames_failed": getattr(job, "frames_failed", 0),
            "progress_percent": (
                getattr(job, "frames_processed", 0) / getattr(job, "total_frames", 1) * 100
            )
            if getattr(job, "total_frames", 0) > 0
            else 0,
            "current_frame": getattr(job, "current_frame", 0),
            "avg_frame_time_ms": getattr(job, "avg_frame_time_ms", 0),
            "created_at": (
                getattr(job, "created_at", datetime.now()).isoformat()
                if hasattr(job, "created_at")
                else None
            ),
            "updated_at": (
                getattr(job, "updated_at", datetime.now()).isoformat()
                if hasattr(job, "updated_at")
                else None
            ),
            "error_message": getattr(job, "error_message", None),
        }

    def _frame_to_dict(self, frame: Any) -> Dict[str, Any]:
        """Convert a frame object to a dictionary."""
        return {
            "frame_number": getattr(frame, "frame_number", 0),
            "state": getattr(frame, "state", {}).value if hasattr(getattr(frame, "state", {}), "value") else str(getattr(frame, "state", "unknown")),
            "input_path": getattr(frame, "input_path", ""),
            "output_path": getattr(frame, "output_path", ""),
            "error_message": getattr(frame, "error_message", None),
        }

    def _get_system_status(self) -> Dict[str, Any]:
        """Get system resource status."""
        status = {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "gpu_name": "N/A",
            "gpu_temp": 0,
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }

        # CPU and RAM via psutil
        try:
            import psutil

            status["cpu_percent"] = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            status["ram_used_gb"] = mem.used / (1024**3)
            status["ram_total_gb"] = mem.total / (1024**3)
        except ImportError:
            pass

        # GPU info via torch
        try:
            import torch

            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["vram_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
                props = torch.cuda.get_device_properties(0)
                status["vram_total_gb"] = props.total_memory / (1024**3)
        except ImportError:
            pass

        # GPU temperature via pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            status["gpu_temp"] = temp
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return status


# =============================================================================
# Threaded Server
# =============================================================================


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that handles each request in a new thread."""

    daemon_threads = True
    allow_reuse_address = True


# =============================================================================
# Dashboard Server
# =============================================================================


class DashboardServer:
    """Main dashboard server class.

    Provides a web interface for monitoring and managing FrameWright
    restoration jobs using only Python standard library.
    """

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        job_store: Optional[Any] = None,
        model_manager: Optional[Any] = None,
    ):
        """Initialize the dashboard server.

        Args:
            config: Server configuration
            job_store: Optional job store for persistence
            model_manager: Optional model manager
        """
        self.config = config or DashboardConfig()
        self.job_store = job_store or self.config.job_store
        self.model_manager = model_manager or self.config.model_manager

        self._server: Optional[ThreadedHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # WebSocket manager
        self.ws_manager = WebSocketManager()

        # Update broadcast thread
        self._update_thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = True) -> None:
        """Start the dashboard server.

        Args:
            blocking: If True, blocks until server stops
        """
        # Configure handler class
        DashboardHandler.config = self.config
        DashboardHandler.job_store = self.job_store
        DashboardHandler.model_manager = self.model_manager
        DashboardHandler.ws_manager = self.ws_manager

        # Create server
        self._server = ThreadedHTTPServer(
            (self.config.host, self.config.port),
            DashboardHandler,
        )

        # Start WebSocket manager
        if self.config.enable_websocket:
            self.ws_manager.start()

        # Start update broadcast thread
        self._running = True
        self._update_thread = threading.Thread(
            target=self._broadcast_updates, daemon=True
        )
        self._update_thread.start()

        url = f"http://{self.config.host}:{self.config.port}"
        logger.info(f"Dashboard server starting at {url}")

        if self.config.require_auth:
            logger.info(f"API Key: {self.config.api_key}")

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                logger.info("Server interrupted")
            finally:
                self.stop()
        else:
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
            )
            self._server_thread.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False

        if self._server:
            self._server.shutdown()
            self._server = None

        self.ws_manager.stop()
        logger.info("Dashboard server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.config.host}:{self.config.port}"

    def broadcast(self, message: Union[str, dict]) -> None:
        """Broadcast a message to all WebSocket clients.

        Args:
            message: Message to broadcast
        """
        self.ws_manager.broadcast(message)

    def _broadcast_updates(self) -> None:
        """Periodically broadcast system updates."""
        while self._running:
            try:
                # Get system status
                status = DashboardHandler._get_system_status(None)

                # Broadcast system update
                self.ws_manager.broadcast({
                    "type": "system_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat(),
                })

                # Broadcast job updates
                if self.job_store:
                    try:
                        jobs = self.job_store.list_jobs()
                        job_data = [
                            {
                                "job_id": getattr(j, "job_id", str(j)),
                                "state": str(getattr(j, "state", "unknown")),
                                "progress": (
                                    getattr(j, "frames_processed", 0)
                                    / max(getattr(j, "total_frames", 1), 1)
                                    * 100
                                ),
                            }
                            for j in jobs
                        ]
                        self.ws_manager.broadcast({
                            "type": "jobs_list",
                            "data": job_data,
                        })
                    except Exception:
                        pass

                time.sleep(self.config.auto_refresh_seconds)

            except Exception as e:
                logger.error(f"Error in update broadcast: {e}")
                time.sleep(5)


# =============================================================================
# Convenience Functions
# =============================================================================


def start_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    job_store: Optional[Any] = None,
    model_manager: Optional[Any] = None,
    require_auth: bool = False,
    api_key: Optional[str] = None,
    blocking: bool = True,
    open_browser: bool = True,
) -> Optional[DashboardServer]:
    """Start the dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
        job_store: Optional job store
        model_manager: Optional model manager
        require_auth: Whether to require API key authentication
        api_key: API key for authentication (generated if not provided)
        blocking: If True, blocks until stopped
        open_browser: If True, opens browser automatically

    Returns:
        DashboardServer instance if non-blocking, None if blocking
    """
    config = DashboardConfig(
        host=host,
        port=port,
        require_auth=require_auth,
        api_key=api_key or "",
    )

    server = DashboardServer(
        config=config,
        job_store=job_store,
        model_manager=model_manager,
    )

    if open_browser:
        import webbrowser

        def open_delayed():
            time.sleep(0.5)
            webbrowser.open(server.url)

        threading.Thread(target=open_delayed, daemon=True).start()

    server.start(blocking=blocking)

    return server if not blocking else None
