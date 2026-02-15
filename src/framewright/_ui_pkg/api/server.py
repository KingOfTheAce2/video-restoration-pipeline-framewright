"""REST API server for FrameWright.

This module provides a JSON-based REST API for video restoration
using only Python standard library (http.server).
"""

import hashlib
import http.server
import json
import logging
import secrets
import socketserver
import threading
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class APIConfig:
    """Configuration for the API server."""

    host: str = "127.0.0.1"
    port: int = 8081
    debug: bool = False

    # Authentication
    require_auth: bool = False
    api_keys: List[str] = field(default_factory=list)

    # Rate limiting
    enable_rate_limit: bool = True
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 60  # seconds

    # CORS
    allow_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # References
    job_store: Optional[Any] = None
    model_manager: Optional[Any] = None
    preset_registry: Optional[Any] = None

    def __post_init__(self):
        """Generate a default API key if auth required but none provided."""
        if self.require_auth and not self.api_keys:
            key = secrets.token_urlsafe(32)
            self.api_keys.append(key)
            logger.info(f"Generated API key: {key}")


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """Check if a request is allowed.

        Args:
            client_id: Client identifier (IP or API key)

        Returns:
            Tuple of (allowed, remaining requests)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Clean old requests
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]

            # Check limit
            current = len(self._requests[client_id])
            if current >= self.max_requests:
                return False, 0

            # Record request
            self._requests[client_id].append(now)
            return True, self.max_requests - current - 1

    def reset(self, client_id: str):
        """Reset rate limit for a client."""
        with self._lock:
            self._requests.pop(client_id, None)


# =============================================================================
# OpenAPI Specification
# =============================================================================


OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "FrameWright API",
        "description": "REST API for video restoration using Real-ESRGAN and related technologies",
        "version": "1.0.0",
        "contact": {
            "name": "FrameWright",
        },
    },
    "servers": [
        {
            "url": "http://localhost:8081/api/v1",
            "description": "Local development server",
        }
    ],
    "security": [
        {"ApiKeyAuth": []},
        {"BearerAuth": []},
    ],
    "paths": {
        "/restore": {
            "post": {
                "summary": "Submit a restoration job",
                "description": "Submit a new video restoration job",
                "operationId": "submitRestoreJob",
                "tags": ["Jobs"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/RestoreRequest"},
                        }
                    },
                },
                "responses": {
                    "201": {
                        "description": "Job created successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/JobResponse"},
                            }
                        },
                    },
                    "400": {"description": "Invalid request"},
                    "401": {"description": "Unauthorized"},
                    "429": {"description": "Rate limit exceeded"},
                },
            }
        },
        "/jobs": {
            "get": {
                "summary": "List all jobs",
                "description": "Get a list of all restoration jobs",
                "operationId": "listJobs",
                "tags": ["Jobs"],
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "description": "Filter by job status",
                        "schema": {"type": "string", "enum": ["pending", "processing", "completed", "failed", "cancelled"]},
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of jobs to return",
                        "schema": {"type": "integer", "default": 50},
                    },
                    {
                        "name": "offset",
                        "in": "query",
                        "description": "Number of jobs to skip",
                        "schema": {"type": "integer", "default": 0},
                    },
                ],
                "responses": {
                    "200": {
                        "description": "List of jobs",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/JobList"},
                            }
                        },
                    },
                },
            }
        },
        "/jobs/{job_id}": {
            "get": {
                "summary": "Get job details",
                "description": "Get detailed information about a specific job",
                "operationId": "getJob",
                "tags": ["Jobs"],
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Job details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Job"},
                            }
                        },
                    },
                    "404": {"description": "Job not found"},
                },
            },
            "delete": {
                "summary": "Cancel a job",
                "description": "Cancel a pending or processing job",
                "operationId": "cancelJob",
                "tags": ["Jobs"],
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "200": {"description": "Job cancelled"},
                    "404": {"description": "Job not found"},
                    "409": {"description": "Job cannot be cancelled"},
                },
            },
        },
        "/presets": {
            "get": {
                "summary": "List available presets",
                "description": "Get a list of available restoration presets",
                "operationId": "listPresets",
                "tags": ["Configuration"],
                "responses": {
                    "200": {
                        "description": "List of presets",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PresetList"},
                            }
                        },
                    },
                },
            }
        },
        "/models": {
            "get": {
                "summary": "List available models",
                "description": "Get a list of available AI models",
                "operationId": "listModels",
                "tags": ["Configuration"],
                "responses": {
                    "200": {
                        "description": "List of models",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ModelList"},
                            }
                        },
                    },
                },
            }
        },
        "/hardware": {
            "get": {
                "summary": "Get hardware information",
                "description": "Get information about available hardware resources",
                "operationId": "getHardware",
                "tags": ["System"],
                "responses": {
                    "200": {
                        "description": "Hardware information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HardwareInfo"},
                            }
                        },
                    },
                },
            }
        },
        "/analyze": {
            "post": {
                "summary": "Analyze a video",
                "description": "Analyze a video file and get restoration recommendations",
                "operationId": "analyzeVideo",
                "tags": ["Analysis"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AnalyzeRequest"},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Analysis results",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AnalysisResult"},
                            }
                        },
                    },
                },
            }
        },
    },
    "components": {
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
            },
        },
        "schemas": {
            "RestoreRequest": {
                "type": "object",
                "required": ["input_path"],
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input video file",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for output video file (optional)",
                    },
                    "preset": {
                        "type": "string",
                        "description": "Restoration preset to use",
                        "default": "balanced",
                    },
                    "scale": {
                        "type": "integer",
                        "description": "Upscaling factor",
                        "default": 4,
                        "enum": [2, 4],
                    },
                    "model": {
                        "type": "string",
                        "description": "Specific model to use (optional)",
                    },
                    "options": {
                        "type": "object",
                        "description": "Additional processing options",
                    },
                },
            },
            "JobResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "job_id": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
            "Job": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "state": {"type": "string"},
                    "total_frames": {"type": "integer"},
                    "frames_processed": {"type": "integer"},
                    "progress_percent": {"type": "number"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"},
                    "error_message": {"type": "string"},
                },
            },
            "JobList": {
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Job"},
                    },
                    "total": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
            },
            "Preset": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "recommended_for": {"type": "string"},
                },
            },
            "PresetList": {
                "type": "object",
                "properties": {
                    "presets": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Preset"},
                    },
                },
            },
            "Model": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "scale": {"type": "integer"},
                    "loaded": {"type": "boolean"},
                },
            },
            "ModelList": {
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Model"},
                    },
                },
            },
            "HardwareInfo": {
                "type": "object",
                "properties": {
                    "cpu_percent": {"type": "number"},
                    "ram_used_gb": {"type": "number"},
                    "ram_total_gb": {"type": "number"},
                    "gpu_name": {"type": "string"},
                    "vram_used_gb": {"type": "number"},
                    "vram_total_gb": {"type": "number"},
                    "gpu_temp": {"type": "integer"},
                },
            },
            "AnalyzeRequest": {
                "type": "object",
                "required": ["input_path"],
                "properties": {
                    "input_path": {"type": "string"},
                },
            },
            "AnalysisResult": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "analysis": {
                        "type": "object",
                        "properties": {
                            "content_type": {"type": "string"},
                            "degradation": {"type": "string"},
                            "resolution": {"type": "string"},
                            "frame_count": {"type": "integer"},
                            "fps": {"type": "number"},
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


# =============================================================================
# API Request Handler
# =============================================================================


class APIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the REST API."""

    # Class-level configuration (set by server)
    config: APIConfig = None
    job_store: Any = None
    model_manager: Any = None
    preset_registry: Any = None
    rate_limiter: RateLimiter = None

    # API version prefix
    API_PREFIX = "/api/v1"

    def log_message(self, format: str, *args):
        """Override to use Python logging."""
        if self.config and self.config.debug:
            logger.debug(f"{self.address_string()} - {format % args}")

    def _get_client_id(self) -> str:
        """Get client identifier for rate limiting."""
        # Use API key if present, otherwise IP
        api_key = self.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{hashlib.md5(api_key.encode()).hexdigest()[:8]}"
        return f"ip:{self.client_address[0]}"

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit.

        Returns:
            True if allowed, False if rate limited
        """
        if not self.config or not self.config.enable_rate_limit:
            return True

        if not self.rate_limiter:
            return True

        client_id = self._get_client_id()
        allowed, remaining = self.rate_limiter.is_allowed(client_id)

        if not allowed:
            self.send_json(
                {
                    "error": "Rate limit exceeded",
                    "retry_after": self.config.rate_limit_window,
                },
                429,
            )
            return False

        return True

    def _check_auth(self) -> bool:
        """Check if request is authenticated.

        Returns:
            True if authenticated or auth not required
        """
        if not self.config or not self.config.require_auth:
            return True

        # Check X-API-Key header
        api_key = self.headers.get("X-API-Key")
        if api_key and api_key in self.config.api_keys:
            return True

        # Check Authorization: Bearer header
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            if token in self.config.api_keys:
                return True

        return False

    def send_json(self, data: Any, status: int = 200):
        """Send a JSON response.

        Args:
            data: Data to serialize
            status: HTTP status code
        """
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _add_cors_headers(self):
        """Add CORS headers if enabled."""
        if self.config and self.config.allow_cors:
            origin = self.headers.get("Origin", "*")
            if "*" in self.config.cors_origins or origin in self.config.cors_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
            self.send_header("Access-Control-Max-Age", "86400")

    def _parse_path(self) -> Tuple[str, Dict[str, str]]:
        """Parse the request path.

        Returns:
            Tuple of (path, query_params)
        """
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Remove API prefix
        if path.startswith(self.API_PREFIX):
            path = path[len(self.API_PREFIX):]

        # Parse query parameters
        query = urllib.parse.parse_qs(parsed.query)
        # Flatten single-value params
        params = {k: v[0] if len(v) == 1 else v for k, v in query.items()}

        return path, params

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body.

        Returns:
            Parsed JSON data or None
        """
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}

        try:
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._add_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if not self._check_rate_limit():
            return

        if not self._check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        path, params = self._parse_path()

        try:
            # Route handling
            if path == "" or path == "/":
                self._handle_root()
            elif path == "/openapi.json" or path == "/swagger.json":
                self._handle_openapi()
            elif path == "/jobs":
                self._handle_list_jobs(params)
            elif path.startswith("/jobs/"):
                job_id = path[6:]
                self._handle_get_job(job_id)
            elif path == "/presets":
                self._handle_list_presets()
            elif path == "/models":
                self._handle_list_models()
            elif path == "/hardware":
                self._handle_hardware()
            elif path == "/health":
                self._handle_health()
            else:
                self.send_json({"error": "Not found"}, 404)

        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        """Handle POST requests."""
        if not self._check_rate_limit():
            return

        if not self._check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        path, params = self._parse_path()

        # Parse JSON body
        data = self._read_json_body()
        if data is None:
            self.send_json({"error": "Invalid JSON body"}, 400)
            return

        try:
            if path == "/restore":
                self._handle_restore(data)
            elif path == "/analyze":
                self._handle_analyze(data)
            else:
                self.send_json({"error": "Not found"}, 404)

        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self.send_json({"error": str(e)}, 500)

    def do_DELETE(self):
        """Handle DELETE requests."""
        if not self._check_rate_limit():
            return

        if not self._check_auth():
            self.send_json({"error": "Unauthorized"}, 401)
            return

        path, params = self._parse_path()

        try:
            if path.startswith("/jobs/"):
                job_id = path[6:]
                self._handle_cancel_job(job_id)
            else:
                self.send_json({"error": "Not found"}, 404)

        except Exception as e:
            logger.error(f"Error handling DELETE {path}: {e}")
            self.send_json({"error": str(e)}, 500)

    # -------------------------------------------------------------------------
    # Route Handlers
    # -------------------------------------------------------------------------

    def _handle_root(self):
        """Handle root endpoint - API info."""
        self.send_json({
            "name": "FrameWright API",
            "version": "1.0.0",
            "documentation": f"{self.API_PREFIX}/openapi.json",
            "endpoints": {
                "restore": f"POST {self.API_PREFIX}/restore",
                "jobs": f"GET {self.API_PREFIX}/jobs",
                "job": f"GET {self.API_PREFIX}/jobs/{{job_id}}",
                "cancel": f"DELETE {self.API_PREFIX}/jobs/{{job_id}}",
                "presets": f"GET {self.API_PREFIX}/presets",
                "models": f"GET {self.API_PREFIX}/models",
                "hardware": f"GET {self.API_PREFIX}/hardware",
                "analyze": f"POST {self.API_PREFIX}/analyze",
            },
        })

    def _handle_openapi(self):
        """Return OpenAPI specification."""
        self.send_json(OPENAPI_SPEC)

    def _handle_health(self):
        """Health check endpoint."""
        self.send_json({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        })

    def _handle_restore(self, data: Dict[str, Any]):
        """Submit a restoration job."""
        input_path = data.get("input_path")
        if not input_path:
            self.send_json({"error": "input_path is required"}, 400)
            return

        if not self.job_store:
            self.send_json({"error": "Job store not configured"}, 503)
            return

        try:
            job_id = self.job_store.create_job(
                input_path=input_path,
                output_path=data.get("output_path"),
                preset=data.get("preset", "balanced"),
                scale=data.get("scale", 4),
                model=data.get("model"),
                options=data.get("options", {}),
            )

            self.send_json({
                "success": True,
                "job_id": job_id,
                "message": "Job submitted successfully",
            }, 201)

        except Exception as e:
            logger.error(f"Error creating job: {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_list_jobs(self, params: Dict[str, str]):
        """List all jobs."""
        status_filter = params.get("status")
        limit = int(params.get("limit", 50))
        offset = int(params.get("offset", 0))

        jobs = []
        total = 0

        if self.job_store:
            try:
                all_jobs = self.job_store.list_jobs()

                # Filter by status
                if status_filter:
                    all_jobs = [
                        j for j in all_jobs
                        if str(getattr(j, "state", "")).lower() == status_filter.lower()
                    ]

                total = len(all_jobs)

                # Paginate
                all_jobs = all_jobs[offset:offset + limit]

                jobs = [self._job_to_dict(j) for j in all_jobs]

            except Exception as e:
                logger.error(f"Error listing jobs: {e}")

        self.send_json({
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "offset": offset,
        })

    def _handle_get_job(self, job_id: str):
        """Get job details."""
        if not self.job_store:
            self.send_json({"error": "Job store not configured"}, 503)
            return

        try:
            job = self.job_store.get_job(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
                return

            result = self._job_to_dict(job)

            # Include frame info if available
            try:
                frames = self.job_store.get_frames(job_id)
                result["frames_info"] = {
                    "total": len(frames),
                    "processed": sum(1 for f in frames if str(getattr(f, "state", "")) == "completed"),
                    "failed": sum(1 for f in frames if str(getattr(f, "state", "")) == "failed"),
                }
            except Exception:
                pass

            self.send_json(result)

        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_cancel_job(self, job_id: str):
        """Cancel a job."""
        if not self.job_store:
            self.send_json({"error": "Job store not configured"}, 503)
            return

        try:
            # Check job exists
            job = self.job_store.get_job(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
                return

            # Check if cancellable
            state = str(getattr(job, "state", ""))
            if state in ("completed", "failed", "cancelled"):
                self.send_json({"error": f"Job cannot be cancelled (state: {state})"}, 409)
                return

            success = self.job_store.cancel_job(job_id)
            if success:
                self.send_json({
                    "success": True,
                    "message": "Job cancelled",
                })
            else:
                self.send_json({"error": "Failed to cancel job"}, 500)

        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            self.send_json({"error": str(e)}, 500)

    def _handle_list_presets(self):
        """List available presets."""
        presets = []

        if self.preset_registry:
            try:
                preset_list = self.preset_registry.list_presets()
                presets = [
                    {
                        "name": p.name,
                        "description": p.description,
                        "recommended_for": p.recommended_for,
                    }
                    for p in preset_list
                ]
            except Exception as e:
                logger.error(f"Error listing presets: {e}")

        # Fallback default presets
        if not presets:
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
                        "size_mb": getattr(m, "size_mb", None),
                    }
                    for m in model_list
                ]
            except Exception as e:
                logger.error(f"Error listing models: {e}")

        # Fallback default models
        if not models:
            models = [
                {"name": "realesr-general-x4v3", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
                {"name": "realesr-animevideov3", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
                {"name": "RealESRGAN_x4plus", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
                {"name": "RealESRGAN_x4plus_anime_6B", "type": "Real-ESRGAN", "scale": 4, "loaded": False},
            ]

        self.send_json({"models": models})

    def _handle_hardware(self):
        """Get hardware information."""
        info = self._get_hardware_info()
        self.send_json(info)

    def _handle_analyze(self, data: Dict[str, Any]):
        """Analyze a video file."""
        input_path = data.get("input_path")
        if not input_path:
            self.send_json({"error": "input_path is required"}, 400)
            return

        # Check file exists
        if not Path(input_path).exists():
            self.send_json({"error": "File not found"}, 404)
            return

        try:
            # Try to use smart analyzer
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
            # Basic analysis fallback
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
    # Helper Methods
    # -------------------------------------------------------------------------

    def _job_to_dict(self, job: Any) -> Dict[str, Any]:
        """Convert job object to dictionary."""
        state = getattr(job, "state", None)
        if hasattr(state, "value"):
            state = state.value
        else:
            state = str(state) if state else "unknown"

        total_frames = getattr(job, "total_frames", 0) or 0
        frames_processed = getattr(job, "frames_processed", 0) or 0

        return {
            "job_id": getattr(job, "job_id", str(job)),
            "input_path": getattr(job, "input_path", ""),
            "output_path": getattr(job, "output_path", ""),
            "state": state,
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "frames_failed": getattr(job, "frames_failed", 0) or 0,
            "progress_percent": (frames_processed / total_frames * 100) if total_frames > 0 else 0,
            "avg_frame_time_ms": getattr(job, "avg_frame_time_ms", 0),
            "created_at": getattr(job, "created_at", datetime.now()).isoformat() if hasattr(job, "created_at") else None,
            "updated_at": getattr(job, "updated_at", datetime.now()).isoformat() if hasattr(job, "updated_at") else None,
            "error_message": getattr(job, "error_message", None),
        }

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        import platform

        info = {
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

        try:
            import psutil

            info["cpu_percent"] = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            info["ram_used_gb"] = mem.used / (1024**3)
            info["ram_total_gb"] = mem.total / (1024**3)
        except ImportError:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["vram_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
                props = torch.cuda.get_device_properties(0)
                info["vram_total_gb"] = props.total_memory / (1024**3)
        except ImportError:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            info["gpu_temp"] = temp
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return info


# =============================================================================
# Threaded Server
# =============================================================================


class ThreadedAPIServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that handles each request in a new thread."""

    daemon_threads = True
    allow_reuse_address = True


# =============================================================================
# API Server
# =============================================================================


class APIServer:
    """REST API server for FrameWright.

    Provides a JSON-based API for submitting and managing
    video restoration jobs.
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        job_store: Optional[Any] = None,
        model_manager: Optional[Any] = None,
        preset_registry: Optional[Any] = None,
    ):
        """Initialize the API server.

        Args:
            config: Server configuration
            job_store: Optional job store for persistence
            model_manager: Optional model manager
            preset_registry: Optional preset registry
        """
        self.config = config or APIConfig()
        self.job_store = job_store or self.config.job_store
        self.model_manager = model_manager or self.config.model_manager
        self.preset_registry = preset_registry or self.config.preset_registry

        self._server: Optional[ThreadedAPIServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_requests,
            window_seconds=self.config.rate_limit_window,
        )

    def start(self, blocking: bool = True) -> None:
        """Start the API server.

        Args:
            blocking: If True, blocks until server stops
        """
        # Configure handler
        APIHandler.config = self.config
        APIHandler.job_store = self.job_store
        APIHandler.model_manager = self.model_manager
        APIHandler.preset_registry = self.preset_registry
        APIHandler.rate_limiter = self.rate_limiter

        # Create server
        self._server = ThreadedAPIServer(
            (self.config.host, self.config.port),
            APIHandler,
        )

        self._running = True

        url = f"http://{self.config.host}:{self.config.port}"
        logger.info(f"API server starting at {url}")
        logger.info(f"API documentation: {url}/api/v1/openapi.json")

        if self.config.require_auth:
            logger.info(f"API keys: {self.config.api_keys}")

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
        """Stop the API server."""
        self._running = False

        if self._server:
            self._server.shutdown()
            self._server = None

        logger.info("API server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def api_url(self) -> str:
        """Get the API base URL."""
        return f"{self.url}/api/v1"


# =============================================================================
# Convenience Functions
# =============================================================================


def start_api_server(
    host: str = "127.0.0.1",
    port: int = 8081,
    job_store: Optional[Any] = None,
    model_manager: Optional[Any] = None,
    require_auth: bool = False,
    api_keys: Optional[List[str]] = None,
    blocking: bool = True,
) -> Optional[APIServer]:
    """Start the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        job_store: Optional job store
        model_manager: Optional model manager
        require_auth: Whether to require API key authentication
        api_keys: List of valid API keys
        blocking: If True, blocks until stopped

    Returns:
        APIServer instance if non-blocking, None if blocking
    """
    config = APIConfig(
        host=host,
        port=port,
        require_auth=require_auth,
        api_keys=api_keys or [],
    )

    server = APIServer(
        config=config,
        job_store=job_store,
        model_manager=model_manager,
    )

    server.start(blocking=blocking)

    return server if not blocking else None
