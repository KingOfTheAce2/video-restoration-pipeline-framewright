"""REST API for FrameWright video restoration.

This module provides a JSON-based REST API for video restoration
using only Python standard library (http.server).

Features:
- RESTful endpoints for job management
- OpenAPI/Swagger specification
- API key authentication (optional)
- Rate limiting
- CORS support
- Python client library

API Endpoints:
- POST /api/v1/restore - Submit restoration job
- GET /api/v1/jobs - List all jobs
- GET /api/v1/jobs/{id} - Get job details
- DELETE /api/v1/jobs/{id} - Cancel job
- GET /api/v1/presets - List available presets
- GET /api/v1/models - List available models
- GET /api/v1/hardware - Get hardware info
- POST /api/v1/analyze - Analyze video

Server Example:

    >>> from framewright.ui.api import APIServer, APIConfig
    >>>
    >>> # Start with default settings
    >>> server = APIServer()
    >>> server.start()  # Blocks until stopped
    >>>
    >>> # Start with custom configuration
    >>> config = APIConfig(
    ...     host="0.0.0.0",
    ...     port=8081,
    ...     require_auth=True,
    ...     api_keys=["your-api-key"],
    ... )
    >>> server = APIServer(config=config)
    >>> server.start(blocking=False)

Client Example:

    >>> from framewright.ui.api import FrameWrightClient
    >>>
    >>> # Connect to API
    >>> client = FrameWrightClient(
    ...     base_url="http://localhost:8081",
    ...     api_key="your-api-key",
    ... )
    >>>
    >>> # Submit a job
    >>> job_id = client.restore("/path/to/video.mp4", preset="quality")
    >>>
    >>> # Wait for completion
    >>> status = client.wait_for_completion(job_id)
    >>> print(f"Output: {status.output_path}")
    >>>
    >>> # Or check status manually
    >>> status = client.get_status(job_id)
    >>> print(f"Progress: {status.progress_percent:.1f}%")

Quick restore function:

    >>> from framewright.ui.api import quick_restore
    >>>
    >>> # One-liner restoration
    >>> result = quick_restore("video.mp4", preset="balanced", wait=True)
    >>> print(f"Done: {result.output_path}")
"""

from .server import (
    APIConfig,
    APIServer,
    APIHandler,
    RateLimiter,
    OPENAPI_SPEC,
    start_api_server,
)

from .client import (
    FrameWrightClient,
    JobStatus,
    Model,
    Preset,
    HardwareInfo,
    AnalysisResult,
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ConnectionError,
    create_client,
    quick_restore,
)

__all__ = [
    # Server
    "APIConfig",
    "APIServer",
    "APIHandler",
    "RateLimiter",
    "OPENAPI_SPEC",
    "start_api_server",
    # Client
    "FrameWrightClient",
    "JobStatus",
    "Model",
    "Preset",
    "HardwareInfo",
    "AnalysisResult",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "ConnectionError",
    "create_client",
    "quick_restore",
]
