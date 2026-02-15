"""Real-time preview server for live video restoration preview.

This module provides a web-based preview interface for video restoration with:
- Before/after comparison slider
- Segment caching for instant replay
- Background rendering queue
- Real-time progress updates
- Side-by-side comparison mode

Quick Start:
    >>> from framewright.ui.preview import preview_video
    >>> preview_video("input.mp4")  # Opens browser at localhost:8080

Advanced Usage:
    >>> from framewright.ui.preview import PreviewServer, PreviewConfig
    >>> config = PreviewConfig(port=8080, quality="high", auto_open_browser=True)
    >>> server = PreviewServer()
    >>> server.start("input.mp4", config)
    >>> # ... use the preview ...
    >>> server.stop()

Factory Functions:
    start_preview_server: Start server and return instance
    preview_video: One-liner that starts server and blocks
"""

from .server import (
    # Configuration
    PreviewConfig,
    PreviewQuality,
    # Main classes
    PreviewServer,
    VideoInfo,
    RenderTask,
    ServerStatus,
    SegmentCache,
    RenderQueue,
    # Factory functions
    start_preview_server,
    preview_video,
)

__all__ = [
    # Configuration
    "PreviewConfig",
    "PreviewQuality",
    # Main classes
    "PreviewServer",
    "VideoInfo",
    "RenderTask",
    "ServerStatus",
    "SegmentCache",
    "RenderQueue",
    # Factory functions
    "start_preview_server",
    "preview_video",
]
