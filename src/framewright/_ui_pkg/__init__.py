"""FrameWright UI Module - Apple-quality terminal user experience.

This module provides:
- Rich-based beautiful terminal output
- Interactive wizard for guided setup
- Smart auto-detection and recommendations
- Progress displays with ETA and statistics
"""

from .terminal import (
    Console,
    Theme,
    create_console,
    print_banner,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_step,
    create_panel,
    create_table,
)

from .progress import (
    ProgressDisplay,
    StageProgress,
    create_progress_display,
    create_spinner,
)

from .wizard import (
    InteractiveWizard,
    WizardResult,
    run_wizard,
)

from .auto_detect import (
    SmartAnalyzer,
    AnalysisResult,
    ContentProfile,
    DegradationProfile,
    analyze_video_smart,
)

from .recommendations import (
    PresetRecommender,
    Recommendation,
    RecommendationReason,
    get_recommendations,
)

# Preview classes (some may not be available)
try:
    from .preview import (
        RealTimePreview,
        PreviewConfig,
        PreviewMode,
        PreviewBackend,
        PreviewFrame,
        PreviewIntegration,
        create_preview,
    )
except ImportError:
    # Use server-based preview as fallback
    from .preview.server import PreviewConfig, PreviewServer
    RealTimePreview = PreviewServer
    PreviewMode = None
    PreviewBackend = None
    PreviewFrame = None
    PreviewIntegration = None
    create_preview = None

# Preview Server (web-based real-time preview)
from .preview.server import (
    PreviewConfig as PreviewServerConfig,
    PreviewQuality,
    PreviewServer,
    VideoInfo,
    RenderTask,
    ServerStatus,
    SegmentCache,
    RenderQueue,
    start_preview_server,
    preview_video,
)

# Web Dashboard (standard library only)
from .dashboard import (
    DashboardConfig,
    DashboardServer,
    DashboardHandler,
    WebSocketConnection,
    WebSocketManager,
    start_dashboard,
    render_dashboard_page,
    render_error_page,
    render_login_page,
)

# REST API (standard library only)
from .api import (
    APIConfig,
    APIServer,
    APIHandler,
    RateLimiter,
    OPENAPI_SPEC,
    start_api_server,
    FrameWrightClient,
    JobStatus,
    Model as APIModel,
    Preset as APIPreset,
    HardwareInfo as APIHardwareInfo,
    AnalysisResult as APIAnalysisResult,
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    create_client,
    quick_restore,
)

__all__ = [
    # Terminal
    "Console",
    "Theme",
    "create_console",
    "print_banner",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_step",
    "create_panel",
    "create_table",
    # Progress
    "ProgressDisplay",
    "StageProgress",
    "create_progress_display",
    "create_spinner",
    # Wizard
    "InteractiveWizard",
    "WizardResult",
    "run_wizard",
    # Auto-detect
    "SmartAnalyzer",
    "AnalysisResult",
    "ContentProfile",
    "DegradationProfile",
    "analyze_video_smart",
    # Recommendations
    "PresetRecommender",
    "Recommendation",
    "RecommendationReason",
    "get_recommendations",
    # Preview
    "RealTimePreview",
    "PreviewConfig",
    "PreviewMode",
    "PreviewBackend",
    "PreviewFrame",
    "PreviewIntegration",
    "create_preview",
    # Preview Server (web-based)
    "PreviewServerConfig",
    "PreviewQuality",
    "PreviewServer",
    "VideoInfo",
    "RenderTask",
    "ServerStatus",
    "SegmentCache",
    "RenderQueue",
    "start_preview_server",
    "preview_video",
    # Web Dashboard
    "DashboardConfig",
    "DashboardServer",
    "DashboardHandler",
    "WebSocketConnection",
    "WebSocketManager",
    "start_dashboard",
    "render_dashboard_page",
    "render_error_page",
    "render_login_page",
    # REST API
    "APIConfig",
    "APIServer",
    "APIHandler",
    "RateLimiter",
    "OPENAPI_SPEC",
    "start_api_server",
    "FrameWrightClient",
    "JobStatus",
    "APIModel",
    "APIPreset",
    "APIHardwareInfo",
    "APIAnalysisResult",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "create_client",
    "quick_restore",
]
