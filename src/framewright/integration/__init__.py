"""Professional integration module for FrameWright.

Provides EDL import/export, LUT support, webhook notifications,
email/SMS notification systems, media library integration,
Archive.org upload, and YouTube upload integration.

Heavy dependencies (notifications, media_libraries, youtube_upload, archive_org)
are lazily loaded to improve import performance.
"""

import importlib
from typing import TYPE_CHECKING

# Lightweight imports - loaded immediately
from .edl import EDLParser, EDLWriter, EDLEvent, EDL
from .lut import LUTManager, LUT, LUTFormat
from .webhooks import WebhookManager, WebhookEvent, WebhookConfig

# Type checking imports for IDE support (not loaded at runtime)
if TYPE_CHECKING:
    from .notifications import (
        NotificationType,
        EmailConfig,
        SMSConfig,
        NotificationConfig,
        EmailSender,
        SMSSender,
        NotificationManager,
        create_email_notifier,
        create_gmail_notifier,
        create_sms_notifier,
        create_notification_manager,
    )
    from .media_libraries import (
        MediaServer,
        MediaServerConfig,
        PlexConnector,
        JellyfinConnector,
        EmbyConnector,
        MediaLibraryManager,
        setup_plex,
        setup_jellyfin,
        setup_emby,
    )
    from .archive_org import (
        ArchiveMediaType,
        ArchiveConfig,
        ArchiveMetadata,
        ArchiveUploader,
        UploadResult,
        upload_to_archive,
        generate_identifier,
    )
    from .youtube_upload import (
        YouTubePrivacy,
        YouTubeConfig,
        VideoMetadata as YouTubeVideoMetadata,
        UploadResult as YouTubeUploadResult,
        YouTubeUploader,
        upload_to_youtube,
    )

# Mapping of lazy-loaded names to their source modules
_lazy_imports = {
    # Notifications (requires smtplib setup)
    "NotificationType": "notifications",
    "EmailConfig": "notifications",
    "SMSConfig": "notifications",
    "NotificationConfig": "notifications",
    "EmailSender": "notifications",
    "SMSSender": "notifications",
    "NotificationManager": "notifications",
    "create_email_notifier": "notifications",
    "create_gmail_notifier": "notifications",
    "create_sms_notifier": "notifications",
    "create_notification_manager": "notifications",
    # Media Libraries (Plex/Jellyfin/Emby API)
    "MediaServer": "media_libraries",
    "MediaServerConfig": "media_libraries",
    "PlexConnector": "media_libraries",
    "JellyfinConnector": "media_libraries",
    "EmbyConnector": "media_libraries",
    "MediaLibraryManager": "media_libraries",
    "setup_plex": "media_libraries",
    "setup_jellyfin": "media_libraries",
    "setup_emby": "media_libraries",
    # Archive.org (S3 API)
    "ArchiveMediaType": "archive_org",
    "ArchiveConfig": "archive_org",
    "ArchiveMetadata": "archive_org",
    "ArchiveUploader": "archive_org",
    "UploadResult": "archive_org",
    "upload_to_archive": "archive_org",
    "generate_identifier": "archive_org",
    # YouTube (requires google-api-python-client)
    "YouTubePrivacy": "youtube_upload",
    "YouTubeConfig": "youtube_upload",
    "YouTubeVideoMetadata": "youtube_upload",
    "YouTubeUploadResult": "youtube_upload",
    "YouTubeUploader": "youtube_upload",
    "upload_to_youtube": "youtube_upload",
}

# Special renames for YouTube module
_lazy_renames = {
    "YouTubeVideoMetadata": "VideoMetadata",
    "YouTubeUploadResult": "UploadResult",
}

# Cache for loaded modules to avoid repeated imports
_loaded_modules = {}


def __getattr__(name: str):
    """Lazy load heavy integration modules on first access."""
    if name in _lazy_imports:
        module_name = _lazy_imports[name]

        # Load module if not already cached
        if module_name not in _loaded_modules:
            _loaded_modules[module_name] = importlib.import_module(
                f".{module_name}", __package__
            )

        module = _loaded_modules[module_name]

        # Handle renamed attributes (e.g., YouTubeVideoMetadata -> VideoMetadata)
        attr_name = _lazy_renames.get(name, name)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy-loaded names in dir() for discoverability."""
    return list(__all__)


__all__ = [
    # EDL (lightweight - loaded immediately)
    "EDLParser",
    "EDLWriter",
    "EDLEvent",
    "EDL",
    # LUT (lightweight - loaded immediately)
    "LUTManager",
    "LUT",
    "LUTFormat",
    # Webhooks (lightweight - loaded immediately)
    "WebhookManager",
    "WebhookEvent",
    "WebhookConfig",
    # Notifications (lazy-loaded)
    "NotificationType",
    "EmailConfig",
    "SMSConfig",
    "NotificationConfig",
    "EmailSender",
    "SMSSender",
    "NotificationManager",
    "create_email_notifier",
    "create_gmail_notifier",
    "create_sms_notifier",
    "create_notification_manager",
    # Media Libraries (lazy-loaded)
    "MediaServer",
    "MediaServerConfig",
    "PlexConnector",
    "JellyfinConnector",
    "EmbyConnector",
    "MediaLibraryManager",
    "setup_plex",
    "setup_jellyfin",
    "setup_emby",
    # Archive.org (lazy-loaded)
    "ArchiveMediaType",
    "ArchiveConfig",
    "ArchiveMetadata",
    "ArchiveUploader",
    "UploadResult",
    "upload_to_archive",
    "generate_identifier",
    # YouTube (lazy-loaded)
    "YouTubePrivacy",
    "YouTubeConfig",
    "YouTubeVideoMetadata",
    "YouTubeUploadResult",
    "YouTubeUploader",
    "upload_to_youtube",
]
