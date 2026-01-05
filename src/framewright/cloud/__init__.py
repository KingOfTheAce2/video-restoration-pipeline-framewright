"""Cloud processing backend for FrameWright video restoration.

This module provides cloud GPU computing integration for video restoration,
supporting providers like RunPod and Vast.ai, as well as cloud storage
backends including AWS S3, Google Cloud Storage, Azure Blob Storage,
and Google Drive (via rclone).
"""

from .base import (
    CloudProvider,
    CloudStorageProvider,
    JobStatus,
    JobState,
    ProcessingConfig,
    CloudError,
    AuthenticationError,
    JobSubmissionError,
    JobExecutionError,
    StorageError,
)
from .runpod import RunPodProvider
from .vastai import VastAIProvider
from .storage import (
    S3Storage,
    GCSStorage,
    AzureStorage,
    get_storage_provider,
)

# Google Drive support (requires rclone)
try:
    from .gdrive import (
        GoogleDriveStorage,
        setup_gdrive_remote,
        check_gdrive_configured,
    )
    _GDRIVE_AVAILABLE = True
except ImportError:
    GoogleDriveStorage = None
    setup_gdrive_remote = None
    check_gdrive_configured = None
    _GDRIVE_AVAILABLE = False

__all__ = [
    # Base classes
    "CloudProvider",
    "CloudStorageProvider",
    "JobStatus",
    "JobState",
    "ProcessingConfig",
    # Errors
    "CloudError",
    "AuthenticationError",
    "JobSubmissionError",
    "JobExecutionError",
    "StorageError",
    # Providers
    "RunPodProvider",
    "VastAIProvider",
    # Storage
    "S3Storage",
    "GCSStorage",
    "AzureStorage",
    "get_storage_provider",
    # Google Drive
    "GoogleDriveStorage",
    "setup_gdrive_remote",
    "check_gdrive_configured",
]
