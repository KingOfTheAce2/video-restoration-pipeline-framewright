"""Cloud processing backend for FrameWright video restoration.

This module provides cloud GPU computing integration for video restoration,
supporting providers like RunPod and Vast.ai, as well as cloud storage
backends including AWS S3, Google Cloud Storage, and Azure Blob Storage.
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
]
