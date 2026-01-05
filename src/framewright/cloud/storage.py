"""Cloud storage providers for video upload/download."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from urllib.parse import urlparse

from .base import CloudStorageProvider, StorageError


class S3Storage(CloudStorageProvider):
    """AWS S3 storage provider.

    Supports both standard S3 and S3-compatible storage (MinIO, DigitalOcean Spaces, etc.).

    Example:
        >>> storage = S3Storage(
        ...     bucket="my-video-bucket",
        ...     credentials={
        ...         "aws_access_key_id": "AKIAXXXXXXXX",
        ...         "aws_secret_access_key": "secret",
        ...         "region_name": "us-east-1",
        ...     }
        ... )
        >>> uri = storage.upload(Path("video.mp4"), "input/video.mp4")
        >>> print(uri)  # s3://my-video-bucket/input/video.mp4
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
    ):
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name.
            credentials: AWS credentials dict with aws_access_key_id,
                        aws_secret_access_key, and optionally region_name.
            endpoint_url: Custom endpoint URL for S3-compatible storage.
            region_name: AWS region name.
        """
        credentials = credentials or {}

        # Load from environment if not provided
        if "aws_access_key_id" not in credentials:
            credentials["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
        if "aws_secret_access_key" not in credentials:
            credentials["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if "region_name" not in credentials:
            credentials["region_name"] = os.environ.get("AWS_DEFAULT_REGION", region_name)

        super().__init__(bucket=bucket, credentials=credentials)
        self._endpoint_url = endpoint_url
        self._client = None

    @property
    def scheme(self) -> str:
        """Get the URI scheme for S3."""
        return "s3"

    def _get_client(self):
        """Get or create boto3 S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config

                config = Config(
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    connect_timeout=30,
                    read_timeout=300,
                )

                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=self._credentials.get("aws_access_key_id"),
                    aws_secret_access_key=self._credentials.get("aws_secret_access_key"),
                    region_name=self._credentials.get("region_name"),
                    endpoint_url=self._endpoint_url,
                    config=config,
                )
            except ImportError:
                raise StorageError(
                    "boto3 library required for S3 storage. "
                    "Install with: pip install framewright[cloud]"
                )
        return self._client

    def upload(
        self,
        local_path: Path,
        remote_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload file to S3.

        Args:
            local_path: Local file path.
            remote_path: Remote path within bucket (key).
            progress_callback: Optional progress callback (0-1).

        Returns:
            Full S3 URI of uploaded file.

        Raises:
            StorageError: If upload fails.
        """
        if not local_path.exists():
            raise StorageError(f"File not found: {local_path}")

        bucket, key = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            file_size = local_path.stat().st_size

            # Create progress callback wrapper
            uploaded = [0]

            def upload_progress(bytes_transferred):
                uploaded[0] += bytes_transferred
                if progress_callback and file_size > 0:
                    progress_callback(uploaded[0] / file_size)

            # Use multipart upload for large files
            client.upload_file(
                str(local_path),
                bucket,
                key,
                Callback=upload_progress if progress_callback else None,
            )

            return self.get_uri(f"{bucket}/{key}")

        except Exception as e:
            raise StorageError(f"S3 upload failed: {e}")

    def download(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download file from S3.

        Args:
            remote_path: Remote path or full S3 URI.
            local_path: Local destination path.
            progress_callback: Optional progress callback (0-1).

        Raises:
            StorageError: If download fails.
        """
        bucket, key = self._resolve_path(remote_path)

        try:
            client = self._get_client()

            # Get file size for progress
            head = client.head_object(Bucket=bucket, Key=key)
            file_size = head.get("ContentLength", 0)

            # Create progress callback wrapper
            downloaded = [0]

            def download_progress(bytes_transferred):
                downloaded[0] += bytes_transferred
                if progress_callback and file_size > 0:
                    progress_callback(downloaded[0] / file_size)

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            client.download_file(
                bucket,
                key,
                str(local_path),
                Callback=download_progress if progress_callback else None,
            )

        except Exception as e:
            raise StorageError(f"S3 download failed: {e}")

    def delete(self, remote_path: str) -> bool:
        """Delete file from S3.

        Args:
            remote_path: Remote path or full S3 URI.

        Returns:
            True if deletion successful.
        """
        bucket, key = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            client.delete_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def exists(self, remote_path: str) -> bool:
        """Check if file exists in S3.

        Args:
            remote_path: Remote path or full S3 URI.

        Returns:
            True if file exists.
        """
        bucket, key = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def list_files(
        self,
        prefix: str = "",
        max_results: int = 1000,
    ) -> List[str]:
        """List files in S3 bucket.

        Args:
            prefix: Path prefix to filter by.
            max_results: Maximum number of results.

        Returns:
            List of file keys.
        """
        bucket = self._bucket

        try:
            client = self._get_client()
            paginator = client.get_paginator("list_objects_v2")

            files = []
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=min(max_results, 1000),
            ):
                for obj in page.get("Contents", []):
                    files.append(obj["Key"])
                    if len(files) >= max_results:
                        return files

            return files

        except Exception:
            return []

    def _resolve_path(self, path: str) -> tuple:
        """Resolve path to bucket and key."""
        if path.startswith("s3://"):
            bucket, key = self.parse_uri(path)
        else:
            bucket = self._bucket
            key = path.lstrip("/")

        if not bucket:
            raise StorageError("No bucket specified")

        return bucket, key

    def generate_presigned_url(
        self,
        remote_path: str,
        expires_in: int = 3600,
        method: str = "get_object",
    ) -> str:
        """Generate a presigned URL for the object.

        Args:
            remote_path: Remote path or full S3 URI.
            expires_in: URL expiry time in seconds.
            method: S3 method (get_object, put_object).

        Returns:
            Presigned URL.
        """
        bucket, key = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            url = client.generate_presigned_url(
                method,
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {e}")


class GCSStorage(CloudStorageProvider):
    """Google Cloud Storage provider.

    Example:
        >>> storage = GCSStorage(
        ...     bucket="my-video-bucket",
        ...     credentials={"service_account_file": "/path/to/key.json"}
        ... )
        >>> uri = storage.upload(Path("video.mp4"), "input/video.mp4")
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
    ):
        """Initialize GCS storage.

        Args:
            bucket: GCS bucket name.
            credentials: Dict with either:
                        - service_account_file: Path to service account JSON
                        - service_account_info: Dict of service account credentials
            project: Google Cloud project ID.
        """
        super().__init__(bucket=bucket, credentials=credentials or {})
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._client = None

    @property
    def scheme(self) -> str:
        """Get the URI scheme for GCS."""
        return "gs"

    def _get_client(self):
        """Get or create GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                from google.oauth2 import service_account

                if "service_account_file" in self._credentials:
                    credentials = service_account.Credentials.from_service_account_file(
                        self._credentials["service_account_file"]
                    )
                    self._client = storage.Client(
                        project=self._project, credentials=credentials
                    )
                elif "service_account_info" in self._credentials:
                    credentials = service_account.Credentials.from_service_account_info(
                        self._credentials["service_account_info"]
                    )
                    self._client = storage.Client(
                        project=self._project, credentials=credentials
                    )
                else:
                    # Use default credentials
                    self._client = storage.Client(project=self._project)

            except ImportError:
                raise StorageError(
                    "google-cloud-storage library required for GCS. "
                    "Install with: pip install framewright[cloud]"
                )
        return self._client

    def upload(
        self,
        local_path: Path,
        remote_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload file to GCS."""
        if not local_path.exists():
            raise StorageError(f"File not found: {local_path}")

        bucket_name, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            file_size = local_path.stat().st_size

            # For large files, use resumable upload with progress
            if file_size > 5 * 1024 * 1024 and progress_callback:  # > 5MB
                from google.cloud.storage import transfer_manager

                # Use transfer manager for progress tracking
                blob.upload_from_filename(str(local_path))
                progress_callback(1.0)
            else:
                blob.upload_from_filename(str(local_path))
                if progress_callback:
                    progress_callback(1.0)

            return self.get_uri(f"{bucket_name}/{blob_name}")

        except Exception as e:
            raise StorageError(f"GCS upload failed: {e}")

    def download(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download file from GCS."""
        bucket_name, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob.download_to_filename(str(local_path))

            if progress_callback:
                progress_callback(1.0)

        except Exception as e:
            raise StorageError(f"GCS download failed: {e}")

    def delete(self, remote_path: str) -> bool:
        """Delete file from GCS."""
        bucket_name, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception:
            return False

    def exists(self, remote_path: str) -> bool:
        """Check if file exists in GCS."""
        bucket_name, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
        except Exception:
            return False

    def list_files(
        self,
        prefix: str = "",
        max_results: int = 1000,
    ) -> List[str]:
        """List files in GCS bucket."""
        try:
            client = self._get_client()
            bucket = client.bucket(self._bucket)
            blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
            return [blob.name for blob in blobs]
        except Exception:
            return []

    def _resolve_path(self, path: str) -> tuple:
        """Resolve path to bucket and blob name."""
        if path.startswith("gs://"):
            bucket, blob = self.parse_uri(path)
        else:
            bucket = self._bucket
            blob = path.lstrip("/")

        if not bucket:
            raise StorageError("No bucket specified")

        return bucket, blob


class AzureStorage(CloudStorageProvider):
    """Azure Blob Storage provider.

    Example:
        >>> storage = AzureStorage(
        ...     bucket="my-container",  # Container name
        ...     credentials={
        ...         "connection_string": "DefaultEndpointsProtocol=https;..."
        ...     }
        ... )
        >>> uri = storage.upload(Path("video.mp4"), "input/video.mp4")
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        account_name: Optional[str] = None,
    ):
        """Initialize Azure Blob storage.

        Args:
            bucket: Azure container name.
            credentials: Dict with one of:
                        - connection_string: Azure storage connection string
                        - account_key: Storage account key (requires account_name)
                        - sas_token: Shared Access Signature token
            account_name: Storage account name.
        """
        credentials = credentials or {}

        # Load from environment if not provided
        if "connection_string" not in credentials:
            conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if conn_str:
                credentials["connection_string"] = conn_str

        super().__init__(bucket=bucket, credentials=credentials)
        self._account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT")
        self._client = None

    @property
    def scheme(self) -> str:
        """Get the URI scheme for Azure Blob."""
        return "azure"

    def _get_client(self):
        """Get or create Azure Blob client."""
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient

                if "connection_string" in self._credentials:
                    self._client = BlobServiceClient.from_connection_string(
                        self._credentials["connection_string"]
                    )
                elif "account_key" in self._credentials and self._account_name:
                    account_url = f"https://{self._account_name}.blob.core.windows.net"
                    self._client = BlobServiceClient(
                        account_url=account_url,
                        credential=self._credentials["account_key"],
                    )
                elif "sas_token" in self._credentials and self._account_name:
                    account_url = f"https://{self._account_name}.blob.core.windows.net"
                    self._client = BlobServiceClient(
                        account_url=f"{account_url}?{self._credentials['sas_token']}"
                    )
                else:
                    raise StorageError(
                        "Azure credentials required. Provide connection_string, "
                        "account_key, or sas_token."
                    )

            except ImportError:
                raise StorageError(
                    "azure-storage-blob library required for Azure Blob. "
                    "Install with: pip install framewright[cloud]"
                )
        return self._client

    def upload(
        self,
        local_path: Path,
        remote_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload file to Azure Blob."""
        if not local_path.exists():
            raise StorageError(f"File not found: {local_path}")

        container, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            container_client = client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)

            file_size = local_path.stat().st_size
            uploaded = [0]

            def progress_hook(response):
                if progress_callback and file_size > 0:
                    current = response.context.get("upload_stream_current", 0)
                    uploaded[0] = current
                    progress_callback(current / file_size)

            with open(local_path, "rb") as f:
                blob_client.upload_blob(
                    f,
                    overwrite=True,
                    raw_response_hook=progress_hook if progress_callback else None,
                )

            if progress_callback:
                progress_callback(1.0)

            return self.get_uri(f"{container}/{blob_name}")

        except Exception as e:
            raise StorageError(f"Azure upload failed: {e}")

    def download(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download file from Azure Blob."""
        container, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            container_client = client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)

            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                download_stream.readinto(f)

            if progress_callback:
                progress_callback(1.0)

        except Exception as e:
            raise StorageError(f"Azure download failed: {e}")

    def delete(self, remote_path: str) -> bool:
        """Delete file from Azure Blob."""
        container, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            container_client = client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            return True
        except Exception:
            return False

    def exists(self, remote_path: str) -> bool:
        """Check if file exists in Azure Blob."""
        container, blob_name = self._resolve_path(remote_path)

        try:
            client = self._get_client()
            container_client = client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            return blob_client.exists()
        except Exception:
            return False

    def list_files(
        self,
        prefix: str = "",
        max_results: int = 1000,
    ) -> List[str]:
        """List files in Azure container."""
        try:
            client = self._get_client()
            container_client = client.get_container_client(self._bucket)
            blobs = container_client.list_blobs(name_starts_with=prefix)

            files = []
            for blob in blobs:
                files.append(blob.name)
                if len(files) >= max_results:
                    break

            return files
        except Exception:
            return []

    def _resolve_path(self, path: str) -> tuple:
        """Resolve path to container and blob name."""
        if path.startswith("azure://"):
            container, blob = self.parse_uri(path)
        else:
            container = self._bucket
            blob = path.lstrip("/")

        if not container:
            raise StorageError("No container specified")

        return container, blob


# Import Google Drive storage
try:
    from .gdrive import GoogleDriveStorage
    _gdrive_available = True
except ImportError:
    _gdrive_available = False

# Storage provider registry
STORAGE_PROVIDERS: Dict[str, Type[CloudStorageProvider]] = {
    "s3": S3Storage,
    "gs": GCSStorage,
    "gcs": GCSStorage,
    "azure": AzureStorage,
}

# Add Google Drive if available
if _gdrive_available:
    STORAGE_PROVIDERS["gdrive"] = GoogleDriveStorage
    STORAGE_PROVIDERS["googledrive"] = GoogleDriveStorage


def get_storage_provider(
    uri_or_type: str,
    bucket: Optional[str] = None,
    credentials: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> CloudStorageProvider:
    """Get appropriate storage provider based on URI or type.

    Args:
        uri_or_type: Storage URI (s3://..., gs://...) or type name (s3, gcs, azure).
        bucket: Default bucket name.
        credentials: Provider-specific credentials.
        **kwargs: Additional provider arguments.

    Returns:
        Configured storage provider instance.

    Raises:
        StorageError: If provider type is not recognized.

    Example:
        >>> storage = get_storage_provider("s3://my-bucket/path")
        >>> storage = get_storage_provider("gcs", bucket="my-bucket")
    """
    # Extract scheme from URI
    if "://" in uri_or_type:
        parsed = urlparse(uri_or_type)
        scheme = parsed.scheme.lower()
        if not bucket:
            bucket = parsed.netloc
    else:
        scheme = uri_or_type.lower()

    provider_class = STORAGE_PROVIDERS.get(scheme)

    if not provider_class:
        available = ", ".join(STORAGE_PROVIDERS.keys())
        raise StorageError(
            f"Unknown storage provider: {scheme}. Available: {available}"
        )

    return provider_class(bucket=bucket, credentials=credentials, **kwargs)
