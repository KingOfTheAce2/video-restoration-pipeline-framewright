"""Archive.org (Internet Archive) upload integration for FrameWright.

Provides upload functionality to the Internet Archive using their S3-like API.
API documentation: https://archive.org/developers/ias3.html
"""

import hashlib
import http.client
import logging
import os
import re
import ssl
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class ArchiveMediaType(Enum):
    """Internet Archive media types."""
    MOVIES = "movies"
    AUDIO = "audio"
    TEXTS = "texts"
    IMAGE = "image"


@dataclass
class ArchiveConfig:
    """Configuration for Archive.org uploads."""
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    default_collection: str = "opensource_movies"
    default_mediatype: ArchiveMediaType = ArchiveMediaType.MOVIES

    def __post_init__(self) -> None:
        """Load credentials from environment if not provided."""
        if self.access_key is None:
            self.access_key = os.environ.get("IA_ACCESS_KEY")
        if self.secret_key is None:
            self.secret_key = os.environ.get("IA_SECRET_KEY")


@dataclass
class ArchiveMetadata:
    """Metadata for an Internet Archive item."""
    identifier: str  # Unique item ID (must be URL-safe)
    title: str
    description: str = ""
    creator: str = ""
    date: str = ""  # YYYY-MM-DD format
    subject: List[str] = field(default_factory=list)  # Tags
    collection: str = "opensource_movies"
    mediatype: ArchiveMediaType = ArchiveMediaType.MOVIES
    licenseurl: str = ""  # Creative Commons URL
    notes: str = ""

    def validate(self) -> List[str]:
        """Validate metadata and return list of errors."""
        errors = []

        if not self.identifier:
            errors.append("Identifier is required")
        elif not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', self.identifier):
            errors.append(
                "Identifier must start with alphanumeric and contain only "
                "alphanumeric, underscore, hyphen, or period characters"
            )

        if not self.title:
            errors.append("Title is required")

        if self.date and not re.match(r'^\d{4}(-\d{2}(-\d{2})?)?$', self.date):
            errors.append("Date must be in YYYY, YYYY-MM, or YYYY-MM-DD format")

        return errors


@dataclass
class UploadResult:
    """Result of an upload operation."""
    success: bool
    item_url: Optional[str] = None
    identifier: Optional[str] = None
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


class ArchiveUploader:
    """Upload files to the Internet Archive."""

    ARCHIVE_HOST = "s3.us.archive.org"
    ARCHIVE_URL = "https://archive.org"

    def __init__(self, config: Optional[ArchiveConfig] = None):
        """Initialize the uploader with configuration."""
        self.config = config or ArchiveConfig()
        self._authenticated = False

    def authenticate(self, access_key: str, secret_key: str) -> bool:
        """Set authentication credentials.

        Args:
            access_key: Internet Archive S3-like access key
            secret_key: Internet Archive S3-like secret key

        Returns:
            True if credentials are set (does not verify with server)
        """
        self.config.access_key = access_key
        self.config.secret_key = secret_key
        self._authenticated = True
        logger.info("Archive.org credentials configured")
        return True

    def is_authenticated(self) -> bool:
        """Check if authentication credentials are configured."""
        return bool(self.config.access_key and self.config.secret_key)

    def check_identifier_available(self, identifier: str) -> bool:
        """Check if an identifier is available on Archive.org.

        Args:
            identifier: The identifier to check

        Returns:
            True if the identifier is available, False if taken or on error
        """
        try:
            context = ssl.create_default_context()
            conn = http.client.HTTPSConnection("archive.org", context=context)

            path = f"/metadata/{urllib.parse.quote(identifier)}"
            conn.request("GET", path)

            response = conn.getresponse()
            data = response.read().decode("utf-8")
            conn.close()

            # If we get a 404 or empty result, identifier is available
            if response.status == 404:
                return True

            # Check if the response indicates no item exists
            if '"error"' in data or data.strip() == "{}":
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking identifier availability: {e}")
            return False

    def upload(
        self,
        video_path: Path,
        metadata: ArchiveMetadata,
    ) -> UploadResult:
        """Upload a video file to the Internet Archive.

        Args:
            video_path: Path to the video file
            metadata: Metadata for the item

        Returns:
            UploadResult with success status and item URL or error message
        """
        # Validate prerequisites
        if not self.is_authenticated():
            return UploadResult(
                success=False,
                error_message="Not authenticated. Call authenticate() first."
            )

        if not video_path.exists():
            return UploadResult(
                success=False,
                error_message=f"Video file not found: {video_path}"
            )

        # Validate metadata
        errors = metadata.validate()
        if errors:
            return UploadResult(
                success=False,
                error_message=f"Metadata validation failed: {'; '.join(errors)}"
            )

        try:
            # Upload the file
            if not self._upload_file(video_path, metadata):
                return UploadResult(
                    success=False,
                    identifier=metadata.identifier,
                    error_message="Upload failed - see logs for details"
                )

            item_url = f"{self.ARCHIVE_URL}/details/{metadata.identifier}"
            logger.info(f"Successfully uploaded to {item_url}")

            return UploadResult(
                success=True,
                item_url=item_url,
                identifier=metadata.identifier
            )

        except Exception as e:
            logger.exception(f"Upload failed: {e}")
            return UploadResult(
                success=False,
                identifier=metadata.identifier,
                error_message=str(e)
            )

    def get_upload_status(self, identifier: str) -> Dict[str, Any]:
        """Get the status of an uploaded item.

        Args:
            identifier: The item identifier

        Returns:
            Dictionary with item status information
        """
        try:
            context = ssl.create_default_context()
            conn = http.client.HTTPSConnection("archive.org", context=context)

            path = f"/metadata/{urllib.parse.quote(identifier)}"
            conn.request("GET", path)

            response = conn.getresponse()
            data = response.read().decode("utf-8")
            conn.close()

            if response.status == 200:
                import json
                return json.loads(data)
            else:
                return {
                    "error": True,
                    "status_code": response.status,
                    "message": f"HTTP {response.status}"
                }

        except Exception as e:
            logger.error(f"Error getting upload status: {e}")
            return {
                "error": True,
                "message": str(e)
            }

    def _create_metadata_xml(self, metadata: ArchiveMetadata) -> str:
        """Create metadata XML for the Internet Archive.

        Args:
            metadata: The metadata to convert to XML

        Returns:
            XML string with metadata
        """
        root = ET.Element("metadata")

        # Add standard fields
        ET.SubElement(root, "identifier").text = metadata.identifier
        ET.SubElement(root, "title").text = metadata.title
        ET.SubElement(root, "mediatype").text = metadata.mediatype.value
        ET.SubElement(root, "collection").text = metadata.collection

        # Add optional fields
        if metadata.description:
            ET.SubElement(root, "description").text = metadata.description

        if metadata.creator:
            ET.SubElement(root, "creator").text = metadata.creator

        if metadata.date:
            ET.SubElement(root, "date").text = metadata.date

        for subject in metadata.subject:
            ET.SubElement(root, "subject").text = subject

        if metadata.licenseurl:
            ET.SubElement(root, "licenseurl").text = metadata.licenseurl

        if metadata.notes:
            ET.SubElement(root, "notes").text = metadata.notes

        return ET.tostring(root, encoding="unicode")

    def _upload_file(
        self,
        video_path: Path,
        metadata: ArchiveMetadata,
    ) -> bool:
        """Upload a file to the Internet Archive using S3-like API.

        Args:
            video_path: Path to the file to upload
            metadata: Metadata for the item

        Returns:
            True if upload succeeded, False otherwise
        """
        identifier = metadata.identifier
        filename = video_path.name

        # Calculate file size and MD5
        file_size = video_path.stat().st_size
        md5_hash = self._calculate_md5(video_path)

        # Build headers
        headers = {
            "Authorization": f"LOW {self.config.access_key}:{self.config.secret_key}",
            "Content-Type": self._get_content_type(video_path),
            "Content-Length": str(file_size),
            "Content-MD5": md5_hash,
            "x-amz-auto-make-bucket": "1",
            "x-archive-queue-derive": "1",
            "x-archive-size-hint": str(file_size),
        }

        # Add metadata headers
        headers["x-archive-meta-title"] = metadata.title
        headers["x-archive-meta-mediatype"] = metadata.mediatype.value
        headers["x-archive-meta-collection"] = metadata.collection

        if metadata.description:
            headers["x-archive-meta-description"] = metadata.description

        if metadata.creator:
            headers["x-archive-meta-creator"] = metadata.creator

        if metadata.date:
            headers["x-archive-meta-date"] = metadata.date

        for i, subject in enumerate(metadata.subject):
            headers[f"x-archive-meta{i:02d}-subject"] = subject

        if metadata.licenseurl:
            headers["x-archive-meta-licenseurl"] = metadata.licenseurl

        if metadata.notes:
            headers["x-archive-meta-notes"] = metadata.notes

        # Encode header values (ASCII only for HTTP headers)
        encoded_headers = {}
        for key, value in headers.items():
            if key.startswith("x-archive-meta"):
                # URI-encode metadata values for non-ASCII content
                encoded_headers[key] = urllib.parse.quote(str(value), safe="")
            else:
                encoded_headers[key] = str(value)

        try:
            context = ssl.create_default_context()
            conn = http.client.HTTPSConnection(self.ARCHIVE_HOST, context=context)

            path = f"/{urllib.parse.quote(identifier)}/{urllib.parse.quote(filename)}"

            logger.info(f"Uploading {filename} ({file_size} bytes) to {identifier}")

            # Open file and upload
            with open(video_path, "rb") as f:
                conn.request("PUT", path, body=f, headers=encoded_headers)

            response = conn.getresponse()
            response_body = response.read().decode("utf-8")
            conn.close()

            if response.status in (200, 201):
                logger.info(f"Upload successful: HTTP {response.status}")
                return True
            else:
                logger.error(
                    f"Upload failed: HTTP {response.status} - {response_body}"
                )
                return False

        except Exception as e:
            logger.exception(f"Upload error: {e}")
            return False

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Base64-encoded MD5 hash
        """
        import base64

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)

        return base64.b64encode(hash_md5.digest()).decode("ascii")

    def _get_content_type(self, file_path: Path) -> str:
        """Get content type for a file based on extension.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string
        """
        extension = file_path.suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".mkv": "video/x-matroska",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".wmv": "video/x-ms-wmv",
            ".flv": "video/x-flv",
            ".webm": "video/webm",
            ".m4v": "video/x-m4v",
            ".mpg": "video/mpeg",
            ".mpeg": "video/mpeg",
            ".ogv": "video/ogg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".pdf": "application/pdf",
        }
        return content_types.get(extension, "application/octet-stream")


def generate_identifier(title: str) -> str:
    """Generate a clean identifier from a title.

    Args:
        title: The title to convert

    Returns:
        A URL-safe identifier suitable for Archive.org
    """
    # Convert to lowercase and replace spaces with underscores
    identifier = title.lower().strip()

    # Replace spaces and common separators with underscores
    identifier = re.sub(r'[\s\-]+', '_', identifier)

    # Remove any character that isn't alphanumeric, underscore, hyphen, or period
    identifier = re.sub(r'[^a-z0-9_.\-]', '', identifier)

    # Remove consecutive underscores
    identifier = re.sub(r'_+', '_', identifier)

    # Remove leading/trailing underscores or periods
    identifier = identifier.strip('_.')

    # Ensure it starts with alphanumeric
    if identifier and not identifier[0].isalnum():
        identifier = 'item_' + identifier

    # Ensure minimum length and add suffix if needed
    if len(identifier) < 3:
        identifier = f"item_{identifier}" if identifier else "item"

    # Truncate to reasonable length (Archive.org limit is 100)
    if len(identifier) > 80:
        identifier = identifier[:80].rstrip('_.')

    return identifier


def upload_to_archive(
    video_path: Path,
    title: str,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    description: str = "",
    creator: str = "",
    date: str = "",
    subject: Optional[List[str]] = None,
    collection: str = "opensource_movies",
    mediatype: ArchiveMediaType = ArchiveMediaType.MOVIES,
    licenseurl: str = "",
    notes: str = "",
    identifier: Optional[str] = None,
) -> UploadResult:
    """Convenience function to upload a video to the Internet Archive.

    Args:
        video_path: Path to the video file
        title: Title for the item
        access_key: Archive.org S3 access key (or set IA_ACCESS_KEY env var)
        secret_key: Archive.org S3 secret key (or set IA_SECRET_KEY env var)
        description: Description of the item
        creator: Creator/author name
        date: Date in YYYY-MM-DD format
        subject: List of tags/subjects
        collection: Archive.org collection name
        mediatype: Type of media (MOVIES, AUDIO, TEXTS, IMAGE)
        licenseurl: URL to the license (e.g., Creative Commons)
        notes: Additional notes
        identifier: Custom identifier (auto-generated from title if not provided)

    Returns:
        UploadResult with success status and item URL or error message
    """
    # Create configuration
    config = ArchiveConfig(
        access_key=access_key,
        secret_key=secret_key,
        default_collection=collection,
        default_mediatype=mediatype,
    )

    # Generate identifier if not provided
    if identifier is None:
        identifier = generate_identifier(title)

    # Create metadata
    metadata = ArchiveMetadata(
        identifier=identifier,
        title=title,
        description=description,
        creator=creator,
        date=date,
        subject=subject or [],
        collection=collection,
        mediatype=mediatype,
        licenseurl=licenseurl,
        notes=notes,
    )

    # Create uploader and upload
    uploader = ArchiveUploader(config)

    if not uploader.is_authenticated():
        return UploadResult(
            success=False,
            error_message=(
                "No authentication credentials provided. "
                "Set access_key/secret_key or IA_ACCESS_KEY/IA_SECRET_KEY env vars."
            )
        )

    return uploader.upload(Path(video_path), metadata)
