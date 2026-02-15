"""YouTube upload integration for FrameWright.

Enables direct upload of restored videos to YouTube with metadata,
thumbnails, and playlist management.

Requires optional dependencies:
    pip install google-api-python-client google-auth-oauthlib
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Track if YouTube API dependencies are available
_YOUTUBE_API_AVAILABLE = False
_MISSING_DEPS_MESSAGE = (
    "YouTube upload requires google-api-python-client and google-auth-oauthlib. "
    "Install with: pip install google-api-python-client google-auth-oauthlib"
)

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    _YOUTUBE_API_AVAILABLE = True
except ImportError:
    # Dependencies not installed - will raise helpful error when used
    Credentials = None
    InstalledAppFlow = None
    build = None
    MediaFileUpload = None
    HttpError = Exception


# YouTube API scopes required for upload
YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_FULL_SCOPE = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


class YouTubePrivacy(Enum):
    """YouTube video privacy settings."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


@dataclass
class YouTubeConfig:
    """Configuration for YouTube upload integration.

    Attributes:
        client_secrets_path: Path to OAuth client secrets JSON file from Google Cloud Console.
            If None, must be set before authentication.
        credentials_path: Path where authenticated credentials will be stored.
        default_privacy: Default privacy setting for uploaded videos.
        default_category: Default YouTube category ID (22 = People & Blogs).
        notify_subscribers: Whether to notify subscribers on public uploads.
    """
    client_secrets_path: Optional[Path] = None
    credentials_path: Path = field(
        default_factory=lambda: Path.home() / ".framewright" / "youtube_credentials.json"
    )
    default_privacy: YouTubePrivacy = YouTubePrivacy.PRIVATE
    default_category: str = "22"  # People & Blogs
    notify_subscribers: bool = False


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video upload.

    Attributes:
        title: Video title (required, max 100 characters).
        description: Video description (max 5000 characters).
        tags: List of video tags for discovery.
        category: YouTube category ID.
        privacy: Video privacy setting.
        playlist_id: Optional playlist to add the video to after upload.
        thumbnail_path: Optional custom thumbnail image path.
    """
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "22"
    privacy: YouTubePrivacy = YouTubePrivacy.PRIVATE
    playlist_id: Optional[str] = None
    thumbnail_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate metadata constraints."""
        if not self.title:
            raise ValueError("Video title is required")
        if len(self.title) > 100:
            raise ValueError(f"Title exceeds 100 characters: {len(self.title)}")
        if len(self.description) > 5000:
            raise ValueError(f"Description exceeds 5000 characters: {len(self.description)}")


@dataclass
class UploadResult:
    """Result of a YouTube upload operation.

    Attributes:
        success: Whether the upload completed successfully.
        video_id: YouTube video ID if successful.
        video_url: Full YouTube URL to the video if successful.
        error_message: Error description if upload failed.
    """
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate video URL from ID if available."""
        if self.video_id and not self.video_url:
            self.video_url = f"https://www.youtube.com/watch?v={self.video_id}"


class YouTubeUploader:
    """Handles authenticated uploads to YouTube.

    Provides OAuth authentication flow, resumable video uploads,
    thumbnail management, and playlist integration.

    Example:
        config = YouTubeConfig(
            client_secrets_path=Path("client_secrets.json")
        )
        uploader = YouTubeUploader(config)

        if uploader.authenticate():
            metadata = VideoMetadata(
                title="My Restored Video",
                description="Restored with FrameWright",
                privacy=YouTubePrivacy.UNLISTED
            )
            result = uploader.upload(Path("output.mp4"), metadata)
            if result.success:
                print(f"Uploaded: {result.video_url}")
    """

    # Retry configuration for resumable upload
    MAX_RETRIES = 10
    RETRY_EXCEPTIONS = (
        "HttpError",
        "IOError",
        "ConnectionResetError",
        "BrokenPipeError",
    )

    def __init__(self, config: YouTubeConfig) -> None:
        """Initialize the uploader with configuration.

        Args:
            config: YouTube configuration settings.

        Raises:
            ImportError: If required dependencies are not installed.
        """
        if not _YOUTUBE_API_AVAILABLE:
            raise ImportError(_MISSING_DEPS_MESSAGE)

        self.config = config
        self._credentials: Optional[Credentials] = None
        self._service: Optional[Any] = None

        # Ensure credentials directory exists
        self.config.credentials_path.parent.mkdir(parents=True, exist_ok=True)

    def authenticate(self) -> bool:
        """Perform OAuth authentication flow.

        If valid credentials exist, they will be loaded. Otherwise,
        launches browser-based OAuth flow to obtain new credentials.

        Returns:
            True if authentication successful, False otherwise.
        """
        try:
            # Try to load existing credentials
            if self.config.credentials_path.exists():
                logger.info("Loading existing YouTube credentials")
                self._credentials = Credentials.from_authorized_user_file(
                    str(self.config.credentials_path),
                    YOUTUBE_FULL_SCOPE
                )

                # Check if credentials are still valid
                if self._credentials and self._credentials.valid:
                    logger.info("YouTube credentials are valid")
                    return True

                # Try to refresh expired credentials
                if self._credentials and self._credentials.expired and self._credentials.refresh_token:
                    try:
                        from google.auth.transport.requests import Request
                        self._credentials.refresh(Request())
                        self._save_credentials()
                        logger.info("YouTube credentials refreshed")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to refresh credentials: {e}")
                        self._credentials = None

            # Need new authentication
            if not self.config.client_secrets_path:
                logger.error("client_secrets_path not configured")
                return False

            if not self.config.client_secrets_path.exists():
                logger.error(f"Client secrets file not found: {self.config.client_secrets_path}")
                return False

            logger.info("Starting OAuth authentication flow")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.config.client_secrets_path),
                YOUTUBE_FULL_SCOPE
            )

            # Run local server for OAuth callback
            self._credentials = flow.run_local_server(
                port=8080,
                prompt="consent",
                access_type="offline"
            )

            self._save_credentials()
            logger.info("YouTube authentication successful")
            return True

        except Exception as e:
            logger.error(f"YouTube authentication failed: {e}")
            return False

    def _save_credentials(self) -> None:
        """Save credentials to disk for future use."""
        if self._credentials:
            with open(self.config.credentials_path, "w") as f:
                f.write(self._credentials.to_json())
            logger.debug(f"Credentials saved to {self.config.credentials_path}")

    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid credentials.

        Returns:
            True if authenticated and credentials are valid.
        """
        return self._credentials is not None and self._credentials.valid

    def _get_authenticated_service(self) -> Any:
        """Get or create authenticated YouTube API service.

        Returns:
            YouTube API service object.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self.is_authenticated():
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        if self._service is None:
            self._service = build("youtube", "v3", credentials=self._credentials)

        return self._service

    def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> UploadResult:
        """Upload a video to YouTube.

        Uses resumable upload for reliability with large files.

        Args:
            video_path: Path to the video file.
            metadata: Video metadata (title, description, etc.).
            progress_callback: Optional callback for upload progress (0.0-1.0).

        Returns:
            UploadResult with success status and video details.
        """
        if not video_path.exists():
            return UploadResult(
                success=False,
                error_message=f"Video file not found: {video_path}"
            )

        try:
            video_id = self._resumable_upload(video_path, metadata, progress_callback)

            # Set custom thumbnail if provided
            if metadata.thumbnail_path:
                if not self.set_thumbnail(video_id, metadata.thumbnail_path):
                    logger.warning(f"Failed to set thumbnail for video {video_id}")

            # Add to playlist if specified
            if metadata.playlist_id:
                if not self.add_to_playlist(video_id, metadata.playlist_id):
                    logger.warning(f"Failed to add video {video_id} to playlist {metadata.playlist_id}")

            return UploadResult(success=True, video_id=video_id)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Upload failed: {error_msg}")
            return UploadResult(success=False, error_message=error_msg)

    def _resumable_upload(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Perform resumable upload with retry logic.

        Args:
            video_path: Path to the video file.
            metadata: Video metadata.
            progress_callback: Optional progress callback.

        Returns:
            Video ID of the uploaded video.

        Raises:
            Exception: If upload fails after all retries.
        """
        service = self._get_authenticated_service()

        # Build request body
        body = {
            "snippet": {
                "title": metadata.title,
                "description": metadata.description,
                "tags": metadata.tags,
                "categoryId": metadata.category,
            },
            "status": {
                "privacyStatus": metadata.privacy.value,
                "selfDeclaredMadeForKids": False,
            },
        }

        # Add notification setting for public videos
        if metadata.privacy == YouTubePrivacy.PUBLIC:
            body["status"]["publishAt"] = None  # Publish immediately
            if self.config.notify_subscribers:
                body["status"]["publicStatsViewable"] = True

        # Create media upload object
        media = MediaFileUpload(
            str(video_path),
            chunksize=1024 * 1024 * 10,  # 10MB chunks
            resumable=True,
            mimetype=self._get_video_mimetype(video_path)
        )

        # Create insert request
        request = service.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
            notifySubscribers=self.config.notify_subscribers
        )

        # Execute with retry logic
        response = None
        error = None
        retry = 0

        while response is None:
            try:
                logger.info(f"Uploading {video_path.name}...")
                status, response = request.next_chunk()

                if status:
                    progress = status.progress()
                    logger.debug(f"Upload progress: {progress * 100:.1f}%")
                    if progress_callback:
                        progress_callback(progress)

            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retriable errors
                    error = e
                    retry += 1
                    if retry > self.MAX_RETRIES:
                        raise Exception(f"Max retries exceeded: {error}")

                    wait_time = 2 ** retry
                    logger.warning(f"Retriable error, waiting {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    # Non-retriable error
                    raise

            except Exception as e:
                error_type = type(e).__name__
                if error_type in self.RETRY_EXCEPTIONS:
                    error = e
                    retry += 1
                    if retry > self.MAX_RETRIES:
                        raise Exception(f"Max retries exceeded: {error}")

                    wait_time = 2 ** retry
                    logger.warning(f"Retriable error, waiting {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise

        video_id = response["id"]
        logger.info(f"Upload complete: https://www.youtube.com/watch?v={video_id}")

        if progress_callback:
            progress_callback(1.0)

        return video_id

    def _get_video_mimetype(self, video_path: Path) -> str:
        """Determine video MIME type from file extension.

        Args:
            video_path: Path to the video file.

        Returns:
            MIME type string.
        """
        extension = video_path.suffix.lower()
        mimetypes = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".wmv": "video/x-ms-wmv",
            ".flv": "video/x-flv",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".m4v": "video/x-m4v",
            ".3gp": "video/3gpp",
            ".3g2": "video/3gpp2",
        }
        return mimetypes.get(extension, "video/*")

    def set_thumbnail(self, video_id: str, thumbnail_path: Path) -> bool:
        """Set a custom thumbnail for an uploaded video.

        Args:
            video_id: YouTube video ID.
            thumbnail_path: Path to thumbnail image (JPEG, PNG, etc.).

        Returns:
            True if thumbnail was set successfully.
        """
        if not thumbnail_path.exists():
            logger.error(f"Thumbnail file not found: {thumbnail_path}")
            return False

        try:
            service = self._get_authenticated_service()

            media = MediaFileUpload(
                str(thumbnail_path),
                mimetype=self._get_image_mimetype(thumbnail_path)
            )

            service.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()

            logger.info(f"Thumbnail set for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set thumbnail: {e}")
            return False

    def _get_image_mimetype(self, image_path: Path) -> str:
        """Determine image MIME type from file extension.

        Args:
            image_path: Path to the image file.

        Returns:
            MIME type string.
        """
        extension = image_path.suffix.lower()
        mimetypes = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        return mimetypes.get(extension, "image/*")

    def add_to_playlist(self, video_id: str, playlist_id: str) -> bool:
        """Add an uploaded video to a playlist.

        Args:
            video_id: YouTube video ID.
            playlist_id: YouTube playlist ID.

        Returns:
            True if video was added successfully.
        """
        try:
            service = self._get_authenticated_service()

            body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }

            service.playlistItems().insert(
                part="snippet",
                body=body
            ).execute()

            logger.info(f"Video {video_id} added to playlist {playlist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add to playlist: {e}")
            return False


def upload_to_youtube(
    video_path: Path,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    privacy: YouTubePrivacy = YouTubePrivacy.PRIVATE,
    category: str = "22",
    client_secrets_path: Optional[Path] = None,
    credentials_path: Optional[Path] = None,
    thumbnail_path: Optional[Path] = None,
    playlist_id: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> UploadResult:
    """Convenience function to upload a video to YouTube.

    Creates a configured uploader, authenticates, and uploads in one call.

    Args:
        video_path: Path to the video file.
        title: Video title.
        description: Video description.
        tags: List of video tags.
        privacy: Privacy setting (PUBLIC, UNLISTED, PRIVATE).
        category: YouTube category ID.
        client_secrets_path: Path to OAuth client secrets JSON.
        credentials_path: Path to store/load credentials.
        thumbnail_path: Optional custom thumbnail image.
        playlist_id: Optional playlist to add video to.
        progress_callback: Optional callback for upload progress.

    Returns:
        UploadResult with success status and video details.

    Example:
        result = upload_to_youtube(
            video_path=Path("restored_video.mp4"),
            title="My Restored Video",
            description="Restored with FrameWright",
            tags=["restoration", "upscale"],
            privacy=YouTubePrivacy.UNLISTED,
            client_secrets_path=Path("client_secrets.json")
        )
        if result.success:
            print(f"Video URL: {result.video_url}")
    """
    if not _YOUTUBE_API_AVAILABLE:
        return UploadResult(
            success=False,
            error_message=_MISSING_DEPS_MESSAGE
        )

    # Build configuration
    config = YouTubeConfig(
        client_secrets_path=client_secrets_path,
        default_privacy=privacy,
        default_category=category,
    )

    if credentials_path:
        config.credentials_path = credentials_path

    # Build metadata
    metadata = VideoMetadata(
        title=title,
        description=description,
        tags=tags or [],
        category=category,
        privacy=privacy,
        thumbnail_path=thumbnail_path,
        playlist_id=playlist_id,
    )

    # Create uploader and authenticate
    try:
        uploader = YouTubeUploader(config)
    except ImportError as e:
        return UploadResult(success=False, error_message=str(e))

    if not uploader.authenticate():
        return UploadResult(
            success=False,
            error_message="Failed to authenticate with YouTube. Check client_secrets_path and try again."
        )

    # Upload
    return uploader.upload(video_path, metadata, progress_callback)
