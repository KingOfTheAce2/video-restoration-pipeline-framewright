"""Google Drive storage provider using rclone for FrameWright.

This module provides Google Drive integration via rclone, enabling:
- Direct upload/download of videos to/from Google Drive
- Streaming support for large files
- Integration with Vast.ai processing workflow

Requirements:
    - rclone installed and configured with a 'gdrive' remote
    - Run `rclone config` to set up Google Drive access
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import re
import json

from .base import CloudStorageProvider, StorageError

logger = logging.getLogger(__name__)


def _find_rclone_binary() -> Optional[str]:
    """Find rclone binary in PATH or common locations.

    Returns:
        Path to rclone binary or None if not found
    """
    path = shutil.which("rclone")
    if path:
        return path

    # Check common locations
    common_paths = [
        Path.home() / ".local" / "bin" / "rclone",
        Path("/usr/local/bin/rclone"),
        Path("/usr/bin/rclone"),
        Path("/opt/homebrew/bin/rclone"),
    ]

    for p in common_paths:
        if p.exists():
            return str(p)

    return None


class GoogleDriveStorage(CloudStorageProvider):
    """Google Drive storage provider using rclone.

    Uses rclone for robust Google Drive access with features like:
    - Resumable uploads/downloads
    - Progress tracking
    - Streaming support

    Example:
        >>> storage = GoogleDriveStorage(
        ...     remote_name="gdrive",  # rclone remote name
        ...     base_path="framewright/videos"  # Base folder in Drive
        ... )
        >>> uri = storage.upload(Path("video.mp4"), "input/video.mp4")
        >>> print(uri)  # gdrive:framewright/videos/input/video.mp4
    """

    def __init__(
        self,
        remote_name: str = "gdrive",
        base_path: str = "",
        bucket: Optional[str] = None,  # Alias for base_path for compatibility
        credentials: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Google Drive storage.

        Args:
            remote_name: Name of the rclone remote (default: gdrive)
            base_path: Base folder path in Google Drive
            bucket: Alias for base_path (for compatibility with other providers)
            credentials: Optional dict with rclone config overrides
        """
        effective_path = bucket or base_path
        super().__init__(bucket=effective_path, credentials=credentials or {})

        self._remote_name = remote_name
        self._base_path = effective_path
        self._rclone_path = _find_rclone_binary()

        if not self._rclone_path:
            raise StorageError(
                "rclone not found. Install with:\n"
                "  curl https://rclone.org/install.sh | sudo bash\n"
                "Then configure Google Drive:\n"
                "  rclone config"
            )

    @property
    def scheme(self) -> str:
        """Get the URI scheme for Google Drive."""
        return "gdrive"

    def _run_rclone(
        self,
        args: List[str],
        progress_callback: Optional[Callable[[float], None]] = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run rclone command with optional progress tracking.

        Args:
            args: Command arguments
            progress_callback: Optional progress callback (0-1)
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        cmd = [self._rclone_path] + args

        if progress_callback:
            # Use --progress for real-time updates
            cmd.append("--progress")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                # Parse progress from rclone output
                # Format: "Transferred: X / Y, 50%, 10 MB/s, ETA 1m30s"
                match = re.search(r"(\d+)%", line)
                if match:
                    progress_callback(int(match.group(1)) / 100)

            process.wait()

            if process.returncode != 0:
                raise StorageError(f"rclone command failed with code {process.returncode}")

            return subprocess.CompletedProcess(cmd, process.returncode)
        else:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise StorageError(f"rclone command failed: {error_msg}")

            return result

    def _get_remote_path(self, path: str) -> str:
        """Get full remote path including remote name and base path.

        Args:
            path: Relative path

        Returns:
            Full rclone remote path (e.g., "gdrive:folder/file.mp4")
        """
        if path.startswith(f"{self._remote_name}:"):
            return path

        # Strip any leading slashes
        path = path.lstrip("/")

        if self._base_path:
            full_path = f"{self._base_path}/{path}"
        else:
            full_path = path

        return f"{self._remote_name}:{full_path}"

    def check_remote_exists(self) -> bool:
        """Check if the rclone remote is configured.

        Returns:
            True if remote exists and is accessible
        """
        try:
            result = self._run_rclone(["listremotes"])
            remotes = result.stdout.strip().split("\n")
            return f"{self._remote_name}:" in remotes
        except Exception:
            return False

    def upload(
        self,
        local_path: Path,
        remote_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Upload file to Google Drive.

        Args:
            local_path: Local file path
            remote_path: Remote path within base folder
            progress_callback: Optional progress callback (0-1)

        Returns:
            Full Google Drive URI of uploaded file
        """
        if not local_path.exists():
            raise StorageError(f"File not found: {local_path}")

        full_remote = self._get_remote_path(remote_path)

        logger.info(f"Uploading {local_path} to {full_remote}")

        self._run_rclone(
            ["copyto", str(local_path), full_remote],
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(1.0)

        return full_remote

    def download(
        self,
        remote_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Download file from Google Drive.

        Args:
            remote_path: Remote path or full URI
            local_path: Local destination path
            progress_callback: Optional progress callback (0-1)
        """
        full_remote = self._get_remote_path(remote_path)

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {full_remote} to {local_path}")

        self._run_rclone(
            ["copyto", full_remote, str(local_path)],
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(1.0)

    def delete(self, remote_path: str) -> bool:
        """Delete file from Google Drive.

        Args:
            remote_path: Remote path or full URI

        Returns:
            True if deletion successful
        """
        full_remote = self._get_remote_path(remote_path)

        try:
            self._run_rclone(["deletefile", full_remote])
            return True
        except Exception:
            return False

    def exists(self, remote_path: str) -> bool:
        """Check if file exists in Google Drive.

        Args:
            remote_path: Remote path or full URI

        Returns:
            True if file exists
        """
        full_remote = self._get_remote_path(remote_path)

        try:
            result = self._run_rclone(["lsf", full_remote])
            return bool(result.stdout.strip())
        except Exception:
            return False

    def list_files(
        self,
        prefix: str = "",
        max_results: int = 1000,
    ) -> List[str]:
        """List files in Google Drive folder.

        Args:
            prefix: Path prefix to filter by
            max_results: Maximum number of results

        Returns:
            List of file paths
        """
        remote_path = self._get_remote_path(prefix) if prefix else f"{self._remote_name}:{self._base_path}"

        try:
            result = self._run_rclone([
                "lsf",
                "--recursive",
                f"--max-count={max_results}",
                remote_path,
            ])

            files = result.stdout.strip().split("\n")
            return [f for f in files if f]  # Filter empty strings

        except Exception:
            return []

    def get_file_size(self, remote_path: str) -> int:
        """Get size of remote file in bytes.

        Args:
            remote_path: Remote path or full URI

        Returns:
            File size in bytes
        """
        full_remote = self._get_remote_path(remote_path)

        try:
            result = self._run_rclone(["size", "--json", full_remote])
            data = json.loads(result.stdout)
            return data.get("bytes", 0)
        except Exception:
            return 0

    def stream_to_drive(
        self,
        stdin_source: str,
        remote_path: str,
    ) -> str:
        """Stream data directly to Google Drive using rclone rcat.

        This allows piping data (e.g., from yt-dlp) directly to Drive
        without storing locally first.

        Args:
            stdin_source: Command to generate data (e.g., "yt-dlp -o - URL")
            remote_path: Destination path in Drive

        Returns:
            Full remote path
        """
        full_remote = self._get_remote_path(remote_path)

        # Create pipeline: source | rclone rcat remote:path
        cmd = f"{stdin_source} | {self._rclone_path} rcat {full_remote}"

        logger.info(f"Streaming to {full_remote}")

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise StorageError(f"Stream to Drive failed: {result.stderr}")

        return full_remote

    def sync_folder(
        self,
        local_folder: Path,
        remote_folder: str,
        direction: str = "upload",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Sync entire folder between local and Drive.

        Args:
            local_folder: Local folder path
            remote_folder: Remote folder path
            direction: "upload" or "download"
            progress_callback: Optional progress callback
        """
        full_remote = self._get_remote_path(remote_folder)

        if direction == "upload":
            source = str(local_folder)
            dest = full_remote
        else:
            source = full_remote
            dest = str(local_folder)
            local_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Syncing {source} -> {dest}")

        self._run_rclone(
            ["sync", source, dest],
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(1.0)


def setup_gdrive_remote(remote_name: str = "gdrive") -> bool:
    """Interactive setup for Google Drive rclone remote.

    This runs `rclone config` to set up a new Google Drive remote.

    Args:
        remote_name: Name for the remote

    Returns:
        True if setup completed successfully
    """
    rclone_path = _find_rclone_binary()
    if not rclone_path:
        print("rclone not found. Install with:")
        print("  curl https://rclone.org/install.sh | sudo bash")
        return False

    print(f"\nSetting up Google Drive remote '{remote_name}'...")
    print("Follow the prompts to authenticate with Google.\n")

    try:
        subprocess.run([rclone_path, "config"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_gdrive_configured(remote_name: str = "gdrive") -> bool:
    """Check if Google Drive is configured in rclone.

    Args:
        remote_name: Name of the rclone remote

    Returns:
        True if configured and accessible
    """
    rclone_path = _find_rclone_binary()
    if not rclone_path:
        return False

    try:
        result = subprocess.run(
            [rclone_path, "listremotes"],
            capture_output=True,
            text=True,
        )
        return f"{remote_name}:" in result.stdout
    except Exception:
        return False
