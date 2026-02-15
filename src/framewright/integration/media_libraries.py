"""Media library integration for Plex, Jellyfin, and Emby.

Provides connectors for popular media servers to automatically add
restored videos to libraries and trigger scans.
"""

import json
import logging
import ssl
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode, quote

logger = logging.getLogger(__name__)


class MediaServer(Enum):
    """Supported media server types."""
    PLEX = "plex"
    JELLYFIN = "jellyfin"
    EMBY = "emby"


@dataclass
class MediaServerConfig:
    """Configuration for a media server connection."""
    server_type: MediaServer
    server_url: str
    api_token: str
    library_name: Optional[str] = None
    auto_scan: bool = True
    verify_ssl: bool = True


class PlexConnector:
    """Connector for Plex Media Server.

    Provides methods to interact with Plex libraries, trigger scans,
    and add restored videos to the media library.
    """

    def __init__(self, config: MediaServerConfig):
        """Initialize Plex connector.

        Args:
            config: Media server configuration with Plex credentials.
        """
        if config.server_type != MediaServer.PLEX:
            raise ValueError("PlexConnector requires PLEX server type")

        self.config = config
        self.base_url = config.server_url.rstrip("/")
        self.token = config.api_token
        self.verify_ssl = config.verify_ssl
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context based on configuration."""
        if self.verify_ssl:
            return ssl.create_default_context()
        else:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Plex API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "X-Plex-Token": self.token,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Plex server.

        Args:
            endpoint: API endpoint path.
            method: HTTP method.
            data: Request body data.
            timeout: Request timeout in seconds.

        Returns:
            JSON response data or None on error.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            body = None
            if data:
                body = json.dumps(data).encode("utf-8")

            request = Request(url, data=body, headers=headers, method=method)
            response = urlopen(request, timeout=timeout, context=self._ssl_context)

            content = response.read().decode("utf-8")
            if content:
                return json.loads(content)
            return {}

        except HTTPError as e:
            logger.error(f"Plex API HTTP error: {e.code} - {e.reason}")
            return None
        except URLError as e:
            logger.error(f"Plex API URL error: {e.reason}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Plex API JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Plex API error: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Plex server.

        Returns:
            True if connection successful, False otherwise.
        """
        result = self._make_request("/")
        if result is not None:
            logger.info("Successfully connected to Plex server")
            return True
        logger.warning("Failed to connect to Plex server")
        return False

    def get_libraries(self) -> List[str]:
        """Get list of library names from Plex server.

        Returns:
            List of library names.
        """
        result = self._make_request("/library/sections")
        if result is None:
            return []

        libraries = []
        try:
            media_container = result.get("MediaContainer", {})
            directories = media_container.get("Directory", [])
            for directory in directories:
                if isinstance(directory, dict):
                    title = directory.get("title")
                    if title:
                        libraries.append(title)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing Plex libraries: {e}")

        return libraries

    def _get_library_key(self, library_name: str) -> Optional[str]:
        """Get library key by name.

        Args:
            library_name: Name of the library.

        Returns:
            Library key or None if not found.
        """
        result = self._make_request("/library/sections")
        if result is None:
            return None

        try:
            media_container = result.get("MediaContainer", {})
            directories = media_container.get("Directory", [])
            for directory in directories:
                if isinstance(directory, dict) and directory.get("title") == library_name:
                    return directory.get("key")
        except (KeyError, TypeError) as e:
            logger.error(f"Error finding library key: {e}")

        return None

    def trigger_scan(self, library_name: Optional[str] = None) -> bool:
        """Trigger library scan on Plex server.

        Args:
            library_name: Specific library to scan, or None for all libraries.

        Returns:
            True if scan triggered successfully.
        """
        if library_name:
            library_key = self._get_library_key(library_name)
            if library_key is None:
                logger.error(f"Library '{library_name}' not found")
                return False
            endpoint = f"/library/sections/{library_key}/refresh"
        else:
            endpoint = "/library/sections/all/refresh"

        result = self._make_request(endpoint, method="GET")
        if result is not None:
            logger.info(f"Triggered Plex library scan: {library_name or 'all'}")
            return True
        return False

    def add_to_library(self, video_path: Path, library_name: str) -> bool:
        """Add a video to a Plex library.

        Note: Plex automatically detects files in library paths.
        This method triggers a scan of the specific library to pick up
        the new file. The video_path should be within a library's
        configured paths.

        Args:
            video_path: Path to the video file.
            library_name: Name of the target library.

        Returns:
            True if operation successful.
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        # Trigger scan to pick up the new file
        if self.trigger_scan(library_name):
            logger.info(f"Added video to Plex library '{library_name}': {video_path}")
            return True
        return False


class JellyfinConnector:
    """Connector for Jellyfin Media Server.

    Provides methods to interact with Jellyfin libraries, trigger scans,
    and add restored videos to the media library.
    """

    def __init__(self, config: MediaServerConfig):
        """Initialize Jellyfin connector.

        Args:
            config: Media server configuration with Jellyfin credentials.
        """
        if config.server_type != MediaServer.JELLYFIN:
            raise ValueError("JellyfinConnector requires JELLYFIN server type")

        self.config = config
        self.base_url = config.server_url.rstrip("/")
        self.token = config.api_token
        self.verify_ssl = config.verify_ssl
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context based on configuration."""
        if self.verify_ssl:
            return ssl.create_default_context()
        else:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Jellyfin API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "X-Emby-Token": self.token,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Jellyfin server.

        Args:
            endpoint: API endpoint path.
            method: HTTP method.
            data: Request body data.
            timeout: Request timeout in seconds.

        Returns:
            JSON response data or None on error.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            body = None
            if data:
                body = json.dumps(data).encode("utf-8")

            request = Request(url, data=body, headers=headers, method=method)
            response = urlopen(request, timeout=timeout, context=self._ssl_context)

            content = response.read().decode("utf-8")
            if content:
                return json.loads(content)
            return {}

        except HTTPError as e:
            logger.error(f"Jellyfin API HTTP error: {e.code} - {e.reason}")
            return None
        except URLError as e:
            logger.error(f"Jellyfin API URL error: {e.reason}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Jellyfin API JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Jellyfin API error: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Jellyfin server.

        Returns:
            True if connection successful, False otherwise.
        """
        result = self._make_request("/System/Info")
        if result is not None:
            server_name = result.get("ServerName", "Unknown")
            logger.info(f"Successfully connected to Jellyfin server: {server_name}")
            return True
        logger.warning("Failed to connect to Jellyfin server")
        return False

    def get_libraries(self) -> List[str]:
        """Get list of library names from Jellyfin server.

        Returns:
            List of library names.
        """
        result = self._make_request("/Library/VirtualFolders")
        if result is None:
            return []

        libraries = []
        try:
            if isinstance(result, list):
                for folder in result:
                    if isinstance(folder, dict):
                        name = folder.get("Name")
                        if name:
                            libraries.append(name)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing Jellyfin libraries: {e}")

        return libraries

    def _get_library_id(self, library_name: str) -> Optional[str]:
        """Get library ID by name.

        Args:
            library_name: Name of the library.

        Returns:
            Library ID or None if not found.
        """
        result = self._make_request("/Library/VirtualFolders")
        if result is None:
            return None

        try:
            if isinstance(result, list):
                for folder in result:
                    if isinstance(folder, dict) and folder.get("Name") == library_name:
                        return folder.get("ItemId")
        except (KeyError, TypeError) as e:
            logger.error(f"Error finding library ID: {e}")

        return None

    def trigger_scan(self, library_name: Optional[str] = None) -> bool:
        """Trigger library scan on Jellyfin server.

        Args:
            library_name: Specific library to scan, or None for all libraries.

        Returns:
            True if scan triggered successfully.
        """
        if library_name:
            library_id = self._get_library_id(library_name)
            if library_id is None:
                logger.error(f"Library '{library_name}' not found")
                return False
            endpoint = f"/Items/{library_id}/Refresh"
        else:
            endpoint = "/Library/Refresh"

        result = self._make_request(endpoint, method="POST")
        if result is not None:
            logger.info(f"Triggered Jellyfin library scan: {library_name or 'all'}")
            return True
        return False

    def add_to_library(self, video_path: Path, library_name: str) -> bool:
        """Add a video to a Jellyfin library.

        Note: Jellyfin automatically detects files in library paths.
        This method triggers a scan of the specific library to pick up
        the new file. The video_path should be within a library's
        configured paths.

        Args:
            video_path: Path to the video file.
            library_name: Name of the target library.

        Returns:
            True if operation successful.
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        # Trigger scan to pick up the new file
        if self.trigger_scan(library_name):
            logger.info(f"Added video to Jellyfin library '{library_name}': {video_path}")
            return True
        return False


class EmbyConnector:
    """Connector for Emby Media Server.

    Provides methods to interact with Emby libraries, trigger scans,
    and add restored videos to the media library.

    Note: Emby API is largely compatible with Jellyfin.
    """

    def __init__(self, config: MediaServerConfig):
        """Initialize Emby connector.

        Args:
            config: Media server configuration with Emby credentials.
        """
        if config.server_type != MediaServer.EMBY:
            raise ValueError("EmbyConnector requires EMBY server type")

        self.config = config
        self.base_url = config.server_url.rstrip("/")
        self.token = config.api_token
        self.verify_ssl = config.verify_ssl
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context based on configuration."""
        if self.verify_ssl:
            return ssl.create_default_context()
        else:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Emby API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "X-Emby-Token": self.token,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Emby server.

        Args:
            endpoint: API endpoint path.
            method: HTTP method.
            data: Request body data.
            timeout: Request timeout in seconds.

        Returns:
            JSON response data or None on error.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            body = None
            if data:
                body = json.dumps(data).encode("utf-8")

            request = Request(url, data=body, headers=headers, method=method)
            response = urlopen(request, timeout=timeout, context=self._ssl_context)

            content = response.read().decode("utf-8")
            if content:
                return json.loads(content)
            return {}

        except HTTPError as e:
            logger.error(f"Emby API HTTP error: {e.code} - {e.reason}")
            return None
        except URLError as e:
            logger.error(f"Emby API URL error: {e.reason}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Emby API JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Emby API error: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Emby server.

        Returns:
            True if connection successful, False otherwise.
        """
        result = self._make_request("/System/Info")
        if result is not None:
            server_name = result.get("ServerName", "Unknown")
            logger.info(f"Successfully connected to Emby server: {server_name}")
            return True
        logger.warning("Failed to connect to Emby server")
        return False

    def get_libraries(self) -> List[str]:
        """Get list of library names from Emby server.

        Returns:
            List of library names.
        """
        result = self._make_request("/Library/VirtualFolders")
        if result is None:
            return []

        libraries = []
        try:
            if isinstance(result, list):
                for folder in result:
                    if isinstance(folder, dict):
                        name = folder.get("Name")
                        if name:
                            libraries.append(name)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing Emby libraries: {e}")

        return libraries

    def _get_library_id(self, library_name: str) -> Optional[str]:
        """Get library ID by name.

        Args:
            library_name: Name of the library.

        Returns:
            Library ID or None if not found.
        """
        result = self._make_request("/Library/VirtualFolders")
        if result is None:
            return None

        try:
            if isinstance(result, list):
                for folder in result:
                    if isinstance(folder, dict) and folder.get("Name") == library_name:
                        return folder.get("ItemId")
        except (KeyError, TypeError) as e:
            logger.error(f"Error finding library ID: {e}")

        return None

    def trigger_scan(self, library_name: Optional[str] = None) -> bool:
        """Trigger library scan on Emby server.

        Args:
            library_name: Specific library to scan, or None for all libraries.

        Returns:
            True if scan triggered successfully.
        """
        if library_name:
            library_id = self._get_library_id(library_name)
            if library_id is None:
                logger.error(f"Library '{library_name}' not found")
                return False
            endpoint = f"/Items/{library_id}/Refresh"
        else:
            endpoint = "/Library/Refresh"

        result = self._make_request(endpoint, method="POST")
        if result is not None:
            logger.info(f"Triggered Emby library scan: {library_name or 'all'}")
            return True
        return False

    def add_to_library(self, video_path: Path, library_name: str) -> bool:
        """Add a video to an Emby library.

        Note: Emby automatically detects files in library paths.
        This method triggers a scan of the specific library to pick up
        the new file. The video_path should be within a library's
        configured paths.

        Args:
            video_path: Path to the video file.
            library_name: Name of the target library.

        Returns:
            True if operation successful.
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        # Trigger scan to pick up the new file
        if self.trigger_scan(library_name):
            logger.info(f"Added video to Emby library '{library_name}': {video_path}")
            return True
        return False


class MediaLibraryManager:
    """Manager for multiple media server connections.

    Provides a unified interface to manage multiple media server
    configurations and perform operations across servers.
    """

    def __init__(self):
        """Initialize media library manager."""
        self._servers: Dict[str, MediaServerConfig] = {}
        self._connectors: Dict[str, Union[PlexConnector, JellyfinConnector, EmbyConnector]] = {}

    def add_server(self, config: MediaServerConfig, name: str) -> None:
        """Add a media server configuration.

        Args:
            config: Media server configuration.
            name: Unique name for this server.
        """
        self._servers[name] = config
        # Create connector based on server type
        if config.server_type == MediaServer.PLEX:
            self._connectors[name] = PlexConnector(config)
        elif config.server_type == MediaServer.JELLYFIN:
            self._connectors[name] = JellyfinConnector(config)
        elif config.server_type == MediaServer.EMBY:
            self._connectors[name] = EmbyConnector(config)
        logger.info(f"Added media server '{name}' ({config.server_type.value})")

    def remove_server(self, name: str) -> bool:
        """Remove a media server configuration.

        Args:
            name: Name of the server to remove.

        Returns:
            True if server was removed, False if not found.
        """
        if name in self._servers:
            del self._servers[name]
            del self._connectors[name]
            logger.info(f"Removed media server '{name}'")
            return True
        logger.warning(f"Media server '{name}' not found")
        return False

    def get_connector(
        self, name: str
    ) -> Union[PlexConnector, JellyfinConnector, EmbyConnector]:
        """Get connector for a named server.

        Args:
            name: Name of the server.

        Returns:
            Connector instance for the server.

        Raises:
            KeyError: If server not found.
        """
        if name not in self._connectors:
            raise KeyError(f"Media server '{name}' not found")
        return self._connectors[name]

    def list_servers(self) -> List[str]:
        """List all configured server names.

        Returns:
            List of server names.
        """
        return list(self._servers.keys())

    def add_restored_video(
        self, video_path: Path, server_name: str, library: str
    ) -> bool:
        """Add a restored video to a media library.

        Args:
            video_path: Path to the restored video file.
            server_name: Name of the media server.
            library: Name of the target library.

        Returns:
            True if video added successfully.
        """
        try:
            connector = self.get_connector(server_name)
            return connector.add_to_library(video_path, library)
        except KeyError as e:
            logger.error(str(e))
            return False

    def save_config(self, path: Path) -> None:
        """Save server configurations to a JSON file.

        Args:
            path: Path to save the configuration file.
        """
        config_data = {}
        for name, config in self._servers.items():
            config_data[name] = {
                "server_type": config.server_type.value,
                "server_url": config.server_url,
                "api_token": config.api_token,
                "library_name": config.library_name,
                "auto_scan": config.auto_scan,
                "verify_ssl": config.verify_ssl,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Saved media server configuration to {path}")

    def load_config(self, path: Path) -> None:
        """Load server configurations from a JSON file.

        Args:
            path: Path to the configuration file.

        Raises:
            FileNotFoundError: If config file not found.
            ValueError: If config file is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        for name, data in config_data.items():
            try:
                server_type = MediaServer(data["server_type"])
                config = MediaServerConfig(
                    server_type=server_type,
                    server_url=data["server_url"],
                    api_token=data["api_token"],
                    library_name=data.get("library_name"),
                    auto_scan=data.get("auto_scan", True),
                    verify_ssl=data.get("verify_ssl", True),
                )
                self.add_server(config, name)
            except (KeyError, ValueError) as e:
                logger.error(f"Error loading server '{name}': {e}")
                raise ValueError(f"Invalid configuration for server '{name}': {e}")

        logger.info(f"Loaded {len(config_data)} media server(s) from {path}")


def setup_plex(url: str, token: str, verify_ssl: bool = True) -> PlexConnector:
    """Factory function to create a Plex connector.

    Args:
        url: Plex server URL.
        token: Plex API token.
        verify_ssl: Whether to verify SSL certificates.

    Returns:
        Configured PlexConnector instance.
    """
    config = MediaServerConfig(
        server_type=MediaServer.PLEX,
        server_url=url,
        api_token=token,
        verify_ssl=verify_ssl,
    )
    return PlexConnector(config)


def setup_jellyfin(url: str, token: str, verify_ssl: bool = True) -> JellyfinConnector:
    """Factory function to create a Jellyfin connector.

    Args:
        url: Jellyfin server URL.
        token: Jellyfin API token.
        verify_ssl: Whether to verify SSL certificates.

    Returns:
        Configured JellyfinConnector instance.
    """
    config = MediaServerConfig(
        server_type=MediaServer.JELLYFIN,
        server_url=url,
        api_token=token,
        verify_ssl=verify_ssl,
    )
    return JellyfinConnector(config)


def setup_emby(url: str, token: str, verify_ssl: bool = True) -> EmbyConnector:
    """Factory function to create an Emby connector.

    Args:
        url: Emby server URL.
        token: Emby API token.
        verify_ssl: Whether to verify SSL certificates.

    Returns:
        Configured EmbyConnector instance.
    """
    config = MediaServerConfig(
        server_type=MediaServer.EMBY,
        server_url=url,
        api_token=token,
        verify_ssl=verify_ssl,
    )
    return EmbyConnector(config)
