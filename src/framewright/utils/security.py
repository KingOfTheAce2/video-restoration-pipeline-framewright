"""Security utilities for input validation and subprocess hardening.

This module provides comprehensive security features:
- PathValidator: Path traversal prevention and file extension validation
- InputSanitizer: User parameter sanitization and validation
- SecureSubprocess: Hardened subprocess execution wrapper
- SecurityAudit: Security event logging and monitoring

Designed to prevent:
- Command injection via ffmpeg/ffprobe parameters
- Path traversal attacks
- Resource exhaustion
- Information leakage through error messages
"""

import hashlib
import logging
import os
import re
import resource
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Allowed URL schemes for video downloads
ALLOWED_URL_SCHEMES: FrozenSet[str] = frozenset({'http', 'https'})

# Allowed video hosting domains (for URL validation)
ALLOWED_VIDEO_HOSTS: FrozenSet[str] = frozenset({
    'youtube.com',
    'www.youtube.com',
    'youtu.be',
    'vimeo.com',
    'www.vimeo.com',
    'dailymotion.com',
    'www.dailymotion.com',
    'twitch.tv',
    'www.twitch.tv',
    'streamable.com',
    'archive.org',
    'www.archive.org',
})

# Dangerous characters that should never appear in file paths
DANGEROUS_PATH_CHARS: FrozenSet[str] = frozenset({
    ';', '|', '&', '$', '`', '\n', '\r', '\x00', '\t',
    '<', '>', '"', "'", '\\', '\v', '\f',
})

# Maximum allowed path length
MAX_PATH_LENGTH: int = 4096

# Maximum filename length
MAX_FILENAME_LENGTH: int = 255

# Allowed file extensions for video files
ALLOWED_VIDEO_EXTENSIONS: FrozenSet[str] = frozenset({
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.mts', '.m2ts',
    '.vob', '.ogv', '.rm', '.rmvb', '.asf', '.divx',
})

# Allowed file extensions for image/frame files
ALLOWED_FRAME_EXTENSIONS: FrozenSet[str] = frozenset({
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp',
    '.ppm', '.pgm', '.pbm', '.exr', '.dpx',
})

# Allowed file extensions for audio files
ALLOWED_AUDIO_EXTENSIONS: FrozenSet[str] = frozenset({
    '.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma',
    '.opus', '.ac3', '.dts', '.pcm',
})

# Allowed video codecs (whitelist for ffmpeg -c:v)
ALLOWED_VIDEO_CODECS: FrozenSet[str] = frozenset({
    'libx264', 'libx265', 'libvpx', 'libvpx-vp9', 'libaom-av1',
    'h264', 'hevc', 'vp8', 'vp9', 'av1', 'mpeg4', 'mpeg2video',
    'rawvideo', 'png', 'mjpeg', 'prores', 'dnxhd', 'ffv1',
    'copy',  # Stream copy (no re-encoding)
})

# Allowed audio codecs (whitelist for ffmpeg -c:a)
ALLOWED_AUDIO_CODECS: FrozenSet[str] = frozenset({
    'aac', 'libmp3lame', 'mp3', 'flac', 'libvorbis', 'vorbis',
    'opus', 'libopus', 'pcm_s16le', 'pcm_s24le', 'pcm_s32le',
    'pcm_f32le', 'ac3', 'eac3', 'dts', 'alac', 'wavpack',
    'copy',  # Stream copy (no re-encoding)
})

# Allowed encoding presets
ALLOWED_ENCODING_PRESETS: FrozenSet[str] = frozenset({
    'ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
    'medium', 'slow', 'slower', 'veryslow', 'placebo',
})

# Allowed Real-ESRGAN model names
ALLOWED_ESRGAN_MODELS: FrozenSet[str] = frozenset({
    'realesrgan-x2plus', 'realesrgan-x4plus', 'realesrgan-x4plus-anime',
    'realesr-animevideov3', 'realesr-general-x4v3',
})

# Allowed RIFE model names
ALLOWED_RIFE_MODELS: FrozenSet[str] = frozenset({
    'rife-v2.3', 'rife-v4.0', 'rife-v4.6', 'rife-v4.7', 'rife-v4.8',
})

# Default subprocess timeout (10 minutes)
DEFAULT_SUBPROCESS_TIMEOUT: float = 600.0

# Maximum subprocess timeout (4 hours)
MAX_SUBPROCESS_TIMEOUT: float = 14400.0

# Default memory limit for subprocesses (8 GB)
DEFAULT_MEMORY_LIMIT_MB: int = 8192


# =============================================================================
# Custom Exceptions
# =============================================================================

class SecurityError(Exception):
    """Base class for security-related errors.

    These errors are intentionally vague to prevent information leakage.
    Detailed information is logged internally.
    """
    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails."""
    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attack is detected."""
    pass


class CommandInjectionError(SecurityError):
    """Raised when command injection attempt is detected."""
    pass


class ResourceLimitError(SecurityError):
    """Raised when resource limits are exceeded."""
    pass


class RateLimitError(SecurityError):
    """Raised when rate limits are exceeded."""
    pass


# =============================================================================
# Security Event Types
# =============================================================================

class SecurityEventType(Enum):
    """Types of security events for auditing."""
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    COMMAND_INJECTION_ATTEMPT = "command_injection_attempt"
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    SUBPROCESS_STARTED = "subprocess_started"
    SUBPROCESS_COMPLETED = "subprocess_completed"
    SUBPROCESS_FAILED = "subprocess_failed"
    SUBPROCESS_TIMEOUT = "subprocess_timeout"
    FILE_ACCESS = "file_access"
    FILE_WRITE = "file_write"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class SecurityEvent:
    """A security-relevant event for auditing."""
    event_type: SecurityEventType
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    source_ip: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging/storage."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
        }


# =============================================================================
# PathValidator Class
# =============================================================================

class PathValidator:
    """Validates file paths for security issues.

    Provides comprehensive path validation including:
    - Path traversal attack prevention (../ injection)
    - File extension whitelist validation
    - Null byte and injection pattern detection
    - Base directory confinement
    - Symlink resolution and validation

    Example:
        >>> validator = PathValidator(
        ...     base_dir=Path("/data/videos"),
        ...     allowed_extensions={".mp4", ".mkv"}
        ... )
        >>> safe_path = validator.validate("/data/videos/input.mp4")
        >>> # Raises PathTraversalError:
        >>> validator.validate("/data/videos/../../../etc/passwd")
    """

    # Patterns that indicate path traversal attempts
    TRAVERSAL_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r'\.\.[\\/]'),  # ../ or ..\
        re.compile(r'[\\/]\.\.'),  # /.. or \..
        re.compile(r'^\.\.'),      # Starts with ..
        re.compile(r'\.\.$'),      # Ends with ..
    )

    # Patterns for null byte and other injection attacks
    INJECTION_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r'\x00'),       # Null byte
        re.compile(r'%00'),        # URL-encoded null
        re.compile(r'%2e%2e'),     # URL-encoded ..
        re.compile(r'\.%00'),      # Null after extension
    )

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        allowed_extensions: Optional[Set[str]] = None,
        max_path_length: int = MAX_PATH_LENGTH,
        follow_symlinks: bool = False,
        audit_logger: Optional["SecurityAudit"] = None,
    ):
        """Initialize PathValidator.

        Args:
            base_dir: If provided, all paths must be within this directory
            allowed_extensions: Set of allowed file extensions (e.g., {".mp4", ".mkv"})
            max_path_length: Maximum allowed path length
            follow_symlinks: Whether to follow and validate symlink targets
            audit_logger: Optional SecurityAudit instance for logging
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.allowed_extensions = (
            frozenset(ext.lower() for ext in allowed_extensions)
            if allowed_extensions else None
        )
        self.max_path_length = max_path_length
        self.follow_symlinks = follow_symlinks
        self.audit = audit_logger

    def validate(
        self,
        path: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
    ) -> Path:
        """Validate a file path for security.

        Args:
            path: Path to validate
            must_exist: If True, path must exist
            must_be_file: If True, path must be a file
            must_be_dir: If True, path must be a directory

        Returns:
            Validated Path object (resolved to absolute path)

        Raises:
            InputValidationError: If path is invalid
            PathTraversalError: If path traversal is detected
        """
        if not path:
            raise InputValidationError("Path cannot be empty")

        path_str = str(path)

        # Check length
        if len(path_str) > self.max_path_length:
            self._log_security_event(
                SecurityEventType.INVALID_INPUT,
                f"Path exceeds maximum length: {len(path_str)} > {self.max_path_length}",
                severity="WARNING",
            )
            raise InputValidationError("Path exceeds maximum allowed length")

        # Check for dangerous characters
        self._check_dangerous_chars(path_str)

        # Check for null bytes and injection patterns
        self._check_injection_patterns(path_str)

        # Check for path traversal patterns BEFORE resolution
        self._check_traversal_patterns(path_str)

        # Convert to Path and resolve
        try:
            path_obj = Path(path)
            # Use strict=False to allow non-existent paths during validation
            resolved = path_obj.resolve()
        except (OSError, RuntimeError) as e:
            raise InputValidationError(f"Invalid path format: {self._sanitize_error(e)}")

        # Validate symlinks if enabled
        if self.follow_symlinks and path_obj.is_symlink():
            self._validate_symlink(path_obj)

        # Check base directory confinement
        if self.base_dir:
            self._check_base_dir_confinement(resolved, path_str)

        # Check file extension
        if self.allowed_extensions and resolved.suffix:
            ext = resolved.suffix.lower()
            if ext not in self.allowed_extensions:
                self._log_security_event(
                    SecurityEventType.INVALID_INPUT,
                    f"Disallowed file extension: {ext}",
                    severity="WARNING",
                )
                raise InputValidationError(
                    f"File extension not allowed. Permitted: {', '.join(sorted(self.allowed_extensions))}"
                )

        # Check existence
        if must_exist and not resolved.exists():
            raise InputValidationError("Path does not exist")

        # Check type
        if must_be_file and resolved.exists() and not resolved.is_file():
            raise InputValidationError("Path is not a file")

        if must_be_dir and resolved.exists() and not resolved.is_dir():
            raise InputValidationError("Path is not a directory")

        # Log successful file access
        if self.audit:
            self._log_security_event(
                SecurityEventType.FILE_ACCESS,
                f"Path validated: {resolved}",
                severity="INFO",
            )

        return resolved

    def validate_video_path(self, path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate a video file path.

        Args:
            path: Path to validate
            must_exist: If True, path must exist

        Returns:
            Validated Path object
        """
        # Temporarily override allowed extensions
        original_extensions = self.allowed_extensions
        self.allowed_extensions = ALLOWED_VIDEO_EXTENSIONS
        try:
            return self.validate(path, must_exist=must_exist, must_be_file=must_exist)
        finally:
            self.allowed_extensions = original_extensions

    def validate_frame_path(self, path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate a frame/image file path.

        Args:
            path: Path to validate
            must_exist: If True, path must exist

        Returns:
            Validated Path object
        """
        original_extensions = self.allowed_extensions
        self.allowed_extensions = ALLOWED_FRAME_EXTENSIONS
        try:
            return self.validate(path, must_exist=must_exist, must_be_file=must_exist)
        finally:
            self.allowed_extensions = original_extensions

    def validate_output_path(self, path: Union[str, Path]) -> Path:
        """Validate an output path (need not exist, but parent must be writable).

        Args:
            path: Path to validate

        Returns:
            Validated Path object

        Raises:
            InputValidationError: If path is invalid or parent is not writable
        """
        validated = self.validate(path, must_exist=False)

        # Check parent directory exists and is writable
        parent = validated.parent
        if not parent.exists():
            raise InputValidationError("Parent directory does not exist")

        if not os.access(parent, os.W_OK):
            raise InputValidationError("Parent directory is not writable")

        return validated

    def _check_dangerous_chars(self, path_str: str) -> None:
        """Check for dangerous characters in path."""
        for char in DANGEROUS_PATH_CHARS:
            if char in path_str:
                self._log_security_event(
                    SecurityEventType.SUSPICIOUS_PATTERN,
                    f"Dangerous character detected in path: {repr(char)}",
                    severity="WARNING",
                    details={"path_hash": self._hash_for_log(path_str)},
                )
                raise InputValidationError("Path contains invalid characters")

    def _check_injection_patterns(self, path_str: str) -> None:
        """Check for null byte and other injection patterns."""
        path_lower = path_str.lower()
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(path_lower):
                self._log_security_event(
                    SecurityEventType.COMMAND_INJECTION_ATTEMPT,
                    "Injection pattern detected in path",
                    severity="ERROR",
                    details={"path_hash": self._hash_for_log(path_str)},
                )
                raise PathTraversalError("Invalid path format")

    def _check_traversal_patterns(self, path_str: str) -> None:
        """Check for path traversal patterns."""
        for pattern in self.TRAVERSAL_PATTERNS:
            if pattern.search(path_str):
                self._log_security_event(
                    SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
                    "Path traversal pattern detected",
                    severity="ERROR",
                    details={"path_hash": self._hash_for_log(path_str)},
                )
                raise PathTraversalError("Path traversal not allowed")

    def _check_base_dir_confinement(self, resolved: Path, original: str) -> None:
        """Ensure resolved path is within base directory."""
        try:
            resolved.relative_to(self.base_dir)
        except ValueError:
            self._log_security_event(
                SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
                f"Path escapes base directory: {self.base_dir}",
                severity="ERROR",
                details={"path_hash": self._hash_for_log(original)},
            )
            raise PathTraversalError("Path is outside allowed directory")

    def _validate_symlink(self, path: Path) -> None:
        """Validate symlink target is within allowed boundaries."""
        try:
            target = path.resolve()
            if self.base_dir:
                target.relative_to(self.base_dir)
        except ValueError:
            raise PathTraversalError("Symlink target is outside allowed directory")
        except OSError:
            raise InputValidationError("Cannot resolve symlink target")

    def _hash_for_log(self, value: str) -> str:
        """Create a hash of a value for secure logging."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _sanitize_error(self, error: Exception) -> str:
        """Sanitize error message to prevent information leakage."""
        # Return generic message, log details internally
        logger.debug(f"Path validation error details: {error}")
        return "Invalid path format"

    def _log_security_event(
        self,
        event_type: SecurityEventType,
        message: str,
        severity: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a security event."""
        if self.audit:
            self.audit.log_event(
                SecurityEvent(
                    event_type=event_type,
                    message=message,
                    severity=severity,
                    details=details or {},
                )
            )
        else:
            log_func = getattr(logger, severity.lower(), logger.info)
            log_func(f"Security: {event_type.value} - {message}")


# =============================================================================
# InputSanitizer Class
# =============================================================================

class InputSanitizer:
    """Sanitizes and validates user-provided parameters.

    Provides validation for:
    - Numeric ranges (FPS, CRF, scale factors)
    - Codec names against whitelists
    - Output filenames
    - FFmpeg/ffprobe parameters

    Example:
        >>> sanitizer = InputSanitizer()
        >>> fps = sanitizer.validate_fps(60)  # OK
        >>> fps = sanitizer.validate_fps(300)  # Raises InputValidationError
        >>> codec = sanitizer.validate_video_codec("libx264")  # OK
        >>> codec = sanitizer.validate_video_codec("; rm -rf /")  # Raises
    """

    # FPS limits
    MIN_FPS: float = 1.0
    MAX_FPS: float = 240.0

    # CRF limits (lower = better quality)
    MIN_CRF: int = 0
    MAX_CRF: int = 51

    # Scale factor limits
    ALLOWED_SCALE_FACTORS: FrozenSet[int] = frozenset({1, 2, 3, 4, 8})

    # Tile size limits
    MIN_TILE_SIZE: int = 32
    MAX_TILE_SIZE: int = 4096

    # Bitrate limits (in kbps)
    MIN_BITRATE: int = 100
    MAX_BITRATE: int = 500000

    # Resolution limits
    MIN_RESOLUTION: int = 1
    MAX_RESOLUTION: int = 16384

    # Pattern for safe filenames
    SAFE_FILENAME_PATTERN: re.Pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')

    # Pattern for FFmpeg filter values (very restrictive)
    SAFE_FILTER_VALUE_PATTERN: re.Pattern = re.compile(r'^[a-zA-Z0-9._:=,/-]+$')

    def __init__(self, audit_logger: Optional["SecurityAudit"] = None):
        """Initialize InputSanitizer.

        Args:
            audit_logger: Optional SecurityAudit instance for logging
        """
        self.audit = audit_logger

    def validate_fps(self, fps: Union[int, float], name: str = "FPS") -> float:
        """Validate frames per second value.

        Args:
            fps: FPS value to validate
            name: Name for error messages

        Returns:
            Validated FPS as float

        Raises:
            InputValidationError: If FPS is out of range
        """
        try:
            fps_float = float(fps)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be a number")

        if fps_float < self.MIN_FPS or fps_float > self.MAX_FPS:
            raise InputValidationError(
                f"{name} must be between {self.MIN_FPS} and {self.MAX_FPS}, got {fps_float}"
            )

        return fps_float

    def validate_crf(self, crf: int, name: str = "CRF") -> int:
        """Validate Constant Rate Factor value.

        Args:
            crf: CRF value to validate
            name: Name for error messages

        Returns:
            Validated CRF as integer

        Raises:
            InputValidationError: If CRF is out of range
        """
        try:
            crf_int = int(crf)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be an integer")

        if crf_int < self.MIN_CRF or crf_int > self.MAX_CRF:
            raise InputValidationError(
                f"{name} must be between {self.MIN_CRF} and {self.MAX_CRF}, got {crf_int}"
            )

        return crf_int

    def validate_scale_factor(self, scale: int, name: str = "Scale factor") -> int:
        """Validate upscaling factor.

        Args:
            scale: Scale factor to validate
            name: Name for error messages

        Returns:
            Validated scale factor

        Raises:
            InputValidationError: If scale factor is not allowed
        """
        try:
            scale_int = int(scale)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be an integer")

        if scale_int not in self.ALLOWED_SCALE_FACTORS:
            raise InputValidationError(
                f"{name} must be one of {sorted(self.ALLOWED_SCALE_FACTORS)}, got {scale_int}"
            )

        return scale_int

    def validate_tile_size(self, tile_size: int, name: str = "Tile size") -> int:
        """Validate tile size for processing.

        Args:
            tile_size: Tile size to validate (0 for auto)
            name: Name for error messages

        Returns:
            Validated tile size

        Raises:
            InputValidationError: If tile size is out of range
        """
        try:
            size_int = int(tile_size)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be an integer")

        # 0 means auto
        if size_int == 0:
            return 0

        if size_int < self.MIN_TILE_SIZE or size_int > self.MAX_TILE_SIZE:
            raise InputValidationError(
                f"{name} must be between {self.MIN_TILE_SIZE} and {self.MAX_TILE_SIZE} "
                f"(or 0 for auto), got {size_int}"
            )

        # Prefer power of 2 for GPU efficiency
        if size_int & (size_int - 1) != 0:
            logger.warning(f"Tile size {size_int} is not a power of 2, may reduce performance")

        return size_int

    def validate_resolution(
        self,
        width: int,
        height: int,
    ) -> Tuple[int, int]:
        """Validate video resolution.

        Args:
            width: Width in pixels
            height: Height in pixels

        Returns:
            Tuple of (width, height)

        Raises:
            InputValidationError: If resolution is invalid
        """
        try:
            w = int(width)
            h = int(height)
        except (TypeError, ValueError):
            raise InputValidationError("Resolution must be integers")

        if w < self.MIN_RESOLUTION or w > self.MAX_RESOLUTION:
            raise InputValidationError(
                f"Width must be between {self.MIN_RESOLUTION} and {self.MAX_RESOLUTION}"
            )

        if h < self.MIN_RESOLUTION or h > self.MAX_RESOLUTION:
            raise InputValidationError(
                f"Height must be between {self.MIN_RESOLUTION} and {self.MAX_RESOLUTION}"
            )

        return (w, h)

    def validate_video_codec(self, codec: str, name: str = "Video codec") -> str:
        """Validate video codec name.

        Args:
            codec: Codec name to validate
            name: Name for error messages

        Returns:
            Validated codec name

        Raises:
            InputValidationError: If codec is not in whitelist
        """
        if not codec or not isinstance(codec, str):
            raise InputValidationError(f"{name} must be a non-empty string")

        codec_clean = codec.strip().lower()

        if codec_clean not in ALLOWED_VIDEO_CODECS:
            self._log_suspicious_input("video_codec", codec)
            raise InputValidationError(
                f"{name} '{codec}' is not allowed. "
                f"Permitted codecs: {', '.join(sorted(ALLOWED_VIDEO_CODECS))}"
            )

        return codec_clean

    def validate_audio_codec(self, codec: str, name: str = "Audio codec") -> str:
        """Validate audio codec name.

        Args:
            codec: Codec name to validate
            name: Name for error messages

        Returns:
            Validated codec name

        Raises:
            InputValidationError: If codec is not in whitelist
        """
        if not codec or not isinstance(codec, str):
            raise InputValidationError(f"{name} must be a non-empty string")

        codec_clean = codec.strip().lower()

        if codec_clean not in ALLOWED_AUDIO_CODECS:
            self._log_suspicious_input("audio_codec", codec)
            raise InputValidationError(
                f"{name} '{codec}' is not allowed. "
                f"Permitted codecs: {', '.join(sorted(ALLOWED_AUDIO_CODECS))}"
            )

        return codec_clean

    def validate_encoding_preset(self, preset: str, name: str = "Preset") -> str:
        """Validate encoding preset name.

        Args:
            preset: Preset name to validate
            name: Name for error messages

        Returns:
            Validated preset name

        Raises:
            InputValidationError: If preset is not in whitelist
        """
        if not preset or not isinstance(preset, str):
            raise InputValidationError(f"{name} must be a non-empty string")

        preset_clean = preset.strip().lower()

        if preset_clean not in ALLOWED_ENCODING_PRESETS:
            self._log_suspicious_input("encoding_preset", preset)
            raise InputValidationError(
                f"{name} '{preset}' is not allowed. "
                f"Permitted presets: {', '.join(sorted(ALLOWED_ENCODING_PRESETS))}"
            )

        return preset_clean

    def validate_esrgan_model(self, model: str, name: str = "Model") -> str:
        """Validate Real-ESRGAN model name.

        Args:
            model: Model name to validate
            name: Name for error messages

        Returns:
            Validated model name

        Raises:
            InputValidationError: If model is not in whitelist
        """
        if not model or not isinstance(model, str):
            raise InputValidationError(f"{name} must be a non-empty string")

        model_clean = model.strip().lower()

        if model_clean not in ALLOWED_ESRGAN_MODELS:
            self._log_suspicious_input("esrgan_model", model)
            raise InputValidationError(
                f"{name} '{model}' is not allowed. "
                f"Permitted models: {', '.join(sorted(ALLOWED_ESRGAN_MODELS))}"
            )

        return model_clean

    def validate_rife_model(self, model: str, name: str = "RIFE model") -> str:
        """Validate RIFE model name.

        Args:
            model: Model name to validate
            name: Name for error messages

        Returns:
            Validated model name

        Raises:
            InputValidationError: If model is not in whitelist
        """
        if not model or not isinstance(model, str):
            raise InputValidationError(f"{name} must be a non-empty string")

        model_clean = model.strip().lower()

        if model_clean not in ALLOWED_RIFE_MODELS:
            self._log_suspicious_input("rife_model", model)
            raise InputValidationError(
                f"{name} '{model}' is not allowed. "
                f"Permitted models: {', '.join(sorted(ALLOWED_RIFE_MODELS))}"
            )

        return model_clean

    def sanitize_filename(
        self,
        filename: str,
        max_length: int = MAX_FILENAME_LENGTH,
        default: str = "output",
    ) -> str:
        """Sanitize a filename to remove dangerous characters.

        Args:
            filename: Original filename
            max_length: Maximum allowed length
            default: Default name if filename becomes empty

        Returns:
            Sanitized filename
        """
        if not filename:
            return default

        # Remove null bytes first
        filename = filename.replace('\x00', '')

        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')

        # Remove other dangerous characters
        for char in DANGEROUS_PATH_CHARS:
            filename = filename.replace(char, '_')

        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)

        # Collapse multiple underscores
        filename = re.sub(r'_+', '_', filename)

        # Remove leading/trailing underscores and dots
        filename = filename.strip('_.')

        # Limit length while preserving extension
        if len(filename) > max_length:
            base, ext = os.path.splitext(filename)
            if ext:
                max_base = max_length - len(ext)
                base = base[:max_base]
                filename = f"{base}{ext}"
            else:
                filename = filename[:max_length]

        # Handle empty result
        if not filename or filename.isspace():
            return default

        return filename

    def validate_ffmpeg_filter_value(self, value: str, name: str = "Filter value") -> str:
        """Validate a value for use in FFmpeg filter expressions.

        This is very restrictive to prevent command injection.

        Args:
            value: Value to validate
            name: Name for error messages

        Returns:
            Validated value

        Raises:
            InputValidationError: If value contains unsafe characters
        """
        if not value or not isinstance(value, str):
            raise InputValidationError(f"{name} must be a non-empty string")

        if not self.SAFE_FILTER_VALUE_PATTERN.match(value):
            self._log_suspicious_input("ffmpeg_filter", value)
            raise InputValidationError(
                f"{name} contains invalid characters. "
                "Only alphanumeric, '.', '_', ':', '=', ',', '/', and '-' are allowed."
            )

        return value

    def validate_integer_range(
        self,
        value: int,
        min_val: int,
        max_val: int,
        name: str = "Value",
    ) -> int:
        """Validate that an integer is within range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages

        Returns:
            Validated integer

        Raises:
            InputValidationError: If value is out of range
        """
        try:
            int_val = int(value)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be an integer")

        if int_val < min_val or int_val > max_val:
            raise InputValidationError(
                f"{name} must be between {min_val} and {max_val}, got {int_val}"
            )

        return int_val

    def validate_float_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        name: str = "Value",
    ) -> float:
        """Validate that a float is within range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages

        Returns:
            Validated float

        Raises:
            InputValidationError: If value is out of range
        """
        try:
            float_val = float(value)
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be a number")

        if float_val < min_val or float_val > max_val:
            raise InputValidationError(
                f"{name} must be between {min_val} and {max_val}, got {float_val}"
            )

        return float_val

    def _log_suspicious_input(self, param_name: str, value: str) -> None:
        """Log suspicious input for security monitoring."""
        if self.audit:
            self.audit.log_event(SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                message=f"Suspicious value for {param_name}",
                severity="WARNING",
                details={
                    "param": param_name,
                    "value_hash": hashlib.sha256(value.encode()).hexdigest()[:16],
                    "value_length": len(value),
                },
            ))


# =============================================================================
# SecureSubprocess Class
# =============================================================================

class SecureSubprocess:
    """Secure wrapper for subprocess execution.

    Provides hardened subprocess execution with:
    - Mandatory list-based commands (no shell=True)
    - Resource limits (timeout, memory)
    - Environment variable isolation
    - Sanitized error output
    - Security event logging

    Example:
        >>> secure = SecureSubprocess(timeout=60, memory_limit_mb=4096)
        >>> result = secure.run(["ffprobe", "-v", "quiet", "-i", "video.mp4"])
        >>> # Raises SecurityError if shell=True is attempted:
        >>> secure.run("ffprobe -v quiet -i video.mp4")  # Raises!
    """

    # Commands that are allowed to be executed
    ALLOWED_COMMANDS: FrozenSet[str] = frozenset({
        'ffmpeg', 'ffprobe', 'yt-dlp', 'youtube-dl',
        'realesrgan-ncnn-vulkan', 'rife-ncnn-vulkan',
        'python', 'python3',
    })

    # Environment variables to preserve (others are filtered)
    SAFE_ENV_VARS: FrozenSet[str] = frozenset({
        'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'TERM',
        'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'VULKAN_SDK',
        'DISPLAY', 'XDG_RUNTIME_DIR', 'TMPDIR', 'TMP', 'TEMP',
    })

    def __init__(
        self,
        timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        allowed_commands: Optional[Set[str]] = None,
        audit_logger: Optional["SecurityAudit"] = None,
        sanitize_env: bool = True,
    ):
        """Initialize SecureSubprocess.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            allowed_commands: Override set of allowed commands
            audit_logger: Optional SecurityAudit instance
            sanitize_env: Whether to filter environment variables
        """
        self.timeout = min(timeout, MAX_SUBPROCESS_TIMEOUT)
        self.memory_limit_mb = memory_limit_mb
        self.allowed_commands = (
            frozenset(allowed_commands) if allowed_commands
            else self.ALLOWED_COMMANDS
        )
        self.audit = audit_logger
        self.sanitize_env = sanitize_env

    def run(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        capture_output: bool = True,
        check: bool = False,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[bytes] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a subprocess command securely.

        Args:
            command: Command as list of arguments (NOT a string!)
            timeout: Override timeout for this call
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise on non-zero exit
            cwd: Working directory for the command
            env: Additional environment variables
            input_data: Data to send to stdin

        Returns:
            CompletedProcess object

        Raises:
            SecurityError: If command is invalid or security check fails
            subprocess.TimeoutExpired: If timeout exceeded
            subprocess.CalledProcessError: If check=True and command fails
        """
        # Validate command format
        self._validate_command_format(command)

        # Validate command is in allowed list
        self._validate_command_allowed(command)

        # Validate arguments don't contain injection patterns
        self._validate_arguments(command)

        # Prepare environment
        exec_env = self._prepare_environment(env)

        # Get timeout
        exec_timeout = min(
            timeout if timeout is not None else self.timeout,
            MAX_SUBPROCESS_TIMEOUT
        )

        # Log subprocess start
        self._log_subprocess_event(
            SecurityEventType.SUBPROCESS_STARTED,
            command,
            f"Starting subprocess with timeout={exec_timeout}s",
        )

        start_time = time.time()

        try:
            # Set resource limits via preexec_fn on Unix
            preexec = self._create_preexec_fn() if os.name != 'nt' else None

            result = subprocess.run(
                command,
                timeout=exec_timeout,
                capture_output=capture_output,
                check=check,
                cwd=cwd,
                env=exec_env,
                input=input_data,
                shell=False,  # NEVER use shell=True
                preexec_fn=preexec,
            )

            elapsed = time.time() - start_time

            self._log_subprocess_event(
                SecurityEventType.SUBPROCESS_COMPLETED,
                command,
                f"Subprocess completed in {elapsed:.2f}s with code {result.returncode}",
            )

            return result

        except subprocess.TimeoutExpired as e:
            self._log_subprocess_event(
                SecurityEventType.SUBPROCESS_TIMEOUT,
                command,
                f"Subprocess timed out after {exec_timeout}s",
                severity="WARNING",
            )
            raise

        except subprocess.CalledProcessError as e:
            # Sanitize error output before logging
            sanitized_stderr = self._sanitize_output(e.stderr) if e.stderr else ""
            self._log_subprocess_event(
                SecurityEventType.SUBPROCESS_FAILED,
                command,
                f"Subprocess failed with code {e.returncode}",
                severity="WARNING",
                details={"stderr_preview": sanitized_stderr[:200]},
            )
            raise

    def run_ffmpeg(
        self,
        args: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Execute FFmpeg with security hardening.

        Args:
            args: FFmpeg arguments (without 'ffmpeg' prefix)
            timeout: Optional timeout override
            **kwargs: Additional arguments for run()

        Returns:
            CompletedProcess object
        """
        # Validate FFmpeg-specific arguments
        self._validate_ffmpeg_args(args)

        command = ['ffmpeg'] + args
        return self.run(command, timeout=timeout, **kwargs)

    def run_ffprobe(
        self,
        args: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Execute ffprobe with security hardening.

        Args:
            args: ffprobe arguments (without 'ffprobe' prefix)
            timeout: Optional timeout override
            **kwargs: Additional arguments for run()

        Returns:
            CompletedProcess object
        """
        command = ['ffprobe'] + args
        return self.run(command, timeout=timeout or 60, **kwargs)

    def run_esrgan(
        self,
        input_path: Path,
        output_path: Path,
        model: str,
        scale: int = 4,
        tile_size: int = 0,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        """Execute Real-ESRGAN with security hardening.

        Args:
            input_path: Input image path
            output_path: Output image path
            model: Model name
            scale: Scale factor
            tile_size: Tile size (0 for auto)
            timeout: Optional timeout override

        Returns:
            CompletedProcess object
        """
        # Validate inputs using sanitizer
        sanitizer = InputSanitizer(audit_logger=self.audit)
        model = sanitizer.validate_esrgan_model(model)
        scale = sanitizer.validate_scale_factor(scale)
        tile_size = sanitizer.validate_tile_size(tile_size)

        command = [
            'realesrgan-ncnn-vulkan',
            '-i', str(input_path),
            '-o', str(output_path),
            '-n', model,
            '-s', str(scale),
            '-f', 'png',
        ]

        if tile_size > 0:
            command.extend(['-t', str(tile_size)])

        return self.run(command, timeout=timeout or 300)

    def _validate_command_format(self, command: Any) -> None:
        """Ensure command is a list, not a string."""
        if isinstance(command, str):
            self._log_security_violation(
                "Command passed as string instead of list",
                {"command_preview": command[:50]},
            )
            raise CommandInjectionError(
                "Command must be a list of arguments. "
                "Example: ['ffmpeg', '-i', 'input.mp4'] instead of 'ffmpeg -i input.mp4'"
            )

        if not isinstance(command, (list, tuple)):
            raise SecurityError("Command must be a list")

        if not command:
            raise SecurityError("Command list cannot be empty")

    def _validate_command_allowed(self, command: List[str]) -> None:
        """Check that the command executable is in the allowed list."""
        executable = Path(command[0]).name.lower()

        # Remove common extensions
        for ext in ('.exe', '.bat', '.cmd', '.sh'):
            if executable.endswith(ext):
                executable = executable[:-len(ext)]

        if executable not in self.allowed_commands:
            self._log_security_violation(
                f"Attempted to run disallowed command: {executable}",
                {"command": executable},
            )
            raise SecurityError(f"Command '{executable}' is not allowed")

    def _validate_arguments(self, command: List[str]) -> None:
        """Validate command arguments for injection patterns."""
        dangerous_patterns = [
            r'[;&|`$]',           # Shell metacharacters
            r'\$\(',              # Command substitution
            r'`',                 # Backtick substitution
            r'\|\|',              # OR operator
            r'&&',                # AND operator
            r'>\s*/',             # Redirect to root
            r'<\(',               # Process substitution
            r'\x00',              # Null byte
        ]

        for i, arg in enumerate(command[1:], 1):  # Skip command name
            arg_str = str(arg)

            for pattern in dangerous_patterns:
                if re.search(pattern, arg_str):
                    self._log_security_violation(
                        f"Dangerous pattern in argument {i}",
                        {"pattern": pattern},
                    )
                    raise CommandInjectionError("Command argument contains forbidden pattern")

    def _validate_ffmpeg_args(self, args: List[str]) -> None:
        """Additional validation for FFmpeg-specific arguments."""
        # Check for dangerous FFmpeg options
        dangerous_options = {
            '-filter_complex',  # Can be dangerous with complex expressions
            '-lavfi',           # Complex filter graph
            '-protocol_whitelist',  # Can enable file:// etc
        }

        for arg in args:
            if arg in dangerous_options:
                logger.warning(f"Using potentially dangerous FFmpeg option: {arg}")

    def _prepare_environment(self, additional_env: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Prepare sanitized environment variables."""
        if not self.sanitize_env:
            result = os.environ.copy()
            if additional_env:
                result.update(additional_env)
            return result

        # Start with only safe environment variables
        result = {}
        for key in self.SAFE_ENV_VARS:
            if key in os.environ:
                result[key] = os.environ[key]

        # Add any additional env vars (after validation)
        if additional_env:
            for key, value in additional_env.items():
                # Validate key name
                if not re.match(r'^[A-Z][A-Z0-9_]*$', key):
                    logger.warning(f"Skipping invalid env var name: {key}")
                    continue
                result[key] = value

        return result

    def _create_preexec_fn(self) -> Callable[[], None]:
        """Create preexec function to set resource limits on Unix."""
        memory_bytes = self.memory_limit_mb * 1024 * 1024

        def set_limits():
            # Set memory limit
            try:
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (memory_bytes, memory_bytes)
                )
            except (ValueError, resource.error):
                pass  # May fail in some environments

            # Set CPU time limit (slightly longer than timeout)
            try:
                cpu_limit = int(MAX_SUBPROCESS_TIMEOUT) + 60
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (cpu_limit, cpu_limit)
                )
            except (ValueError, resource.error):
                pass

        return set_limits

    def _sanitize_output(self, output: Union[str, bytes]) -> str:
        """Sanitize subprocess output for safe logging."""
        if isinstance(output, bytes):
            try:
                output = output.decode('utf-8', errors='replace')
            except Exception:
                return "[binary output]"

        # Remove potential secrets/paths
        output = re.sub(r'/home/[^/\s]+', '/home/[REDACTED]', output)
        output = re.sub(r'password[=:]\S+', 'password=[REDACTED]', output, flags=re.I)
        output = re.sub(r'api[_-]?key[=:]\S+', 'api_key=[REDACTED]', output, flags=re.I)

        # Limit length
        if len(output) > 1000:
            output = output[:1000] + "...[truncated]"

        return output

    def _log_subprocess_event(
        self,
        event_type: SecurityEventType,
        command: List[str],
        message: str,
        severity: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log subprocess event."""
        event_details = details or {}
        event_details["command"] = command[0]  # Only log command name
        event_details["arg_count"] = len(command) - 1

        if self.audit:
            self.audit.log_event(SecurityEvent(
                event_type=event_type,
                message=message,
                severity=severity,
                details=event_details,
            ))
        else:
            log_func = getattr(logger, severity.lower(), logger.info)
            log_func(f"Subprocess: {message}")

    def _log_security_violation(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log security violation."""
        if self.audit:
            self.audit.log_event(SecurityEvent(
                event_type=SecurityEventType.COMMAND_INJECTION_ATTEMPT,
                message=message,
                severity="ERROR",
                details=details or {},
            ))
        else:
            logger.error(f"Security violation: {message}")


# =============================================================================
# SecurityAudit Class
# =============================================================================

class SecurityAudit:
    """Security event logging and monitoring.

    Provides:
    - Security event logging
    - File access pattern tracking
    - Rate limiting for API usage
    - Audit trail for compliance

    Example:
        >>> audit = SecurityAudit(log_file=Path("/var/log/framewright/security.log"))
        >>> audit.log_event(SecurityEvent(
        ...     event_type=SecurityEventType.FILE_ACCESS,
        ...     message="Video file accessed",
        ... ))
        >>> # Rate limiting
        >>> if audit.check_rate_limit("api_calls", limit=100, window_seconds=60):
        ...     process_request()
        ... else:
        ...     raise RateLimitError("Too many requests")
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        max_events: int = 10000,
        enable_rate_limiting: bool = True,
    ):
        """Initialize SecurityAudit.

        Args:
            log_file: Optional file path for persistent logging
            max_events: Maximum events to keep in memory
            enable_rate_limiting: Whether to enable rate limiting
        """
        self.log_file = log_file
        self.max_events = max_events
        self.enable_rate_limiting = enable_rate_limiting

        self._events: List[SecurityEvent] = []
        self._events_lock = threading.Lock()

        # Rate limiting state: {key: [(timestamp, count), ...]}
        self._rate_limits: Dict[str, List[Tuple[float, int]]] = {}
        self._rate_limits_lock = threading.Lock()

        # File access tracking
        self._file_access_counts: Dict[str, int] = {}
        self._file_access_lock = threading.Lock()

        # Setup file handler if log_file provided
        if log_file:
            self._setup_file_logging(log_file)

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event.

        Args:
            event: SecurityEvent to log
        """
        with self._events_lock:
            self._events.append(event)

            # Trim old events if over limit
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

        # Log to standard logger
        log_func = getattr(logger, event.severity.lower(), logger.info)
        log_func(f"[{event.event_type.value}] {event.message}")

        # Log to file if configured
        if self.log_file:
            self._write_to_file(event)

        # Track file access
        if event.event_type == SecurityEventType.FILE_ACCESS:
            self._track_file_access(event)

    def get_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Retrieve logged security events.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            since: Only events after this time
            limit: Maximum events to return

        Returns:
            List of matching SecurityEvent objects
        """
        with self._events_lock:
            events = list(self._events)

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Return most recent events
        return events[-limit:]

    def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: float,
    ) -> bool:
        """Check if an operation is within rate limits.

        Args:
            key: Rate limit key (e.g., "api_calls", "file_reads")
            limit: Maximum operations allowed in window
            window_seconds: Time window in seconds

        Returns:
            True if within limits, False if rate limited
        """
        if not self.enable_rate_limiting:
            return True

        now = time.time()
        window_start = now - window_seconds

        with self._rate_limits_lock:
            # Get or create rate limit entry
            if key not in self._rate_limits:
                self._rate_limits[key] = []

            entries = self._rate_limits[key]

            # Remove old entries
            entries = [(ts, cnt) for ts, cnt in entries if ts >= window_start]

            # Count operations in window
            total = sum(cnt for _, cnt in entries)

            if total >= limit:
                self.log_event(SecurityEvent(
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    message=f"Rate limit exceeded for {key}",
                    severity="WARNING",
                    details={"key": key, "limit": limit, "window": window_seconds},
                ))
                return False

            # Add new entry
            entries.append((now, 1))
            self._rate_limits[key] = entries

            return True

    def increment_rate_limit(self, key: str, count: int = 1) -> None:
        """Increment rate limit counter.

        Args:
            key: Rate limit key
            count: Amount to increment
        """
        now = time.time()

        with self._rate_limits_lock:
            if key not in self._rate_limits:
                self._rate_limits[key] = []
            self._rate_limits[key].append((now, count))

    def get_file_access_stats(self) -> Dict[str, int]:
        """Get file access statistics.

        Returns:
            Dictionary mapping file paths to access counts
        """
        with self._file_access_lock:
            return dict(self._file_access_counts)

    def get_suspicious_patterns(self) -> List[SecurityEvent]:
        """Get events that may indicate security issues.

        Returns:
            List of suspicious SecurityEvent objects
        """
        suspicious_types = {
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            SecurityEventType.COMMAND_INJECTION_ATTEMPT,
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventType.SUSPICIOUS_PATTERN,
        }

        return self.get_events(limit=1000)

    def generate_report(self) -> Dict[str, Any]:
        """Generate a security audit report.

        Returns:
            Dictionary containing security statistics and events
        """
        with self._events_lock:
            events = list(self._events)

        # Count by type
        type_counts = {}
        for event in events:
            type_name = event.event_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Count by severity
        severity_counts = {}
        for event in events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1

        # Get recent critical/error events
        critical_events = [
            e.to_dict() for e in events
            if e.severity in ("ERROR", "CRITICAL")
        ][-20:]

        return {
            "total_events": len(events),
            "events_by_type": type_counts,
            "events_by_severity": severity_counts,
            "recent_critical_events": critical_events,
            "file_access_stats": self.get_file_access_stats(),
            "report_generated": datetime.now().isoformat(),
        }

    def _setup_file_logging(self, log_file: Path) -> None:
        """Setup file-based logging."""
        log_file.parent.mkdir(parents=True, exist_ok=True)

        self._file_handler = logging.FileHandler(log_file)
        self._file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )

    def _write_to_file(self, event: SecurityEvent) -> None:
        """Write event to log file."""
        try:
            import json
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write security log: {e}")

    def _track_file_access(self, event: SecurityEvent) -> None:
        """Track file access patterns."""
        path = event.details.get("path", event.message)

        with self._file_access_lock:
            self._file_access_counts[path] = self._file_access_counts.get(path, 0) + 1


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================

def validate_url(url: str, allow_any_host: bool = False) -> bool:
    """Validate a URL for security.

    Args:
        url: URL to validate
        allow_any_host: If True, allow any host

    Returns:
        True if URL is valid and safe

    Raises:
        InputValidationError: If URL is invalid or potentially dangerous
    """
    if not url or not isinstance(url, str):
        raise InputValidationError("URL must be a non-empty string")

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise InputValidationError(f"Invalid URL format: {e}")

    # Check scheme
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise InputValidationError(
            f"URL scheme '{parsed.scheme}' not allowed. "
            f"Use: {', '.join(ALLOWED_URL_SCHEMES)}"
        )

    # Check for empty host
    if not parsed.netloc:
        raise InputValidationError("URL must have a valid host")

    # Check for suspicious patterns
    suspicious_patterns = [
        r'javascript:', r'data:', r'file:',
        r'//localhost', r'//127\.', r'//0\.', r'//\[',
    ]

    url_lower = url.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, url_lower):
            raise InputValidationError("Suspicious URL pattern detected")

    return True


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = False,
    allowed_extensions: Optional[Set[str]] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Validate a file path for security.

    Args:
        path: Path to validate
        must_exist: If True, path must exist
        allowed_extensions: Optional set of allowed file extensions
        base_dir: If provided, path must be under this directory

    Returns:
        Validated Path object

    Raises:
        InputValidationError: If path is invalid
        PathTraversalError: If path traversal is detected
    """
    validator = PathValidator(
        base_dir=base_dir,
        allowed_extensions=allowed_extensions,
    )
    return validator.validate(path, must_exist=must_exist)


def validate_video_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate a video file path."""
    validator = PathValidator(allowed_extensions=ALLOWED_VIDEO_EXTENSIONS)
    return validator.validate(path, must_exist=must_exist)


def validate_frame_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate a frame/image file path."""
    validator = PathValidator(allowed_extensions=ALLOWED_FRAME_EXTENSIONS)
    return validator.validate(path, must_exist=must_exist)


def sanitize_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """Sanitize a filename to remove dangerous characters."""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_filename(filename, max_length=max_length)


def safe_subprocess_run(
    command: List[str],
    timeout: Optional[float] = None,
    capture_output: bool = True,
    check: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Safely run a subprocess command.

    Args:
        command: Command as list of arguments (NOT a string!)
        timeout: Optional timeout in seconds
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit
        **kwargs: Additional arguments

    Returns:
        CompletedProcess object
    """
    secure = SecureSubprocess(
        timeout=timeout or DEFAULT_SUBPROCESS_TIMEOUT,
    )
    return secure.run(
        command,
        timeout=timeout,
        capture_output=capture_output,
        check=check,
        **kwargs,
    )


def quote_path(path: Union[str, Path]) -> str:
    """Quote a path for safe use in shell commands.

    Note: Prefer using list-based commands instead of quoted strings.
    """
    return shlex.quote(str(path))


def validate_integer_range(
    value: int,
    min_val: int,
    max_val: int,
    name: str = "value",
) -> int:
    """Validate that an integer is within an allowed range."""
    sanitizer = InputSanitizer()
    return sanitizer.validate_integer_range(value, min_val, max_val, name)


def validate_float_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
) -> float:
    """Validate that a float is within an allowed range."""
    sanitizer = InputSanitizer()
    return sanitizer.validate_float_range(value, min_val, max_val, name)
