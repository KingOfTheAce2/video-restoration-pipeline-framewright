"""Enhanced error handling module for FrameWright pipeline.

Provides error classification, retry logic, and detailed error context.
"""
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Error Classification
# =============================================================================

class VideoRestorerError(Exception):
    """Base exception for all VideoRestorer errors."""

    def __init__(self, message: str, context: Optional["ErrorContext"] = None):
        super().__init__(message)
        self.context = context


class TransientError(VideoRestorerError):
    """Recoverable errors that may succeed on retry.

    These errors are typically caused by temporary conditions like
    resource exhaustion, network issues, or race conditions.
    """
    pass


class ResourceError(TransientError):
    """Resource exhaustion errors (VRAM, disk space, memory)."""
    pass


class VRAMError(ResourceError):
    """GPU VRAM exhaustion error."""
    pass


class DiskSpaceError(ResourceError):
    """Insufficient disk space error."""
    pass


class NetworkError(TransientError):
    """Network-related failures."""
    pass


class TimeoutError(TransientError):
    """Operation timeout error."""
    pass


class FatalError(VideoRestorerError):
    """Non-recoverable errors requiring intervention.

    These errors indicate fundamental problems that cannot be
    resolved by retrying.
    """
    pass


class CorruptionError(FatalError):
    """Data corruption requiring manual intervention."""
    pass


class DependencyError(FatalError):
    """Missing or incompatible dependency."""
    pass


class ConfigurationError(FatalError):
    """Invalid configuration."""
    pass


class ValidationError(FatalError):
    """Validation failure for input or output."""
    pass


# Stage-specific errors (can be transient or fatal)
class DownloadError(VideoRestorerError):
    """Error during video download."""
    pass


class MetadataError(VideoRestorerError):
    """Error during metadata extraction."""
    pass


class AudioExtractionError(VideoRestorerError):
    """Error during audio extraction."""
    pass


class FrameExtractionError(VideoRestorerError):
    """Error during frame extraction."""
    pass


class EnhancementError(VideoRestorerError):
    """Error during frame enhancement."""
    pass


class ReassemblyError(VideoRestorerError):
    """Error during video reassembly."""
    pass


class InterpolationError(VideoRestorerError):
    """Error during frame interpolation."""
    pass


# =============================================================================
# Error Context
# =============================================================================

@dataclass
class ErrorContext:
    """Detailed context for debugging errors.

    Captures comprehensive information about the system state
    and operation that caused the error.
    """
    stage: str
    operation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    frame_number: Optional[int] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    command: Optional[List[str]] = None
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    return_code: Optional[int] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "stage": self.stage,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "command": self.command,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "return_code": self.return_code,
            "system_state": self.system_state,
            "additional_info": self.additional_info,
        }

    def __str__(self) -> str:
        """Human-readable error context."""
        lines = [
            f"Stage: {self.stage}",
            f"Operation: {self.operation}",
            f"Timestamp: {self.timestamp}",
        ]

        if self.frame_number is not None:
            lines.append(f"Frame: {self.frame_number}")
        if self.input_file:
            lines.append(f"Input: {self.input_file}")
        if self.output_file:
            lines.append(f"Output: {self.output_file}")
        if self.command:
            lines.append(f"Command: {' '.join(self.command)}")
        if self.return_code is not None:
            lines.append(f"Return code: {self.return_code}")
        if self.stderr:
            lines.append(f"Stderr: {self.stderr[:500]}")
        if self.system_state:
            lines.append(f"System state: {self.system_state}")

        return "\n".join(lines)


def create_error_context(
    stage: str,
    operation: str,
    frame_number: Optional[int] = None,
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    command: Optional[List[str]] = None,
    stderr: Optional[str] = None,
    stdout: Optional[str] = None,
    return_code: Optional[int] = None,
    **additional_info: Any
) -> ErrorContext:
    """Create an error context with system state.

    Automatically captures system state like VRAM usage and disk space.
    """
    from .utils.gpu import get_gpu_memory_info
    from .utils.disk import get_disk_usage

    system_state = {}

    # Capture GPU state
    try:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            system_state["gpu"] = gpu_info
    except Exception:
        pass

    # Capture disk state
    try:
        if input_file:
            disk_info = get_disk_usage(Path(input_file).parent)
            system_state["disk"] = disk_info
    except Exception:
        pass

    return ErrorContext(
        stage=stage,
        operation=operation,
        frame_number=frame_number,
        input_file=str(input_file) if input_file else None,
        output_file=str(output_file) if output_file else None,
        command=command,
        stderr=stderr,
        stdout=stdout,
        return_code=return_code,
        system_state=system_state,
        additional_info=additional_info,
    )


# =============================================================================
# Error Classification Logic
# =============================================================================

def classify_error(
    error: Exception,
    stderr: Optional[str] = None
) -> Type[VideoRestorerError]:
    """Classify an error as transient or fatal based on error message.

    Args:
        error: The exception that occurred
        stderr: Optional stderr output for additional context

    Returns:
        The appropriate error class to use
    """
    error_text = str(error).lower()
    stderr_text = (stderr or "").lower()
    combined = f"{error_text} {stderr_text}"

    # VRAM/GPU errors (transient - can retry with smaller tile size)
    vram_indicators = [
        "cuda out of memory",
        "out of memory",
        "vram",
        "gpu memory",
        "memory allocation failed",
        "insufficient memory",
        "vulkan memory",
    ]
    if any(ind in combined for ind in vram_indicators):
        return VRAMError

    # Disk space errors (transient - may free up space)
    disk_indicators = [
        "no space left on device",
        "disk quota exceeded",
        "not enough space",
        "disk full",
        "write error",
    ]
    if any(ind in combined for ind in disk_indicators):
        return DiskSpaceError

    # Network errors (transient)
    network_indicators = [
        "connection refused",
        "connection reset",
        "network unreachable",
        "timeout",
        "timed out",
        "host not found",
        "dns",
        "ssl",
        "certificate",
    ]
    if any(ind in combined for ind in network_indicators):
        return NetworkError

    # Corruption errors (fatal)
    corruption_indicators = [
        "corrupt",
        "invalid data",
        "checksum",
        "crc error",
        "truncated",
        "malformed",
    ]
    if any(ind in combined for ind in corruption_indicators):
        return CorruptionError

    # Dependency errors (fatal)
    dependency_indicators = [
        "not found",
        "command not found",
        "no such file or directory",
        "cannot find",
        "missing",
        "not installed",
    ]
    if any(ind in combined for ind in dependency_indicators):
        return DependencyError

    # Default to base error
    return VideoRestorerError


# =============================================================================
# Retry Logic
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on: tuple = (TransientError,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        import random

        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (TransientError,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on
        on_retry: Optional callback called before each retry

    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        retry_on=retry_on,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)

                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed. "
                            f"Last error: {e}"
                        )
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error: {e}")
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception

            raise RuntimeError("Unexpected state: no exception but all retries failed")

        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations with custom handling."""

    def __init__(
        self,
        operation_name: str,
        max_attempts: int = 3,
        retry_on: tuple = (TransientError,),
        on_vram_error: Optional[Callable[[], None]] = None,
    ):
        """Initialize retryable operation.

        Args:
            operation_name: Name for logging
            max_attempts: Maximum retry attempts
            retry_on: Exception types to retry
            on_vram_error: Callback for VRAM errors (e.g., reduce tile size)
        """
        self.operation_name = operation_name
        self.max_attempts = max_attempts
        self.retry_on = retry_on
        self.on_vram_error = on_vram_error
        self.attempt = 0
        self.config = RetryConfig(max_attempts=max_attempts, retry_on=retry_on)

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        last_exception: Optional[Exception] = None

        for self.attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except VRAMError as e:
                last_exception = e
                if self.on_vram_error and self.attempt < self.max_attempts - 1:
                    logger.info("VRAM error detected, attempting recovery...")
                    self.on_vram_error()
                    time.sleep(self.config.get_delay(self.attempt))
                elif self.attempt >= self.max_attempts - 1:
                    raise
            except self.retry_on as e:
                last_exception = e
                if self.attempt < self.max_attempts - 1:
                    delay = self.config.get_delay(self.attempt)
                    logger.warning(
                        f"{self.operation_name}: Attempt {self.attempt + 1} failed. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry state")


# =============================================================================
# Error Aggregation
# =============================================================================

@dataclass
class ErrorReport:
    """Aggregated error report for batch operations."""
    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_error(
        self,
        operation_id: Union[int, str],
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> None:
        """Add an error to the report."""
        self.failed += 1
        self.errors.append({
            "operation_id": operation_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context.to_dict() if context else None,
        })

    def add_success(self) -> None:
        """Record a successful operation."""
        self.successful += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful / self.total_operations

    def has_failures(self) -> bool:
        """Check if any operations failed."""
        return self.failed > 0

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Operations: {self.successful}/{self.total_operations} successful "
            f"({self.success_rate:.1%}), {self.failed} failed"
        )
