"""Error recovery system for resilient processing."""

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Strategies for error recovery."""
    RETRY = auto()  # Retry the same operation
    SKIP = auto()  # Skip the failing item
    FALLBACK = auto()  # Use fallback method
    REDUCE_BATCH = auto()  # Reduce batch size
    CLEAR_CACHE = auto()  # Clear model cache
    RESTART_MODEL = auto()  # Reload the model
    ABORT = auto()  # Abort processing


class RecoveryAction(Enum):
    """Actions taken during recovery."""
    SUCCEEDED = auto()
    FAILED = auto()
    SKIPPED = auto()
    ABORTED = auto()


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    error_type: str
    error_message: str
    traceback_str: str
    timestamp: datetime
    frame_number: Optional[int] = None
    stage: str = ""
    batch_size: int = 1
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    action: RecoveryAction
    strategy_used: RecoveryStrategy
    retry_count: int
    error_context: ErrorContext
    recovered_result: Any = None
    notes: str = ""


class ErrorRecoveryManager:
    """Manages error recovery for frame processing."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_consecutive_failures: int = 10,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_consecutive_failures = max_consecutive_failures

        # Recovery handlers
        self._handlers: Dict[Type[Exception], RecoveryStrategy] = {}
        self._fallbacks: Dict[str, Callable] = {}

        # State
        self._consecutive_failures = 0
        self._error_history: List[ErrorContext] = []
        self._recovery_stats = {
            "total_errors": 0,
            "recovered": 0,
            "skipped": 0,
            "aborted": 0,
        }

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default error handlers."""
        # CUDA/GPU errors - usually need model restart
        try:
            import torch
            self._handlers[torch.cuda.OutOfMemoryError] = RecoveryStrategy.REDUCE_BATCH
        except ImportError:
            pass

        # Memory errors
        self._handlers[MemoryError] = RecoveryStrategy.REDUCE_BATCH

        # File errors - skip the frame
        self._handlers[FileNotFoundError] = RecoveryStrategy.SKIP
        self._handlers[PermissionError] = RecoveryStrategy.SKIP

        # Network errors - retry
        self._handlers[ConnectionError] = RecoveryStrategy.RETRY
        self._handlers[TimeoutError] = RecoveryStrategy.RETRY

        # Value errors - usually data issue, skip
        self._handlers[ValueError] = RecoveryStrategy.SKIP

        # Runtime errors - try fallback then skip
        self._handlers[RuntimeError] = RecoveryStrategy.FALLBACK

    def register_handler(
        self,
        error_type: Type[Exception],
        strategy: RecoveryStrategy,
    ) -> None:
        """Register a recovery strategy for an error type."""
        self._handlers[error_type] = strategy

    def register_fallback(
        self,
        stage: str,
        fallback_fn: Callable,
    ) -> None:
        """Register a fallback function for a processing stage."""
        self._fallbacks[stage] = fallback_fn

    def create_context(
        self,
        error: Exception,
        frame_number: Optional[int] = None,
        stage: str = "",
        batch_size: int = 1,
        retry_count: int = 0,
    ) -> ErrorContext:
        """Create error context from an exception."""
        return ErrorContext(
            error=error,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            timestamp=datetime.now(),
            frame_number=frame_number,
            stage=stage,
            batch_size=batch_size,
            retry_count=retry_count,
        )

    def get_strategy(self, error: Exception) -> RecoveryStrategy:
        """Get recovery strategy for an error."""
        # Check specific type first
        for error_type, strategy in self._handlers.items():
            if isinstance(error, error_type):
                return strategy

        # Default strategy based on error characteristics
        error_msg = str(error).lower()

        if "cuda" in error_msg or "gpu" in error_msg or "memory" in error_msg:
            return RecoveryStrategy.REDUCE_BATCH

        if "file" in error_msg or "path" in error_msg:
            return RecoveryStrategy.SKIP

        if "timeout" in error_msg or "connection" in error_msg:
            return RecoveryStrategy.RETRY

        # Default: retry then skip
        return RecoveryStrategy.RETRY

    def attempt_recovery(
        self,
        context: ErrorContext,
        operation: Callable[[], Any],
        fallback: Optional[Callable[[], Any]] = None,
    ) -> RecoveryResult:
        """Attempt to recover from an error.

        Args:
            context: Error context
            operation: The operation that failed
            fallback: Optional fallback operation

        Returns:
            Recovery result
        """
        self._recovery_stats["total_errors"] += 1
        self._error_history.append(context)
        self._consecutive_failures += 1

        # Check for too many consecutive failures
        if self._consecutive_failures > self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures ({self._consecutive_failures}), aborting")
            self._recovery_stats["aborted"] += 1
            return RecoveryResult(
                action=RecoveryAction.ABORTED,
                strategy_used=RecoveryStrategy.ABORT,
                retry_count=context.retry_count,
                error_context=context,
                notes="Exceeded max consecutive failures",
            )

        strategy = self.get_strategy(context.error)
        logger.info(f"Recovery strategy for {context.error_type}: {strategy.name}")

        result = None

        if strategy == RecoveryStrategy.RETRY:
            result = self._handle_retry(context, operation)

        elif strategy == RecoveryStrategy.SKIP:
            result = self._handle_skip(context)

        elif strategy == RecoveryStrategy.FALLBACK:
            fb = fallback or self._fallbacks.get(context.stage)
            result = self._handle_fallback(context, operation, fb)

        elif strategy == RecoveryStrategy.REDUCE_BATCH:
            result = self._handle_reduce_batch(context, operation)

        elif strategy == RecoveryStrategy.CLEAR_CACHE:
            result = self._handle_clear_cache(context, operation)

        elif strategy == RecoveryStrategy.RESTART_MODEL:
            result = self._handle_restart_model(context, operation)

        else:
            result = self._handle_skip(context)

        # Update stats
        if result.action == RecoveryAction.SUCCEEDED:
            self._recovery_stats["recovered"] += 1
            self._consecutive_failures = 0
        elif result.action == RecoveryAction.SKIPPED:
            self._recovery_stats["skipped"] += 1
        elif result.action == RecoveryAction.ABORTED:
            self._recovery_stats["aborted"] += 1

        return result

    def _handle_retry(
        self,
        context: ErrorContext,
        operation: Callable,
    ) -> RecoveryResult:
        """Handle retry strategy."""
        for attempt in range(self.max_retries):
            context.retry_count = attempt + 1
            logger.info(f"Retry attempt {context.retry_count}/{self.max_retries}")

            time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

            try:
                result = operation()
                return RecoveryResult(
                    action=RecoveryAction.SUCCEEDED,
                    strategy_used=RecoveryStrategy.RETRY,
                    retry_count=context.retry_count,
                    error_context=context,
                    recovered_result=result,
                    notes=f"Succeeded on retry {context.retry_count}",
                )
            except Exception as e:
                logger.warning(f"Retry {context.retry_count} failed: {e}")
                context.error = e

        # All retries failed, skip
        return self._handle_skip(context)

    def _handle_skip(self, context: ErrorContext) -> RecoveryResult:
        """Handle skip strategy."""
        logger.warning(f"Skipping frame {context.frame_number} due to {context.error_type}")

        return RecoveryResult(
            action=RecoveryAction.SKIPPED,
            strategy_used=RecoveryStrategy.SKIP,
            retry_count=context.retry_count,
            error_context=context,
            notes=f"Skipped due to unrecoverable {context.error_type}",
        )

    def _handle_fallback(
        self,
        context: ErrorContext,
        operation: Callable,
        fallback: Optional[Callable],
    ) -> RecoveryResult:
        """Handle fallback strategy."""
        if fallback is None:
            logger.warning(f"No fallback for stage {context.stage}, skipping")
            return self._handle_skip(context)

        try:
            result = fallback()
            return RecoveryResult(
                action=RecoveryAction.SUCCEEDED,
                strategy_used=RecoveryStrategy.FALLBACK,
                retry_count=context.retry_count,
                error_context=context,
                recovered_result=result,
                notes="Used fallback method",
            )
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            return self._handle_skip(context)

    def _handle_reduce_batch(
        self,
        context: ErrorContext,
        operation: Callable,
    ) -> RecoveryResult:
        """Handle reduce batch strategy."""
        # This would need to communicate batch size reduction
        # For now, clear GPU cache and retry
        self._clear_gpu_memory()

        try:
            result = operation()
            return RecoveryResult(
                action=RecoveryAction.SUCCEEDED,
                strategy_used=RecoveryStrategy.REDUCE_BATCH,
                retry_count=context.retry_count,
                error_context=context,
                recovered_result=result,
                notes="Succeeded after clearing GPU memory",
            )
        except Exception as e:
            logger.error(f"Still failing after memory clear: {e}")
            return self._handle_skip(context)

    def _handle_clear_cache(
        self,
        context: ErrorContext,
        operation: Callable,
    ) -> RecoveryResult:
        """Handle clear cache strategy."""
        self._clear_gpu_memory()
        return self._handle_retry(context, operation)

    def _handle_restart_model(
        self,
        context: ErrorContext,
        operation: Callable,
    ) -> RecoveryResult:
        """Handle model restart strategy."""
        # This would need model manager integration
        self._clear_gpu_memory()
        time.sleep(2.0)  # Allow GPU to settle
        return self._handle_retry(context, operation)

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared GPU memory cache")
        except ImportError:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            **self._recovery_stats,
            "error_history_count": len(self._error_history),
            "consecutive_failures": self._consecutive_failures,
        }

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type."""
        summary: Dict[str, int] = {}
        for ctx in self._error_history:
            error_type = ctx.error_type
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary

    def reset(self) -> None:
        """Reset error tracking state."""
        self._consecutive_failures = 0
        self._error_history.clear()
        self._recovery_stats = {
            "total_errors": 0,
            "recovered": 0,
            "skipped": 0,
            "aborted": 0,
        }


def with_recovery(
    recovery_manager: ErrorRecoveryManager,
    stage: str = "",
    frame_number: Optional[int] = None,
):
    """Decorator for automatic error recovery.

    Usage:
        @with_recovery(manager, stage="upscale")
        def process_frame(frame):
            return upscaler(frame)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = recovery_manager.create_context(
                    error=e,
                    frame_number=frame_number,
                    stage=stage,
                )
                result = recovery_manager.attempt_recovery(
                    context=context,
                    operation=lambda: func(*args, **kwargs),
                )
                if result.action == RecoveryAction.SUCCEEDED:
                    return result.recovered_result
                elif result.action == RecoveryAction.ABORTED:
                    raise RuntimeError(f"Processing aborted: {e}")
                else:
                    return None  # Skipped

        return wrapper
    return decorator
