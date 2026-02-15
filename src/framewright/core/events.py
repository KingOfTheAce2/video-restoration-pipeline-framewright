"""Event system for FrameWright video restoration pipeline.

This module provides a thread-safe event bus for pub/sub communication
throughout the processing pipeline. Events enable loose coupling between
components and support both synchronous and asynchronous notification.

Example usage:

    >>> from framewright.core.events import EventBus, EventType, Event
    >>>
    >>> # Create event bus
    >>> bus = EventBus()
    >>>
    >>> # Subscribe to events
    >>> def on_progress(event):
    ...     print(f"Progress: {event.data['progress']}%")
    >>>
    >>> bus.subscribe(EventType.PROGRESS, on_progress)
    >>>
    >>> # Emit events
    >>> bus.emit(Event(EventType.PROGRESS, "processor", {"progress": 50}))
    >>>
    >>> # Use async emission for non-blocking
    >>> bus.emit_async(Event(EventType.COMPLETED, "processor", {"frames": 1000}))
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet, ref

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types
# =============================================================================


class EventType(Enum):
    """Types of events in the processing pipeline.

    Events are categorized by their purpose:
    - Lifecycle events (STARTED, COMPLETED)
    - Progress events (PROGRESS)
    - Status events (ERROR, WARNING)
    - Stage-specific events (STAGE_STARTED, STAGE_COMPLETED, etc.)
    """

    # Lifecycle events
    STARTED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    PAUSED = auto()
    RESUMED = auto()

    # Progress events
    PROGRESS = auto()
    FRAME_PROCESSED = auto()
    BATCH_PROCESSED = auto()

    # Status events
    ERROR = auto()
    WARNING = auto()
    INFO = auto()

    # Stage events
    STAGE_STARTED = auto()
    STAGE_COMPLETED = auto()
    STAGE_FAILED = auto()

    # Resource events
    CHECKPOINT_SAVED = auto()
    CHECKPOINT_LOADED = auto()
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()

    # Quality events
    QUALITY_CHECK_PASSED = auto()
    QUALITY_CHECK_FAILED = auto()

    # Hardware events
    GPU_MEMORY_LOW = auto()
    THERMAL_THROTTLE = auto()

    # Custom event type for extensions
    CUSTOM = auto()


# =============================================================================
# Event Classes
# =============================================================================


@dataclass
class Event:
    """Base event class with metadata.

    Attributes:
        event_type: Type of the event.
        source: Name of the component that emitted the event.
        data: Event-specific data dictionary.
        timestamp: When the event was created (auto-generated).
        event_id: Unique identifier for the event (auto-generated).
        correlation_id: Optional ID to link related events.
        priority: Event priority (higher = more important).
    """

    event_type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: Optional[str] = None
    priority: int = 0

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"Event({self.event_type.name}, source={self.source}, "
            f"data_keys={list(self.data.keys())})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_type": self.event_type.name,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary.

        Args:
            data: Dictionary representation of the event.

        Returns:
            Event instance.
        """
        return cls(
            event_type=EventType[data["event_type"]],
            source=data["source"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())[:8]),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 0),
        )


# =============================================================================
# Specialized Event Classes
# =============================================================================


@dataclass
class ProcessingStartedEvent(Event):
    """Event emitted when processing starts.

    Data fields:
        input_path: Path to input video.
        total_frames: Total number of frames to process.
        config: Configuration summary.
    """

    def __init__(
        self,
        source: str,
        input_path: str,
        total_frames: int,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            event_type=EventType.STARTED,
            source=source,
            data={
                "input_path": input_path,
                "total_frames": total_frames,
                "config": config or {},
            },
            **kwargs,
        )


@dataclass
class FrameProcessedEvent(Event):
    """Event emitted when a frame is processed.

    Data fields:
        frame_number: Number of the processed frame.
        total_frames: Total frames to process.
        processing_time: Time taken to process in seconds.
        stage: Current processing stage.
    """

    def __init__(
        self,
        source: str,
        frame_number: int,
        total_frames: int,
        processing_time: float,
        stage: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            event_type=EventType.FRAME_PROCESSED,
            source=source,
            data={
                "frame_number": frame_number,
                "total_frames": total_frames,
                "processing_time": processing_time,
                "stage": stage,
                "progress": (frame_number / total_frames * 100) if total_frames > 0 else 0,
            },
            **kwargs,
        )


@dataclass
class StageCompletedEvent(Event):
    """Event emitted when a processing stage completes.

    Data fields:
        stage_name: Name of the completed stage.
        duration: Stage duration in seconds.
        frames_processed: Number of frames processed.
        metrics: Stage-specific metrics.
    """

    def __init__(
        self,
        source: str,
        stage_name: str,
        duration: float,
        frames_processed: int,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            event_type=EventType.STAGE_COMPLETED,
            source=source,
            data={
                "stage_name": stage_name,
                "duration": duration,
                "frames_processed": frames_processed,
                "metrics": metrics or {},
            },
            **kwargs,
        )


@dataclass
class ProcessingErrorEvent(Event):
    """Event emitted when an error occurs.

    Data fields:
        error_type: Type of the error.
        error_message: Error message.
        stage: Stage where error occurred.
        frame_number: Frame number if applicable.
        recoverable: Whether the error is recoverable.
        traceback: Error traceback if available.
    """

    def __init__(
        self,
        source: str,
        error_type: str,
        error_message: str,
        stage: Optional[str] = None,
        frame_number: Optional[int] = None,
        recoverable: bool = False,
        traceback: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            event_type=EventType.ERROR,
            source=source,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "stage": stage,
                "frame_number": frame_number,
                "recoverable": recoverable,
                "traceback": traceback,
            },
            priority=100,  # High priority for errors
            **kwargs,
        )


@dataclass
class ProgressEvent(Event):
    """Event emitted for progress updates.

    Data fields:
        progress: Progress percentage (0-100).
        stage: Current processing stage.
        eta_seconds: Estimated time remaining.
        current_fps: Current processing speed.
    """

    def __init__(
        self,
        source: str,
        progress: float,
        stage: Optional[str] = None,
        eta_seconds: Optional[float] = None,
        current_fps: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            event_type=EventType.PROGRESS,
            source=source,
            data={
                "progress": progress,
                "stage": stage,
                "eta_seconds": eta_seconds,
                "current_fps": current_fps,
            },
            **kwargs,
        )


@dataclass
class QualityCheckEvent(Event):
    """Event emitted for quality check results.

    Data fields:
        passed: Whether the check passed.
        check_type: Type of quality check.
        actual_value: Measured value.
        threshold: Threshold value.
        frame_number: Frame number if applicable.
    """

    def __init__(
        self,
        source: str,
        passed: bool,
        check_type: str,
        actual_value: float,
        threshold: float,
        frame_number: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        event_type = EventType.QUALITY_CHECK_PASSED if passed else EventType.QUALITY_CHECK_FAILED
        super().__init__(
            event_type=event_type,
            source=source,
            data={
                "passed": passed,
                "check_type": check_type,
                "actual_value": actual_value,
                "threshold": threshold,
                "frame_number": frame_number,
            },
            **kwargs,
        )


# =============================================================================
# Callback Type
# =============================================================================

EventCallback = Callable[[Event], None]


# =============================================================================
# Event Bus
# =============================================================================


class EventBus:
    """Thread-safe event bus for pub/sub communication.

    The EventBus allows components to publish events and subscribe to
    receive notifications. It supports:
    - Multiple subscribers per event type
    - Wildcard subscriptions (receive all events)
    - Synchronous and asynchronous event emission
    - Event filtering
    - Subscriber priority ordering

    Example:
        >>> bus = EventBus()
        >>>
        >>> def handler(event):
        ...     print(f"Received: {event}")
        >>>
        >>> bus.subscribe(EventType.PROGRESS, handler)
        >>> bus.emit(Event(EventType.PROGRESS, "test", {"progress": 50}))
    """

    def __init__(
        self,
        max_workers: int = 4,
        enable_history: bool = False,
        history_size: int = 1000,
    ) -> None:
        """Initialize the event bus.

        Args:
            max_workers: Maximum threads for async emission.
            enable_history: Whether to keep event history.
            history_size: Maximum events to keep in history.
        """
        self._subscribers: Dict[EventType, List[EventCallback]] = {}
        self._wildcard_subscribers: List[EventCallback] = []
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._enable_history = enable_history
        self._history_size = history_size
        self._history: List[Event] = []

        self._paused = False
        self._pause_lock = threading.Condition(self._lock)

        # Statistics
        self._events_emitted = 0
        self._events_by_type: Dict[EventType, int] = {}

    def subscribe(
        self,
        event_type: Union[EventType, None],
        callback: EventCallback,
    ) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to, or None for all.
            callback: Function to call when event is emitted.
        """
        with self._lock:
            if event_type is None:
                # Wildcard subscription
                if callback not in self._wildcard_subscribers:
                    self._wildcard_subscribers.append(callback)
            else:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                if callback not in self._subscribers[event_type]:
                    self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: Union[EventType, None],
        callback: EventCallback,
    ) -> bool:
        """Unsubscribe from events.

        Args:
            event_type: Type of events to unsubscribe from, or None for wildcard.
            callback: Callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        with self._lock:
            if event_type is None:
                if callback in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(callback)
                    return True
            else:
                if event_type in self._subscribers:
                    if callback in self._subscribers[event_type]:
                        self._subscribers[event_type].remove(callback)
                        return True
            return False

    def emit(self, event: Event) -> None:
        """Emit an event synchronously.

        Calls all subscribers in the current thread. Errors in
        callbacks are logged but don't stop other callbacks.

        Args:
            event: Event to emit.
        """
        self._wait_if_paused()

        with self._lock:
            # Update statistics
            self._events_emitted += 1
            self._events_by_type[event.event_type] = (
                self._events_by_type.get(event.event_type, 0) + 1
            )

            # Add to history
            if self._enable_history:
                self._history.append(event)
                if len(self._history) > self._history_size:
                    self._history = self._history[-self._history_size:]

            # Get subscribers (copy to avoid modification during iteration)
            subscribers = list(self._subscribers.get(event.event_type, []))
            wildcards = list(self._wildcard_subscribers)

        # Call subscribers outside lock
        for callback in subscribers:
            self._safe_call(callback, event)

        for callback in wildcards:
            self._safe_call(callback, event)

    def emit_async(self, event: Event) -> None:
        """Emit an event asynchronously.

        Queues the event for emission in a background thread.
        Returns immediately without waiting for callbacks.

        Args:
            event: Event to emit.
        """
        self._executor.submit(self.emit, event)

    async def emit_awaitable(self, event: Event) -> None:
        """Emit an event in an async context.

        This method can be awaited in async code.

        Args:
            event: Event to emit.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.emit, event)

    def _safe_call(self, callback: EventCallback, event: Event) -> None:
        """Call a callback safely, catching and logging errors.

        Args:
            callback: Callback function to call.
            event: Event to pass to callback.
        """
        try:
            callback(event)
        except Exception as e:
            logger.error(
                f"Error in event callback for {event.event_type.name}: {e}",
                exc_info=True,
            )

    def _wait_if_paused(self) -> None:
        """Wait if the bus is paused."""
        with self._pause_lock:
            while self._paused:
                self._pause_lock.wait()

    def pause(self) -> None:
        """Pause event emission.

        Events emitted while paused will block until resumed.
        """
        with self._pause_lock:
            self._paused = True

    def resume(self) -> None:
        """Resume event emission after pause."""
        with self._pause_lock:
            self._paused = False
            self._pause_lock.notify_all()

    @property
    def is_paused(self) -> bool:
        """Check if the bus is paused."""
        return self._paused

    def clear_subscribers(
        self,
        event_type: Optional[EventType] = None,
    ) -> None:
        """Clear all subscribers.

        Args:
            event_type: Type to clear, or None to clear all.
        """
        with self._lock:
            if event_type is None:
                self._subscribers.clear()
                self._wildcard_subscribers.clear()
            elif event_type in self._subscribers:
                self._subscribers[event_type].clear()

    def get_subscriber_count(
        self,
        event_type: Optional[EventType] = None,
    ) -> int:
        """Get count of subscribers.

        Args:
            event_type: Type to count, or None for total.

        Returns:
            Number of subscribers.
        """
        with self._lock:
            if event_type is None:
                count = len(self._wildcard_subscribers)
                for subscribers in self._subscribers.values():
                    count += len(subscribers)
                return count
            else:
                return len(self._subscribers.get(event_type, []))

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Get event history.

        Args:
            event_type: Filter by event type, or None for all.
            limit: Maximum events to return.

        Returns:
            List of past events.
        """
        if not self._enable_history:
            return []

        with self._lock:
            history = list(self._history)

        if event_type is not None:
            history = [e for e in history if e.event_type == event_type]

        if limit is not None:
            history = history[-limit:]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get emission statistics.

        Returns:
            Dictionary with event statistics.
        """
        with self._lock:
            return {
                "total_events_emitted": self._events_emitted,
                "events_by_type": {
                    t.name: c for t, c in self._events_by_type.items()
                },
                "subscriber_count": self.get_subscriber_count(),
                "history_size": len(self._history) if self._enable_history else 0,
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the event bus.

        Args:
            wait: Whether to wait for pending async events.
        """
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "EventBus":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - shutdown bus."""
        self.shutdown()


# =============================================================================
# Event Filtering
# =============================================================================


class EventFilter:
    """Filter for selecting specific events.

    Filters can be combined with & (and) and | (or) operators.

    Example:
        >>> filter = EventFilter.by_type(EventType.PROGRESS)
        >>> filter = filter & EventFilter.by_source("upscaler")
        >>>
        >>> if filter.matches(event):
        ...     print("Event matches!")
    """

    def __init__(
        self,
        predicate: Callable[[Event], bool],
        description: str = "custom",
    ) -> None:
        """Initialize filter with a predicate function.

        Args:
            predicate: Function that returns True for matching events.
            description: Human-readable description of the filter.
        """
        self._predicate = predicate
        self._description = description

    def matches(self, event: Event) -> bool:
        """Check if an event matches this filter.

        Args:
            event: Event to check.

        Returns:
            True if the event matches.
        """
        return self._predicate(event)

    def __and__(self, other: "EventFilter") -> "EventFilter":
        """Combine filters with AND."""
        return EventFilter(
            lambda e: self.matches(e) and other.matches(e),
            f"({self._description} AND {other._description})",
        )

    def __or__(self, other: "EventFilter") -> "EventFilter":
        """Combine filters with OR."""
        return EventFilter(
            lambda e: self.matches(e) or other.matches(e),
            f"({self._description} OR {other._description})",
        )

    def __invert__(self) -> "EventFilter":
        """Negate filter."""
        return EventFilter(
            lambda e: not self.matches(e),
            f"NOT({self._description})",
        )

    def __str__(self) -> str:
        """Return description."""
        return self._description

    @classmethod
    def by_type(cls, event_type: EventType) -> "EventFilter":
        """Create filter for a specific event type.

        Args:
            event_type: Type to filter for.

        Returns:
            EventFilter instance.
        """
        return cls(
            lambda e: e.event_type == event_type,
            f"type={event_type.name}",
        )

    @classmethod
    def by_source(cls, source: str) -> "EventFilter":
        """Create filter for a specific source.

        Args:
            source: Source name to filter for.

        Returns:
            EventFilter instance.
        """
        return cls(
            lambda e: e.source == source,
            f"source={source}",
        )

    @classmethod
    def by_priority(cls, min_priority: int) -> "EventFilter":
        """Create filter for minimum priority.

        Args:
            min_priority: Minimum priority to include.

        Returns:
            EventFilter instance.
        """
        return cls(
            lambda e: e.priority >= min_priority,
            f"priority>={min_priority}",
        )

    @classmethod
    def by_data_key(cls, key: str, value: Any = None) -> "EventFilter":
        """Create filter for events with specific data.

        Args:
            key: Data key that must be present.
            value: Optional value the key must have.

        Returns:
            EventFilter instance.
        """
        if value is None:
            return cls(
                lambda e: key in e.data,
                f"data.{key} exists",
            )
        else:
            return cls(
                lambda e: e.data.get(key) == value,
                f"data.{key}={value}",
            )


# =============================================================================
# Filtered Subscriber
# =============================================================================


class FilteredSubscriber:
    """Subscriber wrapper that filters events before callback.

    Example:
        >>> subscriber = FilteredSubscriber(
        ...     callback=my_handler,
        ...     filter=EventFilter.by_type(EventType.ERROR),
        ... )
        >>> bus.subscribe(None, subscriber)  # Subscribe to all, filter applied
    """

    def __init__(
        self,
        callback: EventCallback,
        event_filter: EventFilter,
    ) -> None:
        """Initialize filtered subscriber.

        Args:
            callback: Callback to call for matching events.
            event_filter: Filter to apply.
        """
        self._callback = callback
        self._filter = event_filter

    def __call__(self, event: Event) -> None:
        """Process event if it matches filter."""
        if self._filter.matches(event):
            self._callback(event)


# =============================================================================
# Global Event Bus Instance
# =============================================================================


_global_bus: Optional[EventBus] = None
_global_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Creates the bus on first call. Thread-safe.

    Returns:
        Global EventBus instance.
    """
    global _global_bus

    if _global_bus is None:
        with _global_bus_lock:
            if _global_bus is None:
                _global_bus = EventBus(enable_history=True)

    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus.

    Shuts down and recreates the global bus. Useful for testing.
    """
    global _global_bus

    with _global_bus_lock:
        if _global_bus is not None:
            _global_bus.shutdown(wait=True)
            _global_bus = None


# =============================================================================
# Convenience Functions
# =============================================================================


def subscribe(
    event_type: Union[EventType, None],
    callback: EventCallback,
) -> None:
    """Subscribe to events on the global bus.

    Args:
        event_type: Type to subscribe to, or None for all.
        callback: Callback function.
    """
    get_event_bus().subscribe(event_type, callback)


def emit(event: Event) -> None:
    """Emit an event on the global bus.

    Args:
        event: Event to emit.
    """
    get_event_bus().emit(event)


def emit_async(event: Event) -> None:
    """Emit an event asynchronously on the global bus.

    Args:
        event: Event to emit.
    """
    get_event_bus().emit_async(event)


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Event types
    "EventType",
    # Event classes
    "Event",
    "ProcessingStartedEvent",
    "FrameProcessedEvent",
    "StageCompletedEvent",
    "ProcessingErrorEvent",
    "ProgressEvent",
    "QualityCheckEvent",
    # Event bus
    "EventBus",
    "EventCallback",
    # Filtering
    "EventFilter",
    "FilteredSubscriber",
    # Global bus
    "get_event_bus",
    "reset_event_bus",
    # Convenience functions
    "subscribe",
    "emit",
    "emit_async",
]
