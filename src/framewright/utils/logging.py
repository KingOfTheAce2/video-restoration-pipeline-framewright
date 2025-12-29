"""Structured logging utilities for FrameWright.

This module provides configurable, structured logging with support for:
- JSON format for production/machine parsing
- Human-readable text format for development
- Component-specific log levels
- Log rotation and file management
- Processing metrics and timing
- Error aggregation

Example usage:
    >>> from framewright.utils.logging import get_logger, LogConfig, configure_logging
    >>>
    >>> # Configure logging at application startup
    >>> config = LogConfig(
    ...     log_level="INFO",
    ...     log_format="json",
    ...     log_file="./logs/framewright.log",
    ...     component_levels={"upscaler": "DEBUG", "interpolator": "WARNING"}
    ... )
    >>> configure_logging(config)
    >>>
    >>> # Get a component logger
    >>> logger = get_logger("upscaler")
    >>> logger.info("Processing frame", frame=1500, total=3000, fps=2.5)
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

# Type aliases
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]


@dataclass
class LogConfig:
    """Configuration for FrameWright logging.

    Attributes:
        log_level: Default log level for all components
        log_format: Output format ('text' for human-readable, 'json' for structured)
        log_file: Optional file path for log output
        component_levels: Dictionary of component-specific log levels
        max_file_size_mb: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of rotated log files to keep (default: 5)
        include_timestamp: Whether to include timestamps in output
        include_source: Whether to include source file/line information
    """

    log_level: LogLevel = "INFO"
    log_format: LogFormat = "text"
    log_file: Optional[str] = None
    component_levels: Dict[str, LogLevel] = field(default_factory=dict)
    max_file_size_mb: int = 10
    backup_count: int = 5
    include_timestamp: bool = True
    include_source: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log_level '{self.log_level}'. "
                f"Must be one of: {valid_levels}"
            )
        if self.log_format not in ("text", "json"):
            raise ValueError(
                f"Invalid log_format '{self.log_format}'. "
                "Must be 'text' or 'json'"
            )
        for component, level in self.component_levels.items():
            if level.upper() not in valid_levels:
                raise ValueError(
                    f"Invalid log level '{level}' for component '{component}'. "
                    f"Must be one of: {valid_levels}"
                )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogConfig":
        """Create LogConfig from dictionary."""
        return cls(
            log_level=data.get("log_level", "INFO"),
            log_format=data.get("log_format", "text"),
            log_file=data.get("log_file"),
            component_levels=data.get("component_levels", {}),
            max_file_size_mb=data.get("max_file_size_mb", 10),
            backup_count=data.get("backup_count", 5),
            include_timestamp=data.get("include_timestamp", True),
            include_source=data.get("include_source", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": self.log_file,
            "component_levels": self.component_levels,
            "max_file_size_mb": self.max_file_size_mb,
            "backup_count": self.backup_count,
            "include_timestamp": self.include_timestamp,
            "include_source": self.include_source,
        }


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs log records as JSON objects with consistent structure:
    {
        "timestamp": "2024-12-29T10:30:45.123Z",
        "level": "INFO",
        "component": "upscaler",
        "message": "Processing frame",
        "frame": 1500,
        "total": 3000,
        "fps": 2.5
    }
    """

    def __init__(self, include_source: bool = False) -> None:
        """Initialize JSON formatter.

        Args:
            include_source: Whether to include source file/line information
        """
        super().__init__()
        self.include_source = include_source

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # Build base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "component": record.name.split(".")[-1],  # Get last part of logger name
            "message": record.getMessage(),
        }

        # Add source information if enabled
        if self.include_source:
            log_entry["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter.

    Outputs log records in format:
    2024-12-29 10:30:45 | INFO     | upscaler | Processing frame 1500/3000 (2.5 fps)
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_source: bool = False,
    ) -> None:
        """Initialize text formatter.

        Args:
            include_timestamp: Whether to include timestamps
            include_source: Whether to include source file/line information
        """
        self.include_timestamp = include_timestamp
        self.include_source = include_source

        if include_timestamp:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
        else:
            fmt = "%(levelname)-8s | %(name)-12s | %(message)s"

        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text string."""
        # Append extra fields to message
        if hasattr(record, "extra_fields") and record.extra_fields:
            extra_str = ", ".join(
                f"{k}={v}" for k, v in record.extra_fields.items()
            )
            record.msg = f"{record.msg} [{extra_str}]"

        # Add source info if enabled
        if self.include_source:
            source_info = f" ({record.filename}:{record.lineno})"
            record.msg = f"{record.msg}{source_info}"

        return super().format(record)


class FramewrightLogger(logging.LoggerAdapter):
    """Enhanced logger for FrameWright with structured logging support.

    Provides convenient methods for logging with extra structured data
    that will be formatted appropriately based on output format.
    """

    def __init__(
        self,
        logger: logging.Logger,
        component: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize FrameWright logger.

        Args:
            logger: Base Python logger
            component: Component name for this logger
            extra: Default extra fields to include in all messages
        """
        super().__init__(logger, extra or {})
        self.component = component

    def process(
        self,
        msg: str,
        kwargs: Dict[str, Any],
    ) -> tuple:
        """Process log message to add extra fields."""
        # Extract extra fields from kwargs
        extra_fields = {}
        for key in list(kwargs.keys()):
            if key not in ("exc_info", "stack_info", "stacklevel", "extra"):
                extra_fields[key] = kwargs.pop(key)

        # Merge with default extra fields
        if self.extra:
            extra_fields.update(self.extra)

        # Add to record
        kwargs.setdefault("extra", {})
        kwargs["extra"]["extra_fields"] = extra_fields

        return msg, kwargs

    def processing_start(
        self,
        operation: str,
        **kwargs: Any,
    ) -> None:
        """Log start of a processing operation."""
        self.info(f"Starting {operation}", operation=operation, **kwargs)

    def processing_complete(
        self,
        operation: str,
        duration_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log completion of a processing operation."""
        if duration_seconds is not None:
            self.info(
                f"Completed {operation}",
                operation=operation,
                duration_seconds=round(duration_seconds, 2),
                **kwargs,
            )
        else:
            self.info(f"Completed {operation}", operation=operation, **kwargs)

    def processing_progress(
        self,
        operation: str,
        current: int,
        total: int,
        fps: Optional[float] = None,
        eta_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log processing progress update."""
        progress_pct = (current / total * 100) if total > 0 else 0
        extra = {
            "operation": operation,
            "current": current,
            "total": total,
            "progress_pct": round(progress_pct, 1),
        }
        if fps is not None:
            extra["fps"] = round(fps, 2)
        if eta_seconds is not None:
            extra["eta_seconds"] = round(eta_seconds, 1)
        extra.update(kwargs)

        self.info(f"{operation} progress: {current}/{total}", **extra)

    def frame_processed(
        self,
        frame_number: int,
        total_frames: int,
        processing_time_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log frame processing completion."""
        extra = {
            "frame": frame_number,
            "total": total_frames,
        }
        if processing_time_ms is not None:
            extra["processing_time_ms"] = round(processing_time_ms, 1)
        extra.update(kwargs)

        self.debug(f"Processed frame {frame_number}/{total_frames}", **extra)

    def metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a metric value."""
        extra = {
            "metric_name": metric_name,
            "metric_value": value,
        }
        if unit:
            extra["metric_unit"] = unit
        extra.update(kwargs)

        self.info(f"Metric: {metric_name}={value}{unit or ''}", **extra)


# Global configuration
_log_config: Optional[LogConfig] = None
_configured_loggers: Dict[str, FramewrightLogger] = {}


def configure_logging(config: Optional[LogConfig] = None) -> None:
    """Configure global logging settings.

    This should be called once at application startup to set up
    logging handlers and formatters.

    Args:
        config: Logging configuration. If None, uses defaults.
    """
    global _log_config

    if config is None:
        config = LogConfig()

    _log_config = config

    # Get or create root framewright logger
    root_logger = logging.getLogger("framewright")
    root_logger.setLevel(getattr(logging, config.log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create appropriate formatter
    if config.log_format == "json":
        formatter = JSONFormatter(include_source=config.include_source)
    else:
        formatter = TextFormatter(
            include_timestamp=config.include_timestamp,
            include_source=config.include_source,
        )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, config.log_level.upper()))
    root_logger.addHandler(console_handler)

    # Add file handler if configured
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, config.log_level.upper()))
        root_logger.addHandler(file_handler)

    # Configure component-specific levels
    for component, level in config.component_levels.items():
        component_logger = logging.getLogger(f"framewright.{component}")
        component_logger.setLevel(getattr(logging, level.upper()))

    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(component: str) -> FramewrightLogger:
    """Get a logger for a specific component.

    Args:
        component: Component name (e.g., 'upscaler', 'interpolator')

    Returns:
        Configured FramewrightLogger instance

    Example:
        >>> logger = get_logger("upscaler")
        >>> logger.info("Processing", frame=100, total=1000)
    """
    global _configured_loggers

    # Configure logging if not already done
    if _log_config is None:
        configure_logging()

    # Return cached logger if available
    if component in _configured_loggers:
        return _configured_loggers[component]

    # Create new logger
    base_logger = logging.getLogger(f"framewright.{component}")

    # Apply component-specific level if configured
    if _log_config and component in _log_config.component_levels:
        level = _log_config.component_levels[component]
        base_logger.setLevel(getattr(logging, level.upper()))

    logger = FramewrightLogger(base_logger, component)
    _configured_loggers[component] = logger

    return logger


def get_config() -> Optional[LogConfig]:
    """Get current logging configuration."""
    return _log_config


def set_level(level: LogLevel, component: Optional[str] = None) -> None:
    """Set log level dynamically.

    Args:
        level: New log level
        component: Component to set level for (None for root)
    """
    if component:
        logger = logging.getLogger(f"framewright.{component}")
    else:
        logger = logging.getLogger("framewright")

    logger.setLevel(getattr(logging, level.upper()))


def add_file_handler(
    log_file: str,
    level: Optional[LogLevel] = None,
    log_format: Optional[LogFormat] = None,
) -> None:
    """Add an additional file handler dynamically.

    Args:
        log_file: Path to log file
        level: Log level for this file (None uses current config)
        log_format: Format for this file (None uses current config)
    """
    config = _log_config or LogConfig()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatter
    fmt = log_format or config.log_format
    if fmt == "json":
        formatter = JSONFormatter(include_source=config.include_source)
    else:
        formatter = TextFormatter(
            include_timestamp=config.include_timestamp,
            include_source=config.include_source,
        )

    # Create and add handler
    handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_file_size_mb * 1024 * 1024,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.setLevel(getattr(logging, (level or config.log_level).upper()))

    root_logger = logging.getLogger("framewright")
    root_logger.addHandler(handler)


@dataclass
class ProcessingMetricsLog:
    """Aggregates processing metrics for logging.

    Use this to track and log processing statistics over time.
    """

    component: str
    operation: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frames_processed: int = 0
    total_frames: int = 0
    errors: int = 0
    warnings: int = 0
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    _logger: Optional[FramewrightLogger] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize logger."""
        self._logger = get_logger(self.component)

    def record_frame(
        self,
        processing_time_ms: float,
        **extra_metrics: float,
    ) -> None:
        """Record a processed frame."""
        self.frames_processed += 1

        # Track processing time
        if "processing_time_ms" not in self.metrics:
            self.metrics["processing_time_ms"] = []
        self.metrics["processing_time_ms"].append(processing_time_ms)

        # Track extra metrics
        for name, value in extra_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def record_error(self, error: Exception) -> None:
        """Record an error."""
        self.errors += 1
        if self._logger:
            self._logger.error(
                f"Error in {self.operation}",
                error_type=type(error).__name__,
                error_message=str(error),
            )

    def record_warning(self, message: str) -> None:
        """Record a warning."""
        self.warnings += 1
        if self._logger:
            self._logger.warning(message, operation=self.operation)

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from recorded metrics."""
        stats: Dict[str, Any] = {
            "operation": self.operation,
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": (
                datetime.now(timezone.utc) - self.start_time
            ).total_seconds(),
        }

        # Calculate stats for each metric
        for name, values in self.metrics.items():
            if values:
                stats[f"{name}_avg"] = sum(values) / len(values)
                stats[f"{name}_min"] = min(values)
                stats[f"{name}_max"] = max(values)

        # Calculate overall FPS
        if stats["duration_seconds"] > 0:
            stats["fps"] = self.frames_processed / stats["duration_seconds"]

        return stats

    def log_summary(self) -> None:
        """Log a summary of processing metrics."""
        if self._logger:
            stats = self.get_statistics()
            self._logger.info(
                f"{self.operation} complete",
                **stats,
            )

    def log_progress(self, interval: int = 100) -> None:
        """Log progress if frames processed is multiple of interval."""
        if self._logger and self.frames_processed % interval == 0:
            duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            fps = self.frames_processed / duration if duration > 0 else 0

            eta = None
            if fps > 0 and self.total_frames > 0:
                remaining = self.total_frames - self.frames_processed
                eta = remaining / fps

            self._logger.processing_progress(
                operation=self.operation,
                current=self.frames_processed,
                total=self.total_frames,
                fps=fps,
                eta_seconds=eta,
            )


class ErrorAggregator:
    """Aggregates errors for batch operations.

    Provides summary statistics and categorization of errors
    that occur during batch processing.
    """

    def __init__(self, component: str) -> None:
        """Initialize error aggregator.

        Args:
            component: Component name for logging
        """
        self.component = component
        self.errors: List[Dict[str, Any]] = []
        self._logger = get_logger(component)

    def add_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an error to the aggregator.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {},
        }
        self.errors.append(error_entry)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of aggregated errors.

        Returns:
            Dictionary with error counts by type and total
        """
        by_type: Dict[str, int] = {}
        for error in self.errors:
            error_type = error["error_type"]
            by_type[error_type] = by_type.get(error_type, 0) + 1

        return {
            "total_errors": len(self.errors),
            "by_type": by_type,
            "errors": self.errors,
        }

    def log_summary(self) -> None:
        """Log a summary of errors."""
        summary = self.get_summary()
        if summary["total_errors"] > 0:
            self._logger.warning(
                f"Error summary: {summary['total_errors']} total errors",
                error_summary=summary["by_type"],
            )
        else:
            self._logger.info("No errors recorded")

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0

    def clear(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()


# Convenience functions for CLI integration


def configure_from_cli(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> LogConfig:
    """Configure logging from CLI arguments.

    Args:
        log_level: Log level from --log-level argument
        log_format: Format from --log-format argument
        log_file: File path from --log-file argument

    Returns:
        Configured LogConfig instance
    """
    config = LogConfig(
        log_level=log_level.upper() if log_level else "INFO",
        log_format=log_format if log_format in ("text", "json") else "text",
        log_file=log_file,
    )
    configure_logging(config)
    return config


def get_cli_args_parser():
    """Get argparse arguments for logging configuration.

    Returns:
        List of argument tuples for argparse.add_argument()
    """
    return [
        (
            ("--log-level",),
            {
                "type": str,
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
                "help": "Set logging level (default: INFO)",
            },
        ),
        (
            ("--log-format",),
            {
                "type": str,
                "choices": ["text", "json"],
                "default": "text",
                "help": "Set logging format (default: text)",
            },
        ),
        (
            ("--log-file",),
            {
                "type": str,
                "default": None,
                "help": "Path to log file (default: stderr only)",
            },
        ),
    ]
