"""Pipeline orchestration for video restoration processing.

This module provides a flexible, event-driven pipeline system for
orchestrating multi-stage video restoration workflows. Key features:

- PipelineStage: Configuration for individual processing stages
- Pipeline: Main orchestrator with validation, execution, and error recovery
- PipelineBuilder: Fluent API for constructing pipelines
- Event emission at each stage for monitoring and logging

Example:
    >>> pipeline = PipelineBuilder()
    ...     .add_analysis()
    ...     .add_denoising(strength=0.5)
    ...     .add_face_restoration()
    ...     .add_upscaling(scale=4)
    ...     .build()
    >>> result = pipeline.run("input.mp4", "output.mp4")
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class PipelineEventType(Enum):
    """Types of events emitted during pipeline execution."""

    PIPELINE_STARTED = auto()
    PIPELINE_COMPLETED = auto()
    PIPELINE_FAILED = auto()
    PIPELINE_CANCELLED = auto()

    STAGE_STARTED = auto()
    STAGE_COMPLETED = auto()
    STAGE_FAILED = auto()
    STAGE_SKIPPED = auto()

    FRAME_PROCESSED = auto()
    PROGRESS_UPDATE = auto()

    CHECKPOINT_CREATED = auto()
    CHECKPOINT_RESTORED = auto()

    WARNING = auto()
    ERROR = auto()


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PipelineStatus(Enum):
    """Status of the overall pipeline."""

    NOT_STARTED = "not_started"
    VALIDATING = "validating"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Protocols for Processors
# =============================================================================


class FrameProcessor(Protocol):
    """Protocol for frame-based processors."""

    def process_frame(self, frame: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Process a single frame.

        Args:
            frame: Input frame as numpy array (BGR format).
            **kwargs: Additional processor-specific arguments.

        Returns:
            Processed frame as numpy array.
        """
        ...


class VideoProcessor(Protocol):
    """Protocol for video-based processors."""

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> bool:
        """Process a video file.

        Args:
            input_path: Path to input video.
            output_path: Path for output video.
            progress_callback: Optional progress callback (0.0-1.0).
            **kwargs: Additional processor-specific arguments.

        Returns:
            True if processing succeeded.
        """
        ...


ProcessorType = Union[FrameProcessor, VideoProcessor, Callable[..., Any]]


# =============================================================================
# Event System
# =============================================================================


@dataclass
class PipelineEvent:
    """Event emitted during pipeline execution.

    Attributes:
        event_type: Type of event.
        timestamp: Unix timestamp when event occurred.
        pipeline_id: Unique identifier for the pipeline run.
        stage_name: Name of the stage (if applicable).
        data: Event-specific data dictionary.
        message: Human-readable event message.
    """

    event_type: PipelineEventType
    timestamp: float = field(default_factory=time.time)
    pipeline_id: str = ""
    stage_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


EventCallback = Callable[[PipelineEvent], None]


class EventEmitter:
    """Manages event subscriptions and emission."""

    def __init__(self) -> None:
        """Initialize the event emitter."""
        self._listeners: Dict[PipelineEventType, List[EventCallback]] = {}
        self._global_listeners: List[EventCallback] = []
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_type: Optional[PipelineEventType],
        callback: EventCallback,
    ) -> Callable[[], None]:
        """Subscribe to events.

        Args:
            event_type: Specific event type or None for all events.
            callback: Function to call when event occurs.

        Returns:
            Unsubscribe function.
        """
        with self._lock:
            if event_type is None:
                self._global_listeners.append(callback)
                return lambda: self._global_listeners.remove(callback)
            else:
                if event_type not in self._listeners:
                    self._listeners[event_type] = []
                self._listeners[event_type].append(callback)
                return lambda: self._listeners[event_type].remove(callback)

    def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all subscribed listeners.

        Args:
            event: The event to emit.
        """
        with self._lock:
            listeners = list(self._global_listeners)
            if event.event_type in self._listeners:
                listeners.extend(self._listeners[event.event_type])

        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")


# =============================================================================
# Pipeline Stage Configuration
# =============================================================================


@dataclass
class StageConfig:
    """Configuration for a pipeline stage.

    Attributes:
        enabled: Whether the stage is enabled.
        params: Stage-specific parameters.
        timeout_seconds: Maximum execution time (None for no limit).
        retry_count: Number of retries on failure.
        retry_delay_seconds: Delay between retries.
        skip_on_failure: Continue pipeline if this stage fails.
        checkpoint_after: Create checkpoint after this stage.
    """

    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    skip_on_failure: bool = False
    checkpoint_after: bool = False


@dataclass
class PipelineStage:
    """A stage in the processing pipeline.

    Attributes:
        name: Unique name for this stage.
        processor: The processor to execute.
        config: Stage configuration.
        dependencies: Names of stages that must complete first.
        description: Human-readable description.
        estimated_time_factor: Relative processing time (1.0 = baseline).
    """

    name: str
    processor: ProcessorType
    config: StageConfig = field(default_factory=StageConfig)
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    estimated_time_factor: float = 1.0

    # Runtime state
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    result: Any = None

    def reset(self) -> None:
        """Reset stage runtime state."""
        self.status = StageStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.result = None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class StageResult:
    """Result of executing a pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        status: Final status.
        elapsed_seconds: Time taken to execute.
        output: Stage output (if any).
        error: Exception if failed.
        retry_count: Number of retries attempted.
    """

    stage_name: str
    status: StageStatus
    elapsed_seconds: float = 0.0
    output: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        pipeline_id: Unique run identifier.
        status: Final pipeline status.
        input_path: Input file path.
        output_path: Output file path.
        stage_results: Results for each stage.
        total_elapsed_seconds: Total execution time.
        frames_processed: Number of frames processed.
        error: Exception if failed.
        metadata: Additional result metadata.
    """

    pipeline_id: str
    status: PipelineStatus
    input_path: Path
    output_path: Path
    stage_results: List[StageResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    frames_processed: int = 0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.status == PipelineStatus.COMPLETED

    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage."""
        for result in self.stage_results:
            if result.stage_name == stage_name:
                return result
        return None


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationResult:
    """Result of pipeline validation.

    Attributes:
        valid: Whether the pipeline is valid.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


class PipelineValidator:
    """Validates pipeline configuration."""

    def validate(self, pipeline: "Pipeline") -> ValidationResult:
        """Validate a pipeline configuration.

        Args:
            pipeline: Pipeline to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        result = ValidationResult()

        # Check for stages
        if not pipeline.stages:
            result.add_error("Pipeline has no stages")
            return result

        # Check for duplicate stage names
        names = [s.name for s in pipeline.stages]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            result.add_error(f"Duplicate stage names: {set(duplicates)}")

        # Validate dependencies
        stage_names = set(names)
        for stage in pipeline.stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    result.add_error(
                        f"Stage '{stage.name}' has unknown dependency '{dep}'"
                    )

        # Check for circular dependencies
        if self._has_circular_dependency(pipeline.stages):
            result.add_error("Pipeline has circular dependencies")

        # Validate processor compatibility
        for stage in pipeline.stages:
            if stage.processor is None:
                result.add_error(f"Stage '{stage.name}' has no processor")

        # Check for disabled critical stages
        critical_stages = {"analysis", "enhancement", "upscaling"}
        disabled_critical = [
            s.name for s in pipeline.stages
            if s.name in critical_stages and not s.config.enabled
        ]
        if disabled_critical:
            result.add_warning(
                f"Critical stages disabled: {disabled_critical}"
            )

        return result

    def _has_circular_dependency(
        self,
        stages: List[PipelineStage],
    ) -> bool:
        """Check for circular dependencies using DFS."""
        stage_map = {s.name: s for s in stages}
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)

            stage = stage_map.get(name)
            if stage:
                for dep in stage.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(name)
            return False

        for stage in stages:
            if stage.name not in visited:
                if has_cycle(stage.name):
                    return True

        return False


# =============================================================================
# Pipeline Execution Context
# =============================================================================


@dataclass
class PipelineContext:
    """Runtime context for pipeline execution.

    Attributes:
        pipeline_id: Unique run identifier.
        input_path: Input file path.
        output_path: Output file path.
        temp_dir: Temporary directory for intermediate files.
        shared_data: Data shared between stages.
        cancelled: Flag indicating cancellation requested.
        paused: Flag indicating pause requested.
    """

    pipeline_id: str
    input_path: Path
    output_path: Path
    temp_dir: Optional[Path] = None
    shared_data: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    paused: bool = False
    _pause_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        """Initialize pause event as not paused."""
        self._pause_event.set()

    def request_cancel(self) -> None:
        """Request pipeline cancellation."""
        self.cancelled = True
        self._pause_event.set()  # Unblock if paused

    def request_pause(self) -> None:
        """Request pipeline pause."""
        self.paused = True
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume paused pipeline."""
        self.paused = False
        self._pause_event.set()

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """Wait if pipeline is paused.

        Args:
            timeout: Maximum time to wait (None for indefinite).

        Returns:
            True if resumed, False if cancelled or timeout.
        """
        if self.cancelled:
            return False
        return self._pause_event.wait(timeout)


# =============================================================================
# Main Pipeline Class
# =============================================================================


class Pipeline:
    """Orchestrates full video restoration pipeline.

    The Pipeline class manages multi-stage video processing with:
    - Dependency-based stage ordering
    - Event emission for monitoring
    - Error handling and recovery
    - Checkpoint support for resume
    - Async execution support

    Example:
        >>> pipeline = Pipeline()
        >>> pipeline.add_stage(analyzer, StageConfig(params={"sample_rate": 10}))
        >>> pipeline.add_stage(denoiser, StageConfig(params={"strength": 0.5}))
        >>> pipeline.add_stage(upscaler, StageConfig(params={"scale": 4}))
        >>> result = pipeline.run("input.mp4", "output.mp4")
    """

    def __init__(
        self,
        name: str = "default",
        max_workers: int = 1,
        checkpoint_manager: Optional[Any] = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            name: Pipeline name for identification.
            max_workers: Maximum concurrent workers for parallel stages.
            checkpoint_manager: Optional checkpoint manager for resume support.
        """
        self.name = name
        self.max_workers = max_workers
        self.checkpoint_manager = checkpoint_manager

        self.stages: List[PipelineStage] = []
        self._event_emitter = EventEmitter()
        self._validator = PipelineValidator()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._context: Optional[PipelineContext] = None
        self._status = PipelineStatus.NOT_STARTED
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Stage Management
    # -------------------------------------------------------------------------

    def add_stage(
        self,
        processor: ProcessorType,
        config: Optional[StageConfig] = None,
        name: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
        estimated_time_factor: float = 1.0,
    ) -> "Pipeline":
        """Add a processing stage to the pipeline.

        Args:
            processor: The processor to execute.
            config: Stage configuration.
            name: Unique stage name (auto-generated if None).
            dependencies: Names of stages that must complete first.
            description: Human-readable description.
            estimated_time_factor: Relative processing time.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If stage name already exists.
        """
        if name is None:
            # Generate name from processor
            if hasattr(processor, "__class__"):
                name = processor.__class__.__name__
            elif hasattr(processor, "__name__"):
                name = processor.__name__
            else:
                name = f"stage_{len(self.stages)}"

        # Ensure unique name
        existing_names = {s.name for s in self.stages}
        if name in existing_names:
            base_name = name
            counter = 1
            while name in existing_names:
                name = f"{base_name}_{counter}"
                counter += 1

        stage = PipelineStage(
            name=name,
            processor=processor,
            config=config or StageConfig(),
            dependencies=dependencies or [],
            description=description,
            estimated_time_factor=estimated_time_factor,
        )

        self.stages.append(stage)
        logger.debug(f"Added stage: {name}")
        return self

    def remove_stage(self, name: str) -> bool:
        """Remove a stage from the pipeline.

        Args:
            name: Name of the stage to remove.

        Returns:
            True if stage was removed.
        """
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                self.stages.pop(i)
                # Remove from other stages' dependencies
                for other in self.stages:
                    if name in other.dependencies:
                        other.dependencies.remove(name)
                logger.debug(f"Removed stage: {name}")
                return True
        return False

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name.

        Args:
            name: Stage name.

        Returns:
            PipelineStage or None if not found.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def enable_stage(self, name: str) -> bool:
        """Enable a stage.

        Args:
            name: Stage name.

        Returns:
            True if stage was found and enabled.
        """
        stage = self.get_stage(name)
        if stage:
            stage.config.enabled = True
            return True
        return False

    def disable_stage(self, name: str) -> bool:
        """Disable a stage.

        Args:
            name: Stage name.

        Returns:
            True if stage was found and disabled.
        """
        stage = self.get_stage(name)
        if stage:
            stage.config.enabled = False
            return True
        return False

    def configure_stage(self, name: str, **params: Any) -> bool:
        """Update stage configuration parameters.

        Args:
            name: Stage name.
            **params: Parameters to update.

        Returns:
            True if stage was found and configured.
        """
        stage = self.get_stage(name)
        if stage:
            stage.config.params.update(params)
            return True
        return False

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on_event(
        self,
        event_type: Optional[PipelineEventType],
        callback: EventCallback,
    ) -> Callable[[], None]:
        """Subscribe to pipeline events.

        Args:
            event_type: Specific event type or None for all events.
            callback: Function to call when event occurs.

        Returns:
            Unsubscribe function.
        """
        return self._event_emitter.subscribe(event_type, callback)

    def _emit(
        self,
        event_type: PipelineEventType,
        stage_name: Optional[str] = None,
        message: str = "",
        **data: Any,
    ) -> None:
        """Emit a pipeline event.

        Args:
            event_type: Type of event.
            stage_name: Name of the stage (if applicable).
            message: Human-readable message.
            **data: Additional event data.
        """
        event = PipelineEvent(
            event_type=event_type,
            pipeline_id=self._context.pipeline_id if self._context else "",
            stage_name=stage_name,
            message=message,
            data=data,
        )
        self._event_emitter.emit(event)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Validate the pipeline configuration.

        Returns:
            ValidationResult with any errors or warnings.
        """
        return self._validator.validate(self)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        resume_from_checkpoint: bool = True,
    ) -> PipelineResult:
        """Execute the pipeline synchronously.

        Args:
            input_path: Path to input video file.
            output_path: Path for output video file.
            progress_callback: Optional callback(progress, message).
            resume_from_checkpoint: Try to resume from checkpoint.

        Returns:
            PipelineResult with execution details.

        Raises:
            ValueError: If pipeline validation fails.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Validate pipeline
        validation = self.validate()
        if not validation.valid:
            raise ValueError(
                f"Pipeline validation failed: {validation.errors}"
            )

        # Create context
        pipeline_id = str(uuid.uuid4())[:8]
        self._context = PipelineContext(
            pipeline_id=pipeline_id,
            input_path=input_path,
            output_path=output_path,
        )

        # Initialize result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.NOT_STARTED,
            input_path=input_path,
            output_path=output_path,
        )

        start_time = time.time()
        self._status = PipelineStatus.RUNNING

        try:
            # Emit start event
            self._emit(
                PipelineEventType.PIPELINE_STARTED,
                message=f"Starting pipeline '{self.name}'",
                input_path=str(input_path),
                output_path=str(output_path),
                stages=[s.name for s in self.stages if s.config.enabled],
            )

            # Check for checkpoint resume
            resume_stage_idx = 0
            if resume_from_checkpoint and self.checkpoint_manager:
                resume_stage_idx = self._try_restore_checkpoint()

            # Execute stages in dependency order
            execution_order = self._get_execution_order()
            total_stages = len([s for s in execution_order if s.config.enabled])
            completed_stages = 0

            for stage in execution_order:
                # Check for cancellation
                if self._context.cancelled:
                    self._status = PipelineStatus.CANCELLED
                    result.status = PipelineStatus.CANCELLED
                    self._emit(
                        PipelineEventType.PIPELINE_CANCELLED,
                        message="Pipeline cancelled by user",
                    )
                    break

                # Wait if paused
                if not self._context.wait_if_paused():
                    # Cancelled during pause
                    self._status = PipelineStatus.CANCELLED
                    result.status = PipelineStatus.CANCELLED
                    break

                # Skip disabled stages
                if not stage.config.enabled:
                    stage.status = StageStatus.SKIPPED
                    result.stage_results.append(
                        StageResult(
                            stage_name=stage.name,
                            status=StageStatus.SKIPPED,
                        )
                    )
                    continue

                # Execute stage
                stage_result = self._execute_stage(stage)
                result.stage_results.append(stage_result)

                if stage_result.status == StageStatus.COMPLETED:
                    completed_stages += 1

                    # Progress callback
                    if progress_callback:
                        progress = completed_stages / total_stages
                        progress_callback(progress, f"Completed: {stage.name}")

                    # Checkpoint after stage if configured
                    if stage.config.checkpoint_after and self.checkpoint_manager:
                        self._create_checkpoint(stage.name)

                elif stage_result.status == StageStatus.FAILED:
                    if not stage.config.skip_on_failure:
                        # Fatal failure
                        self._status = PipelineStatus.FAILED
                        result.status = PipelineStatus.FAILED
                        result.error = stage_result.error
                        self._emit(
                            PipelineEventType.PIPELINE_FAILED,
                            stage_name=stage.name,
                            message=f"Pipeline failed at stage '{stage.name}'",
                            error=str(stage_result.error),
                        )
                        break

            # Check final status
            if self._status == PipelineStatus.RUNNING:
                self._status = PipelineStatus.COMPLETED
                result.status = PipelineStatus.COMPLETED
                self._emit(
                    PipelineEventType.PIPELINE_COMPLETED,
                    message=f"Pipeline '{self.name}' completed successfully",
                    stages_completed=completed_stages,
                )

        except Exception as e:
            self._status = PipelineStatus.FAILED
            result.status = PipelineStatus.FAILED
            result.error = e
            logger.exception(f"Pipeline execution failed: {e}")
            self._emit(
                PipelineEventType.PIPELINE_FAILED,
                message=f"Pipeline failed with exception: {e}",
                error=str(e),
                traceback=traceback.format_exc(),
            )

        finally:
            result.total_elapsed_seconds = time.time() - start_time
            self._context = None

        return result

    def run_async(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        resume_from_checkpoint: bool = True,
    ) -> Future[PipelineResult]:
        """Execute the pipeline asynchronously in a background thread.

        Args:
            input_path: Path to input video file.
            output_path: Path for output video file.
            progress_callback: Optional callback(progress, message).
            resume_from_checkpoint: Try to resume from checkpoint.

        Returns:
            Future that will contain PipelineResult.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

        return self._executor.submit(
            self.run,
            input_path,
            output_path,
            progress_callback,
            resume_from_checkpoint,
        )

    def cancel(self) -> bool:
        """Request pipeline cancellation.

        Returns:
            True if cancellation was requested.
        """
        if self._context:
            self._context.request_cancel()
            return True
        return False

    def pause(self) -> bool:
        """Request pipeline pause.

        Returns:
            True if pause was requested.
        """
        if self._context and self._status == PipelineStatus.RUNNING:
            self._context.request_pause()
            self._status = PipelineStatus.PAUSED
            return True
        return False

    def resume(self) -> bool:
        """Resume paused pipeline.

        Returns:
            True if resume was successful.
        """
        if self._context and self._status == PipelineStatus.PAUSED:
            self._context.resume()
            self._status = PipelineStatus.RUNNING
            return True
        return False

    @property
    def status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self._status

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _get_execution_order(self) -> List[PipelineStage]:
        """Get stages in dependency-resolved execution order.

        Returns:
            List of stages in execution order.
        """
        # Topological sort using Kahn's algorithm
        in_degree: Dict[str, int] = {s.name: 0 for s in self.stages}
        graph: Dict[str, List[str]] = {s.name: [] for s in self.stages}

        for stage in self.stages:
            for dep in stage.dependencies:
                if dep in graph:
                    graph[dep].append(stage.name)
                    in_degree[stage.name] += 1

        # Start with stages that have no dependencies
        queue = [s for s in self.stages if in_degree[s.name] == 0]
        result = []

        while queue:
            # Sort by priority (stages without dependencies first)
            queue.sort(key=lambda s: s.estimated_time_factor)
            stage = queue.pop(0)
            result.append(stage)

            for dependent_name in graph[stage.name]:
                in_degree[dependent_name] -= 1
                if in_degree[dependent_name] == 0:
                    dependent = self.get_stage(dependent_name)
                    if dependent:
                        queue.append(dependent)

        return result

    def _execute_stage(self, stage: PipelineStage) -> StageResult:
        """Execute a single pipeline stage.

        Args:
            stage: Stage to execute.

        Returns:
            StageResult with execution details.
        """
        stage.reset()
        stage.status = StageStatus.RUNNING
        stage.start_time = time.time()
        retry_count = 0

        self._emit(
            PipelineEventType.STAGE_STARTED,
            stage_name=stage.name,
            message=f"Starting stage: {stage.name}",
            params=stage.config.params,
        )

        while True:
            try:
                # Execute the processor
                result = self._run_processor(stage)
                stage.result = result
                stage.status = StageStatus.COMPLETED
                stage.end_time = time.time()

                self._emit(
                    PipelineEventType.STAGE_COMPLETED,
                    stage_name=stage.name,
                    message=f"Stage completed: {stage.name}",
                    elapsed_seconds=stage.elapsed_seconds,
                )

                return StageResult(
                    stage_name=stage.name,
                    status=StageStatus.COMPLETED,
                    elapsed_seconds=stage.elapsed_seconds or 0.0,
                    output=result,
                    retry_count=retry_count,
                )

            except Exception as e:
                logger.warning(f"Stage '{stage.name}' failed: {e}")

                # Check for retry
                if retry_count < stage.config.retry_count:
                    retry_count += 1
                    logger.info(
                        f"Retrying stage '{stage.name}' "
                        f"({retry_count}/{stage.config.retry_count})"
                    )
                    time.sleep(stage.config.retry_delay_seconds)
                    continue

                # No more retries
                stage.status = StageStatus.FAILED
                stage.error = e
                stage.end_time = time.time()

                self._emit(
                    PipelineEventType.STAGE_FAILED,
                    stage_name=stage.name,
                    message=f"Stage failed: {stage.name}",
                    error=str(e),
                    retry_count=retry_count,
                )

                return StageResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    elapsed_seconds=stage.elapsed_seconds or 0.0,
                    error=e,
                    retry_count=retry_count,
                )

    def _run_processor(self, stage: PipelineStage) -> Any:
        """Run a stage's processor.

        Args:
            stage: Stage containing the processor.

        Returns:
            Processor result.
        """
        processor = stage.processor
        params = stage.config.params
        context = self._context

        if context is None:
            raise RuntimeError("No execution context available")

        # Check if processor is a VideoProcessor
        if hasattr(processor, "process_video"):

            def progress_cb(progress: float) -> None:
                self._emit(
                    PipelineEventType.PROGRESS_UPDATE,
                    stage_name=stage.name,
                    progress=progress,
                )

            return processor.process_video(
                context.input_path,
                context.output_path,
                progress_callback=progress_cb,
                **params,
            )

        # Check if processor is a FrameProcessor
        elif hasattr(processor, "process_frame"):
            # For frame processors, we need to handle video I/O
            return self._process_video_frames(processor, stage)

        # Check if processor is a callable
        elif callable(processor):
            return processor(
                input_path=context.input_path,
                output_path=context.output_path,
                **params,
            )

        else:
            raise TypeError(
                f"Unknown processor type for stage '{stage.name}': "
                f"{type(processor)}"
            )

    def _process_video_frames(
        self,
        processor: FrameProcessor,
        stage: PipelineStage,
    ) -> int:
        """Process video frame by frame using a FrameProcessor.

        Args:
            processor: Frame processor to use.
            stage: Current stage.

        Returns:
            Number of frames processed.
        """
        # This is a simplified implementation
        # In production, use the VideoRestorer's frame processing
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for frame processing")

        context = self._context
        if context is None:
            raise RuntimeError("No execution context")

        # Open video
        cap = cv2.VideoCapture(str(context.input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {context.input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create output writer
        out = cv2.VideoWriter(
            str(context.output_path),
            fourcc,
            fps,
            (width, height),
        )

        frames_processed = 0
        params = stage.config.params

        try:
            while True:
                if context.cancelled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed = processor.process_frame(frame, **params)
                out.write(processed)
                frames_processed += 1

                # Emit progress
                if frames_processed % 10 == 0:
                    progress = frames_processed / total_frames
                    self._emit(
                        PipelineEventType.FRAME_PROCESSED,
                        stage_name=stage.name,
                        frame_number=frames_processed,
                        total_frames=total_frames,
                        progress=progress,
                    )

        finally:
            cap.release()
            out.release()

        return frames_processed

    def _try_restore_checkpoint(self) -> int:
        """Try to restore from checkpoint.

        Returns:
            Index of stage to resume from (0 if no checkpoint).
        """
        if not self.checkpoint_manager:
            return 0

        try:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                # Find the stage to resume from
                for i, stage in enumerate(self.stages):
                    if stage.name == checkpoint.get("last_completed_stage"):
                        self._emit(
                            PipelineEventType.CHECKPOINT_RESTORED,
                            message=f"Restored from checkpoint at stage '{stage.name}'",
                            stage_name=stage.name,
                        )
                        return i + 1
        except Exception as e:
            logger.warning(f"Failed to restore checkpoint: {e}")

        return 0

    def _create_checkpoint(self, stage_name: str) -> None:
        """Create a checkpoint after completing a stage.

        Args:
            stage_name: Name of completed stage.
        """
        if not self.checkpoint_manager or not self._context:
            return

        try:
            self.checkpoint_manager.save(
                self._context.pipeline_id,
                {
                    "last_completed_stage": stage_name,
                    "pipeline_name": self.name,
                    "input_path": str(self._context.input_path),
                    "output_path": str(self._context.output_path),
                },
            )
            self._emit(
                PipelineEventType.CHECKPOINT_CREATED,
                stage_name=stage_name,
                message=f"Checkpoint created after stage '{stage_name}'",
            )
        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        stage_names = [s.name for s in self.stages]
        return f"Pipeline(name='{self.name}', stages={stage_names})"


# =============================================================================
# Pipeline Builder
# =============================================================================


class PipelineBuilder:
    """Fluent API for building pipelines.

    Example:
        >>> pipeline = PipelineBuilder()
        ...     .with_name("restoration")
        ...     .add_analysis()
        ...     .add_denoising(strength=0.5)
        ...     .add_face_restoration()
        ...     .add_upscaling(scale=4)
        ...     .with_checkpoint_manager(manager)
        ...     .build()
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._name = "default"
        self._stages: List[Tuple[str, ProcessorType, StageConfig, List[str]]] = []
        self._checkpoint_manager: Optional[Any] = None
        self._max_workers = 1

    def with_name(self, name: str) -> "PipelineBuilder":
        """Set pipeline name.

        Args:
            name: Pipeline name.

        Returns:
            Self for method chaining.
        """
        self._name = name
        return self

    def with_checkpoint_manager(self, manager: Any) -> "PipelineBuilder":
        """Set checkpoint manager.

        Args:
            manager: Checkpoint manager instance.

        Returns:
            Self for method chaining.
        """
        self._checkpoint_manager = manager
        return self

    def with_max_workers(self, workers: int) -> "PipelineBuilder":
        """Set maximum concurrent workers.

        Args:
            workers: Number of workers.

        Returns:
            Self for method chaining.
        """
        self._max_workers = workers
        return self

    def add_stage(
        self,
        name: str,
        processor: ProcessorType,
        config: Optional[StageConfig] = None,
        dependencies: Optional[List[str]] = None,
    ) -> "PipelineBuilder":
        """Add a custom stage.

        Args:
            name: Stage name.
            processor: Processor to use.
            config: Stage configuration.
            dependencies: Stage dependencies.

        Returns:
            Self for method chaining.
        """
        self._stages.append((
            name,
            processor,
            config or StageConfig(),
            dependencies or [],
        ))
        return self

    def add_analysis(
        self,
        sample_rate: int = 10,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add video analysis stage.

        Args:
            sample_rate: Analyze every Nth frame.
            **kwargs: Additional analysis parameters.

        Returns:
            Self for method chaining.
        """

        def analyze(input_path: Path, output_path: Path, **params: Any) -> Dict[str, Any]:
            """Placeholder analysis processor."""
            return {"analyzed": True, "input": str(input_path)}

        config = StageConfig(params={"sample_rate": sample_rate, **kwargs})
        self._stages.append(("analysis", analyze, config, []))
        return self

    def add_denoising(
        self,
        strength: float = 0.5,
        method: str = "temporal",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add denoising stage.

        Args:
            strength: Denoising strength (0.0-1.0).
            method: Denoising method.
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """

        def denoise(input_path: Path, output_path: Path, **params: Any) -> bool:
            """Placeholder denoising processor."""
            return True

        config = StageConfig(params={"strength": strength, "method": method, **kwargs})
        self._stages.append(("denoising", denoise, config, ["analysis"]))
        return self

    def add_face_restoration(
        self,
        model: str = "gfpgan",
        strength: float = 0.8,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add face restoration stage.

        Args:
            model: Face restoration model.
            strength: Restoration strength.
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """

        def restore_faces(input_path: Path, output_path: Path, **params: Any) -> bool:
            """Placeholder face restoration processor."""
            return True

        config = StageConfig(params={"model": model, "strength": strength, **kwargs})
        self._stages.append(("face_restoration", restore_faces, config, ["denoising"]))
        return self

    def add_upscaling(
        self,
        scale: int = 4,
        model: str = "realesrgan",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add upscaling stage.

        Args:
            scale: Upscaling factor (2 or 4).
            model: Upscaling model.
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """

        def upscale(input_path: Path, output_path: Path, **params: Any) -> bool:
            """Placeholder upscaling processor."""
            return True

        config = StageConfig(
            params={"scale": scale, "model": model, **kwargs},
            checkpoint_after=True,
        )
        deps = []
        if any(s[0] == "face_restoration" for s in self._stages):
            deps.append("face_restoration")
        elif any(s[0] == "denoising" for s in self._stages):
            deps.append("denoising")
        self._stages.append(("upscaling", upscale, config, deps))
        return self

    def add_colorization(
        self,
        model: str = "deoldify",
        saturation: float = 1.0,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add colorization stage.

        Args:
            model: Colorization model.
            saturation: Color saturation.
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """

        def colorize(input_path: Path, output_path: Path, **params: Any) -> bool:
            """Placeholder colorization processor."""
            return True

        config = StageConfig(params={"model": model, "saturation": saturation, **kwargs})
        self._stages.append(("colorization", colorize, config, ["analysis"]))
        return self

    def add_interpolation(
        self,
        target_fps: float = 60.0,
        model: str = "rife",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add frame interpolation stage.

        Args:
            target_fps: Target frame rate.
            model: Interpolation model.
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """

        def interpolate(input_path: Path, output_path: Path, **params: Any) -> bool:
            """Placeholder interpolation processor."""
            return True

        config = StageConfig(params={"target_fps": target_fps, "model": model, **kwargs})
        # Interpolation typically comes after upscaling
        deps = []
        if any(s[0] == "upscaling" for s in self._stages):
            deps.append("upscaling")
        self._stages.append(("interpolation", interpolate, config, deps))
        return self

    def add_audio_enhancement(
        self,
        denoise: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add audio enhancement stage.

        Args:
            denoise: Apply audio denoising.
            normalize: Normalize audio levels.
            **kwargs: Additional parameters.

        Returns:
            Self for method chaining.
        """

        def enhance_audio(input_path: Path, output_path: Path, **params: Any) -> bool:
            """Placeholder audio enhancement processor."""
            return True

        config = StageConfig(params={"denoise": denoise, "normalize": normalize, **kwargs})
        self._stages.append(("audio_enhancement", enhance_audio, config, []))
        return self

    def build(self) -> Pipeline:
        """Build the pipeline.

        Returns:
            Configured Pipeline instance.
        """
        pipeline = Pipeline(
            name=self._name,
            max_workers=self._max_workers,
            checkpoint_manager=self._checkpoint_manager,
        )

        for name, processor, config, dependencies in self._stages:
            pipeline.add_stage(
                processor=processor,
                config=config,
                name=name,
                dependencies=dependencies,
            )

        return pipeline


# =============================================================================
# Convenience Functions
# =============================================================================


def create_restoration_pipeline(
    preset: str = "balanced",
    checkpoint_manager: Optional[Any] = None,
) -> Pipeline:
    """Create a standard restoration pipeline.

    Args:
        preset: Preset name ("fast", "balanced", "quality").
        checkpoint_manager: Optional checkpoint manager.

    Returns:
        Configured Pipeline.
    """
    builder = PipelineBuilder().with_name(f"restoration_{preset}")

    if checkpoint_manager:
        builder.with_checkpoint_manager(checkpoint_manager)

    # Add stages based on preset
    builder.add_analysis()

    if preset in ("balanced", "quality"):
        builder.add_denoising(strength=0.3 if preset == "balanced" else 0.5)

    if preset == "quality":
        builder.add_face_restoration()

    builder.add_upscaling(scale=4 if preset == "quality" else 2)

    if preset == "quality":
        builder.add_audio_enhancement()

    return builder.build()


def create_simple_pipeline(
    processors: List[ProcessorType],
) -> Pipeline:
    """Create a simple sequential pipeline.

    Args:
        processors: List of processors to execute in order.

    Returns:
        Configured Pipeline.
    """
    pipeline = Pipeline(name="simple")

    prev_name: Optional[str] = None
    for i, processor in enumerate(processors):
        name = f"stage_{i}"
        deps = [prev_name] if prev_name else []
        pipeline.add_stage(processor, name=name, dependencies=deps)
        prev_name = name

    return pipeline
