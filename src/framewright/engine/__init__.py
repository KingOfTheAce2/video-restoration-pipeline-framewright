"""Engine module for advanced video processing.

This module provides core processing engines for video restoration:

- **Pipeline**: Multi-stage pipeline orchestration with event-driven execution
- **Scheduler**: Job queue management with priority scheduling and persistence
- **Checkpoint**: Crash recovery and resume support for long-running jobs
- **Temporal Consistency**: Color/grain consistency for long-form videos

Example - Pipeline:
    >>> from framewright.engine import PipelineBuilder, Pipeline
    >>> pipeline = PipelineBuilder()
    ...     .add_analysis()
    ...     .add_denoising(strength=0.5)
    ...     .add_upscaling(scale=4)
    ...     .build()
    >>> result = pipeline.run("input.mp4", "output.mp4")

Example - Scheduler:
    >>> from framewright.engine import JobScheduler, Job, JobPriority
    >>> scheduler = JobScheduler(max_concurrent=2, auto_start=True)
    >>> job = Job(input="video.mp4", output="restored.mp4")
    >>> job_id = scheduler.submit(job)
    >>> scheduler.get_status(job_id)

Example - Checkpoint:
    >>> from framewright.engine import CheckpointManager, CheckpointState
    >>> manager = CheckpointManager(checkpoint_dir=Path("./checkpoints"))
    >>> state = CheckpointState(job_id="job_123", frame_index=1500)
    >>> manager.save("job_123", state)
    >>> restored = manager.load("job_123")
"""

# Pipeline orchestration
from framewright.engine.pipeline import (
    # Enums
    PipelineEventType,
    StageStatus,
    PipelineStatus,
    # Event system
    PipelineEvent,
    EventCallback,
    EventEmitter,
    # Stage configuration
    StageConfig,
    PipelineStage,
    StageResult,
    # Pipeline result
    PipelineResult,
    # Validation
    ValidationResult,
    PipelineValidator,
    # Context
    PipelineContext,
    # Main classes
    Pipeline,
    PipelineBuilder,
    # Convenience functions
    create_restoration_pipeline,
    create_simple_pipeline,
)

# Job scheduling
from framewright.engine.scheduler import (
    # Enums
    JobStatus,
    JobPriority,
    JobEventType,
    # Data classes
    JobConfig,
    JobProgress,
    Job,
    JobFilter,
    JobEvent,
    # Callbacks
    JobEventCallback,
    # Main class
    JobScheduler,
    # Convenience functions
    create_scheduler,
    create_job,
)

# Checkpoint management
from framewright.engine.checkpoint import (
    # Data classes
    ProcessorState,
    CheckpointState,
    CheckpointInfo,
    # Main class
    CheckpointManager,
    # Convenience functions
    create_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
    checkpoint_exists,
)

# Temporal consistency (existing)
from framewright.engine.temporal_consistency import (
    GlobalAnchors,
    LongFormConsistencyManager,
    ColorConsistencyEnforcer,
    ChunkedProcessor,
    TemporalConsistencyConfig,
    ConsistencyResult,
)

__all__ = [
    # Pipeline - Enums
    "PipelineEventType",
    "StageStatus",
    "PipelineStatus",
    # Pipeline - Event system
    "PipelineEvent",
    "EventCallback",
    "EventEmitter",
    # Pipeline - Stage configuration
    "StageConfig",
    "PipelineStage",
    "StageResult",
    # Pipeline - Result
    "PipelineResult",
    # Pipeline - Validation
    "ValidationResult",
    "PipelineValidator",
    # Pipeline - Context
    "PipelineContext",
    # Pipeline - Main classes
    "Pipeline",
    "PipelineBuilder",
    # Pipeline - Convenience functions
    "create_restoration_pipeline",
    "create_simple_pipeline",
    # Scheduler - Enums
    "JobStatus",
    "JobPriority",
    "JobEventType",
    # Scheduler - Data classes
    "JobConfig",
    "JobProgress",
    "Job",
    "JobFilter",
    "JobEvent",
    # Scheduler - Callbacks
    "JobEventCallback",
    # Scheduler - Main class
    "JobScheduler",
    # Scheduler - Convenience functions
    "create_scheduler",
    "create_job",
    # Checkpoint - Data classes
    "ProcessorState",
    "CheckpointState",
    "CheckpointInfo",
    # Checkpoint - Main class
    "CheckpointManager",
    # Checkpoint - Convenience functions
    "create_checkpoint_manager",
    "save_checkpoint",
    "load_checkpoint",
    "checkpoint_exists",
    # Temporal consistency (existing)
    "GlobalAnchors",
    "LongFormConsistencyManager",
    "ColorConsistencyEnforcer",
    "ChunkedProcessor",
    "TemporalConsistencyConfig",
    "ConsistencyResult",
]
