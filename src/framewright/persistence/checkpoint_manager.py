"""Checkpoint Manager for True Resume Support.

.. deprecated::
    DEPRECATED: Legacy checkpoint manager. Use `framewright.engine.checkpoint` for new code.

Provides frame-level checkpointing for video restoration, allowing
interrupted jobs to resume exactly where they left off.

Features:
- Frame-level progress tracking
- Atomic checkpoint writes (crash-safe)
- Automatic checkpoint cleanup
- Resume from any point
- Multi-stage checkpoint support

Example:
    >>> manager = CheckpointManager(video_path, output_dir)
    >>> for frame_idx in manager.get_remaining_frames():
    ...     result = process_frame(frame_idx)
    ...     manager.save_frame(frame_idx, result)
    ...     manager.checkpoint()
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FrameCheckpoint:
    """Checkpoint data for a single frame."""
    frame_index: int
    stage: str
    output_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ProcessingCheckpoint:
    """Complete checkpoint state for a restoration job."""
    # Identification
    video_path: str
    video_hash: str  # For verifying same video
    job_id: str

    # Progress
    total_frames: int
    completed_frames: List[int] = field(default_factory=list)
    failed_frames: List[int] = field(default_factory=list)
    current_stage: str = "init"

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: float = field(default_factory=time.time)
    last_checkpoint_at: float = field(default_factory=time.time)
    processing_time_seconds: float = 0.0

    # Stage progress
    stages_completed: List[str] = field(default_factory=list)
    stage_frame_progress: Dict[str, List[int]] = field(default_factory=dict)

    # Metadata
    version: str = "1.0"
    hostname: str = ""
    gpu_name: str = ""

    @property
    def progress_percent(self) -> float:
        """Get overall progress percentage."""
        if self.total_frames == 0:
            return 0.0
        return len(self.completed_frames) / self.total_frames * 100

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return len(self.completed_frames) >= self.total_frames

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingCheckpoint":
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages checkpoints for video restoration jobs.

    Provides crash-safe checkpoint storage and automatic resume
    capability for long-running restoration jobs.
    """

    CHECKPOINT_FILENAME = "checkpoint.json"
    CHECKPOINT_BACKUP = "checkpoint.backup.json"
    FRAMES_DIR = "frames"

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        total_frames: int,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 100,  # Checkpoint every N frames
    ):
        """Initialize checkpoint manager.

        Args:
            video_path: Input video path
            output_dir: Output directory for checkpoints
            total_frames: Total number of frames
            config: Processing configuration
            checkpoint_interval: Frames between automatic checkpoints
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.total_frames = total_frames
        self.checkpoint_interval = checkpoint_interval

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / self.FRAMES_DIR
        self.frames_dir.mkdir(exist_ok=True)

        # Initialize or load checkpoint
        self.checkpoint = self._load_or_create_checkpoint(config)
        self._frames_since_checkpoint = 0

    def _compute_video_hash(self) -> str:
        """Compute hash of video file for verification."""
        import hashlib

        hasher = hashlib.md5()
        try:
            with open(self.video_path, 'rb') as f:
                # Read first and last 1MB for fast hashing
                hasher.update(f.read(1024 * 1024))
                f.seek(-1024 * 1024, 2)
                hasher.update(f.read())
            return hasher.hexdigest()[:16]
        except Exception:
            return "unknown"

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        import uuid
        return f"{self.video_path.stem}_{uuid.uuid4().hex[:8]}"

    def _load_or_create_checkpoint(
        self,
        config: Optional[Dict[str, Any]],
    ) -> ProcessingCheckpoint:
        """Load existing checkpoint or create new one."""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILENAME

        if checkpoint_path.exists():
            try:
                checkpoint = self._load_checkpoint(checkpoint_path)

                # Verify it's for the same video
                current_hash = self._compute_video_hash()
                if checkpoint.video_hash == current_hash:
                    logger.info(
                        f"Resuming from checkpoint: {checkpoint.progress_percent:.1f}% complete "
                        f"({len(checkpoint.completed_frames)}/{checkpoint.total_frames} frames)"
                    )
                    return checkpoint
                else:
                    logger.warning("Video hash mismatch - starting fresh")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        # Create new checkpoint
        import socket

        checkpoint = ProcessingCheckpoint(
            video_path=str(self.video_path),
            video_hash=self._compute_video_hash(),
            job_id=self._generate_job_id(),
            total_frames=self.total_frames,
            config=config or {},
            hostname=socket.gethostname(),
        )

        # Try to get GPU name
        try:
            import torch
            if torch.cuda.is_available():
                checkpoint.gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        self._save_checkpoint(checkpoint)
        return checkpoint

    def _load_checkpoint(self, path: Path) -> ProcessingCheckpoint:
        """Load checkpoint from file."""
        with open(path) as f:
            data = json.load(f)
        return ProcessingCheckpoint.from_dict(data)

    def _save_checkpoint(self, checkpoint: ProcessingCheckpoint) -> None:
        """Save checkpoint atomically."""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILENAME
        backup_path = self.output_dir / self.CHECKPOINT_BACKUP
        temp_path = checkpoint_path.with_suffix('.tmp')

        checkpoint.last_checkpoint_at = time.time()

        # Write to temp file first
        with open(temp_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Backup existing checkpoint
        if checkpoint_path.exists():
            shutil.copy2(checkpoint_path, backup_path)

        # Atomic rename
        temp_path.replace(checkpoint_path)

    def get_remaining_frames(self) -> Iterator[int]:
        """Get iterator of frames that still need processing.

        Yields:
            Frame indices that haven't been completed
        """
        completed = set(self.checkpoint.completed_frames)
        failed = set(self.checkpoint.failed_frames)

        for i in range(self.total_frames):
            if i not in completed and i not in failed:
                yield i

    def get_remaining_count(self) -> int:
        """Get count of remaining frames."""
        completed = set(self.checkpoint.completed_frames)
        failed = set(self.checkpoint.failed_frames)
        return self.total_frames - len(completed) - len(failed)

    def mark_frame_complete(
        self,
        frame_index: int,
        output_path: Optional[Path] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Mark a frame as completed.

        Args:
            frame_index: Frame index
            output_path: Path to output frame
            metrics: Quality metrics for frame
        """
        if frame_index not in self.checkpoint.completed_frames:
            self.checkpoint.completed_frames.append(frame_index)

        # Remove from failed if it was there
        if frame_index in self.checkpoint.failed_frames:
            self.checkpoint.failed_frames.remove(frame_index)

        # Update stage progress
        stage = self.checkpoint.current_stage
        if stage not in self.checkpoint.stage_frame_progress:
            self.checkpoint.stage_frame_progress[stage] = []
        if frame_index not in self.checkpoint.stage_frame_progress[stage]:
            self.checkpoint.stage_frame_progress[stage].append(frame_index)

        # Auto-checkpoint
        self._frames_since_checkpoint += 1
        if self._frames_since_checkpoint >= self.checkpoint_interval:
            self.save_checkpoint()

    def mark_frame_failed(
        self,
        frame_index: int,
        error: str,
    ) -> None:
        """Mark a frame as failed.

        Args:
            frame_index: Frame index
            error: Error message
        """
        if frame_index not in self.checkpoint.failed_frames:
            self.checkpoint.failed_frames.append(frame_index)

        logger.warning(f"Frame {frame_index} failed: {error}")

    def set_stage(self, stage: str) -> None:
        """Set current processing stage.

        Args:
            stage: Stage name
        """
        if self.checkpoint.current_stage != stage:
            if self.checkpoint.current_stage not in self.checkpoint.stages_completed:
                self.checkpoint.stages_completed.append(self.checkpoint.current_stage)
            self.checkpoint.current_stage = stage
            logger.info(f"Entered stage: {stage}")

    def save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        self.checkpoint.processing_time_seconds = (
            time.time() - self.checkpoint.started_at
        )
        self._save_checkpoint(self.checkpoint)
        self._frames_since_checkpoint = 0
        logger.debug(f"Checkpoint saved: {self.checkpoint.progress_percent:.1f}% complete")

    def get_frame_output_path(self, frame_index: int, extension: str = ".png") -> Path:
        """Get output path for a frame.

        Args:
            frame_index: Frame index
            extension: File extension

        Returns:
            Path for frame output
        """
        return self.frames_dir / f"frame_{frame_index:08d}{extension}"

    def cleanup(self, keep_frames: bool = False) -> None:
        """Clean up checkpoint files after completion.

        Args:
            keep_frames: Whether to keep processed frames
        """
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILENAME
        backup_path = self.output_dir / self.CHECKPOINT_BACKUP

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if backup_path.exists():
            backup_path.unlink()

        if not keep_frames and self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)

        logger.info("Checkpoint cleanup complete")

    def get_summary(self) -> Dict[str, Any]:
        """Get checkpoint summary.

        Returns:
            Summary dictionary
        """
        return {
            "job_id": self.checkpoint.job_id,
            "video": self.video_path.name,
            "total_frames": self.checkpoint.total_frames,
            "completed_frames": len(self.checkpoint.completed_frames),
            "failed_frames": len(self.checkpoint.failed_frames),
            "progress_percent": self.checkpoint.progress_percent,
            "current_stage": self.checkpoint.current_stage,
            "stages_completed": self.checkpoint.stages_completed,
            "processing_time": self.checkpoint.processing_time_seconds,
            "is_complete": self.checkpoint.is_complete,
        }


def get_checkpoint_info(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Get checkpoint info without loading full manager.

    Args:
        output_dir: Output directory

    Returns:
        Checkpoint info or None if not found
    """
    checkpoint_path = output_dir / CheckpointManager.CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)

        checkpoint = ProcessingCheckpoint.from_dict(data)
        return {
            "job_id": checkpoint.job_id,
            "video": Path(checkpoint.video_path).name,
            "progress_percent": checkpoint.progress_percent,
            "completed_frames": len(checkpoint.completed_frames),
            "total_frames": checkpoint.total_frames,
            "current_stage": checkpoint.current_stage,
            "last_checkpoint": datetime.fromtimestamp(
                checkpoint.last_checkpoint_at
            ).isoformat(),
            "is_complete": checkpoint.is_complete,
        }
    except Exception as e:
        logger.debug(f"Failed to read checkpoint: {e}")
        return None


def can_resume(output_dir: Path) -> bool:
    """Check if a job can be resumed.

    Args:
        output_dir: Output directory

    Returns:
        True if resumable checkpoint exists
    """
    info = get_checkpoint_info(output_dir)
    return info is not None and not info["is_complete"]
