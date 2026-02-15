"""Checkpoint management for crash recovery and resume support.

This module provides checkpoint functionality for the engine pipeline,
allowing jobs to be resumed after interruption or crash. Key features:

- CheckpointManager: Save and load checkpoint state
- CheckpointState: Dataclass representing checkpoint data
- Auto-checkpoint at configurable intervals
- Atomic file operations for safety
- Compression support for large states

Example:
    >>> manager = CheckpointManager(checkpoint_dir=Path("./checkpoints"))
    >>> manager.save("job_123", CheckpointState(
    ...     job_id="job_123",
    ...     frame_index=1500,
    ...     stage_index=2,
    ...     processor_states={"denoiser": {...}},
    ... ))
    >>> if manager.exists("job_123"):
    ...     state = manager.load("job_123")
    ...     print(f"Resume from frame {state.frame_index}")
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint Data Classes
# =============================================================================


@dataclass
class ProcessorState:
    """State of a single processor for checkpointing.

    Attributes:
        name: Processor name.
        completed: Whether processor completed.
        last_frame: Last frame processed.
        internal_state: Processor-specific internal state.
        config: Processor configuration used.
    """

    name: str
    completed: bool = False
    last_frame: int = 0
    internal_state: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessorState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointState:
    """Complete checkpoint state for a job.

    Attributes:
        job_id: Unique job identifier.
        frame_index: Current frame index (0-based).
        stage_index: Current stage index in pipeline.
        stage_name: Name of current stage.
        total_frames: Total frames to process.
        total_stages: Total stages in pipeline.
        processor_states: State for each processor.
        shared_data: Data shared between stages.
        input_path: Original input file path.
        output_path: Target output file path.
        temp_dir: Temporary working directory.
        timestamp: When checkpoint was created.
        version: Checkpoint format version.
        metadata: Additional metadata.
    """

    job_id: str
    frame_index: int = 0
    stage_index: int = 0
    stage_name: str = ""
    total_frames: int = 0
    total_stages: int = 0
    processor_states: Dict[str, ProcessorState] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    input_path: str = ""
    output_path: str = ""
    temp_dir: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "frame_index": self.frame_index,
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "total_frames": self.total_frames,
            "total_stages": self.total_stages,
            "processor_states": {
                name: state.to_dict()
                for name, state in self.processor_states.items()
            },
            "shared_data": self._serialize_shared_data(),
            "input_path": self.input_path,
            "output_path": self.output_path,
            "temp_dir": self.temp_dir,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }

    def _serialize_shared_data(self) -> Dict[str, Any]:
        """Serialize shared data, handling non-JSON types."""
        result = {}
        for key, value in self.shared_data.items():
            try:
                # Try JSON serialization
                json.dumps(value)
                result[key] = {"type": "json", "data": value}
            except (TypeError, ValueError):
                # Fall back to pickle for complex objects
                import base64
                pickled = pickle.dumps(value)
                result[key] = {
                    "type": "pickle",
                    "data": base64.b64encode(pickled).decode("ascii"),
                }
        return result

    @classmethod
    def _deserialize_shared_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize shared data."""
        result = {}
        for key, value in data.items():
            if isinstance(value, dict) and "type" in value:
                if value["type"] == "json":
                    result[key] = value["data"]
                elif value["type"] == "pickle":
                    import base64
                    pickled = base64.b64decode(value["data"])
                    result[key] = pickle.loads(pickled)
                else:
                    result[key] = value
            else:
                # Legacy format
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        processor_states = {
            name: ProcessorState.from_dict(state)
            for name, state in data.get("processor_states", {}).items()
        }

        shared_data = cls._deserialize_shared_data(data.get("shared_data", {}))

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            job_id=data["job_id"],
            frame_index=data.get("frame_index", 0),
            stage_index=data.get("stage_index", 0),
            stage_name=data.get("stage_name", ""),
            total_frames=data.get("total_frames", 0),
            total_stages=data.get("total_stages", 0),
            processor_states=processor_states,
            shared_data=shared_data,
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            temp_dir=data.get("temp_dir", ""),
            timestamp=timestamp,
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
        )

    @property
    def progress(self) -> float:
        """Calculate overall progress (0.0-1.0)."""
        if self.total_frames == 0:
            return 0.0
        return self.frame_index / self.total_frames

    @property
    def stage_progress(self) -> float:
        """Calculate stage progress (0.0-1.0)."""
        if self.total_stages == 0:
            return 0.0
        return self.stage_index / self.total_stages

    def get_processor_state(self, name: str) -> Optional[ProcessorState]:
        """Get state for a specific processor.

        Args:
            name: Processor name.

        Returns:
            ProcessorState or None if not found.
        """
        return self.processor_states.get(name)

    def update_processor_state(
        self,
        name: str,
        completed: bool = False,
        last_frame: int = 0,
        internal_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update state for a processor.

        Args:
            name: Processor name.
            completed: Whether processor completed.
            last_frame: Last frame processed.
            internal_state: Processor internal state.
            config: Processor configuration.
        """
        if name in self.processor_states:
            state = self.processor_states[name]
            state.completed = completed
            state.last_frame = last_frame
            if internal_state is not None:
                state.internal_state = internal_state
            if config is not None:
                state.config = config
        else:
            self.processor_states[name] = ProcessorState(
                name=name,
                completed=completed,
                last_frame=last_frame,
                internal_state=internal_state or {},
                config=config or {},
            )


@dataclass
class CheckpointInfo:
    """Summary information about a checkpoint.

    Attributes:
        job_id: Job identifier.
        frame_index: Current frame index.
        stage_index: Current stage index.
        stage_name: Current stage name.
        progress: Overall progress (0.0-1.0).
        timestamp: When checkpoint was created.
        file_size: Checkpoint file size in bytes.
        file_path: Path to checkpoint file.
    """

    job_id: str
    frame_index: int
    stage_index: int
    stage_name: str
    progress: float
    timestamp: datetime
    file_size: int
    file_path: Path


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manages checkpoint save/load operations.

    The CheckpointManager provides:
    - Save checkpoint state to disk
    - Load checkpoint state from disk
    - List available checkpoints
    - Auto-checkpoint at intervals
    - Atomic file operations
    - Optional compression

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir=Path("./checkpoints"),
        ...     auto_interval=100,  # Auto-checkpoint every 100 frames
        ...     compress=True,
        ... )
        >>> state = CheckpointState(job_id="job_123", frame_index=500)
        >>> manager.save("job_123", state)
        >>> loaded = manager.load("job_123")
        >>> print(loaded.frame_index)
        500
    """

    CHECKPOINT_EXTENSION = ".ckpt"
    COMPRESSED_EXTENSION = ".ckpt.gz"

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        auto_interval: int = 0,
        compress: bool = False,
        keep_history: int = 3,
        on_save: Optional[Callable[[str, CheckpointState], None]] = None,
        on_load: Optional[Callable[[str, CheckpointState], None]] = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
            auto_interval: Frames between auto-checkpoints (0 to disable).
            compress: Compress checkpoint files with gzip.
            keep_history: Number of historical checkpoints to keep.
            on_save: Callback after saving checkpoint.
            on_load: Callback after loading checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir or Path.cwd() / ".checkpoints"
        self.auto_interval = auto_interval
        self.compress = compress
        self.keep_history = keep_history
        self.on_save = on_save
        self.on_load = on_load

        # Auto-checkpoint tracking
        self._last_checkpoint_frame: Dict[str, int] = {}
        self._lock = threading.Lock()

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def save(self, job_id: str, state: CheckpointState) -> Path:
        """Save a checkpoint state.

        Args:
            job_id: Job identifier.
            state: Checkpoint state to save.

        Returns:
            Path to saved checkpoint file.

        Raises:
            IOError: If save operation fails.
        """
        # Update timestamp
        state.timestamp = datetime.now()

        # Determine file path
        ext = self.COMPRESSED_EXTENSION if self.compress else self.CHECKPOINT_EXTENSION
        file_path = self.checkpoint_dir / f"{job_id}{ext}"

        # Serialize state
        data = state.to_dict()
        json_data = json.dumps(data, indent=2)

        # Write atomically using temp file
        with self._lock:
            try:
                # Create temp file in same directory for atomic rename
                fd, temp_path = tempfile.mkstemp(
                    suffix=ext,
                    dir=self.checkpoint_dir,
                )
                temp_path = Path(temp_path)

                try:
                    if self.compress:
                        with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                            f.write(json_data)
                    else:
                        with open(temp_path, "w", encoding="utf-8") as f:
                            f.write(json_data)

                    # Rotate history before replacing
                    self._rotate_history(job_id, file_path)

                    # Atomic rename
                    temp_path.replace(file_path)

                except Exception:
                    # Clean up temp file on error
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

                finally:
                    os.close(fd)

            except Exception as e:
                logger.error(f"Failed to save checkpoint for {job_id}: {e}")
                raise IOError(f"Checkpoint save failed: {e}") from e

        # Update tracking
        self._last_checkpoint_frame[job_id] = state.frame_index

        logger.debug(
            f"Checkpoint saved: {job_id} at frame {state.frame_index}"
        )

        # Callback
        if self.on_save:
            try:
                self.on_save(job_id, state)
            except Exception as e:
                logger.warning(f"Checkpoint save callback error: {e}")

        return file_path

    def load(self, job_id: str) -> Optional[CheckpointState]:
        """Load a checkpoint state.

        Args:
            job_id: Job identifier.

        Returns:
            CheckpointState or None if not found.

        Raises:
            IOError: If load operation fails.
        """
        # Try compressed first, then uncompressed
        compressed_path = self.checkpoint_dir / f"{job_id}{self.COMPRESSED_EXTENSION}"
        uncompressed_path = self.checkpoint_dir / f"{job_id}{self.CHECKPOINT_EXTENSION}"

        file_path = None
        if compressed_path.exists():
            file_path = compressed_path
        elif uncompressed_path.exists():
            file_path = uncompressed_path

        if file_path is None:
            return None

        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    json_data = f.read()
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = f.read()

            data = json.loads(json_data)
            state = CheckpointState.from_dict(data)

            logger.debug(
                f"Checkpoint loaded: {job_id} at frame {state.frame_index}"
            )

            # Callback
            if self.on_load:
                try:
                    self.on_load(job_id, state)
                except Exception as e:
                    logger.warning(f"Checkpoint load callback error: {e}")

            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint for {job_id}: {e}")
            raise IOError(f"Checkpoint load failed: {e}") from e

    def exists(self, job_id: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            job_id: Job identifier.

        Returns:
            True if checkpoint exists.
        """
        compressed_path = self.checkpoint_dir / f"{job_id}{self.COMPRESSED_EXTENSION}"
        uncompressed_path = self.checkpoint_dir / f"{job_id}{self.CHECKPOINT_EXTENSION}"
        return compressed_path.exists() or uncompressed_path.exists()

    def delete(self, job_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            job_id: Job identifier.

        Returns:
            True if checkpoint was deleted.
        """
        deleted = False

        for ext in (self.CHECKPOINT_EXTENSION, self.COMPRESSED_EXTENSION):
            file_path = self.checkpoint_dir / f"{job_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True

        # Also delete history files
        for history_file in self.checkpoint_dir.glob(f"{job_id}.*.ckpt*"):
            history_file.unlink()
            deleted = True

        if deleted:
            self._last_checkpoint_frame.pop(job_id, None)
            logger.debug(f"Checkpoint deleted: {job_id}")

        return deleted

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints.

        Returns:
            List of CheckpointInfo for each checkpoint.
        """
        checkpoints = []

        # Find all checkpoint files (not history files)
        patterns = [
            f"*{self.CHECKPOINT_EXTENSION}",
            f"*{self.COMPRESSED_EXTENSION}",
        ]

        seen_jobs: Set[str] = set()

        for pattern in patterns:
            for file_path in self.checkpoint_dir.glob(pattern):
                # Skip history files (contain timestamp)
                if "." in file_path.stem and file_path.stem.count(".") > 0:
                    parts = file_path.stem.rsplit(".", 1)
                    if parts[1].isdigit():
                        continue

                # Extract job ID
                job_id = file_path.stem
                if job_id.endswith(".ckpt"):
                    job_id = job_id[:-5]

                if job_id in seen_jobs:
                    continue
                seen_jobs.add(job_id)

                try:
                    # Load just enough to get info
                    state = self.load(job_id)
                    if state:
                        checkpoints.append(CheckpointInfo(
                            job_id=job_id,
                            frame_index=state.frame_index,
                            stage_index=state.stage_index,
                            stage_name=state.stage_name,
                            progress=state.progress,
                            timestamp=state.timestamp,
                            file_size=file_path.stat().st_size,
                            file_path=file_path,
                        ))
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {file_path}: {e}")

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)

        return checkpoints

    def get_info(self, job_id: str) -> Optional[CheckpointInfo]:
        """Get information about a specific checkpoint.

        Args:
            job_id: Job identifier.

        Returns:
            CheckpointInfo or None if not found.
        """
        for info in self.list_checkpoints():
            if info.job_id == job_id:
                return info
        return None

    # -------------------------------------------------------------------------
    # Auto-Checkpoint Support
    # -------------------------------------------------------------------------

    def should_checkpoint(self, job_id: str, current_frame: int) -> bool:
        """Check if auto-checkpoint should be triggered.

        Args:
            job_id: Job identifier.
            current_frame: Current frame index.

        Returns:
            True if checkpoint should be created.
        """
        if self.auto_interval <= 0:
            return False

        last_frame = self._last_checkpoint_frame.get(job_id, 0)
        return current_frame - last_frame >= self.auto_interval

    def auto_checkpoint(
        self,
        job_id: str,
        state: CheckpointState,
        force: bool = False,
    ) -> Optional[Path]:
        """Create checkpoint if auto-interval reached.

        Args:
            job_id: Job identifier.
            state: Current checkpoint state.
            force: Force checkpoint regardless of interval.

        Returns:
            Path to checkpoint file if created, None otherwise.
        """
        if force or self.should_checkpoint(job_id, state.frame_index):
            return self.save(job_id, state)
        return None

    # -------------------------------------------------------------------------
    # History Management
    # -------------------------------------------------------------------------

    def _rotate_history(self, job_id: str, current_path: Path) -> None:
        """Rotate checkpoint history files.

        Args:
            job_id: Job identifier.
            current_path: Current checkpoint file path.
        """
        if self.keep_history <= 0 or not current_path.exists():
            return

        # Get existing history files
        history_pattern = f"{job_id}.*{self.CHECKPOINT_EXTENSION}*"
        history_files = sorted(
            self.checkpoint_dir.glob(history_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Include current file in rotation
        all_files = [current_path] + history_files

        # Remove oldest files beyond keep_history
        for old_file in all_files[self.keep_history:]:
            try:
                old_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")

        # Rename current to history if it exists
        if current_path.exists():
            timestamp = int(time.time())
            ext = self.COMPRESSED_EXTENSION if self.compress else self.CHECKPOINT_EXTENSION
            history_path = self.checkpoint_dir / f"{job_id}.{timestamp}{ext}"
            try:
                shutil.copy2(current_path, history_path)
            except Exception as e:
                logger.warning(f"Failed to create checkpoint history: {e}")

    def list_history(self, job_id: str) -> List[CheckpointInfo]:
        """List checkpoint history for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of historical checkpoints (newest first).
        """
        history = []

        for ext in (self.CHECKPOINT_EXTENSION, self.COMPRESSED_EXTENSION):
            pattern = f"{job_id}.*{ext}"
            for file_path in self.checkpoint_dir.glob(pattern):
                # Extract timestamp from filename
                parts = file_path.stem.rsplit(".", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    timestamp = datetime.fromtimestamp(int(parts[1]))
                    try:
                        # Quick load to get frame info
                        if file_path.suffix == ".gz":
                            with gzip.open(file_path, "rt") as f:
                                data = json.loads(f.read())
                        else:
                            with open(file_path) as f:
                                data = json.load(f)

                        history.append(CheckpointInfo(
                            job_id=job_id,
                            frame_index=data.get("frame_index", 0),
                            stage_index=data.get("stage_index", 0),
                            stage_name=data.get("stage_name", ""),
                            progress=data.get("frame_index", 0) / max(1, data.get("total_frames", 1)),
                            timestamp=timestamp,
                            file_size=file_path.stat().st_size,
                            file_path=file_path,
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to read history file: {e}")

        history.sort(key=lambda c: c.timestamp, reverse=True)
        return history

    def restore_from_history(
        self,
        job_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[CheckpointState]:
        """Restore a checkpoint from history.

        Args:
            job_id: Job identifier.
            timestamp: Specific timestamp to restore (latest if None).

        Returns:
            CheckpointState or None if not found.
        """
        history = self.list_history(job_id)
        if not history:
            return None

        # Find matching or latest
        target = history[0]  # Latest by default
        if timestamp:
            for info in history:
                if info.timestamp == timestamp:
                    target = info
                    break

        # Load from history file
        try:
            if target.file_path.suffix == ".gz":
                with gzip.open(target.file_path, "rt") as f:
                    data = json.loads(f.read())
            else:
                with open(target.file_path) as f:
                    data = json.load(f)

            return CheckpointState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to restore from history: {e}")
            return None

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup_old(self, max_age_days: int = 7) -> int:
        """Remove checkpoints older than specified age.

        Args:
            max_age_days: Maximum age in days.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)

        for file_path in self.checkpoint_dir.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                try:
                    file_path.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint: {e}")

        if removed:
            logger.info(f"Removed {removed} old checkpoint files")

        return removed

    def cleanup_orphaned(self, active_jobs: Set[str]) -> int:
        """Remove checkpoints for jobs no longer active.

        Args:
            active_jobs: Set of active job IDs.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0

        for info in self.list_checkpoints():
            if info.job_id not in active_jobs:
                if self.delete(info.job_id):
                    removed += 1

        if removed:
            logger.info(f"Removed {removed} orphaned checkpoints")

        return removed

    def get_total_size(self) -> int:
        """Get total size of all checkpoint files.

        Returns:
            Total size in bytes.
        """
        total = 0
        for file_path in self.checkpoint_dir.glob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total


# =============================================================================
# Convenience Functions
# =============================================================================


def create_checkpoint_manager(
    checkpoint_dir: Optional[Union[str, Path]] = None,
    auto_interval: int = 100,
    compress: bool = True,
    keep_history: int = 3,
) -> CheckpointManager:
    """Create a checkpoint manager with common settings.

    Args:
        checkpoint_dir: Directory for checkpoints.
        auto_interval: Frames between auto-checkpoints.
        compress: Enable compression.
        keep_history: Number of history files to keep.

    Returns:
        Configured CheckpointManager.
    """
    return CheckpointManager(
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        auto_interval=auto_interval,
        compress=compress,
        keep_history=keep_history,
    )


def save_checkpoint(
    job_id: str,
    frame_index: int,
    stage_name: str,
    total_frames: int,
    checkpoint_dir: Optional[Path] = None,
    **metadata: Any,
) -> Path:
    """Quick save a checkpoint.

    Args:
        job_id: Job identifier.
        frame_index: Current frame index.
        stage_name: Current stage name.
        total_frames: Total frames to process.
        checkpoint_dir: Checkpoint directory.
        **metadata: Additional metadata.

    Returns:
        Path to checkpoint file.
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    state = CheckpointState(
        job_id=job_id,
        frame_index=frame_index,
        stage_name=stage_name,
        total_frames=total_frames,
        metadata=metadata,
    )
    return manager.save(job_id, state)


def load_checkpoint(
    job_id: str,
    checkpoint_dir: Optional[Path] = None,
) -> Optional[CheckpointState]:
    """Quick load a checkpoint.

    Args:
        job_id: Job identifier.
        checkpoint_dir: Checkpoint directory.

    Returns:
        CheckpointState or None if not found.
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.load(job_id)


def checkpoint_exists(
    job_id: str,
    checkpoint_dir: Optional[Path] = None,
) -> bool:
    """Check if a checkpoint exists.

    Args:
        job_id: Job identifier.
        checkpoint_dir: Checkpoint directory.

    Returns:
        True if checkpoint exists.
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.exists(job_id)
