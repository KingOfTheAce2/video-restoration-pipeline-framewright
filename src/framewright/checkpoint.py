"""Checkpointing and recovery module for the FrameWright pipeline.

Provides frame-level checkpointing with resume capability for crash recovery.
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameCheckpoint:
    """Checkpoint data for a single frame."""
    frame_number: int
    input_path: str
    output_path: Optional[str] = None
    checksum: Optional[str] = None
    processed: bool = False
    timestamp: Optional[str] = None


@dataclass
class PipelineCheckpoint:
    """Complete pipeline checkpoint state.

    Attributes:
        stage: Current processing stage
        last_completed_frame: Last successfully processed frame number
        total_frames: Total number of frames to process
        source_path: Path to the source video
        metadata: Video metadata dictionary
        frames: List of frame checkpoint data
        checkpoint_interval: How often to save checkpoints
        created_at: When the checkpoint was first created
        updated_at: Last update timestamp
        config_hash: Hash of the configuration for validation
    """
    stage: Literal["download", "extract", "enhance", "interpolate", "reassemble", "complete"]
    last_completed_frame: int
    total_frames: int
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    frames: List[FrameCheckpoint] = field(default_factory=list)
    checkpoint_interval: int = 100
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for JSON serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineCheckpoint":
        """Create checkpoint from dictionary."""
        frames_data = data.pop("frames", [])
        frames = [
            FrameCheckpoint(**f) if isinstance(f, dict) else f
            for f in frames_data
        ]
        return cls(frames=frames, **data)


class CheckpointManager:
    """Manages pipeline checkpoints for crash recovery.

    Saves checkpoint state to disk periodically and provides resume capability.
    """

    CHECKPOINT_FILENAME = "checkpoint.json"
    CHECKPOINT_DIR = ".framewright"

    def __init__(
        self,
        project_dir: Path,
        checkpoint_interval: int = 100,
        config_hash: Optional[str] = None
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            project_dir: Root project directory
            checkpoint_interval: Save checkpoint every N frames
            config_hash: Hash of current configuration for validation
        """
        self.project_dir = Path(project_dir)
        self.checkpoint_dir = self.project_dir / self.CHECKPOINT_DIR
        self.checkpoint_file = self.checkpoint_dir / self.CHECKPOINT_FILENAME
        self.checkpoint_interval = checkpoint_interval
        self.config_hash = config_hash or ""
        self._checkpoint: Optional[PipelineCheckpoint] = None
        self._frames_since_save = 0

    def _ensure_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def has_checkpoint(self) -> bool:
        """Check if a valid checkpoint exists."""
        return self.checkpoint_file.exists()

    def load_checkpoint(self) -> Optional[PipelineCheckpoint]:
        """Load existing checkpoint from disk.

        Returns:
            PipelineCheckpoint if valid checkpoint exists, None otherwise.
        """
        if not self.has_checkpoint():
            return None

        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)

            checkpoint = PipelineCheckpoint.from_dict(data)

            # Validate config hash if provided
            if self.config_hash and checkpoint.config_hash != self.config_hash:
                logger.warning(
                    "Configuration has changed since checkpoint. "
                    "Resume may produce inconsistent results."
                )

            self._checkpoint = checkpoint
            logger.info(
                f"Loaded checkpoint: stage={checkpoint.stage}, "
                f"frame={checkpoint.last_completed_frame}/{checkpoint.total_frames}"
            )
            return checkpoint

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def create_checkpoint(
        self,
        stage: str,
        total_frames: int,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineCheckpoint:
        """Create a new checkpoint.

        Args:
            stage: Current processing stage
            total_frames: Total number of frames to process
            source_path: Path to source video
            metadata: Video metadata

        Returns:
            New PipelineCheckpoint instance
        """
        self._ensure_checkpoint_dir()

        self._checkpoint = PipelineCheckpoint(
            stage=stage,
            last_completed_frame=0,
            total_frames=total_frames,
            source_path=source_path,
            metadata=metadata or {},
            checkpoint_interval=self.checkpoint_interval,
            config_hash=self.config_hash,
        )

        self._save_checkpoint()
        logger.info(f"Created new checkpoint for {total_frames} frames")
        return self._checkpoint

    def update_frame(
        self,
        frame_number: int,
        input_path: Path,
        output_path: Optional[Path] = None,
        checksum: Optional[str] = None
    ) -> None:
        """Update checkpoint with processed frame.

        Args:
            frame_number: Frame number that was processed
            input_path: Path to input frame
            output_path: Path to output frame (if enhancement complete)
            checksum: SHA256 checksum of output frame
        """
        if self._checkpoint is None:
            raise RuntimeError("No checkpoint active. Call create_checkpoint first.")

        frame_data = FrameCheckpoint(
            frame_number=frame_number,
            input_path=str(input_path),
            output_path=str(output_path) if output_path else None,
            checksum=checksum,
            processed=output_path is not None,
            timestamp=datetime.now().isoformat()
        )

        # Update or append frame data
        existing_idx = next(
            (i for i, f in enumerate(self._checkpoint.frames)
             if f.frame_number == frame_number),
            None
        )

        if existing_idx is not None:
            self._checkpoint.frames[existing_idx] = frame_data
        else:
            self._checkpoint.frames.append(frame_data)

        self._checkpoint.last_completed_frame = frame_number
        self._checkpoint.updated_at = datetime.now().isoformat()

        self._frames_since_save += 1

        # Save periodically
        if self._frames_since_save >= self.checkpoint_interval:
            self._save_checkpoint()
            self._frames_since_save = 0

    def update_stage(self, stage: str) -> None:
        """Update the current processing stage.

        Args:
            stage: New stage name
        """
        if self._checkpoint is None:
            raise RuntimeError("No checkpoint active.")

        self._checkpoint.stage = stage
        self._checkpoint.updated_at = datetime.now().isoformat()
        self._save_checkpoint()
        logger.info(f"Stage updated to: {stage}")

    def _save_checkpoint(self) -> None:
        """Save current checkpoint to disk."""
        if self._checkpoint is None:
            return

        self._ensure_checkpoint_dir()

        # Write to temporary file first, then rename (atomic operation)
        temp_file = self.checkpoint_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                json.dump(self._checkpoint.to_dict(), f, indent=2)

            temp_file.replace(self.checkpoint_file)
            logger.debug(f"Checkpoint saved: frame {self._checkpoint.last_completed_frame}")

        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def force_save(self) -> None:
        """Force immediate checkpoint save."""
        self._save_checkpoint()
        self._frames_since_save = 0

    def get_resume_frame(self) -> int:
        """Get the frame number to resume from.

        Returns:
            Frame number to start processing from (0 if no checkpoint)
        """
        if self._checkpoint is None:
            return 0
        return self._checkpoint.last_completed_frame + 1

    def get_unprocessed_frames(self, all_frames: List[Path]) -> List[Path]:
        """Get list of frames that haven't been processed yet.

        Args:
            all_frames: List of all frame paths

        Returns:
            List of unprocessed frame paths
        """
        if self._checkpoint is None:
            return all_frames

        processed_numbers = {
            f.frame_number for f in self._checkpoint.frames if f.processed
        }

        unprocessed = []
        for frame_path in all_frames:
            # Extract frame number from filename (e.g., "frame_00000001.png" -> 1)
            try:
                frame_num = int(frame_path.stem.split("_")[-1])
                if frame_num not in processed_numbers:
                    unprocessed.append(frame_path)
            except (ValueError, IndexError):
                # If we can't parse the frame number, include it
                unprocessed.append(frame_path)

        return unprocessed

    def validate_checkpoint(self, frames_dir: Path, enhanced_dir: Path) -> bool:
        """Validate checkpoint integrity against actual files.

        Args:
            frames_dir: Directory containing source frames
            enhanced_dir: Directory containing enhanced frames

        Returns:
            True if checkpoint is valid, False otherwise
        """
        if self._checkpoint is None:
            return False

        # Check that processed frames actually exist
        for frame in self._checkpoint.frames:
            if frame.processed and frame.output_path:
                output_path = Path(frame.output_path)
                if not output_path.exists():
                    logger.warning(
                        f"Checkpoint frame {frame.frame_number} marked as processed "
                        f"but output file missing: {output_path}"
                    )
                    return False

                # Optionally verify checksum
                if frame.checksum:
                    actual_checksum = self.compute_file_checksum(output_path)
                    if actual_checksum != frame.checksum:
                        logger.warning(
                            f"Checksum mismatch for frame {frame.frame_number}"
                        )
                        return False

        return True

    def clear_checkpoint(self) -> None:
        """Remove the checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")
        self._checkpoint = None
        self._frames_since_save = 0

    def complete(self) -> None:
        """Mark the pipeline as complete and optionally keep checkpoint for reference."""
        if self._checkpoint is not None:
            self._checkpoint.stage = "complete"
            self._save_checkpoint()
            logger.info("Pipeline marked as complete")

    @staticmethod
    def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
        """Compute checksum of a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use

        Returns:
            Hex digest of file hash
        """
        hasher = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    @staticmethod
    def compute_config_hash(config_dict: Dict[str, Any]) -> str:
        """Compute hash of configuration for change detection.

        Args:
            config_dict: Configuration dictionary

        Returns:
            SHA256 hash of configuration
        """
        # Sort keys for consistent hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
