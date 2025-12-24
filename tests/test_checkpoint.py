"""Tests for the checkpoint module."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from framewright.checkpoint import (
    CheckpointManager,
    PipelineCheckpoint,
    FrameCheckpoint,
)


class TestFrameCheckpoint:
    """Tests for FrameCheckpoint dataclass."""

    def test_create_frame_checkpoint(self):
        """Test creating a frame checkpoint."""
        checkpoint = FrameCheckpoint(
            frame_number=1,
            input_path="/path/to/frame_00000001.png",
            output_path="/path/to/enhanced/frame_00000001.png",
            checksum="abc123",
            processed=True,
        )

        assert checkpoint.frame_number == 1
        assert checkpoint.processed is True
        assert checkpoint.checksum == "abc123"

    def test_frame_checkpoint_defaults(self):
        """Test default values for frame checkpoint."""
        checkpoint = FrameCheckpoint(
            frame_number=5,
            input_path="/path/to/frame.png",
        )

        assert checkpoint.output_path is None
        assert checkpoint.checksum is None
        assert checkpoint.processed is False


class TestPipelineCheckpoint:
    """Tests for PipelineCheckpoint dataclass."""

    def test_create_pipeline_checkpoint(self):
        """Test creating a pipeline checkpoint."""
        checkpoint = PipelineCheckpoint(
            stage="enhance",
            last_completed_frame=50,
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        assert checkpoint.stage == "enhance"
        assert checkpoint.last_completed_frame == 50
        assert checkpoint.total_frames == 100

    def test_to_dict(self):
        """Test converting checkpoint to dictionary."""
        checkpoint = PipelineCheckpoint(
            stage="extract",
            last_completed_frame=0,
            total_frames=1000,
            source_path="/path/to/video.mp4",
            metadata={"width": 1920, "height": 1080},
        )

        data = checkpoint.to_dict()

        assert data["stage"] == "extract"
        assert data["total_frames"] == 1000
        assert data["metadata"]["width"] == 1920

    def test_from_dict(self):
        """Test creating checkpoint from dictionary."""
        data = {
            "stage": "enhance",
            "last_completed_frame": 25,
            "total_frames": 50,
            "source_path": "/path/to/video.mp4",
            "metadata": {},
            "frames": [],
            "checkpoint_interval": 100,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config_hash": "abc123",
        }

        checkpoint = PipelineCheckpoint.from_dict(data)

        assert checkpoint.stage == "enhance"
        assert checkpoint.last_completed_frame == 25


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        return project_dir

    @pytest.fixture
    def manager(self, temp_project_dir):
        """Create a checkpoint manager for testing."""
        return CheckpointManager(
            project_dir=temp_project_dir,
            checkpoint_interval=10,
            config_hash="testhash",
        )

    def test_init(self, manager, temp_project_dir):
        """Test checkpoint manager initialization."""
        assert manager.project_dir == temp_project_dir
        assert manager.checkpoint_interval == 10
        assert manager.config_hash == "testhash"

    def test_has_checkpoint_false(self, manager):
        """Test has_checkpoint returns False when no checkpoint exists."""
        assert manager.has_checkpoint() is False

    def test_create_checkpoint(self, manager):
        """Test creating a new checkpoint."""
        checkpoint = manager.create_checkpoint(
            stage="extract",
            total_frames=100,
            source_path="/path/to/video.mp4",
            metadata={"width": 1920},
        )

        assert checkpoint.stage == "extract"
        assert checkpoint.total_frames == 100
        assert manager.has_checkpoint() is True

    def test_load_checkpoint(self, manager):
        """Test loading an existing checkpoint."""
        # Create checkpoint
        manager.create_checkpoint(
            stage="enhance",
            total_frames=50,
            source_path="/path/to/video.mp4",
        )

        # Create new manager and load
        new_manager = CheckpointManager(
            project_dir=manager.project_dir,
            checkpoint_interval=10,
        )

        loaded = new_manager.load_checkpoint()

        assert loaded is not None
        assert loaded.stage == "enhance"
        assert loaded.total_frames == 50

    def test_update_frame(self, manager):
        """Test updating checkpoint with frame progress."""
        manager.create_checkpoint(
            stage="enhance",
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        manager.update_frame(
            frame_number=5,
            input_path=Path("/path/to/frame_00000005.png"),
            output_path=Path("/path/to/enhanced/frame_00000005.png"),
            checksum="abc123",
        )

        # Verify update
        loaded = manager.load_checkpoint()
        assert loaded.last_completed_frame == 5
        assert len(loaded.frames) == 1

    def test_update_stage(self, manager):
        """Test updating the processing stage."""
        manager.create_checkpoint(
            stage="extract",
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        manager.update_stage("enhance")

        loaded = manager.load_checkpoint()
        assert loaded.stage == "enhance"

    def test_get_resume_frame(self, manager):
        """Test getting the frame to resume from."""
        manager.create_checkpoint(
            stage="enhance",
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        # Update to frame 50
        for i in range(50):
            manager.update_frame(
                frame_number=i,
                input_path=Path(f"/path/to/frame_{i:08d}.png"),
            )
        manager.force_save()

        resume_frame = manager.get_resume_frame()
        assert resume_frame == 50

    def test_get_unprocessed_frames(self, manager, temp_project_dir):
        """Test getting list of unprocessed frames."""
        # Create some frame files
        frames_dir = temp_project_dir / "frames"
        frames_dir.mkdir()

        all_frames = []
        for i in range(10):
            frame = frames_dir / f"frame_{i:08d}.png"
            frame.touch()
            all_frames.append(frame)

        manager.create_checkpoint(
            stage="enhance",
            total_frames=10,
            source_path="/path/to/video.mp4",
        )

        # Mark first 5 as processed
        for i in range(5):
            manager.update_frame(
                frame_number=i,
                input_path=all_frames[i],
                output_path=all_frames[i],
            )
            manager._checkpoint.frames[-1].processed = True
        manager.force_save()

        unprocessed = manager.get_unprocessed_frames(all_frames)
        assert len(unprocessed) == 5

    def test_clear_checkpoint(self, manager):
        """Test clearing the checkpoint."""
        manager.create_checkpoint(
            stage="extract",
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        assert manager.has_checkpoint() is True

        manager.clear_checkpoint()

        assert manager.has_checkpoint() is False

    def test_complete(self, manager):
        """Test marking pipeline as complete."""
        manager.create_checkpoint(
            stage="enhance",
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        manager.complete()

        loaded = manager.load_checkpoint()
        assert loaded.stage == "complete"

    def test_compute_file_checksum(self, temp_project_dir):
        """Test file checksum computation."""
        test_file = temp_project_dir / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = CheckpointManager.compute_file_checksum(test_file)

        assert len(checksum) == 64  # SHA256 hex length
        assert checksum == CheckpointManager.compute_file_checksum(test_file)  # Deterministic

    def test_compute_config_hash(self):
        """Test configuration hash computation."""
        config1 = {"scale": 4, "model": "realesrgan-x4plus"}
        config2 = {"model": "realesrgan-x4plus", "scale": 4}  # Same, different order
        config3 = {"scale": 2, "model": "realesrgan-x2plus"}

        hash1 = CheckpointManager.compute_config_hash(config1)
        hash2 = CheckpointManager.compute_config_hash(config2)
        hash3 = CheckpointManager.compute_config_hash(config3)

        assert hash1 == hash2  # Same config, different order
        assert hash1 != hash3  # Different config

    def test_periodic_save(self, manager):
        """Test that checkpoints are saved periodically."""
        manager.create_checkpoint(
            stage="enhance",
            total_frames=100,
            source_path="/path/to/video.mp4",
        )

        # Update frames up to checkpoint interval
        for i in range(manager.checkpoint_interval):
            manager.update_frame(
                frame_number=i,
                input_path=Path(f"/path/to/frame_{i:08d}.png"),
            )

        # Checkpoint should have been saved
        loaded = manager.load_checkpoint()
        assert loaded.last_completed_frame >= manager.checkpoint_interval - 1
