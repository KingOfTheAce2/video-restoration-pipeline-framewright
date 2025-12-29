"""Unit tests for output directory manager."""
import pytest
from pathlib import Path

from framewright.utils.output_manager import (
    OutputManager,
    OutputPaths,
)


class TestOutputPaths:
    """Tests for OutputPaths dataclass."""

    def test_output_paths_creation(self, temp_dir):
        """Test creating OutputPaths instance."""
        paths = OutputPaths(
            project_dir=temp_dir / "project",
            frames_dir=temp_dir / "frames",
            enhanced_dir=temp_dir / "enhanced",
            interpolated_dir=temp_dir / "interpolated",
            colorized_dir=temp_dir / "colorized",
            watermark_removed_dir=temp_dir / "watermark_removed",
            output_dir=temp_dir / "output",
            temp_dir=temp_dir / "temp",
        )
        assert paths.project_dir == temp_dir / "project"
        assert paths.frames_dir == temp_dir / "frames"
        assert paths.enhanced_dir == temp_dir / "enhanced"
        assert paths.interpolated_dir == temp_dir / "interpolated"
        assert paths.colorized_dir == temp_dir / "colorized"
        assert paths.watermark_removed_dir == temp_dir / "watermark_removed"
        assert paths.output_dir == temp_dir / "output"
        assert paths.temp_dir == temp_dir / "temp"

    def test_path_conversion(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        paths = OutputPaths(
            project_dir=str(temp_dir / "project"),
            frames_dir=str(temp_dir / "frames"),
            enhanced_dir=str(temp_dir / "enhanced"),
            interpolated_dir=str(temp_dir / "interpolated"),
            colorized_dir=str(temp_dir / "colorized"),
            watermark_removed_dir=str(temp_dir / "watermark_removed"),
            output_dir=str(temp_dir / "output"),
            temp_dir=str(temp_dir / "temp"),
        )
        assert isinstance(paths.project_dir, Path)
        assert isinstance(paths.frames_dir, Path)


class TestOutputManager:
    """Tests for OutputManager."""

    def test_init_default(self, temp_dir):
        """Test default initialization."""
        manager = OutputManager(project_dir=temp_dir)
        assert manager.project_dir == temp_dir.resolve()

    def test_init_with_custom_output(self, temp_dir):
        """Test initialization with custom output directory."""
        output_dir = temp_dir / "custom_output"
        manager = OutputManager(project_dir=temp_dir, output_dir=output_dir)
        assert manager.output_dir == output_dir.resolve()

    def test_init_with_custom_subdirs(self, temp_dir):
        """Test initialization with custom subdirectory names."""
        manager = OutputManager(
            project_dir=temp_dir,
            frames_subdir="my_frames",
            enhanced_subdir="my_enhanced",
            output_subdir="my_output"
        )
        assert manager.frames_subdir == "my_frames"
        assert manager.enhanced_subdir == "my_enhanced"
        assert manager.output_subdir == "my_output"

    def test_create_directories(self, temp_dir):
        """Test that create_directories creates all directories."""
        manager = OutputManager(project_dir=temp_dir)
        manager.create_directories()
        paths = manager.get_paths()

        # All directories should exist
        assert paths.project_dir.exists()
        assert paths.frames_dir.exists()
        assert paths.enhanced_dir.exists()
        assert paths.output_dir.exists()
        assert paths.temp_dir.exists()

    def test_get_paths_returns_output_paths(self, temp_dir):
        """Test that get_paths returns OutputPaths."""
        manager = OutputManager(project_dir=temp_dir)
        paths = manager.get_paths()

        assert isinstance(paths, OutputPaths)

    def test_get_output_path(self, temp_dir):
        """Test getting output video path."""
        manager = OutputManager(project_dir=temp_dir)
        manager.create_directories()

        video_path = manager.get_output_path("restored_video.mp4")
        assert video_path.parent == manager.output_dir
        assert video_path.name == "restored_video.mp4"

    def test_cleanup_temp(self, temp_dir):
        """Test cleaning up temporary directory."""
        manager = OutputManager(project_dir=temp_dir)
        manager.create_directories()
        paths = manager.get_paths()

        # Create temp file
        temp_file = paths.temp_dir / "temp.txt"
        temp_file.write_text("temporary data")
        assert temp_file.exists()

        manager.cleanup_temp()
        assert not paths.temp_dir.exists()


class TestOutputManagerIntegration:
    """Integration tests for OutputManager."""

    def test_full_workflow(self, temp_dir):
        """Test complete output management workflow."""
        # Initialize
        manager = OutputManager(
            project_dir=temp_dir,
            output_dir=temp_dir / "final_output",
        )

        # Create directories
        manager.create_directories()
        paths = manager.get_paths()

        # Verify all directories exist
        assert paths.project_dir.exists()
        assert paths.frames_dir.exists()
        assert paths.enhanced_dir.exists()
        assert paths.output_dir.exists()
        assert paths.temp_dir.exists()

        # Get output path
        output_path = manager.get_output_path("restored_video.mp4")
        assert output_path.parent == manager.output_dir
        assert output_path.name == "restored_video.mp4"

        # Cleanup temp
        manager.cleanup_temp()
        assert not paths.temp_dir.exists()
