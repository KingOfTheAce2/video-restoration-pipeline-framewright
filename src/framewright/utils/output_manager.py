"""Output directory management for FrameWright video restoration pipeline.

This module provides utilities for managing output directories, temporary files,
and path resolution across the restoration pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import logging
import shutil
import os

logger = logging.getLogger(__name__)


@dataclass
class OutputPaths:
    """Container for all output directory paths used in the pipeline."""

    project_dir: Path
    frames_dir: Path
    enhanced_dir: Path
    interpolated_dir: Path
    colorized_dir: Path
    watermark_removed_dir: Path
    output_dir: Path
    temp_dir: Path

    def __post_init__(self):
        """Ensure all paths are Path objects."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))


class OutputManager:
    """Manages output directories and file paths for the restoration pipeline.

    This class handles:
    - Directory structure creation and management
    - Path resolution for different processing stages
    - Temporary file cleanup
    - Disk usage monitoring

    Attributes:
        project_dir: Root directory for all project files
        output_dir: Final output directory (user-configurable)
        frames_subdir: Subdirectory name for extracted frames
        enhanced_subdir: Subdirectory name for enhanced frames
        output_subdir: Subdirectory name for final output
    """

    def __init__(
        self,
        project_dir: Path,
        output_dir: Optional[Path] = None,
        frames_subdir: str = "frames",
        enhanced_subdir: str = "enhanced",
        output_subdir: str = "output",
    ):
        """Initialize OutputManager with directory configuration.

        Args:
            project_dir: Root directory for the project
            output_dir: Custom output directory (defaults to project_dir/output_subdir)
            frames_subdir: Name of frames subdirectory
            enhanced_subdir: Name of enhanced frames subdirectory
            output_subdir: Name of output subdirectory
        """
        self.project_dir = Path(project_dir).resolve()
        self.frames_subdir = frames_subdir
        self.enhanced_subdir = enhanced_subdir
        self.output_subdir = output_subdir

        # Set output directory (user-defined or default)
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = self.project_dir / output_subdir

        # Build paths structure
        self._paths = self._build_paths()

        logger.info(f"OutputManager initialized with project_dir: {self.project_dir}")
        logger.info(f"Final output directory: {self.output_dir}")

    def _build_paths(self) -> OutputPaths:
        """Build the complete directory structure.

        Returns:
            OutputPaths object containing all directory paths
        """
        frames_dir = self.project_dir / self.frames_subdir
        enhanced_dir = self.project_dir / self.enhanced_subdir
        temp_dir = self.project_dir / "temp"

        return OutputPaths(
            project_dir=self.project_dir,
            frames_dir=frames_dir,
            enhanced_dir=enhanced_dir,
            interpolated_dir=enhanced_dir / "interpolated",
            colorized_dir=enhanced_dir / "colorized",
            watermark_removed_dir=enhanced_dir / "watermark_removed",
            output_dir=self.output_dir,
            temp_dir=temp_dir,
        )

    def get_paths(self) -> OutputPaths:
        """Get all configured directory paths.

        Returns:
            OutputPaths object containing all directory paths
        """
        return self._paths

    def create_directories(self, exist_ok: bool = True) -> None:
        """Create all necessary directories in the pipeline.

        Args:
            exist_ok: If True, don't raise error if directories exist

        Raises:
            PermissionError: If insufficient permissions to create directories
            OSError: If directory creation fails for other reasons
        """
        paths = self.get_paths()
        directories = [
            paths.project_dir,
            paths.frames_dir,
            paths.enhanced_dir,
            paths.interpolated_dir,
            paths.colorized_dir,
            paths.watermark_removed_dir,
            paths.output_dir,
            paths.temp_dir,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=exist_ok)
                logger.debug(f"Created directory: {directory}")
            except PermissionError as e:
                logger.error(f"Permission denied creating directory: {directory}")
                raise PermissionError(
                    f"Insufficient permissions to create directory: {directory}"
                ) from e
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise

    def get_output_path(self, filename: str) -> Path:
        """Get full path for an output file.

        Args:
            filename: Name of the output file

        Returns:
            Full path to the output file in the output directory
        """
        return self.output_dir / filename

    def get_frames_path(self, filename: str) -> Path:
        """Get full path for a frames file.

        Args:
            filename: Name of the frames file

        Returns:
            Full path to the frames file
        """
        return self._paths.frames_dir / filename

    def get_enhanced_path(self, subdir: Optional[str] = None, filename: Optional[str] = None) -> Path:
        """Get path in the enhanced directory.

        Args:
            subdir: Optional subdirectory (interpolated, colorized, etc.)
            filename: Optional filename

        Returns:
            Full path in the enhanced directory
        """
        if subdir:
            base_path = self._paths.enhanced_dir / subdir
        else:
            base_path = self._paths.enhanced_dir

        if filename:
            return base_path / filename
        return base_path

    def get_temp_path(self, subdir: Optional[str] = None) -> Path:
        """Get temporary directory path.

        Args:
            subdir: Optional subdirectory within temp

        Returns:
            Path to temp directory or subdirectory
        """
        if subdir:
            temp_path = self._paths.temp_dir / subdir
            temp_path.mkdir(parents=True, exist_ok=True)
            return temp_path
        return self._paths.temp_dir

    def cleanup_temp(self) -> None:
        """Remove temporary files and directories.

        This removes the entire temp directory and its contents.
        """
        temp_dir = self._paths.temp_dir
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except PermissionError as e:
                logger.error(f"Permission denied cleaning temp directory: {temp_dir}")
                logger.warning("Some temporary files may remain")
            except OSError as e:
                logger.error(f"Failed to clean temp directory {temp_dir}: {e}")
                raise

    def cleanup_frames(self) -> None:
        """Remove extracted frames directory."""
        frames_dir = self._paths.frames_dir
        if frames_dir.exists():
            try:
                shutil.rmtree(frames_dir)
                logger.info(f"Cleaned up frames directory: {frames_dir}")
            except PermissionError as e:
                logger.error(f"Permission denied cleaning frames directory: {frames_dir}")
                logger.warning("Some frame files may remain")
            except OSError as e:
                logger.error(f"Failed to clean frames directory {frames_dir}: {e}")
                raise

    def cleanup_enhanced(self) -> None:
        """Remove enhanced frames directory."""
        enhanced_dir = self._paths.enhanced_dir
        if enhanced_dir.exists():
            try:
                shutil.rmtree(enhanced_dir)
                logger.info(f"Cleaned up enhanced directory: {enhanced_dir}")
            except PermissionError as e:
                logger.error(f"Permission denied cleaning enhanced directory: {enhanced_dir}")
                logger.warning("Some enhanced files may remain")
            except OSError as e:
                logger.error(f"Failed to clean enhanced directory {enhanced_dir}: {e}")
                raise

    def cleanup_all(self, keep_output: bool = True) -> None:
        """Remove all generated files.

        Args:
            keep_output: If True, preserve the final output directory
        """
        self.cleanup_temp()
        self.cleanup_frames()
        self.cleanup_enhanced()

        if not keep_output and self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up output directory: {self.output_dir}")
            except PermissionError as e:
                logger.error(f"Permission denied cleaning output directory: {self.output_dir}")
                logger.warning("Some output files may remain")
            except OSError as e:
                logger.error(f"Failed to clean output directory {self.output_dir}: {e}")
                raise

    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics for each directory.

        Returns:
            Dictionary mapping directory names to sizes in MB
        """
        usage = {}
        paths = self.get_paths()

        directories = {
            "frames": paths.frames_dir,
            "enhanced": paths.enhanced_dir,
            "interpolated": paths.interpolated_dir,
            "colorized": paths.colorized_dir,
            "watermark_removed": paths.watermark_removed_dir,
            "output": paths.output_dir,
            "temp": paths.temp_dir,
            "total": paths.project_dir,
        }

        for name, directory in directories.items():
            if directory.exists():
                size_bytes = self._get_directory_size(directory)
                usage[name] = round(size_bytes / (1024 * 1024), 2)  # Convert to MB
            else:
                usage[name] = 0.0

        return usage

    @staticmethod
    def _get_directory_size(directory: Path) -> int:
        """Calculate total size of a directory.

        Args:
            directory: Directory to measure

        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except (PermissionError, OSError):
                        # Skip files we can't access
                        continue
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {directory}")

        return total_size

    @staticmethod
    def resolve_output_path(
        output_arg: Optional[str],
        output_dir_arg: Optional[str],
        default_filename: str,
        output_format: str,
    ) -> Path:
        """Resolve output path from CLI arguments.

        This method handles the logic for determining the final output file path
        based on various CLI argument combinations.

        Args:
            output_arg: Value from --output argument (full file path)
            output_dir_arg: Value from --output-dir argument (directory only)
            default_filename: Default filename to use
            output_format: File extension (without dot)

        Returns:
            Resolved output file path

        Examples:
            >>> OutputManager.resolve_output_path(
            ...     output_arg="/path/to/video.mp4",
            ...     output_dir_arg=None,
            ...     default_filename="output",
            ...     output_format="mp4"
            ... )
            Path('/path/to/video.mp4')

            >>> OutputManager.resolve_output_path(
            ...     output_arg=None,
            ...     output_dir_arg="/custom/dir",
            ...     default_filename="output",
            ...     output_format="mp4"
            ... )
            Path('/custom/dir/output.mp4')
        """
        # Case 1: Full output path specified
        if output_arg:
            output_path = Path(output_arg)
            # Ensure correct extension
            if output_path.suffix.lower() != f".{output_format.lower()}":
                output_path = output_path.with_suffix(f".{output_format}")
            return output_path.resolve()

        # Case 2: Output directory specified
        if output_dir_arg:
            output_dir = Path(output_dir_arg)
            filename = f"{default_filename}.{output_format}"
            return (output_dir / filename).resolve()

        # Case 3: Use current directory with default filename
        filename = f"{default_filename}.{output_format}"
        return Path.cwd() / filename

    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all paths are accessible and writable.

        Returns:
            Dictionary mapping directory names to validation status
        """
        validation = {}
        paths = self.get_paths()

        directories = {
            "project_dir": paths.project_dir,
            "frames_dir": paths.frames_dir,
            "enhanced_dir": paths.enhanced_dir,
            "output_dir": paths.output_dir,
            "temp_dir": paths.temp_dir,
        }

        for name, directory in directories.items():
            try:
                # Check if we can create the directory
                directory.mkdir(parents=True, exist_ok=True)

                # Check if we can write to it
                test_file = directory / ".write_test"
                test_file.touch()
                test_file.unlink()

                validation[name] = True
                logger.debug(f"Validated {name}: {directory}")
            except (PermissionError, OSError) as e:
                validation[name] = False
                logger.error(f"Validation failed for {name} ({directory}): {e}")

        return validation

    def __repr__(self) -> str:
        """String representation of OutputManager."""
        return (
            f"OutputManager("
            f"project_dir={self.project_dir}, "
            f"output_dir={self.output_dir})"
        )
