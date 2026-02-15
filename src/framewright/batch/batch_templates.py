"""Batch Templates for Applying Settings to Multiple Folders.

Provides a system for creating and applying batch processing templates
to multiple folders with consistent settings.

Features:
- Create reusable batch templates
- Folder pattern matching
- Per-folder overrides
- Progress tracking across batches
- Template import/export

Example:
    >>> template = BatchTemplate("archive_restoration")
    >>> template.set_config({"preset": "archive", "scale_factor": 4})
    >>> template.add_folder("./films/1950s", priority=1)
    >>> template.add_folder("./films/1960s", priority=2)
    >>> runner = BatchRunner(template)
    >>> runner.run()
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FolderConfig:
    """Configuration for a single folder in batch."""
    path: Path
    priority: int = 5  # 1 = highest, 10 = lowest
    enabled: bool = True
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[Path] = None
    recursive: bool = False
    file_patterns: List[str] = field(default_factory=lambda: ["*.mp4", "*.mkv", "*.avi", "*.mov"])

    # Status tracking
    status: str = "pending"  # pending, processing, completed, failed, skipped
    videos_found: int = 0
    videos_processed: int = 0
    videos_failed: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["path"] = str(self.path)
        if self.output_dir:
            d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderConfig":
        """Create from dictionary."""
        data["path"] = Path(data["path"])
        if data.get("output_dir"):
            data["output_dir"] = Path(data["output_dir"])
        return cls(**data)


@dataclass
class BatchTemplate:
    """A reusable batch processing template."""
    name: str
    description: str = ""

    # Base configuration applied to all folders
    base_config: Dict[str, Any] = field(default_factory=dict)

    # Folder configurations
    folders: List[FolderConfig] = field(default_factory=list)

    # Template metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    version: str = "1.0"

    # Processing options
    stop_on_error: bool = False
    max_concurrent: int = 1
    notify_on_complete: bool = True

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set base configuration.

        Args:
            config: Configuration dictionary
        """
        self.base_config = config
        self.updated_at = datetime.now().isoformat()

    def add_folder(
        self,
        path: Path,
        priority: int = 5,
        recursive: bool = False,
        output_dir: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        file_patterns: Optional[List[str]] = None,
    ) -> None:
        """Add a folder to the batch.

        Args:
            path: Folder path
            priority: Processing priority (1=highest)
            recursive: Search subdirectories
            output_dir: Custom output directory
            config_overrides: Override base config for this folder
            file_patterns: Custom file patterns
        """
        folder = FolderConfig(
            path=Path(path),
            priority=priority,
            recursive=recursive,
            output_dir=Path(output_dir) if output_dir else None,
            config_overrides=config_overrides or {},
            file_patterns=file_patterns or ["*.mp4", "*.mkv", "*.avi", "*.mov"],
        )
        self.folders.append(folder)
        self.updated_at = datetime.now().isoformat()

    def remove_folder(self, path: Path) -> bool:
        """Remove a folder from the batch.

        Args:
            path: Folder path

        Returns:
            True if removed
        """
        path = Path(path)
        for i, folder in enumerate(self.folders):
            if folder.path == path:
                self.folders.pop(i)
                return True
        return False

    def get_folder_config(self, folder: FolderConfig) -> Dict[str, Any]:
        """Get merged configuration for a folder.

        Args:
            folder: Folder configuration

        Returns:
            Merged config dictionary
        """
        config = self.base_config.copy()
        config.update(folder.config_overrides)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "base_config": self.base_config,
            "folders": [f.to_dict() for f in self.folders],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "version": self.version,
            "stop_on_error": self.stop_on_error,
            "max_concurrent": self.max_concurrent,
            "notify_on_complete": self.notify_on_complete,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchTemplate":
        """Create from dictionary."""
        folders = [FolderConfig.from_dict(f) for f in data.pop("folders", [])]
        template = cls(**data)
        template.folders = folders
        return template

    def save(self, path: Path) -> bool:
        """Save template to file.

        Args:
            path: Output path

        Returns:
            True if saved
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            return False

    @classmethod
    def load(cls, path: Path) -> Optional["BatchTemplate"]:
        """Load template from file.

        Args:
            path: Template file path

        Returns:
            BatchTemplate or None
        """
        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            return None


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""
    total_folders: int = 0
    completed_folders: int = 0
    failed_folders: int = 0
    current_folder: str = ""
    current_folder_progress: float = 0.0
    total_videos: int = 0
    completed_videos: int = 0
    failed_videos: int = 0
    start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def folder_progress_percent(self) -> float:
        if self.total_folders == 0:
            return 0.0
        return self.completed_folders / self.total_folders * 100

    @property
    def video_progress_percent(self) -> float:
        if self.total_videos == 0:
            return 0.0
        return self.completed_videos / self.total_videos * 100


class BatchRunner:
    """Runs batch processing from a template."""

    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm', '.m4v', '.flv'}

    def __init__(
        self,
        template: BatchTemplate,
        process_callback: Optional[Callable[[Path, Path, Dict[str, Any]], bool]] = None,
    ):
        """Initialize batch runner.

        Args:
            template: Batch template to run
            process_callback: Callback for processing each video
                             (input_path, output_path, config) -> success
        """
        self.template = template
        self.process_callback = process_callback
        self.progress = BatchProgress()
        self._stopped = False

    def find_videos(self, folder: FolderConfig) -> List[Path]:
        """Find videos in a folder.

        Args:
            folder: Folder configuration

        Returns:
            List of video paths
        """
        videos = []
        patterns = folder.file_patterns

        if folder.recursive:
            for pattern in patterns:
                videos.extend(folder.path.rglob(pattern))
        else:
            for pattern in patterns:
                videos.extend(folder.path.glob(pattern))

        # Filter to actual video files
        videos = [
            v for v in videos
            if v.is_file() and v.suffix.lower() in self.VIDEO_EXTENSIONS
        ]

        return sorted(videos)

    def run(
        self,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ) -> BatchProgress:
        """Run the batch processing.

        Args:
            progress_callback: Callback for progress updates

        Returns:
            Final BatchProgress
        """
        self._stopped = False
        self.progress = BatchProgress()

        # Sort folders by priority
        folders = sorted(
            [f for f in self.template.folders if f.enabled],
            key=lambda f: f.priority
        )

        self.progress.total_folders = len(folders)

        # Count total videos
        for folder in folders:
            videos = self.find_videos(folder)
            folder.videos_found = len(videos)
            self.progress.total_videos += len(videos)

        logger.info(
            f"Starting batch: {self.progress.total_folders} folders, "
            f"{self.progress.total_videos} videos"
        )

        # Process each folder
        for folder in folders:
            if self._stopped:
                break

            self._process_folder(folder, progress_callback)

            if folder.status == "failed" and self.template.stop_on_error:
                logger.error("Stopping batch due to error")
                break

        # Notify completion
        if self.template.notify_on_complete:
            self._send_notification()

        return self.progress

    def _process_folder(
        self,
        folder: FolderConfig,
        progress_callback: Optional[Callable[[BatchProgress], None]],
    ) -> None:
        """Process a single folder."""
        folder.status = "processing"
        self.progress.current_folder = str(folder.path)

        logger.info(f"Processing folder: {folder.path}")

        try:
            videos = self.find_videos(folder)
            config = self.template.get_folder_config(folder)

            # Determine output directory
            output_dir = folder.output_dir or (folder.path / "restored")
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, video_path in enumerate(videos):
                if self._stopped:
                    break

                self.progress.current_folder_progress = (i + 1) / len(videos)

                if progress_callback:
                    progress_callback(self.progress)

                # Process video
                output_path = output_dir / f"{video_path.stem}_restored.mp4"

                try:
                    if self.process_callback:
                        success = self.process_callback(video_path, output_path, config)
                    else:
                        success = self._default_process(video_path, output_path, config)

                    if success:
                        folder.videos_processed += 1
                        self.progress.completed_videos += 1
                    else:
                        folder.videos_failed += 1
                        self.progress.failed_videos += 1

                except Exception as e:
                    logger.error(f"Failed to process {video_path}: {e}")
                    folder.videos_failed += 1
                    self.progress.failed_videos += 1
                    self.progress.errors.append(f"{video_path.name}: {str(e)}")

            folder.status = "completed" if folder.videos_failed == 0 else "failed"
            self.progress.completed_folders += 1

        except Exception as e:
            folder.status = "failed"
            folder.error = str(e)
            self.progress.failed_folders += 1
            self.progress.errors.append(f"{folder.path}: {str(e)}")
            logger.error(f"Folder processing failed: {e}")

    def _default_process(
        self,
        input_path: Path,
        output_path: Path,
        config: Dict[str, Any],
    ) -> bool:
        """Default processing using FrameWright."""
        try:
            from ..cli_simple import run_best_restore
            from ..ui.terminal import create_console

            console = create_console(quiet=True)
            run_best_restore(input_path, output_path, console)
            return output_path.exists()

        except Exception as e:
            logger.error(f"Default processing failed: {e}")
            return False

    def _send_notification(self) -> None:
        """Send completion notification."""
        try:
            from ..cli_advanced import send_notification

            title = "FrameWright Batch Complete"
            message = (
                f"Processed {self.progress.completed_videos}/{self.progress.total_videos} videos "
                f"in {self.progress.completed_folders} folders"
            )

            if self.progress.failed_videos > 0:
                message += f" ({self.progress.failed_videos} failed)"

            send_notification(title, message)

        except Exception:
            pass

    def stop(self) -> None:
        """Stop batch processing."""
        self._stopped = True
        logger.info("Batch processing stop requested")

    def get_summary(self) -> Dict[str, Any]:
        """Get batch summary.

        Returns:
            Summary dictionary
        """
        return {
            "template": self.template.name,
            "total_folders": self.progress.total_folders,
            "completed_folders": self.progress.completed_folders,
            "failed_folders": self.progress.failed_folders,
            "total_videos": self.progress.total_videos,
            "completed_videos": self.progress.completed_videos,
            "failed_videos": self.progress.failed_videos,
            "elapsed_seconds": self.progress.elapsed_seconds,
            "errors": self.progress.errors,
        }


class TemplateManager:
    """Manages batch templates."""

    TEMPLATES_DIR = "batch_templates"

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize template manager.

        Args:
            storage_dir: Directory for storing templates
        """
        self.storage_dir = storage_dir or (Path.home() / ".framewright" / self.TEMPLATES_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates.

        Returns:
            List of template info
        """
        templates = []

        for path in self.storage_dir.glob("*.json"):
            try:
                template = BatchTemplate.load(path)
                if template:
                    templates.append({
                        "name": template.name,
                        "description": template.description,
                        "folders": len(template.folders),
                        "created_at": template.created_at,
                        "path": str(path),
                    })
            except Exception:
                pass

        return templates

    def save_template(self, template: BatchTemplate) -> bool:
        """Save a template.

        Args:
            template: Template to save

        Returns:
            True if saved
        """
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in template.name)
        path = self.storage_dir / f"{safe_name}.json"
        return template.save(path)

    def load_template(self, name: str) -> Optional[BatchTemplate]:
        """Load a template by name.

        Args:
            name: Template name

        Returns:
            BatchTemplate or None
        """
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = self.storage_dir / f"{safe_name}.json"

        if path.exists():
            return BatchTemplate.load(path)

        return None

    def delete_template(self, name: str) -> bool:
        """Delete a template.

        Args:
            name: Template name

        Returns:
            True if deleted
        """
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = self.storage_dir / f"{safe_name}.json"

        if path.exists():
            path.unlink()
            return True

        return False


# Convenience functions
def create_template(
    name: str,
    config: Dict[str, Any],
    folders: List[Path],
    description: str = "",
) -> BatchTemplate:
    """Create a batch template.

    Args:
        name: Template name
        config: Base configuration
        folders: List of folder paths
        description: Template description

    Returns:
        BatchTemplate
    """
    template = BatchTemplate(name=name, description=description)
    template.set_config(config)

    for folder in folders:
        template.add_folder(folder)

    return template


def run_batch(template: BatchTemplate) -> BatchProgress:
    """Run a batch template.

    Args:
        template: Template to run

    Returns:
        BatchProgress
    """
    runner = BatchRunner(template)
    return runner.run()
