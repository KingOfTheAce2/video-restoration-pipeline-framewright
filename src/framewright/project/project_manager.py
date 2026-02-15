"""Project file management for FrameWright.

Provides project-based workflow with save/load capabilities,
version tracking, and restoration state management.
"""

import logging
import json
import shutil
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

PROJECT_VERSION = "1.0.0"
PROJECT_EXTENSION = ".fwproj"


class ProjectStatus(Enum):
    """Project status."""
    NEW = "new"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(Enum):
    """Stage processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class SourceFile:
    """Information about source file."""
    path: str
    filename: str
    size_bytes: int = 0
    duration_seconds: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    codec: str = ""
    hash_md5: str = ""
    added_at: str = ""


@dataclass
class StageProgress:
    """Progress for a processing stage."""
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    frames_processed: int = 0
    frames_total: int = 0
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProjectVersion:
    """Version snapshot of project state."""
    version: int
    created_at: str
    description: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    stages_snapshot: List[Dict] = field(default_factory=list)
    auto_save: bool = True


@dataclass
class ProjectMetadata:
    """Project metadata."""
    name: str
    description: str = ""
    created_at: str = ""
    modified_at: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Project:
    """Complete project data structure."""
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = PROJECT_VERSION
    metadata: ProjectMetadata = field(default_factory=lambda: ProjectMetadata(name="Untitled"))

    # Source
    source: Optional[SourceFile] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    preset_name: str = "balanced"

    # Processing state
    status: ProjectStatus = ProjectStatus.NEW
    stages: List[StageProgress] = field(default_factory=list)
    current_stage: Optional[str] = None

    # Paths
    project_dir: str = ""
    output_path: Optional[str] = None
    temp_dir: Optional[str] = None

    # History
    versions: List[ProjectVersion] = field(default_factory=list)
    current_version: int = 0

    # Results
    result_path: Optional[str] = None
    qa_report_path: Optional[str] = None
    processing_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value

        # Convert stage statuses
        for stage in data["stages"]:
            stage["status"] = stage["status"].value if isinstance(stage["status"], StageStatus) else stage["status"]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Create from dictionary."""
        data = data.copy()

        # Convert enums
        data["status"] = ProjectStatus(data.get("status", "new"))

        # Convert metadata
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = ProjectMetadata(**data["metadata"])

        # Convert source
        if "source" in data and data["source"] and isinstance(data["source"], dict):
            data["source"] = SourceFile(**data["source"])

        # Convert stages
        stages = []
        for stage_data in data.get("stages", []):
            if isinstance(stage_data, dict):
                stage_data["status"] = StageStatus(stage_data.get("status", "pending"))
                stages.append(StageProgress(**stage_data))
        data["stages"] = stages

        # Convert versions
        versions = []
        for ver_data in data.get("versions", []):
            if isinstance(ver_data, dict):
                versions.append(ProjectVersion(**ver_data))
        data["versions"] = versions

        return cls(**data)


class ProjectManager:
    """Manages FrameWright projects.

    Provides project creation, save/load, version control,
    and state management for restoration workflows.
    """

    def __init__(self, projects_dir: Optional[Path] = None):
        """Initialize project manager.

        Args:
            projects_dir: Base directory for projects
        """
        if projects_dir is None:
            projects_dir = Path.home() / ".framewright" / "projects"
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        self._current_project: Optional[Project] = None

    @property
    def current_project(self) -> Optional[Project]:
        """Get current project."""
        return self._current_project

    def create_project(
        self,
        name: str,
        source_path: Optional[Path] = None,
        preset: str = "balanced",
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> Project:
        """Create a new project.

        Args:
            name: Project name
            source_path: Optional source video path
            preset: Restoration preset name
            config: Additional configuration
            description: Project description
            author: Author name
            tags: Project tags

        Returns:
            Created Project
        """
        now = datetime.now().isoformat()

        project = Project(
            metadata=ProjectMetadata(
                name=name,
                description=description,
                created_at=now,
                modified_at=now,
                author=author,
                tags=tags or [],
            ),
            preset_name=preset,
            config=config or {},
            status=ProjectStatus.NEW,
        )

        # Set up project directory
        safe_name = self._sanitize_name(name)
        project_dir = self.projects_dir / f"{safe_name}_{project.id[:8]}"
        project_dir.mkdir(parents=True, exist_ok=True)
        project.project_dir = str(project_dir)
        project.temp_dir = str(project_dir / "temp")

        # Add source if provided
        if source_path:
            self._add_source(project, source_path)

        # Initialize stages
        project.stages = self._create_stages(preset)

        # Create initial version
        self._create_version(project, "Initial project creation")

        # Save project
        self.save_project(project)

        self._current_project = project
        logger.info(f"Created project: {name} ({project.id})")

        return project

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for filesystem."""
        # Remove/replace invalid characters
        invalid = '<>:"/\\|?*'
        for char in invalid:
            name = name.replace(char, "_")
        return name[:50]

    def _add_source(self, project: Project, source_path: Path) -> None:
        """Add source file to project."""
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Calculate hash
        hash_md5 = self._calculate_hash(source_path)

        # Get video info
        info = self._get_video_info(source_path)

        project.source = SourceFile(
            path=str(source_path),
            filename=source_path.name,
            size_bytes=source_path.stat().st_size,
            duration_seconds=info.get("duration", 0),
            resolution=info.get("resolution", ""),
            fps=info.get("fps", 0),
            codec=info.get("codec", ""),
            hash_md5=hash_md5,
            added_at=datetime.now().isoformat(),
        )

    def _calculate_hash(self, path: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of file."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_video_info(self, path: Path) -> Dict[str, Any]:
        """Get video information using FFprobe."""
        import subprocess
        import json

        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    str(path)
                ],
                capture_output=True,
                text=True,
            )
            data = json.loads(result.stdout)

            info = {}
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["resolution"] = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        info["fps"] = float(num) / float(den) if float(den) > 0 else 0
                    info["codec"] = stream.get("codec_name", "")
                    break

            fmt = data.get("format", {})
            info["duration"] = float(fmt.get("duration", 0))

            return info

        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return {}

    def _create_stages(self, preset: str) -> List[StageProgress]:
        """Create processing stages based on preset."""
        # Define all possible stages
        all_stages = [
            "analyze",
            "qp_artifact_removal",
            "tap_denoise",
            "frame_generation",
            "deduplication",
            "extract_frames",
            "enhance",
            "face_restore",
            "interpolate",
            "temporal_consistency",
            "colorize",
            "reassemble",
            "audio_restore",
            "validate",
        ]

        # Stages active by preset
        preset_stages = {
            "fast": ["analyze", "extract_frames", "enhance", "reassemble"],
            "balanced": ["analyze", "extract_frames", "enhance", "face_restore", "reassemble"],
            "quality": ["analyze", "extract_frames", "enhance", "face_restore", "temporal_consistency", "reassemble", "validate"],
            "ultimate": all_stages,
            "archive": all_stages,
        }

        active = preset_stages.get(preset, preset_stages["balanced"])

        stages = []
        for name in all_stages:
            status = StageStatus.PENDING if name in active else StageStatus.SKIPPED
            stages.append(StageProgress(name=name, status=status))

        return stages

    def _create_version(self, project: Project, description: str = "") -> ProjectVersion:
        """Create a new version snapshot."""
        project.current_version += 1

        version = ProjectVersion(
            version=project.current_version,
            created_at=datetime.now().isoformat(),
            description=description,
            config_snapshot=project.config.copy(),
            stages_snapshot=[asdict(s) for s in project.stages],
        )

        project.versions.append(version)

        # Keep only last 20 versions
        if len(project.versions) > 20:
            project.versions = project.versions[-20:]

        return version

    def save_project(self, project: Optional[Project] = None) -> Path:
        """Save project to file.

        Args:
            project: Project to save (uses current if None)

        Returns:
            Path to saved project file
        """
        project = project or self._current_project
        if not project:
            raise ValueError("No project to save")

        project.metadata.modified_at = datetime.now().isoformat()

        project_file = Path(project.project_dir) / f"{self._sanitize_name(project.metadata.name)}{PROJECT_EXTENSION}"

        with open(project_file, "w") as f:
            json.dump(project.to_dict(), f, indent=2)

        logger.info(f"Project saved: {project_file}")
        return project_file

    def load_project(self, path: Union[str, Path]) -> Project:
        """Load project from file.

        Args:
            path: Path to project file or directory

        Returns:
            Loaded Project
        """
        path = Path(path)

        # If directory, find project file
        if path.is_dir():
            project_files = list(path.glob(f"*{PROJECT_EXTENSION}"))
            if not project_files:
                raise FileNotFoundError(f"No project file found in {path}")
            path = project_files[0]

        with open(path) as f:
            data = json.load(f)

        project = Project.from_dict(data)
        self._current_project = project

        logger.info(f"Project loaded: {project.metadata.name}")
        return project

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects.

        Returns:
            List of project summaries
        """
        projects = []

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_files = list(project_dir.glob(f"*{PROJECT_EXTENSION}"))
            if not project_files:
                continue

            try:
                with open(project_files[0]) as f:
                    data = json.load(f)

                projects.append({
                    "id": data.get("id", ""),
                    "name": data.get("metadata", {}).get("name", "Unknown"),
                    "status": data.get("status", "unknown"),
                    "created_at": data.get("metadata", {}).get("created_at", ""),
                    "modified_at": data.get("metadata", {}).get("modified_at", ""),
                    "path": str(project_files[0]),
                })
            except Exception as e:
                logger.warning(f"Could not read project {project_dir}: {e}")

        # Sort by modification time
        projects.sort(key=lambda p: p.get("modified_at", ""), reverse=True)
        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete a project.

        Args:
            project_id: Project ID

        Returns:
            True if deleted
        """
        for project_dir in self.projects_dir.iterdir():
            if project_id in project_dir.name:
                shutil.rmtree(project_dir)
                logger.info(f"Deleted project: {project_dir}")
                return True
        return False

    def update_stage(
        self,
        stage_name: str,
        status: Optional[StageStatus] = None,
        frames_processed: Optional[int] = None,
        frames_total: Optional[int] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update stage progress.

        Args:
            stage_name: Name of stage to update
            status: New status
            frames_processed: Frames completed
            frames_total: Total frames
            error_message: Error message if failed
            output_path: Output path if completed
            metrics: Quality metrics
        """
        if not self._current_project:
            return

        for stage in self._current_project.stages:
            if stage.name == stage_name:
                now = datetime.now().isoformat()

                if status:
                    stage.status = status
                    if status == StageStatus.RUNNING:
                        stage.started_at = now
                        self._current_project.current_stage = stage_name
                    elif status in [StageStatus.COMPLETED, StageStatus.FAILED]:
                        stage.completed_at = now

                if frames_processed is not None:
                    stage.frames_processed = frames_processed
                if frames_total is not None:
                    stage.frames_total = frames_total
                if error_message:
                    stage.error_message = error_message
                if output_path:
                    stage.output_path = output_path
                if metrics:
                    stage.metrics.update(metrics)

                break

        # Auto-save
        self.save_project()

    def set_status(self, status: ProjectStatus) -> None:
        """Set project status.

        Args:
            status: New status
        """
        if self._current_project:
            self._current_project.status = status
            self.save_project()

    def add_log(self, message: str) -> None:
        """Add message to processing log.

        Args:
            message: Log message
        """
        if self._current_project:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._current_project.processing_log.append(f"[{timestamp}] {message}")

    def revert_to_version(self, version_number: int) -> bool:
        """Revert project to a previous version.

        Args:
            version_number: Version to revert to

        Returns:
            True if reverted
        """
        if not self._current_project:
            return False

        for version in self._current_project.versions:
            if version.version == version_number:
                self._current_project.config = version.config_snapshot.copy()

                # Restore stages
                self._current_project.stages = []
                for stage_data in version.stages_snapshot:
                    stage_data["status"] = StageStatus(stage_data.get("status", "pending"))
                    self._current_project.stages.append(StageProgress(**stage_data))

                self._create_version(self._current_project, f"Reverted to version {version_number}")
                self.save_project()

                logger.info(f"Reverted to version {version_number}")
                return True

        return False

    def generate_changelog(
        self,
        project: Optional[Project] = None,
        format: str = "text",
    ) -> str:
        """Generate human-readable changelog of project history.

        Args:
            project: Project (uses current if None)
            format: Output format ("text" or "markdown")

        Returns:
            Changelog string
        """
        project = project or self._current_project
        if not project:
            return "No project loaded."

        lines = []

        if format == "markdown":
            lines.append(f"# Changelog: {project.metadata.name}")
            lines.append("")
            lines.append(f"**Project ID:** {project.id}")
            lines.append(f"**Created:** {project.metadata.created_at}")
            lines.append(f"**Status:** {project.status.value}")
            lines.append("")
            lines.append("## Version History")
            lines.append("")

            for version in reversed(project.versions):
                lines.append(f"### Version {version.version}")
                lines.append(f"- **Date:** {version.created_at}")
                lines.append(f"- **Description:** {version.description or 'No description'}")
                if version.config_snapshot:
                    lines.append(f"- **Config changes:** {len(version.config_snapshot)} settings")
                lines.append("")

            lines.append("## Processing Stages")
            lines.append("")
            for stage in project.stages:
                status_icon = {
                    StageStatus.COMPLETED: "[x]",
                    StageStatus.RUNNING: "[~]",
                    StageStatus.FAILED: "[!]",
                    StageStatus.SKIPPED: "[-]",
                    StageStatus.PENDING: "[ ]",
                }.get(stage.status, "[ ]")

                lines.append(f"- {status_icon} **{stage.name}**")
                if stage.started_at:
                    lines.append(f"  - Started: {stage.started_at}")
                if stage.completed_at:
                    lines.append(f"  - Completed: {stage.completed_at}")
                if stage.frames_processed > 0:
                    lines.append(f"  - Frames: {stage.frames_processed}/{stage.frames_total}")
                if stage.error_message:
                    lines.append(f"  - Error: {stage.error_message}")
                if stage.metrics:
                    metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in stage.metrics.items())
                    lines.append(f"  - Metrics: {metrics_str}")

            if project.processing_log:
                lines.append("")
                lines.append("## Processing Log")
                lines.append("")
                lines.append("```")
                for entry in project.processing_log[-20:]:  # Last 20 entries
                    lines.append(entry)
                lines.append("```")

        else:  # text format
            lines.append(f"CHANGELOG: {project.metadata.name}")
            lines.append("=" * 60)
            lines.append(f"Project ID: {project.id}")
            lines.append(f"Created: {project.metadata.created_at}")
            lines.append(f"Status: {project.status.value}")
            lines.append("")
            lines.append("VERSION HISTORY:")
            lines.append("-" * 40)

            for version in reversed(project.versions):
                lines.append(f"  v{version.version} ({version.created_at})")
                lines.append(f"    {version.description or 'No description'}")

            lines.append("")
            lines.append("PROCESSING STAGES:")
            lines.append("-" * 40)

            for stage in project.stages:
                status_str = stage.status.value.upper()
                lines.append(f"  [{status_str:^10}] {stage.name}")
                if stage.frames_processed > 0:
                    lines.append(f"               Frames: {stage.frames_processed}/{stage.frames_total}")
                if stage.error_message:
                    lines.append(f"               Error: {stage.error_message}")

            if project.processing_log:
                lines.append("")
                lines.append("RECENT LOG ENTRIES:")
                lines.append("-" * 40)
                for entry in project.processing_log[-10:]:
                    lines.append(f"  {entry}")

        return "\n".join(lines)

    def export_changelog(
        self,
        output_path: Path,
        project: Optional[Project] = None,
        format: str = "markdown",
    ) -> Path:
        """Export changelog to file.

        Args:
            output_path: Output file path
            project: Project (uses current if None)
            format: Output format

        Returns:
            Path to exported file
        """
        changelog = self.generate_changelog(project, format)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(changelog)
        logger.info(f"Changelog exported: {output_path}")
        return output_path

    def get_project_summary(self, project: Optional[Project] = None) -> Dict[str, Any]:
        """Get project summary.

        Args:
            project: Project (uses current if None)

        Returns:
            Summary dictionary
        """
        project = project or self._current_project
        if not project:
            return {}

        completed = sum(1 for s in project.stages if s.status == StageStatus.COMPLETED)
        total = sum(1 for s in project.stages if s.status != StageStatus.SKIPPED)

        return {
            "name": project.metadata.name,
            "status": project.status.value,
            "progress": f"{completed}/{total} stages",
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "current_stage": project.current_stage,
            "source": project.source.filename if project.source else None,
            "created": project.metadata.created_at,
            "modified": project.metadata.modified_at,
            "versions": len(project.versions),
        }

    def export_project(self, output_path: Path) -> Path:
        """Export project as portable archive.

        Args:
            output_path: Output archive path

        Returns:
            Path to archive
        """
        if not self._current_project:
            raise ValueError("No project to export")

        output_path = Path(output_path)
        project_dir = Path(self._current_project.project_dir)

        # Create archive
        shutil.make_archive(
            str(output_path.with_suffix("")),
            "zip",
            project_dir.parent,
            project_dir.name,
        )

        archive_path = output_path.with_suffix(".zip")
        logger.info(f"Project exported: {archive_path}")
        return archive_path

    def import_project(self, archive_path: Path) -> Project:
        """Import project from archive.

        Args:
            archive_path: Path to project archive

        Returns:
            Imported Project
        """
        archive_path = Path(archive_path)

        # Extract archive
        extract_dir = self.projects_dir / f"import_{uuid.uuid4().hex[:8]}"
        shutil.unpack_archive(archive_path, extract_dir)

        # Find and load project
        project_files = list(extract_dir.rglob(f"*{PROJECT_EXTENSION}"))
        if not project_files:
            shutil.rmtree(extract_dir)
            raise ValueError("No project file found in archive")

        project = self.load_project(project_files[0])

        # Update project directory
        project.project_dir = str(project_files[0].parent)
        self.save_project(project)

        logger.info(f"Project imported: {project.metadata.name}")
        return project


def create_project(
    name: str,
    source_path: Optional[Path] = None,
    preset: str = "balanced",
    **kwargs,
) -> Project:
    """Quick function to create a project.

    Args:
        name: Project name
        source_path: Source video path
        preset: Restoration preset
        **kwargs: Additional options

    Returns:
        Created Project
    """
    manager = ProjectManager()
    return manager.create_project(
        name=name,
        source_path=source_path,
        preset=preset,
        **kwargs,
    )


def open_project(path: Union[str, Path]) -> Project:
    """Quick function to open a project.

    Args:
        path: Project path

    Returns:
        Loaded Project
    """
    manager = ProjectManager()
    return manager.load_project(path)
