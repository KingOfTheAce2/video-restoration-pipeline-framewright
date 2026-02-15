"""Project management module for FrameWright.

Provides project-based workflow with save/load and version control.
"""

from .project_manager import (
    ProjectManager,
    Project,
    ProjectStatus,
    StageStatus,
    StageProgress,
    ProjectVersion,
    ProjectMetadata,
    SourceFile,
    create_project,
    open_project,
    PROJECT_VERSION,
    PROJECT_EXTENSION,
)

__all__ = [
    "ProjectManager",
    "Project",
    "ProjectStatus",
    "StageStatus",
    "StageProgress",
    "ProjectVersion",
    "ProjectMetadata",
    "SourceFile",
    "create_project",
    "open_project",
    "PROJECT_VERSION",
    "PROJECT_EXTENSION",
]
