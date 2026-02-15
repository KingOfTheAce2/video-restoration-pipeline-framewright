"""FrameWright Simplified CLI - It Just Works.

This module provides the simplified command-line interface for FrameWright
with the "it just works" philosophy.

Primary usage:
    framewright video.mp4

Everything else is automatic: hardware detection, content analysis,
and optimal settings selection.

Wizard mode for interactive guidance:
    framewright video.mp4 --wizard
"""

from .main import cli, main
from .wizard import (
    WizardConfig,
    RestorationWizard,
    run_wizard,
    create_wizard,
)

__all__ = [
    "cli",
    "main",
    "WizardConfig",
    "RestorationWizard",
    "run_wizard",
    "create_wizard",
]
