"""Mixins for VideoRestorer to split restorer.py into manageable modules.

This package contains mixin classes that separate concerns:
- FrameProcessingMixin: Frame enhancement methods
- VideoAssemblyMixin: Video/audio extraction and reassembly methods
"""

from .frame_processing import FrameProcessingMixin
from .video_assembly import VideoAssemblyMixin

__all__ = [
    'FrameProcessingMixin',
    'VideoAssemblyMixin',
]
