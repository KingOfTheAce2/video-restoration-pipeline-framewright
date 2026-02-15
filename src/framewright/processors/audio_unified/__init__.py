"""Unified audio processing module for FrameWright.

This package provides a unified interface to all audio enhancement capabilities
with automatic backend selection and graceful fallback chains.

Usage:
    >>> from framewright.processors.audio_unified import AudioEnhancer, AudioConfig
    >>> config = AudioConfig(
    ...     enable_noise_reduction=True,
    ...     enable_normalization=True,
    ...     target_loudness_lufs=-14.0
    ... )
    >>> enhancer = AudioEnhancer(config)
    >>> result = enhancer.enhance("input.wav", "output.wav")
    >>> print(f"Processed with: {result.backend_used.name}")

Or use the convenience functions:
    >>> from framewright.processors.audio_unified import enhance_audio, auto_enhance_audio
    >>> result = auto_enhance_audio("input.wav", "output.wav")
"""

from .enhancer import (
    # Main unified class
    AudioEnhancer,
    # Configuration and results
    AudioConfig,
    AudioResult,
    BackendType,
    EnhancementFeature,
    # Backend classes for direct access
    AudioBackend,
    FFmpegAudioBackend,
    TraditionalEnhancerBackend,
    AIAudioBackend,
    RestorationBackend,
    # Convenience functions
    enhance_audio,
    auto_enhance_audio,
    get_available_backends,
)

__all__ = [
    # Main unified class
    "AudioEnhancer",
    # Configuration
    "AudioConfig",
    "AudioResult",
    "BackendType",
    "EnhancementFeature",
    # Backend base class
    "AudioBackend",
    # Backend implementations
    "FFmpegAudioBackend",
    "TraditionalEnhancerBackend",
    "AIAudioBackend",
    "RestorationBackend",
    # Convenience functions
    "enhance_audio",
    "auto_enhance_audio",
    "get_available_backends",
]
