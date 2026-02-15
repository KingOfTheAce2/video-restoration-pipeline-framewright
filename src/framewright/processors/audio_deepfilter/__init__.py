"""DeepFilterNet audio enhancement subpackage for FrameWright.

This package provides state-of-the-art audio enhancement capabilities including:

- DeepFilterNet 3 integration for 10-20ms latency noise suppression
- Multiple backend support with automatic fallback
- Real-time streaming enhancement
- Video audio extraction and enhancement

Available modules:
- deepfilter: DeepFilterNet 3 integration for 10-20ms latency enhancement

Usage:
    >>> from framewright.processors.audio_deepfilter import (
    ...     DeepFilterEnhancer, DeepFilterConfig,
    ...     create_deepfilter_enhancer, enhance_audio
    ... )
    >>>
    >>> # Quick enhancement
    >>> result = enhance_audio("noisy.wav", "clean.wav")
    >>>
    >>> # Or with configuration
    >>> config = DeepFilterConfig(denoise=True, dereverb=True, normalize=True)
    >>> enhancer = create_deepfilter_enhancer(config)
    >>> result = enhancer.enhance("input.wav", "output.wav")
"""

from .deepfilter import (
    # Configuration
    DeepFilterConfig,
    BackendPriority,
    # Analysis
    AudioAnalysis,
    AudioAnalyzer,
    # Results
    EnhancementResult,
    # Backend base class
    DeepFilterBackend,
    # Backend implementations
    DeepFilterNet3Backend,
    DeepFilterLiteBackend,
    SpeechBrainBackend,
    TraditionalFilterBackend,
    # Main enhancer class
    DeepFilterEnhancer,
    # Factory functions
    create_deepfilter_enhancer,
    enhance_audio,
    enhance_video_audio,
    analyze_audio,
    get_available_backends,
)

__all__ = [
    # Configuration
    "DeepFilterConfig",
    "BackendPriority",
    # Analysis
    "AudioAnalysis",
    "AudioAnalyzer",
    # Results
    "EnhancementResult",
    # Backend base class
    "DeepFilterBackend",
    # Backend implementations
    "DeepFilterNet3Backend",
    "DeepFilterLiteBackend",
    "SpeechBrainBackend",
    "TraditionalFilterBackend",
    # Main enhancer class
    "DeepFilterEnhancer",
    # Factory functions
    "create_deepfilter_enhancer",
    "enhance_audio",
    "enhance_video_audio",
    "analyze_audio",
    "get_available_backends",
]
