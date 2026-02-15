"""Unified Audio Enhancement Processor for FrameWright.

This module provides a single AudioEnhancer class that wraps all existing audio
processing backends with automatic backend selection and graceful fallback chains.

Backends (in order of capability):
- ffmpeg: Always available, basic processing (extraction, filters, normalization)
- traditional: FFmpeg-based from audio_enhance.py (noise reduction, declipping, EQ)
- ai: Neural network-based from audio_enhance.py (denoiser, demucs)
- restoration: Full restoration suite from audio_restoration.py (comprehensive)

Example usage:
    >>> config = AudioConfig(
    ...     enable_noise_reduction=True,
    ...     enable_normalization=True,
    ...     target_loudness_lufs=-14.0
    ... )
    >>> enhancer = AudioEnhancer(config)
    >>> result = enhancer.enhance(Path("input.wav"), Path("output.wav"))
    >>> print(f"Processed with backend: {result.backend_used}")
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available audio processing backends."""
    FFMPEG = auto()           # Basic FFmpeg processing (always available)
    TRADITIONAL = auto()      # FFmpeg-based from audio_enhance.py
    AI = auto()               # Neural network-based (denoiser/demucs)
    RESTORATION = auto()      # Full restoration suite


class EnhancementFeature(Enum):
    """Available enhancement features."""
    NOISE_REDUCTION = "noise_reduction"
    DECLIPPING = "declipping"
    NORMALIZATION = "normalization"
    DEHUM = "dehum"
    DEREVERB = "dereverb"
    SPEECH_ENHANCEMENT = "speech_enhancement"
    DECLICK = "declick"
    DIALOG_ENHANCE = "dialog_enhance"
    UPMIX = "upmix"


@dataclass
class AudioConfig:
    """Unified configuration for audio enhancement.

    Combines settings from all backends into a single configuration object.

    Attributes:
        # Noise reduction
        enable_noise_reduction: Apply noise reduction.
        noise_reduction_strength: Noise reduction intensity (0.0-1.0).

        # Declipping
        enable_declipping: Repair clipped/distorted audio peaks.

        # Normalization
        enable_normalization: Apply loudness normalization.
        target_loudness_lufs: Target loudness in LUFS (-14 YouTube, -16 podcast, -23 broadcast).

        # Hum removal
        enable_dehum: Remove electrical hum (50/60Hz).
        hum_frequency: Hum frequency (0 = auto-detect, 50 or 60 Hz).

        # De-reverb
        enable_dereverb: Reduce reverb/echo.
        dereverb_strength: Dereverb intensity (0.0-1.0).

        # Click/pop removal
        enable_declick: Remove clicks and pops.
        click_sensitivity: Click detection sensitivity (0.0-1.0).

        # Dialog enhancement
        enable_dialog_enhance: Enhance speech clarity.
        dialog_boost_db: Dialog boost in dB.

        # AI features
        enable_ai_enhancement: Use AI models when available.
        ai_model: AI model type ("speech", "music", "general").

        # Upmixing
        enable_upmix: Upmix mono to stereo.

        # Backend selection
        preferred_backend: Preferred backend (None = auto-select).
        fallback_enabled: Enable fallback to simpler backends on failure.

        # Output settings
        output_sample_rate: Output sample rate in Hz.
        output_bit_depth: Output bit depth.
    """
    # Noise reduction
    enable_noise_reduction: bool = True
    noise_reduction_strength: float = 0.5

    # Declipping
    enable_declipping: bool = True

    # Normalization
    enable_normalization: bool = True
    target_loudness_lufs: float = -14.0

    # Hum removal
    enable_dehum: bool = False
    hum_frequency: int = 0  # 0 = auto-detect

    # De-reverb
    enable_dereverb: bool = False
    dereverb_strength: float = 0.5

    # Click/pop removal
    enable_declick: bool = False
    click_sensitivity: float = 0.5

    # Dialog enhancement
    enable_dialog_enhance: bool = False
    dialog_boost_db: float = 3.0

    # AI features
    enable_ai_enhancement: bool = False
    ai_model: str = "speech"  # "speech", "music", "general"

    # Upmixing
    enable_upmix: bool = False

    # Backend selection
    preferred_backend: Optional[BackendType] = None
    fallback_enabled: bool = True

    # Output settings
    output_sample_rate: int = 48000
    output_bit_depth: int = 24

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.noise_reduction_strength <= 1.0:
            raise ValueError(
                f"noise_reduction_strength must be between 0.0 and 1.0, "
                f"got {self.noise_reduction_strength}"
            )

        if self.hum_frequency not in (0, 50, 60):
            raise ValueError(
                f"hum_frequency must be 0 (auto), 50, or 60 Hz, "
                f"got {self.hum_frequency}"
            )

        if not -70.0 <= self.target_loudness_lufs <= 0.0:
            raise ValueError(
                f"target_loudness_lufs must be between -70.0 and 0.0, "
                f"got {self.target_loudness_lufs}"
            )

        if not 0.0 <= self.dereverb_strength <= 1.0:
            raise ValueError(
                f"dereverb_strength must be between 0.0 and 1.0, "
                f"got {self.dereverb_strength}"
            )

        if not 0.0 <= self.click_sensitivity <= 1.0:
            raise ValueError(
                f"click_sensitivity must be between 0.0 and 1.0, "
                f"got {self.click_sensitivity}"
            )

        valid_ai_models = ("speech", "music", "general")
        if self.ai_model not in valid_ai_models:
            raise ValueError(
                f"ai_model must be one of {valid_ai_models}, got {self.ai_model}"
            )


@dataclass
class AudioResult:
    """Result of audio enhancement processing.

    Attributes:
        success: Whether enhancement completed successfully.
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        backend_used: Backend that processed the audio.
        stages_applied: List of processing stages applied.
        processing_time_seconds: Total processing time.
        fallback_chain: List of backends attempted (if fallback occurred).
        error_message: Error message if processing failed.
        metadata: Additional metadata from processing.
    """
    success: bool
    input_path: Path
    output_path: Path
    backend_used: Optional[BackendType] = None
    stages_applied: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    fallback_chain: List[BackendType] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "backend_used": self.backend_used.name if self.backend_used else None,
            "stages_applied": self.stages_applied,
            "processing_time_seconds": self.processing_time_seconds,
            "fallback_chain": [b.name for b in self.fallback_chain],
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class AudioBackend(ABC):
    """Abstract base class for audio processing backends."""

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def get_supported_features(self) -> List[EnhancementFeature]:
        """Return list of supported features."""
        pass

    @abstractmethod
    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: AudioConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        """Process audio file with this backend.

        Args:
            input_path: Path to input audio file.
            output_path: Path to output audio file.
            config: Enhancement configuration.
            progress_callback: Optional callback(stage, progress).

        Returns:
            AudioResult with processing details.
        """
        pass


class FFmpegAudioBackend(AudioBackend):
    """Basic FFmpeg-based audio processing backend.

    Always available when FFmpeg is installed. Provides basic audio
    extraction, filtering, and normalization.

    Wraps functionality from: audio.py
    """

    def __init__(self) -> None:
        self._processor = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FFMPEG

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from framewright.processors.audio import AudioProcessor
                self._processor = AudioProcessor()
                self._available = True
                logger.debug("FFmpeg audio backend is available")
            except Exception as e:
                logger.debug(f"FFmpeg audio backend not available: {e}")
                self._available = False
        return self._available

    def get_supported_features(self) -> List[EnhancementFeature]:
        return [
            EnhancementFeature.NOISE_REDUCTION,  # Basic via filters
            EnhancementFeature.NORMALIZATION,
        ]

    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: AudioConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        import time
        start_time = time.time()
        stages_applied = []

        if not self.is_available():
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                error_message="FFmpeg backend not available"
            )

        try:
            from framewright.processors.audio import AudioProcessor

            processor = AudioProcessor()
            current_path = input_path

            # Progress wrapper for the simple callback
            def simple_progress(msg: str) -> None:
                if progress_callback:
                    progress_callback(msg, 0.5)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                step = 0

                # Apply enhancement (combines filtering and normalization)
                if config.enable_noise_reduction or config.enable_normalization:
                    step += 1
                    temp_output = temp_dir_path / f"step{step}_enhanced.wav"

                    if progress_callback:
                        progress_callback("Applying FFmpeg enhancement", 0.3)

                    processor.enhance(
                        str(current_path),
                        str(temp_output),
                        progress_callback=simple_progress
                    )
                    current_path = temp_output
                    stages_applied.append("ffmpeg_enhance")

                # Apply specific normalization if different target
                if config.enable_normalization and config.target_loudness_lufs != -16.0:
                    step += 1
                    temp_output = temp_dir_path / f"step{step}_normalized.wav"

                    if progress_callback:
                        progress_callback("Normalizing loudness", 0.7)

                    processor.normalize(
                        str(current_path),
                        str(temp_output),
                        target_loudness=config.target_loudness_lufs,
                        progress_callback=simple_progress
                    )
                    current_path = temp_output
                    stages_applied.append("normalization")

                # Copy final result
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if current_path != input_path:
                    shutil.copy2(current_path, output_path)
                else:
                    shutil.copy2(input_path, output_path)

            return AudioResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=stages_applied,
                processing_time_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"FFmpeg backend processing failed: {e}")
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=stages_applied,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )


class TraditionalEnhancerBackend(AudioBackend):
    """Traditional FFmpeg-based enhancement backend.

    Provides comprehensive FFmpeg-based processing including noise reduction,
    declipping, hum removal, and EBU R128 normalization.

    Wraps functionality from: audio_enhance.py (TraditionalAudioEnhancer)
    """

    def __init__(self) -> None:
        self._enhancer = None
        self._analyzer = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.TRADITIONAL

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from framewright.processors.audio_enhance import (
                    TraditionalAudioEnhancer,
                    AudioAnalyzer
                )
                self._enhancer = TraditionalAudioEnhancer()
                self._analyzer = AudioAnalyzer()
                self._available = True
                logger.debug("Traditional audio enhancer backend is available")
            except Exception as e:
                logger.debug(f"Traditional audio enhancer not available: {e}")
                self._available = False
        return self._available

    def get_supported_features(self) -> List[EnhancementFeature]:
        return [
            EnhancementFeature.NOISE_REDUCTION,
            EnhancementFeature.DECLIPPING,
            EnhancementFeature.NORMALIZATION,
            EnhancementFeature.DEHUM,
        ]

    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: AudioConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        import time
        start_time = time.time()
        stages_applied = []

        if not self.is_available():
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                error_message="Traditional enhancer backend not available"
            )

        try:
            from framewright.processors.audio_enhance import (
                AudioEnhanceConfig,
                TraditionalAudioEnhancer,
            )

            # Convert unified config to audio_enhance config
            enhance_config = AudioEnhanceConfig(
                enable_noise_reduction=config.enable_noise_reduction,
                noise_reduction_strength=config.noise_reduction_strength,
                enable_declipping=config.enable_declipping,
                enable_dehum=config.enable_dehum,
                hum_frequency=config.hum_frequency if config.hum_frequency > 0 else 60,
                enable_normalization=config.enable_normalization,
                target_loudness=config.target_loudness_lufs,
                enable_ai_enhancement=False,  # AI handled by separate backend
                ai_model=config.ai_model,
            )

            # Simple progress callback adapter
            def simple_progress(msg: str) -> None:
                if progress_callback:
                    progress_callback(msg, 0.5)

            enhancer = TraditionalAudioEnhancer()
            result = enhancer.enhance_full_pipeline(
                str(input_path),
                str(output_path),
                enhance_config,
                progress_callback=simple_progress
            )

            return AudioResult(
                success=result.success,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=result.stages_applied,
                processing_time_seconds=time.time() - start_time,
                error_message=result.error_message,
            )

        except Exception as e:
            logger.error(f"Traditional enhancer processing failed: {e}")
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=stages_applied,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )


class AIAudioBackend(AudioBackend):
    """AI-powered audio enhancement backend.

    Provides neural network-based audio processing using denoiser and demucs
    models for superior restoration of heavily degraded audio.

    Wraps functionality from: audio_enhance.py (AIAudioEnhancer)
    """

    def __init__(self) -> None:
        self._ai_enhancer = None
        self._traditional_enhancer = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.AI

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from framewright.processors.audio_enhance import AIAudioEnhancer
                self._ai_enhancer = AIAudioEnhancer()
                self._available = self._ai_enhancer.is_available()
                if self._available:
                    logger.debug(
                        f"AI audio backend available with models: "
                        f"{self._ai_enhancer.get_available_models()}"
                    )
                else:
                    logger.debug("AI audio models not installed")
            except Exception as e:
                logger.debug(f"AI audio backend not available: {e}")
                self._available = False
        return self._available

    def get_supported_features(self) -> List[EnhancementFeature]:
        features = [
            EnhancementFeature.NOISE_REDUCTION,
            EnhancementFeature.SPEECH_ENHANCEMENT,
        ]
        # Add normalization if we can use traditional enhancer for post-processing
        try:
            from framewright.processors.audio_enhance import TraditionalAudioEnhancer
            TraditionalAudioEnhancer()
            features.append(EnhancementFeature.NORMALIZATION)
        except Exception:
            pass
        return features

    def get_available_models(self) -> List[str]:
        """Get list of available AI models."""
        if self.is_available() and self._ai_enhancer:
            return self._ai_enhancer.get_available_models()
        return []

    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: AudioConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        import time
        start_time = time.time()
        stages_applied = []

        if not self.is_available():
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                error_message="AI audio backend not available (install denoiser/demucs)"
            )

        try:
            from framewright.processors.audio_enhance import (
                AIAudioEnhancer,
                TraditionalAudioEnhancer,
            )

            ai_enhancer = AIAudioEnhancer()

            # Simple progress callback adapter
            def simple_progress(msg: str) -> None:
                if progress_callback:
                    progress_callback(msg, 0.5)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)

                # Apply AI enhancement
                if progress_callback:
                    progress_callback("Running AI enhancement", 0.2)

                ai_output = temp_dir_path / "ai_enhanced.wav"
                ai_success = False

                if config.ai_model == "speech":
                    ai_success = ai_enhancer.enhance_speech(
                        str(input_path),
                        str(ai_output),
                        progress_callback=simple_progress
                    )
                    if ai_success:
                        stages_applied.append("ai_speech_enhancement")
                elif config.ai_model == "music":
                    ai_success = ai_enhancer.enhance_music(
                        str(input_path),
                        str(ai_output),
                        progress_callback=simple_progress
                    )
                    if ai_success:
                        stages_applied.append("ai_music_enhancement")
                else:
                    ai_success = ai_enhancer.enhance_general(
                        str(input_path),
                        str(ai_output),
                        progress_callback=simple_progress
                    )
                    if ai_success:
                        stages_applied.append("ai_general_enhancement")

                if not ai_success:
                    return AudioResult(
                        success=False,
                        input_path=input_path,
                        output_path=output_path,
                        backend_used=self.backend_type,
                        stages_applied=stages_applied,
                        processing_time_seconds=time.time() - start_time,
                        error_message="AI enhancement failed"
                    )

                current_path = ai_output

                # Apply post-processing normalization if enabled
                if config.enable_normalization:
                    try:
                        if progress_callback:
                            progress_callback("Normalizing output", 0.8)

                        traditional = TraditionalAudioEnhancer()
                        traditional.normalize_loudness(
                            str(current_path),
                            str(output_path),
                            target_lufs=config.target_loudness_lufs,
                            progress_callback=simple_progress
                        )
                        stages_applied.append("normalization")
                    except Exception as e:
                        logger.warning(f"Post-normalization failed: {e}, copying AI output")
                        shutil.copy2(current_path, output_path)
                else:
                    shutil.copy2(current_path, output_path)

            return AudioResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=stages_applied,
                processing_time_seconds=time.time() - start_time,
                metadata={"ai_models": self.get_available_models()}
            )

        except Exception as e:
            logger.error(f"AI audio backend processing failed: {e}")
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=stages_applied,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )


class RestorationBackend(AudioBackend):
    """Full audio restoration suite backend.

    Provides comprehensive restoration including noise reduction, click removal,
    dialog enhancement, source separation, declipping, dereverb, and upmixing.

    Wraps functionality from: audio_restoration.py (AudioRestorer)
    """

    def __init__(self) -> None:
        self._restorer = None
        self._available: Optional[bool] = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.RESTORATION

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from framewright.processors.audio_restoration import AudioRestorer
                self._restorer = AudioRestorer()
                self._available = True
                logger.debug("Audio restoration backend is available")
            except Exception as e:
                logger.debug(f"Audio restoration backend not available: {e}")
                self._available = False
        return self._available

    def get_supported_features(self) -> List[EnhancementFeature]:
        return [
            EnhancementFeature.NOISE_REDUCTION,
            EnhancementFeature.DECLIPPING,
            EnhancementFeature.NORMALIZATION,
            EnhancementFeature.DEHUM,
            EnhancementFeature.DEREVERB,
            EnhancementFeature.DECLICK,
            EnhancementFeature.DIALOG_ENHANCE,
            EnhancementFeature.UPMIX,
        ]

    def process(
        self,
        input_path: Path,
        output_path: Path,
        config: AudioConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        import time
        start_time = time.time()

        if not self.is_available():
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                error_message="Restoration backend not available"
            )

        try:
            from framewright.processors.audio_restoration import (
                RestorationConfig,
                AudioRestorer,
            )

            # Convert unified config to restoration config
            restoration_config = RestorationConfig(
                enable_denoise=config.enable_noise_reduction,
                denoise_strength=config.noise_reduction_strength,
                enable_dehum=config.enable_dehum,
                hum_frequency=float(config.hum_frequency) if config.hum_frequency > 0 else 0.0,
                enable_declick=config.enable_declick,
                click_sensitivity=config.click_sensitivity,
                enable_dialog_enhance=config.enable_dialog_enhance,
                dialog_boost_db=config.dialog_boost_db,
                enable_declip=config.enable_declipping,
                enable_normalize=config.enable_normalization,
                target_loudness_lufs=config.target_loudness_lufs,
                enable_dereverb=config.enable_dereverb,
                dereverb_strength=config.dereverb_strength,
                enable_upmix=config.enable_upmix,
                output_sample_rate=config.output_sample_rate,
                output_bit_depth=config.output_bit_depth,
            )

            restorer = AudioRestorer(restoration_config)
            result = restorer.restore(
                input_path,
                output_path,
                progress_callback=progress_callback
            )

            return AudioResult(
                success=result.success,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                stages_applied=result.stages_applied,
                processing_time_seconds=time.time() - start_time,
                error_message="; ".join(result.warnings) if result.warnings else None,
                metadata={
                    "noise_reduction_db": result.noise_reduction_db,
                    "dynamic_range_improvement_db": result.dynamic_range_improvement_db,
                    "clicks_removed": result.clicks_removed,
                }
            )

        except Exception as e:
            logger.error(f"Restoration backend processing failed: {e}")
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_used=self.backend_type,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )


class AudioEnhancer:
    """Unified audio enhancement processor with multiple backends.

    Automatically selects the optimal backend based on:
    1. User preference (if specified)
    2. Feature requirements from configuration
    3. Available dependencies

    Provides graceful fallback chain when preferred backend fails.

    Backends (in order of capability/complexity):
    - RESTORATION: Full restoration suite (most features)
    - AI: Neural network enhancement (best for heavy degradation)
    - TRADITIONAL: FFmpeg-based enhancement (reliable, fast)
    - FFMPEG: Basic processing (always available)

    Example:
        >>> config = AudioConfig(
        ...     enable_noise_reduction=True,
        ...     noise_reduction_strength=0.7,
        ...     enable_normalization=True,
        ...     target_loudness_lufs=-14.0
        ... )
        >>> enhancer = AudioEnhancer(config)
        >>>
        >>> # Process a single file
        >>> result = enhancer.enhance(Path("input.wav"), Path("output.wav"))
        >>> print(f"Success: {result.success}, Backend: {result.backend_used.name}")
        >>>
        >>> # Check available backends
        >>> print(f"Available: {enhancer.get_available_backends()}")
    """

    # Backend classes in order of capability (most capable first)
    BACKENDS: Dict[BackendType, type] = {
        BackendType.RESTORATION: RestorationBackend,
        BackendType.AI: AIAudioBackend,
        BackendType.TRADITIONAL: TraditionalEnhancerBackend,
        BackendType.FFMPEG: FFmpegAudioBackend,
    }

    # Fallback order (from most capable to simplest)
    FALLBACK_ORDER: List[BackendType] = [
        BackendType.RESTORATION,
        BackendType.AI,
        BackendType.TRADITIONAL,
        BackendType.FFMPEG,
    ]

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        """Initialize the unified audio enhancer.

        Args:
            config: Enhancement configuration. If None, uses default config.
        """
        self.config = config or AudioConfig()
        self._backends: Dict[BackendType, AudioBackend] = {}
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize all available backends."""
        for backend_type, backend_class in self.BACKENDS.items():
            try:
                backend = backend_class()
                if backend.is_available():
                    self._backends[backend_type] = backend
                    logger.debug(f"Backend {backend_type.name} initialized and available")
            except Exception as e:
                logger.debug(f"Backend {backend_type.name} initialization failed: {e}")

    def get_available_backends(self) -> List[BackendType]:
        """Get list of available backends.

        Returns:
            List of available BackendType values.
        """
        return list(self._backends.keys())

    def get_backend(self, backend_type: BackendType) -> Optional[AudioBackend]:
        """Get a specific backend instance.

        Args:
            backend_type: The backend type to retrieve.

        Returns:
            Backend instance or None if not available.
        """
        return self._backends.get(backend_type)

    def _select_optimal_backend(self) -> Optional[BackendType]:
        """Select the optimal backend based on configuration and availability.

        Selection criteria:
        1. If preferred_backend is set and available, use it
        2. If AI enhancement is enabled and AI backend is available, prefer it
        3. Select based on required features
        4. Fall back to most capable available backend

        Returns:
            Selected BackendType or None if no backend is available.
        """
        # Check preferred backend first
        if self.config.preferred_backend:
            if self.config.preferred_backend in self._backends:
                return self.config.preferred_backend
            logger.warning(
                f"Preferred backend {self.config.preferred_backend.name} not available"
            )

        # If AI enhancement is explicitly requested
        if self.config.enable_ai_enhancement and BackendType.AI in self._backends:
            return BackendType.AI

        # Build list of required features
        required_features = []
        if self.config.enable_noise_reduction:
            required_features.append(EnhancementFeature.NOISE_REDUCTION)
        if self.config.enable_declipping:
            required_features.append(EnhancementFeature.DECLIPPING)
        if self.config.enable_normalization:
            required_features.append(EnhancementFeature.NORMALIZATION)
        if self.config.enable_dehum:
            required_features.append(EnhancementFeature.DEHUM)
        if self.config.enable_dereverb:
            required_features.append(EnhancementFeature.DEREVERB)
        if self.config.enable_declick:
            required_features.append(EnhancementFeature.DECLICK)
        if self.config.enable_dialog_enhance:
            required_features.append(EnhancementFeature.DIALOG_ENHANCE)
        if self.config.enable_upmix:
            required_features.append(EnhancementFeature.UPMIX)

        # Find backend that supports all required features
        for backend_type in self.FALLBACK_ORDER:
            if backend_type not in self._backends:
                continue
            backend = self._backends[backend_type]
            supported = set(backend.get_supported_features())
            if all(f in supported for f in required_features):
                return backend_type

        # No backend supports all features, use most capable available
        for backend_type in self.FALLBACK_ORDER:
            if backend_type in self._backends:
                logger.warning(
                    f"No backend supports all requested features. "
                    f"Using {backend_type.name} with partial support."
                )
                return backend_type

        return None

    def enhance(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        """Enhance audio file using optimal backend with fallback.

        Args:
            input_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional callback(stage, progress) for progress reporting.

        Returns:
            AudioResult with processing details.

        Example:
            >>> enhancer = AudioEnhancer(AudioConfig(enable_noise_reduction=True))
            >>> result = enhancer.enhance("noisy.wav", "clean.wav")
            >>> if result.success:
            ...     print(f"Processed with {result.backend_used.name}")
        """
        import time
        start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)

        # Validate input
        if not input_path.exists():
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                error_message=f"Input file does not exist: {input_path}"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select optimal backend
        selected_backend = self._select_optimal_backend()

        if not selected_backend:
            return AudioResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                error_message="No audio processing backends available"
            )

        fallback_chain = []

        # Try selected backend with fallback
        if self.config.fallback_enabled:
            # Build fallback chain starting from selected backend
            start_idx = self.FALLBACK_ORDER.index(selected_backend)
            backends_to_try = self.FALLBACK_ORDER[start_idx:]
        else:
            backends_to_try = [selected_backend]

        for backend_type in backends_to_try:
            if backend_type not in self._backends:
                continue

            fallback_chain.append(backend_type)
            backend = self._backends[backend_type]

            logger.info(f"Attempting audio enhancement with {backend_type.name} backend")

            if progress_callback:
                progress_callback(f"Processing with {backend_type.name}", 0.1)

            result = backend.process(
                input_path,
                output_path,
                self.config,
                progress_callback
            )

            if result.success:
                result.fallback_chain = fallback_chain
                result.processing_time_seconds = time.time() - start_time

                if len(fallback_chain) > 1:
                    logger.info(
                        f"Audio enhancement completed after fallback: "
                        f"{' -> '.join(b.name for b in fallback_chain)}"
                    )
                else:
                    logger.info(f"Audio enhancement completed with {backend_type.name}")

                return result

            logger.warning(
                f"Backend {backend_type.name} failed: {result.error_message}. "
                f"Trying fallback..."
            )

        # All backends failed
        return AudioResult(
            success=False,
            input_path=input_path,
            output_path=output_path,
            fallback_chain=fallback_chain,
            processing_time_seconds=time.time() - start_time,
            error_message="All backends failed to process audio"
        )

    def analyze(self, audio_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Analyze audio file for quality issues.

        Uses the most capable available analyzer backend.

        Args:
            audio_path: Path to audio file to analyze.

        Returns:
            Analysis results dictionary or None if analysis failed.
        """
        audio_path = Path(audio_path)

        # Try restoration backend analyzer first (most comprehensive)
        if BackendType.RESTORATION in self._backends:
            try:
                from framewright.processors.audio_restoration import AudioAnalyzer
                analyzer = AudioAnalyzer()
                analysis = analyzer.analyze(audio_path)
                return {
                    "sample_rate": analysis.sample_rate,
                    "channels": analysis.channels,
                    "duration_seconds": analysis.duration_seconds,
                    "bit_depth": analysis.bit_depth,
                    "peak_db": analysis.peak_db,
                    "rms_db": analysis.rms_db,
                    "dynamic_range_db": analysis.dynamic_range_db,
                    "noise_floor_db": analysis.noise_floor_db,
                    "has_clipping": analysis.has_clipping,
                    "clipping_percentage": analysis.clipping_percentage,
                    "has_hum": analysis.has_hum,
                    "hum_frequency": analysis.hum_frequency,
                    "quality": analysis.quality.value,
                    "summary": analysis.summary(),
                }
            except Exception as e:
                logger.warning(f"Restoration analyzer failed: {e}")

        # Try traditional enhancer analyzer
        if BackendType.TRADITIONAL in self._backends:
            try:
                from framewright.processors.audio_enhance import AudioAnalyzer
                analyzer = AudioAnalyzer()
                analysis = analyzer.analyze(str(audio_path))
                return {
                    "sample_rate": analysis.sample_rate,
                    "channels": analysis.channels,
                    "duration_seconds": analysis.duration_seconds,
                    "bit_depth": analysis.bit_depth,
                    "loudness_lufs": analysis.loudness_lufs,
                    "loudness_range_lu": analysis.loudness_range_lu,
                    "true_peak_dbtp": analysis.true_peak_dbtp,
                    "dynamic_range_db": analysis.dynamic_range_db,
                    "noise_level_db": analysis.noise_level_db,
                    "has_clipping": analysis.has_clipping,
                    "clipping_percentage": analysis.clipping_percentage,
                    "has_hum": analysis.has_hum,
                    "detected_hum_frequency": analysis.detected_hum_frequency,
                }
            except Exception as e:
                logger.warning(f"Traditional analyzer failed: {e}")

        logger.error("No audio analyzer available")
        return None

    def auto_enhance(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AudioResult:
        """Automatically analyze and enhance audio based on detected issues.

        Analyzes the input audio, generates optimal configuration based on
        detected quality issues, and applies enhancement.

        Args:
            input_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional callback for progress reporting.

        Returns:
            AudioResult with processing details.
        """
        import time
        start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)

        if progress_callback:
            progress_callback("Analyzing audio", 0.1)

        # Analyze input
        analysis = self.analyze(input_path)

        if analysis is None:
            logger.warning("Could not analyze audio, using default config")
            return self.enhance(input_path, output_path, progress_callback)

        # Build optimized config based on analysis
        auto_config = AudioConfig(
            # Noise reduction based on noise floor
            enable_noise_reduction=analysis.get("noise_floor_db", -60) > -50,
            noise_reduction_strength=min(1.0, max(0.3,
                (analysis.get("noise_floor_db", -60) + 60) / 30
            )),

            # Declipping if detected
            enable_declipping=analysis.get("has_clipping", False),

            # Always normalize
            enable_normalization=True,
            target_loudness_lufs=self.config.target_loudness_lufs,

            # Hum removal if detected
            enable_dehum=analysis.get("has_hum", False),
            hum_frequency=int(analysis.get("hum_frequency", 0) or
                             analysis.get("detected_hum_frequency", 0) or 0),

            # Preserve other settings from original config
            enable_dereverb=self.config.enable_dereverb,
            dereverb_strength=self.config.dereverb_strength,
            enable_declick=self.config.enable_declick,
            click_sensitivity=self.config.click_sensitivity,
            enable_dialog_enhance=self.config.enable_dialog_enhance,
            dialog_boost_db=self.config.dialog_boost_db,
            enable_ai_enhancement=self.config.enable_ai_enhancement,
            ai_model=self.config.ai_model,
            enable_upmix=self.config.enable_upmix,
            preferred_backend=self.config.preferred_backend,
            fallback_enabled=self.config.fallback_enabled,
            output_sample_rate=self.config.output_sample_rate,
            output_bit_depth=self.config.output_bit_depth,
        )

        # Temporarily use auto config
        original_config = self.config
        self.config = auto_config

        try:
            result = self.enhance(input_path, output_path, progress_callback)
            result.metadata["auto_analysis"] = analysis
            result.metadata["auto_config"] = {
                "noise_reduction_strength": auto_config.noise_reduction_strength,
                "enable_declipping": auto_config.enable_declipping,
                "enable_dehum": auto_config.enable_dehum,
            }
            return result
        finally:
            # Restore original config
            self.config = original_config


# Convenience functions

def enhance_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[AudioConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> AudioResult:
    """Convenience function for audio enhancement.

    Args:
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        config: Optional enhancement configuration.
        progress_callback: Optional progress callback.

    Returns:
        AudioResult with processing details.
    """
    enhancer = AudioEnhancer(config)
    return enhancer.enhance(input_path, output_path, progress_callback)


def auto_enhance_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> AudioResult:
    """Convenience function for automatic audio enhancement.

    Analyzes audio and applies appropriate enhancement automatically.

    Args:
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        progress_callback: Optional progress callback.

    Returns:
        AudioResult with processing details.
    """
    enhancer = AudioEnhancer()
    return enhancer.auto_enhance(input_path, output_path, progress_callback)


def get_available_backends() -> List[str]:
    """Get list of available audio backend names.

    Returns:
        List of available backend names.
    """
    enhancer = AudioEnhancer()
    return [b.name for b in enhancer.get_available_backends()]


__all__ = [
    # Main class
    "AudioEnhancer",
    # Configuration and results
    "AudioConfig",
    "AudioResult",
    "BackendType",
    "EnhancementFeature",
    # Backend classes
    "AudioBackend",
    "FFmpegAudioBackend",
    "TraditionalEnhancerBackend",
    "AIAudioBackend",
    "RestorationBackend",
    # Convenience functions
    "enhance_audio",
    "auto_enhance_audio",
    "get_available_backends",
]
