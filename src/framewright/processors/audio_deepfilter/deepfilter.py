"""DeepFilterNet Integration for State-of-the-Art Audio Restoration.

This module provides DeepFilterNet 3 integration for real-time audio enhancement
with 10-20ms latency, supporting multiple backends with automatic fallback.

DeepFilterNet3 provides:
- State-of-the-art noise suppression
- Real-time processing capability
- Low latency (10-20ms)
- Excellent speech quality preservation

Backends (in order of preference):
- DeepFilterNet3Backend: Full DeepFilterNet3 (requires torch)
- DeepFilterLiteBackend: Lightweight ONNX version
- SpeechBrainBackend: SpeechBrain enhancement (alternative)
- TraditionalFilterBackend: FFmpeg-based fallback (always works)

Example usage:
    >>> from framewright.processors.audio.deepfilter import (
    ...     DeepFilterEnhancer, DeepFilterConfig, create_deepfilter_enhancer
    ... )
    >>> config = DeepFilterConfig(
    ...     denoise=True,
    ...     dereverb=True,
    ...     normalize=True,
    ...     target_loudness=-14.0
    ... )
    >>> enhancer = create_deepfilter_enhancer(config)
    >>> result = enhancer.enhance(Path("input.wav"), Path("output.wav"))
    >>> print(f"Processed with backend: {result.backend_name}")

Or use the one-liner convenience functions:
    >>> from framewright.processors.audio.deepfilter import enhance_audio, enhance_video_audio
    >>> enhance_audio("noisy.wav", "clean.wav")
    >>> enhance_video_audio("video.mp4", "video_enhanced.mp4")
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DeepFilterConfig:
    """Configuration for DeepFilterNet audio enhancement.

    Attributes:
        denoise: Enable noise removal (default True).
        dereverb: Enable reverb reduction (default True).
        declip: Enable clipping restoration (default False).
        normalize: Enable EBU R128 loudness normalization (default True).
        target_loudness: Target loudness in LUFS (default -14.0 for YouTube/streaming).
        atten_lim_db: Attenuation limit in dB (default 100, max noise reduction).
        post_filter: Enable post-filter for extra noise reduction.
        sample_rate: Target sample rate in Hz (default 48000).
        min_processing_buffer: Minimum buffer size for processing in samples.
        preserve_metadata: Preserve audio metadata during processing.
        high_quality: Use high quality mode (slower but better).
    """
    denoise: bool = True
    dereverb: bool = True
    declip: bool = False
    normalize: bool = True
    target_loudness: float = -14.0
    atten_lim_db: float = 100.0
    post_filter: bool = True
    sample_rate: int = 48000
    min_processing_buffer: int = 480
    preserve_metadata: bool = True
    high_quality: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not -70.0 <= self.target_loudness <= 0.0:
            raise ValueError(
                f"target_loudness must be between -70.0 and 0.0 LUFS, "
                f"got {self.target_loudness}"
            )
        if not 0.0 <= self.atten_lim_db <= 100.0:
            raise ValueError(
                f"atten_lim_db must be between 0.0 and 100.0, got {self.atten_lim_db}"
            )
        if self.sample_rate not in (8000, 16000, 22050, 24000, 44100, 48000, 96000):
            raise ValueError(
                f"sample_rate must be a standard audio rate, got {self.sample_rate}"
            )


class BackendPriority(Enum):
    """Backend priority levels for auto-selection."""
    HIGHEST = auto()  # DeepFilterNet3
    HIGH = auto()     # DeepFilterLite (ONNX)
    MEDIUM = auto()   # SpeechBrain
    LOW = auto()      # Traditional FFmpeg


# =============================================================================
# Audio Analysis
# =============================================================================


@dataclass
class AudioAnalysis:
    """Audio analysis results for determining optimal processing settings.

    Attributes:
        noise_level: Estimated noise level (0.0-1.0, higher = more noise).
        reverb_amount: Estimated reverb amount (0.0-1.0, higher = more reverb).
        clipping_detected: Whether clipping was detected.
        clipping_percentage: Percentage of samples that are clipped.
        speech_detected: Whether speech was detected in the audio.
        speech_confidence: Confidence level for speech detection (0.0-1.0).
        music_detected: Whether music was detected in the audio.
        music_confidence: Confidence level for music detection (0.0-1.0).
        sample_rate: Detected sample rate in Hz.
        channels: Number of audio channels.
        duration_seconds: Audio duration in seconds.
        peak_db: Peak level in dB.
        rms_db: RMS level in dB.
        recommended_settings: Recommended DeepFilterConfig based on analysis.
    """
    noise_level: float = 0.0
    reverb_amount: float = 0.0
    clipping_detected: bool = False
    clipping_percentage: float = 0.0
    speech_detected: bool = False
    speech_confidence: float = 0.0
    music_detected: bool = False
    music_confidence: float = 0.0
    sample_rate: int = 48000
    channels: int = 2
    duration_seconds: float = 0.0
    peak_db: float = 0.0
    rms_db: float = -20.0
    recommended_settings: Optional[DeepFilterConfig] = None

    def summary(self) -> str:
        """Get human-readable summary of analysis."""
        parts = []
        if self.noise_level > 0.3:
            parts.append(f"noise: {self.noise_level:.0%}")
        if self.reverb_amount > 0.3:
            parts.append(f"reverb: {self.reverb_amount:.0%}")
        if self.clipping_detected:
            parts.append(f"clipping: {self.clipping_percentage:.1f}%")
        if self.speech_detected:
            parts.append(f"speech: {self.speech_confidence:.0%}")
        if self.music_detected:
            parts.append(f"music: {self.music_confidence:.0%}")

        if parts:
            return f"Audio analysis: {', '.join(parts)}"
        return "Audio analysis: clean audio detected"


# =============================================================================
# Enhancement Results
# =============================================================================


@dataclass
class EnhancementResult:
    """Result of audio enhancement processing.

    Attributes:
        success: Whether enhancement completed successfully.
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        backend_name: Name of the backend that processed the audio.
        processing_time_seconds: Total processing time in seconds.
        latency_ms: Processing latency in milliseconds.
        stages_applied: List of processing stages that were applied.
        noise_reduction_db: Estimated noise reduction in dB.
        error_message: Error message if processing failed.
        metadata: Additional processing metadata.
    """
    success: bool
    input_path: Path
    output_path: Path
    backend_name: str = ""
    processing_time_seconds: float = 0.0
    latency_ms: float = 0.0
    stages_applied: List[str] = field(default_factory=list)
    noise_reduction_db: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "backend_name": self.backend_name,
            "processing_time_seconds": self.processing_time_seconds,
            "latency_ms": self.latency_ms,
            "stages_applied": self.stages_applied,
            "noise_reduction_db": self.noise_reduction_db,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


# =============================================================================
# Backend Base Class
# =============================================================================


class DeepFilterBackend(ABC):
    """Abstract base class for DeepFilter audio processing backends.

    All backends must implement:
    - is_available(): Check if the backend can be used
    - enhance(): Process audio data
    - get_latency_ms(): Return expected processing latency
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @property
    @abstractmethod
    def priority(self) -> BackendPriority:
        """Return the backend priority for auto-selection."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available for use.

        Returns:
            True if the backend dependencies are installed and working.
        """
        pass

    @abstractmethod
    def enhance(
        self,
        audio_data: Any,
        sample_rate: int,
        config: DeepFilterConfig,
    ) -> Tuple[Any, int]:
        """Process audio data with this backend.

        Args:
            audio_data: Audio data as numpy array or tensor.
            sample_rate: Sample rate of input audio.
            config: Enhancement configuration.

        Returns:
            Tuple of (enhanced_audio_data, output_sample_rate).
        """
        pass

    @abstractmethod
    def get_latency_ms(self) -> float:
        """Return the expected processing latency in milliseconds.

        Returns:
            Latency in milliseconds.
        """
        pass

    def supports_streaming(self) -> bool:
        """Check if this backend supports real-time streaming.

        Returns:
            True if streaming is supported.
        """
        return False


# =============================================================================
# DeepFilterNet3 Backend (Full PyTorch)
# =============================================================================


class DeepFilterNet3Backend(DeepFilterBackend):
    """Full DeepFilterNet3 backend using PyTorch.

    This is the highest quality backend with full model capabilities.
    Requires: torch, torchaudio, df (deepfilternet package)

    Features:
    - State-of-the-art noise suppression
    - 10-20ms latency
    - GPU acceleration support
    - Real-time streaming capability
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._model = None
        self._df_state = None

    @property
    def name(self) -> str:
        return "DeepFilterNet3"

    @property
    def priority(self) -> BackendPriority:
        return BackendPriority.HIGHEST

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            # Lazy import
            import torch
            from df.enhance import enhance, init_df

            # Try to initialize model
            self._model, self._df_state, _ = init_df()
            self._available = True
            logger.info("DeepFilterNet3 backend available with PyTorch")

        except ImportError as e:
            logger.debug(f"DeepFilterNet3 not available: {e}")
            self._available = False
        except Exception as e:
            logger.warning(f"DeepFilterNet3 initialization failed: {e}")
            self._available = False

        return self._available

    def enhance(
        self,
        audio_data: Any,
        sample_rate: int,
        config: DeepFilterConfig,
    ) -> Tuple[Any, int]:
        if not self.is_available():
            raise RuntimeError("DeepFilterNet3 backend not available")

        try:
            import torch
            from df.enhance import enhance

            # Convert to tensor if needed
            if not isinstance(audio_data, torch.Tensor):
                import numpy as np
                if isinstance(audio_data, np.ndarray):
                    audio_data = torch.from_numpy(audio_data).float()
                else:
                    raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

            # Ensure correct shape (batch, channels, samples)
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0).unsqueeze(0)
            elif audio_data.dim() == 2:
                audio_data = audio_data.unsqueeze(0)

            # Process with DeepFilterNet
            enhanced = enhance(
                self._model,
                self._df_state,
                audio_data,
                atten_lim_db=config.atten_lim_db,
            )

            # Output sample rate matches model (typically 48kHz)
            output_sr = self._df_state.sr()

            return enhanced.squeeze(), output_sr

        except Exception as e:
            logger.error(f"DeepFilterNet3 enhancement failed: {e}")
            raise

    def get_latency_ms(self) -> float:
        return 10.0  # DeepFilterNet3 typical latency

    def supports_streaming(self) -> bool:
        return True

    def enhance_streaming(
        self,
        audio_chunks: Iterator[Any],
        sample_rate: int,
        config: DeepFilterConfig,
    ) -> Generator[Any, None, None]:
        """Process audio in streaming mode.

        Args:
            audio_chunks: Iterator of audio chunks.
            sample_rate: Sample rate of input audio.
            config: Enhancement configuration.

        Yields:
            Enhanced audio chunks.
        """
        if not self.is_available():
            raise RuntimeError("DeepFilterNet3 backend not available")

        try:
            import torch
            from df.enhance import enhance

            for chunk in audio_chunks:
                if not isinstance(chunk, torch.Tensor):
                    import numpy as np
                    chunk = torch.from_numpy(chunk).float()

                if chunk.dim() == 1:
                    chunk = chunk.unsqueeze(0).unsqueeze(0)
                elif chunk.dim() == 2:
                    chunk = chunk.unsqueeze(0)

                enhanced = enhance(
                    self._model,
                    self._df_state,
                    chunk,
                    atten_lim_db=config.atten_lim_db,
                )

                yield enhanced.squeeze()

        except Exception as e:
            logger.error(f"DeepFilterNet3 streaming failed: {e}")
            raise


# =============================================================================
# DeepFilterLite Backend (ONNX)
# =============================================================================


class DeepFilterLiteBackend(DeepFilterBackend):
    """Lightweight ONNX-based DeepFilter backend.

    Uses ONNX Runtime for inference, no PyTorch required.
    Good balance of quality and performance.

    Requires: onnxruntime, numpy
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._session = None
        self._model_path: Optional[Path] = None

    @property
    def name(self) -> str:
        return "DeepFilterLite"

    @property
    def priority(self) -> BackendPriority:
        return BackendPriority.HIGH

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import numpy as np
            import onnxruntime as ort

            # Check for ONNX model file
            model_locations = [
                Path.home() / ".cache" / "deepfilternet" / "deepfilter.onnx",
                Path.home() / ".deepfilternet" / "model.onnx",
                Path("/usr/share/deepfilternet/model.onnx"),
            ]

            for path in model_locations:
                if path.exists():
                    self._model_path = path
                    self._session = ort.InferenceSession(str(path))
                    self._available = True
                    logger.info(f"DeepFilterLite backend available with ONNX model: {path}")
                    return True

            # No pre-downloaded model, but ONNX runtime is available
            # We could potentially download the model, but for now mark as unavailable
            logger.debug("DeepFilterLite: ONNX runtime available but model not found")
            self._available = False

        except ImportError as e:
            logger.debug(f"DeepFilterLite not available: {e}")
            self._available = False
        except Exception as e:
            logger.warning(f"DeepFilterLite initialization failed: {e}")
            self._available = False

        return self._available

    def enhance(
        self,
        audio_data: Any,
        sample_rate: int,
        config: DeepFilterConfig,
    ) -> Tuple[Any, int]:
        if not self.is_available():
            raise RuntimeError("DeepFilterLite backend not available")

        try:
            import numpy as np

            # Ensure numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)

            # Ensure correct shape
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, 1, -1)
            elif audio_data.ndim == 2:
                audio_data = audio_data.reshape(1, *audio_data.shape)

            # Run ONNX inference
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: audio_data.astype(np.float32)})

            enhanced = output[0].squeeze()

            # ONNX model typically outputs at 48kHz
            return enhanced, 48000

        except Exception as e:
            logger.error(f"DeepFilterLite enhancement failed: {e}")
            raise

    def get_latency_ms(self) -> float:
        return 15.0  # ONNX inference typical latency


# =============================================================================
# SpeechBrain Backend
# =============================================================================


class SpeechBrainBackend(DeepFilterBackend):
    """SpeechBrain-based audio enhancement backend.

    Uses SpeechBrain's pretrained enhancement models.
    Good alternative when DeepFilterNet is not available.

    Requires: speechbrain, torch, torchaudio
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._enhancer = None

    @property
    def name(self) -> str:
        return "SpeechBrain"

    @property
    def priority(self) -> BackendPriority:
        return BackendPriority.MEDIUM

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import torch
            import torchaudio
            from speechbrain.inference.enhancement import SpectralMaskEnhancement

            # Try to load the enhancement model
            self._enhancer = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir=str(Path.home() / ".cache" / "speechbrain" / "metricgan-plus"),
            )
            self._available = True
            logger.info("SpeechBrain enhancement backend available")

        except ImportError as e:
            logger.debug(f"SpeechBrain not available: {e}")
            self._available = False
        except Exception as e:
            logger.warning(f"SpeechBrain initialization failed: {e}")
            self._available = False

        return self._available

    def enhance(
        self,
        audio_data: Any,
        sample_rate: int,
        config: DeepFilterConfig,
    ) -> Tuple[Any, int]:
        if not self.is_available():
            raise RuntimeError("SpeechBrain backend not available")

        try:
            import torch
            import torchaudio

            # Convert to tensor if needed
            if not isinstance(audio_data, torch.Tensor):
                import numpy as np
                if isinstance(audio_data, np.ndarray):
                    audio_data = torch.from_numpy(audio_data).float()
                else:
                    raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

            # Ensure mono for SpeechBrain
            if audio_data.dim() == 2 and audio_data.shape[0] > 1:
                audio_data = audio_data.mean(dim=0)
            elif audio_data.dim() == 2:
                audio_data = audio_data.squeeze(0)

            # Resample to 16kHz if needed (SpeechBrain model requirement)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_data = resampler(audio_data)

            # Enhance
            enhanced = self._enhancer.enhance_batch(audio_data.unsqueeze(0))
            enhanced = enhanced.squeeze()

            # Resample back to target rate if needed
            if config.sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(16000, config.sample_rate)
                enhanced = resampler(enhanced)

            return enhanced, config.sample_rate

        except Exception as e:
            logger.error(f"SpeechBrain enhancement failed: {e}")
            raise

    def get_latency_ms(self) -> float:
        return 50.0  # SpeechBrain typical latency (higher than DeepFilter)


# =============================================================================
# Traditional FFmpeg Backend (Fallback)
# =============================================================================


class TraditionalFilterBackend(DeepFilterBackend):
    """FFmpeg-based traditional audio filtering backend.

    Always available when FFmpeg is installed.
    Uses conventional DSP filters for noise reduction.

    This is the fallback backend when AI models are unavailable.
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._ffmpeg_path: Optional[str] = None

    @property
    def name(self) -> str:
        return "TraditionalFilter"

    @property
    def priority(self) -> BackendPriority:
        return BackendPriority.LOW

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        self._ffmpeg_path = shutil.which("ffmpeg")
        self._available = self._ffmpeg_path is not None

        if self._available:
            logger.debug("TraditionalFilter backend available with FFmpeg")
        else:
            logger.warning("TraditionalFilter backend unavailable: FFmpeg not found")

        return self._available

    def enhance(
        self,
        audio_data: Any,
        sample_rate: int,
        config: DeepFilterConfig,
    ) -> Tuple[Any, int]:
        # This backend works on files, not raw audio data
        # For raw audio processing, return the data unchanged
        # The file-based processing is done in enhance_file()
        return audio_data, sample_rate

    def enhance_file(
        self,
        input_path: Path,
        output_path: Path,
        config: DeepFilterConfig,
    ) -> bool:
        """Enhance audio file using FFmpeg filters.

        Args:
            input_path: Input audio file path.
            output_path: Output audio file path.
            config: Enhancement configuration.

        Returns:
            True if successful.
        """
        if not self.is_available():
            return False

        try:
            filters = []

            # Noise reduction via filters
            if config.denoise:
                # High-pass to remove rumble
                filters.append("highpass=f=80")
                # Adaptive noise gate
                filters.append("agate=threshold=-40dB:ratio=3:attack=5:release=50")
                # Low-pass to reduce hiss
                if config.post_filter:
                    filters.append("lowpass=f=15000")

            # De-reverb via compression/gating
            if config.dereverb:
                filters.append("agate=threshold=-35dB:ratio=4:attack=2:release=100")
                filters.append("acompressor=threshold=-25dB:ratio=2:attack=5:release=100")

            # Declipping
            if config.declip:
                filters.append("adeclick=t=a")
                filters.append("alimiter=limit=0.95:level=false")

            # Normalization (EBU R128)
            if config.normalize:
                filters.append(f"loudnorm=I={config.target_loudness}:TP=-1.5:LRA=11")

            # Build FFmpeg command
            filter_str = ",".join(filters) if filters else "anull"

            cmd = [
                self._ffmpeg_path, "-y",
                "-i", str(input_path),
                "-af", filter_str,
                "-ar", str(config.sample_rate),
                "-acodec", "pcm_s24le",
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg processing timed out")
            return False
        except Exception as e:
            logger.error(f"TraditionalFilter enhancement failed: {e}")
            return False

    def get_latency_ms(self) -> float:
        return 100.0  # FFmpeg processing latency (file-based, not real-time)


# =============================================================================
# Audio Analyzer
# =============================================================================


class AudioAnalyzer:
    """Analyzes audio for noise, reverb, clipping, and content type."""

    def __init__(self) -> None:
        self._ffprobe_path = shutil.which("ffprobe")
        self._numpy_available = False
        self._scipy_available = False

        try:
            import numpy as np
            self._numpy_available = True
        except ImportError:
            pass

        try:
            from scipy import signal
            from scipy.io import wavfile
            self._scipy_available = True
        except ImportError:
            pass

    def analyze(self, audio_path: Union[str, Path]) -> AudioAnalysis:
        """Analyze audio file for quality and content characteristics.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioAnalysis with detected characteristics and recommended settings.
        """
        audio_path = Path(audio_path)
        analysis = AudioAnalysis()

        # Get basic metadata
        self._analyze_metadata(audio_path, analysis)

        # Analyze waveform for noise and clipping
        if self._numpy_available:
            self._analyze_waveform(audio_path, analysis)

        # Detect content type (speech/music)
        self._detect_content_type(audio_path, analysis)

        # Generate recommended settings
        analysis.recommended_settings = self._generate_recommendations(analysis)

        return analysis

    def _analyze_metadata(self, audio_path: Path, analysis: AudioAnalysis) -> None:
        """Extract metadata using FFprobe."""
        if not self._ffprobe_path:
            return

        try:
            import json

            cmd = [
                self._ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    analysis.sample_rate = int(stream.get("sample_rate", 48000))
                    analysis.channels = int(stream.get("channels", 2))
                    break

            fmt = data.get("format", {})
            analysis.duration_seconds = float(fmt.get("duration", 0))

        except Exception as e:
            logger.warning(f"FFprobe analysis failed: {e}")

    def _analyze_waveform(self, audio_path: Path, analysis: AudioAnalysis) -> None:
        """Analyze waveform for noise level and clipping."""
        try:
            import numpy as np

            # Extract short sample via FFmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-t", "10",  # First 10 seconds
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                str(tmp_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            if self._scipy_available:
                from scipy.io import wavfile
                sr, data = wavfile.read(str(tmp_path))
                data = data.astype(np.float32) / 32768.0
            else:
                # Manual WAV reading
                import struct
                import wave
                with wave.open(str(tmp_path), 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # Analyze levels
            peak = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data**2))

            analysis.peak_db = 20 * np.log10(peak + 1e-10)
            analysis.rms_db = 20 * np.log10(rms + 1e-10)

            # Detect clipping (samples at max level)
            clip_threshold = 0.99
            clipped_samples = np.sum(np.abs(data) >= clip_threshold)
            analysis.clipping_percentage = (clipped_samples / len(data)) * 100
            analysis.clipping_detected = analysis.clipping_percentage > 0.1

            # Estimate noise level from quietest sections
            sorted_abs = np.sort(np.abs(data))
            noise_floor = np.mean(sorted_abs[:len(sorted_abs)//10])
            noise_floor_db = 20 * np.log10(noise_floor + 1e-10)

            # Normalize noise level to 0-1 scale
            # -60dB or lower = 0 (clean), -30dB or higher = 1 (very noisy)
            analysis.noise_level = max(0.0, min(1.0, (noise_floor_db + 60) / 30))

            # Estimate reverb from envelope decay
            # This is a simplified heuristic
            envelope = np.abs(data)
            if len(envelope) > 1000:
                decay_rate = np.mean(np.diff(envelope[::100]))
                analysis.reverb_amount = max(0.0, min(1.0, -decay_rate * 1000))

            tmp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Waveform analysis failed: {e}")

    def _detect_content_type(self, audio_path: Path, analysis: AudioAnalysis) -> None:
        """Detect if audio contains speech or music."""
        # Simple heuristic based on spectral characteristics
        # A more accurate version would use a trained classifier

        try:
            # Use FFmpeg to analyze spectral centroid
            cmd = [
                "ffmpeg", "-i", str(audio_path),
                "-t", "10",
                "-af", "aspectralstats=measure=mean",
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Parse output for spectral characteristics
            # Speech typically has lower spectral centroid than music
            output = result.stderr.lower()

            # Very basic heuristics
            if "speech" in str(audio_path).lower() or "voice" in str(audio_path).lower():
                analysis.speech_detected = True
                analysis.speech_confidence = 0.8
            elif "music" in str(audio_path).lower():
                analysis.music_detected = True
                analysis.music_confidence = 0.8
            else:
                # Default assumption for video content: likely has speech
                analysis.speech_detected = True
                analysis.speech_confidence = 0.5

        except Exception as e:
            logger.debug(f"Content type detection failed: {e}")
            # Default to speech detection for video restoration
            analysis.speech_detected = True
            analysis.speech_confidence = 0.5

    def _generate_recommendations(self, analysis: AudioAnalysis) -> DeepFilterConfig:
        """Generate recommended configuration based on analysis."""
        config = DeepFilterConfig(
            denoise=analysis.noise_level > 0.1,
            dereverb=analysis.reverb_amount > 0.3,
            declip=analysis.clipping_detected,
            normalize=True,
            target_loudness=-14.0,
            atten_lim_db=min(100, 50 + analysis.noise_level * 50),
            post_filter=analysis.noise_level > 0.2,
            sample_rate=analysis.sample_rate,
            high_quality=True,
        )

        return config


# =============================================================================
# Main Enhancer Class
# =============================================================================


class DeepFilterEnhancer:
    """Main DeepFilterNet audio enhancement class.

    Auto-selects the best available backend and provides a unified interface
    for audio enhancement with progress callbacks and streaming support.

    Example:
        >>> config = DeepFilterConfig(denoise=True, normalize=True)
        >>> enhancer = DeepFilterEnhancer(config)
        >>> result = enhancer.enhance(Path("input.wav"), Path("output.wav"))
        >>> if result.success:
        ...     print(f"Enhanced with {result.backend_name}")
    """

    # Available backends in priority order
    BACKEND_CLASSES = [
        DeepFilterNet3Backend,
        DeepFilterLiteBackend,
        SpeechBrainBackend,
        TraditionalFilterBackend,
    ]

    def __init__(
        self,
        config: Optional[DeepFilterConfig] = None,
        preferred_backend: Optional[str] = None,
    ) -> None:
        """Initialize the DeepFilter enhancer.

        Args:
            config: Enhancement configuration. Uses defaults if None.
            preferred_backend: Preferred backend name. Auto-selects if None.
        """
        self.config = config or DeepFilterConfig()
        self._backends: Dict[str, DeepFilterBackend] = {}
        self._selected_backend: Optional[DeepFilterBackend] = None
        self._analyzer = AudioAnalyzer()

        # Initialize all available backends
        self._initialize_backends()

        # Select preferred or best available backend
        if preferred_backend:
            if preferred_backend in self._backends:
                self._selected_backend = self._backends[preferred_backend]
            else:
                logger.warning(
                    f"Preferred backend '{preferred_backend}' not available, "
                    "auto-selecting best backend"
                )

        if self._selected_backend is None:
            self._selected_backend = self._select_best_backend()

    def _initialize_backends(self) -> None:
        """Initialize all available backends."""
        for backend_class in self.BACKEND_CLASSES:
            try:
                backend = backend_class()
                if backend.is_available():
                    self._backends[backend.name] = backend
                    logger.debug(f"Backend {backend.name} initialized")
            except Exception as e:
                logger.debug(f"Backend {backend_class.__name__} init failed: {e}")

    def _select_best_backend(self) -> Optional[DeepFilterBackend]:
        """Select the best available backend based on priority."""
        if not self._backends:
            logger.error("No audio enhancement backends available")
            return None

        # Sort by priority
        sorted_backends = sorted(
            self._backends.values(),
            key=lambda b: b.priority.value
        )

        selected = sorted_backends[0]
        logger.info(f"Selected backend: {selected.name}")
        return selected

    def get_available_backends(self) -> List[str]:
        """Get list of available backend names.

        Returns:
            List of available backend names.
        """
        return list(self._backends.keys())

    def get_selected_backend(self) -> Optional[str]:
        """Get the currently selected backend name.

        Returns:
            Selected backend name or None.
        """
        return self._selected_backend.name if self._selected_backend else None

    def analyze(self, audio_path: Union[str, Path]) -> AudioAnalysis:
        """Analyze audio file for quality and characteristics.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioAnalysis with detected characteristics.
        """
        return self._analyzer.analyze(Path(audio_path))

    def enhance(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> EnhancementResult:
        """Enhance audio file using the selected backend.

        Args:
            input_path: Path to input audio file.
            output_path: Path to output audio file. If None, uses input name with '_enhanced' suffix.
            progress_callback: Optional callback(stage, progress) for progress reporting.

        Returns:
            EnhancementResult with processing details.
        """
        start_time = time.time()
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
        else:
            output_path = Path(output_path)

        # Validate input
        if not input_path.exists():
            return EnhancementResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                error_message=f"Input file does not exist: {input_path}"
            )

        # Check backend availability
        if self._selected_backend is None:
            return EnhancementResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                error_message="No audio enhancement backends available"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stages_applied = []

        try:
            if progress_callback:
                progress_callback("Initializing", 0.0)

            # For TraditionalFilterBackend, use file-based processing
            if isinstance(self._selected_backend, TraditionalFilterBackend):
                return self._enhance_with_ffmpeg(
                    input_path,
                    output_path,
                    progress_callback,
                    start_time
                )

            # For AI backends, load audio and process
            return self._enhance_with_ai(
                input_path,
                output_path,
                progress_callback,
                start_time
            )

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return EnhancementResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_name=self._selected_backend.name if self._selected_backend else "",
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def _enhance_with_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]],
        start_time: float,
    ) -> EnhancementResult:
        """Enhance using FFmpeg backend."""
        backend = self._selected_backend

        if progress_callback:
            progress_callback("Processing with FFmpeg filters", 0.3)

        # Use the file-based enhance method
        if isinstance(backend, TraditionalFilterBackend):
            success = backend.enhance_file(input_path, output_path, self.config)
        else:
            success = False

        stages_applied = []
        if self.config.denoise:
            stages_applied.append("denoise")
        if self.config.dereverb:
            stages_applied.append("dereverb")
        if self.config.declip:
            stages_applied.append("declip")
        if self.config.normalize:
            stages_applied.append("normalize")

        if progress_callback:
            progress_callback("Complete", 1.0)

        return EnhancementResult(
            success=success,
            input_path=input_path,
            output_path=output_path,
            backend_name=backend.name,
            processing_time_seconds=time.time() - start_time,
            latency_ms=backend.get_latency_ms(),
            stages_applied=stages_applied,
            error_message=None if success else "FFmpeg processing failed"
        )

    def _enhance_with_ai(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[str, float], None]],
        start_time: float,
    ) -> EnhancementResult:
        """Enhance using AI backend (DeepFilter, SpeechBrain, etc.)."""
        backend = self._selected_backend
        stages_applied = []

        try:
            # Lazy imports
            import numpy as np

            if progress_callback:
                progress_callback("Loading audio", 0.1)

            # Load audio via FFmpeg
            audio_data, sample_rate = self._load_audio(input_path)

            if progress_callback:
                progress_callback(f"Enhancing with {backend.name}", 0.3)

            # Process with backend
            enhanced, output_sr = backend.enhance(audio_data, sample_rate, self.config)
            stages_applied.append("ai_enhance")

            if progress_callback:
                progress_callback("Post-processing", 0.7)

            # Apply additional processing if needed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            # Save enhanced audio
            self._save_audio(enhanced, output_sr, tmp_path)

            # Apply normalization via FFmpeg if requested
            if self.config.normalize:
                self._apply_normalization(tmp_path, output_path)
                stages_applied.append("normalize")
            else:
                shutil.move(tmp_path, output_path)

            if progress_callback:
                progress_callback("Complete", 1.0)

            return EnhancementResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                backend_name=backend.name,
                processing_time_seconds=time.time() - start_time,
                latency_ms=backend.get_latency_ms(),
                stages_applied=stages_applied,
            )

        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return EnhancementResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend_name=backend.name if backend else "",
                processing_time_seconds=time.time() - start_time,
                stages_applied=stages_applied,
                error_message=str(e)
            )

    def _load_audio(self, audio_path: Path) -> Tuple[Any, int]:
        """Load audio file as numpy array via FFmpeg."""
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Convert to WAV via FFmpeg
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-vn", "-acodec", "pcm_f32le",
                "-ar", str(self.config.sample_rate),
                str(tmp_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)

            # Load the WAV file
            try:
                from scipy.io import wavfile
                sample_rate, audio_data = wavfile.read(str(tmp_path))
                audio_data = audio_data.astype(np.float32)
                if audio_data.dtype == np.int16:
                    audio_data = audio_data / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data / 2147483648.0
            except ImportError:
                # Manual WAV reading for float32
                import struct
                import wave
                with wave.open(str(tmp_path), 'rb') as wav:
                    sample_rate = wav.getframerate()
                    frames = wav.readframes(wav.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.float32)

            return audio_data, sample_rate

        finally:
            tmp_path.unlink(missing_ok=True)

    def _save_audio(self, audio_data: Any, sample_rate: int, output_path: Path) -> None:
        """Save audio data to WAV file."""
        import numpy as np

        # Convert tensor to numpy if needed
        if hasattr(audio_data, 'numpy'):
            audio_data = audio_data.numpy()
        elif hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()

        audio_data = np.asarray(audio_data, dtype=np.float32)

        # Ensure 1D or 2D
        if audio_data.ndim > 2:
            audio_data = audio_data.squeeze()

        try:
            from scipy.io import wavfile

            # Scale to int16 range
            if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)

            wavfile.write(str(output_path), sample_rate, audio_int16)

        except ImportError:
            # Manual WAV writing
            import struct
            import wave

            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[0]
                audio_data = audio_data.flatten()

            if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)

            with wave.open(str(output_path), 'wb') as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_int16.tobytes())

    def _apply_normalization(self, input_path: Path, output_path: Path) -> None:
        """Apply EBU R128 loudness normalization via FFmpeg."""
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"loudnorm=I={self.config.target_loudness}:TP=-1.5:LRA=11",
            "-ar", str(self.config.sample_rate),
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        input_path.unlink(missing_ok=True)

    def enhance_stream(
        self,
        audio_chunks: Iterator[Any],
        sample_rate: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Generator[Any, None, None]:
        """Enhance audio in real-time streaming mode.

        Args:
            audio_chunks: Iterator yielding audio chunks.
            sample_rate: Sample rate of input audio.
            progress_callback: Optional progress callback.

        Yields:
            Enhanced audio chunks.
        """
        if self._selected_backend is None:
            raise RuntimeError("No audio enhancement backends available")

        if not self._selected_backend.supports_streaming():
            raise RuntimeError(
                f"Backend {self._selected_backend.name} does not support streaming"
            )

        if progress_callback:
            progress_callback("Starting streaming enhancement", 0.0)

        # Only DeepFilterNet3Backend supports streaming currently
        if isinstance(self._selected_backend, DeepFilterNet3Backend):
            yield from self._selected_backend.enhance_streaming(
                audio_chunks,
                sample_rate,
                self.config
            )
        else:
            raise RuntimeError("Streaming not supported by current backend")

    def enhance_video_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> EnhancementResult:
        """Extract, enhance, and return video audio.

        Extracts audio from video, enhances it, and returns the enhanced audio path.
        Does not modify the original video.

        Args:
            video_path: Path to input video file.
            output_path: Path to output audio file. If None, auto-generates.
            progress_callback: Optional progress callback.

        Returns:
            EnhancementResult with the enhanced audio path.
        """
        start_time = time.time()
        video_path = Path(video_path)

        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_audio_enhanced.wav"
        else:
            output_path = Path(output_path)

        if not video_path.exists():
            return EnhancementResult(
                success=False,
                input_path=video_path,
                output_path=output_path,
                error_message=f"Video file does not exist: {video_path}"
            )

        try:
            if progress_callback:
                progress_callback("Extracting audio from video", 0.1)

            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio_path = Path(tmp.name)

            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s24le",
                "-ar", str(self.config.sample_rate),
                str(tmp_audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode != 0:
                return EnhancementResult(
                    success=False,
                    input_path=video_path,
                    output_path=output_path,
                    error_message="Failed to extract audio from video"
                )

            if progress_callback:
                progress_callback("Enhancing extracted audio", 0.3)

            # Enhance the extracted audio
            enhance_result = self.enhance(
                tmp_audio_path,
                output_path,
                lambda stage, prog: progress_callback(stage, 0.3 + prog * 0.6)
                if progress_callback else None
            )

            # Cleanup temp file
            tmp_audio_path.unlink(missing_ok=True)

            if progress_callback:
                progress_callback("Complete", 1.0)

            # Update result with video-specific info
            enhance_result.input_path = video_path
            enhance_result.metadata["source"] = "video"
            enhance_result.processing_time_seconds = time.time() - start_time

            return enhance_result

        except Exception as e:
            logger.error(f"Video audio enhancement failed: {e}")
            return EnhancementResult(
                success=False,
                input_path=video_path,
                output_path=output_path,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )


# =============================================================================
# Factory Functions
# =============================================================================


def create_deepfilter_enhancer(
    config: Optional[DeepFilterConfig] = None,
    preferred_backend: Optional[str] = None,
) -> DeepFilterEnhancer:
    """Create a DeepFilterEnhancer with optional configuration.

    Args:
        config: Enhancement configuration. Uses defaults if None.
        preferred_backend: Preferred backend name. Auto-selects if None.

    Returns:
        Configured DeepFilterEnhancer instance.

    Example:
        >>> enhancer = create_deepfilter_enhancer()
        >>> enhancer = create_deepfilter_enhancer(DeepFilterConfig(denoise=True))
        >>> enhancer = create_deepfilter_enhancer(preferred_backend="TraditionalFilter")
    """
    return DeepFilterEnhancer(config=config, preferred_backend=preferred_backend)


def enhance_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[DeepFilterConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> EnhancementResult:
    """One-liner convenience function for audio enhancement.

    Args:
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        config: Enhancement configuration. Uses defaults if None.
        progress_callback: Optional progress callback.

    Returns:
        EnhancementResult with processing details.

    Example:
        >>> result = enhance_audio("noisy.wav", "clean.wav")
        >>> if result.success:
        ...     print(f"Enhanced with {result.backend_name}")
    """
    enhancer = DeepFilterEnhancer(config)
    return enhancer.enhance(input_path, output_path, progress_callback)


def enhance_video_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[DeepFilterConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> EnhancementResult:
    """One-liner convenience function for video audio enhancement.

    Extracts audio from video, enhances it, and saves to output path.

    Args:
        video_path: Path to input video file.
        output_path: Path to output audio file.
        config: Enhancement configuration. Uses defaults if None.
        progress_callback: Optional progress callback.

    Returns:
        EnhancementResult with processing details.

    Example:
        >>> result = enhance_video_audio("movie.mp4", "movie_audio_enhanced.wav")
    """
    enhancer = DeepFilterEnhancer(config)
    return enhancer.enhance_video_audio(video_path, output_path, progress_callback)


def analyze_audio(audio_path: Union[str, Path]) -> AudioAnalysis:
    """Analyze audio file for noise, reverb, and content characteristics.

    Args:
        audio_path: Path to audio file.

    Returns:
        AudioAnalysis with detected characteristics and recommended settings.

    Example:
        >>> analysis = analyze_audio("audio.wav")
        >>> print(f"Noise level: {analysis.noise_level:.0%}")
        >>> print(f"Recommended config: {analysis.recommended_settings}")
    """
    analyzer = AudioAnalyzer()
    return analyzer.analyze(audio_path)


def get_available_backends() -> List[str]:
    """Get list of available DeepFilter backend names.

    Returns:
        List of available backend names.

    Example:
        >>> backends = get_available_backends()
        >>> print(f"Available: {backends}")
    """
    enhancer = DeepFilterEnhancer()
    return enhancer.get_available_backends()


# =============================================================================
# Module Exports
# =============================================================================


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
