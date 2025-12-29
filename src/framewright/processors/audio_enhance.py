"""Enhanced audio processing module for video restoration pipeline.

This module provides comprehensive audio restoration capabilities combining
traditional FFmpeg-based processing with optional AI-powered enhancement.

Traditional vs AI Enhancement:
-----------------------------
**Traditional (FFmpeg-based)**:
- Fast processing, reliable results
- Lower computational requirements
- Best for light cleanup and normalization
- Consistent, predictable output

**AI-based (denoiser/demucs)**:
- Slower processing, requires GPU
- Superior results for heavily degraded audio
- Best for old recordings, heavy noise, speech enhancement
- May introduce artifacts if overused

LUFS Targets by Platform:
------------------------
- YouTube: -14 LUFS
- Podcasts: -16 LUFS
- Broadcast (EBU R128): -23 LUFS
- Spotify: -14 LUFS
- Apple Music: -16 LUFS
- CD/Streaming: -14 to -16 LUFS

When to Use Each Approach:
-------------------------
- **Traditional only**: Light background noise, hum removal, volume normalization
- **AI + Traditional**: Old recordings, heavy degradation, speech clarity
- **AI for speech**: Interviews, documentaries, dialogue-heavy content
- **AI for music**: Music preservation where detail matters
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class AudioEnhanceError(Exception):
    """Base exception for audio enhancement errors."""
    pass


class AIModelType(str, Enum):
    """AI model types for audio enhancement."""
    SPEECH = "speech"
    MUSIC = "music"
    GENERAL = "general"


@dataclass
class AudioEnhanceConfig:
    """Configuration for audio enhancement pipeline.

    Attributes:
        enable_noise_reduction: Apply FFT-based noise reduction.
        noise_reduction_strength: Noise reduction intensity (0.0-1.0).
            0.0 = minimal, 1.0 = aggressive
        enable_declipping: Repair clipped/distorted audio peaks.
        enable_dehum: Remove electrical hum (50/60Hz).
        hum_frequency: Hum frequency to remove.
            50 Hz for Europe/Asia, 60 Hz for Americas.
        enable_normalization: Apply loudness normalization.
        target_loudness: Target loudness in LUFS.
            -14 for YouTube, -16 for podcast, -23 for broadcast.
        enable_ai_enhancement: Use AI models for enhancement.
        ai_model: AI model type (speech, music, general).

    Example:
        >>> config = AudioEnhanceConfig(
        ...     enable_noise_reduction=True,
        ...     noise_reduction_strength=0.6,
        ...     enable_dehum=True,
        ...     hum_frequency=60,
        ...     target_loudness=-14.0
        ... )
    """
    enable_noise_reduction: bool = True
    noise_reduction_strength: float = 0.5
    enable_declipping: bool = True
    enable_dehum: bool = True
    hum_frequency: int = 60
    enable_normalization: bool = True
    target_loudness: float = -14.0
    enable_ai_enhancement: bool = False
    ai_model: str = "speech"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.noise_reduction_strength <= 1.0:
            raise ValueError(
                f"noise_reduction_strength must be between 0.0 and 1.0, "
                f"got {self.noise_reduction_strength}"
            )

        if self.hum_frequency not in (50, 60):
            raise ValueError(
                f"hum_frequency must be 50 or 60 Hz, got {self.hum_frequency}"
            )

        if not -70.0 <= self.target_loudness <= 0.0:
            raise ValueError(
                f"target_loudness must be between -70.0 and 0.0 LUFS, "
                f"got {self.target_loudness}"
            )

        valid_models = [m.value for m in AIModelType]
        if self.ai_model not in valid_models:
            raise ValueError(
                f"ai_model must be one of {valid_models}, got {self.ai_model}"
            )


@dataclass
class AudioAnalysis:
    """Results from audio analysis.

    Attributes:
        noise_level_db: Estimated background noise level in dB.
        has_clipping: Whether audio contains clipped samples.
        clipping_percentage: Percentage of samples that are clipped.
        has_hum: Whether electrical hum is detected.
        detected_hum_frequency: Detected hum frequency (50 or 60 Hz).
        loudness_lufs: Integrated loudness in LUFS.
        loudness_range_lu: Loudness range in LU.
        true_peak_dbtp: True peak level in dBTP.
        dynamic_range_db: Dynamic range in dB.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        duration_seconds: Audio duration in seconds.
        bit_depth: Audio bit depth (if available).
        recommended_config: Recommended enhancement configuration.
    """
    noise_level_db: float
    has_clipping: bool
    clipping_percentage: float
    has_hum: bool
    detected_hum_frequency: Optional[int]
    loudness_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    dynamic_range_db: float
    sample_rate: int
    channels: int
    duration_seconds: float
    bit_depth: Optional[int] = None
    recommended_config: Optional[AudioEnhanceConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        result = {
            "noise_level_db": self.noise_level_db,
            "has_clipping": self.has_clipping,
            "clipping_percentage": self.clipping_percentage,
            "has_hum": self.has_hum,
            "detected_hum_frequency": self.detected_hum_frequency,
            "loudness_lufs": self.loudness_lufs,
            "loudness_range_lu": self.loudness_range_lu,
            "true_peak_dbtp": self.true_peak_dbtp,
            "dynamic_range_db": self.dynamic_range_db,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "bit_depth": self.bit_depth,
        }
        return result


@dataclass
class EnhancementResult:
    """Results from audio enhancement processing.

    Attributes:
        success: Whether enhancement completed successfully.
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        stages_applied: List of enhancement stages applied.
        processing_time_seconds: Total processing time.
        before_analysis: Audio analysis before enhancement.
        after_analysis: Audio analysis after enhancement (if available).
        ai_used: Whether AI enhancement was used.
        error_message: Error message if enhancement failed.
    """
    success: bool
    input_path: str
    output_path: str
    stages_applied: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    before_analysis: Optional[AudioAnalysis] = None
    after_analysis: Optional[AudioAnalysis] = None
    ai_used: bool = False
    error_message: Optional[str] = None


class TraditionalAudioEnhancer:
    """FFmpeg-based audio enhancement processor.

    Provides reliable, fast audio processing using FFmpeg filters:
    - afftdn: FFT-based noise reduction
    - highpass/lowpass: Frequency filtering for hum removal
    - loudnorm: EBU R128 loudness normalization
    - acompressor/dynaudnorm: Dynamic range control
    - declip: Audio declipping

    Example:
        >>> enhancer = TraditionalAudioEnhancer()
        >>> enhancer.reduce_noise("input.wav", "output.wav")
        >>> enhancer.normalize_loudness("input.wav", "output.wav", target_lufs=-14.0)
    """

    def __init__(self) -> None:
        """Initialize the traditional audio enhancer.

        Raises:
            AudioEnhanceError: If FFmpeg is not available.
        """
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available with required filters."""
        if not shutil.which("ffmpeg"):
            raise AudioEnhanceError(
                "FFmpeg is not installed or not in PATH. "
                "Please install FFmpeg to use audio enhancement."
            )

        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True,
                text=True,
                timeout=30
            )

            required_filters = ["afftdn", "loudnorm", "highpass", "lowpass"]
            available_filters = result.stdout

            for filter_name in required_filters:
                if filter_name not in available_filters:
                    logger.warning(
                        f"FFmpeg filter '{filter_name}' may not be available. "
                        "Some features may not work."
                    )

        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg filter check timed out")
        except Exception as e:
            logger.warning(f"Could not verify FFmpeg filters: {e}")

    def _validate_input(self, audio_path: str) -> Path:
        """Validate input file exists."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioEnhanceError(f"Input file does not exist: {audio_path}")
        if not path.is_file():
            raise AudioEnhanceError(f"Input path is not a file: {audio_path}")
        return path

    def _ensure_output_dir(self, output_path: str) -> Path:
        """Ensure output directory exists."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _run_ffmpeg(
        self,
        command: List[str],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> subprocess.CompletedProcess:
        """Run FFmpeg command with error handling.

        Args:
            command: FFmpeg command arguments.
            progress_callback: Optional progress callback.

        Returns:
            Completed process result.

        Raises:
            AudioEnhanceError: If command fails.
        """
        logger.debug(f"Running FFmpeg: {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stderr_lines = []
            for line in process.stderr:
                line = line.strip()
                if line:
                    stderr_lines.append(line)
                    logger.debug(f"FFmpeg: {line}")
                    if progress_callback:
                        progress_callback(line)

            return_code = process.wait()

            if return_code != 0:
                error_output = "\n".join(stderr_lines[-10:])
                raise AudioEnhanceError(
                    f"FFmpeg failed with code {return_code}: {error_output}"
                )

            return subprocess.CompletedProcess(
                command, return_code, "", "\n".join(stderr_lines)
            )

        except subprocess.SubprocessError as e:
            raise AudioEnhanceError(f"FFmpeg execution failed: {e}") from e

    def reduce_noise(
        self,
        audio_path: str,
        output_path: str,
        strength: float = 0.5,
        profile_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply FFT-based noise reduction.

        Uses FFmpeg's afftdn filter for spectral noise reduction.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            strength: Noise reduction strength (0.0-1.0).
            profile_path: Optional noise profile file (not currently used).
            progress_callback: Optional progress callback.

        Raises:
            AudioEnhanceError: If processing fails.
        """
        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        # Map strength (0-1) to noise reduction amount (0-97 dB)
        # and noise floor (-80 to -20 dB)
        nr_amount = int(strength * 50 + 10)  # 10-60 range
        noise_floor = -30 - (strength * 30)  # -30 to -60 range

        filter_str = f"afftdn=nr={nr_amount}:nf={noise_floor}:tn=1"

        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        logger.info(f"Reducing noise: strength={strength}")
        self._run_ffmpeg(command, progress_callback)

    def remove_hum(
        self,
        audio_path: str,
        output_path: str,
        frequency: int = 60,
        harmonics: int = 4,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Remove electrical hum and its harmonics.

        Uses a series of notch filters to remove the fundamental
        frequency and its harmonics.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            frequency: Hum frequency (50 or 60 Hz).
            harmonics: Number of harmonics to remove.
            progress_callback: Optional progress callback.

        Raises:
            AudioEnhanceError: If processing fails.
        """
        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        if frequency not in (50, 60):
            raise AudioEnhanceError(f"Hum frequency must be 50 or 60 Hz")

        # Build filter chain for fundamental and harmonics
        filters = []
        for i in range(1, harmonics + 1):
            freq = frequency * i
            if freq < 20000:  # Stay below Nyquist for 48kHz
                # Notch filter with Q factor of 5
                filters.append(f"equalizer=f={freq}:t=q:w=5:g=-30")

        filter_str = ",".join(filters)

        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        logger.info(f"Removing {frequency}Hz hum with {harmonics} harmonics")
        self._run_ffmpeg(command, progress_callback)

    def normalize_loudness(
        self,
        audio_path: str,
        output_path: str,
        target_lufs: float = -14.0,
        true_peak: float = -1.0,
        loudness_range: float = 11.0,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply EBU R128 loudness normalization.

        Uses FFmpeg's loudnorm filter for standard-compliant
        loudness normalization.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            target_lufs: Target integrated loudness in LUFS.
            true_peak: Maximum true peak level in dBTP.
            loudness_range: Target loudness range in LU.
            progress_callback: Optional progress callback.

        Common targets:
            - YouTube: -14 LUFS
            - Podcast: -16 LUFS
            - Broadcast: -23 LUFS

        Raises:
            AudioEnhanceError: If processing fails.
        """
        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        filter_str = (
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"
            ":print_format=summary"
        )

        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        logger.info(f"Normalizing loudness to {target_lufs} LUFS")
        self._run_ffmpeg(command, progress_callback)

    def declip(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Repair clipped audio samples.

        Uses FFmpeg's declip filter to reconstruct
        clipped audio peaks.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional progress callback.

        Raises:
            AudioEnhanceError: If processing fails.
        """
        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        # declip filter with default settings
        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", "adeclip=window=55:overlap=75:arorder=8:threshold=10",
            "-ar", "48000",
            str(output_path)
        ]

        logger.info("Applying declipping filter")
        self._run_ffmpeg(command, progress_callback)

    def apply_eq(
        self,
        audio_path: str,
        output_path: str,
        eq_settings: Dict[str, float],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply parametric equalization.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            eq_settings: Dictionary mapping frequency bands to gain in dB.
                Example: {"100": -3, "1000": 2, "8000": 1.5}
            progress_callback: Optional progress callback.

        Raises:
            AudioEnhanceError: If processing fails.
        """
        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        if not eq_settings:
            # No EQ settings, just copy
            shutil.copy2(audio_path, output_path)
            return

        # Build equalizer filter chain
        filters = []
        for freq_str, gain in eq_settings.items():
            freq = int(freq_str)
            if -30 <= gain <= 30:  # Reasonable gain range
                # Width (Q) of 1.4 is a good default
                filters.append(f"equalizer=f={freq}:t=q:w=1.4:g={gain}")

        if not filters:
            shutil.copy2(audio_path, output_path)
            return

        filter_str = ",".join(filters)

        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        logger.info(f"Applying EQ with {len(filters)} bands")
        self._run_ffmpeg(command, progress_callback)

    def apply_compression(
        self,
        audio_path: str,
        output_path: str,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 20.0,
        release: float = 250.0,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply dynamic range compression.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            threshold: Compression threshold in dB.
            ratio: Compression ratio (e.g., 4 means 4:1).
            attack: Attack time in milliseconds.
            release: Release time in milliseconds.
            progress_callback: Optional progress callback.

        Raises:
            AudioEnhanceError: If processing fails.
        """
        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        # Convert ms to seconds for FFmpeg
        attack_s = attack / 1000.0
        release_s = release / 1000.0

        filter_str = (
            f"acompressor=threshold={threshold}dB:ratio={ratio}:"
            f"attack={attack_s}:release={release_s}:makeup=2"
        )

        command = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        logger.info(f"Applying compression: threshold={threshold}dB, ratio={ratio}:1")
        self._run_ffmpeg(command, progress_callback)

    def enhance_full_pipeline(
        self,
        audio_path: str,
        output_path: str,
        config: AudioEnhanceConfig,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> EnhancementResult:
        """Apply full enhancement pipeline based on configuration.

        Processes audio through multiple stages in optimal order:
        1. Declipping (if enabled)
        2. Noise reduction (if enabled)
        3. Hum removal (if enabled)
        4. Normalization (if enabled)

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            config: Enhancement configuration.
            progress_callback: Optional progress callback.

        Returns:
            EnhancementResult with processing details.
        """
        import time
        start_time = time.time()

        self._validate_input(audio_path)
        self._ensure_output_dir(output_path)

        stages_applied = []
        current_input = audio_path

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                stage_num = 0

                # Stage 1: Declipping
                if config.enable_declipping:
                    stage_num += 1
                    temp_output = str(temp_dir_path / f"stage{stage_num}_declip.wav")
                    self.declip(current_input, temp_output, progress_callback)
                    current_input = temp_output
                    stages_applied.append("declip")

                # Stage 2: Noise reduction
                if config.enable_noise_reduction:
                    stage_num += 1
                    temp_output = str(temp_dir_path / f"stage{stage_num}_denoise.wav")
                    self.reduce_noise(
                        current_input,
                        temp_output,
                        strength=config.noise_reduction_strength,
                        progress_callback=progress_callback
                    )
                    current_input = temp_output
                    stages_applied.append("noise_reduction")

                # Stage 3: Hum removal
                if config.enable_dehum:
                    stage_num += 1
                    temp_output = str(temp_dir_path / f"stage{stage_num}_dehum.wav")
                    self.remove_hum(
                        current_input,
                        temp_output,
                        frequency=config.hum_frequency,
                        progress_callback=progress_callback
                    )
                    current_input = temp_output
                    stages_applied.append("dehum")

                # Stage 4: Normalization
                if config.enable_normalization:
                    stage_num += 1
                    temp_output = str(temp_dir_path / f"stage{stage_num}_normalize.wav")
                    self.normalize_loudness(
                        current_input,
                        temp_output,
                        target_lufs=config.target_loudness,
                        progress_callback=progress_callback
                    )
                    current_input = temp_output
                    stages_applied.append("normalization")

                # Copy final result to output
                if current_input != audio_path:
                    shutil.copy2(current_input, output_path)
                else:
                    # No processing was enabled, just copy
                    shutil.copy2(audio_path, output_path)

            processing_time = time.time() - start_time

            return EnhancementResult(
                success=True,
                input_path=audio_path,
                output_path=output_path,
                stages_applied=stages_applied,
                processing_time_seconds=processing_time,
                ai_used=False
            )

        except Exception as e:
            logger.error(f"Enhancement pipeline failed: {e}")
            return EnhancementResult(
                success=False,
                input_path=audio_path,
                output_path=output_path,
                stages_applied=stages_applied,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )


class AIAudioEnhancer:
    """AI-powered audio enhancement using deep learning models.

    Uses denoiser/demucs models for superior audio restoration,
    particularly effective for:
    - Old recordings with heavy degradation
    - Speech enhancement and clarity
    - Music preservation and restoration

    Falls back gracefully to traditional processing if AI
    models are not available.

    Example:
        >>> ai_enhancer = AIAudioEnhancer()
        >>> if ai_enhancer.is_available():
        ...     ai_enhancer.enhance_speech("input.wav", "output.wav")
        ... else:
        ...     print("AI enhancement not available, using traditional")
    """

    def __init__(self) -> None:
        """Initialize AI audio enhancer."""
        self._denoiser_available: Optional[bool] = None
        self._demucs_available: Optional[bool] = None

    def _check_denoiser(self) -> bool:
        """Check if denoiser package is available."""
        if self._denoiser_available is None:
            try:
                import denoiser
                self._denoiser_available = True
                logger.info("Denoiser AI model is available")
            except ImportError:
                self._denoiser_available = False
                logger.debug("Denoiser package not installed")
        return self._denoiser_available

    def _check_demucs(self) -> bool:
        """Check if demucs package is available."""
        if self._demucs_available is None:
            try:
                import demucs
                self._demucs_available = True
                logger.info("Demucs AI model is available")
            except ImportError:
                self._demucs_available = False
                logger.debug("Demucs package not installed")
        return self._demucs_available

    def is_available(self) -> bool:
        """Check if any AI enhancement model is available.

        Returns:
            True if at least one AI model is available.
        """
        return self._check_denoiser() or self._check_demucs()

    def get_available_models(self) -> List[str]:
        """Get list of available AI models.

        Returns:
            List of available model names.
        """
        models = []
        if self._check_denoiser():
            models.append("denoiser")
        if self._check_demucs():
            models.append("demucs")
        return models

    def enhance_speech(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Enhance speech audio using AI models.

        Uses denoiser model optimized for speech enhancement.
        Improves clarity, removes background noise, and
        enhances intelligibility.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional progress callback.

        Returns:
            True if AI enhancement succeeded, False otherwise.

        Note:
            Falls back to False if denoiser is not available.
            Caller should handle fallback to traditional processing.
        """
        if not self._check_denoiser():
            logger.warning("Denoiser not available for speech enhancement")
            return False

        try:
            import torch
            import torchaudio
            from denoiser import pretrained
            from denoiser.dsp import convert_audio

            logger.info("Loading denoiser model for speech enhancement")
            if progress_callback:
                progress_callback("Loading AI speech model...")

            # Load pretrained model
            model = pretrained.dns64()

            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            if progress_callback:
                progress_callback(f"Processing on {device}...")

            # Load audio
            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(device)

            # Convert to model's expected format
            wav = convert_audio(wav, sr, model.sample_rate, model.chin)

            # Process
            with torch.no_grad():
                denoised = model(wav[None])[0]

            # Save output
            denoised = denoised.cpu()
            torchaudio.save(output_path, denoised, model.sample_rate)

            logger.info("AI speech enhancement completed")
            if progress_callback:
                progress_callback("AI speech enhancement complete")

            return True

        except Exception as e:
            logger.error(f"AI speech enhancement failed: {e}")
            return False

    def enhance_music(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Enhance music audio using AI models.

        Uses demucs model for music enhancement and
        source separation if needed.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional progress callback.

        Returns:
            True if AI enhancement succeeded, False otherwise.
        """
        if not self._check_demucs():
            logger.warning("Demucs not available for music enhancement")
            return False

        try:
            # Demucs is primarily for source separation
            # For music enhancement, we use it to separate and remix
            # with cleaner artifacts

            logger.info("Loading demucs model for music enhancement")
            if progress_callback:
                progress_callback("Loading AI music model...")

            # For now, return False as demucs setup is more complex
            # and requires specific model downloads
            logger.warning(
                "Demucs music enhancement requires model setup. "
                "Falling back to traditional processing."
            )
            return False

        except Exception as e:
            logger.error(f"AI music enhancement failed: {e}")
            return False

    def enhance_general(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """General-purpose AI audio enhancement.

        Attempts speech enhancement first, then music enhancement.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional progress callback.

        Returns:
            True if any AI enhancement succeeded, False otherwise.
        """
        # Try speech enhancement first (more commonly useful)
        if self.enhance_speech(audio_path, output_path, progress_callback):
            return True

        # Fall back to music enhancement
        if self.enhance_music(audio_path, output_path, progress_callback):
            return True

        return False


class AudioAnalyzer:
    """Analyze audio files for quality assessment and enhancement recommendations.

    Provides detailed analysis of audio characteristics including:
    - Noise level estimation
    - Clipping detection
    - Hum detection
    - Loudness measurement (EBU R128)
    - Dynamic range analysis

    Example:
        >>> analyzer = AudioAnalyzer()
        >>> analysis = analyzer.analyze("audio.wav")
        >>> print(f"Loudness: {analysis.loudness_lufs} LUFS")
        >>> print(f"Recommended config: {analysis.recommended_config}")
    """

    def __init__(self) -> None:
        """Initialize audio analyzer."""
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available."""
        if not shutil.which("ffmpeg"):
            raise AudioEnhanceError(
                "FFmpeg is required for audio analysis"
            )

    def _run_ffmpeg_analysis(
        self,
        audio_path: str,
        filter_name: str,
        extra_args: Optional[List[str]] = None
    ) -> str:
        """Run FFmpeg analysis filter and capture output."""
        command = [
            "ffmpeg", "-i", str(audio_path),
            "-af", filter_name,
            "-f", "null", "-"
        ]

        if extra_args:
            command.extend(extra_args)

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

        return result.stderr

    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get basic audio file information using ffprobe."""
        command = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(audio_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise AudioEnhanceError(f"ffprobe failed: {result.stderr}")

        return json.loads(result.stdout)

    def _measure_loudness(self, audio_path: str) -> Dict[str, float]:
        """Measure EBU R128 loudness metrics."""
        output = self._run_ffmpeg_analysis(
            audio_path,
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json"
        )

        # Parse loudnorm JSON output from stderr
        loudness_data = {
            "input_i": -24.0,
            "input_tp": -1.0,
            "input_lra": 7.0,
            "input_thresh": -34.0
        }

        # Find JSON in output
        import re
        json_match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', output)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                loudness_data.update({
                    "input_i": float(parsed.get("input_i", -24.0)),
                    "input_tp": float(parsed.get("input_tp", -1.0)),
                    "input_lra": float(parsed.get("input_lra", 7.0)),
                    "input_thresh": float(parsed.get("input_thresh", -34.0))
                })
            except (json.JSONDecodeError, ValueError):
                pass

        return loudness_data

    def _detect_clipping(self, audio_path: str) -> Tuple[bool, float]:
        """Detect audio clipping."""
        output = self._run_ffmpeg_analysis(
            audio_path,
            "astats=metadata=1:reset=1"
        )

        # Look for peak levels close to 0 dB
        import re
        peak_match = re.search(r'Peak level dB:\s*([-\d.]+)', output)

        if peak_match:
            peak_db = float(peak_match.group(1))
            # Consider clipping if peak is above -0.3 dB
            has_clipping = peak_db > -0.3
            # Estimate clipping percentage (rough approximation)
            clipping_pct = max(0, (peak_db + 0.3) * 10) if has_clipping else 0
            return has_clipping, min(clipping_pct, 100)

        return False, 0.0

    def _estimate_noise_level(self, audio_path: str) -> float:
        """Estimate background noise level."""
        output = self._run_ffmpeg_analysis(
            audio_path,
            "astats=metadata=1:reset=1"
        )

        # Look for RMS level (rough noise estimate)
        import re
        rms_match = re.search(r'RMS level dB:\s*([-\d.]+)', output)

        if rms_match:
            rms_db = float(rms_match.group(1))
            # Noise floor is typically 20-30 dB below RMS
            return rms_db - 25

        return -60.0  # Default assumption

    def _detect_hum(self, audio_path: str) -> Tuple[bool, Optional[int]]:
        """Detect electrical hum (50/60 Hz)."""
        # Use FFT analysis to check for peaks at 50/60 Hz
        # This is a simplified detection

        output = self._run_ffmpeg_analysis(
            audio_path,
            "aspectralstats=measure=flat"
        )

        # For now, return a heuristic based on common issues
        # Real implementation would analyze frequency spectrum
        return False, None

    def analyze(self, audio_path: str) -> AudioAnalysis:
        """Perform comprehensive audio analysis.

        Args:
            audio_path: Path to audio file to analyze.

        Returns:
            AudioAnalysis with detailed metrics and recommendations.

        Raises:
            AudioEnhanceError: If analysis fails.
        """
        path = Path(audio_path)
        if not path.exists():
            raise AudioEnhanceError(f"File not found: {audio_path}")

        logger.info(f"Analyzing audio: {audio_path}")

        # Get basic info
        info = self._get_audio_info(audio_path)

        # Extract stream info
        audio_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            raise AudioEnhanceError("No audio stream found in file")

        # Get format info
        format_info = info.get("format", {})
        duration = float(format_info.get("duration", 0))

        # Get stream properties
        sample_rate = int(audio_stream.get("sample_rate", 48000))
        channels = int(audio_stream.get("channels", 2))
        bit_depth = audio_stream.get("bits_per_sample")
        if bit_depth:
            bit_depth = int(bit_depth)

        # Measure loudness
        loudness = self._measure_loudness(audio_path)

        # Detect clipping
        has_clipping, clipping_pct = self._detect_clipping(audio_path)

        # Estimate noise level
        noise_level = self._estimate_noise_level(audio_path)

        # Detect hum
        has_hum, hum_freq = self._detect_hum(audio_path)

        # Calculate dynamic range
        dynamic_range = loudness["input_lra"] * 1.5  # Approximation

        # Build analysis result
        analysis = AudioAnalysis(
            noise_level_db=noise_level,
            has_clipping=has_clipping,
            clipping_percentage=clipping_pct,
            has_hum=has_hum,
            detected_hum_frequency=hum_freq,
            loudness_lufs=loudness["input_i"],
            loudness_range_lu=loudness["input_lra"],
            true_peak_dbtp=loudness["input_tp"],
            dynamic_range_db=dynamic_range,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration,
            bit_depth=bit_depth
        )

        # Generate recommended configuration
        analysis.recommended_config = self._generate_recommendations(analysis)

        logger.info(f"Analysis complete: {analysis.loudness_lufs:.1f} LUFS")
        return analysis

    def _generate_recommendations(
        self,
        analysis: AudioAnalysis
    ) -> AudioEnhanceConfig:
        """Generate enhancement recommendations based on analysis.

        Args:
            analysis: Audio analysis results.

        Returns:
            Recommended AudioEnhanceConfig.
        """
        config = AudioEnhanceConfig()

        # Noise reduction recommendation
        if analysis.noise_level_db > -50:
            config.enable_noise_reduction = True
            # Scale strength based on noise level
            # -50dB = 0.3 strength, -30dB = 0.8 strength
            config.noise_reduction_strength = min(0.9, max(0.3,
                (analysis.noise_level_db + 50) / 25 * 0.5 + 0.3
            ))
        else:
            config.enable_noise_reduction = False

        # Declipping recommendation
        config.enable_declipping = analysis.has_clipping

        # Hum removal recommendation
        config.enable_dehum = analysis.has_hum
        if analysis.detected_hum_frequency:
            config.hum_frequency = analysis.detected_hum_frequency

        # Normalization recommendation (almost always beneficial)
        config.enable_normalization = True
        config.target_loudness = -14.0  # YouTube default

        # AI enhancement for heavily degraded audio
        if (analysis.noise_level_db > -40 or
            analysis.has_clipping and analysis.clipping_percentage > 5):
            config.enable_ai_enhancement = True
            config.ai_model = "speech"  # Default to speech

        return config


def create_audio_enhancer(
    use_ai: bool = False
) -> Tuple[TraditionalAudioEnhancer, Optional[AIAudioEnhancer]]:
    """Factory function to create audio enhancer instances.

    Args:
        use_ai: Whether to include AI enhancer.

    Returns:
        Tuple of (TraditionalAudioEnhancer, AIAudioEnhancer or None).
    """
    traditional = TraditionalAudioEnhancer()
    ai_enhancer = None

    if use_ai:
        ai_enhancer = AIAudioEnhancer()
        if not ai_enhancer.is_available():
            logger.warning(
                "AI enhancement requested but no AI models available. "
                "Install 'denoiser' package for AI features."
            )

    return traditional, ai_enhancer


def enhance_audio_auto(
    audio_path: str,
    output_path: str,
    config: Optional[AudioEnhanceConfig] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> EnhancementResult:
    """Convenience function for automatic audio enhancement.

    Analyzes audio, generates recommendations if config not provided,
    and applies appropriate enhancement pipeline.

    Args:
        audio_path: Path to input audio file.
        output_path: Path to output audio file.
        config: Optional enhancement configuration.
        progress_callback: Optional progress callback.

    Returns:
        EnhancementResult with processing details.

    Example:
        >>> result = enhance_audio_auto("input.wav", "output.wav")
        >>> if result.success:
        ...     print(f"Enhanced in {result.processing_time_seconds:.1f}s")
    """
    import time
    start_time = time.time()

    # Analyze audio if no config provided
    if config is None:
        analyzer = AudioAnalyzer()
        analysis = analyzer.analyze(audio_path)
        config = analysis.recommended_config
        logger.info(f"Using auto-detected config: noise_strength={config.noise_reduction_strength:.2f}")

    # Try AI enhancement if enabled
    if config.enable_ai_enhancement:
        ai_enhancer = AIAudioEnhancer()

        if ai_enhancer.is_available():
            if progress_callback:
                progress_callback("Attempting AI enhancement...")

            ai_success = False

            if config.ai_model == "speech":
                ai_success = ai_enhancer.enhance_speech(
                    audio_path, output_path, progress_callback
                )
            elif config.ai_model == "music":
                ai_success = ai_enhancer.enhance_music(
                    audio_path, output_path, progress_callback
                )
            else:
                ai_success = ai_enhancer.enhance_general(
                    audio_path, output_path, progress_callback
                )

            if ai_success:
                # Apply traditional post-processing
                traditional = TraditionalAudioEnhancer()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_path = tmp.name

                try:
                    # Apply normalization after AI processing
                    if config.enable_normalization:
                        traditional.normalize_loudness(
                            output_path,
                            temp_path,
                            target_lufs=config.target_loudness,
                            progress_callback=progress_callback
                        )
                        shutil.copy2(temp_path, output_path)
                finally:
                    Path(temp_path).unlink(missing_ok=True)

                return EnhancementResult(
                    success=True,
                    input_path=audio_path,
                    output_path=output_path,
                    stages_applied=["ai_enhancement", "normalization"],
                    processing_time_seconds=time.time() - start_time,
                    ai_used=True
                )
            else:
                logger.info("AI enhancement not available, using traditional")

    # Use traditional enhancement
    traditional = TraditionalAudioEnhancer()
    result = traditional.enhance_full_pipeline(
        audio_path,
        output_path,
        config,
        progress_callback
    )

    result.processing_time_seconds = time.time() - start_time
    return result


__all__ = [
    # Configuration
    "AudioEnhanceConfig",
    "AudioEnhanceError",
    "AIModelType",
    # Results
    "AudioAnalysis",
    "EnhancementResult",
    # Main classes
    "TraditionalAudioEnhancer",
    "AIAudioEnhancer",
    "AudioAnalyzer",
    # Factory and convenience functions
    "create_audio_enhancer",
    "enhance_audio_auto",
]
