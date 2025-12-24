"""Audio processing module for video restoration pipeline.

This module provides audio extraction, enhancement, normalization, and filtering
capabilities using FFmpeg.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AudioProcessorError(Exception):
    """Base exception for audio processing errors."""
    pass


class AudioProcessor:
    """Handles audio extraction and enhancement operations.

    This class provides methods for extracting audio from video files,
    applying noise reduction, normalization, and various audio filters
    using FFmpeg as the backend processing engine.
    """

    def __init__(self):
        """Initialize the AudioProcessor.

        Raises:
            AudioProcessorError: If FFmpeg is not available in the system PATH.
        """
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify that FFmpeg is installed and accessible.

        Raises:
            AudioProcessorError: If FFmpeg is not found.
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"FFmpeg version: {result.stdout.split()[2]}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise AudioProcessorError(
                "FFmpeg is not installed or not found in PATH. "
                "Please install FFmpeg to use audio processing features."
            ) from e

    def _validate_input_file(self, file_path: str) -> Path:
        """Validate that input file exists and is accessible.

        Args:
            file_path: Path to the input file.

        Returns:
            Path object for the validated file.

        Raises:
            AudioProcessorError: If file doesn't exist or is not accessible.
        """
        path = Path(file_path)
        if not path.exists():
            raise AudioProcessorError(f"Input file does not exist: {file_path}")
        if not path.is_file():
            raise AudioProcessorError(f"Input path is not a file: {file_path}")
        return path

    def _ensure_output_dir(self, output_path: str) -> Path:
        """Ensure output directory exists.

        Args:
            output_path: Path to the output file.

        Returns:
            Path object for the output file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _run_ffmpeg(
        self,
        command: list[str],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Run FFmpeg command with error handling and progress reporting.

        Args:
            command: FFmpeg command as list of arguments.
            progress_callback: Optional callback function for progress updates.

        Raises:
            AudioProcessorError: If FFmpeg command fails.
        """
        logger.info(f"Running FFmpeg command: {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )

            # Read stderr for progress information
            for line in process.stderr:
                line = line.strip()
                if line:
                    logger.debug(f"FFmpeg: {line}")
                    if progress_callback:
                        progress_callback(line)

            return_code = process.wait()

            if return_code != 0:
                stdout, stderr = process.communicate()
                error_msg = f"FFmpeg command failed with return code {return_code}"
                if stderr:
                    error_msg += f"\nError output: {stderr}"
                raise AudioProcessorError(error_msg)

            logger.info("FFmpeg command completed successfully")

        except subprocess.SubprocessError as e:
            raise AudioProcessorError(f"Failed to execute FFmpeg: {str(e)}") from e

    def extract(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Extract audio from video file.

        Args:
            video_path: Path to input video file.
            output_path: Path to output audio file.
            progress_callback: Optional callback for progress updates.

        Raises:
            AudioProcessorError: If extraction fails.
        """
        logger.info(f"Extracting audio from {video_path} to {output_path}")

        self._validate_input_file(video_path)
        self._ensure_output_dir(output_path)

        command = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Use PCM for lossless extraction
            '-ar', '48000',  # 48kHz sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            str(output_path)
        ]

        self._run_ffmpeg(command, progress_callback)
        logger.info(f"Audio extracted successfully to {output_path}")

    def enhance(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply noise reduction and normalization to audio.

        This is a convenience method that combines denoising and normalization
        with default settings for general audio enhancement.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            progress_callback: Optional callback for progress updates.

        Raises:
            AudioProcessorError: If enhancement fails.
        """
        logger.info(f"Enhancing audio from {audio_path} to {output_path}")

        self._validate_input_file(audio_path)
        self._ensure_output_dir(output_path)

        command = [
            'ffmpeg',
            '-i', str(audio_path),
            # Apply highpass filter to remove low-frequency noise
            '-af', 'highpass=f=80,lowpass=f=12000,loudnorm=I=-16:TP=-1.5:LRA=11',
            '-ar', '48000',
            '-ac', '2',
            '-y',
            str(output_path)
        ]

        self._run_ffmpeg(command, progress_callback)
        logger.info(f"Audio enhanced successfully to {output_path}")

    def normalize(
        self,
        audio_path: str,
        output_path: str,
        target_loudness: float = -16.0,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply loudness normalization to audio.

        Uses FFmpeg's loudnorm filter to normalize audio to a target loudness
        level according to EBU R128 standard.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            target_loudness: Target integrated loudness in LUFS (default: -16.0).
            progress_callback: Optional callback for progress updates.

        Raises:
            AudioProcessorError: If normalization fails.
        """
        logger.info(
            f"Normalizing audio from {audio_path} to {output_path} "
            f"(target: {target_loudness} LUFS)"
        )

        self._validate_input_file(audio_path)
        self._ensure_output_dir(output_path)

        command = [
            'ffmpeg',
            '-i', str(audio_path),
            '-af', f'loudnorm=I={target_loudness}:TP=-1.5:LRA=11',
            '-ar', '48000',
            '-y',
            str(output_path)
        ]

        self._run_ffmpeg(command, progress_callback)
        logger.info(f"Audio normalized successfully to {output_path}")

    def denoise(
        self,
        audio_path: str,
        output_path: str,
        noise_floor: float = -20.0,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply FFT-based noise reduction to audio.

        Uses FFmpeg's afftdn (FFT denoise) filter to reduce background noise.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            noise_floor: Noise floor in dB (default: -20.0). Lower values are more aggressive.
            progress_callback: Optional callback for progress updates.

        Raises:
            AudioProcessorError: If denoising fails.
        """
        logger.info(
            f"Denoising audio from {audio_path} to {output_path} "
            f"(noise floor: {noise_floor} dB)"
        )

        self._validate_input_file(audio_path)
        self._ensure_output_dir(output_path)

        # Convert noise floor to noise reduction amount (0-97 dB)
        # More negative noise floor = more noise reduction
        noise_reduction = min(97, max(0, abs(noise_floor)))

        command = [
            'ffmpeg',
            '-i', str(audio_path),
            '-af', f'afftdn=nr={noise_reduction}:nf={noise_floor}',
            '-ar', '48000',
            '-y',
            str(output_path)
        ]

        self._run_ffmpeg(command, progress_callback)
        logger.info(f"Audio denoised successfully to {output_path}")

    def apply_filters(
        self,
        audio_path: str,
        output_path: str,
        highpass: int = 80,
        lowpass: int = 12000,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Apply frequency filtering to audio.

        Applies highpass and lowpass filters to remove unwanted frequencies.

        Args:
            audio_path: Path to input audio file.
            output_path: Path to output audio file.
            highpass: Highpass filter frequency in Hz (default: 80).
            lowpass: Lowpass filter frequency in Hz (default: 12000).
            progress_callback: Optional callback for progress updates.

        Raises:
            AudioProcessorError: If filtering fails.
        """
        logger.info(
            f"Applying filters to audio from {audio_path} to {output_path} "
            f"(highpass: {highpass} Hz, lowpass: {lowpass} Hz)"
        )

        self._validate_input_file(audio_path)
        self._ensure_output_dir(output_path)

        if highpass >= lowpass:
            raise AudioProcessorError(
                f"Highpass frequency ({highpass} Hz) must be less than "
                f"lowpass frequency ({lowpass} Hz)"
            )

        command = [
            'ffmpeg',
            '-i', str(audio_path),
            '-af', f'highpass=f={highpass},lowpass=f={lowpass}',
            '-ar', '48000',
            '-y',
            str(output_path)
        ]

        self._run_ffmpeg(command, progress_callback)
        logger.info(f"Filters applied successfully to {output_path}")
