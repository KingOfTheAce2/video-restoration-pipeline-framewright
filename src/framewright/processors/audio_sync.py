"""Audio-visual synchronization repair module for FrameWright.

This module provides audio-video synchronization detection and correction
capabilities using signal processing techniques for precise sync alignment.

Key Features:
-------------
- Audio peak/fingerprint analysis for sync detection
- Visual motion/scene change detection for sync reference points
- Cross-correlation based offset calculation
- Automatic drift correction with progressive drift detection
- Quality-preserving audio resampling and timing adjustment

Sync Drift Causes:
------------------
- Variable frame rate source material
- Telecine pulldown conversion
- Frame interpolation changing video duration
- Damaged film with missing frames
- Improper capture/digitization
- Audio/video clock mismatch during recording

Detection Methods:
-----------------
- Audio fingerprint: Uses audio peaks and transients
- Onset detection: Detects sound onsets (speech, music, effects)
- Scene detection: Correlates audio with scene changes and motion peaks
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from framewright.utils.dependencies import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger(__name__)

# Check for numpy availability
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("NumPy not available, using basic peak detection")


class AudioSyncError(Exception):
    """Base exception for audio sync errors."""
    pass


@dataclass
class SyncAnalysis:
    """Results from audio-video sync analysis.

    Attributes:
        drift_ms: Average drift in milliseconds.
            Positive = audio leads video, Negative = audio lags video.
        drift_direction: Direction of drift ("audio_ahead" or "video_ahead").
        confidence: Confidence score for the detected offset (0.0-1.0).
        samples_analyzed: Number of audio/video event pairs analyzed.
        is_progressive: True if drift increases over time (clock mismatch).
    """
    drift_ms: float
    drift_direction: str  # "audio_ahead" or "video_ahead"
    confidence: float
    samples_analyzed: int
    is_progressive: bool  # True if drift increases over time

    def __post_init__(self) -> None:
        """Validate and normalize fields."""
        # Ensure drift_direction is valid
        if self.drift_direction not in ("audio_ahead", "video_ahead"):
            # Auto-determine from drift_ms
            self.drift_direction = "audio_ahead" if self.drift_ms > 0 else "video_ahead"

        # Clamp confidence to valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def needs_correction(self) -> bool:
        """Check if sync correction is recommended.

        Returns:
            True if drift exceeds threshold and confidence is sufficient.
        """
        return abs(self.drift_ms) > 40.0 and self.confidence >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "drift_ms": self.drift_ms,
            "drift_direction": self.drift_direction,
            "confidence": self.confidence,
            "samples_analyzed": self.samples_analyzed,
            "is_progressive": self.is_progressive,
            "needs_correction": self.needs_correction,
        }


@dataclass
class SyncConfig:
    """Configuration for audio sync detection and correction.

    Attributes:
        detection_method: Method for sync detection.
            "audio_fingerprint" - Uses audio peaks and transients
            "onset_detection" - Uses sound onset detection (librosa)
        max_drift_ms: Maximum drift to attempt to correct (milliseconds).
        correction_method: Method for applying sync correction.
            "resample" - Resample audio to match video duration
            "frame_shift" - Shift audio start/end points
            "pts_adjust" - Adjust presentation timestamps (fastest)
    """
    detection_method: str = "audio_fingerprint"
    max_drift_ms: float = 500.0
    correction_method: str = "resample"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_detection = ("audio_fingerprint", "onset_detection")
        if self.detection_method not in valid_detection:
            raise ValueError(
                f"Invalid detection_method: {self.detection_method}. "
                f"Must be one of {valid_detection}"
            )

        valid_correction = ("resample", "frame_shift", "pts_adjust")
        if self.correction_method not in valid_correction:
            raise ValueError(
                f"Invalid correction_method: {self.correction_method}. "
                f"Must be one of {valid_correction}"
            )

        if self.max_drift_ms <= 0:
            raise ValueError("max_drift_ms must be positive")


# Legacy dataclasses for backward compatibility
@dataclass
class AudioWaveformInfo:
    """Information about audio waveform characteristics.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        duration: Audio duration in seconds.
        peak_positions: List of peak sample positions.
        rms_levels: RMS level per analysis window.
        channels: Number of audio channels.
        bit_depth: Audio bit depth (if available).
    """
    sample_rate: int
    duration: float
    peak_positions: List[int] = field(default_factory=list)
    rms_levels: List[float] = field(default_factory=list)
    channels: int = 2
    bit_depth: Optional[int] = None

    def peak_times(self) -> List[float]:
        """Convert peak positions to timestamps.

        Returns:
            List of peak timestamps in seconds.
        """
        return [pos / self.sample_rate for pos in self.peak_positions]


@dataclass
class SyncCorrection:
    """Results from audio sync correction.

    Attributes:
        original_duration: Original audio duration in seconds.
        adjusted_duration: Adjusted audio duration in seconds.
        stretch_factor: Time stretch factor applied.
            >1.0 = audio slowed down, <1.0 = audio sped up.
        offset_applied_ms: Offset correction applied in milliseconds.
        output_path: Path to corrected audio file.
        quality_preserved: Whether quality was preserved during adjustment.
    """
    original_duration: float
    adjusted_duration: float
    stretch_factor: float
    offset_applied_ms: float = 0.0
    output_path: str = ""
    quality_preserved: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert correction to dictionary."""
        return {
            "original_duration": self.original_duration,
            "adjusted_duration": self.adjusted_duration,
            "stretch_factor": self.stretch_factor,
            "offset_applied_ms": self.offset_applied_ms,
            "output_path": self.output_path,
            "quality_preserved": self.quality_preserved,
        }


class AudioSyncAnalyzer:
    """Analyzes audio-video synchronization using signal processing.

    Detects sync drift by analyzing audio peaks/fingerprints and
    correlating with visual motion events and scene changes.

    Example:
        >>> analyzer = AudioSyncAnalyzer()
        >>> analysis = analyzer.analyze_sync(Path("video.mp4"))
        >>> if analysis.needs_correction:
        ...     print(f"Drift: {analysis.drift_ms}ms ({analysis.drift_direction})")
    """

    # Threshold for considering sync correction needed (in ms)
    SYNC_THRESHOLD_MS = 40.0

    # Minimum confidence for valid detection
    MIN_CONFIDENCE = 0.5

    def __init__(self, config: Optional[SyncConfig] = None) -> None:
        """Initialize the sync analyzer.

        Args:
            config: Optional SyncConfig for customizing detection.
        """
        self.config = config or SyncConfig()
        self._verify_ffmpeg()
        self._librosa_available = self._check_librosa()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available."""
        try:
            get_ffmpeg_path()
        except FileNotFoundError as e:
            raise AudioSyncError(
                "FFmpeg is not installed or not in PATH. "
                "Please install FFmpeg for audio sync detection."
            ) from e

    def _check_librosa(self) -> bool:
        """Check if librosa is available for advanced onset detection."""
        try:
            import librosa
            return True
        except ImportError:
            logger.debug("librosa not available, using FFmpeg for onset detection")
            return False

    def analyze_sync(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SyncAnalysis:
        """Analyze audio-video synchronization.

        Extracts audio and video events, then calculates the drift
        between them using cross-correlation.

        Args:
            video_path: Path to video file to analyze.
            progress_callback: Optional callback for progress updates.

        Returns:
            SyncAnalysis with drift information.

        Raises:
            AudioSyncError: If analysis fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioSyncError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing sync for: {video_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "extracted_audio.wav"

            if progress_callback:
                progress_callback("Extracting audio...")

            # Extract audio from video
            self._extract_audio(video_path, audio_path)

            if progress_callback:
                progress_callback("Detecting audio peaks...")

            # Detect audio peaks/events
            audio_events = self._detect_audio_peaks(audio_path)

            if progress_callback:
                progress_callback("Detecting video events...")

            # Detect video events (scene changes, motion peaks)
            video_events = self._detect_video_events(video_path)

            if progress_callback:
                progress_callback("Calculating drift...")

            # Calculate drift
            drift_ms, confidence = self._calculate_drift(audio_events, video_events)

            # Check for progressive drift
            is_progressive = self._check_progressive_drift(
                audio_events, video_events, audio_path
            )

            # Determine drift direction
            drift_direction = "audio_ahead" if drift_ms > 0 else "video_ahead"

            samples_analyzed = min(len(audio_events), len(video_events))

            if progress_callback:
                progress_callback("Analysis complete")

            return SyncAnalysis(
                drift_ms=drift_ms,
                drift_direction=drift_direction,
                confidence=confidence,
                samples_analyzed=samples_analyzed,
                is_progressive=is_progressive
            )

    def _extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        sample_rate: int = 48000
    ) -> None:
        """Extract audio from video to WAV format.

        Args:
            video_path: Path to input video file.
            output_path: Path for output WAV file.
            sample_rate: Target sample rate.

        Raises:
            AudioSyncError: If extraction fails.
        """
        command = [
            get_ffmpeg_path(), "-y",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono for analysis
            str(output_path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise AudioSyncError(f"Audio extraction failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Audio extraction timed out")

    def _detect_audio_peaks(self, audio_path: Path) -> List[float]:
        """Detect audio peaks and transients in audio file.

        Uses either librosa onset detection or FFmpeg-based peak detection
        depending on the configured method and available libraries.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of peak timestamps in seconds.

        Raises:
            AudioSyncError: If detection fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioSyncError(f"Audio file not found: {audio_path}")

        logger.info(f"Detecting audio peaks in {audio_path}")

        if self.config.detection_method == "onset_detection" and self._librosa_available:
            return self._detect_peaks_librosa(audio_path)
        else:
            return self._detect_peaks_ffmpeg(audio_path)

    def _detect_peaks_librosa(self, audio_path: Path) -> List[float]:
        """Detect audio peaks using librosa onset detection.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of peak timestamps.
        """
        import librosa

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        events = librosa.frames_to_time(onset_frames, sr=sr)

        logger.info(f"Detected {len(events)} audio peaks (librosa)")
        return events.tolist()

    def _detect_peaks_ffmpeg(self, audio_path: Path) -> List[float]:
        """Detect audio peaks using FFmpeg silencedetect filter.

        Falls back to this when librosa is not available. Uses silence
        boundaries as proxy for audio events.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of peak timestamps.
        """
        # Use silencedetect to find non-silent segments (onsets)
        command = [
            get_ffmpeg_path(),
            "-i", str(audio_path),
            "-af", "silencedetect=noise=-40dB:d=0.05",
            "-f", "null", "-"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse silence end times (these are onset times)
        events = []
        for line in result.stderr.split('\n'):
            if 'silence_end' in line:
                try:
                    parts = line.split('silence_end:')
                    if len(parts) > 1:
                        time_str = parts[1].strip().split()[0]
                        events.append(float(time_str))
                except (ValueError, IndexError):
                    continue

        # Add beginning if audio starts with sound
        if events and events[0] > 0.1:
            events.insert(0, 0.0)

        # If too few events, use volume-based peak detection
        if len(events) < 10:
            volume_events = self._detect_volume_peaks(audio_path)
            events.extend(volume_events)
            events = sorted(set(events))

        logger.info(f"Detected {len(events)} audio peaks (FFmpeg)")
        return events

    def _detect_volume_peaks(self, audio_path: Path) -> List[float]:
        """Detect volume peaks using FFmpeg volumedetect.

        Analyzes audio volume over time to find significant peaks.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of peak timestamps.
        """
        if not NUMPY_AVAILABLE:
            return []

        # Extract raw audio samples
        command = [
            get_ffmpeg_path(),
            "-i", str(audio_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", "8000",  # Lower sample rate for faster processing
            "pipe:1"
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=300
            )

            if result.returncode != 0:
                return []

            audio = np.frombuffer(result.stdout, dtype=np.float32)
            sr = 8000

            # Calculate envelope
            window_size = int(sr * 0.05)  # 50ms windows
            envelope = []
            times = []

            for i in range(0, len(audio) - window_size, window_size):
                window = audio[i:i + window_size]
                envelope.append(np.sqrt(np.mean(window ** 2)))
                times.append(i / sr)

            if not envelope:
                return []

            envelope = np.array(envelope)
            threshold = np.mean(envelope) + np.std(envelope) * 1.5

            # Find peaks above threshold
            peaks = []
            min_distance = int(0.2 / (window_size / sr))  # 200ms minimum

            for i in range(1, len(envelope) - 1):
                if envelope[i] > threshold:
                    if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
                        if not peaks or (i - peaks[-1]) >= min_distance:
                            peaks.append(i)

            return [times[p] for p in peaks]

        except Exception as e:
            logger.warning(f"Volume peak detection failed: {e}")
            return []

    def _detect_video_events(self, video_path: Path) -> List[float]:
        """Detect video events (scene changes, motion peaks).

        Uses FFmpeg scene detection and motion analysis to find
        visual events that can correlate with audio.

        Args:
            video_path: Path to video file.

        Returns:
            List of event timestamps in seconds.

        Raises:
            AudioSyncError: If detection fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioSyncError(f"Video file not found: {video_path}")

        logger.info(f"Detecting video events in {video_path}")

        events = []

        # Scene change detection using FFmpeg
        scene_events = self._detect_scene_changes(video_path)
        events.extend(scene_events)

        # Motion peak detection
        motion_events = self._detect_motion_peaks(video_path)
        events.extend(motion_events)

        # Sort and remove duplicates (within 0.1s tolerance)
        events = sorted(set(events))
        if len(events) > 1:
            filtered = [events[0]]
            for e in events[1:]:
                if e - filtered[-1] > 0.1:
                    filtered.append(e)
            events = filtered

        logger.info(f"Detected {len(events)} video events")
        return events

    def _detect_scene_changes(self, video_path: Path) -> List[float]:
        """Detect scene changes using FFmpeg.

        Args:
            video_path: Path to video file.

        Returns:
            List of scene change timestamps.
        """
        command = [
            get_ffmpeg_path(),
            "-i", str(video_path),
            "-vf", "select='gt(scene,0.3)',showinfo",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600
            )
        except subprocess.TimeoutExpired:
            logger.warning("Scene detection timed out")
            return []

        events = []
        for line in result.stderr.split('\n'):
            if 'pts_time' in line:
                try:
                    parts = line.split('pts_time:')
                    if len(parts) > 1:
                        time_str = parts[1].strip().split()[0]
                        events.append(float(time_str))
                except (ValueError, IndexError):
                    continue

        return events

    def _detect_motion_peaks(self, video_path: Path) -> List[float]:
        """Detect motion peaks in video using frame differencing.

        Uses a lower threshold than scene detection to capture
        more subtle motion events.

        Args:
            video_path: Path to video file.

        Returns:
            List of motion peak timestamps.
        """
        command = [
            get_ffmpeg_path(),
            "-i", str(video_path),
            "-vf", "select='gt(scene,0.1)',showinfo",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600
            )
        except subprocess.TimeoutExpired:
            logger.warning("Motion detection timed out")
            return []

        events = []
        for line in result.stderr.split('\n'):
            if 'pts_time' in line:
                try:
                    parts = line.split('pts_time:')
                    if len(parts) > 1:
                        time_str = parts[1].strip().split()[0]
                        events.append(float(time_str))
                except (ValueError, IndexError):
                    continue

        return events

    def _calculate_drift(
        self,
        audio_events: List[float],
        video_events: List[float]
    ) -> Tuple[float, float]:
        """Calculate drift between audio and video events.

        Uses cross-correlation to find the offset that best aligns
        audio and visual events.

        Args:
            audio_events: List of audio event timestamps (seconds).
            video_events: List of video event timestamps (seconds).

        Returns:
            Tuple of (drift_ms, confidence).
            Positive drift means audio leads video.
        """
        if not audio_events or not video_events:
            logger.warning("Insufficient events for drift calculation")
            return 0.0, 0.0

        logger.info(
            f"Calculating drift from {len(audio_events)} audio and "
            f"{len(video_events)} video events"
        )

        if not NUMPY_AVAILABLE:
            return self._calculate_drift_basic(audio_events, video_events)

        # Convert to numpy arrays
        audio = np.array(audio_events)
        video = np.array(video_events)

        # Search range
        max_offset_s = self.config.max_drift_ms / 1000.0
        step_s = 0.001  # 1ms resolution
        offsets = np.arange(-max_offset_s, max_offset_s, step_s)

        best_offset = 0.0
        best_score = 0.0

        # For each offset, count how many audio events match video events
        tolerance = 0.05  # 50ms tolerance for matching

        for offset in offsets:
            shifted_audio = audio + offset
            matches = 0

            for a_time in shifted_audio:
                if len(video) > 0:
                    closest_idx = np.argmin(np.abs(video - a_time))
                    if np.abs(video[closest_idx] - a_time) < tolerance:
                        matches += 1

            # Normalize score
            score = matches / max(len(audio_events), 1)

            if score > best_score:
                best_score = score
                best_offset = offset

        # Convert to milliseconds
        drift_ms = best_offset * 1000.0

        logger.info(
            f"Best drift: {drift_ms:.1f}ms "
            f"(confidence: {best_score:.3f})"
        )

        return drift_ms, best_score

    def _calculate_drift_basic(
        self,
        audio_events: List[float],
        video_events: List[float]
    ) -> Tuple[float, float]:
        """Calculate drift without numpy (basic implementation).

        Args:
            audio_events: List of audio event timestamps.
            video_events: List of video event timestamps.

        Returns:
            Tuple of (drift_ms, confidence).
        """
        max_offset_s = self.config.max_drift_ms / 1000.0
        step_s = 0.005  # 5ms resolution for basic method
        tolerance = 0.05

        best_offset = 0.0
        best_score = 0.0

        offset = -max_offset_s
        while offset <= max_offset_s:
            matches = 0

            for a_time in audio_events:
                shifted = a_time + offset
                for v_time in video_events:
                    if abs(shifted - v_time) < tolerance:
                        matches += 1
                        break

            score = matches / max(len(audio_events), 1)

            if score > best_score:
                best_score = score
                best_offset = offset

            offset += step_s

        return best_offset * 1000.0, best_score

    def _check_progressive_drift(
        self,
        audio_events: List[float],
        video_events: List[float],
        audio_path: Path
    ) -> bool:
        """Check if drift is progressive (increases over time).

        Compares drift in first and second halves of content to
        detect clock mismatch or progressive sync loss.

        Args:
            audio_events: Audio event timestamps.
            video_events: Video event timestamps.
            audio_path: Path to audio file for duration.

        Returns:
            True if drift is progressive.
        """
        if len(audio_events) < 10 or len(video_events) < 10:
            return False

        # Get audio duration
        try:
            duration = self._get_duration(audio_path)
        except Exception:
            duration = max(
                max(audio_events) if audio_events else 0,
                max(video_events) if video_events else 0
            )

        if duration <= 0:
            return False

        midpoint = duration / 2

        # Split events
        audio_first = [e for e in audio_events if e < midpoint]
        audio_second = [e for e in audio_events if e >= midpoint]
        video_first = [e for e in video_events if e < midpoint]
        video_second = [e for e in video_events if e >= midpoint]

        # Calculate drift for each half
        drift_first, _ = self._calculate_drift(audio_first, video_first)
        drift_second, _ = self._calculate_drift(audio_second, video_second)

        # Check if drift increased significantly
        drift_change = abs(drift_second - drift_first)
        is_progressive = drift_change > 20.0  # 20ms threshold

        logger.info(
            f"Drift analysis: first half={drift_first:.1f}ms, "
            f"second half={drift_second:.1f}ms, "
            f"progressive={is_progressive}"
        )

        return is_progressive

    def _get_duration(self, file_path: Path) -> float:
        """Get media duration in seconds."""
        command = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(file_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))


class AudioSyncCorrector:
    """Corrects audio-video synchronization issues.

    Applies timing adjustments to audio, handles time stretching
    for duration matching, and preserves audio quality during correction.

    Example:
        >>> corrector = AudioSyncCorrector()
        >>> analysis = SyncAnalysis(
        ...     drift_ms=-100,
        ...     drift_direction="video_ahead",
        ...     confidence=0.85,
        ...     samples_analyzed=50,
        ...     is_progressive=False
        ... )
        >>> output = corrector.correct_sync(
        ...     Path("video.mp4"),
        ...     analysis,
        ...     Path("output.mp4")
        ... )
    """

    # Maximum stretch factor before quality concerns
    MAX_STRETCH_FACTOR = 1.05  # 5% stretch/compress

    def __init__(self, config: Optional[SyncConfig] = None) -> None:
        """Initialize the sync corrector.

        Args:
            config: Optional SyncConfig for customizing correction.
        """
        self.config = config or SyncConfig()
        self._verify_ffmpeg()
        self._rubberband_available = self._check_rubberband()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available with required filters."""
        try:
            ffmpeg = get_ffmpeg_path()
        except FileNotFoundError as e:
            raise AudioSyncError(
                "FFmpeg is not installed. Required for audio sync correction."
            ) from e

        # Check for required filters
        result = subprocess.run(
            [ffmpeg, "-filters"],
            capture_output=True,
            text=True,
            timeout=30
        )

        required = ["atempo", "adelay"]
        for filter_name in required:
            if filter_name not in result.stdout:
                logger.warning(f"FFmpeg filter '{filter_name}' may not be available")

    def _check_rubberband(self) -> bool:
        """Check if rubberband is available for high-quality time stretching."""
        result = subprocess.run(
            [get_ffmpeg_path(), "-filters"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return "rubberband" in result.stdout

    def correct_sync(
        self,
        video_path: Path,
        analysis: SyncAnalysis,
        output_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Path:
        """Apply sync correction to video based on analysis.

        Corrects audio-video synchronization by adjusting audio timing
        according to the detected drift.

        Args:
            video_path: Path to input video file.
            analysis: SyncAnalysis from AudioSyncAnalyzer.
            output_path: Path for output corrected video.
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to corrected video file.

        Raises:
            AudioSyncError: If correction fails or drift exceeds limits.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise AudioSyncError(f"Video file not found: {video_path}")

        # Check if drift exceeds maximum
        if abs(analysis.drift_ms) > self.config.max_drift_ms:
            raise AudioSyncError(
                f"Drift ({analysis.drift_ms}ms) exceeds maximum "
                f"({self.config.max_drift_ms}ms)"
            )

        if not analysis.needs_correction:
            logger.info("No sync correction needed according to analysis")
            shutil.copy2(video_path, output_path)
            return output_path

        logger.info(
            f"Applying sync correction: {analysis.drift_ms}ms "
            f"({analysis.drift_direction})"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if progress_callback:
                progress_callback("Extracting audio...")

            # Extract audio
            audio_path = temp_path / "original_audio.wav"
            self._extract_audio(video_path, audio_path)

            if progress_callback:
                progress_callback(f"Adjusting audio timing ({analysis.drift_ms}ms)...")

            # Adjust audio timing based on correction method
            adjusted_audio = temp_path / "adjusted_audio.wav"

            if self.config.correction_method == "resample":
                self._resample_audio(
                    audio_path,
                    analysis.drift_ms,
                    adjusted_audio
                )
            elif self.config.correction_method == "frame_shift":
                self._adjust_audio_timing(
                    audio_path,
                    analysis.drift_ms,
                    adjusted_audio
                )
            else:  # pts_adjust
                self._adjust_audio_timing(
                    audio_path,
                    analysis.drift_ms,
                    adjusted_audio
                )

            if progress_callback:
                progress_callback("Remuxing video...")

            # Remux video with adjusted audio
            self._remux_video(video_path, adjusted_audio, output_path)

        logger.info(f"Sync correction complete: {output_path}")
        return output_path

    def _extract_audio(self, video_path: Path, output_path: Path) -> None:
        """Extract audio from video file."""
        command = [
            get_ffmpeg_path(), "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise AudioSyncError(f"Audio extraction failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Audio extraction timed out")

    def _adjust_audio_timing(
        self,
        audio_path: Path,
        drift_ms: float,
        output_path: Path
    ) -> None:
        """Adjust audio timing by adding delay or trimming.

        Args:
            audio_path: Path to input audio file.
            drift_ms: Drift to correct in milliseconds.
                Positive = audio leads (add delay)
                Negative = audio lags (trim start)
            output_path: Path for output audio file.
        """
        if abs(drift_ms) < 1:
            shutil.copy2(audio_path, output_path)
            return

        if drift_ms > 0:
            # Audio leads video - add delay at start
            delay_samples = int(drift_ms)
            filter_str = f"adelay={delay_samples}|{delay_samples}"
        else:
            # Audio lags video - trim from start
            trim_seconds = abs(drift_ms) / 1000.0
            filter_str = f"atrim=start={trim_seconds}"

        command = [
            get_ffmpeg_path(), "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise AudioSyncError(f"Audio timing adjustment failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Audio timing adjustment timed out")

    def _resample_audio(
        self,
        audio_path: Path,
        drift_ms: float,
        output_path: Path
    ) -> None:
        """Resample audio to correct drift using time stretching.

        Uses time stretching to subtly adjust audio speed to match
        video timing. Preserves pitch where possible.

        Args:
            audio_path: Path to input audio file.
            drift_ms: Drift to correct in milliseconds.
            output_path: Path for output audio file.
        """
        # Get audio duration
        duration = self._get_duration(audio_path)

        if duration <= 0:
            shutil.copy2(audio_path, output_path)
            return

        # Calculate stretch factor
        # If audio leads (positive drift), we need to slow it down slightly
        # If audio lags (negative drift), we need to speed it up slightly
        drift_seconds = drift_ms / 1000.0
        target_duration = duration + drift_seconds
        stretch_factor = target_duration / duration

        # atempo uses speed factor (inverse of stretch)
        tempo_factor = 1.0 / stretch_factor

        logger.info(f"Resampling audio: tempo factor = {tempo_factor:.4f}")

        # Build filter chain
        if self._rubberband_available and abs(tempo_factor - 1.0) < 0.1:
            # Use rubberband for high quality on small adjustments
            filter_str = f"rubberband=tempo={tempo_factor}:pitch=1.0"
        else:
            # Use atempo (may need chaining for large values)
            filter_parts = []
            remaining = tempo_factor

            while remaining > 2.0:
                filter_parts.append("atempo=2.0")
                remaining /= 2.0
            while remaining < 0.5:
                filter_parts.append("atempo=0.5")
                remaining /= 0.5

            if 0.5 <= remaining <= 2.0:
                filter_parts.append(f"atempo={remaining}")

            filter_str = ",".join(filter_parts) if filter_parts else "anull"

        command = [
            get_ffmpeg_path(), "-y",
            "-i", str(audio_path),
            "-af", filter_str,
            "-ar", "48000",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                raise AudioSyncError(f"Audio resampling failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Audio resampling timed out")

    def _remux_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> None:
        """Remux video with new audio track.

        Args:
            video_path: Path to input video file.
            audio_path: Path to audio file.
            output_path: Path for output video file.
        """
        command = [
            get_ffmpeg_path(), "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v",  # Video from first input
            "-map", "1:a",  # Audio from second input
            "-c:v", "copy",  # Copy video codec
            "-c:a", "aac",  # Encode audio as AAC
            "-b:a", "192k",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                raise AudioSyncError(f"Video remuxing failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Video remuxing timed out")

    def _get_duration(self, file_path: Path) -> float:
        """Get media duration in seconds."""
        command = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(file_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))


# Legacy class aliases for backward compatibility
AudioSyncDetector = AudioSyncAnalyzer


# Factory functions

def analyze_av_sync(video_path: Path) -> SyncAnalysis:
    """Analyze audio-visual synchronization for a video file.

    Factory function for quick sync analysis.

    Args:
        video_path: Path to video file.

    Returns:
        SyncAnalysis with drift information.

    Example:
        >>> analysis = analyze_av_sync(Path("video.mp4"))
        >>> if analysis.needs_correction:
        ...     print(f"Drift: {analysis.drift_ms}ms ({analysis.drift_direction})")
    """
    analyzer = AudioSyncAnalyzer()
    return analyzer.analyze_sync(Path(video_path))


def fix_av_sync(video_path: Path, output_path: Path) -> Path:
    """Analyze and fix audio-visual synchronization.

    Factory function that analyzes sync issues and applies correction.

    Args:
        video_path: Path to input video file.
        output_path: Path for output corrected video.

    Returns:
        Path to corrected video file.

    Example:
        >>> corrected = fix_av_sync(
        ...     Path("input.mp4"),
        ...     Path("output.mp4")
        ... )
        >>> print(f"Corrected video: {corrected}")
    """
    analyzer = AudioSyncAnalyzer()
    analysis = analyzer.analyze_sync(Path(video_path))

    corrector = AudioSyncCorrector()
    return corrector.correct_sync(
        Path(video_path),
        analysis,
        Path(output_path)
    )


# Legacy convenience functions for backward compatibility

def analyze_audio_sync(
    video_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None
) -> SyncAnalysis:
    """Analyze audio-video sync for a video file.

    Legacy convenience function - wraps analyze_av_sync.

    Args:
        video_path: Path to video file.
        audio_path: Optional path to separate audio file (ignored, for compatibility).

    Returns:
        SyncAnalysis with sync information.
    """
    return analyze_av_sync(Path(video_path))


def correct_audio_sync(
    audio_path: Union[str, Path],
    offset_ms: float,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Apply sync correction to audio file.

    Legacy convenience function for direct audio adjustment.

    Args:
        audio_path: Path to audio file.
        offset_ms: Offset to apply in milliseconds.
        output_path: Optional output path.

    Returns:
        Path to corrected audio file.
    """
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.parent / f"{audio_path.stem}_synced{audio_path.suffix}"
    output_path = Path(output_path)

    corrector = AudioSyncCorrector()
    corrector._adjust_audio_timing(audio_path, offset_ms, output_path)
    return output_path


def sync_audio_to_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    auto_correct: bool = True
) -> Tuple[SyncAnalysis, Optional[SyncCorrection]]:
    """Analyze and optionally correct audio sync.

    Complete workflow for sync detection and correction.

    Args:
        video_path: Path to video file.
        audio_path: Path to audio file.
        output_path: Optional output path for corrected audio.
        auto_correct: Whether to automatically apply correction.

    Returns:
        Tuple of (SyncAnalysis, SyncCorrection or None).
    """
    analyzer = AudioSyncAnalyzer()
    analysis = analyzer.analyze_sync(Path(video_path))

    correction = None
    if auto_correct and analysis.needs_correction:
        audio_path = Path(audio_path)
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_synced{audio_path.suffix}"

        corrector = AudioSyncCorrector()
        original_duration = corrector._get_duration(audio_path)

        corrector._adjust_audio_timing(
            audio_path,
            analysis.drift_ms,
            Path(output_path)
        )

        adjusted_duration = corrector._get_duration(Path(output_path))

        correction = SyncCorrection(
            original_duration=original_duration,
            adjusted_duration=adjusted_duration,
            stretch_factor=1.0,
            offset_applied_ms=analysis.drift_ms,
            output_path=str(output_path),
            quality_preserved=True
        )

    return analysis, correction


__all__ = [
    # Exceptions
    "AudioSyncError",
    # Configuration
    "SyncConfig",
    # Data classes
    "SyncAnalysis",
    "SyncCorrection",
    "AudioWaveformInfo",
    # Main classes
    "AudioSyncAnalyzer",
    "AudioSyncCorrector",
    "AudioSyncDetector",  # Legacy alias
    # Factory functions (new)
    "analyze_av_sync",
    "fix_av_sync",
    # Convenience functions (legacy)
    "analyze_audio_sync",
    "correct_audio_sync",
    "sync_audio_to_video",
]
