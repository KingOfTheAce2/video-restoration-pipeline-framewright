"""AI-Powered Audio Synchronization for video restoration pipeline.

This module provides audio-video synchronization detection and correction
capabilities using signal processing techniques for precise sync alignment.

Key Features:
-------------
- Audio waveform analysis for beats and speech patterns
- Visual motion detection for sync reference points
- Cross-correlation based offset calculation
- Automatic drift correction for frame interpolation
- Quality-preserving audio stretching/compression

Sync Drift Causes:
------------------
- Variable frame rate source material
- Telecine pulldown conversion
- Frame interpolation changing video duration
- Damaged film with missing frames
- Improper capture/digitization

Detection Methods:
-----------------
- Audio onset detection (transients, beats, speech)
- Visual motion peaks (scene cuts, action)
- Cross-correlation for offset calculation
- Drift analysis over time
"""

import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any, Union

import numpy as np

from framewright.utils.dependencies import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger(__name__)


class AudioSyncError(Exception):
    """Base exception for audio sync errors."""
    pass


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
class SyncAnalysis:
    """Results from audio-video sync analysis.

    Attributes:
        offset_ms: Detected offset in milliseconds.
            Positive = audio leads video, Negative = audio lags video.
        confidence: Confidence score for the detected offset (0.0-1.0).
        drift_per_minute_ms: Detected drift rate in ms per minute.
            Non-zero indicates progressive sync loss.
        needs_correction: Whether sync correction is recommended.
        audio_events_count: Number of audio events detected.
        visual_events_count: Number of visual events detected.
        correlation_score: Cross-correlation peak score.
        analysis_method: Method used for analysis.
    """
    offset_ms: float
    confidence: float
    drift_per_minute_ms: float
    needs_correction: bool
    audio_events_count: int = 0
    visual_events_count: int = 0
    correlation_score: float = 0.0
    analysis_method: str = "cross-correlation"

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "offset_ms": self.offset_ms,
            "confidence": self.confidence,
            "drift_per_minute_ms": self.drift_per_minute_ms,
            "needs_correction": self.needs_correction,
            "audio_events_count": self.audio_events_count,
            "visual_events_count": self.visual_events_count,
            "correlation_score": self.correlation_score,
            "analysis_method": self.analysis_method,
        }


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


class AudioSyncDetector:
    """Detects audio-video synchronization issues.

    Analyzes audio waveform for beats, speech patterns, and transients,
    then correlates with visual motion events to detect sync issues.

    Example:
        >>> detector = AudioSyncDetector()
        >>> analysis = detector.analyze_sync(
        ...     Path("video.mp4"),
        ...     Path("audio.wav")
        ... )
        >>> if analysis.needs_correction:
        ...     print(f"Offset: {analysis.offset_ms}ms")
    """

    # Threshold for considering sync correction needed (in ms)
    SYNC_THRESHOLD_MS = 40.0

    # Minimum confidence for valid detection
    MIN_CONFIDENCE = 0.5

    # Drift threshold (ms per minute) for flagging issues
    DRIFT_THRESHOLD_MS_PER_MIN = 5.0

    def __init__(self) -> None:
        """Initialize the sync detector."""
        self._verify_ffmpeg()
        self._scipy_available = self._check_scipy()
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

    def _check_scipy(self) -> bool:
        """Check if scipy is available."""
        try:
            import scipy.signal
            return True
        except ImportError:
            logger.debug("scipy not available, using basic signal processing")
            return False

    def _check_librosa(self) -> bool:
        """Check if librosa is available."""
        try:
            import librosa
            return True
        except ImportError:
            logger.debug("librosa not available, using FFmpeg for onset detection")
            return False

    def _extract_audio_to_wav(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 48000
    ) -> None:
        """Extract audio from video to WAV format.

        Args:
            input_path: Path to input video/audio file.
            output_path: Path for output WAV file.
            sample_rate: Target sample rate.

        Raises:
            AudioSyncError: If extraction fails.
        """
        command = [
            get_ffmpeg_path(), "-y",
            "-i", str(input_path),
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

    def _load_audio_data(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio data from file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (audio_samples, sample_rate).
        """
        if self._librosa_available:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            return audio, sr

        # Fallback: use FFmpeg to extract raw samples
        command = [
            get_ffmpeg_path(),
            "-i", str(audio_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", "48000",
            "pipe:1"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            timeout=300
        )

        if result.returncode != 0:
            raise AudioSyncError("Failed to load audio data")

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio, 48000

    def get_waveform_info(self, audio_path: Path) -> AudioWaveformInfo:
        """Analyze audio waveform characteristics.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioWaveformInfo with waveform characteristics.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioSyncError(f"Audio file not found: {audio_path}")

        # Get basic info via ffprobe
        command = [
            get_ffprobe_path(), "-v", "quiet",
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
            raise AudioSyncError(f"ffprobe failed: {result.stderr}")

        info = json.loads(result.stdout)

        # Find audio stream
        audio_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            raise AudioSyncError("No audio stream found")

        sample_rate = int(audio_stream.get("sample_rate", 48000))
        channels = int(audio_stream.get("channels", 2))
        duration = float(info.get("format", {}).get("duration", 0))
        bit_depth = audio_stream.get("bits_per_sample")

        # Load audio for peak detection
        audio, sr = self._load_audio_data(audio_path)

        # Detect peaks (local maxima above threshold)
        threshold = np.std(audio) * 2
        peak_positions = self._find_peaks(audio, threshold, min_distance=int(sr * 0.05))

        # Calculate RMS levels in windows
        window_size = int(sr * 0.1)  # 100ms windows
        rms_levels = []
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_levels.append(float(rms))

        return AudioWaveformInfo(
            sample_rate=sr,
            duration=duration,
            peak_positions=peak_positions,
            rms_levels=rms_levels,
            channels=channels,
            bit_depth=int(bit_depth) if bit_depth else None
        )

    def _find_peaks(
        self,
        signal: np.ndarray,
        threshold: float,
        min_distance: int
    ) -> List[int]:
        """Find peaks in signal above threshold.

        Args:
            signal: Input signal.
            threshold: Minimum peak height.
            min_distance: Minimum samples between peaks.

        Returns:
            List of peak positions.
        """
        if self._scipy_available:
            from scipy.signal import find_peaks as scipy_find_peaks
            peaks, _ = scipy_find_peaks(
                np.abs(signal),
                height=threshold,
                distance=min_distance
            )
            return peaks.tolist()

        # Basic peak finding
        peaks = []
        last_peak = -min_distance

        for i in range(1, len(signal) - 1):
            if i - last_peak < min_distance:
                continue

            val = abs(signal[i])
            if val > threshold:
                if val > abs(signal[i - 1]) and val > abs(signal[i + 1]):
                    peaks.append(i)
                    last_peak = i

        return peaks

    def detect_audio_events(
        self,
        audio_path: Path,
        method: str = "onset"
    ) -> List[float]:
        """Detect audio events (onsets/transients) in audio file.

        Uses onset detection to find transients, beats, and speech onset
        points that can be correlated with visual events.

        Args:
            audio_path: Path to audio file.
            method: Detection method ("onset", "beat", "speech").

        Returns:
            List of event timestamps in seconds.

        Raises:
            AudioSyncError: If detection fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioSyncError(f"Audio file not found: {audio_path}")

        logger.info(f"Detecting audio events in {audio_path} (method: {method})")

        if self._librosa_available:
            return self._detect_events_librosa(audio_path, method)
        else:
            return self._detect_events_ffmpeg(audio_path, method)

    def _detect_events_librosa(
        self,
        audio_path: Path,
        method: str
    ) -> List[float]:
        """Detect audio events using librosa.

        Args:
            audio_path: Path to audio file.
            method: Detection method.

        Returns:
            List of event timestamps.
        """
        import librosa

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)

        if method == "beat":
            # Beat detection
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            events = librosa.frames_to_time(beat_frames, sr=sr)

        elif method == "speech":
            # Speech onset using spectral flux + high-frequency energy
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Focus on speech frequencies (300-3000 Hz)
            stft = librosa.stft(y)
            freqs = librosa.fft_frequencies(sr=sr)
            speech_mask = (freqs >= 300) & (freqs <= 3000)
            speech_energy = np.sum(np.abs(stft[speech_mask, :]), axis=0)

            # Combine onset strength with speech energy
            combined = onset_env * (speech_energy / np.max(speech_energy) + 0.5)

            # Peak picking
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=combined,
                sr=sr,
                units='frames'
            )
            events = librosa.frames_to_time(onset_frames, sr=sr)

        else:  # "onset" (default)
            # General onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            events = librosa.frames_to_time(onset_frames, sr=sr)

        logger.info(f"Detected {len(events)} audio events")
        return events.tolist()

    def _detect_events_ffmpeg(
        self,
        audio_path: Path,
        method: str
    ) -> List[float]:
        """Detect audio events using FFmpeg silencedetect filter.

        Args:
            audio_path: Path to audio file.
            method: Detection method (simplified with FFmpeg).

        Returns:
            List of event timestamps.
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
                # Parse: [silencedetect @ ...] silence_end: 1.234
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

        logger.info(f"Detected {len(events)} audio events (FFmpeg)")
        return events

    def detect_visual_events(
        self,
        video_path: Path,
        method: str = "motion"
    ) -> List[float]:
        """Detect visual events (motion peaks/scene changes) in video.

        Uses motion analysis to find visual events that can be correlated
        with audio events for sync detection.

        Args:
            video_path: Path to video file.
            method: Detection method ("motion", "scene", "both").

        Returns:
            List of event timestamps in seconds.

        Raises:
            AudioSyncError: If detection fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioSyncError(f"Video file not found: {video_path}")

        logger.info(f"Detecting visual events in {video_path} (method: {method})")

        events = []

        if method in ("scene", "both"):
            # Scene change detection using FFmpeg
            scene_events = self._detect_scene_changes(video_path)
            events.extend(scene_events)

        if method in ("motion", "both"):
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

        logger.info(f"Detected {len(events)} visual events")
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

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600
        )

        events = []
        for line in result.stderr.split('\n'):
            if 'pts_time' in line:
                try:
                    # Parse: [Parsed_showinfo...] ... pts_time:1.234567
                    parts = line.split('pts_time:')
                    if len(parts) > 1:
                        time_str = parts[1].strip().split()[0]
                        events.append(float(time_str))
                except (ValueError, IndexError):
                    continue

        return events

    def _detect_motion_peaks(self, video_path: Path) -> List[float]:
        """Detect motion peaks in video using frame differencing.

        Args:
            video_path: Path to video file.

        Returns:
            List of motion peak timestamps.
        """
        # Use FFmpeg to extract motion data
        command = [
            get_ffmpeg_path(),
            "-i", str(video_path),
            "-vf", "mestimate=epzs,codecview=mv=pf+bf+bb",
            "-f", "null", "-"
        ]

        # For simpler approach, use frame difference magnitude
        # This is a simplified implementation using select filter
        command = [
            get_ffmpeg_path(),
            "-i", str(video_path),
            "-vf", "select='gt(scene,0.1)',showinfo",
            "-f", "null", "-"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600
        )

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

    def calculate_offset(
        self,
        audio_events: List[float],
        visual_events: List[float],
        max_offset_ms: float = 2000.0
    ) -> Tuple[float, float]:
        """Calculate optimal offset between audio and visual events.

        Uses cross-correlation to find the offset that best aligns
        audio and visual events.

        Args:
            audio_events: List of audio event timestamps (seconds).
            visual_events: List of visual event timestamps (seconds).
            max_offset_ms: Maximum offset to search (milliseconds).

        Returns:
            Tuple of (offset_ms, correlation_score).
            Positive offset means audio leads video.
        """
        if not audio_events or not visual_events:
            logger.warning("Insufficient events for offset calculation")
            return 0.0, 0.0

        logger.info(
            f"Calculating offset from {len(audio_events)} audio and "
            f"{len(visual_events)} visual events"
        )

        # Convert to numpy arrays
        audio = np.array(audio_events)
        visual = np.array(visual_events)

        # Search range
        max_offset_s = max_offset_ms / 1000.0
        step_s = 0.001  # 1ms resolution
        offsets = np.arange(-max_offset_s, max_offset_s, step_s)

        best_offset = 0.0
        best_score = 0.0

        # For each offset, count how many audio events are within tolerance
        # of a visual event
        tolerance = 0.05  # 50ms tolerance for matching

        for offset in offsets:
            shifted_audio = audio + offset
            matches = 0

            for a_time in shifted_audio:
                # Find closest visual event
                if len(visual) > 0:
                    closest_idx = np.argmin(np.abs(visual - a_time))
                    if np.abs(visual[closest_idx] - a_time) < tolerance:
                        matches += 1

            # Normalize score
            score = matches / max(len(audio_events), 1)

            if score > best_score:
                best_score = score
                best_offset = offset

        # Convert to milliseconds
        offset_ms = best_offset * 1000.0

        logger.info(
            f"Best offset: {offset_ms:.1f}ms "
            f"(correlation score: {best_score:.3f})"
        )

        return offset_ms, best_score

    def analyze_sync(
        self,
        video_path: Path,
        audio_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SyncAnalysis:
        """Perform comprehensive audio-video sync analysis.

        Analyzes both audio and video to detect sync issues including
        constant offset and drift over time.

        Args:
            video_path: Path to video file.
            audio_path: Path to separate audio file (uses video's audio if None).
            progress_callback: Optional progress callback.

        Returns:
            SyncAnalysis with detailed sync information.

        Raises:
            AudioSyncError: If analysis fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioSyncError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing sync for: {video_path}")

        if progress_callback:
            progress_callback("Extracting audio events...")

        # Use video's audio if no separate audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if audio_path is None:
                audio_path = temp_path / "extracted_audio.wav"
                self._extract_audio_to_wav(video_path, audio_path)

            if progress_callback:
                progress_callback("Detecting audio events...")

            # Detect audio events
            audio_events = self.detect_audio_events(audio_path)

            if progress_callback:
                progress_callback("Detecting visual events...")

            # Detect visual events
            visual_events = self.detect_visual_events(video_path, method="both")

            if progress_callback:
                progress_callback("Calculating offset...")

            # Calculate offset
            offset_ms, correlation = self.calculate_offset(audio_events, visual_events)

            # Analyze drift by comparing first and second halves
            drift_per_minute = 0.0
            if len(audio_events) > 10 and len(visual_events) > 10:
                drift_per_minute = self._analyze_drift(
                    audio_events, visual_events, audio_path
                )

            # Determine confidence
            confidence = correlation
            if len(audio_events) < 5 or len(visual_events) < 5:
                confidence *= 0.5  # Low event count reduces confidence

            # Determine if correction is needed
            needs_correction = (
                abs(offset_ms) > self.SYNC_THRESHOLD_MS or
                abs(drift_per_minute) > self.DRIFT_THRESHOLD_MS_PER_MIN
            ) and confidence >= self.MIN_CONFIDENCE

            if progress_callback:
                progress_callback("Analysis complete")

            return SyncAnalysis(
                offset_ms=offset_ms,
                confidence=confidence,
                drift_per_minute_ms=drift_per_minute,
                needs_correction=needs_correction,
                audio_events_count=len(audio_events),
                visual_events_count=len(visual_events),
                correlation_score=correlation,
                analysis_method="cross-correlation"
            )

    def _analyze_drift(
        self,
        audio_events: List[float],
        visual_events: List[float],
        audio_path: Path
    ) -> float:
        """Analyze sync drift over time.

        Compares offset in first and second half of content to detect
        progressive sync loss.

        Args:
            audio_events: Audio event timestamps.
            visual_events: Visual event timestamps.
            audio_path: Path to audio file for duration.

        Returns:
            Drift rate in milliseconds per minute.
        """
        # Get audio duration
        try:
            waveform_info = self.get_waveform_info(audio_path)
            duration = waveform_info.duration
        except Exception:
            duration = max(max(audio_events), max(visual_events))

        midpoint = duration / 2

        # Split events
        audio_first = [e for e in audio_events if e < midpoint]
        audio_second = [e for e in audio_events if e >= midpoint]
        visual_first = [e for e in visual_events if e < midpoint]
        visual_second = [e for e in visual_events if e >= midpoint]

        # Calculate offset for each half
        offset_first, _ = self.calculate_offset(audio_first, visual_first)
        offset_second, _ = self.calculate_offset(audio_second, visual_second)

        # Calculate drift rate
        if duration > 0:
            drift_total = offset_second - offset_first
            drift_per_minute = (drift_total / duration) * 60
        else:
            drift_per_minute = 0.0

        logger.info(
            f"Drift analysis: first half={offset_first:.1f}ms, "
            f"second half={offset_second:.1f}ms, "
            f"drift={drift_per_minute:.2f}ms/min"
        )

        return drift_per_minute


class AudioSyncCorrector:
    """Corrects audio-video synchronization issues.

    Applies micro-adjustments to audio timing, handles time stretching
    for duration matching, and preserves audio quality during correction.

    Example:
        >>> corrector = AudioSyncCorrector()
        >>> correction = corrector.correct_sync(
        ...     Path("audio.wav"),
        ...     offset_ms=-100  # Audio 100ms late
        ... )
        >>> print(f"Corrected audio: {correction.output_path}")
    """

    # Maximum stretch factor before quality concerns
    MAX_STRETCH_FACTOR = 1.05  # 5% stretch/compress

    def __init__(self) -> None:
        """Initialize the sync corrector."""
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

    def _validate_input(self, audio_path: Path) -> None:
        """Validate input file exists."""
        if not audio_path.exists():
            raise AudioSyncError(f"Audio file not found: {audio_path}")

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        command = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))

    def correct_sync(
        self,
        audio_path: Path,
        offset_ms: float,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Path:
        """Apply offset correction to audio.

        Shifts audio timing by adding silence (positive offset) or
        trimming beginning (negative offset).

        Args:
            audio_path: Path to input audio file.
            offset_ms: Offset to apply in milliseconds.
                Positive = audio leads (add delay to start)
                Negative = audio lags (trim start)
            output_path: Path for output file (auto-generated if None).
            progress_callback: Optional progress callback.

        Returns:
            Path to corrected audio file.

        Raises:
            AudioSyncError: If correction fails.
        """
        audio_path = Path(audio_path)
        self._validate_input(audio_path)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_synced{audio_path.suffix}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying sync correction: {offset_ms}ms offset")

        if progress_callback:
            progress_callback(f"Applying {offset_ms}ms offset...")

        if abs(offset_ms) < 1:
            # No significant correction needed
            shutil.copy2(audio_path, output_path)
            return output_path

        if offset_ms > 0:
            # Audio leads video - add delay at start
            delay_samples = int(offset_ms)  # adelay uses milliseconds
            filter_str = f"adelay={delay_samples}|{delay_samples}"
        else:
            # Audio lags video - trim from start
            trim_seconds = abs(offset_ms) / 1000.0
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
                raise AudioSyncError(f"Sync correction failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Sync correction timed out")

        logger.info(f"Sync correction applied: {output_path}")
        return output_path

    def adjust_for_interpolation(
        self,
        audio_path: Path,
        source_fps: float,
        target_fps: float,
        output_path: Optional[Path] = None,
        high_quality: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Path:
        """Adjust audio duration for frame interpolation.

        When video is interpolated from source_fps to target_fps, the video
        duration changes. This method adjusts audio to match.

        Args:
            audio_path: Path to input audio file.
            source_fps: Original video frame rate.
            target_fps: Target video frame rate after interpolation.
            output_path: Path for output file (auto-generated if None).
            high_quality: Use rubberband for quality (if available).
            progress_callback: Optional progress callback.

        Returns:
            Path to adjusted audio file.

        Raises:
            AudioSyncError: If adjustment fails.

        Note:
            Frame interpolation itself doesn't change video duration (it adds
            frames but keeps the same total time). This method is for cases
            where the interpolation process somehow affects duration (e.g.,
            frame dropping during decimation to exact target fps).
        """
        audio_path = Path(audio_path)
        self._validate_input(audio_path)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_adjusted{audio_path.suffix}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Frame interpolation typically doesn't change duration, but
        # decimation during fps conversion can. Calculate stretch factor.
        # If source had N frames at source_fps, duration = N/source_fps
        # After interpolation to target_fps with exact frame count target,
        # we may need slight adjustment.

        # For most interpolation cases, no duration change is needed
        if abs(source_fps - target_fps) < 0.01:
            shutil.copy2(audio_path, output_path)
            return output_path

        original_duration = self._get_audio_duration(audio_path)

        # Calculate expected video duration change
        # This handles cases like 24fps -> 23.976fps conversions
        duration_ratio = source_fps / target_fps

        if abs(duration_ratio - 1.0) < 0.001:
            # Negligible change
            shutil.copy2(audio_path, output_path)
            return output_path

        logger.info(
            f"Adjusting audio for fps change: {source_fps} -> {target_fps} "
            f"(ratio: {duration_ratio:.4f})"
        )

        return self._time_stretch(
            audio_path,
            output_path,
            duration_ratio,
            high_quality,
            progress_callback
        )

    def stretch_to_duration(
        self,
        audio_path: Path,
        target_duration: float,
        output_path: Optional[Path] = None,
        high_quality: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SyncCorrection:
        """Stretch or compress audio to match target duration.

        Uses time-stretching to adjust audio duration while preserving
        pitch and quality as much as possible.

        Args:
            audio_path: Path to input audio file.
            target_duration: Target duration in seconds.
            output_path: Path for output file (auto-generated if None).
            high_quality: Use rubberband for quality (if available).
            progress_callback: Optional progress callback.

        Returns:
            SyncCorrection with adjustment details.

        Raises:
            AudioSyncError: If stretch factor exceeds safe limits.
        """
        audio_path = Path(audio_path)
        self._validate_input(audio_path)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_stretched{audio_path.suffix}"
        output_path = Path(output_path)

        original_duration = self._get_audio_duration(audio_path)

        if original_duration <= 0:
            raise AudioSyncError("Could not determine audio duration")

        stretch_factor = target_duration / original_duration

        logger.info(
            f"Stretching audio: {original_duration:.2f}s -> {target_duration:.2f}s "
            f"(factor: {stretch_factor:.4f})"
        )

        # Check stretch limits
        if stretch_factor > self.MAX_STRETCH_FACTOR or stretch_factor < (1 / self.MAX_STRETCH_FACTOR):
            logger.warning(
                f"Stretch factor {stretch_factor:.4f} exceeds recommended range "
                f"({1/self.MAX_STRETCH_FACTOR:.2f} - {self.MAX_STRETCH_FACTOR:.2f}). "
                "Audio quality may be affected."
            )

        if progress_callback:
            progress_callback(f"Time-stretching audio (factor: {stretch_factor:.3f})...")

        result_path = self._time_stretch(
            audio_path,
            output_path,
            stretch_factor,
            high_quality,
            progress_callback
        )

        # Verify output duration
        adjusted_duration = self._get_audio_duration(result_path)

        return SyncCorrection(
            original_duration=original_duration,
            adjusted_duration=adjusted_duration,
            stretch_factor=stretch_factor,
            output_path=str(result_path),
            quality_preserved=high_quality and abs(stretch_factor - 1.0) < 0.03
        )

    def _time_stretch(
        self,
        audio_path: Path,
        output_path: Path,
        stretch_factor: float,
        high_quality: bool,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Path:
        """Apply time stretching to audio.

        Args:
            audio_path: Input audio path.
            output_path: Output audio path.
            stretch_factor: Stretch factor (>1 = slower, <1 = faster).
            high_quality: Use rubberband if available.
            progress_callback: Optional progress callback.

        Returns:
            Path to stretched audio.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # atempo range is 0.5 to 2.0, chain multiple for larger changes
        tempo_factor = 1.0 / stretch_factor  # atempo speeds up, we want stretch

        if high_quality and self._rubberband_available:
            # Use rubberband for highest quality
            filter_str = f"rubberband=tempo={tempo_factor}:pitch=1.0"
        else:
            # Use atempo (may need chaining for extreme values)
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

        logger.debug(f"Time stretch command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                raise AudioSyncError(f"Time stretch failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise AudioSyncError("Time stretch timed out")

        return output_path

    def correct_with_analysis(
        self,
        audio_path: Path,
        analysis: SyncAnalysis,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SyncCorrection:
        """Apply sync correction based on analysis results.

        Combines offset correction and drift compensation based on
        the provided SyncAnalysis.

        Args:
            audio_path: Path to input audio file.
            analysis: SyncAnalysis from AudioSyncDetector.
            output_path: Path for output file.
            progress_callback: Optional progress callback.

        Returns:
            SyncCorrection with full correction details.
        """
        audio_path = Path(audio_path)
        self._validate_input(audio_path)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_synced{audio_path.suffix}"

        original_duration = self._get_audio_duration(audio_path)

        if not analysis.needs_correction:
            logger.info("No sync correction needed according to analysis")
            shutil.copy2(audio_path, output_path)
            return SyncCorrection(
                original_duration=original_duration,
                adjusted_duration=original_duration,
                stretch_factor=1.0,
                offset_applied_ms=0.0,
                output_path=str(output_path),
                quality_preserved=True
            )

        logger.info(
            f"Applying correction: offset={analysis.offset_ms}ms, "
            f"drift={analysis.drift_per_minute_ms}ms/min"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            current_path = audio_path

            # Step 1: Apply offset correction
            if abs(analysis.offset_ms) >= 1:
                if progress_callback:
                    progress_callback("Applying offset correction...")

                offset_output = temp_path / "offset_corrected.wav"
                self.correct_sync(
                    current_path,
                    analysis.offset_ms,
                    offset_output
                )
                current_path = offset_output

            # Step 2: Apply drift correction via time stretch
            if abs(analysis.drift_per_minute_ms) > 0.5:
                if progress_callback:
                    progress_callback("Applying drift correction...")

                # Calculate stretch factor from drift
                total_drift_ms = (analysis.drift_per_minute_ms * original_duration) / 60
                stretch_factor = (original_duration * 1000) / (original_duration * 1000 + total_drift_ms)

                drift_output = temp_path / "drift_corrected.wav"
                self._time_stretch(
                    current_path,
                    drift_output,
                    stretch_factor,
                    high_quality=True
                )
                current_path = drift_output
            else:
                stretch_factor = 1.0

            # Copy final result
            shutil.copy2(current_path, output_path)

        adjusted_duration = self._get_audio_duration(output_path)

        return SyncCorrection(
            original_duration=original_duration,
            adjusted_duration=adjusted_duration,
            stretch_factor=stretch_factor,
            offset_applied_ms=analysis.offset_ms,
            output_path=str(output_path),
            quality_preserved=abs(stretch_factor - 1.0) < 0.03
        )


# Convenience functions

def analyze_audio_sync(
    video_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None
) -> SyncAnalysis:
    """Analyze audio-video sync for a video file.

    Convenience function for quick sync analysis.

    Args:
        video_path: Path to video file.
        audio_path: Optional path to separate audio file.

    Returns:
        SyncAnalysis with sync information.

    Example:
        >>> analysis = analyze_audio_sync("video.mp4")
        >>> if analysis.needs_correction:
        ...     print(f"Audio is {analysis.offset_ms}ms out of sync")
    """
    detector = AudioSyncDetector()
    return detector.analyze_sync(
        Path(video_path),
        Path(audio_path) if audio_path else None
    )


def correct_audio_sync(
    audio_path: Union[str, Path],
    offset_ms: float,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Apply sync correction to audio file.

    Convenience function for quick sync correction.

    Args:
        audio_path: Path to audio file.
        offset_ms: Offset to apply in milliseconds.
        output_path: Optional output path.

    Returns:
        Path to corrected audio file.

    Example:
        >>> corrected = correct_audio_sync("audio.wav", offset_ms=50)
        >>> print(f"Corrected audio: {corrected}")
    """
    corrector = AudioSyncCorrector()
    return corrector.correct_sync(
        Path(audio_path),
        offset_ms,
        Path(output_path) if output_path else None
    )


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

    Example:
        >>> analysis, correction = sync_audio_to_video(
        ...     "video.mp4", "audio.wav",
        ...     auto_correct=True
        ... )
        >>> if correction:
        ...     print(f"Audio corrected to: {correction.output_path}")
    """
    detector = AudioSyncDetector()
    analysis = detector.analyze_sync(Path(video_path), Path(audio_path))

    correction = None
    if auto_correct and analysis.needs_correction:
        corrector = AudioSyncCorrector()
        correction = corrector.correct_with_analysis(
            Path(audio_path),
            analysis,
            Path(output_path) if output_path else None
        )

    return analysis, correction


__all__ = [
    # Exceptions
    "AudioSyncError",
    # Data classes
    "AudioWaveformInfo",
    "SyncAnalysis",
    "SyncCorrection",
    # Main classes
    "AudioSyncDetector",
    "AudioSyncCorrector",
    # Convenience functions
    "analyze_audio_sync",
    "correct_audio_sync",
    "sync_audio_to_video",
]
