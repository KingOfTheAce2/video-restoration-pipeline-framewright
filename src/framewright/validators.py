"""Quality validation module for FrameWright pipeline.

Provides frame integrity validation, quality metrics, artifact detection,
and temporal consistency checking.
"""
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QualityMetrics:
    """Quality metrics for a frame or video."""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    vmaf: Optional[float] = None
    mse: Optional[float] = None

    def meets_threshold(
        self,
        min_psnr: float = 25.0,
        min_ssim: float = 0.85,
        min_vmaf: float = 70.0,
    ) -> bool:
        """Check if metrics meet quality thresholds."""
        if self.psnr is not None and self.psnr < min_psnr:
            return False
        if self.ssim is not None and self.ssim < min_ssim:
            return False
        if self.vmaf is not None and self.vmaf < min_vmaf:
            return False
        return True


@dataclass
class FrameValidation:
    """Frame validation result."""
    frame_path: Path
    is_valid: bool
    width: int = 0
    height: int = 0
    file_size: int = 0
    error_message: Optional[str] = None
    quality: Optional[QualityMetrics] = None


@dataclass
class SequenceReport:
    """Frame sequence validation report."""
    total_frames: int
    expected_frames: int
    missing_frames: List[int] = field(default_factory=list)
    duplicate_frames: List[int] = field(default_factory=list)
    invalid_frames: List[Path] = field(default_factory=list)
    is_complete: bool = True

    @property
    def missing_count(self) -> int:
        return len(self.missing_frames)

    @property
    def has_issues(self) -> bool:
        return bool(self.missing_frames or self.duplicate_frames or self.invalid_frames)


@dataclass
class ArtifactReport:
    """Artifact detection report."""
    frame_path: Path
    has_artifacts: bool
    artifacts: List[str] = field(default_factory=list)
    severity: str = "none"  # none, low, medium, high
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalReport:
    """Temporal consistency report."""
    frames_analyzed: int
    brightness_variance: float
    color_variance: float
    flickering_detected: bool
    flicker_frames: List[int] = field(default_factory=list)
    severity: str = "none"


@dataclass
class AudioValidation:
    """Audio stream validation result."""
    has_audio: bool
    codec: Optional[str] = None
    sample_rate: int = 0
    channels: int = 0
    duration: float = 0.0
    bit_depth: Optional[int] = None
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class AudioIssue:
    """Represents a detected audio issue."""
    issue_type: str  # SILENCE, CLIPPING, NOISE, DROPOUT
    start_time: float
    end_time: Optional[float] = None
    severity: str = "low"  # low, medium, high
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioQualityReport:
    """Audio quality analysis report."""
    audio_path: Path
    max_volume_db: float = 0.0
    mean_volume_db: float = -20.0
    clipping_detected: bool = False
    clipping_count: int = 0
    silence_segments: List[Tuple[float, float]] = field(default_factory=list)
    noise_floor_db: float = -60.0
    dynamic_range_db: float = 0.0
    issues: List[AudioIssue] = field(default_factory=list)
    is_acceptable: bool = True

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0 or self.clipping_detected


# =============================================================================
# Frame Integrity Validation
# =============================================================================

def validate_frame_integrity(frame_path: Path) -> FrameValidation:
    """Validate that a frame file is valid and not corrupted.

    Args:
        frame_path: Path to frame file

    Returns:
        FrameValidation with results
    """
    result = FrameValidation(frame_path=frame_path, is_valid=False)

    if not frame_path.exists():
        result.error_message = "File does not exist"
        return result

    result.file_size = frame_path.stat().st_size

    if result.file_size == 0:
        result.error_message = "File is empty"
        return result

    # Use ffprobe to validate image
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
            str(frame_path)
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if proc.returncode != 0:
            result.error_message = f"ffprobe error: {proc.stderr}"
            return result

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])

        if not streams:
            result.error_message = "No video stream found"
            return result

        stream = streams[0]
        result.width = int(stream.get("width", 0))
        result.height = int(stream.get("height", 0))

        if result.width == 0 or result.height == 0:
            result.error_message = "Invalid dimensions"
            return result

        result.is_valid = True

    except subprocess.TimeoutExpired:
        result.error_message = "Validation timed out"
    except json.JSONDecodeError as e:
        result.error_message = f"Invalid ffprobe output: {e}"
    except Exception as e:
        result.error_message = f"Validation error: {e}"

    return result


def validate_frame_batch(
    frame_paths: List[Path],
    parallel: bool = True,
    max_workers: int = 4,
) -> List[FrameValidation]:
    """Validate multiple frames.

    Args:
        frame_paths: List of frame paths
        parallel: Use parallel processing
        max_workers: Number of parallel workers

    Returns:
        List of FrameValidation results
    """
    if not parallel or len(frame_paths) <= 4:
        return [validate_frame_integrity(p) for p in frame_paths]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(validate_frame_integrity, p): p
            for p in frame_paths
        }

        for future in as_completed(future_to_path):
            results.append(future.result())

    return results


# =============================================================================
# Frame Sequence Validation
# =============================================================================

def validate_frame_sequence(
    frame_dir: Path,
    pattern: str = "frame_*.png",
    expected_count: Optional[int] = None,
) -> SequenceReport:
    """Validate frame sequence for completeness and ordering.

    Args:
        frame_dir: Directory containing frames
        pattern: Glob pattern for frame files
        expected_count: Expected number of frames (optional)

    Returns:
        SequenceReport with validation results
    """
    frames = sorted(frame_dir.glob(pattern))
    total = len(frames)

    if total == 0:
        return SequenceReport(
            total_frames=0,
            expected_frames=expected_count or 0,
            is_complete=False,
        )

    # Extract frame numbers
    numbers = []
    for frame in frames:
        match = re.search(r"(\d+)", frame.stem)
        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        return SequenceReport(
            total_frames=total,
            expected_frames=expected_count or total,
            is_complete=False,
        )

    # Check for gaps
    min_num = min(numbers)
    max_num = max(numbers)
    expected_set = set(range(min_num, max_num + 1))
    actual_set = set(numbers)

    missing = sorted(expected_set - actual_set)

    # Check for duplicates
    seen = set()
    duplicates = []
    for num in numbers:
        if num in seen:
            duplicates.append(num)
        seen.add(num)

    # Validate frame integrity for a sample
    sample_size = min(10, total)
    sample_frames = frames[::max(1, total // sample_size)][:sample_size]
    invalid_frames = []

    for frame in sample_frames:
        validation = validate_frame_integrity(frame)
        if not validation.is_valid:
            invalid_frames.append(frame)

    expected = expected_count if expected_count else (max_num - min_num + 1)

    return SequenceReport(
        total_frames=total,
        expected_frames=expected,
        missing_frames=missing,
        duplicate_frames=duplicates,
        invalid_frames=invalid_frames,
        is_complete=len(missing) == 0 and len(duplicates) == 0,
    )


# =============================================================================
# Quality Metrics
# =============================================================================

def compute_quality_metrics(
    distorted: Path,
    reference: Path,
    metrics: List[str] = None,
) -> Optional[QualityMetrics]:
    """Compute quality metrics between distorted and reference frames.

    Requires ffmpeg-quality-metrics to be installed.

    Args:
        distorted: Path to distorted/processed frame
        reference: Path to reference frame
        metrics: List of metrics to compute (psnr, ssim, vmaf)

    Returns:
        QualityMetrics or None if computation fails
    """
    if metrics is None:
        metrics = ["psnr", "ssim"]

    # Check if ffmpeg-quality-metrics is available
    if not shutil.which("ffmpeg-quality-metrics"):
        logger.warning("ffmpeg-quality-metrics not installed, skipping quality check")
        return None

    try:
        cmd = [
            "ffmpeg-quality-metrics",
            str(distorted),
            str(reference),
            "--metrics",
        ] + metrics + [
            "--output-format", "json",
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if proc.returncode != 0:
            logger.warning(f"Quality metrics computation failed: {proc.stderr}")
            return None

        data = json.loads(proc.stdout)

        result = QualityMetrics()

        if "psnr" in data:
            psnr_data = data["psnr"]
            if isinstance(psnr_data, dict):
                result.psnr = psnr_data.get("psnr_avg") or psnr_data.get("psnr_y")
            elif isinstance(psnr_data, list) and psnr_data:
                result.psnr = psnr_data[0].get("psnr_avg") or psnr_data[0].get("psnr_y")

        if "ssim" in data:
            ssim_data = data["ssim"]
            if isinstance(ssim_data, dict):
                result.ssim = ssim_data.get("ssim_avg") or ssim_data.get("ssim_y")
            elif isinstance(ssim_data, list) and ssim_data:
                result.ssim = ssim_data[0].get("ssim_avg") or ssim_data[0].get("ssim_y")

        if "vmaf" in data:
            vmaf_data = data["vmaf"]
            if isinstance(vmaf_data, dict):
                result.vmaf = vmaf_data.get("vmaf")
            elif isinstance(vmaf_data, list) and vmaf_data:
                result.vmaf = vmaf_data[0].get("vmaf")

        return result

    except subprocess.TimeoutExpired:
        logger.warning("Quality metrics computation timed out")
        return None
    except Exception as e:
        logger.warning(f"Quality metrics error: {e}")
        return None


def validate_enhancement_quality(
    original: Path,
    enhanced: Path,
    min_psnr: float = 25.0,
    min_ssim: float = 0.85,
) -> Tuple[bool, Optional[QualityMetrics]]:
    """Validate that enhanced frame meets quality thresholds.

    Args:
        original: Path to original frame
        enhanced: Path to enhanced frame
        min_psnr: Minimum PSNR threshold
        min_ssim: Minimum SSIM threshold

    Returns:
        Tuple of (passes_threshold, metrics)
    """
    metrics = compute_quality_metrics(enhanced, original)

    if metrics is None:
        # If we can't compute metrics, assume it's okay
        return True, None

    passes = metrics.meets_threshold(min_psnr=min_psnr, min_ssim=min_ssim)

    return passes, metrics


# =============================================================================
# Artifact Detection
# =============================================================================

def detect_artifacts(frame_path: Path) -> ArtifactReport:
    """Detect common enhancement artifacts in a frame.

    Detects:
    - Tiling artifacts (repetitive patterns)
    - Halo artifacts (over-sharpening)
    - Color banding
    - Blockiness

    Args:
        frame_path: Path to frame to analyze

    Returns:
        ArtifactReport with detected artifacts
    """
    report = ArtifactReport(frame_path=frame_path, has_artifacts=False)

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("NumPy/Pillow not available for artifact detection")
        return report

    try:
        img = Image.open(frame_path)
        img_array = np.array(img)

        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        artifacts = []
        details = {}

        # Detect color banding (low color variance in smooth regions)
        color_variance = np.var(img_array)
        if color_variance < 100:  # Very low variance might indicate banding
            # Check for distinct color levels
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1] if len(img_array.shape) == 3 else 1), axis=0))
            if unique_colors < 1000:
                artifacts.append("COLOR_BANDING")
                details["unique_colors"] = unique_colors

        # Detect blockiness (8x8 DCT block artifacts)
        block_size = 8
        h, w = gray.shape
        block_h, block_w = h // block_size, w // block_size

        if block_h > 2 and block_w > 2:
            # Calculate variance at block boundaries
            h_edges = gray[:, block_size-1::block_size] - gray[:, block_size::block_size]
            v_edges = gray[block_size-1::block_size, :] - gray[block_size::block_size, :]

            edge_variance = np.mean(np.abs(h_edges)) + np.mean(np.abs(v_edges))

            if edge_variance > 10:  # Threshold for blockiness
                artifacts.append("BLOCKINESS")
                details["edge_variance"] = float(edge_variance)

        # Detect over-sharpening halos
        # Look for high frequency ringing around edges
        from scipy import ndimage
        laplacian = ndimage.laplace(gray.astype(float))
        halo_score = np.percentile(np.abs(laplacian), 99)

        if halo_score > 50:  # High frequency content threshold
            artifacts.append("HALO_ARTIFACT")
            details["halo_score"] = float(halo_score)

        # Update report
        if artifacts:
            report.has_artifacts = True
            report.artifacts = artifacts
            report.details = details

            # Determine severity
            if len(artifacts) >= 3:
                report.severity = "high"
            elif len(artifacts) >= 2:
                report.severity = "medium"
            else:
                report.severity = "low"

    except ImportError:
        # scipy not available, skip advanced detection
        pass
    except Exception as e:
        logger.warning(f"Artifact detection error: {e}")

    return report


# =============================================================================
# Temporal Consistency
# =============================================================================

def validate_temporal_consistency(
    frame_dir: Path,
    sample_rate: int = 10,
    flicker_threshold: float = 15.0,
) -> TemporalReport:
    """Check for temporal inconsistencies like flickering.

    Args:
        frame_dir: Directory containing frames
        sample_rate: Analyze every Nth frame
        flicker_threshold: Variance threshold for flicker detection

    Returns:
        TemporalReport with analysis results
    """
    frames = sorted(frame_dir.glob("*.png"))

    if len(frames) < 3:
        return TemporalReport(
            frames_analyzed=len(frames),
            brightness_variance=0.0,
            color_variance=0.0,
            flickering_detected=False,
        )

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("NumPy/Pillow not available for temporal analysis")
        return TemporalReport(
            frames_analyzed=0,
            brightness_variance=0.0,
            color_variance=0.0,
            flickering_detected=False,
        )

    brightness_values = []
    color_values = []
    frame_indices = []

    try:
        for i, frame in enumerate(frames[::sample_rate]):
            img = Image.open(frame)
            img_array = np.array(img)

            # Calculate mean brightness
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
                color_mean = np.mean(img_array, axis=(0, 1))
                color_values.append(color_mean)
            else:
                gray = img_array

            brightness_values.append(np.mean(gray))
            frame_indices.append(i * sample_rate)

        brightness_array = np.array(brightness_values)
        brightness_variance = float(np.var(brightness_array))

        # Calculate frame-to-frame differences
        brightness_diffs = np.abs(np.diff(brightness_array))

        # Detect flickering (large sudden changes)
        flicker_frames = []
        for i, diff in enumerate(brightness_diffs):
            if diff > flicker_threshold:
                flicker_frames.append(frame_indices[i + 1])

        # Color variance
        color_variance = 0.0
        if color_values:
            color_array = np.array(color_values)
            color_variance = float(np.mean(np.var(color_array, axis=0)))

        flickering_detected = len(flicker_frames) > len(frames) * 0.01  # >1% frames

        severity = "none"
        if flickering_detected:
            flicker_ratio = len(flicker_frames) / len(brightness_values)
            if flicker_ratio > 0.1:
                severity = "high"
            elif flicker_ratio > 0.05:
                severity = "medium"
            else:
                severity = "low"

        return TemporalReport(
            frames_analyzed=len(brightness_values),
            brightness_variance=brightness_variance,
            color_variance=color_variance,
            flickering_detected=flickering_detected,
            flicker_frames=flicker_frames,
            severity=severity,
        )

    except Exception as e:
        logger.warning(f"Temporal analysis error: {e}")
        return TemporalReport(
            frames_analyzed=0,
            brightness_variance=0.0,
            color_variance=0.0,
            flickering_detected=False,
        )


# =============================================================================
# Audio Validation
# =============================================================================

def validate_audio_stream(audio_path: Path) -> AudioValidation:
    """Validate audio stream properties.

    Args:
        audio_path: Path to audio file

    Returns:
        AudioValidation with stream info
    """
    if not audio_path.exists():
        return AudioValidation(
            has_audio=False,
            is_valid=False,
            error_message="File does not exist",
        )

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration,bits_per_raw_sample",
            "-of", "json",
            str(audio_path)
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if proc.returncode != 0:
            return AudioValidation(
                has_audio=False,
                is_valid=False,
                error_message=f"ffprobe error: {proc.stderr}",
            )

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])

        if not streams:
            return AudioValidation(has_audio=False, is_valid=True)

        stream = streams[0]

        return AudioValidation(
            has_audio=True,
            codec=stream.get("codec_name"),
            sample_rate=int(stream.get("sample_rate", 0)),
            channels=int(stream.get("channels", 0)),
            duration=float(stream.get("duration", 0)),
            bit_depth=int(stream.get("bits_per_raw_sample", 0)) or None,
            is_valid=True,
        )

    except Exception as e:
        return AudioValidation(
            has_audio=False,
            is_valid=False,
            error_message=str(e),
        )


def validate_av_sync(
    video_path: Path,
    audio_path: Path,
    tolerance_ms: float = 100.0,
) -> Tuple[bool, float]:
    """Validate audio-video synchronization.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        tolerance_ms: Maximum allowed difference in milliseconds

    Returns:
        Tuple of (is_synced, difference_ms)
    """
    try:
        # Get video duration
        video_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]
        video_proc = subprocess.run(video_cmd, capture_output=True, text=True, timeout=10)
        video_data = json.loads(video_proc.stdout)
        video_duration = float(video_data.get("format", {}).get("duration", 0))

        # Get audio duration
        audio_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(audio_path)
        ]
        audio_proc = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=10)
        audio_data = json.loads(audio_proc.stdout)
        audio_duration = float(audio_data.get("format", {}).get("duration", 0))

        diff_ms = abs(video_duration - audio_duration) * 1000
        is_synced = diff_ms <= tolerance_ms

        if not is_synced:
            logger.warning(
                f"A/V sync issue: video={video_duration:.2f}s, "
                f"audio={audio_duration:.2f}s, diff={diff_ms:.0f}ms"
            )

        return is_synced, diff_ms

    except Exception as e:
        logger.warning(f"A/V sync check failed: {e}")
        return True, 0.0  # Assume synced if we can't check


def detect_audio_issues(
    audio_path: Path,
    silence_threshold_db: float = -50.0,
    silence_duration: float = 1.0,
) -> List[AudioIssue]:
    """Detect problematic audio segments including silence.

    Uses FFmpeg's silencedetect filter to find silent portions.

    Args:
        audio_path: Path to audio file
        silence_threshold_db: Volume threshold for silence detection (dB)
        silence_duration: Minimum duration to count as silence (seconds)

    Returns:
        List of detected AudioIssue objects
    """
    issues: List[AudioIssue] = []

    if not audio_path.exists():
        return issues

    try:
        # Run silence detection
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-af", f"silencedetect=noise={silence_threshold_db}dB:d={silence_duration}",
            "-f", "null",
            "-"
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse silence detection output from stderr
        silence_start = None
        for line in proc.stderr.split("\n"):
            if "silence_start:" in line:
                match = re.search(r"silence_start:\s*([\d.]+)", line)
                if match:
                    silence_start = float(match.group(1))

            elif "silence_end:" in line and silence_start is not None:
                match_end = re.search(r"silence_end:\s*([\d.]+)", line)
                match_dur = re.search(r"silence_duration:\s*([\d.]+)", line)

                if match_end:
                    silence_end = float(match_end.group(1))
                    duration = float(match_dur.group(1)) if match_dur else (silence_end - silence_start)

                    # Determine severity based on duration
                    if duration > 10.0:
                        severity = "high"
                    elif duration > 5.0:
                        severity = "medium"
                    else:
                        severity = "low"

                    issues.append(AudioIssue(
                        issue_type="SILENCE",
                        start_time=silence_start,
                        end_time=silence_end,
                        severity=severity,
                        details={"duration": duration},
                    ))
                    silence_start = None

    except subprocess.TimeoutExpired:
        logger.warning("Audio issue detection timed out")
    except Exception as e:
        logger.warning(f"Audio issue detection error: {e}")

    return issues


def analyze_audio_quality(audio_path: Path) -> AudioQualityReport:
    """Analyze audio for clipping, noise, and volume issues.

    Uses FFmpeg's volumedetect and astats filters.

    Args:
        audio_path: Path to audio file

    Returns:
        AudioQualityReport with analysis results
    """
    report = AudioQualityReport(audio_path=audio_path)

    if not audio_path.exists():
        report.is_acceptable = False
        return report

    try:
        # Run volume detection
        vol_cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-af", "volumedetect",
            "-f", "null",
            "-"
        ]

        vol_proc = subprocess.run(
            vol_cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse volume detection output
        for line in vol_proc.stderr.split("\n"):
            if "max_volume:" in line:
                match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
                if match:
                    report.max_volume_db = float(match.group(1))
                    # Check for clipping (max volume at or above 0 dB)
                    if report.max_volume_db >= 0.0:
                        report.clipping_detected = True

            elif "mean_volume:" in line:
                match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", line)
                if match:
                    report.mean_volume_db = float(match.group(1))

        # Run audio stats for more detailed analysis
        stats_cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-af", "astats=metadata=1:reset=1",
            "-f", "null",
            "-"
        ]

        stats_proc = subprocess.run(
            stats_cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse audio stats
        for line in stats_proc.stderr.split("\n"):
            if "Number of samples:" in line or "Number of NaNs:" in line:
                continue  # Skip

            # Look for clipping info
            if "Number of Infs:" in line:
                match = re.search(r"Number of Infs:\s*(\d+)", line)
                if match:
                    report.clipping_count = int(match.group(1))

            # Noise floor estimation from minimum RMS
            if "RMS level dB:" in line:
                match = re.search(r"RMS level dB:\s*([-\d.]+)", line)
                if match:
                    rms = float(match.group(1))
                    if rms < report.noise_floor_db:
                        report.noise_floor_db = rms

            # Dynamic range
            if "Dynamic range:" in line:
                match = re.search(r"Dynamic range:\s*([\d.]+)", line)
                if match:
                    report.dynamic_range_db = float(match.group(1))

        # Calculate dynamic range if not found
        if report.dynamic_range_db == 0.0:
            report.dynamic_range_db = report.max_volume_db - report.noise_floor_db

        # Detect silence segments
        silence_issues = detect_audio_issues(audio_path)
        for issue in silence_issues:
            if issue.issue_type == "SILENCE" and issue.end_time is not None:
                report.silence_segments.append((issue.start_time, issue.end_time))
                report.issues.append(issue)

        # Add clipping as issue if detected
        if report.clipping_detected:
            report.issues.append(AudioIssue(
                issue_type="CLIPPING",
                start_time=0.0,
                severity="high" if report.clipping_count > 100 else "medium",
                details={"count": report.clipping_count, "max_db": report.max_volume_db},
            ))

        # Check if audio is too quiet
        if report.mean_volume_db < -40.0:
            report.issues.append(AudioIssue(
                issue_type="LOW_VOLUME",
                start_time=0.0,
                severity="low",
                details={"mean_db": report.mean_volume_db},
            ))

        # Determine overall acceptability
        high_severity_count = sum(1 for i in report.issues if i.severity == "high")
        report.is_acceptable = high_severity_count == 0

    except subprocess.TimeoutExpired:
        logger.warning("Audio quality analysis timed out")
    except Exception as e:
        logger.warning(f"Audio quality analysis error: {e}")

    return report
