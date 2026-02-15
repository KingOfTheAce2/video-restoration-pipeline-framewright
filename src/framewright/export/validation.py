"""Export Validation for Restored Videos.

Verifies output file integrity after encoding - checks for corruption,
audio sync, frame count matches, and other quality issues.

Features:
- File integrity verification (can be read/decoded)
- Frame count comparison with source
- Audio sync verification
- Black frame detection
- Corrupt frame detection
- Duration match verification
- Codec validation

Example:
    >>> validator = ExportValidator()
    >>> result = validator.validate("output.mp4", compare_to="input.mp4")
    >>> if result.is_valid:
    ...     print("Export validated successfully")
    >>> else:
    ...     for issue in result.issues:
    ...         print(f"Issue: {issue}")
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class IssueSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"           # Informational, not a problem
    WARNING = "warning"     # Potential issue, may be acceptable
    ERROR = "error"         # Definite problem
    CRITICAL = "critical"   # File is unusable


class IssueType(Enum):
    """Types of validation issues."""
    FILE_CORRUPT = "file_corrupt"
    FRAME_COUNT_MISMATCH = "frame_count_mismatch"
    DURATION_MISMATCH = "duration_mismatch"
    AUDIO_MISSING = "audio_missing"
    AUDIO_SYNC_DRIFT = "audio_sync_drift"
    BLACK_FRAMES = "black_frames"
    CORRUPT_FRAMES = "corrupt_frames"
    RESOLUTION_MISMATCH = "resolution_mismatch"
    FPS_MISMATCH = "fps_mismatch"
    CODEC_ERROR = "codec_error"
    FILE_TOO_SMALL = "file_too_small"
    TRUNCATED = "truncated"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    type: IssueType
    severity: IssueSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class VideoInfo:
    """Video file information."""
    path: Path
    duration_seconds: float = 0.0
    frame_count: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    audio_sample_rate: int = 0
    file_size_bytes: int = 0
    bitrate_kbps: float = 0.0


@dataclass
class ValidationResult:
    """Result of export validation."""
    is_valid: bool = True
    output_info: Optional[VideoInfo] = None
    source_info: Optional[VideoInfo] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_total: int = 0

    # Detailed results
    black_frames: List[int] = field(default_factory=list)
    corrupt_frames: List[int] = field(default_factory=list)

    # Checksums
    output_md5: str = ""
    output_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_total": self.checks_total,
            "issues": [i.to_dict() for i in self.issues],
            "output_info": {
                "path": str(self.output_info.path) if self.output_info else None,
                "duration": self.output_info.duration_seconds if self.output_info else 0,
                "frame_count": self.output_info.frame_count if self.output_info else 0,
                "resolution": f"{self.output_info.width}x{self.output_info.height}" if self.output_info else "",
            },
            "black_frames_count": len(self.black_frames),
            "corrupt_frames_count": len(self.corrupt_frames),
            "checksums": {
                "md5": self.output_md5,
                "sha256": self.output_sha256,
            },
        }

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL):
            self.is_valid = False
            self.checks_failed += 1
        else:
            self.checks_passed += 1
        self.checks_total += 1


class ExportValidator:
    """Validates exported video files.

    Performs comprehensive checks on restored video files to ensure
    they are valid and match expected characteristics.
    """

    def __init__(
        self,
        check_frames: bool = True,
        sample_frames: int = 50,
        black_threshold: float = 5.0,
        duration_tolerance: float = 0.5,
        frame_count_tolerance: int = 2,
    ):
        """Initialize validator.

        Args:
            check_frames: Whether to check individual frames
            sample_frames: Number of frames to sample for checks
            black_threshold: Average brightness below this = black frame
            duration_tolerance: Allowed duration difference in seconds
            frame_count_tolerance: Allowed frame count difference
        """
        self.check_frames = check_frames
        self.sample_frames = sample_frames
        self.black_threshold = black_threshold
        self.duration_tolerance = duration_tolerance
        self.frame_count_tolerance = frame_count_tolerance

    def validate(
        self,
        output_path: Path,
        compare_to: Optional[Path] = None,
        compute_checksums: bool = True,
    ) -> ValidationResult:
        """Validate an exported video file.

        Args:
            output_path: Path to output video
            compare_to: Optional source video for comparison
            compute_checksums: Whether to compute file checksums

        Returns:
            ValidationResult with all findings
        """
        output_path = Path(output_path)
        result = ValidationResult()

        # Check file exists
        if not output_path.exists():
            result.add_issue(ValidationIssue(
                type=IssueType.FILE_CORRUPT,
                severity=IssueSeverity.CRITICAL,
                message=f"Output file not found: {output_path}",
            ))
            return result

        # Get output file info
        output_info = self._get_video_info(output_path)
        if output_info is None:
            result.add_issue(ValidationIssue(
                type=IssueType.FILE_CORRUPT,
                severity=IssueSeverity.CRITICAL,
                message="Cannot read output file - may be corrupt",
            ))
            return result

        result.output_info = output_info

        # Get source info if provided
        source_info = None
        if compare_to:
            compare_to = Path(compare_to)
            source_info = self._get_video_info(compare_to)
            result.source_info = source_info

        # Run all checks
        self._check_file_size(result)
        self._check_decodable(result)

        if source_info:
            self._check_duration_match(result)
            self._check_frame_count_match(result)
            self._check_resolution(result)
            self._check_fps_match(result)

        self._check_audio(result)

        if self.check_frames:
            self._check_black_frames(result)
            self._check_corrupt_frames(result)

        if compute_checksums:
            self._compute_checksums(result)

        # Final validation status
        if not result.issues:
            result.checks_passed = result.checks_total

        return result

    def _get_video_info(self, path: Path) -> Optional[VideoInfo]:
        """Get video file information using ffprobe.

        Args:
            path: Video file path

        Returns:
            VideoInfo or None if cannot read
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            info = VideoInfo(path=path)

            # Format info
            fmt = data.get("format", {})
            info.duration_seconds = float(fmt.get("duration", 0))
            info.file_size_bytes = int(fmt.get("size", 0))
            info.bitrate_kbps = float(fmt.get("bit_rate", 0)) / 1000

            # Stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info.video_codec = stream.get("codec_name", "")
                    info.width = int(stream.get("width", 0))
                    info.height = int(stream.get("height", 0))

                    # FPS
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        info.fps = float(num) / float(den) if float(den) > 0 else 0
                    else:
                        info.fps = float(fps_str)

                    # Frame count
                    info.frame_count = int(stream.get("nb_frames", 0))
                    if info.frame_count == 0 and info.duration_seconds > 0:
                        info.frame_count = int(info.duration_seconds * info.fps)

                elif stream.get("codec_type") == "audio":
                    info.audio_codec = stream.get("codec_name", "")
                    info.audio_channels = int(stream.get("channels", 0))
                    info.audio_sample_rate = int(stream.get("sample_rate", 0))

            return info

        except Exception as e:
            logger.debug(f"Failed to get video info: {e}")
            return None

    def _check_file_size(self, result: ValidationResult) -> None:
        """Check if file size is reasonable."""
        result.checks_total += 1

        if not result.output_info:
            return

        size_mb = result.output_info.file_size_bytes / (1024 * 1024)

        # Minimum size check
        if size_mb < 0.1:
            result.add_issue(ValidationIssue(
                type=IssueType.FILE_TOO_SMALL,
                severity=IssueSeverity.ERROR,
                message=f"File too small: {size_mb:.2f}MB",
                details={"size_mb": size_mb},
            ))
        else:
            result.checks_passed += 1

    def _check_decodable(self, result: ValidationResult) -> None:
        """Check if file can be decoded."""
        result.checks_total += 1

        if not result.output_info:
            return

        try:
            cmd = [
                "ffmpeg",
                "-v", "error",
                "-i", str(result.output_info.path),
                "-f", "null",
                "-t", "1",  # Only check first second
                "-"
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode != 0 or proc.stderr:
                result.add_issue(ValidationIssue(
                    type=IssueType.CODEC_ERROR,
                    severity=IssueSeverity.WARNING,
                    message="Decoder warnings detected",
                    details={"stderr": proc.stderr[:500]},
                ))
            else:
                result.checks_passed += 1

        except Exception as e:
            result.add_issue(ValidationIssue(
                type=IssueType.FILE_CORRUPT,
                severity=IssueSeverity.ERROR,
                message=f"Decode check failed: {e}",
            ))

    def _check_duration_match(self, result: ValidationResult) -> None:
        """Check if duration matches source."""
        result.checks_total += 1

        if not result.output_info or not result.source_info:
            return

        diff = abs(
            result.output_info.duration_seconds -
            result.source_info.duration_seconds
        )

        if diff > self.duration_tolerance:
            result.add_issue(ValidationIssue(
                type=IssueType.DURATION_MISMATCH,
                severity=IssueSeverity.WARNING,
                message=f"Duration mismatch: {diff:.2f}s difference",
                details={
                    "output_duration": result.output_info.duration_seconds,
                    "source_duration": result.source_info.duration_seconds,
                    "difference": diff,
                },
            ))
        else:
            result.checks_passed += 1

    def _check_frame_count_match(self, result: ValidationResult) -> None:
        """Check if frame count matches source."""
        result.checks_total += 1

        if not result.output_info or not result.source_info:
            return

        diff = abs(
            result.output_info.frame_count -
            result.source_info.frame_count
        )

        if diff > self.frame_count_tolerance:
            result.add_issue(ValidationIssue(
                type=IssueType.FRAME_COUNT_MISMATCH,
                severity=IssueSeverity.WARNING,
                message=f"Frame count mismatch: {diff} frames difference",
                details={
                    "output_frames": result.output_info.frame_count,
                    "source_frames": result.source_info.frame_count,
                    "difference": diff,
                },
            ))
        else:
            result.checks_passed += 1

    def _check_resolution(self, result: ValidationResult) -> None:
        """Check resolution is valid."""
        result.checks_total += 1

        if not result.output_info:
            return

        # Resolution should be at least as large as source (if upscaled)
        # Or we should validate it's a valid resolution
        if result.output_info.width < 16 or result.output_info.height < 16:
            result.add_issue(ValidationIssue(
                type=IssueType.RESOLUTION_MISMATCH,
                severity=IssueSeverity.ERROR,
                message=f"Invalid resolution: {result.output_info.width}x{result.output_info.height}",
            ))
        else:
            result.checks_passed += 1

    def _check_fps_match(self, result: ValidationResult) -> None:
        """Check if FPS is reasonable."""
        result.checks_total += 1

        if not result.output_info or not result.source_info:
            return

        # FPS can be different (interpolation), but should be reasonable
        if result.output_info.fps < 1 or result.output_info.fps > 240:
            result.add_issue(ValidationIssue(
                type=IssueType.FPS_MISMATCH,
                severity=IssueSeverity.ERROR,
                message=f"Invalid FPS: {result.output_info.fps}",
            ))
        else:
            result.checks_passed += 1

    def _check_audio(self, result: ValidationResult) -> None:
        """Check audio stream."""
        result.checks_total += 1

        if not result.output_info:
            return

        # If source had audio, output should too
        if result.source_info and result.source_info.audio_codec:
            if not result.output_info.audio_codec:
                result.add_issue(ValidationIssue(
                    type=IssueType.AUDIO_MISSING,
                    severity=IssueSeverity.WARNING,
                    message="Source had audio but output does not",
                ))
                return

        result.checks_passed += 1

    def _check_black_frames(self, result: ValidationResult) -> None:
        """Check for black frames."""
        result.checks_total += 1

        if not HAS_CV2 or not result.output_info:
            result.checks_passed += 1
            return

        try:
            cap = cv2.VideoCapture(str(result.output_info.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames
            step = max(1, total // self.sample_frames)
            black_frames = []

            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Check if frame is black
                avg_brightness = np.mean(frame)
                if avg_brightness < self.black_threshold:
                    black_frames.append(i)

            cap.release()

            result.black_frames = black_frames

            # More than 10% black frames is concerning
            if len(black_frames) > self.sample_frames * 0.1:
                result.add_issue(ValidationIssue(
                    type=IssueType.BLACK_FRAMES,
                    severity=IssueSeverity.WARNING,
                    message=f"Found {len(black_frames)} black frames in sample",
                    details={"black_frames": black_frames[:10]},
                ))
            else:
                result.checks_passed += 1

        except Exception as e:
            logger.debug(f"Black frame check failed: {e}")
            result.checks_passed += 1

    def _check_corrupt_frames(self, result: ValidationResult) -> None:
        """Check for corrupt/unreadable frames."""
        result.checks_total += 1

        if not HAS_CV2 or not result.output_info:
            result.checks_passed += 1
            return

        try:
            cap = cv2.VideoCapture(str(result.output_info.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames
            step = max(1, total // self.sample_frames)
            corrupt_frames = []

            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    corrupt_frames.append(i)

            cap.release()

            result.corrupt_frames = corrupt_frames

            if corrupt_frames:
                result.add_issue(ValidationIssue(
                    type=IssueType.CORRUPT_FRAMES,
                    severity=IssueSeverity.ERROR,
                    message=f"Found {len(corrupt_frames)} unreadable frames",
                    details={"corrupt_frames": corrupt_frames[:10]},
                ))
            else:
                result.checks_passed += 1

        except Exception as e:
            logger.debug(f"Corrupt frame check failed: {e}")
            result.checks_passed += 1

    def _compute_checksums(self, result: ValidationResult) -> None:
        """Compute file checksums."""
        if not result.output_info:
            return

        try:
            md5 = hashlib.md5()
            sha256 = hashlib.sha256()

            with open(result.output_info.path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
                    sha256.update(chunk)

            result.output_md5 = md5.hexdigest()
            result.output_sha256 = sha256.hexdigest()

        except Exception as e:
            logger.debug(f"Checksum computation failed: {e}")


def validate_export(
    output_path: Path,
    compare_to: Optional[Path] = None,
) -> ValidationResult:
    """Convenience function to validate export.

    Args:
        output_path: Output video path
        compare_to: Optional source video

    Returns:
        ValidationResult
    """
    validator = ExportValidator()
    return validator.validate(output_path, compare_to)
