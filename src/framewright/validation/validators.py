"""Frame and output validation."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ValidationIssue:
    """A validation issue detected in a frame or output."""
    severity: IssueSeverity
    code: str
    message: str
    frame_number: Optional[int] = None
    location: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL):
            self.passed = False


class FrameValidator:
    """Validates individual frames for quality issues."""

    def __init__(
        self,
        check_black_frames: bool = True,
        check_white_frames: bool = True,
        check_corrupt: bool = True,
        check_blur: bool = True,
        check_artifacts: bool = True,
        blur_threshold: float = 100.0,
        black_threshold: float = 5.0,
        white_threshold: float = 250.0,
    ):
        self.check_black_frames = check_black_frames
        self.check_white_frames = check_white_frames
        self.check_corrupt = check_corrupt
        self.check_blur = check_blur
        self.check_artifacts = check_artifacts
        self.blur_threshold = blur_threshold
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold

    def validate(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        original: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """Validate a frame.

        Args:
            frame: Frame to validate (BGR, uint8)
            frame_number: Frame index
            original: Optional original frame for comparison

        Returns:
            Validation result
        """
        result = ValidationResult(passed=True)

        if frame is None:
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="NULL_FRAME",
                message="Frame is null",
                frame_number=frame_number,
            ))
            return result

        # Basic checks
        if self.check_corrupt:
            self._check_corrupt(frame, frame_number, result)

        if self.check_black_frames:
            self._check_black_frame(frame, frame_number, result)

        if self.check_white_frames:
            self._check_white_frame(frame, frame_number, result)

        if self.check_blur:
            self._check_blur(frame, frame_number, result)

        if self.check_artifacts:
            self._check_artifacts(frame, frame_number, result)

        # Comparison with original
        if original is not None:
            self._check_vs_original(frame, original, frame_number, result)

        return result

    def _check_corrupt(
        self,
        frame: np.ndarray,
        frame_number: int,
        result: ValidationResult,
    ) -> None:
        """Check for corrupt frames."""
        # Check for NaN or Inf values
        if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="CORRUPT_VALUES",
                message="Frame contains NaN or Inf values",
                frame_number=frame_number,
            ))
            return

        # Check for valid dimensions
        if len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="INVALID_DIMENSIONS",
                message=f"Invalid frame dimensions: {frame.shape}",
                frame_number=frame_number,
            ))

    def _check_black_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        result: ValidationResult,
    ) -> None:
        """Check for black frames."""
        mean_value = np.mean(frame)
        result.metrics["mean_brightness"] = float(mean_value)

        if mean_value < self.black_threshold:
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="BLACK_FRAME",
                message=f"Frame appears black (mean: {mean_value:.1f})",
                frame_number=frame_number,
                details={"mean_value": float(mean_value)},
            ))

    def _check_white_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        result: ValidationResult,
    ) -> None:
        """Check for white frames."""
        mean_value = np.mean(frame)

        if mean_value > self.white_threshold:
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="WHITE_FRAME",
                message=f"Frame appears white (mean: {mean_value:.1f})",
                frame_number=frame_number,
                details={"mean_value": float(mean_value)},
            ))

    def _check_blur(
        self,
        frame: np.ndarray,
        frame_number: int,
        result: ValidationResult,
    ) -> None:
        """Check for blurry frames."""
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            result.metrics["sharpness"] = float(laplacian_var)

            if laplacian_var < self.blur_threshold:
                result.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="BLURRY_FRAME",
                    message=f"Frame appears blurry (sharpness: {laplacian_var:.1f})",
                    frame_number=frame_number,
                    details={"laplacian_var": float(laplacian_var)},
                ))
        except ImportError:
            pass

    def _check_artifacts(
        self,
        frame: np.ndarray,
        frame_number: int,
        result: ValidationResult,
    ) -> None:
        """Check for processing artifacts."""
        # Check for large uniform regions (potential artifacts)
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Check for blocks of identical values
            h, w = gray.shape
            block_size = 16
            uniform_blocks = 0
            total_blocks = 0

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    if np.std(block) < 1.0:
                        uniform_blocks += 1
                    total_blocks += 1

            if total_blocks > 0:
                uniform_ratio = uniform_blocks / total_blocks
                result.metrics["uniform_ratio"] = uniform_ratio

                if uniform_ratio > 0.3:
                    result.add_issue(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        code="POTENTIAL_ARTIFACT",
                        message=f"High uniform block ratio ({uniform_ratio:.1%})",
                        frame_number=frame_number,
                        details={"uniform_ratio": uniform_ratio},
                    ))
        except ImportError:
            pass

    def _check_vs_original(
        self,
        frame: np.ndarray,
        original: np.ndarray,
        frame_number: int,
        result: ValidationResult,
    ) -> None:
        """Compare against original frame."""
        if frame.shape != original.shape:
            # Size changed (expected for upscaling)
            return

        # Calculate PSNR
        mse = np.mean((frame.astype(float) - original.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
            result.metrics["psnr_vs_original"] = float(psnr)

            # If PSNR is very low, output differs significantly
            if psnr < 10:
                result.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="HIGH_DIFF_ORIGINAL",
                    message=f"Large difference from original (PSNR: {psnr:.1f}dB)",
                    frame_number=frame_number,
                    details={"psnr": float(psnr)},
                ))


class OutputValidator:
    """Validates output video files."""

    def __init__(
        self,
        check_duration: bool = True,
        check_codec: bool = True,
        check_audio: bool = True,
        max_duration_diff_seconds: float = 1.0,
    ):
        self.check_duration = check_duration
        self.check_codec = check_codec
        self.check_audio = check_audio
        self.max_duration_diff_seconds = max_duration_diff_seconds

    def validate(
        self,
        output_path: Path,
        input_path: Optional[Path] = None,
    ) -> ValidationResult:
        """Validate output video file.

        Args:
            output_path: Path to output video
            input_path: Optional input path for comparison

        Returns:
            Validation result
        """
        result = ValidationResult(passed=True)
        output_path = Path(output_path)

        # Check file exists
        if not output_path.exists():
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="FILE_NOT_FOUND",
                message=f"Output file not found: {output_path}",
            ))
            return result

        # Check file size
        file_size = output_path.stat().st_size
        result.metrics["file_size_mb"] = file_size / (1024 * 1024)

        if file_size < 1024:  # Less than 1KB
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="FILE_TOO_SMALL",
                message=f"Output file too small: {file_size} bytes",
            ))
            return result

        # Get output metadata
        output_info = self._get_video_info(output_path)
        if not output_info:
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.ERROR,
                code="UNREADABLE",
                message="Could not read output video metadata",
            ))
            return result

        result.metrics.update(output_info)

        # Compare with input if available
        if input_path:
            input_info = self._get_video_info(input_path)
            if input_info:
                self._compare_videos(input_info, output_info, result)

        # Check video integrity
        self._check_integrity(output_path, result)

        return result

    def _get_video_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """Get video metadata."""
        import json
        import subprocess

        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                info = {"streams": []}

                # Format info
                fmt = data.get("format", {})
                info["duration"] = float(fmt.get("duration", 0))
                info["bitrate"] = int(fmt.get("bit_rate", 0))

                # Stream info
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        info["video_codec"] = stream.get("codec_name")
                        info["width"] = stream.get("width")
                        info["height"] = stream.get("height")
                        info["fps"] = eval(stream.get("r_frame_rate", "0/1"))
                    elif stream.get("codec_type") == "audio":
                        info["has_audio"] = True
                        info["audio_codec"] = stream.get("codec_name")

                return info

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")

        return None

    def _compare_videos(
        self,
        input_info: Dict,
        output_info: Dict,
        result: ValidationResult,
    ) -> None:
        """Compare input and output videos."""
        # Duration check
        if self.check_duration:
            in_dur = input_info.get("duration", 0)
            out_dur = output_info.get("duration", 0)
            diff = abs(in_dur - out_dur)

            if diff > self.max_duration_diff_seconds:
                result.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="DURATION_MISMATCH",
                    message=f"Duration differs by {diff:.2f}s",
                    details={"input_duration": in_dur, "output_duration": out_dur},
                ))

        # Audio check
        if self.check_audio:
            in_audio = input_info.get("has_audio", False)
            out_audio = output_info.get("has_audio", False)

            if in_audio and not out_audio:
                result.add_issue(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="AUDIO_MISSING",
                    message="Output is missing audio that was present in input",
                ))

    def _check_integrity(self, path: Path, result: ValidationResult) -> None:
        """Check video file integrity."""
        import subprocess

        try:
            # Quick integrity check with ffmpeg
            cmd = [
                "ffmpeg", "-v", "error",
                "-i", str(path),
                "-f", "null", "-",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if proc.stderr:
                errors = proc.stderr.strip().split("\n")
                for error in errors[:5]:  # Limit to first 5 errors
                    result.add_issue(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        code="STREAM_ERROR",
                        message=error.strip(),
                    ))

        except subprocess.TimeoutExpired:
            result.add_issue(ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="CHECK_TIMEOUT",
                message="Integrity check timed out",
            ))
        except Exception as e:
            logger.warning(f"Integrity check failed: {e}")
