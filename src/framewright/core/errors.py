"""Unified error handling system for FrameWright.

This module provides a comprehensive, unified exception hierarchy for all
FrameWright operations. Each exception includes:
- Clear, user-friendly error messages
- Suggested fixes (Apple-like helpfulness)
- Detailed context for debugging
- Proper exception chaining

Exception Hierarchy:
    FramewrightError (base)
    +-- ConfigurationError
    +-- ProcessingError
    |   +-- EnhancementError
    |   +-- InterpolationError
    |   +-- FrameExtractionError
    |   +-- ReassemblyError
    +-- ModelError
    +-- HardwareError
    |   +-- GPUError
    |   |   +-- VRAMError (alias: OutOfMemoryError)
    |   +-- CPUFallbackError
    |   +-- GPURequiredError
    +-- VideoError
    |   +-- MetadataError
    |   +-- AudioExtractionError
    +-- NetworkError
    |   +-- DownloadError
    |   +-- TimeoutError
    +-- StorageError
    |   +-- DiskSpaceError
    |   +-- CheckpointError
    +-- ValidationError
    +-- DependencyError
    +-- CorruptionError

Transient vs Fatal Classification:
    TransientError - May succeed on retry (VRAM, disk, network)
    FatalError - Requires user intervention (corruption, missing deps)
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Base Exception with User-Friendly Features
# =============================================================================


class FramewrightError(Exception):
    """Base exception for all FrameWright errors.

    Provides user-friendly error messages with suggested fixes,
    detailed context, and proper exception chaining.

    Attributes:
        message: Human-readable error description
        details: Optional dictionary with additional context
        cause: Original exception that caused this error (if any)
        context: ErrorContext object for detailed debugging
    """

    # Default user-friendly messages and fixes (override in subclasses)
    _default_user_message = "An unexpected error occurred during video processing."
    _default_suggested_fix = "Please check the logs for more details and try again."

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        """Initialize FramewrightError.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
            cause: Original exception that caused this error
            context: ErrorContext object for detailed debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.context = context

        # Chain the cause exception for proper traceback
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return string representation with details if available."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message

    def user_message(self) -> str:
        """Return a user-friendly error message.

        Override in subclasses for specific, helpful messages.

        Returns:
            A clear, non-technical description of what went wrong.
        """
        return self._default_user_message

    def suggested_fix(self) -> str:
        """Return a suggested fix for this error.

        Override in subclasses for specific, actionable suggestions.

        Returns:
            A clear, actionable suggestion for resolving the error.
        """
        return self._default_suggested_fix

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization.

        Returns:
            Dictionary with error type, message, details, and suggested fix.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "user_message": self.user_message(),
            "suggested_fix": self.suggested_fix(),
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
            "context": self.context.to_dict() if self.context else None,
        }

    @property
    def is_transient(self) -> bool:
        """Check if this error is potentially transient (may succeed on retry).

        Transient errors include VRAM exhaustion, disk space, and network issues.
        Override in subclasses to indicate transient nature.

        Returns:
            True if the error may succeed on retry, False otherwise.
        """
        return isinstance(self, TransientError)

    @property
    def is_fatal(self) -> bool:
        """Check if this error is fatal (requires user intervention).

        Fatal errors include configuration issues, missing dependencies, and corruption.
        Override in subclasses to indicate fatal nature.

        Returns:
            True if the error requires user intervention, False otherwise.
        """
        return isinstance(self, FatalError)

    def format_for_user(self) -> str:
        """Format error for display to end user.

        Returns:
            A nicely formatted error message with suggested fix.
        """
        error_type = "RECOVERABLE ERROR" if self.is_transient else "ERROR"
        lines = [
            "",
            "=" * 60,
            f"{error_type}: {self.__class__.__name__}",
            "=" * 60,
            "",
            self.user_message(),
            "",
            "What to try:",
            "  " + self.suggested_fix(),
            "",
        ]
        if self.details:
            lines.append("Details:")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Transient vs Fatal Base Classes
# =============================================================================


class TransientError(FramewrightError):
    """Recoverable errors that may succeed on retry.

    These errors are typically caused by temporary conditions like
    resource exhaustion, network issues, or race conditions.
    """

    _default_user_message = "A temporary error occurred that may resolve on retry."
    _default_suggested_fix = "Wait a moment and try again. The operation may succeed."


class FatalError(FramewrightError):
    """Non-recoverable errors requiring intervention.

    These errors indicate fundamental problems that cannot be
    resolved by retrying.
    """

    _default_user_message = "An error occurred that requires your attention to resolve."
    _default_suggested_fix = "Please review the error details and take corrective action."


# =============================================================================
# Hardware Errors
# =============================================================================


class HardwareError(FramewrightError):
    """Hardware compatibility error.

    Base class for hardware-related errors including GPU detection
    failures, driver issues, and hardware capability problems.
    """

    _default_user_message = "A hardware compatibility issue was detected."
    _default_suggested_fix = (
        "Ensure your hardware meets the requirements and drivers are up to date."
    )

    def __init__(
        self,
        message: str,
        hardware_type: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if hardware_type:
            details["hardware_type"] = hardware_type
        if device_info:
            details["device_info"] = device_info
        super().__init__(message, details=details, cause=cause, context=context)


class GPUError(HardwareError):
    """GPU-specific error.

    Raised for GPU-related issues that are not memory-related,
    such as driver issues, compute capability problems, or
    CUDA/Vulkan initialization failures.
    """

    _default_user_message = (
        "Your GPU encountered an error during video processing."
    )
    _default_suggested_fix = (
        "Try updating your GPU drivers. If using NVIDIA, ensure CUDA is properly installed. "
        "You can also try running with --cpu-only flag."
    )

    def __init__(
        self,
        message: str,
        gpu_id: Optional[int] = None,
        gpu_name: Optional[str] = None,
        driver_version: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        device_info = {}
        if gpu_id is not None:
            device_info["gpu_id"] = gpu_id
        if gpu_name:
            device_info["gpu_name"] = gpu_name
        if driver_version:
            device_info["driver_version"] = driver_version
        super().__init__(
            message,
            hardware_type="gpu",
            device_info=device_info if device_info else None,
            cause=cause,
            context=context,
        )


class VRAMError(GPUError):
    """GPU VRAM exhaustion error.

    Raised when GPU video memory is exhausted during processing.
    This error is often recoverable by reducing tile size
    or processing fewer frames in parallel.
    """

    _default_user_message = (
        "Your GPU ran out of video memory (VRAM) while processing frames."
    )
    _default_suggested_fix = (
        "Try reducing the tile size with --tile-size 128 or lower. "
        "You can also close other GPU-intensive applications or try processing "
        "at a lower resolution first."
    )

    def __init__(
        self,
        message: str,
        required_mb: Optional[int] = None,
        available_mb: Optional[int] = None,
        gpu_id: Optional[int] = None,
        gpu_name: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            gpu_id=gpu_id,
            gpu_name=gpu_name,
            cause=cause,
            context=context,
        )
        if required_mb is not None:
            self.details["required_mb"] = required_mb
        if available_mb is not None:
            self.details["available_mb"] = available_mb

    @property
    def recoverable(self) -> bool:
        """Check if error is potentially recoverable by reducing tile size."""
        return True

    @property
    def is_transient(self) -> bool:
        """VRAM errors are transient - may succeed with smaller tile size."""
        return True

    def user_message(self) -> str:
        if self.details.get("required_mb") and self.details.get("available_mb"):
            return (
                f"Your GPU ran out of video memory. "
                f"Needed: {self.details['required_mb']}MB, "
                f"Available: {self.details['available_mb']}MB."
            )
        return self._default_user_message


# Alias for backward compatibility
OutOfMemoryError = VRAMError


class GPURequiredError(HardwareError):
    """GPU required but not available or not working.

    Raised when require_gpu=True but no GPU is detected or GPU
    processing fails and would fall back to CPU.
    """

    _default_user_message = (
        "GPU processing was required, but no compatible GPU was found."
    )
    _default_suggested_fix = (
        "Ensure you have a compatible NVIDIA GPU with CUDA support, or "
        "run with --allow-cpu to enable CPU fallback (much slower)."
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            hardware_type="gpu",
            cause=cause,
            context=context,
        )


class CPUFallbackError(HardwareError):
    """CPU fallback detected when GPU required.

    Raised when processing starts using CPU instead of GPU
    and require_gpu=True in config.
    """

    _default_user_message = (
        "Processing fell back to CPU, but GPU-only mode was enabled."
    )
    _default_suggested_fix = (
        "Check that your GPU drivers are properly installed and CUDA is working. "
        "Run 'nvidia-smi' to verify GPU status. If you want to allow CPU processing, "
        "remove the --require-gpu flag."
    )

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            hardware_type="cpu_fallback",
            cause=cause,
            context=context,
        )


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(FatalError):
    """Invalid configuration.

    Raised when configuration values are invalid, missing required
    fields, or have incompatible combinations.
    """

    _default_user_message = "The provided configuration is invalid."
    _default_suggested_fix = (
        "Review your configuration file or command-line arguments. "
        "Run with --help to see valid options."
    )

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        valid_values: Optional[list] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value
        if valid_values:
            details["valid_values"] = valid_values
        super().__init__(message, details=details, cause=cause, context=context)

    def user_message(self) -> str:
        if self.details.get("config_key"):
            key = self.details["config_key"]
            if self.details.get("valid_values"):
                return (
                    f"Invalid value for '{key}'. "
                    f"Valid options: {self.details['valid_values']}"
                )
            return f"Invalid configuration for '{key}'."
        return self._default_user_message


# =============================================================================
# Processing Errors
# =============================================================================


class ProcessingError(FramewrightError):
    """Error during frame processing.

    Raised when frame enhancement, interpolation, or other
    processing operations fail.
    """

    _default_user_message = "An error occurred while processing video frames."
    _default_suggested_fix = (
        "Try processing with a smaller tile size or check if the input video is valid."
    )

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        frame_number: Optional[int] = None,
        frame_path: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if stage:
            details["stage"] = stage
        if frame_number is not None:
            details["frame_number"] = frame_number
        if frame_path:
            details["frame_path"] = frame_path
        super().__init__(message, details=details, cause=cause, context=context)

    def user_message(self) -> str:
        if self.details.get("stage") and self.details.get("frame_number") is not None:
            return (
                f"Processing failed at {self.details['stage']} stage, "
                f"frame {self.details['frame_number']}."
            )
        elif self.details.get("stage"):
            return f"Processing failed during {self.details['stage']} stage."
        return self._default_user_message


class EnhancementError(ProcessingError):
    """Frame enhancement error.

    Raised when Real-ESRGAN or other enhancement operations fail.
    """

    _default_user_message = "Frame enhancement (upscaling) failed."
    _default_suggested_fix = (
        "Try using a smaller tile size (--tile-size 128) or a different model. "
        "If using custom models, verify the model file is not corrupted."
    )

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        scale_factor: Optional[int] = None,
        tile_size: Optional[int] = None,
        frame_number: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            stage="enhancement",
            frame_number=frame_number,
            cause=cause,
            context=context,
        )
        if model_name:
            self.details["model_name"] = model_name
        if scale_factor is not None:
            self.details["scale_factor"] = scale_factor
        if tile_size is not None:
            self.details["tile_size"] = tile_size


class InterpolationError(ProcessingError):
    """Frame interpolation error.

    Raised when RIFE or other interpolation operations fail.
    """

    _default_user_message = "Frame interpolation (FPS boost) failed."
    _default_suggested_fix = (
        "Try a lower target FPS or disable interpolation with --no-interpolation. "
        "Scene cuts can sometimes cause issues - try with --scene-detection enabled."
    )

    def __init__(
        self,
        message: str,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        model: Optional[str] = None,
        frame_number: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            stage="interpolation",
            frame_number=frame_number,
            cause=cause,
            context=context,
        )
        if source_fps is not None:
            self.details["source_fps"] = source_fps
        if target_fps is not None:
            self.details["target_fps"] = target_fps
        if model:
            self.details["model"] = model


class FrameExtractionError(ProcessingError):
    """Error during frame extraction.

    Raised when extracting frames from the source video fails.
    """

    _default_user_message = "Failed to extract frames from the video."
    _default_suggested_fix = (
        "The video file may be corrupted or use an unsupported codec. "
        "Try converting it to a standard format (MP4/H.264) first using ffmpeg."
    )

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        frame_number: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            stage="frame_extraction",
            frame_number=frame_number,
            cause=cause,
            context=context,
        )
        if video_path:
            self.details["video_path"] = video_path


class ReassemblyError(ProcessingError):
    """Error during video reassembly.

    Raised when assembling processed frames back into a video fails.
    """

    _default_user_message = "Failed to reassemble processed frames into a video."
    _default_suggested_fix = (
        "Check that you have enough disk space and ffmpeg is properly installed. "
        "Try a different output codec with --codec libx264."
    )

    def __init__(
        self,
        message: str,
        output_path: Optional[str] = None,
        codec: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            stage="reassembly",
            cause=cause,
            context=context,
        )
        if output_path:
            self.details["output_path"] = output_path
        if codec:
            self.details["codec"] = codec


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(FramewrightError):
    """Model loading or inference error.

    Raised when AI models fail to load, have incorrect format,
    or encounter errors during inference.
    """

    _default_user_message = "Failed to load or run the AI model."
    _default_suggested_fix = (
        "Try re-downloading the model with 'framewright models download'. "
        "If using a custom model, verify it's compatible with FrameWright."
    )

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        if model_type:
            details["model_type"] = model_type
        super().__init__(message, details=details, cause=cause, context=context)

    def user_message(self) -> str:
        if self.details.get("model_name"):
            return f"Failed to load model '{self.details['model_name']}'."
        return self._default_user_message


# =============================================================================
# Video Errors
# =============================================================================


class VideoError(FramewrightError):
    """Video file reading/writing error.

    Raised when video files cannot be read, written, or parsed.
    """

    _default_user_message = "An error occurred while reading or writing the video file."
    _default_suggested_fix = (
        "Verify the video file exists and is not corrupted. "
        "Try playing it in a video player first. For output errors, "
        "check write permissions and available disk space."
    )

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        operation: Optional[str] = None,
        codec: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if video_path:
            details["video_path"] = video_path
        if operation:
            details["operation"] = operation
        if codec:
            details["codec"] = codec
        super().__init__(message, details=details, cause=cause, context=context)


class MetadataError(VideoError):
    """Error during metadata extraction.

    Raised when video metadata cannot be read or is invalid.
    """

    _default_user_message = "Failed to read video metadata (duration, resolution, etc.)."
    _default_suggested_fix = (
        "The video file may be corrupted or have missing headers. "
        "Try remuxing it with: ffmpeg -i input.mp4 -c copy output.mp4"
    )

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        missing_fields: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            video_path=video_path,
            operation="metadata_extraction",
            cause=cause,
            context=context,
        )
        if missing_fields:
            self.details["missing_fields"] = missing_fields


class AudioExtractionError(VideoError):
    """Error during audio extraction.

    Raised when audio stream cannot be extracted from video.
    """

    _default_user_message = "Failed to extract audio from the video."
    _default_suggested_fix = (
        "The video may not have an audio track, or the audio codec is unsupported. "
        "Try processing with --no-audio to skip audio processing."
    )

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        audio_codec: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(
            message,
            video_path=video_path,
            operation="audio_extraction",
            codec=audio_codec,
            cause=cause,
            context=context,
        )


# =============================================================================
# Network Errors
# =============================================================================


class NetworkError(TransientError):
    """Network-related failures.

    Base class for network errors that are typically transient.
    """

    _default_user_message = "A network error occurred."
    _default_suggested_fix = (
        "Check your internet connection and try again. "
        "If behind a firewall or proxy, ensure the required URLs are accessible."
    )

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if url:
            details["url"] = url
        super().__init__(message, details=details, cause=cause, context=context)


class DownloadError(NetworkError):
    """Model or video download error.

    Raised when downloading files fails due to network issues,
    authentication problems, or server errors.
    """

    _default_user_message = "Failed to download a required file."
    _default_suggested_fix = (
        "Check your internet connection and try again. "
        "You can also manually download models and place them in the models directory."
    )

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        downloaded_bytes: Optional[int] = None,
        expected_bytes: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(message, url=url, cause=cause, context=context)
        if status_code is not None:
            self.details["status_code"] = status_code
        if downloaded_bytes is not None:
            self.details["downloaded_bytes"] = downloaded_bytes
        if expected_bytes is not None:
            self.details["expected_bytes"] = expected_bytes

    def user_message(self) -> str:
        if self.details.get("status_code") == 404:
            return "The requested file was not found on the server."
        elif self.details.get("status_code"):
            return f"Download failed with HTTP error {self.details['status_code']}."
        return self._default_user_message


class TimeoutError(NetworkError):
    """Operation timeout error.

    Raised when an operation takes too long to complete.
    """

    _default_user_message = "The operation timed out."
    _default_suggested_fix = (
        "Try again - the server may be temporarily slow. "
        "You can increase the timeout with --timeout option if needed."
    )

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        super().__init__(message, cause=cause, context=context)
        if timeout_seconds is not None:
            self.details["timeout_seconds"] = timeout_seconds
        if operation:
            self.details["operation"] = operation


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(FramewrightError):
    """Storage-related error base class."""

    _default_user_message = "A storage error occurred."
    _default_suggested_fix = "Check disk space and file permissions."


class DiskSpaceError(StorageError):
    """Insufficient disk space error.

    Raised when there is not enough disk space for operations.
    """

    _default_user_message = "Not enough disk space available."
    _default_suggested_fix = (
        "Free up disk space by deleting unnecessary files. "
        "Video processing requires significant temporary space - "
        "aim for at least 10GB free for HD content, more for 4K."
    )

    def __init__(
        self,
        message: str,
        required_gb: Optional[float] = None,
        available_gb: Optional[float] = None,
        path: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if required_gb is not None:
            details["required_gb"] = required_gb
        if available_gb is not None:
            details["available_gb"] = available_gb
        if path:
            details["path"] = path
        super().__init__(message, details=details, cause=cause, context=context)

    @property
    def is_transient(self) -> bool:
        """Disk space errors are transient - may succeed if space is freed."""
        return True

    def user_message(self) -> str:
        if self.details.get("required_gb") and self.details.get("available_gb"):
            return (
                f"Not enough disk space. "
                f"Need: {self.details['required_gb']:.1f}GB, "
                f"Available: {self.details['available_gb']:.1f}GB."
            )
        return self._default_user_message


class CheckpointError(StorageError):
    """Checkpoint save/load error.

    Raised when checkpoints cannot be saved, loaded, or verified.
    """

    _default_user_message = "Failed to save or load processing checkpoint."
    _default_suggested_fix = (
        "Check disk space and permissions for the checkpoint directory. "
        "You can disable checkpointing with --no-checkpoint or specify "
        "a different directory with --checkpoint-dir."
    )

    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        operation: Optional[str] = None,
        checkpoint_version: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        if operation:
            details["operation"] = operation
        if checkpoint_version:
            details["checkpoint_version"] = checkpoint_version
        super().__init__(message, details=details, cause=cause, context=context)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(FatalError):
    """Validation failure for input or output.

    Raised when quality validation fails, input is invalid,
    or output does not meet requirements.
    """

    _default_user_message = "Validation failed."
    _default_suggested_fix = (
        "Review the validation requirements and ensure your input meets them. "
        "You can bypass validation with --skip-validation if needed."
    )

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if validation_type:
            details["validation_type"] = validation_type
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        super().__init__(message, details=details, cause=cause, context=context)

    def user_message(self) -> str:
        if self.details.get("validation_type"):
            return f"Validation failed: {self.details['validation_type']}"
        return self._default_user_message


# =============================================================================
# Dependency Errors
# =============================================================================


class DependencyError(FatalError):
    """Missing or incompatible dependency.

    Raised when required external tools or libraries are not
    available or have incompatible versions.
    """

    _default_user_message = "A required dependency is missing or incompatible."
    _default_suggested_fix = (
        "Install the missing dependency. Run 'framewright check-deps' to see "
        "all required dependencies and their status."
    )

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        installed_version: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if dependency_name:
            details["dependency_name"] = dependency_name
        if required_version:
            details["required_version"] = required_version
        if installed_version:
            details["installed_version"] = installed_version
        super().__init__(message, details=details, cause=cause, context=context)

    def user_message(self) -> str:
        if self.details.get("dependency_name"):
            name = self.details["dependency_name"]
            if self.details.get("required_version") and self.details.get("installed_version"):
                return (
                    f"'{name}' version mismatch. "
                    f"Required: {self.details['required_version']}, "
                    f"Installed: {self.details['installed_version']}"
                )
            return f"Required dependency '{name}' is not installed."
        return self._default_user_message


# =============================================================================
# Data Corruption Errors
# =============================================================================


class CorruptionError(FatalError):
    """Data corruption requiring manual intervention.

    Raised when data is corrupted and cannot be automatically recovered.
    """

    _default_user_message = "Data corruption was detected."
    _default_suggested_fix = (
        "The file may be corrupted. Try re-downloading or using a backup. "
        "If this is a video file, try running: ffmpeg -v error -i file.mp4 -f null -"
    )

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        corruption_type: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional["ErrorContext"] = None,
    ) -> None:
        details = {}
        if file_path:
            details["file_path"] = file_path
        if corruption_type:
            details["corruption_type"] = corruption_type
        super().__init__(message, details=details, cause=cause, context=context)


# =============================================================================
# Legacy Compatibility - ResourceError hierarchy
# =============================================================================


class ResourceError(TransientError):
    """Resource exhaustion errors (VRAM, disk space, memory).

    This is a legacy class for backward compatibility.
    Prefer using specific error types (VRAMError, DiskSpaceError) directly.
    """

    _default_user_message = "A system resource was exhausted."
    _default_suggested_fix = (
        "Free up system resources and try again. Close other applications "
        "to free up memory, or delete files to free disk space."
    )


# =============================================================================
# Legacy Compatibility - VideoRestorerError alias
# =============================================================================


# Alias for backward compatibility with errors.py
VideoRestorerError = FramewrightError


# =============================================================================
# Error Context
# =============================================================================


@dataclass
class ErrorContext:
    """Detailed context for debugging errors.

    Captures comprehensive information about the system state
    and operation that caused the error.
    """

    stage: str
    operation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    frame_number: Optional[int] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    command: Optional[List[str]] = None
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    return_code: Optional[int] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "stage": self.stage,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "command": self.command,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "return_code": self.return_code,
            "system_state": self.system_state,
            "additional_info": self.additional_info,
        }

    def __str__(self) -> str:
        """Human-readable error context."""
        lines = [
            f"Stage: {self.stage}",
            f"Operation: {self.operation}",
            f"Timestamp: {self.timestamp}",
        ]

        if self.frame_number is not None:
            lines.append(f"Frame: {self.frame_number}")
        if self.input_file:
            lines.append(f"Input: {self.input_file}")
        if self.output_file:
            lines.append(f"Output: {self.output_file}")
        if self.command:
            lines.append(f"Command: {' '.join(self.command)}")
        if self.return_code is not None:
            lines.append(f"Return code: {self.return_code}")
        if self.stderr:
            lines.append(f"Stderr: {self.stderr[:500]}")
        if self.system_state:
            lines.append(f"System state: {self.system_state}")

        return "\n".join(lines)


def create_error_context(
    stage: str,
    operation: str,
    frame_number: Optional[int] = None,
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    command: Optional[List[str]] = None,
    stderr: Optional[str] = None,
    stdout: Optional[str] = None,
    return_code: Optional[int] = None,
    **additional_info: Any,
) -> ErrorContext:
    """Create an error context with system state.

    Automatically captures system state like VRAM usage and disk space.
    """
    system_state: Dict[str, Any] = {}

    # Capture GPU state (safely)
    try:
        from ..utils.gpu import get_gpu_memory_info

        gpu_info = get_gpu_memory_info()
        if gpu_info:
            system_state["gpu"] = gpu_info
    except Exception:
        pass

    # Capture disk state (safely)
    try:
        from ..utils.disk import get_disk_usage

        if input_file:
            disk_info = get_disk_usage(Path(input_file).parent)
            system_state["disk"] = disk_info
    except Exception:
        pass

    return ErrorContext(
        stage=stage,
        operation=operation,
        frame_number=frame_number,
        input_file=str(input_file) if input_file else None,
        output_file=str(output_file) if output_file else None,
        command=command,
        stderr=stderr,
        stdout=stdout,
        return_code=return_code,
        system_state=system_state,
        additional_info=additional_info,
    )


# =============================================================================
# Error Classification Logic
# =============================================================================


def classify_error(
    error: Exception, stderr: Optional[str] = None
) -> Type[FramewrightError]:
    """Classify an error as transient or fatal based on error message.

    Args:
        error: The exception that occurred
        stderr: Optional stderr output for additional context

    Returns:
        The appropriate error class to use
    """
    error_text = str(error).lower()
    stderr_text = (stderr or "").lower()
    combined = f"{error_text} {stderr_text}"

    # VRAM/GPU errors (transient - can retry with smaller tile size)
    vram_indicators = [
        "cuda out of memory",
        "out of memory",
        "vram",
        "gpu memory",
        "memory allocation failed",
        "insufficient memory",
        "vulkan memory",
    ]
    if any(ind in combined for ind in vram_indicators):
        return VRAMError

    # Disk space errors (transient - may free up space)
    disk_indicators = [
        "no space left on device",
        "disk quota exceeded",
        "not enough space",
        "disk full",
        "write error",
    ]
    if any(ind in combined for ind in disk_indicators):
        return DiskSpaceError

    # Network errors (transient)
    network_indicators = [
        "connection refused",
        "connection reset",
        "network unreachable",
        "timeout",
        "timed out",
        "host not found",
        "dns",
        "ssl",
        "certificate",
    ]
    if any(ind in combined for ind in network_indicators):
        return NetworkError

    # Corruption errors (fatal)
    corruption_indicators = [
        "corrupt",
        "invalid data",
        "checksum",
        "crc error",
        "truncated",
        "malformed",
    ]
    if any(ind in combined for ind in corruption_indicators):
        return CorruptionError

    # Dependency errors (fatal)
    dependency_indicators = [
        "not found",
        "command not found",
        "no such file or directory",
        "cannot find",
        "missing",
        "not installed",
    ]
    if any(ind in combined for ind in dependency_indicators):
        return DependencyError

    # Default to base error
    return FramewrightError


# =============================================================================
# Retry Logic
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on: tuple = (TransientError,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        import random

        delay = min(
            self.initial_delay * (self.exponential_base**attempt), self.max_delay
        )

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (TransientError,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on
        on_retry: Optional callback called before each retry

    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        retry_on=retry_on,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)

                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed. "
                            f"Last error: {e}"
                        )
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error: {e}")
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception

            raise RuntimeError("Unexpected state: no exception but all retries failed")

        return wrapper

    return decorator


class RetryableOperation:
    """Context manager for retryable operations with custom handling."""

    def __init__(
        self,
        operation_name: str,
        max_attempts: int = 3,
        retry_on: tuple = (TransientError,),
        on_vram_error: Optional[Callable[[], None]] = None,
    ):
        """Initialize retryable operation.

        Args:
            operation_name: Name for logging
            max_attempts: Maximum retry attempts
            retry_on: Exception types to retry
            on_vram_error: Callback for VRAM errors (e.g., reduce tile size)
        """
        self.operation_name = operation_name
        self.max_attempts = max_attempts
        self.retry_on = retry_on
        self.on_vram_error = on_vram_error
        self.attempt = 0
        self.config = RetryConfig(max_attempts=max_attempts, retry_on=retry_on)

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        last_exception: Optional[Exception] = None

        for self.attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except VRAMError as e:
                last_exception = e
                if self.on_vram_error and self.attempt < self.max_attempts - 1:
                    logger.info("VRAM error detected, attempting recovery...")
                    self.on_vram_error()
                    time.sleep(self.config.get_delay(self.attempt))
                elif self.attempt >= self.max_attempts - 1:
                    raise
            except self.retry_on as e:
                last_exception = e
                if self.attempt < self.max_attempts - 1:
                    delay = self.config.get_delay(self.attempt)
                    logger.warning(
                        f"{self.operation_name}: Attempt {self.attempt + 1} failed. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry state")


# =============================================================================
# Error Aggregation
# =============================================================================


@dataclass
class ErrorReport:
    """Aggregated error report for batch operations."""

    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_error(
        self,
        operation_id: Union[int, str],
        error: Exception,
        context: Optional[ErrorContext] = None,
    ) -> None:
        """Add an error to the report."""
        self.failed += 1
        self.errors.append(
            {
                "operation_id": operation_id,
                "error_type": type(error).__name__,
                "message": str(error),
                "context": context.to_dict() if context else None,
            }
        )

    def add_success(self) -> None:
        """Record a successful operation."""
        self.successful += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful / self.total_operations

    def has_failures(self) -> bool:
        """Check if any operations failed."""
        return self.failed > 0

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Operations: {self.successful}/{self.total_operations} successful "
            f"({self.success_rate:.1%}), {self.failed} failed"
        )


# =============================================================================
# Exception Mapping
# =============================================================================


EXCEPTION_MAP = {
    "configuration": ConfigurationError,
    "processing": ProcessingError,
    "model": ModelError,
    "hardware": HardwareError,
    "gpu": GPUError,
    "vram": VRAMError,
    "memory": VRAMError,  # Alias
    "out_of_memory": VRAMError,  # Alias
    "video": VideoError,
    "download": DownloadError,
    "checkpoint": CheckpointError,
    "interpolation": InterpolationError,
    "enhancement": EnhancementError,
    "validation": ValidationError,
    "dependency": DependencyError,
    "disk_space": DiskSpaceError,
    "corruption": CorruptionError,
    "network": NetworkError,
    "timeout": TimeoutError,
    "frame_extraction": FrameExtractionError,
    "reassembly": ReassemblyError,
    "metadata": MetadataError,
    "audio_extraction": AudioExtractionError,
    "gpu_required": GPURequiredError,
    "cpu_fallback": CPUFallbackError,
    "storage": StorageError,
    "resource": ResourceError,
    "transient": TransientError,
    "fatal": FatalError,
}


def get_exception_class(error_type: str) -> Type[FramewrightError]:
    """Get exception class by name.

    Args:
        error_type: Error type name (lowercase, underscores ok)

    Returns:
        Exception class

    Raises:
        KeyError: If error type not found
    """
    return EXCEPTION_MAP[error_type.lower().replace("-", "_")]


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Base classes
    "FramewrightError",
    "TransientError",
    "FatalError",
    # Hardware errors
    "HardwareError",
    "GPUError",
    "VRAMError",
    "OutOfMemoryError",  # Alias for VRAMError
    "GPURequiredError",
    "CPUFallbackError",
    # Configuration
    "ConfigurationError",
    # Processing errors
    "ProcessingError",
    "EnhancementError",
    "InterpolationError",
    "FrameExtractionError",
    "ReassemblyError",
    # Model errors
    "ModelError",
    # Video errors
    "VideoError",
    "MetadataError",
    "AudioExtractionError",
    # Network errors
    "NetworkError",
    "DownloadError",
    "TimeoutError",
    # Storage errors
    "StorageError",
    "DiskSpaceError",
    "CheckpointError",
    # Validation and dependency
    "ValidationError",
    "DependencyError",
    # Corruption
    "CorruptionError",
    # Legacy compatibility
    "ResourceError",
    "VideoRestorerError",
    # Context and utilities
    "ErrorContext",
    "create_error_context",
    "classify_error",
    # Retry utilities
    "RetryConfig",
    "retry_with_backoff",
    "RetryableOperation",
    # Error reporting
    "ErrorReport",
    # Exception mapping
    "EXCEPTION_MAP",
    "get_exception_class",
]
