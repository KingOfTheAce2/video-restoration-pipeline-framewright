"""Standardized exception hierarchy for FrameWright.

This module provides a clean, standardized exception hierarchy for all
FrameWright operations. Each exception is designed to be:
- Self-descriptive with clear error messages
- Categorized for appropriate error handling
- Chainable for preserving original exception context

Exception Hierarchy:
    FramewrightError (base)
    +-- ConfigurationError
    +-- ProcessingError
    +-- ModelError
    +-- HardwareError
    |   +-- GPUError
    |   +-- OutOfMemoryError
    +-- VideoError
    +-- DownloadError
    +-- CheckpointError
"""

from typing import Any, Dict, Optional


class FramewrightError(Exception):
    """Base exception for all FrameWright errors.

    All FrameWright-specific exceptions inherit from this class,
    allowing for broad exception handling when needed.

    Attributes:
        message: Human-readable error description
        details: Optional dictionary with additional context
        cause: Original exception that caused this error (if any)
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize FramewrightError.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

        # Chain the cause exception for proper traceback
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return string representation with details if available."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization.

        Returns:
            Dictionary with error type, message, and details
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(FramewrightError):
    """Invalid configuration.

    Raised when configuration values are invalid, missing required
    fields, or have incompatible combinations.

    Examples:
        - Invalid scale factor (not 2 or 4)
        - Invalid CRF value (outside 0-51 range)
        - Incompatible model for scale factor
        - Missing required configuration field
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        valid_values: Optional[list] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Description of configuration error
            config_key: Name of the invalid configuration key
            config_value: The invalid value provided
            valid_values: List of valid values (if applicable)
            cause: Original exception
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value
        if valid_values:
            details["valid_values"] = valid_values
        super().__init__(message, details=details, cause=cause)


class ProcessingError(FramewrightError):
    """Error during frame processing.

    Raised when frame enhancement, interpolation, or other
    processing operations fail.

    Examples:
        - Frame enhancement failed
        - Interpolation computation error
        - Color correction failure
        - Face restoration error
    """

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        frame_number: Optional[int] = None,
        frame_path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize ProcessingError.

        Args:
            message: Description of processing error
            stage: Processing stage where error occurred
            frame_number: Frame number if applicable
            frame_path: Path to the frame file
            cause: Original exception
        """
        details = {}
        if stage:
            details["stage"] = stage
        if frame_number is not None:
            details["frame_number"] = frame_number
        if frame_path:
            details["frame_path"] = frame_path
        super().__init__(message, details=details, cause=cause)


class ModelError(FramewrightError):
    """Model loading or inference error.

    Raised when AI models fail to load, have incorrect format,
    or encounter errors during inference.

    Examples:
        - Model file not found
        - Corrupted model weights
        - Model version mismatch
        - Inference failure
        - Unsupported model format
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize ModelError.

        Args:
            message: Description of model error
            model_name: Name of the model
            model_path: Path to model file
            model_type: Type of model (realesrgan, rife, etc.)
            cause: Original exception
        """
        details = {}
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        if model_type:
            details["model_type"] = model_type
        super().__init__(message, details=details, cause=cause)


class HardwareError(FramewrightError):
    """Hardware compatibility error.

    Base class for hardware-related errors including GPU detection
    failures, driver issues, and hardware capability problems.

    Examples:
        - No compatible GPU found
        - Driver version mismatch
        - Unsupported hardware feature
        - Hardware initialization failure
    """

    def __init__(
        self,
        message: str,
        hardware_type: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize HardwareError.

        Args:
            message: Description of hardware error
            hardware_type: Type of hardware (gpu, cpu, etc.)
            device_info: Dictionary with device information
            cause: Original exception
        """
        details = {}
        if hardware_type:
            details["hardware_type"] = hardware_type
        if device_info:
            details["device_info"] = device_info
        super().__init__(message, details=details, cause=cause)


class GPUError(HardwareError):
    """GPU-specific error.

    Raised for GPU-related issues that are not memory-related,
    such as driver issues, compute capability problems, or
    CUDA/Vulkan initialization failures.

    Examples:
        - CUDA initialization failed
        - Vulkan not available
        - GPU driver too old
        - Compute capability too low
    """

    def __init__(
        self,
        message: str,
        gpu_id: Optional[int] = None,
        gpu_name: Optional[str] = None,
        driver_version: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize GPUError.

        Args:
            message: Description of GPU error
            gpu_id: GPU device ID
            gpu_name: GPU device name
            driver_version: GPU driver version
            cause: Original exception
        """
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
        )


class OutOfMemoryError(GPUError):
    """GPU out of memory error.

    Raised when GPU VRAM is exhausted during processing.
    This error is often recoverable by reducing tile size
    or processing fewer frames in parallel.

    Examples:
        - VRAM exhausted during enhancement
        - Model too large for available memory
        - Batch size too large
    """

    def __init__(
        self,
        message: str,
        required_mb: Optional[int] = None,
        available_mb: Optional[int] = None,
        gpu_id: Optional[int] = None,
        gpu_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize OutOfMemoryError.

        Args:
            message: Description of memory error
            required_mb: Memory required in MB
            available_mb: Memory available in MB
            gpu_id: GPU device ID
            gpu_name: GPU device name
            cause: Original exception
        """
        super().__init__(
            message,
            gpu_id=gpu_id,
            gpu_name=gpu_name,
            cause=cause,
        )
        if required_mb is not None:
            self.details["required_mb"] = required_mb
        if available_mb is not None:
            self.details["available_mb"] = available_mb

    @property
    def recoverable(self) -> bool:
        """Check if error is potentially recoverable by reducing tile size."""
        return True


class VideoError(FramewrightError):
    """Video file reading/writing error.

    Raised when video files cannot be read, written, or parsed.
    Includes issues with containers, codecs, and corrupted files.

    Examples:
        - Video file not found
        - Unsupported codec
        - Corrupted video file
        - Write permission denied
        - Invalid video format
    """

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        operation: Optional[str] = None,
        codec: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize VideoError.

        Args:
            message: Description of video error
            video_path: Path to video file
            operation: Operation that failed (read, write, parse)
            codec: Codec involved in the error
            cause: Original exception
        """
        details = {}
        if video_path:
            details["video_path"] = video_path
        if operation:
            details["operation"] = operation
        if codec:
            details["codec"] = codec
        super().__init__(message, details=details, cause=cause)


class DownloadError(FramewrightError):
    """Model or video download error.

    Raised when downloading files fails due to network issues,
    authentication problems, or server errors.

    Examples:
        - Network connection failed
        - URL not found (404)
        - Download timeout
        - Incomplete download
        - Checksum verification failed
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        downloaded_bytes: Optional[int] = None,
        expected_bytes: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize DownloadError.

        Args:
            message: Description of download error
            url: URL that failed to download
            status_code: HTTP status code if applicable
            downloaded_bytes: Bytes downloaded before failure
            expected_bytes: Expected total bytes
            cause: Original exception
        """
        details = {}
        if url:
            details["url"] = url
        if status_code is not None:
            details["status_code"] = status_code
        if downloaded_bytes is not None:
            details["downloaded_bytes"] = downloaded_bytes
        if expected_bytes is not None:
            details["expected_bytes"] = expected_bytes
        super().__init__(message, details=details, cause=cause)


class CheckpointError(FramewrightError):
    """Checkpoint save/load error.

    Raised when checkpoints cannot be saved, loaded, or verified.
    Includes issues with corrupted checkpoints and version mismatches.

    Examples:
        - Checkpoint file corrupted
        - Checkpoint version mismatch
        - Cannot write checkpoint file
        - Checkpoint directory not writable
    """

    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        operation: Optional[str] = None,
        checkpoint_version: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize CheckpointError.

        Args:
            message: Description of checkpoint error
            checkpoint_path: Path to checkpoint file
            operation: Operation that failed (save, load, verify)
            checkpoint_version: Checkpoint version if applicable
            cause: Original exception
        """
        details = {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        if operation:
            details["operation"] = operation
        if checkpoint_version:
            details["checkpoint_version"] = checkpoint_version
        super().__init__(message, details=details, cause=cause)


# Additional specialized exceptions for completeness


class InterpolationError(ProcessingError):
    """Frame interpolation error.

    Raised when RIFE or other interpolation operations fail.
    """

    def __init__(
        self,
        message: str,
        source_fps: Optional[float] = None,
        target_fps: Optional[float] = None,
        model: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize InterpolationError.

        Args:
            message: Description of interpolation error
            source_fps: Source frame rate
            target_fps: Target frame rate
            model: Interpolation model name
            cause: Original exception
        """
        super().__init__(message, stage="interpolation", cause=cause)
        if source_fps is not None:
            self.details["source_fps"] = source_fps
        if target_fps is not None:
            self.details["target_fps"] = target_fps
        if model:
            self.details["model"] = model


class EnhancementError(ProcessingError):
    """Frame enhancement error.

    Raised when Real-ESRGAN or other enhancement operations fail.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        scale_factor: Optional[int] = None,
        tile_size: Optional[int] = None,
        frame_number: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize EnhancementError.

        Args:
            message: Description of enhancement error
            model_name: Enhancement model name
            scale_factor: Upscaling factor
            tile_size: Tile size used
            frame_number: Frame number that failed
            cause: Original exception
        """
        super().__init__(
            message,
            stage="enhancement",
            frame_number=frame_number,
            cause=cause,
        )
        if model_name:
            self.details["model_name"] = model_name
        if scale_factor is not None:
            self.details["scale_factor"] = scale_factor
        if tile_size is not None:
            self.details["tile_size"] = tile_size


class ValidationError(FramewrightError):
    """Validation failure for input or output.

    Raised when quality validation fails, input is invalid,
    or output does not meet requirements.
    """

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Description of validation error
            validation_type: Type of validation (ssim, psnr, format, etc.)
            expected: Expected value
            actual: Actual value
            cause: Original exception
        """
        details = {}
        if validation_type:
            details["validation_type"] = validation_type
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        super().__init__(message, details=details, cause=cause)


class DependencyError(FramewrightError):
    """Missing or incompatible dependency.

    Raised when required external tools or libraries are not
    available or have incompatible versions.
    """

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        installed_version: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize DependencyError.

        Args:
            message: Description of dependency error
            dependency_name: Name of the dependency
            required_version: Required version
            installed_version: Currently installed version
            cause: Original exception
        """
        details = {}
        if dependency_name:
            details["dependency_name"] = dependency_name
        if required_version:
            details["required_version"] = required_version
        if installed_version:
            details["installed_version"] = installed_version
        super().__init__(message, details=details, cause=cause)


class DiskSpaceError(FramewrightError):
    """Insufficient disk space error.

    Raised when there is not enough disk space for operations.
    """

    def __init__(
        self,
        message: str,
        required_gb: Optional[float] = None,
        available_gb: Optional[float] = None,
        path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize DiskSpaceError.

        Args:
            message: Description of disk space error
            required_gb: Required space in GB
            available_gb: Available space in GB
            path: Path where space is needed
            cause: Original exception
        """
        details = {}
        if required_gb is not None:
            details["required_gb"] = required_gb
        if available_gb is not None:
            details["available_gb"] = available_gb
        if path:
            details["path"] = path
        super().__init__(message, details=details, cause=cause)


# Exception mapping for easy lookup
EXCEPTION_MAP = {
    "configuration": ConfigurationError,
    "processing": ProcessingError,
    "model": ModelError,
    "hardware": HardwareError,
    "gpu": GPUError,
    "memory": OutOfMemoryError,
    "video": VideoError,
    "download": DownloadError,
    "checkpoint": CheckpointError,
    "interpolation": InterpolationError,
    "enhancement": EnhancementError,
    "validation": ValidationError,
    "dependency": DependencyError,
    "disk_space": DiskSpaceError,
}


def get_exception_class(error_type: str) -> type:
    """Get exception class by name.

    Args:
        error_type: Error type name (lowercase)

    Returns:
        Exception class

    Raises:
        KeyError: If error type not found
    """
    return EXCEPTION_MAP[error_type.lower()]
