"""Tests for the errors module."""
import pytest
import time
from unittest.mock import MagicMock, patch

from framewright.errors import (
    VideoRestorerError,
    TransientError,
    ResourceError,
    VRAMError,
    DiskSpaceError,
    NetworkError,
    FatalError,
    CorruptionError,
    DependencyError,
    ConfigurationError,
    ErrorContext,
    RetryConfig,
    retry_with_backoff,
    RetryableOperation,
    ErrorReport,
    classify_error,
)


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_transient_error_is_video_restorer_error(self):
        """Test TransientError inherits from VideoRestorerError."""
        error = TransientError("test")
        assert isinstance(error, VideoRestorerError)

    def test_resource_error_is_transient(self):
        """Test ResourceError inherits from TransientError."""
        error = ResourceError("test")
        assert isinstance(error, TransientError)

    def test_vram_error_is_resource_error(self):
        """Test VRAMError inherits from ResourceError."""
        error = VRAMError("test")
        assert isinstance(error, ResourceError)
        assert isinstance(error, TransientError)

    def test_fatal_error_is_video_restorer_error(self):
        """Test FatalError inherits from VideoRestorerError."""
        error = FatalError("test")
        assert isinstance(error, VideoRestorerError)

    def test_corruption_error_is_fatal(self):
        """Test CorruptionError inherits from FatalError."""
        error = CorruptionError("test")
        assert isinstance(error, FatalError)


class TestErrorContext:
    """Tests for ErrorContext class."""

    def test_create_context(self):
        """Test creating an error context."""
        context = ErrorContext(
            stage="enhance",
            operation="enhance_frame",
            frame_number=42,
            input_file="/path/to/input.png",
        )

        assert context.stage == "enhance"
        assert context.frame_number == 42
        assert context.timestamp is not None

    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = ErrorContext(
            stage="download",
            operation="yt-dlp",
            stderr="Connection refused",
        )

        data = context.to_dict()

        assert data["stage"] == "download"
        assert data["stderr"] == "Connection refused"

    def test_context_str(self):
        """Test string representation of context."""
        context = ErrorContext(
            stage="enhance",
            operation="realesrgan",
            frame_number=100,
            return_code=1,
        )

        str_repr = str(context)

        assert "enhance" in str_repr
        assert "100" in str_repr
        assert "Return code: 1" in str_repr


class TestClassifyError:
    """Tests for error classification function."""

    def test_classify_vram_error(self):
        """Test classification of VRAM errors."""
        error = Exception("CUDA out of memory")
        result = classify_error(error)
        assert result == VRAMError

    def test_classify_disk_error(self):
        """Test classification of disk space errors."""
        error = Exception("No space left on device")
        result = classify_error(error)
        assert result == DiskSpaceError

    def test_classify_network_error(self):
        """Test classification of network errors."""
        error = Exception("Connection refused")
        result = classify_error(error)
        assert result == NetworkError

    def test_classify_corruption_error(self):
        """Test classification of corruption errors."""
        error = Exception("Checksum mismatch in file")
        result = classify_error(error)
        assert result == CorruptionError

    def test_classify_dependency_error(self):
        """Test classification of dependency errors."""
        error = Exception("ffmpeg not found")
        result = classify_error(error)
        assert result == DependencyError

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = Exception("Something went wrong")
        result = classify_error(error)
        assert result == VideoRestorerError

    def test_classify_from_stderr(self):
        """Test classification using stderr."""
        error = Exception("Error occurred")
        result = classify_error(error, stderr="GPU memory allocation failed")
        assert result == VRAMError


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.exponential_base == 2.0

    def test_get_delay(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=0.0,  # No jitter for deterministic test
        )

        # First attempt: 1 * 2^0 = 1
        assert config.get_delay(0) == 1.0

        # Second attempt: 1 * 2^1 = 2
        assert config.get_delay(1) == 2.0

        # Third attempt: 1 * 2^2 = 4
        assert config.get_delay(2) == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=0.0,
        )

        # Large attempt number should be capped
        delay = config.get_delay(10)
        assert delay <= 10.0


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_first_try(self):
        """Test function succeeds on first try."""
        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Test function retries on transient error."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Temporary failure")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_fatal_error(self):
        """Test function doesn't retry on fatal error."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, retry_on=(TransientError,))
        def fatal_func():
            nonlocal call_count
            call_count += 1
            raise FatalError("Fatal failure")

        with pytest.raises(FatalError):
            fatal_func()

        assert call_count == 1

    def test_max_attempts_exceeded(self):
        """Test function raises after max attempts."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TransientError("Always fails")

        with pytest.raises(TransientError):
            always_fails()

        assert call_count == 3


class TestRetryableOperation:
    """Tests for RetryableOperation class."""

    def test_successful_operation(self):
        """Test successful operation execution."""
        op = RetryableOperation(
            operation_name="test",
            max_attempts=3,
        )

        result = op.execute(lambda: "success")

        assert result == "success"

    def test_retry_operation(self):
        """Test operation with retries."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Flaky")
            return "success"

        op = RetryableOperation(
            operation_name="test",
            max_attempts=3,
            retry_on=(TransientError,),
        )

        # Mock sleep to speed up test
        with patch('framewright.errors.time.sleep'):
            result = op.execute(flaky)

        assert result == "success"
        assert call_count == 2

    def test_vram_error_callback(self):
        """Test VRAM error triggers callback."""
        callback_called = False

        def on_vram():
            nonlocal callback_called
            callback_called = True

        call_count = 0

        def vram_hungry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise VRAMError("Out of memory")
            return "success"

        op = RetryableOperation(
            operation_name="test",
            max_attempts=3,
            on_vram_error=on_vram,
        )

        with patch('framewright.errors.time.sleep'):
            result = op.execute(vram_hungry)

        assert result == "success"
        assert callback_called is True


class TestErrorReport:
    """Tests for ErrorReport class."""

    def test_empty_report(self):
        """Test empty error report."""
        report = ErrorReport()

        assert report.total_operations == 0
        assert report.successful == 0
        assert report.failed == 0
        assert not report.has_failures()

    def test_add_success(self):
        """Test adding successful operations."""
        report = ErrorReport(total_operations=10)

        for _ in range(5):
            report.add_success()

        assert report.successful == 5
        assert report.success_rate == 0.5

    def test_add_error(self):
        """Test adding error operations."""
        report = ErrorReport(total_operations=10)

        report.add_error(
            operation_id=1,
            error=VRAMError("Out of memory"),
        )

        assert report.failed == 1
        assert report.has_failures()
        assert len(report.errors) == 1
        assert report.errors[0]["error_type"] == "VRAMError"

    def test_summary(self):
        """Test report summary."""
        report = ErrorReport(total_operations=10)

        for _ in range(8):
            report.add_success()
        for i in range(2):
            report.add_error(i, Exception("Error"))

        summary = report.summary()

        assert "8/10" in summary
        assert "80.0%" in summary
        assert "2 failed" in summary
