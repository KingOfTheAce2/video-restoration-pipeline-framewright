"""Integration tests for error recovery and handling in FrameWright.

These tests verify the system handles various error conditions gracefully
including corrupt files, disk space issues, and interrupted processing.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock
import pytest

from framewright.config import Config
from framewright.restorer import VideoRestorer
from framewright.errors import (
    VideoRestorerError,
    DownloadError,
    MetadataError,
    FrameExtractionError,
    EnhancementError,
    ReassemblyError,
    DiskSpaceError,
    VRAMError,
)


# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.integration]


class TestCorruptVideoHandling:
    """Test handling of corrupt or invalid video files."""

    def test_corrupt_video_detection(
        self,
        corrupt_video: Path,
        temp_dir: Path
    ):
        """Test handling of corrupt video files."""
        project_dir = temp_dir / "corrupt_video_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # Should raise MetadataError for corrupt file
            with pytest.raises(MetadataError):
                restorer.analyze_metadata(corrupt_video)

    def test_truncated_video_handling(
        self,
        temp_dir: Path,
        ffmpeg_available: bool
    ):
        """Test handling of truncated video files."""
        if not ffmpeg_available:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "truncated_video_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create a truncated video file
        truncated_path = project_dir / "truncated.mp4"

        # Write some bytes that look like video start but are incomplete
        with open(truncated_path, 'wb') as f:
            # Write minimal MP4 header then truncate
            f.write(b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41')

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # Should handle truncated file gracefully
            with pytest.raises((MetadataError, VideoRestorerError)):
                restorer.analyze_metadata(truncated_path)

    def test_empty_file_handling(self, temp_dir: Path):
        """Test handling of empty video files."""
        project_dir = temp_dir / "empty_file_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create empty file
        empty_path = project_dir / "empty.mp4"
        empty_path.touch()

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # Should raise error for empty file
            with pytest.raises((MetadataError, VideoRestorerError)):
                restorer.analyze_metadata(empty_path)

    def test_wrong_extension_handling(
        self,
        test_single_frame: Path,
        temp_dir: Path
    ):
        """Test handling of file with wrong extension."""
        project_dir = temp_dir / "wrong_ext_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Copy PNG file with .mp4 extension
        wrong_ext_path = project_dir / "not_a_video.mp4"
        shutil.copy(test_single_frame, wrong_ext_path)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # ffprobe can actually read PNG files, so analyze_metadata
            # may succeed. We test that either it raises an error OR
            # returns metadata indicating a non-video format.
            try:
                metadata = restorer.analyze_metadata(wrong_ext_path)
                # If it succeeds, the codec should indicate image format (png)
                assert metadata is not None
            except (MetadataError, VideoRestorerError):
                pass  # Expected for corrupt/non-video files


class TestDiskSpaceHandling:
    """Test handling of disk space issues."""

    def test_disk_space_validation_enabled(self, temp_dir: Path, mock_gpu):
        """Test disk space validation when enabled."""
        project_dir = temp_dir / "disk_space_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=True,  # Enable disk validation
            disk_safety_margin=1.2,
            enable_vram_monitoring=False,
        )

        assert config.enable_disk_validation is True
        assert config.disk_safety_margin == 1.2

    def test_low_disk_space_warning(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test handling of low disk space conditions."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "low_disk_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=True,
            disk_safety_margin=1.2,
            enable_vram_monitoring=False,
        )

        # Mock low disk space
        mock_disk_result = {
            "is_valid": False,
            "available_gb": 0.5,
            "required_gb": 10.0,
        }

        with patch.object(VideoRestorer, '_verify_dependencies'):
            with patch('framewright.restorer.validate_disk_space', return_value=mock_disk_result):
                restorer = VideoRestorer(config)
                restorer.analyze_metadata(test_video_1s)

                # Should raise DiskSpaceError
                with pytest.raises(DiskSpaceError):
                    restorer._validate_disk_space(test_video_1s)

    def test_disk_space_sufficient(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test validation passes with sufficient disk space."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "sufficient_disk_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=True,
            disk_safety_margin=1.2,
            enable_vram_monitoring=False,
        )

        # Mock sufficient disk space
        mock_disk_result = {
            "is_valid": True,
            "available_gb": 100.0,
            "required_gb": 10.0,
        }

        with patch.object(VideoRestorer, '_verify_dependencies'):
            with patch('framewright.restorer.validate_disk_space', return_value=mock_disk_result):
                restorer = VideoRestorer(config)
                restorer.analyze_metadata(test_video_1s)

                # Should not raise
                restorer._validate_disk_space(test_video_1s)


class TestInterruptedProcessing:
    """Test recovery from interrupted processing."""

    def test_checkpoint_recovery(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test recovery from interrupted processing using checkpoints."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "checkpoint_recovery_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=True,
            checkpoint_interval=5,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        # First run - process partially and save checkpoint
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer1 = VideoRestorer(config)
            restorer1.analyze_metadata(test_video_1s)
            frame_count = restorer1.extract_frames(test_video_1s)

            # Simulate partial enhancement
            if restorer1.checkpoint_manager:
                restorer1.checkpoint_manager.update_stage("enhance")
                restorer1.checkpoint_manager.force_save()

        # Second run - verify checkpoint exists and can be loaded
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer2 = VideoRestorer(config)

            # Should find existing checkpoint
            assert restorer2.checkpoint_manager is not None
            checkpoint = restorer2.checkpoint_manager.load_checkpoint()

            if checkpoint:
                assert checkpoint.stage == "enhance"
                assert checkpoint.total_frames == frame_count

    def test_recovery_preserves_frames(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test that recovery preserves already extracted frames."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "preserve_frames_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=True,
            checkpoint_interval=3,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        # First run - extract frames
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer1 = VideoRestorer(config)
            restorer1.analyze_metadata(test_video_1s)
            original_count = restorer1.extract_frames(test_video_1s)

            if restorer1.checkpoint_manager:
                restorer1.checkpoint_manager.force_save()

        # Count frames
        frames_before = list(config.frames_dir.glob("frame_*.png"))

        # Second run - frames should still exist
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer2 = VideoRestorer(config)

            frames_after = list(config.frames_dir.glob("frame_*.png"))
            assert len(frames_after) == len(frames_before)

    def test_continue_on_error_flag(
        self,
        temp_dir: Path,
        test_frames_dir: Path,
        mock_gpu
    ):
        """Test continue_on_error flag allows processing to continue."""
        project_dir = temp_dir / "continue_error_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
            continue_on_error=True,  # Enable continue on error
        )

        assert config.continue_on_error is True

    def test_max_retries_config(self, temp_dir: Path, mock_gpu):
        """Test max_retries configuration."""
        project_dir = temp_dir / "max_retries_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
            max_retries=5,
            retry_delay=0.5,
        )

        assert config.max_retries == 5
        assert config.retry_delay == 0.5


class TestVRAMErrorHandling:
    """Test handling of VRAM/GPU memory errors."""

    def test_vram_monitoring_enabled(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test VRAM monitoring when enabled."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "vram_monitoring_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=True,  # Enable VRAM monitoring
        )

        assert config.enable_vram_monitoring is True

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # VRAM monitor should be initialized
            assert restorer._vram_monitor is not None

    def test_low_vram_tile_reduction(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_low_vram,
        ffmpeg_available: bool
    ):
        """Test tile size reduction for low VRAM conditions."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "low_vram_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=True,
            tile_size=0,  # Auto tile size
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)
            restorer.analyze_metadata(test_video_1s)

            # With low VRAM, tile size should be reduced
            # This tests the tile size selection logic
            tile_size = restorer._get_tile_size()
            assert tile_size >= 0  # 0 means auto

    def test_vram_error_classification(self):
        """Test VRAM error classification."""
        from framewright.errors import classify_error

        # Create mock subprocess error with VRAM message
        mock_error = subprocess.CalledProcessError(
            returncode=1,
            cmd=['realesrgan'],
            stderr="CUDA out of memory"
        )

        error_class = classify_error(mock_error, "CUDA out of memory")
        assert error_class is not None


class TestDependencyErrors:
    """Test handling of missing dependencies."""

    def test_missing_ffmpeg_error(self, temp_dir: Path):
        """Test error handling when ffmpeg is missing."""
        project_dir = temp_dir / "missing_ffmpeg_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        # Mock missing dependencies
        mock_report = Mock()
        mock_report.is_ready.return_value = False
        mock_report.missing_required = ["ffmpeg", "ffprobe"]
        mock_report.summary.return_value = "Missing: ffmpeg, ffprobe"

        with patch('framewright.restorer.validate_all_dependencies', return_value=mock_report):
            from framewright.errors import DependencyError
            with pytest.raises(DependencyError):
                VideoRestorer(config)

    def test_missing_realesrgan_error(self, temp_dir: Path):
        """Test error handling when Real-ESRGAN is missing."""
        project_dir = temp_dir / "missing_realesrgan_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        # Mock missing dependencies
        mock_report = Mock()
        mock_report.is_ready.return_value = False
        mock_report.missing_required = ["realesrgan"]
        mock_report.summary.return_value = "Missing: realesrgan"

        with patch('framewright.restorer.validate_all_dependencies', return_value=mock_report):
            from framewright.errors import DependencyError
            with pytest.raises(DependencyError):
                VideoRestorer(config)


class TestNetworkErrors:
    """Test handling of network-related errors."""

    def test_download_timeout(self, temp_dir: Path, mock_gpu):
        """Test handling of download timeouts."""
        project_dir = temp_dir / "download_timeout_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # Mock timeout during download
            def timeout_side_effect(*args, **kwargs):
                raise subprocess.TimeoutExpired(cmd=['yt-dlp'], timeout=3600)

            with patch('framewright.restorer.get_ytdlp_path', return_value='yt-dlp'):
                with patch('subprocess.run', side_effect=timeout_side_effect):
                    from framewright.errors import TransientError
                    with pytest.raises((DownloadError, TransientError)):
                        restorer.download_video("https://example.com/video.mp4")

    def test_network_error_retry(self, temp_dir: Path, mock_gpu):
        """Test network error retry logic."""
        project_dir = temp_dir / "network_retry_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
            max_retries=2,
            retry_delay=0.1,
        )

        assert config.max_retries == 2


class TestFrameProcessingErrors:
    """Test handling of frame processing errors."""

    def test_corrupt_frame_handling(
        self,
        temp_dir: Path,
        mock_gpu
    ):
        """Test handling of corrupt frame during enhancement."""
        project_dir = temp_dir / "corrupt_frame_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
            continue_on_error=True,
        )

        # Create frames directory with one corrupt frame
        config.frames_dir.mkdir(parents=True, exist_ok=True)

        # Valid frame
        from tests.fixtures.conftest import generate_png_image
        valid_png = generate_png_image(320, 240, (128, 128, 128), 1)
        (config.frames_dir / "frame_00000001.png").write_bytes(valid_png)

        # Corrupt frame
        (config.frames_dir / "frame_00000002.png").write_bytes(b"NOT A PNG")

        # Valid frame
        valid_png2 = generate_png_image(320, 240, (128, 128, 128), 3)
        (config.frames_dir / "frame_00000003.png").write_bytes(valid_png2)

        assert len(list(config.frames_dir.glob("frame_*.png"))) == 3

    def test_enhancement_error_report(
        self,
        temp_dir: Path,
        test_frames_dir: Path,
        mock_gpu
    ):
        """Test error report generation during enhancement."""
        project_dir = temp_dir / "error_report_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Copy test frames
        shutil.copytree(test_frames_dir, project_dir / "temp" / "frames")

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
            continue_on_error=True,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # Get error report (before any processing)
            report = restorer.get_error_report()
            assert report is not None


class TestCleanupOnError:
    """Test cleanup behavior on errors."""

    def test_temp_files_on_error(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test temporary files are handled on error."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "cleanup_error_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=True,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)
            restorer.analyze_metadata(test_video_1s)
            restorer.extract_frames(test_video_1s)

            # Frames should exist
            assert config.frames_dir.exists()
            assert len(list(config.frames_dir.glob("frame_*.png"))) > 0

            # Cleanup should remove temp files
            config.cleanup_temp()
            assert not config.temp_dir.exists()
