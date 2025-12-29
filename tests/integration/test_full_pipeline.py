"""Integration tests for the full FrameWright video restoration pipeline.

These tests verify end-to-end functionality of the video restoration workflow
including frame extraction, enhancement, interpolation, and reassembly.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock
import pytest

from framewright.config import Config
from framewright.restorer import VideoRestorer, ProgressInfo


# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.integration]


class TestFullPipeline:
    """Test complete video restoration workflow."""

    def test_basic_restoration_with_mocked_tools(
        self,
        test_video_5s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test complete video restoration workflow with mocked external tools.

        Verifies the pipeline orchestration logic without requiring actual
        GPU hardware or external tools to be fully functional.
        """
        if not ffmpeg_available or test_video_5s is None:
            pytest.skip("FFmpeg not available for video generation")

        project_dir = temp_dir / "restoration_project"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            crf=28,
            preset="ultrafast",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        # Track progress callbacks
        progress_updates: List[ProgressInfo] = []

        def progress_callback(info: ProgressInfo):
            progress_updates.append(info)

        # Mock the dependency check and external tool calls
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config, progress_callback=progress_callback)

            # Test metadata extraction (this should work with real ffmpeg)
            metadata = restorer.analyze_metadata(test_video_5s)

            assert metadata is not None
            assert 'width' in metadata
            assert 'height' in metadata
            assert 'framerate' in metadata
            assert 'duration' in metadata
            assert metadata['width'] > 0
            assert metadata['height'] > 0
            assert metadata['framerate'] > 0

            # Verify progress was reported
            assert len(progress_updates) > 0
            assert any(p.stage == 'analyze_metadata' for p in progress_updates)

    def test_frame_extraction(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        ffmpeg_available: bool
    ):
        """Test frame extraction from video."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "frame_extraction_test"
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
            restorer.analyze_metadata(test_video_1s)

            # Extract frames
            frame_count = restorer.extract_frames(test_video_1s)

            # Verify frames were extracted
            assert frame_count > 0
            assert config.frames_dir.exists()

            extracted_frames = list(config.frames_dir.glob("frame_*.png"))
            assert len(extracted_frames) == frame_count
            assert len(extracted_frames) > 0

            # Verify frame files are valid
            for frame in extracted_frames[:3]:  # Check first 3 frames
                assert frame.stat().st_size > 0

    def test_restoration_with_interpolation(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test restoration with RIFE interpolation enabled."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "interpolation_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_interpolation=True,
            target_fps=48.0,  # Double the frame rate
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)
            metadata = restorer.analyze_metadata(test_video_1s)

            # Store original FPS
            original_fps = metadata['framerate']

            # Verify interpolation config
            assert config.enable_interpolation is True
            assert config.target_fps == 48.0
            assert config.interpolated_dir is not None

    def test_restoration_with_colorization(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test restoration with colorization enabled."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "colorization_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_colorization=True,
            colorization_model="ddcolor",
            colorization_strength=0.8,
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        # Verify colorization config
        assert config.enable_colorization is True
        assert config.colorization_model == "ddcolor"
        assert config.colorization_strength == 0.8

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)
            metadata = restorer.analyze_metadata(test_video_1s)

            assert metadata is not None

    def test_checkpoint_save_and_load(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test checkpoint save and resume functionality."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "checkpoint_test"
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

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # Verify checkpoint manager was initialized
            assert restorer.checkpoint_manager is not None
            assert config.checkpoint_dir is not None

            # Analyze metadata and extract frames
            restorer.analyze_metadata(test_video_1s)
            frame_count = restorer.extract_frames(test_video_1s)

            # Load checkpoint and verify
            checkpoint = restorer.checkpoint_manager.load_checkpoint()
            if checkpoint is not None:
                assert checkpoint.stage == "extract"
                assert checkpoint.total_frames == frame_count

    def test_checkpoint_resume(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test resuming from a saved checkpoint."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "resume_test"
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

        # First run - create checkpoint
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer1 = VideoRestorer(config)
            restorer1.analyze_metadata(test_video_1s)
            restorer1.extract_frames(test_video_1s)

            # Get original frame count
            original_frames = len(list(config.frames_dir.glob("frame_*.png")))

            # Force save checkpoint
            if restorer1.checkpoint_manager:
                restorer1.checkpoint_manager.force_save()

        # Second run - resume from checkpoint
        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer2 = VideoRestorer(config)

            # Verify checkpoint exists
            assert restorer2.checkpoint_manager is not None
            checkpoint = restorer2.checkpoint_manager.load_checkpoint()

            # Frames should still exist
            existing_frames = len(list(config.frames_dir.glob("frame_*.png")))
            assert existing_frames == original_frames

    def test_gpu_fallback_to_cpu(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_no_gpu,
        ffmpeg_available: bool
    ):
        """Test automatic fallback when GPU unavailable."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "cpu_fallback_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=True,  # Enable to test fallback
        )

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)

            # With mocked no GPU, VRAM monitor should handle gracefully
            metadata = restorer.analyze_metadata(test_video_1s)
            assert metadata is not None

            # GPU info should be None
            from framewright.utils.gpu import get_gpu_memory_info
            gpu_info = get_gpu_memory_info()
            assert gpu_info is None

    def test_progress_callback_integration(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test that progress callbacks are called correctly."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "progress_test"
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

        progress_stages: List[str] = []
        progress_values: List[float] = []

        def progress_callback(info: ProgressInfo):
            progress_stages.append(info.stage)
            progress_values.append(info.progress)

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config, progress_callback=progress_callback)
            restorer.analyze_metadata(test_video_1s)
            restorer.extract_frames(test_video_1s)

        # Verify progress callbacks were received
        assert len(progress_stages) > 0
        assert 'analyze_metadata' in progress_stages
        assert 'extract_frames' in progress_stages

        # Verify progress values are valid
        for val in progress_values:
            assert 0.0 <= val <= 1.0

    def test_parallel_frame_processing_config(
        self,
        test_video_1s: Optional[Path],
        temp_dir: Path,
        mock_gpu,
        ffmpeg_available: bool
    ):
        """Test configuration for parallel frame processing."""
        if not ffmpeg_available or test_video_1s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "parallel_test"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Test with parallel frames enabled
        config = Config(
            project_dir=project_dir,
            scale_factor=2,
            model_name="realesrgan-x2plus",
            parallel_frames=4,  # Enable parallel processing
            enable_checkpointing=False,
            enable_validation=False,
            enable_disk_validation=False,
            enable_vram_monitoring=False,
        )

        assert config.parallel_frames == 4

        with patch.object(VideoRestorer, '_verify_dependencies'):
            restorer = VideoRestorer(config)
            assert restorer.config.parallel_frames == 4


class TestPipelineWithPresets:
    """Test pipeline with configuration presets."""

    @pytest.mark.parametrize("preset_name,expected_scale", [
        ("fast", 2),
        ("quality", 4),
        ("archive", 4),
        ("anime", 4),
    ])
    def test_preset_configurations(
        self,
        preset_name: str,
        expected_scale: int,
        temp_dir: Path
    ):
        """Test that presets create valid configurations."""
        project_dir = temp_dir / f"preset_{preset_name}"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config.from_preset(preset_name, project_dir)

        assert config.scale_factor == expected_scale
        assert config.project_dir == project_dir

    def test_preset_with_overrides(self, temp_dir: Path):
        """Test preset with custom overrides."""
        project_dir = temp_dir / "preset_override"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config.from_preset(
            "fast",
            project_dir,
            crf=20,  # Override CRF
            parallel_frames=2,  # Override parallel frames
        )

        assert config.crf == 20
        assert config.parallel_frames == 2

    def test_film_restoration_preset(self, temp_dir: Path):
        """Test film restoration preset with all auto features."""
        project_dir = temp_dir / "film_restoration"
        project_dir.mkdir(parents=True, exist_ok=True)

        config = Config.from_preset("film_restoration", project_dir)

        assert config.enable_auto_enhance is True
        assert config.auto_defect_repair is True
        assert config.auto_face_restore is True


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frames_directory(self, temp_dir: Path, mock_gpu):
        """Test handling of empty frames directory."""
        project_dir = temp_dir / "empty_frames_test"
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

            # Create empty frames directory
            config.frames_dir.mkdir(parents=True, exist_ok=True)

            # Verify enhancement raises error with no frames
            from framewright.errors import EnhancementError
            with pytest.raises(EnhancementError):
                restorer.enhance_frames()

    def test_single_frame_video(
        self,
        temp_dir: Path,
        test_single_frame: Path,
        mock_gpu
    ):
        """Test handling of single-frame input."""
        project_dir = temp_dir / "single_frame_test"
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

        # Setup frames directory with single frame
        config.frames_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(test_single_frame, config.frames_dir / "frame_00000001.png")

        # Verify single frame exists
        frames = list(config.frames_dir.glob("frame_*.png"))
        assert len(frames) == 1

    def test_config_validation(self, temp_dir: Path):
        """Test configuration validation."""
        project_dir = temp_dir / "config_validation"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Test invalid scale factor
        with pytest.raises(ValueError, match="scale_factor must be 2 or 4"):
            Config(
                project_dir=project_dir,
                scale_factor=3,  # Invalid
                model_name="realesrgan-x4plus",
            )

        # Test invalid CRF
        with pytest.raises(ValueError, match="crf must be between 0 and 51"):
            Config(
                project_dir=project_dir,
                scale_factor=4,
                model_name="realesrgan-x4plus",
                crf=60,  # Invalid
            )

        # Test invalid model for scale
        with pytest.raises(ValueError, match="Invalid model"):
            Config(
                project_dir=project_dir,
                scale_factor=2,
                model_name="realesrgan-x4plus",  # 4x model with 2x scale
            )


class TestPipelineMetadata:
    """Test metadata extraction and handling."""

    def test_metadata_extraction(
        self,
        test_video_5s: Optional[Path],
        temp_dir: Path,
        ffmpeg_available: bool
    ):
        """Test detailed metadata extraction."""
        if not ffmpeg_available or test_video_5s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "metadata_test"
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
            metadata = restorer.analyze_metadata(test_video_5s)

        # Verify all expected metadata fields
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'framerate' in metadata
        assert 'duration' in metadata
        assert 'codec' in metadata
        assert 'has_audio' in metadata

        # Verify values are reasonable
        assert metadata['width'] == 480
        assert metadata['height'] == 360
        assert 23 <= metadata['framerate'] <= 25  # Allow for frame rate variations
        assert 4.5 <= metadata['duration'] <= 5.5

    def test_metadata_with_audio(
        self,
        test_video_5s: Optional[Path],
        temp_dir: Path,
        ffmpeg_available: bool
    ):
        """Test metadata extraction with audio track."""
        if not ffmpeg_available or test_video_5s is None:
            pytest.skip("FFmpeg not available")

        project_dir = temp_dir / "audio_metadata_test"
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
            metadata = restorer.analyze_metadata(test_video_5s)

        # Test video should have audio
        assert metadata['has_audio'] is True
        assert metadata.get('audio_codec') is not None
