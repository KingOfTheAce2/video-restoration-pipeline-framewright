"""Tests for temporal denoising module.

Tests cover:
- Configuration validation
- Optical flow estimation
- Flicker detection and reduction
- Temporal consistency filtering
- Multi-frame denoising
- Auto denoiser
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framewright.processors.temporal_denoise import (
    AutoTemporalDenoiser,
    DenoiseMethod,
    FlickerMode,
    FlickerReducer,
    FlowField,
    OpticalFlowEstimator,
    OpticalFlowMethod,
    TemporalConsistencyFilter,
    TemporalDenoiseConfig,
    TemporalDenoiseResult,
    TemporalDenoiser,
    auto_denoise_video,
    create_temporal_denoiser,
    denoise_video_frames,
)


# Mark tests that require OpenCV
requires_opencv = pytest.mark.skipif(
    not pytest.importorskip("cv2", reason="OpenCV not installed"),
    reason="OpenCV required"
)


class TestTemporalDenoiseConfig:
    """Test TemporalDenoiseConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TemporalDenoiseConfig()

        assert config.temporal_radius == 3
        assert config.noise_strength == 0.5
        assert config.method == DenoiseMethod.OPTICAL_FLOW_WARP
        assert config.enable_optical_flow is True
        assert config.enable_flicker_reduction is True
        assert config.flicker_mode == FlickerMode.ADAPTIVE
        assert config.preserve_edges is True
        assert config.scene_change_threshold == 0.7
        assert config.chunk_size == 50

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TemporalDenoiseConfig(
            temporal_radius=5,
            noise_strength=0.8,
            method=DenoiseMethod.NON_LOCAL_MEANS_TEMPORAL,
            enable_optical_flow=False,
            flicker_mode=FlickerMode.AGGRESSIVE,
        )

        assert config.temporal_radius == 5
        assert config.noise_strength == 0.8
        assert config.method == DenoiseMethod.NON_LOCAL_MEANS_TEMPORAL
        assert config.enable_optical_flow is False
        assert config.flicker_mode == FlickerMode.AGGRESSIVE

    def test_invalid_temporal_radius(self):
        """Test that invalid temporal_radius raises ValueError."""
        with pytest.raises(ValueError, match="temporal_radius must be >= 1"):
            TemporalDenoiseConfig(temporal_radius=0)

    def test_invalid_noise_strength(self):
        """Test that invalid noise_strength raises ValueError."""
        with pytest.raises(ValueError, match="noise_strength must be 0-1"):
            TemporalDenoiseConfig(noise_strength=1.5)

        with pytest.raises(ValueError, match="noise_strength must be 0-1"):
            TemporalDenoiseConfig(noise_strength=-0.1)

    def test_invalid_weight_decay(self):
        """Test that invalid temporal_weight_decay raises ValueError."""
        with pytest.raises(ValueError, match="temporal_weight_decay must be 0-1"):
            TemporalDenoiseConfig(temporal_weight_decay=2.0)

    def test_invalid_scene_threshold(self):
        """Test that invalid scene_change_threshold raises ValueError."""
        with pytest.raises(ValueError, match="scene_change_threshold must be 0-1"):
            TemporalDenoiseConfig(scene_change_threshold=1.5)

    def test_invalid_chunk_size(self):
        """Test that invalid chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be >= 10"):
            TemporalDenoiseConfig(chunk_size=5)


class TestFlowField:
    """Test FlowField data class."""

    def test_flow_field_creation(self):
        """Test creating a FlowField."""
        h, w = 100, 150
        flow_x = np.random.randn(h, w).astype(np.float32)
        flow_y = np.random.randn(h, w).astype(np.float32)
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        confidence = np.ones((h, w), dtype=np.float32)

        flow = FlowField(
            flow_x=flow_x,
            flow_y=flow_y,
            magnitude=magnitude,
            confidence=confidence,
            frame_idx_from=0,
            frame_idx_to=1
        )

        assert flow.flow_x.shape == (h, w)
        assert flow.flow_y.shape == (h, w)
        assert flow.magnitude.shape == (h, w)
        assert flow.confidence.shape == (h, w)
        assert flow.frame_idx_from == 0
        assert flow.frame_idx_to == 1


class TestOpticalFlowEstimator:
    """Test optical flow estimation."""

    @requires_opencv
    def test_farneback_estimation(self):
        """Test Farneback optical flow estimation."""
        import cv2

        estimator = OpticalFlowEstimator(method=OpticalFlowMethod.FARNEBACK)

        # Create test frames with known motion
        h, w = 100, 100
        frame1 = np.zeros((h, w, 3), dtype=np.uint8)
        frame1[40:60, 40:60] = 255  # White square

        frame2 = np.zeros((h, w, 3), dtype=np.uint8)
        frame2[40:60, 45:65] = 255  # Shifted right by 5 pixels

        flow = estimator.estimate(frame1, frame2)

        assert isinstance(flow, FlowField)
        assert flow.flow_x.shape == (h, w)
        assert flow.flow_y.shape == (h, w)
        assert flow.magnitude.shape == (h, w)
        assert flow.confidence.shape == (h, w)

    @requires_opencv
    def test_warp_frame(self):
        """Test frame warping with optical flow."""
        import cv2

        estimator = OpticalFlowEstimator(method=OpticalFlowMethod.FARNEBACK)

        h, w = 100, 100
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 255

        # Create synthetic flow field
        flow_x = np.full((h, w), 5.0, dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        magnitude = np.abs(flow_x)
        confidence = np.ones((h, w), dtype=np.float32)

        flow = FlowField(
            flow_x=flow_x,
            flow_y=flow_y,
            magnitude=magnitude,
            confidence=confidence,
            frame_idx_from=0,
            frame_idx_to=1
        )

        warped = estimator.warp_frame(frame, flow)

        assert warped.shape == frame.shape
        assert warped.dtype == frame.dtype

    def test_unsupported_method_fallback(self):
        """Test that unsupported methods fall back gracefully."""
        # RAFT requires PyTorch and model weights
        estimator = OpticalFlowEstimator(method=OpticalFlowMethod.RAFT)
        # Should initialize without error, may have limited functionality


class TestFlickerReducer:
    """Test flicker detection and reduction."""

    def test_init(self):
        """Test FlickerReducer initialization."""
        reducer = FlickerReducer(mode=FlickerMode.MEDIUM)
        assert reducer.mode == FlickerMode.MEDIUM
        assert reducer.preserve_brightness_changes is True

    def test_adaptive_mode(self):
        """Test adaptive flicker mode."""
        reducer = FlickerReducer(mode=FlickerMode.ADAPTIVE)
        assert reducer.mode == FlickerMode.ADAPTIVE

    @pytest.fixture
    def temp_frames_dir(self):
        """Create a temporary directory with test frames."""
        temp_dir = tempfile.mkdtemp()
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir()

        # Create simple test frames with varying brightness
        try:
            import cv2
            for i in range(20):
                # Add flickering by varying brightness
                brightness = 128 + int(30 * np.sin(i * 0.5))
                frame = np.full((100, 100, 3), brightness, dtype=np.uint8)
                cv2.imwrite(str(frames_dir / f"frame_{i:08d}.png"), frame)
        except ImportError:
            # Create placeholder files if OpenCV not available
            for i in range(20):
                (frames_dir / f"frame_{i:08d}.png").touch()

        yield frames_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @requires_opencv
    def test_analyze_flicker(self, temp_frames_dir):
        """Test flicker analysis on frame sequence."""
        reducer = FlickerReducer(mode=FlickerMode.ADAPTIVE)

        metrics = reducer.analyze_flicker(temp_frames_dir)

        assert "severity" in metrics
        assert "temporal_variance" in metrics
        assert "frequency" in metrics
        assert "recommended_mode" in metrics

        assert 0 <= metrics["severity"] <= 1
        assert metrics["recommended_mode"] in [m.value for m in FlickerMode]

    @requires_opencv
    def test_reduce_flicker(self, temp_frames_dir):
        """Test flicker reduction processing."""
        reducer = FlickerReducer(mode=FlickerMode.LIGHT)

        output_dir = temp_frames_dir.parent / "output"

        result = reducer.reduce_flicker(temp_frames_dir, output_dir)

        assert "frames_processed" in result or "success" in result
        assert output_dir.exists()


class TestTemporalConsistencyFilter:
    """Test temporal consistency filtering."""

    def test_init(self):
        """Test filter initialization."""
        filter = TemporalConsistencyFilter(
            strength=0.7,
            temporal_radius=3,
            use_optical_flow=True
        )

        assert filter.strength == 0.7
        assert filter.temporal_radius == 3
        assert filter.use_optical_flow is True

    def test_default_flow_estimator(self):
        """Test that default flow estimator is created."""
        filter = TemporalConsistencyFilter()
        assert filter.flow_estimator is not None

    def test_custom_flow_estimator(self):
        """Test using custom flow estimator."""
        custom_estimator = OpticalFlowEstimator(method=OpticalFlowMethod.DIS)
        filter = TemporalConsistencyFilter(flow_estimator=custom_estimator)

        assert filter.flow_estimator is custom_estimator

    @requires_opencv
    def test_apply_filter(self, tmp_path):
        """Test applying temporal consistency filter."""
        import cv2

        # Create test frames
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(10):
            frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), frame)

        filter = TemporalConsistencyFilter(strength=0.5, temporal_radius=2)
        result = filter.apply(input_dir, output_dir)

        assert result.get("success", False) or result.get("frames_processed", 0) > 0
        assert output_dir.exists()


class TestTemporalDenoiser:
    """Test main temporal denoiser class."""

    def test_init_default_config(self):
        """Test denoiser initialization with default config."""
        denoiser = TemporalDenoiser()

        assert denoiser.config is not None
        assert denoiser._flow_estimator is not None
        assert denoiser._flicker_reducer is not None
        assert denoiser._consistency_filter is not None

    def test_init_custom_config(self):
        """Test denoiser initialization with custom config."""
        config = TemporalDenoiseConfig(
            temporal_radius=5,
            noise_strength=0.8,
            enable_optical_flow=False
        )
        denoiser = TemporalDenoiser(config)

        assert denoiser.config.temporal_radius == 5
        assert denoiser.config.noise_strength == 0.8
        assert denoiser.config.enable_optical_flow is False

    @requires_opencv
    def test_analyze_empty_dir(self, tmp_path):
        """Test analysis of empty directory."""
        denoiser = TemporalDenoiser()
        analysis = denoiser.analyze(tmp_path)

        assert "error" in analysis or analysis.get("total_frames", 0) == 0

    @requires_opencv
    def test_analyze_with_frames(self, tmp_path):
        """Test analysis with actual frames."""
        import cv2

        # Create test frames
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        for i in range(30):
            # Add some noise variation between frames
            base = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
            noise = np.random.randint(0, 30, (50, 50, 3), dtype=np.uint8)
            frame = np.clip(base + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:08d}.png"), frame)

        denoiser = TemporalDenoiser()
        analysis = denoiser.analyze(frames_dir)

        assert analysis.get("total_frames", 0) > 0
        assert "noise_level" in analysis
        assert "flicker_metrics" in analysis
        assert "scene_changes" in analysis
        assert "recommended_config" in analysis

    @requires_opencv
    def test_denoise_frames(self, tmp_path):
        """Test full denoising pipeline."""
        import cv2

        # Create noisy test frames
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(20):
            frame = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
            noise = np.random.randint(0, 50, (50, 50, 3), dtype=np.uint8)
            noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), noisy_frame)

        config = TemporalDenoiseConfig(
            temporal_radius=2,
            noise_strength=0.5,
            enable_flicker_reduction=False,  # Faster for testing
            enable_optical_flow=False,  # Faster for testing
        )
        denoiser = TemporalDenoiser(config)

        result = denoiser.denoise_frames(input_dir, output_dir)

        assert isinstance(result, TemporalDenoiseResult)
        assert result.frames_processed > 0
        assert output_dir.exists()


class TestAutoTemporalDenoiser:
    """Test automatic temporal denoiser."""

    def test_presets(self):
        """Test that all presets are available."""
        for preset in ["fast", "balanced", "quality"]:
            denoiser = AutoTemporalDenoiser(quality_preset=preset)
            assert preset in denoiser.presets

    def test_fast_preset(self):
        """Test fast preset configuration."""
        denoiser = AutoTemporalDenoiser(quality_preset="fast")
        config = denoiser.presets["fast"]

        assert config.temporal_radius == 2
        assert config.enable_optical_flow is False
        assert config.flicker_mode == FlickerMode.LIGHT

    def test_balanced_preset(self):
        """Test balanced preset configuration."""
        denoiser = AutoTemporalDenoiser(quality_preset="balanced")
        config = denoiser.presets["balanced"]

        assert config.temporal_radius == 3
        assert config.enable_optical_flow is True
        assert config.flicker_mode == FlickerMode.ADAPTIVE

    def test_quality_preset(self):
        """Test quality preset configuration."""
        denoiser = AutoTemporalDenoiser(quality_preset="quality")
        config = denoiser.presets["quality"]

        assert config.temporal_radius == 4
        assert config.enable_optical_flow is True
        assert config.preserve_edges is True

    @requires_opencv
    def test_process(self, tmp_path):
        """Test auto denoiser processing."""
        import cv2

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(15):
            frame = np.random.randint(100, 200, (40, 40, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), frame)

        denoiser = AutoTemporalDenoiser(quality_preset="fast")
        result, analysis = denoiser.process(input_dir, output_dir)

        assert isinstance(result, TemporalDenoiseResult)
        assert isinstance(analysis, dict)
        assert result.output_dir == output_dir


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_temporal_denoiser(self):
        """Test create_temporal_denoiser function."""
        denoiser = create_temporal_denoiser(
            strength=0.7,
            temporal_radius=4,
            enable_optical_flow=True,
            enable_flicker_reduction=False,
        )

        assert isinstance(denoiser, TemporalDenoiser)
        assert denoiser.config.noise_strength == 0.7
        assert denoiser.config.temporal_radius == 4
        assert denoiser.config.enable_optical_flow is True
        assert denoiser.config.enable_flicker_reduction is False

    @requires_opencv
    def test_denoise_video_frames(self, tmp_path):
        """Test denoise_video_frames convenience function."""
        import cv2

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(10):
            frame = np.random.randint(100, 200, (30, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), frame)

        result = denoise_video_frames(input_dir, output_dir, strength=0.3)

        assert isinstance(result, TemporalDenoiseResult)

    @requires_opencv
    def test_auto_denoise_video(self, tmp_path):
        """Test auto_denoise_video convenience function."""
        import cv2

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(10):
            frame = np.random.randint(100, 200, (30, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), frame)

        result, analysis = auto_denoise_video(input_dir, output_dir, quality="fast")

        assert isinstance(result, TemporalDenoiseResult)
        assert isinstance(analysis, dict)


class TestDenoiseMethodEnum:
    """Test DenoiseMethod enum."""

    def test_all_methods_exist(self):
        """Test that all expected methods exist."""
        expected = [
            "MULTI_FRAME_AVERAGE",
            "OPTICAL_FLOW_WARP",
            "NON_LOCAL_MEANS_TEMPORAL",
            "BILATERAL_TEMPORAL",
            "VBM4D",
        ]

        for method_name in expected:
            assert hasattr(DenoiseMethod, method_name)


class TestFlickerModeEnum:
    """Test FlickerMode enum."""

    def test_all_modes_exist(self):
        """Test that all expected modes exist."""
        expected = ["LIGHT", "MEDIUM", "AGGRESSIVE", "ADAPTIVE"]

        for mode_name in expected:
            assert hasattr(FlickerMode, mode_name)

    def test_mode_values(self):
        """Test FlickerMode values."""
        assert FlickerMode.LIGHT.value == "light"
        assert FlickerMode.MEDIUM.value == "medium"
        assert FlickerMode.AGGRESSIVE.value == "aggressive"
        assert FlickerMode.ADAPTIVE.value == "adaptive"


class TestOpticalFlowMethodEnum:
    """Test OpticalFlowMethod enum."""

    def test_all_methods_exist(self):
        """Test that all expected methods exist."""
        expected = ["FARNEBACK", "LUCAS_KANADE", "DIS", "RAFT", "RIFE"]

        for method_name in expected:
            assert hasattr(OpticalFlowMethod, method_name)

    def test_method_values(self):
        """Test OpticalFlowMethod values."""
        assert OpticalFlowMethod.FARNEBACK.value == "farneback"
        assert OpticalFlowMethod.DIS.value == "dis"
        assert OpticalFlowMethod.RAFT.value == "raft"


class TestIntegration:
    """Integration tests for the temporal denoising pipeline."""

    @requires_opencv
    @pytest.mark.slow
    def test_full_pipeline_with_scene_changes(self, tmp_path):
        """Test full pipeline with scene changes."""
        import cv2

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create frames with a scene change in the middle
        for i in range(30):
            if i < 15:
                # Scene 1: dark frames
                frame = np.full((50, 50, 3), 50, dtype=np.uint8)
            else:
                # Scene 2: bright frames
                frame = np.full((50, 50, 3), 200, dtype=np.uint8)

            # Add noise
            noise = np.random.randint(-20, 20, (50, 50, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), frame)

        config = TemporalDenoiseConfig(
            temporal_radius=3,
            noise_strength=0.5,
            scene_change_threshold=0.5,
        )
        denoiser = TemporalDenoiser(config)

        # Analyze first
        analysis = denoiser.analyze(input_dir)

        # Should detect scene change around frame 15
        scene_changes = analysis.get("scene_changes", [])
        assert len(scene_changes) > 0

        # Process
        result = denoiser.denoise_frames(input_dir, output_dir)

        assert result.frames_processed > 0
        assert result.scene_changes_detected == scene_changes

    @requires_opencv
    @pytest.mark.slow
    def test_progress_callback(self, tmp_path):
        """Test that progress callback is called correctly."""
        import cv2

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for i in range(10):
            frame = np.random.randint(100, 200, (30, 30, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"frame_{i:08d}.png"), frame)

        progress_values = []

        def progress_callback(p):
            progress_values.append(p)

        config = TemporalDenoiseConfig(
            temporal_radius=1,
            enable_flicker_reduction=False,
            enable_optical_flow=False,
        )
        denoiser = TemporalDenoiser(config)
        denoiser.denoise_frames(input_dir, output_dir, progress_callback)

        # Progress should increase and end at 1.0
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0
        assert all(0 <= p <= 1 for p in progress_values)
        # Should be monotonically increasing (with possible repeated values)
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]
