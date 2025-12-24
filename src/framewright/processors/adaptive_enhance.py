"""Adaptive enhancement processor with content-aware parameter tuning.

This module automatically adjusts enhancement parameters based on
content analysis, optimizing for different types of video content.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from .analyzer import FrameAnalyzer, VideoAnalysis, ContentType, DegradationType
from .defect_repair import AutoDefectProcessor, DefectRepairResult, DefectType
from .face_restore import FaceRestorer, FaceRestorationResult, FaceModel

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveEnhanceResult:
    """Result of adaptive enhancement processing."""
    analysis: Optional[VideoAnalysis] = None
    defect_repair: Optional[DefectRepairResult] = None
    face_restoration: Optional[FaceRestorationResult] = None
    frames_processed: int = 0
    stages_applied: List[str] = None

    def __post_init__(self):
        if self.stages_applied is None:
            self.stages_applied = []

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Frames processed: {self.frames_processed}"]
        lines.append(f"Stages applied: {', '.join(self.stages_applied) or 'none'}")

        if self.analysis:
            lines.append(f"Content type: {self.analysis.primary_content.name}")
            lines.append(f"Degradation: {self.analysis.degradation_severity}")

        if self.defect_repair:
            lines.append(f"Defects repaired: {self.defect_repair.defects_repaired}")

        if self.face_restoration:
            lines.append(f"Faces restored: {self.face_restoration.faces_restored}")

        return "\n".join(lines)


class AdaptiveEnhancer:
    """Content-aware adaptive enhancement pipeline.

    This processor automatically:
    1. Analyzes video content and degradation
    2. Applies defect repairs (scratches, dust, grain)
    3. Applies face restoration if faces detected
    4. Adjusts parameters based on content type

    All operations are fully automatic with no manual intervention required.
    """

    def __init__(
        self,
        enable_analysis: bool = True,
        enable_defect_repair: bool = True,
        enable_face_restoration: bool = True,
        face_model: FaceModel = FaceModel.GFPGAN_V1_4,
        scratch_sensitivity: float = 0.5,
        dust_sensitivity: float = 0.5,
        grain_reduction: float = 0.3,
    ):
        """Initialize adaptive enhancer.

        Args:
            enable_analysis: Run pre-scan analysis
            enable_defect_repair: Apply defect repairs
            enable_face_restoration: Apply face restoration
            face_model: Face restoration model to use
            scratch_sensitivity: Sensitivity for scratch detection (0-1)
            dust_sensitivity: Sensitivity for dust detection (0-1)
            grain_reduction: Default grain reduction strength (0-1)
        """
        self.enable_analysis = enable_analysis
        self.enable_defect_repair = enable_defect_repair
        self.enable_face_restoration = enable_face_restoration
        self.face_model = face_model
        self.scratch_sensitivity = scratch_sensitivity
        self.dust_sensitivity = dust_sensitivity
        self.grain_reduction = grain_reduction

        # Initialize sub-processors
        self.analyzer = FrameAnalyzer() if enable_analysis else None
        self.defect_processor = AutoDefectProcessor(
            scratch_sensitivity=scratch_sensitivity,
            dust_sensitivity=dust_sensitivity,
            grain_reduction=grain_reduction,
        ) if enable_defect_repair else None
        self.face_restorer = FaceRestorer(
            model=face_model,
        ) if enable_face_restoration else None

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AdaptiveEnhanceResult:
        """Process video with adaptive enhancement.

        Args:
            video_path: Path to source video
            output_dir: Directory for output
            progress_callback: Callback(stage, progress) for updates

        Returns:
            AdaptiveEnhanceResult with processing details
        """
        result = AdaptiveEnhanceResult()

        # Phase 1: Analysis
        if self.analyzer:
            if progress_callback:
                progress_callback("analysis", 0.0)

            logger.info("Running video analysis...")
            result.analysis = self.analyzer.analyze_video(video_path)
            result.stages_applied.append("analysis")

            logger.info(
                f"Analysis complete: {result.analysis.primary_content.name}, "
                f"degradation={result.analysis.degradation_severity}"
            )

            if progress_callback:
                progress_callback("analysis", 1.0)

        return result

    def process_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        analysis: Optional[VideoAnalysis] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AdaptiveEnhanceResult:
        """Process frames with adaptive enhancement.

        Args:
            input_dir: Directory with input frames
            output_dir: Directory for output frames
            analysis: Pre-computed analysis (if None, will analyze)
            progress_callback: Callback(stage, progress) for updates

        Returns:
            AdaptiveEnhanceResult with processing details
        """
        import shutil

        result = AdaptiveEnhanceResult()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Analysis (if not provided)
        if analysis is None and self.analyzer:
            if progress_callback:
                progress_callback("analysis", 0.0)

            logger.info("Analyzing frames...")
            analysis = self.analyzer.analyze_frames_dir(input_dir)
            result.stages_applied.append("analysis")

            if progress_callback:
                progress_callback("analysis", 1.0)

        result.analysis = analysis

        # Determine what processing to apply based on analysis
        apply_defect_repair = self.enable_defect_repair
        apply_face_restore = self.enable_face_restoration

        if analysis:
            # Override based on analysis recommendations
            if analysis.enable_scratch_removal or \
               analysis.degradation_severity in ("moderate", "heavy"):
                apply_defect_repair = True

            if analysis.enable_face_restoration or analysis.face_frame_ratio > 0.3:
                apply_face_restore = True

            logger.info(
                f"Adaptive decisions: defect_repair={apply_defect_repair}, "
                f"face_restore={apply_face_restore}"
            )

        # Track current frames directory (for chaining operations)
        current_frames_dir = input_dir

        # Phase 2: Defect repair
        if apply_defect_repair and self.defect_processor:
            if progress_callback:
                progress_callback("defect_repair", 0.0)

            logger.info("Applying defect repairs...")
            defect_output = output_dir / "_defect_repair"
            defect_output.mkdir(parents=True, exist_ok=True)

            def defect_progress(p):
                if progress_callback:
                    progress_callback("defect_repair", p)

            defect_result, detected_types = self.defect_processor.process(
                current_frames_dir,
                defect_output,
                defect_progress,
            )

            result.defect_repair = defect_result
            result.stages_applied.append("defect_repair")
            current_frames_dir = defect_output

            logger.info(
                f"Defect repair complete: {defect_result.defects_repaired} repaired"
            )

        # Phase 3: Face restoration
        if apply_face_restore and self.face_restorer and self.face_restorer.is_available():
            if progress_callback:
                progress_callback("face_restoration", 0.0)

            logger.info("Applying face restoration...")
            face_output = output_dir / "_face_restore"
            face_output.mkdir(parents=True, exist_ok=True)

            def face_progress(p):
                if progress_callback:
                    progress_callback("face_restoration", p)

            face_result = self.face_restorer.restore_frames(
                current_frames_dir,
                face_output,
                face_progress,
            )

            result.face_restoration = face_result
            result.stages_applied.append("face_restoration")
            current_frames_dir = face_output

            logger.info(
                f"Face restoration complete: {face_result.faces_restored} restored"
            )
        elif apply_face_restore and self.face_restorer:
            logger.warning(
                "Face restoration requested but GFPGAN/CodeFormer not available"
            )

        # Phase 4: Copy final frames to output
        if current_frames_dir != output_dir:
            final_output = output_dir / "final"
            final_output.mkdir(parents=True, exist_ok=True)

            frames = sorted(current_frames_dir.glob("*.png"))
            for frame in frames:
                shutil.copy(frame, final_output / frame.name)

            result.frames_processed = len(frames)

            # Cleanup intermediate directories
            for subdir in ["_defect_repair", "_face_restore"]:
                subdir_path = output_dir / subdir
                if subdir_path.exists() and subdir_path != current_frames_dir:
                    shutil.rmtree(subdir_path)
        else:
            result.frames_processed = len(list(output_dir.glob("*.png")))

        if progress_callback:
            progress_callback("complete", 1.0)

        return result

    def get_recommended_config(
        self,
        analysis: VideoAnalysis,
    ) -> Dict[str, Any]:
        """Get recommended configuration based on analysis.

        Args:
            analysis: Video analysis results

        Returns:
            Dictionary of recommended configuration values
        """
        config = {
            "scale_factor": analysis.recommended_scale,
            "model_name": analysis.recommended_model,
            "enable_face_restoration": analysis.enable_face_restoration,
            "enable_defect_repair": analysis.enable_scratch_removal,
            "grain_reduction": analysis.recommended_denoise,
            "enable_interpolation": analysis.recommended_target_fps is not None,
            "target_fps": analysis.recommended_target_fps,
        }

        # Content-specific adjustments
        if analysis.primary_content == ContentType.ANIMATION:
            config["model_name"] = "realesrgan-x4plus-anime"
            config["grain_reduction"] = 0.0  # Preserve anime style
        elif analysis.primary_content == ContentType.LOW_LIGHT:
            config["grain_reduction"] = min(0.5, config["grain_reduction"] + 0.2)
        elif analysis.primary_content in (ContentType.FACE_PORTRAIT, ContentType.FACE_GROUP):
            config["enable_face_restoration"] = True

        return config


class AutoEnhancePipeline:
    """Fully automated enhancement pipeline.

    Combines analysis, defect repair, face restoration, and parameter
    optimization into a single automated workflow.
    """

    def __init__(self):
        """Initialize the auto-enhancement pipeline."""
        self.enhancer = AdaptiveEnhancer(
            enable_analysis=True,
            enable_defect_repair=True,
            enable_face_restoration=True,
        )

    def enhance(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AdaptiveEnhanceResult:
        """Run complete auto-enhancement pipeline.

        Args:
            input_dir: Directory with input frames
            output_dir: Directory for output frames
            progress_callback: Optional progress callback

        Returns:
            AdaptiveEnhanceResult with all processing details
        """
        logger.info("Starting auto-enhancement pipeline...")

        result = self.enhancer.process_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        logger.info(f"Auto-enhancement complete:\n{result.summary()}")
        return result

    def get_config_recommendations(
        self,
        video_path: Path,
    ) -> Dict[str, Any]:
        """Analyze video and get configuration recommendations.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary of recommended configuration values
        """
        analyzer = FrameAnalyzer()
        analysis = analyzer.analyze_video(video_path)
        return self.enhancer.get_recommended_config(analysis)
