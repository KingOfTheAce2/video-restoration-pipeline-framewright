"""Preset recommendation system for FrameWright.

Provides intelligent preset and setting recommendations based on:
- Video analysis results
- Available hardware
- User preferences
- Content characteristics
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging

from .auto_detect import (
    AnalysisResult,
    ContentType,
    DegradationType,
    Era,
    SmartAnalyzer,
)

logger = logging.getLogger(__name__)


class RecommendationReason(Enum):
    """Reasons for recommendations."""
    CONTENT_TYPE = "content_type"
    DEGRADATION = "degradation"
    ERA = "era"
    RESOLUTION = "resolution"
    HARDWARE = "hardware"
    USER_PREFERENCE = "user_preference"
    FACES_DETECTED = "faces_detected"
    BW_FOOTAGE = "bw_footage"
    LOW_FPS = "low_fps"
    MISSING_FRAMES = "missing_frames"


@dataclass
class Recommendation:
    """A single recommendation with explanation."""
    setting: str
    value: Any
    reason: RecommendationReason
    explanation: str
    confidence: float = 0.8  # 0-1, how confident we are
    priority: int = 1  # 1=high, 3=low

    def __str__(self) -> str:
        return f"{self.setting}={self.value} ({self.explanation})"


@dataclass
class RecommendationSet:
    """Complete set of recommendations."""
    preset: str
    preset_explanation: str
    recommendations: List[Recommendation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_quality_score: float = 0.8  # Expected output quality 0-1
    estimated_vram_gb: float = 4.0
    processing_stages: List[str] = field(default_factory=list)

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert recommendations to config dictionary."""
        config = {"preset": self.preset}
        for rec in self.recommendations:
            config[rec.setting] = rec.value
        return config

    def get_high_priority(self) -> List[Recommendation]:
        """Get high priority recommendations."""
        return [r for r in self.recommendations if r.priority == 1]

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Recommended Preset: {self.preset.upper()}",
            f"  {self.preset_explanation}",
            "",
            "Key Settings:",
        ]
        for rec in self.get_high_priority():
            lines.append(f"  - {rec.setting}: {rec.value}")
            lines.append(f"    ({rec.explanation})")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)


class PresetRecommender:
    """Intelligent preset and settings recommender.

    Analyzes video characteristics and generates optimal
    restoration settings with explanations.

    Example:
        >>> recommender = PresetRecommender()
        >>> recs = recommender.recommend("old_film.mp4")
        >>> print(recs.summary())
        >>> config = recs.to_config_dict()
    """

    # Preset characteristics
    PRESET_INFO = {
        "fast": {
            "quality": 0.6,
            "speed": 0.9,
            "vram_gb": 2,
            "description": "Quick processing with good quality",
        },
        "balanced": {
            "quality": 0.75,
            "speed": 0.6,
            "vram_gb": 4,
            "description": "Best balance of speed and quality",
        },
        "quality": {
            "quality": 0.85,
            "speed": 0.3,
            "vram_gb": 8,
            "description": "High quality with moderate processing time",
        },
        "ultimate": {
            "quality": 0.95,
            "speed": 0.1,
            "vram_gb": 16,
            "description": "Maximum quality using all available techniques",
        },
    }

    def __init__(self, available_vram_gb: float = 8.0):
        """Initialize recommender.

        Args:
            available_vram_gb: Available GPU VRAM in GB
        """
        self.available_vram_gb = available_vram_gb
        self._analyzer = SmartAnalyzer()

    def recommend(
        self,
        video_path: Optional[Path] = None,
        analysis: Optional[AnalysisResult] = None,
        user_priority: str = "balanced",  # speed, balanced, quality, maximum
    ) -> RecommendationSet:
        """Generate recommendations for a video.

        Args:
            video_path: Path to video file (analyzed if no analysis provided)
            analysis: Pre-computed analysis result
            user_priority: User's priority preference

        Returns:
            RecommendationSet with preset and detailed recommendations
        """
        # Get analysis
        if analysis is None and video_path:
            analysis = self._analyzer.analyze(video_path)
        elif analysis is None:
            # No video to analyze, return defaults
            return self._default_recommendations(user_priority)

        # Start building recommendations
        recs = RecommendationSet(
            preset="balanced",
            preset_explanation="Good balance of quality and speed",
        )

        # Determine base preset
        self._recommend_preset(analysis, user_priority, recs)

        # Add specific recommendations
        self._recommend_preprocessing(analysis, recs)
        self._recommend_super_resolution(analysis, recs)
        self._recommend_face_enhancement(analysis, recs)
        self._recommend_frame_generation(analysis, recs)
        self._recommend_interpolation(analysis, recs)
        self._recommend_temporal(analysis, recs)
        self._recommend_colorization(analysis, recs)

        # Calculate estimates
        recs.estimated_vram_gb = self._estimate_vram(recs)
        recs.estimated_quality_score = self._estimate_quality(analysis, recs)

        # Add hardware warnings
        if recs.estimated_vram_gb > self.available_vram_gb:
            recs.warnings.append(
                f"Settings may require {recs.estimated_vram_gb:.1f}GB VRAM "
                f"(you have {self.available_vram_gb:.1f}GB)"
            )

        # Build processing stages list
        recs.processing_stages = self._build_stages_list(recs)

        return recs

    def _recommend_preset(
        self,
        analysis: AnalysisResult,
        user_priority: str,
        recs: RecommendationSet,
    ) -> None:
        """Determine the best preset."""
        # Start with user preference
        priority_map = {
            "speed": "fast",
            "balanced": "balanced",
            "quality": "quality",
            "maximum": "ultimate",
        }
        base_preset = priority_map.get(user_priority, "balanced")

        # Upgrade for archive footage
        if analysis.content.era in (Era.SILENT_ERA, Era.EARLY_SOUND, Era.CLASSIC):
            if base_preset in ("fast", "balanced"):
                base_preset = "quality"
                recs.recommendations.append(Recommendation(
                    setting="preset_upgrade",
                    value="quality",
                    reason=RecommendationReason.ERA,
                    explanation=f"Archive footage ({analysis.content.era.value}) benefits from higher quality",
                    priority=1,
                ))

        # Upgrade for severe degradation
        if analysis.degradation.severity in ("severe", "heavy"):
            if base_preset != "ultimate":
                base_preset = "ultimate" if self.available_vram_gb >= 16 else "quality"
                recs.recommendations.append(Recommendation(
                    setting="preset_upgrade",
                    value=base_preset,
                    reason=RecommendationReason.DEGRADATION,
                    explanation=f"Severe degradation detected - using advanced restoration",
                    priority=1,
                ))

        # Check VRAM constraints
        preset_vram = self.PRESET_INFO[base_preset]["vram_gb"]
        if preset_vram > self.available_vram_gb:
            # Downgrade if needed
            for preset in ["quality", "balanced", "fast"]:
                if self.PRESET_INFO[preset]["vram_gb"] <= self.available_vram_gb:
                    base_preset = preset
                    recs.warnings.append(
                        f"Preset downgraded to '{preset}' due to VRAM limitations"
                    )
                    break

        recs.preset = base_preset
        recs.preset_explanation = self.PRESET_INFO[base_preset]["description"]

    def _recommend_preprocessing(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend preprocessing stages."""
        # QP artifact removal for compressed video
        if (DegradationType.COMPRESSION in analysis.degradation.degradations or
            analysis.degradation.compression_level > 0.3):
            recs.recommendations.append(Recommendation(
                setting="enable_qp_artifact_removal",
                value=True,
                reason=RecommendationReason.DEGRADATION,
                explanation="Compression artifacts detected - deblocking recommended",
                confidence=min(1.0, analysis.degradation.compression_level + 0.3),
                priority=1,
            ))

        # TAP denoising for noisy footage
        if (DegradationType.NOISE in analysis.degradation.degradations or
            analysis.degradation.noise_level > 0.3):
            recs.recommendations.append(Recommendation(
                setting="enable_tap_denoise",
                value=True,
                reason=RecommendationReason.DEGRADATION,
                explanation="Noise detected - neural denoising recommended",
                confidence=min(1.0, analysis.degradation.noise_level + 0.3),
                priority=1,
            ))

            # Adjust strength based on noise level
            strength = min(1.0, analysis.degradation.noise_level * 1.5)
            recs.recommendations.append(Recommendation(
                setting="tap_strength",
                value=round(strength, 2),
                reason=RecommendationReason.DEGRADATION,
                explanation=f"Noise level {analysis.degradation.noise_level:.0%} - strength adjusted",
                priority=2,
            ))

        # Deduplication for old film
        if analysis.content.era in (Era.SILENT_ERA, Era.EARLY_SOUND):
            recs.recommendations.append(Recommendation(
                setting="enable_deduplication",
                value=True,
                reason=RecommendationReason.ERA,
                explanation="Old film often has duplicate frames from FPS conversion",
                priority=2,
            ))

    def _recommend_super_resolution(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend super-resolution settings."""
        # Scale factor based on input resolution
        if analysis.width < 480:
            scale = 4
            reason = "Very low resolution - maximum upscaling recommended"
        elif analysis.width < 720:
            scale = 4
            reason = "Low resolution - 4x upscaling to HD"
        elif analysis.width < 1080:
            scale = 2
            reason = "SD resolution - 2x upscaling to Full HD"
        else:
            scale = 2
            reason = "HD resolution - modest upscaling"

        recs.recommendations.append(Recommendation(
            setting="scale_factor",
            value=scale,
            reason=RecommendationReason.RESOLUTION,
            explanation=reason,
            priority=1,
        ))

        # SR model selection
        if recs.preset == "ultimate" and self.available_vram_gb >= 12:
            recs.recommendations.append(Recommendation(
                setting="sr_model",
                value="diffusion",
                reason=RecommendationReason.HARDWARE,
                explanation="Diffusion SR provides best quality for archive footage",
                priority=1,
            ))
        elif analysis.content.content_type == ContentType.ANIMATION:
            recs.recommendations.append(Recommendation(
                setting="model_name",
                value="realesr-animevideov3",
                reason=RecommendationReason.CONTENT_TYPE,
                explanation="Animation-optimized model for cartoons/anime",
                priority=1,
            ))

    def _recommend_face_enhancement(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend face enhancement settings."""
        if analysis.content.has_faces and analysis.content.face_percentage > 10:
            recs.recommendations.append(Recommendation(
                setting="auto_face_restore",
                value=True,
                reason=RecommendationReason.FACES_DETECTED,
                explanation=f"Faces detected in {analysis.content.face_percentage:.0f}% of frames",
                priority=1,
            ))

            # Face model selection
            if recs.preset in ("quality", "ultimate"):
                recs.recommendations.append(Recommendation(
                    setting="face_model",
                    value="aesrgan",
                    reason=RecommendationReason.FACES_DETECTED,
                    explanation="AESRGAN provides better detail than GFPGAN",
                    priority=2,
                ))
        else:
            recs.recommendations.append(Recommendation(
                setting="auto_face_restore",
                value=False,
                reason=RecommendationReason.FACES_DETECTED,
                explanation="Few faces detected - skipping face enhancement",
                priority=3,
            ))

    def _recommend_frame_generation(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend frame generation for damaged footage."""
        if analysis.degradation.frame_damage_ratio > 0.01:
            recs.recommendations.append(Recommendation(
                setting="enable_frame_generation",
                value=True,
                reason=RecommendationReason.MISSING_FRAMES,
                explanation=f"~{analysis.degradation.frame_damage_ratio*100:.1f}% frames appear missing/damaged",
                priority=1,
            ))
            recs.warnings.append(
                f"Approximately {int(analysis.total_frames * analysis.degradation.frame_damage_ratio)} "
                "frames may need reconstruction"
            )

    def _recommend_interpolation(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend frame interpolation settings."""
        if analysis.fps < 24:
            recs.recommendations.append(Recommendation(
                setting="enable_interpolation",
                value=True,
                reason=RecommendationReason.LOW_FPS,
                explanation=f"Low frame rate ({analysis.fps:.1f}fps) - interpolation recommended",
                priority=1,
            ))

            # Target FPS
            if analysis.fps < 15:
                target = 30
            elif analysis.fps < 20:
                target = 25
            else:
                target = 24

            recs.recommendations.append(Recommendation(
                setting="target_fps",
                value=target,
                reason=RecommendationReason.LOW_FPS,
                explanation=f"Interpolate from {analysis.fps:.1f}fps to {target}fps",
                priority=2,
            ))

    def _recommend_temporal(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend temporal consistency settings."""
        if (DegradationType.FLICKER in analysis.degradation.degradations or
            analysis.degradation.flicker_intensity > 0.3):
            method = "hybrid" if recs.preset in ("quality", "ultimate") else "optical_flow"
            recs.recommendations.append(Recommendation(
                setting="temporal_method",
                value=method,
                reason=RecommendationReason.DEGRADATION,
                explanation="Flicker detected - temporal consistency recommended",
                priority=1,
            ))

    def _recommend_colorization(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> None:
        """Recommend colorization settings."""
        if analysis.content.is_black_and_white:
            recs.recommendations.append(Recommendation(
                setting="colorization_available",
                value=True,
                reason=RecommendationReason.BW_FOOTAGE,
                explanation="B&W footage detected - colorization available if desired",
                priority=2,
            ))
            recs.warnings.append(
                "For best colorization, provide reference color images of similar subjects"
            )

    def _estimate_vram(self, recs: RecommendationSet) -> float:
        """Estimate required VRAM."""
        base = self.PRESET_INFO[recs.preset]["vram_gb"]

        # Add for specific features
        config = recs.to_config_dict()
        if config.get("enable_tap_denoise"):
            base += 2
        if config.get("sr_model") == "diffusion":
            base += 8
        if config.get("enable_frame_generation"):
            base += 4

        return base

    def _estimate_quality(
        self,
        analysis: AnalysisResult,
        recs: RecommendationSet,
    ) -> float:
        """Estimate output quality score."""
        # Start with preset quality
        base = self.PRESET_INFO[recs.preset]["quality"]

        # Adjust based on degradation (worse input = lower ceiling)
        degradation_penalty = {
            "minimal": 0,
            "light": 0.05,
            "moderate": 0.1,
            "heavy": 0.15,
            "severe": 0.2,
        }
        penalty = degradation_penalty.get(analysis.degradation.severity, 0.1)

        return min(1.0, base - penalty + 0.05)  # Small bonus for optimized settings

    def _build_stages_list(self, recs: RecommendationSet) -> List[str]:
        """Build list of processing stages."""
        stages = []
        config = recs.to_config_dict()

        if config.get("enable_qp_artifact_removal"):
            stages.append("QP Artifact Removal")
        if config.get("enable_tap_denoise"):
            stages.append("TAP Neural Denoising")
        if config.get("enable_frame_generation"):
            stages.append("Missing Frame Generation")

        sr_model = config.get("sr_model", "realesrgan")
        scale = config.get("scale_factor", 2)
        stages.append(f"Super-Resolution ({sr_model}) {scale}x")

        if config.get("auto_face_restore"):
            face_model = config.get("face_model", "gfpgan")
            stages.append(f"Face Enhancement ({face_model})")

        if config.get("enable_interpolation"):
            target_fps = config.get("target_fps", 30)
            stages.append(f"Frame Interpolation → {target_fps}fps")

        if config.get("temporal_method"):
            stages.append("Temporal Consistency")

        stages.append("Video Reassembly")

        return stages

    def _default_recommendations(self, user_priority: str) -> RecommendationSet:
        """Return default recommendations when no video provided."""
        priority_map = {
            "speed": "fast",
            "balanced": "balanced",
            "quality": "quality",
            "maximum": "ultimate",
        }
        preset = priority_map.get(user_priority, "balanced")

        return RecommendationSet(
            preset=preset,
            preset_explanation=self.PRESET_INFO[preset]["description"],
            recommendations=[
                Recommendation(
                    setting="scale_factor",
                    value=2,
                    reason=RecommendationReason.USER_PREFERENCE,
                    explanation="Default 2x upscaling",
                    priority=1,
                ),
            ],
            estimated_quality_score=self.PRESET_INFO[preset]["quality"],
            estimated_vram_gb=self.PRESET_INFO[preset]["vram_gb"],
            processing_stages=["Super-Resolution (realesrgan) 2x", "Video Reassembly"],
        )


def get_recommendations(
    video_path: Optional[Path] = None,
    analysis: Optional[AnalysisResult] = None,
    user_priority: str = "balanced",
    available_vram_gb: float = 8.0,
) -> RecommendationSet:
    """Get recommendations for a video.

    Convenience function for getting recommendations.

    Args:
        video_path: Path to video file
        analysis: Pre-computed analysis
        user_priority: User's priority (speed/balanced/quality/maximum)
        available_vram_gb: Available GPU VRAM

    Returns:
        RecommendationSet with recommendations
    """
    recommender = PresetRecommender(available_vram_gb=available_vram_gb)
    return recommender.recommend(video_path, analysis, user_priority)
