"""Smart preset selector for automatic optimal configuration.

Analyzes video characteristics and hardware to automatically
select the best preset configuration.

Example:
    >>> from pathlib import Path
    >>> from framewright.presets.smart_selector import SmartPresetSelector
    >>> from framewright.presets.registry import HardwareInfo
    >>>
    >>> selector = SmartPresetSelector()
    >>> hardware = HardwareInfo.detect()
    >>>
    >>> # Auto-select optimal preset for video
    >>> config = selector.select(Path("old_vhs_tape.mp4"), hardware)
    >>> print(f"Selected: {config.name}")
    >>> print(f"Reasoning: {selector.last_reasoning}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analyzer import (
    VideoAnalyzer,
    VideoCharacteristics,
    VideoEra,
    VideoSource,
    DefectType,
)
from .registry import (
    HardwareInfo,
    PresetConfig,
    PresetRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionReasoning:
    """Explains why a particular preset was selected.

    Attributes:
        video_characteristics: Summary of detected video properties
        hardware_tier: Selected hardware tier
        base_preset: Selected base preset name
        style: Selected style (if any)
        decisions: List of decision explanations
        warnings: List of warnings about the selection
        estimated_time_minutes: Estimated processing time
        estimated_quality_improvement: Expected quality gain description
    """
    video_characteristics: Dict[str, Any] = field(default_factory=dict)
    hardware_tier: str = ""
    base_preset: str = ""
    style: Optional[str] = None
    decisions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_time_minutes: float = 0.0
    estimated_quality_improvement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_characteristics": self.video_characteristics,
            "hardware_tier": self.hardware_tier,
            "base_preset": self.base_preset,
            "style": self.style,
            "decisions": self.decisions,
            "warnings": self.warnings,
            "estimated_time_minutes": self.estimated_time_minutes,
            "estimated_quality_improvement": self.estimated_quality_improvement,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Selected: {self.base_preset}" + (f" + {self.style}" if self.style else ""),
            f"Hardware Tier: {self.hardware_tier}",
            "",
            "Decisions:",
        ]
        for decision in self.decisions:
            lines.append(f"  - {decision}")

        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings:
                lines.append(f"  ! {warning}")

        if self.estimated_time_minutes > 0:
            lines.append(f"\nEstimated Time: {self.estimated_time_minutes:.1f} minutes")

        if self.estimated_quality_improvement:
            lines.append(f"Expected Quality: {self.estimated_quality_improvement}")

        return "\n".join(lines)


class SmartPresetSelector:
    """Automatically selects optimal preset based on video and hardware.

    Analyzes the input video to detect characteristics like:
    - Content era (silent film, VHS era, modern, etc.)
    - Source format (film, VHS, DVD, digital)
    - Defects present (noise, scratches, compression artifacts)
    - Content type indicators (faces, animation style)

    Then selects the best preset combination considering:
    - Available hardware capabilities
    - User's quality preference
    - Video duration and complexity
    - Specific defects that need addressing

    Attributes:
        quality_preference: Default quality level ("fast", "balanced", "best")
        preserve_authenticity: Whether to prioritize era-appropriate processing
        analyzer: VideoAnalyzer instance for content analysis
        last_reasoning: Reasoning for the most recent selection
    """

    def __init__(
        self,
        quality_preference: str = "balanced",
        preserve_authenticity: bool = True,
        sample_frames: int = 30,
    ):
        """Initialize the smart selector.

        Args:
            quality_preference: Default quality ("fast", "balanced", "best")
            preserve_authenticity: Prioritize authentic period look
            sample_frames: Number of frames to sample for analysis
        """
        self.quality_preference = quality_preference
        self.preserve_authenticity = preserve_authenticity
        self.analyzer = VideoAnalyzer(sample_frames=sample_frames)
        self.last_reasoning: Optional[SelectionReasoning] = None

    def select(
        self,
        video_path: Path,
        hardware: Optional[HardwareInfo] = None,
        quality_override: Optional[str] = None,
        style_override: Optional[str] = None,
    ) -> PresetConfig:
        """Select optimal preset for a video.

        Args:
            video_path: Path to the video file.
            hardware: Hardware info. Auto-detected if None.
            quality_override: Override quality preference for this selection.
            style_override: Force a specific style instead of auto-detecting.

        Returns:
            PresetConfig optimized for the video and hardware.

        Raises:
            FileNotFoundError: If video file doesn't exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Auto-detect hardware if not provided
        if hardware is None:
            hardware = HardwareInfo.detect()

        # Initialize reasoning
        reasoning = SelectionReasoning()
        reasoning.hardware_tier = PresetRegistry.get_hardware_tier(hardware)

        # Analyze video
        try:
            chars = self.analyzer.analyze(video_path)
            reasoning.video_characteristics = chars.to_dict()
        except Exception as e:
            logger.warning(f"Video analysis failed: {e}. Using defaults.")
            chars = VideoCharacteristics()
            reasoning.warnings.append(f"Analysis failed: {e}")

        # Select base preset
        quality = quality_override or self.quality_preference
        base_preset = self._select_base_preset(chars, hardware, quality, reasoning)

        # Select style
        if style_override:
            style = style_override
            reasoning.decisions.append(f"Style override: {style}")
        else:
            style = self._select_style(chars, reasoning)

        reasoning.base_preset = base_preset
        reasoning.style = style

        # Get the merged preset
        if style:
            preset = PresetRegistry.with_style(base_preset, style, hardware)
        else:
            preset = PresetRegistry.get_for_hardware(base_preset, hardware)

        # Apply video-specific adjustments
        self._apply_video_adjustments(preset, chars, hardware, reasoning)

        # Estimate processing time
        self._estimate_processing_time(preset, chars, hardware, reasoning)

        # Store reasoning
        self.last_reasoning = reasoning
        preset.warnings.extend(reasoning.warnings)

        logger.info(f"Selected preset: {preset.name}")
        logger.debug(str(reasoning))

        return preset

    def _select_base_preset(
        self,
        chars: VideoCharacteristics,
        hardware: HardwareInfo,
        quality_pref: str,
        reasoning: SelectionReasoning,
    ) -> str:
        """Select base preset based on content and hardware.

        Args:
            chars: Video characteristics.
            hardware: Hardware info.
            quality_pref: User's quality preference.
            reasoning: Reasoning object to update.

        Returns:
            Base preset name.
        """
        # Check if hardware constrains quality
        tier = reasoning.hardware_tier

        # CPU-only or very low VRAM: force fast
        if tier == "cpu_only" or hardware.vram_gb < 4:
            if quality_pref != "fast":
                reasoning.decisions.append(
                    f"Downgraded from '{quality_pref}' to 'fast' due to limited hardware"
                )
                reasoning.warnings.append(
                    f"Hardware tier '{tier}' can only support 'fast' preset"
                )
            return "fast"

        # Low VRAM (4-6GB): cap at balanced
        if hardware.vram_gb < 6 and quality_pref == "best":
            reasoning.decisions.append(
                "Downgraded from 'best' to 'balanced' due to 4-6GB VRAM limit"
            )
            return "balanced"

        # Consider video complexity
        complexity_score = self._calculate_complexity(chars)

        if complexity_score > 0.8 and quality_pref == "best":
            # Very complex video - check if we have enough resources
            if hardware.vram_gb < 12:
                reasoning.decisions.append(
                    f"Complex video (score: {complexity_score:.2f}) would strain "
                    f"{hardware.vram_gb:.1f}GB VRAM. Using 'balanced'."
                )
                return "balanced"

        # Consider video duration
        if chars.duration_seconds > 7200:  # > 2 hours
            if quality_pref == "best":
                reasoning.decisions.append(
                    "Long video (>2h) - consider 'balanced' for reasonable processing time"
                )
                reasoning.warnings.append(
                    "Very long video. 'best' preset may take many hours to process."
                )

        # Use requested quality
        reasoning.decisions.append(f"Using requested quality tier: {quality_pref}")
        return quality_pref

    def _select_style(
        self,
        chars: VideoCharacteristics,
        reasoning: SelectionReasoning,
    ) -> Optional[str]:
        """Auto-detect appropriate style based on video content.

        Args:
            chars: Video characteristics.
            reasoning: Reasoning object to update.

        Returns:
            Style name or None.
        """
        # VHS/analog detection
        if chars.has_vhs_artifacts or chars.source == VideoSource.VHS:
            reasoning.decisions.append(
                "Detected VHS artifacts - applying 'home_video' style"
            )
            return "home_video"

        if chars.source in (VideoSource.BETAMAX, VideoSource.HI8):
            reasoning.decisions.append(
                f"Detected {chars.source.value} source - applying 'home_video' style"
            )
            return "home_video"

        # Film detection
        if chars.source in (VideoSource.FILM_35MM, VideoSource.FILM_16MM, VideoSource.FILM_8MM):
            reasoning.decisions.append(
                f"Detected {chars.source.value} film source - applying 'film' style"
            )
            return "film"

        if chars.has_film_grain and chars.grain_intensity > 0.4:
            reasoning.decisions.append(
                f"Detected significant film grain (intensity: {chars.grain_intensity:.2f}) - "
                "applying 'film' style"
            )
            return "film"

        # Era-based selection
        if chars.era in (VideoEra.SILENT_FILM, VideoEra.EARLY_SOUND, VideoEra.GOLDEN_AGE):
            reasoning.decisions.append(
                f"Detected {chars.era.value} era content - applying 'archive' style"
            )
            return "archive"

        # Broadcast detection
        if chars.source == VideoSource.BROADCAST or (
            chars.is_interlaced and chars.width in (720, 1920)
        ):
            reasoning.decisions.append(
                "Detected broadcast/interlaced content - applying 'broadcast' style"
            )
            return "broadcast"

        # DVD/compression artifact heavy content
        if (
            DefectType.COMPRESSION_ARTIFACTS in chars.defects
            and chars.defect_severity.get(DefectType.COMPRESSION_ARTIFACTS, 0) > 0.5
        ):
            reasoning.decisions.append(
                "Heavy compression artifacts detected - applying 'web_video' style"
            )
            return "web_video"

        # No specific style detected
        reasoning.decisions.append(
            "No specific content type detected - using base preset without style"
        )
        return None

    def _calculate_complexity(self, chars: VideoCharacteristics) -> float:
        """Calculate video complexity score (0-1).

        Higher complexity means more processing required.

        Args:
            chars: Video characteristics.

        Returns:
            Complexity score between 0 and 1.
        """
        score = 0.0

        # Resolution factor
        pixels = chars.width * chars.height
        if pixels > 1920 * 1080:
            score += 0.2
        elif pixels > 1280 * 720:
            score += 0.1

        # Defect count
        defect_score = min(0.3, len(chars.defects) * 0.05)
        score += defect_score

        # Defect severity
        if chars.defect_severity:
            avg_severity = sum(chars.defect_severity.values()) / len(chars.defect_severity)
            score += avg_severity * 0.2

        # Face complexity
        if chars.has_faces:
            score += 0.1 + min(0.1, chars.face_count_avg * 0.02)

        # Noise level
        if chars.noise_level > 50:
            score += 0.1

        # Film grain (requires grain-preserving processing)
        if chars.has_film_grain:
            score += 0.1

        return min(1.0, score)

    def _apply_video_adjustments(
        self,
        preset: PresetConfig,
        chars: VideoCharacteristics,
        hardware: HardwareInfo,
        reasoning: SelectionReasoning,
    ) -> None:
        """Apply video-specific adjustments to preset.

        Args:
            preset: Preset to modify.
            chars: Video characteristics.
            hardware: Hardware info.
            reasoning: Reasoning object to update.
        """
        settings = preset.settings

        # Adjust denoising strength based on detected noise
        if chars.noise_level > 0 and "denoise_strength" not in settings:
            strength = min(1.0, chars.noise_level / 100 * 1.2)
            settings["denoise_strength"] = strength
            reasoning.decisions.append(
                f"Set denoise_strength to {strength:.2f} based on detected noise level"
            )

        # Enable deinterlacing if interlaced
        if chars.is_interlaced and not settings.get("enable_deinterlace"):
            settings["enable_deinterlace"] = True
            settings["deinterlace_method"] = "yadif"
            settings["field_order"] = chars.field_order
            reasoning.decisions.append(
                f"Enabled deinterlacing ({chars.field_order}) for interlaced content"
            )

        # Enable face restoration if faces detected
        if chars.has_faces and chars.face_count_avg > 0.5:
            if not settings.get("enable_face_restore"):
                settings["enable_face_restore"] = True
                reasoning.decisions.append(
                    f"Enabled face restoration (avg {chars.face_count_avg:.1f} faces/frame)"
                )

        # Adjust for grayscale content
        if chars.is_grayscale:
            settings["enable_colorization"] = False
            reasoning.decisions.append(
                "Grayscale content - colorization available but disabled by default"
            )

        # Enable frame interpolation for low fps
        if chars.fps > 0 and chars.fps < 20 and hardware.vram_gb >= 8:
            if not settings.get("enable_interpolation"):
                settings["enable_interpolation"] = True
                settings["target_fps"] = 24
                reasoning.decisions.append(
                    f"Enabled frame interpolation: {chars.fps:.1f} -> 24 fps"
                )

        # Compression artifact removal
        if DefectType.COMPRESSION_ARTIFACTS in chars.defects:
            severity = chars.defect_severity.get(DefectType.COMPRESSION_ARTIFACTS, 0.5)
            if not settings.get("enable_qp_artifact_removal"):
                settings["enable_qp_artifact_removal"] = True
                settings["qp_strength"] = min(1.0, severity * 1.2)
                reasoning.decisions.append(
                    f"Enabled compression artifact removal (severity: {severity:.1%})"
                )

        # Era-based authenticity limits
        if self.preserve_authenticity and chars.era != VideoEra.UNKNOWN:
            era_limits = self._get_era_limits(chars.era)
            for key, limit in era_limits.items():
                if key.endswith("_max"):
                    base_key = key[:-4]
                    if base_key in settings:
                        if settings[base_key] > limit:
                            settings[base_key] = limit
                            reasoning.decisions.append(
                                f"Capped {base_key} to {limit} for {chars.era.value} authenticity"
                            )

    def _get_era_limits(self, era: VideoEra) -> Dict[str, Any]:
        """Get processing limits for an era.

        Args:
            era: Video era.

        Returns:
            Dictionary of limit settings.
        """
        limits = {
            VideoEra.SILENT_FILM: {
                "scale_factor_max": 2,
                "face_restore_strength_max": 0.3,
                "denoise_strength_max": 0.5,
            },
            VideoEra.EARLY_SOUND: {
                "scale_factor_max": 2,
                "face_restore_strength_max": 0.4,
                "denoise_strength_max": 0.6,
            },
            VideoEra.GOLDEN_AGE: {
                "scale_factor_max": 4,
                "face_restore_strength_max": 0.5,
                "denoise_strength_max": 0.7,
            },
            VideoEra.HOME_VIDEO: {
                "face_restore_strength_max": 0.8,
            },
        }
        return limits.get(era, {})

    def _estimate_processing_time(
        self,
        preset: PresetConfig,
        chars: VideoCharacteristics,
        hardware: HardwareInfo,
        reasoning: SelectionReasoning,
    ) -> None:
        """Estimate processing time for the video.

        Args:
            preset: Selected preset.
            chars: Video characteristics.
            hardware: Hardware info.
            reasoning: Reasoning object to update.
        """
        # Base time per frame in milliseconds
        base_ms_per_frame = 50

        settings = preset.settings

        # Add time for each enabled feature
        if settings.get("enable_tap_denoise"):
            model = settings.get("tap_model", "nafnet")
            base_ms_per_frame += {"restormer": 200, "nafnet": 50, "tap": 150}.get(model, 50)

        if settings.get("enable_upscale", True):
            model = settings.get("sr_model", "realesrgan")
            scale = settings.get("scale_factor", 2)
            time_mult = {"diffusion": 50, "realesrgan": 1, "basicvsr": 2}.get(model, 1)
            base_ms_per_frame += 100 * scale * time_mult

        if settings.get("enable_face_restore"):
            base_ms_per_frame += 100 * max(1, chars.face_count_avg)

        if settings.get("enable_interpolation"):
            base_ms_per_frame += 150

        if settings.get("enable_qp_artifact_removal"):
            base_ms_per_frame += 30

        # Adjust for hardware
        vram_factor = 1.0
        if hardware.vram_gb >= 24:
            vram_factor = 0.5
        elif hardware.vram_gb >= 16:
            vram_factor = 0.7
        elif hardware.vram_gb >= 8:
            vram_factor = 1.0
        elif hardware.vram_gb >= 4:
            vram_factor = 2.0
        else:
            vram_factor = 10.0  # CPU processing

        adjusted_ms_per_frame = base_ms_per_frame * vram_factor

        # Total time
        total_frames = chars.total_frames or int(chars.duration_seconds * (chars.fps or 24))
        total_ms = adjusted_ms_per_frame * total_frames
        total_minutes = total_ms / 60000

        reasoning.estimated_time_minutes = total_minutes

        # Quality improvement estimate
        if len(chars.defects) > 3:
            reasoning.estimated_quality_improvement = "Significant improvement expected"
        elif len(chars.defects) > 1:
            reasoning.estimated_quality_improvement = "Moderate improvement expected"
        elif chars.defects:
            reasoning.estimated_quality_improvement = "Minor improvement expected"
        else:
            reasoning.estimated_quality_improvement = "Enhancement/upscaling only"

    def get_reasoning(self) -> Optional[SelectionReasoning]:
        """Get reasoning for the last selection.

        Returns:
            SelectionReasoning or None if no selection made.
        """
        return self.last_reasoning

    def explain_selection(self) -> str:
        """Get human-readable explanation of last selection.

        Returns:
            Explanation string.
        """
        if self.last_reasoning is None:
            return "No selection has been made yet."
        return str(self.last_reasoning)


def auto_select_preset(
    video_path: Path,
    hardware: Optional[HardwareInfo] = None,
    quality: str = "balanced",
    preserve_authenticity: bool = True,
) -> Tuple[PresetConfig, SelectionReasoning]:
    """Convenience function for automatic preset selection.

    Args:
        video_path: Path to the video file.
        hardware: Hardware info. Auto-detected if None.
        quality: Quality preference ("fast", "balanced", "best").
        preserve_authenticity: Preserve era-appropriate aesthetics.

    Returns:
        Tuple of (PresetConfig, SelectionReasoning).
    """
    selector = SmartPresetSelector(
        quality_preference=quality,
        preserve_authenticity=preserve_authenticity,
    )
    preset = selector.select(video_path, hardware)
    return preset, selector.last_reasoning
