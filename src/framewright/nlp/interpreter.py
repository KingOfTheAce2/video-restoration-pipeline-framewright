"""Command interpreter that converts parsed commands to restoration plans."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parser import ParsedCommand, CommandIntent

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStage:
    """A single stage in the restoration pipeline."""
    name: str
    processor: str
    settings: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    order: int = 0
    description: str = ""


@dataclass
class RestorationPlan:
    """Complete restoration plan generated from natural language."""
    # Source command
    original_command: str = ""

    # Input/Output
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    # Pipeline stages
    stages: List[ProcessingStage] = field(default_factory=list)

    # Global settings
    preset: str = "balanced"
    scale_factor: float = 1.0
    target_fps: Optional[float] = None
    output_format: str = "mp4"
    codec: str = "libx264"
    crf: int = 18

    # Authenticity settings
    preserve_authenticity: bool = True
    authenticity_level: float = 0.7  # 0 = modern look, 1 = fully authentic
    source_era: Optional[str] = None
    source_format: Optional[str] = None

    # Quality targets
    target_psnr: float = 35.0
    target_ssim: float = 0.90
    max_processing_time: Optional[float] = None

    # Estimated resources
    estimated_vram_gb: float = 0.0
    estimated_time_per_frame: float = 0.0

    # User feedback
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_config(self) -> Dict[str, Any]:
        """Convert plan to configuration dictionary."""
        config = {
            "preset": self.preset,
            "scale_factor": self.scale_factor,
            "output_format": self.output_format,
            "codec": self.codec,
            "crf": self.crf,
            "preserve_authenticity": self.preserve_authenticity,
            "authenticity_level": self.authenticity_level,
        }

        if self.target_fps:
            config["target_fps"] = self.target_fps

        if self.source_era:
            config["source_era"] = self.source_era

        if self.source_format:
            config["source_format"] = self.source_format

        # Add stage-specific settings
        for stage in self.stages:
            if stage.enabled:
                config[f"enable_{stage.name}"] = True
                for key, value in stage.settings.items():
                    config[f"{stage.name}_{key}"] = value

        return config

    def get_summary(self) -> str:
        """Get human-readable summary of the plan."""
        lines = ["Restoration Plan:"]
        lines.append(f"  Preset: {self.preset}")
        lines.append(f"  Scale: {self.scale_factor}x")

        if self.target_fps:
            lines.append(f"  Target FPS: {self.target_fps}")

        lines.append("")
        lines.append("Pipeline Stages:")

        for stage in sorted(self.stages, key=lambda s: s.order):
            if stage.enabled:
                lines.append(f"  {stage.order}. {stage.name}: {stage.description}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


class CommandInterpreter:
    """Interprets parsed commands into restoration plans."""

    def __init__(self):
        self._preset_configs = self._build_presets()
        self._era_configs = self._build_era_configs()
        self._format_configs = self._build_format_configs()

    def _build_presets(self) -> Dict[str, Dict[str, Any]]:
        """Build preset configurations."""
        return {
            "draft": {
                "scale_factor": 1,
                "crf": 23,
                "denoise_strength": 0.3,
                "sharpen_strength": 0.3,
                "use_ai_upscale": False,
                "subsample": 5,
            },
            "fast": {
                "scale_factor": 2,
                "crf": 20,
                "denoise_strength": 0.4,
                "sharpen_strength": 0.4,
                "use_ai_upscale": True,
                "model": "realesrgan",
            },
            "balanced": {
                "scale_factor": 2,
                "crf": 18,
                "denoise_strength": 0.5,
                "sharpen_strength": 0.5,
                "use_ai_upscale": True,
                "model": "realesrgan",
                "face_restore": True,
            },
            "quality": {
                "scale_factor": 4,
                "crf": 16,
                "denoise_strength": 0.6,
                "sharpen_strength": 0.6,
                "use_ai_upscale": True,
                "model": "realesrgan",
                "face_restore": True,
                "temporal_consistency": True,
            },
            "ultimate": {
                "scale_factor": 4,
                "crf": 14,
                "denoise_strength": 0.7,
                "sharpen_strength": 0.7,
                "use_ai_upscale": True,
                "model": "diffusion",
                "face_restore": True,
                "face_model": "aesrgan",
                "temporal_consistency": True,
                "temporal_method": "hybrid",
                "enable_tap_denoise": True,
            },
        }

    def _build_era_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build era-specific configurations."""
        return {
            "silent": {
                "source_era": "silent_film",
                "preserve_grain": True,
                "max_denoise": 0.3,
                "max_sharpen": 0.3,
                "typical_fps": 16,
                "typical_issues": ["scratches", "flicker", "damage"],
            },
            "1920s": {
                "source_era": "silent_film",
                "preserve_grain": True,
                "max_denoise": 0.3,
            },
            "1930s": {
                "source_era": "early_talkies",
                "preserve_grain": True,
                "max_denoise": 0.4,
            },
            "1940s": {
                "source_era": "golden_age",
                "preserve_grain": True,
                "max_denoise": 0.4,
            },
            "1950s": {
                "source_era": "golden_age",
                "preserve_grain": True,
                "max_denoise": 0.5,
            },
            "1960s": {
                "source_era": "new_hollywood",
                "preserve_grain": True,
                "max_denoise": 0.5,
            },
            "1970s": {
                "source_era": "new_hollywood",
                "preserve_grain": True,
                "max_denoise": 0.5,
            },
            "1980s": {
                "source_era": "video_era",
                "preserve_grain": False,
                "max_denoise": 0.7,
            },
            "1990s": {
                "source_era": "video_era",
                "preserve_grain": False,
                "max_denoise": 0.8,
            },
            "home_video": {
                "source_era": "video_era",
                "preserve_grain": False,
                "max_denoise": 0.8,
                "typical_issues": ["shake", "noise", "tracking"],
            },
        }

    def _build_format_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build format-specific configurations."""
        return {
            "vhs": {
                "source_format": "vhs",
                "enable_vhs_restoration": True,
                "fix_tracking": True,
                "fix_dropout": True,
                "fix_chroma_bleed": True,
            },
            "betamax": {
                "source_format": "betamax",
                "enable_vhs_restoration": True,
            },
            "hi8": {
                "source_format": "hi8",
                "enable_vhs_restoration": True,
            },
            "super8": {
                "source_format": "super8",
                "preserve_grain": True,
                "fix_flicker": True,
            },
            "16mm": {
                "source_format": "16mm",
                "preserve_grain": True,
            },
            "35mm": {
                "source_format": "35mm",
                "preserve_grain": True,
            },
            "nitrate": {
                "source_format": "nitrate",
                "preserve_grain": True,
                "max_denoise": 0.2,
            },
        }

    def interpret(self, command: ParsedCommand) -> RestorationPlan:
        """Interpret a parsed command into a restoration plan."""
        plan = RestorationPlan(
            original_command=command.raw_input,
            input_path=command.input_path,
            output_path=command.output_path,
            preset=command.quality_preset,
            preserve_authenticity=command.preserve_authenticity,
        )

        # Apply preset settings
        preset_config = self._preset_configs.get(command.quality_preset, {})

        # Apply scale factor
        if command.scale_factor:
            plan.scale_factor = command.scale_factor
        elif command.target_resolution and command.input_path:
            # Calculate scale from target resolution (would need to probe input)
            plan.scale_factor = preset_config.get("scale_factor", 2.0)
        else:
            plan.scale_factor = preset_config.get("scale_factor", 2.0)

        # Apply FPS
        plan.target_fps = command.target_fps

        # Apply era settings
        if command.source_era and command.source_era in self._era_configs:
            era_config = self._era_configs[command.source_era]
            plan.source_era = era_config.get("source_era")

            if era_config.get("preserve_grain", False):
                plan.authenticity_level = max(plan.authenticity_level, 0.7)

        # Apply format settings
        if command.source_format and command.source_format in self._format_configs:
            format_config = self._format_configs[command.source_format]
            plan.source_format = format_config.get("source_format")

        # Build pipeline stages based on intent and settings
        plan.stages = self._build_pipeline(command, preset_config)

        # Apply authenticity constraints
        if command.preserve_authenticity:
            self._apply_authenticity_constraints(plan, command)

        # Calculate resource estimates
        self._estimate_resources(plan)

        # Generate explanation
        plan.explanation = command.explanation
        plan.suggestions = command.suggestions

        # Add warnings
        plan.warnings = self._generate_warnings(command, plan)

        return plan

    def _build_pipeline(
        self,
        command: ParsedCommand,
        preset_config: Dict[str, Any]
    ) -> List[ProcessingStage]:
        """Build the processing pipeline stages."""
        stages = []
        order = 0

        # Stage 1: Input analysis (always)
        order += 1
        stages.append(ProcessingStage(
            name="analysis",
            processor="scene_intelligence",
            settings={"detect_content": True, "detect_era": True},
            order=order,
            description="Analyze video content and detect characteristics",
        ))

        # Stage 2: Format-specific preprocessing
        if command.source_format in ("vhs", "betamax", "hi8"):
            order += 1
            stages.append(ProcessingStage(
                name="vhs_restoration",
                processor="vhs_restorer",
                settings={
                    "remove_tracking": "tracking" in command.fix_issues,
                    "remove_dropout": "dropout" in command.fix_issues,
                    "fix_chroma": "color_bleed" in command.fix_issues,
                },
                order=order,
                description="Fix VHS-specific artifacts",
            ))

        # Stage 3: Stabilization
        if (command.intent == CommandIntent.STABILIZE or
            "shake" in command.fix_issues or
            "jitter" in command.fix_issues):
            order += 1
            stages.append(ProcessingStage(
                name="stabilization",
                processor="stabilizer",
                settings={"strength": 0.8},
                order=order,
                description="Stabilize shaky footage",
            ))

        # Stage 4: Denoising
        if (command.intent == CommandIntent.DENOISE or
            "noise" in command.fix_issues or
            "grain" in command.fix_issues or
            command.quality_preset in ("quality", "ultimate")):
            order += 1

            denoise_strength = preset_config.get("denoise_strength", 0.5)
            if "grain" not in command.preserve_aspects:
                denoise_strength = min(denoise_strength, 0.4)

            stages.append(ProcessingStage(
                name="denoise",
                processor="tap_denoise" if command.quality_preset == "ultimate" else "denoiser",
                settings={
                    "strength": denoise_strength,
                    "preserve_grain": "grain" in command.preserve_aspects,
                },
                order=order,
                description="Reduce noise and clean up image",
            ))

        # Stage 5: Artifact removal
        if ("artifacts" in command.fix_issues or
            "scratches" in command.fix_issues or
            "damage" in command.fix_issues):
            order += 1
            stages.append(ProcessingStage(
                name="artifact_removal",
                processor="defect_repair",
                settings={
                    "fix_scratches": "scratches" in command.fix_issues,
                    "fix_damage": "damage" in command.fix_issues,
                },
                order=order,
                description="Remove scratches and damage",
            ))

        # Stage 6: Upscaling
        if (command.intent == CommandIntent.UPSCALE or
            command.scale_factor and command.scale_factor > 1 or
            command.target_resolution):
            order += 1
            stages.append(ProcessingStage(
                name="upscale",
                processor=preset_config.get("model", "realesrgan"),
                settings={
                    "scale_factor": command.scale_factor or preset_config.get("scale_factor", 2),
                    "model": preset_config.get("model", "realesrgan"),
                },
                order=order,
                description=f"Upscale video {command.scale_factor or 2}x",
            ))

        # Stage 7: Face enhancement
        if (preset_config.get("face_restore", False) or
            command.quality_preset in ("balanced", "quality", "ultimate")):
            order += 1
            stages.append(ProcessingStage(
                name="face_enhance",
                processor=preset_config.get("face_model", "gfpgan"),
                settings={
                    "strength": 0.7 if command.preserve_authenticity else 1.0,
                },
                order=order,
                description="Enhance and restore faces",
            ))

        # Stage 8: Frame interpolation
        if (command.intent == CommandIntent.INTERPOLATE or
            command.target_fps):
            order += 1
            stages.append(ProcessingStage(
                name="interpolation",
                processor="rife",
                settings={
                    "target_fps": command.target_fps or 60,
                },
                order=order,
                description=f"Interpolate to {command.target_fps or 60}fps",
            ))

        # Stage 9: Colorization
        if command.intent == CommandIntent.COLORIZE:
            order += 1
            stages.append(ProcessingStage(
                name="colorize",
                processor="ddcolor",
                settings={
                    "preserve_bw_option": True,
                },
                order=order,
                description="Colorize black and white footage",
            ))

        # Stage 10: Temporal consistency
        if preset_config.get("temporal_consistency", False):
            order += 1
            stages.append(ProcessingStage(
                name="temporal",
                processor="cross_attention_temporal",
                settings={
                    "method": preset_config.get("temporal_method", "optical_flow"),
                },
                order=order,
                description="Apply temporal consistency",
            ))

        # Stage 11: Color correction
        order += 1
        stages.append(ProcessingStage(
            name="color_correction",
            processor="auto_enhance",
            settings={
                "auto_white_balance": True,
                "auto_exposure": True,
                "preserve_look": command.preserve_authenticity,
            },
            order=order,
            description="Auto color correction",
        ))

        # Stage 12: Final sharpening
        if preset_config.get("sharpen_strength", 0) > 0:
            order += 1
            stages.append(ProcessingStage(
                name="sharpen",
                processor="sharpener",
                settings={
                    "strength": preset_config.get("sharpen_strength", 0.5),
                },
                order=order,
                description="Apply final sharpening",
            ))

        return stages

    def _apply_authenticity_constraints(
        self,
        plan: RestorationPlan,
        command: ParsedCommand
    ) -> None:
        """Apply authenticity constraints to limit over-processing."""
        if not command.preserve_authenticity:
            return

        # Reduce processing strength for authentic restoration
        for stage in plan.stages:
            if stage.name == "denoise":
                current = stage.settings.get("strength", 0.5)
                stage.settings["strength"] = min(current, 0.4)
                stage.settings["preserve_grain"] = True

            if stage.name == "face_enhance":
                current = stage.settings.get("strength", 1.0)
                stage.settings["strength"] = min(current, 0.6)

            if stage.name == "sharpen":
                current = stage.settings.get("strength", 0.5)
                stage.settings["strength"] = min(current, 0.3)

            if stage.name == "color_correction":
                stage.settings["preserve_look"] = True
                stage.settings["max_correction"] = 0.3

    def _estimate_resources(self, plan: RestorationPlan) -> None:
        """Estimate VRAM and time requirements."""
        vram = 2.0  # Base VRAM

        for stage in plan.stages:
            if stage.enabled:
                if stage.processor in ("realesrgan", "basicvsr"):
                    vram += 4.0
                elif stage.processor == "diffusion":
                    vram += 12.0
                elif stage.processor in ("gfpgan", "codeformer"):
                    vram += 2.0
                elif stage.processor == "rife":
                    vram += 3.0
                elif stage.processor == "tap_denoise":
                    vram += 6.0

        plan.estimated_vram_gb = vram

        # Estimate time (very rough)
        time_per_frame = 0.1  # Base
        for stage in plan.stages:
            if stage.enabled:
                if stage.processor == "diffusion":
                    time_per_frame += 5.0
                elif stage.processor in ("realesrgan", "basicvsr"):
                    time_per_frame += 0.3
                elif stage.processor in ("gfpgan", "codeformer"):
                    time_per_frame += 0.1
                elif stage.processor == "rife":
                    time_per_frame += 0.2

        plan.estimated_time_per_frame = time_per_frame

    def _generate_warnings(
        self,
        command: ParsedCommand,
        plan: RestorationPlan
    ) -> List[str]:
        """Generate warnings about the plan."""
        warnings = []

        if plan.estimated_vram_gb > 24:
            warnings.append(
                f"This plan requires approximately {plan.estimated_vram_gb:.1f}GB VRAM. "
                "Consider a lower quality preset if you have less than 32GB VRAM."
            )

        if plan.scale_factor > 4:
            warnings.append(
                f"Upscaling {plan.scale_factor}x may introduce artifacts. "
                "Consider 4x maximum for best quality."
            )

        if "grain" in command.fix_issues and "grain" in command.preserve_aspects:
            warnings.append(
                "You requested both removing and preserving grain. "
                "Will reduce grain while maintaining some film character."
            )

        if command.intent == CommandIntent.COLORIZE and not command.preserve_authenticity:
            warnings.append(
                "AI colorization works best when preserving some original character. "
                "Consider enabling authenticity preservation for more natural colors."
            )

        return warnings


def interpret_command(text: str) -> RestorationPlan:
    """Quick helper to interpret a natural language command."""
    from .parser import NLPCommandParser

    parser = NLPCommandParser()
    interpreter = CommandInterpreter()

    command = parser.parse(text)
    plan = interpreter.interpret(command)

    return plan
