"""Preset generator for automatic configuration based on video analysis."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .analyzer import (
    VideoCharacteristics,
    VideoEra,
    VideoSource,
    DefectType,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedPreset:
    """A generated preset with configuration and reasoning."""
    name: str
    description: str
    config: Dict[str, Any]

    # Reasoning for each setting
    reasoning: Dict[str, str] = field(default_factory=dict)

    # Estimated resource usage
    estimated_vram_gb: float = 0.0
    estimated_time_per_frame_ms: float = 0.0

    # Quality expectations
    expected_psnr_gain: float = 0.0
    expected_ssim_gain: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "reasoning": self.reasoning,
            "estimated_vram_gb": self.estimated_vram_gb,
            "estimated_time_per_frame_ms": self.estimated_time_per_frame_ms,
            "expected_quality_improvement": {
                "psnr_gain_db": self.expected_psnr_gain,
                "ssim_gain": self.expected_ssim_gain,
            },
            "warnings": self.warnings,
        }


class PresetGenerator:
    """Generates optimal presets based on video characteristics."""

    def __init__(
        self,
        available_vram_gb: float = 8.0,
        target_quality: str = "balanced",  # fast, balanced, quality, ultimate
        preserve_authenticity: bool = True,
    ):
        self.available_vram_gb = available_vram_gb
        self.target_quality = target_quality
        self.preserve_authenticity = preserve_authenticity

    def generate(self, chars: VideoCharacteristics) -> GeneratedPreset:
        """Generate optimal preset for video characteristics."""
        preset = GeneratedPreset(
            name=self._generate_name(chars),
            description=self._generate_description(chars),
            config={},
        )

        # Build configuration
        self._add_base_config(preset, chars)
        self._add_denoise_config(preset, chars)
        self._add_upscale_config(preset, chars)
        self._add_face_config(preset, chars)
        self._add_temporal_config(preset, chars)
        self._add_defect_repair_config(preset, chars)
        self._add_color_config(preset, chars)
        self._add_output_config(preset, chars)

        # Add authenticity constraints
        if self.preserve_authenticity:
            self._add_authenticity_constraints(preset, chars)

        # Estimate resources
        self._estimate_resources(preset, chars)

        # Validate against available VRAM
        self._validate_and_adjust(preset, chars)

        return preset

    def _generate_name(self, chars: VideoCharacteristics) -> str:
        """Generate a descriptive preset name."""
        parts = []

        # Era
        era_names = {
            VideoEra.SILENT_FILM: "silent",
            VideoEra.EARLY_SOUND: "classic",
            VideoEra.GOLDEN_AGE: "vintage",
            VideoEra.NEW_HOLLYWOOD: "retro",
            VideoEra.HOME_VIDEO: "vhs",
            VideoEra.DIGITAL_EARLY: "earlydigital",
            VideoEra.MODERN: "modern",
        }
        parts.append(era_names.get(chars.era, "auto"))

        # Quality tier
        parts.append(self.target_quality)

        # Resolution
        if chars.width <= 480:
            parts.append("sd")
        elif chars.width <= 720:
            parts.append("dvd")
        elif chars.width <= 1920:
            parts.append("hd")
        else:
            parts.append("uhd")

        return "_".join(parts)

    def _generate_description(self, chars: VideoCharacteristics) -> str:
        """Generate description of the preset."""
        desc_parts = []

        desc_parts.append(f"Optimized for {chars.era.value.replace('_', ' ')} era")
        desc_parts.append(f"{chars.width}x{chars.height} {chars.source.value.replace('_', ' ')} source")

        if chars.defects:
            defect_names = [d.name.replace('_', ' ').lower() for d in chars.defects[:3]]
            desc_parts.append(f"addresses {', '.join(defect_names)}")

        return ". ".join(desc_parts) + "."

    def _add_base_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Add base configuration."""
        config = preset.config

        # Quality tiers affect model selection
        quality_settings = {
            "fast": {"models": "lightweight", "passes": 1},
            "balanced": {"models": "standard", "passes": 1},
            "quality": {"models": "full", "passes": 2},
            "ultimate": {"models": "best", "passes": 3},
        }

        tier = quality_settings.get(self.target_quality, quality_settings["balanced"])

        config["quality_tier"] = self.target_quality
        config["processing_passes"] = tier["passes"]

        preset.reasoning["quality_tier"] = f"Using {self.target_quality} tier for {tier['models']} models"

    def _add_denoise_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure denoising based on detected noise."""
        config = preset.config

        noise_level = chars.noise_level
        has_grain = chars.has_film_grain

        if noise_level < 10 and not has_grain:
            config["enable_denoise"] = False
            preset.reasoning["denoise"] = "Low noise level, denoising skipped"
            return

        config["enable_denoise"] = True

        # Choose denoiser based on noise type
        if has_grain and self.preserve_authenticity:
            config["denoise_method"] = "tap"
            config["tap_preserve_grain"] = True
            config["denoise_strength"] = min(0.7, noise_level / 100)
            preset.reasoning["denoise"] = "TAP denoising with grain preservation for film source"
        elif noise_level > 50:
            config["denoise_method"] = "restormer"
            config["denoise_strength"] = min(1.0, noise_level / 80)
            preset.reasoning["denoise"] = f"Strong Restormer denoising for high noise ({noise_level:.0f}%)"
        else:
            config["denoise_method"] = "nafnet"
            config["denoise_strength"] = noise_level / 100
            preset.reasoning["denoise"] = f"NAFNet denoising for moderate noise ({noise_level:.0f}%)"

    def _add_upscale_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure upscaling."""
        config = preset.config

        current_width = chars.width
        current_height = chars.height

        # Determine target resolution
        if current_width < 720:
            target_scale = 4 if self.target_quality in ("quality", "ultimate") else 2
        elif current_width < 1280:
            target_scale = 2
        elif current_width < 1920:
            target_scale = 2 if self.target_quality in ("quality", "ultimate") else 1
        else:
            target_scale = 1

        if target_scale == 1:
            config["enable_upscale"] = False
            preset.reasoning["upscale"] = "Source already HD+, upscaling disabled"
            return

        config["enable_upscale"] = True
        config["scale_factor"] = target_scale

        # Choose upscaler based on quality tier
        if self.target_quality == "ultimate" and self.available_vram_gb >= 12:
            config["sr_model"] = "diffusion"
            config["diffusion_steps"] = 20
            preset.reasoning["upscale"] = f"Diffusion SR ({target_scale}x) for maximum quality"
        elif self.target_quality in ("quality", "ultimate"):
            config["sr_model"] = "realesrgan"
            preset.reasoning["upscale"] = f"Real-ESRGAN ({target_scale}x) for high quality"
        else:
            config["sr_model"] = "realesrgan"
            config["sr_model_variant"] = "fast"
            preset.reasoning["upscale"] = f"Real-ESRGAN fast ({target_scale}x) for balanced quality/speed"

    def _add_face_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure face restoration."""
        config = preset.config

        if not chars.has_faces or chars.face_count_avg < 0.1:
            config["enable_face_restore"] = False
            preset.reasoning["face_restore"] = "No faces detected, face restoration disabled"
            return

        config["enable_face_restore"] = True
        config["face_detection_threshold"] = 0.5

        # Choose face model based on quality tier
        if self.target_quality == "ultimate":
            config["face_model"] = "aesrgan"
            preset.reasoning["face_restore"] = "AESRGAN for natural face restoration"
        elif self.target_quality == "quality":
            config["face_model"] = "codeformer"
            config["codeformer_fidelity"] = 0.7
            preset.reasoning["face_restore"] = "CodeFormer with balanced fidelity"
        else:
            config["face_model"] = "gfpgan"
            preset.reasoning["face_restore"] = "GFPGAN for efficient face restoration"

        # Adjust for era
        if chars.era in (VideoEra.SILENT_FILM, VideoEra.EARLY_SOUND) and self.preserve_authenticity:
            config["face_restore_strength"] = 0.5
            preset.reasoning["face_restore"] += " (reduced strength for era authenticity)"

    def _add_temporal_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure temporal processing."""
        config = preset.config

        # Frame interpolation
        if chars.fps < 24:
            if self.target_quality in ("quality", "ultimate"):
                config["enable_interpolation"] = True
                config["target_fps"] = 24
                config["interpolation_model"] = "rife"
                preset.reasoning["interpolation"] = f"RIFE interpolation {chars.fps:.1f} -> 24 fps"
            else:
                config["enable_interpolation"] = False
                preset.reasoning["interpolation"] = "Low fps but skipped for speed"
        else:
            config["enable_interpolation"] = False

        # Temporal consistency
        if self.target_quality == "ultimate":
            config["temporal_method"] = "cross_attention"
            preset.reasoning["temporal"] = "Cross-attention for flicker-free output"
        elif self.target_quality == "quality":
            config["temporal_method"] = "optical_flow"
            preset.reasoning["temporal"] = "Optical flow temporal consistency"
        else:
            config["temporal_method"] = "blend"
            preset.reasoning["temporal"] = "Simple blending for temporal smoothness"

        # Deinterlacing
        if chars.is_interlaced:
            config["enable_deinterlace"] = True
            config["deinterlace_method"] = "yadif"
            preset.reasoning["deinterlace"] = f"YADIF deinterlacing ({chars.field_order})"

    def _add_defect_repair_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure defect-specific repairs."""
        config = preset.config

        # Compression artifacts
        if DefectType.COMPRESSION_ARTIFACTS in chars.defects:
            severity = chars.defect_severity.get(DefectType.COMPRESSION_ARTIFACTS, 0.5)
            config["enable_qp_artifact_removal"] = True
            config["qp_strength"] = severity
            preset.reasoning["qp_artifact"] = f"QP artifact removal (severity: {severity:.1%})"

        # Scratches
        if DefectType.SCRATCHES in chars.defects:
            config["enable_scratch_removal"] = True
            config["scratch_detection_sensitivity"] = 0.7
            preset.reasoning["scratch"] = "Automatic scratch detection and removal"

        # VHS artifacts
        if DefectType.HEAD_SWITCHING in chars.defects or chars.has_vhs_artifacts:
            config["enable_vhs_restoration"] = True
            config["vhs_head_switch_repair"] = True
            config["vhs_tracking_repair"] = True
            preset.reasoning["vhs"] = "VHS-specific artifact restoration"

        # Stabilization
        if DefectType.JITTER in chars.defects:
            config["enable_stabilization"] = True
            config["stabilization_strength"] = 0.7
            preset.reasoning["stabilization"] = "Video stabilization for jitter"

    def _add_color_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure color processing."""
        config = preset.config

        if chars.is_grayscale:
            # Don't auto-colorize unless explicitly requested
            config["enable_colorize"] = False
            preset.reasoning["color"] = "Grayscale source, colorization available on request"
        else:
            # Color correction
            if DefectType.COLOR_FADING in chars.defects:
                config["enable_color_correction"] = True
                config["color_correction_strength"] = 0.5
                preset.reasoning["color"] = "Color correction for fading"

    def _add_output_config(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Configure output encoding."""
        config = preset.config

        # CRF based on quality tier
        crf_map = {"fast": 23, "balanced": 18, "quality": 16, "ultimate": 14}
        config["crf"] = crf_map.get(self.target_quality, 18)

        # Encoder preset
        preset_map = {"fast": "medium", "balanced": "slow", "quality": "slower", "ultimate": "veryslow"}
        config["encoder_preset"] = preset_map.get(self.target_quality, "slow")

        # Codec selection
        config["video_codec"] = "libx265" if self.target_quality in ("quality", "ultimate") else "libx264"

        # HDR for quality outputs
        if self.target_quality == "ultimate" and chars.bit_depth >= 10:
            config["enable_hdr"] = True

        preset.reasoning["output"] = f"{config['video_codec']} CRF {config['crf']} {config['encoder_preset']}"

    def _add_authenticity_constraints(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Add constraints to preserve era authenticity."""
        config = preset.config

        # Limit processing based on era
        era_limits = {
            VideoEra.SILENT_FILM: {
                "max_upscale": 2,
                "preserve_grain": True,
                "face_restore_strength": 0.3,
                "denoise_strength_cap": 0.5,
            },
            VideoEra.EARLY_SOUND: {
                "max_upscale": 2,
                "preserve_grain": True,
                "face_restore_strength": 0.4,
                "denoise_strength_cap": 0.6,
            },
            VideoEra.GOLDEN_AGE: {
                "max_upscale": 4,
                "preserve_grain": True,
                "face_restore_strength": 0.5,
                "denoise_strength_cap": 0.7,
            },
            VideoEra.HOME_VIDEO: {
                "max_upscale": 4,
                "preserve_grain": False,
                "face_restore_strength": 0.7,
            },
        }

        limits = era_limits.get(chars.era, {})

        if limits:
            # Apply upscale limit
            if "max_upscale" in limits and config.get("scale_factor", 1) > limits["max_upscale"]:
                config["scale_factor"] = limits["max_upscale"]
                preset.warnings.append(f"Upscale limited to {limits['max_upscale']}x for era authenticity")

            # Apply grain preservation
            if limits.get("preserve_grain") and config.get("enable_denoise"):
                config["denoise_preserve_grain"] = True
                if "denoise_strength_cap" in limits:
                    if config.get("denoise_strength", 0) > limits["denoise_strength_cap"]:
                        config["denoise_strength"] = limits["denoise_strength_cap"]

            # Apply face restoration limit
            if "face_restore_strength" in limits:
                config["face_restore_strength"] = limits["face_restore_strength"]

        config["preserve_authenticity"] = True
        config["detected_era"] = chars.era.value

    def _estimate_resources(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Estimate resource requirements."""
        config = preset.config

        vram = 0.5  # Base overhead
        time_per_frame = 50  # Base ms

        # Denoise
        if config.get("enable_denoise"):
            method = config.get("denoise_method", "nafnet")
            vram += {"restormer": 6, "nafnet": 2, "tap": 6}.get(method, 2)
            time_per_frame += {"restormer": 200, "nafnet": 50, "tap": 150}.get(method, 50)

        # Upscale
        if config.get("enable_upscale"):
            model = config.get("sr_model", "realesrgan")
            scale = config.get("scale_factor", 2)
            vram += {"diffusion": 12, "realesrgan": 3, "basicvsr": 8}.get(model, 3) * (scale / 2)
            time_per_frame += {"diffusion": 5000, "realesrgan": 100, "basicvsr": 200}.get(model, 100)

        # Face restoration
        if config.get("enable_face_restore"):
            model = config.get("face_model", "gfpgan")
            vram += {"aesrgan": 2, "codeformer": 2, "gfpgan": 1.5}.get(model, 1.5)
            time_per_frame += 100 * chars.face_count_avg

        # Interpolation
        if config.get("enable_interpolation"):
            vram += 3
            time_per_frame += 150

        preset.estimated_vram_gb = vram
        preset.estimated_time_per_frame_ms = time_per_frame

        # Quality estimates based on defects and processing
        psnr_gain = 0
        ssim_gain = 0

        if config.get("enable_denoise"):
            psnr_gain += 2 * config.get("denoise_strength", 0.5)
            ssim_gain += 0.02

        if config.get("enable_qp_artifact_removal"):
            psnr_gain += 1.5
            ssim_gain += 0.01

        preset.expected_psnr_gain = psnr_gain
        preset.expected_ssim_gain = ssim_gain

    def _validate_and_adjust(self, preset: GeneratedPreset, chars: VideoCharacteristics) -> None:
        """Validate preset against available resources and adjust if needed."""
        config = preset.config

        if preset.estimated_vram_gb > self.available_vram_gb:
            # Need to reduce VRAM usage
            preset.warnings.append(
                f"Estimated VRAM ({preset.estimated_vram_gb:.1f}GB) exceeds available "
                f"({self.available_vram_gb:.1f}GB). Adjusting settings."
            )

            # Prioritized reductions
            reductions = [
                ("diffusion SR", lambda: config.get("sr_model") == "diffusion",
                 lambda: config.update({"sr_model": "realesrgan"})),
                ("cross-attention temporal", lambda: config.get("temporal_method") == "cross_attention",
                 lambda: config.update({"temporal_method": "optical_flow"})),
                ("face restoration", lambda: config.get("enable_face_restore"),
                 lambda: config.update({"enable_face_restore": False})),
                ("interpolation", lambda: config.get("enable_interpolation"),
                 lambda: config.update({"enable_interpolation": False})),
            ]

            for name, check, apply in reductions:
                if check():
                    apply()
                    logger.info(f"Disabled {name} to fit VRAM budget")
                    # Recalculate
                    self._estimate_resources(preset, chars)
                    if preset.estimated_vram_gb <= self.available_vram_gb:
                        break
