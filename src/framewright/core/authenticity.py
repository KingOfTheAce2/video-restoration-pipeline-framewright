"""Authenticity Preservation System for FrameWright.

Core philosophy: Restore, don't modernize. Make historical footage accessible
while preserving its authentic character and historical value.

This module ensures that restoration enhances accessibility without
over-processing footage to look artificially modern.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json

logger = logging.getLogger(__name__)


class Era(Enum):
    """Historical era classification."""
    SILENT_FILM = "silent_film"          # Pre-1930
    EARLY_TALKIES = "early_talkies"      # 1930-1940
    GOLDEN_AGE = "golden_age"            # 1940-1960
    NEW_HOLLYWOOD = "new_hollywood"      # 1960-1980
    VIDEO_ERA = "video_era"              # 1980-2000
    DIGITAL_TRANSITION = "digital_transition"  # 2000-2010
    MODERN = "modern"                    # 2010+


class SourceMedium(Enum):
    """Original recording medium."""
    NITRATE_FILM = "nitrate"             # Pre-1950
    ACETATE_FILM = "acetate"             # 1950s-1980s
    POLYESTER_FILM = "polyester"         # 1980s+
    VHS = "vhs"
    BETAMAX = "betamax"
    UMATIC = "umatic"
    BETACAM = "betacam"
    DIGITAL_TAPE = "digital_tape"        # DV, HDV
    DIGITAL_FILE = "digital_file"


class RestorationPhilosophy(Enum):
    """Restoration approach philosophy."""
    ARCHIVAL = "archival"                # Maximum authenticity, minimal processing
    ACCESSIBLE = "accessible"            # Balance authenticity with watchability
    ENHANCED = "enhanced"                # More aggressive enhancement
    PRESENTATION = "presentation"        # For theatrical/broadcast presentation


@dataclass
class AuthenticityProfile:
    """Profile defining what makes footage authentic to its era."""
    era: Era
    source_medium: SourceMedium

    # Characteristics to PRESERVE (not remove)
    preserve_grain: bool = True
    preserve_flicker_character: bool = False  # Subtle period flicker
    preserve_vignetting: bool = False
    preserve_aspect_ratio: bool = True
    preserve_frame_rate_feel: bool = True     # Don't over-interpolate
    preserve_color_palette: bool = True       # Era-appropriate colors
    preserve_dynamic_range: bool = True       # Don't expand to modern HDR

    # Artifacts to REMOVE (degradation, not character)
    remove_physical_damage: bool = True       # Scratches, tears, dust
    remove_chemical_degradation: bool = True  # Color fading, vinegar syndrome
    remove_electronic_artifacts: bool = True  # Dropout, tracking errors
    remove_compression_artifacts: bool = True # Modern encoding damage

    # Enhancement LIMITS (prevent over-processing)
    max_sharpening: float = 0.3              # 0-1, limit sharpening
    max_noise_reduction: float = 0.5         # 0-1, preserve some texture
    max_color_correction: float = 0.4        # 0-1, gentle color work
    max_upscale_factor: int = 4              # Don't upscale beyond recognition
    max_interpolation_factor: float = 1.5    # Limit frame rate increase
    face_enhancement_strength: float = 0.5   # Subtle, not plastic

    # Target quality (don't exceed era-appropriate quality)
    target_grain_level: float = 0.3          # Maintain some grain
    target_softness: float = 0.2             # Don't over-sharpen

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "era": self.era.value,
            "source_medium": self.source_medium.value,
            "preserve_grain": self.preserve_grain,
            "preserve_flicker_character": self.preserve_flicker_character,
            "preserve_vignetting": self.preserve_vignetting,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "preserve_frame_rate_feel": self.preserve_frame_rate_feel,
            "preserve_color_palette": self.preserve_color_palette,
            "preserve_dynamic_range": self.preserve_dynamic_range,
            "remove_physical_damage": self.remove_physical_damage,
            "remove_chemical_degradation": self.remove_chemical_degradation,
            "remove_electronic_artifacts": self.remove_electronic_artifacts,
            "remove_compression_artifacts": self.remove_compression_artifacts,
            "max_sharpening": self.max_sharpening,
            "max_noise_reduction": self.max_noise_reduction,
            "max_color_correction": self.max_color_correction,
            "max_upscale_factor": self.max_upscale_factor,
            "max_interpolation_factor": self.max_interpolation_factor,
            "face_enhancement_strength": self.face_enhancement_strength,
            "target_grain_level": self.target_grain_level,
            "target_softness": self.target_softness,
        }


# Predefined authenticity profiles for different eras
ERA_PROFILES: Dict[Era, AuthenticityProfile] = {
    Era.SILENT_FILM: AuthenticityProfile(
        era=Era.SILENT_FILM,
        source_medium=SourceMedium.NITRATE_FILM,
        preserve_grain=True,
        preserve_flicker_character=True,  # Subtle flicker is period-authentic
        preserve_vignetting=True,
        max_sharpening=0.2,
        max_noise_reduction=0.3,          # Keep texture
        max_color_correction=0.2,         # Respect original tinting
        max_interpolation_factor=1.3,     # Don't make it too smooth
        face_enhancement_strength=0.3,    # Very subtle
        target_grain_level=0.5,           # More grain expected
        target_softness=0.3,              # Period lenses were softer
    ),

    Era.EARLY_TALKIES: AuthenticityProfile(
        era=Era.EARLY_TALKIES,
        source_medium=SourceMedium.NITRATE_FILM,
        preserve_grain=True,
        preserve_flicker_character=False,
        max_sharpening=0.25,
        max_noise_reduction=0.35,
        max_color_correction=0.3,
        max_interpolation_factor=1.3,
        face_enhancement_strength=0.35,
        target_grain_level=0.45,
        target_softness=0.25,
    ),

    Era.GOLDEN_AGE: AuthenticityProfile(
        era=Era.GOLDEN_AGE,
        source_medium=SourceMedium.ACETATE_FILM,
        preserve_grain=True,
        preserve_color_palette=True,      # Technicolor/era colors
        max_sharpening=0.3,
        max_noise_reduction=0.4,
        max_color_correction=0.35,
        max_interpolation_factor=1.4,
        face_enhancement_strength=0.4,
        target_grain_level=0.35,
        target_softness=0.2,
    ),

    Era.NEW_HOLLYWOOD: AuthenticityProfile(
        era=Era.NEW_HOLLYWOOD,
        source_medium=SourceMedium.ACETATE_FILM,
        preserve_grain=True,
        max_sharpening=0.35,
        max_noise_reduction=0.45,
        max_color_correction=0.4,
        max_interpolation_factor=1.5,
        face_enhancement_strength=0.45,
        target_grain_level=0.3,
        target_softness=0.15,
    ),

    Era.VIDEO_ERA: AuthenticityProfile(
        era=Era.VIDEO_ERA,
        source_medium=SourceMedium.VHS,
        preserve_grain=False,             # Video noise is not "grain"
        max_sharpening=0.4,
        max_noise_reduction=0.6,          # More aggressive for video noise
        max_color_correction=0.5,
        max_interpolation_factor=2.0,     # Interlaced to progressive OK
        face_enhancement_strength=0.5,
        target_grain_level=0.1,
        target_softness=0.1,
    ),

    Era.DIGITAL_TRANSITION: AuthenticityProfile(
        era=Era.DIGITAL_TRANSITION,
        source_medium=SourceMedium.DIGITAL_TAPE,
        preserve_grain=False,
        max_sharpening=0.5,
        max_noise_reduction=0.7,
        max_color_correction=0.6,
        max_interpolation_factor=2.0,
        face_enhancement_strength=0.6,
        target_grain_level=0.05,
        target_softness=0.05,
    ),

    Era.MODERN: AuthenticityProfile(
        era=Era.MODERN,
        source_medium=SourceMedium.DIGITAL_FILE,
        preserve_grain=False,
        max_sharpening=0.7,
        max_noise_reduction=0.8,
        max_color_correction=0.8,
        max_interpolation_factor=2.5,
        face_enhancement_strength=0.7,
        target_grain_level=0.0,
        target_softness=0.0,
    ),
}


@dataclass
class AuthenticityGuard:
    """Guards against over-processing during restoration.

    Monitors processing parameters and clips them to era-appropriate limits.
    """
    profile: AuthenticityProfile
    philosophy: RestorationPhilosophy = RestorationPhilosophy.ACCESSIBLE

    # Multipliers based on philosophy
    _philosophy_multipliers: Dict[RestorationPhilosophy, float] = field(
        default_factory=lambda: {
            RestorationPhilosophy.ARCHIVAL: 0.5,      # Half the limits
            RestorationPhilosophy.ACCESSIBLE: 1.0,   # Standard limits
            RestorationPhilosophy.ENHANCED: 1.5,     # 50% more
            RestorationPhilosophy.PRESENTATION: 1.3, # 30% more
        }
    )

    def get_multiplier(self) -> float:
        """Get the philosophy-based multiplier."""
        return self._philosophy_multipliers.get(self.philosophy, 1.0)

    def limit_sharpening(self, requested: float) -> float:
        """Limit sharpening to era-appropriate level."""
        max_allowed = self.profile.max_sharpening * self.get_multiplier()
        limited = min(requested, max_allowed)
        if limited < requested:
            logger.debug(f"Sharpening limited: {requested:.2f} -> {limited:.2f}")
        return limited

    def limit_noise_reduction(self, requested: float) -> float:
        """Limit noise reduction to preserve texture."""
        max_allowed = self.profile.max_noise_reduction * self.get_multiplier()
        limited = min(requested, max_allowed)
        if limited < requested:
            logger.debug(f"Noise reduction limited: {requested:.2f} -> {limited:.2f}")
        return limited

    def limit_color_correction(self, requested: float) -> float:
        """Limit color correction to preserve period palette."""
        max_allowed = self.profile.max_color_correction * self.get_multiplier()
        limited = min(requested, max_allowed)
        if limited < requested:
            logger.debug(f"Color correction limited: {requested:.2f} -> {limited:.2f}")
        return limited

    def limit_upscale(self, requested: int) -> int:
        """Limit upscale factor."""
        max_allowed = self.profile.max_upscale_factor
        limited = min(requested, max_allowed)
        if limited < requested:
            logger.debug(f"Upscale limited: {requested}x -> {limited}x")
        return limited

    def limit_interpolation(self, source_fps: float, target_fps: float) -> float:
        """Limit frame rate interpolation."""
        requested_factor = target_fps / source_fps if source_fps > 0 else 1.0
        max_factor = self.profile.max_interpolation_factor * self.get_multiplier()

        if requested_factor > max_factor:
            limited_fps = source_fps * max_factor
            logger.debug(f"Interpolation limited: {target_fps:.1f} -> {limited_fps:.1f} fps")
            return limited_fps
        return target_fps

    def limit_face_enhancement(self, requested: float) -> float:
        """Limit face enhancement to avoid plastic look."""
        max_allowed = self.profile.face_enhancement_strength * self.get_multiplier()
        limited = min(requested, max_allowed)
        if limited < requested:
            logger.debug(f"Face enhancement limited: {requested:.2f} -> {limited:.2f}")
        return limited

    def should_preserve_grain(self) -> bool:
        """Check if grain should be preserved."""
        return self.profile.preserve_grain

    def get_target_grain_level(self) -> float:
        """Get target grain level after processing."""
        return self.profile.target_grain_level

    def should_remove_artifact(self, artifact_type: str) -> bool:
        """Check if an artifact type should be removed."""
        artifact_map = {
            "scratch": self.profile.remove_physical_damage,
            "dust": self.profile.remove_physical_damage,
            "tear": self.profile.remove_physical_damage,
            "color_fade": self.profile.remove_chemical_degradation,
            "vinegar_syndrome": self.profile.remove_chemical_degradation,
            "dropout": self.profile.remove_electronic_artifacts,
            "tracking": self.profile.remove_electronic_artifacts,
            "head_switching": self.profile.remove_electronic_artifacts,
            "blocking": self.profile.remove_compression_artifacts,
            "ringing": self.profile.remove_compression_artifacts,
            "banding": self.profile.remove_compression_artifacts,
        }
        return artifact_map.get(artifact_type, True)

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust a restoration config to respect authenticity.

        Args:
            config: Restoration configuration dictionary

        Returns:
            Adjusted configuration with era-appropriate limits
        """
        adjusted = config.copy()

        # Limit enhancement parameters
        if "sharpening" in adjusted:
            adjusted["sharpening"] = self.limit_sharpening(adjusted["sharpening"])

        if "denoise_strength" in adjusted:
            adjusted["denoise_strength"] = self.limit_noise_reduction(
                adjusted["denoise_strength"]
            )

        if "color_correction_strength" in adjusted:
            adjusted["color_correction_strength"] = self.limit_color_correction(
                adjusted["color_correction_strength"]
            )

        if "scale_factor" in adjusted:
            adjusted["scale_factor"] = self.limit_upscale(adjusted["scale_factor"])

        if "target_fps" in adjusted and "source_fps" in adjusted:
            adjusted["target_fps"] = self.limit_interpolation(
                adjusted["source_fps"],
                adjusted["target_fps"]
            )

        if "face_enhancement" in adjusted:
            adjusted["face_enhancement"] = self.limit_face_enhancement(
                adjusted["face_enhancement"]
            )

        # Handle grain preservation
        if self.should_preserve_grain():
            adjusted["preserve_grain"] = True
            adjusted["target_grain_level"] = self.get_target_grain_level()

            # Reduce noise reduction if grain should be preserved
            if "denoise_strength" in adjusted:
                adjusted["denoise_strength"] *= 0.7

        return adjusted

    def get_recommendations(self) -> List[str]:
        """Get restoration recommendations based on profile."""
        recommendations = []

        if self.profile.era in [Era.SILENT_FILM, Era.EARLY_TALKIES]:
            recommendations.append(
                "Preserve period character: avoid over-sharpening and excessive noise reduction"
            )
            recommendations.append(
                "Consider keeping subtle flicker for authenticity"
            )

        if self.profile.preserve_grain:
            recommendations.append(
                f"Maintain grain level around {self.profile.target_grain_level:.0%} for period authenticity"
            )

        if self.profile.source_medium == SourceMedium.VHS:
            recommendations.append(
                "Focus on removing tape artifacts (dropout, tracking) while preserving era aesthetics"
            )

        if self.profile.preserve_color_palette:
            recommendations.append(
                "Preserve original color science - avoid modern color grading"
            )

        if self.profile.preserve_aspect_ratio:
            recommendations.append(
                "Maintain original aspect ratio - do not crop or stretch"
            )

        return recommendations


class EraDetector:
    """Automatically detects the era and source medium of footage."""

    def __init__(self):
        self._cv2 = None
        self._np = None

    def _ensure_deps(self) -> bool:
        """Ensure OpenCV is available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            return False

    def detect_era(
        self,
        video_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Era, SourceMedium, float]:
        """Detect era and source medium of footage.

        Args:
            video_path: Path to video file
            metadata: Optional metadata hints

        Returns:
            Tuple of (Era, SourceMedium, confidence)
        """
        indicators = {
            "aspect_ratio": None,
            "frame_rate": None,
            "is_color": None,
            "has_grain": None,
            "has_interlacing": None,
            "resolution": None,
            "sound_type": None,
            "edge_characteristics": None,
        }

        # Analyze video
        if self._ensure_deps():
            indicators.update(self._analyze_video(video_path))

        # Use metadata hints
        if metadata:
            if "year" in metadata:
                indicators["year"] = metadata["year"]
            if "format" in metadata:
                indicators["format_hint"] = metadata["format"]

        # Determine era
        era, era_confidence = self._classify_era(indicators)
        medium, medium_confidence = self._classify_medium(indicators, era)

        overall_confidence = (era_confidence + medium_confidence) / 2

        logger.info(
            f"Detected era: {era.value}, medium: {medium.value}, "
            f"confidence: {overall_confidence:.0%}"
        )

        return era, medium, overall_confidence

    def _analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """Analyze video characteristics."""
        cv2 = self._cv2
        np = self._np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {}

        result = {}

        # Basic properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result["frame_rate"] = fps
        result["resolution"] = (width, height)
        result["aspect_ratio"] = width / height if height > 0 else 1.33

        # Sample frames for analysis
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        if not frames:
            return result

        # Color analysis
        saturation_values = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation_values.append(np.mean(hsv[:, :, 1]))

        result["is_color"] = np.mean(saturation_values) > 25

        # Grain analysis
        noise_levels = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_levels.append(laplacian.std())

        result["has_grain"] = np.mean(noise_levels) > 8
        result["grain_intensity"] = np.mean(noise_levels)

        # Interlacing detection
        interlace_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Check for comb pattern
            diff_odd = np.abs(gray[::2, :].astype(float) - gray[1::2, :].astype(float))
            interlace_scores.append(np.mean(diff_odd))

        result["has_interlacing"] = np.mean(interlace_scores) > 20

        # Edge softness (older lenses are softer)
        sharpness_values = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_values.append(laplacian.var())

        result["edge_sharpness"] = np.mean(sharpness_values)

        return result

    def _classify_era(self, indicators: Dict[str, Any]) -> Tuple[Era, float]:
        """Classify footage era based on indicators."""
        scores = {era: 0.0 for era in Era}

        # Year-based classification (if available)
        if "year" in indicators:
            year = indicators["year"]
            if year < 1930:
                scores[Era.SILENT_FILM] += 5.0
            elif year < 1940:
                scores[Era.EARLY_TALKIES] += 5.0
            elif year < 1960:
                scores[Era.GOLDEN_AGE] += 5.0
            elif year < 1980:
                scores[Era.NEW_HOLLYWOOD] += 5.0
            elif year < 2000:
                scores[Era.VIDEO_ERA] += 5.0
            elif year < 2010:
                scores[Era.DIGITAL_TRANSITION] += 5.0
            else:
                scores[Era.MODERN] += 5.0

        # Color/B&W
        if indicators.get("is_color") is False:
            scores[Era.SILENT_FILM] += 2.0
            scores[Era.EARLY_TALKIES] += 1.5

        # Aspect ratio
        aspect = indicators.get("aspect_ratio", 1.33)
        if abs(aspect - 1.33) < 0.05:  # 4:3
            scores[Era.SILENT_FILM] += 1.0
            scores[Era.EARLY_TALKIES] += 1.0
            scores[Era.VIDEO_ERA] += 0.5
        elif abs(aspect - 1.37) < 0.05:  # Academy
            scores[Era.GOLDEN_AGE] += 1.0
        elif abs(aspect - 1.85) < 0.05 or abs(aspect - 2.35) < 0.1:
            scores[Era.NEW_HOLLYWOOD] += 1.0
            scores[Era.MODERN] += 0.5

        # Frame rate
        fps = indicators.get("frame_rate", 24)
        if fps and fps < 20:
            scores[Era.SILENT_FILM] += 2.0
        elif fps and abs(fps - 24) < 1:
            scores[Era.GOLDEN_AGE] += 0.5
            scores[Era.NEW_HOLLYWOOD] += 0.5

        # Interlacing suggests video era
        if indicators.get("has_interlacing"):
            scores[Era.VIDEO_ERA] += 2.0
            scores[Era.DIGITAL_TRANSITION] += 1.0

        # Grain characteristics
        if indicators.get("has_grain"):
            grain = indicators.get("grain_intensity", 10)
            if grain > 15:
                scores[Era.SILENT_FILM] += 1.0
                scores[Era.EARLY_TALKIES] += 0.5
            elif grain > 10:
                scores[Era.GOLDEN_AGE] += 0.5

        # Edge sharpness (modern is sharper)
        sharpness = indicators.get("edge_sharpness", 100)
        if sharpness < 50:
            scores[Era.SILENT_FILM] += 1.0
        elif sharpness > 200:
            scores[Era.MODERN] += 1.0

        # Find best match
        best_era = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_era] / total_score if total_score > 0 else 0.5

        return best_era, confidence

    def _classify_medium(
        self,
        indicators: Dict[str, Any],
        era: Era,
    ) -> Tuple[SourceMedium, float]:
        """Classify source medium based on indicators and era."""
        # Default medium based on era
        era_defaults = {
            Era.SILENT_FILM: SourceMedium.NITRATE_FILM,
            Era.EARLY_TALKIES: SourceMedium.NITRATE_FILM,
            Era.GOLDEN_AGE: SourceMedium.ACETATE_FILM,
            Era.NEW_HOLLYWOOD: SourceMedium.ACETATE_FILM,
            Era.VIDEO_ERA: SourceMedium.VHS,
            Era.DIGITAL_TRANSITION: SourceMedium.DIGITAL_TAPE,
            Era.MODERN: SourceMedium.DIGITAL_FILE,
        }

        medium = era_defaults.get(era, SourceMedium.DIGITAL_FILE)
        confidence = 0.7

        # Refine based on indicators
        if indicators.get("has_interlacing"):
            if era == Era.VIDEO_ERA:
                # Check resolution for VHS vs Betacam
                res = indicators.get("resolution", (640, 480))
                if res[0] < 500:
                    medium = SourceMedium.VHS
                else:
                    medium = SourceMedium.BETACAM
                confidence = 0.8

        if indicators.get("format_hint"):
            hint = indicators["format_hint"].lower()
            if "vhs" in hint:
                medium = SourceMedium.VHS
                confidence = 0.95
            elif "betamax" in hint:
                medium = SourceMedium.BETAMAX
                confidence = 0.95
            elif "film" in hint:
                if era in [Era.SILENT_FILM, Era.EARLY_TALKIES]:
                    medium = SourceMedium.NITRATE_FILM
                else:
                    medium = SourceMedium.ACETATE_FILM
                confidence = 0.9

        return medium, confidence


class AuthenticityManager:
    """Main manager for authenticity preservation.

    Provides the interface between the restoration pipeline and
    authenticity constraints.
    """

    def __init__(
        self,
        philosophy: RestorationPhilosophy = RestorationPhilosophy.ACCESSIBLE,
        custom_profile: Optional[AuthenticityProfile] = None,
    ):
        """Initialize authenticity manager.

        Args:
            philosophy: Overall restoration philosophy
            custom_profile: Custom profile (auto-detected if None)
        """
        self.philosophy = philosophy
        self.custom_profile = custom_profile
        self.detector = EraDetector()
        self._current_guard: Optional[AuthenticityGuard] = None

    def analyze_and_configure(
        self,
        video_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuthenticityGuard:
        """Analyze video and create appropriate authenticity guard.

        Args:
            video_path: Path to video file
            metadata: Optional metadata hints

        Returns:
            Configured AuthenticityGuard
        """
        if self.custom_profile:
            profile = self.custom_profile
        else:
            era, medium, confidence = self.detector.detect_era(video_path, metadata)

            # Get base profile for era
            profile = ERA_PROFILES.get(era, ERA_PROFILES[Era.MODERN])

            # Adjust for detected medium
            profile = AuthenticityProfile(
                era=era,
                source_medium=medium,
                **{k: v for k, v in profile.to_dict().items()
                   if k not in ["era", "source_medium"]}
            )

        self._current_guard = AuthenticityGuard(
            profile=profile,
            philosophy=self.philosophy,
        )

        return self._current_guard

    def get_guard(self) -> Optional[AuthenticityGuard]:
        """Get current authenticity guard."""
        return self._current_guard

    def validate_restoration_config(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and adjust restoration configuration.

        Args:
            config: Restoration configuration

        Returns:
            Adjusted configuration respecting authenticity
        """
        if self._current_guard is None:
            logger.warning("No authenticity guard configured, using defaults")
            return config

        return self._current_guard.validate_config(config)

    def get_processing_limits(self) -> Dict[str, float]:
        """Get current processing limits."""
        if self._current_guard is None:
            return {}

        profile = self._current_guard.profile
        mult = self._current_guard.get_multiplier()

        return {
            "max_sharpening": profile.max_sharpening * mult,
            "max_noise_reduction": profile.max_noise_reduction * mult,
            "max_color_correction": profile.max_color_correction * mult,
            "max_upscale_factor": profile.max_upscale_factor,
            "max_interpolation_factor": profile.max_interpolation_factor * mult,
            "max_face_enhancement": profile.face_enhancement_strength * mult,
            "target_grain_level": profile.target_grain_level,
        }

    def get_era_info(self) -> Dict[str, Any]:
        """Get information about detected era."""
        if self._current_guard is None:
            return {}

        profile = self._current_guard.profile

        return {
            "era": profile.era.value,
            "era_name": profile.era.name.replace("_", " ").title(),
            "source_medium": profile.source_medium.value,
            "philosophy": self.philosophy.value,
            "preserve_grain": profile.preserve_grain,
            "preserve_color_palette": profile.preserve_color_palette,
            "recommendations": self._current_guard.get_recommendations(),
        }


def create_authenticity_manager(
    philosophy: str = "accessible",
    era_hint: Optional[str] = None,
    medium_hint: Optional[str] = None,
) -> AuthenticityManager:
    """Create an authenticity manager with optional hints.

    Args:
        philosophy: Restoration philosophy (archival, accessible, enhanced, presentation)
        era_hint: Optional era hint (silent_film, golden_age, etc.)
        medium_hint: Optional medium hint (vhs, film, etc.)

    Returns:
        Configured AuthenticityManager
    """
    # Parse philosophy
    try:
        phil = RestorationPhilosophy(philosophy)
    except ValueError:
        phil = RestorationPhilosophy.ACCESSIBLE

    # Create custom profile if hints provided
    custom_profile = None
    if era_hint:
        try:
            era = Era(era_hint)
            base_profile = ERA_PROFILES.get(era, ERA_PROFILES[Era.MODERN])

            medium = SourceMedium.DIGITAL_FILE
            if medium_hint:
                try:
                    medium = SourceMedium(medium_hint)
                except ValueError:
                    pass

            custom_profile = AuthenticityProfile(
                era=era,
                source_medium=medium,
                **{k: v for k, v in base_profile.to_dict().items()
                   if k not in ["era", "source_medium"]}
            )
        except ValueError:
            pass

    return AuthenticityManager(
        philosophy=phil,
        custom_profile=custom_profile,
    )
