"""Perceptual tuning processor for FrameWright.

This module provides perceptual tuning capabilities that allow users to balance
between faithful preservation of original content and enhanced visual appeal.

Key features:
- Three perceptual modes: Faithful, Balanced, Enhanced
- Configurable balance between preservation and enhancement
- Content-type aware adjustments
- Grain and color cast preservation options
- Sharpening and denoise limits to avoid artifacts
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PerceptualMode(Enum):
    """Perceptual tuning modes for restoration."""
    FAITHFUL = "faithful"  # Preserve original look
    BALANCED = "balanced"  # Balance preservation and enhancement
    ENHANCED = "enhanced"  # Maximize visual appeal


@dataclass
class PerceptualConfig:
    """Configuration for perceptual tuning.

    Attributes:
        mode: Perceptual mode (faithful, balanced, or enhanced)
        balance: Balance factor (0.0 = faithful, 1.0 = enhanced)
        preserve_grain: Whether to preserve film grain
        preserve_color_cast: Keep period color tinting (e.g., sepia, warmth)
        sharpening_limit: Maximum sharpening to avoid artifacts (0.0-1.0)
        denoise_limit: Maximum denoise to preserve texture (0.0-1.0)
    """
    mode: PerceptualMode = PerceptualMode.BALANCED
    balance: float = 0.5  # 0.0 = faithful, 1.0 = enhanced
    preserve_grain: bool = True
    preserve_color_cast: bool = False  # Keep period color tinting
    sharpening_limit: float = 0.4  # Max sharpening to avoid artifacts
    denoise_limit: float = 0.6  # Max denoise to preserve texture

    def __post_init__(self):
        """Validate configuration values."""
        self.balance = max(0.0, min(1.0, self.balance))
        self.sharpening_limit = max(0.0, min(1.0, self.sharpening_limit))
        self.denoise_limit = max(0.0, min(1.0, self.denoise_limit))


@dataclass
class PerceptualProfile:
    """Perceptual profile containing adjustment parameters.

    Attributes:
        sharpening: Sharpening strength (0.0-1.0)
        denoise: Denoising strength (0.0-1.0)
        color_enhancement: Color enhancement strength (0.0-1.0)
        detail_recovery: Detail recovery strength (0.0-1.0)
        grain_preservation: Grain preservation factor (0.0-1.0)
    """
    sharpening: float = 0.0
    denoise: float = 0.0
    color_enhancement: float = 0.0
    detail_recovery: float = 0.0
    grain_preservation: float = 1.0

    def __post_init__(self):
        """Validate profile values."""
        self.sharpening = max(0.0, min(1.0, self.sharpening))
        self.denoise = max(0.0, min(1.0, self.denoise))
        self.color_enhancement = max(0.0, min(1.0, self.color_enhancement))
        self.detail_recovery = max(0.0, min(1.0, self.detail_recovery))
        self.grain_preservation = max(0.0, min(1.0, self.grain_preservation))

    def to_dict(self) -> Dict[str, float]:
        """Convert profile to dictionary."""
        return {
            "sharpening": self.sharpening,
            "denoise": self.denoise,
            "color_enhancement": self.color_enhancement,
            "detail_recovery": self.detail_recovery,
            "grain_preservation": self.grain_preservation,
        }


# Predefined profiles for each mode
_FAITHFUL_PROFILE = PerceptualProfile(
    sharpening=0.1,
    denoise=0.1,
    color_enhancement=0.0,
    detail_recovery=0.2,
    grain_preservation=1.0,
)

_ENHANCED_PROFILE = PerceptualProfile(
    sharpening=0.6,
    denoise=0.7,
    color_enhancement=0.5,
    detail_recovery=0.8,
    grain_preservation=0.2,
)


class PerceptualTuner:
    """Perceptual tuning processor for video restoration.

    This class manages perceptual adjustments during restoration to balance
    between faithful preservation of the original content and enhanced
    visual appeal.

    Example:
        >>> config = PerceptualConfig(mode=PerceptualMode.BALANCED, balance=0.5)
        >>> tuner = PerceptualTuner(config)
        >>> profile = tuner.get_profile()
        >>> adjusted_settings = tuner.adjust_settings(base_settings, "film")
    """

    def __init__(self, config: Optional[PerceptualConfig] = None):
        """Initialize the perceptual tuner.

        Args:
            config: Perceptual configuration. Uses default balanced config if None.
        """
        self.config = config or PerceptualConfig()
        self._profile: Optional[PerceptualProfile] = None
        self._cv2 = None
        self._np = None

        logger.info(
            f"PerceptualTuner initialized: mode={self.config.mode.value}, "
            f"balance={self.config.balance}"
        )

    def _ensure_deps(self) -> bool:
        """Ensure OpenCV and numpy are available."""
        try:
            import cv2
            import numpy as np
            self._cv2 = cv2
            self._np = np
            return True
        except ImportError:
            logger.warning("OpenCV/numpy not available for frame processing")
            return False

    def get_profile(self) -> PerceptualProfile:
        """Get the perceptual profile based on current configuration.

        The profile is computed based on the mode and balance settings.
        Faithful mode prioritizes preservation, enhanced mode prioritizes
        visual appeal, and balanced mode interpolates between them.

        Returns:
            PerceptualProfile with computed adjustment parameters.
        """
        if self._profile is not None:
            return self._profile

        self._profile = self._calculate_perceptual_adjustments(
            self.config.mode,
            self.config.balance
        )

        # Apply limits from config
        if self._profile.sharpening > self.config.sharpening_limit:
            self._profile.sharpening = self.config.sharpening_limit

        if self._profile.denoise > self.config.denoise_limit:
            self._profile.denoise = self.config.denoise_limit

        # Override grain preservation if configured
        if self.config.preserve_grain:
            self._profile.grain_preservation = max(
                self._profile.grain_preservation, 0.7
            )

        logger.debug(f"Computed profile: {self._profile.to_dict()}")
        return self._profile

    def adjust_settings(
        self,
        base_settings: Dict[str, Any],
        content_type: str = "general"
    ) -> Dict[str, Any]:
        """Adjust restoration settings based on perceptual profile.

        Takes base restoration settings and adjusts them according to the
        perceptual profile and content type.

        Args:
            base_settings: Base restoration settings dictionary
            content_type: Content type for adjustments. Supported types:
                - "general": Default adjustments
                - "film": Film footage adjustments
                - "animation": Animation/anime adjustments
                - "documentary": Documentary footage
                - "lowlight": Low-light footage
                - "faces": Face-focused content

        Returns:
            Adjusted settings dictionary
        """
        profile = self.get_profile()
        settings = base_settings.copy()

        # Apply profile-based adjustments
        if "sharpening" in settings:
            settings["sharpening"] = self._scale_setting(
                settings["sharpening"], profile.sharpening
            )
        else:
            settings["sharpening"] = profile.sharpening

        if "denoise" in settings:
            settings["denoise"] = self._scale_setting(
                settings["denoise"], profile.denoise
            )
        else:
            settings["denoise"] = profile.denoise

        if "color_enhancement" in settings:
            settings["color_enhancement"] = self._scale_setting(
                settings["color_enhancement"], profile.color_enhancement
            )
        else:
            settings["color_enhancement"] = profile.color_enhancement

        if "detail_recovery" in settings:
            settings["detail_recovery"] = self._scale_setting(
                settings["detail_recovery"], profile.detail_recovery
            )
        else:
            settings["detail_recovery"] = profile.detail_recovery

        # Grain preservation
        settings["grain_preservation"] = profile.grain_preservation

        # Content-type specific adjustments
        settings = self._apply_content_type_adjustments(settings, content_type)

        # Apply color cast preservation if configured
        if self.config.preserve_color_cast:
            settings["preserve_color_cast"] = True
            settings["color_enhancement"] = min(settings.get("color_enhancement", 0), 0.2)

        logger.debug(f"Adjusted settings for {content_type}: {settings}")
        return settings

    def apply_to_frame(self, frame) -> Any:
        """Apply perceptual adjustments to a single frame.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            Processed frame as numpy array, or original frame if deps unavailable
        """
        if not self._ensure_deps():
            return frame

        cv2 = self._cv2
        np = self._np
        profile = self.get_profile()

        result = frame.copy()

        # Apply denoising (respecting grain preservation)
        if profile.denoise > 0.1 and profile.grain_preservation < 0.9:
            denoise_strength = profile.denoise * (1 - profile.grain_preservation)
            h_value = int(denoise_strength * 10)  # Map to cv2 parameter range
            if h_value > 0:
                result = cv2.fastNlMeansDenoisingColored(
                    result, None, h_value, h_value, 7, 21
                )

        # Apply sharpening
        if profile.sharpening > 0.1:
            # Unsharp mask technique
            blur = cv2.GaussianBlur(result, (0, 0), 3)
            amount = profile.sharpening * 1.5
            result = cv2.addWeighted(result, 1 + amount, blur, -amount, 0)

        # Apply color enhancement
        if profile.color_enhancement > 0.1:
            # Increase saturation
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + profile.color_enhancement * 0.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply detail recovery (contrast enhancement)
        if profile.detail_recovery > 0.2:
            # CLAHE for local contrast enhancement
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            clahe_limit = 1.0 + profile.detail_recovery * 2.0
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Blend with original based on grain preservation (for grain texture)
        if self.config.preserve_grain and profile.grain_preservation > 0.3:
            # Extract high-frequency grain from original
            original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # High-pass filter to get grain
            blur_original = cv2.GaussianBlur(original_gray, (5, 5), 0)
            grain = cv2.subtract(original_gray, blur_original)

            # Add grain back to result
            grain_amount = profile.grain_preservation * 0.5
            grain_3ch = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(result, 1.0, grain_3ch, grain_amount, 0)

        return result

    def _calculate_perceptual_adjustments(
        self,
        mode: PerceptualMode,
        balance: float
    ) -> PerceptualProfile:
        """Calculate perceptual adjustments based on mode and balance.

        Args:
            mode: The perceptual mode
            balance: Balance factor (0.0 = faithful, 1.0 = enhanced)

        Returns:
            Computed PerceptualProfile
        """
        if mode == PerceptualMode.FAITHFUL:
            return PerceptualProfile(
                sharpening=_FAITHFUL_PROFILE.sharpening,
                denoise=_FAITHFUL_PROFILE.denoise,
                color_enhancement=_FAITHFUL_PROFILE.color_enhancement,
                detail_recovery=_FAITHFUL_PROFILE.detail_recovery,
                grain_preservation=_FAITHFUL_PROFILE.grain_preservation,
            )
        elif mode == PerceptualMode.ENHANCED:
            return PerceptualProfile(
                sharpening=_ENHANCED_PROFILE.sharpening,
                denoise=_ENHANCED_PROFILE.denoise,
                color_enhancement=_ENHANCED_PROFILE.color_enhancement,
                detail_recovery=_ENHANCED_PROFILE.detail_recovery,
                grain_preservation=_ENHANCED_PROFILE.grain_preservation,
            )
        else:
            # Balanced mode - blend based on balance factor
            return self._blend_profiles(
                _FAITHFUL_PROFILE,
                _ENHANCED_PROFILE,
                balance
            )

    def _blend_profiles(
        self,
        faithful: PerceptualProfile,
        enhanced: PerceptualProfile,
        balance: float
    ) -> PerceptualProfile:
        """Blend two perceptual profiles based on balance factor.

        Args:
            faithful: The faithful (preservation) profile
            enhanced: The enhanced (visual appeal) profile
            balance: Balance factor (0.0 = faithful, 1.0 = enhanced)

        Returns:
            Blended PerceptualProfile
        """
        def lerp(a: float, b: float, t: float) -> float:
            """Linear interpolation."""
            return a + (b - a) * t

        return PerceptualProfile(
            sharpening=lerp(faithful.sharpening, enhanced.sharpening, balance),
            denoise=lerp(faithful.denoise, enhanced.denoise, balance),
            color_enhancement=lerp(
                faithful.color_enhancement, enhanced.color_enhancement, balance
            ),
            detail_recovery=lerp(
                faithful.detail_recovery, enhanced.detail_recovery, balance
            ),
            grain_preservation=lerp(
                faithful.grain_preservation, enhanced.grain_preservation, balance
            ),
        )

    def _scale_setting(self, base_value: float, profile_value: float) -> float:
        """Scale a base setting value by the profile value.

        Args:
            base_value: Original setting value
            profile_value: Profile-based scaling factor

        Returns:
            Scaled setting value
        """
        # Use profile value as a blend between 0 and 2x base value
        return base_value * (0.5 + profile_value)

    def _apply_content_type_adjustments(
        self,
        settings: Dict[str, Any],
        content_type: str
    ) -> Dict[str, Any]:
        """Apply content-type specific adjustments to settings.

        Args:
            settings: Current settings dictionary
            content_type: Content type identifier

        Returns:
            Adjusted settings dictionary
        """
        adjustments = {
            "film": {
                "grain_preservation": min(settings.get("grain_preservation", 1.0) + 0.2, 1.0),
                "denoise": max(settings.get("denoise", 0) - 0.1, 0),
            },
            "animation": {
                "grain_preservation": 0.0,  # Animation has no grain
                "sharpening": min(settings.get("sharpening", 0) + 0.1, 1.0),
                "denoise": min(settings.get("denoise", 0) + 0.2, 1.0),
            },
            "documentary": {
                "grain_preservation": min(settings.get("grain_preservation", 1.0) + 0.1, 1.0),
                "color_enhancement": max(settings.get("color_enhancement", 0) - 0.1, 0),
            },
            "lowlight": {
                "denoise": min(settings.get("denoise", 0) + 0.2, 1.0),
                "detail_recovery": min(settings.get("detail_recovery", 0) + 0.2, 1.0),
            },
            "faces": {
                "denoise": min(settings.get("denoise", 0) + 0.1, 1.0),
                "detail_recovery": min(settings.get("detail_recovery", 0) + 0.15, 1.0),
                "sharpening": max(settings.get("sharpening", 0) - 0.05, 0),
            },
        }

        if content_type in adjustments:
            for key, value in adjustments[content_type].items():
                settings[key] = value

        return settings


def create_perceptual_tuner(mode: str = "balanced") -> PerceptualTuner:
    """Factory function to create a perceptual tuner with specified mode.

    Args:
        mode: Perceptual mode name ("faithful", "balanced", or "enhanced")

    Returns:
        Configured PerceptualTuner instance

    Raises:
        ValueError: If mode is not recognized
    """
    mode_map = {
        "faithful": PerceptualMode.FAITHFUL,
        "balanced": PerceptualMode.BALANCED,
        "enhanced": PerceptualMode.ENHANCED,
    }

    mode_lower = mode.lower()
    if mode_lower not in mode_map:
        valid_modes = ", ".join(mode_map.keys())
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {valid_modes}")

    perceptual_mode = mode_map[mode_lower]

    # Set balance based on mode
    if perceptual_mode == PerceptualMode.FAITHFUL:
        balance = 0.0
    elif perceptual_mode == PerceptualMode.ENHANCED:
        balance = 1.0
    else:
        balance = 0.5

    config = PerceptualConfig(mode=perceptual_mode, balance=balance)
    return PerceptualTuner(config)


def get_perceptual_profile(balance: float) -> PerceptualProfile:
    """Get a perceptual profile for the given balance value.

    This is a convenience function that creates a balanced-mode tuner
    with the specified balance and returns its profile.

    Args:
        balance: Balance factor (0.0 = faithful, 1.0 = enhanced)

    Returns:
        PerceptualProfile computed for the given balance
    """
    balance = max(0.0, min(1.0, balance))
    config = PerceptualConfig(mode=PerceptualMode.BALANCED, balance=balance)
    tuner = PerceptualTuner(config)
    return tuner.get_profile()
