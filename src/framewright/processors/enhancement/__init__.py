"""Enhancement processors for video restoration.

This module contains unified enhancement processors that combine multiple
backends with automatic hardware-based selection.

Denoising Example:
    >>> from framewright.processors.enhancement import Denoiser, detect_hardware
    >>> hardware = detect_hardware()
    >>> denoiser = Denoiser(hardware=hardware)
    >>> result = denoiser.denoise(input_dir, output_dir)

    # Or use the factory for automatic setup:
    >>> from framewright.processors.enhancement import denoise_frames
    >>> result = denoise_frames(input_dir, output_dir, strength=0.7)

Super-Resolution Example:
    >>> from framewright.processors.enhancement import SuperResolution, detect_hardware
    >>> hardware = detect_hardware()
    >>> sr = SuperResolution(hardware=hardware)
    >>> result = sr.upscale(input_dir, output_dir)

    # Or use the factory for automatic setup:
    >>> from framewright.processors.enhancement import upscale_frames
    >>> result = upscale_frames(input_dir, output_dir, scale=4)
"""

from .denoising import (
    # Configuration
    DenoiserConfig,
    # Core types
    HardwareTier,
    HardwareInfo,
    GPUVendor,
    DenoiseResult,
    # Backend protocols
    DenoiserBackend,
    # Backend implementations (for advanced use)
    TraditionalDenoiser,
    TemporalDenoiserBackend,
    TAPDenoiserBackend,
    NCNNDenoiser,
    # Main class
    Denoiser,
    # Factory functions
    create_denoiser,
    denoise_frames,
    # Hardware detection
    detect_hardware,
    get_hardware_tier,
)

from .super_resolution import (
    # Enums
    SRBackendType,
    # Configuration
    SRConfig,
    # Result types
    SRResult,
    # Backend protocol
    SRBackend,
    # Backend implementations
    NCNNVulkanSRBackend,
    RealESRGANBackend,
    HATBackend,
    BasicVSRPPBackend,
    VRTBackend,
    DiffusionSRBackend,
    EnsembleSRBackend,
    # Main class
    SuperResolution,
    # Factory functions
    create_super_resolution,
    upscale_frames,
    get_recommended_backend,
    list_available_backends,
)

# One-step diffusion super resolution (FlashVSR-style)
from .diffusion_sr import (
    # Configuration
    FlashSRConfig,
    PrecisionMode,
    # Result type
    DiffusionSRResult,
    # Backend base class (renamed to avoid conflict with super_resolution.DiffusionSRBackend)
    DiffusionSRBackend as FlashDiffusionBackend,
    # Backend implementations
    FlashVSRBackend,
    StableSRBackend,
    SwinIRDiffusionBackend,
    FallbackInterpolationBackend,
    # Main class
    DiffusionSuperResolution,
    # Factory functions
    create_diffusion_sr,
    upscale_video_diffusion,
    get_available_backends as get_diffusion_backends,
    estimate_vram_requirement,
)

# Cross-Attention Temporal VAE for consistency (TE-3DVAE style)
from .temporal_vae import (
    # Configuration
    TemporalVAEConfig,
    ConsistencyMode,
    PrecisionMode as VAEPrecisionMode,
    # Result type
    TemporalVAEResult,
    # Main classes
    TemporalVAE,
    ConsistencyEnforcer,
    # Integration
    TemporalVAEProcessor,
    # Factory functions
    create_temporal_vae,
    enforce_temporal_consistency,
    get_vae_info,
)

# Text-guided super resolution (Upscale-A-Video style)
from .guided_sr import (
    # Configuration
    GuidedSRConfig,
    # Result type
    GuidedSRResult,
    # Style presets
    StylePresets,
    # Text encoding
    TextEncoder,
    # Backend base class
    GuidedDiffusionBackend,
    # Backend implementations
    SDGuidedSRBackend,
    FallbackGuidedBackend,
    # Texture generation
    TextureGenerator,
    # Main class
    GuidedSuperResolution,
    # Factory functions
    create_guided_sr,
    upscale_with_guidance,
    upscale_with_style,
    list_style_presets,
    get_style_preset_info,
)

# HDR and Dolby Vision export capabilities
from .hdr_export import (
    # Enums
    HDRFormat,
    ToneMappingAlgorithm,
    ColorSpaceType,
    # Configuration
    HDRConfig,
    HDRMetadata,
    # Result types
    HDRAnalysisResult,
    HDRExportResult,
    # Main classes
    ToneMapper,
    ColorSpaceConverter,
    HDRAnalyzer,
    HDRExporter,
    SDRtoHDR,
    # Factory functions
    create_hdr_exporter,
    export_as_hdr,
    analyze_hdr,
)

__all__ = [
    # ===== Denoising =====
    # Configuration
    "DenoiserConfig",
    # Core types
    "HardwareTier",
    "HardwareInfo",
    "GPUVendor",
    "DenoiseResult",
    # Backend protocols
    "DenoiserBackend",
    # Backend implementations (for advanced use)
    "TraditionalDenoiser",
    "TemporalDenoiserBackend",
    "TAPDenoiserBackend",
    "NCNNDenoiser",
    # Main class
    "Denoiser",
    # Factory functions
    "create_denoiser",
    "denoise_frames",
    # Hardware detection
    "detect_hardware",
    "get_hardware_tier",
    # ===== Super-Resolution =====
    # Enums
    "SRBackendType",
    # Configuration
    "SRConfig",
    # Result types
    "SRResult",
    # Backend protocol
    "SRBackend",
    # Backend implementations
    "NCNNVulkanSRBackend",
    "RealESRGANBackend",
    "HATBackend",
    "BasicVSRPPBackend",
    "VRTBackend",
    "DiffusionSRBackend",
    "EnsembleSRBackend",
    # Main class
    "SuperResolution",
    # Factory functions
    "create_super_resolution",
    "upscale_frames",
    "get_recommended_backend",
    "list_available_backends",
    # ===== One-Step Diffusion Super-Resolution (FlashVSR) =====
    # Configuration
    "FlashSRConfig",
    "PrecisionMode",
    # Result type
    "DiffusionSRResult",
    # Backend base class
    "FlashDiffusionBackend",
    # Backend implementations
    "FlashVSRBackend",
    "StableSRBackend",
    "SwinIRDiffusionBackend",
    "FallbackInterpolationBackend",
    # Main class
    "DiffusionSuperResolution",
    # Factory functions
    "create_diffusion_sr",
    "upscale_video_diffusion",
    "get_diffusion_backends",
    "estimate_vram_requirement",
    # ===== Cross-Attention Temporal VAE (TE-3DVAE) =====
    # Configuration
    "TemporalVAEConfig",
    "ConsistencyMode",
    "VAEPrecisionMode",
    # Result type
    "TemporalVAEResult",
    # Main classes
    "TemporalVAE",
    "ConsistencyEnforcer",
    # Integration
    "TemporalVAEProcessor",
    # Factory functions
    "create_temporal_vae",
    "enforce_temporal_consistency",
    "get_vae_info",
    # ===== Text-Guided Super-Resolution (Upscale-A-Video) =====
    # Configuration
    "GuidedSRConfig",
    # Result type
    "GuidedSRResult",
    # Style presets
    "StylePresets",
    # Text encoding
    "TextEncoder",
    # Backend base class
    "GuidedDiffusionBackend",
    # Backend implementations
    "SDGuidedSRBackend",
    "FallbackGuidedBackend",
    # Texture generation
    "TextureGenerator",
    # Main class
    "GuidedSuperResolution",
    # Factory functions
    "create_guided_sr",
    "upscale_with_guidance",
    "upscale_with_style",
    "list_style_presets",
    "get_style_preset_info",
    # ===== HDR and Dolby Vision Export =====
    # Enums
    "HDRFormat",
    "ToneMappingAlgorithm",
    "ColorSpaceType",
    # Configuration
    "HDRConfig",
    "HDRMetadata",
    # Result types
    "HDRAnalysisResult",
    "HDRExportResult",
    # Main classes
    "ToneMapper",
    "ColorSpaceConverter",
    "HDRAnalyzer",
    "HDRExporter",
    "SDRtoHDR",
    # Factory functions
    "create_hdr_exporter",
    "export_as_hdr",
    "analyze_hdr",
]
