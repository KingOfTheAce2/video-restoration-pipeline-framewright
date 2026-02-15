"""Processors module for framewright video restoration pipeline.

This module contains various processors for audio and video manipulation:

- analyzer: Automated content and degradation detection
- colorization: DeOldify/DDColor B&W colorization
- defect_repair: Scratch, dust, and grain removal
- face_restore: GFPGAN/CodeFormer face enhancement
- adaptive_enhance: Content-aware adaptive processing
- interpolation: RIFE frame interpolation
- audio: Basic audio processing (FFmpeg)
- audio_enhance: Enhanced audio restoration (AI + traditional)
- audio_restoration: Full audio restoration suite (dereverb, declick, etc.)
- audio_unified: **Unified audio enhancer** - Single interface with multiple backends:
    - FFmpeg backend (always available)
    - Traditional backend (from audio_enhance.py)
    - AI backend (denoiser/demucs neural models)
    - Restoration backend (from audio_restoration.py)
- audio_deepfilter: **DeepFilterNet 3 integration** - State-of-the-art audio enhancement:
    - 10-20ms latency real-time processing
    - DeepFilterNet3Backend (full PyTorch, highest quality)
    - DeepFilterLiteBackend (ONNX, lightweight)
    - SpeechBrainBackend (alternative AI)
    - TraditionalFilterBackend (FFmpeg fallback)
- preview: Before/after comparison and quality metrics
- streaming: Chunk-based streaming processing
- watermark_removal: Watermark detection and removal
- subtitles: Subtitle preservation, extraction, timing sync, and embedding
- advanced_models: BasicVSR++, VRT, and Real-BasicVSR temporal video SR models
- audio_sync: AI-powered audio-video synchronization detection and correction
- stabilization: Video stabilization using FFmpeg vidstab and OpenCV
- subtitle_removal: Burnt-in subtitle detection and removal using OCR + inpainting
- temporal_denoise: Advanced temporal denoising with optical flow and flicker reduction
- aspect_correction: Aspect ratio detection and correction (PAL, anamorphic, letterbox)
- telecine: Inverse telecine (IVTC) for removing 3:2/2:3/2:2 pulldown patterns
- interlace_handler: Interlacing detection and deinterlacing (VHS, DVD, broadcast)
- letterbox_handler: Letterbox/pillarbox detection and cropping
- film_stock_detector: Film stock type detection and era-specific color correction
- benchmark: Restoration performance benchmarking
- frame_quality_scorer: Frame quality analysis and problem frame detection
- credits_detector: Intro/credits/title card detection
"""

from .audio import AudioProcessor, AudioProcessorError
from .interpolation import FrameInterpolator, InterpolationError

from .audio_enhance import (
    # Configuration
    AudioEnhanceConfig,
    AudioEnhanceError,
    AIModelType,
    # Results
    AudioAnalysis,
    EnhancementResult,
    # Main classes
    TraditionalAudioEnhancer,
    AIAudioEnhancer,
    AudioAnalyzer,
    # Factory and convenience functions
    create_audio_enhancer,
    enhance_audio_auto,
)

from .analyzer import (
    # Core analyzer
    FrameAnalyzer,
    VideoAnalysis,
    ContentType,
    DegradationType,
    FrameMetrics,
    # Time estimation
    TimeEstimate,
    format_time_estimate,
    # Content detection
    ContentMix,
    ProcessingSegment,
    # Quality warnings
    QualityWarning,
    WarningSeverity,
    # Hardware recommendations
    HardwareInfo,
    ProcessingPlan,
    get_hardware_info,
    # Pre-flight checks
    PreflightResult,
    PreflightBlocker,
    BlockerType,
    # Batch analysis
    BatchAnalysis,
    BatchVideoInfo,
)

from .defect_repair import (
    DefectDetector,
    DefectRepairer,
    AutoDefectProcessor,
    DefectType,
    DefectMap,
    DefectRepairResult,
)

from .face_restore import (
    FaceRestorer,
    AutoFaceRestorer,
    FaceModel,
    FaceRestorationResult,
)

from .adaptive_enhance import (
    AdaptiveEnhancer,
    AutoEnhancePipeline,
    AdaptiveEnhanceResult,
)

from .streaming import (
    # Core streaming
    StreamingProcessor,
    StreamingConfig,
    StreamingState,
    ChunkInfo,
    create_streaming_restorer,
    # Distribution strategy
    DistributionStrategy,
    # Memory management
    MemoryManager,
    MemorySnapshot,
    # GPU distribution
    GPUDistributor,
    get_available_gpus,
    # Streaming pipeline
    StreamingPipeline,
    FrameBuffer,
    PipelineFrame,
    # Batch processing
    BatchFrameProcessor,
    BatchResult,
)

from .colorization import (
    Colorizer,
    AutoColorizer,
    ColorModel,
    ArtisticStyle,
    ColorizationConfig,
    ColorizationResult,
)

from .watermark_removal import (
    WatermarkRemover,
    WatermarkConfig,
    WatermarkRemovalResult,
    WatermarkPosition,
    remove_watermark_from_image,
)

from .preview import (
    PreviewGenerator,
    PreviewConfig,
    PreviewMode,
    LivePreview,
    QualityComparison,
    ComparisonMetrics,
    create_gradio_slider_preview,
)

from .advanced_models import (
    # Enums and configs
    AdvancedModel,
    AdvancedModelConfig,
    ModelRequirements,
    ProcessingResult,
    # Processors
    BasicVSRPP,
    VRTProcessor,
    # Selector
    ModelSelector,
    # Factory
    get_advanced_processor,
)

from .subtitles import (
    # Configuration
    SubtitleConfig,
    SubtitleError,
    OCREngine,
    SubtitleFormat,
    # Data classes
    BoundingBox,
    SubtitleLine,
    SubtitleTrack,
    SubtitleStreamInfo,
    # Main classes
    SubtitleExtractor,
    SubtitleTimeSync,
    SubtitleEnhancer,
    SubtitleMerger,
    # Convenience functions
    detect_burned_subtitles,
    extract_subtitles,
    remove_subtitles,
    preserve_subtitles_during_restoration,
)

from .scene_detection import (
    # Enums
    SceneType,
    TransitionType,
    # Data classes
    Scene,
    SceneEnhancementParams,
    SceneAnalysisResult,
    # Main classes
    SceneDetector,
    SceneAnalyzer,
    # Convenience function
    detect_and_analyze_scenes,
)

from .audio_sync import (
    # Exception
    AudioSyncError,
    # Configuration
    SyncConfig,
    # Data classes
    AudioWaveformInfo,
    SyncAnalysis,
    SyncCorrection,
    # Main classes
    AudioSyncAnalyzer,
    AudioSyncCorrector,
    AudioSyncDetector,  # Legacy alias for AudioSyncAnalyzer
    # Factory functions
    analyze_av_sync,
    fix_av_sync,
    # Legacy convenience functions
    analyze_audio_sync,
    correct_audio_sync,
    sync_audio_to_video,
)

from .stabilization import (
    # Enums
    StabilizationAlgorithm,
    SmoothingMode,
    # Data classes
    MotionVector,
    StabilizationConfig,
    StabilizationResult,
    # Main classes
    MotionAnalyzer,
    VideoStabilizer,
    # Convenience functions
    detect_shake_severity,
    stabilize_video,
    stabilize_frames,
    create_stabilizer,
)

from .subtitle_removal import (
    # Enums
    OCREngine as SubtitleOCREngine,
    SubtitleRegion,
    # Data classes
    SubtitleBox,
    SubtitleRemovalConfig,
    SubtitleRemovalResult,
    # Main classes
    SubtitleDetector,
    SubtitleRemover,
    AutoSubtitleRemover,
    # Convenience functions
    detect_burnt_subtitles,
    remove_burnt_subtitles,
    check_ocr_available,
)

from .temporal_denoise import (
    # Enums
    DenoiseMethod,
    FlickerMode,
    OpticalFlowMethod,
    # Data classes
    TemporalDenoiseConfig,
    TemporalDenoiseResult,
    FlowField,
    # Main classes
    OpticalFlowEstimator,
    FlickerReducer,
    TemporalConsistencyFilter,
    TemporalDenoiser,
    AutoTemporalDenoiser,
    # Convenience functions
    create_temporal_denoiser,
    denoise_video_frames,
    auto_denoise_video,
)

# Advanced features (Ultimate preset)
from .tap_denoise import (
    TAPModel,
    TAPDenoiseConfig,
    TAPDenoiseResult,
    TAPDenoiser,
    AutoTAPDenoiser,
    create_tap_denoiser,
    # v2.1 Motion-Adaptive Denoising
    MotionAdaptiveConfig,
    MotionAdaptiveTAPDenoiser,
)

from .aesrgan_face import (
    FaceDetectorType,
    AESRGANFaceConfig,
    FaceBox,
    AESRGANFaceResult,
    AESRGANFaceRestorer,
    create_aesrgan_restorer,
)

from .diffusion_sr import (
    DiffusionModel,
    DiffusionSRConfig,
    DiffusionSRResult,
    DiffusionSRProcessor,
    AutoDiffusionSR,
    create_diffusion_sr,
)

from .qp_artifact_removal import (
    ArtifactType,
    CompressionLevel,
    QPArtifactConfig,
    QPArtifactResult,
    QPArtifactRemover,
    create_qp_artifact_remover,
)

from .swintexco_colorize import (
    ColorPropagationMode,
    ExemplarColorizeConfig,
    ExemplarColorizeResult,
    SwinTExCoColorizer,
    create_swintexco_colorizer,
)

from .frame_generation import (
    GenerationModel,
    FrameGenerationConfig,
    GapInfo,
    FrameGenerationResult,
    MissingFrameGenerator,
    create_frame_generator,
)

from .cross_attention_temporal import (
    TemporalMethod,
    CrossAttentionConfig,
    TemporalConsistencyResult,
    CrossAttentionTemporalProcessor,
    create_temporal_processor,
)

# v2.0 Advanced Features
from .scene_intelligence import (
    ContentType as SceneContentType,
    SceneAnalysis,
    AdaptiveSettings,
    SceneIntelligence,
    create_scene_intelligence,
    # v2.1 Scene-Aware Processing
    SceneAdaptiveConfig,
    SceneAdaptiveProcessor,
)

from .vhs_restoration import (
    AnalogFormat,
    AnalogFormatProfile,
    ANALOG_PROFILES,
    VHSArtifactAnalysis,
    VHSRestorationConfig,
    VHSArtifactDetector,
    VHSRestorer,
    create_vhs_restorer,
)

from .film_restoration import (
    FilmType,
    FilmEra,
    FilmCharacteristics,
    FilmRestorationConfig,
    FilmAnalyzer,
    FilmRestorer,
    create_film_restorer,
)

from .telecine import (
    # Enums
    TelecinePattern,
    FieldOrder,
    # Data classes
    TelecineAnalysis,
    IVTCConfig,
    # Exception
    IVTCError,
    # Main class
    InverseTelecine,
    # Factory functions
    analyze_telecine,
    apply_ivtc,
    create_ivtc_processor,
)

from .aspect_correction import (
    # Enums
    AspectRatio,
    CorrectionMethod,
    # Data classes
    AspectConfig,
    AspectAnalysis,
    # Exception
    AspectCorrectionError,
    # Main class
    AspectCorrector,
    # Factory functions
    fix_aspect_ratio,
    create_aspect_corrector,
)

from .hdr_expansion import (
    # Enums
    HDRFormat,
    ToneMappingMethod,
    # Data classes
    HDRConfig,
    HDRMetadata,
    HDRExpansionResult,
    # Main class
    HDRExpander,
    # Factory function
    expand_to_hdr,
    create_hdr_expander,
)

from .perceptual_tuning import (
    # Enums
    PerceptualMode,
    # Data classes
    PerceptualConfig,
    PerceptualProfile,
    # Main class
    PerceptualTuner,
    # Factory functions
    create_perceptual_tuner,
    get_perceptual_profile,
)

# RTX 5090 / High-End GPU Optimizations
from .temporal_colorization import (
    # Enums
    PropagationMode,
    BlendMethod,
    # Data classes
    TemporalColorizationConfig,
    TemporalColorizationResult,
    # Main classes
    OpticalFlowColorPropagator,
    TemporalColorizationProcessor,
    # Factory functions
    create_temporal_colorization_processor,
    apply_temporal_colorization,
)

from .raft_flow import (
    # Data class
    RAFTFlowField,
    # Main class
    RAFTFlowEstimator,
    # Factory functions
    create_raft_estimator,
    estimate_flow_raft,
)

from .hat_upscaler import (
    # Enums
    HATModelSize,
    # Data classes
    HATConfig,
    HATResult,
    # Main class
    HATUpscaler,
    # Factory function
    create_hat_upscaler,
)

from .ensemble_sr import (
    # Enums
    VotingMethod,
    SRModel,
    # Data classes
    EnsembleConfig,
    EnsembleResult,
    # Main classes
    ModelProcessor,
    QualityMetrics,
    EnsembleSR,
    # Factory function
    create_ensemble_sr,
)

from .interlace_handler import (
    # Enums
    InterlaceType,
    DeinterlaceMethod,
    # Data classes
    InterlaceAnalysis,
    DeinterlaceResult,
    # Main classes
    InterlaceDetector,
    Deinterlacer,
    # Convenience functions
    analyze_interlacing,
    deinterlace_video,
)

from .letterbox_handler import (
    # Data classes
    CropRegion,
    AspectRatio as LetterboxAspectRatio,
    LetterboxAnalysis,
    # Main classes
    LetterboxDetector,
    LetterboxCropper,
    # Constants
    ASPECT_RATIOS,
    # Convenience functions
    detect_letterbox,
    crop_letterbox,
)

from .film_stock_detector import (
    # Enums
    FilmStock,
    FilmEra as FilmStockEra,
    # Data classes
    ColorProfile,
    FilmStockAnalysis,
    # Constants
    STOCK_PROFILES,
    # Main classes
    FilmStockDetector,
    FilmStockCorrector,
    # Convenience functions
    detect_film_stock,
    get_correction_for_stock,
)

from .noise_profiler import (
    # Enums
    NoiseType,
    DenoiserType,
    # Data classes
    NoiseCharacteristics,
    NoiseProfile,
    # Main class
    NoiseProfiler,
    # Convenience function
    analyze_noise,
)

from .upscale_detector import (
    # Enums
    UpscaleMethod,
    # Data classes
    Resolution,
    UpscaleAnalysis,
    # Constants
    COMMON_RESOLUTIONS,
    # Main class
    UpscaleDetector,
    # Convenience function
    detect_upscaling,
)

from .quick_preview import (
    # Data classes
    PreviewConfig as QuickPreviewConfig,
    PreviewFrame,
    QuickPreviewResult,
    # Main class
    QuickPreviewGenerator,
    # Convenience function
    generate_quick_preview,
)

from .benchmark import (
    # Enums
    BenchmarkType,
    # Data classes
    BenchmarkResult,
    BenchmarkReport,
    # Main class
    RestorationBenchmark,
)

from .frame_quality_scorer import (
    # Enums
    QualityIssue,
    # Data classes
    FrameScore,
    QualityReport,
    # Main class
    FrameQualityScorer,
)

from .credits_detector import (
    # Enums
    SegmentType,
    # Data classes
    DetectedSegment,
    CreditsAnalysis,
    # Main class
    CreditsDetector,
)

# Unified audio enhancer with multiple backends
from .audio_unified.enhancer import (
    # Main unified class
    AudioEnhancer,
    # Configuration and results
    AudioConfig,
    AudioResult,
    BackendType,
    EnhancementFeature,
    # Backend classes for direct access
    AudioBackend,
    FFmpegAudioBackend,
    TraditionalEnhancerBackend,
    AIAudioBackend,
    RestorationBackend,
    # Convenience functions
    enhance_audio,
    auto_enhance_audio,
    get_available_backends,
)

# Unified restoration processors
from .restoration.grain_manager import (
    # Enums
    FilmStockGrainType,
    # Data classes
    GrainProfile,
    GrainConfig,
    GrainRemovalResult,
    # Constants
    FILM_STOCK_PROFILES,
    # Main class
    GrainManager,
    # Factory functions
    create_grain_manager,
    extract_grain_profile,
    process_with_grain_preservation,
)

# DeepFilterNet audio enhancement (state-of-the-art, 10-20ms latency)
from .audio_deepfilter.deepfilter import (
    # Configuration
    DeepFilterConfig,
    BackendPriority as DeepFilterBackendPriority,
    # Analysis
    AudioAnalysis as DeepFilterAudioAnalysis,
    AudioAnalyzer as DeepFilterAudioAnalyzer,
    # Results
    EnhancementResult as DeepFilterEnhancementResult,
    # Backend base class
    DeepFilterBackend,
    # Backend implementations
    DeepFilterNet3Backend,
    DeepFilterLiteBackend,
    SpeechBrainBackend,
    TraditionalFilterBackend,
    # Main enhancer class
    DeepFilterEnhancer,
    # Factory functions
    create_deepfilter_enhancer,
    enhance_audio as deepfilter_enhance_audio,
    enhance_video_audio as deepfilter_enhance_video_audio,
    analyze_audio as deepfilter_analyze_audio,
    get_available_backends as deepfilter_get_available_backends,
)

__all__ = [
    # Audio (basic)
    "AudioProcessor",
    "AudioProcessorError",
    # Audio Enhancement (advanced - legacy)
    "AudioEnhanceConfig",
    "AudioEnhanceError",
    "AIModelType",
    "AudioAnalysis",
    "EnhancementResult",
    "TraditionalAudioEnhancer",
    "AIAudioEnhancer",
    "AudioAnalyzer",
    "create_audio_enhancer",
    "enhance_audio_auto",
    # Audio Enhancement (unified - recommended)
    "AudioEnhancer",
    "AudioConfig",
    "AudioResult",
    "BackendType",
    "EnhancementFeature",
    "AudioBackend",
    "FFmpegAudioBackend",
    "TraditionalEnhancerBackend",
    "AIAudioBackend",
    "RestorationBackend",
    "enhance_audio",
    "auto_enhance_audio",
    "get_available_backends",
    # Interpolation
    "FrameInterpolator",
    "InterpolationError",
    # Analyzer (core)
    "FrameAnalyzer",
    "VideoAnalysis",
    "ContentType",
    "DegradationType",
    "FrameMetrics",
    # Analyzer (time estimation)
    "TimeEstimate",
    "format_time_estimate",
    # Analyzer (content detection)
    "ContentMix",
    "ProcessingSegment",
    # Analyzer (quality warnings)
    "QualityWarning",
    "WarningSeverity",
    # Analyzer (hardware recommendations)
    "HardwareInfo",
    "ProcessingPlan",
    "get_hardware_info",
    # Analyzer (pre-flight checks)
    "PreflightResult",
    "PreflightBlocker",
    "BlockerType",
    # Analyzer (batch analysis)
    "BatchAnalysis",
    "BatchVideoInfo",
    # Defect repair
    "DefectDetector",
    "DefectRepairer",
    "AutoDefectProcessor",
    "DefectType",
    "DefectMap",
    "DefectRepairResult",
    # Face restoration
    "FaceRestorer",
    "AutoFaceRestorer",
    "FaceModel",
    "FaceRestorationResult",
    # Adaptive enhancement
    "AdaptiveEnhancer",
    "AutoEnhancePipeline",
    "AdaptiveEnhanceResult",
    # Streaming output
    "StreamingProcessor",
    "StreamingConfig",
    "StreamingState",
    "ChunkInfo",
    "create_streaming_restorer",
    # Distribution strategy
    "DistributionStrategy",
    # Memory management
    "MemoryManager",
    "MemorySnapshot",
    # GPU distribution
    "GPUDistributor",
    "get_available_gpus",
    # Streaming pipeline
    "StreamingPipeline",
    "FrameBuffer",
    "PipelineFrame",
    # Batch processing
    "BatchFrameProcessor",
    "BatchResult",
    # Colorization
    "Colorizer",
    "AutoColorizer",
    "ColorModel",
    "ArtisticStyle",
    "ColorizationConfig",
    "ColorizationResult",
    # Watermark removal
    "WatermarkRemover",
    "WatermarkConfig",
    "WatermarkRemovalResult",
    "WatermarkPosition",
    "remove_watermark_from_image",
    # Preview
    "PreviewGenerator",
    "PreviewConfig",
    "PreviewMode",
    "LivePreview",
    "QualityComparison",
    "ComparisonMetrics",
    "create_gradio_slider_preview",
    # Advanced models (temporal video SR)
    "AdvancedModel",
    "AdvancedModelConfig",
    "ModelRequirements",
    "ProcessingResult",
    "BasicVSRPP",
    "VRTProcessor",
    "ModelSelector",
    "get_advanced_processor",
    # Subtitles
    "SubtitleConfig",
    "SubtitleError",
    "OCREngine",
    "SubtitleFormat",
    "BoundingBox",
    "SubtitleLine",
    "SubtitleTrack",
    "SubtitleStreamInfo",
    "SubtitleExtractor",
    "SubtitleTimeSync",
    "SubtitleEnhancer",
    "SubtitleMerger",
    "detect_burned_subtitles",
    "extract_subtitles",
    "remove_subtitles",
    "preserve_subtitles_during_restoration",
    # Scene detection
    "SceneType",
    "TransitionType",
    "Scene",
    "SceneEnhancementParams",
    "SceneAnalysisResult",
    "SceneDetector",
    "SceneAnalyzer",
    "detect_and_analyze_scenes",
    # Audio Sync
    "AudioSyncError",
    "SyncConfig",
    "AudioWaveformInfo",
    "SyncAnalysis",
    "SyncCorrection",
    "AudioSyncAnalyzer",
    "AudioSyncCorrector",
    "AudioSyncDetector",  # Legacy alias
    "analyze_av_sync",
    "fix_av_sync",
    "analyze_audio_sync",
    "correct_audio_sync",
    "sync_audio_to_video",
    # Stabilization
    "StabilizationAlgorithm",
    "SmoothingMode",
    "MotionVector",
    "StabilizationConfig",
    "StabilizationResult",
    "MotionAnalyzer",
    "VideoStabilizer",
    "detect_shake_severity",
    "stabilize_video",
    "stabilize_frames",
    "create_stabilizer",
    # Burnt-in subtitle removal
    "SubtitleOCREngine",
    "SubtitleRegion",
    "SubtitleBox",
    "SubtitleRemovalConfig",
    "SubtitleRemovalResult",
    "SubtitleDetector",
    "SubtitleRemover",
    "AutoSubtitleRemover",
    "detect_burnt_subtitles",
    "remove_burnt_subtitles",
    "check_ocr_available",
    # Temporal Denoising
    "DenoiseMethod",
    "FlickerMode",
    "OpticalFlowMethod",
    "TemporalDenoiseConfig",
    "TemporalDenoiseResult",
    "FlowField",
    "OpticalFlowEstimator",
    "FlickerReducer",
    "TemporalConsistencyFilter",
    "TemporalDenoiser",
    "AutoTemporalDenoiser",
    "create_temporal_denoiser",
    "denoise_video_frames",
    "auto_denoise_video",
    # TAP Neural Denoising (Ultimate preset)
    "TAPModel",
    "TAPDenoiseConfig",
    "TAPDenoiseResult",
    "TAPDenoiser",
    "AutoTAPDenoiser",
    "create_tap_denoiser",
    # v2.1 Motion-Adaptive Denoising
    "MotionAdaptiveConfig",
    "MotionAdaptiveTAPDenoiser",
    # AESRGAN Face Enhancement
    "FaceDetectorType",
    "AESRGANFaceConfig",
    "FaceBox",
    "AESRGANFaceResult",
    "AESRGANFaceRestorer",
    "create_aesrgan_restorer",
    # Diffusion Super-Resolution
    "DiffusionModel",
    "DiffusionSRConfig",
    "DiffusionSRResult",
    "DiffusionSRProcessor",
    "AutoDiffusionSR",
    "create_diffusion_sr",
    # QP Artifact Removal
    "ArtifactType",
    "CompressionLevel",
    "QPArtifactConfig",
    "QPArtifactResult",
    "QPArtifactRemover",
    "create_qp_artifact_remover",
    # SwinTExCo Exemplar Colorization
    "ColorPropagationMode",
    "ExemplarColorizeConfig",
    "ExemplarColorizeResult",
    "SwinTExCoColorizer",
    "create_swintexco_colorizer",
    # Missing Frame Generation
    "GenerationModel",
    "FrameGenerationConfig",
    "GapInfo",
    "FrameGenerationResult",
    "MissingFrameGenerator",
    "create_frame_generator",
    # Cross-Attention Temporal Consistency
    "TemporalMethod",
    "CrossAttentionConfig",
    "TemporalConsistencyResult",
    "CrossAttentionTemporalProcessor",
    "create_temporal_processor",
    # v2.0 Scene Intelligence
    "SceneContentType",
    "SceneAnalysis",
    "AdaptiveSettings",
    "SceneIntelligence",
    "create_scene_intelligence",
    # v2.1 Scene-Aware Processing
    "SceneAdaptiveConfig",
    "SceneAdaptiveProcessor",
    # v2.0 VHS/Analog Restoration
    "AnalogFormat",
    "AnalogFormatProfile",
    "ANALOG_PROFILES",
    "VHSArtifactAnalysis",
    "VHSRestorationConfig",
    "VHSArtifactDetector",
    "VHSRestorer",
    "create_vhs_restorer",
    # v2.0 Film Restoration
    "FilmType",
    "FilmEra",
    "FilmCharacteristics",
    "FilmRestorationConfig",
    "FilmAnalyzer",
    "FilmRestorer",
    "create_film_restorer",
    # Aspect Ratio Correction
    "AspectRatio",
    "CorrectionMethod",
    "AspectConfig",
    "AspectAnalysis",
    "AspectCorrectionError",
    "AspectCorrector",
    "fix_aspect_ratio",
    "create_aspect_corrector",
    # Inverse Telecine (IVTC)
    "TelecinePattern",
    "FieldOrder",
    "TelecineAnalysis",
    "IVTCConfig",
    "IVTCError",
    "InverseTelecine",
    "analyze_telecine",
    "apply_ivtc",
    "create_ivtc_processor",
    # v2.1 HDR Expansion
    "HDRFormat",
    "ToneMappingMethod",
    "HDRConfig",
    "HDRMetadata",
    "HDRExpansionResult",
    "HDRExpander",
    "expand_to_hdr",
    "create_hdr_expander",
    # v2.1 Perceptual Tuning
    "PerceptualMode",
    "PerceptualConfig",
    "PerceptualProfile",
    "PerceptualTuner",
    "create_perceptual_tuner",
    "get_perceptual_profile",
    # RTX 5090 / High-End GPU - Temporal Colorization
    "PropagationMode",
    "BlendMethod",
    "TemporalColorizationConfig",
    "TemporalColorizationResult",
    "OpticalFlowColorPropagator",
    "TemporalColorizationProcessor",
    "create_temporal_colorization_processor",
    "apply_temporal_colorization",
    # RTX 5090 / High-End GPU - RAFT Optical Flow
    "RAFTFlowField",
    "RAFTFlowEstimator",
    "create_raft_estimator",
    "estimate_flow_raft",
    # RTX 5090 / High-End GPU - HAT Upscaler
    "HATModelSize",
    "HATConfig",
    "HATResult",
    "HATUpscaler",
    "create_hat_upscaler",
    # RTX 5090 / High-End GPU - Ensemble SR
    "VotingMethod",
    "SRModel",
    "EnsembleConfig",
    "EnsembleResult",
    "ModelProcessor",
    "QualityMetrics",
    "EnsembleSR",
    "create_ensemble_sr",
    # Interlace Detection & Deinterlacing
    "InterlaceType",
    "DeinterlaceMethod",
    "InterlaceAnalysis",
    "DeinterlaceResult",
    "InterlaceDetector",
    "Deinterlacer",
    "analyze_interlacing",
    "deinterlace_video",
    # Letterbox/Pillarbox Detection
    "CropRegion",
    "LetterboxAspectRatio",
    "LetterboxAnalysis",
    "LetterboxDetector",
    "LetterboxCropper",
    "ASPECT_RATIOS",
    "detect_letterbox",
    "crop_letterbox",
    # Film Stock Detection
    "FilmStock",
    "FilmStockEra",
    "ColorProfile",
    "FilmStockAnalysis",
    "STOCK_PROFILES",
    "FilmStockDetector",
    "FilmStockCorrector",
    "detect_film_stock",
    "get_correction_for_stock",
    # Noise Profiling
    "NoiseType",
    "DenoiserType",
    "NoiseCharacteristics",
    "NoiseProfile",
    "NoiseProfiler",
    "analyze_noise",
    # Upscale Detection
    "UpscaleMethod",
    "Resolution",
    "UpscaleAnalysis",
    "COMMON_RESOLUTIONS",
    "UpscaleDetector",
    "detect_upscaling",
    # Quick Preview
    "QuickPreviewConfig",
    "PreviewFrame",
    "QuickPreviewResult",
    "QuickPreviewGenerator",
    "generate_quick_preview",
    # Benchmark
    "BenchmarkType",
    "BenchmarkResult",
    "BenchmarkReport",
    "RestorationBenchmark",
    # Frame Quality Scoring
    "QualityIssue",
    "FrameScore",
    "QualityReport",
    "FrameQualityScorer",
    # Credits Detection
    "SegmentType",
    "DetectedSegment",
    "CreditsAnalysis",
    "CreditsDetector",
    # Grain Management (Film Grain Preservation)
    "FilmStockGrainType",
    "GrainProfile",
    "GrainConfig",
    "GrainRemovalResult",
    "FILM_STOCK_PROFILES",
    "GrainManager",
    "create_grain_manager",
    "extract_grain_profile",
    "process_with_grain_preservation",
    # DeepFilterNet Audio Enhancement (state-of-the-art)
    "DeepFilterConfig",
    "DeepFilterBackendPriority",
    "DeepFilterAudioAnalysis",
    "DeepFilterAudioAnalyzer",
    "DeepFilterEnhancementResult",
    "DeepFilterBackend",
    "DeepFilterNet3Backend",
    "DeepFilterLiteBackend",
    "SpeechBrainBackend",
    "TraditionalFilterBackend",
    "DeepFilterEnhancer",
    "create_deepfilter_enhancer",
    "deepfilter_enhance_audio",
    "deepfilter_enhance_video_audio",
    "deepfilter_analyze_audio",
    "deepfilter_get_available_backends",
]
