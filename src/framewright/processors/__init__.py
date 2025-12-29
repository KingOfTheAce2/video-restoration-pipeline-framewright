"""Processors module for framewright video restoration pipeline.

This module contains various processors for audio and video manipulation:

- analyzer: Automated content and degradation detection
- colorization: DeOldify/DDColor B&W colorization
- defect_repair: Scratch, dust, and grain removal
- face_restore: GFPGAN/CodeFormer face enhancement
- adaptive_enhance: Content-aware adaptive processing
- interpolation: RIFE frame interpolation
- audio: Audio processing
- audio_enhance: Enhanced audio restoration (AI + traditional)
- preview: Before/after comparison and quality metrics
- streaming: Chunk-based streaming processing
- watermark_removal: Watermark detection and removal
- subtitles: Subtitle preservation, extraction, timing sync, and embedding
- advanced_models: BasicVSR++, VRT, and Real-BasicVSR temporal video SR models
- audio_sync: AI-powered audio-video synchronization detection and correction
- stabilization: Video stabilization using FFmpeg vidstab and OpenCV
- subtitle_removal: Burnt-in subtitle detection and removal using OCR + inpainting
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
    # Data classes
    AudioWaveformInfo,
    SyncAnalysis,
    SyncCorrection,
    # Main classes
    AudioSyncDetector,
    AudioSyncCorrector,
    # Convenience functions
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

__all__ = [
    # Audio (basic)
    "AudioProcessor",
    "AudioProcessorError",
    # Audio Enhancement (advanced)
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
    "AudioWaveformInfo",
    "SyncAnalysis",
    "SyncCorrection",
    "AudioSyncDetector",
    "AudioSyncCorrector",
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
]
