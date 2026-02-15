"""Unified content analysis module for FrameWright.

This module consolidates all video analysis capabilities into a single
coherent API with lazy loading of heavy components.

Key classes:
- ContentAnalyzer: Main unified analyzer combining all analysis capabilities
- ContentAnalysis: Complete analysis results dataclass
- QualityScorer: Comprehensive quality assessment with multiple metrics
- DegradationDetector: Detection of noise, blur, artifacts, and damage

Example:
    >>> from framewright.processors.analysis import ContentAnalyzer
    >>> analyzer = ContentAnalyzer()
    >>> analysis = analyzer.analyze("video.mp4")
    >>> print(f"Content type: {analysis.content_type}")
    >>> print(f"Recommended preset: {analysis.recommended_preset}")
    >>>
    >>> # Quality scoring
    >>> from framewright.processors.analysis import QualityScorer
    >>> scorer = QualityScorer()
    >>> metrics = scorer.score_frame(frame)
    >>> print(f"Overall quality: {metrics.overall_score:.1f}")
    >>>
    >>> # Degradation detection
    >>> from framewright.processors.analysis import DegradationDetector
    >>> detector = DegradationDetector()
    >>> degradations = detector.detect(frame)
    >>> recommendations = detector.suggest_processors(degradations)
"""

from .content_analyzer import (
    # Main classes
    ContentAnalyzer,
    # Configuration
    AnalyzerConfig,
    # Results dataclasses
    ContentAnalysis,
    # Enums
    ContentType,
    SourceFormat,
    DegradationType,
    # Supporting types
    NoiseProfile,
    Scene,
    # Factory function
    create_content_analyzer,
    # Convenience function
    analyze_video,
    quick_analyze,
)

from .quality_scorer import (
    # Data classes
    QualityMetrics,
    VideoQualityMetrics,
    ComparisonResult,
    # Report generation
    QualityReport,
    # Main class
    QualityScorer,
    # Convenience functions
    score_frame,
    score_video,
    compare_quality,
    generate_quality_report,
)

from .degradation_detector import (
    # Enums
    DegradationType as DegradationTypeDetector,  # Alias to avoid conflict
    SeverityLevel,
    # Data classes
    DegradationRegion,
    DegradationInfo,
    VideoAnalysisResult,
    ProcessorRecommendation,
    # Main class
    DegradationDetector,
    # Convenience functions
    detect_degradations,
    analyze_video_degradations,
    suggest_restoration_pipeline,
)

__all__ = [
    # Content Analyzer
    "ContentAnalyzer",
    "AnalyzerConfig",
    "ContentAnalysis",
    "ContentType",
    "SourceFormat",
    "DegradationType",
    "NoiseProfile",
    "Scene",
    "create_content_analyzer",
    "analyze_video",
    "quick_analyze",
    # Quality Scorer
    "QualityMetrics",
    "VideoQualityMetrics",
    "ComparisonResult",
    "QualityReport",
    "QualityScorer",
    "score_frame",
    "score_video",
    "compare_quality",
    "generate_quality_report",
    # Degradation Detector
    "DegradationTypeDetector",
    "SeverityLevel",
    "DegradationRegion",
    "DegradationInfo",
    "VideoAnalysisResult",
    "ProcessorRecommendation",
    "DegradationDetector",
    "detect_degradations",
    "analyze_video_degradations",
    "suggest_restoration_pipeline",
]
