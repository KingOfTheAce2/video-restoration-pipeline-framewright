"""Reports module for FrameWright.

Provides QA reporting, quality analysis, trend tracking, and cost estimation capabilities.
"""

from .qa_report import (
    QAReportGenerator,
    QAReport,
    QualityGrade,
    FrameMetrics,
    SegmentAnalysis,
    FaceQualityReport,
    AudioQualityReport,
    ProcessingStats,
    generate_qa_report,
)

from .trends import (
    QualityDataPoint,
    TrendAnalysis,
    QualityTrends,
    create_quality_tracker,
)

from .cost_calculator import (
    CostEstimate,
    CostConfig,
    CostCalculator,
    estimate_processing_cost,
    DEFAULT_GPU_POWER_WATTS,
    DEFAULT_CLOUD_RATES,
)

__all__ = [
    # QA Report
    "QAReportGenerator",
    "QAReport",
    "QualityGrade",
    "FrameMetrics",
    "SegmentAnalysis",
    "FaceQualityReport",
    "AudioQualityReport",
    "ProcessingStats",
    "generate_qa_report",
    # Quality Trends
    "QualityDataPoint",
    "TrendAnalysis",
    "QualityTrends",
    "create_quality_tracker",
    # Cost Calculator
    "CostEstimate",
    "CostConfig",
    "CostCalculator",
    "estimate_processing_cost",
    "DEFAULT_GPU_POWER_WATTS",
    "DEFAULT_CLOUD_RATES",
]
