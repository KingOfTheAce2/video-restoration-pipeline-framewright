"""FrameWright Benchmarking Suite.

Provides standardized performance testing for the video restoration pipeline,
including profiling, benchmarking, and performance analysis tools.
"""

from .benchmark_suite import (
    BenchmarkRunner,
    BenchmarkMetrics,
    BenchmarkReporter,
    StandardTestSuite,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkType,
    DeviceType,
    ResourceMonitor,
    SystemProfiler,
    TestVideoGenerator,
    QualityAnalyzer,
)

from .profiler import (
    PerformanceProfiler,
    ProfileReport,
    ProcessingStage,
    StageMetrics,
    ProfileSummary,
    analyze_profile,
)

__all__ = [
    # Benchmark classes
    "BenchmarkRunner",
    "BenchmarkMetrics",
    "BenchmarkReporter",
    "StandardTestSuite",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkType",
    "DeviceType",
    "ResourceMonitor",
    "SystemProfiler",
    "TestVideoGenerator",
    "QualityAnalyzer",
    # Profiler classes
    "PerformanceProfiler",
    "ProfileReport",
    "ProcessingStage",
    "StageMetrics",
    "ProfileSummary",
    "analyze_profile",
]
