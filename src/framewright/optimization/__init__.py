"""Optimization module for high-performance processing."""

from .pipeline import (
    PipelineOptimizer,
    BatchProcessor,
    FramePrefetcher,
    ProcessingStage,
)
from .mixed_precision import (
    MixedPrecisionManager,
    PrecisionMode,
    get_optimal_precision,
)

__all__ = [
    "PipelineOptimizer",
    "BatchProcessor",
    "FramePrefetcher",
    "ProcessingStage",
    "MixedPrecisionManager",
    "PrecisionMode",
    "get_optimal_precision",
]
