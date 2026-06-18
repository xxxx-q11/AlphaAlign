"""Weight allocators."""

from .mwu_allocator import MWUAllocator
from .normalized_score_allocator import NormalizedScoreAllocator
from .regression_allocator import RegressionAllocator
from .score_ir_allocator import ScoreIRAllocator

__all__ = ["NormalizedScoreAllocator", "RegressionAllocator", "ScoreIRAllocator", "MWUAllocator"]
