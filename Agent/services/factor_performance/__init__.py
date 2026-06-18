"""Factor performance computation profiles."""

from .cross_sectional_profile import (
    CROSS_SECTIONAL_PROFILE_NAME,
    build_cross_sectional_snapshot,
    compute_cross_sectional_daily_metrics,
)
from .topk_return_profile import (
    TOPK_RETURN_PROFILE_NAME,
    build_topk_return_snapshot,
    compute_topk_return_daily_metrics,
)

__all__ = [
    "CROSS_SECTIONAL_PROFILE_NAME",
    "TOPK_RETURN_PROFILE_NAME",
    "build_cross_sectional_snapshot",
    "build_topk_return_snapshot",
    "compute_cross_sectional_daily_metrics",
    "compute_topk_return_daily_metrics",
]
