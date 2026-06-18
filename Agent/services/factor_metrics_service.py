"""
Factor Metrics Service

Responsible for:
1. Unified organization of train / valid / test metric structures
2. Providing scoring logic for conflict comparisons
3. Providing recent performance proxy values for the linear weighting module
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


class FactorMetricsService:
    """Unified processing of structured factor metrics."""

    DEFAULT_SPLITS = ("train", "valid", "test")

    def ensure_metrics(self, factor: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in standardized metric structures for the factor."""
        metrics = deepcopy(factor.get("metrics", {}))
        if not isinstance(metrics, dict):
            metrics = {}

        train_metrics = metrics.get("train", {}) if isinstance(metrics.get("train", {}), dict) else {}
        valid_metrics = metrics.get("valid", {}) if isinstance(metrics.get("valid", {}), dict) else {}
        test_metrics = metrics.get("test", {}) if isinstance(metrics.get("test", {}), dict) else {}

        train_ic_raw = factor.get("train_ic")
        if train_ic_raw is None:
            train_ic_raw = train_metrics.get("ic", factor.get("ic"))
        train_ic = self._safe_float(train_ic_raw, default=0.0)

        train_rank_ic_raw = factor.get("train_rank_ic")
        if train_rank_ic_raw is None:
            train_rank_ic_raw = train_metrics.get("rank_ic", factor.get("rank_ic"))
        train_rank_ic = self._safe_float(train_rank_ic_raw, default=0.0)

        valid_ic_raw = factor.get("valid_ic")
        if valid_ic_raw is None:
            valid_ic_raw = valid_metrics.get("ic", factor.get("ic_valid"))
        valid_ic = self._safe_float(valid_ic_raw, default=0.0)

        valid_rank_ic_raw = factor.get("valid_rank_ic")
        if valid_rank_ic_raw is None:
            valid_rank_ic_raw = valid_metrics.get("rank_ic", factor.get("rank_ic_valid"))
        valid_rank_ic = self._safe_float(valid_rank_ic_raw, default=0.0)

        test_ic_raw = factor.get("test_ic")
        if test_ic_raw is None:
            test_ic_raw = test_metrics.get("ic", factor.get("ic_test"))
        test_ic = self._safe_float(test_ic_raw, default=0.0)

        test_rank_ic_raw = factor.get("test_rank_ic")
        if test_rank_ic_raw is None:
            test_rank_ic_raw = test_metrics.get("rank_ic", factor.get("rank_ic_test"))
        test_rank_ic = self._safe_float(test_rank_ic_raw, default=0.0)

        metrics["train"] = {
            "ic": train_ic,
            "rank_ic": train_rank_ic,
            "icir": self._safe_float(train_metrics.get("icir"), 0.0),
            "rank_icir": self._safe_float(train_metrics.get("rank_icir"), 0.0),
        }
        metrics["valid"] = {
            "ic": valid_ic,
            "rank_ic": valid_rank_ic,
            "icir": self._safe_float(valid_metrics.get("icir"), 0.0),
            "rank_icir": self._safe_float(valid_metrics.get("rank_icir"), 0.0),
        }
        metrics["test"] = {
            "ic": test_ic,
            "rank_ic": test_rank_ic,
            "icir": self._safe_float(test_metrics.get("icir"), 0.0),
            "rank_icir": self._safe_float(test_metrics.get("rank_icir"), 0.0),
        }

        factor["metrics"] = metrics
        factor["train_ic"] = train_ic
        factor["train_rank_ic"] = train_rank_ic
        factor["valid_ic"] = valid_ic
        factor["valid_rank_ic"] = valid_rank_ic
        factor["test_ic"] = test_ic
        factor["test_rank_ic"] = test_rank_ic
        return factor

    def get_conflict_score(self, factor: Dict[str, Any]) -> float:
        """
        Priority score for high-correlation conflicts.

        Preferentially use validation set Rank IC; fall back to validation set IC if missing.
        """
        metrics = factor.get("metrics", {})
        valid_metrics = metrics.get("valid", {})

        rank_ic = self._safe_float(
            factor.get("valid_rank_ic", valid_metrics.get("rank_ic")),
            default=0.0,
        )
        if rank_ic != 0:
            return rank_ic

        return self._safe_float(
            factor.get("valid_ic", valid_metrics.get("ic")),
            default=0.0,
        )

    def get_recent_excess_return_series(self, factor: Dict[str, Any], window_days: int) -> List[float]:
        """
        Get the factor's recent excess return series.

        Preferentially read real historical data from the factor object; if no real rolling returns are available yet,
        fall back to proxy values constructed from test/valid/train metrics to ensure the linear weighting module can run initially.
        """
        snapshot = self.get_recent_performance_snapshot(factor, window_days=window_days)
        return snapshot["series"]

    def get_recent_performance_snapshot(self, factor: Dict[str, Any], window_days: int) -> Dict[str, Any]:
        """
        Get a recent performance snapshot.

        Returns a unified structure, explicitly marking whether the linear weighting is using real 7-day rolling excess returns,
        or proxy values constructed by degrading historical metrics.
        """
        real_series = factor.get("recent_topk_excess_returns")
        if real_series:
            series = [self._safe_float(v, 0.0) for v in real_series[-window_days:]]
            return {
                "series": series,
                "source": "recent_topk_excess_returns",
                "is_proxy": False,
            }

        metrics = factor.get("metrics", {})
        proxy_score = (
            self._safe_float(metrics.get("test", {}).get("rank_ic"), 0.0)
            or self._safe_float(metrics.get("valid", {}).get("rank_ic"), 0.0)
            or self._safe_float(metrics.get("train", {}).get("rank_ic"), 0.0)
            or self._safe_float(metrics.get("test", {}).get("ic"), 0.0)
            or self._safe_float(metrics.get("train", {}).get("ic"), 0.0)
        )
        return {
            "series": [proxy_score for _ in range(window_days)],
            "source": "metrics_proxy",
            "is_proxy": True,
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert to float."""
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
