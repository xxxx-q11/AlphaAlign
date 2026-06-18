"""
Factor Linear Weighting Service

Supports four weighting schemes:
1. normalized: Direct normalization based on recent average excess return
2. regression: Least-squares regression on recent window return series
3. score_ir: Non-negative weighting based on the product of recent strength and stability
4. mwu: Multiplicative weight update on recent window return series
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from Strategy.Allocator.mwu_allocator import MWUAllocator

from .factor_metrics_service import FactorMetricsService
from .rolling_factor_performance_service import RollingFactorPerformanceService


class FactorWeightingService:
    """Handles 7-day rolling factor selection and linear weighting."""

    def __init__(self) -> None:
        self.metrics_service = FactorMetricsService()
        self.rolling_performance_service = RollingFactorPerformanceService()

    def build_weighting_result(
        self,
        factor_library: List[Dict[str, Any]],
        weighting_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build the current-round linear weighting result from the factor library."""
        config = {
            "window_days": 7,
            "top_n": 5,
            "top_k": 5,
            "recent_perf_candidate_limit": None,
            "recent_perf_batch_size": 8,
            "instrument": "csi300",
            "benchmark": "csi300",
            "selection_date": "2023-01-01",
            "provider_uri": None,
            "return_expression": "Ref($close, -11)/Ref($close, -1) - 1",
            "weighting_method": "normalized",
            "mwu_learning_rate": 0.15,
            "mwu_reward_cap": 0.05,
            "mwu_explore_rate": 0.03,
            "mwu_max_weight": 0.15,
        }
        if weighting_config:
            config.update(weighting_config)

        if not factor_library:
            return {
                "status": "empty_factor_library",
                "selected_factors": [],
                "weights": [],
                "combined_expression": None,
                "config": config,
            }

        candidate_limit = self._resolve_recent_perf_candidate_limit(
            factor_library=factor_library,
            top_n=int(config["top_n"]),
            configured_limit=config.get("recent_perf_candidate_limit"),
        )
        evaluation_candidates = self._select_recent_performance_candidates(
            factor_library=factor_library,
            candidate_limit=candidate_limit,
        )

        recent_performance_context = self.rolling_performance_service.enrich_factor_library(
            evaluation_candidates,
            window_days=int(config["window_days"]),
            top_k=int(config["top_k"]),
            selection_date=str(config["selection_date"]),
            instrument=str(config["instrument"]),
            benchmark=str(config["benchmark"]),
            provider_uri=config.get("provider_uri"),
            return_expression=str(config["return_expression"]),
            batch_size=int(config["recent_perf_batch_size"]),
            metric_profile=RollingFactorPerformanceService.PROFILE_TOPK_RETURN,
        )
        recent_performance_context["candidate_factor_count"] = len(evaluation_candidates)
        recent_performance_context["candidate_limit"] = candidate_limit

        window_days = int(config["window_days"])
        require_real_recent_performance = bool(
            recent_performance_context.get("enriched_factor_count", 0) > 0
        )
        ranking_source = evaluation_candidates if require_real_recent_performance else factor_library
        ranked_candidates = self._rank_factors_by_recent_performance(
            ranking_source,
            window_days,
            require_real_recent_performance=require_real_recent_performance,
        )
        selected_items = ranked_candidates[: int(config["top_n"])]

        if not selected_items:
            return {
                "status": "no_selectable_factor",
                "selected_factors": [],
                "weights": [],
                "combined_expression": None,
                "recent_performance_context": recent_performance_context,
                "config": config,
            }

        method = str(config.get("weighting_method") or "normalized")
        if method == "mwu":
            weights = self._compute_mwu_weights(selected_items, config)
        elif method == "regression":
            weights = self._compute_regression_weights(selected_items)
        elif method == "score_ir":
            weights = self._compute_score_ir_weights(selected_items)
        else:
            method = "normalized"
            weights = self._compute_normalized_weights(selected_items)

        # If the current scheme fails to produce valid weights, fall back to direct normalization.
        if not weights:
            method = "normalized"
            weights = self._compute_normalized_weights(selected_items)

        weighted_factors = []
        for item, weight in zip(selected_items, weights):
            factor_expr = item["factor"].get("qlib_expression")
            if not factor_expr or float(weight) <= 0:
                continue

            weighted_factors.append(
                {
                    "factor_id": item["factor"]["factor_id"],
                    "qlib_expression": factor_expr,
                    "recent_score": item["recent_score"],
                    "recent_series": item["recent_series"],
                    "recent_score_source": item["recent_score_source"],
                    "is_proxy_score": item["is_proxy_score"],
                    "weight": float(weight),
                }
            )

        if not weighted_factors:
            return {
                "status": "no_positive_weight_factor",
                "method": method,
                "window_days": window_days,
                "top_n": int(config["top_n"]),
                "top_k": int(config["top_k"]),
                "benchmark": config["benchmark"],
                "selected_factors": [],
                "weights": [],
                "combined_expression": None,
                "recent_performance_context": recent_performance_context,
                "config": config,
            }

        weight_sum = float(sum(item["weight"] for item in weighted_factors))
        if weight_sum > 0:
            for item in weighted_factors:
                item["weight"] = float(item["weight"] / weight_sum)

        combined_expression = self._build_combined_expression(weighted_factors)
        proxy_factor_count = sum(1 for item in weighted_factors if item["is_proxy_score"])
        return {
            "status": "success",
            "method": method,
            "window_days": window_days,
            "top_n": int(config["top_n"]),
            "top_k": int(config["top_k"]),
            "instrument": config["instrument"],
            "benchmark": config["benchmark"],
            "selection_date": config["selection_date"],
            "selected_factors": weighted_factors,
            "weights": [item["weight"] for item in weighted_factors],
            "combined_expression": combined_expression,
            "recent_performance_context": recent_performance_context,
            "score_source_summary": {
                "real_recent_excess_return_factor_count": len(weighted_factors) - proxy_factor_count,
                "proxy_metric_factor_count": proxy_factor_count,
                "uses_proxy_scores": proxy_factor_count > 0,
            },
            "config": config,
        }

    def _resolve_recent_perf_candidate_limit(
        self,
        *,
        factor_library: List[Dict[str, Any]],
        top_n: int,
        configured_limit: Any,
    ) -> int:
        """Determine the candidate pool size for real rolling performance evaluation."""
        factor_count = len(factor_library)
        if factor_count <= 0:
            return 0

        if configured_limit is not None:
            try:
                return max(min(int(configured_limit), factor_count), min(top_n, factor_count))
            except (TypeError, ValueError):
                pass

        default_limit = max(top_n * 3, 20)
        return min(default_limit, factor_count)

    def _select_recent_performance_candidates(
        self,
        *,
        factor_library: List[Dict[str, Any]],
        candidate_limit: int,
    ) -> List[Dict[str, Any]]:
        """Pre-screen candidates using static proxy metrics to reduce the scale of Qlib real-time evaluation."""
        if candidate_limit <= 0 or candidate_limit >= len(factor_library):
            return factor_library

        ranked_by_proxy = []
        for factor in factor_library:
            proxy_score = self._get_proxy_metric_score(factor)
            ranked_by_proxy.append((proxy_score, factor))

        ranked_by_proxy.sort(key=lambda item: item[0], reverse=True)
        return [factor for _, factor in ranked_by_proxy[:candidate_limit]]

    def _get_proxy_metric_score(self, factor: Dict[str, Any]) -> float:
        """Use historical metrics as a cheap proxy score."""
        metrics = factor.get("metrics", {})
        candidate_values = [
            metrics.get("test", {}).get("rank_ic"),
            metrics.get("valid", {}).get("rank_ic"),
            metrics.get("train", {}).get("rank_ic"),
            metrics.get("test", {}).get("ic"),
            metrics.get("valid", {}).get("ic"),
            metrics.get("train", {}).get("ic"),
        ]
        score = 0.0
        for value in candidate_values:
            try:
                score = value if value is not None else score
                numeric_value = float(score)
            except (TypeError, ValueError):
                continue
            if abs(numeric_value) > 0:
                return numeric_value
        return 0.0

    def _rank_factors_by_recent_performance(
        self,
        factor_library: List[Dict[str, Any]],
        window_days: int,
        require_real_recent_performance: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rank factors by their recent window performance."""
        ranked = []
        for factor in factor_library:
            snapshot = self.metrics_service.get_recent_performance_snapshot(
                factor,
                window_days=window_days,
            )
            recent_series = snapshot["series"]
            if not recent_series:
                continue
            if require_real_recent_performance and bool(snapshot["is_proxy"]):
                continue

            recent_score = float(np.mean(recent_series))
            ranked.append(
                {
                    "factor": factor,
                    "recent_series": recent_series,
                    "recent_score": recent_score,
                    "recent_score_source": snapshot["source"],
                    "is_proxy_score": bool(snapshot["is_proxy"]),
                }
            )

        ranked.sort(key=lambda item: item["recent_score"], reverse=True)
        return ranked

    def _compute_normalized_weights(self, selected_items: List[Dict[str, Any]]) -> List[float]:
        """Direct normalization based on recent average excess return."""
        positive_scores = [max(item["recent_score"], 0.0) for item in selected_items]
        total_score = sum(positive_scores)

        # If all scores are non-positive, fall back to equal weights.
        if total_score <= 0:
            equal_weight = 1.0 / len(selected_items)
            return [equal_weight for _ in selected_items]

        return [score / total_score for score in positive_scores]

    def _compute_regression_weights(self, selected_items: List[Dict[str, Any]]) -> List[float]:
        """Least-squares regression on recent window return series."""
        if not selected_items:
            return []

        try:
            feature_matrix = np.array([item["recent_series"] for item in selected_items], dtype=float).T
            target = np.mean(feature_matrix, axis=1)
            coefficients, *_ = np.linalg.lstsq(feature_matrix, target, rcond=None)
            coefficients = np.maximum(coefficients, 0.0)

            if np.sum(coefficients) <= 0:
                return []

            normalized = coefficients / np.sum(coefficients)
            return [float(value) for value in normalized]
        except Exception:
            return []

    def _compute_score_ir_weights(self, selected_items: List[Dict[str, Any]]) -> List[float]:
        """Non-negative weighting based on the product of recent strength (recent_score) and stability (recent_ir)."""
        if not selected_items:
            return []

        raw_weights: List[float] = []
        for item in selected_items:
            recent_score = max(float(item.get("recent_score", 0.0)), 0.0)
            recent_series = np.array(item.get("recent_series", []) or [], dtype=float)
            recent_std = float(np.std(recent_series)) if recent_series.size > 0 else 0.0
            if recent_std > 1e-12:
                recent_ir = recent_score / recent_std
            elif recent_score > 0:
                recent_ir = 10.0
            else:
                recent_ir = 0.0
            raw_weights.append(float(recent_score * max(min(recent_ir, 10.0), 0.0)))

        raw_weight_sum = float(sum(raw_weights))
        if raw_weight_sum > 0:
            return [weight / raw_weight_sum for weight in raw_weights]

        return self._compute_normalized_weights(selected_items)

    def _compute_mwu_weights(self, selected_items: List[Dict[str, Any]], config: Dict[str, Any]) -> List[float]:
        """Update expert weights using MWU on the recent window return series."""
        if not selected_items:
            return []

        allocator = MWUAllocator(
            learning_rate=float(config.get("mwu_learning_rate", 0.15)),
            reward_cap=float(config.get("mwu_reward_cap", 0.05)),
            exploration_rate=float(config.get("mwu_explore_rate", 0.03)),
            max_weight=float(config.get("mwu_max_weight", 0.15)),
        )
        allocation_result = allocator.allocate(selected_items, context=None)
        weighted_factors = allocation_result.get("weighted_factors", []) or []
        weights_by_factor = {
            self._factor_weight_key(item): float(item.get("weight", 0.0))
            for item in weighted_factors
        }
        weights = [
            weights_by_factor.get(self._factor_weight_key(item.get("factor", {})), 0.0)
            for item in selected_items
        ]
        weight_sum = float(sum(weights))
        if weight_sum <= 0:
            return []
        return [float(weight / weight_sum) for weight in weights]

    @staticmethod
    def _factor_weight_key(factor: Dict[str, Any]) -> tuple[Any, Any, int]:
        """Build a stable factor key used for MWU weight backfill."""
        try:
            expert_direction = int(factor.get("expert_direction", 1))
        except (TypeError, ValueError):
            expert_direction = 1
        return (
            factor.get("factor_id"),
            factor.get("qlib_expression"),
            expert_direction,
        )

    def _build_combined_expression(self, weighted_factors: List[Dict[str, Any]]) -> str | None:
        """Combine weights and qlib expressions into a linear combination expression."""
        if not weighted_factors:
            return None

        expression = None
        for factor in weighted_factors:
            factor_expr = factor.get("qlib_expression")
            if not factor_expr:
                continue

            weighted_expression = f"Mul({factor['weight']:.6f}, {factor_expr})"
            if expression is None:
                expression = weighted_expression
            else:
                expression = f"Add({expression}, {weighted_expression})"

        return expression
