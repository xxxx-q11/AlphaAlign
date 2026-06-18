from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from Agent.services.factor_metrics_service import FactorMetricsService
from Agent.services.rolling_factor_performance_service import RollingFactorPerformanceService
from Strategy.base.base_selector import BaseSelector
from Strategy.runtime.window_context import WindowContext


class RecentPerformanceSelector(BaseSelector):
    """Select factors using recent rolling performance with proxy fallback."""

    def __init__(
        self,
        *,
        recent_perf_candidate_limit: int | None = None,
        recent_perf_batch_size: int = 8,
    ) -> None:
        self.recent_perf_candidate_limit = recent_perf_candidate_limit
        self.recent_perf_batch_size = int(recent_perf_batch_size)
        self.metrics_service = FactorMetricsService()
        self.rolling_service = RollingFactorPerformanceService()

    def select(
        self,
        factor_library: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not factor_library:
            return {
                "selected_items": [],
                "selection_context": {"status": "empty_factor_library"},
            }

        working_library = deepcopy(factor_library)
        candidate_limit = self._resolve_candidate_limit(
            factor_library=working_library,
            top_n=context.top_n,
            configured_limit=self.recent_perf_candidate_limit,
        )
        evaluation_candidates = self._select_candidates(
            factor_library=working_library,
            candidate_limit=candidate_limit,
        )

        selection_context = self.rolling_service.enrich_factor_library(
            evaluation_candidates,
            window_days=context.window_days,
            top_k=context.top_k,
            selection_date=context.selection_date.strftime("%Y-%m-%d"),
            instrument=context.instrument,
            benchmark=context.benchmark,
            provider_uri=str(context.provider_uri),
            return_expression=context.return_expression,
            batch_size=self.recent_perf_batch_size,
            metric_profile=RollingFactorPerformanceService.PROFILE_TOPK_RETURN,
        )
        selection_context["candidate_factor_count"] = len(evaluation_candidates)
        selection_context["candidate_limit"] = candidate_limit

        require_real_recent_performance = bool(selection_context.get("enriched_factor_count", 0) > 0)
        ranking_source = evaluation_candidates if require_real_recent_performance else working_library
        ranked_candidates = self._rank_factors(
            ranking_source,
            window_days=context.window_days,
            require_real_recent_performance=require_real_recent_performance,
        )
        return {
            "selected_items": ranked_candidates[: context.top_n],
            "selection_context": selection_context,
        }

    def _resolve_candidate_limit(
        self,
        *,
        factor_library: list[dict[str, Any]],
        top_n: int,
        configured_limit: int | None,
    ) -> int:
        factor_count = len(factor_library)
        if factor_count <= 0:
            return 0
        if configured_limit is not None:
            return max(min(int(configured_limit), factor_count), min(top_n, factor_count))
        default_limit = max(top_n * 3, 20)
        return min(default_limit, factor_count)

    def _select_candidates(
        self,
        *,
        factor_library: list[dict[str, Any]],
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        if candidate_limit <= 0 or candidate_limit >= len(factor_library):
            return factor_library

        ranked_by_proxy = []
        for factor in factor_library:
            proxy_score = self._get_proxy_metric_score(factor)
            ranked_by_proxy.append((proxy_score, factor))
        ranked_by_proxy.sort(key=lambda item: item[0], reverse=True)
        return [factor for _, factor in ranked_by_proxy[:candidate_limit]]

    def _get_proxy_metric_score(self, factor: dict[str, Any]) -> float:
        metrics = factor.get("metrics", {})
        candidate_values = [
            metrics.get("test", {}).get("rank_ic"),
            metrics.get("valid", {}).get("rank_ic"),
            metrics.get("train", {}).get("rank_ic"),
            metrics.get("test", {}).get("ic"),
            metrics.get("valid", {}).get("ic"),
            metrics.get("train", {}).get("ic"),
        ]
        for value in candidate_values:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if abs(numeric_value) > 0:
                return numeric_value
        return 0.0

    def _rank_factors(
        self,
        factor_library: list[dict[str, Any]],
        *,
        window_days: int,
        require_real_recent_performance: bool,
    ) -> list[dict[str, Any]]:
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
