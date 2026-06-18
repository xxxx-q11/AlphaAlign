from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from Agent.services.rolling_factor_performance_service import RollingFactorPerformanceService
from Strategy.base.base_selector import BaseSelector
from Strategy.runtime.window_context import WindowContext


class AFFRecentPerformanceSelector(BaseSelector):
    """AFF-style selector: threshold recent mean/IR, then keep the best factors."""

    def __init__(
        self,
        *,
        recent_perf_candidate_limit: int | None = None,
        recent_perf_batch_size: int = 8,
        score_threshold: float = 0.02,
        ir_threshold: float = 0.2,
        fallback_count: int = 1,
    ) -> None:
        self.recent_perf_candidate_limit = recent_perf_candidate_limit
        self.recent_perf_batch_size = int(recent_perf_batch_size)
        self.score_threshold = float(score_threshold)
        self.ir_threshold = float(ir_threshold)
        self.fallback_count = max(int(fallback_count), 1)
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
            metric_profile=RollingFactorPerformanceService.PROFILE_CROSS_SECTIONAL,
        )
        selection_context["candidate_factor_count"] = len(evaluation_candidates)
        selection_context["candidate_limit"] = candidate_limit
        selection_context["selector_mode"] = "aff_recent"
        selection_context["metric_source"] = "recent_rank_ics"
        selection_context["score_threshold"] = self.score_threshold
        selection_context["ir_threshold"] = self.ir_threshold
        selection_context["fallback_count"] = self.fallback_count

        require_real_recent_performance = bool(selection_context.get("enriched_factor_count", 0) > 0)
        ranking_source = evaluation_candidates if require_real_recent_performance else working_library
        ranked_candidates = self._rank_factors(
            ranking_source,
            window_days=context.window_days,
            require_real_recent_performance=require_real_recent_performance,
        )
        eligible_candidates = [
            item
            for item in ranked_candidates
            # if item["recent_score"] > self.score_threshold and item["recent_ir"] > self.ir_threshold
        ]
        selected_items = eligible_candidates[: context.top_n]
        if not selected_items:
            selected_items = ranked_candidates[: min(context.top_n, self.fallback_count)]

        selection_context["ranked_factor_count"] = len(ranked_candidates)
        selection_context["eligible_factor_count"] = len(eligible_candidates)
        selection_context["selected_factor_count"] = len(selected_items)
        selection_context["used_fallback"] = not bool(eligible_candidates)
        return {
            "selected_items": selected_items,
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
            snapshot = self._get_rank_ic_snapshot(factor, window_days=window_days)
            recent_series = snapshot["series"]
            if not recent_series:
                continue
            if require_real_recent_performance and bool(snapshot["is_proxy"]):
                continue

            recent_score = float(np.mean(recent_series))
            recent_std = float(np.std(recent_series))
            recent_ir = self._safe_information_ratio(recent_score, recent_std)
            if not bool(snapshot["is_proxy"]):
                recent_ir = self._safe_float(factor.get("recent_rank_icir"), recent_ir)
            if bool(snapshot["is_proxy"]) and abs(self._safe_float(snapshot.get("proxy_ir"))) > 0:
                recent_ir = self._safe_float(snapshot.get("proxy_ir"))
            ranked.append(
                {
                    "factor": factor,
                    "recent_series": recent_series,
                    "recent_score": recent_score,
                    "recent_std": recent_std,
                    "recent_ir": recent_ir,
                    "recent_score_source": snapshot["source"],
                    "is_proxy_score": bool(snapshot["is_proxy"]),
                }
            )

        ranked.sort(
            key=lambda item: (
                abs(float(item["recent_ir"])),
                float(item["recent_score"]),
            ),
            reverse=True,
        )
        return ranked

    def _get_rank_ic_snapshot(self, factor: dict[str, Any], window_days: int) -> dict[str, Any]:
        real_series = factor.get("recent_rank_ics")
        if real_series:
            series = [self._safe_float(value) for value in real_series[-window_days:]]
            if series:
                return {
                    "series": series,
                    "source": "recent_rank_ics",
                    "is_proxy": False,
                }

        proxy_rank_ic, _proxy_rank_icir = self._get_proxy_rank_metrics(factor)
        return {
            "series": [proxy_rank_ic for _ in range(window_days)],
            "source": "metrics_rank_ic_proxy",
            "is_proxy": True,
            "proxy_ir": _proxy_rank_icir,
        }

    def _get_proxy_rank_metrics(self, factor: dict[str, Any]) -> tuple[float, float]:
        metrics = factor.get("metrics", {})
        for split in ("test", "valid", "train"):
            split_metrics = metrics.get(split, {})
            rank_ic = self._safe_float(split_metrics.get("rank_ic"))
            rank_icir = self._safe_float(split_metrics.get("rank_icir"))
            if abs(rank_ic) > 0 or abs(rank_icir) > 0:
                return rank_ic, rank_icir

        rank_ic = self._safe_float(factor.get("train_rank_ic"))
        if abs(rank_ic) > 0:
            return rank_ic, 0.0
        return 0.0, 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_information_ratio(mean_value: float, std_value: float) -> float:
        if std_value > 1e-12:
            return float(mean_value / std_value)
        if mean_value > 0:
            return float("inf")
        if mean_value < 0:
            return float("-inf")
        return 0.0
