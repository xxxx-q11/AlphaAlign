from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from Agent.services.rolling_factor_performance_service import RollingFactorPerformanceService
from Strategy.base.base_selector import BaseSelector
from Strategy.runtime.window_context import WindowContext


class MWUAllExpertSelector(BaseSelector):
    """Evaluate all factors as experts with rolling top-k excess-return rewards."""

    def __init__(
        self,
        *,
        recent_perf_batch_size: int = 8,
        include_inverse_experts: bool = True,
        min_history_days: int = 5,
    ) -> None:
        self.recent_perf_batch_size = int(recent_perf_batch_size)
        self.include_inverse_experts = bool(include_inverse_experts)
        self.min_history_days = max(int(min_history_days), 1)
        self.rolling_service = RollingFactorPerformanceService()

    def select(
        self,
        factor_library: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not factor_library:
            return {
                "selected_items": [],
                "selection_context": {"status": "empty_factor_library", "selector_mode": "mwu"},
            }

        expert_library = self._build_expert_library(deepcopy(factor_library))
        selection_context = self.rolling_service.enrich_factor_library(
            expert_library,
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

        selected_items = self._build_selected_items(expert_library)
        selection_context["selector_mode"] = "mwu"
        selection_context["metric_source"] = "recent_topk_excess_returns"
        selection_context["base_factor_count"] = len(factor_library)
        selection_context["expert_factor_count"] = len(expert_library)
        selection_context["include_inverse_experts"] = self.include_inverse_experts
        selection_context["selected_factor_count"] = len(selected_items)
        selection_context["min_history_days"] = self.min_history_days

        return {
            "selected_items": selected_items,
            "selection_context": selection_context,
        }

    def _build_expert_library(self, factor_library: list[dict[str, Any]]) -> list[dict[str, Any]]:
        experts: list[dict[str, Any]] = []
        for index, factor in enumerate(factor_library):
            expression = str(factor.get("qlib_expression") or "").strip()
            if not expression:
                continue

            base_factor_id = str(factor.get("factor_id") or f"factor_{index}")
            experts.append(
                self._build_expert_factor(
                    factor=factor,
                    base_factor_id=base_factor_id,
                    direction=1,
                    expression=expression,
                    base_expression=expression,
                )
            )
            if self.include_inverse_experts:
                experts.append(
                    self._build_expert_factor(
                        factor=factor,
                        base_factor_id=base_factor_id,
                        direction=-1,
                        expression=f"Mul(-1, {expression})",
                        base_expression=expression,
                    )
                )
        return experts

    def _build_expert_factor(
        self,
        *,
        factor: dict[str, Any],
        base_factor_id: str,
        direction: int,
        expression: str,
        base_expression: str,
    ) -> dict[str, Any]:
        expert_factor = deepcopy(factor)
        expert_label = "long" if direction > 0 else "short"
        expert_factor["base_factor_id"] = base_factor_id
        expert_factor["base_qlib_expression"] = base_expression
        expert_factor["expert_direction"] = int(direction)
        expert_factor["expert_label"] = expert_label
        expert_factor["factor_id"] = f"{base_factor_id}__{expert_label}"
        expert_factor["qlib_expression"] = expression
        return expert_factor

    def _build_selected_items(self, expert_library: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected_items: list[dict[str, Any]] = []
        for factor in expert_library:
            recent_series = [self._safe_float(value) for value in factor.get("recent_topk_excess_returns", [])]
            recent_eval_dates = [str(value) for value in factor.get("recent_topk_eval_dates", [])]
            recent_topk_returns = [self._safe_float(value) for value in factor.get("recent_topk_returns", [])]
            recent_benchmark_returns = [
                self._safe_float(value) for value in factor.get("recent_topk_benchmark_returns", [])
            ]

            aligned_length = min(
                len(recent_series),
                len(recent_eval_dates) if recent_eval_dates else len(recent_series),
                len(recent_topk_returns) if recent_topk_returns else len(recent_series),
                len(recent_benchmark_returns) if recent_benchmark_returns else len(recent_series),
            )
            if aligned_length < self.min_history_days:
                continue

            if len(recent_eval_dates) != aligned_length:
                recent_eval_dates = recent_eval_dates[-aligned_length:]
            if len(recent_series) != aligned_length:
                recent_series = recent_series[-aligned_length:]
            if len(recent_topk_returns) == aligned_length:
                recent_topk_returns = recent_topk_returns[-aligned_length:]
            else:
                recent_topk_returns = []
            if len(recent_benchmark_returns) == aligned_length:
                recent_benchmark_returns = recent_benchmark_returns[-aligned_length:]
            else:
                recent_benchmark_returns = []

            recent_score = float(np.mean(recent_series))
            recent_std = float(np.std(np.asarray(recent_series, dtype=float)))
            recent_ir = self._safe_information_ratio(recent_score, recent_std)
            selected_items.append(
                {
                    "factor": factor,
                    "recent_series": recent_series,
                    "recent_eval_dates": recent_eval_dates,
                    "recent_topk_returns": recent_topk_returns,
                    "recent_benchmark_returns": recent_benchmark_returns,
                    "recent_score": recent_score,
                    "recent_std": recent_std,
                    "recent_ir": recent_ir,
                    "recent_score_source": "recent_topk_excess_returns",
                    "is_proxy_score": False,
                    "base_factor_id": factor.get("base_factor_id"),
                    "expert_direction": int(factor.get("expert_direction", 1)),
                    "expert_label": factor.get("expert_label"),
                }
            )

        selected_items.sort(
            key=lambda item: (
                float(item.get("recent_score", 0.0)),
                float(item.get("recent_ir", 0.0)),
                str(item.get("factor", {}).get("factor_id", "")),
            ),
            reverse=True,
        )
        return selected_items

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
