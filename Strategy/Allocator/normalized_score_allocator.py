from __future__ import annotations

from typing import Any

from Strategy.base.base_allocator import BaseAllocator
from Strategy.runtime.window_context import WindowContext


class NormalizedScoreAllocator(BaseAllocator):
    """Allocate weights by normalizing positive recent scores."""

    def allocate(
        self,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not selected_items:
            return {"weighted_factors": [], "allocation_context": {"method": "normalized"}}

        positive_scores = [max(float(item.get("recent_score", 0.0)), 0.0) for item in selected_items]
        score_sum = sum(positive_scores)
        if score_sum <= 0:
            weights = [1.0 / len(selected_items) for _ in selected_items]
        else:
            weights = [score / score_sum for score in positive_scores]

        weighted_factors = self._build_weighted_factors(selected_items, weights)
        return {
            "weighted_factors": weighted_factors,
            "allocation_context": {"method": "normalized", "selected_factor_count": len(weighted_factors)},
        }

    def _build_weighted_factors(
        self,
        selected_items: list[dict[str, Any]],
        weights: list[float],
    ) -> list[dict[str, Any]]:
        weighted_factors: list[dict[str, Any]] = []
        for item, weight in zip(selected_items, weights):
            factor_expr = item["factor"].get("qlib_expression")
            if not factor_expr or float(weight) <= 0:
                continue
            weighted_factors.append(
                {
                    "factor_id": item["factor"].get("factor_id"),
                    "qlib_expression": factor_expr,
                    "recent_score": float(item.get("recent_score", 0.0)),
                    "recent_series": item.get("recent_series", []),
                    "recent_score_source": item.get("recent_score_source"),
                    "is_proxy_score": bool(item.get("is_proxy_score", False)),
                    "weight": float(weight),
                }
            )

        weight_sum = float(sum(item["weight"] for item in weighted_factors))
        if weight_sum > 0:
            for item in weighted_factors:
                item["weight"] = float(item["weight"] / weight_sum)
        return weighted_factors
