from __future__ import annotations

from typing import Any

import math
import numpy as np

from Strategy.base.base_allocator import BaseAllocator
from Strategy.runtime.window_context import WindowContext


class ScoreIRAllocator(BaseAllocator):
    """Allocate weights by recent strength times recent stability."""

    def __init__(self, *, ir_cap: float = 10.0) -> None:
        self.ir_cap = float(ir_cap)

    def allocate(
        self,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not selected_items:
            return {"weighted_factors": [], "allocation_context": {"method": "score_ir"}}

        raw_weights: list[float] = []
        ir_values: list[float] = []
        ir_sources: list[str] = []
        std_values: list[float] = []
        for item in selected_items:
            recent_score = max(self._safe_float(item.get("recent_score", 0.0)), 0.0)
            recent_ir, ir_source, recent_std = self._resolve_recent_ir(item)
            effective_ir = max(min(recent_ir, self.ir_cap), 0.0)
            raw_weights.append(float(recent_score * effective_ir))
            ir_values.append(float(recent_ir))
            ir_sources.append(ir_source)
            std_values.append(float(recent_std))

        raw_weight_sum = float(sum(raw_weights))
        used_fallback = raw_weight_sum <= 0
        if used_fallback:
            weights = self._fallback_weights(selected_items)
        else:
            weights = [weight / raw_weight_sum for weight in raw_weights]

        weighted_factors = self._build_weighted_factors(
            selected_items=selected_items,
            weights=weights,
            ir_values=ir_values,
            ir_sources=ir_sources,
            std_values=std_values,
            raw_weights=raw_weights,
        )
        return {
            "weighted_factors": weighted_factors,
            "allocation_context": {
                "method": "score_ir",
                "ir_cap": self.ir_cap,
                "selected_factor_count": len(weighted_factors),
                "used_fallback": used_fallback,
                "raw_weight_sum": raw_weight_sum,
            },
        }

    def _fallback_weights(self, selected_items: list[dict[str, Any]]) -> list[float]:
        positive_scores = [max(self._safe_float(item.get("recent_score", 0.0)), 0.0) for item in selected_items]
        score_sum = float(sum(positive_scores))
        if score_sum > 0:
            return [score / score_sum for score in positive_scores]
        equal_weight = 1.0 / len(selected_items)
        return [equal_weight for _ in selected_items]

    def _build_weighted_factors(
        self,
        *,
        selected_items: list[dict[str, Any]],
        weights: list[float],
        ir_values: list[float],
        ir_sources: list[str],
        std_values: list[float],
        raw_weights: list[float],
    ) -> list[dict[str, Any]]:
        weighted_factors: list[dict[str, Any]] = []
        for item, weight, recent_ir, ir_source, recent_std, raw_weight in zip(
            selected_items,
            weights,
            ir_values,
            ir_sources,
            std_values,
            raw_weights,
        ):
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
                    "recent_ir": float(recent_ir),
                    "recent_ir_source": ir_source,
                    "recent_std": float(recent_std),
                    "raw_weight": float(raw_weight),
                    "weight": float(weight),
                }
            )

        weight_sum = float(sum(item["weight"] for item in weighted_factors))
        if weight_sum > 0:
            for item in weighted_factors:
                item["weight"] = float(item["weight"] / weight_sum)
        return weighted_factors

    def _resolve_recent_ir(self, item: dict[str, Any]) -> tuple[float, str, float]:
        recent_score = self._safe_float(item.get("recent_score", 0.0))
        explicit_ir = item.get("recent_ir")
        if explicit_ir is not None and explicit_ir != "":
            return (
                self._clip_ir(self._safe_float(explicit_ir, 0.0)),
                "selected_item.recent_ir",
                self._resolve_recent_std(item),
            )

        recent_std = self._resolve_recent_std(item)
        if recent_std > 1e-12:
            return (self._clip_ir(recent_score / recent_std), "recent_series", recent_std)
        if recent_score > 0:
            return (self.ir_cap, "recent_series_flat_positive", recent_std)
        if recent_score < 0:
            return (-self.ir_cap, "recent_series_flat_negative", recent_std)
        return (0.0, "recent_series_flat_zero", recent_std)

    def _resolve_recent_std(self, item: dict[str, Any]) -> float:
        explicit_std = item.get("recent_std")
        if explicit_std is not None and explicit_std != "":
            return max(self._safe_float(explicit_std, 0.0), 0.0)

        recent_series = item.get("recent_series", []) or []
        numeric_series = [self._safe_float(value, 0.0) for value in recent_series]
        if not numeric_series:
            return 0.0
        return float(np.std(np.asarray(numeric_series, dtype=float)))

    def _clip_ir(self, value: float) -> float:
        if not math.isfinite(value):
            return self.ir_cap if value > 0 else -self.ir_cap
        return float(max(min(value, self.ir_cap), -self.ir_cap))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
