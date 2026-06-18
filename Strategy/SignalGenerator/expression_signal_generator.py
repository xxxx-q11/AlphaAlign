from __future__ import annotations

from typing import Any

from qlib.data import D

from Strategy.base.base_signal_generator import BaseSignalGenerator
from Strategy.runtime.window_context import WindowContext


class ExpressionSignalGenerator(BaseSignalGenerator):
    """Build a linear expression and evaluate it on the holding window."""

    def generate(
        self,
        weighted_factors: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        combined_expression = self._build_combined_expression(weighted_factors)
        if not combined_expression:
            return {
                "signal": None,
                "combined_expression": None,
                "generation_context": {"status": "no_positive_weight_factor"},
            }

        signal = D.features(
            D.instruments(context.instrument),
            [combined_expression],
            start_time=context.window_start.strftime("%Y-%m-%d"),
            end_time=context.window_end.strftime("%Y-%m-%d"),
        )
        signal.columns = ["score"]
        signal = signal.dropna()
        return {
            "signal": signal if not signal.empty else None,
            "combined_expression": combined_expression,
            "generation_context": {"status": "success", "signal_rows": 0 if signal.empty else len(signal)},
        }

    def _build_combined_expression(self, weighted_factors: list[dict[str, Any]]) -> str | None:
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
