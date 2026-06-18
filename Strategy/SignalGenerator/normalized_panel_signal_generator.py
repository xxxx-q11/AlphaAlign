from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from qlib.data import D

from Strategy.base.base_signal_generator import BaseSignalGenerator
from Strategy.runtime.window_context import WindowContext


class NormalizedPanelSignalGenerator(BaseSignalGenerator):
    """Load factor panels, normalize them cross-sectionally, then aggregate by weight."""

    def generate(
        self,
        weighted_factors: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        active_factors = [factor for factor in weighted_factors if factor.get("qlib_expression") and factor.get("weight", 0)]
        if not active_factors:
            return {
                "signal": None,
                "combined_expression": None,
                "generation_context": {"status": "no_active_weighted_factor"},
            }

        expressions = [str(factor["qlib_expression"]) for factor in active_factors]
        feature_names = [f"factor_{index}" for index in range(len(expressions))]
        factor_panel = D.features(
            D.instruments(context.instrument),
            expressions,
            start_time=context.window_start.strftime("%Y-%m-%d"),
            end_time=context.window_end.strftime("%Y-%m-%d"),
        )
        if factor_panel is None or factor_panel.empty:
            return {
                "signal": None,
                "combined_expression": None,
                "generation_context": {"status": "empty_factor_panel"},
            }

        factor_panel.columns = feature_names
        factor_panel = self._canonicalize_panel(factor_panel)
        normalized_panel = self._cross_sectional_rank_normalize(factor_panel, feature_names).fillna(0.0)

        score = np.zeros(len(normalized_panel), dtype=float)
        for feature_name, factor in zip(feature_names, active_factors):
            score += float(factor.get("weight", 0.0)) * normalized_panel[feature_name].to_numpy(dtype=float)

        signal = pd.DataFrame({"score": score}, index=normalized_panel.index).dropna()
        combined_expression = self._build_combined_expression(active_factors)
        return {
            "signal": signal if not signal.empty else None,
            "combined_expression": combined_expression,
            "generation_context": {
                "status": "success",
                "signal_rows": 0 if signal.empty else int(len(signal)),
                "factor_count": len(active_factors),
                "normalization": "cross_sectional_rank",
            },
        }

    def _build_combined_expression(self, weighted_factors: list[dict[str, Any]]) -> str:
        top_components = sorted(
            weighted_factors,
            key=lambda item: float(item.get("weight", 0.0)),
            reverse=True,
        )[:10]
        summary = ", ".join(
            f"{component.get('factor_id', 'unknown')}:{float(component.get('weight', 0.0)):.4f}"
            for component in top_components
        )
        return f"normalized_panel_signal[{len(weighted_factors)} experts; top={summary}]"

    def _cross_sectional_rank_normalize(
        self,
        panel: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        if not feature_names:
            return pd.DataFrame(index=panel.index)

        date_level = "datetime" if "datetime" in panel.index.names else 0
        ranks = panel.loc[:, feature_names].groupby(level=date_level).rank(pct=True, method="average")
        return (ranks - 0.5) * 2.0

    def _canonicalize_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(panel.index, pd.MultiIndex):
            return panel

        index_names = list(panel.index.names)
        if "datetime" not in index_names or "instrument" not in index_names:
            inferred = self._infer_index_names(panel.index)
            if inferred is not None:
                panel.index = panel.index.set_names(inferred)
                index_names = list(panel.index.names)

        if "datetime" in index_names and "instrument" in index_names and index_names != ["datetime", "instrument"]:
            panel = panel.reorder_levels(["datetime", "instrument"]).sort_index()
        return panel

    @staticmethod
    def _infer_index_names(index: pd.MultiIndex) -> list[str] | None:
        if index.nlevels != 2:
            return None

        level0 = index.get_level_values(0)
        level1 = index.get_level_values(1)
        level0_is_datetime = NormalizedPanelSignalGenerator._looks_like_datetime_level(level0)
        level1_is_datetime = NormalizedPanelSignalGenerator._looks_like_datetime_level(level1)

        if level0_is_datetime and not level1_is_datetime:
            return ["datetime", "instrument"]
        if level1_is_datetime and not level0_is_datetime:
            return ["instrument", "datetime"]
        return None

    @staticmethod
    def _looks_like_datetime_level(level_values: pd.Index, sample_size: int = 50) -> bool:
        if pd.api.types.is_datetime64_any_dtype(level_values):
            return True

        sample = pd.Index(level_values).dropna().unique()[:sample_size]
        if len(sample) == 0:
            return False

        parsed = pd.to_datetime(sample, errors="coerce")
        parsed_ratio = float(parsed.notna().sum()) / float(len(sample))
        return parsed_ratio >= 0.8
