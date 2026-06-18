from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Strategy.base.base_allocator import BaseAllocator
from Strategy.runtime.window_context import WindowContext


class RegressionAllocator(BaseAllocator):
    """Fit linear factor weights on historical stock panels with percentile labels."""

    def __init__(
        self,
        *,
        fit_window_days: int | None = None,
        target_horizon_days: int | None = 10,
        target_expression: str | None = None,
        ridge_alpha: float = 1e-6,
        center_target: bool = True,
    ) -> None:
        self.fit_window_days = fit_window_days
        self.target_horizon_days = target_horizon_days
        self.target_expression = target_expression
        self.ridge_alpha = max(float(ridge_alpha), 0.0)
        self.center_target = bool(center_target)
        self._qlib_initialized = False
        self._provider_uri: Path | None = None
        self._D = None

    def allocate(
        self,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not selected_items:
            return {"weighted_factors": [], "allocation_context": {"method": "regression"}}

        weights, regression_context = self._compute_regression_weights(selected_items, context)
        used_fallback = not weights
        if used_fallback:
            weights = self._fallback_weights(selected_items)
            raw_weights = list(weights)
        else:
            raw_weights = list(regression_context.get("item_raw_weights", weights))

        weighted_factors = self._build_weighted_factors(
            selected_items=selected_items,
            weights=weights,
            raw_weights=raw_weights,
        )
        allocation_context = {
            "method": "regression",
            "selected_factor_count": len(weighted_factors),
            "used_fallback": used_fallback,
        }
        allocation_context.update(regression_context)
        print(
            "[RegressionAllocator] "
            f"window_index={context.window_index}, "
            f"fit_status={allocation_context.get('fit_status')}, "
            f"used_fallback={allocation_context.get('used_fallback')}, "
            f"selected_factor_count={allocation_context.get('selected_factor_count')}, "
            f"fit_factor_count={allocation_context.get('fit_factor_count')}, "
            f"sample_count={allocation_context.get('sample_count')}"
        )
        return {
            "weighted_factors": weighted_factors,
            "allocation_context": allocation_context,
        }

    def _compute_regression_weights(
        self,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> tuple[list[float], dict[str, Any]]:
        horizon_days = max(int(self.target_horizon_days or 1), 1)
        label_lookahead_days = horizon_days + 1
        fit_window_days = max(int(self.fit_window_days or context.window_days), 1)
        target_expression = self.target_expression or self._build_target_expression(horizon_days)
        context_payload: dict[str, Any] = {
            "fit_window_days": fit_window_days,
            "target_horizon_days": horizon_days,
            "target_expression": target_expression,
            "target_label_mode": "cross_sectional_percentile",
            "center_target": self.center_target,
            "ridge_alpha": self.ridge_alpha,
        }

        indexed_items = [
            (index, item)
            for index, item in enumerate(selected_items)
            if item.get("factor", {}).get("qlib_expression")
        ]
        if not indexed_items:
            context_payload["fit_status"] = "no_factor_expression"
            return [], context_payload

        if not self._ensure_qlib_initialized(context.provider_uri):
            context_payload["fit_status"] = "qlib_init_failed"
            context_payload["provider_uri"] = str(self._provider_uri) if self._provider_uri else None
            return [], context_payload

        training_dates = self._resolve_training_dates(
            selection_date=context.selection_date,
            fit_window_days=fit_window_days,
            horizon_days=label_lookahead_days,
        )
        if not training_dates:
            context_payload["fit_status"] = "insufficient_training_dates"
            context_payload["provider_uri"] = str(self._provider_uri) if self._provider_uri else None
            return [], context_payload

        start_date = training_dates[0].strftime("%Y-%m-%d")
        end_date = training_dates[-1].strftime("%Y-%m-%d")
        context_payload["provider_uri"] = str(self._provider_uri) if self._provider_uri else None
        context_payload["training_start"] = start_date
        context_payload["training_end"] = end_date
        context_payload["training_date_count"] = len(training_dates)

        factor_frame, valid_items, invalid_factor_ids, load_error = self._load_training_frame(
            indexed_items=indexed_items,
            instrument=context.instrument,
            start_date=start_date,
            end_date=end_date,
            target_expression=target_expression,
        )
        if load_error:
            context_payload["data_load_warning"] = load_error
        if invalid_factor_ids:
            context_payload["invalid_factor_ids"] = invalid_factor_ids
        if factor_frame is None or factor_frame.empty or not valid_items:
            context_payload["fit_status"] = "training_frame_unavailable"
            return [], context_payload

        feature_names = [f"factor_{offset}" for offset in range(len(valid_items))]
        try:
            prepared = self._prepare_training_frame(
                factor_frame=factor_frame,
                feature_names=feature_names,
            )
        except Exception as exc:
            context_payload["fit_status"] = "training_frame_prepare_failed"
            context_payload["fit_error"] = f"{type(exc).__name__}: {exc}"
            return [], context_payload

        if prepared is None:
            context_payload["fit_status"] = "training_samples_unavailable"
            return [], context_payload

        training_frame, feature_matrix, target_vector = prepared
        coefficients, fit_metrics = self._solve_coefficients(
            feature_matrix=feature_matrix,
            target_vector=target_vector,
            training_frame=training_frame,
        )
        if not coefficients:
            context_payload["fit_status"] = "coefficient_solve_failed"
            context_payload.update(fit_metrics)
            return [], context_payload

        weights = [0.0 for _ in selected_items]
        raw_weights = [0.0 for _ in selected_items]
        for (item_index, _), raw_weight, normalized_weight in zip(
            valid_items,
            fit_metrics["raw_weights"],
            coefficients,
        ):
            raw_weights[item_index] = float(raw_weight)
            weights[item_index] = float(normalized_weight)

        context_payload.update(fit_metrics)
        context_payload["fit_status"] = "success"
        context_payload["fit_factor_count"] = len(valid_items)
        context_payload["sample_count"] = int(len(target_vector))
        context_payload["effective_training_date_count"] = int(
            training_frame.index.get_level_values(self._date_level(training_frame)).nunique()
        )
        context_payload["nonzero_weight_count"] = int(sum(abs(weight) > 1e-12 for weight in weights))
        context_payload["item_raw_weights"] = [float(value) for value in raw_weights]
        return weights, context_payload

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
        raw_weights: list[float],
    ) -> list[dict[str, Any]]:
        weighted_factors: list[dict[str, Any]] = []
        for item, weight, raw_weight in zip(selected_items, weights, raw_weights):
            factor_expr = item.get("factor", {}).get("qlib_expression")
            if not factor_expr or abs(float(weight)) <= 1e-12:
                continue
            weighted_factors.append(
                {
                    "factor_id": item.get("factor", {}).get("factor_id"),
                    "qlib_expression": factor_expr,
                    "recent_score": float(item.get("recent_score", 0.0)),
                    "recent_series": item.get("recent_series", []),
                    "recent_score_source": item.get("recent_score_source"),
                    "is_proxy_score": bool(item.get("is_proxy_score", False)),
                    "recent_ir": self._safe_float(item.get("recent_ir", 0.0)),
                    "recent_ir_source": item.get("recent_ir_source"),
                    "recent_std": self._safe_float(item.get("recent_std", 0.0)),
                    "raw_weight": float(raw_weight),
                    "weight": float(weight),
                }
            )
        return weighted_factors

    def _ensure_qlib_initialized(self, provider_uri: Path | None) -> bool:
        if self._qlib_initialized and self._D is not None:
            return True

        try:
            from qlib.data import D

            D.calendar(start_time="2005-01-01", end_time="2005-01-10")
            self._D = D
            self._qlib_initialized = True
            return True
        except Exception:
            pass

        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D

            resolved_provider = self._resolve_provider_uri(provider_uri)
            qlib.init(provider_uri=str(resolved_provider), region=REG_CN)
            self._provider_uri = resolved_provider
            self._D = D
            self._qlib_initialized = True
            return True
        except Exception:
            return False

    def _resolve_provider_uri(self, provider_uri: Path | None) -> Path:
        candidates: list[Path] = []
        if provider_uri:
            candidates.append(Path(provider_uri).expanduser())
        candidates.extend(
            [
                Path("~/.qlib/qlib_data/cn_data").expanduser(),
                Path("/home/batchcom/.qlib/qlib_data/cn_data"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("No Qlib data path found for regression allocator.")

    def _resolve_training_dates(
        self,
        *,
        selection_date: pd.Timestamp,
        fit_window_days: int,
        horizon_days: int,
    ) -> list[pd.Timestamp]:
        selection_ts = pd.Timestamp(selection_date)
        calendar_start = (selection_ts - pd.Timedelta(days=max((fit_window_days + horizon_days) * 12, 120))).strftime(
            "%Y-%m-%d"
        )
        calendar = self._D.calendar(
            start_time=calendar_start,
            end_time=selection_ts.strftime("%Y-%m-%d"),
        )
        all_dates = [pd.Timestamp(value) for value in calendar if pd.Timestamp(value) <= selection_ts]
        if len(all_dates) <= horizon_days:
            return []
        available_feature_dates = all_dates[:-horizon_days]
        return available_feature_dates[-fit_window_days:]

    def _load_training_frame(
        self,
        *,
        indexed_items: list[tuple[int, dict[str, Any]]],
        instrument: str,
        start_date: str,
        end_date: str,
        target_expression: str,
    ) -> tuple[pd.DataFrame | None, list[tuple[int, dict[str, Any]]], list[str], str | None]:
        expressions = [item["factor"]["qlib_expression"] for _, item in indexed_items] + [target_expression]
        instruments = self._D.instruments(instrument)
        try:
            frame = self._D.features(
                instruments,
                expressions,
                start_time=start_date,
                end_time=end_date,
            )
            return frame, indexed_items, [], None
        except Exception as exc:
            invalid_factor_ids: list[str] = []
            valid_items: list[tuple[int, dict[str, Any]]] = []
            for index, item in indexed_items:
                expression = item["factor"].get("qlib_expression")
                if not expression:
                    invalid_factor_ids.append(str(item.get("factor", {}).get("factor_id") or f"index_{index}"))
                    continue
                try:
                    single_frame = self._D.features(
                        instruments,
                        [expression, target_expression],
                        start_time=start_date,
                        end_time=end_date,
                    )
                    if single_frame is None or single_frame.empty:
                        invalid_factor_ids.append(str(item.get("factor", {}).get("factor_id") or f"index_{index}"))
                        continue
                    valid_items.append((index, item))
                except Exception:
                    invalid_factor_ids.append(str(item.get("factor", {}).get("factor_id") or f"index_{index}"))

            if not valid_items:
                return None, [], invalid_factor_ids, f"{type(exc).__name__}: {exc}"

            frame = self._D.features(
                instruments,
                [item["factor"]["qlib_expression"] for _, item in valid_items] + [target_expression],
                start_time=start_date,
                end_time=end_date,
            )
            return frame, valid_items, invalid_factor_ids, f"{type(exc).__name__}: {exc}"

    def _prepare_training_frame(
        self,
        *,
        factor_frame: pd.DataFrame,
        feature_names: list[str],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
        if factor_frame is None or factor_frame.empty:
            return None

        target_column = "target_return"
        prepared = factor_frame.copy()
        prepared.columns = feature_names + [target_column]
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        prepared = prepared.dropna()
        if prepared.empty:
            return None

        date_level = self._date_level(prepared)
        ranked_frames: list[pd.DataFrame] = []
        for _, daily_frame in prepared.groupby(level=date_level, sort=True):
            if len(daily_frame) < 2:
                continue
            ranked_daily = daily_frame.copy()
            ranked_daily["target_label"] = ranked_daily[target_column].rank(method="average", pct=True)
            if self.center_target:
                ranked_daily["target_label"] = ranked_daily["target_label"] - 0.5
            ranked_frames.append(ranked_daily)

        if not ranked_frames:
            return None

        training_frame = pd.concat(ranked_frames, axis=0).sort_index()
        feature_matrix = training_frame[feature_names].to_numpy(dtype=float)
        target_vector = training_frame["target_label"].to_numpy(dtype=float)
        return training_frame, feature_matrix, target_vector

    def _solve_coefficients(
        self,
        *,
        feature_matrix: np.ndarray,
        target_vector: np.ndarray,
        training_frame: pd.DataFrame,
    ) -> tuple[list[float], dict[str, Any]]:
        if feature_matrix.size == 0 or target_vector.size == 0:
            return [], {}

        feature_scale = np.std(feature_matrix, axis=0)
        feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1e-12), feature_scale, 1.0)
        scaled_matrix = feature_matrix / feature_scale
        if self.ridge_alpha > 0:
            ridge_matrix = np.sqrt(self.ridge_alpha) * np.eye(scaled_matrix.shape[1], dtype=float)
            solve_matrix = np.vstack([scaled_matrix, ridge_matrix])
            solve_target = np.concatenate([target_vector, np.zeros(scaled_matrix.shape[1], dtype=float)])
        else:
            solve_matrix = scaled_matrix
            solve_target = target_vector

        try:
            scaled_coefficients, *_ = np.linalg.lstsq(solve_matrix, solve_target, rcond=None)
        except Exception as exc:
            return [], {"fit_error": f"{type(exc).__name__}: {exc}"}

        raw_coefficients = scaled_coefficients / feature_scale
        raw_coefficients = np.where(np.isfinite(raw_coefficients), raw_coefficients, 0.0)
        coefficient_abs_sum = float(np.sum(np.abs(raw_coefficients)))
        if coefficient_abs_sum <= 1e-12:
            return [], {
                "feature_scale": [float(value) for value in feature_scale],
                "raw_weights": [float(value) for value in raw_coefficients],
            }

        normalized_coefficients = raw_coefficients / coefficient_abs_sum
        prediction = feature_matrix @ raw_coefficients
        mse = float(np.mean((prediction - target_vector) ** 2))
        centered_target = target_vector - float(np.mean(target_vector))
        target_var = float(np.sum(centered_target**2))
        r2 = None
        if target_var > 1e-12:
            r2 = float(1.0 - np.sum((prediction - target_vector) ** 2) / target_var)

        mean_rank_corr = self._mean_daily_rank_corr(training_frame, prediction)
        return [float(value) for value in normalized_coefficients], {
            "feature_scale": [float(value) for value in feature_scale],
            "raw_weights": [float(value) for value in raw_coefficients],
            "weight_normalization": "l1_abs",
            "fit_mse": mse,
            "fit_r2": r2,
            "fit_mean_daily_rank_corr": mean_rank_corr,
        }

    def _mean_daily_rank_corr(self, training_frame: pd.DataFrame, prediction: np.ndarray) -> float | None:
        evaluation_frame = training_frame[["target_label"]].copy()
        evaluation_frame["prediction"] = prediction
        date_level = self._date_level(evaluation_frame)
        daily_corrs: list[float] = []
        for _, daily_frame in evaluation_frame.groupby(level=date_level, sort=True):
            if len(daily_frame) < 2:
                continue
            corr = daily_frame["prediction"].corr(daily_frame["target_label"], method="spearman")
            if corr is None or pd.isna(corr):
                continue
            daily_corrs.append(float(corr))
        if not daily_corrs:
            return None
        return float(np.mean(daily_corrs))

    @staticmethod
    def _build_target_expression(horizon_days: int) -> str:
        return f"Ref($close, -{int(horizon_days) + 1})/Ref($close, -1) - 1"

    @staticmethod
    def _date_level(frame: pd.DataFrame) -> str | int:
        index_names = list(frame.index.names)
        if "datetime" in index_names:
            return "datetime"
        if index_names and index_names[0] is not None:
            return index_names[0]
        return 0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
