"""Cross-sectional rolling factor metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CROSS_SECTIONAL_PROFILE_NAME = "cross_sectional"


def compute_cross_sectional_daily_metrics(
    factor_frame: pd.DataFrame,
    *,
    feature_names: list[str],
    evaluation_dates: list[pd.Timestamp],
) -> dict[str, dict[pd.Timestamp, dict[str, Any]]]:
    """Compute daily IC and Rank IC for each factor column."""
    if factor_frame is None or factor_frame.empty or not feature_names or not evaluation_dates:
        return {}

    frame = factor_frame.reset_index()
    frame = frame[frame["datetime"].isin(evaluation_dates)]
    if frame.empty:
        return {}

    results = {feature_name: {} for feature_name in feature_names}
    for eval_date, day_df in frame.groupby("datetime", sort=True):
        factor_matrix = day_df[feature_names].to_numpy(dtype=float, copy=False)
        next_return = day_df["next_return"].to_numpy(dtype=float, copy=False)

        ic_values, universe_counts = _batch_corr(factor_matrix, next_return)

        ranked_matrix = pd.DataFrame(factor_matrix, columns=feature_names).rank(
            method="average",
            na_option="keep",
        )
        ranked_returns = pd.Series(next_return).rank(method="average")
        rank_ic_values, _ = _batch_corr(
            ranked_matrix.to_numpy(dtype=float, copy=False),
            ranked_returns.to_numpy(dtype=float, copy=False),
        )

        eval_ts = pd.Timestamp(eval_date)
        for feature_index, feature_name in enumerate(feature_names):
            universe_count = int(universe_counts[feature_index])
            if universe_count < 3:
                continue

            ic_value = ic_values[feature_index]
            rank_ic_value = rank_ic_values[feature_index]
            if not np.isfinite(ic_value) and not np.isfinite(rank_ic_value):
                continue

            results[feature_name][eval_ts] = {
                "date": eval_ts.strftime("%Y-%m-%d"),
                "ic": 0.0 if not np.isfinite(ic_value) else float(ic_value),
                "rank_ic": 0.0 if not np.isfinite(rank_ic_value) else float(rank_ic_value),
                "universe_count": universe_count,
            }

    return {feature_name: payload for feature_name, payload in results.items() if payload}


def build_cross_sectional_snapshot(
    daily_metrics: dict[pd.Timestamp, dict[str, Any]],
    *,
    evaluation_dates: list[pd.Timestamp],
) -> dict[str, Any] | None:
    """Build a selector-facing rolling snapshot from cached daily IC metrics."""
    if not daily_metrics or not evaluation_dates:
        return None

    ordered_payload = [daily_metrics[eval_date] for eval_date in evaluation_dates if eval_date in daily_metrics]
    if not ordered_payload:
        return None

    recent_ics = [float(item["ic"]) for item in ordered_payload]
    recent_rank_ics = [float(item["rank_ic"]) for item in ordered_payload]
    return {
        "recent_ic_eval_dates": [item["date"] for item in ordered_payload],
        "recent_ics": recent_ics,
        "recent_rank_ics": recent_rank_ics,
        "recent_ic_universe_counts": [int(item["universe_count"]) for item in ordered_payload],
        "recent_ic_mean": float(np.mean(recent_ics)) if recent_ics else 0.0,
        "recent_rank_ic_mean": float(np.mean(recent_rank_ics)) if recent_rank_ics else 0.0,
        "recent_icir": _safe_information_ratio(recent_ics),
        "recent_rank_icir": _safe_information_ratio(recent_rank_ics),
    }


def _batch_corr(matrix: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute correlation between many columns and one vector in a masked batch."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    aligned_vector = np.broadcast_to(vector[:, None], matrix.shape)
    mask = np.isfinite(matrix) & np.isfinite(aligned_vector)
    counts = mask.sum(axis=0).astype(int)

    masked_matrix = np.where(mask, matrix, 0.0)
    masked_vector = np.where(mask, aligned_vector, 0.0)

    sum_x = masked_matrix.sum(axis=0)
    sum_y = masked_vector.sum(axis=0)
    sum_xx = np.square(masked_matrix).sum(axis=0)
    sum_yy = np.square(masked_vector).sum(axis=0)
    sum_xy = (masked_matrix * masked_vector).sum(axis=0)

    numerator = counts * sum_xy - sum_x * sum_y
    denominator = (counts * sum_xx - np.square(sum_x)) * (counts * sum_yy - np.square(sum_y))

    corr = np.full(matrix.shape[1], np.nan, dtype=float)
    valid = (counts >= 3) & (denominator > 1e-24)
    corr[valid] = numerator[valid] / np.sqrt(denominator[valid])
    return corr, counts


def _safe_information_ratio(values: list[float]) -> float:
    if not values:
        return 0.0

    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    if std_value > 1e-12:
        return float(mean_value / std_value)
    if mean_value > 0:
        return float("inf")
    if mean_value < 0:
        return float("-inf")
    return 0.0
