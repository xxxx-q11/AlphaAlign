"""Top-k excess return rolling factor metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TOPK_RETURN_PROFILE_NAME = "topk_return"


def compute_topk_return_daily_metrics(
    factor_frame: pd.DataFrame,
    *,
    feature_names: list[str],
    evaluation_dates: list[pd.Timestamp],
    benchmark_returns: dict[pd.Timestamp, float],
    top_k: int,
    score_multipliers: dict[str, float] | None = None,
) -> dict[str, dict[pd.Timestamp, dict[str, Any]]]:
    """Compute daily top-k return metrics for each factor column."""
    if factor_frame is None or factor_frame.empty or not feature_names or not evaluation_dates or top_k <= 0:
        return {}

    frame = factor_frame.reset_index()
    frame = frame[frame["datetime"].isin(evaluation_dates)]
    if frame.empty:
        return {}

    results = {feature_name: {} for feature_name in feature_names}
    normalized_top_k = max(int(top_k), 1)
    multiplier_vector = None
    if score_multipliers:
        multiplier_vector = np.asarray(
            [float(score_multipliers.get(feature_name, 1.0)) for feature_name in feature_names],
            dtype=float,
        )

    for eval_date, day_df in frame.groupby("datetime", sort=True):
        benchmark_return = benchmark_returns.get(pd.Timestamp(eval_date))
        if benchmark_return is None:
            continue

        factor_matrix = day_df[feature_names].to_numpy(dtype=float, copy=False)
        if multiplier_vector is not None:
            factor_matrix = factor_matrix * multiplier_vector
        next_return = day_df["next_return"].to_numpy(dtype=float, copy=False)
        topk_returns, topk_counts = _batch_topk_mean_returns(
            factor_matrix,
            next_return,
            top_k=normalized_top_k,
        )

        eval_ts = pd.Timestamp(eval_date)
        for feature_index, feature_name in enumerate(feature_names):
            topk_return = topk_returns[feature_index]
            topk_count = int(topk_counts[feature_index])
            if topk_count <= 0 or not np.isfinite(topk_return):
                continue

            results[feature_name][eval_ts] = {
                "date": eval_ts.strftime("%Y-%m-%d"),
                "topk_return": float(topk_return),
                "benchmark_return": float(benchmark_return),
                "excess_return": float(topk_return - benchmark_return),
                "universe_count": topk_count,
            }

    return {feature_name: payload for feature_name, payload in results.items() if payload}


def build_topk_return_snapshot(
    daily_metrics: dict[pd.Timestamp, dict[str, Any]],
    *,
    evaluation_dates: list[pd.Timestamp],
) -> dict[str, Any] | None:
    """Build a selector-facing rolling snapshot from cached daily top-k metrics."""
    if not daily_metrics or not evaluation_dates:
        return None

    ordered_payload = [daily_metrics[eval_date] for eval_date in evaluation_dates if eval_date in daily_metrics]
    if not ordered_payload:
        return None

    return {
        "recent_topk_eval_dates": [item["date"] for item in ordered_payload],
        "recent_topk_returns": [float(item["topk_return"]) for item in ordered_payload],
        "recent_topk_benchmark_returns": [float(item["benchmark_return"]) for item in ordered_payload],
        "recent_topk_excess_returns": [float(item["excess_return"]) for item in ordered_payload],
        "recent_topk_universe_counts": [int(item["universe_count"]) for item in ordered_payload],
    }


def _batch_topk_mean_returns(
    matrix: np.ndarray,
    next_return: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k mean returns for many factor columns."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    aligned_return = np.broadcast_to(next_return[:, None], matrix.shape)
    valid_mask = np.isfinite(matrix) & np.isfinite(aligned_return)
    valid_counts = valid_mask.sum(axis=0).astype(int)

    topk_returns = np.full(matrix.shape[1], np.nan, dtype=float)
    if not np.any(valid_counts > 0):
        return topk_returns, valid_counts

    safe_scores = np.where(valid_mask, matrix, -np.inf)
    use_all_indices = np.where((valid_counts > 0) & (valid_counts <= top_k))[0]
    for feature_index in use_all_indices:
        topk_returns[feature_index] = float(
            np.mean(next_return[valid_mask[:, feature_index]])
        )

    partition_indices = np.where(valid_counts > top_k)[0]
    if partition_indices.size > 0:
        partition_scores = safe_scores[:, partition_indices]
        partition_order = np.argpartition(
            partition_scores,
            kth=partition_scores.shape[0] - top_k,
            axis=0,
        )
        top_indices = partition_order[-top_k:, :]
        partition_returns = np.broadcast_to(next_return[:, None], partition_scores.shape)
        gathered_returns = np.take_along_axis(partition_returns, top_indices, axis=0)
        topk_returns[partition_indices] = gathered_returns.mean(axis=0)

    topk_counts = np.minimum(valid_counts, top_k)
    return topk_returns, topk_counts
