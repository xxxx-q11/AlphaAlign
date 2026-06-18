"""Reusable Qlib backtest service for linear weighting outputs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

BACKTEST_IMPORT_ERROR: Exception | None = None

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import qlib
    from Strategy import DynamicWindowTopkStrategy, NewsAwareTopkStrategy
    from Strategy.Allocator.mwu_allocator import MWUAllocator
    from Strategy.Model.lgbm_window_trainer import LGBMWindowTrainer
    from Strategy.qlib.model_window_signal import ModelWindowSignal
    from Strategy.Allocator.normalized_score_allocator import NormalizedScoreAllocator
    from Strategy.Allocator.regression_allocator import RegressionAllocator
    from Strategy.Allocator.score_ir_allocator import ScoreIRAllocator
    from Strategy.Selector import AFFRecentPerformanceSelector, MWUAllExpertSelector, RecentPerformanceSelector
    from Strategy.SignalGenerator.expression_signal_generator import ExpressionSignalGenerator
    from Strategy.SignalGenerator.normalized_panel_signal_generator import NormalizedPanelSignalGenerator
    from Strategy.base.base_allocator import BaseAllocator
    from Strategy.base.base_signal_generator import BaseSignalGenerator
    from Strategy.base.base_selector import BaseSelector
    from Strategy.news import NewsReviewService
    from Strategy.runtime import BacktestEventLogger
    from Strategy.runtime.rebalance_scheduler import RebalanceScheduler
    from Strategy.runtime.window_context import WindowContext
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.config import REG_CN
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data import D
except Exception as exc:  # pragma: no cover - environment dependent
    BACKTEST_IMPORT_ERROR = exc
    plt = None
    np = None
    pd = None
    qlib = None
    DynamicWindowTopkStrategy = None
    NewsAwareTopkStrategy = None
    MWUAllocator = None
    LGBMWindowTrainer = None
    ModelWindowSignal = None
    NormalizedScoreAllocator = None
    RegressionAllocator = None
    ScoreIRAllocator = None
    AFFRecentPerformanceSelector = None
    MWUAllExpertSelector = None
    RecentPerformanceSelector = None
    ExpressionSignalGenerator = None
    NormalizedPanelSignalGenerator = None
    BaseAllocator = Any
    BaseSignalGenerator = Any
    BaseSelector = Any
    NewsReviewService = Any
    BacktestEventLogger = Any
    RebalanceScheduler = None
    WindowContext = None
    backtest = None
    SimulatorExecutor = None
    REG_CN = None
    risk_analysis = None
    D = None


DEFAULT_BACKTEST_CONFIG: dict[str, Any] = {
    "provider_uri": None,
    "start_date": "2025-01-01",
    "end_date": "2025-10-01",
    "instrument": "csi300",
    "benchmark": "csi300",
    "topk": 50,
    "n_drop": 5,
    "portfolio_mode": "fixed_horizon",
    "holding_period_days": 10,
    "daily_buy_topk": 5,
    "factor_eval_top_k": 5,
    "top_n": None,
    "signal_mode": "rolling",
    "rebalance_window_days": 5,
    "window_days": 20,
    "model_train_window_days": 120,
    "model_label_horizon_days": 10,
    "model_label_expression": "Ref($close, -11)/Ref($close, -1) - 1",
    "recent_perf_candidate_limit": None,
    "recent_perf_batch_size": 8,
    "selector_mode": "recent",
    "selector_score_threshold": 0.02,
    "selector_ir_threshold": 0.2,
    "mwu_learning_rate": 0.15,
    "mwu_reward_cap": 0.05,
    "mwu_explore_rate": 0.03,
    "mwu_max_weight": 0.15,
    "mwu_use_dual_experts": True,
    "mwu_enable_tail_switch": False,
    "mwu_tail_switch_mode": "hard",
    "mwu_direction_rank_ic_threshold": 0.0,
    "mwu_direction_top_bottom_k": 5,
    "mwu_direction_validation_days": 30,
    "mwu_direction_spread_threshold": 0.0,
    "mwu_bayes_half_life": 15.0,
    "mwu_bayes_prior_strength": 10.0,
    "mwu_bayes_hurdle": 0.0,
    "weighting_method": None,
    "enable_news_review": False,
    "news_data_path": None,
    "news_batch_size": 10,
    "news_candidate_pool_multiplier": 3,
    "news_llm_config_path": None,
    "news_confidence_threshold": 0.3,
    "account": 100000000,
    "return_expression": "Ref($close, -11)/Ref($close, -1) - 1",
    "exchange_kwargs": {
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}


def _resolve_provider_uri(provider_uri: str | None) -> Path:
    candidates = []
    if provider_uri:
        candidates.append(Path(provider_uri).expanduser())
    candidates.extend(
        [
            Path("~/.qlib/qlib_data/cn_data").expanduser(),
            Path("/home/batchcom/.qlib/qlib_data/cn_data"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Qlib data path found. Please pass provider_uri explicitly.")


def _load_weighting_result(weighting_path: Path | None, data_dir: Path) -> tuple[dict[str, Any], Path]:
    if weighting_path is None:
        history_dir = data_dir / "weighting_history"
        candidates = sorted(history_dir.glob("weighting_*.json"))
        if not candidates:
            raise FileNotFoundError("No weighting history found under data/weighting_history")
        weighting_path = candidates[-1]

    with open(weighting_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    weighting_result = payload.get("weighting_result", payload)
    if not isinstance(weighting_result, dict):
        raise ValueError(f"Invalid weighting payload: {weighting_path}")
    return weighting_result, weighting_path


def _load_factor_library(data_dir: Path) -> list[dict[str, Any]]:
    factor_library_path = data_dir / "factor_library.json"
    if not factor_library_path.exists():
        raise FileNotFoundError(f"factor_library.json is missing under {data_dir}")

    with open(factor_library_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"Invalid factor_library payload: {factor_library_path}")
    return payload


def _benchmark_code(benchmark: str | None) -> str:
    mapping = {
        "csi300": "SH000300",
        "hs300": "SH000300",
        "csi500": "SH000905",
        "zz500": "SH000905",
    }
    if not benchmark:
        return "SH000300"
    return mapping.get(benchmark.lower(), benchmark)


def _risk_analysis_to_summary(risk_df: pd.DataFrame) -> dict[str, float]:
    return {
        key: float(risk_df.loc[key, "risk"])
        for key in ("mean", "std", "annualized_return", "information_ratio", "max_drawdown")
    }


def _cost_series(report_df: pd.DataFrame) -> pd.Series:
    if "cost" not in report_df.columns:
        return pd.Series(0.0, index=report_df.index)
    return report_df["cost"].fillna(0.0)


def _strategy_return_with_cost(report_df: pd.DataFrame, cost_series: pd.Series) -> pd.Series:
    net_return = report_df["return"] - cost_series
    if "account" not in report_df.columns or report_df.empty:
        return net_return

    account_return = report_df["account"].astype(float).pct_change()
    account_return.iloc[0] = net_return.iloc[0]
    return account_return.fillna(net_return)


def _strategy_nav_summary(strategy_return: pd.Series) -> dict[str, float]:
    final_nav = float((1.0 + strategy_return.fillna(0.0)).prod())
    return {
        "final_nav": final_nav,
        "total_return": final_nav - 1.0,
        "cagr_like_return": float(final_nav ** (238.0 / len(strategy_return)) - 1.0),
    }


def _summarize_risk(report_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    cost_series = _cost_series(report_df)
    strategy_return = _strategy_return_with_cost(report_df, cost_series)
    strategy_risk = risk_analysis(strategy_return, freq="day")
    excess_risk = risk_analysis(report_df["return"] - report_df["bench"], freq="day")
    excess_with_cost_risk = risk_analysis(report_df["return"] - report_df["bench"] - cost_series, freq="day")
    strategy_summary = _risk_analysis_to_summary(strategy_risk)
    strategy_summary.update(_strategy_nav_summary(strategy_return))

    return {
        "strategy": strategy_summary,
        "excess_return_without_cost": _risk_analysis_to_summary(excess_risk),
        "excess_return_with_cost": _risk_analysis_to_summary(excess_with_cost_risk),
    }


def _stringify_position_keys(positions: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in positions.items()}


def _build_selected_items_by_date(rolling_windows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for window_payload in rolling_windows:
        selection_date = str(window_payload.get("selection_date") or window_payload.get("holding_start") or "unknown")
        selected_items = window_payload.get("selected_items", []) or []
        grouped[selection_date] = selected_items
    return grouped


def _nav_from_returns(return_series: pd.Series) -> pd.Series:
    nav = (1 + return_series.fillna(0.0)).cumprod()
    if nav.empty:
        return nav
    return nav / float(nav.iloc[0])


def _plot_net_value_curve(report_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = report_df.copy()
    if "datetime" in plot_df.columns:
        plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
        plot_df = plot_df.set_index("datetime")
    else:
        plot_df.index = pd.to_datetime(plot_df.index)

    cost_series = plot_df["cost"].fillna(0.0) if "cost" in plot_df.columns else pd.Series(0.0, index=plot_df.index)
    strategy_nav = _nav_from_returns(plot_df["return"])
    benchmark_nav = _nav_from_returns(plot_df["bench"])
    strategy_with_cost_nav = _nav_from_returns(plot_df["return"] - cost_series)
    excess_with_cost_nav = _nav_from_returns(plot_df["return"] - plot_df["bench"] - cost_series)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy_nav.index, strategy_nav, label="Strategy NAV (Gross)", linewidth=2.0)
    ax.plot(
        strategy_with_cost_nav.index,
        strategy_with_cost_nav,
        label="Strategy NAV (Net of Cost)",
        linewidth=1.8,
    )
    ax.plot(benchmark_nav.index, benchmark_nav, label="Benchmark NAV", linewidth=1.8)
    ax.plot(
        excess_with_cost_nav.index,
        excess_with_cost_nav,
        label="Excess NAV vs Benchmark (Net of Cost)",
        linewidth=1.8,
    )
    ax.set_title("Net Value Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _normalize_signal_frame(signal: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(signal, pd.Series):
        signal_df = signal.to_frame("score")
    else:
        signal_df = signal.copy()

    if "score" not in signal_df.columns:
        if len(signal_df.columns) != 1:
            raise ValueError("Signal dataframe must contain a 'score' column or exactly one value column.")
        signal_df = signal_df.rename(columns={signal_df.columns[0]: "score"})

    if not isinstance(signal_df.index, pd.MultiIndex):
        raise ValueError("Signal dataframe index must be a MultiIndex containing datetime and instrument.")

    index_names = list(signal_df.index.names)
    if "datetime" not in index_names or "instrument" not in index_names:
        inferred_names = _infer_signal_index_names(signal_df.index)
        if inferred_names is not None:
            signal_df.index = signal_df.index.set_names(inferred_names)
            index_names = list(signal_df.index.names)

    if "datetime" not in index_names or "instrument" not in index_names:
        raise ValueError(
            f"Signal dataframe index names must include 'datetime' and 'instrument', got {index_names!r}."
        )

    if index_names != ["datetime", "instrument"]:
        signal_df = signal_df.reorder_levels(["datetime", "instrument"]).sort_index()

    signal_df.index = _coerce_signal_multiindex(signal_df.index)
    return signal_df[["score"]]


def _infer_signal_index_names(index: pd.MultiIndex) -> list[str] | None:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        return None

    level0 = index.get_level_values(0)
    level1 = index.get_level_values(1)
    level0_is_datetime = _looks_like_datetime_level(level0)
    level1_is_datetime = _looks_like_datetime_level(level1)

    if level0_is_datetime and not level1_is_datetime:
        return ["datetime", "instrument"]
    if level1_is_datetime and not level0_is_datetime:
        return ["instrument", "datetime"]
    return None


def _looks_like_datetime_level(level_values: pd.Index, sample_size: int = 50) -> bool:
    if pd.api.types.is_datetime64_any_dtype(level_values):
        return True

    sample = pd.Index(level_values).dropna().unique()[:sample_size]
    if len(sample) == 0:
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    parsed_ratio = float(parsed.notna().sum()) / float(len(sample))
    return parsed_ratio >= 0.8


def _coerce_signal_multiindex(index: pd.MultiIndex) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        return index

    index = index.set_names(["datetime", "instrument"])
    datetime_level = pd.to_datetime(index.get_level_values("datetime"), errors="coerce")
    instrument_level = index.get_level_values("instrument")
    return pd.MultiIndex.from_arrays(
        [datetime_level, instrument_level],
        names=["datetime", "instrument"],
    )


def _calculate_test_signal_metrics(
    signal: pd.DataFrame | pd.Series | None,
    *,
    instrument: str,
    label_expression: str,
) -> dict[str, Any] | None:
    if signal is None:
        return None

    signal_df = _normalize_signal_frame(signal).dropna()
    if signal_df.empty:
        return None

    signal_start = pd.Timestamp(signal_df.index.get_level_values("datetime").min()).strftime("%Y-%m-%d")
    signal_end = pd.Timestamp(signal_df.index.get_level_values("datetime").max()).strftime("%Y-%m-%d")

    label_df = D.features(
        D.instruments(instrument),
        [label_expression],
        start_time=signal_start,
        end_time=signal_end,
    )
    label_df.columns = ["label"]
    merged = signal_df.join(label_df, how="inner").dropna()
    if merged.empty:
        return None

    daily_metrics = (
        merged.groupby(level="datetime")
        .apply(
            lambda frame: pd.Series(
                {
                    "ic": frame["score"].corr(frame["label"], method="pearson"),
                    "rank_ic": frame["score"].corr(frame["label"], method="spearman"),
                    "sample_count": int(len(frame)),
                }
            )
        )
        .dropna(subset=["ic", "rank_ic"])
    )
    if daily_metrics.empty:
        return None

    ic_mean = float(daily_metrics["ic"].mean())
    rank_ic_mean = float(daily_metrics["rank_ic"].mean())
    ic_std = float(daily_metrics["ic"].std(ddof=0))
    rank_ic_std = float(daily_metrics["rank_ic"].std(ddof=0))

    return {
        "label_expression": label_expression,
        "signal_start": signal_start,
        "signal_end": signal_end,
        "signal_rows_used": int(len(merged)),
        "days": int(len(daily_metrics)),
        "mean_sample_count": float(daily_metrics["sample_count"].mean()),
        "ic": ic_mean,
        "rank_ic": rank_ic_mean,
        "icir": _safe_ratio(ic_mean, ic_std),
        "rank_icir": _safe_ratio(rank_ic_mean, rank_ic_std),
    }


def _calculate_signal_direction_metrics(
    signal: pd.DataFrame | pd.Series | None,
    *,
    instrument: str,
    label_expression: str,
    top_bottom_k: int,
) -> dict[str, Any] | None:
    if signal is None:
        return None

    signal_df = _normalize_signal_frame(signal).dropna()
    if signal_df.empty:
        return None

    signal_start = pd.Timestamp(signal_df.index.get_level_values("datetime").min()).strftime("%Y-%m-%d")
    signal_end = pd.Timestamp(signal_df.index.get_level_values("datetime").max()).strftime("%Y-%m-%d")

    label_df = D.features(
        D.instruments(instrument),
        [label_expression],
        start_time=signal_start,
        end_time=signal_end,
    )
    label_df.columns = ["label"]
    merged = signal_df.join(label_df, how="inner").dropna()
    if merged.empty:
        return None

    effective_top_bottom_k = max(int(top_bottom_k), 1)

    def _daily_direction_stats(frame: pd.DataFrame) -> pd.Series:
        bucket_size = min(effective_top_bottom_k, len(frame) // 2)
        if bucket_size <= 0:
            return pd.Series(
                {
                    "ic": np.nan,
                    "rank_ic": np.nan,
                    "sample_count": int(len(frame)),
                    "bucket_size": 0,
                    "top_mean": np.nan,
                    "bottom_mean": np.nan,
                    "top_bottom_spread": np.nan,
                }
            )

        sorted_frame = frame.sort_values("score", ascending=False)
        top_mean = float(sorted_frame["label"].iloc[:bucket_size].mean())
        bottom_mean = float(sorted_frame["label"].iloc[-bucket_size:].mean())
        return pd.Series(
            {
                "ic": frame["score"].corr(frame["label"], method="pearson"),
                "rank_ic": frame["score"].corr(frame["label"], method="spearman"),
                "sample_count": int(len(frame)),
                "bucket_size": int(bucket_size),
                "top_mean": top_mean,
                "bottom_mean": bottom_mean,
                "top_bottom_spread": float(top_mean - bottom_mean),
            }
        )

    daily_metrics = merged.groupby(level="datetime").apply(_daily_direction_stats)
    daily_metrics = daily_metrics.dropna(subset=["rank_ic", "top_bottom_spread"])
    if daily_metrics.empty:
        return None

    rank_ic_mean = float(daily_metrics["rank_ic"].mean())
    rank_ic_std = float(daily_metrics["rank_ic"].std(ddof=0))
    spread_mean = float(daily_metrics["top_bottom_spread"].mean())
    spread_std = float(daily_metrics["top_bottom_spread"].std(ddof=0))

    return {
        "label_expression": label_expression,
        "signal_start": signal_start,
        "signal_end": signal_end,
        "signal_rows_used": int(len(merged)),
        "days": int(len(daily_metrics)),
        "mean_sample_count": float(daily_metrics["sample_count"].mean()),
        "top_bottom_k": effective_top_bottom_k,
        "mean_bucket_size": float(daily_metrics["bucket_size"].mean()),
        "rank_ic": rank_ic_mean,
        "rank_icir": _safe_ratio(rank_ic_mean, rank_ic_std),
        "top_mean": float(daily_metrics["top_mean"].mean()),
        "bottom_mean": float(daily_metrics["bottom_mean"].mean()),
        "top_bottom_spread": spread_mean,
        "top_bottom_spread_ir": _safe_ratio(spread_mean, spread_std),
        "daily_eval_dates": [
            pd.Timestamp(value).strftime("%Y-%m-%d")
            for value in daily_metrics.index.tolist()
        ],
        "daily_rank_ics": [float(value) for value in daily_metrics["rank_ic"].tolist()],
        "daily_top_means": [float(value) for value in daily_metrics["top_mean"].tolist()],
        "daily_bottom_means": [float(value) for value in daily_metrics["bottom_mean"].tolist()],
        "daily_top_bottom_spreads": [
            float(value) for value in daily_metrics["top_bottom_spread"].tolist()
        ],
    }


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def _calculate_bayesian_direction_posterior(
    spreads: list[Any] | tuple[Any, ...],
    *,
    half_life: float,
    prior_strength: float,
    hurdle: float,
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(value)
            for value in spreads
            if value is not None and np.isfinite(float(value))
        ],
        dtype=float,
    )
    if values.size <= 0:
        return {
            "status": "no_observations",
            "observation_count": 0,
            "p_head": 0.5,
            "p_tail": 0.5,
            "p_none": 0.0,
            "direction_multiplier": 0.0,
            "head_fraction": 0.5,
        }

    half_life_value = max(float(half_life), 1e-6)
    ages = np.arange(values.size - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / half_life_value)
    weight_sum = float(weights.sum())
    weighted_mean = float(np.sum(weights * values) / weight_sum)
    weighted_var = float(np.sum(weights * np.square(values - weighted_mean)) / weight_sum)

    if values.size <= 1:
        sigma = max(abs(weighted_mean), float(hurdle), 0.01)
    else:
        sigma = max(math.sqrt(max(weighted_var, 0.0)), 1e-4)

    prior = max(float(prior_strength), 0.0)
    posterior_precision = max(prior + weight_sum, 1e-12)
    posterior_mean = float((weight_sum * weighted_mean) / posterior_precision)
    posterior_std = float(sigma / math.sqrt(posterior_precision))
    hurdle_value = max(float(hurdle), 0.0)

    p_head = _normal_cdf((posterior_mean - hurdle_value) / posterior_std)
    p_tail = _normal_cdf((-hurdle_value - posterior_mean) / posterior_std)
    p_head = min(max(float(p_head), 0.0), 1.0)
    p_tail = min(max(float(p_tail), 0.0), 1.0)
    p_none = min(max(float(1.0 - p_head - p_tail), 0.0), 1.0)
    direction_multiplier = min(max(float(p_head - p_tail), -1.0), 1.0)
    head_fraction = min(max(float(0.5 * (1.0 + direction_multiplier)), 0.0), 1.0)
    effective_sample_size = float(weight_sum**2 / max(float(np.sum(np.square(weights))), 1e-12))

    return {
        "status": "success",
        "observation_count": int(values.size),
        "half_life": half_life_value,
        "prior_strength": prior,
        "hurdle": hurdle_value,
        "weighted_mean": weighted_mean,
        "weighted_std": sigma,
        "weight_sum": weight_sum,
        "effective_sample_size": effective_sample_size,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "p_head": p_head,
        "p_tail": p_tail,
        "p_none": p_none,
        "direction_multiplier": direction_multiplier,
        "head_fraction": head_fraction,
    }


def _pop_next_unselected(candidates: list[Any], selected: set[Any]) -> Any | None:
    while candidates:
        candidate = candidates.pop(0)
        if candidate in selected:
            continue
        selected.add(candidate)
        return candidate
    return None


def _apply_bayesian_head_tail_rank_mix(
    signal: pd.DataFrame | pd.Series,
    *,
    direction_multiplier: float,
    daily_buy_topk: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal_df = _normalize_signal_frame(signal).dropna()
    if signal_df.empty:
        return signal_df, {"status": "empty_signal"}

    head_fraction = min(max(float(0.5 * (1.0 + direction_multiplier)), 0.0), 1.0)
    buy_slots = max(int(daily_buy_topk), 1)
    mixed_frames: list[pd.DataFrame] = []
    daily_slot_counts: list[dict[str, Any]] = []
    head_slot_carry = 0.0

    for raw_date, group in signal_df.groupby(level="datetime", sort=True):
        ranked_group = group.sort_values("score", ascending=False)
        group_size = int(len(ranked_group))
        if group_size <= 0:
            continue

        effective_buy_slots = min(buy_slots, group_size)
        head_slot_carry += effective_buy_slots * head_fraction
        head_slots = int(math.floor(head_slot_carry + 1e-12))
        head_slots = min(max(head_slots, 0), effective_buy_slots)
        head_slot_carry -= head_slots
        tail_slots = effective_buy_slots - head_slots

        head_queue = list(ranked_group.index)
        tail_queue = list(reversed(head_queue))
        selected: set[Any] = set()
        ordered_index: list[Any] = []

        for _ in range(head_slots):
            candidate = _pop_next_unselected(head_queue, selected)
            if candidate is not None:
                ordered_index.append(candidate)
        for _ in range(tail_slots):
            candidate = _pop_next_unselected(tail_queue, selected)
            if candidate is not None:
                ordered_index.append(candidate)

        head_used = head_slots
        for target_size in range(len(ordered_index) + 1, group_size + 1):
            desired_head_used = int(math.floor(target_size * head_fraction + 0.5))
            prefer_head = head_used < desired_head_used
            if prefer_head:
                candidate = _pop_next_unselected(head_queue, selected)
                if candidate is not None:
                    head_used += 1
                else:
                    candidate = _pop_next_unselected(tail_queue, selected)
            else:
                candidate = _pop_next_unselected(tail_queue, selected)
                if candidate is None:
                    candidate = _pop_next_unselected(head_queue, selected)
                    if candidate is not None:
                        head_used += 1
            if candidate is not None:
                ordered_index.append(candidate)

        if len(ordered_index) < group_size:
            for candidate in ranked_group.index:
                if candidate not in selected:
                    ordered_index.append(candidate)
                    selected.add(candidate)

        if group_size == 1:
            mixed_scores = np.asarray([1.0], dtype=float)
        else:
            mixed_scores = np.linspace(1.0, -1.0, num=group_size, dtype=float)
        mixed_index = pd.MultiIndex.from_tuples(ordered_index, names=signal_df.index.names)
        mixed_frames.append(pd.DataFrame({"score": mixed_scores}, index=mixed_index))
        daily_slot_counts.append(
            {
                "date": pd.Timestamp(raw_date).strftime("%Y-%m-%d"),
                "head_slots": int(head_slots),
                "tail_slots": int(tail_slots),
                "buy_slots": int(effective_buy_slots),
            }
        )

    mixed_signal = pd.concat(mixed_frames).sort_index() if mixed_frames else signal_df
    mean_head_slots = (
        float(np.mean([item["head_slots"] for item in daily_slot_counts]))
        if daily_slot_counts
        else 0.0
    )
    mean_tail_slots = (
        float(np.mean([item["tail_slots"] for item in daily_slot_counts]))
        if daily_slot_counts
        else 0.0
    )
    return mixed_signal, {
        "status": "success",
        "direction_multiplier": float(direction_multiplier),
        "head_fraction": head_fraction,
        "daily_buy_topk": int(buy_slots),
        "mean_head_slots": mean_head_slots,
        "mean_tail_slots": mean_tail_slots,
        "daily_slot_counts": daily_slot_counts,
    }


def _resolve_recent_validation_range(
    *,
    reward_start: str | pd.Timestamp,
    reward_end: str | pd.Timestamp,
    validation_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, int] | None:
    start_ts = pd.Timestamp(reward_start)
    end_ts = pd.Timestamp(reward_end)
    if end_ts < start_ts:
        return None

    calendar = D.calendar(
        start_time=start_ts.strftime("%Y-%m-%d"),
        end_time=end_ts.strftime("%Y-%m-%d"),
    )
    trading_dates = [
        pd.Timestamp(value)
        for value in calendar
        if start_ts <= pd.Timestamp(value) <= end_ts
    ]
    if not trading_dates:
        return None

    effective_days = max(int(validation_days), 1)
    validation_dates = trading_dates[-effective_days:]
    return validation_dates[0], validation_dates[-1], len(validation_dates)


def _filter_selected_items_before_date(
    selected_items: list[dict[str, Any]],
    *,
    cutoff_date: str | pd.Timestamp,
    min_history_days: int = 5,
) -> list[dict[str, Any]]:
    cutoff_ts = pd.Timestamp(cutoff_date)
    filtered_items: list[dict[str, Any]] = []

    for item in selected_items:
        raw_dates = item.get("recent_eval_dates", []) or []
        raw_series = item.get("recent_series", []) or []
        if not raw_dates or not raw_series:
            continue

        aligned_length = min(len(raw_dates), len(raw_series))
        if aligned_length <= 0:
            continue

        dates = list(raw_dates)[-aligned_length:]
        series = list(raw_series)[-aligned_length:]
        topk_returns = item.get("recent_topk_returns", []) or []
        benchmark_returns = item.get("recent_benchmark_returns", []) or []
        aligned_topk_returns = list(topk_returns)[-aligned_length:] if len(topk_returns) >= aligned_length else []
        aligned_benchmark_returns = (
            list(benchmark_returns)[-aligned_length:] if len(benchmark_returns) >= aligned_length else []
        )

        keep_offsets = [
            offset
            for offset, raw_date in enumerate(dates)
            if pd.Timestamp(raw_date) < cutoff_ts
        ]
        if len(keep_offsets) < int(min_history_days):
            continue

        kept_series = [float(series[offset]) for offset in keep_offsets]
        filtered_item = deepcopy(item)
        filtered_item["recent_eval_dates"] = [str(dates[offset]) for offset in keep_offsets]
        filtered_item["recent_series"] = kept_series
        if aligned_topk_returns:
            filtered_item["recent_topk_returns"] = [float(aligned_topk_returns[offset]) for offset in keep_offsets]
        if aligned_benchmark_returns:
            filtered_item["recent_benchmark_returns"] = [
                float(aligned_benchmark_returns[offset]) for offset in keep_offsets
            ]

        recent_score = float(np.mean(kept_series))
        recent_std = float(np.std(np.asarray(kept_series, dtype=float)))
        filtered_item["recent_score"] = recent_score
        filtered_item["recent_std"] = recent_std
        filtered_item["recent_ir"] = _safe_ratio(recent_score, recent_std)
        filtered_items.append(filtered_item)

    return filtered_items


def _previous_trading_date(target_date: str | pd.Timestamp) -> pd.Timestamp:
    target_ts = pd.Timestamp(target_date)
    calendar_start = (target_ts - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    calendar = D.calendar(start_time=calendar_start, end_time=target_ts.strftime("%Y-%m-%d"))
    previous_dates = [pd.Timestamp(value) for value in calendar if pd.Timestamp(value) < target_ts]
    if not previous_dates:
        raise ValueError(f"No previous trading date found before {target_ts.strftime('%Y-%m-%d')}")
    return previous_dates[-1]


def _build_static_signal(expression: str, instrument: str, start_date: str, end_date: str) -> pd.DataFrame:
    signal_start = _previous_trading_date(start_date)
    signal_end = _previous_trading_date(end_date)
    signal = D.features(
        D.instruments(instrument),
        [expression],
        start_time=signal_start.strftime("%Y-%m-%d"),
        end_time=signal_end.strftime("%Y-%m-%d"),
    )
    signal.columns = ["score"]
    signal = signal.dropna()
    if signal.empty:
        raise ValueError("Signal dataframe is empty after D.features().")
    return signal


def _build_selector(
    *,
    selector_mode: str,
    weighting_method: str,
    recent_perf_candidate_limit: int | None,
    recent_perf_batch_size: int,
    score_threshold: float,
    ir_threshold: float,
    mwu_use_dual_experts: bool,
) -> BaseSelector:
    if selector_mode == "mwu" or weighting_method == "mwu":
        return MWUAllExpertSelector(
            recent_perf_batch_size=recent_perf_batch_size,
            include_inverse_experts=mwu_use_dual_experts,
        )
    if selector_mode == "aff":
        return AFFRecentPerformanceSelector(
            recent_perf_candidate_limit=recent_perf_candidate_limit,
            recent_perf_batch_size=recent_perf_batch_size,
            score_threshold=score_threshold,
            ir_threshold=ir_threshold,
        )
    return RecentPerformanceSelector(
        recent_perf_candidate_limit=recent_perf_candidate_limit,
        recent_perf_batch_size=recent_perf_batch_size,
    )


def _build_allocator(
    weighting_method: str,
    *,
    mwu_learning_rate: float,
    mwu_reward_cap: float,
    mwu_explore_rate: float,
    mwu_max_weight: float,
) -> BaseAllocator:
    if weighting_method == "mwu":
        return MWUAllocator(
            learning_rate=mwu_learning_rate,
            reward_cap=mwu_reward_cap,
            exploration_rate=mwu_explore_rate,
            max_weight=mwu_max_weight,
        )
    if weighting_method == "regression":
        return RegressionAllocator()
    if weighting_method == "score_ir":
        return ScoreIRAllocator()
    return NormalizedScoreAllocator()


def _build_signal_generator(weighting_method: str) -> BaseSignalGenerator:
    if weighting_method == "mwu":
        return NormalizedPanelSignalGenerator()
    return ExpressionSignalGenerator()


def _build_news_review_service(
    *,
    enable_news_review: bool,
    news_data_path: str | None,
    news_batch_size: int,
    news_llm_config_path: str | None,
    news_confidence_threshold: float,
    event_logger: BacktestEventLogger | None,
) -> NewsReviewService | None:
    if not enable_news_review:
        return None
    return NewsReviewService.from_env_config(
        news_data_path=news_data_path,
        news_batch_size=news_batch_size,
        llm_config_path=news_llm_config_path,
        confidence_threshold=news_confidence_threshold,
        event_logger=event_logger,
    )


def _build_rolling_signal(
    factor_library: list[dict[str, Any]],
    selector: BaseSelector,
    allocator: BaseAllocator,
    signal_generator: BaseSignalGenerator,
    *,
    provider_uri: Path,
    start_date: str,
    end_date: str,
    instrument: str,
    benchmark: str,
    top_k: int,
    top_n: int,
    daily_buy_topk: int,
    window_days: int,
    rebalance_window_days: int,
    return_expression: str,
    mwu_enable_tail_switch: bool,
    mwu_tail_switch_mode: str,
    mwu_direction_rank_ic_threshold: float,
    mwu_direction_top_bottom_k: int,
    mwu_direction_validation_days: int,
    mwu_direction_spread_threshold: float,
    mwu_bayes_half_life: float,
    mwu_bayes_prior_strength: float,
    mwu_bayes_hurdle: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scheduler = RebalanceScheduler(
        start_date=start_date,
        end_date=end_date,
        rebalance_window_days=rebalance_window_days,
    )
    windows = scheduler.get_windows()
    if not windows:
        raise ValueError("No trading dates found in the requested backtest range.")

    signal_frames: list[pd.DataFrame] = []
    rolling_windows: list[dict[str, Any]] = []

    total_windows = len(windows)
    for window in windows:
        holding_start = window.start.strftime("%Y-%m-%d")
        holding_end = window.end.strftime("%Y-%m-%d")
        loop_start = perf_counter()

        context = WindowContext(
            window_index=window.window_index,
            selection_date=window.selection_date,
            window_start=window.signal_start,
            window_end=window.signal_end,
            instrument=instrument,
            benchmark=benchmark,
            provider_uri=provider_uri,
            top_n=int(top_n),
            top_k=int(top_k),
            window_days=int(window_days),
            rebalance_window_days=int(rebalance_window_days),
            return_expression=str(return_expression),
        )
        selection_result = selector.select(factor_library, context)
        selected_items = selection_result.get("selected_items", []) or []
        allocation_result = allocator.allocate(selected_items, context)
        weighted_factors = allocation_result.get("weighted_factors", []) or []
        generation_result = signal_generator.generate(weighted_factors, context)

        expression = generation_result.get("combined_expression")
        if not expression:
            raise ValueError(
                f"Rolling signal generation failed for holding_start={holding_start}: "
                f"selection_status={selection_result.get('selection_context', {}).get('status')}"
            )

        window_signal = generation_result.get("signal")
        if window_signal is None or window_signal.empty:
            raise ValueError(f"Window signal is empty for holding_start={holding_start}")

        direction_context = {
            "enabled": bool(mwu_enable_tail_switch),
            "applied": False,
            "flip_to_tail": False,
            "mode": str(mwu_tail_switch_mode),
            "rank_ic_threshold": float(mwu_direction_rank_ic_threshold),
            "spread_threshold": float(mwu_direction_spread_threshold),
            "top_bottom_k": int(mwu_direction_top_bottom_k),
            "daily_buy_topk": int(daily_buy_topk),
            "validation_days": int(mwu_direction_validation_days),
            "bayes_half_life": float(mwu_bayes_half_life),
            "bayes_prior_strength": float(mwu_bayes_prior_strength),
            "bayes_hurdle": float(mwu_bayes_hurdle),
            "status": "disabled",
        }
        allocation_method = str(allocation_result.get("allocation_context", {}).get("method"))
        if bool(mwu_enable_tail_switch) and weighted_factors and allocation_method == "mwu":
            reward_start = allocation_result.get("allocation_context", {}).get("reward_start")
            reward_end = allocation_result.get("allocation_context", {}).get("reward_end")
            if reward_start and reward_end:
                validation_range = _resolve_recent_validation_range(
                    reward_start=reward_start,
                    reward_end=reward_end,
                    validation_days=mwu_direction_validation_days,
                )
                if validation_range is None:
                    direction_context = {
                        "enabled": True,
                        "applied": False,
                        "flip_to_tail": False,
                        "rank_ic_threshold": float(mwu_direction_rank_ic_threshold),
                        "spread_threshold": float(mwu_direction_spread_threshold),
                        "top_bottom_k": int(mwu_direction_top_bottom_k),
                        "validation_days": int(mwu_direction_validation_days),
                        "status": "validation_window_unavailable",
                        "reward_start": str(reward_start),
                        "reward_end": str(reward_end),
                    }
                else:
                    validation_start, validation_end, actual_validation_days = validation_range
                    validation_train_items = _filter_selected_items_before_date(
                        selected_items,
                        cutoff_date=validation_start,
                    )
                    validation_allocation_result = allocator.allocate(validation_train_items, context)
                    validation_weighted_factors = validation_allocation_result.get("weighted_factors", []) or []
                    if not validation_weighted_factors:
                        direction_context = {
                            "enabled": True,
                            "applied": False,
                            "flip_to_tail": False,
                            "rank_ic_threshold": float(mwu_direction_rank_ic_threshold),
                            "spread_threshold": float(mwu_direction_spread_threshold),
                            "top_bottom_k": int(mwu_direction_top_bottom_k),
                            "validation_days": int(mwu_direction_validation_days),
                            "actual_validation_days": int(actual_validation_days),
                            "status": "validation_training_unavailable",
                            "reward_start": str(reward_start),
                            "reward_end": str(reward_end),
                            "validation_signal_start": validation_start.strftime("%Y-%m-%d"),
                            "validation_signal_end": validation_end.strftime("%Y-%m-%d"),
                            "validation_train_factor_count": len(validation_train_items),
                            "validation_weight_source": "pre_validation_reward_window",
                        }
                    else:
                        validation_context = replace(
                            context,
                            window_start=validation_start,
                            window_end=validation_end,
                        )
                        validation_generation = signal_generator.generate(validation_weighted_factors, validation_context)
                        validation_signal = validation_generation.get("signal")
                        validation_metrics = _calculate_signal_direction_metrics(
                            validation_signal,
                            instrument=instrument,
                            label_expression=return_expression,
                            top_bottom_k=mwu_direction_top_bottom_k,
                        )
                        direction_context = {
                            "enabled": True,
                            "applied": validation_metrics is not None,
                            "flip_to_tail": False,
                            "rank_ic_threshold": float(mwu_direction_rank_ic_threshold),
                            "spread_threshold": float(mwu_direction_spread_threshold),
                            "top_bottom_k": int(mwu_direction_top_bottom_k),
                            "validation_days": int(mwu_direction_validation_days),
                            "actual_validation_days": int(actual_validation_days),
                            "status": "metrics_unavailable" if validation_metrics is None else "evaluated",
                            "reward_start": str(reward_start),
                            "reward_end": str(reward_end),
                            "validation_signal_start": validation_start.strftime("%Y-%m-%d"),
                            "validation_signal_end": validation_end.strftime("%Y-%m-%d"),
                            "validation_train_factor_count": len(validation_train_items),
                            "validation_weight_source": "pre_validation_reward_window",
                            "validation_allocation_context": validation_allocation_result.get("allocation_context"),
                            "validation_generation_context": validation_generation.get("generation_context"),
                            "validation_metrics": validation_metrics,
                        }
                        if validation_metrics is not None:
                            tail_switch_mode = str(mwu_tail_switch_mode)
                            if tail_switch_mode == "bayesian":
                                posterior_context = _calculate_bayesian_direction_posterior(
                                    validation_metrics.get("daily_top_bottom_spreads", []) or [],
                                    half_life=mwu_bayes_half_life,
                                    prior_strength=mwu_bayes_prior_strength,
                                    hurdle=mwu_bayes_hurdle,
                                )
                                direction_multiplier = float(
                                    posterior_context.get("direction_multiplier", 0.0)
                                )
                                mixed_signal, mix_context = _apply_bayesian_head_tail_rank_mix(
                                    window_signal,
                                    direction_multiplier=direction_multiplier,
                                    daily_buy_topk=daily_buy_topk,
                                )
                                window_signal = mixed_signal
                                direction_context["applied"] = (
                                    posterior_context.get("status") == "success"
                                    and mix_context.get("status") == "success"
                                )
                                direction_context["flip_to_tail"] = bool(direction_multiplier < 0.0)
                                direction_context["posterior_context"] = posterior_context
                                direction_context["mix_context"] = mix_context
                                if direction_multiplier < -1e-12:
                                    direction_context["status"] = "bayesian_tail_biased"
                                elif direction_multiplier > 1e-12:
                                    direction_context["status"] = "bayesian_head_biased"
                                else:
                                    direction_context["status"] = "bayesian_neutral"
                                expression = (
                                    f"bayesian_head_tail_mix(g={direction_multiplier:.4f}; {expression})"
                                )
                            else:
                                should_flip = (
                                    float(validation_metrics.get("rank_ic", 0.0))
                                    < float(mwu_direction_rank_ic_threshold)
                                    and float(validation_metrics.get("top_bottom_spread", 0.0))
                                    < float(mwu_direction_spread_threshold)
                                )
                                direction_context["flip_to_tail"] = bool(should_flip)
                                if should_flip:
                                    window_signal = _normalize_signal_frame(window_signal).copy()
                                    window_signal["score"] = -window_signal["score"].astype(float)
                                    expression = f"tail_switched({expression})"
                                    direction_context["status"] = "flipped_to_tail"
                                else:
                                    direction_context["status"] = "kept_head"
            else:
                direction_context["status"] = "missing_reward_window"

        if direction_context.get("enabled"):
            direction_context.setdefault("mode", str(mwu_tail_switch_mode))
            direction_context.setdefault("daily_buy_topk", int(daily_buy_topk))
            direction_context.setdefault("bayes_half_life", float(mwu_bayes_half_life))
            direction_context.setdefault("bayes_prior_strength", float(mwu_bayes_prior_strength))
            direction_context.setdefault("bayes_hurdle", float(mwu_bayes_hurdle))

        signal_elapsed = perf_counter() - loop_start
        signal_frames.append(window_signal)

        rolling_windows.append(
            {
                "window_index": window.window_index,
                "selection_date": context.selection_date.strftime("%Y-%m-%d"),
                "holding_start": holding_start,
                "holding_end": holding_end,
                "signal_start": context.window_start.strftime("%Y-%m-%d"),
                "signal_end": context.window_end.strftime("%Y-%m-%d"),
                "combined_expression": expression,
                "selected_items": selected_items,
                "selected_factor_ids": [item.get("factor_id") for item in weighted_factors],
                "selected_factors": weighted_factors,
                "selection_context": selection_result.get("selection_context"),
                "allocation_context": allocation_result.get("allocation_context"),
                "generation_context": generation_result.get("generation_context"),
                "direction_context": direction_context,
                "elapsed_seconds": float(signal_elapsed),
            }
        )

    signal = pd.concat(signal_frames).sort_index()
    signal = signal[~signal.index.duplicated(keep="last")].dropna()
    if signal.empty:
        raise ValueError("Rolling signal dataframe is empty after concatenation.")
    return signal, rolling_windows


def _merge_backtest_config(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base_config)
    for key, value in overrides.items():
        if key == "exchange_kwargs" and isinstance(value, dict):
            merged_exchange_kwargs = dict(merged.get("exchange_kwargs", {}))
            merged_exchange_kwargs.update(value)
            merged["exchange_kwargs"] = merged_exchange_kwargs
        elif value is not None:
            merged[key] = value
    return merged


def _validate_backtest_config(config: dict[str, Any]) -> None:
    if int(config["rebalance_window_days"]) <= 0:
        raise ValueError("rebalance_window_days must be a positive integer")
    if int(config["window_days"]) <= 0:
        raise ValueError("window_days must be a positive integer")
    top_n = config.get("top_n")
    if top_n is not None and int(top_n) <= 0:
        raise ValueError("top_n must be a positive integer")
    if int(config["topk"]) <= 0:
        raise ValueError("topk must be a positive integer")
    if int(config["n_drop"]) < 0:
        raise ValueError("n_drop must be a non-negative integer")
    if str(config.get("portfolio_mode", "topk_dropout")) not in {"topk_dropout", "fixed_horizon"}:
        raise ValueError("portfolio_mode must be one of: topk_dropout, fixed_horizon")
    if int(config.get("holding_period_days", 1)) <= 0:
        raise ValueError("holding_period_days must be a positive integer")
    if int(config.get("daily_buy_topk", 1)) <= 0:
        raise ValueError("daily_buy_topk must be a positive integer")
    if int(config.get("factor_eval_top_k", config.get("daily_buy_topk", 1))) <= 0:
        raise ValueError("factor_eval_top_k must be a positive integer")
    if int(config["news_batch_size"]) <= 0:
        raise ValueError("news_batch_size must be a positive integer")
    if int(config.get("news_candidate_pool_multiplier", 1)) <= 0:
        raise ValueError("news_candidate_pool_multiplier must be a positive integer")
    if int(config["model_train_window_days"]) <= 0:
        raise ValueError("model_train_window_days must be a positive integer")
    if int(config["model_label_horizon_days"]) <= 0:
        raise ValueError("model_label_horizon_days must be a positive integer")
    recent_perf_candidate_limit = config.get("recent_perf_candidate_limit")
    if recent_perf_candidate_limit is not None and int(recent_perf_candidate_limit) <= 0:
        raise ValueError("recent_perf_candidate_limit must be a positive integer")
    if str(config["signal_mode"]) not in {"rolling", "static", "dynamic", "model"}:
        raise ValueError("signal_mode must be one of: rolling, static, dynamic, model")
    if str(config["selector_mode"]) not in {"recent", "aff", "mwu"}:
        raise ValueError("selector_mode must be one of: recent, aff, mwu")
    weighting_method = config.get("weighting_method")
    if weighting_method is not None and str(weighting_method) not in {"normalized", "regression", "score_ir", "mwu"}:
        raise ValueError("weighting_method must be one of: normalized, regression, score_ir, mwu")
    if float(config.get("mwu_learning_rate", 0.0)) < 0:
        raise ValueError("mwu_learning_rate must be non-negative")
    if float(config.get("mwu_reward_cap", 0.0)) <= 0:
        raise ValueError("mwu_reward_cap must be a positive float")
    if not 0.0 <= float(config.get("mwu_explore_rate", 0.0)) <= 1.0:
        raise ValueError("mwu_explore_rate must be within [0, 1]")
    if not 0.0 < float(config.get("mwu_max_weight", 0.0)) <= 1.0:
        raise ValueError("mwu_max_weight must be within (0, 1]")
    if int(config.get("mwu_direction_top_bottom_k", 0)) <= 0:
        raise ValueError("mwu_direction_top_bottom_k must be a positive integer")
    if int(config.get("mwu_direction_validation_days", 0)) <= 0:
        raise ValueError("mwu_direction_validation_days must be a positive integer")
    if str(config.get("mwu_tail_switch_mode", "hard")) not in {"hard", "bayesian"}:
        raise ValueError("mwu_tail_switch_mode must be one of: hard, bayesian")
    if float(config.get("mwu_bayes_half_life", 0.0)) <= 0:
        raise ValueError("mwu_bayes_half_life must be a positive float")
    if float(config.get("mwu_bayes_prior_strength", 0.0)) < 0:
        raise ValueError("mwu_bayes_prior_strength must be non-negative")
    if float(config.get("mwu_bayes_hurdle", 0.0)) < 0:
        raise ValueError("mwu_bayes_hurdle must be non-negative")


class QlibBacktestService:
    """Run the linear-weighting Qlib backtest inside the refactored workflow."""

    def __init__(self, repo_root: Path | None = None, data_dir: Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir is not None else self.repo_root / "data"
        self.output_root = self.data_dir / "backtest_outputs"
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_default_config() -> dict[str, Any]:
        """Return a deep-copied default backtest config."""
        return deepcopy(DEFAULT_BACKTEST_CONFIG)

    def run(
        self,
        *,
        weighting_result: dict[str, Any] | None = None,
        factor_library: list[dict[str, Any]] | None = None,
        backtest_config: dict[str, Any] | None = None,
        weighting_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run a Qlib backtest and return structured artifacts plus summary metrics."""
        if BACKTEST_IMPORT_ERROR is not None:
            return {
                "status": "missing_runtime_dependency",
                "message": f"Qlib backtest runtime is unavailable: {BACKTEST_IMPORT_ERROR}",
                "error": str(BACKTEST_IMPORT_ERROR),
            }

        config = _merge_backtest_config(self.build_default_config(), backtest_config or {})
        _validate_backtest_config(config)

        signal_mode = str(config["signal_mode"])
        requires_weighting_result = signal_mode == "static"

        resolved_weighting_path: Path | None = None
        resolved_weighting_result = weighting_result
        if resolved_weighting_result is None and (requires_weighting_result or weighting_path):
            resolved_weighting_result, resolved_weighting_path = _load_weighting_result(
                Path(weighting_path).expanduser() if weighting_path else None,
                self.data_dir,
            )
        elif resolved_weighting_result is not None:
            if weighting_path:
                resolved_weighting_path = Path(weighting_path).expanduser()
            else:
                save_path = resolved_weighting_result.get("save_path")
                if save_path:
                    resolved_weighting_path = Path(str(save_path)).expanduser()

        if requires_weighting_result and not resolved_weighting_result:
            return {
                "status": "pending_weighting_result",
                "config": config,
                "message": "No weighting_result is available for static backtest.",
            }

        weighting_payload = resolved_weighting_result or {}
        weighting_status = str(weighting_payload.get("status", ""))
        selected_factors = weighting_payload.get("selected_factors", []) or []
        selected_factor_count = len(selected_factors)
        if requires_weighting_result and weighting_status and weighting_status != "success":
            return {
                "status": "pending_weighting_result",
                "config": config,
                "weighting_status": weighting_status,
                "weighting_path": str(resolved_weighting_path) if resolved_weighting_path else None,
                "selected_factor_count": selected_factor_count,
                "combined_expression": weighting_payload.get("combined_expression"),
                "message": f"Weighting result is not ready for backtest: status={weighting_status}",
            }

        if requires_weighting_result and selected_factor_count <= 0:
            return {
                "status": "pending_weighting_result",
                "config": config,
                "weighting_status": weighting_status or "missing_selected_factors",
                "weighting_path": str(resolved_weighting_path) if resolved_weighting_path else None,
                "selected_factor_count": 0,
                "combined_expression": weighting_payload.get("combined_expression"),
                "message": "Weighting result does not contain selected factors.",
            }

        resolved_factor_library = factor_library
        if resolved_factor_library is None and signal_mode in {"rolling", "dynamic", "model"}:
            resolved_factor_library = _load_factor_library(self.data_dir)

        if signal_mode in {"rolling", "dynamic", "model"} and not resolved_factor_library:
            return {
                "status": "missing_factor_library",
                "config": config,
                "weighting_path": str(resolved_weighting_path) if resolved_weighting_path else None,
                "selected_factor_count": selected_factor_count,
                "combined_expression": weighting_payload.get("combined_expression"),
                "message": "Rolling, dynamic, or model backtest requires a non-empty factor_library.",
            }

        weighting_config = (
            weighting_payload.get("config", {})
            if isinstance(weighting_payload.get("config"), dict)
            else {}
        )
        provider_uri = _resolve_provider_uri(config.get("provider_uri") or weighting_config.get("provider_uri"))
        benchmark_code = _benchmark_code(
            str(config.get("benchmark") or weighting_payload.get("benchmark") or weighting_config.get("benchmark"))
        )
        resolved_window_days = int(
            config["window_days"]
            if config.get("window_days") is not None
            else weighting_payload.get("window_days", weighting_config.get("window_days", 5))
        )
        portfolio_mode = str(config.get("portfolio_mode", "topk_dropout"))
        holding_period_days = int(config.get("holding_period_days", 10))
        daily_buy_topk = int(config.get("daily_buy_topk", config.get("n_drop", 5)))
        factor_eval_top_k = int(config.get("factor_eval_top_k") or daily_buy_topk)
        resolved_recent_perf_candidate_limit = config.get("recent_perf_candidate_limit")
        if resolved_recent_perf_candidate_limit is None:
            resolved_recent_perf_candidate_limit = weighting_config.get("recent_perf_candidate_limit")
        resolved_recent_perf_batch_size = int(
            config.get("recent_perf_batch_size") or weighting_config.get("recent_perf_batch_size", 8)
        )
        return_expression = str(config.get("return_expression") or weighting_config.get("return_expression"))
        resolved_model_label_expression = str(config.get("model_label_expression") or "").strip()
        if not resolved_model_label_expression:
            resolved_model_label_expression = (
                f"Ref($close, -{int(config['model_label_horizon_days']) + 1})/Ref($close, -1) - 1"
            )
        resolved_weighting_method = str(
            config.get("weighting_method")
            or (resolved_weighting_result or {}).get("method")
            or weighting_config.get("weighting_method")
            or "normalized"
        )
        configured_selector_mode = str(config["selector_mode"])
        if configured_selector_mode == "mwu":
            resolved_weighting_method = "mwu"
        resolved_selector_mode = "mwu" if resolved_weighting_method == "mwu" else configured_selector_mode
        top_n = int(
            config["top_n"]
            if config.get("top_n") is not None
            else weighting_payload.get("top_n", weighting_config.get("top_n", 10))
        )

        run_dir = self.output_root / f"linear_weighting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        live_log_path = run_dir / "backtest_live_log.jsonl"
        event_logger = BacktestEventLogger(live_log_path)

        qlib.init(provider_uri=str(provider_uri), region=REG_CN)
        event_logger.log(
            "backtest_run_started",
            {
                "provider_uri": str(provider_uri),
                "weighting_path": str(resolved_weighting_path) if resolved_weighting_path else None,
                "start_date": config["start_date"],
                "end_date": config["end_date"],
                "signal_mode": signal_mode,
                "portfolio_mode": portfolio_mode,
                "holding_period_days": holding_period_days,
                "daily_buy_topk": daily_buy_topk,
                "factor_eval_top_k": factor_eval_top_k,
                "selector_mode": resolved_selector_mode,
                "weighting_method": resolved_weighting_method,
                "mwu_learning_rate": float(config["mwu_learning_rate"]),
                "mwu_reward_cap": float(config["mwu_reward_cap"]),
                "mwu_explore_rate": float(config["mwu_explore_rate"]),
                "mwu_max_weight": float(config["mwu_max_weight"]),
                "mwu_use_dual_experts": bool(config["mwu_use_dual_experts"]),
                "mwu_enable_tail_switch": bool(config["mwu_enable_tail_switch"]),
                "mwu_tail_switch_mode": str(config["mwu_tail_switch_mode"]),
                "mwu_direction_rank_ic_threshold": float(config["mwu_direction_rank_ic_threshold"]),
                "mwu_direction_top_bottom_k": int(config["mwu_direction_top_bottom_k"]),
                "mwu_direction_validation_days": int(config["mwu_direction_validation_days"]),
                "mwu_direction_spread_threshold": float(config["mwu_direction_spread_threshold"]),
                "mwu_bayes_half_life": float(config["mwu_bayes_half_life"]),
                "mwu_bayes_prior_strength": float(config["mwu_bayes_prior_strength"]),
                "mwu_bayes_hurdle": float(config["mwu_bayes_hurdle"]),
                "enable_news_review": config["enable_news_review"],
                "news_candidate_pool_multiplier": int(config["news_candidate_pool_multiplier"]),
                "model_train_window_days": int(config["model_train_window_days"]),
                "model_label_horizon_days": int(config["model_label_horizon_days"]),
                "model_label_expression": resolved_model_label_expression,
                "run_dir": str(run_dir),
            },
        )

        selector = _build_selector(
            selector_mode=resolved_selector_mode,
            weighting_method=resolved_weighting_method,
            recent_perf_candidate_limit=(
                int(resolved_recent_perf_candidate_limit)
                if resolved_recent_perf_candidate_limit is not None
                else None
            ),
            recent_perf_batch_size=resolved_recent_perf_batch_size,
            score_threshold=float(config["selector_score_threshold"]),
            ir_threshold=float(config["selector_ir_threshold"]),
            mwu_use_dual_experts=bool(config["mwu_use_dual_experts"]),
        )
        allocator = _build_allocator(
            resolved_weighting_method,
            mwu_learning_rate=float(config["mwu_learning_rate"]),
            mwu_reward_cap=float(config["mwu_reward_cap"]),
            mwu_explore_rate=float(config["mwu_explore_rate"]),
            mwu_max_weight=float(config["mwu_max_weight"]),
        )
        signal_generator = _build_signal_generator(resolved_weighting_method)
        news_review_service = _build_news_review_service(
            enable_news_review=bool(config["enable_news_review"]),
            news_data_path=config.get("news_data_path"),
            news_batch_size=int(config["news_batch_size"]),
            news_llm_config_path=config.get("news_llm_config_path"),
            news_confidence_threshold=float(config["news_confidence_threshold"]),
            event_logger=event_logger,
        )
        news_review_status = news_review_service.get_status() if news_review_service is not None else None
        portfolio_kwargs = {
            "portfolio_mode": portfolio_mode,
            "holding_period_days": holding_period_days,
            "daily_buy_topk": daily_buy_topk,
            "news_candidate_pool_multiplier": int(config["news_candidate_pool_multiplier"]),
        }

        rolling_windows: list[dict[str, Any]] = []
        latest_expression: str | None = None
        signal: pd.DataFrame | None = None
        model_window_signal: ModelWindowSignal | None = None

        if signal_mode == "static":
            expression = resolved_weighting_result.get("combined_expression")
            if not expression:
                raise ValueError("combined_expression is missing for static backtest mode.")
            signal = _build_static_signal(
                expression=expression,
                instrument=str(config["instrument"]),
                start_date=str(config["start_date"]),
                end_date=str(config["end_date"]),
            )
            latest_expression = expression
            strategy = NewsAwareTopkStrategy(
                signal=signal,
                topk=int(config["topk"]),
                n_drop=int(config["n_drop"]),
                **portfolio_kwargs,
                news_review_service=news_review_service,
                event_logger=event_logger,
            )
        elif signal_mode in {"rolling", "dynamic"}:
            strategy = DynamicWindowTopkStrategy(
                factor_library=list(resolved_factor_library or []),
                start_date=str(config["start_date"]),
                end_date=str(config["end_date"]),
                instrument=str(config["instrument"]),
                benchmark=benchmark_code,
                provider_uri=str(provider_uri),
                topk=int(config["topk"]),
                n_drop=int(config["n_drop"]),
                factor_eval_top_k=factor_eval_top_k,
                top_n=top_n,
                window_days=resolved_window_days,
                rebalance_window_days=int(config["rebalance_window_days"]),
                return_expression=return_expression,
                selector=selector,
                allocator=allocator,
                signal_generator=signal_generator,
                **portfolio_kwargs,
                news_review_service=news_review_service,
                event_logger=event_logger,
            )
        elif signal_mode == "model":
            trainer = LGBMWindowTrainer(
                train_window_days=int(config["model_train_window_days"]),
                label_horizon_days=int(config["model_label_horizon_days"]),
                label_expression=resolved_model_label_expression,
            )
            model_window_signal = ModelWindowSignal(
                factor_library=list(resolved_factor_library or []),
                selector=selector,
                trainer=trainer,
                scheduler=RebalanceScheduler(
                    start_date=str(config["start_date"]),
                    end_date=str(config["end_date"]),
                    rebalance_window_days=int(config["rebalance_window_days"]),
                ),
                instrument=str(config["instrument"]),
                benchmark=benchmark_code,
                provider_uri=str(provider_uri),
                top_n=top_n,
                top_k=factor_eval_top_k,
                window_days=resolved_window_days,
                rebalance_window_days=int(config["rebalance_window_days"]),
                return_expression=return_expression,
            )
            strategy = NewsAwareTopkStrategy(
                signal=model_window_signal,
                topk=int(config["topk"]),
                n_drop=int(config["n_drop"]),
                **portfolio_kwargs,
                news_review_service=news_review_service,
                event_logger=event_logger,
            )

        executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
        portfolio_metric_dict, indicator_dict = backtest(
            start_time=str(config["start_date"]),
            end_time=str(config["end_date"]),
            strategy=strategy,
            executor=executor,
            benchmark=benchmark_code,
            account=float(config["account"]),
            exchange_kwargs=dict(config["exchange_kwargs"]),
        )

        report_df, positions = portfolio_metric_dict["1day"]
        summary = _summarize_risk(report_df)
        if signal_mode in {"rolling", "dynamic"}:
            rolling_windows = strategy.get_window_history()
            signal = strategy.get_signal_cache()
            latest_expression = rolling_windows[-1]["combined_expression"] if rolling_windows else None
        elif signal_mode == "model":
            rolling_windows = model_window_signal.get_window_history() if model_window_signal is not None else []
            signal = model_window_signal.get_signal_cache() if model_window_signal is not None else None
            latest_expression = None
        news_review_history = (
            strategy.get_news_review_history() if hasattr(strategy, "get_news_review_history") else []
        )
        direction_flip_count = sum(
            1
            for window_payload in rolling_windows
            if bool((window_payload.get("direction_context") or {}).get("flip_to_tail"))
        )
        test_signal_metrics = _calculate_test_signal_metrics(
            signal,
            instrument=str(config["instrument"]),
            label_expression=resolved_model_label_expression if signal_mode == "model" else return_expression,
        )

        signal_path = run_dir / "signal.pkl"
        report_path = run_dir / "report_1day.csv"
        positions_path = run_dir / "positions.json"
        metrics_path = run_dir / "metrics.json"
        rolling_windows_path = run_dir / "rolling_windows.json"
        selected_items_path = run_dir / "selected_items.json"
        net_value_curve_path = run_dir / "net_value_curve.png"
        news_review_path = run_dir / "news_review_history.json"

        if signal is not None:
            signal.to_pickle(signal_path)
        report_df.to_csv(report_path)
        _plot_net_value_curve(report_df, net_value_curve_path)
        with open(positions_path, "w", encoding="utf-8") as file:
            json.dump(_stringify_position_keys(positions), file, ensure_ascii=False, indent=2, default=str)
        if signal_mode in {"rolling", "dynamic", "model"}:
            with open(rolling_windows_path, "w", encoding="utf-8") as file:
                json.dump(rolling_windows, file, ensure_ascii=False, indent=2, default=str)
            with open(selected_items_path, "w", encoding="utf-8") as file:
                json.dump(
                    _build_selected_items_by_date(rolling_windows),
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
        if news_review_history:
            with open(news_review_path, "w", encoding="utf-8") as file:
                json.dump(news_review_history, file, ensure_ascii=False, indent=2, default=str)

        metrics_payload = {
            "weighting_path": str(resolved_weighting_path) if resolved_weighting_path else None,
            "provider_uri": str(provider_uri),
            "instrument": config["instrument"],
            "benchmark": benchmark_code,
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "topk": int(config["topk"]),
            "n_drop": int(config["n_drop"]),
            "portfolio_mode": portfolio_mode,
            "holding_period_days": holding_period_days,
            "daily_buy_topk": daily_buy_topk,
            "factor_eval_top_k": factor_eval_top_k,
            "top_n": top_n,
            "signal_mode": signal_mode,
            "selector_mode": resolved_selector_mode,
            "weighting_method": resolved_weighting_method,
            "selector_score_threshold": float(config["selector_score_threshold"]),
            "selector_ir_threshold": float(config["selector_ir_threshold"]),
            "mwu_learning_rate": float(config["mwu_learning_rate"]),
            "mwu_reward_cap": float(config["mwu_reward_cap"]),
            "mwu_explore_rate": float(config["mwu_explore_rate"]),
            "mwu_max_weight": float(config["mwu_max_weight"]),
            "mwu_use_dual_experts": bool(config["mwu_use_dual_experts"]),
            "mwu_enable_tail_switch": bool(config["mwu_enable_tail_switch"]),
            "mwu_tail_switch_mode": str(config["mwu_tail_switch_mode"]),
            "mwu_direction_rank_ic_threshold": float(config["mwu_direction_rank_ic_threshold"]),
            "mwu_direction_top_bottom_k": int(config["mwu_direction_top_bottom_k"]),
            "mwu_direction_validation_days": int(config["mwu_direction_validation_days"]),
            "mwu_direction_spread_threshold": float(config["mwu_direction_spread_threshold"]),
            "mwu_bayes_half_life": float(config["mwu_bayes_half_life"]),
            "mwu_bayes_prior_strength": float(config["mwu_bayes_prior_strength"]),
            "mwu_bayes_hurdle": float(config["mwu_bayes_hurdle"]),
            "window_days": resolved_window_days,
            "rebalance_window_days": int(config["rebalance_window_days"]),
            "model_train_window_days": int(config["model_train_window_days"]),
            "model_label_horizon_days": int(config["model_label_horizon_days"]),
            "model_label_expression": resolved_model_label_expression,
            "return_expression": return_expression,
            "rolling_window_count": len(rolling_windows),
            "direction_flip_count": direction_flip_count,
            "rolling_windows_path": str(rolling_windows_path) if signal_mode in {"rolling", "dynamic", "model"} else None,
            "selected_items_path": str(selected_items_path) if signal_mode in {"rolling", "dynamic", "model"} else None,
            "net_value_curve_path": str(net_value_curve_path),
            "enable_news_review": bool(config["enable_news_review"]),
            "news_candidate_pool_multiplier": int(config["news_candidate_pool_multiplier"]),
            "news_review_status": news_review_status,
            "news_review_days": len(news_review_history),
            "news_review_path": str(news_review_path) if news_review_history else None,
            "live_log_path": str(live_log_path),
            "combined_expression": latest_expression if signal_mode == "static" else None,
            "latest_combined_expression": latest_expression,
            "signal_rows": len(signal) if signal is not None else 0,
            "summary": summary,
            "test_signal_metrics": test_signal_metrics,
            "indicator_keys": list(indicator_dict.keys()),
        }
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, ensure_ascii=False, indent=2)

        event_logger.log(
            "backtest_run_completed",
            {
                "run_dir": str(run_dir),
                "signal_path": str(signal_path) if signal is not None else None,
                "report_path": str(report_path),
                "positions_path": str(positions_path),
                "metrics_path": str(metrics_path),
                "net_value_curve_path": str(net_value_curve_path),
                "news_review_path": str(news_review_path) if news_review_history else None,
                "summary": summary,
                "test_signal_metrics": test_signal_metrics,
            },
        )

        return {
            "status": "success",
            "message": "Qlib backtest finished successfully.",
            "config": config,
            "resolved_config": {
                "provider_uri": str(provider_uri),
                "instrument": config["instrument"],
                "benchmark": benchmark_code,
                "top_n": top_n,
                "topk": int(config["topk"]),
                "n_drop": int(config["n_drop"]),
                "portfolio_mode": portfolio_mode,
                "holding_period_days": holding_period_days,
                "daily_buy_topk": daily_buy_topk,
                "factor_eval_top_k": factor_eval_top_k,
                "signal_mode": signal_mode,
                "window_days": resolved_window_days,
                "rebalance_window_days": int(config["rebalance_window_days"]),
                "model_train_window_days": int(config["model_train_window_days"]),
                "model_label_horizon_days": int(config["model_label_horizon_days"]),
                "model_label_expression": resolved_model_label_expression,
                "selector_mode": resolved_selector_mode,
                "weighting_method": resolved_weighting_method,
                "mwu_learning_rate": float(config["mwu_learning_rate"]),
                "mwu_reward_cap": float(config["mwu_reward_cap"]),
                "mwu_explore_rate": float(config["mwu_explore_rate"]),
                "mwu_max_weight": float(config["mwu_max_weight"]),
                "mwu_use_dual_experts": bool(config["mwu_use_dual_experts"]),
                "mwu_enable_tail_switch": bool(config["mwu_enable_tail_switch"]),
                "mwu_tail_switch_mode": str(config["mwu_tail_switch_mode"]),
                "mwu_direction_rank_ic_threshold": float(config["mwu_direction_rank_ic_threshold"]),
                "mwu_direction_top_bottom_k": int(config["mwu_direction_top_bottom_k"]),
                "mwu_direction_validation_days": int(config["mwu_direction_validation_days"]),
                "mwu_direction_spread_threshold": float(config["mwu_direction_spread_threshold"]),
                "mwu_bayes_half_life": float(config["mwu_bayes_half_life"]),
                "mwu_bayes_prior_strength": float(config["mwu_bayes_prior_strength"]),
                "mwu_bayes_hurdle": float(config["mwu_bayes_hurdle"]),
                "news_candidate_pool_multiplier": int(config["news_candidate_pool_multiplier"]),
                "return_expression": return_expression,
            },
            "weighting_path": str(resolved_weighting_path) if resolved_weighting_path else None,
            "selected_factor_count": selected_factor_count,
            "combined_expression": latest_expression if signal_mode == "static" else None,
            "latest_combined_expression": latest_expression,
            "news_review_status": news_review_status,
            "news_review_days": len(news_review_history),
            "signal_rows": len(signal) if signal is not None else 0,
            "rolling_window_count": len(rolling_windows),
            "direction_flip_count": direction_flip_count,
            "summary": summary,
            "test_signal_metrics": test_signal_metrics,
            "indicator_keys": list(indicator_dict.keys()),
            "run_dir": str(run_dir),
            "signal_path": str(signal_path) if signal is not None else None,
            "report_path": str(report_path),
            "positions_path": str(positions_path),
            "metrics_path": str(metrics_path),
            "rolling_windows_path": str(rolling_windows_path) if signal_mode in {"rolling", "dynamic", "model"} else None,
            "selected_items_path": str(selected_items_path) if signal_mode in {"rolling", "dynamic", "model"} else None,
            "net_value_curve_path": str(net_value_curve_path),
            "news_review_path": str(news_review_path) if news_review_history else None,
            "live_log_path": str(live_log_path),
        }
