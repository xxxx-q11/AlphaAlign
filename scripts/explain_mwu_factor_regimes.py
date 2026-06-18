#!/usr/bin/env python3
"""Explain MWU factor regimes from a rolling backtest output directory.

The script turns MWU long/short expert weights into a reproducible explanation
dataset, then optionally asks an LLM to write period-by-period interpretations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data"
    / "backtest_outputs"
    / "IC_01_No_Base_pool"
)
DEFAULT_FACTOR_DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read MWU backtest outputs, compute expert/base/direction explanation "
            "statistics, and optionally call an LLM for holding-period narratives."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Backtest output directory containing MWU CSVs and rolling_windows.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to RUN_DIR/mwu_factor_regime_explanations.",
    )
    parser.add_argument(
        "--factor-data-dir",
        type=Path,
        default=DEFAULT_FACTOR_DATA_DIR,
        help="Directory containing factor_library.json and factor_library_metrics.json.",
    )
    parser.add_argument(
        "--llm-config-path",
        type=Path,
        default=None,
        help="Optional config/env.yaml path. If omitted, config/env.yaml is auto-detected.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("per-period", "none"),
        default="per-period",
        help="Use per-period LLM calls, or skip LLM and use local template explanations.",
    )
    parser.add_argument(
        "--max-llm-periods",
        type=int,
        default=None,
        help="Debug limit for LLM calls. Statistics are still computed for all periods.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top factors/experts to include in prompts and Markdown.",
    )
    parser.add_argument(
        "--market-up-threshold",
        type=float,
        default=0.02,
        help="Period benchmark return above this value is labeled as rising market.",
    )
    parser.add_argument(
        "--market-down-threshold",
        type=float,
        default=-0.02,
        help="Period benchmark return below this value is labeled as falling market.",
    )
    parser.add_argument(
        "--expert-significant-threshold",
        type=float,
        default=None,
        help="Expert weight threshold for significant experts. Defaults to max(3/N, 3%%).",
    )
    parser.add_argument(
        "--expert-core-threshold",
        type=float,
        default=None,
        help="Expert weight threshold for core experts. Defaults to max(5/N, 5%%).",
    )
    parser.add_argument(
        "--expert-dominant-threshold",
        type=float,
        default=None,
        help="Expert weight threshold for dominant experts. Defaults to max(10/N, 10%%).",
    )
    parser.add_argument(
        "--base-significant-threshold",
        type=float,
        default=None,
        help="Base factor total weight threshold. Defaults to max(2/M, 4%%).",
    )
    parser.add_argument(
        "--base-core-threshold",
        type=float,
        default=None,
        help="Base factor total weight core threshold. Defaults to max(3/M, 6%%).",
    )
    parser.add_argument(
        "--base-dominant-threshold",
        type=float,
        default=None,
        help="Base factor total weight dominant threshold. Defaults to max(5/M, 10%%).",
    )
    parser.add_argument(
        "--direction-significant-share",
        type=float,
        default=0.05,
        help="abs(long-short contribution share) threshold for significant direction factors.",
    )
    parser.add_argument(
        "--direction-dominant-share",
        type=float,
        default=0.10,
        help="abs(long-short contribution share) threshold for dominant direction factors.",
    )
    return parser.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def resolve_factor_data_paths(factor_data_dir: Path) -> tuple[Path, Path]:
    factor_data_dir = factor_data_dir.expanduser().resolve()
    if not factor_data_dir.is_dir():
        raise FileNotFoundError(f"Factor data directory not found: {factor_data_dir}")
    factor_library_path = factor_data_dir / "factor_library.json"
    factor_metrics_path = factor_data_dir / "factor_library_metrics.json"
    missing = [str(path) for path in (factor_library_path, factor_metrics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Factor data directory must contain factor_library.json and "
            f"factor_library_metrics.json. Missing: {', '.join(missing)}"
        )
    return factor_library_path, factor_metrics_path


def clean_factor_id(value: Any) -> str:
    return str(value or "").strip()


def split_expert_id(factor_id: str) -> tuple[str, str]:
    match = re.match(r"^(.*)__(long|short)$", factor_id)
    if match:
        return match.group(1), match.group(2)
    return factor_id, ""


def period_key(window: dict[str, Any]) -> str:
    index = int(window.get("window_index") or 0)
    start = str(window.get("holding_start") or window.get("selection_date") or "")
    end = str(window.get("holding_end") or "")
    if start and end:
        return f"W{index:02d}_{start}_to_{end}"
    if start:
        return f"W{index:02d}_{start}"
    return f"W{index:02d}"


def short_period_label(period: str) -> str:
    match = re.match(r"^(W\d+)_(\d{4}-\d{2}-\d{2})(?:_to_(\d{4}-\d{2}-\d{2}))?$", period)
    if not match:
        return period
    window, start, end = match.groups()
    if end:
        return f"{window} {start}~{end}"
    return f"{window} {start}"


def load_windows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "rolling_windows.json"
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    windows = [item for item in payload if isinstance(item, dict)]
    if not windows:
        raise ValueError(f"No rolling windows found in {path}.")
    return sorted(windows, key=lambda item: int(item.get("window_index") or 0))


def iter_selected_factors(window: dict[str, Any]) -> list[dict[str, Any]]:
    selected = window.get("selected_factors")
    if isinstance(selected, list) and selected:
        return [item for item in selected if isinstance(item, dict)]
    selected = window.get("selected_items")
    if not isinstance(selected, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        factor = item.get("factor")
        merged = dict(item)
        if isinstance(factor, dict):
            merged = {**factor, **item}
        rows.append(merged)
    return rows


def load_factor_info(
    factor_library_path: Path,
    factor_metrics_path: Path,
    windows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}

    library = read_json(factor_library_path, default=[])
    if isinstance(library, list):
        for item in library:
            if not isinstance(item, dict):
                continue
            factor_id = clean_factor_id(item.get("factor_id"))
            if not factor_id:
                continue
            info[factor_id] = {
                "factor_id": factor_id,
                "source": item.get("source"),
                "round_index": item.get("round_index"),
                "qlib_expression": item.get("qlib_expression") or item.get("expression"),
                "economics_passed": item.get("economics_passed"),
                "economics_reason": item.get("economics_reason"),
            }

    metrics = read_json(factor_metrics_path, default=[])
    if isinstance(metrics, list):
        for item in metrics:
            if not isinstance(item, dict):
                continue
            factor_id = clean_factor_id(item.get("factor_id"))
            if not factor_id:
                continue
            row = info.setdefault(factor_id, {"factor_id": factor_id})
            if item.get("qlib_expression") and not row.get("qlib_expression"):
                row["qlib_expression"] = item.get("qlib_expression")
            if item.get("economics_passed") is not None:
                row["economics_passed"] = item.get("economics_passed")
            if item.get("economics_reason") and not row.get("economics_reason"):
                row["economics_reason"] = item.get("economics_reason")
            metric = item.get("metrics")
            if isinstance(metric, dict):
                test_metric = metric.get("test") or {}
                if isinstance(test_metric, dict):
                    row["test_ic"] = test_metric.get("ic")
                    row["test_rank_ic"] = test_metric.get("rank_ic")

    for window in windows:
        for item in iter_selected_factors(window):
            factor_id = clean_factor_id(item.get("factor_id"))
            base_id = clean_factor_id(item.get("base_factor_id")) or split_expert_id(factor_id)[0]
            if not base_id:
                continue
            row = info.setdefault(base_id, {"factor_id": base_id})
            if item.get("qlib_expression") and not row.get("qlib_expression"):
                row["qlib_expression"] = item.get("qlib_expression")
            if item.get("economics_passed") is not None:
                row["economics_passed"] = item.get("economics_passed")
            if item.get("economics_reason") and not row.get("economics_reason"):
                row["economics_reason"] = item.get("economics_reason")
            factor = item.get("factor")
            if isinstance(factor, dict):
                if factor.get("qlib_expression") and not row.get("qlib_expression"):
                    row["qlib_expression"] = factor.get("qlib_expression")
                if factor.get("source") and not row.get("source"):
                    row["source"] = factor.get("source")
                if factor.get("round_index") is not None and row.get("round_index") is None:
                    row["round_index"] = factor.get("round_index")
                if factor.get("economics_passed") is not None:
                    row["economics_passed"] = factor.get("economics_passed")
                if factor.get("economics_reason") and not row.get("economics_reason"):
                    row["economics_reason"] = factor.get("economics_reason")

    for factor_id, row in info.items():
        row.update(classify_factor(factor_id, row.get("qlib_expression") or ""))
        row["economic_meaning"] = row.get("economics_reason")
    return info


def classify_factor(factor_id: str, expression: str) -> dict[str, str]:
    expr = expression or ""
    lower = expr.lower().replace(" ", "")

    if factor_id == "base_009" or "ref($close,60)/$close" in lower:
        return {
            "factor_style": "60-Day Price Position / Momentum Reversal",
            "style_group": "Price Position",
            "high_value_meaning": "High factor value generally indicates the current price is lower relative to 60 days ago; the stock is in a relative weakness or low-level recovery state.",
            "long_meaning": "Prefers low-level recovery or mean reversion.",
            "short_meaning": "Prefers relatively strong price positions, reflecting momentum or trend continuation.",
        }
    if "wma($open" in lower or "ema($close" in lower or "mean($open" in lower:
        return {
            "factor_style": "Medium-to-Long Term Moving Average / Price Position",
            "style_group": "Price Position",
            "high_value_meaning": "The factor value characterizes the deviation of medium-to-long-term average price from the current price center; higher values often correspond to relatively low positions or price recovery potential.",
            "long_meaning": "Prefers relatively low positions, recovery, or mean-reversion characteristics.",
            "short_meaning": "Prefers stocks where current price is stronger relative to long-term center, reflecting strong momentum.",
        }
    if "mean($low" in lower or "mean($close" in lower and "div(" in lower:
        return {
            "factor_style": "Recent Price Position / Low-Point Relationship",
            "style_group": "Price Position",
            "high_value_meaning": "The factor value reflects the relative position of recent lows, average prices, and closing price centers.",
            "long_meaning": "Prefers stocks with stable price positions or low-level recovery characteristics.",
            "short_meaning": "Prefers stocks with stronger price centers or more evident upward momentum.",
        }
    if "rsquare($close" in lower:
        return {
            "factor_style": "Trend Fit Quality",
            "style_group": "Price Trend",
            "high_value_meaning": "High factor value indicates the price path is closer to a linear trend, with a clearer trend pattern.",
            "long_meaning": "Prefers stocks with clearer trends.",
            "short_meaning": "Prefers stocks with lower trend fit, structural changes, or stronger rotational dynamics.",
        }
    if "resi($close" in lower:
        return {
            "factor_style": "Price Residual / Mean Reversion",
            "style_group": "Price Position",
            "high_value_meaning": "The factor value characterizes the residual deviation of price relative to a short-term fitted trend.",
            "long_meaning": "Prefers positive deviations or short-term strength.",
            "short_meaning": "Prefers recovery after negative deviations, or avoiding short-term overheating.",
        }
    if "$high-$low" in lower or "std(abs($close/ref($close,1)-1)*$volume" in lower or "std($close" in lower:
        return {
            "factor_style": "Volatility / Risk Shock",
            "style_group": "Volatility Risk",
            "high_value_meaning": "High factor value indicates greater price fluctuations, intraday amplitude, or volume-price shocks.",
            "long_meaning": "Prefers high-volatility risk compensation or risk appetite expansion.",
            "short_meaning": "Prefers low volatility, stability, or defensive attributes.",
        }
    if "corr" in lower and ("$vwap" in lower or "$volume" in lower or "log($volume" in lower):
        return {
            "factor_style": "Volume-Price Synergy / Liquidity Structure",
            "style_group": "Volume-Price Liquidity",
            "high_value_meaning": "High factor value indicates that the relationships among price, volume, or VWAP are stronger, more stable, or more consistent.",
            "long_meaning": "Prefers stocks with volume-price confirmation and clearer liquidity structure.",
            "short_meaning": "Prefers stocks with volume-price relationship deviations, structural changes, or breakout patterns.",
        }
    if "$volume" in lower or "mad($volume" in lower:
        return {
            "factor_style": "Volume Stability / Activity",
            "style_group": "Liquidity",
            "high_value_meaning": "High factor value generally indicates more prominent volume levels, stability, or activity.",
            "long_meaning": "Prefers stocks with more active trading or more stable liquidity.",
            "short_meaning": "Prefers stocks with abnormal trading volume, low stability, or greater liquidity changes.",
        }
    if "corr" in lower:
        return {
            "factor_style": "Correlation Structure",
            "style_group": "Structural Correlation",
            "high_value_meaning": "High factor value indicates a stronger correlation structure among the expression's internal variables.",
            "long_meaning": "Prefers stocks with more stable or stronger correlation structures.",
            "short_meaning": "Prefers stocks with weaker, deviating, or more variable correlation structures.",
        }
    return {
        "factor_style": "Other Composite Factor",
        "style_group": "Other",
        "high_value_meaning": "The factor value meaning requires further interpretation in conjunction with the expression.",
        "long_meaning": "Prefers stocks with higher factor values.",
        "short_meaning": "Prefers stocks with lower factor values.",
    }


def importance_level(value: float, significant: float, core: float, dominant: float) -> str:
    if value >= dominant:
        return "dominant"
    if value >= core:
        return "core"
    if value >= significant:
        return "significant"
    return "normal"


def compute_thresholds(args: argparse.Namespace, expert_count: int, base_count: int) -> dict[str, float]:
    expert_avg = 1.0 / max(expert_count, 1)
    base_avg = 1.0 / max(base_count, 1)
    return {
        "expert_average_weight": expert_avg,
        "base_average_weight": base_avg,
        "expert_significant": (
            args.expert_significant_threshold
            if args.expert_significant_threshold is not None
            else max(3.0 * expert_avg, 0.03)
        ),
        "expert_core": (
            args.expert_core_threshold
            if args.expert_core_threshold is not None
            else max(5.0 * expert_avg, 0.05)
        ),
        "expert_dominant": (
            args.expert_dominant_threshold
            if args.expert_dominant_threshold is not None
            else max(10.0 * expert_avg, 0.10)
        ),
        "base_significant": (
            args.base_significant_threshold
            if args.base_significant_threshold is not None
            else max(2.0 * base_avg, 0.04)
        ),
        "base_core": (
            args.base_core_threshold
            if args.base_core_threshold is not None
            else max(3.0 * base_avg, 0.06)
        ),
        "base_dominant": (
            args.base_dominant_threshold
            if args.base_dominant_threshold is not None
            else max(5.0 * base_avg, 0.10)
        ),
        "direction_significant_share": args.direction_significant_share,
        "direction_dominant_share": args.direction_dominant_share,
    }


def build_selected_factor_frame(windows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for window in windows:
        period = period_key(window)
        window_index = int(window.get("window_index") or 0)
        for item in iter_selected_factors(window):
            factor_id = clean_factor_id(item.get("factor_id"))
            if not factor_id:
                continue
            base_id = clean_factor_id(item.get("base_factor_id")) or split_expert_id(factor_id)[0]
            expert_label = clean_factor_id(item.get("expert_label")).lower()
            if expert_label not in {"long", "short"}:
                _, expert_label = split_expert_id(factor_id)
            records.append(
                {
                    "window_index": window_index,
                    "period": period,
                    "factor_id": factor_id,
                    "base_factor_id": base_id,
                    "expert_label": expert_label,
                    "recent_score": item.get("recent_score"),
                    "recent_ir": item.get("recent_ir"),
                    "recent_std": item.get("recent_std"),
                    "transformed_reward_mean": item.get("transformed_reward_mean"),
                    "raw_weight": item.get("raw_weight"),
                    "selected_weight": item.get("weight"),
                    "qlib_expression": item.get("qlib_expression"),
                    "recent_score_source": item.get("recent_score_source"),
                }
            )
    return pd.DataFrame(records)


def build_expert_stats(
    run_dir: Path,
    windows: list[dict[str, Any]],
    factor_info: dict[str, dict[str, Any]],
    selected_frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    path = run_dir / "mwu_all_expert_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    matrix = pd.read_csv(path)
    if "factor_id" not in matrix.columns:
        raise ValueError(f"{path} must contain factor_id column.")

    period_to_window = {period_key(window): window for window in windows}
    period_columns = [column for column in matrix.columns if column != "factor_id"]
    records: list[dict[str, Any]] = []
    for _, row in matrix.iterrows():
        factor_id = clean_factor_id(row["factor_id"])
        base_id, expert_label = split_expert_id(factor_id)
        info = factor_info.get(base_id, {})
        for period in period_columns:
            window = period_to_window.get(period, {})
            weight = float(row[period])
            level = importance_level(
                weight,
                thresholds["expert_significant"],
                thresholds["expert_core"],
                thresholds["expert_dominant"],
            )
            records.append(
                {
                    "window_index": int(window.get("window_index") or 0),
                    "period": period,
                    "holding_start": window.get("holding_start"),
                    "holding_end": window.get("holding_end"),
                    "factor_id": factor_id,
                    "base_factor_id": base_id,
                    "expert_label": expert_label,
                    "weight": weight,
                    "importance_level": level,
                    "is_high_weight": level != "normal",
                    "factor_style": info.get("factor_style"),
                    "style_group": info.get("style_group"),
                    "economic_meaning": info.get("economic_meaning"),
                    "economics_reason": info.get("economics_reason"),
                    "economics_passed": info.get("economics_passed"),
                    "qlib_expression": info.get("qlib_expression"),
                    "test_ic": info.get("test_ic"),
                    "test_rank_ic": info.get("test_rank_ic"),
                }
            )
    expert_stats = pd.DataFrame(records)
    if not selected_frame.empty:
        merge_cols = [
            "period",
            "factor_id",
            "recent_score",
            "recent_ir",
            "recent_std",
            "transformed_reward_mean",
            "recent_score_source",
        ]
        available_cols = [column for column in merge_cols if column in selected_frame.columns]
        expert_stats = expert_stats.merge(
            selected_frame[available_cols].drop_duplicates(["period", "factor_id"]),
            on=["period", "factor_id"],
            how="left",
        )
    return expert_stats.sort_values(["window_index", "weight"], ascending=[True, False]).reset_index(drop=True)


def build_base_stats(
    expert_stats: pd.DataFrame,
    factor_info: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    grouped = (
        expert_stats.groupby(["window_index", "period", "holding_start", "holding_end", "base_factor_id", "expert_label"])[
            "weight"
        ]
        .sum()
        .reset_index()
    )
    wide = grouped.pivot_table(
        index=["window_index", "period", "holding_start", "holding_end", "base_factor_id"],
        columns="expert_label",
        values="weight",
        fill_value=0.0,
        aggfunc="sum",
    ).reset_index()
    for column in ("long", "short"):
        if column not in wide.columns:
            wide[column] = 0.0
    wide = wide.rename(columns={"long": "long_weight", "short": "short_weight"})
    wide["total_weight"] = wide["long_weight"] + wide["short_weight"]
    wide["net_long_minus_short"] = wide["long_weight"] - wide["short_weight"]
    wide["dominant_direction"] = wide["net_long_minus_short"].apply(lambda value: "long" if value >= 0 else "short")
    wide["base_importance_level"] = wide["total_weight"].apply(
        lambda value: importance_level(
            float(value),
            thresholds["base_significant"],
            thresholds["base_core"],
            thresholds["base_dominant"],
        )
    )
    wide["is_high_base_weight"] = wide["base_importance_level"] != "normal"

    abs_denominator = wide.groupby("period")["net_long_minus_short"].transform(lambda col: col.abs().sum())
    wide["direction_abs_share"] = wide["net_long_minus_short"].abs() / abs_denominator.replace(0.0, pd.NA)
    wide["direction_abs_share"] = wide["direction_abs_share"].fillna(0.0)
    wide["direction_signed_share"] = wide["net_long_minus_short"] / abs_denominator.replace(0.0, pd.NA)
    wide["direction_signed_share"] = wide["direction_signed_share"].fillna(0.0)
    wide["direction_importance_level"] = wide["direction_abs_share"].apply(
        lambda value: (
            "dominant"
            if value >= thresholds["direction_dominant_share"]
            else "significant"
            if value >= thresholds["direction_significant_share"]
            else "normal"
        )
    )
    wide["is_high_direction_contribution"] = wide["direction_importance_level"] != "normal"

    for column in (
        "factor_style",
        "style_group",
        "economic_meaning",
        "economics_reason",
        "economics_passed",
        "high_value_meaning",
        "long_meaning",
        "short_meaning",
        "qlib_expression",
        "test_ic",
        "test_rank_ic",
    ):
        wide[column] = wide["base_factor_id"].map(lambda fid: factor_info.get(fid, {}).get(column))
    wide["direction_meaning"] = wide.apply(
        lambda row: format_direction_meaning(row),
        axis=1,
    )
    wide["implied_style"] = wide.apply(infer_implied_style, axis=1)
    return wide.sort_values(["window_index", "total_weight"], ascending=[True, False]).reset_index(drop=True)


def format_direction_meaning(row: pd.Series) -> str:
    economic_meaning = row.get("economic_meaning") or row.get("economics_reason")
    direction = str(row.get("dominant_direction") or "")
    if economic_meaning:
        return f"{direction} direction based on the factor's economic meaning: {economic_meaning}"
    return f"{direction} direction: no economics_reason provided; cannot derive original economic meaning."


def infer_implied_style(row: pd.Series) -> str:
    style_group = str(row.get("style_group") or "")
    direction = str(row.get("dominant_direction") or "")
    if style_group == "Price Position":
        return "Strong Momentum / Trend Continuation" if direction == "short" else "Low-Level Reversal / Mean Reversion"
    if style_group == "Price Trend":
        return "Trend Weakening / Rotation" if direction == "short" else "Trend Continuation"
    if style_group == "Volume-Price Liquidity":
        return "Volume-Price Divergence / Structural Change" if direction == "short" else "Volume-Price Confirmation / Liquidity Quality"
    if style_group == "Liquidity":
        return "Trading Abnormality / Liquidity Change" if direction == "short" else "Stable Trading / Active Liquidity"
    if style_group == "Volatility Risk":
        return "Low Volatility Defense" if direction == "short" else "High Volatility Risk Appetite"
    return str(row.get("factor_style") or "Other")


def build_market_stats(
    run_dir: Path,
    windows: list[dict[str, Any]],
    up_threshold: float,
    down_threshold: float,
) -> pd.DataFrame:
    path = run_dir / "report_1day.csv"
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "window_index": int(window.get("window_index") or 0),
                    "period": period_key(window),
                    "holding_start": window.get("holding_start"),
                    "holding_end": window.get("holding_end"),
                    "market_style": "unknown",
                }
                for window in windows
            ]
        )

    report = pd.read_csv(path, parse_dates=["datetime"])
    report = report.sort_values("datetime")
    volatility_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for window in windows:
        start = pd.to_datetime(window.get("holding_start"))
        end = pd.to_datetime(window.get("holding_end"))
        period = period_key(window)
        subset = report[(report["datetime"] >= start) & (report["datetime"] <= end)].copy()
        if subset.empty:
            rows.append(
                {
                    "window_index": int(window.get("window_index") or 0),
                    "period": period,
                    "holding_start": window.get("holding_start"),
                    "holding_end": window.get("holding_end"),
                    "strategy_return": None,
                    "benchmark_return": None,
                    "excess_return": None,
                    "benchmark_volatility_annualized": None,
                    "mean_turnover": None,
                }
            )
            volatility_values.append(0.0)
            continue
        strategy_return = float(subset["account"].iloc[-1] / subset["account"].iloc[0] - 1.0)
        benchmark_return = float((1.0 + subset["bench"].astype(float)).prod() - 1.0)
        excess_return = float((1.0 + strategy_return) / (1.0 + benchmark_return) - 1.0)
        volatility = float(subset["bench"].astype(float).std(ddof=0) * math.sqrt(252)) if len(subset) > 1 else 0.0
        volatility_values.append(volatility)
        rows.append(
            {
                "window_index": int(window.get("window_index") or 0),
                "period": period,
                "holding_start": window.get("holding_start"),
                "holding_end": window.get("holding_end"),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "benchmark_volatility_annualized": volatility,
                "mean_turnover": float(subset["turnover"].astype(float).mean()) if "turnover" in subset else None,
            }
        )

    non_null_vol = [value for value in volatility_values if value is not None]
    median_volatility = float(pd.Series(non_null_vol).median()) if non_null_vol else 0.0
    for row in rows:
        benchmark_return = row.get("benchmark_return")
        volatility = row.get("benchmark_volatility_annualized")
        if benchmark_return is None:
            trend_label = "unknown"
        elif benchmark_return > up_threshold:
            trend_label = "rising"
        elif benchmark_return < down_threshold:
            trend_label = "falling"
        else:
            trend_label = "ranging"
        if volatility is None:
            volatility_label = "unknown volatility"
        else:
            volatility_label = "high volatility" if float(volatility) >= median_volatility else "low volatility"
        row["market_trend_label"] = trend_label
        row["market_volatility_label"] = volatility_label
        row["market_style"] = f"{trend_label}-{volatility_label}"
        row["benchmark_volatility_median"] = median_volatility
    return pd.DataFrame(rows)


def top_records(df: pd.DataFrame, sort_column: str, n: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = df.sort_values(sort_column, ascending=False).head(n)
    return [json_clean(record) for record in rows.to_dict("records")]


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_period_payloads(
    windows: list[dict[str, Any]],
    expert_stats: pd.DataFrame,
    base_stats: pd.DataFrame,
    market_stats: pd.DataFrame,
    thresholds: dict[str, float],
    top_n: int,
) -> list[dict[str, Any]]:
    market_by_period = market_stats.set_index("period").to_dict("index") if not market_stats.empty else {}
    payloads: list[dict[str, Any]] = []
    for window in windows:
        period = period_key(window)
        period_experts = expert_stats[expert_stats["period"] == period]
        period_bases = base_stats[base_stats["period"] == period]
        high_experts = period_experts[period_experts["is_high_weight"]].sort_values("weight", ascending=False)
        if high_experts.empty:
            high_experts = period_experts.sort_values("weight", ascending=False).head(top_n)

        high_bases = period_bases[period_bases["is_high_base_weight"]].sort_values("total_weight", ascending=False)
        if high_bases.empty:
            high_bases = period_bases.sort_values("total_weight", ascending=False).head(top_n)

        high_directions = period_bases[period_bases["is_high_direction_contribution"]].sort_values(
            "direction_abs_share", ascending=False
        )
        if high_directions.empty:
            high_directions = period_bases.sort_values("direction_abs_share", ascending=False).head(top_n)

        style_exposure = (
            period_bases.groupby(["style_group", "factor_style"], dropna=False)["total_weight"]
            .sum()
            .reset_index()
            .sort_values("total_weight", ascending=False)
        )
        implied_style = (
            high_directions.groupby("implied_style", dropna=False)["direction_abs_share"]
            .sum()
            .reset_index()
            .sort_values("direction_abs_share", ascending=False)
        )

        weights = period_experts["weight"].sort_values(ascending=False)
        effective_n = float(1.0 / (period_experts["weight"].pow(2).sum())) if not period_experts.empty else 0.0
        long_total = float(period_experts.loc[period_experts["expert_label"] == "long", "weight"].sum())
        short_total = float(period_experts.loc[period_experts["expert_label"] == "short", "weight"].sum())
        concentration = {
            "top1_weight": float(weights.iloc[0]) if len(weights) else 0.0,
            "top5_weight_sum": float(weights.head(5).sum()),
            "top10_weight_sum": float(weights.head(10).sum()),
            "effective_expert_count": effective_n,
            "long_total_weight": long_total,
            "short_total_weight": short_total,
            "net_long_minus_short": long_total - short_total,
        }

        payload = {
            "window_index": int(window.get("window_index") or 0),
            "period": period,
            "period_label": short_period_label(period),
            "holding_start": window.get("holding_start"),
            "holding_end": window.get("holding_end"),
            "market": json_clean(market_by_period.get(period, {})),
            "thresholds": json_clean(thresholds),
            "concentration": json_clean(concentration),
            "top_high_experts": top_records(high_experts, "weight", top_n),
            "top_high_base_factors": top_records(high_bases, "total_weight", top_n),
            "top_direction_contributors": top_records(high_directions, "direction_abs_share", top_n),
            "style_exposure": top_records(style_exposure, "total_weight", top_n),
            "implied_style_exposure": top_records(implied_style, "direction_abs_share", top_n),
        }
        payload["local_explanation"] = build_local_explanation(payload)
        payloads.append(payload)
    return payloads


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "NA"


def build_local_explanation(payload: dict[str, Any]) -> dict[str, str]:
    market = payload.get("market") or {}
    market_style = market.get("market_style") or "unknown"
    strategy_return = pct(market.get("strategy_return"))
    benchmark_return = pct(market.get("benchmark_return"))
    excess_return = pct(market.get("excess_return"))

    styles = payload.get("style_exposure") or []
    style_text = ", ".join(
        f"{item.get('factor_style') or item.get('style_group')}({pct(item.get('total_weight'))})"
        for item in styles[:3]
    )
    directions = payload.get("top_direction_contributors") or []
    direction_text = "; ".join(
        (
            f"{item.get('base_factor_id')} {item.get('dominant_direction')} "
            f"{pct(item.get('direction_abs_share'))}, "
            f"{factor_economic_meaning(item)}"
        )
        for item in directions[:3]
    )
    high_base = payload.get("top_high_base_factors") or []
    base_text = ", ".join(
        (
            f"{item.get('base_factor_id')}({pct(item.get('total_weight'))}, "
            f"{factor_economic_meaning(item)})"
        )
        for item in high_base[:3]
    )
    period_explanation = (
        f"{payload['period_label']} Market state: {market_style}. Strategy return: {strategy_return}. "
        f"Benchmark return: {benchmark_return}. Excess return: {excess_return}. "
        f"High-weight base factors are concentrated in: {base_text or 'no significantly high-weight factors'}. "
        f"Style exposure is mainly: {style_text or 'no obvious style concentration'}. "
        f"Direction contributions show: {direction_text or 'direction contributions are fairly dispersed'}."
    )
    paper_sentence = (
        f"During the {market_style} phase, MWU primarily constructs signals through {style_text or 'diversified factor styles'}, "
        f"and at the directional level this manifests as {direction_text or 'balanced multi-factor allocation'}."
    )
    return {
        "market_style_summary": (
            f"Market: {market_style}. Strategy: {strategy_return}. Benchmark: {benchmark_return}. Excess: {excess_return}."
        ),
        "dominant_factor_styles": style_text or "diversified styles",
        "direction_interpretation": direction_text or "direction contributions are dispersed",
        "period_explanation": period_explanation,
        "risk_or_caveat": "This explanation is based on index returns, volatility, and MWU weights from the backtest output, without introducing additional sector or macro external data.",
        "paper_sentence": paper_sentence,
    }


def factor_economic_meaning(record: dict[str, Any]) -> str:
    return str(record.get("economic_meaning") or record.get("economics_reason") or "no economics_reason provided")


def discover_llm_config(path: Path | None) -> Path | None:
    if path is not None:
        return path
    default_path = REPO_ROOT / "config" / "env.yaml"
    if default_path.exists():
        return default_path
    return None


def create_llm_agent(config_path: Path | None) -> Any | None:
    resolved = discover_llm_config(config_path)
    if resolved is None:
        print("[WARN] No LLM config found. Use --llm-config-path or create config/env.yaml. Falling back locally.")
        return None
    if not resolved.exists():
        print(f"[WARN] LLM config not found: {resolved}. Falling back locally.")
        return None
    sys.path.insert(0, str(REPO_ROOT))
    from Agent.agent_factory import create_agent, load_env_config

    config = dict(load_env_config(str(resolved)))
    provider = config.get("provider", "qwen")
    api_key = config.get("api_key")
    if not api_key or str(api_key).startswith("YOUR_"):
        print(f"[WARN] LLM api_key is not configured in {resolved}. Falling back locally.")
        return None
    return create_agent(
        provider=provider,
        api_key=api_key,
        model=config.get("model"),
        base_url=config.get("base_url"),
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=config.get("max_tokens") or 1200,
        timeout=config.get("timeout", 300),
    )


def response_content(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {
        "market_style_summary": "",
        "dominant_factor_styles": "",
        "direction_interpretation": "",
        "period_explanation": text,
        "risk_or_caveat": "LLM did not return parseable JSON; raw text preserved.",
        "paper_sentence": "",
    }


def build_llm_prompt(payload: dict[str, Any]) -> str:
    compact_payload = {
        "period_label": payload["period_label"],
        "market": payload["market"],
        "concentration": payload["concentration"],
        "thresholds": payload["thresholds"],
        "top_high_experts": trim_records(payload["top_high_experts"]),
        "top_high_base_factors": trim_records(payload["top_high_base_factors"]),
        "top_direction_contributors": trim_records(payload["top_direction_contributors"]),
        "style_exposure": trim_records(payload["style_exposure"]),
        "implied_style_exposure": trim_records(payload["implied_style_exposure"]),
        "local_explanation": payload["local_explanation"],
    }
    return (
        "You are a quantitative investment paper writing assistant. Please strictly explain the holding period of an MWU "
        "long-short expert weighted model based on the JSON data below; do not fabricate external market events.\n"
        "Please primarily refer to top_direction_contributors to explain the model's style selection and directional implications for the period; "
        "top_high_experts, top_high_base_factors, style_exposure, and implied_style_exposure "
        "serve only as auxiliary context and must not replace the main thread of direction-contributing factors.\n"
        "When explaining the economic meaning of factors, you must prioritize economics_reason/economic_meaning; "
        "factor_style, style_group, and implied_style are only coarse-grained style labels and must not replace economic meaning.\n"
        "When explaining direction-contributing factors, please prioritize direction_abs_share, direction_signed_share, "
        "dominant_direction, long_weight, short_weight, net_long_minus_short, economics_reason, "
        "economic_meaning, direction_meaning, factor_style, style_group, implied_style, and qlib_expression.\n"
        "You should explain: 1) the current market state; 2) which factors contribute most to direction and their styles; "
        '3) prefix long factors with "+" and short factors with "-", explaining the meaning of these direction-contributing factors in the current market; '
        "4) whether high-weight experts/base factors support this directional contribution main thread; 5) how this explanation contributes to a paper narrative; "
        "6) necessary risk caveats.\n"
        "Please return JSON with fixed fields: market_style_summary, dominant_factor_styles, "
        "direction_interpretation, period_explanation, risk_or_caveat, paper_sentence.\n\n"
        f"Data:\n{json.dumps(compact_payload, ensure_ascii=False, indent=2)}"
    )


def trim_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "factor_id",
        "base_factor_id",
        "expert_label",
        "weight",
        "total_weight",
        "long_weight",
        "short_weight",
        "net_long_minus_short",
        "dominant_direction",
        "direction_abs_share",
        "direction_signed_share",
        "factor_style",
        "style_group",
        "economic_meaning",
        "economics_reason",
        "economics_passed",
        "implied_style",
        "direction_meaning",
        "recent_score",
        "transformed_reward_mean",
        "test_ic",
        "test_rank_ic",
        "qlib_expression",
    }
    trimmed: list[dict[str, Any]] = []
    for record in records:
        trimmed.append({key: record.get(key) for key in keep if key in record})
    return trimmed


def call_llm_for_payloads(
    payloads: list[dict[str, Any]],
    agent: Any | None,
    max_periods: int | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    prompt_path = output_dir / "period_llm_prompts.jsonl"
    response_path = output_dir / "period_llm_responses.jsonl"
    explained: list[dict[str, Any]] = []
    with prompt_path.open("w", encoding="utf-8") as prompt_file, response_path.open("w", encoding="utf-8") as response_file:
        for index, payload in enumerate(payloads):
            prompt = build_llm_prompt(payload)
            prompt_file.write(json.dumps({"period": payload["period"], "prompt": prompt}, ensure_ascii=False) + "\n")
            if agent is None or (max_periods is not None and index >= max_periods):
                llm_result = payload["local_explanation"]
                raw_response = ""
                status = "local_fallback"
            else:
                try:
                    response = agent.call(
                        prompt=prompt,
                        system_prompt="You only provide quantitative factor explanations based on user-supplied data, and output in JSON.",
                        stream=False,
                    )
                    raw_response = response_content(response)
                    llm_result = parse_llm_json(raw_response)
                    status = "llm_success"
                except Exception as exc:
                    raw_response = str(exc)
                    llm_result = {
                        **payload["local_explanation"],
                        "risk_or_caveat": f"LLM call failed; using local template explanation: {exc}",
                    }
                    status = "llm_failed_local_fallback"
            payload = dict(payload)
            payload["explanation"] = llm_result
            payload["llm_status"] = status
            explained.append(payload)
            response_file.write(
                json.dumps(
                    {
                        "period": payload["period"],
                        "status": status,
                        "raw_response": raw_response,
                        "parsed": llm_result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[Explain] {payload['period_label']}: {status}")
    return explained


def flatten_period_explanations(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        market = payload.get("market") or {}
        concentration = payload.get("concentration") or {}
        explanation = payload.get("explanation") or payload.get("local_explanation") or {}
        top_base = payload.get("top_high_base_factors") or []
        top_direction = payload.get("top_direction_contributors") or []
        rows.append(
            {
                "window_index": payload.get("window_index"),
                "period": payload.get("period"),
                "holding_start": payload.get("holding_start"),
                "holding_end": payload.get("holding_end"),
                "market_style": market.get("market_style"),
                "strategy_return": market.get("strategy_return"),
                "benchmark_return": market.get("benchmark_return"),
                "excess_return": market.get("excess_return"),
                "benchmark_volatility_annualized": market.get("benchmark_volatility_annualized"),
                "long_total_weight": concentration.get("long_total_weight"),
                "short_total_weight": concentration.get("short_total_weight"),
                "top5_weight_sum": concentration.get("top5_weight_sum"),
                "effective_expert_count": concentration.get("effective_expert_count"),
                "top_base_factors": "; ".join(
                    (
                        f"{item.get('base_factor_id')}:{pct(item.get('total_weight'))}:"
                        f"{factor_economic_meaning(item)}"
                    )
                    for item in top_base[:5]
                ),
                "top_direction_factors": "; ".join(
                    (
                        f"{item.get('base_factor_id')}:{item.get('dominant_direction')}:"
                        f"{pct(item.get('direction_abs_share'))}:"
                        f"{factor_economic_meaning(item)}"
                    )
                    for item in top_direction[:5]
                ),
                "llm_status": payload.get("llm_status"),
                "market_style_summary": explanation.get("market_style_summary"),
                "dominant_factor_styles": explanation.get("dominant_factor_styles"),
                "direction_interpretation": explanation.get("direction_interpretation"),
                "period_explanation": explanation.get("period_explanation"),
                "risk_or_caveat": explanation.get("risk_or_caveat"),
                "paper_sentence": explanation.get("paper_sentence"),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    run_dir: Path,
    thresholds: dict[str, float],
    payloads: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# MWU Factor Style Explanation Report")
    lines.append("")
    lines.append(f"- Backtest directory: `{run_dir}`")
    lines.append(f"- Expert significant/core/dominant thresholds: {pct(thresholds['expert_significant'])} / {pct(thresholds['expert_core'])} / {pct(thresholds['expert_dominant'])}")
    lines.append(f"- Base factor significant/core/dominant thresholds: {pct(thresholds['base_significant'])} / {pct(thresholds['base_core'])} / {pct(thresholds['base_dominant'])}")
    lines.append(f"- Direction contribution significant/dominant thresholds: {pct(thresholds['direction_significant_share'])} / {pct(thresholds['direction_dominant_share'])}")
    lines.append("")

    for payload in payloads:
        market = payload.get("market") or {}
        concentration = payload.get("concentration") or {}
        explanation = payload.get("explanation") or payload.get("local_explanation") or {}
        lines.append(f"## {payload['period_label']}")
        lines.append("")
        lines.append(
            "- Market & Performance: "
            f"{market.get('market_style', 'unknown')}, "
            f"strategy {pct(market.get('strategy_return'))}, "
            f"benchmark {pct(market.get('benchmark_return'))}, "
            f"excess {pct(market.get('excess_return'))}, "
            f"ann. volatility {pct(market.get('benchmark_volatility_annualized'))}."
        )
        lines.append(
            "- Weight Concentration: "
            f"Top5 experts total {pct(concentration.get('top5_weight_sum'))}, "
            f"effective expert count {concentration.get('effective_expert_count', 0):.1f}, "
            f"long/short total weight {pct(concentration.get('long_total_weight'))}/"
            f"{pct(concentration.get('short_total_weight'))}."
        )
        lines.append("- High-Weight Base Factors:")
        for item in (payload.get("top_high_base_factors") or [])[:5]:
            lines.append(
                f"  - `{item.get('base_factor_id')}` {pct(item.get('total_weight'))}, "
                f"{factor_economic_meaning(item)}, "
                f"direction `{item.get('dominant_direction')}`, "
                f"{item.get('direction_meaning')}"
            )
        lines.append("- Direction Contribution Factors:")
        for item in (payload.get("top_direction_contributors") or [])[:5]:
            lines.append(
                f"  - `{item.get('base_factor_id')}` {item.get('dominant_direction')}, "
                f"direction contribution {pct(item.get('direction_abs_share'))}, "
                f"economic meaning: {factor_economic_meaning(item)}"
            )
        lines.append("")
        lines.append(explanation.get("period_explanation") or "")
        if explanation.get("paper_sentence"):
            lines.append("")
            lines.append(f"> {explanation['paper_sentence']}")
        if explanation.get("risk_or_caveat"):
            lines.append("")
            lines.append(f"Risk note: {explanation['risk_or_caveat']}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "mwu_factor_regime_explanations")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_library_path, factor_metrics_path = resolve_factor_data_paths(args.factor_data_dir)

    windows = load_windows(run_dir)
    selected_frame = build_selected_factor_frame(windows)
    factor_info = load_factor_info(factor_library_path, factor_metrics_path, windows)

    raw_weights = pd.read_csv(run_dir / "mwu_all_expert_weights.csv")
    expert_count = int(raw_weights["factor_id"].nunique())
    base_count = int(raw_weights["factor_id"].map(lambda value: split_expert_id(clean_factor_id(value))[0]).nunique())
    thresholds = compute_thresholds(args, expert_count=expert_count, base_count=base_count)

    expert_stats = build_expert_stats(run_dir, windows, factor_info, selected_frame, thresholds)
    base_stats = build_base_stats(expert_stats, factor_info, thresholds)
    market_stats = build_market_stats(
        run_dir,
        windows,
        up_threshold=args.market_up_threshold,
        down_threshold=args.market_down_threshold,
    )
    payloads = build_period_payloads(
        windows=windows,
        expert_stats=expert_stats,
        base_stats=base_stats,
        market_stats=market_stats,
        thresholds=thresholds,
        top_n=args.top_n,
    )

    agent = None
    if args.llm_mode != "none":
        agent = create_llm_agent(args.llm_config_path)
    explained_payloads = call_llm_for_payloads(
        payloads=payloads,
        agent=agent,
        max_periods=args.max_llm_periods,
        output_dir=output_dir,
    )

    expert_stats.to_csv(output_dir / "all_expert_period_stats.csv", index=False)
    expert_stats[expert_stats["is_high_weight"]].to_csv(output_dir / "high_weight_expert_stats.csv", index=False)
    base_stats.to_csv(output_dir / "base_factor_period_stats.csv", index=False)
    base_stats[base_stats["is_high_base_weight"]].to_csv(output_dir / "high_weight_base_factor_stats.csv", index=False)
    base_stats[base_stats["is_high_direction_contribution"]].to_csv(
        output_dir / "high_direction_contribution_stats.csv", index=False
    )
    market_stats.to_csv(output_dir / "period_market_style.csv", index=False)
    flatten_period_explanations(explained_payloads).to_csv(output_dir / "period_factor_explanations.csv", index=False)
    write_json(output_dir / "period_explanation_payloads.json", explained_payloads)
    write_json(
        output_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "factor_data_dir": str(args.factor_data_dir.expanduser().resolve()),
            "factor_library_path": str(factor_library_path),
            "factor_metrics_path": str(factor_metrics_path),
            "expert_count": expert_count,
            "base_factor_count": base_count,
            "period_count": len(windows),
            "thresholds": thresholds,
            "llm_mode": args.llm_mode,
        },
    )
    write_markdown_report(
        output_path=output_dir / "period_factor_explanations.md",
        run_dir=run_dir,
        thresholds=thresholds,
        payloads=explained_payloads,
    )

    print(f"[Done] Wrote explanation outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
