#!/usr/bin/env python3
"""Recompute train/valid/test IC metrics for a Qlib factor library.

The IC and Rank IC means follow AlphaSAGE/train_GP.py:
factor values are normalized by day, daily cross-sectional Pearson/Spearman
correlations are computed against a future-return target, non-finite daily
correlations are converted to 0, and then averaged across days.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
ALPHASAGE_SRC = ROOT / "Qlib_MCP" / "workspace" / "AlphaSAGE" / "src"
DEFAULT_INPUT_DIR = ROOT / "data" / "backup_files" / "10day_IC_2"
DEFAULT_PROVIDER_URI = Path("~/.qlib/qlib_data/cn_data").expanduser()

if str(ALPHASAGE_SRC) not in sys.path:
    sys.path.insert(0, str(ALPHASAGE_SRC))

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr  # noqa: E402
from alphagen.utils.pytorch_utils import normalize_by_day  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train/valid/test IC and Rank IC for factor_library.json."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing factor_library.json and factor_library_metrics.json.",
    )
    parser.add_argument("--instruments", default="csi300", help="Qlib instrument universe.")
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2020,
        help="Same split rule as train_GP.py: train ends at YEAR-12-31.",
    )
    parser.add_argument(
        "--target-horizon-days",
        type=int,
        default=10,
        help="10 means Ref($close, -11)/Ref($close, -1)-1.",
    )
    parser.add_argument(
        "--provider-uri",
        type=Path,
        default=DEFAULT_PROVIDER_URI,
        help="Qlib provider URI.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Qlib expression batch size.")
    parser.add_argument(
        "--output-library",
        type=Path,
        default=None,
        help="Output factor library JSON. Defaults to INPUT_DIR/factor_library_recomputed_ic.json.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Output compact metrics JSON. Defaults to INPUT_DIR/factor_library_metrics_recomputed_ic.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output summary CSV. Defaults to INPUT_DIR/factor_ic_summary.csv.",
    )
    return parser.parse_args()


def load_factor_library(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [item for item in data if isinstance(item, dict)]


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_ranges(train_end_year: int) -> dict[str, tuple[str, str]]:
    return {
        "train": ("2010-01-01", f"{train_end_year}-12-31"),
        "valid": (f"{train_end_year + 1}-01-01", f"{train_end_year + 1}-12-31"),
        "test": (f"{train_end_year + 2}-01-01", f"{train_end_year + 4}-12-31"),
    }


def init_qlib(provider_uri: Path) -> None:
    import qlib
    from qlib.config import REG_CN

    resolved = provider_uri.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Qlib provider URI does not exist: {resolved}")
    qlib.init(provider_uri=str(resolved), region=REG_CN)


def load_instruments(instruments: str, start_date: str, end_date: str) -> list[str]:
    from qlib.data import D

    values = D.list_instruments(
        instruments=D.instruments(instruments),
        start_time=start_date,
        end_time=end_date,
        as_list=True,
    )
    if not values:
        raise RuntimeError(f"No instruments loaded for {instruments} {start_date}->{end_date}")
    return list(values)


def load_feature_frame(
    instruments: list[str],
    expressions: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    from qlib.data import D

    frame = D.features(instruments, expressions, start_time=start_date, end_time=end_date)
    if frame is None or frame.empty:
        raise RuntimeError(f"Qlib returned empty features for {start_date}->{end_date}")
    return frame.sort_index()


def to_day_stock_tensor(series: pd.Series, dates: list[pd.Timestamp], instruments: list[str]) -> torch.Tensor:
    panel = series.unstack("instrument")
    panel = panel.reindex(index=dates, columns=instruments)
    values = panel.to_numpy(dtype=np.float32, copy=False)
    return torch.from_numpy(values.copy())


def safe_ir(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    if std_value > 1e-12:
        return mean_value / std_value
    return 0.0


def clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isfinite(float(value)):
        return float(value)
    return 0.0


def compute_one_factor_metrics(
    factor_series: pd.Series,
    target_tensor: torch.Tensor,
    dates: list[pd.Timestamp],
    instruments: list[str],
) -> dict[str, Any]:
    factor_tensor = to_day_stock_tensor(factor_series, dates, instruments)
    factor_tensor = normalize_by_day(factor_tensor)

    daily_ic_tensor = torch.nan_to_num(
        batch_pearsonr(factor_tensor, target_tensor),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    daily_rank_ic_tensor = torch.nan_to_num(
        batch_spearmanr(factor_tensor, target_tensor),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    daily_ics = [float(value) for value in daily_ic_tensor.detach().cpu().numpy()]
    daily_rank_ics = [float(value) for value in daily_rank_ic_tensor.detach().cpu().numpy()]
    valid_counts = (
        torch.isfinite(factor_tensor) & torch.isfinite(target_tensor)
    ).sum(dim=1).detach().cpu().numpy()

    return {
        "ic": clean_float(float(np.mean(daily_ics)) if daily_ics else 0.0),
        "rank_ic": clean_float(float(np.mean(daily_rank_ics)) if daily_rank_ics else 0.0),
        "icir": clean_float(safe_ir(daily_ics)),
        "rank_icir": clean_float(safe_ir(daily_rank_ics)),
        "ic_std": clean_float(float(np.std(daily_ics)) if daily_ics else 0.0),
        "rank_ic_std": clean_float(float(np.std(daily_rank_ics)) if daily_rank_ics else 0.0),
        "ic_win_rate": clean_float(float(np.mean(np.array(daily_ics) > 0)) if daily_ics else 0.0),
        "rank_ic_win_rate": clean_float(
            float(np.mean(np.array(daily_rank_ics) > 0)) if daily_rank_ics else 0.0
        ),
        "trading_days": len(daily_ics),
        "avg_universe_count": clean_float(float(np.mean(valid_counts)) if len(valid_counts) else 0.0),
    }


def compute_split_metrics(
    factors: list[dict[str, Any]],
    split_name: str,
    start_date: str,
    end_date: str,
    instruments_name: str,
    target_expression: str,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    from qlib.data import D

    print(f"[IC] Loading {split_name}: {start_date}->{end_date}")
    instrument_list = load_instruments(instruments_name, start_date, end_date)
    calendar = [pd.Timestamp(value) for value in D.calendar(start_time=start_date, end_time=end_date)]
    if not calendar:
        raise RuntimeError(f"No calendar dates for {split_name}: {start_date}->{end_date}")

    target_frame = load_feature_frame(instrument_list, [target_expression], start_date, end_date)
    target_tensor = to_day_stock_tensor(target_frame.iloc[:, 0], calendar, instrument_list)

    result: dict[str, dict[str, Any]] = {}
    valid_factors = [
        factor
        for factor in factors
        if factor.get("qlib_expression") and factor.get("is_valid", True)
    ]
    total_batches = (len(valid_factors) + max(1, batch_size) - 1) // max(1, batch_size)

    for batch_index, start_idx in enumerate(range(0, len(valid_factors), max(1, batch_size)), start=1):
        batch = valid_factors[start_idx : start_idx + max(1, batch_size)]
        expressions = [str(factor["qlib_expression"]) for factor in batch]
        print(f"[IC] {split_name} batch {batch_index}/{total_batches}: factors={len(batch)}")
        try:
            factor_frame = load_feature_frame(instrument_list, expressions, start_date, end_date)
            factor_frame.columns = [f"factor_{idx}" for idx in range(len(batch))]
            for offset, factor in enumerate(batch):
                factor_id = str(factor.get("factor_id") or factor.get("qlib_expression"))
                result[factor_id] = compute_one_factor_metrics(
                    factor_frame.iloc[:, offset],
                    target_tensor,
                    calendar,
                    instrument_list,
                )
        except Exception as batch_exc:
            print(f"[IC] {split_name} batch failed, falling back to single factors: {batch_exc}")
            for factor in batch:
                factor_id = str(factor.get("factor_id") or factor.get("qlib_expression"))
                expression = str(factor["qlib_expression"])
                try:
                    factor_frame = load_feature_frame(instrument_list, [expression], start_date, end_date)
                    result[factor_id] = compute_one_factor_metrics(
                        factor_frame.iloc[:, 0],
                        target_tensor,
                        calendar,
                        instrument_list,
                    )
                except Exception as exc:
                    print(f"[IC] {split_name} factor failed: {factor_id}, error={exc}")
                    result[factor_id] = {
                        "ic": None,
                        "rank_ic": None,
                        "icir": None,
                        "rank_icir": None,
                        "error": str(exc),
                    }

    return result


def build_compact_metrics(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for factor in factors:
        compact.append(
            {
                "factor_id": factor.get("factor_id"),
                "qlib_expression": factor.get("qlib_expression"),
                "metrics": factor.get("metrics", {}),
                "economics_passed": factor.get("economics_passed"),
                "round_index": factor.get("round_index"),
            }
        )
    return compact


def build_summary_rows(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for factor in factors:
        row = {
            "factor_id": factor.get("factor_id"),
            "qlib_expression": factor.get("qlib_expression"),
        }
        for split_name in ("train", "valid", "test"):
            metrics = factor.get("metrics", {}).get(split_name, {})
            for key in (
                "ic",
                "rank_ic",
                "icir",
                "rank_icir",
                "ic_std",
                "rank_ic_std",
                "ic_win_rate",
                "rank_ic_win_rate",
                "trading_days",
                "avg_universe_count",
            ):
                row[f"{split_name}_{key}"] = metrics.get(key)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    input_library = input_dir / "factor_library.json"
    output_library = args.output_library or input_dir / "factor_library_recomputed_ic.json"
    output_metrics = args.output_metrics or input_dir / "factor_library_metrics_recomputed_ic.json"
    output_csv = args.output_csv or input_dir / "factor_ic_summary.csv"

    factors = load_factor_library(input_library)
    target_horizon_days = max(int(args.target_horizon_days), 1)
    target_expression = f"Ref($close, -{target_horizon_days + 1})/Ref($close, -1) - 1"

    print(f"[IC] Loaded factors: {len(factors)} from {input_library}")
    print(f"[IC] Target expression: {target_expression}")
    init_qlib(args.provider_uri)

    all_split_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, (start_date, end_date) in split_ranges(args.train_end_year).items():
        all_split_metrics[split_name] = compute_split_metrics(
            factors=factors,
            split_name=split_name,
            start_date=start_date,
            end_date=end_date,
            instruments_name=args.instruments,
            target_expression=target_expression,
            batch_size=max(1, int(args.batch_size)),
        )

    for factor in factors:
        factor_id = str(factor.get("factor_id") or factor.get("qlib_expression"))
        metrics = factor.setdefault("metrics", {})
        for split_name in ("train", "valid", "test"):
            split_metrics = all_split_metrics.get(split_name, {}).get(factor_id)
            if split_metrics is None:
                split_metrics = {
                    "ic": None,
                    "rank_ic": None,
                    "icir": None,
                    "rank_icir": None,
                    "error": "factor was skipped because qlib_expression is missing or invalid",
                }
            metrics[split_name] = split_metrics
            factor[f"{split_name}_ic"] = split_metrics.get("ic")
            factor[f"{split_name}_rank_ic"] = split_metrics.get("rank_ic")

    dump_json(output_library, factors)
    dump_json(output_metrics, build_compact_metrics(factors))
    pd.DataFrame(build_summary_rows(factors)).to_csv(output_csv, index=False)

    print(f"[IC] Wrote library: {output_library}")
    print(f"[IC] Wrote metrics: {output_metrics}")
    print(f"[IC] Wrote summary CSV: {output_csv}")


if __name__ == "__main__":
    main()
