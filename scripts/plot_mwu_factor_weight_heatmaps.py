#!/usr/bin/env python3
"""Plot MWU factor/expert weights from rolling backtest outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data"
    / "backtest_outputs"
    / "IC_01_No_Base_pool"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read MWU rolling_windows.json and plot five charts: "
            "all long/short expert weights, base-factor long-short weight differences, "
            "column-normalized long-short differences, signed column-share differences, "
            "and base-factor column-share bubbles."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Backtest output directory containing rolling_windows.json.",
    )
    parser.add_argument(
        "--rolling-windows-path",
        type=Path,
        default=None,
        help="Optional explicit path to rolling_windows.json. Overrides --run-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory for heatmap PNG and CSV outputs. Defaults to RUN_DIR/mwu_weight_heatmaps.",
    )
    parser.add_argument(
        "--weight-field",
        choices=("weight", "raw_weight"),
        default="weight",
        help="Weight field to read from selected_factors.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("mining_order", "first_window", "mean_weight", "factor_id"),
        default="mining_order",
        help="Row order for the all-expert heatmap.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved heatmap images.",
    )
    parser.add_argument(
        "--weight-vmax-percentile",
        type=float,
        default=95.0,
        help="Color-scale upper percentile for the all-expert weight heatmap.",
    )
    parser.add_argument(
        "--diff-vmax-percentile",
        type=float,
        default=95.0,
        help="Symmetric color-scale percentile for abs(long-short diff).",
    )
    parser.add_argument(
        "--share-marker-max-size",
        type=float,
        default=220.0,
        help="Maximum marker area for the base-factor share chart.",
    )
    parser.add_argument(
        "--signed-share-color-limit",
        type=float,
        default=0.4,
        help="Symmetric color limit for signed column-share heatmap. Default 0.4 means +/-40%%.",
    )
    return parser.parse_args()


def load_windows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    windows = [item for item in payload if isinstance(item, dict)]
    if not windows:
        raise ValueError(f"No rolling windows found in {path}.")
    return sorted(windows, key=lambda item: int(item.get("window_index") or 0))


def clean_factor_id(value: Any) -> str:
    return str(value or "").strip()


def base_factor_id(factor_id: str, item: dict[str, Any]) -> str:
    explicit = clean_factor_id(item.get("base_factor_id"))
    if explicit:
        return explicit
    return re.sub(r"__(long|short)$", "", factor_id)


def expert_label(factor_id: str, item: dict[str, Any]) -> str:
    label = clean_factor_id(item.get("expert_label")).lower()
    if label in {"long", "short"}:
        return label
    match = re.search(r"__(long|short)$", factor_id)
    if match:
        return match.group(1)
    direction = item.get("expert_direction")
    if direction == 1:
        return "long"
    if direction == -1:
        return "short"
    return ""


def mining_order_key(factor_id: str) -> tuple[Any, ...]:
    base_id = re.sub(r"__(long|short)$", "", factor_id)
    base_match = re.match(r"^base_(\d+)$", base_id)
    if base_match:
        return (0, int(base_match.group(1)), base_id)

    mined_match = re.match(r"^fac_(\d+)_(\d+)_([A-Za-z0-9]+)$", base_id)
    if mined_match:
        round_index, factor_index, hash_suffix = mined_match.groups()
        return (1, int(round_index), int(factor_index), hash_suffix, base_id)

    return (2, base_id)


def expert_mining_order_key(factor_id: str) -> tuple[Any, ...]:
    label = "long" if factor_id.endswith("__long") else "short" if factor_id.endswith("__short") else ""
    direction_rank = {"long": 0, "short": 1}.get(label, 2)
    return (*mining_order_key(factor_id), direction_rank, factor_id)


def period_key(window: dict[str, Any]) -> str:
    index = int(window.get("window_index") or 0)
    start = str(window.get("holding_start") or window.get("selection_date") or "")
    end = str(window.get("holding_end") or "")
    if start and end:
        return f"W{index:02d}_{start}_to_{end}"
    if start:
        return f"W{index:02d}_{start}"
    return f"W{index:02d}"


def display_period_label(label: str) -> str:
    match = re.match(r"^(W\d+)_(\d{4}-\d{2}-\d{2})(?:_to_(\d{4}-\d{2}-\d{2}))?$", label)
    if not match:
        return label
    window, start, end = match.groups()
    if end:
        return f"{window}\n{start[5:]}-{end[5:]}"
    return f"{window}\n{start[5:]}"


def iter_selected_factors(window: dict[str, Any]) -> list[dict[str, Any]]:
    selected = window.get("selected_factors")
    if isinstance(selected, list) and selected:
        return [item for item in selected if isinstance(item, dict)]
    selected = window.get("selected_items")
    if isinstance(selected, list) and selected:
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
    return []


def build_matrices(
    windows: list[dict[str, Any]], weight_field: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expert_records: list[dict[str, Any]] = []
    direction_records: list[dict[str, Any]] = []
    base_order: list[str] = []
    expert_order: list[str] = []

    for window in windows:
        window_index = int(window.get("window_index") or 0)
        label = period_key(window)
        for item in iter_selected_factors(window):
            factor_id = clean_factor_id(item.get("factor_id"))
            if not factor_id:
                continue
            weight = item.get(weight_field)
            if weight is None:
                continue
            weight_value = float(weight)
            base_id = base_factor_id(factor_id, item)
            direction = expert_label(factor_id, item)

            if factor_id not in expert_order:
                expert_order.append(factor_id)
            if base_id and base_id not in base_order:
                base_order.append(base_id)

            expert_records.append(
                {
                    "window_index": window_index,
                    "period": label,
                    "factor_id": factor_id,
                    "weight": weight_value,
                }
            )
            if direction in {"long", "short"} and base_id:
                direction_records.append(
                    {
                        "window_index": window_index,
                        "period": label,
                        "base_factor_id": base_id,
                        "direction": direction,
                        "weight": weight_value,
                    }
                )

    if not expert_records:
        raise ValueError(f"No selected_factors with {weight_field!r} were found.")

    expert_df = pd.DataFrame(expert_records)
    expert_matrix = expert_df.pivot_table(
        index="factor_id",
        columns="period",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )

    periods = [period_key(window) for window in windows]
    expert_matrix = expert_matrix.reindex(columns=periods, fill_value=0.0)

    direction_df = pd.DataFrame(direction_records)
    direction_wide = direction_df.pivot_table(
        index=["base_factor_id", "period"],
        columns="direction",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for direction in ("long", "short"):
        if direction not in direction_wide.columns:
            direction_wide[direction] = 0.0
    direction_wide["long_minus_short"] = direction_wide["long"] - direction_wide["short"]
    direction_wide["dominant_direction"] = np.where(
        direction_wide["long"] >= direction_wide["short"], "long", "short"
    )

    diff_matrix = direction_wide.pivot(
        index="base_factor_id", columns="period", values="long_minus_short"
    ).fillna(0.0)
    diff_matrix = diff_matrix.reindex(index=base_order, columns=periods, fill_value=0.0)

    dominant_matrix = direction_wide.pivot(
        index="base_factor_id", columns="period", values="dominant_direction"
    ).fillna("")
    dominant_matrix = dominant_matrix.reindex(index=base_order, columns=periods, fill_value="")

    expert_matrix.attrs["expert_order"] = expert_order
    return expert_matrix, diff_matrix, dominant_matrix


def sort_expert_matrix(matrix: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "mining_order":
        return matrix.reindex(sorted(matrix.index, key=expert_mining_order_key))
    if sort_by == "factor_id":
        return matrix.sort_index()
    if sort_by == "mean_weight":
        return matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    expert_order = matrix.attrs.get("expert_order")
    if isinstance(expert_order, list):
        return matrix.reindex([factor_id for factor_id in expert_order if factor_id in matrix.index])
    return matrix


def sort_base_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.reindex(sorted(matrix.index, key=mining_order_key))


def figure_size(row_count: int, col_count: int) -> tuple[float, float]:
    width = min(26.0, max(14.0, 5.0 + col_count * 0.62))
    height = min(34.0, max(10.0, 2.5 + row_count * 0.23))
    return width, height


def percentile_limit(values: np.ndarray, percentile: float, fallback: float) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback
    percentile = float(np.clip(percentile, 0.0, 100.0))
    limit = float(np.nanpercentile(finite_values, percentile))
    if limit <= 0:
        return fallback
    return limit


def plot_weight_heatmap(
    matrix: pd.DataFrame, output_path: Path, dpi: int, vmax_percentile: float
) -> float:
    values = matrix.to_numpy(dtype=float)
    positive_values = values[values > 0]
    fallback = float(np.nanmax(values)) if values.size else 1.0
    vmax = percentile_limit(positive_values, vmax_percentile, max(fallback, 1e-12))
    fig, ax = plt.subplots(figsize=figure_size(len(matrix.index), len(matrix.columns)))
    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            linewidths=0.0,
            cbar_kws={"label": f"MWU weight (clipped at p{vmax_percentile:g})"},
        )
    else:
        image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        fig.colorbar(image, ax=ax, label=f"MWU weight (clipped at p{vmax_percentile:g})")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)

    ax.set_title(f"MWU Long/Short Expert Weights ({len(matrix.index)} experts)")
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Long/short expert")
    ax.set_xticklabels([display_period_label(label.get_text()) for label in ax.get_xticklabels()])
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return vmax


def plot_diff_heatmap(
    matrix: pd.DataFrame, output_path: Path, dpi: int, vmax_percentile: float
) -> float:
    values = matrix.to_numpy(dtype=float)
    abs_values = np.abs(values)
    max_abs = float(np.nanmax(abs_values)) if abs_values.size else 0.0
    limit = percentile_limit(abs_values, vmax_percentile, max(max_abs, 1e-12))
    fig, ax = plt.subplots(figsize=figure_size(len(matrix.index), len(matrix.columns)))
    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.0,
            cbar_kws={"label": f"Long weight - short weight (clipped at p{vmax_percentile:g})"},
        )
    else:
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        fig.colorbar(image, ax=ax, label=f"Long weight - short weight (clipped at p{vmax_percentile:g})")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)

    ax.set_title(f"MWU Long-Short Weight Difference ({len(matrix.index)} base factors)")
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Base factor")
    ax.set_xticklabels([display_period_label(label.get_text()) for label in ax.get_xticklabels()])
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return limit


def normalize_columns_by_abs_max(matrix: pd.DataFrame) -> pd.DataFrame:
    denominators = matrix.abs().max(axis=0).replace(0.0, np.nan)
    normalized = matrix.divide(denominators, axis=1).fillna(0.0)
    return normalized.clip(lower=-1.0, upper=1.0)


def column_shares_from_abs_values(matrix: pd.DataFrame) -> pd.DataFrame:
    abs_values = matrix.abs()
    denominators = abs_values.sum(axis=0).replace(0.0, np.nan)
    return abs_values.divide(denominators, axis=1).fillna(0.0)


def signed_column_shares(matrix: pd.DataFrame) -> pd.DataFrame:
    denominators = matrix.abs().sum(axis=0).replace(0.0, np.nan)
    return matrix.divide(denominators, axis=1).fillna(0.0)


def plot_normalized_diff_heatmap(matrix: pd.DataFrame, output_path: Path, dpi: int) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figure_size(len(matrix.index), len(matrix.columns)))
    if sns is not None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.0,
            cbar_kws={"label": "Column-normalized long-short diff"},
        )
    else:
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(image, ax=ax, label="Column-normalized long-short diff")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)

    ax.set_title(f"Column-Normalized MWU Long-Short Difference ({len(matrix.index)} base factors)")
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Base factor")
    ax.set_xticklabels([display_period_label(label.get_text()) for label in ax.get_xticklabels()])
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_signed_share_heatmap(
    matrix: pd.DataFrame, output_path: Path, dpi: int, color_limit: float
) -> None:
    values = matrix.to_numpy(dtype=float)
    limit = max(float(color_limit), 1e-12)
    fig, ax = plt.subplots(figsize=figure_size(len(matrix.index), len(matrix.columns)))
    if sns is not None:
        plot = sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.0,
            cbar_kws={
                "label": f"Signed column share (red=long, blue=short; clipped at +/-{limit:.0%})"
            },
        )
        colorbar = plot.collections[0].colorbar
    else:
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        colorbar = fig.colorbar(
            image,
            ax=ax,
            label=f"Signed column share (red=long, blue=short; clipped at +/-{limit:.0%})",
        )
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
    if colorbar is not None:
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.set_title(f"Signed Column-Share MWU Long-Short Difference ({len(matrix.index)} base factors)")
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Base factor")
    ax.set_xticklabels([display_period_label(label.get_text()) for label in ax.get_xticklabels()])
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_share_bubble_chart(
    share_matrix: pd.DataFrame,
    signed_matrix: pd.DataFrame,
    output_path: Path,
    dpi: int,
    max_marker_size: float,
) -> None:
    fig, ax = plt.subplots(figsize=figure_size(len(share_matrix.index), len(share_matrix.columns)))
    values = share_matrix.to_numpy(dtype=float)
    signed_values = signed_matrix.reindex_like(share_matrix).to_numpy(dtype=float)
    max_share = float(np.nanmax(values)) if values.size else 0.0
    size_scale = max(float(max_marker_size), 1.0)
    marker_sizes = np.where(max_share > 0, values / max_share * size_scale, 0.0)

    x_positions = np.tile(np.arange(len(share_matrix.columns)), len(share_matrix.index))
    y_positions = np.repeat(np.arange(len(share_matrix.index)), len(share_matrix.columns))

    scatter = ax.scatter(
        x_positions,
        y_positions,
        s=marker_sizes.ravel(),
        c=signed_values.ravel(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        alpha=0.9,
        linewidths=0.15,
        edgecolors="0.35",
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.78)
    colorbar.set_label("Column-normalized long-short diff")

    if max_share > 0:
        legend_shares = sorted(
            {
                value
                for value in (0.01, 0.03, 0.05, 0.10)
                if value <= max_share * 1.05
            }
        )
        if not legend_shares:
            legend_shares = [max_share]
        handles = [
            ax.scatter(
                [],
                [],
                s=max(value / max_share * size_scale, 1.0),
                facecolors="none",
                edgecolors="0.35",
                linewidths=0.6,
                label=f"{value:.0%}",
            )
            for value in legend_shares
        ]
        ax.legend(
            handles=handles,
            title="Column share",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )

    ax.set_title(f"Base-Factor Long-Short Share by Holding Period ({len(share_matrix.index)} base factors)")
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Base factor")
    ax.set_xticks(np.arange(len(share_matrix.columns)))
    ax.set_xticklabels([display_period_label(str(column)) for column in share_matrix.columns])
    ax.set_yticks(np.arange(len(share_matrix.index)))
    ax.set_yticklabels(share_matrix.index)
    ax.set_xlim(-0.5, len(share_matrix.columns) - 0.5)
    ax.set_ylim(len(share_matrix.index) - 0.5, -0.5)
    ax.grid(True, which="major", axis="both", linewidth=0.25, color="0.88")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    rolling_windows_path = (
        args.rolling_windows_path.expanduser().resolve()
        if args.rolling_windows_path is not None
        else run_dir / "rolling_windows.json"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "mwu_weight_heatmaps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(rolling_windows_path)
    expert_matrix, diff_matrix, dominant_matrix = build_matrices(windows, args.weight_field)
    expert_matrix = sort_expert_matrix(expert_matrix, args.sort_by)
    diff_matrix = sort_base_matrix(diff_matrix)
    dominant_matrix = sort_base_matrix(dominant_matrix)

    expert_csv_path = output_dir / "mwu_all_expert_weights.csv"
    diff_csv_path = output_dir / "mwu_base_factor_long_short_diff.csv"
    normalized_diff_csv_path = output_dir / "mwu_base_factor_long_short_diff_column_normalized.csv"
    signed_share_csv_path = output_dir / "mwu_base_factor_long_short_diff_signed_column_share.csv"
    share_csv_path = output_dir / "mwu_base_factor_long_short_diff_column_share.csv"
    dominant_csv_path = output_dir / "mwu_base_factor_dominant_direction.csv"
    expert_png_path = output_dir / "mwu_all_expert_weights_heatmap.png"
    diff_png_path = output_dir / "mwu_base_factor_long_short_diff_heatmap.png"
    normalized_diff_png_path = output_dir / "mwu_base_factor_long_short_diff_column_normalized_heatmap.png"
    signed_share_png_path = output_dir / "mwu_base_factor_long_short_diff_signed_column_share_heatmap.png"
    share_png_path = output_dir / "mwu_base_factor_long_short_diff_column_share_bubble.png"

    normalized_diff_matrix = normalize_columns_by_abs_max(diff_matrix)
    signed_share_matrix = signed_column_shares(normalized_diff_matrix)
    share_matrix = column_shares_from_abs_values(normalized_diff_matrix)

    expert_matrix.to_csv(expert_csv_path, encoding="utf-8")
    diff_matrix.to_csv(diff_csv_path, encoding="utf-8")
    normalized_diff_matrix.to_csv(normalized_diff_csv_path, encoding="utf-8")
    signed_share_matrix.to_csv(signed_share_csv_path, encoding="utf-8")
    share_matrix.to_csv(share_csv_path, encoding="utf-8")
    dominant_matrix.to_csv(dominant_csv_path, encoding="utf-8")
    weight_vmax = plot_weight_heatmap(
        expert_matrix, expert_png_path, args.dpi, args.weight_vmax_percentile
    )
    diff_limit = plot_diff_heatmap(diff_matrix, diff_png_path, args.dpi, args.diff_vmax_percentile)
    plot_normalized_diff_heatmap(normalized_diff_matrix, normalized_diff_png_path, args.dpi)
    plot_signed_share_heatmap(
        signed_share_matrix,
        signed_share_png_path,
        args.dpi,
        args.signed_share_color_limit,
    )
    plot_share_bubble_chart(
        share_matrix,
        normalized_diff_matrix,
        share_png_path,
        args.dpi,
        args.share_marker_max_size,
    )

    print(f"windows: {len(windows)}")
    expert_count = len(expert_matrix.index)
    base_factor_count = len(diff_matrix.index)
    print(f"expert_count: {expert_count}")
    print(f"base_factor_count: {base_factor_count}")
    if base_factor_count:
        print(f"expert_to_base_ratio: {expert_count / base_factor_count:.6g}")
    if expert_count % 2 == 0 and base_factor_count != expert_count // 2:
        print(
            "warning: base_factor_count is not exactly half of expert_count; "
            "check whether every base factor has both long and short experts."
        )
    print(f"expert_weights_csv: {expert_csv_path}")
    print(f"long_short_diff_csv: {diff_csv_path}")
    print(f"column_normalized_long_short_diff_csv: {normalized_diff_csv_path}")
    print(f"signed_column_share_long_short_diff_csv: {signed_share_csv_path}")
    print(f"column_share_long_short_diff_csv: {share_csv_path}")
    print(f"dominant_direction_csv: {dominant_csv_path}")
    print(f"expert_weights_heatmap: {expert_png_path}")
    print(f"long_short_diff_heatmap: {diff_png_path}")
    print(f"column_normalized_long_short_diff_heatmap: {normalized_diff_png_path}")
    print(f"signed_column_share_long_short_diff_heatmap: {signed_share_png_path}")
    print(f"column_share_long_short_diff_bubble_chart: {share_png_path}")
    print(f"weight_color_vmax: {weight_vmax:.8g}")
    print(f"diff_color_abs_limit: {diff_limit:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
