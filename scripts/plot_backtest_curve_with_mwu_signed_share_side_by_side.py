#!/usr/bin/env python3
"""Overlay net value curves with an MWU signed-share factor heatmap."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from plot_mwu_factor_weight_heatmaps import (
    build_matrices,
    load_windows,
    normalize_columns_by_abs_max,
    signed_column_shares,
    sort_base_matrix,
)


RUN_DIR = Path(
)
DEFAULT_OUTPUT_PATH = RUN_DIR / "net_value_curve_overlay_mwu_signed_share_heatmap.png"
DEFAULT_FIG_WIDTH = 7.1
DEFAULT_FIG_HEIGHT = 3.2
DEFAULT_TITLE = "Temporal Evolution of Signed Factor Weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an overlaid backtest case figure: NAV curves on the left axis, factor heatmap on the right."
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--rolling-windows-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--weight-field", choices=("weight", "raw_weight"), default="weight")
    parser.add_argument("--signed-share-color-limit", type=float, default=0.4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-nav",
        action="store_true",
        help=(
            "Plot only the signed-share heatmap. The x-axis still follows the "
            "report date range so it aligns with the NAV overlay figure."
        ),
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=DEFAULT_FIG_WIDTH,
        help="Figure width in inches. Default is sized for a double-column paper figure.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=DEFAULT_FIG_HEIGHT,
        help="Figure height in inches. Keep compact for a double-column paper figure.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Figure title. Use an empty string to omit the title for paper captions.",
    )
    return parser.parse_args()


def _nav_from_returns(return_series: pd.Series) -> pd.Series:
    nav = (1.0 + return_series.fillna(0.0)).cumprod()
    if nav.empty:
        return nav
    return nav / float(nav.iloc[0])


def load_report(path: Path) -> pd.DataFrame:
    report_df = pd.read_csv(path)
    if "datetime" not in report_df.columns:
        raise ValueError(f"{path} must contain a datetime column.")
    missing = {"return", "bench"} - set(report_df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    report_df["datetime"] = pd.to_datetime(report_df["datetime"])
    return report_df.set_index("datetime").sort_index()


def apply_report_time_axis(ax: plt.Axes, report_df: pd.DataFrame) -> None:
    if report_df.empty:
        raise ValueError("report_df must contain at least one row to set the time axis.")

    start_date = report_df.index.min()
    end_date = report_df.index.max()
    start_num, end_num = mdates.date2num([start_date, end_date])
    ax.set_xlim(start_num, end_num)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=25, pad=1.5)


def display_factor_label(factor_id: str) -> str:
    base_id = re.sub(r"__(long|short)$", "", factor_id)
    mined_match = re.match(r"^fac_(\d+)_(\d+)(?:_[A-Za-z0-9]+)?$", base_id)
    if mined_match:
        round_index, factor_index = mined_match.groups()
        return f"F{int(round_index):02d}-{int(factor_index):02d}"
    base_match = re.match(r"^base_(\d+)$", base_id)
    if base_match:
        return f"F00-{int(base_match.group(1)):02d}"
    return base_id


def plot_nav_curves(ax: plt.Axes, report_df: pd.DataFrame) -> None:
    cost_series = (
        report_df["cost"].fillna(0.0)
        if "cost" in report_df.columns
        else pd.Series(0.0, index=report_df.index)
    )
    strategy_with_cost_nav = _nav_from_returns(report_df["return"] - cost_series)
    benchmark_nav = _nav_from_returns(report_df["bench"])

    ax.plot(
        strategy_with_cost_nav.index,
        strategy_with_cost_nav,
        label="Strategy NAV (net)",
        color="#988ED5",
        linewidth=1.1,
    )
    ax.plot(
        benchmark_nav.index,
        benchmark_nav,
        label="Benchmark NAV",
        color="#6E6E6E",
        linewidth=1.1,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Net Value")
    legend = ax.legend(
        loc="upper left",
        framealpha=0.82,
        borderpad=0.35,
        handlelength=1.8,
        labelspacing=0.28,
    )
    legend.get_frame().set_edgecolor("0.85")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=25, pad=1.5)
    ax.tick_params(axis="y", pad=1.5)


def column_date_edges(columns: pd.Index) -> np.ndarray:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for column in columns:
        label = str(column)
        match = re.match(r"^W\d+_(\d{4}-\d{2}-\d{2})(?:_to_(\d{4}-\d{2}-\d{2}))?$", label)
        if not match:
            raise ValueError(f"Cannot parse holding-period label: {label}")
        start_text, end_text = match.groups()
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text) if end_text else start
        starts.append(start)
        ends.append(end)

    if not starts:
        raise ValueError("No holding periods found for heatmap columns.")
    final_edge = ends[-1]
    if final_edge <= starts[-1]:
        final_edge = starts[-1] + pd.Timedelta(days=1)
    return mdates.date2num([*starts, final_edge])


def plot_signed_share_overlay(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    figure: plt.Figure,
    color_limit: float,
) -> None:
    values = matrix.to_numpy(dtype=float)
    limit = max(float(color_limit), 1e-12)
    image = ax.pcolormesh(
        column_date_edges(matrix.columns),
        np.arange(len(matrix.index) + 1),
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="flat",
        alpha=1.0,
        zorder=0,
    )
    colorbar = figure.colorbar(image, ax=ax, pad=0.035, fraction=0.03, aspect=30)
    colorbar.set_label("Signed weight share", labelpad=2)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    colorbar.ax.tick_params(labelsize=6.5, pad=1)

    ax.set_ylabel("Factor", labelpad=3)
    ax.set_ylim(len(matrix.index), 0)
    ax.set_yticks(np.arange(len(matrix.index)) + 0.5)
    ax.set_yticklabels([display_factor_label(str(factor_id)) for factor_id in matrix.index])
    ax.tick_params(axis="y", labelsize=5.8, pad=1.5)
    ax.grid(False)


def build_signed_share_matrix(windows: list[dict[str, Any]], weight_field: str) -> pd.DataFrame:
    _, diff_matrix, _ = build_matrices(windows, weight_field)
    diff_matrix = sort_base_matrix(diff_matrix)
    return signed_column_shares(normalize_columns_by_abs_max(diff_matrix))


def apply_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else run_dir / "report_1day.csv"
    )
    rolling_windows_path = (
        args.rolling_windows_path.expanduser().resolve()
        if args.rolling_windows_path is not None
        else run_dir / "rolling_windows.json"
    )
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_df = load_report(report_path)
    windows = load_windows(rolling_windows_path)
    signed_share_matrix = build_signed_share_matrix(windows, args.weight_field)
    no_nav = args.no_nav or output_path.stem.endswith("_no_nav")

    apply_paper_style()
    if no_nav:
        fig, heatmap_ax = plt.subplots(
            figsize=(args.fig_width, args.fig_height),
            constrained_layout=True,
        )
        plot_signed_share_overlay(
            heatmap_ax,
            signed_share_matrix,
            fig,
            args.signed_share_color_limit,
        )
        apply_report_time_axis(heatmap_ax, report_df)
        if args.title:
            heatmap_ax.set_title(args.title, pad=3.0)
    else:
        fig, curve_ax = plt.subplots(
            figsize=(args.fig_width, args.fig_height),
            constrained_layout=True,
        )
        heatmap_ax = curve_ax.twinx()
        heatmap_ax.set_zorder(0)
        curve_ax.set_zorder(1)
        heatmap_ax.patch.set_alpha(0.0)
        curve_ax.patch.set_alpha(0.0)
        plot_signed_share_overlay(
            heatmap_ax,
            signed_share_matrix,
            fig,
            args.signed_share_color_limit,
        )
        plot_nav_curves(curve_ax, report_df)
        apply_report_time_axis(curve_ax, report_df)
        if args.title:
            curve_ax.set_title(args.title, pad=3.0)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"report_path: {report_path}")
    print(f"rolling_windows_path: {rolling_windows_path}")
    print(f"windows: {len(windows)}")
    print(f"base_factor_count: {len(signed_share_matrix.index)}")
    print(f"no_nav: {no_nav}")
    print(f"overlay_figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
