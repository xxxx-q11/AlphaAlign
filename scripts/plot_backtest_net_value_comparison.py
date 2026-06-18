"""Compare net-of-cost NAV curves from multiple Qlib backtest outputs."""
from __future__ import annotations

import argparse
import fnmatch
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATHS = (REPO_ROOT / "data" / "backtest_outputs",)
DEFAULT_REPORT_NAME = "report_1day.csv"

# Optional edit-only filters. Leave empty to include every discovered report.
# Examples: ("linear_weighting_20260413_223554", "ExperimentMWU*")
DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = ("ExperimentMWU_rebalance_window_10_trainday_90_IC_06_2025_ValidationSet_IC_Filter_Descending_50Factors_Rolling","ExperimentMWU_rebalance_window_10_trainday_90_IC_06_2025_ValidationSet_IC_Filter_Descending_50Factors_Rolling_News")
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot net-of-cost NAV curves for several backtest outputs and add a benchmark baseline. "
            "Inputs can be a backtest root directory, run directories, or report_1day.csv files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUT_PATHS),
        help="Input paths. Defaults to data/backtest_outputs under this repo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to comparison_net_value_<timestamp>.png under the first input directory.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=list(DEFAULT_INCLUDE_PATTERNS),
        help="Only include runs matching this glob pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=list(DEFAULT_EXCLUDE_PATTERNS),
        help="Exclude runs matching this glob pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap after sorting report paths. Useful when a folder contains many runs.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date, e.g. 2025-01-01.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date, e.g. 2025-07-01.")
    parser.add_argument(
        "--net-source",
        choices=("auto", "account", "return_minus_cost"),
        default="auto",
        help="Source for strategy net NAV. auto prefers account and falls back to return - cost.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("first", "all", "none"),
        default="first",
        help="Plot one benchmark baseline from the first included report, all benchmark baselines, or none.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("name", "relative", "path"),
        default="name",
        help="How to label each run in the legend.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser


def resolve_input_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to create a unique output path for {path}")


def discover_report_paths(paths: list[Path], report_name: str = DEFAULT_REPORT_NAME) -> list[Path]:
    report_paths: set[Path] = set()
    for raw_path in paths:
        path = resolve_input_path(raw_path)
        if path.is_file():
            if path.name != report_name:
                raise ValueError(f"Input file is not {report_name}: {path}")
            report_paths.add(path)
            continue

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        direct_report = path / report_name
        if direct_report.exists():
            report_paths.add(direct_report)
        else:
            report_paths.update(path.rglob(report_name))

    return sorted(report_paths)


def match_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def filter_report_paths(report_paths: list[Path], include: list[str], exclude: list[str], max_runs: int | None) -> list[Path]:
    selected: list[Path] = []
    for report_path in report_paths:
        run_dir = report_path.parent
        labels = (run_dir.name, str(run_dir), str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else "")
        if include and not any(match_any(label, include) for label in labels):
            continue
        if exclude and any(match_any(label, exclude) for label in labels):
            continue
        selected.append(report_path)

    if max_runs is not None:
        selected = selected[: max(max_runs, 0)]
    return selected


def load_report(report_path: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    report = pd.read_csv(report_path, parse_dates=["datetime"])
    required_columns = {"datetime", "return", "bench"}
    missing_columns = sorted(required_columns.difference(report.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {report_path}: {missing_columns}")

    report = report.sort_values("datetime").set_index("datetime")
    if start_date:
        report = report.loc[report.index >= pd.Timestamp(start_date)]
    if end_date:
        report = report.loc[report.index <= pd.Timestamp(end_date)]
    return report


def nav_from_returns(return_series: pd.Series) -> pd.Series:
    nav = (1.0 + pd.to_numeric(return_series, errors="coerce").fillna(0.0)).cumprod()
    if nav.empty:
        return nav
    first_value = float(nav.iloc[0])
    if first_value == 0.0:
        return nav
    return nav / first_value


def strategy_net_nav(report: pd.DataFrame, net_source: str) -> tuple[pd.Series, str]:
    if net_source in {"auto", "account"} and "account" in report.columns:
        account = pd.to_numeric(report["account"], errors="coerce").dropna()
        if not account.empty and float(account.iloc[0]) != 0.0:
            return account / float(account.iloc[0]), "account"
        if net_source == "account":
            raise ValueError("account column is empty or starts from zero.")

    cost = pd.to_numeric(report["cost"], errors="coerce").fillna(0.0) if "cost" in report.columns else 0.0
    return nav_from_returns(report["return"] - cost), "return - cost"


def benchmark_nav(report: pd.DataFrame) -> pd.Series:
    return nav_from_returns(report["bench"])


def make_label(report_path: Path, label_mode: str) -> str:
    run_dir = report_path.parent
    if label_mode == "path":
        return str(run_dir)
    if label_mode == "relative":
        try:
            return str(run_dir.relative_to(REPO_ROOT))
        except ValueError:
            return str(run_dir)
    return run_dir.name


def default_output_path(first_input: Path) -> Path:
    resolved = resolve_input_path(first_input)
    output_dir = resolved if resolved.is_dir() else resolved.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"comparison_net_value_{timestamp}.png"


def plot_comparison(
    loaded_reports: list[tuple[Path, pd.DataFrame, pd.Series, str]],
    output_path: Path,
    *,
    benchmark_mode: str,
    label_mode: str,
    dpi: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 7.2))

    for report_path, _, nav, _ in loaded_reports:
        ax.plot(nav.index, nav, label=make_label(report_path, label_mode), linewidth=1.8)

    if benchmark_mode != "none":
        benchmark_reports = loaded_reports[:1] if benchmark_mode == "first" else loaded_reports
        for report_path, report, _, _ in benchmark_reports:
            label = "Benchmark" if benchmark_mode == "first" else f"Benchmark - {make_label(report_path, label_mode)}"
            ax.plot(benchmark_nav(report).index, benchmark_nav(report), label=label, linewidth=2.0, linestyle="--", color="black" if benchmark_mode == "first" else None, alpha=0.85)

    ax.axhline(1.0, color="#555555", linewidth=0.9, alpha=0.45)
    ax.set_title("Backtest Net Value Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value, normalized to 1.0")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    report_paths = discover_report_paths(args.paths)
    report_paths = filter_report_paths(report_paths, args.include, args.exclude, args.max_runs)
    if not report_paths:
        raise FileNotFoundError("No report_1day.csv files matched the input paths and filters.")

    loaded_reports: list[tuple[Path, pd.DataFrame, pd.Series, str]] = []
    for report_path in report_paths:
        report = load_report(report_path, args.start_date, args.end_date)
        if report.empty:
            print(f"skip_empty_after_date_filter: {report_path}")
            continue
        net_nav, source = strategy_net_nav(report, args.net_source)
        loaded_reports.append((report_path, report, net_nav, source))

    if not loaded_reports:
        raise ValueError("All matched reports are empty after applying date filters.")

    output_path = args.output.expanduser() if args.output is not None else default_output_path(args.paths[0])
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path = unique_path(output_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparison(
        loaded_reports,
        output_path,
        benchmark_mode=args.benchmark_mode,
        label_mode=args.label_mode,
        dpi=args.dpi,
    )

    print(f"output_file: {output_path}")
    print(f"report_count: {len(loaded_reports)}")
    for report_path, _, nav, source in loaded_reports:
        print(f"read_file: {report_path}")
        print(f"  net_source: {source}")
        print(f"  final_nav: {float(nav.iloc[-1]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
