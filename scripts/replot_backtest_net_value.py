"""Replot a Qlib backtest net value curve without overwriting existing output."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUN_DIR = Path(
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replot net value curves from a Qlib report_1day.csv file.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Backtest output directory containing report_1day.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. If it exists, a numeric suffix is appended.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 10000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to create a unique output path for {path}")


def nav_from_returns(return_series: pd.Series) -> pd.Series:
    nav = (1.0 + return_series.fillna(0.0)).cumprod()
    if nav.empty:
        return nav
    return nav / float(nav.iloc[0])


def load_report(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_path}")

    report = pd.read_csv(report_path, parse_dates=["datetime"])
    required_columns = {"datetime", "account", "return", "cost", "bench"}
    missing_columns = sorted(required_columns.difference(report.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {report_path}: {missing_columns}")

    report = report.sort_values("datetime").set_index("datetime")
    return report


def plot_net_value(report: pd.DataFrame, output_path: Path, dpi: int) -> None:
    account_nav = report["account"] / float(report["account"].iloc[0])
    strategy_gross_nav = nav_from_returns(report["return"])
    strategy_net_nav = nav_from_returns(report["return"] - report["cost"])
    benchmark_nav = nav_from_returns(report["bench"])
    excess_net_nav = nav_from_returns(report["return"] - report["bench"] - report["cost"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.plot(account_nav.index, account_nav, label="Strategy NAV (account, net of cost)", linewidth=2.2)
    ax.plot(strategy_gross_nav.index, strategy_gross_nav, label="Strategy NAV (gross)", linewidth=1.5, alpha=0.75)
    ax.plot(strategy_net_nav.index, strategy_net_nav, label="Strategy NAV (return - cost)", linewidth=1.5, alpha=0.75)
    ax.plot(benchmark_nav.index, benchmark_nav, label="Benchmark NAV", linewidth=1.8)
    ax.plot(excess_net_nav.index, excess_net_nav, label="Excess NAV vs Benchmark (net of cost)", linewidth=1.6)

    ax.set_title("Backtest Net Value Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Net Value")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def default_output_path(run_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / f"net_value_curve_replotted_{timestamp}.png"


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "report_1day.csv"
    output_path = args.output.expanduser().resolve() if args.output is not None else default_output_path(run_dir)
    output_path = unique_path(output_path)

    report = load_report(report_path)
    plot_net_value(report, output_path, args.dpi)

    print(f"read_file: {report_path}")
    print(f"output_file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
