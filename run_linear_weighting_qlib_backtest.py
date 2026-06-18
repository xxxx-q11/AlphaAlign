"""CLI wrapper for the reusable Qlib backtest service."""
from __future__ import annotations

import argparse
from pathlib import Path

from Agent.services.qlib_backtest_service import QlibBacktestService


def _print_metric_tree(payload: dict, *, indent: int = 2) -> None:
    prefix = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_metric_tree(value, indent=indent + 2)
        else:
            print(f"{prefix}{key}: {float(value):.6f}")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser."""
    parser = argparse.ArgumentParser(description="Step-by-step Qlib backtest for linear weighting output")
    parser.add_argument("--weighting-path", type=str, default=None, help="Path to a weighting_*.json file")
    parser.add_argument("--provider-uri", type=str, default=None, help="Qlib data path")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Backtest start date")
    parser.add_argument("--end-date", type=str, default="2026-01-01", help="Backtest end date")
    parser.add_argument("--instrument", type=str, default="csi300", help="Qlib instrument universe")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark code, e.g. SH000300")
    parser.add_argument("--topk", type=int, default=50, help="Total target holdings")
    parser.add_argument("--n-drop", type=int, default=5, help="Number of names replaced each rebalance in topk_dropout mode")
    parser.add_argument(
        "--portfolio-mode",
        type=str,
        default="fixed_horizon",
        choices=["fixed_horizon"],
        help="fixed_horizon: buy one top-N group per day and sell it after a fixed holding period",
    )
    parser.add_argument(
        "--holding-period-days",
        type=int,
        default=10,
        help="Trading days each fixed-horizon group is held before attempted liquidation",
    )
    parser.add_argument(
        "--daily-buy-topk",
        type=int,
        default=5,
        help="Maximum stocks bought into each daily fixed-horizon group",
    )
    parser.add_argument(
        "--factor-eval-top-k",
        type=int,
        default=5,
        help="Top-k used when evaluating factor recent top-k returns",
    )
    parser.add_argument(
        "--signal-mode",
        type=str,
        default="rolling",
        choices=["rolling"],
        help="rolling: recompute factors and lazily build each window signal during backtest",
    )
    parser.add_argument(
        "--rebalance-window-days",
        type=int,
        default=5,
        help="Trading-day window for rolling factor reweighting or model retraining",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=20,
        help="Trading-day lookback window used for factor selection and model validation",
    )
    parser.add_argument(
        "--model-train-window-days",
        type=int,
        default=120,
        help="Training-window length in trading days for model-based signal mode",
    )
    parser.add_argument(
        "--model-label-horizon-days",
        type=int,
        default=10,
        help="Forward-return horizon in trading days for model-based signal mode labels",
    )
    parser.add_argument(
        "--model-label-expression",
        type=str,
        default="Ref($close, -11)/Ref($close, -1) - 1",
        help=(
            "Qlib label expression for model-based signal mode. "
            "If empty, the label falls back to --model-label-horizon-days."
        ),
    )
    parser.add_argument(
        "--return-expression",
        type=str,
        default="Ref($close, -11)/Ref($close, -1) - 1",
        help="Qlib forward-return expression used by factor IC/rank IC and selector evaluation",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of factors selected into the linear weighting combination or model feature set",
    )
    parser.add_argument(
        "--recent-perf-candidate-limit",
        type=int,
        default=None,
        help="Number of candidate factors evaluated by recent rolling performance",
    )
    parser.add_argument(
        "--recent-perf-batch-size",
        type=int,
        default=None,
        help="Batch size for recent rolling performance evaluation",
    )
    parser.add_argument(
        "--selector-mode",
        type=str,
        default="aff",
        choices=["recent", "aff", "mwu"],
        help="Factor selector used by rolling, dynamic, or model signal generation",
    )
    parser.add_argument(
        "--selector-score-threshold",
        type=float,
        default=0.02,
        help="Mean score threshold used by AFF selector",
    )
    parser.add_argument(
        "--selector-ir-threshold",
        type=float,
        default=0.2,
        help="Information ratio threshold used by AFF selector",
    )
    parser.add_argument(
        "--weighting-method",
        type=str,
        default=None,
        choices=["mwu"],
        help="Allocator used by rolling/dynamic signal generation",
    )
    parser.add_argument(
        "--mwu-learning-rate",
        type=float,
        default=0.15,
        help="Multiplicative Weights Update learning rate",
    )
    parser.add_argument(
        "--mwu-reward-cap",
        type=float,
        default=0.05,
        help="Absolute cap applied to per-day expert excess-return rewards before MWU update",
    )
    parser.add_argument(
        "--mwu-explore-rate",
        type=float,
        default=0.03,
        help="Uniform prior mixing rate applied after each MWU update",
    )
    parser.add_argument(
        "--mwu-max-weight",
        type=float,
        default=0.15,
        help="Hard cap for a single expert weight after MWU updates",
    )
    parser.add_argument(
        "--mwu-disable-dual-experts",
        action="store_true",
        help="Disable +/- factor expert splitting in MWU mode",
    )
    parser.add_argument(
        "--mwu-enable-tail-switch",
        action="store_true",
        help="Flip the final MWU window signal to the tail when both validation rank_ic and TopK-BottomK spread are negative",
    )
    parser.add_argument(
        "--mwu-tail-switch-mode",
        type=str,
        default="hard",
        choices=["hard"],
        help="hard: legacy full head/tail switch",
    )
    parser.add_argument(
        "--mwu-direction-rank-ic-threshold",
        type=float,
        default=0.0,
        help="Validation rank_ic threshold below which MWU tail switching becomes eligible",
    )
    parser.add_argument(
        "--mwu-direction-top-bottom-k",
        type=int,
        default=5,
        help="Top/Bottom bucket size used by the MWU validation spread check",
    )
    parser.add_argument(
        "--mwu-direction-validation-days",
        type=int,
        default=30,
        help="Most recent reward-window trading days held out for the MWU validation direction check",
    )
    parser.add_argument(
        "--mwu-direction-spread-threshold",
        type=float,
        default=0.0,
        help="Validation TopK-BottomK spread threshold below which MWU tail switching becomes eligible",
    )
    parser.add_argument(
        "--mwu-bayes-half-life",
        type=float,
        default=15.0,
        help="Half-life in validation observations for Bayesian MWU direction evidence",
    )
    parser.add_argument(
        "--mwu-bayes-prior-strength",
        type=float,
        default=10.0,
        help="Prior strength, in observation-equivalent units, shrinking Bayesian MWU direction edge toward zero",
    )
    parser.add_argument(
        "--mwu-bayes-hurdle",
        type=float,
        default=0.0,
        help="Minimum TopK-BottomK spread required before Bayesian direction evidence counts as head or tail edge",
    )
    parser.add_argument(
        "--enable-news-review",
        action="store_true",
        help="Review buy/sell candidates with stock news before generating orders",
    )
    parser.add_argument(
        "--news-data-path",
        type=str,
        default=None,
        help="Directory or JSON file containing stock news data",
    )
    parser.add_argument(
        "--news-batch-size",
        type=int,
        default=10,
        help="Number of candidate stocks reviewed per LLM batch",
    )
    parser.add_argument(
        "--news-candidate-pool-multiplier",
        type=int,
        default=3,
        help="Multiplier for the ranked buy-candidate pool reviewed by news before refilling buys",
    )
    parser.add_argument(
        "--news-llm-config-path",
        type=str,
        default=None,
        help="Optional path to config/env.yaml for the news review LLM",
    )
    parser.add_argument(
        "--news-confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum LLM confidence required before vetoing a buy/sell candidate",
    )
    return parser


def _build_backtest_config(args: argparse.Namespace) -> dict:
    """Convert parsed CLI args into service config."""
    config = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "instrument": args.instrument,
        "topk": args.topk,
        "n_drop": args.n_drop,
        "portfolio_mode": args.portfolio_mode,
        "holding_period_days": args.holding_period_days,
        "daily_buy_topk": args.daily_buy_topk,
        "factor_eval_top_k": args.factor_eval_top_k,
        "signal_mode": args.signal_mode,
        "rebalance_window_days": args.rebalance_window_days,
        "window_days": args.window_days,
        "model_train_window_days": args.model_train_window_days,
        "model_label_horizon_days": args.model_label_horizon_days,
        "model_label_expression": args.model_label_expression,
        "return_expression": args.return_expression,
        "selector_mode": args.selector_mode,
        "selector_score_threshold": args.selector_score_threshold,
        "selector_ir_threshold": args.selector_ir_threshold,
        "mwu_learning_rate": args.mwu_learning_rate,
        "mwu_reward_cap": args.mwu_reward_cap,
        "mwu_explore_rate": args.mwu_explore_rate,
        "mwu_max_weight": args.mwu_max_weight,
        "mwu_use_dual_experts": not args.mwu_disable_dual_experts,
        "mwu_enable_tail_switch": args.mwu_enable_tail_switch,
        "mwu_tail_switch_mode": args.mwu_tail_switch_mode,
        "mwu_direction_rank_ic_threshold": args.mwu_direction_rank_ic_threshold,
        "mwu_direction_top_bottom_k": args.mwu_direction_top_bottom_k,
        "mwu_direction_validation_days": args.mwu_direction_validation_days,
        "mwu_direction_spread_threshold": args.mwu_direction_spread_threshold,
        "mwu_bayes_half_life": args.mwu_bayes_half_life,
        "mwu_bayes_prior_strength": args.mwu_bayes_prior_strength,
        "mwu_bayes_hurdle": args.mwu_bayes_hurdle,
        "enable_news_review": args.enable_news_review,
        "news_batch_size": args.news_batch_size,
        "news_candidate_pool_multiplier": args.news_candidate_pool_multiplier,
        "news_confidence_threshold": args.news_confidence_threshold,
    }

    optional_values = {
        "provider_uri": args.provider_uri,
        "benchmark": args.benchmark,
        "top_n": args.top_n,
        "recent_perf_candidate_limit": args.recent_perf_candidate_limit,
        "recent_perf_batch_size": args.recent_perf_batch_size,
        "weighting_method": args.weighting_method,
        "news_data_path": args.news_data_path,
        "news_llm_config_path": args.news_llm_config_path,
    }
    for key, value in optional_values.items():
        if value is not None:
            config[key] = value
    return config


def main() -> int:
    """Run the standalone backtest wrapper."""
    args = build_parser().parse_args()
    service = QlibBacktestService(repo_root=Path(__file__).resolve().parent)
    result = service.run(
        weighting_path=args.weighting_path,
        backtest_config=_build_backtest_config(args),
    )

    print(f"status: {result.get('status')}")
    if result.get("message"):
        print(f"message: {result.get('message')}")
    if result.get("weighting_path"):
        print(f"weighting_path: {result.get('weighting_path')}")
    if result.get("run_dir"):
        print(f"run_dir: {result.get('run_dir')}")
    if result.get("signal_path"):
        print(f"signal_path: {result.get('signal_path')}")
    if result.get("report_path"):
        print(f"report_path: {result.get('report_path')}")
    if result.get("positions_path"):
        print(f"positions_path: {result.get('positions_path')}")
    if result.get("metrics_path"):
        print(f"metrics_path: {result.get('metrics_path')}")
    if result.get("net_value_curve_path"):
        print(f"net_value_curve_path: {result.get('net_value_curve_path')}")
    if result.get("rolling_windows_path"):
        print(f"rolling_windows_path: {result.get('rolling_windows_path')}")
    if result.get("selected_items_path"):
        print(f"selected_items_path: {result.get('selected_items_path')}")
    if result.get("news_review_path"):
        print(f"news_review_path: {result.get('news_review_path')}")
    if result.get("live_log_path"):
        print(f"live_log_path: {result.get('live_log_path')}")

    summary = result.get("summary") or {}
    if summary:
        print("summary:")
        _print_metric_tree(summary)

    test_signal_metrics = result.get("test_signal_metrics") or {}
    if test_signal_metrics:
        print("test_signal_metrics:")
        for key in ("ic", "rank_ic", "icir", "rank_icir"):
            if key in test_signal_metrics:
                print(f"  {key}: {float(test_signal_metrics[key]):.6f}")

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
