"""Linear weighting node."""
from __future__ import annotations

from typing import Any, Dict

from state import AgentState
from Agent.linear_weighting_agent import LinearWeightingAgent


def linear_weighting_node(state: AgentState) -> Dict[str, Any]:
    """Execute rolling factor selection and linear weighting."""
    logs = ["[LinearWeighting] Starting linear weighting execution"]
    backtest_config = state.get("backtest_config") or {}

    weighting_config = {
        "window_days": int(backtest_config.get("window_days", 7)),
        "top_n": int(backtest_config.get("top_n", 10)),
        "top_k": int(
            backtest_config.get(
                "factor_eval_top_k",
                backtest_config.get("daily_buy_topk", backtest_config.get("top_k", 5)),
            )
        ),
        "instrument": str(backtest_config.get("instrument", "csi300")),
        "benchmark": str(backtest_config.get("benchmark", "csi300")),
        "provider_uri": backtest_config.get("provider_uri"),
        "return_expression": str(backtest_config.get("return_expression", "Ref($close, -11)/Ref($close, -1) - 1")),
        "weighting_method": str(backtest_config.get("weighting_method") or "normalized"),
    }
    for key in ("mwu_learning_rate", "mwu_reward_cap", "mwu_explore_rate", "mwu_max_weight"):
        if backtest_config.get(key) is not None:
            weighting_config[key] = backtest_config[key]
    recent_perf_candidate_limit = backtest_config.get("recent_perf_candidate_limit")
    if recent_perf_candidate_limit is not None:
        weighting_config["recent_perf_candidate_limit"] = int(recent_perf_candidate_limit)
    recent_perf_batch_size = backtest_config.get("recent_perf_batch_size")
    if recent_perf_batch_size is not None:
        weighting_config["recent_perf_batch_size"] = int(recent_perf_batch_size)
    selection_date = backtest_config.get("selection_date")
    if selection_date:
        weighting_config["selection_date"] = str(selection_date)

    try:
        agent = LinearWeightingAgent()
        weighting_result = agent.process(
            factor_library=state.get("factor_library", []),
            weighting_config=weighting_config,
        )
        logs.append(
            f"[LinearWeighting] Currently selected factor count: {len(weighting_result.get('selected_factors', []))}"
        )
        logs.append(
            "[LinearWeighting] Config: "
            f"top_n={weighting_config['top_n']}, "
            f"window_days={weighting_config['window_days']}, "
            f"weighting_method={weighting_config['weighting_method']}"
        )

        return {
            "weighting_result": weighting_result,
            "logs": logs,
            "current_node": "linear_weighting",
        }
    except Exception as exc:
        logs.append(f"[LinearWeighting] Linear weighting exception: {exc}")
        return {
            "weighting_result": {"status": "error", "error": str(exc)},
            "logs": logs,
            "current_node": "linear_weighting",
        }
