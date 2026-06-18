"""News backtest node."""
from __future__ import annotations

from typing import Any, Dict

from state import AgentState
from Agent.news_backtest_agent import NewsBacktestAgent


def news_backtest_node(state: AgentState) -> Dict[str, Any]:
    """Execute Qlib backtest, optionally integrating news review."""
    logs = ["[NewsBacktest] Starting Qlib backtest execution"]

    try:
        agent = NewsBacktestAgent()
        result = agent.process(
            weighting_result=state.get("weighting_result"),
            factor_library=state.get("factor_library"),
            backtest_config=state.get("backtest_config"),
        )
        logs.append(f"[NewsBacktest] Current status: {result.get('status')}")
        if result.get("metrics_path"):
            logs.append(f"[NewsBacktest] Metrics output: {result.get('metrics_path')}")
        return {
            "news_backtest_result": result,
            "logs": logs,
            "current_node": "news_backtest",
        }
    except Exception as exc:
        logs.append(f"[NewsBacktest] News backtest module exception: {exc}")
        return {
            "news_backtest_result": {"status": "error", "error": str(exc)},
            "logs": logs,
            "current_node": "news_backtest",
        }
