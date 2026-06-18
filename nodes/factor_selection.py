"""Factor selection node."""
from __future__ import annotations

from typing import Any, Dict

from state import AgentState
from Agent.factor_selection_agent import FactorSelectionAgent


def factor_selection_node(state: AgentState) -> Dict[str, Any]:
    """Execute factor selection, deduplication, and factor library construction."""
    logs = ["[FactorSelection] Starting factor selection execution"]
    llm_service = get_llm_service()

    try:
        agent = FactorSelectionAgent(llm_service=llm_service)
        result = agent.process(
            candidates=state.get("gp_candidates", []),
            factor_library=state.get("factor_library"),
            mining_iteration=int(state.get("mining_iteration", 0)),
            selection_config={
                "train_ic_threshold": 0.02,
                "correlation_threshold": 0.90,
                "target_library_size": int(state.get("factor_library_size_target", 50)),
            },
            max_mining_rounds=int(state.get("max_mining_rounds", 10)),
        )
        logs.extend(result.get("logs", []))

        return {
            "factor_library": result.get("factor_library", []),
            "selected_candidates": result.get("selected_candidates", []),
            "rejected_candidates": result.get("rejected_candidates", []),
            "selection_summary": result.get("selection_summary", {}),
            "mining_feedback": result.get("mining_feedback", {}),
            "should_continue_mining": result.get("should_continue_mining", False),
            "logs": logs,
            "current_node": "factor_selection",
        }
    except Exception as exc:
        logs.append(f"[FactorSelection] Factor selection exception: {exc}")
        return {
            "logs": logs,
            "should_continue_mining": False,
            "current_node": "factor_selection",
        }


def get_llm_service():
    """Read configuration and create an LLM service instance."""
    from Agent.agent_factory import load_env_config, create_agent

    config = load_env_config()
    return create_agent(
        provider=config.get("provider", "qwen"),
        api_key=config.get("api_key"),
        model=config.get("model"),
        base_url=config.get("base_url"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens"),
        timeout=300,
    )
