"""Factor mining node."""
from __future__ import annotations

from typing import Any, Dict, List

from state import AgentState
from Agent.FactorMiningAgent import FactorMiningAgent
from Agent.services.factor_expression_converter import FactorExpressionConverterService
from Agent.services.factor_library_manager import FactorLibraryManager


def factor_mining_node(state: AgentState) -> Dict[str, Any]:
    """Execute a single round of GP factor mining and output structured candidate factors."""
    logs = ["[FactorMining] Starting factor mining execution"]
    mining_iteration = int(state.get("mining_iteration", 0)) + 1
    llm_service = get_llm_service()
    converter_service = FactorExpressionConverterService()
    library_manager = FactorLibraryManager()

    try:
        agent = FactorMiningAgent(llm_service)
        previous_selection_result = state.get("selection_result")
        mining_feedback = state.get("mining_feedback")

        raw_candidates, selection_result = agent.process(
            previous_selection_result=previous_selection_result,
            mining_feedback=mining_feedback,
        )

        normalized_candidates = converter_service.normalize_candidates(
            raw_factors=raw_candidates,
            round_index=mining_iteration,
            source="gp",
        )
        library_manager.save_candidate_round(mining_iteration, normalized_candidates)

        logs.append(f"[FactorMining] Candidate factors mined this round: {len(raw_candidates)}")
        logs.append(f"[FactorMining] Normalized candidate factors: {len(normalized_candidates)}")

        return {
            "gp_candidates": normalized_candidates,
            "selection_result": selection_result,
            "mining_iteration": mining_iteration,
            "logs": logs,
            "current_node": "factor_mining",
        }

    except Exception as exc:
        logs.append(f"[FactorMining] Factor mining error: {exc}")
        return {
            "gp_candidates": [],
            "mining_iteration": mining_iteration,
            "logs": logs,
            "current_node": "factor_mining",
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
    )
