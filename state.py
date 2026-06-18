"""
State definition module

Unified state structure for AlphaAlign's LangGraph flow.
After refactoring, the state is centered on a structured factor library rather than the legacy model/strategy approach.
"""
from typing import TypedDict, List, Dict, Any

try:
    from typing import Annotated
except ImportError:  # pragma: no cover - Python 3.8 compatibility
    from typing_extensions import Annotated


def merge_lists(left: List[Any], right: List[Any]) -> List[Any]:
    """Merge log lists in LangGraph."""
    return left + right


class AgentState(TypedDict, total=False):
    """AlphaAlign workflow state."""

    # Input information
    task: str
    github_repo_url: str

    # Basic flow control
    current_node: str
    logs: Annotated[List[str], merge_lists]

    # Factor mining phase
    gp_candidates: List[Dict[str, Any]]
    mining_feedback: Dict[str, Any]
    mining_iteration: int
    max_mining_rounds: int

    # Factor screening and library phase
    factor_library: List[Dict[str, Any]]
    selected_candidates: List[Dict[str, Any]]
    rejected_candidates: List[Dict[str, Any]]
    selection_summary: Dict[str, Any]
    factor_library_size_target: int
    should_continue_mining: bool

    # Linear weighting phase
    weighting_result: Dict[str, Any]

    # News backtesting phase
    news_backtest_result: Dict[str, Any]
    backtest_config: Dict[str, Any]

    # Legacy state fields for backward compatibility, to avoid import errors in other modules
    factors: List[Dict[str, Any]]
    model: Dict[str, Any]
    strategy: Dict[str, Any]
    risk_report: Dict[str, Any]
    sota_pool_list: List[str]
    factor_pool_analysis_result_history: List[Dict[str, Any]]
    selection_result: Dict[str, Any]
    current_holdings: Dict[str, float]
    top_k_recommendations: List[Dict[str, Any]]
    trade_decisions: List[Dict[str, Any]]
    turnover_rate: float
