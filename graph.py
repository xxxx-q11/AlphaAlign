"""
AlphaAlign main graph definition

Core workflow after refactoring:
Factor Mining -> Factor Selection -> Linear Weighting -> News Backtest
"""
from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    factor_mining_node,
    factor_selection_node,
    linear_weighting_node,
    news_backtest_node,
)


def route_after_factor_selection(state: AgentState) -> str:
    """
    Routing logic after factor selection completes.

    When the factor pool has not yet reached the target size and the maximum
    mining rounds have not been exceeded, loop back to factor mining;
    otherwise proceed to the linear weighting stage.
    """
    if state.get("should_continue_mining", False):
        return "factor_mining"
    return "linear_weighting"


def build_graph() -> StateGraph:
    """Build the refactored AlphaAlign workflow graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("factor_mining", factor_mining_node)
    graph.add_node("factor_selection", factor_selection_node)
    graph.add_node("linear_weighting", linear_weighting_node)
    graph.add_node("news_backtest", news_backtest_node)

    # Set entry point
    graph.set_entry_point("factor_mining")

    # Define edges
    graph.add_edge("factor_mining", "factor_selection")
    graph.add_conditional_edges(
        "factor_selection",
        route_after_factor_selection,
        {
            "factor_mining": "factor_mining",
            "linear_weighting": "linear_weighting",
        },
    )
    graph.add_edge("linear_weighting", "news_backtest")
    graph.add_edge("news_backtest", END)

    return graph


def create_agent():
    """Create and compile the LangGraph Agent."""
    return build_graph().compile()
