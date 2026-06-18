"""
Agent module - Unified multi-LLM API calling interface

Uses safe imports to avoid making the entire module unavailable due to missing dependencies
"""
from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider

# Safe import factory functions
try:
    from Agent.agent_factory import create_agent, create_agent_from_config
except ImportError:
    create_agent = None
    create_agent_from_config = None

# Safe import providers (providers/__init__.py also does safe imports internally)
try:
    from Agent.providers import (
        QwenAgent,
        ClaudeAgent,
        GoogleAgent,
        OpenAIAgent,
        ThirdPartyAgent,
    )
except ImportError:
    QwenAgent = None
    ClaudeAgent = None
    GoogleAgent = None
    OpenAIAgent = None
    ThirdPartyAgent = None

try:
    from Agent.factor_selection_agent import FactorSelectionAgent
except ImportError:
    FactorSelectionAgent = None

try:
    from Agent.linear_weighting_agent import LinearWeightingAgent
except ImportError:
    LinearWeightingAgent = None

try:
    from Agent.news_backtest_agent import NewsBacktestAgent
except ImportError:
    NewsBacktestAgent = None

__all__ = [
    # Base classes
    "BaseAgent",
    "Message",
    "LLMResponse",
    "LLMProvider",
    # Factory functions
    "create_agent",
    "create_agent_from_config",
    # Concrete implementations
    "QwenAgent",
    "ClaudeAgent",
    "GoogleAgent",
    "OpenAIAgent",
    "ThirdPartyAgent",
    "FactorSelectionAgent",
    "LinearWeightingAgent",
    "NewsBacktestAgent",
]
