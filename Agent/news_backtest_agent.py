"""Qlib backtest agent with optional news-aware trade review."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .services.qlib_backtest_service import QlibBacktestService


class NewsBacktestAgent:
    """Workflow wrapper around the reusable Qlib backtest service."""

    def __init__(self, llm_service=None) -> None:
        self.llm = llm_service
        self.backtest_service = QlibBacktestService()

    def process(
        self,
        weighting_result: Optional[Dict[str, Any]] = None,
        factor_library: Optional[List[Dict[str, Any]]] = None,
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the shared Qlib backtest implementation."""
        return self.backtest_service.run(
            weighting_result=weighting_result,
            factor_library=factor_library,
            backtest_config=backtest_config,
        )
