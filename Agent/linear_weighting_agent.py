"""
Linear Weighting Agent

Responsibilities:
1. Select the best-performing factors from the factor library in the recent window
2. Support different linear weighting schemes
3. Output composite expressions for subsequent stock selection or backtesting
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .services.factor_library_manager import FactorLibraryManager
from .services.factor_weighting_service import FactorWeightingService


class LinearWeightingAgent:
    """Refactored Linear Weighting Agent."""

    def __init__(self, llm_service=None) -> None:
        self.llm = llm_service
        self.library_manager = FactorLibraryManager()
        self.weighting_service = FactorWeightingService()

    def process(
        self,
        factor_library: Optional[List[Dict[str, Any]]] = None,
        weighting_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute 7-day rolling factor selection and linear weighting."""
        factor_library = factor_library or self.library_manager.load_factor_library()
        result = self.weighting_service.build_weighting_result(
            factor_library=factor_library,
            weighting_config=weighting_config,
        )
        self.library_manager.save_factor_library(factor_library)
        save_path = self.library_manager.save_weighting_result(result)
        result["save_path"] = save_path
        return result
