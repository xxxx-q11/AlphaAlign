from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from Strategy.runtime.window_context import WindowContext


class BaseSelector(ABC):
    """Select a subset of factors for the current rebalance window."""

    @abstractmethod
    def select(
        self,
        factor_library: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        """
        Return a dict containing at least `selected_items`.

        Each selected item should contain:
        - factor
        - recent_series
        - recent_score
        - recent_score_source
        - is_proxy_score
        """
