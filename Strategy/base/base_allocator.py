from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from Strategy.runtime.window_context import WindowContext


class BaseAllocator(ABC):
    """Allocate weights for selected factors."""

    @abstractmethod
    def allocate(
        self,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        """
        Return a dict containing at least `weighted_factors`.

        Each weighted factor should contain:
        - factor_id
        - qlib_expression
        - weight
        """
