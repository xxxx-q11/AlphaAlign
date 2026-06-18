from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from Strategy.runtime.window_context import WindowContext


class BaseSignalGenerator(ABC):
    """Generate a tradable signal for a future holding window."""

    @abstractmethod
    def generate(
        self,
        weighted_factors: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        """
        Return a dict containing at least:
        - signal: pandas.DataFrame | None
        - combined_expression: str | None
        """
