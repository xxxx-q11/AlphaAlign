from __future__ import annotations

from abc import ABC
from typing import Any

from Strategy.base.base_allocator import BaseAllocator
from Strategy.base.base_selector import BaseSelector
from Strategy.base.base_signal_generator import BaseSignalGenerator
from Strategy.qlib.news_aware_topk_strategy import NewsAwareTopkStrategy


class BaseWindowStrategy(NewsAwareTopkStrategy, ABC):
    """Top-k strategy backed by pluggable selector/allocator/signal modules."""

    def __init__(
        self,
        *,
        selector: BaseSelector,
        allocator: BaseAllocator,
        signal_generator: BaseSignalGenerator,
        window_signal: Any,
        topk: int,
        n_drop: int,
        event_logger: Any = None,
        **kwargs: Any,
    ) -> None:
        self.selector = selector
        self.allocator = allocator
        self.signal_generator = signal_generator
        self.window_signal = window_signal
        super().__init__(
            signal=window_signal,
            topk=topk,
            n_drop=n_drop,
            event_logger=event_logger,
            **kwargs,
        )

    def get_window_history(self) -> list[dict[str, Any]]:
        if hasattr(self.window_signal, "get_window_history"):
            return self.window_signal.get_window_history()
        return []

    def get_signal_cache(self):
        if hasattr(self.window_signal, "get_signal_cache"):
            return self.window_signal.get_signal_cache()
        return None
