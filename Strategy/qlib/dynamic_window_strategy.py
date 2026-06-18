from __future__ import annotations

from typing import Any

from Strategy.Allocator.normalized_score_allocator import NormalizedScoreAllocator
from Strategy.SignalGenerator.expression_signal_generator import ExpressionSignalGenerator
from Strategy.Selector.recent_performance_selector import RecentPerformanceSelector
from Strategy.base.base_allocator import BaseAllocator
from Strategy.base.base_selector import BaseSelector
from Strategy.base.base_signal_generator import BaseSignalGenerator
from Strategy.base.base_window_strategy import BaseWindowStrategy
from Strategy.qlib.window_signal import RollingWindowSignal
from Strategy.runtime.rebalance_scheduler import RebalanceScheduler


class DynamicWindowTopkStrategy(BaseWindowStrategy):
    """Qlib top-k strategy with modular rolling factor selection and weighting."""

    def __init__(
        self,
        *,
        factor_library: list[dict[str, Any]],
        start_date: str,
        end_date: str,
        instrument: str,
        benchmark: str,
        provider_uri: str,
        topk: int,
        n_drop: int,
        factor_eval_top_k: int | None = None,
        top_n: int = 10,
        window_days: int = 5,
        rebalance_window_days: int = 10,
        return_expression: str = "Ref($close, -11)/Ref($close, -1) - 1",
        selector: BaseSelector | None = None,
        allocator: BaseAllocator | None = None,
        signal_generator: BaseSignalGenerator | None = None,
        news_review_service: Any | None = None,
        event_logger: Any | None = None,
        **kwargs: Any,
    ) -> None:
        selector = selector or RecentPerformanceSelector()
        allocator = allocator or NormalizedScoreAllocator()
        signal_generator = signal_generator or ExpressionSignalGenerator()
        scheduler = RebalanceScheduler(
            start_date=start_date,
            end_date=end_date,
            rebalance_window_days=rebalance_window_days,
        )
        window_signal = RollingWindowSignal(
            factor_library=factor_library,
            selector=selector,
            allocator=allocator,
            signal_generator=signal_generator,
            scheduler=scheduler,
            instrument=instrument,
            benchmark=benchmark,
            provider_uri=provider_uri,
            top_n=top_n,
            top_k=int(factor_eval_top_k or topk),
            window_days=window_days,
            rebalance_window_days=rebalance_window_days,
            return_expression=return_expression,
        )
        super().__init__(
            selector=selector,
            allocator=allocator,
            signal_generator=signal_generator,
            window_signal=window_signal,
            topk=topk,
            n_drop=n_drop,
            news_review_service=news_review_service,
            event_logger=event_logger,
            **kwargs,
        )
