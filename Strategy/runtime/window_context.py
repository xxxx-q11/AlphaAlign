from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class WindowContext:
    """Immutable context passed to selector/allocator/signal modules."""

    window_index: int
    selection_date: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    instrument: str
    benchmark: str
    provider_uri: Path
    top_n: int
    top_k: int
    window_days: int
    rebalance_window_days: int
    return_expression: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "window_index": self.window_index,
            "selection_date": self.selection_date.strftime("%Y-%m-%d"),
            "window_start": self.window_start.strftime("%Y-%m-%d"),
            "window_end": self.window_end.strftime("%Y-%m-%d"),
            "instrument": self.instrument,
            "benchmark": self.benchmark,
            "provider_uri": str(self.provider_uri),
            "top_n": self.top_n,
            "top_k": self.top_k,
            "window_days": self.window_days,
            "rebalance_window_days": self.rebalance_window_days,
            "return_expression": self.return_expression,
        }
        payload.update(self.extra)
        return payload
