from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from qlib.backtest.signal import Signal
from qlib.data.dataset.utils import convert_index_format
from qlib.utils.resam import resam_ts_data

from Strategy.base.base_allocator import BaseAllocator
from Strategy.base.base_selector import BaseSelector
from Strategy.base.base_signal_generator import BaseSignalGenerator
from Strategy.runtime.rebalance_scheduler import RebalanceScheduler, ScheduledWindow
from Strategy.runtime.window_context import WindowContext


class RollingWindowSignal(Signal):
    """Lazily build and cache one signal block per rebalance window."""

    def __init__(
        self,
        *,
        factor_library: list[dict[str, Any]],
        selector: BaseSelector,
        allocator: BaseAllocator,
        signal_generator: BaseSignalGenerator,
        scheduler: RebalanceScheduler,
        instrument: str,
        benchmark: str,
        provider_uri: str,
        top_n: int,
        top_k: int,
        window_days: int,
        rebalance_window_days: int,
        return_expression: str,
    ) -> None:
        self.factor_library = deepcopy(factor_library)
        self.selector = selector
        self.allocator = allocator
        self.signal_generator = signal_generator
        self.scheduler = scheduler
        self.instrument = instrument
        self.benchmark = benchmark
        self.provider_uri = provider_uri
        self.top_n = int(top_n)
        self.top_k = int(top_k)
        self.window_days = int(window_days)
        self.rebalance_window_days = int(rebalance_window_days)
        self.return_expression = return_expression
        self.signal_cache: pd.DataFrame | None = None
        self.window_signal_cache: dict[int, pd.DataFrame] = {}
        self.window_cache: dict[int, dict[str, Any]] = {}
        self.window_history: list[dict[str, Any]] = []

    def get_signal(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.Series | pd.DataFrame | None:
        window = self.scheduler.get_window_for_date(pd.Timestamp(start_time))
        if window is None:
            return None
        if window.window_index not in self.window_cache:
            self._build_window(window)
        signal_block = self.window_signal_cache.get(window.window_index)
        if signal_block is None:
            return None
        return resam_ts_data(signal_block, start_time=start_time, end_time=end_time, method="last")

    def get_window_history(self) -> list[dict[str, Any]]:
        return list(self.window_history)

    def get_signal_cache(self) -> pd.DataFrame | None:
        if self.signal_cache is None:
            return None
        return self._canonicalize_signal(self.signal_cache.copy())

    def _build_window(self, window: ScheduledWindow) -> None:
        context = WindowContext(
            window_index=window.window_index,
            selection_date=window.selection_date,
            window_start=window.signal_start,
            window_end=window.signal_end,
            instrument=self.instrument,
            benchmark=self.benchmark,
            provider_uri=Path(self.provider_uri),
            top_n=self.top_n,
            top_k=self.top_k,
            window_days=self.window_days,
            rebalance_window_days=self.rebalance_window_days,
            return_expression=self.return_expression,
        )

        selection_result = self.selector.select(self.factor_library, context)
        selected_items = selection_result.get("selected_items", []) or []
        allocation_result = self.allocator.allocate(selected_items, context)
        weighted_factors = allocation_result.get("weighted_factors", []) or []
        generation_result = self.signal_generator.generate(weighted_factors, context)
        signal = generation_result.get("signal")

        if signal is not None and not signal.empty:
            signal = signal[~signal.index.duplicated(keep="last")]
            self.window_signal_cache[window.window_index] = convert_index_format(signal.copy(), level="datetime")
            self._append_signal_cache(signal)

        window_payload = {
            "window_index": window.window_index,
            "holding_start": window.start.strftime("%Y-%m-%d"),
            "holding_end": window.end.strftime("%Y-%m-%d"),
            "selection_date": context.selection_date.strftime("%Y-%m-%d"),
            "signal_start": context.window_start.strftime("%Y-%m-%d"),
            "signal_end": context.window_end.strftime("%Y-%m-%d"),
            "selected_items": selected_items,
            "selected_factor_ids": [item.get("factor_id") for item in weighted_factors],
            "selected_factors": weighted_factors,
            "combined_expression": generation_result.get("combined_expression"),
            "selection_context": selection_result.get("selection_context"),
            "allocation_context": allocation_result.get("allocation_context"),
            "generation_context": generation_result.get("generation_context"),
        }
        self.window_cache[window.window_index] = window_payload
        self.window_history.append(window_payload)

    def _append_signal_cache(self, signal: pd.DataFrame) -> None:
        signal = self._canonicalize_signal(signal.copy())
        existing = None if self.signal_cache is None else self._canonicalize_signal(self.signal_cache.copy())
        combined = signal if existing is None else pd.concat([existing, signal]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        self.signal_cache = self._canonicalize_signal(combined)

    @classmethod
    def _canonicalize_signal(cls, signal: pd.DataFrame) -> pd.DataFrame:
        signal = cls._ensure_named_index(signal)
        if isinstance(signal.index, pd.MultiIndex) and signal.index.nlevels == 2:
            index_names = list(signal.index.names)
            if "datetime" in index_names and "instrument" in index_names:
                signal = convert_index_format(signal, level="datetime")
                signal = signal.reorder_levels(["datetime", "instrument"]).sort_index()
        return signal

    @staticmethod
    def _ensure_named_index(signal: pd.DataFrame) -> pd.DataFrame:
        if isinstance(signal.index, pd.MultiIndex) and signal.index.nlevels == 2:
            level0 = signal.index.get_level_values(0)
            level1 = signal.index.get_level_values(1)
            if RollingWindowSignal._looks_like_datetime_level(level0):
                signal.index = signal.index.set_names(["datetime", "instrument"])
            elif RollingWindowSignal._looks_like_datetime_level(level1):
                signal.index = signal.index.set_names(["instrument", "datetime"])
        return signal

    @staticmethod
    def _looks_like_datetime_level(level_values: pd.Index, sample_size: int = 50) -> bool:
        if pd.api.types.is_datetime64_any_dtype(level_values):
            return True

        sample = pd.Index(level_values).dropna().unique()[:sample_size]
        if len(sample) == 0:
            return False

        parsed = pd.to_datetime(sample, errors="coerce")
        parsed_ratio = float(parsed.notna().sum()) / float(len(sample))
        return parsed_ratio >= 0.8
