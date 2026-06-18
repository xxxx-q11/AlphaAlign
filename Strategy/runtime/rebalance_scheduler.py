from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from qlib.data import D


@dataclass(frozen=True, slots=True)
class ScheduledWindow:
    """A future holding window built from the trading calendar."""

    window_index: int
    selection_date: pd.Timestamp
    signal_start: pd.Timestamp
    signal_end: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


class RebalanceScheduler:
    """Split the backtest trading calendar into fixed-size rolling windows."""

    def __init__(self, *, start_date: str, end_date: str, rebalance_window_days: int) -> None:
        if rebalance_window_days <= 0:
            raise ValueError("rebalance_window_days must be positive")
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.rebalance_window_days = int(rebalance_window_days)
        self._windows: list[ScheduledWindow] | None = None

    def get_windows(self) -> list[ScheduledWindow]:
        if self._windows is not None:
            return self._windows

        calendar_start = (self.start_date - pd.Timedelta(days=max(self.rebalance_window_days * 20, 120))).strftime(
            "%Y-%m-%d"
        )
        calendar = D.calendar(
            start_time=calendar_start,
            end_time=self.end_date.strftime("%Y-%m-%d"),
        )
        full_trading_dates = [pd.Timestamp(value) for value in calendar if pd.Timestamp(value) <= self.end_date]
        trading_dates = [value for value in full_trading_dates if self.start_date <= value <= self.end_date]
        trading_date_positions = {value: index for index, value in enumerate(full_trading_dates)}
        windows: list[ScheduledWindow] = []
        for index, start_idx in enumerate(range(0, len(trading_dates), self.rebalance_window_days), start=1):
            window_dates = trading_dates[start_idx : start_idx + self.rebalance_window_days]
            if not window_dates:
                continue
            holding_start = window_dates[0]
            holding_end = window_dates[-1]
            holding_start_pos = trading_date_positions.get(holding_start)
            holding_end_pos = trading_date_positions.get(holding_end)
            if holding_start_pos is None or holding_end_pos is None:
                raise ValueError("Failed to locate holding window boundary in trading calendar.")
            if holding_start_pos == 0:
                raise ValueError(
                    f"No previous trading date found before holding_start={holding_start.strftime('%Y-%m-%d')}"
                )
            selection_date = full_trading_dates[holding_start_pos - 1]
            signal_end = full_trading_dates[holding_end_pos - 1]
            windows.append(
                ScheduledWindow(
                    window_index=index,
                    selection_date=selection_date,
                    signal_start=selection_date,
                    signal_end=signal_end,
                    start=holding_start,
                    end=holding_end,
                )
            )
        self._windows = windows
        return windows

    def get_window_for_date(self, query_date: pd.Timestamp) -> ScheduledWindow | None:
        query_ts = pd.Timestamp(query_date)
        windows = self.get_windows()
        if not windows:
            return None

        if query_ts < windows[0].signal_start:
            return None

        if query_ts <= windows[0].signal_end:
            return windows[0]

        for window in windows:
            if window.signal_start <= query_ts <= window.signal_end:
                return window

        return None
