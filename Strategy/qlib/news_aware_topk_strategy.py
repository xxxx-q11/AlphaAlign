from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy import TopkDropoutStrategy

from Strategy.news.news_review_service import NewsReviewService
from Strategy.runtime import BacktestEventLogger

logger = logging.getLogger(__name__)


class NewsAwareTopkStrategy(TopkDropoutStrategy):
    """Apply news veto logic on top of the standard TopkDropout candidate lists."""

    def __init__(
        self,
        *,
        news_review_service: NewsReviewService | None = None,
        event_logger: BacktestEventLogger | None = None,
        **kwargs: Any,
    ) -> None:
        self.portfolio_mode = str(kwargs.pop("portfolio_mode", "topk_dropout"))
        self.holding_period_days = max(int(kwargs.pop("holding_period_days", 1)), 1)
        self.daily_buy_topk = max(int(kwargs.pop("daily_buy_topk", kwargs.get("n_drop", 1))), 1)
        self.news_candidate_pool_multiplier = max(
            int(kwargs.pop("news_candidate_pool_multiplier", 3)),
            1,
        )
        self.fixed_horizon_buckets: list[list[str]] = [[] for _ in range(self.holding_period_days)]
        self.news_review_service = news_review_service
        self.news_review_history: list[dict[str, Any]] = []
        self.event_logger = event_logger
        super().__init__(**kwargs)

    def get_news_review_history(self) -> list[dict[str, Any]]:
        return list(self.news_review_history)

    def generate_trade_decision(self, execute_result=None):
        news_review_enabled = self.news_review_service is not None and self.news_review_service.is_enabled()
        if self.portfolio_mode == "fixed_horizon":
            try:
                return self._generate_fixed_horizon_trade_decision(
                    execute_result,
                    enable_news_review=news_review_enabled,
                )
            except Exception as exc:  # pragma: no cover - runtime integration dependent
                logger.warning("[NewsAwareTopkStrategy] Fixed-horizon decision failed: %s", exc, exc_info=True)
                self._log_event(
                    "trade_decision_fallback",
                    {
                        "error": str(exc),
                        "portfolio_mode": self.portfolio_mode,
                    },
                )
                return TradeDecisionWO([], self)

        if not news_review_enabled and self.event_logger is None:
            return super().generate_trade_decision(execute_result)

        try:
            return self._generate_trade_decision_with_news_review(
                execute_result,
                enable_news_review=news_review_enabled,
            )
        except Exception as exc:  # pragma: no cover - runtime integration dependent
            logger.warning("[NewsAwareTopkStrategy] Fallback to raw TopkDropoutStrategy: %s", exc, exc_info=True)
            self._log_event(
                "trade_decision_fallback",
                {
                    "error": str(exc),
                },
            )
            return super().generate_trade_decision(execute_result)

    def _generate_fixed_horizon_trade_decision(self, execute_result=None, *, enable_news_review: bool):
        del execute_result

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        pred_score = pred_score.dropna().sort_values(ascending=False)
        current_temp: Position = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()
        current_stock_set = set(current_stock_list)
        bucket_index = trade_step % self.holding_period_days
        self._sync_fixed_horizon_buckets(current_stock_set, fallback_bucket_index=bucket_index)
        expiring_bucket = self.fixed_horizon_buckets[bucket_index]
        expiring_holdings = [code for code in expiring_bucket if code in current_stock_set]
        raw_sell = pd.Index(
            [
                code
                for code in expiring_holdings
                if self._is_sellable_now(code, current_temp, trade_start_time, trade_end_time)
            ]
        )
        retained_before_review = [code for code in expiring_holdings if code not in set(raw_sell)]
        raw_buy_slots = max(self.daily_buy_topk - len(retained_before_review), 0)
        buy_candidate_pool_size = self._buy_candidate_pool_size(
            raw_buy_slots,
            enable_news_review=enable_news_review,
            universe_count=len(pred_score),
        )
        buy_candidate_pool = self._select_fixed_horizon_buy_candidates(
            pred_score=pred_score,
            current_stock_list=current_stock_list,
            sell_candidates=raw_sell,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            buy_slots=buy_candidate_pool_size,
        )
        raw_buy = pd.Index(list(buy_candidate_pool)[:raw_buy_slots])

        current_holdings = self._position_to_holdings(current_temp)
        self._log_event(
            "trade_decision_candidates",
            {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "trade_step": trade_step,
                "portfolio_mode": self.portfolio_mode,
                "holding_period_days": self.holding_period_days,
                "daily_buy_topk": self.daily_buy_topk,
                "bucket_index": bucket_index,
                "current_holdings": current_holdings,
                "bucket_holdings": list(expiring_holdings),
                "retained_before_review": list(retained_before_review),
                "topk_recommendations": self._topk_recommendations(pred_score, top_n=self.daily_buy_topk),
                "raw_buy_candidates": self._build_review_candidates(raw_buy, pred_score, current_temp),
                "buy_candidate_pool": self._build_review_candidates(
                    buy_candidate_pool,
                    pred_score,
                    current_temp,
                ),
                "raw_sell_candidates": self._build_review_candidates(raw_sell, pred_score, current_temp),
                "news_review_enabled": enable_news_review,
            },
        )

        if enable_news_review and self.news_review_service is not None:
            review_payload = self.news_review_service.review(
                trade_date=pd.Timestamp(trade_start_time),
                buy_candidates=self._build_review_candidates(buy_candidate_pool, pred_score, current_temp),
                sell_candidates=self._build_review_candidates(raw_sell, pred_score, current_temp),
                current_holdings=current_holdings,
            )
        else:
            review_payload = {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "status": "disabled",
                "remove_from_buy": [],
                "remove_from_sell": [],
                "analysis_by_stock": {},
                "batch_summaries": [],
            }

        remove_from_buy = set(review_payload.get("remove_from_buy", []))
        remove_from_sell = set(review_payload.get("remove_from_sell", []))
        final_sell = pd.Index([code for code in raw_sell if code not in remove_from_sell])
        retained_after_review = [code for code in expiring_holdings if code not in set(final_sell)]
        buy_slots_after_review = max(self.daily_buy_topk - len(retained_after_review), 0)
        still_held_after_sell = current_stock_set - set(final_sell)
        final_buy = pd.Index(
            [
                code
                for code in buy_candidate_pool
                if code not in remove_from_buy and code not in still_held_after_sell
            ][:buy_slots_after_review]
        )

        self.news_review_history.append(
            {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "status": review_payload.get("status"),
                "portfolio_mode": self.portfolio_mode,
                "bucket_index": bucket_index,
                "raw_buy": list(raw_buy),
                "buy_candidate_pool": list(buy_candidate_pool),
                "raw_sell": list(raw_sell),
                "final_buy": list(final_buy),
                "final_sell": list(final_sell),
                "remove_from_buy": list(remove_from_buy),
                "remove_from_sell": list(remove_from_sell),
                "analysis_by_stock": review_payload.get("analysis_by_stock", {}),
                "batch_summaries": review_payload.get("batch_summaries", []),
            }
        )
        self._log_event("trade_decision_review_result", self.news_review_history[-1])

        return self._generate_fixed_horizon_orders(
            current_temp=current_temp,
            current_stock_list=current_stock_list,
            buy=final_buy,
            sell=final_sell,
            bucket_index=bucket_index,
            expiring_holdings=expiring_holdings,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )

    def _generate_trade_decision_with_news_review(self, execute_result=None, *, enable_news_review: bool):
        del execute_result

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        get_first_n, get_last_n, filter_stock = self._build_tradable_helpers(
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )

        current_temp: Position = copy.deepcopy(self.trade_position)
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        topk_recommendations = self._topk_recommendations(pred_score)
        raw_sell, raw_buy, buy_candidate_pool = self._resolve_raw_candidates(
            pred_score=pred_score,
            current_stock_list=current_stock_list,
            get_first_n=get_first_n,
            get_last_n=get_last_n,
            filter_stock=filter_stock,
            enable_news_review=enable_news_review,
        )
        current_holdings = self._position_to_holdings(current_temp)
        self._log_event(
            "trade_decision_candidates",
            {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "trade_step": trade_step,
                "current_holdings": current_holdings,
                "topk_recommendations": topk_recommendations,
                "raw_buy_candidates": self._build_review_candidates(raw_buy, pred_score, current_temp),
                "buy_candidate_pool": self._build_review_candidates(
                    buy_candidate_pool,
                    pred_score,
                    current_temp,
                ),
                "raw_sell_candidates": self._build_review_candidates(raw_sell, pred_score, current_temp),
                "news_review_enabled": enable_news_review,
            },
        )

        if enable_news_review and self.news_review_service is not None:
            review_payload = self.news_review_service.review(
                trade_date=pd.Timestamp(trade_start_time),
                buy_candidates=self._build_review_candidates(buy_candidate_pool, pred_score, current_temp),
                sell_candidates=self._build_review_candidates(raw_sell, pred_score, current_temp),
                current_holdings=current_holdings,
            )
        else:
            review_payload = {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "status": "disabled",
                "remove_from_buy": [],
                "remove_from_sell": [],
                "analysis_by_stock": {},
                "batch_summaries": [],
            }

        remove_from_buy = set(review_payload.get("remove_from_buy", []))
        remove_from_sell = set(review_payload.get("remove_from_sell", []))
        final_sell = pd.Index([code for code in raw_sell if code not in remove_from_sell])
        still_held_after_sell = set(current_stock_list) - set(final_sell)
        buy_slots_after_review = max(len(final_sell) + self.topk - len(current_stock_list), 0)
        final_buy = pd.Index(
            [
                code
                for code in buy_candidate_pool
                if code not in remove_from_buy and code not in still_held_after_sell
            ][:buy_slots_after_review]
        )

        self.news_review_history.append(
            {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "status": review_payload.get("status"),
                "raw_buy": list(raw_buy),
                "buy_candidate_pool": list(buy_candidate_pool),
                "raw_sell": list(raw_sell),
                "final_buy": list(final_buy),
                "final_sell": list(final_sell),
                "remove_from_buy": list(remove_from_buy),
                "remove_from_sell": list(remove_from_sell),
                "analysis_by_stock": review_payload.get("analysis_by_stock", {}),
                "batch_summaries": review_payload.get("batch_summaries", []),
            }
        )
        self._log_event(
            "trade_decision_review_result",
            self.news_review_history[-1],
        )

        return self._generate_orders(
            current_temp=current_temp,
            cash=cash,
            current_stock_list=current_stock_list,
            buy=final_buy,
            sell=final_sell,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )

    def _resolve_raw_candidates(
        self,
        *,
        pred_score: pd.Series,
        current_stock_list: list[str],
        get_first_n: Callable[[Any, int], list[str] | pd.Index],
        get_last_n: Callable[[Any, int], list[str] | pd.Index],
        filter_stock: Callable[[Any], list[str] | pd.Index],
        enable_news_review: bool,
    ) -> tuple[pd.Index, pd.Index, pd.Index]:
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index

        if self.method_buy == "top":
            raw_candidate_count = max(self.n_drop + self.topk - len(last), 0)
            candidate_count = self._buy_candidate_pool_size(
                raw_candidate_count,
                enable_news_review=enable_news_review,
                universe_count=len(pred_score),
            )
            if candidate_count <= 0:
                today = []
            else:
                today = get_first_n(
                    pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                    candidate_count,
                )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = [stock_id for stock_id in topk_candi if stock_id not in last]
            raw_candidate_count = max(self.n_drop + self.topk - len(last), 0)
            candidate_count = self._buy_candidate_pool_size(
                raw_candidate_count,
                enable_news_review=enable_news_review,
                universe_count=len(candi),
            )
            if candidate_count <= 0:
                today = []
            else:
                try:
                    today = np.random.choice(candi, candidate_count, replace=False)
                except ValueError:
                    today = candi
        else:
            raise NotImplementedError(f"Unsupported method_buy={self.method_buy}")

        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = pd.Index(candi)
        else:
            raise NotImplementedError(f"Unsupported method_sell={self.method_sell}")

        raw_buy_count = max(len(sell) + self.topk - len(last), 0)
        buy_pool_count = self._buy_candidate_pool_size(
            raw_buy_count,
            enable_news_review=enable_news_review,
            universe_count=len(today),
        )
        buy_candidate_pool = pd.Index(today[:buy_pool_count])
        buy = pd.Index(list(buy_candidate_pool)[:raw_buy_count])
        return pd.Index(sell), buy, buy_candidate_pool

    def _buy_candidate_pool_size(
        self,
        target_count: int,
        *,
        enable_news_review: bool,
        universe_count: int | None = None,
    ) -> int:
        target = max(int(target_count), 0)
        if target <= 0:
            return 0
        if not enable_news_review:
            return target

        pool_size = max(target, target * self.news_candidate_pool_multiplier)
        if universe_count is not None:
            pool_size = min(pool_size, max(int(universe_count), 0))
        return max(pool_size, target)

    def _generate_orders(
        self,
        *,
        current_temp: Position,
        cash: float,
        current_stock_list: list[str],
        buy: pd.Index,
        sell: pd.Index,
        trade_start_time,
        trade_end_time,
    ) -> TradeDecisionWO:
        sell_order_list = []
        buy_order_list = []

        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code not in sell:
                continue

            time_per_step = self.trade_calendar.get_freq()
            if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                continue

            sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    sell_order,
                    position=current_temp,
                )
                del trade_price
                cash += trade_val - trade_cost

        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue

            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)

        self._log_event(
            "trade_decision_orders",
            {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "final_sell_candidates": list(sell),
                "final_buy_candidates": list(buy),
                "sell_orders": [self._order_to_dict(order) for order in sell_order_list],
                "buy_orders": [self._order_to_dict(order) for order in buy_order_list],
            },
        )
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

    def _generate_fixed_horizon_orders(
        self,
        *,
        current_temp: Position,
        current_stock_list: list[str],
        buy: pd.Index,
        sell: pd.Index,
        bucket_index: int,
        expiring_holdings: list[str],
        trade_start_time,
        trade_end_time,
    ) -> TradeDecisionWO:
        sell_order_list = []
        buy_order_list = []
        sold_codes: list[str] = []
        cash = current_temp.get_cash()

        for code in current_stock_list:
            if code not in sell:
                continue
            if not self._is_sellable_now(code, current_temp, trade_start_time, trade_end_time):
                continue

            sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                sold_codes.append(code)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    sell_order,
                    position=current_temp,
                )
                del trade_price
                cash += trade_val - trade_cost

        retained_codes = [code for code in expiring_holdings if code not in set(sold_codes)]
        buy_slots = max(self.daily_buy_topk - len(retained_codes), 0)
        held_after_sells = set(current_stock_list) - set(sold_codes)
        sell_set = set(sell)
        buy = pd.Index(
            [code for code in buy if code not in held_after_sells and code not in sell_set][:buy_slots]
        )
        total_value = current_temp.calculate_value() - current_temp.get_cash() + cash
        target_stock_value = (
            total_value * self.risk_degree / (self.holding_period_days * self.daily_buy_topk)
        )
        value = min(target_stock_value, cash * self.risk_degree / len(buy)) if len(buy) > 0 else 0
        bought_codes: list[str] = []

        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue

            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            if buy_price is None or not np.isfinite(buy_price) or buy_price <= 0:
                continue

            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            if buy_amount > 0:
                buy_order_list.append(buy_order)
                bought_codes.append(code)

        self.fixed_horizon_buckets[bucket_index] = retained_codes + bought_codes
        self._log_event(
            "trade_decision_orders",
            {
                "trade_date": pd.Timestamp(trade_start_time).strftime("%Y-%m-%d"),
                "portfolio_mode": self.portfolio_mode,
                "bucket_index": bucket_index,
                "retained_codes": retained_codes,
                "sold_codes": sold_codes,
                "bought_codes": bought_codes,
                "final_sell_candidates": list(sell),
                "final_buy_candidates": list(buy),
                "sell_orders": [self._order_to_dict(order) for order in sell_order_list],
                "buy_orders": [self._order_to_dict(order) for order in buy_order_list],
            },
        )
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

    def _sync_fixed_horizon_buckets(self, current_stock_set: set[str], *, fallback_bucket_index: int = 0) -> None:
        tracked: set[str] = set()
        synced_buckets: list[list[str]] = []
        for bucket in self.fixed_horizon_buckets:
            synced_bucket = []
            for code in bucket:
                if code in current_stock_set and code not in tracked:
                    synced_bucket.append(code)
                    tracked.add(code)
            synced_buckets.append(synced_bucket)

        untracked_holdings = [code for code in current_stock_set if code not in tracked]
        if untracked_holdings and synced_buckets:
            synced_buckets[fallback_bucket_index % len(synced_buckets)].extend(untracked_holdings)
        self.fixed_horizon_buckets = synced_buckets

    def _is_sellable_now(self, code: str, position: Position, trade_start_time, trade_end_time) -> bool:
        if not self.trade_exchange.is_stock_tradable(
            stock_id=code,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
        ):
            return False

        time_per_step = self.trade_calendar.get_freq()
        return position.get_stock_count(code, bar=time_per_step) >= self.hold_thresh

    def _select_fixed_horizon_buy_candidates(
        self,
        *,
        pred_score: pd.Series,
        current_stock_list: list[str],
        sell_candidates: pd.Index,
        trade_start_time,
        trade_end_time,
        buy_slots: int,
    ) -> pd.Index:
        if buy_slots <= 0:
            return pd.Index([])

        excluded = set(current_stock_list) | set(sell_candidates)
        selected: list[str] = []
        for code in pred_score.index:
            if code in excluded:
                continue
            if self.only_tradable and not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            selected.append(code)
            if len(selected) >= buy_slots:
                break
        return pd.Index(selected)

    def _build_tradable_helpers(self, *, trade_start_time, trade_end_time):
        if self.only_tradable:

            def get_first_n(li, n, reverse: bool = False):
                current_count = 0
                result = []
                iterable = reversed(li) if reverse else li
                for stock_id in iterable:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=stock_id,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    ):
                        result.append(stock_id)
                        current_count += 1
                        if current_count >= n:
                            break
                return result[::-1] if reverse else result

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    stock_id
                    for stock_id in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=stock_id,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    )
                ]

            return get_first_n, get_last_n, filter_stock

        def get_first_n(li, n):
            return list(li)[:n]

        def get_last_n(li, n):
            return list(li)[-n:]

        def filter_stock(li):
            return li

        return get_first_n, get_last_n, filter_stock

    @staticmethod
    def _position_to_holdings(position: Position) -> dict[str, float]:
        holdings: dict[str, float] = {}
        total_value = position.calculate_value()
        if total_value <= 0:
            return holdings

        for stock_id in position.get_stock_list():
            stock_value = position.get_stock_amount(stock_id) * position.get_stock_price(stock_id)
            holdings[stock_id] = stock_value / total_value
        return holdings

    @staticmethod
    def _build_review_candidates(
        stock_ids: pd.Index,
        pred_score: pd.Series,
        position: Position,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        current_holdings = NewsAwareTopkStrategy._position_to_holdings(position)
        for stock_id in stock_ids:
            score = pred_score.get(stock_id, np.nan)
            candidates.append(
                {
                    "stock_id": stock_id,
                    "score": None if pd.isna(score) else float(score),
                    "current_weight": float(current_holdings.get(stock_id, 0.0)),
                }
            )
        return candidates

    def _topk_recommendations(self, pred_score: pd.Series, top_n: int | None = None) -> list[dict[str, Any]]:
        recommendations = pred_score.sort_values(ascending=False).head(int(top_n or self.topk))
        payload: list[dict[str, Any]] = []
        for stock_id, score in recommendations.items():
            payload.append(
                {
                    "stock_id": stock_id,
                    "score": None if pd.isna(score) else float(score),
                }
            )
        return payload

    @staticmethod
    def _order_to_dict(order: Order) -> dict[str, Any]:
        return {
            "stock_id": order.stock_id,
            "amount": float(order.amount),
            "direction": int(order.direction),
            "start_time": order.start_time,
            "end_time": order.end_time,
        }

    def _log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, payload)
