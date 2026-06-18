from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from Agent.agent_factory import create_agent_from_config, load_env_config
from Strategy.runtime import BacktestEventLogger

logger = logging.getLogger(__name__)


class NewsReviewService:
    """Review buy/sell candidates with stock news and return veto lists."""

    def __init__(
        self,
        *,
        llm_service: Any | None = None,
        news_data_path: str | Path | None = None,
        news_batch_size: int = 10,
        confidence_threshold: float = 0.3,
        event_logger: BacktestEventLogger | None = None,
    ) -> None:
        self.llm = llm_service
        self.news_batch_size = max(1, int(news_batch_size))
        self.confidence_threshold = float(confidence_threshold)
        self.news_data_path = self._resolve_news_data_path(news_data_path)
        self._news_data_cache: dict[str, dict[str, Any]] = {}
        self.event_logger = event_logger

    @classmethod
    def from_env_config(
        cls,
        *,
        news_data_path: str | Path | None = None,
        news_batch_size: int = 10,
        llm_config_path: str | None = None,
        confidence_threshold: float = 0.3,
        event_logger: BacktestEventLogger | None = None,
    ) -> "NewsReviewService":
        llm_service = None
        try:
            config = load_env_config(llm_config_path)
            if config:
                llm_service = create_agent_from_config(dict(config))
                logger.info("[NewsReview] LLM service initialized for candidate review")
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            logger.warning("[NewsReview] Failed to initialize LLM service: %s", exc)

        return cls(
            llm_service=llm_service,
            news_data_path=news_data_path,
            news_batch_size=news_batch_size,
            confidence_threshold=confidence_threshold,
            event_logger=event_logger,
        )

    def is_enabled(self) -> bool:
        return self.llm is not None and self.news_data_path is not None and self.news_data_path.exists()

    def get_status(self) -> dict[str, Any]:
        disabled_reason = None
        if self.llm is None:
            disabled_reason = "llm_unavailable"
        elif self.news_data_path is None:
            disabled_reason = "news_data_path_unresolved"
        elif not self.news_data_path.exists():
            disabled_reason = "news_data_missing"

        return {
            "enabled": self.is_enabled(),
            "llm_available": self.llm is not None,
            "news_data_path": str(self.news_data_path) if self.news_data_path is not None else None,
            "news_data_exists": bool(self.news_data_path and self.news_data_path.exists()),
            "news_batch_size": self.news_batch_size,
            "confidence_threshold": self.confidence_threshold,
            "disabled_reason": disabled_reason,
        }

    def review(
        self,
        *,
        trade_date: pd.Timestamp | str,
        buy_candidates: list[dict[str, Any]],
        sell_candidates: list[dict[str, Any]],
        current_holdings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        del current_holdings  # reserved for future prompt enrichment

        formatted_date = self._normalize_date(trade_date)
        result = {
            "trade_date": formatted_date,
            "status": "disabled",
            "remove_from_buy": [],
            "remove_from_sell": [],
            "reviewed_buy_count": 0,
            "reviewed_sell_count": 0,
            "buy_candidates_with_news": [],
            "sell_candidates_with_news": [],
            "analysis_by_stock": {},
            "batch_summaries": [],
        }

        if not self.is_enabled():
            result["status"] = "disabled"
            result["disabled_reason"] = self.get_status().get("disabled_reason")
            self._log_event("news_review_disabled", result)
            return result

        buy_news_candidates = self._attach_news(buy_candidates, formatted_date)
        sell_news_candidates = self._attach_news(sell_candidates, formatted_date)
        result["buy_candidates_with_news"] = [item["stock_id"] for item in buy_news_candidates]
        result["sell_candidates_with_news"] = [item["stock_id"] for item in sell_news_candidates]
        self._log_event(
            "news_review_started",
            {
                "trade_date": formatted_date,
                "buy_candidates": buy_news_candidates,
                "sell_candidates": sell_news_candidates,
            },
        )

        if not buy_news_candidates and not sell_news_candidates:
            result["status"] = "no_relevant_news"
            self._log_event("news_review_no_relevant_news", result)
            return result

        try:
            buy_result = self._review_action_candidates("buy", buy_news_candidates, formatted_date)
            sell_result = self._review_action_candidates("sell", sell_news_candidates, formatted_date)
        except Exception as exc:  # pragma: no cover - downstream runtime/network dependent
            logger.warning("[NewsReview] Review failed for %s: %s", formatted_date, exc)
            result["status"] = "review_error"
            result["error"] = str(exc)
            self._log_event("news_review_error", result)
            return result

        result["status"] = "success"
        result["remove_from_buy"] = buy_result["removed"]
        result["remove_from_sell"] = sell_result["removed"]
        result["reviewed_buy_count"] = buy_result["reviewed_count"]
        result["reviewed_sell_count"] = sell_result["reviewed_count"]
        result["analysis_by_stock"].update(buy_result["analysis_by_stock"])
        result["analysis_by_stock"].update(sell_result["analysis_by_stock"])
        result["batch_summaries"].extend(buy_result["batch_summaries"])
        result["batch_summaries"].extend(sell_result["batch_summaries"])
        self._log_event("news_review_completed", result)
        return result

    def _attach_news(self, candidates: list[dict[str, Any]], trade_date: str) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            stock_id = str(candidate.get("stock_id", "")).strip()
            if not stock_id:
                continue
            stock_news = self._get_stock_news(stock_id, trade_date)
            if not stock_news:
                continue
            payload = dict(candidate)
            payload["stock_id"] = stock_id
            payload["news"] = stock_news
            enriched.append(payload)
        return enriched

    def _review_action_candidates(
        self,
        action: str,
        candidates: list[dict[str, Any]],
        trade_date: str,
    ) -> dict[str, Any]:
        removed: list[str] = []
        analysis_by_stock: dict[str, Any] = {}
        batch_summaries: list[dict[str, Any]] = []

        if not candidates:
            return {
                "removed": removed,
                "analysis_by_stock": analysis_by_stock,
                "batch_summaries": batch_summaries,
                "reviewed_count": 0,
            }

        for batch_start in range(0, len(candidates), self.news_batch_size):
            batch = candidates[batch_start : batch_start + self.news_batch_size]
            prompt = self._build_action_prompt(action=action, candidates=batch, trade_date=trade_date)
            self._log_event(
                "news_review_batch_request",
                {
                    "trade_date": trade_date,
                    "action": action,
                    "batch_index": batch_start // self.news_batch_size + 1,
                    "stock_ids": [item["stock_id"] for item in batch],
                    "batch_candidates": batch,
                    "prompt": prompt,
                },
            )
            response = self.llm.call(prompt=prompt, stream=False)
            parsed = self.llm.parse_json_response(response)
            stock_analysis = parsed.get("stocks_analysis", parsed if isinstance(parsed, dict) else {})
            if not isinstance(stock_analysis, dict):
                stock_analysis = {}
            self._log_event(
                "news_review_batch_response",
                {
                    "trade_date": trade_date,
                    "action": action,
                    "batch_index": batch_start // self.news_batch_size + 1,
                    "stock_ids": [item["stock_id"] for item in batch],
                    "raw_response": getattr(response, "content", response),
                    "parsed_response": parsed,
                },
            )

            removed.extend(self._extract_removed_codes(action=action, batch=batch, stock_analysis=stock_analysis))
            analysis_by_stock.update(stock_analysis)
            batch_summaries.append(
                {
                    "action": action,
                    "batch_index": batch_start // self.news_batch_size + 1,
                    "stock_ids": [item["stock_id"] for item in batch],
                    "summary": parsed.get("batch_summary", "") if isinstance(parsed, dict) else "",
                }
            )

        return {
            "removed": self._deduplicate_preserve_order(removed),
            "analysis_by_stock": analysis_by_stock,
            "batch_summaries": batch_summaries,
            "reviewed_count": len(candidates),
        }

    def _extract_removed_codes(
        self,
        *,
        action: str,
        batch: list[dict[str, Any]],
        stock_analysis: dict[str, Any],
    ) -> list[str]:
        removed: list[str] = []
        for candidate in batch:
            stock_id = candidate["stock_id"]
            analysis = stock_analysis.get(stock_id, {})
            if not isinstance(analysis, dict):
                continue
            recommendation = str(analysis.get("recommendation", "")).strip().lower()
            confidence = self._safe_float(analysis.get("confidence"), default=0.5)
            if confidence < self.confidence_threshold:
                continue

            if action == "buy" and recommendation in {"hold", "sell"}:
                removed.append(stock_id)
            if action == "sell" and recommendation in {"hold", "buy"}:
                removed.append(stock_id)
        return removed

    def _build_action_prompt(
        self,
        *,
        action: str,
        candidates: list[dict[str, Any]],
        trade_date: str,
    ) -> str:
        action_label = "Buy Candidate" if action == "buy" else "Sell Candidate"
        default_recommendation = "buy" if action == "buy" else "sell"
        veto_recommendation = "hold"

        stock_blocks: list[str] = []
        for candidate in candidates:
            news_lines = []
            for index, article in enumerate(candidate.get("news", [])[:5], start=1):
                title = str(article.get("news_title", "")).strip()
                source = str(article.get("news_source", "")).strip()
                publish_date = str(article.get("publish_date", "")).strip()
                content = str(article.get("content", "")).strip().replace("\n", " ")
                content = content[:240]
                news_lines.append(
                    f"    [{index}] Title: {title}\n"
                    f"        Source: {source}\n"
                    f"        Date: {publish_date}\n"
                    f"        Content: {content}"
                )

            score = self._safe_float(candidate.get("score"), default=0.0)
            current_weight = self._safe_float(candidate.get("current_weight"), default=0.0)
            stock_blocks.append(
                "\n".join(
                    [
                        f"Symbol: {candidate['stock_id']}",
                        f"Action: {action_label}",
                        f"Score: {score:.6f}",
                        f"Weight: {current_weight:.4%}",
                        "Related News:",
                        *news_lines,
                    ]
                )
            )

        if action == "buy":
            decision_rule = (
                "If the news shows clear negative signals, major risks, regulatory penalties, "
                "significantly below-expectation earnings, or escalating negative events, "
                f"then return recommendation=\"{veto_recommendation}\"; otherwise keep the original recommendation=\"{default_recommendation}\"."
            )
        else:
            decision_rule = (
                "If the news shows clear positive signals, better-than-expected improvements, major orders, "
                "significantly above-expectation earnings, or ongoing positive catalysts, "
                f"then return recommendation=\"{veto_recommendation}\"; otherwise keep the original recommendation=\"{default_recommendation}\"."
            )

        return f"""You are a news review agent in a quantitative trading backtest. Your task is NOT to re-select stocks, but to review trading candidates already selected by the model.

Trading Date: {trade_date}
Candidate Type: {action_label}

Rules:
1. Judge whether to veto the original action based solely on the given news.
2. Pay close attention to news publication dates -- the closer to the trading date, the more important.
3. If the evidence is weak, keep the original action and do not over-intervene.
4. {decision_rule}

Candidate Stocks and News:

{chr(10).join(stock_blocks)}

Please output JSON:
```json
{{
  "stocks_analysis": {{
    "<STOCK_CODE>": {{
      "news_sentiment": "positive/negative/neutral",
      "recommendation": "{default_recommendation}/{veto_recommendation}",
      "confidence": 0.85,
      "reason": "Brief explanation"
    }}
  }},
  "batch_summary": "Overall conclusion for this batch"
}}
```

Output only JSON. Do not add any explanation."""

    def _resolve_news_data_path(self, news_data_path: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if news_data_path is not None:
            candidates.append(Path(news_data_path).expanduser())

        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                repo_root / "data" / "news_data",
                repo_root / "data" / "news_data" / "csi_300",
                repo_root / "Qlib_MCP" / "workspace" / "news_data" / "csi_300",
                repo_root.parent / "trading_agent" / "Qlib_MCP" / "workspace" / "news_data" / "csi_300",
                Path("/workspace/news_data/csi_300"),
            ]
        )

        for candidate in candidates:
            normalized_candidate = self._normalize_news_data_path(candidate)
            if normalized_candidate.exists():
                return normalized_candidate

        return candidates[0] if candidates else None

    @staticmethod
    def _normalize_news_data_path(candidate: Path) -> Path:
        if not candidate.exists() or candidate.is_file():
            return candidate

        if (candidate / "Macro_News.json").exists() or any(candidate.glob("eastmoney_news_*")):
            return candidate

        csi300_dir = candidate / "csi_300"
        if csi300_dir.exists():
            return csi300_dir

        return candidate

    def _get_month_from_date(self, trade_date: str | pd.Timestamp) -> str:
        ts = pd.Timestamp(trade_date)
        return ts.strftime("%Y-%m")

    def _get_news_file_path(self, month: str) -> Path | None:
        if self.news_data_path is None:
            return None
        if self.news_data_path.is_file():
            return self.news_data_path

        year, month_num = month.split("-", maxsplit=1)
        exact_name = f"eastmoney_news_{year}_processed_{year}_{month_num}.json"
        exact_path = self.news_data_path / exact_name
        if exact_path.exists():
            return exact_path

        matched = sorted(self.news_data_path.glob(f"eastmoney_news_*_processed_{year}_{month_num}.json"))
        return matched[0] if matched else None

    def _load_news_data_by_month(self, month: str) -> dict[str, Any]:
        if month in self._news_data_cache:
            return self._news_data_cache[month]

        file_path = self._get_news_file_path(month)
        if file_path is None or not file_path.exists():
            self._news_data_cache[month] = {}
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - IO/runtime dependent
            logger.warning("[NewsReview] Failed to load %s: %s", file_path, exc)
            payload = {}

        if not isinstance(payload, dict):
            logger.warning("[NewsReview] Unexpected news payload type in %s", file_path)
            payload = {}

        self._news_data_cache[month] = payload
        return payload

    def _get_stock_news(self, stock_id: str, trade_date: str | pd.Timestamp) -> list[dict[str, Any]]:
        formatted_date = self._normalize_date(trade_date)
        month = self._get_month_from_date(formatted_date)
        news_data = self._load_news_data_by_month(month)
        if not news_data:
            return []

        date_news = news_data.get(formatted_date, {})
        if not date_news:
            for key, value in news_data.items():
                if str(key).startswith(formatted_date):
                    date_news = value
                    break
        if not isinstance(date_news, dict):
            return []

        stock_news = date_news.get(stock_id, [])
        if not isinstance(stock_news, list):
            return []
        return stock_news

    @staticmethod
    def _normalize_date(trade_date: str | pd.Timestamp) -> str:
        return pd.Timestamp(trade_date).strftime("%Y-%m-%d")

    @staticmethod
    def _deduplicate_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduplicated: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduplicated.append(value)
        return deduplicated

    @staticmethod
    def _safe_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, payload)
