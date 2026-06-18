from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


def _clip(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_timestamp(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "-"
    text = text.replace("T", " ")
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _list_preview(values: Iterable[Any], limit: int = 8) -> str:
    items = [str(item) for item in values]
    if not items:
        return "-"
    preview = items[:limit]
    if len(items) > limit:
        preview.append(f"...(+{len(items) - limit})")
    return ", ".join(preview)


def _candidate_preview(candidates: list[dict[str, Any]], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for item in candidates[:limit]:
        stock_id = item.get("stock_id", "-")
        score = _fmt_float(item.get("score"))
        weight = _fmt_float(item.get("current_weight"))
        news = item.get("news") or []
        if news:
            first_news = news[0]
            title = _clip(first_news.get("news_title"), 50)
            publish_date = first_news.get("publish_date", "-")
            lines.append(
                f"  - {stock_id} score={score} weight={weight} news={len(news)} "
                f"[{publish_date}] {title}"
            )
        else:
            lines.append(f"  - {stock_id} score={score} weight={weight}")
    extra = len(candidates) - limit
    if extra > 0:
        lines.append(f"  - ... and {extra} more candidate(s)")
    return lines


def _analysis_preview(analysis_by_stock: dict[str, Any], limit: int = 8) -> list[str]:
    lines: list[str] = []
    items = list(analysis_by_stock.items())
    for stock_id, result in items[:limit]:
        if not isinstance(result, dict):
            lines.append(f"  - {stock_id}: {result}")
            continue
        recommendation = result.get("recommendation", "-")
        sentiment = result.get("news_sentiment", "-")
        confidence = _fmt_float(result.get("confidence"), digits=2)
        reason = _clip(result.get("reason"), 80)
        lines.append(
            f"  - {stock_id}: rec={recommendation} sentiment={sentiment} "
            f"conf={confidence} reason={reason}"
        )
    extra = len(items) - limit
    if extra > 0:
        lines.append(f"  - ... and {extra} more analysis result(s)")
    return lines


def _orders_preview(orders: list[dict[str, Any]], limit: int = 8) -> list[str]:
    lines: list[str] = []
    for order in orders[:limit]:
        lines.append(
            f"  - {order.get('stock_id', '-')} amount={_fmt_num(order.get('amount'))} "
            f"dir={order.get('direction', '-')}"
        )
    extra = len(orders) - limit
    if extra > 0:
        lines.append(f"  - ... and {extra} more order(s)")
    return lines


def _render_record(record: dict[str, Any], show_prompt: bool, show_raw: bool) -> str:
    timestamp = _fmt_timestamp(record.get("logged_at"))
    event_type = record.get("event_type", "unknown_event")
    payload = record.get("payload") or {}
    lines = [f"[{timestamp}] {event_type}"]

    if event_type == "backtest_run_started":
        lines.append(
            "  "
            f"range={payload.get('start_date', '-')} -> {payload.get('end_date', '-')} "
            f"signal={payload.get('signal_mode', '-')} selector={payload.get('selector_mode', '-')} "
            f"weighting={payload.get('weighting_method', '-')} "
            f"news_review={payload.get('enable_news_review', False)}"
        )
        lines.append(f"  run_dir={payload.get('run_dir', '-')}")
        lines.append(f"  weighting_path={payload.get('weighting_path', '-')}")
    elif event_type == "trade_decision_candidates":
        topk = payload.get("topk_recommendations") or []
        raw_buy = payload.get("raw_buy_candidates") or []
        raw_sell = payload.get("raw_sell_candidates") or []
        holdings = payload.get("current_holdings") or {}
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} step={payload.get('trade_step', '-')} "
            f"holdings={len(holdings)} topk={len(topk)} raw_buy={len(raw_buy)} raw_sell={len(raw_sell)}"
        )
        if topk:
            topk_preview = [
                f"{item.get('stock_id', '-')}({_fmt_float(item.get('score'))})"
                for item in topk[:8]
            ]
            if len(topk) > 8:
                topk_preview.append(f"...(+{len(topk) - 8})")
            lines.append(f"  topk: {', '.join(topk_preview)}")
    elif event_type == "news_review_started":
        buy_candidates = payload.get("buy_candidates") or []
        sell_candidates = payload.get("sell_candidates") or []
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} "
            f"buy_with_news={len(buy_candidates)} sell_with_news={len(sell_candidates)}"
        )
        if buy_candidates:
            lines.append("  buy candidates:")
            lines.extend(_candidate_preview(buy_candidates))
        if sell_candidates:
            lines.append("  sell candidates:")
            lines.extend(_candidate_preview(sell_candidates))
    elif event_type == "news_review_batch_request":
        batch_candidates = payload.get("batch_candidates") or []
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} action={payload.get('action', '-')} "
            f"batch={payload.get('batch_index', '-')} stock_ids={_list_preview(payload.get('stock_ids') or [])}"
        )
        lines.extend(_candidate_preview(batch_candidates, limit=8))
        if show_prompt and payload.get("prompt"):
            lines.append(f"  prompt={_clip(payload.get('prompt'), 600)}")
    elif event_type == "news_review_batch_response":
        parsed_response = payload.get("parsed_response")
        stock_analysis = {}
        if isinstance(parsed_response, dict):
            candidate_analysis = parsed_response.get("stocks_analysis", parsed_response)
            if isinstance(candidate_analysis, dict):
                stock_analysis = candidate_analysis
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} action={payload.get('action', '-')} "
            f"batch={payload.get('batch_index', '-')} stock_ids={_list_preview(payload.get('stock_ids') or [])}"
        )
        if stock_analysis:
            lines.extend(_analysis_preview(stock_analysis))
        if isinstance(parsed_response, dict) and parsed_response.get("batch_summary"):
            lines.append(f"  summary={_clip(parsed_response.get('batch_summary'), 120)}")
        if show_raw and payload.get("raw_response"):
            lines.append(f"  raw_response={_clip(payload.get('raw_response'), 600)}")
    elif event_type in {"news_review_completed", "news_review_no_relevant_news"}:
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} status={payload.get('status', '-')} "
            f"reviewed_buy={payload.get('reviewed_buy_count', 0)} "
            f"reviewed_sell={payload.get('reviewed_sell_count', 0)}"
        )
        lines.append(
            "  "
            f"remove_from_buy={_list_preview(payload.get('remove_from_buy') or [])} "
            f"remove_from_sell={_list_preview(payload.get('remove_from_sell') or [])}"
        )
        analysis = payload.get("analysis_by_stock") or {}
        if analysis:
            lines.extend(_analysis_preview(analysis))
        batch_summaries = payload.get("batch_summaries") or []
        for batch_summary in batch_summaries[:3]:
            lines.append(
                f"  batch_summary[{batch_summary.get('batch_index', '-')}]="
                f"{_clip(batch_summary.get('summary'), 120)}"
            )
        if len(batch_summaries) > 3:
            lines.append(f"  ... and {len(batch_summaries) - 3} more batch summar(y/ies)")
    elif event_type == "trade_decision_review_result":
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} status={payload.get('status', '-')} "
            f"raw_buy={len(payload.get('raw_buy') or [])} raw_sell={len(payload.get('raw_sell') or [])} "
            f"final_buy={len(payload.get('final_buy') or [])} final_sell={len(payload.get('final_sell') or [])}"
        )
        lines.append(
            "  "
            f"remove_from_buy={_list_preview(payload.get('remove_from_buy') or [])} "
            f"remove_from_sell={_list_preview(payload.get('remove_from_sell') or [])}"
        )
        analysis = payload.get("analysis_by_stock") or {}
        if analysis:
            lines.extend(_analysis_preview(analysis))
    elif event_type == "trade_decision_orders":
        buy_orders = payload.get("buy_orders") or []
        sell_orders = payload.get("sell_orders") or []
        total_buy = sum(float(item.get("amount", 0.0)) for item in buy_orders)
        total_sell = sum(float(item.get("amount", 0.0)) for item in sell_orders)
        lines.append(
            "  "
            f"trade_date={payload.get('trade_date', '-')} "
            f"buy_orders={len(buy_orders)} total_buy={_fmt_num(total_buy)} "
            f"sell_orders={len(sell_orders)} total_sell={_fmt_num(total_sell)}"
        )
        if buy_orders:
            lines.append("  buy orders:")
            lines.extend(_orders_preview(buy_orders))
        if sell_orders:
            lines.append("  sell orders:")
            lines.extend(_orders_preview(sell_orders))
    else:
        lines.append(f"  payload={_clip(json.dumps(payload, ensure_ascii=False, default=str), 400)}")

    return "\n".join(lines)


def _print_record(line: str, *, show_prompt: bool, show_raw: bool) -> None:
    line = line.strip()
    if not line:
        return
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        print(f"[invalid_json] {_clip(line, 300)}", flush=True)
        return
    print(_render_record(record, show_prompt=show_prompt, show_raw=show_raw), flush=True)


def _follow_file(path: Path, show_prompt: bool, show_raw: bool, sleep_seconds: float = 0.5) -> None:
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                _print_record(line, show_prompt=show_prompt, show_raw=show_raw)
                continue
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print backtest JSONL logs.")
    parser.add_argument("path", type=Path, help="Path to backtest_live_log.jsonl")
    parser.add_argument("--follow", action="store_true", help="Follow appended log lines")
    parser.add_argument("--tail", type=int, default=0, help="Print only the last N existing lines before follow")
    parser.add_argument("--show-prompt", action="store_true", help="Show clipped LLM prompts")
    parser.add_argument("--show-raw", action="store_true", help="Show clipped raw LLM responses")
    args = parser.parse_args()

    path = args.path.expanduser()
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if args.tail > 0:
        lines = lines[-args.tail :]

    for line in lines:
        _print_record(line, show_prompt=args.show_prompt, show_raw=args.show_raw)

    if args.follow:
        _follow_file(path, show_prompt=args.show_prompt, show_raw=args.show_raw)


if __name__ == "__main__":
    main()
