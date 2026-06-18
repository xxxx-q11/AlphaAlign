from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class BacktestEventLogger:
    """Append structured backtest events to a jsonl file."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if self.path is None:
            return

        record = {
            "logged_at": datetime.now().isoformat(),
            "event_type": event_type,
            "payload": payload or {},
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")
