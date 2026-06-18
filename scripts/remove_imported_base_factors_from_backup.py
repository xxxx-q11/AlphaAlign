#!/usr/bin/env python3
"""Remove imported base-pool factors from factor library files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTOR_LIBRARY_PATH = ROOT / "data" / "factor_library.json"
DEFAULT_FACTOR_LIBRARY_METRICS_PATH = ROOT / "data" / "factor_library_metrics.json"
BASE_SOURCE = "base_pool"
BASE_REASON = "Initial import from base factor library"
BASE_ID_PREFIX = "base_"


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def dump_json(path: Path, data: list[dict]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def is_imported_base_factor(item: dict) -> bool:
    factor_id = str(item.get("factor_id") or "")
    source = item.get("source")
    economics_reason = item.get("economics_reason")
    return (
        source == BASE_SOURCE
        or economics_reason == BASE_REASON
        or factor_id.startswith(BASE_ID_PREFIX)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove imported base factors from factor_library.json and "
            "factor_library_metrics.json, keeping only GP/LLM-mined factors."
        )
    )
    parser.add_argument(
        "--factor-library",
        type=Path,
        default=DEFAULT_FACTOR_LIBRARY_PATH,
        help=f"Path to factor_library.json. Default: {DEFAULT_FACTOR_LIBRARY_PATH}",
    )
    parser.add_argument(
        "--factor-metrics",
        type=Path,
        default=DEFAULT_FACTOR_LIBRARY_METRICS_PATH,
        help=(
            "Path to factor_library_metrics.json. "
            f"Default: {DEFAULT_FACTOR_LIBRARY_METRICS_PATH}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the filtering result without writing files.",
    )
    args = parser.parse_args()

    factor_library = load_json(args.factor_library)
    factor_metrics = load_json(args.factor_metrics)

    removed_factor_ids = {
        item["factor_id"] for item in factor_library if is_imported_base_factor(item)
    }
    kept_factor_ids = {
        item["factor_id"]
        for item in factor_library
        if item.get("factor_id") not in removed_factor_ids
    }

    cleaned_factor_library = [
        item
        for item in factor_library
        if item.get("factor_id") in kept_factor_ids
    ]
    cleaned_factor_metrics = [
        item
        for item in factor_metrics
        if item.get("factor_id") in kept_factor_ids
    ]

    metrics_only_factor_ids = sorted(
        item["factor_id"]
        for item in factor_metrics
        if item.get("factor_id") not in {entry.get("factor_id") for entry in factor_library}
    )

    print(
        f"{args.factor_library.name}: {len(factor_library)} -> {len(cleaned_factor_library)} "
        f"(removed {len(factor_library) - len(cleaned_factor_library)})"
    )
    print(
        f"{args.factor_metrics.name}: {len(factor_metrics)} -> {len(cleaned_factor_metrics)} "
        f"(removed {len(factor_metrics) - len(cleaned_factor_metrics)})"
    )
    print(f"Kept mined factors: {len(kept_factor_ids)}")
    print(f"Removed imported base factors: {len(removed_factor_ids)}")

    if removed_factor_ids:
        print("Removed factor_id values:")
        for factor_id in sorted(removed_factor_ids):
            print(f"  - {factor_id}")

    if metrics_only_factor_ids:
        print("Metrics-only factor_id values skipped:")
        for factor_id in metrics_only_factor_ids:
            print(f"  - {factor_id}")

    if args.dry_run:
        return

    dump_json(args.factor_library, cleaned_factor_library)
    dump_json(args.factor_metrics, cleaned_factor_metrics)


if __name__ == "__main__":
    main()
