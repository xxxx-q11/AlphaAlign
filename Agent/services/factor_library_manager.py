"""
Factor Library Management Service

Responsible for:
1. Factor library main file read/write
2. Metrics file read/write
3. Persisting candidate factors, mining feedback, and linear weighting results
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class FactorLibraryManager:
    """Structured factor library manager."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.base_pool_path = self.data_dir / "base_pool.json"
        self.factor_library_path = self.data_dir / "factor_library.json"
        self.factor_metrics_path = self.data_dir / "factor_library_metrics.json"
        self.candidate_dir = self.data_dir / "candidate_factors"
        self.feedback_dir = self.data_dir / "mining_feedback"
        self.weighting_dir = self.data_dir / "weighting_history"

        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.weighting_dir.mkdir(parents=True, exist_ok=True)

    def load_factor_library(self) -> List[Dict[str, Any]]:
        """Load structured factor library; initialize from base_pool if not found."""
        if self.factor_library_path.exists():
            content = self._read_json_file(self.factor_library_path, default=[])
            if isinstance(content, list):
                # Backward compatibility: if the file still contains expression string lists, auto-upgrade to structured factor library.
                if content and all(isinstance(item, str) for item in content):
                    upgraded_library = []
                    for index, expression in enumerate(content, start=1):
                        upgraded_library.append(
                            {
                                "factor_id": f"legacy_{index:03d}",
                                "source": "legacy_factor_library",
                                "round_index": 0,
                                "gp_expression": None,
                                "qlib_expression": expression,
                                "is_valid": True,
                                "invalid_reason": None,
                                "needs_cs_rank": False,
                                "economics_passed": True,
                                "economics_reason": "Legacy factor library auto-upgraded",
                                "train_ic": 0.0,
                                "train_rank_ic": 0.0,
                                "valid_ic": 0.0,
                                "valid_rank_ic": 0.0,
                                "test_ic": 0.0,
                                "test_rank_ic": 0.0,
                                "metrics": {
                                    "train": {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0},
                                    "valid": {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0},
                                    "test": {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0},
                                },
                                "metadata": {},
                            }
                        )
                    self.save_factor_library(upgraded_library)
                    return upgraded_library

                return content

        base_pool = self.load_base_pool()
        factor_library = []
        for index, expression in enumerate(base_pool, start=1):
            factor_library.append(
                {
                    "factor_id": f"base_{index:03d}",
                    "source": "base_pool",
                    "round_index": 0,
                    "gp_expression": None,
                    "qlib_expression": expression,
                    "is_valid": True,
                    "invalid_reason": None,
                    "needs_cs_rank": False,
                    "economics_passed": True,
                    "economics_reason": "Base factor library initialization import",
                    "train_ic": 0.0,
                    "train_rank_ic": 0.0,
                    "valid_ic": 0.0,
                    "valid_rank_ic": 0.0,
                    "test_ic": 0.0,
                    "test_rank_ic": 0.0,
                    "metrics": {
                        "train": {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0},
                        "valid": {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0},
                        "test": {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0},
                    },
                    "metadata": {},
                }
            )

        if factor_library:
            self.save_factor_library(factor_library)
        return factor_library

    def save_factor_library(self, factor_library: List[Dict[str, Any]]) -> str:
        """Save structured factor library."""
        return self._write_json_file(self.factor_library_path, factor_library)

    def save_factor_metrics(self, factor_library: List[Dict[str, Any]]) -> str:
        """Extract and save factor metrics file."""
        metrics_payload = []
        for factor in factor_library:
            metrics_payload.append(
                {
                    "factor_id": factor.get("factor_id"),
                    "qlib_expression": factor.get("qlib_expression"),
                    "metrics": factor.get("metrics", {}),
                    "economics_passed": factor.get("economics_passed"),
                    "round_index": factor.get("round_index"),
                }
            )
        return self._write_json_file(self.factor_metrics_path, metrics_payload)

    def load_base_pool(self) -> List[str]:
        """Load base factor pool."""
        content = self._read_json_file(self.base_pool_path, default=[])
        return content if isinstance(content, list) else []

    def save_candidate_round(self, round_index: int, candidates: List[Dict[str, Any]]) -> str:
        """Save single-round mining candidate factors."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.candidate_dir / f"round_{round_index:03d}_{timestamp}.json"
        payload = {
            "round_index": round_index,
            "timestamp": timestamp,
            "candidates": candidates,
        }
        return self._write_json_file(path, payload)

    def save_mining_feedback(self, round_index: int, mining_feedback: Dict[str, Any]) -> str:
        """Save next-round mining feedback."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.feedback_dir / f"feedback_round_{round_index:03d}_{timestamp}.json"
        payload = {
            "round_index": round_index,
            "timestamp": timestamp,
            "mining_feedback": mining_feedback,
        }
        return self._write_json_file(path, payload)

    def save_weighting_result(self, weighting_result: Dict[str, Any]) -> str:
        """Save linear weighting result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.weighting_dir / f"weighting_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "weighting_result": weighting_result,
        }
        return self._write_json_file(path, payload)

    def _read_json_file(self, path: Path, default: Any) -> Any:
        """Read JSON file."""
        if not path.exists():
            return default

        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return default

    def _write_json_file(self, path: Path, payload: Any) -> str:
        """Write JSON file."""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return str(path)
