"""
Factor Expression Conversion Service

Responsibilities:
1. Unify and standardize GP/AlphaSAGE/qlib factor expressions
2. Output a uniform structure before candidate factors enter the filtering module
3. Preserve the original expression as much as possible for later tracing and further mining
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class FactorExpressionConverterService:
    """Unified conversion entry from GP expressions to qlib expressions."""

    def __init__(self) -> None:
        self._converter = None
        self._converter_load_error: Optional[str] = None

    def normalize_candidates(
        self,
        raw_factors: List[Dict[str, Any]],
        round_index: int,
        source: str = "gp",
    ) -> List[Dict[str, Any]]:
        """Standardize a list of raw candidate factors into a uniform structure."""
        normalized: List[Dict[str, Any]] = []

        for index, raw_factor in enumerate(raw_factors, start=1):
            if not isinstance(raw_factor, dict):
                continue

            normalized_factor = self.normalize_single_factor(
                raw_factor=raw_factor,
                round_index=round_index,
                factor_index=index,
                source=source,
            )
            if normalized_factor:
                normalized.append(normalized_factor)

        return normalized

    def normalize_single_factor(
        self,
        raw_factor: Dict[str, Any],
        round_index: int,
        factor_index: int,
        source: str = "gp",
    ) -> Optional[Dict[str, Any]]:
        """Standardize a single candidate factor."""
        gp_expression = (
            raw_factor.get("gp_expression")
            or raw_factor.get("original_expression")
            or raw_factor.get("expression")
        )
        qlib_expression = raw_factor.get("qlib_expression")
        needs_cs_rank = bool(raw_factor.get("needs_cs_rank", False))
        is_valid = bool(raw_factor.get("is_valid", True))
        invalid_reason = raw_factor.get("invalid_reason")

        # If there is no qlib expression but there is an original GP expression, attempt conversion.
        if not qlib_expression and gp_expression:
            conversion_result = self.convert_expression(gp_expression)
            qlib_expression = conversion_result.get("qlib_expression")
            needs_cs_rank = conversion_result.get("needs_cs_rank", False)
            is_valid = conversion_result.get("is_valid", False)
            invalid_reason = conversion_result.get("invalid_reason")
        elif qlib_expression:
            is_valid = bool(is_valid and qlib_expression)

        # Discard directly when neither expression exists.
        if not gp_expression and not qlib_expression:
            return None

        train_ic_raw = raw_factor.get("train_ic")
        if train_ic_raw is None:
            train_ic_raw = raw_factor.get("ic")
        train_ic = self._safe_float(train_ic_raw, default=0.0)

        train_rank_ic_raw = raw_factor.get("train_rank_ic")
        if train_rank_ic_raw is None:
            train_rank_ic_raw = raw_factor.get("rank_ic")
        train_rank_ic = self._safe_float(train_rank_ic_raw, default=0.0)

        valid_ic_raw = raw_factor.get("valid_ic")
        if valid_ic_raw is None:
            valid_ic_raw = raw_factor.get("ic_valid")
        valid_ic = self._safe_float(valid_ic_raw, default=0.0)

        valid_rank_ic_raw = raw_factor.get("valid_rank_ic")
        if valid_rank_ic_raw is None:
            valid_rank_ic_raw = raw_factor.get("rank_ic_valid")
        valid_rank_ic = self._safe_float(valid_rank_ic_raw, default=0.0)

        test_ic_raw = raw_factor.get("test_ic")
        if test_ic_raw is None:
            test_ic_raw = raw_factor.get("ic_test")
        test_ic = self._safe_float(test_ic_raw, default=0.0)

        test_rank_ic_raw = raw_factor.get("test_rank_ic")
        if test_rank_ic_raw is None:
            test_rank_ic_raw = raw_factor.get("rank_ic_test")
        test_rank_ic = self._safe_float(test_rank_ic_raw, default=0.0)
        metrics = self._build_metrics(
            raw_factor=raw_factor,
            train_ic=train_ic,
            train_rank_ic=train_rank_ic,
            valid_ic=valid_ic,
            valid_rank_ic=valid_rank_ic,
            test_ic=test_ic,
            test_rank_ic=test_rank_ic,
        )

        factor_id = raw_factor.get("factor_id") or self._build_factor_id(
            qlib_expression=qlib_expression,
            round_index=round_index,
            factor_index=factor_index,
        )

        normalized_factor = {
            "factor_id": factor_id,
            "source": source,
            "round_index": round_index,
            "gp_expression": gp_expression,
            "qlib_expression": qlib_expression,
            "is_valid": is_valid,
            "invalid_reason": invalid_reason,
            "needs_cs_rank": needs_cs_rank,
            "train_ic": train_ic,
            "train_rank_ic": train_rank_ic,
            "valid_ic": valid_ic,
            "valid_rank_ic": valid_rank_ic,
            "test_ic": test_ic,
            "test_rank_ic": test_rank_ic,
            "metrics": metrics,
            "metadata": raw_factor.get("metadata", {}),
            "raw_factor": raw_factor,
        }

        return normalized_factor

    def convert_expression(self, gp_expression: str) -> Dict[str, Any]:
        """
        Convert a single GP expression to a qlib expression.

        If the converter fails to load, attempt graceful degradation:
        - Preserve the original expression by default
        - Mark it as invalid to prevent erroneous expressions from entering the official factor library
        """
        converter = self._load_converter()
        if converter is None:
            return {
                "qlib_expression": None,
                "needs_cs_rank": False,
                "is_valid": False,
                "invalid_reason": self._converter_load_error or "Expression converter unavailable",
            }

        try:
            result = converter.convert_with_metadata(gp_expression)
            return {
                "qlib_expression": result.get("qlib_expression"),
                "needs_cs_rank": bool(result.get("needs_cs_rank", False)),
                "is_valid": bool(result.get("is_valid", False)),
                "invalid_reason": result.get("invalid_reason"),
            }
        except Exception as exc:
            return {
                "qlib_expression": None,
                "needs_cs_rank": False,
                "is_valid": False,
                "invalid_reason": f"Expression conversion failed: {exc}",
            }

    def _load_converter(self):
        """Lazily load the AlphaSAGE -> qlib converter."""
        if self._converter is not None:
            return self._converter

        try:
            workspace_src = (
                Path(__file__).parent.parent.parent
                / "Qlib_MCP"
                / "workspace"
                / "AlphaSAGE"
                / "src"
            )
            if str(workspace_src) not in sys.path:
                sys.path.insert(0, str(workspace_src))

            from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter

            self._converter = AlphaSAGEToQlibConverter()
        except Exception as exc:
            self._converter = None
            self._converter_load_error = f"Failed to load expression converter: {exc}"

        return self._converter

    def _build_factor_id(self, qlib_expression: Optional[str], round_index: int, factor_index: int) -> str:
        """Generate a stable factor ID from the expression."""
        content = qlib_expression or f"round_{round_index}_factor_{factor_index}"
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
        return f"fac_{round_index:03d}_{factor_index:03d}_{digest}"

    def _build_metrics(
        self,
        raw_factor: Dict[str, Any],
        train_ic: float,
        train_rank_ic: float,
        valid_ic: float,
        valid_rank_ic: float,
        test_ic: float,
        test_rank_ic: float,
    ) -> Dict[str, Any]:
        """Unified construction of train/valid/test metric structure."""
        metrics = raw_factor.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        train_metrics = metrics.get("train", {}) if isinstance(metrics.get("train", {}), dict) else {}
        valid_metrics = metrics.get("valid", {}) if isinstance(metrics.get("valid", {}), dict) else {}
        test_metrics = metrics.get("test", {}) if isinstance(metrics.get("test", {}), dict) else {}

        return {
            "train": {
                "ic": train_ic,
                "rank_ic": train_rank_ic,
                "icir": self._safe_float(train_metrics.get("icir"), 0.0),
                "rank_icir": self._safe_float(train_metrics.get("rank_icir"), 0.0),
            },
            "valid": {
                "ic": valid_ic,
                "rank_ic": valid_rank_ic,
                "icir": self._safe_float(valid_metrics.get("icir"), 0.0),
                "rank_icir": self._safe_float(valid_metrics.get("rank_icir"), 0.0),
            },
            "test": {
                "ic": test_ic,
                "rank_ic": test_rank_ic,
                "icir": self._safe_float(test_metrics.get("icir"), 0.0),
                "rank_icir": self._safe_float(test_metrics.get("rank_icir"), 0.0),
            },
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert any numeric value to float."""
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
