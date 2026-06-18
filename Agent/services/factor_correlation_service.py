"""
Factor Correlation Service

Prioritizes using qlib to compute real factor series correlations;
if the environment is incomplete, degrades to expression-level approximate similarity
to ensure the pipeline can continue executing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..evaluators.correlation_evaluator import CorrelationEvaluator


class FactorCorrelationService:
    """Service for comparing correlation between candidate factors and the factor library."""

    def __init__(self) -> None:
        self.evaluator = CorrelationEvaluator()
        repo_root = Path(__file__).resolve().parents[2]
        self.cache_dir = repo_root / "data" / "factor_value_cache"
        self.series_cache_dir = self.cache_dir / "series"
        self.series_cache_dir.mkdir(parents=True, exist_ok=True)
        self.library_matrix_path = self.cache_dir / "library_matrix.pkl"
        self.library_matrix_meta_path = self.cache_dir / "library_matrix_meta.json"
        self.default_instruments = "csi300"
        self.default_start_date = "2020-01-01"
        self.default_end_date = "2023-12-31"
        self._series_memory_cache: Dict[str, pd.Series] = {}
        self._library_matrix_cache: Optional[pd.DataFrame] = None
        self._library_factor_by_column: Dict[str, Dict[str, Any]] = {}
        self._library_matrix_signature: Optional[str] = None
        self.factor_value_timeout_seconds = self._read_int_env(
            "FACTOR_VALUE_TIMEOUT_SECONDS",
            300,
        )

    def find_high_correlation_matches(
        self,
        candidate: Dict[str, Any],
        factor_library: List[Dict[str, Any]],
        threshold: float = 0.90,
    ) -> List[Dict[str, Any]]:
        """Identify library factors that are highly correlated with the candidate factor."""
        matches: List[Dict[str, Any]] = []
        candidate_expression = candidate.get("qlib_expression")
        if not candidate_expression:
            return matches

        factor_id = candidate.get("factor_id") or "-"
        print(
            "[FactorSelection][Correlation] Starting correlation check: "
            f"factor_id={factor_id}, library_size={len(factor_library)}"
        )

        candidate_values, error = self._get_or_calculate_factor_values(candidate)
        if candidate_values is None:
            print(
                "[FactorSelection][Correlation] qlib factor values unavailable, falling back to expression similarity: "
                f"factor_id={factor_id}, reason={error or 'unknown'}"
            )
            return self._find_matches_with_expression_similarity(candidate, factor_library, threshold)

        library_matrix, factor_by_column = self._get_library_matrix(factor_library)
        correlations = self._compute_correlations_to_matrix(candidate_values, library_matrix)

        compared_factor_ids = set()
        for column, library_factor in factor_by_column.items():
            compared_factor_ids.add(id(library_factor))
            correlation = correlations.get(column)
            if correlation is None:
                correlation = self._compute_with_expression_similarity(candidate, library_factor)
            if correlation >= threshold:
                matches.append(
                    {
                        "factor": library_factor,
                        "correlation": correlation,
                    }
                )

        for library_factor in factor_library:
            if id(library_factor) in compared_factor_ids:
                continue
            correlation = self._compute_with_expression_similarity(candidate, library_factor)
            if correlation >= threshold:
                matches.append({"factor": library_factor, "correlation": correlation})

        matches.sort(key=lambda item: item["correlation"], reverse=True)
        print(
            "[FactorSelection][Correlation] Correlation check completed: "
            f"factor_id={factor_id}, qlib_compared={len(factor_by_column)}, "
            f"high_corr_matches={len(matches)}"
        )
        return matches

    def validate_candidate_for_library_entry(
        self,
        candidate: Dict[str, Any],
        *,
        instruments: str = "csi300",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ) -> Dict[str, Any]:
        """Validate that a candidate factor can be calculated by qlib before library entry."""
        expression = candidate.get("qlib_expression")
        if not expression:
            return {
                "is_calculable": False,
                "reason": "Candidate factor lacks qlib_expression, unable to perform qlib computability validation",
            }

        factor_id = candidate.get("factor_id") or "-"
        print(
            "[FactorSelection][QlibValidation] Starting computability validation: "
            f"factor_id={factor_id}, expression={expression}",
            flush=True,
        )
        factor_values, error = self._get_or_calculate_factor_values(
            candidate,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
        )
        if factor_values is None:
            print(
                "[FactorSelection][QlibValidation] Computability validation failed: "
                f"factor_id={factor_id}, reason={error or 'unknown'}",
                flush=True,
            )
            return {
                "is_calculable": False,
                "reason": error or "Qlib cannot compute this factor expression",
            }

        print(
            "[FactorSelection][QlibValidation] Computability validation completed: "
            f"factor_id={factor_id}, value_count={len(factor_values)}",
            flush=True,
        )
        return {
            "is_calculable": True,
            "reason": "",
            "value_count": int(len(factor_values)),
        }

    def compute_correlation(
        self,
        factor_left: Dict[str, Any],
        factor_right: Dict[str, Any],
    ) -> float:
        """Compute the correlation between two factors."""
        correlation = self._compute_with_qlib(factor_left, factor_right)
        if correlation is not None:
            return correlation

        return self._compute_with_expression_similarity(factor_left, factor_right)

    def _compute_with_qlib(
        self,
        factor_left: Dict[str, Any],
        factor_right: Dict[str, Any],
    ) -> float | None:
        """Compute real factor series correlation using qlib."""
        left_expression = factor_left.get("qlib_expression")
        right_expression = factor_right.get("qlib_expression")
        if not left_expression or not right_expression:
            return None

        try:
            left_values, _ = self._get_or_calculate_factor_values(factor_left)
            right_values, _ = self._get_or_calculate_factor_values(factor_right)
            if left_values is None or right_values is None:
                return None

            return self._compute_series_correlation(left_values, right_values)
        except Exception:
            return None

    def _get_or_calculate_factor_values(
        self,
        factor: Dict[str, Any],
        *,
        instruments: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Load factor values from cache or calculate them once with qlib."""
        expression = factor.get("qlib_expression")
        if not expression:
            return None, "qlib_expression is empty"

        instruments = instruments or self.default_instruments
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date
        cache_key, cache_payload = self._factor_value_cache_key(
            expression=expression,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            needs_cs_rank=bool(factor.get("needs_cs_rank", False)),
        )

        cached_values = self._load_cached_series(cache_key)
        if cached_values is not None:
            return cached_values, None

        factor_id = factor.get("factor_id") or "-"
        if not self.evaluator._init_qlib():
            return None, "Qlib initialization failed, cannot compute factor values"

        started_at = time.monotonic()
        print(
            "[FactorSelection][QlibValue] Starting factor value computation: "
            f"factor_id={factor_id}, timeout={self.factor_value_timeout_seconds}s, "
            f"expression={expression}",
            flush=True,
        )
        try:
            with self._factor_value_time_limit(factor_id, expression):
                factor_values, error = self.evaluator._calculate_factor_values_with_error(
                    expression=expression,
                    instruments=instruments,
                    start_date=start_date,
                    end_date=end_date,
                    needs_cs_rank=bool(factor.get("needs_cs_rank", False)),
                )
        except TimeoutError as exc:
            elapsed = time.monotonic() - started_at
            print(
                "[FactorSelection][QlibValue] Factor value computation timed out: "
                f"factor_id={factor_id}, elapsed={elapsed:.1f}s, error={exc}",
                flush=True,
            )
            return None, str(exc)

        normalized_values = self._normalize_factor_values(factor_values)
        if normalized_values is None:
            elapsed = time.monotonic() - started_at
            print(
                "[FactorSelection][QlibValue] Factor value computation failed: "
                f"factor_id={factor_id}, elapsed={elapsed:.1f}s, "
                f"reason={error or 'Qlib computation result is empty or all missing values'}",
                flush=True,
            )
            return None, error or "Qlib computation result is empty or all missing values"

        self._store_cached_series(cache_key, normalized_values, cache_payload, factor)
        elapsed = time.monotonic() - started_at
        print(
            "[FactorSelection][QlibValue] Factor value computation completed: "
            f"factor_id={factor_id}, value_count={len(normalized_values)}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        return normalized_values, None

    @contextmanager
    def _factor_value_time_limit(
        self,
        factor_id: str,
        expression: str,
    ) -> Iterator[None]:
        seconds = int(self.factor_value_timeout_seconds)
        if (
            seconds <= 0
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")
        ):
            yield
            return

        previous_handler = signal.getsignal(signal.SIGALRM)

        def _raise_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError(
                "Qlib factor value computation timed out"
                f"({seconds}s): factor_id={factor_id}, expression={expression[:200]}"
            )

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _get_library_matrix(
        self,
        factor_library: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Build or load the factor library value matrix."""
        library_columns = self._build_library_columns(factor_library)
        signature = self._library_matrix_cache_signature(library_columns)
        if (
            self._library_matrix_cache is not None
            and self._library_matrix_signature == signature
        ):
            return self._library_matrix_cache, self._library_factor_by_column

        loaded = self._load_library_matrix_snapshot(signature, library_columns)
        if loaded is not None:
            matrix, factor_by_column = loaded
            self._library_matrix_cache = matrix
            self._library_factor_by_column = factor_by_column
            self._library_matrix_signature = signature
            return matrix, factor_by_column

        series_list: List[pd.Series] = []
        factor_by_column: Dict[str, Dict[str, Any]] = {}
        failed_count = 0
        for column, _, library_factor in library_columns:
            values, error = self._get_or_calculate_factor_values(library_factor)
            if values is None:
                failed_count += 1
                print(
                    "[FactorSelection][Correlation] Skipping incomputable library factor: "
                    f"factor_id={library_factor.get('factor_id') or '-'}, "
                    f"reason={error or 'unknown'}"
                )
                continue
            series_list.append(values.rename(column))
            factor_by_column[column] = library_factor

        matrix = pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
        self._library_matrix_cache = matrix
        self._library_factor_by_column = factor_by_column
        self._library_matrix_signature = signature
        self._store_library_matrix_snapshot(matrix, signature, library_columns, failed_count)
        return matrix, factor_by_column

    def _compute_series_correlation(
        self,
        left_values: pd.Series,
        right_values: pd.Series,
    ) -> Optional[float]:
        correlations = self._compute_correlations_to_matrix(
            left_values,
            right_values.rename("right").to_frame(),
        )
        correlation = correlations.get("right")
        return float(correlation) if correlation is not None else None

    def _compute_correlations_to_matrix(
        self,
        candidate_values: pd.Series,
        library_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute mean daily cross-sectional correlations against a matrix."""
        candidate_values = self._normalize_factor_values(candidate_values)
        if candidate_values is None or library_matrix.empty:
            return {}

        candidate_name = "__candidate__"
        aligned = library_matrix.join(candidate_values.rename(candidate_name), how="inner")
        if len(aligned) < 20:
            return {}
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        aligned = aligned.dropna(subset=[candidate_name])
        if len(aligned) < 20:
            return {}

        if not isinstance(aligned.index, pd.MultiIndex):
            factor_block = aligned.drop(columns=[candidate_name])
            valid_counts = factor_block.notna().mul(aligned[candidate_name].notna(), axis=0).sum()
            correlations = factor_block.corrwith(aligned[candidate_name])
            correlations = correlations[(valid_counts >= 3) & correlations.notna()]
            return {str(column): float(value) for column, value in correlations.items()}

        columns = [column for column in aligned.columns if column != candidate_name]
        sums = pd.Series(0.0, index=columns, dtype="float64")
        counts = pd.Series(0, index=columns, dtype="int64")
        for _, group in aligned.groupby(level=0, sort=False):
            if len(group) < 3:
                continue
            candidate_group = group[candidate_name]
            factor_group = group[columns]
            valid_counts = factor_group.notna().mul(candidate_group.notna(), axis=0).sum()
            daily_correlations = factor_group.corrwith(candidate_group)
            daily_correlations = daily_correlations[
                (valid_counts >= 3) & daily_correlations.notna()
            ]
            if daily_correlations.empty:
                continue
            sums.loc[daily_correlations.index] += daily_correlations.astype("float64")
            counts.loc[daily_correlations.index] += 1

        valid = counts > 0
        if not valid.any():
            return {}
        means = (sums[valid] / counts[valid]).dropna()
        return {str(column): float(value) for column, value in means.items()}

    def _find_matches_with_expression_similarity(
        self,
        candidate: Dict[str, Any],
        factor_library: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for library_factor in factor_library:
            correlation = self._compute_with_expression_similarity(candidate, library_factor)
            if correlation >= threshold:
                matches.append({"factor": library_factor, "correlation": correlation})
        matches.sort(key=lambda item: item["correlation"], reverse=True)
        return matches

    def _factor_value_cache_key(
        self,
        *,
        expression: str,
        instruments: str,
        start_date: str,
        end_date: str,
        needs_cs_rank: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "cache_version": 1,
            "expression": expression.strip(),
            "instruments": instruments,
            "start_date": start_date,
            "end_date": end_date,
            "needs_cs_rank": bool(needs_cs_rank),
            "qlib_path": str(self.evaluator._qlib_path or ""),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), payload

    def _series_cache_path(self, cache_key: str) -> Path:
        return self.series_cache_dir / f"{cache_key}.pkl"

    def _series_meta_path(self, cache_key: str) -> Path:
        return self.series_cache_dir / f"{cache_key}.json"

    def _load_cached_series(self, cache_key: str) -> Optional[pd.Series]:
        if cache_key in self._series_memory_cache:
            return self._series_memory_cache[cache_key]

        path = self._series_cache_path(cache_key)
        if not path.exists():
            return None

        try:
            values = pd.read_pickle(path)
            values = self._normalize_factor_values(values)
            if values is None:
                return None
            self._series_memory_cache[cache_key] = values
            return values
        except Exception as exc:
            print(f"Warning: Failed to load factor value cache {path}: {exc}")
            return None

    def _store_cached_series(
        self,
        cache_key: str,
        values: pd.Series,
        cache_payload: Dict[str, Any],
        factor: Dict[str, Any],
    ) -> None:
        self._series_memory_cache[cache_key] = values
        path = self._series_cache_path(cache_key)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            values.to_pickle(temp_path)
            temp_path.replace(path)
            metadata = {
                **cache_payload,
                "factor_id": factor.get("factor_id"),
                "value_count": int(len(values)),
                "dtype": str(values.dtype),
            }
            self._write_json_atomic(self._series_meta_path(cache_key), metadata)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _normalize_factor_values(self, values: Any) -> Optional[pd.Series]:
        if values is None:
            return None
        if isinstance(values, pd.DataFrame):
            if values.empty:
                return None
            series = values.iloc[:, 0]
        elif isinstance(values, pd.Series):
            series = values
        else:
            series = pd.Series(values)

        series = pd.to_numeric(series, errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return None
        if isinstance(series.index, pd.MultiIndex) and series.index.nlevels == 2:
            series.index.names = ["datetime", "instrument"]
        return series.astype("float32", copy=False)

    def _build_library_columns(
        self,
        factor_library: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        columns: List[Tuple[str, str, Dict[str, Any]]] = []
        seen_columns: set[str] = set()
        for index, factor in enumerate(factor_library):
            expression = factor.get("qlib_expression")
            if not expression:
                continue
            cache_key, _ = self._factor_value_cache_key(
                expression=expression,
                instruments=self.default_instruments,
                start_date=self.default_start_date,
                end_date=self.default_end_date,
                needs_cs_rank=bool(factor.get("needs_cs_rank", False)),
            )
            base_column = str(factor.get("factor_id") or cache_key[:16])
            column = base_column
            if column in seen_columns:
                column = f"{base_column}__{index}"
            seen_columns.add(column)
            columns.append((column, cache_key, factor))
        return columns

    def _library_matrix_cache_signature(
        self,
        library_columns: List[Tuple[str, str, Dict[str, Any]]],
    ) -> str:
        payload = {
            "cache_version": 1,
            "items": [
                {
                    "column": column,
                    "value_key": cache_key,
                    "factor_id": factor.get("factor_id"),
                }
                for column, cache_key, factor in library_columns
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_library_matrix_snapshot(
        self,
        signature: str,
        library_columns: List[Tuple[str, str, Dict[str, Any]]],
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]]:
        if not self.library_matrix_path.exists() or not self.library_matrix_meta_path.exists():
            return None
        try:
            metadata = self._read_json(self.library_matrix_meta_path, default={})
            if metadata.get("signature") != signature:
                return None
            matrix = pd.read_pickle(self.library_matrix_path)
            expected_columns = [column for column, _, _ in library_columns]
            missing_columns = [column for column in expected_columns if column not in matrix.columns]
            if missing_columns:
                return None
            matrix = matrix[expected_columns]
            factor_by_column = {
                column: factor
                for column, _, factor in library_columns
            }
            return matrix, factor_by_column
        except Exception as exc:
            print(f"Warning: Failed to load library factor matrix cache: {exc}")
            return None

    def _store_library_matrix_snapshot(
        self,
        matrix: pd.DataFrame,
        signature: str,
        library_columns: List[Tuple[str, str, Dict[str, Any]]],
        failed_count: int,
    ) -> None:
        temp_path = self.library_matrix_path.with_name(
            f".{self.library_matrix_path.name}.{os.getpid()}.tmp"
        )
        try:
            matrix.to_pickle(temp_path)
            temp_path.replace(self.library_matrix_path)
            metadata = {
                "signature": signature,
                "cache_version": 1,
                "factor_count": int(len(matrix.columns)),
                "failed_count": int(failed_count),
                "rows": int(len(matrix)),
                "columns": list(matrix.columns),
                "library_factors": [
                    {
                        "column": column,
                        "value_key": cache_key,
                        "factor_id": factor.get("factor_id"),
                    }
                    for column, cache_key, factor in library_columns
                ],
            }
            self._write_json_atomic(self.library_matrix_meta_path, metadata)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return default

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            return default

    def _compute_with_expression_similarity(
        self,
        factor_left: Dict[str, Any],
        factor_right: Dict[str, Any],
    ) -> float:
        """
        Use expression token similarity as an approximate substitute for correlation.

        This is a pipeline fallback and should not replace real return series correlation.
        """
        left_expression = factor_left.get("qlib_expression", "")
        right_expression = factor_right.get("qlib_expression", "")
        if not left_expression or not right_expression:
            return 0.0

        if left_expression == right_expression:
            return 1.0

        left_tokens = self._tokenize_expression(left_expression)
        right_tokens = self._tokenize_expression(right_expression)
        if not left_tokens or not right_tokens:
            return 0.0

        intersection = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(intersection) / max(len(union), 1)

    @staticmethod
    def _tokenize_expression(expression: str) -> set[str]:
        """Tokenize the expression into operators and field tokens."""
        tokens = re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", expression or "")
        return {token for token in tokens if token}
