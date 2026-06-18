"""
Rolling Factor Performance Service

Responsible for:
1. Initializing the Qlib data environment
2. Computing rolling metrics according to the profiles required by the selector
3. Writing real rolling performance metrics back into the structured factor library
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import pandas as pd

from Agent.services.factor_performance import (
    CROSS_SECTIONAL_PROFILE_NAME,
    TOPK_RETURN_PROFILE_NAME,
    build_cross_sectional_snapshot,
    build_topk_return_snapshot,
    compute_cross_sectional_daily_metrics,
    compute_topk_return_daily_metrics,
)


class RollingFactorPerformanceService:
    """Compute real rolling factor performance based on Qlib."""

    DEFAULT_RETURN_EXPRESSION = "Ref($close, -11)/Ref($close, -1) - 1"
    PROFILE_FULL = "full"
    PROFILE_CROSS_SECTIONAL = CROSS_SECTIONAL_PROFILE_NAME
    PROFILE_TOPK_RETURN = TOPK_RETURN_PROFILE_NAME

    def __init__(self) -> None:
        self._qlib_initialized = False
        self._provider_uri: Optional[Path] = None
        self._D = None
        self._CSRankNorm = None
        self._instrument_cache: Dict[tuple[str, str, str], List[str]] = {}
        self._invalid_factor_ids: set[str] = set()
        self._daily_metric_cache: Dict[str, Dict[tuple[str, str], Dict[str, Any]]] = {
            self.PROFILE_CROSS_SECTIONAL: {},
            self.PROFILE_TOPK_RETURN: {},
        }

    def enrich_factor_library(
        self,
        factor_library: List[Dict[str, Any]],
        *,
        window_days: int,
        top_k: int,
        batch_size: int = 8,
        selection_date: str,
        instrument: str,
        benchmark: str,
        provider_uri: Optional[str] = None,
        return_expression: Optional[str] = None,
        metric_profile: str = PROFILE_FULL,
    ) -> Dict[str, Any]:
        """
        Fill in real rolling metrics for the factor library.

        If Qlib initialization or single-factor computation fails, do not raise an exception;
        instead preserve existing fields so upstream logic can continue to run
        in degraded mode using fallback values.
        """
        resolved_profiles = self._resolve_metric_profiles(metric_profile)
        context = {
            "status": "skipped",
            "selection_date": selection_date,
            "window_days": int(window_days),
            "top_k": int(top_k),
            "instrument": instrument,
            "benchmark": benchmark,
            "provider_uri": None,
            "evaluation_dates": [],
            "enriched_factor_count": 0,
            "failed_factor_count": 0,
            "return_expression": return_expression or self.DEFAULT_RETURN_EXPRESSION,
            "metric_profile": metric_profile,
            "resolved_metric_profiles": list(resolved_profiles),
        }
        context["label_lookahead_days"] = self._infer_label_lookahead_days(context["return_expression"])

        if not factor_library:
            context["status"] = "empty_factor_library"
            return context

        enrich_start = perf_counter()
        if not self._init_qlib(provider_uri):
            context["status"] = "qlib_init_failed"
            return context

        evaluation_dates = self._resolve_evaluation_dates(
            selection_date=selection_date,
            window_days=int(window_days),
            label_lookahead_days=int(context["label_lookahead_days"]),
        )
        if not evaluation_dates:
            context["status"] = "insufficient_calendar"
            context["provider_uri"] = str(self._provider_uri) if self._provider_uri else None
            return context

        start_date = evaluation_dates[0].strftime("%Y-%m-%d")
        end_date = evaluation_dates[-1].strftime("%Y-%m-%d")
        instrument_list = self._load_instrument_list(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
        )
        benchmark_returns: Dict[pd.Timestamp, float] = {}
        if self.PROFILE_TOPK_RETURN in resolved_profiles:
            benchmark_returns = self._load_benchmark_returns(
                benchmark=benchmark,
                start_date=start_date,
                end_date=end_date,
                return_expression=context["return_expression"],
            )

        context["provider_uri"] = str(self._provider_uri) if self._provider_uri else None
        context["evaluation_dates"] = [value.strftime("%Y-%m-%d") for value in evaluation_dates]

        if not instrument_list:
            context["status"] = "market_data_unavailable"
            return context
        if self.PROFILE_TOPK_RETURN in resolved_profiles and not benchmark_returns:
            context["status"] = "market_data_unavailable"
            return context

        print(
            "[RollingFactorPerformance] start batch evaluation: "
            f"factors={len(factor_library)}, instrument_count={len(instrument_list)}, "
            f"dates={len(evaluation_dates)}, range={start_date}->{end_date}, "
            f"batch_size={int(batch_size)}, profiles={resolved_profiles}"
        )
        batch_start = perf_counter()
        batch_snapshots = self._compute_batch_recent_performance(
            factor_library=factor_library,
            instrument_list=instrument_list,
            evaluation_dates=evaluation_dates,
            benchmark_returns=benchmark_returns,
            top_k=int(top_k),
            batch_size=int(batch_size),
            return_expression=context["return_expression"],
            metric_profiles=resolved_profiles,
        )
        print(
            "[RollingFactorPerformance] batch evaluation finished: "
            f"elapsed={perf_counter() - batch_start:.2f}s, snapshots={len(batch_snapshots)}"
        )

        enriched_count = 0
        failed_count = 0
        for index, factor in enumerate(factor_library):
            snapshot = batch_snapshots.get(index)
            factor.update(self._stale_profile_payloads(resolved_profiles))
            if snapshot is None:
                failed_count += 1
                factor.update(self._empty_profile_payloads(resolved_profiles))
                factor["recent_performance_source"] = "qlib_profile_failed"
                factor["recent_performance_error"] = factor.get(
                    "recent_performance_error",
                    "Batch rolling performance evaluation failed.",
                )
                continue

            factor.update(snapshot)
            enriched_count += 1

        context["status"] = "success" if enriched_count > 0 else "factor_evaluation_failed"
        context["enriched_factor_count"] = enriched_count
        context["failed_factor_count"] = failed_count
        print(
            "[RollingFactorPerformance] enrich finished: "
            f"elapsed={perf_counter() - enrich_start:.2f}s, "
            f"enriched={enriched_count}, failed={failed_count}"
        )
        return context

    def _init_qlib(self, provider_uri: Optional[str]) -> bool:
        if self._qlib_initialized and self._provider_uri is not None:
            return True

        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            from qlib.data.dataset.processor import CSRankNorm

            resolved_provider_uri = self._resolve_provider_uri(provider_uri)
            qlib.init(provider_uri=str(resolved_provider_uri), region=REG_CN)
            self._provider_uri = resolved_provider_uri
            self._D = D
            self._CSRankNorm = CSRankNorm
            self._qlib_initialized = True
            return True
        except Exception:
            return False

    def _resolve_provider_uri(self, provider_uri: Optional[str]) -> Path:
        candidates = []
        if provider_uri:
            candidates.append(Path(provider_uri).expanduser())

        candidates.extend(
            [
                Path("~/.qlib/qlib_data/cn_data").expanduser(),
                Path("/home/batchcom/.qlib/qlib_data/cn_data"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError("No Qlib data path found for rolling factor performance.")

    def _resolve_evaluation_dates(
        self,
        selection_date: str,
        window_days: int,
        label_lookahead_days: int = 1,
    ) -> List[pd.Timestamp]:
        selection_ts = pd.Timestamp(selection_date)
        lookahead_days = max(int(label_lookahead_days), 1)
        calendar_start = (selection_ts - pd.Timedelta(days=max((window_days + lookahead_days) * 10, 60))).strftime(
            "%Y-%m-%d"
        )
        calendar = self._D.calendar(start_time=calendar_start, end_time=selection_date)
        all_dates = [pd.Timestamp(value) for value in calendar if pd.Timestamp(value) <= selection_ts]
        if len(all_dates) <= lookahead_days:
            return []
        dates = all_dates[:-lookahead_days]
        return dates[-window_days:]

    def _load_instrument_list(self, instrument: str, start_date: str, end_date: str) -> List[str]:
        cache_key = (instrument, start_date, end_date)
        cached = self._instrument_cache.get(cache_key)
        if cached is not None:
            return cached

        instrument_list = self._D.list_instruments(
            instruments=self._D.instruments(instrument),
            start_time=start_date,
            end_time=end_date,
            as_list=True,
        )
        result = list(instrument_list) if instrument_list else []
        self._instrument_cache[cache_key] = result
        return result

    def _load_benchmark_returns(
        self,
        *,
        benchmark: str,
        start_date: str,
        end_date: str,
        return_expression: str,
    ) -> Dict[pd.Timestamp, float]:
        benchmark_code = self._benchmark_code(benchmark)
        benchmark_df = self._D.features(
            [benchmark_code],
            [return_expression],
            start_time=start_date,
            end_time=end_date,
        )
        if benchmark_df is None or benchmark_df.empty:
            return {}

        benchmark_df = benchmark_df.reset_index()
        if return_expression not in benchmark_df.columns:
            return {}

        result: Dict[pd.Timestamp, float] = {}
        for _, row in benchmark_df.iterrows():
            value = row.get(return_expression)
            if pd.isna(value):
                continue
            result[pd.Timestamp(row["datetime"])] = float(value)
        return result

    def _compute_batch_recent_performance(
        self,
        *,
        factor_library: List[Dict[str, Any]],
        instrument_list: List[str],
        evaluation_dates: List[pd.Timestamp],
        benchmark_returns: Dict[pd.Timestamp, float],
        top_k: int,
        batch_size: int,
        return_expression: str,
        metric_profiles: List[str],
    ) -> Dict[int, Dict[str, Any] | None]:
        indexed_factors = [
            (index, factor)
            for index, factor in enumerate(factor_library)
            if factor.get("qlib_expression") and factor.get("factor_id") not in self._invalid_factor_ids
        ]
        if not indexed_factors or not evaluation_dates:
            return {}

        pending_factors, missing_dates = self._collect_pending_factors(
            indexed_factors=indexed_factors,
            evaluation_dates=evaluation_dates,
            metric_profiles=metric_profiles,
        )
        if pending_factors and missing_dates:
            start_date = missing_dates[0].strftime("%Y-%m-%d")
            end_date = missing_dates[-1].strftime("%Y-%m-%d")
            try:
                load_start = perf_counter()
                directional_plan = (
                    self._build_directional_topk_load_plan(pending_factors)
                    if self._should_use_directional_topk_cache(
                        indexed_factors=pending_factors,
                        metric_profiles=metric_profiles,
                    )
                    else None
                )
                if directional_plan is not None:
                    factor_df = self._load_batch_factor_frame(
                        indexed_factors=directional_plan["load_factors"],
                        instrument_list=instrument_list,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        return_expression=return_expression,
                    )
                    print(
                        "[RollingFactorPerformance] loaded directional base factor frame: "
                        f"expert_count={len(pending_factors)}, "
                        f"base_factor_count={len(directional_plan['load_factors'])}, "
                        f"elapsed={perf_counter() - load_start:.2f}s"
                    )
                    if factor_df is not None and not factor_df.empty:
                        self._cache_directional_topk_metrics_from_factor_frame(
                            indexed_factors=pending_factors,
                            load_plan=directional_plan,
                            factor_df=factor_df,
                            evaluation_dates=missing_dates,
                            benchmark_returns=benchmark_returns,
                            top_k=top_k,
                        )
                else:
                    factor_df = self._load_batch_factor_frame(
                        indexed_factors=pending_factors,
                        instrument_list=instrument_list,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        return_expression=return_expression,
                    )
                    print(
                        "[RollingFactorPerformance] loaded batch factor frame: "
                        f"factor_count={len(pending_factors)}, elapsed={perf_counter() - load_start:.2f}s"
                    )
                    if factor_df is not None and not factor_df.empty:
                        self._cache_daily_metrics_from_factor_frame(
                            indexed_factors=pending_factors,
                            factor_df=factor_df,
                            evaluation_dates=missing_dates,
                            benchmark_returns=benchmark_returns,
                            top_k=top_k,
                            metric_profiles=metric_profiles,
                        )
            except Exception as exc:
                error_message = str(exc)
                print(
                    "[RollingFactorPerformance] batch evaluation fallback triggered: "
                    f"{type(exc).__name__}: {error_message}"
                )
                compatible_factors, invalid_factors = self._partition_batch_compatible_factors(
                    indexed_factors=pending_factors,
                    instrument_list=instrument_list,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=batch_size,
                    return_expression=return_expression,
                )

                for _, factor in invalid_factors:
                    factor_id = str(factor.get("factor_id", ""))
                    if factor_id:
                        self._invalid_factor_ids.add(factor_id)
                    print(
                        "[RollingFactorPerformance] invalid factor skipped: "
                        f"factor_id={factor_id or 'unknown'}, "
                        f"expression={factor.get('qlib_expression')}"
                    )
                    factor["recent_performance_error"] = error_message

                if compatible_factors:
                    factor_df = self._load_batch_factor_frame(
                        indexed_factors=compatible_factors,
                        instrument_list=instrument_list,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        return_expression=return_expression,
                    )
                    if factor_df is not None and not factor_df.empty:
                        self._cache_daily_metrics_from_factor_frame(
                            indexed_factors=compatible_factors,
                            factor_df=factor_df,
                            evaluation_dates=missing_dates,
                            benchmark_returns=benchmark_returns,
                            top_k=top_k,
                            metric_profiles=metric_profiles,
                        )

        return self._build_snapshots_from_cache(
            indexed_factors=indexed_factors,
            evaluation_dates=evaluation_dates,
            metric_profiles=metric_profiles,
        )

    def _load_batch_factor_frame(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        instrument_list: List[str],
        start_date: str,
        end_date: str,
        batch_size: int,
        return_expression: str,
    ):
        normalized_batch_size = max(1, int(batch_size))
        if len(indexed_factors) <= normalized_batch_size:
            expressions = [factor["qlib_expression"] for _, factor in indexed_factors] + [return_expression]
            return self._D.features(
                instrument_list,
                expressions,
                start_time=start_date,
                end_time=end_date,
            )

        feature_frames = []
        return_series = None
        total_batches = (len(indexed_factors) + normalized_batch_size - 1) // normalized_batch_size
        for batch_index, start_idx in enumerate(range(0, len(indexed_factors), normalized_batch_size), start=1):
            factor_batch = indexed_factors[start_idx : start_idx + normalized_batch_size]
            batch_expressions = [factor["qlib_expression"] for _, factor in factor_batch] + [return_expression]
            batch_start = perf_counter()
            batch_df = self._D.features(
                instrument_list,
                batch_expressions,
                start_time=start_date,
                end_time=end_date,
            )
            print(
                "[RollingFactorPerformance] feature batch loaded: "
                f"{batch_index}/{total_batches}, factor_count={len(factor_batch)}, "
                f"elapsed={perf_counter() - batch_start:.2f}s"
            )
            if batch_df is None or batch_df.empty:
                continue

            feature_frames.append(batch_df.iloc[:, :-1])
            if return_series is None:
                return_series = batch_df.iloc[:, -1].rename(return_expression)

        if not feature_frames:
            return None
        if return_series is None:
            return pd.concat(feature_frames, axis=1)
        return pd.concat([*feature_frames, return_series], axis=1)

    def _partition_batch_compatible_factors(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        instrument_list: List[str],
        start_date: str,
        end_date: str,
        batch_size: int,
        return_expression: str,
    ) -> tuple[List[tuple[int, Dict[str, Any]]], List[tuple[int, Dict[str, Any]]]]:
        if not indexed_factors:
            return [], []

        valid_factors: List[tuple[int, Dict[str, Any]]] = []
        invalid_factors: List[tuple[int, Dict[str, Any]]] = []

        print(
            "[RollingFactorPerformance] validating factors one by one after batch failure: "
            f"count={len(indexed_factors)}"
        )
        for factor_index, factor in indexed_factors:
            expression = factor.get("qlib_expression")
            if not expression:
                invalid_factors.append((factor_index, factor))
                continue

            validate_start = perf_counter()
            try:
                factor_df = self._D.features(
                    instrument_list,
                    [expression, return_expression],
                    start_time=start_date,
                    end_time=end_date,
                )
                if factor_df is None or factor_df.empty:
                    factor["recent_performance_error"] = "Single-factor validation returned empty data."
                    invalid_factors.append((factor_index, factor))
                    print(
                        "[RollingFactorPerformance] factor validation failed: "
                        f"factor_id={factor.get('factor_id', 'unknown')}, "
                        "reason=empty_data"
                    )
                    continue

                valid_factors.append((factor_index, factor))
                print(
                    "[RollingFactorPerformance] factor validation passed: "
                    f"factor_id={factor.get('factor_id', 'unknown')}, "
                    f"elapsed={perf_counter() - validate_start:.2f}s"
                )
            except Exception as exc:
                factor["recent_performance_error"] = str(exc)
                invalid_factors.append((factor_index, factor))
                print(
                    "[RollingFactorPerformance] factor validation failed: "
                    f"factor_id={factor.get('factor_id', 'unknown')}, "
                    f"error={type(exc).__name__}: {exc}"
                )

        return valid_factors, invalid_factors

    def _should_use_directional_topk_cache(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        metric_profiles: List[str],
    ) -> bool:
        if set(metric_profiles) != {self.PROFILE_TOPK_RETURN}:
            return False
        return any(
            self._factor_expert_direction(factor) < 0 and self._base_factor_expression(factor)
            for _, factor in indexed_factors
        )

    def _build_directional_topk_load_plan(
        self,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
    ) -> Dict[str, Any] | None:
        load_factors: List[tuple[int, Dict[str, Any]]] = []
        assignments: Dict[int, Dict[str, Any]] = {}
        base_key_to_feature_name: Dict[tuple[str, bool], str] = {}

        for factor_index, factor in indexed_factors:
            base_expression = self._base_factor_expression(factor)
            if not base_expression:
                return None

            direction = self._factor_expert_direction(factor)
            needs_cs_rank = bool(factor.get("needs_cs_rank", False))
            base_key = (base_expression, needs_cs_rank)
            feature_name = base_key_to_feature_name.get(base_key)
            if feature_name is None:
                load_index = len(load_factors)
                feature_name = f"factor_{load_index}"
                base_key_to_feature_name[base_key] = feature_name
                load_factors.append(
                    (
                        load_index,
                        {
                            "factor_id": str(factor.get("base_factor_id") or factor.get("factor_id") or load_index),
                            "qlib_expression": base_expression,
                            "needs_cs_rank": needs_cs_rank,
                        },
                    )
                )

            assignments[factor_index] = {
                "feature_name": feature_name,
                "direction": direction,
            }

        if not load_factors or len(load_factors) >= len(indexed_factors):
            return None
        return {
            "load_factors": load_factors,
            "assignments": assignments,
        }

    def _cache_directional_topk_metrics_from_factor_frame(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        load_plan: Dict[str, Any],
        factor_df: pd.DataFrame,
        evaluation_dates: List[pd.Timestamp],
        benchmark_returns: Dict[pd.Timestamp, float],
        top_k: int,
    ) -> None:
        load_factors: List[tuple[int, Dict[str, Any]]] = load_plan["load_factors"]
        assignments: Dict[int, Dict[str, Any]] = load_plan["assignments"]
        feature_names = [f"factor_{offset}" for offset in range(len(load_factors))]
        factor_df.columns = feature_names + ["next_return"]
        factor_df = factor_df.sort_index()

        normalization_start = perf_counter()
        for feature_name, (_, factor) in zip(feature_names, load_factors):
            if not factor.get("needs_cs_rank", False):
                continue
            cs_rank_df = self._CSRankNorm()(factor_df[[feature_name]])
            factor_df.loc[:, feature_name] = cs_rank_df[feature_name]
        print(
            "[RollingFactorPerformance] factor normalization finished: "
            f"elapsed={perf_counter() - normalization_start:.2f}s"
        )

        compute_start = perf_counter()
        top_profile = compute_topk_return_daily_metrics(
            factor_df,
            feature_names=feature_names,
            evaluation_dates=evaluation_dates,
            benchmark_returns=benchmark_returns,
            top_k=top_k,
        )
        bottom_profile = compute_topk_return_daily_metrics(
            factor_df,
            feature_names=feature_names,
            evaluation_dates=evaluation_dates,
            benchmark_returns=benchmark_returns,
            top_k=top_k,
            score_multipliers={feature_name: -1.0 for feature_name in feature_names},
        )
        print(
            "[RollingFactorPerformance] profile computation finished: "
            f"elapsed={perf_counter() - compute_start:.2f}s, profiles=['topk_return'], "
            "directional=top_bottom"
        )

        cache = self._daily_metric_cache[self.PROFILE_TOPK_RETURN]
        for factor_index, factor in indexed_factors:
            assignment = assignments.get(factor_index)
            if not assignment:
                continue

            feature_name = str(assignment["feature_name"])
            profile_payload = bottom_profile if int(assignment["direction"]) < 0 else top_profile
            factor_cache_key = self._factor_cache_key(factor)
            for eval_ts, payload in profile_payload.get(feature_name, {}).items():
                cache[(factor_cache_key, eval_ts.strftime("%Y-%m-%d"))] = payload

    def _cache_daily_metrics_from_factor_frame(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        factor_df: pd.DataFrame,
        evaluation_dates: List[pd.Timestamp],
        benchmark_returns: Dict[pd.Timestamp, float],
        top_k: int,
        metric_profiles: List[str],
    ) -> None:
        feature_names = [f"factor_{offset}" for offset in range(len(indexed_factors))]
        factor_df.columns = feature_names + ["next_return"]
        factor_df = factor_df.sort_index()

        normalization_start = perf_counter()
        for feature_name, (_, factor) in zip(feature_names, indexed_factors):
            if not factor.get("needs_cs_rank", False):
                continue
            cs_rank_df = self._CSRankNorm()(factor_df[[feature_name]])
            factor_df.loc[:, feature_name] = cs_rank_df[feature_name]
        print(
            "[RollingFactorPerformance] factor normalization finished: "
            f"elapsed={perf_counter() - normalization_start:.2f}s"
        )

        profile_results: Dict[str, Dict[str, Dict[pd.Timestamp, Dict[str, Any]]]] = {}
        compute_start = perf_counter()
        if self.PROFILE_CROSS_SECTIONAL in metric_profiles:
            profile_results[self.PROFILE_CROSS_SECTIONAL] = compute_cross_sectional_daily_metrics(
                factor_df,
                feature_names=feature_names,
                evaluation_dates=evaluation_dates,
            )
        if self.PROFILE_TOPK_RETURN in metric_profiles:
            profile_results[self.PROFILE_TOPK_RETURN] = compute_topk_return_daily_metrics(
                factor_df,
                feature_names=feature_names,
                evaluation_dates=evaluation_dates,
                benchmark_returns=benchmark_returns,
                top_k=top_k,
            )
        print(
            "[RollingFactorPerformance] profile computation finished: "
            f"elapsed={perf_counter() - compute_start:.2f}s, profiles={list(profile_results)}"
        )

        for profile_name, profile_payload in profile_results.items():
            cache = self._daily_metric_cache[profile_name]
            for feature_name, (_, factor) in zip(feature_names, indexed_factors):
                factor_cache_key = self._factor_cache_key(factor)
                for eval_ts, payload in profile_payload.get(feature_name, {}).items():
                    cache[(factor_cache_key, eval_ts.strftime("%Y-%m-%d"))] = payload

    def _build_snapshots_from_cache(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        evaluation_dates: List[pd.Timestamp],
        metric_profiles: List[str],
    ) -> Dict[int, Dict[str, Any] | None]:
        snapshots: Dict[int, Dict[str, Any] | None] = {}
        for factor_index, factor in indexed_factors:
            combined_payload: Dict[str, Any] = {}
            has_cross_sectional = False
            has_topk_return = False

            if self.PROFILE_CROSS_SECTIONAL in metric_profiles:
                daily_metrics = self._load_cached_daily_metrics(
                    factor=factor,
                    evaluation_dates=evaluation_dates,
                    profile_name=self.PROFILE_CROSS_SECTIONAL,
                )
                cross_snapshot = build_cross_sectional_snapshot(
                    daily_metrics,
                    evaluation_dates=evaluation_dates,
                )
                if cross_snapshot:
                    combined_payload.update(cross_snapshot)
                    has_cross_sectional = True

            if self.PROFILE_TOPK_RETURN in metric_profiles:
                daily_metrics = self._load_cached_daily_metrics(
                    factor=factor,
                    evaluation_dates=evaluation_dates,
                    profile_name=self.PROFILE_TOPK_RETURN,
                )
                topk_snapshot = build_topk_return_snapshot(
                    daily_metrics,
                    evaluation_dates=evaluation_dates,
                )
                if topk_snapshot:
                    combined_payload.update(topk_snapshot)
                    has_topk_return = True

            if not combined_payload:
                snapshots[factor_index] = None
                continue

            combined_payload["recent_performance_source"] = self._build_recent_performance_source(
                has_cross_sectional=has_cross_sectional,
                has_topk_return=has_topk_return,
            )
            combined_payload["recent_performance_error"] = None
            snapshots[factor_index] = combined_payload

        return snapshots

    def _collect_pending_factors(
        self,
        *,
        indexed_factors: List[tuple[int, Dict[str, Any]]],
        evaluation_dates: List[pd.Timestamp],
        metric_profiles: List[str],
    ) -> tuple[List[tuple[int, Dict[str, Any]]], List[pd.Timestamp]]:
        pending_factors: List[tuple[int, Dict[str, Any]]] = []
        pending_dates: set[pd.Timestamp] = set()
        for factor_index, factor in indexed_factors:
            factor_cache_key = self._factor_cache_key(factor)
            factor_missing_dates = False
            for profile_name in metric_profiles:
                cache = self._daily_metric_cache[profile_name]
                for eval_date in evaluation_dates:
                    cache_key = (factor_cache_key, eval_date.strftime("%Y-%m-%d"))
                    if cache_key in cache:
                        continue
                    pending_dates.add(eval_date)
                    factor_missing_dates = True
            if factor_missing_dates:
                pending_factors.append((factor_index, factor))

        return pending_factors, sorted(pending_dates)

    def _load_cached_daily_metrics(
        self,
        *,
        factor: Dict[str, Any],
        evaluation_dates: List[pd.Timestamp],
        profile_name: str,
    ) -> Dict[pd.Timestamp, Dict[str, Any]]:
        cache = self._daily_metric_cache[profile_name]
        factor_cache_key = self._factor_cache_key(factor)
        payload: Dict[pd.Timestamp, Dict[str, Any]] = {}
        for eval_date in evaluation_dates:
            cache_value = cache.get((factor_cache_key, eval_date.strftime("%Y-%m-%d")))
            if cache_value is None:
                continue
            payload[eval_date] = cache_value
        return payload

    def _resolve_metric_profiles(self, metric_profile: str) -> List[str]:
        normalized_profile = str(metric_profile or self.PROFILE_FULL).lower()
        if normalized_profile == self.PROFILE_CROSS_SECTIONAL:
            return [self.PROFILE_CROSS_SECTIONAL]
        if normalized_profile == self.PROFILE_TOPK_RETURN:
            return [self.PROFILE_TOPK_RETURN]
        return [self.PROFILE_CROSS_SECTIONAL, self.PROFILE_TOPK_RETURN]

    def _build_recent_performance_source(
        self,
        *,
        has_cross_sectional: bool,
        has_topk_return: bool,
    ) -> str:
        if has_cross_sectional and has_topk_return:
            return "qlib_rank_ic_and_topk_excess"
        if has_cross_sectional:
            return "qlib_rank_ic"
        return "qlib_topk_excess"

    def _empty_profile_payloads(self, metric_profiles: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.PROFILE_CROSS_SECTIONAL in metric_profiles:
            payload.update(
                {
                    "recent_ic_eval_dates": [],
                    "recent_ics": [],
                    "recent_rank_ics": [],
                    "recent_ic_universe_counts": [],
                    "recent_ic_mean": 0.0,
                    "recent_rank_ic_mean": 0.0,
                    "recent_icir": 0.0,
                    "recent_rank_icir": 0.0,
                }
            )
        if self.PROFILE_TOPK_RETURN in metric_profiles:
            payload.update(
                {
                    "recent_topk_eval_dates": [],
                    "recent_topk_returns": [],
                    "recent_topk_benchmark_returns": [],
                    "recent_topk_excess_returns": [],
                    "recent_topk_universe_counts": [],
                }
            )
        return payload

    def _stale_profile_payloads(self, metric_profiles: List[str]) -> Dict[str, Any]:
        stale_profiles = [
            profile_name
            for profile_name in (self.PROFILE_CROSS_SECTIONAL, self.PROFILE_TOPK_RETURN)
            if profile_name not in metric_profiles
        ]
        if not stale_profiles:
            return {}
        return self._empty_profile_payloads(stale_profiles)

    def _base_factor_expression(self, factor: Dict[str, Any]) -> str | None:
        base_expression = str(factor.get("base_qlib_expression") or "").strip()
        if base_expression:
            return base_expression

        expression = str(factor.get("qlib_expression") or "").strip()
        if self._factor_expert_direction(factor) > 0:
            return expression or None

        return self._unwrap_negative_expression(expression)

    @staticmethod
    def _factor_expert_direction(factor: Dict[str, Any]) -> int:
        try:
            direction = int(factor.get("expert_direction", 1))
        except (TypeError, ValueError):
            expert_label = str(factor.get("expert_label") or "").lower()
            factor_id = str(factor.get("factor_id") or "").lower()
            direction = -1 if expert_label == "short" or factor_id.endswith("__short") else 1
        return -1 if direction < 0 else 1

    @staticmethod
    def _unwrap_negative_expression(expression: str) -> str | None:
        if not expression:
            return None

        import re

        match = re.match(r"^Mul\s*\(\s*-1(?:\.0)?\s*,\s*(.*)\)\s*$", expression)
        if not match:
            return None
        base_expression = match.group(1).strip()
        return base_expression or None

    def _factor_cache_key(self, factor: Dict[str, Any]) -> str:
        factor_id = factor.get("factor_id")
        if factor_id:
            return str(factor_id)
        return str(factor.get("qlib_expression", ""))

    @staticmethod
    def _benchmark_code(benchmark: str) -> str:
        mapping = {
            "csi300": "SH000300",
            "hs300": "SH000300",
            "csi500": "SH000905",
            "zz500": "SH000905",
        }
        return mapping.get(str(benchmark).lower(), benchmark)

    @staticmethod
    def _infer_label_lookahead_days(return_expression: str) -> int:
        import re

        offsets = []
        for match in re.finditer(r"Ref\s*\([^,]+,\s*(-\d+)\s*\)", return_expression or ""):
            try:
                offsets.append(abs(int(match.group(1))))
            except (TypeError, ValueError):
                continue
        return max(offsets) if offsets else 1
