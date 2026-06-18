"""
Factor Selection Agent

Responsibilities:
1. Process all qualified candidate factors from a single GP round
2. Use LLM for economic rationale binary classification
3. Use correlation service for factor deduplication
4. Maintain a structured factor library
5. Generate mining feedback for the next GP round
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .prompts import FACTOR_POOL_ANALYSIS_PROMPT
from .services.factor_correlation_service import FactorCorrelationService
from .services.factor_library_manager import FactorLibraryManager
from .services.factor_metrics_service import FactorMetricsService


class FactorSelectionAgent:
    """Refactored factor selection and factor library construction agent."""

    def __init__(self, llm_service=None) -> None:
        self.llm = llm_service
        self.library_manager = FactorLibraryManager()
        self.metrics_service = FactorMetricsService()
        self.correlation_service = FactorCorrelationService()

    def process(
        self,
        candidates: List[Dict[str, Any]],
        factor_library: Optional[List[Dict[str, Any]]] = None,
        mining_iteration: int = 0,
        selection_config: Optional[Dict[str, Any]] = None,
        max_mining_rounds: int = 10,
    ) -> Dict[str, Any]:
        """Execute a single round of candidate factor filtering, deduplication, and library ingestion."""
        config = {
            "train_ic_threshold": 0.02,
            "correlation_threshold": 0.90,
            "target_library_size": 50,
            "max_accept_per_round": 5,
        }
        if selection_config:
            config.update(selection_config)

        logs: List[str] = []
        factor_library = deepcopy(factor_library) if factor_library else self.library_manager.load_factor_library()
        accepted_candidates: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []

        filtered_candidates = self._filter_candidates(
            candidates,
            config["train_ic_threshold"],
        )
        logs.append(f"[FactorSelection] Qualified candidate count for current round: {len(filtered_candidates)}")
        target_library_size = int(config["target_library_size"])

        for candidate in filtered_candidates:
            remaining_slots = max(target_library_size - len(factor_library), 0)
            if remaining_slots <= 0:
                skipped_factor = deepcopy(candidate)
                skipped_factor["rejection_reason"] = (
                    f"Factor library has reached target size ({target_library_size}); subsequent candidates not ingested"
                )
                rejected_candidates.append(skipped_factor)
                logs.append(
                    "[FactorSelection] Factor library has reached target size; candidate skipped and passed to next stage: "
                    f"{skipped_factor.get('qlib_expression')}"
                )
                continue

            if len(accepted_candidates) >= int(config["max_accept_per_round"]):
                skipped_factor = deepcopy(candidate)
                skipped_factor["rejection_reason"] = (
                    f"Reached single-round ingestion cap ({int(config['max_accept_per_round'])}); no more factors accepted this round"
                )
                rejected_candidates.append(skipped_factor)
                logs.append(
                    "[FactorSelection] Reached single-round ingestion cap; candidate skipped and rejected: "
                    f"{skipped_factor.get('qlib_expression')}"
                )
                continue

            decision, updated_library, decision_log = self._evaluate_candidate(
                candidate=candidate,
                factor_library=factor_library,
                correlation_threshold=float(config["correlation_threshold"]),
            )
            logs.extend(decision_log)
            factor_library = updated_library

            if decision.get("status") == "accepted":
                accepted_candidates.append(decision["factor"])
            else:
                rejected_candidates.append(decision["factor"])

        # After each round, uniformly persist the factor library and metrics.
        self.library_manager.save_factor_library(factor_library)
        self.library_manager.save_factor_metrics(factor_library)

        mining_feedback = self._build_mining_feedback(
            factor_library=factor_library,
            accepted_candidates=accepted_candidates,
            mining_iteration=mining_iteration,
            target_library_size=target_library_size,
        )
        feedback_path = self.library_manager.save_mining_feedback(mining_iteration, mining_feedback)

        should_continue_mining = len(factor_library) < target_library_size and mining_iteration < max_mining_rounds

        selection_summary = {
            "round_index": mining_iteration,
            "accepted_count": len(accepted_candidates),
            "rejected_count": len(rejected_candidates),
            "max_accept_per_round": int(config["max_accept_per_round"]),
            "factor_library_size": len(factor_library),
            "target_library_size": target_library_size,
            "should_continue_mining": should_continue_mining,
            "feedback_path": feedback_path,
        }
        logs.append(f"[FactorSelection] Factor library current size: {len(factor_library)} / {target_library_size}")

        return {
            "status": "success",
            "logs": logs,
            "factor_library": factor_library,
            "selected_candidates": accepted_candidates,
            "rejected_candidates": rejected_candidates,
            "selection_summary": selection_summary,
            "mining_feedback": mining_feedback,
            "should_continue_mining": should_continue_mining,
        }

    def _filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        train_ic_threshold: float,
    ) -> List[Dict[str, Any]]:
        """First filter by validity and training-set IC, then sort by validation-set Rank IC strength."""
        filtered = []
        for candidate in candidates:
            if not candidate.get("is_valid", False):
                continue
            train_ic = self._safe_float(candidate.get("train_ic"), 0.0)
            if train_ic <= train_ic_threshold:
                continue

            candidate = self.metrics_service.ensure_metrics(candidate)
            filtered.append(candidate)

        filtered.sort(
            key=lambda item: self.metrics_service.get_conflict_score(item),
            reverse=True,
        )
        return filtered

    def _evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        factor_library: List[Dict[str, Any]],
        correlation_threshold: float,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
        """Perform economic rationale screening and correlation screening for a single candidate factor."""
        logs: List[str] = []
        factor = deepcopy(candidate)
        expression = factor.get("qlib_expression")
        factor_id = factor.get("factor_id") or "-"
        print(
            "[FactorSelection] Begin processing candidate factor: "
            f"factor_id={factor_id}, expression={expression}",
            flush=True,
        )
        factor["economics_passed"] = False
        factor["economics_reason"] = ""
        qlib_validation = self.correlation_service.validate_candidate_for_library_entry(factor)
        factor["qlib_validation_passed"] = bool(qlib_validation.get("is_calculable", False))
        factor["qlib_validation_reason"] = str(qlib_validation.get("reason", "") or "")

        if not factor["qlib_validation_passed"]:
            rejection_reason = (
                "Transformed factor expression does not meet qlib computability requirements; rejected: "
                f"{factor['qlib_validation_reason'] or 'Unknown reason'}"
            )
            factor["invalid_reason"] = factor["qlib_validation_reason"] or factor.get("invalid_reason")
            factor["rejection_reason"] = rejection_reason
            logs.append(f"[FactorSelection] Factor filtered out by qlib computability check: {expression}; {rejection_reason}")
            return {"status": "rejected", "factor": factor}, factor_library, logs

        economics_result = self._judge_economic_rationale(factor)
        factor["economics_passed"] = economics_result["is_explainable"]
        factor["economics_reason"] = economics_result["reason"]

        if not economics_result["is_explainable"]:
            factor["rejection_reason"] = f"Failed economic rationale check: {factor['economics_reason']}"
            logs.append(f"[FactorSelection] Factor filtered out by economic rationale check: {expression}")
            return {"status": "rejected", "factor": factor}, factor_library, logs

        high_corr_matches = self.correlation_service.find_high_correlation_matches(
            candidate=factor,
            factor_library=factor_library,
            threshold=correlation_threshold,
        )

        if not high_corr_matches:
            logs.append(f"[FactorSelection] Factor passed correlation check and ingested into library: {expression}")
            print(
                "[FactorSelection][Decision] Factor passed correlation check and ingested into library: "
                f"factor_id={factor_id}, expression={expression}",
                flush=True,
            )
            return {"status": "accepted", "factor": factor}, factor_library + [factor], logs

        best_existing_match = max(
            high_corr_matches,
            key=lambda item: self.metrics_service.get_conflict_score(item["factor"]),
        )
        existing_factor = best_existing_match["factor"]
        current_score = self.metrics_service.get_conflict_score(factor)
        existing_score = self.metrics_service.get_conflict_score(existing_factor)

        if current_score > existing_score:
            logs.append(
                "[FactorSelection] Factor is highly correlated with a library entry, but current factor has better Rank IC/IC; performing replacement: "
                f"{existing_factor.get('qlib_expression')} -> {expression}"
            )
            print(
                "[FactorSelection][Decision] High-correlation factor replacement: "
                f"factor_id={factor_id}, replaced_factor_id={existing_factor.get('factor_id')}, "
                f"correlation={best_existing_match['correlation']:.4f}",
                flush=True,
            )
            updated_library = [
                item for item in factor_library if item.get("factor_id") != existing_factor.get("factor_id")
            ]
            factor["replaced_factor_id"] = existing_factor.get("factor_id")
            factor["max_correlation"] = best_existing_match["correlation"]
            return {"status": "accepted", "factor": factor}, updated_library + [factor], logs

        logs.append(
            "[FactorSelection] Factor is highly correlated with a library entry and has a lower score; rejected: "
            f"{expression}"
        )
        print(
            "[FactorSelection][Decision] High correlation with lower score; rejected: "
            f"factor_id={factor_id}, conflicted_factor_id={existing_factor.get('factor_id')}, "
            f"correlation={best_existing_match['correlation']:.4f}",
            flush=True,
        )
        factor["max_correlation"] = best_existing_match["correlation"]
        factor["conflicted_factor_id"] = existing_factor.get("factor_id")
        factor["rejection_reason"] = (
            f"Highly correlated with a library entry and has a lower score: conflicted_factor_id={factor['conflicted_factor_id']}"
        )
        return {"status": "rejected", "factor": factor}, factor_library, logs

    def _judge_economic_rationale(self, factor: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to perform binary classification on factor economic explainability."""
        expression = factor.get("qlib_expression") or factor.get("gp_expression") or ""
        factor_id = factor.get("factor_id") or "-"

        print(f"[FactorSelection][LLM] Begin evaluating factor: factor_id={factor_id}, expression={expression}")

        if not self.llm:
            # When no LLM is configured, treat as non-explainable but preserve the reason to avoid silently passing through.
            print(f"[FactorSelection][LLM] Evaluation skipped: factor_id={factor_id}, reason=LLM not configured")
            return {
                "is_explainable": False,
                "reason": "LLM not configured; defaulting to non-explainable",
            }

#         prompt = f"""
# You are a quantitative researcher. Judge whether the following factor has a reasonable economic or financial explanation, or whether it is the product of overfitting.
# The factor uses qlib format.
# Factor expression:
# {expression}

# Please return strictly as JSON, do not output any other content:
# {{
#     "reason": "brief explanation of whether this factor has an economic rationale",
#     "is_explainable": true or false, based on your binary classification judgment
# }}
# """
        prompt = f"""
You are a quantitative researcher. Judge whether the following factor has a reasonable economic or financial explanation, or whether it is the product of overfitting.

The factor uses qlib expression format. Note the following operator semantics:

1. Basic fields
- $open: opening price
- $close: closing price
- $high: highest price
- $low: lowest price
- $volume: trading volume
- $vwap: volume-weighted average price

2. Unary operators
- Abs(x): absolute value
- Sign(x): sign function
- Log(x): natural logarithm

3. Binary operators
- Add(x, y): x + y
- Sub(x, y): x - y
- Mul(x, y): x * y
- Div(x, y): x / y
- Power(x, y): x raised to the power of y
- Greater(x, y): comparison result of x > y
- Less(x, y): comparison result of x < y

4. Time-series rolling operators
Note: the second parameter N of the following operators represents the rolling window size, i.e., the window of past N periods, not a constant threshold.
- Ref(x, N): x from N periods ago
- Mean(x, N): mean over the past N periods
- Sum(x, N): sum over the past N periods
- Std(x, N): standard deviation over the past N periods
- Var(x, N): variance over the past N periods
- Skew(x, N): skewness over the past N periods
- Kurt(x, N): kurtosis over the past N periods
- Max(x, N): maximum over the past N periods
- Min(x, N): minimum over the past N periods
- Med(x, N): median over the past N periods
- Mad(x, N): mean absolute deviation over the past N periods
- Rank(x, N): time-series rank of the current value within the past N period window
- Delta(x, N): change in current value relative to N periods ago
- WMA(x, N): weighted moving average over the past N periods
- EMA(x, N): exponential moving average over the past N periods

5. Paired rolling operators
- Cov(x, y, N): covariance of x and y over the past N periods
- Corr(x, y, N): correlation coefficient of x and y over the past N periods

6. Common structures expanded from GP operators
- Div(Mean(x, N), Std(x, N)): analogous to mean / standard deviation over the past N periods, interpretable as time-series stability or risk-adjusted strength
- Sub(Max(x, N), Min(x, N)): range width over the past N periods, representing amplitude, volatility range, or trend swing magnitude
- Sub(x, Max(x, N)): distance of current value from the maximum over the past N periods
- Sub(x, Min(x, N)): distance of current value from the minimum over the past N periods
- Div(x, Mean(x, N)): ratio of current value to the mean over the past N periods
- Div(Sub(x, Ref(x, N-1)), Ref(x, N-1)): rate of change over approximately the past N periods

Judgment criteria:
- Do not interpret the window parameter N of rolling operators as a fixed threshold.
- For example, Max(EMA($close, 10), 40) means "the maximum of EMA($close, 10) over the past 40 periods", not "the larger of EMA($close, 10) and the constant 40".
- If the factor corresponds to common financial concepts such as momentum, reversal, volatility, volume/liquidity, price-volume relationship, trend strength, range position, or risk-adjusted return, it can be considered to have some degree of explainability.
- If the factor is primarily composed of complex nesting, arbitrary constants, comparisons lacking financial meaning, or non-intuitive combinations, lean toward judging it as overfitting.
- If the factor uses raw price level differences, note that it may be affected by stock price scale and have weak cross-stock comparability.

Factor expression:
{expression}

Please return strictly as JSON, do not output any other content:
{{
    "reason": "brief explanation of whether this factor has an economic rationale",
    "is_explainable": true or false
}}
"""
        try:
            response = self.llm.call(prompt=prompt, stream=False)
            parsed = self.llm.parse_json_response(response)
            result = {
                "is_explainable": bool(parsed.get("is_explainable", False)),
                "reason": str(parsed.get("reason", "LLM did not return a reason")),
            }
            # print(
            #     "[FactorSelection][LLM] Evaluation result: "
            #     f"factor_id={factor_id}, is_explainable={result['is_explainable']}, "
            #     f"reason={result['reason']}"
            # )
            return result
        except Exception as exc:
            # When LLM fails, do not interrupt the pipeline, but treat as non-explainable and record the reason.
            print(f"[FactorSelection][LLM] Evaluation failed: factor_id={factor_id}, error={exc}")
            return {
                "is_explainable": False,
                "reason": f"LLM judgment failed; defaulting to non-explainable: {exc}",
            }

    def _build_mining_feedback(
        self,
        factor_library: List[Dict[str, Any]],
        accepted_candidates: List[Dict[str, Any]],
        mining_iteration: int,
        target_library_size: int,
    ) -> Dict[str, Any]:
        """Build feedback for the next GP mining round."""
        print(
            "[FactorSelection][Feedback] Begin building next-round mining feedback: "
            f"iteration={mining_iteration}, library_size={len(factor_library)}, "
            f"accepted_count={len(accepted_candidates)}",
            flush=True,
        )
        pool_report = self._build_pool_report(factor_library, accepted_candidates, target_library_size)
        llm_result = self._analyze_pool_with_llm(pool_report)

        return {
            "iteration": mining_iteration + 1,
            "current_library_size": len(factor_library),
            "target_library_size": target_library_size,
            "pool_report": pool_report,
            "pool_weaknesses": llm_result.get("pool_weaknesses", []),
            "suggested_directions": llm_result.get("suggested_directions", []),
            "suggested_seeds": llm_result.get("suggested_seeds", []),
            "gp_strategy_hints": llm_result.get(
                "gp_strategy_hints",
                {
                    "preferred_operators": ["TsCorr", "TsStd", "TsMean"],
                    "preferred_features": ["volume", "vwap", "high"],
                    "preferred_windows": [10, 20, 40],
                    "avoid_patterns": [],
                },
            ),
            "convergence_info": {
                "current_pool_size": len(factor_library),
                "is_target_reached": len(factor_library) >= target_library_size,
            },
        }

    def _build_pool_report(
        self,
        factor_library: List[Dict[str, Any]],
        accepted_candidates: List[Dict[str, Any]],
        target_library_size: int,
    ) -> str:
        """Build a factor library report for the next-round LLM analysis."""
        lines = [
            f"Current factor library size: {len(factor_library)} / {target_library_size}",
            "Current factor library expressions and economic rationales:",
        ]
        for index, factor in enumerate(factor_library[:80], start=1):
            lines.extend(self._format_factor_for_pool_report(index, factor))

        lines.append("")
        lines.append("Newly accepted factors this round and their economic rationales:")
        if accepted_candidates:
            for index, factor in enumerate(accepted_candidates, start=1):
                lines.extend(self._format_factor_for_pool_report(index, factor))
        else:
            lines.append("No new factors entered the library this round.")

        return "\n".join(lines)

    @staticmethod
    def _format_factor_for_pool_report(index: int, factor: Dict[str, Any]) -> List[str]:
        """Format factor expression and economic rationale for the next-round LLM analysis."""
        expression = factor.get("qlib_expression") or factor.get("gp_expression") or ""
        economics_reason = str(factor.get("economics_reason") or "").strip()
        if not economics_reason:
            economics_reason = "No economic rationale recorded"

        return [
            f"{index}. Expression: {expression}",
            f"   Economic rationale: {economics_reason}",
        ]

    def _analyze_pool_with_llm(self, pool_report: str) -> Dict[str, Any]:
        """Call LLM to generate next-round seed factors and operator suggestions."""
        if not self.llm:
            return {}

        prompt = FACTOR_POOL_ANALYSIS_PROMPT.format(pool_report=pool_report)
        try:
            print("[FactorSelection][Feedback] Starting LLM analysis of factor library", flush=True)
            response = self.llm.call(prompt=prompt, stream=False)
            parsed = self.llm.parse_json_response(response)
            print("[FactorSelection][Feedback] Completed LLM analysis of factor library", flush=True)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            print(f"[FactorSelection][Feedback] LLM analysis of factor library failed: {exc}", flush=True)
            return {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert to float."""
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
