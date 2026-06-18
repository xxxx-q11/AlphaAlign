from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Strategy.base.base_allocator import BaseAllocator
from Strategy.runtime.window_context import WindowContext


class MWUAllocator(BaseAllocator):
    """Learn expert weights with multiplicative-weights updates."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.15,
        reward_cap: float = 0.05,
        exploration_rate: float = 0.03,
        max_weight: float = 0.15,
    ) -> None:
        self.learning_rate = max(float(learning_rate), 0.0)
        self.reward_cap = max(float(reward_cap), 1e-6)
        self.exploration_rate = min(max(float(exploration_rate), 0.0), 1.0)
        self.max_weight = min(max(float(max_weight), 0.0), 1.0)

    def allocate(
        self,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not selected_items:
            return {"weighted_factors": [], "allocation_context": {"method": "mwu"}}

        reward_dates, reward_matrix = self._build_reward_matrix(selected_items)
        expert_count = len(selected_items)
        if expert_count <= 0:
            return {"weighted_factors": [], "allocation_context": {"method": "mwu"}}

        weights = np.full(expert_count, 1.0 / expert_count, dtype=float)
        effective_cap = max(self.max_weight, 1.0 / expert_count)
        reward_history: list[np.ndarray] = []

        for step_rewards in reward_matrix:
            reward_history.append(step_rewards)
            multiplicative_update = np.exp(np.clip(self.learning_rate * step_rewards, -50.0, 50.0))
            weights = weights * multiplicative_update
            weights = self._normalize_weights(weights)
            weights = self._mix_uniform_prior(weights)
            weights = self._apply_weight_cap(weights, cap=effective_cap)

        transformed_mean_rewards = (
            np.mean(np.vstack(reward_history), axis=0)
            if reward_history
            else np.zeros(expert_count, dtype=float)
        )
        raw_mean_rewards = np.asarray(
            [self._safe_float(item.get("recent_score", 0.0)) for item in selected_items],
            dtype=float,
        )
        weighted_factors = self._build_weighted_factors(
            selected_items=selected_items,
            weights=weights,
            transformed_mean_rewards=transformed_mean_rewards,
            raw_mean_rewards=raw_mean_rewards,
        )
        allocation_context = {
            "method": "mwu",
            "learning_rate": self.learning_rate,
            "reward_cap": self.reward_cap,
            "exploration_rate": self.exploration_rate,
            "max_weight": self.max_weight,
            "effective_max_weight": effective_cap,
            "selected_factor_count": len(weighted_factors),
            "expert_count": expert_count,
            "reward_step_count": int(reward_matrix.shape[0]),
            "reward_start": reward_dates[0] if reward_dates else None,
            "reward_end": reward_dates[-1] if reward_dates else None,
            "top_experts": [
                {
                    "factor_id": factor.get("factor_id"),
                    "weight": float(factor.get("weight", 0.0)),
                    "recent_score": float(factor.get("recent_score", 0.0)),
                    "transformed_reward_mean": float(factor.get("transformed_reward_mean", 0.0)),
                }
                for factor in sorted(weighted_factors, key=lambda item: float(item.get("weight", 0.0)), reverse=True)[:10]
            ],
        }
        return {
            "weighted_factors": weighted_factors,
            "allocation_context": allocation_context,
        }

    def _build_reward_matrix(
        self,
        selected_items: list[dict[str, Any]],
    ) -> tuple[list[str], np.ndarray]:
        date_index: list[str] = []
        date_set: set[str] = set()
        for item in selected_items:
            for raw_date in item.get("recent_eval_dates", []) or []:
                date_key = self._normalize_date_key(raw_date)
                if not date_key or date_key in date_set:
                    continue
                date_set.add(date_key)
                date_index.append(date_key)

        if date_index:
            date_index.sort()
            reward_matrix = np.zeros((len(date_index), len(selected_items)), dtype=float)
            for item_index, item in enumerate(selected_items):
                reward_by_date = self._reward_mapping(item)
                for date_offset, date_key in enumerate(date_index):
                    reward_matrix[date_offset, item_index] = reward_by_date.get(date_key, 0.0)
            return date_index, reward_matrix

        max_length = max(len(item.get("recent_series", []) or []) for item in selected_items)
        if max_length <= 0:
            return [], np.zeros((0, len(selected_items)), dtype=float)

        reward_matrix = np.zeros((max_length, len(selected_items)), dtype=float)
        for item_index, item in enumerate(selected_items):
            series = [self._transform_reward(value) for value in item.get("recent_series", []) or []]
            if not series:
                continue
            reward_matrix[max_length - len(series) :, item_index] = np.asarray(series, dtype=float)
        return [], reward_matrix

    def _reward_mapping(self, item: dict[str, Any]) -> dict[str, float]:
        reward_by_date: dict[str, float] = {}
        raw_dates = item.get("recent_eval_dates", []) or []
        raw_series = item.get("recent_series", []) or []
        for raw_date, raw_reward in zip(raw_dates, raw_series):
            date_key = self._normalize_date_key(raw_date)
            if not date_key:
                continue
            reward_by_date[date_key] = self._transform_reward(raw_reward)
        return reward_by_date

    def _transform_reward(self, reward: Any) -> float:
        raw_reward = self._safe_float(reward, 0.0)
        clipped_reward = float(np.clip(raw_reward, -self.reward_cap, self.reward_cap))
        return float(clipped_reward / self.reward_cap)

    def _mix_uniform_prior(self, weights: np.ndarray) -> np.ndarray:
        if self.exploration_rate <= 0 or weights.size <= 0:
            return weights
        uniform = np.full(weights.size, 1.0 / weights.size, dtype=float)
        mixed = (1.0 - self.exploration_rate) * weights + self.exploration_rate * uniform
        return self._normalize_weights(mixed)

    def _apply_weight_cap(self, weights: np.ndarray, *, cap: float) -> np.ndarray:
        normalized = self._normalize_weights(weights)
        if normalized.size <= 0 or cap >= 1.0:
            return normalized

        capped = normalized.copy()
        for _ in range(max(normalized.size * 2, 1)):
            over_mask = capped > cap + 1e-12
            if not np.any(over_mask):
                break
            excess = float((capped[over_mask] - cap).sum())
            capped[over_mask] = cap
            under_mask = ~over_mask
            if not np.any(under_mask):
                break
            under_total = float(capped[under_mask].sum())
            if under_total <= 1e-12:
                capped = np.full(capped.size, 1.0 / capped.size, dtype=float)
                break
            capped[under_mask] = capped[under_mask] + excess * (capped[under_mask] / under_total)
        return self._normalize_weights(capped)

    def _build_weighted_factors(
        self,
        *,
        selected_items: list[dict[str, Any]],
        weights: np.ndarray,
        transformed_mean_rewards: np.ndarray,
        raw_mean_rewards: np.ndarray,
    ) -> list[dict[str, Any]]:
        weighted_factors: list[dict[str, Any]] = []
        for item, weight, transformed_reward_mean, raw_mean_reward in zip(
            selected_items,
            weights.tolist(),
            transformed_mean_rewards.tolist(),
            raw_mean_rewards.tolist(),
        ):
            factor = item.get("factor", {})
            factor_expr = factor.get("qlib_expression")
            if not factor_expr or float(weight) <= 0:
                continue
            weighted_factors.append(
                {
                    "factor_id": factor.get("factor_id"),
                    "base_factor_id": factor.get("base_factor_id"),
                    "expert_direction": int(factor.get("expert_direction", 1)),
                    "expert_label": factor.get("expert_label"),
                    "qlib_expression": factor_expr,
                    "recent_score": float(item.get("recent_score", raw_mean_reward)),
                    "recent_series": item.get("recent_series", []),
                    "recent_eval_dates": item.get("recent_eval_dates", []),
                    "recent_score_source": item.get("recent_score_source"),
                    "is_proxy_score": bool(item.get("is_proxy_score", False)),
                    "recent_ir": float(item.get("recent_ir", 0.0)),
                    "recent_std": float(item.get("recent_std", 0.0)),
                    "raw_weight": float(weight),
                    "weight": float(weight),
                    "transformed_reward_mean": float(transformed_reward_mean),
                }
            )

        weight_sum = float(sum(item["weight"] for item in weighted_factors))
        if weight_sum > 0:
            for item in weighted_factors:
                item["weight"] = float(item["weight"] / weight_sum)
        weighted_factors.sort(key=lambda item: float(item.get("weight", 0.0)), reverse=True)
        return weighted_factors

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        nonnegative = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        weight_sum = float(nonnegative.sum())
        if weight_sum <= 1e-12:
            if nonnegative.size <= 0:
                return nonnegative
            return np.full(nonnegative.size, 1.0 / nonnegative.size, dtype=float)
        return nonnegative / weight_sum

    @staticmethod
    def _normalize_date_key(value: Any) -> str | None:
        if value is None or value == "":
            return None
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
