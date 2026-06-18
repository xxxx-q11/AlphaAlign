from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.data import D

from Strategy.runtime.window_context import WindowContext


ROBUST_ZSCORE_EPS = 1e-12
FEATURE_PROCESSORS = [
    "RobustZScoreNorm(fields_group=feature, clip_outlier=True)",
    "Fillna(fields_group=feature, fill_value=0)",
]
LABEL_PROCESSORS = [
    "DropnaLabel(fields_group=label)",
    "CSRankNorm(fields_group=label)",
]


@dataclass(slots=True)
class WindowDatasetSplit:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


class LGBMWindowTrainer:
    """Train one LightGBM model per rebalance window and return window predictions."""

    def __init__(
        self,
        *,
        train_window_days: int = 120,
        label_horizon_days: int = 1,
        label_expression: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.train_window_days = max(int(train_window_days), 1)
        self.label_horizon_days = max(int(label_horizon_days), 1)
        self.label_expression = str(label_expression).strip() if label_expression else None
        self.model_kwargs = dict(model_kwargs or {})
        self.fit_kwargs = dict(fit_kwargs or {})
        self._provider_uri: str | None = None
        self._qlib_initialized = False

    def train_and_predict(
        self,
        *,
        selected_items: list[dict[str, Any]],
        context: WindowContext,
    ) -> dict[str, Any]:
        if not selected_items:
            return self._empty_result(
                fit_status="no_selected_items",
                selected_factor_count=0,
            )

        valid_items = [item for item in selected_items if item.get("factor", {}).get("qlib_expression")]
        if not valid_items:
            return self._empty_result(
                fit_status="no_factor_expression",
                selected_factor_count=0,
            )

        self._ensure_qlib_initialized(str(context.provider_uri))
        split = self._build_dataset_split(context)
        feature_names = [f"factor_{index}" for index, _ in enumerate(valid_items)]
        label_expression = self._resolve_label_expression()

        frame = self._load_feature_frame(
            valid_items=valid_items,
            instrument=context.instrument,
            split=split,
            label_expression=label_expression,
        )
        prepared = self._prepare_dataset(
            frame=frame,
            feature_names=feature_names,
            split=split,
        )
        if prepared is None:
            return self._empty_result(
                fit_status="insufficient_training_samples",
                selected_factor_count=len(valid_items),
                split=split,
                label_expression=label_expression,
            )

        train_frame, valid_frame, test_frame = prepared
        if train_frame.empty or valid_frame.empty or test_frame.empty:
            return self._empty_result(
                fit_status="empty_train_valid_or_test_frame",
                selected_factor_count=len(valid_items),
                split=split,
                label_expression=label_expression,
            )

        model = LGBModel(**self.model_kwargs)
        dataset = _FrameDataset(train_frame=train_frame, valid_frame=valid_frame, test_frame=test_frame)
        evals_result: dict[str, Any] = {}
        model.fit(dataset, evals_result=evals_result, **self.fit_kwargs)
        prediction = model.predict(dataset, segment="test").rename("score").dropna()
        if prediction.empty:
            return self._empty_result(
                fit_status="empty_prediction",
                selected_factor_count=len(valid_items),
                split=split,
                label_expression=label_expression,
            )

        signal = prediction.to_frame("score")
        if isinstance(signal.index, pd.MultiIndex):
            index_names = list(signal.index.names)
            if "datetime" in index_names and "instrument" in index_names:
                signal = signal.reorder_levels(["datetime", "instrument"]).sort_index()
            else:
                signal.index = signal.index.set_names(["instrument", "datetime"])
                signal = signal.reorder_levels(["datetime", "instrument"]).sort_index()
        else:
            signal.index = signal.index.rename("datetime")
            signal = signal.sort_index()
        generation_context = {
            "status": "success",
            "model_class": "LGBModel",
            "selected_factor_count": len(valid_items),
            "selected_factor_ids": [item.get("factor", {}).get("factor_id") for item in valid_items],
            "train_sample_count": int(len(train_frame)),
            "valid_sample_count": int(len(valid_frame)),
            "test_sample_count": int(len(test_frame)),
            "prediction_rows": int(len(signal)),
            "train_start": split.train_start,
            "train_end": split.train_end,
            "valid_start": split.valid_start,
            "valid_end": split.valid_end,
            "test_start": split.test_start,
            "test_end": split.test_end,
            "label_expression": label_expression,
            "label_horizon_days": self.label_horizon_days,
            "train_window_days": self.train_window_days,
            "feature_processors": FEATURE_PROCESSORS,
            "label_processors": LABEL_PROCESSORS,
            "provider_uri": self._provider_uri,
            "feature_names": feature_names,
            "best_iteration": getattr(model.model, "best_iteration", None),
            "evals_result": evals_result,
        }
        return {
            "signal": signal,
            "generation_context": generation_context,
        }

    def _ensure_qlib_initialized(self, provider_uri: str) -> None:
        if self._qlib_initialized and self._provider_uri == provider_uri:
            return
        try:
            D.calendar(start_time="2005-01-01", end_time="2005-01-10")
            self._provider_uri = provider_uri
            self._qlib_initialized = True
            return
        except Exception:
            pass
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        self._provider_uri = provider_uri
        self._qlib_initialized = True

    def _build_dataset_split(self, context: WindowContext) -> WindowDatasetSplit:
        selection_ts = pd.Timestamp(context.selection_date)
        label_lookahead_days = self.label_horizon_days + 1
        calendar_start = (
            selection_ts
            - pd.Timedelta(days=max((self.train_window_days + context.window_days + label_lookahead_days) * 12, 240))
        ).strftime("%Y-%m-%d")
        calendar_end = (
            selection_ts + pd.Timedelta(days=max(int(context.rebalance_window_days) * 12, 120))
        ).strftime("%Y-%m-%d")
        calendar = D.calendar(start_time=calendar_start, end_time=calendar_end)
        all_dates = [pd.Timestamp(value) for value in calendar]
        selection_pos = all_dates.index(selection_ts)
        valid_end_pos = selection_pos - label_lookahead_days
        if valid_end_pos < 0:
            raise ValueError("Insufficient trading dates before validation window.")
        valid_start_pos = max(valid_end_pos - int(context.window_days) + 1, 0)
        train_end_pos = valid_start_pos - 1
        if train_end_pos < 0:
            raise ValueError("Insufficient trading dates before validation window.")
        train_start_pos = max(train_end_pos - self.train_window_days + 1, 0)
        signal_end_ts = pd.Timestamp(context.window_end)
        test_start_pos = selection_pos
        try:
            test_end_pos = all_dates.index(signal_end_ts)
        except ValueError as exc:
            raise ValueError("Failed to locate signal window end in trading calendar.") from exc
        if test_end_pos < test_start_pos:
            raise ValueError("Signal window end must not be earlier than selection date.")
        return WindowDatasetSplit(
            train_start=all_dates[train_start_pos].strftime("%Y-%m-%d"),
            train_end=all_dates[train_end_pos].strftime("%Y-%m-%d"),
            valid_start=all_dates[valid_start_pos].strftime("%Y-%m-%d"),
            valid_end=all_dates[valid_end_pos].strftime("%Y-%m-%d"),
            test_start=all_dates[test_start_pos].strftime("%Y-%m-%d"),
            test_end=all_dates[test_end_pos].strftime("%Y-%m-%d"),
        )

    def _load_feature_frame(
        self,
        *,
        valid_items: list[dict[str, Any]],
        instrument: str,
        split: WindowDatasetSplit,
        label_expression: str,
    ) -> pd.DataFrame:
        expressions = [item["factor"]["qlib_expression"] for item in valid_items] + [label_expression]
        return D.features(
            D.instruments(instrument),
            expressions,
            start_time=split.train_start,
            end_time=split.test_end,
        )

    def _prepare_dataset(
        self,
        *,
        frame: pd.DataFrame,
        feature_names: list[str],
        split: WindowDatasetSplit,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        if frame is None or frame.empty:
            return None

        prepared = frame.copy()
        prepared.columns = feature_names + ["label"]
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        date_level = self._date_level(prepared)

        train_frame = self._slice_by_date(prepared, split.train_start, split.train_end, date_level)
        valid_frame = self._slice_by_date(prepared, split.valid_start, split.valid_end, date_level)
        test_frame = self._slice_by_date(prepared, split.test_start, split.test_end, date_level)

        train_frame, valid_frame, test_frame = self._apply_feature_processors(
            train_frame=train_frame.copy(),
            valid_frame=valid_frame.copy(),
            test_frame=test_frame.copy(),
            feature_names=feature_names,
        )

        train_frame = self._apply_label_processors(train_frame, date_level).sort_index()
        valid_frame = self._apply_label_processors(valid_frame, date_level).sort_index()
        test_frame = test_frame.sort_index()
        if test_frame.empty:
            return None
        test_frame = test_frame.copy()
        test_frame["label"] = test_frame["label"].fillna(0.0)
        return train_frame, valid_frame, test_frame

    def _apply_feature_processors(
        self,
        *,
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        feature_names: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Apply RobustZScoreNorm and Fillna semantics to feature columns."""
        train_features = train_frame[feature_names].to_numpy(dtype=float, copy=True)
        median_train = np.nanmedian(train_features, axis=0)
        mad_train = np.nanmedian(np.abs(train_features - median_train), axis=0)
        std_train = (mad_train + ROBUST_ZSCORE_EPS) * 1.4826

        def transform(frame: pd.DataFrame) -> pd.DataFrame:
            transformed = frame.copy()
            features = transformed[feature_names].to_numpy(dtype=float, copy=True)
            features = (features - median_train) / std_train
            features = np.clip(features, -3, 3)
            transformed.loc[:, feature_names] = features.astype(np.float32)
            transformed.loc[:, feature_names] = transformed.loc[:, feature_names].fillna(0.0)
            return transformed

        return transform(train_frame), transform(valid_frame), transform(test_frame)

    def _apply_label_processors(self, frame: pd.DataFrame, date_level: str | int) -> pd.DataFrame:
        """Apply DropnaLabel and CSRankNorm semantics to the label column."""
        processed = frame.dropna(subset=["label"]).copy()
        if processed.empty:
            return processed
        ranked_label = processed.groupby(level=date_level)["label"].rank(pct=True).sub(0.5).mul(3.46)
        processed.loc[:, "label"] = ranked_label.astype(np.float32)
        return processed

    def _slice_by_date(
        self,
        frame: pd.DataFrame,
        start_date: str,
        end_date: str,
        date_level: str | int,
    ) -> pd.DataFrame:
        dates = pd.to_datetime(frame.index.get_level_values(date_level))
        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        return frame[mask]

    @staticmethod
    def _build_label_expression(horizon_days: int) -> str:
        return f"Ref($close, -{int(horizon_days) + 1})/Ref($close, -1) - 1"

    def _resolve_label_expression(self) -> str:
        return self.label_expression or self._build_label_expression(self.label_horizon_days)

    @staticmethod
    def _date_level(frame: pd.DataFrame) -> str | int:
        index_names = list(frame.index.names)
        if "datetime" in index_names:
            return "datetime"
        if index_names and index_names[0] is not None:
            return index_names[0]
        return 0

    def _empty_result(
        self,
        *,
        fit_status: str,
        selected_factor_count: int,
        split: WindowDatasetSplit | None = None,
        label_expression: str | None = None,
    ) -> dict[str, Any]:
        generation_context = {
            "status": fit_status,
            "model_class": "LGBModel",
            "selected_factor_count": selected_factor_count,
            "label_horizon_days": self.label_horizon_days,
            "train_window_days": self.train_window_days,
            "feature_processors": FEATURE_PROCESSORS,
            "label_processors": LABEL_PROCESSORS,
            "provider_uri": self._provider_uri,
        }
        if split is not None:
            generation_context.update(
                {
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "valid_start": split.valid_start,
                    "valid_end": split.valid_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                }
            )
        if label_expression is not None:
            generation_context["label_expression"] = label_expression
        return {
            "signal": None,
            "generation_context": generation_context,
        }


class _FrameDataset:
    """Minimal DatasetH-compatible wrapper over prepared panel frames."""

    def __init__(self, *, train_frame: pd.DataFrame, valid_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
        self.frames = {
            "train": train_frame,
            "valid": valid_frame,
            "test": test_frame,
        }
        self.segments = {key: key for key in self.frames}

    def prepare(self, segment, col_set="feature", data_key=None):  # noqa: ANN001,ARG002
        if isinstance(segment, list):
            return tuple(self.prepare(item, col_set=col_set, data_key=data_key) for item in segment)
        frame = self.frames[str(segment)]
        if col_set == "feature":
            return frame.drop(columns=["label"])
        if col_set == ["feature", "label"]:
            feature_frame = frame.drop(columns=["label"]).copy()
            label_frame = frame[["label"]].copy()
            feature_frame.columns = pd.MultiIndex.from_product([["feature"], feature_frame.columns])
            label_frame.columns = pd.MultiIndex.from_product([["label"], label_frame.columns])
            return pd.concat([feature_frame, label_frame], axis=1)
        raise ValueError(f"Unsupported col_set={col_set!r}")
