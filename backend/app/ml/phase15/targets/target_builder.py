"""
Phase 15 forecasting and execution-aligned target generation.

Research-only module.

Forecast targets:
    Close[t+h] / Close[t] - 1

Execution targets:
    Close[t+h] / Open[t+1] - 1

Horizons are measured in trading sessions, not calendar days.

Future prices are labels only. They must never be included in model inputs.
"""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.feature_builder import (
    PHASE15_FEATURE_COLUMNS,
)


DEFAULT_HORIZONS = (1, 3, 7)


class TargetBuilderError(ValueError):
    """Raised when target-generation inputs are invalid."""


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize forecast horizons."""

    if isinstance(horizons, (str, bytes)):
        raise TargetBuilderError("Horizons must be a sequence of integers.")

    try:
        values = list(horizons)
    except TypeError as exc:
        raise TargetBuilderError(
            "Horizons must be an iterable of positive integers."
        ) from exc

    if not values:
        raise TargetBuilderError("At least one horizon is required.")

    normalized = []

    for value in values:
        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise TargetBuilderError(
                f"Invalid horizon: {value!r}. Use positive integers."
            )

        value = int(value)

        if value <= 0:
            raise TargetBuilderError(
                "All horizons must be positive."
            )

        normalized.append(value)

    if len(set(normalized)) != len(normalized):
        raise TargetBuilderError("Duplicate horizons are not allowed.")

    return tuple(sorted(normalized))


def _validate_stock_rows(
    stock_data: Dict[str, Any],
) -> list[dict]:
    """Validate stock sessions and the OHLC prices required for targets."""

    if not isinstance(stock_data, dict):
        raise TargetBuilderError(
            "stock_data must be a normalized dictionary."
        )

    rows = stock_data.get("rows")

    if not isinstance(rows, list) or len(rows) < 2:
        raise TargetBuilderError(
            "At least two historical candles are required."
        )

    normalized_rows = []
    previous_date = None

    for index, row in enumerate(rows):

        if not isinstance(row, dict):
            raise TargetBuilderError(
                f"Invalid candle structure at index {index}."
            )

        raw_date = row.get("session_date")

        try:
            session_date = date.fromisoformat(str(raw_date))
        except (TypeError, ValueError) as exc:
            raise TargetBuilderError(
                f"Invalid session date at index {index}: {raw_date!r}"
            ) from exc

        if session_date.isoformat() != str(raw_date):
            raise TargetBuilderError(
                f"Session date is not canonical ISO format: {raw_date!r}"
            )

        if previous_date is not None:
            if session_date <= previous_date:
                raise TargetBuilderError(
                    "Stock sessions must be unique and sorted oldest to newest."
                )

        previous_date = session_date

        try:
            open_price = float(row["open"])
            close_price = float(row["close"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TargetBuilderError(
                f"Invalid open or close price at {session_date}."
            ) from exc

        if not isfinite(open_price) or not isfinite(close_price):
            raise TargetBuilderError(
                f"Non-finite price at {session_date}."
            )

        if open_price <= 0 or close_price <= 0:
            raise TargetBuilderError(
                f"Non-positive price at {session_date}."
            )

        normalized_rows.append({
            "session_date": session_date.isoformat(),
            "open": open_price,
            "close": close_price,
        })

    return normalized_rows


def build_return_targets(
    stock_data: Dict[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Build close-to-close and next-open-to-close return targets.

    One output row is created for every historical forecast origin.

    The final rows have unavailable future targets represented by NaN.
    They are not silently removed.

    Returns:
        session_date
        base_close
        next_session_open
        next_session_date
        close_to_close_return_{h}s
        next_open_to_close_return_{h}s
        target_close_{h}s
        target_exit_session_{h}s
        max_horizon_target_session
        all_targets_available
    """

    selected_horizons = _validate_horizons(horizons)
    rows = _validate_stock_rows(stock_data)

    opens = np.asarray(
        [row["open"] for row in rows],
        dtype=np.float64,
    )

    closes = np.asarray(
        [row["close"] for row in rows],
        dtype=np.float64,
    )

    dates = [row["session_date"] for row in rows]

    count = len(rows)
    max_horizon = max(selected_horizons)

    output = []

    for base_idx in range(count):

        entry_idx = base_idx + 1

        has_entry = entry_idx < count

        record = {
            "session_date": dates[base_idx],
            "base_close": float(closes[base_idx]),
            "next_session_open": (
                float(opens[entry_idx]) if has_entry else np.nan
            ),
            "next_session_date": (
                dates[entry_idx] if has_entry else None
            ),
            "max_horizon_target_session": (
                dates[base_idx + max_horizon]
                if base_idx + max_horizon < count
                else None
            ),
        }

        all_available = True

        for horizon in selected_horizons:

            exit_idx = base_idx + horizon

            if exit_idx >= count or not has_entry:
                forecast_return = np.nan
                execution_return = np.nan
                target_close = np.nan
                target_session = None

                all_available = False

            else:
                target_close = float(closes[exit_idx])

                forecast_return = (
                    target_close / closes[base_idx] - 1.0
                )

                execution_return = (
                    target_close / opens[entry_idx] - 1.0
                )

                target_session = dates[exit_idx]

            record[f"close_to_close_return_{horizon}s"] = (
                forecast_return
            )

            record[f"next_open_to_close_return_{horizon}s"] = (
                execution_return
            )

            record[f"target_close_{horizon}s"] = target_close
            record[f"target_exit_session_{horizon}s"] = target_session

        record["all_targets_available"] = all_available

        output.append(record)

    return pd.DataFrame(output)


def build_phase15_training_frame(
    features: pd.DataFrame,
    stock_data: Dict[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Combine the 31 input features and future-return labels.

    Feature dates and target dates must match exactly.

    Warm-up feature rows and unavailable future-target rows remain present.
    The training_eligible column identifies rows where all requested
    features and all forecast/execution labels are available.

    IMPORTANT:
        Never pass the entire returned DataFrame directly as model X.
        Model inputs must be selected exclusively from
        PHASE15_FEATURE_COLUMNS.
    """

    if not isinstance(features, pd.DataFrame):
        raise TargetBuilderError("features must be a pandas DataFrame.")

    expected_columns = (
        ["session_date"] + PHASE15_FEATURE_COLUMNS
    )

    if list(features.columns) != expected_columns:
        raise TargetBuilderError(
            "Feature frame does not match the official Phase 15 "
            "31-feature schema."
        )

    if features["session_date"].duplicated().any():
        raise TargetBuilderError(
            "Duplicate feature session dates detected."
        )

    if not features["session_date"].is_monotonic_increasing:
        raise TargetBuilderError(
            "Feature sessions must be sorted oldest to newest."
        )

    target_frame = build_return_targets(
        stock_data=stock_data,
        horizons=horizons,
    )

    if len(features) != len(target_frame):
        raise TargetBuilderError(
            "Feature and target row counts differ."
        )

    feature_dates = features["session_date"].tolist()
    target_dates = target_frame["session_date"].tolist()

    if feature_dates != target_dates:
        raise TargetBuilderError(
            "Feature and target session dates are not identical."
        )

    selected_horizons = _validate_horizons(horizons)

    result = pd.concat(
        [
            features.reset_index(drop=True),
            target_frame.drop(
                columns=["session_date"]
            ).reset_index(drop=True),
        ],
        axis=1,
    )

    numeric_features = result[
        PHASE15_FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    if np.isinf(numeric_features).any():
        raise TargetBuilderError(
            "Feature matrix contains infinite values."
        )

    feature_ready = result[
        PHASE15_FEATURE_COLUMNS
    ].notna().all(axis=1)

    target_columns = []

    for horizon in selected_horizons:
        target_columns.extend([
            f"close_to_close_return_{horizon}s",
            f"next_open_to_close_return_{horizon}s",
        ])

    targets_ready = result[
        target_columns
    ].notna().all(axis=1)

    result["feature_ready"] = feature_ready
    result["targets_ready"] = targets_ready

    result["training_eligible"] = (
        feature_ready & targets_ready
    )

    return result
