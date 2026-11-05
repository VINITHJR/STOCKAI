"""Leakage-safe chronological walk-forward splitting for Phase 15.

Input:
    A combined panel containing one row per instrument and session_date.
    Required columns:
        instrument_key
        session_date
        max_horizon_target_session
        training_eligible

The panel should be assembled from build_phase15_training_frame() outputs.

Rules:
    - Every instrument must have the same session calendar.
    - Training eligibility must agree across instruments on each date.
    - The maximum target date for an eligible row must be exactly
      max_horizon trading sessions after its feature date.
    - A training row is allowed only when its maximum target date is
      strictly earlier than the validation start.
    - The final holdout is not included in any walk-forward fold.
    - Splits are deterministic; no random splitting is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


class WalkForwardError(ValueError):
    """Raised when the panel cannot safely be split."""


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-training, forward-validation fold.

    Indices are positional indices into WalkForwardPlan.panel.
    Always select data with .iloc[indices], not .loc[indices].
    """

    fold_number: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_start_date: str
    validation_end_date: str
    train_start_date: str
    train_end_date: str
    train_session_count: int
    validation_session_count: int
    purged_sample_count: int
    purged_session_count: int


@dataclass(frozen=True)
class WalkForwardPlan:
    """Complete walk-forward plan and separately reserved final holdout."""

    panel: pd.DataFrame
    folds: tuple[WalkForwardFold, ...]
    final_holdout_indices: tuple[int, ...]
    final_train_indices: tuple[int, ...]
    final_train_start_date: str
    final_train_end_date: str
    final_train_session_count: int
    final_purged_sample_count: int
    final_purged_session_count: int
    final_holdout_start_date: str
    final_holdout_end_date: str
    final_holdout_session_count: int
    holdout_embargo_session_count: int
    instrument_keys: tuple[str, ...]
    max_horizon: int


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise WalkForwardError(f"{name} must be a positive integer.")

    value = int(value)

    if value <= 0:
        raise WalkForwardError(f"{name} must be a positive integer.")

    return value


def combine_training_frames(
    training_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine per-instrument training frames into a single panel.

    Example:
        frames = {
            instrument_key: build_phase15_training_frame(...),
            ...
        }
        panel = combine_training_frames(frames)

    The mapping key becomes the instrument_key column. If a frame already
    contains instrument_key, its values must agree with the mapping key.
    """

    if not isinstance(training_frames, Mapping) or not training_frames:
        raise WalkForwardError(
            "training_frames must be a non-empty mapping of "
            "instrument keys to DataFrames."
        )

    combined = []

    for raw_key, frame in training_frames.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise WalkForwardError(
                f"Invalid instrument key: {raw_key!r}"
            )

        instrument_key = raw_key.strip()

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise WalkForwardError(
                f"{instrument_key}: training frame must be a non-empty DataFrame."
            )

        required = {
            "session_date",
            "max_horizon_target_session",
            "training_eligible",
        }
        missing = required.difference(frame.columns)

        if missing:
            raise WalkForwardError(
                f"{instrument_key}: missing columns: {sorted(missing)}"
            )

        item = frame.copy()

        if "instrument_key" in item.columns:
            existing = item["instrument_key"].dropna().astype(str).unique()

            if len(existing) != 1 or existing[0] != instrument_key:
                raise WalkForwardError(
                    f"{instrument_key}: existing instrument_key values "
                    "do not match the mapping key."
                )
        else:
            item["instrument_key"] = instrument_key

        combined.append(item)

    panel = pd.concat(combined, ignore_index=True, sort=False)

    return panel


def _parse_date_column(
    values: pd.Series,
    column_name: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    """Convert a date column to timezone-naive normalized timestamps."""

    try:
        parsed = pd.to_datetime(values, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise WalkForwardError(
            f"Could not parse {column_name} as dates."
        ) from exc

    if not isinstance(parsed, pd.Series):
        parsed = pd.Series(parsed, index=values.index)

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)

    if not allow_missing and parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise WalkForwardError(
            f"{column_name} contains {bad_count} missing or invalid dates."
        )

    return parsed.dt.normalize()


def _validate_panel(
    panel: pd.DataFrame,
    max_horizon: int,
) -> tuple[pd.DataFrame, tuple[str, ...], list[pd.Timestamp], pd.Series]:
    """Validate panel structure and return normalized working data."""

    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise WalkForwardError("panel must be a non-empty DataFrame.")

    required = {
        "instrument_key",
        "session_date",
        "max_horizon_target_session",
        "training_eligible",
    }
    missing = required.difference(panel.columns)

    if missing:
        raise WalkForwardError(
            f"Panel is missing required columns: {sorted(missing)}"
        )

    work = panel.copy().reset_index(drop=True)

    if work["instrument_key"].isna().any():
        raise WalkForwardError("instrument_key contains missing values.")

    work["instrument_key"] = work["instrument_key"].astype(str).str.strip()

    if (work["instrument_key"] == "").any():
        raise WalkForwardError("instrument_key contains blank values.")

    # Do not silently turn text such as "False" into True.
    valid_boolean_values = work["training_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )

    if not valid_boolean_values.all():
        raise WalkForwardError(
            "training_eligible must contain actual boolean values."
        )

    work["_eligible"] = work["training_eligible"].astype(bool)

    work["_session_date"] = _parse_date_column(
        work["session_date"],
        "session_date",
        allow_missing=False,
    )

    work["_target_end_date"] = _parse_date_column(
        work["max_horizon_target_session"],
        "max_horizon_target_session",
        allow_missing=True,
    )

    if work.duplicated(["instrument_key", "_session_date"]).any():
        raise WalkForwardError(
            "Duplicate instrument_key/session_date rows detected."
        )

    instrument_keys = tuple(sorted(work["instrument_key"].unique()))

    if not instrument_keys:
        raise WalkForwardError("No instruments were found in the panel.")

    calendars = {}

    for instrument_key, group in work.groupby("instrument_key", sort=False):
        calendars[instrument_key] = tuple(
            pd.Timestamp(value)
            for value in group["_session_date"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    reference_calendar = calendars[instrument_keys[0]]

    for instrument_key in instrument_keys[1:]:
        if calendars[instrument_key] != reference_calendar:
            raise WalkForwardError(
                "Trading calendar mismatch for "
                f"{instrument_key}. All instruments must have identical "
                "session dates."
            )

    # All instruments must agree on whether a session is eligible.
    eligibility_counts = (
        work.groupby("_session_date")["_eligible"].nunique()
    )

    if (eligibility_counts > 1).any():
        bad_dates = eligibility_counts[eligibility_counts > 1].index
        examples = [
            pd.Timestamp(value).strftime("%Y-%m-%d")
            for value in list(bad_dates[:5])
        ]
        raise WalkForwardError(
            "training_eligible differs across instruments on shared dates. "
            f"Example dates: {examples}"
        )

    calendar = list(reference_calendar)
    date_positions = {
        session_date: position
        for position, session_date in enumerate(calendar)
    }

    eligible_rows = work["_eligible"]

    if not eligible_rows.any():
        raise WalkForwardError("No training-eligible samples were found.")

    eligible_target_dates = work.loc[
        eligible_rows, "_target_end_date"
    ]

    if eligible_target_dates.isna().any():
        raise WalkForwardError(
            "An eligible sample is missing max_horizon_target_session."
        )

    eligible_feature_dates = work.loc[
        eligible_rows, "_session_date"
    ]

    for feature_date, target_date in zip(
        eligible_feature_dates,
        eligible_target_dates,
    ):
        if target_date not in date_positions:
            raise WalkForwardError(
                "An eligible sample's target date is outside the "
                f"shared calendar: {target_date}"
            )

        feature_position = date_positions[feature_date]
        target_position = date_positions[target_date]

        if target_position - feature_position != max_horizon:
            raise WalkForwardError(
                "Eligible sample has an unexpected maximum target horizon: "
                f"feature_date={feature_date.date()}, "
                f"target_date={target_date.date()}, "
                f"expected_sessions={max_horizon}, "
                f"actual_sessions={target_position - feature_position}."
            )

    # Order records chronologically, then by instrument.
    work = work.sort_values(
        ["_session_date", "instrument_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    eligible_by_date = (
        work.groupby("_session_date")["_eligible"].first()
    )

    eligible_dates = [
        pd.Timestamp(session_date)
        for session_date, is_eligible in eligible_by_date.items()
        if bool(is_eligible)
    ]

    if not eligible_dates:
        raise WalkForwardError("No shared eligible session dates found.")

    return work, instrument_keys, eligible_dates, work["_eligible"]


def build_walk_forward_splits(
    panel: pd.DataFrame,
    *,
    n_splits: int = 5,
    initial_train_size: int = 252,
    validation_size: int | None = None,
    final_holdout_size: int = 126,
    max_horizon: int = 7,
) -> WalkForwardPlan:
    """Build expanding walk-forward folds and reserve a final holdout.

    Parameters
    ----------
    panel:
        Combined multi-instrument panel. Use combine_training_frames().

    n_splits:
        Number of sequential validation windows.

    initial_train_size:
        Number of eligible shared session dates reserved before the
        first validation window. The effective training history will be
        shorter after the purge is applied.

    validation_size:
        Optional fixed number of eligible sessions per validation fold.
        If omitted, the available development validation dates are split
        as evenly as possible across n_splits.

    final_holdout_size:
        Number of final eligible shared session dates reserved for the
        final holdout. These dates are excluded from every model-selection
        training and validation fold.

    max_horizon:
        Maximum target horizon in trading sessions. Must match the
        max_horizon_target_session field. Phase 15 uses 7.

    Returns
    -------
    WalkForwardPlan
        The plan contains a chronologically sorted panel. All indices
        refer to plan.panel and must be selected with .iloc.

    Notes
    -----
    For each fold, a training sample is permitted only when:

        max_horizon_target_session < validation_start_date

    Equality is excluded. This purges samples whose target reaches the
    validation start, preventing labels from crossing the boundary.
    """

    n_splits = _validate_positive_integer(n_splits, "n_splits")
    initial_train_size = _validate_positive_integer(
        initial_train_size,
        "initial_train_size",
    )
    final_holdout_size = _validate_positive_integer(
        final_holdout_size,
        "final_holdout_size",
    )
    max_horizon = _validate_positive_integer(
        max_horizon,
        "max_horizon",
    )

    if validation_size is not None:
        validation_size = _validate_positive_integer(
            validation_size,
            "validation_size",
        )

    work, instrument_keys, eligible_dates, _ = _validate_panel(
        panel,
        max_horizon,
    )

    if final_holdout_size >= len(eligible_dates):
        raise WalkForwardError(
            "final_holdout_size must be smaller than the number "
            "of eligible shared session dates."
        )

    development_dates = eligible_dates[:-final_holdout_size]
    holdout_dates = eligible_dates[-final_holdout_size:]

    # Keep the final holdout sealed from validation outcomes as well.
    # The final max_horizon eligible sessions before the holdout are
    # embargoed because their labels would extend into the holdout period.
    if len(development_dates) <= initial_train_size + max_horizon:
        raise WalkForwardError(
            "Insufficient development history after reserving the final "
            "holdout and its validation embargo."
        )

    validation_safe_dates = development_dates[:-max_horizon]
    validation_candidates = validation_safe_dates[initial_train_size:]

    if validation_size is None:
        if n_splits > len(validation_candidates):
            raise WalkForwardError(
                "n_splits cannot exceed the number of available "
                "validation dates."
            )

        validation_windows = [
            list(window)
            for window in np.array_split(
                np.asarray(validation_candidates, dtype=object),
                n_splits,
            )
            if len(window) > 0
        ]
    else:
        required_validation_dates = n_splits * validation_size

        if required_validation_dates > len(validation_candidates):
            raise WalkForwardError(
                "Not enough validation dates for the requested "
                "n_splits and validation_size."
            )

        validation_windows = []

        for fold_index in range(n_splits):
            start = fold_index * validation_size
            end = start + validation_size

            validation_windows.append(
                validation_candidates[start:end]
            )

    holdout_set = set(holdout_dates)
    eligible_mask = work["_eligible"]
    session_dates = work["_session_date"]
    target_end_dates = work["_target_end_date"]

    holdout_mask = (
        eligible_mask
        & session_dates.isin(holdout_set)
    )
    holdout_indices = tuple(
        int(value) for value in np.flatnonzero(holdout_mask.to_numpy())
    )

    expected_holdout_rows = (
        len(holdout_dates) * len(instrument_keys)
    )

    if len(holdout_indices) != expected_holdout_rows:
        raise WalkForwardError(
            "Final holdout does not contain every instrument on every "
            "reserved date."
        )

    # Build the final model's development-training set.
    # Every training label must end strictly before the holdout begins.
    final_holdout_start = pd.Timestamp(holdout_dates[0])

    final_prior_mask = (
        eligible_mask
        & session_dates.lt(final_holdout_start)
    )

    final_train_mask = (
        final_prior_mask
        & target_end_dates.lt(final_holdout_start)
    )

    final_purged_mask = (
        final_prior_mask
        & target_end_dates.ge(final_holdout_start)
    )

    final_train_indices = tuple(
        int(value)
        for value in np.flatnonzero(final_train_mask.to_numpy())
    )

    if not final_train_indices:
        raise WalkForwardError(
            "Final training set is empty after applying the holdout purge."
        )

    final_training_dates = session_dates.iloc[
        list(final_train_indices)
    ]

    final_training_targets = target_end_dates.iloc[
        list(final_train_indices)
    ]

    if not (final_training_targets < final_holdout_start).all():
        raise WalkForwardError(
            "Final training contains a target reaching the holdout boundary."
        )

    if set(final_train_indices).intersection(holdout_indices):
        raise WalkForwardError(
            "Final training and final holdout indices overlap."
        )

    final_train_session_count = int(
        final_training_dates.nunique()
    )

    final_purged_sample_count = int(final_purged_mask.sum())

    final_purged_session_count = int(
        session_dates.loc[final_purged_mask].nunique()
    )

    folds = []

    previous_train_count = -1

    for fold_number, validation_dates in enumerate(
        validation_windows,
        start=1,
    ):
        if not validation_dates:
            raise WalkForwardError(
                f"Validation fold {fold_number} has no dates."
            )

        validation_start = pd.Timestamp(validation_dates[0])
        validation_end = pd.Timestamp(validation_dates[-1])
        validation_set = set(validation_dates)

        # Candidate training samples must precede validation.
        prior_samples = (
            eligible_mask
            & session_dates.lt(validation_start)
        )

        # Strictly earlier: equality is purged too.
        allowed_training = (
            prior_samples
            & target_end_dates.lt(validation_start)
        )

        purged_samples = (
            prior_samples
            & target_end_dates.ge(validation_start)
        )

        validation_mask = (
            eligible_mask
            & session_dates.isin(validation_set)
        )

        train_indices = tuple(
            int(value)
            for value in np.flatnonzero(allowed_training.to_numpy())
        )
        validation_indices = tuple(
            int(value)
            for value in np.flatnonzero(validation_mask.to_numpy())
        )

        # Validation outcomes must also finish before the final holdout.
        validation_target_dates = target_end_dates.iloc[
            list(validation_indices)
        ]

        if (
            validation_target_dates.isna().any()
            or not (validation_target_dates < final_holdout_start).all()
        ):
            raise WalkForwardError(
                f"Fold {fold_number} has validation targets that reach "
                "or cross the final holdout boundary."
            )

        if not train_indices:
            raise WalkForwardError(
                f"Fold {fold_number} has no training samples after purge."
            )

        if not validation_indices:
            raise WalkForwardError(
                f"Fold {fold_number} has no validation samples."
            )

        expected_validation_rows = (
            len(validation_dates) * len(instrument_keys)
        )

        if len(validation_indices) != expected_validation_rows:
            raise WalkForwardError(
                f"Fold {fold_number} does not contain every instrument "
                "on every validation date."
            )

        train_date_count = int(
            session_dates.iloc[list(train_indices)].nunique()
        )
        validation_date_count = int(
            session_dates.iloc[list(validation_indices)].nunique()
        )

        if train_date_count <= previous_train_count:
            raise WalkForwardError(
                "Training windows are not expanding. Check the panel "
                "eligibility and target dates."
            )

        previous_train_count = train_date_count

        train_dates = session_dates.iloc[list(train_indices)]
        max_train_target_date = target_end_dates.iloc[
            list(train_indices)
        ].max()

        if not (max_train_target_date < validation_start):
            raise WalkForwardError(
                f"Fold {fold_number} contains target leakage across "
                "the validation boundary."
            )

        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                validation_start_date=validation_start.strftime("%Y-%m-%d"),
                validation_end_date=validation_end.strftime("%Y-%m-%d"),
                train_start_date=train_dates.min().strftime("%Y-%m-%d"),
                train_end_date=train_dates.max().strftime("%Y-%m-%d"),
                train_session_count=train_date_count,
                validation_session_count=validation_date_count,
                purged_sample_count=int(purged_samples.sum()),
                purged_session_count=int(
                    session_dates.loc[purged_samples].nunique()
                ),
            )
        )

    # The final holdout must occur strictly after all model-selection folds.
    last_validation_date = max(
        pd.Timestamp(fold.validation_end_date)
        for fold in folds
    )

    if not (
        last_validation_date
        < pd.Timestamp(holdout_dates[0])
    ):
        raise WalkForwardError(
            "Final holdout overlaps the walk-forward validation period."
        )

    # No final holdout row may appear in any fold.
    holdout_index_set = set(holdout_indices)

    for fold in folds:
        if holdout_index_set.intersection(fold.train_indices):
            raise WalkForwardError(
                f"Final holdout leaked into fold {fold.fold_number} training."
            )

        if holdout_index_set.intersection(fold.validation_indices):
            raise WalkForwardError(
                f"Final holdout leaked into fold {fold.fold_number} validation."
            )

    # Remove internal validation columns from the public panel.
    public_panel = work.drop(
        columns=[
            "_eligible",
            "_session_date",
            "_target_end_date",
        ]
    ).reset_index(drop=True)

    return WalkForwardPlan(
        panel=public_panel,
        folds=tuple(folds),
        final_holdout_indices=holdout_indices,
        final_train_indices=final_train_indices,
        final_train_start_date=final_training_dates.min().strftime("%Y-%m-%d"),
        final_train_end_date=final_training_dates.max().strftime("%Y-%m-%d"),
        final_train_session_count=final_train_session_count,
        final_purged_sample_count=final_purged_sample_count,
        final_purged_session_count=final_purged_session_count,
        final_holdout_start_date=holdout_dates[0].strftime("%Y-%m-%d"),
        final_holdout_end_date=holdout_dates[-1].strftime("%Y-%m-%d"),
        final_holdout_session_count=len(holdout_dates),
        holdout_embargo_session_count=max_horizon,
        instrument_keys=instrument_keys,
        max_horizon=max_horizon,
    )
