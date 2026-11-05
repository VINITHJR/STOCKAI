"""Leakage-conscious walk-forward model benchmarking for Phase 15.

This module:
- Uses exactly PHASE15_FEATURE_COLUMNS as model inputs.
- Supports one target per benchmark run.
- Fits imputers and scalers independently inside each fold.
- Uses no random train/test split.
- Compares zero and training-mean baselines with Ridge, Random Forest,
  and Extra Trees regressors.
- Never reads final-holdout feature rows or target values.
- Does not write models, predictions, scalers, or reports to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.ml.phase15.features.feature_builder import (
    PHASE15_FEATURE_COLUMNS,
)
from backend.app.ml.phase15.validation.walk_forward import (
    WalkForwardPlan,
)


class ModelBenchmarkError(ValueError):
    """Raised when benchmark data or configuration is invalid."""


PRIMARY_TARGET = "next_open_to_close_return_7s"
FORECAST_TARGET = "close_to_close_return_7s"

ALLOWED_TARGETS = (
    PRIMARY_TARGET,
    FORECAST_TARGET,
)

DEFAULT_MODELS = (
    "zero_return",
    "train_mean",
    "ridge",
    "random_forest",
    "extra_trees",
)

MODEL_NAMES = frozenset(DEFAULT_MODELS)

DIRECTION_EPSILON = 1e-12


@dataclass(frozen=True)
class ModelBenchmarkResult:
    """In-memory output from one target's walk-forward benchmark."""

    target_column: str
    feature_columns: tuple[str, ...]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fit_audit: pd.DataFrame
    holdout_used: bool = False


def _validate_configuration(
    plan: WalkForwardPlan,
    target_column: str,
    model_names: Sequence[str],
) -> tuple[str, ...]:

    if not isinstance(plan, WalkForwardPlan):
        raise ModelBenchmarkError(
            "plan must be a WalkForwardPlan instance."
        )

    if target_column not in ALLOWED_TARGETS:
        raise ModelBenchmarkError(
            f"Unsupported target {target_column!r}. "
            f"Choose exactly one of: {ALLOWED_TARGETS}."
        )

    if plan.max_horizon != 7:
        raise ModelBenchmarkError(
            "This benchmark currently requires a seven-session "
            "walk-forward plan."
        )

    if not model_names:
        raise ModelBenchmarkError(
            "At least one benchmark model must be selected."
        )

    normalized = tuple(model_names)

    if len(set(normalized)) != len(normalized):
        raise ModelBenchmarkError(
            "Duplicate model names are not allowed."
        )

    unknown = set(normalized).difference(MODEL_NAMES)

    if unknown:
        raise ModelBenchmarkError(
            f"Unknown model names: {sorted(unknown)}."
        )

    if not plan.folds:
        raise ModelBenchmarkError(
            "Walk-forward plan contains no validation folds."
        )

    return normalized


def _validate_required_columns(
    panel: pd.DataFrame,
    target_column: str,
) -> None:

    required = {
        "instrument_key",
        "session_date",
        "max_horizon_target_session",
        "training_eligible",
        target_column,
        *PHASE15_FEATURE_COLUMNS,
    }

    missing = required.difference(panel.columns)

    if missing:
        raise ModelBenchmarkError(
            f"Benchmark panel is missing columns: {sorted(missing)}"
        )

    if len(PHASE15_FEATURE_COLUMNS) != 31:
        raise ModelBenchmarkError(
            "Phase 15 feature schema must contain exactly 31 columns."
        )

    if len(set(PHASE15_FEATURE_COLUMNS)) != 31:
        raise ModelBenchmarkError(
            "Phase 15 feature schema contains duplicate names."
        )

    forbidden = {
        target_column,
        "close_to_close_return_7s",
        "next_open_to_close_return_7s",
        "target_close_7s",
        "target_exit_session_7s",
        "max_horizon_target_session",
        "base_close",
        "next_session_open",
        "next_session_date",
        "all_targets_available",
        "feature_ready",
        "targets_ready",
        "training_eligible",
    }

    if forbidden.intersection(PHASE15_FEATURE_COLUMNS):
        raise ModelBenchmarkError(
            "A target or eligibility column appears in the feature schema."
        )


def _extract_features(
    frame: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DataFrame:
    """Extract only the approved 31 features."""

    # Deliberately do not use select_dtypes, numeric-column discovery,
    # or all columns except the target. The feature list is explicit.
    X = frame.loc[:, list(PHASE15_FEATURE_COLUMNS)].copy()

    for column in PHASE15_FEATURE_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    values = X.to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ModelBenchmarkError(
            f"{split_name}: feature matrix contains infinity."
        )

    return X


def _extract_target(
    frame: pd.DataFrame,
    target_column: str,
    *,
    split_name: str,
) -> np.ndarray:

    values = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ModelBenchmarkError(
            f"{split_name}: target contains missing or non-finite values."
        )

    return values


def _validate_split(
    plan: WalkForwardPlan,
    fold,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    panel = plan.panel

    train_indices = tuple(fold.train_indices)
    validation_indices = tuple(fold.validation_indices)
    holdout_indices = set(plan.final_holdout_indices)

    if not train_indices or not validation_indices:
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: empty training or validation split."
        )

    if len(set(train_indices)) != len(train_indices):
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: duplicate training indices."
        )

    if len(set(validation_indices)) != len(validation_indices):
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: duplicate validation indices."
        )

    if set(train_indices).intersection(validation_indices):
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: train/validation overlap."
        )

    if holdout_indices.intersection(train_indices):
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: final holdout entered training."
        )

    if holdout_indices.intersection(validation_indices):
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: final holdout entered validation."
        )

    train = panel.iloc[list(train_indices)].copy()
    validation = panel.iloc[list(validation_indices)].copy()

    if not train["training_eligible"].astype(bool).all():
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: ineligible training row detected."
        )

    if not validation["training_eligible"].astype(bool).all():
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: ineligible validation row detected."
        )

    train_target_dates = pd.to_datetime(
        train["max_horizon_target_session"],
        errors="coerce",
    )

    validation_target_dates = pd.to_datetime(
        validation["max_horizon_target_session"],
        errors="coerce",
    )

    if train_target_dates.isna().any():
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: missing training target-end date."
        )

    if validation_target_dates.isna().any():
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: missing validation target-end date."
        )

    validation_start = pd.Timestamp(fold.validation_start_date)

    if not (train_target_dates < validation_start).all():
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: training target crosses "
            "the validation boundary."
        )

    holdout_start = pd.Timestamp(plan.final_holdout_start_date)

    if not (validation_target_dates < holdout_start).all():
        raise ModelBenchmarkError(
            f"Fold {fold.fold_number}: validation target crosses "
            "the final holdout boundary."
        )

    return train, validation


def _make_estimator(model_name: str) -> Pipeline:

    imputer = SimpleImputer(strategy="median")

    if model_name == "ridge":
        return Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])

    if model_name == "random_forest":
        return Pipeline([
            ("imputer", imputer),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])

    if model_name == "extra_trees":
        return Pipeline([
            ("imputer", imputer),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])

    raise ModelBenchmarkError(
        f"No estimator is defined for {model_name!r}."
    )


def _calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ModelBenchmarkError(
            "Metrics require equally sized non-empty arrays."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ModelBenchmarkError(
            "Metrics cannot be calculated with non-finite values."
        )

    errors = predicted - actual

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    actual_mean = float(np.mean(actual))

    total_sum_squares = float(
        np.sum(np.square(actual - actual_mean))
    )

    residual_sum_squares = float(
        np.sum(np.square(errors))
    )

    # Identify constant or numerically indistinguishable targets
    # using their range, rather than variance around a rounded mean.
    actual_min = float(np.min(actual))
    actual_max = float(np.max(actual))
    actual_range = actual_max - actual_min

    actual_scale = max(
        1.0,
        abs(actual_min),
        abs(actual_max),
    )

    constant_tolerance = (
        np.finfo(np.float64).eps
        * actual_scale
        * 10.0
    )

    is_effectively_constant = (
        actual_range <= constant_tolerance
    )

    if not is_effectively_constant and total_sum_squares > 0.0:
        r2 = float(
            1.0 - residual_sum_squares / total_sum_squares
        )
    else:
        r2 = float("nan")

    active_predictions = (
        np.abs(predicted) > DIRECTION_EPSILON
    )

    coverage = float(
        np.mean(active_predictions) * 100.0
    )

    if active_predictions.any():
        directional_accuracy = float(
            np.mean(
                np.sign(predicted[active_predictions])
                == np.sign(actual[active_predictions])
            ) * 100.0
        )
    else:
        # A zero prediction has no direction, so it is not counted
        # as either a correct or incorrect directional prediction.
        directional_accuracy = float("nan")

    return {
        "sample_count": int(len(actual)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy_pct": directional_accuracy,
        "directional_coverage_pct": coverage,
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_std": float(np.std(actual, ddof=0)),
        "predicted_std": float(np.std(predicted, ddof=0)),
    }


def benchmark_walk_forward(
    plan: WalkForwardPlan,
    *,
    target_column: str = PRIMARY_TARGET,
    model_names: Sequence[str] = DEFAULT_MODELS,
) -> ModelBenchmarkResult:
    """Benchmark models on the walk-forward folds for one target.

    The final holdout is deliberately not selected, transformed, fitted,
    predicted, or scored in this function.

    Call this function separately for:
        PRIMARY_TARGET:
            next_open_to_close_return_7s

        FORECAST_TARGET:
            close_to_close_return_7s
    """

    selected_models = _validate_configuration(
        plan,
        target_column,
        model_names,
    )

    panel = plan.panel

    _validate_required_columns(panel, target_column)

    prediction_frames = []
    fit_audit_rows = []
    metric_rows = []

    for fold in plan.folds:
        train, validation = _validate_split(plan, fold)

        X_train = _extract_features(
            train,
            split_name=f"fold {fold.fold_number} training",
        )

        X_validation = _extract_features(
            validation,
            split_name=f"fold {fold.fold_number} validation",
        )

        y_train = _extract_target(
            train,
            target_column,
            split_name=f"fold {fold.fold_number} training",
        )

        y_validation = _extract_target(
            validation,
            target_column,
            split_name=f"fold {fold.fold_number} validation",
        )

        # An entirely missing training feature cannot be imputed safely.
        all_missing_training_columns = [
            column
            for column in PHASE15_FEATURE_COLUMNS
            if X_train[column].notna().sum() == 0
        ]

        if all_missing_training_columns:
            raise ModelBenchmarkError(
                f"Fold {fold.fold_number}: these features contain no "
                "finite training observations: "
                f"{all_missing_training_columns}"
            )

        fold_predictions = {}

        for model_name in selected_models:

            if model_name == "zero_return":
                predictions = np.zeros(
                    len(validation),
                    dtype=float,
                )
                imputer_scope = "not_applicable"
                scaler_scope = "not_applicable"
                target_fit_scope = "not_applicable"

            elif model_name == "train_mean":
                training_mean = float(np.mean(y_train))

                predictions = np.full(
                    len(validation),
                    training_mean,
                    dtype=float,
                )
                imputer_scope = "not_applicable"
                scaler_scope = "not_applicable"
                target_fit_scope = "training_only"

            else:
                # A fresh pipeline is constructed and fitted per fold.
                # Its imputer and scaler see training rows only.
                estimator = _make_estimator(model_name)

                estimator.fit(X_train, y_train)

                predictions = np.asarray(
                    estimator.predict(X_validation),
                    dtype=float,
                )

                imputer_scope = "training_only"

                scaler_scope = (
                    "training_only"
                    if model_name == "ridge"
                    else "not_used"
                )

                target_fit_scope = "training_only"

            if predictions.shape != y_validation.shape:
                raise ModelBenchmarkError(
                    f"Fold {fold.fold_number}, {model_name}: "
                    "prediction shape mismatch."
                )

            if not np.isfinite(predictions).all():
                raise ModelBenchmarkError(
                    f"Fold {fold.fold_number}, {model_name}: "
                    "predictions contain non-finite values."
                )

            fold_predictions[model_name] = predictions

            prediction_frame = pd.DataFrame({
                "fold_number": fold.fold_number,
                "model": model_name,
                "instrument_key": validation[
                    "instrument_key"
                ].astype(str).to_numpy(),
                "session_date": validation[
                    "session_date"
                ].astype(str).to_numpy(),
                "target_column": target_column,
                "actual_return": y_validation,
                "predicted_return": predictions,
            })

            prediction_frames.append(prediction_frame)

            scores = _calculate_metrics(
                y_validation,
                predictions,
            )

            metric_rows.append({
                "scope": "fold",
                "fold_number": fold.fold_number,
                "model": model_name,
                "target_column": target_column,
                **scores,
            })

            fit_audit_rows.append({
                "fold_number": fold.fold_number,
                "model": model_name,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "feature_count": len(PHASE15_FEATURE_COLUMNS),
                "training_start_date": train[
                    "session_date"
                ].min(),
                "training_end_date": train[
                    "session_date"
                ].max(),
                "maximum_training_target_end": pd.to_datetime(
                    train["max_horizon_target_session"]
                ).max().strftime("%Y-%m-%d"),
                "validation_start_date": fold.validation_start_date,
                "maximum_validation_target_end": pd.to_datetime(
                    validation["max_horizon_target_session"]
                ).max().strftime("%Y-%m-%d"),
                "model_fit_scope": target_fit_scope,
                "imputer_fit_scope": imputer_scope,
                "scaler_fit_scope": scaler_scope,
                "final_holdout_used": False,
            })

    predictions_frame = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # Aggregate the same out-of-fold predictions for each model.
    for model_name in selected_models:
        model_predictions = predictions_frame.loc[
            predictions_frame["model"] == model_name
        ]

        scores = _calculate_metrics(
            model_predictions["actual_return"].to_numpy(dtype=float),
            model_predictions["predicted_return"].to_numpy(dtype=float),
        )

        metric_rows.append({
            "scope": "pooled_oof",
            "fold_number": None,
            "model": model_name,
            "target_column": target_column,
            **scores,
        })

    metrics_frame = pd.DataFrame(metric_rows)
    fit_audit_frame = pd.DataFrame(fit_audit_rows)

    predictions_frame = predictions_frame.sort_values(
        ["fold_number", "model", "session_date", "instrument_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    return ModelBenchmarkResult(
        target_column=target_column,
        feature_columns=tuple(PHASE15_FEATURE_COLUMNS),
        metrics=metrics_frame,
        predictions=predictions_frame,
        fit_audit=fit_audit_frame,
        holdout_used=False,
    )
