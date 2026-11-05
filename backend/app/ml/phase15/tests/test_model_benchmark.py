import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.feature_builder import (
    PHASE15_FEATURE_COLUMNS,
)
from backend.app.ml.phase15.validation import model_benchmark as benchmark
from backend.app.ml.phase15.validation.walk_forward import (
    build_walk_forward_splits,
    combine_training_frames,
)


INSTRUMENTS = [
    f"NSE_EQ|BENCH{i:04d}"
    for i in range(20)
]

ROWS_PER_STOCK = 100

BENCHMARK_MODELS = (
    "zero_return",
    "train_mean",
    "ridge",
    "random_forest",
    "extra_trees",
)


def make_training_frame(instrument_number, rows=ROWS_PER_STOCK):
    """Create deterministic synthetic features and two separate targets."""

    dates = pd.bdate_range("2025-01-01", periods=rows)
    output = []

    for i, session in enumerate(dates):
        target_position = i + 7

        target_date = (
            dates[target_position].strftime("%Y-%m-%d")
            if target_position < rows
            else None
        )

        eligible = (
            i >= 5
            and target_position < rows
        )

        row = {
            "session_date": session.strftime("%Y-%m-%d"),
            "max_horizon_target_session": target_date,
            "training_eligible": eligible,
            "feature_ready": i >= 5,
            "targets_ready": target_position < rows,
        }

        # Deliberately create distinct numerical features.
        for feature_number, column in enumerate(
            PHASE15_FEATURE_COLUMNS
        ):
            value = (
                np.sin(
                    i / (3 + feature_number % 5)
                    + (instrument_number + 1)
                    * (feature_number + 1)
                    * 0.013
                )
                + 0.05 * instrument_number
                + 0.001 * feature_number
            )

            # Preserve warm-up NaNs.
            if i < 5:
                value = np.nan

            # Add a small number of missing observations to test
            # training-only imputation.
            if feature_number == 0 and i in (10, 11):
                value = np.nan

            row[column] = value

        # Two separate synthetic return targets.
        primary_return = (
            0.01 * np.sin(i / 7 + instrument_number * 0.07)
            + 0.001 * instrument_number
        )

        forecast_return = (
            0.012 * np.cos(i / 8 + instrument_number * 0.05)
            - 0.0005 * instrument_number
        )

        row[benchmark.PRIMARY_TARGET] = (
            primary_return if eligible else np.nan
        )

        row[benchmark.FORECAST_TARGET] = (
            forecast_return if eligible else np.nan
        )

        output.append(row)

    return pd.DataFrame(output)


def make_synthetic_plan():
    """Create a 20-instrument chronological walk-forward plan."""

    frames = {
        instrument_key: make_training_frame(number)
        for number, instrument_key in enumerate(INSTRUMENTS)
    }

    panel = combine_training_frames(frames)

    return build_walk_forward_splits(
        panel,
        n_splits=3,
        initial_train_size=20,
        final_holdout_size=12,
        max_horizon=7,
    )


class TestModelBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = make_synthetic_plan()

        cls.result = benchmark.benchmark_walk_forward(
            cls.plan,
            target_column=benchmark.PRIMARY_TARGET,
            model_names=BENCHMARK_MODELS,
        )

    def test_plan_contains_twenty_synthetic_instruments(self):
        self.assertEqual(
            len(self.plan.instrument_keys),
            20,
        )

    def test_exactly_thirty_one_features_are_used(self):
        self.assertEqual(
            len(self.result.feature_columns),
            31,
        )

        self.assertEqual(
            self.result.feature_columns,
            tuple(PHASE15_FEATURE_COLUMNS),
        )

        self.assertEqual(
            len(set(self.result.feature_columns)),
            31,
        )

    def test_target_and_metadata_columns_are_not_features(self):
        forbidden = {
            benchmark.PRIMARY_TARGET,
            benchmark.FORECAST_TARGET,
            "target_close_7s",
            "target_exit_session_7s",
            "max_horizon_target_session",
            "base_close",
            "next_session_open",
            "next_session_date",
            "training_eligible",
            "feature_ready",
            "targets_ready",
        }

        self.assertTrue(
            forbidden.isdisjoint(self.result.feature_columns)
        )

    def test_all_five_models_are_benchmarked(self):
        pooled = self.result.metrics.loc[
            self.result.metrics["scope"] == "pooled_oof"
        ]

        self.assertEqual(
            set(pooled["model"]),
            set(BENCHMARK_MODELS),
        )

    def test_every_model_uses_the_same_validation_samples(self):
        predictions = self.result.predictions

        for fold_number in range(1, len(self.plan.folds) + 1):
            fold_data = predictions.loc[
                predictions["fold_number"] == fold_number
            ]

            reference = None

            for model_name in BENCHMARK_MODELS:
                current = fold_data.loc[
                    fold_data["model"] == model_name,
                    [
                        "instrument_key",
                        "session_date",
                        "actual_return",
                    ],
                ].sort_values(
                    ["instrument_key", "session_date"]
                ).reset_index(drop=True)

                if reference is None:
                    reference = current
                else:
                    pd.testing.assert_frame_equal(
                        reference,
                        current,
                    )

            expected_rows = len(
                self.plan.folds[fold_number - 1].validation_indices
            )

            self.assertEqual(len(reference), expected_rows)

    def test_each_model_has_metrics_for_every_fold(self):
        fold_metrics = self.result.metrics.loc[
            self.result.metrics["scope"] == "fold"
        ]

        expected = len(self.plan.folds) * len(BENCHMARK_MODELS)

        self.assertEqual(len(fold_metrics), expected)

    def test_prediction_values_are_finite(self):
        prediction_values = self.result.predictions[
            "predicted_return"
        ].to_numpy(dtype=float)

        self.assertTrue(
            np.isfinite(prediction_values).all()
        )

        actual_values = self.result.predictions[
            "actual_return"
        ].to_numpy(dtype=float)

        self.assertTrue(
            np.isfinite(actual_values).all()
        )

    def test_prediction_count_matches_validation_rows_and_models(self):
        validation_rows = sum(
            len(fold.validation_indices)
            for fold in self.plan.folds
        )

        expected = validation_rows * len(BENCHMARK_MODELS)

        self.assertEqual(
            len(self.result.predictions),
            expected,
        )

    def test_final_holdout_rows_are_not_predicted(self):
        holdout = self.plan.panel.iloc[
            list(self.plan.final_holdout_indices)
        ]

        holdout_keys = set(
            zip(
                holdout["instrument_key"].astype(str),
                holdout["session_date"].astype(str),
            )
        )

        prediction_keys = set(
            zip(
                self.result.predictions["instrument_key"].astype(str),
                self.result.predictions["session_date"].astype(str),
            )
        )

        self.assertTrue(
            holdout_keys.isdisjoint(prediction_keys)
        )

        self.assertFalse(self.result.holdout_used)

    def test_holdout_values_do_not_change_benchmark_results(self):
        """Changing holdout values must not affect any benchmark output."""

        primary_result = benchmark.benchmark_walk_forward(
            self.plan,
            target_column=benchmark.PRIMARY_TARGET,
            model_names=("ridge",),
        )

        changed_panel = self.plan.panel.copy()

        holdout_indices = list(
            self.plan.final_holdout_indices
        )

        # Make holdout data unusable. A benchmark that accesses these
        # rows for fitting or prediction should fail or change results.
        for column in PHASE15_FEATURE_COLUMNS:
            changed_panel.loc[holdout_indices, column] = np.inf

        changed_panel.loc[
            holdout_indices,
            benchmark.PRIMARY_TARGET,
        ] = np.nan

        changed_panel.loc[
            holdout_indices,
            benchmark.FORECAST_TARGET,
        ] = np.nan

        modified_plan = replace(
            self.plan,
            panel=changed_panel,
        )

        modified_result = benchmark.benchmark_walk_forward(
            modified_plan,
            target_column=benchmark.PRIMARY_TARGET,
            model_names=("ridge",),
        )

        pd.testing.assert_frame_equal(
            primary_result.predictions,
            modified_result.predictions,
        )

        pd.testing.assert_frame_equal(
            primary_result.metrics,
            modified_result.metrics,
        )

    def test_other_target_does_not_leak_into_primary_experiment(self):
        original = benchmark.benchmark_walk_forward(
            self.plan,
            target_column=benchmark.PRIMARY_TARGET,
            model_names=("ridge",),
        )

        changed_panel = self.plan.panel.copy()

        changed_panel[benchmark.FORECAST_TARGET] = 999.0

        modified_plan = replace(
            self.plan,
            panel=changed_panel,
        )

        changed = benchmark.benchmark_walk_forward(
            modified_plan,
            target_column=benchmark.PRIMARY_TARGET,
            model_names=("ridge",),
        )

        pd.testing.assert_frame_equal(
            original.predictions,
            changed.predictions,
        )

    def test_imputer_and_scaler_fit_on_training_rows_only(self):
        captured_estimators = []

        original_factory = benchmark._make_estimator

        def capture_estimator(model_name):
            estimator = original_factory(model_name)
            captured_estimators.append(estimator)
            return estimator

        with patch.object(
            benchmark,
            "_make_estimator",
            side_effect=capture_estimator,
        ):
            result = benchmark.benchmark_walk_forward(
                self.plan,
                target_column=benchmark.PRIMARY_TARGET,
                model_names=("ridge",),
            )

        self.assertEqual(
            len(captured_estimators),
            len(self.plan.folds),
        )

        for fold, estimator in zip(
            self.plan.folds,
            captured_estimators,
        ):
            train = self.plan.panel.iloc[
                list(fold.train_indices)
            ]

            X_train = train.loc[
                :,
                list(PHASE15_FEATURE_COLUMNS),
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            imputer = estimator.named_steps["imputer"]
            scaler = estimator.named_steps["scaler"]

            expected_medians = np.nanmedian(
                X_train.to_numpy(dtype=float),
                axis=0,
            )

            np.testing.assert_allclose(
                imputer.statistics_,
                expected_medians,
                rtol=1e-10,
                atol=1e-12,
            )

            imputed_training = imputer.transform(X_train)

            expected_scaler_mean = imputed_training.mean(axis=0)

            np.testing.assert_allclose(
                scaler.mean_,
                expected_scaler_mean,
                rtol=1e-10,
                atol=1e-12,
            )

            self.assertEqual(
                len(imputer.statistics_),
                31,
            )

        self.assertFalse(result.holdout_used)

    def test_fit_audit_confirms_training_only_preprocessing(self):
        audit = self.result.fit_audit

        self.assertEqual(
            set(audit["imputer_fit_scope"]),
            {"not_applicable", "training_only"},
        )

        self.assertEqual(
            set(audit["scaler_fit_scope"]),
            {"not_applicable", "not_used", "training_only"},
        )

        self.assertTrue(
            (audit["feature_count"] == 31).all()
        )

        self.assertFalse(
            audit["final_holdout_used"].any()
        )

    def test_zero_return_baseline_has_zero_directional_coverage(self):
        zero_predictions = self.result.predictions.loc[
            self.result.predictions["model"] == "zero_return",
            "predicted_return",
        ]

        self.assertTrue(
            (zero_predictions == 0.0).all()
        )

        pooled = self.result.metrics.loc[
            (self.result.metrics["scope"] == "pooled_oof")
            & (self.result.metrics["model"] == "zero_return")
        ].iloc[0]

        self.assertAlmostEqual(
            pooled["directional_coverage_pct"],
            0.0,
        )

        self.assertTrue(
            math.isnan(pooled["directional_accuracy_pct"])
        )

    def test_train_mean_baseline_uses_training_target_mean(self):
        for fold in self.plan.folds:
            train = self.plan.panel.iloc[
                list(fold.train_indices)
            ]

            expected_mean = train[
                benchmark.PRIMARY_TARGET
            ].mean()

            predictions = self.result.predictions.loc[
                (self.result.predictions["fold_number"] == fold.fold_number)
                & (self.result.predictions["model"] == "train_mean"),
                "predicted_return",
            ]

            self.assertTrue(
                np.allclose(
                    predictions.to_numpy(dtype=float),
                    expected_mean,
                )
            )

    def test_constant_target_returns_nan_r_squared(self):
        actual = np.full(10, 0.01)
        predicted = np.full(10, 0.01)

        scores = benchmark._calculate_metrics(
            actual,
            predicted,
        )

        self.assertAlmostEqual(scores["mae"], 0.0)
        self.assertAlmostEqual(scores["rmse"], 0.0)

        self.assertTrue(
            math.isnan(scores["r2"])
        )

    def test_constant_target_returns_nan_r_squared_with_imperfect_predictions(self):
        actual = np.full(10, 0.01, dtype=float)
        predicted = np.full(10, 0.02, dtype=float)

        scores = benchmark._calculate_metrics(
            actual,
            predicted,
        )

        self.assertGreater(scores["mae"], 0.0)
        self.assertTrue(math.isnan(scores["r2"]))

    def test_zero_predictions_are_not_counted_as_directional_successes(self):
        actual = np.array(
            [0.01, -0.02, 0.03, -0.01],
            dtype=float,
        )

        predicted = np.zeros(4, dtype=float)

        scores = benchmark._calculate_metrics(
            actual,
            predicted,
        )

        self.assertEqual(
            scores["directional_coverage_pct"],
            0.0,
        )

        self.assertTrue(
            math.isnan(scores["directional_accuracy_pct"])
        )

    def test_primary_and_forecast_experiments_are_separate(self):
        primary = benchmark.benchmark_walk_forward(
            self.plan,
            target_column=benchmark.PRIMARY_TARGET,
            model_names=("zero_return",),
        )

        forecast = benchmark.benchmark_walk_forward(
            self.plan,
            target_column=benchmark.FORECAST_TARGET,
            model_names=("zero_return",),
        )

        self.assertEqual(
            primary.target_column,
            benchmark.PRIMARY_TARGET,
        )

        self.assertEqual(
            forecast.target_column,
            benchmark.FORECAST_TARGET,
        )

        self.assertEqual(
            set(primary.predictions["target_column"]),
            {benchmark.PRIMARY_TARGET},
        )

        self.assertEqual(
            set(forecast.predictions["target_column"]),
            {benchmark.FORECAST_TARGET},
        )

        self.assertFalse(primary.holdout_used)
        self.assertFalse(forecast.holdout_used)

        self.assertFalse(
            np.allclose(
                primary.predictions["actual_return"].to_numpy(),
                forecast.predictions["actual_return"].to_numpy(),
            )
        )

    def test_unsupported_target_is_rejected(self):
        with self.assertRaises(benchmark.ModelBenchmarkError):
            benchmark.benchmark_walk_forward(
                self.plan,
                target_column="target_close_7s",
                model_names=("ridge",),
            )

    def test_duplicate_model_names_are_rejected(self):
        with self.assertRaises(benchmark.ModelBenchmarkError):
            benchmark.benchmark_walk_forward(
                self.plan,
                target_column=benchmark.PRIMARY_TARGET,
                model_names=("ridge", "ridge"),
            )


if __name__ == "__main__":
    unittest.main()
