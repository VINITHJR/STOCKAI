import unittest

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.feature_builder import (
    PHASE15_FEATURE_COLUMNS,
)
from backend.app.ml.phase15.targets.target_builder import (
    TargetBuilderError,
    build_return_targets,
    build_phase15_training_frame,
)


INSTRUMENT_KEY = "NSE_EQ|INE009A01021"


def make_stock_data(rows=15):
    """Build synthetic data using the historical loader's return structure."""
    dates = pd.bdate_range("2025-01-01", periods=rows)
    candles = []

    for i, session in enumerate(dates):
        close = 100.0 + (i * 2.0)
        open_price = 99.0 + (i * 1.5)
        session_date = session.strftime("%Y-%m-%d")

        candles.append({
            "instrument_key": INSTRUMENT_KEY,
            "timestamp": f"{session_date}T09:15:00+05:30",
            "timestamp_ist": f"{session_date}T15:30:00+05:30",
            "session_date": session_date,
            "open": open_price,
            "high": max(open_price, close) + 1.0,
            "low": min(open_price, close) - 1.0,
            "close": close,
            "volume": 1000 + (i * 10),
        })

    return {
        "instrument_key": INSTRUMENT_KEY,
        "start_date": candles[0]["session_date"],
        "end_date": candles[-1]["session_date"],
        "candle_count": len(candles),
        "first_session": candles[0]["session_date"],
        "last_session": candles[-1]["session_date"],
        "ordering": "ascending",
        "rows": candles,
    }


def make_features(stock_data):
    """Create a DataFrame using the exact Phase 15 feature schema."""
    rows = stock_data["rows"]

    frame = pd.DataFrame({
        "session_date": [
            row["session_date"] for row in rows
        ]
    })

    for index, column in enumerate(PHASE15_FEATURE_COLUMNS):
        frame[column] = 0.1 + (index * 0.01)

    return frame


class TestReturnTargets(unittest.TestCase):

    def setUp(self):
        self.stock_data = make_stock_data(15)
        self.rows = self.stock_data["rows"]

        self.targets = build_return_targets(
            self.stock_data,
            horizons=(1, 3, 7),
        )

    def test_output_has_one_row_per_input_session(self):
        self.assertEqual(len(self.targets), len(self.rows))

    def test_required_target_columns_exist(self):
        required = {
            "session_date",
            "base_close",
            "next_session_open",
            "next_session_date",
            "max_horizon_target_session",
            "all_targets_available",
        }

        for horizon in (1, 3, 7):
            required.update({
                f"close_to_close_return_{horizon}s",
                f"next_open_to_close_return_{horizon}s",
                f"target_close_{horizon}s",
                f"target_exit_session_{horizon}s",
            })

        self.assertTrue(
            required.issubset(set(self.targets.columns))
        )

    def test_one_session_close_to_close_formula(self):
        expected = (
            self.rows[1]["close"] / self.rows[0]["close"]
        ) - 1.0

        actual = self.targets.loc[
            0, "close_to_close_return_1s"
        ]

        self.assertAlmostEqual(actual, expected, places=12)

    def test_one_session_execution_formula_uses_next_open(self):
        expected = (
            self.rows[1]["close"] / self.rows[1]["open"]
        ) - 1.0

        actual = self.targets.loc[
            0, "next_open_to_close_return_1s"
        ]

        self.assertAlmostEqual(actual, expected, places=12)

    def test_three_session_forecast_formula(self):
        expected = (
            self.rows[3]["close"] / self.rows[0]["close"]
        ) - 1.0

        actual = self.targets.loc[
            0, "close_to_close_return_3s"
        ]

        self.assertAlmostEqual(actual, expected, places=12)

    def test_three_session_execution_formula(self):
        expected = (
            self.rows[3]["close"] / self.rows[1]["open"]
        ) - 1.0

        actual = self.targets.loc[
            0, "next_open_to_close_return_3s"
        ]

        self.assertAlmostEqual(actual, expected, places=12)

    def test_seven_session_forecast_formula(self):
        expected = (
            self.rows[7]["close"] / self.rows[0]["close"]
        ) - 1.0

        actual = self.targets.loc[
            0, "close_to_close_return_7s"
        ]

        self.assertAlmostEqual(actual, expected, places=12)

    def test_seven_session_execution_formula(self):
        expected = (
            self.rows[7]["close"] / self.rows[1]["open"]
        ) - 1.0

        actual = self.targets.loc[
            0, "next_open_to_close_return_7s"
        ]

        self.assertAlmostEqual(actual, expected, places=12)

    def test_next_session_open_is_taken_from_next_row(self):
        self.assertAlmostEqual(
            self.targets.loc[0, "next_session_open"],
            self.rows[1]["open"],
            places=12,
        )

    def test_final_seven_rows_do_not_have_all_targets(self):
        self.assertTrue(
            self.targets["all_targets_available"].iloc[:-7].all()
        )

        self.assertFalse(
            self.targets["all_targets_available"].iloc[-7:].any()
        )

    def test_unavailable_seven_session_targets_are_nan(self):
        columns = (
            "close_to_close_return_7s",
            "next_open_to_close_return_7s",
            "target_close_7s",
        )

        for column in columns:
            with self.subTest(column=column):
                self.assertTrue(
                    self.targets[column].iloc[-7:].isna().all()
                )

    def test_first_row_exit_session_matches_seven_session_target_date(self):
        expected_date = self.rows[7]["session_date"]
        actual_date = self.targets.loc[
            0, "target_exit_session_7s"
        ]

        self.assertEqual(
            pd.Timestamp(actual_date).strftime("%Y-%m-%d"),
            expected_date,
        )

    def test_duplicate_session_dates_are_rejected(self):
        bad_data = make_stock_data(10)
        bad_data["rows"][3]["session_date"] = (
            bad_data["rows"][2]["session_date"]
        )

        with self.assertRaises(TargetBuilderError):
            build_return_targets(bad_data, horizons=(1, 3, 7))

    def test_unsorted_session_dates_are_rejected(self):
        bad_data = make_stock_data(10)
        rows = bad_data["rows"]

        rows[3], rows[4] = rows[4], rows[3]

        with self.assertRaises(TargetBuilderError):
            build_return_targets(bad_data, horizons=(1, 3, 7))

    def test_non_positive_prices_are_rejected(self):
        bad_data = make_stock_data(10)
        bad_data["rows"][4]["close"] = 0.0

        with self.assertRaises(TargetBuilderError):
            build_return_targets(bad_data, horizons=(1, 3, 7))

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(TargetBuilderError):
            build_return_targets(
                self.stock_data,
                horizons=(1, 0, 7),
            )

    def test_duplicate_horizons_are_rejected(self):
        with self.assertRaises(TargetBuilderError):
            build_return_targets(
                self.stock_data,
                horizons=(1, 3, 3),
            )


class TestPhase15TrainingFrame(unittest.TestCase):

    def setUp(self):
        self.stock_data = make_stock_data(15)
        self.features = make_features(self.stock_data)

    def test_training_frame_has_expected_number_of_rows(self):
        frame = build_phase15_training_frame(
            self.features,
            self.stock_data,
            horizons=(1, 3, 7),
        )

        self.assertEqual(len(frame), len(self.stock_data["rows"]))

    def test_training_frame_preserves_all_feature_columns(self):
        frame = build_phase15_training_frame(
            self.features,
            self.stock_data,
            horizons=(1, 3, 7),
        )

        for column in PHASE15_FEATURE_COLUMNS:
            self.assertIn(column, frame.columns)

    def test_training_eligibility_requires_features_and_targets(self):
        features = self.features.copy()

        features.loc[0, PHASE15_FEATURE_COLUMNS[0]] = np.nan

        frame = build_phase15_training_frame(
            features,
            self.stock_data,
            horizons=(1, 3, 7),
        )

        self.assertFalse(bool(frame.loc[0, "feature_ready"]))
        self.assertFalse(bool(frame.loc[0, "training_eligible"]))

        self.assertFalse(
            frame["targets_ready"].iloc[-7:].any()
        )

        self.assertFalse(
            frame["training_eligible"].iloc[-7:].any()
        )

    def test_training_eligible_is_exact_intersection(self):
        frame = build_phase15_training_frame(
            self.features,
            self.stock_data,
            horizons=(1, 3, 7),
        )

        expected = frame["feature_ready"] & frame["targets_ready"]

        pd.testing.assert_series_equal(
            frame["training_eligible"].astype(bool),
            expected.astype(bool),
            check_names=False,
        )

    def test_mismatched_feature_dates_are_rejected(self):
        features = self.features.copy()
        features.loc[0, "session_date"] = "2024-12-31"

        with self.assertRaises(TargetBuilderError):
            build_phase15_training_frame(
                features,
                self.stock_data,
                horizons=(1, 3, 7),
            )

    def test_missing_feature_column_is_rejected(self):
        features = self.features.drop(
            columns=[PHASE15_FEATURE_COLUMNS[0]]
        )

        with self.assertRaises(TargetBuilderError):
            build_phase15_training_frame(
                features,
                self.stock_data,
                horizons=(1, 3, 7),
            )


if __name__ == "__main__":
    unittest.main()
