import unittest

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.technical_features import (
    STOCK_TECHNICAL_FEATURES,
    TechnicalFeatureError,
    build_stock_technical_features,
)


def make_stock_dataset(
    periods=80,
    volume_values=None,
    flat_prices=False,
):
    dates = pd.bdate_range("2026-01-01", periods=periods)

    if flat_prices:
        close = np.full(periods, 100.0)
    else:
        close = 100.0 * (1.01 ** np.arange(periods))

    if volume_values is None:
        volume_values = np.full(periods, 1000.0)

    rows = []

    for i, (session, price) in enumerate(zip(dates, close)):
        rows.append({
            "session_date": session.strftime("%Y-%m-%d"),
            "open": float(price * 0.995),
            "high": float(price * 1.01),
            "low": float(price * 0.99),
            "close": float(price),
            "volume": float(volume_values[i]),
        })

    return {
        "instrument_key": "NSE_EQ|INFY",
        "rows": rows,
        "candle_count": len(rows),
    }


class TechnicalFeatureTests(unittest.TestCase):

    def test_returns_exactly_18_features(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        self.assertEqual(len(STOCK_TECHNICAL_FEATURES), 18)
        self.assertEqual(
            list(result.columns),
            ["session_date"] + STOCK_TECHNICAL_FEATURES,
        )
        self.assertEqual(len(result), 80)

    def test_one_day_return_calculation(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[20, "return_1d"],
            0.01,
            places=10,
        )

    def test_five_day_return_calculation(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        expected = (1.01 ** 5) - 1.0

        self.assertAlmostEqual(
            result.loc[20, "return_5d"],
            expected,
            places=10,
        )

    def test_intraday_return_calculation(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        expected = (1.0 / 0.995) - 1.0

        self.assertAlmostEqual(
            result.loc[20, "intraday_return"],
            expected,
            places=10,
        )

    def test_close_location_calculation(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[20, "close_location"],
            0.5,
            places=10,
        )

    def test_relative_volume_uses_previous_20_sessions(self):
        volumes = np.full(80, 1000.0)
        volumes[20] = 2000.0

        dataset = make_stock_dataset(volume_values=volumes)

        result = build_stock_technical_features(dataset)

        expected = np.log1p(2.0)

        self.assertAlmostEqual(
            result.loc[20, "relative_volume_log_20"],
            expected,
            places=10,
        )

    def test_today_volume_is_not_in_its_own_baseline(self):
        volumes_a = np.full(80, 1000.0)
        volumes_b = np.full(80, 1000.0)

        volumes_b[20] = 5000.0

        result_a = build_stock_technical_features(
            make_stock_dataset(volume_values=volumes_a)
        )

        result_b = build_stock_technical_features(
            make_stock_dataset(volume_values=volumes_b)
        )

        # Prior 20-session average is identical.
        # Current relative volume must reflect the current volume change.
        self.assertAlmostEqual(
            result_a.loc[20, "relative_volume_log_20"],
            np.log1p(1.0),
            places=10,
        )

        self.assertAlmostEqual(
            result_b.loc[20, "relative_volume_log_20"],
            np.log1p(5.0),
            places=10,
        )

    def test_wilder_rsi_on_consistently_rising_prices(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[20, "rsi_wilder_14"],
            1.0,
            places=10,
        )

    def test_flat_market_rsi_is_neutral(self):
        dataset = make_stock_dataset(flat_prices=True)

        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[20, "rsi_wilder_14"],
            0.5,
            places=10,
        )

    def test_wilder_atr_percentage(self):
        dataset = make_stock_dataset(flat_prices=True)

        # With constant prices of 100, high=101 and low=99,
        # true range is 2 for every session.
        for row in dataset["rows"]:
            row["high"] = 101.0
            row["low"] = 99.0
            row["open"] = 100.0
            row["close"] = 100.0

        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[20, "atr_pct_14"],
            0.02,
            places=10,
        )

    def test_flat_candle_close_location_is_neutral(self):
        dataset = make_stock_dataset(flat_prices=True)

        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[30, "close_location"],
            0.5,
            places=10,
        )

    def test_warmup_values_are_not_filled_with_zero(self):
        dataset = make_stock_dataset()

        result = build_stock_technical_features(dataset)

        self.assertTrue(
            result.iloc[:19][STOCK_TECHNICAL_FEATURES]
            .isna().any(axis=1).all()
        )

        self.assertTrue(
            result.iloc[49][STOCK_TECHNICAL_FEATURES]
            .notna().all()
        )

    def test_trend_efficiency_is_bounded(self):
        dataset = make_stock_dataset()
        result = build_stock_technical_features(dataset)

        values = result["trend_efficiency_20d"].dropna().to_numpy()

        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= -1.0 - 1e-12).all())
        self.assertTrue((values <= 1.0 + 1e-12).all())

        # A continuously rising price path has efficiency close to 1.
        self.assertAlmostEqual(
            result.loc[20, "trend_efficiency_20d"],
            1.0,
            places=10,
        )

    def test_flat_market_trend_efficiency_is_zero(self):
        dataset = make_stock_dataset(flat_prices=True)
        result = build_stock_technical_features(dataset)

        self.assertAlmostEqual(
            result.loc[30, "trend_efficiency_20d"],
            0.0,
            places=10,
        )

    def test_no_future_data_changes_past_features(self):
        original_data = make_stock_dataset()

        original = build_stock_technical_features(original_data)

        changed_data = make_stock_dataset()

        # Modify only observations after position 59.
        for row in changed_data["rows"][60:]:
            row["open"] *= 2.0
            row["high"] *= 2.0
            row["low"] *= 2.0
            row["close"] *= 2.0
            row["volume"] *= 3.0

        modified = build_stock_technical_features(changed_data)

        pd.testing.assert_frame_equal(
            original.iloc[:60].reset_index(drop=True),
            modified.iloc[:60].reset_index(drop=True),
            check_exact=True,
        )

    def test_rejects_duplicate_sessions(self):
        dataset = make_stock_dataset()

        dataset["rows"][20]["session_date"] = (
            dataset["rows"][19]["session_date"]
        )

        with self.assertRaisesRegex(
            TechnicalFeatureError,
            "Duplicate",
        ):
            build_stock_technical_features(dataset)

    def test_rejects_unsorted_sessions(self):
        dataset = make_stock_dataset()

        dataset["rows"].reverse()

        with self.assertRaisesRegex(
            TechnicalFeatureError,
            "sorted oldest to newest",
        ):
            build_stock_technical_features(dataset)

    def test_rejects_invalid_ohlc(self):
        dataset = make_stock_dataset()

        dataset["rows"][20]["high"] = 1.0

        with self.assertRaisesRegex(
            TechnicalFeatureError,
            "Invalid high prices",
        ):
            build_stock_technical_features(dataset)

    def test_rejects_negative_volume(self):
        dataset = make_stock_dataset()

        dataset["rows"][20]["volume"] = -100.0

        with self.assertRaisesRegex(
            TechnicalFeatureError,
            "Volume cannot be negative",
        ):
            build_stock_technical_features(dataset)

    def test_rejects_insufficient_history(self):
        dataset = make_stock_dataset(periods=40)

        with self.assertRaisesRegex(
            TechnicalFeatureError,
            "At least 60",
        ):
            build_stock_technical_features(dataset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
