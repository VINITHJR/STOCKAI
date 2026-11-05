import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.market_context import (
    MARKET_CONTEXT_FEATURES,
    MarketContextError,
    build_market_context_features,
)


def make_dataset(key, close_values, dates):
    rows = []

    for session_date, close in zip(dates, close_values):
        close = float(close)

        rows.append({
            "instrument_key": key,
            "session_date": session_date.strftime("%Y-%m-%d"),
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100000,
        })

    return {
        "instrument_key": key,
        "rows": rows,
        "candle_count": len(rows),
    }


def make_test_datasets():
    dates = pd.bdate_range("2026-01-01", periods=80)

    stock_close = 100 * (1.01 ** np.arange(80))
    nifty_close = 25000 * (1.005 ** np.arange(80))
    vix_close = 14 + 0.05 * np.arange(80)

    stock = make_dataset("NSE_EQ|INFY", stock_close, dates)
    nifty = make_dataset("NSE_INDEX|Nifty 50", nifty_close, dates)
    vix = make_dataset("NSE_INDEX|India VIX", vix_close, dates)

    return stock, nifty, vix


class MarketContextTests(unittest.TestCase):

    def test_returns_expected_13_features(self):
        stock, nifty, vix = make_test_datasets()

        result = build_market_context_features(stock, nifty, vix)

        self.assertEqual(len(MARKET_CONTEXT_FEATURES), 13)
        self.assertEqual(
            list(result.columns),
            ["session_date"] + MARKET_CONTEXT_FEATURES,
        )
        self.assertEqual(len(result), 80)

    def test_warmup_rows_remain_nan(self):
        stock, nifty, vix = make_test_datasets()

        result = build_market_context_features(stock, nifty, vix)

        # The 50-session SMA requires 50 observations.
        self.assertTrue(
            result.iloc[:49][MARKET_CONTEXT_FEATURES]
            .isna().any(axis=1).all()
        )

        # From the 50th observation onwards all features are available.
        self.assertTrue(
            result.iloc[49:][MARKET_CONTEXT_FEATURES]
            .notna().all().all()
        )

    def test_relative_return_calculation(self):
        stock, nifty, vix = make_test_datasets()

        result = build_market_context_features(stock, nifty, vix)

        expected = (1.01 / 1.005) - 1.0
        actual = result.loc[1, "relative_return_1d"]

        self.assertAlmostEqual(actual, expected, places=10)

    def test_nifty_one_day_return(self):
        stock, nifty, vix = make_test_datasets()

        result = build_market_context_features(stock, nifty, vix)

        self.assertAlmostEqual(
            result.loc[1, "nifty_return_1d"],
            0.005,
            places=10,
        )

    def test_future_data_does_not_change_past_features(self):
        stock, nifty, vix = make_test_datasets()

        original = build_market_context_features(stock, nifty, vix)

        changed_stock = deepcopy(stock)
        changed_nifty = deepcopy(nifty)
        changed_vix = deepcopy(vix)

        # Modify observations strictly after position 59.
        for dataset in (
            changed_stock,
            changed_nifty,
            changed_vix,
        ):
            for row in dataset["rows"][60:]:
                row["close"] *= 2.0

        modified = build_market_context_features(
            changed_stock,
            changed_nifty,
            changed_vix,
        )

        pd.testing.assert_frame_equal(
            original.iloc[:60].reset_index(drop=True),
            modified.iloc[:60].reset_index(drop=True),
            check_exact=True,
        )

    def test_rejects_mismatched_vix_calendar(self):
        stock, nifty, vix = make_test_datasets()

        vix["rows"].pop(30)

        with self.assertRaisesRegex(
            MarketContextError,
            "India VIX calendars",
        ):
            build_market_context_features(stock, nifty, vix)

    def test_rejects_mismatched_nifty_calendar(self):
        stock, nifty, vix = make_test_datasets()

        nifty["rows"].pop(30)

        with self.assertRaisesRegex(
            MarketContextError,
            "NIFTY calendars",
        ):
            build_market_context_features(stock, nifty, vix)

    def test_rejects_unsorted_stock_sessions(self):
        stock, nifty, vix = make_test_datasets()

        stock["rows"].reverse()

        with self.assertRaisesRegex(
            MarketContextError,
            "sorted oldest to newest",
        ):
            build_market_context_features(stock, nifty, vix)

    def test_rejects_duplicate_sessions(self):
        stock, nifty, vix = make_test_datasets()

        stock["rows"][20]["session_date"] = (
            stock["rows"][19]["session_date"]
        )

        with self.assertRaisesRegex(
            MarketContextError,
            "duplicate trading sessions",
        ):
            build_market_context_features(stock, nifty, vix)

    def test_rejects_invalid_close_price(self):
        stock, nifty, vix = make_test_datasets()

        stock["rows"][20]["close"] = float("nan")

        with self.assertRaisesRegex(
            MarketContextError,
            "NaN or infinity",
        ):
            build_market_context_features(stock, nifty, vix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
