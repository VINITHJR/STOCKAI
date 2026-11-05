import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.feature_builder import (
    PHASE15_FEATURE_COLUMNS,
    FeatureBuilderError,
    build_phase15_features,
)

from backend.app.ml.phase15.features.market_context import MarketContextError


def make_dataset(key, close_values, dates, volumes=None):
    if volumes is None:
        volumes = np.full(len(close_values), 100000.0)

    rows = []

    for session, close, volume in zip(dates, close_values, volumes):
        close = float(close)

        rows.append({
            "instrument_key": key,
            "session_date": session.strftime("%Y-%m-%d"),
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": float(volume),
        })

    return {
        "instrument_key": key,
        "rows": rows,
        "candle_count": len(rows),
    }


def make_all_datasets(periods=100):
    dates = pd.bdate_range("2026-01-01", periods=periods)
    x = np.arange(periods, dtype=np.float64)

    stock = make_dataset(
        "NSE_EQ|INFY",
        100 * (1.01 ** x) * (1 + 0.002 * np.sin(x)),
        dates,
    )

    nifty = make_dataset(
        "NSE_INDEX|Nifty 50",
        25000 * (1.005 ** x) * (1 + 0.001 * np.sin(x)),
        dates,
    )

    vix = make_dataset(
        "NSE_INDEX|India VIX",
        15 + 0.02 * x + 0.5 * np.sin(x / 5),
        dates,
    )

    return stock, nifty, vix


class FeatureBuilderTests(unittest.TestCase):

    def test_builds_exactly_31_unique_features(self):
        stock, nifty, vix = make_all_datasets()

        result = build_phase15_features(stock, nifty, vix)

        self.assertEqual(len(PHASE15_FEATURE_COLUMNS), 31)
        self.assertEqual(len(set(PHASE15_FEATURE_COLUMNS)), 31)

        self.assertEqual(
            list(result.columns),
            ["session_date"] + list(PHASE15_FEATURE_COLUMNS),
        )

    def test_preserves_all_input_sessions(self):
        stock, nifty, vix = make_all_datasets()

        result = build_phase15_features(stock, nifty, vix)

        self.assertEqual(len(result), len(stock["rows"]))

        self.assertEqual(
            result["session_date"].tolist(),
            [row["session_date"] for row in stock["rows"]],
        )

    def test_output_is_chronologically_sorted(self):
        stock, nifty, vix = make_all_datasets()

        result = build_phase15_features(stock, nifty, vix)

        dates = pd.to_datetime(result["session_date"])

        self.assertTrue(dates.is_monotonic_increasing)
        self.assertFalse(dates.duplicated().any())

    def test_warmup_nans_are_preserved(self):
        stock, nifty, vix = make_all_datasets()

        result = build_phase15_features(stock, nifty, vix)

        numeric = result[list(PHASE15_FEATURE_COLUMNS)]

        self.assertTrue(numeric.iloc[:49].isna().any().any())
        self.assertTrue(numeric.iloc[49:].notna().all().all())

    def test_no_infinite_feature_values(self):
        stock, nifty, vix = make_all_datasets()

        result = build_phase15_features(stock, nifty, vix)

        values = result[list(PHASE15_FEATURE_COLUMNS)].to_numpy(
            dtype=np.float64
        )

        self.assertFalse(np.isinf(values).any())

    def test_future_data_does_not_change_past_combined_features(self):
        stock, nifty, vix = make_all_datasets()

        original = build_phase15_features(stock, nifty, vix)

        changed_stock = deepcopy(stock)
        changed_nifty = deepcopy(nifty)
        changed_vix = deepcopy(vix)

        for dataset in (
            changed_stock,
            changed_nifty,
            changed_vix,
        ):
            for row in dataset["rows"][70:]:
                row["open"] *= 1.8
                row["high"] *= 1.8
                row["low"] *= 1.8
                row["close"] *= 1.8
                row["volume"] *= 2.0

        modified = build_phase15_features(
            changed_stock,
            changed_nifty,
            changed_vix,
        )

        pd.testing.assert_frame_equal(
            original.iloc[:70].reset_index(drop=True),
            modified.iloc[:70].reset_index(drop=True),
            check_exact=True,
        )

    def test_rejects_stock_and_vix_calendar_mismatch(self):
        stock, nifty, vix = make_all_datasets()

        vix["rows"].pop(30)

        with self.assertRaises(MarketContextError):
            build_phase15_features(stock, nifty, vix)

    def test_rejects_stock_and_nifty_calendar_mismatch(self):
        stock, nifty, vix = make_all_datasets()

        nifty["rows"].pop(30)

        with self.assertRaises(MarketContextError):
            build_phase15_features(stock, nifty, vix)

    def test_output_has_unique_feature_names(self):
        stock, nifty, vix = make_all_datasets()

        result = build_phase15_features(stock, nifty, vix)

        # The output includes session_date plus 31 numeric features.
        self.assertEqual(len(result.columns), 32)

        # No duplicate names are allowed in the actual output.
        self.assertEqual(
            len(result.columns),
            len(set(result.columns)),
        )

        self.assertEqual(
            len(PHASE15_FEATURE_COLUMNS),
            len(set(PHASE15_FEATURE_COLUMNS)),
        )

    def test_rejects_invalid_market_data(self):
        stock, nifty, vix = make_all_datasets()

        nifty["rows"][50]["close"] = float("nan")

        with self.assertRaises(MarketContextError):
            build_phase15_features(stock, nifty, vix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
