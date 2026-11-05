import unittest

from backend.app.ml.phase15.data.historical_loader import (
    HistoricalDataError,
    compare_session_calendars,
    fetch_daily_candles,
    validate_daily_candles,
)


KEY = "NSE_EQ|INE009A01021"


def candle(timestamp, open_=100, high=105, low=98, close=103, volume=1000):
    return [timestamp, open_, high, low, close, volume, 0]


class HistoricalLoaderTests(unittest.TestCase):

    def test_sorts_candles_oldest_to_newest(self):
        data = [
            candle("2026-10-02T00:00:00+05:30"),
            candle("2026-10-01T00:00:00+05:30"),
        ]

        result = validate_daily_candles(
            KEY, data, "2026-10-01", "2026-10-02"
        )

        self.assertEqual(result["ordering"], "ascending")
        self.assertEqual(
            [r["session_date"] for r in result["rows"]],
            ["2026-10-01", "2026-10-02"],
        )

    def test_converts_utc_timestamp_to_ist_session_date(self):
        data = [
            candle("2026-10-01T18:30:00Z")
        ]

        result = validate_daily_candles(
            KEY, data, "2026-10-02", "2026-10-02"
        )

        self.assertEqual(
            result["rows"][0]["session_date"], "2026-10-02"
        )
        self.assertIn("+05:30", result["rows"][0]["timestamp_ist"])

    def test_rejects_duplicate_sessions(self):
        data = [
            candle("2026-10-01T09:15:00+05:30"),
            candle("2026-10-01T15:30:00+05:30"),
        ]

        with self.assertRaisesRegex(HistoricalDataError, "duplicate"):
            validate_daily_candles(
                KEY, data, "2026-10-01", "2026-10-01"
            )

    def test_rejects_timezone_naive_timestamp(self):
        data = [candle("2026-10-01T00:00:00")]

        with self.assertRaisesRegex(HistoricalDataError, "Timezone missing"):
            validate_daily_candles(
                KEY, data, "2026-10-01", "2026-10-01"
            )

    def test_rejects_invalid_high_price(self):
        data = [
            candle(
                "2026-10-01T00:00:00+05:30",
                open_=100, high=102, low=98, close=104,
            )
        ]

        with self.assertRaisesRegex(HistoricalDataError, "invalid high"):
            validate_daily_candles(
                KEY, data, "2026-10-01", "2026-10-01"
            )

    def test_rejects_negative_volume(self):
        data = [
            candle("2026-10-01T00:00:00+05:30", volume=-10)
        ]

        with self.assertRaisesRegex(HistoricalDataError, "negative volume"):
            validate_daily_candles(
                KEY, data, "2026-10-01", "2026-10-01"
            )

    def test_rejects_non_finite_prices(self):
        data = [
            candle("2026-10-01T00:00:00+05:30", close=float("nan"))
        ]

        with self.assertRaisesRegex(HistoricalDataError, "non-finite"):
            validate_daily_candles(
                KEY, data, "2026-10-01", "2026-10-01"
            )

    def test_rejects_candle_outside_requested_range(self):
        data = [candle("2026-10-02T00:00:00+05:30")]

        with self.assertRaisesRegex(HistoricalDataError, "outside"):
            validate_daily_candles(
                KEY, data, "2026-10-01", "2026-10-01"
            )

    def test_rejects_empty_candles(self):
        with self.assertRaisesRegex(HistoricalDataError, "No historical"):
            validate_daily_candles(
                KEY, [], "2026-10-01", "2026-10-02"
            )

    def test_compares_session_calendars(self):
        reference = validate_daily_candles(
            "NSE_INDEX|Nifty 50",
            [
                candle("2026-10-01T00:00:00+05:30"),
                candle("2026-10-02T00:00:00+05:30"),
            ],
            "2026-10-01",
            "2026-10-02",
        )

        candidate = validate_daily_candles(
            KEY,
            [
                candle("2026-10-01T00:00:00+05:30"),
                candle("2026-10-03T00:00:00+05:30"),
            ],
            "2026-10-01",
            "2026-10-03",
        )

        result = compare_session_calendars(reference, candidate)

        self.assertEqual(result["missing_sessions"], ["2026-10-02"])
        self.assertEqual(result["extra_sessions"], ["2026-10-03"])
        self.assertFalse(result["exact_match"])

    def test_fetch_function_uses_injected_api_client(self):
        received_arguments = {}

        def fake_fetch(**kwargs):
            received_arguments.update(kwargs)
            return {
                "status": "success",
                "data": {
                    "candles": [
                        candle("2026-10-01T00:00:00+05:30")
                    ]
                },
            }

        result = fetch_daily_candles(
            KEY,
            "2026-10-01",
            "2026-10-01",
            fetch_function=fake_fetch,
        )

        self.assertEqual(result["candle_count"], 1)
        self.assertEqual(received_arguments["instrument_key"], KEY)
        self.assertEqual(received_arguments["unit"], "days")
        self.assertEqual(received_arguments["interval"], "1")
        self.assertEqual(received_arguments["from_date"], "2026-10-01")
        self.assertEqual(received_arguments["to_date"], "2026-10-01")

    def test_fetch_rejects_unsuccessful_api_response(self):
        def fake_fetch(**kwargs):
            return {"status": "error", "data": {"candles": []}}

        with self.assertRaisesRegex(HistoricalDataError, "unsuccessful status"):
            fetch_daily_candles(
                KEY,
                "2026-10-01",
                "2026-10-01",
                fetch_function=fake_fetch,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
