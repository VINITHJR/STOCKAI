from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from backend.app.backtest.historical_data import _normalize, validate_normalized_candles
from backend.app.backtest.cache import HistoricalDataCache


def row(ts, o=100, h=103, low=99, c=101, volume=1000, oi=0):
    return [ts, o, h, low, c, volume, oi]


def good_candle(ts="2025-01-01T09:15:00+05:30"):
    return {
        "timestamp": ts,
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000,
        "open_interest": 0.0,
    }


def test_normalize_sorts_and_deduplicates_keep_last_valid_row():
    raw = {"data": {"candles": [
        row("2025-01-02T09:15:00+05:30", c=102),
        row("2025-01-01T09:15:00+05:30", c=101),
        row("2025-01-01T09:15:00+05:30", c=102),
    ]}}
    result = _normalize(raw)
    assert len(result) == 2
    assert result[0]["timestamp"].startswith("2025-01-01")
    assert result[0]["close"] == 102.0
    assert result[1]["timestamp"].startswith("2025-01-02")


def test_normalize_discards_invalid_rows():
    raw = {"data": {"candles": [
        row("not-a-timestamp"),
        row("2025-01-01T09:15:00+05:30", c=float("nan")),
        row("2025-01-02T09:15:00+05:30", h=95, c=101),
        row("2025-01-03T09:15:00+05:30", volume=-1),
        row("2025-01-04T09:15:00+05:30", o=0),
        row("2025-01-05T09:15:00+05:30", volume=1.5),
        row("2025-01-06T09:15:00+05:30"),
    ]}}
    result = _normalize(raw)
    assert len(result) == 1
    assert result[0]["timestamp"].startswith("2025-01-06")


@pytest.mark.parametrize("changes", [
    {"timestamp": "not-a-timestamp"},
    {"open": float("nan")},
    {"close": float("inf")},
    {"open": 0},
    {"high": 95},
    {"low": 105},
    {"low": 104, "high": 103},
    {"volume": -1},
    {"volume": float("nan")},
    {"volume": 1.5},
    {"open_interest": -1},
])
def test_strict_validator_rejects_invalid_normalized_candle(changes):
    candle = good_candle()
    candle.update(changes)
    with pytest.raises(ValueError):
        validate_normalized_candles([candle])


def test_strict_validator_rejects_duplicate_timestamps():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_normalized_candles(
            [good_candle(), good_candle()], reject_duplicates=True
        )


def test_cache_round_trip_and_rejects_corrupt_cache(tmp_path: Path):
    cache = HistoricalDataCache(str(tmp_path))
    args = ("NSE_EQ|TEST", "days", "1", "2025-01-01", "2025-01-02")
    candles = [good_candle(), good_candle("2025-01-02T09:15:00+05:30")]

    cache.set(*args, candles)
    assert cache.get(*args) is not None
    assert len(cache.get(*args)) == 2

    cache_file = cache._path(*args)
    payload = json.loads(cache_file.read_text())
    payload["candles"][0]["close"] = float("nan")
    cache_file.write_text(json.dumps(payload))
    assert cache.get(*args) is None


def test_cache_set_refuses_invalid_candle(tmp_path: Path):
    cache = HistoricalDataCache(str(tmp_path))
    args = ("NSE_EQ|TEST", "days", "1", "2025-01-01", "2025-01-01")
    invalid = good_candle()
    invalid["volume"] = -3
    with pytest.raises(ValueError):
        cache.set(*args, [invalid])
