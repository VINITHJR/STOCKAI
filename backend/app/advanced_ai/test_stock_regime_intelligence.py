from __future__ import annotations

import math

from stock_regime_intelligence import (
    StockRegimeIntelligence,
)
from stock_regime_models import (
    StockRegimeInput,
)


def make_bullish_candles(count: int = 120):
    candles = []

    price = 100.0

    for i in range(count):

        close = price + 1.0
        open_price = price
        high = close + 0.5
        low = open_price - 0.2
        volume = 1000 + i * 2

        candles.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        price = close

    return candles


def make_bearish_candles(count: int = 120):
    candles = []

    price = 200.0

    for i in range(count):

        close = price - 1.0
        open_price = price
        high = open_price + 0.2
        low = close - 0.5
        volume = 1000 + i * 2

        candles.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        price = close

    return candles


def make_flat_candles(count: int = 120):
    return [
        {
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
            "volume": 1000,
        }
        for _ in range(count)
    ]


def main():

    engine = StockRegimeIntelligence()

    bullish = engine.analyze(
        StockRegimeInput(
            instrument_key="TEST|BULL",
            candles=make_bullish_candles(),
        )
    )

    assert bullish.stock_direction == "bullish"
    assert bullish.stock_regime in {
        "bullish_trend",
        "strong_bullish_trend",
    }

    bearish = engine.analyze(
        StockRegimeInput(
            instrument_key="TEST|BEAR",
            candles=make_bearish_candles(),
        )
    )

    assert bearish.stock_direction == "bearish"
    assert bearish.stock_regime in {
        "bearish_trend",
        "strong_bearish_trend",
    }

    flat = engine.analyze(
        StockRegimeInput(
            instrument_key="TEST|FLAT",
            candles=make_flat_candles(),
        )
    )

    assert flat.stock_direction == "neutral"
    assert flat.stock_regime in {
        "sideways",
        "volatile_sideways",
    }

    # Determinism
    result_a = engine.analyze(
        StockRegimeInput(
            instrument_key="TEST|DET",
            candles=make_bullish_candles(),
        )
    )

    result_b = engine.analyze(
        StockRegimeInput(
            instrument_key="TEST|DET",
            candles=make_bullish_candles(),
        )
    )

    assert result_a.model_dump() == result_b.model_dump()

    # Insufficient data
    try:
        engine.analyze(
            StockRegimeInput(
                instrument_key="TEST|SHORT",
                candles=make_bullish_candles(20),
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for insufficient candles"
        )

    # Confidence must remain valid
    assert 0.0 <= bullish.stock_regime_confidence <= 1.0
    assert 0.0 <= bearish.stock_regime_confidence <= 1.0
    assert 0.0 <= flat.stock_regime_confidence <= 1.0

    assert math.isfinite(
        bullish.stock_regime_confidence
    )

    print("PHASE14_2_STOCK_REGIME_TESTS_OK")


if __name__ == "__main__":
    main()