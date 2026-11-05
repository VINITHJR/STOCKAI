import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("technical_analysis_intelligence.py")
spec = importlib.util.spec_from_file_location("technical_analysis_intelligence", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
TA = module.TechnicalAnalysisIntelligence


def candles(n: int, trend: float, volume: float = 100_000.0):
    rows = []
    price = 100.0
    for _ in range(n):
        previous = price
        price += trend
        rows.append({
            "open": previous,
            "high": max(previous, price) + 1.0,
            "low": min(previous, price) - 1.0,
            "close": price,
            "volume": volume,
        })
    return rows


# 1. Strong rising series must be bullish and deterministic.
up = TA.analyze(candles(100, 1.0))
up_again = TA.analyze(candles(100, 1.0))
assert up == up_again
assert up["signal"] == "bullish"
assert 0.0 <= up["confidence"] <= 1.0
assert up["directional_score"] > 0
assert up["indicators"]["rsi_14"] == 100.0
assert up["indicators"]["adx_14"] is not None

# 2. Strong falling series must be bearish.
down = TA.analyze(candles(100, -1.0))
assert down["signal"] == "bearish"
assert down["directional_score"] < 0
assert down["indicators"]["rsi_14"] == 0.0

# 3. Flat series must remain neutral with low confidence.
flat = TA.analyze(candles(100, 0.0))
assert flat["signal"] == "neutral"
assert flat["confidence"] < 0.5
assert abs(flat["directional_score"]) < 1e-12

# 4. Invalid/non-finite rows are ignored; valid data still works.
rows = candles(100, 0.5)
rows.insert(0, {"open": float("nan"), "high": 1, "low": 1, "close": 1, "volume": 1})
result = TA.analyze(rows)
assert result["candle_count"] == 100

# 5. Insufficient usable candles must fail explicitly.
try:
    TA.analyze(candles(59, 1.0))
except ValueError:
    pass
else:
    raise AssertionError("Expected ValueError for insufficient candles")

print("STEP1_TECHNICAL_TESTS_OK")
