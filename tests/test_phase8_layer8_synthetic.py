"""Synthetic regression tests for Phase 8 Layer 8; no API/network calls."""
from datetime import date, timedelta
from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.layer8 import BacktestPeriod, RobustnessBacktestRequest, _run_period
from backend.app.backtest.models import BacktestRequest


def make_candles(prices, start):
    rows = []
    for i, price in enumerate(prices):
        d = start + timedelta(days=i)
        p = float(price)
        rows.append({"timestamp": d.isoformat(), "open": p, "high": p + 1,
                     "low": max(0.01, p - 1), "close": p, "volume": 1000})
    return rows


prices = [100,99,98,99,101,104,108,112,110,106,102,99,103,108,114,118,115,111,107,104,
          101,99,102,106,111,116,119,117,113,109]
rows = make_candles(prices, date(2025, 1, 1))
base = BacktestRequest(instrument_key="SYNTHETIC", unit="days", interval="1",
    from_date="2025-01-01", to_date="2025-01-30", initial_capital=100000,
    quantity=1, short_window=2, long_window=3)
result = BacktestEngine(base).run(rows)
assert len(result["equity_curve"]) == len(rows)
assert abs(result["final_capital"] - result["initial_capital"] - result["total_pnl"]) < 0.02
assert abs(result["equity_curve"][-1]["equity"] - result["final_capital"]) < 0.02
assert all(t["execution_cost"] >= 0 for t in result["trades"])
print("PASS: engine equity curve and final capital reconcile")

request = RobustnessBacktestRequest(instrument_key="SYNTHETIC", periods=[
    {"from_date":"2025-01-11","to_date":"2025-01-20"},
    {"from_date":"2025-01-21","to_date":"2025-01-30"}], unit="days", interval="1",
    initial_capital=100000, quantity=1, short_window=2, long_window=3)
for period in [BacktestPeriod(from_date="2025-01-11", to_date="2025-01-20"),
               BacktestPeriod(from_date="2025-01-21", to_date="2025-01-30")]:
    item = _run_period(request, period, rows)
    assert item["candle_count"] == 10
    assert item["warmup_candle_count"] >= 2
    assert "cagr_percent" in item and "sharpe_ratio" in item
    assert abs(item["final_capital"] - 100000 - item["total_pnl"]) < 0.02
    assert len(item["equity_curve"]) == 10
print("PASS: Layer 8 excludes warm-up bars from evaluation results")
print("PASS: Layer 8 metrics and final capital reconcile")
