from datetime import date, timedelta
import pytest
from backend.app.backtest.costs import calculate_costs, calculate_entry_costs, calculate_exit_costs
from backend.app.backtest.layer8 import BacktestPeriod, RobustnessBacktestRequest, _run_period
from backend.app.backtest.models import ExecutionConfigRequest


def candles(start, count):
    out=[]
    for i in range(count):
        p=100.0 + i
        out.append({"timestamp":(start+timedelta(days=i)).isoformat(),"open":p,"high":p+1,"low":p-1,"close":p,"volume":100})
    return out


def request():
    return RobustnessBacktestRequest(instrument_key="TEST", periods=[
        {"from_date":"2025-01-11","to_date":"2025-01-15"},
        {"from_date":"2025-01-16","to_date":"2025-01-20"}], unit="days", interval="1",
        initial_capital=100000, quantity=1, short_window=2, long_window=3)


def test_entry_exit_costs_sum_to_legacy_round_trip():
    cfg=ExecutionConfigRequest()
    buy=calculate_entry_costs(100, 2, cfg)
    sell=calculate_exit_costs(110, 2, cfg)
    total=calculate_costs(100, 110, 2, cfg)
    assert total.total_cost == pytest.approx(buy.total_cost + sell.total_cost)
    assert total.stt == pytest.approx(sell.stt)
    assert total.stamp_duty == pytest.approx(buy.stamp_duty)


def test_layer8_rejects_insufficient_warmup():
    req=request()
    period=BacktestPeriod(from_date="2025-01-11", to_date="2025-01-15")
    with pytest.raises(ValueError, match="Insufficient warm-up"):
        _run_period(req, period, candles(date(2025,1,10), 6))


def test_metrics_raise_if_engine_final_capital_does_not_reconcile():
    from backend.app.backtest.metrics import calculate_metrics
    trades=[{"pnl":10.0,"exit_time":"2025-01-12"}]
    with pytest.raises(ValueError, match="does not reconcile"):
        calculate_metrics(1000, trades, [], engine_final_capital=1050)


def test_equity_curve_execution_timestamps_match_trade_timestamps():
    from backend.app.backtest.engine import BacktestEngine
    from backend.app.backtest.models import BacktestRequest

    # A deterministic series that produces multiple SMA crossings.
    prices = [100, 99, 98, 99, 101, 104, 108, 112, 110, 106, 102, 99,
              103, 108, 114, 118, 115, 111, 107, 104, 101, 99, 102, 106]
    rows = []
    start = date(2025, 1, 1)
    for i, price in enumerate(prices):
        stamp = (start + timedelta(days=i)).isoformat()
        rows.append({"timestamp": stamp, "open": float(price), "high": float(price + 1),
                     "low": float(price - 1), "close": float(price), "volume": 1000})

    req = BacktestRequest(instrument_key="TIMING", unit="days", interval="1",
        from_date="2025-01-01", to_date="2025-01-24", initial_capital=100000,
        quantity=1, short_window=2, long_window=3)
    result = BacktestEngine(req).run(rows)
    points = {p["timestamp"]: p for p in result["equity_curve"]}

    for trade in result["trades"]:
        assert trade["entry_time"] in points
        assert points[trade["entry_time"]]["executed_action"] == "BUY"
        assert trade["exit_time"] in points
        exit_point = points[trade["exit_time"]]
        assert exit_point["executed_action"] == "SELL" or exit_point["forced_close"] is True

    assert len(result["equity_curve"]) == len(rows)
    assert result["equity_curve"][-1]["equity"] == pytest.approx(result["final_capital"], abs=0.02)


def test_layer8_aggregate_preserves_small_return_variability():
    from backend.app.backtest.layer8 import _aggregate

    periods = [
        {"initial_capital": 100000.0, "final_capital": 99999.4936,
         "total_return_percent": -0.0005, "max_drawdown_percent": 0.0215,
         "total_trades": 1, "cagr_percent": -0.0168, "sharpe_ratio": -0.0862},
        {"initial_capital": 100000.0, "final_capital": 99999.4572,
         "total_return_percent": -0.0005, "max_drawdown_percent": 0.0143,
         "total_trades": 1, "cagr_percent": -0.0180, "sharpe_ratio": -0.1370},
    ]
    summary = _aggregate(periods, 100000.0)
    assert summary["negative_periods"] == 2
    assert summary["return_stddev_percent"] > 0
    assert summary["return_stddev_percent"] == pytest.approx(0.0000182, rel=0.03)
