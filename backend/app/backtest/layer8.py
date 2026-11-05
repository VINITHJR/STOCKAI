from __future__ import annotations

from datetime import date, datetime, timedelta
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from backend.app.backtest.cache import HistoricalDataCache
from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.historical_data import fetch_historical_candles
from backend.app.backtest.metrics import calculate_metrics
from backend.app.backtest.models import BacktestRequest, ExecutionConfigRequest

router = APIRouter(prefix="/api/backtest/robustness", tags=["Phase 8 Layer 8 - Multi-Period Robustness"])


class BacktestPeriod(BaseModel):
    from_date: str = Field(..., min_length=10, max_length=10)
    to_date: str = Field(..., min_length=10, max_length=10)
    @model_validator(mode="after")
    def validate_dates(self):
        try:
            start, end = date.fromisoformat(self.from_date), date.fromisoformat(self.to_date)
        except ValueError as exc:
            raise ValueError("Period dates must use YYYY-MM-DD format.") from exc
        if start > end:
            raise ValueError("from_date must be earlier than or equal to to_date.")
        return self


class RobustnessBacktestRequest(BaseModel):
    instrument_key: str = Field(..., min_length=1)
    periods: List[BacktestPeriod] = Field(..., min_length=2, max_length=12)
    unit: str = Field("days", min_length=1)
    interval: str = Field("1", min_length=1)
    initial_capital: float = Field(100000.0, gt=0)
    quantity: int = Field(1, gt=0)
    short_window: int = Field(20, ge=2)
    long_window: int = Field(50, ge=3)
    execution: ExecutionConfigRequest = Field(default_factory=ExecutionConfigRequest)
    @model_validator(mode="after")
    def validate_windows(self):
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        return self


def _validate_periods(periods: List[BacktestPeriod]) -> List[BacktestPeriod]:
    ordered = sorted(periods, key=lambda p: (p.from_date, p.to_date))
    for prev, current in zip(ordered, ordered[1:]):
        if current.from_date <= prev.to_date:
            raise ValueError(f"Robustness periods must not overlap: {prev.from_date}..{prev.to_date} and {current.from_date}..{current.to_date}.")
    return ordered


def _periods_per_year(unit: str, interval: str) -> float:
    try: step = max(float(interval), 0.000001)
    except (TypeError, ValueError): step = 1.0
    unit = unit.lower().strip()
    if unit in {"day", "days"}: return 252.0 / step
    if unit in {"minute", "minutes", "min"}: return 252.0 * 375.0 / step
    if unit in {"hour", "hours"}: return 252.0 * 6.25 / step
    if unit in {"week", "weeks"}: return 52.0 / step
    if unit in {"month", "months"}: return 12.0 / step
    if unit in {"year", "years"}: return 1.0 / step
    return 252.0


def _lookback_start(request: RobustnessBacktestRequest, period: BacktestPeriod) -> str:
    unit = request.unit.lower().strip()
    try: step = max(float(request.interval), 1.0)
    except (TypeError, ValueError): step = 1.0
    if unit in {"minute", "minutes", "min"}: calendar_days = int(request.long_window * step / 375 * 3 + 10)
    elif unit in {"hour", "hours"}: calendar_days = int(request.long_window * step / 6.25 * 3 + 10)
    elif unit in {"week", "weeks"}: calendar_days = int(request.long_window * step * 7 * 3 + 10)
    elif unit in {"month", "months"}: calendar_days = int(request.long_window * step * 31 * 3 + 10)
    elif unit in {"year", "years"}: calendar_days = int(request.long_window * step * 366 * 3 + 10)
    else: calendar_days = int(request.long_window * step * 3 + 10)
    return (date.fromisoformat(period.from_date) - timedelta(days=max(calendar_days, 14))).isoformat()


async def _load_candles(request: RobustnessBacktestRequest, period: BacktestPeriod) -> Tuple[List[Dict[str, Any]], str]:
    cache = HistoricalDataCache()
    load_from = _lookback_start(request, period)
    cached = cache.get(request.instrument_key, request.unit, request.interval, load_from, period.to_date)
    if cached is not None:
        if not cached: raise ValueError(f"Cached candles are empty for {load_from}..{period.to_date}.")
        return cached, "cache"
    candles = await fetch_historical_candles(instrument_key=request.instrument_key, unit=request.unit,
        interval=request.interval, from_date=load_from, to_date=period.to_date)
    if not candles: raise ValueError(f"No historical candles found for {request.instrument_key} during {load_from}..{period.to_date}.")
    cache.set(request.instrument_key, request.unit, request.interval, load_from, period.to_date, candles)
    return candles, "upstox"


def _candle_date(candle: Dict[str, Any]) -> date:
    raw = str(candle.get("timestamp", ""))
    try: return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try: return date.fromisoformat(raw[:10])
        except ValueError as exc: raise ValueError(f"Invalid candle timestamp: {raw!r}") from exc


def _run_period(request: RobustnessBacktestRequest, period: BacktestPeriod, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candles: raise ValueError(f"No candles supplied for {period.from_date}..{period.to_date}.")
    start, end = date.fromisoformat(period.from_date), date.fromisoformat(period.to_date)
    ordered = sorted(candles, key=lambda c: str(c.get("timestamp", "")))
    warmup = [c for c in ordered if _candle_date(c) < start]
    evaluation = [c for c in ordered if start <= _candle_date(c) <= end]
    if not evaluation: raise ValueError(f"No evaluation candles found inside {period.from_date}..{period.to_date}.")
    required = request.long_window - 1
    if len(warmup) < required:
        raise ValueError(f"Insufficient warm-up history for {period.from_date}: need at least {required} candles before the period, received {len(warmup)}. Widen the historical-data lookback or verify the provider's returned history.")
    combined = warmup + evaluation
    backtest_request = BacktestRequest(instrument_key=request.instrument_key, unit=request.unit, interval=request.interval,
        from_date=period.from_date, to_date=period.to_date, initial_capital=request.initial_capital,
        quantity=request.quantity, short_window=request.short_window, long_window=request.long_window, execution=request.execution)
    result = BacktestEngine(backtest_request).run(combined, warmup_candle_count=len(warmup))
    metrics = calculate_metrics(initial_capital=request.initial_capital, trades=result["trades"],
        equity_curve=result["equity_curve"], from_date=period.from_date, to_date=period.to_date,
        periods_per_year=_periods_per_year(request.unit, request.interval), engine_final_capital=result["final_capital"])
    return {"from_date": period.from_date, "to_date": period.to_date,
        "candle_count": len(evaluation), "warmup_candle_count": len(warmup),
        "initial_capital": round(float(request.initial_capital), 4), "final_capital": metrics["final_capital"],
        "total_pnl": metrics["total_pnl"], "total_return_percent": round(metrics["total_return_percent"], 4),
        "total_trades": metrics["total_trades"], "winning_trades": metrics["winning_trades"],
        "losing_trades": metrics["losing_trades"], "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"], "max_drawdown": metrics["max_drawdown"],
        "max_drawdown_percent": metrics["max_drawdown_percent"], "cagr_percent": metrics["cagr_percent"],
        "sharpe_ratio": metrics["sharpe_ratio"], "periodic_return_count": metrics["periodic_return_count"],
        "strategy": result["strategy"], "equity_curve": metrics["equity_curve"], "trades": result["trades"],
        "total_execution_cost": result["total_execution_cost"]}


def _aggregate(periods: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
    if not periods: raise ValueError("At least one period result is required.")
    # Reconstruct returns from capital values instead of the rounded display
    # percentage. This preserves small differences between otherwise similar
    # periods for sign classification and standard-deviation calculations.
    returns = [
        (float(x["final_capital"]) - float(x.get("initial_capital", initial_capital)))
        / float(x.get("initial_capital", initial_capital)) * 100.0
        for x in periods
    ]
    drawdowns = [float(x["max_drawdown_percent"]) for x in periods]
    counts = [int(x["total_trades"]) for x in periods]
    cagrs = [float(x["cagr_percent"]) for x in periods if x.get("cagr_percent") is not None]
    sharpes = [float(x["sharpe_ratio"]) for x in periods if x.get("sharpe_ratio") is not None]
    positive = sum(x > 0 for x in returns)
    return {"period_count": len(periods), "positive_periods": positive,
        "negative_periods": sum(x < 0 for x in returns), "flat_periods": sum(x == 0 for x in returns),
        "positive_period_rate": round(positive / len(periods) * 100, 4),
        "average_return_percent": round(mean(returns), 4), "median_return_percent": round(median(returns), 4),
        "return_stddev_percent": round(pstdev(returns), 8) if len(returns) > 1 else 0.0,
        "best_period_return_percent": round(max(returns), 4), "worst_period_return_percent": round(min(returns), 4),
        "average_max_drawdown_percent": round(mean(drawdowns), 4), "worst_max_drawdown_percent": round(max(drawdowns), 4),
        "average_cagr_percent": round(mean(cagrs), 4) if cagrs else None,
        "average_sharpe_ratio": round(mean(sharpes), 4) if sharpes else None,
        "total_trades": sum(counts), "average_trades_per_period": round(mean(counts), 4),
        "initial_capital_per_period": round(float(initial_capital), 4)}


@router.post("/run")
async def run_robustness_backtest(request: RobustnessBacktestRequest):
    try:
        periods = _validate_periods(request.periods)
        results, sources = [], []
        for period in periods:
            candles, source = await _load_candles(request, period)
            sources.append(source)
            results.append(_run_period(request, period, candles))
        return {"success": True, "layer": "Phase 8 / Layer 8", "analysis": "Multi-period robustness backtest",
            "instrument_key": request.instrument_key,
            "data_source": "cache" if sources and all(s == "cache" for s in sources) else "mixed_or_upstox",
            "periods": results, "summary": _aggregate(results, request.initial_capital)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
