from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.backtest.cache import HistoricalDataCache
from backend.app.backtest.historical_data import fetch_historical_candles


router = APIRouter(
    prefix="/api/backtest/benchmark",
    tags=["Phase 8 Layer 7 - Benchmark Comparison"],
)


class EquityPoint(BaseModel):
    timestamp: str
    equity: float


class BenchmarkRequest(BaseModel):
    strategy_equity_curve: List[EquityPoint] = Field(..., min_length=2)
    benchmark_instrument_key: str = Field(..., min_length=1)
    unit: str = Field("days", min_length=1)
    interval: str = Field("1", min_length=1)
    from_date: str = Field(..., min_length=10, max_length=10)
    to_date: str = Field(..., min_length=10, max_length=10)
    initial_capital: float = Field(100000.0, gt=0)


def _normalize_timestamp(value: str) -> str:
    return str(value)


def _return_percent(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return (end - start) / start * 100.0


def _build_benchmark_curve(
    candles: List[Dict[str, Any]],
    initial_capital: float,
) -> List[Dict[str, Any]]:
    if not candles:
        return []

    first_close = float(candles[0]["close"])
    if first_close <= 0:
        raise ValueError("Benchmark first closing price must be greater than zero.")

    return [
        {
            "timestamp": _normalize_timestamp(candle["timestamp"]),
            "close": round(float(candle["close"]), 6),
            "equity": round(initial_capital * float(candle["close"]) / first_close, 4),
        }
        for candle in candles
    ]


def _align_strategy_and_benchmark(
    strategy_curve: List[EquityPoint],
    benchmark_curve: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    benchmark_by_time = {
        str(point["timestamp"]): point for point in benchmark_curve
    }

    rows: List[Dict[str, Any]] = []
    for point in strategy_curve:
        timestamp = str(point.timestamp)
        benchmark = benchmark_by_time.get(timestamp)
        if benchmark is None:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "strategy_equity": round(float(point.equity), 4),
                "benchmark_equity": round(float(benchmark["equity"]), 4),
                "benchmark_close": round(float(benchmark["close"]), 6),
            }
        )

    return rows


def _calculate_comparison(
    strategy_curve: List[EquityPoint],
    benchmark_curve: List[Dict[str, Any]],
    initial_capital: float,
) -> Dict[str, Any]:
    if len(strategy_curve) < 2:
        raise ValueError("At least two strategy equity points are required.")
    if not benchmark_curve:
        raise ValueError("No benchmark candles were returned.")

    aligned = _align_strategy_and_benchmark(strategy_curve, benchmark_curve)
    if len(aligned) < 2:
        # Some feeds can use equivalent dates with different timestamp formats.
        # Fall back to an index-based alignment while preserving chronological order.
        count = min(len(strategy_curve), len(benchmark_curve))
        if count < 2:
            raise ValueError("Insufficient overlapping strategy and benchmark data.")
        aligned = [
            {
                "timestamp": str(strategy_curve[i].timestamp),
                "strategy_equity": round(float(strategy_curve[i].equity), 4),
                "benchmark_equity": round(float(benchmark_curve[i]["equity"]), 4),
                "benchmark_close": round(float(benchmark_curve[i]["close"]), 6),
            }
            for i in range(count)
        ]

    strategy_start = float(aligned[0]["strategy_equity"])
    strategy_end = float(aligned[-1]["strategy_equity"])
    benchmark_start = float(aligned[0]["benchmark_equity"])
    benchmark_end = float(aligned[-1]["benchmark_equity"])

    strategy_return = _return_percent(strategy_start, strategy_end)
    benchmark_return = _return_percent(benchmark_start, benchmark_end)
    excess_return = strategy_return - benchmark_return

    # Alpha here is the simple absolute excess return over the requested period,
    # not a CAPM/Jensen alpha estimate.
    strategy_pnl = strategy_end - initial_capital
    benchmark_pnl = benchmark_end - initial_capital

    comparison_curve: List[Dict[str, Any]] = []
    for row in aligned:
        comparison_curve.append(
            {
                **row,
                "strategy_return_percent": round(
                    _return_percent(initial_capital, row["strategy_equity"]), 4
                ),
                "benchmark_return_percent": round(
                    _return_percent(initial_capital, row["benchmark_equity"]), 4
                ),
                "excess_return_percent": round(
                    _return_percent(initial_capital, row["strategy_equity"])
                    - _return_percent(initial_capital, row["benchmark_equity"]),
                    4,
                ),
            }
        )

    return {
        "initial_capital": round(initial_capital, 4),
        "strategy": {
            "start_equity": round(strategy_start, 4),
            "final_equity": round(strategy_end, 4),
            "pnl": round(strategy_pnl, 4),
            "return_percent": round(strategy_return, 4),
        },
        "benchmark": {
            "start_equity": round(benchmark_start, 4),
            "final_equity": round(benchmark_end, 4),
            "pnl": round(benchmark_pnl, 4),
            "return_percent": round(benchmark_return, 4),
        },
        "excess_return_percent": round(excess_return, 4),
        "outperformance_percent": round(excess_return, 4),
        "aligned_points": len(aligned),
        "comparison_curve": comparison_curve,
    }


async def _load_benchmark(request: BenchmarkRequest) -> tuple[List[Dict[str, Any]], str]:
    cache = HistoricalDataCache()
    cached = cache.get(
        request.benchmark_instrument_key,
        request.unit,
        request.interval,
        request.from_date,
        request.to_date,
    )
    if cached is not None:
        return cached, "cache"

    candles = await fetch_historical_candles(
        instrument_key=request.benchmark_instrument_key,
        unit=request.unit,
        interval=request.interval,
        from_date=request.from_date,
        to_date=request.to_date,
    )
    if not candles:
        raise ValueError(
            f"No benchmark historical candles found for {request.benchmark_instrument_key}."
        )

    cache.set(
        request.benchmark_instrument_key,
        request.unit,
        request.interval,
        request.from_date,
        request.to_date,
        candles,
    )
    return candles, "upstox"


@router.post("/run")
async def run_benchmark_comparison(request: BenchmarkRequest):
    try:
        candles, source = await _load_benchmark(request)
        benchmark_curve = _build_benchmark_curve(candles, request.initial_capital)
        analysis = _calculate_comparison(
            request.strategy_equity_curve,
            benchmark_curve,
            request.initial_capital,
        )

        return {
            "success": True,
            "layer": "Phase 8 / Layer 7",
            "data_source": source,
            "benchmark_instrument_key": request.benchmark_instrument_key,
            "benchmark_candle_count": len(candles),
            "analysis": analysis,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
