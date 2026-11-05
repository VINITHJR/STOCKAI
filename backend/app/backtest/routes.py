from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.backtest.cache import HistoricalDataCache
from backend.app.backtest.engine import BacktestEngine
from backend.app.backtest.historical_data import fetch_historical_candles
from backend.app.backtest.metrics import calculate_metrics
from backend.app.backtest.models import BacktestRequest


router = APIRouter(
    prefix="/api/backtest",
    tags=["Phase 8 Backtest"],
)


@router.post("/run")
async def run_backtest(request: BacktestRequest):
    try:
        cache = HistoricalDataCache()

        candles = cache.get(
            request.instrument_key,
            request.unit,
            request.interval,
            request.from_date,
            request.to_date,
        )

        cache_hit = candles is not None

        if candles is None:
            candles = await fetch_historical_candles(
                instrument_key=request.instrument_key,
                unit=request.unit,
                interval=request.interval,
                from_date=request.from_date,
                to_date=request.to_date,
            )

            if not candles:
                raise HTTPException(
                    status_code=404,
                    detail="No historical candles found.",
                )

            cache.set(
                request.instrument_key,
                request.unit,
                request.interval,
                request.from_date,
                request.to_date,
                candles,
            )

        engine = BacktestEngine(request)
        data = engine.run(candles)

        # Reconstruct lightweight objects for metrics.
        class TradeProxy:
            def __init__(self, payload):
                self.pnl = payload["pnl"]
                self.exit_time = payload["exit_time"]

        proxies = [
            TradeProxy(t) for t in data["trades"]
        ]

        metrics = calculate_metrics(
            initial_capital=request.initial_capital,
            trades=proxies,
            equity_curve=data["equity_curve"],
        )

        return {
            "success": True,
            "layer": "Phase 8 / Layer 1",
            "data_source": "cache" if cache_hit else "upstox",
            "candle_count": len(candles),
            "data": data,
            "metrics": metrics,
            "execution_costs": request.execution.model_dump(),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.delete("/cache")
def clear_backtest_cache():
    try:
        deleted = HistoricalDataCache().clear()
        return {
            "success": True,
            "deleted_files": deleted,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
