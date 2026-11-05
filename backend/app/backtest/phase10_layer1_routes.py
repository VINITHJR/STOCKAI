from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.backtest.ai_layer1 import run_ai_signal_backtest

router = APIRouter(
    prefix="/api/backtest/ai",
    tags=["Phase 10 Layer 1 - AI Validation"],
)


@router.get("/run")
async def run_phase10_layer1(
    instrument_key: str = Query(..., min_length=1),
    from_date: str = Query(..., min_length=10, max_length=10),
    to_date: str = Query(..., min_length=10, max_length=10),
    unit: str = Query("days", min_length=1),
    interval: str = Query("1", min_length=1),
    initial_capital: float = Query(100000.0, gt=0),
    quantity: int = Query(1, gt=0),
    forward_days: int = Query(5, ge=1, le=30),
    step_days: int = Query(5, ge=1, le=30),
    neutral_threshold_percent: float = Query(1.0, ge=0, le=20),
):
    try:
        return await run_ai_signal_backtest(
            instrument_key=instrument_key.strip(),
            from_date=from_date,
            to_date=to_date,
            unit=unit,
            interval=interval,
            initial_capital=initial_capital,
            quantity=quantity,
            forward_days=forward_days,
            step_days=step_days,
            neutral_threshold_percent=neutral_threshold_percent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
