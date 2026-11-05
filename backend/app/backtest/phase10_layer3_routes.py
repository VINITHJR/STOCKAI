"""
Phase 10 / Layer 3 / Part 3
API adapter.

This route deliberately accepts a prepared Layer 3 evaluation payload rather
than changing the existing market-data services. It is a safe integration
point for the existing Part 1/Part 2 pipeline.

Use POST /api/backtest/ai/multi-period/metrics with the JSON produced by
evaluate_multi_period_signals().
"""

from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from .ai_layer3_metrics import aggregate_multi_period_metrics

router = APIRouter(
    prefix="/api/backtest/ai",
    tags=["Phase 10 - Layer 3"],
)


@router.post("/multi-period/metrics")
def multi_period_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return aggregate_multi_period_metrics(
            payload,
            initial_capital=float(
                (payload.get("configuration") or {}).get(
                    "initial_capital", 100000
                )
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Phase 10 Layer 3 metrics error: {exc}",
        ) from exc
