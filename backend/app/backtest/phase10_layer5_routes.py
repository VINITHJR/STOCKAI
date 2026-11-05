"""Phase 10 / Layer 5 / Part 3 API routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from .ai_layer5_metrics import aggregate_ai_baseline_metrics


router = APIRouter(
    prefix="/api/backtest/ai",
    tags=["Phase 10 - Layer 5"],
)


@router.post("/ai-vs-baseline/metrics")
def ai_vs_baseline_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate a Layer 5 Part 2 AI-vs-baseline comparison."""
    try:
        return aggregate_ai_baseline_metrics(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Phase 10 Layer 5 metrics error: {exc}",
        ) from exc


__all__ = ["router"]
