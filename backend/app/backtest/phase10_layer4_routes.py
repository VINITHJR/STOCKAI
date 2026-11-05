"""Phase 10 / Layer 4 / Part 3 API routes."""
from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from .ai_layer4_metrics import aggregate_signal_quality_metrics
router=APIRouter(prefix='/api/backtest/ai',tags=['Phase 10 - Layer 4'])
@router.post('/signal-performance/metrics')
def signal_performance_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    try: return aggregate_signal_quality_metrics(payload)
    except Exception as exc: raise HTTPException(status_code=400,detail=f'Phase 10 Layer 4 metrics error: {exc}') from exc
__all__=['router']
