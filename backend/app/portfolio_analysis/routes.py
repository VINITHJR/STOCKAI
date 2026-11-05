"""
Phase 13 - Layer 7
Portfolio Analysis API Routes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .aggregator import PortfolioAnalysisAggregator


router = APIRouter(
    prefix="/api/portfolio-analysis",
    tags=["Portfolio Analysis"],
)


class PortfolioPositionRequest(BaseModel):
    instrument_key: str
    symbol: str = ""
    quantity: float = Field(
        ge=0
    )
    average_price: float = Field(
        ge=0
    )
    current_price: float = Field(
        ge=0
    )
    sector: str = "Unknown"
    exchange: str = ""
    asset_type: str = "EQUITY"


class PortfolioAnalysisRequest(BaseModel):
    positions: List[
        PortfolioPositionRequest
    ] = Field(
        default_factory=list
    )

    cash: float = Field(
        default=0.0,
        ge=0,
    )

    initial_capital: float = Field(
        default=0.0,
        ge=0,
    )

    realized_pnl: float = 0.0

    timestamp: Optional[str] = None

    currency: str = "INR"


@router.get("/health")
def portfolio_analysis_health() -> Dict[str, Any]:
    """
    Phase 13 Layer 7 health check.
    """

    return {
        "status": "healthy",
        "phase": 13,
        "layer": 7,
        "component": "portfolio_analysis",
        "layers": {
            "layer_1_portfolio_analyzer": "ready",
            "layer_2_performance_analysis": "ready",
            "layer_3_allocation_analysis": "ready",
            "layer_4_diversification_analysis": "ready",
            "layer_5_attribution_analysis": "ready",
            "layer_6_portfolio_health": "ready",
            "layer_7_aggregator": "ready",
        },
    }


@router.post("/analyze")
def analyze_portfolio(
    request: PortfolioAnalysisRequest,
) -> Dict[str, Any]:
    """
    Run the complete Phase 13 portfolio analysis.
    """

    try:
        positions = [
            position.model_dump()
            for position in request.positions
        ]

        analyzer = PortfolioAnalysisAggregator(
            currency=request.currency
        )

        result = analyzer.analyze(
            positions=positions,
            cash=request.cash,
            initial_capital=request.initial_capital,
            realized_pnl=request.realized_pnl,
            timestamp=request.timestamp,
        )

        return {
            "status": "success",
            "data": result.to_dict(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc