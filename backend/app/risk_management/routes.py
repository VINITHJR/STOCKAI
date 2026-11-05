"""
Phase 12 - Risk Management API Routes

API integration layer for the completed Phase 12 risk-management
components.

Layers 1-7 remain unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .alerts import RiskAlertManager
from .limits import RiskLimitManager
from .portfolio_risk import PortfolioRiskManager
from .position_risk import PositionRiskManager
from .risk_engine import RiskEngine
from .risk_metrics import RiskMetricsManager


router = APIRouter(
    prefix="/api/risk",
    tags=["Phase 12 - Risk Management"],
)


# ============================================================
# SHARED MANAGERS
# ============================================================

risk_engine = RiskEngine()
position_risk_manager = PositionRiskManager()
portfolio_risk_manager = PortfolioRiskManager()
risk_limit_manager = RiskLimitManager()
risk_metrics_manager = RiskMetricsManager()
risk_alert_manager = RiskAlertManager()


# ============================================================
# REQUEST MODELS
# ============================================================


class PositionSizeRequest(BaseModel):
    position_value: float = Field(..., gt=0)
    portfolio_equity: float = Field(..., gt=0)
    instrument_key: Optional[str] = None


class PortfolioExposureRequest(BaseModel):
    invested_value: float = Field(..., ge=0)
    portfolio_equity: float = Field(..., gt=0)


class CapitalRequest(BaseModel):
    required_capital: float = Field(..., ge=0)
    available_capital: float = Field(..., ge=0)


class DrawdownRequest(BaseModel):
    drawdown_percent: float = Field(..., ge=0)


class PositionRequest(BaseModel):
    instrument_key: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    portfolio_equity: float = Field(..., gt=0)


class StopLossRequest(BaseModel):
    triggered: bool
    instrument_key: str = Field(..., min_length=1)
    current_price: float = Field(..., gt=0)
    stop_loss_price: float = Field(..., gt=0)


class TakeProfitRequest(BaseModel):
    triggered: bool
    instrument_key: str = Field(..., min_length=1)
    current_price: float = Field(..., gt=0)
    take_profit_price: float = Field(..., gt=0)


# ============================================================
# HEALTH
# ============================================================


@router.get("/health")
def risk_health() -> Dict[str, Any]:
    """
    Basic Phase 12 API health check.
    """

    return {
        "status": "success",
        "phase": 12,
        "module": "risk_management",
        "layers": {
            "layer_1_models": "ready",
            "layer_2_risk_engine": "ready",
            "layer_3_position_risk": "ready",
            "layer_4_portfolio_risk": "ready",
            "layer_5_risk_limits": "ready",
            "layer_6_risk_metrics": "ready",
            "layer_7_risk_alerts": "ready",
        },
    }


# ============================================================
# LAYER 2 — RISK ENGINE
# ============================================================


@router.post("/evaluate/position-size")
def evaluate_position_size(
    request: PositionSizeRequest,
) -> Dict[str, Any]:
    """
    Evaluate position size against configured risk limits.
    """

    try:
        evaluation = risk_engine.evaluate_position_size(
            position_value=request.position_value,
            portfolio_equity=request.portfolio_equity,
            instrument_key=request.instrument_key,
        )

        return evaluation.__dict__

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Position-size risk evaluation failed: {exc}",
        ) from exc


@router.post("/evaluate/portfolio-exposure")
def evaluate_portfolio_exposure(
    request: PortfolioExposureRequest,
) -> Dict[str, Any]:
    """
    Evaluate portfolio exposure against configured limits.
    """

    try:
        evaluation = risk_engine.evaluate_portfolio_exposure(
            invested_value=request.invested_value,
            portfolio_equity=request.portfolio_equity,
        )

        return evaluation.__dict__

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Portfolio-exposure evaluation failed: {exc}",
        ) from exc


@router.post("/evaluate/capital")
def evaluate_capital(
    request: CapitalRequest,
) -> Dict[str, Any]:
    """
    Evaluate available capital.
    """

    try:
        evaluation = risk_engine.evaluate_capital(
            required_capital=request.required_capital,
            available_capital=request.available_capital,
        )

        return evaluation.__dict__

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Capital risk evaluation failed: {exc}",
        ) from exc


@router.post("/evaluate/drawdown")
def evaluate_drawdown(
    request: DrawdownRequest,
) -> Dict[str, Any]:
    """
    Evaluate portfolio drawdown.
    """

    try:
        evaluation = risk_engine.evaluate_drawdown(
            drawdown_percent=request.drawdown_percent,
        )

        return evaluation.__dict__

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Drawdown evaluation failed: {exc}",
        ) from exc


# ============================================================
# LAYER 3 — POSITION RISK
# ============================================================


@router.post("/position")
def calculate_position_risk(
    request: PositionRequest,
) -> Dict[str, Any]:
    """
    Calculate complete position-level risk.
    """

    try:
        result = position_risk_manager.analyze_position(
            instrument_key=request.instrument_key,
            quantity=request.quantity,
            entry_price=request.entry_price,
            current_price=request.current_price,
            portfolio_equity=request.portfolio_equity,
        )

        if hasattr(result, "__dict__"):
            return result.__dict__

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Position-risk calculation failed: {exc}",
        ) from exc


# ============================================================
# LAYER 4 — PORTFOLIO RISK
# ============================================================


@router.post("/portfolio")
def calculate_portfolio_risk(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate portfolio-level risk.

    The payload is passed through to the existing portfolio
    risk manager without changing its implementation.
    """

    try:
        result = portfolio_risk_manager.analyze_portfolio(
            **payload
        )

        if hasattr(result, "__dict__"):
            return result.__dict__

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Portfolio-risk calculation failed: {exc}",
        ) from exc


# ============================================================
# LAYER 5 — RISK LIMIT VALIDATION
# ============================================================


@router.post("/limits/validate")
def validate_risk_limits(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate a trade against the configured risk limits.
    """

    try:
        result = risk_limit_manager.validate_trade(
            **payload
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Risk-limit validation failed: {exc}",
        ) from exc


# ============================================================
# LAYER 6 — RISK METRICS
# ============================================================


@router.post("/metrics")
def calculate_risk_metrics(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate portfolio risk metrics.
    """

    try:
        result = risk_metrics_manager.calculate_portfolio_metrics(
            **payload
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Risk-metrics calculation failed: {exc}",
        ) from exc


# ============================================================
# LAYER 7 — ALERTS
# ============================================================


@router.post("/alerts/stop-loss")
def check_stop_loss(
    request: StopLossRequest,
) -> Dict[str, Any]:
    """
    Generate a stop-loss alert when triggered.
    """

    try:
        alert = risk_alert_manager.check_stop_loss(
            triggered=request.triggered,
            instrument_key=request.instrument_key,
            current_price=request.current_price,
            stop_loss_price=request.stop_loss_price,
        )

        if alert is None:
            return {
                "status": "success",
                "alert": None,
            }

        return {
            "status": "success",
            "alert": alert.__dict__,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Stop-loss alert check failed: {exc}",
        ) from exc


@router.post("/alerts/take-profit")
def check_take_profit(
    request: TakeProfitRequest,
) -> Dict[str, Any]:
    """
    Generate a take-profit alert when triggered.
    """

    try:
        alert = risk_alert_manager.check_take_profit(
            triggered=request.triggered,
            instrument_key=request.instrument_key,
            current_price=request.current_price,
            take_profit_price=request.take_profit_price,
        )

        if alert is None:
            return {
                "status": "success",
                "alert": None,
            }

        return {
            "status": "success",
            "alert": alert.__dict__,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Take-profit alert check failed: {exc}",
        ) from exc


@router.get("/alerts")
def get_risk_alerts() -> Dict[str, Any]:
    """
    Return all currently stored risk alerts.
    """

    try:
        alerts = risk_alert_manager.get_alerts()

        return {
            "status": "success",
            "count": len(alerts),
            "alerts": [
                alert.__dict__
                for alert in alerts
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve risk alerts: {exc}",
        ) from exc


@router.get("/alerts/summary")
def get_risk_alert_summary() -> Dict[str, Any]:
    """
    Return the current risk-alert summary.
    """

    try:
        return {
            "status": "success",
            **risk_alert_manager.summary(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve alert summary: {exc}",
        ) from exc


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_risk_alert(
    alert_id: str,
) -> Dict[str, Any]:
    """
    Acknowledge a risk alert.
    """

    try:
        alert = risk_alert_manager.acknowledge_alert(alert_id)

        if alert is None:
            raise HTTPException(
                status_code=404,
                detail=f"Risk alert '{alert_id}' was not found.",
            )

        return {
            "status": "success",
            "alert": alert.__dict__,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to acknowledge risk alert: {exc}",
        ) from exc


# ============================================================
# ROUTER EXPORT
# ============================================================

__all__ = ["router"]