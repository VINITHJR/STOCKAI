from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/backtest/risk",
    tags=["Phase 8 Layer 6 - Risk & Drawdown Analysis"],
)


class EquityPoint(BaseModel):
    timestamp: str
    equity: float


class RiskAnalysisRequest(BaseModel):
    initial_capital: float = Field(..., gt=0)
    equity_curve: List[EquityPoint] = Field(..., min_length=2)


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _days_between(start: str, end: str) -> Optional[float]:
    a = _parse_timestamp(start)
    b = _parse_timestamp(end)
    if a is None or b is None or b < a:
        return None
    return round((b - a).total_seconds() / 86400.0, 4)


def _calculate_drawdown_analysis(
    initial_capital: float,
    points: List[EquityPoint],
) -> Dict[str, Any]:
    """Calculate a chronological drawdown series without changing Layer 1 metrics."""

    peak_equity = float(initial_capital)
    peak_time: Optional[str] = None

    curve: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []

    current: Optional[Dict[str, Any]] = None
    max_dd = 0.0
    max_dd_pct = 0.0
    max_dd_peak_time: Optional[str] = None
    max_dd_trough_time: Optional[str] = None
    max_dd_recovery_time: Optional[str] = None

    for point in points:
        equity = float(point.equity)
        timestamp = point.timestamp

        # Start/update a drawdown episode before handling a new peak.
        if equity < peak_equity:
            drawdown = peak_equity - equity
            drawdown_pct = (drawdown / peak_equity * 100.0) if peak_equity else 0.0

            if current is None:
                current = {
                    "peak_time": peak_time,
                    "peak_equity": peak_equity,
                    "start_time": timestamp,
                    "trough_time": timestamp,
                    "trough_equity": equity,
                }
            elif equity < float(current["trough_equity"]):
                current["trough_time"] = timestamp
                current["trough_equity"] = equity

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct
                max_dd_peak_time = peak_time
                max_dd_trough_time = timestamp
                max_dd_recovery_time = None

        else:
            drawdown = 0.0
            drawdown_pct = 0.0

            # Equity has recovered to the previous peak.
            if current is not None:
                current["recovery_time"] = timestamp
                current["duration_days"] = _days_between(
                    str(current["start_time"]),
                    timestamp,
                )
                current["recovery_period_days"] = _days_between(
                    str(current["trough_time"]),
                    timestamp,
                )
                episodes.append(current)

                if (
                    max_dd_trough_time == current["trough_time"]
                    and max_dd_peak_time == current["peak_time"]
                ):
                    max_dd_recovery_time = timestamp

                current = None

            # Establish a new running peak.
            if equity > peak_equity:
                peak_equity = equity
                peak_time = timestamp
            elif peak_time is None:
                peak_time = timestamp

        curve.append(
            {
                "timestamp": timestamp,
                "equity": round(equity, 4),
                "peak_equity": round(peak_equity, 4),
                "drawdown": round(drawdown, 4),
                "drawdown_percent": round(drawdown_pct, 4),
                "is_peak": equity >= peak_equity,
            }
        )

    # An open drawdown at the end has not recovered.
    if current is not None:
        current["recovery_time"] = None
        current["duration_days"] = _days_between(
            str(current["start_time"]),
            str(points[-1].timestamp),
        )
        current["recovery_period_days"] = None
        current["status"] = "not_recovered"
        episodes.append(current)

    # Identify the longest drawdown episode.
    longest = None
    if episodes:
        longest = max(
            episodes,
            key=lambda item: float(item.get("duration_days") or 0.0),
        )

    final_equity = float(points[-1].equity)
    total_return = (
        (final_equity - initial_capital) / initial_capital * 100.0
        if initial_capital
        else 0.0
    )

    return {
        "initial_capital": round(initial_capital, 4),
        "final_equity": round(final_equity, 4),
        "total_return_percent": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_percent": round(max_dd_pct, 4),
        "max_drawdown_peak_time": max_dd_peak_time,
        "max_drawdown_trough_time": max_dd_trough_time,
        "max_drawdown_recovery_time": max_dd_recovery_time,
        "recovery_period_days": (
            _days_between(max_dd_trough_time, max_dd_recovery_time)
            if max_dd_trough_time and max_dd_recovery_time
            else None
        ),
        "longest_drawdown": longest,
        "drawdown_curve": curve,
    }


@router.post("/run")
def run_risk_analysis(request: RiskAnalysisRequest):
    try:
        return {
            "success": True,
            "layer": "Phase 8 / Layer 6",
            "analysis": _calculate_drawdown_analysis(
                request.initial_capital,
                request.equity_curve,
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
