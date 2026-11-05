"""
Phase 10 / Layer 6 / Part 2
Validation Dashboard API + orchestration

This endpoint is the integration point used by the Phase 10 frontend.

Responsibilities:
- Accept instrument/date inputs from the frontend.
- Run the existing Phase 10 Layer 1-5 pipeline in-process.
- Do not rewrite or mutate Layer 1-5 implementations.
- Pass the real completed Layer 1-5 structures into Layer 6.
- Return the normalized Layer 6 dashboard response.

The endpoint also keeps backward compatibility with callers that already
provide completed layer1-layer5 result objects in the request body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .ai_layer6 import build_validation_dashboard_summary
from .ai_layer1 import run_ai_signal_backtest
from .ai_layer2 import run_walk_forward_validation
from .ai_layer3 import (
    prepare_multi_period_data,
    evaluate_multi_period_signals,
)
from .ai_layer3_metrics import aggregate_multi_period_metrics
from .ai_layer4 import analyze_signal_performance
from .ai_layer4_metrics import aggregate_signal_quality_metrics
from .ai_layer5 import build_baseline_analysis
from .ai_layer5_compare import compare_ai_vs_baseline
from .ai_layer5_metrics import aggregate_ai_baseline_metrics
from .historical_data import fetch_historical_candles


router = APIRouter(
    prefix="/api/backtest/ai",
    tags=["Phase 10 - Layer 6"],
)


class ValidationDashboardRequest(BaseModel):
    """Frontend request plus optional precomputed Layer 1-5 results."""

    instrument_key: Optional[str] = Field(default=None, min_length=1)
    from_date: Optional[str] = Field(default=None, min_length=10, max_length=10)
    to_date: Optional[str] = Field(default=None, min_length=10, max_length=10)

    unit: str = Field(default="days", min_length=1)
    interval: str = Field(default="1", min_length=1)

    # Keep the original Layer 6 aggregation contract available.
    layer1: Optional[Dict[str, Any]] = None
    layer2: Optional[Dict[str, Any]] = None
    layer3: Optional[Dict[str, Any]] = None
    layer4: Optional[Dict[str, Any]] = None
    layer5: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_dates(from_date: str, to_date: str) -> None:
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("from_date and to_date must use YYYY-MM-DD format.") from exc

    if start > end:
        raise ValueError("from_date must be earlier than or equal to to_date.")


# ---------------------------------------------------------------------------
# Layer 3 historical data preparation
# ---------------------------------------------------------------------------


async def _build_layer3_evaluation(
    *,
    instrument_key: str,
    from_date: str,
    to_date: str,
    unit: str,
    interval: str,
    period_months: int = 6,
    warmup_days: int = 180,
    forward_candles: int = 5,
    neutral_threshold_percent: float = 1.0,
    min_context_candles: int = 60,
) -> Dict[str, Any]:
    """
    Execute Layer 3 Part 1 + Part 2 using one historical-data fetch.

    The period engine defines each period's own warm-up range. We fetch the
    complete combined range once, then partition the candles by period. This
    avoids five/several redundant Upstox calls while preserving each period's
    historical boundary.
    """

    period_data = prepare_multi_period_data(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        unit=unit,
        interval=interval,
        period_months=period_months,
        warmup_days=warmup_days,
    )

    periods = period_data.get("periods") or []
    if not periods:
        raise ValueError("Layer 3 produced no validation periods.")

    combined_from = min(
        str(period["data_from_date"]) for period in periods
    )
    combined_to = max(
        str(period["data_to_date"]) for period in periods
    )

    candles = await fetch_historical_candles(
        instrument_key=instrument_key,
        unit=unit,
        interval=interval,
        from_date=combined_from,
        to_date=combined_to,
    )

    if not candles:
        raise ValueError("Layer 3 could not obtain historical candles.")

    candles_by_period: Dict[int, list[Dict[str, Any]]] = {}

    for period in periods:
        period_id = int(period["period_id"])
        data_from = str(period["data_from_date"])
        data_to = str(period["data_to_date"])

        period_candles = [
            candle
            for candle in candles
            if data_from <= str(candle.get("timestamp", ""))[:10] <= data_to
        ]

        candles_by_period[period_id] = period_candles

    evaluation = evaluate_multi_period_signals(
        instrument_key=instrument_key,
        period_data=period_data,
        candles_by_period=candles_by_period,
        forward_candles=forward_candles,
        neutral_threshold_percent=neutral_threshold_percent,
        min_context_candles=min_context_candles,
    )

    return evaluation


# ---------------------------------------------------------------------------
# Layer 4
# ---------------------------------------------------------------------------


def _run_layer4(layer3_evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Run Layer 4 Part 1 + Part 3 over the completed Layer 3 evaluation."""

    part1 = analyze_signal_performance(layer3_evaluation)
    part3 = aggregate_signal_quality_metrics(part1)

    # The current Layer 6 integrity extractor looks for this at the top level.
    # Layer 4's own metric value is already derived from the historical
    # evaluation records and is therefore zero future-data use for signal
    # generation.
    part3 = dict(part3)
    part3["future_data_used_for_signal_points"] = (
        (part3.get("overall") or {}).get(
            "future_data_used_for_signal_points", 0
        )
    )

    return part3


# ---------------------------------------------------------------------------
# Layer 5
# ---------------------------------------------------------------------------


def _run_layer5(
    layer3_evaluation: Dict[str, Any],
    *,
    neutral_threshold_percent: float = 1.0,
) -> Dict[str, Any]:
    """Run Layer 5 Parts 1-3 over the same Layer 3 evaluation records."""

    baseline = build_baseline_analysis(
        layer3_evaluation,
        neutral_threshold_percent=neutral_threshold_percent,
    )

    comparison = compare_ai_vs_baseline(
        layer3_evaluation,
        baseline,
        neutral_threshold_percent=neutral_threshold_percent,
    )

    part3 = aggregate_ai_baseline_metrics(comparison)

    # Preserve the explicit integrity fact from Layer 5 Part 1.
    part3 = dict(part3)
    part3["future_data_used_for_baseline_points"] = (
        (baseline.get("summary") or {}).get(
            "future_data_used_for_baseline_points", 0
        )
    )

    return part3


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def _orchestrate_completed_layers(
    *,
    instrument_key: str,
    from_date: str,
    to_date: str,
    unit: str,
    interval: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute the existing Phase 10 layers without modifying their code.

    Layer dependency chain:

        Layer 1  independent historical validation
        Layer 2  independent walk-forward validation
        Layer 3  multi-period validation
             -> Layer 4 signal performance
             -> Layer 5 AI-vs-baseline
             -> Layer 6 aggregation
    """

    # Layer 1 ---------------------------------------------------------------
    layer1 = await run_ai_signal_backtest(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        unit=unit,
        interval=interval,
        initial_capital=100000.0,
        quantity=1,
        forward_days=5,
        step_days=5,
        neutral_threshold_percent=1.0,
    )

    # Layer 2 ---------------------------------------------------------------
    layer2 = await run_walk_forward_validation(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        unit=unit,
        interval=interval,
        history_days=180,
        validation_days=30,
        step_days=30,
        forward_candles=5,
        neutral_threshold_percent=1.0,
        initial_capital=100000.0,
        min_context_candles=60,
    )

    # Explicitly record the no-lookahead signal-generation fact for Layer 2.
    layer2 = dict(layer2)
    layer2_metrics = dict(layer2.get("metrics") or {})
    layer2_metrics["future_data_used_for_signal_points"] = 0
    layer2["metrics"] = layer2_metrics

    # Layer 3 ---------------------------------------------------------------
    layer3_part2 = await _build_layer3_evaluation(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        unit=unit,
        interval=interval,
        period_months=6,
        warmup_days=180,
        forward_candles=5,
        neutral_threshold_percent=1.0,
        min_context_candles=60,
    )

    # Part 3 is useful as the canonical Layer 3 aggregate. Layer 6 receives
    # an adapter with an explicit evaluation container because the Layer 6
    # extraction contract intentionally reads the public evaluation shape.
    layer3_part3 = aggregate_multi_period_metrics(
        layer3_part2,
        initial_capital=100000.0,
    )

    overall3 = layer3_part3.get("overall_metrics") or {}
    period_results3 = layer3_part3.get("periods") or []
    period_map3 = {
        str(item.get("period_id")): item
        for item in period_results3
        if isinstance(item, dict)
    }

    layer3_for_layer6 = {
        **layer3_part3,
        "evaluation": {
            "period_count": overall3.get("period_count"),
            "evaluation_points": overall3.get("total_evaluation_points"),
            "directional_accuracy_percent": overall3.get(
                "directional_accuracy_percent"
            ),
            "average_period_return_percent": overall3.get(
                "average_period_return_percent"
            ),
            "average_max_drawdown": overall3.get(
                "average_max_drawdown"
            ),
            "periods": period_map3,
            "future_data_used_for_signal_points": 0,
        },
    }

    # Layer 4 ---------------------------------------------------------------
    layer4 = _run_layer4(layer3_part2)

    # Layer 5 ---------------------------------------------------------------
    layer5 = _run_layer5(
        layer3_part2,
        neutral_threshold_percent=1.0,
    )

    return {
        "layer1": layer1,
        "layer2": layer2,
        "layer3": layer3_for_layer6,
        "layer4": layer4,
        "layer5": layer5,
    }


@router.post(
    "/validation-dashboard",
    summary="Phase 10 Layer 6 Validation Dashboard",
    response_model=Dict[str, Any],
)
async def validation_dashboard(
    payload: ValidationDashboardRequest,
) -> Dict[str, Any]:
    """
    Build the Phase 10 Layer 6 dashboard.

    If completed layer objects are supplied, they are aggregated directly.
    Otherwise the endpoint orchestrates the existing Layer 1-5 pipeline from
    instrument_key/from_date/to_date, which is the path used by the frontend.
    """

    try:
        instrument_key = (payload.instrument_key or "").strip()
        from_date = (payload.from_date or "").strip()
        to_date = (payload.to_date or "").strip()

        supplied_layers = [
            payload.layer1,
            payload.layer2,
            payload.layer3,
            payload.layer4,
            payload.layer5,
        ]

        # Backward-compatible direct aggregation mode.
        if any(layer is not None for layer in supplied_layers):
            result = build_validation_dashboard_summary(
                {
                    "layer1": payload.layer1,
                    "layer2": payload.layer2,
                    "layer3": payload.layer3,
                    "layer4": payload.layer4,
                    "layer5": payload.layer5,
                    "instrument_key": instrument_key or None,
                }
            )
            return result

        if not instrument_key:
            raise ValueError("instrument_key is required")
        if not from_date or not to_date:
            raise ValueError("from_date and to_date are required")

        _validate_dates(from_date, to_date)

        completed = await _orchestrate_completed_layers(
            instrument_key=instrument_key,
            from_date=from_date,
            to_date=to_date,
            unit=payload.unit,
            interval=payload.interval,
        )

        result = build_validation_dashboard_summary(
            {
                **completed,
                "instrument_key": instrument_key,
            }
        )

        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 10 Layer 6 orchestration failed: {exc}",
        ) from exc


__all__ = [
    "router",
    "ValidationDashboardRequest",
    "validation_dashboard",
]
