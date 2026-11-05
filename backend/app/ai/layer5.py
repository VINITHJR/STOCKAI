from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.trend_service import get_trend_prediction
from backend.app.services.prediction_service import get_prediction
from backend.app.services.fundamentals_service import get_fundamentals
from backend.app.services.news_event_service import get_news_events

from backend.app.ai.layer2 import _build_ensemble
from backend.app.ai.layer3 import _fundamental_context, _news_context
from backend.app.ai.layer4 import (
    HistoricalMarketDataError,
    InsufficientDailyCandlesError,
    load_regime_analysis,
)


router = APIRouter(
    prefix="/api/ai/decision",
    tags=["Phase 9 Layer 5 - AI Decision Intelligence"],
)


def _direction(value: str | None) -> str:
    value = (value or "neutral").strip().lower()
    if value in {"bullish", "positive", "up"}:
        return "bullish"
    if value in {"bearish", "negative", "down"}:
        return "bearish"
    return "neutral"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _symbol_from_instrument(instrument_key: str) -> str | None:
    """Best-effort symbol resolution using the existing stock-search service."""
    try:
        from backend.app.services.upstox_service import search_stocks

        result = search_stocks(instrument_key)
        if isinstance(result, dict):
            items = result.get("data") or result.get("results") or []
        else:
            items = result or []

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = item.get("instrument_key") or item.get("instrumentKey")
                if key == instrument_key:
                    value = (
                        item.get("trading_symbol")
                        or item.get("tradingsymbol")
                        or item.get("symbol")
                    )
                    return str(value).strip().upper() if value else None
    except Exception:
        pass
    return None


def _regime_context(regime_result: dict) -> dict:
    """Normalize regime output and retain current-price provenance."""
    regime = regime_result.get("regime", "range_bound")
    direction = _direction(regime_result.get("direction"))
    confidence = _clamp(float(regime_result.get("confidence") or 0.0))
    metrics = regime_result.get("metrics") or {}

    if regime in {"bullish_trend", "volatile_bullish"}:
        score = confidence
    elif regime in {"bearish_trend", "volatile_bearish"}:
        score = -confidence
    else:
        score = 0.0

    return {
        "signal": direction,
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "regime": regime,
        "current_price": metrics.get("current_price"),
        "current_price_source": metrics.get("current_price_source"),
        "current_price_timestamp": metrics.get("current_price_timestamp"),
        "last_completed_daily_close": metrics.get("last_completed_daily_close"),
        "last_completed_daily_timestamp": metrics.get("last_completed_daily_timestamp"),
    }


def _combine(
    ensemble: dict,
    fundamentals: dict,
    news: dict,
    regime: dict,
) -> dict:
    """Combine Layer 2 with bounded fundamental, news and regime context."""
    base_signal = _direction(ensemble.get("signal"))
    base_conf = _clamp(float(ensemble.get("confidence") or 0.0))

    components = [
        ("ensemble", base_signal, base_conf, 0.60),
        ("fundamentals", _direction(fundamentals.get("signal")), _clamp(float(fundamentals.get("confidence") or 0.0)), 0.15),
        ("news", _direction(news.get("signal")), _clamp(float(news.get("confidence") or 0.0)), 0.10),
        ("regime", _direction(regime.get("signal")), _clamp(float(regime.get("confidence") or 0.0)), 0.15),
    ]

    weighted = 0.0
    usable_context = 0
    supporting = 0
    opposing = 0

    for name, signal, confidence, weight in components:
        if name != "ensemble" and (confidence <= 0 or signal == "neutral"):
            continue

        if name != "ensemble":
            usable_context += 1
            if base_signal != "neutral":
                if signal == base_signal:
                    supporting += 1
                else:
                    opposing += 1

        if signal == "bullish":
            weighted += confidence * weight
        elif signal == "bearish":
            weighted -= confidence * weight

    # Context is bounded; a neutral base model is not converted into a
    # directional signal unless all three context sources agree directionally.
    if base_signal == "bullish":
        final_signal = "bullish" if weighted >= 0.10 else "neutral"
    elif base_signal == "bearish":
        final_signal = "bearish" if weighted <= -0.10 else "neutral"
    else:
        context_directions = [
            _direction(fundamentals.get("signal")),
            _direction(news.get("signal")),
            _direction(regime.get("signal")),
        ]
        if context_directions.count("bullish") == 3:
            final_signal = "bullish"
        elif context_directions.count("bearish") == 3:
            final_signal = "bearish"
        else:
            final_signal = "neutral"

    # This is a consensus measure, not a calibrated probability of return.
    if base_signal != "neutral" and usable_context:
        alignment = supporting / usable_context
        opposition_penalty = opposing / usable_context
        final_conf = base_conf + (0.10 * alignment) - (0.15 * opposition_penalty)
    elif base_signal == "neutral":
        directional = max(
            abs(float(fundamentals.get("score") or 0.0)),
            abs(float(news.get("score") or 0.0)),
            abs(float(regime.get("score") or 0.0)),
        )
        final_conf = min(0.35, directional * 0.50)
    else:
        final_conf = base_conf

    if final_signal == "neutral":
        final_conf = min(final_conf, 0.35)

    return {
        "signal": final_signal,
        "confidence": round(_clamp(final_conf), 4),
        "base_signal": base_signal,
        "base_confidence": round(base_conf, 4),
        "supporting_contexts": supporting,
        "opposing_contexts": opposing,
        "usable_contexts": usable_context,
        "weighted_context_score": round(weighted, 4),
    }


@router.get("/run")
def run_ai_decision(
    instrument_key: str = Query(..., min_length=1),
    symbol: str | None = Query(default=None, min_length=1),
    news_days: int = Query(default=7, ge=1, le=30),
    news_limit: int = Query(default=20, ge=1, le=50),
    lookback_days: int = Query(default=180, ge=60, le=365),
):
    """Phase 9 Layer 5 — AI Decision Intelligence.

    Uses the same regime loader as Layer 4 so both layers share the same
    intraday-price selection, completed-daily-candle window and indicator logic.
    """
    key = instrument_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    resolved_symbol = (symbol or "").strip().upper() or _symbol_from_instrument(key)
    if not resolved_symbol:
        raise HTTPException(
            status_code=400,
            detail="NSE symbol could not be resolved; provide the symbol query parameter.",
        )

    errors: list[dict] = []

    try:
        rf_result = get_trend_prediction(key)
    except Exception as exc:
        rf_result = None
        errors.append({"model": "RandomForest", "message": str(exc)})

    try:
        lstm_result = get_prediction(key)
    except Exception as exc:
        lstm_result = None
        errors.append({"model": "LSTM", "message": str(exc)})

    try:
        ensemble = _build_ensemble(rf_result, lstm_result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        fundamentals_result = get_fundamentals(resolved_symbol)
    except Exception as exc:
        fundamentals_result = None
        errors.append({"source": "Fundamentals", "message": str(exc)})

    try:
        news_result = get_news_events(
            resolved_symbol,
            days=news_days,
            limit=news_limit,
        )
    except Exception as exc:
        news_result = None
        errors.append({"source": "News", "message": str(exc)})

    fundamental_context = _fundamental_context(fundamentals_result)
    news_context = _news_context(news_result, symbol=resolved_symbol)

    try:
        regime_bundle = load_regime_analysis(key, lookback_days)
    except InsufficientDailyCandlesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HistoricalMarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market regime calculation failed: {exc}") from exc

    regime_analysis = regime_bundle["analysis"]
    regime_context = _regime_context(regime_analysis)

    final = _combine(
        ensemble,
        fundamental_context,
        news_context,
        regime_context,
    )

    explanation = [
        f'Layer 2 ensemble is {final["base_signal"]} with {final["base_confidence"]:.1%} confidence.',
        f'Fundamental context is {fundamental_context["signal"]}.',
        f'News context is {news_context["signal"]} based on {news_context.get("total_items", 0)} relevant recent items.',
        f'Market regime is {regime_context["regime"]} with {regime_context["confidence"]:.1%} regime confidence.',
        (
            f'Regime current price is {regime_context["current_price"]} from '
            f'{regime_context["current_price_source"]} at '
            f'{regime_context["current_price_timestamp"]}.'
        ),
    ]

    if final["supporting_contexts"]:
        explanation.append(
            f'{final["supporting_contexts"]} context source(s) support the Layer 2 direction.'
        )
    if final["opposing_contexts"]:
        explanation.append(
            f'{final["opposing_contexts"]} context source(s) oppose the Layer 2 direction, reducing confidence.'
        )

    explanation.append(
        "Layer 5 is a transparent cross-layer synthesis; its confidence is an ensemble/context-consensus measure, not a probability of future return."
    )

    return {
        "status": "success",
        "layer": "Phase 9 / Layer 5",
        "analysis": "AI Decision Intelligence",
        "instrument_key": key,
        "symbol": resolved_symbol,
        "signal": final["signal"],
        "confidence": final["confidence"],
        "decision_context": {
            "base_layer2": {
                "signal": final["base_signal"],
                "confidence": final["base_confidence"],
                "model_count": ensemble["model_count"],
                "agreement": ensemble["agreement"],
                "conflict": ensemble["conflict"],
                "conflict_level": ensemble["conflict_level"],
            },
            "fundamentals": fundamental_context,
            "news": news_context,
            "market_regime": regime_context,
        },
        "synthesis": {
            "supporting_contexts": final["supporting_contexts"],
            "opposing_contexts": final["opposing_contexts"],
            "usable_contexts": final["usable_contexts"],
            "weighted_context_score": final["weighted_context_score"],
        },
        "explanation": explanation,
        "source_outputs": {
            "random_forest": rf_result,
            "lstm": lstm_result,
            "fundamentals": fundamentals_result,
            "news": news_result,
            "market_regime": regime_analysis,
            "regime_daily_candle_count": regime_bundle["candle_count"],
        },
        "model_errors": errors,
    }
