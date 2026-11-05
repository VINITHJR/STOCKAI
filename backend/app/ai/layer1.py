from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.trend_service import get_trend_prediction
from backend.app.services.prediction_service import get_prediction


router = APIRouter(
    prefix="/api/ai/signal",
    tags=["Phase 9 Layer 1 - AI Signal Intelligence"],
)


def _normalise_direction(value: str | None) -> str:
    value = (value or "neutral").strip().lower()
    if value in {"bullish", "positive", "up"}:
        return "bullish"
    if value in {"bearish", "negative", "down"}:
        return "bearish"
    return "neutral"


def _ensemble(rf: dict | None, lstm: dict | None) -> dict:
    votes: list[tuple[str, float, str]] = []

    def safe_confidence(value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if 0.0 <= number <= 1.0 else 0.0

    if isinstance(rf, dict) and rf.get("status") == "success":
        direction = _normalise_direction(rf.get("trend"))
        confidence = safe_confidence(rf.get("confidence"))
        votes.append(("RandomForest", confidence, direction))

    if isinstance(lstm, dict) and lstm.get("status") == "success":
        predictions = lstm.get("predictions") or []
        if predictions:
            weighted_change = 0.0
            weight_total = 0.0
            confidence_total = 0.0
            for item in predictions:
                try:
                    change = float(item.get("change_percent") or 0.0)
                    confidence = safe_confidence(item.get("confidence"))
                    horizon = max(float(item.get("horizon_sessions") or item.get("horizon_days") or 1.0), 1.0)
                except (TypeError, ValueError):
                    continue
                if not (-100.0 <= change <= 100.0):
                    continue
                weight = 1.0 / horizon
                weighted_change += change * weight
                weight_total += weight
                confidence_total += confidence * weight
            avg_change = weighted_change / weight_total if weight_total else 0.0
            confidence = confidence_total / weight_total if weight_total else 0.0
            # Match the per-horizon neutral band; tiny predicted moves should not
            # create a directional vote just because floating-point change is nonzero.
            direction = "bullish" if avg_change > 0.05 else "bearish" if avg_change < -0.05 else "neutral"
            votes.append(("LSTM", confidence, direction))

    if not votes:
        raise RuntimeError("No trained AI model produced a signal.")

    weights = {"RandomForest": 0.55, "LSTM": 0.45}
    score = 0.0
    available_weight = 0.0

    for name, confidence, direction in votes:
        weight = weights.get(name, 1.0)

        # Keep confidence within its valid range.
        confidence = max(0.0, min(1.0, confidence))

        signed = (
            1.0 if direction == "bullish"
            else -1.0 if direction == "bearish"
            else 0.0
        )

        score += signed * confidence * weight
        available_weight += weight

    normalised = (
        score / available_weight
        if available_weight > 0
        else 0.0
    )

    if normalised > 0.15:
        signal = "bullish"
    elif normalised < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    agreement = sum(
        1 for _, _, direction in votes
        if direction == signal
    )
    directional_agreement_count = (
        agreement if signal in {"bullish", "bearish"} else 0
    )
    neutral_agreement_count = sum(
        1 for _, _, direction in votes if direction == "neutral"
    )
    if signal in {"bullish", "bearish"} and agreement == len(votes):
        agreement_type = "directional_consensus"
    elif signal in {"bullish", "bearish"} and agreement > 0:
        agreement_type = "partial_directional_agreement"
    elif neutral_agreement_count == len(votes):
        agreement_type = "neutral_consensus"
    else:
        agreement_type = "no_directional_consensus"

    # Do not add an arbitrary 10% confidence bonus.
    confidence = min(1.0, abs(normalised))

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "model_count": len(votes),
        "agreement": f"{agreement}/{len(votes)}",
        "directional_agreement_count": directional_agreement_count,
        "neutral_agreement_count": neutral_agreement_count,
        "agreement_type": agreement_type,
        "models": [
            {"model": name, "signal": direction, "confidence": round(conf, 4)}
            for name, conf, direction in votes
        ],
    }


@router.get("/run")
def run_ai_signal(
    instrument_key: str = Query(..., min_length=1),
):
    """Combine existing trained AI model outputs into one transparent signal."""
    key = instrument_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    # Run both trained models concurrently so Layer 1 executes the
    # Random Forest and LSTM inference in the same analysis pass.
    # Each model has its own exception boundary so one failure does not
    # hide the other model's result.
    rf_result = None
    lstm_result = None
    errors: list[dict] = []

    model_jobs = {
        "RandomForest": get_trend_prediction,
        "LSTM": get_prediction,
    }

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="phase9-ai") as executor:
        futures = {
            executor.submit(func, key): name
            for name, func in model_jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "RandomForest":
                    rf_result = result
                else:
                    lstm_result = result
            except Exception as exc:
                errors.append({"model": name, "message": str(exc)})

    try:
        ensemble = _ensemble(rf_result, lstm_result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    explanations = []
    for model in ensemble["models"]:
        explanations.append(
            f'{model["model"]} produced a {model["signal"]} signal with {model["confidence"]:.1%} confidence.'
        )
    if ensemble["model_count"] > 1:
        explanations.append(
            f'Model agreement is {ensemble["agreement"]}; the final signal combines the available model outputs.'
        )
    else:
        explanations.append(
            'Only one trained model produced a usable output, so the final signal uses that available model.'
        )

    return {
        "status": "success",
        "layer": "Phase 9 / Layer 1",
        "analysis": "AI Signal Intelligence",
        "instrument_key": key,
        "signal": ensemble["signal"],
        "confidence": ensemble["confidence"],
        "model_count": ensemble["model_count"],
        "agreement": ensemble["agreement"],
        # Additive fields: retain the legacy agreement string for API compatibility.
        "directional_agreement_count": ensemble["directional_agreement_count"],
        "neutral_agreement_count": ensemble["neutral_agreement_count"],
        "agreement_type": ensemble["agreement_type"],
        "models": ensemble["models"],
        "explanation": explanations,
        "source_outputs": {
            "random_forest": rf_result,
            "lstm": lstm_result,
        },
        "model_errors": errors,
    }
