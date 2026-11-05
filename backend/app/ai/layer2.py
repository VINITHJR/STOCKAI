from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.trend_service import get_trend_prediction
from backend.app.services.prediction_service import get_prediction


router = APIRouter(
    prefix="/api/ai/ensemble",
    tags=["Phase 9 Layer 2 - Multi-Model Ensemble"],
)

MODEL_WEIGHTS = {
    "RandomForest": 0.55,
    "LSTM": 0.45,
}


def _returned_model_error(model_name: str, result: object) -> dict | None:
    """Describe a model failure returned as data, not raised as an exception."""
    if isinstance(result, dict) and result.get("status") == "success":
        return None

    if isinstance(result, dict):
        return {
            "model": model_name,
            "error_code": result.get("error_code") or "model_unavailable",
            "message": str(
                result.get("message")
                or result.get("detail")
                or f"{model_name} did not return a successful prediction."
            ),
        }

    return {
        "model": model_name,
        "error_code": "invalid_model_response",
        "message": f"{model_name} returned no valid response.",
    }


def _direction(value: str | None) -> str:
    value = (value or "neutral").strip().lower()
    if value in {"bullish", "positive", "up"}:
        return "bullish"
    if value in {"bearish", "negative", "down"}:
        return "bearish"
    return "neutral"


def _signed(direction: str) -> float:
    if direction == "bullish":
        return 1.0
    if direction == "bearish":
        return -1.0
    return 0.0


def _lstm_summary(result: dict) -> dict | None:
    predictions = result.get("predictions") or []
    if not predictions:
        return None

    weighted_change = 0.0
    weight_total = 0.0
    confidence_total = 0.0
    directions: list[str] = []

    for item in predictions:
        try:
            horizon = max(float(item.get("horizon_sessions") or item.get("horizon_days") or 1.0), 1.0)
            change = float(item.get("change_percent") or 0.0)
            confidence = float(item.get("confidence") or 0.0)
            if not (-100.0 <= change <= 100.0):
                continue
        except (TypeError, ValueError):
            continue

        weight = 1.0 / horizon
        weighted_change += change * weight
        confidence_total += confidence * weight
        weight_total += weight
        directions.append(_direction(item.get("direction")))

    if not weight_total:
        return None

    average_change = weighted_change / weight_total
    average_confidence = confidence_total / weight_total

    bullish = sum(1 for value in directions if value == "bullish")
    bearish = sum(1 for value in directions if value == "bearish")
    neutral = sum(1 for value in directions if value == "neutral")

    if average_change > 0.05:
        aggregate_direction = "bullish"
    elif average_change < -0.05:
        aggregate_direction = "bearish"
    else:
        aggregate_direction = "neutral"

    return {
        "direction": aggregate_direction,
        "confidence": round(min(1.0, max(0.0, average_confidence)), 4),
        "weighted_change_percent": round(average_change, 4),
        "horizon_count": len(directions),
        "horizon_agreement": {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "agreement": f"{max(bullish, bearish, neutral)}/{len(directions)}",
        },
    }



def _build_ensemble(
    rf_result: dict | None,
    lstm_result: dict | None,
) -> dict:
    model_outputs: list[dict] = []

    def safe_confidence(value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not 0.0 <= confidence <= 1.0:
            return 0.0

        return confidence

    if (
        isinstance(rf_result, dict)
        and rf_result.get("status") == "success"
    ):
        rf_direction = _direction(rf_result.get("trend"))
        rf_confidence = safe_confidence(rf_result.get("confidence"))

        model_outputs.append({
            "model": "RandomForest",
            "signal": rf_direction,
            "confidence": round(rf_confidence, 4),
            "weight": MODEL_WEIGHTS["RandomForest"],
        })

    lstm_summary = None

    if (
        isinstance(lstm_result, dict)
        and lstm_result.get("status") == "success"
    ):
        lstm_summary = _lstm_summary(lstm_result)

        if lstm_summary:
            model_outputs.append({
                "model": "LSTM",
                "signal": lstm_summary["direction"],
                "confidence": safe_confidence(
                    lstm_summary["confidence"]
                ),
                "weight": MODEL_WEIGHTS["LSTM"],
            })

    if not model_outputs:
        raise RuntimeError(
            "No trained AI model produced an ensemble output."
        )

    weighted_score = 0.0
    available_weight = 0.0

    for item in model_outputs:
        confidence = safe_confidence(item["confidence"])
        weight = item["weight"]

        # Store the validated confidence in the output too.
        item["confidence"] = round(confidence, 4)

        weighted_score += (
            _signed(item["signal"]) * confidence * weight
        )

        # FIX: do not multiply the denominator by confidence.
        available_weight += weight

    normalised_score = (
        weighted_score / available_weight
        if available_weight > 0
        else 0.0
    )

    if normalised_score > 0.15:
        final_signal = "bullish"
    elif normalised_score < -0.15:
        final_signal = "bearish"
    else:
        final_signal = "neutral"

    signal_directions = [
        item["signal"] for item in model_outputs
    ]

    agreeing_models = sum(
        1 for value in signal_directions
        if value == final_signal
    )

    directional_agreement_count = (
        agreeing_models
        if final_signal in {"bullish", "bearish"}
        else 0
    )
    neutral_agreement_count = sum(
        1 for value in signal_directions if value == "neutral"
    )
    if final_signal in {"bullish", "bearish"} and agreeing_models == len(signal_directions):
        agreement_type = "directional_consensus"
    elif final_signal in {"bullish", "bearish"} and agreeing_models > 0:
        agreement_type = "partial_directional_agreement"
    elif neutral_agreement_count == len(signal_directions):
        agreement_type = "neutral_consensus"
    else:
        agreement_type = "no_directional_consensus"

    # Backward-compatible conflict semantics: `conflict` remains True for
    # any model-signal disagreement. Add fields to distinguish neutral-vs-
    # directional disagreement from opposing bullish/bearish predictions.
    signal_set = set(signal_directions)
    conflict = len(signal_set) > 1 if len(signal_directions) > 1 else False
    directional_conflict = {"bullish", "bearish"}.issubset(signal_set)

    if not conflict:
        disagreement_type = "none"
    elif directional_conflict:
        disagreement_type = "opposing_directions"
    elif "neutral" in signal_set:
        disagreement_type = "directional_neutral_disagreement"
    else:
        disagreement_type = "model_disagreement"

    # Keep legacy conflict_level behavior; this additive change does not alter
    # existing Phase 14 consumers or response compatibility.
    if conflict:
        conflict_level = (
            "high" if agreeing_models == 0 else "medium"
        )
    else:
        conflict_level = "none"

    # Confidence is the magnitude of the normalized score.
    # Model disagreement already affects the score.
    raw_confidence = min(1.0, abs(normalised_score))
    final_confidence = raw_confidence

    return {
        "signal": final_signal,
        "confidence": round(final_confidence, 4),
        "raw_ensemble_score": round(normalised_score, 4),
        "model_count": len(model_outputs),
        "agreement": f"{agreeing_models}/{len(model_outputs)}",
        "directional_agreement_count": directional_agreement_count,
        "neutral_agreement_count": neutral_agreement_count,
        "agreement_type": agreement_type,
        # `conflict` is legacy and deliberately keeps its previous meaning.
        "conflict": conflict,
        "conflict_level": conflict_level,
        "directional_conflict": directional_conflict,
        "disagreement_type": disagreement_type,
        "models": model_outputs,
        "lstm_analysis": lstm_summary,
    }



@router.get("/run")
def run_multi_model_ensemble(
    instrument_key: str = Query(..., min_length=1),
):
    """
    Phase 9 Layer 2 — Multi-Model Ensemble.

    Uses the already-trained Random Forest and LSTM models.
    No model is retrained here and Layer 1 is not modified.
    """
    key = instrument_key.strip()

    if not key:
        raise HTTPException(
            status_code=400,
            detail="instrument_key is required",
        )

    rf_result = None
    lstm_result = None
    errors: list[dict] = []

    try:
        rf_result = get_trend_prediction(key)
    except Exception as exc:
        errors.append(
            {
                "model": "RandomForest",
                "error_code": "prediction_exception",
                "message": str(exc),
            }
        )

    # A service can return {"status": "error", ...} without raising.
    # Record that failure explicitly; exception-only handling misses it.
    rf_error = _returned_model_error("RandomForest", rf_result)
    if rf_error is not None and not any(
        item.get("model") == "RandomForest" for item in errors
    ):
        errors.append(rf_error)

    try:
        lstm_result = get_prediction(key)
    except Exception as exc:
        errors.append(
            {
                "model": "LSTM",
                "error_code": "prediction_exception",
                "message": str(exc),
            }
        )

    # This catches the current price_outside_training_range response.
    lstm_error = _returned_model_error("LSTM", lstm_result)
    if lstm_error is not None and not any(
        item.get("model") == "LSTM" for item in errors
    ):
        errors.append(lstm_error)
    elif (
        isinstance(lstm_result, dict)
        and lstm_result.get("status") == "success"
        and _lstm_summary(lstm_result) is None
    ):
        errors.append({
            "model": "LSTM",
            "error_code": "invalid_prediction_payload",
            "message": "LSTM returned success but did not provide usable predictions.",
        })

    try:
        ensemble = _build_ensemble(
            rf_result,
            lstm_result,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    explanations = []

    for model in ensemble["models"]:
        explanations.append(
            f'{model["model"]} contributes a '
            f'{model["signal"]} signal with '
            f'{model["confidence"]:.1%} confidence at '
            f'{model["weight"]:.0%} ensemble weight.'
        )

    if ensemble["directional_conflict"]:
        explanations.append(
            "The available models contain opposing bullish and bearish signals; "
            "this is a direct directional conflict. Treat the ensemble cautiously."
        )
    elif ensemble["conflict"]:
        explanations.append(
            "The available models disagree between a directional signal and "
            "neutral; this is not a direct bullish-versus-bearish conflict. "
            "Treat the ensemble cautiously."
        )
    elif ensemble["model_count"] == 1:
        explanations.append(
            "Only one model produced a valid prediction; the ensemble result "
            "is provisional and does not represent cross-model agreement."
        )
    else:
        explanations.append(
            f'Model agreement is {ensemble["agreement"]}; '
            "the ensemble signal is based on the available model outputs."
        )

    if isinstance(lstm_result, dict) and lstm_result.get("status") != "success":
        explanations.append(
            "LSTM is unavailable for this instrument: "
            + str(lstm_result.get("message") or lstm_result.get("detail") or "prediction failed.")
        )

    if ensemble["lstm_analysis"]:
        horizon = ensemble["lstm_analysis"]["horizon_agreement"]
        explanations.append(
            "LSTM horizon agreement is "
            f'{horizon["agreement"]} '
            f'(bullish={horizon["bullish"]}, '
            f'bearish={horizon["bearish"]}, '
            f'neutral={horizon["neutral"]}).'
        )

    return {
        "status": "success",
        "layer": "Phase 9 / Layer 2",
        "analysis": "Multi-Model Ensemble",
        "instrument_key": key,
        "signal": ensemble["signal"],
        "confidence": ensemble["confidence"],
        "raw_ensemble_score": ensemble["raw_ensemble_score"],
        "model_count": ensemble["model_count"],
        "agreement": ensemble["agreement"],
        # Additive fields: retain all legacy fields and endpoint behavior.
        "directional_agreement_count": ensemble["directional_agreement_count"],
        "neutral_agreement_count": ensemble["neutral_agreement_count"],
        "agreement_type": ensemble["agreement_type"],
        "conflict": ensemble["conflict"],
        "conflict_level": ensemble["conflict_level"],
        "directional_conflict": ensemble["directional_conflict"],
        "disagreement_type": ensemble["disagreement_type"],
        "models": ensemble["models"],
        "lstm_analysis": ensemble["lstm_analysis"],
        "explanation": explanations,
        "source_outputs": {
            "random_forest": rf_result,
            "lstm": lstm_result,
        },
        "model_errors": errors,
    }
