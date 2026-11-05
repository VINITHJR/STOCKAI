from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, Dict, List

import pandas as pd

from backend.app.backtest.historical_data import fetch_historical_candles
from backend.app.backtest.cache import HistoricalDataCache


class AIBacktestError(RuntimeError):
    pass


def _direction(value: str | None) -> str:
    value = (value or "neutral").strip().lower()
    if value in {"bullish", "positive", "up", "buy"}:
        return "bullish"
    if value in {"bearish", "negative", "down", "sell"}:
        return "bearish"
    return "neutral"


def _load_models():
    """Load the already-trained Phase 9 models. No retraining is performed."""
    from backend.app.ml.rf_trend_classifier import RFTrendClassifier
    from backend.app.ml.lstm_model import LSTMPricePredictor

    rf = RFTrendClassifier()
    rf.load_model()

    lstm = LSTMPricePredictor(
        sequence_length=60,
        prediction_horizons=[1, 3, 7],
    )
    if not lstm.load():
        raise AIBacktestError("Saved LSTM model could not be loaded.")

    return rf, lstm


def _rf_signal(rf_model, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(candles)
    result = rf_model.predict_trend(df)
    return {
        "signal": _direction(result.get("trend")),
        "confidence": float(result.get("confidence") or 0.0),
        "probabilities": result.get("probabilities") or {},
    }


def _lstm_signal(lstm_model, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    current_price = float(candles[-1]["close"])
    result = lstm_model.predict(candles, current_price)
    if result.get("status") != "success":
        raise AIBacktestError(result.get("message") or "LSTM prediction failed.")

    predictions = result.get("predictions") or []
    if not predictions:
        raise AIBacktestError("LSTM returned no predictions.")

    weighted_change = 0.0
    weight_total = 0.0
    confidence_total = 0.0

    for item in predictions:
        horizon = max(float(item.get("horizon_days") or 1.0), 1.0)
        weight = 1.0 / horizon
        weighted_change += float(item.get("change_percent") or 0.0) * weight
        confidence_total += float(item.get("confidence") or 0.0) * weight
        weight_total += weight

    avg_change = weighted_change / weight_total if weight_total else 0.0
    confidence = confidence_total / weight_total if weight_total else 0.0

    return {
        "signal": "bullish" if avg_change > 0 else "bearish" if avg_change < 0 else "neutral",
        "confidence": max(0.0, min(1.0, confidence)),
        "weighted_change_percent": avg_change,
        "predictions": predictions,
    }


def _ensemble(rf: Dict[str, Any], lstm: Dict[str, Any]) -> Dict[str, Any]:
    weights = {"RandomForest": 0.55, "LSTM": 0.45}
    outputs = [
        {"model": "RandomForest", **rf, "weight": weights["RandomForest"]},
        {"model": "LSTM", **lstm, "weight": weights["LSTM"]},
    ]

    score = 0.0
    weight_total = 0.0
    for item in outputs:
        signed = 1.0 if item["signal"] == "bullish" else -1.0 if item["signal"] == "bearish" else 0.0
        score += signed * item["confidence"] * item["weight"]
        weight_total += item["confidence"] * item["weight"]

    normalized = score / weight_total if weight_total else 0.0
    if normalized > 0.15:
        signal = "bullish"
    elif normalized < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    conflict = len({item["signal"] for item in outputs}) > 1
    confidence = min(1.0, abs(normalized))
    if conflict:
        confidence *= 0.5
    elif all(item["signal"] == signal for item in outputs):
        confidence = min(1.0, confidence + 0.10)

    return {
        "signal": signal,
        "confidence": confidence,
        "raw_score": normalized,
        "conflict": conflict,
        "models": outputs,
    }


def _outcome(signal: str, forward_return_percent: float, neutral_threshold_percent: float) -> Dict[str, Any]:
    actual_direction = (
        "bullish" if forward_return_percent > neutral_threshold_percent
        else "bearish" if forward_return_percent < -neutral_threshold_percent
        else "neutral"
    )

    if signal == "neutral":
        correct = actual_direction == "neutral"
    else:
        correct = signal == actual_direction

    return {
        "actual_direction": actual_direction,
        "correct": correct,
    }


async def run_ai_signal_backtest(
    instrument_key: str,
    from_date: str,
    to_date: str,
    unit: str = "days",
    interval: str = "1",
    initial_capital: float = 100000.0,
    quantity: int = 1,
    forward_days: int = 5,
    step_days: int = 5,
    neutral_threshold_percent: float = 1.0,
) -> Dict[str, Any]:
    """
    Phase 10 / Layer 1: Historical AI Signal Validation.

    For each historical evaluation point, only candles available up to that
    point are passed to the existing Phase 9 RF and LSTM models. Future
    candles are used only after the signal is generated to measure outcome.
    No model is retrained.
    """
    if forward_days < 1:
        raise ValueError("forward_days must be >= 1")
    if step_days < 1:
        raise ValueError("step_days must be >= 1")
    if quantity < 1:
        raise ValueError("quantity must be >= 1")
    if initial_capital <= 0 or not isfinite(float(initial_capital)):
        raise ValueError("initial_capital must be a finite value > 0")
    if neutral_threshold_percent < 0 or not isfinite(float(neutral_threshold_percent)):
        raise ValueError("neutral_threshold_percent must be a finite value >= 0")

    # The models need historical warm-up candles before the requested
    # evaluation period. Fetch extra calendar time so that the first
    # requested evaluation date can still use a full model context.
    try:
        requested_from = datetime.strptime(from_date, "%Y-%m-%d").date()
        requested_to = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("from_date and to_date must use YYYY-MM-DD format.") from exc

    if requested_from > requested_to:
        raise ValueError("from_date must be earlier than or equal to to_date.")

    warmup_calendar_days = 180
    data_from_date = (requested_from - timedelta(days=warmup_calendar_days)).isoformat()

    cache = HistoricalDataCache()
    candles = cache.get(instrument_key, unit, interval, data_from_date, to_date)
    cache_hit = candles is not None

    if candles is None:
        candles = await fetch_historical_candles(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            from_date=data_from_date,
            to_date=to_date,
        )
        if not candles:
            raise ValueError("No historical candles found.")
        cache.set(instrument_key, unit, interval, data_from_date, to_date, candles)

    candles = sorted(candles, key=lambda x: x["timestamp"])
    minimum_history = 80
    if len(candles) < minimum_history + forward_days:
        raise ValueError(
            f"Insufficient historical data. Need at least {minimum_history + forward_days} candles including warm-up history, got {len(candles)}."
        )

    rf_model, lstm_model = _load_models()

    evaluation_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # Start at the first candle whose timestamp is on/after the user's
    # requested from_date. Warm-up candles before that date are available
    # to the models but are never counted as evaluation points.
    start_index = None
    for idx in range(minimum_history - 1, len(candles)):
        timestamp_text = str(candles[idx]["timestamp"])
        if timestamp_text[:10] >= from_date:
            start_index = idx
            break

    if start_index is None:
        raise ValueError("No evaluation candles are available in the requested period.")

    last_index = len(candles) - forward_days - 1
    if start_index > last_index:
        raise ValueError("The requested period does not contain enough candles for the forward evaluation horizon.")

    evaluation_candidates = list(range(start_index, last_index + 1, step_days))
    skipped_warmup_candles = start_index

    for index in evaluation_candidates:
        history = candles[: index + 1]
        signal_time = str(history[-1]["timestamp"])
        current_price = float(history[-1]["close"])
        future_price = float(candles[index + forward_days]["close"])
        if not isfinite(current_price) or not isfinite(future_price) or current_price <= 0:
            errors.append({"timestamp": signal_time, "model": "outcome",
                           "message": "Invalid current/future close price."})
            continue
        forward_return = (future_price - current_price) / current_price * 100.0

        try:
            rf = _rf_signal(rf_model, history)
        except Exception as exc:
            errors.append({"timestamp": signal_time, "model": "RandomForest", "message": str(exc)})
            continue

        try:
            lstm = _lstm_signal(lstm_model, history)
        except Exception as exc:
            errors.append({"timestamp": signal_time, "model": "LSTM", "message": str(exc)})
            continue

        ensemble = _ensemble(rf, lstm)
        outcome = _outcome(ensemble["signal"], forward_return, neutral_threshold_percent)
        model_agreement = rf["signal"] == lstm["signal"]
        model_conflict = not model_agreement

        evaluation_rows.append({
            "timestamp": signal_time,
            "current_price": round(current_price, 4),
            "future_timestamp": str(candles[index + forward_days]["timestamp"]),
            "future_price": round(future_price, 4),
            "forward_days": forward_days,
            "forward_return_percent": round(forward_return, 4),
            "signal": ensemble["signal"],
            "confidence": round(ensemble["confidence"], 4),
            "conflict": ensemble["conflict"],
            "model_agreement": model_agreement,
            "model_conflict": model_conflict,
            "available_candle_count": len(history),
            "future_data_used_for_signal": False,
            "random_forest": {
                "signal": rf["signal"],
                "confidence": round(rf["confidence"], 4),
            },
            "lstm": {
                "signal": lstm["signal"],
                "confidence": round(lstm["confidence"], 4),
            },
            "actual_direction": outcome["actual_direction"],
            "correct": outcome["correct"],
        })

    total = len(evaluation_rows)
    correct = sum(1 for row in evaluation_rows if row["correct"])
    bullish_rows = [row for row in evaluation_rows if row["signal"] == "bullish"]
    bearish_rows = [row for row in evaluation_rows if row["signal"] == "bearish"]
    neutral_rows = [row for row in evaluation_rows if row["signal"] == "neutral"]

    # Agreement/conflict analysis compares the two underlying Phase 9 models
    # independently of the ensemble's final signal. This is validation only;
    # it does not change the Phase 9 ensemble weights or model behaviour.
    agreement_rows = [
        row for row in evaluation_rows
        if row["random_forest"]["signal"] == row["lstm"]["signal"]
    ]
    conflict_rows = [
        row for row in evaluation_rows
        if row["random_forest"]["signal"] != row["lstm"]["signal"]
    ]

    def accuracy(rows: List[Dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return round(sum(1 for row in rows if row["correct"]) / len(rows) * 100.0, 4)

    def avg_return(rows: List[Dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return round(
            sum(row["forward_return_percent"] for row in rows) / len(rows),
            4,
        )

    def positive_return_rate(rows: List[Dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return round(
            sum(1 for row in rows if row["forward_return_percent"] > 0)
            / len(rows)
            * 100.0,
            4,
        )

    avg_forward_return = (
        sum(row["forward_return_percent"] for row in evaluation_rows) / total
        if total else 0.0
    )

    # A simple signal-following equity simulation. This is intentionally a
    # validation metric, not a replacement for the existing Phase 8 engine.
    capital = float(initial_capital)
    equity_curve = [{"timestamp": from_date, "equity": round(capital, 4)}]
    for row in evaluation_rows:
        if row["signal"] == "neutral":
            trade_return = 0.0
        elif row["signal"] == "bullish":
            trade_return = row["forward_return_percent"] / 100.0
        else:
            trade_return = -row["forward_return_percent"] / 100.0
        capital *= 1.0 + trade_return
        equity_curve.append({
            "timestamp": row["future_timestamp"],
            "equity": round(capital, 4),
        })

    peak = float(initial_capital)
    max_drawdown_amount = 0.0
    max_drawdown_percent = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        drawdown_amount = max(0.0, peak - equity)
        drawdown_percent = (drawdown_amount / peak * 100.0) if peak > 0 else 0.0
        max_drawdown_amount = max(max_drawdown_amount, drawdown_amount)
        max_drawdown_percent = max(max_drawdown_percent, drawdown_percent)

    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100.0

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 1",
        "analysis": "Historical AI Signal Validation",
        "instrument_key": instrument_key,
        "data_source": "cache" if cache_hit else "upstox",
        "period": {
            "from_date": from_date,
            "to_date": to_date,
            "data_from_date": data_from_date,
            "warmup_calendar_days": warmup_calendar_days,
        },
        "configuration": {
            "forward_days": forward_days,
            "step_days": step_days,
            "neutral_threshold_percent": neutral_threshold_percent,
            "initial_capital": initial_capital,
            "quantity": quantity,
            "execution_note": "Signal is generated only from candles available at the evaluation timestamp; future candles are used only for outcome measurement.",
        },
        "model_usage": {
            "random_forest": "existing trained model",
            "lstm": "existing trained model",
            "retrained": False,
        },
        "metrics": {
            "evaluation_points": total,
            "candidate_evaluation_points": len(evaluation_candidates),
            "correct_signals": correct,
            "skipped_warmup_candles": skipped_warmup_candles,
            "model_error_count": len(errors),
            "directional_accuracy_percent": round(correct / total * 100.0, 4) if total else 0.0,
            "average_forward_return_percent": round(avg_forward_return, 4),
            "bullish_signals": len(bullish_rows),
            "bullish_accuracy_percent": accuracy(bullish_rows),
            "bearish_signals": len(bearish_rows),
            "bearish_accuracy_percent": accuracy(bearish_rows),
            "neutral_signals": len(neutral_rows),
            "neutral_accuracy_percent": accuracy(neutral_rows),
            "conflicting_model_points": sum(1 for row in evaluation_rows if row["conflict"]),
            "model_agreement_analysis": {
                "agreement_points": len(agreement_rows),
                "conflict_points": len(conflict_rows),
                "agreement_rate_percent": round(len(agreement_rows) / total * 100.0, 4) if total else 0.0,
                "conflict_rate_percent": round(len(conflict_rows) / total * 100.0, 4) if total else 0.0,
                "agreement_accuracy_percent": accuracy(agreement_rows),
                "conflict_accuracy_percent": accuracy(conflict_rows),
                "agreement_average_forward_return_percent": avg_return(agreement_rows),
                "conflict_average_forward_return_percent": avg_return(conflict_rows),
                "agreement_positive_forward_return_rate_percent": positive_return_rate(agreement_rows),
                "conflict_positive_forward_return_rate_percent": positive_return_rate(conflict_rows),
            },
            "simulated_final_capital": round(final_capital, 4),
            "simulated_return_percent": round(total_return, 4),
            "simulated_max_drawdown": round(max_drawdown_percent, 4),
            "simulated_max_drawdown_percent": round(max_drawdown_percent, 4),
            "simulated_max_drawdown_amount": round(max_drawdown_amount, 4),
        },
        "equity_curve": equity_curve,
        "signals": evaluation_rows,
        "errors": errors,
        "warning": "Historical validation does not guarantee future returns. This layer measures historical signal behavior and is not a live trading recommendation.",
        "validation_notes": [
            "Warm-up candles before the requested from_date are used only to provide model context and are not counted as evaluation points.",
            "A model error is recorded separately and does not count as a correct or incorrect signal.",
            "Signals are generated only from candles available at the signal timestamp; future candles are used only for outcome measurement.",
            "Agreement analysis compares the existing RandomForest and LSTM outputs; it does not retrain models or change Phase 9 ensemble weights.",
            "Agreement/conflict accuracy uses the final Layer 1 signal against the same forward outcome definition used by the overall validation.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
