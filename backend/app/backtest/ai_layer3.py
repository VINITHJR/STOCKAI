
"""
Phase 10 / Layer 3
Multi-Period Validation - Data, Window, AI Evaluation Engine

This module contains:
    Part 1 - deterministic multi-period validation windows
    Part 2 - chronological Phase 9 RF + LSTM evaluation

Part 3 metrics aggregation remains in ai_layer3_metrics.py.
API routing remains in phase10_layer3_routes.py.

Important design rules:
- No current/latest market endpoint is called for historical inference.
- The current evaluation candle is NOT passed to the models.
- Future candles are used only to calculate the outcome.
- Phase 9 models are loaded ONCE per Layer 3 run, not once per signal.
- Existing public function signatures are preserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# PART 1 - CONFIGURATION / PERIOD WINDOWS
# ============================================================================

@dataclass(frozen=True)
class ValidationPeriod:
    period_id: int
    validation_from_date: str
    validation_to_date: str
    data_from_date: str
    data_to_date: str


@dataclass(frozen=True)
class MultiPeriodConfig:
    from_date: str
    to_date: str
    period_months: int = 6
    warmup_days: int = 180


@dataclass
class _Phase9Models:
    """One loaded copy of the Phase 9 models reused for the entire run."""

    rf: Any
    lstm: Any
    ensemble: Any


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Date must use YYYY-MM-DD format: {value!r}"
        ) from exc


def _format_date(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def _add_months(value: date, months: int) -> date:
    if months < 0:
        raise ValueError("months must be >= 0")

    month_index = (value.year * 12 + (value.month - 1)) + months
    year = month_index // 12
    month = month_index % 12 + 1

    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(value.day, last_day))


def validate_multi_period_config(config: MultiPeriodConfig) -> None:
    start = _parse_date(config.from_date)
    end = _parse_date(config.to_date)

    if start > end:
        raise ValueError("from_date must be <= to_date")

    if config.period_months <= 0:
        raise ValueError("period_months must be > 0")

    if config.warmup_days < 0:
        raise ValueError("warmup_days must be >= 0")


def build_multi_period_windows(
    from_date: str,
    to_date: str,
    *,
    period_months: int = 6,
    warmup_days: int = 180,
) -> List[ValidationPeriod]:
    """
    Build contiguous, non-overlapping validation periods.

    Warm-up data is placed before every validation period and is never counted
    as a validation result.
    """
    config = MultiPeriodConfig(
        from_date=from_date,
        to_date=to_date,
        period_months=period_months,
        warmup_days=warmup_days,
    )
    validate_multi_period_config(config)

    start = _parse_date(from_date)
    end = _parse_date(to_date)

    periods: List[ValidationPeriod] = []
    current_start = start
    period_id = 1

    while current_start <= end:
        next_boundary = _add_months(current_start, period_months)
        current_end = min(next_boundary - timedelta(days=1), end)
        data_from = current_start - timedelta(days=warmup_days)

        periods.append(
            ValidationPeriod(
                period_id=period_id,
                validation_from_date=_format_date(current_start),
                validation_to_date=_format_date(current_end),
                data_from_date=_format_date(data_from),
                data_to_date=_format_date(current_end),
            )
        )

        current_start = current_end + timedelta(days=1)
        period_id += 1

    return periods


def prepare_multi_period_data(
    instrument_key: str,
    from_date: str,
    to_date: str,
    *,
    unit: str = "days",
    interval: str = "1",
    period_months: int = 6,
    warmup_days: int = 180,
) -> Dict[str, Any]:
    """
    Part 1: prepare deterministic period definitions.

    No market data is fetched and no model is executed here.
    """
    if not instrument_key or not instrument_key.strip():
        raise ValueError("instrument_key is required")

    periods = build_multi_period_windows(
        from_date,
        to_date,
        period_months=period_months,
        warmup_days=warmup_days,
    )

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 3",
        "part": "Part 1",
        "analysis": "Multi-Period Validation Window Engine",
        "instrument_key": instrument_key.strip(),
        "configuration": {
            "unit": unit,
            "interval": interval,
            "period_months": period_months,
            "warmup_days": warmup_days,
        },
        "period_count": len(periods),
        "periods": [asdict(period) for period in periods],
        "execution_note": (
            "Periods are contiguous and non-overlapping. "
            "Warm-up data is placed before each validation period "
            "and is not counted as validation data. "
            "No AI inference or performance calculation is performed in Part 1."
        ),
    }


# ============================================================================
# PART 2 - HISTORICAL AI EVALUATION
# ============================================================================

def _timestamp(value: Any) -> datetime:
    """
    Normalize ISO timestamps to timezone-naive datetimes.

    Upstox candles may contain +05:30 while period boundaries are date-only.
    Removing tzinfo after parsing keeps chronological comparisons consistent.
    """
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Empty timestamp")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        result = datetime.fromisoformat(text)

    if result.tzinfo is not None:
        result = result.replace(tzinfo=None)

    return result


def _normalise_signal(value: Any) -> str:
    signal = str(value or "neutral").strip().lower()

    if signal in {"bullish", "buy", "long", "positive", "up"}:
        return "bullish"

    if signal in {"bearish", "sell", "short", "negative", "down"}:
        return "bearish"

    return "neutral"


def _extract_timestamp(candle: Dict[str, Any]) -> Optional[Any]:
    return (
        candle.get("timestamp")
        or candle.get("datetime")
        or candle.get("time")
    )


def _extract_price(candle: Dict[str, Any]) -> Optional[float]:
    for key in (
        "close",
        "close_price",
        "closing_price",
        "c",
    ):
        value = candle.get(key)

        if value is None:
            continue

        try:
            price = float(value)
        except (TypeError, ValueError):
            continue

        if price > 0:
            return price

    return None


def _sort_candles(
    candles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []

    for candle in candles:
        if not isinstance(candle, dict):
            continue

        ts = _extract_timestamp(candle)

        if ts is None:
            continue

        try:
            _timestamp(ts)
        except (TypeError, ValueError):
            continue

        valid.append(candle)

    return sorted(
        valid,
        key=lambda row: _timestamp(_extract_timestamp(row)),
    )


# ============================================================================
# PHASE 9 MODEL LOADING
# ============================================================================

def _load_phase9_models() -> _Phase9Models:
    """
    Load Phase 9 model artifacts exactly once for a Layer 3 execution.

    This is the critical performance fix.

    The previous implementation created RFTrendClassifier and
    LSTMPricePredictor and loaded their artifacts inside every evaluation
    point. For ~239 evaluation points this caused hundreds of model loads,
    TensorFlow retracing warnings, extreme runtime, and made cancellation
    difficult.
    """
    try:
        from ..ai.layer1 import _ensemble
        from ..ml.rf_trend_classifier import RFTrendClassifier
        from ..ml.lstm_model import (
            LSTMPricePredictor,
            TENSORFLOW_AVAILABLE,
        )
    except Exception as exc:
        raise RuntimeError(
            "Phase 9 model components could not be imported."
        ) from exc

    rf_model = RFTrendClassifier()

    try:
        loaded = rf_model.load_model()

        # Some implementations return None on successful load.
        # Treat an explicit False as failure.
        if loaded is False:
            raise RuntimeError("Random Forest model could not be loaded.")
    except Exception as exc:
        raise RuntimeError(
            f"Random Forest model initialization failed: {exc}"
        ) from exc

    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available; Phase 10 Layer 3 requires "
            "the Phase 9 LSTM model."
        )

    lstm_model = LSTMPricePredictor(
        sequence_length=60,
        prediction_horizons=[1, 3, 7],
    )

    try:
        loaded = lstm_model.load()

        if loaded is False:
            raise RuntimeError("Saved LSTM model could not be loaded.")
    except Exception as exc:
        raise RuntimeError(
            f"LSTM model initialization failed: {exc}"
        ) from exc

    return _Phase9Models(
        rf=rf_model,
        lstm=lstm_model,
        ensemble=_ensemble,
    )


def _get_signal_from_phase9(
    instrument_key: str,
    context_candles: List[Dict[str, Any]],
    models: Optional[_Phase9Models] = None,
) -> Dict[str, Any]:
    """
    Generate a Phase 9 RF + LSTM signal using historical context only.

    ``models`` is optional for backward compatibility. Layer 3 passes the
    single shared model bundle so models are NOT reloaded for every point.
    """
    if not context_candles:
        raise RuntimeError("No historical context candles available.")

    if models is None:
        models = _load_phase9_models()

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for RF inference.") from exc

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    try:
        import warnings
        df = pd.DataFrame(context_candles)

        # The saved RF artifact was trained under a newer scikit-learn
        # release. During prediction, older scikit-learn/joblib combinations
        # can emit the `sklearn.utils.parallel.delayed` warning repeatedly.
        # It is a warning rather than a prediction failure; keep it local to
        # this inference call so Layer 3 logs remain usable.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
                category=UserWarning,
            )
            raw_rf = models.rf.predict_trend(df)

        if not isinstance(raw_rf, dict):
            raise RuntimeError(
                "Random Forest returned a non-dictionary result."
            )

        rf_signal = _normalise_signal(
            raw_rf.get("trend")
            or raw_rf.get("signal")
            or raw_rf.get("prediction")
        )

        try:
            rf_confidence = float(raw_rf.get("confidence") or 0.0)
        except (TypeError, ValueError):
            rf_confidence = 0.0

        rf_result: Dict[str, Any] = {
            "status": "success",
            "instrument_key": instrument_key,
            "signal": rf_signal,
            "confidence": max(0.0, min(1.0, rf_confidence)),
            "probabilities": raw_rf.get("probabilities") or {},
        }

        # Preserve useful fields from the actual Phase 9 RF result.
        for key in (
            "trend",
            "prediction",
            "features",
            "model_version",
        ):
            if key in raw_rf:
                rf_result[key] = raw_rf[key]

    except Exception as exc:
        rf_result = {
            "status": "error",
            "instrument_key": instrument_key,
            "signal": "neutral",
            "confidence": 0.0,
            "probabilities": {},
            "message": str(exc),
        }

    # ------------------------------------------------------------------
    # LSTM
    # ------------------------------------------------------------------
    try:
        current_price = _extract_price(context_candles[-1])

        if current_price is None:
            raise RuntimeError(
                "The latest historical context candle has no valid close."
            )

        lstm_raw = models.lstm.predict(
            context_candles,
            current_price,
        )

        if not isinstance(lstm_raw, dict):
            raise RuntimeError(
                "LSTM returned a non-dictionary result."
            )

        if lstm_raw.get("status") != "success":
            raise RuntimeError(
                lstm_raw.get("message")
                or "LSTM prediction failed."
            )

        predictions = lstm_raw.get("predictions") or []

        if not predictions:
            raise RuntimeError("LSTM returned no predictions.")

        weighted_change = 0.0
        confidence_total = 0.0
        weight_total = 0.0

        for item in predictions:
            if not isinstance(item, dict):
                continue

            try:
                horizon = max(
                    float(item.get("horizon_days") or 1.0),
                    1.0,
                )
                change = float(
                    item.get("change_percent") or 0.0
                )
                confidence = float(
                    item.get("confidence") or 0.0
                )
            except (TypeError, ValueError):
                continue

            weight = 1.0 / horizon

            weighted_change += change * weight
            confidence_total += confidence * weight
            weight_total += weight

        avg_change = (
            weighted_change / weight_total
            if weight_total
            else 0.0
        )

        lstm_confidence = (
            confidence_total / weight_total
            if weight_total
            else 0.0
        )

        lstm_signal = (
            "bullish"
            if avg_change > 0
            else "bearish"
            if avg_change < 0
            else "neutral"
        )

        lstm_result: Dict[str, Any] = {
            "status": "success",
            "instrument_key": instrument_key,
            "signal": lstm_signal,
            "confidence": max(
                0.0,
                min(1.0, lstm_confidence),
            ),
            "weighted_change_percent": avg_change,
            "predictions": predictions,
        }

        for key in (
            "current_price",
            "sequence_length",
            "model_version",
        ):
            if key in lstm_raw:
                lstm_result[key] = lstm_raw[key]

    except Exception as exc:
        lstm_result = {
            "status": "error",
            "instrument_key": instrument_key,
            "signal": "neutral",
            "confidence": 0.0,
            "predictions": [],
            "message": str(exc),
        }

    # ------------------------------------------------------------------
    # Existing Phase 9 ensemble logic
    # ------------------------------------------------------------------
    try:
        ensemble = models.ensemble(
            rf_result,
            lstm_result,
        )

        if not isinstance(ensemble, dict):
            raise RuntimeError(
                "Phase 9 ensemble returned a non-dictionary result."
            )

    except Exception as exc:
        raise RuntimeError(
            f"Phase 9 RF/LSTM ensemble failed: {exc}"
        ) from exc

    ensemble_signal = _normalise_signal(
        ensemble.get("signal")
        or ensemble.get("final_signal")
        or ensemble.get("prediction")
    )

    try:
        ensemble_confidence = float(
            ensemble.get("confidence") or 0.0
        )
    except (TypeError, ValueError):
        ensemble_confidence = 0.0

    model_outputs = ensemble.get("models") or []

    model_errors: List[Dict[str, Any]] = []

    if rf_result.get("status") != "success":
        model_errors.append(
            {
                "model": "RandomForest",
                "message": rf_result.get("message"),
            }
        )

    if lstm_result.get("status") != "success":
        model_errors.append(
            {
                "model": "LSTM",
                "message": lstm_result.get("message"),
            }
        )

    return {
        "status": "success",
        "layer": "Phase 9 / Layer 1",
        "analysis": "AI Signal Intelligence",
        "instrument_key": instrument_key,
        "signal": ensemble_signal,
        "confidence": max(
            0.0,
            min(1.0, ensemble_confidence),
        ),
        "model_count": len(model_outputs),
        "agreement": ensemble.get("agreement"),
        "conflict": ensemble.get("conflict"),
        "raw_score": ensemble.get("raw_score"),
        "models": model_outputs,
        "explanation": [
            (
                f'{model.get("model", "Model")} produced a '
                f'{model.get("signal", "neutral")} signal with '
                f'{float(model.get("confidence") or 0.0):.1%} confidence.'
            )
            for model in model_outputs
            if isinstance(model, dict)
        ],
        "source_outputs": {
            "random_forest": rf_result,
            "lstm": lstm_result,
        },
        "model_errors": model_errors,
    }


def _extract_model_outputs(
    result: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Support both Phase 9 response structures:
        source_outputs.random_forest / source_outputs.lstm
    and direct random_forest / lstm keys.
    """
    source = result.get("source_outputs")

    if not isinstance(source, dict):
        source = {}

    rf = (
        source.get("random_forest")
        or source.get("rf")
        or result.get("random_forest")
        or {}
    )

    lstm = (
        source.get("lstm")
        or result.get("lstm")
        or {}
    )

    return (
        rf if isinstance(rf, dict) else {},
        lstm if isinstance(lstm, dict) else {},
    )


# ============================================================================
# SINGLE PERIOD EVALUATION
# ============================================================================

def _evaluate_period(
    *,
    instrument_key: str,
    period: ValidationPeriod,
    candles: List[Dict[str, Any]],
    models: _Phase9Models,
    forward_candles: int = 5,
    neutral_threshold_percent: float = 1.0,
    min_context_candles: int = 60,
) -> Dict[str, Any]:
    """
    Evaluate one validation period chronologically.

    The critical invariant is:

        model_context = candles strictly BEFORE evaluation candle

        future_candles = used ONLY after signal generation for outcome

    Therefore future_data_used_for_signal is always False for successful
    evaluation points.
    """
    if forward_candles <= 0:
        raise ValueError("forward_candles must be > 0")

    if min_context_candles <= 0:
        raise ValueError("min_context_candles must be > 0")

    ordered = _sort_candles(candles)

    validation_from = _timestamp(
        period.validation_from_date + "T00:00:00"
    )
    validation_to = _timestamp(
        period.validation_to_date + "T23:59:59"
    )

    results: List[Dict[str, Any]] = []
    model_errors = 0
    outcome_skipped = 0
    candidate_points = 0

    for index, candle in enumerate(ordered):
        ts_raw = _extract_timestamp(candle)

        if ts_raw is None:
            continue

        ts = _timestamp(ts_raw)

        if ts < validation_from or ts > validation_to:
            continue

        candidate_points += 1

        current_price = _extract_price(candle)

        if current_price is None:
            continue

        # --------------------------------------------------------------
        # NO LOOK-AHEAD:
        # current candle and every later candle are excluded.
        # --------------------------------------------------------------
        context = ordered[:index]

        if len(context) < min_context_candles:
            continue

        # --------------------------------------------------------------
        # Outcome:
        # only after signal context has been established.
        # --------------------------------------------------------------
        future_index = index + forward_candles

        if future_index >= len(ordered):
            outcome_skipped += 1
            continue

        future_candle = ordered[future_index]

        future_price = _extract_price(future_candle)
        future_ts_raw = _extract_timestamp(future_candle)

        if (
            future_price is None
            or future_ts_raw is None
        ):
            outcome_skipped += 1
            continue

        try:
            future_ts = _timestamp(future_ts_raw)
        except (TypeError, ValueError):
            outcome_skipped += 1
            continue

        # --------------------------------------------------------------
        # Phase 9 inference using the shared model instances.
        # --------------------------------------------------------------
        try:
            signal_result = _get_signal_from_phase9(
                instrument_key,
                context,
                models,
            )
        except Exception as exc:
            model_errors += 1

            results.append(
                {
                    "timestamp": ts.isoformat(),
                    "current_price": round(
                        current_price,
                        6,
                    ),
                    "signal": "error",
                    "confidence": 0.0,
                    "model_error": str(exc),
                    "available_candle_count": len(context),
                    "future_data_used_for_signal": False,
                }
            )

            continue

        signal = _normalise_signal(
            signal_result.get("signal")
            or signal_result.get("final_signal")
            or signal_result.get("prediction")
        )

        try:
            confidence = float(
                signal_result.get("confidence") or 0.0
            )
        except (TypeError, ValueError):
            confidence = 0.0

        rf, lstm = _extract_model_outputs(signal_result)

        rf_signal = _normalise_signal(
            rf.get("signal")
            or rf.get("trend")
            or rf.get("prediction")
        )

        lstm_signal = _normalise_signal(
            lstm.get("signal")
            or lstm.get("trend")
            or lstm.get("prediction")
        )

        model_agreement = (
            rf_signal != "neutral"
            and lstm_signal != "neutral"
            and rf_signal == lstm_signal
        )

        model_conflict = (
            rf_signal != "neutral"
            and lstm_signal != "neutral"
            and rf_signal != lstm_signal
        )

        forward_return = (
            (future_price - current_price)
            / current_price
        ) * 100.0

        if forward_return > neutral_threshold_percent:
            actual_direction = "bullish"
        elif forward_return < -neutral_threshold_percent:
            actual_direction = "bearish"
        else:
            actual_direction = "neutral"

        # Preserve the existing Layer 3 correctness rule.
        correct = signal == actual_direction

        results.append(
            {
                "timestamp": ts.isoformat(),
                "current_price": round(
                    current_price,
                    6,
                ),
                "signal": signal,
                "confidence": round(
                    confidence,
                    6,
                ),
                "random_forest": rf,
                "lstm": lstm,
                "model_agreement": model_agreement,
                "model_conflict": model_conflict,
                "available_candle_count": len(context),

                # Explicit integrity flag:
                "future_data_used_for_signal": False,

                # Future values are outcome-only fields.
                "future_timestamp": future_ts.isoformat(),
                "future_price": round(
                    future_price,
                    6,
                ),
                "forward_return_percent": round(
                    forward_return,
                    6,
                ),
                "actual_direction": actual_direction,
                "correct": correct,
            }
        )

    return {
        "period_id": period.period_id,
        "validation_from_date": period.validation_from_date,
        "validation_to_date": period.validation_to_date,
        "data_from_date": period.data_from_date,
        "data_to_date": period.data_to_date,
        "candidate_points": candidate_points,
        "evaluation_points": len(results),
        "model_error_points": model_errors,
        "outcome_skipped_points": outcome_skipped,
        "signals": results,
    }


# ============================================================================
# PUBLIC PART 2 API
# ============================================================================

def evaluate_multi_period_signals(
    instrument_key: str,
    period_data: Dict[str, Any],
    candles_by_period: Dict[int, List[Dict[str, Any]]],
    *,
    forward_candles: int = 5,
    neutral_threshold_percent: float = 1.0,
    min_context_candles: int = 60,
) -> Dict[str, Any]:
    """
    Part 2: evaluate every Layer 3 period chronologically.

    IMPORTANT PERFORMANCE BEHAVIOUR:
        Phase 9 models are loaded exactly once here and passed to every
        period/evaluation point.
    """
    if not instrument_key or not instrument_key.strip():
        raise ValueError("instrument_key is required")

    if not isinstance(period_data, dict):
        raise TypeError("period_data must be a dictionary")

    if not isinstance(candles_by_period, dict):
        raise TypeError(
            "candles_by_period must be a dictionary keyed by period_id"
        )

    if forward_candles <= 0:
        raise ValueError("forward_candles must be > 0")

    if not 0.0 <= neutral_threshold_percent:
        raise ValueError(
            "neutral_threshold_percent must be >= 0"
        )

    if min_context_candles <= 0:
        raise ValueError(
            "min_context_candles must be > 0"
        )

    periods: List[ValidationPeriod] = []

    for item in period_data.get("periods") or []:
        if not isinstance(item, dict):
            continue

        periods.append(
            ValidationPeriod(
                period_id=int(item["period_id"]),
                validation_from_date=str(
                    item["validation_from_date"]
                ),
                validation_to_date=str(
                    item["validation_to_date"]
                ),
                data_from_date=str(
                    item["data_from_date"]
                ),
                data_to_date=str(
                    item["data_to_date"]
                ),
            )
        )

    # --------------------------------------------------------------
    # CRITICAL FIX:
    # Load RF + LSTM ONCE, not once per evaluation point.
    # --------------------------------------------------------------
    models = _load_phase9_models()

    evaluations: List[Dict[str, Any]] = []

    for period in periods:
        raw_candles = candles_by_period.get(
            period.period_id,
            [],
        )

        if not isinstance(raw_candles, list):
            raw_candles = []

        evaluations.append(
            _evaluate_period(
                instrument_key=instrument_key.strip(),
                period=period,
                candles=raw_candles,
                models=models,
                forward_candles=forward_candles,
                neutral_threshold_percent=neutral_threshold_percent,
                min_context_candles=min_context_candles,
            )
        )

    total_evaluations = sum(
        int(item.get("evaluation_points") or 0)
        for item in evaluations
    )

    total_model_errors = sum(
        int(item.get("model_error_points") or 0)
        for item in evaluations
    )

    total_outcome_skipped = sum(
        int(item.get("outcome_skipped_points") or 0)
        for item in evaluations
    )

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 3",
        "part": "Part 2",
        "analysis": "Multi-Period AI Signal Evaluation",
        "instrument_key": instrument_key.strip(),

        "configuration": {
            "forward_candles": forward_candles,
            "neutral_threshold_percent": (
                neutral_threshold_percent
            ),
            "model_context_candles": min_context_candles,
            "models_retrained": False,
            "models_loaded_once": True,
            "execution_note": (
                "Each signal is generated chronologically from "
                "candles strictly before the evaluation timestamp. "
                "Future candles are used only for outcome measurement. "
                "Phase 9 RF and LSTM model artifacts are loaded once "
                "for the complete Layer 3 execution."
            ),
        },

        "period_count": len(evaluations),
        "evaluation_points": total_evaluations,
        "model_error_points": total_model_errors,
        "outcome_skipped_points": total_outcome_skipped,
        "future_data_used_for_signal": 0,

        "periods": evaluations,
    }


__all__ = [
    "ValidationPeriod",
    "MultiPeriodConfig",
    "validate_multi_period_config",
    "build_multi_period_windows",
    "prepare_multi_period_data",
    "evaluate_multi_period_signals",
]
