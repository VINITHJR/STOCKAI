from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from backend.app.backtest.historical_data import fetch_historical_candles


# =========================================================
# PHASE 10 / LAYER 2 / PART 1
# Walk-Forward Data & Window Engine
# =========================================================
# This module only prepares chronological walk-forward windows.
# It does NOT run RF/LSTM, retrain models, calculate performance,
# register an API route, or modify Phase 10 Layer 1.


DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class WalkForwardWindow:
    """One chronological walk-forward evaluation window."""

    window_id: int
    history_from_date: str
    history_to_date: str
    validation_from_date: str
    validation_to_date: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for deterministic walk-forward window generation."""

    from_date: str
    to_date: str
    history_days: int = 180
    validation_days: int = 30
    step_days: int = 30

    def validate(self) -> None:
        start = _parse_date(self.from_date, "from_date")
        end = _parse_date(self.to_date, "to_date")

        if end <= start:
            raise ValueError("to_date must be later than from_date")

        if self.history_days < 1:
            raise ValueError("history_days must be at least 1")

        if self.validation_days < 1:
            raise ValueError("validation_days must be at least 1")

        if self.step_days < 1:
            raise ValueError("step_days must be at least 1")


def _parse_date(value: str, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must use YYYY-MM-DD format")

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format"
        ) from exc

    if parsed.isoformat() != value:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format"
        )

    return parsed


def build_walk_forward_windows(
    from_date: str,
    to_date: str,
    *,
    history_days: int = 180,
    validation_days: int = 30,
    step_days: int = 30,
) -> List[WalkForwardWindow]:
    """
    Build chronological walk-forward windows.

    The requested validation range is divided into consecutive validation
    windows. Each validation window receives a historical context window
    immediately before its validation start. History and validation never
    overlap.

    Example with validation_days=30 and step_days=30:

        history       validation
        |------------| |----------|
                     ^
                     validation start

        |------------------------| |----------|
                                 ^
                                 next validation start

    The final validation window may be shorter than validation_days so that
    the requested to_date is not exceeded.
    """

    config = WalkForwardConfig(
        from_date=from_date,
        to_date=to_date,
        history_days=history_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    config.validate()

    requested_start = _parse_date(from_date, "from_date")
    requested_end = _parse_date(to_date, "to_date")

    windows: List[WalkForwardWindow] = []
    validation_start = requested_start
    window_id = 1

    while validation_start < requested_end:
        validation_end = min(
            validation_start + timedelta(days=validation_days - 1),
            requested_end,
        )

        history_end = validation_start - timedelta(days=1)
        history_start = history_end - timedelta(days=history_days - 1)

        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                history_from_date=history_start.isoformat(),
                history_to_date=history_end.isoformat(),
                validation_from_date=validation_start.isoformat(),
                validation_to_date=validation_end.isoformat(),
            )
        )

        window_id += 1
        validation_start += timedelta(days=step_days)

    return windows


def _filter_candles(
    candles: List[Dict[str, Any]],
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """Keep candles whose calendar date is inside the inclusive range."""

    filtered: List[Dict[str, Any]] = []

    for candle in candles:
        timestamp = candle.get("timestamp")
        if not timestamp:
            continue

        try:
            candle_date = date.fromisoformat(str(timestamp)[:10])
        except ValueError:
            continue

        if start_date <= candle_date <= end_date:
            filtered.append(candle)

    return filtered


async def prepare_walk_forward_data(
    instrument_key: str,
    unit: str,
    interval: str,
    from_date: str,
    to_date: str,
    *,
    history_days: int = 180,
    validation_days: int = 30,
    step_days: int = 30,
) -> Dict[str, Any]:
    """
    Fetch the required historical range once and split it into walk-forward
    windows.

    This function intentionally performs no model inference. It only prepares
    data for Part 2.
    """

    if not instrument_key or not str(instrument_key).strip():
        raise ValueError("instrument_key is required")

    config = WalkForwardConfig(
        from_date=from_date,
        to_date=to_date,
        history_days=history_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    config.validate()

    windows = build_walk_forward_windows(
        from_date,
        to_date,
        history_days=history_days,
        validation_days=validation_days,
        step_days=step_days,
    )

    if not windows:
        raise ValueError("No walk-forward windows could be created")

    requested_start = _parse_date(from_date, "from_date")
    requested_end = _parse_date(to_date, "to_date")

    # Fetch enough history for the first window's warm-up/context period.
    first_history_start = _parse_date(
        windows[0].history_from_date,
        "history_from_date",
    )

    # Upstox receives the expanded range. The requested validation period is
    # still controlled by the windows above and is never changed by this call.
    candles = await fetch_historical_candles(
        instrument_key=instrument_key,
        unit=unit,
        interval=interval,
        from_date=first_history_start.isoformat(),
        to_date=requested_end.isoformat(),
    )

    prepared_windows: List[Dict[str, Any]] = []

    for window in windows:
        history_start = _parse_date(window.history_from_date, "history_from_date")
        history_end = _parse_date(window.history_to_date, "history_to_date")
        validation_start = _parse_date(
            window.validation_from_date,
            "validation_from_date",
        )
        validation_end = _parse_date(window.validation_to_date, "validation_to_date")

        history_candles = _filter_candles(candles, history_start, history_end)
        validation_candles = _filter_candles(
            candles,
            validation_start,
            validation_end,
        )

        prepared_windows.append(
            {
                **window.to_dict(),
                "history_candle_count": len(history_candles),
                "validation_candle_count": len(validation_candles),
                "history_candles": history_candles,
                "validation_candles": validation_candles,
            }
        )

    return {
        "instrument_key": instrument_key,
        "unit": unit,
        "interval": interval,
        "requested_from_date": requested_start.isoformat(),
        "requested_to_date": requested_end.isoformat(),
        "data_from_date": first_history_start.isoformat(),
        "history_days": history_days,
        "validation_days": validation_days,
        "step_days": step_days,
        "window_count": len(prepared_windows),
        "windows": prepared_windows,
    }


# =========================================================
# PHASE 10 / LAYER 2 / PART 2
# Walk-Forward AI Signal Evaluation
# =========================================================
# This section consumes the Part 1 windows and evaluates the existing
# Phase 9 Random Forest + LSTM models chronologically. It does NOT retrain
# models and does NOT calculate final performance metrics.


class WalkForwardEvaluationError(RuntimeError):
    """Raised when a walk-forward AI evaluation point cannot be evaluated."""


def _timestamp_date(value: Any) -> date:
    """Extract a YYYY-MM-DD date from a candle timestamp."""
    raw = str(value or "")[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid candle timestamp: {value}") from exc


def _unique_candles(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort candles chronologically and remove duplicate timestamps."""
    ordered = sorted(candles, key=lambda item: str(item.get("timestamp") or ""))
    seen = set()
    result: List[Dict[str, Any]] = []

    for candle in ordered:
        timestamp = candle.get("timestamp")
        if not timestamp or timestamp in seen:
            continue
        seen.add(timestamp)
        result.append(candle)

    return result


def _load_walk_forward_models():
    """Load the existing Phase 9 models once for the whole walk-forward run."""
    from backend.app.backtest.ai_layer1 import _load_models

    try:
        return _load_models()
    except Exception as exc:
        raise WalkForwardEvaluationError(
            f"Unable to load existing Phase 9 models: {exc}"
        ) from exc


def _evaluate_validation_candle(
    rf_model: Any,
    lstm_model: Any,
    context_candles: List[Dict[str, Any]],
    validation_candle: Dict[str, Any],
    *,
    min_context_candles: int = 60,
) -> Dict[str, Any]:
    """
    Generate one AI signal using only candles available at the evaluation
    timestamp.

    The current validation candle is included in context. No later validation
    candle is passed to either model.
    """
    from backend.app.backtest.ai_layer1 import _ensemble, _lstm_signal, _rf_signal

    timestamp = validation_candle.get("timestamp")
    if not timestamp:
        raise WalkForwardEvaluationError("Validation candle has no timestamp.")

    current_price = float(validation_candle["close"])
    available_candles = _unique_candles(
        list(context_candles) + [validation_candle]
    )

    if len(available_candles) < 60:
        raise WalkForwardEvaluationError(
            f"Insufficient model context at {timestamp}: "
            f"{len(available_candles)} candles available, {min_context_candles} required."
        )

    rf = _rf_signal(rf_model, available_candles)
    lstm = _lstm_signal(lstm_model, available_candles)
    ensemble = _ensemble(rf, lstm)

    return {
        "timestamp": timestamp,
        "current_price": current_price,
        "signal": ensemble["signal"],
        "confidence": ensemble["confidence"],
        "raw_score": ensemble["raw_score"],
        "conflict": ensemble["conflict"],
        "random_forest": rf,
        "lstm": lstm,
        "available_candle_count": len(available_candles),
        "future_data_used_for_signal": False,
    }


async def evaluate_walk_forward_signals(
    prepared_data: Dict[str, Any],
    *,
    min_context_candles: int = 60,
) -> Dict[str, Any]:
    """
    Phase 10 / Layer 2 / Part 2.

    Evaluate the existing Phase 9 AI models over the validation candles
    produced by Part 1. Each validation candle receives a signal generated
    strictly from candles available at that timestamp.

    This function intentionally does not calculate final returns, accuracy,
    drawdown, or other performance metrics. Those belong to Part 3.
    """
    if not isinstance(prepared_data, dict):
        raise ValueError("prepared_data must be a dictionary from Part 1")

    if min_context_candles < 1:
        raise ValueError("min_context_candles must be >= 1")

    windows = prepared_data.get("windows") or []
    if not windows:
        raise ValueError("prepared_data contains no walk-forward windows")

    rf_model, lstm_model = _load_walk_forward_models()

    evaluated_windows: List[Dict[str, Any]] = []
    total_points = 0
    total_errors = 0

    for window in windows:
        history_candles = _unique_candles(window.get("history_candles") or [])
        validation_candles = _unique_candles(window.get("validation_candles") or [])

        signals: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        # Only candles up to the current validation timestamp are allowed.
        # Earlier validation candles become part of the available history for
        # later points in the same walk-forward window.
        prior_validation: List[Dict[str, Any]] = []

        for validation_candle in validation_candles:
            timestamp = validation_candle.get("timestamp")

            try:
                context = history_candles + prior_validation
                if len(context) + 1 < min_context_candles:
                    raise WalkForwardEvaluationError(
                        f"Insufficient model context at {timestamp}: "
                        f"{len(context) + 1} candles available, "
                        f"{min_context_candles} required."
                    )

                result = _evaluate_validation_candle(
                    rf_model,
                    lstm_model,
                    context,
                    validation_candle,
                    min_context_candles=min_context_candles,
                )
                signals.append(result)
                total_points += 1
            except Exception as exc:
                errors.append(
                    {
                        "timestamp": timestamp,
                        "message": str(exc),
                    }
                )
                total_errors += 1

            # This candle becomes available only AFTER its own signal has been
            # generated, so it cannot leak into its own prediction context.
            prior_validation.append(validation_candle)

        evaluated_windows.append(
            {
                "window_id": window.get("window_id"),
                "history_from_date": window.get("history_from_date"),
                "history_to_date": window.get("history_to_date"),
                "validation_from_date": window.get("validation_from_date"),
                "validation_to_date": window.get("validation_to_date"),
                "history_candle_count": len(history_candles),
                "validation_candle_count": len(validation_candles),
                "evaluated_signal_count": len(signals),
                "error_count": len(errors),
                "signals": signals,
                "errors": errors,
            }
        )

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 2 / Part 2",
        "analysis": "Walk-Forward AI Signal Evaluation",
        "instrument_key": prepared_data.get("instrument_key"),
        "period": {
            "from_date": prepared_data.get("requested_from_date"),
            "to_date": prepared_data.get("requested_to_date"),
            "data_from_date": prepared_data.get("data_from_date"),
        },
        "configuration": {
            "history_days": prepared_data.get("history_days"),
            "validation_days": prepared_data.get("validation_days"),
            "step_days": prepared_data.get("step_days"),
            "model_context_candles": min_context_candles,
            "models_retrained": False,
            "execution_note": (
                "Each signal is generated only from candles available at "
                "its evaluation timestamp. Later validation candles are not "
                "included in that signal's model input."
            ),
        },
        "metrics": {
            "window_count": len(evaluated_windows),
            "evaluated_signal_points": total_points,
            "model_error_points": total_errors,
        },
        "windows": evaluated_windows,
        "validation_notes": [
            "Part 1 provides the chronological history and validation windows.",
            "Existing Phase 9 Random Forest and LSTM models are reused; no retraining occurs.",
            "A validation candle is added to prior context only after its own signal is generated.",
            "Part 2 records signals and model outputs; final performance metrics belong to Part 3.",
        ],
    }


# =========================================================
# PHASE 10 / LAYER 2 / PART 3
# Walk-Forward Metrics + API Support
# =========================================================
# Part 3 consumes the Part 1 prepared windows and Part 2 signals. It uses
# future candles only AFTER each signal timestamp to measure the outcome.
# It does not retrain models and does not modify Phase 10 Layer 1.


def _all_prepared_candles(prepared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build one chronological, de-duplicated candle series from prepared windows."""
    all_candles: List[Dict[str, Any]] = []
    for window in prepared_data.get("windows") or []:
        all_candles.extend(window.get("history_candles") or [])
        all_candles.extend(window.get("validation_candles") or [])
    return _unique_candles(all_candles)


def _max_drawdown_from_equity(equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return absolute and peak-relative percentage maximum drawdown."""
    peak = 0.0
    max_drawdown_amount = 0.0
    max_drawdown_percent = 0.0

    for point in equity_curve:
        equity = float(point.get("equity") or 0.0)
        peak = max(peak, equity)
        drawdown_amount = max(0.0, peak - equity)
        drawdown_percent = (drawdown_amount / peak * 100.0) if peak > 0.0 else 0.0
        max_drawdown_amount = max(max_drawdown_amount, drawdown_amount)
        max_drawdown_percent = max(max_drawdown_percent, drawdown_percent)

    return {
        "max_drawdown_amount": round(max_drawdown_amount, 4),
        "max_drawdown_percent": round(max_drawdown_percent, 4),
    }


def _safe_percent(numerator: float, denominator: int) -> float:
    return round((numerator / denominator) * 100.0, 4) if denominator else 0.0


def _forward_outcome(
    candles: List[Dict[str, Any]],
    timestamp: str,
    forward_candles: int,
    neutral_threshold_percent: float,
) -> Dict[str, Any] | None:
    """Measure a signal's future outcome using only candles after its timestamp."""
    if forward_candles < 1:
        raise ValueError("forward_candles must be >= 1")

    timestamps = [str(c.get("timestamp") or "") for c in candles]
    try:
        index = timestamps.index(timestamp)
    except ValueError:
        return None

    future_index = index + forward_candles
    if future_index >= len(candles):
        return None

    current_price = float(candles[index]["close"])
    future_price = float(candles[future_index]["close"])
    if current_price == 0:
        return None

    forward_return = (future_price - current_price) / current_price * 100.0
    actual_direction = (
        "bullish" if forward_return > neutral_threshold_percent
        else "bearish" if forward_return < -neutral_threshold_percent
        else "neutral"
    )

    return {
        "future_timestamp": str(candles[future_index]["timestamp"]),
        "future_price": round(future_price, 4),
        "forward_return_percent": round(forward_return, 4),
        "actual_direction": actual_direction,
    }


def _signal_correct(signal: str, actual_direction: str) -> bool:
    """Compare the final AI signal with the forward-outcome direction."""
    if signal == "neutral":
        return actual_direction == "neutral"
    return signal == actual_direction


def _calculate_period_metrics(
    rows: List[Dict[str, Any]],
    *,
    initial_capital: float,
    forward_candles: int = 5,
) -> Dict[str, Any]:
    """Calculate validation metrics for one walk-forward period."""
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    positive = sum(1 for row in rows if row["forward_return_percent"] > 0)

    equity = float(initial_capital)
    equity_curve = [{"timestamp": rows[0]["timestamp"], "equity": round(equity, 4)}] if rows else []

    # Each signal represents a forward-horizon trade. Because validation
    # signals are generated more frequently than the forward horizon, applying
    # every signal to the same capital would overlap trades and compound the
    # same time period repeatedly. Simulate only non-overlapping trades.
    simulated_trade_points = 0
    next_available_timestamp: Optional[str] = None
    for row in rows:
        timestamp = str(row["timestamp"])
        if next_available_timestamp is not None and timestamp < next_available_timestamp:
            continue

        signal = row["signal"]
        forward_return = float(row["forward_return_percent"]) / 100.0
        trade_return = (
            forward_return if signal == "bullish"
            else -forward_return if signal == "bearish"
            else 0.0
        )
        equity *= 1.0 + trade_return
        simulated_trade_points += 1
        next_available_timestamp = str(row["future_timestamp"])
        equity_curve.append({
            "timestamp": row["future_timestamp"],
            "equity": round(equity, 4),
        })

    final_capital = equity
    drawdown = _max_drawdown_from_equity(equity_curve)
    return {
        "evaluation_points": total,
        "correct_signals": correct,
        "directional_accuracy_percent": _safe_percent(correct, total),
        "average_forward_return_percent": round(
            sum(float(r["forward_return_percent"]) for r in rows) / total, 4
        ) if total else 0.0,
        "positive_forward_return_rate_percent": _safe_percent(positive, total),
        "agreement_points": sum(1 for r in rows if r["model_agreement"]),
        "conflict_points": sum(1 for r in rows if r["model_conflict"]),
        "simulated_final_capital": round(final_capital, 4),
        "simulated_return_percent": round((final_capital - initial_capital) / initial_capital * 100.0, 4),
        "simulated_trade_points": simulated_trade_points,
        # Keep the legacy max_drawdown field as an absolute currency amount.
        # Use max_drawdown_percent for the peak-relative percentage.
        "max_drawdown": drawdown["max_drawdown_amount"],
        "max_drawdown_amount": drawdown["max_drawdown_amount"],
        "max_drawdown_percent": drawdown["max_drawdown_percent"],
        "equity_curve": equity_curve,
    }


async def run_walk_forward_validation(
    instrument_key: str,
    from_date: str,
    to_date: str,
    *,
    unit: str = "days",
    interval: str = "1",
    history_days: int = 180,
    validation_days: int = 30,
    step_days: int = 30,
    forward_candles: int = 5,
    neutral_threshold_percent: float = 1.0,
    initial_capital: float = 100000.0,
    min_context_candles: int = 60,
) -> Dict[str, Any]:
    """Phase 10 / Layer 2 complete walk-forward validation."""
    if not instrument_key or not str(instrument_key).strip():
        raise ValueError("instrument_key is required")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    if forward_candles < 1:
        raise ValueError("forward_candles must be >= 1")
    if neutral_threshold_percent < 0:
        raise ValueError("neutral_threshold_percent must be >= 0")

    prepared = await prepare_walk_forward_data(
        instrument_key.strip(),
        unit,
        interval,
        from_date,
        to_date,
        history_days=history_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    evaluated = await evaluate_walk_forward_signals(
        prepared,
        min_context_candles=min_context_candles,
    )

    candles = _all_prepared_candles(prepared)
    evaluated_windows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    outcome_skipped = 0

    for window in evaluated["windows"]:
        rows: List[Dict[str, Any]] = []
        for signal in window["signals"]:
            outcome = _forward_outcome(
                candles,
                signal["timestamp"],
                forward_candles,
                neutral_threshold_percent,
            )
            if outcome is None:
                outcome_skipped += 1
                continue

            rf_signal = (signal.get("random_forest") or {}).get("signal")
            lstm_signal = (signal.get("lstm") or {}).get("signal")
            agreement = (
                rf_signal is not None
                and lstm_signal is not None
                and rf_signal == lstm_signal
                and rf_signal != "neutral"
            )
            conflict = (
                rf_signal is not None
                and lstm_signal is not None
                and rf_signal != lstm_signal
            )

            row = {
                **signal,
                **outcome,
                "correct": _signal_correct(signal["signal"], outcome["actual_direction"]),
                "model_agreement": agreement,
                "model_conflict": conflict,
            }
            rows.append(row)
            all_rows.append(row)

        metrics = _calculate_period_metrics(rows, initial_capital=initial_capital, forward_candles=forward_candles)
        evaluated_windows.append({
            "window_id": window.get("window_id"),
            "history_from_date": window.get("history_from_date"),
            "history_to_date": window.get("history_to_date"),
            "validation_from_date": window.get("validation_from_date"),
            "validation_to_date": window.get("validation_to_date"),
            "metrics": {k: v for k, v in metrics.items() if k != "equity_curve"},
            "equity_curve": metrics["equity_curve"],
            "signals": rows,
            "source_signal_count": len(window["signals"]),
        })

    total = len(all_rows)
    correct = sum(1 for r in all_rows if r["correct"])
    positive = sum(1 for r in all_rows if r["forward_return_percent"] > 0)
    agreement_rows = [r for r in all_rows if r["model_agreement"]]
    conflict_rows = [r for r in all_rows if r["model_conflict"]]

    # Use the same sequential signal-following simulation as the period metrics.
    overall_metrics = _calculate_period_metrics(all_rows, initial_capital=initial_capital, forward_candles=forward_candles)

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 2",
        "analysis": "Walk-Forward Validation",
        "instrument_key": prepared["instrument_key"],
        "period": {
            "from_date": prepared["requested_from_date"],
            "to_date": prepared["requested_to_date"],
            "data_from_date": prepared["data_from_date"],
        },
        "configuration": {
            "history_days": history_days,
            "validation_days": validation_days,
            "step_days": step_days,
            "forward_candles": forward_candles,
            "neutral_threshold_percent": neutral_threshold_percent,
            "initial_capital": initial_capital,
            "model_context_candles": min_context_candles,
            "models_retrained": False,
            "execution_note": (
                "Signals are generated chronologically using only information "
                "available at each timestamp. Future candles are used only "
                "afterward for outcome measurement."
            ),
        },
        "metrics": {
            "window_count": len(evaluated_windows),
            "evaluation_points": total,
            "correct_signals": correct,
            "directional_accuracy_percent": _safe_percent(correct, total),
            "positive_forward_return_rate_percent": _safe_percent(positive, total),
            "average_forward_return_percent": round(
                sum(float(r["forward_return_percent"]) for r in all_rows) / total, 4
            ) if total else 0.0,
            "agreement_points": len(agreement_rows),
            "conflict_points": len(conflict_rows),
            "agreement_accuracy_percent": _safe_percent(
                sum(1 for r in agreement_rows if r["correct"]), len(agreement_rows)
            ),
            "conflict_accuracy_percent": _safe_percent(
                sum(1 for r in conflict_rows if r["correct"]), len(conflict_rows)
            ),
            "simulated_final_capital": overall_metrics["simulated_final_capital"],
            "simulated_return_percent": overall_metrics["simulated_return_percent"],
            "simulated_trade_points": overall_metrics["simulated_trade_points"],
            "simulated_max_drawdown": overall_metrics["max_drawdown_amount"],
            "simulated_max_drawdown_amount": overall_metrics["max_drawdown_amount"],
            "simulated_max_drawdown_percent": overall_metrics["max_drawdown_percent"],
            "outcome_skipped_points": outcome_skipped,
            "model_error_points": evaluated["metrics"]["model_error_points"],
        },
        "overall_equity_curve": overall_metrics["equity_curve"],
        "windows": evaluated_windows,
        "validation_notes": [
            "Part 1 creates chronological history and validation windows.",
            "Part 2 generates AI signals from the existing Phase 9 Random Forest and LSTM models without retraining.",
            "The current signal candle is never used to generate its own signal before the signal is created.",
            "Future candles are used only after the signal timestamp to measure forward outcome.",
            "Walk-forward metrics are validation measurements and do not guarantee future returns.",
        ],
    }
