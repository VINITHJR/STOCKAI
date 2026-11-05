"""
Phase 10 / Layer 5 / Part 1
AI vs Baseline - Baseline Engine

This module creates deterministic non-AI baselines from the existing
Phase 10 Layer 3 Part 2 evaluation output.

Baselines:
1. always_bullish  - predicts bullish at every resolved signal point
2. always_bearish  - predicts bearish at every resolved signal point
3. buy_and_hold    - period-level market return using the first and last
                     available validation prices in the Layer 3 signal data

The baselines use exactly the same forward outcomes already produced by
Layer 3. They do not fetch new data, retrain models, or modify any previous
Phase 10 layer.

Part 1 only builds baseline observations and metrics. Part 2 will perform
the actual AI-vs-baseline comparison.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Dict, List, Optional


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _direction(forward_return: Optional[float], threshold: float) -> str:
    if forward_return is None:
        return "unresolved"
    if forward_return > threshold:
        return "bullish"
    if forward_return < -threshold:
        return "bearish"
    return "neutral"


def _resolved_records(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for period in evaluation.get("periods") or []:
        if not isinstance(period, dict):
            continue
        for signal in period.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            if _num(signal.get("forward_return_percent")) is None:
                continue
            records.append({
                **signal,
                "period_id": period.get("period_id"),
            })
    return records


def _accuracy(predictions: List[str], actuals: List[str]) -> Optional[float]:
    pairs = [
        (p, a)
        for p, a in zip(predictions, actuals)
        if a != "unresolved" and p != "unresolved"
    ]
    if not pairs:
        return None
    return sum(p == a for p, a in pairs) / len(pairs) * 100.0


def _period_records(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("period_id"))].append(record)
    return dict(groups)


def _period_buy_hold(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate a period return from the earliest current_price and the latest
    future_price available in that period.

    This is intentionally a simple descriptive market baseline, not a
    tradable execution simulation.
    """
    if not records:
        return {
            "status": "unresolved",
            "return_percent": None,
            "start_price": None,
            "end_price": None,
        }

    ordered = sorted(records, key=lambda r: str(r.get("timestamp", "")))

    start_price = _num(ordered[0].get("current_price"))
    end_price = _num(ordered[-1].get("future_price"))

    if start_price is None or end_price is None or start_price == 0:
        return {
            "status": "unresolved",
            "return_percent": None,
            "start_price": start_price,
            "end_price": end_price,
        }

    return {
        "status": "success",
        "return_percent": (end_price - start_price) / start_price * 100.0,
        "start_price": start_price,
        "end_price": end_price,
    }


def build_baseline_analysis(
    evaluation: Dict[str, Any],
    *,
    neutral_threshold_percent: float = 1.0,
) -> Dict[str, Any]:
    """
    Build baseline observations aligned to the Layer 3 signal timestamps.

    The returned records retain the original forward outcome and add
    deterministic baseline directions. No AI prediction is changed.
    """
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a dictionary")
    if not isfinite(float(neutral_threshold_percent)) or neutral_threshold_percent < 0:
        raise ValueError("neutral_threshold_percent must be a finite value >= 0")

    records = _resolved_records(evaluation)

    observations: List[Dict[str, Any]] = []

    for record in records:
        forward_return = _num(record.get("forward_return_percent"))
        actual = _direction(forward_return, neutral_threshold_percent)

        observations.append({
            "period_id": record.get("period_id"),
            "timestamp": record.get("timestamp"),
            "current_price": _num(record.get("current_price")),
            "future_price": _num(record.get("future_price")),
            "forward_return_percent": forward_return,
            "actual_direction": actual,
            "always_bullish_prediction": "bullish",
            "always_bearish_prediction": "bearish",
            "threshold_actual_direction": actual,
            "future_data_used_for_signal": bool(
                record.get("future_data_used_for_signal", False)
            ),
        })

    actuals = [x["threshold_actual_direction"] for x in observations]
    bullish_predictions = ["bullish"] * len(observations)
    bearish_predictions = ["bearish"] * len(observations)

    period_baselines: Dict[str, Any] = {}
    for period_id, period_items in _period_records(observations).items():
        period_baselines[period_id] = {
            "signal_points": len(period_items),
            "always_bullish_accuracy_percent": _accuracy(
                ["bullish"] * len(period_items),
                [x["threshold_actual_direction"] for x in period_items],
            ),
            "always_bearish_accuracy_percent": _accuracy(
                ["bearish"] * len(period_items),
                [x["threshold_actual_direction"] for x in period_items],
            ),
            "buy_and_hold": _period_buy_hold(
                [r for r in records if str(r.get("period_id")) == period_id]
            ),
        }

    positive = sum(
        1 for x in observations
        if x["forward_return_percent"] is not None
        and x["forward_return_percent"] > neutral_threshold_percent
    )
    negative = sum(
        1 for x in observations
        if x["forward_return_percent"] is not None
        and x["forward_return_percent"] < -neutral_threshold_percent
    )
    neutral = len(observations) - positive - negative

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 5",
        "part": "Part 1",
        "analysis": "AI vs Baseline - Baseline Engine",
        "instrument_key": evaluation.get("instrument_key"),
        "configuration": {
            "neutral_threshold_percent": neutral_threshold_percent,
            "baseline_definitions": {
                "always_bullish": "Predict bullish at every resolved evaluation point.",
                "always_bearish": "Predict bearish at every resolved evaluation point.",
                "buy_and_hold": (
                    "Descriptive period return from the first available "
                    "validation current price to the last available future price."
                ),
            },
            "data_source": "Phase 10 Layer 3 Part 2 evaluation",
            "future_data_used_to_create_baseline": False,
        },
        "summary": {
            "resolved_signal_points": len(observations),
            "bullish_outcome_points": positive,
            "bearish_outcome_points": negative,
            "neutral_outcome_points": neutral,
            "always_bullish_accuracy_percent": _accuracy(
                bullish_predictions, actuals
            ),
            "always_bearish_accuracy_percent": _accuracy(
                bearish_predictions, actuals
            ),
            "future_data_used_for_baseline_points": sum(
                x["future_data_used_for_signal"] for x in observations
            ),
        },
        "periods": period_baselines,
        "observations": observations,
    }


__all__ = ["build_baseline_analysis"]
