"""
Phase 10 / Layer 5 / Part 2
AI vs Baseline - Comparison Engine

Compares the existing Phase 10 Layer 3 AI signal records against the
deterministic baselines created by Layer 5 Part 1.

Comparison target:
- The same threshold-based actual direction is used for AI and baselines.
- Only Layer 3 forward outcomes are used as historical outcomes.
- No new market-data request and no model retraining.

This part is descriptive: it reports side-by-side metrics and differences.
It does not select or rank a strategy as an investment recommendation.
"""

from __future__ import annotations

from collections import defaultdict
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


def _records(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for period in evaluation.get("periods") or []:
        if not isinstance(period, dict):
            continue
        period_id = period.get("period_id")
        for signal in period.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            forward_return = _num(signal.get("forward_return_percent"))
            if forward_return is None:
                continue
            records.append({
                **signal,
                "period_id": period_id,
            })
    return records


def _accuracy(predictions: List[str], actuals: List[str]) -> Optional[float]:
    pairs = [
        (p, a)
        for p, a in zip(predictions, actuals)
        if p != "unresolved" and a != "unresolved"
    ]
    if not pairs:
        return None
    return sum(p == a for p, a in pairs) / len(pairs) * 100.0


def _average(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _strategy_metrics(
    predictions: List[str],
    actuals: List[str],
    returns: List[float],
) -> Dict[str, Any]:
    accuracy = _accuracy(predictions, actuals)
    resolved = [
        (p, a, r)
        for p, a, r in zip(predictions, actuals, returns)
        if p != "unresolved" and a != "unresolved"
    ]

    directional_correct = sum(p == a for p, a, _ in resolved)

    # A simple signal-aligned return measure:
    # bullish => +forward return, bearish => -forward return, neutral => 0.
    # This is descriptive, not a transaction-cost/execution simulation.
    aligned_returns = []
    for prediction, ret in zip(predictions, returns):
        if prediction == "bullish":
            aligned_returns.append(ret)
        elif prediction == "bearish":
            aligned_returns.append(-ret)
        else:
            aligned_returns.append(0.0)

    positive = sum(x > 0 for x in aligned_returns)
    negative = sum(x < 0 for x in aligned_returns)

    return {
        "signal_points": len(predictions),
        "resolved_points": len(resolved),
        "correct_direction_points": directional_correct,
        "incorrect_direction_points": len(resolved) - directional_correct,
        "directional_accuracy_percent": accuracy,
        "average_forward_return_percent": _average(returns),
        "average_signal_aligned_return_percent": _average(aligned_returns),
        "positive_aligned_return_points": positive,
        "negative_aligned_return_points": negative,
        "positive_aligned_return_rate_percent": (
            positive / len(aligned_returns) * 100.0
            if aligned_returns else None
        ),
    }


def _period_comparison(
    records: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("period_id"))].append(record)

    result = {}
    for period_id in sorted(
        grouped,
        key=lambda x: int(x) if x.lstrip("-").isdigit() else x,
    ):
        items = grouped[period_id]
        actuals = [
            _direction(_num(r.get("forward_return_percent")), threshold)
            for r in items
        ]
        returns = [_num(r.get("forward_return_percent")) or 0.0 for r in items]

        ai_predictions = [
            str(r.get("signal") or "neutral").strip().lower()
            for r in items
        ]

        result[period_id] = {
            "signal_points": len(items),
            "ai": _strategy_metrics(ai_predictions, actuals, returns),
            "always_bullish": _strategy_metrics(
                ["bullish"] * len(items), actuals, returns
            ),
            "always_bearish": _strategy_metrics(
                ["bearish"] * len(items), actuals, returns
            ),
        }

    return result


def compare_ai_vs_baseline(
    evaluation: Dict[str, Any],
    baseline_analysis: Dict[str, Any],
    *,
    neutral_threshold_percent: float = 1.0,
) -> Dict[str, Any]:
    """
    Compare AI, always-bullish, and always-bearish predictions using the
    exact same Layer 3 evaluation records.

    baseline_analysis is accepted explicitly so the comparison is tied to
    the output of Layer 5 Part 1. Its observations are validated by their
    count/order before comparison.
    """
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a dictionary")
    if not isinstance(baseline_analysis, dict):
        raise TypeError("baseline_analysis must be a dictionary")
    if neutral_threshold_percent < 0:
        raise ValueError("neutral_threshold_percent must be >= 0")

    records = _records(evaluation)
    baseline_observations = baseline_analysis.get("observations") or []

    if len(baseline_observations) != len(records):
        raise ValueError(
            "Layer 5 Part 1 observation count does not match Layer 3 "
            "evaluation record count"
        )

    for index, (record, observation) in enumerate(zip(records, baseline_observations)):
        if (str(record.get("period_id")) != str(observation.get("period_id"))
                or str(record.get("timestamp") or "") != str(observation.get("timestamp") or "")):
            raise ValueError(
                f"Layer 5 Part 1 observation mismatch at index {index}: "
                "period_id/timestamp do not match Layer 3."
            )

    actuals = [
        _direction(_num(r.get("forward_return_percent")), neutral_threshold_percent)
        for r in records
    ]
    returns = [_num(r.get("forward_return_percent")) or 0.0 for r in records]
    ai_predictions = [
        (
            str(r.get("signal") or "neutral").strip().lower()
            if str(r.get("signal") or "neutral").strip().lower()
            in {"bullish", "bearish", "neutral"}
            else "neutral"
        )
        for r in records
    ]

    ai = _strategy_metrics(ai_predictions, actuals, returns)
    always_bullish = _strategy_metrics(
        ["bullish"] * len(records), actuals, returns
    )
    always_bearish = _strategy_metrics(
        ["bearish"] * len(records), actuals, returns
    )

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 5",
        "part": "Part 2",
        "analysis": "AI vs Baseline Comparison",
        "instrument_key": evaluation.get("instrument_key"),
        "configuration": {
            "neutral_threshold_percent": neutral_threshold_percent,
            "comparison_points": len(records),
            "same_historical_outcomes_for_all_methods": True,
            "no_new_market_data_used": True,
            "models_retrained": (
                (evaluation.get("configuration") or {}).get("models_retrained")
            ),
        },
        "comparison": {
            "ai": ai,
            "always_bullish": always_bullish,
            "always_bearish": always_bearish,
        },
        "periods": _period_comparison(records, neutral_threshold_percent),
    }


__all__ = ["compare_ai_vs_baseline"]
