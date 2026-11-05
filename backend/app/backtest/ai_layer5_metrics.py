"""Phase 10 / Layer 5 / Part 3: AI vs Baseline Metrics API."""

from __future__ import annotations

from typing import Any, Dict, Optional


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


def _comparison_delta(ai: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Return descriptive AI-minus-baseline differences."""
    fields = (
        "directional_accuracy_percent",
        "average_signal_aligned_return_percent",
        "positive_aligned_return_rate_percent",
    )
    result: Dict[str, Any] = {}
    for field in fields:
        a = _num(ai.get(field))
        b = _num(baseline.get(field))
        result[field] = (a - b) if a is not None and b is not None else None
    return result


def aggregate_ai_baseline_metrics(
    part2_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert the Layer 5 Part 2 comparison into compact API-ready metrics.

    No new market data is fetched and no model is retrained.
    """
    if not isinstance(part2_result, dict):
        raise TypeError("part2_result must be a dictionary")

    comparison = part2_result.get("comparison") or {}
    if not isinstance(comparison, dict):
        raise ValueError("Layer 5 Part 2 comparison must be an object")
    ai = comparison.get("ai") or {}
    bullish = comparison.get("always_bullish") or {}
    bearish = comparison.get("always_bearish") or {}

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 5",
        "part": "Part 3",
        "analysis": "AI vs Baseline Metrics API",
        "instrument_key": part2_result.get("instrument_key"),
        "configuration": part2_result.get("configuration") or {},
        "overall": {
            "comparison_points": ai.get("signal_points", 0),
            "ai": ai,
            "always_bullish": bullish,
            "always_bearish": bearish,
            "ai_minus_always_bullish": _comparison_delta(ai, bullish),
            "ai_minus_always_bearish": _comparison_delta(ai, bearish),
        },
        "periods": part2_result.get("periods") or {},
    }


__all__ = ["aggregate_ai_baseline_metrics"]
