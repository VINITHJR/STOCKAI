"""
Phase 10 / Layer 4 / Part 1
Signal Performance Analysis

Consumes ONLY the completed Phase 10 Layer 3 evaluation.

Rules:
- Does not fetch market data.
- Does not retrain models.
- Does not modify Layer 3.
- Preserves every Layer 3 signal.
- Resolved means Layer 3 supplied a boolean `correct`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional


SIGNAL_TYPES = ("bullish", "bearish", "neutral")


# ============================================================
# Helpers
# ============================================================

def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if isfinite(value) else None


def _int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def _normalize_signal(value: Any) -> str:
    signal = str(value or "neutral").strip().lower()

    if signal in SIGNAL_TYPES:
        return signal

    return "neutral"


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in SIGNAL_TYPES:
        return value

    return None


def _confidence_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unknown"

    if value < 0.40:
        return "low"

    if value < 0.70:
        return "medium"

    if value < 0.85:
        return "high"

    return "very_high"


def _return_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unresolved"

    if value <= -3.0:
        return "large_loss"

    if value < 0:
        return "loss"

    if value < 3.0:
        return "small_gain"

    return "large_gain"


def _outcome(correct: Optional[bool]) -> str:
    """
    IMPORTANT:
    A record is resolved ONLY when Layer 3 explicitly provides
    a boolean `correct`.

    We do not invent correctness from forward return.
    """
    if correct is True:
        return "correct"

    if correct is False:
        return "incorrect"

    return "unresolved"


# ============================================================
# Layer 3 traversal
# ============================================================

def _iter_periods(
    evaluation: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:

    periods = evaluation.get("periods") or []

    if not isinstance(periods, list):
        return []

    return (
        period
        for period in periods
        if isinstance(period, dict)
    )


def _iter_signals(
    evaluation: Dict[str, Any],
) -> Iterable[tuple[int, Dict[str, Any]]]:

    for period in _iter_periods(evaluation):

        period_id = _int(period.get("period_id"))

        if period_id is None:
            period_id = 0

        signals = period.get("signals") or []

        if not isinstance(signals, list):
            continue

        for signal in signals:

            if isinstance(signal, dict):
                yield period_id, signal


# ============================================================
# Normalized record
# ============================================================

@dataclass
class SignalPerformanceRecord:

    period_id: int
    timestamp: str

    signal: str
    confidence: Optional[float]

    current_price: Optional[float]
    future_price: Optional[float]

    forward_return_percent: Optional[float]

    actual_direction: Optional[str]
    correct: Optional[bool]

    model_agreement: bool
    model_conflict: bool

    available_candle_count: Optional[int]

    future_data_used_for_signal: bool

    outcome: str

    confidence_bucket: str
    return_bucket: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Build normalized records
# ============================================================

def build_signal_performance_records(
    evaluation: Dict[str, Any],
) -> List[SignalPerformanceRecord]:

    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a dictionary")

    records: List[SignalPerformanceRecord] = []

    for period_id, item in _iter_signals(evaluation):

        confidence = _number(item.get("confidence"))

        if confidence is not None:
            confidence = max(
                0.0,
                min(1.0, confidence),
            )

        forward_return = _number(
            item.get("forward_return_percent")
        )

        correct = item.get("correct")

        if not isinstance(correct, bool):
            correct = None

        actual_direction = _normalize_direction(
            item.get("actual_direction")
        )

        record = SignalPerformanceRecord(
            period_id=period_id,

            timestamp=str(
                item.get("timestamp") or ""
            ),

            signal=_normalize_signal(
                item.get("signal")
            ),

            confidence=confidence,

            current_price=_number(
                item.get("current_price")
            ),

            future_price=_number(
                item.get("future_price")
            ),

            forward_return_percent=forward_return,

            actual_direction=actual_direction,

            correct=correct,

            model_agreement=_bool(
                item.get("model_agreement")
            ),

            model_conflict=_bool(
                item.get("model_conflict")
            ),

            available_candle_count=_int(
                item.get("available_candle_count")
            ),

            future_data_used_for_signal=_bool(
                item.get(
                    "future_data_used_for_signal"
                )
            ),

            outcome=_outcome(correct),

            confidence_bucket=_confidence_bucket(
                confidence
            ),

            return_bucket=_return_bucket(
                forward_return
            ),
        )

        records.append(record)

    return records


# ============================================================
# Group summary
# ============================================================

def _average(
    values: List[float],
) -> Optional[float]:

    if not values:
        return None

    return sum(values) / len(values)


def _group_summary(
    records: List[SignalPerformanceRecord],
) -> Dict[str, Any]:

    signal_points = len(records)

    resolved = [
        record
        for record in records
        if isinstance(record.correct, bool)
    ]

    correct = [
        record
        for record in resolved
        if record.correct is True
    ]

    incorrect = [
        record
        for record in resolved
        if record.correct is False
    ]

    returns = [
        record.forward_return_percent
        for record in records
        if record.forward_return_percent is not None
    ]

    confidences = [
        record.confidence
        for record in records
        if record.confidence is not None
    ]

    return {
        "signal_points": signal_points,

        "resolved_points": len(resolved),

        "correct_points": len(correct),

        "incorrect_points": len(incorrect),

        "accuracy_percent": (
            len(correct) / len(resolved) * 100.0
            if resolved
            else None
        ),

        "average_forward_return_percent": _average(
            returns
        ),

        "positive_forward_return_points": sum(
            value > 0
            for value in returns
        ),

        "negative_forward_return_points": sum(
            value < 0
            for value in returns
        ),

        "positive_forward_return_rate_percent": (
            sum(value > 0 for value in returns) / len(returns) * 100.0
            if returns else None
        ),

        "negative_forward_return_rate_percent": (
            sum(value < 0 for value in returns) / len(returns) * 100.0
            if returns else None
        ),

        "average_confidence": _average(
            confidences
        ),
    }


# ============================================================
# Main Layer 4 Part 1
# ============================================================

def analyze_signal_performance(
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:

    records = build_signal_performance_records(
        evaluation
    )

    by_signal = defaultdict(list)
    by_period = defaultdict(list)
    by_outcome = defaultdict(list)
    by_confidence = defaultdict(list)
    by_return = defaultdict(list)

    for record in records:

        by_signal[
            record.signal
        ].append(record)

        by_period[
            record.period_id
        ].append(record)

        by_outcome[
            record.outcome
        ].append(record)

        by_confidence[
            record.confidence_bucket
        ].append(record)

        by_return[
            record.return_bucket
        ].append(record)

    def summarize_groups(groups):

        return {
            str(key): _group_summary(items)
            for key, items in sorted(
                groups.items(),
                key=lambda item: str(item[0]),
            )
        }

    resolved = sum(
        record.correct is not None
        for record in records
    )

    correct = sum(
        record.correct is True
        for record in records
    )

    incorrect = sum(
        record.correct is False
        for record in records
    )

    agreement = sum(
        record.model_agreement
        for record in records
    )

    conflict = sum(
        record.model_conflict
        for record in records
    )

    future_data = sum(
        record.future_data_used_for_signal
        for record in records
    )

    return {
        "status": "success",

        "layer": "Phase 10 / Layer 4",

        "part": "Part 1",

        "analysis": (
            "Signal Performance Analysis"
        ),

        "instrument_key": evaluation.get(
            "instrument_key"
        ),

        "source": {
            "layer": evaluation.get("layer"),
            "part": evaluation.get("part"),
            "period_count": len(
                list(_iter_periods(evaluation))
            ),
            "models_retrained": (
                (
                    evaluation.get(
                        "configuration"
                    ) or {}
                ).get("models_retrained")
            ),
        },

        "summary": {
            "signal_points": len(records),

            "resolved_points": resolved,

            "correct_points": correct,

            "incorrect_points": incorrect,

            "accuracy_percent": (
                correct / resolved * 100.0
                if resolved
                else None
            ),

            "average_forward_return_percent": (
                _average([
                    record.forward_return_percent
                    for record in records
                    if record.forward_return_percent is not None
                ])
            ),

            "positive_forward_return_points": sum(
                1
                for record in records
                if record.forward_return_percent is not None
                and record.forward_return_percent > 0
            ),

            "negative_forward_return_points": sum(
                1
                for record in records
                if record.forward_return_percent is not None
                and record.forward_return_percent < 0
            ),

            "positive_forward_return_rate_percent": (
                sum(
                    1
                    for record in records
                    if record.forward_return_percent is not None
                    and record.forward_return_percent > 0
                )
                / sum(
                    1
                    for record in records
                    if record.forward_return_percent is not None
                )
                * 100.0
                if any(record.forward_return_percent is not None for record in records)
                else None
            ),

            "negative_forward_return_rate_percent": (
                sum(
                    1
                    for record in records
                    if record.forward_return_percent is not None
                    and record.forward_return_percent < 0
                )
                / sum(
                    1
                    for record in records
                    if record.forward_return_percent is not None
                )
                * 100.0
                if any(record.forward_return_percent is not None for record in records)
                else None
            ),

            "agreement_points": agreement,

            "conflict_points": conflict,

            "future_data_used_for_signal_points": (
                future_data
            ),

            "model_consistency_points": agreement + conflict,

            "outcome_counts": dict(
                Counter(
                    record.outcome
                    for record in records
                )
            ),
        },

        "by_signal": summarize_groups(
            by_signal
        ),

        "by_period": summarize_groups(
            by_period
        ),

        "by_outcome": summarize_groups(
            by_outcome
        ),

        "by_confidence_bucket": summarize_groups(
            by_confidence
        ),

        "by_return_bucket": summarize_groups(
            by_return
        ),

        "records": [
            record.to_dict()
            for record in records
        ],
    }


__all__ = [
    "SignalPerformanceRecord",
    "build_signal_performance_records",
    "analyze_signal_performance",
]