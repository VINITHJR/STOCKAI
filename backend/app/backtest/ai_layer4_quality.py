"""
Phase 10 / Layer 4 / Part 2
Signal Quality & Breakdown
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def _num(value: Any) -> Optional[float]:

    if value is None or isinstance(value, bool):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value != value:
        return None

    if value in (
        float("inf"),
        float("-inf"),
    ):
        return None

    return value


def _average(
    values: List[float],
) -> Optional[float]:

    if not values:
        return None

    return sum(values) / len(values)


def _group_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:

    resolved = [
        record
        for record in records
        if isinstance(
            record.get("correct"),
            bool,
        )
    ]

    correct = [
        record
        for record in resolved
        if record["correct"] is True
    ]

    incorrect = [
        record
        for record in resolved
        if record["correct"] is False
    ]

    returns = [
        value
        for value in (
            _num(
                record.get(
                    "forward_return_percent"
                )
            )
            for record in records
        )
        if value is not None
    ]

    confidences = [
        value
        for value in (
            _num(
                record.get("confidence")
            )
            for record in records
        )
        if value is not None
    ]

    return {
        "signal_points": len(records),

        "resolved_points": len(resolved),

        "correct_points": len(correct),

        "incorrect_points": len(incorrect),

        "accuracy_percent": (
            len(correct) / len(resolved) * 100.0
            if resolved
            else None
        ),

        "average_forward_return_percent": (
            _average(returns)
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
            sum(value > 0 for value in returns)
            / len(returns)
            * 100.0
            if returns
            else None
        ),

        "average_confidence": (
            _average(confidences)
        ),
    }


def _groups(
    records: List[Dict[str, Any]],
    field: str,
):

    result = defaultdict(list)

    for record in records:

        result[
            str(record.get(field))
        ].append(record)

    return dict(result)


def _summarize(
    groups,
):

    return {
        key: _group_metrics(records)
        for key, records in sorted(
            groups.items()
        )
    }


def _agreement(
    records: List[Dict[str, Any]],
):

    agreement = [
        record
        for record in records
        if record.get(
            "model_agreement"
        ) is True
    ]

    conflict = [
        record
        for record in records
        if record.get(
            "model_conflict"
        ) is True
    ]

    neither = [
        record
        for record in records
        if (
            record.get(
                "model_agreement"
            ) is not True
            and
            record.get(
                "model_conflict"
            ) is not True
        )
    ]

    return {
        "agreement": _group_metrics(
            agreement
        ),

        "conflict": _group_metrics(
            conflict
        ),

        "neither": _group_metrics(
            neither
        ),
    }


def _periods(
    records: List[Dict[str, Any]],
):

    groups = _groups(
        records,
        "period_id",
    )

    result = {}

    for key in sorted(
        groups,
        key=lambda x: (
            int(x)
            if str(x).lstrip("-").isdigit()
            else str(x)
        ),
    ):

        period_records = groups[key]

        distribution = defaultdict(int)

        for record in period_records:

            signal = str(
                record.get(
                    "signal",
                    "neutral",
                )
            )

            distribution[signal] += 1

        metrics = _group_metrics(
            period_records
        )

        metrics[
            "signal_distribution"
        ] = dict(
            sorted(distribution.items())
        )

        result[key] = metrics

    return result


def _extremes(
    records: List[Dict[str, Any]],
    limit: int,
):

    resolved = [
        record
        for record in records
        if _num(
            record.get(
                "forward_return_percent"
            )
        ) is not None
    ]

    def compact(record):

        return {
            "period_id": record.get(
                "period_id"
            ),

            "timestamp": record.get(
                "timestamp"
            ),

            "signal": record.get(
                "signal"
            ),

            "confidence": record.get(
                "confidence"
            ),

            "forward_return_percent":
                record.get(
                    "forward_return_percent"
                ),

            "actual_direction":
                record.get(
                    "actual_direction"
                ),

            "correct":
                record.get("correct"),
        }

    best = sorted(
        resolved,
        key=lambda record:
            _num(
                record.get(
                    "forward_return_percent"
                )
            ) or float("-inf"),
        reverse=True,
    )[:limit]

    worst = sorted(
        resolved,
        key=lambda record:
            _num(
                record.get(
                    "forward_return_percent"
                )
            ) or float("inf"),
    )[:limit]

    return {
        "best_forward_return_signals": [
            compact(record)
            for record in best
        ],

        "worst_forward_return_signals": [
            compact(record)
            for record in worst
        ],
    }


def analyze_signal_quality(
    part1_result: Dict[str, Any],
    *,
    extreme_signal_limit: int = 5,
) -> Dict[str, Any]:

    if not isinstance(
        part1_result,
        dict,
    ):
        raise TypeError(
            "part1_result must be a dictionary"
        )

    if extreme_signal_limit < 1:
        raise ValueError(
            "extreme_signal_limit must be >= 1"
        )

    records = (
        part1_result.get("records")
        or []
    )

    if not isinstance(
        records,
        list,
    ):
        raise ValueError(
            "part1_result.records must be a list"
        )

    records = [
        record
        for record in records
        if isinstance(record, dict)
    ]

    resolved = sum(
        isinstance(
            record.get("correct"),
            bool,
        )
        for record in records
    )

    correct = sum(
        record.get("correct") is True
        for record in records
    )

    incorrect = sum(
        record.get("correct") is False
        for record in records
    )

    positive = sum(
        1
        for record in records
        if (
            _num(
                record.get(
                    "forward_return_percent"
                )
            ) is not None
            and
            _num(
                record.get(
                    "forward_return_percent"
                )
            ) > 0
        )
    )

    negative = sum(
        1
        for record in records
        if (
            _num(
                record.get(
                    "forward_return_percent"
                )
            ) is not None
            and
            _num(
                record.get(
                    "forward_return_percent"
                )
            ) < 0
        )
    )

    return {
        "status": "success",

        "layer": "Phase 10 / Layer 4",

        "part": "Part 2",

        "analysis": (
            "Signal Quality & Breakdown"
        ),

        "instrument_key":
            part1_result.get(
                "instrument_key"
            ),

        "source":
            part1_result.get(
                "source",
                {},
            ),

        "quality_summary": {

            "signal_points":
                len(records),

            "resolved_points":
                resolved,

            "correct_points":
                correct,

            "incorrect_points":
                incorrect,

            "accuracy_percent":
                (
                    correct / resolved * 100.0
                    if resolved
                    else None
                ),

            "positive_return_points":
                positive,

            "negative_return_points":
                negative,

            "agreement_points":
                sum(
                    record.get(
                        "model_agreement"
                    ) is True
                    for record in records
                ),

            "conflict_points":
                sum(
                    record.get(
                        "model_conflict"
                    ) is True
                    for record in records
                ),

            "future_data_used_for_signal_points":
                sum(
                    record.get(
                        "future_data_used_for_signal"
                    ) is True
                    for record in records
                ),
        },

        "by_signal":
            _summarize(
                _groups(
                    records,
                    "signal",
                )
            ),

        "by_confidence_bucket":
            _summarize(
                _groups(
                    records,
                    "confidence_bucket",
                )
            ),

        "by_return_bucket":
            _summarize(
                _groups(
                    records,
                    "return_bucket",
                )
            ),

        "by_outcome":
            _summarize(
                _groups(
                    records,
                    "outcome",
                )
            ),

        "by_model_agreement_conflict":
            _agreement(records),

        "by_period":
            _periods(records),

        "extreme_outcomes":
            _extremes(
                records,
                extreme_signal_limit,
            ),
    }


__all__ = [
    "analyze_signal_quality"
]