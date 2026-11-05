"""
Phase 10 / Layer 4 / Part 3
Signal Performance Metrics
"""

from __future__ import annotations

from typing import Any, Dict


def aggregate_signal_quality_metrics(
    part2_result: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(
        part2_result,
        dict,
    ):
        raise TypeError(
            "part2_result must be a dictionary"
        )

    # Layer 4 Part 1 exposes its aggregate values under ``summary``.
    # Accept both the canonical Part 1 shape and an already-aggregated
    # ``overall`` shape so the adapter is safe for both direct and
    # orchestrated calls.
    summary = (
        part2_result.get("summary")
        or part2_result.get("quality_summary")
        or part2_result.get("overall")
        or {}
    )

    return {
        "status": "success",

        "layer": "Phase 10 / Layer 4",

        "part": "Part 3",

        "analysis":
            "Signal Performance Metrics",

        "instrument_key":
            part2_result.get(
                "instrument_key"
            ),

        "source":
            part2_result.get(
                "source",
                {},
            ),

        "overall": {

            "signal_points":
                summary.get(
                    "signal_points",
                    0,
                ),

            "resolved_points":
                summary.get(
                    "resolved_points",
                    0,
                ),

            "correct_points":
                summary.get(
                    "correct_points",
                    0,
                ),

            "incorrect_points":
                summary.get(
                    "incorrect_points",
                    0,
                ),

            "accuracy_percent":
                summary.get(
                    "accuracy_percent"
                ),

            "average_forward_return_percent":
                summary.get(
                    "average_forward_return_percent"
                ),

            "positive_return_points":
                summary.get(
                    "positive_return_points",
                    summary.get("positive_forward_return_points", 0),
                ),

            "negative_return_points":
                summary.get(
                    "negative_return_points",
                    summary.get("negative_forward_return_points", 0),
                ),

            "positive_return_rate_percent":
                summary.get("positive_forward_return_rate_percent"),

            "negative_return_rate_percent":
                summary.get("negative_forward_return_rate_percent"),

            "agreement_points":
                summary.get(
                    "agreement_points",
                    0,
                ),

            "conflict_points":
                summary.get(
                    "conflict_points",
                    0,
                ),

            "future_data_used_for_signal_points":
                summary.get(
                    "future_data_used_for_signal_points",
                    0,
                ),
        },

        "by_signal":
            part2_result.get(
                "by_signal",
                {},
            ),

        "by_confidence_bucket":
            part2_result.get(
                "by_confidence_bucket",
                {},
            ),

        "by_return_bucket":
            part2_result.get(
                "by_return_bucket",
                {},
            ),

        "by_outcome":
            part2_result.get(
                "by_outcome",
                {},
            ),

        "by_model_agreement_conflict":
            part2_result.get(
                "by_model_agreement_conflict",
                {},
            ),

        "by_period":
            part2_result.get(
                "by_period",
                {},
            ),

        "extreme_outcomes":
            part2_result.get(
                "extreme_outcomes",
                {},
            ),
    }


__all__ = [
    "aggregate_signal_quality_metrics"
]