"""
Phase 10 / Layer 3 / Part 3
Multi-Period Validation - Metrics Aggregation

This module aggregates the period-level signal evaluations produced by
Phase 10 / Layer 3 / Part 2.

It intentionally does not modify Part 1 or Part 2.
No market-data fetching or model inference is performed here.
"""

from __future__ import annotations

from math import isfinite
from statistics import mean, median, pstdev
from typing import Any, Dict, List


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _period_return(period: Dict[str, Any]) -> float:
    signals = period.get("signals") or []
    equity = 100000.0
    traded = 0

    for row in signals:
        signal = str(row.get("signal") or "neutral").lower()
        forward = _number(row.get("forward_return_percent"))
        if signal == "bullish":
            equity *= 1.0 + forward / 100.0
            traded += 1
        elif signal == "bearish":
            equity *= 1.0 - forward / 100.0
            traded += 1

    return (equity / 100000.0 - 1.0) * 100.0 if traded else 0.0


def _non_overlapping_simulation(
    signals: List[Dict[str, Any]],
    initial_capital: float,
) -> Dict[str, Any]:
    """Apply non-overlapping forward outcomes and calculate equity drawdown."""
    rows = sorted(
        [r for r in signals if str(r.get("signal") or "").lower() != "error"],
        key=lambda r: str(r.get("timestamp") or ""),
    )
    equity = float(initial_capital)
    peak = equity
    max_drawdown_amount = 0.0
    max_drawdown_percent = 0.0
    trade_points = 0
    next_available = None
    curve: List[Dict[str, Any]] = []

    for row in rows:
        timestamp = str(row.get("timestamp") or "")
        if next_available is not None and timestamp < next_available:
            continue
        signal = str(row.get("signal") or "neutral").strip().lower()
        if signal not in {"bullish", "bearish"}:
            curve.append({"timestamp": timestamp, "equity": round(equity, 4)})
            continue

        forward_return = _number(row.get("forward_return_percent"))
        trade_return = forward_return / 100.0 if signal == "bullish" else -forward_return / 100.0
        new_equity = equity * (1.0 + trade_return)
        if not isfinite(new_equity) or new_equity < 0:
            continue

        equity = new_equity
        trade_points += 1
        future_timestamp = row.get("future_timestamp")
        next_available = str(future_timestamp) if future_timestamp else timestamp

        peak = max(peak, equity)
        drawdown_amount = max(0.0, peak - equity)
        drawdown_percent = drawdown_amount / peak * 100.0 if peak > 0 else 0.0
        max_drawdown_amount = max(max_drawdown_amount, drawdown_amount)
        max_drawdown_percent = max(max_drawdown_percent, drawdown_percent)

        curve.append({"timestamp": str(future_timestamp or timestamp),
                      "equity": round(equity, 4)})

    return {
        "final_capital": round(equity, 4),
        "return_percent": round(((equity / initial_capital) - 1.0) * 100.0, 4),
        "simulated_trade_points": trade_points,
        "max_drawdown": round(max_drawdown_percent, 4),
        "max_drawdown_percent": round(max_drawdown_percent, 4),
        "max_drawdown_amount": round(max_drawdown_amount, 4),
        "equity_curve": curve,
    }


def _period_metrics(
    period: Dict[str, Any],
    *,
    initial_capital: float,
) -> Dict[str, Any]:
    signals = [
        row for row in (period.get("signals") or [])
        if row.get("signal") != "error"
    ]

    evaluation_points = len(signals)
    correct = sum(1 for row in signals if _bool(row.get("correct")))
    positive = sum(
        1 for row in signals
        if _number(row.get("forward_return_percent")) > 0
    )

    agreement_rows = [r for r in signals if _bool(r.get("model_agreement"))]
    conflict_rows = [r for r in signals if _bool(r.get("model_conflict"))]

    agreement_correct = sum(1 for r in agreement_rows if _bool(r.get("correct")))
    conflict_correct = sum(1 for r in conflict_rows if _bool(r.get("correct")))

    returns = [
        _number(row.get("forward_return_percent"))
        for row in signals
    ]

    simulation = _non_overlapping_simulation(
        signals,
        initial_capital,
    )

    return {
        "evaluation_points": evaluation_points,
        "correct_signals": correct,
        "directional_accuracy_percent": round(
            correct / evaluation_points * 100.0, 4
        ) if evaluation_points else 0.0,
        "positive_forward_return_rate_percent": round(
            positive / evaluation_points * 100.0, 4
        ) if evaluation_points else 0.0,
        "average_forward_return_percent": round(
            mean(returns), 4
        ) if returns else 0.0,
        "median_forward_return_percent": round(
            median(returns), 4
        ) if returns else 0.0,
        "return_std_dev_percent": round(
            pstdev(returns), 4
        ) if len(returns) > 1 else 0.0,
        "agreement_points": len(agreement_rows),
        "conflict_points": len(conflict_rows),
        "agreement_accuracy_percent": round(
            agreement_correct / len(agreement_rows) * 100.0, 4
        ) if agreement_rows else 0.0,
        "conflict_accuracy_percent": round(
            conflict_correct / len(conflict_rows) * 100.0, 4
        ) if conflict_rows else 0.0,
        "model_error_points": int(period.get("model_error_points") or 0),
        "outcome_skipped_points": int(period.get("outcome_skipped_points") or 0),
        "simulated_final_capital": simulation["final_capital"],
        "simulated_return_percent": simulation["return_percent"],
        "simulated_trade_points": simulation["simulated_trade_points"],
        "simulated_max_drawdown": simulation["max_drawdown_percent"],
        "simulated_max_drawdown_percent": simulation["max_drawdown_percent"],
        "simulated_max_drawdown_amount": simulation["max_drawdown_amount"],
        "equity_curve": simulation["equity_curve"],
    }


def aggregate_multi_period_metrics(
    evaluation: Dict[str, Any],
    *,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Aggregate Layer 3 Part 2 results without changing the underlying signals.
    """
    if not isfinite(float(initial_capital)) or initial_capital <= 0:
        raise ValueError("initial_capital must be a finite value > 0")
    period_results: List[Dict[str, Any]] = []

    for period in evaluation.get("periods") or []:
        metrics = _period_metrics(
            period,
            initial_capital=initial_capital,
        )

        period_results.append(
            {
                "period_id": period.get("period_id"),
                "validation_from_date": period.get("validation_from_date"),
                "validation_to_date": period.get("validation_to_date"),
                "data_from_date": period.get("data_from_date"),
                "data_to_date": period.get("data_to_date"),
                "metrics": metrics,
            }
        )

    returns = [
        _number(item["metrics"].get("simulated_return_percent"))
        for item in period_results
    ]
    accuracies = [
        _number(item["metrics"].get("directional_accuracy_percent"))
        for item in period_results
    ]
    drawdowns_percent = [
        _number(item["metrics"].get("simulated_max_drawdown_percent"))
        for item in period_results
    ]
    drawdowns_amount = [
        _number(item["metrics"].get("simulated_max_drawdown_amount"))
        for item in period_results
    ]

    positive_periods = sum(1 for value in returns if value > 0)
    negative_periods = sum(1 for value in returns if value < 0)
    flat_periods = len(returns) - positive_periods - negative_periods

    best = max(
        period_results,
        key=lambda x: _number(x["metrics"].get("simulated_return_percent")),
        default=None,
    )
    worst = min(
        period_results,
        key=lambda x: _number(x["metrics"].get("simulated_return_percent")),
        default=None,
    )

    total_evaluation_points = sum(
        int(item["metrics"].get("evaluation_points") or 0)
        for item in period_results
    )
    total_correct = sum(
        int(item["metrics"].get("correct_signals") or 0)
        for item in period_results
    )

    overall_accuracy = (
        total_correct / total_evaluation_points * 100.0
        if total_evaluation_points else 0.0
    )

    total_trades = sum(
        int(item["metrics"].get("simulated_trade_points") or 0)
        for item in period_results
    )

    overall = {
        "period_count": len(period_results),
        "positive_periods": positive_periods,
        "negative_periods": negative_periods,
        "flat_periods": flat_periods,
        "positive_period_rate_percent": round(
            positive_periods / len(period_results) * 100.0, 4
        ) if period_results else 0.0,
        "total_evaluation_points": total_evaluation_points,
        "total_correct_signals": total_correct,
        "directional_accuracy_percent": round(overall_accuracy, 4),
        "average_period_return_percent": round(mean(returns), 4) if returns else 0.0,
        "median_period_return_percent": round(median(returns), 4) if returns else 0.0,
        "period_return_std_dev_percent": round(
            pstdev(returns), 4
        ) if len(returns) > 1 else 0.0,
        "average_period_accuracy_percent": round(
            mean(accuracies), 4
        ) if accuracies else 0.0,
        "average_max_drawdown": round(mean(drawdowns_percent), 4) if drawdowns_percent else 0.0,
        "worst_max_drawdown": round(max(drawdowns_percent), 4) if drawdowns_percent else 0.0,
        "average_max_drawdown_percent": round(mean(drawdowns_percent), 4) if drawdowns_percent else 0.0,
        "worst_max_drawdown_percent": round(max(drawdowns_percent), 4) if drawdowns_percent else 0.0,
        "average_max_drawdown_amount": round(mean(drawdowns_amount), 4) if drawdowns_amount else 0.0,
        "worst_max_drawdown_amount": round(max(drawdowns_amount), 4) if drawdowns_amount else 0.0,
        "total_simulated_trade_points": total_trades,
        "best_period": (
            {
                "period_id": best.get("period_id"),
                "validation_from_date": best.get("validation_from_date"),
                "validation_to_date": best.get("validation_to_date"),
                "return_percent": best["metrics"].get("simulated_return_percent"),
            }
            if best else None
        ),
        "worst_period": (
            {
                "period_id": worst.get("period_id"),
                "validation_from_date": worst.get("validation_from_date"),
                "validation_to_date": worst.get("validation_to_date"),
                "return_percent": worst["metrics"].get("simulated_return_percent"),
            }
            if worst else None
        ),
    }

    return {
        "status": "success",
        "layer": "Phase 10 / Layer 3",
        "part": "Part 3",
        "analysis": "Multi-Period Validation Metrics",
        "instrument_key": evaluation.get("instrument_key"),
        "configuration": {
            **(evaluation.get("configuration") or {}),
            "initial_capital": initial_capital,
        },
        "overall_metrics": overall,
        "periods": period_results,
    }


__all__ = ["aggregate_multi_period_metrics"]
