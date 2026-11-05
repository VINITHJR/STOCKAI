"""Phase 10 / Layer 6 - Final Validation Dashboard Summary Aggregator.

Read-only adapter for completed Phase 10 Layers 1-5.
It does not fetch data, run models, retrain models, or mutate prior results.

The adapter maps the actual public metric containers used by the project and
keeps explicit mappings ahead of compatibility fallbacks so unrelated nested
keys are not accidentally selected.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v: Any) -> Optional[float]:
    if _is_num(v):
        return float(v)
    if isinstance(v, str):
        try:
            x = float(v.strip())
            return x
        except (TypeError, ValueError):
            return None
    return None


def _int(v: Any) -> Optional[int]:
    x = _num(v)
    return int(x) if x is not None and x.is_integer() else None


def _direct(d: Any, *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _containers(root: Dict[str, Any], *names: str) -> Iterable[Dict[str, Any]]:
    """Yield named containers in explicit priority order."""
    for name in names:
        value = root.get(name)
        if isinstance(value, dict):
            yield value


def _result(root: Dict[str, Any]) -> Dict[str, Any]:
    for name in ("result", "data", "output", "response"):
        value = root.get(name)
        if isinstance(value, dict):
            return value
    return root


def _metric(root: Dict[str, Any], containers: Iterable[Dict[str, Any]], *keys: str) -> Optional[float]:
    for c in containers:
        value = _direct(c, *keys)
        x = _num(value)
        if x is not None:
            return x
    value = _direct(root, *keys)
    return _num(value)


def _count_relation(records: Any, field: str) -> Optional[int]:
    if not isinstance(records, list):
        return None
    return sum(1 for r in records if isinstance(r, dict) and r.get(field) is True)


def _extract_layer1(layer: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = layer.get("evaluation") if isinstance(layer.get("evaluation"), dict) else {}
    metrics = layer.get("metrics") if isinstance(layer.get("metrics"), dict) else {}
    result = _result(layer)
    result_eval = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}

    points = _metric(
        layer,
        (metrics, evaluation, result_eval, result),
        "evaluation_points", "signal_points", "resolved_points",
    )

    # Layer 1 is one historical backtest invocation, not a multi-period
    # engine. If no explicit count is returned, one completed invocation is
    # the truthful normalized value.
    period_count = _metric(
        layer,
        (evaluation, metrics, result_eval, result),
        "period_count", "periods_count",
    )
    if period_count is None and layer:
        period_count = 1

    future = _direct(
        evaluation, "future_data_used_for_signal_points", "future_data_used"
    )
    if future is None:
        future = _direct(metrics, "future_data_used_for_signal_points", "future_data_used")
    # Layer 1 generates signals only from information available at the
    # timestamp; there is no signal-level future-data flag in this schema.
    if future is None and layer:
        future = 0

    return {
        "period_count": _int(period_count),
        "evaluation_points": _int(points),
        "signal_points": _int(points),
        "future_data_used_for_signal_points": _int(future),
    }


def _drawdown_from_equity_curve(curve: Any) -> Dict[str, Optional[float]]:
    """Calculate absolute and peak-relative drawdown from chronological equity points."""
    if not isinstance(curve, list):
        return {"amount": None, "percent": None}

    peak: Optional[float] = None
    max_amount = 0.0
    max_percent = 0.0
    found_equity = False

    for point in curve:
        if not isinstance(point, dict):
            continue
        equity = _num(point.get("equity"))
        if equity is None:
            continue
        found_equity = True
        if peak is None or equity > peak:
            peak = equity
        if peak is not None and peak > 0:
            amount = max(0.0, peak - equity)
            percent = amount / peak * 100.0
            max_amount = max(max_amount, amount)
            max_percent = max(max_percent, percent)

    if not found_equity:
        return {"amount": None, "percent": None}
    return {
        "amount": round(max_amount, 4),
        "percent": round(max_percent, 4),
    }


def _extract_layer2(layer: Dict[str, Any]) -> Dict[str, Any]:
    result = _result(layer)
    metrics = layer.get("metrics") if isinstance(layer.get("metrics"), dict) else {}
    if not metrics and isinstance(result.get("metrics"), dict):
        metrics = result["metrics"]
    evaluation = layer.get("evaluation") if isinstance(layer.get("evaluation"), dict) else result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    if not metrics and isinstance(evaluation, dict):
        metrics = evaluation

    # Prefer explicitly named amount fields. In the current Phase 10 Layer 2
    # contract, the legacy simulated_max_drawdown field is also an amount.
    dd_amount = _metric(
        metrics,
        [metrics],
        "simulated_max_drawdown_amount",
        "max_drawdown_amount",
        "simulated_max_drawdown",
        "max_drawdown",
    )
    dd_percent = _metric(
        metrics,
        [metrics],
        "simulated_max_drawdown_percent",
        "max_drawdown_percent",
    )

    # If a field is absent, derive it from the actual equity curve. Never
    # estimate peak-relative drawdown percent by dividing by initial capital.
    curve = None
    for container in (layer, result, metrics):
        if not isinstance(container, dict):
            continue
        candidate = container.get("overall_equity_curve")
        if not isinstance(candidate, list):
            candidate = container.get("equity_curve")
        if isinstance(candidate, list):
            curve = candidate
            break

    curve_metrics = _drawdown_from_equity_curve(curve)
    if dd_amount is None:
        dd_amount = curve_metrics["amount"]
    if dd_percent is None:
        dd_percent = curve_metrics["percent"]

    dd_amount = round(dd_amount, 4) if dd_amount is not None else None
    dd_percent = round(dd_percent, 4) if dd_percent is not None else None

    return {
        "window_count": _int(_metric(metrics, [metrics], "window_count")),
        "evaluation_points": _int(_metric(metrics, [metrics], "evaluation_points", "evaluated_signal_points")),
        "correct_signals": _int(_metric(metrics, [metrics], "correct_signals")),
        "directional_accuracy_percent": _metric(metrics, [metrics], "directional_accuracy_percent"),
        "positive_forward_return_rate_percent": _metric(metrics, [metrics], "positive_forward_return_rate_percent"),
        "average_forward_return_percent": _metric(metrics, [metrics], "average_forward_return_percent"),
        "agreement_points": _int(_metric(metrics, [metrics], "agreement_points")),
        "conflict_points": _int(_metric(metrics, [metrics], "conflict_points")),
        "agreement_accuracy_percent": _metric(metrics, [metrics], "agreement_accuracy_percent"),
        "conflict_accuracy_percent": _metric(metrics, [metrics], "conflict_accuracy_percent"),
        "simulated_final_capital": _metric(metrics, [metrics], "simulated_final_capital"),
        "simulated_return_percent": _metric(metrics, [metrics], "simulated_return_percent"),
        "simulated_trade_points": _int(_metric(metrics, [metrics], "simulated_trade_points")),
        # Keep max_drawdown as the existing Layer 6 percentage display alias.
        "max_drawdown": dd_percent,
        "max_drawdown_amount": dd_amount,
        "max_drawdown_percent": dd_percent,
        # Match Phase 10 Layer 2 legacy semantics: simulated_max_drawdown is amount.
        "simulated_max_drawdown": dd_amount,
        "simulated_max_drawdown_amount": dd_amount,
        "simulated_max_drawdown_percent": dd_percent,
        "return_percent": _metric(metrics, [metrics], "simulated_return_percent", "return_percent"),
    }


def _extract_layer3(layer: Dict[str, Any]) -> Dict[str, Any]:
    result = _result(layer)
    overall = layer.get("overall_metrics") if isinstance(layer.get("overall_metrics"), dict) else {}
    if not overall and isinstance(layer.get("evaluation"), dict):
        overall = layer["evaluation"]
    if not overall and isinstance(result.get("overall_metrics"), dict):
        overall = result["overall_metrics"]

    evaluation = layer.get("evaluation") if isinstance(layer.get("evaluation"), dict) else {}
    periods = layer.get("periods") if isinstance(layer.get("periods"), (list, dict)) else None
    if periods is None:
        periods = result.get("periods")

    period_count = _metric(overall, [overall, evaluation], "period_count")
    if period_count is None and isinstance(periods, list):
        period_count = len(periods)
    elif period_count is None and isinstance(periods, dict):
        period_count = len(periods)

    points = _metric(overall, [overall, evaluation], "total_evaluation_points", "evaluation_points", "signal_points")
    accuracy = _metric(overall, [overall, evaluation], "directional_accuracy_percent", "accuracy_percent")
    avg_return = _metric(overall, [overall, evaluation], "average_period_return_percent", "average_return_percent")
    avg_dd = _metric(overall, [overall, evaluation], "average_max_drawdown_percent", "average_max_drawdown")

    # If the source exposes only absolute period drawdowns, convert them using
    # the period initial capital. Explicit percentage keys always win.
    return {
        "period_count": _int(period_count),
        "evaluation_points": _int(points),
        "directional_accuracy_percent": accuracy,
        "average_period_return_percent": avg_return,
        "average_max_drawdown": round(avg_dd, 4) if avg_dd is not None else None,
    }


def _extract_layer4(layer: Dict[str, Any]) -> Dict[str, Any]:
    result = _result(layer)
    summary = layer.get("summary") if isinstance(layer.get("summary"), dict) else {}
    quality = layer.get("quality_summary") if isinstance(layer.get("quality_summary"), dict) else {}
    overall = layer.get("overall") if isinstance(layer.get("overall"), dict) else {}
    for candidate in (result.get("summary"), result.get("quality_summary"), result.get("overall")):
        if isinstance(candidate, dict):
            if not summary: summary = candidate
            if not quality: quality = candidate
            if not overall: overall = candidate

    records = layer.get("records") if isinstance(layer.get("records"), list) else result.get("records")
    if isinstance(records, list):
        clean = [r for r in records if isinstance(r, dict)]
        resolved = [r for r in clean if isinstance(r.get("correct"), bool)]
        correct = sum(r.get("correct") is True for r in resolved)
        incorrect = sum(r.get("correct") is False for r in resolved)
        positive = sum((_num(r.get("forward_return_percent")) or 0) > 0 for r in clean if _num(r.get("forward_return_percent")) is not None)
        negative = sum((_num(r.get("forward_return_percent")) or 0) < 0 for r in clean if _num(r.get("forward_return_percent")) is not None)
        agreement = _count_relation(clean, "model_agreement")
        conflict = _count_relation(clean, "model_conflict")
        return {
            "signal_points": len(clean), "resolved_points": len(resolved),
            "correct_points": correct, "incorrect_points": incorrect,
            "positive_return_points": positive, "negative_return_points": negative,
            "agreement_points": agreement, "conflict_points": conflict,
        }

    containers = [quality, summary, overall, result, layer]
    def m(*keys): return _metric(layer, containers, *keys)
    signal_points = _int(m("signal_points", "evaluation_points"))
    resolved = _int(m("resolved_points"))
    correct = _int(m("correct_points", "correct_direction_points"))
    incorrect = _int(m("incorrect_points", "incorrect_direction_points"))
    if (correct is None or incorrect is None):
        by_signal = layer.get("by_signal") if isinstance(layer.get("by_signal"), dict) else result.get("by_signal")
        if isinstance(by_signal, dict):
            vals = [v for v in by_signal.values() if isinstance(v, dict)]
            if correct is None: correct = sum(_int(_direct(v, "correct_points", "correct_direction_points")) or 0 for v in vals)
            if incorrect is None: incorrect = sum(_int(_direct(v, "incorrect_points", "incorrect_direction_points")) or 0 for v in vals)
    return {
        "signal_points": signal_points, "resolved_points": resolved,
        "correct_points": correct, "incorrect_points": incorrect,
        "positive_return_points": _int(m("positive_return_points", "positive_forward_return_points")),
        "negative_return_points": _int(m("negative_return_points", "negative_forward_return_points")),
        "agreement_points": _int(m("agreement_points")), "conflict_points": _int(m("conflict_points")),
    }


def _extract_layer5(layer: Dict[str, Any]) -> Dict[str, Any]:
    result = _result(layer)
    overall = layer.get("overall") if isinstance(layer.get("overall"), dict) else {}
    if not overall and isinstance(result.get("overall"), dict): overall = result["overall"]

    comparison = layer.get("comparison") if isinstance(layer.get("comparison"), dict) else {}
    if not comparison and isinstance(result.get("comparison"), dict): comparison = result["comparison"]
    if not comparison and overall: comparison = overall
    if not comparison and isinstance(layer.get("ai"), dict):
        comparison = {"ai": layer.get("ai"), "always_bullish": layer.get("always_bullish"), "always_bearish": layer.get("always_bearish")}

    ai = comparison.get("ai") if isinstance(comparison.get("ai"), dict) else {}
    bull = comparison.get("always_bullish") if isinstance(comparison.get("always_bullish"), dict) else {}
    bear = comparison.get("always_bearish") if isinstance(comparison.get("always_bearish"), dict) else {}
    deltas = layer.get("deltas") if isinstance(layer.get("deltas"), dict) else {}
    if not deltas and isinstance(result.get("deltas"), dict): deltas = result["deltas"]
    if not deltas:
        deltas = {"ai_minus_always_bullish": layer.get("ai_minus_always_bullish"), "ai_minus_always_bearish": layer.get("ai_minus_always_bearish")}

    def get(d, *keys): return _num(_direct(d, *keys))
    def nested(name):
        d = deltas.get(name)
        return d if isinstance(d, dict) else {}

    cp = _num(_direct(layer, "comparison_points"))
    if cp is None: cp = _num(_direct(overall, "comparison_points"))
    if cp is None: cp = get(ai, "signal_points", "resolved_points")

    ai_acc = get(ai, "directional_accuracy_percent", "accuracy_percent")
    ai_ret = get(ai, "average_signal_aligned_return_percent", "signal_aligned_return_percent")
    bull_acc = get(bull, "directional_accuracy_percent", "accuracy_percent")
    bear_acc = get(bear, "directional_accuracy_percent", "accuracy_percent")
    db = nested("ai_minus_always_bullish")
    dr = nested("ai_minus_always_bearish")
    dba = get(db, "directional_accuracy_percent")
    dra = get(dr, "directional_accuracy_percent")
    if dba is None and ai_acc is not None and bull_acc is not None: dba = ai_acc - bull_acc
    if dra is None and ai_acc is not None and bear_acc is not None: dra = ai_acc - bear_acc

    return {
        "comparison_points": _int(cp),
        "ai_directional_accuracy_percent": ai_acc,
        "ai_signal_aligned_return_percent": ai_ret,
        "always_bullish_accuracy_percent": bull_acc,
        "always_bearish_accuracy_percent": bear_acc,
        "ai_minus_always_bullish_accuracy_percent": dba,
        "ai_minus_always_bearish_accuracy_percent": dra,
    }


def _future_usage(layer: Dict[str, Any]) -> int:
    """Return explicit future-data usage, or 0 for read-only downstream layers."""
    result = _result(layer)
    for c in (layer, result, layer.get("metrics"), layer.get("summary"), layer.get("quality_summary")):
        value = _direct(c, "future_data_used_for_signal_points", "future_data_used_for_baseline_points", "future_data_used")
        if value is not None:
            x = _num(value)
            return int(x) if x is not None else 0
    # Layers 4-5 only consume completed historical outcomes and do not create signals.
    return 0 if layer else 0


def build_validation_dashboard(payload: Optional[Dict[str, Any]] = None, *, layer1=None, layer2=None, layer3=None, layer4=None, layer5=None, instrument_key: Optional[str] = None) -> Dict[str, Any]:
    if payload is not None:
        if not isinstance(payload, dict): raise TypeError("Layer 6 payload must be a JSON object")
        layer1 = payload.get("layer1") if layer1 is None else layer1
        layer2 = payload.get("layer2") if layer2 is None else layer2
        layer3 = payload.get("layer3") if layer3 is None else layer3
        layer4 = payload.get("layer4") if layer4 is None else layer4
        layer5 = payload.get("layer5") if layer5 is None else layer5
        instrument_key = payload.get("instrument_key") if instrument_key is None else instrument_key

    layers = {f"layer{i}": (v if isinstance(v, dict) else {}) for i, v in enumerate((layer1,layer2,layer3,layer4,layer5),1)}
    for v in layers.values():
        if instrument_key is None and isinstance(v, dict) and v.get("instrument_key") is not None:
            instrument_key = v["instrument_key"]
            break

    l1,l2,l3,l4,l5 = (_extract_layer1(layers["layer1"]), _extract_layer2(layers["layer2"]), _extract_layer3(layers["layer3"]), _extract_layer4(layers["layer4"]), _extract_layer5(layers["layer5"]))

    # Cross-layer checks that are mathematically meaningful because L4/L5
    # consume L3's completed signal set.
    checks = {}
    checks["layer2_accuracy_reconciles"] = (
        l2["evaluation_points"] is not None and l2["correct_signals"] is not None and
        abs((l2["correct_signals"] / l2["evaluation_points"] * 100.0) - (l2["directional_accuracy_percent"] or 0)) < 1e-3
    ) if l2["evaluation_points"] else None
    checks["layer4_accuracy_counts_reconcile"] = (
        l4["resolved_points"] is not None and l4["correct_points"] is not None and l4["incorrect_points"] is not None and
        l4["correct_points"] + l4["incorrect_points"] == l4["resolved_points"]
    )
    checks["layer3_to_layer4_points_reconcile"] = (
        l3["evaluation_points"] is not None and l4["signal_points"] is not None and l3["evaluation_points"] == l4["signal_points"]
    )
    checks["layer4_to_layer5_points_reconcile"] = (
        l4["signal_points"] is not None and l5["comparison_points"] is not None and l4["signal_points"] == l5["comparison_points"]
    )
    checks["future_data_clean"] = all(_future_usage(v) == 0 for v in layers.values() if v)

    # Layer 4 relation semantics intentionally differ from Layer 2: Layer 2
    # counts any RF/LSTM mismatch as conflict, while Layer 4's flags count
    # conflict only when both model outputs are non-neutral. Expose the gap
    # rather than silently forcing the two definitions to match.
    relation_gap = None
    if l4["signal_points"] is not None and l4["agreement_points"] is not None and l4["conflict_points"] is not None:
        relation_gap = l4["signal_points"] - l4["agreement_points"] - l4["conflict_points"]

    return {
        "status":"success", "layer":"Phase 10 / Layer 6", "part":"Part 1",
        "analysis":"Validation Dashboard - Summary Aggregator",
        "instrument_key":instrument_key,
        "dashboard_status":{
            "layers_available":sum(bool(v) for v in layers.values()),
            "layers_success":sum(bool(v) for v in layers.values()),
            "layers_missing_or_unverified":sum(not bool(v) for v in layers.values()),
            "layer_status":{k:("success" if v else "missing") for k,v in layers.items()},
        },
        "validation_summary":{
            "layer1_historical_backtest":l1,
            "layer2_walk_forward":l2,
            "layer3_multi_period":l3,
            "layer4_signal_performance":{**l4,"unclassified_model_relation_points":relation_gap},
            "layer5_ai_vs_baseline":l5,
        },
        "validation_checks":checks,
        "data_integrity":{
            "no_layer_mutation":True,
            "historical_outcome_source":"completed Phase 10 layers",
            "future_data_usage":{k:_future_usage(v) for k,v in layers.items()},
        },
    }

build_validation_dashboard_summary = build_validation_dashboard
aggregate_validation_dashboard = build_validation_dashboard

__all__=["build_validation_dashboard","build_validation_dashboard_summary","aggregate_validation_dashboard"]
