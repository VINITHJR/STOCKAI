from __future__ import annotations

import math
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.app.ai.layer2 import _build_ensemble
from backend.app.services.trend_service import get_trend_prediction
from backend.app.services.prediction_service import get_prediction
from backend.app.services.fundamentals_service import get_fundamentals
from backend.app.services.news_event_service import get_news_events


router = APIRouter(
    prefix="/api/ai/context",
    tags=["Phase 9 Layer 3 - Context-Aware AI Intelligence"],
)


# These are explicit, explainable heuristics, not learned or calibrated models.
# Thresholds are intended as broad context rules, not universal investment rules.
_FUNDAMENTAL_THRESHOLDS = {
    "roe_positive": 15.0,
    "roe_negative": 5.0,
    "roa_positive": 5.0,
    "roa_negative": 2.0,
    "roce_positive": 15.0,
    "roce_negative": 8.0,
    "profit_margin_positive": 10.0,
    "debt_to_equity_positive_max": 0.5,
    "debt_to_equity_negative_min": 2.0,
    "current_ratio_positive": 1.5,
    "current_ratio_negative": 1.0,
}

# Related names are limited to aliases that are reasonably specific to the issuer.
# The issuer's company name is also used dynamically when available.
_COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "RELIANCE": ("jio", "jio platforms", "reliance industries"),
}

_GENERIC_COMPANY_WORDS = {
    "ltd", "limited", "industries", "industry", "company", "co", "corp",
    "corporation", "holdings", "holding", "group", "india", "indian",
    "private", "public", "pvt", "inc", "incorporated", "plc", "services",
    "service", "technologies", "technology", "enterprises", "enterprise",
    "international", "global",
}

_POSITIVE_NEWS_TERMS = (
    "beats estimates", "beat estimates", "profit rises", "profit jumps",
    "profit grows", "record profit", "strong results", "strong earnings",
    "surge", "surges", "surged", "rally", "rallies", "rallied", "gains",
    "gained", "rises", "rising", "rose", "growth", "grows", "upgrade",
    "upgraded", "approved", "approval", "clearance", "expansion", "expands",
    "launches", "launch", "wins contract", "deal wins", "outperform", "welcomes",
    "dividend raised", "dividend hike", "orders secured",
)

_NEGATIVE_NEWS_TERMS = (
    "misses estimates", "missed estimates", "profit falls", "profit drops",
    "profit declines", "weak results", "weak earnings", "falls", "fell", "drops",
    "dropped", "decline", "declines", "declined", "slides", "slid", "slump",
    "loss", "losses", "downgrade", "downgraded", "probe", "investigation",
    "penalty", "fraud", "scam", "lawsuit", "default", "bankruptcy", "insolvency",
    "regulatory action", "ban imposed", "warning issued", "forecast cut",
    "guidance cut", "earnings cut", "plant shuts", "shutdown", "recall",
    "layoffs", "layoff", "criminal charges", "accounting irregularities",
)

_HIGH_RISK_TERMS = (
    "fraud", "scam", "probe", "investigation", "raid", "penalty", "regulatory action",
    "ban imposed", "default", "bankruptcy", "insolvency", "lawsuit", "criminal charges",
    "accounting irregularities", "major breach", "money laundering", "arrested",
)

_MEDIUM_RISK_TERMS = (
    "downgrade", "downgraded", "warning", "weak results", "weak earnings", "loss",
    "losses", "decline", "declines", "slump", "debt", "delay", "delayed", "competition",
    "forecast cut", "guidance cut", "earnings cut", "layoffs", "layoff", "risk", "risks",
    "volatility", "margin pressure", "misses estimates", "missed estimates", "falls", "drops",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_float(value: Any) -> float | None:
    """Parse a finite numeric value, accepting strings such as '8.94%' or '1,234'."""
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip().replace(",", "").replace("%", "")
        if not text:
            return None
        number = float(text)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _direction(value: str | None) -> str:
    value = (value or "neutral").strip().lower()
    if value in {"bullish", "positive", "up"}:
        return "bullish"
    if value in {"bearish", "negative", "down"}:
        return "bearish"
    return "neutral"


def _symbol_from_instrument(instrument_key: str) -> str | None:
    """Resolve an instrument key to its NSE trading symbol when possible."""
    try:
        from backend.app.services.upstox_service import get_nse_equity_universe

        for item in get_nse_equity_universe():
            if item.get("instrument_key") == instrument_key:
                symbol = str(item.get("symbol") or "").strip().upper()
                return symbol or None
    except Exception:
        # The endpoint can still work when the caller supplies symbol explicitly.
        return None
    return None


def _fundamental_context(result: dict | None) -> dict:
    """Score available fundamentals from the actual nested Upstox response shape."""
    if not isinstance(result, dict) or result.get("status") != "success":
        return {
            "status": "unavailable",
            "signal": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "positive_factors": 0,
            "negative_factors": 0,
            "factor_count": 0,
            "factors": [],
            "message": "Fundamentals data is unavailable.",
        }

    nested = result.get("fundamentals")
    nested = nested if isinstance(nested, dict) else {}

    def metric(name: str) -> float | None:
        # The current fundamentals service puts ratios under result['fundamentals'];
        # the outer-level lookup supports older/alternate response versions.
        value = _as_float(nested.get(name))
        if value is None:
            value = _as_float(result.get(name))
        return value

    roe = metric("roe")
    roa = metric("roa")
    roce = metric("roce")
    profit_margin = metric("profit_margin")
    revenue_growth = metric("revenue_growth")
    earnings_growth = metric("earnings_growth")
    debt_to_equity = metric("debt_to_equity")
    current_ratio = metric("current_ratio")

    # Some fundamentals payloads include current assets/liabilities but no current ratio.
    if current_ratio is None:
        current_assets = metric("current_assets")
        current_liabilities = metric("current_liabilities")
        if (
            current_assets is not None
            and current_liabilities is not None
            and current_assets >= 0
            and current_liabilities > 0
        ):
            current_ratio = current_assets / current_liabilities

    factors: list[dict[str, Any]] = []
    positive = 0
    negative = 0

    def add(
        name: str,
        value: float | None,
        signal: str,
        reason: str,
        unit: str | None = None,
    ) -> None:
        nonlocal positive, negative
        if value is None:
            return
        factor: dict[str, Any] = {
            "factor": name,
            "value": round(value, 4),
            "signal": signal,
            "reason": reason,
        }
        if unit:
            factor["unit"] = unit
        factors.append(factor)
        if signal == "positive":
            positive += 1
        elif signal == "negative":
            negative += 1

    t = _FUNDAMENTAL_THRESHOLDS

    if roe is not None:
        signal = "positive" if roe >= t["roe_positive"] else "negative" if roe < t["roe_negative"] else "neutral"
        reason = (
            f"ROE is at or above {t['roe_positive']}%." if signal == "positive" else
            f"ROE is below {t['roe_negative']}%." if signal == "negative" else
            "ROE is between the configured rule thresholds."
        )
        add("roe", roe, signal, reason, "%")

    if roa is not None:
        signal = "positive" if roa >= t["roa_positive"] else "negative" if roa < t["roa_negative"] else "neutral"
        reason = (
            f"ROA is at or above {t['roa_positive']}%." if signal == "positive" else
            f"ROA is below {t['roa_negative']}%." if signal == "negative" else
            "ROA is between the configured rule thresholds."
        )
        add("roa", roa, signal, reason, "%")

    if roce is not None:
        signal = "positive" if roce >= t["roce_positive"] else "negative" if roce < t["roce_negative"] else "neutral"
        reason = (
            f"ROCE is at or above {t['roce_positive']}%." if signal == "positive" else
            f"ROCE is below {t['roce_negative']}%." if signal == "negative" else
            "ROCE is between the configured rule thresholds."
        )
        add("roce", roce, signal, reason, "%")

    if profit_margin is not None:
        signal = "positive" if profit_margin >= t["profit_margin_positive"] else "negative" if profit_margin < 0 else "neutral"
        reason = (
            f"Profit margin is at or above {t['profit_margin_positive']}%." if signal == "positive" else
            "Profit margin is negative." if signal == "negative" else
            "Profit margin is non-negative but below the configured threshold."
        )
        add("profit_margin", profit_margin, signal, reason, "%")

    for name, value in (("revenue_growth", revenue_growth), ("earnings_growth", earnings_growth)):
        if value is None:
            continue
        signal = "positive" if value > 0 else "negative" if value < 0 else "neutral"
        reason = "Growth is positive." if signal == "positive" else "Growth is negative." if signal == "negative" else "Growth is flat."
        add(name, value, signal, reason, "%")

    if debt_to_equity is not None:
        # This field is a ratio (e.g. 1.0059), not a percentage (100.59).
        signal = (
            "negative" if debt_to_equity < 0 or debt_to_equity > t["debt_to_equity_negative_min"] else
            "positive" if debt_to_equity <= t["debt_to_equity_positive_max"] else
            "neutral"
        )
        reason = (
            f"Debt-to-equity is above {t['debt_to_equity_negative_min']} or negative." if signal == "negative" else
            f"Debt-to-equity is at or below {t['debt_to_equity_positive_max']}." if signal == "positive" else
            "Debt-to-equity is between the configured rule thresholds."
        )
        add("debt_to_equity", debt_to_equity, signal, reason, "ratio")

    if current_ratio is not None:
        signal = "positive" if current_ratio >= t["current_ratio_positive"] else "negative" if current_ratio < t["current_ratio_negative"] else "neutral"
        reason = (
            f"Current ratio is at or above {t['current_ratio_positive']}." if signal == "positive" else
            f"Current ratio is below {t['current_ratio_negative']}." if signal == "negative" else
            "Current ratio is between the configured rule thresholds."
        )
        add("current_ratio", current_ratio, signal, reason, "ratio")

    if not factors:
        return {
            "status": "insufficient_data",
            "signal": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "positive_factors": 0,
            "negative_factors": 0,
            "factor_count": 0,
            "factors": [],
            "message": "Fundamentals were retrieved, but no supported scoring fields were available.",
        }

    # Every evaluated metric contributes +1, 0 or -1. Neutral metrics are
    # counted in the denominator rather than silently dropped.
    signed_score = positive - negative
    score = signed_score / len(factors)
    signal = "bullish" if score >= 0.15 else "bearish" if score <= -0.15 else "neutral"

    return {
        "status": "success",
        "signal": signal,
        "score": round(score, 4),
        "confidence": round(_clamp(abs(score)), 4),
        "positive_factors": positive,
        "negative_factors": negative,
        "factor_count": len(factors),
        "factors": factors,
        "scoring_method": "transparent_rule_based_fundamentals_v2",
    }


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(_normalise_text(term)).replace(r"\ ", r"\s+")
    return bool(escaped and re.search(rf"\b{escaped}\b", text))


def _news_is_relevant(
    title: str,
    symbol: str | None,
    company_name: str | None = None,
) -> bool:
    """Only score headlines that mention the requested ticker/company or known alias."""
    text = _normalise_text(title)
    if not text:
        return False

    clean_symbol = _normalise_text(symbol)
    if clean_symbol and _contains_term(text, clean_symbol):
        return True

    if clean_symbol:
        compact_symbol = re.sub(r"[^a-z0-9]", "", clean_symbol)
        aliases = _COMPANY_ALIASES.get(compact_symbol.upper(), ())
        if any(_contains_term(text, alias) for alias in aliases):
            return True

    name_text = _normalise_text(company_name)
    if name_text:
        tokens = [
            token for token in name_text.split()
            if len(token) >= 4 and token not in _GENERIC_COMPANY_WORDS
        ]
        if any(_contains_term(text, token) for token in tokens):
            return True

    return False


def _classify_news_item(title: str) -> tuple[str, str]:
    """Return (sentiment, risk). This keyword heuristic is intentionally transparent."""
    text = _normalise_text(title)

    positive_hits = sum(_contains_term(text, term) for term in _POSITIVE_NEWS_TERMS)
    negative_hits = sum(_contains_term(text, term) for term in _NEGATIVE_NEWS_TERMS)

    if positive_hits > negative_hits:
        sentiment = "positive"
    elif negative_hits > positive_hits:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    if any(_contains_term(text, term) for term in _HIGH_RISK_TERMS):
        risk = "high"
    elif any(_contains_term(text, term) for term in _MEDIUM_RISK_TERMS):
        risk = "medium"
    else:
        risk = "low"

    return sentiment, risk


def _news_context(
    result: dict | None,
    symbol: str | None = None,
    company_name: str | None = None,
) -> dict:
    """Filter issuer-relevant headlines, then score sentiment and risk separately."""
    if not isinstance(result, dict) or result.get("status") != "success":
        return {
            "status": "unavailable",
            "signal": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "total_items": 0,
            "retrieved_items": 0,
            "excluded_irrelevant": 0,
            "risk_counts": {"high": 0, "medium": 0, "low": 0},
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "message": "News data is unavailable.",
        }

    if result.get("source") == "unavailable":
        return {
            "status": "unavailable",
            "signal": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "total_items": 0,
            "retrieved_items": 0,
            "excluded_irrelevant": 0,
            "risk_counts": {"high": 0, "medium": 0, "low": 0},
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "message": result.get("message", "News feed unavailable."),
        }

    if symbol is None:
        symbol = str(result.get("symbol") or "").strip().upper() or None

    raw_items = result.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    valid_items = [item for item in raw_items if isinstance(item, dict)]
    relevant_items = [
        item for item in valid_items
        if _news_is_relevant(
            str(item.get("title") or ""),
            symbol,
            company_name,
        )
    ]

    risk_counts = {"high": 0, "medium": 0, "low": 0}
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for item in relevant_items:
        sentiment, risk = _classify_news_item(str(item.get("title") or ""))
        sentiment_counts[sentiment] += 1
        risk_counts[risk] += 1

    total = len(relevant_items)
    if total:
        score = (sentiment_counts["positive"] - sentiment_counts["negative"]) / total
    else:
        score = 0.0

    # Require a minimum number of relevant headlines before emitting a directional
    # signal. This is a keyword heuristic, not a calibrated sentiment probability.
    minimum_for_direction = 3
    if total >= minimum_for_direction and score >= 0.15:
        signal = "bullish"
    elif total >= minimum_for_direction and score <= -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    sample_strength = min(1.0, total / 5.0)
    confidence = _clamp(abs(score) * sample_strength)

    message = None
    if total == 0:
        message = "No headlines matched the requested company/ticker; news signal is not inferred."
    elif total < minimum_for_direction:
        message = f"Only {total} relevant headline(s) were found; insufficient for a directional news signal."

    return {
        "status": "success" if total else "insufficient_data",
        "signal": signal,
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "total_items": total,
        "retrieved_items": len(valid_items),
        "excluded_irrelevant": max(0, len(valid_items) - total),
        "risk_counts": risk_counts,
        "sentiment_counts": sentiment_counts,
        "classification_method": "title_keyword_heuristic_v1",
        **({"message": message} if message else {}),
    }


def _build_context_signal(
    ensemble: dict,
    fundamentals: dict,
    news: dict,
) -> dict:
    """Combine Layer 2 with valid fundamentals/news context conservatively."""
    ai_signal = _direction(ensemble.get("signal"))
    parsed_ai_confidence = _as_float(ensemble.get("confidence"))
    ai_confidence = _clamp(parsed_ai_confidence if parsed_ai_confidence is not None else 0.0)

    context_parts = [fundamentals, news]
    supporting = 0
    opposing = 0
    usable = 0
    support_strength = 0.0
    oppose_strength = 0.0

    for context in context_parts:
        if not isinstance(context, dict) or context.get("status") != "success":
            continue
        context_signal = _direction(context.get("signal"))
        context_confidence_value = _as_float(context.get("confidence"))
        context_confidence = _clamp(context_confidence_value or 0.0)
        if context_signal == "neutral" or context_confidence <= 0:
            continue

        usable += 1
        if context_signal == ai_signal:
            supporting += 1
            support_strength += context_confidence
        elif ai_signal != "neutral":
            opposing += 1
            oppose_strength += context_confidence

    # Context evidence changes the ensemble confidence proportionally to its
    # own heuristic strength; it cannot dominate the trained-model ensemble.
    adjusted_confidence = ai_confidence
    adjusted_confidence += min(0.12, 0.08 * support_strength)
    adjusted_confidence -= min(0.18, 0.12 * oppose_strength)
    adjusted_confidence = _clamp(adjusted_confidence)

    final_signal = ai_signal
    if ai_signal == "neutral" and usable == 2:
        f_signal = _direction(fundamentals.get("signal"))
        n_signal = _direction(news.get("signal"))
        if f_signal == n_signal and f_signal != "neutral":
            final_signal = f_signal
            f_conf = _as_float(fundamentals.get("confidence")) or 0.0
            n_conf = _as_float(news.get("confidence")) or 0.0
            # When the ensemble is neutral, require both context sources to
            # agree and base the capped confidence on the weaker context source.
            agreement_confidence = min(_clamp(f_conf), _clamp(n_conf))
            adjusted_confidence = min(0.35, max(ai_confidence, agreement_confidence))

    return {
        "signal": final_signal,
        "confidence": round(adjusted_confidence, 4),
        "base_layer2_signal": ai_signal,
        "base_layer2_confidence": round(ai_confidence, 4),
        "supporting_contexts": supporting,
        "opposing_contexts": opposing,
        "usable_contexts": usable,
    }


@router.get("/run")
def run_context_aware_ai(
    instrument_key: str = Query(..., min_length=1),
    symbol: str | None = Query(
        default=None,
        min_length=1,
        description="Optional NSE trading symbol. If omitted, it is resolved from the instrument key.",
    ),
    news_days: int = Query(default=7, ge=1, le=30),
    news_limit: int = Query(default=20, ge=1, le=50),
):
    """Phase 9 Layer 3 — Context-Aware AI Intelligence."""
    key = instrument_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    resolved_symbol = (symbol or "").strip().upper() or _symbol_from_instrument(key)
    if not resolved_symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                "NSE symbol could not be resolved from instrument_key. "
                "Provide the symbol query parameter."
            ),
        )

    rf_result = None
    lstm_result = None
    errors: list[dict[str, str]] = []

    try:
        rf_result = get_trend_prediction(key)
    except Exception as exc:
        errors.append({"model": "RandomForest", "message": str(exc)})

    try:
        lstm_result = get_prediction(key)
    except Exception as exc:
        errors.append({"model": "LSTM", "message": str(exc)})

    try:
        ensemble = _build_ensemble(rf_result, lstm_result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    fundamentals_result = None
    try:
        fundamentals_result = get_fundamentals(resolved_symbol)
    except Exception as exc:
        errors.append({"source": "Fundamentals", "message": str(exc)})

    news_result = None
    try:
        news_result = get_news_events(
            resolved_symbol,
            days=news_days,
            limit=news_limit,
        )
    except Exception as exc:
        errors.append({"source": "News", "message": str(exc)})

    company_name = None
    if isinstance(fundamentals_result, dict):
        company_name = str(fundamentals_result.get("name") or "").strip() or None

    fundamental_context = _fundamental_context(fundamentals_result)
    news_context = _news_context(
        news_result,
        symbol=resolved_symbol,
        company_name=company_name,
    )

    final = _build_context_signal(
        ensemble,
        fundamental_context,
        news_context,
    )

    explanations = [
        f'Layer 2 ensemble signal is {ensemble["signal"]} with {ensemble["confidence"]:.1%} ensemble confidence.',
        (
            f'Fundamental context is {fundamental_context["signal"]} '
            f'from {fundamental_context.get("factor_count", 0)} evaluated factor(s).'
        ),
        (
            f'News context is {news_context["signal"]} based on '
            f'{news_context.get("total_items", 0)} relevant headline(s).'
        ),
    ]

    if fundamental_context.get("message"):
        explanations.append(str(fundamental_context["message"]))
    if news_context.get("message"):
        explanations.append(str(news_context["message"]))
    if final["supporting_contexts"]:
        explanations.append(
            f'{final["supporting_contexts"]} context source(s) support the Layer 2 signal.'
        )
    if final["opposing_contexts"]:
        explanations.append(
            f'{final["opposing_contexts"]} context source(s) differ from the Layer 2 signal; confidence is reduced.'
        )

    return {
        "status": "success",
        "layer": "Phase 9 / Layer 3",
        "analysis": "Context-Aware AI Intelligence",
        "instrument_key": key,
        "symbol": resolved_symbol,
        "signal": final["signal"],
        "confidence": final["confidence"],
        "base_layer2": {
            "signal": final["base_layer2_signal"],
            "confidence": final["base_layer2_confidence"],
            "model_count": ensemble["model_count"],
            "agreement": ensemble["agreement"],
            "conflict": ensemble["conflict"],
            "conflict_level": ensemble["conflict_level"],
        },
        "context": {
            "fundamentals": fundamental_context,
            "news": news_context,
        },
        "explanation": explanations,
        "source_outputs": {
            "random_forest": rf_result,
            "lstm": lstm_result,
            "fundamentals": fundamentals_result,
            "news": news_result,
        },
        "model_errors": errors,
    }
