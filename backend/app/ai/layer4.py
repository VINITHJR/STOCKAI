from __future__ import annotations

from datetime import date, datetime, timedelta
from math import isfinite, sqrt
from statistics import mean, stdev
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.upstox_service import (
    get_historical_candles,
    get_intraday_candles,
)


router = APIRouter(
    prefix="/api/ai/regime",
    tags=["Phase 9 Layer 4 - Market Regime Intelligence"],
)

INDIA_TZ = ZoneInfo("Asia/Kolkata")
MIN_DAILY_CANDLES = 50
INTRADAY_UNIT = "minutes"
INTRADAY_INTERVAL = "30"


class HistoricalMarketDataError(RuntimeError):
    """Historical candle retrieval failed."""


class InsufficientDailyCandlesError(ValueError):
    """There are not enough completed daily candles to compute a regime."""


def _num(value) -> float | None:
    """Convert a value to a finite float; reject NaN and infinity."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _parse_timestamp(value) -> datetime | None:
    """Parse an ISO timestamp and normalize it to India local time."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError, OverflowError):
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=INDIA_TZ)
    return parsed.astimezone(INDIA_TZ)


def _extract_candles(payload: dict | None) -> list[dict]:
    """Normalize Upstox candles into chronologically ordered OHLCV rows."""
    data = payload.get("data") if isinstance(payload, dict) else None
    candles = data.get("candles") if isinstance(data, dict) else None
    if candles is None and isinstance(payload, dict):
        candles = payload.get("candles")
    if not isinstance(candles, list):
        return []

    rows: list[dict] = []
    for candle in candles:
        if isinstance(candle, (list, tuple)) and len(candle) >= 6:
            timestamp = candle[0]
            open_price = _num(candle[1])
            high = _num(candle[2])
            low = _num(candle[3])
            close = _num(candle[4])
            volume = _num(candle[5])
        elif isinstance(candle, dict):
            timestamp = candle.get("timestamp") or candle.get("time")
            open_price = _num(candle.get("open"))
            high = _num(candle.get("high"))
            low = _num(candle.get("low"))
            close = _num(candle.get("close"))
            volume = _num(candle.get("volume"))
        else:
            continue

        # Prices must be finite and positive; OHLC ranges must be sensible.
        if (
            close is None or close <= 0
            or high is None or high <= 0
            or low is None or low <= 0
            or high < low
        ):
            continue

        rows.append({
            "timestamp": str(timestamp) if timestamp is not None else None,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(0.0, volume) if volume is not None else 0.0,
        })

    # Known timestamps first, chronologically. Rows without a parseable
    # timestamp are kept at the beginning so they cannot become the latest bar.
    def sort_key(row: dict):
        parsed = _parse_timestamp(row.get("timestamp"))
        if parsed is None:
            return (0, datetime.min.replace(tzinfo=INDIA_TZ), str(row.get("timestamp") or ""))
        return (1, parsed, str(row.get("timestamp") or ""))

    rows.sort(key=sort_key)
    return rows


def _sma(values: list[float], window: int) -> float | None:
    if window <= 0 or len(values) < window:
        return None
    return mean(values[-window:])


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    window: int = 14,
) -> float | None:
    if window <= 0 or len(closes) < window + 1:
        return None

    true_ranges = []
    for index in range(1, len(closes)):
        true_ranges.append(max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        ))

    return mean(true_ranges[-window:]) if len(true_ranges) >= window else None


def _latest_same_day_intraday_reference(
    instrument_key: str,
    session_date: date,
) -> dict | None:
    """Return the latest 30-minute close for the current India session.

    This is only a price reference. Intraday bars are never mixed into the
    daily series used for SMA, ATR, historical volatility, or volume metrics.
    Intraday retrieval failure is non-fatal: callers fall back to daily data.
    """
    try:
        payload = get_intraday_candles(
            instrument_key,
            unit=INTRADAY_UNIT,
            interval=INTRADAY_INTERVAL,
        )
    except Exception:
        return None

    rows = _extract_candles(payload)
    now = datetime.now(INDIA_TZ)
    candidates: list[tuple[datetime, dict]] = []

    for row in rows:
        timestamp = _parse_timestamp(row.get("timestamp"))
        close = _num(row.get("close"))
        if timestamp is None or close is None or close <= 0:
            continue
        if timestamp.date() != session_date:
            continue
        # Reject clearly future-dated feed records.
        if timestamp > now + timedelta(minutes=5):
            continue
        candidates.append((timestamp, row))

    if not candidates:
        return None

    timestamp, latest = max(candidates, key=lambda item: item[0])
    return {
        "price": float(latest["close"]),
        "timestamp": timestamp.isoformat(),
        "date": timestamp.date().isoformat(),
        "source": "intraday_30m_close",
    }


def _completed_daily_rows(rows: list[dict], before_date: date) -> list[dict]:
    """Exclude daily rows dated on/after a live intraday reference date."""
    completed = []
    for row in rows:
        timestamp = _parse_timestamp(row.get("timestamp"))
        # When a live reference is used, timestamp-less rows cannot safely be
        # proved to precede that session, so exclude them from the indicators.
        if timestamp is None:
            continue
        if timestamp.date() < before_date:
            completed.append(row)
    return completed


def _regime(rows: list[dict], current_reference: dict | None = None) -> dict:
    """Calculate regime indicators from daily candles and an optional live price.

    Daily SMAs, ATR, volatility and volume ratio use only completed daily rows.
    A same-session 30-minute close may be used for current price, momentum,
    current-vs-SMA comparisons, and ATR percentage. It is not inserted into the
    daily history, preventing mixed-interval indicators.
    """
    if not isinstance(rows, list) or len(rows) < MIN_DAILY_CANDLES:
        raise InsufficientDailyCandlesError(
            f"At least {MIN_DAILY_CANDLES} usable completed daily candles are required; "
            f"received {len(rows) if isinstance(rows, list) else 0}."
        )

    closes = [_num(row.get("close")) for row in rows]
    highs = [_num(row.get("high")) for row in rows]
    lows = [_num(row.get("low")) for row in rows]
    volumes = [_num(row.get("volume")) or 0.0 for row in rows]

    # Rows are validated by _extract_candles, but validate again for safe use
    # when this helper is called independently by other modules/tests.
    if any(value is None or value <= 0 for value in closes + highs + lows):
        raise ValueError("Daily candles contain invalid close/high/low values.")

    closes = [float(value) for value in closes]
    highs = [float(value) for value in highs]
    lows = [float(value) for value in lows]

    latest_daily_row = rows[-1]
    latest_daily_close = closes[-1]
    latest_daily_timestamp = latest_daily_row.get("timestamp")
    latest_daily_dt = _parse_timestamp(latest_daily_timestamp)

    reference_price = _num(current_reference.get("price")) if isinstance(current_reference, dict) else None
    reference_timestamp = current_reference.get("timestamp") if isinstance(current_reference, dict) else None
    reference_dt = _parse_timestamp(reference_timestamp)

    # Only trust the current reference when it is positive and clearly newer
    # than the last completed daily candle.
    use_reference = bool(
        reference_price is not None
        and reference_price > 0
        and reference_dt is not None
        and (latest_daily_dt is None or reference_dt.date() > latest_daily_dt.date())
    )

    if use_reference:
        current = float(reference_price)
        current_source = str(current_reference.get("source") or "intraday_30m_close")
        current_timestamp = reference_dt.isoformat()
    else:
        current = latest_daily_close
        current_source = "latest_completed_daily_close"
        current_timestamp = str(latest_daily_timestamp) if latest_daily_timestamp is not None else None

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    atr14 = _atr(highs, lows, closes, 14)

    returns = [
        (closes[index] / closes[index - 1]) - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    recent_returns = returns[-20:]
    volatility_daily = stdev(recent_returns) if len(recent_returns) >= 2 else 0.0
    volatility_annual = volatility_daily * sqrt(252)

    # When the reference is a newer session than the daily series, compare it
    # to the close 20 trading sessions earlier. For a daily-close fallback,
    # preserve the original 20-close-to-close window.
    if use_reference:
        lookback = min(20, len(closes) - 1)
        baseline_index = max(0, len(closes) - lookback)
    else:
        lookback = min(20, len(closes) - 1)
        baseline_index = len(closes) - 1 - lookback

    baseline_price = closes[baseline_index] if lookback > 0 else current
    momentum20 = (current / baseline_price) - 1.0 if baseline_price > 0 else 0.0
    slope = momentum20 / max(lookback, 1)

    volume_ratio = None
    if len(volumes) >= 20:
        average_volume = mean(volumes[-20:])
        if average_volume > 0:
            volume_ratio = volumes[-1] / average_volume

    atr_percent = (atr14 / current) if atr14 is not None and current > 0 else 0.0

    # Descriptive deterministic regime thresholds; not a forecast guarantee.
    high_vol = volatility_annual >= 0.35 or atr_percent >= 0.025
    low_vol = volatility_annual <= 0.15 and atr_percent <= 0.015
    bullish_trend = bool(
        sma20 is not None and sma50 is not None
        and current > sma20 > sma50 and momentum20 > 0.02
    )
    bearish_trend = bool(
        sma20 is not None and sma50 is not None
        and current < sma20 < sma50 and momentum20 < -0.02
    )

    if high_vol and bullish_trend:
        regime = "volatile_bullish"
    elif high_vol and bearish_trend:
        regime = "volatile_bearish"
    elif bullish_trend:
        regime = "bullish_trend"
    elif bearish_trend:
        regime = "bearish_trend"
    elif high_vol:
        regime = "high_volatility"
    elif low_vol:
        regime = "low_volatility_range"
    else:
        regime = "range_bound"

    if regime in {"bullish_trend", "volatile_bullish"}:
        direction = "bullish"
    elif regime in {"bearish_trend", "volatile_bearish"}:
        direction = "bearish"
    else:
        direction = "neutral"

    trend_strength = min(1.0, abs(momentum20) / 0.10)
    volatility_strength = min(1.0, volatility_annual / 0.50)
    confidence = max(trend_strength, volatility_strength * 0.75)
    if regime in {"range_bound", "low_volatility_range", "high_volatility"}:
        confidence *= 0.70

    return {
        "regime": regime,
        "direction": direction,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "metrics": {
            "current_price": round(current, 4),
            "current_price_source": current_source,
            "current_price_timestamp": current_timestamp,
            "last_completed_daily_close": round(latest_daily_close, 4),
            "last_completed_daily_timestamp": str(latest_daily_timestamp) if latest_daily_timestamp is not None else None,
            "daily_candle_count_for_indicators": len(rows),
            "intraday_reference_used": use_reference,
            "sma_20": round(sma20, 4) if sma20 is not None else None,
            "sma_50": round(sma50, 4) if sma50 is not None else None,
            "momentum_20d_percent": round(momentum20 * 100, 4),
            "trend_slope": round(slope, 6),
            "atr_14": round(atr14, 4) if atr14 is not None else None,
            "atr_percent": round(atr_percent * 100, 4),
            "volatility_annualized": round(volatility_annual, 4),
            "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
        },
    }


def load_regime_analysis(instrument_key: str, lookback_days: int = 180) -> dict:
    """Shared regime loader used by both Layer 4 and Layer 5.

    The historical series is limited to completed daily candles. If a valid
    current-session 30-minute candle exists, its close is used as the current
    price reference; otherwise the latest completed daily close is used.
    """
    key = (instrument_key or "").strip()
    if not key:
        raise ValueError("instrument_key is required")

    today = datetime.now(INDIA_TZ).date()
    from_date = today - timedelta(days=int(lookback_days))
    try:
        payload = get_historical_candles(
            key,
            unit="days",
            interval="1",
            to_date=today.isoformat(),
            from_date=from_date.isoformat(),
        )
    except Exception as exc:
        raise HistoricalMarketDataError(f"Historical market data unavailable: {exc}") from exc

    rows = _extract_candles(payload)
    reference = _latest_same_day_intraday_reference(key, today)

    if reference is not None:
        rows = _completed_daily_rows(rows, today)
    else:
        # Never treat a current-date daily candle as a fully completed session
        # when intraday is unavailable; use only dated candles before today.
        rows = _completed_daily_rows(rows, today)

    if len(rows) < MIN_DAILY_CANDLES:
        raise InsufficientDailyCandlesError(
            f"At least {MIN_DAILY_CANDLES} usable completed daily candles are required; "
            f"received {len(rows)}."
        )

    analysis = _regime(rows, current_reference=reference)
    return {
        "analysis": analysis,
        "candle_count": len(rows),
        "lookback_days": int(lookback_days),
    }


@router.get("/run")
def run_market_regime(
    instrument_key: str = Query(..., min_length=1),
    lookback_days: int = Query(default=180, ge=60, le=365),
):
    """Phase 9 Layer 4 — Market Regime Intelligence."""
    key = instrument_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="instrument_key is required")

    try:
        result = load_regime_analysis(key, lookback_days)
    except InsufficientDailyCandlesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HistoricalMarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    analysis = result["analysis"]
    metrics = analysis["metrics"]
    reference_description = (
        "the latest same-session 30-minute candle"
        if metrics["intraday_reference_used"]
        else "the latest completed daily candle (intraday reference unavailable)"
    )

    return {
        "status": "success",
        "layer": "Phase 9 / Layer 4",
        "analysis": "Market Regime Intelligence",
        "instrument_key": key,
        "lookback_days": result["lookback_days"],
        "candle_count": result["candle_count"],
        "regime": analysis["regime"],
        "direction": analysis["direction"],
        "confidence": analysis["confidence"],
        "metrics": metrics,
        "explanation": [
            f'Current market regime is classified as {analysis["regime"]}.',
            f'Regime direction is {analysis["direction"]} with {analysis["confidence"]:.1%} regime confidence.',
            f'Current price reference uses {reference_description}.',
            'SMA, ATR, historical volatility and volume ratio use completed daily candles only; the current-session intraday close is not mixed into the daily series.',
        ],
    }
