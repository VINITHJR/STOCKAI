from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from backend.app.services.upstox_service import get_historical_candles


def _parse_timestamp(value: Any) -> Tuple[datetime, str]:
    """Parse an ISO timestamp and return (UTC instant, original string)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")

    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {text!r}") from exc

    if parsed.tzinfo is None:
        # Treat timezone-naive candle timestamps consistently as UTC.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc), text


def _validated_candle(candle: Dict[str, Any]) -> Tuple[Dict[str, Any], datetime]:
    """Validate a normalized candle dictionary and return it with UTC time."""
    if not isinstance(candle, dict):
        raise ValueError("candle must be an object")

    timestamp_dt, timestamp_text = _parse_timestamp(candle.get("timestamp"))

    values: Dict[str, float] = {}
    for name in ("open", "high", "low", "close"):
        try:
            value = float(candle[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
        values[name] = value

    if values["high"] < max(values["open"], values["close"]):
        raise ValueError("high must be >= max(open, close)")
    if values["low"] > min(values["open"], values["close"]):
        raise ValueError("low must be <= min(open, close)")
    if values["low"] > values["high"]:
        raise ValueError("low must be <= high")

    try:
        volume_float = float(candle.get("volume", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("volume must be numeric") from exc
    if not math.isfinite(volume_float) or volume_float < 0:
        raise ValueError("volume must be finite and non-negative")
    if not volume_float.is_integer():
        raise ValueError("volume must be a whole number")

    try:
        open_interest = float(candle.get("open_interest", 0) or 0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("open_interest must be numeric") from exc
    if not math.isfinite(open_interest) or open_interest < 0:
        raise ValueError("open_interest must be finite and non-negative")

    normalized = {
        "timestamp": timestamp_text,
        "open": values["open"],
        "high": values["high"],
        "low": values["low"],
        "close": values["close"],
        "volume": int(volume_float),
        "open_interest": open_interest,
    }
    return normalized, timestamp_dt


def validate_normalized_candles(
    candles: List[Dict[str, Any]],
    *,
    reject_duplicates: bool = True,
) -> List[Dict[str, Any]]:
    """Validate normalized candle dictionaries and return them chronologically.

    Cache reads/writes use reject_duplicates=True: any invalid candle makes the
    cached dataset unusable, causing callers to fetch clean data again.
    """
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    parsed: List[Tuple[datetime, Dict[str, Any]]] = []
    seen = set()
    for index, candle in enumerate(candles):
        try:
            normalized, timestamp_dt = _validated_candle(candle)
        except ValueError as exc:
            raise ValueError(f"Invalid candle at index {index}: {exc}") from exc

        timestamp_key = timestamp_dt.isoformat()
        if reject_duplicates and timestamp_key in seen:
            raise ValueError(
                f"Duplicate candle timestamp at index {index}: "
                f"{normalized['timestamp']}"
            )
        seen.add(timestamp_key)
        parsed.append((timestamp_dt, normalized))

    parsed.sort(key=lambda item: item[0])
    return [item[1] for item in parsed]


def _normalize(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Upstox candles, discarding invalid records and deduplicating.

    When multiple valid rows represent the same instant, the last valid row in
    the provider response is retained. Output is sorted chronologically.
    """
    candles = (raw or {}).get("data", {}).get("candles", []) or []
    by_timestamp: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}

    for candle in candles:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue
        try:
            timestamp_dt, timestamp_text = _parse_timestamp(candle[0])
            raw_candle = {
                "timestamp": timestamp_text,
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5] if candle[5] is not None else 0,
                "open_interest": (
                    candle[6] if len(candle) > 6 and candle[6] is not None else 0
                ),
            }
            normalized, timestamp_dt = _validated_candle(raw_candle)
        except (ValueError, TypeError, KeyError, OverflowError):
            continue

        # Same instant in different timezone offsets is treated as a duplicate.
        by_timestamp[timestamp_dt.isoformat()] = (timestamp_dt, normalized)

    return [
        item[1]
        for item in sorted(by_timestamp.values(), key=lambda item: item[0])
    ]


async def fetch_historical_candles(
    instrument_key: str,
    unit: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    # Existing Upstox service is synchronous; keep it off the FastAPI event loop.
    raw = await asyncio.to_thread(
        get_historical_candles,
        instrument_key=instrument_key,
        unit=unit,
        interval=interval,
        to_date=to_date,
        from_date=from_date,
    )
    return _normalize(raw)
