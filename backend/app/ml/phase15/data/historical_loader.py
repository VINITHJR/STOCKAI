"""
Phase 15 historical data loader.

Research-only module.
Does not modify production models, scalers, Phase 9 or Phase 14.
"""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any, Callable, Dict, List
from zoneinfo import ZoneInfo


INDIA_TZ = ZoneInfo("Asia/Kolkata")


class HistoricalDataError(RuntimeError):
    """Raised when historical market data fails validation."""


def _to_date(value: date | str) -> date:
    """Convert a date or ISO date string to datetime.date."""
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def _parse_session_date(timestamp: Any) -> tuple[datetime, date]:
    """
    Parse a candle timestamp and return its IST datetime and session date.

    Naive timestamps are rejected because their timezone cannot be
    established reliably.
    """
    try:
        raw = str(timestamp).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise HistoricalDataError(
            f"Invalid candle timestamp: {timestamp!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise HistoricalDataError(
            f"Timezone missing from candle timestamp: {timestamp!r}"
        )

    ist_timestamp = parsed.astimezone(INDIA_TZ)
    return ist_timestamp, ist_timestamp.date()


def validate_daily_candles(
    instrument_key: str,
    candles: List[Any],
    start_date: date | str,
    end_date: date | str,
) -> Dict[str, Any]:
    """
    Validate and normalize Upstox daily candles.

    Expected Upstox candle format:
    [timestamp, open, high, low, close, volume, optional ...]

    The returned rows are sorted oldest-to-newest.
    Invalid rows and duplicate sessions are rejected rather than silently
    discarded.
    """
    key = str(instrument_key).strip()

    parts = key.split("|")

    if (
        len(parts) != 2
        or not parts[0]
        or not parts[1]
        or parts[0] != parts[0].strip()
        or parts[1] != parts[1].strip()
        or "#" in key
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
    ):
        raise ValueError(f"Invalid instrument key: {instrument_key!r}")

    start = _to_date(start_date)
    end = _to_date(end_date)

    if start > end:
        raise ValueError("start_date must not be later than end_date.")

    if not isinstance(candles, list) or not candles:
        raise HistoricalDataError(
            f"No historical candles returned for {key}."
        )

    rows: List[Dict[str, Any]] = []
    seen_dates = set()

    for index, candle in enumerate(candles):

        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            raise HistoricalDataError(
                f"{key}: malformed candle at response index {index}."
            )

        ist_timestamp, session_date = _parse_session_date(candle[0])

        if not (start <= session_date <= end):
            raise HistoricalDataError(
                f"{key}: session {session_date} is outside "
                f"the requested range {start} to {end}."
            )

        if session_date in seen_dates:
            raise HistoricalDataError(
                f"{key}: duplicate daily session {session_date}."
            )

        try:
            open_price, high_price, low_price, close_price, volume = (
                float(value) for value in candle[1:6]
            )
        except (TypeError, ValueError) as exc:
            raise HistoricalDataError(
                f"{key}: non-numeric OHLCV at {session_date}."
            ) from exc

        values = [
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
        ]

        if not all(isfinite(value) for value in values):
            raise HistoricalDataError(
                f"{key}: non-finite OHLCV at {session_date}."
            )

        if min(open_price, high_price, low_price, close_price) <= 0:
            raise HistoricalDataError(
                f"{key}: non-positive price at {session_date}."
            )

        if volume < 0:
            raise HistoricalDataError(
                f"{key}: negative volume at {session_date}."
            )

        if high_price < max(open_price, low_price, close_price):
            raise HistoricalDataError(
                f"{key}: invalid high price at {session_date}."
            )

        if low_price > min(open_price, high_price, close_price):
            raise HistoricalDataError(
                f"{key}: invalid low price at {session_date}."
            )

        seen_dates.add(session_date)

        rows.append({
            "instrument_key": key,
            "timestamp": str(candle[0]),
            "timestamp_ist": ist_timestamp.isoformat(),
            "session_date": session_date.isoformat(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        })

    rows.sort(key=lambda row: row["session_date"])

    return {
        "instrument_key": key,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "candle_count": len(rows),
        "first_session": rows[0]["session_date"],
        "last_session": rows[-1]["session_date"],
        "ordering": "ascending",
        "rows": rows,
    }


def fetch_daily_candles(
    instrument_key: str,
    start_date: date | str,
    end_date: date | str,
    fetch_function: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Fetch and validate daily candles through the existing Upstox service.

    A custom fetch_function can be supplied for isolated testing.
    """
    start = _to_date(start_date)
    end = _to_date(end_date)

    if start > end:
        raise ValueError("start_date must not be later than end_date.")

    if fetch_function is None:
        from backend.app.services.upstox_service import (
            get_historical_candles,
        )
        fetch_function = get_historical_candles

    try:
        result = fetch_function(
            instrument_key=instrument_key,
            unit="days",
            interval="1",
            to_date=end.isoformat(),
            from_date=start.isoformat(),
        )
    except Exception as exc:
        raise HistoricalDataError(
            f"Upstox request failed for {instrument_key}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise HistoricalDataError("Upstox returned a non-dictionary response.")

    if result.get("status") != "success":
        raise HistoricalDataError(
            f"Upstox returned unsuccessful status for {instrument_key}: "
            f"{result.get('status')!r}"
        )

    data = result.get("data")

    if not isinstance(data, dict):
        raise HistoricalDataError(
            f"Invalid response data for {instrument_key}."
        )

    candles = data.get("candles")

    if not isinstance(candles, list):
        raise HistoricalDataError(
            f"Missing or invalid candles for {instrument_key}."
        )

    return validate_daily_candles(
        instrument_key=instrument_key,
        candles=candles,
        start_date=start,
        end_date=end,
    )


def compare_session_calendars(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two validated datasets by their trading-session dates."""
    reference_dates = {
        row["session_date"] for row in reference["rows"]
    }
    candidate_dates = {
        row["session_date"] for row in candidate["rows"]
    }

    missing = sorted(reference_dates - candidate_dates)
    extra = sorted(candidate_dates - reference_dates)

    return {
        "reference_instrument": reference["instrument_key"],
        "candidate_instrument": candidate["instrument_key"],
        "reference_sessions": len(reference_dates),
        "candidate_sessions": len(candidate_dates),
        "missing_sessions": missing,
        "extra_sessions": extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "exact_match": not missing and not extra,
    }
