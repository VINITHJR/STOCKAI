from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.backtest.historical_data import validate_normalized_candles


class HistoricalDataCache:
    """JSON cache for validated historical candles.

    Cache keys remain compatible with the existing instrument/unit/interval/date
    scheme. Invalid or stale cache entries return None so the caller can refetch.
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = str(
                Path(__file__).resolve().parents[2] / "data" / "historical_cache"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(
        self,
        instrument_key: str,
        unit: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> Path:
        raw = "|".join([instrument_key, unit, interval, from_date, to_date])
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    def get(
        self,
        instrument_key: str,
        unit: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> Optional[List[Dict[str, Any]]]:
        path = self._path(instrument_key, unit, interval, from_date, to_date)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None

            expected = {
                "instrument_key": instrument_key,
                "unit": unit,
                "interval": interval,
                "from_date": from_date,
                "to_date": to_date,
            }
            # Refuse a file whose metadata does not match the requested cache key.
            if any(payload.get(key) != value for key, value in expected.items()):
                return None

            candles = payload.get("candles")
            if not isinstance(candles, list):
                return None

            # Strict validation: don't serve old cache files containing invalid
            # candles or duplicate timestamps. A cache miss lets the caller refetch.
            validated = validate_normalized_candles(candles, reject_duplicates=True)
            if len(validated) != len(candles):
                return None
            return validated
        except (OSError, json.JSONDecodeError, ValueError, TypeError, OverflowError):
            return None

    def set(
        self,
        instrument_key: str,
        unit: str,
        interval: str,
        from_date: str,
        to_date: str,
        candles: List[Dict[str, Any]],
    ) -> None:
        validated = validate_normalized_candles(candles, reject_duplicates=True)
        path = self._path(instrument_key, unit, interval, from_date, to_date)
        payload = {
            "instrument_key": instrument_key,
            "unit": unit,
            "interval": interval,
            "from_date": from_date,
            "to_date": to_date,
            "candle_count": len(validated),
            "candles": validated,
        }

        # allow_nan=False prevents writing NaN/Infinity to cache JSON.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def clear(self) -> int:
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        return count
