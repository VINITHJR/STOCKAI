"""Service layer for the versioned StockAI LSTM predictor."""
from __future__ import annotations

from datetime import date, timedelta, datetime
import math
import threading
from typing import Dict, List

from .upstox_service import get_historical_candles, get_market_quotes, find_quote

_cache: Dict[str, dict] = {}
_cache_times: Dict[str, float] = {}
_lock = threading.RLock()
_max_cache = 64
_cache_ttl_seconds = 60


def _candles(instrument_key: str, days: int = 420) -> List[dict]:
    to_date = date.today()
    from_date = to_date - timedelta(days=days)
    result = get_historical_candles(
        instrument_key, "days", "1", to_date.isoformat(), from_date.isoformat()
    )
    raw = result.get("data", {}).get("candles", []) if isinstance(result, dict) else []
    output = []
    for candle in raw:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue
        try:
            item = {
                "timestamp": candle[0], "open": float(candle[1]), "high": float(candle[2]),
                "low": float(candle[3]), "close": float(candle[4]),
                "volume": float(candle[5] or 0),
            }
            if all(math.isfinite(item[k]) for k in ("open", "high", "low", "close", "volume")):
                output.append(item)
        except (TypeError, ValueError):
            continue
    output.sort(key=lambda x: str(x["timestamp"]))
    return output


def _current_price(instrument_key: str) -> float:
    quote = find_quote(get_market_quotes([instrument_key]), instrument_key)
    if not isinstance(quote, dict) or quote.get("last_price") is None:
        raise RuntimeError("Current market quote unavailable")
    price = float(quote["last_price"])
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError("Current market quote is not a finite positive price")
    return price


def get_prediction(instrument_key: str):
    key = str(instrument_key or "").strip()
    if not key:
        return {"status": "error", "error_code": "missing_instrument_key", "message": "instrument_key is required."}
    now = datetime.now().timestamp()
    with _lock:
        cached = _cache.get(key)
        if cached and now - _cache_times.get(key, 0) < _cache_ttl_seconds:
            return cached
        _cache.pop(key, None)
        _cache_times.pop(key, None)

    try:
        from backend.app.ml.lstm_model import LSTMPricePredictor, TENSORFLOW_AVAILABLE
        if not TENSORFLOW_AVAILABLE:
            return {"status": "error", "error_code": "tensorflow_unavailable", "message": "TensorFlow is unavailable for LSTM prediction."}
        candles = _candles(key)
        price = _current_price(key)
        predictor = LSTMPricePredictor(sequence_length=60, prediction_horizons=[1, 3, 7])
        if not predictor.load():
            return {
                "status": "error", "error_code": "v2_model_not_trained",
                "message": "The multi-stock LSTM v2 model is not installed. Run the supplied multi-stock training script first.",
                "required_model_file": predictor.model_file, "required_scaler_file": predictor.scaler_file,
            }
        result = predictor.predict(candles, price)
        if isinstance(result, dict) and result.get("status") == "success":
            with _lock:
                if len(_cache) >= _max_cache:
                    oldest = next(iter(_cache))
                    _cache.pop(oldest, None)
                    _cache_times.pop(oldest, None)
                _cache[key] = result
                _cache_times[key] = datetime.now().timestamp()
        # Never cache errors; they should clear as soon as artifacts/data are fixed.
        return result
    except Exception as exc:
        return {"status": "error", "error_code": "prediction_service_error", "message": str(exc)}


def get_prediction_with_analysis(instrument_key: str):
    result = get_prediction(instrument_key)
    count = 0
    if result.get("status") == "success":
        try:
            count = len(_candles(str(instrument_key).strip()))
        except Exception:
            count = 0
    return {**result, "analysis": {"instrument_key": instrument_key, "historical_candles_used": count}}


def clear_prediction_cache():
    with _lock:
        _cache.clear()
        _cache_times.clear()
    return {"status": "success", "message": "Prediction cache cleared"}


def get_cache_stats():
    with _lock:
        return {"size": len(_cache), "max_size": _max_cache, "ttl_seconds": _cache_ttl_seconds}
