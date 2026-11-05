from __future__ import annotations
from .prediction_service import _candles
import pandas as pd
import threading
_cache={}; _lock=threading.RLock(); _max_cache=64

def get_trend_prediction(instrument_key:str):
    key=instrument_key.strip()
    with _lock:
        if key in _cache: return _cache[key]
    from backend.app.ml.rf_trend_classifier import RFTrendClassifier
    candles=_candles(key); df=pd.DataFrame(candles)
    model=RFTrendClassifier()
    try:
        model.load_model()
    except Exception as exc:
        raise RuntimeError(f"Saved Random Forest trend model could not be loaded: {exc}") from exc
    value=model.predict_trend(df); value.update({"status":"success","instrument_key":key})
    with _lock:
        if len(_cache)>=_max_cache: _cache.pop(next(iter(_cache)))
        _cache[key]=value
    return value

def get_trend_with_context(instrument_key:str):
    result=get_trend_prediction(instrument_key); return {**result,"context":{"model":"RandomForest","feature_source":"historical OHLCV + technical indicators"}}

def clear_trend_cache():
    with _lock: _cache.clear()
    return {"status":"success","message":"Trend cache cleared"}

def get_trend_cache_stats():
    with _lock: return {"size":len(_cache),"max_size":_max_cache}
