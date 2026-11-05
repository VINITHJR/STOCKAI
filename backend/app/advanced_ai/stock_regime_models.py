from typing import List, Dict, Any
from pydantic import BaseModel, Field


class StockRegimeInput(BaseModel):
    """
    Phase 14.2 input.

    Stock regime is calculated independently from the selected
    stock's OHLCV candles.

    This module must not consume:
    - RandomForest predictions
    - LSTM predictions
    - Phase 9 signals
    - Phase 14 Layer 1 decisions
    """

    instrument_key: str

    candles: List[Dict[str, Any]]

    min_candles: int = Field(
        default=60,
        ge=20,
    )


class StockRegimeResult(BaseModel):
    """
    Independent stock-regime intelligence result.
    """

    instrument_key: str

    stock_regime: str
    stock_direction: str
    stock_regime_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    trend_strength: float = Field(
        ge=0.0,
        le=1.0,
    )

    momentum_regime: str
    volatility_regime: str
    price_structure: str

    current_price: float

    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None

    momentum_20d_percent: float = 0.0

    atr_percent: float = 0.0

    adx_14: float = 0.0

    plus_di: float = 0.0
    minus_di: float = 0.0

    candle_count: int

    evidence: List[str] = Field(
        default_factory=list,
    )

    method: str = (
        "deterministic_ohlcv_stock_regime_v1"
    )

    status: str = "success"