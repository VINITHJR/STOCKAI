from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionConfigRequest(BaseModel):
    brokerage_rate: float = Field(0.0003, ge=0)
    brokerage_cap_per_order: float = Field(20.0, ge=0)
    stt_rate_sell: float = Field(0.00025, ge=0)
    exchange_rate: float = Field(0.0000297, ge=0)
    gst_rate: float = Field(0.18, ge=0)
    sebi_rate: float = Field(0.000001, ge=0)
    stamp_rate_buy: float = Field(0.00003, ge=0)
    slippage_rate: float = Field(0.0002, ge=0)


class BacktestRequest(BaseModel):
    instrument_key: str = Field(..., min_length=1)
    unit: str = Field("days", min_length=1)
    interval: str = Field("1", min_length=1)
    from_date: str = Field(..., min_length=10, max_length=10)
    to_date: str = Field(..., min_length=10, max_length=10)
    initial_capital: float = Field(100000.0, gt=0)
    quantity: int = Field(1, gt=0)
    short_window: int = Field(20, ge=2)
    long_window: int = Field(50, ge=3)
    execution: ExecutionConfigRequest = Field(
        default_factory=ExecutionConfigRequest
    )


class Trade(BaseModel):
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    execution_cost: float
    pnl: float
    return_percent: float


class BacktestResponse(BaseModel):
    success: bool
    layer: str
    data_source: str
    candle_count: int
    data: Dict[str, Any]
    metrics: Dict[str, Any]
    execution_costs: Dict[str, Any]
