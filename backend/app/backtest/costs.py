from __future__ import annotations

from dataclasses import dataclass
from backend.app.backtest.models import ExecutionConfigRequest


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage_cost: float
    total_cost: float


def _side_costs(price: float, quantity: int, config: ExecutionConfigRequest, *, is_buy: bool) -> CostBreakdown:
    value = float(price) * int(quantity)
    brokerage = min(value * config.brokerage_rate, config.brokerage_cap_per_order)
    stt = 0.0 if is_buy else value * config.stt_rate_sell
    exchange = value * config.exchange_rate
    gst = (brokerage + exchange) * config.gst_rate
    sebi = value * config.sebi_rate
    stamp = value * config.stamp_rate_buy if is_buy else 0.0
    slippage = value * config.slippage_rate
    total = brokerage + stt + exchange + gst + sebi + stamp + slippage
    return CostBreakdown(brokerage, stt, exchange, gst, sebi, stamp, slippage, total)


def calculate_entry_costs(entry_price: float, quantity: int, config: ExecutionConfigRequest) -> CostBreakdown:
    """Costs charged when opening a long position (buy side only)."""
    return _side_costs(entry_price, quantity, config, is_buy=True)


def calculate_exit_costs(exit_price: float, quantity: int, config: ExecutionConfigRequest) -> CostBreakdown:
    """Costs charged when closing a long position (sell side only)."""
    return _side_costs(exit_price, quantity, config, is_buy=False)


def calculate_costs(entry_price: float, exit_price: float, quantity: int, config: ExecutionConfigRequest) -> CostBreakdown:
    """Backward-compatible round-trip cost breakdown.

    The signature and fields are retained for existing Phase 8 callers.
    """
    buy = calculate_entry_costs(entry_price, quantity, config)
    sell = calculate_exit_costs(exit_price, quantity, config)
    return CostBreakdown(
        brokerage=buy.brokerage + sell.brokerage,
        stt=sell.stt,
        exchange_charges=buy.exchange_charges + sell.exchange_charges,
        gst=buy.gst + sell.gst,
        sebi_charges=buy.sebi_charges + sell.sebi_charges,
        stamp_duty=buy.stamp_duty,
        slippage_cost=buy.slippage_cost + sell.slippage_cost,
        total_cost=buy.total_cost + sell.total_cost,
    )
