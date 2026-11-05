from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.backtest.cache import HistoricalDataCache
from backend.app.backtest.costs import calculate_costs
from backend.app.backtest.historical_data import fetch_historical_candles
from backend.app.backtest.models import ExecutionConfigRequest, Trade
from backend.app.backtest.strategies import create_strategy


router = APIRouter(
    prefix="/api/backtest/portfolio",
    tags=["Phase 8 Layer 5 - Portfolio Backtest"],
)


class PortfolioAsset(BaseModel):
    instrument_key: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0, le=1)
    strategy: str = Field("sma", min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    quantity: int = Field(1, gt=0)


class PortfolioBacktestRequest(BaseModel):
    assets: List[PortfolioAsset] = Field(..., min_length=2)
    unit: str = Field("days", min_length=1)
    interval: str = Field("1", min_length=1)
    from_date: str = Field(..., min_length=10, max_length=10)
    to_date: str = Field(..., min_length=10, max_length=10)
    initial_capital: float = Field(100000.0, gt=0)
    execution: ExecutionConfigRequest = Field(
        default_factory=ExecutionConfigRequest
    )


def _validate_assets(assets: List[PortfolioAsset]) -> None:
    keys = [asset.instrument_key for asset in assets]
    if len(set(keys)) != len(keys):
        raise ValueError("Portfolio contains duplicate instrument_key values.")

    total_weight = sum(float(asset.weight) for asset in assets)
    if total_weight > 1.000001:
        raise ValueError(
            f"Portfolio weights must sum to 1.0 or less; got {total_weight:.6f}."
        )


async def _load_candles(
    asset: PortfolioAsset,
    request: PortfolioBacktestRequest,
) -> tuple[List[Dict[str, Any]], str]:
    cache = HistoricalDataCache()
    cached = cache.get(
        asset.instrument_key,
        request.unit,
        request.interval,
        request.from_date,
        request.to_date,
    )
    if cached is not None:
        return cached, "cache"

    candles = await fetch_historical_candles(
        instrument_key=asset.instrument_key,
        unit=request.unit,
        interval=request.interval,
        from_date=request.from_date,
        to_date=request.to_date,
    )
    if not candles:
        raise ValueError(
            f"No historical candles found for {asset.instrument_key}."
        )

    cache.set(
        asset.instrument_key,
        request.unit,
        request.interval,
        request.from_date,
        request.to_date,
        candles,
    )
    return candles, "upstox"


def _prepare_asset(
    asset: PortfolioAsset,
    candles: List[Dict[str, Any]],
    initial_allocation: float,
    execution: ExecutionConfigRequest,
) -> Dict[str, Any]:
    strategy = create_strategy(
        asset.strategy,
        asset.params,
        default_short_window=20,
        default_long_window=50,
    )

    # Use a separate cash ledger for each sleeve. This keeps portfolio
    # allocations independent while the portfolio equity is aggregated.
    cash = float(initial_allocation)
    position: Dict[str, Any] | None = None
    trades: List[Trade] = []
    equity_curve: List[Dict[str, Any]] = []
    total_cost = 0.0

    for index, candle in enumerate(candles):
        close = float(candle["close"])
        timestamp = str(candle["timestamp"])
        signal = strategy.generate_signal(candles, index)
        next_candle = candles[index + 1] if index + 1 < len(candles) else None

        if next_candle is not None:
            next_open = float(next_candle["open"])
            next_timestamp = str(next_candle["timestamp"])

            if position is None and signal == "BUY":
                quantity = int(asset.quantity)
                notional = next_open * quantity
                costs = calculate_costs(
                    next_open,
                    next_open,
                    quantity,
                    # Execution config is attached by caller after preparation.
                    # This placeholder is replaced before execution.
                    execution,
                )
                entry_cost = costs.buy_cost
                if cash >= notional + entry_cost:
                    cash -= notional + entry_cost
                    position = {
                        "entry_time": next_timestamp,
                        "entry_price": next_open,
                        "entry_cost": entry_cost,
                        "quantity": quantity,
                    }
                    total_cost += entry_cost

            elif position is not None and signal == "SELL":
                quantity = int(position["quantity"])
                entry = float(position["entry_price"])
                costs = calculate_costs(
                    entry,
                    next_open,
                    quantity,
                    execution,
                )
                gross = (next_open - entry) * quantity
                net = gross - costs.total_cost
                base = entry * quantity
                trade = Trade(
                    entry_time=str(position["entry_time"]),
                    exit_time=next_timestamp,
                    entry_price=entry,
                    exit_price=next_open,
                    quantity=quantity,
                    gross_pnl=round(gross, 4),
                    execution_cost=round(costs.total_cost, 4),
                    pnl=round(net, 4),
                    return_percent=round((net / base) * 100, 4) if base else 0.0,
                )
                cash += next_open * quantity - costs.total_cost
                total_cost += costs.total_cost
                trades.append(trade)
                position = None

        position_value = (
            close * int(position["quantity"])
            if position is not None
            else 0.0
        )
        equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": round(cash + position_value, 4),
                "cash": round(cash, 4),
                "position_value": round(position_value, 4),
                "signal": signal,
            }
        )

    if position is not None:
        last = candles[-1]
        exit_price = float(last["close"])
        exit_time = str(last["timestamp"])
        quantity = int(position["quantity"])
        entry = float(position["entry_price"])
        costs = calculate_costs(
            entry,
            exit_price,
            quantity,
            execution,
        )
        gross = (exit_price - entry) * quantity
        net = gross - costs.total_cost
        base = entry * quantity
        trade = Trade(
            entry_time=str(position["entry_time"]),
            exit_time=exit_time,
            entry_price=entry,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=round(gross, 4),
            execution_cost=round(costs.total_cost, 4),
            pnl=round(net, 4),
            return_percent=round((net / base) * 100, 4) if base else 0.0,
        )
        cash += exit_price * quantity - costs.total_cost
        total_cost += costs.total_cost
        trades.append(trade)
        if equity_curve:
            equity_curve[-1]["equity"] = round(cash, 4)
            equity_curve[-1]["cash"] = round(cash, 4)
            equity_curve[-1]["position_value"] = 0.0

    return {
        "instrument_key": asset.instrument_key,
        "weight": float(asset.weight),
        "initial_allocation": round(initial_allocation, 4),
        "final_value": round(cash, 4),
        "pnl": round(cash - initial_allocation, 4),
        "return_percent": round(
            ((cash - initial_allocation) / initial_allocation) * 100, 4
        ) if initial_allocation else 0.0,
        "strategy": strategy.describe(),
        "total_trades": len(trades),
        "execution_cost": round(total_cost, 4),
        "trades": [trade.model_dump() for trade in trades],
        "equity_curve": equity_curve,
    }



@router.post("/run")
async def run_portfolio_backtest(request: PortfolioBacktestRequest):
    try:
        _validate_assets(request.assets)

        total_weight = sum(float(asset.weight) for asset in request.assets)
        unallocated_capital = request.initial_capital * (1.0 - total_weight)

        asset_results: List[Dict[str, Any]] = []
        source_values: List[str] = []

        for asset in request.assets:
            candles, source = await _load_candles(asset, request)
            source_values.append(source)
            allocation = request.initial_capital * float(asset.weight)
            result = _prepare_asset(asset, candles, allocation, request.execution)
            result["candle_count"] = len(candles)
            asset_results.append(result)

        # Align each sleeve's equity curve by timestamp. Missing timestamps
        # carry the previous sleeve value so the portfolio can be aggregated
        # without changing any Layer 1 historical data behavior.
        by_timestamp: Dict[str, Dict[str, float]] = {}
        for asset_result in asset_results:
            previous = float(asset_result["initial_allocation"])
            for point in asset_result["equity_curve"]:
                timestamp = str(point["timestamp"])
                value = float(point["equity"])
                by_timestamp.setdefault(timestamp, {})[
                    asset_result["instrument_key"]
                ] = value
                previous = value

        timestamps = sorted(by_timestamp)
        last_values = {
            result["instrument_key"]: float(result["initial_allocation"])
            for result in asset_results
        }
        portfolio_curve: List[Dict[str, Any]] = []

        for timestamp in timestamps:
            row = by_timestamp[timestamp]
            for instrument_key, value in row.items():
                last_values[instrument_key] = value
            invested_value = sum(last_values.values())
            total_equity = invested_value + unallocated_capital
            portfolio_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": round(total_equity, 4),
                    "invested_value": round(invested_value, 4),
                    "unallocated_cash": round(unallocated_capital, 4),
                }
            )

        final_capital = (
            portfolio_curve[-1]["equity"]
            if portfolio_curve
            else request.initial_capital
        )
        total_pnl = final_capital - request.initial_capital
        total_trades = sum(
            int(result["total_trades"]) for result in asset_results
        )
        total_cost = sum(
            float(result["execution_cost"]) for result in asset_results
        )

        return {
            "success": True,
            "layer": "Phase 8 / Layer 5",
            "data_source": "cache" if source_values and all(
                source == "cache" for source in source_values
            ) else "upstox",
            "asset_count": len(asset_results),
            "initial_capital": request.initial_capital,
            "unallocated_cash": round(unallocated_capital, 4),
            "final_capital": round(final_capital, 4),
            "total_pnl": round(total_pnl, 4),
            "total_return_percent": round(
                (total_pnl / request.initial_capital) * 100, 4
            ),
            "total_trades": total_trades,
            "total_execution_cost": round(total_cost, 4),
            "allocation": {
                "total_weight": round(total_weight, 6),
                "unallocated_weight": round(1.0 - total_weight, 6),
            },
            "assets": asset_results,
            "portfolio_equity_curve": portfolio_curve,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
