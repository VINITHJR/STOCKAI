from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


def _trade_value(trade: Any, key: str) -> float:
    value = trade.get(key, 0.0) if isinstance(trade, dict) else getattr(trade, key, 0.0)
    return float(value or 0.0)


def _date_value(value: str):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return None


def calculate_metrics(
    initial_capital: float,
    trades: List[Any],
    equity_curve: List[Dict[str, Any]],
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    periods_per_year: Optional[float] = None,
    engine_final_capital: Optional[float] = None,
) -> Dict[str, Any]:
    initial = float(initial_capital)
    if not isfinite(initial) or initial <= 0:
        raise ValueError("initial_capital must be finite and positive.")

    pnls = [_trade_value(trade, "pnl") for trade in trades]
    if not all(isfinite(pnl) for pnl in pnls):
        raise ValueError("Trade P&L contains a non-finite value.")
    ledger_final = initial + sum(pnls)
    if engine_final_capital is None:
        final = ledger_final
    else:
        final = float(engine_final_capital)
        if not isfinite(final) or final < 0:
            raise ValueError("engine_final_capital must be finite and non-negative.")
        tolerance = max(0.02, len(trades) * 0.0002 + 0.005)
        if abs(final - ledger_final) > tolerance:
            raise ValueError(
                "Trade ledger does not reconcile with engine final capital "
                f"(difference={final - ledger_final:.4f}, tolerance={tolerance:.4f})."
            )

    total_pnl = final - initial
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    wins = sum(p > 0 for p in pnls)
    losses = sum(p < 0 for p in pnls)

    # Drawdown is computed from bar-level equity (which now reflects actions at
    # that bar's open) and the initial capital. Trade-level drawdown is a
    # fallback for callers that do not provide a usable equity curve.
    max_drawdown = 0.0
    max_drawdown_percent = 0.0
    bar_peak = initial
    valid_equity_points = []
    for point in equity_curve or []:
        try:
            value = float(point["equity"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isfinite(value) or value < 0:
            continue
        valid_equity_points.append(point)
        bar_peak = max(bar_peak, value)
        drawdown = bar_peak - value
        drawdown_percent = drawdown / bar_peak * 100 if bar_peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_percent = max(max_drawdown_percent, drawdown_percent)

    if not valid_equity_points:
        equity = initial
        peak = initial
        for trade, pnl in zip(trades, pnls):
            equity += pnl
            peak = max(peak, equity)
            drawdown = peak - equity
            drawdown_percent = drawdown / peak * 100 if peak else 0.0
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_percent = max(max_drawdown_percent, drawdown_percent)

    cagr = None
    start = _date_value(from_date) if from_date else None
    end = _date_value(to_date) if to_date else None
    if start and end and end > start and final > 0:
        years = (end - start).days / 365.25
        if years > 0:
            cagr = ((final / initial) ** (1 / years) - 1) * 100

    returns: List[float] = []
    prior = initial
    for point in valid_equity_points:
        try:
            current = float(point["equity"])
        except (KeyError, TypeError, ValueError):
            continue
        if prior > 0 and isfinite(current):
            returns.append(current / prior - 1.0)
        if isfinite(current):
            prior = current

    sharpe = None
    if len(returns) >= 2:
        volatility = pstdev(returns)
        annualizer = (
            float(periods_per_year) ** 0.5
            if periods_per_year is not None and isfinite(float(periods_per_year)) and float(periods_per_year) > 0
            else None
        )
        if volatility > 0 and annualizer:
            sharpe = mean(returns) / volatility * annualizer

    return {
        "final_capital": round(final, 4),
        "total_pnl": round(total_pnl, 4),
        "total_return_percent": total_pnl / initial * 100,
        "total_trades": len(trades),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": round(wins / len(trades) * 100, 4) if trades else 0.0,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "average_win": round(gross_profit / wins, 4) if wins else 0.0,
        "average_loss": round(gross_loss / losses, 4) if losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else None,
        "max_drawdown": round(max_drawdown, 4),
        "max_drawdown_percent": round(max_drawdown_percent, 4),
        "cagr_percent": round(cagr, 4) if cagr is not None and isfinite(cagr) else None,
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None and isfinite(sharpe) else None,
        "periodic_return_count": len(returns),
        "equity_curve": valid_equity_points,
    }
