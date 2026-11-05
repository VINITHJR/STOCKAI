from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.app.backtest.costs import calculate_entry_costs, calculate_exit_costs
from backend.app.backtest.models import BacktestRequest, Trade


class BacktestEngine:
    """Long-only SMA backtest engine with explicit bar timing.

    The signal is computed after a candle closes. That signal can only execute
    at the *next evaluation candle's open*. The equity point for a candle is
    marked after any action at that candle's open and at that candle's close.
    Warm-up candles update indicators only; they do not create trades/equity
    points. Any open position is liquidated at the final evaluation candle's
    close so reported final capital is cash.
    """

    def __init__(self, request: BacktestRequest):
        if request.short_window >= request.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        self.request = request

    @staticmethod
    def sma(values: List[float], window: int) -> float:
        if len(values) < window:
            raise ValueError("Not enough values for SMA.")
        return sum(values[-window:]) / window

    def _signal(self, closes: List[float]) -> str:
        if len(closes) < self.request.long_window:
            return "HOLD"
        short = self.sma(closes, self.request.short_window)
        long = self.sma(closes, self.request.long_window)
        if short > long:
            return "BUY"
        if short < long:
            return "SELL"
        return "HOLD"

    def run(self, candles: List[Dict[str, Any]], warmup_candle_count: int = 0) -> Dict[str, Any]:
        if not candles:
            raise ValueError("No historical candles available.")
        if isinstance(warmup_candle_count, bool) or not isinstance(warmup_candle_count, int):
            raise ValueError("warmup_candle_count must be a non-negative integer.")
        if warmup_candle_count < 0 or warmup_candle_count >= len(candles):
            if warmup_candle_count != 0:
                raise ValueError("warmup_candle_count must be smaller than the candle count.")

        initial = float(self.request.initial_capital)
        cash = initial
        position: Optional[Dict[str, Any]] = None
        closes: List[float] = []
        trades: List[Trade] = []
        equity_curve: List[Dict[str, Any]] = []
        total_execution_cost = 0.0
        evaluation = candles[warmup_candle_count:]
        if not evaluation:
            raise ValueError("No evaluation candles remain after warm-up.")

        # A signal generated at the prior close is executed at this candle's open.
        pending_signal: Optional[str] = None
        for index, candle in enumerate(candles):
            close = float(candle["close"])
            timestamp = str(candle["timestamp"])
            closes.append(close)
            if index < warmup_candle_count:
                continue

            open_price = float(candle["open"])
            executed_action = "NONE"
            if pending_signal == "BUY" and position is None:
                quantity = int(self.request.quantity)
                entry_cost = calculate_entry_costs(
                    open_price, quantity, self.request.execution
                ).total_cost
                notional = open_price * quantity
                if cash >= notional + entry_cost:
                    cash -= notional + entry_cost
                    total_execution_cost += entry_cost
                    position = {
                        "entry_time": timestamp,
                        "entry_price": open_price,
                        "signal_time": str(candles[index - 1]["timestamp"]),
                        "entry_cost": entry_cost,
                    }
                    executed_action = "BUY"
            elif pending_signal == "SELL" and position is not None:
                trade, exit_cost = self._close(position, timestamp, open_price)
                cash += open_price * int(self.request.quantity) - exit_cost
                total_execution_cost += exit_cost
                trades.append(trade)
                position = None
                executed_action = "SELL"

            position_value = close * int(self.request.quantity) if position is not None else 0.0
            estimated_exit_cost = (
                calculate_exit_costs(close, int(self.request.quantity), self.request.execution).total_cost
                if position is not None else 0.0
            )
            equity = cash + position_value - estimated_exit_cost

            # The current close produces a signal for the next evaluation candle.
            signal = self._signal(closes)
            equity_curve.append({
                "timestamp": timestamp,
                "equity": round(equity, 4),
                "cash": round(cash, 4),
                "position_value": round(position_value, 4),
                "estimated_exit_cost": round(estimated_exit_cost, 4),
                "signal": signal,
                "executed_action": executed_action,
                "forced_close": False,
            })
            pending_signal = signal

        # A strategy cannot execute a signal after the final candle. Liquidate
        # any position still open at the final evaluation close.
        if position is not None:
            last = evaluation[-1]
            close = float(last["close"])
            trade, exit_cost = self._close(position, str(last["timestamp"]), close)
            cash += close * int(self.request.quantity) - exit_cost
            total_execution_cost += exit_cost
            trades.append(trade)
            equity_curve[-1].update({
                "equity": round(cash, 4),
                "cash": round(cash, 4),
                "position_value": 0.0,
                "estimated_exit_cost": 0.0,
                "forced_close": True,
            })

        final_capital = cash
        total_pnl = final_capital - initial
        return {
            "instrument_key": self.request.instrument_key,
            "from_date": self.request.from_date,
            "to_date": self.request.to_date,
            "initial_capital": initial,
            "final_capital": round(final_capital, 4),
            "total_pnl": round(total_pnl, 4),
            "total_return_percent": total_pnl / initial * 100,
            "total_trades": len(trades),
            "total_execution_cost": round(total_execution_cost, 4),
            "warmup_candle_count": warmup_candle_count,
            "evaluation_candle_count": len(evaluation),
            "strategy": {
                "type": "SMA foundation test",
                "short_window": self.request.short_window,
                "long_window": self.request.long_window,
                "execution_rule": "completed-candle signal -> next evaluation-candle open",
                "cost_accounting": "buy costs at entry; sell costs at exit; open-position equity net of estimated exit costs",
            },
            "trades": [trade.model_dump() for trade in trades],
            "equity_curve": equity_curve,
        }

    def _close(self, position: Dict[str, Any], exit_time: str, exit_price: float):
        entry = float(position["entry_price"])
        quantity = int(self.request.quantity)
        gross = (exit_price - entry) * quantity
        entry_cost = float(position["entry_cost"])
        exit_cost = calculate_exit_costs(exit_price, quantity, self.request.execution).total_cost
        total_cost = entry_cost + exit_cost
        net = gross - total_cost
        base = entry * quantity
        trade = Trade(
            entry_time=str(position["entry_time"]),
            exit_time=exit_time,
            entry_price=entry,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=round(gross, 4),
            execution_cost=round(total_cost, 4),
            pnl=round(net, 4),
            return_percent=round(net / base * 100 if base else 0.0, 4),
        )
        return trade, exit_cost
