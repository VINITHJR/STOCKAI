from __future__ import annotations

from typing import Any, Dict, List

from backend.app.backtest.strategies.base import BaseStrategy, Signal


class SMACrossoverStrategy(BaseStrategy):
    """Simple moving-average crossover strategy used by Layer 1 by default."""

    name = "sma"

    def __init__(self, short_window: int = 20, long_window: int = 50):
        short_window = int(short_window)
        long_window = int(long_window)

        if short_window < 2:
            raise ValueError("short_window must be at least 2.")
        if long_window < 3:
            raise ValueError("long_window must be at least 3.")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window.")

        self.short_window = short_window
        self.long_window = long_window

    @staticmethod
    def _sma(values: List[float], window: int) -> float:
        if len(values) < window:
            raise ValueError("Not enough values for SMA.")
        return sum(values[-window:]) / window

    def generate_signal(
        self,
        candles: List[Dict[str, Any]],
        index: int,
    ) -> Signal:
        if index < 0 or index >= len(candles):
            raise IndexError("Strategy candle index is out of range.")

        closes = [float(candle["close"]) for candle in candles[: index + 1]]
        if len(closes) < self.long_window:
            return "HOLD"

        short = self._sma(closes, self.short_window)
        long = self._sma(closes, self.long_window)

        if short > long:
            return "BUY"
        if short < long:
            return "SELL"
        return "HOLD"

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "SMA foundation test",
            "short_window": self.short_window,
            "long_window": self.long_window,
        }
