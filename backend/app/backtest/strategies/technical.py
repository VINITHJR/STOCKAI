from __future__ import annotations

from typing import Any, Dict, List
import math

from backend.app.backtest.strategies.base import BaseStrategy, Signal



def _closes(candles: List[Dict[str, Any]], index: int) -> List[float]:
    if index < 0 or index >= len(candles):
        raise IndexError("Strategy candle index is out of range.")
    return [float(candle["close"]) for candle in candles[: index + 1]]


def _sma(values: List[float], window: int) -> float:
    if len(values) < window:
        raise ValueError("Not enough values for SMA.")
    return sum(values[-window:]) / window


def _ema_series(values: List[float], window: int) -> List[float]:
    if len(values) < window:
        return []
    multiplier = 2.0 / (window + 1.0)
    ema = sum(values[:window]) / window
    result = [ema]
    for value in values[window:]:
        ema = (value - ema) * multiplier + ema
        result.append(ema)
    return result


def _rsi(values: List[float], window: int) -> float | None:
    if len(values) <= window:
        return None

    gains: List[float] = []
    losses: List[float] = []
    for previous, current in zip(values[-(window + 1):-1], values[-window:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    average_gain = sum(gains) / window
    average_loss = sum(losses) / window

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


class EMACrossoverStrategy(BaseStrategy):
    """Layer 3 technical strategy: fast EMA crossing slow EMA."""

    name = "ema"

    def __init__(self, fast_window: int = 12, slow_window: int = 26):
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        if self.fast_window < 2:
            raise ValueError("fast_window must be at least 2.")
        if self.slow_window < 3:
            raise ValueError("slow_window must be at least 3.")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window.")

    def generate_signal(self, candles: List[Dict[str, Any]], index: int) -> Signal:
        closes = _closes(candles, index)
        if len(closes) < self.slow_window:
            return "HOLD"
        fast = _ema_series(closes, self.fast_window)[-1]
        slow = _ema_series(closes, self.slow_window)[-1]
        if fast > slow:
            return "BUY"
        if fast < slow:
            return "SELL"
        return "HOLD"

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "EMA crossover",
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
        }


class RSIStrategy(BaseStrategy):
    """Layer 3 technical strategy: RSI oversold/overbought signals."""

    name = "rsi"

    def __init__(self, window: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.window = int(window)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if not 0 <= self.oversold < self.overbought <= 100:
            raise ValueError("RSI thresholds must satisfy 0 <= oversold < overbought <= 100.")

    def generate_signal(self, candles: List[Dict[str, Any]], index: int) -> Signal:
        closes = _closes(candles, index)
        value = _rsi(closes, self.window)
        if value is None:
            return "HOLD"
        if value <= self.oversold:
            return "BUY"
        if value >= self.overbought:
            return "SELL"
        return "HOLD"

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "RSI mean-reversion thresholds",
            "window": self.window,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }


class MACDStrategy(BaseStrategy):
    """Layer 3 technical strategy: MACD line versus signal line."""

    name = "macd"

    def __init__(self, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.signal_window = int(signal_window)
        if self.fast_window < 2:
            raise ValueError("fast_window must be at least 2.")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window.")
        if self.signal_window < 2:
            raise ValueError("signal_window must be at least 2.")

    def generate_signal(self, candles: List[Dict[str, Any]], index: int) -> Signal:
        closes = _closes(candles, index)
        if len(closes) < self.slow_window + self.signal_window - 1:
            return "HOLD"

        fast_series = _ema_series(closes, self.fast_window)
        slow_series = _ema_series(closes, self.slow_window)
        offset = self.slow_window - self.fast_window
        macd_series = [fast_series[i + offset] - slow_series[i] for i in range(len(slow_series))]
        if len(macd_series) < self.signal_window:
            return "HOLD"

        signal_series = _ema_series(macd_series, self.signal_window)
        if not signal_series:
            return "HOLD"
        macd_value = macd_series[-1]
        signal_value = signal_series[-1]
        if macd_value > signal_value:
            return "BUY"
        if macd_value < signal_value:
            return "SELL"
        return "HOLD"

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "MACD",
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "signal_window": self.signal_window,
        }


class BollingerBandsStrategy(BaseStrategy):
    """Layer 3 technical strategy: Bollinger Band threshold signals."""

    name = "bollinger"

    def __init__(self, window: int = 20, standard_deviations: float = 2.0):
        self.window = int(window)
        self.standard_deviations = float(standard_deviations)
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.standard_deviations <= 0:
            raise ValueError("standard_deviations must be greater than 0.")

    def generate_signal(self, candles: List[Dict[str, Any]], index: int) -> Signal:
        closes = _closes(candles, index)
        if len(closes) < self.window:
            return "HOLD"
        window_values = closes[-self.window:]
        middle = sum(window_values) / self.window
        variance = sum((value - middle) ** 2 for value in window_values) / self.window
        deviation = math.sqrt(variance)
        upper = middle + self.standard_deviations * deviation
        lower = middle - self.standard_deviations * deviation
        current = closes[-1]
        if current <= lower:
            return "BUY"
        if current >= upper:
            return "SELL"
        return "HOLD"

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "Bollinger Bands",
            "window": self.window,
            "standard_deviations": self.standard_deviations,
        }
