from __future__ import annotations

from typing import Any, Dict

from backend.app.backtest.strategies.base import BaseStrategy
from backend.app.backtest.strategies.sma import SMACrossoverStrategy
from backend.app.backtest.strategies.technical import (
    BollingerBandsStrategy,
    EMACrossoverStrategy,
    MACDStrategy,
    RSIStrategy,
)


_STRATEGIES = {
    "sma": SMACrossoverStrategy,
    "sma_crossover": SMACrossoverStrategy,
    "ema": EMACrossoverStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerBandsStrategy,
    "bollinger_bands": BollingerBandsStrategy,
}


def create_strategy(
    name: str,
    params: Dict[str, Any] | None = None,
    *,
    default_short_window: int = 20,
    default_long_window: int = 50,
) -> BaseStrategy:
    strategy_name = str(name or "sma").strip().lower()
    strategy_class = _STRATEGIES.get(strategy_name)

    if strategy_class is None:
        supported = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"Unknown strategy '{name}'. Supported strategies: {supported}."
        )

    supplied = dict(params or {})

    if strategy_name in {"ema", "ema_crossover"}:
        fast_window = supplied.pop("fast_window", 12)
        slow_window = supplied.pop("slow_window", 26)
        if supplied:
            unknown = ", ".join(sorted(str(key) for key in supplied))
            raise ValueError(f"Unsupported parameters for EMA strategy: {unknown}.")
        return strategy_class(fast_window=fast_window, slow_window=slow_window)

    if strategy_name == "rsi":
        window = supplied.pop("window", 14)
        oversold = supplied.pop("oversold", 30.0)
        overbought = supplied.pop("overbought", 70.0)
        if supplied:
            unknown = ", ".join(sorted(str(key) for key in supplied))
            raise ValueError(f"Unsupported parameters for RSI strategy: {unknown}.")
        return strategy_class(window=window, oversold=oversold, overbought=overbought)

    if strategy_name == "macd":
        fast_window = supplied.pop("fast_window", 12)
        slow_window = supplied.pop("slow_window", 26)
        signal_window = supplied.pop("signal_window", 9)
        if supplied:
            unknown = ", ".join(sorted(str(key) for key in supplied))
            raise ValueError(f"Unsupported parameters for MACD strategy: {unknown}.")
        return strategy_class(
            fast_window=fast_window,
            slow_window=slow_window,
            signal_window=signal_window,
        )

    if strategy_name in {"bollinger", "bollinger_bands"}:
        window = supplied.pop("window", 20)
        standard_deviations = supplied.pop("standard_deviations", 2.0)
        if supplied:
            unknown = ", ".join(sorted(str(key) for key in supplied))
            raise ValueError(f"Unsupported parameters for Bollinger strategy: {unknown}.")
        return strategy_class(window=window, standard_deviations=standard_deviations)

    if strategy_name in {"sma", "sma_crossover"}:
        short_window = supplied.pop("short_window", default_short_window)
        long_window = supplied.pop("long_window", default_long_window)
        if supplied:
            unknown = ", ".join(sorted(str(key) for key in supplied))
            raise ValueError(
                f"Unsupported parameters for SMA strategy: {unknown}."
            )
        return strategy_class(
            short_window=short_window,
            long_window=long_window,
        )

    # Kept as a defensive fallback if more strategies are registered later.
    return strategy_class(**supplied)
