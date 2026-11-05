from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

Signal = Literal["BUY", "SELL", "HOLD"]


class BaseStrategy(ABC):
    """Common interface for Phase 8 backtest strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signal(
        self,
        candles: List[Dict[str, Any]],
        index: int,
    ) -> Signal:
        """Return the signal available at the close of ``index``."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Return serialisable strategy metadata for the API response."""
        raise NotImplementedError
