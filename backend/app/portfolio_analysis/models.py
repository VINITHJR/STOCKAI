"""
Phase 13 - Layer 1
Portfolio Analysis Models

Defines the core data structures used by the Portfolio Analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PortfolioPosition:
    """
    Represents one position in the portfolio.
    """

    instrument_key: str
    symbol: str = ""
    quantity: float = 0.0
    average_price: float = 0.0
    current_price: float = 0.0

    sector: str = ""
    exchange: str = ""
    asset_type: str = "EQUITY"

    @property
    def invested_value(self) -> float:
        """Total capital invested in this position."""
        return self.quantity * self.average_price

    @property
    def market_value(self) -> float:
        """Current market value of this position."""
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        """Absolute profit or loss."""
        return self.market_value - self.invested_value

    @property
    def pnl_percent(self) -> float:
        """Percentage profit or loss."""
        if self.invested_value <= 0:
            return 0.0

        return (self.pnl / self.invested_value) * 100


@dataclass
class PortfolioSnapshot:
    """
    Snapshot of the complete portfolio at a point in time.
    """

    positions: List[PortfolioPosition] = field(default_factory=list)

    initial_capital: float = 0.0
    cash: float = 0.0

    timestamp: Optional[str] = None

    @property
    def invested_value(self) -> float:
        """Total amount invested across all positions."""
        return sum(position.invested_value for position in self.positions)

    @property
    def market_value(self) -> float:
        """Current market value of all positions."""
        return sum(position.market_value for position in self.positions)

    @property
    def total_value(self) -> float:
        """Portfolio value including available cash."""
        return self.market_value + self.cash

    @property
    def total_pnl(self) -> float:
        """Total portfolio profit or loss."""
        return sum(position.pnl for position in self.positions)

    @property
    def total_pnl_percent(self) -> float:
        """Total portfolio P&L percentage."""
        if self.invested_value <= 0:
            return 0.0

        return (self.total_pnl / self.invested_value) * 100


@dataclass
class PositionAnalysis:
    """
    Analysis result for a single portfolio position.
    """

    instrument_key: str
    symbol: str

    quantity: float

    invested_value: float
    market_value: float

    pnl: float
    pnl_percent: float

    portfolio_weight: float

    sector: str = ""
    asset_type: str = ""

    status: str = "NEUTRAL"

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioAnalysis:
    """
    Complete Layer 1 portfolio analysis result.
    """

    position_count: int

    initial_capital: float
    cash: float

    invested_value: float
    market_value: float
    total_value: float

    total_pnl: float
    total_pnl_percent: float

    invested_percent: float
    cash_percent: float

    positions: List[PositionAnalysis] = field(default_factory=list)

    status: str = "HEALTHY"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result into a JSON-compatible dictionary."""

        return {
            "position_count": self.position_count,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "invested_value": self.invested_value,
            "market_value": self.market_value,
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_pnl_percent": self.total_pnl_percent,
            "invested_percent": self.invested_percent,
            "cash_percent": self.cash_percent,
            "status": self.status,
            "positions": [
                {
                    "instrument_key": position.instrument_key,
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "invested_value": position.invested_value,
                    "market_value": position.market_value,
                    "pnl": position.pnl,
                    "pnl_percent": position.pnl_percent,
                    "portfolio_weight": position.portfolio_weight,
                    "sector": position.sector,
                    "asset_type": position.asset_type,
                    "status": position.status,
                    "metadata": position.metadata,
                }
                for position in self.positions
            ],
            "metadata": self.metadata,
        }