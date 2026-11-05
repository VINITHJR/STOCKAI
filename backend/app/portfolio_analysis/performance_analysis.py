"""
Phase 13 - Layer 2
Performance Analysis

Analyzes portfolio performance using the portfolio positions
and Layer 1 portfolio analysis.

This layer does not modify:
- Phase 11 Paper Trading
- Phase 12 Risk Management
- Phase 13 Layer 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .portfolio_analyzer import PortfolioAnalyzer


@dataclass
class PerformanceAnalysis:
    """Portfolio performance analysis result."""

    initial_capital: float
    invested_value: float
    market_value: float
    cash: float
    total_value: float

    total_pnl: float
    total_return_percent: float

    realized_pnl: float
    unrealized_pnl: float

    winning_positions: int
    losing_positions: int
    neutral_positions: int

    win_rate_percent: float

    best_position: Optional[Dict[str, Any]] = None
    worst_position: Optional[Dict[str, Any]] = None

    average_position_return_percent: float = 0.0

    status: str = "NEUTRAL"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible representation."""

        return {
            "initial_capital": self.initial_capital,
            "invested_value": self.invested_value,
            "market_value": self.market_value,
            "cash": self.cash,
            "total_value": self.total_value,
            "total_pnl": self.total_pnl,
            "total_return_percent": self.total_return_percent,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "winning_positions": self.winning_positions,
            "losing_positions": self.losing_positions,
            "neutral_positions": self.neutral_positions,
            "win_rate_percent": self.win_rate_percent,
            "best_position": self.best_position,
            "worst_position": self.worst_position,
            "average_position_return_percent": (
                self.average_position_return_percent
            ),
            "status": self.status,
            "metadata": self.metadata,
        }


class PerformanceAnalyzer:
    """
    Phase 13 Layer 2 performance analyzer.

    Layer 2 calculates:
    - total portfolio return
    - position-level performance
    - winning positions
    - losing positions
    - neutral positions
    - win rate
    - best position
    - worst position
    - average position return

    Realized P&L is only calculated when explicitly supplied.
    It is not guessed from current holdings.
    """

    def __init__(self, currency: str = "INR") -> None:
        self.currency = currency
        self.portfolio_analyzer = PortfolioAnalyzer(currency=currency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        positions: Optional[Iterable[Mapping[str, Any]]] = None,
        cash: float = 0.0,
        initial_capital: float = 0.0,
        realized_pnl: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> PerformanceAnalysis:
        """
        Analyze portfolio performance.

        Parameters
        ----------
        positions:
            Portfolio position dictionaries.

        cash:
            Current available cash.

        initial_capital:
            Original portfolio capital.

        realized_pnl:
            Explicitly supplied realized P&L.
            Defaults to zero because it cannot safely be inferred
            from open positions.

        timestamp:
            Optional analysis timestamp.
        """

        portfolio = self.portfolio_analyzer.analyze(
            positions=positions,
            cash=cash,
            initial_capital=initial_capital,
            timestamp=timestamp,
        )

        winning_positions = 0
        losing_positions = 0
        neutral_positions = 0

        best_position: Optional[Dict[str, Any]] = None
        worst_position: Optional[Dict[str, Any]] = None

        total_position_return = 0.0

        for position in portfolio.positions:
            pnl_percent = float(position.pnl_percent)

            total_position_return += pnl_percent

            if pnl_percent > 0:
                winning_positions += 1

            elif pnl_percent < 0:
                losing_positions += 1

            else:
                neutral_positions += 1

            position_data = {
                "instrument_key": position.instrument_key,
                "symbol": position.symbol,
                "pnl": position.pnl,
                "pnl_percent": position.pnl_percent,
                "market_value": position.market_value,
                "invested_value": position.invested_value,
            }

            if (
                best_position is None
                or pnl_percent > best_position["pnl_percent"]
            ):
                best_position = position_data

            if (
                worst_position is None
                or pnl_percent < worst_position["pnl_percent"]
            ):
                worst_position = position_data

        total_positions = (
            winning_positions
            + losing_positions
            + neutral_positions
        )

        win_rate_percent = self._calculate_win_rate(
            winning_positions=winning_positions,
            losing_positions=losing_positions,
        )

        average_position_return_percent = self._calculate_average_return(
            total_position_return=total_position_return,
            position_count=total_positions,
        )

        unrealized_pnl = portfolio.total_pnl

        realized_pnl = self._safe_float(realized_pnl)

        total_pnl = realized_pnl + unrealized_pnl

        total_return_percent = self._calculate_total_return(
            total_pnl=total_pnl,
            initial_capital=portfolio.initial_capital,
        )

        status = self._determine_status(
            total_return_percent=total_return_percent,
            win_rate_percent=win_rate_percent,
        )

        return PerformanceAnalysis(
            initial_capital=self._round(portfolio.initial_capital),
            invested_value=self._round(portfolio.invested_value),
            market_value=self._round(portfolio.market_value),
            cash=self._round(portfolio.cash),
            total_value=self._round(portfolio.total_value),
            total_pnl=self._round(total_pnl),
            total_return_percent=self._round(total_return_percent),
            realized_pnl=self._round(realized_pnl),
            unrealized_pnl=self._round(unrealized_pnl),
            winning_positions=winning_positions,
            losing_positions=losing_positions,
            neutral_positions=neutral_positions,
            win_rate_percent=self._round(win_rate_percent),
            best_position=best_position,
            worst_position=worst_position,
            average_position_return_percent=self._round(
                average_position_return_percent
            ),
            status=status,
            metadata={
                "currency": self.currency,
                "layer": 2,
                "component": "performance_analysis",
                "realized_pnl_source": (
                    "explicit_input"
                    if realized_pnl != 0
                    else "not_provided"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Performance calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_win_rate(
        winning_positions: int,
        losing_positions: int,
    ) -> float:
        """
        Calculate win rate using only winning and losing positions.

        Neutral positions are excluded from the denominator.
        """

        evaluated_positions = winning_positions + losing_positions

        if evaluated_positions == 0:
            return 0.0

        return (
            winning_positions
            / evaluated_positions
        ) * 100

    @staticmethod
    def _calculate_average_return(
        total_position_return: float,
        position_count: int,
    ) -> float:
        """Calculate average percentage return per position."""

        if position_count <= 0:
            return 0.0

        return total_position_return / position_count

    @staticmethod
    def _calculate_total_return(
        total_pnl: float,
        initial_capital: float,
    ) -> float:
        """
        Calculate total portfolio return.

        Return is based on initial capital.
        """

        if initial_capital <= 0:
            return 0.0

        return (total_pnl / initial_capital) * 100

    @staticmethod
    def _determine_status(
        total_return_percent: float,
        win_rate_percent: float,
    ) -> str:
        """Determine descriptive performance status."""

        if total_return_percent >= 10:
            return "STRONG"

        if total_return_percent > 0:
            return "POSITIVE"

        if total_return_percent <= -10:
            return "WEAK"

        if total_return_percent < 0:
            return "NEGATIVE"

        if win_rate_percent >= 50:
            return "STABLE"

        return "NEUTRAL"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert a value to float."""

        try:
            number = float(value)

            if number != number:
                return 0.0

            if number in (float("inf"), float("-inf")):
                return 0.0

            return number

        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _round(value: float) -> float:
        """Standardize numeric precision."""

        return round(float(value), 4)