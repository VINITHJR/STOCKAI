from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import PaperTrade
from .portfolio import PaperPortfolio


@dataclass
class PaperTradingMetrics:
    """
    Performance metrics for a paper-trading session.
    """

    initial_capital: float

    final_equity: float

    total_pnl: float

    realized_pnl: float

    unrealized_pnl: float

    total_return_percent: float

    total_trades: int

    winning_trades: int

    losing_trades: int

    breakeven_trades: int

    win_rate_percent: float

    average_win: float

    average_loss: float

    profit_factor: Optional[float]

    maximum_drawdown: float

    maximum_drawdown_percent: float


class PaperTradingMetricsCalculator:
    """
    Phase 11 - Layer 6

    Calculates performance statistics for paper trading.

    This layer does not execute orders and does not modify
    the portfolio.
    """

    # ============================================================
    # PORTFOLIO METRICS
    # ============================================================

    def calculate(
        self,
        portfolio: PaperPortfolio,
        trades: Optional[List[PaperTrade]] = None,
    ) -> PaperTradingMetrics:
        """
        Calculate complete paper-trading performance metrics.

        Args:
            portfolio:
                Current virtual portfolio.

            trades:
                Optional list of completed paper trades.

        Returns:
            PaperTradingMetrics
        """

        if trades is None:
            trades = []

        initial_capital = portfolio.initial_capital
        final_equity = portfolio.total_equity

        total_pnl = portfolio.total_pnl
        realized_pnl = portfolio.realized_pnl
        unrealized_pnl = portfolio.unrealized_pnl

        total_return_percent = (
            portfolio.total_return_percent
        )

        total_trades = len(trades)

        winning_trades = sum(
            1 for trade in trades
            if trade.net_pnl > 0
        )

        losing_trades = sum(
            1 for trade in trades
            if trade.net_pnl < 0
        )

        breakeven_trades = sum(
            1 for trade in trades
            if trade.net_pnl == 0
        )

        win_rate_percent = self._calculate_win_rate(
            total_trades,
            winning_trades,
        )

        average_win = self._calculate_average_win(
            trades
        )

        average_loss = self._calculate_average_loss(
            trades
        )

        profit_factor = self._calculate_profit_factor(
            trades
        )

        maximum_drawdown = 0.0
        maximum_drawdown_percent = 0.0

        return PaperTradingMetrics(
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_return_percent=total_return_percent,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate_percent=win_rate_percent,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            maximum_drawdown=maximum_drawdown,
            maximum_drawdown_percent=maximum_drawdown_percent,
        )

    # ============================================================
    # WIN RATE
    # ============================================================

    @staticmethod
    def _calculate_win_rate(
        total_trades: int,
        winning_trades: int,
    ) -> float:
        """
        Calculate winning-trade percentage.
        """

        if total_trades == 0:
            return 0.0

        return (
            winning_trades
            / total_trades
        ) * 100.0

    # ============================================================
    # AVERAGE WIN
    # ============================================================

    @staticmethod
    def _calculate_average_win(
        trades: List[PaperTrade],
    ) -> float:
        """
        Calculate average P&L of winning trades.
        """

        winning_pnls = [
            trade.net_pnl
            for trade in trades
            if trade.net_pnl > 0
        ]

        if not winning_pnls:
            return 0.0

        return sum(winning_pnls) / len(winning_pnls)

    # ============================================================
    # AVERAGE LOSS
    # ============================================================

    @staticmethod
    def _calculate_average_loss(
        trades: List[PaperTrade],
    ) -> float:
        """
        Calculate average absolute loss.

        Returned as a positive value for easier reporting.
        """

        losing_pnls = [
            abs(trade.net_pnl)
            for trade in trades
            if trade.net_pnl < 0
        ]

        if not losing_pnls:
            return 0.0

        return sum(losing_pnls) / len(losing_pnls)

    # ============================================================
    # PROFIT FACTOR
    # ============================================================

    @staticmethod
    def _calculate_profit_factor(
        trades: List[PaperTrade],
    ) -> Optional[float]:
        """
        Profit Factor = Gross Profit / Gross Loss.

        Returns None when there are no losing trades.
        """

        gross_profit = sum(
            trade.net_pnl
            for trade in trades
            if trade.net_pnl > 0
        )

        gross_loss = sum(
            abs(trade.net_pnl)
            for trade in trades
            if trade.net_pnl < 0
        )

        if gross_loss == 0:
            if gross_profit > 0:
                return None

            return 0.0

        return gross_profit / gross_loss

    # ============================================================
    # DRAWDOWN
    # ============================================================

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: List[float],
    ) -> tuple[float, float]:
        """
        Calculate maximum absolute and percentage drawdown.

        Example:

            Equity:
            100000
            105000
            102000
            98000

        Peak = 105000
        Trough = 98000

        Drawdown = 7000
        Drawdown % = 6.666...%
        """

        if not equity_curve:
            return 0.0, 0.0

        peak = equity_curve[0]

        maximum_drawdown = 0.0
        maximum_drawdown_percent = 0.0

        for equity in equity_curve:

            if equity > peak:
                peak = equity

            if peak <= 0:
                continue

            drawdown = peak - equity

            drawdown_percent = (
                drawdown / peak
            ) * 100.0

            if drawdown > maximum_drawdown:
                maximum_drawdown = drawdown

            if (
                drawdown_percent
                > maximum_drawdown_percent
            ):
                maximum_drawdown_percent = (
                    drawdown_percent
                )

        return (
            maximum_drawdown,
            maximum_drawdown_percent,
        )