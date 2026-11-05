"""
Phase 12 - Layer 6
Risk Metrics

Calculates higher-level risk and performance metrics from
portfolio and position data.

This layer does not modify positions or execute trades.
"""

from typing import Iterable, List, Optional

from .models import (
    PortfolioRisk,
    PositionRisk,
    RiskConfig,
)


class RiskMetricsManager:
    """
    Calculates portfolio and position-level risk metrics.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ---------------------------------------------------------
    # Portfolio return metrics
    # ---------------------------------------------------------

    @staticmethod
    def calculate_total_return_percent(
        total_pnl: float,
        initial_capital: float,
    ) -> float:
        """
        Calculate total portfolio return percentage.
        """

        if initial_capital <= 0:
            raise ValueError(
                "Initial capital must be greater than zero."
            )

        return (total_pnl / initial_capital) * 100.0

    @staticmethod
    def calculate_realized_return_percent(
        realized_pnl: float,
        initial_capital: float,
    ) -> float:
        """
        Calculate realized return percentage.
        """

        if initial_capital <= 0:
            raise ValueError(
                "Initial capital must be greater than zero."
            )

        return (realized_pnl / initial_capital) * 100.0

    @staticmethod
    def calculate_unrealized_return_percent(
        unrealized_pnl: float,
        initial_capital: float,
    ) -> float:
        """
        Calculate unrealized return percentage.
        """

        if initial_capital <= 0:
            raise ValueError(
                "Initial capital must be greater than zero."
            )

        return (unrealized_pnl / initial_capital) * 100.0

    # ---------------------------------------------------------
    # Drawdown metrics
    # ---------------------------------------------------------

    @staticmethod
    def calculate_current_drawdown_percent(
        current_equity: float,
        peak_equity: float,
    ) -> float:
        """
        Calculate current drawdown from peak equity.
        """

        if current_equity < 0:
            raise ValueError(
                "Current equity cannot be negative."
            )

        if peak_equity <= 0:
            raise ValueError(
                "Peak equity must be greater than zero."
            )

        if current_equity >= peak_equity:
            return 0.0

        return (
            (peak_equity - current_equity)
            / peak_equity
        ) * 100.0

    @staticmethod
    def calculate_drawdown_amount(
        current_equity: float,
        peak_equity: float,
    ) -> float:
        """
        Calculate drawdown amount in currency.
        """

        if current_equity < 0:
            raise ValueError(
                "Current equity cannot be negative."
            )

        if peak_equity <= 0:
            raise ValueError(
                "Peak equity must be greater than zero."
            )

        return max(
            0.0,
            peak_equity - current_equity,
        )

    # ---------------------------------------------------------
    # Exposure metrics
    # ---------------------------------------------------------

    @staticmethod
    def calculate_exposure_percent(
        invested_value: float,
        total_equity: float,
    ) -> float:
        """
        Calculate portfolio exposure percentage.
        """

        if invested_value < 0:
            raise ValueError(
                "Invested value cannot be negative."
            )

        if total_equity <= 0:
            raise ValueError(
                "Total equity must be greater than zero."
            )

        return (
            invested_value / total_equity
        ) * 100.0

    @staticmethod
    def calculate_cash_percent(
        cash: float,
        total_equity: float,
    ) -> float:
        """
        Calculate cash percentage.
        """

        if cash < 0:
            raise ValueError(
                "Cash cannot be negative."
            )

        if total_equity <= 0:
            raise ValueError(
                "Total equity must be greater than zero."
            )

        return (
            cash / total_equity
        ) * 100.0

    # ---------------------------------------------------------
    # Daily loss
    # ---------------------------------------------------------

    @staticmethod
    def calculate_daily_loss_percent(
        daily_loss: float,
        starting_day_equity: float,
    ) -> float:
        """
        Calculate daily loss percentage.
        """

        if daily_loss < 0:
            raise ValueError(
                "Daily loss cannot be negative."
            )

        if starting_day_equity <= 0:
            raise ValueError(
                "Starting day equity must be greater than zero."
            )

        return (
            daily_loss / starting_day_equity
        ) * 100.0

    # ---------------------------------------------------------
    # Position statistics
    # ---------------------------------------------------------

    @staticmethod
    def count_positions(
        positions: Iterable[PositionRisk],
    ) -> int:
        """
        Count open positions.
        """

        return len(list(positions))

    @staticmethod
    def count_winning_positions(
        positions: Iterable[PositionRisk],
    ) -> int:
        """
        Count positions with positive unrealized P&L.
        """

        return sum(
            1
            for position in positions
            if position.unrealized_pnl > 0
        )

    @staticmethod
    def count_losing_positions(
        positions: Iterable[PositionRisk],
    ) -> int:
        """
        Count positions with negative unrealized P&L.
        """

        return sum(
            1
            for position in positions
            if position.unrealized_pnl < 0
        )

    @staticmethod
    def count_break_even_positions(
        positions: Iterable[PositionRisk],
    ) -> int:
        """
        Count positions with zero unrealized P&L.
        """

        return sum(
            1
            for position in positions
            if position.unrealized_pnl == 0
        )

    @staticmethod
    def calculate_win_rate(
        positions: Iterable[PositionRisk],
    ) -> float:
        """
        Calculate win rate based on open positions.

        Break-even positions are included in the total count
        but are not counted as wins.
        """

        position_list: List[PositionRisk] = list(positions)

        if not position_list:
            return 0.0

        winning_positions = sum(
            1
            for position in position_list
            if position.unrealized_pnl > 0
        )

        return (
            winning_positions
            / len(position_list)
        ) * 100.0

    # ---------------------------------------------------------
    # Average gain/loss
    # ---------------------------------------------------------

    @staticmethod
    def calculate_average_gain(
        positions: Iterable[PositionRisk],
    ) -> float:
        """
        Calculate average positive unrealized P&L.
        """

        gains = [
            position.unrealized_pnl
            for position in positions
            if position.unrealized_pnl > 0
        ]

        if not gains:
            return 0.0

        return sum(gains) / len(gains)

    @staticmethod
    def calculate_average_loss(
        positions: Iterable[PositionRisk],
    ) -> float:
        """
        Calculate average absolute loss.

        Returned value is positive.
        """

        losses = [
            abs(position.unrealized_pnl)
            for position in positions
            if position.unrealized_pnl < 0
        ]

        if not losses:
            return 0.0

        return sum(losses) / len(losses)

    @staticmethod
    def calculate_profit_factor(
        positions: Iterable[PositionRisk],
    ) -> float:
        """
        Calculate profit factor:

        total gains / total losses

        Returns 0 when there are no gains.
        Returns infinity when there are gains but no losses.
        """

        total_gains = sum(
            position.unrealized_pnl
            for position in positions
            if position.unrealized_pnl > 0
        )

        total_losses = sum(
            abs(position.unrealized_pnl)
            for position in positions
            if position.unrealized_pnl < 0
        )

        if total_losses == 0:
            if total_gains > 0:
                return float("inf")

            return 0.0

        return total_gains / total_losses

    # ---------------------------------------------------------
    # Position concentration
    # ---------------------------------------------------------

    @staticmethod
    def calculate_largest_position_percent(
        positions: Iterable[PositionRisk],
        total_equity: float,
    ) -> float:
        """
        Calculate the largest individual position as a percentage
        of total portfolio equity.
        """

        if total_equity <= 0:
            raise ValueError(
                "Total equity must be greater than zero."
            )

        position_list = list(positions)

        if not position_list:
            return 0.0

        largest_value = max(
            position.market_value
            for position in position_list
        )

        return (
            largest_value / total_equity
        ) * 100.0

    # ---------------------------------------------------------
    # Risk / reward
    # ---------------------------------------------------------

    @staticmethod
    def calculate_position_risk_reward(
        position: PositionRisk,
    ) -> float:
        """
        Calculate position-level risk/reward ratio.

        Risk = entry price - stop loss
        Reward = take profit - entry price

        Returns 0 when required levels are unavailable.
        """

        if position.stop_loss_price is None:
            return 0.0

        if position.take_profit_price is None:
            return 0.0

        risk = (
            position.entry_price
            - position.stop_loss_price
        )

        reward = (
            position.take_profit_price
            - position.entry_price
        )

        if risk <= 0:
            return 0.0

        return reward / risk

    # ---------------------------------------------------------
    # Complete portfolio metrics
    # ---------------------------------------------------------

    def calculate_portfolio_metrics(
        self,
        portfolio: PortfolioRisk,
        positions: Iterable[PositionRisk],
        starting_day_equity: Optional[float] = None,
    ) -> dict:
        """
        Calculate a complete portfolio risk-metrics snapshot.
        """

        position_list = list(positions)

        if starting_day_equity is None:
            starting_day_equity = portfolio.total_equity

        total_return_percent = (
            self.calculate_total_return_percent(
                portfolio.total_pnl,
                self.config.initial_capital,
            )
        )

        realized_return_percent = (
            self.calculate_realized_return_percent(
                portfolio.realized_pnl,
                self.config.initial_capital,
            )
        )

        unrealized_return_percent = (
            self.calculate_unrealized_return_percent(
                portfolio.unrealized_pnl,
                self.config.initial_capital,
            )
        )

        current_drawdown_percent = (
            self.calculate_current_drawdown_percent(
                portfolio.total_equity,
                portfolio.peak_equity,
            )
        )

        drawdown_amount = (
            self.calculate_drawdown_amount(
                portfolio.total_equity,
                portfolio.peak_equity,
            )
        )

        exposure_percent = (
            self.calculate_exposure_percent(
                portfolio.invested_value,
                portfolio.total_equity,
            )
        )

        cash_percent = (
            self.calculate_cash_percent(
                portfolio.cash,
                portfolio.total_equity,
            )
        )

        daily_loss_percent = (
            self.calculate_daily_loss_percent(
                portfolio.daily_loss,
                starting_day_equity,
            )
        )

        position_count = len(position_list)

        winning_positions = (
            self.count_winning_positions(position_list)
        )

        losing_positions = (
            self.count_losing_positions(position_list)
        )

        break_even_positions = (
            self.count_break_even_positions(position_list)
        )

        win_rate = self.calculate_win_rate(
            position_list
        )

        average_gain = self.calculate_average_gain(
            position_list
        )

        average_loss = self.calculate_average_loss(
            position_list
        )

        profit_factor = self.calculate_profit_factor(
            position_list
        )

        largest_position_percent = (
            self.calculate_largest_position_percent(
                position_list,
                portfolio.total_equity,
            )
        )

        position_risk_reward = [
            {
                "instrument_key": position.instrument_key,
                "risk_reward_ratio": (
                    self.calculate_position_risk_reward(
                        position
                    )
                ),
            }
            for position in position_list
        ]

        return {
            "total_return_percent": total_return_percent,
            "realized_return_percent": realized_return_percent,
            "unrealized_return_percent": unrealized_return_percent,
            "current_drawdown_percent": current_drawdown_percent,
            "drawdown_amount": drawdown_amount,
            "exposure_percent": exposure_percent,
            "cash_percent": cash_percent,
            "daily_loss_percent": daily_loss_percent,
            "position_count": position_count,
            "winning_positions": winning_positions,
            "losing_positions": losing_positions,
            "break_even_positions": break_even_positions,
            "win_rate_percent": win_rate,
            "average_gain": average_gain,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "largest_position_percent": largest_position_percent,
            "position_risk_reward": position_risk_reward,
        }


# Backward-compatible alias
RiskMetrics = RiskMetricsManager