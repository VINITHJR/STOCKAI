"""
Phase 12 - Layer 4
Portfolio Risk

Handles portfolio-level risk calculations and evaluation.
"""

from typing import Iterable, List, Optional

from .models import (
    PortfolioRisk,
    PositionRisk,
    RiskAction,
    RiskConfig,
    RiskEvaluation,
    RiskLevel,
    RiskStatus,
    RiskType,
)


class PortfolioRiskManager:
    """
    Calculates and evaluates risk for the complete portfolio.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ---------------------------------------------------------
    # Portfolio calculations
    # ---------------------------------------------------------

    @staticmethod
    def calculate_invested_value(
        positions: Iterable[PositionRisk],
    ) -> float:
        """
        Calculate total current market value invested in open positions.
        """
        return sum(
            max(0.0, float(position.market_value))
            for position in positions
        )

    @staticmethod
    def calculate_unrealized_pnl(
        positions: Iterable[PositionRisk],
    ) -> float:
        """
        Calculate total unrealized P&L across open positions.
        """
        return sum(
            float(position.unrealized_pnl)
            for position in positions
        )

    @staticmethod
    def calculate_exposure_percent(
        invested_value: float,
        total_equity: float,
    ) -> float:
        """
        Calculate portfolio exposure as a percentage of total equity.
        """
        if invested_value < 0:
            raise ValueError("Invested value cannot be negative.")

        if total_equity <= 0:
            raise ValueError("Total equity must be greater than zero.")

        return (invested_value / total_equity) * 100.0

    @staticmethod
    def calculate_cash_percent(
        cash: float,
        total_equity: float,
    ) -> float:
        """
        Calculate cash percentage of total equity.
        """
        if cash < 0:
            raise ValueError("Cash cannot be negative.")

        if total_equity <= 0:
            raise ValueError("Total equity must be greater than zero.")

        return (cash / total_equity) * 100.0

    @staticmethod
    def calculate_total_pnl(
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> float:
        """
        Calculate total portfolio P&L.
        """
        return realized_pnl + unrealized_pnl

    @staticmethod
    def calculate_pnl_percent(
        total_pnl: float,
        initial_capital: float,
    ) -> float:
        """
        Calculate portfolio P&L as a percentage of initial capital.
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")

        return (total_pnl / initial_capital) * 100.0

    @staticmethod
    def calculate_drawdown_percent(
        current_equity: float,
        peak_equity: float,
    ) -> float:
        """
        Calculate current portfolio drawdown from peak equity.

        Returns 0 when current equity is at or above the peak.
        """
        if current_equity < 0:
            raise ValueError("Current equity cannot be negative.")

        if peak_equity <= 0:
            raise ValueError("Peak equity must be greater than zero.")

        if current_equity >= peak_equity:
            return 0.0

        return ((peak_equity - current_equity) / peak_equity) * 100.0

    @staticmethod
    def calculate_daily_loss_percent(
        daily_loss: float,
        starting_day_equity: float,
    ) -> float:
        """
        Calculate daily loss percentage.

        A positive daily_loss represents a loss amount.
        """
        if daily_loss < 0:
            raise ValueError("Daily loss cannot be negative.")

        if starting_day_equity <= 0:
            raise ValueError(
                "Starting day equity must be greater than zero."
            )

        return (daily_loss / starting_day_equity) * 100.0

    # ---------------------------------------------------------
    # Build portfolio risk
    # ---------------------------------------------------------

    def build_portfolio_risk(
        self,
        total_equity: float,
        cash: float,
        positions: Iterable[PositionRisk],
        realized_pnl: float = 0.0,
        peak_equity: Optional[float] = None,
        daily_loss: float = 0.0,
    ) -> PortfolioRisk:
        """
        Build a complete PortfolioRisk object.
        """

        if total_equity <= 0:
            raise ValueError("Total equity must be greater than zero.")

        if cash < 0:
            raise ValueError("Cash cannot be negative.")

        if cash > total_equity:
            raise ValueError("Cash cannot exceed total equity.")

        if realized_pnl is None:
            realized_pnl = 0.0

        if daily_loss < 0:
            raise ValueError("Daily loss cannot be negative.")

        position_list: List[PositionRisk] = list(positions)

        invested_value = self.calculate_invested_value(
            position_list
        )

        exposure_percent = self.calculate_exposure_percent(
            invested_value,
            total_equity,
        )

        cash_percent = self.calculate_cash_percent(
            cash,
            total_equity,
        )

        unrealized_pnl = self.calculate_unrealized_pnl(
            position_list
        )

        total_pnl = self.calculate_total_pnl(
            realized_pnl,
            unrealized_pnl,
        )

        if peak_equity is None:
            peak_equity = max(
                self.config.initial_capital,
                total_equity,
            )

        drawdown_percent = self.calculate_drawdown_percent(
            total_equity,
            peak_equity,
        )

        portfolio = PortfolioRisk(
            total_equity=total_equity,
            cash=cash,
            invested_value=invested_value,
            exposure_percent=exposure_percent,
            cash_percent=cash_percent,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            peak_equity=peak_equity,
            drawdown_percent=drawdown_percent,
            daily_loss=daily_loss,
            open_positions=len(position_list),
        )

        return portfolio

    # ---------------------------------------------------------
    # Portfolio exposure evaluation
    # ---------------------------------------------------------

    def evaluate_exposure(
        self,
        portfolio: PortfolioRisk,
    ) -> RiskEvaluation:
        """
        Evaluate total portfolio exposure.
        """

        exposure_percent = portfolio.exposure_percent
        limit = self.config.max_portfolio_exposure_percent

        if exposure_percent > limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.HIGH,
                action=RiskAction.BLOCK,
                risk_type=RiskType.PORTFOLIO_EXPOSURE,
                message=(
                    f"Portfolio exposure {exposure_percent:.2f}% "
                    f"exceeds maximum allowed {limit:.2f}%."
                ),
                current_value=exposure_percent,
                limit_value=limit,
                utilization_percent=(
                    exposure_percent / limit * 100.0
                    if limit > 0
                    else 0.0
                ),
            )

        warning_threshold = limit * 0.80

        if exposure_percent >= warning_threshold:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.PORTFOLIO_EXPOSURE,
                message=(
                    f"Portfolio exposure {exposure_percent:.2f}% "
                    f"is approaching maximum allowed {limit:.2f}%."
                ),
                current_value=exposure_percent,
                limit_value=limit,
                utilization_percent=(
                    exposure_percent / limit * 100.0
                    if limit > 0
                    else 0.0
                ),
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.PORTFOLIO_EXPOSURE,
            message=(
                f"Portfolio exposure {exposure_percent:.2f}% "
                f"is within the allowed limit of {limit:.2f}%."
            ),
            current_value=exposure_percent,
            limit_value=limit,
            utilization_percent=(
                exposure_percent / limit * 100.0
                if limit > 0
                else 0.0
            ),
        )

    # ---------------------------------------------------------
    # Drawdown evaluation
    # ---------------------------------------------------------

    def evaluate_drawdown(
        self,
        portfolio: PortfolioRisk,
    ) -> RiskEvaluation:
        """
        Evaluate portfolio drawdown against configured limits.
        """

        drawdown = portfolio.drawdown_percent

        max_limit = self.config.max_drawdown_percent
        warning_limit = self.config.warning_drawdown_percent

        if drawdown >= max_limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.CRITICAL,
                action=RiskAction.BLOCK,
                risk_type=RiskType.DRAWDOWN,
                message=(
                    f"Portfolio drawdown {drawdown:.2f}% "
                    f"has reached the maximum limit of "
                    f"{max_limit:.2f}%."
                ),
                current_value=drawdown,
                limit_value=max_limit,
                utilization_percent=(
                    drawdown / max_limit * 100.0
                    if max_limit > 0
                    else 0.0
                ),
            )

        if drawdown >= warning_limit:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.HIGH,
                action=RiskAction.WARN,
                risk_type=RiskType.DRAWDOWN,
                message=(
                    f"Portfolio drawdown {drawdown:.2f}% "
                    f"is approaching the maximum limit of "
                    f"{max_limit:.2f}%."
                ),
                current_value=drawdown,
                limit_value=max_limit,
                utilization_percent=(
                    drawdown / max_limit * 100.0
                    if max_limit > 0
                    else 0.0
                ),
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.DRAWDOWN,
            message=(
                f"Portfolio drawdown {drawdown:.2f}% "
                f"is within the allowed limit."
            ),
            current_value=drawdown,
            limit_value=max_limit,
            utilization_percent=(
                drawdown / max_limit * 100.0
                if max_limit > 0
                else 0.0
            ),
        )

    # ---------------------------------------------------------
    # Daily loss evaluation
    # ---------------------------------------------------------

    def evaluate_daily_loss(
        self,
        portfolio: PortfolioRisk,
        starting_day_equity: float,
    ) -> RiskEvaluation:
        """
        Evaluate daily portfolio loss.
        """

        daily_loss_percent = self.calculate_daily_loss_percent(
            portfolio.daily_loss,
            starting_day_equity,
        )

        limit = self.config.max_daily_loss_percent

        if daily_loss_percent >= limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.CRITICAL,
                action=RiskAction.BLOCK,
                risk_type=RiskType.LOSS_LIMIT,
                message=(
                    f"Daily loss {daily_loss_percent:.2f}% "
                    f"has reached the maximum limit of {limit:.2f}%."
                ),
                current_value=daily_loss_percent,
                limit_value=limit,
                utilization_percent=(
                    daily_loss_percent / limit * 100.0
                    if limit > 0
                    else 0.0
                ),
            )

        warning_threshold = limit * 0.80

        if daily_loss_percent >= warning_threshold:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.HIGH,
                action=RiskAction.WARN,
                risk_type=RiskType.LOSS_LIMIT,
                message=(
                    f"Daily loss {daily_loss_percent:.2f}% "
                    f"is approaching the maximum limit of {limit:.2f}%."
                ),
                current_value=daily_loss_percent,
                limit_value=limit,
                utilization_percent=(
                    daily_loss_percent / limit * 100.0
                    if limit > 0
                    else 0.0
                ),
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.LOSS_LIMIT,
            message=(
                f"Daily loss {daily_loss_percent:.2f}% "
                f"is within the allowed limit."
            ),
            current_value=daily_loss_percent,
            limit_value=limit,
            utilization_percent=(
                daily_loss_percent / limit * 100.0
                if limit > 0
                else 0.0
            ),
        )

    # ---------------------------------------------------------
    # Complete portfolio analysis
    # ---------------------------------------------------------

    def analyze_portfolio(
        self,
        total_equity: float,
        cash: float,
        positions: Iterable[PositionRisk],
        realized_pnl: float = 0.0,
        peak_equity: Optional[float] = None,
        daily_loss: float = 0.0,
        starting_day_equity: Optional[float] = None,
    ) -> dict:
        """
        Build and evaluate complete portfolio risk.
        """

        portfolio = self.build_portfolio_risk(
            total_equity=total_equity,
            cash=cash,
            positions=positions,
            realized_pnl=realized_pnl,
            peak_equity=peak_equity,
            daily_loss=daily_loss,
        )

        exposure_evaluation = self.evaluate_exposure(
            portfolio
        )

        drawdown_evaluation = self.evaluate_drawdown(
            portfolio
        )

        if starting_day_equity is None:
            starting_day_equity = total_equity

        daily_loss_evaluation = self.evaluate_daily_loss(
            portfolio,
            starting_day_equity,
        )

        evaluations = [
            exposure_evaluation,
            drawdown_evaluation,
            daily_loss_evaluation,
        ]

        blocked = next(
            (
                evaluation
                for evaluation in evaluations
                if evaluation.status == RiskStatus.BLOCKED
            ),
            None,
        )

        if blocked is not None:
            overall_status = RiskStatus.BLOCKED
            overall_level = RiskLevel.CRITICAL
        elif any(
            evaluation.status == RiskStatus.WARNING
            for evaluation in evaluations
        ):
            overall_status = RiskStatus.WARNING
            overall_level = RiskLevel.HIGH
        else:
            overall_status = RiskStatus.SAFE
            overall_level = RiskLevel.LOW

        portfolio.risk_status = overall_status
        portfolio.risk_level = overall_level

        return {
            "portfolio": portfolio,
            "exposure_evaluation": exposure_evaluation,
            "drawdown_evaluation": drawdown_evaluation,
            "daily_loss_evaluation": daily_loss_evaluation,
        }


# Backward-compatible alias
PortfolioRiskEngine = PortfolioRiskManager