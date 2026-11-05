"""
Phase 12 - Layer 5
Risk Limits

Centralizes validation of configured risk limits.
"""

from typing import Optional

from .models import (
    RiskAction,
    RiskConfig,
    RiskEvaluation,
    RiskLevel,
    RiskStatus,
    RiskType,
)


class RiskLimitManager:
    """
    Centralized risk-limit validation.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ---------------------------------------------------------
    # Position-size limit
    # ---------------------------------------------------------

    def check_position_size(
        self,
        position_value: float,
        portfolio_equity: float,
        instrument_key: Optional[str] = None,
    ) -> RiskEvaluation:
        """
        Validate a position against maximum position size.
        """

        if position_value < 0:
            raise ValueError("Position value cannot be negative.")

        if portfolio_equity <= 0:
            raise ValueError("Portfolio equity must be greater than zero.")

        exposure_percent = (
            position_value / portfolio_equity
        ) * 100.0

        limit = self.config.max_position_percent

        if exposure_percent > limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.HIGH,
                action=RiskAction.BLOCK,
                risk_type=RiskType.POSITION_SIZE,
                message=(
                    f"Position size {exposure_percent:.2f}% "
                    f"exceeds maximum allowed {limit:.2f}%."
                ),
                current_value=exposure_percent,
                limit_value=limit,
                utilization_percent=(
                    exposure_percent / limit * 100.0
                    if limit > 0 else 0.0
                ),
                instrument_key=instrument_key,
            )

        warning_threshold = limit * 0.80

        if exposure_percent >= warning_threshold:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.POSITION_SIZE,
                message=(
                    f"Position size {exposure_percent:.2f}% "
                    f"is approaching maximum allowed {limit:.2f}%."
                ),
                current_value=exposure_percent,
                limit_value=limit,
                utilization_percent=(
                    exposure_percent / limit * 100.0
                    if limit > 0 else 0.0
                ),
                instrument_key=instrument_key,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.POSITION_SIZE,
            message=(
                f"Position size {exposure_percent:.2f}% "
                f"is within the allowed limit of {limit:.2f}%."
            ),
            current_value=exposure_percent,
            limit_value=limit,
            utilization_percent=(
                exposure_percent / limit * 100.0
                if limit > 0 else 0.0
            ),
            instrument_key=instrument_key,
        )

    # ---------------------------------------------------------
    # Portfolio exposure limit
    # ---------------------------------------------------------

    def check_portfolio_exposure(
        self,
        invested_value: float,
        portfolio_equity: float,
    ) -> RiskEvaluation:
        """
        Validate total portfolio exposure.
        """

        if invested_value < 0:
            raise ValueError("Invested value cannot be negative.")

        if portfolio_equity <= 0:
            raise ValueError("Portfolio equity must be greater than zero.")

        exposure_percent = (
            invested_value / portfolio_equity
        ) * 100.0

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
                    if limit > 0 else 0.0
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
                    if limit > 0 else 0.0
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
                if limit > 0 else 0.0
            ),
        )

    # ---------------------------------------------------------
    # Drawdown limit
    # ---------------------------------------------------------

    def check_drawdown(
        self,
        drawdown_percent: float,
    ) -> RiskEvaluation:
        """
        Validate portfolio drawdown.
        """

        if drawdown_percent < 0:
            raise ValueError("Drawdown cannot be negative.")

        warning_limit = self.config.warning_drawdown_percent
        max_limit = self.config.max_drawdown_percent

        if drawdown_percent >= max_limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.CRITICAL,
                action=RiskAction.BLOCK,
                risk_type=RiskType.DRAWDOWN,
                message=(
                    f"Drawdown {drawdown_percent:.2f}% "
                    f"has reached the maximum limit of "
                    f"{max_limit:.2f}%."
                ),
                current_value=drawdown_percent,
                limit_value=max_limit,
                utilization_percent=(
                    drawdown_percent / max_limit * 100.0
                    if max_limit > 0 else 0.0
                ),
            )

        if drawdown_percent >= warning_limit:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.HIGH,
                action=RiskAction.WARN,
                risk_type=RiskType.DRAWDOWN,
                message=(
                    f"Drawdown {drawdown_percent:.2f}% "
                    f"is approaching the maximum limit of "
                    f"{max_limit:.2f}%."
                ),
                current_value=drawdown_percent,
                limit_value=max_limit,
                utilization_percent=(
                    drawdown_percent / max_limit * 100.0
                    if max_limit > 0 else 0.0
                ),
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.DRAWDOWN,
            message=(
                f"Drawdown {drawdown_percent:.2f}% "
                f"is within the allowed limit."
            ),
            current_value=drawdown_percent,
            limit_value=max_limit,
            utilization_percent=(
                drawdown_percent / max_limit * 100.0
                if max_limit > 0 else 0.0
            ),
        )

    # ---------------------------------------------------------
    # Daily-loss limit
    # ---------------------------------------------------------

    def check_daily_loss(
        self,
        daily_loss_percent: float,
    ) -> RiskEvaluation:
        """
        Validate daily portfolio loss.
        """

        if daily_loss_percent < 0:
            raise ValueError("Daily loss percentage cannot be negative.")

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
                    if limit > 0 else 0.0
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
                    if limit > 0 else 0.0
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
                if limit > 0 else 0.0
            ),
        )

    # ---------------------------------------------------------
    # Capital availability
    # ---------------------------------------------------------

    def check_capital(
        self,
        required_capital: float,
        available_capital: float,
    ) -> RiskEvaluation:
        """
        Validate whether enough capital is available.
        """

        if required_capital < 0:
            raise ValueError("Required capital cannot be negative.")

        if available_capital < 0:
            raise ValueError("Available capital cannot be negative.")

        if required_capital > available_capital:
            utilization = (
                required_capital / available_capital * 100.0
                if available_capital > 0 else 100.0
            )

            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.CRITICAL,
                action=RiskAction.BLOCK,
                risk_type=RiskType.CAPITAL,
                message=(
                    f"Required capital ₹{required_capital:.2f} "
                    f"exceeds available capital "
                    f"₹{available_capital:.2f}."
                ),
                current_value=required_capital,
                limit_value=available_capital,
                utilization_percent=utilization,
            )

        utilization = (
            required_capital / available_capital * 100.0
            if available_capital > 0 else 0.0
        )

        if utilization >= 80.0:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.CAPITAL,
                message=(
                    f"Capital utilization {utilization:.2f}% "
                    f"is approaching available capital."
                ),
                current_value=required_capital,
                limit_value=available_capital,
                utilization_percent=utilization,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.CAPITAL,
            message=(
                f"Required capital ₹{required_capital:.2f} "
                f"is within available capital."
            ),
            current_value=required_capital,
            limit_value=available_capital,
            utilization_percent=utilization,
        )

    # ---------------------------------------------------------
    # Concentration limit
    # ---------------------------------------------------------

    def check_concentration(
        self,
        position_value: float,
        portfolio_equity: float,
        instrument_key: Optional[str] = None,
    ) -> RiskEvaluation:
        """
        Validate concentration of one instrument within the portfolio.
        """

        if position_value < 0:
            raise ValueError("Position value cannot be negative.")

        if portfolio_equity <= 0:
            raise ValueError("Portfolio equity must be greater than zero.")

        concentration_percent = (
            position_value / portfolio_equity
        ) * 100.0

        limit = self.config.max_concentration_percent

        if concentration_percent > limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.HIGH,
                action=RiskAction.BLOCK,
                risk_type=RiskType.CONCENTRATION,
                message=(
                    f"Instrument concentration {concentration_percent:.2f}% "
                    f"exceeds maximum allowed {limit:.2f}%."
                ),
                current_value=concentration_percent,
                limit_value=limit,
                utilization_percent=(
                    concentration_percent / limit * 100.0
                    if limit > 0 else 0.0
                ),
                instrument_key=instrument_key,
            )

        warning_threshold = limit * 0.80

        if concentration_percent >= warning_threshold:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.CONCENTRATION,
                message=(
                    f"Instrument concentration {concentration_percent:.2f}% "
                    f"is approaching maximum allowed {limit:.2f}%."
                ),
                current_value=concentration_percent,
                limit_value=limit,
                utilization_percent=(
                    concentration_percent / limit * 100.0
                    if limit > 0 else 0.0
                ),
                instrument_key=instrument_key,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.CONCENTRATION,
            message=(
                f"Instrument concentration {concentration_percent:.2f}% "
                f"is within the allowed limit of {limit:.2f}%."
            ),
            current_value=concentration_percent,
            limit_value=limit,
            utilization_percent=(
                concentration_percent / limit * 100.0
                if limit > 0 else 0.0
            ),
            instrument_key=instrument_key,
        )

    # ---------------------------------------------------------
    # Combined limit validation
    # ---------------------------------------------------------

    def validate_trade(
        self,
        position_value: float,
        portfolio_equity: float,
        available_capital: float,
        current_invested_value: float = 0.0,
        drawdown_percent: float = 0.0,
        daily_loss_percent: float = 0.0,
        instrument_key: Optional[str] = None,
    ) -> dict:
        """
        Run all major risk-limit checks for a proposed trade.
        """

        if current_invested_value < 0:
            raise ValueError(
                "Current invested value cannot be negative."
            )

        evaluations = [
            self.check_capital(
                position_value,
                available_capital,
            ),
            self.check_position_size(
                position_value,
                portfolio_equity,
                instrument_key,
            ),
            self.check_portfolio_exposure(
                current_invested_value + position_value,
                portfolio_equity,
            ),
            self.check_drawdown(
                drawdown_percent,
            ),
            self.check_daily_loss(
                daily_loss_percent,
            ),
            self.check_concentration(
                position_value,
                portfolio_equity,
                instrument_key,
            ),
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
            overall = blocked
        elif any(
            evaluation.status == RiskStatus.WARNING
            for evaluation in evaluations
        ):
            overall = next(
                evaluation
                for evaluation in evaluations
                if evaluation.status == RiskStatus.WARNING
            )
        else:
            overall = RiskEvaluation(
                status=RiskStatus.SAFE,
                level=RiskLevel.LOW,
                action=RiskAction.ALLOW,
                risk_type=RiskType.POSITION_SIZE,
                message="All configured risk limits are satisfied.",
                current_value=0.0,
                limit_value=0.0,
                utilization_percent=0.0,
                instrument_key=instrument_key,
            )

        return {
            "allowed": overall.action != RiskAction.BLOCK,
            "overall": overall,
            "evaluations": evaluations,
        }


# Backward-compatible alias
RiskLimits = RiskLimitManager