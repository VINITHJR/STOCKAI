from __future__ import annotations

from typing import Optional

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


class RiskEngine:
    """
    Core Phase 12 risk evaluation engine.

    Layer 2 responsibilities:
    - Validate proposed position size
    - Validate portfolio exposure
    - Validate available capital
    - Evaluate individual positions
    - Return a standardized RiskEvaluation

    This layer does NOT execute trades and does NOT modify
    the Phase 11 paper-trading engine.
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
    ) -> None:
        self.config = config or RiskConfig()

    # ========================================================
    # POSITION SIZE
    # ========================================================

    def evaluate_position_size(
        self,
        position_value: float,
        portfolio_equity: float,
        instrument_key: Optional[str] = None,
    ) -> RiskEvaluation:
        """
        Check whether a proposed position is within the
        configured maximum position-size limit.
        """

        self._validate_positive_value(
            position_value,
            "position_value",
        )

        self._validate_positive_value(
            portfolio_equity,
            "portfolio_equity",
        )

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
                    "Position size exceeds the configured "
                    "maximum position limit."
                ),
                current_value=position_value,
                limit_value=(
                    portfolio_equity * limit / 100.0
                ),
                utilization_percent=exposure_percent,
                instrument_key=instrument_key,
            )

        warning_limit = limit * 0.80

        if exposure_percent >= warning_limit:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.POSITION_SIZE,
                message=(
                    "Position size is approaching the "
                    "configured maximum position limit."
                ),
                current_value=position_value,
                limit_value=(
                    portfolio_equity * limit / 100.0
                ),
                utilization_percent=exposure_percent,
                instrument_key=instrument_key,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.POSITION_SIZE,
            message="Position size is within the configured limit.",
            current_value=position_value,
            limit_value=(
                portfolio_equity * limit / 100.0
            ),
            utilization_percent=exposure_percent,
            instrument_key=instrument_key,
        )

    # ========================================================
    # PORTFOLIO EXPOSURE
    # ========================================================

    def evaluate_portfolio_exposure(
        self,
        invested_value: float,
        portfolio_equity: float,
    ) -> RiskEvaluation:
        """
        Check total portfolio exposure.
        """

        self._validate_non_negative_value(
            invested_value,
            "invested_value",
        )

        self._validate_positive_value(
            portfolio_equity,
            "portfolio_equity",
        )

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
                    "Portfolio exposure exceeds the "
                    "configured maximum exposure limit."
                ),
                current_value=invested_value,
                limit_value=(
                    portfolio_equity * limit / 100.0
                ),
                utilization_percent=exposure_percent,
            )

        warning_limit = limit * 0.80

        if exposure_percent >= warning_limit:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.PORTFOLIO_EXPOSURE,
                message=(
                    "Portfolio exposure is approaching the "
                    "configured maximum exposure limit."
                ),
                current_value=invested_value,
                limit_value=(
                    portfolio_equity * limit / 100.0
                ),
                utilization_percent=exposure_percent,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.PORTFOLIO_EXPOSURE,
            message=(
                "Portfolio exposure is within the configured limit."
            ),
            current_value=invested_value,
            limit_value=(
                portfolio_equity * limit / 100.0
            ),
            utilization_percent=exposure_percent,
        )

    # ========================================================
    # CAPITAL CHECK
    # ========================================================

    def evaluate_capital(
        self,
        required_capital: float,
        available_capital: float,
    ) -> RiskEvaluation:
        """
        Check whether sufficient capital is available.
        """

        self._validate_positive_value(
            required_capital,
            "required_capital",
        )

        self._validate_non_negative_value(
            available_capital,
            "available_capital",
        )

        if required_capital > available_capital:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.CRITICAL,
                action=RiskAction.BLOCK,
                risk_type=RiskType.CAPITAL,
                message=(
                    "Required capital exceeds available capital."
                ),
                current_value=required_capital,
                limit_value=available_capital,
                utilization_percent=(
                    required_capital
                    / available_capital
                    * 100.0
                    if available_capital > 0
                    else 0.0
                ),
            )

        utilization_percent = (
            required_capital / available_capital
        ) * 100.0 if available_capital > 0 else 0.0

        if utilization_percent >= 80.0:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.CAPITAL,
                message=(
                    "Trade uses a high percentage of "
                    "available capital."
                ),
                current_value=required_capital,
                limit_value=available_capital,
                utilization_percent=utilization_percent,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.CAPITAL,
            message="Sufficient capital is available.",
            current_value=required_capital,
            limit_value=available_capital,
            utilization_percent=utilization_percent,
        )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    def evaluate_drawdown(
        self,
        drawdown_percent: float,
    ) -> RiskEvaluation:
        """
        Evaluate current portfolio drawdown.
        """

        self._validate_non_negative_value(
            drawdown_percent,
            "drawdown_percent",
        )

        limit = self.config.max_drawdown_percent
        warning_limit = self.config.warning_drawdown_percent

        if drawdown_percent >= limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.CRITICAL,
                action=RiskAction.BLOCK,
                risk_type=RiskType.DRAWDOWN,
                message=(
                    "Portfolio drawdown has reached or exceeded "
                    "the configured maximum drawdown."
                ),
                current_value=drawdown_percent,
                limit_value=limit,
                utilization_percent=(
                    drawdown_percent / limit
                ) * 100.0,
            )

        if drawdown_percent >= warning_limit:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.HIGH,
                action=RiskAction.WARN,
                risk_type=RiskType.DRAWDOWN,
                message=(
                    "Portfolio drawdown has reached the "
                    "configured warning threshold."
                ),
                current_value=drawdown_percent,
                limit_value=limit,
                utilization_percent=(
                    drawdown_percent / limit
                ) * 100.0,
            )

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.DRAWDOWN,
            message="Portfolio drawdown is within the configured limit.",
            current_value=drawdown_percent,
            limit_value=limit,
            utilization_percent=(
                drawdown_percent / limit
            ) * 100.0,
        )

    # ========================================================
    # POSITION EVALUATION
    # ========================================================

    def evaluate_position(
        self,
        position: PositionRisk,
        portfolio_equity: float,
    ) -> RiskEvaluation:
        """
        Evaluate an existing position.
        """

        if not position.instrument_key:
            raise ValueError(
                "position.instrument_key is required"
            )

        return self.evaluate_position_size(
            position_value=position.market_value,
            portfolio_equity=portfolio_equity,
            instrument_key=position.instrument_key,
        )

    # ========================================================
    # PORTFOLIO EVALUATION
    # ========================================================

    def evaluate_portfolio(
        self,
        portfolio: PortfolioRisk,
    ) -> RiskEvaluation:
        """
        Evaluate overall portfolio exposure.
        """

        return self.evaluate_portfolio_exposure(
            invested_value=portfolio.invested_value,
            portfolio_equity=portfolio.total_equity,
        )

    # ========================================================
    # COMBINED TRADE CHECK
    # ========================================================

    def evaluate_trade(
        self,
        position_value: float,
        portfolio_equity: float,
        available_capital: float,
        current_invested_value: float = 0.0,
        instrument_key: Optional[str] = None,
    ) -> RiskEvaluation:
        """
        Run the main Layer 2 checks for a proposed trade.

        The most restrictive result is returned.
        """

        self._validate_non_negative_value(
            current_invested_value,
            "current_invested_value",
        )

        capital_result = self.evaluate_capital(
            required_capital=position_value,
            available_capital=available_capital,
        )

        if capital_result.status == RiskStatus.BLOCKED:
            return capital_result

        position_result = self.evaluate_position_size(
            position_value=position_value,
            portfolio_equity=portfolio_equity,
            instrument_key=instrument_key,
        )

        if position_result.status == RiskStatus.BLOCKED:
            return position_result

        total_exposure = (
            current_invested_value + position_value
        )

        portfolio_result = self.evaluate_portfolio_exposure(
            invested_value=total_exposure,
            portfolio_equity=portfolio_equity,
        )

        if portfolio_result.status == RiskStatus.BLOCKED:
            return portfolio_result

        results = [
            capital_result,
            position_result,
            portfolio_result,
        ]

        if any(
            result.status == RiskStatus.WARNING
            for result in results
        ):
            warning_result = next(
                result
                for result in results
                if result.status == RiskStatus.WARNING
            )

            return warning_result

        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=None,
            message="Trade passed all Layer 2 risk checks.",
            current_value=position_value,
            limit_value=portfolio_equity,
            utilization_percent=(
                position_value / portfolio_equity * 100.0
            ),
            instrument_key=instrument_key,
        )

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    @staticmethod
    def _validate_positive_value(
        value: float,
        field_name: str,
    ) -> None:
        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than zero"
            )

    @staticmethod
    def _validate_non_negative_value(
        value: float,
        field_name: str,
    ) -> None:
        if value < 0:
            raise ValueError(
                f"{field_name} cannot be negative"
            )