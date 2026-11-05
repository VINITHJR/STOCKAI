"""
Phase 12 - Layer 3
Position Risk

Handles position-level risk calculations:
- Market value
- Exposure percentage
- Unrealized P&L
- Unrealized P&L percentage
- Stop-loss price
- Take-profit price
- Position-level risk evaluation
"""

from typing import Optional

from .models import (
    PositionRisk,
    RiskAction,
    RiskConfig,
    RiskEvaluation,
    RiskLevel,
    RiskStatus,
    RiskType,
)


class PositionRiskManager:
    """
    Calculates and evaluates risk for an individual position.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ---------------------------------------------------------
    # Basic calculations
    # ---------------------------------------------------------

    @staticmethod
    def calculate_market_value(
        quantity: float,
        current_price: float,
    ) -> float:
        """
        Calculate current market value of a position.
        """
        if quantity < 0:
            raise ValueError("Quantity cannot be negative.")

        if current_price < 0:
            raise ValueError("Current price cannot be negative.")

        return quantity * current_price

    @staticmethod
    def calculate_invested_value(
        quantity: float,
        entry_price: float,
    ) -> float:
        """
        Calculate original invested value.
        """
        if quantity < 0:
            raise ValueError("Quantity cannot be negative.")

        if entry_price < 0:
            raise ValueError("Entry price cannot be negative.")

        return quantity * entry_price

    @staticmethod
    def calculate_unrealized_pnl(
        quantity: float,
        entry_price: float,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized profit/loss.
        """
        if quantity < 0:
            raise ValueError("Quantity cannot be negative.")

        if entry_price < 0 or current_price < 0:
            raise ValueError("Prices cannot be negative.")

        return (current_price - entry_price) * quantity

    @staticmethod
    def calculate_unrealized_pnl_percent(
        entry_price: float,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized P&L percentage.
        """
        if entry_price <= 0:
            raise ValueError("Entry price must be greater than zero.")

        return ((current_price - entry_price) / entry_price) * 100.0

    # ---------------------------------------------------------
    # Exposure
    # ---------------------------------------------------------

    @staticmethod
    def calculate_exposure_percent(
        market_value: float,
        portfolio_equity: float,
    ) -> float:
        """
        Calculate position exposure as a percentage of portfolio equity.
        """
        if market_value < 0:
            raise ValueError("Market value cannot be negative.")

        if portfolio_equity <= 0:
            raise ValueError("Portfolio equity must be greater than zero.")

        return (market_value / portfolio_equity) * 100.0

    # ---------------------------------------------------------
    # Stop-loss / Take-profit
    # ---------------------------------------------------------

    def calculate_stop_loss_price(
        self,
        entry_price: float,
        stop_loss_percent: Optional[float] = None,
    ) -> float:
        """
        Calculate stop-loss price for a long position.
        """
        if entry_price <= 0:
            raise ValueError("Entry price must be greater than zero.")

        percent = (
            self.config.default_stop_loss_percent
            if stop_loss_percent is None
            else stop_loss_percent
        )

        if percent < 0 or percent >= 100:
            raise ValueError(
                "Stop-loss percentage must be between 0 and 100."
            )

        return entry_price * (1.0 - percent / 100.0)

    def calculate_take_profit_price(
        self,
        entry_price: float,
        take_profit_percent: Optional[float] = None,
    ) -> float:
        """
        Calculate take-profit price for a long position.
        """
        if entry_price <= 0:
            raise ValueError("Entry price must be greater than zero.")

        percent = (
            self.config.default_take_profit_percent
            if take_profit_percent is None
            else take_profit_percent
        )

        if percent < 0:
            raise ValueError("Take-profit percentage cannot be negative.")

        return entry_price * (1.0 + percent / 100.0)

    # ---------------------------------------------------------
    # Build / update PositionRisk
    # ---------------------------------------------------------

    def build_position_risk(
        self,
        instrument_key: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        portfolio_equity: float,
        stop_loss_percent: Optional[float] = None,
        take_profit_percent: Optional[float] = None,
    ) -> PositionRisk:
        """
        Build a complete PositionRisk object from raw position data.
        """

        if not instrument_key:
            raise ValueError("Instrument key is required.")

        if quantity <= 0:
            raise ValueError("Quantity must be greater than zero.")

        if entry_price <= 0:
            raise ValueError("Entry price must be greater than zero.")

        if current_price < 0:
            raise ValueError("Current price cannot be negative.")

        if portfolio_equity <= 0:
            raise ValueError("Portfolio equity must be greater than zero.")

        market_value = self.calculate_market_value(
            quantity,
            current_price,
        )

        invested_value = self.calculate_invested_value(
            quantity,
            entry_price,
        )

        unrealized_pnl = self.calculate_unrealized_pnl(
            quantity,
            entry_price,
            current_price,
        )

        unrealized_pnl_percent = self.calculate_unrealized_pnl_percent(
            entry_price,
            current_price,
        )

        exposure_percent = self.calculate_exposure_percent(
            market_value,
            portfolio_equity,
        )

        stop_loss_price = self.calculate_stop_loss_price(
            entry_price,
            stop_loss_percent,
        )

        take_profit_price = self.calculate_take_profit_price(
            entry_price,
            take_profit_percent,
        )

        position = PositionRisk(
            instrument_key=instrument_key,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            market_value=market_value,
            exposure_percent=exposure_percent,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        return position

    # ---------------------------------------------------------
    # Position evaluation
    # ---------------------------------------------------------

    def evaluate_position(
        self,
        position: PositionRisk,
        portfolio_equity: float,
    ) -> RiskEvaluation:
        """
        Evaluate the position against the configured maximum
        position-size limit.
        """

        if portfolio_equity <= 0:
            raise ValueError("Portfolio equity must be greater than zero.")

        exposure_percent = self.calculate_exposure_percent(
            position.market_value,
            portfolio_equity,
        )

        limit = self.config.max_position_percent

        # Above maximum position size
        if exposure_percent > limit:
            return RiskEvaluation(
                status=RiskStatus.BLOCKED,
                level=RiskLevel.HIGH,
                action=RiskAction.BLOCK,
                risk_type=RiskType.POSITION_SIZE,
                message=(
                    f"Position exposure {exposure_percent:.2f}% "
                    f"exceeds maximum allowed {limit:.2f}%."
                ),
                current_value=exposure_percent,
                limit_value=limit,
                utilization_percent=(
                    exposure_percent / limit * 100.0
                    if limit > 0
                    else 0.0
                ),
                instrument_key=position.instrument_key,
            )

        # At or above 80% of maximum
        warning_threshold = limit * 0.80

        if exposure_percent >= warning_threshold:
            return RiskEvaluation(
                status=RiskStatus.WARNING,
                level=RiskLevel.MEDIUM,
                action=RiskAction.WARN,
                risk_type=RiskType.POSITION_SIZE,
                message=(
                    f"Position exposure {exposure_percent:.2f}% "
                    f"is approaching maximum allowed {limit:.2f}%."
                ),
                current_value=exposure_percent,
                limit_value=limit,
                utilization_percent=(
                    exposure_percent / limit * 100.0
                    if limit > 0
                    else 0.0
                ),
                instrument_key=position.instrument_key,
            )

        # Safe
        return RiskEvaluation(
            status=RiskStatus.SAFE,
            level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
            risk_type=RiskType.POSITION_SIZE,
            message=(
                f"Position exposure {exposure_percent:.2f}% "
                f"is within the allowed limit of {limit:.2f}%."
            ),
            current_value=exposure_percent,
            limit_value=limit,
            utilization_percent=(
                exposure_percent / limit * 100.0
                if limit > 0
                else 0.0
            ),
            instrument_key=position.instrument_key,
        )

    # ---------------------------------------------------------
    # Stop-loss / take-profit status
    # ---------------------------------------------------------

    @staticmethod
    def check_stop_loss(
        position: PositionRisk,
    ) -> bool:
        """
        Returns True when the current price has reached or crossed
        the stop-loss price.
        """
        if position.stop_loss_price is None:
            return False

        return position.current_price <= position.stop_loss_price

    @staticmethod
    def check_take_profit(
        position: PositionRisk,
    ) -> bool:
        """
        Returns True when the current price has reached or crossed
        the take-profit price.
        """
        if position.take_profit_price is None:
            return False

        return position.current_price >= position.take_profit_price

    # ---------------------------------------------------------
    # Full position analysis
    # ---------------------------------------------------------

    def analyze_position(
        self,
        instrument_key: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        portfolio_equity: float,
        stop_loss_percent: Optional[float] = None,
        take_profit_percent: Optional[float] = None,
    ) -> dict:
        """
        Build and evaluate a complete position.
        """

        position = self.build_position_risk(
            instrument_key=instrument_key,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            portfolio_equity=portfolio_equity,
            stop_loss_percent=stop_loss_percent,
            take_profit_percent=take_profit_percent,
        )

        evaluation = self.evaluate_position(
            position,
            portfolio_equity,
        )

        stop_loss_hit = self.check_stop_loss(position)
        take_profit_hit = self.check_take_profit(position)

        return {
            "position": position,
            "evaluation": evaluation,
            "stop_loss_hit": stop_loss_hit,
            "take_profit_hit": take_profit_hit,
        }


# Backward-compatible alias
PositionRiskEngine = PositionRiskManager