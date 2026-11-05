from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ============================================================
# ENUMS
# ============================================================


class RiskStatus(str, Enum):
    """Overall status of a risk evaluation."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


class RiskLevel(str, Enum):
    """Severity level of a detected risk."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskType(str, Enum):
    """Types of risk monitored by Phase 12."""

    POSITION_SIZE = "POSITION_SIZE"
    POSITION_EXPOSURE = "POSITION_EXPOSURE"
    PORTFOLIO_EXPOSURE = "PORTFOLIO_EXPOSURE"
    DRAWDOWN = "DRAWDOWN"
    LOSS_LIMIT = "LOSS_LIMIT"
    STOP_LOSS = "STOP_LOSS"
    CONCENTRATION = "CONCENTRATION"
    CAPITAL = "CAPITAL"


class RiskAction(str, Enum):
    """Action suggested by the risk system."""

    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"


# ============================================================
# RISK CONFIGURATION
# ============================================================


@dataclass
class RiskConfig:
    """
    Global risk-management configuration.

    This class only stores configuration.
    Risk calculations are handled by later Phase 12 layers.
    """

    initial_capital: float = 100000.0

    max_position_percent: float = 20.0
    max_portfolio_exposure_percent: float = 80.0

    max_drawdown_percent: float = 10.0
    warning_drawdown_percent: float = 5.0

    max_daily_loss_percent: float = 3.0

    default_stop_loss_percent: float = 5.0
    default_take_profit_percent: float = 10.0

    max_concentration_percent: float = 25.0

    currency: str = "INR"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(
                "initial_capital must be greater than zero"
            )

        if not 0 < self.max_position_percent <= 100:
            raise ValueError(
                "max_position_percent must be between 0 and 100"
            )

        if not 0 < self.max_portfolio_exposure_percent <= 100:
            raise ValueError(
                "max_portfolio_exposure_percent must be between 0 and 100"
            )

        if not 0 < self.max_drawdown_percent <= 100:
            raise ValueError(
                "max_drawdown_percent must be between 0 and 100"
            )

        if not 0 <= self.warning_drawdown_percent <= 100:
            raise ValueError(
                "warning_drawdown_percent must be between 0 and 100"
            )

        if (
            self.warning_drawdown_percent
            > self.max_drawdown_percent
        ):
            raise ValueError(
                "warning_drawdown_percent cannot exceed "
                "max_drawdown_percent"
            )

        if not 0 < self.max_daily_loss_percent <= 100:
            raise ValueError(
                "max_daily_loss_percent must be between 0 and 100"
            )

        if not 0 < self.default_stop_loss_percent <= 100:
            raise ValueError(
                "default_stop_loss_percent must be between 0 and 100"
            )

        if self.default_take_profit_percent < 0:
            raise ValueError(
                "default_take_profit_percent cannot be negative"
            )

        if not 0 < self.max_concentration_percent <= 100:
            raise ValueError(
                "max_concentration_percent must be between 0 and 100"
            )

        if not self.currency:
            raise ValueError("currency is required")


# ============================================================
# RISK LIMIT
# ============================================================


@dataclass
class RiskLimit:
    """
    Represents one individual risk limit.

    Actual limit checking is performed by Layer 5.
    """

    risk_type: RiskType
    limit_percent: float
    warning_percent: Optional[float] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.limit_percent <= 100:
            raise ValueError(
                "limit_percent must be between 0 and 100"
            )

        if self.warning_percent is not None:
            if not 0 <= self.warning_percent <= 100:
                raise ValueError(
                    "warning_percent must be between 0 and 100"
                )

            if self.warning_percent > self.limit_percent:
                raise ValueError(
                    "warning_percent cannot exceed limit_percent"
                )


# ============================================================
# POSITION RISK
# ============================================================


@dataclass
class PositionRisk:
    """
    Risk information for one position.
    """

    instrument_key: str

    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0

    market_value: float = 0.0
    exposure_percent: float = 0.0

    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0

    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    risk_status: RiskStatus = RiskStatus.SAFE
    risk_level: RiskLevel = RiskLevel.LOW

    def __post_init__(self) -> None:
        if not self.instrument_key:
            raise ValueError("instrument_key is required")

        if self.quantity < 0:
            raise ValueError("quantity cannot be negative")

        if self.entry_price < 0:
            raise ValueError("entry_price cannot be negative")

        if self.current_price < 0:
            raise ValueError("current_price cannot be negative")

        if self.market_value < 0:
            raise ValueError("market_value cannot be negative")

        if self.exposure_percent < 0:
            raise ValueError(
                "exposure_percent cannot be negative"
            )


# ============================================================
# PORTFOLIO RISK
# ============================================================


@dataclass
class PortfolioRisk:
    """
    Overall portfolio-level risk information.
    """

    total_equity: float = 0.0
    cash: float = 0.0
    invested_value: float = 0.0

    exposure_percent: float = 0.0
    cash_percent: float = 0.0

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0

    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    drawdown_percent: float = 0.0

    daily_loss: float = 0.0
    daily_loss_percent: float = 0.0

    open_positions: int = 0

    risk_status: RiskStatus = RiskStatus.SAFE
    risk_level: RiskLevel = RiskLevel.LOW

    def __post_init__(self) -> None:
        if self.total_equity < 0:
            raise ValueError(
                "total_equity cannot be negative"
            )

        if self.cash < 0:
            raise ValueError(
                "cash cannot be negative"
            )

        if self.invested_value < 0:
            raise ValueError(
                "invested_value cannot be negative"
            )

        if self.exposure_percent < 0:
            raise ValueError(
                "exposure_percent cannot be negative"
            )

        if self.cash_percent < 0:
            raise ValueError(
                "cash_percent cannot be negative"
            )

        if self.peak_equity < 0:
            raise ValueError(
                "peak_equity cannot be negative"
            )

        if self.current_drawdown < 0:
            raise ValueError(
                "current_drawdown cannot be negative"
            )

        if self.drawdown_percent < 0:
            raise ValueError(
                "drawdown_percent cannot be negative"
            )

        if self.daily_loss_percent < 0:
            raise ValueError(
                "daily_loss_percent cannot be negative"
            )

        if self.open_positions < 0:
            raise ValueError(
                "open_positions cannot be negative"
            )


# ============================================================
# RISK EVALUATION
# ============================================================


@dataclass
class RiskEvaluation:
    """
    Result of evaluating a proposed trade or portfolio state.

    Layer 2 and later layers will populate this object.
    """

    status: RiskStatus = RiskStatus.SAFE
    level: RiskLevel = RiskLevel.LOW
    action: RiskAction = RiskAction.ALLOW

    risk_type: Optional[RiskType] = None

    message: str = ""

    current_value: float = 0.0
    limit_value: float = 0.0
    utilization_percent: float = 0.0

    instrument_key: Optional[str] = None

    evaluated_at: datetime = field(
        default_factory=datetime.utcnow
    )

    def __post_init__(self) -> None:
        if self.current_value < 0:
            raise ValueError(
                "current_value cannot be negative"
            )

        if self.limit_value < 0:
            raise ValueError(
                "limit_value cannot be negative"
            )

        if self.utilization_percent < 0:
            raise ValueError(
                "utilization_percent cannot be negative"
            )


# ============================================================
# RISK ALERT
# ============================================================


@dataclass
class RiskAlert:
    """
    Represents a risk event generated by Phase 12.
    """

    alert_id: str
    risk_type: RiskType
    level: RiskLevel
    status: RiskStatus

    message: str

    instrument_key: Optional[str] = None

    current_value: Optional[float] = None
    limit_value: Optional[float] = None

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    acknowledged: bool = False

    def __post_init__(self) -> None:
        if not self.alert_id:
            raise ValueError("alert_id is required")

        if not self.message:
            raise ValueError("message is required")

        if (
            self.current_value is not None
            and self.current_value < 0
        ):
            raise ValueError(
                "current_value cannot be negative"
            )

        if (
            self.limit_value is not None
            and self.limit_value < 0
        ):
            raise ValueError(
                "limit_value cannot be negative"
            )