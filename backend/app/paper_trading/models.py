from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ============================================================
# ENUMS
# ============================================================

class TradingMode(str, Enum):
    PAPER = "paper"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionSide(str, Enum):
    LONG = "LONG"


class TradingSessionStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


# ============================================================
# PAPER TRADING CONFIGURATION
# ============================================================

@dataclass
class PaperTradingConfig:
    """
    Configuration for a Phase 11 paper-trading session.
    """

    instrument_key: str
    initial_capital: float = 100000.0

    trading_mode: TradingMode = TradingMode.PAPER

    commission_percent: float = 0.0
    slippage_percent: float = 0.0

    allow_fractional_quantity: bool = False

    max_position_value_percent: float = 100.0

    currency: str = "INR"

    def __post_init__(self) -> None:
        if not self.instrument_key:
            raise ValueError("instrument_key is required")

        if self.initial_capital <= 0:
            raise ValueError(
                "initial_capital must be greater than zero"
            )

        if self.commission_percent < 0:
            raise ValueError(
                "commission_percent cannot be negative"
            )

        if self.slippage_percent < 0:
            raise ValueError(
                "slippage_percent cannot be negative"
            )

        if not 0 < self.max_position_value_percent <= 100:
            raise ValueError(
                "max_position_value_percent must be between "
                "0 and 100"
            )


# ============================================================
# TRADING SESSION
# ============================================================

@dataclass
class TradingSession:
    """
    Represents one paper-trading session.
    """

    session_id: str
    config: PaperTradingConfig

    status: TradingSessionStatus = (
        TradingSessionStatus.CREATED
    )

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    total_orders: int = 0
    total_trades: int = 0


# ============================================================
# ORDER
# ============================================================

@dataclass
class PaperOrder:
    """
    Represents a simulated paper-trading order.
    """

    order_id: str
    session_id: str

    instrument_key: str

    side: OrderSide
    order_type: OrderType = OrderType.MARKET

    quantity: float = 0.0
    requested_price: Optional[float] = None

    status: OrderStatus = OrderStatus.PENDING

    filled_price: Optional[float] = None
    filled_quantity: float = 0.0

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    filled_at: Optional[datetime] = None

    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id is required")

        if not self.session_id:
            raise ValueError("session_id is required")

        if not self.instrument_key:
            raise ValueError("instrument_key is required")

        if self.quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero"
            )


# ============================================================
# POSITION
# ============================================================

@dataclass
class PaperPosition:
    """
    Represents an open simulated position.
    """

    instrument_key: str

    quantity: float = 0.0

    average_entry_price: float = 0.0

    current_price: float = 0.0

    realized_pnl: float = 0.0

    unrealized_pnl: float = 0.0

    opened_at: Optional[datetime] = None
    updated_at: datetime = field(
        default_factory=datetime.utcnow
    )

    @property
    def market_value(self) -> float:
        return (
            self.quantity *
            self.current_price
        )

    @property
    def invested_value(self) -> float:
        return (
            self.quantity *
            self.average_entry_price
        )

    def update_market_price(
        self,
        current_price: float,
    ) -> None:

        if current_price < 0:
            raise ValueError(
                "current_price cannot be negative"
            )

        self.current_price = current_price

        self.unrealized_pnl = (
            self.current_price
            - self.average_entry_price
        ) * self.quantity

        self.updated_at = datetime.utcnow()


# ============================================================
# COMPLETED TRADE
# ============================================================

@dataclass
class PaperTrade:
    """
    Represents a completed simulated trade.
    """

    trade_id: str
    session_id: str

    instrument_key: str

    side: OrderSide

    quantity: float

    entry_price: float
    exit_price: float

    entry_time: datetime
    exit_time: datetime

    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0

    net_pnl: float = 0.0

    signal: Optional[str] = None
    confidence: Optional[float] = None

    def calculate_pnl(self) -> float:
        """
        Calculate gross and net trade P&L.
        """

        if self.side == OrderSide.BUY:
            self.gross_pnl = (
                self.exit_price
                - self.entry_price
            ) * self.quantity

        else:
            self.gross_pnl = (
                self.entry_price
                - self.exit_price
            ) * self.quantity

        self.net_pnl = (
            self.gross_pnl
            - self.commission
            - self.slippage_cost
        )

        return self.net_pnl


# ============================================================
# PORTFOLIO SNAPSHOT
# ============================================================

@dataclass
class PortfolioSnapshot:
    """
    Point-in-time snapshot of the simulated portfolio.
    """

    timestamp: datetime

    cash: float
    invested_value: float
    market_value: float

    realized_pnl: float
    unrealized_pnl: float

    total_pnl: float

    total_return_percent: float

    open_positions: int

    total_trades: int

    @property
    def total_equity(self) -> float:
        return self.cash + self.market_value