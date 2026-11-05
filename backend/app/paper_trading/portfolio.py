from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from .models import (
    OrderSide,
    OrderStatus,
    PaperOrder,
    PaperPosition,
    PaperTradingConfig,
)


class PaperPortfolio:
    """
    Phase 11 - Layer 4

    Virtual paper-trading portfolio.

    Responsibilities:
    - Maintain virtual cash
    - Maintain open positions
    - Apply filled BUY orders
    - Apply filled SELL orders
    - Calculate average entry price
    - Calculate realized P&L
    - Calculate unrealized P&L
    - Calculate market value and total equity

    This layer does NOT execute orders.
    Execution belongs to Layer 5.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        config: PaperTradingConfig,
    ) -> None:

        self.config = config

        self.initial_capital = config.initial_capital
        self.cash = config.initial_capital

        self.positions: Dict[str, PaperPosition] = {}

        self.realized_pnl = 0.0

        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at

        self.total_orders_processed = 0
        self.total_filled_orders = 0

    # ============================================================
    # APPLY FILLED ORDER
    # ============================================================

    def apply_order(
        self,
        order: PaperOrder,
    ) -> PaperPosition | None:
        """
        Apply a FILLED order to the virtual portfolio.

        BUY:
            Cash decreases.
            Position increases.

        SELL:
            Cash increases.
            Position decreases.
            Realized P&L is calculated.

        The order must already be FILLED.
        """

        if order.status != OrderStatus.FILLED:
            raise ValueError(
                "Only FILLED orders can be applied to the portfolio"
            )

        if order.filled_price is None:
            raise ValueError(
                "Filled order must contain filled_price"
            )

        if order.filled_quantity <= 0:
            raise ValueError(
                "Filled order must contain filled_quantity > 0"
            )

        self.total_orders_processed += 1
        self.total_filled_orders += 1

        if order.side == OrderSide.BUY:
            position = self._apply_buy(order)

        elif order.side == OrderSide.SELL:
            position = self._apply_sell(order)

        else:
            raise ValueError(
                f"Unsupported order side: {order.side}"
            )

        self.updated_at = datetime.utcnow()

        return position

    # ============================================================
    # BUY
    # ============================================================

    def _apply_buy(
        self,
        order: PaperOrder,
    ) -> PaperPosition:

        filled_price = order.filled_price
        filled_quantity = order.filled_quantity

        assert filled_price is not None

        purchase_value = (
            filled_price * filled_quantity
        )

        if purchase_value > self.cash:
            raise ValueError(
                "Insufficient virtual cash for BUY order"
            )

        self.cash -= purchase_value

        instrument_key = order.instrument_key

        if instrument_key not in self.positions:

            position = PaperPosition(
                instrument_key=instrument_key,
                quantity=filled_quantity,
                average_entry_price=filled_price,
                current_price=filled_price,
                opened_at=datetime.utcnow(),
            )

            self.positions[instrument_key] = position

        else:

            position = self.positions[instrument_key]

            old_quantity = position.quantity
            old_average = position.average_entry_price

            new_quantity = (
                old_quantity + filled_quantity
            )

            new_average = (
                (
                    old_quantity * old_average
                )
                +
                (
                    filled_quantity * filled_price
                )
            ) / new_quantity

            position.quantity = new_quantity
            position.average_entry_price = new_average
            position.current_price = filled_price

            position.updated_at = datetime.utcnow()

        return self.positions[instrument_key]

    # ============================================================
    # SELL
    # ============================================================

    def _apply_sell(
        self,
        order: PaperOrder,
    ) -> PaperPosition | None:

        filled_price = order.filled_price
        filled_quantity = order.filled_quantity

        assert filled_price is not None

        instrument_key = order.instrument_key

        if instrument_key not in self.positions:
            raise ValueError(
                "Cannot SELL: no open position exists"
            )

        position = self.positions[instrument_key]

        if filled_quantity > position.quantity:
            raise ValueError(
                "Cannot SELL more quantity than current position"
            )

        # --------------------------------------------------------
        # Realized P&L
        # --------------------------------------------------------

        realized_trade_pnl = (
            filled_price
            - position.average_entry_price
        ) * filled_quantity

        self.realized_pnl += realized_trade_pnl

        # --------------------------------------------------------
        # Return cash
        # --------------------------------------------------------

        sale_value = (
            filled_price * filled_quantity
        )

        self.cash += sale_value

        # --------------------------------------------------------
        # Reduce position
        # --------------------------------------------------------

        position.quantity -= filled_quantity
        position.current_price = filled_price

        position.realized_pnl += realized_trade_pnl

        position.updated_at = datetime.utcnow()

        # --------------------------------------------------------
        # Position completely closed
        # --------------------------------------------------------

        if position.quantity == 0:

            del self.positions[instrument_key]

            return None

        # --------------------------------------------------------
        # Position remains open
        # --------------------------------------------------------

        position.unrealized_pnl = (
            position.current_price
            - position.average_entry_price
        ) * position.quantity

        return position

    # ============================================================
    # UPDATE MARKET PRICE
    # ============================================================

    def update_market_price(
        self,
        instrument_key: str,
        current_price: float,
    ) -> PaperPosition:

        if current_price <= 0:
            raise ValueError(
                "current_price must be greater than zero"
            )

        if instrument_key not in self.positions:
            raise KeyError(
                f"No open position for {instrument_key}"
            )

        position = self.positions[instrument_key]

        position.update_market_price(current_price)

        self.updated_at = datetime.utcnow()

        return position

    # ============================================================
    # GET POSITION
    # ============================================================

    def get_position(
        self,
        instrument_key: str,
    ) -> PaperPosition:

        if instrument_key not in self.positions:
            raise KeyError(
                f"No open position for {instrument_key}"
            )

        return self.positions[instrument_key]

    # ============================================================
    # LIST POSITIONS
    # ============================================================

    def list_positions(self) -> List[PaperPosition]:
        """
        Return all currently open positions.
        """

        return list(self.positions.values())

    # ============================================================
    # PORTFOLIO MARKET VALUE
    # ============================================================

    @property
    def invested_value(self) -> float:
        """
        Total amount currently invested at average entry prices.
        """

        return sum(
            position.invested_value
            for position in self.positions.values()
        )

    @property
    def market_value(self) -> float:
        """
        Current market value of all open positions.
        """

        return sum(
            position.market_value
            for position in self.positions.values()
        )

    @property
    def unrealized_pnl(self) -> float:
        """
        Total unrealized P&L across open positions.
        """

        return sum(
            position.unrealized_pnl
            for position in self.positions.values()
        )

    @property
    def total_pnl(self) -> float:
        """
        Realized + unrealized P&L.
        """

        return (
            self.realized_pnl
            + self.unrealized_pnl
        )

    @property
    def total_equity(self) -> float:
        """
        Current portfolio equity.

        Cash + current market value.
        """

        return self.cash + self.market_value

    @property
    def total_return_percent(self) -> float:
        """
        Portfolio return relative to initial capital.
        """

        if self.initial_capital <= 0:
            return 0.0

        return (
            (
                self.total_equity
                - self.initial_capital
            )
            / self.initial_capital
        ) * 100.0

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(self) -> dict:
        """
        Return a compact portfolio summary.
        """

        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "invested_value": self.invested_value,
            "market_value": self.market_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_equity": self.total_equity,
            "total_return_percent": self.total_return_percent,
            "open_positions": len(self.positions),
            "total_orders_processed": (
                self.total_orders_processed
            ),
            "total_filled_orders": (
                self.total_filled_orders
            ),
            "updated_at": self.updated_at,
        }