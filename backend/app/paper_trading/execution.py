from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .models import (
    OrderSide,
    OrderStatus,
    PaperOrder,
    PaperTradingConfig,
)


# ============================================================
# EXECUTION RESULT
# ============================================================

@dataclass
class ExecutionResult:
    """
    Result of a simulated paper-trading execution.
    """

    order_id: str

    status: OrderStatus

    requested_price: float

    execution_price: float

    requested_quantity: float

    executed_quantity: float

    gross_value: float

    commission: float

    slippage_cost: float

    total_cost: float

    executed_at: datetime


# ============================================================
# PAPER EXECUTION ENGINE
# ============================================================

class PaperExecutionEngine:
    """
    Phase 11 - Layer 5

    Simulates execution of paper-trading orders.

    Responsibilities:
    - Validate pending orders
    - Determine simulated execution price
    - Apply configurable slippage
    - Calculate commission
    - Fill the order
    - Return execution details

    This layer NEVER sends orders to a real broker.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        config: PaperTradingConfig,
    ) -> None:

        self.config = config

    # ========================================================
    # EXECUTE ORDER
    # ========================================================

    def execute(
        self,
        order: PaperOrder,
        market_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a pending paper order using simulated execution.

        If market_price is supplied, it becomes the base execution
        price.

        Otherwise, requested_price must be available.
        """

        # ----------------------------------------------------
        # Validate order status
        # ----------------------------------------------------

        if order.status != OrderStatus.PENDING:
            raise ValueError(
                "Only PENDING orders can be executed"
            )

        # ----------------------------------------------------
        # Validate quantity
        # ----------------------------------------------------

        if order.quantity <= 0:
            raise ValueError(
                "Order quantity must be greater than zero"
            )

        # ----------------------------------------------------
        # Determine base price
        # ----------------------------------------------------

        if market_price is not None:

            if market_price <= 0:
                raise ValueError(
                    "market_price must be greater than zero"
                )

            base_price = market_price

        elif order.requested_price is not None:

            if order.requested_price <= 0:
                raise ValueError(
                    "requested_price must be greater than zero"
                )

            base_price = order.requested_price

        else:

            raise ValueError(
                "A market price or requested price is required"
            )

        # ----------------------------------------------------
        # Calculate execution price
        # ----------------------------------------------------

        execution_price = self._apply_slippage(
            order.side,
            base_price,
        )

        # ----------------------------------------------------
        # Execution quantity
        # ----------------------------------------------------

        executed_quantity = order.quantity

        # ----------------------------------------------------
        # Gross trade value
        # ----------------------------------------------------

        gross_value = (
            execution_price
            * executed_quantity
        )

        # ----------------------------------------------------
        # Commission
        # ----------------------------------------------------

        commission = (
            gross_value
            * self.config.commission_percent
            / 100.0
        )

        # ----------------------------------------------------
        # Slippage cost
        # ----------------------------------------------------

        slippage_cost = (
            abs(execution_price - base_price)
            * executed_quantity
        )

        # ----------------------------------------------------
        # Total transaction cost
        # ----------------------------------------------------

        total_cost = (
            gross_value
            + commission
        )

        # ----------------------------------------------------
        # Update order
        # ----------------------------------------------------

        executed_at = datetime.utcnow()

        order.status = OrderStatus.FILLED
        order.filled_price = execution_price
        order.filled_quantity = executed_quantity
        order.filled_at = executed_at

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return ExecutionResult(
            order_id=order.order_id,
            status=order.status,
            requested_price=base_price,
            execution_price=execution_price,
            requested_quantity=order.quantity,
            executed_quantity=executed_quantity,
            gross_value=gross_value,
            commission=commission,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            executed_at=executed_at,
        )

    # ========================================================
    # SLIPPAGE
    # ========================================================

    def _apply_slippage(
        self,
        side: OrderSide,
        base_price: float,
    ) -> float:
        """
        Apply configured slippage.

        BUY:
            Price increases.

        SELL:
            Price decreases.

        Example:

            base price = 100
            slippage = 0.10%

            BUY  -> 100.10
            SELL -> 99.90
        """

        slippage_rate = (
            self.config.slippage_percent
            / 100.0
        )

        if side == OrderSide.BUY:

            return (
                base_price
                * (1.0 + slippage_rate)
            )

        if side == OrderSide.SELL:

            return (
                base_price
                * (1.0 - slippage_rate)
            )

        raise ValueError(
            f"Unsupported order side: {side}"
        )