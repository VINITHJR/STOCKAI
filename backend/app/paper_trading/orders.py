from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from .models import (
    OrderStatus,
    PaperOrder,
)


class PaperOrderManager:
    """
    Phase 11 - Layer 3

    Manages the lifecycle of paper-trading orders.

    Responsibilities:
    - Create and register orders
    - Validate orders
    - Retrieve orders
    - List orders
    - Cancel pending orders
    - Reject pending orders
    - Mark pending orders as filled

    This layer does NOT execute trades.
    Actual simulated execution belongs to Layer 5.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self) -> None:
        self.orders: Dict[str, PaperOrder] = {}

    # ============================================================
    # CREATE / REGISTER ORDER
    # ============================================================

    def create_order(
        self,
        order: PaperOrder,
    ) -> PaperOrder:
        """
        Validate and register a new paper order.

        New orders must start in PENDING state.
        """

        if order.order_id in self.orders:
            raise ValueError(
                f"Order already exists: {order.order_id}"
            )

        self.validate_order(order)

        if order.status != OrderStatus.PENDING:
            raise ValueError(
                "New orders must have PENDING status"
            )

        self.orders[order.order_id] = order

        return order

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def validate_order(
        order: PaperOrder,
    ) -> bool:
        """
        Validate basic order properties.
        """

        if not order.order_id:
            raise ValueError(
                "order_id is required"
            )

        if not order.session_id:
            raise ValueError(
                "session_id is required"
            )

        if not order.instrument_key:
            raise ValueError(
                "instrument_key is required"
            )

        if order.quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero"
            )

        if order.requested_price is not None:
            if order.requested_price <= 0:
                raise ValueError(
                    "requested_price must be greater than zero"
                )

        return True

    # ============================================================
    # GET ORDER
    # ============================================================

    def get_order(
        self,
        order_id: str,
    ) -> PaperOrder:
        """
        Retrieve an order by order ID.
        """

        if order_id not in self.orders:
            raise KeyError(
                f"Order not found: {order_id}"
            )

        return self.orders[order_id]

    # ============================================================
    # LIST ORDERS
    # ============================================================

    def list_orders(
        self,
        session_id: str | None = None,
    ) -> List[PaperOrder]:
        """
        Return all orders.

        If session_id is supplied, only orders belonging
        to that session are returned.
        """

        if session_id is None:
            return list(self.orders.values())

        return [
            order
            for order in self.orders.values()
            if order.session_id == session_id
        ]

    # ============================================================
    # CANCEL ORDER
    # ============================================================

    def cancel_order(
        self,
        order_id: str,
    ) -> PaperOrder:
        """
        Cancel a pending order.

        Only PENDING orders can be cancelled.
        """

        order = self.get_order(order_id)

        if order.status != OrderStatus.PENDING:
            raise ValueError(
                f"Only PENDING orders can be cancelled. "
                f"Current status: {order.status.value}"
            )

        order.status = OrderStatus.CANCELLED

        return order

    # ============================================================
    # REJECT ORDER
    # ============================================================

    def reject_order(
        self,
        order_id: str,
        reason: str,
    ) -> PaperOrder:
        """
        Reject a pending order.

        A rejection reason is required.
        """

        order = self.get_order(order_id)

        if order.status != OrderStatus.PENDING:
            raise ValueError(
                f"Only PENDING orders can be rejected. "
                f"Current status: {order.status.value}"
            )

        if not reason or not reason.strip():
            raise ValueError(
                "rejection reason is required"
            )

        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason.strip()

        return order

    # ============================================================
    # MARK ORDER AS FILLED
    # ============================================================

    def mark_filled(
        self,
        order_id: str,
        filled_price: float,
        filled_quantity: float | None = None,
    ) -> PaperOrder:
        """
        Mark a pending order as FILLED.

        This method only records the fill.
        It does NOT perform broker execution.
        """

        order = self.get_order(order_id)

        if order.status != OrderStatus.PENDING:
            raise ValueError(
                f"Only PENDING orders can be filled. "
                f"Current status: {order.status.value}"
            )

        if filled_price <= 0:
            raise ValueError(
                "filled_price must be greater than zero"
            )

        if filled_quantity is None:
            filled_quantity = order.quantity

        if filled_quantity <= 0:
            raise ValueError(
                "filled_quantity must be greater than zero"
            )

        if filled_quantity > order.quantity:
            raise ValueError(
                "filled_quantity cannot exceed order quantity"
            )

        order.filled_price = filled_price
        order.filled_quantity = filled_quantity
        order.filled_at = datetime.utcnow()
        order.status = OrderStatus.FILLED

        return order

    # ============================================================
    # STATUS HELPERS
    # ============================================================

    def is_pending(
        self,
        order_id: str,
    ) -> bool:
        """
        Check whether an order is pending.
        """

        return (
            self.get_order(order_id).status
            == OrderStatus.PENDING
        )

    def is_filled(
        self,
        order_id: str,
    ) -> bool:
        """
        Check whether an order is filled.
        """

        return (
            self.get_order(order_id).status
            == OrderStatus.FILLED
        )

    def is_cancelled(
        self,
        order_id: str,
    ) -> bool:
        """
        Check whether an order is cancelled.
        """

        return (
            self.get_order(order_id).status
            == OrderStatus.CANCELLED
        )

    def is_rejected(
        self,
        order_id: str,
    ) -> bool:
        """
        Check whether an order is rejected.
        """

        return (
            self.get_order(order_id).status
            == OrderStatus.REJECTED
        )