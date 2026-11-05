from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from .models import (
    OrderSide,
    OrderStatus,
    OrderType,
    PaperOrder,
    PaperTradingConfig,
    TradingSession,
    TradingSessionStatus,
)


class PaperTradingEngine:
    """
    Phase 11 - Layer 2

    Main controller for paper-trading sessions.

    Responsibilities:
    - Create paper-trading sessions
    - Start / pause / stop sessions
    - Maintain session state
    - Accept AI signals
    - Convert actionable signals into paper orders
    - Track prepared orders

    This layer does NOT execute orders.
    Actual execution will be handled by later Phase 11 layers.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self) -> None:
        self.sessions: Dict[str, TradingSession] = {}
        self.orders: Dict[str, PaperOrder] = {}

    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================

    def create_session(
        self,
        config: PaperTradingConfig,
    ) -> TradingSession:
        """
        Create a new paper-trading session.

        The session starts in CREATED state.
        """

        if config.trading_mode.value != "paper":
            raise ValueError(
                "Phase 11 supports paper trading only"
            )

        session_id = self._generate_session_id()

        session = TradingSession(
            session_id=session_id,
            config=config,
            status=TradingSessionStatus.CREATED,
        )

        self.sessions[session_id] = session

        return session

    def start_session(
        self,
        session_id: str,
    ) -> TradingSession:
        """
        Start a CREATED or PAUSED paper-trading session.
        """

        session = self._get_session(session_id)

        if session.status == TradingSessionStatus.STOPPED:
            raise ValueError(
                "Stopped session cannot be started again"
            )

        if session.status == TradingSessionStatus.COMPLETED:
            raise ValueError(
                "Completed session cannot be started again"
            )

        if session.status == TradingSessionStatus.RUNNING:
            return session

        session.status = TradingSessionStatus.RUNNING

        if session.started_at is None:
            session.started_at = datetime.utcnow()

        return session

    def pause_session(
        self,
        session_id: str,
    ) -> TradingSession:
        """
        Pause a running paper-trading session.
        """

        session = self._get_session(session_id)

        if session.status != TradingSessionStatus.RUNNING:
            raise ValueError(
                "Only a RUNNING session can be paused"
            )

        session.status = TradingSessionStatus.PAUSED

        return session

    def stop_session(
        self,
        session_id: str,
    ) -> TradingSession:
        """
        Stop a running or paused paper-trading session.
        """

        session = self._get_session(session_id)

        if session.status == TradingSessionStatus.COMPLETED:
            raise ValueError(
                "Completed session cannot be stopped"
            )

        if session.status == TradingSessionStatus.STOPPED:
            return session

        session.status = TradingSessionStatus.STOPPED
        session.stopped_at = datetime.utcnow()

        return session

    def complete_session(
        self,
        session_id: str,
    ) -> TradingSession:
        """
        Mark a paper-trading session as completed.
        """

        session = self._get_session(session_id)

        if session.status == TradingSessionStatus.RUNNING:
            raise ValueError(
                "Stop the running session before completing it"
            )

        if session.status == TradingSessionStatus.COMPLETED:
            return session

        session.status = TradingSessionStatus.COMPLETED

        if session.stopped_at is None:
            session.stopped_at = datetime.utcnow()

        return session

    # ============================================================
    # SESSION LOOKUP
    # ============================================================

    def get_session(
        self,
        session_id: str,
    ) -> TradingSession:
        """
        Return a session by ID.
        """

        return self._get_session(session_id)

    def list_sessions(self) -> List[TradingSession]:
        """
        Return all paper-trading sessions.
        """

        return list(self.sessions.values())

    # ============================================================
    # SIGNAL -> ORDER
    # ============================================================

    def create_order_from_signal(
        self,
        session_id: str,
        signal: str,
        confidence: Optional[float],
        current_price: float,
        quantity: float,
    ) -> Optional[PaperOrder]:
        """
        Convert an AI signal into a paper order.

        Supported signals:

        BUY side:
            bullish
            buy
            long

        SELL side:
            bearish
            sell
            short

        No order:
            neutral
            hold

        The order is only CREATED here.
        It is not executed in Layer 2.

        Returns:
            PaperOrder for actionable signals
            None for neutral/hold signals
        """

        session = self._get_session(session_id)

        if session.status != TradingSessionStatus.RUNNING:
            raise ValueError(
                "Orders can only be created for a RUNNING session"
            )

        if current_price <= 0:
            raise ValueError(
                "current_price must be greater than zero"
            )

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero"
            )

        normalized_signal = self._normalize_signal(signal)

        # --------------------------------------------------------
        # Neutral / Hold = No Order
        # --------------------------------------------------------

        if normalized_signal is None:
            return None

        # --------------------------------------------------------
        # Validate confidence if supplied
        # --------------------------------------------------------

        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0 and 1"
                )

        # --------------------------------------------------------
        # Determine order side
        # --------------------------------------------------------

        if normalized_signal == "bullish":
            side = OrderSide.BUY

        elif normalized_signal == "bearish":
            side = OrderSide.SELL

        else:
            raise ValueError(
                f"Unsupported normalized signal: {normalized_signal}"
            )

        # --------------------------------------------------------
        # Create order
        # --------------------------------------------------------

        order_id = self._generate_order_id()

        order = PaperOrder(
            order_id=order_id,
            session_id=session_id,
            instrument_key=session.config.instrument_key,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            requested_price=current_price,
            status=OrderStatus.PENDING,
        )

        self.orders[order_id] = order

        session.total_orders += 1

        return order

    # ============================================================
    # ORDER LOOKUP
    # ============================================================

    def get_order(
        self,
        order_id: str,
    ) -> PaperOrder:
        """
        Return a prepared paper order.
        """

        if order_id not in self.orders:
            raise KeyError(
                f"Order not found: {order_id}"
            )

        return self.orders[order_id]

    def list_orders(
        self,
        session_id: Optional[str] = None,
    ) -> List[PaperOrder]:
        """
        Return all orders.

        If session_id is provided, only orders belonging
        to that session are returned.
        """

        if session_id is None:
            return list(self.orders.values())

        self._get_session(session_id)

        return [
            order
            for order in self.orders.values()
            if order.session_id == session_id
        ]

    # ============================================================
    # SIGNAL NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_signal(
        signal: str,
    ) -> Optional[str]:
        """
        Normalize AI signal names.

        Returns:
            bullish
            bearish
            None for neutral/hold
        """

        if not isinstance(signal, str):
            raise ValueError(
                "signal must be a string"
            )

        normalized = signal.strip().lower()

        bullish_signals = {
            "bullish",
            "buy",
            "long",
        }

        bearish_signals = {
            "bearish",
            "sell",
            "short",
        }

        neutral_signals = {
            "neutral",
            "hold",
            "none",
        }

        if normalized in bullish_signals:
            return "bullish"

        if normalized in bearish_signals:
            return "bearish"

        if normalized in neutral_signals:
            return None

        raise ValueError(
            f"Unsupported trading signal: {signal}"
        )

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _get_session(
        self,
        session_id: str,
    ) -> TradingSession:
        """
        Retrieve a session or raise a clear error.
        """

        if session_id not in self.sessions:
            raise KeyError(
                f"Session not found: {session_id}"
            )

        return self.sessions[session_id]

    @staticmethod
    def _generate_session_id() -> str:
        """
        Generate a unique session ID.
        """

        return f"PT-{uuid4().hex[:12].upper()}"

    @staticmethod
    def _generate_order_id() -> str:
        """
        Generate a unique order ID.
        """

        return f"ORD-{uuid4().hex[:12].upper()}"