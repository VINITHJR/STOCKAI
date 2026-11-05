from __future__ import annotations

from typing import Dict, List, Optional

from .models import PaperTrade


class PaperTradeHistory:
    """
    Phase 11 - Layer 7

    Stores and retrieves completed paper trades.

    Responsibilities:
    - Record completed trades
    - Prevent duplicate trade IDs
    - Retrieve individual trades
    - List trades
    - Filter trades by session
    - Filter trades by instrument
    - Calculate historical trade statistics

    This layer does NOT execute orders and does not modify
    the portfolio.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self) -> None:
        self.trades: Dict[str, PaperTrade] = {}

    # ============================================================
    # RECORD TRADE
    # ============================================================

    def record_trade(
        self,
        trade: PaperTrade,
    ) -> PaperTrade:
        """
        Record a completed paper trade.

        Duplicate trade IDs are rejected.
        """

        if trade.trade_id in self.trades:
            raise ValueError(
                f"Trade already exists: {trade.trade_id}"
            )

        self._validate_trade(trade)

        self.trades[trade.trade_id] = trade

        return trade

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_trade(
        trade: PaperTrade,
    ) -> None:
        """
        Validate basic completed-trade fields.
        """

        if not trade.trade_id:
            raise ValueError(
                "trade_id is required"
            )

        if not trade.session_id:
            raise ValueError(
                "session_id is required"
            )

        if not trade.instrument_key:
            raise ValueError(
                "instrument_key is required"
            )

        if trade.quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero"
            )

        if trade.entry_price <= 0:
            raise ValueError(
                "entry_price must be greater than zero"
            )

        if trade.exit_price <= 0:
            raise ValueError(
                "exit_price must be greater than zero"
            )

        if trade.exit_time < trade.entry_time:
            raise ValueError(
                "exit_time cannot be earlier than entry_time"
            )

    # ============================================================
    # GET TRADE
    # ============================================================

    def get_trade(
        self,
        trade_id: str,
    ) -> PaperTrade:
        """
        Retrieve a trade by trade ID.
        """

        if trade_id not in self.trades:
            raise KeyError(
                f"Trade not found: {trade_id}"
            )

        return self.trades[trade_id]

    # ============================================================
    # LIST ALL TRADES
    # ============================================================

    def list_trades(self) -> List[PaperTrade]:
        """
        Return all recorded trades.
        """

        return list(self.trades.values())

    # ============================================================
    # FILTER BY SESSION
    # ============================================================

    def get_session_trades(
        self,
        session_id: str,
    ) -> List[PaperTrade]:
        """
        Return all trades belonging to a session.
        """

        return [
            trade
            for trade in self.trades.values()
            if trade.session_id == session_id
        ]

    # ============================================================
    # FILTER BY INSTRUMENT
    # ============================================================

    def get_instrument_trades(
        self,
        instrument_key: str,
    ) -> List[PaperTrade]:
        """
        Return all trades for an instrument.
        """

        return [
            trade
            for trade in self.trades.values()
            if trade.instrument_key == instrument_key
        ]

    # ============================================================
    # FILTER BY SESSION + INSTRUMENT
    # ============================================================

    def get_filtered_trades(
        self,
        session_id: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> List[PaperTrade]:
        """
        Return trades matching the supplied filters.

        If neither filter is supplied, all trades are returned.
        """

        trades = list(self.trades.values())

        if session_id is not None:
            trades = [
                trade
                for trade in trades
                if trade.session_id == session_id
            ]

        if instrument_key is not None:
            trades = [
                trade
                for trade in trades
                if trade.instrument_key == instrument_key
            ]

        return trades

    # ============================================================
    # TRADE COUNT
    # ============================================================

    def trade_count(
        self,
        session_id: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> int:
        """
        Return the number of trades matching the filters.
        """

        return len(
            self.get_filtered_trades(
                session_id=session_id,
                instrument_key=instrument_key,
            )
        )

    # ============================================================
    # TOTAL P&L
    # ============================================================

    def total_pnl(
        self,
        session_id: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> float:
        """
        Calculate total net P&L from recorded trades.
        """

        trades = self.get_filtered_trades(
            session_id=session_id,
            instrument_key=instrument_key,
        )

        return sum(
            trade.net_pnl
            for trade in trades
        )

    # ============================================================
    # WINNING TRADES
    # ============================================================

    def winning_trades(
        self,
        session_id: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> List[PaperTrade]:
        """
        Return profitable trades.
        """

        return [
            trade
            for trade in self.get_filtered_trades(
                session_id=session_id,
                instrument_key=instrument_key,
            )
            if trade.net_pnl > 0
        ]

    # ============================================================
    # LOSING TRADES
    # ============================================================

    def losing_trades(
        self,
        session_id: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> List[PaperTrade]:
        """
        Return losing trades.
        """

        return [
            trade
            for trade in self.get_filtered_trades(
                session_id=session_id,
                instrument_key=instrument_key,
            )
            if trade.net_pnl < 0
        ]

    # ============================================================
    # WIN RATE
    # ============================================================

    def win_rate_percent(
        self,
        session_id: Optional[str] = None,
        instrument_key: Optional[str] = None,
    ) -> float:
        """
        Calculate winning trade percentage.
        """

        trades = self.get_filtered_trades(
            session_id=session_id,
            instrument_key=instrument_key,
        )

        if not trades:
            return 0.0

        winners = sum(
            1
            for trade in trades
            if trade.net_pnl > 0
        )

        return (
            winners
            / len(trades)
        ) * 100.0

    # ============================================================
    # CLEAR HISTORY
    # ============================================================

    def clear(self) -> None:
        """
        Remove all recorded trades.

        Intended for isolated testing or resetting a paper
        trading session.
        """

        self.trades.clear()