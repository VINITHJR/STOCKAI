"""
Phase 13 - Layer 1
Portfolio Analyzer / Foundation

Responsible for building the base portfolio-level analysis.

This layer does NOT:
- manage risk limits
- execute trades
- modify Phase 11 paper trading
- modify Phase 12 risk management
- calculate advanced performance metrics
- calculate diversification scores

Those responsibilities belong to later layers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .models import (
    PortfolioAnalysis,
    PortfolioPosition,
    PortfolioSnapshot,
    PositionAnalysis,
)


class PortfolioAnalyzer:
    """
    Phase 13 Layer 1 portfolio analyzer.

    The analyzer accepts portfolio position data and produces
    a normalized portfolio-level analysis.
    """

    def __init__(self, currency: str = "INR") -> None:
        self.currency = currency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        positions: Optional[Iterable[Mapping[str, Any]]] = None,
        cash: float = 0.0,
        initial_capital: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> PortfolioAnalysis:
        """
        Analyze the supplied portfolio.

        Parameters
        ----------
        positions:
            Iterable containing portfolio position dictionaries.

        cash:
            Available uninvested cash.

        initial_capital:
            Original portfolio capital.

        timestamp:
            Optional portfolio snapshot timestamp.

        Returns
        -------
        PortfolioAnalysis
        """

        normalized_positions = self._normalize_positions(positions)

        snapshot = PortfolioSnapshot(
            positions=normalized_positions,
            initial_capital=self._safe_float(initial_capital),
            cash=self._safe_float(cash),
            timestamp=timestamp,
        )

        position_analysis = self._analyze_positions(snapshot)

        total_value = snapshot.total_value

        invested_percent = self._calculate_percentage(
            snapshot.invested_value,
            total_value,
        )

        cash_percent = self._calculate_percentage(
            snapshot.cash,
            total_value,
        )

        status = self._determine_portfolio_status(
            snapshot=snapshot,
            invested_percent=invested_percent,
        )

        return PortfolioAnalysis(
            position_count=len(normalized_positions),
            initial_capital=snapshot.initial_capital,
            cash=self._round(snapshot.cash),
            invested_value=self._round(snapshot.invested_value),
            market_value=self._round(snapshot.market_value),
            total_value=self._round(snapshot.total_value),
            total_pnl=self._round(snapshot.total_pnl),
            total_pnl_percent=self._round(snapshot.total_pnl_percent),
            invested_percent=self._round(invested_percent),
            cash_percent=self._round(cash_percent),
            positions=position_analysis,
            status=status,
            metadata={
                "currency": self.currency,
                "layer": 1,
                "component": "portfolio_analyzer",
            },
        )

    def analyze_snapshot(
        self,
        snapshot: PortfolioSnapshot,
    ) -> PortfolioAnalysis:
        """
        Analyze an already-created PortfolioSnapshot.
        """

        return self.analyze(
            positions=[
                {
                    "instrument_key": position.instrument_key,
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "average_price": position.average_price,
                    "current_price": position.current_price,
                    "sector": position.sector,
                    "exchange": position.exchange,
                    "asset_type": position.asset_type,
                }
                for position in snapshot.positions
            ],
            cash=snapshot.cash,
            initial_capital=snapshot.initial_capital,
            timestamp=snapshot.timestamp,
        )

    # ------------------------------------------------------------------
    # Position normalization
    # ------------------------------------------------------------------

    def _normalize_positions(
        self,
        positions: Optional[Iterable[Mapping[str, Any]]],
    ) -> List[PortfolioPosition]:
        """
        Normalize incoming position dictionaries.

        Supports common field aliases so the analyzer can consume
        data from existing paper-trading/portfolio structures without
        changing those structures.
        """

        if positions is None:
            return []

        normalized: List[PortfolioPosition] = []

        for raw_position in positions:
            if not isinstance(raw_position, Mapping):
                continue

            instrument_key = self._first_value(
                raw_position,
                "instrument_key",
                "instrumentToken",
                "instrument",
                "token",
            )

            symbol = self._first_value(
                raw_position,
                "symbol",
                "trading_symbol",
                "tradingsymbol",
                "ticker",
                "name",
            )

            quantity = self._first_numeric(
                raw_position,
                "quantity",
                "qty",
                "units",
            )

            average_price = self._first_numeric(
                raw_position,
                "average_price",
                "avg_price",
                "averagePrice",
                "buy_price",
                "entry_price",
            )

            current_price = self._first_numeric(
                raw_position,
                "current_price",
                "market_price",
                "ltp",
                "last_price",
                "price",
            )

            sector = self._first_value(
                raw_position,
                "sector",
                "sector_name",
            )

            exchange = self._first_value(
                raw_position,
                "exchange",
            )

            asset_type = self._first_value(
                raw_position,
                "asset_type",
                "instrument_type",
            )

            # Skip completely invalid positions.
            if not instrument_key and not symbol:
                continue

            normalized.append(
                PortfolioPosition(
                    instrument_key=str(instrument_key or ""),
                    symbol=str(symbol or ""),
                    quantity=quantity,
                    average_price=average_price,
                    current_price=current_price,
                    sector=str(sector or ""),
                    exchange=str(exchange or ""),
                    asset_type=str(asset_type or "EQUITY"),
                )
            )

        return normalized

    # ------------------------------------------------------------------
    # Position analysis
    # ------------------------------------------------------------------

    def _analyze_positions(
        self,
        snapshot: PortfolioSnapshot,
    ) -> List[PositionAnalysis]:
        """
        Produce normalized analysis for every position.
        """

        total_market_value = snapshot.market_value

        results: List[PositionAnalysis] = []

        for position in snapshot.positions:
            portfolio_weight = self._calculate_percentage(
                position.market_value,
                total_market_value,
            )

            status = self._determine_position_status(
                pnl_percent=position.pnl_percent,
                portfolio_weight=portfolio_weight,
            )

            results.append(
                PositionAnalysis(
                    instrument_key=position.instrument_key,
                    symbol=position.symbol,
                    quantity=self._round(position.quantity),
                    invested_value=self._round(position.invested_value),
                    market_value=self._round(position.market_value),
                    pnl=self._round(position.pnl),
                    pnl_percent=self._round(position.pnl_percent),
                    portfolio_weight=self._round(portfolio_weight),
                    sector=position.sector,
                    asset_type=position.asset_type,
                    status=status,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Status calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_position_status(
        pnl_percent: float,
        portfolio_weight: float,
    ) -> str:
        """
        Basic descriptive position status.

        This is intentionally not a risk decision.
        Phase 12 remains responsible for risk decisions.
        """

        if portfolio_weight >= 40:
            return "CONCENTRATED"

        if pnl_percent >= 10:
            return "PROFITABLE"

        if pnl_percent <= -10:
            return "LOSS"

        return "NEUTRAL"

    @staticmethod
    def _determine_portfolio_status(
        snapshot: PortfolioSnapshot,
        invested_percent: float,
    ) -> str:
        """
        Determine a simple descriptive portfolio status.

        This is not the Phase 12 risk engine and does not replace
        risk-management decisions.
        """

        if len(snapshot.positions) == 0:
            return "EMPTY"

        if snapshot.total_value <= 0:
            return "NO_VALUE"

        if snapshot.total_pnl_percent >= 10:
            return "POSITIVE"

        if snapshot.total_pnl_percent <= -10:
            return "NEGATIVE"

        if invested_percent >= 95:
            return "FULLY_INVESTED"

        return "HEALTHY"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_value(
        data: Mapping[str, Any],
        *keys: str,
    ) -> Any:
        """
        Return the first non-empty value from the supplied keys.
        """

        for key in keys:
            value = data.get(key)

            if value is not None and value != "":
                return value

        return None

    @classmethod
    def _first_numeric(
        cls,
        data: Mapping[str, Any],
        *keys: str,
    ) -> float:
        """
        Return the first valid numeric value from the supplied keys.
        """

        for key in keys:
            value = data.get(key)

            if value is None or value == "":
                continue

            numeric = cls._safe_float(value)

            if numeric != 0.0:
                return numeric

            # Preserve explicit zero when it is the supplied value.
            try:
                if float(value) == 0.0:
                    return 0.0
            except (TypeError, ValueError):
                continue

        return 0.0

    @staticmethod
    def _safe_float(value: Any) -> float:
        """
        Convert a value into a finite float.
        """

        try:
            number = float(value)

            if number != number:  # NaN
                return 0.0

            if number in (float("inf"), float("-inf")):
                return 0.0

            return number

        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _calculate_percentage(
        numerator: float,
        denominator: float,
    ) -> float:
        """
        Safely calculate numerator / denominator * 100.
        """

        if denominator <= 0:
            return 0.0

        return (numerator / denominator) * 100

    @staticmethod
    def _round(value: float) -> float:
        """
        Standardize numeric output precision.
        """

        return round(float(value), 4)