"""
Phase 13 - Layer 5
Position & Sector Attribution

Analyzes how individual positions and sectors contribute to
overall portfolio profit and loss.

This layer is descriptive only.

It does NOT:
- modify Phase 11 Paper Trading
- modify Phase 12 Risk Management
- modify Phase 13 Layers 1-4
- execute trades
- block trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .portfolio_analyzer import PortfolioAnalyzer


@dataclass
class AttributionItem:
    """
    Represents the performance contribution of a position
    or sector.
    """

    name: str

    pnl: float
    contribution_percent: float

    invested_value: float
    market_value: float

    return_percent: float

    position_count: int = 0

    status: str = "NEUTRAL"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert attribution item into a JSON-compatible dictionary."""

        return {
            "name": self.name,
            "pnl": self.pnl,
            "contribution_percent": (
                self.contribution_percent
            ),
            "invested_value": self.invested_value,
            "market_value": self.market_value,
            "return_percent": self.return_percent,
            "position_count": self.position_count,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class AttributionAnalysis:
    """
    Complete Layer 5 position and sector attribution result.
    """

    total_pnl: float

    position_attributions: List[AttributionItem] = field(
        default_factory=list
    )

    sector_attributions: List[AttributionItem] = field(
        default_factory=list
    )

    best_position: Optional[Dict[str, Any]] = None
    worst_position: Optional[Dict[str, Any]] = None

    best_sector: Optional[Dict[str, Any]] = None
    worst_sector: Optional[Dict[str, Any]] = None

    positive_position_count: int = 0
    negative_position_count: int = 0
    neutral_position_count: int = 0

    positive_sector_count: int = 0
    negative_sector_count: int = 0
    neutral_sector_count: int = 0

    attribution_status: str = "NEUTRAL"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result into JSON-compatible dictionary."""

        return {
            "total_pnl": self.total_pnl,
            "position_attributions": [
                item.to_dict()
                for item in self.position_attributions
            ],
            "sector_attributions": [
                item.to_dict()
                for item in self.sector_attributions
            ],
            "best_position": self.best_position,
            "worst_position": self.worst_position,
            "best_sector": self.best_sector,
            "worst_sector": self.worst_sector,
            "positive_position_count": (
                self.positive_position_count
            ),
            "negative_position_count": (
                self.negative_position_count
            ),
            "neutral_position_count": (
                self.neutral_position_count
            ),
            "positive_sector_count": (
                self.positive_sector_count
            ),
            "negative_sector_count": (
                self.negative_sector_count
            ),
            "neutral_sector_count": (
                self.neutral_sector_count
            ),
            "attribution_status": (
                self.attribution_status
            ),
            "metadata": self.metadata,
        }


class PositionSectorAttributionAnalyzer:
    """
    Phase 13 Layer 5 analyzer.

    Calculates:

        Position attribution
        Sector attribution
        Best/worst contributors
        Positive/negative contributors

    Contribution percentage is calculated as:

        position P&L / total portfolio P&L × 100

    When total P&L is zero, contribution percentage is
    reported as zero to avoid misleading infinite values.
    """

    def __init__(
        self,
        currency: str = "INR",
    ) -> None:
        self.currency = currency

        self.portfolio_analyzer = PortfolioAnalyzer(
            currency=currency
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        positions: Optional[
            Iterable[Mapping[str, Any]]
        ] = None,
        cash: float = 0.0,
        initial_capital: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> AttributionAnalysis:
        """
        Analyze position and sector performance attribution.
        """

        portfolio = self.portfolio_analyzer.analyze(
            positions=positions,
            cash=cash,
            initial_capital=initial_capital,
            timestamp=timestamp,
        )

        total_pnl = self._safe_float(
            portfolio.total_pnl
        )

        position_attributions = (
            self._build_position_attributions(
                portfolio.positions,
                total_pnl,
            )
        )

        sector_attributions = (
            self._build_sector_attributions(
                portfolio.positions,
                total_pnl,
            )
        )

        best_position = self._best_item(
            position_attributions
        )

        worst_position = self._worst_item(
            position_attributions
        )

        best_sector = self._best_item(
            sector_attributions
        )

        worst_sector = self._worst_item(
            sector_attributions
        )

        (
            positive_position_count,
            negative_position_count,
            neutral_position_count,
        ) = self._count_statuses(
            position_attributions
        )

        (
            positive_sector_count,
            negative_sector_count,
            neutral_sector_count,
        ) = self._count_statuses(
            sector_attributions
        )

        attribution_status = (
            self._determine_status(
                total_pnl=total_pnl,
                positive_positions=(
                    positive_position_count
                ),
                negative_positions=(
                    negative_position_count
                ),
            )
        )

        return AttributionAnalysis(
            total_pnl=self._round(
                total_pnl
            ),
            position_attributions=(
                position_attributions
            ),
            sector_attributions=(
                sector_attributions
            ),
            best_position=best_position,
            worst_position=worst_position,
            best_sector=best_sector,
            worst_sector=worst_sector,
            positive_position_count=(
                positive_position_count
            ),
            negative_position_count=(
                negative_position_count
            ),
            neutral_position_count=(
                neutral_position_count
            ),
            positive_sector_count=(
                positive_sector_count
            ),
            negative_sector_count=(
                negative_sector_count
            ),
            neutral_sector_count=(
                neutral_sector_count
            ),
            attribution_status=(
                attribution_status
            ),
            metadata={
                "currency": self.currency,
                "layer": 5,
                "component": (
                    "position_sector_attribution"
                ),
                "attribution_basis": "pnl",
            },
        )

    # ------------------------------------------------------------------
    # Position attribution
    # ------------------------------------------------------------------

    def _build_position_attributions(
        self,
        positions: List[Any],
        total_pnl: float,
    ) -> List[AttributionItem]:
        """
        Build P&L attribution for each position.
        """

        results: List[AttributionItem] = []

        for position in positions:
            pnl = self._safe_float(
                position.pnl
            )

            invested_value = self._safe_float(
                position.invested_value
            )

            market_value = self._safe_float(
                position.market_value
            )

            return_percent = self._percentage(
                pnl,
                invested_value,
            )

            contribution_percent = (
                self._contribution_percentage(
                    pnl,
                    total_pnl,
                )
            )

            status = self._pnl_status(
                pnl
            )

            results.append(
                AttributionItem(
                    name=(
                        position.symbol
                        or position.instrument_key
                        or "UNKNOWN"
                    ),
                    pnl=self._round(pnl),
                    contribution_percent=self._round(
                        contribution_percent
                    ),
                    invested_value=self._round(
                        invested_value
                    ),
                    market_value=self._round(
                        market_value
                    ),
                    return_percent=self._round(
                        return_percent
                    ),
                    position_count=1,
                    status=status,
                    metadata={
                        "instrument_key": (
                            position.instrument_key
                        ),
                        "sector": (
                            position.sector
                            or "UNKNOWN"
                        ),
                        "asset_type": (
                            position.asset_type
                            or "UNKNOWN"
                        ),
                    },
                )
            )

        return self._sort_by_pnl(
            results
        )

    # ------------------------------------------------------------------
    # Sector attribution
    # ------------------------------------------------------------------

    def _build_sector_attributions(
        self,
        positions: List[Any],
        total_pnl: float,
    ) -> List[AttributionItem]:
        """
        Aggregate P&L attribution by sector.
        """

        sectors: Dict[
            str,
            Dict[str, float],
        ] = {}

        for position in positions:
            sector = (
                str(position.sector).strip()
                if position.sector
                else "UNKNOWN"
            )

            pnl = self._safe_float(
                position.pnl
            )

            invested_value = self._safe_float(
                position.invested_value
            )

            market_value = self._safe_float(
                position.market_value
            )

            if sector not in sectors:
                sectors[sector] = {
                    "pnl": 0.0,
                    "invested_value": 0.0,
                    "market_value": 0.0,
                    "position_count": 0.0,
                }

            sectors[sector]["pnl"] += pnl
            sectors[sector]["invested_value"] += (
                invested_value
            )
            sectors[sector]["market_value"] += (
                market_value
            )
            sectors[sector]["position_count"] += 1

        results: List[AttributionItem] = []

        for sector, values in sectors.items():
            pnl = values["pnl"]
            invested_value = values[
                "invested_value"
            ]
            market_value = values[
                "market_value"
            ]

            results.append(
                AttributionItem(
                    name=sector,
                    pnl=self._round(pnl),
                    contribution_percent=self._round(
                        self._contribution_percentage(
                            pnl,
                            total_pnl,
                        )
                    ),
                    invested_value=self._round(
                        invested_value
                    ),
                    market_value=self._round(
                        market_value
                    ),
                    return_percent=self._round(
                        self._percentage(
                            pnl,
                            invested_value,
                        )
                    ),
                    position_count=int(
                        values["position_count"]
                    ),
                    status=self._pnl_status(
                        pnl
                    ),
                )
            )

        return self._sort_by_pnl(
            results
        )

    # ------------------------------------------------------------------
    # Best / worst
    # ------------------------------------------------------------------

    @staticmethod
    def _best_item(
        items: List[AttributionItem],
    ) -> Optional[Dict[str, Any]]:
        """Return the highest P&L contributor."""

        if not items:
            return None

        item = max(
            items,
            key=lambda value: value.pnl,
        )

        return {
            "name": item.name,
            "pnl": item.pnl,
            "contribution_percent": (
                item.contribution_percent
            ),
            "return_percent": (
                item.return_percent
            ),
        }

    @staticmethod
    def _worst_item(
        items: List[AttributionItem],
    ) -> Optional[Dict[str, Any]]:
        """Return the lowest P&L contributor."""

        if not items:
            return None

        item = min(
            items,
            key=lambda value: value.pnl,
        )

        return {
            "name": item.name,
            "pnl": item.pnl,
            "contribution_percent": (
                item.contribution_percent
            ),
            "return_percent": (
                item.return_percent
            ),
        }

    # ------------------------------------------------------------------
    # Status counts
    # ------------------------------------------------------------------

    @staticmethod
    def _count_statuses(
        items: List[AttributionItem],
    ) -> tuple[int, int, int]:
        """
        Count positive, negative and neutral items.
        """

        positive = sum(
            1
            for item in items
            if item.status == "POSITIVE"
        )

        negative = sum(
            1
            for item in items
            if item.status == "NEGATIVE"
        )

        neutral = sum(
            1
            for item in items
            if item.status == "NEUTRAL"
        )

        return positive, negative, neutral

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _pnl_status(
        pnl: float,
    ) -> str:
        """Classify P&L direction."""

        if pnl > 0:
            return "POSITIVE"

        if pnl < 0:
            return "NEGATIVE"

        return "NEUTRAL"

    @staticmethod
    def _determine_status(
        total_pnl: float,
        positive_positions: int,
        negative_positions: int,
    ) -> str:
        """
        Determine overall attribution status.
        """

        if total_pnl > 0:
            if positive_positions > negative_positions:
                return "POSITIVE"

            return "MIXED_POSITIVE"

        if total_pnl < 0:
            if negative_positions > positive_positions:
                return "NEGATIVE"

            return "MIXED_NEGATIVE"

        if (
            positive_positions > 0
            and negative_positions > 0
        ):
            return "MIXED_NEUTRAL"

        return "NEUTRAL"

    # ------------------------------------------------------------------
    # Calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _contribution_percentage(
        pnl: float,
        total_pnl: float,
    ) -> float:
        """
        Calculate contribution to total P&L.

        If total P&L is zero, contribution is reported as
        zero because percentage contribution would otherwise
        be undefined.
        """

        if total_pnl == 0:
            return 0.0

        return (
            pnl
            / total_pnl
        ) * 100

    @staticmethod
    def _percentage(
        numerator: float,
        denominator: float,
    ) -> float:
        """Calculate percentage safely."""

        if denominator <= 0:
            return 0.0

        return (
            numerator
            / denominator
        ) * 100

    @staticmethod
    def _sort_by_pnl(
        items: List[AttributionItem],
    ) -> List[AttributionItem]:
        """Sort attribution from highest P&L to lowest."""

        return sorted(
            items,
            key=lambda item: item.pnl,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        """Safely convert a value to a finite float."""

        try:
            number = float(value)

            if number != number:
                return 0.0

            if number in (
                float("inf"),
                float("-inf"),
            ):
                return 0.0

            return number

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _round(
        value: float,
    ) -> float:
        """Round numeric values."""

        return round(
            float(value),
            4,
        )