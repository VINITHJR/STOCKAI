"""
Phase 13 - Layer 4
Diversification Analysis

Analyzes portfolio diversification and concentration.

This layer is descriptive only.

It does NOT:
- modify Phase 11 Paper Trading
- modify Phase 12 Risk Management
- modify Phase 13 Layers 1-3
- execute trades
- block trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .portfolio_analyzer import PortfolioAnalyzer


@dataclass
class DiversificationAnalysis:
    """
    Complete Layer 4 diversification analysis result.
    """

    position_count: int
    sector_count: int
    asset_type_count: int

    largest_position_percent: float
    largest_sector_percent: float
    largest_asset_type_percent: float

    position_hhi: float
    sector_hhi: float
    asset_type_hhi: float

    position_diversification_score: float
    sector_diversification_score: float
    overall_diversification_score: float

    concentration_level: str
    diversification_status: str

    concentration_warnings: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result into a JSON-compatible dictionary."""

        return {
            "position_count": self.position_count,
            "sector_count": self.sector_count,
            "asset_type_count": self.asset_type_count,
            "largest_position_percent": (
                self.largest_position_percent
            ),
            "largest_sector_percent": (
                self.largest_sector_percent
            ),
            "largest_asset_type_percent": (
                self.largest_asset_type_percent
            ),
            "position_hhi": self.position_hhi,
            "sector_hhi": self.sector_hhi,
            "asset_type_hhi": self.asset_type_hhi,
            "position_diversification_score": (
                self.position_diversification_score
            ),
            "sector_diversification_score": (
                self.sector_diversification_score
            ),
            "overall_diversification_score": (
                self.overall_diversification_score
            ),
            "concentration_level": (
                self.concentration_level
            ),
            "diversification_status": (
                self.diversification_status
            ),
            "concentration_warnings": (
                self.concentration_warnings
            ),
            "metadata": self.metadata,
        }


class DiversificationAnalyzer:
    """
    Phase 13 Layer 4 diversification analyzer.

    Diversification is measured using portfolio market value.

    HHI is calculated from allocation percentages expressed
    as decimal weights.

    Example:

        50% / 50%

        HHI = 0.50² + 0.50²
            = 0.50

    Lower HHI means better diversification.

    The HHI is normalized into a diversification score:

        score = (1 - HHI) * 100

    A score closer to 100 indicates greater diversification.

    Note:
        HHI is descriptive here. It is not a Phase 12 risk limit.
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
    ) -> DiversificationAnalysis:
        """
        Analyze portfolio diversification.

        Parameters
        ----------
        positions:
            Portfolio position dictionaries.

        cash:
            Current available cash.

        initial_capital:
            Original portfolio capital.

        timestamp:
            Optional analysis timestamp.
        """

        portfolio = self.portfolio_analyzer.analyze(
            positions=positions,
            cash=cash,
            initial_capital=initial_capital,
            timestamp=timestamp,
        )

        position_weights = self._position_weights(
            portfolio.positions
        )

        sector_weights = self._sector_weights(
            portfolio.positions
        )

        asset_type_weights = self._asset_type_weights(
            portfolio.positions
        )

        position_hhi = self._calculate_hhi(
            position_weights
        )

        sector_hhi = self._calculate_hhi(
            sector_weights
        )

        asset_type_hhi = self._calculate_hhi(
            asset_type_weights
        )

        position_score = self._hhi_to_score(
            position_hhi
        )

        sector_score = self._hhi_to_score(
            sector_hhi
        )

        asset_type_score = self._hhi_to_score(
            asset_type_hhi
        )

        overall_score = self._calculate_overall_score(
            position_score=position_score,
            sector_score=sector_score,
            asset_type_score=asset_type_score,
        )

        largest_position = self._largest_percentage(
            position_weights
        )

        largest_sector = self._largest_percentage(
            sector_weights
        )

        largest_asset_type = self._largest_percentage(
            asset_type_weights
        )

        concentration_level = (
            self._determine_concentration_level(
                largest_position=largest_position,
                largest_sector=largest_sector,
                position_hhi=position_hhi,
                sector_hhi=sector_hhi,
            )
        )

        diversification_status = (
            self._determine_diversification_status(
                overall_score
            )
        )

        warnings = self._build_warnings(
            largest_position=largest_position,
            largest_sector=largest_sector,
            largest_asset_type=largest_asset_type,
            position_hhi=position_hhi,
            sector_hhi=sector_hhi,
        )

        return DiversificationAnalysis(
            position_count=len(
                portfolio.positions
            ),
            sector_count=len(
                sector_weights
            ),
            asset_type_count=len(
                asset_type_weights
            ),
            largest_position_percent=self._round(
                largest_position
            ),
            largest_sector_percent=self._round(
                largest_sector
            ),
            largest_asset_type_percent=self._round(
                largest_asset_type
            ),
            position_hhi=self._round(
                position_hhi
            ),
            sector_hhi=self._round(
                sector_hhi
            ),
            asset_type_hhi=self._round(
                asset_type_hhi
            ),
            position_diversification_score=self._round(
                position_score
            ),
            sector_diversification_score=self._round(
                sector_score
            ),
            overall_diversification_score=self._round(
                overall_score
            ),
            concentration_level=concentration_level,
            diversification_status=(
                diversification_status
            ),
            concentration_warnings=warnings,
            metadata={
                "currency": self.currency,
                "layer": 4,
                "component": (
                    "diversification_analysis"
                ),
                "allocation_basis": "market_value",
                "hhi_scale": "0_to_1",
            },
        )

    # ------------------------------------------------------------------
    # Allocation weights
    # ------------------------------------------------------------------

    @staticmethod
    def _position_weights(
        positions: List[Any],
    ) -> List[float]:
        """
        Calculate normalized position weights.

        The weights use total market value of positions,
        excluding cash.
        """

        total_market_value = sum(
            DiversificationAnalyzer._safe_float(
                position.market_value
            )
            for position in positions
        )

        if total_market_value <= 0:
            return []

        return [
            DiversificationAnalyzer._safe_float(
                position.market_value
            ) / total_market_value
            for position in positions
            if DiversificationAnalyzer._safe_float(
                position.market_value
            ) > 0
        ]

    @staticmethod
    def _sector_weights(
        positions: List[Any],
    ) -> List[float]:
        """Calculate sector allocation weights."""

        sector_values: Dict[str, float] = {}

        for position in positions:
            sector = (
                str(position.sector).strip()
                if position.sector
                else "UNKNOWN"
            )

            value = (
                DiversificationAnalyzer._safe_float(
                    position.market_value
                )
            )

            sector_values[sector] = (
                sector_values.get(sector, 0.0)
                + value
            )

        return DiversificationAnalyzer._normalize_values(
            list(sector_values.values())
        )

    @staticmethod
    def _asset_type_weights(
        positions: List[Any],
    ) -> List[float]:
        """Calculate asset-type allocation weights."""

        asset_values: Dict[str, float] = {}

        for position in positions:
            asset_type = (
                str(position.asset_type).strip()
                if position.asset_type
                else "UNKNOWN"
            )

            value = (
                DiversificationAnalyzer._safe_float(
                    position.market_value
                )
            )

            asset_values[asset_type] = (
                asset_values.get(asset_type, 0.0)
                + value
            )

        return DiversificationAnalyzer._normalize_values(
            list(asset_values.values())
        )

    @staticmethod
    def _normalize_values(
        values: List[float],
    ) -> List[float]:
        """Convert values into normalized weights."""

        positive_values = [
            float(value)
            for value in values
            if float(value) > 0
        ]

        total = sum(positive_values)

        if total <= 0:
            return []

        return [
            value / total
            for value in positive_values
        ]

    # ------------------------------------------------------------------
    # HHI
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_hhi(
        weights: List[float],
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index.

        HHI range:

            1/N  -> perfectly equal N allocations
            1.0   -> completely concentrated

        Example:

            [0.5, 0.5]

            HHI = 0.25 + 0.25
                = 0.50
        """

        if not weights:
            return 0.0

        return sum(
            weight * weight
            for weight in weights
        )

    @staticmethod
    def _hhi_to_score(
        hhi: float,
    ) -> float:
        """
        Convert HHI to a 0-100 diversification score.

        Lower HHI = higher diversification score.
        """

        if hhi <= 0:
            return 0.0

        score = (
            1.0 - hhi
        ) * 100

        return max(
            0.0,
            min(100.0, score),
        )

    @staticmethod
    def _calculate_overall_score(
        position_score: float,
        sector_score: float,
        asset_type_score: float,
    ) -> float:
        """
        Calculate weighted overall diversification score.

        Position diversification receives the highest weight,
        followed by sector and asset type diversification.
        """

        return (
            position_score * 0.50
            + sector_score * 0.35
            + asset_type_score * 0.15
        )

    # ------------------------------------------------------------------
    # Largest allocations
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_percentage(
        weights: List[float],
    ) -> float:
        """Return largest allocation percentage."""

        if not weights:
            return 0.0

        return max(weights) * 100

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_concentration_level(
        largest_position: float,
        largest_sector: float,
        position_hhi: float,
        sector_hhi: float,
    ) -> str:
        """
        Determine descriptive concentration level.
        """

        if (
            largest_position >= 50
            or largest_sector >= 60
            or position_hhi >= 0.50
            or sector_hhi >= 0.60
        ):
            return "HIGH"

        if (
            largest_position >= 30
            or largest_sector >= 40
            or position_hhi >= 0.30
            or sector_hhi >= 0.40
        ):
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _determine_diversification_status(
        score: float,
    ) -> str:
        """Classify overall diversification."""

        if score >= 75:
            return "WELL_DIVERSIFIED"

        if score >= 50:
            return "MODERATELY_DIVERSIFIED"

        if score >= 25:
            return "LOW_DIVERSIFICATION"

        return "HIGHLY_CONCENTRATED"

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warnings(
        largest_position: float,
        largest_sector: float,
        largest_asset_type: float,
        position_hhi: float,
        sector_hhi: float,
    ) -> List[str]:
        """
        Build descriptive concentration warnings.
        """

        warnings: List[str] = []

        if largest_position >= 50:
            warnings.append(
                "A single position represents "
                "50% or more of the invested portfolio."
            )
        elif largest_position >= 30:
            warnings.append(
                "The largest position represents "
                "30% or more of the invested portfolio."
            )

        if largest_sector >= 60:
            warnings.append(
                "A single sector represents "
                "60% or more of the invested portfolio."
            )
        elif largest_sector >= 40:
            warnings.append(
                "The largest sector represents "
                "40% or more of the invested portfolio."
            )

        if largest_asset_type >= 80:
            warnings.append(
                "A single asset type represents "
                "80% or more of the invested portfolio."
            )

        if position_hhi >= 0.50:
            warnings.append(
                "Position-level concentration is high "
                "according to HHI."
            )
        elif position_hhi >= 0.30:
            warnings.append(
                "Position-level concentration is moderate "
                "according to HHI."
            )

        if sector_hhi >= 0.60:
            warnings.append(
                "Sector-level concentration is high "
                "according to HHI."
            )
        elif sector_hhi >= 0.40:
            warnings.append(
                "Sector-level concentration is moderate "
                "according to HHI."
            )

        return warnings

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