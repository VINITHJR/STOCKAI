"""
Phase 13 - Layer 3
Allocation Analysis

Analyzes how portfolio capital and market value are distributed
across positions, sectors, asset types, and cash.

This layer is descriptive only.

It does NOT:
- modify Phase 11 Paper Trading
- modify Phase 12 Risk Management
- change Layer 1 Portfolio Analyzer
- change Layer 2 Performance Analysis
- block trades
- execute trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .portfolio_analyzer import PortfolioAnalyzer


@dataclass
class AllocationBucket:
    """
    Represents one allocation category.

    Examples:
        - AAPL position
        - Banking sector
        - EQUITY asset type
        - CASH
    """

    name: str

    value: float
    percentage: float

    position_count: int = 0

    invested_value: float = 0.0
    market_value: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation bucket to JSON-compatible dictionary."""

        return {
            "name": self.name,
            "value": self.value,
            "percentage": self.percentage,
            "position_count": self.position_count,
            "invested_value": self.invested_value,
            "market_value": self.market_value,
            "metadata": self.metadata,
        }


@dataclass
class AllocationAnalysis:
    """
    Complete Layer 3 allocation analysis result.
    """

    total_portfolio_value: float
    invested_value: float
    cash_value: float

    invested_percent: float
    cash_percent: float

    position_allocations: List[AllocationBucket] = field(
        default_factory=list
    )

    sector_allocations: List[AllocationBucket] = field(
        default_factory=list
    )

    asset_type_allocations: List[AllocationBucket] = field(
        default_factory=list
    )

    largest_position: Optional[Dict[str, Any]] = None
    largest_sector: Optional[Dict[str, Any]] = None
    largest_asset_type: Optional[Dict[str, Any]] = None

    allocation_status: str = "BALANCED"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result into JSON-compatible dictionary."""

        return {
            "total_portfolio_value": self.total_portfolio_value,
            "invested_value": self.invested_value,
            "cash_value": self.cash_value,
            "invested_percent": self.invested_percent,
            "cash_percent": self.cash_percent,
            "position_allocations": [
                bucket.to_dict()
                for bucket in self.position_allocations
            ],
            "sector_allocations": [
                bucket.to_dict()
                for bucket in self.sector_allocations
            ],
            "asset_type_allocations": [
                bucket.to_dict()
                for bucket in self.asset_type_allocations
            ],
            "largest_position": self.largest_position,
            "largest_sector": self.largest_sector,
            "largest_asset_type": self.largest_asset_type,
            "allocation_status": self.allocation_status,
            "metadata": self.metadata,
        }


class AllocationAnalyzer:
    """
    Phase 13 Layer 3 allocation analyzer.

    Calculates portfolio allocation across:

        1. Individual positions
        2. Sectors
        3. Asset types
        4. Cash

    The analyzer uses current market value as the primary
    allocation basis.

    This is intentionally independent of Phase 12 risk limits.
    """

    def __init__(self, currency: str = "INR") -> None:
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
    ) -> AllocationAnalysis:
        """
        Analyze portfolio allocation.

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

        Returns
        -------
        AllocationAnalysis
        """

        portfolio = self.portfolio_analyzer.analyze(
            positions=positions,
            cash=cash,
            initial_capital=initial_capital,
            timestamp=timestamp,
        )

        total_value = self._safe_float(
            portfolio.total_value
        )

        invested_value = self._safe_float(
            portfolio.market_value
        )

        cash_value = self._safe_float(
            portfolio.cash
        )

        invested_percent = self._percentage(
            invested_value,
            total_value,
        )

        cash_percent = self._percentage(
            cash_value,
            total_value,
        )

        position_allocations = (
            self._build_position_allocations(
                portfolio.positions,
                total_value,
            )
        )

        sector_allocations = (
            self._build_sector_allocations(
                portfolio.positions,
                total_value,
            )
        )

        asset_type_allocations = (
            self._build_asset_type_allocations(
                portfolio.positions,
                total_value,
            )
        )

        largest_position = self._largest_bucket(
            position_allocations
        )

        largest_sector = self._largest_bucket(
            sector_allocations
        )

        largest_asset_type = self._largest_bucket(
            asset_type_allocations
        )

        allocation_status = (
            self._determine_allocation_status(
                largest_position=largest_position,
                largest_sector=largest_sector,
                invested_percent=invested_percent,
                cash_percent=cash_percent,
            )
        )

        return AllocationAnalysis(
            total_portfolio_value=self._round(
                total_value
            ),
            invested_value=self._round(
                invested_value
            ),
            cash_value=self._round(
                cash_value
            ),
            invested_percent=self._round(
                invested_percent
            ),
            cash_percent=self._round(
                cash_percent
            ),
            position_allocations=position_allocations,
            sector_allocations=sector_allocations,
            asset_type_allocations=asset_type_allocations,
            largest_position=largest_position,
            largest_sector=largest_sector,
            largest_asset_type=largest_asset_type,
            allocation_status=allocation_status,
            metadata={
                "currency": self.currency,
                "layer": 3,
                "component": "allocation_analysis",
                "allocation_basis": "market_value",
            },
        )

    # ------------------------------------------------------------------
    # Position allocation
    # ------------------------------------------------------------------

    def _build_position_allocations(
        self,
        positions: List[Any],
        total_value: float,
    ) -> List[AllocationBucket]:
        """
        Build allocation for every individual position.
        """

        allocations: List[AllocationBucket] = []

        for position in positions:
            market_value = self._safe_float(
                position.market_value
            )

            invested_value = self._safe_float(
                position.invested_value
            )

            percentage = self._percentage(
                market_value,
                total_value,
            )

            allocations.append(
                AllocationBucket(
                    name=(
                        position.symbol
                        or position.instrument_key
                        or "UNKNOWN"
                    ),
                    value=self._round(market_value),
                    percentage=self._round(percentage),
                    position_count=1,
                    invested_value=self._round(
                        invested_value
                    ),
                    market_value=self._round(
                        market_value
                    ),
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

        return self._sort_allocations(
            allocations
        )

    # ------------------------------------------------------------------
    # Sector allocation
    # ------------------------------------------------------------------

    def _build_sector_allocations(
        self,
        positions: List[Any],
        total_value: float,
    ) -> List[AllocationBucket]:
        """
        Aggregate portfolio allocation by sector.
        """

        buckets: Dict[str, Dict[str, float]] = {}

        for position in positions:
            sector = (
                str(position.sector).strip()
                if position.sector
                else "UNKNOWN"
            )

            market_value = self._safe_float(
                position.market_value
            )

            invested_value = self._safe_float(
                position.invested_value
            )

            if sector not in buckets:
                buckets[sector] = {
                    "market_value": 0.0,
                    "invested_value": 0.0,
                    "position_count": 0,
                }

            buckets[sector]["market_value"] += market_value
            buckets[sector]["invested_value"] += invested_value
            buckets[sector]["position_count"] += 1

        allocations: List[AllocationBucket] = []

        for sector, values in buckets.items():
            market_value = values["market_value"]

            allocations.append(
                AllocationBucket(
                    name=sector,
                    value=self._round(
                        market_value
                    ),
                    percentage=self._round(
                        self._percentage(
                            market_value,
                            total_value,
                        )
                    ),
                    position_count=int(
                        values["position_count"]
                    ),
                    invested_value=self._round(
                        values["invested_value"]
                    ),
                    market_value=self._round(
                        market_value
                    ),
                )
            )

        return self._sort_allocations(
            allocations
        )

    # ------------------------------------------------------------------
    # Asset type allocation
    # ------------------------------------------------------------------

    def _build_asset_type_allocations(
        self,
        positions: List[Any],
        total_value: float,
    ) -> List[AllocationBucket]:
        """
        Aggregate portfolio allocation by asset type.
        """

        buckets: Dict[str, Dict[str, float]] = {}

        for position in positions:
            asset_type = (
                str(position.asset_type).strip()
                if position.asset_type
                else "UNKNOWN"
            )

            market_value = self._safe_float(
                position.market_value
            )

            invested_value = self._safe_float(
                position.invested_value
            )

            if asset_type not in buckets:
                buckets[asset_type] = {
                    "market_value": 0.0,
                    "invested_value": 0.0,
                    "position_count": 0,
                }

            buckets[asset_type]["market_value"] += market_value
            buckets[asset_type]["invested_value"] += invested_value
            buckets[asset_type]["position_count"] += 1

        allocations: List[AllocationBucket] = []

        for asset_type, values in buckets.items():
            market_value = values["market_value"]

            allocations.append(
                AllocationBucket(
                    name=asset_type,
                    value=self._round(
                        market_value
                    ),
                    percentage=self._round(
                        self._percentage(
                            market_value,
                            total_value,
                        )
                    ),
                    position_count=int(
                        values["position_count"]
                    ),
                    invested_value=self._round(
                        values["invested_value"]
                    ),
                    market_value=self._round(
                        market_value
                    ),
                )
            )

        return self._sort_allocations(
            allocations
        )

    # ------------------------------------------------------------------
    # Largest allocation
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_bucket(
        buckets: List[AllocationBucket],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the largest allocation bucket.
        """

        if not buckets:
            return None

        largest = max(
            buckets,
            key=lambda bucket: bucket.value,
        )

        return {
            "name": largest.name,
            "value": largest.value,
            "percentage": largest.percentage,
            "position_count": (
                largest.position_count
            ),
        }

    # ------------------------------------------------------------------
    # Allocation status
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_allocation_status(
        largest_position: Optional[Dict[str, Any]],
        largest_sector: Optional[Dict[str, Any]],
        invested_percent: float,
        cash_percent: float,
    ) -> str:
        """
        Determine descriptive allocation status.

        These thresholds are descriptive only and are NOT
        Phase 12 risk limits.
        """

        if (
            largest_position
            and largest_position["percentage"] >= 50
        ):
            return "HIGHLY_CONCENTRATED"

        if (
            largest_sector
            and largest_sector["percentage"] >= 60
        ):
            return "SECTOR_CONCENTRATED"

        if invested_percent >= 95:
            return "HIGHLY_INVESTED"

        if cash_percent >= 50:
            return "CASH_HEAVY"

        if invested_percent == 0:
            return "FULLY_CASH"

        return "BALANCED"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    def _safe_float(
        value: Any,
    ) -> float:
        """Safely convert value to finite float."""

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

    @staticmethod
    def _sort_allocations(
        allocations: List[AllocationBucket],
    ) -> List[AllocationBucket]:
        """Sort allocations from largest to smallest."""

        return sorted(
            allocations,
            key=lambda bucket: bucket.value,
            reverse=True,
        )