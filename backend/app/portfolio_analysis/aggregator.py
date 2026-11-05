"""
Phase 13 - Layer 7
Portfolio Analysis Aggregator

Combines the outputs of Phase 13 Layers 1-6 into a single
portfolio analysis response.

This layer does not modify any previous layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from .portfolio_analyzer import PortfolioAnalyzer
from .performance_analysis import PerformanceAnalyzer
from .allocation_analysis import AllocationAnalyzer
from .diversification_analysis import DiversificationAnalyzer
from .position_sector_attribution import (
    PositionSectorAttributionAnalyzer,
)
from .portfolio_health import PortfolioHealthAnalyzer


@dataclass
class PortfolioAnalysisResult:
    """
    Final aggregated Phase 13 portfolio analysis.
    """

    portfolio: Dict[str, Any]
    performance: Dict[str, Any]
    allocation: Dict[str, Any]
    diversification: Dict[str, Any]
    attribution: Dict[str, Any]
    health: Dict[str, Any]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible dictionary."""

        return {
            "portfolio": self.portfolio,
            "performance": self.performance,
            "allocation": self.allocation,
            "diversification": self.diversification,
            "attribution": self.attribution,
            "health": self.health,
            "summary": self.summary,
            "metadata": self.metadata,
        }


class PortfolioAnalysisAggregator:
    """
    Phase 13 Layer 7 aggregator.

    It orchestrates Layers 1-6 and returns one unified
    portfolio analysis response.
    """

    def __init__(
        self,
        currency: str = "INR",
    ) -> None:
        self.currency = currency

        self.portfolio_analyzer = PortfolioAnalyzer(
            currency=currency
        )

        self.performance_analyzer = PerformanceAnalyzer(
            currency=currency
        )

        self.allocation_analyzer = AllocationAnalyzer(
            currency=currency
        )

        self.diversification_analyzer = (
            DiversificationAnalyzer(
                currency=currency
            )
        )

        self.attribution_analyzer = (
            PositionSectorAttributionAnalyzer(
                currency=currency
            )
        )

        self.health_analyzer = PortfolioHealthAnalyzer(
            currency=currency
        )

    def analyze(
        self,
        positions: Optional[
            Iterable[Mapping[str, Any]]
        ] = None,
        cash: float = 0.0,
        initial_capital: float = 0.0,
        realized_pnl: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> PortfolioAnalysisResult:
        """
        Run the complete Phase 13 analysis pipeline.
        """

        position_data = list(
            positions or []
        )

        # --------------------------------------------------------------
        # Layer 1 - Portfolio Analyzer
        # --------------------------------------------------------------

        portfolio_result = (
            self.portfolio_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        portfolio_dict = self._to_dict(
            portfolio_result
        )

        # --------------------------------------------------------------
        # Layer 2 - Performance Analysis
        # --------------------------------------------------------------

        performance_result = (
            self.performance_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                realized_pnl=realized_pnl,
                timestamp=timestamp,
            )
        )

        performance_dict = self._to_dict(
            performance_result
        )

        # --------------------------------------------------------------
        # Layer 3 - Allocation Analysis
        # --------------------------------------------------------------

        allocation_result = (
            self.allocation_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        allocation_dict = self._to_dict(
            allocation_result
        )

        # --------------------------------------------------------------
        # Layer 4 - Diversification Analysis
        # --------------------------------------------------------------

        diversification_result = (
            self.diversification_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        diversification_dict = self._to_dict(
            diversification_result
        )

        # --------------------------------------------------------------
        # Layer 5 - Position & Sector Attribution
        # --------------------------------------------------------------

        attribution_result = (
            self.attribution_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        attribution_dict = self._to_dict(
            attribution_result
        )

        # --------------------------------------------------------------
        # Layer 6 - Portfolio Health
        # --------------------------------------------------------------

        health_result = (
            self.health_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                realized_pnl=realized_pnl,
                timestamp=timestamp,
            )
        )

        health_dict = self._to_dict(
            health_result
        )

        # --------------------------------------------------------------
        # Final summary
        # --------------------------------------------------------------

        summary = self._build_summary(
            portfolio=portfolio_dict,
            performance=performance_dict,
            allocation=allocation_dict,
            diversification=diversification_dict,
            attribution=attribution_dict,
            health=health_dict,
        )

        return PortfolioAnalysisResult(
            portfolio=portfolio_dict,
            performance=performance_dict,
            allocation=allocation_dict,
            diversification=diversification_dict,
            attribution=attribution_dict,
            health=health_dict,
            summary=summary,
            metadata={
                "currency": self.currency,
                "phase": 13,
                "layer": 7,
                "component": (
                    "portfolio_analysis_aggregator"
                ),
                "layers_included": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ],
            },
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        portfolio: Dict[str, Any],
        performance: Dict[str, Any],
        allocation: Dict[str, Any],
        diversification: Dict[str, Any],
        attribution: Dict[str, Any],
        health: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the compact top-level portfolio summary.
        """

        return {
            "position_count": portfolio.get(
                "position_count",
                0,
            ),
            "total_value": portfolio.get(
                "total_value",
                0.0,
            ),
            "total_pnl": portfolio.get(
                "total_pnl",
                0.0,
            ),
            "total_return_percent": performance.get(
                "total_return_percent",
                0.0,
            ),
            "invested_percent": allocation.get(
                "invested_percent",
                0.0,
            ),
            "cash_percent": allocation.get(
                "cash_percent",
                0.0,
            ),
            "diversification_score": (
                diversification.get(
                    "overall_diversification_score",
                    0.0,
                )
            ),
            "health_score": health.get(
                "health_score",
                0.0,
            ),
            "health_status": health.get(
                "health_status",
                "UNKNOWN",
            ),
            "portfolio_status": portfolio.get(
                "status",
                "UNKNOWN",
            ),
            "allocation_status": allocation.get(
                "allocation_status",
                "UNKNOWN",
            ),
            "diversification_status": (
                diversification.get(
                    "diversification_status",
                    "UNKNOWN",
                )
            ),
            "attribution_status": attribution.get(
                "attribution_status",
                "UNKNOWN",
            ),
        }

    # ------------------------------------------------------------------
    # Serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(
        result: Any,
    ) -> Dict[str, Any]:
        """
        Convert supported analysis result objects to dictionaries.
        """

        if hasattr(result, "to_dict"):
            return result.to_dict()

        if hasattr(result, "__dict__"):
            return dict(result.__dict__)

        if isinstance(result, dict):
            return result

        raise TypeError(
            f"Unsupported analysis result type: "
            f"{type(result).__name__}"
        )