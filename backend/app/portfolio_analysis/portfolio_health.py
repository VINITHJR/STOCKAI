"""
Phase 13 - Layer 6
Portfolio Health Score

Combines portfolio performance, diversification, allocation,
attribution, and cash structure into a single descriptive
portfolio health score.

This layer does NOT:
- modify Phase 11 Paper Trading
- modify Phase 12 Risk Management
- modify Phase 13 Layers 1-5
- execute trades
- block trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .allocation_analysis import AllocationAnalyzer
from .diversification_analysis import DiversificationAnalyzer
from .performance_analysis import PerformanceAnalyzer
from .position_sector_attribution import (
    PositionSectorAttributionAnalyzer,
)


@dataclass
class PortfolioHealthAnalysis:
    """
    Complete Phase 13 Layer 6 health analysis.
    """

    health_score: float
    health_status: str

    performance_score: float
    diversification_score: float
    allocation_score: float
    attribution_score: float
    cash_score: float

    performance_weight: float
    diversification_weight: float
    allocation_weight: float
    attribution_weight: float
    cash_weight: float

    strengths: list[str] = field(
        default_factory=list
    )

    concerns: list[str] = field(
        default_factory=list
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-compatible dictionary."""

        return {
            "health_score": self.health_score,
            "health_status": self.health_status,
            "performance_score": self.performance_score,
            "diversification_score": (
                self.diversification_score
            ),
            "allocation_score": self.allocation_score,
            "attribution_score": self.attribution_score,
            "cash_score": self.cash_score,
            "performance_weight": (
                self.performance_weight
            ),
            "diversification_weight": (
                self.diversification_weight
            ),
            "allocation_weight": (
                self.allocation_weight
            ),
            "attribution_weight": (
                self.attribution_weight
            ),
            "cash_weight": self.cash_weight,
            "strengths": self.strengths,
            "concerns": self.concerns,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class PortfolioHealthAnalyzer:
    """
    Phase 13 Layer 6 Portfolio Health Analyzer.

    The health score is composed from five dimensions:

        Performance       30%
        Diversification   25%
        Allocation        20%
        Attribution      15%
        Cash              10%

    Total:

        30 + 25 + 20 + 15 + 10 = 100%

    The score is descriptive and should not be interpreted
    as a trading signal or risk-limit decision.
    """

    PERFORMANCE_WEIGHT = 0.30
    DIVERSIFICATION_WEIGHT = 0.25
    ALLOCATION_WEIGHT = 0.20
    ATTRIBUTION_WEIGHT = 0.15
    CASH_WEIGHT = 0.10

    def __init__(
        self,
        currency: str = "INR",
    ) -> None:
        self.currency = currency

        self.performance_analyzer = (
            PerformanceAnalyzer(
                currency=currency
            )
        )

        self.allocation_analyzer = (
            AllocationAnalyzer(
                currency=currency
            )
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
        realized_pnl: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> PortfolioHealthAnalysis:
        """
        Calculate overall portfolio health.

        Parameters
        ----------
        positions:
            Portfolio position dictionaries.

        cash:
            Current available cash.

        initial_capital:
            Original portfolio capital.

        realized_pnl:
            Explicitly supplied realized P&L.

        timestamp:
            Optional analysis timestamp.
        """

        position_data = list(
            positions or []
        )

        performance = (
            self.performance_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                realized_pnl=realized_pnl,
                timestamp=timestamp,
            )
        )

        allocation = (
            self.allocation_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        diversification = (
            self.diversification_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        attribution = (
            self.attribution_analyzer.analyze(
                positions=position_data,
                cash=cash,
                initial_capital=initial_capital,
                timestamp=timestamp,
            )
        )

        performance_score = (
            self._calculate_performance_score(
                performance.total_return_percent
            )
        )

        diversification_score = (
            self._clamp_score(
                diversification.overall_diversification_score
            )
        )

        allocation_score = (
            self._calculate_allocation_score(
                allocation_status=(
                    allocation.allocation_status
                ),
                invested_percent=(
                    allocation.invested_percent
                ),
                largest_position_percent=(
                    (
                        allocation.largest_position
                        or {}
                    ).get(
                        "percentage",
                        0.0,
                    )
                ),
            )
        )

        attribution_score = (
            self._calculate_attribution_score(
                total_pnl=performance.total_pnl,
                positive_positions=(
                    attribution.positive_position_count
                ),
                negative_positions=(
                    attribution.negative_position_count
                ),
            )
        )

        cash_score = (
            self._calculate_cash_score(
                cash_percent=allocation.cash_percent
            )
        )

        health_score = (
            performance_score
            * self.PERFORMANCE_WEIGHT
            + diversification_score
            * self.DIVERSIFICATION_WEIGHT
            + allocation_score
            * self.ALLOCATION_WEIGHT
            + attribution_score
            * self.ATTRIBUTION_WEIGHT
            + cash_score
            * self.CASH_WEIGHT
        )

        health_score = self._clamp_score(
            health_score
        )

        health_status = (
            self._determine_health_status(
                health_score
            )
        )

        strengths = self._build_strengths(
            performance_score=performance_score,
            diversification_score=(
                diversification_score
            ),
            allocation_score=allocation_score,
            attribution_score=attribution_score,
            cash_score=cash_score,
        )

        concerns = self._build_concerns(
            performance=performance,
            allocation=allocation,
            diversification=diversification,
            attribution=attribution,
            performance_score=performance_score,
            diversification_score=(
                diversification_score
            ),
            allocation_score=allocation_score,
            attribution_score=attribution_score,
            cash_score=cash_score,
        )

        recommendations = (
            self._build_recommendations(
                allocation=allocation,
                diversification=diversification,
                performance=performance,
                attribution=attribution,
            )
        )

        return PortfolioHealthAnalysis(
            health_score=self._round(
                health_score
            ),
            health_status=health_status,
            performance_score=self._round(
                performance_score
            ),
            diversification_score=self._round(
                diversification_score
            ),
            allocation_score=self._round(
                allocation_score
            ),
            attribution_score=self._round(
                attribution_score
            ),
            cash_score=self._round(
                cash_score
            ),
            performance_weight=(
                self.PERFORMANCE_WEIGHT
            ),
            diversification_weight=(
                self.DIVERSIFICATION_WEIGHT
            ),
            allocation_weight=(
                self.ALLOCATION_WEIGHT
            ),
            attribution_weight=(
                self.ATTRIBUTION_WEIGHT
            ),
            cash_weight=self.CASH_WEIGHT,
            strengths=strengths,
            concerns=concerns,
            recommendations=recommendations,
            metadata={
                "currency": self.currency,
                "layer": 6,
                "component": "portfolio_health",
                "score_range": "0_to_100",
                "risk_control": False,
            },
        )

    # ------------------------------------------------------------------
    # Performance score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_performance_score(
        return_percent: float,
    ) -> float:
        """
        Convert portfolio return into a health score.

        Mapping:

            +10% or higher  -> 100
            +5%             -> 80
             0%             -> 60
            -5%             -> 40
            -10% or lower   -> 20

        Values between these points are interpolated.
        """

        points = [
            (-10.0, 20.0),
            (-5.0, 40.0),
            (0.0, 60.0),
            (5.0, 80.0),
            (10.0, 100.0),
        ]

        return PortfolioHealthAnalyzer._interpolate(
            return_percent,
            points,
        )

    # ------------------------------------------------------------------
    # Allocation score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_allocation_score(
        allocation_status: str,
        invested_percent: float,
        largest_position_percent: float,
    ) -> float:
        """
        Calculate allocation health score.

        This is descriptive and is not a Phase 12 risk limit.
        """

        status_scores = {
            "BALANCED": 90.0,
            "CASH_HEAVY": 70.0,
            "HIGHLY_INVESTED": 65.0,
            "FULLY_CASH": 50.0,
            "SECTOR_CONCENTRATED": 45.0,
            "HIGHLY_CONCENTRATED": 25.0,
        }

        score = status_scores.get(
            allocation_status,
            60.0,
        )

        if largest_position_percent >= 70:
            score -= 20

        elif largest_position_percent >= 50:
            score -= 10

        return max(
            0.0,
            min(100.0, score),
        )

    # ------------------------------------------------------------------
    # Attribution score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_attribution_score(
        total_pnl: float,
        positive_positions: int,
        negative_positions: int,
    ) -> float:
        """
        Calculate attribution health.

        More positive contributors and positive total P&L
        increase the score.
        """

        total_positions = (
            positive_positions
            + negative_positions
        )

        if total_positions == 0:
            return 50.0

        positive_ratio = (
            positive_positions
            / total_positions
        )

        score = 50.0 + (
            positive_ratio
            - 0.5
        ) * 60.0

        if total_pnl > 0:
            score += 10.0

        elif total_pnl < 0:
            score -= 10.0

        return max(
            0.0,
            min(100.0, score),
        )

    # ------------------------------------------------------------------
    # Cash score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_cash_score(
        cash_percent: float,
    ) -> float:
        """
        Evaluate cash allocation.

        This does not determine whether a cash level is
        financially suitable for a specific user.

        It only measures portfolio structure.
        """

        if cash_percent <= 10:
            return 65.0

        if cash_percent <= 25:
            return 90.0

        if cash_percent <= 40:
            return 85.0

        if cash_percent <= 50:
            return 75.0

        if cash_percent <= 70:
            return 60.0

        return 45.0

    # ------------------------------------------------------------------
    # Health classification
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_health_status(
        score: float,
    ) -> str:
        """Convert score into health status."""

        if score >= 85:
            return "EXCELLENT"

        if score >= 70:
            return "GOOD"

        if score >= 50:
            return "FAIR"

        if score >= 30:
            return "WEAK"

        return "CRITICAL"

    # ------------------------------------------------------------------
    # Strengths
    # ------------------------------------------------------------------

    @staticmethod
    def _build_strengths(
        performance_score: float,
        diversification_score: float,
        allocation_score: float,
        attribution_score: float,
        cash_score: float,
    ) -> list[str]:
        """Build descriptive strengths."""

        strengths: list[str] = []

        if performance_score >= 70:
            strengths.append(
                "Portfolio performance is positive or stable."
            )

        if diversification_score >= 70:
            strengths.append(
                "Portfolio diversification is strong."
            )

        if allocation_score >= 70:
            strengths.append(
                "Portfolio allocation structure is healthy."
            )

        if attribution_score >= 70:
            strengths.append(
                "A healthy proportion of positions are contributing positively."
            )

        if cash_score >= 70:
            strengths.append(
                "Cash allocation provides a balanced portfolio structure."
            )

        if not strengths:
            strengths.append(
                "Portfolio has no major strength identified by the current scoring model."
            )

        return strengths

    # ------------------------------------------------------------------
    # Concerns
    # ------------------------------------------------------------------

    @staticmethod
    def _build_concerns(
        performance: Any,
        allocation: Any,
        diversification: Any,
        attribution: Any,
        performance_score: float,
        diversification_score: float,
        allocation_score: float,
        attribution_score: float,
        cash_score: float,
    ) -> list[str]:
        """Build descriptive portfolio concerns."""

        concerns: list[str] = []

        if performance_score < 50:
            concerns.append(
                "Portfolio performance is currently negative."
            )

        if diversification_score < 50:
            concerns.append(
                "Portfolio diversification is low."
            )

        if allocation_score < 50:
            concerns.append(
                "Portfolio allocation is concentrated."
            )

        if attribution_score < 50:
            concerns.append(
                "Negative contributors are materially affecting portfolio performance."
            )

        if cash_score < 50:
            concerns.append(
                "Cash allocation is outside the scoring model's preferred structural range."
            )

        if getattr(
            diversification,
            "concentration_warnings",
            None,
        ):
            concerns.extend(
                diversification.concentration_warnings
            )

        if not concerns:
            concerns.append(
                "No major structural concerns identified."
            )

        # Remove duplicates while preserving order.
        return list(
            dict.fromkeys(concerns)
        )

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendations(
        allocation: Any,
        diversification: Any,
        performance: Any,
        attribution: Any,
    ) -> list[str]:
        """
        Generate descriptive recommendations.

        These are informational and are not trading instructions.
        """

        recommendations: list[str] = []

        if (
            diversification.overall_diversification_score
            < 50
        ):
            recommendations.append(
                "Review concentration across positions and sectors."
            )

        if (
            allocation.largest_position
            and allocation.largest_position.get(
                "percentage",
                0.0,
            ) >= 40
        ):
            recommendations.append(
                "Review the portfolio's largest position concentration."
            )

        if (
            performance.total_return_percent
            < 0
        ):
            recommendations.append(
                "Review the positions contributing most to portfolio losses."
            )

        if (
            attribution.negative_position_count
            > attribution.positive_position_count
        ):
            recommendations.append(
                "Review the negative P&L contributors and their portfolio weights."
            )

        if not recommendations:
            recommendations.append(
                "Continue monitoring portfolio performance and diversification."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(
        value: float,
        points: list[tuple[float, float]],
    ) -> float:
        """
        Linearly interpolate a value between score points.
        """

        if not points:
            return 0.0

        if value <= points[0][0]:
            return points[0][1]

        if value >= points[-1][0]:
            return points[-1][1]

        for index in range(
            1,
            len(points),
        ):
            x1, y1 = points[index - 1]
            x2, y2 = points[index]

            if value <= x2:
                ratio = (
                    value - x1
                ) / (
                    x2 - x1
                )

                return y1 + (
                    ratio
                    * (y2 - y1)
                )

        return points[-1][1]

    @staticmethod
    def _clamp_score(
        value: float,
    ) -> float:
        """Keep score within 0-100."""

        return max(
            0.0,
            min(
                100.0,
                float(value),
            ),
        )

    @staticmethod
    def _round(
        value: float,
    ) -> float:
        """Round numeric values."""

        return round(
            float(value),
            4,
        )