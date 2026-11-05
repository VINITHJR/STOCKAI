# ============================================================
# PHASE 14.3 — FUNDAMENTAL INTELLIGENCE
# ============================================================
#
# Purpose:
#   Independent deterministic fundamental analysis based on
#   the fundamental fields already exposed by StockAI's
#   existing Upstox fundamentals service.
#
# Important:
#   - No ML models
#   - No technical indicators
#   - No stock-regime calculations
#   - No Phase 9 model outputs
#   - No BUY/HOLD/SELL decision
#   - Missing values are ignored, not treated as bearish
#
# ============================================================

from __future__ import annotations

import math
from typing import List, Optional, Tuple

try:
    from .fundamental_intelligence_models import (
        FundamentalAnalysisInput,
        FundamentalAnalysisResult,
    )
except ImportError:
    from fundamental_intelligence_models import (
        FundamentalAnalysisInput,
        FundamentalAnalysisResult,
    )


class FundamentalIntelligence:
    """
    Independent deterministic fundamental intelligence engine.

    The thresholds used here are intentionally simple v1
    heuristics because the current backend does not provide
    sector-relative valuation benchmarks.

    This engine produces:
        bullish
        bearish
        neutral

    It does NOT produce:
        BUY
        HOLD
        SELL
    """

    METHOD = (
        "deterministic_fundamental_intelligence_v1"
    )

    # Metrics actually available from the current
    # fundamentals_service output.

    TOTAL_METRICS = 9

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _valid(value: Optional[float]) -> bool:
        if value is None:
            return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        return math.isfinite(value)

    @staticmethod
    def _clip(
        value: float,
        minimum: float = -1.0,
        maximum: float = 1.0,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    @staticmethod
    def _assessment(score: float) -> str:

        if score >= 0.50:
            return "STRONG"

        if score >= 0.15:
            return "POSITIVE"

        if score <= -0.50:
            return "WEAK"

        if score <= -0.15:
            return "NEGATIVE"

        return "NEUTRAL"

    # ========================================================
    # Metric scoring
    # ========================================================

    @staticmethod
    def _score_pe(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value <= 15:
            return 1.0

        if value <= 20:
            return 0.5

        if value <= 25:
            return 0.0

        if value <= 35:
            return -0.5

        return -1.0

    @staticmethod
    def _score_pb(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value <= 2:
            return 1.0

        if value <= 3:
            return 0.5

        if value <= 5:
            return 0.0

        if value <= 8:
            return -0.5

        return -1.0

    @staticmethod
    def _score_ev_ebitda(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value <= 10:
            return 1.0

        if value <= 15:
            return 0.5

        if value <= 20:
            return 0.0

        if value <= 30:
            return -0.5

        return -1.0

    @staticmethod
    def _score_roe(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value >= 20:
            return 1.0

        if value >= 15:
            return 0.5

        if value >= 8:
            return 0.0

        if value >= 0:
            return -0.5

        return -1.0

    @staticmethod
    def _score_roa(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value >= 10:
            return 1.0

        if value >= 7:
            return 0.5

        if value >= 3:
            return 0.0

        if value >= 0:
            return -0.5

        return -1.0

    @staticmethod
    def _score_roce(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value >= 20:
            return 1.0

        if value >= 15:
            return 0.5

        if value >= 8:
            return 0.0

        if value >= 0:
            return -0.5

        return -1.0

    @staticmethod
    def _score_quick_ratio(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value >= 1.5:
            return 1.0

        if value >= 1.0:
            return 0.5

        if value >= 0.7:
            return 0.0

        if value >= 0.5:
            return -0.5

        return -1.0

    @staticmethod
    def _score_debt_to_equity(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value <= 0.5:
            return 1.0

        if value <= 1.0:
            return 0.5

        if value <= 2.0:
            return 0.0

        if value <= 3.0:
            return -0.5

        return -1.0

    @staticmethod
    def _score_dividend_yield(
        value: Optional[float],
    ) -> Optional[float]:

        if not FundamentalIntelligence._valid(value):
            return None

        value = float(value)

        if value >= 4.0:
            return 1.0

        if value >= 2.0:
            return 0.5

        if value >= 0.5:
            return 0.0

        if value >= 0:
            return -0.25

        return -1.0

    # ========================================================
    # Group scoring
    # ========================================================

    @classmethod
    def _average(
        cls,
        values: List[Optional[float]],
    ) -> Tuple[float, int]:

        valid_values = [
            float(value)
            for value in values
            if value is not None
            and cls._valid(value)
        ]

        if not valid_values:
            return 0.0, 0

        return (
            sum(valid_values) / len(valid_values),
            len(valid_values),
        )

    # ========================================================
    # Main analysis
    # ========================================================

    def analyze(
        self,
        data: FundamentalAnalysisInput,
    ) -> FundamentalAnalysisResult:

        # ----------------------------------------------------
        # Individual metric scores
        # ----------------------------------------------------

        pe_score = self._score_pe(
            data.pe_ratio
        )

        pb_score = self._score_pb(
            data.pb_ratio
        )

        ev_score = self._score_ev_ebitda(
            data.ev_ebitda
        )

        roe_score = self._score_roe(
            data.roe
        )

        roa_score = self._score_roa(
            data.roa
        )

        roce_score = self._score_roce(
            data.roce
        )

        quick_score = self._score_quick_ratio(
            data.quick_ratio
        )

        debt_score = self._score_debt_to_equity(
            data.debt_to_equity
        )

        dividend_score = self._score_dividend_yield(
            data.dividend_yield
        )

        # ----------------------------------------------------
        # Category scores
        # ----------------------------------------------------

        valuation_score, valuation_count = (
            self._average(
                [
                    pe_score,
                    pb_score,
                    ev_score,
                ]
            )
        )

        profitability_score, profitability_count = (
            self._average(
                [
                    roe_score,
                    roa_score,
                    roce_score,
                ]
            )
        )

        financial_health_score, health_count = (
            self._average(
                [
                    quick_score,
                    debt_score,
                ]
            )
        )

        shareholder_return_score, return_count = (
            self._average(
                [
                    dividend_score,
                ]
            )
        )

        # ----------------------------------------------------
        # Weighted overall score
        #
        # Valuation          30%
        # Profitability      35%
        # Financial health   25%
        # Shareholder return 10%
        #
        # Missing categories do not contribute as bearish.
        # The weights are renormalized over available
        # categories.
        # ----------------------------------------------------

        category_values = []

        if valuation_count > 0:
            category_values.append(
                (valuation_score, 0.30)
            )

        if profitability_count > 0:
            category_values.append(
                (profitability_score, 0.35)
            )

        if health_count > 0:
            category_values.append(
                (financial_health_score, 0.25)
            )

        if return_count > 0:
            category_values.append(
                (shareholder_return_score, 0.10)
            )

        if category_values:

            weighted_total = sum(
                score * weight
                for score, weight
                in category_values
            )

            total_weight = sum(
                weight
                for _, weight
                in category_values
            )

            fundamental_score = (
                weighted_total / total_weight
                if total_weight > 0
                else 0.0
            )

        else:

            fundamental_score = 0.0

        fundamental_score = self._clip(
            fundamental_score
        )

        # ----------------------------------------------------
        # Data completeness
        # ----------------------------------------------------

        available_metric_count = sum(
            1
            for value in (
                data.pe_ratio,
                data.pb_ratio,
                data.ev_ebitda,
                data.roe,
                data.roa,
                data.roce,
                data.quick_ratio,
                data.debt_to_equity,
                data.dividend_yield,
            )
            if self._valid(value)
        )

        data_completeness = (
            available_metric_count
            / self.TOTAL_METRICS
        )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if available_metric_count == 0:

            signal = "neutral"

        elif fundamental_score >= 0.20:

            signal = "bullish"

        elif fundamental_score <= -0.20:

            signal = "bearish"

        else:

            signal = "neutral"

        # ----------------------------------------------------
        # Confidence
        #
        # Confidence depends on:
        #   1. data completeness
        #   2. directional strength
        #
        # This deliberately does not turn missing data into
        # negative evidence.
        # ----------------------------------------------------

        directional_strength = abs(
            fundamental_score
        )

        if available_metric_count == 0:

            confidence = 0.0

        else:

            confidence = (
                0.35 * data_completeness
                + 0.65 * directional_strength
            )

            confidence = self._clip(
                confidence,
                0.0,
                1.0,
            )

        # ----------------------------------------------------
        # Assessments
        # ----------------------------------------------------

        valuation_assessment = (
            self._assessment(
                valuation_score
            )
            if valuation_count > 0
            else "UNAVAILABLE"
        )

        profitability_assessment = (
            self._assessment(
                profitability_score
            )
            if profitability_count > 0
            else "UNAVAILABLE"
        )

        financial_health_assessment = (
            self._assessment(
                financial_health_score
            )
            if health_count > 0
            else "UNAVAILABLE"
        )

        shareholder_return_assessment = (
            self._assessment(
                shareholder_return_score
            )
            if return_count > 0
            else "UNAVAILABLE"
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        bullish_evidence: List[str] = []
        bearish_evidence: List[str] = []
        neutral_evidence: List[str] = []
        warnings: List[str] = []

        # Valuation evidence

        if pe_score is not None:

            if pe_score > 0:
                bullish_evidence.append(
                    f"P/E of {data.pe_ratio:.2f} "
                    "is favorable under the v1 valuation thresholds."
                )

            elif pe_score < 0:
                bearish_evidence.append(
                    f"P/E of {data.pe_ratio:.2f} "
                    "is elevated under the v1 valuation thresholds."
                )

            else:
                neutral_evidence.append(
                    f"P/E of {data.pe_ratio:.2f} "
                    "falls in the neutral valuation range."
                )

        if pb_score is not None:

            if pb_score > 0:
                bullish_evidence.append(
                    f"P/B of {data.pb_ratio:.2f} "
                    "is favorable under the v1 valuation thresholds."
                )

            elif pb_score < 0:
                bearish_evidence.append(
                    f"P/B of {data.pb_ratio:.2f} "
                    "is elevated under the v1 valuation thresholds."
                )

            else:
                neutral_evidence.append(
                    f"P/B of {data.pb_ratio:.2f} "
                    "falls in the neutral valuation range."
                )

        if ev_score is not None:

            if ev_score > 0:
                bullish_evidence.append(
                    f"EV/EBITDA of {data.ev_ebitda:.2f} "
                    "is favorable under the v1 valuation thresholds."
                )

            elif ev_score < 0:
                bearish_evidence.append(
                    f"EV/EBITDA of {data.ev_ebitda:.2f} "
                    "is elevated under the v1 valuation thresholds."
                )

            else:
                neutral_evidence.append(
                    f"EV/EBITDA of {data.ev_ebitda:.2f} "
                    "falls in the neutral valuation range."
                )

        # Profitability evidence

        if roe_score is not None:

            if roe_score > 0:
                bullish_evidence.append(
                    f"ROE of {data.roe:.2f}% "
                    "supports profitability."
                )

            elif roe_score < 0:
                bearish_evidence.append(
                    f"ROE of {data.roe:.2f}% "
                    "is weak under the v1 thresholds."
                )

        if roa_score is not None:

            if roa_score > 0:
                bullish_evidence.append(
                    f"ROA of {data.roa:.2f}% "
                    "supports asset profitability."
                )

            elif roa_score < 0:
                bearish_evidence.append(
                    f"ROA of {data.roa:.2f}% "
                    "is weak under the v1 thresholds."
                )

        if roce_score is not None:

            if roce_score > 0:
                bullish_evidence.append(
                    f"ROCE of {data.roce:.2f}% "
                    "supports capital efficiency."
                )

            elif roce_score < 0:
                bearish_evidence.append(
                    f"ROCE of {data.roce:.2f}% "
                    "is weak under the v1 thresholds."
                )

        # Financial health

        if quick_score is not None:

            if quick_score > 0:
                bullish_evidence.append(
                    f"Quick ratio of {data.quick_ratio:.2f} "
                    "supports short-term liquidity."
                )

            elif quick_score < 0:
                bearish_evidence.append(
                    f"Quick ratio of {data.quick_ratio:.2f} "
                    "indicates weaker short-term liquidity."
                )

        if debt_score is not None:

            if debt_score > 0:
                bullish_evidence.append(
                    f"Debt-to-equity of "
                    f"{data.debt_to_equity:.2f} "
                    "is favorable under the v1 leverage thresholds."
                )

            elif debt_score < 0:
                bearish_evidence.append(
                    f"Debt-to-equity of "
                    f"{data.debt_to_equity:.2f} "
                    "is elevated under the v1 leverage thresholds."
                )

        # Dividend

        if dividend_score is not None:

            if dividend_score > 0:
                bullish_evidence.append(
                    f"Dividend yield of "
                    f"{data.dividend_yield:.2f}% "
                    "supports shareholder return."
                )

            elif dividend_score < 0:
                bearish_evidence.append(
                    f"Dividend yield of "
                    f"{data.dividend_yield:.2f}% "
                    "provides limited shareholder-return support."
                )

        # ----------------------------------------------------
        # Missing-data warnings
        # ----------------------------------------------------

        if data.pe_ratio is None:
            warnings.append("P/E is unavailable.")

        if data.pb_ratio is None:
            warnings.append("P/B is unavailable.")

        if data.ev_ebitda is None:
            warnings.append("EV/EBITDA is unavailable.")

        if data.roe is None:
            warnings.append("ROE is unavailable.")

        if data.roa is None:
            warnings.append("ROA is unavailable.")

        if data.roce is None:
            warnings.append("ROCE is unavailable.")

        if data.quick_ratio is None:
            warnings.append(
                "Quick ratio is unavailable."
            )

        if data.debt_to_equity is None:
            warnings.append(
                "Debt-to-equity is unavailable."
            )

        if data.dividend_yield is None:
            warnings.append(
                "Dividend yield is unavailable."
            )

        if data_completeness < 0.50:

            warnings.append(
                "Fundamental data completeness is below 50%; "
                "confidence is limited."
            )

        # ----------------------------------------------------
        # Combined evidence
        # ----------------------------------------------------

        evidence = (
            bullish_evidence
            + bearish_evidence
            + neutral_evidence
        )

        return FundamentalAnalysisResult(

            instrument_key=data.instrument_key,

            symbol=data.symbol,

            sector=data.sector,

            signal=signal,

            confidence=round(
                confidence,
                6,
            ),

            fundamental_score=round(
                fundamental_score,
                6,
            ),

            valuation_score=round(
                valuation_score,
                6,
            ),

            profitability_score=round(
                profitability_score,
                6,
            ),

            financial_health_score=round(
                financial_health_score,
                6,
            ),

            shareholder_return_score=round(
                shareholder_return_score,
                6,
            ),

            valuation_assessment=(
                valuation_assessment
            ),

            profitability_assessment=(
                profitability_assessment
            ),

            financial_health_assessment=(
                financial_health_assessment
            ),

            shareholder_return_assessment=(
                shareholder_return_assessment
            ),

            available_metric_count=(
                available_metric_count
            ),

            total_metric_count=(
                self.TOTAL_METRICS
            ),

            data_completeness=round(
                data_completeness,
                6,
            ),

            bullish_evidence=(
                bullish_evidence
            ),

            bearish_evidence=(
                bearish_evidence
            ),

            neutral_evidence=(
                neutral_evidence
            ),

            evidence=evidence,

            warnings=warnings,

            method=self.METHOD,
        )