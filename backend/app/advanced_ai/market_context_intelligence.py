from .layer4_models import (
    AIMarketContextInput,
    AIMarketContextResult,
)


class MarketContextIntelligence:
    """
    Phase 14 — Layer 4

    Market Context Intelligence evaluates whether the broader
    market and sector environment supports or conflicts with
    the stock-level AI decision.

    IMPORTANT:
    - Does NOT create a new BUY/SELL/HOLD decision.
    - Does NOT override Layer 1.
    - Does NOT recalculate Layer 2 reliability.
    - Provides contextual intelligence for Layer 5.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):
        self.name = "Market Context Intelligence"
        self.layer = 4

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_signal(value: str) -> str:
        """
        Normalize signal/direction values.
        """

        if not value:
            return "neutral"

        value = str(value).strip().lower()

        bullish_values = {
            "bullish",
            "positive",
            "up",
            "strong_bullish",
            "bull"
        }

        bearish_values = {
            "bearish",
            "negative",
            "down",
            "strong_bearish",
            "bear"
        }

        neutral_values = {
            "neutral",
            "sideways",
            "flat",
            "unknown",
            "none"
        }

        if value in bullish_values:
            return "bullish"

        if value in bearish_values:
            return "bearish"

        if value in neutral_values:
            return "neutral"

        return "neutral"

    # ========================================================
    # DECISION DIRECTION
    # ========================================================

    @staticmethod
    def _decision_direction(decision: str) -> str:
        """
        Convert Layer 1 decision into a directional category.
        """

        decision = str(decision).strip().upper()

        if decision == "BUY":
            return "bullish"

        if decision == "SELL":
            return "bearish"

        return "neutral"

    # ========================================================
    # ALIGNMENT
    # ========================================================

    @staticmethod
    def _calculate_alignment(
        decision_direction: str,
        context_direction: str
    ) -> str:
        """
        Compare stock decision direction against market/sector.
        """

        if context_direction == "neutral":
            return "NEUTRAL"

        if decision_direction == "neutral":
            return "NEUTRAL"

        if decision_direction == context_direction:
            return "SUPPORTIVE"

        return "CONFLICTING"

    # ========================================================
    # CONTEXT SCORE
    # ========================================================

    @staticmethod
    def _calculate_context_score(
        decision_direction: str,
        context_direction: str,
        confidence: float
    ) -> float:
        """
        Calculate contextual support.

        Score interpretation:

        0.00 = strongly conflicting
        0.50 = neutral
        1.00 = strongly supportive
        """

        confidence = max(0.0, min(1.0, confidence))

        # Neutral context gives neutral support.
        if context_direction == "neutral":
            return 0.50

        # HOLD does not have a directional bias.
        if decision_direction == "neutral":
            return 0.50

        # Context supports decision.
        if decision_direction == context_direction:
            return round(0.50 + (0.50 * confidence), 4)

        # Context conflicts with decision.
        return round(0.50 - (0.50 * confidence), 4)

    # ========================================================
    # OVERALL ALIGNMENT
    # ========================================================

    @staticmethod
    def _calculate_overall_alignment(
        market_alignment: str,
        sector_alignment: str
    ) -> str:
        """
        Combine market and sector alignment.
        """

        alignments = {
            market_alignment,
            sector_alignment
        }

        if (
            market_alignment == "SUPPORTIVE"
            and sector_alignment == "SUPPORTIVE"
        ):
            return "STRONGLY_SUPPORTIVE"

        if (
            market_alignment == "CONFLICTING"
            and sector_alignment == "CONFLICTING"
        ):
            return "STRONGLY_CONFLICTING"

        if (
            "SUPPORTIVE" in alignments
            and "CONFLICTING" in alignments
        ):
            return "MIXED"

        if "SUPPORTIVE" in alignments:
            return "SUPPORTIVE"

        if "CONFLICTING" in alignments:
            return "CONFLICTING"

        return "NEUTRAL"

    # ========================================================
    # OVERALL SCORE
    # ========================================================

    @staticmethod
    def _calculate_overall_score(
        market_score: float,
        sector_score: float
    ) -> float:
        """
        Market receives slightly higher weight than sector.

        Market: 60%
        Sector: 40%
        """

        score = (
            (market_score * 0.60)
            + (sector_score * 0.40)
        )

        return round(score, 4)

    # ========================================================
    # CONTEXT STRENGTH
    # ========================================================

    @staticmethod
    def _context_strength(score: float) -> str:

        if score >= 0.75:
            return "STRONG"

        if score >= 0.60:
            return "MODERATE"

        if score >= 0.40:
            return "WEAK"

        return "VERY_WEAK"

    # ========================================================
    # CONTEXT RISK
    # ========================================================

    @staticmethod
    def _context_risk(
        score: float,
        reliability_score: float,
        risk_level: str
    ) -> str:
        """
        Determine contextual risk.

        Strong conflict + low reliability = HIGH risk.
        """

        risk_level = str(risk_level).upper()

        if score < 0.30:
            return "HIGH"

        if (
            score < 0.45
            or reliability_score < 0.35
            or risk_level == "HIGH"
        ):
            return "MEDIUM"

        return "LOW"

    # ========================================================
    # OVERALL CONTEXT
    # ========================================================

    @staticmethod
    def _overall_context(
        overall_score: float,
        overall_alignment: str
    ) -> str:

        if overall_alignment == "STRONGLY_SUPPORTIVE":
            return "SUPPORTIVE"

        if overall_alignment == "STRONGLY_CONFLICTING":
            return "CONFLICTING"

        if overall_alignment == "MIXED":
            return "MIXED"

        if overall_score >= 0.60:
            return "SUPPORTIVE"

        if overall_score <= 0.40:
            return "CONFLICTING"

        return "NEUTRAL"

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze(
        self,
        data: AIMarketContextInput
    ) -> AIMarketContextResult:
        """
        Execute Layer 4 Market Context Intelligence.
        """

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        decision_direction = self._decision_direction(
            data.decision
        )

        market_direction = self._normalize_signal(
            data.market_direction
        )

        sector_direction = self._normalize_signal(
            data.sector_direction
        )

        # ----------------------------------------------------
        # Market Alignment
        # ----------------------------------------------------

        market_alignment = self._calculate_alignment(
            decision_direction,
            market_direction
        )

        # ----------------------------------------------------
        # Sector Alignment
        # ----------------------------------------------------

        sector_alignment = self._calculate_alignment(
            decision_direction,
            sector_direction
        )

        # ----------------------------------------------------
        # Context Scores
        # ----------------------------------------------------

        market_score = self._calculate_context_score(
            decision_direction,
            market_direction,
            data.market_confidence
        )

        sector_score = self._calculate_context_score(
            decision_direction,
            sector_direction,
            data.sector_confidence
        )

        overall_score = self._calculate_overall_score(
            market_score,
            sector_score
        )

        # ----------------------------------------------------
        # Overall Alignment
        # ----------------------------------------------------

        overall_alignment = self._calculate_overall_alignment(
            market_alignment,
            sector_alignment
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        context_strength = self._context_strength(
            overall_score
        )

        context_risk = self._context_risk(
            overall_score,
            data.reliability_score,
            data.risk_level
        )

        overall_context = self._overall_context(
            overall_score,
            overall_alignment
        )

        # ----------------------------------------------------
        # Explanation Lists
        # ----------------------------------------------------

        supporting_factors = []
        conflicting_factors = []
        warnings = []

        # ----------------------------------------------------
        # Market explanation
        # ----------------------------------------------------

        if market_alignment == "SUPPORTIVE":

            supporting_factors.append(
                f"Market regime {data.market_regime} "
                f"supports the {data.decision} decision "
                f"with {data.market_confidence:.2%} confidence."
            )

        elif market_alignment == "CONFLICTING":

            conflicting_factors.append(
                f"Market regime {data.market_regime} "
                f"is {market_direction} while the AI decision "
                f"is {data.decision}."
            )

            warnings.append(
                "Broader market direction conflicts with "
                "the stock-level AI decision."
            )

        else:

            supporting_factors.append(
                "Market context does not provide strong "
                "directional support or opposition."
            )

        # ----------------------------------------------------
        # Sector explanation
        # ----------------------------------------------------

        if sector_alignment == "SUPPORTIVE":

            supporting_factors.append(
                f"The {data.sector} sector is {sector_direction} "
                f"with {data.sector_confidence:.2%} confidence."
            )

        elif sector_alignment == "CONFLICTING":

            conflicting_factors.append(
                f"The {data.sector} sector is {sector_direction} "
                f"while the AI decision is {data.decision}."
            )

            warnings.append(
                "Sector direction conflicts with "
                "the stock-level AI decision."
            )

        else:

            supporting_factors.append(
                f"The {data.sector} sector provides "
                "limited directional context."
            )

        # ----------------------------------------------------
        # Reliability warning
        # ----------------------------------------------------

        if data.reliability_score < 0.35:

            warnings.append(
                "Overall AI reliability is very low; "
                "market context should be interpreted cautiously."
            )

        elif data.reliability_score < 0.50:

            warnings.append(
                "AI reliability is limited; contextual signals "
                "should not be treated as confirmation by themselves."
            )

        # ----------------------------------------------------
        # High-risk warning
        # ----------------------------------------------------

        if context_risk == "HIGH":

            warnings.append(
                "Market and/or sector context presents "
                "high contextual risk."
            )

        # ----------------------------------------------------
        # HOLD-specific interpretation
        # ----------------------------------------------------

        if decision_direction == "neutral":

            if overall_alignment == "MIXED":

                warnings.append(
                    "The HOLD decision is supported by mixed "
                    "market and sector conditions."
                )

            elif overall_alignment == "STRONGLY_CONFLICTING":

                warnings.append(
                    "Market and sector conditions are broadly "
                    "opposed to the current directional evidence."
                )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return AIMarketContextResult(
            instrument_key=data.instrument_key,

            decision=data.decision,
            decision_confidence=data.decision_confidence,

            market_context_score=market_score,
            sector_context_score=sector_score,
            overall_context_score=overall_score,

            market_alignment=market_alignment,
            sector_alignment=sector_alignment,
            overall_alignment=overall_alignment,

            context_strength=context_strength,
            context_risk=context_risk,
            overall_context=overall_context,

            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            warnings=warnings,

            market_regime=data.market_regime,
            market_direction=market_direction,
            market_confidence=data.market_confidence,

            sector=data.sector,
            sector_direction=sector_direction,
            sector_confidence=data.sector_confidence,

            reliability_score=data.reliability_score,
            risk_level=data.risk_level,

            status="success"
        )