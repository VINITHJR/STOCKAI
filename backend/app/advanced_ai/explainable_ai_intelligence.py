from __future__ import annotations

from .layer3_models import (
    AIExplanationInput,
    AIExplanationResult,
)


class ExplainableAIIntelligence:
    """
    Phase 14 - Layer 3
    Explainable AI Intelligence.

    This layer does NOT create a new trading decision.

    It explains:

        - why Layer 1 produced the decision
        - what factors supported it
        - what factors opposed it
        - why confidence is high/low
        - why reliability is high/low
        - what risks are present

    The output is deterministic and human-readable.
    """

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        data: AIExplanationInput,
    ) -> AIExplanationResult:

        bullish_factors: list[str] = []
        bearish_factors: list[str] = []
        neutral_factors: list[str] = []

        supporting_factors: list[str] = []
        conflicting_factors: list[str] = []
        risk_factors: list[str] = []

        # =====================================================
        # 1. AI MODEL
        # =====================================================

        ai_signal = self._normalize_signal(
            data.ai_signal
        )

        if ai_signal == "bullish":

            factor = (
                f"AI model is bullish with "
                f"{data.ai_confidence:.2%} confidence."
            )

            bullish_factors.append(factor)

            if data.decision == "BUY":

                supporting_factors.append(
                    factor
                )

            else:

                conflicting_factors.append(
                    "The AI model is bullish, "
                    f"but the final decision is {data.decision}."
                )

        elif ai_signal == "bearish":

            factor = (
                f"AI model is bearish with "
                f"{data.ai_confidence:.2%} confidence."
            )

            bearish_factors.append(factor)

            if data.decision == "SELL":

                supporting_factors.append(
                    factor
                )

            else:

                conflicting_factors.append(
                    "The AI model is bearish, "
                    f"but the final decision is {data.decision}."
                )

        else:

            neutral_factors.append(
                "AI model does not provide a strong "
                "directional signal."
            )

        # =====================================================
        # 2. ENSEMBLE
        # =====================================================

        ensemble_signal = self._normalize_signal(
            data.ensemble_signal
        )

        if ensemble_signal == "bullish":

            factor = (
                f"Ensemble signal is bullish with "
                f"{data.ensemble_confidence:.2%} confidence."
            )

            bullish_factors.append(factor)

            if data.decision == "BUY":

                supporting_factors.append(
                    factor
                )

            else:

                conflicting_factors.append(
                    "The ensemble is bullish but does not "
                    f"match the final {data.decision} decision."
                )

        elif ensemble_signal == "bearish":

            factor = (
                f"Ensemble signal is bearish with "
                f"{data.ensemble_confidence:.2%} confidence."
            )

            bearish_factors.append(factor)

            if data.decision == "SELL":

                supporting_factors.append(
                    factor
                )

            else:

                conflicting_factors.append(
                    "The ensemble is bearish but does not "
                    f"match the final {data.decision} decision."
                )

        else:

            neutral_factors.append(
                "The ensemble does not provide a strong "
                "directional signal."
            )

        # =====================================================
        # 3. MARKET REGIME
        # =====================================================

        market_direction = self._normalize_signal(
            data.market_direction
        )

        if market_direction == "bullish":

            factor = (
                f"Market regime is bullish with "
                f"{data.market_confidence:.2%} confidence."
            )

            bullish_factors.append(factor)

            if data.market_alignment == "supportive":

                supporting_factors.append(
                    "The bullish market regime supports "
                    "the AI direction."
                )

            else:

                conflicting_factors.append(
                    "The bullish market regime does not "
                    "fully align with the AI decision."
                )

        elif market_direction == "bearish":

            factor = (
                f"Market regime is bearish with "
                f"{data.market_confidence:.2%} confidence."
            )

            bearish_factors.append(factor)

            if data.market_alignment == "supportive":

                supporting_factors.append(
                    "The bearish market regime supports "
                    "the AI direction."
                )

            else:

                conflicting_factors.append(
                    "The bearish market regime conflicts "
                    "with the AI direction."
                )

        else:

            neutral_factors.append(
                "The current market regime does not "
                "provide a strong directional bias."
            )

        # =====================================================
        # 4. TECHNICAL ANALYSIS
        # =====================================================

        technical_signal = self._normalize_signal(
            data.technical_signal
        )

        self._add_supporting_signal(
            signal=technical_signal,
            name="Technical analysis",
            decision=data.decision,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
        )

        # =====================================================
        # 5. FUNDAMENTAL ANALYSIS
        # =====================================================

        fundamental_signal = self._normalize_signal(
            data.fundamental_signal
        )

        self._add_supporting_signal(
            signal=fundamental_signal,
            name="Fundamental analysis",
            decision=data.decision,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
        )

        # =====================================================
        # 6. NEWS ANALYSIS
        # =====================================================

        news_signal = self._normalize_signal(
            data.news_signal
        )

        self._add_supporting_signal(
            signal=news_signal,
            name="News and event analysis",
            decision=data.decision,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
        )

        # =====================================================
        # 7. MODEL CONFLICT
        # =====================================================

        if data.directional_conflict:
            conflicting_factors.append(
                "Models issued opposing bullish and bearish directional signals "
                f"(legacy conflict level: {data.conflict_level})."
            )
            risk_factors.append(
                "Opposing directional model evidence reduces confidence in the decision."
            )
        elif data.disagreement_type == "directional_neutral_disagreement":
            conflicting_factors.append(
                "At least one model is directional while another is neutral; "
                "this is partial directional agreement, not opposing-direction conflict."
            )
            risk_factors.append(
                "Directional confirmation is partial because one or more models are neutral."
            )
        elif data.conflict:
            conflicting_factors.append(
                f"Model outputs disagree (legacy conflict level: {data.conflict_level})."
            )
            risk_factors.append(
                "Model disagreement warrants caution when interpreting the decision."
            )

        # =====================================================
        # 8. MODEL AGREEMENT
        # =====================================================

        if data.model_count > 0:
            ensemble_signal = str(data.ensemble_signal or "neutral").strip().lower()
            if ensemble_signal not in {"bullish", "bearish"}:
                risk_factors.append(
                    "The ensemble signal is neutral; no bullish or bearish "
                    "directional confirmation is available."
                )
            elif data.agreement_count == data.model_count:
                supporting_factors.append(
                    f"All {data.model_count} available models confirm the "
                    f"{ensemble_signal} direction."
                )
            elif data.agreement_count == 0:
                risk_factors.append(
                    f"No model confirms the {ensemble_signal} direction."
                )
            else:
                risk_factors.append(
                    f"Only {data.agreement_count} of {data.model_count} "
                    f"models confirm the {ensemble_signal} direction."
                )

        # =====================================================
        # 9. LAYER 2 RELIABILITY
        # =====================================================

        if data.reliability_level in {
            "VERY_LOW",
            "LOW",
        }:

            risk_factors.append(
                f"Overall AI reliability is "
                f"{data.reliability_level.lower().replace('_', ' ')}."
            )

        elif data.reliability_level == "MODERATE":

            supporting_factors.append(
                "AI reliability is at a moderate level."
            )

        else:

            supporting_factors.append(
                "AI reliability is sufficiently strong."
            )

        # =====================================================
        # 10. RISK LEVEL
        # =====================================================

        if data.risk_level in {
            "HIGH",
            "MEDIUM",
        }:

            risk_factors.append(
                f"Current AI risk level is "
                f"{data.risk_level.lower()}."
            )

        # =====================================================
        # 11. DECISION SUMMARY
        # =====================================================

        decision_summary = (
            self._build_decision_summary(
                data
            )
        )

        # =====================================================
        # 12. CONFIDENCE EXPLANATION
        # =====================================================

        confidence_explanation = (
            self._build_confidence_explanation(
                data
            )
        )

        # =====================================================
        # 13. RELIABILITY EXPLANATION
        # =====================================================

        reliability_explanation = (
            self._build_reliability_explanation(
                data
            )
        )

        # =====================================================
        # 14. STRONGEST FACTOR
        # =====================================================

        strongest_factor = (
            self._find_strongest_factor(
                data
            )
        )

        strongest_risk = (
            self._find_strongest_risk(
                data
            )
        )

        # =====================================================
        # 15. FINAL EXPLANATION
        # =====================================================

        final_explanation = (
            self._build_final_explanation(
                data,
                supporting_factors,
                conflicting_factors,
                risk_factors,
            )
        )

        # =====================================================
        # 16. EXPLANATION QUALITY
        # =====================================================

        explanation_quality = (
            self._explanation_quality(
                data,
                bullish_factors,
                bearish_factors,
                supporting_factors,
                conflicting_factors,
            )
        )

        # =====================================================
        # 17. RETURN
        # =====================================================

        return AIExplanationResult(
            instrument_key=data.instrument_key,

            decision=data.decision,
            decision_confidence=data.decision_confidence,

            decision_summary=decision_summary,

            confidence_explanation=(
                confidence_explanation
            ),

            reliability_explanation=(
                reliability_explanation
            ),

            final_explanation=final_explanation,

            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,

            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            risk_factors=risk_factors,

            strongest_factor=strongest_factor,
            strongest_risk=strongest_risk,

            explanation_quality=explanation_quality,
            directional_conflict=data.directional_conflict,
            disagreement_type=data.disagreement_type,
            agreement_type=data.agreement_type,
            neutral_agreement_count=data.neutral_agreement_count,

            status="success",
        )

    # =========================================================
    # SUPPORTING SIGNAL HANDLER
    # =========================================================

    def _add_supporting_signal(
        self,
        signal: str,
        name: str,
        decision: str,
        bullish_factors: list[str],
        bearish_factors: list[str],
        neutral_factors: list[str],
        supporting_factors: list[str],
        conflicting_factors: list[str],
    ) -> None:

        if signal == "bullish":

            message = (
                f"{name} is bullish."
            )

            bullish_factors.append(
                message
            )

            if decision == "BUY":

                supporting_factors.append(
                    message
                )

            elif decision == "SELL":

                conflicting_factors.append(
                    f"{name} is bullish but the "
                    "AI decision is bearish."
                )

        elif signal == "bearish":

            message = (
                f"{name} is bearish."
            )

            bearish_factors.append(
                message
            )

            if decision == "SELL":

                supporting_factors.append(
                    message
                )

            elif decision == "BUY":

                conflicting_factors.append(
                    f"{name} is bearish but the "
                    "AI decision is bullish."
                )

        else:

            neutral_factors.append(
                f"{name} is neutral."
            )

    # =========================================================
    # DECISION SUMMARY
    # =========================================================

    @staticmethod
    def _build_decision_summary(
        data: AIExplanationInput,
    ) -> str:

        decision = data.decision.upper()

        if decision == "BUY":

            return (
                "The AI system currently favors a BUY decision "
                "based on the strongest available bullish evidence."
            )

        if decision == "SELL":

            return (
                "The AI system currently favors a SELL decision "
                "based on the strongest available bearish evidence."
            )

        return (
            "The AI system currently favors HOLD because the "
            "available evidence is not strong or consistent "
            "enough for a high-confidence directional decision."
        )

    # =========================================================
    # CONFIDENCE EXPLANATION
    # =========================================================

    @staticmethod
    def _build_confidence_explanation(
        data: AIExplanationInput,
    ) -> str:
        confidence = max(
            0.0,
            min(1.0, float(data.decision_confidence or 0.0)),
        )

        if confidence >= 0.80:
            level = "very high"
        elif confidence >= 0.65:
            level = "high"
        elif confidence >= 0.55:
            level = "moderate"
        elif confidence >= 0.40:
            level = "low"
        else:
            level = "very low"

        reliability = max(
            0.0,
            min(1.0, float(data.reliability_score or 0.0)),
        )

        return (
            f"Decision confidence is {confidence:.2%} "
            f"({level}), representing the relative strength "
            f"of the evidence for the selected decision. "
            f"Overall reliability is {reliability:.2%} "
            f"({data.reliability_level}) and is assessed "
            f"separately using model agreement, market context, "
            f"and supporting-analysis quality. Decision "
            f"confidence is not a calibrated probability of "
            f"profit; neither metric guarantees future performance."
        )

    # =========================================================
    # RELIABILITY EXPLANATION
    # =========================================================

    @staticmethod
    def _build_reliability_explanation(
        data: AIExplanationInput,
    ) -> str:

        return (
            f"Overall reliability is "
            f"{data.reliability_score:.2%} "
            f"({data.reliability_level}). "
            f"Model reliability is "
            f"{data.model_reliability:.2%}, "
            f"market reliability is "
            f"{data.market_reliability:.2%}, "
            f"signal reliability is "
            f"{data.signal_reliability:.2%}, "
            f"and supporting-analysis reliability is "
            f"{data.supporting_reliability:.2%}."
        )

    # =========================================================
    # FINAL EXPLANATION
    # =========================================================

    @staticmethod
    def _build_final_explanation(
        data: AIExplanationInput,
        supporting_factors: list[str],
        conflicting_factors: list[str],
        risk_factors: list[str],
    ) -> str:

        decision = data.decision.upper()

        explanation = (
            f"The final AI decision is {decision}. "
            f"Decision confidence is "
            f"{data.decision_confidence:.2%}, "
            f"while overall reliability is "
            f"{data.reliability_score:.2%}."
        )

        if supporting_factors:

            explanation += (
                " Supporting evidence includes "
                + "; ".join(
                    supporting_factors[:3]
                )
                + "."
            )

        if conflicting_factors:

            explanation += (
                " However, conflicting evidence includes "
                + "; ".join(
                    conflicting_factors[:3]
                )
                + "."
            )

        if risk_factors:

            explanation += (
                " Main risks are "
                + "; ".join(
                    risk_factors[:3]
                )
                + "."
            )

        return explanation

    # =========================================================
    # STRONGEST FACTOR
    # =========================================================

    @staticmethod
    def _find_strongest_factor(
        data: AIExplanationInput,
    ) -> str:

        candidates = [
            (
                data.ai_confidence,
                f"AI model confidence at {data.ai_confidence:.2%}.",
            ),
            (
                data.ensemble_confidence,
                f"Ensemble confidence at {data.ensemble_confidence:.2%}.",
            ),
            (
                data.market_confidence,
                f"Market regime confidence at {data.market_confidence:.2%}.",
            ),
            (
                data.technical_confidence,
                f"Technical analysis confidence at {data.technical_confidence:.2%}.",
            ),
            (
                data.fundamental_confidence,
                f"Fundamental analysis confidence at {data.fundamental_confidence:.2%}.",
            ),
            (
                data.news_confidence,
                f"News and event analysis confidence at {data.news_confidence:.2%}.",
            ),
        ]

        return max(
            candidates,
            key=lambda item: item[0],
        )[1]

    # =========================================================
    # STRONGEST RISK
    # =========================================================

    @staticmethod
    def _find_strongest_risk(
        data: AIExplanationInput,
    ) -> str:

        if data.directional_conflict and data.conflict_level.lower() == "high":
            return "High opposing-direction model conflict."

        if data.directional_conflict and data.conflict_level.lower() == "medium":
            return "Medium opposing-direction conflict reduces directional confidence."

        if data.disagreement_type == "directional_neutral_disagreement":
            return "Partial directional agreement because at least one model is neutral."

        if data.conflict:
            return "Model outputs disagree, but opposing directions are not confirmed."

        if data.market_alignment == "conflicting":

            return (
                "Market regime conflicts with the AI direction."
            )

        if data.signal_alignment == "conflicting":

            return (
                "AI and ensemble signals are conflicting."
            )

        if data.reliability_score < 0.35:

            return (
                "Very low overall AI reliability."
            )

        return (
            "No single dominant risk identified."
        )

    # =========================================================
    # EXPLANATION QUALITY
    # =========================================================

    @staticmethod
    def _explanation_quality(
        data: AIExplanationInput,
        bullish_factors: list[str],
        bearish_factors: list[str],
        supporting_factors: list[str],
        conflicting_factors: list[str],
    ) -> str:

        factor_count = (
            len(bullish_factors)
            + len(bearish_factors)
            + len(supporting_factors)
            + len(conflicting_factors)
        )

        if factor_count >= 8:

            return "DETAILED"

        if factor_count >= 5:

            return "GOOD"

        return "STANDARD"

    # =========================================================
    # SIGNAL NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_signal(
        signal: str,
    ) -> str:

        if not signal:

            return "neutral"

        value = (
            str(signal)
            .strip()
            .lower()
        )

        mapping = {
            "bullish": "bullish",
            "buy": "bullish",
            "long": "bullish",

            "bearish": "bearish",
            "sell": "bearish",
            "short": "bearish",

            "neutral": "neutral",
            "hold": "neutral",
            "sideways": "neutral",
            "unknown": "neutral",
        }

        return mapping.get(
            value,
            "neutral",
        )