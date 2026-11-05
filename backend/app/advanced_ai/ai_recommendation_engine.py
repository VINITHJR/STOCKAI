from __future__ import annotations

from .layer5_models import AIRecommendationInput, AIRecommendationResult


class AIRecommendationEngine:
    """
    Phase 14 — Layer 5.

    Converts the validated intelligence from Layers 1–4 into a
    presentation-level recommendation.

    Layer 5 does not invent a new BUY/SELL/HOLD direction.
    It preserves Layer 1's decision and adjusts confidence,
    strength and risk according to reliability, context and model conflict.
    """

    HIGH_CONFLICT_FACTOR = 0.75
    MEDIUM_CONFLICT_FACTOR = 0.88
    LOW_CONFLICT_FACTOR = 0.95
    NON_DIRECTIONAL_DISAGREEMENT_FACTOR = 0.97

    @staticmethod
    def _normalize_decision(value: str) -> str:
        value = str(value or "").strip().upper()
        if value in {"BUY", "BULLISH"}:
            return "BUY"
        if value in {"SELL", "BEARISH"}:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _directional_agreement_count(
        data: AIRecommendationInput,
    ) -> int:
        """Return a bounded directional agreement count.

        New callers should provide ``directional_agreement_count`` explicitly.
        For backward compatibility, older callers fall back to ``agreement_count``.
        A neutral ensemble never receives directional agreement credit.
        """
        ensemble_signal = str(
            getattr(data, "ensemble_signal", "neutral") or "neutral"
        ).strip().lower()
        if ensemble_signal not in {"bullish", "bearish"}:
            return 0

        raw_count = getattr(data, "directional_agreement_count", None)
        if raw_count is None:
            raw_count = getattr(data, "agreement_count", 0)

        try:
            count = int(raw_count or 0)
            model_count = max(0, int(getattr(data, "model_count", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

        return max(0, min(count, model_count))

    @staticmethod
    def _conflict_factor(
        conflict: bool,
        level: str,
        directional_conflict: bool = False,
    ) -> float:
        """Apply stronger penalties to truly opposing bullish/bearish signals."""
        if directional_conflict:
            level = str(level or "").strip().lower()
            if level == "high":
                return 0.75
            if level == "medium":
                return 0.88
            return 0.95
        if conflict:
            return AIRecommendationEngine.NON_DIRECTIONAL_DISAGREEMENT_FACTOR
        return 1.0

    @staticmethod
    def _recommendation_quality(
        decision: str,
        reliability: float,
        context_score: float,
        risk_level: str,
        conflict: bool,
        conflict_level: str,
        directional_conflict: bool = False,
    ) -> str:
        reliability = max(0.0, min(1.0, reliability))
        context_score = max(0.0, min(1.0, context_score))
        risk_level = str(risk_level or "").upper()
        conflict_level = str(conflict_level or "").lower()

        if reliability < 0.35:
            return "UNCERTAIN"

        if directional_conflict and conflict_level == "high":
            return "CAUTIOUS"

        if decision == "HOLD":
            if reliability >= 0.70 and context_score >= 0.60 and not conflict:
                return "HIGH_QUALITY"
            if reliability >= 0.50:
                return "REASONABLE"
            return "CAUTIOUS"

        if risk_level == "HIGH":
            return "CAUTIOUS"

        if reliability >= 0.75 and context_score >= 0.65 and not conflict:
            return "HIGH_QUALITY"

        if reliability >= 0.60 and context_score >= 0.55:
            return "GOOD"

        if reliability >= 0.45:
            return "CAUTIOUS"

        return "UNCERTAIN"

    @staticmethod
    def _recommendation_strength(
        confidence: float,
        reliability: float,
        conflict: bool = False,
        conflict_level: str = "none",
        directional_conflict: bool = False,
    ) -> str:
        combined = 0.50 * confidence + 0.50 * reliability

        if conflict or directional_conflict:
            combined *= AIRecommendationEngine._conflict_factor(
                conflict, conflict_level, directional_conflict
            )

        if combined >= 0.80:
            return "VERY_STRONG"
        if combined >= 0.65:
            return "STRONG"
        if combined >= 0.50:
            return "MODERATE"
        if combined >= 0.35:
            return "WEAK"
        return "VERY_WEAK"

    @staticmethod
    def _recommendation_confidence(
        decision_confidence: float,
        reliability_score: float,
        context_score: float,
        conflict: bool = False,
        conflict_level: str = "none",
        directional_conflict: bool = False,
    ) -> float:
        score = (
            0.40 * decision_confidence
            + 0.35 * reliability_score
            + 0.25 * context_score
        )

        if conflict or directional_conflict:
            score *= AIRecommendationEngine._conflict_factor(
                conflict, conflict_level, directional_conflict
            )

        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _recommendation_risk(
        reliability_score: float,
        context_score: float,
        risk_level: str,
        context_risk: str,
        conflict: bool = False,
        conflict_level: str = "none",
        directional_conflict: bool = False,
        disagreement_type: str = "none",
    ) -> str:
        """Classify risk from evidence quality and conflict semantics.

        A bullish/neutral disagreement is reported as partial confirmation and
        receives a mild confidence/strength penalty elsewhere. It does not, by
        itself, force MEDIUM risk. Opposing directions remain severity-weighted.
        Legacy callers with an unknown generic conflict keep the conservative
        MEDIUM-risk behavior.
        """
        reliability_score = max(0.0, min(1.0, float(reliability_score)))
        context_score = max(0.0, min(1.0, float(context_score)))
        risk_level = str(risk_level or "").upper()
        context_risk = str(context_risk or "").upper()
        conflict_level = str(conflict_level or "").upper()
        disagreement_type = str(disagreement_type or "none").strip().lower()
        partial_non_opposing = (
            not directional_conflict
            and disagreement_type == "directional_neutral_disagreement"
        )

        if (
            reliability_score < 0.35
            or risk_level == "HIGH"
            or context_risk == "HIGH"
            or context_score < 0.30
            or (directional_conflict and conflict_level == "HIGH")
        ):
            return "HIGH"

        conflict_requires_medium = (
            (directional_conflict and conflict_level == "MEDIUM")
            or (
                conflict
                and not partial_non_opposing
                and (conflict_level == "MEDIUM" or not directional_conflict)
            )
        )
        if (
            reliability_score < 0.50
            or risk_level == "MEDIUM"
            or context_risk == "MEDIUM"
            or context_score < 0.45
            or conflict_requires_medium
        ):
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _calculate_directional_scores(
        decision: str,
        decision_confidence: float,
        reliability_score: float,
        context_score: float,
    ):
        directional_strength = max(
            0.0,
            min(
                1.0,
                0.45 * decision_confidence
                + 0.30 * reliability_score
                + 0.25 * context_score,
            ),
        )

        decision = str(decision or "").upper()

        if decision == "BUY":
            bullish_score = directional_strength
            bearish_score = 0.25 * (1.0 - context_score)
            neutral_score = max(0.0, 1.0 - bullish_score - bearish_score)
        elif decision == "SELL":
            bearish_score = directional_strength
            bullish_score = 0.25 * (1.0 - context_score)
            neutral_score = max(0.0, 1.0 - bearish_score - bullish_score)
        else:
            bullish_score = 0.0
            bearish_score = 0.0
            neutral_score = 1.0

        return (
            round(bullish_score, 4),
            round(bearish_score, 4),
            round(neutral_score, 4),
        )

    @staticmethod
    def _build_supporting_factors(data: AIRecommendationInput, decision: str):
        factors = []

        if data.reliability_score >= 0.65:
            factors.append(
                f"AI reliability is {data.reliability_score:.2%}."
            )

        if data.overall_context_score >= 0.65:
            factors.append("Market and sector context provide supportive evidence.")

        if data.overall_alignment in {"SUPPORTIVE", "STRONGLY_SUPPORTIVE"}:
            factors.append(
                f"Broader market context supports the {decision} direction."
            )

        ensemble_signal = str(data.ensemble_signal or "neutral").strip().lower()
        directional_count = AIRecommendationEngine._directional_agreement_count(data)
        if (
            ensemble_signal in {"bullish", "bearish"}
            and not data.conflict
            and data.model_count >= 2
            and directional_count == data.model_count
        ):
            factors.append(
                f"All {directional_count} available models confirm the "
                f"{ensemble_signal} direction."
            )

        if data.strongest_factor:
            factors.append(f"Strongest evidence: {data.strongest_factor}")

        return list(dict.fromkeys(factors))

    @staticmethod
    def _build_conflicting_factors(data: AIRecommendationInput, decision: str):
        factors = []

        if data.reliability_score < 0.50:
            factors.append(
                f"AI reliability is limited at {data.reliability_score:.2%}."
            )

        if data.overall_alignment in {"CONFLICTING", "STRONGLY_CONFLICTING"}:
            factors.append(
                f"Market and/or sector context conflicts with the {decision} direction."
            )

        if data.overall_alignment == "MIXED":
            factors.append("Market and sector conditions provide mixed contextual evidence.")

        if data.directional_conflict:
            factors.append(
                "Opposing bullish and bearish model directions are present "
                f"(conflict level: {data.conflict_level})."
            )
        elif data.disagreement_type == "directional_neutral_disagreement":
            existing_partial = any(
                "directional" in str(item).lower()
                and "neutral" in str(item).lower()
                for item in data.conflicting_factors
            )
            if not existing_partial:
                factors.append(
                    "Model confirmation is partial: a directional model and a neutral model disagree; "
                    "they do not directly oppose each other."
                )
        elif data.conflict:
            factors.append("Model outputs disagree; inspect individual model signals.")

        if data.strongest_risk:
            factors.append(f"Strongest risk: {data.strongest_risk}")

        factors.extend(data.conflicting_factors)
        return list(dict.fromkeys(factors))

    @staticmethod
    def _build_warnings(
        data: AIRecommendationInput,
        recommendation_risk: str,
        recommendation_quality: str,
    ):
        warnings = []

        if recommendation_risk == "HIGH":
            warnings.append(
                "Recommendation carries high risk and should not be treated as a strong trading signal."
            )

        if recommendation_quality == "UNCERTAIN":
            warnings.append(
                "Recommendation reliability is insufficient for a high-confidence action."
            )

        if data.directional_conflict:
            warnings.append(
                "Opposing bullish and bearish directional model signals are present; "
                "recommendation confidence and reliability are reduced."
            )
        elif data.disagreement_type == "directional_neutral_disagreement":
            warnings.append(
                "Models disagree between a directional and neutral signal. "
                "This is partial confirmation, not an opposing-direction conflict."
            )
        elif data.conflict:
            warnings.append("Model outputs disagree; inspect individual signals before relying on direction.")

        if str(data.ensemble_signal or "neutral").strip().lower() not in {"bullish", "bearish"}:
            warnings.append(
                "The ensemble has no bullish or bearish directional signal; "
                "model consensus must not be interpreted as trade confirmation."
            )

        if data.context_risk in {"HIGH", "MEDIUM"}:
            warnings.append(
                f"Market context risk is {data.context_risk.lower()}."
            )

        if data.reliability_score < 0.35:
            warnings.append("Overall AI reliability is very low.")

        return list(dict.fromkeys(warnings))

    @staticmethod
    def _action_guidance(
        decision: str,
        quality: str,
        risk: str,
        ensemble_signal: str = "neutral",
        directional_conflict: bool = False,
        disagreement_type: str = "none",
    ) -> str:
        ensemble_signal = str(ensemble_signal or "neutral").strip().lower()
        disagreement_type = str(disagreement_type or "none").strip().lower()

        if decision == "BUY":
            if quality in {"HIGH_QUALITY", "GOOD"} and risk == "LOW":
                return "BUY conditions are supported by the available AI evidence and contextual signals."
            return (
                "BUY evidence exists, but confirmation is limited. "
                "Use caution and avoid treating this as a high-confidence entry signal."
            )

        if decision == "SELL":
            if quality in {"HIGH_QUALITY", "GOOD"} and risk == "LOW":
                return "SELL conditions are supported by the available AI evidence and contextual signals."
            return (
                "SELL evidence exists, but confirmation is limited. "
                "Use caution and avoid treating this as a high-confidence exit signal."
            )

        if directional_conflict:
            return (
                "HOLD: bullish and bearish model directions oppose each other. "
                "Wait for the disagreement and risk evidence to improve before treating the output as directional confirmation."
            )

        if disagreement_type == "directional_neutral_disagreement":
            return (
                "HOLD: model confirmation is partial because one model is directional and another is neutral. "
                "This is not direct bullish-versus-bearish opposition; wait for stronger confirmation."
            )

        if ensemble_signal not in {"bullish", "bearish"}:
            return (
                "HOLD: the ensemble has no bullish or bearish direction. "
                "Neutral consensus describes the model output and must not be treated as trade confirmation."
            )

        return (
            "HOLD: the weighted evidence or directional threshold does not justify a strong directional action. "
            "Review the decision reason and evidence scores before acting."
        )

    @staticmethod
    def _position_guidance(decision: str, quality: str, risk: str, reliability: float) -> str:
        if decision == "HOLD":
            return "Avoid increasing directional exposure until stronger confirmation becomes available."

        if risk == "HIGH" or reliability < 0.35:
            return "If action is considered, use reduced exposure and apply strict risk controls."

        if quality in {"HIGH_QUALITY", "GOOD"} and risk == "LOW":
            return (
                "Normal position sizing may be considered subject to the portfolio "
                "risk-management rules."
            )

        return "Use conservative position sizing because recommendation confirmation is limited."

    def analyze(self, data: AIRecommendationInput) -> AIRecommendationResult:
        decision = self._normalize_decision(data.decision)

        recommendation_confidence = self._recommendation_confidence(
            data.decision_confidence,
            data.reliability_score,
            data.overall_context_score,
            data.conflict,
            data.conflict_level,
            data.directional_conflict,
        )

        recommendation_quality = self._recommendation_quality(
            decision,
            data.reliability_score,
            data.overall_context_score,
            data.risk_level,
            data.conflict,
            data.conflict_level,
            data.directional_conflict,
        )

        recommendation_strength = self._recommendation_strength(
            data.decision_confidence,
            data.reliability_score,
            data.conflict,
            data.conflict_level,
            data.directional_conflict,
        )

        recommendation_risk = self._recommendation_risk(
            data.reliability_score,
            data.overall_context_score,
            data.risk_level,
            data.context_risk,
            data.conflict,
            data.conflict_level,
            data.directional_conflict,
            data.disagreement_type,
        )

        bullish_score, bearish_score, neutral_score = self._calculate_directional_scores(
            decision,
            data.decision_confidence,
            data.reliability_score,
            data.overall_context_score,
        )

        supporting_factors = self._build_supporting_factors(data, decision)
        conflicting_factors = self._build_conflicting_factors(data, decision)
        warnings = self._build_warnings(
            data,
            recommendation_risk,
            recommendation_quality,
        )

        action_guidance = self._action_guidance(
            decision,
            recommendation_quality,
            recommendation_risk,
            ensemble_signal=data.ensemble_signal,
            directional_conflict=data.directional_conflict,
            disagreement_type=data.disagreement_type,
        )
        position_guidance = self._position_guidance(
            decision,
            recommendation_quality,
            recommendation_risk,
            data.reliability_score,
        )

        recommendation_score = round(
            recommendation_confidence,
            4,
        )

        return AIRecommendationResult(
            instrument_key=data.instrument_key,
            ai_decision=decision,
            ai_confidence=data.decision_confidence,
            recommendation=decision,
            recommendation_confidence=recommendation_confidence,
            recommendation_quality=recommendation_quality,
            recommendation_strength=recommendation_strength,
            risk_level=data.risk_level,
            recommendation_risk=recommendation_risk,
            market_context=data.overall_context,
            market_alignment=data.market_alignment,
            sector_alignment=data.sector_alignment,
            overall_alignment=data.overall_alignment,
            recommendation_score=recommendation_score,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            warnings=warnings,
            action_guidance=action_guidance,
            position_guidance=position_guidance,
            strongest_factor=data.strongest_factor,
            strongest_risk=data.strongest_risk,
            reliability_score=data.reliability_score,
            overall_context_score=data.overall_context_score,
            model_count=data.model_count,
            agreement_count=data.agreement_count,
            directional_agreement_count=(
                AIRecommendationEngine._directional_agreement_count(data)
            ),
            neutral_agreement_count=data.neutral_agreement_count,
            agreement_type=data.agreement_type,
            conflict=data.conflict,
            conflict_level=data.conflict_level,
            directional_conflict=data.directional_conflict,
            disagreement_type=data.disagreement_type,
            status="success",
        )
