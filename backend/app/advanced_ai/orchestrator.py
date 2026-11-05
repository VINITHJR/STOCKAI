# ============================================================
# PHASE 14 — ADVANCED AI
# ORCHESTRATOR
# ============================================================
#
# Connects Phase 14 Layers 1 → 7.
#
# IMPORTANT:
# - Does not modify Layer 1–7 logic.
# - Does not create new AI logic.
# - Does not execute trades.
# - Does not use a database.
#
# ============================================================

from __future__ import annotations

from typing import Any, Dict

from .layer1_models import AIDecisionInput
from .ai_decision_intelligence import AIDecisionIntelligence

from .layer2_models import AIConfidenceInput
from .ai_confidence_reliability import AIConfidenceReliability

from .layer3_models import AIExplanationInput
from .explainable_ai_intelligence import ExplainableAIIntelligence

from .layer4_models import AIMarketContextInput
from .market_context_intelligence import MarketContextIntelligence

from .layer5_models import AIRecommendationInput
from .ai_recommendation_engine import AIRecommendationEngine

from .layer6_models import AIValidationInput
from .advanced_ai_validation import AdvancedAIValidation

from .layer7_models import FinalAIIntelligenceInput
from .final_ai_intelligence_aggregator import (
    FinalAIIntelligenceAggregator,
)


class AdvancedAIOrchestrator:

    def __init__(self) -> None:

        self.layer1 = AIDecisionIntelligence()
        self.layer2 = AIConfidenceReliability()
        self.layer3 = ExplainableAIIntelligence()
        self.layer4 = MarketContextIntelligence()
        self.layer5 = AIRecommendationEngine()
        self.layer6 = AdvancedAIValidation()
        self.layer7 = FinalAIIntelligenceAggregator()

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _signal(
        payload: Dict[str, Any],
        key: str,
    ) -> str:

        value = payload.get(key)

        if value is None:
            return "neutral"

        return str(value)

    @staticmethod
    def _float(
        payload: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:

        value = payload.get(key)

        if value is None:
            return default

        return float(value)

    @staticmethod
    def _int(
        payload: Dict[str, Any],
        key: str,
        default: int = 0,
    ) -> int:

        value = payload.get(key)

        if value is None:
            return default

        return int(value)

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:

        instrument_key = str(
            payload["instrument_key"]
        )

        # ====================================================
        # COMMON INPUTS
        # ====================================================

        ai_signal = self._signal(
            payload,
            "ai_signal",
        )

        ai_confidence = self._float(
            payload,
            "ai_confidence",
        )

        ensemble_signal = self._signal(
            payload,
            "ensemble_signal",
        )

        ensemble_confidence = self._float(
            payload,
            "ensemble_confidence",
        )

        ensemble_score = self._float(
            payload,
            "ensemble_score",
        )

        model_count = self._int(
            payload,
            "model_count",
        )

        agreement_count = self._int(payload, "agreement_count")
        directional_agreement_count = self._int(
            payload,
            "directional_agreement_count",
            agreement_count,
        )
        neutral_agreement_count = self._int(payload, "neutral_agreement_count")
        agreement_type = str(
            payload.get("agreement_type", "no_directional_consensus")
            or "no_directional_consensus"
        )
        directional_conflict = bool(payload.get("directional_conflict", False))
        disagreement_type = str(payload.get("disagreement_type", "none") or "none")

        # Backward compatibility for callers that only supply legacy fields.
        directional_agreement_count = max(0, directional_agreement_count)
        neutral_agreement_count = max(0, neutral_agreement_count)
        if not directional_conflict and disagreement_type == "opposing_directions":
            directional_conflict = True

        conflict = bool(
            payload.get(
                "conflict",
                False,
            )
        )

        conflict_level = str(
            payload.get(
                "conflict_level",
                "none",
            )
        )

        market_regime = str(
            payload.get(
                "market_regime",
                "unknown",
            )
        )

        market_direction = self._signal(
            payload,
            "market_direction",
        )

        market_confidence = self._float(
            payload,
            "market_confidence",
        )

        technical_signal = self._signal(
            payload,
            "technical_signal",
        )

        technical_confidence = self._float(
            payload,
            "technical_confidence",
        )

        fundamental_signal = self._signal(
            payload,
            "fundamental_signal",
        )

        fundamental_confidence = self._float(
            payload,
            "fundamental_confidence",
        )

        news_signal = self._signal(
            payload,
            "news_signal",
        )

        news_confidence = self._float(
            payload,
            "news_confidence",
        )

        sector = str(
            payload.get(
                "sector",
                "Unknown",
            )
            or "Unknown"
        )

        sector_direction = self._signal(
            payload,
            "sector_direction",
        )

        sector_confidence = self._float(
            payload,
            "sector_confidence",
        )

        # ====================================================
        # LAYER 1
        # AI DECISION INTELLIGENCE
        # ====================================================

        layer1_input = AIDecisionInput(

            instrument_key=instrument_key,

            ai_signal=ai_signal,
            ai_confidence=ai_confidence,

            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            ensemble_score=ensemble_score,

            model_count=model_count,
            agreement_count=agreement_count,
            directional_agreement_count=directional_agreement_count,
            neutral_agreement_count=neutral_agreement_count,
            agreement_type=agreement_type,

            conflict=conflict,
            conflict_level=conflict_level,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,

            market_regime=market_regime,
            market_direction=market_direction,
            market_confidence=market_confidence,

            technical_signal=technical_signal,
            technical_confidence=technical_confidence,
            fundamental_signal=fundamental_signal,
            fundamental_confidence=fundamental_confidence,
            news_signal=news_signal,
            news_confidence=news_confidence,
        )

        layer1_result = self.layer1.analyze(
            layer1_input
        )

        # ====================================================
        # LAYER 2
        # AI CONFIDENCE & RELIABILITY
        # ====================================================

        layer2_input = AIConfidenceInput(

            instrument_key=instrument_key,

            decision=layer1_result.decision,
            decision_confidence=layer1_result.confidence,

            signal_alignment=(
                layer1_result.signal_alignment
            ),

            market_alignment=(
                layer1_result.market_alignment
            ),


            model_count=model_count,
            agreement_count=agreement_count,
            directional_agreement_count=directional_agreement_count,

            conflict=conflict,
            conflict_level=conflict_level,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,
            agreement_type=agreement_type,
            neutral_agreement_count=neutral_agreement_count,

            ai_signal=ai_signal,
            ai_confidence=ai_confidence,

            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            ensemble_score=ensemble_score,

            market_regime=market_regime,
            market_direction=market_direction,
            market_confidence=market_confidence,

            technical_signal=technical_signal,
            fundamental_signal=fundamental_signal,
            news_signal=news_signal,
            technical_confidence=technical_confidence,
            fundamental_confidence=fundamental_confidence,
            news_confidence=news_confidence,
        )

        layer2_result = self.layer2.analyze(
            layer2_input
        )

        # ====================================================
        # LAYER 3
        # EXPLAINABLE AI INTELLIGENCE
        # ====================================================

        layer3_input = AIExplanationInput(

            instrument_key=instrument_key,

            # Layer 1
            decision=layer1_result.decision,
            decision_confidence=layer1_result.confidence,

            bullish_score=(
                layer1_result.bullish_score
            ),

            bearish_score=(
                layer1_result.bearish_score
            ),

            neutral_score=(
                layer1_result.neutral_score
            ),

            signal_alignment=(
                layer1_result.signal_alignment
            ),

            market_alignment=(
                layer1_result.market_alignment
            ),

            # Layer 2
            reliability_score=(
                layer2_result.reliability_score
            ),

            confidence_grade=(
                layer2_result.confidence_grade
            ),

            confidence_band=(
                layer2_result.confidence_band
            ),

            reliability_level=(
                layer2_result.reliability_level
            ),

            recommendation_quality=(
                layer2_result.recommendation_quality
            ),

            decision_strength=(
                layer2_result.decision_strength
            ),

            risk_level=(
                layer2_result.risk_level
            ),

            model_reliability=(
                layer2_result.model_reliability
            ),

            market_reliability=(
                layer2_result.market_reliability
            ),

            signal_reliability=(
                layer2_result.signal_reliability
            ),

            supporting_reliability=(
                layer2_result.supporting_reliability
            ),

            # Model information
            model_count=model_count,
            agreement_count=agreement_count,

            conflict=conflict,
            conflict_level=conflict_level,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,
            agreement_type=agreement_type,
            neutral_agreement_count=neutral_agreement_count,

            # AI
            ai_signal=ai_signal,
            ai_confidence=ai_confidence,

            # Ensemble
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            ensemble_score=ensemble_score,

            # Market
            market_regime=market_regime,
            market_direction=market_direction,
            market_confidence=market_confidence,

            # Supporting
            technical_signal=technical_signal,
            technical_confidence=technical_confidence,
            fundamental_signal=fundamental_signal,
            fundamental_confidence=fundamental_confidence,
            news_signal=news_signal,
            news_confidence=news_confidence,
        )

        layer3_result = self.layer3.analyze(
            layer3_input
        )

        # ====================================================
        # LAYER 4
        # MARKET CONTEXT INTELLIGENCE
        # ====================================================

        layer4_input = AIMarketContextInput(

            instrument_key=instrument_key,

            decision=layer1_result.decision,
            decision_confidence=layer1_result.confidence,

            reliability_score=(
                layer2_result.reliability_score
            ),

            risk_level=(
                layer2_result.risk_level
            ),

            market_regime=market_regime,
            market_direction=market_direction,
            market_confidence=market_confidence,

            sector=sector,
            sector_direction=sector_direction,
            sector_confidence=sector_confidence,

            technical_signal=technical_signal,
            fundamental_signal=fundamental_signal,
            news_signal=news_signal,
        )

        layer4_result = self.layer4.analyze(
            layer4_input
        )

        # ====================================================
        # LAYER 5
        # AI RECOMMENDATION ENGINE
        # ====================================================

        layer5_input = AIRecommendationInput(

            instrument_key=instrument_key,

            # Layer 1
            decision=layer1_result.decision,
            decision_confidence=layer1_result.confidence,

            # Layer 2
            reliability_score=(
                layer2_result.reliability_score
            ),

            confidence_grade=(
                layer2_result.confidence_grade
            ),

            confidence_band=(
                layer2_result.confidence_band
            ),

            reliability_level=(
                layer2_result.reliability_level
            ),

            recommendation_quality=(
                layer2_result.recommendation_quality
            ),

            decision_strength=(
                layer2_result.decision_strength
            ),

            risk_level=(
                layer2_result.risk_level
            ),

            model_reliability=(
                layer2_result.model_reliability
            ),

            market_reliability=(
                layer2_result.market_reliability
            ),

            signal_reliability=(
                layer2_result.signal_reliability
            ),

            supporting_reliability=(
                layer2_result.supporting_reliability
            ),

            model_count=model_count,
            agreement_count=agreement_count,
            directional_agreement_count=directional_agreement_count,
            neutral_agreement_count=neutral_agreement_count,
            agreement_type=agreement_type,
            ensemble_signal=ensemble_signal,

            conflict=conflict,
            conflict_level=conflict_level,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,

            # Layer 3
            strongest_factor=(
                layer3_result.strongest_factor
            ),

            strongest_risk=(
                layer3_result.strongest_risk
            ),

            bullish_factors=(
                layer3_result.bullish_factors
            ),

            bearish_factors=(
                layer3_result.bearish_factors
            ),

            neutral_factors=(
                layer3_result.neutral_factors
            ),

            supporting_factors=(
                layer3_result.supporting_factors
            ),

            conflicting_factors=(
                layer3_result.conflicting_factors
            ),

            explanation_risk_factors=(
                layer3_result.risk_factors
            ),

            # Layer 4
            market_context_score=(
                layer4_result.market_context_score
            ),

            sector_context_score=(
                layer4_result.sector_context_score
            ),

            overall_context_score=(
                layer4_result.overall_context_score
            ),

            market_alignment=(
                layer4_result.market_alignment
            ),

            sector_alignment=(
                layer4_result.sector_alignment
            ),

            overall_alignment=(
                layer4_result.overall_alignment
            ),

            context_strength=(
                layer4_result.context_strength
            ),

            context_risk=(
                layer4_result.context_risk
            ),

            overall_context=(
                layer4_result.overall_context
            ),

            market_regime=market_regime,
            market_direction=market_direction,
            market_confidence=market_confidence,

            sector=sector,
            sector_direction=sector_direction,
            sector_confidence=sector_confidence,
        )

        layer5_result = self.layer5.analyze(
            layer5_input
        )

        # ====================================================
        # LAYER 6
        # ADVANCED AI VALIDATION
        # ====================================================

        layer6_input = AIValidationInput(

            instrument_key=instrument_key,

            # Layer 1
            ai_decision=layer1_result.decision,
            ai_confidence=layer1_result.confidence,

            # Layer 2
            reliability_score=(
                layer2_result.reliability_score
            ),

            confidence_grade=(
                layer2_result.confidence_grade
            ),

            confidence_band=(
                layer2_result.confidence_band
            ),

            reliability_level=(
                layer2_result.reliability_level
            ),

            decision_strength=(
                layer2_result.decision_strength
            ),

            risk_level=(
                layer2_result.risk_level
            ),

            model_reliability=(
                layer2_result.model_reliability
            ),

            market_reliability=(
                layer2_result.market_reliability
            ),

            signal_reliability=(
                layer2_result.signal_reliability
            ),

            supporting_reliability=(
                layer2_result.supporting_reliability
            ),

            model_count=model_count,
            agreement_count=agreement_count,

            conflict=conflict,
            conflict_level=conflict_level,

            # Layer 4
            market_context_score=(
                layer4_result.market_context_score
            ),

            sector_context_score=(
                layer4_result.sector_context_score
            ),

            overall_context_score=(
                layer4_result.overall_context_score
            ),

            market_alignment=(
                layer4_result.market_alignment
            ),

            sector_alignment=(
                layer4_result.sector_alignment
            ),

            overall_alignment=(
                layer4_result.overall_alignment
            ),

            context_strength=(
                layer4_result.context_strength
            ),

            context_risk=(
                layer4_result.context_risk
            ),

            overall_context=(
                layer4_result.overall_context
            ),

            # Layer 5
            recommendation=(
                layer5_result.recommendation
            ),

            recommendation_confidence=(
                layer5_result.recommendation_confidence
            ),

            recommendation_quality=(
                layer5_result.recommendation_quality
            ),

            recommendation_strength=(
                layer5_result.recommendation_strength
            ),

            recommendation_risk=(
                layer5_result.recommendation_risk
            ),

            recommendation_score=(
                layer5_result.recommendation_score
            ),

            bullish_score=(
                layer5_result.bullish_score
            ),

            bearish_score=(
                layer5_result.bearish_score
            ),

            neutral_score=(
                layer5_result.neutral_score
            ),
        )

        layer6_result = self.layer6.validate(
            layer6_input
        )

        # ====================================================
        # LAYER 7
        # FINAL AI INTELLIGENCE AGGREGATOR
        # ====================================================

        layer7_input = FinalAIIntelligenceInput(

            instrument_key=instrument_key,

            # ------------------------------------------------
            # Layer 1
            # ------------------------------------------------

            decision=layer1_result.decision,

            decision_confidence=(
                layer1_result.confidence
            ),

            bullish_score=(
                layer1_result.bullish_score
            ),

            bearish_score=(
                layer1_result.bearish_score
            ),

            neutral_score=(
                layer1_result.neutral_score
            ),

            signal_alignment=(
                layer1_result.signal_alignment
            ),

            market_alignment_layer1=(
                layer1_result.market_alignment
            ),

            # ------------------------------------------------
            # Layer 2
            # ------------------------------------------------

            reliability_score=(
                layer2_result.reliability_score
            ),

            confidence_grade=(
                layer2_result.confidence_grade
            ),

            confidence_band=(
                layer2_result.confidence_band
            ),

            reliability_level=(
                layer2_result.reliability_level
            ),

            recommendation_quality_layer2=(
                layer2_result.recommendation_quality
            ),

            decision_strength=(
                layer2_result.decision_strength
            ),

            model_reliability=(
                layer2_result.model_reliability
            ),

            market_reliability=(
                layer2_result.market_reliability
            ),

            signal_reliability=(
                layer2_result.signal_reliability
            ),

            supporting_reliability=(
                layer2_result.supporting_reliability
            ),

            model_count=model_count,
            agreement_count=agreement_count,
            directional_agreement_count=directional_agreement_count,
            neutral_agreement_count=neutral_agreement_count,
            agreement_type=agreement_type,

            conflict=conflict,
            conflict_level=conflict_level,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,

            # ------------------------------------------------
            # Layer 3
            # ------------------------------------------------

            decision_summary=(
                layer3_result.decision_summary
            ),

            confidence_explanation=(
                layer3_result.confidence_explanation
            ),

            reliability_explanation=(
                layer3_result.reliability_explanation
            ),

            final_explanation=(
                layer3_result.final_explanation
            ),

            bullish_factors=(
                layer3_result.bullish_factors
            ),

            bearish_factors=(
                layer3_result.bearish_factors
            ),

            neutral_factors=(
                layer3_result.neutral_factors
            ),

            explanation_supporting_factors=(
                layer3_result.supporting_factors
            ),

            conflicting_factors=(
                layer3_result.conflicting_factors
            ),

            explanation_risk_factors=(
                layer3_result.risk_factors
            ),

            strongest_factor=(
                layer3_result.strongest_factor
            ),

            strongest_risk=(
                layer3_result.strongest_risk
            ),

            explanation_quality=(
                layer3_result.explanation_quality
            ),

            # ------------------------------------------------
            # Layer 4
            # ------------------------------------------------

            market_context_score=(
                layer4_result.market_context_score
            ),

            sector_context_score=(
                layer4_result.sector_context_score
            ),

            overall_context_score=(
                layer4_result.overall_context_score
            ),

            market_alignment=(
                layer4_result.market_alignment
            ),

            sector_alignment=(
                layer4_result.sector_alignment
            ),

            overall_alignment=(
                layer4_result.overall_alignment
            ),

            context_strength=(
                layer4_result.context_strength
            ),

            context_risk=(
                layer4_result.context_risk
            ),

            overall_context=(
                layer4_result.overall_context
            ),

            market_regime=market_regime,
            market_direction=market_direction,
            market_confidence=market_confidence,

            sector=sector,
            sector_direction=sector_direction,
            sector_confidence=sector_confidence,

            # ------------------------------------------------
            # Layer 5
            # ------------------------------------------------

            recommendation=(
                layer5_result.recommendation
            ),

            recommendation_confidence=(
                layer5_result.recommendation_confidence
            ),

            recommendation_quality=(
                layer5_result.recommendation_quality
            ),

            recommendation_strength=(
                layer5_result.recommendation_strength
            ),

            recommendation_risk=(
                layer5_result.recommendation_risk
            ),

            recommendation_score=(
                layer5_result.recommendation_score
            ),

            recommendation_supporting_factors=(
                layer5_result.supporting_factors
            ),

            recommendation_conflicting_factors=(
                layer5_result.conflicting_factors
            ),

            recommendation_warnings=(
                layer5_result.warnings
            ),

            action_guidance=(
                layer5_result.action_guidance
            ),

            position_guidance=(
                layer5_result.position_guidance
            ),

            # ------------------------------------------------
            # Layer 6
            # ------------------------------------------------

            validation_status=(
                layer6_result.validation_status
            ),

            validation_score=(
                layer6_result.validation_score
            ),

            validation_quality=(
                layer6_result.validation_quality
            ),

            decision_consistency=(
                layer6_result.decision_consistency
            ),

            confidence_consistency=(
                layer6_result.confidence_consistency
            ),

            reliability_consistency=(
                layer6_result.reliability_consistency
            ),

            context_consistency=(
                layer6_result.context_consistency
            ),

            risk_consistency=(
                layer6_result.risk_consistency
            ),

            conflict_consistency=(
                layer6_result.conflict_consistency
            ),

            passed_checks=(
                layer6_result.passed_checks
            ),

            failed_checks=(
                layer6_result.failed_checks
            ),

            validation_warnings=(
                layer6_result.warnings
            ),

            validation_risk=(
                layer6_result.validation_risk
            ),

            recommendation_valid=(
                layer6_result.recommendation_valid
            ),

            recommendation_trust=(
                layer6_result.recommendation_trust
            ),
        )

        layer7_result = self.layer7.aggregate(
            layer7_input
        )

        # ====================================================
        # SERIALIZE FINAL RESULT
        # ====================================================

        if hasattr(layer7_result, "model_dump"):
            final_result = layer7_result.model_dump()
        elif hasattr(layer7_result, "dict"):
            final_result = layer7_result.dict()
        elif hasattr(layer7_result, "to_dict"):
            final_result = layer7_result.to_dict()
        elif hasattr(layer7_result, "__dict__"):
            final_result = dict(layer7_result.__dict__)
        else:
            final_result = layer7_result

        # Layer 7 stays aggregation-only. Expose Layer 1 diagnostics as
        # additive API fields so users can understand a HOLD decision.

        if isinstance(final_result, dict):
            final_result.update({
                "ensemble_signal": ensemble_signal,
                "directional_agreement_count": directional_agreement_count,
                "neutral_agreement_count": neutral_agreement_count,
                "agreement_type": agreement_type,
                "directional_conflict": directional_conflict,
                "disagreement_type": disagreement_type,
                "neutral_agreement_count": neutral_agreement_count,
                "agreement_type": agreement_type,
                "directional_conflict": directional_conflict,
                "disagreement_type": disagreement_type,
                "decision_reason_code": layer1_result.decision_reason_code,
                "decision_reason": layer1_result.decision_reason,
                "leading_direction": layer1_result.leading_direction,
                "leading_direction_score": layer1_result.leading_direction_score,
                "directional_threshold": layer1_result.directional_threshold,
                "confidence_interpretation": layer1_result.confidence_interpretation,
                "decision_bullish_score": layer1_result.bullish_score,
                "decision_bearish_score": layer1_result.bearish_score,
                "decision_neutral_score": layer1_result.neutral_score,
                "decision_reasons": list(layer1_result.reasons),
                "decision_warnings": list(layer1_result.warnings),
            })
        return final_result