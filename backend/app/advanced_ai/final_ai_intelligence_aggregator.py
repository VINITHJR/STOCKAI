from .layer7_models import (
    FinalAIIntelligenceInput,
    FinalAIIntelligenceResult,
)


class FinalAIIntelligenceAggregator:
    """
    Phase 14 — Layer 7

    Final AI Intelligence Aggregator.

    Layer 7 combines the outputs of Layers 1–6 into one unified
    Phase 14 result.

    IMPORTANT:
    - Does not create a new AI decision.
    - Does not recalculate reliability.
    - Does not change the recommendation.
    - Does not override validation.
    - Does not execute trades.
    - Does not modify Layers 1–6.
    """

    def __init__(self):
        self.name = "Final AI Intelligence Aggregator"
        self.layer = 7

    # ========================================================
    # OVERALL STATUS
    # ========================================================

    @staticmethod
    def _calculate_overall_status(
        validation_status: str,
        recommendation_valid: bool,
        recommendation_risk: str,
    ) -> str:
        """
        Determine the final system status.

        Layer 7 does not reinterpret the recommendation.
        It only summarizes the validated state.
        """

        validation_status = str(
            validation_status or ""
        ).upper()

        recommendation_risk = str(
            recommendation_risk or ""
        ).upper()

        if (
            validation_status == "INVALID"
            or not recommendation_valid
        ):
            return "INVALID"

        if recommendation_risk == "HIGH":
            return "VALID_WITH_HIGH_RISK"

        if validation_status == "VALID_WITH_WARNINGS":
            return "VALID_WITH_WARNINGS"

        return "VALID"

    # ========================================================
    # COMBINE WARNINGS
    # ========================================================

    @staticmethod
    def _combine_warnings(
        recommendation_warnings,
        validation_warnings,
        reliability_score,
        conflict,
        context_risk,
        directional_conflict=False,
        disagreement_type="none",
    ):
        """
        Combine warnings from Layers 5 and 6 without creating
        duplicate messages.
        """

        warnings = []

        warnings.extend(
            recommendation_warnings or []
        )

        warnings.extend(
            validation_warnings or []
        )

        if reliability_score < 0.35:
            warnings.append(
                "Overall AI reliability is very low."
            )

        if directional_conflict:
            if not any("opposing bullish and bearish" in str(item).lower() for item in warnings):
                warnings.append("Models produced opposing bullish and bearish directional signals.")
        elif disagreement_type == "directional_neutral_disagreement":
            if not any(
                "directional and neutral" in str(item).lower()
                or "partial confirmation" in str(item).lower()
                for item in warnings
            ):
                warnings.append(
                    "Models disagree between a directional and neutral signal; "
                    "this is partial confirmation, not opposing-direction conflict."
                )
        elif conflict:
            warnings.append("Model outputs disagree; inspect individual signals.")

        if str(context_risk).upper() == "HIGH":
            warnings.append(
                "Market context risk is high."
            )

        return list(
            dict.fromkeys(warnings)
        )

    # ========================================================
    # COMBINE SUPPORTING FACTORS
    # ========================================================

    @staticmethod
    def _combine_supporting_factors(
        explanation_factors,
        recommendation_factors,
    ):
        factors = []

        factors.extend(
            explanation_factors or []
        )

        factors.extend(
            recommendation_factors or []
        )

        return list(
            dict.fromkeys(factors)
        )

    # ========================================================
    # COMBINE CONFLICTING FACTORS
    # ========================================================

    @staticmethod
    def _combine_conflicting_factors(
        explanation_factors,
        recommendation_factors,
    ):
        factors = []

        factors.extend(
            explanation_factors or []
        )

        factors.extend(
            recommendation_factors or []
        )

        return list(
            dict.fromkeys(factors)
        )

    # ========================================================
    # MAIN AGGREGATION
    # ========================================================

    def aggregate(
        self,
        data: FinalAIIntelligenceInput,
    ) -> FinalAIIntelligenceResult:
        """
        Aggregate Layers 1–6 into the final Phase 14 result.
        """

        # ----------------------------------------------------
        # Overall status
        # ----------------------------------------------------

        overall_status = (
            self._calculate_overall_status(
                data.validation_status,
                data.recommendation_valid,
                data.recommendation_risk,
            )
        )

        # ----------------------------------------------------
        # Combined supporting factors
        # ----------------------------------------------------

        supporting_factors = (
            self._combine_supporting_factors(
                data.explanation_supporting_factors,
                data.recommendation_supporting_factors,
            )
        )

        # ----------------------------------------------------
        # Combined conflicting factors
        # ----------------------------------------------------

        conflicting_factors = (
            self._combine_conflicting_factors(
                data.conflicting_factors,
                data.recommendation_conflicting_factors,
            )
        )

        # ----------------------------------------------------
        # Combined warnings
        # ----------------------------------------------------

        warnings = (
            self._combine_warnings(
                data.recommendation_warnings,
                data.validation_warnings,
                data.reliability_score,
                data.conflict,
                data.context_risk,
                data.directional_conflict,
                data.disagreement_type,
            )
        )

        # ----------------------------------------------------
        # Layer status
        # ----------------------------------------------------

        layer_status = {
            "layer_1": "complete",
            "layer_2": "complete",
            "layer_3": "complete",
            "layer_4": "complete",
            "layer_5": "complete",
            "layer_6": "complete",
            "layer_7": "complete",
        }

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "phase": 14,
            "layer": 7,
            "component": "final_ai_intelligence_aggregator",
            "layers_included": [1, 2, 3, 4, 5, 6, 7],
            "recommendation_source": "layer_5",
            "validation_source": "layer_6",
            "aggregation_only": True,
            "validation_scope": "internal_pipeline_consistency_only",
            "model_evidence_note": (
                "AI signal and ensemble are correlated summaries of the same "
                "underlying Phase 9 model outputs and are not treated as independent "
                "top-level evidence."
            ),
            "technical_evidence_note": (
                "Phase 14.1 independent OHLCV technical intelligence is registered and contributes 7% of Layer 1 directional evidence."
            ),
            "agreement_semantics": (
                "agreement_count/directional_agreement_count count models supporting a bullish or bearish ensemble direction; "
                "neutral_agreement_count is reported separately. Legacy conflict means any model disagreement, while "
                "directional_conflict means at least one bullish and one bearish model signal."
            ),
        }

        # ----------------------------------------------------
        # Return final result
        # ----------------------------------------------------

        return FinalAIIntelligenceResult(
            instrument_key=data.instrument_key,

            # ------------------------------------------------
            # Decision
            # ------------------------------------------------
            decision=data.decision,
            decision_confidence=data.decision_confidence,

            # ------------------------------------------------
            # Recommendation
            # ------------------------------------------------
            recommendation=data.recommendation,
            recommendation_confidence=(
                data.recommendation_confidence
            ),
            recommendation_quality=(
                data.recommendation_quality
            ),
            recommendation_strength=(
                data.recommendation_strength
            ),
            recommendation_risk=(
                data.recommendation_risk
            ),

            # ------------------------------------------------
            # Reliability
            # ------------------------------------------------
            reliability_score=data.reliability_score,
            confidence_grade=data.confidence_grade,
            confidence_band=data.confidence_band,
            reliability_level=data.reliability_level,

            # ------------------------------------------------
            # Market Context
            # ------------------------------------------------
            market_context=data.overall_context,
            market_context_score=data.market_context_score,
            sector_context_score=data.sector_context_score,
            overall_context_score=data.overall_context_score,

            market_alignment=data.market_alignment,
            sector_alignment=data.sector_alignment,
            overall_alignment=data.overall_alignment,

            context_strength=data.context_strength,
            context_risk=data.context_risk,

            market_regime=data.market_regime,
            market_direction=data.market_direction,
            sector=data.sector,
            sector_direction=data.sector_direction,

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------
            validation_status=data.validation_status,
            validation_score=data.validation_score,
            validation_quality=data.validation_quality,

            recommendation_valid=(
                data.recommendation_valid
            ),
            recommendation_trust=(
                data.recommendation_trust
            ),
            validation_risk=data.validation_risk,

            # ------------------------------------------------
            # Explainability
            # ------------------------------------------------
            decision_summary=data.decision_summary,
            confidence_explanation=(
                data.confidence_explanation
            ),
            reliability_explanation=(
                data.reliability_explanation
            ),
            final_explanation=data.final_explanation,

            strongest_factor=data.strongest_factor,
            strongest_risk=data.strongest_risk,

            bullish_factors=data.bullish_factors,
            bearish_factors=data.bearish_factors,
            neutral_factors=data.neutral_factors,

            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            warnings=warnings,

            # ------------------------------------------------
            # Guidance
            # ------------------------------------------------
            action_guidance=data.action_guidance,
            position_guidance=data.position_guidance,

            # ------------------------------------------------
            # Model information
            # ------------------------------------------------
            model_count=data.model_count,
            agreement_count=data.agreement_count,
            conflict=data.conflict,
            conflict_level=data.conflict_level,
            directional_agreement_count=data.directional_agreement_count,
            neutral_agreement_count=data.neutral_agreement_count,
            agreement_type=data.agreement_type,
            directional_conflict=data.directional_conflict,
            disagreement_type=data.disagreement_type,

            # ------------------------------------------------
            # Layer summary
            # ------------------------------------------------
            layers_completed=[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],

            layer_status=layer_status,

            # ------------------------------------------------
            # Final status
            # ------------------------------------------------
            overall_status=overall_status,
            status="success",

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------
            metadata=metadata,
        )