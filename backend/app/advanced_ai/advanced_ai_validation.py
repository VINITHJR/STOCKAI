from __future__ import annotations

from .layer6_models import AIValidationInput, AIValidationResult


class AdvancedAIValidation:
    """
    Phase 14 — Layer 6.

    Validates internal consistency of the Phase 14 pipeline.

    This is NOT a claim that the recommendation will be profitable or
    that the future market outcome is correct. It checks whether the
    recommendation is consistent with the preceding layers.
    """

    def __init__(self):
        self.name = "Advanced AI Validation"
        self.layer = 6

    @staticmethod
    def _normalize_decision(value: str) -> str:
        value = str(value or "").strip().upper()
        if value in {"BUY", "BULLISH"}:
            return "BUY"
        if value in {"SELL", "BEARISH"}:
            return "SELL"
        return "HOLD"


    @classmethod
    def _check_decision_consistency(
        cls,
        ai_decision: str,
        recommendation: str,
    ) -> bool:
        valid_decisions = {
            "BUY", "BULLISH",
            "SELL", "BEARISH",
            "HOLD", "NEUTRAL",
        }

        ai_value = str(ai_decision or "").strip().upper()
        recommendation_value = str(
            recommendation or ""
        ).strip().upper()

        # Invalid or missing decisions must never pass validation.
        if (
            ai_value not in valid_decisions
            or recommendation_value not in valid_decisions
        ):
            return False

        return (
            cls._normalize_decision(ai_value)
            == cls._normalize_decision(recommendation_value)
        )

    @staticmethod
    def _check_confidence_consistency(
        ai_confidence: float,
        recommendation_confidence: float,
    ) -> bool:
        # Recommendation confidence can be modestly higher because it
        # incorporates contextual information, but not by a large jump.
        return recommendation_confidence <= ai_confidence + 0.10

    @staticmethod
    def _check_reliability_consistency(
        reliability_score: float,
        recommendation_confidence: float,
        recommendation_quality: str,
    ) -> bool:
        quality = str(recommendation_quality or "").upper()

        if reliability_score < 0.35:
            return (
                quality not in {"HIGH_QUALITY", "GOOD", "HIGH_QUALITY_HOLD"}
                and recommendation_confidence <= 0.60
            )

        if reliability_score < 0.50:
            return quality not in {"HIGH_QUALITY", "HIGH_QUALITY_HOLD"}

        return True

    @staticmethod
    def _check_context_consistency(
        overall_context_score: float,
        recommendation_confidence: float,
        recommendation_quality: str,
    ) -> bool:
        quality = str(recommendation_quality or "").upper()

        if overall_context_score < 0.30:
            if quality in {"HIGH_QUALITY", "GOOD", "HIGH_QUALITY_HOLD"}:
                return False
            if recommendation_confidence > 0.70:
                return False

        return True

    @staticmethod
    def _check_risk_consistency(
        reliability_score: float,
        recommendation_risk: str,
        context_risk: str,
        source_risk_level: str,
        conflict: bool,
        conflict_level: str,
    ) -> bool:
        recommendation_risk = str(recommendation_risk or "").upper()
        context_risk = str(context_risk or "").upper()
        source_risk_level = str(source_risk_level or "").upper()
        conflict_level = str(conflict_level or "").upper()

        if (
            reliability_score < 0.35
            or context_risk == "HIGH"
            or source_risk_level == "HIGH"
            or conflict_level == "HIGH"
        ):
            return recommendation_risk == "HIGH"

        if (
            reliability_score < 0.50
            or context_risk == "MEDIUM"
            or source_risk_level == "MEDIUM"
            or (conflict and conflict_level == "MEDIUM")
        ):
            return recommendation_risk in {"HIGH", "MEDIUM"}

        return recommendation_risk in {"LOW", "MEDIUM", "HIGH"}

    @staticmethod
    def _check_conflict_consistency(
        conflict: bool,
        conflict_level: str,
        recommendation_confidence: float,
        recommendation_quality: str,
        recommendation_strength: str,
    ) -> bool:
        level = str(conflict_level or "").lower()
        quality = str(recommendation_quality or "").upper()
        strength = str(recommendation_strength or "").upper()

        if conflict and level == "high":
            if quality in {"HIGH_QUALITY", "GOOD"}:
                return False
            if strength in {"VERY_STRONG", "STRONG"}:
                return False
            if recommendation_confidence > 0.70:
                return False

        if conflict and level == "medium":
            if strength == "VERY_STRONG":
                return False

        return True

    @staticmethod
    def _calculate_validation_score(checks) -> float:
        if not checks:
            return 0.0
        return round(sum(bool(value) for value in checks) / len(checks), 4)

    @staticmethod
    def _validation_quality(score: float, failed_count: int) -> str:
        if failed_count == 0 and score >= 0.90:
            return "EXCELLENT"
        if score >= 0.80:
            return "GOOD"
        if score >= 0.60:
            return "MODERATE"
        if score >= 0.40:
            return "WEAK"
        return "POOR"

    @staticmethod
    def _validation_status(score: float, failed_count: int) -> str:
        if failed_count == 0:
            return "VALID"
        if score >= 0.60:
            return "VALID_WITH_WARNINGS"
        return "INVALID"

    @staticmethod
    def _validation_risk(status: str, score: float, recommendation_risk: str) -> str:
        recommendation_risk = str(recommendation_risk or "").upper()

        if status == "INVALID":
            return "HIGH"
        if recommendation_risk == "HIGH" or score < 0.70:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _recommendation_trust(
        validation_status: str,
        validation_score: float,
        recommendation_risk: str,
    ) -> str:
        recommendation_risk = str(recommendation_risk or "").upper()

        if validation_status == "INVALID":
            return "DO_NOT_TRUST"

        if validation_status == "VALID" and validation_score >= 0.90 and recommendation_risk == "LOW":
            return "HIGH"

        if validation_score >= 0.75:
            return "MODERATE"

        return "LOW"

    def validate(self, data: AIValidationInput) -> AIValidationResult:
        decision_consistency = self._check_decision_consistency(
            data.ai_decision,
            data.recommendation,
        )
        confidence_consistency = self._check_confidence_consistency(
            data.ai_confidence,
            data.recommendation_confidence,
        )
        reliability_consistency = self._check_reliability_consistency(
            data.reliability_score,
            data.recommendation_confidence,
            data.recommendation_quality,
        )
        context_consistency = self._check_context_consistency(
            data.overall_context_score,
            data.recommendation_confidence,
            data.recommendation_quality,
        )
        risk_consistency = self._check_risk_consistency(
            data.reliability_score,
            data.recommendation_risk,
            data.context_risk,
            data.risk_level,
            data.conflict,
            data.conflict_level,
        )
        conflict_consistency = self._check_conflict_consistency(
            data.conflict,
            data.conflict_level,
            data.recommendation_confidence,
            data.recommendation_quality,
            data.recommendation_strength,
        )

        checks = [
            decision_consistency,
            confidence_consistency,
            reliability_consistency,
            context_consistency,
            risk_consistency,
            conflict_consistency,
        ]

        named_checks = [
            ("Decision consistency", decision_consistency),
            ("Confidence consistency", confidence_consistency),
            ("Reliability consistency", reliability_consistency),
            ("Market context consistency", context_consistency),
            ("Risk consistency", risk_consistency),
            ("Model conflict consistency", conflict_consistency),
        ]

        passed_checks = [name for name, passed in named_checks if passed]
        failed_checks = [name for name, passed in named_checks if not passed]

        validation_score = self._calculate_validation_score(checks)
        validation_quality = self._validation_quality(
            validation_score,
            len(failed_checks),
        )
        validation_status = self._validation_status(
            validation_score,
            len(failed_checks),
        )
        validation_risk = self._validation_risk(
            validation_status,
            validation_score,
            data.recommendation_risk,
        )
        recommendation_trust = self._recommendation_trust(
            validation_status,
            validation_score,
            data.recommendation_risk,
        )

        warnings = []

        if not decision_consistency:
            warnings.append("Layer 5 recommendation does not match the Layer 1 decision.")
        if not confidence_consistency:
            warnings.append("Recommendation confidence is higher than the Layer 1 confidence allows.")
        if not reliability_consistency:
            warnings.append("Recommendation presentation is too strong for the measured reliability.")
        if not context_consistency:
            warnings.append("Recommendation presentation is too strong for the available context.")
        if not risk_consistency:
            warnings.append("Recommendation risk is understated relative to reliability, context or model conflict.")
        if not conflict_consistency:
            warnings.append("Recommendation presentation is too strong for the detected model conflict.")

        if data.reliability_score < 0.35:
            warnings.append("AI reliability is very low.")
        if data.conflict and str(data.conflict_level).lower() == "high":
            warnings.append("High model conflict is present.")
        if str(data.context_risk).upper() == "HIGH":
            warnings.append("Market context risk is high.")

        warnings.append(
            "Validation checks internal pipeline consistency; it does not predict future market performance."
        )

        return AIValidationResult(
            instrument_key=data.instrument_key,
            recommendation=data.recommendation,
            recommendation_confidence=data.recommendation_confidence,
            validation_status=validation_status,
            validation_score=validation_score,
            validation_quality=validation_quality,
            decision_consistency=decision_consistency,
            confidence_consistency=confidence_consistency,
            reliability_consistency=reliability_consistency,
            context_consistency=context_consistency,
            risk_consistency=risk_consistency,
            conflict_consistency=conflict_consistency,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=list(dict.fromkeys(warnings)),
            validation_risk=validation_risk,
            recommendation_valid=(validation_status != "INVALID"),
            recommendation_trust=recommendation_trust,
            ai_decision=data.ai_decision,
            ai_confidence=data.ai_confidence,
            reliability_score=data.reliability_score,
            overall_context_score=data.overall_context_score,
            recommendation_risk=data.recommendation_risk,
            status="success",
        )
