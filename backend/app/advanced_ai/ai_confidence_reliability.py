from __future__ import annotations

from .layer2_models import AIConfidenceInput, AIConfidenceResult


class AIConfidenceReliability:
    """
    Phase 14 - Layer 2
    AI Confidence & Reliability Intelligence.

    Reliability is intentionally different from Layer 1 confidence.

    It measures:
    - quality of the multi-model evidence
    - model agreement/conflict
    - quality of the market-context source
    - coherence of the model summaries
    - availability/confirmation from supporting analyses

    It does NOT feed Layer 1 confidence back into itself. This avoids
    circular "confidence -> reliability -> recommendation confidence"
    inflation.
    """

    MODEL_WEIGHT = 0.40
    MARKET_WEIGHT = 0.20
    SIGNAL_WEIGHT = 0.20
    SUPPORTING_WEIGHT = 0.20

    HIGH_CONFLICT_FACTOR = 0.75
    MEDIUM_CONFLICT_FACTOR = 0.88
    LOW_CONFLICT_FACTOR = 0.95
    NON_DIRECTIONAL_DISAGREEMENT_FACTOR = 0.97

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalize_signal(signal: str) -> str:
        value = str(signal or "").strip().lower()
        if value in {"bullish", "buy", "long", "positive", "up"}:
            return "bullish"
        if value in {"bearish", "sell", "short", "negative", "down"}:
            return "bearish"
        return "neutral"


    @classmethod
    def _agreement_ratio(cls, data: AIConfidenceInput) -> float:
        """Calculate reliability using directional agreement only."""
        try:
            model_count = max(0, int(data.model_count or 0))
        except (TypeError, ValueError, OverflowError):
            return 0.0

        if model_count == 0:
            return 0.0

        # Neutral ensemble signals never provide directional confirmation.
        if cls._normalize_signal(data.ensemble_signal) == "neutral":
            return 0.0

        # Prefer the explicit directional count.
        # Fall back to the legacy field for older callers.
        raw_count = getattr(data, "directional_agreement_count", None)
        if raw_count is None:
            raw_count = data.agreement_count

        try:
            directional_count = int(raw_count or 0)
        except (TypeError, ValueError, OverflowError):
            return 0.0

        directional_count = max(0, min(directional_count, model_count))
        return directional_count / model_count

    @classmethod
    def _get_conflict_factor(
        cls,
        conflict: bool,
        conflict_level: str,
        directional_conflict: bool = False,
    ) -> float:
        """Penalize opposing directions more than neutral disagreement.

        Legacy ``conflict`` remains the any-disagreement flag. If directions
        are not directly opposed, apply a small caution factor only.
        """
        if directional_conflict:
            level = str(conflict_level or "").strip().lower()
            if level == "high":
                return cls.HIGH_CONFLICT_FACTOR
            if level == "medium":
                return cls.MEDIUM_CONFLICT_FACTOR
            return cls.LOW_CONFLICT_FACTOR
        if conflict:
            return cls.NON_DIRECTIONAL_DISAGREEMENT_FACTOR
        return 1.0

    def _calculate_model_reliability(self, data: AIConfidenceInput) -> float:
        """
        Use the Phase 9 ensemble as the primary model-quality measure.

        The AI signal is a reconstruction from the same RF/LSTM outputs,
        so it is not added as a second independent reliability source.
        """
        if data.model_count <= 0:
            return 0.20

        ensemble_confidence = self._clamp(data.ensemble_confidence)
        agreement = self._agreement_ratio(data)

        reliability = (
            0.70 * ensemble_confidence
            + 0.30 * agreement
        )
        return self._clamp(reliability)

    def _calculate_market_reliability(self, data: AIConfidenceInput) -> float:
        confidence = self._clamp(data.market_confidence)
        alignment = str(data.market_alignment or "").lower()

        if alignment == "supportive":
            return self._clamp(0.50 + 0.50 * confidence)

        if alignment == "conflicting":
            return self._clamp(0.50 * confidence)

        # Neutral alignment means the source exists but gives limited
        # directional confirmation.
        return self._clamp(0.25 + 0.25 * confidence)

    def _calculate_signal_reliability(self, data: AIConfidenceInput) -> float:
        """
        AI and ensemble are correlated summaries. Their alignment is
        therefore treated as coherence, not as independent confirmation.
        """
        agreement = self._agreement_ratio(data)

        if data.model_count <= 0:
            return 0.25

        # 1/2 agreement => 0.75, 2/2 => 1.00, 0/2 => 0.50.
        coherence = 0.50 + 0.50 * agreement

        if data.signal_alignment == "conflicting":
            coherence *= 0.75

        return self._clamp(coherence)

    def _calculate_supporting_reliability(self, data: AIConfidenceInput) -> float:
        """Score supporting signals with their source confidence.

        Neutral signals are not directional confirmation. Missing confidence
        (None) defaults to 1.0 for backward compatibility; explicit 0.0
        confidence contributes no directional evidence.
        """
        sources = [
            (self._normalize_signal(data.technical_signal), data.technical_confidence),
            (self._normalize_signal(data.fundamental_signal), data.fundamental_confidence),
            (self._normalize_signal(data.news_signal), data.news_confidence),
        ]
        directional = []
        for signal, raw_confidence in sources:
            if signal not in {"bullish", "bearish"}:
                continue
            weight = 1.0 if raw_confidence is None else self._clamp(raw_confidence)
            directional.append((signal, weight))

        if not directional:
            return 0.50

        total_weight = sum(weight for _, weight in directional)
        if total_weight <= 0.0:
            return 0.50

        decision = self._normalize_signal(data.decision)
        confidence_coverage = self._clamp(total_weight / len(directional))
        if decision == "neutral":
            return self._clamp(0.50 - 0.05 * confidence_coverage)

        matching_weight = sum(weight for signal, weight in directional if signal == decision)
        opposing_weight = total_weight - matching_weight
        agreement = matching_weight / total_weight
        opposition = opposing_weight / total_weight
        score = 0.50 + confidence_coverage * (0.30 * agreement - 0.25 * opposition)
        return self._clamp(score)

    @staticmethod
    def _confidence_grade(score: float) -> str:
        if score >= 0.85:
            return "A+"
        if score >= 0.75:
            return "A"
        if score >= 0.65:
            return "B"
        if score >= 0.55:
            return "C"
        if score >= 0.40:
            return "D"
        return "E"

    @staticmethod
    def _confidence_band(score: float) -> str:
        if score >= 0.80:
            return "VERY_HIGH"
        if score >= 0.65:
            return "HIGH"
        if score >= 0.50:
            return "MODERATE"
        if score >= 0.35:
            return "LOW"
        return "VERY_LOW"

    @staticmethod
    def _reliability_level(score: float) -> str:
        if score >= 0.80:
            return "VERY_HIGH"
        if score >= 0.65:
            return "HIGH"
        if score >= 0.50:
            return "MODERATE"
        if score >= 0.35:
            return "LOW"
        return "VERY_LOW"

    @staticmethod
    def _decision_strength(decision: str, confidence: float) -> str:
        decision = str(decision or "").upper()

        if decision == "HOLD":
            if confidence >= 0.70:
                return "STRONG_HOLD"
            if confidence >= 0.50:
                return "MODERATE_HOLD"
            return "WEAK_HOLD"

        if confidence >= 0.80:
            return "VERY_STRONG"
        if confidence >= 0.65:
            return "STRONG"
        if confidence >= 0.55:
            return "MODERATE"
        return "WEAK"

    @staticmethod
    def _risk_level(reliability_score: float, data: AIConfidenceInput) -> str:
        conflict_level = str(data.conflict_level or "").lower()

        if data.directional_conflict and conflict_level == "high":
            return "HIGH"

        if reliability_score < 0.35:
            return "HIGH"

        if (
            (data.directional_conflict and conflict_level == "medium")
            or (data.conflict and conflict_level == "medium")
            or reliability_score < 0.50
        ):
            return "MEDIUM"

        if data.conflict and not data.directional_conflict:
            return "MEDIUM" if reliability_score < 0.70 else "LOW"

        if reliability_score < 0.70:
            return "LOW"

        return "VERY_LOW"

    @staticmethod
    def _recommendation_quality(decision: str, reliability_score: float) -> str:
        if decision == "HOLD":
            if reliability_score >= 0.70:
                return "HIGH_QUALITY_HOLD"
            if reliability_score >= 0.50:
                return "REASONABLE_HOLD"
            if reliability_score >= 0.35:
                return "CAUTIOUS_HOLD"
            return "UNCERTAIN_HOLD"

        if reliability_score >= 0.80:
            return "HIGH_QUALITY"
        if reliability_score >= 0.65:
            return "GOOD"
        if reliability_score >= 0.50:
            return "MODERATE"
        if reliability_score >= 0.35:
            return "CAUTIOUS"
        return "LOW"

    @staticmethod
    def _evaluate_decision_confidence(confidence: float) -> list[str]:
        if confidence >= 0.80:
            return ["Layer 1 decision confidence is very high."]
        if confidence >= 0.65:
            return ["Layer 1 decision confidence is high."]
        if confidence >= 0.55:
            return ["Layer 1 decision confidence is moderate."]
        return ["Layer 1 decision confidence is below the moderate threshold."]

    @staticmethod
    def _evaluate_decision_risk(confidence: float) -> list[str]:
        if confidence < 0.55:
            return ["Layer 1 decision has limited directional separation."]
        return []

    def analyze(self, data: AIConfidenceInput) -> AIConfidenceResult:
        model_reliability = self._calculate_model_reliability(data)
        market_reliability = self._calculate_market_reliability(data)
        signal_reliability = self._calculate_signal_reliability(data)
        supporting_reliability = self._calculate_supporting_reliability(data)

        reliability_score = (
            model_reliability * self.MODEL_WEIGHT
            + market_reliability * self.MARKET_WEIGHT
            + signal_reliability * self.SIGNAL_WEIGHT
            + supporting_reliability * self.SUPPORTING_WEIGHT
        )

        conflict_factor = self._get_conflict_factor(
            data.conflict,
            data.conflict_level,
            data.directional_conflict,
        )
        if data.conflict or data.directional_conflict:
            reliability_score *= conflict_factor

        reliability_score = self._clamp(reliability_score)

        confidence_factors = self._evaluate_decision_confidence(
            self._clamp(data.decision_confidence)
        )
        risk_factors = self._evaluate_decision_risk(
            self._clamp(data.decision_confidence)
        )

        if model_reliability >= 0.75:
            confidence_factors.append("Model evidence provides strong reliability.")
        elif model_reliability >= 0.55:
            confidence_factors.append("Model evidence provides moderate reliability.")
        else:
            risk_factors.append("Model evidence provides limited reliability.")

        if data.market_alignment == "supportive":
            confidence_factors.append("The market regime supports the AI direction.")
        elif data.market_alignment == "conflicting":
            risk_factors.append("The market regime conflicts with the AI direction.")
        else:
            risk_factors.append("The market regime provides limited confirmation.")

        if data.signal_alignment == "aligned":
            confidence_factors.append("AI and ensemble summaries are coherent.")
        elif data.signal_alignment == "conflicting":
            risk_factors.append("AI and ensemble summaries are conflicting.")
        else:
            risk_factors.append("AI and ensemble summaries provide limited coherence.")

        if supporting_reliability >= 0.75:
            confidence_factors.append("Supporting analysis provides strong confirmation.")
        elif supporting_reliability >= 0.55:
            confidence_factors.append("Supporting analysis provides moderate confirmation.")
        else:
            risk_factors.append("Supporting analysis provides limited confirmation.")

        ensemble_signal = self._normalize_signal(data.ensemble_signal)
        if data.directional_conflict:
            level = str(data.conflict_level or "unknown").lower()
            risk_factors.append(
                "Models produced opposing bullish and bearish directional signals "
                f"(conflict level: {level}); the directional-conflict penalty applies."
            )
        elif data.disagreement_type == "directional_neutral_disagreement":
            risk_factors.append(
                "Models disagree between a directional signal and neutral; "
                "confirmation is partial, but the models are not directly opposed."
            )
        elif data.conflict:
            risk_factors.append(
                "Model outputs disagree; inspect individual signals before relying on direction."
            )


        if data.model_count <= 0:
            risk_factors.append("No model count is available.")

        elif ensemble_signal == "neutral":
            risk_factors.append(
                "The ensemble signal is neutral; there is no bullish or bearish "
                "directional confirmation. Neutral consensus is not directional evidence."
            )

        else:
            raw_directional_count = getattr(
                data, "directional_agreement_count", None
            )
            if raw_directional_count is None:
                raw_directional_count = data.agreement_count

            try:
                directional_count = max(
                    0,
                    min(
                        int(raw_directional_count or 0),
                        int(data.model_count),
                    ),
                )
            except (TypeError, ValueError, OverflowError):
                directional_count = 0

            if directional_count == data.model_count and data.model_count >= 2:
                confidence_factors.append(
                    f"All {directional_count} models confirm "
                    f"the {ensemble_signal} direction."
                )
            elif directional_count > 0:
                risk_factors.append(
                    f"Only {directional_count} of {data.model_count} models confirm "
                    f"the {ensemble_signal} direction."
                )
            else:
                risk_factors.append(
                    f"No models confirm the {ensemble_signal} direction."
                )

        raw_directional_count = getattr(
            data, "directional_agreement_count", None
        )

        if ensemble_signal == "neutral":
            directional_result_count = 0
        else:
            if raw_directional_count is None:
                raw_directional_count = data.agreement_count

            try:
                directional_result_count = max(
                    0,
                    min(
                        int(raw_directional_count or 0),
                        int(data.model_count),
                    ),
                )
            except (TypeError, ValueError, OverflowError):
                directional_result_count = 0

        return AIConfidenceResult(
            instrument_key=data.instrument_key,
            decision=data.decision,
            decision_confidence=self._clamp(data.decision_confidence),
            reliability_score=round(reliability_score, 4),
            confidence_grade=self._confidence_grade(reliability_score),
            confidence_band=self._confidence_band(reliability_score),
            reliability_level=self._reliability_level(reliability_score),
            recommendation_quality=self._recommendation_quality(
                data.decision,
                reliability_score,
            ),
            decision_strength=self._decision_strength(
                data.decision,
                self._clamp(data.decision_confidence),
            ),
            risk_level=self._risk_level(reliability_score, data),
            model_reliability=round(model_reliability, 4),
            market_reliability=round(market_reliability, 4),
            signal_reliability=round(signal_reliability, 4),
            supporting_reliability=round(supporting_reliability, 4),
            confidence_factors=list(dict.fromkeys(confidence_factors)),
            risk_factors=list(dict.fromkeys(risk_factors)),
            model_count=data.model_count,
            agreement_count=data.agreement_count,
            directional_agreement_count=directional_result_count,
            conflict=data.conflict,
            conflict_level=data.conflict_level,
            directional_conflict=data.directional_conflict,
            disagreement_type=data.disagreement_type,
            agreement_type=data.agreement_type,
            neutral_agreement_count=data.neutral_agreement_count,
            market_regime=data.market_regime,
            market_direction=data.market_direction,
            status="success",
        )
