from __future__ import annotations

from .layer1_models import AIDecisionInput, AIDecisionResult


class AIDecisionIntelligence:
    """
    Phase 14 - Layer 1
    AI Decision Intelligence.

    Layer 1 converts the stock-specific Phase 9 model/ensemble
    evidence plus independent contextual signals into one decision.

    IMPORTANT:
    - AI and ensemble are correlated summaries of the same Phase 9
      RF/LSTM outputs, but Phase 14 uses the configured 40% / 25%
      source weights for traceable scoring.
    - Phase 14.1 is an independent OHLCV technical source and carries
      the configured 7% Layer 1 weight.
    - Phase 14.2 stock regime is contextual/diagnostic and carries no
      duplicate Layer 1 weight.
    - Model conflict penalties affect only the AI/ensemble bucket.
    """

    # Phase 14 Layer 1 evidence weights.
    # These six sources sum to 1.0.
    AI_WEIGHT = 0.40
    ENSEMBLE_WEIGHT = 0.25
    MARKET_WEIGHT = 0.20
    TECHNICAL_WEIGHT = 0.07
    FUNDAMENTAL_WEIGHT = 0.05
    NEWS_WEIGHT = 0.03

    HIGH_CONFLICT_PENALTY = 0.75
    MEDIUM_CONFLICT_PENALTY = 0.90
    LOW_CONFLICT_PENALTY = 0.95

    # A directional result below this threshold becomes HOLD.
    DIRECTIONAL_THRESHOLD = 0.55

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
        if value in {"bullish", "buy", "positive", "up", "long", "bull"}:
            return "bullish"
        if value in {"bearish", "sell", "negative", "down", "short", "bear"}:
            return "bearish"
        return "neutral"

    @staticmethod
    def _calculate_alignment(first: str, second: str) -> str:
        if first == "neutral" or second == "neutral":
            return "neutral"
        return "aligned" if first == second else "conflicting"

    @staticmethod
    def _calculate_market_alignment(
        ai_signal: str,
        ensemble_signal: str,
        market_direction: str,
    ) -> str:
        model_directions = [
            value for value in (ai_signal, ensemble_signal)
            if value != "neutral"
        ]
        if market_direction == "neutral" or not model_directions:
            return "neutral"

        if all(value == market_direction for value in model_directions):
            return "supportive"

        if any(value != market_direction for value in model_directions):
            return "conflicting"

        return "neutral"

    NON_DIRECTIONAL_DISAGREEMENT_PENALTY = 0.97

    @classmethod
    def _conflict_factor(
        cls,
        conflict: bool,
        level: str,
        directional_conflict: bool = False,
        disagreement_type: str = "none",
    ) -> float:
        """Return a stronger penalty only for genuinely opposing directions.

        ``conflict`` remains the legacy any-disagreement flag. When newer
        metadata is available, bullish/neutral disagreement gets a mild
        penalty; bullish/bearish opposition receives the configured severity
        penalty. This keeps old callers working.
        """
        disagreement_type = str(disagreement_type or "none").strip().lower()
        if directional_conflict or disagreement_type == "opposing_directions":
            level = str(level or "").strip().lower()
            if level == "high":
                return cls.HIGH_CONFLICT_PENALTY
            if level == "medium":
                return cls.MEDIUM_CONFLICT_PENALTY
            return cls.LOW_CONFLICT_PENALTY

        if conflict or disagreement_type in {
            "directional_neutral_disagreement",
            "model_disagreement",
        }:
            return cls.NON_DIRECTIONAL_DISAGREEMENT_PENALTY

        return 1.0

    @classmethod
    def _weighted_signal_evidence(
        cls,
        signal: str,
        confidence: float,
        weight: float,
    ) -> tuple[float, float, float]:
        """Convert one source into weighted bullish/bearish/neutral evidence."""
        confidence = cls._clamp(confidence)

        if signal == "bullish":
            return weight * confidence, 0.0, weight * (1.0 - confidence)
        if signal == "bearish":
            return 0.0, weight * confidence, weight * (1.0 - confidence)
        return 0.0, 0.0, weight

    @classmethod
    def _evidence_scores(
        cls,
        *,
        ai_signal: str,
        ai_confidence: float,
        ensemble_signal: str,
        ensemble_confidence: float,
        market_direction: str,
        market_confidence: float,
        technical_signal: str,
        technical_confidence: float,
        fundamental_signal: str,
        news_signal: str,
        fundamental_confidence: float = 0.0,
        news_confidence: float = 0.0,
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Calculate Layer 1 evidence using the Phase 14 source weights.

        The AI/ensemble bucket is returned separately so model-conflict
        penalties cannot accidentally reduce independent evidence.
        """
        bullish = bearish = neutral = 0.0
        model_bullish = model_bearish = 0.0

        for signal, confidence, weight in (
            (ai_signal, ai_confidence, cls.AI_WEIGHT),
            (ensemble_signal, ensemble_confidence, cls.ENSEMBLE_WEIGHT),
        ):
            b, s, n = cls._weighted_signal_evidence(signal, confidence, weight)
            model_bullish += b
            model_bearish += s
            bullish += b
            bearish += s
            neutral += n

        for signal, confidence, weight in (
            (market_direction, market_confidence, cls.MARKET_WEIGHT),
            (fundamental_signal, fundamental_confidence, cls.FUNDAMENTAL_WEIGHT),
            (news_signal, news_confidence, cls.NEWS_WEIGHT),
        ):
            b, s, n = cls._weighted_signal_evidence(signal, confidence, weight)
            bullish += b
            bearish += s
            neutral += n

        technical_bullish, technical_bearish, technical_neutral = cls._weighted_signal_evidence(
            technical_signal,
            technical_confidence,
            cls.TECHNICAL_WEIGHT,
        )
        bullish += technical_bullish
        bearish += technical_bearish
        neutral += technical_neutral

        return (
            bullish, bearish, neutral,
            model_bullish, model_bearish,
            technical_bullish, technical_bearish,
        )

    def analyze(self, data: AIDecisionInput) -> AIDecisionResult:
        ai_signal = self._normalize_signal(data.ai_signal)
        ensemble_signal = self._normalize_signal(data.ensemble_signal)
        market_direction = self._normalize_signal(data.market_direction)
        fundamental_signal = self._normalize_signal(data.fundamental_signal)
        news_signal = self._normalize_signal(data.news_signal)

        ai_confidence = self._clamp(data.ai_confidence)
        ensemble_confidence = self._clamp(data.ensemble_confidence)
        market_confidence = self._clamp(data.market_confidence)
        technical_confidence = self._clamp(data.technical_confidence)

        (
            bullish_score,
            bearish_score,
            neutral_score,
            model_bullish,
            model_bearish,
            technical_bullish,
            technical_bearish,
        ) = self._evidence_scores(
            ai_signal=ai_signal,
            ai_confidence=ai_confidence,
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            market_direction=market_direction,
            market_confidence=market_confidence,
            technical_signal=self._normalize_signal(data.technical_signal),
            technical_confidence=technical_confidence,
            fundamental_signal=fundamental_signal,
            news_signal=news_signal,
            fundamental_confidence=self._clamp(data.fundamental_confidence),
            news_confidence=self._clamp(data.news_confidence),
        )

        reasons: list[str] = []
        warnings: list[str] = []

        agreement_type = str(
            getattr(data, "agreement_type", "no_directional_consensus")
            or "no_directional_consensus"
        ).strip().lower()
        disagreement_type = str(
            getattr(data, "disagreement_type", "none") or "none"
        ).strip().lower()
        directional_conflict = bool(
            getattr(data, "directional_conflict", False)
            or disagreement_type == "opposing_directions"
        )
        legacy_conflict = bool(getattr(data, "conflict", False))
        # Never infer directional agreement from the legacy count when the
        # ensemble itself is neutral. For directional legacy callers, fall
        # back to agreement_count only if the additive field was not populated.
        if ensemble_signal == "neutral":
            directional_agreement_count = 0
        else:
            directional_agreement_count = max(
                0,
                int(
                    getattr(data, "directional_agreement_count", 0)
                    or data.agreement_count
                    or 0
                ),
            )
        neutral_agreement_count = max(
            0,
            int(getattr(data, "neutral_agreement_count", 0) or 0),
        )

        if agreement_type == "neutral_consensus" or (
            ensemble_signal == "neutral" and neutral_agreement_count > 0
        ):
            reasons.append(
                "Phase 9 model consensus is neutral; this is not bullish or bearish trade confirmation."
            )
        elif agreement_type == "directional_consensus":
            reasons.append(
                f"Phase 9 models provide directional consensus on {ensemble_signal}."
            )
        elif disagreement_type == "directional_neutral_disagreement":
            reasons.append(
                "Phase 9 confirmation is partial: a directional model and a neutral model disagree, "
                "but they do not directly oppose each other."
            )
        elif directional_conflict:
            warnings.append(
                "Phase 9 models contain opposing bullish and bearish directions."
            )
        elif legacy_conflict:
            warnings.append(
                "Phase 9 models disagree; the legacy conflict flag does not by itself prove opposing directions."
            )

        if ai_signal == "bullish":
            reasons.append(f"AI model signal is bullish with {ai_confidence:.2%} confidence.")
        elif ai_signal == "bearish":
            reasons.append(f"AI model signal is bearish with {ai_confidence:.2%} confidence.")
        else:
            reasons.append("AI model signal is neutral.")

        if ensemble_signal == "bullish":
            reasons.append(f"Ensemble signal is bullish with {ensemble_confidence:.2%} confidence.")
        elif ensemble_signal == "bearish":
            reasons.append(f"Ensemble signal is bearish with {ensemble_confidence:.2%} confidence.")
        else:
            reasons.append("Ensemble signal is neutral.")

        signal_alignment = self._calculate_alignment(ai_signal, ensemble_signal)
        if signal_alignment == "aligned":
            reasons.append("AI and ensemble signals are aligned.")
        elif signal_alignment == "conflicting":
            warnings.append("AI and ensemble signals are conflicting.")
        else:
            reasons.append("AI and ensemble signals provide limited independent direction.")

        # Explain market evidence. The market contribution has already been
        # added exactly once inside _evidence_scores() using MARKET_WEIGHT.
        if market_direction == "bullish":
            reasons.append(f"Market regime is bullish with {market_confidence:.2%} confidence.")
        elif market_direction == "bearish":
            reasons.append(f"Market regime is bearish with {market_confidence:.2%} confidence.")
        else:
            reasons.append("Market regime provides no directional bias.")

        market_alignment = self._calculate_market_alignment(
            ai_signal,
            ensemble_signal,
            market_direction,
        )
        if market_alignment == "supportive":
            reasons.append("Market regime supports the model direction.")
        elif market_alignment == "conflicting":
            warnings.append("Model direction conflicts with the current market regime.")
        else:
            reasons.append("Market regime provides limited confirmation.")

        # Explain fundamental evidence. The fundamental contribution has
        # already been added exactly once inside _evidence_scores() using
        # FUNDAMENTAL_WEIGHT.
        fundamental_confidence = self._clamp(data.fundamental_confidence)
        if fundamental_signal == "bullish":
            reasons.append(
                f"Fundamental analysis is supportive with {fundamental_confidence:.2%} source confidence."
            )
        elif fundamental_signal == "bearish":
            reasons.append(
                f"Fundamental analysis is negative with {fundamental_confidence:.2%} source confidence."
            )
        else:
            reasons.append("Fundamental analysis is neutral or has no directional signal.")

        # Explain news evidence. The news contribution has already been
        # added exactly once inside _evidence_scores() using NEWS_WEIGHT.
        news_confidence = self._clamp(data.news_confidence)
        if news_signal == "bullish":
            reasons.append(
                f"News and event analysis is supportive with {news_confidence:.2%} source confidence."
            )
        elif news_signal == "bearish":
            reasons.append(
                f"News and event analysis is negative with {news_confidence:.2%} source confidence."
            )
        else:
            reasons.append("News and event analysis is neutral or has no directional signal.")

        # Phase 14.1 is a real independent source and contributes its 7%
        # Layer 1 weight.
        technical_signal = self._normalize_signal(data.technical_signal)
        if technical_signal == "bullish":
            reasons.append(
                f"Technical intelligence is bullish with {technical_confidence:.2%} confidence."
            )
        elif technical_signal == "bearish":
            reasons.append(
                f"Technical intelligence is bearish with {technical_confidence:.2%} confidence."
            )
        else:
            reasons.append("Technical intelligence is neutral or unavailable.")

        # Penalize only the correlated AI/ensemble evidence bucket.
        # Independent market, technical, fundamental and news evidence is
        # never changed by Phase 9 model disagreement.
        apply_conflict_penalty = bool(
            legacy_conflict
            or directional_conflict
            or disagreement_type in {
                "directional_neutral_disagreement",
                "model_disagreement",
                "opposing_directions",
            }
        )
        conflict_factor = self._conflict_factor(
            apply_conflict_penalty,
            data.conflict_level,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,
        )

        if apply_conflict_penalty:
            bullish_score -= model_bullish
            bearish_score -= model_bearish
            bullish_score += model_bullish * conflict_factor
            bearish_score += model_bearish * conflict_factor
            neutral_score += (
                (model_bullish + model_bearish) * (1.0 - conflict_factor)
            )

            if directional_conflict:
                warnings.append(
                    f"Opposing bullish and bearish model directions are present "
                    f"(conflict level: {data.conflict_level})."
                )
            elif disagreement_type == "directional_neutral_disagreement":
                warnings.append(
                    "Model confirmation is partial (directional versus neutral); "
                    "a mild evidence penalty was applied, not an opposing-direction penalty."
                )
            else:
                warnings.append(
                    "Model outputs disagree; a mild evidence penalty was applied because "
                    "directly opposing directions were not established."
                )

        if technical_bullish > 0.0:
            reasons.append(
                f"Technical intelligence contributes {technical_bullish:.2%} bullish evidence."
            )
        elif technical_bearish > 0.0:
            reasons.append(
                f"Technical intelligence contributes {technical_bearish:.2%} bearish evidence."
            )

        # Normalize small floating-point drift before selection.
        total_score = bullish_score + bearish_score + neutral_score
        if total_score <= 0.0:
            bullish_score = bearish_score = 0.0
            neutral_score = 1.0
            score_map = {"BUY": 0.0, "SELL": 0.0, "HOLD": 1.0}
            decision = "HOLD"
            decision_reason_code = "no_valid_evidence"
        else:
            bullish_score /= total_score
            bearish_score /= total_score
            neutral_score /= total_score
            score_map = {
                "BUY": bullish_score,
                "SELL": bearish_score,
                "HOLD": neutral_score,
            }
            max_score = max(score_map.values())
            leaders = [
                label for label, score in score_map.items()
                if abs(score - max_score) <= 1e-12
            ]
            # Resolve exact ties conservatively instead of defaulting to BUY
            # because of dictionary insertion order.
            if len(leaders) > 1:
                decision = "HOLD"
                decision_reason_code = "evidence_tie_conservative_hold"
                warnings.append(
                    "Evidence scores are tied; the engine selected HOLD conservatively."
                )
            else:
                decision = leaders[0]
                decision_reason_code = "directional_evidence_dominates" if decision in {"BUY", "SELL"} else "neutral_evidence_dominates"

            if decision in {"BUY", "SELL"} and score_map[decision] < self.DIRECTIONAL_THRESHOLD:
                rejected_direction = "bullish" if decision == "BUY" else "bearish"
                rejected_score = score_map[decision]
                warnings.append(
                    f"The leading {rejected_direction} evidence score ({rejected_score:.2%}) "
                    f"is below the directional decision threshold ({self.DIRECTIONAL_THRESHOLD:.2%}); "
                    "the final decision is HOLD."
                )
                decision = "HOLD"
                decision_reason_code = "directional_threshold_not_met"

        # Confidence is the normalized evidence share assigned to the final
        # decision. It is a heuristic score, not a calibrated probability.
        confidence = (
            0.0
            if decision_reason_code == "no_valid_evidence"
            else score_map.get(decision, 0.0)
        )

        if bullish_score > bearish_score:
            leading_direction = "bullish"
        elif bearish_score > bullish_score:
            leading_direction = "bearish"
        else:
            leading_direction = "balanced"
        leading_direction_score = max(bullish_score, bearish_score)

        if decision == "BUY":
            decision_reason = (
                f"BUY selected because bullish evidence ({bullish_score:.2%}) "
                "is the largest evidence category and meets the directional threshold."
            )
            reasons.append(decision_reason)
        elif decision == "SELL":
            decision_reason = (
                f"SELL selected because bearish evidence ({bearish_score:.2%}) "
                "is the largest evidence category and meets the directional threshold."
            )
            reasons.append(decision_reason)
        elif decision_reason_code == "directional_threshold_not_met":
            decision_reason = (
                f"HOLD selected because the leading {leading_direction} evidence score "
                f"({leading_direction_score:.2%}) did not meet the required "
                f"{self.DIRECTIONAL_THRESHOLD:.2%} directional threshold. "
                "Review the evidence scores and validation history before taking directional action."
            )
            reasons.append(decision_reason)
        elif decision_reason_code == "evidence_tie_conservative_hold":
            decision_reason = (
                "HOLD selected conservatively because the highest evidence scores were tied."
            )
            reasons.append(decision_reason)
        elif decision_reason_code == "no_valid_evidence":
            decision_reason = "HOLD selected because no usable evidence score was available."
            reasons.append(decision_reason)
        else:
            decision_reason = (
                f"HOLD selected because neutral evidence ({neutral_score:.2%}) "
                f"is greater than bullish evidence ({bullish_score:.2%}) and "
                f"bearish evidence ({bearish_score:.2%})."
            )
            reasons.append(decision_reason)

        return AIDecisionResult(
            instrument_key=data.instrument_key,
            decision=decision,
            confidence=round(confidence, 4),
            bullish_score=round(bullish_score, 4),
            bearish_score=round(bearish_score, 4),
            neutral_score=round(neutral_score, 4),
            signal_alignment=signal_alignment,
            market_alignment=market_alignment,
            reasons=list(dict.fromkeys(reasons)),
            warnings=list(dict.fromkeys(warnings)),
            model_count=data.model_count,
            conflict=legacy_conflict,
            conflict_level=data.conflict_level,
            market_regime=data.market_regime,
            market_direction=market_direction,
            status="success",
            directional_agreement_count=directional_agreement_count,
            neutral_agreement_count=neutral_agreement_count,
            agreement_type=agreement_type,
            directional_conflict=directional_conflict,
            disagreement_type=disagreement_type,
            leading_direction=leading_direction,
            leading_direction_score=round(leading_direction_score, 4),
            directional_threshold=self.DIRECTIONAL_THRESHOLD,
            decision_reason_code=decision_reason_code,
            decision_reason=decision_reason,
            confidence_interpretation=(
                "Normalized heuristic evidence share for the selected decision; "
                "not a calibrated probability of a profitable trade."
            ),
        )
