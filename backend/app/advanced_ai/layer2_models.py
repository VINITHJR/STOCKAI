from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AIConfidenceInput:
    """
    Phase 14 - Layer 2 input.

    Evaluates the confidence and reliability of the
    Phase 14 Layer 1 decision.
    """

    instrument_key: str

    # ---------------------------------------------------------
    # Layer 1 decision
    # ---------------------------------------------------------

    decision: str = "HOLD"
    decision_confidence: float = 0.0

    # ---------------------------------------------------------
    # Layer 1 alignment
    # ---------------------------------------------------------

    signal_alignment: str = "neutral"
    market_alignment: str = "neutral"

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    model_count: int = 0
    agreement_count: int = 0
    directional_agreement_count: int | None = None

    conflict: bool = False
    conflict_level: str = "none"
    directional_conflict: bool = False
    disagreement_type: str = "none"
    agreement_type: str = "no_directional_consensus"
    neutral_agreement_count: int = 0

    # ---------------------------------------------------------
    # AI model
    # ---------------------------------------------------------

    ai_signal: str = "neutral"
    ai_confidence: float = 0.0

    # ---------------------------------------------------------
    # Ensemble
    # ---------------------------------------------------------

    ensemble_signal: str = "neutral"
    ensemble_confidence: float = 0.0
    ensemble_score: float = 0.0

    # ---------------------------------------------------------
    # Market regime
    # ---------------------------------------------------------

    market_regime: str = "unknown"
    market_direction: str = "neutral"
    market_confidence: float = 0.0

    # ---------------------------------------------------------
    # Supporting analysis
    # ---------------------------------------------------------

    technical_signal: str = "neutral"
    fundamental_signal: str = "neutral"
    news_signal: str = "neutral"

    # None means source confidence was not supplied; 0.0 is explicit zero.
    # Defaults keep older internal callers compatible.
    technical_confidence: float | None = None
    fundamental_confidence: float | None = None
    news_confidence: float | None = None


@dataclass
class AIConfidenceResult:
    """
    Phase 14 - Layer 2 output.

    Describes how reliable the Layer 1 decision is.
    """

    instrument_key: str

    # ---------------------------------------------------------
    # Decision
    # ---------------------------------------------------------

    decision: str
    decision_confidence: float

    # ---------------------------------------------------------
    # Reliability
    # ---------------------------------------------------------

    reliability_score: float
    confidence_grade: str
    confidence_band: str

    reliability_level: str
    recommendation_quality: str
    decision_strength: str
    risk_level: str

    # ---------------------------------------------------------
    # Reliability components
    # ---------------------------------------------------------

    model_reliability: float
    market_reliability: float
    signal_reliability: float
    supporting_reliability: float

    # ---------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------

    confidence_factors: List[str] = field(
        default_factory=list
    )

    risk_factors: List[str] = field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    model_count: int = 0
    agreement_count: int = 0
    directional_agreement_count: int = 0

    conflict: bool = False
    conflict_level: str = "none"
    directional_conflict: bool = False
    disagreement_type: str = "none"
    agreement_type: str = "no_directional_consensus"
    neutral_agreement_count: int = 0

    # ---------------------------------------------------------
    # Market information
    # ---------------------------------------------------------

    market_regime: str = "unknown"
    market_direction: str = "neutral"

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    status: str = "success"

    def to_dict(self) -> dict:
        return {
            "instrument_key": self.instrument_key,

            "decision": self.decision,
            "decision_confidence": round(
                self.decision_confidence,
                4,
            ),

            "reliability_score": round(
                self.reliability_score,
                4,
            ),

            "confidence_grade": self.confidence_grade,
            "confidence_band": self.confidence_band,

            "reliability_level": self.reliability_level,
            "recommendation_quality": (
                self.recommendation_quality
            ),
            "decision_strength": self.decision_strength,
            "risk_level": self.risk_level,

            "model_reliability": round(
                self.model_reliability,
                4,
            ),

            "market_reliability": round(
                self.market_reliability,
                4,
            ),

            "signal_reliability": round(
                self.signal_reliability,
                4,
            ),

            "supporting_reliability": round(
                self.supporting_reliability,
                4,
            ),

            "confidence_factors": self.confidence_factors,
            "risk_factors": self.risk_factors,


            "model_count": self.model_count,
            "agreement_count": self.agreement_count,
            "directional_agreement_count": self.directional_agreement_count,
            "neutral_agreement_count": self.neutral_agreement_count,
            "agreement_type": self.agreement_type,

            "conflict": self.conflict,
            "conflict_level": self.conflict_level,
            "directional_conflict": self.directional_conflict,
            "disagreement_type": self.disagreement_type,

            "market_regime": self.market_regime,
            "market_direction": self.market_direction,

            "status": self.status,
        }