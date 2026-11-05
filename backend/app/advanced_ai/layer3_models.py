from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AIExplanationInput:
    """
    Phase 14 - Layer 3 input.

    Contains the outputs and supporting information required
    to explain the Layer 1 AI decision and Layer 2 reliability.
    """

    instrument_key: str

    # ---------------------------------------------------------
    # Layer 1 decision
    # ---------------------------------------------------------

    decision: str = "HOLD"
    decision_confidence: float = 0.0

    bullish_score: float = 0.0
    bearish_score: float = 0.0
    neutral_score: float = 0.0

    signal_alignment: str = "neutral"
    market_alignment: str = "neutral"

    # ---------------------------------------------------------
    # Layer 2 reliability
    # ---------------------------------------------------------

    reliability_score: float = 0.0
    confidence_grade: str = "E"
    confidence_band: str = "VERY_LOW"

    reliability_level: str = "VERY_LOW"
    recommendation_quality: str = "UNCERTAIN_HOLD"

    decision_strength: str = "WEAK_HOLD"
    risk_level: str = "HIGH"

    model_reliability: float = 0.0
    market_reliability: float = 0.0
    signal_reliability: float = 0.0
    supporting_reliability: float = 0.0

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    model_count: int = 0
    agreement_count: int = 0

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
    technical_confidence: float = 0.0

    fundamental_signal: str = "neutral"
    fundamental_confidence: float = 0.0

    news_signal: str = "neutral"
    news_confidence: float = 0.0


@dataclass
class AIExplanationResult:
    """
    Phase 14 - Layer 3 output.

    Provides a human-readable explanation of the AI decision.
    """

    instrument_key: str

    # ---------------------------------------------------------
    # Decision
    # ---------------------------------------------------------

    decision: str
    decision_confidence: float

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    decision_summary: str
    confidence_explanation: str
    reliability_explanation: str

    final_explanation: str

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    bullish_factors: List[str] = field(
        default_factory=list
    )

    bearish_factors: List[str] = field(
        default_factory=list
    )

    neutral_factors: List[str] = field(
        default_factory=list
    )

    supporting_factors: List[str] = field(
        default_factory=list
    )

    conflicting_factors: List[str] = field(
        default_factory=list
    )

    risk_factors: List[str] = field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Key metrics
    # ---------------------------------------------------------

    strongest_factor: str = ""
    strongest_risk: str = ""

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    explanation_quality: str = "STANDARD"
    directional_conflict: bool = False
    disagreement_type: str = "none"
    agreement_type: str = "no_directional_consensus"
    neutral_agreement_count: int = 0

    status: str = "success"

    def to_dict(self) -> dict:
        return {
            "instrument_key": self.instrument_key,

            "decision": self.decision,
            "decision_confidence": round(
                self.decision_confidence,
                4,
            ),

            "decision_summary": self.decision_summary,

            "confidence_explanation": (
                self.confidence_explanation
            ),

            "reliability_explanation": (
                self.reliability_explanation
            ),

            "final_explanation": self.final_explanation,

            "bullish_factors": self.bullish_factors,
            "bearish_factors": self.bearish_factors,
            "neutral_factors": self.neutral_factors,

            "supporting_factors": (
                self.supporting_factors
            ),

            "conflicting_factors": (
                self.conflicting_factors
            ),

            "risk_factors": self.risk_factors,

            "strongest_factor": self.strongest_factor,
            "strongest_risk": self.strongest_risk,

            "explanation_quality": self.explanation_quality,
            "directional_conflict": self.directional_conflict,
            "disagreement_type": self.disagreement_type,
            "agreement_type": self.agreement_type,
            "neutral_agreement_count": self.neutral_agreement_count,

            "status": self.status,
        }