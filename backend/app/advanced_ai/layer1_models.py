from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AIDecisionInput:
    """Validated, stock-specific input for Phase 14 Layer 1.

    Newly added agreement fields are optional/defaulted so existing callers
    that construct this dataclass with the legacy fields remain compatible.
    Confidence inputs are evidence scores, not calibrated probabilities.
    """

    instrument_key: str

    # Phase 9 AI signal
    ai_signal: str = "neutral"
    ai_confidence: float = 0.0

    # Phase 9 Layer 2 ensemble
    ensemble_signal: str = "neutral"
    ensemble_confidence: float = 0.0
    ensemble_score: float = 0.0

    # Legacy-compatible model agreement fields
    model_count: int = 0
    agreement_count: int = 0
    conflict: bool = False
    conflict_level: str = "none"

    # Additive agreement metadata from Phase 9.
    directional_agreement_count: int = 0
    neutral_agreement_count: int = 0
    agreement_type: str = "no_directional_consensus"
    directional_conflict: bool = False
    disagreement_type: str = "none"

    # Market regime
    market_regime: str = "unknown"
    market_direction: str = "neutral"
    market_confidence: float = 0.0

    # Supporting information
    technical_signal: str = "neutral"
    technical_confidence: float = 0.0
    fundamental_signal: str = "neutral"
    fundamental_confidence: float = 0.0
    news_signal: str = "neutral"
    news_confidence: float = 0.0


@dataclass
class AIDecisionResult:
    instrument_key: str

    decision: str
    confidence: float

    bullish_score: float
    bearish_score: float
    neutral_score: float

    signal_alignment: str
    market_alignment: str

    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    model_count: int = 0
    conflict: bool = False
    conflict_level: str = "none"

    market_regime: str = "unknown"
    market_direction: str = "neutral"
    status: str = "success"

    # Phase 9 agreement metadata carried into Phase 14 Layer 1.
    directional_agreement_count: int = 0
    neutral_agreement_count: int = 0
    agreement_type: str = "no_directional_consensus"
    directional_conflict: bool = False
    disagreement_type: str = "none"

    # Diagnostic fields that explain the selected decision.
    leading_direction: str = "balanced"
    leading_direction_score: float = 0.0
    directional_threshold: float = 0.55
    decision_reason_code: str = "unknown"
    decision_reason: str = ""
    confidence_interpretation: str = (
        "Normalized heuristic evidence share; not a calibrated probability."
    )

    def to_dict(self) -> dict:
        return {
            "instrument_key": self.instrument_key,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "bullish_score": round(self.bullish_score, 4),
            "bearish_score": round(self.bearish_score, 4),
            "neutral_score": round(self.neutral_score, 4),
            "signal_alignment": self.signal_alignment,
            "market_alignment": self.market_alignment,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "model_count": self.model_count,
            "conflict": self.conflict,
            "conflict_level": self.conflict_level,
            "market_regime": self.market_regime,
            "market_direction": self.market_direction,
            "status": self.status,
            "directional_agreement_count": self.directional_agreement_count,
            "neutral_agreement_count": self.neutral_agreement_count,
            "agreement_type": self.agreement_type,
            "directional_conflict": self.directional_conflict,
            "disagreement_type": self.disagreement_type,
            "leading_direction": self.leading_direction,
            "leading_direction_score": round(self.leading_direction_score, 4),
            "directional_threshold": self.directional_threshold,
            "decision_reason_code": self.decision_reason_code,
            "decision_reason": self.decision_reason,
            "confidence_interpretation": self.confidence_interpretation,
        }
