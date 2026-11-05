from typing import List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# PHASE 14 — LAYER 7
# FINAL AI INTELLIGENCE AGGREGATOR
# ============================================================


class FinalAIIntelligenceInput(BaseModel):
    """
    Input for Phase 14 Layer 7.

    Layer 7 receives the validated outputs of Layers 1–6 and
    combines them into one final AI intelligence structure.

    Layer 7 does not create a new trading decision.
    """

    # --------------------------------------------------------
    # Instrument
    # --------------------------------------------------------
    instrument_key: str

    # --------------------------------------------------------
    # Layer 1 — AI Decision
    # --------------------------------------------------------
    decision: str
    decision_confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    bullish_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    bearish_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    neutral_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    signal_alignment: str = "neutral"
    market_alignment_layer1: str = "neutral"

    # --------------------------------------------------------
    # Layer 2 — Confidence & Reliability
    # --------------------------------------------------------
    reliability_score: float = Field(
        ge=0.0,
        le=1.0
    )

    confidence_grade: str = "E"
    confidence_band: str = "VERY_LOW"
    reliability_level: str = "VERY_LOW"
    recommendation_quality_layer2: str = "UNCERTAIN"
    decision_strength: str = "WEAK"

    model_reliability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    market_reliability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    signal_reliability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    supporting_reliability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    model_count: int = Field(
        default=0,
        ge=0
    )

    agreement_count: int = Field(
        default=0,
        ge=0
    )

    conflict: bool = False
    conflict_level: str = "none"
    directional_agreement_count: int = Field(default=0, ge=0)
    neutral_agreement_count: int = Field(default=0, ge=0)
    agreement_type: str = "no_directional_consensus"
    directional_conflict: bool = False
    disagreement_type: str = "none"

    # --------------------------------------------------------
    # Layer 3 — Explainable AI
    # --------------------------------------------------------
    decision_summary: str = ""
    confidence_explanation: str = ""
    reliability_explanation: str = ""
    final_explanation: str = ""

    bullish_factors: List[str] = Field(
        default_factory=list
    )

    bearish_factors: List[str] = Field(
        default_factory=list
    )

    neutral_factors: List[str] = Field(
        default_factory=list
    )

    explanation_supporting_factors: List[str] = Field(
        default_factory=list
    )

    conflicting_factors: List[str] = Field(
        default_factory=list
    )

    explanation_risk_factors: List[str] = Field(
        default_factory=list
    )

    strongest_factor: str = ""
    strongest_risk: str = ""
    explanation_quality: str = "GOOD"

    # --------------------------------------------------------
    # Layer 4 — Market Context
    # --------------------------------------------------------
    market_context_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )

    sector_context_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )

    overall_context_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )

    market_alignment: str = "NEUTRAL"
    sector_alignment: str = "NEUTRAL"
    overall_alignment: str = "NEUTRAL"

    context_strength: str = "WEAK"
    context_risk: str = "MEDIUM"
    overall_context: str = "NEUTRAL"

    market_regime: str = "unknown"
    market_direction: str = "neutral"
    market_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    sector: str = "Unknown"
    sector_direction: str = "neutral"
    sector_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    # --------------------------------------------------------
    # Layer 5 — Recommendation
    # --------------------------------------------------------
    recommendation: str
    recommendation_confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    recommendation_quality: str = "UNCERTAIN"
    recommendation_strength: str = "WEAK"

    recommendation_risk: str = "HIGH"

    recommendation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    recommendation_supporting_factors: List[str] = Field(
        default_factory=list
    )

    recommendation_conflicting_factors: List[str] = Field(
        default_factory=list
    )

    recommendation_warnings: List[str] = Field(
        default_factory=list
    )

    action_guidance: str = ""
    position_guidance: str = ""

    # --------------------------------------------------------
    # Layer 6 — Advanced AI Validation
    # --------------------------------------------------------
    validation_status: str = "INVALID"
    validation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    validation_quality: str = "POOR"

    decision_consistency: bool = False
    confidence_consistency: bool = False
    reliability_consistency: bool = False
    context_consistency: bool = False
    risk_consistency: bool = False
    conflict_consistency: bool = False

    passed_checks: List[str] = Field(
        default_factory=list
    )

    failed_checks: List[str] = Field(
        default_factory=list
    )

    validation_warnings: List[str] = Field(
        default_factory=list
    )

    validation_risk: str = "HIGH"

    recommendation_valid: bool = False
    recommendation_trust: str = "LOW"


class FinalAIIntelligenceResult(BaseModel):
    """
    Final unified Phase 14 AI intelligence result.

    This is the main structure that the future frontend/API
    can consume.
    """

    # --------------------------------------------------------
    # Identification
    # --------------------------------------------------------
    instrument_key: str

    # --------------------------------------------------------
    # Final Decision
    # --------------------------------------------------------
    decision: str
    decision_confidence: float

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------
    recommendation: str
    recommendation_confidence: float
    recommendation_quality: str
    recommendation_strength: str
    recommendation_risk: str

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------
    reliability_score: float
    confidence_grade: str
    confidence_band: str
    reliability_level: str

    # --------------------------------------------------------
    # Market Context
    # --------------------------------------------------------
    market_context: str
    market_context_score: float
    sector_context_score: float
    overall_context_score: float

    market_alignment: str
    sector_alignment: str
    overall_alignment: str

    context_strength: str
    context_risk: str

    market_regime: str
    market_direction: str
    sector: str
    sector_direction: str

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    validation_status: str
    validation_score: float
    validation_quality: str
    recommendation_valid: bool
    recommendation_trust: str
    validation_risk: str

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------
    decision_summary: str
    confidence_explanation: str
    reliability_explanation: str
    final_explanation: str

    strongest_factor: str
    strongest_risk: str

    bullish_factors: List[str] = Field(
        default_factory=list
    )

    bearish_factors: List[str] = Field(
        default_factory=list
    )

    neutral_factors: List[str] = Field(
        default_factory=list
    )

    supporting_factors: List[str] = Field(
        default_factory=list
    )

    conflicting_factors: List[str] = Field(
        default_factory=list
    )

    warnings: List[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Guidance
    # --------------------------------------------------------
    action_guidance: str
    position_guidance: str

    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------
    model_count: int
    agreement_count: int
    conflict: bool
    conflict_level: str
    directional_agreement_count: int = 0
    neutral_agreement_count: int = 0
    agreement_type: str = "no_directional_consensus"
    directional_conflict: bool = False
    disagreement_type: str = "none"

    # --------------------------------------------------------
    # Layer Summary
    # --------------------------------------------------------
    layers_completed: List[int] = Field(
        default_factory=list
    )

    layer_status: Dict[str, str] = Field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Overall Status
    # --------------------------------------------------------
    overall_status: str
    status: str = "success"

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )