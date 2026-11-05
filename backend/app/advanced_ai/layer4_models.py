from typing import List
from pydantic import BaseModel, Field


# ============================================================
# PHASE 14 — LAYER 4
# MARKET CONTEXT INTELLIGENCE
# ============================================================


class AIMarketContextInput(BaseModel):
    """
    Input data for Phase 14 Layer 4.

    Layer 4 evaluates how the broader market and sector context
    supports or conflicts with the stock-level AI decision.
    """

    # --------------------------------------------------------
    # Instrument
    # --------------------------------------------------------
    instrument_key: str

    # --------------------------------------------------------
    # Layer 1 — AI Decision
    # --------------------------------------------------------
    decision: str
    decision_confidence: float = Field(ge=0.0, le=1.0)

    # --------------------------------------------------------
    # Layer 2 — Reliability
    # --------------------------------------------------------
    reliability_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = "LOW"

    # --------------------------------------------------------
    # Market Regime
    # --------------------------------------------------------
    market_regime: str = "unknown"
    market_direction: str = "neutral"
    market_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    # --------------------------------------------------------
    # Sector Context
    # --------------------------------------------------------
    sector: str = "Unknown"
    sector_direction: str = "neutral"
    sector_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    # --------------------------------------------------------
    # Stock-Level Supporting Signals
    # --------------------------------------------------------
    technical_signal: str = "neutral"
    fundamental_signal: str = "neutral"
    news_signal: str = "neutral"


class AIMarketContextResult(BaseModel):
    """
    Output produced by Phase 14 Layer 4.
    """

    # --------------------------------------------------------
    # Instrument
    # --------------------------------------------------------
    instrument_key: str

    # --------------------------------------------------------
    # Original AI Decision
    # --------------------------------------------------------
    decision: str
    decision_confidence: float

    # --------------------------------------------------------
    # Context Score
    # --------------------------------------------------------
    market_context_score: float
    sector_context_score: float
    overall_context_score: float

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------
    market_alignment: str
    sector_alignment: str
    overall_alignment: str

    # --------------------------------------------------------
    # Context Classification
    # --------------------------------------------------------
    context_strength: str
    context_risk: str
    overall_context: str

    # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------
    supporting_factors: List[str] = Field(default_factory=list)
    conflicting_factors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # --------------------------------------------------------
    # Source Context
    # --------------------------------------------------------
    market_regime: str
    market_direction: str
    market_confidence: float

    sector: str
    sector_direction: str
    sector_confidence: float

    # --------------------------------------------------------
    # Layer 2 Context
    # --------------------------------------------------------
    reliability_score: float
    risk_level: str

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------
    status: str = "success"