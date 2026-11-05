# ============================================================
# PHASE 14.3 — FUNDAMENTAL INTELLIGENCE
# MODELS
# ============================================================

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FundamentalAnalysisInput(BaseModel):
    """
    Input for Phase 14.3 Fundamental Intelligence.

    Values correspond to fields exposed by the existing
    StockAI fundamentals service.

    Missing values remain None and are NOT interpreted as
    bearish evidence.
    """

    instrument_key: str

    symbol: Optional[str] = None
    sector: Optional[str] = None

    # ========================================================
    # Valuation
    # ========================================================

    pe_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    pb_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    ev_ebitda: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    # ========================================================
    # Profitability
    # ========================================================

    roe: Optional[float] = None
    roa: Optional[float] = None
    roce: Optional[float] = None

    # ========================================================
    # Financial health
    # ========================================================

    quick_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    debt_to_equity: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    # ========================================================
    # Shareholder return
    # ========================================================

    dividend_yield: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    # ========================================================
    # Descriptive values
    # ========================================================

    eps: Optional[float] = None
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None

    book_value: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    market_cap: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    total_assets: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    current_assets: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    current_liabilities: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    non_current_liabilities: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    equity_capital: Optional[float] = Field(
        default=None,
        ge=0.0,
    )


class FundamentalAnalysisResult(BaseModel):
    """
    Output produced by Phase 14.3.

    This component provides fundamental evidence only.
    It does NOT produce BUY/HOLD/SELL.
    """

    instrument_key: str

    symbol: Optional[str] = None
    sector: Optional[str] = None

    # ========================================================
    # Final fundamental signal
    # ========================================================

    signal: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    fundamental_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    # ========================================================
    # Component scores
    # ========================================================

    valuation_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    profitability_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    financial_health_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    shareholder_return_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    # ========================================================
    # Assessments
    # ========================================================

    valuation_assessment: str
    profitability_assessment: str
    financial_health_assessment: str
    shareholder_return_assessment: str

    # ========================================================
    # Data quality
    # ========================================================

    available_metric_count: int = Field(
        ge=0,
    )

    total_metric_count: int = Field(
        ge=1,
    )

    data_completeness: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ========================================================
    # Explainability
    # ========================================================

    bullish_evidence: List[str] = Field(
        default_factory=list,
    )

    bearish_evidence: List[str] = Field(
        default_factory=list,
    )

    neutral_evidence: List[str] = Field(
        default_factory=list,
    )

    evidence: List[str] = Field(
        default_factory=list,
    )

    warnings: List[str] = Field(
        default_factory=list,
    )

    # ========================================================
    # Metadata
    # ========================================================

    method: str = (
        "deterministic_fundamental_intelligence_v1"
    )