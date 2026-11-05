# ============================================================
# PHASE 14 — ADVANCED AI
# ROUTES
# ============================================================

"""
Phase 14 is stock-driven.

The frontend only needs to send:

    instrument_key

This route builds the Phase 14 input from the existing Phase 9
intelligence/services for that SAME instrument.

Pipeline:

    Phase 9 Layer 2 models/ensemble
        ->
    Layer 1 AI signal reconstructed from same model outputs
        ->
    Fundamentals + News context
        ->
    Market regime
        ->
    Phase 14 Layers 1 -> 7
"""

# ============================================================
# IMPORTS
# ============================================================

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------
# Phase 9
# ------------------------------------------------------------

from backend.app.ai.layer1 import _ensemble as phase9_ensemble

from backend.app.ai.layer2 import run_multi_model_ensemble

from backend.app.ai.layer3 import (
    _fundamental_context,
    _news_context,
    _symbol_from_instrument,
)

from backend.app.ai.layer4 import (
    _extract_candles,
    _regime,
)

# ------------------------------------------------------------
# Existing data services
# ------------------------------------------------------------

from backend.app.services.fundamentals_service import (
    get_fundamentals,
)

from backend.app.services.news_event_service import (
    get_news_events,
)

from backend.app.services.upstox_service import (
    get_historical_candles,
)

# ------------------------------------------------------------
# Phase 14
# ------------------------------------------------------------

from .orchestrator import AdvancedAIOrchestrator

# IMPORTANT:
# Phase 14.3 independent Fundamental Intelligence
from .fundamental_intelligence import FundamentalIntelligence

from .fundamental_intelligence_adapter import (
    fundamentals_response_to_input,
)

# Phase 14.1 — Independent Technical Intelligence
from .technical_analysis_intelligence import (
    TechnicalAnalysisIntelligence,
)

# Phase 14.2 — Independent Stock Regime Intelligence
# The existing 14.2 implementation imports its model using the
# legacy top-level module name ``stock_regime_models``. Register
# the already-loaded package module under that name here so the
# existing 14.2 source code can be reused unchanged.
import sys as _sys
from . import stock_regime_models as _stock_regime_models

_sys.modules.setdefault(
    "stock_regime_models",
    _stock_regime_models,
)

from .stock_regime_intelligence import StockRegimeIntelligence
from .stock_regime_models import StockRegimeInput


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/advanced-ai",
    tags=["Phase 14 - Advanced AI"],
)


_orchestrator = AdvancedAIOrchestrator()


# ============================================================
# REQUEST MODEL
# ============================================================

class AdvancedAIRequest(BaseModel):
    """
    Phase 14 request contract.

    Analytical values are intentionally NOT accepted as
    authoritative frontend inputs.

    Phase 14 derives analytical values from the selected
    instrument so one stock cannot accidentally reuse
    another stock's analysis.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    instrument_key: str = Field(
        min_length=1,
        description="Upstox instrument key, e.g. NSE_EQ|INE009A01021",
    )


# ============================================================
# HELPERS
# ============================================================

def _parse_agreement(
    value: Any,
    model_count: int,
) -> int:
    """
    Convert Phase 9 agreement values such as '1/2'
    into an integer.
    """

    if isinstance(value, int):
        return max(0, min(value, model_count))

    text = str(value or "").strip()

    if "/" in text:
        try:
            numerator = int(
                text.split("/", 1)[0].strip()
            )

            return max(
                0,
                min(numerator, model_count),
            )

        except (TypeError, ValueError):
            pass

    try:
        return max(
            0,
            min(int(float(text)), model_count),
        )

    except (TypeError, ValueError):
        return 0


def _normalize_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"bullish", "buy", "long", "positive", "up"}:
        return "bullish"
    if text in {"bearish", "sell", "short", "negative", "down"}:
        return "bearish"
    return "neutral"


def _directional_agreement_count(
    ensemble_result: dict[str, Any],
    model_count: int,
) -> int:
    """Count only models confirming a bullish/bearish ensemble direction.

    Neutral consensus is valid descriptive information, but it is not
    directional confirmation and must not inflate Phase 14 reliability.
    """
    signal = _normalize_direction(ensemble_result.get("signal"))
    if model_count <= 0 or signal == "neutral":
        return 0

    models = ensemble_result.get("models")
    if isinstance(models, list):
        valid_signals = [
            _normalize_direction(item.get("signal"))
            for item in models
            if isinstance(item, dict)
        ]
        if valid_signals:
            return min(model_count, sum(value == signal for value in valid_signals))

    # Compatibility fallback for older Phase 9 responses without model details.
    return _parse_agreement(ensemble_result.get("agreement"), model_count)


def _agreement_metadata(
    ensemble_result: dict[str, Any],
    model_count: int,
) -> dict[str, Any]:
    """Normalize Phase 9 agreement data for a consistent Phase 14 contract.

    `conflict` remains the legacy any-disagreement flag. `directional_conflict`
    specifically means the model list contains both bullish and bearish signals.
    """
    raw_models = ensemble_result.get("models")
    model_signals = [
        _normalize_direction(item.get("signal"))
        for item in raw_models
        if isinstance(item, dict) and item.get("signal") is not None
    ] if isinstance(raw_models, list) else []

    ensemble_signal = _normalize_direction(ensemble_result.get("signal"))
    count = max(0, int(model_count or len(model_signals) or 0))
    legacy_conflict = bool(ensemble_result.get(
        "conflict", len(set(model_signals)) > 1 if model_signals else False
    ))

    if model_signals:
        directional_count = (
            sum(value == ensemble_signal for value in model_signals)
            if ensemble_signal in {"bullish", "bearish"} else 0
        )
        neutral_count = sum(value == "neutral" for value in model_signals)
        signals = set(model_signals)
        directional_conflict = {"bullish", "bearish"}.issubset(signals)
        if not legacy_conflict:
            disagreement_type = "none"
        elif directional_conflict:
            disagreement_type = "opposing_directions"
        elif "neutral" in signals and signals.intersection({"bullish", "bearish"}):
            disagreement_type = "directional_neutral_disagreement"
        else:
            disagreement_type = "model_disagreement"
    else:
        directional_count = _directional_agreement_count(ensemble_result, count)
        try:
            neutral_count = max(0, min(
                int(ensemble_result.get("neutral_agreement_count", 0) or 0), count
            ))
        except (TypeError, ValueError):
            neutral_count = 0
        directional_conflict = bool(ensemble_result.get("directional_conflict", False))
        disagreement_type = str(ensemble_result.get("disagreement_type") or (
            "opposing_directions" if directional_conflict else
            "directional_neutral_disagreement" if legacy_conflict else "none"
        ))

    agreement_type = ensemble_result.get("agreement_type")
    valid_types = {
        "directional_consensus", "partial_directional_agreement",
        "neutral_consensus", "no_directional_consensus",
    }
    if agreement_type not in valid_types:
        if ensemble_signal in {"bullish", "bearish"} and count > 0:
            if directional_count == count:
                agreement_type = "directional_consensus"
            elif directional_count > 0:
                agreement_type = "partial_directional_agreement"
            else:
                agreement_type = "no_directional_consensus"
        elif model_signals and neutral_count == len(model_signals):
            agreement_type = "neutral_consensus"
        else:
            agreement_type = "no_directional_consensus"

    return {
        "agreement_type": str(agreement_type),
        "directional_agreement_count": max(0, min(int(directional_count), count)),
        "neutral_agreement_count": max(0, min(int(neutral_count), count)),
        "directional_conflict": bool(directional_conflict),
        "disagreement_type": str(disagreement_type),
        "conflict": legacy_conflict,
        "conflict_level": str(ensemble_result.get("conflict_level") or "low"),
        "model_signals": model_signals,
    }


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


# ============================================================
# BUILD STOCK-SPECIFIC PHASE 14 PAYLOAD
# ============================================================

def _build_phase14_payload(
    instrument_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build a stock-specific Phase 14 payload from existing
    project sources.

    Returns:

        payload
        source_metadata
    """

    key = instrument_key.strip()

    if not key:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "instrument_key cannot be empty.",
            },
        )

    # ========================================================
    # Phase 9 Layer 2
    # ========================================================

    """
    Execute existing RF + LSTM inference once.

    We reuse those source outputs to reconstruct the
    Phase 9 Layer 1 signal without running the models
    a second time.
    """

    try:

        ensemble_result = run_multi_model_ensemble(
            instrument_key=key,
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Phase 9 model inference failed.",
                "source": "Phase 9 Layer 2",
                "error": str(exc),
            },
        ) from exc

    source_outputs = (
        ensemble_result.get("source_outputs")
        or {}
    )

    rf_result = source_outputs.get(
        "random_forest"
    )

    lstm_result = source_outputs.get(
        "lstm"
    )

    # ========================================================
    # Phase 9 Layer 1
    # ========================================================

    """
    Exact Phase 9 Layer 1 aggregation over the same
    RF + LSTM outputs.
    """

    try:

        ai_result = phase9_ensemble(
            rf_result,
            lstm_result,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Phase 9 Layer 1 signal "
                    "reconstruction failed."
                ),
                "source": "Phase 9 Layer 1",
                "error": str(exc),
            },
        ) from exc

    model_count = int(
        ensemble_result.get("model_count")
        or ai_result.get("model_count")
        or 0
    )

    # Keep the legacy field, but count only directional confirmation.
    agreement_meta = _agreement_metadata(ensemble_result, model_count)
    agreement_count = agreement_meta["directional_agreement_count"]

    # ========================================================
    # Symbol
    # ========================================================

    symbol = _symbol_from_instrument(key)

    if not symbol:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "NSE symbol could not be resolved "
                    "from instrument_key."
                ),
                "instrument_key": key,
            },
        )

    # ========================================================
    # Fundamentals + News
    # ========================================================

    fundamentals_result: dict[str, Any] | None

    try:

        fundamentals_result = get_fundamentals(
            symbol
        )

    except Exception as exc:

        fundamentals_result = None
        fundamentals_error = str(exc)

    else:

        fundamentals_error = None

    try:

        news_result = get_news_events(
            symbol,
            days=7,
            limit=20,
        )

    except Exception as exc:

        news_result = None
        news_error = str(exc)

    else:

        news_error = None

    # ========================================================
    # Existing Phase 9 contexts
    # ========================================================

    fundamental_context = _fundamental_context(
        fundamentals_result
    )

    news_context = _news_context(
        news_result
    )

    # ========================================================
    # Sector
    # ========================================================

    """
    Resolve sector BEFORE Phase 14.3.

    This is important because the Phase 14.3 fallback
    and successful result should both contain the actual
    stock sector whenever the fundamentals service provides it.
    """

    sector = "Unknown"

    if isinstance(
        fundamentals_result,
        dict,
    ):

        sector_value = fundamentals_result.get(
            "sector"
        )

        if sector_value:

            sector = str(
                sector_value
            ).strip()

    # ========================================================
    # Phase 14.3 — Independent Fundamental Intelligence
    # ========================================================

    """
    Reuse the SAME fundamentals service response already
    fetched above.

    This prevents a second Upstox fundamentals request
    and guarantees that:

        existing fundamental context

    and:

        Phase 14.3 fundamental intelligence

    use the same stock-specific source data.
    """

    fundamental_analysis_result: dict[str, Any]

    fundamental_analysis_error: str | None = None

    if (
        isinstance(
            fundamentals_result,
            dict,
        )
        and fundamentals_result.get("status")
        == "success"
    ):

        try:

            # ------------------------------------------------
            # Convert existing service response into the
            # independent Phase 14.3 input model.
            # ------------------------------------------------

            fundamental_input = (
                fundamentals_response_to_input(
                    fundamentals_result
                )
            )

            # ------------------------------------------------
            # Run independent deterministic fundamental
            # intelligence.
            # ------------------------------------------------

            fundamental_analysis = (
                FundamentalIntelligence().analyze(
                    fundamental_input
                )
            )

            # ------------------------------------------------
            # Convert Pydantic/model result into dict.
            # ------------------------------------------------

            if hasattr(
                fundamental_analysis,
                "model_dump",
            ):

                fundamental_analysis_result = (
                    fundamental_analysis.model_dump()
                )

            elif hasattr(
                fundamental_analysis,
                "dict",
            ):

                fundamental_analysis_result = (
                    fundamental_analysis.dict()
                )

            else:

                fundamental_analysis_result = dict(
                    fundamental_analysis.__dict__
                )

            # ------------------------------------------------
            # Ensure sector is not accidentally lost.
            # ------------------------------------------------

            if (
                not fundamental_analysis_result.get(
                    "sector"
                )
                or fundamental_analysis_result.get(
                    "sector"
                )
                == "Unknown"
            ):

                fundamental_analysis_result[
                    "sector"
                ] = sector

            # ------------------------------------------------
            # Ensure instrument key is correct.
            # ------------------------------------------------

            fundamental_analysis_result[
                "instrument_key"
            ] = key

            # ------------------------------------------------
            # Ensure symbol is correct.
            # ------------------------------------------------

            fundamental_analysis_result[
                "symbol"
            ] = symbol

        except Exception as exc:

            fundamental_analysis_result = {

                "instrument_key": key,

                "symbol": symbol,

                "sector": sector,

                "signal": "neutral",

                "confidence": 0.0,

                "fundamental_score": 0.0,

                "warnings": [
                    (
                        "Phase 14.3 fundamental "
                        f"analysis failed: {exc}"
                    )
                ],

                "method": (
                    "deterministic_fundamental_"
                    "intelligence_v1"
                ),
            }

            fundamental_analysis_error = str(exc)

    else:

        fundamental_analysis_result = {

            "instrument_key": key,

            "symbol": symbol,

            "sector": sector,

            "signal": "neutral",

            "confidence": 0.0,

            "fundamental_score": 0.0,

            "warnings": [
                "Fundamental source data unavailable."
            ],

            "method": (
                "deterministic_fundamental_"
                "intelligence_v1"
            ),
        }

        fundamental_analysis_error = (
            "Fundamental source data unavailable."
        )

    # ========================================================
    # Phase 9 Layer 4 — Market Regime
    # ========================================================

    try:

        to_date = date.today()

        from_date = (
            to_date - timedelta(days=180)
        )

        candles_payload = get_historical_candles(
            key,
            unit="days",
            interval="1",
            to_date=to_date.isoformat(),
            from_date=from_date.isoformat(),
        )

        candles = _extract_candles(
            candles_payload
        )

        if len(candles) < 50:

            raise ValueError(
                "At least 50 usable daily candles "
                f"are required; received {len(candles)}."
            )

        regime_result = _regime(
            candles
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Market regime calculation failed."
                ),
                "source": "Phase 9 Layer 4",
                "instrument_key": key,
                "error": str(exc),
            },
        ) from exc

    # ========================================================
    # Phase 14.1 — Independent Technical Intelligence
    # ========================================================

    technical_analysis_result: dict[str, Any]
    technical_analysis_error: str | None = None

    try:
        technical_analysis_result = (
            TechnicalAnalysisIntelligence.analyze(
                candles
            )
        )
    except Exception as exc:
        technical_analysis_result = {
            "signal": "neutral",
            "confidence": 0.0,
            "directional_score": 0.0,
            "vote_count": 0,
            "active_vote_count": 0,
            "agreement_count": 0,
            "indicators": {},
            "explanation": [],
            "method": "deterministic_ohlcv_multi_indicator_v1",
            "candle_count": len(candles),
        }
        technical_analysis_error = str(exc)

    # ========================================================
    # Phase 14.2 — Independent Stock Regime Intelligence
    # ========================================================

    stock_regime_result: dict[str, Any]
    stock_regime_error: str | None = None

    try:
        stock_regime_input = StockRegimeInput(
            instrument_key=key,
            candles=candles,
            min_candles=StockRegimeIntelligence.MIN_CANDLES,
        )
        stock_regime_analysis = StockRegimeIntelligence().analyze(
            stock_regime_input
        )

        if hasattr(stock_regime_analysis, "model_dump"):
            stock_regime_result = stock_regime_analysis.model_dump()
        elif hasattr(stock_regime_analysis, "dict"):
            stock_regime_result = stock_regime_analysis.dict()
        else:
            stock_regime_result = dict(stock_regime_analysis.__dict__)

        stock_regime_result["instrument_key"] = key

    except Exception as exc:
        stock_regime_result = {
            "instrument_key": key,
            "stock_regime": "unknown",
            "stock_direction": "neutral",
            "stock_regime_confidence": 0.0,
            "trend_strength": 0.0,
            "momentum_regime": "unknown",
            "volatility_regime": "unknown",
            "price_structure": "unknown",
            "current_price": 0.0,
            "ema_20": None,
            "ema_50": None,
            "ema_200": None,
            "momentum_20d_percent": 0.0,
            "atr_percent": 0.0,
            "adx_14": 0.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "candle_count": len(candles),
            "evidence": [],
            "method": "deterministic_ohlcv_stock_regime_v1",
            "status": "error",
        }
        stock_regime_error = str(exc)

    # ========================================================
    # Phase 14 normalized input
    # ========================================================

    payload: dict[str, Any] = {

        "instrument_key": key,

        # ----------------------------------------------------
        # Phase 9 Layer 1
        # ----------------------------------------------------

        "ai_signal": ai_result.get(
            "signal",
            "neutral",
        ),

        "ai_confidence": _safe_float(
            ai_result.get("confidence")
        ),

        # ----------------------------------------------------
        # Phase 9 Layer 2
        # ----------------------------------------------------

        "ensemble_signal": ensemble_result.get(
            "signal",
            "neutral",
        ),

        "ensemble_confidence": _safe_float(
            ensemble_result.get("confidence")
        ),

        "ensemble_score": _safe_float(
            ensemble_result.get(
                "raw_ensemble_score"
            )
        ),

        "model_count": model_count,

        # Preserve legacy keys; add explicit directional/neutral semantics.
        "agreement_count": agreement_count,
        "directional_agreement_count": agreement_meta["directional_agreement_count"],
        "neutral_agreement_count": agreement_meta["neutral_agreement_count"],
        "agreement_type": agreement_meta["agreement_type"],
        "directional_conflict": agreement_meta["directional_conflict"],
        "disagreement_type": agreement_meta["disagreement_type"],
        "conflict": agreement_meta["conflict"],
        "conflict_level": agreement_meta["conflict_level"],

        # ----------------------------------------------------
        # Phase 9 Layer 4
        # ----------------------------------------------------

        "market_regime": str(
            regime_result.get(
                "regime"
            )
            or "unknown"
        ),

        "market_direction": str(
            regime_result.get(
                "direction"
            )
            or "neutral"
        ),

        "market_confidence": _safe_float(
            regime_result.get(
                "confidence"
            )
        ),

        # ----------------------------------------------------
        # Sector context
        # ----------------------------------------------------

        "sector": sector,

        "sector_direction": "neutral",

        "sector_confidence": 0.0,

        # ----------------------------------------------------
        # Phase 14.1 — Independent technical intelligence
        # ----------------------------------------------------

        "technical_signal": technical_analysis_result.get(
            "signal",
            "neutral",
        ),

        "technical_confidence": _safe_float(
            technical_analysis_result.get(
                "confidence"
            )
        ),

        # Phase 14.2 is intentionally carried as independent
        # stock-regime context. It is NOT substituted for the
        # broader Phase 9 market regime above and does not
        # invent a new Layer 1 weight.
        "stock_regime": stock_regime_result.get(
            "stock_regime",
            "unknown",
        ),

        "stock_direction": stock_regime_result.get(
            "stock_direction",
            "neutral",
        ),

        "stock_regime_confidence": _safe_float(
            stock_regime_result.get(
                "stock_regime_confidence"
            )
        ),

        # ----------------------------------------------------
        # Phase 14.3 — Independent fundamental intelligence
        # ----------------------------------------------------

        "fundamental_signal": (
            fundamental_analysis_result.get(
                "signal",
                "neutral",
            )
        ),

        "fundamental_confidence": _safe_float(
            fundamental_analysis_result.get(
                "confidence"
            )
        ),

        # ----------------------------------------------------
        # News
        # ----------------------------------------------------

        "news_signal": news_context.get(
            "signal",
            "neutral",
        ),

        "news_confidence": _safe_float(
            news_context.get("confidence")
        ),
    }

    # ========================================================
    # Source metadata
    # ========================================================

    source_metadata = {

        "symbol": symbol,

        # ----------------------------------------------------
        # Phase 9 Layer 1
        # ----------------------------------------------------

        "phase9_layer1_signal": ai_result,

        # ----------------------------------------------------
        # Phase 9 Layer 2
        # ----------------------------------------------------

        "phase9_layer2": {

            "signal": ensemble_result.get(
                "signal"
            ),

            "confidence": ensemble_result.get(
                "confidence"
            ),

            "raw_ensemble_score": (
                ensemble_result.get(
                    "raw_ensemble_score"
                )
            ),

            "model_count": ensemble_result.get(
                "model_count"
            ),

            "agreement": ensemble_result.get("agreement"),
            "agreement_type": agreement_meta["agreement_type"],
            "directional_agreement_count": agreement_meta["directional_agreement_count"],
            "neutral_agreement_count": agreement_meta["neutral_agreement_count"],
            "directional_conflict": agreement_meta["directional_conflict"],
            "disagreement_type": agreement_meta["disagreement_type"],
            "model_signals": agreement_meta["model_signals"],
            "conflict": agreement_meta["conflict"],
            "conflict_level": agreement_meta["conflict_level"],
        },

        # ----------------------------------------------------
        # Phase 9 Layer 4
        # ----------------------------------------------------

        "phase9_layer4": {

            "regime": regime_result.get(
                "regime"
            ),

            "direction": regime_result.get(
                "direction"
            ),

            "confidence": regime_result.get(
                "confidence"
            ),

            "candle_count": len(
                candles
            ),
        },

        # ----------------------------------------------------
        # Phase 14.1 / 14.2 independent analysis
        # ----------------------------------------------------

        "technical_intelligence": technical_analysis_result,

        "stock_regime_intelligence": stock_regime_result,

        # Explicit Phase 14 integration contract.
        # 14.1 contributes to Layer 1; 14.2 remains contextual only.
        "phase14_1": {
            "status": "error" if technical_analysis_error else "success",
            "role": "layer_1_directional_evidence",
            "decision_weight": 0.07,
            "signal": technical_analysis_result.get("signal", "neutral"),
            "confidence": _safe_float(technical_analysis_result.get("confidence")),
            "candle_count": technical_analysis_result.get("candle_count", len(candles)),
            "method": technical_analysis_result.get("method"),
            "error": technical_analysis_error,
        },
        "phase14_2": {
            "status": "error" if stock_regime_error else "success",
            "role": "stock_context_diagnostic",
            "decision_weight": 0.0,
            "stock_regime": stock_regime_result.get("stock_regime", "unknown"),
            "stock_direction": stock_regime_result.get("stock_direction", "neutral"),
            "confidence": _safe_float(stock_regime_result.get("stock_regime_confidence")),
            "candle_count": stock_regime_result.get("candle_count", len(candles)),
            "method": stock_regime_result.get("method"),
            "error": stock_regime_error,
        },

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        "context": {

            # Phase 14.1 independent technical intelligence
            "technical_intelligence": technical_analysis_result,

            # Explicit Phase 14.1 contract
            "phase14_1": {
                "signal": technical_analysis_result.get("signal", "neutral"),
                "confidence": _safe_float(technical_analysis_result.get("confidence")),
                "decision_weight": 0.07,
                "status": "error" if technical_analysis_error else "success",
            },

            # Phase 14.2 independent selected-stock regime
            "stock_regime_intelligence": stock_regime_result,

            # Explicit Phase 14.2 contract; contextual only.
            "phase14_2": {
                "stock_regime": stock_regime_result.get("stock_regime", "unknown"),
                "stock_direction": stock_regime_result.get("stock_direction", "neutral"),
                "confidence": _safe_float(stock_regime_result.get("stock_regime_confidence")),
                "decision_weight": 0.0,
                "status": "error" if stock_regime_error else "success",
            },

            # Phase 14.3 is the authoritative fundamental source.
            # Keep the nested context aligned with the same signal used
            # by Layer 1 so diagnostics cannot contradict the pipeline.
            "fundamental_signal": (
                fundamental_analysis_result.get(
                    "signal",
                    "neutral",
                )
            ),

            "fundamental_context_confidence": _safe_float(
                fundamental_analysis_result.get(
                    "confidence"
                )
            ),

            # Phase 14.3 independent fundamental intelligence
            "fundamental_intelligence": (
                fundamental_analysis_result
            ),

            "fundamental_intelligence_confidence": (
                fundamental_analysis_result.get(
                    "confidence"
                )
            ),

            # News
            "news_signal": news_context.get(
                "signal"
            ),

            "news_confidence": news_context.get(
                "confidence"
            ),

            # Sector
            "sector": sector,
        },

        # ----------------------------------------------------
        # Warnings
        # ----------------------------------------------------

        "warnings": [

            warning

            for warning in [

                (
                    "Fundamentals unavailable: "
                    f"{fundamentals_error}"
                    if fundamentals_error
                    else None
                ),

                (
                    "Phase 14.3 fundamental "
                    "intelligence unavailable: "
                    f"{fundamental_analysis_error}"
                    if fundamental_analysis_error
                    else None
                ),

                (
                    "News unavailable: "
                    f"{news_error}"
                    if news_error
                    else None
                ),

                (
                    "Phase 14.1 technical intelligence unavailable: "
                    f"{technical_analysis_error}"
                    if technical_analysis_error
                    else None
                ),

                (
                    "Phase 14.2 stock regime intelligence unavailable: "
                    f"{stock_regime_error}"
                    if stock_regime_error
                    else None
                ),

                (
                    "No sector-direction "
                    "intelligence source exists in "
                    "the current backend; sector "
                    "direction is neutral."
                ),

            ]

            if warning
        ],
    }

    return (
        payload,
        source_metadata,
    )


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def advanced_ai_health():

    return {

        "phase": 14,

        "module": "advanced_ai",

        "status": "success",

        "message": (
            "Phase 14 Advanced AI API is running."
        ),
    }


# ============================================================
# RUN COMPLETE PHASE 14 PIPELINE
# ============================================================

@router.post("/run")
def run_advanced_ai(
    request: AdvancedAIRequest,
):
    """
    Execute Phase 14 for the selected instrument.

    The client sends only instrument_key.

    All analytical inputs are derived server-side
    from the existing StockAI Phase 9/data services.
    """

    key = request.instrument_key.strip()

    try:

        payload, source_metadata = (
            _build_phase14_payload(
                key
            )
        )

        result = _orchestrator.run(
            payload
        )

        # ----------------------------------------------------
        # Preserve existing response envelope while adding
        # source traceability.
        # ----------------------------------------------------

        result[
            "phase14_input_source"
        ] = source_metadata

        return {

            "status": "success",

            "phase": 14,

            "instrument_key": key,

            "result": result,
        }

    except HTTPException:
        raise

    except KeyError as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Phase 14 internal input "
                    "mapping failed."
                ),
                "field": str(exc).strip("'"),
            },
        ) from exc

    except (TypeError, ValueError) as exc:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Invalid Phase 14 pipeline value."
                ),
                "error": str(exc),
            },
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Phase 14 Advanced AI "
                    "execution failed."
                ),
                "error": str(exc),
            },
        ) from exc