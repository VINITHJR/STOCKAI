# ============================================================
# PHASE 14.3 — FUNDAMENTAL INTELLIGENCE ADAPTER
# ============================================================
#
# Purpose:
#   Convert the existing StockAI fundamentals-service response
#   into FundamentalAnalysisInput.
#
# Important:
#   This file DOES NOT call Upstox directly.
#
#   Existing service:
#       backend.app.services.fundamentals_service
#
#   Flow:
#       Existing fundamentals service
#               ↓
#       Adapter
#               ↓
#       FundamentalAnalysisInput
#               ↓
#       FundamentalIntelligence
#
# ============================================================

from __future__ import annotations

import math
from typing import Any

# ============================================================
# PROJECT IMPORTS
# ============================================================

try:
    # Normal package import when StockAI is started as a package.
    from backend.app.services.fundamentals_service import (
        get_fundamentals,
    )
except ModuleNotFoundError:
    # Direct execution from:
    # backend/app/advanced_ai/
    #
    # Add the project root so that:
    # backend.app.services...
    # can be resolved.
    import sys
    from pathlib import Path

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    from backend.app.services.fundamentals_service import (
        get_fundamentals,
    )

try:
    from .fundamental_intelligence_models import (
        FundamentalAnalysisInput,
    )
except ImportError:
    from fundamental_intelligence_models import (
        FundamentalAnalysisInput,
    )


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value: Any) -> float | None:
    """
    Convert a service value into a finite float.

    Returns None for:
      - None
      - empty strings
      - invalid numeric values
      - NaN
      - positive/negative infinity
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric):
        return None

    return numeric


def _clean_symbol(value: Any) -> str | None:
    """Return a clean symbol or None."""

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _clean_sector(value: Any) -> str | None:
    """Return a clean sector or None."""

    if value is None:
        return None

    text = str(value).strip()

    return text or None


# ============================================================
# SERVICE RESPONSE VALIDATION
# ============================================================

def _validate_service_response(
    response: Any,
) -> dict[str, Any]:
    """
    Validate the response returned by the existing
    fundamentals service.

    The adapter never assumes a successful response.
    """

    if not isinstance(response, dict):

        raise ValueError(
            "Fundamentals service returned a non-dictionary response."
        )

    status = str(
        response.get("status", "")
    ).strip().lower()

    if status != "success":

        message = response.get(
            "message",
            "Unknown fundamentals-service error.",
        )

        raise ValueError(
            f"Fundamentals service failed: {message}"
        )

    fundamentals = response.get(
        "fundamentals"
    )

    if not isinstance(fundamentals, dict):

        raise ValueError(
            "Fundamentals service response does not contain "
            "a valid 'fundamentals' object."
        )

    return response


# ============================================================
# SERVICE RESPONSE → MODEL INPUT
# ============================================================

def fundamentals_response_to_input(
    response: dict[str, Any],
) -> FundamentalAnalysisInput:
    """
    Convert an existing get_fundamentals() response into
    FundamentalAnalysisInput.

    Only metrics actually supplied by the existing service
    are mapped.

    No growth metrics are invented.

    No technical data is used.

    No AI model output is used.
    """

    response = _validate_service_response(
        response
    )

    fundamentals = response[
        "fundamentals"
    ]

    instrument_key = str(
        response.get(
            "instrument_key",
            "",
        )
    ).strip()

    if not instrument_key:

        raise ValueError(
            "Fundamentals response does not contain "
            "a valid instrument_key."
        )

    return FundamentalAnalysisInput(

        instrument_key=instrument_key,

        symbol=_clean_symbol(
            response.get("symbol")
        ),

        sector=_clean_sector(
            response.get("sector")
        ),

        # ----------------------------------------------------
        # Valuation
        # ----------------------------------------------------

        pe_ratio=_safe_float(
            fundamentals.get("pe_ratio")
        ),

        pb_ratio=_safe_float(
            fundamentals.get("pb_ratio")
        ),

        ev_ebitda=_safe_float(
            fundamentals.get("ev_ebitda")
        ),

        # ----------------------------------------------------
        # Profitability
        # ----------------------------------------------------

        roe=_safe_float(
            fundamentals.get("roe")
        ),

        roa=_safe_float(
            fundamentals.get("roa")
        ),

        roce=_safe_float(
            fundamentals.get("roce")
        ),

        # ----------------------------------------------------
        # Financial health
        # ----------------------------------------------------

        quick_ratio=_safe_float(
            fundamentals.get("quick_ratio")
        ),

        debt_to_equity=_safe_float(
            fundamentals.get("debt_to_equity")
        ),

        # ----------------------------------------------------
        # Shareholder return
        # ----------------------------------------------------

        dividend_yield=_safe_float(
            fundamentals.get("dividend_yield")
        ),

        # ----------------------------------------------------
        # Additional descriptive fields
        # ----------------------------------------------------
        #
        # These are accepted by FundamentalAnalysisInput
        # but are not currently used as scoring metrics.
        #
        eps=_safe_float(
            fundamentals.get("eps")
        ),

        eps_basic=_safe_float(
            fundamentals.get("eps_basic")
        ),

        eps_diluted=_safe_float(
            fundamentals.get("eps_diluted")
        ),

        book_value=_safe_float(
            fundamentals.get("book_value")
        ),

        market_cap=_safe_float(
            fundamentals.get("market_cap")
        ),

        total_assets=_safe_float(
            fundamentals.get("total_assets")
        ),

        current_assets=_safe_float(
            fundamentals.get("current_assets")
        ),

        current_liabilities=_safe_float(
            fundamentals.get("current_liabilities")
        ),

        non_current_liabilities=_safe_float(
            fundamentals.get(
                "non_current_liabilities"
            )
        ),

        equity_capital=_safe_float(
            fundamentals.get("equity_capital")
        ),
    )


# ============================================================
# SYMBOL → FUNDAMENTAL INPUT
# ============================================================

def get_fundamental_analysis_input(
    symbol: str,
) -> FundamentalAnalysisInput:
    """
    Fetch real fundamentals using the EXISTING StockAI
    fundamentals service and convert them into the
    Phase 14.3 input model.
    """

    clean_symbol = str(
        symbol or ""
    ).strip()

    if not clean_symbol:

        raise ValueError(
            "symbol is required"
        )

    response = get_fundamentals(
        clean_symbol
    )

    return fundamentals_response_to_input(
        response
    )


# ============================================================
# SYMBOL → FUNDAMENTAL INPUT + RAW RESPONSE
# ============================================================

def get_fundamental_source(
    symbol: str,
) -> tuple[
    FundamentalAnalysisInput,
    dict[str, Any],
]:
    """
    Fetch real fundamentals and return both:

        1. normalized FundamentalAnalysisInput
        2. original service response

    Useful for testing and debugging.
    """

    clean_symbol = str(
        symbol or ""
    ).strip()

    if not clean_symbol:

        raise ValueError(
            "symbol is required"
        )

    response = get_fundamentals(
        clean_symbol
    )

    analysis_input = fundamentals_response_to_input(
        response
    )

    return (
        analysis_input,
        response,
    )