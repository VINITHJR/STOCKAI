"""
Phase 15 unified feature builder.

Combines:
- 18 stock technical features
- 13 market-context features

Research-only. Does not modify production model interfaces.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from backend.app.ml.phase15.features.technical_features import (
    STOCK_TECHNICAL_FEATURES,
    build_stock_technical_features,
)

from backend.app.ml.phase15.features.market_context import (
    MARKET_CONTEXT_FEATURES,
    build_market_context_features,
)


class FeatureBuilderError(ValueError):
    """Raised when unified feature construction fails."""


PHASE15_FEATURE_COLUMNS = (
    STOCK_TECHNICAL_FEATURES + MARKET_CONTEXT_FEATURES
)


def build_phase15_features(
    stock_data: Dict[str, Any],
    nifty_data: Dict[str, Any],
    vix_data: Dict[str, Any],
) -> pd.DataFrame:
    """
    Construct the complete 31-feature dataset for one stock.

    All three inputs must be normalized by historical_loader.py.

    Returns:
        DataFrame containing:
            session_date
            18 stock technical features
            13 market-context features

    Rolling warm-up NaNs are intentionally preserved.
    The caller must handle them explicitly before model training.
    """

    stock_features = build_stock_technical_features(stock_data)

    market_features = build_market_context_features(
        stock_data=stock_data,
        nifty_data=nifty_data,
        vix_data=vix_data,
        require_exact_calendar=True,
    )

    stock_dates = stock_features["session_date"].tolist()
    market_dates = market_features["session_date"].tolist()

    if stock_dates != market_dates:
        raise FeatureBuilderError(
            "Stock and market feature session dates are not identical."
        )

    overlap = (
        set(STOCK_TECHNICAL_FEATURES)
        & set(MARKET_CONTEXT_FEATURES)
    )

    if overlap:
        raise FeatureBuilderError(
            f"Duplicate feature names detected: {sorted(overlap)}"
        )

    if len(stock_features) != len(market_features):
        raise FeatureBuilderError(
            "Stock and market feature row counts differ."
        )

    market_only = market_features.drop(
        columns=["session_date"]
    )

    combined = pd.concat(
        [
            stock_features.reset_index(drop=True),
            market_only.reset_index(drop=True),
        ],
        axis=1,
    )

    expected_columns = (
        ["session_date"] + PHASE15_FEATURE_COLUMNS
    )

    if list(combined.columns) != expected_columns:
        raise FeatureBuilderError(
            "Final feature order does not match the Phase 15 schema."
        )

    numeric = combined[PHASE15_FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    if np.isinf(numeric).any():
        raise FeatureBuilderError(
            "Infinite values detected in final feature matrix."
        )

    return combined
