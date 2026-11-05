"""
Phase 15 market-context feature engineering.

Combines individual-stock data with NIFTY 50 and India VIX information.

All features at session t use current or past session data only.
No forward filling or future-data access is performed.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class MarketContextError(ValueError):
    """Raised when stock and index data cannot be safely aligned."""


MARKET_CONTEXT_FEATURES = [
    "nifty_return_1d",
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_realized_volatility_20d",
    "nifty_close_vs_sma20",
    "nifty_close_vs_sma50",
    "india_vix_close",
    "india_vix_return_1d",
    "india_vix_return_5d",
    "india_vix_close_vs_sma20",
    "relative_return_1d",
    "relative_return_5d",
    "relative_return_20d",
]


def _prepare_close_frame(
    dataset: Dict[str, Any],
    name: str,
) -> pd.DataFrame:
    """
    Extract and validate session dates and closing prices.

    Input must be the normalized output of historical_loader.py.
    The function rejects duplicate or unordered session dates.
    """

    if not isinstance(dataset, dict):
        raise MarketContextError(f"{name}: dataset must be a dictionary.")

    rows = dataset.get("rows")

    if not isinstance(rows, list) or not rows:
        raise MarketContextError(f"{name}: no normalized rows supplied.")

    frame = pd.DataFrame(rows)

    required = {"session_date", "close"}
    missing = required - set(frame.columns)

    if missing:
        raise MarketContextError(
            f"{name}: missing required fields: {sorted(missing)}"
        )

    try:
        dates = pd.to_datetime(
            frame["session_date"],
            format="%Y-%m-%d",
            errors="raise",
        )

        close = pd.to_numeric(frame["close"], errors="coerce")

    except (TypeError, ValueError) as exc:
        raise MarketContextError(
            f"{name}: could not parse session dates or closing prices."
        ) from exc

    if dates.isna().any():
        raise MarketContextError(f"{name}: missing session dates.")

    if dates.duplicated().any():
        raise MarketContextError(f"{name}: duplicate trading sessions.")

    if not dates.is_monotonic_increasing:
        raise MarketContextError(
            f"{name}: sessions must be sorted oldest to newest."
        )

    close_values = close.to_numpy(dtype=np.float64)

    if not np.isfinite(close_values).all():
        raise MarketContextError(
            f"{name}: closing prices contain NaN or infinity."
        )

    if (close_values <= 0).any():
        raise MarketContextError(
            f"{name}: closing prices must be positive."
        )

    return pd.DataFrame({
        "session_date": dates.dt.strftime("%Y-%m-%d"),
        "close": close_values,
    })


def _percentage_return(
    close: pd.Series,
    periods: int,
) -> pd.Series:
    """Calculate close-to-close return without filling missing observations."""
    return close.pct_change(
        periods=periods,
        fill_method=None,
    )


def build_market_context_features(
    stock_data: Dict[str, Any],
    nifty_data: Dict[str, Any],
    vix_data: Dict[str, Any],
    require_exact_calendar: bool = True,
) -> pd.DataFrame:
    """
    Build market-context features for one stock.

    Required inputs:
        stock_data: Validated stock dataset.
        nifty_data: Validated NIFTY 50 dataset.
        vix_data: Validated India VIX dataset.

    Output:
        One row per stock trading session.
        Contains session_date and MARKET_CONTEXT_FEATURES.

    Feature definitions:

        NIFTY returns:
            1, 5 and 20 session close-to-close returns.

        NIFTY volatility:
            Rolling 20-session standard deviation of daily returns.

        NIFTY trend:
            NIFTY close relative to its 20 and 50 session averages.

        India VIX:
            Closing level, 1 and 5 session returns,
            and closing level relative to its 20 session average.

        Relative performance:
            Stock cumulative return relative to NIFTY cumulative return
            over 1, 5 and 20 sessions.

    Important:
        The India VIX and NIFTY volume fields are deliberately excluded.
        Index volume is not ordinary stock trading volume.

        Rolling warm-up periods remain NaN. They are not replaced with zero.
    """

    stock = _prepare_close_frame(stock_data, "STOCK")
    nifty = _prepare_close_frame(nifty_data, "NIFTY 50")
    vix = _prepare_close_frame(vix_data, "INDIA VIX")

    stock_dates = stock["session_date"]
    nifty_dates = nifty["session_date"]
    vix_dates = vix["session_date"]

    if require_exact_calendar:

        if not stock_dates.equals(nifty_dates):
            missing = sorted(set(nifty_dates) - set(stock_dates))
            extra = sorted(set(stock_dates) - set(nifty_dates))

            raise MarketContextError(
                "Stock and NIFTY calendars do not match exactly. "
                f"Missing stock sessions: {missing[:10]}; "
                f"extra stock sessions: {extra[:10]}."
            )

        if not stock_dates.equals(vix_dates):
            missing = sorted(set(vix_dates) - set(stock_dates))
            extra = sorted(set(stock_dates) - set(vix_dates))

            raise MarketContextError(
                "Stock and India VIX calendars do not match exactly. "
                f"Missing stock sessions: {missing[:10]}; "
                f"extra stock sessions: {extra[:10]}."
            )

    else:
        raise NotImplementedError(
            "Non-exact calendar alignment is not supported yet. "
            "Do not forward-fill index data."
        )

    stock_close = stock["close"]
    nifty_close = nifty["close"]
    vix_close = vix["close"]

    # --------------------------------------------------
    # NIFTY 50 context
    # --------------------------------------------------

    nifty_return_1d = _percentage_return(nifty_close, 1)
    nifty_return_5d = _percentage_return(nifty_close, 5)
    nifty_return_20d = _percentage_return(nifty_close, 20)

    nifty_volatility_20d = (
        nifty_return_1d
        .rolling(window=20, min_periods=20)
        .std(ddof=1)
    )

    nifty_sma20 = nifty_close.rolling(
        window=20, min_periods=20
    ).mean()

    nifty_sma50 = nifty_close.rolling(
        window=50, min_periods=50
    ).mean()

    nifty_close_vs_sma20 = nifty_close / nifty_sma20 - 1.0
    nifty_close_vs_sma50 = nifty_close / nifty_sma50 - 1.0

    # --------------------------------------------------
    # India VIX context
    # --------------------------------------------------

    vix_return_1d = _percentage_return(vix_close, 1)
    vix_return_5d = _percentage_return(vix_close, 5)

    vix_sma20 = vix_close.rolling(
        window=20, min_periods=20
    ).mean()

    vix_close_vs_sma20 = vix_close / vix_sma20 - 1.0

    # --------------------------------------------------
    # Stock relative performance against NIFTY
    # --------------------------------------------------

    stock_return_1d = _percentage_return(stock_close, 1)
    stock_return_5d = _percentage_return(stock_close, 5)
    stock_return_20d = _percentage_return(stock_close, 20)

    relative_return_1d = (
        (1.0 + stock_return_1d) / (1.0 + nifty_return_1d)
    ) - 1.0

    relative_return_5d = (
        (1.0 + stock_return_5d) / (1.0 + nifty_return_5d)
    ) - 1.0

    relative_return_20d = (
        (1.0 + stock_return_20d) / (1.0 + nifty_return_20d)
    ) - 1.0

    # --------------------------------------------------
    # Final feature frame
    # --------------------------------------------------

    features = pd.DataFrame({
        "session_date": stock["session_date"],

        "nifty_return_1d": nifty_return_1d,
        "nifty_return_5d": nifty_return_5d,
        "nifty_return_20d": nifty_return_20d,

        "nifty_realized_volatility_20d": nifty_volatility_20d,

        "nifty_close_vs_sma20": nifty_close_vs_sma20,
        "nifty_close_vs_sma50": nifty_close_vs_sma50,

        "india_vix_close": vix_close,
        "india_vix_return_1d": vix_return_1d,
        "india_vix_return_5d": vix_return_5d,
        "india_vix_close_vs_sma20": vix_close_vs_sma20,

        "relative_return_1d": relative_return_1d,
        "relative_return_5d": relative_return_5d,
        "relative_return_20d": relative_return_20d,
    })

    numeric_features = features[MARKET_CONTEXT_FEATURES]

    features.loc[:, MARKET_CONTEXT_FEATURES] = (
        numeric_features.replace([np.inf, -np.inf], np.nan)
    )

    return features
