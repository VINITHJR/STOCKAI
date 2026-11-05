"""
Phase 15 stock-level technical feature engineering.

Research-only module.

Features are based exclusively on current and historical OHLCV observations.
All price-derived features are scale-normalized.

No production LSTM V2, Phase 9 or Phase 14 files are modified.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class TechnicalFeatureError(ValueError):
    """Raised when stock data cannot produce valid technical features."""


RSI_PERIOD = 14
ATR_PERIOD = 14
SHORT_MA_PERIOD = 20
LONG_MA_PERIOD = 50
VOLATILITY_PERIOD = 20
VOLUME_PERIOD = 20
BOLLINGER_PERIOD = 20

STOCK_TECHNICAL_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "overnight_gap_return",
    "intraday_return",
    "daily_range_pct",
    "close_location",
    "realized_volatility_20d",
    "sma20_gap",
    "sma50_gap",
    "rsi_wilder_14",
    "macd_pct",
    "macd_signal_pct",
    "bb_width_20",
    "bb_percent_b_20",
    "atr_pct_14",
    "relative_volume_log_20",
    "trend_efficiency_20d",
]


def _prepare_ohlcv(dataset: Dict[str, Any]) -> pd.DataFrame:
    """Validate normalized daily OHLCV data without silently dropping rows."""

    if not isinstance(dataset, dict):
        raise TechnicalFeatureError("Dataset must be a dictionary.")

    rows = dataset.get("rows")

    if not isinstance(rows, list) or len(rows) < 60:
        raise TechnicalFeatureError(
            "At least 60 normalized daily candles are required."
        )

    frame = pd.DataFrame(rows)

    required = {
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(frame.columns)

    if missing:
        raise TechnicalFeatureError(
            f"Missing required columns: {sorted(missing)}"
        )

    dates = pd.to_datetime(
        frame["session_date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    if dates.isna().any():
        raise TechnicalFeatureError("Invalid or missing session dates.")

    if dates.duplicated().any():
        raise TechnicalFeatureError("Duplicate trading sessions detected.")

    if not dates.is_monotonic_increasing:
        raise TechnicalFeatureError(
            "Candles must be sorted oldest to newest."
        )

    frame = frame.copy()
    frame["session_date"] = dates.dt.strftime("%Y-%m-%d")

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    values = frame[
        ["open", "high", "low", "close", "volume"]
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise TechnicalFeatureError(
            "OHLCV contains missing or non-finite values."
        )

    if (values[:, :4] <= 0).any():
        raise TechnicalFeatureError("OHLC prices must be positive.")

    if (frame["volume"] < 0).any():
        raise TechnicalFeatureError("Volume cannot be negative.")

    if (
        frame["high"]
        < frame[["open", "low", "close"]].max(axis=1)
    ).any():
        raise TechnicalFeatureError("Invalid high prices detected.")

    if (
        frame["low"]
        > frame[["open", "high", "close"]].min(axis=1)
    ).any():
        raise TechnicalFeatureError("Invalid low prices detected.")

    return frame.reset_index(drop=True)


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothed moving average.

    Seed = arithmetic mean of the first `period` valid observations.
    Subsequent value = (previous * (period - 1) + current) / period.
    """

    if period < 1:
        raise ValueError("Wilder period must be positive.")

    values = series.to_numpy(dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)

    valid_positions = np.flatnonzero(np.isfinite(values))

    if len(valid_positions) < period:
        return pd.Series(output, index=series.index)

    seed_positions = valid_positions[:period]
    seed_index = int(seed_positions[-1])

    previous = float(np.mean(values[seed_positions]))
    output[seed_index] = previous

    for index in range(seed_index + 1, len(values)):
        current = values[index]

        if not np.isfinite(current):
            continue

        previous = (
            previous * (period - 1) + current
        ) / period

        output[index] = previous

    return pd.Series(output, index=series.index)


def _wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI using Wilder's initial average and recursive smoothing."""

    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)

    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100.0,
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50.0,
    )

    return rsi / 100.0


def _calculate_true_range(frame: pd.DataFrame) -> pd.Series:
    """Calculate daily true range using the previous session close."""

    previous_close = frame["close"].shift(1)

    components = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    return components.max(axis=1)


def build_stock_technical_features(
    stock_data: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build 18 scale-normalized stock technical features.

    Input:
        Normalized dataset returned by historical_loader.py.

    Output:
        session_date + STOCK_TECHNICAL_FEATURES.

    All features use current or past session information only.
    Rolling warm-up values remain NaN.
    """

    frame = _prepare_ohlcv(stock_data)

    open_price = frame["open"]
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    volume = frame["volume"]

    # --------------------------------------------------
    # Returns and price behaviour
    # --------------------------------------------------

    return_1d = close.pct_change(1, fill_method=None)
    return_5d = close.pct_change(5, fill_method=None)
    return_20d = close.pct_change(20, fill_method=None)

    overnight_gap_return = (
        open_price / close.shift(1) - 1.0
    )

    intraday_return = close / open_price - 1.0

    daily_range_pct = (high - low) / close

    candle_range = high - low

    close_location = (
        (close - low) / candle_range.replace(0.0, np.nan)
    )

    # A completely flat candle is assigned a neutral close location.
    close_location = close_location.mask(
        candle_range == 0.0,
        0.5,
    )

    # --------------------------------------------------
    # Trend and volatility
    # --------------------------------------------------

    realized_volatility_20d = (
        return_1d
        .rolling(
            window=VOLATILITY_PERIOD,
            min_periods=VOLATILITY_PERIOD,
        )
        .std(ddof=1)
    )

    sma20 = close.rolling(
        window=SHORT_MA_PERIOD,
        min_periods=SHORT_MA_PERIOD,
    ).mean()

    sma50 = close.rolling(
        window=LONG_MA_PERIOD,
        min_periods=LONG_MA_PERIOD,
    ).mean()

    sma20_gap = close / sma20 - 1.0
    sma50_gap = close / sma50 - 1.0

    daily_log_return = np.log(close / close.shift(1))
    net_log_return_20d = np.log(close / close.shift(20))

    absolute_log_path_20d = daily_log_return.abs().rolling(
        window=20,
        min_periods=20,
    ).sum()

    trend_efficiency_20d = (
        net_log_return_20d
        / absolute_log_path_20d.replace(0.0, np.nan)
    )

    # A completely flat 20-session path has neutral efficiency.
    trend_efficiency_20d = trend_efficiency_20d.mask(
        absolute_log_path_20d == 0.0,
        0.0,
    )

    # --------------------------------------------------
    # Wilder RSI
    # --------------------------------------------------

    rsi_wilder_14 = _wilder_rsi(close, RSI_PERIOD)

    # --------------------------------------------------
    # MACD
    # --------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    macd_pct = macd / close
    macd_signal_pct = macd_signal / close

    # --------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------

    bb_middle = close.rolling(
        window=BOLLINGER_PERIOD,
        min_periods=BOLLINGER_PERIOD,
    ).mean()

    bb_std = close.rolling(
        window=BOLLINGER_PERIOD,
        min_periods=BOLLINGER_PERIOD,
    ).std(ddof=0)

    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std

    bb_width_20 = (
        (bb_upper - bb_lower) / bb_middle
    )

    bb_denominator = bb_upper - bb_lower

    bb_percent_b_20 = (
        (close - bb_lower)
        / bb_denominator.replace(0.0, np.nan)
    )

    # The middle band is not exported as a separate feature.
    # It would duplicate the SMA20 calculation.

    # --------------------------------------------------
    # Wilder ATR
    # --------------------------------------------------

    true_range = _calculate_true_range(frame)

    atr = _wilder_rma(true_range, ATR_PERIOD)

    atr_pct_14 = atr / close

    # --------------------------------------------------
    # Relative volume
    # --------------------------------------------------

    # The baseline uses only the PREVIOUS 20 sessions.
    # Today's volume is compared against that prior-history average.
    previous_volume_average = (
        volume.shift(1)
        .rolling(
            window=VOLUME_PERIOD,
            min_periods=VOLUME_PERIOD,
        )
        .mean()
    )

    relative_volume = (
        volume / previous_volume_average.replace(0.0, np.nan)
    )

    relative_volume_log_20 = np.log1p(relative_volume)

    # --------------------------------------------------
    # Final features
    # --------------------------------------------------

    features = pd.DataFrame({
        "session_date": frame["session_date"],

        "return_1d": return_1d,
        "return_5d": return_5d,
        "return_20d": return_20d,

        "overnight_gap_return": overnight_gap_return,
        "intraday_return": intraday_return,
        "daily_range_pct": daily_range_pct,
        "close_location": close_location,

        "realized_volatility_20d": realized_volatility_20d,

        "sma20_gap": sma20_gap,
        "sma50_gap": sma50_gap,

        "rsi_wilder_14": rsi_wilder_14,

        "macd_pct": macd_pct,
        "macd_signal_pct": macd_signal_pct,

        "bb_width_20": bb_width_20,
        "bb_percent_b_20": bb_percent_b_20,

        "atr_pct_14": atr_pct_14,

        "relative_volume_log_20": relative_volume_log_20,

        "trend_efficiency_20d": trend_efficiency_20d,
    })

    numeric = features[STOCK_TECHNICAL_FEATURES]

    features.loc[:, STOCK_TECHNICAL_FEATURES] = (
        numeric.replace([np.inf, -np.inf], np.nan)
    )

    return features
