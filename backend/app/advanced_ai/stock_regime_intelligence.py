from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math

import numpy as np

from stock_regime_models import (
    StockRegimeInput,
    StockRegimeResult,
)


class StockRegimeIntelligence:
    """
    Phase 14.2 — Independent Stock Regime Intelligence.

    Purpose
    -------
    Determine the current regime of the SELECTED STOCK using only
    its OHLCV candle history.

    This component is intentionally independent from:
        - RandomForest
        - LSTM
        - Phase 9
        - Phase 14 Layer 1
        - recommendation logic
        - market context logic

    It does NOT produce BUY / SELL / HOLD.
    It only describes the stock's current regime.
    """

    MIN_CANDLES = 60

    def __init__(self) -> None:
        self.name = "Stock Regime Intelligence"
        self.layer = "14.2"

    # =========================================================
    # BASIC HELPERS
    # =========================================================

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")

        if not math.isfinite(number):
            return float("nan")

        return number

    @staticmethod
    def _extract_candle(candle: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        """
        Accept common OHLCV key variants.
        """

        def pick(*keys: str) -> float:
            for key in keys:
                if key in candle:
                    return StockRegimeIntelligence._safe_float(
                        candle[key]
                    )
            return float("nan")

        timestamp = pick(
            "timestamp",
            "time",
            "ts",
            "date",
        )

        open_price = pick(
            "open",
            "o",
        )

        high = pick(
            "high",
            "h",
        )

        low = pick(
            "low",
            "l",
        )

        close = pick(
            "close",
            "c",
        )

        volume = pick(
            "volume",
            "v",
        )

        return (
            open_price,
            high,
            low,
            close,
            volume,
        )

    @classmethod
    def _prepare_candles(
        cls,
        candles: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[float] = []

        for candle in candles:

            if not isinstance(candle, dict):
                continue

            open_price, high, low, close, volume = (
                cls._extract_candle(candle)
            )

            if not all(
                math.isfinite(x)
                for x in (
                    open_price,
                    high,
                    low,
                    close,
                    volume,
                )
            ):
                continue

            if open_price <= 0:
                continue

            if close <= 0:
                continue

            if high < low:
                continue

            if high < max(open_price, close):
                continue

            if low > min(open_price, close):
                continue

            if volume < 0:
                continue

            opens.append(open_price)
            highs.append(high)
            lows.append(low)
            closes.append(close)
            volumes.append(volume)

        return (
            np.asarray(opens, dtype=float),
            np.asarray(highs, dtype=float),
            np.asarray(lows, dtype=float),
            np.asarray(closes, dtype=float),
            np.asarray(volumes, dtype=float),
        )

    # =========================================================
    # EMA
    # =========================================================

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        if len(values) < period:
            return np.full(len(values), np.nan)

        alpha = 2.0 / (period + 1.0)

        result = np.full(len(values), np.nan)
        result[period - 1] = np.mean(
            values[:period]
        )

        for i in range(period, len(values)):
            result[i] = (
                alpha * values[i]
                + (1.0 - alpha) * result[i - 1]
            )

        return result

    # =========================================================
    # ATR / ADX
    # =========================================================

    @staticmethod
    def _wilder_smoothing(
        values: np.ndarray,
        period: int,
    ) -> np.ndarray:

        result = np.full(len(values), np.nan)

        if len(values) < period:
            return result

        result[period - 1] = np.mean(
            values[:period]
        )

        for i in range(period, len(values)):
            result[i] = (
                (
                    result[i - 1] * (period - 1)
                )
                + values[i]
            ) / period

        return result

    @classmethod
    def _atr_adx(
        cls,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> Tuple[float, float, float, float]:

        if len(closes) < period + 1:
            return 0.0, 0.0, 0.0, 0.0

        previous_close = closes[:-1]

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - previous_close),
                np.abs(lows[1:] - previous_close),
            ),
        )

        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        )

        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        )

        atr = cls._wilder_smoothing(
            tr,
            period,
        )

        plus_smoothed = cls._wilder_smoothing(
            plus_dm,
            period,
        )

        minus_smoothed = cls._wilder_smoothing(
            minus_dm,
            period,
        )

        plus_di = np.where(
            atr > 0,
            100.0 * plus_smoothed / atr,
            0.0,
        )

        minus_di = np.where(
            atr > 0,
            100.0 * minus_smoothed / atr,
            0.0,
        )

        denominator = plus_di + minus_di

        dx = np.zeros_like(denominator, dtype=float)

        valid_denominator = denominator > 0

        np.divide(
            100.0 * np.abs(plus_di - minus_di),
            denominator,
            out=dx,
            where=valid_denominator,
        )

        adx = cls._wilder_smoothing(
            dx,
            period,
        )

        atr_value = float(
            atr[-1]
        ) if np.isfinite(atr[-1]) else 0.0

        adx_value = float(
            adx[-1]
        ) if np.isfinite(adx[-1]) else 0.0

        plus_di_value = float(
            plus_di[-1]
        ) if np.isfinite(plus_di[-1]) else 0.0

        minus_di_value = float(
            minus_di[-1]
        ) if np.isfinite(minus_di[-1]) else 0.0

        return (
            atr_value,
            adx_value,
            plus_di_value,
            minus_di_value,
        )

    # =========================================================
    # REGIME
    # =========================================================

    @staticmethod
    def _classify_momentum(
        momentum_percent: float,
    ) -> str:

        if momentum_percent >= 5.0:
            return "strong_positive"

        if momentum_percent >= 1.0:
            return "positive"

        if momentum_percent <= -5.0:
            return "strong_negative"

        if momentum_percent <= -1.0:
            return "negative"

        return "neutral"

    @staticmethod
    def _classify_volatility(
        atr_percent: float,
    ) -> str:

        if atr_percent >= 4.0:
            return "high"

        if atr_percent >= 2.0:
            return "elevated"

        if atr_percent <= 0.75:
            return "low"

        return "normal"

    @staticmethod
    def _price_structure(
        price: float,
        ema20: float | None,
        ema50: float | None,
        ema200: float | None,
    ) -> str:

        if ema20 is None or ema50 is None:
            return "insufficient_ema_history"

        if ema200 is not None:

            if (
                price > ema20 > ema50 > ema200
            ):
                return "strong_bullish_structure"

            if (
                price < ema20 < ema50 < ema200
            ):
                return "strong_bearish_structure"

        if price > ema20 and ema20 > ema50:
            return "bullish_structure"

        if price < ema20 and ema20 < ema50:
            return "bearish_structure"

        return "mixed_structure"

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        data: StockRegimeInput,
    ) -> StockRegimeResult:

        (
            opens,
            highs,
            lows,
            closes,
            volumes,
        ) = self._prepare_candles(
            data.candles
        )

        candle_count = len(closes)

        minimum = max(
            self.MIN_CANDLES,
            int(data.min_candles),
        )

        if candle_count < minimum:
            raise ValueError(
                f"Insufficient valid OHLCV candles: "
                f"{candle_count}. Minimum required: {minimum}."
            )

        price = float(closes[-1])

        ema20_series = self._ema(
            closes,
            20,
        )

        ema50_series = self._ema(
            closes,
            50,
        )

        ema200_series = (
            self._ema(closes, 200)
            if candle_count >= 200
            else np.full(
                candle_count,
                np.nan,
            )
        )

        ema20 = float(
            ema20_series[-1]
        )

        ema50 = float(
            ema50_series[-1]
        )

        ema200 = (
            float(ema200_series[-1])
            if np.isfinite(
                ema200_series[-1]
            )
            else None
        )

        (
            atr,
            adx,
            plus_di,
            minus_di,
        ) = self._atr_adx(
            highs,
            lows,
            closes,
            period=14,
        )

        atr_percent = (
            (atr / price) * 100.0
            if price > 0
            else 0.0
        )

        if candle_count >= 21:
            momentum_percent = (
                (
                    price
                    / closes[-21]
                ) - 1.0
            ) * 100.0
        else:
            momentum_percent = 0.0

        momentum_regime = (
            self._classify_momentum(
                momentum_percent
            )
        )

        volatility_regime = (
            self._classify_volatility(
                atr_percent
            )
        )

        price_structure = (
            self._price_structure(
                price,
                ema20,
                ema50,
                ema200,
            )
        )

        # =====================================================
        # Independent regime votes
        # =====================================================

        bullish_votes = 0
        bearish_votes = 0
        active_votes = 0

        evidence: List[str] = []

        # EMA structure
        if (
            price > ema20
            and ema20 > ema50
        ):
            bullish_votes += 1
            active_votes += 1
            evidence.append(
                "Price is above EMA20 and EMA20 is above EMA50."
            )

        elif (
            price < ema20
            and ema20 < ema50
        ):
            bearish_votes += 1
            active_votes += 1
            evidence.append(
                "Price is below EMA20 and EMA20 is below EMA50."
            )

        # ADX / DMI
        if adx >= 20.0:

            if plus_di > minus_di:
                bullish_votes += 1
                active_votes += 1
                evidence.append(
                    "ADX confirms a directional trend with "
                    "+DI above -DI."
                )

            elif minus_di > plus_di:
                bearish_votes += 1
                active_votes += 1
                evidence.append(
                    "ADX confirms a directional trend with "
                    "-DI above +DI."
                )

        # Momentum
        if momentum_percent >= 1.0:
            bullish_votes += 1
            active_votes += 1
            evidence.append(
                f"20-day momentum is positive at "
                f"{momentum_percent:.2f}%."
            )

        elif momentum_percent <= -1.0:
            bearish_votes += 1
            active_votes += 1
            evidence.append(
                f"20-day momentum is negative at "
                f"{momentum_percent:.2f}%."
            )

        # Long-term EMA if available
        if ema200 is not None:

            if (
                price > ema200
                and ema50 > ema200
            ):
                bullish_votes += 1
                active_votes += 1
                evidence.append(
                    "Price and EMA50 are above EMA200."
                )

            elif (
                price < ema200
                and ema50 < ema200
            ):
                bearish_votes += 1
                active_votes += 1
                evidence.append(
                    "Price and EMA50 are below EMA200."
                )

        # =====================================================
        # Direction
        # =====================================================

        if bullish_votes > bearish_votes:
            direction = "bullish"

        elif bearish_votes > bullish_votes:
            direction = "bearish"

        else:
            direction = "neutral"

        # =====================================================
        # Trend strength
        # =====================================================

        adx_strength = self._clamp(
            adx / 50.0
        )

        if active_votes > 0:
            directional_agreement = (
                max(
                    bullish_votes,
                    bearish_votes,
                )
                / active_votes
            )
        else:
            directional_agreement = 0.0

        trend_strength = self._clamp(
            (
                adx_strength * 0.60
                + directional_agreement * 0.40
            )
        )

        # =====================================================
        # Confidence
        # =====================================================

        if direction == "neutral":

            confidence = self._clamp(
                0.35
                + (
                    0.25
                    * (
                        1.0
                        - directional_agreement
                    )
                )
            )

        else:

            confidence = self._clamp(
                (
                    directional_agreement * 0.65
                )
                + (
                    adx_strength * 0.35
                )
            )

        # Avoid falsely high confidence when ADX is weak.
        if adx < 15.0:
            confidence *= 0.75

        confidence = round(
            self._clamp(confidence),
            4,
        )

        # =====================================================
        # Regime classification
        # =====================================================

        if direction == "bullish":

            if trend_strength >= 0.70:
                regime = "strong_bullish_trend"
            else:
                regime = "bullish_trend"

        elif direction == "bearish":

            if trend_strength >= 0.70:
                regime = "strong_bearish_trend"
            else:
                regime = "bearish_trend"

        else:

            if volatility_regime in {
                "high",
                "elevated",
            }:
                regime = "volatile_sideways"

            else:
                regime = "sideways"

        evidence.append(
            f"Stock regime classified as {regime} "
            f"with {confidence:.2%} confidence."
        )

        return StockRegimeResult(
            instrument_key=data.instrument_key,
            stock_regime=regime,
            stock_direction=direction,
            stock_regime_confidence=confidence,
            trend_strength=round(
                self._clamp(
                    trend_strength
                ),
                4,
            ),
            momentum_regime=momentum_regime,
            volatility_regime=volatility_regime,
            price_structure=price_structure,
            current_price=round(
                price,
                4,
            ),
            ema_20=round(
                ema20,
                4,
            ),
            ema_50=round(
                ema50,
                4,
            ),
            ema_200=(
                round(ema200, 4)
                if ema200 is not None
                else None
            ),
            momentum_20d_percent=round(
                momentum_percent,
                4,
            ),
            atr_percent=round(
                atr_percent,
                4,
            ),
            adx_14=round(
                adx,
                4,
            ),
            plus_di=round(
                plus_di,
                4,
            ),
            minus_di=round(
                minus_di,
                4,
            ),
            candle_count=candle_count,
            evidence=evidence,
            status="success",
        )