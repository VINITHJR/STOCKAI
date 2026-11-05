from __future__ import annotations

from math import isfinite
from statistics import mean, stdev
from typing import Any


class TechnicalAnalysisIntelligence:
    """Phase 14.1 — independent deterministic technical analysis.

    Uses only OHLCV candles. It is intentionally independent from the RF/LSTM
    outputs so Phase 14 does not double-count the same model evidence.
    """

    MIN_CANDLES = 60

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _sma(values: list[float], window: int) -> float | None:
        return mean(values[-window:]) if len(values) >= window else None

    @staticmethod
    def _ema(values: list[float], window: int) -> float | None:
        if len(values) < window:
            return None
        alpha = 2.0 / (window + 1.0)
        ema = mean(values[:window])
        for value in values[window:]:
            ema = alpha * value + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def _rsi(values: list[float], window: int = 14) -> float | None:
        if len(values) < window + 1:
            return None
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        gains = [max(change, 0.0) for change in changes]
        losses = [max(-change, 0.0) for change in changes]
        avg_gain = mean(gains[:window])
        avg_loss = mean(losses[:window])
        for i in range(window, len(changes)):
            avg_gain = ((avg_gain * (window - 1)) + gains[i]) / window
            avg_loss = ((avg_loss * (window - 1)) + losses[i]) / window
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(highs: list[float], lows: list[float], closes: list[float], window: int = 14) -> float | None:
        if len(closes) < window + 1:
            return None
        trs = []
        for i in range(1, len(closes)):
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
        if len(trs) < window:
            return None

        # Wilder-style ATR smoothing, matching the convention used by
        # the directional indicators below.
        atr = mean(trs[:window])
        for tr in trs[window:]:
            atr = ((atr * (window - 1)) + tr) / window
        return atr

    @classmethod
    def _adx(cls, highs: list[float], lows: list[float], closes: list[float], window: int = 14) -> tuple[float | None, float | None, float | None]:
        if len(closes) < (window * 2) + 1:
            return None, None, None

        trs: list[float] = []
        plus_dm: list[float] = []
        minus_dm: list[float] = []
        for i in range(1, len(closes)):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0.0)
            minus_dm.append(down if down > up and down > 0 else 0.0)
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

        if len(trs) < window:
            return None, None, None

        atr = mean(trs[:window])
        p = mean(plus_dm[:window])
        m = mean(minus_dm[:window])
        dx_values: list[float] = []
        plus_di = minus_di = 0.0

        for i in range(window, len(trs)):
            atr = ((atr * (window - 1)) + trs[i]) / window
            p = ((p * (window - 1)) + plus_dm[i]) / window
            m = ((m * (window - 1)) + minus_dm[i]) / window
            plus_di = 100.0 * p / atr if atr else 0.0
            minus_di = 100.0 * m / atr if atr else 0.0
            denom = plus_di + minus_di
            dx_values.append(100.0 * abs(plus_di - minus_di) / denom if denom else 0.0)

        if len(dx_values) < window:
            return None, None, None
        adx = mean(dx_values[:window])
        for dx in dx_values[window:]:
            adx = ((adx * (window - 1)) + dx) / window
        return adx, plus_di, minus_di

    @staticmethod
    def _normalize_candles(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        normalized = []
        for row in rows:
            try:
                close = float(row["close"])
                open_price = (
                    float(row["open"])
                    if row.get("open") is not None
                    else close
                )
                high = float(row["high"])
                low = float(row["low"])
                volume = float(row.get("volume") or 0.0)

                values = (open_price, high, low, close, volume)
                if not all(isfinite(value) for value in values):
                    continue
                if high < low or close <= 0 or volume < 0:
                    continue

                normalized.append({
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                })
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return normalized

    @classmethod
    def analyze(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        candles = cls._normalize_candles(rows)
        if len(candles) < cls.MIN_CANDLES:
            raise ValueError(
                f"At least {cls.MIN_CANDLES} usable candles are required for technical analysis; received {len(candles)}."
            )

        closes = [r["close"] for r in candles]
        highs = [r["high"] for r in candles]
        lows = [r["low"] for r in candles]
        volumes = [r["volume"] for r in candles]
        current = closes[-1]

        ema20 = cls._ema(closes, 20)
        ema50 = cls._ema(closes, 50)
        ema200 = cls._ema(closes, 200)
        sma20 = cls._sma(closes, 20)
        rsi14 = cls._rsi(closes, 14)
        atr14 = cls._atr(highs, lows, closes, 14)
        adx14, plus_di, minus_di = cls._adx(highs, lows, closes, 14)

        macd_line = None
        macd_signal = None
        macd_hist = None
        if len(closes) >= 35:
            # Calculate the full MACD series to avoid using a single unstable point.
            alpha12 = 2.0 / 13.0
            alpha26 = 2.0 / 27.0
            e12 = mean(closes[:12])
            e26 = mean(closes[:26])
            macd_series: list[float] = []
            for i, price in enumerate(closes):
                if i >= 12:
                    e12 = alpha12 * price + (1 - alpha12) * e12
                if i >= 26:
                    e26 = alpha26 * price + (1 - alpha26) * e26
                if i >= 26:
                    macd_series.append(e12 - e26)
            if len(macd_series) >= 9:
                macd_line = macd_series[-1]
                macd_signal = cls._ema(macd_series, 9)
                if macd_signal is not None:
                    macd_hist = macd_line - macd_signal

        bb_mid = sma20
        bb_std = stdev(closes[-20:]) if len(closes) >= 20 else None
        bb_upper = bb_lower = bb_position = None
        if bb_mid is not None and bb_std is not None:
            bb_upper = bb_mid + 2.0 * bb_std
            bb_lower = bb_mid - 2.0 * bb_std
            width = bb_upper - bb_lower
            bb_position = (current - bb_lower) / width if width else 0.5

        volume_ratio = None
        if len(volumes) >= 20:
            avg_volume = mean(volumes[-20:])
            if avg_volume > 0:
                volume_ratio = volumes[-1] / avg_volume

        momentum20 = (current / closes[-21] - 1.0) if closes[-21] else 0.0

        # Independent indicator votes. Each indicator contributes at most one vote.
        votes: list[tuple[str, int, float]] = []
        if ema20 is not None and ema50 is not None:
            if current > ema20 > ema50:
                votes.append(("EMA trend", 1, 0.90))
            elif current < ema20 < ema50:
                votes.append(("EMA trend", -1, 0.90))
            else:
                votes.append(("EMA trend", 0, 0.45))

        if rsi14 is not None:
            # RSI is used here as a momentum/trend confirmation signal.
            # Extreme readings do not automatically reverse the direction:
            # overbought/oversold are conditions, not guaranteed reversals.
            if rsi14 >= 55:
                strength = cls._clamp(0.55 + abs(rsi14 - 50) / 40.0)
                votes.append(("RSI", 1, strength))
            elif rsi14 <= 45:
                strength = cls._clamp(0.55 + abs(rsi14 - 50) / 40.0)
                votes.append(("RSI", -1, strength))
            else:
                votes.append(("RSI", 0, 0.40))

        if macd_hist is not None:
            scale = max(abs(macd_line or 0.0), current * 0.002, 1e-9)
            strength = cls._clamp(abs(macd_hist) / scale)
            votes.append(("MACD", 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0, max(0.45, strength)))

        if adx14 is not None and plus_di is not None and minus_di is not None:
            if adx14 >= 20:
                direction = 1 if plus_di > minus_di else -1 if minus_di > plus_di else 0
                votes.append(("ADX/DMI", direction, cls._clamp(adx14 / 50.0)))
            else:
                votes.append(("ADX/DMI", 0, 0.35))

        if bb_position is not None:
            if bb_position > 0.80 and momentum20 > 0:
                votes.append(("Bollinger", 1, 0.60))
            elif bb_position < 0.20 and momentum20 < 0:
                votes.append(("Bollinger", -1, 0.60))
            else:
                votes.append(("Bollinger", 0, 0.35))

        if volume_ratio is not None and volume_ratio >= 1.20 and momentum20 != 0:
            votes.append(("Volume confirmation", 1 if momentum20 > 0 else -1, cls._clamp(volume_ratio / 2.0)))

        weighted = sum(direction * confidence for _, direction, confidence in votes)
        weight_total = sum(confidence for _, _, confidence in votes) or 1.0
        directional_score = weighted / weight_total

        if directional_score >= 0.20:
            signal = "bullish"
        elif directional_score <= -0.20:
            signal = "bearish"
        else:
            signal = "neutral"

        target_direction = 1 if signal == "bullish" else -1 if signal == "bearish" else 0
        agreement = sum(1 for _, d, _ in votes if d == target_direction)
        active_votes = sum(1 for _, d, _ in votes if d != 0)
        if signal == "neutral":
            agreement_ratio = agreement / max(len(votes), 1)
        else:
            agreement_ratio = agreement / max(active_votes, 1)
        confidence = cls._clamp(0.45 * abs(directional_score) + 0.35 * agreement_ratio + 0.20 * min(1.0, (adx14 or 0.0) / 40.0))
        if signal == "neutral":
            confidence *= 0.75

        explanation = []
        if ema20 is not None and ema50 is not None:
            explanation.append(f"EMA structure is {'bullish' if current > ema20 > ema50 else 'bearish' if current < ema20 < ema50 else 'mixed'}.")
        if rsi14 is not None:
            explanation.append(f"RSI(14) is {rsi14:.2f}.")
        if macd_hist is not None:
            explanation.append(f"MACD histogram is {'positive' if macd_hist > 0 else 'negative' if macd_hist < 0 else 'flat'}.")
        if adx14 is not None:
            explanation.append(f"ADX(14) is {adx14:.2f}.")
        if volume_ratio is not None:
            explanation.append(f"Volume is {volume_ratio:.2f}x the 20-day average.")

        return {
            "signal": signal,
            "confidence": round(cls._clamp(confidence), 4),
            "directional_score": round(directional_score, 4),
            "vote_count": len(votes),
            "active_vote_count": active_votes,
            "agreement_count": agreement,
            "indicators": {
                "current_price": round(current, 4),
                "rsi_14": round(rsi14, 4) if rsi14 is not None else None,
                "ema_20": round(ema20, 4) if ema20 is not None else None,
                "ema_50": round(ema50, 4) if ema50 is not None else None,
                "ema_200": round(ema200, 4) if ema200 is not None else None,
                "macd": round(macd_line, 6) if macd_line is not None else None,
                "macd_signal": round(macd_signal, 6) if macd_signal is not None else None,
                "macd_histogram": round(macd_hist, 6) if macd_hist is not None else None,
                "adx_14": round(adx14, 4) if adx14 is not None else None,
                "plus_di": round(plus_di, 4) if plus_di is not None else None,
                "minus_di": round(minus_di, 4) if minus_di is not None else None,
                "bollinger_middle": round(bb_mid, 4) if bb_mid is not None else None,
                "bollinger_upper": round(bb_upper, 4) if bb_upper is not None else None,
                "bollinger_lower": round(bb_lower, 4) if bb_lower is not None else None,
                "bollinger_position": round(bb_position, 4) if bb_position is not None else None,
                "atr_14": round(atr14, 4) if atr14 is not None else None,
                "atr_percent": round((atr14 / current) * 100, 4) if atr14 is not None and current else None,
                "momentum_20d_percent": round(momentum20 * 100, 4),
                "volume_ratio_20d": round(volume_ratio, 4) if volume_ratio is not None else None,
            },
            "explanation": explanation,
            "method": "deterministic_ohlcv_multi_indicator_v1",
            "candle_count": len(candles),
        }
