import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type LogicalRange,
  type Logical,
  type Time,
} from "lightweight-charts";

type ChartMode = "line" | "candle";

export type LiveChartPeriod =
  | "1D"
  | "1W"
  | "1M"
  | "3M"
  | "6M"
  | "1Y"
  | "3Y"
  | "5Y"
  | "MAX";

type Period = LiveChartPeriod;

type RawCandle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

type RawVolume = {
  time: number;
  volume: number;
};

type Candle = {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
};

export type TechnicalSummary = {
  score: number;
  bias: "Bullish" | "Bearish" | "Neutral";
  trend: "Bullish" | "Bearish" | "Neutral";
  momentum: "Positive" | "Negative" | "Neutral";
  volume: "Strong" | "Weak" | "Neutral" | "Unavailable";
  rsi: number | null;
  rsiStatus: "Overbought" | "Oversold" | "Neutral" | "Unavailable";
  macdStatus: "Bullish" | "Bearish" | "Neutral" | "Unavailable";
  vwapStatus: "Above VWAP" | "Below VWAP" | "At VWAP" | "Unavailable";
  price: number | null;
  sma20: number | null;
  sma50: number | null;
  vwap: number | null;
  support1: number | null;
  support2: number | null;
  resistance1: number | null;
  resistance2: number | null;
  signal: "BUY" | "HOLD" | "SELL";
  direction: "LONG" | "WAIT" | "SHORT";
  confidence: number;
  riskLevel: "Low" | "Medium" | "High";
  setupQuality: "Strong" | "Good" | "Fair" | "Poor";
  entry: number | null;
  entryZone: string | null;
  stopLoss: number | null;
  target1: number | null;
  target2: number | null;
  riskReward: number | null;
  atr: number | null;
  atrPercent: number | null;
  dataQuality: "High" | "Medium" | "Low";
  volumeRatio: number | null;
  volumeConfirmation: "Confirmed" | "Weak" | "Neutral" | "Unavailable";
  actionTrigger: string;
  invalidation: string;
  scoreBreakdown: { label: string; score: number; weight: number; reason: string }[];
  confirmationGates: { label: string; passed: boolean; detail: string }[];
  decisionReasons: string[];
  notes: string[];
};

type LiveMarketChartProps = {
  instrumentKey: string;
  symbol?: string;
  name?: string;
  currentPrice?: number | null;
  currentChange?: number | null;
  marketStatus?: string;
  isIndex?: boolean;
  period?: LiveChartPeriod;
  onPeriodChange?: (period: LiveChartPeriod) => void;
  onTechnicalSummaryChange?: (summary: TechnicalSummary | null) => void;
};

type PeriodConfig = {
  label: Period;
  unit: "minutes" | "hours" | "days" | "weeks" | "months";
  interval: string;
  days: number | null;
  live: boolean;
};

const API_BASE = "http://127.0.0.1:8000";
const IST_OFFSET_SECONDS = 5 * 60 * 60 + 30 * 60;

const PERIODS: PeriodConfig[] = [
  { label: "1D", unit: "minutes", interval: "1", days: 1, live: true },
  { label: "1W", unit: "minutes", interval: "15", days: 8, live: false },
  { label: "1M", unit: "days", interval: "1", days: 35, live: false },
  { label: "3M", unit: "days", interval: "1", days: 100, live: false },
  { label: "6M", unit: "days", interval: "1", days: 190, live: false },
  { label: "1Y", unit: "days", interval: "1", days: 380, live: false },
  { label: "3Y", unit: "weeks", interval: "1", days: 3 * 365 + 30, live: false },
  { label: "5Y", unit: "weeks", interval: "1", days: 5 * 365 + 30, live: false },
  { label: "MAX", unit: "months", interval: "1", days: null, live: false },
];

function toISTChartTime(seconds: number): number {
  return seconds + IST_OFFSET_SECONDS;
}

function formatISTDateTimeFromChartTime(chartTime: number): string {
  const date = new Date((chartTime - IST_OFFSET_SECONDS) * 1000);

  return date.toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "short",
    year: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

function formatISTTickFromChartTime(chartTime: number): string {
  const date = new Date((chartTime - IST_OFFSET_SECONDS) * 1000);

  const time = date.toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  const day = date.toLocaleDateString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "short",
  });

  return `${day} ${time}`;
}

function timestampToSeconds(value: unknown): number | null {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;
    return value > 10_000_000_000
      ? Math.floor(value / 1000)
      : Math.floor(value);
  }

  if (typeof value === "string") {
    const numeric = Number(value);

    if (Number.isFinite(numeric)) {
      return numeric > 10_000_000_000
        ? Math.floor(numeric / 1000)
        : Math.floor(numeric);
    }

    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) return Math.floor(parsed / 1000);
  }

  return null;
}

function parseCandle(item: unknown): RawCandle | null {
  if (Array.isArray(item)) {
    const time = timestampToSeconds(item[0]);
    const open = Number(item[1]);
    const high = Number(item[2]);
    const low = Number(item[3]);
    const close = Number(item[4]);
    const volume = Number(item[5]);

    if (
      time === null ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }

    return {
      time,
      open,
      high,
      low,
      close,
      volume: Number.isFinite(volume) ? volume : undefined,
    };
  }

  if (item && typeof item === "object") {
    const obj = item as Record<string, unknown>;
    const time = timestampToSeconds(
      obj.time ?? obj.timestamp ?? obj.ts ?? obj.datetime ?? obj.date
    );
    const open = Number(obj.open ?? obj.o);
    const high = Number(obj.high ?? obj.h);
    const low = Number(obj.low ?? obj.l);
    const close = Number(obj.close ?? obj.c);
    const volume = Number(obj.volume ?? obj.v);

    if (
      time === null ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }

    return {
      time,
      open,
      high,
      low,
      close,
      volume: Number.isFinite(volume) ? volume : undefined,
    };
  }

  return null;
}

function parseVolume(item: unknown): RawVolume | null {
  if (!item || typeof item !== "object") return null;

  const obj = item as Record<string, unknown>;
  const time = timestampToSeconds(
    obj.time ?? obj.timestamp ?? obj.ts ?? obj.datetime ?? obj.date
  );
  const volume = Number(obj.volume ?? obj.value ?? obj.v);

  if (time === null || !Number.isFinite(volume) || volume < 0) return null;

  return { time, volume };
}

function extractCandles(payload: any): RawCandle[] {
  const candidates = [
    payload?.data?.candles,
    payload?.candles,
    payload?.data?.data?.candles,
    payload?.data,
  ];

  for (const candidate of candidates) {
    if (!Array.isArray(candidate)) continue;

    const parsed = candidate
      .map(parseCandle)
      .filter((item): item is RawCandle => item !== null);

    if (parsed.length > 0) return parsed;
  }

  return [];
}

function extractIndexVolumes(payload: any): RawVolume[] {
  const candidates = [
    payload?.candles,
    payload?.data?.candles,
    payload?.data,
  ];

  for (const candidate of candidates) {
    if (!Array.isArray(candidate)) continue;

    const parsed = candidate
      .map(parseVolume)
      .filter((item): item is RawVolume => item !== null);

    if (parsed.length > 0) return parsed;
  }

  return [];
}

function sortAndDeduplicate(candles: RawCandle[]): RawCandle[] {
  const map = new Map<number, RawCandle>();

  for (const candle of candles) map.set(candle.time, candle);

  return Array.from(map.values()).sort((a, b) => a.time - b.time);
}

function minuteKey(seconds: number): number {
  return Math.floor(seconds / 60) * 60;
}

function normalizeVolumes(volumes: RawVolume[]): RawVolume[] {
  const map = new Map<number, RawVolume>();

  for (const item of volumes) {
    const time = minuteKey(item.time);
    map.set(time, { time, volume: item.volume });
  }

  return Array.from(map.values()).sort((a, b) => a.time - b.time);
}

function calculateSMA(candles: RawCandle[], period: number) {
  if (period <= 0) return [] as { time: Time; value: number }[];

  const result: { time: Time; value: number }[] = [];
  let sum = 0;

  for (let i = 0; i < candles.length; i += 1) {
    sum += candles[i].close;
    if (i >= period) sum -= candles[i - period].close;

    if (i >= period - 1) {
      result.push({
        time: candles[i].time as Time,
        value: sum / period,
      });
    }
  }

  return result;
}

function calculateBollingerBands(candles: RawCandle[], period = 20, multiplier = 2) {
  const result: { time: Time; middle: number; upper: number; lower: number }[] = [];
  if (period <= 0 || candles.length < period) return result;

  let sum = 0;
  for (let i = 0; i < candles.length; i += 1) {
    sum += candles[i].close;
    if (i >= period) sum -= candles[i - period].close;

    if (i >= period - 1) {
      const middle = sum / period;
      let varianceSum = 0;
      for (let j = i - period + 1; j <= i; j += 1) {
        const diff = candles[j].close - middle;
        varianceSum += diff * diff;
      }
      const standardDeviation = Math.sqrt(varianceSum / period);
      result.push({
        time: candles[i].time as Time,
        middle,
        upper: middle + multiplier * standardDeviation,
        lower: middle - multiplier * standardDeviation,
      });
    }
  }

  return result;
}

function calculateRSI(candles: RawCandle[], period = 14) {
  const result: { time: Time; value: number }[] = [];
  if (candles.length <= period || period <= 0) return result;

  let gainSum = 0;
  let lossSum = 0;

  for (let i = 1; i <= period; i += 1) {
    const change = candles[i].close - candles[i - 1].close;
    if (change >= 0) gainSum += change;
    else lossSum += Math.abs(change);
  }

  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;

  const firstRSI = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  result.push({ time: candles[period].time as Time, value: firstRSI });

  for (let i = period + 1; i < candles.length; i += 1) {
    const change = candles[i].close - candles[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    result.push({ time: candles[i].time as Time, value: rsi });
  }

  return result;
}


function calculateEMA(values: number[], period: number): number[] {
  const result = new Array<number>(values.length).fill(Number.NaN);
  if (period <= 0 || values.length < period) return result;
  let sum = 0;
  for (let i = 0; i < period; i += 1) sum += values[i];
  let ema = sum / period;
  result[period - 1] = ema;
  const multiplier = 2 / (period + 1);
  for (let i = period; i < values.length; i += 1) {
    ema = (values[i] - ema) * multiplier + ema;
    result[i] = ema;
  }
  return result;
}

function calculateMACD(candles: RawCandle[], fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  const result: { time: Time; macd: number; signal: number; histogram: number }[] = [];
  if (candles.length < slowPeriod + signalPeriod - 1) return result;
  const closes = candles.map((c) => c.close);
  const fast = calculateEMA(closes, fastPeriod);
  const slow = calculateEMA(closes, slowPeriod);
  const macdValues = new Array<number>(closes.length).fill(Number.NaN);
  for (let i = slowPeriod - 1; i < closes.length; i += 1) {
    if (Number.isFinite(fast[i]) && Number.isFinite(slow[i])) macdValues[i] = fast[i] - slow[i];
  }
  const valid = macdValues.filter((v): v is number => Number.isFinite(v));
  const signals = calculateEMA(valid, signalPeriod);
  let validIndex = 0;
  for (let i = slowPeriod - 1; i < candles.length; i += 1) {
    if (!Number.isFinite(macdValues[i])) continue;
    const signal = signals[validIndex++];
    if (!Number.isFinite(signal)) continue;
    const macd = macdValues[i];
    result.push({ time: candles[i].time as Time, macd, signal, histogram: macd - signal });
  }
  return result;
}
function calculateATR(candles: RawCandle[], period = 14): number | null {
  if (candles.length < 2 || period <= 0) return null;

  const trueRanges: number[] = [];
  for (let i = 1; i < candles.length; i += 1) {
    const current = candles[i];
    const previousClose = candles[i - 1].close;
    const tr = Math.max(
      current.high - current.low,
      Math.abs(current.high - previousClose),
      Math.abs(current.low - previousClose),
    );
    if (Number.isFinite(tr) && tr >= 0) trueRanges.push(tr);
  }

  if (trueRanges.length < period) return null;

  let atr = trueRanges.slice(0, period).reduce((sum, value) => sum + value, 0) / period;
  for (let i = period; i < trueRanges.length; i += 1) {
    atr = ((atr * (period - 1)) + trueRanges[i]) / period;
  }
  return Number.isFinite(atr) ? atr : null;
}

export function calculateTechnicalSummary(
  candles: RawCandle[],
  indexInstrument: boolean,
  indexVolumes: RawVolume[],
  currentPrice: number | null
): TechnicalSummary | null {
  if (candles.length < 20) return null;

  const latest = candles[candles.length - 1];
  const price = Number.isFinite(currentPrice) && currentPrice !== null
    ? currentPrice
    : latest.close;
  if (!Number.isFinite(price) || price <= 0) return null;

  const sma20 = calculateSMA(candles, 20);
  const sma50 = calculateSMA(candles, 50);
  const rsiData = calculateRSI(candles, 14);
  const macdData = calculateMACD(candles, 12, 26, 9);
  const atr = calculateATR(candles, 14);
  const atrPercent = atr !== null ? (atr / price) * 100 : null;

  const vwapCandles = indexInstrument
    ? candles.map((candle) => ({
        ...candle,
        volume: indexVolumes.find(
          (item) => minuteKey(item.time) === minuteKey(candle.time)
        )?.volume,
      }))
    : candles;

  const vwapData = calculateVWAP(vwapCandles);
  const sr = calculateSupportResistance(candles);

  const latestSma20 = sma20.at(-1)?.value ?? null;
  const latestSma50 = sma50.at(-1)?.value ?? null;
  const latestRsi = rsiData.at(-1)?.value ?? null;
  const latestMacd = macdData.at(-1) ?? null;
  const previousMacd = macdData.at(-2) ?? null;
  const latestVwap = vwapData.at(-1)?.value ?? null;

  // The latest 1-minute candle can still be forming while the market is open.
  // For signal confirmation, deliberately exclude the newest volume observation
  // and compare completed candles instead of allowing a partial candle to distort
  // the volume signal.
  const rawVolumeValues = (
    indexInstrument
      ? indexVolumes.map((v) => v.volume)
      : candles.map((c) => c.volume)
  ).filter((v): v is number => typeof v === "number" && Number.isFinite(v) && v > 0);

  const completedVolumeValues = rawVolumeValues.length > 1
    ? rawVolumeValues.slice(0, -1)
    : rawVolumeValues;

  let volumeStatus: TechnicalSummary["volume"] = "Unavailable";
  let volumeRatio: number | null = null;
  let volumeConfirmation: TechnicalSummary["volumeConfirmation"] = "Unavailable";

  if (completedVolumeValues.length >= 10) {
    const recentWindow = completedVolumeValues.slice(-5);
    const baselineWindow = completedVolumeValues.slice(-20, -5);
    const recentAvg = recentWindow.reduce((a, b) => a + b, 0) / recentWindow.length;
    const baselineAvg = baselineWindow.length
      ? baselineWindow.reduce((a, b) => a + b, 0) / baselineWindow.length
      : recentAvg;

    if (baselineAvg > 0) {
      volumeRatio = recentAvg / baselineAvg;
      volumeStatus = volumeRatio >= 1.20
        ? "Strong"
        : volumeRatio <= 0.80
          ? "Weak"
          : "Neutral";
      volumeConfirmation = volumeStatus === "Strong"
        ? "Confirmed"
        : volumeStatus === "Weak"
          ? "Weak"
          : "Neutral";
    }
  }

  let vwapStatus: TechnicalSummary["vwapStatus"] = "Unavailable";
  if (latestVwap !== null && Number.isFinite(latestVwap)) {
    const diff = Math.abs(price - latestVwap) / Math.max(Math.abs(latestVwap), 1);
    vwapStatus = diff <= 0.001
      ? "At VWAP"
      : price > latestVwap
        ? "Above VWAP"
        : "Below VWAP";
  }

  const rsiStatus: TechnicalSummary["rsiStatus"] =
    latestRsi === null
      ? "Unavailable"
      : latestRsi >= 70
        ? "Overbought"
        : latestRsi <= 30
          ? "Oversold"
          : "Neutral";

  const macdStatus: TechnicalSummary["macdStatus"] =
    latestMacd === null
      ? "Unavailable"
      : latestMacd.macd > latestMacd.signal
        ? "Bullish"
        : latestMacd.macd < latestMacd.signal
          ? "Bearish"
          : "Neutral";

  // -------------------------------------------------------------
  // Weighted signal model
  // -------------------------------------------------------------
  // We score independent evidence rather than adding arbitrary points.
  // The weights total 100 when every component is available.
  type Component = {
    label: string;
    weight: number;
    direction: number; // -1 bearish, 0 neutral, +1 bullish
    reason: string;
  };

  const components: Component[] = [];

  if (latestSma20 !== null) {
    components.push({
      label: "SMA 20",
      weight: 10,
      direction: price > latestSma20 ? 1 : price < latestSma20 ? -1 : 0,
      reason: price > latestSma20
        ? "Price is above the short-term SMA 20."
        : price < latestSma20
          ? "Price is below the short-term SMA 20."
          : "Price is at SMA 20.",
    });
  }

  if (latestSma50 !== null) {
    components.push({
      label: "SMA 50",
      weight: 10,
      direction: price > latestSma50 ? 1 : price < latestSma50 ? -1 : 0,
      reason: price > latestSma50
        ? "Price is above the medium-term SMA 50."
        : price < latestSma50
          ? "Price is below the medium-term SMA 50."
          : "Price is at SMA 50.",
    });
  }

  if (latestSma20 !== null && latestSma50 !== null) {
    components.push({
      label: "MA structure",
      weight: 10,
      direction: latestSma20 > latestSma50 ? 1 : latestSma20 < latestSma50 ? -1 : 0,
      reason: latestSma20 > latestSma50
        ? "SMA 20 is above SMA 50, supporting an upward structure."
        : latestSma20 < latestSma50
          ? "SMA 20 is below SMA 50, supporting a downward structure."
          : "SMA 20 and SMA 50 are nearly equal.",
    });
  }

  let rsiDirection = 0;
  let rsiReason = "RSI is in a neutral zone.";
  if (latestRsi !== null) {
    if (latestRsi >= 55 && latestRsi < 70) {
      rsiDirection = 1;
      rsiReason = "RSI has positive momentum without being overbought.";
    } else if (latestRsi > 50 && latestRsi < 55) {
      rsiDirection = 0.35;
      rsiReason = "RSI is mildly positive.";
    } else if (latestRsi > 45 && latestRsi <= 50) {
      rsiDirection = 0;
      rsiReason = "RSI is close to neutral.";
    } else if (latestRsi > 30 && latestRsi <= 45) {
      rsiDirection = -1;
      rsiReason = "RSI shows weak momentum.";
    } else if (latestRsi <= 30) {
      rsiDirection = -0.45;
      rsiReason = "RSI is oversold; downside momentum is weak but reversal risk is elevated.";
    } else if (latestRsi >= 70) {
      rsiDirection = 0.25;
      rsiReason = "RSI is overbought; bullish momentum exists but chasing risk is elevated.";
    }
    components.push({ label: "RSI 14", weight: 10, direction: rsiDirection, reason: rsiReason });
  }

  if (latestMacd !== null) {
    let macdDirection = latestMacd.macd > latestMacd.signal
      ? 1
      : latestMacd.macd < latestMacd.signal
        ? -1
        : 0;
    let macdReason = macdDirection > 0
      ? "MACD is above its signal line."
      : macdDirection < 0
        ? "MACD is below its signal line."
        : "MACD and signal are aligned.";

    if (previousMacd) {
      if (macdDirection > 0 && latestMacd.histogram < previousMacd.histogram) {
        macdDirection *= 0.60;
        macdReason += " Bullish momentum is weakening."
      } else if (macdDirection < 0 && latestMacd.histogram > previousMacd.histogram) {
        macdDirection *= 0.60;
        macdReason += " Bearish momentum is weakening."
      }
    }
    components.push({ label: "MACD", weight: 15, direction: macdDirection, reason: macdReason });
  }

  if (vwapStatus !== "Unavailable") {
    components.push({
      label: "VWAP",
      weight: 15,
      direction: vwapStatus === "Above VWAP" ? 1 : vwapStatus === "Below VWAP" ? -1 : 0,
      reason: vwapStatus === "Above VWAP"
        ? "Price is above the session VWAP."
        : vwapStatus === "Below VWAP"
          ? "Price is below the session VWAP."
          : "Price is close to the session VWAP.",
    });
  }

  if (volumeStatus !== "Unavailable") {
    // Volume confirms direction only when it is actually strong. Weak volume
    // should reduce conviction, not automatically become bearish.
    let direction = 0;
    let reason = `Recent completed volume is ${volumeStatus.toLowerCase()}.`;
    if (volumeStatus === "Strong") {
      const priceDirection = latest.close >= latest.open ? 1 : -1;
      direction = priceDirection;
      reason = priceDirection > 0
        ? "Strong recent completed volume supports the latest upward candle direction."
        : "Strong recent completed volume supports the latest downward candle direction.";
    } else if (volumeStatus === "Weak") {
      reason = "Recent completed volume is weak, so directional confirmation is limited.";
    }
    components.push({ label: "Volume", weight: 10, direction, reason });
  }

  const nearSupport = sr.support1 !== null && Math.abs(price - sr.support1) / price <= 0.008;
  const nearResistance = sr.resistance1 !== null && Math.abs(sr.resistance1 - price) / price <= 0.008;

  if (sr.support1 !== null || sr.resistance1 !== null) {
    let direction = 0;
    let reason = "No immediate support/resistance proximity is strong enough to add directional bias.";

    // Keep S/R scoring consistent with the same proximity test used by the
    // decision gates. When price is close to BOTH levels, S/R is contextually
    // important but directionally neutral: support can attract demand while
    // resistance can cap upside. It must not silently become a bearish or
    // bullish score merely because one level is marginally closer.
    if (nearSupport && nearResistance) {
      reason = `Price is near both support at ₹${sr.support1!.toFixed(2)} and resistance at ₹${sr.resistance1!.toFixed(2)}; S/R context is neutral until one level breaks.`;
    } else if (nearSupport) {
      direction = 0.35;
      reason = `Price is near support at ₹${sr.support1!.toFixed(2)}; downside may find demand.`;
    } else if (nearResistance) {
      direction = -0.35;
      reason = `Price is near resistance at ₹${sr.resistance1!.toFixed(2)}; upside may face supply.`;
    }

    components.push({ label: "S/R context", weight: 10, direction, reason });
  }

  const totalWeight = components.reduce((sum, item) => sum + item.weight, 0);
  const weightedDirection = totalWeight > 0
    ? components.reduce((sum, item) => sum + item.direction * item.weight, 0) / totalWeight
    : 0;

  // Signal Score is directional strength: 0 = weak/no edge, 100 = very strong.
  // Direction is supplied separately by the bias, so a bearish setup can have
  // a high score instead of the old confusing "14/100 = bearish" convention.
  const signalScore = Math.max(0, Math.min(100, Math.round(Math.abs(weightedDirection) * 100)));
  const bias: TechnicalSummary["bias"] =
    weightedDirection >= 0.20 ? "Bullish"
      : weightedDirection <= -0.20 ? "Bearish"
        : "Neutral";

  // Directional agreement is deliberately independent of indicator weight.
  // This prevents one heavy indicator from creating a trade by itself.
  const nonNeutral = components.filter((item) => Math.abs(item.direction) >= 0.20);
  const signedAgreement = nonNeutral.length
    ? nonNeutral.reduce((sum, item) => sum + Math.sign(item.direction), 0) / nonNeutral.length
    : 0;
  const agreement = Math.abs(signedAgreement);
  const availabilityRatio = totalWeight / 100;
  const directionalComponentCount = nonNeutral.length;

  const bullishAgreement = signedAgreement >= 0.67;
  const bearishAgreement = signedAgreement <= -0.67;

  // Confirmation layer: a directional score is not enough. Strong setups use
  // directional volume or a confirmed breakout/breakdown; normal setups may
  // qualify from clean structure when volume is not weak.
  const confirmationCandle = candles.length > 1 ? candles[candles.length - 2] : latest;

  const volumeConfirmedForDirection =
    volumeConfirmation === "Confirmed" &&
    ((bias === "Bullish" && confirmationCandle.close >= confirmationCandle.open) ||
      (bias === "Bearish" && confirmationCandle.close < confirmationCandle.open));

  // Compare the current candle with levels that existed before it. This gives
  // us a genuine breakdown/breakout check rather than comparing price with a
  // support/resistance level that was recalculated around the current candle.
  const previousSr = candles.length > 20
    ? calculateSupportResistance(candles.slice(0, -1))
    : { support1: null, support2: null, resistance1: null, resistance2: null };

  const supportBreakdown =
    bias === "Bearish" &&
    previousSr.support1 !== null &&
    confirmationCandle.close < previousSr.support1;

  const resistanceBreakout =
    bias === "Bullish" &&
    previousSr.resistance1 !== null &&
    confirmationCandle.close > previousSr.resistance1;

  // Safety filters requested for the decision hierarchy:
  // bearish + deeply oversold + near support => HOLD/WAIT unless support
  // has actually broken with confirmation.
  const bearishOversoldTrap =
    bias === "Bearish" &&
    latestRsi !== null &&
    latestRsi <= 30 &&
    nearSupport &&
    !supportBreakdown;

  const bullishOverboughtTrap =
    bias === "Bullish" &&
    latestRsi !== null &&
    latestRsi >= 70 &&
    nearResistance &&
    !resistanceBreakout;

  const dataQuality: TechnicalSummary["dataQuality"] =
    availabilityRatio >= 0.90 && candles.length >= 80
      ? "High"
      : availabilityRatio >= 0.70 && candles.length >= 50
        ? "Medium"
        : "Low";

  // ---------------------------------------------------------------------------
  // Tiered trade-decision model
  // ---------------------------------------------------------------------------
  // A technical bias and a trade entry are deliberately different things.
  // The engine may be bullish/bearish while still returning HOLD when the
  // setup is late, trapped between S/R, overextended, or poorly confirmed.
  //
  // Tier 1 (strong): score >= 70 + >=67% agreement + 4 directional signals
  //                   + directional volume + clean price location.
  // Tier 2 (normal): score >= 60 + >=67% agreement + 4 signals + either
  //                   directional volume OR a confirmed breakout/breakdown.
  // Extreme RSI/SR traps always block a fresh entry until confirmation.

  const minimumDirectionalComponents = directionalComponentCount >= 4;
  const agreementConfirmed = agreement >= 0.67 && minimumDirectionalComponents;

  const bullishVolumeConfirmed =
    volumeConfirmation === "Confirmed" && confirmationCandle.close >= confirmationCandle.open;
  const bearishVolumeConfirmed =
    volumeConfirmation === "Confirmed" && confirmationCandle.close < confirmationCandle.open;

  const bullishCleanLocation =
    (!nearResistance || resistanceBreakout) && !bullishOverboughtTrap;
  const bearishCleanLocation =
    (!nearSupport || supportBreakdown) && !bearishOversoldTrap;

  // A breakout/breakdown is stronger than simply being on the correct side of
  // VWAP or a moving average because the level existed before the latest bar.
  const bullishConfirmation = bullishVolumeConfirmed || resistanceBreakout;
  const bearishConfirmation = bearishVolumeConfirmed || supportBreakdown;

  // Estimate the raw structural reward/risk before deciding whether a trade is
  // worth taking. This uses the same support/resistance + ATR framework later
  // used to display the actual stop and targets, so the signal cannot qualify
  // on a setup whose nearby structure offers poor reward/risk.
  const safeAtrForCandidate = atr !== null && atr > 0 ? atr : price * 0.01;
  const candidateLongStop = sr.support1 !== null && sr.support1 < price
    ? Math.min(price - safeAtrForCandidate * 0.80, sr.support1 - safeAtrForCandidate * 0.15)
    : price - safeAtrForCandidate;
  const candidateLongTarget = sr.resistance1 !== null && sr.resistance1 > price
    ? sr.resistance1
    : price + safeAtrForCandidate * 1.5;
  const candidateShortStop = sr.resistance1 !== null && sr.resistance1 > price
    ? Math.max(price + safeAtrForCandidate * 0.80, sr.resistance1 + safeAtrForCandidate * 0.15)
    : price + safeAtrForCandidate;
  const candidateShortTarget = sr.support1 !== null && sr.support1 < price
    ? sr.support1
    : Math.max(0.01, price - safeAtrForCandidate * 1.5);

  const candidateLongRisk = Math.max(price - candidateLongStop, safeAtrForCandidate * 0.50);
  const candidateShortRisk = Math.max(candidateShortStop - price, safeAtrForCandidate * 0.50);
  const candidateLongReward = Math.max(0, candidateLongTarget - price);
  const candidateShortReward = Math.max(0, price - candidateShortTarget);
  const candidateLongRR = candidateLongRisk > 0 ? candidateLongReward / candidateLongRisk : 0;
  const candidateShortRR = candidateShortRisk > 0 ? candidateShortReward / candidateShortRisk : 0;

  // Normal setups can qualify without a fresh breakout if the price is not
  // trapped at the opposite S/R level. Strong setups still require actual
  // directional volume or a confirmed breakout/breakdown.
  const normalBullishConfirmation = bullishConfirmation ||
    (volumeStatus !== "Weak" && !nearResistance && !bullishOverboughtTrap);
  const normalBearishConfirmation = bearishConfirmation ||
    (volumeStatus !== "Weak" && !nearSupport && !bearishOversoldTrap);

  const strongBullishSetup =
    bias === "Bullish" &&
    signalScore >= 70 &&
    agreementConfirmed &&
    bullishConfirmation &&
    bullishCleanLocation &&
    candidateLongRR >= 1.50;

  const strongBearishSetup =
    bias === "Bearish" &&
    signalScore >= 70 &&
    agreementConfirmed &&
    bearishConfirmation &&
    bearishCleanLocation &&
    candidateShortRR >= 1.50;

  const normalBullishSetup =
    bias === "Bullish" &&
    signalScore >= 60 &&
    agreementConfirmed &&
    normalBullishConfirmation &&
    bullishCleanLocation &&
    candidateLongRR >= 1.50;

  const normalBearishSetup =
    bias === "Bearish" &&
    signalScore >= 60 &&
    agreementConfirmed &&
    normalBearishConfirmation &&
    bearishCleanLocation &&
    candidateShortRR >= 1.50;

  let signal: TechnicalSummary["signal"] = "HOLD";

  if (strongBullishSetup || normalBullishSetup) {
    signal = "BUY";
  } else if (strongBearishSetup || normalBearishSetup) {
    signal = "SELL";
  }

  // ---------------------------------------------------------------------------
  // Confidence is a confidence-in-signal-quality score, NOT a probability of
  // profit. Strong directional agreement earns confidence; contradictory or
  // missing confirmation reduces it.
  // ---------------------------------------------------------------------------
  let confidence = 40 + signalScore * 0.35 + agreement * 15 + availabilityRatio * 8;
  if (bullishVolumeConfirmed || bearishVolumeConfirmed) confidence += 7;
  if (resistanceBreakout || supportBreakdown) confidence += 5;
  if (signal === "HOLD") confidence -= 4;
  if (bearishOversoldTrap || bullishOverboughtTrap) confidence -= 10;
  if (volumeStatus === "Weak") confidence -= 8;
  if (dataQuality === "Low") confidence -= 12;
  confidence = Math.max(35, Math.min(92, Math.round(confidence)));

  const confirmationGates = [
    {
      label: "Directional score",
      passed: signalScore >= 60,
      detail: `${signalScore}/100 (60+ minimum; 70+ for strong setup)`,
    },
    {
      label: "Indicator agreement",
      passed: agreementConfirmed,
      detail: `${Math.round(agreement * 100)}% across ${directionalComponentCount} directional indicators (67% + 4 required)`,
    },
    {
      label: "Volume confirmation",
      passed: (bias === "Bullish" ? normalBullishConfirmation : bias === "Bearish" ? normalBearishConfirmation : false),
      detail: (bias === "Bullish" ? normalBullishConfirmation : bias === "Bearish" ? normalBearishConfirmation : false)
        ? (bullishVolumeConfirmed || bearishVolumeConfirmed || resistanceBreakout || supportBreakdown
          ? `Strong confirmation for ${bias.toLowerCase()} direction`
          : `Clean structure; directional volume is not weak`)

        : volumeStatus === "Unavailable"
          ? "Unavailable"
          : `${volumeStatus} / ${volumeRatio !== null ? `${volumeRatio.toFixed(2)}x baseline` : "no ratio"}`,
    },
    {
      label: "Price location",
      passed: bias === "Bearish"
        ? bearishCleanLocation
        : bias === "Bullish"
          ? bullishCleanLocation
          : false,
      detail: bias === "Bearish"
        ? bearishOversoldTrap
          ? "Blocked: deeply oversold near support"
          : supportBreakdown
            ? "Support breakdown confirmed"
            : nearSupport && nearResistance
              ? "Near support and resistance; bearish entry needs a confirmed support break"
              : nearSupport
                ? "Near support"
                : "Clear of immediate support"
        : bias === "Bullish"
          ? bullishOverboughtTrap
            ? "Blocked: overbought near resistance"
            : resistanceBreakout
              ? "Resistance breakout confirmed"
              : nearResistance && nearSupport
                ? "Near support and resistance; bullish entry needs a confirmed resistance break"
                : nearResistance
                  ? "Near resistance"
                  : "Clear of immediate resistance"
          : "No directional setup",
    },
  ];

  const safeAtr = atr !== null && atr > 0 ? atr : price * 0.01;
  const roundPrice = (value: number) => Number(value.toFixed(2));

  let entry: number | null = null;
  let entryZone: string | null = null;
  let stopLoss: number | null = null;
  let target1: number | null = null;
  let target2: number | null = null;

  if (signal === "BUY") {
    entry = roundPrice(price);
    const zoneLow = Math.max(0.01, price - safeAtr * 0.25);
    const zoneHigh = price + safeAtr * 0.10;
    entryZone = `${formatPrice(roundPrice(zoneLow))} – ${formatPrice(roundPrice(zoneHigh))}`;

    const structuralStop = sr.support1 !== null && sr.support1 < price
      ? sr.support1 - safeAtr * 0.15
      : price - safeAtr * 1.25;
    stopLoss = roundPrice(Math.max(0.01, Math.min(structuralStop, price - safeAtr * 0.80)));

    const risk = Math.max(price - stopLoss, safeAtr * 0.50);
    const resistanceTarget = sr.resistance1 !== null && sr.resistance1 > price
      ? sr.resistance1
      : price + safeAtr * 1.5;
    target1 = roundPrice(Math.max(resistanceTarget, price + risk * 1.50));

    const secondReference = sr.resistance2 !== null && sr.resistance2 > target1
      ? sr.resistance2
      : price + risk * 2.50;
    target2 = roundPrice(Math.max(secondReference, target1 + risk * 0.75));
  } else if (signal === "SELL") {
    entry = roundPrice(price);
    const zoneHigh = price + safeAtr * 0.25;
    const zoneLow = Math.max(0.01, price - safeAtr * 0.10);
    entryZone = `${formatPrice(roundPrice(zoneLow))} – ${formatPrice(roundPrice(zoneHigh))}`;

    const structuralStop = sr.resistance1 !== null && sr.resistance1 > price
      ? sr.resistance1 + safeAtr * 0.15
      : price + safeAtr * 1.25;
    stopLoss = roundPrice(Math.max(price + safeAtr * 0.80, structuralStop));

    const risk = Math.max(stopLoss - price, safeAtr * 0.50);
    const supportTarget = sr.support1 !== null && sr.support1 < price
      ? sr.support1
      : price - safeAtr * 1.5;
    target1 = roundPrice(Math.min(supportTarget, price - risk * 1.50));

    const secondReference = sr.support2 !== null && sr.support2 < target1
      ? sr.support2
      : Math.max(0.01, price - risk * 2.50);
    target2 = roundPrice(Math.min(secondReference, target1 - risk * 0.75));
  }

  const riskDistance = stopLoss !== null ? Math.abs(price - stopLoss) : null;
  const rewardDistance = target1 !== null ? Math.abs(target1 - price) : null;
  const riskReward = riskDistance && rewardDistance
    ? Number((rewardDistance / riskDistance).toFixed(2))
    : null;
  const stopPercent = riskDistance !== null ? (riskDistance / price) * 100 : null;

  let riskLevel: TechnicalSummary["riskLevel"] = "Medium";
  if (signal === "HOLD") {
    riskLevel = atrPercent !== null && atrPercent > 3.5 ? "High" : "Medium";
  } else if (
    atrPercent !== null &&
    stopPercent !== null &&
    atrPercent <= 1.5 &&
    stopPercent <= 2.0 &&
    (riskReward === null || riskReward >= 1.5)
  ) {
    riskLevel = "Low";
  } else if (
    (atrPercent !== null && atrPercent > 3.5) ||
    (stopPercent !== null && stopPercent > 4) ||
    (riskReward !== null && riskReward < 1.2)
  ) {
    riskLevel = "High";
  }

  const setupQuality: TechnicalSummary["setupQuality"] =
    signal === "HOLD"
      ? agreement >= 0.67 && signalScore >= 60 ? "Fair" : "Poor"
      : (signalScore >= 70 && agreement >= 0.80 && dataQuality === "High") ? "Strong"
        : (signalScore >= 60 && agreement >= 0.67 && dataQuality !== "Low") ? "Good"
          : "Fair";

  const direction: TechnicalSummary["direction"] =
    signal === "BUY" ? "LONG" : signal === "SELL" ? "SHORT" : "WAIT";

  let actionTrigger = "Wait for stronger agreement between trend, momentum and volume before taking a directional trade.";
  let invalidation = "No active trade setup. Recalculate after a confirmed breakout or breakdown.";

  if (signal === "BUY") {
    actionTrigger = sr.resistance1 !== null
      ? `Prefer a confirmed move above ₹${sr.resistance1.toFixed(2)} with stronger completed volume.`
      : "Prefer a continuation move with stronger completed volume and price above the key moving averages.";
    invalidation = stopLoss !== null
      ? `Bullish setup invalidates below ₹${stopLoss.toFixed(2)} or if momentum loses confirmation.`
      : "Bullish setup invalidates if trend and momentum turn bearish.";
  } else if (signal === "SELL") {
    actionTrigger = sr.support1 !== null
      ? `Prefer a confirmed move below ₹${sr.support1.toFixed(2)} with stronger completed selling volume.`
      : "Prefer a breakdown with stronger completed volume and price below the key moving averages.";
    invalidation = stopLoss !== null
      ? `Bearish setup invalidates above ₹${stopLoss.toFixed(2)} or if momentum turns bullish.`
      : "Bearish setup invalidates if trend and momentum turn bullish.";
  } else {
    if (bias === "Bearish") {
      actionTrigger = sr.resistance1 !== null && sr.support1 !== null
        ? `Wait for rejection near ₹${sr.resistance1.toFixed(2)} or a confirmed breakdown below ₹${sr.support1.toFixed(2)} with volume.`
        : "Wait for clearer bearish confirmation from price, momentum and volume.";
    } else if (bias === "Bullish") {
      actionTrigger = sr.resistance1 !== null
        ? `Wait for a confirmed breakout above ₹${sr.resistance1.toFixed(2)} with stronger completed volume.`
        : "Wait for stronger bullish confirmation from price, momentum and volume.";
    }
  }

  const decisionReasons: string[] = [];
  if (trendPointsFromSummary(latestSma20, latestSma50, price) > 0) {
    decisionReasons.push("Price/trend structure is bullish versus the moving averages.");
  } else if (trendPointsFromSummary(latestSma20, latestSma50, price) < 0) {
    decisionReasons.push("Price/trend structure is bearish versus the moving averages.");
  } else {
    decisionReasons.push("Moving-average trend evidence is mixed.");
  }

  if (latestRsi !== null) {
    decisionReasons.push(`RSI 14 is ${latestRsi.toFixed(1)} (${rsiStatus.toLowerCase()}).`);
  }
  if (macdStatus !== "Unavailable") decisionReasons.push(`MACD is ${macdStatus.toLowerCase()}.`);
  if (vwapStatus !== "Unavailable") decisionReasons.push(`Price is ${vwapStatus.toLowerCase()}.`);
  if (volumeStatus === "Strong") decisionReasons.push(`Completed volume is strong (${volumeRatio!.toFixed(2)}x baseline).`);
  if (volumeStatus === "Weak") decisionReasons.push(`Completed volume is weak (${volumeRatio!.toFixed(2)}x baseline), reducing confirmation.`);
  if (nearResistance && sr.resistance1 !== null) decisionReasons.push(`Price is near resistance at ₹${sr.resistance1.toFixed(2)}.`);
  if (nearSupport && sr.support1 !== null) decisionReasons.push(`Price is near support at ₹${sr.support1.toFixed(2)}.`);

  if (signal === "HOLD") {
    if (bias === "Bearish") {
      if (signalScore < 60) decisionReasons.push(`Bearish score is ${signalScore}/100, below the 60-point minimum SELL threshold.`);
      else if (!strongBearishSetup) decisionReasons.push(`Bearish setup qualifies only as a normal-tier candidate; strong confirmation requires 70+ strength plus directional volume or a confirmed breakdown.`);
      if (!bearishAgreement) decisionReasons.push(`Bearish indicator agreement is ${(agreement * 100).toFixed(0)}%, below the 67% confirmation threshold.`);
      if (!volumeConfirmedForDirection) decisionReasons.push("Selling volume is not confirmed, so a bearish bias cannot become a SELL signal.");
      if (bearishOversoldTrap) decisionReasons.push("RSI is deeply oversold near support; fresh short entry is blocked until support breaks.");
    } else if (bias === "Bullish") {
      if (signalScore < 60) decisionReasons.push(`Bullish score is ${signalScore}/100, below the 60-point minimum BUY threshold.`);
      else if (!strongBullishSetup) decisionReasons.push(`Bullish setup qualifies only as a normal-tier candidate; strong confirmation requires 70+ strength plus directional volume or a confirmed breakout.`);
      if (!bullishAgreement) decisionReasons.push(`Bullish indicator agreement is ${(agreement * 100).toFixed(0)}%, below the 67% confirmation threshold.`);
      if (!volumeConfirmedForDirection) decisionReasons.push("Buying volume is not confirmed, so a bullish bias cannot become a BUY signal.");
      if (bullishOverboughtTrap) decisionReasons.push("RSI is overbought near resistance; fresh long entry is blocked until breakout confirmation.");
    } else {
      decisionReasons.push("Signals are mixed, so the engine is waiting for a clearer directional setup.");
    }
  }

  const notes: string[] = [
    bias === "Bullish"
      ? `Technical bias is bullish; directional strength is ${signalScore}/100.`
      : bias === "Bearish"
        ? `Technical bias is bearish; directional strength is ${signalScore}/100.`
        : "Technical bias is neutral because weighted evidence is balanced.",
    signal === "HOLD"
      ? "HOLD means wait for confirmation. A bearish/bullish bias alone is never enough for a trade entry."
      : `${signal} passed the score, agreement, volume and price-location confirmation gates.`,
    "Strong BUY/SELL requires 70/100 strength, 67% indicator agreement, 4+ directional signals and directional confirmation. Normal BUY/SELL can qualify from 60/100 when confirmation and price location are clean.",
    volumeStatus === "Unavailable"
      ? "Volume confirmation is unavailable."
      : "Volume analysis uses completed observations to reduce partial-candle distortion.",
  ];

  const scoreBreakdown = components.map((item) => ({
    label: item.label,
    score: Math.round(item.direction * item.weight),
    weight: item.weight,
    reason: item.reason,
  }));

  return {
    score: signalScore,
    bias,
    trend:
      trendPointsFromSummary(latestSma20, latestSma50, price) >= 2
        ? "Bullish"
        : trendPointsFromSummary(latestSma20, latestSma50, price) <= -2
          ? "Bearish"
          : "Neutral",
    momentum:
      ((latestRsi !== null && rsiDirection > 0) ? 1 : 0) +
      ((latestMacd !== null && latestMacd.macd > latestMacd.signal) ? 1 : 0) >= 1
        ? "Positive"
        : ((latestRsi !== null && rsiDirection < 0) ? 1 : 0) +
          ((latestMacd !== null && latestMacd.macd < latestMacd.signal) ? 1 : 0) >= 1
          ? "Negative"
          : "Neutral",
    volume: volumeStatus,
    rsi: latestRsi,
    rsiStatus,
    macdStatus,
    vwapStatus,
    price,
    sma20: latestSma20,
    sma50: latestSma50,
    vwap: latestVwap,
    support1: sr.support1,
    support2: sr.support2,
    resistance1: sr.resistance1,
    resistance2: sr.resistance2,
    signal,
    direction,
    confidence,
    riskLevel,
    setupQuality,
    entry,
    entryZone,
    stopLoss,
    target1,
    target2,
    riskReward,
    atr,
    atrPercent,
    dataQuality,
    volumeRatio,
    volumeConfirmation,
    actionTrigger,
    invalidation,
    scoreBreakdown,
    confirmationGates,
    decisionReasons: decisionReasons.slice(0, 8),
    notes: notes.slice(0, 5),
  };
}

function trendPointsFromSummary(
  sma20: number | null,
  sma50: number | null,
  price: number
): number {
  let points = 0;
  if (sma20 !== null) points += price > sma20 ? 1 : price < sma20 ? -1 : 0;
  if (sma50 !== null) points += price > sma50 ? 1 : price < sma50 ? -1 : 0;
  if (sma20 !== null && sma50 !== null) points += sma20 > sma50 ? 1 : sma20 < sma50 ? -1 : 0;
  return points;
}

function getTradingSessionKey(chartTime: number): string {
  const date = new Date((chartTime - IST_OFFSET_SECONDS) * 1000);
  return date.toLocaleDateString("en-CA", {
    timeZone: "Asia/Kolkata",
  });
}

function calculateVWAP(candles: RawCandle[]) {
  const result: { time: Time; value: number }[] = [];

  let sessionKey = "";
  let cumulativePriceVolume = 0;
  let cumulativeVolume = 0;
  let lastVWAP: number | null = null;

  for (const candle of candles) {
    const currentSession = getTradingSessionKey(candle.time);

    if (currentSession !== sessionKey) {
      sessionKey = currentSession;
      cumulativePriceVolume = 0;
      cumulativeVolume = 0;
      lastVWAP = null;
    }

    const volume = Number(candle.volume);
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;

    if (
      Number.isFinite(volume) &&
      volume > 0 &&
      Number.isFinite(typicalPrice)
    ) {
      cumulativePriceVolume += typicalPrice * volume;
      cumulativeVolume += volume;
      lastVWAP = cumulativePriceVolume / cumulativeVolume;
    }

    // Keep the line continuous through zero/missing-volume candles when
    // a valid VWAP already exists for the current trading session.
    if (lastVWAP !== null && Number.isFinite(lastVWAP)) {
      result.push({
        time: candle.time as Time,
        value: lastVWAP,
      });
    }
  }

  return result;
}

function calculateSupportResistance(candles: RawCandle[], pivotWindow = 5, lookback = 200) {
  const empty = {
    support1: null as number | null,
    support2: null as number | null,
    resistance1: null as number | null,
    resistance2: null as number | null,
  };
  if (candles.length < pivotWindow * 2 + 1) return empty;

  const recent = candles.slice(-lookback);
  const current = recent[recent.length - 1]?.close;
  if (!Number.isFinite(current)) return empty;

  const highs: number[] = [];
  const lows: number[] = [];

  for (let i = pivotWindow; i < recent.length - pivotWindow; i += 1) {
    const candle = recent[i];
    let isHigh = true;
    let isLow = true;
    for (let j = i - pivotWindow; j <= i + pivotWindow; j += 1) {
      if (j === i) continue;
      if (recent[j].high > candle.high) isHigh = false;
      if (recent[j].low < candle.low) isLow = false;
      if (!isHigh && !isLow) break;
    }
    if (isHigh) highs.push(candle.high);
    if (isLow) lows.push(candle.low);
  }

  const tolerance = Math.max(Math.abs(current) * 0.0025, 0.05);
  const cluster = (values: number[]) => {
    const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b);
    const groups: number[][] = [];
    for (const value of sorted) {
      const last = groups[groups.length - 1];
      if (last && Math.abs(value - last[last.length - 1]) <= tolerance) last.push(value);
      else groups.push([value]);
    }
    return groups.map((g) => g.reduce((sum, value) => sum + value, 0) / g.length);
  };

  const supports = cluster(lows).filter((level) => level < current).sort((a, b) => b - a);
  const resistances = cluster(highs).filter((level) => level > current).sort((a, b) => a - b);

  if (!supports.length) {
    const low = Math.min(...recent.map((c) => c.low));
    if (Number.isFinite(low) && low < current) supports.push(low);
  }
  if (!resistances.length) {
    const high = Math.max(...recent.map((c) => c.high));
    if (Number.isFinite(high) && high > current) resistances.push(high);
  }

  return {
    support1: supports[0] ?? null,
    support2: supports[1] ?? null,
    resistance1: resistances[0] ?? null,
    resistance2: resistances[1] ?? null,
  };
}

function mergeVolumes(
  candles: RawCandle[],
  volumes: RawVolume[]
): RawCandle[] {
  const volumeMap = new Map<number, number>();

  for (const item of normalizeVolumes(volumes)) {
    volumeMap.set(minuteKey(item.time), item.volume);
  }

  return candles.map((candle) => ({
    ...candle,
    volume: volumeMap.get(minuteKey(candle.time)),
  }));
}

function extractLivePrice(payload: any): number | null {
  const candidates = [
    payload?.ltpc?.ltp,
    payload?.ltpc?.last_price,
    payload?.data?.ltpc?.ltp,
    payload?.data?.ltpc?.last_price,
    payload?.last_price,
    payload?.ltp,
    payload?.data?.last_price,
    payload?.data?.ltp,
  ];

  for (const value of candidates) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) return number;
  }

  return null;
}

function extractLiveVolume(payload: any): number | null {
  const candidates = [
    payload?.ltpc?.volume,
    payload?.ltpc?.vtt,
    payload?.data?.ltpc?.volume,
    payload?.data?.ltpc?.vtt,
    payload?.volume,
    payload?.data?.volume,
  ];

  for (const value of candidates) {
    const number = Number(value);
    if (Number.isFinite(number) && number >= 0) return number;
  }

  return null;
}

function formatPrice(price: number | null): string {
  if (price === null || !Number.isFinite(price)) return "—";

  return `₹${price.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}



function formatDateForApi(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function getISTSessionDateFromUnix(seconds: number): string {
  const date = new Date((seconds + IST_OFFSET_SECONDS) * 1000);
  return date.toISOString().slice(0, 10);
}

function getISTTodayKey(): string {
  return getISTSessionDateFromUnix(Math.floor(Date.now() / 1000));
}

function isNSEMarketOpenNow(): boolean {
  const now = new Date();
  const ist = new Date(now.getTime() + IST_OFFSET_SECONDS * 1000);
  const day = ist.getUTCDay();
  if (day === 0 || day === 6) return false;

  const minutes = ist.getUTCHours() * 60 + ist.getUTCMinutes();
  return minutes >= 555 && minutes <= 930;
}

function getDateRange(days: number | null) {
  const to = new Date();

  if (days === null) {
    return { fromDate: undefined, toDate: formatDateForApi(to) };
  }

  const from = new Date(to);
  from.setDate(from.getDate() - days);

  return {
    fromDate: formatDateForApi(from),
    toDate: formatDateForApi(to),
  };
}

function normalizeInstrumentKey(value: string): string {
  return value.trim();
}

function isIndexInstrument(key: string): boolean {
  return key.startsWith("NSE_INDEX|") || key.startsWith("BSE_INDEX|");
}

export default function LiveMarketChart({
  instrumentKey,
  symbol = "Stock",
  name,
  currentPrice = null,
  currentChange = null,
  marketStatus = "NORMAL_OPEN",
  isIndex: isIndexProp,
  period: controlledPeriod,
  onPeriodChange,
  onTechnicalSummaryChange,
}: LiveMarketChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const sma20SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const sma50SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bbMiddleSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bbUpperSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bbLowerSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const vwapSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const support1SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const support2SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const resistance1SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const resistance2SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  // Live market price is rendered as a horizontal price line. This avoids
  // inventing a fake historical candle and keeps the historical data intact.
  const currentPriceLineCandleRef = useRef<any>(null);
  const currentPriceLineLineRef = useRef<any>(null);
  const rsiContainerRef = useRef<HTMLDivElement | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const macdContainerRef = useRef<HTMLDivElement | null>(null);
  const macdChartRef = useRef<IChartApi | null>(null);
  const macdLineRef = useRef<ISeriesApi<"Line"> | null>(null);
  const macdSignalRef = useRef<ISeriesApi<"Line"> | null>(null);
  const macdHistogramRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const candlesRef = useRef<RawCandle[]>([]);
  const indexVolumesRef = useRef<RawVolume[]>([]);
  const livePriceRef = useRef<number | null>(currentPrice);

  const [internalPeriod, setInternalPeriod] = useState<Period>("1D");
  const period = controlledPeriod ?? internalPeriod;
  const setPeriod = useCallback((next: Period) => {
    setInternalPeriod(next);
    onPeriodChange?.(next);
  }, [onPeriodChange]);
  const [chartMode, setChartMode] = useState<ChartMode>("candle");
  const [candles, setCandles] = useState<RawCandle[]>([]);
  const [indexVolumes, setIndexVolumes] = useState<RawVolume[]>([]);
  const [livePrice, setLivePrice] = useState<number | null>(currentPrice);
  const [liveChange, setLiveChange] = useState<number | null>(currentChange);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastTick, setLastTick] = useState<Date | null>(null);
  const [liveSessionActive, setLiveSessionActive] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [showSMA20, setShowSMA20] = useState(false);
  const [showSMA50, setShowSMA50] = useState(false);
  const [showRSI, setShowRSI] = useState(true);
  const [showMACD, setShowMACD] = useState(true);
  const [showBollinger, setShowBollinger] = useState(false);
  const [showVWAP, setShowVWAP] = useState(false);
  const [showSupportResistance, setShowSupportResistance] = useState(false);
  const [crosshairInfo, setCrosshairInfo] = useState<{
    x: number;
    y: number;
    time: number;
    price: number;
    volume: number | null;
    sma20: number | null;
    sma50: number | null;
    rsi14: number | null;
    macd: number | null;
    macdSignal: number | null;
    macdHistogram: number | null;
    bbMiddle: number | null;
    bbUpper: number | null;
    bbLower: number | null;
    vwap: number | null;
    support1: number | null;
    support2: number | null;
    resistance1: number | null;
    resistance2: number | null;
  } | null>(null);
  const [indexVolumeActive, setIndexVolumeActive] = useState(false);
  const [indexVolumeMatched, setIndexVolumeMatched] = useState<number | null>(null);

  const safeInstrumentKey = useMemo(
    () => normalizeInstrumentKey(instrumentKey),
    [instrumentKey]
  );

  const selectedConfig = useMemo(
    () => PERIODS.find((item) => item.label === period) ?? PERIODS[0],
    [period]
  );

  const indexInstrument = isIndexProp ?? isIndexInstrument(safeInstrumentKey);

  const loadIndexVolume = useCallback(async (): Promise<RawVolume[]> => {
    if (!indexInstrument) return [];

    const range = getDateRange(selectedConfig.days);
    const params = new URLSearchParams({
      instrument_key: safeInstrumentKey,
      unit: selectedConfig.unit,
      interval: selectedConfig.interval,
      to_date: range.toDate,
    });

    if (range.fromDate) params.set("from_date", range.fromDate);

    const response = await fetch(
      `${API_BASE}/api/indices/volume?${params.toString()}`
    );

    if (!response.ok) {
      throw new Error(`Index volume request failed: ${response.status}`);
    }

    const payload = await response.json();
    const volumes = extractIndexVolumes(payload);

    setIndexVolumeActive(payload?.source === "aggregated_constituent_volume");
    setIndexVolumeMatched(
      Number.isFinite(Number(payload?.matched_constituents))
        ? Number(payload.matched_constituents)
        : null
    );

    return volumes;
  }, [indexInstrument, safeInstrumentKey, selectedConfig]);

  const loadChartData = useCallback(async () => {
    if (!safeInstrumentKey) {
      setError("Instrument key is missing.");
      setLoading(false);
      setLiveSessionActive(false);
      return;
    }

    setLoading(true);
    setError(null);
    setLiveSessionActive(false);

    try {
      const params = new URLSearchParams({
        instrument_key: safeInstrumentKey,
        unit: selectedConfig.unit,
        interval: selectedConfig.interval,
      });

      const range = getDateRange(selectedConfig.days);
      if (range.fromDate) params.set("from_date", range.fromDate);
      params.set("to_date", range.toDate);

      let parsed: RawCandle[] = [];
      let showingLiveToday = false;

      if (period === "1D") {
        // 1D: use today's intraday session when available.
        const intradayEndpoint = indexInstrument
          ? "/api/indices/intraday"
          : "/api/stocks/intraday";

        const intradayResponse = await fetch(
          `${API_BASE}${intradayEndpoint}?${params.toString()}`
        );

        if (intradayResponse.ok) {
          parsed = extractCandles(await intradayResponse.json());
        }

        // When the market is closed (weekends/holidays/after the session),
        // Upstox may return no intraday candles. Keep the 1D chart useful by
        // falling back to the most recent completed trading session only.
        if (parsed.length === 0) {
          const fallbackFrom = new Date();
          fallbackFrom.setDate(fallbackFrom.getDate() - 7);

          const historyParams = new URLSearchParams({
            instrument_key: safeInstrumentKey,
            unit: "minutes",
            interval: "1",
            from_date: formatDateForApi(fallbackFrom),
            to_date: formatDateForApi(new Date()),
          });

          const historyEndpoint = indexInstrument
            ? "/api/indices/history"
            : "/api/stocks/history";

          const historyResponse = await fetch(
            `${API_BASE}${historyEndpoint}?${historyParams.toString()}`
          );

          if (historyResponse.ok) {
            const historical = extractCandles(await historyResponse.json());
            if (historical.length > 0) {
              const sessionDates = Array.from(
                new Set(
                  historical.map((candle) =>
                    getISTSessionDateFromUnix(candle.time)
                  )
                )
              ).sort();

              const latestSession = sessionDates[sessionDates.length - 1];
              parsed = historical.filter(
                (candle) =>
                  getISTSessionDateFromUnix(candle.time) === latestSession
              );
            }
          }
        } else {
          const latestSession = getISTSessionDateFromUnix(
            parsed[parsed.length - 1].time
          );
          showingLiveToday =
            latestSession === getISTTodayKey() && isNSEMarketOpenNow();
        }
      } else {
        const endpoint = indexInstrument
          ? "/api/indices/history"
          : "/api/stocks/history";

        const response = await fetch(`${API_BASE}${endpoint}?${params.toString()}`);

        if (!response.ok) {
          throw new Error(`${period} price request failed: ${response.status}`);
        }

        parsed = extractCandles(await response.json());
      }

      if (parsed.length === 0) {
        throw new Error(`No ${period} chart data was returned by Upstox.`);
      }

      parsed = sortAndDeduplicate(parsed);

      // Upstox timestamps are absolute UTC instants. Shift the chart
      // display coordinate so NSE/BSE market time is shown on the axis.
      parsed = parsed.map((candle) => ({
        ...candle,
        time: toISTChartTime(candle.time),
      }));

      if (indexInstrument) {
        const volumes = await loadIndexVolume();
        const shiftedVolumes = normalizeVolumes(volumes).map((item) => ({
          ...item,
          time: toISTChartTime(item.time),
        }));

        indexVolumesRef.current = shiftedVolumes;
        setIndexVolumes(shiftedVolumes);
        parsed = mergeVolumes(parsed, shiftedVolumes);
      } else {
        indexVolumesRef.current = [];
        setIndexVolumes([]);
        setIndexVolumeActive(false);
        setIndexVolumeMatched(null);
      }

      candlesRef.current = parsed;
      setCandles(parsed);
      setLiveSessionActive(period === "1D" && showingLiveToday);

      const last = parsed[parsed.length - 1];
      if (last) {
        // For 1D, the websocket/current quote should win when available.
        // For historical periods, keep the historical candles untouched and
        // use the parent's current quote as the live reference price.
        const validPropPrice =
          Number.isFinite(currentPrice) && currentPrice !== null && currentPrice > 0
            ? currentPrice
            : null;

        const nextDisplayPrice =
          period === "1D"
            ? (showingLiveToday ? last.close : validPropPrice ?? last.close)
            : (validPropPrice ?? last.close);

        livePriceRef.current = nextDisplayPrice;
        setLivePrice(nextDisplayPrice);

        if (period === "1D" && showingLiveToday) {
          const first = parsed[0];
          if (first.open > 0) {
            setLiveChange(((last.close - first.open) / first.open) * 100);
          }
        } else if (validPropPrice !== null) {
          // For historical ranges, the percentage change displayed beside
          // the live quote should remain the parent-provided market change.
          setLiveChange(
            Number.isFinite(currentChange) && currentChange !== null
              ? currentChange
              : null
          );
        } else {
          const first = parsed[0];
          if (first.open > 0) {
            setLiveChange(((last.close - first.open) / first.open) * 100);
          }
        }
      }

      setLoading(false);
    } catch (err) {
      console.error("Chart data error:", err);
      setError(
        err instanceof Error
          ? err.message
          : `Unable to load ${period} chart data.`
      );
      setLoading(false);
      setLiveSessionActive(false);
    }
  }, [
    safeInstrumentKey,
    period,
    selectedConfig,
    indexInstrument,
    loadIndexVolume,
    currentPrice,
    currentChange,
  ]);

  useEffect(() => {
    loadChartData();
  }, [loadChartData]);

  // Refresh index constituent volume periodically during the 1D view.
  useEffect(() => {
    if (!indexInstrument || period !== "1D") return;

    const timer = window.setInterval(async () => {
      try {
        const volumes = await loadIndexVolume();
        if (!volumes.length) return;

        const shifted = normalizeVolumes(volumes).map((item) => ({
          ...item,
          time: toISTChartTime(item.time),
        }));

        indexVolumesRef.current = shifted;
        setIndexVolumes(shifted);

        const merged = mergeVolumes(candlesRef.current, shifted);
        candlesRef.current = merged;
        setCandles(merged);
      } catch (err) {
        console.warn("Index volume refresh failed:", err);
      }
    }, 30000);

    return () => window.clearInterval(timer);
  }, [indexInstrument, period, loadIndexVolume]);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: 560,
      layout: {
        background: { type: ColorType.Solid, color: "#071022" },
        textColor: "#8fa4c7",
      },
      grid: {
        vertLines: { color: "#172642" },
        horzLines: { color: "#172642" },
      },
      crosshair: {
        vertLine: { color: "#52698f", width: 1, style: 2 },
        horzLine: { color: "#52698f", width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: "#233452",
        scaleMargins: { top: 0.06, bottom: 0.30 },
      },
      timeScale: {
        borderColor: "#233452",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        barSpacing: 8,
        minBarSpacing: 2,
        tickMarkFormatter: (time: Time) => formatISTTickFromChartTime(Number(time)),
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: true,
        axisDoubleClickReset: true,
      },
      localization: {
        priceFormatter: (price: number) =>
          `₹${price.toLocaleString("en-IN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}`,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    const lineSeries = chart.addSeries(LineSeries, {
      color: "#22d3ee",
      lineWidth: 2,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      priceLineVisible: true,
      lastValueVisible: true,
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      lastValueVisible: false,
      priceLineVisible: false,
    });

    const sma20Series = chart.addSeries(LineSeries, {
      color: "#f59e0b",
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    const sma50Series = chart.addSeries(LineSeries, {
      color: "#a855f7",
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    const bbUpperSeries = chart.addSeries(LineSeries, {
      color: "#38bdf8",
      lineWidth: 1,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    const bbMiddleSeries = chart.addSeries(LineSeries, {
      color: "#94a3b8",
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    const bbLowerSeries = chart.addSeries(LineSeries, {
      color: "#38bdf8",
      lineWidth: 1,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    const vwapSeries = chart.addSeries(LineSeries, {
      color: "#eab308",
      lineWidth: 2,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 3,
      priceLineVisible: false,
      lastValueVisible: true,
      visible: false,
    });

    const support1Series = chart.addSeries(LineSeries, { color: "#22c55e", lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: true, visible: false });
    const support2Series = chart.addSeries(LineSeries, { color: "#16a34a", lineWidth: 1, lineStyle: 3, crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: true, visible: false });
    const resistance1Series = chart.addSeries(LineSeries, { color: "#ef4444", lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: true, visible: false });
    const resistance2Series = chart.addSeries(LineSeries, { color: "#dc2626", lineWidth: 1, lineStyle: 3, crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: true, visible: false });

    // Reserve a real bottom pane for volume. The volume series has its own
    // price scale, so it remains visible for both stocks and indices.
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.74, bottom: 0.02 },
      borderVisible: false,
      autoScale: true,
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    lineSeriesRef.current = lineSeries;
    volumeSeriesRef.current = volumeSeries;
    sma20SeriesRef.current = sma20Series;
    sma50SeriesRef.current = sma50Series;
    bbUpperSeriesRef.current = bbUpperSeries;
    bbMiddleSeriesRef.current = bbMiddleSeries;
    bbLowerSeriesRef.current = bbLowerSeries;
    vwapSeriesRef.current = vwapSeries;
    support1SeriesRef.current = support1Series;
    support2SeriesRef.current = support2Series;
    resistance1SeriesRef.current = resistance1Series;
    resistance2SeriesRef.current = resistance2Series;

    const handleCrosshairMove = (param: any) => {
      if (!containerRef.current || !param?.point || param.time == null) {
        setCrosshairInfo(null);
        return;
      }

      if (
        param.point.x < 0 ||
        param.point.y < 0 ||
        param.point.x > containerRef.current.clientWidth ||
        param.point.y > containerRef.current.clientHeight
      ) {
        setCrosshairInfo(null);
        return;
      }

      const candleData = param.seriesData?.get(candleSeries);
      const lineData = param.seriesData?.get(lineSeries);
      const volumeData = param.seriesData?.get(volumeSeries);
      const sma20Data = param.seriesData?.get(sma20Series);
      const sma50Data = param.seriesData?.get(sma50Series);
      const bbUpperData = param.seriesData?.get(bbUpperSeries);
      const bbMiddleData = param.seriesData?.get(bbMiddleSeries);
      const bbLowerData = param.seriesData?.get(bbLowerSeries);
      const vwapData = param.seriesData?.get(vwapSeries);
      const support1Data = param.seriesData?.get(support1Series);
      const support2Data = param.seriesData?.get(support2Series);
      const resistance1Data = param.seriesData?.get(resistance1Series);
      const resistance2Data = param.seriesData?.get(resistance2Series);

      let rsi14: number | null = null;
      const rsiData = calculateRSI(candlesRef.current, 14);
      const matchedRSI = rsiData.find((item) => Number(item.time) === Number(param.time));
      if (matchedRSI) rsi14 = matchedRSI.value;

      let macd: number | null = null;
      let macdSignal: number | null = null;
      let macdHistogram: number | null = null;
      let bbMiddle: number | null = Number.isFinite(Number(bbMiddleData?.value)) ? Number(bbMiddleData?.value) : null;
      let bbUpper: number | null = Number.isFinite(Number(bbUpperData?.value)) ? Number(bbUpperData?.value) : null;
      let bbLower: number | null = Number.isFinite(Number(bbLowerData?.value)) ? Number(bbLowerData?.value) : null;
      const macdData = calculateMACD(candlesRef.current, 12, 26, 9);
      const sr = calculateSupportResistance(candlesRef.current);
      const matchedMACD = macdData.find((item) => Number(item.time) === Number(param.time));
      if (matchedMACD) {
        macd = matchedMACD.macd;
        macdSignal = matchedMACD.signal;
        macdHistogram = matchedMACD.histogram;
      }

      let price = Number.NaN;
      if (chartMode === "line") {
        price = Number(lineData?.value);
      } else {
        price = Number(candleData?.close);
      }

      if (!Number.isFinite(price)) {
        price = Number(candleData?.close ?? lineData?.value);
      }

      if (!Number.isFinite(price)) {
        setCrosshairInfo(null);
        return;
      }

      let volume = Number(volumeData?.value);

      // Lightweight Charts only supplies seriesData when the series has an
      // exact data point at the crosshair time. For index volume, keep the
      // tooltip useful even if the API/candle timestamps differ by seconds.
      if (!Number.isFinite(volume)) {
        const crosshairMinute = minuteKey(Number(param.time));
        const nearest = indexInstrument
          ? indexVolumes.find((item) => minuteKey(item.time) === crosshairMinute)
          : candlesRef.current.find((item) => minuteKey(item.time) === crosshairMinute);
        volume = Number(nearest?.volume);
      }

      setCrosshairInfo({
        x: param.point.x,
        y: param.point.y,
        time: Number(param.time),
        price,
        volume: Number.isFinite(volume) ? volume : null,
        sma20: Number.isFinite(Number(sma20Data?.value)) ? Number(sma20Data.value) : null,
        sma50: Number.isFinite(Number(sma50Data?.value)) ? Number(sma50Data.value) : null,
        rsi14,
        macd,
        macdSignal,
        macdHistogram,
        bbMiddle,
        bbUpper,
        bbLower,
        vwap: Number.isFinite(Number(vwapData?.value)) ? Number(vwapData.value) : null,
        support1: Number.isFinite(Number(support1Data?.value)) ? Number(support1Data.value) : sr.support1,
        support2: Number.isFinite(Number(support2Data?.value)) ? Number(support2Data.value) : sr.support2,
        resistance1: Number.isFinite(Number(resistance1Data?.value)) ? Number(resistance1Data.value) : sr.resistance1,
        resistance2: Number.isFinite(Number(resistance2Data?.value)) ? Number(resistance2Data.value) : sr.resistance2,
      });
    };

    chart.subscribeCrosshairMove(handleCrosshairMove);

    const resizeObserver = new ResizeObserver(() => {
      if (!containerRef.current || !chartRef.current) return;
      chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
    });

    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.unsubscribeCrosshairMove(handleCrosshairMove);
      if (currentPriceLineCandleRef.current) {
        try {
          candleSeries.removePriceLine(currentPriceLineCandleRef.current);
        } catch {}
        currentPriceLineCandleRef.current = null;
      }
      if (currentPriceLineLineRef.current) {
        try {
          lineSeries.removePriceLine(currentPriceLineLineRef.current);
        } catch {}
        currentPriceLineLineRef.current = null;
      }

      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      lineSeriesRef.current = null;
      volumeSeriesRef.current = null;
      sma20SeriesRef.current = null;
      sma50SeriesRef.current = null;
      vwapSeriesRef.current = null;
      support1SeriesRef.current = null;
      support2SeriesRef.current = null;
      resistance1SeriesRef.current = null;
      resistance2SeriesRef.current = null;
    };
  }, [chartMode]);

  useEffect(() => {
    if (!chartRef.current || !candleSeriesRef.current || !lineSeriesRef.current) return;

    const candleData: Candle[] = candles.map((candle) => ({
      time: candle.time as Time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));

    const lineData = candles.map((candle) => ({
      time: candle.time as Time,
      value: candle.close,
    }));

    // For indices, render the API's aggregated constituent volume directly
    // against its own timestamps. This is deliberately independent from the
    // candle volume field so a timestamp mismatch cannot make the histogram
    // disappear. Stocks continue to use the volume embedded in their candles.
    const volumeData = indexInstrument
      ? indexVolumes.map((item) => {
          const candle = candles.find(
            (candidate) => minuteKey(candidate.time) === minuteKey(item.time)
          );

          return {
            time: item.time as Time,
            value: item.volume,
            color:
              candle && candle.close < candle.open
                ? "rgba(239,68,68,0.70)"
                : "rgba(34,197,94,0.70)",
          };
        })
      : candles
          .filter(
            (candle) =>
              candle.volume !== undefined &&
              Number.isFinite(candle.volume) &&
              candle.volume >= 0
          )
          .map((candle) => ({
            time: candle.time as Time,
            value: candle.volume as number,
            color:
              candle.close >= candle.open
                ? "rgba(34,197,94,0.70)"
                : "rgba(239,68,68,0.70)",
          }));

    const sma20Data = calculateSMA(candles, 20);
    const sma50Data = calculateSMA(candles, 50);
    const bbData = calculateBollingerBands(candles, 20, 2);
    const vwapData = calculateVWAP(candles);
    const sr = calculateSupportResistance(candles);
    const firstTime = candles[0]?.time as Time | undefined;
    const lastTime = candles[candles.length - 1]?.time as Time | undefined;
    const horizontalData = (level: number | null) =>
      level !== null && firstTime !== undefined && lastTime !== undefined
        ? [{ time: firstTime, value: level }, { time: lastTime, value: level }]
        : [];

    candleSeriesRef.current.setData(candleData);
    lineSeriesRef.current.setData(lineData);
    volumeSeriesRef.current?.setData(volumeData);

    // Show the actual current market price inside the chart without adding a
    // synthetic candle. Historical OHLC data therefore remains untouched.
    if (currentPriceLineCandleRef.current) {
      candleSeriesRef.current.removePriceLine(currentPriceLineCandleRef.current);
      currentPriceLineCandleRef.current = null;
    }
    if (currentPriceLineLineRef.current) {
      lineSeriesRef.current.removePriceLine(currentPriceLineLineRef.current);
      currentPriceLineLineRef.current = null;
    }

    const chartLivePrice =
      Number.isFinite(livePrice) && livePrice !== null && livePrice > 0
        ? livePrice
        : Number.isFinite(currentPrice) && currentPrice !== null && currentPrice > 0
          ? currentPrice
          : null;

    if (chartLivePrice !== null) {
      const priceLineOptions = {
        price: chartLivePrice,
        color: "#12cdea",
        lineWidth: 2 as const,
        lineStyle: 2 as const,
        axisLabelVisible: true,
        title: "LIVE",
      };

      if (chartMode === "candle") {
        currentPriceLineCandleRef.current =
          candleSeriesRef.current.createPriceLine(priceLineOptions);
      } else {
        currentPriceLineLineRef.current =
          lineSeriesRef.current.createPriceLine(priceLineOptions);
      }
    }
    sma20SeriesRef.current?.setData(sma20Data);
    sma50SeriesRef.current?.setData(sma50Data);
    bbMiddleSeriesRef.current?.setData(bbData.map((d) => ({ time: d.time, value: d.middle })));
    bbUpperSeriesRef.current?.setData(bbData.map((d) => ({ time: d.time, value: d.upper })));
    bbLowerSeriesRef.current?.setData(bbData.map((d) => ({ time: d.time, value: d.lower })));
    vwapSeriesRef.current?.setData(vwapData);
    support1SeriesRef.current?.setData(horizontalData(sr.support1));
    support2SeriesRef.current?.setData(horizontalData(sr.support2));
    resistance1SeriesRef.current?.setData(horizontalData(sr.resistance1));
    resistance2SeriesRef.current?.setData(horizontalData(sr.resistance2));

    candleSeriesRef.current.applyOptions({ visible: chartMode === "candle" });
    lineSeriesRef.current.applyOptions({ visible: chartMode === "line" });
    sma20SeriesRef.current?.applyOptions({ visible: showSMA20 });
    sma50SeriesRef.current?.applyOptions({ visible: showSMA50 });
    bbMiddleSeriesRef.current?.applyOptions({ visible: showBollinger });
    bbUpperSeriesRef.current?.applyOptions({ visible: showBollinger });
    bbLowerSeriesRef.current?.applyOptions({ visible: showBollinger });
    vwapSeriesRef.current?.applyOptions({ visible: showVWAP });
    support1SeriesRef.current?.applyOptions({ visible: showSupportResistance && sr.support1 !== null });
    support2SeriesRef.current?.applyOptions({ visible: showSupportResistance && sr.support2 !== null });
    resistance1SeriesRef.current?.applyOptions({ visible: showSupportResistance && sr.resistance1 !== null });
    resistance2SeriesRef.current?.applyOptions({ visible: showSupportResistance && sr.resistance2 !== null });

    if (candles.length > 0) chartRef.current.timeScale().fitContent();
  }, [
    candles,
    chartMode,
    currentPrice,
    livePrice,
    indexInstrument,
    indexVolumes,
    showSMA20,
    showSMA50,
    showBollinger,
    showVWAP,
    showSupportResistance,
  ]);

  useEffect(() => {
    if (!onTechnicalSummaryChange) return;
    onTechnicalSummaryChange(
      calculateTechnicalSummary(candles, indexInstrument, indexVolumes, livePrice ?? currentPrice)
    );
  }, [candles, indexInstrument, indexVolumes, livePrice, currentPrice, onTechnicalSummaryChange]);

  useEffect(() => {
    if (!rsiContainerRef.current || !showRSI) {
      if (rsiChartRef.current) {
        rsiChartRef.current.remove();
        rsiChartRef.current = null;
        rsiSeriesRef.current = null;
      }
      return;
    }

    const container = rsiContainerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 150,
      layout: {
        background: { type: ColorType.Solid, color: "#071022" },
        textColor: "#7186aa",
      },
      grid: {
        vertLines: { color: "#172642" },
        horzLines: { color: "#172642" },
      },
      rightPriceScale: {
        borderColor: "#233452",
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        visible: true,
        borderColor: "#233452",
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: Time) => formatISTTickFromChartTime(Number(time)),
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true, axisDoubleClickReset: true },
      localization: { priceFormatter: (value: number) => value.toFixed(0) },
    });

    const rsiSeries = chart.addSeries(LineSeries, {
      color: "#22d3ee",
      lineWidth: 2,
      crosshairMarkerVisible: true,
      priceLineVisible: false,
      lastValueVisible: true,
      autoscaleInfoProvider: (original: any) => {
        const info = original();
        return info;
      },
    });

    rsiSeries.createPriceLine({
      price: 70,
      color: "rgba(239,68,68,0.55)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: "70",
    });
    rsiSeries.createPriceLine({
      price: 30,
      color: "rgba(34,197,94,0.55)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: "30",
    });

    rsiSeriesRef.current = rsiSeries;
    rsiChartRef.current = chart;

    const resizeObserver = new ResizeObserver(() => {
      if (rsiContainerRef.current && rsiChartRef.current) {
        rsiChartRef.current.applyOptions({ width: rsiContainerRef.current.clientWidth });
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      rsiChartRef.current = null;
      rsiSeriesRef.current = null;
    };
  }, [showRSI]);

  useEffect(() => {
    if (!rsiChartRef.current || !rsiSeriesRef.current || !showRSI) return;
    const data = calculateRSI(candles, 14);
    rsiSeriesRef.current.setData(data);
    const chart = rsiChartRef.current;
    const start = data[0]?.time;
    const end = data[data.length - 1]?.time;
    if (start !== undefined && end !== undefined) {
      chart.timeScale().setVisibleRange({ from: start as Time, to: end as Time });
    }
  }, [candles, showRSI]);

  useEffect(() => {
    if (!macdContainerRef.current || !showMACD) {
      if (macdChartRef.current) macdChartRef.current.remove();
      macdChartRef.current = null;
      macdLineRef.current = null;
      macdSignalRef.current = null;
      macdHistogramRef.current = null;
      return;
    }
    const container = macdContainerRef.current;
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 180,
      layout: { background: { type: ColorType.Solid, color: "#071022" }, textColor: "#7186aa" },
      grid: { vertLines: { color: "#172642" }, horzLines: { color: "#172642" } },
      rightPriceScale: { borderColor: "#233452", scaleMargins: { top: 0.08, bottom: 0.08 } },
      timeScale: { visible: true, borderColor: "#233452", timeVisible: true, secondsVisible: false, tickMarkFormatter: (time: Time) => formatISTTickFromChartTime(Number(time)) },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true, axisDoubleClickReset: true },
      localization: { priceFormatter: (value: number) => value.toFixed(2) },
    });
    const histogram = chart.addSeries(HistogramSeries, { priceFormat: { type: "price", precision: 2, minMove: 0.01 }, priceLineVisible: false, lastValueVisible: false, base: 0 });
    const macdLine = chart.addSeries(LineSeries, { color: "#22d3ee", lineWidth: 2, crosshairMarkerVisible: true, priceLineVisible: false, lastValueVisible: true });
    const signalLine = chart.addSeries(LineSeries, { color: "#f59e0b", lineWidth: 2, crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: true });
    macdLine.createPriceLine({ price: 0, color: "rgba(148,163,184,0.55)", lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "0" });
    macdLineRef.current = macdLine;
    macdSignalRef.current = signalLine;
    macdHistogramRef.current = histogram;
    macdChartRef.current = chart;
    const resizeObserver = new ResizeObserver(() => {
      if (macdContainerRef.current && macdChartRef.current) macdChartRef.current.applyOptions({ width: macdContainerRef.current.clientWidth });
    });
    resizeObserver.observe(container);
    return () => { resizeObserver.disconnect(); chart.remove(); macdChartRef.current = null; macdLineRef.current = null; macdSignalRef.current = null; macdHistogramRef.current = null; };
  }, [showMACD]);

  useEffect(() => {
    if (!macdChartRef.current || !macdLineRef.current || !macdSignalRef.current || !macdHistogramRef.current || !showMACD) return;
    const data = calculateMACD(candles, 12, 26, 9);
    macdLineRef.current.setData(data.map((d) => ({ time: d.time, value: d.macd })));
    macdSignalRef.current.setData(data.map((d) => ({ time: d.time, value: d.signal })));
    macdHistogramRef.current.setData(data.map((d) => ({ time: d.time, value: d.histogram, color: d.histogram >= 0 ? "#22c55e" : "#ef4444" })));
    const start = data[0]?.time; const end = data[data.length - 1]?.time;
    if (start !== undefined && end !== undefined) macdChartRef.current.timeScale().setVisibleRange({ from: start as Time, to: end as Time });
  }, [candles, showMACD]);

  useEffect(() => {
    if (!safeInstrumentKey || period !== "1D" || !liveSessionActive) {
      setConnected(false);
      return;
    }

    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let stopped = false;

    const connect = () => {
      if (stopped) return;

      const encoded = encodeURIComponent(safeInstrumentKey);
      socket = new WebSocket(
        `ws://127.0.0.1:8000/ws/market?instrument_key=${encoded}`
      );

      socket.onopen = () => setConnected(true);

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const price = extractLivePrice(payload);
          if (price === null) return;

          const now = new Date();
          livePriceRef.current = price;
          setLivePrice(price);
          setLastTick(now);

          const first = candlesRef.current[0];
          if (first?.open > 0) {
            setLiveChange(((price - first.open) / first.open) * 100);
          }

          const minuteTimestamp = toISTChartTime(
            Math.floor(now.getTime() / 60000) * 60
          );

          const existing = candlesRef.current[candlesRef.current.length - 1];
          if (!existing) return;

          const tickVolume = extractLiveVolume(payload);

          if (existing.time === minuteTimestamp) {
            const updated: RawCandle = {
              ...existing,
              high: Math.max(existing.high, price),
              low: Math.min(existing.low, price),
              close: price,
              volume: tickVolume ?? existing.volume,
            };

            candlesRef.current = [...candlesRef.current.slice(0, -1), updated];
            setCandles([...candlesRef.current]);
          } else if (minuteTimestamp > existing.time) {
            const newCandle: RawCandle = {
              time: minuteTimestamp,
              open: price,
              high: price,
              low: price,
              close: price,
              volume: tickVolume ?? undefined,
            };

            candlesRef.current = [...candlesRef.current, newCandle];
            setCandles([...candlesRef.current]);
          }
        } catch (error) {
          console.error("Live WebSocket parse error:", error);
        }
      };

      socket.onerror = () => setConnected(false);

      socket.onclose = () => {
        setConnected(false);
        if (!stopped) reconnectTimer = window.setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      stopped = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close();
      setConnected(false);
    };
  }, [safeInstrumentKey, period, liveSessionActive]);

  const zoom = useCallback((factor: number) => {
    const chart = chartRef.current;
    if (!chart) return;

    const currentRange = chart.timeScale().getVisibleLogicalRange();
    if (!currentRange) return;

    const center = (currentRange.from + currentRange.to) / 2;
    const currentWidth = currentRange.to - currentRange.from;
    const maxWidth = Math.max(8, candlesRef.current.length);
    const newWidth = Math.max(8, Math.min(maxWidth, currentWidth * factor));

    const newRange: LogicalRange = {
      from: (center - newWidth / 2) as Logical,
      to: (center + newWidth / 2) as Logical,
    };

    chart.timeScale().setVisibleLogicalRange(newRange);
    setZoomLevel(
      Math.round(
        Math.max(10, Math.min(1000, (candlesRef.current.length / newWidth) * 100))
      )
    );
  }, []);

  const displayPrice =
    period === "1D"
      ? (livePrice ?? currentPrice)
      : (Number.isFinite(currentPrice) && currentPrice !== null && currentPrice > 0
          ? currentPrice
          : livePrice);

  const displayChange =
    period === "1D"
      ? (liveChange ?? currentChange)
      : (currentChange ?? liveChange);

  // Keep the chart's live reference synchronized with the quote supplied by
  // StockDetails. Historical candles are never replaced by this update.
  useEffect(() => {
    if (!Number.isFinite(currentPrice) || currentPrice === null || currentPrice <= 0) return;

    if (period !== "1D" || !connected) {
      livePriceRef.current = currentPrice;
      setLivePrice(currentPrice);
      if (Number.isFinite(currentChange) && currentChange !== null) {
        setLiveChange(currentChange);
      }
    }
  }, [currentPrice, currentChange, period, connected]);
  const decisionSummary = useMemo(
    () => calculateTechnicalSummary(candles, indexInstrument, indexVolumes, displayPrice),
    [candles, indexInstrument, indexVolumes, displayPrice]
  );
  const marketOpen =
    marketStatus === "NORMAL_OPEN" ||
    marketStatus === "NORMAL_PREOPEN" ||
    marketStatus === "PRE_OPEN";
  const livePeriod = period === "1D";

  const rangeDescription: Record<Period, string> = {
    "1D": "Today's intraday market movement",
    "1W": "Recent 8 calendar days of 15-minute history",
    "1M": "Recent 35 calendar days of daily history",
    "3M": "Recent 100 calendar days of daily history",
    "6M": "Recent 190 calendar days of daily history",
    "1Y": "Recent 380 calendar days of daily history",
    "3Y": "Recent 3+ years of weekly history",
    "5Y": "Recent 5+ years of weekly history",
    MAX: "Maximum available history",
  };

  return (
    <div
      style={{
        width: "100%",
        background: "#071022",
        border: "1px solid #1c2c49",
        borderRadius: "18px",
        padding: "22px",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "20px",
          flexWrap: "wrap",
          marginBottom: "15px",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0, color: "#f5f7fb", fontSize: "22px", fontWeight: 700 }}>
              Live Market Chart
            </h2>
            <span
              style={{
                padding: "6px 10px",
                borderRadius: "20px",
                background: marketOpen ? "rgba(34,197,94,0.12)" : "rgba(148,163,184,0.12)",
                color: marketOpen ? "#22c55e" : "#94a3b8",
                fontSize: "12px",
                fontWeight: 600,
              }}
            >
              ● {marketOpen ? "Market Open" : "Market Closed"}
            </span>
            {livePeriod && connected && (
              <span style={{ color: "#22c55e", fontSize: "12px", fontWeight: 600 }}>● Live</span>
            )}
          </div>

          <div style={{ marginTop: "5px", color: "#7186aa", fontSize: "14px" }}>
            {symbol}{name ? ` · ${name}` : ""}
          </div>
        </div>

        <div style={{ display: "flex", gap: "6px", background: "#101d34", borderRadius: "10px", padding: "4px" }}>
          {(["line", "candle"] as ChartMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setChartMode(mode)}
              style={{
                border: 0,
                borderRadius: "7px",
                padding: "9px 15px",
                cursor: "pointer",
                background: chartMode === mode ? "#12cdea" : "transparent",
                color: chartMode === mode ? "#03121d" : "#91a5c6",
                fontWeight: 600,
              }}
            >
              {mode === "line" ? "Line" : "Candles"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap", marginBottom: "12px" }}>
        <span style={{ color: "#7186aa", fontSize: "12px", fontWeight: 600 }}>Indicators:</span>
        <button
          type="button"
          onClick={() => setShowSMA20((value) => !value)}
          style={{
            border: `1px solid ${showSMA20 ? "#f59e0b" : "#24395d"}`,
            borderRadius: "8px",
            padding: "7px 11px",
            cursor: "pointer",
            background: showSMA20 ? "rgba(245,158,11,0.14)" : "#14233d",
            color: showSMA20 ? "#f59e0b" : "#91a5c6",
            fontWeight: 700,
            fontSize: "12px",
          }}
        >
          SMA 20
        </button>
        <button
          type="button"
          onClick={() => setShowSMA50((value) => !value)}
          style={{
            border: `1px solid ${showSMA50 ? "#a855f7" : "#24395d"}`,
            borderRadius: "8px",
            padding: "7px 11px",
            cursor: "pointer",
            background: showSMA50 ? "rgba(168,85,247,0.14)" : "#14233d",
            color: showSMA50 ? "#c084fc" : "#91a5c6",
            fontWeight: 700,
            fontSize: "12px",
          }}
        >
          SMA 50
        </button>
        <button
          type="button"
          onClick={() => setShowRSI((value) => !value)}
          style={{
            border: `1px solid ${showRSI ? "#22d3ee" : "#24395d"}`,
            borderRadius: "8px",
            padding: "7px 11px",
            cursor: "pointer",
            background: showRSI ? "rgba(34,211,238,0.14)" : "#14233d",
            color: showRSI ? "#22d3ee" : "#91a5c6",
            fontWeight: 700,
            fontSize: "12px",
          }}
        >
          RSI 14
        </button>
        <button type="button" onClick={() => setShowMACD((v) => !v)} style={{ border: `1px solid ${showMACD ? "#22d3ee" : "#24395d"}`, borderRadius: "8px", padding: "7px 11px", cursor: "pointer", background: showMACD ? "rgba(34,211,238,0.14)" : "#14233d", color: showMACD ? "#22d3ee" : "#91a5c6", fontWeight: 700, fontSize: "12px" }}>
          MACD
        </button>
        <button type="button" onClick={() => setShowBollinger((v) => !v)} style={{ border: `1px solid ${showBollinger ? "#38bdf8" : "#24395d"}`, borderRadius: "8px", padding: "7px 11px", cursor: "pointer", background: showBollinger ? "rgba(56,189,248,0.14)" : "#14233d", color: showBollinger ? "#38bdf8" : "#91a5c6", fontWeight: 700, fontSize: "12px" }}>
          Bollinger Bands
        </button>
        <button type="button" onClick={() => setShowVWAP((v) => !v)} style={{ border: `1px solid ${showVWAP ? "#eab308" : "#24395d"}`, borderRadius: "8px", padding: "7px 11px", cursor: "pointer", background: showVWAP ? "rgba(234,179,8,0.14)" : "#14233d", color: showVWAP ? "#eab308" : "#91a5c6", fontWeight: 700, fontSize: "12px" }}>
          VWAP
        </button>
        <button type="button" onClick={() => setShowSupportResistance((v) => !v)} style={{ border: `1px solid ${showSupportResistance ? "#ef4444" : "#24395d"}`, borderRadius: "8px", padding: "7px 11px", cursor: "pointer", background: showSupportResistance ? "rgba(239,68,68,0.14)" : "#14233d", color: showSupportResistance ? "#fca5a5" : "#91a5c6", fontWeight: 700, fontSize: "12px" }}>
          Support / Resistance
        </button>
        <span style={{ color: "#607698", fontSize: "12px" }}>
          SMA 20 = short-term · SMA 50 = medium-term · RSI 14 = momentum · MACD = trend/momentum · BB = volatility · VWAP = volume-weighted average price · S/R = automatic support and resistance
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "baseline", gap: "14px", flexWrap: "wrap", marginBottom: "13px" }}>
        <span style={{ color: "#ffffff", fontSize: "34px", fontWeight: 700 }}>
          {formatPrice(displayPrice)}
        </span>
        {displayChange !== null && Number.isFinite(displayChange) && (
          <span style={{ color: displayChange >= 0 ? "#22c55e" : "#ef4444", fontSize: "17px", fontWeight: 600 }}>
            {displayChange >= 0 ? "+" : ""}{displayChange.toFixed(2)}%
          </span>
        )}
        {lastTick && livePeriod && (
          <span style={{ color: "#7186aa", fontSize: "12px" }}>
            Last tick {lastTick.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "7px", flexWrap: "wrap", marginBottom: "12px" }}>
        {PERIODS.map((item) => {
          const active = period === item.label;
          return (
            <button
              key={item.label}
              type="button"
              onClick={() => setPeriod(item.label)}
              title={rangeDescription[item.label]}
              style={{
                border: "1px solid #24395d",
                borderRadius: "8px",
                padding: "8px 12px",
                cursor: "pointer",
                background: active ? "#12cdea" : "#14233d",
                color: active ? "#03121d" : "#91a5c6",
                fontWeight: 700,
                fontSize: "12px",
              }}
            >
              {item.label}
            </button>
          );
        })}
        <span style={{ marginLeft: "4px", color: "#607698", fontSize: "12px" }}>
          {rangeDescription[period]}
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "7px", marginBottom: "10px", flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={() => zoom(0.7)}
          title="Zoom in"
          style={{ background: "#14233d", border: "1px solid #24395d", color: "#dbe7fa", borderRadius: "8px", width: "42px", height: "38px", cursor: "pointer", fontSize: "20px" }}
        >+</button>
        <button
          type="button"
          onClick={() => zoom(1.4)}
          title="Zoom out"
          style={{ background: "#14233d", border: "1px solid #24395d", color: "#dbe7fa", borderRadius: "8px", width: "42px", height: "38px", cursor: "pointer", fontSize: "20px" }}
        >−</button>
        <button
          type="button"
          onClick={() => {
            chartRef.current?.timeScale().fitContent();
            setZoomLevel(100);
          }}
          title="Fit entire chart"
          style={{ background: "#14233d", border: "1px solid #24395d", color: "#dbe7fa", borderRadius: "8px", padding: "0 14px", height: "38px", cursor: "pointer", fontWeight: 600 }}
        >Fit</button>
        <span style={{ marginLeft: "5px", color: "#7186aa", fontSize: "12px" }}>Zoom {zoomLevel}%</span>
        <span style={{ marginLeft: "auto", color: "#607698", fontSize: "12px" }}>Scroll to zoom · Drag to pan</span>
      </div>

      {indexInstrument && indexVolumeActive && (
        <div style={{ marginBottom: "9px", color: "#6f89ad", fontSize: "12px" }}>
          Volume = aggregated traded volume of {indexVolumeMatched ?? "index"} matched constituents
        </div>
      )}

      <div style={{ position: "relative", width: "100%", minHeight: "560px" }}>
        {loading && (
          <div style={{ position: "absolute", inset: 0, zIndex: 5, display: "flex", justifyContent: "center", alignItems: "center", color: "#8fa4c7", background: "rgba(7,16,34,0.8)", borderRadius: "12px" }}>
            Loading {period} market data...
          </div>
        )}

        {error && !loading && (
          <div style={{ position: "absolute", inset: 0, zIndex: 5, display: "flex", justifyContent: "center", alignItems: "center", flexDirection: "column", gap: "10px", color: "#ef4444", background: "#071022", borderRadius: "12px" }}>
            <strong>Unable to load {period} chart</strong>
            <span style={{ color: "#7186aa", fontSize: "13px", textAlign: "center", maxWidth: "600px" }}>{error}</span>
            <button type="button" onClick={loadChartData} style={{ marginTop: "5px", border: 0, borderRadius: "8px", padding: "9px 16px", background: "#12cdea", color: "#03121d", cursor: "pointer", fontWeight: 600 }}>
              Retry
            </button>
          </div>
        )}

        {/* Visual separation between the price pane and the volume pane.
            Both remain inside the same Lightweight Charts canvas and share
            exactly the same time axis/crosshair. */}
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: "74%",
            borderTop: "1px solid #233452",
            zIndex: 3,
            pointerEvents: "none",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: 10,
            top: "8px",
            zIndex: 3,
            pointerEvents: "none",
            color: "#52698f",
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.08em",
          }}
        >
          PRICE
        </div>
        <div
          style={{
            position: "absolute",
            left: 10,
            top: "76%",
            zIndex: 3,
            pointerEvents: "none",
            color: "#52698f",
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.08em",
          }}
        >
          VOLUME
        </div>

        {crosshairInfo && (
          <div
            style={{
              position: "absolute",
              left: Math.min(Math.max(crosshairInfo.x + 14, 8), Math.max(8, (containerRef.current?.clientWidth ?? 600) - 190)),
              top: Math.min(Math.max(crosshairInfo.y - 75, 8), 450),
              zIndex: 10,
              pointerEvents: "none",
              minWidth: "165px",
              padding: "9px 11px",
              background: "#101d34",
              border: "1px solid #2a3d60",
              borderRadius: "8px",
              boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
              color: "#dbe7fa",
              fontSize: "12px",
              lineHeight: 1.45,
            }}
          >
            <div style={{ color: "#f5f7fb", fontWeight: 700, marginBottom: "3px" }}>
              {formatISTDateTimeFromChartTime(crosshairInfo.time)}
            </div>
            <div style={{ color: "#12cdea", fontWeight: 600 }}>
              Price: {formatPrice(crosshairInfo.price)}
            </div>
            {showSMA20 && crosshairInfo.sma20 !== null && (
              <div style={{ color: "#f59e0b", fontWeight: 600, marginTop: "2px" }}>
                SMA 20: {formatPrice(crosshairInfo.sma20)}
              </div>
            )}
            {showSMA50 && crosshairInfo.sma50 !== null && (
              <div style={{ color: "#c084fc", fontWeight: 600, marginTop: "2px" }}>
                SMA 50: {formatPrice(crosshairInfo.sma50)}
              </div>
            )}
            {showBollinger && crosshairInfo.bbMiddle !== null && (
              <>
                <div style={{ color: "#38bdf8", fontWeight: 600, marginTop: "2px" }}>
                  BB Upper: {formatPrice(crosshairInfo.bbUpper ?? crosshairInfo.bbMiddle)}
                </div>
                <div style={{ color: "#94a3b8", fontWeight: 600, marginTop: "2px" }}>
                  BB Middle: {formatPrice(crosshairInfo.bbMiddle)}
                </div>
                <div style={{ color: "#38bdf8", fontWeight: 600, marginTop: "2px" }}>
                  BB Lower: {formatPrice(crosshairInfo.bbLower ?? crosshairInfo.bbMiddle)}
                </div>
              </>
            )}
            {showVWAP && crosshairInfo.vwap !== null && (
              <div style={{ color: "#eab308", fontWeight: 600, marginTop: "2px" }}>
                VWAP: {formatPrice(crosshairInfo.vwap)}
              </div>
            )}
            {showSupportResistance && (
              <>
                {crosshairInfo.support1 !== null && <div style={{ color: "#22c55e", fontWeight: 600, marginTop: "2px" }}>Support 1: {formatPrice(crosshairInfo.support1)}</div>}
                {crosshairInfo.support2 !== null && <div style={{ color: "#16a34a", fontWeight: 600, marginTop: "2px" }}>Support 2: {formatPrice(crosshairInfo.support2)}</div>}
                {crosshairInfo.resistance1 !== null && <div style={{ color: "#ef4444", fontWeight: 600, marginTop: "2px" }}>Resistance 1: {formatPrice(crosshairInfo.resistance1)}</div>}
                {crosshairInfo.resistance2 !== null && <div style={{ color: "#dc2626", fontWeight: 600, marginTop: "2px" }}>Resistance 2: {formatPrice(crosshairInfo.resistance2)}</div>}
              </>
            )}
            {crosshairInfo.volume !== null && (
              <div style={{ color: "#94a3b8", fontWeight: 600, marginTop: "2px" }}>
                Volume: {crosshairInfo.volume.toLocaleString("en-IN")}
              </div>
            )}
          </div>
        )}

        <div ref={containerRef} style={{ width: "100%", height: "560px" }} />

        {showRSI && (
          <div style={{ marginTop: "8px", borderTop: "1px solid #233452", paddingTop: "8px" }}>
            <div style={{ color: "#52698f", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", margin: "0 0 4px 10px" }}>
              RSI (14) · 30 OVERSOLD · 70 OVERBOUGHT
            </div>
            <div ref={rsiContainerRef} style={{ width: "100%", height: "150px" }} />
          </div>
        )}

        {showMACD && (
          <div style={{ marginTop: "8px", borderTop: "1px solid #233452", paddingTop: "8px" }}>
            <div style={{ color: "#52698f", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", margin: "0 0 4px 10px" }}>MACD (12, 26, 9) · CYAN MACD · ORANGE SIGNAL · HISTOGRAM</div>
            <div ref={macdContainerRef} style={{ width: "100%", height: "180px" }} />
          </div>
        )}

      </div>

      {decisionSummary && (
        <div
          style={{
            marginTop: "14px",
            padding: "16px",
            border: "1px solid #1c2c49",
            borderRadius: "14px",
            background: "#081329",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "14px", flexWrap: "wrap" }}>
            <div>
              <div style={{ color: "#607698", fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em" }}>
                STOCKAI DECISION ENGINE
              </div>
              <div style={{ color: "#f5f7fb", fontSize: "19px", fontWeight: 700, marginTop: "4px" }}>
                Technical trade setup
              </div>
              <div style={{ color: "#7186aa", fontSize: "11px", marginTop: "3px" }}>
                Rule-based analysis of the loaded OHLC, completed volume and technical indicators
              </div>
            </div>

            <div
              style={{
                minWidth: "128px",
                padding: "9px 13px",
                borderRadius: "10px",
                textAlign: "center",
                background:
                  decisionSummary.signal === "BUY"
                    ? "rgba(34,197,94,0.12)"
                    : decisionSummary.signal === "SELL"
                      ? "rgba(239,68,68,0.12)"
                      : "rgba(234,179,8,0.12)",
                border: `1px solid ${
                  decisionSummary.signal === "BUY"
                    ? "rgba(34,197,94,0.35)"
                    : decisionSummary.signal === "SELL"
                      ? "rgba(239,68,68,0.35)"
                      : "rgba(234,179,8,0.35)"
                }`,
              }}
            >
              <div style={{
                color: decisionSummary.signal === "BUY" ? "#22c55e" : decisionSummary.signal === "SELL" ? "#ef4444" : "#eab308",
                fontSize: "21px",
                fontWeight: 800,
              }}>
                {decisionSummary.signal}
              </div>
              <div style={{ color: "#91a5c6", fontSize: "11px", marginTop: "2px" }}>
                Confidence {decisionSummary.confidence}%
              </div>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(135px, 1fr))", gap: "8px", marginTop: "13px" }}>
            {[
              ["Signal Score", `${decisionSummary.score}/100`],
              ["Market Bias", decisionSummary.bias],
              ["Risk Level", decisionSummary.riskLevel],
              ["Setup Quality", decisionSummary.setupQuality],
              ["Direction", decisionSummary.direction],
              ["Data Quality", decisionSummary.dataQuality],
              ["ATR (14)", decisionSummary.atr !== null ? `${formatPrice(decisionSummary.atr)} (${decisionSummary.atrPercent?.toFixed(2) ?? "—"}%)` : "—"],
              ["Volume", decisionSummary.volumeRatio !== null ? `${decisionSummary.volumeRatio.toFixed(2)}x · ${decisionSummary.volumeConfirmation}` : decisionSummary.volume],
            ].map(([label, value]) => (
              <div key={label} style={{ padding: "10px 11px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
                <div style={{ color: "#607698", fontSize: "10px", marginBottom: "4px" }}>{label}</div>
                <div style={{ color: "#dbe7fa", fontSize: "13px", fontWeight: 700 }}>{value}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: "8px", marginTop: "8px" }}>
            {[
              ["Entry", formatPrice(decisionSummary.entry)],
              ["Entry Zone", decisionSummary.entryZone ?? "Wait for confirmation"],
              ["Stop Loss", formatPrice(decisionSummary.stopLoss)],
              ["Target 1", formatPrice(decisionSummary.target1)],
              ["Target 2", formatPrice(decisionSummary.target2)],
              ["Risk / Reward", decisionSummary.riskReward !== null ? `${decisionSummary.riskReward}:1` : "—"],
            ].map(([label, value]) => (
              <div key={label} style={{ padding: "10px 11px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
                <div style={{ color: "#607698", fontSize: "10px", marginBottom: "4px" }}>{label}</div>
                <div style={{ color: "#dbe7fa", fontSize: "13px", fontWeight: 700 }}>{value}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "8px", marginTop: "10px" }}>
            <div style={{ padding: "11px 12px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
              <div style={{ color: "#607698", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", marginBottom: "6px" }}>ACTION / TRIGGER</div>
              <div style={{ color: "#b7c6dd", fontSize: "12px", lineHeight: 1.5 }}>{decisionSummary.actionTrigger}</div>
            </div>
            <div style={{ padding: "11px 12px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
              <div style={{ color: "#607698", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", marginBottom: "6px" }}>INVALIDATION</div>
              <div style={{ color: "#f1b8b8", fontSize: "12px", lineHeight: 1.5 }}>{decisionSummary.invalidation}</div>
            </div>
          </div>

          <div style={{ marginTop: "10px", padding: "11px 12px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
            <div style={{ color: "#607698", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", marginBottom: "6px" }}>
              WHY THIS SIGNAL
            </div>
            {decisionSummary.decisionReasons.map((reason, index) => (
              <div key={`${reason}-${index}`} style={{ color: "#b7c6dd", fontSize: "12px", lineHeight: 1.5, marginTop: index ? "3px" : 0 }}>
                • {reason}
              </div>
            ))}
          </div>

          <div style={{ marginTop: "10px", padding: "11px 12px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
            <div style={{ color: "#607698", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", marginBottom: "8px" }}>
              CONFIRMATION GATES
            </div>
            <div style={{ display: "grid", gap: "6px" }}>
              {decisionSummary.confirmationGates.map((gate) => (
                <div key={gate.label} style={{ display: "grid", gridTemplateColumns: "24px 150px 1fr", gap: "8px", alignItems: "center", fontSize: "11px" }}>
                  <span style={{ color: gate.passed ? "#22c55e" : "#ef4444", fontWeight: 800 }}>{gate.passed ? "✓" : "✕"}</span>
                  <span style={{ color: "#91a5c6", fontWeight: 600 }}>{gate.label}</span>
                  <span style={{ color: gate.passed ? "#b7c6dd" : "#f1b8b8" }}>{gate.detail}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: "10px", padding: "11px 12px", borderRadius: "9px", background: "#0b1830", border: "1px solid #1a2b48" }}>
            <div style={{ color: "#607698", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", marginBottom: "8px" }}>
              SIGNAL SCORE BREAKDOWN
            </div>
            <div style={{ display: "grid", gap: "6px" }}>
              {decisionSummary.scoreBreakdown.map((item) => (
                <div key={item.label} style={{ display: "grid", gridTemplateColumns: "100px 52px 1fr", gap: "8px", alignItems: "start", fontSize: "11px" }}>
                  <span style={{ color: "#91a5c6" }}>{item.label}</span>
                  <span style={{ color: item.score > 0 ? "#22c55e" : item.score < 0 ? "#ef4444" : "#eab308", fontWeight: 700 }}>{item.score > 0 ? "+" : ""}{item.score}</span>
                  <span style={{ color: "#7186aa" }}>{item.reason}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: "10px", color: "#607698", fontSize: "10px", lineHeight: 1.5 }}>
            Confidence measures technical-signal agreement and data completeness; it is not a probability of profit. Entry, stop-loss and target levels are reference estimates derived from observed market structure and ATR, not guaranteed prices or financial advice.
          </div>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "space-between", gap: "10px", flexWrap: "wrap", marginTop: "12px", color: "#607698", fontSize: "12px" }}>
        <span>{candles.length.toLocaleString("en-IN")} candles</span>
        <span>
          {chartMode === "candle" ? "OHLC Candlestick" : "Close Price Line"} · {period}
        </span>
        <span>
          {indexInstrument && indexVolumeActive
            ? `Upstox · ${indexVolumeMatched ?? "constituent"} volume${indexVolumes.length ? ` · ${indexVolumes.length} bars` : ""}`
            : livePeriod && connected
              ? "Upstox · Live"
              : "Upstox · Historical"}
        </span>
      </div>
    </div>
  );
}
