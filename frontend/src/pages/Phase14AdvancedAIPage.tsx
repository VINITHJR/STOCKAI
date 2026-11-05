import React, { useMemo, useState } from "react";

/* ============================================================
   PHASE 14 — ADVANCED AI
   FRONTEND
   ============================================================

   IMPORTANT ARCHITECTURE

   Frontend sends ONLY:

       {
         instrument_key: "NSE_EQ|..."
       }

   Backend calculates:
     Phase 14.1 Technical Intelligence
     Phase 14.2 Stock Regime Intelligence
     Phase 14.3 Fundamental Intelligence
     Phase 14 Layers 1 -> 7

   The frontend DOES NOT calculate or override AI values.
   ============================================================ */

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

/* ============================================================
   TYPES
   ============================================================ */

type TechnicalIntelligence = {
  signal?: string | null;
  confidence?: number | null;
  directional_score?: number | null;

  bullish_votes?: number | null;
  bearish_votes?: number | null;
  neutral_votes?: number | null;

  current_price?: number | null;
  rsi_14?: number | null;
  ema_20?: number | null;
  ema_50?: number | null;
  ema_200?: number | null;
  macd?: number | null;
  macd_signal?: number | null;
  macd_histogram?: number | null;
  adx_14?: number | null;
  plus_di?: number | null;
  plus_di_14?: number | null;
  minus_di?: number | null;
  minus_di_14?: number | null;
  bollinger_upper?: number | null;
  bollinger_middle?: number | null;
  bollinger_lower?: number | null;
  bollinger_position?: number | null;
  atr_14?: number | null;
  atr_percent?: number | null;
  momentum_20d_percent?: number | null;
  volume_ratio_20d?: number | null;
  volume_ratio?: number | null;
  candle_count?: number | null;

  indicators?: {
    current_price?: number | null;
    rsi_14?: number | null;
    ema_20?: number | null;
    ema_50?: number | null;
    ema_200?: number | null;
    macd?: number | null;
    macd_signal?: number | null;
    macd_histogram?: number | null;
    adx_14?: number | null;
    plus_di?: number | null;
    plus_di_14?: number | null;
    minus_di?: number | null;
    minus_di_14?: number | null;
    bollinger_upper?: number | null;
    bollinger_middle?: number | null;
    bollinger_lower?: number | null;
    bollinger_position?: number | null;
    atr_14?: number | null;
    atr_percent?: number | null;
    momentum_20d_percent?: number | null;
    volume_ratio_20d?: number | null;
    volume_ratio?: number | null;
    [key: string]: any;
  } | null;

  explanation?: string | null;
  method?: string | null;
};

type StockRegimeIntelligence = {
  stock_regime?: string | null;
  stock_direction?: string | null;
  stock_regime_confidence?: number | null;

  trend_strength?: number | null;
  momentum_regime?: string | null;
  volatility_regime?: string | null;
  price_structure?: string | null;

  current_price?: number | null;

  ema_20?: number | null;
  ema_50?: number | null;
  ema_200?: number | null;

  momentum_20d_percent?: number | null;
  atr_percent?: number | null;
  adx?: number | null;
  adx_14?: number | null;
  plus_di?: number | null;
  minus_di?: number | null;

  candle_count?: number | null;

  evidence?: string[] | null;
  method?: string | null;
};

type FundamentalIntelligence = {
  signal?: string | null;
  confidence?: number | null;

  fundamental_score?: number | null;

  valuation_score?: number | null;
  profitability_score?: number | null;
  financial_health_score?: number | null;
  shareholder_return_score?: number | null;

  valuation_assessment?: string | null;
  profitability_assessment?: string | null;
  financial_health_assessment?: string | null;
  shareholder_return_assessment?: string | null;

  metric_count?: number | null;
  available_metric_count?: number | null;
  total_metric_count?: number | null;

  completeness?: number | null;
  completeness_percent?: number | null;

  evidence?: string[] | null;
  warnings?: string[] | null;

  method?: string | null;
};

type Phase14Context = {
  technical_intelligence?: TechnicalIntelligence | null;
  stock_regime_intelligence?: StockRegimeIntelligence | null;
  fundamental_intelligence?: FundamentalIntelligence | null;

  [key: string]: any;
};

type Phase14SourceDetail = {
  signal?: string | null;
  confidence?: number | null;
  decision_weight?: number | null;
  role?: string | null;
  stock_regime?: string | null;
  stock_direction?: string | null;
  error?: string | null;
  candle_count?: number | null;
  method?: string | null;
  [key: string]: any;
};

type Phase14InputSource = {
  instrument_key?: string | null;
  symbol?: string | null;
  sector?: string | null;

  context?: Phase14Context | null;

  technical_signal?: string | null;
  technical_confidence?: number | null;

  stock_regime?: string | null;
  stock_regime_confidence?: number | null;

  fundamental_signal?: string | null;
  fundamental_context_confidence?: number | null;

  phase14_1?: Phase14SourceDetail | null;
  phase14_2?: Phase14SourceDetail | null;

  [key: string]: any;
};

type Phase14Result = {
  instrument_key?: string | null;

  decision?: string | null;
  decision_confidence?: number | null;

  recommendation?: string | null;
  recommendation_confidence?: number | null;

  recommendation_quality?: string | null;
  recommendation_strength?: string | null;
  recommendation_risk?: string | null;

  reliability_score?: number | null;
  confidence_grade?: string | null;
  confidence_band?: string | null;
  reliability_level?: string | null;

  market_context?: string | null;
  market_context_score?: number | null;

  sector_context_score?: number | null;
  overall_context_score?: number | null;

  market_alignment?: string | null;
  sector_alignment?: string | null;
  overall_alignment?: string | null;

  context_strength?: string | null;
  context_risk?: string | null;

  market_regime?: string | null;
  market_direction?: string | null;

  sector?: string | null;
  sector_direction?: string | null;

  validation_status?: string | null;
  validation_score?: number | null;
  validation_quality?: string | null;

  recommendation_valid?: boolean | null;
  recommendation_trust?: string | null;
  validation_risk?: string | null;

  decision_summary?: string | null;
  confidence_explanation?: string | null;
  reliability_explanation?: string | null;
  final_explanation?: string | null;

  strongest_factor?: string | null;
  strongest_risk?: string | null;

  bullish_factors?: string[] | null;
  bearish_factors?: string[] | null;
  neutral_factors?: string[] | null;

  supporting_factors?: string[] | null;
  conflicting_factors?: string[] | null;

  warnings?: string[] | null;

  action_guidance?: string | null;
  position_guidance?: string | null;

  model_count?: number | null;
  agreement_count?: number | null;
  conflict?: boolean | null;
  conflict_level?: string | null;

  layers_completed?: number[] | null;

  layer_status?: Record<string, string> | null;

  overall_status?: string | null;
  status?: string | null;

  phase14_input_source?: Phase14InputSource | null;

  metadata?: {
    phase?: number;
    layer?: number;
    component?: string;
    layers_included?: number[];
    recommendation_source?: string;
    validation_source?: string;
    aggregation_only?: boolean;
  } | null;

  [key: string]: any;
};

type ApiResponse = {
  status?: string;
  phase?: number;
  instrument_key?: string;
  result?: Phase14Result;
  detail?: any;
};

type StockSearchResult = {
  instrument_key?: string;
  instrumentKey?: string;
  instrument_token?: string;
  key?: string;

  symbol?: string;
  trading_symbol?: string;
  tradingSymbol?: string;

  name?: string;
  company_name?: string;
  companyName?: string;

  sector?: string;

  [key: string]: any;
};

/* ============================================================
   HELPER FUNCTIONS
   ============================================================ */

function formatPercent(value?: number | null): string {
  if (
    value === undefined ||
    value === null ||
    !Number.isFinite(Number(value))
  ) {
    return "--";
  }

  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatPercentFromNumber(value?: number | null): string {
  if (
    value === undefined ||
    value === null ||
    !Number.isFinite(Number(value))
  ) {
    return "--";
  }

  return `${Number(value).toFixed(2)}%`;
}

function formatNumber(
  value?: number | null,
  decimals = 2,
): string {
  if (
    value === undefined ||
    value === null ||
    !Number.isFinite(Number(value))
  ) {
    return "--";
  }

  return Number(value).toFixed(decimals);
}

function formatBoolean(value?: boolean | null): string {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "--";
}

function labelize(value?: string | null): string {
  if (!value) return "--";

  return value
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase(),
    );
}

function getSignalTone(
  value?: string | null,
): "bullish" | "bearish" | "neutral" {
  const normalized = String(value || "").toLowerCase();

  if (
    normalized.includes("bullish") ||
    normalized.includes("buy")
  ) {
    return "bullish";
  }

  if (
    normalized.includes("bearish") ||
    normalized.includes("sell")
  ) {
    return "bearish";
  }

  return "neutral";
}

function signalTextClass(value?: string | null): string {
  const tone = getSignalTone(value);

  if (tone === "bullish") {
    return "text-emerald-400";
  }

  if (tone === "bearish") {
    return "text-red-400";
  }

  return "text-slate-300";
}

function signalBgClass(value?: string | null): string {
  const tone = getSignalTone(value);

  if (tone === "bullish") {
    return "border-emerald-500/30 bg-emerald-500/10";
  }

  if (tone === "bearish") {
    return "border-red-500/30 bg-red-500/10";
  }

  return "border-slate-500/30 bg-slate-500/10";
}

function riskClass(value?: string | null): string {
  const normalized = String(value || "").toLowerCase();

  if (
    normalized.includes("very_high") ||
    normalized.includes("high")
  ) {
    return "text-red-400";
  }

  if (
    normalized.includes("medium") ||
    normalized.includes("moderate") ||
    normalized.includes("uncertain")
  ) {
    return "text-amber-400";
  }

  if (
    normalized.includes("low") ||
    normalized.includes("strong") ||
    normalized.includes("valid")
  ) {
    return "text-emerald-400";
  }

  return "text-slate-300";
}

function validationClass(
  value?: string | null,
): string {
  const normalized = String(value || "").toLowerCase();

  if (normalized.includes("valid")) {
    return "text-emerald-400";
  }

  if (
    normalized.includes("invalid") ||
    normalized.includes("fail")
  ) {
    return "text-red-400";
  }

  return "text-amber-400";
}

function normalizeArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(
      (item): item is string =>
        typeof item === "string",
    )
    .map((item) => item.trim())
    .filter(Boolean);
}

function extractSearchResults(
  json: any,
): StockSearchResult[] {
  if (Array.isArray(json)) {
    return json;
  }

  if (Array.isArray(json?.results)) {
    return json.results;
  }

  if (Array.isArray(json?.data)) {
    return json.data;
  }

  if (Array.isArray(json?.stocks)) {
    return json.stocks;
  }

  return [];
}

function getInstrumentKey(
  stock: StockSearchResult,
): string {
  return (
    stock.instrument_key ||
    stock.instrumentKey ||
    stock.instrument_token ||
    stock.key ||
    ""
  );
}

function getStockSymbol(
  stock: StockSearchResult,
): string {
  return (
    stock.trading_symbol ||
    stock.tradingSymbol ||
    stock.symbol ||
    stock.name ||
    stock.company_name ||
    stock.companyName ||
    ""
  );
}

function getStockName(
  stock: StockSearchResult,
): string {
  return (
    stock.name ||
    stock.company_name ||
    stock.companyName ||
    getStockSymbol(stock)
  );
}

/* ============================================================
   REUSABLE UI COMPONENTS
   ============================================================ */

function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border border-[#1e293b] bg-[#0b1224] p-5 shadow-sm ${className}`}
    >
      <div className="mb-5">
        <h2 className="text-base font-semibold text-white">
          {title}
        </h2>

        {subtitle && (
          <p className="mt-1 text-xs text-slate-400">
            {subtitle}
          </p>
        )}
      </div>

      {children}
    </section>
  );
}

function MetricCard({
  label,
  value,
  subValue,
  tone = "neutral",
}: {
  label: string;
  value: string;
  subValue?: string;
  tone?: string | null;
}) {
  return (
    <div className="rounded-xl border border-[#1e293b] bg-[#080f20] p-4">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
        {label}
      </div>

      <div
        className={`mt-2 text-xl font-bold ${signalTextClass(
          tone,
        )}`}
      >
        {value}
      </div>

      {subValue && (
        <div className="mt-1 text-xs text-slate-500">
          {subValue}
        </div>
      )}
    </div>
  );
}

function DataRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-[#172238] py-2.5 last:border-b-0">
      <span className="text-sm text-slate-400">
        {label}
      </span>

      <span className="text-right text-sm font-medium text-slate-200">
        {value}
      </span>
    </div>
  );
}

function FactorList({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[];
  tone: "bullish" | "bearish" | "neutral";
}) {
  const dotClass =
    tone === "bullish"
      ? "bg-emerald-400"
      : tone === "bearish"
        ? "bg-red-400"
        : "bg-slate-400";

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-slate-200">
        {title}
      </h3>

      {items.length === 0 ? (
        <div className="rounded-lg border border-[#1e293b] bg-[#080f20] p-3 text-sm text-slate-500">
          No data available.
        </div>
      ) : (
        <div className="space-y-2">
          {items.map((item, index) => (
            <div
              key={`${item}-${index}`}
              className="flex gap-3 rounded-lg border border-[#1e293b] bg-[#080f20] p-3"
            >
              <span
                className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${dotClass}`}
              />

              <span className="text-sm leading-5 text-slate-300">
                {item}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   PHASE 14.1 — TECHNICAL INTELLIGENCE
   ============================================================ */

function TechnicalIntelligenceCard({
  technical,
}: {
  technical: TechnicalIntelligence | null;
}) {
  if (!technical) {
    return (
      <Card
        title="Phase 14.1 — Technical Intelligence"
        subtitle="Independent OHLCV-based technical analysis"
      >
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-300">
          Technical intelligence was not returned by
          the backend for this analysis.
        </div>
      </Card>
    );
  }

  // Phase 14.1 returns detailed indicators under technical.indicators.
  // Normalize them here so the UI never reads the wrong nesting level.
  const indicators = technical.indicators || {};
  const currentPrice =
    indicators.current_price ?? technical.current_price;
  const ema20 = indicators.ema_20 ?? technical.ema_20;
  const ema50 = indicators.ema_50 ?? technical.ema_50;
  const ema200 = indicators.ema_200 ?? technical.ema_200;
  const rsi14 = indicators.rsi_14 ?? technical.rsi_14;
  const macd = indicators.macd ?? technical.macd;
  const macdSignal = indicators.macd_signal ?? technical.macd_signal;
  const macdHistogram = indicators.macd_histogram ?? technical.macd_histogram;
  const adx14 = indicators.adx_14 ?? technical.adx_14;
  const plusDI =
    indicators.plus_di ?? indicators.plus_di_14 ?? technical.plus_di ?? technical.plus_di_14;
  const minusDI =
    indicators.minus_di ?? indicators.minus_di_14 ?? technical.minus_di ?? technical.minus_di_14;
  const bollingerUpper = indicators.bollinger_upper ?? technical.bollinger_upper;
  const bollingerMiddle = indicators.bollinger_middle ?? technical.bollinger_middle;
  const bollingerLower = indicators.bollinger_lower ?? technical.bollinger_lower;
  const bollingerPosition = indicators.bollinger_position ?? technical.bollinger_position;
  const atr14 = indicators.atr_14 ?? technical.atr_14;
  const atrPercent = indicators.atr_percent ?? technical.atr_percent;
  const momentum20d = indicators.momentum_20d_percent ?? technical.momentum_20d_percent;
  const volumeRatio =
    indicators.volume_ratio_20d ?? indicators.volume_ratio ?? technical.volume_ratio_20d ?? technical.volume_ratio;

  return (
    <Card
      title="Phase 14.1 — Technical Intelligence"
      subtitle="Independent deterministic OHLCV technical analysis"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Signal"
          value={labelize(technical.signal)}
          subValue={`Confidence ${formatPercent(
            technical.confidence,
          )}`}
          tone={technical.signal}
        />

        <MetricCard
          label="Confidence"
          value={formatPercent(
            technical.confidence,
          )}
          subValue={`Directional ${formatNumber(
            technical.directional_score,
          )}`}
        />

        <MetricCard
          label="Directional Score"
          value={formatNumber(
            technical.directional_score,
          )}
        />

        <MetricCard
          label="Candles"
          value={formatNumber(
            technical.candle_count,
            0,
          )}
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Trend & Momentum
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="EMA 20"
              value={formatNumber(
                ema20,
              )}
            />

            <DataRow
              label="EMA 50"
              value={formatNumber(
                ema50,
              )}
            />

            <DataRow
              label="EMA 200"
              value={formatNumber(
                ema200,
              )}
            />

            <DataRow
              label="RSI 14"
              value={formatNumber(
                rsi14,
              )}
            />

            <DataRow
              label="20D Momentum"
              value={formatPercentFromNumber(
                momentum20d,
              )}
            />

            <DataRow
              label="Volume Ratio"
              value={
                volumeRatio !==
                undefined &&
                volumeRatio !== null
                  ? `${formatNumber(
                      volumeRatio,
                    )}x`
                  : "--"
              }
            />
          </div>
        </div>

        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            MACD & Trend Strength
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="MACD"
              value={formatNumber(
                macd,
              )}
            />

            <DataRow
              label="MACD Signal"
              value={formatNumber(
                macdSignal,
              )}
            />

            <DataRow
              label="MACD Histogram"
              value={formatNumber(
                macdHistogram,
              )}
            />

            <DataRow
              label="ADX 14"
              value={formatNumber(
                adx14,
              )}
            />

            <DataRow
              label="+DI 14"
              value={formatNumber(
                plusDI,
              )}
            />

            <DataRow
              label="-DI 14"
              value={formatNumber(
                minusDI,
              )}
            />
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Volatility
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="ATR 14"
              value={formatNumber(
                atr14,
              )}
            />

            <DataRow
              label="ATR %"
              value={formatPercentFromNumber(
                atrPercent,
              )}
            />

            <DataRow
              label="Bollinger Upper"
              value={formatNumber(
                bollingerUpper,
              )}
            />

            <DataRow
              label="Bollinger Middle"
              value={formatNumber(
                bollingerMiddle,
              )}
            />

            <DataRow
              label="Bollinger Lower"
              value={formatNumber(
                bollingerLower,
              )}
            />

            <DataRow
              label="Bollinger Position"
              value={formatNumber(
                bollingerPosition,
              )}
            />

            <DataRow
              label="Current Price"
              value={formatNumber(
                currentPrice,
              )}
            />
          </div>
        </div>

        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Voting
          </h3>

          <div className="grid grid-cols-3 gap-3">
            <MetricCard
              label="Bullish Votes"
              value={formatNumber(
                technical.bullish_votes,
                0,
              )}
              tone="bullish"
            />

            <MetricCard
              label="Bearish Votes"
              value={formatNumber(
                technical.bearish_votes,
                0,
              )}
              tone="bearish"
            />

            <MetricCard
              label="Neutral Votes"
              value={formatNumber(
                technical.neutral_votes,
                0,
              )}
              tone="neutral"
            />
          </div>
        </div>
      </div>

      {technical.explanation && (
        <div className="mt-6 rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-cyan-300">
            Technical Explanation
          </div>

          <p className="mt-2 text-sm leading-6 text-slate-300">
            {technical.explanation}
          </p>
        </div>
      )}

      {technical.method && (
        <div className="mt-3 text-xs text-slate-500">
          Method: {technical.method}
        </div>
      )}
    </Card>
  );
}

/* ============================================================
   PHASE 14.2 — STOCK REGIME INTELLIGENCE
   ============================================================ */

function StockRegimeCard({
  regime,
}: {
  regime: StockRegimeIntelligence | null;
}) {
  if (!regime) {
    return (
      <Card
        title="Phase 14.2 — Stock Regime Intelligence"
        subtitle="Independent selected-stock regime analysis"
      >
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-300">
          Stock regime intelligence was not returned
          by the backend for this analysis.
        </div>
      </Card>
    );
  }

  const evidence = normalizeArray(
    regime.evidence,
  );

  return (
    <Card
      title="Phase 14.2 — Stock Regime Intelligence"
      subtitle="Independent OHLCV-based stock regime classification"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Stock Regime"
          value={labelize(regime.stock_regime)}
          subValue={labelize(
            regime.stock_direction,
          )}
          tone={regime.stock_direction}
        />

        <MetricCard
          label="Regime Confidence"
          value={formatPercent(
            regime.stock_regime_confidence,
          )}
        />

        <MetricCard
          label="Trend Strength"
          value={formatPercent(
            regime.trend_strength,
          )}
        />

        <MetricCard
          label="Current Price"
          value={formatNumber(
            regime.current_price,
          )}
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Regime Classification
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="Stock Direction"
              value={labelize(
                regime.stock_direction,
              )}
            />

            <DataRow
              label="Momentum Regime"
              value={labelize(
                regime.momentum_regime,
              )}
            />

            <DataRow
              label="Volatility Regime"
              value={labelize(
                regime.volatility_regime,
              )}
            />

            <DataRow
              label="Price Structure"
              value={labelize(
                regime.price_structure,
              )}
            />

            <DataRow
              label="20D Momentum"
              value={formatPercentFromNumber(
                regime.momentum_20d_percent,
              )}
            />

            <DataRow
              label="ATR %"
              value={formatPercentFromNumber(
                regime.atr_percent,
              )}
            />
          </div>
        </div>

        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Trend Indicators
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="EMA 20"
              value={formatNumber(
                regime.ema_20,
              )}
            />

            <DataRow
              label="EMA 50"
              value={formatNumber(
                regime.ema_50,
              )}
            />

            <DataRow
              label="EMA 200"
              value={formatNumber(
                regime.ema_200,
              )}
            />

            <DataRow
              label="ADX"
              value={formatNumber(
                regime.adx_14 ?? regime.adx,
              )}
            />

            <DataRow
              label="+DI"
              value={formatNumber(
                regime.plus_di,
              )}
            />

            <DataRow
              label="-DI"
              value={formatNumber(
                regime.minus_di,
              )}
            />
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Evidence
          </h3>

          {evidence.length === 0 ? (
            <div className="rounded-xl border border-[#1e293b] bg-[#080f20] p-4 text-sm text-slate-500">
              No regime evidence returned.
            </div>
          ) : (
            <div className="space-y-2">
              {evidence.map((item, index) => (
                <div
                  key={`${item}-${index}`}
                  className="rounded-lg border border-[#1e293b] bg-[#080f20] p-3 text-sm text-slate-300"
                >
                  {item}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-xl border border-[#1e293b] bg-[#080f20] p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Data Coverage
          </div>

          <div className="mt-3">
            <DataRow
              label="Candle Count"
              value={formatNumber(
                regime.candle_count,
                0,
              )}
            />

            <DataRow
              label="Method"
              value={labelize(regime.method)}
            />
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ============================================================
   PHASE 14.3 — FUNDAMENTAL INTELLIGENCE
   ============================================================ */

function FundamentalCard({
  fundamental,
}: {
  fundamental: FundamentalIntelligence | null;
}) {
  if (!fundamental) {
    return (
      <Card
        title="Phase 14.3 — Fundamental Intelligence"
        subtitle="Independent deterministic fundamental analysis"
      >
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-300">
          Fundamental intelligence was not returned
          by the backend for this analysis.
        </div>
      </Card>
    );
  }

  const evidence = normalizeArray(
    fundamental.evidence,
  );

  const warnings = normalizeArray(
    fundamental.warnings,
  );

  const metricCount =
    fundamental.metric_count ??
    fundamental.available_metric_count;

  let completeness = "--";

  if (
    fundamental.completeness_percent !==
      undefined &&
    fundamental.completeness_percent !== null
  ) {
    completeness =
      `${Number(
        fundamental.completeness_percent,
      ).toFixed(2)}%`;
  } else if (
    fundamental.completeness !==
      undefined &&
    fundamental.completeness !== null
  ) {
    completeness = formatPercent(
      fundamental.completeness,
    );
  } else if (
    metricCount !== undefined &&
    metricCount !== null &&
    fundamental.total_metric_count
  ) {
    completeness =
      `${(
        (Number(metricCount) /
          Number(
            fundamental.total_metric_count,
          )) *
        100
      ).toFixed(2)}%`;
  }

  return (
    <Card
      title="Phase 14.3 — Fundamental Intelligence"
      subtitle="Independent deterministic fundamental analysis"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Fundamental Signal"
          value={labelize(
            fundamental.signal,
          )}
          subValue={`Confidence ${formatPercent(
            fundamental.confidence,
          )}`}
          tone={fundamental.signal}
        />

        <MetricCard
          label="Confidence"
          value={formatPercent(
            fundamental.confidence,
          )}
        />

        <MetricCard
          label="Fundamental Score"
          value={formatNumber(
            fundamental.fundamental_score,
          )}
        />

        <MetricCard
          label="Completeness"
          value={completeness}
          subValue={
            metricCount !== undefined &&
            metricCount !== null
              ? `${metricCount} metrics available`
              : undefined
          }
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Fundamental Scores
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="Valuation"
              value={formatNumber(
                fundamental.valuation_score,
              )}
            />

            <DataRow
              label="Profitability"
              value={formatNumber(
                fundamental.profitability_score,
              )}
            />

            <DataRow
              label="Financial Health"
              value={formatNumber(
                fundamental.financial_health_score,
              )}
            />

            <DataRow
              label="Shareholder Return"
              value={formatNumber(
                fundamental.shareholder_return_score,
              )}
            />

            <DataRow
              label="Metric Count"
              value={formatNumber(
                metricCount,
                0,
              )}
            />

            <DataRow
              label="Total Metrics"
              value={formatNumber(
                fundamental.total_metric_count,
                0,
              )}
            />
          </div>
        </div>

        <div>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Assessments
          </h3>

          <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
            <DataRow
              label="Valuation"
              value={
                fundamental.valuation_assessment ||
                "--"
              }
            />

            <DataRow
              label="Profitability"
              value={
                fundamental.profitability_assessment ||
                "--"
              }
            />

            <DataRow
              label="Financial Health"
              value={
                fundamental.financial_health_assessment ||
                "--"
              }
            />

            <DataRow
              label="Shareholder Return"
              value={
                fundamental.shareholder_return_assessment ||
                "--"
              }
            />
          </div>
        </div>
      </div>

      {evidence.length > 0 && (
        <div className="mt-6">
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Fundamental Evidence
          </h3>

          <div className="grid gap-2 md:grid-cols-2">
            {evidence.map((item, index) => (
              <div
                key={`${item}-${index}`}
                className="rounded-lg border border-[#1e293b] bg-[#080f20] p-3 text-sm text-slate-300"
              >
                {item}
              </div>
            ))}
          </div>
        </div>
      )}

      {warnings.length > 0 && (
        <div className="mt-6">
          <h3 className="mb-3 text-sm font-semibold text-amber-300">
            Fundamental Warnings
          </h3>

          <div className="space-y-2">
            {warnings.map((item, index) => (
              <div
                key={`${item}-${index}`}
                className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-sm text-amber-300"
              >
                {item}
              </div>
            ))}
          </div>
        </div>
      )}

      {fundamental.method && (
        <div className="mt-4 text-xs text-slate-500">
          Method: {fundamental.method}
        </div>
      )}
    </Card>
  );
}

/* ============================================================
   MAIN PAGE
   ============================================================ */

export default function Phase14AdvancedAIPage() {
  const [instrumentKey, setInstrumentKey] =
    useState("NSE_EQ|INE009A01021");

  const [stockSearch, setStockSearch] =
    useState("INFY");

  const [stockResults, setStockResults] =
    useState<StockSearchResult[]>([]);

  const [showStockResults, setShowStockResults] =
    useState(false);

  const [stockSearchLoading, setStockSearchLoading] =
    useState(false);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [data, setData] =
    useState<Phase14Result | null>(null);

  /* ----------------------------------------------------------
     STOCK SEARCH
     ---------------------------------------------------------- */

  const searchStocks = async (
    value: string,
  ) => {
    setStockSearch(value);

    if (value.trim().length < 2) {
      setStockResults([]);
      setShowStockResults(false);
      return;
    }

    setStockSearchLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE}/api/stocks/search?query=${encodeURIComponent(
          value.trim(),
        )}`,
      );

      if (!response.ok) {
        throw new Error(
          `Stock search failed (${response.status}).`,
        );
      }

      const json = await response.json();

      const results =
        extractSearchResults(json);

      setStockResults(results);
      setShowStockResults(true);
    } catch (err) {
      setStockResults([]);
      setShowStockResults(false);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to search stocks.",
      );
    } finally {
      setStockSearchLoading(false);
    }
  };

  /* ----------------------------------------------------------
     SELECT STOCK
     ---------------------------------------------------------- */

  const selectStock = (
    stock: StockSearchResult,
  ) => {
    const key = getInstrumentKey(stock);

    if (!key) {
      setError(
        "The selected stock does not contain a valid instrument key.",
      );
      return;
    }

    const symbol = getStockSymbol(stock);

    setInstrumentKey(key);
    setStockSearch(symbol);
    setShowStockResults(false);
    setError("");

    /*
     * Clear old analysis when changing stock.
     *
     * This is important:
     * otherwise the UI could display the previous
     * stock's result while the new stock is selected.
     */
    setData(null);
  };

  /* ----------------------------------------------------------
     RUN PHASE 14
     ---------------------------------------------------------- */


  /* ----------------------------------------------------------
     RUN PHASE 14
     ---------------------------------------------------------- */

  const runAnalysis = async () => {
    const key = instrumentKey.trim();

    if (!key) {
      setError(
        "Please select a stock or enter a valid instrument key.",
      );
      return;
    }

    const controller = new AbortController();
    let timedOut = false;

    // Allow up to 120 seconds for Phase 14 orchestration.
    const timeoutId = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 120_000);

    setLoading(true);
    setError("");
    setData(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/advanced-ai/run`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            instrument_key: key,
          }),
          signal: controller.signal,
        },
      );

      // Read as text first so an HTML error page or empty
      // response does not cause an unexplained JSON error.
      const rawBody = await response.text();

      let json: ApiResponse | null = null;

      if (rawBody.trim()) {
        try {
          json = JSON.parse(rawBody) as ApiResponse;
        } catch {
          throw new Error(
            `Phase 14 returned an invalid JSON response (HTTP ${response.status}).`,
          );
        }
      }

      if (!response.ok) {
        let message =
          `Phase 14 request failed (HTTP ${response.status}).`;

        if (typeof json?.detail === "string") {
          message = json.detail;
        } else if (json?.detail) {
          message = JSON.stringify(
            json.detail,
            null,
            2,
          );
        }

        throw new Error(message);
      }

      if (!json?.result) {
        throw new Error(
          "Phase 14 API returned no result object.",
        );
      }

      const returnedKey =
        json.result.instrument_key ||
        json.instrument_key ||
        "";

      if (returnedKey && returnedKey !== key) {
        throw new Error(
          `Instrument mismatch. Requested ${key}, but backend returned ${returnedKey}.`,
        );
      }

      setData(json.result);
    } catch (err) {
      setData(null);

      if (timedOut) {
        setError(
          "Phase 14 analysis exceeded 120 seconds. Check the backend terminal to determine which layer is taking too long.",
        );
      } else if (
        err instanceof Error &&
        err.name === "AbortError"
      ) {
        setError(
          "Phase 14 request was cancelled.",
        );
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to run Phase 14 analysis.",
        );
      }
    } finally {
      window.clearTimeout(timeoutId);
      setLoading(false);
    }
  };


  /* ----------------------------------------------------------
     DERIVED DATA
     ---------------------------------------------------------- */

  const inputSource =
    data?.phase14_input_source || null;

  const context =
    inputSource?.context || null;

  const technical =
    context?.technical_intelligence || null;

  const regime =
    context?.stock_regime_intelligence || null;

  const fundamental =
    context?.fundamental_intelligence || null;

  const completedLayers =
    data?.layers_completed?.length || 0;

  const symbol =
    inputSource?.symbol ||
    data?.instrument_key ||
    instrumentKey;

  const marketLabel = [
    labelize(data?.market_regime),
    labelize(data?.market_direction),
  ]
    .filter(
      (value) => value !== "--",
    )
    .join(" / ");

  const layerStatus = useMemo(
    () => data?.layer_status || {},
    [data],
  );

  return (
    <div className="min-h-screen bg-[#020617] p-4 md:p-6 lg:p-8">
      <div className="mx-auto max-w-7xl space-y-6">

        {/* ====================================================
            HEADER
            ==================================================== */}

        <div className="rounded-2xl border border-[#1e293b] bg-[#0b1224] p-6 shadow-sm">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="mb-2 inline-flex rounded-full bg-cyan-400 px-3 py-1 text-xs font-bold text-slate-950">
                PHASE 14
              </div>

              <h1 className="text-2xl font-bold text-white md:text-3xl">
                Advanced AI Intelligence
              </h1>

              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                Stock-specific AI decision intelligence
                with independent technical, stock-regime
                and fundamental analysis.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4 py-3 text-center">
                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                  Pipeline
                </div>

                <div className="mt-1 text-lg font-bold text-white">
                  {completedLayers}/7
                </div>
              </div>

              <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4 py-3 text-center">
                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                  Validation
                </div>

                <div
                  className={`mt-1 text-sm font-bold ${validationClass(
                    data?.validation_status,
                  )}`}
                >
                  {data?.validation_status ||
                    "--"}
                </div>
              </div>

              <div className="col-span-2 rounded-xl border border-[#1e293b] bg-[#080f20] px-4 py-3 text-center sm:col-span-1">
                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                  Stock
                </div>

                <div className="mt-1 truncate text-sm font-bold text-white">
                  {symbol || "--"}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ====================================================
            STOCK INPUT
            ==================================================== */}

        <Card
          title="Select Stock"
          subtitle="Phase 14 is stock-driven. Only the instrument key is sent to the backend."
        >
          <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
            <div className="relative">
              <label className="mb-2 block text-xs font-medium text-slate-300">
                Search Company / Symbol
              </label>

              <input
                value={stockSearch}
                onChange={(event) =>
                  searchStocks(
                    event.target.value,
                  )
                }
                onFocus={() => {
                  if (
                    stockResults.length > 0
                  ) {
                    setShowStockResults(true);
                  }
                }}
                onBlur={() => {
                  /*
                   * Delay allows click on dropdown item
                   * before closing it.
                   */
                  window.setTimeout(() => {
                    setShowStockResults(false);
                  }, 150);
                }}
                className="w-full rounded-lg border border-[#26344d] bg-[#080f20] px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20"
                placeholder="Search INFY, TCS, RELIANCE..."
              />

              {stockSearchLoading && (
                <div className="absolute right-3 top-[39px] text-xs text-cyan-300">
                  Searching...
                </div>
              )}

              {showStockResults && (
                <div className="absolute left-0 right-0 z-50 mt-1 max-h-72 overflow-y-auto rounded-xl border border-[#26344d] bg-[#0b1224] shadow-2xl">
                  {stockResults.length > 0 ? (
                    stockResults.map(
                      (stock, index) => {
                        const stockKey =
                          getInstrumentKey(
                            stock,
                          );

                        const stockSymbol =
                          getStockSymbol(
                            stock,
                          );

                        const stockName =
                          getStockName(
                            stock,
                          );

                        return (
                          <button
                            type="button"
                            key={`${stockKey || stockSymbol}-${index}`}
                            onMouseDown={(event) =>
                              event.preventDefault()
                            }
                            onClick={() =>
                              selectStock(
                                stock,
                              )
                            }
                            className="block w-full border-b border-[#172238] px-4 py-3 text-left transition last:border-b-0 hover:bg-[#111c33]"
                          >
                            <div className="font-semibold text-white">
                              {stockSymbol ||
                                "Unknown"}
                            </div>

                            {stockName &&
                              stockName !==
                                stockSymbol && (
                                <div className="mt-1 text-xs text-slate-400">
                                  {stockName}
                                </div>
                              )}

                            {stockKey && (
                              <div className="mt-1 text-[10px] text-slate-500">
                                {stockKey}
                              </div>
                            )}
                          </button>
                        );
                      },
                    )
                  ) : (
                    <div className="px-4 py-3 text-sm text-slate-500">
                      No stocks found.
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="lg:min-w-[320px]">
              <label className="mb-2 block text-xs font-medium text-slate-300">
                Instrument Key
              </label>

              <input
                value={instrumentKey}
                onChange={(event) =>
                  setInstrumentKey(
                    event.target.value,
                  )
                }
                className="w-full rounded-lg border border-[#26344d] bg-[#080f20] px-4 py-3 font-mono text-xs text-white outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20"
                placeholder="NSE_EQ|INE..."
              />
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => {
                setInstrumentKey(
                  "NSE_EQ|INE009A01021",
                );
                setStockSearch("INFY");
                setData(null);
                setError("");
              }}
              className="rounded-lg border border-[#26344d] bg-[#080f20] px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-cyan-400 hover:text-cyan-300"
            >
              INFY
            </button>

            <button
              type="button"
              onClick={() => {
                setInstrumentKey(
                  "NSE_EQ|INE002A01018",
                );
                setStockSearch("RELIANCE");
                setData(null);
                setError("");
              }}
              className="rounded-lg border border-[#26344d] bg-[#080f20] px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-cyan-400 hover:text-cyan-300"
            >
              RELIANCE
            </button>

            <button
              type="button"
              onClick={() => {
                setInstrumentKey(
                  "NSE_EQ|INE467B01029",
                );
                setStockSearch("TCS");
                setData(null);
                setError("");
              }}
              className="rounded-lg border border-[#26344d] bg-[#080f20] px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-cyan-400 hover:text-cyan-300"
            >
              TCS
            </button>

            <button
              type="button"
              onClick={() => {
                setInstrumentKey(
                  "NSE_EQ|INE040A01034",
                );
                setStockSearch("HDFCBANK");
                setData(null);
                setError("");
              }}
              className="rounded-lg border border-[#26344d] bg-[#080f20] px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-cyan-400 hover:text-cyan-300"
            >
              HDFCBANK
            </button>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={runAnalysis}
              disabled={
                loading ||
                !instrumentKey.trim()
              }
              className="rounded-lg bg-cyan-400 px-6 py-3 text-sm font-bold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading
                ? "Running Phase 14..."
                : "Run Advanced AI"}
            </button>

            {data && (
              <button
                type="button"
                onClick={() => {
                  setData(null);
                  setError("");
                }}
                className="rounded-lg border border-[#26344d] bg-[#080f20] px-5 py-3 text-sm font-medium text-slate-300 transition hover:border-slate-500"
              >
                Clear Result
              </button>
            )}
          </div>

          {error && (
            <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
              <div className="text-sm font-semibold text-red-300">
                Phase 14 Error
              </div>

              <pre className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-red-200">
                {error}
              </pre>
            </div>
          )}
        </Card>

        {/* ====================================================
            LOADING
            ==================================================== */}

        {loading && (
          <div className="rounded-2xl border border-cyan-500/20 bg-[#0b1224] p-8 text-center">
            <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />

            <div className="mt-4 text-sm font-semibold text-white">
              Running Phase 14 Advanced AI
            </div>

            <div className="mt-1 text-xs text-slate-500">
              Processing the selected stock through
              Layers 1 → 7.
            </div>
          </div>
        )}

        {/* ====================================================
            EMPTY STATE
            ==================================================== */}

        {!data && !loading && (
          <div className="rounded-2xl border border-dashed border-[#26344d] bg-[#0b1224] p-12 text-center">
            <div className="text-lg font-semibold text-slate-200">
              Ready for Advanced AI Analysis
            </div>

            <p className="mx-auto mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              Select a stock and run the Phase 14
              pipeline. The frontend will display the
              actual backend-generated Technical,
              Stock Regime and Fundamental intelligence.
            </p>
          </div>
        )}

        {/* ====================================================
            RESULTS
            ==================================================== */}

        {data && !loading && (
          <div className="space-y-6">

            {/* ================================================
                FINAL DECISION
                ================================================ */}

            <Card
              title="Final AI Decision"
              subtitle={`${symbol} • ${data.instrument_key || instrumentKey}`}
            >
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <div
                  className={`rounded-xl border p-5 ${signalBgClass(
                    data.decision,
                  )}`}
                >
                  <div className="text-xs uppercase tracking-wide text-slate-400">
                    Decision
                  </div>

                  <div
                    className={`mt-2 text-4xl font-black ${signalTextClass(
                      data.decision,
                    )}`}
                  >
                    {labelize(data.decision)}
                  </div>
                </div>

                <MetricCard
                  label="Decision Confidence"
                  value={formatPercent(
                    data.decision_confidence,
                  )}
                />

                <MetricCard
                  label="Reliability"
                  value={formatPercent(
                    data.reliability_score,
                  )}
                  subValue={
                    data.reliability_level
                      ? labelize(
                          data.reliability_level,
                        )
                      : undefined
                  }
                />

                <MetricCard
                  label="Confidence Grade"
                  value={
                    data.confidence_grade ||
                    "--"
                  }
                  subValue={
                    data.confidence_band
                      ? labelize(
                          data.confidence_band,
                        )
                      : undefined
                  }
                />
              </div>

              <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Recommendation"
                  value={labelize(
                    data.recommendation,
                  )}
                  subValue={`Confidence ${formatPercent(
                    data.recommendation_confidence,
                  )}`}
                  tone={data.recommendation}
                />

                <MetricCard
                  label="Recommendation Quality"
                  value={labelize(
                    data.recommendation_quality,
                  )}
                />

                <MetricCard
                  label="Recommendation Strength"
                  value={labelize(
                    data.recommendation_strength,
                  )}
                />

                <MetricCard
                  label="Recommendation Risk"
                  value={labelize(
                    data.recommendation_risk,
                  )}
                  tone={data.recommendation_risk}
                />
              </div>
            </Card>

            {/* ================================================
                PHASE 14.1
                ================================================ */}

            <TechnicalIntelligenceCard
              technical={technical}
            />

            {/* ================================================
                PHASE 14.2
                ================================================ */}

            <StockRegimeCard
              regime={regime}
            />

            {/* ================================================
                PHASE 14.3
                ================================================ */}

            <FundamentalCard
              fundamental={fundamental}
            />

            {/* ================================================
                PHASE 14 SOURCE WIRING
                ================================================ */}

            <Card
              title="Phase 14 Source Wiring"
              subtitle="Backend registration of independent Phase 14 evidence sources"
            >
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <MetricCard
                  label="Phase 14.1 Technical"
                  value={labelize(
                    inputSource?.phase14_1?.signal ||
                      inputSource?.technical_signal,
                  )}
                  subValue={`Confidence ${formatPercent(
                    inputSource?.phase14_1?.confidence ??
                      inputSource?.technical_confidence,
                  )} • Weight ${formatPercent(
                    inputSource?.phase14_1?.decision_weight,
                  )}`}
                  tone={
                    inputSource?.phase14_1?.signal ||
                    inputSource?.technical_signal
                  }
                />

                <MetricCard
                  label="Phase 14.2 Stock Regime"
                  value={labelize(
                    inputSource?.phase14_2?.stock_regime ||
                      inputSource?.stock_regime,
                  )}
                  subValue={`Confidence ${formatPercent(
                    inputSource?.phase14_2?.confidence ??
                      inputSource?.stock_regime_confidence,
                  )} • Weight ${formatPercent(
                    inputSource?.phase14_2?.decision_weight,
                  )}`}
                  tone={
                    inputSource?.phase14_2?.stock_direction
                  }
                />

                <MetricCard
                  label="Phase 14.3 Fundamental"
                  value={labelize(
                    fundamental?.signal ||
                      inputSource?.fundamental_signal,
                  )}
                  subValue={`Confidence ${formatPercent(
                    fundamental?.confidence ??
                      inputSource?.fundamental_context_confidence,
                  )}`}
                  tone={
                    fundamental?.signal ||
                    inputSource?.fundamental_signal
                  }
                />
              </div>

              <div className="mt-5 rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
                <DataRow
                  label="Technical Role"
                  value={
                    inputSource?.phase14_1?.role ||
                    "--"
                  }
                />
                <DataRow
                  label="Stock Regime Role"
                  value={
                    inputSource?.phase14_2?.role ||
                    "--"
                  }
                />
                <DataRow
                  label="Technical Candles"
                  value={formatNumber(
                    inputSource?.phase14_1?.candle_count ??
                      technical?.candle_count,
                    0,
                  )}
                />
                <DataRow
                  label="Regime Candles"
                  value={formatNumber(
                    inputSource?.phase14_2?.candle_count ??
                      regime?.candle_count,
                    0,
                  )}
                />
              </div>
            </Card>

            {/* ================================================
                MARKET / SECTOR CONTEXT
                ================================================ */}

            <Card
              title="Market & Sector Context"
              subtitle="Context supplied to the Phase 14 decision pipeline"
            >
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Market Regime"
                  value={labelize(
                    data.market_regime,
                  )}
                  subValue={
                    data.market_direction
                      ? labelize(
                          data.market_direction,
                        )
                      : undefined
                  }
                  tone={data.market_direction}
                />

                <MetricCard
                  label="Market Context"
                  value={labelize(
                    data.market_context,
                  )}
                  subValue={`Score ${formatPercent(
                    data.market_context_score,
                  )}`}
                />

                <MetricCard
                  label="Sector Context"
                  value={formatPercent(
                    data.sector_context_score,
                  )}
                  subValue={
                    data.sector
                      ? data.sector
                      : undefined
                  }
                />

                <MetricCard
                  label="Overall Context"
                  value={formatPercent(
                    data.overall_context_score,
                  )}
                  subValue={
                    data.overall_alignment
                      ? labelize(
                          data.overall_alignment,
                        )
                      : undefined
                  }
                />
              </div>

              <div className="mt-5 rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
                <DataRow
                  label="Market"
                  value={
                    marketLabel || "--"
                  }
                />

                <DataRow
                  label="Sector"
                  value={
                    data.sector || "--"
                  }
                />

                <DataRow
                  label="Sector Direction"
                  value={labelize(
                    data.sector_direction,
                  )}
                />

                <DataRow
                  label="Market Alignment"
                  value={labelize(
                    data.market_alignment,
                  )}
                />

                <DataRow
                  label="Sector Alignment"
                  value={labelize(
                    data.sector_alignment,
                  )}
                />

                <DataRow
                  label="Context Strength"
                  value={labelize(
                    data.context_strength,
                  )}
                />

                <DataRow
                  label="Context Risk"
                  value={labelize(
                    data.context_risk,
                  )}
                />
              </div>
            </Card>

            {/* ================================================
                MODEL AGREEMENT
                ================================================ */}

            <Card
              title="Model Agreement & Conflict"
              subtitle="Phase 9 model context used by Phase 14"
            >
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Models"
                  value={formatNumber(
                    data.model_count,
                    0,
                  )}
                />

                <MetricCard
                  label="Agreement"
                  value={formatNumber(
                    data.agreement_count,
                    0,
                  )}
                />

                <MetricCard
                  label="Conflict"
                  value={formatBoolean(
                    data.conflict,
                  )}
                />

                <MetricCard
                  label="Conflict Level"
                  value={labelize(
                    data.conflict_level,
                  )}
                  tone={
                    data.conflict_level
                  }
                />
              </div>
            </Card>

            {/* ================================================
                EXPLANATION
                ================================================ */}

            <Card
              title="AI Explanation"
              subtitle="Backend-generated explanation"
            >
              <div className="space-y-4">
                {data.decision_summary && (
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-wide text-cyan-300">
                      Decision Summary
                    </div>

                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {data.decision_summary}
                    </p>
                  </div>
                )}

                {data.confidence_explanation && (
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Confidence
                    </div>

                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {
                        data.confidence_explanation
                      }
                    </p>
                  </div>
                )}

                {data.reliability_explanation && (
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Reliability
                    </div>

                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {
                        data.reliability_explanation
                      }
                    </p>
                  </div>
                )}

                {data.final_explanation && (
                  <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wide text-cyan-300">
                      Final Explanation
                    </div>

                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {data.final_explanation}
                    </p>
                  </div>
                )}
              </div>
            </Card>

            {/* ================================================
                FACTORS
                ================================================ */}

            <Card
              title="Decision Factors"
              subtitle="Factors returned by the Phase 14 backend"
            >
              <div className="grid gap-6 lg:grid-cols-3">
                <FactorList
                  title="Bullish Factors"
                  items={normalizeArray(
                    data.bullish_factors,
                  )}
                  tone="bullish"
                />

                <FactorList
                  title="Bearish Factors"
                  items={normalizeArray(
                    data.bearish_factors,
                  )}
                  tone="bearish"
                />

                <FactorList
                  title="Neutral Factors"
                  items={normalizeArray(
                    data.neutral_factors,
                  )}
                  tone="neutral"
                />
              </div>

              <div className="mt-6 grid gap-6 lg:grid-cols-2">
                <FactorList
                  title="Supporting Factors"
                  items={normalizeArray(
                    data.supporting_factors,
                  )}
                  tone="bullish"
                />

                <FactorList
                  title="Conflicting Factors"
                  items={normalizeArray(
                    data.conflicting_factors,
                  )}
                  tone="bearish"
                />
              </div>
            </Card>

            {/* ================================================
                STRONGEST FACTOR
                ================================================ */}

            <Card title="Strongest Factor & Risk">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-5">
                  <div className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
                    Strongest Factor
                  </div>

                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    {data.strongest_factor ||
                      "No strongest factor returned."}
                  </p>
                </div>

                <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-5">
                  <div className="text-xs font-semibold uppercase tracking-wide text-red-300">
                    Strongest Risk
                  </div>

                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    {data.strongest_risk ||
                      "No strongest risk returned."}
                  </p>
                </div>
              </div>
            </Card>

            {/* ================================================
                GUIDANCE
                ================================================ */}

            <Card title="AI Guidance">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-xl border border-[#1e293b] bg-[#080f20] p-5">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Action Guidance
                  </div>

                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    {data.action_guidance ||
                      "--"}
                  </p>
                </div>

                <div className="rounded-xl border border-[#1e293b] bg-[#080f20] p-5">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Position Guidance
                  </div>

                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    {data.position_guidance ||
                      "--"}
                  </p>
                </div>
              </div>
            </Card>

            {/* ================================================
                VALIDATION
                ================================================ */}

            <Card
              title="Validation"
              subtitle="Phase 14 Layer 6 validation result"
            >
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                <MetricCard
                  label="Status"
                  value={
                    data.validation_status ||
                    "--"
                  }
                  tone={
                    data.validation_status
                  }
                />

                <MetricCard
                  label="Validation Score"
                  value={formatPercent(
                    data.validation_score,
                  )}
                />

                <MetricCard
                  label="Quality"
                  value={labelize(
                    data.validation_quality,
                  )}
                />

                <MetricCard
                  label="Recommendation Valid"
                  value={formatBoolean(
                    data.recommendation_valid,
                  )}
                />

                <MetricCard
                  label="Trust"
                  value={labelize(
                    data.recommendation_trust,
                  )}
                />
              </div>

              <div className="mt-4 rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
                <DataRow
                  label="Validation Risk"
                  value={labelize(
                    data.validation_risk,
                  )}
                />
              </div>
            </Card>

            {/* ================================================
                LAYER STATUS
                ================================================ */}

            <Card
              title="Phase 14 Pipeline Status"
              subtitle="Layers returned by the backend"
            >
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {[1, 2, 3, 4, 5, 6, 7].map(
                  (layer) => {
                    const completed =
                      data.layers_completed?.includes(
                        layer,
                      );

                    const status =
                      layerStatus[
                        String(layer)
                      ];

                    return (
                      <div
                        key={layer}
                        className={`rounded-xl border p-4 ${
                          completed
                            ? "border-emerald-500/20 bg-emerald-500/5"
                            : "border-[#1e293b] bg-[#080f20]"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-semibold text-slate-200">
                            Layer {layer}
                          </span>

                          <span
                            className={
                              completed
                                ? "text-emerald-400"
                                : "text-slate-500"
                            }
                          >
                            {completed
                              ? "✓"
                              : "—"}
                          </span>
                        </div>

                        <div className="mt-2 text-xs text-slate-500">
                          {status ||
                            (completed
                              ? "Completed"
                              : "Not completed")}
                        </div>
                      </div>
                    );
                  },
                )}
              </div>
            </Card>

            {/* ================================================
                WARNINGS
                ================================================ */}

            {normalizeArray(
              data.warnings,
            ).length > 0 && (
              <Card
                title="Warnings"
                subtitle="Warnings returned by the Phase 14 pipeline"
              >
                <div className="space-y-2">
                  {normalizeArray(
                    data.warnings,
                  ).map((warning, index) => (
                    <div
                      key={`${warning}-${index}`}
                      className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-sm text-amber-300"
                    >
                      {warning}
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* ================================================
                SYSTEM STATUS
                ================================================ */}

            <Card
              title="System Status"
              subtitle="Final Phase 14 execution metadata"
            >
              <div className="rounded-xl border border-[#1e293b] bg-[#080f20] px-4">
                <DataRow
                  label="Overall Status"
                  value={
                    data.overall_status ||
                    "--"
                  }
                />

                <DataRow
                  label="Result Status"
                  value={
                    data.status || "--"
                  }
                />

                <DataRow
                  label="Validation"
                  value={
                    data.validation_status ||
                    "--"
                  }
                />

                <DataRow
                  label="Completed Layers"
                  value={
                    data.layers_completed?.join(
                      ", ",
                    ) || "--"
                  }
                />

                <DataRow
                  label="Phase"
                  value={
                    data.metadata?.phase ??
                    14
                  }
                />

                <DataRow
                  label="Final Layer"
                  value={
                    data.metadata?.layer ??
                    7
                  }
                />

                <DataRow
                  label="Component"
                  value={
                    data.metadata?.component ||
                    "--"
                  }
                />

                <DataRow
                  label="Recommendation Source"
                  value={
                    data.metadata
                      ?.recommendation_source ||
                    "--"
                  }
                />

                <DataRow
                  label="Validation Source"
                  value={
                    data.metadata
                      ?.validation_source ||
                    "--"
                  }
                />

                <DataRow
                  label="Aggregation Only"
                  value={formatBoolean(
                    data.metadata
                      ?.aggregation_only,
                  )}
                />
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}