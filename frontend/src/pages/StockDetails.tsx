import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import LiveMarketChart, { type LiveChartPeriod, type TechnicalSummary } from "../components/LiveMarketChart";
import { MLPredictions } from "../components/MLPredictions";
import { API_BASE_URL, getMarketStatus, type MarketStatus } from "../services/marketApi";

interface QuoteData {
  status: string;
  symbol: string;
  value: number | null;
  change: number | null;
  net_change: number | null;
  previous_close: number | null;
  open: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
  timestamp?: string | null;
}

interface Fundamentals {
  market_cap: number | null;
  pe_ratio: number | string | null;
  pb_ratio: number | string | null;
  eps: number | null;
  eps_basic: number | null;
  eps_diluted: number | null;
  dividend_yield: number | null;
  book_value: number | null;
  roe: number | string | null;
  roa: number | string | null;
  roce: number | string | null;
  ev_ebitda: number | string | null;
  quick_ratio: number | string | null;
  total_assets: number | null;
  current_assets: number | null;
  current_liabilities: number | null;
  non_current_liabilities: number | null;
  equity_capital: number | null;
  debt_to_equity: number | null;
}

interface FundamentalsResponse {
  status: string;
  symbol?: string;
  name?: string;
  sector?: string;
  fundamentals?: Fundamentals;
  fundamental_period?: string;
  fundamental_type?: string;
  balance_sheet_period?: string;
  balance_sheet_type?: string;
  note?: string;
  message?: string;
}

function formatPrice(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `₹${value.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatNumber(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-IN");
}

function statusText(status: string | null | undefined) {
  switch (status) {
    case "NORMAL_OPEN":
      return "Market Open";
    case "PRE_OPEN_START":
    case "PRE_OPEN_END":
      return "Pre-open";
    case "CLOSING_START":
      return "Closing session";
    case "NORMAL_CLOSE":
    case "CLOSING_END":
      return "Market Closed";
    default:
      return "Market status unavailable";
  }
}

export default function StockDetails() {
  const { symbol } = useParams();
  const [searchParams] = useSearchParams();
  const instrumentKey = searchParams.get("instrument_key") || "";

  const isIndex = useMemo(
    () => instrumentKey.startsWith("NSE_INDEX|") || instrumentKey.startsWith("BSE_INDEX|"),
    [instrumentKey]
  );

  const exchange = instrumentKey.startsWith("BSE_") ? "BSE" : "NSE";
  const [stock, setStock] = useState<QuoteData | null>(null);
  const [quoteLoading, setQuoteLoading] = useState(true);
  const [quoteError, setQuoteError] = useState("");
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<LiveChartPeriod>("1D");
  const [technicalSummary, setTechnicalSummary] = useState<TechnicalSummary | null>(null);
  const [fundamentals, setFundamentals] = useState<FundamentalsResponse | null>(null);
  const [fundamentalsLoading, setFundamentalsLoading] = useState(false);
  const [fundamentalsError, setFundamentalsError] = useState("");

  useEffect(() => {
    if (!instrumentKey) {
      setQuoteError("Instrument key is missing.");
      setQuoteLoading(false);
      return;
    }

    let alive = true;
    let timer: number | undefined;

    const loadQuote = async () => {
      try {
        const endpoint = isIndex ? "/api/indices/quote" : "/api/stocks/quote";
        const response = await fetch(
          `${API_BASE_URL}${endpoint}?instrument_key=${encodeURIComponent(instrumentKey)}`
        );
        if (!response.ok) throw new Error(`Quote request failed: ${response.status}`);
        const data = await response.json();
        if (data.status !== "success") throw new Error(data.message || "Unable to load quote");
        if (alive) {
          setStock(data);
          setQuoteError("");
          setQuoteLoading(false);
        }
      } catch (error) {
        console.error("Quote error:", error);
        if (alive) {
          setQuoteError(error instanceof Error ? error.message : "Unable to load quote");
          setQuoteLoading(false);
        }
      }
    };

    loadQuote();
    timer = window.setInterval(loadQuote, 15000);

    return () => {
      alive = false;
      if (timer) window.clearInterval(timer);
    };
  }, [instrumentKey, isIndex]);

  // Load company fundamentals for NSE/BSE equities only.
  useEffect(() => {
    if (!symbol || isIndex) {
      setFundamentals(null);
      setFundamentalsError("");
      setFundamentalsLoading(false);
      return;
    }

    let alive = true;

    const loadFundamentals = async () => {
      try {
        setFundamentalsLoading(true);
        setFundamentalsError("");

        const response = await fetch(
          `${API_BASE_URL}/api/fundamentals/${encodeURIComponent(symbol)}`
        );

        const data: FundamentalsResponse = await response.json();

        if (!response.ok || data.status !== "success") {
          throw new Error(
            data.message || `Fundamentals request failed: ${response.status}`
          );
        }

        if (alive) {
          setFundamentals(data);
        }
      } catch (error) {
        console.error("Fundamentals error:", error);

        if (alive) {
          setFundamentals(null);
          setFundamentalsError(
            error instanceof Error
              ? error.message
              : "Unable to load fundamentals."
          );
        }
      } finally {
        if (alive) {
          setFundamentalsLoading(false);
        }
      }
    };

    loadFundamentals();

    return () => {
      alive = false;
    };
  }, [symbol, isIndex]);

  useEffect(() => {
    if (!instrumentKey) return;
    let alive = true;

    const loadStatus = async () => {
      try {
        const data = await getMarketStatus(exchange);
        if (alive) setMarketStatus(data);
      } catch (error) {
        console.error("Market status error:", error);
      }
    };

    loadStatus();
    const timer = window.setInterval(loadStatus, 30000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [instrumentKey, exchange]);

  const positive = (stock?.change ?? 0) >= 0;
  const displaySymbol = stock?.symbol || symbol || "Market";
  const status = statusText(marketStatus?.market_status);
  const marketOpen = marketStatus?.market_status === "NORMAL_OPEN";

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <main className="mx-auto max-w-7xl px-6 py-8">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white"
        >
          ← Back to Dashboard
        </Link>

        <section className="mt-8">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className="text-slate-500">{exchange} · {isIndex ? "Index" : "Equity"}</span>
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                marketOpen
                  ? "bg-emerald-400/10 text-emerald-400"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              <span className="mr-1">{marketOpen ? "●" : "○"}</span>
              {status}
            </span>
          </div>

          <h1 className="mt-3 text-3xl font-bold">{displaySymbol}</h1>
          <p className="mt-1 text-sm text-slate-500">
            {isIndex ? "Indian market index · Upstox live market data" : "NSE equity · Upstox live market data"}
          </p>

          {quoteLoading && !stock ? (
            <p className="mt-6 text-3xl text-slate-500">Loading market data...</p>
          ) : quoteError && !stock ? (
            <p className="mt-6 text-sm text-red-400">{quoteError}</p>
          ) : (
            <div className="mt-6 flex flex-wrap items-end gap-4">
              <div className="text-5xl font-bold tracking-tight">{formatPrice(stock?.value)}</div>
              <div className={`pb-1 text-lg font-semibold ${positive ? "text-emerald-400" : "text-red-400"}`}>
                {positive ? "+" : ""}{formatPrice(stock?.net_change)}
                <span className="ml-2">({positive ? "+" : ""}{stock?.change?.toFixed(2) ?? "—"}%)</span>
              </div>
              {stock?.timestamp && (
                <div className="pb-1 text-xs text-slate-500">
                  Quote updated {new Date(stock.timestamp).toLocaleTimeString("en-IN", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </div>
              )}
            </div>
          )}
        </section>

        <LiveMarketChart
          instrumentKey={instrumentKey}
          symbol={displaySymbol}
          isIndex={isIndex}
          period={selectedPeriod}
          onPeriodChange={setSelectedPeriod}
          currentPrice={stock?.value ?? null}
          currentChange={stock?.change ?? null}
          marketStatus={marketStatus?.market_status ?? "NORMAL_OPEN"}
          onTechnicalSummaryChange={setTechnicalSummary}
        />

        {!isIndex && stock?.value && (
          <MLPredictions
            instrumentKey={instrumentKey}
            currentPrice={stock.value}
          />
        )}

        <section className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard title="Open" value={formatPrice(stock?.open)} />
          <StatCard title="Day High" value={formatPrice(stock?.high)} />
          <StatCard title="Day Low" value={formatPrice(stock?.low)} />
          <StatCard title="Volume" value={formatNumber(stock?.volume)} />
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-3">
          <InfoCard title="Previous Close" value={formatPrice(stock?.previous_close)} />
          <InfoCard title="Day Change" value={`${positive ? "+" : ""}${stock?.change?.toFixed(2) ?? "—"}%`} />
          <InfoCard title="Instrument" value={instrumentKey || "—"} />
        </section>

        {!isIndex && (
          <FundamentalsCard
            symbol={displaySymbol}
            sector={fundamentals?.sector}
            data={fundamentals?.fundamentals ?? null}
            period={fundamentals?.fundamental_period}
            type={fundamentals?.fundamental_type}
            loading={fundamentalsLoading}
            error={fundamentalsError}
          />
        )}

        <TechnicalSummaryCard summary={technicalSummary} period={selectedPeriod} />

        <footer className="py-8 text-center text-xs text-slate-600">
          StockAI provides market information and AI-generated analysis for educational and informational purposes only. It is not financial advice.
        </footer>
      </main>
    </div>
  );
}

function FundamentalsCard({
  symbol,
  sector,
  data,
  period,
  type,
  loading,
  error,
}: {
  symbol: string;
  sector?: string;
  data: Fundamentals | null;
  period?: string;
  type?: string;
  loading: boolean;
  error: string;
}) {
  const formatCr = (value: number | null | undefined) => {
    if (value == null || !Number.isFinite(value)) return "—";
    return `₹${value.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })} Cr`;
  };

  const formatMetric = (value: number | string | null | undefined) => {
    if (value == null || value === "") return "—";
    if (typeof value === "number") {
      return Number.isFinite(value) ? value.toFixed(2) : "—";
    }
    return value;
  };

  const formatPercent = (value: number | string | null | undefined) => {
    if (value == null || value === "") return "—";
    if (typeof value === "number") {
      return Number.isFinite(value) ? `${value.toFixed(2)}%` : "—";
    }
    return value.includes("%") ? value : `${value}%`;
  };

  const formatRupees = (value: number | null | undefined) => {
    if (value == null || !Number.isFinite(value)) return "—";
    return `₹${value.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  if (loading) {
    return (
      <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <p className="text-sm text-cyan-400">Fundamental Analysis</p>
        <h2 className="mt-1 text-2xl font-bold">Company Fundamentals</h2>
        <p className="mt-4 text-sm text-slate-500">
          Loading fundamental data for {symbol}...
        </p>
      </section>
    );
  }

  if (error || !data) {
    return (
      <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <p className="text-sm text-cyan-400">Fundamental Analysis</p>
        <h2 className="mt-1 text-2xl font-bold">Company Fundamentals</h2>
        <p className="mt-4 text-sm text-slate-500">
          {error || "Fundamental data is unavailable for this stock."}
        </p>
      </section>
    );
  }

  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm text-cyan-400">Fundamental Analysis</p>
          <h2 className="mt-1 text-2xl font-bold">Company Fundamentals</h2>
          <p className="mt-1 text-sm text-slate-500">
            {symbol} · {sector || "Sector unavailable"}
          </p>
        </div>

        {(period || type) && (
          <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-right text-xs text-slate-500">
            <div>{period || "Yearly"} financial data</div>
            <div className="mt-1 capitalize">{type || "Consolidated"}</div>
          </div>
        )}
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryMetric label="Sector" value={sector || "—"} />
        <SummaryMetric label="Market Cap" value={formatCr(data.market_cap)} />
        <SummaryMetric label="P/E" value={formatMetric(data.pe_ratio)} />
        <SummaryMetric label="P/B" value={formatMetric(data.pb_ratio)} />

        <SummaryMetric label="EPS" value={formatRupees(data.eps)} />
        <SummaryMetric label="Dividend Yield" value={formatPercent(data.dividend_yield)} />
        <SummaryMetric label="Book Value" value={formatRupees(data.book_value)} />
        <SummaryMetric label="Debt-to-Equity" value={formatMetric(data.debt_to_equity)} />

        <SummaryMetric label="ROE" value={formatPercent(data.roe)} />
        <SummaryMetric label="ROA" value={formatPercent(data.roa)} />
        <SummaryMetric label="ROCE" value={formatPercent(data.roce)} />
        <SummaryMetric label="EV/EBITDA" value={formatMetric(data.ev_ebitda)} />

        <SummaryMetric label="Quick Ratio" value={formatMetric(data.quick_ratio)} />
        <SummaryMetric label="Total Assets" value={formatCr(data.total_assets)} />
        <SummaryMetric label="Current Assets" value={formatCr(data.current_assets)} />
        <SummaryMetric label="Current Liabilities" value={formatCr(data.current_liabilities)} />

        <SummaryMetric label="Equity Capital" value={formatCr(data.equity_capital)} />
      </div>

      <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Additional EPS Data
        </p>

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <SummaryMetric label="EPS Basic" value={formatRupees(data.eps_basic)} />
          <SummaryMetric label="EPS Diluted" value={formatRupees(data.eps_diluted)} />
        </div>
      </div>

      <p className="mt-4 text-xs leading-5 text-slate-500">
        Market cap and book value are derived by the backend. Debt-to-equity uses
        total balance-sheet liabilities divided by equity capital for this StockAI
        fundamentals view.
      </p>
    </section>
  );
}

function TechnicalSummaryCard({
  summary,
  period,
}: {
  summary: TechnicalSummary | null;
  period: LiveChartPeriod;
}) {
  if (!summary) {
    return (
      <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <p className="text-sm text-slate-400">StockAI Analysis</p>
        <h2 className="mt-1 text-2xl font-bold">🤖 Technical Summary</h2>
        <p className="mt-3 text-sm text-slate-500">Waiting for market candles to calculate the analysis...</p>
      </section>
    );
  }

  const biasClass =
    summary.bias === "Bullish"
      ? "bg-emerald-400/10 text-emerald-400 border-emerald-400/20"
      : summary.bias === "Bearish"
        ? "bg-red-400/10 text-red-400 border-red-400/20"
        : "bg-amber-400/10 text-amber-300 border-amber-400/20";

  const statusClass = (value: string) =>
    value === "Bullish" || value === "Positive" || value === "Strong" || value === "Above VWAP"
      ? "text-emerald-400"
      : value === "Bearish" || value === "Negative" || value === "Weak" || value === "Below VWAP"
        ? "text-red-400"
        : "text-amber-300";

  const formatPrice = (value: number | null) =>
    value == null || !Number.isFinite(value)
      ? "—"
      : `₹${value.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-400">StockAI Analysis · {period}</p>
          <h2 className="mt-1 text-2xl font-bold">🤖 Technical Summary</h2>
          <p className="mt-1 text-xs text-slate-500">
            Calculated from the selected chart's real OHLC and volume data.
          </p>
        </div>

        <div className={`rounded-xl border px-4 py-3 text-center ${biasClass}`}>
          <div className="text-xs uppercase tracking-wide opacity-80">Overall Bias</div>
          <div className="mt-1 text-xl font-bold">{summary.bias}</div>
          <div className="text-xs opacity-80">Score {summary.score}/100</div>
        </div>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryMetric label="Trend" value={summary.trend} className={statusClass(summary.trend)} />
        <SummaryMetric label="Momentum" value={summary.momentum} className={statusClass(summary.momentum)} />
        <SummaryMetric label="Volume" value={summary.volume} className={statusClass(summary.volume)} />
        <SummaryMetric label="RSI 14" value={summary.rsi == null ? "—" : `${summary.rsi.toFixed(1)} · ${summary.rsiStatus}`} className={statusClass(summary.rsiStatus)} />
        <SummaryMetric label="MACD" value={summary.macdStatus} className={statusClass(summary.macdStatus)} />
        <SummaryMetric label="VWAP" value={summary.vwapStatus} className={statusClass(summary.vwapStatus)} />
        <SummaryMetric label="SMA 20" value={formatPrice(summary.sma20)} />
        <SummaryMetric label="SMA 50" value={formatPrice(summary.sma50)} />
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <LevelCard label="Support 1" value={formatPrice(summary.support1)} positive />
        <LevelCard label="Support 2" value={formatPrice(summary.support2)} positive />
        <LevelCard label="Resistance 1" value={formatPrice(summary.resistance1)} />
        <LevelCard label="Resistance 2" value={formatPrice(summary.resistance2)} />
      </div>

      <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">📊 Analysis</p>
        <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-300">
          {summary.notes.map((note, index) => (
            <li key={`${index}-${note}`} className="flex gap-2">
              <span className="text-slate-600">•</span>
              <span>{note}</span>
            </li>
          ))}
        </ul>
      </div>

      <p className="mt-4 text-xs leading-5 text-slate-500">
        ⚠️ This is rule-based technical analysis from market data, not a guaranteed prediction or financial advice.
      </p>
    </section>
  );
}

function SummaryMetric({
  label,
  value,
  className = "text-slate-200",
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${className}`}>{value}</p>
    </div>
  );
}

function LevelCard({
  label,
  value,
  positive = false,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${positive ? "text-emerald-400" : "text-red-400"}`}>
        {value}
      </p>
    </div>
  );
}

function StatCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <p className="text-sm text-slate-500">{title}</p>
      <p className="mt-2 text-lg font-semibold">{value}</p>
    </div>
  );
}

function InfoCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <p className="text-xs uppercase tracking-wide text-slate-500">{title}</p>
      <p className="mt-2 break-all text-sm font-semibold text-slate-200">{value}</p>
    </div>
  );
}
