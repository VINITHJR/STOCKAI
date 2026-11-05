import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "../services/marketApi";

type MarketData = {
  last_price?: number | null;
  open?: number | null;
  high?: number | null;
  low?: number | null;
  previous_close?: number | null;
  volume?: number | null;
  net_change?: number | null;
};

type Fundamentals = {
  market_cap?: number | null;
  pe_ratio?: number | string | null;
  pb_ratio?: number | string | null;
  eps?: number | null;
  eps_basic?: number | null;
  eps_diluted?: number | null;
  dividend_yield?: number | null;
  book_value?: number | null;
  roe?: number | string | null;
  roa?: number | string | null;
  roce?: number | string | null;
  ev_ebitda?: number | string | null;
  quick_ratio?: number | string | null;
  total_assets?: number | null;
  current_assets?: number | null;
  current_liabilities?: number | null;
  non_current_liabilities?: number | null;
  equity_capital?: number | null;
  debt_to_equity?: number | null;
};

type StockSearchResult = {
  symbol?: string;
  name?: string;
  company_name?: string;
  trading_symbol?: string;
  short_name?: string;
  instrument_key?: string;
  exchange?: string;
  isin?: string;
  instrument_type?: string;
  segment?: string;
};

type FundamentalData = {
  status: string;
  source?: string;
  symbol?: string;
  name?: string;
  company_name?: string;
  instrument_key?: string;
  isin?: string;
  exchange?: string;
  sector?: string | null;
  market_data?: MarketData;
  fundamentals?: Fundamentals;
  fundamental_period?: string;
  fundamental_type?: string;
  balance_sheet_period?: string;
  balance_sheet_type?: string;
  note?: string;
  message?: string;
};

const DEFAULT_SYMBOLS = [
  "RELIANCE",
  "TCS",
  "INFY",
  "HDFCBANK",
  "ICICIBANK",
  "SBIN",
];

function num(value: number | string | null | undefined, digits = 2) {
  if (value == null) return "—";

  const numeric =
    typeof value === "number"
      ? value
      : Number.parseFloat(String(value).replace("%", "").trim());

  if (!Number.isFinite(numeric)) {
    return String(value);
  }

  return numeric.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function pct(value: number | string | null | undefined) {
  if (value == null) return "—";

  if (typeof value === "string" && value.includes("%")) {
    return value;
  }

  const numeric =
    typeof value === "number"
      ? value
      : Number.parseFloat(String(value).replace("%", "").trim());

  if (!Number.isFinite(numeric)) {
    return String(value);
  }

  // Backend dividend yield is already in percentage points.
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(2)}%`;
}

function compactMoneyCr(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `₹${value.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} Cr`;
}

function money(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `₹${value.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function integer(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-IN");
}

function Metric({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-100">{value}</p>
      {description ? (
        <p className="mt-1 text-[11px] leading-4 text-slate-600">
          {description}
        </p>
      ) : null}
    </div>
  );
}

function SectionTitle({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <div className="mb-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
        {title}
      </h3>
      {description ? (
        <p className="mt-1 text-xs text-slate-500">{description}</p>
      ) : null}
    </div>
  );
}

export default function FundamentalAnalysis() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOLS[0]);
  const [searchResults, setSearchResults] = useState<StockSearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [data, setData] = useState<FundamentalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const searchStocks = useCallback(async (query: string) => {
    const clean = query.trim();
    if (clean.length < 2) {
      setSearchResults([]);
      setShowSearchResults(false);
      setSearchLoading(false);
      return;
    }
    try {
      setSearchLoading(true);
      const response = await fetch(
        `${API_BASE_URL}/api/stocks/search?query=${encodeURIComponent(clean)}`,
        { headers: { Accept: "application/json" } }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.message || `Stock search returned HTTP ${response.status}`);
      const rawResults = Array.isArray(payload) ? payload : Array.isArray(payload?.data) ? payload.data : Array.isArray(payload?.results) ? payload.results : [];
      const results: StockSearchResult[] = rawResults
        .map((item: any) => ({
          symbol: item.symbol || item.trading_symbol || item.tradingsymbol || item.short_name,
          name: item.name || item.company_name || item.companyName,
          company_name: item.company_name || item.name || item.companyName,
          trading_symbol: item.trading_symbol,
          short_name: item.short_name,
          instrument_key: item.instrument_key || item.instrumentKey,
          exchange: item.exchange,
          isin: item.isin,
          instrument_type: item.instrument_type,
          segment: item.segment,
        }))
        .filter((item: StockSearchResult) => item.instrument_key && (!item.segment || item.segment === "NSE_EQ") && (!item.instrument_type || item.instrument_type === "EQ" || item.instrument_type === "EQUITY"))
        .slice(0, 10);
      setSearchResults(results);
      setShowSearchResults(true);
    } catch {
      setSearchResults([]);
      setShowSearchResults(true);
    } finally {
      setSearchLoading(false);
    }
  }, []);

  const handleSymbolChange = (value: string) => {
    const next = value.toUpperCase();
    setSymbol(next);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => searchStocks(next), 300);
  };

  const selectSearchResult = (stock: StockSearchResult) => {
    const selected = (stock.symbol || stock.trading_symbol || stock.short_name || "").toUpperCase();
    if (!selected) return;
    setSymbol(selected);
    setSearchResults([]);
    setShowSearchResults(false);
    load(selected);
  };

  const load = useCallback(async (requested: string = symbol) => {
    const clean = requested.trim().toUpperCase();

    if (!clean) {
      setError("Please enter an NSE stock symbol.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        `${API_BASE_URL}/api/fundamentals/${encodeURIComponent(clean)}`,
        {
          headers: { Accept: "application/json" },
        }
      );

      const payload: FundamentalData = await response.json();

      if (!response.ok || payload.status !== "success") {
        throw new Error(
          payload.message ||
            `Fundamental API returned HTTP ${response.status}`
        );
      }

      setData(payload);
      setSymbol(clean);
    } catch (err) {
      setData(null);
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load fundamental data."
      );
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    load(DEFAULT_SYMBOLS[0]);
    // Load the default stock once when the component mounts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, []);

  const fundamentals = data?.fundamentals;
  const marketData = data?.market_data;

  return (
    <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm text-cyan-400">Phase 5 · Layer 6</p>

          <h2 className="mt-1 text-2xl font-semibold text-white">
            Fundamental Analysis
          </h2>

          <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">
            Descriptive company fundamentals for an NSE equity. This layer
            shows financial metrics from the backend and does not change the
            existing technical signals, setup ranking, or chart.
          </p>
        </div>

        <div className="relative w-full lg:w-[620px]">
          <input
            value={symbol}
            onChange={(event) => handleSymbolChange(event.target.value)}
            onFocus={() => {
              if (symbol.trim().length >= 2 && searchResults.length > 0) setShowSearchResults(true);
            }}
            onBlur={() => window.setTimeout(() => setShowSearchResults(false), 180)}
            placeholder="Search stock symbol or company name"
            autoComplete="off"
            className="w-full rounded-xl border border-cyan-500 bg-slate-950 px-5 py-3 text-base font-medium text-white outline-none placeholder:text-slate-500 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400"
          />
          {showSearchResults && symbol.trim().length >= 2 ? (
            <div className="absolute left-0 right-0 top-full z-50 mt-2 max-h-[420px] overflow-y-auto rounded-2xl border border-slate-700 bg-slate-950 shadow-2xl">
              {searchLoading ? (
                <div className="px-5 py-6 text-sm text-slate-400">Searching stocks...</div>
              ) : searchResults.length > 0 ? (
                searchResults.map((stock, index) => {
                  const stockSymbol = stock.symbol || stock.trading_symbol || stock.short_name || "—";
                  const companyName = stock.name || stock.company_name || "Company name unavailable";
                  return (
                    <button
                      key={`${stock.instrument_key || stockSymbol}-${index}`}
                      type="button"
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => selectSearchResult(stock)}
                      className="flex w-full items-center justify-between border-b border-slate-800 px-5 py-4 text-left last:border-b-0 hover:bg-slate-900"
                    >
                      <div className="min-w-0">
                        <p className="text-lg font-semibold text-white">{stockSymbol}</p>
                        <p className="mt-1 truncate text-sm text-slate-400">{companyName} · {stock.exchange || "NSE"}</p>
                        {stock.instrument_key ? <p className="mt-2 text-xs text-slate-600">{stock.instrument_key}</p> : null}
                      </div>
                      <span className="ml-4 shrink-0 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-300">{stock.exchange || "NSE"}</span>
                    </button>
                  );
                })
              ) : (
                <div className="px-5 py-6 text-sm text-slate-500">No matching NSE stocks found.</div>
              )}
            </div>
          ) : null}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {DEFAULT_SYMBOLS.map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => load(item)}
            disabled={loading}
            className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:border-cyan-500 hover:text-cyan-300 disabled:opacity-50"
          >
            {item}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/40 p-6 text-sm text-slate-400">
          Loading fundamentals for {symbol}...
        </div>
      ) : error ? (
        <div className="mt-5 rounded-xl border border-red-900/60 bg-red-950/30 p-5 text-sm text-red-300">
          {error}
        </div>
      ) : data && fundamentals ? (
        <>
          {/* Company header */}
          <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <p className="text-xl font-semibold text-white">
                  {data.name || data.company_name || data.symbol}
                </p>

                <p className="text-sm text-slate-500">
                  {data.symbol || symbol}
                  {" · "}
                  {data.sector || "Sector unavailable"}
                  {data.exchange ? ` · ${data.exchange}` : ""}
                </p>

                {data.instrument_key ? (
                  <p className="mt-1 text-[11px] text-slate-600">
                    Instrument: {data.instrument_key}
                  </p>
                ) : null}
              </div>

              <div className="text-left md:text-right">
                <p className="text-xs text-slate-500">Current Price</p>
                <p className="text-lg font-semibold text-slate-200">
                  {money(marketData?.last_price)}
                </p>

                {marketData?.net_change != null ? (
                  <p
                    className={`text-xs ${
                      marketData.net_change >= 0
                        ? "text-emerald-400"
                        : "text-red-400"
                    }`}
                  >
                    {marketData.net_change >= 0 ? "+" : ""}
                    {num(marketData.net_change)}
                  </p>
                ) : null}
              </div>
            </div>
          </div>

          {/* Market snapshot */}
          {marketData ? (
            <div className="mt-5">
              <SectionTitle
                title="Market Snapshot"
                description="Current market information returned with the fundamentals."
              />

              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Metric
                  label="Last Price"
                  value={money(marketData.last_price)}
                />
                <Metric label="Open" value={money(marketData.open)} />
                <Metric label="Day High" value={money(marketData.high)} />
                <Metric label="Day Low" value={money(marketData.low)} />
                <Metric
                  label="Previous Close"
                  value={money(marketData.previous_close)}
                />
                <Metric
                  label="Net Change"
                  value={money(marketData.net_change)}
                />
                <Metric
                  label="Volume"
                  value={integer(marketData.volume)}
                />
                <Metric
                  label="Sector"
                  value={data.sector || "—"}
                />
              </div>
            </div>
          ) : null}

          {/* Valuation */}
          <div className="mt-6">
            <SectionTitle
              title="Valuation"
              description="Key valuation measures provided by the fundamentals service."
            />

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric
                label="Market Cap"
                value={compactMoneyCr(fundamentals.market_cap)}
                description="Backend-derived market capitalization"
              />
              <Metric
                label="P/E Ratio"
                value={num(fundamentals.pe_ratio)}
              />
              <Metric
                label="P/B Ratio"
                value={num(fundamentals.pb_ratio)}
              />
              <Metric
                label="Book Value"
                value={money(fundamentals.book_value)}
                description="Derived from current price ÷ P/B"
              />
              <Metric
                label="EV / EBITDA"
                value={num(fundamentals.ev_ebitda)}
              />
              <Metric
                label="EPS"
                value={money(fundamentals.eps)}
              />
              <Metric
                label="EPS Basic"
                value={money(fundamentals.eps_basic)}
              />
              <Metric
                label="EPS Diluted"
                value={money(fundamentals.eps_diluted)}
              />
            </div>
          </div>

          {/* Profitability */}
          <div className="mt-6">
            <SectionTitle
              title="Profitability & Returns"
              description="Profitability ratios reported by the backend fundamentals service."
            />

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric label="ROE" value={pct(fundamentals.roe)} />
              <Metric label="ROA" value={pct(fundamentals.roa)} />
              <Metric label="ROCE" value={pct(fundamentals.roce)} />
              <Metric
                label="Dividend Yield"
                value={pct(fundamentals.dividend_yield)}
              />
            </div>
          </div>

          {/* Balance sheet */}
          <div className="mt-6">
            <SectionTitle
              title="Balance Sheet & Financial Strength"
              description="Balance-sheet values and liquidity/leverage metrics returned by the backend."
            />

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric
                label="Quick Ratio"
                value={num(fundamentals.quick_ratio)}
              />
              <Metric
                label="Debt / Equity"
                value={num(fundamentals.debt_to_equity)}
                description="Total liabilities ÷ equity capital"
              />
              <Metric
                label="Total Assets"
                value={compactMoneyCr(fundamentals.total_assets)}
              />
              <Metric
                label="Current Assets"
                value={compactMoneyCr(fundamentals.current_assets)}
              />
              <Metric
                label="Current Liabilities"
                value={compactMoneyCr(fundamentals.current_liabilities)}
              />
              <Metric
                label="Non-current Liabilities"
                value={compactMoneyCr(fundamentals.non_current_liabilities)}
              />
              <Metric
                label="Equity Capital"
                value={compactMoneyCr(fundamentals.equity_capital)}
              />
            </div>
          </div>

          {/* Data source */}
          <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950/40 p-4 text-xs leading-5 text-slate-500">
            <p>
              <strong className="text-slate-300">Data source:</strong>{" "}
              {data.source || "Upstox Fundamentals APIs"}
            </p>

            <p className="mt-1">
              Fundamental period:{" "}
              <span className="text-slate-400">
                {data.fundamental_period || "—"}
              </span>
              {" · "}
              Type:{" "}
              <span className="text-slate-400">
                {data.fundamental_type || "—"}
              </span>
            </p>

            {data.balance_sheet_period ? (
              <p className="mt-1">
                Balance sheet:{" "}
                <span className="text-slate-400">
                  {data.balance_sheet_period}
                  {data.balance_sheet_type
                    ? ` · ${data.balance_sheet_type}`
                    : ""}
                </span>
              </p>
            ) : null}

            {data.note ? (
              <p className="mt-2 text-slate-600">{data.note}</p>
            ) : null}

            <p className="mt-2 text-slate-600">
              A missing metric is displayed as — rather than being estimated
              by the frontend.
            </p>
          </div>
        </>
      ) : null}
    </section>
  );
}
