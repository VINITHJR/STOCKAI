import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const STOCKS: Record<string, string> = {
  RELIANCE: "NSE_EQ|INE002A01018",
  TCS: "NSE_EQ|INE467B01029",
  INFY: "NSE_EQ|INE009A01021",
  HDFCBANK: "NSE_EQ|INE040A01034",
  ICICIBANK: "NSE_EQ|INE090A01021",
  SBIN: "NSE_EQ|INE062A01020",
};

const EXECUTION = {
  brokerage_rate: 0.0003,
  brokerage_cap_per_order: 20,
  stt_rate_sell: 0.00025,
  exchange_rate: 0.0000297,
  gst_rate: 0.18,
  sebi_rate: 0.000001,
  stamp_rate_buy: 0.00003,
  slippage_rate: 0.0002,
};

type Tab = "historical" | "ml" | "portfolio" | "risk" | "benchmark";
type CurvePoint = { timestamp: string; equity: number; drawdown?: number; drawdown_percent?: number };

type AnyResult = Record<string, any>;
type StockSearchItem = {
  symbol: string;
  name: string;
  instrument_key: string;
};

function money(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) ? `₹${n.toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : "—";
}

function pct(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n >= 0 ? "+" : ""}${n.toFixed(2)}%` : "—";
}

function dateLabel(value: string) {
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
}

async function postJson(path: string, body: unknown) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((x: any) => `${(x.loc || []).join(".")}: ${x.msg || "validation error"}`).join("; ")
      : payload.detail || payload.message || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return payload as AnyResult;
}

function Card({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-2 text-xl font-semibold text-white">{value}</p>
      {sub && <p className="mt-1 text-[11px] text-slate-600">{sub}</p>}
    </div>
  );
}

function Input({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (v: string) => void; type?: string }) {
  return (
    <label className="text-xs text-slate-500">
      {label}
      <input type={type} value={value} onChange={(e) => onChange(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
    </label>
  );
}

export default function Phase8Dashboard() {
  const [tab, setTab] = useState<Tab>("historical");
  const [symbol, setSymbol] = useState("RELIANCE");
  const [instrumentKey, setInstrumentKey] = useState(STOCKS.RELIANCE);
  const [stockQuery, setStockQuery] = useState("");
  const [stockResults, setStockResults] = useState<StockSearchItem[]>([]);
  const [stockSearchLoading, setStockSearchLoading] = useState(false);
  const [stockSearchOpen, setStockSearchOpen] = useState(false);
  const [fromDate, setFromDate] = useState("2024-01-01");
  const [toDate, setToDate] = useState("2025-12-31");
  const [capital, setCapital] = useState("100000");
  const [quantity, setQuantity] = useState("10");
  const [strategy, setStrategy] = useState("sma");
  const [fast, setFast] = useState("20");
  const [slow, setSlow] = useState("50");
  const [result, setResult] = useState<AnyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [mlResult, setMlResult] = useState<AnyResult | null>(null);
  const [portfolioResult, setPortfolioResult] = useState<AnyResult | null>(null);
  const [riskResult, setRiskResult] = useState<AnyResult | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<AnyResult | null>(null);
  const [benchmarkSymbol, setBenchmarkSymbol] = useState("NIFTY50");
  const [benchmarkInstrumentKey, setBenchmarkInstrumentKey] = useState("NSE_INDEX|Nifty 50");
  const [benchmarkQuery, setBenchmarkQuery] = useState("");
  const [benchmarkResults, setBenchmarkResults] = useState<StockSearchItem[]>([]);
  const [benchmarkSearchLoading, setBenchmarkSearchLoading] = useState(false);
  const [benchmarkSearchOpen, setBenchmarkSearchOpen] = useState(false);

  useEffect(() => {
    const query = stockQuery.trim();
    if (!query) {
      setStockResults([]);
      setStockSearchOpen(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        setStockSearchLoading(true);
        const response = await fetch(
          `${API_BASE_URL}/api/stocks/search?query=${encodeURIComponent(query)}`,
          { headers: { Accept: "application/json" }, signal: controller.signal }
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
        }

        const raw = Array.isArray(payload?.data) ? payload.data : [];
        const normalised: StockSearchItem[] = raw
          .map((item: any) => ({
            symbol: String(item.symbol || item.trading_symbol || item.short_name || "").trim(),
            name: String(item.name || item.company_name || item.short_name || "").trim(),
            instrument_key: String(item.instrument_key || "").trim(),
          }))
          .filter((item: StockSearchItem) => item.instrument_key)
          .slice(0, 20);

        setStockResults(normalised);
        setStockSearchOpen(true);
      } catch (e) {
        if ((e as Error)?.name !== "AbortError") {
          setStockResults([]);
          setStockSearchOpen(false);
        }
      } finally {
        setStockSearchLoading(false);
      }
    }, 300);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [stockQuery]);

  useEffect(() => {
    const query = benchmarkQuery.trim();
    if (!query) {
      setBenchmarkResults([]);
      setBenchmarkSearchOpen(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        setBenchmarkSearchLoading(true);
        const response = await fetch(
          `${API_BASE_URL}/api/stocks/search?query=${encodeURIComponent(query)}`,
          { headers: { Accept: "application/json" }, signal: controller.signal }
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
        }

        const raw = Array.isArray(payload?.data) ? payload.data : [];
        const normalised: StockSearchItem[] = raw
          .map((item: any) => ({
            symbol: String(item.symbol || item.trading_symbol || item.short_name || "").trim(),
            name: String(item.name || item.company_name || item.short_name || "").trim(),
            instrument_key: String(item.instrument_key || "").trim(),
          }))
          .filter((item: StockSearchItem) => item.instrument_key.startsWith("NSE_EQ|"))
          .slice(0, 20);

        setBenchmarkResults(normalised);
        setBenchmarkSearchOpen(true);
      } catch (e) {
        if ((e as Error)?.name !== "AbortError") {
          setBenchmarkResults([]);
          setBenchmarkSearchOpen(false);
        }
      } finally {
        setBenchmarkSearchLoading(false);
      }
    }, 300);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [benchmarkQuery]);

  const equityCurve: CurvePoint[] = useMemo(() => {
    const curve = result?.metrics?.equity_curve || result?.data?.equity_curve || [];
    return Array.isArray(curve) ? curve : [];
  }, [result]);

  const buildHistoricalRequest = () => {
    let params: Record<string, any> = {};
    if (strategy === "sma") params = { short_window: Number(fast), long_window: Number(slow) };
    if (strategy === "ema") params = { fast_window: Number(fast), slow_window: Number(slow) };
    if (strategy === "rsi") params = { window: Number(fast), oversold: 30, overbought: 70 };
    if (strategy === "macd") params = { fast_window: 12, slow_window: 26, signal_window: 9 };
    if (strategy === "bollinger") params = { window: Number(fast), standard_deviations: 2 };

    return {
      instrument_key: instrumentKey,
      unit: "days",
      interval: "1",
      from_date: fromDate,
      to_date: toDate,
      initial_capital: Number(capital),
      quantity: Number(quantity),
      short_window: Number(fast),
      long_window: Number(slow),
      strategy: { name: strategy, params },
      execution: EXECUTION,
    };
  };

  const runHistorical = async () => {
    if (!instrumentKey || fromDate >= toDate) {
      setError("Choose a valid stock and a date range where From date is earlier than To date.");
      return;
    }
    try {
      setLoading(true);
      setError("");
      setResult(null);
      const payload = await postJson("/api/backtest/run", buildHistoricalRequest());
      setResult(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Historical backtest failed.");
    } finally {
      setLoading(false);
    }
  };

  const runML = async () => {
    try {
      setLoading(true); setError(""); setMlResult(null);
      const payload = await postJson("/api/backtest/ml/run", {
        instrument_key: instrumentKey, unit: "days", interval: "1", from_date: fromDate, to_date: toDate,
        initial_capital: Number(capital), quantity: Number(quantity), forward_days: 5, execution: EXECUTION,
      });
      setMlResult(payload);
    } catch (e) { setError(e instanceof Error ? e.message : "ML backtest failed."); }
    finally { setLoading(false); }
  };

  const runPortfolio = async () => {
    const second = symbol === "RELIANCE" ? "TCS" : "RELIANCE";
    try {
      setLoading(true); setError(""); setPortfolioResult(null);
      const payload = await postJson("/api/backtest/portfolio/run", {
        assets: [
          { instrument_key: instrumentKey, weight: 0.5, strategy: "sma", params: { short_window: 20, long_window: 50 }, quantity: 1 },
          { instrument_key: STOCKS[second], weight: 0.5, strategy: "ema", params: { fast_window: 12, slow_window: 26 }, quantity: 1 },
        ], unit: "days", interval: "1", from_date: fromDate, to_date: toDate,
        initial_capital: Number(capital), execution: EXECUTION,
      });
      setPortfolioResult(payload);
    } catch (e) { setError(e instanceof Error ? e.message : "Portfolio backtest failed."); }
    finally { setLoading(false); }
  };

  const runRisk = async () => {
    if (equityCurve.length < 2) { setError("Run a Historical Backtest first so Layer 6 can analyze its equity curve."); return; }
    try {
      setLoading(true); setError("");
      const payload = await postJson("/api/backtest/risk/run", { initial_capital: Number(capital), equity_curve: equityCurve });
      setRiskResult(payload);
      setTab("risk");
    } catch (e) { setError(e instanceof Error ? e.message : "Risk analysis failed."); }
    finally { setLoading(false); }
  };

  const runBenchmark = async () => {
    if (!instrumentKey || !benchmarkInstrumentKey) {
      setError("Choose both a strategy stock and a benchmark stock.");
      return;
    }
    if (fromDate >= toDate) {
      setError("Choose a valid date range where From date is earlier than To date.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setBenchmarkResult(null);

      // Layer 7 needs the strategy equity curve from Layer 1.
      // If the user has not run Layer 1 in this browser session,
      // run the same Layer 1 request automatically instead of
      // stopping with the old "Run a Historical Backtest first" error.
      let strategyCurve = equityCurve;

      if (strategyCurve.length < 2) {
        const historicalPayload = await postJson(
          "/api/backtest/run",
          buildHistoricalRequest()
        );

        setResult(historicalPayload);

        const generatedCurve =
          historicalPayload?.metrics?.equity_curve ||
          historicalPayload?.data?.equity_curve ||
          [];

        strategyCurve = Array.isArray(generatedCurve) ? generatedCurve : [];
      }

      if (strategyCurve.length < 2) {
        throw new Error("Historical backtest completed, but no equity curve was returned for the benchmark comparison.");
      }

      const payload = await postJson("/api/backtest/benchmark/run", {
        strategy_equity_curve: strategyCurve.map((p) => ({
          timestamp: p.timestamp,
          equity: p.equity,
        })),
        benchmark_instrument_key: benchmarkInstrumentKey,
        unit: "days",
        interval: "1",
        from_date: fromDate,
        to_date: toDate,
        initial_capital: Number(capital),
      });

      setBenchmarkResult(payload);
      setTab("benchmark");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Benchmark comparison failed.");
    } finally {
      setLoading(false);
    }
  };

  const tabs: Array<[Tab, string]> = [
    ["historical", "Historical / Technical"], ["ml", "AI / ML"], ["portfolio", "Portfolio"], ["risk", "Risk & Drawdown"], ["benchmark", "Benchmark"],
  ];

  return (
    <section className="mt-6 rounded-2xl border border-cyan-900/40 bg-slate-900/60 p-6">
      <div className="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm text-cyan-400">Phase 8 · Backtesting Suite</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">StockAI Backtest Lab</h2>
          <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">One frontend workspace for the completed Phase 8 backend layers. Existing Layer 1–7 APIs remain unchanged.</p>
        </div>
        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-300">Backend API connected</span>
      </div>

      <div className="mt-6 flex flex-wrap gap-2 border-b border-slate-800 pb-3">
        {tabs.map(([key, label]) => (
          <button key={key} type="button" onClick={() => { setTab(key); setError(""); }} className={`rounded-lg px-3 py-2 text-xs font-medium ${tab === key ? "bg-cyan-400/15 text-cyan-300" : "text-slate-500 hover:bg-slate-800 hover:text-slate-200"}`}>{label}</button>
        ))}
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <div className="relative text-xs text-slate-500 md:col-span-2 xl:col-span-2">
          <label>
            Search stock
            <div className="relative mt-2">
              <input
                value={stockQuery}
                onChange={(e) => setStockQuery(e.target.value)}
                onFocus={() => stockQuery.trim() && setStockSearchOpen(true)}
                onBlur={() => window.setTimeout(() => setStockSearchOpen(false), 150)}
                placeholder="Search symbol or company name (e.g. RELIANCE, TATA, HDFC)"
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 pr-24 text-sm text-white outline-none focus:border-cyan-400"
              />
              {stockSearchLoading && (
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-cyan-300">
                  Searching...
                </span>
              )}
            </div>
          </label>

          {stockSearchOpen && stockQuery.trim() && (
            <div className="absolute left-0 right-0 top-full z-30 mt-1 max-h-72 overflow-auto rounded-xl border border-slate-700 bg-slate-950 shadow-2xl">
              {stockResults.length > 0 ? (
                stockResults.map((item) => (
                  <button
                    key={item.instrument_key}
                    type="button"
                    onMouseDown={() => {
                      setSymbol(item.symbol || item.instrument_key.split("|")[1] || "STOCK");
                      setInstrumentKey(item.instrument_key);
                      setStockQuery("");
                      setStockResults([]);
                      setStockSearchOpen(false);
                    }}
                    className="block w-full border-b border-slate-800 px-4 py-3 text-left last:border-b-0 hover:bg-slate-900"
                  >
                    <div className="text-sm font-medium text-white">
                      {item.symbol || "—"}
                    </div>
                    <div className="mt-0.5 truncate text-[11px] text-slate-500">
                      {item.name || "NSE Equity"} · {item.instrument_key}
                    </div>
                  </button>
                ))
              ) : (
                <div className="px-4 py-3 text-xs text-slate-500">
                  No NSE equity results found.
                </div>
              )}
            </div>
          )}

          <div className="mt-2 flex items-center gap-2">
            <span className="text-[11px] text-slate-600">Selected:</span>
            <span className="rounded-md bg-cyan-400/10 px-2 py-1 text-[11px] font-medium text-cyan-300">
              {symbol}
            </span>
            <span className="truncate text-[10px] text-slate-600">{instrumentKey}</span>
          </div>
        </div>

        <label className="text-xs text-slate-500">Quick select
          <select
            value={symbol}
            onChange={(e) => {
              const next = e.target.value;
              setSymbol(next);
              setInstrumentKey(STOCKS[next] || instrumentKey);
            }}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white"
          >
            {Object.keys(STOCKS).map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <Input label="From date" type="date" value={fromDate} onChange={setFromDate} />
        <Input label="To date" type="date" value={toDate} onChange={setToDate} />
        <Input label="Initial capital" type="number" value={capital} onChange={setCapital} />
        <Input label="Quantity" type="number" value={quantity} onChange={setQuantity} />
      </div>

      {tab === "historical" && (
        <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
          <div className="grid gap-3 md:grid-cols-3">
            <label className="text-xs text-slate-500">Strategy<select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white"><option value="sma">SMA Crossover</option><option value="ema">EMA Crossover</option><option value="rsi">RSI</option><option value="macd">MACD</option><option value="bollinger">Bollinger Bands</option></select></label>
            <Input label={strategy === "rsi" || strategy === "bollinger" ? "Window" : "Fast / Short window"} type="number" value={fast} onChange={setFast} />
            <Input label="Slow / Long window" type="number" value={slow} onChange={setSlow} />
          </div>
          <button type="button" onClick={runHistorical} disabled={loading} className="mt-4 rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 disabled:opacity-50">{loading ? "Running..." : "Run Historical Backtest"}</button>
        </div>
      )}

      {tab === "ml" && <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4"><p className="text-sm text-slate-300">Run the existing Random Forest trend-prediction replay for {symbol}.</p><button onClick={runML} disabled={loading} className="mt-4 rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 disabled:opacity-50">{loading ? "Running ML backtest..." : "Run AI / ML Backtest"}</button></div>}
      {tab === "portfolio" && <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4"><p className="text-sm text-slate-300">Run a 50/50 two-stock portfolio using SMA on the selected stock and EMA on a second stock.</p><button onClick={runPortfolio} disabled={loading} className="mt-4 rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 disabled:opacity-50">{loading ? "Running portfolio..." : "Run Portfolio Backtest"}</button></div>}
      {tab === "risk" && <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4"><p className="text-sm text-slate-300">Risk analysis uses the equity curve generated by the latest Historical Backtest.</p><button onClick={runRisk} disabled={loading} className="mt-4 rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 disabled:opacity-50">{loading ? "Calculating risk..." : "Analyze Drawdown"}</button></div>}
      {tab === "benchmark" && (
        <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="relative text-xs text-slate-500">
              <label>
                Search benchmark stock
                <div className="relative mt-2">
                  <input
                    value={benchmarkQuery}
                    onChange={(e) => setBenchmarkQuery(e.target.value)}
                    onFocus={() => benchmarkQuery.trim() && setBenchmarkSearchOpen(true)}
                    onBlur={() => window.setTimeout(() => setBenchmarkSearchOpen(false), 150)}
                    placeholder="Search any NSE stock (e.g. TATA, HDFC, INFY)"
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 pr-24 text-sm text-white outline-none focus:border-cyan-400"
                  />
                  {benchmarkSearchLoading && (
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-cyan-300">
                      Searching...
                    </span>
                  )}
                </div>
              </label>

              {benchmarkSearchOpen && benchmarkQuery.trim() && (
                <div className="absolute left-0 right-0 top-full z-30 mt-1 max-h-72 overflow-auto rounded-xl border border-slate-700 bg-slate-950 shadow-2xl">
                  {benchmarkResults.length > 0 ? (
                    benchmarkResults
                      .filter((item) => item.instrument_key !== instrumentKey)
                      .map((item) => (
                        <button
                          key={item.instrument_key}
                          type="button"
                          onMouseDown={() => {
                            setBenchmarkSymbol(item.symbol || item.instrument_key.split("|")[1] || "STOCK");
                            setBenchmarkInstrumentKey(item.instrument_key);
                            setBenchmarkQuery("");
                            setBenchmarkResults([]);
                            setBenchmarkSearchOpen(false);
                          }}
                          className="block w-full border-b border-slate-800 px-4 py-3 text-left last:border-b-0 hover:bg-slate-900"
                        >
                          <div className="text-sm font-medium text-white">{item.symbol || "—"}</div>
                          <div className="mt-0.5 truncate text-[11px] text-slate-500">
                            {item.name || "NSE Equity"} · {item.instrument_key}
                          </div>
                        </button>
                      ))
                  ) : (
                    <div className="px-4 py-3 text-xs text-slate-500">
                      No NSE equity benchmark results found.
                    </div>
                  )}
                </div>
              )}

              <div className="mt-2 flex items-center gap-2">
                <span className="text-[11px] text-slate-600">Selected:</span>
                <span className="rounded-md bg-cyan-400/10 px-2 py-1 text-[11px] font-medium text-cyan-300">
                  {benchmarkSymbol}
                </span>
                <span className="truncate text-[10px] text-slate-600">{benchmarkInstrumentKey}</span>
              </div>
            </div>

            <label className="text-xs text-slate-500">
              Quick benchmark
              <select
                value={benchmarkSymbol}
                onChange={(e) => {
                  const next = e.target.value;
                  setBenchmarkSymbol(next);
                  if (next === "NIFTY50") {
                    setBenchmarkInstrumentKey("NSE_INDEX|Nifty 50");
                  } else {
                    setBenchmarkInstrumentKey(STOCKS[next] || benchmarkInstrumentKey);
                  }
                }}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white"
              >
                <option>NIFTY50</option>
                {Object.keys(STOCKS).filter((s) => s !== symbol).map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
          </div>

          <button
            onClick={runBenchmark}
            disabled={loading || !benchmarkInstrumentKey}
            className="mt-4 rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 disabled:opacity-50"
          >
            {loading ? "Preparing comparison..." : "Compare With Benchmark"}
          </button>
        </div>
      )}

      {error && <div className="mt-4 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-3 text-sm text-red-300">{error}</div>}

      {tab === "historical" && result && <HistoricalResult result={result} equityCurve={equityCurve} />}
      {tab === "ml" && mlResult && <MLResult result={mlResult} />}
      {tab === "portfolio" && portfolioResult && <PortfolioResult result={portfolioResult} />}
      {tab === "risk" && riskResult && <RiskResult result={riskResult} />}
      {tab === "benchmark" && benchmarkResult && <BenchmarkResult result={benchmarkResult} />}
    </section>
  );
}

function HistoricalResult({ result, equityCurve }: { result: AnyResult; equityCurve: CurvePoint[] }) {
  const metrics = result.metrics || {};
  return <div className="mt-5 space-y-4">
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Card label="Final Capital" value={money(metrics.final_capital)} /><Card label="Total P&L" value={money(metrics.total_pnl)} /><Card label="Return" value={pct(metrics.total_return_percent)} /><Card label="Max Drawdown" value={pct(-Math.abs(Number(metrics.max_drawdown_percent || 0)))} /></div>
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4"><h3 className="text-lg font-semibold text-white">Equity Curve</h3><div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><AreaChart data={equityCurve.map((p) => ({ ...p, label: dateLabel(p.timestamp) }))}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={35} /><YAxis tick={{ fill: "#64748b", fontSize: 10 }} /><Tooltip /><Area type="monotone" dataKey="equity" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.12} /></AreaChart></ResponsiveContainer></div></div>
  </div>;
}

function MLResult({ result }: { result: AnyResult }) {
  const evaluation = result.prediction_evaluation || {};
  const predictions = result.predictions || [];
  return <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Card label="Model" value="Random Forest" /><Card label="Predictions" value={String(result.prediction_count ?? predictions.length)} /><Card label="Evaluated" value={String(evaluation.evaluated_predictions ?? 0)} /><Card label="Accuracy" value={pct(evaluation.accuracy_percent)} /></div>;
}

function PortfolioResult({ result }: { result: AnyResult }) {
  const curve = result.portfolio_equity_curve || [];
  return <div className="mt-5 space-y-4"><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Card label="Final Capital" value={money(result.final_capital)} /><Card label="P&L" value={money(result.total_pnl)} /><Card label="Return" value={pct(result.total_return_percent)} /><Card label="Trades" value={String(result.total_trades ?? 0)} /></div><div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4"><h3 className="text-lg font-semibold text-white">Portfolio Equity</h3><div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><AreaChart data={curve.map((p: any) => ({ ...p, label: dateLabel(p.timestamp) }))}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} /><YAxis tick={{ fill: "#64748b", fontSize: 10 }} /><Tooltip /><Area dataKey="equity" type="monotone" stroke="#34d399" fill="#34d399" fillOpacity={0.1} /></AreaChart></ResponsiveContainer></div></div></div>;
}

function RiskResult({ result }: { result: AnyResult }) {
  const a = result.analysis || result;
  const curve = a.drawdown_curve || a.curve || [];
  return <div className="mt-5 space-y-4"><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Card label="Maximum Drawdown" value={money(a.max_drawdown)} /><Card label="Maximum Drawdown %" value={pct(-Math.abs(Number(a.max_drawdown_percent || 0)))} /><Card label="Recovery" value={a.recovery_time_days != null ? `${a.recovery_time_days} days` : "—"} /><Card label="Recovered" value={a.recovered == null ? "—" : a.recovered ? "Yes" : "No"} /></div>{curve.length > 1 && <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4"><h3 className="text-lg font-semibold text-white">Drawdown Curve</h3><div className="mt-4 h-64"><ResponsiveContainer width="100%" height="100%"><LineChart data={curve}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="timestamp" tick={{ fill: "#64748b", fontSize: 10 }} /><YAxis tick={{ fill: "#64748b", fontSize: 10 }} /><Tooltip /><Line type="monotone" dataKey="drawdown_percent" stroke="#f87171" dot={false} /></LineChart></ResponsiveContainer></div></div>}</div>;
}

function BenchmarkResult({ result }: { result: AnyResult }) {
  const a = result.analysis || {};
  const curve = a.comparison_curve || [];
  return <div className="mt-5 space-y-4"><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Card label="Strategy Return" value={pct(a.strategy?.return_percent)} /><Card label="Benchmark Return" value={pct(a.benchmark?.return_percent)} /><Card label="Excess Return" value={pct(a.excess_return_percent)} /><Card label="Aligned Points" value={String(a.aligned_points ?? 0)} /></div>{curve.length > 1 && <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4"><h3 className="text-lg font-semibold text-white">Strategy vs Benchmark</h3><div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><LineChart data={curve.map((p: any) => ({ ...p, label: dateLabel(p.timestamp) }))}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} /><YAxis tick={{ fill: "#64748b", fontSize: 10 }} /><Tooltip /><Line type="monotone" dataKey="strategy_equity" stroke="#22d3ee" dot={false} name="Strategy" /><Line type="monotone" dataKey="benchmark_equity" stroke="#a78bfa" dot={false} name="Benchmark" /></LineChart></ResponsiveContainer></div></div>}</div>;
}
