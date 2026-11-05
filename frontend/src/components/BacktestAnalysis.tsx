import { useCallback, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
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

type Trade = {
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  gross_pnl: number;
  execution_cost: number;
  pnl: number;
  return_percent: number;
};

type EquityPoint = {
  timestamp: string;
  equity: number;
  cash: number;
  position_value: number;
  signal: string;
};

type BacktestResponse = {
  success: boolean;
  layer: string;
  data_source: string;
  candle_count: number;
  data: {
    instrument_key: string;
    from_date: string;
    to_date: string;
    initial_capital: number;
    final_capital: number;
    total_pnl: number;
    total_return_percent: number;
    total_trades: number;
    total_execution_cost: number;
    strategy: {
      type: string;
      short_window: number;
      long_window: number;
      execution_rule: string;
    };
    trades: Trade[];
    equity_curve: EquityPoint[];
  };
  metrics: {
    final_capital: number;
    total_pnl: number;
    total_return_percent: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    gross_profit: number;
    gross_loss: number;
    average_win: number;
    average_loss: number;
    profit_factor: number | null;
    max_drawdown: number;
    max_drawdown_percent: number;
    cagr_percent?: number;
    sharpe_ratio?: number | null;
    equity_curve: Array<{
      timestamp: string;
      equity: number;
      drawdown: number;
      drawdown_percent: number;
    }>;
  };
  execution_costs: Record<string, number>;
};

function money(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function pct(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function dateLabel(value: string) {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "2-digit" });
}

function StatCard({ label, value, sub, positive }: { label: string; value: string; sub?: string; positive?: boolean }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-2 text-xl font-semibold ${positive === true ? "text-emerald-400" : positive === false ? "text-red-400" : "text-white"}`}>
        {value}
      </p>
      {sub && <p className="mt-1 text-[11px] text-slate-600">{sub}</p>}
    </div>
  );
}

export default function BacktestAnalysis() {
  const [symbol, setSymbol] = useState("RELIANCE");
  const [fromDate, setFromDate] = useState("2024-01-01");
  const [toDate, setToDate] = useState("2025-12-31");
  const [capital, setCapital] = useState("100000");
  const [quantity, setQuantity] = useState("10");
  const [shortWindow, setShortWindow] = useState("20");
  const [longWindow, setLongWindow] = useState("50");
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runBacktest = useCallback(async () => {
    const clean = symbol.trim().toUpperCase();
    const instrumentKey = STOCKS[clean];
    if (!instrumentKey) {
      setError(`Instrument mapping for ${clean} is not configured. Choose one of RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK or SBIN.`);
      return;
    }

    if (fromDate >= toDate) {
      setError("From date must be earlier than To date.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const response = await fetch(`${API_BASE_URL}/api/backtest/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          instrument_key: instrumentKey,
          unit: "days",
          interval: "1",
          from_date: fromDate,
          to_date: toDate,
          initial_capital: Number(capital),
          quantity: Number(quantity),
          short_window: Number(shortWindow),
          long_window: Number(longWindow),
          execution: {
            brokerage_rate: 0.0003,
            brokerage_cap_per_order: 20,
            stt_rate_sell: 0.00025,
            exchange_rate: 0.0000297,
            gst_rate: 0.18,
            sebi_rate: 0.000001,
            stamp_rate_buy: 0.00003,
            slippage_rate: 0.0002,
          },
        }),
      });

      const payload = await response.json();
      if (!response.ok || !payload.success) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail.map((x: { loc?: string[]; msg?: string }) => `${(x.loc || []).join(".")}: ${x.msg || "validation error"}`).join("; ")
          : payload.detail || payload.message || `Backtest API returned HTTP ${response.status}`;
        throw new Error(detail);
      }

      setResult(payload as BacktestResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to run backtest.");
    } finally {
      setLoading(false);
    }
  }, [symbol, fromDate, toDate, capital, quantity, shortWindow, longWindow]);

  const equityData = useMemo(() => {
    if (!result) return [];
    return result.metrics.equity_curve.map((p) => ({
      ...p,
      label: dateLabel(p.timestamp),
    }));
  }, [result]);

  return (
    <section className="mt-6 rounded-2xl border border-cyan-900/40 bg-slate-900/60 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm text-cyan-400">Phase 8 · Layer 1</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Historical Backtest Engine</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
            Validate the StockAI historical trading foundation using real Upstox daily candles, configurable SMA signals, execution costs, trade accounting and performance metrics.
          </p>
        </div>
        <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1.5 text-xs text-cyan-300">
          Backend connected
        </span>
      </div>

      <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-xs text-slate-500">Stock
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400">
            {Object.keys(STOCKS).map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        <label className="text-xs text-slate-500">From date
          <input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
        </label>
        <label className="text-xs text-slate-500">To date
          <input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
        </label>
        <label className="text-xs text-slate-500">Initial capital
          <input type="number" min="1" value={capital} onChange={(e) => setCapital(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
        </label>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-3">
        <label className="text-xs text-slate-500">Quantity
          <input type="number" min="1" value={quantity} onChange={(e) => setQuantity(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
        </label>
        <label className="text-xs text-slate-500">Short SMA
          <input type="number" min="2" value={shortWindow} onChange={(e) => setShortWindow(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
        </label>
        <label className="text-xs text-slate-500">Long SMA
          <input type="number" min="3" value={longWindow} onChange={(e) => setLongWindow(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400" />
        </label>
      </div>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <button type="button" onClick={runBacktest} disabled={loading} className="rounded-lg border border-cyan-400/60 bg-cyan-400/10 px-6 py-2.5 text-sm font-medium text-cyan-300 hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50">
          {loading ? "Running historical backtest..." : "▶ Run Backtest"}
        </button>
        <p className="text-xs text-slate-600">Execution: completed-candle signal → next-candle open</p>
      </div>

      {error && <div className="mt-5 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-3 text-sm text-red-300">{error}</div>}

      {result && !error && (
        <>
          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-3">
            <div>
              <p className="text-xs text-slate-500">Backtest result</p>
              <p className="mt-1 text-sm font-medium text-white">{symbol} · {result.candle_count.toLocaleString("en-IN")} candles</p>
            </div>
            <div className="text-right text-xs text-slate-500">Source: <span className="text-slate-300">{result.data_source}</span></div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Initial Capital" value={money(result.data.initial_capital)} />
            <StatCard label="Final Capital" value={money(result.metrics.final_capital)} />
            <StatCard label="Total Return" value={pct(result.metrics.total_return_percent)} positive={result.metrics.total_return_percent >= 0} />
            <StatCard label="Total P&L" value={money(result.metrics.total_pnl)} positive={result.metrics.total_pnl >= 0} />
            <StatCard label="Win Rate" value={`${result.metrics.win_rate.toFixed(2)}%`} />
            <StatCard label="Trades" value={String(result.metrics.total_trades)} sub={`${result.metrics.winning_trades} wins · ${result.metrics.losing_trades} losses`} />
            <StatCard label="Max Drawdown" value={pct(-Math.abs(result.metrics.max_drawdown_percent))} positive={false} sub={money(result.metrics.max_drawdown)} />
            <StatCard label="Execution Costs" value={money(result.data.total_execution_cost)} />
          </div>

          <div className="mt-5 grid gap-5 xl:grid-cols-[1.5fr_0.8fr]">
            <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs text-slate-500">Portfolio value</p>
                  <h3 className="mt-1 text-lg font-semibold text-white">Equity Curve</h3>
                </div>
                <span className="text-xs text-slate-600">Daily</span>
              </div>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={equityData}>
                    <defs>
                      <linearGradient id="backtestEquity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.28} />
                        <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={35} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} tickFormatter={(v) => `₹${Math.round(v / 1000)}k`} />
                    <Tooltip
                      formatter={(value) => [
                        money(
                          typeof value === "number"
                            ? value
                            : typeof value === "string"
                              ? Number(value)
                              : undefined
                        ),
                        "Equity",
                      ]}
                      labelFormatter={(l) => String(l)}
                    />
                    <Area type="monotone" dataKey="equity" stroke="#22d3ee" fill="url(#backtestEquity)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
              <p className="text-xs text-slate-500">Strategy configuration</p>
              <h3 className="mt-1 text-lg font-semibold text-white">Layer 1 Foundation</h3>
              <dl className="mt-5 space-y-3 text-sm">
                <div className="flex justify-between gap-4"><dt className="text-slate-500">Strategy</dt><dd className="text-right text-slate-200">SMA {result.data.strategy.short_window}/{result.data.strategy.long_window}</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-slate-500">Execution</dt><dd className="text-right text-slate-200">Next candle open</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-slate-500">Profit factor</dt><dd className="text-right text-slate-200">{result.metrics.profit_factor?.toFixed(2) ?? "—"}</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-slate-500">CAGR</dt><dd className="text-right text-slate-200">{result.metrics.cagr_percent != null ? pct(result.metrics.cagr_percent) : "—"}</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-slate-500">Sharpe</dt><dd className="text-right text-slate-200">{result.metrics.sharpe_ratio != null ? result.metrics.sharpe_ratio.toFixed(2) : "—"}</dd></div>
              </dl>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
            <div className="flex items-center justify-between gap-3">
              <div><p className="text-xs text-slate-500">Executed positions</p><h3 className="mt-1 text-lg font-semibold text-white">Trade History</h3></div>
              <span className="text-xs text-slate-600">{result.data.trades.length} completed</span>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[820px] text-sm">
                <thead><tr className="border-b border-slate-800 text-left text-xs text-slate-500"><th className="pb-3 pr-4">Entry</th><th className="pb-3 pr-4">Exit</th><th className="pb-3 pr-4 text-right">Entry</th><th className="pb-3 pr-4 text-right">Exit</th><th className="pb-3 pr-4 text-right">Qty</th><th className="pb-3 pr-4 text-right">Costs</th><th className="pb-3 text-right">Net P&L</th></tr></thead>
                <tbody>
                  {result.data.trades.map((trade, index) => (
                    <tr key={`${trade.entry_time}-${index}`} className="border-b border-slate-900 last:border-0">
                      <td className="py-3 pr-4 text-slate-300">{dateLabel(trade.entry_time)}</td><td className="py-3 pr-4 text-slate-300">{dateLabel(trade.exit_time)}</td><td className="py-3 pr-4 text-right text-slate-300">{money(trade.entry_price)}</td><td className="py-3 pr-4 text-right text-slate-300">{money(trade.exit_price)}</td><td className="py-3 pr-4 text-right text-slate-400">{trade.quantity}</td><td className="py-3 pr-4 text-right text-slate-500">{money(trade.execution_cost)}</td><td className={`py-3 text-right font-semibold ${trade.pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>{money(trade.pnl)}<span className="ml-2 text-xs">{pct(trade.return_percent)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.data.trades.length === 0 && <div className="px-4 py-8 text-center text-sm text-slate-500">No completed trades were generated for this period.</div>}
            </div>
          </div>

          <p className="mt-4 text-[11px] leading-5 text-slate-600">
            Historical results are a backtest, not live trading performance. Transaction costs are configurable and the foundation strategy uses completed candles with next-candle-open execution to avoid look-ahead bias.
          </p>
        </>
      )}
    </section>
  );
}
