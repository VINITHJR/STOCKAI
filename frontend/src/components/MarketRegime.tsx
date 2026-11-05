import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getMarketActivity,
  getMarketOverview,
  getMarketStatus,
  type MarketActivity,
  type MarketOverview,
  type MarketStatus,
} from "../services/marketApi";

type Regime = "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish" | "Volatile";

type RegimeAnalysis = {
  regime: Regime;
  averageIndexChange: number;
  breadthRatio: number;
  vixChange: number;
  positiveIndexes: number;
  negativeIndexes: number;
  rationale: string;
};

function numberOrZero(value: number | null | undefined) {
  return value != null && Number.isFinite(Number(value)) ? Number(value) : 0;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function changeText(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function changeClass(value: number | null | undefined) {
  const n = numberOrZero(value);
  if (n > 0) return "text-emerald-400";
  if (n < 0) return "text-red-400";
  return "text-slate-400";
}

function regimeClass(regime: Regime) {
  if (regime === "Strong Bullish" || regime === "Bullish") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-300";
  if (regime === "Strong Bearish" || regime === "Bearish") return "border-red-500/40 bg-red-500/10 text-red-300";
  if (regime === "Volatile") return "border-amber-500/40 bg-amber-500/10 text-amber-300";
  return "border-slate-600 bg-slate-800/60 text-slate-300";
}

function calculateRegime(market: MarketOverview, activity: MarketActivity): RegimeAnalysis {
  const indexChanges = [
    numberOrZero(market.nifty?.change),
    numberOrZero(market.sensex?.change),
    numberOrZero(market.bank_nifty?.change),
  ];

  const averageIndexChange = indexChanges.reduce((sum, value) => sum + value, 0) / indexChanges.length;
  const positiveIndexes = indexChanges.filter((value) => value > 0).length;
  const negativeIndexes = indexChanges.filter((value) => value < 0).length;
  const breadthRatio = activity.breadth?.total > 0
    ? (activity.breadth.advances / activity.breadth.total) * 100
    : 50;
  const vixChange = numberOrZero(market.india_vix?.change);

  // Rule-based classification only. This describes current market conditions;
  // it does not create a stock-specific trade signal or prediction.
  let regime: Regime = "Neutral";

  if (Math.abs(averageIndexChange) < 0.3 && Math.abs(vixChange) >= 5) {
    regime = "Volatile";
  } else if (averageIndexChange >= 1 && breadthRatio >= 60 && positiveIndexes >= 2) {
    regime = "Strong Bullish";
  } else if (averageIndexChange <= -1 && breadthRatio <= 40 && negativeIndexes >= 2) {
    regime = "Strong Bearish";
  } else if (averageIndexChange >= 0.3 && breadthRatio >= 52 && positiveIndexes >= 2) {
    regime = "Bullish";
  } else if (averageIndexChange <= -0.3 && breadthRatio <= 48 && negativeIndexes >= 2) {
    regime = "Bearish";
  }

  const direction = averageIndexChange > 0.05 ? "indexes are positive" : averageIndexChange < -0.05 ? "indexes are negative" : "indexes are mixed/flat";
  const breadth = activity.breadth?.total
    ? `${activity.breadth.advances} advances vs ${activity.breadth.declines} declines`
    : "breadth unavailable";
  const vix = `India VIX ${changeText(vixChange)}`;

  return {
    regime,
    averageIndexChange,
    breadthRatio,
    vixChange,
    positiveIndexes,
    negativeIndexes,
    rationale: `${direction}; ${breadth}; ${vix}.`,
  };
}

export default function MarketRegime() {
  const [market, setMarket] = useState<MarketOverview | null>(null);
  const [activity, setActivity] = useState<MarketActivity | null>(null);
  const [status, setStatus] = useState<MarketStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async (refresh = false) => {
    try {
      if (refresh) setRefreshing(true); else setLoading(true);
      setError("");
      const [overview, marketActivity, marketStatus] = await Promise.all([
        getMarketOverview(),
        getMarketActivity(10),
        getMarketStatus("NSE"),
      ]);
      setMarket(overview);
      setActivity(marketActivity);
      setStatus(marketStatus);
    } catch (err) {
      console.error("Market regime error:", err);
      setError(err instanceof Error ? err.message : "Unable to load market regime data.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = window.setInterval(() => load(true), 60_000);
    return () => window.clearInterval(interval);
  }, [load]);

  const analysis = useMemo(() => {
    if (!market || !activity) return null;
    return calculateRegime(market, activity);
  }, [market, activity]);

  const indexRows = useMemo(() => {
    if (!market) return [];
    return [
      { label: "NIFTY 50", data: market.nifty },
      { label: "SENSEX", data: market.sensex },
      { label: "BANK NIFTY", data: market.bank_nifty },
      { label: "INDIA VIX", data: market.india_vix },
    ];
  }, [market]);

  return (
    <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm text-cyan-400">Phase 5 · Layer 4</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Market Regime</h2>
          <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">
            Describes the current broad-market environment using NIFTY 50, SENSEX, BANK NIFTY, India VIX and NSE market breadth. It does not change the stock-level Layer 2 signal or Layer 3 setup ranking.
          </p>
        </div>
        <button
          type="button"
          onClick={() => load(true)}
          disabled={loading || refreshing}
          className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:border-cyan-500 disabled:opacity-50"
        >
          {refreshing ? "Refreshing..." : "↻ Refresh"}
        </button>
      </div>

      {loading ? (
        <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/60 p-8 text-center text-sm text-slate-500">
          Loading current market regime data...
        </div>
      ) : error ? (
        <div className="mt-5 rounded-xl border border-red-900/60 bg-red-950/30 p-5 text-sm text-red-300">
          {error}
        </div>
      ) : analysis ? (
        <>
          <div className="mt-5 grid gap-3 lg:grid-cols-[1.2fr_1fr_1fr_1fr]">
            <div className={`rounded-xl border p-4 ${regimeClass(analysis.regime)}`}>
              <p className="text-xs uppercase tracking-wide opacity-70">Current regime</p>
              <p className="mt-1 text-2xl font-bold">{analysis.regime}</p>
              <p className="mt-2 text-xs leading-5 opacity-80">{analysis.rationale}</p>
            </div>
            <RegimeStat label="Index average" value={changeText(analysis.averageIndexChange)} className={changeClass(analysis.averageIndexChange)} />
            <RegimeStat label="Market breadth" value={`${formatNumber(analysis.breadthRatio, 1)}% advances`} />
            <RegimeStat label="India VIX change" value={changeText(analysis.vixChange)} className={changeClass(analysis.vixChange)} />
          </div>

          <div className="mt-5 overflow-x-auto rounded-xl border border-slate-800">
            <table className="min-w-[720px] w-full text-left text-sm">
              <thead className="border-b border-slate-800 bg-slate-950/80 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  {['Index', 'Value', 'Change', 'Open', 'High', 'Low'].map((heading) => <th key={heading} className="px-4 py-3 whitespace-nowrap">{heading}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {indexRows.map(({ label, data }) => (
                  <tr key={label} className="hover:bg-slate-800/40">
                    <td className="px-4 py-3 font-semibold text-slate-100">{label}</td>
                    <td className="px-4 py-3 text-slate-200">{formatNumber(data?.value, 2)}</td>
                    <td className={`px-4 py-3 font-semibold ${changeClass(data?.change)}`}>{changeText(data?.change)}</td>
                    <td className="px-4 py-3 text-slate-300">{formatNumber(data?.open, 2)}</td>
                    <td className="px-4 py-3 text-slate-300">{formatNumber(data?.high, 2)}</td>
                    <td className="px-4 py-3 text-slate-300">{formatNumber(data?.low, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <RegimeStat label="Advances" value={`${activity?.breadth.advances ?? 0}`} />
            <RegimeStat label="Declines" value={`${activity?.breadth.declines ?? 0}`} />
            <RegimeStat label="Unchanged" value={`${activity?.breadth.unchanged ?? 0}`} />
          </div>

          <div className="mt-4 flex flex-col gap-1 text-xs text-slate-500 sm:flex-row sm:justify-between">
            <span>NSE status: {status?.market_status || "Unavailable"}</span>
            <span>Source: Upstox live market data</span>
          </div>
        </>
      ) : null}

      <p className="mt-4 text-xs leading-5 text-slate-600">
        Regime classification is a rule-based description of current market data, not a forecast and not a stock-specific trading signal. Thresholds are transparent and can be adjusted later without changing Layers 2 or 3.
      </p>
    </section>
  );
}

function RegimeStat({ label, value, className = "text-slate-100" }: { label: string; value: string; className?: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 text-lg font-semibold ${className}`}>{value}</p>
    </div>
  );
}
