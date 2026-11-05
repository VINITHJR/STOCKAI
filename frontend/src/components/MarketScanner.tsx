import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getIntraday,
  getHistory,
  getMarketScanner,
  type MarketScannerStock,
  type HistoryCandle,
} from "../services/marketApi";
import { calculateTechnicalSummary, type TechnicalSummary } from "./LiveMarketChart";
import MarketRegime from "./MarketRegime";
import SectorStrength from "./SectorStrength";
import FundamentalAnalysis from "./FundamentalAnalysis";
import NewsEventAnalysis from "./NewsEventAnalysis";

type SortBy = "volume" | "change" | "value";
type SetupType =
  | "Breakout"
  | "Breakdown"
  | "Pullback"
  | "Rejection"
  | "Trend Continuation"
  | "Oversold Watch"
  | "Overbought Watch"
  | "No Clear Setup";

type Opportunity = {
  score: number;
  setup: SetupType;
  rank: number;
  reason: string;
};

type TechnicalRow = MarketScannerStock & {
  technical?: TechnicalSummary | null;
  technicalError?: string;
};

function formatPrice(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `₹${Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
function formatVolume(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString("en-IN");
}
function signalClass(signal?: string) {
  if (signal === "BUY") return "text-emerald-400";
  if (signal === "SELL") return "text-red-400";
  return "text-amber-300";
}
function toRawCandles(candles: HistoryCandle[]) {
  return candles.map((c) => ({
    time: Math.floor(new Date(c.timestamp).getTime() / 1000),
    open: Number(c.open), high: Number(c.high), low: Number(c.low), close: Number(c.close), volume: Number(c.volume ?? 0),
  })).filter((c) => Number.isFinite(c.time) && Number.isFinite(c.close));
}

function nearLevel(price: number | null, level: number | null, tolerance = 0.012) {
  if (price == null || level == null || !Number.isFinite(price) || !Number.isFinite(level) || price <= 0) return false;
  return Math.abs(price - level) / price <= tolerance;
}

function deriveOpportunity(technical: TechnicalSummary | null | undefined): Opportunity | null {
  if (!technical) return null;

  const price = technical.price;
  const nearSupport = nearLevel(price, technical.support1);
  const nearResistance = nearLevel(price, technical.resistance1);
  const volumeConfirmed = technical.volumeConfirmation === "Confirmed";
  const volumeWeak = technical.volumeConfirmation === "Weak";
  const bullish = technical.bias === "Bullish";
  const bearish = technical.bias === "Bearish";

  let setup: SetupType = "No Clear Setup";
  let reason = "No clear technical setup is confirmed yet.";
  let setupBonus = 0;

  // Breakout/breakdown are only classified when the price has crossed the
  // relevant level and volume confirms the move. Merely touching a level is
  // treated as a watch condition, not a completed breakout/breakdown.
  if (price != null && technical.resistance1 != null && price > technical.resistance1 && volumeConfirmed && bullish) {
    setup = "Breakout";
    reason = "Price is above resistance with bullish bias and confirmed volume.";
    setupBonus = 18;
  } else if (price != null && technical.support1 != null && price < technical.support1 && volumeConfirmed && bearish) {
    setup = "Breakdown";
    reason = "Price is below support with bearish bias and confirmed volume.";
    setupBonus = 18;
  } else if (bullish && nearSupport && technical.vwapStatus !== "Below VWAP" && !volumeWeak) {
    setup = "Pullback";
    reason = "Bullish structure is near support without weak volume confirmation.";
    setupBonus = 14;
  } else if (bearish && nearResistance && technical.vwapStatus !== "Above VWAP" && !volumeWeak) {
    setup = "Rejection";
    reason = "Bearish structure is near resistance without weak volume confirmation.";
    setupBonus = 14;
  } else if (technical.signal !== "HOLD" && technical.score >= 60 && !volumeWeak) {
    setup = "Trend Continuation";
    reason = `${technical.signal} bias has a directional score of ${technical.score}/100 with no weak-volume warning.`;
    setupBonus = 10;
  } else if (technical.rsiStatus === "Oversold" && nearSupport) {
    setup = "Oversold Watch";
    reason = "RSI is oversold and price is near support; confirmation is still required.";
    setupBonus = 7;
  } else if (technical.rsiStatus === "Overbought" && nearResistance) {
    setup = "Overbought Watch";
    reason = "RSI is overbought and price is near resistance; chasing risk is elevated.";
    setupBonus = 5;
  }

  let score = technical.score;
  if (setup !== "No Clear Setup") score += setupBonus;
  if (volumeConfirmed) score += 5;
  if (volumeWeak) score -= 8;
  if (technical.dataQuality === "Low") score -= 10;
  if (technical.riskLevel === "High") score -= 5;
  if (technical.riskReward != null && technical.riskReward >= 2) score += 5;
  else if (technical.riskReward != null && technical.riskReward < 1) score -= 8;

  return {
    score: Math.max(0, Math.min(100, Math.round(score))),
    setup,
    rank: 0,
    reason,
  };
}

function setupClass(setup: SetupType) {
  if (setup === "Breakout" || setup === "Trend Continuation" || setup === "Pullback") return "text-emerald-400";
  if (setup === "Breakdown" || setup === "Rejection") return "text-red-400";
  if (setup === "Oversold Watch" || setup === "Overbought Watch") return "text-amber-300";
  return "text-slate-400";
}

export default function MarketScanner() {
  const [stocks, setStocks] = useState<TechnicalRow[]>([]);
  const [totalUniverse, setTotalUniverse] = useState(0);
  const [quoted, setQuoted] = useState(0);
  const [sortBy, setSortBy] = useState<SortBy>("volume");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(0);
  const [error, setError] = useState("");
  const abortRef = useRef(false);

  const loadScanner = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true); else setLoading(true);
      setError("");
      const data = await getMarketScanner(50, sortBy);
      if (data.status !== "success") throw new Error(data.message || "Unable to load market scanner.");
      setStocks(Array.isArray(data.stocks) ? data.stocks : []);
      setTotalUniverse(Number(data.total_universe || 0));
      setQuoted(Number(data.quoted_instruments || 0));
      setAnalyzed(0);
    } catch (err) {
      console.error("Market scanner error:", err);
      setStocks([]);
      setError(err instanceof Error ? err.message : "Unable to load the market scanner.");
    } finally { setLoading(false); setRefreshing(false); }
  }, [sortBy]);

  useEffect(() => {
    loadScanner();
    const interval = window.setInterval(() => loadScanner(true), 60_000);
    return () => window.clearInterval(interval);
  }, [loadScanner]);

  const analyzeTechnical = useCallback(async () => {
    if (analyzing || stocks.length === 0) return;
    abortRef.current = false;
    setAnalyzing(true); setAnalyzed(0); setError("");
    const next = [...stocks];
    const batchSize = 3;

    const formatDate = (date: Date) => {
      const y = date.getFullYear();
      const m = String(date.getMonth() + 1).padStart(2, "0");
      const d = String(date.getDate()).padStart(2, "0");
      return `${y}-${m}-${d}`;
    };

    const latestTradingSession = (candles: HistoryCandle[]) => {
      if (!candles.length) return [];
      const dates = candles.map((c) => new Date(c.timestamp).toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" }));
      const latest = dates.sort().at(-1);
      if (!latest) return [];
      return candles.filter((c) => new Date(c.timestamp).toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" }) === latest);
    };

    const analyzeOne = async (stock: TechnicalRow): Promise<TechnicalRow> => {
      try {
        // First use the same intraday endpoint as Stock Details.
        let candles = await getIntraday(stock.instrument_key, false, "minutes", "1");

        // On weekends/market-closed periods Upstox can return no current intraday
        // candles. Fall back to recent 1-minute history and use the latest
        // completed NSE session. This does not change the technical engine.
        if (candles.length < 20) {
          const to = new Date();
          const from = new Date(to);
          from.setDate(from.getDate() - 7);
          const history = await getHistory(
            stock.instrument_key,
            false,
            "minutes",
            "1",
            formatDate(from),
            formatDate(to),
          );
          candles = latestTradingSession(history);
        }

        if (candles.length < 20) {
          return { ...stock, technical: null, technicalError: `Only ${candles.length} usable candles available` };
        }

        const raw = toRawCandles(candles);
        const summary = calculateTechnicalSummary(raw, false, [], stock.value);
        return summary ? { ...stock, technical: summary, technicalError: undefined } : { ...stock, technical: null, technicalError: "Technical summary unavailable" };
      } catch (err) {
        return { ...stock, technical: null, technicalError: err instanceof Error ? err.message : "Analysis failed" };
      }
    };

    try {
      for (let start = 0; start < next.length; start += batchSize) {
        if (abortRef.current) break;
        const batch = next.slice(start, start + batchSize);
        const results = await Promise.all(batch.map(analyzeOne));
        results.forEach((result) => {
          const index = next.findIndex((item) => item.instrument_key === result.instrument_key);
          if (index >= 0) next[index] = result;
        });
        setStocks([...next]);
        setAnalyzed(Math.min(start + batch.length, next.length));
      }
    } finally { setAnalyzing(false); }
  }, [analyzing, stocks]);

  const filteredStocks = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return stocks;
    return stocks.filter((stock) => [stock.symbol, stock.name].filter(Boolean).some((v) => v.toLowerCase().includes(query)));
  }, [search, stocks]);

  const handleSelectStock = useCallback((stock: MarketScannerStock) => {
    window.dispatchEvent(
      new CustomEvent("stockai:selected-stock", {
        detail: {
          symbol: stock.symbol,
          name: stock.name,
          instrument_key: stock.instrument_key,
          exchange: stock.exchange,
          price: stock.value,
        },
      })
    );
  }, []);

  const opportunityRows = useMemo(() => {
    const rows = stocks
      .map((stock) => ({ stock, opportunity: deriveOpportunity(stock.technical) }))
      .filter((row): row is { stock: TechnicalRow; opportunity: Opportunity } => row.opportunity !== null)
      .sort((a, b) => b.opportunity.score - a.opportunity.score);
    return rows.map((row, index) => ({ ...row, opportunity: { ...row.opportunity, rank: index + 1 } }));
  }, [stocks]);

  return <section id="scanner" className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
      <div><p className="text-sm text-cyan-400">Phase 5 · Market Scanner · Layer 2</p><h2 className="mt-1 text-2xl font-semibold text-white">NSE Equity Scanner + Technical Engine</h2>
        <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">The scanner uses the existing StockAI technical decision engine. It does not create a second BUY/HOLD/SELL scoring system.</p></div>
      <div className="flex gap-2"><button type="button" onClick={() => loadScanner(true)} disabled={loading || refreshing || analyzing} className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:border-cyan-500 disabled:opacity-50">{refreshing ? "Refreshing..." : "↻ Refresh"}</button>
      <button type="button" onClick={analyzeTechnical} disabled={loading || refreshing || analyzing || stocks.length === 0} className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-50">{analyzing ? `Analyzing ${analyzed}/${stocks.length}...` : "Analyze Technical"}</button></div>
    </div>
    <div className="mt-5 grid gap-3 sm:grid-cols-3"><ScannerStat label="NSE Universe" value={totalUniverse.toLocaleString("en-IN")} /><ScannerStat label="Quoted" value={quoted.toLocaleString("en-IN")} /><ScannerStat label="Displayed" value={filteredStocks.length.toLocaleString("en-IN")} /></div>
    <div className="mt-5 flex flex-col gap-3 md:flex-row"><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search symbol or company..." className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-500" />
      <select value={sortBy} onChange={(e) => setSortBy(e.target.value as SortBy)} disabled={analyzing} className="rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-slate-200 outline-none focus:border-cyan-500"><option value="volume">Sort: Volume</option><option value="change">Sort: Change</option><option value="value">Sort: Price</option></select></div>
    <div className="mt-5 overflow-x-auto rounded-xl border border-slate-800">
      {loading ? <div className="p-8 text-center text-sm text-slate-500">Loading NSE market scanner...</div> : error ? <div className="p-8 text-center text-sm text-red-400">{error}</div> :
      <table className="min-w-[1180px] w-full text-left text-sm"><thead className="border-b border-slate-800 bg-slate-950/80 text-xs uppercase tracking-wide text-slate-500"><tr>
        {['#','Stock','Price','Change','Volume','RSI','Trend','MACD','VWAP','Score','Signal','Confidence','Risk','Setup','Opportunity'].map((h) => <th key={h} className="px-3 py-3 whitespace-nowrap">{h}</th>)}
      </tr></thead><tbody className="divide-y divide-slate-800">{filteredStocks.map((stock) => { const t = stock.technical; const positive = stock.change > 0; const negative = stock.change < 0; return <tr key={stock.instrument_key} className="hover:bg-slate-800/40">
        <td className="px-3 py-3 text-slate-500">{stock.rank}</td><td className="px-3 py-3 min-w-[190px]">
          <button
            type="button"
            onClick={() => handleSelectStock(stock)}
            className="text-left"
            title="Analyze this stock in Phase 5 Layers 6 and 7"
          >
            <div className="font-semibold text-cyan-300 hover:text-cyan-200">{stock.symbol}</div>
            <div className="max-w-[220px] truncate text-xs text-slate-500">{stock.name}</div>
          </button>
        </td>
        <td className="px-3 py-3 whitespace-nowrap font-medium text-slate-200">{formatPrice(stock.value)}</td><td className={`px-3 py-3 whitespace-nowrap font-semibold ${positive ? 'text-emerald-400' : negative ? 'text-red-400' : 'text-slate-400'}`}>{positive ? '+' : ''}{stock.change.toFixed(2)}%</td><td className="px-3 py-3 whitespace-nowrap text-slate-300">{formatVolume(stock.volume)}</td>
        <td className="px-3 py-3 whitespace-nowrap text-slate-200">{t?.rsi == null ? '—' : t.rsi.toFixed(1)}</td><td className="px-3 py-3 whitespace-nowrap text-slate-300">{t?.trend ?? '—'}</td><td className="px-3 py-3 whitespace-nowrap text-slate-300">{t?.macdStatus ?? '—'}</td><td className="px-3 py-3 whitespace-nowrap text-slate-300">{t?.vwapStatus ?? '—'}</td>
        <td className="px-3 py-3 whitespace-nowrap font-semibold text-slate-100">{t ? `${t.score}/100` : '—'}</td><td className={`px-3 py-3 whitespace-nowrap font-bold ${signalClass(t?.signal)}`}>{t?.signal ?? '—'}</td><td className="px-3 py-3 whitespace-nowrap text-slate-300">{t ? `${t.confidence}%` : '—'}</td><td className="px-3 py-3 whitespace-nowrap text-slate-300">{t?.riskLevel ?? '—'}</td>
        <td className={`px-3 py-3 whitespace-nowrap font-semibold ${t ? setupClass(deriveOpportunity(t)?.setup ?? "No Clear Setup") : 'text-slate-500'}`}>{t ? (deriveOpportunity(t)?.setup ?? 'No Clear Setup') : '—'}</td>
        <td className="px-3 py-3 whitespace-nowrap font-semibold text-cyan-300">{t ? `${deriveOpportunity(t)?.score ?? 0}/100` : '—'}</td>
      </tr>; })}</tbody></table>}
    </div>

    <div className="mt-6 rounded-xl border border-cyan-900/60 bg-slate-950/50 p-4">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-cyan-400">Phase 5 · Layer 3</p>
          <h3 className="mt-1 text-lg font-semibold text-white">Opportunity Ranking + Setup Detection</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">Derived from the existing Layer 2 technical summary. This layer does not change BUY/HOLD/SELL logic.</p>
        </div>
        <div className="text-xs text-slate-500">{opportunityRows.length} analyzed stocks{analyzing ? ` · ${analyzed}/${stocks.length} processed` : ""}</div>
      </div>

      {opportunityRows.length === 0 ? (
        <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950 px-4 py-5 text-sm text-slate-500">Run <span className="font-semibold text-slate-300">Analyze Technical</span> to calculate setup detection and opportunity ranking.</div>
      ) : (
        <div className="mt-4 overflow-x-auto rounded-lg border border-slate-800">
          <table className="min-w-[760px] w-full text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-950 text-xs uppercase tracking-wide text-slate-500"><tr>
              {['Rank','Stock','Opportunity','Setup','Signal','Score','Volume','Reason'].map((h) => <th key={h} className="px-3 py-3 whitespace-nowrap">{h}</th>)}
            </tr></thead>
            <tbody className="divide-y divide-slate-800">
              {opportunityRows.slice(0, 10).map(({ stock, opportunity }) => <tr key={`op-${stock.instrument_key}`} className="hover:bg-slate-800/40">
                <td className="px-3 py-3 font-semibold text-cyan-300">#{opportunity.rank}</td>
                <td className="px-3 py-3"><div className="font-semibold text-slate-100">{stock.symbol}</div><div className="text-xs text-slate-500">{stock.name}</div></td>
                <td className="px-3 py-3 font-semibold text-cyan-300">{opportunity.score}/100</td>
                <td className={`px-3 py-3 font-semibold ${setupClass(opportunity.setup)}`}>{opportunity.setup}</td>
                <td className={`px-3 py-3 font-bold ${signalClass(stock.technical?.signal)}`}>{stock.technical?.signal ?? '—'}</td>
                <td className="px-3 py-3 text-slate-300">{stock.technical?.score ?? '—'}/100</td>
                <td className="px-3 py-3 text-slate-300">{stock.technical?.volumeConfirmation ?? '—'}</td>
                <td className="min-w-[280px] px-3 py-3 text-xs leading-5 text-slate-400">{opportunity.reason}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      )}
    </div>

    <p className="mt-3 text-xs leading-5 text-slate-600">Technical analysis is rule-based market-data analysis, not a guaranteed prediction of returns. Layer 3 ranks detected setups for review; it does not guarantee future performance or change the underlying Layer 2 signal.</p>

    {/* Phase 5 · Layers 4–7 */}
    <div className="mt-8 space-y-8">
      <MarketRegime />
      <SectorStrength />

      <FundamentalAnalysis />

      <NewsEventAnalysis />
    </div>
  </section>;
}
function ScannerStat({label,value}:{label:string;value:string}) { return <div className="rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-3"><p className="text-xs text-slate-500">{label}</p><p className="mt-1 text-lg font-semibold text-slate-100">{value}</p></div>; }
