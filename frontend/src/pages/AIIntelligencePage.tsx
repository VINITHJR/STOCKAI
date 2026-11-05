import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  ChevronDown,
  Gauge,
  Layers3,
  Newspaper,
  RefreshCw,
  Search,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  XCircle,
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type AnyRecord = Record<string, any>;

type SearchItem = {
  symbol: string;
  name?: string;
  instrument_key: string;
  exchange?: string;
  instrument_type?: string;
};

const QUICK_STOCKS: SearchItem[] = [
  { symbol: "RELIANCE", name: "Reliance Industries", instrument_key: "NSE_EQ|INE002A01018" },
  { symbol: "TCS", name: "Tata Consultancy Services", instrument_key: "NSE_EQ|INE467B01029" },
  { symbol: "INFY", name: "Infosys", instrument_key: "NSE_EQ|INE009A01021" },
  { symbol: "HDFCBANK", name: "HDFC Bank", instrument_key: "NSE_EQ|INE040A01034" },
  { symbol: "ICICIBANK", name: "ICICI Bank", instrument_key: "NSE_EQ|INE090A01021" },
  { symbol: "SBIN", name: "State Bank of India", instrument_key: "NSE_EQ|INE062A01020" },
];

async function getJson(path: string): Promise<AnyRecord> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Accept: "application/json" },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload?.status === "error") {
    throw new Error(
      payload?.message ||
        payload?.detail ||
        `API request failed: HTTP ${response.status}`
    );
  }
  return payload;
}

function pct(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—";
}

function num(value: unknown, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function money(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n)
    ? `₹${n.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`
    : "—";
}

function signalTone(signal: string | undefined) {
  if (signal === "bullish") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
  if (signal === "bearish") return "border-red-500/30 bg-red-500/10 text-red-300";
  return "border-amber-500/30 bg-amber-500/10 text-amber-300";
}

function SignalBadge({ signal }: { signal?: string }) {
  const normalized = signal || "unknown";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold uppercase ${signalTone(
        normalized
      )}`}
    >
      {normalized === "bullish" ? (
        <TrendingUp size={14} />
      ) : normalized === "bearish" ? (
        <TrendingDown size={14} />
      ) : (
        <Activity size={14} />
      )}
      {normalized}
    </span>
  );
}

function Metric({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-2 text-lg font-semibold text-white">{value}</p>
      {sub && <p className="mt-1 text-[11px] text-slate-500">{sub}</p>}
    </div>
  );
}

function Section({
  icon,
  title,
  subtitle,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-xl shadow-black/10">
      <div className="mb-5 flex items-start gap-3">
        <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-300">
          {icon}
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-sm text-red-300">
      <XCircle size={17} className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export default function AIIntelligencePage() {
  const [symbol, setSymbol] = useState("RELIANCE");
  const [instrumentKey, setInstrumentKey] = useState(QUICK_STOCKS[0].instrument_key);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchItem[]>([]);
  const [searchOpen, setSearchOpen] = useState(false);

  const [layer1, setLayer1] = useState<AnyRecord | null>(null);
  const [layer2, setLayer2] = useState<AnyRecord | null>(null);
  const [layer3, setLayer3] = useState<AnyRecord | null>(null);
  const [layer4, setLayer4] = useState<AnyRecord | null>(null);
  const [layer5, setLayer5] = useState<AnyRecord | null>(null);

  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const filteredQuick = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return QUICK_STOCKS;
    return QUICK_STOCKS.filter(
      (s) =>
        s.symbol.toLowerCase().includes(q) ||
        (s.name || "").toLowerCase().includes(q)
    );
  }, [query]);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults([]);
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        const response = await getJson(`/api/search?query=${encodeURIComponent(q)}`);
        const data = Array.isArray(response.data) ? response.data : [];
        setResults(
          data
            .filter((item: SearchItem) => item.instrument_key?.startsWith("NSE_EQ|"))
            .slice(0, 8)
        );
      } catch {
        setResults([]);
      }
    }, 350);

    return () => window.clearTimeout(timer);
  }, [query]);

  const selectStock = (stock: SearchItem) => {
    setSymbol(stock.symbol);
    setInstrumentKey(stock.instrument_key);
    setQuery("");
    setResults([]);
    setSearchOpen(false);
  };

  const runAnalysis = async () => {
    setLoading(true);
    setErrors({});
    setLayer1(null);
    setLayer2(null);
    setLayer3(null);
    setLayer4(null);
    setLayer5(null);

    const key = encodeURIComponent(instrumentKey);

    const requests: Array<[string, string, (v: AnyRecord) => void]> = [
      ["layer1", `/api/ai/signal/run?instrument_key=${key}`, setLayer1],
      ["layer2", `/api/ai/ensemble/run?instrument_key=${key}`, setLayer2],
      [
        "layer3",
        `/api/ai/context/run?instrument_key=${key}&symbol=${encodeURIComponent(symbol)}`,
        setLayer3,
      ],
      ["layer4", `/api/ai/regime/run?instrument_key=${key}`, setLayer4],
      [
        "layer5",
        `/api/ai/decision/run?instrument_key=${key}&symbol=${encodeURIComponent(symbol)}`,
        setLayer5,
      ],
    ];

    const settled = await Promise.allSettled(
      requests.map(async ([name, path, setter]) => {
        const data = await getJson(path);
        setter(data);
        return name;
      })
    );

    const nextErrors: Record<string, string> = {};
    settled.forEach((result, index) => {
      if (result.status === "rejected") {
        nextErrors[requests[index][0]] =
          result.reason instanceof Error
            ? result.reason.message
            : "Request failed";
      }
    });
    setErrors(nextErrors);
    setLoading(false);
  };

  useEffect(() => {
    runAnalysis();
    // Initial analysis only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const decision = layer5?.signal;
  const decisionConfidence = layer5?.confidence;

  const models = Array.isArray(layer2?.models) ? layer2.models : [];
  const lstmPredictions =
    layer2?.source_outputs?.lstm?.predictions ||
    layer1?.source_outputs?.lstm?.predictions ||
    [];

  const regimeMetrics = layer4?.metrics || {};
  const fundamentals = layer3?.context?.fundamentals || layer3?.decision_context?.fundamentals;
  const news = layer3?.context?.news || layer3?.decision_context?.news;

  return (
    <div className="min-h-screen bg-slate-950 text-white">

      <main className="mx-auto max-w-7xl px-6 py-8">
        <section className="mb-7">
          <div className="flex items-center gap-2 text-sm text-cyan-400">
            <Brain size={17} />
            Phase 9 · Advanced AI Intelligence
          </div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight md:text-4xl">
            AI Intelligence
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
            Unified view of the trained models, ensemble analysis, contextual
            signals, market regime, and final AI decision for the selected NSE stock.
          </p>
        </section>

        <section className="mb-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end">
            <div className="relative flex-1">
              <label className="text-xs font-medium text-slate-500">
                Search NSE stock
              </label>
              <div className="relative mt-2">
                <Search
                  size={17}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                />
                <input
                  value={query}
                  onFocus={() => setSearchOpen(true)}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setSearchOpen(true);
                  }}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 py-3 pl-10 pr-10 text-sm text-white outline-none focus:border-cyan-400"
                  placeholder="Search RELIANCE, TCS, INFY..."
                />
                <ChevronDown
                  size={17}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500"
                />
              </div>

              {searchOpen && (query || results.length > 0) && (
                <div className="absolute z-40 mt-2 w-full overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
                  {(results.length ? results : filteredQuick).map((stock) => (
                    <button
                      key={stock.instrument_key}
                      type="button"
                      onMouseDown={() => selectStock(stock)}
                      className="flex w-full items-center justify-between border-b border-slate-800 px-4 py-3 text-left hover:bg-slate-800"
                    >
                      <div>
                        <p className="text-sm font-semibold text-white">{stock.symbol}</p>
                        <p className="text-xs text-slate-500">{stock.name || "NSE Equity"}</p>
                      </div>
                      <span className="text-[11px] text-slate-600">NSE</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 lg:min-w-64">
              <p className="text-[11px] uppercase tracking-wide text-slate-600">Selected stock</p>
              <p className="mt-1 font-semibold text-white">{symbol}</p>
              <p className="mt-0.5 text-[10px] text-slate-600">{instrumentKey}</p>
            </div>

            <button
              type="button"
              onClick={runAnalysis}
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Analyzing..." : "Run AI Analysis"}
            </button>
          </div>
        </section>

        {loading && (
          <div className="mb-6 rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4 text-sm text-cyan-300">
            Running Phase 9 Layers 1–5 for <strong>{symbol}</strong>...
          </div>
        )}

        {/* Final decision */}
        <section className={`mb-6 rounded-2xl border p-6 ${signalTone(decision)}`}>
          <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] opacity-70">
                <Gauge size={15} />
                Layer 5 · AI Decision Intelligence
              </div>
              <h2 className="mt-2 text-3xl font-bold">
                {decision ? decision.toUpperCase() : "—"}
              </h2>
              <p className="mt-2 max-w-2xl text-sm opacity-80">
                Final cross-layer synthesis. Confidence is a contextual consensus
                measure, not a probability of future return.
              </p>
            </div>
            <div className="rounded-2xl border border-current/20 bg-slate-950/30 px-6 py-5 text-center">
              <p className="text-xs opacity-70">Confidence</p>
              <p className="mt-1 text-3xl font-bold">
                {decisionConfidence != null ? pct(decisionConfidence) : "—"}
              </p>
            </div>
          </div>
        </section>

        <div className="grid gap-6 xl:grid-cols-2">
          <Section
            icon={<Brain size={20} />}
            title="Layer 1 — AI Signal Intelligence"
            subtitle="Random Forest + LSTM model outputs"
          >
            {errors.layer1 && <ErrorBox message={errors.layer1} />}
            {layer1 && (
              <>
                <div className="grid gap-3 sm:grid-cols-3">
                  <Metric label="Signal" value={String(layer1.signal || "—").toUpperCase()} />
                  <Metric label="Ensemble confidence" value={pct(layer1.confidence)} />
                  <Metric label="Model count" value={String(layer1.model_count ?? "—")} />
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {(layer1.models || []).map((model: AnyRecord) => (
                    <div key={model.model} className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-white">{model.model}</span>
                        <SignalBadge signal={model.signal} />
                      </div>
                      <p className="mt-3 text-sm text-slate-400">
                        Confidence: <span className="text-white">{pct(model.confidence)}</span>
                      </p>
                    </div>
                  ))}
                </div>
                {Array.isArray(layer1.model_errors) && layer1.model_errors.length > 0 && (
                  <div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-200">
                    <p className="font-semibold">Model execution warnings</p>
                    {layer1.model_errors.map((item: AnyRecord, index: number) => (
                      <p key={`${item.model || "model"}-${index}`} className="mt-1">
                        {String(item.model || "Model")}: {String(item.message || "Unknown error")}
                      </p>
                    ))}
                  </div>
                )}
              </>
            )}
          </Section>

          <Section
            icon={<Layers3 size={20} />}
            title="Layer 2 — Multi-Model Ensemble"
            subtitle="Weighted model combination and conflict detection"
          >
            {errors.layer2 && <ErrorBox message={errors.layer2} />}
            {layer2 && (
              <>
                <div className="grid gap-3 sm:grid-cols-4">
                  <Metric label="Signal" value={String(layer2.signal || "—").toUpperCase()} />
                  <Metric label="Confidence" value={pct(layer2.confidence)} />
                  <Metric label="Agreement" value={String(layer2.agreement || "—")} />
                  <Metric label="Conflict" value={layer2.conflict ? "High" : "None"} />
                </div>
                <div className="mt-4 space-y-2">
                  {models.map((model: AnyRecord) => (
                    <div key={model.model} className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/50 px-4 py-3">
                      <div className="flex items-center gap-3">
                        {model.signal === "bullish" ? <TrendingUp size={17} className="text-emerald-400" /> : <TrendingDown size={17} className="text-red-400" />}
                        <span className="text-sm font-medium text-white">{model.model}</span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-slate-400">
                        <SignalBadge signal={model.signal} />
                        <span>{pct(model.confidence)}</span>
                        <span>{pct(model.weight)} weight</span>
                      </div>
                    </div>
                  ))}
                </div>
                {lstmPredictions.length > 0 && (
                  <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    {lstmPredictions.map((p: AnyRecord) => (
                      <Metric
                        key={p.horizon_days}
                        label={`${p.horizon_days}-day LSTM`}
                        value={p.change_percent != null ? `${Number(p.change_percent) >= 0 ? "+" : ""}${num(p.change_percent)}%` : "—"}
                        sub={`${String(p.direction || "—")} · ${pct(p.confidence)}`}
                      />
                    ))}
                  </div>
                )}
              </>
            )}
          </Section>

          <Section
            icon={<Activity size={20} />}
            title="Layer 3 — Context-Aware AI"
            subtitle="Fundamental and news context around the AI signal"
          >
            {errors.layer3 && <ErrorBox message={errors.layer3} />}
            {layer3 && (
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">Fundamentals</span>
                    <SignalBadge signal={fundamentals?.signal} />
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <Metric label="Score" value={num(fundamentals?.score)} />
                    <Metric label="Factors" value={String(fundamentals?.factor_count ?? "—")} />
                    <Metric label="Positive" value={String(fundamentals?.positive_factors ?? 0)} />
                    <Metric label="Negative" value={String(fundamentals?.negative_factors ?? 0)} />
                  </div>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">News & Events</span>
                    <SignalBadge signal={news?.signal} />
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <Metric label="Recent items" value={String(news?.total_items ?? "—")} />
                    <Metric label="Confidence" value={pct(news?.confidence)} />
                    <Metric label="High risk" value={String(news?.risk_counts?.high ?? 0)} />
                    <Metric label="Medium risk" value={String(news?.risk_counts?.medium ?? 0)} />
                  </div>
                </div>
              </div>
            )}
          </Section>

          <Section
            icon={<Gauge size={20} />}
            title="Layer 4 — Market Regime"
            subtitle="Trend, momentum, volatility and volume regime"
          >
            {errors.layer4 && <ErrorBox message={errors.layer4} />}
            {layer4 && (
              <>
                <div className="flex flex-wrap items-center gap-3">
                  <SignalBadge signal={layer4.direction} />
                  <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400">
                    {String(layer4.regime || "unknown").replaceAll("_", " ")}
                  </span>
                  <span className="text-xs text-slate-500">
                    {pct(layer4.confidence)} regime confidence
                  </span>
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <Metric label="Current price" value={money(regimeMetrics.current_price)} />
                  <Metric label="SMA 20" value={money(regimeMetrics.sma_20)} />
                  <Metric label="SMA 50" value={money(regimeMetrics.sma_50)} />
                  <Metric label="20D momentum" value={`${num(regimeMetrics.momentum_20d_percent)}%`} />
                  <Metric label="ATR %" value={`${num(regimeMetrics.atr_percent)}%`} />
                  <Metric label="Annualized volatility" value={pct(regimeMetrics.volatility_annualized)} />
                </div>
              </>
            )}
          </Section>
        </div>

        <Section
          icon={<ShieldAlert size={20} />}
          title="Layer 5 — Decision Context"
          subtitle="Why the final decision is what it is"
        >
          {errors.layer5 && <ErrorBox message={errors.layer5} />}
          {layer5 && (
            <>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Metric label="Final signal" value={String(layer5.signal || "—").toUpperCase()} />
                <Metric label="Confidence" value={pct(layer5.confidence)} />
                <Metric
                  label="Supporting contexts"
                  value={String(layer5.synthesis?.supporting_contexts ?? "—")}
                />
                <Metric
                  label="Opposing contexts"
                  value={String(layer5.synthesis?.opposing_contexts ?? "—")}
                />
              </div>

              <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
                  <Newspaper size={16} />
                  AI Explanation
                </h3>
                <div className="mt-3 space-y-2">
                  {(layer5.explanation || []).map((item: string, index: number) => (
                    <div key={`${item}-${index}`} className="flex gap-2 text-sm text-slate-400">
                      <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-cyan-400" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              {layer5.decision_context?.market_regime && (
                <div className="mt-4 rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-200">
                  <div className="flex items-center gap-2 font-semibold">
                    <AlertTriangle size={16} />
                    Market regime context
                  </div>
                  <p className="mt-1 text-amber-200/70">
                    {String(layer5.decision_context.market_regime.regime).replaceAll("_", " ")}
                    {" · "}
                    {String(layer5.decision_context.market_regime.direction)}
                    {" · "}
                    {pct(layer5.decision_context.market_regime.confidence)} confidence
                  </p>
                </div>
              )}
            </>
          )}
        </Section>

        <p className="mt-6 text-center text-xs text-slate-600">
          Phase 9 frontend reads the existing AI APIs and does not retrain or modify
          the underlying ML models.
        </p>
      </main>
    </div>
  );
}
