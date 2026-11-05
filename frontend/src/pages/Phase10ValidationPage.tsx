import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Gauge,
  Layers3,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const VALIDATION_ENDPOINT =
  import.meta.env.VITE_PHASE10_VALIDATION_ENDPOINT ||
  "/api/backtest/ai/validation-dashboard";

type AnyRecord = Record<string, any>;

type StockSearchResult = {
  name?: string;
  symbol?: string;
  trading_symbol?: string;
  exchange?: string;
  instrument_key?: string;
  isin?: string;
  short_name?: string;
  instrument_type?: string;
  segment?: string;
};

type Phase10Response = {
  status?: string;
  layer?: string;
  part?: string;
  analysis?: string;
  instrument_key?: string | null;
  dashboard_status?: {
    layers_available?: number;
    layers_success?: number;
    layers_missing_or_unverified?: number;
    layer_status?: Record<string, string>;
  };
  validation_summary?: {
    layer1_historical_backtest?: AnyRecord;
    layer2_walk_forward?: AnyRecord;
    layer3_multi_period?: AnyRecord;
    layer4_signal_performance?: AnyRecord;
    layer5_ai_vs_baseline?: AnyRecord;
  };
  data_integrity?: {
    no_layer_mutation?: boolean;
    historical_outcome_source?: string;
    future_data_usage?: Record<string, number | null>;
  };
};

const QUICK_INSTRUMENTS = [
  { symbol: "RELIANCE", key: "NSE_EQ|INE002A01018" },
  { symbol: "TCS", key: "NSE_EQ|INE467B01029" },
  { symbol: "INFY", key: "NSE_EQ|INE009A01021" },
  { symbol: "HDFCBANK", key: "NSE_EQ|INE040A01034" },
  { symbol: "ICICIBANK", key: "NSE_EQ|INE090A01021" },
  { symbol: "SBIN", key: "NSE_EQ|INE062A01020" },
];

function formatValue(value: unknown, suffix = "") {
  if (value === null || value === undefined || value === "") return "—";

  if (typeof value === "number") {
    return `${value.toLocaleString("en-IN", {
      maximumFractionDigits: 4,
    })}${suffix}`;
  }

  return `${value}${suffix}`;
}

function statusIsSuccess(status?: string) {
  return status?.toLowerCase() === "success";
}

function StatusBadge({ status }: { status?: string }) {
  const success = statusIsSuccess(status);

  return (
    <span
      className={
        success
          ? "inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300"
          : "inline-flex items-center gap-1.5 rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-300"
      }
    >
      {success ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
      {success ? "Success" : status || "Unknown"}
    </span>
  );
}

function MetricRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-slate-800/80 py-2.5 last:border-b-0">
      <span className="text-xs text-slate-500">{label}</span>
      <strong className="text-right text-xs font-semibold text-slate-200">
        {value as React.ReactNode}
      </strong>
    </div>
  );
}

function MetricCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-black/10">
      <div className="mb-4 flex items-center gap-3">
        <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-300">
          {icon}
        </div>
        <h2 className="text-base font-semibold text-white">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function LayerStatus({
  name,
  status,
}: {
  name: string;
  status?: string;
}) {
  const success = statusIsSuccess(status);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
        {name}
      </div>
      <span
        className={
          success
            ? "inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-semibold text-emerald-300"
            : "inline-flex items-center gap-1.5 rounded-full border border-red-500/30 bg-red-500/10 px-2.5 py-1 text-[11px] font-semibold text-red-300"
        }
      >
        {success ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
        {status || "missing"}
      </span>
    </div>
  );
}

export default function Phase10ValidationPage() {
  const [instrumentKey, setInstrumentKey] = useState("NSE_EQ|INE002A01018");
  const [symbol, setSymbol] = useState("RELIANCE");
  const [fromDate, setFromDate] = useState("2024-01-01");
  const [toDate, setToDate] = useState("2025-12-31");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<Phase10Response | null>(null);

  const [searchQuery, setSearchQuery] = useState("RELIANCE");
  const [searchResults, setSearchResults] = useState<StockSearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearchResults, setShowSearchResults] = useState(false);

  useEffect(() => {
    const query = searchQuery.trim();

    if (query.length < 2) {
      setSearchResults([]);
      setShowSearchResults(false);
      setSearchLoading(false);
      return;
    }

    let cancelled = false;

    const timer = window.setTimeout(async () => {
      setSearchLoading(true);

      try {
        const response = await fetch(
          `${API_BASE_URL}/api/stocks/search?query=${encodeURIComponent(query)}`
        );

        const payload = await response.json().catch(() => null);

        if (!response.ok) {
          throw new Error(
            payload?.detail ||
              payload?.message ||
              `Stock search returned HTTP ${response.status}`
          );
        }

        const rawResults = Array.isArray(payload)
          ? payload
          : Array.isArray(payload?.data)
            ? payload.data
            : Array.isArray(payload?.results)
              ? payload.results
              : [];

        const results = rawResults
          .map((item: AnyRecord) => ({
            name: item.name,
            symbol: item.symbol || item.trading_symbol || item.short_name,
            trading_symbol: item.trading_symbol,
            exchange: item.exchange,
            instrument_key: item.instrument_key,
            isin: item.isin,
            short_name: item.short_name,
            instrument_type: item.instrument_type,
            segment: item.segment,
          }))
          .filter(
            (item: StockSearchResult) =>
              item.instrument_key &&
              (!item.segment || item.segment === "NSE_EQ") &&
              (!item.instrument_type ||
                item.instrument_type === "EQ" ||
                item.instrument_type === "EQUITY")
          )
          .slice(0, 10);

        if (!cancelled) {
          setSearchResults(results);
          setShowSearchResults(true);
        }
      } catch {
        if (!cancelled) {
          setSearchResults([]);
          setShowSearchResults(true);
        }
      } finally {
        if (!cancelled) {
          setSearchLoading(false);
        }
      }
    }, 600);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [searchQuery]);

  const selectSearchResult = (stock: StockSearchResult) => {
    if (!stock.instrument_key) return;

    setSymbol(
      stock.symbol ||
        stock.trading_symbol ||
        stock.short_name ||
        ""
    );
    setInstrumentKey(stock.instrument_key);
    setSearchQuery(
      stock.symbol ||
        stock.trading_symbol ||
        stock.short_name ||
        stock.name ||
        ""
    );
    setSearchResults([]);
    setShowSearchResults(false);
  };

  const runValidation = useCallback(async () => {
    const key = instrumentKey.trim();

    if (!key) {
      setError("Please enter a valid instrument key.");
      return;
    }

    if (fromDate > toDate) {
      setError("From date cannot be later than the to date.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}${VALIDATION_ENDPOINT}`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          instrument_key: key,
          from_date: fromDate,
          to_date: toDate,
        }),
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const message =
          payload?.detail ||
          payload?.message ||
          `Phase 10 API returned HTTP ${response.status}`;
        throw new Error(message);
      }

      const normalized =
        payload?.data && typeof payload.data === "object"
          ? payload.data
          : payload;

      if (!normalized || typeof normalized !== "object") {
        throw new Error("Phase 10 API returned an invalid response body.");
      }

      if (normalized.status === "error") {
        throw new Error(
          normalized.message || normalized.detail || "Phase 10 validation failed."
        );
      }

      setData(normalized as Phase10Response);
    } catch (requestError) {
      setData(null);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to run Phase 10 validation."
      );
    } finally {
      setLoading(false);
    }
  }, [fromDate, instrumentKey, toDate]);

  const selectInstrument = (value: (typeof QUICK_INSTRUMENTS)[number]) => {
    setSymbol(value.symbol);
    setInstrumentKey(value.key);
    setSearchQuery(value.symbol);
    setSearchResults([]);
    setShowSearchResults(false);
  };

  const dashboard = data?.dashboard_status;
  const summary = data?.validation_summary;
  const integrity = data?.data_integrity;

  return (
    <div className="min-h-screen bg-slate-950 text-white">

      <main className="mx-auto max-w-7xl px-6 py-8">
        <section className="mb-7">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
            Phase 10 · Validation
          </p>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Validation Dashboard
              </h1>
              <p className="mt-2 max-w-3xl text-sm text-slate-400">
                Aggregates the completed Phase 10 Layers 1–5 without mutating
                their results.
              </p>
            </div>
            {data && <StatusBadge status={data.status} />}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-black/10">
          <div className="mb-4 flex items-center gap-3">
            <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-300">
              <Gauge size={18} />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">
                Validation Inputs
              </h2>
              <p className="text-xs text-slate-500">
                Select an instrument and the historical validation range.
              </p>
            </div>
          </div>

          <div className="mb-4 flex flex-wrap gap-2">
            {QUICK_INSTRUMENTS.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => selectInstrument(item)}
                className={
                  item.symbol === symbol
                    ? "rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-300"
                    : "rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-1.5 text-xs text-slate-400 hover:border-slate-600 hover:text-white"
                }
              >
                {item.symbol}
              </button>
            ))}
          </div>

          <div className="grid gap-3 lg:grid-cols-[2fr_1fr_1fr_auto]">
            <div className="relative">
              <input
                value={searchQuery}
                onChange={(event) => {
                  setSearchQuery(event.target.value);
                  setSymbol("");
                  setInstrumentKey("");
                }}
                onFocus={() => {
                  if (searchResults.length > 0) {
                    setShowSearchResults(true);
                  }
                }}
                aria-label="Search stock"
                placeholder="Search stock by name or symbol..."
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 pr-10 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-500"
              />

              {searchLoading && (
                <RefreshCw
                  size={15}
                  className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin text-cyan-400"
                />
              )}

              {showSearchResults && searchQuery.trim().length >= 2 && (
                <div className="absolute left-0 right-0 top-full z-40 mt-2 max-h-80 overflow-y-auto rounded-xl border border-slate-700 bg-slate-950 shadow-2xl">
                  {searchLoading ? (
                    <div className="px-4 py-3 text-xs text-slate-500">
                      Searching stocks...
                    </div>
                  ) : searchResults.length > 0 ? (
                    searchResults.map((stock, index) => (
                      <button
                        key={`${stock.instrument_key}-${index}`}
                        type="button"
                        onClick={() => selectSearchResult(stock)}
                        className="w-full border-b border-slate-800 px-4 py-3 text-left last:border-b-0 hover:bg-slate-900"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-semibold text-white">
                              {stock.symbol ||
                                stock.trading_symbol ||
                                stock.short_name ||
                                "Unknown"}
                            </p>
                            <p className="mt-0.5 truncate text-xs text-slate-500">
                              {stock.name || stock.short_name || "Stock"}
                              {stock.exchange ? ` · ${stock.exchange}` : ""}
                            </p>
                          </div>
                          <span className="shrink-0 rounded-md border border-cyan-500/20 bg-cyan-500/10 px-2 py-1 text-[10px] font-medium text-cyan-300">
                            {stock.exchange || "NSE"}
                          </span>
                        </div>
                        <p className="mt-2 truncate font-mono text-[10px] text-slate-600">
                          {stock.instrument_key}
                        </p>
                      </button>
                    ))
                  ) : (
                    <div className="px-4 py-3 text-xs text-slate-500">
                      No matching NSE equity found.
                    </div>
                  )}
                </div>
              )}

              {instrumentKey && (
                <p className="mt-1.5 truncate px-1 text-[10px] text-slate-600">
                  Instrument Key: {instrumentKey}
                </p>
              )}
            </div>
            <input
              type="date"
              value={fromDate}
              onChange={(event) => setFromDate(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-500"
            />
            <input
              type="date"
              value={toDate}
              onChange={(event) => setToDate(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-500"
            />
            <button
              type="button"
              onClick={runValidation}
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-500 px-5 py-2.5 text-sm font-bold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-wait disabled:opacity-60"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Running..." : "Run Validation"}
            </button>
          </div>
        </section>

        {error && (
          <div className="mt-5 flex items-start gap-3 rounded-2xl border border-red-500/25 bg-red-500/5 p-4 text-sm text-red-300">
            <XCircle size={18} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">Phase 10 request failed</p>
              <p className="mt-1 text-xs text-red-300/80">{error}</p>
            </div>
          </div>
        )}

        {dashboard && (
          <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-black/10">
            <div className="mb-5 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-300">
                  <Layers3 size={18} />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-white">
                    Phase 10 Pipeline Status
                  </h2>
                  <p className="text-xs text-slate-500">
                    {data?.instrument_key || instrumentKey}
                  </p>
                </div>
              </div>
              <div className="text-right text-xs text-slate-500">
                <span className="font-semibold text-slate-200">
                  {dashboard.layers_success ?? 0}
                </span>{" "}
                / {dashboard.layers_available ?? 0} layers available
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              {["layer1", "layer2", "layer3", "layer4", "layer5"].map(
                (layer) => (
                  <LayerStatus
                    key={layer}
                    name={layer.replace("layer", "Layer ")}
                    status={dashboard.layer_status?.[layer]}
                  />
                )
              )}
            </div>
          </section>
        )}

        {summary && (
          <div className="mt-6 grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
            <MetricCard title="Layer 1 — Historical Backtest" icon={<Database size={18} />}>
              <MetricRow label="Period Count" value={formatValue(summary.layer1_historical_backtest?.period_count)} />
              <MetricRow label="Evaluation Points" value={formatValue(summary.layer1_historical_backtest?.evaluation_points)} />
              <MetricRow label="Signal Points" value={formatValue(summary.layer1_historical_backtest?.signal_points)} />
              <MetricRow label="Future Data Used" value={formatValue(summary.layer1_historical_backtest?.future_data_used_for_signal_points)} />
            </MetricCard>

            <MetricCard title="Layer 2 — Walk Forward" icon={<Gauge size={18} />}>
              <MetricRow label="Window Count" value={formatValue(summary.layer2_walk_forward?.window_count)} />
              <MetricRow label="Evaluation Points" value={formatValue(summary.layer2_walk_forward?.evaluation_points)} />
              <MetricRow label="Correct Signals" value={formatValue(summary.layer2_walk_forward?.correct_signals)} />
              <MetricRow label="Directional Accuracy" value={formatValue(summary.layer2_walk_forward?.directional_accuracy_percent, "%")} />
              <MetricRow label="Positive Forward Return Rate" value={formatValue(summary.layer2_walk_forward?.positive_forward_return_rate_percent, "%")} />
              <MetricRow label="Average Forward Return" value={formatValue(summary.layer2_walk_forward?.average_forward_return_percent, "%")} />
              <MetricRow label="Agreement Points" value={formatValue(summary.layer2_walk_forward?.agreement_points)} />
              <MetricRow label="Conflict Points" value={formatValue(summary.layer2_walk_forward?.conflict_points)} />
              <MetricRow label="Simulated Final Capital" value={formatValue(summary.layer2_walk_forward?.simulated_final_capital)} />
              <MetricRow label="Simulated Return" value={formatValue(summary.layer2_walk_forward?.simulated_return_percent, "%")} />
              <MetricRow label="Simulated Trade Points" value={formatValue(summary.layer2_walk_forward?.simulated_trade_points)} />
              <MetricRow label="Max Drawdown" value={formatValue(summary.layer2_walk_forward?.max_drawdown, "%")} />
            </MetricCard>

            <MetricCard title="Layer 3 — Multi Period" icon={<Layers3 size={18} />}>
              <MetricRow label="Period Count" value={formatValue(summary.layer3_multi_period?.period_count)} />
              <MetricRow label="Evaluation Points" value={formatValue(summary.layer3_multi_period?.evaluation_points)} />
              <MetricRow label="Directional Accuracy" value={formatValue(summary.layer3_multi_period?.directional_accuracy_percent, "%")} />
              <MetricRow label="Average Period Return" value={formatValue(summary.layer3_multi_period?.average_period_return_percent, "%")} />
              <MetricRow label="Average Max Drawdown" value={formatValue(summary.layer3_multi_period?.average_max_drawdown, "%")} />
            </MetricCard>

            <MetricCard title="Layer 4 — Signal Performance" icon={<Layers3 size={18} />}>
              <MetricRow label="Signal Points" value={formatValue(summary.layer4_signal_performance?.signal_points)} />
              <MetricRow label="Resolved Points" value={formatValue(summary.layer4_signal_performance?.resolved_points)} />
              <MetricRow label="Correct Points" value={formatValue(summary.layer4_signal_performance?.correct_points)} />
              <MetricRow label="Incorrect Points" value={formatValue(summary.layer4_signal_performance?.incorrect_points)} />
              <MetricRow label="Positive Return Points" value={formatValue(summary.layer4_signal_performance?.positive_return_points)} />
              <MetricRow label="Negative Return Points" value={formatValue(summary.layer4_signal_performance?.negative_return_points)} />
              <MetricRow label="Agreement Points" value={formatValue(summary.layer4_signal_performance?.agreement_points)} />
              <MetricRow label="Conflict Points" value={formatValue(summary.layer4_signal_performance?.conflict_points)} />
            </MetricCard>

            <MetricCard title="Layer 5 — AI vs Baseline" icon={<ShieldCheck size={18} />}>
              <MetricRow label="Comparison Points" value={formatValue(summary.layer5_ai_vs_baseline?.comparison_points)} />
              <MetricRow label="AI Directional Accuracy" value={formatValue(summary.layer5_ai_vs_baseline?.ai_directional_accuracy_percent, "%")} />
              <MetricRow label="AI Signal-Aligned Return" value={formatValue(summary.layer5_ai_vs_baseline?.ai_signal_aligned_return_percent, "%")} />
              <MetricRow label="Always Bullish Accuracy" value={formatValue(summary.layer5_ai_vs_baseline?.always_bullish_accuracy_percent, "%")} />
              <MetricRow label="Always Bearish Accuracy" value={formatValue(summary.layer5_ai_vs_baseline?.always_bearish_accuracy_percent, "%")} />
              <MetricRow label="AI − Always Bullish" value={formatValue(summary.layer5_ai_vs_baseline?.ai_minus_always_bullish_accuracy_percent, "%")} />
              <MetricRow label="AI − Always Bearish" value={formatValue(summary.layer5_ai_vs_baseline?.ai_minus_always_bearish_accuracy_percent, "%")} />
            </MetricCard>
          </div>
        )}

        {integrity && (
          <MetricCard title="Data Integrity" icon={<ShieldCheck size={18} />}>
            <MetricRow
              label="Layer Mutation"
              value={
                integrity.no_layer_mutation ? (
                  <span className="text-emerald-300">✓ None</span>
                ) : (
                  <span className="text-red-300">⚠ Detected</span>
                )
              }
            />
            <MetricRow
              label="Historical Outcome Source"
              value={integrity.historical_outcome_source || "—"}
            />
            {Object.entries(integrity.future_data_usage || {}).map(
              ([layer, value]) => (
                <MetricRow
                  key={layer}
                  label={`Future Data — ${layer}`}
                  value={formatValue(value)}
                />
              )
            )}
          </MetricCard>
        )}

        {!data && !loading && !error && (
          <div className="mt-6 rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 p-12 text-center">
            <Gauge className="mx-auto text-slate-700" size={34} />
            <h2 className="mt-4 text-sm font-semibold text-slate-300">
              Phase 10 validation is ready
            </h2>
            <p className="mt-2 text-xs text-slate-600">
              Select an instrument and click Run Validation to load the real
              Layer 1–5 validation results.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
