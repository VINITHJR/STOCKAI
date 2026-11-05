import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "../services/marketApi";

type EventItem = {
  symbol?: string;
  company_name?: string;
  subject?: string;
  details?: string;
  category?: string;
  attention?: string;
  attention_note?: string;
  broadcast_time?: string;
  attachment_url?: string;
  source?: string;

  // Layer 7 service response fields.
  title?: string;
  summary?: string;
  published?: string;
  url?: string;
  risk_level?: string;
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

type NewsEventResponse = {
  status: string;
  source?: string;
  official_reference?: string;
  symbol?: string;
  days?: number;
  total_events?: number;
  attention_summary?: {
    overall?: string;
    high?: number;
    medium?: number;
    low?: number;
    review?: number;
  };
  counts?: {
    high?: number;
    medium?: number;
    low?: number;
    review?: number;
  };
  summary?: {
    total?: number;
    high?: number;
    medium?: number;
    low?: number;
    review?: number;
  };
  events?: EventItem[];
  items?: EventItem[];
  risk_level?: string;
  message?: string;
  warnings?: string[];
};

const DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"];
const PERIODS = [
  { label: "1D", value: 1 },
  { label: "7D", value: 7 },
  { label: "30D", value: 30 },
];

function normalizeAttention(value?: string) {
  const text = (value || "").trim().toLowerCase();

  if (text.includes("high")) return "High attention";
  if (text.includes("medium")) return "Medium attention";
  if (text.includes("low")) return "Low attention";
  if (text.includes("review")) return "Review";

  return "Review";
}

function attentionClass(attention: string) {
  if (attention === "High attention") return "bg-red-400/10 text-red-300 border-red-400/20";
  if (attention === "Medium attention") return "bg-amber-400/10 text-amber-300 border-amber-400/20";
  if (attention === "Low attention") return "bg-emerald-400/10 text-emerald-300 border-emerald-400/20";
  return "bg-slate-400/10 text-slate-300 border-slate-400/20";
}

function formatDate(value?: string) {
  if (!value) return "—";

  const parsed = new Date(value.replace(" ", "T"));
  if (Number.isNaN(parsed.getTime())) return value;

  return parsed.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function eventTitle(event: EventItem) {
  return event.title?.trim() || event.subject?.trim() || "Corporate announcement";
}

function eventDetails(event: EventItem) {
  return event.summary?.trim() || event.details?.trim() || "";
}

function eventDate(event: EventItem) {
  return event.published || event.broadcast_time || "";
}

function eventUrl(event: EventItem) {
  return event.url || event.attachment_url || "";
}

function eventAttention(event: EventItem) {
  return normalizeAttention(event.attention || event.risk_level);
}

function getEvents(data: NewsEventResponse) {
  return data.events?.length ? data.events : data.items || [];
}

function getCounts(data: NewsEventResponse, events: EventItem[]) {
  const derived = events.reduce(
    (counts, event) => {
      const attention = eventAttention(event);
      if (attention === "High attention") counts.high += 1;
      else if (attention === "Medium attention") counts.medium += 1;
      else if (attention === "Low attention") counts.low += 1;
      else counts.review += 1;
      return counts;
    },
    { high: 0, medium: 0, low: 0, review: 0 }
  );

  // Prefer counts derived from the actual records when records are present.
  // This keeps the UI correct even when an older backend sends stale/zero
  // aggregate counters while the event records themselves are populated.
  const hasClassification = events.some(
    (event) => Boolean(event.attention || event.risk_level)
  );

  if (events.length > 0 && hasClassification) {
    return derived;
  }

  const source = data.attention_summary || data.counts || data.summary;
  if (source) {
    return {
      high: Number(source.high || 0),
      medium: Number(source.medium || 0),
      low: Number(source.low || 0),
      review: Number(source.review || 0),
    };
  }

  return derived;
}

function getOverall(data: NewsEventResponse, counts: ReturnType<typeof getCounts>) {
  const explicit = data.attention_summary?.overall || data.risk_level;
  if (explicit) return normalizeAttention(explicit);

  if (counts.high > 0) return "High attention";
  if (counts.medium > 0) return "Medium attention";
  if (counts.review > 0) return "Review";
  if (counts.low > 0) return "Low attention";
  return "No recent events";
}

export default function NewsEventAnalysis() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOLS[0]);
  const [searchResults, setSearchResults] = useState<StockSearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [days, setDays] = useState(7);
  const [data, setData] = useState<NewsEventResponse | null>(null);
  const [, setLoading] = useState(false);
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
    load(selected, days);
  };

  const load = useCallback(async (requested: string, requestedDays: number) => {
    const clean = requested.trim().toUpperCase();
    if (!clean) return;

    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        `${API_BASE_URL}/api/news-events/${encodeURIComponent(clean)}?days=${requestedDays}&limit=20`,
        { headers: { Accept: "application/json" } }
      );

      const payload: NewsEventResponse = await response.json();
      if (!response.ok || payload.status !== "success") {
        throw new Error(payload.message || `News/Event API returned HTTP ${response.status}`);
      }

      setData(payload);
      setSymbol(clean);
      setDays(requestedDays);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Unable to load NSE announcements.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(DEFAULT_SYMBOLS[0], 7);
  }, [load]);

  useEffect(() => {
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
  }, []);

  const events = data ? getEvents(data) : [];
  const counts = data ? getCounts(data, events) : { high: 0, medium: 0, low: 0, review: 0 };
  const overall = data ? getOverall(data, counts) : "Review";
  const totalEvents = events.length > 0 ? events.length : data?.total_events ?? data?.summary?.total ?? 0;

  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm text-cyan-400">Phase 5 · Layer 7</p>
          <h2 className="mt-1 text-2xl font-semibold">News &amp; Event Analysis</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
            NSE corporate announcements classified into event categories and attention levels.
            This layer provides risk context; it does not generate a BUY, HOLD, or SELL signal.
          </p>
        </div>

        <div className="relative w-full lg:max-w-[620px]">
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
            onClick={() => load(item, days)}
            className="rounded-full border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-xs text-slate-300 hover:border-cyan-400/60 hover:text-cyan-300"
          >
            {item}
          </button>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="mr-1 text-xs text-slate-500">Lookback:</span>
        {PERIODS.map((period) => (
          <button
            key={period.value}
            type="button"
            onClick={() => load(symbol, period.value)}
            className={`rounded-full border px-3 py-1.5 text-xs ${
              days === period.value
                ? "border-cyan-400/70 bg-cyan-400/10 text-cyan-300"
                : "border-slate-700 bg-slate-950/60 text-slate-400 hover:text-slate-200"
            }`}
          >
            {period.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="mt-5 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {data && !error && (
        <>
          <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950/40 p-5">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs text-slate-500">Event risk context</p>
                <div className="mt-1 flex flex-wrap items-center gap-3">
                  <h3 className="text-xl font-semibold text-white">{data.symbol}</h3>
                  <span className={`rounded-full border px-2.5 py-1 text-xs font-medium ${attentionClass(overall)}`}>
                    {overall}
                  </span>
                </div>
                <p className="mt-1 text-sm text-slate-500">
                  {totalEvents} announcement{totalEvents === 1 ? "" : "s"} in the last {data.days} day{data.days === 1 ? "" : "s"}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-lg border border-red-400/10 bg-red-400/5 px-3 py-2 text-center">
                  <p className="text-[10px] text-red-300">High</p>
                  <p className="mt-1 text-lg font-semibold text-white">{counts.high}</p>
                </div>
                <div className="rounded-lg border border-amber-400/10 bg-amber-400/5 px-3 py-2 text-center">
                  <p className="text-[10px] text-amber-300">Medium</p>
                  <p className="mt-1 text-lg font-semibold text-white">{counts.medium}</p>
                </div>
                <div className="rounded-lg border border-emerald-400/10 bg-emerald-400/5 px-3 py-2 text-center">
                  <p className="text-[10px] text-emerald-300">Low</p>
                  <p className="mt-1 text-lg font-semibold text-white">{counts.low}</p>
                </div>
                <div className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-center">
                  <p className="text-[10px] text-slate-400">Review</p>
                  <p className="mt-1 text-lg font-semibold text-white">{counts.review}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 space-y-3">
            {events.map((event, index) => (
              <article
                key={`${eventDate(event)}-${eventTitle(event)}-${index}`}
                className="rounded-xl border border-slate-800 bg-slate-950/40 p-4"
              >
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${attentionClass(eventAttention(event))}`}>
                        {eventAttention(event)}
                      </span>
                      <span className="rounded-full border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400">
                        {event.category || "general_news"}
                      </span>
                    </div>
                    <h4 className="mt-2 text-sm font-semibold text-slate-100">{eventTitle(event)}</h4>
                    {eventDetails(event) && (
                      <p className="mt-1 text-xs leading-5 text-slate-400">{eventDetails(event)}</p>
                    )}
                  </div>

                  <div className="shrink-0 text-left lg:text-right">
                    <p className="text-xs text-slate-500">{formatDate(eventDate(event))}</p>
                    {eventUrl(event) && (
                      <a
                        href={eventUrl(event)}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-2 inline-block text-xs text-cyan-400 hover:text-cyan-300"
                      >
                        View NSE filing →
                      </a>
                    )}
                  </div>
                </div>
                <p className="mt-3 border-t border-slate-800 pt-3 text-[11px] text-slate-500">
                  {event.attention_note || (eventAttention(event) === "High attention" ? "Potentially material event; review the original source." : eventAttention(event) === "Medium attention" ? "Potentially relevant company event; review the source for context." : eventAttention(event) === "Low attention" ? "Routine or informational event based on the available classification." : "Event type is not confidently classified; review the original source.")}
                </p>
              </article>
            ))}

            {events.length === 0 && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-8 text-center text-sm text-slate-500">
                No NSE corporate announcements were returned for this symbol and period.
              </div>
            )}
          </div>

          <p className="mt-4 text-[11px] leading-5 text-slate-600">
            Source: {data.source || "News/Event service"}. Review the original source before relying on an event classification.
          </p>
        </>
      )}
    </section>
  );
}
