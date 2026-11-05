import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

interface SearchResult {
  name: string;
  symbol: string;
  exchange: string;
  instrument_key: string;
  isin?: string;
  short_name?: string;
  instrument_type?: "EQUITY" | "INDEX";
  segment?: string;
}

const API_BASE_URL = "http://127.0.0.1:8000";

function StockSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery) {
      setResults([]);
      setLoading(false);
      return;
    }

    const controller = new AbortController();

    const timeout = setTimeout(async () => {
      try {
        setLoading(true);

        const response = await fetch(
          `${API_BASE_URL}/api/search?query=${encodeURIComponent(
            trimmedQuery
          )}`,
          {
            signal: controller.signal,
          }
        );

        if (!response.ok) {
          throw new Error(`Search failed: ${response.status}`);
        }

        const result = await response.json();

        if (result.status === "success") {
          setResults(Array.isArray(result.data) ? result.data : []);
        } else {
          setResults([]);
        }
      } catch (error) {
        if ((error as DOMException).name === "AbortError") {
          return;
        }

        console.error("Dashboard search error:", error);
        setResults([]);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }, 300);

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [query]);

  const handleResultClick = (item: SearchResult) => {
    setQuery("");
    setResults([]);

    const safeSymbol =
      item.symbol || item.name || "STOCK";

    navigate(
      `/stock/${encodeURIComponent(safeSymbol)}?instrument_key=${encodeURIComponent(
        item.instrument_key
      )}`
    );
  };

  const isIndex = (item: SearchResult) =>
    item.instrument_type === "INDEX" ||
    item.segment === "NSE_INDEX" ||
    item.segment === "BSE_INDEX" ||
    item.instrument_key.includes("_INDEX|");

  return (
    <div className="relative w-full max-w-xl">
      <div className="flex items-center rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 transition focus-within:border-cyan-400">
        <span className="mr-3 text-lg text-slate-400">🔍</span>

        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search stocks, companies or indices..."
          className="w-full bg-transparent text-sm text-white outline-none placeholder:text-slate-500"
        />

        {query && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setResults([]);
            }}
            className="ml-2 text-slate-500 hover:text-white"
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {query.trim() && (
        <div className="absolute left-0 right-0 top-full z-50 mt-2 overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
          {loading ? (
            <div className="px-4 py-6 text-center text-sm text-slate-500">
              Searching Upstox...
            </div>
          ) : results.length > 0 ? (
            results.map((item) => {
              const index = isIndex(item);

              return (
                <button
                  type="button"
                  key={item.instrument_key}
                  onClick={() => handleResultClick(item)}
                  className="flex w-full items-center justify-between border-b border-slate-800 px-4 py-4 text-left transition last:border-0 hover:bg-slate-800"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-white">
                        {item.symbol}
                      </p>

                      <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
                        {index ? "INDEX" : item.exchange}
                      </span>
                    </div>

                    <p className="mt-1 truncate text-xs text-slate-500">
                      {item.name}
                    </p>
                  </div>

                  <span className="ml-4 text-slate-500">→</span>
                </button>
              );
            })
          ) : (
            <div className="px-4 py-6 text-center text-sm text-slate-500">
              No stocks or indices found
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default StockSearch;
