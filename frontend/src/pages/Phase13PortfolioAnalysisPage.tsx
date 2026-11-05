import React, { useMemo, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";

type Position = {
  instrument_key: string;
  symbol: string;
  quantity: number;
  average_price: number;
  current_price: number;
  sector: string;
  exchange: string;
  asset_type: string;
};

type AnalysisResponse = {
  status: string;
  data: {
    portfolio: {
      position_count: number;
      initial_capital: number;
      cash: number;
      invested_value: number;
      market_value: number;
      total_value: number;
      total_pnl: number;
      total_pnl_percent: number;
      invested_percent: number;
      cash_percent: number;
      status: string;
      positions: Array<{
        instrument_key: string;
        symbol: string;
        quantity: number;
        invested_value: number;
        market_value: number;
        pnl: number;
        pnl_percent: number;
        portfolio_weight: number;
        sector: string;
        asset_type: string;
        status: string;
      }>;
    };

    performance: {
      initial_capital: number;
      invested_value: number;
      market_value: number;
      cash: number;
      total_value: number;
      total_pnl: number;
      total_return_percent: number;
      realized_pnl: number;
      unrealized_pnl: number;
      winning_positions: number;
      losing_positions: number;
      neutral_positions: number;
      win_rate_percent: number;
      best_position: {
        symbol: string;
        pnl: number;
        pnl_percent: number;
        market_value: number;
        invested_value: number;
      } | null;
      worst_position: {
        symbol: string;
        pnl: number;
        pnl_percent: number;
        market_value: number;
        invested_value: number;
      } | null;
      average_position_return_percent: number;
      status: string;
    };

    allocation: {
      total_portfolio_value: number;
      invested_value: number;
      cash_value: number;
      invested_percent: number;
      cash_percent: number;
      position_allocations: Array<{
        name: string;
        value: number;
        percentage: number;
        position_count: number;
        invested_value: number;
        market_value: number;
      }>;
      sector_allocations: Array<{
        name: string;
        value: number;
        percentage: number;
        position_count: number;
        invested_value: number;
        market_value: number;
      }>;
      asset_type_allocations: Array<{
        name: string;
        value: number;
        percentage: number;
        position_count: number;
        invested_value: number;
        market_value: number;
      }>;
      largest_position: {
        name: string;
        value: number;
        percentage: number;
        position_count: number;
      } | null;
      largest_sector: {
        name: string;
        value: number;
        percentage: number;
        position_count: number;
      } | null;
      largest_asset_type: {
        name: string;
        value: number;
        percentage: number;
        position_count: number;
      } | null;
      allocation_status: string;
    };

    diversification: {
      position_count: number;
      sector_count: number;
      asset_type_count: number;
      largest_position_percent: number;
      largest_sector_percent: number;
      largest_asset_type_percent: number;
      position_hhi: number;
      sector_hhi: number;
      asset_type_hhi: number;
      position_diversification_score: number;
      sector_diversification_score: number;
      overall_diversification_score: number;
      concentration_level: string;
      diversification_status: string;
      concentration_warnings: string[];
    };

    attribution: {
      total_pnl: number;
      position_attributions: Array<{
        name: string;
        pnl: number;
        contribution_percent: number;
        invested_value: number;
        market_value: number;
        return_percent: number;
        position_count: number;
        status: string;
        metadata?: {
          instrument_key?: string;
          sector?: string;
          asset_type?: string;
        };
      }>;
      sector_attributions: Array<{
        name: string;
        pnl: number;
        contribution_percent: number;
        invested_value: number;
        market_value: number;
        return_percent: number;
        position_count: number;
        status: string;
      }>;
      best_position: {
        name: string;
        pnl: number;
        contribution_percent: number;
        return_percent: number;
      } | null;
      worst_position: {
        name: string;
        pnl: number;
        contribution_percent: number;
        return_percent: number;
      } | null;
      best_sector: {
        name: string;
        pnl: number;
        contribution_percent: number;
        return_percent: number;
      } | null;
      worst_sector: {
        name: string;
        pnl: number;
        contribution_percent: number;
        return_percent: number;
      } | null;
      positive_position_count: number;
      negative_position_count: number;
      neutral_position_count: number;
      positive_sector_count: number;
      negative_sector_count: number;
      neutral_sector_count: number;
      attribution_status: string;
    };

    health: {
      health_score: number;
      health_status: string;
      performance_score: number;
      diversification_score: number;
      allocation_score: number;
      attribution_score: number;
      cash_score: number;
      performance_weight: number;
      diversification_weight: number;
      allocation_weight: number;
      attribution_weight: number;
      cash_weight: number;
      strengths: string[];
      concerns: string[];
      recommendations: string[];
    };

    summary: {
      position_count: number;
      total_value: number;
      total_pnl: number;
      total_return_percent: number;
      invested_percent: number;
      cash_percent: number;
      diversification_score: number;
      health_score: number;
      health_status: string;
      portfolio_status: string;
      allocation_status: string;
      diversification_status: string;
      attribution_status: string;
    };

    metadata: {
      currency: string;
      phase: number;
      layer: number;
      component: string;
      layers_included: number[];
    };
  };
};

type StockSearchResult = {
  instrument_key: string;
  symbol?: string;
  name?: string;
  short_name?: string;
  exchange?: string;
  sector?: string;
  asset_type?: string;
};

function formatCurrency(value: number) {
  return `₹${Number(value || 0).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatNumber(value: number, digits = 2) {
  return Number(value || 0).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatPercent(value: number, digits = 2) {
  return `${formatNumber(value, digits)}%`;
}

function getStatusClass(status: string) {
  const value = status.toUpperCase();

  if (
    value.includes("EXCELLENT") ||
    value.includes("HEALTHY") ||
    value.includes("PROFITABLE") ||
    value.includes("POSITIVE") ||
    value === "BALANCED"
  ) {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-400";
  }

  if (
    value.includes("GOOD") ||
    value.includes("FAIR") ||
    value.includes("NEUTRAL") ||
    value.includes("CASH_HEAVY")
  ) {
    return "border-amber-500/30 bg-amber-500/10 text-amber-400";
  }

  return "border-red-500/30 bg-red-500/10 text-red-400";
}

function getPnlClass(value: number) {
  if (value > 0) return "text-emerald-400";
  if (value < 0) return "text-red-400";
  return "text-slate-300";
}

function ScoreBar({
  label,
  score,
}: {
  label: string;
  score: number;
}) {
  const safeScore = Math.max(0, Math.min(100, score));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-sm font-semibold text-white">
          {formatNumber(safeScore)}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-cyan-400 transition-all duration-500"
          style={{ width: `${safeScore}%` }}
        />
      </div>
    </div>
  );
}

function AllocationBar({
  name,
  percentage,
}: {
  name: string;
  percentage: number;
}) {
  const safePercentage = Math.max(
    0,
    Math.min(100, percentage)
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-4">
        <span className="truncate text-sm text-slate-300">
          {name}
        </span>

        <span className="shrink-0 text-sm font-medium text-white">
          {formatPercent(percentage)}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-cyan-400"
          style={{ width: `${safePercentage}%` }}
        />
      </div>
    </div>
  );
}

const Phase13PortfolioAnalysisPage: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [cash, setCash] = useState("5000");
  const [initialCapital, setInitialCapital] = useState("10000");
  const [realizedPnl, setRealizedPnl] = useState("0");

  const [stockSearch, setStockSearch] = useState("");
  const [stockSearchResults, setStockSearchResults] = useState<
    StockSearchResult[]
  >([]);
  const [stockSearchOpen, setStockSearchOpen] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);

  const [selectedStock, setSelectedStock] =
    useState<StockSearchResult | null>(null);

  const [quantity, setQuantity] = useState("");
  const [averagePrice, setAveragePrice] = useState("");
  const [currentPrice, setCurrentPrice] = useState("");
  const [currentPriceLoading, setCurrentPriceLoading] = useState(false);
  const [currentPriceError, setCurrentPriceError] = useState("");

  const [analysis, setAnalysis] =
    useState<AnalysisResponse["data"] | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canAddPosition = useMemo(() => {
    return (
      selectedStock !== null &&
      Number(quantity) > 0 &&
      Number(averagePrice) >= 0 &&
      Number(currentPrice) >= 0
    );
  }, [
    selectedStock,
    quantity,
    averagePrice,
    currentPrice,
  ]);

  const searchStocks = async (query: string) => {
    setStockSearch(query);

    if (query.trim().length < 2) {
      setStockSearchResults([]);
      setStockSearchOpen(false);
      return;
    }

    setSearchLoading(true);
    setStockSearchOpen(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/stocks/search?query=${encodeURIComponent(
          query.trim()
        )}`
      );

      if (!response.ok) {
        throw new Error(
          `Stock search failed (${response.status})`
        );
      }

      const data = await response.json();

      const results = Array.isArray(data)
        ? data
        : Array.isArray(data?.results)
        ? data.results
        : Array.isArray(data?.data)
        ? data.data
        : [];

      setStockSearchResults(results);
    } catch (err) {
      setStockSearchResults([]);
      setError(
        err instanceof Error
          ? err.message
          : "Unable to search stocks."
      );
    } finally {
      setSearchLoading(false);
    }
  };

  const selectStock = async (
    stock: StockSearchResult
  ) => {
    const symbol =
      stock.symbol ||
      stock.short_name ||
      stock.name ||
      "";

    setSelectedStock(stock);
    setStockSearch(symbol);
    setStockSearchOpen(false);
    setError("");
    setCurrentPrice("");
    setCurrentPriceError("");
    setCurrentPriceLoading(true);

    // Fetch the live price and sector independently.
    // Sector failure does not block the live price.
    const quotePromise = fetch(
      `${API_BASE_URL}/api/stocks/quote?instrument_key=${encodeURIComponent(
        stock.instrument_key
      )}`
    );

    const sectorPromise = symbol
      ? fetch(
          `${API_BASE_URL}/api/fundamentals/${encodeURIComponent(symbol)}`
        ).catch(() => null)
      : Promise.resolve(null);

    try {
      const [quoteResponse, sectorResponse] = await Promise.all([
        quotePromise,
        sectorPromise,
      ]);

      // -------------------------------------------------------
      // Live current price
      // -------------------------------------------------------
      if (!quoteResponse.ok) {
        throw new Error(
          `Unable to fetch current price (${quoteResponse.status})`
        );
      }

      const quoteData = await quoteResponse.json();

      if (
        quoteData?.status === "error" ||
        typeof quoteData?.value !== "number" ||
        !Number.isFinite(quoteData.value) ||
        quoteData.value <= 0
      ) {
        throw new Error(
          quoteData?.message ||
            "Current market price is unavailable."
        );
      }

      setCurrentPrice(String(quoteData.value));

      // -------------------------------------------------------
      // Stock sector
      // -------------------------------------------------------
      if (sectorResponse?.ok) {
        const fundamentalData = await sectorResponse.json();
        const sector =
          typeof fundamentalData?.sector === "string"
            ? fundamentalData.sector.trim()
            : "";

        setSelectedStock((previous) =>
          previous
            ? { ...previous, sector: sector || "Unknown" }
            : previous
        );
      } else {
        setSelectedStock((previous) =>
          previous
            ? { ...previous, sector: "Unknown" }
            : previous
        );
      }
    } catch (err) {
      setCurrentPrice("");
      setCurrentPriceError(
        err instanceof Error
          ? err.message
          : "Unable to fetch the current market price."
      );
    } finally {
      setCurrentPriceLoading(false);
    }
  };

  const addPosition = () => {
    if (!canAddPosition || !selectedStock) {
      setError(
        "Select a stock and enter valid quantity and prices."
      );
      return;
    }

    const newPosition: Position = {
      instrument_key:
        selectedStock.instrument_key,
      symbol:
        selectedStock.symbol ||
        selectedStock.short_name ||
        selectedStock.name ||
        selectedStock.instrument_key,
      quantity: Number(quantity),
      average_price: Number(averagePrice),
      current_price: Number(currentPrice),
      sector:
        selectedStock.sector ||
        "Unknown",
      exchange:
        selectedStock.exchange ||
        "NSE",
      asset_type:
        selectedStock.asset_type ||
        "EQUITY",
    };

    setPositions((previous) => [
      ...previous,
      newPosition,
    ]);

    setSelectedStock(null);
    setStockSearch("");
    setQuantity("");
    setAveragePrice("");
    setCurrentPrice("");
    setError("");
  };

  const removePosition = (index: number) => {
    setPositions((previous) =>
      previous.filter(
        (_, positionIndex) =>
          positionIndex !== index
      )
    );
  };

  const loadSamplePortfolio = () => {
    setPositions([
      {
        instrument_key: "NSE_EQ|TEST1",
        symbol: "TEST1",
        quantity: 10,
        average_price: 100,
        current_price: 110,
        sector: "Technology",
        exchange: "NSE",
        asset_type: "EQUITY",
      },
      {
        instrument_key: "NSE_EQ|TEST2",
        symbol: "TEST2",
        quantity: 20,
        average_price: 200,
        current_price: 190,
        sector: "Banking",
        exchange: "NSE",
        asset_type: "EQUITY",
      },
      {
        instrument_key: "NSE_EQ|TEST3",
        symbol: "TEST3",
        quantity: 10,
        average_price: 100,
        current_price: 100,
        sector: "Technology",
        exchange: "NSE",
        asset_type: "EQUITY",
      },
    ]);

    setCash("5000");
    setInitialCapital("10000");
    setRealizedPnl("0");
    setAnalysis(null);
    setError("");
  };

  const clearPortfolio = () => {
    setPositions([]);
    setAnalysis(null);
    setError("");
  };

  const runAnalysis = async () => {
    setError("");

    if (positions.length === 0) {
      setError(
        "Add at least one portfolio position before running analysis."
      );
      return;
    }

    if (Number(initialCapital) <= 0) {
      setError(
        "Initial capital must be greater than zero."
      );
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/portfolio-analysis/analyze`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            positions,
            cash: Number(cash),
            initial_capital: Number(
              initialCapital
            ),
            realized_pnl: Number(realizedPnl),
            currency: "INR",
          }),
        }
      );

      const result: AnalysisResponse =
        await response.json();

      if (!response.ok) {
        throw new Error(
          typeof result === "object" &&
            result !== null
            ? "Portfolio analysis request failed."
            : `Request failed (${response.status})`
        );
      }

      if (result.status !== "success") {
        throw new Error(
          "Portfolio analysis returned an unsuccessful status."
        );
      }

      setAnalysis(result.data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to analyze portfolio."
      );
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-6 text-white md:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-cyan-400">
                Phase 13
              </span>

              <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Portfolio Analysis
              </span>
            </div>

            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
              Portfolio Analysis Dashboard
            </h1>

            <p className="mt-1 max-w-2xl text-sm text-slate-400">
              Analyze portfolio performance, allocation,
              diversification, attribution, and overall
              portfolio health.
            </p>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={loadSamplePortfolio}
              className="rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-4 py-2.5 text-sm font-medium text-cyan-400 transition hover:bg-cyan-500/20"
            >
              Load Test Portfolio
            </button>

            <button
              type="button"
              onClick={clearPortfolio}
              className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-slate-800"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Portfolio Input */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl">
          <div className="mb-5">
            <h2 className="text-base font-semibold text-white">
              Portfolio Input
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Add the positions you want to analyze.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-4">
            {/* Stock Search */}
            <div className="relative lg:col-span-2">
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Search Stock
              </label>

              <input
                type="text"
                value={stockSearch}
                onChange={(event) => {
                  setCurrentPrice("");
                  setCurrentPriceError("");
                  setSelectedStock(null);
                  searchStocks(event.target.value);
                }}
                onFocus={() => {
                  if (
                    stockSearch.trim().length >= 2
                  ) {
                    setStockSearchOpen(true);
                  }
                }}
                placeholder="Search symbol or company"
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-400"
              />

              {selectedStock && (
                <div className="mt-2 flex items-center gap-2 text-[11px]">
                  <span className="text-slate-500">Sector:</span>
                  <span className="font-medium text-cyan-400">
                    {selectedStock.sector || "Loading..."}
                  </span>
                </div>
              )}

              {stockSearchOpen && (
                <div className="absolute left-0 right-0 top-full z-30 mt-2 max-h-64 overflow-y-auto rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
                  {searchLoading ? (
                    <div className="px-4 py-3 text-xs text-slate-400">
                      Searching stocks...
                    </div>
                  ) : stockSearchResults.length ===
                    0 ? (
                    <div className="px-4 py-3 text-xs text-slate-500">
                      No matching stocks found.
                    </div>
                  ) : (
                    stockSearchResults.map(
                      (stock) => (
                        <button
                          key={
                            stock.instrument_key
                          }
                          type="button"
                          onClick={() =>
                            selectStock(stock)
                          }
                          className="flex w-full items-center justify-between border-b border-slate-800 px-4 py-3 text-left transition last:border-b-0 hover:bg-slate-800"
                        >
                          <div>
                            <div className="text-sm font-medium text-white">
                              {stock.symbol ||
                                stock.short_name ||
                                stock.name}
                            </div>

                            <div className="mt-0.5 text-[11px] text-slate-500">
                              {stock.name ||
                                stock.short_name ||
                                stock.instrument_key}
                            </div>
                          </div>

                          <div className="text-right text-[10px] text-slate-500">
                            {stock.exchange ||
                              "NSE"}
                          </div>
                        </button>
                      )
                    )
                  )}
                </div>
              )}
            </div>

            {/* Quantity */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Quantity
              </label>

              <input
                type="number"
                min="0"
                step="1"
                value={quantity}
                onChange={(event) =>
                  setQuantity(event.target.value)
                }
                placeholder="10"
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400"
              />
            </div>

            {/* Average Price */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Average Price
              </label>

              <input
                type="number"
                min="0"
                step="0.01"
                value={averagePrice}
                onChange={(event) =>
                  setAveragePrice(
                    event.target.value
                  )
                }
                placeholder="100.00"
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400"
              />
            </div>

            {/* Current Price */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Current Price
              </label>

              <div className="relative">
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={currentPrice}
                  onChange={(event) =>
                    setCurrentPrice(
                      event.target.value
                    )
                  }
                  placeholder={
                    currentPriceLoading
                      ? "Loading..."
                      : "Select a stock to load price"
                  }
                  className="w-full rounded-xl border border-emerald-500/30 bg-slate-950 px-3 py-2.5 pr-16 text-sm text-white outline-none focus:border-cyan-400"
                />

                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
                  {currentPriceLoading
                    ? "Loading"
                    : currentPrice
                    ? "Live"
                    : ""}
                </span>
              </div>

              {currentPriceError && (
                <p className="mt-1.5 text-[11px] text-red-400">
                  {currentPriceError}
                </p>
              )}

              {!currentPriceError &&
                selectedStock &&
                currentPrice &&
                !currentPriceLoading && (
                  <p className="mt-1.5 text-[11px] text-emerald-400">
                    Latest market price loaded automatically.
                  </p>
                )}
            </div>

            {/* Add Button */}
            <div className="flex items-end">
              <button
                type="button"
                onClick={addPosition}
                disabled={!canAddPosition}
                className="w-full rounded-xl bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Add Position
              </button>
            </div>

            {/* Cash */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Cash
              </label>

              <input
                type="number"
                min="0"
                step="0.01"
                value={cash}
                onChange={(event) =>
                  setCash(event.target.value)
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400"
              />
            </div>

            {/* Initial Capital */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Initial Capital
              </label>

              <input
                type="number"
                min="0"
                step="0.01"
                value={initialCapital}
                onChange={(event) =>
                  setInitialCapital(
                    event.target.value
                  )
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400"
              />
            </div>

            {/* Realized P&L */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-400">
                Realized P&L
              </label>

              <input
                type="number"
                step="0.01"
                value={realizedPnl}
                onChange={(event) =>
                  setRealizedPnl(
                    event.target.value
                  )
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-400"
              />
            </div>
          </div>

          {/* Current Positions */}
          {positions.length > 0 && (
            <div className="mt-6 overflow-hidden rounded-xl border border-slate-800">
              <div className="border-b border-slate-800 bg-slate-950/70 px-4 py-3">
                <div className="text-sm font-semibold text-white">
                  Positions ({positions.length})
                </div>
              </div>

              <div className="divide-y divide-slate-800">
                {positions.map(
                  (position, index) => (
                    <div
                      key={`${position.instrument_key}-${index}`}
                      className="grid gap-3 px-4 py-3 md:grid-cols-6 md:items-center"
                    >
                      <div>
                        <div className="text-sm font-medium text-white">
                          {position.symbol}
                        </div>
                        <div className="text-[10px] text-slate-500">
                          {position.instrument_key}
                        </div>
                      </div>

                      <div className="text-xs text-slate-400">
                        Qty{" "}
                        <span className="text-white">
                          {formatNumber(
                            position.quantity,
                            0
                          )}
                        </span>
                      </div>

                      <div className="text-xs text-slate-400">
                        Avg{" "}
                        <span className="text-white">
                          {formatCurrency(
                            position.average_price
                          )}
                        </span>
                      </div>

                      <div className="text-xs text-slate-400">
                        Current{" "}
                        <span className="text-white">
                          {formatCurrency(
                            position.current_price
                          )}
                        </span>
                      </div>

                      <div className="text-xs text-slate-400">
                        Sector{" "}
                        <span className="text-white">
                          {position.sector}
                        </span>
                      </div>

                      <div className="flex justify-end">
                        <button
                          type="button"
                          onClick={() =>
                            removePosition(index)
                          }
                          className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-1.5 text-xs font-medium text-red-400 transition hover:bg-red-500/20"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}

          <div className="mt-5 flex justify-end">
            <button
              type="button"
              onClick={runAnalysis}
              disabled={
                loading || positions.length === 0
              }
              className="rounded-xl bg-cyan-500 px-6 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading
                ? "Analyzing Portfolio..."
                : "Analyze Portfolio"}
            </button>
          </div>
        </section>

        {/* Empty State */}
        {!analysis && (
          <section className="rounded-2xl border border-dashed border-slate-800 bg-slate-900/40 p-10 text-center">
            <div className="mx-auto max-w-lg">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-400">
                📊
              </div>

              <h2 className="text-lg font-semibold text-white">
                Portfolio analysis is ready
              </h2>

              <p className="mt-2 text-sm leading-6 text-slate-500">
                Add your portfolio positions and run the
                analysis to see performance, allocation,
                diversification, attribution, and health
                metrics.
              </p>
            </div>
          </section>
        )}

        {analysis && (
          <>
            {/* Top Summary */}
            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Portfolio Health
                </div>

                <div className="mt-3 flex items-end justify-between">
                  <div>
                    <div className="text-3xl font-bold text-white">
                      {formatNumber(
                        analysis.health.health_score
                      )}
                    </div>

                    <div
                      className={`mt-2 inline-flex rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase ${getStatusClass(
                        analysis.health
                          .health_status
                      )}`}
                    >
                      {analysis.health.health_status}
                    </div>
                  </div>

                  <div className="text-4xl">
                    {analysis.health.health_score >=
                    70
                      ? "🟢"
                      : analysis.health
                          .health_score >= 50
                      ? "🟡"
                      : "🔴"}
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Total Portfolio Value
                </div>

                <div className="mt-3 text-3xl font-bold text-white">
                  {formatCurrency(
                    analysis.summary.total_value
                  )}
                </div>

                <div
                  className={`mt-2 text-sm font-medium ${getPnlClass(
                    analysis.summary.total_pnl
                  )}`}
                >
                  {analysis.summary.total_pnl >=
                  0
                    ? "+"
                    : ""}
                  {formatCurrency(
                    analysis.summary.total_pnl
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Total Return
                </div>

                <div
                  className={`mt-3 text-3xl font-bold ${getPnlClass(
                    analysis.summary
                      .total_return_percent
                  )}`}
                >
                  {analysis.summary
                    .total_return_percent >= 0
                    ? "+"
                    : ""}
                  {formatPercent(
                    analysis.summary
                      .total_return_percent
                  )}
                </div>

                <div className="mt-2 text-xs text-slate-500">
                  {analysis.performance
                    .winning_positions} winning ·{" "}
                  {analysis.performance
                    .losing_positions} losing
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Positions
                </div>

                <div className="mt-3 text-3xl font-bold text-white">
                  {analysis.summary.position_count}
                </div>

                <div
                  className={`mt-2 inline-flex rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase ${getStatusClass(
                    analysis.summary
                      .portfolio_status
                  )}`}
                >
                  {analysis.summary.portfolio_status}
                </div>
              </div>
            </section>

            {/* Capital Overview */}
            <section className="grid gap-4 md:grid-cols-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs text-slate-500">
                  Initial Capital
                </div>
                <div className="mt-2 text-xl font-semibold text-white">
                  {formatCurrency(
                    analysis.portfolio
                      .initial_capital
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs text-slate-500">
                  Invested Value
                </div>
                <div className="mt-2 text-xl font-semibold text-white">
                  {formatCurrency(
                    analysis.portfolio
                      .market_value
                  )}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {formatPercent(
                    analysis.summary
                      .invested_percent
                  )}{" "}
                  of portfolio
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs text-slate-500">
                  Cash
                </div>
                <div className="mt-2 text-xl font-semibold text-white">
                  {formatCurrency(
                    analysis.portfolio.cash
                  )}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {formatPercent(
                    analysis.summary.cash_percent
                  )}{" "}
                  of portfolio
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="text-xs text-slate-500">
                  Unrealized P&L
                </div>
                <div
                  className={`mt-2 text-xl font-semibold ${getPnlClass(
                    analysis.performance
                      .unrealized_pnl
                  )}`}
                >
                  {analysis.performance
                    .unrealized_pnl >= 0
                    ? "+"
                    : ""}
                  {formatCurrency(
                    analysis.performance
                      .unrealized_pnl
                  )}
                </div>
              </div>
            </section>

            {/* Health Score Breakdown */}
            <section className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="mb-5">
                  <h2 className="text-base font-semibold text-white">
                    Health Score Breakdown
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Phase 13 Layer 6 health components.
                  </p>
                </div>

                <div className="space-y-5">
                  <ScoreBar
                    label={`Performance · ${formatPercent(
                      analysis.health
                        .performance_weight *
                        100,
                        0
                    )}`}
                    score={
                      analysis.health
                        .performance_score
                    }
                  />

                  <ScoreBar
                    label={`Diversification · ${formatPercent(
                      analysis.health
                        .diversification_weight *
                        100,
                        0
                    )}`}
                    score={
                      analysis.health
                        .diversification_score
                    }
                  />

                  <ScoreBar
                    label={`Allocation · ${formatPercent(
                      analysis.health
                        .allocation_weight *
                        100,
                        0
                    )}`}
                    score={
                      analysis.health
                        .allocation_score
                    }
                  />

                  <ScoreBar
                    label={`Attribution · ${formatPercent(
                      analysis.health
                        .attribution_weight *
                        100,
                        0
                    )}`}
                    score={
                      analysis.health
                        .attribution_score
                    }
                  />

                  <ScoreBar
                    label={`Cash · ${formatPercent(
                      analysis.health
                        .cash_weight *
                        100,
                        0
                    )}`}
                    score={
                      analysis.health.cash_score
                    }
                  />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-white">
                      Diversification
                    </h2>
                    <p className="mt-1 text-xs text-slate-500">
                      Portfolio concentration analysis.
                    </p>
                  </div>

                  <span
                    className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase ${getStatusClass(
                      analysis.diversification
                        .diversification_status
                    )}`}
                  >
                    {
                      analysis.diversification
                        .diversification_status
                    }
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-xl bg-slate-950 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">
                      Positions
                    </div>
                    <div className="mt-2 text-xl font-semibold text-white">
                      {
                        analysis.diversification
                          .position_count
                      }
                    </div>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">
                      Sectors
                    </div>
                    <div className="mt-2 text-xl font-semibold text-white">
                      {
                        analysis.diversification
                          .sector_count
                      }
                    </div>
                  </div>

                  <div className="rounded-xl bg-slate-950 p-4">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">
                      Asset Types
                    </div>
                    <div className="mt-2 text-xl font-semibold text-white">
                      {
                        analysis.diversification
                          .asset_type_count
                      }
                    </div>
                  </div>
                </div>

                <div className="mt-5 space-y-4">
                  <ScoreBar
                    label="Overall Diversification"
                    score={
                      analysis.diversification
                        .overall_diversification_score
                    }
                  />

                  <ScoreBar
                    label="Position Diversification"
                    score={
                      analysis.diversification
                        .position_diversification_score
                    }
                  />

                  <ScoreBar
                    label="Sector Diversification"
                    score={
                      analysis.diversification
                        .sector_diversification_score
                    }
                  />
                </div>

                <div className="mt-5 flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950 px-4 py-3">
                  <span className="text-xs text-slate-500">
                    Concentration Level
                  </span>

                  <span className="text-sm font-semibold uppercase text-red-400">
                    {
                      analysis.diversification
                        .concentration_level
                    }
                  </span>
                </div>
              </div>
            </section>

            {/* Allocation */}
            <section className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-white">
                      Sector Allocation
                    </h2>
                    <p className="mt-1 text-xs text-slate-500">
                      Market-value based allocation.
                    </p>
                  </div>

                  <span
                    className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase ${getStatusClass(
                      analysis.allocation
                        .allocation_status
                    )}`}
                  >
                    {
                      analysis.allocation
                        .allocation_status
                    }
                  </span>
                </div>

                <div className="space-y-5">
                  {analysis.allocation
                    .sector_allocations.length ===
                  0 ? (
                    <div className="text-sm text-slate-500">
                      No sector allocation data.
                    </div>
                  ) : (
                    analysis.allocation.sector_allocations.map(
                      (sector) => (
                        <AllocationBar
                          key={sector.name}
                          name={sector.name}
                          percentage={
                            sector.percentage
                          }
                        />
                      )
                    )
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="mb-5">
                  <h2 className="text-base font-semibold text-white">
                    Position Allocation
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Largest positions by current market value.
                  </p>
                </div>

                <div className="space-y-5">
                  {analysis.allocation
                    .position_allocations.length ===
                  0 ? (
                    <div className="text-sm text-slate-500">
                      No position allocation data.
                    </div>
                  ) : (
                    analysis.allocation.position_allocations.map(
                      (position) => (
                        <AllocationBar
                          key={
                            position.name
                          }
                          name={
                            position.name
                          }
                          percentage={
                            position.percentage
                          }
                        />
                      )
                    )
                  )}
                </div>
              </div>
            </section>

            {/* Position Table */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900/70">
              <div className="border-b border-slate-800 px-5 py-4">
                <h2 className="text-base font-semibold text-white">
                  Position Analysis
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  Performance and portfolio contribution by position.
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[850px]">
                  <thead>
                    <tr className="border-b border-slate-800 text-left">
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Position
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Quantity
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Invested
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Market Value
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        P&L
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Return
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Weight
                      </th>
                      <th className="px-5 py-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                        Status
                      </th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-slate-800">
                    {analysis.portfolio.positions.map(
                      (position) => {
                        const sourcePosition =
                          positions.find(
                            (item) =>
                              item.instrument_key ===
                              position.instrument_key
                          );

                        return (
                          <tr
                            key={
                              position.instrument_key
                            }
                            className="transition hover:bg-slate-800/30"
                          >
                            <td className="px-5 py-4">
                              <div className="font-medium text-white">
                                {position.symbol}
                              </div>
                              <div className="mt-0.5 text-[10px] text-slate-500">
                                {position.sector}
                              </div>
                            </td>

                            <td className="px-5 py-4 text-sm text-slate-300">
                              {formatNumber(
                                sourcePosition
                                  ?.quantity ||
                                  position.quantity,
                                0
                              )}
                            </td>

                            <td className="px-5 py-4 text-sm text-slate-300">
                              {formatCurrency(
                                position.invested_value
                              )}
                            </td>

                            <td className="px-5 py-4 text-sm text-slate-300">
                              {formatCurrency(
                                position.market_value
                              )}
                            </td>

                            <td
                              className={`px-5 py-4 text-sm font-semibold ${getPnlClass(
                                position.pnl
                              )}`}
                            >
                              {position.pnl >=
                              0
                                ? "+"
                                : ""}
                              {formatCurrency(
                                position.pnl
                              )}
                            </td>

                            <td
                              className={`px-5 py-4 text-sm font-semibold ${getPnlClass(
                                position.pnl_percent
                              )}`}
                            >
                              {position.pnl_percent >=
                              0
                                ? "+"
                                : ""}
                              {formatPercent(
                                position.pnl_percent
                              )}
                            </td>

                            <td className="px-5 py-4 text-sm text-slate-300">
                              {formatPercent(
                                position.portfolio_weight
                              )}
                            </td>

                            <td className="px-5 py-4">
                              <span
                                className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase ${getStatusClass(
                                  position.status
                                )}`}
                              >
                                {
                                  position.status
                                }
                              </span>
                            </td>
                          </tr>
                        );
                      }
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Attribution */}
            <section className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="mb-5">
                  <h2 className="text-base font-semibold text-white">
                    Position Attribution
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Positions contributing to total P&L.
                  </p>
                </div>

                <div className="space-y-3">
                  {analysis.attribution
                    .position_attributions.map(
                      (item) => (
                        <div
                          key={item.name}
                          className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950 px-4 py-3"
                        >
                          <div>
                            <div className="text-sm font-medium text-white">
                              {item.name}
                            </div>
                            <div className="mt-1 text-[11px] text-slate-500">
                              Return{" "}
                              {item.return_percent >=
                              0
                                ? "+"
                                : ""}
                              {formatPercent(
                                item.return_percent
                              )}
                            </div>
                          </div>

                          <div className="text-right">
                            <div
                              className={`text-sm font-semibold ${getPnlClass(
                                item.pnl
                              )}`}
                            >
                              {item.pnl >=
                              0
                                ? "+"
                                : ""}
                              {formatCurrency(
                                item.pnl
                              )}
                            </div>

                            <div className="mt-1 text-[10px] text-slate-500">
                              {formatPercent(
                                item.contribution_percent
                              )}{" "}
                              contribution
                            </div>
                          </div>
                        </div>
                      )
                    )}
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                <div className="mb-5">
                  <h2 className="text-base font-semibold text-white">
                    Sector Attribution
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Sector-level contribution to portfolio P&L.
                  </p>
                </div>

                <div className="space-y-3">
                  {analysis.attribution
                    .sector_attributions.map(
                      (item) => (
                        <div
                          key={item.name}
                          className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950 px-4 py-3"
                        >
                          <div>
                            <div className="text-sm font-medium text-white">
                              {item.name}
                            </div>
                            <div className="mt-1 text-[11px] text-slate-500">
                              {item.position_count}{" "}
                              position
                              {item.position_count ===
                              1
                                ? ""
                                : "s"}
                            </div>
                          </div>

                          <div className="text-right">
                            <div
                              className={`text-sm font-semibold ${getPnlClass(
                                item.pnl
                              )}`}
                            >
                              {item.pnl >=
                              0
                                ? "+"
                                : ""}
                              {formatCurrency(
                                item.pnl
                              )}
                            </div>

                            <div className="mt-1 text-[10px] text-slate-500">
                              {item.return_percent >=
                              0
                                ? "+"
                                : ""}
                              {formatPercent(
                                item.return_percent
                              )}
                            </div>
                          </div>
                        </div>
                      )
                    )}
                </div>
              </div>
            </section>

            {/* Best / Worst */}
            <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-5">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Best Position
                </div>

                <div className="mt-3 text-lg font-semibold text-white">
                  {analysis.attribution
                    .best_position?.name ||
                    "N/A"}
                </div>

                <div className="mt-1 text-sm font-semibold text-emerald-400">
                  {analysis.attribution
                    .best_position
                    ? `+${formatCurrency(
                        analysis.attribution
                          .best_position
                          .pnl
                      )}`
                    : "N/A"}
                </div>
              </div>

              <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-5">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Worst Position
                </div>

                <div className="mt-3 text-lg font-semibold text-white">
                  {analysis.attribution
                    .worst_position?.name ||
                    "N/A"}
                </div>

                <div className="mt-1 text-sm font-semibold text-red-400">
                  {analysis.attribution
                    .worst_position
                    ? formatCurrency(
                        analysis.attribution
                          .worst_position
                          .pnl
                      )
                    : "N/A"}
                </div>
              </div>

              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-5">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Best Sector
                </div>

                <div className="mt-3 text-lg font-semibold text-white">
                  {analysis.attribution
                    .best_sector?.name ||
                    "N/A"}
                </div>

                <div className="mt-1 text-sm font-semibold text-emerald-400">
                  {analysis.attribution
                    .best_sector
                    ? `+${formatCurrency(
                        analysis.attribution
                          .best_sector
                          .pnl
                      )}`
                    : "N/A"}
                </div>
              </div>

              <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-5">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Worst Sector
                </div>

                <div className="mt-3 text-lg font-semibold text-white">
                  {analysis.attribution
                    .worst_sector?.name ||
                    "N/A"}
                </div>

                <div className="mt-1 text-sm font-semibold text-red-400">
                  {analysis.attribution
                    .worst_sector
                    ? formatCurrency(
                        analysis.attribution
                          .worst_sector
                          .pnl
                      )
                    : "N/A"}
                </div>
              </div>
            </section>

            {/* Concerns / Recommendations */}
            <section className="grid gap-5 lg:grid-cols-3">
              <div className="rounded-2xl border border-emerald-500/20 bg-slate-900/70 p-5">
                <h2 className="text-base font-semibold text-white">
                  Strengths
                </h2>

                <div className="mt-4 space-y-3">
                  {analysis.health.strengths.map(
                    (item, index) => (
                      <div
                        key={index}
                        className="flex gap-3 rounded-xl border border-emerald-500/10 bg-emerald-500/5 p-3"
                      >
                        <span className="text-emerald-400">
                          ✓
                        </span>

                        <span className="text-xs leading-5 text-slate-300">
                          {item}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-amber-500/20 bg-slate-900/70 p-5">
                <h2 className="text-base font-semibold text-white">
                  Concerns
                </h2>

                <div className="mt-4 space-y-3">
                  {analysis.health.concerns.map(
                    (item, index) => (
                      <div
                        key={index}
                        className="flex gap-3 rounded-xl border border-amber-500/10 bg-amber-500/5 p-3"
                      >
                        <span className="text-amber-400">
                          !
                        </span>

                        <span className="text-xs leading-5 text-slate-300">
                          {item}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-cyan-500/20 bg-slate-900/70 p-5">
                <h2 className="text-base font-semibold text-white">
                  Recommendations
                </h2>

                <div className="mt-4 space-y-3">
                  {analysis.health.recommendations.map(
                    (item, index) => (
                      <div
                        key={index}
                        className="flex gap-3 rounded-xl border border-cyan-500/10 bg-cyan-500/5 p-3"
                      >
                        <span className="text-cyan-400">
                          →
                        </span>

                        <span className="text-xs leading-5 text-slate-300">
                          {item}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </div>
            </section>

            {/* Footer Metadata */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900/50 px-5 py-4">
              <div className="flex flex-col gap-2 text-[11px] text-slate-500 md:flex-row md:items-center md:justify-between">
                <div>
                  Phase{" "}
                  {analysis.metadata.phase} · Layer{" "}
                  {analysis.metadata.layer} ·{" "}
                  {analysis.metadata.component}
                </div>

                <div>
                  Currency:{" "}
                  {analysis.metadata.currency} ·
                  Layers included:{" "}
                  {analysis.metadata.layers_included.join(
                    ", "
                  )}
                </div>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
};

export default Phase13PortfolioAnalysisPage;