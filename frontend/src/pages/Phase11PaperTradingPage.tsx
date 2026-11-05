import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleDollarSign,
  Clock3,
  Play,
  Pause,
  Square,
  RefreshCw,
  Search,
  ShoppingCart,
  TrendingDown,
  TrendingUp,
  Wallet,
  XCircle,
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type SessionStatus =
  | "CREATED"
  | "RUNNING"
  | "PAUSED"
  | "STOPPED"
  | "COMPLETED"
  | string;

type OrderStatus =
  | "PENDING"
  | "FILLED"
  | "CANCELLED"
  | "REJECTED"
  | string;

type OrderSide = "BUY" | "SELL";

type Session = {
  session_id: string;
  status: SessionStatus;
  instrument_key: string;
  created_at?: string;
  started_at?: string | null;
  stopped_at?: string | null;
  total_orders?: number;
  total_trades?: number;
  config?: {
    instrument_key?: string;
    initial_capital?: number;
    commission_percent?: number;
    slippage_percent?: number;
    max_position_value_percent?: number;
    currency?: string;
  };
};

type Order = {
  order_id: string;
  session_id: string;
  instrument_key: string;
  side: OrderSide;
  order_type?: string;
  quantity: number;
  requested_price?: number | null;
  status: OrderStatus;
  filled_price?: number | null;
  filled_quantity?: number;
  created_at?: string;
  filled_at?: string | null;
  rejection_reason?: string | null;
};

type Portfolio = {
  cash?: number;
  invested_value?: number;
  market_value?: number;
  total_equity?: number;
  realized_pnl?: number;
  unrealized_pnl?: number;
  total_pnl?: number;
  total_return_percent?: number;
  open_positions?: number;
  total_trades?: number;
};

type Metrics = {
  initial_capital?: number;
  final_equity?: number;
  total_pnl?: number;
  total_return_percent?: number;
  total_trades?: number;
  winning_trades?: number;
  losing_trades?: number;
  breakeven_trades?: number;
  win_rate?: number;
  average_win?: number;
  average_loss?: number;
  profit_factor?: number;
  max_drawdown?: number;
  max_drawdown_percent?: number;
};

type Trade = {
  trade_id: string;
  session_id: string;
  instrument_key: string;
  side: OrderSide;
  quantity: number;
  entry_price: number;
  exit_price: number;
  entry_time?: string;
  exit_time?: string;
  gross_pnl?: number;
  commission?: number;
  slippage_cost?: number;
  net_pnl?: number;
  signal?: string | null;
  confidence?: number | null;
};

type StockSearchResult = {
  name?: string;
  symbol?: string;
  exchange?: string;
  instrument_key: string;
  isin?: string;
  short_name?: string;
  instrument_type?: string;
  segment?: string;
};

function formatCurrency(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "₹--";
  }

  return `₹${Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "--";
  }

  return Number(value).toFixed(digits);
}

function formatPercent(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "--";
  }

  return `${Number(value).toFixed(2)}%`;
}

function extractObject<T>(
  response: Record<string, any>,
  keys: string[]
): T | null {
  for (const key of keys) {
    if (response[key] && typeof response[key] === "object") {
      return response[key] as T;
    }
  }

  return null;
}

function StatusBadge({ status }: { status: SessionStatus }) {
  const normalized = status.toUpperCase();

  const classes =
    normalized === "RUNNING"
      ? "bg-emerald-400/10 text-emerald-300 border-emerald-400/20"
      : normalized === "PAUSED"
      ? "bg-amber-400/10 text-amber-300 border-amber-400/20"
      : normalized === "STOPPED" || normalized === "COMPLETED"
      ? "bg-red-400/10 text-red-300 border-red-400/20"
      : "bg-slate-700/50 text-slate-300 border-slate-700";

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${classes}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {normalized}
    </span>
  );
}

function Section({
  title,
  icon,
  children,
  right,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-xl shadow-black/10">
      <div className="mb-5 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-slate-800 p-2 text-slate-200">
            {icon}
          </div>

          <div>
            <h2 className="text-lg font-semibold text-white">{title}</h2>
          </div>
        </div>

        {right}
      </div>

      {children}
    </section>
  );
}

function MetricCard({
  label,
  value,
  icon,
  positive,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  positive?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">{label}</span>

        <span
          className={
            positive === true
              ? "text-emerald-400"
              : positive === false
              ? "text-red-400"
              : "text-slate-500"
          }
        >
          {icon}
        </span>
      </div>

      <p className="mt-2 text-xl font-semibold text-white">{value}</p>
    </div>
  );
}

export default function Phase11PaperTradingPage() {
  const [session, setSession] = useState<Session | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);

  const [instrumentKey, setInstrumentKey] = useState(
    "NSE_EQ|INE081A01020"
  );

  const [stockSearch, setStockSearch] = useState("TATASTEEL");
  const [stockSearchResults, setStockSearchResults] = useState<
    StockSearchResult[]
  >([]);
  const [stockSearchLoading, setStockSearchLoading] = useState(false);
  const [stockSearchOpen, setStockSearchOpen] = useState(false);

  const [initialCapital, setInitialCapital] = useState("100000");
  const [commissionPercent, setCommissionPercent] = useState("0");
  const [slippagePercent, setSlippagePercent] = useState("0");

  const [signal, setSignal] = useState("BUY");
  const [confidence, setConfidence] = useState("0.80");
  const [currentPrice, setCurrentPrice] = useState("");
  const [currentPriceLoading, setCurrentPriceLoading] = useState(false);
  const [currentPriceError, setCurrentPriceError] = useState("");
  const [quantity, setQuantity] = useState("1");

  const [marketPrice, setMarketPrice] = useState("");

  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const apiRequest = useCallback(
    async <T,>(url: string, options?: RequestInit): Promise<T> => {
      const response = await fetch(`${API_BASE_URL}${url}`, {
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          ...(options?.headers || {}),
        },
        ...options,
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            `API request failed with HTTP ${response.status}`
        );
      }

      if (data?.status === "error") {
        throw new Error(data.message || "API returned an error");
      }

      return data as T;
    },
    []
  );

  useEffect(() => {
    const query = stockSearch.trim();

    if (query.length < 2 || session) {
      setStockSearchResults([]);
      setStockSearchLoading(false);
      return;
    }

    let cancelled = false;

    const timer = window.setTimeout(async () => {
      try {
        setStockSearchLoading(true);

        const response = await apiRequest<Record<string, any>>(
          `/api/stocks/search?query=${encodeURIComponent(query)}`
        );

        const results = Array.isArray(response?.data)
          ? response.data
          : [];

        if (!cancelled) {
          setStockSearchResults(results as StockSearchResult[]);
          setStockSearchOpen(true);
        }
      } catch (err) {
        if (!cancelled) {
          setStockSearchResults([]);
          setError(
            err instanceof Error
              ? err.message
              : "Unable to search stocks."
          );
        }
      } finally {
        if (!cancelled) {
          setStockSearchLoading(false);
        }
      }
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [stockSearch, session, apiRequest]);

  const loadSessionData = useCallback(
    async (sessionId: string, showRefreshing = false) => {
      try {
        if (showRefreshing) {
          setRefreshing(true);
        }

        const [
          sessionResponse,
          ordersResponse,
          portfolioResponse,
          metricsResponse,
          tradesResponse,
        ] = await Promise.all([
          apiRequest<Record<string, any>>(
            `/api/paper-trading/sessions/${encodeURIComponent(sessionId)}`
          ),
          apiRequest<Record<string, any>>(
            `/api/paper-trading/sessions/${encodeURIComponent(
              sessionId
            )}/orders`
          ),
          apiRequest<Record<string, any>>(
            `/api/paper-trading/sessions/${encodeURIComponent(
              sessionId
            )}/portfolio`
          ),
          apiRequest<Record<string, any>>(
            `/api/paper-trading/sessions/${encodeURIComponent(
              sessionId
            )}/metrics`
          ),
          apiRequest<Record<string, any>>(
            `/api/paper-trading/sessions/${encodeURIComponent(
              sessionId
            )}/trades`
          ),
        ]);

        const sessionData = extractObject<Session>(sessionResponse, [
          "session",
          "data",
        ]);

        const portfolioData = extractObject<Portfolio>(
          portfolioResponse,
          ["portfolio", "data"]
        );

        const metricsData = extractObject<Metrics>(metricsResponse, [
          "metrics",
          "data",
        ]);

        const tradeData =
          (Array.isArray(tradesResponse.trades)
            ? tradesResponse.trades
            : Array.isArray(tradesResponse.data)
            ? tradesResponse.data
            : []) as Trade[];

        const orderData =
          (Array.isArray(ordersResponse.orders)
            ? ordersResponse.orders
            : Array.isArray(ordersResponse.data)
            ? ordersResponse.data
            : []) as Order[];

        if (sessionData) {
          setSession(sessionData);
        }

        if (portfolioData) {
          setPortfolio(portfolioData);
        }

        if (metricsData) {
          setMetrics(metricsData);
        }

        setOrders(orderData);
        setTrades(tradeData);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to refresh Phase 11 data."
        );
      } finally {
        setRefreshing(false);
      }
    },
    [apiRequest]
  );

  useEffect(() => {
    if (!session?.session_id) {
      return;
    }

    const interval = window.setInterval(() => {
      loadSessionData(session.session_id);
    }, 5000);

    return () => {
      window.clearInterval(interval);
    };
  }, [session?.session_id, loadSessionData]);

  const selectStock = async (stock: StockSearchResult) => {
    setInstrumentKey(stock.instrument_key);
    setStockSearch(stock.symbol || stock.name || "");
    setStockSearchResults([]);
    setStockSearchOpen(false);
    setError("");
    setCurrentPriceError("");
    setCurrentPriceLoading(true);

    try {
      const response = await apiRequest<Record<string, any>>(
        `/api/stocks/quote?instrument_key=${encodeURIComponent(
          stock.instrument_key
        )}`
      );

      if (
        response?.status === "error" ||
        typeof response?.value !== "number" ||
        !Number.isFinite(response.value) ||
        response.value <= 0
      ) {
        throw new Error(
          response?.message || "Current market price is unavailable."
        );
      }

      setCurrentPrice(String(response.value));
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

  const createSession = async () => {
    try {
      setLoading(true);
      setError("");
      setSuccess("");

      const response = await apiRequest<Record<string, any>>(
        "/api/paper-trading/sessions",
        {
          method: "POST",
          body: JSON.stringify({
            instrument_key: instrumentKey.trim(),
            initial_capital: Number(initialCapital),
            commission_percent: Number(commissionPercent),
            slippage_percent: Number(slippagePercent),
            max_position_value_percent: 100,
          }),
        }
      );

      const newSession = extractObject<Session>(response, [
        "session",
        "data",
      ]);

      if (!newSession?.session_id) {
        throw new Error("Session was created but no session_id was returned.");
      }

      setSession(newSession);
      setStockSearchOpen(false);
      setStockSearchResults([]);
      setOrders([]);
      setPortfolio(null);
      setMetrics(null);
      setTrades([]);

      setSuccess(`Paper trading session ${newSession.session_id} created.`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to create session."
      );
    } finally {
      setLoading(false);
    }
  };

  const sessionAction = async (
    action: "start" | "pause" | "stop"
  ) => {
    if (!session?.session_id) return;

    try {
      setLoading(true);
      setError("");
      setSuccess("");

      await apiRequest(
        `/api/paper-trading/sessions/${encodeURIComponent(
          session.session_id
        )}/${action}`,
        {
          method: "POST",
        }
      );

      await loadSessionData(session.session_id);

      setSuccess(`Session ${action} request completed.`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : `Unable to ${action} session.`
      );
    } finally {
      setLoading(false);
    }
  };

  const createOrder = async () => {
    if (!session?.session_id) {
      setError("Create a paper trading session first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setSuccess("");

      await apiRequest(
        `/api/paper-trading/sessions/${encodeURIComponent(
          session.session_id
        )}/orders`,
        {
          method: "POST",
          body: JSON.stringify({
            signal,
            confidence: Number(confidence),
            current_price: Number(currentPrice),
            quantity: Number(quantity),
          }),
        }
      );

      await loadSessionData(session.session_id);

      setSuccess(`${signal} order created successfully.`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to create order."
      );
    } finally {
      setLoading(false);
    }
  };

  const executeOrder = async (orderId: string) => {
    if (!session?.session_id) return;

    try {
      setLoading(true);
      setError("");
      setSuccess("");

      await apiRequest(
        `/api/paper-trading/orders/${encodeURIComponent(
          orderId
        )}/execute`,
        {
          method: "POST",
          body: JSON.stringify({
            market_price: Number(marketPrice),
          }),
        }
      );

      await loadSessionData(session.session_id);

      setSuccess(`Order ${orderId} executed successfully.`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to execute order."
      );
    } finally {
      setLoading(false);
    }
  };

  const cancelOrder = async (orderId: string) => {
    try {
      setLoading(true);
      setError("");
      setSuccess("");

      await apiRequest(
        `/api/paper-trading/orders/${encodeURIComponent(
          orderId
        )}/cancel`,
        {
          method: "POST",
        }
      );

      if (session?.session_id) {
        await loadSessionData(session.session_id);
      }

      setSuccess(`Order ${orderId} cancelled.`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to cancel order."
      );
    } finally {
      setLoading(false);
    }
  };

  const updatePrice = async () => {
    if (!session?.session_id) {
      setError("Create a session first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setSuccess("");

      await apiRequest(
        `/api/paper-trading/sessions/${encodeURIComponent(
          session.session_id
        )}/portfolio/price`,
        {
          method: "POST",
          body: JSON.stringify({
            current_price: Number(marketPrice),
          }),
        }
      );

      await loadSessionData(session.session_id);

      setSuccess("Portfolio market price updated.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to update portfolio price."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetPage = () => {
    setSession(null);
    setStockSearch("TATASTEEL");
    setInstrumentKey("NSE_EQ|INE081A01020");
    setStockSearchResults([]);
    setStockSearchOpen(false);
    setCurrentPrice("");
    setCurrentPriceError("");
    setCurrentPriceLoading(false);
    setOrders([]);
    setPortfolio(null);
    setMetrics(null);
    setTrades([]);
    setError("");
    setSuccess("");
  };

  const sessionRunning = session?.status === "RUNNING";
  const hasPendingOrders = orders.some(
    (order) => order.status === "PENDING"
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              <Activity size={14} />
              Phase 11
            </div>

            <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Paper Trading
            </h1>

            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Simulated trading environment using the Phase 11 paper
              trading engine. No real broker orders are sent.
            </p>
          </div>

          <div className="flex gap-2">
            {session && (
              <button
                onClick={() =>
                  loadSessionData(session.session_id, true)
                }
                disabled={refreshing}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-slate-800 disabled:opacity-50"
              >
                <RefreshCw
                  size={16}
                  className={refreshing ? "animate-spin" : ""}
                />
                Refresh
              </button>
            )}

            <button
              onClick={resetPage}
              className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-400 transition hover:bg-slate-800 hover:text-white"
            >
              New Session
            </button>
          </div>
        </div>

        {/* Alerts */}
        {error && (
          <div className="mb-5 flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
            <XCircle size={18} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="mb-5 flex items-start gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4 text-sm text-emerald-300">
            <CheckCircle2 size={18} className="mt-0.5 shrink-0" />
            <span>{success}</span>
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Session configuration */}
          <Section
            title="Trading Session"
            icon={<Clock3 size={18} />}
            right={
              session ? <StatusBadge status={session.status} /> : null
            }
          >
            {!session ? (
              <div className="space-y-4">
                <div className="relative">
                  <label className="mb-1.5 block text-xs font-medium text-slate-400">
                    Search Stock
                  </label>

                  <div className="relative">
                    <Search
                      size={16}
                      className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                    />

                    <input
                      value={stockSearch}
                      onChange={(e) => {
                        setStockSearch(e.target.value);
                        setStockSearchOpen(true);
                        setInstrumentKey("");
                        setCurrentPrice("");
                        setCurrentPriceError("");
                      }}
                      onFocus={() => {
                        if (stockSearchResults.length > 0) {
                          setStockSearchOpen(true);
                        }
                      }}
                      placeholder="Search TATASTEEL, TCS, INFY..."
                      className="w-full rounded-xl border border-slate-700 bg-slate-950 py-2.5 pl-10 pr-10 text-sm text-white outline-none transition focus:border-slate-500"
                    />

                    {stockSearchLoading && (
                      <RefreshCw
                        size={15}
                        className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin text-slate-500"
                      />
                    )}
                  </div>

                  {stockSearchOpen && stockSearch.trim().length >= 2 && (
                    <div className="absolute left-0 right-0 z-30 mt-2 max-h-64 overflow-y-auto rounded-xl border border-slate-700 bg-slate-950 shadow-2xl">
                      {stockSearchResults.length > 0 ? (
                        stockSearchResults.map((stock) => (
                          <button
                            key={stock.instrument_key}
                            type="button"
                            onClick={() => selectStock(stock)}
                            className="w-full border-b border-slate-800 px-3 py-3 text-left transition last:border-b-0 hover:bg-slate-900"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <p className="truncate text-sm font-semibold text-white">
                                  {stock.symbol || stock.short_name || "--"}
                                </p>
                                <p className="truncate text-xs text-slate-400">
                                  {stock.name || "--"} · {stock.exchange || "NSE"}
                                </p>
                                <p className="mt-1 truncate font-mono text-[10px] text-slate-600">
                                  {stock.instrument_key}
                                </p>
                              </div>

                              <span className="shrink-0 rounded-md border border-cyan-400/20 bg-cyan-400/10 px-2 py-1 text-[10px] font-medium text-cyan-300">
                                NSE
                              </span>
                            </div>
                          </button>
                        ))
                      ) : stockSearchLoading ? (
                        <div className="px-3 py-4 text-center text-xs text-slate-500">
                          Searching stocks...
                        </div>
                      ) : (
                        <div className="px-3 py-4 text-center text-xs text-slate-500">
                          No NSE stocks found.
                        </div>
                      )}
                    </div>
                  )}

                  {instrumentKey && (
                    <div className="mt-2 rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-wide text-slate-600">
                        Selected Instrument Key
                      </p>
                      <p className="mt-1 break-all font-mono text-[11px] text-slate-400">
                        {instrumentKey}
                      </p>
                    </div>
                  )}

                  <p className="mt-1.5 text-[11px] text-slate-500">
                    Search and select an NSE stock. The instrument key is filled automatically.
                  </p>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium text-slate-400">
                    Initial Capital
                  </label>

                  <input
                    type="number"
                    min="1"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(e.target.value)}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-slate-400">
                      Commission %
                    </label>

                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={commissionPercent}
                      onChange={(e) =>
                        setCommissionPercent(e.target.value)
                      }
                      className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                    />
                  </div>

                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-slate-400">
                      Slippage %
                    </label>

                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={slippagePercent}
                      onChange={(e) =>
                        setSlippagePercent(e.target.value)
                      }
                      className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                    />
                  </div>
                </div>

                <button
                  onClick={createSession}
                  disabled={
                    loading ||
                    !instrumentKey.trim() ||
                    Number(initialCapital) <= 0
                  }
                  className="w-full rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? "Creating..." : "Create Paper Session"}
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
                  <p className="text-xs text-slate-500">Session ID</p>
                  <p className="mt-1 break-all font-mono text-sm text-white">
                    {session.session_id}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-slate-950/50 p-3">
                    <p className="text-[11px] text-slate-500">
                      Instrument
                    </p>
                    <p className="mt-1 truncate text-sm font-medium text-white">
                      {session.instrument_key}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-950/50 p-3">
                    <p className="text-[11px] text-slate-500">
                      Orders
                    </p>
                    <p className="mt-1 text-sm font-medium text-white">
                      {session.total_orders ?? orders.length}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {session.status === "CREATED" && (
                    <button
                      onClick={() => sessionAction("start")}
                      disabled={loading}
                      className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
                    >
                      <Play size={15} />
                      Start
                    </button>
                  )}

                  {session.status === "RUNNING" && (
                    <button
                      onClick={() => sessionAction("pause")}
                      disabled={loading}
                      className="inline-flex items-center gap-2 rounded-xl bg-amber-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-amber-400 disabled:opacity-50"
                    >
                      <Pause size={15} />
                      Pause
                    </button>
                  )}

                  {session.status === "PAUSED" && (
                    <button
                      onClick={() => sessionAction("start")}
                      disabled={loading}
                      className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
                    >
                      <Play size={15} />
                      Resume
                    </button>
                  )}

                  {(session.status === "RUNNING" ||
                    session.status === "PAUSED") && (
                    <button
                      onClick={() => sessionAction("stop")}
                      disabled={loading}
                      className="inline-flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm font-semibold text-red-300 hover:bg-red-500/20 disabled:opacity-50"
                    >
                      <Square size={15} />
                      Stop
                    </button>
                  )}
                </div>
              </div>
            )}
          </Section>

          {/* Order creation */}
          <Section
            title="Create Order"
            icon={<ShoppingCart size={18} />}
            right={
              <span className="rounded-full bg-slate-800 px-2.5 py-1 text-[10px] text-slate-400">
                Simulated
              </span>
            }
          >
            <div className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-slate-400">
                  Signal
                </label>

                <select
                  value={signal}
                  onChange={(e) => setSignal(e.target.value)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                >
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                  <option value="LONG">LONG</option>
                  <option value="SHORT">SHORT</option>
                  <option value="BULLISH">BULLISH</option>
                  <option value="BEARISH">BEARISH</option>
                  <option value="NEUTRAL">NEUTRAL</option>
                  <option value="HOLD">HOLD</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1.5 block text-xs font-medium text-slate-400">
                    Confidence
                  </label>

                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={confidence}
                    onChange={(e) => setConfidence(e.target.value)}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                  />
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium text-slate-400">
                    Quantity
                  </label>

                  <input
                    type="number"
                    min="0.0001"
                    step="1"
                    value={quantity}
                    onChange={(e) => setQuantity(e.target.value)}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                  />
                </div>
              </div>

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
                    onChange={(e) => setCurrentPrice(e.target.value)}
                    placeholder="Select a stock to load price"
                    className="w-full rounded-xl border border-emerald-500/30 bg-slate-950 px-3 py-2.5 pr-20 text-sm text-white outline-none focus:border-cyan-400"
                  />

                  <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
                    {currentPriceLoading ? "Loading..." : "Live"}
                  </span>
                </div>

                {currentPriceError ? (
                  <p className="mt-1.5 text-[11px] text-red-400">
                    {currentPriceError}
                  </p>
                ) : (
                  <p className="mt-1.5 text-[11px] text-slate-500">
                    Automatically filled from the selected stock's latest quote.
                  </p>
                )}
              </div>

              <button
                onClick={createOrder}
                disabled={
                  loading ||
                  !sessionRunning ||
                  Number(currentPrice) <= 0 ||
                  Number(quantity) <= 0
                }
                className="w-full rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Processing..." : "Create Order"}
              </button>

              {!sessionRunning && (
                <p className="text-xs text-slate-500">
                  Start the paper trading session before creating an order.
                </p>
              )}
            </div>
          </Section>

          {/* Market price */}
          <Section
            title="Portfolio Price"
            icon={<CircleDollarSign size={18} />}
          >
            <div className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-slate-400">
                  Current Market Price
                </label>

                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={marketPrice}
                  onChange={(e) => setMarketPrice(e.target.value)}
                  placeholder="Example: 152.50"
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none focus:border-slate-500"
                />
              </div>

              <button
                onClick={updatePrice}
                disabled={
                  loading ||
                  !session ||
                  Number(marketPrice) <= 0
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-800 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Update Portfolio Price
              </button>

              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <p className="text-xs text-slate-500">
                  Purpose
                </p>

                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Updating the price recalculates the open position's
                  market value and unrealized P&L.
                </p>
              </div>
            </div>
          </Section>
        </div>

        {/* Portfolio */}
        <div className="mt-6">
          <Section
            title="Virtual Portfolio"
            icon={<Wallet size={18} />}
            right={
              portfolio ? (
                <span className="text-xs text-slate-500">
                  Paper account
                </span>
              ) : null
            }
          >
            {!portfolio ? (
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/30 p-8 text-center">
                <Wallet
                  size={28}
                  className="mx-auto text-slate-700"
                />
                <p className="mt-3 text-sm text-slate-500">
                  Portfolio data will appear after a session is created.
                </p>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Cash"
                  value={formatCurrency(portfolio.cash)}
                  icon={<Wallet size={16} />}
                />

                <MetricCard
                  label="Market Value"
                  value={formatCurrency(portfolio.market_value)}
                  icon={<BarChart3 size={16} />}
                />

                <MetricCard
                  label="Total Equity"
                  value={formatCurrency(portfolio.total_equity)}
                  icon={<CircleDollarSign size={16} />}
                />

                <MetricCard
                  label="Total P&L"
                  value={formatCurrency(portfolio.total_pnl)}
                  positive={
                    portfolio.total_pnl != null
                      ? portfolio.total_pnl > 0
                      : undefined
                  }
                  icon={
                    (portfolio.total_pnl ?? 0) >= 0 ? (
                      <TrendingUp size={16} />
                    ) : (
                      <TrendingDown size={16} />
                    )
                  }
                />

                <MetricCard
                  label="Realized P&L"
                  value={formatCurrency(portfolio.realized_pnl)}
                  positive={
                    portfolio.realized_pnl != null
                      ? portfolio.realized_pnl > 0
                      : undefined
                  }
                  icon={<CheckCircle2 size={16} />}
                />

                <MetricCard
                  label="Unrealized P&L"
                  value={formatCurrency(portfolio.unrealized_pnl)}
                  positive={
                    portfolio.unrealized_pnl != null
                      ? portfolio.unrealized_pnl > 0
                      : undefined
                  }
                  icon={<Activity size={16} />}
                />

                <MetricCard
                  label="Return"
                  value={formatPercent(
                    portfolio.total_return_percent
                  )}
                  positive={
                    portfolio.total_return_percent != null
                      ? portfolio.total_return_percent > 0
                      : undefined
                  }
                  icon={<TrendingUp size={16} />}
                />

                <MetricCard
                  label="Open Positions"
                  value={String(portfolio.open_positions ?? 0)}
                  icon={<ShoppingCart size={16} />}
                />
              </div>
            )}
          </Section>
        </div>

        {/* Orders */}
        <div className="mt-6">
          <Section
            title="Orders"
            icon={<ShoppingCart size={18} />}
            right={
              <span className="text-xs text-slate-500">
                {orders.length} order{orders.length === 1 ? "" : "s"}
              </span>
            }
          >
            {orders.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/30 p-8 text-center">
                <ShoppingCart
                  size={28}
                  className="mx-auto text-slate-700"
                />
                <p className="mt-3 text-sm text-slate-500">
                  No orders have been created yet.
                </p>
              </div>
            ) : (
              <>
                {hasPendingOrders && (
                  <div className="mb-4 rounded-xl border border-blue-500/20 bg-blue-500/10 p-3 text-xs text-blue-300">
                    Enter a market price above to execute a pending
                    order.
                  </div>
                )}

                <div className="overflow-x-auto">
                  <table className="w-full min-w-[900px] text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-800 text-xs text-slate-500">
                        <th className="px-3 py-3">Order</th>
                        <th className="px-3 py-3">Side</th>
                        <th className="px-3 py-3">Quantity</th>
                        <th className="px-3 py-3">Requested</th>
                        <th className="px-3 py-3">Filled</th>
                        <th className="px-3 py-3">Status</th>
                        <th className="px-3 py-3">Action</th>
                      </tr>
                    </thead>

                    <tbody>
                      {orders.map((order) => (
                        <tr
                          key={order.order_id}
                          className="border-b border-slate-900 last:border-0"
                        >
                          <td className="px-3 py-4">
                            <p className="font-mono text-xs text-slate-300">
                              {order.order_id}
                            </p>
                          </td>

                          <td className="px-3 py-4">
                            <span
                              className={
                                order.side === "BUY"
                                  ? "rounded-full bg-emerald-400/10 px-2.5 py-1 text-xs font-medium text-emerald-300"
                                  : "rounded-full bg-red-400/10 px-2.5 py-1 text-xs font-medium text-red-300"
                              }
                            >
                              {order.side}
                            </span>
                          </td>

                          <td className="px-3 py-4 text-slate-300">
                            {formatNumber(order.quantity, 4)}
                          </td>

                          <td className="px-3 py-4 text-slate-300">
                            {formatCurrency(order.requested_price)}
                          </td>

                          <td className="px-3 py-4 text-slate-300">
                            {formatCurrency(order.filled_price)}
                          </td>

                          <td className="px-3 py-4">
                            <span className="text-xs text-slate-400">
                              {order.status}
                            </span>
                          </td>

                          <td className="px-3 py-4">
                            {order.status === "PENDING" ? (
                              <div className="flex gap-2">
                                <button
                                  onClick={() =>
                                    executeOrder(order.order_id)
                                  }
                                  disabled={
                                    loading ||
                                    Number(marketPrice) <= 0
                                  }
                                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-40"
                                >
                                  <Play size={13} />
                                  Execute
                                </button>

                                <button
                                  onClick={() =>
                                    cancelOrder(order.order_id)
                                  }
                                  disabled={loading}
                                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-40"
                                >
                                  <XCircle size={13} />
                                  Cancel
                                </button>
                              </div>
                            ) : (
                              <span className="text-xs text-slate-600">
                                —
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </Section>
        </div>

        {/* Metrics */}
        <div className="mt-6">
          <Section
            title="Trading Metrics"
            icon={<BarChart3 size={18} />}
          >
            {!metrics ? (
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/30 p-8 text-center">
                <BarChart3
                  size={28}
                  className="mx-auto text-slate-700"
                />
                <p className="mt-3 text-sm text-slate-500">
                  Metrics will appear after trades are recorded.
                </p>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Initial Capital"
                  value={formatCurrency(metrics.initial_capital)}
                  icon={<CircleDollarSign size={16} />}
                />

                <MetricCard
                  label="Final Equity"
                  value={formatCurrency(metrics.final_equity)}
                  icon={<Wallet size={16} />}
                />

                <MetricCard
                  label="Total P&L"
                  value={formatCurrency(metrics.total_pnl)}
                  positive={
                    metrics.total_pnl != null
                      ? metrics.total_pnl > 0
                      : undefined
                  }
                  icon={<TrendingUp size={16} />}
                />

                <MetricCard
                  label="Return"
                  value={formatPercent(
                    metrics.total_return_percent
                  )}
                  positive={
                    metrics.total_return_percent != null
                      ? metrics.total_return_percent > 0
                      : undefined
                  }
                  icon={<TrendingUp size={16} />}
                />

                <MetricCard
                  label="Total Trades"
                  value={String(metrics.total_trades ?? 0)}
                  icon={<Activity size={16} />}
                />

                <MetricCard
                  label="Win Rate"
                  value={formatPercent(metrics.win_rate)}
                  icon={<CheckCircle2 size={16} />}
                />

                <MetricCard
                  label="Profit Factor"
                  value={formatNumber(metrics.profit_factor)}
                  icon={<BarChart3 size={16} />}
                />

                <MetricCard
                  label="Max Drawdown"
                  value={formatCurrency(metrics.max_drawdown)}
                  positive={false}
                  icon={<TrendingDown size={16} />}
                />
              </div>
            )}
          </Section>
        </div>

        {/* Trade History */}
        <div className="mt-6">
          <Section
            title="Trade History"
            icon={<Activity size={18} />}
            right={
              <span className="text-xs text-slate-500">
                {trades.length} completed trade
                {trades.length === 1 ? "" : "s"}
              </span>
            }
          >
            {trades.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/30 p-8 text-center">
                <Activity
                  size={28}
                  className="mx-auto text-slate-700"
                />

                <p className="mt-3 text-sm text-slate-500">
                  No completed trades yet.
                </p>

                <p className="mt-1 text-xs text-slate-600">
                  A completed trade requires an entry and exit.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[950px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 text-xs text-slate-500">
                      <th className="px-3 py-3">Trade</th>
                      <th className="px-3 py-3">Side</th>
                      <th className="px-3 py-3">Qty</th>
                      <th className="px-3 py-3">Entry</th>
                      <th className="px-3 py-3">Exit</th>
                      <th className="px-3 py-3">Gross P&L</th>
                      <th className="px-3 py-3">Net P&L</th>
                      <th className="px-3 py-3">Confidence</th>
                    </tr>
                  </thead>

                  <tbody>
                    {trades.map((trade) => (
                      <tr
                        key={trade.trade_id}
                        className="border-b border-slate-900 last:border-0"
                      >
                        <td className="px-3 py-4">
                          <span className="font-mono text-xs text-slate-300">
                            {trade.trade_id}
                          </span>
                        </td>

                        <td className="px-3 py-4">
                          <span
                            className={
                              trade.side === "BUY"
                                ? "rounded-full bg-emerald-400/10 px-2.5 py-1 text-xs text-emerald-300"
                                : "rounded-full bg-red-400/10 px-2.5 py-1 text-xs text-red-300"
                            }
                          >
                            {trade.side}
                          </span>
                        </td>

                        <td className="px-3 py-4 text-slate-300">
                          {formatNumber(trade.quantity, 4)}
                        </td>

                        <td className="px-3 py-4 text-slate-300">
                          {formatCurrency(trade.entry_price)}
                        </td>

                        <td className="px-3 py-4 text-slate-300">
                          {formatCurrency(trade.exit_price)}
                        </td>

                        <td className="px-3 py-4 text-slate-300">
                          {formatCurrency(trade.gross_pnl)}
                        </td>

                        <td
                          className={`px-3 py-4 font-semibold ${
                            (trade.net_pnl ?? 0) >= 0
                              ? "text-emerald-400"
                              : "text-red-400"
                          }`}
                        >
                          {formatCurrency(trade.net_pnl)}
                        </td>

                        <td className="px-3 py-4 text-slate-400">
                          {trade.confidence != null
                            ? formatNumber(trade.confidence, 3)
                            : "--"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>
        </div>

        {/* Footer note */}
        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-center">
          <p className="text-xs leading-5 text-slate-500">
            Phase 11 is a virtual paper-trading environment. Orders are
            simulated by the backend execution engine and are not sent to
            Upstox or any real broker.
          </p>
        </div>
      </main>
    </div>
  );
}