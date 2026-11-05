import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import StockDetails from "./pages/StockDetails";
import ScannerPage from "./pages/ScannerPage";
import StockSearch from "./components/StockSearch";
import Phase8Dashboard from "./components/Phase8Dashboard";
import AIIntelligencePage from "./pages/AIIntelligencePage";
import Phase10ValidationPage from "./pages/Phase10ValidationPage";
import Phase11PaperTradingPage from "./pages/Phase11PaperTradingPage";
import Phase12RiskManagementPage from "./pages/Phase12RiskManagementPage";
import Phase13PortfolioAnalysisPage from "./pages/Phase13PortfolioAnalysisPage";
import Phase14AdvancedAIPage from "./pages/Phase14AdvancedAIPage";
import Phase15RiskStabilityPage from "./pages/Phase15RiskStabilityPage";
import { useEffect, useState } from "react";

import {
  getMarketOverview,
  getMarketActivity,
  type MarketOverview,
  type MarketActivity,
} from "./services/marketApi";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type ChartPoint = {
  time: string;
  value: number;
};

type Mover = {
  name: string;
  symbol: string;
  price: string;
  change: string;
  netChange?: number;
  volume?: number | null;
};

type MoversResponse = {
  status: string;
  source?: string;
  total_instruments?: number;
  quoted_instruments?: number;

  gainers?: Array<{
    name?: string;
    symbol?: string;
    value?: number;
    change?: number;
    net_change?: number;
    volume?: number | null;
  }>;

  losers?: Array<{
    name?: string;
    symbol?: string;
    value?: number;
    change?: number;
    net_change?: number;
    volume?: number | null;
  }>;

  message?: string;
};

type Period = "1W" | "1M" | "6M" | "1Y";

const watchlist = [
  {
    symbol: "RELIANCE",
    name: "Reliance Industries",
    price: "₹1,384.20",
    change: "+1.24%",
  },
  {
    symbol: "TCS",
    name: "Tata Consultancy Services",
    price: "₹3,841.70",
    change: "-2.34%",
  },
  {
    symbol: "INFY",
    name: "Infosys",
    price: "₹1,482.40",
    change: "-3.42%",
  },
];

/* =========================================================
   COMMON APPLICATION HEADER
   ========================================================= */

function AppHeader() {
  const location = useLocation();

  const isActive = (path: string) => {
    if (path === "/") {
      return location.pathname === "/";
    }

    return location.pathname === path;
  };

  const linkClass = (path: string) =>
    `transition ${
      isActive(path)
        ? "font-medium text-cyan-400"
        : "text-slate-400 hover:text-white"
    }`;

  return (
    <header className="border-b border-slate-800 bg-slate-950/95">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-10">
          {/* Logo */}
          <a
            href="/"
            className="flex items-center gap-2.5 text-white no-underline"
          >
            <img
              src="/logo.png"
              alt="StockAI"
              className="h-9 w-9 rounded-lg object-cover"
            />

            <span className="text-xl font-bold tracking-tight">
              StockAI
            </span>
          </a>

          {/* Main Navigation */}
          <nav className="hidden gap-7 text-sm md:flex">
            <a href="/" className={linkClass("/")}>
              Dashboard
            </a>

            <a href="/markets" className={linkClass("/markets")}>
              Markets
            </a>

            <a href="/scanner" className={linkClass("/scanner")}>
              Scanner
            </a>

            <a href="/watchlist" className={linkClass("/watchlist")}>
              Watchlist
            </a>

            <a href="/ai" className={linkClass("/ai")}>
              AI Intelligence
            </a>

            <a href="/phase10" className={linkClass("/phase10")}>
              AI Validation
            </a>

            <a href="/phase11" className={linkClass("/phase11")}>
              Paper Trading
            </a>
            <a href="/phase12" className={linkClass("/phase12")}>
              Risk Management
            </a>
            <a href="/phase13" className={linkClass("/phase13")}>
              Portfolio Analysis
            </a>
            <a href="/phase14" className={linkClass("/phase14")}>
              Advanced AI
            </a>
            <a href="/phase15" className={linkClass("/phase15")}>
              Risk & Stability
            </a>
          </nav>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-4">
          <button
            type="button"
            className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:border-slate-500"
            title="Notifications"
          >
            🔔
          </button>

          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-cyan-500 font-semibold text-slate-950">
            V
          </div>
        </div>
      </div>
    </header>
  );
}

/* =========================================================
   DASHBOARD
   ========================================================= */

function Dashboard() {
  const [market, setMarket] = useState<MarketOverview | null>(null);
  const [loading, setLoading] = useState(true);

  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [chartLoading, setChartLoading] = useState(true);
  const [chartError, setChartError] = useState("");
  const [selectedPeriod, setSelectedPeriod] =
    useState<Period>("1M");

  const [movers, setMovers] = useState<{
    gainers: Mover[];
    losers: Mover[];
    source: string;
    message: string;
  }>({
    gainers: [],
    losers: [],
    source: "",
    message: "",
  });

  const [moversLoading, setMoversLoading] = useState(true);
  const [moversError, setMoversError] = useState("");

  const [marketActivity, setMarketActivity] =
    useState<MarketActivity | null>(null);

  const [activityLoading, setActivityLoading] = useState(true);
  const [activityError, setActivityError] = useState("");

  /* ---------------------------------------------------------
     Load live market overview
     --------------------------------------------------------- */

  useEffect(() => {
    let mounted = true;

    const loadMarket = async () => {
      try {
        const data = await getMarketOverview();

        if (mounted) {
          setMarket(data);
        }
      } catch (error) {
        console.error("Market API error:", error);

        if (mounted) {
          setMarket(null);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadMarket();

    const interval = window.setInterval(loadMarket, 30_000);

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  /* ---------------------------------------------------------
     Load REAL Top Gainers / Top Losers from Upstox
     --------------------------------------------------------- */

  useEffect(() => {
    let mounted = true;

    const loadMovers = async () => {
      try {
        if (mounted) {
          setMoversLoading(true);
          setMoversError("");
        }

        const response = await fetch(
          `${API_BASE_URL}/api/market/movers?limit=5`,
          {
            method: "GET",
            headers: {
              Accept: "application/json",
            },
          }
        );

        if (!response.ok) {
          throw new Error(
            `Movers API returned HTTP ${response.status}`
          );
        }

        const data: MoversResponse = await response.json();

        if (data.status !== "success") {
          throw new Error(
            data.message || "Unable to load market movers."
          );
        }

        const mapMover = (stock: {
          name?: string;
          symbol?: string;
          value?: number;
          change?: number;
          net_change?: number;
          volume?: number | null;
        }): Mover => {
          const change = Number(stock.change ?? 0);
          const value = Number(stock.value ?? 0);

          return {
            name: stock.name || stock.symbol || "Unknown",
            symbol: stock.symbol || "UNKNOWN",
            price: Number.isFinite(value)
              ? `₹${value.toLocaleString("en-IN", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}`
              : "₹--",
            change: `${change > 0 ? "+" : ""}${change.toFixed(2)}%`,
            netChange:
              stock.net_change != null
                ? Number(stock.net_change)
                : undefined,
            volume: stock.volume ?? null,
          };
        };

        const realGainers = (data.gainers || [])
          .map(mapMover)
          .filter(
            (stock) =>
              Number.parseFloat(stock.change) > 0
          );

        const realLosers = (data.losers || [])
          .map(mapMover)
          .filter(
            (stock) =>
              Number.parseFloat(stock.change) < 0
          );

        if (!mounted) return;

        setMovers({
          gainers: realGainers,
          losers: realLosers,
          source: data.source || "upstox_live",
          message: data.message || "",
        });
      } catch (error) {
        console.error("Market movers API error:", error);

        if (mounted) {
          setMovers({
            gainers: [],
            losers: [],
            source: "",
            message: "",
          });

          setMoversError(
            error instanceof Error
              ? error.message
              : "Unable to load Top Gainers / Top Losers."
          );
        }
      } finally {
        if (mounted) {
          setMoversLoading(false);
        }
      }
    };

    loadMovers();

    const interval = window.setInterval(
      loadMovers,
      60_000
    );

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  /* ---------------------------------------------------------
     Load Market Breadth + Most Active Stocks
     --------------------------------------------------------- */

  useEffect(() => {
    let mounted = true;

    const loadMarketActivity = async () => {
      try {
        if (mounted) {
          setActivityLoading(true);
          setActivityError("");
        }

        const data = await getMarketActivity(10);

        if (!mounted) return;

        if (data.status !== "success") {
          throw new Error(
            data.message ||
              "Unable to load market activity."
          );
        }

        setMarketActivity(data);
      } catch (error) {
        console.error(
          "Market activity API error:",
          error
        );

        if (mounted) {
          setMarketActivity(null);

          setActivityError(
            error instanceof Error
              ? error.message
              : "Unable to load market breadth and active stocks."
          );
        }
      } finally {
        if (mounted) {
          setActivityLoading(false);
        }
      }
    };

    loadMarketActivity();

    const interval = window.setInterval(
      loadMarketActivity,
      60_000
    );

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  /* ---------------------------------------------------------
     Calculate date range for NIFTY historical chart
     --------------------------------------------------------- */

  const getDateRange = (period: Period) => {
    const today = new Date();

    const toDate = today
      .toISOString()
      .split("T")[0];

    const from = new Date(today);

    switch (period) {
      case "1W":
        from.setDate(from.getDate() - 7);
        break;

      case "1M":
        from.setMonth(from.getMonth() - 1);
        break;

      case "6M":
        from.setMonth(from.getMonth() - 6);
        break;

      case "1Y":
        from.setFullYear(from.getFullYear() - 1);
        break;
    }

    return {
      fromDate: from.toISOString().split("T")[0],
      toDate,
    };
  };

  /* ---------------------------------------------------------
     Load REAL NIFTY historical data from backend / Upstox
     --------------------------------------------------------- */

  useEffect(() => {
    let cancelled = false;

    const loadNiftyHistory = async () => {
      try {
        setChartLoading(true);
        setChartError("");

        const { fromDate, toDate } =
          getDateRange(selectedPeriod);

        const url =
          `${API_BASE_URL}/api/stocks/history` +
          `?instrument_key=${encodeURIComponent(
            "NSE_INDEX|Nifty 50"
          )}` +
          `&unit=days` +
          `&interval=1` +
          `&to_date=${encodeURIComponent(toDate)}` +
          `&from_date=${encodeURIComponent(fromDate)}`;

        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(
            `Historical API returned HTTP ${response.status}`
          );
        }

        const data = await response.json();

        if (data.status !== "success") {
          throw new Error(
            data.message ||
              "Unable to load NIFTY historical data"
          );
        }

        const candles = Array.isArray(data.candles)
          ? data.candles
          : [];

        if (candles.length === 0) {
          throw new Error(
            "Upstox returned no historical NIFTY candles for this period."
          );
        }

        const formatted: ChartPoint[] = [...candles]
          .reverse()
          .filter(
            (candle) =>
              candle &&
              typeof candle === "object" &&
              typeof candle.timestamp === "string" &&
              typeof candle.close === "number"
          )
          .map((candle) => {
            const date = new Date(
              candle.timestamp
            );

            return {
              time: date.toLocaleDateString(
                "en-IN",
                {
                  day: "2-digit",
                  month: "short",
                }
              ),
              value: Number(candle.close),
            };
          });

        if (formatted.length === 0) {
          throw new Error(
            "No valid candle data was returned by Upstox."
          );
        }

        if (!cancelled) {
          setChartData(formatted);
        }
      } catch (error) {
        console.error(
          "NIFTY history error:",
          error
        );

        if (!cancelled) {
          setChartData([]);

          setChartError(
            error instanceof Error
              ? error.message
              : "Unable to load NIFTY historical data."
          );
        }
      } finally {
        if (!cancelled) {
          setChartLoading(false);
        }
      }
    };

    loadNiftyHistory();

    return () => {
      cancelled = true;
    };
  }, [selectedPeriod]);

  const niftyChange =
    market?.nifty?.change ?? 0;

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <main className="mx-auto max-w-7xl px-6 py-8">

        {/* Welcome */}
        <section className="mb-8">
          <p className="text-sm text-slate-400">
            {new Date().toLocaleDateString(
              "en-IN",
              {
                weekday: "long",
                month: "long",
                day: "numeric",
                year: "numeric",
              }
            )}
          </p>

          <h1 className="mt-2 text-3xl font-bold tracking-tight">
            Good evening, Vinith 👋
          </h1>

          <p className="mt-2 text-slate-400">
            Here&apos;s what&apos;s happening in the Indian market today.
          </p>

          <div className="mt-6">
            <StockSearch />
          </div>
        </section>

        {/* Market Cards */}
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MarketCard
            name="NIFTY 50"
            value={
              loading
                ? "Loading..."
                : market?.nifty?.value != null
                  ? market.nifty.value.toLocaleString(
                      "en-IN",
                      {
                        minimumFractionDigits: 2,
                      }
                    )
                  : "Unavailable"
            }
            change={
              loading
                ? "..."
                : market?.nifty?.change != null
                  ? `${
                      market.nifty.change > 0
                        ? "+"
                        : ""
                    }${market.nifty.change.toFixed(2)}%`
                  : "--"
            }
            negative={niftyChange < 0}
          />

          <MarketCard
            name="SENSEX"
            value={
              loading
                ? "Loading..."
                : market?.sensex?.value != null
                  ? market.sensex.value.toLocaleString(
                      "en-IN",
                      {
                        minimumFractionDigits: 2,
                      }
                    )
                  : "Unavailable"
            }
            change={
              loading
                ? "..."
                : market?.sensex?.change != null
                  ? `${
                      market.sensex.change > 0
                        ? "+"
                        : ""
                    }${market.sensex.change.toFixed(2)}%`
                  : "--"
            }
            negative={
              (market?.sensex?.change ?? 0) < 0
            }
          />

          <MarketCard
            name="BANK NIFTY"
            value={
              loading
                ? "Loading..."
                : market?.bank_nifty?.value != null
                  ? market.bank_nifty.value.toLocaleString(
                      "en-IN",
                      {
                        minimumFractionDigits: 2,
                      }
                    )
                  : "Unavailable"
            }
            change={
              loading
                ? "..."
                : market?.bank_nifty?.change != null
                  ? `${
                      market.bank_nifty.change > 0
                        ? "+"
                        : ""
                    }${market.bank_nifty.change.toFixed(2)}%`
                  : "--"
            }
            negative={
              (market?.bank_nifty?.change ?? 0) < 0
            }
          />

          <MarketCard
            name="INDIA VIX"
            value={
              loading
                ? "Loading..."
                : market?.india_vix?.value != null
                  ? market.india_vix.value.toFixed(2)
                  : "Unavailable"
            }
            change={
              loading
                ? "..."
                : market?.india_vix?.change != null
                  ? `${
                      market.india_vix.change > 0
                        ? "+"
                        : ""
                    }${market.india_vix.change.toFixed(2)}%`
                  : "--"
            }
            negative={
              (market?.india_vix?.change ?? 0) > 0
            }
          />
        </section>

        {/* Chart + AI Score */}
        <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_320px]">

          {/* NIFTY Chart */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
              <div>
                <p className="text-sm text-slate-400">
                  NIFTY 50 • Upstox Historical Data
                </p>

                <div className="mt-1 flex items-end gap-3">
                  <h2 className="text-3xl font-bold">
                    {market?.nifty?.value != null
                      ? market.nifty.value.toLocaleString(
                          "en-IN",
                          {
                            minimumFractionDigits: 2,
                          }
                        )
                      : "Loading..."}
                  </h2>

                  <span
                    className={`pb-1 text-sm font-medium ${
                      niftyChange < 0
                        ? "text-red-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {market?.nifty?.change != null
                      ? `${
                          market.nifty.change > 0
                            ? "+"
                            : ""
                        }${market.nifty.change.toFixed(2)}%`
                      : "..."}
                  </span>
                </div>
              </div>

              <div className="flex gap-2">
                {(
                  ["1W", "1M", "6M", "1Y"] as Period[]
                ).map((period) => (
                  <button
                    type="button"
                    key={period}
                    onClick={() =>
                      setSelectedPeriod(period)
                    }
                    className={`rounded-lg px-3 py-2 text-xs font-medium transition ${
                      selectedPeriod === period
                        ? "bg-cyan-400 text-slate-950"
                        : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white"
                    }`}
                  >
                    {period}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-6 h-[310px]">
              {chartLoading ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  Loading real NIFTY historical data...
                </div>
              ) : chartError ? (
                <div className="flex h-full flex-col items-center justify-center px-6 text-center">
                  <p className="text-sm text-red-400">
                    Unable to load NIFTY chart
                  </p>

                  <p className="mt-2 max-w-lg text-xs leading-5 text-slate-500">
                    {chartError}
                  </p>
                </div>
              ) : chartData.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  No NIFTY historical data available.
                </div>
              ) : (
                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >
                  <AreaChart
                    data={chartData}
                    margin={{
                      top: 10,
                      right: 10,
                      left: 0,
                      bottom: 0,
                    }}
                  >
                    <defs>
                      <linearGradient
                        id="marketGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopOpacity={0.35}
                        />
                        <stop
                          offset="100%"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>

                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1e293b"
                    />

                    <XAxis
                      dataKey="time"
                      tick={{
                        fill: "#64748b",
                        fontSize: 12,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      domain={[
                        "dataMin - 100",
                        "dataMax + 100",
                      ]}
                      tick={{
                        fill: "#64748b",
                        fontSize: 12,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <Tooltip
                      formatter={(value) => [
                        Number(value).toLocaleString(
                          "en-IN",
                          {
                            minimumFractionDigits: 2,
                          }
                        ),
                        "NIFTY 50",
                      ]}
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: "10px",
                        color: "#fff",
                      }}
                    />

                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#22d3ee"
                      strokeWidth={2}
                      fill="url(#marketGradient)"
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* AI Market Score */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">
                  AI Market Score
                </p>

                <h2 className="mt-1 text-lg font-semibold">
                  Today&apos;s Outlook
                </h2>
              </div>

              <span className="rounded-full bg-yellow-400/10 px-3 py-1 text-xs font-medium text-yellow-300">
                Neutral
              </span>
            </div>

            <div className="my-8 flex justify-center">
              <div className="flex h-40 w-40 items-center justify-center rounded-full border-[12px] border-yellow-400/70">
                <div className="text-center">
                  <div className="text-4xl font-bold">
                    64
                  </div>

                  <div className="text-xs text-slate-400">
                    / 100
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <ScoreRow name="Technical" score={72} />
              <ScoreRow name="Momentum" score={68} />
              <ScoreRow name="Sentiment" score={61} />
              <ScoreRow name="Volume" score={57} />
            </div>

            <p className="mt-6 rounded-xl bg-slate-800/70 p-4 text-xs leading-5 text-slate-400">
              AI predictions will be connected in a later milestone.
              The market cards and NIFTY chart are connected to
              the backend/Upstox data.
            </p>
          </div>
        </section>

        {/* REAL Top Gainers / Top Losers */}
        <section className="mt-6 grid gap-6 lg:grid-cols-2">
          {moversLoading ? (
            <>
              <StockListLoading title="Top Gainers" />
              <StockListLoading title="Top Losers" />
            </>
          ) : moversError ? (
            <>
              <StockList
                title="Top Gainers"
                subtitle="Unable to load live Upstox movers"
                stocks={[]}
                positive
                emptyMessage={moversError}
              />

              <StockList
                title="Top Losers"
                subtitle="Unable to load live Upstox movers"
                stocks={[]}
                emptyMessage={moversError}
              />
            </>
          ) : (
            <>
              <StockList
                title="Top Gainers"
                subtitle={
                  movers.source ===
                  "latest_trading_session"
                    ? "Latest available trading session"
                    : "Live Upstox market data"
                }
                stocks={movers.gainers}
                positive
                emptyMessage={
                  movers.message ||
                  "No positive movers are available right now."
                }
              />

              <StockList
                title="Top Losers"
                subtitle={
                  movers.source ===
                  "latest_trading_session"
                    ? "Latest available trading session"
                    : "Live Upstox market data"
                }
                stocks={movers.losers}
                emptyMessage={
                  movers.message ||
                  "No negative movers are available right now."
                }
              />
            </>
          )}
        </section>

        {movers.source && (
          <p className="mt-3 text-right text-xs text-slate-600">
            Movers source:{" "}
            {movers.source ===
            "latest_trading_session"
              ? "latest trading session"
              : "live Upstox"}
          </p>
        )}

        {/* Market Breadth + Most Active Stocks */}
        <section className="mt-6 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">

          {/* Market Breadth */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-slate-400">
                  NSE Market Breadth
                </p>

                <h2 className="mt-1 text-xl font-semibold">
                  Advances vs Declines
                </h2>
              </div>

              <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-[10px] font-medium text-cyan-300">
                {marketActivity?.source ===
                "upstox_live"
                  ? "Upstox"
                  : marketActivity?.source ||
                    "Market"}
              </span>
            </div>

            {activityLoading ? (
              <MarketActivityLoading />
            ) : activityError ? (
              <div className="mt-6 rounded-xl border border-red-500/20 bg-red-500/5 p-5">
                <p className="text-sm text-red-400">
                  Unable to load market breadth
                </p>

                <p className="mt-2 text-xs leading-5 text-slate-500">
                  {activityError}
                </p>
              </div>
            ) : (
              <>
                <div className="mt-7 grid grid-cols-3 gap-3">
                  <BreadthStat
                    label="Advances"
                    value={
                      marketActivity?.breadth
                        .advances ?? 0
                    }
                    ratio={
                      marketActivity?.breadth
                        .advance_ratio ?? 0
                    }
                    positive
                  />

                  <BreadthStat
                    label="Declines"
                    value={
                      marketActivity?.breadth
                        .declines ?? 0
                    }
                    ratio={
                      marketActivity?.breadth
                        .decline_ratio ?? 0
                    }
                  />

                  <BreadthStat
                    label="Unchanged"
                    value={
                      marketActivity?.breadth
                        .unchanged ?? 0
                    }
                    ratio={
                      marketActivity?.breadth
                        .unchanged_ratio ?? 0
                    }
                    neutral
                  />
                </div>

                <div className="mt-6">
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="text-slate-500">
                      Market breadth
                    </span>

                    <span className="text-slate-400">
                      {marketActivity?.breadth
                        .total ?? 0}{" "}
                      stocks
                    </span>
                  </div>

                  <div className="flex h-3 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="bg-emerald-400"
                      style={{
                        width: `${
                          marketActivity?.breadth
                            .advance_ratio ?? 0
                        }%`,
                      }}
                    />

                    <div
                      className="bg-red-400"
                      style={{
                        width: `${
                          marketActivity?.breadth
                            .decline_ratio ?? 0
                        }%`,
                      }}
                    />

                    <div
                      className="bg-slate-500"
                      style={{
                        width: `${
                          marketActivity?.breadth
                            .unchanged_ratio ?? 0
                        }%`,
                      }}
                    />
                  </div>

                  <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-500">
                    <span>
                      <span className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-400" />
                      Advances
                    </span>

                    <span>
                      <span className="mr-1 inline-block h-2 w-2 rounded-full bg-red-400" />
                      Declines
                    </span>

                    <span>
                      <span className="mr-1 inline-block h-2 w-2 rounded-full bg-slate-500" />
                      Unchanged
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Most Active Stocks */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-slate-400">
                  NSE Most Active
                </p>

                <h2 className="mt-1 text-xl font-semibold">
                  Highest Traded Volume
                </h2>
              </div>

              <span className="rounded-full bg-yellow-400/10 px-3 py-1 text-[10px] font-medium text-yellow-300">
                Live
              </span>
            </div>

            {activityLoading ? (
              <MarketActivityTableLoading />
            ) : activityError ? (
              <div className="mt-6 rounded-xl border border-red-500/20 bg-red-500/5 p-5">
                <p className="text-sm text-red-400">
                  Unable to load active stocks
                </p>

                <p className="mt-2 text-xs leading-5 text-slate-500">
                  {activityError}
                </p>
              </div>
            ) : (
              <div className="mt-5 overflow-x-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 text-left text-xs text-slate-500">
                      <th className="pb-3 pr-3 font-medium">
                        #
                      </th>

                      <th className="pb-3 pr-3 font-medium">
                        Stock
                      </th>

                      <th className="pb-3 pr-3 text-right font-medium">
                        Price
                      </th>

                      <th className="pb-3 pr-3 text-right font-medium">
                        Change
                      </th>

                      <th className="pb-3 text-right font-medium">
                        Volume
                      </th>
                    </tr>
                  </thead>

                  <tbody>
                    {(
                      marketActivity?.most_active ||
                      []
                    ).map((stock, index) => (
                      <tr
                        key={`${stock.instrument_key}-${index}`}
                        className="border-b border-slate-800/70 last:border-0"
                      >
                        <td className="py-3 pr-3 text-xs text-slate-600">
                          {index + 1}
                        </td>

                        <td className="py-3 pr-3">
                          <p className="max-w-[180px] truncate font-medium text-slate-200">
                            {stock.symbol}
                          </p>

                          <p className="max-w-[180px] truncate text-[11px] text-slate-600">
                            {stock.name}
                          </p>
                        </td>

                        <td className="py-3 pr-3 text-right text-slate-300">
                          ₹
                          {Number(
                            stock.value
                          ).toLocaleString(
                            "en-IN",
                            {
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2,
                            }
                          )}
                        </td>

                        <td
                          className={`py-3 pr-3 text-right font-medium ${
                            stock.change >= 0
                              ? "text-emerald-400"
                              : "text-red-400"
                          }`}
                        >
                          {stock.change >= 0
                            ? "+"
                            : ""}
                          {stock.change.toFixed(2)}%
                        </td>

                        <td className="py-3 text-right text-slate-400">
                          {formatVolume(
                            stock.volume
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {(!marketActivity?.most_active ||
                  marketActivity.most_active
                    .length === 0) && (
                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-8 text-center text-sm text-slate-500">
                    No active-stock data is available right now.
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* Phase 8 — Backtesting Suite */}
        <Phase8Dashboard />

        {/* Watchlist */}
        <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">
                ⭐ My Watchlist
              </h2>

              <p className="mt-1 text-sm text-slate-400">
                Stocks you&apos;re keeping an eye on.
              </p>
            </div>

            <button
              type="button"
              className="rounded-lg bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
            >
              + Add Stock
            </button>
          </div>

          <div className="mt-5 divide-y divide-slate-800">
            {watchlist.map((stock) => (
              <div
                key={stock.symbol}
                className="flex items-center justify-between py-4"
              >
                <div>
                  <p className="font-medium">
                    {stock.symbol}
                  </p>

                  <p className="text-xs text-slate-500">
                    {stock.name}
                  </p>
                </div>

                <div className="text-right">
                  <p className="font-medium">
                    {stock.price}
                  </p>

                  <p
                    className={`text-sm ${
                      stock.change.startsWith("+")
                        ? "text-emerald-400"
                        : "text-red-400"
                    }`}
                  >
                    {stock.change}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Disclaimer */}
        <footer className="py-8 text-center text-xs text-slate-600">
          StockAI provides AI-generated market analysis for
          educational and informational purposes only. It is not
          financial advice.
        </footer>
      </main>
    </div>
  );
}

/* =========================================================
   DASHBOARD COMPONENTS
   ========================================================= */

function MarketCard({
  name,
  value,
  change,
  negative = false,
}: {
  name: string;
  value: string;
  change: string;
  negative?: boolean;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 transition hover:border-slate-700">
      <p className="text-sm text-slate-400">
        {name}
      </p>

      <div className="mt-3 flex items-end justify-between gap-2">
        <p className="text-xl font-semibold">
          {value}
        </p>

        <span
          className={`text-sm font-medium ${
            negative
              ? "text-red-400"
              : "text-emerald-400"
          }`}
        >
          {change}
        </span>
      </div>

      <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full ${
            negative
              ? "bg-red-400"
              : "bg-emerald-400"
          }`}
          style={{
            width: negative ? "38%" : "68%",
          }}
        />
      </div>
    </div>
  );
}

function ScoreRow({
  name,
  score,
}: {
  name: string;
  score: number;
}) {
  return (
    <div>
      <div className="mb-2 flex justify-between text-xs">
        <span className="text-slate-400">
          {name}
        </span>

        <span className="font-medium text-slate-200">
          {score}
        </span>
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-cyan-400"
          style={{
            width: `${score}%`,
          }}
        />
      </div>
    </div>
  );
}

function BreadthStat({
  label,
  value,
  ratio,
  positive = false,
  neutral = false,
}: {
  label: string;
  value: number;
  ratio: number;
  positive?: boolean;
  neutral?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
      <p
        className={`text-xs ${
          positive
            ? "text-emerald-400"
            : neutral
              ? "text-slate-400"
              : "text-red-400"
        }`}
      >
        {label}
      </p>

      <p className="mt-2 text-2xl font-bold text-white">
        {value.toLocaleString("en-IN")}
      </p>

      <p className="mt-1 text-xs text-slate-500">
        {ratio.toFixed(2)}%
      </p>
    </div>
  );
}

function formatVolume(volume: number): string {
  if (!Number.isFinite(volume) || volume <= 0) {
    return "--";
  }

  if (volume >= 10_000_000) {
    return `${(volume / 10_000_000).toFixed(2)} Cr`;
  }

  if (volume >= 100_000) {
    return `${(volume / 100_000).toFixed(2)} L`;
  }

  if (volume >= 1_000) {
    return `${(volume / 1_000).toFixed(1)} K`;
  }

  return volume.toLocaleString("en-IN");
}

function MarketActivityLoading() {
  return (
    <div className="mt-7 grid grid-cols-3 gap-3">
      {[1, 2, 3].map((item) => (
        <div
          key={item}
          className="h-28 animate-pulse rounded-xl bg-slate-800/60"
        />
      ))}
    </div>
  );
}

function MarketActivityTableLoading() {
  return (
    <div className="mt-5 space-y-3">
      {[1, 2, 3, 4, 5].map((item) => (
        <div
          key={item}
          className="flex items-center justify-between rounded-lg bg-slate-950/30 p-3"
        >
          <div className="space-y-2">
            <div className="h-4 w-28 animate-pulse rounded bg-slate-800" />
            <div className="h-3 w-20 animate-pulse rounded bg-slate-800" />
          </div>

          <div className="h-4 w-24 animate-pulse rounded bg-slate-800" />
        </div>
      ))}
    </div>
  );
}

function StockList({
  title,
  subtitle,
  stocks,
  positive = false,
  emptyMessage = "No market data available.",
}: {
  title: string;
  subtitle: string;
  stocks: Mover[];
  positive?: boolean;
  emptyMessage?: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">
            {title}
          </h2>

          <p className="mt-1 text-sm text-slate-400">
            {subtitle}
          </p>
        </div>

        <span
          className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${
            positive
              ? "bg-emerald-400/10 text-emerald-300"
              : "bg-red-400/10 text-red-300"
          }`}
        >
          Upstox
        </span>
      </div>

      {stocks.length === 0 ? (
        <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-8 text-center">
          <p className="text-sm text-slate-400">
            {emptyMessage}
          </p>
        </div>
      ) : (
        <div className="mt-5 divide-y divide-slate-800">
          {stocks.map((stock, index) => (
            <div
              key={`${stock.symbol}-${index}`}
              className="flex items-center justify-between gap-4 py-4"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-medium text-slate-600">
                    #{index + 1}
                  </span>

                  <div className="min-w-0">
                    <p className="truncate font-medium text-white">
                      {stock.name}
                    </p>

                    <p className="mt-0.5 text-xs text-slate-500">
                      {stock.symbol}
                    </p>
                  </div>
                </div>
              </div>

              <div className="shrink-0 text-right">
                <p className="font-medium text-slate-200">
                  {stock.price}
                </p>

                <p
                  className={`text-sm font-semibold ${
                    positive
                      ? "text-emerald-400"
                      : "text-red-400"
                  }`}
                >
                  {stock.change}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StockListLoading({
  title,
}: {
  title: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold">
            {title}
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Loading real Upstox market data...
          </p>
        </div>

        <span className="rounded-full bg-slate-800 px-2.5 py-1 text-[10px] text-slate-500">
          Live
        </span>
      </div>

      <div className="mt-5 space-y-4">
        {[1, 2, 3, 4, 5].map((item) => (
          <div
            key={item}
            className="flex items-center justify-between border-b border-slate-800 py-3 last:border-0"
          >
            <div className="space-y-2">
              <div className="h-4 w-36 animate-pulse rounded bg-slate-800" />
              <div className="h-3 w-20 animate-pulse rounded bg-slate-800" />
            </div>

            <div className="space-y-2 text-right">
              <div className="ml-auto h-4 w-20 animate-pulse rounded bg-slate-800" />
              <div className="ml-auto h-3 w-14 animate-pulse rounded bg-slate-800" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* =========================================================
   APPLICATION
   ========================================================= */

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-950 text-white">

        {/* ONE COMMON HEADER FOR ALL PAGES */}
        <AppHeader />

        <Routes>
          {/* Main Dashboard */}
          <Route
            path="/"
            element={<Dashboard />}
          />

          {/* Stock Details */}
          <Route
            path="/stock/:symbol"
            element={<StockDetails />}
          />

          {/* Phase 5 Scanner */}
          <Route
            path="/scanner"
            element={<ScannerPage />}
          />

          {/* Phase 9 AI */}
          <Route
            path="/ai"
            element={<AIIntelligencePage />}
          />

          {/* Phase 10 */}
          <Route
            path="/phase10"
            element={<Phase10ValidationPage />}
          />

          {/* Phase 11 */}
          <Route
            path="/phase11"
            element={<Phase11PaperTradingPage />}
          />
          {/* Phase 12 */}
          <Route
            path="/phase12"
            element={<Phase12RiskManagementPage />}
          />
          {/* Phase 13 */}
          <Route
            path="/phase13"
            element={<Phase13PortfolioAnalysisPage />}
          />
          <Route
            path="/phase14"
            element={<Phase14AdvancedAIPage />}
          />
          <Route
            path="/phase15"
            element={<Phase15RiskStabilityPage />}
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
