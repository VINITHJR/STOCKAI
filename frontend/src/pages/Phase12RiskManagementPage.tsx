import { useCallback, useEffect, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type RiskEvaluation = {
  status: string;
  level: string;
  action: string;
  risk_type: string;
  message: string;
  current_value: number;
  limit_value: number;
  utilization_percent: number;
  instrument_key: string | null;
  evaluated_at: string;
};

type PositionResponse = {
  position: {
    instrument_key: string;
    quantity: number;
    entry_price: number;
    current_price: number;
    market_value: number;
    exposure_percent: number;
    unrealized_pnl: number;
    unrealized_pnl_percent: number;
    stop_loss_price: number;
    take_profit_price: number;
    risk_status: string;
    risk_level: string;
  };
  evaluation: RiskEvaluation;
  stop_loss_hit: boolean;
  take_profit_hit: boolean;
};

type Alert = {
  alert_id: string;
  risk_type: string;
  level: string;
  status: string;
  message: string;
  instrument_key: string | null;
  current_value: number;
  limit_value: number;
  created_at: string;
  acknowledged: boolean;
};

type HealthResponse = {
  status: string;
  phase: number;
  module: string;
  layers: Record<string, string>;
};

function formatNumber(value: number, digits = 2) {
  return Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function statusClass(status: string) {
  switch (status.toUpperCase()) {
    case "SAFE":
      return "bg-emerald-400/10 text-emerald-300 border-emerald-400/20";
    case "WARNING":
      return "bg-yellow-400/10 text-yellow-300 border-yellow-400/20";
    case "BLOCKED":
      return "bg-red-400/10 text-red-300 border-red-400/20";
    default:
      return "bg-slate-800 text-slate-300 border-slate-700";
  }
}

function StatusBadge({ value }: { value: string }) {
  return (
    <span
      className={`rounded-full border px-2.5 py-1 text-xs font-medium ${statusClass(
        value
      )}`}
    >
      {value}
    </span>
  );
}

function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description?: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-bold text-white">{value}</p>
      {description && (
        <p className="mt-1 text-xs text-slate-500">{description}</p>
      )}
    </div>
  );
}

export default function Phase12RiskManagementPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [position, setPosition] = useState<PositionResponse | null>(null);
  const [positionLoading, setPositionLoading] = useState(false);
  const [positionError, setPositionError] = useState("");
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertsLoading, setAlertsLoading] = useState(false);
  const [alertsError, setAlertsError] = useState("");
  const [alertSummary, setAlertSummary] = useState<Record<string, unknown> | null>(null);

  const [instrumentKey, setInstrumentKey] = useState("NSE_EQ|INE002A01018");
  const [stockSearch, setStockSearch] = useState("");
  const [stockResults, setStockResults] = useState<
    Array<{
      name: string;
      symbol: string;
      instrument_key: string;
      exchange?: string;
    }>
  >([]);
  const [stockSearchLoading, setStockSearchLoading] = useState(false);
  const [showStockResults, setShowStockResults] = useState(false);
  const [currentPriceLoading, setCurrentPriceLoading] = useState(false);
  const [currentPriceError, setCurrentPriceError] = useState("");
  const [quantity, setQuantity] = useState("100");
  const [entryPrice, setEntryPrice] = useState("150");
  const [currentPrice, setCurrentPrice] = useState("160");
  const [portfolioEquity, setPortfolioEquity] = useState("100000");
  const [drawdown, setDrawdown] = useState("6");
  const [riskEvaluation, setRiskEvaluation] = useState<RiskEvaluation | null>(null);

  const loadHealth = useCallback(async () => {
    try {
      setHealthError("");
      const response = await fetch(`${API_BASE_URL}/api/risk/health`);
      if (!response.ok) throw new Error(`Risk health API returned HTTP ${response.status}`);
      setHealth(await response.json());
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : "Unable to load Phase 12 health.");
    }
  }, []);

  const loadAlerts = useCallback(async () => {
    try {
      setAlertsLoading(true);
      setAlertsError("");
      const response = await fetch(`${API_BASE_URL}/api/risk/alerts`);
      if (!response.ok) throw new Error(`Alerts API returned HTTP ${response.status}`);
      const data = await response.json();
      setAlerts(Array.isArray(data) ? data : Array.isArray(data.alerts) ? data.alerts : []);
    } catch (error) {
      setAlertsError(error instanceof Error ? error.message : "Unable to load alerts.");
    } finally {
      setAlertsLoading(false);
    }
  }, []);

  const loadAlertSummary = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/risk/alerts/summary`);
      if (!response.ok) throw new Error();
      setAlertSummary(await response.json());
    } catch {
      setAlertSummary(null);
    }
  }, []);

  useEffect(() => {
    loadHealth();
    loadAlerts();
    loadAlertSummary();
  }, [loadHealth, loadAlerts, loadAlertSummary]);

  // Search the existing StockAI backend for the full NSE equity universe.
  // No hard-coded stock list is used here.
  useEffect(() => {
    const query = stockSearch.trim();

    if (!query || query.includes(" — ")) {
      setStockResults([]);
      setStockSearchLoading(false);
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        setStockSearchLoading(true);

        const response = await fetch(
          `${API_BASE_URL}/api/stocks/search?query=${encodeURIComponent(query)}`
        );

        if (!response.ok) {
          throw new Error(`Stock search API returned HTTP ${response.status}`);
        }

        const data = await response.json();
        const results = Array.isArray(data?.data) ? data.data : [];

        setStockResults(
          results
            .filter(
              (stock: {
                instrument_key?: string;
                symbol?: string;
              }) =>
                stock.instrument_key?.startsWith("NSE_EQ|") &&
                Boolean(stock.symbol)
            )
            .slice(0, 12)
        );
        setShowStockResults(true);
      } catch {
        setStockResults([]);
      } finally {
        setStockSearchLoading(false);
      }
    }, 350);

    return () => window.clearTimeout(timer);
  }, [stockSearch]);

  const selectStock = async (stock: {
    name: string;
    symbol: string;
    instrument_key: string;
    exchange?: string;
  }) => {
    setInstrumentKey(stock.instrument_key);
    setStockSearch(
      stock.name ? `${stock.symbol} — ${stock.name}` : stock.symbol
    );
    setStockResults([]);
    setShowStockResults(false);

    // Automatically fetch the latest traded price for the selected stock.
    setCurrentPriceLoading(true);
    setCurrentPriceError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/stocks/quote?instrument_key=${encodeURIComponent(
          stock.instrument_key
        )}`
      );

      if (!response.ok) {
        throw new Error(`Quote API returned HTTP ${response.status}`);
      }

      const data = await response.json();

      if (
        data?.status === "error" ||
        typeof data?.value !== "number"
      ) {
        throw new Error(data?.message || "Current price is unavailable.");
      }

      setCurrentPrice(String(data.value));
    } catch (error) {
      setCurrentPriceError(
        error instanceof Error
          ? error.message
          : "Unable to load current stock price."
      );
    } finally {
      setCurrentPriceLoading(false);
    }
  };

  const evaluatePosition = async () => {
    try {
      setPositionLoading(true);
      setPositionError("");
      const response = await fetch(`${API_BASE_URL}/api/risk/position`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          instrument_key: instrumentKey,
          quantity: Number(quantity),
          entry_price: Number(entryPrice),
          current_price: Number(currentPrice),
          portfolio_equity: Number(portfolioEquity),
        }),
      });
      if (!response.ok) throw new Error(`Position API returned HTTP ${response.status}`);
      const data: PositionResponse = await response.json();
      setPosition(data);
      setRiskEvaluation(data.evaluation);
    } catch (error) {
      setPositionError(error instanceof Error ? error.message : "Unable to evaluate position.");
    } finally {
      setPositionLoading(false);
    }
  };

  const evaluateDrawdown = async () => {
    try {
      setPositionError("");
      const response = await fetch(`${API_BASE_URL}/api/risk/evaluate/drawdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ drawdown_percent: Number(drawdown) }),
      });
      if (!response.ok) throw new Error(`Drawdown API returned HTTP ${response.status}`);
      setRiskEvaluation(await response.json());
    } catch (error) {
      setPositionError(error instanceof Error ? error.message : "Unable to evaluate drawdown.");
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/risk/alerts/${encodeURIComponent(alertId)}/acknowledge`,
        { method: "POST", headers: { Accept: "application/json" } }
      );
      if (!response.ok) throw new Error(`Acknowledge API returned HTTP ${response.status}`);
      await loadAlerts();
      await loadAlertSummary();
    } catch (error) {
      setAlertsError(error instanceof Error ? error.message : "Unable to acknowledge alert.");
    }
  };

  const layerCount = health ? Object.keys(health.layers || {}).length : 7;
  const activeCount =
    alertSummary && typeof alertSummary.active_count === "number"
      ? alertSummary.active_count
      : alerts.filter((a) => !a.acknowledged).length;

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <main className="mx-auto max-w-7xl px-6 py-8">
        <section className="mb-8">
          <p className="text-sm text-slate-400">Phase 12</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">Risk Management</h1>
          <p className="mt-2 max-w-3xl text-slate-400">
            Monitor position risk, configured limits, drawdown, and risk alerts.
          </p>
        </section>

        <section className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Module" value={health?.module || "risk_management"} description={healthError || "Phase 12 backend"} />
          <MetricCard label="Risk Layers" value={`${layerCount}/7`} description={health ? "Backend layers ready" : "Checking backend"} />
          <MetricCard label="Active Alerts" value={String(activeCount)} description="Unacknowledged alerts" />
          <MetricCard label="System Status" value={health?.status === "success" ? "READY" : "CHECKING"} description="Phase 12 health" />
        </section>

        <section className="mb-6 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <p className="text-sm text-slate-400">Position Risk</p>
              <h2 className="mt-1 text-xl font-semibold">Evaluate a position</h2>
            </div>
            {position?.position && <StatusBadge value={position.position.risk_status} />}
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            <div className="relative text-sm text-slate-400">
              Stock Search
              <input
                type="text"
                value={stockSearch}
                placeholder="Search any NSE stock..."
                onFocus={() => {
                  if (stockSearch.trim() && !stockSearch.includes(" — ")) {
                    setShowStockResults(true);
                  }
                }}
                onChange={(e) => {
                  setStockSearch(e.target.value);
                  setShowStockResults(true);
                }}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              />

              {showStockResults && stockSearch.trim() && (
                <div className="absolute left-0 right-0 top-full z-30 mt-1 max-h-72 overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 shadow-2xl">
                  {stockSearchLoading ? (
                    <div className="px-3 py-4 text-xs text-slate-400">
                      Searching stocks...
                    </div>
                  ) : stockResults.length > 0 ? (
                    stockResults.map((stock) => (
                      <button
                        key={stock.instrument_key}
                        type="button"
                        onClick={() => selectStock(stock)}
                        className="block w-full border-b border-slate-800 px-3 py-3 text-left last:border-0 hover:bg-slate-800"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="font-medium text-white">
                            {stock.symbol}
                          </span>
                          <span className="text-[10px] uppercase text-cyan-400">
                            {stock.exchange || "NSE"}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-slate-400">
                          {stock.name || "NSE Equity"}
                        </div>
                        <div className="mt-1 break-all text-[10px] text-slate-600">
                          {stock.instrument_key}
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="px-3 py-4 text-xs text-slate-500">
                      No NSE stock found. Try the company name or symbol.
                    </div>
                  )}
                </div>
              )}

              <p className="mt-1 break-all text-xs text-slate-600">
                Instrument Key: {instrumentKey}
              </p>
            </div>

            {[
              ["Quantity", quantity, setQuantity, "number"],
              ["Entry Price", entryPrice, setEntryPrice, "number"],
            ].map(([label, value, setter, type]) => (
              <label key={label as string} className="text-sm text-slate-400">
                {label as string}
                <input
                  type={type as string}
                  value={value as string}
                  onChange={(e) => (setter as (v: string) => void)(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
                />
              </label>
            ))}

            <label className="text-sm text-slate-400">
              Current Price
              <div className="relative">
                <input
                  type="number"
                  value={currentPrice}
                  onChange={(e) => setCurrentPrice(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-emerald-500/30 bg-slate-950 px-3 py-2 pr-20 text-sm text-white outline-none focus:border-cyan-400"
                />
                <span className="pointer-events-none absolute right-3 top-1/2 mt-1 -translate-y-1/2 text-[10px] font-medium uppercase text-emerald-400">
                  {currentPriceLoading ? "Loading..." : "Live"}
                </span>
              </div>
              {currentPriceError ? (
                <p className="mt-1 text-xs text-yellow-400">
                  {currentPriceError}
                </p>
              ) : (
                <p className="mt-1 text-xs text-slate-600">
                  Auto-filled from the selected stock quote.
                </p>
              )}
            </label>

            <label className="text-sm text-slate-400">
              Portfolio Equity
              <input
                type="number"
                value={portfolioEquity}
                onChange={(e) => setPortfolioEquity(e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={evaluatePosition}
            disabled={positionLoading}
            className="mt-5 rounded-lg bg-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
          >
            {positionLoading ? "Evaluating..." : "Evaluate Position"}
          </button>

          {positionError && (
            <p className="mt-4 rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-sm text-red-400">{positionError}</p>
          )}

          {position && (
            <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="Market Value" value={`₹${formatNumber(position.position.market_value)}`} />
              <MetricCard label="Exposure" value={`${formatNumber(position.position.exposure_percent)}%`} />
              <MetricCard label="Unrealized P&L" value={`₹${formatNumber(position.position.unrealized_pnl)}`} />
              <MetricCard label="P&L %" value={`${formatNumber(position.position.unrealized_pnl_percent)}%`} />
            </div>
          )}

          {position?.evaluation && (
            <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950/50 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm text-slate-400">Risk Evaluation</p>
                  <p className="mt-1 text-sm text-slate-300">{position.evaluation.message}</p>
                </div>
                <div className="flex gap-2">
                  <StatusBadge value={position.evaluation.status} />
                  <StatusBadge value={position.evaluation.action} />
                </div>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div><p className="text-xs text-slate-500">Current</p><p className="mt-1 font-medium">{formatNumber(position.evaluation.current_value)}%</p></div>
                <div><p className="text-xs text-slate-500">Limit</p><p className="mt-1 font-medium">{formatNumber(position.evaluation.limit_value)}%</p></div>
                <div><p className="text-xs text-slate-500">Utilization</p><p className="mt-1 font-medium">{formatNumber(position.evaluation.utilization_percent)}%</p></div>
              </div>
            </div>
          )}
        </section>

        <section className="mb-6 grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <p className="text-sm text-slate-400">Drawdown Monitor</p>
            <h2 className="mt-1 text-xl font-semibold">Check portfolio drawdown</h2>
            <div className="mt-5 flex gap-3">
              <input
                type="number"
                min="0"
                value={drawdown}
                onChange={(e) => setDrawdown(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              />
              <button type="button" onClick={evaluateDrawdown} className="shrink-0 rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700">
                Evaluate
              </button>
            </div>
            {riskEvaluation && (
              <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-slate-400">{riskEvaluation.risk_type}</span>
                  <StatusBadge value={riskEvaluation.status} />
                </div>
                <p className="mt-3 text-sm text-slate-300">{riskEvaluation.message}</p>
                <p className="mt-2 text-xs text-slate-500">
                  Current: {formatNumber(riskEvaluation.current_value)}% · Limit: {formatNumber(riskEvaluation.limit_value)}% · Utilization: {formatNumber(riskEvaluation.utilization_percent)}%
                </p>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
            <p className="text-sm text-slate-400">Configured Risk Limits</p>
            <h2 className="mt-1 text-xl font-semibold">Current controls</h2>
            <div className="mt-5 space-y-3 text-sm">
              {[
                ["Maximum position", "20%"],
                ["Maximum portfolio exposure", "80%"],
                ["Warning drawdown", "5%"],
                ["Maximum drawdown", "10%"],
                ["Maximum daily loss", "3%"],
                ["Maximum concentration", "25%"],
                ["Default stop loss", "5%"],
                ["Default take profit", "10%"],
              ].map(([label, value]) => (
                <div key={label} className="flex items-center justify-between border-b border-slate-800/70 pb-3 last:border-0 last:pb-0">
                  <span className="text-slate-400">{label}</span>
                  <span className="font-medium text-slate-200">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <p className="text-sm text-slate-400">Risk Alerts</p>
              <h2 className="mt-1 text-xl font-semibold">Recent risk events</h2>
            </div>
            <button
              type="button"
              onClick={() => { loadAlerts(); loadAlertSummary(); }}
              className="rounded-lg bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700"
            >
              Refresh
            </button>
          </div>

          {alertsError && <p className="mt-4 rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-sm text-red-400">{alertsError}</p>}

          {alertsLoading ? (
            <div className="mt-6 rounded-xl bg-slate-950/40 p-8 text-center text-sm text-slate-500">Loading risk alerts...</div>
          ) : alerts.length === 0 ? (
            <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950/40 p-8 text-center text-sm text-slate-500">No risk alerts available.</div>
          ) : (
            <div className="mt-5 space-y-3">
              {alerts.map((alert) => (
                <div key={alert.alert_id} className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <StatusBadge value={alert.status} />
                        <span className="text-xs text-slate-500">{alert.risk_type}</span>
                        <span className="text-xs text-slate-600">{alert.level}</span>
                      </div>
                      <p className="mt-3 text-sm text-slate-300">{alert.message}</p>
                      <p className="mt-2 text-xs text-slate-600">
                        {alert.instrument_key || "Portfolio"} · {new Date(alert.created_at).toLocaleString("en-IN")}
                      </p>
                    </div>
                    {!alert.acknowledged && (
                      <button
                        type="button"
                        onClick={() => acknowledgeAlert(alert.alert_id)}
                        className="shrink-0 rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:border-slate-500 hover:text-white"
                      >
                        Acknowledge
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
