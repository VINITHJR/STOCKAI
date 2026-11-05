import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "../services/marketApi";

type Sector = {
  key: string;
  label: string;
  group: string;
};

type SectorQuote = {
  key: string;
  label: string;
  group: string;
  value: number | null;
  change: number | null;
  status: string;
  error?: string;
};

const SECTORS: Sector[] = [
  { key: "NIFTYIT", label: "NIFTY IT", group: "Technology" },
  { key: "NIFTYAUTO", label: "NIFTY Auto", group: "Auto" },
  { key: "NIFTYPHARMA", label: "NIFTY Pharma", group: "Pharma" },
  { key: "NIFTYFMCG", label: "NIFTY FMCG", group: "FMCG" },
  { key: "NIFTYMETAL", label: "NIFTY Metal", group: "Metals" },
  { key: "NIFTYPSUBANK", label: "NIFTY PSU Bank", group: "Banking" },
  { key: "NIFTYPRIVATEBANK", label: "NIFTY Private Bank", group: "Banking" },
  { key: "NIFTYFINANCIAL", label: "NIFTY Financial", group: "Financial Services" },
];

function formatNumber(value: number | null, digits = 2) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatChange(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function changeClass(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "text-slate-400";
  if (value > 0) return "text-emerald-400";
  if (value < 0) return "text-red-400";
  return "text-slate-400";
}

function strengthLabel(change: number | null, average: number) {
  if (change == null || !Number.isFinite(change)) return "Unavailable";
  const relative = change - average;
  if (relative >= 1.0) return "Strong";
  if (relative >= 0.25) return "Leading";
  if (relative <= -1.0) return "Weak";
  if (relative <= -0.25) return "Lagging";
  return "Neutral";
}

function strengthClass(label: string) {
  if (label === "Strong" || label === "Leading") return "text-emerald-400";
  if (label === "Weak" || label === "Lagging") return "text-red-400";
  return "text-slate-400";
}

export default function SectorStrength() {
  const [rows, setRows] = useState<SectorQuote[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async (refresh = false) => {
    try {
      if (refresh) setRefreshing(true); else setLoading(true);
      setError("");

      const results = await Promise.all(
        SECTORS.map(async (sector): Promise<SectorQuote> => {
          try {
            const response = await fetch(
              `${API_BASE_URL}/api/indices/${encodeURIComponent(sector.key)}`,
              { headers: { Accept: "application/json" } }
            );
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            const data = payload?.data ?? {};
            return {
              ...sector,
              value: Number.isFinite(Number(data.value)) ? Number(data.value) : null,
              change: Number.isFinite(Number(data.change)) ? Number(data.change) : null,
              status: payload?.status || "success",
            };
          } catch (err) {
            return {
              ...sector,
              value: null,
              change: null,
              status: "error",
              error: err instanceof Error ? err.message : "Unable to load sector data",
            };
          }
        })
      );

      setRows(results);
      if (results.every((row) => row.change == null)) {
        setError("Sector index data could not be loaded. Check the Upstox connection and backend index mappings.");
      }
    } catch (err) {
      console.error("Sector strength error:", err);
      setError(err instanceof Error ? err.message : "Unable to load sector strength.");
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

  const average = useMemo(() => {
    const valid = rows.map((row) => row.change).filter((value): value is number => value != null && Number.isFinite(value));
    return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : 0;
  }, [rows]);

  const ranked = useMemo(
    () => [...rows].sort((a, b) => (b.change ?? -Infinity) - (a.change ?? -Infinity)),
    [rows]
  );

  const leaders = useMemo(() => ranked.filter((row) => (row.change ?? -Infinity) > average), [ranked, average]);
  const laggards = useMemo(() => ranked.filter((row) => (row.change ?? Infinity) < average), [ranked, average]);

  return (
    <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm text-cyan-400">Phase 5 · Layer 5</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Sector Strength &amp; Rotation</h2>
          <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">
            Compares major NSE sector indices by current percentage change. Relative strength is measured against the average change of the displayed sectors; this layer describes current leadership and lagging sectors without changing Layers 2–4 signals.
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
          Loading sector strength data...
        </div>
      ) : error ? (
        <div className="mt-5 rounded-xl border border-red-900/60 bg-red-950/30 p-5 text-sm text-red-300">{error}</div>
      ) : (
        <>
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <p className="text-xs text-slate-500">Sector average</p>
              <p className={`mt-1 text-2xl font-bold ${changeClass(average)}`}>{formatChange(average)}</p>
            </div>
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
              <p className="text-xs text-slate-500">Leading sectors</p>
              <p className="mt-1 text-2xl font-bold text-emerald-400">{leaders.length}</p>
            </div>
            <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
              <p className="text-xs text-slate-500">Lagging sectors</p>
              <p className="mt-1 text-2xl font-bold text-red-400">{laggards.length}</p>
            </div>
          </div>

          <div className="mt-5 overflow-x-auto rounded-xl border border-slate-800">
            <table className="min-w-[760px] w-full text-left text-sm">
              <thead className="border-b border-slate-800 bg-slate-950/80 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-3">Rank</th>
                  <th className="px-4 py-3">Sector</th>
                  <th className="px-4 py-3">Group</th>
                  <th className="px-4 py-3">Value</th>
                  <th className="px-4 py-3">Change</th>
                  <th className="px-4 py-3">Relative Strength</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {ranked.map((row, index) => {
                  const strength = strengthLabel(row.change, average);
                  return (
                    <tr key={row.key} className="hover:bg-slate-800/40">
                      <td className="px-4 py-3 font-semibold text-cyan-300">#{index + 1}</td>
                      <td className="px-4 py-3 font-semibold text-slate-100">{row.label}</td>
                      <td className="px-4 py-3 text-slate-400">{row.group}</td>
                      <td className="px-4 py-3 text-slate-300">{formatNumber(row.value)}</td>
                      <td className={`px-4 py-3 font-semibold ${changeClass(row.change)}`}>{formatChange(row.change)}</td>
                      <td className={`px-4 py-3 font-semibold ${strengthClass(strength)}`}>{strength}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/40 p-4 text-xs leading-5 text-slate-500">
            <strong className="text-slate-300">Rotation context:</strong> a sector is marked Leading when its daily change is at least 0.25 percentage points above the displayed-sector average, and Lagging when it is at least 0.25 points below. This is a current relative-strength view, not a forecast or a claim that capital has definitively rotated between sectors.
          </div>
        </>
      )}
    </section>
  );
}
