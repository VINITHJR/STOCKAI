import { API_BASE_URL } from "./marketApi";
export type Row = Record<string, string | number | boolean | null>;
export interface Overview {
  status: string; phase: number; part: string; run_id: string; title: string;
  research_only: boolean; source_runs: Record<string, string>; portfolios: string[];
  slippage_scenarios_bps_per_side: number[]; holdout: Record<string, unknown>;
  integrity: Record<string, unknown>; limitations: string[]; portfolio_risk_metrics: Row[];
}
export interface Report { status: string; report_key: string; rows: Row[]; }
async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}/api/phase15${path}`, {
    method: "GET", headers: { Accept: "application/json" }, cache: "no-store",
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); if (typeof body?.detail === "string") detail = body.detail; } catch { /* keep status */ }
    throw new Error(`Phase 15 API: ${detail}`);
  }
  return response.json() as Promise<T>;
}
export const getOverview = () => get<Overview>("/overview");
export const getReport = (key: string) => get<Report>(`/reports/${encodeURIComponent(key)}`);
