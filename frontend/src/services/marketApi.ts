export interface MarketIndex {
  symbol: string;
  value: number | null;
  change: number | null;
  net_change: number | null;
  previous_close: number | null;
  open: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
  timestamp?: string | null;
  status: string;
}

export interface MarketOverview {
  nifty: MarketIndex;
  sensex: MarketIndex;
  bank_nifty: MarketIndex;
  india_vix: MarketIndex;
}

export interface MarketBreadth {
  advances: number;
  declines: number;
  unchanged: number;
  total: number;
  advance_ratio: number;
  decline_ratio: number;
  unchanged_ratio: number;
}

export interface MostActiveStock {
  name: string;
  symbol: string;
  exchange: string;
  instrument_key: string;
  value: number;
  change: number;
  net_change: number;
  previous_close: number;
  volume: number;
  average_price: number | null;
  traded_value: number | null;
  timestamp?: string | null;
}

export interface MarketActivity {
  status: string;
  source: string;
  total_instruments: number;
  quoted_instruments: number;
  breadth: MarketBreadth;
  most_active: MostActiveStock[];
  message?: string;
}

export interface MarketStatus {
  status: string;
  exchange: string;
  market_status: string | null;
  last_updated: number | null;
  cas_eligible_status?: {
    status?: string;
    last_updated?: number;
  } | null;
}

export interface SearchResult {
  name: string;
  symbol: string;
  exchange: string;
  instrument_key: string;
  isin?: string;
  short_name?: string;
  instrument_type?: "EQUITY" | "INDEX";
  segment?: string;
}

export interface HistoryCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
  open_interest: number;
}

export interface LiveTick {
  type: "tick";
  instrument_key: string;
  ltp: number;
  cp: number | null;
  net_change: number | null;
  change: number | null;
  ltt?: string | null;
  ltq?: string | null;
  current_ts?: string | null;
}

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  if (data?.status === "error") {
    throw new Error(data.message || "API returned an error");
  }
  return data as T;
}

export async function getMarketOverview(): Promise<MarketOverview> {
  return fetchJson<MarketOverview>(`${API_BASE_URL}/api/market/overview`);
}

export async function getMarketActivity(limit: number = 10): Promise<MarketActivity> {
  return fetchJson<MarketActivity>(
    `${API_BASE_URL}/api/market/activity?limit=${encodeURIComponent(limit)}`
  );
}

export async function getMarketStatus(exchange: string = "NSE"): Promise<MarketStatus> {
  return fetchJson<MarketStatus>(
    `${API_BASE_URL}/api/market/status?exchange=${encodeURIComponent(exchange)}`
  );
}

export async function searchMarket(query: string): Promise<SearchResult[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  const data = await fetchJson<{ data: SearchResult[] }>(
    `${API_BASE_URL}/api/search?query=${encodeURIComponent(trimmed)}`
  );
  return Array.isArray(data.data) ? data.data : [];
}

export async function getIntraday(
  instrumentKey: string,
  isIndex: boolean,
  unit: string = "minutes",
  interval: string = "1"
): Promise<HistoryCandle[]> {
  const endpoint = isIndex ? "/api/indices/intraday" : "/api/stocks/intraday";
  const data = await fetchJson<{ candles: HistoryCandle[] }>(
    `${API_BASE_URL}${endpoint}` +
      `?instrument_key=${encodeURIComponent(instrumentKey)}` +
      `&unit=${encodeURIComponent(unit)}` +
      `&interval=${encodeURIComponent(interval)}`
  );
  return Array.isArray(data.candles) ? data.candles : [];
}

export async function getHistory(
  instrumentKey: string,
  isIndex: boolean,
  unit: string,
  interval: string,
  fromDate: string,
  toDate: string
): Promise<HistoryCandle[]> {
  const endpoint = isIndex ? "/api/indices/history" : "/api/stocks/history";
  const data = await fetchJson<{ candles: HistoryCandle[] }>(
    `${API_BASE_URL}${endpoint}` +
      `?instrument_key=${encodeURIComponent(instrumentKey)}` +
      `&unit=${encodeURIComponent(unit)}` +
      `&interval=${encodeURIComponent(interval)}` +
      `&from_date=${encodeURIComponent(fromDate)}` +
      `&to_date=${encodeURIComponent(toDate)}`
  );
  return Array.isArray(data.candles) ? data.candles : [];
}

export function marketWebSocketUrl(instrumentKey: string): string {
  const wsBase = API_BASE_URL.replace(/^http/, "ws");
  return `${wsBase}/ws/market?instrument_key=${encodeURIComponent(instrumentKey)}`;
}


export interface MarketScannerStock {
  rank: number;
  symbol: string;
  name: string;
  exchange: string;
  instrument_key: string;
  value: number;
  change: number;
  net_change: number;
  previous_close: number;
  open: number | null;
  high: number | null;
  low: number | null;
  volume: number;
  timestamp?: string | null;
  status: string;
}

export interface MarketScannerResponse {
  status: string;
  source: string;
  sort_by: "volume" | "change" | "value";
  total_universe: number;
  quoted_instruments: number;
  returned: number;
  stocks: MarketScannerStock[];
  message?: string;
}

export async function getMarketScanner(
  limit: number = 50,
  sortBy: "volume" | "change" | "value" = "volume"
): Promise<MarketScannerResponse> {
  return fetchJson<MarketScannerResponse>(
    `${API_BASE_URL}/api/market/scanner` +
      `?limit=${encodeURIComponent(limit)}` +
      `&sort_by=${encodeURIComponent(sortBy)}`
  );
}
