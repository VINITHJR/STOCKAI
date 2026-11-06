from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.backtest.phase10_layer2_routes import router as phase10_layer2_router
from backend.app.backtest.phase10_layer1_routes import router as phase10_layer1_router
from backend.app.backtest.phase10_layer3_routes import router as phase10_layer3_router
from backend.app.backtest.phase10_layer4_routes import router as phase10_layer4_router
from backend.app.backtest.phase10_layer5_routes import router as phase10_layer5_router
from .backtest.ai_layer6_api import router as ai_layer6_router
from backend.app.paper_trading.api import router as phase11_paper_trading_router
from backend.app.risk_management.routes import router as phase12_risk_management_router
from backend.app.portfolio_analysis.api import (api_router as portfolio_analysis_router,)
from backend.app.advanced_ai.api import api_router as advanced_ai_router
from backend.app.ml.phase15.report_api import router as phase15_report_router
from backend.app.services.fundamentals_service import get_fundamentals
from backend.app.services.news_event_service import get_news_events
from backend.app.backtest.api import router as backtest_router
from backend.app.ai.layer1 import router as phase9_layer1_router
from backend.app.ai.layer2 import router as phase9_layer2_router
from backend.app.ai.layer3 import router as phase9_layer3_router
from backend.app.ai.layer4 import router as phase9_layer4_router
from backend.app.ai.layer5 import router as phase9_layer5_router
from backend.app.services.upstox_service import (
    get_market_quotes,
    get_market_quotes_batched,
    get_market_movers,
    get_market_activity,
    get_market_scanner,
    search_stocks,
    find_quote,
    get_historical_candles,
    get_intraday_candles,
    get_index_quote,
    search_index_instrument,
    get_market_status,
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="StockAI API",
    description="AI-powered stock market analysis platform",
    version="1.2.0",
)


# =========================================================
# STARTUP: AUTO-TRAIN MODELS IF MISSING
# =========================================================

@app.on_event("startup")
async def startup_event():
    """Run startup tasks including ML model initialization."""
    from backend.app.startup import startup_checks
    startup_checks()


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://stockai-five-opal.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(backtest_router)
app.include_router(phase9_layer1_router)
app.include_router(phase9_layer2_router)
app.include_router(phase9_layer3_router)
app.include_router(phase9_layer4_router)
app.include_router(phase9_layer5_router)
app.include_router(phase10_layer1_router)
app.include_router(phase10_layer2_router)
app.include_router(phase10_layer3_router)
app.include_router(phase10_layer4_router)
app.include_router(phase10_layer5_router)
app.include_router(ai_layer6_router)
app.include_router(phase11_paper_trading_router)
app.include_router(phase12_risk_management_router)
app.include_router(portfolio_analysis_router)
app.include_router(advanced_ai_router)
app.include_router(phase15_report_router)
# =========================================================
# HELPERS
# =========================================================

def _change_percent(
    last_price,
    net_change,
):
    if last_price is None:
        return None

    try:
        last_price = float(last_price)
        net_change = float(net_change or 0)
    except (TypeError, ValueError):
        return None

    previous_close = last_price - net_change

    if previous_close == 0:
        return 0

    return round(
        (net_change / previous_close) * 100,
        2,
    )


def _format_quote(
    symbol: str,
    quote: dict,
):
    """
    Convert an Upstox full-market quote into the format
    used by the frontend.
    """

    if not quote:
        return {
            "symbol": symbol,
            "value": None,
            "change": None,
            "net_change": None,
            "previous_close": None,
            "open": None,
            "high": None,
            "low": None,
            "volume": None,
            "timestamp": None,
            "status": "unavailable",
        }

    last_price = quote.get("last_price")

    net_change = quote.get(
        "net_change",
        0,
    ) or 0

    ohlc = quote.get("ohlc") or {}

    percentage_change = quote.get(
        "percentage_change"
    )


    if percentage_change is None:
        percentage_change = _change_percent(
            last_price,
            net_change,
        )

    # Derive the previous close from the last price and net change
    # when both values are valid. Some quote payloads return an OHLC
    # close equal to the latest value, so prefer the implied close.
    previous_close = None
    raw_net_change = quote.get("net_change")

    if last_price is not None and raw_net_change is not None:
        try:
            last_price_num = float(last_price)
            net_change_num = float(raw_net_change)
            implied_previous_close = last_price_num - net_change_num

            if math.isfinite(implied_previous_close) and implied_previous_close > 0:
                previous_close = implied_previous_close
        except (TypeError, ValueError):
            previous_close = None

    # Use OHLC close as a fallback only when the implied previous close
    # could not be calculated safely.
    if previous_close is None:
        try:
            ohlc_close = float(ohlc.get("close"))
            if math.isfinite(ohlc_close) and ohlc_close > 0:
                previous_close = ohlc_close
        except (TypeError, ValueError):
            previous_close = None

    quote_symbol = quote.get("symbol")
    if (
        not quote_symbol
        or str(quote_symbol).strip().upper()
        in {"NA", "N/A", "NONE", "NULL"}
    ):
        quote_symbol = symbol

    return {
        "symbol": quote_symbol,
        "value": last_price,
        "change": (
            round(
                float(percentage_change),
                2,
            )
            if percentage_change is not None
            else None
        ),
        "net_change": (
            round(
                float(net_change),
                2,
            )
            if net_change is not None
            else None
        ),
        "previous_close": (
            round(
                float(previous_close),
                2,
            )
            if previous_close is not None
            else None
        ),
        "open": ohlc.get("open"),
        "high": ohlc.get("high"),
        "low": ohlc.get("low"),
        "volume": quote.get("volume"),
        "timestamp": quote.get("timestamp"),
        "status": (
            "success"
            if last_price is not None
            else "unavailable"
        ),
    }


def _normalise_search_item(
    item: dict,
    instrument_type: str,
):
    return {
        "name": (
            item.get("name")
            or item.get("short_name")
            or ""
        ),
        "symbol": (
            item.get("trading_symbol")
            or item.get("tradingsymbol")
            or item.get("symbol")
            or ""
        ),
        "exchange": (
            item.get("exchange")
            or (
                "NSE"
                if instrument_type == "INDEX"
                else "NSE"
            )
        ),
        "instrument_key": (
            item.get("instrument_key")
            or ""
        ),
        "isin": (
            item.get("isin")
            or ""
        ),
        "short_name": (
            item.get("short_name")
            or ""
        ),
        "instrument_type": instrument_type,
        "segment": (
            item.get("segment")
            or ""
        ),
    }


def _format_candles(
    result: dict,
    instrument_key: str,
    unit: str,
    interval: str,
):
    candles = (
        (result or {})
        .get("data", {})
        .get("candles", [])
        or []
    )

    formatted = []

    for candle in candles:
        if (
            not isinstance(
                candle,
                (list, tuple),
            )
            or len(candle) < 5
        ):
            continue

        formatted.append({
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": (
                candle[5]
                if len(candle) > 5
                else None
            ),
            "open_interest": (
                candle[6]
                if len(candle) > 6
                else 0
            ),
        })

    return {
        "status": "success",
        "instrument_key": instrument_key,
        "unit": unit,
        "interval": interval,
        "candles": formatted,
    }


# =========================================================
# ROOT
# =========================================================



@app.get("/api/fundamentals/{symbol}")
def fundamentals(symbol: str):
    """Return a descriptive fundamental snapshot for an NSE equity symbol."""
    try:
        return get_fundamentals(symbol)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "symbol": symbol.upper()}

@app.get("/api/news-events/{symbol}")
def news_events(symbol: str, days: int = 7, limit: int = 20):
    """Return recent NSE corporate announcements and event-attention context."""
    try:
        return get_news_events(symbol, days=days, limit=limit)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "symbol": symbol.upper()}
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "symbol": symbol.upper(),
            "events": [],
        }

@app.get("/")
def root():
    return {
        "message": "StockAI API is running",
        "version": "1.2.0",
        "status": "success",
    }


# =========================================================
# MARKET OVERVIEW
# =========================================================

@app.get("/api/market/nifty")
def get_nifty():
    instrument_key = (
        "NSE_INDEX|Nifty 50"
    )

    data = get_market_quotes(
        [instrument_key]
    )

    quote = find_quote(
        data,
        instrument_key,
    )

    if quote is None:
        return {
            "status": "error",
            "message": (
                "NIFTY data was not returned "
                "by Upstox"
            ),
            "available_keys": list(
                data.get(
                    "data",
                    {},
                ).keys()
            ),
        }

    return _format_quote(
        "NIFTY 50",
        quote,
    )


@app.get("/api/market/overview")
def get_market_overview():

    instruments = {
        "nifty": {
            "key": "NSE_INDEX|Nifty 50",
            "name": "NIFTY 50",
        },
        "sensex": {
            "key": "BSE_INDEX|SENSEX",
            "name": "SENSEX",
        },
        "bank_nifty": {
            "key": "NSE_INDEX|Nifty Bank",
            "name": "BANK NIFTY",
        },
        "india_vix": {
            "key": "NSE_INDEX|India VIX",
            "name": "INDIA VIX",
        },
    }

    data = get_market_quotes(
        [
            item["key"]
            for item in instruments.values()
        ]
    )

    result = {}

    for field, instrument in instruments.items():

        quote = find_quote(
            data,
            instrument["key"],
        )

        result[field] = _format_quote(
            instrument["name"],
            quote,
        )

    return result


# =========================================================
# REAL TOP GAINERS / LOSERS
# =========================================================

@app.get("/api/market/movers")
def get_market_movers_endpoint(
    limit: int = 5,
):
    """
    Return today's real NSE Top Gainers and Top Losers.

    Example:
        /api/market/movers
        /api/market/movers?limit=10
    """

    try:

        if limit < 1:
            limit = 5

        if limit > 20:
            limit = 20

        result = get_market_movers(
            limit=limit
        )

        return result

    except Exception as error:

        print(
            "Market movers error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
            "gainers": [],
            "losers": [],
        }


# =========================================================
# MARKET BREADTH + MOST ACTIVE STOCKS
# =========================================================

@app.get("/api/market/activity")
def get_market_activity_endpoint(
    limit: int = 10,
):
    """
    Return NSE Market Breadth and Most Active Stocks.

    Example:
        /api/market/activity
        /api/market/activity?limit=20
    """

    try:
        if limit < 1:
            limit = 10

        if limit > 50:
            limit = 50

        result = get_market_activity(
            limit=limit
        )

        return result

    except Exception as error:

        print(
            "Market activity error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
            "breadth": {
                "advances": 0,
                "declines": 0,
                "unchanged": 0,
                "total": 0,
                "advance_ratio": 0,
                "decline_ratio": 0,
                "unchanged_ratio": 0,
            },
            "most_active": [],
        }


# =========================================================
# MARKET SCANNER — PHASE 5, LAYER 1
# =========================================================

@app.get("/api/market/scanner")
def get_market_scanner_endpoint(
    limit: int = 50,
    sort_by: str = "volume",
):
    """
    Return the first-layer StockAI Market Scanner table.

    This endpoint currently provides the NSE equity universe with
    live quote data. Technical analysis, ML signals and setup
    detection are intentionally left for later scanner layers.
    """

    try:
        if limit < 1:
            limit = 50

        if limit > 100:
            limit = 100

        if sort_by not in {"volume", "change", "value"}:
            sort_by = "volume"

        return get_market_scanner(
            limit=limit,
            sort_by=sort_by,
        )

    except Exception as error:
        print(
            "Market scanner error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
            "source": "error",
            "total_universe": 0,
            "quoted_instruments": 0,
            "returned": 0,
            "stocks": [],
        }


# =========================================================
# STOCK QUOTE
# =========================================================

@app.get("/api/stocks/quote")
def get_stock_quote(
    instrument_key: str,
):

    if not instrument_key:
        return {
            "status": "error",
            "message": (
                "instrument_key is required"
            ),
        }

    try:

        data = get_market_quotes(
            [instrument_key]
        )

        quote = find_quote(
            data,
            instrument_key,
        )

        if not quote:
            return {
                "status": "error",
                "message": (
                    "Stock quote not found"
                ),
            }

        return _format_quote(
            quote.get("symbol")
            or "STOCK",
            quote,
        )

    except Exception as error:

        print(
            "Stock quote error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# INDEX QUOTE
# =========================================================

@app.get("/api/indices/quote")
def get_index_quote_by_instrument(
    instrument_key: str,
):

    if not instrument_key:
        return {
            "status": "error",
            "message": (
                "instrument_key is required"
            ),
        }

    if not (
        instrument_key.startswith(
            "NSE_INDEX|"
        )
        or instrument_key.startswith(
            "BSE_INDEX|"
        )
    ):
        return {
            "status": "error",
            "message": (
                "instrument_key is not "
                "an index instrument"
            ),
        }

    try:

        data = get_market_quotes(
            [instrument_key]
        )

        quote = find_quote(
            data,
            instrument_key,
        )

        if not quote:
            return {
                "status": "error",
                "message": (
                    "Index quote not found"
                ),
            }

        return _format_quote(
            quote.get("symbol")
            or instrument_key.split(
                "|",
                1,
            )[1],
            quote,
        )

    except Exception as error:

        print(
            "Index quote error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# STOCK SEARCH
# =========================================================

@app.get("/api/stocks/search")
def search_stock(
    query: str,
):

    query = query.strip()

    if not query:
        return {
            "status": "success",
            "data": [],
        }

    try:

        response = search_stocks(
            query
        )

        raw_data = (
            response.get("data", [])
            if isinstance(
                response,
                dict,
            )
            else []
        )

        if not isinstance(
            raw_data,
            list,
        ):
            return {
                "status": "error",
                "message": (
                    "Unexpected Upstox "
                    "data format"
                ),
            }

        stocks = [
            _normalise_search_item(
                item,
                "EQUITY",
            )
            for item in raw_data
            if isinstance(
                item,
                dict,
            )
        ]

        return {
            "status": "success",
            "data": stocks,
        }

    except Exception as error:

        print(
            "Stock search error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# INDEX SEARCH
# =========================================================

@app.get("/api/indices/search/{query}")
def search_index(
    query: str,
):

    try:

        result = search_index_instrument(
            query
        )

        raw_data = (
            result.get("data", [])
            if isinstance(
                result,
                dict,
            )
            else []
        )

        return {
            "status": "success",
            "query": query,
            "data": [
                _normalise_search_item(
                    item,
                    "INDEX",
                )
                for item in raw_data
                if isinstance(
                    item,
                    dict,
                )
            ],
        }

    except Exception as error:

        print(
            "Index search error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# UNIFIED SEARCH
# =========================================================

@app.get("/api/search")
def unified_search(
    query: str,
):

    query = query.strip()

    if not query:
        return {
            "status": "success",
            "data": [],
        }

    combined = []
    seen = set()

    # -----------------------------
    # Stocks
    # -----------------------------

    try:

        stock_response = search_stocks(
            query
        )

        stock_data = (
            stock_response.get(
                "data",
                [],
            )
            if isinstance(
                stock_response,
                dict,
            )
            else []
        )

        for item in stock_data:

            if not isinstance(
                item,
                dict,
            ):
                continue

            normalised = (
                _normalise_search_item(
                    item,
                    "EQUITY",
                )
            )

            key = normalised[
                "instrument_key"
            ]

            if (
                key
                and key not in seen
            ):
                combined.append(
                    normalised
                )

                seen.add(key)

    except Exception as error:

        print(
            "Unified stock search error:",
            repr(error),
        )

    # -----------------------------
    # Indices
    # -----------------------------

    try:

        index_response = (
            search_index_instrument(
                query
            )
        )

        index_data = (
            index_response.get(
                "data",
                [],
            )
            if isinstance(
                index_response,
                dict,
            )
            else []
        )

        for item in index_data:

            if not isinstance(
                item,
                dict,
            ):
                continue

            normalised = (
                _normalise_search_item(
                    item,
                    "INDEX",
                )
            )

            key = normalised[
                "instrument_key"
            ]

            if (
                key
                and key not in seen
            ):
                combined.append(
                    normalised
                )

                seen.add(key)

    except Exception as error:

        print(
            "Unified index search error:",
            repr(error),
        )

    return {
        "status": "success",
        "query": query,
        "data": combined,
    }


# =========================================================
# INDEX CONSTITUENT VOLUME
# =========================================================
#
# Spot indices do not have traded volume of their own.  For the
# chart we therefore calculate an index-activity volume by summing
# the traded volume of the index constituents for each candle.
#
# The NIFTY 50 list below is aligned with the constituent list
# published by NSE Indices in August 2026.  The Sensex list is
# aligned with the May 2026 Sensex portfolio and should be refreshed
# when BSE changes the index basket.

NIFTY50_CONSTITUENTS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "TMPV", "TRENT", "ULTRACEMCO", "WIPRO",
]

SENSEX_CONSTITUENTS = [
    "HDFCBANK", "ICICIBANK", "RELIANCE", "BHARTIARTL", "LT", "INFY",
    "SBIN", "AXISBANK", "KOTAKBANK", "ITC", "M&M", "BAJFINANCE", "TCS",
    "SUNPHARMA", "HINDUNILVR", "NTPC", "ETERNAL", "MARUTI", "TATASTEEL",
    "TITAN", "BEL", "ULTRACEMCO", "ADANIPORTS", "POWERGRID", "HCLTECH",
    "ASIANPAINT", "BAJAJFINSV", "INDIGO", "TECHM", "TRENT",
]

_INDEX_VOLUME_CONFIG = {
    "NSE_INDEX|Nifty 50": {
        "name": "NIFTY 50",
        "exchange": "NSE",
        "symbols": NIFTY50_CONSTITUENTS,
    },
    "BSE_INDEX|SENSEX": {
        "name": "SENSEX",
        "exchange": "BSE",
        "symbols": SENSEX_CONSTITUENTS,
    },
}

_index_constituent_key_cache = {}
_index_volume_cache = {}
_index_volume_lock = Lock()


def _search_equity_key(symbol: str, exchange: str) -> str | None:
    """Resolve an equity instrument key from Upstox's stock search."""
    try:
        response = search_stocks(symbol)
        rows = response.get("data", []) if isinstance(response, dict) else []
        if not isinstance(rows, list):
            return None

        symbol_upper = symbol.upper()
        for item in rows:
            if not isinstance(item, dict):
                continue

            candidate_symbol = str(
                item.get("trading_symbol")
                or item.get("tradingsymbol")
                or item.get("symbol")
                or ""
            ).upper()
            isin = str(item.get("isin") or "").strip()
            instrument_key = str(item.get("instrument_key") or "").strip()

            if exchange == "NSE":
                if candidate_symbol == symbol_upper and instrument_key.startswith("NSE_EQ|"):
                    return instrument_key

            # Upstox search is NSE equity search in the current service.
            # BSE listed shares use the same ISIN, so construct BSE_EQ
            # from that ISIN when we need Sensex volume.
            if exchange == "BSE" and candidate_symbol == symbol_upper and isin:
                return f"BSE_EQ|{isin}"

        # A few symbols can be returned without an exact symbol match.
        # Use the first NSE result only when the query itself is exact.
        if exchange == "NSE" and rows:
            for item in rows:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("instrument_key") or "")
                if key.startswith("NSE_EQ|"):
                    return key
    except Exception as error:
        print(f"Index constituent lookup failed for {exchange}:{symbol}: {error}")

    return None


def _get_index_constituent_keys(index_instrument_key: str) -> list[str]:
    config = _INDEX_VOLUME_CONFIG.get(index_instrument_key)
    if not config:
        return []

    cache_key = index_instrument_key
    cached = _index_constituent_key_cache.get(cache_key)
    if cached:
        return cached

    print(f"Resolving {config['name']} constituent instrument keys...")
    keys = []

    # Search is network-bound, so resolve several constituents concurrently.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _search_equity_key,
                symbol,
                config["exchange"],
            ): symbol
            for symbol in config["symbols"]
        }

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                key = future.result()
                if key:
                    keys.append(key)
            except Exception as error:
                print(f"Constituent key error {symbol}: {error}")

    # Preserve deterministic basket order and remove duplicates.
    keys = list(dict.fromkeys(keys))
    _index_constituent_key_cache[cache_key] = keys

    print(
        f"{config['name']} volume basket resolved: "
        f"{len(keys)}/{len(config['symbols'])} constituents"
    )
    return keys


def _extract_raw_candles(result: dict) -> list:
    candles = (result or {}).get("data", {}).get("candles", []) or []
    return candles if isinstance(candles, list) else []


def _fetch_constituent_volume_candles(
    instrument_key: str,
    unit: str,
    interval: str,
    to_date: str,
    from_date: str | None,
):
    """Fetch one constituent's candles and return timestamp -> volume."""
    try:
        if unit in ("minutes", "hours") and from_date is None:
            result = get_intraday_candles(
                instrument_key=instrument_key,
                unit=unit,
                interval=interval,
            )
        else:
            result = get_historical_candles(
                instrument_key=instrument_key,
                unit=unit,
                interval=interval,
                to_date=to_date,
                from_date=from_date,
            )
    except Exception as error:
        print(f"Constituent intraday error {instrument_key}: {error}")
        return {}

    output = {}
    for candle in _extract_raw_candles(result):
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue
        try:
            timestamp = candle[0]
            if isinstance(timestamp, str):
                # Keep the ISO timestamp as a stable dictionary key.
                timestamp_key = timestamp
            else:
                timestamp_key = str(timestamp)
            volume = float(candle[5])
            if volume >= 0:
                output[timestamp_key] = volume
        except (TypeError, ValueError):
            continue

    return output


def _get_index_constituent_volume(
    index_instrument_key: str,
    unit: str,
    interval: str,
    to_date: str,
    from_date: str | None,
):
    """Aggregate constituent traded volume into index-volume candles."""
    config = _INDEX_VOLUME_CONFIG.get(index_instrument_key)
    if not config:
        return {
            "status": "error",
            "message": "Constituent volume is currently supported for NIFTY 50 and SENSEX only.",
            "candles": [],
        }

    cache_key = (index_instrument_key, unit, interval, to_date, from_date)
    now = time.time()
    cached = _index_volume_cache.get(cache_key)
    if cached and now - cached["timestamp"] < 20:
        return cached["data"]

    keys = _get_index_constituent_keys(index_instrument_key)
    if not keys:
        return {
            "status": "error",
            "message": f"Unable to resolve {config['name']} constituent instruments.",
            "candles": [],
        }

    aggregate = {}
    successful = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _fetch_constituent_volume_candles,
                key,
                unit,
                interval,
                to_date,
                from_date,
            ): key
            for key in keys
        }

        for future in as_completed(futures):
            try:
                volumes = future.result()
                if volumes:
                    successful += 1
                for timestamp_key, volume in volumes.items():
                    aggregate[timestamp_key] = aggregate.get(timestamp_key, 0) + volume
            except Exception as error:
                print("Index volume aggregation error:", error)

    # The same Upstox timestamp strings are retained as returned by the
    # constituent candle endpoint. The frontend already knows how to parse them.
    candles = [
        {
            "timestamp": timestamp_key,
            "volume": round(volume, 2),
        }
        for timestamp_key, volume in sorted(aggregate.items(), key=lambda item: item[0])
    ]

    data = {
        "status": "success",
        "instrument_key": index_instrument_key,
        "index": config["name"],
        "source": "aggregated_constituent_volume",
        "exchange": config["exchange"],
        "constituents": len(config["symbols"]),
        "matched_constituents": len(keys),
        "successful_constituents": successful,
        "unit": unit,
        "interval": interval,
        "candles": candles,
    }

    _index_volume_cache[cache_key] = {
        "timestamp": now,
        "data": data,
    }
    return data


@app.get("/api/indices/volume")
def get_index_constituent_volume(
    instrument_key: str,
    unit: str = "minutes",
    interval: str = "1",
    to_date: str = "",
    from_date: str = "",
):
    """Return real aggregated constituent volume for NIFTY 50/SENSEX."""
    if instrument_key not in _INDEX_VOLUME_CONFIG:
        return {
            "status": "error",
            "message": "Supported index volume instruments: NSE_INDEX|Nifty 50 and BSE_INDEX|SENSEX",
            "candles": [],
        }

    try:
        if not to_date:
            to_date = date.today().isoformat()
        return _get_index_constituent_volume(
            instrument_key,
            unit,
            interval,
            to_date,
            from_date or None,
        )
    except Exception as error:
        print("Index constituent volume error:", repr(error))
        return {
            "status": "error",
            "message": str(error),
            "candles": [],
        }


# =========================================================
# LIVE INTRADAY CANDLES
# =========================================================

@app.get("/api/stocks/intraday")
def get_stock_intraday(
    instrument_key: str,
    unit: str = "minutes",
    interval: str = "1",
):
    """Return today's live-session candles for an NSE equity."""

    if not instrument_key.startswith("NSE_EQ|"):
        return {
            "status": "error",
            "message": "A valid NSE_EQ instrument_key is required",
        }

    try:
        result = get_intraday_candles(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
        )
        return _format_candles(
            result,
            instrument_key,
            unit,
            interval,
        )
    except Exception as error:
        print("Intraday stock data error:", repr(error))
        return {
            "status": "error",
            "message": str(error),
            "candles": [],
        }


@app.get("/api/indices/intraday")
def get_index_intraday(
    instrument_key: str,
    unit: str = "minutes",
    interval: str = "1",
):
    """Return today's live-session candles for an NSE/BSE index."""

    if not instrument_key.startswith(("NSE_INDEX|", "BSE_INDEX|")):
        return {
            "status": "error",
            "message": "A valid NSE_INDEX or BSE_INDEX instrument_key is required",
        }

    try:
        result = get_intraday_candles(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
        )
        return _format_candles(
            result,
            instrument_key,
            unit,
            interval,
        )
    except Exception as error:
        print("Intraday index data error:", repr(error))
        return {
            "status": "error",
            "message": str(error),
            "candles": [],
        }


# =========================================================
# STOCK HISTORY
# =========================================================

@app.get("/api/stocks/history")
def get_stock_history(
    instrument_key: str,
    unit: str = "days",
    interval: str = "1",
    to_date: str = "",
    from_date: str = "",
):

    try:

        if not to_date:
            to_date = (
                date.today().isoformat()
            )

        result = get_historical_candles(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            to_date=to_date,
            from_date=(
                from_date
                or None
            ),
        )

        return _format_candles(
            result,
            instrument_key,
            unit,
            interval,
        )

    except Exception as error:

        print(
            "Historical stock data error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# INDEX HISTORY
# =========================================================

@app.get("/api/indices/history")
def get_index_history(
    instrument_key: str,
    unit: str = "days",
    interval: str = "1",
    to_date: str = "",
    from_date: str = "",
):

    if not instrument_key.startswith(
        (
            "NSE_INDEX|",
            "BSE_INDEX|",
        )
    ):
        return {
            "status": "error",
            "message": (
                "A valid NSE_INDEX or "
                "BSE_INDEX instrument_key "
                "is required"
            ),
        }

    try:

        if not to_date:
            to_date = (
                date.today().isoformat()
            )

        result = get_historical_candles(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            to_date=to_date,
            from_date=(
                from_date
                or None
            ),
        )

        return _format_candles(
            result,
            instrument_key,
            unit,
            interval,
        )

    except Exception as error:

        print(
            "Historical index data error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# INDEX DETAILS
# =========================================================

@app.get("/api/indices/{index_name}")
def get_index_details(
    index_name: str,
):

    try:

        result = get_index_quote(
            index_name
        )

        quote = result["quote"]

        return {
            "status": "success",
            "index": index_name.upper(),
            "instrument_key": (
                result["instrument_key"]
            ),
            "data": _format_quote(
                quote.get("symbol")
                or index_name.upper(),
                quote,
            ),
        }

    except ValueError as error:

        return {
            "status": "error",
            "message": str(error),
        }

    except Exception as error:

        print(
            "Index quote error:",
            repr(error),
        )

        return {
            "status": "error",
            "message": str(error),
        }

# =========================================================
# LIVE MARKET STATUS
# =========================================================

@app.get("/api/market/status")
def market_status(exchange: str = "NSE"):
    """Return the live exchange status reported by Upstox."""
    try:
        result = get_market_status(exchange)
        data = result.get("data") or {}
        return {
            "status": "success",
            "exchange": data.get("exchange", exchange.upper()),
            "market_status": data.get("status"),
            "last_updated": data.get("last_updated"),
            "cas_eligible_status": data.get("cas_eligible_status"),
        }
    except Exception as error:
        print("Market status error:", repr(error))
        return {
            "status": "error",
            "message": str(error),
        }


# =========================================================
# LIVE MARKET DATA WEBSOCKET
# =========================================================

from fastapi import WebSocket, WebSocketDisconnect
from backend.app.services.market_stream import market_stream_manager


@app.websocket("/ws/market")
async def live_market_websocket(websocket: WebSocket):
    """Bridge Upstox Market Data Feed V3 to the browser."""
    await websocket.accept()
    instrument_key = websocket.query_params.get("instrument_key", "").strip()

    if not instrument_key or "|" not in instrument_key:
        await websocket.send_json({
            "type": "stream_error",
            "message": "A valid instrument_key query parameter is required.",
        })
        await websocket.close(code=1008)
        return

    queue = None
    try:
        queue = await market_stream_manager.register(instrument_key)
        await websocket.send_json({
            "type": "subscription",
            "instrument_key": instrument_key,
            "status": "subscribed",
        })

        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception as error:
        print("Browser market websocket error:", repr(error))
        try:
            await websocket.send_json({"type": "stream_error", "message": str(error)})
        except Exception:
            pass
    finally:
        if queue is not None:
            market_stream_manager.unregister(instrument_key, queue)


# =========================================================
# ML PRICE PREDICTIONS
# =========================================================

from backend.app.services.prediction_service import (
    get_prediction,
    get_prediction_with_analysis,
    clear_prediction_cache,
    get_cache_stats
)

from backend.app.services.trend_service import (
    get_trend_prediction,
    get_trend_with_context,
    clear_trend_cache,
    get_trend_cache_stats
)


@app.get("/api/predictions/{instrument_key}")
def get_price_prediction(instrument_key: str):
    """
    Get LSTM price predictions for an instrument.

    Returns predictions for 1-day, 3-day, and 7-day horizons.

    Example:
        /api/predictions/NSE_EQ|INE002A01018
    """
    try:
        result = get_prediction(instrument_key)
        return result
    except Exception as error:
        print(f"Prediction endpoint error: {repr(error)}")
        return {
            "status": "error",
            "message": str(error)
        }


@app.get("/api/predictions/{instrument_key}/detailed")
def get_detailed_prediction(instrument_key: str):
    """
    Get price predictions with additional technical analysis context.

    Example:
        /api/predictions/NSE_EQ|INE002A01018/detailed
    """
    try:
        result = get_prediction_with_analysis(instrument_key)
        return result
    except Exception as error:
        print(f"Detailed prediction endpoint error: {repr(error)}")
        return {
            "status": "error",
            "message": str(error)
        }


@app.post("/api/predictions/cache/clear")
def clear_cache():
    """Clear prediction cache (admin endpoint)."""
    try:
        clear_prediction_cache()
        return {
            "status": "success",
            "message": "Prediction cache cleared"
        }
    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }


@app.get("/api/predictions/cache/stats")
def cache_statistics():
    """Get prediction cache statistics."""
    try:
        stats = get_cache_stats()
        return {
            "status": "success",
            "cache": stats
        }
    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }


# =========================================================
# TREND CLASSIFICATION ENDPOINTS
# =========================================================

@app.get("/api/trends/{instrument_key}")
def get_trend_classification(instrument_key: str):
    """
    Get Random Forest trend classification for an instrument.

    Returns trend direction (bullish/bearish/neutral) with confidence.

    Example:
        /api/trends/NSE_EQ|INE002A01018
    """
    try:
        result = get_trend_prediction(instrument_key)
        return result
    except Exception as error:
        print(f"Trend classification endpoint error: {repr(error)}")
        return {
            "status": "error",
            "message": str(error)
        }


@app.get("/api/trends/{instrument_key}/detailed")
def get_detailed_trend(instrument_key: str):
    """
    Get trend classification with additional technical context.

    Example:
        /api/trends/NSE_EQ|INE002A01018/detailed
    """
    try:
        result = get_trend_with_context(instrument_key)
        return result
    except Exception as error:
        print(f"Detailed trend endpoint error: {repr(error)}")
        return {
            "status": "error",
            "message": str(error)
        }


@app.get("/api/combined/{instrument_key}")
def get_combined_prediction(instrument_key: str):
    """
    Get both LSTM price predictions AND RF trend classification together.

    Combines both models for a complete market outlook.

    Example:
        /api/combined/NSE_EQ|INE002A01018
    """
    try:
        # Get both predictions
        price_pred = get_prediction(instrument_key)
        trend_pred = get_trend_prediction(instrument_key)

        # Check for errors
        if price_pred.get('status') == 'error':
            price_pred = None
        if trend_pred.get('status') == 'error':
            trend_pred = None

        return {
            "status": "success",
            "instrument_key": instrument_key,
            "price_prediction": price_pred,
            "trend_classification": trend_pred,
            "timestamp": trend_pred.get('timestamp') if trend_pred else None
        }
    except Exception as error:
        print(f"Combined prediction endpoint error: {repr(error)}")
        return {
            "status": "error",
            "message": str(error)
        }


@app.post("/api/trends/cache/clear")
def clear_trend_cache_endpoint():
    """Clear trend classification cache (admin endpoint)."""
    try:
        clear_trend_cache()
        return {
            "status": "success",
            "message": "Trend cache cleared"
        }
    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }


@app.get("/api/trends/cache/stats")
def trend_cache_statistics():
    """Get trend classification cache statistics."""
    try:
        stats = get_trend_cache_stats()
        return {
            "status": "success",
            "cache": stats
        }
    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }
