import gzip
import json
import os
import time
from typing import Optional
from datetime import datetime
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------

load_dotenv("backend/.env")

UPSTOX_BASE_URL = "https://api.upstox.com/v2"
UPSTOX_V3_BASE_URL = "https://api.upstox.com/v3"

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

# Upstox allows up to 500 instruments for the full market
# quote request. We deliberately stay at 500.
QUOTE_BATCH_SIZE = 500

# Public Upstox instrument master.
#
# Upstox currently documents the CSV instrument files as
# deprecated, so this is being isolated behind one loader.
# If Upstox replaces this file with a newer instrument-master
# endpoint, only this function needs to change.
INSTRUMENT_MASTER_URL = (
    "https://assets.upstox.com/"
    "market-quote/instruments/exchange/NSE.json.gz"
)

# Keep the instrument universe in memory for this many seconds.
# 6 hours is enough because the instrument universe does not
# need to be downloaded for every dashboard refresh.
INSTRUMENT_CACHE_TTL = 6 * 60 * 60


# ---------------------------------------------------------
# Internal cache
# ---------------------------------------------------------

_INSTRUMENT_CACHE: Optional[list[dict]] = None
_INSTRUMENT_CACHE_TIME: float = 0.0

# Cache the complete NSE quote snapshot briefly so Dashboard
# requests for movers + market activity do not trigger two
# separate 2,544-instrument Upstox downloads within seconds.
_MARKET_QUOTE_SNAPSHOT_CACHE: Optional[dict] = None
_MARKET_QUOTE_SNAPSHOT_TIME: float = 0.0
MARKET_QUOTE_SNAPSHOT_TTL = 45  # seconds

# =========================================================
# MARKET MOVERS CACHE
# =========================================================

MOVERS_CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "movers_cache.json",
)

MOVERS_CACHE_TTL = 24 * 60 * 60  # 24 hours
# ---------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------

def _require_access_token():
    if not ACCESS_TOKEN:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN is missing. "
            "Check backend/.env"
        )


def _auth_headers():
    _require_access_token()

    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }


# ---------------------------------------------------------
# Full Market Quotes
# ---------------------------------------------------------

def get_market_quotes(instrument_keys: list[str]) -> dict:
    """
    Get full market quotes for one or more instruments.

    Upstox supports up to 500 instrument keys per request.
    The caller should batch larger lists before calling this
    function.
    """

    if not instrument_keys:
        return {
            "status": "success",
            "data": {},
        }

    if len(instrument_keys) > QUOTE_BATCH_SIZE:
        raise ValueError(
            f"Maximum {QUOTE_BATCH_SIZE} instruments are allowed "
            "per market quote request."
        )

    url = f"{UPSTOX_BASE_URL}/market-quote/quotes"

    params = {
        "instrument_key": ",".join(instrument_keys)
    }

    response = requests.get(
        url,
        headers=_auth_headers(),
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected Upstox market quote response: {result}"
        )

    return result


def get_market_quotes_batched(
    instrument_keys: list[str],
    batch_size: int = QUOTE_BATCH_SIZE,
) -> dict:
    """
    Get market quotes for an arbitrary number of instruments.

    The request is split into batches because Upstox limits
    the number of instrument keys in a single full-market-quote
    request.
    """

    if not instrument_keys:
        return {
            "status": "success",
            "data": {},
        }

    all_quotes = {}

    for start in range(0, len(instrument_keys), batch_size):
        batch = instrument_keys[start:start + batch_size]

        print(
            f"Fetching market quotes "
            f"{start + 1}-{start + len(batch)} "
            f"of {len(instrument_keys)}"
        )

        result = get_market_quotes(batch)

        data = result.get("data", {})

        if isinstance(data, dict):
            all_quotes.update(data)

    return {
        "status": "success",
        "data": all_quotes,
    }


# ---------------------------------------------------------
# Find a quote safely
# ---------------------------------------------------------

def find_quote(data: dict, instrument_key: str):
    """
    Find an Upstox quote using pipe or colon instrument-key
    format, with instrument_token as a fallback.
    """

    if not isinstance(data, dict):
        return None

    quotes = data.get("data", {})

    if not isinstance(quotes, dict):
        return None

    # Direct pipe-key match
    if instrument_key in quotes:
        return quotes[instrument_key]

    # Upstox responses often use ":" instead of "|"
    colon_key = instrument_key.replace("|", ":")

    if colon_key in quotes:
        return quotes[colon_key]

    # Fallback: inspect instrument_token
    for quote in quotes.values():
        if not isinstance(quote, dict):
            continue

        if quote.get("instrument_token") == instrument_key:
            return quote

    return None


# ---------------------------------------------------------
# Stock search
# ---------------------------------------------------------

from functools import lru_cache
import requests


@lru_cache(maxsize=100)
def search_stocks(query: str):

    if not ACCESS_TOKEN:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing")

    query = query.strip()

    if not query:
        return {
            "status": "success",
            "data": [],
        }

    url = f"{UPSTOX_BASE_URL}/instruments/search"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

    params = {
        "query": query,
        "exchanges": "NSE",
        "segments": "EQ",
        "page_number": 1,
        "records": 20,
    }

    print("Searching Upstox:", query)

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=(5, 20),
        )

        print(
            "Upstox HTTP status:",
            response.status_code,
        )

        response.raise_for_status()

        result = response.json()

        if not isinstance(result, dict):
            raise RuntimeError(
                "Unexpected Upstox response format"
            )

        return result

    except requests.exceptions.Timeout as error:
        print(
            "Upstox stock search timeout:",
            repr(error),
        )

        raise RuntimeError(
            "Upstox stock search timed out. "
            "Please try again."
        ) from error

    except requests.exceptions.RequestException as error:
        print(
            "Upstox stock search request error:",
            repr(error),
        )

        raise RuntimeError(
            "Unable to reach Upstox stock search."
        ) from error


# ---------------------------------------------------------
# Historical candles
# ---------------------------------------------------------

def get_historical_candles(
    instrument_key: str,
    unit: str,
    interval: str,
    to_date: str,
    from_date: str | None = None,
):
    """
    Upstox V3 historical candle endpoint.
    """

    _require_access_token()

    url = (
        f"{UPSTOX_V3_BASE_URL}/historical-candle/"
        f"{instrument_key}/{unit}/{interval}/{to_date}"
    )

    if from_date:
        url += f"/{from_date}"

    response = requests.get(
        url,
        headers=_auth_headers(),
        timeout=15,
    )

    print(
        "Historical API status:",
        response.status_code,
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected historical response: {result}"
        )

    return result


# ---------------------------------------------------------
# Major indices
# ---------------------------------------------------------

INDEX_INSTRUMENTS = {
    "NIFTY50": "NSE_INDEX|Nifty 50",
    "SENSEX": "BSE_INDEX|SENSEX",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "INDIAVIX": "NSE_INDEX|India VIX",

    "NIFTYNEXT50": "NSE_INDEX|Nifty Next 50",
    "NIFTYIT": "NSE_INDEX|Nifty IT",
    "NIFTYAUTO": "NSE_INDEX|Nifty Auto",
    "NIFTYPHARMA": "NSE_INDEX|Nifty Pharma",
    "NIFTYFMCG": "NSE_INDEX|Nifty FMCG",
    "NIFTYMETAL": "NSE_INDEX|Nifty Metal",
    "NIFTYPSUBANK": "NSE_INDEX|Nifty PSU Bank",
    "NIFTYPRIVATEBANK": "NSE_INDEX|Nifty Private Bank",
    "NIFTYFINANCIAL": "NSE_INDEX|Nifty Financial",
}


def get_index_quote(index_name: str):
    """
    Get a quote for a supported index.
    """

    index_name = index_name.upper().strip()

    instrument_key = INDEX_INSTRUMENTS.get(index_name)

    if not instrument_key:
        raise ValueError(
            f"Unsupported index: {index_name}"
        )

    data = get_market_quotes([instrument_key])

    quote = find_quote(
        data,
        instrument_key,
    )

    if quote is None:
        raise RuntimeError(
            f"No quote returned for {index_name}"
        )

    return {
        "instrument_key": instrument_key,
        "quote": quote,
    }


# ---------------------------------------------------------
# Index search
# ---------------------------------------------------------

def search_index_instrument(query: str):
    """
    Search Upstox index instruments.
    """

    _require_access_token()

    query = query.strip()

    if not query:
        return {
            "status": "success",
            "data": [],
        }

    url = f"{UPSTOX_BASE_URL}/instruments/search"

    params = {
        "query": query,
        "segments": "INDEX",
        "page_number": 1,
        "records": 30,
    }

    response = requests.get(
        url,
        headers=_auth_headers(),
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected Upstox index-search response: {result}"
        )

    return result


# =========================================================
# NSE INSTRUMENT UNIVERSE
# =========================================================

def _download_instrument_master() -> list[dict]:
    """
    Download the current NSE instrument master from Upstox.

    Upstox's current instrument JSON uses:
        segment       = NSE_EQ
        instrument_type = EQ

    We keep only normal NSE equity instruments because these
    are the instruments needed for Top Gainers / Top Losers.
    """

    print("Downloading Upstox NSE instrument master...")

    response = requests.get(
        INSTRUMENT_MASTER_URL,
        timeout=60,
    )

    print(
        "Instrument master HTTP status:",
        response.status_code,
    )

    response.raise_for_status()

    raw_bytes = response.content

    # Upstox provides the JSON file as gzip compressed data.
    try:
        raw_bytes = gzip.decompress(raw_bytes)
    except OSError:
        # requests/proxy may already have decompressed it.
        pass

    try:
        data = json.loads(
            raw_bytes.decode(
                "utf-8-sig"
            )
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to parse Upstox NSE instrument JSON: "
            f"{error}"
        )

    if not isinstance(data, list):
        raise RuntimeError(
            "Unexpected Upstox instrument JSON format. "
            f"Expected list, got {type(data).__name__}"
        )

    instruments = []

    for row in data:

        if not isinstance(row, dict):
            continue

        segment = str(
            row.get("segment") or ""
        ).strip().upper()

        instrument_type = str(
            row.get("instrument_type") or ""
        ).strip().upper()

        exchange = str(
            row.get("exchange") or ""
        ).strip().upper()

        instrument_key = str(
            row.get("instrument_key") or ""
        ).strip()

        trading_symbol = str(
            row.get("trading_symbol") or ""
        ).strip()

        name = str(
            row.get("name") or ""
        ).strip()

        # -------------------------------------------------
        # Only NSE cash-market equities
        # -------------------------------------------------

        if segment != "NSE_EQ":
            continue

        if instrument_type != "EQ":
            continue

        if exchange != "NSE":
            continue

        if not instrument_key.startswith(
            "NSE_EQ|"
        ):
            continue

        if not trading_symbol:
            continue

        # -------------------------------------------------
        # Skip ETF-like products
        # -------------------------------------------------

        name_upper = name.upper()

        if "ETF" in name_upper:
            continue

        instruments.append({
            "instrument_key": instrument_key,
            "symbol": trading_symbol,
            "name": name,
            "exchange": exchange,
            "segment": segment,
            "instrument_type": instrument_type,
            "isin": row.get("isin") or "",
        })

    # -----------------------------------------------------
    # Remove duplicate instrument keys
    # -----------------------------------------------------

    unique = {}

    for instrument in instruments:
        unique[
            instrument["instrument_key"]
        ] = instrument

    instruments = list(
        unique.values()
    )

    # -----------------------------------------------------
    # Sort alphabetically
    # -----------------------------------------------------

    instruments.sort(
        key=lambda item: item[
            "symbol"
        ].upper()
    )

    print(
        "NSE equity universe loaded:",
        len(instruments),
        "instruments",
    )

    # Show a few instruments for debugging
    if instruments:
        print(
            "Sample NSE instruments:"
        )

        for instrument in instruments[:5]:
            print(
                " ",
                instrument["symbol"],
                "->",
                instrument["instrument_key"],
            )

    return instruments


def get_nse_equity_universe(
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return the cached NSE equity universe.

    The instrument list is refreshed after the cache TTL or
    when force_refresh=True.
    """

    global _INSTRUMENT_CACHE
    global _INSTRUMENT_CACHE_TIME

    now = time.time()

    cache_is_valid = (
        _INSTRUMENT_CACHE is not None
        and (now - _INSTRUMENT_CACHE_TIME)
        < INSTRUMENT_CACHE_TTL
    )

    if cache_is_valid and not force_refresh:
        return _INSTRUMENT_CACHE

    _INSTRUMENT_CACHE = _download_instrument_master()
    _INSTRUMENT_CACHE_TIME = now

    return _INSTRUMENT_CACHE


# =========================================================
# NSE MARKET QUOTE SNAPSHOT
# =========================================================

def _get_nse_quote_snapshot(
    force_refresh: bool = False,
) -> tuple[list[dict], dict]:
    """
    Return the current NSE equity universe and its full quote
    snapshot.

    A short in-memory cache prevents the Dashboard's movers and
    activity endpoints from fetching the same 2,500+ quotes twice
    within a few seconds.
    """

    global _MARKET_QUOTE_SNAPSHOT_CACHE
    global _MARKET_QUOTE_SNAPSHOT_TIME

    now = time.time()

    if (
        not force_refresh
        and _MARKET_QUOTE_SNAPSHOT_CACHE is not None
        and (now - _MARKET_QUOTE_SNAPSHOT_TIME)
        < MARKET_QUOTE_SNAPSHOT_TTL
    ):
        cached = _MARKET_QUOTE_SNAPSHOT_CACHE
        return cached["universe"], cached["quotes"]

    universe = get_nse_equity_universe()

    if not universe:
        raise RuntimeError(
            "NSE equity instrument universe is empty."
        )

    instrument_keys = [
        item["instrument_key"]
        for item in universe
    ]

    quote_data = get_market_quotes_batched(
        instrument_keys
    )

    quotes = quote_data.get("data", {})

    if not isinstance(quotes, dict):
        raise RuntimeError(
            "Invalid market quote response from Upstox."
        )

    _MARKET_QUOTE_SNAPSHOT_CACHE = {
        "universe": universe,
        "quotes": quote_data,
    }
    _MARKET_QUOTE_SNAPSHOT_TIME = now

    print(
        f"NSE market quote snapshot ready: "
        f"{len(quotes)} quotes"
    )

    return universe, quote_data


# =========================================================
# TOP GAINERS / LOSERS
# =========================================================

def _quote_change_percent(quote: dict):
    """
    Calculate the percentage change for a stock.

    Upstox provides net_change as the absolute change
    from yesterday's close to the current last traded price.

    Priority:
    1. net_change + previous close
    2. percentage_change if available
    3. last_price vs OHLC close as final fallback
    """

    if not isinstance(quote, dict):
        return None

    # ---------------------------------------------------------
    # Current market price
    # ---------------------------------------------------------

    last_price = quote.get("last_price")

    try:
        last_price = float(last_price)
    except (TypeError, ValueError):
        return None

    if last_price <= 0:
        return None

    # ---------------------------------------------------------
    # Upstox net_change
    # ---------------------------------------------------------

    net_change = quote.get("net_change")

    try:
        if net_change is not None:
            net_change = float(net_change)
        else:
            net_change = None
    except (TypeError, ValueError):
        net_change = None

    # ---------------------------------------------------------
    # Previous close
    #
    # If net_change exists:
    #
    # previous_close = last_price - net_change
    # ---------------------------------------------------------

    if net_change is not None:

        previous_close = last_price - net_change

        if previous_close > 0:

            return (
                net_change / previous_close
            ) * 100

    # ---------------------------------------------------------
    # Fallback: percentage_change
    # ---------------------------------------------------------

    percentage_change = quote.get("percentage_change")

    try:
        if percentage_change is not None:
            return float(percentage_change)
    except (TypeError, ValueError):
        pass

    # ---------------------------------------------------------
    # Final fallback: OHLC close
    # ---------------------------------------------------------

    ohlc = quote.get("ohlc") or {}

    previous_close = ohlc.get("close")

    try:
        previous_close = float(previous_close)
    except (TypeError, ValueError):
        return None

    if previous_close <= 0:
        return None

    return (
        (last_price - previous_close)
        / previous_close
    ) * 100


def _build_mover(
    instrument: dict,
    quote: dict,
):
    """
    Convert an Upstox quote into a valid dashboard mover object.

    Invalid quotes such as:
        last_price = 0
        previous_close = 0
        incomplete OHLC data

    are ignored.
    """

    if not isinstance(quote, dict):
        return None

    # ---------------------------------------------------------
    # Current market price
    # ---------------------------------------------------------

    last_price = quote.get("last_price")

    try:
        last_price = float(last_price)
    except (TypeError, ValueError):
        return None

    # Never treat zero/negative prices as real market data
    if last_price <= 0:
        return None

    # ---------------------------------------------------------
    # OHLC
    # ---------------------------------------------------------

    ohlc = quote.get("ohlc") or {}

    # ---------------------------------------------------------
    # Previous trading-session close
    # ---------------------------------------------------------

    net_change = quote.get("net_change")

    try:
        net_change = float(net_change)
    except (TypeError, ValueError):
        net_change = None

    if net_change is not None:

        previous_close = last_price - net_change

    else:

        try:
            previous_close = float(
                ohlc.get("close")
            )
        except (TypeError, ValueError):
            return None

    if previous_close <= 0:
        return None

    # ---------------------------------------------------------
    # Calculate percentage change
    # ---------------------------------------------------------

    change = _quote_change_percent(quote)

    if change is None:
        return None

    try:
        change = float(change)
    except (TypeError, ValueError):
        return None

    # ---------------------------------------------------------
    # Net change
    # ---------------------------------------------------------

    net_change = quote.get("net_change")

    try:
        if net_change is None:
            net_change = last_price - previous_close
        else:
            net_change = float(net_change)
    except (TypeError, ValueError):
        net_change = last_price - previous_close

    # ---------------------------------------------------------
    # Build mover
    # ---------------------------------------------------------

    return {
        "symbol": instrument["symbol"],
        "name": instrument["name"],
        "exchange": instrument["exchange"],
        "instrument_key": instrument["instrument_key"],

        "value": round(last_price, 2),

        "change": round(change, 2),

        "net_change": round(net_change, 2),

        "previous_close": round(previous_close, 2),

        "open": ohlc.get("open"),
        "high": ohlc.get("high"),
        "low": ohlc.get("low"),

        "volume": quote.get("volume"),

        "timestamp": quote.get("timestamp"),

        "status": "success",
    }

# =========================================================
# MARKET BREADTH + MOST ACTIVE STOCKS
# =========================================================

def _build_market_activity(
    universe: list[dict],
    quote_data: dict,
    limit: int = 10,
) -> dict:
    """
    Build Market Breadth and Most Active Stocks from the same
    full NSE quote snapshot used by the movers system.

    Breadth:
        advances  -> percentage change > 0
        declines  -> percentage change < 0
        unchanged -> percentage change == 0

    Most active:
        sorted by today's traded volume, descending.
    """

    quotes = quote_data.get("data", {})

    if not isinstance(quotes, dict):
        raise RuntimeError(
            "Invalid market quote data."
        )

    advances = 0
    declines = 0
    unchanged = 0
    active = []

    for instrument in universe:
        quote = find_quote(
            quote_data,
            instrument["instrument_key"],
        )

        if not isinstance(quote, dict):
            continue

        # Reuse the exact validation/change calculation used by
        # Top Gainers / Top Losers.
        mover = _build_mover(
            instrument,
            quote,
        )

        if mover is None:
            continue

        change = mover["change"]

        if change > 0:
            advances += 1
        elif change < 0:
            declines += 1
        else:
            unchanged += 1

        raw_volume = quote.get("volume")

        try:
            volume = float(raw_volume)
        except (TypeError, ValueError):
            volume = 0.0

        if volume <= 0:
            continue

        raw_average_price = quote.get(
            "average_price"
        )

        try:
            average_price = float(
                raw_average_price
            )
            if average_price <= 0:
                average_price = None
        except (TypeError, ValueError):
            average_price = None

        traded_value = None

        if average_price is not None:
            traded_value = volume * average_price

        active.append({
            "symbol": mover["symbol"],
            "name": mover["name"],
            "exchange": mover["exchange"],
            "instrument_key": mover["instrument_key"],
            "value": mover["value"],
            "change": mover["change"],
            "net_change": mover["net_change"],
            "previous_close": mover["previous_close"],
            "volume": int(volume),
            "average_price": (
                round(average_price, 2)
                if average_price is not None
                else None
            ),
            "traded_value": (
                round(traded_value, 2)
                if traded_value is not None
                else None
            ),
            "timestamp": mover["timestamp"],
        })

    total = advances + declines + unchanged

    if total > 0:
        advance_ratio = round(
            (advances / total) * 100,
            2,
        )
        decline_ratio = round(
            (declines / total) * 100,
            2,
        )
        unchanged_ratio = round(
            (unchanged / total) * 100,
            2,
        )
    else:
        advance_ratio = 0.0
        decline_ratio = 0.0
        unchanged_ratio = 0.0

    active.sort(
        key=lambda item: item["volume"],
        reverse=True,
    )

    return {
        "status": "success",
        "source": "upstox_live",
        "total_instruments": len(universe),
        "quoted_instruments": len(quotes),
        "breadth": {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "total": total,
            "advance_ratio": advance_ratio,
            "decline_ratio": decline_ratio,
            "unchanged_ratio": unchanged_ratio,
        },
        "most_active": active[:limit],
    }


def get_market_activity(
    limit: int = 10,
) -> dict:
    """
    Return current NSE Market Breadth and Most Active Stocks.

    Uses the same short-lived full-NSE quote snapshot as the
    movers endpoint. This keeps the Dashboard efficient and
    avoids duplicate Upstox quote requests.
    """

    if limit < 1:
        limit = 10

    limit = min(limit, 50)

    universe, quote_data = _get_nse_quote_snapshot()

    result = _build_market_activity(
        universe,
        quote_data,
        limit=limit,
    )

    if (
        result["breadth"]["total"] == 0
        and not result["most_active"]
    ):
        result["source"] = "no_trading_data"
        result["message"] = (
            "No current NSE trading data is available."
        )

    return result


# =========================================================
# MOVERS CACHE HELPERS
# =========================================================

def _save_movers_cache(data: dict):
    """
    Save the latest successful trading-session movers.
    """

    try:
        payload = {
            "saved_at": time.time(),
            "data": data,
        }

        with open(
            MOVERS_CACHE_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        print(
            "Market movers cache saved:",
            MOVERS_CACHE_FILE,
        )

    except Exception as exc:
        # Cache failure should never break the API.
        print(
            "Could not save movers cache:",
            exc,
        )


def _load_movers_cache():
    """
    Load the latest successful movers cache.
    """

    if not os.path.exists(MOVERS_CACHE_FILE):
        print("No movers cache exists.")
        return None

    try:
        with open(
            MOVERS_CACHE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            return None

        data = payload.get("data")

        if not isinstance(data, dict):
            return None

        saved_at = payload.get("saved_at")

        if saved_at:
            age = time.time() - float(saved_at)

            print(
                f"Movers cache age: "
                f"{round(age / 3600, 2)} hours"
            )

        return data

    except Exception as exc:
        print(
            "Could not load movers cache:",
            exc,
        )
        return None

def get_market_movers(
    limit: int = 5,
) -> dict:
    """
    Get real NSE Top Gainers and Top Losers.

    During a trading session:
        - Fetch current Upstox quotes
        - Calculate movers
        - Save successful result to cache

    When the market is closed:
        - Use the latest successful movers cache
        - This prevents empty gainers/losers on
          Sundays and market holidays.
    """

    if limit < 1:
        limit = 5

    limit = min(limit, 20)

    # -----------------------------------------------------
    # Load one short-lived NSE quote snapshot
    # -----------------------------------------------------

    universe, quote_data = _get_nse_quote_snapshot()

    quotes = quote_data.get(
        "data",
        {},
    )

    movers = []

    # -----------------------------------------------------
    # Build movers
    # -----------------------------------------------------

    for instrument in universe:

        quote = find_quote(
            quote_data,
            instrument["instrument_key"],
        )

        mover = _build_mover(
            instrument,
            quote,
        )

        if mover is not None:
            movers.append(mover)

    print(
        f"Valid mover records: {len(movers)}"
    )

    # -----------------------------------------------------
    # Gainers
    # -----------------------------------------------------

    gainers = sorted(
        [
            item
            for item in movers
            if item["change"] > 0
        ],
        key=lambda item: item["change"],
        reverse=True,
    )

    # -----------------------------------------------------
    # Losers
    # -----------------------------------------------------

    losers = sorted(
        [
            item
            for item in movers
            if item["change"] < 0
        ],
        key=lambda item: item["change"],
    )

    # -----------------------------------------------------
    # CASE 1:
    # We have actual movers
    # -----------------------------------------------------

    if gainers or losers:

        result = {
            "status": "success",
            "source": "upstox_live",
            "total_instruments": len(universe),
            "quoted_instruments": len(quotes),
            "gainers": gainers[:limit],
            "losers": losers[:limit],
        }

        # Save latest successful session
        _save_movers_cache(result)

        return result

    # -----------------------------------------------------
    # CASE 2:
    # Market closed / no movement available
    # -----------------------------------------------------

    print(
        "No current gainers/losers found."
    )

    print(
        "Trying latest successful movers cache..."
    )

    cached = _load_movers_cache()

    if cached:

        cached_gainers = cached.get(
            "gainers",
            [],
        )

        cached_losers = cached.get(
            "losers",
            [],
        )

        return {
            "status": "success",
            "source": "latest_trading_session",
            "total_instruments": len(universe),
            "quoted_instruments": len(quotes),
            "gainers": cached_gainers[:limit],
            "losers": cached_losers[:limit],
        }

    # -----------------------------------------------------
    # CASE 3:
    # No live movers AND no cache
    # -----------------------------------------------------

    return {
        "status": "success",
        "source": "no_trading_data",
        "total_instruments": len(universe),
        "quoted_instruments": len(quotes),
        "gainers": [],
        "losers": [],
        "message": (
            "Market is closed and no previous "
            "trading-session movers are cached yet."
        ),
    }

# =========================================================
# Compatibility helpers used by StockAI main.py
# =========================================================

def get_intraday_candles(instrument_key: str, unit: str = "minutes", interval: str = "1") -> dict:
    _require_access_token()
    if not instrument_key or "|" not in instrument_key:
        raise ValueError("A valid Upstox instrument_key is required")
    url = f"{UPSTOX_V3_BASE_URL}/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
    response = requests.get(url, headers=_auth_headers(), timeout=15)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected intraday response: {result}")
    return result


def get_market_status(exchange: str = "NSE") -> dict:
    _require_access_token()
    exchange = (exchange or "NSE").upper().strip()
    url = f"{UPSTOX_BASE_URL}/market/status/{exchange}"
    response = requests.get(url, headers=_auth_headers(), timeout=15)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected market-status response: {result}")
    return result


def get_market_activity(limit: int = 10) -> dict:
    limit = max(1, min(int(limit), 50))
    universe = get_nse_equity_universe()
    if not universe:
        return {"status":"success","source":"upstox","breadth":{"advances":0,"declines":0,"unchanged":0,"total":0,"advance_ratio":0,"decline_ratio":0,"unchanged_ratio":0},"most_active":[]}
    keys = [x["instrument_key"] for x in universe]
    qdata = get_market_quotes_batched(keys)
    quotes = qdata.get("data", {}) if isinstance(qdata, dict) else {}
    rows=[]; advances=declines=unchanged=0
    for item in universe:
        q=find_quote(qdata,item["instrument_key"])
        if not isinstance(q,dict): continue
        lp=q.get("last_price")
        try: lp=float(lp)
        except (TypeError,ValueError): continue
        if lp<=0: continue
        ch=_quote_change_percent(q)
        if ch is None: continue
        if ch>0: advances+=1
        elif ch<0: declines+=1
        else: unchanged+=1
        rows.append({"symbol":item["symbol"],"name":item["name"],"exchange":item["exchange"],"instrument_key":item["instrument_key"],"last_price":round(lp,2),"change":round(float(ch),2),"volume":q.get("volume") or 0,"value":(q.get("volume") or 0)*lp})
    rows.sort(key=lambda x: float(x.get("volume") or 0), reverse=True)
    total=advances+declines+unchanged
    return {"status":"success","source":"upstox_live","breadth":{"advances":advances,"declines":declines,"unchanged":unchanged,"total":total,"advance_ratio":round(advances/total,4) if total else 0,"decline_ratio":round(declines/total,4) if total else 0,"unchanged_ratio":round(unchanged/total,4) if total else 0},"most_active":rows[:limit]}

# =========================================================
# MARKET SCANNER — PHASE 5, LAYER 1
# =========================================================

def get_market_scanner(
    limit: int = 50,
    sort_by: str = "volume",
) -> dict:
    """
    Return the live NSE equity market scanner.

    This layer only provides:
        - NSE equity universe
        - live quote data
        - price
        - percentage change
        - previous close
        - OHLC
        - volume

    Technical analysis and opportunity detection are handled
    by the frontend scanner analysis layer and are NOT generated
    here.
    """

    if limit < 1:
        limit = 50

    limit = min(limit, 100)

    if sort_by not in {
        "volume",
        "change",
        "value",
    }:
        sort_by = "volume"

    # ---------------------------------------------------------
    # Get the same NSE snapshot already used by movers/activity
    # ---------------------------------------------------------

    universe, quote_data = _get_nse_quote_snapshot()

    quotes = quote_data.get(
        "data",
        {},
    )

    if not isinstance(quotes, dict):
        quotes = {}

    stocks = []

    # ---------------------------------------------------------
    # Build scanner records
    # ---------------------------------------------------------

    for instrument in universe:

        quote = find_quote(
            quote_data,
            instrument["instrument_key"],
        )

        if not isinstance(quote, dict):
            continue

        mover = _build_mover(
            instrument,
            quote,
        )

        if mover is None:
            continue

        # -----------------------------------------------------
        # Volume
        # -----------------------------------------------------

        raw_volume = quote.get("volume")

        try:
            volume = float(raw_volume)
        except (TypeError, ValueError):
            volume = 0.0

        if volume < 0:
            volume = 0.0

        # -----------------------------------------------------
        # OHLC
        # -----------------------------------------------------

        open_price = mover.get("open")
        high_price = mover.get("high")
        low_price = mover.get("low")

        try:
            open_price = (
                float(open_price)
                if open_price is not None
                else None
            )
        except (TypeError, ValueError):
            open_price = None

        try:
            high_price = (
                float(high_price)
                if high_price is not None
                else None
            )
        except (TypeError, ValueError):
            high_price = None

        try:
            low_price = (
                float(low_price)
                if low_price is not None
                else None
            )
        except (TypeError, ValueError):
            low_price = None

        # -----------------------------------------------------
        # Scanner row
        # -----------------------------------------------------

        stocks.append({
            "rank": 0,

            "symbol": mover["symbol"],
            "name": mover["name"],
            "exchange": mover["exchange"],
            "instrument_key": mover["instrument_key"],

            "value": mover["value"],
            "change": mover["change"],
            "net_change": mover["net_change"],
            "previous_close": mover["previous_close"],

            "open": open_price,
            "high": high_price,
            "low": low_price,

            "volume": int(volume),

            "timestamp": mover.get(
                "timestamp"
            ),

            "status": "success",
        })

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    if sort_by == "volume":

        stocks.sort(
            key=lambda item: item.get(
                "volume",
                0,
            ),
            reverse=True,
        )

    elif sort_by == "change":

        stocks.sort(
            key=lambda item: item.get(
                "change",
                0,
            ),
            reverse=True,
        )

    elif sort_by == "value":

        stocks.sort(
            key=lambda item: item.get(
                "value",
                0,
            ),
            reverse=True,
        )

    # ---------------------------------------------------------
    # Assign rank after sorting
    # ---------------------------------------------------------

    stocks = stocks[:limit]

    for index, stock in enumerate(
        stocks,
        start=1,
    ):
        stock["rank"] = index

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------

    return {
        "status": "success",
        "source": "upstox_live",
        "sort_by": sort_by,

        "total_universe": len(
            universe
        ),

        "quoted_instruments": len(
            quotes
        ),

        "returned": len(
            stocks
        ),

        "stocks": stocks,
    }

def get_market_scanner(limit: int = 50, sort_by: str = "volume") -> dict:
    limit=max(1,min(int(limit),100)); sort_by=sort_by if sort_by in {"volume","change","value"} else "volume"
    universe=get_nse_equity_universe()
    keys=[x["instrument_key"] for x in universe]
    qdata=get_market_quotes_batched(keys); rows=[]
    for item in universe:
        q=find_quote(qdata,item["instrument_key"])
        if not isinstance(q,dict): continue
        try: lp=float(q.get("last_price"))
        except (TypeError,ValueError): continue
        if lp<=0: continue
        ch=_quote_change_percent(q)
        if ch is None: ch=0.0
        vol=q.get("volume") or 0
        try: value=float(vol)*lp
        except (TypeError,ValueError): value=0.0
        rows.append({"symbol":item["symbol"],"name":item["name"],"exchange":item["exchange"],"instrument_key":item["instrument_key"],"last_price":round(lp,2),"change":round(float(ch),2),"volume":vol,"value":round(value,2),"open":(q.get("ohlc") or {}).get("open"),"high":(q.get("ohlc") or {}).get("high"),"low":(q.get("ohlc") or {}).get("low"),"previous_close":(q.get("ohlc") or {}).get("close"),"timestamp":q.get("timestamp")})
    rows.sort(key=lambda x: float(x.get(sort_by) or 0), reverse=True)
    return {"status":"success","source":"upstox_live","total_universe":len(universe),"quoted_instruments":len(rows),"returned":min(limit,len(rows)),"stocks":rows[:limit]}
