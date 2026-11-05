import requests

from .upstox_service import (
    ACCESS_TOKEN,
    search_stocks,
    get_market_quotes,
    find_quote,
)


UPSTOX_BASE_URL = "https://api.upstox.com/v2"


# ============================================================
# Generic Upstox GET
# ============================================================

def _upstox_get(url: str, params: dict | None = None) -> dict:
    """
    Make an authenticated request to Upstox.
    """

    if not ACCESS_TOKEN:
        return {}

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
        )

        if response.status_code != 200:
            return {}

        payload = response.json()

        if isinstance(payload, dict):
            return payload

    except Exception:
        pass

    return {}


# ============================================================
# Company Profile
# ============================================================

def _get_company_profile(isin: str) -> dict:

    if not isin:
        return {}

    url = f"{UPSTOX_BASE_URL}/fundamentals/{isin}/profile"

    payload = _upstox_get(url)

    data = payload.get("data", {})

    return data if isinstance(data, dict) else {}


# ============================================================
# Key Ratios
# ============================================================

def _get_key_ratios(isin: str) -> dict:

    if not isin:
        return {}

    url = f"{UPSTOX_BASE_URL}/fundamentals/{isin}/key-ratios"

    payload = _upstox_get(url)

    data = payload.get("data", [])

    if not isinstance(data, list):
        return {}

    ratios = {}

    for item in data:

        if not isinstance(item, dict):
            continue

        name = item.get("name")

        if isinstance(name, str):
            ratios[name.strip()] = item.get(
                "company_value"
            )

    return ratios


# ============================================================
# Income Statement
# ============================================================

def _get_income_statement(isin: str) -> dict:

    if not isin:
        return {}

    url = f"{UPSTOX_BASE_URL}/fundamentals/{isin}/income-statement"

    params = {
        "type": "consolidated",
        "time_period": "yearly",
        "fs": "true",
    }

    payload = _upstox_get(
        url,
        params=params,
    )

    data = payload.get("data", {})

    return data if isinstance(data, dict) else {}


# ============================================================
# Balance Sheet
# ============================================================

def _get_balance_sheet(isin: str) -> dict:

    if not isin:
        return {}

    url = f"{UPSTOX_BASE_URL}/fundamentals/{isin}/balance-sheet"

    params = {
        "type": "consolidated",
        "fs": "true",
    }

    payload = _upstox_get(
        url,
        params=params,
    )

    data = payload.get("data", {})

    return data if isinstance(data, dict) else {}


# ============================================================
# Corporate Actions
# ============================================================

def _get_corporate_actions(isin: str) -> list:

    """
    Get corporate actions.

    Used primarily for dividend information.
    """

    if not isin:
        return []

    url = (
        f"{UPSTOX_BASE_URL}"
        f"/fundamentals/{isin}/corporate-actions"
    )

    payload = _upstox_get(url)

    data = payload.get("data", [])

    return data if isinstance(data, list) else []


# ============================================================
# Latest History Value
# ============================================================

def _latest_history_value(history: list) -> float | None:

    if not isinstance(history, list) or not history:
        return None

    first = history[0]

    if not isinstance(first, dict):
        return None

    value = first.get("value")

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Find Full Statement Value
# ============================================================

def _find_full_statement_value(
    full_statement: list,
    particular: str,
) -> float | None:

    if not isinstance(full_statement, list):
        return None

    target = particular.strip().lower()

    for item in full_statement:

        if not isinstance(item, dict):
            continue

        name = item.get("particular")

        if not isinstance(name, str):
            continue

        if name.strip().lower() != target:
            continue

        return _latest_history_value(
            item.get("history", [])
        )

    return None


# ============================================================
# Find Full Statement Value From Multiple Names
# ============================================================

def _find_first_full_statement_value(
    full_statement: list,
    names: list[str],
) -> float | None:

    for name in names:

        value = _find_full_statement_value(
            full_statement,
            name,
        )

        if value is not None:
            return value

    return None


# ============================================================
# Extract ISIN
# ============================================================

def _extract_isin(
    stock: dict,
    instrument_key: str,
) -> str | None:

    if isinstance(stock, dict):

        isin = stock.get("isin")

        if isinstance(isin, str) and isin.strip():
            return isin.strip()

    if (
        isinstance(instrument_key, str)
        and "|" in instrument_key
    ):

        possible_isin = instrument_key.split(
            "|",
            1,
        )[1].strip()

        if possible_isin:
            return possible_isin

    return None


# ============================================================
# Extract Latest Price
# ============================================================

def _to_float(value) -> float | None:

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(
            str(value)
            .replace(",", "")
            .replace("%", "")
            .strip()
        )
    except (TypeError, ValueError):
        return None


# ============================================================
# Market Data
# ============================================================

def _extract_market_data(
    quote: dict | None,
) -> dict:
    """Extract and reconcile market data from an Upstox quote.

    Priority:
    1. Calculate previous close from last_price - net_change.
    2. Fall back to OHLC close or an explicit previous-close field.
    3. Derive net_change from last_price and previous_close
       when the source net_change is missing or invalid.
    """

    import math

    market_data = {
        "last_price": None,
        "open": None,
        "high": None,
        "low": None,
        "previous_close": None,
        "volume": None,
        "net_change": None,
    }

    if not isinstance(quote, dict):
        return market_data

    def valid_number(value) -> float | None:
        number = _to_float(value)

        if number is None or not math.isfinite(number):
            return None

        return number

    # Current traded price
    last_price = valid_number(
        quote.get("last_price", quote.get("ltp"))
    )

    # Absolute price change from the previous close
    net_change = valid_number(quote.get("net_change"))

    if net_change is None:
        net_change = valid_number(quote.get("change"))

    ohlc = quote.get("ohlc")
    if not isinstance(ohlc, dict):
        ohlc = {}

    # Extract OHLC fields
    market_data["last_price"] = last_price

    market_data["open"] = ohlc.get(
        "open",
        quote.get("open"),
    )

    market_data["high"] = ohlc.get(
        "high",
        quote.get("high"),
    )

    market_data["low"] = ohlc.get(
        "low",
        quote.get("low"),
    )

    market_data["volume"] = quote.get("volume")

    # Preserve available previous-close values as fallbacks.
    fallback_close = valid_number(ohlc.get("close"))

    if fallback_close is None or fallback_close <= 0:
        fallback_close = valid_number(
            quote.get(
                "previous_close",
                quote.get("prev_close"),
            )
        )

    # Prefer the value implied by last_price and net_change
    # whenever it produces a valid positive previous close.
    calculated_close = None

    if (
        last_price is not None
        and last_price > 0
        and net_change is not None
    ):
        candidate = last_price - net_change

        if math.isfinite(candidate) and candidate > 0:
            calculated_close = candidate

    if calculated_close is not None:
        previous_close = calculated_close
    else:
        previous_close = fallback_close

    # Ensure the returned price change agrees with the selected
    # previous close when the source change is absent or unusable.
    if (
        last_price is not None
        and last_price > 0
        and previous_close is not None
        and previous_close > 0
    ):
        if (
            net_change is None
            or calculated_close is None
        ):
            net_change = last_price - previous_close

    market_data["previous_close"] = previous_close
    market_data["net_change"] = net_change

    return market_data


# ============================================================
# Parse Percentage
# ============================================================

def _parse_percentage(value) -> float | None:

    if value is None:
        return None

    try:
        return float(
            str(value)
            .replace("%", "")
            .replace(",", "")
            .strip()
        )

    except (TypeError, ValueError):
        return None


# ============================================================
# Dividend Yield
# ============================================================

def _calculate_dividend_yield(
    corporate_actions: list,
    current_price: float | None,
) -> float | None:

    """
    Calculate dividend yield using dividend events.

    Uses the latest 12-month window approximately by
    collecting dividend events whose expiry/ex-dividend
    date falls within the latest year.

    If dates cannot be parsed reliably, the latest
    dividend events are not blindly treated as annual
    dividends.
    """

    if (
        not corporate_actions
        or current_price is None
        or current_price <= 0
    ):
        return None

    from datetime import datetime, timedelta

    today = datetime.now().date()

    one_year_ago = today - timedelta(days=365)

    total_dividend = 0.0

    valid_dividend_count = 0

    for action in corporate_actions:

        if not isinstance(action, dict):
            continue

        if str(
            action.get("name", "")
        ).strip().lower() != "dividend":
            continue

        date_text = action.get(
            "expiry_date"
        )

        amount = _to_float(
            action.get("amount")
        )

        if not date_text or amount is None:
            continue

        parsed_date = None

        # Upstox commonly returns dates such as:
        # "14 Aug 2025"
        for date_format in (
            "%d %b %Y",
            "%d %B %Y",
            "%Y-%m-%d",
        ):

            try:
                parsed_date = datetime.strptime(
                    str(date_text),
                    date_format,
                ).date()

                break

            except ValueError:
                continue

        if parsed_date is None:
            continue

        if one_year_ago <= parsed_date <= today:

            total_dividend += amount

            valid_dividend_count += 1

    if valid_dividend_count == 0:
        return None

    return round(
        (total_dividend / current_price) * 100,
        4,
    )


# ============================================================
# Main Fundamentals
# ============================================================

def get_fundamentals(symbol: str) -> dict:

    try:

        # ====================================================
        # 1. Search instrument
        # ====================================================

        search_result = search_stocks(symbol)

        stocks = []

        if isinstance(search_result, dict):
            stocks = search_result.get(
                "data",
                [],
            )

        elif isinstance(search_result, list):
            stocks = search_result

        if not stocks:

            return {
                "status": "error",
                "message": f"Stock not found: {symbol}",
            }

        # Prefer exact symbol match
        stock = stocks[0]

        requested_symbol = symbol.strip().upper()

        for candidate in stocks:

            if not isinstance(candidate, dict):
                continue

            candidate_symbol = (
                candidate.get("trading_symbol")
                or candidate.get("symbol")
                or ""
            )

            if (
                isinstance(candidate_symbol, str)
                and candidate_symbol.upper()
                == requested_symbol
            ):

                stock = candidate
                break

        # ====================================================
        # 2. Instrument
        # ====================================================

        instrument_key = stock.get(
            "instrument_key"
        )

        if not instrument_key:

            return {
                "status": "error",
                "message": (
                    f"Instrument key not found for "
                    f"{symbol}"
                ),
            }

        isin = _extract_isin(
            stock,
            instrument_key,
        )

        # ====================================================
        # 3. Market Quote
        # ====================================================

        quote_response = get_market_quotes(
            [instrument_key]
        )

        quote = find_quote(
            quote_response,
            instrument_key,
        )

        market_data = _extract_market_data(
            quote
        )

        current_price = _to_float(
            market_data.get("last_price")
        )

        # ====================================================
        # 4. Company Profile
        # ====================================================

        profile = _get_company_profile(
            isin
        )

        sector = profile.get(
            "sector",
            "Unknown",
        )

        if not isinstance(sector, str):
            sector = "Unknown"

        sector = sector.strip() or "Unknown"

        # ====================================================
        # 5. Key Ratios
        # ====================================================

        ratios = _get_key_ratios(
            isin
        )

        pe_ratio = ratios.get("P/E")

        pb_ratio = ratios.get("P/B")

        roa = ratios.get("ROA")

        roe = ratios.get("ROE")

        roce = ratios.get("ROCE")

        ev_ebitda = ratios.get(
            "EV/EBITDA"
        )

        quick_ratio = ratios.get(
            "Quick Ratio"
        )

        # ====================================================
        # 6. Income Statement
        # ====================================================

        income_statement = _get_income_statement(
            isin
        )

        income_full_statement = (
            income_statement.get(
                "full_statement",
                [],
            )
        )

        eps_basic = _find_full_statement_value(
            income_full_statement,
            "EPS - Basic",
        )

        eps_diluted = _find_full_statement_value(
            income_full_statement,
            "EPS - Diluted",
        )

        # ====================================================
        # 7. Balance Sheet
        # ====================================================

        balance_sheet = _get_balance_sheet(
            isin
        )

        balance_full_statement = (
            balance_sheet.get(
                "full_statement",
                [],
            )
        )

        total_assets = _find_full_statement_value(
            balance_full_statement,
            "Total Assets",
        )

        current_assets = _find_full_statement_value(
            balance_full_statement,
            "Current Assets",
        )

        current_liabilities = (
            _find_full_statement_value(
                balance_full_statement,
                "Current Liabilities",
            )
        )

        non_current_liabilities = (
            _find_full_statement_value(
                balance_full_statement,
                "Non-Current Liabilities",
            )
        )

        equity_capital = (
            _find_full_statement_value(
                balance_full_statement,
                "Equity Capital",
            )
        )

        # ====================================================
        # 8. Book Value Per Share
        # ====================================================

        book_value = None

        pb_numeric = _to_float(
            pb_ratio
        )

        if (
            current_price is not None
            and current_price > 0
            and pb_numeric is not None
            and pb_numeric > 0
        ):

            # Upstox P/B definition:
            #
            # P/B = Market Price / Book Value Per Share
            #
            # Therefore:
            #
            # Book Value Per Share =
            # Market Price / P/B

            book_value = round(
                current_price / pb_numeric,
                2,
            )

        # ====================================================
        # 9. Market Capitalization
        # ====================================================

        market_cap = None

        if (
            equity_capital is not None
            and pb_numeric is not None
            and pb_numeric > 0
        ):

            # Since:
            #
            # Market Cap / Equity =
            # Market Price / Book Value
            #
            # and:
            #
            # Market Price / Book Value = P/B
            #
            # therefore:
            #
            # Market Cap = Equity × P/B
            #
            # Both values are in crore.

            market_cap = round(
                equity_capital * pb_numeric,
                2,
            )

        # ====================================================
        # 10. Corporate Actions
        # ====================================================

        corporate_actions = (
            _get_corporate_actions(
                isin
            )
        )

        dividend_yield = (
            _calculate_dividend_yield(
                corporate_actions,
                current_price,
            )
        )

        # ====================================================
        # 11. Debt-to-Equity
        # ====================================================

        debt_to_equity = None

        # Upstox Balance Sheet provides total liabilities
        # and equity capital.
        #
        # For this StockAI fundamentals metric we use:
        #
        # Debt-to-Equity = Total Liabilities / Equity
        #
        # This is the balance-sheet liabilities-to-equity
        # convention, not a pure interest-bearing-debt ratio.

        total_liabilities = None

        # First try the summary history.
        balance_history = balance_sheet.get(
            "history",
            [],
        )

        if isinstance(balance_history, list) and balance_history:

            latest_balance = balance_history[0]

            if isinstance(latest_balance, dict):

                total_liabilities = _to_float(
                    latest_balance.get(
                        "total_liability"
                    )
                )

        # Fallback: calculate total liabilities from
        # current + non-current liabilities if necessary.
        if total_liabilities is None:

            if (
                current_liabilities is not None
                and non_current_liabilities is not None
            ):

                total_liabilities = (
                    current_liabilities
                    + non_current_liabilities
                )

        # Calculate D/E only when both values exist.
        if (
            total_liabilities is not None
            and equity_capital is not None
            and equity_capital > 0
        ):

            debt_to_equity = round(
                total_liabilities / equity_capital,
                4,
            )

        # ====================================================
        # 12. Response
        # ====================================================

        return {

            "status": "success",

            "symbol": stock.get(
                "trading_symbol",
                stock.get(
                    "symbol",
                    symbol,
                ),
            ),

            "name": stock.get(
                "name",
                stock.get(
                    "short_name",
                    symbol,
                ),
            ),

            "instrument_key": instrument_key,

            "isin": isin,

            "exchange": stock.get(
                "exchange",
                "NSE",
            ),

            # ------------------------------------------------
            # Company Profile
            # ------------------------------------------------

            "sector": sector,

            # ------------------------------------------------
            # Market Data
            # ------------------------------------------------

            "market_data": market_data,

            # ------------------------------------------------
            # Fundamentals
            # ------------------------------------------------

            "fundamentals": {

                "market_cap": market_cap,

                "pe_ratio": pe_ratio,

                "pb_ratio": pb_ratio,

                "eps": eps_basic,

                "eps_basic": eps_basic,

                "eps_diluted": eps_diluted,

                "dividend_yield": dividend_yield,

                "book_value": book_value,

                "roe": roe,

                "roa": roa,

                "roce": roce,

                "ev_ebitda": ev_ebitda,

                "quick_ratio": quick_ratio,

                "total_assets": total_assets,

                "current_assets": current_assets,

                "current_liabilities": current_liabilities,

                "non_current_liabilities": (
                    non_current_liabilities
                ),

                "equity_capital": equity_capital,

                "debt_to_equity": debt_to_equity,
            },

            # ------------------------------------------------
            # Period information
            # ------------------------------------------------

            "fundamental_period": (
                income_statement.get(
                    "time_period",
                    "yearly",
                )
            ),

            "fundamental_type": (
                income_statement.get(
                    "type",
                    "consolidated",
                )
            ),

            "balance_sheet_period": (
                balance_sheet.get(
                    "time_period",
                    "yearly",
                )
            ),

            "balance_sheet_type": (
                balance_sheet.get(
                    "type",
                    "consolidated",
                )
            ),

            "note": (
                "Fundamental data is sourced from "
                "Upstox Company Fundamentals APIs. "
                "Book value per share is derived from "
                "current price divided by P/B. "
                "Market capitalization is derived from "
                "equity capital multiplied by P/B. "
                "Dividend yield is calculated from "
                "dividend corporate actions and current "
                "market price. Debt-to-equity remains "
                "null because the tested balance sheet "
                "does not provide a dedicated debt value."
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": str(exc),
        }