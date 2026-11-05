"""Lightweight public-news/event retrieval for StockAI.

Uses Google News RSS to retrieve recent articles. Publication dates are
validated locally as well as constrained in the Google News search query.
If the feed is unavailable, an empty list is returned so the dashboard can
continue operating.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


_DEFAULT_DAYS = 7
_DEFAULT_LIMIT = 20
_MAX_DAYS = 30
_MAX_LIMIT = 50
_REQUEST_TIMEOUT_SECONDS = 10
_FUTURE_DATE_TOLERANCE = timedelta(hours=1)


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Convert a value to an integer and clamp it to a safe range."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _parse_published_date(value: str, local_timezone) -> datetime | None:
    """Parse an RSS publication date and return it in the requested timezone.

    Dates without an explicit timezone are treated as UTC. Unparseable dates
    are rejected rather than being assumed to be recent.
    """
    if not value:
        return None

    try:
        published = parsedate_to_datetime(value)
        if published is None:
            return None
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return published.astimezone(local_timezone)
    except (TypeError, ValueError, OverflowError):
        return None


def get_news_events(symbol: str, days: int = _DEFAULT_DAYS, limit: int = _DEFAULT_LIMIT) -> dict:
    """Return up to ``limit`` news items published within the last ``days``.

    The returned dictionary keeps the existing StockAI response fields:
    status, symbol, days, source, items, and optional message.
    """
    symbol = str(symbol or "").strip().upper()
    days = _bounded_int(days, default=_DEFAULT_DAYS, minimum=1, maximum=_MAX_DAYS)
    limit = _bounded_int(limit, default=_DEFAULT_LIMIT, minimum=1, maximum=_MAX_LIMIT)

    if not symbol:
        return {
            "status": "error",
            "message": "symbol is required",
            "items": [],
        }

    now = datetime.now().astimezone()
    cutoff = now - timedelta(days=days)

    # Ask Google News for recent results, then enforce the date window below.
    query = f"{symbol} NSE India when:{days}d"
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        response = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "StockAI/1.0"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)

        dated_items: list[tuple[datetime, dict[str, str]]] = []

        for node in root.findall("./channel/item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            published_at = (node.findtext("pubDate") or "").strip()
            source = (node.findtext("source") or "").strip()

            if not title or not link:
                continue

            published_date = _parse_published_date(published_at, now.tzinfo)
            if published_date is None:
                # Exclude undated or malformed articles to avoid false recency.
                continue

            if published_date < cutoff or published_date > now + _FUTURE_DATE_TOLERANCE:
                continue

            dated_items.append(
                (
                    published_date,
                    {
                        "title": title,
                        "url": link,
                        "published_at": published_at,
                        "source": source,
                    },
                )
            )

        # Sort the filtered articles newest-first before applying the limit.
        dated_items.sort(key=lambda item: item[0], reverse=True)
        items = [item for _, item in dated_items[:limit]]

        return {
            "status": "success",
            "symbol": symbol,
            "days": days,
            "source": "Google News RSS",
            "items": items,
        }

    except (requests.RequestException, ET.ParseError, ValueError) as exc:
        # Keep the existing graceful-degradation behavior for StockAI callers.
        return {
            "status": "success",
            "symbol": symbol,
            "days": days,
            "source": "unavailable",
            "items": [],
            "message": f"News feed unavailable: {exc}",
        }
