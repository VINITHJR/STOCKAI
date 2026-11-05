"""Create a candidate multi-stock key list from StockAI's live Upstox NSE universe.

Run from the StockAI project root:
python -m backend.app.ml.lstm_prepare_instrument_list
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.services.upstox_service import get_nse_equity_universe

# The existing two test stocks are intentionally first; other symbols are
# resolved against the live Upstox instrument master, not hard-coded ISINs.
PREFERRED_SYMBOLS = [
    "INFY", "20MICRONS", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK",
    "ITC", "SBIN", "BHARTIARTL", "LT", "HINDUNILVR", "AXISBANK",
    "KOTAKBANK", "MARUTI", "TATAMOTORS", "TATASTEEL", "WIPRO",
    "BAJFINANCE", "SUNPHARMA", "TITAN", "ASIANPAINT", "NTPC",
    "POWERGRID", "ONGC", "ADANIPORTS", "ULTRACEMCO",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a candidate Upstox NSE instrument-key list.")
    parser.add_argument(
        "--output",
        default="backend/app/ml/models/lstm_instrument_keys.txt",
        help="Output text file path (default: backend/app/ml/models/lstm_instrument_keys.txt)",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of preferred symbols to write.")
    args = parser.parse_args()

    universe = get_nse_equity_universe()
    if not universe:
        raise RuntimeError("Upstox NSE instrument universe is empty; cannot create key list.")

    by_symbol = {}
    for item in universe:
        symbol = str(item.get("symbol") or "").strip().upper()
        key = str(item.get("instrument_key") or "").strip()
        if symbol and key:
            by_symbol[symbol] = key

    found, missing = [], []
    for symbol in PREFERRED_SYMBOLS:
        key = by_symbol.get(symbol)
        if key:
            found.append((symbol, key))
        else:
            missing.append(symbol)

    found = found[:max(1, args.limit)]
    if len(found) < 8:
        raise RuntimeError(
            f"Only {len(found)} preferred symbols were found in the current instrument master. "
            f"Missing: {missing}. Inspect your NSE universe symbols and update PREFERRED_SYMBOLS."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# StockAI LSTM v2 multi-stock instrument keys",
        "# Generated from the current Upstox NSE equity instrument master.",
        "# Review these symbols before training; each must have enough daily history.",
        "# One Upstox instrument_key per line.",
    ]
    lines.extend(f"{key}  # {symbol}" for symbol, key in found)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(found)} candidate keys to: {output}")
    print("\nSelected symbols:")
    for symbol, key in found:
        print(f"{symbol:14} {key}")
    if missing:
        print("\nPreferred symbols not found in this instrument master:")
        print(", ".join(missing))
    print("\nReview the file and keep 8-20 instruments with sufficient historical candles.")


if __name__ == "__main__":
    main()
