"""Phase 15 Part 4.2: realistic, OHLC-based trading strategy simulation.

Research-only. This module:
- consumes existing primary-target out-of-fold predictions (no model fitting);
- fetches daily OHLCV through the Phase 15 historical loader;
- enters at next-session OPEN and exits at the seventh holding-session CLOSE;
- runs seven independently funded sleeves to avoid reusing capital prematurely;
- marks open positions to market at every session close;
- models integer shares, configurable delivery costs, slippage and a lagged ADV
  participation cap; and
- never reads the reserved final holdout period.

Run from the StockAI project root:
    python -m backend.app.ml.phase15.validation.trading_strategy_simulation

The cost profile is an editable Upstox NSE-equity-delivery assumption. Update
part4_2_cost_config.json when the actual broker, exchange, contract note or
execution assumptions differ. Slippage scenarios are assumptions, not fees.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("backend/app/ml/phase15")
REPORT_DIR = ROOT / "reports"
COST_CONFIG_PATH = ROOT / "validation" / "part4_2_cost_config.json"
KEYS_FILE = Path("backend/app/ml/models/lstm_instrument_keys.txt")

BENCHMARK_RUN_ID = "20261031_131021"
PRIMARY_TARGET = "next_open_to_close_return_7s"
OOF_PATH = REPORT_DIR / (
    f"phase15_benchmark_{BENCHMARK_RUN_ID}_primary_oof_predictions.csv"
)

START_DATE = "2024-12-01"  # warm-up for 20-session ADV known at signal close
MAX_OHLC_DATE = "2026-04-16"  # last allowed date; final holdout starts 2026-04-17
FINAL_HOLDOUT_START = "2026-04-17"
HOLDING_SESSIONS = 7
TOP_K = 5
SLEEVE_COUNT = 7
INITIAL_CAPITAL_INR = 1_000_000.0
ADV_WINDOW = 20
FETCH_PAUSE_SECONDS = 0.35
TRADING_DAYS_PER_YEAR = 252

EXPECTED_MODELS = {"zero_return", "train_mean", "ridge", "random_forest", "extra_trees"}
MODELS_TO_SIMULATE = ("extra_trees", "random_forest")
FOLD_EXPECTED = {
    1: {"sessions": 61, "rows": 1220, "start": "2025-01-14", "end": "2025-04-11"},
    2: {"sessions": 61, "rows": 1220, "start": "2025-04-15", "end": "2025-07-10"},
    3: {"sessions": 61, "rows": 1220, "start": "2025-07-11", "end": "2025-10-08"},
    4: {"sessions": 60, "rows": 1200, "start": "2025-10-09", "end": "2026-01-05"},
    5: {"sessions": 60, "rows": 1200, "start": "2026-01-06", "end": "2026-04-06"},
}
OOF_COLUMNS = [
    "fold_number", "model", "instrument_key", "session_date",
    "target_column", "actual_return", "predicted_return",
]


@dataclass(frozen=True)
class CostConfig:
    broker_name: str
    exchange: str
    brokerage_per_order_inr: float
    stt_buy_bps: float
    stt_sell_bps: float
    stamp_duty_buy_bps: float
    exchange_transaction_bps_before_2026_03_01: float
    exchange_transaction_bps_from_2026_03_01: float
    sebi_fee_per_crore_inr: float
    ipft_per_crore_inr: float
    gst_rate: float
    dp_charge_per_scrip_per_sell_day_inr: float
    slippage_bps_per_side: float
    max_adv_participation: float
    slippage_scenarios_bps: tuple[float, ...]

    def validate(self) -> None:
        nonnegative = [
            self.brokerage_per_order_inr, self.stt_buy_bps, self.stt_sell_bps,
            self.stamp_duty_buy_bps, self.exchange_transaction_bps_before_2026_03_01,
            self.exchange_transaction_bps_from_2026_03_01,
            self.sebi_fee_per_crore_inr, self.ipft_per_crore_inr,
            self.gst_rate, self.dp_charge_per_scrip_per_sell_day_inr,
            self.slippage_bps_per_side,
        ]
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in nonnegative):
            raise ValueError("Cost configuration contains a negative/non-finite value.")
        if not 0 <= self.max_adv_participation <= 1:
            raise ValueError("max_adv_participation must be between 0 and 1.")
        if self.gst_rate > 1:
            raise ValueError("gst_rate must be a decimal fraction, e.g. 0.18.")
        if not self.slippage_scenarios_bps:
            raise ValueError("At least one slippage scenario is required.")
        if any((not math.isfinite(x) or x < 0) for x in self.slippage_scenarios_bps):
            raise ValueError("Invalid slippage scenario.")


@dataclass
class Sleeve:
    sleeve_id: int
    cash: float
    positions: list[dict[str, Any]]
    opened_signal_date: str | None = None


def load_cost_config(path: Path = COST_CONFIG_PATH) -> CostConfig:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cost configuration: {path}. Copy the supplied example config "
            "to this path and review it against your broker's current tariff."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "broker_name", "exchange", "brokerage_per_order_inr", "stt_buy_bps",
        "stt_sell_bps", "stamp_duty_buy_bps",
        "exchange_transaction_bps_before_2026_03_01",
        "exchange_transaction_bps_from_2026_03_01",
        "sebi_fee_per_crore_inr", "ipft_per_crore_inr", "gst_rate",
        "dp_charge_per_scrip_per_sell_day_inr", "slippage_bps_per_side",
        "max_adv_participation", "slippage_scenarios_bps",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Cost config missing keys: {sorted(missing)}")
    config = CostConfig(
        broker_name=str(payload["broker_name"]),
        exchange=str(payload["exchange"]),
        brokerage_per_order_inr=float(payload["brokerage_per_order_inr"]),
        stt_buy_bps=float(payload["stt_buy_bps"]),
        stt_sell_bps=float(payload["stt_sell_bps"]),
        stamp_duty_buy_bps=float(payload["stamp_duty_buy_bps"]),
        exchange_transaction_bps_before_2026_03_01=float(
            payload["exchange_transaction_bps_before_2026_03_01"]
        ),
        exchange_transaction_bps_from_2026_03_01=float(
            payload["exchange_transaction_bps_from_2026_03_01"]
        ),
        sebi_fee_per_crore_inr=float(payload["sebi_fee_per_crore_inr"]),
        ipft_per_crore_inr=float(payload["ipft_per_crore_inr"]),
        gst_rate=float(payload["gst_rate"]),
        dp_charge_per_scrip_per_sell_day_inr=float(
            payload["dp_charge_per_scrip_per_sell_day_inr"]
        ),
        slippage_bps_per_side=float(payload["slippage_bps_per_side"]),
        max_adv_participation=float(payload["max_adv_participation"]),
        slippage_scenarios_bps=tuple(float(x) for x in payload["slippage_scenarios_bps"]),
    )
    config.validate()
    return config


def load_instrument_keys(path: Path = KEYS_FILE) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    keys = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        key = raw.split("#", 1)[0].strip()
        if key:
            keys.append(key)
    if len(keys) != 20 or len(set(keys)) != 20:
        raise ValueError(f"Expected 20 unique instrument keys, found {len(keys)}.")
    return keys


def load_oof_predictions(path: Path = OOF_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=OOF_COLUMNS)
    frame["fold_number"] = pd.to_numeric(frame["fold_number"], errors="raise").astype(int)
    for col in ("actual_return", "predicted_return"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    if not np.isfinite(frame[["actual_return", "predicted_return"]].to_numpy()).all():
        raise ValueError("OOF predictions contain NaN or infinite values.")
    frame["session_date"] = pd.to_datetime(
        frame["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if set(frame["target_column"].unique()) != {PRIMARY_TARGET}:
        raise ValueError("OOF file does not contain the required primary target.")
    if set(frame["model"].unique()) != EXPECTED_MODELS:
        raise ValueError("OOF file model list does not match the Phase 15 benchmark.")
    if len(frame) != 30_300:
        raise ValueError(f"Expected 30,300 OOF rows, found {len(frame)}.")
    key_cols = ["fold_number", "model", "instrument_key", "session_date"]
    if frame.duplicated(key_cols).any():
        raise ValueError("Duplicate OOF prediction keys found.")
    if frame["session_date"].max() >= FINAL_HOLDOUT_START:
        raise ValueError("STOP: OOF predictions overlap the reserved final holdout.")
    for fold, spec in FOLD_EXPECTED.items():
        f = frame[frame["fold_number"] == fold]
        if f.empty:
            raise ValueError(f"Missing OOF fold {fold}.")
        for model in EXPECTED_MODELS:
            m = f[f["model"] == model]
            if len(m) != spec["rows"]:
                raise ValueError(f"Fold {fold}/{model}: unexpected row count {len(m)}.")
            dates = sorted(m["session_date"].unique())
            if len(dates) != spec["sessions"] or dates[0] != spec["start"] or dates[-1] != spec["end"]:
                raise ValueError(f"Fold {fold}/{model}: validation date range mismatch.")
            if not (m.groupby("session_date").size() == 20).all():
                raise ValueError(f"Fold {fold}/{model}: expected 20 stocks per date.")
    actual_counts = frame.groupby(["fold_number", "instrument_key", "session_date"])[
        "actual_return"
    ].nunique()
    if (actual_counts > 1).any():
        raise ValueError("Actual target differs across models for the same stock/date.")
    if set(frame["instrument_key"].unique()) != set(load_instrument_keys()):
        raise ValueError("OOF stock universe differs from the instrument-key file.")
    return frame


def fetch_ohlcv_universe(stock_keys: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch daily OHLCV only through the permitted pre-holdout exit date."""
    from backend.app.ml.phase15.data.historical_loader import (
        compare_session_calendars,
        fetch_daily_candles,
    )

    stock_bars: dict[str, pd.DataFrame] = {}
    reference = fetch_daily_candles("NSE_INDEX|Nifty 50", START_DATE, MAX_OHLC_DATE)
    reference_dates = [r["session_date"] for r in reference["rows"]]
    if len(reference_dates) < ADV_WINDOW + 10:
        raise ValueError("NIFTY history is too short for evaluation and warm-up.")
    index_df = pd.DataFrame(reference["rows"])
    index_df["instrument_key"] = "NIFTY50_INDEX"
    print(f"NIFTY 50 OHLCV: {len(index_df)} sessions")
    time.sleep(FETCH_PAUSE_SECONDS)

    for number, key in enumerate(stock_keys, start=1):
        print(f"[{number:02d}/20] Fetching OHLCV: {key}")
        dataset = fetch_daily_candles(key, START_DATE, MAX_OHLC_DATE)
        comparison = compare_session_calendars(reference, dataset)
        if not comparison["exact_match"]:
            raise ValueError(
                f"Calendar mismatch for {key}: missing={comparison['missing_count']}, "
                f"extra={comparison['extra_count']}. No forward-fill/intersection permitted."
            )
        df = pd.DataFrame(dataset["rows"])
        required = {"session_date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"{key}: missing OHLCV columns.")
        values = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: non-finite OHLCV values.")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"{key}: non-positive OHLC price.")
        if (df["volume"] < 0).any():
            raise ValueError(f"{key}: negative volume.")
        if (df["high"] < df[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError(f"{key}: invalid high values.")
        if (df["low"] > df[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError(f"{key}: invalid low values.")
        if df["session_date"].tolist() != reference_dates:
            raise ValueError(f"{key}: calendar order differs from NIFTY.")
        stock_bars[key] = df
        time.sleep(FETCH_PAUSE_SECONDS)
    index_df = index_df[["instrument_key", "session_date", "open", "high", "low", "close", "volume"]]
    return stock_bars, index_df


def build_bar_lookup(stock_bars: dict[str, pd.DataFrame], index_df: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    all_frames = list(stock_bars.items()) + [("NIFTY50_INDEX", index_df)]
    first = next(iter(stock_bars.values()))
    calendar = first["session_date"].tolist()
    lookup: dict[str, dict[str, dict[str, float]]] = {}
    for key, frame in all_frames:
        if frame["session_date"].tolist() != calendar:
            raise ValueError(f"{key}: OHLC calendar is not identical to stock calendar.")
        lookup[key] = {
            row.session_date: {
                "open": float(row.open), "high": float(row.high),
                "low": float(row.low), "close": float(row.close),
                "volume": float(row.volume),
            }
            for row in frame.itertuples(index=False)
        }
    return calendar, lookup


def reconcile_oof_targets_with_ohlc(oof: pd.DataFrame, calendar: list[str], bars: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    selected = oof[oof["model"].isin(MODELS_TO_SIMULATE)]
    checked = 0
    max_abs_diff = 0.0
    for row in selected.itertuples(index=False):
        if row.session_date not in date_to_idx:
            raise ValueError(f"OOF signal date {row.session_date} absent from OHLC data.")
        i = date_to_idx[row.session_date]
        entry_i = i + 1
        exit_i = i + HOLDING_SESSIONS
        if i < ADV_WINDOW - 1:
            raise ValueError("Insufficient pre-signal history for ADV calculation.")
        if exit_i >= len(calendar):
            raise ValueError(f"Insufficient OHLC exit data for signal {row.session_date}.")
        if calendar[exit_i] >= FINAL_HOLDOUT_START:
            raise ValueError("STOP: Required exit date falls inside final holdout.")
        entry_date, exit_date = calendar[entry_i], calendar[exit_i]
        expected = bars[row.instrument_key][exit_date]["close"] / bars[row.instrument_key][entry_date]["open"] - 1.0
        diff = abs(float(row.actual_return) - expected)
        max_abs_diff = max(max_abs_diff, diff)
        if not np.isclose(float(row.actual_return), expected, rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"OOF/ohlc target mismatch: {row.instrument_key} {row.session_date}; "
                f"OOF={row.actual_return:.12f}, OHLC={expected:.12f}, diff={diff:.3g}"
            )
        checked += 1
    return {
        "passed": True,
        "selected_oof_rows_checked": checked,
        "max_absolute_return_difference": max_abs_diff,
        "target_formula": "close[signal_index+7] / open[signal_index+1] - 1",
    }


def top_k_fractional_weights(scores: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Top-k weights with equal fractional allocation at a tied boundary."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < k or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector for top-k selection.")
    if np.unique(scores).size < 2:
        raise ValueError("Constant predictions cannot rank stocks.")
    weights = np.zeros(len(scores), dtype=float)
    remaining = float(k)
    for score in np.sort(np.unique(scores))[::-1]:
        mask = scores == score
        count = int(mask.sum())
        allocation = min(1.0, remaining / count)
        weights[mask] = allocation
        remaining -= count * allocation
        if remaining <= 1e-12:
            break
    if not np.isclose(weights.sum(), float(k), atol=1e-10):
        raise RuntimeError("Top-k weights do not sum to k.")
    return weights


def fee_breakdown(
    notional: float, side: str, trade_date: str, config: CostConfig
) -> dict[str, float]:
    """Calculate NSE delivery fees using the exchange rate effective on trade_date."""
    if not math.isfinite(notional) or notional < 0:
        raise ValueError("Notional must be finite and non-negative.")
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if not trade_date:
        raise ValueError("trade_date is required for date-effective fee calculation.")
    exchange_bps = (
        config.exchange_transaction_bps_from_2026_03_01
        if trade_date >= "2026-03-01"
        else config.exchange_transaction_bps_before_2026_03_01
    )
    brokerage = config.brokerage_per_order_inr if notional > 0 else 0.0
    exchange = notional * exchange_bps / 10_000.0
    sebi = notional * config.sebi_fee_per_crore_inr / 10_000_000.0
    ipft = notional * config.ipft_per_crore_inr / 10_000_000.0
    stt_rate = config.stt_buy_bps if side == "BUY" else config.stt_sell_bps
    stt = notional * stt_rate / 10_000.0
    stamp = notional * config.stamp_duty_buy_bps / 10_000.0 if side == "BUY" else 0.0
    dp = config.dp_charge_per_scrip_per_sell_day_inr if side == "SELL" and notional > 0 else 0.0
    # Upstox's published delivery tariff states GST on brokerage, transaction,
    # demat and IPFT charges; STT, stamp duty and SEBI fees are not in this base.
    gst_base = brokerage + exchange + ipft + dp
    gst = gst_base * config.gst_rate
    return {
        "brokerage_inr": brokerage,
        "exchange_inr": exchange,
        "sebi_inr": sebi,
        "ipft_inr": ipft,
        "stt_inr": stt,
        "stamp_duty_inr": stamp,
        "dp_inr": dp,
        "gst_inr": gst,
        "total_fees_inr": brokerage + exchange + sebi + ipft + stt + stamp + dp + gst,
    }


def max_affordable_shares(
    budget: float, fill_price: float, volume_cap: int, trade_date: str, config: CostConfig
) -> int:
    """Integer-share binary search; includes buy fees in the cash constraint."""
    if budget <= 0 or fill_price <= 0 or volume_cap <= 0:
        return 0
    low, high = 0, min(volume_cap, int(math.floor(budget / fill_price)) + 1)
    while low < high:
        mid = (low + high + 1) // 2
        notional = mid * fill_price
        fees = fee_breakdown(notional, "BUY", trade_date, config)["total_fees_inr"]
        if notional + fees <= budget + 1e-8:
            low = mid
        else:
            high = mid - 1
    return low


def trailing_adv20(bars_for_stock: dict[str, dict[str, float]], calendar: list[str], signal_idx: int) -> float:
    """Average volume through the signal close; no future-volume leakage."""
    start = signal_idx - ADV_WINDOW + 1
    if start < 0:
        raise ValueError("Not enough past sessions for ADV20.")
    values = [bars_for_stock[calendar[i]]["volume"] for i in range(start, signal_idx + 1)]
    adv = float(np.mean(values))
    if not math.isfinite(adv) or adv <= 0:
        return 0.0
    return adv


def _mark_position_net(
    position: dict[str, Any], close: float, mark_date: str, config: CostConfig
) -> float:
    exit_fill = close * (1.0 - config.slippage_bps_per_side / 10_000.0)
    notional = position["qty"] * exit_fill
    fees = fee_breakdown(notional, "SELL", mark_date, config)["total_fees_inr"]
    return max(0.0, notional - fees)


def simulate_portfolio(
    portfolio_name: str,
    basket_type: str,
    signals: pd.DataFrame,
    calendar: list[str],
    bars: dict[str, dict[str, dict[str, float]]],
    stock_keys: list[str],
    config: CostConfig,
    initial_capital: float = INITIAL_CAPITAL_INR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one portfolio variant; return daily equity, closed trades, skipped signals."""
    if basket_type not in {"top_k", "equal_weight_20", "nifty50_price_only"}:
        raise ValueError(f"Unsupported basket type: {basket_type}")
    if initial_capital <= 0 or not math.isfinite(initial_capital):
        raise ValueError("Initial capital must be finite and positive.")
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    sleeves = [Sleeve(i + 1, initial_capital / SLEEVE_COUNT, []) for i in range(SLEEVE_COUNT)]

    schedule: dict[int, list[dict[str, Any]]] = {}
    for signal_date, group in signals.groupby("session_date", sort=True):
        if signal_date not in date_to_idx:
            raise ValueError(f"Signal date {signal_date} missing from OHLC calendar.")
        signal_idx = date_to_idx[signal_date]
        entry_idx = signal_idx + 1
        exit_idx = signal_idx + HOLDING_SESSIONS
        if exit_idx >= len(calendar) or calendar[exit_idx] >= FINAL_HOLDOUT_START:
            raise ValueError(f"Trade from {signal_date} would enter reserved holdout or lack exit data.")
        schedule.setdefault(entry_idx, []).append({
            "signal_date": signal_date,
            "fold_number": int(group["fold_number"].iloc[0]),
            "signal_idx": signal_idx,
            "exit_idx": exit_idx,
            "group": group.copy(),
        })

    first_eval_idx = min(date_to_idx[d] for d in signals["session_date"].unique())
    last_exit_idx = max(item["exit_idx"] for values in schedule.values() for item in values)
    trade_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    realized_costs_to_date = 0.0
    slippage_rate = config.slippage_bps_per_side / 10_000.0

    for day_idx in range(first_eval_idx, last_exit_idx + 1):
        session_date = calendar[day_idx]

        # Entries at today's open use signals produced after the previous close.
        for entry in schedule.get(day_idx, []):
            available = next((s for s in sleeves if not s.positions), None)
            if available is None:
                skipped_rows.append({
                    "portfolio": portfolio_name, "session_date": entry["signal_date"],
                    "entry_date": session_date, "fold_number": entry["fold_number"],
                    "reason": "no_free_capital_sleeve",
                })
                continue

            group = entry["group"].copy()
            if basket_type == "top_k":
                group = group.sort_values("instrument_key").reset_index(drop=True)
                scores = group["predicted_return"].to_numpy(dtype=float)
                if np.unique(scores).size < 2:
                    skipped_rows.append({
                        "portfolio": portfolio_name,
                        "session_date": entry["signal_date"],
                        "entry_date": session_date,
                        "fold_number": entry["fold_number"],
                        "reason": "constant_predictions_no_cross_sectional_rank",
                    })
                    continue
                raw_weights = top_k_fractional_weights(scores, TOP_K)
                candidates = [(str(group.iloc[i]["instrument_key"]), float(raw_weights[i]))
                              for i in range(len(group)) if raw_weights[i] > 0]
            elif basket_type == "equal_weight_20":
                candidates = [(key, 1.0) for key in stock_keys]
            else:
                candidates = [("NIFTY50_INDEX", 1.0)]

            weight_sum = sum(w for _, w in candidates)
            created_positions = []
            sleeve_budget = available.cash
            for instrument_key, slot_weight in candidates:
                bar = bars[instrument_key][session_date]
                raw_open = float(bar["open"])
                fill_price = raw_open * (1.0 + slippage_rate)
                if basket_type == "nifty50_price_only":
                    # Index is a non-tradable price reference: fractional units, no fees.
                    allocation = sleeve_budget * slot_weight / weight_sum
                    qty = allocation / fill_price
                    notional = qty * fill_price
                    fees = {"total_fees_inr": 0.0}
                    adv20 = float("nan")
                else:
                    adv20 = trailing_adv20(bars[instrument_key], calendar, entry["signal_idx"])
                    volume_cap = int(math.floor(adv20 * config.max_adv_participation))
                    allocation = sleeve_budget * slot_weight / weight_sum
                    qty = max_affordable_shares(
                        allocation, fill_price, volume_cap, session_date, config
                    )
                    if qty <= 0:
                        skipped_rows.append({
                            "portfolio": portfolio_name,
                            "session_date": entry["signal_date"],
                            "entry_date": session_date,
                            "fold_number": entry["fold_number"],
                            "instrument_key": instrument_key,
                            "reason": "zero_shares_after_cash_or_adv20_cap",
                            "adv20_shares": adv20,
                            "adv20_cap_shares": volume_cap,
                        })
                        continue
                    notional = qty * fill_price
                    fees = fee_breakdown(notional, "BUY", session_date, config)
                cash_outflow = notional + fees["total_fees_inr"]
                if cash_outflow > available.cash + 1e-6:
                    raise RuntimeError("Buy execution exceeded sleeve cash.")
                available.cash -= cash_outflow
                realized_costs_to_date += (
                    float(fees["total_fees_inr"])
                    + qty * (fill_price - raw_open)
                )
                created_positions.append({
                    "portfolio": portfolio_name,
                    "fold_number": entry["fold_number"],
                    "signal_date": entry["signal_date"],
                    "entry_date": session_date,
                    "exit_date": calendar[entry["exit_idx"]],
                    "entry_idx": day_idx,
                    "exit_idx": entry["exit_idx"],
                    "instrument_key": instrument_key,
                    "qty": qty,
                    "entry_open_raw": raw_open,
                    "entry_fill_price": fill_price,
                    "entry_notional_raw": (qty * raw_open),
                    "entry_notional_filled": notional,
                    "entry_fees": fees["total_fees_inr"],
                    "entry_fee_breakdown": fees,
                    "entry_slippage_inr": qty * (fill_price - raw_open),
                    "entry_cash_outflow": cash_outflow,
                    "adv20_at_signal": adv20,
                    "sleeve_id": available.sleeve_id,
                    "basket_type": basket_type,
                })
            if created_positions:
                available.positions = created_positions
                available.opened_signal_date = entry["signal_date"]
            else:
                skipped_rows.append({
                    "portfolio": portfolio_name, "session_date": entry["signal_date"],
                    "entry_date": session_date, "fold_number": entry["fold_number"],
                    "reason": "no_position_filled",
                })

        # Exit at today's close; the same sleeve is reusable from the next open.
        for sleeve in sleeves:
            if not sleeve.positions:
                continue
            remaining = []
            for position in sleeve.positions:
                if position["exit_idx"] == day_idx:
                    bar = bars[position["instrument_key"]][session_date]
                    raw_close = float(bar["close"])
                    fill_close = raw_close * (1.0 - slippage_rate)
                    sale_notional = position["qty"] * fill_close
                    if position["basket_type"] == "nifty50_price_only":
                        exit_fees = {"total_fees_inr": 0.0}
                    else:
                        exit_fees = fee_breakdown(sale_notional, "SELL", session_date, config)
                    sale_proceeds = sale_notional - exit_fees["total_fees_inr"]
                    sleeve.cash += sale_proceeds
                    gross_pnl = position["qty"] * (raw_close - position["entry_open_raw"])
                    net_pnl = sale_proceeds - position["entry_cash_outflow"]
                    exit_slippage = position["qty"] * (raw_close - fill_close)
                    entry_fee = float(position["entry_fees"])
                    exit_fee = float(exit_fees["total_fees_inr"])
                    total_cost = entry_fee + exit_fee + position["entry_slippage_inr"] + exit_slippage
                    realized_costs_to_date += exit_fee + exit_slippage
                    prior = max(0, day_idx - ADV_WINDOW)
                    if position["basket_type"] == "nifty50_price_only":
                        adv_exit = float("nan")
                        exit_adv_pct = float("nan")
                        exit_liquidity_flag = False
                    else:
                        hist = [bars[position["instrument_key"]][calendar[j]]["volume"]
                                for j in range(prior, day_idx)]
                        adv_exit = float(np.mean(hist)) if hist else float("nan")
                        exit_adv_pct = (position["qty"] / adv_exit) if adv_exit > 0 else float("inf")
                        exit_liquidity_flag = bool(exit_adv_pct > config.max_adv_participation)
                    trade_rows.append({
                        "portfolio": portfolio_name,
                        "fold_number": position["fold_number"],
                        "sleeve_id": position["sleeve_id"],
                        "instrument_key": position["instrument_key"],
                        "signal_date": position["signal_date"],
                        "entry_date": position["entry_date"],
                        "exit_date": session_date,
                        "holding_sessions": position["exit_idx"] - position["entry_idx"] + 1,
                        "quantity": position["qty"],
                        "entry_open_raw": position["entry_open_raw"],
                        "entry_fill_price": position["entry_fill_price"],
                        "exit_close_raw": raw_close,
                        "exit_fill_price": fill_close,
                        "entry_notional_raw_inr": position["entry_notional_raw"],
                        "exit_notional_raw_inr": position["qty"] * raw_close,
                        "gross_pnl_inr": gross_pnl,
                        "entry_fees_inr": entry_fee,
                        "exit_fees_inr": exit_fee,
                        "entry_slippage_inr": position["entry_slippage_inr"],
                        "exit_slippage_inr": exit_slippage,
                        "total_costs_inr": total_cost,
                        "net_pnl_inr": net_pnl,
                        "net_return_on_entry_cash_pct": 100.0 * net_pnl / position["entry_cash_outflow"],
                        "adv20_at_signal_shares": position["adv20_at_signal"],
                        "entry_adv_participation_pct": (
                            100.0 * position["qty"] / position["adv20_at_signal"]
                            if position["adv20_at_signal"] > 0 else np.nan
                        ),
                        "adv20_before_exit_shares": adv_exit,
                        "exit_adv_participation_pct": 100.0 * exit_adv_pct,
                        "exit_liquidity_warning": exit_liquidity_flag,
                    })
                else:
                    remaining.append(position)
            sleeve.positions = remaining
            if not remaining:
                sleeve.opened_signal_date = None

        # Close-of-session equity: raw gross value and conservative liquidation value.
        cash_total = sum(s.cash for s in sleeves)
        raw_market_value = 0.0
        net_liquidation_value = 0.0
        active_positions = 0
        for sleeve in sleeves:
            for position in sleeve.positions:
                close = bars[position["instrument_key"]][session_date]["close"]
                raw_value = position["qty"] * close
                raw_market_value += raw_value
                if position["basket_type"] == "nifty50_price_only":
                    net_liquidation_value += raw_value
                else:
                    net_liquidation_value += _mark_position_net(
                        position, close, session_date, config
                    )
                active_positions += 1
        net_equity = cash_total + net_liquidation_value
        estimated_exit_costs_open = raw_market_value - net_liquidation_value
        # Add back realized costs and estimated exit costs to obtain a true
        # pre-cost equity curve while retaining actual net cash accounting.
        gross_equity = (
            net_equity + realized_costs_to_date + estimated_exit_costs_open
        )
        equity_rows.append({
            "portfolio": portfolio_name,
            "session_date": session_date,
            "cash_inr": cash_total,
            "open_positions": active_positions,
            "raw_market_value_inr": raw_market_value,
            "estimated_exit_costs_open_inr": estimated_exit_costs_open,
            "realized_costs_to_date_inr": realized_costs_to_date,
            "gross_equity_before_costs_inr": gross_equity,
            "net_liquidation_equity_inr": net_equity,
            "daily_exposure_pct": 100.0 * raw_market_value / gross_equity if gross_equity > 0 else 0.0,
        })

    still_open = [p for s in sleeves for p in s.positions]
    if still_open:
        raise RuntimeError(f"Simulation ended with {len(still_open)} unclosed positions.")
    if not equity_rows:
        raise RuntimeError("Simulation produced no equity observations.")
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    skipped = pd.DataFrame(skipped_rows)
    return equity, trades, skipped


def calculate_portfolio_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float, scenario: str) -> dict[str, Any]:
    net = equity["net_liquidation_equity_inr"].astype(float)
    gross = equity["gross_equity_before_costs_inr"].astype(float)
    daily_returns = net.pct_change().dropna()
    periods = max(1, len(equity) - 1)
    ending = float(net.iloc[-1])
    gross_ending = float(gross.iloc[-1])
    cagr = (ending / initial_capital) ** (TRADING_DAYS_PER_YEAR / periods) - 1.0 if ending > 0 else -1.0
    vol = float(daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns) > 1 else 0.0
    mean_daily = float(daily_returns.mean()) if len(daily_returns) else 0.0
    daily_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = mean_daily / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR) if daily_std > 0 else 0.0
    drawdown = net / net.cummax() - 1.0
    total_costs = float(trades["total_costs_inr"].sum()) if not trades.empty else 0.0
    gross_pnl = float(trades["gross_pnl_inr"].sum()) if not trades.empty else 0.0
    net_trade_pnl = float(trades["net_pnl_inr"].sum()) if not trades.empty else 0.0
    turnover = float(
        trades["entry_notional_raw_inr"].sum() + trades["exit_notional_raw_inr"].sum()
    ) if not trades.empty else 0.0
    wins = int((trades["net_pnl_inr"] > 0).sum()) if not trades.empty else 0
    trade_count = len(trades)
    return {
        "portfolio": str(equity["portfolio"].iloc[0]),
        "slippage_scenario": scenario,
        "initial_capital_inr": initial_capital,
        "ending_net_liquidation_equity_inr": ending,
        "ending_gross_equity_inr": gross_ending,
        "net_total_return_pct": 100.0 * (ending / initial_capital - 1.0),
        "gross_total_return_before_costs_pct": 100.0 * (gross_ending / initial_capital - 1.0),
        "net_cagr_pct": 100.0 * cagr,
        "annualized_volatility_pct": 100.0 * vol,
        "sharpe_rf_zero": sharpe,
        "max_drawdown_pct": 100.0 * float(drawdown.min()),
        "average_daily_exposure_pct": float(equity["daily_exposure_pct"].mean()),
        "maximum_daily_exposure_pct": float(equity["daily_exposure_pct"].max()),
        "completed_stock_legs": trade_count,
        "winning_stock_legs_pct": 100.0 * wins / trade_count if trade_count else 0.0,
        "gross_trade_pnl_inr": gross_pnl,
        "net_trade_pnl_inr": net_trade_pnl,
        "total_transaction_and_slippage_costs_inr": total_costs,
        "turnover_inr": turnover,
        "turnover_over_initial_capital": turnover / initial_capital,
        "days": len(equity),
        "exit_liquidity_warnings": int(trades["exit_liquidity_warning"].sum()) if not trades.empty else 0,
    }


def flatten_ohlcv(stock_bars: dict[str, pd.DataFrame], index_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for key, frame in stock_bars.items():
        temp = frame[["session_date", "open", "high", "low", "close", "volume"]].copy()
        temp.insert(0, "instrument_key", key)
        frames.append(temp)
    frames.append(index_df[["instrument_key", "session_date", "open", "high", "low", "close", "volume"]].copy())
    return pd.concat(frames, ignore_index=True).sort_values(["instrument_key", "session_date"])


def main() -> int:
    from backend.app.ml.phase15.validation.real_data_benchmark import (
        discover_protected_files,
        fingerprint,
    )

    print("=" * 100)
    print("PHASE 15 PART 4.2 - REALISTIC OHLC TRADING STRATEGY SIMULATION")
    print("=" * 100)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    protected_files = sorted(discover_protected_files())
    if len(protected_files) != 11:
        raise RuntimeError(f"Expected 11 protected files, found {len(protected_files)}.")
    protected_before = {str(p): fingerprint(p) for p in protected_files}

    config = load_cost_config()
    stock_keys = load_instrument_keys()
    print("[1/7] Loading primary-target OOF predictions...")
    oof = load_oof_predictions()
    selected_signals = {
        model: oof[oof["model"] == model].copy()
        for model in MODELS_TO_SIMULATE
    }
    print(f"Validated OOF rows: {len(oof):,}; dates: {oof['session_date'].nunique()}; stocks: {len(stock_keys)}")

    print("\n[2/7] Fetching historical daily OHLCV (holdout boundary enforced)...")
    stock_bars, index_df = fetch_ohlcv_universe(stock_keys)
    calendar, bars = build_bar_lookup(stock_bars, index_df)
    if max(calendar) != MAX_OHLC_DATE:
        # Holidays may mean last exchange session is earlier than requested end.
        if max(calendar) > MAX_OHLC_DATE:
            raise RuntimeError("OHLC data extends into the reserved holdout.")
    print(f"Common OHLC calendar: {len(calendar)} sessions, {calendar[0]} to {calendar[-1]}")

    print("\n[3/7] Reconciling every selected OOF actual return against OHLC...")
    target_reconciliation = reconcile_oof_targets_with_ohlc(oof, calendar, bars)
    print(json.dumps(target_reconciliation, indent=2))

    signal_dates = sorted(selected_signals["extra_trees"]["session_date"].unique())
    benchmark_signals = selected_signals["extra_trees"][["fold_number", "session_date"]].drop_duplicates()
    if set(signal_dates) != set(selected_signals["random_forest"]["session_date"].unique()):
        raise RuntimeError("Extra Trees and Random Forest signal calendars differ.")

    all_equity, all_trades, all_skipped, metric_rows = [], [], [], []
    scenarios = config.slippage_scenarios_bps
    run_specs = []
    for slip in scenarios:
        scenario_config = replace(config, slippage_bps_per_side=slip)
        scenario = f"slippage_{slip:g}bps_per_side"
        run_specs.extend([
            ("extra_trees_top5", "top_k", selected_signals["extra_trees"], scenario_config, scenario),
            ("random_forest_top5", "top_k", selected_signals["random_forest"], scenario_config, scenario),
            ("equal_weight_20_stock", "equal_weight_20", benchmark_signals, scenario_config, scenario),
        ])
    zero_cost_index = replace(
        config,
        brokerage_per_order_inr=0.0,
        stt_buy_bps=0.0,
        stt_sell_bps=0.0,
        stamp_duty_buy_bps=0.0,
        exchange_transaction_bps_before_2026_03_01=0.0,
        exchange_transaction_bps_from_2026_03_01=0.0,
        sebi_fee_per_crore_inr=0.0,
        ipft_per_crore_inr=0.0,
        dp_charge_per_scrip_per_sell_day_inr=0.0,
        slippage_bps_per_side=0.0,
        max_adv_participation=1.0,
    )
    run_specs.append(("nifty50_price_only_reference", "nifty50_price_only", benchmark_signals, zero_cost_index, "price_only_no_costs"))

    print("\n[4/7] Simulating seven-sleeve portfolios...")
    for number, (name, basket_type, signal_frame, run_config, scenario) in enumerate(run_specs, start=1):
        print(f"[{number:02d}/{len(run_specs)}] {name} | {scenario}")
        eq, trades, skipped = simulate_portfolio(
            portfolio_name=name,
            basket_type=basket_type,
            signals=signal_frame,
            calendar=calendar,
            bars=bars,
            stock_keys=stock_keys,
            config=run_config,
            initial_capital=INITIAL_CAPITAL_INR,
        )
        eq["slippage_scenario"] = scenario
        if not trades.empty:
            trades["slippage_scenario"] = scenario
        if not skipped.empty:
            skipped["slippage_scenario"] = scenario
        all_equity.append(eq)
        all_trades.append(trades)
        all_skipped.append(skipped)
        metric_rows.append(calculate_portfolio_metrics(eq, trades, INITIAL_CAPITAL_INR, scenario))

    print("\n[5/7] Building performance and risk reports...")
    equity_report = pd.concat(all_equity, ignore_index=True)
    trades_report = pd.concat([x for x in all_trades if not x.empty], ignore_index=True) if any(not x.empty for x in all_trades) else pd.DataFrame()
    skipped_report = pd.concat([x for x in all_skipped if not x.empty], ignore_index=True) if any(not x.empty for x in all_skipped) else pd.DataFrame()
    metrics_report = pd.DataFrame(metric_rows)
    if metrics_report.empty:
        raise RuntimeError("No portfolio metrics produced.")

    # Capital accounting must reconcile exactly after all trades have closed.
    for row in metrics_report.itertuples(index=False):
        expected_net_pnl = row.ending_net_liquidation_equity_inr - INITIAL_CAPITAL_INR
        actual_net_pnl = row.net_trade_pnl_inr
        if not np.isclose(expected_net_pnl, actual_net_pnl, rtol=1e-7, atol=0.05):
            raise RuntimeError(
                f"Capital accounting mismatch for {row.portfolio}/{row.slippage_scenario}: "
                f"equity delta={expected_net_pnl:.4f}, trade net PnL={actual_net_pnl:.4f}"
            )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {
        "ohlcv": REPORT_DIR / f"phase15_part4_2_{run_id}_ohlcv.csv",
        "daily_equity": REPORT_DIR / f"phase15_part4_2_{run_id}_daily_equity.csv",
        "trades": REPORT_DIR / f"phase15_part4_2_{run_id}_trades.csv",
        "skipped_signals": REPORT_DIR / f"phase15_part4_2_{run_id}_skipped_signals.csv",
        "portfolio_metrics": REPORT_DIR / f"phase15_part4_2_{run_id}_portfolio_metrics.csv",
        "summary": REPORT_DIR / f"phase15_part4_2_{run_id}_summary.json",
    }
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")

    print("\n[6/7] Saving research reports...")
    flatten_ohlcv(stock_bars, index_df).to_csv(outputs["ohlcv"], index=False)
    equity_report.to_csv(outputs["daily_equity"], index=False)
    trades_report.to_csv(outputs["trades"], index=False)
    skipped_report.to_csv(outputs["skipped_signals"], index=False)
    metrics_report.to_csv(outputs["portfolio_metrics"], index=False)

    summary = {
        "run_id": run_id,
        "phase": 15,
        "part": "4.2",
        "title": "Realistic OHLC Trading Strategy Simulation and Net Performance Evaluation",
        "benchmark_run_id": BENCHMARK_RUN_ID,
        "primary_target": PRIMARY_TARGET,
        "models_simulated": list(MODELS_TO_SIMULATE),
        "strategy": {
            "selection": "top five of 20 stocks, tie-aware fractional allocation at score boundary",
            "signal_timing": "after signal-day close",
            "entry": "next trading session open",
            "exit": "close of seventh holding session",
            "holding_sessions": HOLDING_SESSIONS,
            "sleeves": SLEEVE_COUNT,
            "capital_per_portfolio_inr": INITIAL_CAPITAL_INR,
            "share_quantity": "integer shares for equities; fractional index units only for non-tradable NIFTY price reference",
            "mark_to_market": "daily close; net equity marks open positions after estimated liquidation costs; gross equity adds back realized and estimated costs",
        },
        "cost_profile": asdict(config),
        "cost_source_note": "Editable Upstox NSE equity-delivery profile; confirm against the account's current tariff/contract note. Slippage and ADV cap are modelling assumptions.",
        "data": {
            "requested_start_date": START_DATE,
            "requested_end_date": MAX_OHLC_DATE,
            "actual_first_ohlc_date": calendar[0],
            "actual_last_ohlc_date": calendar[-1],
            "final_holdout_start": FINAL_HOLDOUT_START,
            "final_holdout_used": False,
            "stock_count": len(stock_keys),
            "common_calendar_sessions": len(calendar),
            "nifty_reference": "NSE_INDEX|Nifty 50",
            "calendar_policy": "exact calendar match required; no forward fill or silent intersection",
            "adv": f"{ADV_WINDOW}-session trailing average volume through signal close; cap uses only past/current signal-date volume",
        },
        "actual_target_reconciliation": target_reconciliation,
        "benchmarks": {
            "equal_weight_20_stock": "same seven-sleeve schedule, integer shares and delivery cost model",
            "nifty50_price_only_reference": "same signal/holding schedule; non-tradable index price-only reference with no investable costs",
        },
        "limitations": [
            "Daily OHLC does not reveal the opening/closing auction order book; slippage is an explicit assumption, not observed fills.",
            "ADV20 participation is a lagged liquidity proxy; exit-liquidity warnings flag trades exceeding the configured ratio using pre-exit ADV.",
            "No market impact curve, partial fills, circuit limits, corporate-action cash flows, dividends, or borrow/shorting are modeled.",
            "Results are development-stage OOF simulations, not independent final-holdout evidence or a live-trading recommendation.",
            "Seven-session target labels overlap; daily staggered sleeves are capital-accounted separately.",
        ],
        "integrity": {
            "protected_artifact_count": len(protected_files),
            "protected_artifacts_modified": False,
            "phase9_modified": False,
            "phase14_modified": False,
            "production_lstm_modified": False,
            "v4_candidates_modified": False,
            "models_fitted": False,
            "hyperparameters_changed": False,
            "final_holdout_used": False,
        },
        "capital_accounting": "For every portfolio/scenario, ending net liquidation equity minus initial capital reconciles to completed trade net PnL; gross equity adds back realized and estimated liquidation costs.",
        "reports": {key: path.name for key, path in outputs.items()},
    }
    with outputs["summary"].open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\n[7/7] Verifying protected artifacts...")
    protected_after = {str(p): fingerprint(p) for p in protected_files}
    if protected_before != protected_after:
        raise RuntimeError("Protected production/V4 artifact fingerprints changed.")

    print("\n" + "=" * 100)
    print("PHASE 15 PART 4.2 COMPLETED")
    print("=" * 100)
    print(metrics_report[[
        "portfolio", "slippage_scenario", "net_total_return_pct",
        "max_drawdown_pct", "annualized_volatility_pct", "sharpe_rf_zero",
        "total_transaction_and_slippage_costs_inr", "completed_stock_legs",
    ]].round(4).to_string(index=False))
    print("\nReports:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("\nOOF/ OHLC target reconciliation: PASSED")
    print("Capital accounting: RECONCILED")
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
