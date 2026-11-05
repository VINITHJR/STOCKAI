"""Phase 15 Part 4.2.1: benchmark correction, cost attribution and fold diagnostics.

Research-only follow-up to Part 4.2. Reads the existing OOF predictions and the
OHLCV snapshot produced by Part 4.2. It does not call Upstox, fit models, alter
predictions, change Part 4.2 outputs, or access the final holdout.

Run from the StockAI project root:
    python -m backend.app.ml.phase15.validation.part4_2_1_diagnostics

Outputs are new files under backend/app/ml/phase15/reports/.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("backend/app/ml/phase15")
REPORT_DIR = ROOT / "reports"
COST_SCHEDULE_PATH = ROOT / "validation" / "part4_2_1_cost_schedule.json"

PART42_RUN_ID = "20261101_054819"
BENCHMARK_RUN_ID = "20261031_131021"
PRIMARY_TARGET = "next_open_to_close_return_7s"
FINAL_HOLDOUT_START = "2026-04-17"
MAX_ALLOWED_OHLC_DATE = "2026-04-16"
HOLDING_SESSIONS = 7
TOP_K = 5
SLEEVE_COUNT = 7
INITIAL_CAPITAL_INR = 1_000_000.0
ADV_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252
MODEL_PORTFOLIOS = {
    "extra_trees_top5": "extra_trees",
    "random_forest_top5": "random_forest",
}
SCENARIOS_BPS = (0.0, 5.0, 10.0, 20.0)
OOF_COLUMNS = [
    "fold_number", "model", "instrument_key", "session_date",
    "target_column", "actual_return", "predicted_return",
]
PROTECTED_COUNT = 11
FOLD_EXPECTED = {
    1: {"sessions": 61, "rows": 1220, "start": "2025-01-14", "end": "2025-04-11"},
    2: {"sessions": 61, "rows": 1220, "start": "2025-04-15", "end": "2025-07-10"},
    3: {"sessions": 61, "rows": 1220, "start": "2025-07-11", "end": "2025-10-08"},
    4: {"sessions": 60, "rows": 1200, "start": "2025-10-09", "end": "2026-01-05"},
    5: {"sessions": 60, "rows": 1200, "start": "2026-01-06", "end": "2026-04-06"},
}


@dataclass(frozen=True)
class FeeSchedule:
    brokerage_per_order_inr: float
    stt_buy_bps: float
    stt_sell_bps: float
    stamp_duty_buy_bps: float
    exchange_before_per_crore_inr: float
    ipft_before_per_crore_inr: float
    exchange_from_per_crore_inr: float
    ipft_from_per_crore_inr: float
    sebi_per_crore_inr: float
    dp_per_scrip_sell_day_inr: float
    gst_rate: float
    max_adv_participation: float
    effective_change_date: str

    def validate(self) -> None:
        numeric = [
            self.brokerage_per_order_inr, self.stt_buy_bps, self.stt_sell_bps,
            self.stamp_duty_buy_bps, self.exchange_before_per_crore_inr,
            self.ipft_before_per_crore_inr, self.exchange_from_per_crore_inr,
            self.ipft_from_per_crore_inr, self.sebi_per_crore_inr,
            self.dp_per_scrip_sell_day_inr, self.gst_rate,
            self.max_adv_participation,
        ]
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in numeric):
            raise ValueError("Fee schedule contains a negative or non-finite value.")
        if self.gst_rate > 1:
            raise ValueError("GST rate must be a decimal fraction, e.g. 0.18.")
        if not 0 <= self.max_adv_participation <= 1:
            raise ValueError("ADV participation must be between zero and one.")
        if len(self.effective_change_date) != 10:
            raise ValueError("Invalid fee schedule effective date.")


def load_fee_schedule(path: Path = COST_SCHEDULE_PATH) -> FeeSchedule:
    if not path.is_file():
        raise FileNotFoundError(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    schedule = FeeSchedule(
        brokerage_per_order_inr=float(d["brokerage_per_order_inr"]),
        stt_buy_bps=float(d["stt_buy_bps"]),
        stt_sell_bps=float(d["stt_sell_bps"]),
        stamp_duty_buy_bps=float(d["stamp_duty_buy_bps"]),
        exchange_before_per_crore_inr=float(d["exchange_transaction_inr_per_crore_before_2026_03_01"]),
        ipft_before_per_crore_inr=float(d["ipft_inr_per_crore_before_2026_03_01"]),
        exchange_from_per_crore_inr=float(d["exchange_transaction_inr_per_crore_from_2026_03_01"]),
        ipft_from_per_crore_inr=float(d["ipft_inr_per_crore_from_2026_03_01"]),
        sebi_per_crore_inr=float(d["sebi_fee_inr_per_crore"]),
        dp_per_scrip_sell_day_inr=float(d["dp_charge_per_scrip_per_sell_day_inr"]),
        gst_rate=float(d["gst_rate"]),
        max_adv_participation=float(d["max_adv_participation"]),
        effective_change_date=str(d["effective_change_date"]),
    )
    schedule.validate()
    return schedule


def fee_breakdown_21(notional_inr: float, side: str, trade_date: str,
                     schedule: FeeSchedule) -> dict[str, float]:
    """Return date-effective NSE delivery charges for one executed order.

    Amounts are calculated on filled notional. Exchange and IPFT components
    are split explicitly as per the NSE circular effective 2026-03-01.
    """
    notional = float(notional_inr)
    side = str(side).upper()
    if not math.isfinite(notional) or notional < 0:
        raise ValueError("Notional must be finite and non-negative.")
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL.")
    if len(trade_date) != 10:
        raise ValueError("Trade date must be YYYY-MM-DD.")

    if trade_date < schedule.effective_change_date:
        exchange_per_crore = schedule.exchange_before_per_crore_inr
        ipft_per_crore = schedule.ipft_before_per_crore_inr
    else:
        exchange_per_crore = schedule.exchange_from_per_crore_inr
        ipft_per_crore = schedule.ipft_from_per_crore_inr

    brokerage = schedule.brokerage_per_order_inr if notional > 0 else 0.0
    exchange = notional * exchange_per_crore / 10_000_000.0
    ipft = notional * ipft_per_crore / 10_000_000.0
    sebi = notional * schedule.sebi_per_crore_inr / 10_000_000.0
    stt_bps = schedule.stt_buy_bps if side == "BUY" else schedule.stt_sell_bps
    stt = notional * stt_bps / 10_000.0
    stamp = notional * schedule.stamp_duty_buy_bps / 10_000.0 if side == "BUY" else 0.0
    dp = schedule.dp_per_scrip_sell_day_inr if side == "SELL" and notional > 0 else 0.0
    gst_base = brokerage + exchange + ipft + dp
    gst = gst_base * schedule.gst_rate
    total = brokerage + exchange + ipft + sebi + stt + stamp + dp + gst
    return {
        "brokerage_inr": brokerage,
        "exchange_transaction_inr": exchange,
        "ipft_inr": ipft,
        "sebi_fee_inr": sebi,
        "stt_inr": stt,
        "stamp_duty_inr": stamp,
        "dp_charge_inr": dp,
        "gst_inr": gst,
        "gst_base_inr": gst_base,
        "total_fees_inr": total,
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Protected/input file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    st = path.stat()
    return {"sha256": digest.hexdigest(), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _load_protection_helpers():
    from backend.app.ml.phase15.validation.real_data_benchmark import (
        discover_protected_files, fingerprint,
    )
    protected = sorted(discover_protected_files())
    if len(protected) != PROTECTED_COUNT:
        raise RuntimeError(f"Expected {PROTECTED_COUNT} protected artifacts; found {len(protected)}.")
    return protected, fingerprint


def _part42_paths() -> dict[str, Path]:
    stem = f"phase15_part4_2_{PART42_RUN_ID}_"
    return {
        "ohlcv": REPORT_DIR / f"{stem}ohlcv.csv",
        "daily_equity": REPORT_DIR / f"{stem}daily_equity.csv",
        "trades": REPORT_DIR / f"{stem}trades.csv",
        "skipped_signals": REPORT_DIR / f"{stem}skipped_signals.csv",
        "portfolio_metrics": REPORT_DIR / f"{stem}portfolio_metrics.csv",
        "summary": REPORT_DIR / f"{stem}summary.json",
        "oof": REPORT_DIR / f"phase15_benchmark_{BENCHMARK_RUN_ID}_primary_oof_predictions.csv",
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, Path]]:
    paths = _part42_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Part 4.2 input missing ({name}): {path}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("run_id") != PART42_RUN_ID:
        raise ValueError("Part 4.2 summary run ID does not match configured run.")
    if summary.get("benchmark_run_id") != BENCHMARK_RUN_ID:
        raise ValueError("Part 4.2 summary references an unexpected OOF benchmark run.")
    if summary.get("integrity", {}).get("final_holdout_used") is not False:
        raise ValueError("STOP: Part 4.2 summary does not confirm holdout exclusion.")
    if summary.get("data", {}).get("final_holdout_used") is not False:
        raise ValueError("STOP: Part 4.2 data section does not confirm holdout exclusion.")

    ohlcv = pd.read_csv(paths["ohlcv"])
    required_ohlcv = {"instrument_key", "session_date", "open", "high", "low", "close", "volume"}
    if not required_ohlcv.issubset(ohlcv.columns):
        raise ValueError(f"OHLCV report missing columns: {sorted(required_ohlcv-set(ohlcv.columns))}")
    ohlcv["session_date"] = pd.to_datetime(ohlcv["session_date"], errors="raise").dt.strftime("%Y-%m-%d")
    for c in ("open", "high", "low", "close", "volume"):
        ohlcv[c] = pd.to_numeric(ohlcv[c], errors="raise").astype(float)
    if not np.isfinite(ohlcv[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("OHLCV input contains non-finite values.")
    if ohlcv.duplicated(["instrument_key", "session_date"]).any():
        raise ValueError("Duplicate instrument/date OHLCV rows.")
    if (ohlcv[["open", "high", "low", "close"]] <= 0).any().any() or (ohlcv["volume"] < 0).any():
        raise ValueError("Invalid OHLCV values.")
    if (ohlcv["high"] < ohlcv[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("Invalid OHLC high values.")
    if (ohlcv["low"] > ohlcv[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC low values.")
    if ohlcv["session_date"].max() > MAX_ALLOWED_OHLC_DATE:
        raise ValueError("STOP: OHLCV input reaches the reserved final holdout.")
    if ohlcv["instrument_key"].nunique() != 21:
        raise ValueError(f"Expected 20 stocks plus NIFTY50_INDEX; found {ohlcv['instrument_key'].nunique()} instruments.")
    date_counts = ohlcv.groupby("instrument_key")["session_date"].nunique()
    if date_counts.nunique() != 1:
        raise ValueError("OHLCV instruments do not have a common calendar length.")
    calendars = ohlcv.groupby("instrument_key")["session_date"].apply(lambda s: tuple(sorted(s)))
    first_calendar = calendars.iloc[0]
    if any(cal != first_calendar for cal in calendars):
        raise ValueError("OHLCV instruments do not share an identical calendar.")

    oof = pd.read_csv(paths["oof"], usecols=OOF_COLUMNS)
    oof["fold_number"] = pd.to_numeric(oof["fold_number"], errors="raise").astype(int)
    for c in ("actual_return", "predicted_return"):
        oof[c] = pd.to_numeric(oof[c], errors="raise").astype(float)
    oof["session_date"] = pd.to_datetime(oof["session_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if len(oof) != 30_300 or not np.isfinite(oof[["actual_return", "predicted_return"]].to_numpy()).all():
        raise ValueError("Primary OOF report has unexpected row count or invalid numeric values.")
    if set(oof["target_column"].unique()) != {PRIMARY_TARGET}:
        raise ValueError("OOF report is not the primary execution-aligned target.")
    if set(oof["model"].unique()) != {"zero_return", "train_mean", "ridge", "random_forest", "extra_trees"}:
        raise ValueError("Unexpected OOF model universe.")
    if oof["session_date"].max() >= FINAL_HOLDOUT_START:
        raise ValueError("STOP: OOF predictions overlap final holdout.")
    key_cols = ["fold_number", "model", "instrument_key", "session_date"]
    if oof.duplicated(key_cols).any():
        raise ValueError("Duplicate OOF prediction keys.")
    if set(oof["instrument_key"].unique()) != set(ohlcv.loc[ohlcv.instrument_key != "NIFTY50_INDEX", "instrument_key"].unique()):
        raise ValueError("OOF stock universe and OHLCV stock universe differ.")
    if not set(oof["session_date"].unique()).issubset(set(first_calendar)):
        raise ValueError("Some OOF dates are missing from the saved OHLCV calendar.")
    for fold, spec in FOLD_EXPECTED.items():
        for model in ("zero_return", "train_mean", "ridge", "random_forest", "extra_trees"):
            part = oof[(oof.fold_number == fold) & (oof.model == model)]
            if len(part) != spec["rows"]:
                raise ValueError(f"OOF fold/model row count mismatch: Fold {fold}/{model}.")
            dates = sorted(part.session_date.unique())
            if len(dates) != spec["sessions"] or dates[0] != spec["start"] or dates[-1] != spec["end"]:
                raise ValueError(f"OOF fold/model date boundary mismatch: Fold {fold}/{model}.")
            if not (part.groupby("session_date").size() == 20).all():
                raise ValueError(f"OOF cross-section is not exactly 20 stocks: Fold {fold}/{model}.")
    actual_counts = oof.groupby(["fold_number", "instrument_key", "session_date"])["actual_return"].nunique()
    if (actual_counts > 1).any():
        raise ValueError("OOF actual returns are inconsistent across models.")
    return oof, ohlcv, summary, paths


def build_bars(ohlcv: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    calendar = sorted(ohlcv["session_date"].unique().tolist())
    bars: dict[str, dict[str, dict[str, float]]] = {}
    for key, group in ohlcv.groupby("instrument_key", sort=False):
        bars[str(key)] = {
            r.session_date: {
                "open": float(r.open), "high": float(r.high), "low": float(r.low),
                "close": float(r.close), "volume": float(r.volume),
            }
            for r in group.sort_values("session_date").itertuples(index=False)
        }
        if list(bars[str(key)].keys()) != calendar:
            raise ValueError(f"{key}: date calendar/order mismatch.")
    if "NIFTY50_INDEX" not in bars:
        raise ValueError("NIFTY50_INDEX OHLCV row is missing.")
    stock_keys = sorted(k for k in bars if k != "NIFTY50_INDEX")
    if len(stock_keys) != 20:
        raise ValueError("Expected exactly 20 equity OHLCV instruments.")
    return calendar, bars


def reconcile_targets(oof: pd.DataFrame, calendar: list[str], bars: dict) -> dict[str, Any]:
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    selected = oof[oof.model.isin(["extra_trees", "random_forest"])]
    max_diff = 0.0
    checked = 0
    for r in selected.itertuples(index=False):
        i = date_to_idx[r.session_date]
        entry_i, exit_i = i + 1, i + HOLDING_SESSIONS
        if exit_i >= len(calendar) or calendar[exit_i] >= FINAL_HOLDOUT_START:
            raise ValueError(f"STOP: OOF trade reaches holdout or lacks exit data: {r.session_date}")
        entry_date, exit_date = calendar[entry_i], calendar[exit_i]
        actual = bars[r.instrument_key][exit_date]["close"] / bars[r.instrument_key][entry_date]["open"] - 1.0
        diff = abs(float(r.actual_return) - actual)
        max_diff = max(max_diff, diff)
        if not np.isclose(float(r.actual_return), actual, rtol=1e-8, atol=1e-10):
            raise ValueError(f"OHLC/OOF mismatch {r.instrument_key} {r.session_date}: {diff}")
        checked += 1
    if checked != 12_120:
        raise ValueError(f"Expected 12,120 selected OOF target rows; checked {checked}.")
    return {
        "passed": True,
        "selected_rows_checked": checked,
        "max_absolute_return_difference": max_diff,
        "formula": "close[signal_index+7] / open[signal_index+1] - 1",
    }


def top_k_fractional_weights(scores: np.ndarray, k: int = TOP_K) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < k or not np.isfinite(scores).all():
        raise ValueError("Invalid scores for top-k ranking.")
    if np.unique(scores).size < 2:
        raise ValueError("Constant scores cannot rank stocks.")
    weights = np.zeros(len(scores), dtype=float)
    remaining = float(k)
    for score in np.sort(np.unique(scores))[::-1]:
        mask = scores == score
        n = int(mask.sum())
        alloc = min(1.0, remaining / n)
        weights[mask] = alloc
        remaining -= n * alloc
        if remaining <= 1e-12:
            break
    if not np.isclose(weights.sum(), float(k), atol=1e-10):
        raise RuntimeError("Top-k weights do not sum to k.")
    return weights


def max_affordable_shares_21(budget: float, fill_price: float, volume_cap: int,
                             trade_date: str, schedule: FeeSchedule) -> int:
    if budget <= 0 or fill_price <= 0 or volume_cap <= 0:
        return 0
    lo, hi = 0, min(volume_cap, int(math.floor(budget / fill_price)) + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        notional = mid * fill_price
        fees = fee_breakdown_21(notional, "BUY", trade_date, schedule)["total_fees_inr"]
        if notional + fees <= budget + 1e-8:
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_redistributed_equal_weight(
    candidates: list[dict[str, Any]], sleeve_budget: float,
    trade_date: str, schedule: FeeSchedule,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Remove zero-share names iteratively, then redistribute their slots.

    The per-name budget is recomputed across the remaining eligible names until
    every retained name can buy at least one share within the ADV cap. The
    residual caused by whole-share rounding remains as cash and is reported.
    """
    if not candidates:
        return [], [], 0.0
    active = [dict(c) for c in candidates]
    excluded: list[dict[str, Any]] = []
    while active:
        per_name_budget = sleeve_budget / len(active)
        rejected = []
        for c in active:
            cap = int(c["volume_cap"])
            one_share_notional = float(c["fill_price"])
            one_share_fees = fee_breakdown_21(one_share_notional, "BUY", trade_date, schedule)["total_fees_inr"]
            if cap < 1:
                rejected.append((c, "adv20_cap_below_one_share"))
            elif one_share_notional + one_share_fees > per_name_budget + 1e-8:
                rejected.append((c, "one_share_exceeds_equal_weight_budget"))
        if not rejected:
            break
        rejected_ids = {c["instrument_key"] for c, _ in rejected}
        for c, reason in rejected:
            excluded.append({"instrument_key": c["instrument_key"], "reason": reason,
                             "volume_cap_shares": c["volume_cap"],
                             "one_share_cash_cost_inr": c["fill_price"] + fee_breakdown_21(
                                 c["fill_price"], "BUY", trade_date, schedule)["total_fees_inr"]})
        active = [c for c in active if c["instrument_key"] not in rejected_ids]
    if not active:
        return [], excluded, 0.0
    per_name_budget = sleeve_budget / len(active)
    for c in active:
        c["allocation_weight"] = 1.0 / len(active)
        c["allocation_budget_inr"] = per_name_budget
    return active, excluded, per_name_budget


def _adv20(bars_for_stock: dict[str, dict[str, float]], calendar: list[str], signal_idx: int) -> float:
    start = signal_idx - ADV_WINDOW + 1
    if start < 0:
        raise ValueError("Insufficient history for trailing ADV20.")
    value = float(np.mean([bars_for_stock[calendar[j]]["volume"] for j in range(start, signal_idx + 1)]))
    return value if math.isfinite(value) and value > 0 else 0.0


def _equity_metrics(equity: pd.DataFrame, trades: pd.DataFrame,
                    initial_capital: float, portfolio: str, scenario: str) -> dict[str, Any]:
    net = equity["net_liquidation_equity_inr"].astype(float)
    gross = equity["gross_equity_before_costs_inr"].astype(float)
    rets = net.pct_change().dropna()
    n = max(1, len(equity) - 1)
    ending = float(net.iloc[-1])
    gross_end = float(gross.iloc[-1])
    vol = float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(rets) > 1 else 0.0
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0
    dd = net / net.cummax() - 1.0
    gross_pnl = float(trades["gross_pnl_inr"].sum()) if len(trades) else 0.0
    net_pnl = float(trades["net_pnl_inr"].sum()) if len(trades) else 0.0
    total_fees = float(trades["total_fees_inr"].sum()) if len(trades) else 0.0
    total_slippage = float(trades["total_slippage_inr"].sum()) if len(trades) else 0.0
    turnover = float((trades["entry_notional_raw_inr"].sum() + trades["exit_notional_raw_inr"].sum())) if len(trades) else 0.0
    cagr = (ending / initial_capital) ** (TRADING_DAYS_PER_YEAR / n) - 1.0 if ending > 0 else -1.0
    wins = int((trades["net_pnl_inr"] > 0).sum()) if len(trades) else 0
    return {
        "portfolio": portfolio, "slippage_bps_per_side": float(scenario),
        "initial_capital_inr": initial_capital,
        "ending_net_liquidation_equity_inr": ending,
        "ending_gross_equity_inr": gross_end,
        "net_total_return_pct": 100.0 * (ending / initial_capital - 1.0),
        "gross_total_return_before_costs_pct": 100.0 * (gross_end / initial_capital - 1.0),
        "net_cagr_pct": 100.0 * cagr,
        "annualized_volatility_pct": 100.0 * vol,
        "sharpe_rf_zero": sharpe,
        "max_drawdown_pct": 100.0 * float(dd.min()),
        "average_daily_exposure_pct": float(equity["daily_exposure_pct"].mean()),
        "maximum_daily_exposure_pct": float(equity["daily_exposure_pct"].max()),
        "completed_stock_legs": int(len(trades)),
        "winning_stock_legs_pct": 100.0 * wins / len(trades) if len(trades) else 0.0,
        "gross_trade_pnl_inr": gross_pnl,
        "net_trade_pnl_inr": net_pnl,
        "total_fees_inr": total_fees,
        "total_slippage_inr": total_slippage,
        "total_costs_inr": total_fees + total_slippage,
        "turnover_inr": turnover,
        "days": int(len(equity)),
    }


def simulate_variant(
    portfolio: str, kind: str, signals: pd.DataFrame, calendar: list[str],
    bars: dict[str, dict[str, dict[str, float]]], stock_keys: list[str],
    schedule: FeeSchedule, slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay one strategy; returns equity, trades, skipped and rebalance audit."""
    if kind not in {"top_k", "equal_weight_redistributed"}:
        raise ValueError(f"Unsupported variant kind: {kind}")
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    sleeves = [{"sleeve_id": i + 1, "cash": INITIAL_CAPITAL_INR / SLEEVE_COUNT, "positions": []}
               for i in range(SLEEVE_COUNT)]
    event_schedule: dict[int, list[dict[str, Any]]] = {}
    for sig_date, group in signals.groupby("session_date", sort=True):
        if sig_date not in date_to_idx:
            raise ValueError(f"Signal date absent from OHLC: {sig_date}")
        si = date_to_idx[sig_date]
        ei, xi = si + 1, si + HOLDING_SESSIONS
        if xi >= len(calendar) or calendar[xi] >= FINAL_HOLDOUT_START:
            raise ValueError(f"Signal {sig_date} would enter final holdout or lacks exit data.")
        event_schedule.setdefault(ei, []).append({
            "signal_date": sig_date, "fold_number": int(group["fold_number"].iloc[0]),
            "signal_idx": si, "exit_idx": xi, "group": group.copy(),
        })

    first_idx = min(date_to_idx[d] for d in signals.session_date.unique())
    last_idx = max(e["exit_idx"] for events in event_schedule.values() for e in events)
    slip = slippage_bps / 10_000.0
    realized_costs = 0.0
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    order_cost_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []

    for day_idx in range(first_idx, last_idx + 1):
        day = calendar[day_idx]
        for event in event_schedule.get(day_idx, []):
            benchmark_audit_row = None
            sleeve = next((s for s in sleeves if not s["positions"]), None)
            if sleeve is None:
                skipped_rows.append({"portfolio": portfolio, "signal_date": event["signal_date"],
                                     "entry_date": day, "fold_number": event["fold_number"],
                                     "instrument_key": None, "reason": "no_free_capital_sleeve"})
                continue
            budget = float(sleeve["cash"])
            if kind == "top_k":
                group = event["group"].sort_values("instrument_key").reset_index(drop=True)
                score_values = group["predicted_return"].to_numpy(float)
                if np.unique(score_values).size < 2:
                    skipped_rows.append({"portfolio": portfolio, "signal_date": event["signal_date"],
                                         "entry_date": day, "fold_number": event["fold_number"],
                                         "instrument_key": None,
                                         "reason": "constant_predictions_no_cross_sectional_rank"})
                    continue
                weights = top_k_fractional_weights(score_values, TOP_K)
                candidates = []
                for i, weight in enumerate(weights):
                    if weight <= 0:
                        continue
                    key = str(group.iloc[i]["instrument_key"])
                    raw_open = bars[key][day]["open"]
                    fill = raw_open * (1.0 + slip)
                    adv = _adv20(bars[key], calendar, event["signal_idx"])
                    cap = int(math.floor(adv * schedule.max_adv_participation))
                    candidates.append({"instrument_key": key, "slot_weight": float(weight),
                                       "fill_price": fill, "raw_open": raw_open,
                                       "adv20": adv, "volume_cap": cap})
                excluded = []
                allocation_per_weight = budget / float(TOP_K)
                for c in candidates:
                    c["allocation_budget_inr"] = allocation_per_weight * c["slot_weight"]
                    c["allocation_weight"] = c["slot_weight"] / TOP_K
            else:
                candidates = []
                for key in stock_keys:
                    raw_open = bars[key][day]["open"]
                    fill = raw_open * (1.0 + slip)
                    adv = _adv20(bars[key], calendar, event["signal_idx"])
                    cap = int(math.floor(adv * schedule.max_adv_participation))
                    candidates.append({"instrument_key": key, "slot_weight": 1.0,
                                       "fill_price": fill, "raw_open": raw_open,
                                       "adv20": adv, "volume_cap": cap})
                candidates, excluded, per_name_budget = plan_redistributed_equal_weight(
                    candidates, budget, day, schedule
                )
                for x in excluded:
                    skipped_rows.append({"portfolio": portfolio, "signal_date": event["signal_date"],
                                         "entry_date": day, "fold_number": event["fold_number"],
                                         "instrument_key": x["instrument_key"], "reason": x["reason"],
                                         "adv20_cap_shares": x["volume_cap_shares"],
                                         "one_share_cash_cost_inr": x["one_share_cash_cost_inr"]})
                benchmark_audit_row = {
                    "portfolio": portfolio, "signal_date": event["signal_date"],
                    "entry_date": day, "fold_number": event["fold_number"],
                    "requested_stock_count": len(stock_keys),
                    "eligible_stock_count_after_redistribution": len(candidates),
                    "excluded_stock_count": len(excluded),
                    "per_eligible_stock_budget_inr": per_name_budget,
                    "excluded_instruments": ";".join(x["instrument_key"] for x in excluded),
                    "exclusion_reasons": ";".join(f"{x['instrument_key']}:{x['reason']}" for x in excluded),
                }

            created = []
            total_cash_out = 0.0
            for c in candidates:
                key = c["instrument_key"]
                alloc = float(c["allocation_budget_inr"])
                qty = max_affordable_shares_21(
                    alloc, c["fill_price"], int(c["volume_cap"]), day, schedule
                )
                if qty <= 0:
                    skipped_rows.append({"portfolio": portfolio, "signal_date": event["signal_date"],
                                         "entry_date": day, "fold_number": event["fold_number"],
                                         "instrument_key": key, "reason": "zero_shares_after_allocation",
                                         "adv20_cap_shares": c["volume_cap"]})
                    continue
                fill_notional = qty * c["fill_price"]
                fees = fee_breakdown_21(fill_notional, "BUY", day, schedule)
                cash_out = fill_notional + fees["total_fees_inr"]
                if total_cash_out + cash_out > budget + 1e-6:
                    raise RuntimeError("Planned entry orders exceed sleeve cash.")
                total_cash_out += cash_out
                entry_slip = qty * (c["fill_price"] - c["raw_open"])
                realized_costs += fees["total_fees_inr"] + entry_slip
                pos = {
                    "portfolio": portfolio, "fold_number": event["fold_number"],
                    "signal_date": event["signal_date"], "entry_date": day,
                    "exit_date": calendar[event["exit_idx"]],
                    "entry_idx": day_idx, "exit_idx": event["exit_idx"],
                    "instrument_key": key, "qty": qty,
                    "entry_open_raw": c["raw_open"], "entry_fill_price": c["fill_price"],
                    "entry_notional_raw": qty * c["raw_open"],
                    "entry_notional_fill": fill_notional,
                    "entry_fees": fees, "entry_slippage": entry_slip,
                    "entry_cash_outflow": cash_out, "adv20_at_signal": c["adv20"],
                    "sleeve_id": sleeve["sleeve_id"],
                }
                created.append(pos)
                order_cost_rows.append({
                    "portfolio": portfolio, "slippage_bps_per_side": slippage_bps,
                    "fold_number": event["fold_number"], "signal_date": event["signal_date"],
                    "trade_date": day, "side": "BUY", "instrument_key": key,
                    "quantity": qty, "filled_notional_inr": fill_notional,
                    "slippage_inr": entry_slip, **fees,
                })
            sleeve["cash"] -= total_cash_out
            if created:
                sleeve["positions"] = created
            elif kind == "top_k":
                skipped_rows.append({"portfolio": portfolio, "signal_date": event["signal_date"],
                                     "entry_date": day, "fold_number": event["fold_number"],
                                     "instrument_key": None, "reason": "no_position_filled"})
            if benchmark_audit_row is not None:
                benchmark_audit_row["filled_stock_count"] = len(created)
                benchmark_audit_row["filled_instruments"] = ";".join(p["instrument_key"] for p in created)
                benchmark_audit_row["filled_notional_inr"] = float(sum(p["entry_notional_fill"] for p in created))
                benchmark_audit_row["entry_fees_inr"] = float(sum(p["entry_fees"]["total_fees_inr"] for p in created))
                benchmark_audit_row["unused_sleeve_cash_after_entry_inr"] = float(sleeve["cash"])
                rebalance_rows.append(benchmark_audit_row)

        # Close positions at today's close. Released sleeves can be reused next session.
        for sleeve in sleeves:
            if not sleeve["positions"]:
                continue
            remain = []
            for p in sleeve["positions"]:
                if p["exit_idx"] != day_idx:
                    remain.append(p)
                    continue
                raw_close = bars[p["instrument_key"]][day]["close"]
                exit_fill = raw_close * (1.0 - slip)
                exit_notional = p["qty"] * exit_fill
                exit_fees = fee_breakdown_21(exit_notional, "SELL", day, schedule)
                proceeds = exit_notional - exit_fees["total_fees_inr"]
                sleeve["cash"] += proceeds
                exit_slip = p["qty"] * (raw_close - exit_fill)
                gross_pnl = p["qty"] * (raw_close - p["entry_open_raw"])
                net_pnl = proceeds - p["entry_cash_outflow"]
                total_fees = p["entry_fees"]["total_fees_inr"] + exit_fees["total_fees_inr"]
                total_slip = p["entry_slippage"] + exit_slip
                realized_costs += exit_fees["total_fees_inr"] + exit_slip
                prior = max(0, day_idx - ADV_WINDOW)
                exit_adv = float(np.mean([
                    bars[p["instrument_key"]][calendar[j]]["volume"]
                    for j in range(prior, day_idx)
                ])) if day_idx > prior else float("nan")
                exit_part = p["qty"] / exit_adv if exit_adv > 0 else float("inf")
                trade = {
                    "portfolio": portfolio, "slippage_bps_per_side": slippage_bps,
                    "fold_number": p["fold_number"], "sleeve_id": p["sleeve_id"],
                    "instrument_key": p["instrument_key"], "signal_date": p["signal_date"],
                    "entry_date": p["entry_date"], "exit_date": day,
                    "holding_sessions": p["exit_idx"] - p["entry_idx"] + 1,
                    "quantity": p["qty"], "entry_open_raw": p["entry_open_raw"],
                    "entry_fill_price": p["entry_fill_price"], "exit_close_raw": raw_close,
                    "exit_fill_price": exit_fill,
                    "entry_notional_raw_inr": p["entry_notional_raw"],
                    "exit_notional_raw_inr": p["qty"] * raw_close,
                    "entry_filled_notional_inr": p["entry_notional_fill"],
                    "exit_filled_notional_inr": exit_notional,
                    "gross_pnl_inr": gross_pnl, "net_pnl_inr": net_pnl,
                    "entry_fees_inr": p["entry_fees"]["total_fees_inr"],
                    "exit_fees_inr": exit_fees["total_fees_inr"],
                    "total_fees_inr": total_fees,
                    "entry_slippage_inr": p["entry_slippage"],
                    "exit_slippage_inr": exit_slip,
                    "total_slippage_inr": total_slip,
                    "total_costs_inr": total_fees + total_slip,
                    "net_return_on_entry_cash_pct": 100.0 * net_pnl / p["entry_cash_outflow"],
                    "adv20_at_signal_shares": p["adv20_at_signal"],
                    "entry_adv_participation_pct": 100.0 * p["qty"] / p["adv20_at_signal"] if p["adv20_at_signal"] > 0 else np.nan,
                    "adv20_before_exit_shares": exit_adv,
                    "exit_adv_participation_pct": 100.0 * exit_part,
                    "exit_liquidity_warning": bool(exit_part > schedule.max_adv_participation),
                }
                # Keep individual component costs in the trade ledger as well.
                for comp in ("brokerage_inr", "exchange_transaction_inr", "ipft_inr",
                             "sebi_fee_inr", "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr"):
                    trade[f"entry_{comp}"] = p["entry_fees"][comp]
                    trade[f"exit_{comp}"] = exit_fees[comp]
                    trade[f"total_{comp}"] = p["entry_fees"][comp] + exit_fees[comp]
                trade_rows.append(trade)
                order_cost_rows.append({
                    "portfolio": portfolio, "slippage_bps_per_side": slippage_bps,
                    "fold_number": p["fold_number"], "signal_date": p["signal_date"],
                    "trade_date": day, "side": "SELL", "instrument_key": p["instrument_key"],
                    "quantity": p["qty"], "filled_notional_inr": exit_notional,
                    "slippage_inr": exit_slip, **exit_fees,
                })
            sleeve["positions"] = remain

        cash_total = sum(float(s["cash"]) for s in sleeves)
        raw_market_value = 0.0
        estimated_exit_costs = 0.0
        open_count = 0
        for sleeve in sleeves:
            for p in sleeve["positions"]:
                close = bars[p["instrument_key"]][day]["close"]
                raw_value = p["qty"] * close
                mark_fill = close * (1.0 - slip)
                mark_fees = fee_breakdown_21(p["qty"] * mark_fill, "SELL", day, schedule)["total_fees_inr"]
                raw_market_value += raw_value
                estimated_exit_costs += raw_value - (p["qty"] * mark_fill - mark_fees)
                open_count += 1
        net_equity = cash_total + raw_market_value - estimated_exit_costs
        gross_equity = net_equity + realized_costs + estimated_exit_costs
        equity_rows.append({
            "portfolio": portfolio, "session_date": day,
            "cash_inr": cash_total, "open_stock_legs": open_count,
            "raw_market_value_inr": raw_market_value,
            "estimated_exit_costs_open_inr": estimated_exit_costs,
            "realized_costs_to_date_inr": realized_costs,
            "gross_equity_before_costs_inr": gross_equity,
            "net_liquidation_equity_inr": net_equity,
            "daily_exposure_pct": 100.0 * raw_market_value / gross_equity if gross_equity > 0 else 0.0,
        })

    leftovers = [p for s in sleeves for p in s["positions"]]
    if leftovers:
        raise RuntimeError(f"Simulation finished with {len(leftovers)} unclosed positions.")
    eq = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    skipped = pd.DataFrame(skipped_rows)
    orders = pd.DataFrame(order_cost_rows)
    return eq, trades, skipped, orders, pd.DataFrame(rebalance_rows)


def aggregate_costs(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the full charge stack by portfolio, scenario and order side."""
    if orders.empty:
        return pd.DataFrame()
    components = ["brokerage_inr", "exchange_transaction_inr", "ipft_inr", "sebi_fee_inr",
                  "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr", "total_fees_inr", "slippage_inr"]
    grouped = orders.groupby(["portfolio", "slippage_bps_per_side", "side"], as_index=False)[components].sum()
    return grouped


def fold_attribution(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    cost_cols = [f"total_{x}" for x in ("brokerage_inr", "exchange_transaction_inr", "ipft_inr",
                  "sebi_fee_inr", "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr")]
    rows = []
    for (portfolio, slip, fold), g in trades.groupby(["portfolio", "slippage_bps_per_side", "fold_number"], sort=True):
        gross = float(g["gross_pnl_inr"].sum())
        net = float(g["net_pnl_inr"].sum())
        row = {
            "portfolio": portfolio, "slippage_bps_per_side": float(slip),
            "fold_number": int(fold), "completed_stock_legs": int(len(g)),
            "gross_pnl_inr": gross, "net_pnl_inr": net,
            "entry_fees_inr": float(g["entry_fees_inr"].sum()),
            "exit_fees_inr": float(g["exit_fees_inr"].sum()),
            "total_fees_inr": float(g["total_fees_inr"].sum()),
            "entry_slippage_inr": float(g["entry_slippage_inr"].sum()),
            "exit_slippage_inr": float(g["exit_slippage_inr"].sum()),
            "total_slippage_inr": float(g["total_slippage_inr"].sum()),
            "total_costs_inr": float(g["total_costs_inr"].sum()),
            "gross_pnl_contribution_pct_of_full_initial_capital": 100.0 * gross / initial_capital,
            "net_pnl_contribution_pct_of_full_initial_capital": 100.0 * net / initial_capital,
            "costs_as_pct_of_abs_gross_pnl": 100.0 * float(g["total_costs_inr"].sum()) / abs(gross) if abs(gross) > 1e-12 else np.nan,
        }
        for c in cost_cols:
            row[c] = float(g[c].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _slippage_from_label(label: str) -> float | None:
    text = str(label)
    if text.startswith("slippage_") and text.endswith("bps_per_side"):
        middle = text[len("slippage_"):-len("bps_per_side")]
        try:
            return float(middle)
        except ValueError:
            return None
    return None


def compare_original_equal_weight(old_metrics: pd.DataFrame,
                                  new_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare old non-redistributed benchmark with the corrected integer benchmark."""
    old = old_metrics[old_metrics["portfolio"] == "equal_weight_20_stock"].copy()
    new = new_metrics[new_metrics["portfolio"] == "equal_weight_20_integer_redistributed"].copy()
    old["slippage_bps_per_side"] = old["slippage_scenario"].map(_slippage_from_label)
    old = old.rename(columns={"net_total_return_pct": "old_net_return_pct",
                              "gross_total_return_before_costs_pct": "old_gross_return_pct",
                              "average_daily_exposure_pct": "old_average_exposure_pct",
                              "completed_stock_legs": "old_completed_legs"})
    new = new.rename(columns={"net_total_return_pct": "corrected_net_return_pct",
                              "gross_total_return_before_costs_pct": "corrected_gross_return_pct",
                              "average_daily_exposure_pct": "corrected_average_exposure_pct",
                              "completed_stock_legs": "corrected_completed_legs"})
    merged = old.merge(new, on="slippage_bps_per_side", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["net_return_change_pp"] = merged["corrected_net_return_pct"] - merged["old_net_return_pct"]
    merged["gross_return_change_pp"] = merged["corrected_gross_return_pct"] - merged["old_gross_return_pct"]
    merged["exposure_change_pp"] = merged["corrected_average_exposure_pct"] - merged["old_average_exposure_pct"]
    merged["completed_legs_change"] = merged["corrected_completed_legs"] - merged["old_completed_legs"]
    keep = ["slippage_bps_per_side", "old_net_return_pct", "corrected_net_return_pct",
            "net_return_change_pp", "old_gross_return_pct", "corrected_gross_return_pct",
            "gross_return_change_pp", "old_average_exposure_pct", "corrected_average_exposure_pct",
            "exposure_change_pp", "old_completed_legs", "corrected_completed_legs", "completed_legs_change"]
    return merged[keep]


def compare_original_model_portfolios(old_metrics: pd.DataFrame,
                                      new_metrics: pd.DataFrame) -> pd.DataFrame:
    """Show impact of corrected fee split/replay on ET and RF at original 5/10/20 bps."""
    old = old_metrics[old_metrics["portfolio"].isin(MODEL_PORTFOLIOS)].copy()
    old["slippage_bps_per_side"] = old["slippage_scenario"].map(_slippage_from_label)
    new = new_metrics[new_metrics["portfolio"].isin(MODEL_PORTFOLIOS)].copy()
    old = old.rename(columns={"net_total_return_pct": "old_net_return_pct",
                              "gross_total_return_before_costs_pct": "old_gross_return_pct",
                              "completed_stock_legs": "old_completed_legs"})
    new = new.rename(columns={"net_total_return_pct": "replay_net_return_pct",
                              "gross_total_return_before_costs_pct": "replay_gross_return_pct",
                              "completed_stock_legs": "replay_completed_legs"})
    merged = old.merge(new, on=["portfolio", "slippage_bps_per_side"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["net_return_change_pp"] = merged["replay_net_return_pct"] - merged["old_net_return_pct"]
    merged["gross_return_change_pp"] = merged["replay_gross_return_pct"] - merged["old_gross_return_pct"]
    merged["completed_legs_change"] = merged["replay_completed_legs"] - merged["old_completed_legs"]
    return merged[["portfolio", "slippage_bps_per_side", "old_net_return_pct", "replay_net_return_pct",
                   "net_return_change_pp", "old_gross_return_pct", "replay_gross_return_pct",
                   "gross_return_change_pp", "old_completed_legs", "replay_completed_legs",
                   "completed_legs_change"]]


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    protected, protect_fn = _load_protection_helpers()
    protected_before = {str(p): protect_fn(p) for p in protected}

    oof, ohlcv, old_summary, paths = load_inputs()
    source_inputs_before = {str(p): _fingerprint(p) for p in paths.values()}
    calendar, bars = build_bars(ohlcv)
    stocks = sorted(k for k in bars if k != "NIFTY50_INDEX")
    schedule = load_fee_schedule()
    target_check = reconcile_targets(oof, calendar, bars)

    old_metrics = pd.read_csv(paths["portfolio_metrics"])
    old_skipped = pd.read_csv(paths["skipped_signals"])
    if old_summary.get("run_id") != PART42_RUN_ID:
        raise ValueError("Part 4.2 source run changed during load.")

    print("=" * 100)
    print("PHASE 15 PART 4.2.1 - BENCHMARK, COST AND FOLD DIAGNOSTICS")
    print("=" * 100)
    print(f"[1/7] Loaded Part 4.2 run {PART42_RUN_ID}: {len(ohlcv):,} OHLCV rows, {len(oof):,} OOF rows.")
    print(f"[2/7] OHLC/OOF reconciliation: {target_check['selected_rows_checked']:,} rows; max diff={target_check['max_absolute_return_difference']:.3g}.")
    print("[3/7] Applying date-effective NSE fee/IPFT split and preparing benchmark variants...")

    model_signals = {
        model: oof[oof.model == model].copy()
        for model in ("extra_trees", "random_forest")
    }
    benchmark_signals = oof[oof.model == "extra_trees"][["fold_number", "session_date"]].drop_duplicates()
    if set(model_signals["extra_trees"].session_date.unique()) != set(model_signals["random_forest"].session_date.unique()):
        raise ValueError("Model signal calendars differ.")

    all_equity, all_trades, all_skipped, all_orders, all_audit, metrics = [], [], [], [], [], []
    portfolios = [
        ("extra_trees_top5", "top_k", model_signals["extra_trees"]),
        ("random_forest_top5", "top_k", model_signals["random_forest"]),
        ("equal_weight_20_integer_redistributed", "equal_weight_redistributed", benchmark_signals),
    ]
    total_runs = len(portfolios) * len(SCENARIOS_BPS)
    run_no = 0
    for slip in SCENARIOS_BPS:
        for name, kind, signal_frame in portfolios:
            run_no += 1
            print(f"[{run_no:02d}/{total_runs}] {name} | {slip:g} bps/side")
            eq, tr, sk, orders, audit = simulate_variant(
                name, kind, signal_frame, calendar, bars, stocks, schedule, slip
            )
            if not eq.empty:
                all_equity.append(eq)
            if not tr.empty:
                all_trades.append(tr)
            if not sk.empty:
                all_skipped.append(sk)
            if not orders.empty:
                all_orders.append(orders)
            if not audit.empty:
                all_audit.append(audit)
            m = _equity_metrics(eq, tr, INITIAL_CAPITAL_INR, name, str(slip))
            if not np.isclose(m["ending_net_liquidation_equity_inr"] - INITIAL_CAPITAL_INR,
                              m["net_trade_pnl_inr"], rtol=1e-7, atol=0.05):
                raise RuntimeError(f"Net capital reconciliation failed: {name} at {slip} bps.")
            if not np.isclose(m["gross_trade_pnl_inr"] - m["net_trade_pnl_inr"],
                              m["total_costs_inr"], rtol=1e-7, atol=0.05):
                raise RuntimeError(f"Gross/net/cost reconciliation failed: {name} at {slip} bps.")
            if not np.isclose(m["ending_gross_equity_inr"] - INITIAL_CAPITAL_INR,
                              m["gross_trade_pnl_inr"], rtol=1e-7, atol=0.05):
                raise RuntimeError(f"Gross equity reconciliation failed: {name} at {slip} bps.")
            if not tr.empty and not (tr["holding_sessions"] == HOLDING_SESSIONS).all():
                raise RuntimeError(f"Unexpected holding duration in {name}.")
            metrics.append(m)

    equity_report = pd.concat(all_equity, ignore_index=True)
    trades_report = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    skipped_report = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    orders_report = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    metrics_report = pd.DataFrame(metrics)
    costs_report = aggregate_costs(orders_report)
    fold_report = fold_attribution(trades_report, INITIAL_CAPITAL_INR)
    benchmark_comparison = compare_original_equal_weight(old_metrics, metrics_report)
    model_replay_comparison = compare_original_model_portfolios(old_metrics, metrics_report)

    # Validate detailed cost attribution sums back to completed trades.
    if not trades_report.empty:
        comp_cols = [f"total_{x}" for x in ("brokerage_inr", "exchange_transaction_inr", "ipft_inr",
                     "sebi_fee_inr", "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr")]
        for r in trades_report.itertuples(index=False):
            components = sum(float(getattr(r, c)) for c in comp_cols)
            if not np.isclose(components, float(r.total_fees_inr), rtol=1e-8, atol=0.02):
                raise RuntimeError(f"Fee component reconciliation failed for {r.instrument_key}/{r.signal_date}.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {
        "replayed_ohlc_equity": REPORT_DIR / f"phase15_part4_2_1_{run_id}_daily_equity.csv",
        "replayed_trades": REPORT_DIR / f"phase15_part4_2_1_{run_id}_trades.csv",
        "skipped_signals": REPORT_DIR / f"phase15_part4_2_1_{run_id}_skipped_signals.csv",
        "order_cost_attribution": REPORT_DIR / f"phase15_part4_2_1_{run_id}_order_cost_attribution.csv",
        "cost_summary": REPORT_DIR / f"phase15_part4_2_1_{run_id}_cost_summary.csv",
        "fold_performance": REPORT_DIR / f"phase15_part4_2_1_{run_id}_fold_performance.csv",
        "portfolio_metrics": REPORT_DIR / f"phase15_part4_2_1_{run_id}_portfolio_metrics.csv",
        "benchmark_comparison": REPORT_DIR / f"phase15_part4_2_1_{run_id}_benchmark_comparison.csv",
        "model_replay_comparison": REPORT_DIR / f"phase15_part4_2_1_{run_id}_model_replay_comparison.csv",
        "benchmark_rebalance_audit": REPORT_DIR / f"phase15_part4_2_1_{run_id}_benchmark_rebalance_audit.csv",
        "summary": REPORT_DIR / f"phase15_part4_2_1_{run_id}_summary.json",
    }
    for p in outputs.values():
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite output: {p}")

    print("[4/7] Replays completed: 12 portfolios (ET/RF/redistributed equal-weight × 0/5/10/20 bps).")
    print("[5/7] Reconciling fold, fee and gross/net ledgers...")
    print("[6/7] Saving new Part 4.2.1 diagnostic reports...")
    equity_report.to_csv(outputs["replayed_ohlc_equity"], index=False)
    trades_report.to_csv(outputs["replayed_trades"], index=False)
    skipped_report.to_csv(outputs["skipped_signals"], index=False)
    orders_report.to_csv(outputs["order_cost_attribution"], index=False)
    costs_report.to_csv(outputs["cost_summary"], index=False)
    fold_report.to_csv(outputs["fold_performance"], index=False)
    metrics_report.to_csv(outputs["portfolio_metrics"], index=False)
    benchmark_comparison.to_csv(outputs["benchmark_comparison"], index=False)
    model_replay_comparison.to_csv(outputs["model_replay_comparison"], index=False)
    (pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame()).to_csv(
        outputs["benchmark_rebalance_audit"], index=False
    )

    original_equal_skips = old_skipped[old_skipped.portfolio == "equal_weight_20_stock"]
    corrected_equal_skips = skipped_report[skipped_report.portfolio == "equal_weight_20_integer_redistributed"] if not skipped_report.empty else pd.DataFrame()
    fee_doc = json.loads(COST_SCHEDULE_PATH.read_text(encoding="utf-8"))
    summary = {
        "run_id": run_id,
        "phase": 15,
        "part": "4.2.1",
        "title": "Benchmark Correction, Transaction Cost Attribution and Fold-Level Trading Diagnostics",
        "source_part4_2_run_id": PART42_RUN_ID,
        "source_benchmark_run_id": BENCHMARK_RUN_ID,
        "research_only": True,
        "data_reused_from_saved_part4_2_ohlcv": True,
        "upstox_api_called": False,
        "models_fitted": False,
        "predictions_modified": False,
        "hyperparameters_changed": False,
        "final_holdout_used": False,
        "final_holdout_start": FINAL_HOLDOUT_START,
        "maximum_allowed_ohlc_date": MAX_ALLOWED_OHLC_DATE,
        "target_reconciliation": target_check,
        "model_replay_comparison": "Original Part 4.2 ET/RF performance is compared against the Part 4.2.1 replay at matching 5/10/20 bps; small changes can arise from the corrected date-effective IPFT split and recomputed whole-share affordability.",
        "benchmark_correction": {
            "old_benchmark": "equal_weight_20_stock; integer shares; no redistribution of cash from names that could not buy one share",
            "new_benchmark": "equal_weight_20_integer_redistributed; removes names that cannot buy one share at equal allocation or fail the ADV cap, then recalculates equal allocations across remaining eligible names",
            "whole_shares_only": True,
            "cash_from_excluded_names_redistributed": True,
            "remaining_cash_from_whole_share_rounding_retained": True,
            "old_benchmark_skipped_candidate_rows": int(len(original_equal_skips)),
            "new_benchmark_skipped_candidate_rows": int(len(corrected_equal_skips)),
            "benchmark_is_not_guaranteed_to_hold_all_20_names": True,
            "eligible_name_counts_are_reported": True,
        },
        "fee_method": fee_doc,
        "cost_scenarios_bps_per_side": list(SCENARIOS_BPS),
        "zero_slippage_keeps_brokerage_and_statutory_fees": True,
        "gst_base": "brokerage + exchange transaction charge + DP charge + IPFT; excludes STT, stamp duty and SEBI fee",
        "fold_analysis_method": "Trades are attributed to the fold of their signal_date. Fold P&L is shown as a contribution to the full strategy's original INR 1,000,000 capital, not as a separately compounded fold return.",
        "accounting_checks": {
            "ending_equity_minus_initial_matches_net_trade_pnl": True,
            "gross_trade_pnl_minus_net_trade_pnl_matches_fees_plus_slippage": True,
            "fee_component_sums_match_order_fees": True,
            "all_trades_hold_exactly_seven_sessions": True,
        },
        "source_report_integrity": {
            "source_files_fingerprinted": len(source_inputs_before),
            "source_part4_2_and_oof_reports_modified": False,
        },
        "protected_artifacts": {
            "count": PROTECTED_COUNT,
            "modified": False,
            "phase9_modified": False,
            "phase14_modified": False,
            "production_lstm_modified": False,
            "v4_candidates_modified": False,
        },
        "limitations": [
            "Upstox is an assumed broker profile, not a verified user account tariff; confirm every item against contract notes.",
            "Fees are calculated at unrounded theoretical amounts; broker/exchange contract-note rounding can differ by paise.",
            "The corrected equal-weight benchmark redistributes cash among names meeting one-share affordability and ADV eligibility; excluded names and eligible counts are reported.",
            "The equal-weight integer benchmark may hold fewer than 20 stocks and is labelled accordingly; it is not a strict 20-stock equal-weight exposure on dates with exclusions.",
            "Fold attribution groups complete trades by signal fold; it is not a standalone independently compounded fold portfolio return.",
            "Daily OHLC does not model order-book depth, market impact, partial fills, price bands, corporate actions or dividends.",
            "This uses already-reviewed OOF predictions and is a development-stage diagnostic, not independent holdout evidence or a live-trading recommendation.",
        ],
        "reports": {k: p.name for k, p in outputs.items()},
    }
    with outputs["summary"].open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[7/7] Rechecking protected production/V4 artifact fingerprints...")
    protected_after = {str(p): protect_fn(p) for p in protected}
    if protected_before != protected_after:
        raise RuntimeError("Protected artifact fingerprint changed during Part 4.2.1.")
    source_inputs_after = {str(p): _fingerprint(p) for p in paths.values()}
    if source_inputs_before != source_inputs_after:
        raise RuntimeError("A source Part 4.2/OOF report changed during Part 4.2.1.")

    # Confirm all new files exist and inputs remain unmodified.
    for name, path in outputs.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Output was not written correctly: {name} -> {path}")
    print("\n" + "=" * 100)
    print("PHASE 15 PART 4.2.1 COMPLETED")
    print("=" * 100)
    print(metrics_report[["portfolio", "slippage_bps_per_side", "gross_total_return_before_costs_pct",
                          "net_total_return_pct", "max_drawdown_pct", "total_fees_inr",
                          "total_slippage_inr", "completed_stock_legs"]].round(4).to_string(index=False))
    print("\nFold-level trade cohort attribution:", len(fold_report), "rows")
    print("Detailed order cost records:", len(orders_report), "rows")
    print("OOF/OHLC reconciliation: PASSED")
    print("Fee component reconciliation: PASSED")
    print("Capital accounting: RECONCILED")
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Part 4.2 and OOF source reports: UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("\nReports:")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
