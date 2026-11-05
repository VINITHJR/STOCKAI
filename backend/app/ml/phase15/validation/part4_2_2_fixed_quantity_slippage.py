"""Phase 15 Part 4.2.2: fixed-quantity slippage isolation.

Research-only diagnostic. It reads the saved Part 4.2 OHLCV/OOF inputs and
Part 4.2.1 replay reports. The 20 bps Part 4.2.1 replay is the conservative
quantity anchor. The exact same portfolio, signal dates, stock selections,
sleeve IDs and integer share quantities are replayed at 0/5/10/20 bps per side.
Brokerage and date-effective statutory charges remain active in every scenario.

It does not call Upstox, fit models, modify predictions, change source reports,
modify production/V4 artifacts, or access the final holdout.

Run from the StockAI project root:
    python -m backend.app.ml.phase15.validation.part4_2_2_fixed_quantity_slippage
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.phase15.validation import part4_2_1_diagnostics as p421

ROOT = Path("backend/app/ml/phase15")
REPORT_DIR = ROOT / "reports"
PART421_RUN_ID = "20261101_064311"
ANCHOR_SLIPPAGE_BPS = 20.0
SCENARIOS_BPS = (0.0, 5.0, 10.0, 20.0)
PORTFOLIOS = (
    "extra_trees_top5",
    "random_forest_top5",
    "equal_weight_20_integer_redistributed",
)
INITIAL_CAPITAL_INR = p421.INITIAL_CAPITAL_INR
SLEEVE_COUNT = p421.SLEEVE_COUNT
HOLDING_SESSIONS = p421.HOLDING_SESSIONS
FINAL_HOLDOUT_START = p421.FINAL_HOLDOUT_START
REQUIRED_TRADE_COLUMNS = {
    "portfolio", "slippage_bps_per_side", "fold_number", "sleeve_id",
    "instrument_key", "signal_date", "entry_date", "exit_date",
    "holding_sessions", "quantity", "entry_open_raw", "exit_close_raw",
}


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"sha256": digest.hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def schedule_signature(frame: pd.DataFrame) -> str:
    columns = [
        "portfolio", "fold_number", "signal_date", "entry_date", "exit_date",
        "sleeve_id", "instrument_key", "quantity",
    ]
    ordered = frame[columns].sort_values(
        ["portfolio", "signal_date", "entry_date", "sleeve_id", "instrument_key"],
        kind="mergesort",
    )
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _part421_paths() -> dict[str, Path]:
    stem = f"phase15_part4_2_1_{PART421_RUN_ID}_"
    return {
        "trades": REPORT_DIR / f"{stem}trades.csv",
        "daily_equity": REPORT_DIR / f"{stem}daily_equity.csv",
        "portfolio_metrics": REPORT_DIR / f"{stem}portfolio_metrics.csv",
        "summary": REPORT_DIR / f"{stem}summary.json",
    }



def ensure_daily_equity_scenario_labels(daily: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Add scenario labels when the Part 4.2.1 daily writer omitted them.

    Part 4.2.1 appends four complete equity blocks per portfolio in ascending
    scenario order (0, 5, 10, 20 bps), but its daily-equity rows do not contain
    a scenario field. This helper reconstructs labels only if each portfolio
    has four equal-length blocks with exactly matching date sequences.
    """
    result = daily.copy()
    if "slippage_bps_per_side" in result.columns:
        result["slippage_bps_per_side"] = pd.to_numeric(
            result["slippage_bps_per_side"], errors="raise"
        ).astype(float)
        return result

    scenario_values = sorted(
        pd.to_numeric(metrics["slippage_bps_per_side"], errors="raise").unique().tolist()
    )
    expected_scenarios = [float(x) for x in SCENARIOS_BPS]
    if scenario_values != expected_scenarios:
        raise ValueError(
            "Cannot reconstruct Part 4.2.1 daily-equity scenario labels: "
            f"expected {expected_scenarios}, found {scenario_values}."
        )

    result["slippage_bps_per_side"] = np.nan
    for portfolio in PORTFOLIOS:
        indices = result.index[result["portfolio"] == portfolio].tolist()
        if not indices or len(indices) % len(scenario_values) != 0:
            raise ValueError(
                f"Cannot safely assign slippage labels to daily equity for {portfolio}."
            )
        block_size = len(indices) // len(scenario_values)
        baseline_dates = result.loc[indices[:block_size], "session_date"].tolist()
        if not baseline_dates or len(set(baseline_dates)) != len(baseline_dates):
            raise ValueError(f"Part 4.2.1 daily-equity date block is invalid for {portfolio}.")
        for scenario_index, scenario in enumerate(scenario_values):
            block_indices = indices[scenario_index * block_size:(scenario_index + 1) * block_size]
            block_dates = result.loc[block_indices, "session_date"].tolist()
            if block_dates != baseline_dates:
                raise ValueError(
                    f"Part 4.2.1 daily-equity block ordering cannot be verified for {portfolio}."
                )
            result.loc[block_indices, "slippage_bps_per_side"] = float(scenario)

    if result["slippage_bps_per_side"].isna().any():
        raise ValueError("Some Part 4.2.1 daily-equity rows could not be assigned a scenario.")
    return result

def load_part421_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, Path]]:
    paths = _part421_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Part 4.2.1 input missing ({name}): {path}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("run_id") != PART421_RUN_ID:
        raise ValueError("Part 4.2.1 summary run ID mismatch.")
    if summary.get("source_part4_2_run_id") != p421.PART42_RUN_ID:
        raise ValueError("Part 4.2.1 summary references an unexpected Part 4.2 run.")
    if summary.get("source_benchmark_run_id") != p421.BENCHMARK_RUN_ID:
        raise ValueError("Part 4.2.1 summary references an unexpected OOF run.")
    if summary.get("final_holdout_used") is not False:
        raise ValueError("STOP: Part 4.2.1 summary does not explicitly confirm holdout exclusion.")
    if summary.get("protected_artifacts", {}).get("modified") is not False:
        raise ValueError("STOP: Part 4.2.1 summary does not confirm protected artifacts unchanged.")

    trades = pd.read_csv(paths["trades"])
    daily = pd.read_csv(paths["daily_equity"])
    metrics = pd.read_csv(paths["portfolio_metrics"])

    missing = REQUIRED_TRADE_COLUMNS - set(trades.columns)
    if missing:
        raise ValueError(f"Part 4.2.1 trade ledger missing columns: {sorted(missing)}")
    if not {"portfolio", "session_date", "net_liquidation_equity_inr",
            "gross_equity_before_costs_inr"}.issubset(daily.columns):
        raise ValueError("Part 4.2.1 daily equity report has an unexpected schema.")
    if not {"portfolio", "slippage_bps_per_side", "net_total_return_pct",
            "gross_total_return_before_costs_pct", "completed_stock_legs"}.issubset(metrics.columns):
        raise ValueError("Part 4.2.1 portfolio metrics report has an unexpected schema.")

    for col in ("slippage_bps_per_side", "quantity", "fold_number", "sleeve_id", "holding_sessions"):
        trades[col] = pd.to_numeric(trades[col], errors="raise")
    for col in ("entry_open_raw", "exit_close_raw"):
        trades[col] = pd.to_numeric(trades[col], errors="raise")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"], errors="raise").dt.strftime("%Y-%m-%d")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="raise").dt.strftime("%Y-%m-%d")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="raise").dt.strftime("%Y-%m-%d")
    daily["session_date"] = pd.to_datetime(daily["session_date"], errors="raise").dt.strftime("%Y-%m-%d")
    daily = ensure_daily_equity_scenario_labels(daily, metrics)
    for frame in (trades, daily, metrics):
        if "slippage_bps_per_side" in frame:
            frame["slippage_bps_per_side"] = pd.to_numeric(frame["slippage_bps_per_side"], errors="raise").astype(float)
    return trades, daily, metrics, summary, paths


def make_fixed_quantity_schedule(
    source_trades: pd.DataFrame,
    calendar: list[str],
    bars: dict[str, dict[str, dict[str, float]]],
    fee_schedule: p421.FeeSchedule,
) -> pd.DataFrame:
    """Freeze stock selection and quantities from the conservative 20 bps replay."""
    anchor = source_trades[np.isclose(source_trades["slippage_bps_per_side"], ANCHOR_SLIPPAGE_BPS)].copy()
    if anchor.empty:
        raise ValueError("No 20 bps Part 4.2.1 trades available as the fixed-quantity anchor.")
    if set(anchor["portfolio"].unique()) != set(PORTFOLIOS):
        raise ValueError("20 bps anchor does not contain all three expected portfolios.")

    date_to_idx = {d: i for i, d in enumerate(calendar)}
    numeric_quantity = anchor["quantity"].to_numpy(dtype=float)
    if not np.isfinite(numeric_quantity).all() or (numeric_quantity <= 0).any():
        raise ValueError("Anchor trade quantities must be finite and positive.")
    if not np.allclose(numeric_quantity, np.round(numeric_quantity), atol=0, rtol=0):
        raise ValueError("Anchor quantities must be whole shares.")
    if not (anchor["holding_sessions"].astype(int) == HOLDING_SESSIONS).all():
        raise ValueError("Anchor trade ledger contains an unexpected holding duration.")
    if (anchor["signal_date"] >= FINAL_HOLDOUT_START).any() or (anchor["exit_date"] >= FINAL_HOLDOUT_START).any():
        raise ValueError("STOP: fixed quantity schedule reaches the final holdout.")

    anchor["quantity"] = np.round(numeric_quantity).astype(int)
    anchor["fold_number"] = anchor["fold_number"].astype(int)
    anchor["sleeve_id"] = anchor["sleeve_id"].astype(int)

    for row in anchor.itertuples(index=False):
        if row.signal_date not in date_to_idx or row.entry_date not in date_to_idx or row.exit_date not in date_to_idx:
            raise ValueError(f"Anchor trade has a date outside the OHLC calendar: {row.instrument_key}/{row.signal_date}")
        signal_idx = date_to_idx[row.signal_date]
        if date_to_idx[row.entry_date] != signal_idx + 1 or date_to_idx[row.exit_date] != signal_idx + HOLDING_SESSIONS:
            raise ValueError(f"Anchor trade has incorrect entry/exit dates: {row.instrument_key}/{row.signal_date}")
        if row.sleeve_id < 1 or row.sleeve_id > SLEEVE_COUNT:
            raise ValueError(f"Invalid sleeve ID: {row.sleeve_id}")
        raw_open = bars[row.instrument_key][row.entry_date]["open"]
        raw_close = bars[row.instrument_key][row.exit_date]["close"]
        if not np.isclose(float(row.entry_open_raw), raw_open, rtol=1e-10, atol=1e-8):
            raise ValueError(f"Anchor entry OHLC price mismatch: {row.instrument_key}/{row.entry_date}")
        if not np.isclose(float(row.exit_close_raw), raw_close, rtol=1e-10, atol=1e-8):
            raise ValueError(f"Anchor exit OHLC price mismatch: {row.instrument_key}/{row.exit_date}")
        adv = p421._adv20(bars[row.instrument_key], calendar, signal_idx)
        volume_cap = int(math.floor(adv * fee_schedule.max_adv_participation))
        if int(row.quantity) > volume_cap:
            raise ValueError(f"Anchor quantity breaches ADV cap: {row.instrument_key}/{row.signal_date}")

    dup_cols = ["portfolio", "signal_date", "instrument_key"]
    if anchor.duplicated(dup_cols).any():
        raise ValueError("Duplicate stock position in fixed-quantity schedule.")

    event_cols = ["portfolio", "signal_date"]
    for (portfolio, signal_date), group in anchor.groupby(event_cols, sort=False):
        if group["entry_date"].nunique() != 1 or group["exit_date"].nunique() != 1:
            raise ValueError(f"Inconsistent entry/exit dates in {portfolio}/{signal_date} event.")
        if group["fold_number"].nunique() != 1 or group["sleeve_id"].nunique() != 1:
            raise ValueError(f"Inconsistent fold/sleeve within {portfolio}/{signal_date} event.")

    # The same sleeve cannot be occupied by two overlapping holding windows.
    for (portfolio, sleeve_id), group in anchor.groupby(["portfolio", "sleeve_id"], sort=False):
        events = group[["signal_date", "entry_date", "exit_date"]].drop_duplicates().copy()
        events["entry_idx"] = events["entry_date"].map(date_to_idx)
        events["exit_idx"] = events["exit_date"].map(date_to_idx)
        events = events.sort_values("entry_idx")
        previous_exit = -1
        for r in events.itertuples(index=False):
            if int(r.entry_idx) <= previous_exit:
                raise ValueError(f"Overlapping fixed positions in {portfolio}/sleeve {sleeve_id}.")
            previous_exit = int(r.exit_idx)

    anchor["anchor_slippage_bps_per_side"] = ANCHOR_SLIPPAGE_BPS
    anchor["signal_index"] = anchor["signal_date"].map(date_to_idx).astype(int)
    anchor["entry_index"] = anchor["entry_date"].map(date_to_idx).astype(int)
    anchor["exit_index"] = anchor["exit_date"].map(date_to_idx).astype(int)
    anchor = anchor.sort_values(
        ["portfolio", "entry_index", "sleeve_id", "instrument_key"], kind="mergesort"
    ).reset_index(drop=True)
    return anchor


def replay_fixed_schedule(
    portfolio: str,
    fixed_schedule: pd.DataFrame,
    calendar: list[str],
    bars: dict[str, dict[str, dict[str, float]]],
    fee_schedule: p421.FeeSchedule,
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay frozen entries/quantities; only fill prices and charges vary by scenario."""
    if slippage_bps not in SCENARIOS_BPS:
        raise ValueError(f"Unsupported slippage scenario: {slippage_bps}")
    schedule = fixed_schedule[fixed_schedule["portfolio"] == portfolio].copy()
    if schedule.empty:
        raise ValueError(f"No fixed quantity positions for {portfolio}.")

    sleeves = [
        {"sleeve_id": i + 1, "cash": INITIAL_CAPITAL_INR / SLEEVE_COUNT, "positions": []}
        for i in range(SLEEVE_COUNT)
    ]
    sleeve_by_id = {s["sleeve_id"]: s for s in sleeves}
    events: dict[int, list[pd.DataFrame]] = {}
    for (_, _, _), group in schedule.groupby(["entry_index", "signal_date", "sleeve_id"], sort=True):
        entry_idx = int(group["entry_index"].iloc[0])
        events.setdefault(entry_idx, []).append(group.copy())

    first_idx = int(schedule["signal_index"].min())
    last_idx = int(schedule["exit_index"].max())
    slip = float(slippage_bps) / 10_000.0
    realized_costs = 0.0
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []

    for day_idx in range(first_idx, last_idx + 1):
        day = calendar[day_idx]
        # Buy events first, matching the Part 4.2.1 event ordering.
        for event_frame in events.get(day_idx, []):
            sleeve_id = int(event_frame["sleeve_id"].iloc[0])
            sleeve = sleeve_by_id[sleeve_id]
            if sleeve["positions"]:
                raise RuntimeError(f"Frozen sleeve {sleeve_id} is still occupied on {day}.")
            total_cash_out = 0.0
            created: list[dict[str, Any]] = []
            for row in event_frame.itertuples(index=False):
                key = str(row.instrument_key)
                qty = int(row.quantity)
                raw_open = float(bars[key][day]["open"])
                fill_price = raw_open * (1.0 + slip)
                fill_notional = qty * fill_price
                fees = p421.fee_breakdown_21(fill_notional, "BUY", day, fee_schedule)
                cash_out = fill_notional + fees["total_fees_inr"]
                total_cash_out += cash_out
                adv20 = p421._adv20(bars[key], calendar, int(row.signal_index))
                volume_cap = int(math.floor(adv20 * fee_schedule.max_adv_participation))
                if qty > volume_cap:
                    raise RuntimeError(f"Fixed quantity breaches entry ADV cap: {key}/{day}.")
                entry_slippage = qty * (fill_price - raw_open)
                realized_costs += fees["total_fees_inr"] + entry_slippage
                created.append({
                    "portfolio": portfolio,
                    "fold_number": int(row.fold_number),
                    "signal_date": str(row.signal_date),
                    "entry_date": day,
                    "exit_date": str(row.exit_date),
                    "entry_idx": day_idx,
                    "exit_idx": int(row.exit_index),
                    "instrument_key": key,
                    "qty": qty,
                    "entry_open_raw": raw_open,
                    "entry_fill_price": fill_price,
                    "entry_notional_raw": qty * raw_open,
                    "entry_notional_fill": fill_notional,
                    "entry_fees": fees,
                    "entry_slippage": entry_slippage,
                    "entry_cash_outflow": cash_out,
                    "adv20_at_signal": adv20,
                    "sleeve_id": sleeve_id,
                })
                order_rows.append({
                    "portfolio": portfolio, "slippage_bps_per_side": float(slippage_bps),
                    "fold_number": int(row.fold_number), "signal_date": str(row.signal_date),
                    "trade_date": day, "side": "BUY", "instrument_key": key,
                    "quantity": qty, "filled_notional_inr": fill_notional,
                    "slippage_inr": entry_slippage, **fees,
                })
            if total_cash_out > float(sleeve["cash"]) + 1e-6:
                raise RuntimeError(
                    f"Fixed quantities exceed sleeve cash for {portfolio}/{day}/sleeve {sleeve_id}: "
                    f"need={total_cash_out:.4f}, available={sleeve['cash']:.4f}."
                )
            sleeve["cash"] -= total_cash_out
            sleeve["positions"] = created

        # Exit only on the fixed schedule's seventh holding-session close.
        for sleeve in sleeves:
            if not sleeve["positions"]:
                continue
            keep = []
            for position in sleeve["positions"]:
                if int(position["exit_idx"]) != day_idx:
                    keep.append(position)
                    continue
                key = position["instrument_key"]
                qty = int(position["qty"])
                raw_close = float(bars[key][day]["close"])
                exit_fill = raw_close * (1.0 - slip)
                exit_notional = qty * exit_fill
                exit_fees = p421.fee_breakdown_21(exit_notional, "SELL", day, fee_schedule)
                proceeds = exit_notional - exit_fees["total_fees_inr"]
                sleeve["cash"] += proceeds
                exit_slippage = qty * (raw_close - exit_fill)
                gross_pnl = qty * (raw_close - float(position["entry_open_raw"]))
                net_pnl = proceeds - float(position["entry_cash_outflow"])
                total_fees = float(position["entry_fees"]["total_fees_inr"]) + float(exit_fees["total_fees_inr"])
                total_slippage = float(position["entry_slippage"]) + exit_slippage
                realized_costs += exit_fees["total_fees_inr"] + exit_slippage

                prior = max(0, day_idx - p421.ADV_WINDOW)
                exit_adv = float(np.mean([
                    bars[key][calendar[j]]["volume"] for j in range(prior, day_idx)
                ])) if day_idx > prior else float("nan")
                exit_participation = qty / exit_adv if exit_adv > 0 else float("inf")
                trade = {
                    "portfolio": portfolio,
                    "slippage_bps_per_side": float(slippage_bps),
                    "fold_number": int(position["fold_number"]),
                    "sleeve_id": int(position["sleeve_id"]),
                    "instrument_key": key,
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": day,
                    "holding_sessions": int(position["exit_idx"] - position["entry_idx"] + 1),
                    "quantity": qty,
                    "entry_open_raw": float(position["entry_open_raw"]),
                    "entry_fill_price": float(position["entry_fill_price"]),
                    "exit_close_raw": raw_close,
                    "exit_fill_price": exit_fill,
                    "entry_notional_raw_inr": float(position["entry_notional_raw"]),
                    "exit_notional_raw_inr": qty * raw_close,
                    "entry_filled_notional_inr": float(position["entry_notional_fill"]),
                    "exit_filled_notional_inr": exit_notional,
                    "gross_pnl_inr": gross_pnl,
                    "net_pnl_inr": net_pnl,
                    "entry_fees_inr": float(position["entry_fees"]["total_fees_inr"]),
                    "exit_fees_inr": float(exit_fees["total_fees_inr"]),
                    "total_fees_inr": total_fees,
                    "entry_slippage_inr": float(position["entry_slippage"]),
                    "exit_slippage_inr": exit_slippage,
                    "total_slippage_inr": total_slippage,
                    "total_costs_inr": total_fees + total_slippage,
                    "net_return_on_entry_cash_pct": 100.0 * net_pnl / float(position["entry_cash_outflow"]),
                    "adv20_at_signal_shares": float(position["adv20_at_signal"]),
                    "entry_adv_participation_pct": 100.0 * qty / float(position["adv20_at_signal"]) if position["adv20_at_signal"] > 0 else np.nan,
                    "adv20_before_exit_shares": exit_adv,
                    "exit_adv_participation_pct": 100.0 * exit_participation,
                    "exit_liquidity_warning": bool(exit_participation > fee_schedule.max_adv_participation),
                    "anchor_slippage_bps_per_side": ANCHOR_SLIPPAGE_BPS,
                }
                for component in ("brokerage_inr", "exchange_transaction_inr", "ipft_inr", "sebi_fee_inr",
                                  "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr"):
                    trade[f"entry_{component}"] = float(position["entry_fees"][component])
                    trade[f"exit_{component}"] = float(exit_fees[component])
                    trade[f"total_{component}"] = float(position["entry_fees"][component]) + float(exit_fees[component])
                trade_rows.append(trade)
                order_rows.append({
                    "portfolio": portfolio, "slippage_bps_per_side": float(slippage_bps),
                    "fold_number": int(position["fold_number"]), "signal_date": position["signal_date"],
                    "trade_date": day, "side": "SELL", "instrument_key": key,
                    "quantity": qty, "filled_notional_inr": exit_notional,
                    "slippage_inr": exit_slippage, **exit_fees,
                })
            sleeve["positions"] = keep

        cash_total = float(sum(float(s["cash"]) for s in sleeves))
        raw_market_value = 0.0
        estimated_exit_costs = 0.0
        open_count = 0
        for sleeve in sleeves:
            for position in sleeve["positions"]:
                key = position["instrument_key"]
                close = float(bars[key][day]["close"])
                raw_value = int(position["qty"]) * close
                estimated_fill = close * (1.0 - slip)
                estimated_fees = p421.fee_breakdown_21(
                    int(position["qty"]) * estimated_fill, "SELL", day, fee_schedule
                )["total_fees_inr"]
                raw_market_value += raw_value
                estimated_exit_costs += raw_value - (int(position["qty"]) * estimated_fill - estimated_fees)
                open_count += 1
        net_equity = cash_total + raw_market_value - estimated_exit_costs
        gross_equity = net_equity + realized_costs + estimated_exit_costs
        equity_rows.append({
            "portfolio": portfolio,
            "slippage_bps_per_side": float(slippage_bps),
            "session_date": day,
            "cash_inr": cash_total,
            "open_stock_legs": open_count,
            "raw_market_value_inr": raw_market_value,
            "estimated_exit_costs_open_inr": estimated_exit_costs,
            "realized_costs_to_date_inr": realized_costs,
            "gross_equity_before_costs_inr": gross_equity,
            "net_liquidation_equity_inr": net_equity,
            "daily_exposure_pct": 100.0 * raw_market_value / gross_equity if gross_equity > 0 else 0.0,
        })

    leftovers = [p for sleeve in sleeves for p in sleeve["positions"]]
    if leftovers:
        raise RuntimeError(f"Fixed schedule ends with {len(leftovers)} unclosed positions.")
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    orders = pd.DataFrame(order_rows)
    if len(trades) != len(schedule):
        raise RuntimeError(f"Executed {len(trades)} legs but fixed schedule contains {len(schedule)}.")
    if not (trades["holding_sessions"] == HOLDING_SESSIONS).all():
        raise RuntimeError("Fixed replay contains a trade with incorrect holding duration.")
    return equity, trades, orders


def build_fold_report(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    component_columns = [f"total_{c}" for c in (
        "brokerage_inr", "exchange_transaction_inr", "ipft_inr", "sebi_fee_inr",
        "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr",
    )]
    rows = []
    for (portfolio, slip, fold), group in trades.groupby(
        ["portfolio", "slippage_bps_per_side", "fold_number"], sort=True
    ):
        gross = float(group["gross_pnl_inr"].sum())
        net = float(group["net_pnl_inr"].sum())
        fees = float(group["total_fees_inr"].sum())
        slippage = float(group["total_slippage_inr"].sum())
        row = {
            "portfolio": portfolio, "slippage_bps_per_side": float(slip),
            "fold_number": int(fold), "completed_stock_legs": int(len(group)),
            "gross_pnl_inr": gross, "total_fees_inr": fees,
            "total_slippage_inr": slippage, "total_costs_inr": fees + slippage,
            "net_pnl_inr": net,
            "gross_pnl_contribution_pct_initial_capital": 100.0 * gross / INITIAL_CAPITAL_INR,
            "net_pnl_contribution_pct_initial_capital": 100.0 * net / INITIAL_CAPITAL_INR,
        }
        for c in component_columns:
            row[c] = float(group[c].sum())
        rows.append(row)
    return pd.DataFrame(rows)



def compare_anchor_replay(metrics: pd.DataFrame, source_metrics: pd.DataFrame,
                          equity_all: pd.DataFrame, source_equity: pd.DataFrame,
                          trades_all: pd.DataFrame, source_trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "gross_total_return_before_costs_pct", "net_total_return_pct",
        "max_drawdown_pct", "annualized_volatility_pct", "sharpe_rf_zero",
        "total_fees_inr", "total_slippage_inr", "completed_stock_legs",
        "gross_trade_pnl_inr", "net_trade_pnl_inr",
    ]
    for portfolio in PORTFOLIOS:
        new_m = metrics[(metrics.portfolio == portfolio) & np.isclose(metrics.slippage_bps_per_side, ANCHOR_SLIPPAGE_BPS)].iloc[0]
        old_m = source_metrics[(source_metrics.portfolio == portfolio) & np.isclose(source_metrics.slippage_bps_per_side, ANCHOR_SLIPPAGE_BPS)].iloc[0]
        checks = {}
        for col in metric_cols:
            a, b = float(new_m[col]), float(old_m[col])
            checks[col] = abs(a - b)
            if not np.isclose(a, b, rtol=1e-8, atol=0.05):
                raise RuntimeError(f"20 bps anchor metric does not reproduce Part 4.2.1: {portfolio}/{col}: {a} vs {b}")

        new_e = equity_all[(equity_all.portfolio == portfolio) & np.isclose(equity_all.slippage_bps_per_side, ANCHOR_SLIPPAGE_BPS)].sort_values("session_date")
        old_e = source_equity[(source_equity.portfolio == portfolio) & np.isclose(source_equity.slippage_bps_per_side, ANCHOR_SLIPPAGE_BPS)].sort_values("session_date")
        if new_e["session_date"].tolist() != old_e["session_date"].tolist():
            raise RuntimeError(f"20 bps anchor equity calendar differs for {portfolio}.")
        for col in ("net_liquidation_equity_inr", "gross_equity_before_costs_inr"):
            if not np.allclose(new_e[col].to_numpy(float), old_e[col].to_numpy(float), rtol=1e-8, atol=0.05):
                raise RuntimeError(f"20 bps anchor daily equity does not reproduce Part 4.2.1: {portfolio}/{col}.")

        new_t = trades_all[(trades_all.portfolio == portfolio) & np.isclose(trades_all.slippage_bps_per_side, ANCHOR_SLIPPAGE_BPS)]
        old_t = source_trades[(source_trades.portfolio == portfolio) & np.isclose(source_trades.slippage_bps_per_side, ANCHOR_SLIPPAGE_BPS)]
        key = ["signal_date", "entry_date", "exit_date", "sleeve_id", "instrument_key", "quantity"]
        new_keys = new_t[key].copy()
        old_keys = old_t[key].copy()
        for key_frame in (new_keys, old_keys):
            for numeric_key in ("sleeve_id", "quantity"):
                key_frame[numeric_key] = pd.to_numeric(key_frame[numeric_key], errors="raise").astype(int)
            key_frame["signal_date"] = pd.to_datetime(key_frame["signal_date"], errors="raise").dt.strftime("%Y-%m-%d")
            key_frame["entry_date"] = pd.to_datetime(key_frame["entry_date"], errors="raise").dt.strftime("%Y-%m-%d")
            key_frame["exit_date"] = pd.to_datetime(key_frame["exit_date"], errors="raise").dt.strftime("%Y-%m-%d")
        new_keys = new_keys.sort_values(key).reset_index(drop=True)
        old_keys = old_keys.sort_values(key).reset_index(drop=True)
        if not new_keys.equals(old_keys):
            raise RuntimeError(f"20 bps anchor trade selections/quantities differ from Part 4.2.1: {portfolio}.")
        rows.append({
            "portfolio": portfolio, "anchor_slippage_bps_per_side": ANCHOR_SLIPPAGE_BPS,
            "trade_count_matches": True, "trade_selection_quantity_sleeve_matches": True,
            "daily_equity_matches": True, "all_metric_checks_pass": True,
            "maximum_metric_absolute_difference": max(checks.values()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    protected, protected_fingerprint = p421._load_protection_helpers()
    protected_before = {str(path): protected_fingerprint(path) for path in protected}

    oof, ohlcv, part42_summary, part42_paths = p421.load_inputs()
    part421_trades, part421_equity, part421_metrics, part421_summary, part421_paths = load_part421_artifacts()
    input_paths = {**part42_paths, **part421_paths}
    source_before = {str(path): sha256_file(path) for path in input_paths.values()}

    calendar, bars = p421.build_bars(ohlcv)
    fee_schedule = p421.load_fee_schedule()
    target_check = p421.reconcile_targets(oof, calendar, bars)

    schedule = make_fixed_quantity_schedule(part421_trades, calendar, bars, fee_schedule)
    schedule_by_portfolio = {
        portfolio: schedule[schedule.portfolio == portfolio].copy()
        for portfolio in PORTFOLIOS
    }
    schedule_audit_rows = []
    for portfolio, frame in schedule_by_portfolio.items():
        schedule_audit_rows.append({
            "portfolio": portfolio,
            "anchor_slippage_bps_per_side": ANCHOR_SLIPPAGE_BPS,
            "scheduled_stock_legs": len(frame),
            "signal_dates": int(frame.signal_date.nunique()),
            "unique_instruments": int(frame.instrument_key.nunique()),
            "total_shares_all_legs": int(frame.quantity.sum()),
            "schedule_sha256": schedule_signature(frame),
            "excluded_or_unfilled_anchor_legs": "Not present in 20 bps completed-trade anchor; no new scenario-specific additions are allowed.",
        })
    schedule_audit = pd.DataFrame(schedule_audit_rows)

    print("=" * 100)
    print("PHASE 15 PART 4.2.2 - FIXED-QUANTITY SLIPPAGE ISOLATION")
    print("=" * 100)
    print(f"[1/7] Loaded Part 4.2 OOF/OHLC and Part 4.2.1 run {PART421_RUN_ID}.")
    print(f"      OOF rows={len(oof):,}; OHLCV rows={len(ohlcv):,}; 20 bps anchor legs={len(schedule):,}.")
    print("[2/7] Rechecking execution-target/OHLC reconciliation...")
    print(json.dumps(target_check, indent=2))
    print("[3/7] Frozen canonical positions using the 20 bps conservative sizing anchor.")
    for portfolio in PORTFOLIOS:
        print(f"      {portfolio}: {len(schedule_by_portfolio[portfolio]):,} fixed stock legs; "
              f"{schedule_by_portfolio[portfolio].signal_date.nunique():,} signal dates.")

    all_equity: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_orders: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    total_runs = len(PORTFOLIOS) * len(SCENARIOS_BPS)
    run_no = 0
    for slip in SCENARIOS_BPS:
        for portfolio in PORTFOLIOS:
            run_no += 1
            print(f"[{run_no:02d}/{total_runs}] Replaying {portfolio} | {slip:g} bps/side with frozen quantities")
            eq, trades, orders = replay_fixed_schedule(
                portfolio, schedule, calendar, bars, fee_schedule, slip
            )
            all_equity.append(eq)
            all_trades.append(trades)
            all_orders.append(orders)
            metrics = p421._equity_metrics(eq, trades, INITIAL_CAPITAL_INR, portfolio, str(slip))
            if not np.isclose(metrics["ending_net_liquidation_equity_inr"] - INITIAL_CAPITAL_INR,
                              metrics["net_trade_pnl_inr"], rtol=1e-8, atol=0.05):
                raise RuntimeError(f"Net capital accounting failed: {portfolio}/{slip} bps.")
            if not np.isclose(metrics["gross_trade_pnl_inr"] - metrics["net_trade_pnl_inr"],
                              metrics["total_costs_inr"], rtol=1e-8, atol=0.05):
                raise RuntimeError(f"Gross/net/cost accounting failed: {portfolio}/{slip} bps.")
            metrics_rows.append(metrics)

    equity_report = pd.concat(all_equity, ignore_index=True)
    trades_report = pd.concat(all_trades, ignore_index=True)
    orders_report = pd.concat(all_orders, ignore_index=True)
    metrics_report = pd.DataFrame(metrics_rows)

    # Prove the quantity schedule is identical in every scenario and gross P&L
    # is invariant. Only fill prices and costs are allowed to change.
    invariance_rows = []
    for portfolio in PORTFOLIOS:
        base_signature = schedule_signature(schedule_by_portfolio[portfolio])
        gross_values = []
        for slip in SCENARIOS_BPS:
            subset = trades_report[(trades_report.portfolio == portfolio) & np.isclose(trades_report.slippage_bps_per_side, slip)]
            sig = schedule_signature(subset)
            if sig != base_signature:
                raise RuntimeError(f"Selection/quantity/sleeve invariance failed: {portfolio}/{slip} bps.")
            gross = float(subset.gross_pnl_inr.sum())
            gross_values.append(gross)
            invariance_rows.append({
                "portfolio": portfolio, "slippage_bps_per_side": float(slip),
                "completed_stock_legs": len(subset),
                "signal_dates": int(subset.signal_date.nunique()),
                "unique_instruments": int(subset.instrument_key.nunique()),
                "total_shares_all_legs": int(subset.quantity.sum()),
                "schedule_sha256": sig, "same_as_20bps_anchor": True,
                "gross_pnl_inr": gross,
            })
        if not np.allclose(gross_values, gross_values[0], rtol=1e-10, atol=0.01):
            raise RuntimeError(f"Gross P&L is not invariant to slippage: {portfolio}.")
        ordered_net = metrics_report[metrics_report.portfolio == portfolio].sort_values("slippage_bps_per_side")["net_trade_pnl_inr"].to_numpy(float)
        if np.any(np.diff(ordered_net) > 0.05):
            raise RuntimeError(f"Net P&L unexpectedly improves as slippage rises: {portfolio}.")

    invariance_report = pd.DataFrame(invariance_rows)
    fold_report = build_fold_report(trades_report)
    cost_report = p421.aggregate_costs(orders_report)
    anchor_comparison = compare_anchor_replay(
        metrics_report, part421_metrics, equity_report, part421_equity,
        trades_report, part421_trades,
    )

    scenario_rows = []
    for portfolio, group in metrics_report.groupby("portfolio", sort=True):
        group = group.sort_values("slippage_bps_per_side")
        zero = group[np.isclose(group.slippage_bps_per_side, 0.0)].iloc[0]
        for r in group.itertuples(index=False):
            scenario_rows.append({
                "portfolio": portfolio,
                "slippage_bps_per_side": float(r.slippage_bps_per_side),
                "gross_total_return_before_costs_pct": float(r.gross_total_return_before_costs_pct),
                "net_total_return_pct": float(r.net_total_return_pct),
                "net_return_change_vs_zero_slippage_pp": float(r.net_total_return_pct - zero.net_total_return_pct),
                "gross_trade_pnl_inr": float(r.gross_trade_pnl_inr),
                "net_trade_pnl_inr": float(r.net_trade_pnl_inr),
                "total_fees_inr": float(r.total_fees_inr),
                "total_slippage_inr": float(r.total_slippage_inr),
                "completed_stock_legs": int(r.completed_stock_legs),
                "quantity_schedule_invariant": True,
            })
    scenario_report = pd.DataFrame(scenario_rows)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {
        "fixed_quantity_schedule": REPORT_DIR / f"phase15_part4_2_2_{run_id}_fixed_quantity_schedule.csv",
        "daily_equity": REPORT_DIR / f"phase15_part4_2_2_{run_id}_daily_equity.csv",
        "trades": REPORT_DIR / f"phase15_part4_2_2_{run_id}_trades.csv",
        "order_cost_attribution": REPORT_DIR / f"phase15_part4_2_2_{run_id}_order_cost_attribution.csv",
        "scenario_comparison": REPORT_DIR / f"phase15_part4_2_2_{run_id}_scenario_comparison.csv",
        "fold_performance": REPORT_DIR / f"phase15_part4_2_2_{run_id}_fold_performance.csv",
        "quantity_invariance_audit": REPORT_DIR / f"phase15_part4_2_2_{run_id}_quantity_invariance_audit.csv",
        "anchor_reconciliation": REPORT_DIR / f"phase15_part4_2_2_{run_id}_anchor_reconciliation.csv",
        "portfolio_metrics": REPORT_DIR / f"phase15_part4_2_2_{run_id}_portfolio_metrics.csv",
        "summary": REPORT_DIR / f"phase15_part4_2_2_{run_id}_summary.json",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    print("[4/7] Fixed-quantity replays completed: 3 portfolios × 4 slippage scenarios.")
    print("[5/7] Checking 20 bps anchor reproduction, invariant gross P&L and scenario cash feasibility...")
    print("[6/7] Saving new Part 4.2.2 diagnostic reports...")
    schedule.to_csv(outputs["fixed_quantity_schedule"], index=False)
    equity_report.to_csv(outputs["daily_equity"], index=False)
    trades_report.to_csv(outputs["trades"], index=False)
    orders_report_costs = cost_report
    orders_report_costs.to_csv(outputs["order_cost_attribution"], index=False)
    scenario_report.to_csv(outputs["scenario_comparison"], index=False)
    fold_report.to_csv(outputs["fold_performance"], index=False)
    invariance_report.to_csv(outputs["quantity_invariance_audit"], index=False)
    anchor_comparison.to_csv(outputs["anchor_reconciliation"], index=False)
    metrics_report.to_csv(outputs["portfolio_metrics"], index=False)

    source_after = {str(path): sha256_file(path) for path in input_paths.values()}
    protected_after = {str(path): protected_fingerprint(path) for path in protected}
    if source_before != source_after:
        raise RuntimeError("Part 4.2/4.2.1 source reports changed during Part 4.2.2.")
    if protected_before != protected_after:
        raise RuntimeError("Protected production/V4 artifact fingerprints changed during Part 4.2.2.")

    summary = {
        "run_id": run_id,
        "phase": 15,
        "part": "4.2.2",
        "title": "Fixed-Quantity Slippage Isolation and Execution Sensitivity Validation",
        "research_only": True,
        "source_part4_2_run_id": p421.PART42_RUN_ID,
        "source_part4_2_1_run_id": PART421_RUN_ID,
        "source_oof_run_id": p421.BENCHMARK_RUN_ID,
        "quantity_anchor": {
            "slippage_bps_per_side": ANCHOR_SLIPPAGE_BPS,
            "meaning": "Exact successfully executed stock selections, integer quantities and sleeve IDs from the Part 4.2.1 20 bps conservative replay are frozen and reused in every scenario.",
            "scenario_specific_reselection": False,
            "scenario_specific_resizing": False,
            "scenario_specific_redistribution": False,
        },
        "scenarios_bps_per_side": list(SCENARIOS_BPS),
        "brokerage_and_statutory_fees_retained_at_zero_slippage": True,
        "fee_method": "Same date-effective Part 4.2.1 Upstox NSE equity-delivery diagnostic schedule; verify against actual broker contract notes.",
        "execution_model": {
            "entry": "raw next-session open * (1 + slippage_bps/10000)",
            "exit": "raw seventh-holding-session close * (1 - slippage_bps/10000)",
            "holding_sessions": HOLDING_SESSIONS,
            "sleeves": SLEEVE_COUNT,
            "whole_shares": True,
            "mark_to_market": "Daily close with estimated liquidation costs for open positions.",
            "fixed_quantity_rule": "Each portfolio's trade schedule is anchored to its own 20 bps Part 4.2.1 trades; quantity/selection/sleeve/date remain identical at 0, 5, 10 and 20 bps.",
        },
        "target_reconciliation": target_check,
        "invariance_checks": {
            "same_trade_count_all_scenarios": True,
            "same_stock_selection_all_scenarios": True,
            "same_share_quantity_all_scenarios": True,
            "same_sleeve_assignment_all_scenarios": True,
            "gross_pnl_invariant_to_slippage": True,
            "net_pnl_non_increasing_as_slippage_rises": True,
        },
        "anchor_reconciliation_to_part4_2_1_20bps": {
            "passed": True,
            "portfolios_checked": len(anchor_comparison),
            "daily_equity_checked": True,
            "trade_selection_quantity_checked": True,
            "metrics_checked": True,
        },
        "holdout": {
            "final_holdout_start": FINAL_HOLDOUT_START,
            "final_holdout_used": False,
            "maximum_ohlc_date": p421.MAX_ALLOWED_OHLC_DATE,
        },
        "models_fitted": False,
        "predictions_modified": False,
        "hyperparameters_changed": False,
        "upstox_api_called": False,
        "part4_2_and_part4_2_1_sources_modified": False,
        "protected_artifact_count": len(protected),
        "protected_artifacts_modified": False,
        "source_fingerprints_unchanged": True,
        "reports": {name: path.name for name, path in outputs.items()},
        "limitations": [
            "Quantities are anchored to the conservative 20 bps Part 4.2.1 replay; this isolates slippage conditional on that baseline sizing, not the optimal sizing under every scenario.",
            "Selections and quantities that did not execute in the 20 bps anchor remain excluded from all four scenarios.",
            "Scenario fee amounts still vary slightly because statutory charges apply to scenario-specific filled notionals, while the quantity is fixed.",
            "Daily OHLC does not model order-book depth, market impact, partial fills, price bands, corporate actions or dividends.",
            "This is a development-stage replay using previously reviewed OOF signals, not independent final-holdout evidence or a live-trading recommendation.",
        ],
    }
    with outputs["summary"].open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("[7/7] Protected artifacts and all source reports verified unchanged.")
    print("\n" + "=" * 100)
    print("PHASE 15 PART 4.2.2 COMPLETED")
    print("=" * 100)
    print(metrics_report[[
        "portfolio", "slippage_bps_per_side", "gross_total_return_before_costs_pct",
        "net_total_return_pct", "max_drawdown_pct", "total_fees_inr",
        "total_slippage_inr", "completed_stock_legs",
    ]].round(4).to_string(index=False))
    print("\nFixed quantity schedules:", len(schedule), "stock legs")
    print("Quantity invariance audit:", len(invariance_report), "rows")
    print("20 bps anchor reconciliation:", len(anchor_comparison), "portfolios PASSED")
    print("OOF/OHLC reconciliation: PASSED")
    print("Capital accounting: RECONCILED")
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Part 4.2/4.2.1 source reports: UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("\nReports:")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
