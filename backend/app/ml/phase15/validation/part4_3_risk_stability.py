"""Phase 15 Part 4.3: portfolio risk, drawdown, stability and regime diagnostics.

Research-only, read-only analysis of the saved Part 4.2.2 fixed-quantity
slippage replay. It does not fetch data, fit models, change predictions,
replay trades, write to source reports, alter production/V4 artifacts or use
the reserved final holdout.

The diagnostic measures:
- portfolio drawdown depth, duration and recovery;
- daily return risk (volatility, Sharpe, Sortino, VaR/CVaR, Ulcer Index);
- security-level loss concentration;
- validation-fold trade-cohort and calendar-period performance;
- trade performance and exposure by NIFTY trend, market-risk and VIX regimes;
- exposure when a *past-only, maturity-lagged* rolling measure of ranking
  quality is weak, mixed or positive.

The reliability state is a diagnostic proxy, not a model confidence score or
production risk gate. It uses the previous 20 matured validation-date
observations of daily Spearman IC and top-minus-bottom realised return. The
seven-session maturity lag prevents a signal from using outcomes that would
not yet have completed at the current signal close.

Run from the StockAI project root:
    python -m backend.app.ml.phase15.validation.part4_3_risk_stability
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



ROOT = Path("backend/app/ml/phase15")
REPORT_DIR = ROOT / "reports"
PART422_RUN_ID = "20261101_140850"
PART3_RUN_ID = "20261031_145936"
PART2_RUN_ID = "20261031_144125"
PART3_5_RUN_ID = "20261031_153835"
FINAL_HOLDOUT_START = "2026-04-17"
MAX_ALLOWED_DATE = "2026-04-16"
INITIAL_CAPITAL_INR = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252
HOLDING_SESSIONS = 7
RELIABILITY_LOOKBACK = 20
RELIABILITY_MATURITY_LAG = HOLDING_SESSIONS
SCENARIOS_BPS = (0.0, 5.0, 10.0, 20.0)
PORTFOLIOS = (
    "extra_trees_top5",
    "random_forest_top5",
    "equal_weight_20_integer_redistributed",
)
PORTFOLIO_MODEL = {
    "extra_trees_top5": "extra_trees",
    "random_forest_top5": "random_forest",
}
EXPECTED_FOLD_DATES = {
    1: ("2025-01-14", "2025-04-11", 61),
    2: ("2025-04-15", "2025-07-10", 61),
    3: ("2025-07-11", "2025-10-08", 61),
    4: ("2025-10-09", "2026-01-05", 60),
    5: ("2026-01-06", "2026-04-06", 60),
}


def sha256_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"sha256": digest.hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _part422_paths() -> dict[str, Path]:
    stem = f"phase15_part4_2_2_{PART422_RUN_ID}_"
    return {
        "daily_equity": REPORT_DIR / f"{stem}daily_equity.csv",
        "trades": REPORT_DIR / f"{stem}trades.csv",
        "fixed_quantity_schedule": REPORT_DIR / f"{stem}fixed_quantity_schedule.csv",
        "portfolio_metrics": REPORT_DIR / f"{stem}portfolio_metrics.csv",
        "summary": REPORT_DIR / f"{stem}summary.json",
    }


def _context_paths() -> dict[str, Path]:
    return {
        "part3_daily_discrimination": REPORT_DIR / f"phase15_part3_{PART3_RUN_ID}_daily_discrimination.csv",
        "part2_daily_market_context": REPORT_DIR / f"phase15_part2_{PART2_RUN_ID}_daily_market_context.csv",
        "part2_summary": REPORT_DIR / f"phase15_part2_{PART2_RUN_ID}_summary.json",
        "part3_5_sample_sizes": REPORT_DIR / f"phase15_part3_5_{PART3_5_RUN_ID}_regime_sample_sizes.csv",
        "part3_5_summary": REPORT_DIR / f"phase15_part3_5_{PART3_5_RUN_ID}_summary.json",
    }


def _normalize_date_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"Required date column {column!r} is missing.")
    frame[column] = pd.to_datetime(frame[column], errors="raise").dt.strftime("%Y-%m-%d")


def load_part422_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, Path], pd.DataFrame]:
    paths = _part422_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Part 4.2.2 input missing ({name}): {path}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("run_id") != PART422_RUN_ID or summary.get("part") != "4.2.2":
        raise ValueError("Part 4.2.2 summary run ID or part identifier mismatch.")
    if summary.get("holdout", {}).get("final_holdout_used") is not False:
        raise ValueError("STOP: Part 4.2.2 summary does not confirm final holdout exclusion.")
    if summary.get("protected_artifacts_modified") is not False:
        raise ValueError("STOP: Part 4.2.2 summary does not confirm protected artifacts unchanged.")
    if summary.get("source_fingerprints_unchanged") is not True:
        raise ValueError("STOP: Part 4.2.2 source report fingerprints were not confirmed.")

    daily = pd.read_csv(paths["daily_equity"])
    trades = pd.read_csv(paths["trades"])
    schedule = pd.read_csv(paths["fixed_quantity_schedule"])
    metrics = pd.read_csv(paths["portfolio_metrics"])

    daily_required = {
        "portfolio", "slippage_bps_per_side", "session_date",
        "cash_inr", "open_stock_legs", "raw_market_value_inr",
        "gross_equity_before_costs_inr", "net_liquidation_equity_inr",
        "daily_exposure_pct",
    }
    trade_required = {
        "portfolio", "slippage_bps_per_side", "fold_number", "sleeve_id",
        "instrument_key", "signal_date", "entry_date", "exit_date",
        "holding_sessions", "quantity", "gross_pnl_inr", "net_pnl_inr",
        "total_fees_inr", "total_slippage_inr", "total_costs_inr",
        "net_return_on_entry_cash_pct",
    }
    metric_required = {
        "portfolio", "slippage_bps_per_side", "net_total_return_pct",
        "gross_total_return_before_costs_pct", "max_drawdown_pct",
        "annualized_volatility_pct", "sharpe_rf_zero", "average_daily_exposure_pct", "completed_stock_legs",
    }
    schedule_required = {
        "portfolio", "fold_number", "sleeve_id", "instrument_key",
        "signal_date", "entry_date", "exit_date", "quantity",
    }
    for label, frame, required in (
        ("daily equity", daily, daily_required),
        ("trades", trades, trade_required),
        ("fixed quantity schedule", schedule, schedule_required),
        ("portfolio metrics", metrics, metric_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Part 4.2.2 {label} report missing columns: {sorted(missing)}")

    for frame, date_cols in (
        (daily, ["session_date"]),
        (trades, ["signal_date", "entry_date", "exit_date"]),
        (schedule, ["signal_date", "entry_date", "exit_date"]),
    ):
        for col in date_cols:
            _normalize_date_column(frame, col)

    numeric_daily = [
        "slippage_bps_per_side", "cash_inr", "raw_market_value_inr",
        "gross_equity_before_costs_inr", "net_liquidation_equity_inr",
        "daily_exposure_pct", "open_stock_legs",
    ]
    numeric_trades = [
        "slippage_bps_per_side", "fold_number", "sleeve_id", "holding_sessions",
        "quantity", "gross_pnl_inr", "net_pnl_inr", "total_fees_inr",
        "total_slippage_inr", "total_costs_inr", "net_return_on_entry_cash_pct",
    ]
    for frame, columns in ((daily, numeric_daily), (trades, numeric_trades),
                           (schedule, ["fold_number", "sleeve_id", "quantity"]),
                           (metrics, ["slippage_bps_per_side", "net_total_return_pct", "gross_total_return_before_costs_pct", "max_drawdown_pct", "annualized_volatility_pct", "sharpe_rf_zero", "average_daily_exposure_pct", "completed_stock_legs"])):
        for col in columns:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite numeric values detected in {columns}.")

    if set(daily["portfolio"].unique()) != set(PORTFOLIOS):
        raise ValueError("Unexpected portfolio universe in daily-equity report.")
    if set(trades["portfolio"].unique()) != set(PORTFOLIOS):
        raise ValueError("Unexpected portfolio universe in trade report.")
    if set(daily["slippage_bps_per_side"].unique()) != set(SCENARIOS_BPS):
        raise ValueError("Daily-equity report does not contain exactly the four expected slippage scenarios.")
    if set(trades["slippage_bps_per_side"].unique()) != set(SCENARIOS_BPS):
        raise ValueError("Trade report does not contain exactly the four expected slippage scenarios.")
    if daily.duplicated(["portfolio", "slippage_bps_per_side", "session_date"]).any():
        raise ValueError("Duplicate portfolio/scenario/session rows in daily equity.")
    if trades.duplicated(["portfolio", "slippage_bps_per_side", "signal_date", "instrument_key"]).any():
        raise ValueError("Duplicate portfolio/scenario/signal/instrument rows in trade ledger.")
    if (trades["holding_sessions"].astype(int) != HOLDING_SESSIONS).any():
        raise ValueError("Trade ledger contains a holding period other than seven sessions.")
    if (trades["quantity"] <= 0).any() or not np.allclose(trades["quantity"], np.round(trades["quantity"])):
        raise ValueError("Trade quantities must be positive whole shares.")
    if (trades[["signal_date", "entry_date", "exit_date"]] >= FINAL_HOLDOUT_START).any().any():
        raise ValueError("STOP: Trade ledger contains a date in the reserved holdout.")
    if daily["session_date"].max() > MAX_ALLOWED_DATE or trades["exit_date"].max() > MAX_ALLOWED_DATE:
        raise ValueError("STOP: Part 4.2.2 risk inputs extend beyond the approved pre-holdout date boundary.")

    # Validate scenario calendars: each portfolio must use the same date path
    # at every slippage level. This is needed for apples-to-apples risk metrics.
    if metrics.duplicated(["portfolio", "slippage_bps_per_side"]).any() or len(metrics) != len(PORTFOLIOS) * len(SCENARIOS_BPS):
        raise ValueError("Part 4.2.2 portfolio metrics must contain exactly 12 unique portfolio/scenario rows.")

    for portfolio, group in daily.groupby("portfolio"):
        calendars = [
            tuple(sorted(g["session_date"].tolist()))
            for _, g in group.groupby("slippage_bps_per_side")
        ]
        if len(calendars) != len(SCENARIOS_BPS) or any(c != calendars[0] for c in calendars[1:]):
            raise ValueError(f"Scenario daily-equity calendars differ for {portfolio}.")

    # Reconcile the complete 20 bps execution schedule to the canonical fixed
    # quantity schedule and reconcile every trade ledger to ending portfolio
    # equity before performing risk analysis.
    schedule_key = ["portfolio", "fold_number", "sleeve_id", "instrument_key", "signal_date", "entry_date", "exit_date", "quantity"]
    anchor_trades = trades[np.isclose(trades["slippage_bps_per_side"], 20.0)].copy()
    if set(anchor_trades["portfolio"].unique()) != set(PORTFOLIOS):
        raise ValueError("20 bps trade anchor does not contain all expected portfolios.")
    schedule_checks = []
    for portfolio in PORTFOLIOS:
        left = anchor_trades[anchor_trades["portfolio"] == portfolio][schedule_key].copy()
        right = schedule[schedule["portfolio"] == portfolio][schedule_key].copy()
        for frame in (left, right):
            frame["fold_number"] = frame["fold_number"].astype(int)
            frame["sleeve_id"] = frame["sleeve_id"].astype(int)
            frame["quantity"] = frame["quantity"].astype(int)
        left = left.sort_values(schedule_key).reset_index(drop=True)
        right = right.sort_values(schedule_key).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Part 4.2.2 20 bps trades do not match fixed quantity schedule for {portfolio}.")
        schedule_checks.append({"portfolio": portfolio, "anchor_trade_schedule_matches": True, "stock_legs": len(left)})

    for _, metric_row in metrics.iterrows():
        portfolio = str(metric_row["portfolio"])
        slip = float(metric_row["slippage_bps_per_side"])
        tg = trades[(trades["portfolio"] == portfolio) & np.isclose(trades["slippage_bps_per_side"], slip)]
        dg = daily[(daily["portfolio"] == portfolio) & np.isclose(daily["slippage_bps_per_side"], slip)].sort_values("session_date")
        if len(tg) != int(metric_row["completed_stock_legs"]):
            raise ValueError(f"Trade count does not match portfolio metrics for {portfolio}/{slip} bps.")
        if not np.isclose(float(dg["net_liquidation_equity_inr"].iloc[-1]) - INITIAL_CAPITAL_INR,
                           float(tg["net_pnl_inr"].sum()), rtol=1e-8, atol=0.05):
            raise ValueError(f"Trade net P&L does not reconcile to ending equity for {portfolio}/{slip} bps.")
        if not np.isclose(float(tg["gross_pnl_inr"].sum() - tg["net_pnl_inr"].sum()),
                           float(tg["total_costs_inr"].sum()), rtol=1e-8, atol=0.05):
            raise ValueError(f"Trade gross/net/cost ledger does not reconcile for {portfolio}/{slip} bps.")

    summary["part4_2_2_source_integrity_checks"] = {
        "fixed_quantity_20bps_anchor_matches": True,
        "trade_counts_match_portfolio_metrics": True,
        "net_pnl_matches_ending_equity": True,
        "gross_net_cost_ledger_reconciles": True,
    }
    return daily, trades, metrics, summary, paths, pd.DataFrame(schedule_checks)


def load_regime_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = _context_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required regime source missing ({name}): {path}")

    daily = pd.read_csv(paths["part3_daily_discrimination"])
    market = pd.read_csv(paths["part2_daily_market_context"])
    sample_sizes = pd.read_csv(paths["part3_5_sample_sizes"])
    part2_summary = json.loads(paths["part2_summary"].read_text(encoding="utf-8"))
    part3_5_summary = json.loads(paths["part3_5_summary"].read_text(encoding="utf-8"))

    if part2_summary.get("final_holdout_used") is True or part2_summary.get("holdout_used") is True:
        raise ValueError("STOP: Part 2 source summary indicates final holdout use.")
    if part3_5_summary.get("final_holdout_used") is True or part3_5_summary.get("holdout_used") is True:
        raise ValueError("STOP: Part 3.5 source summary indicates final holdout use.")

    daily_required = {
        "experiment", "fold_number", "model", "session_date",
        "nifty_trend_state", "market_risk_state",
        "cross_sectional_spearman_ic", "top_minus_bottom_actual_return",
    }
    if not daily_required.issubset(daily.columns):
        raise ValueError(f"Part 3 daily discrimination report missing columns: {sorted(daily_required-set(daily.columns))}")
    if not {"session_date", "india_vix_close"}.issubset(market.columns):
        raise ValueError("Part 2 daily market context must contain session_date and india_vix_close.")
    if not {"regime_type", "regime", "validation_dates"}.issubset(sample_sizes.columns):
        raise ValueError("Part 3.5 regime sample-size report has an unexpected schema.")

    _normalize_date_column(daily, "session_date")
    _normalize_date_column(market, "session_date")
    for col in ("fold_number", "cross_sectional_spearman_ic", "top_minus_bottom_actual_return"):
        daily[col] = pd.to_numeric(daily[col], errors="raise")
    market["india_vix_close"] = pd.to_numeric(market["india_vix_close"], errors="raise")
    used_ranking_rows = daily[
        (daily["experiment"].astype(str).str.lower() == "primary")
        & daily["model"].isin(["extra_trees", "random_forest"])
    ]
    if used_ranking_rows.empty or not np.isfinite(
        used_ranking_rows[["cross_sectional_spearman_ic", "top_minus_bottom_actual_return"]].to_numpy(float)
    ).all():
        raise ValueError("Part 3 Extra Trees/Random Forest primary ranking metrics contain missing or non-finite values.")
    if not np.isfinite(market["india_vix_close"].to_numpy(float)).all() or (market["india_vix_close"] <= 0).any():
        raise ValueError("Part 2 VIX observations must be finite and positive.")
    if daily["session_date"].max() > "2026-04-06":
        raise ValueError("Part 3 daily discrimination extends beyond the approved validation boundary.")
    if market["session_date"].max() > MAX_ALLOWED_DATE:
        raise ValueError(
            "STOP: Part 2 daily market context contains dates beyond the approved "
            "pre-holdout boundary. Refusing to inspect or filter a source that includes holdout observations."
        )
    if market.duplicated(["session_date"]).any():
        raise ValueError("Duplicate dates in Part 2 daily market context.")

    return daily, market, sample_sizes, paths


def build_market_regimes(
    part3_daily: pd.DataFrame,
    market: pd.DataFrame,
    sample_sizes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join validated trend/risk states and derive fold-specific elevated VIX.

    VIX q75 uses only the last 120 observations dated on/before each fold's
    training-end date. Validation-date VIX is then compared to that frozen
    threshold. This follows the prior Part 3.5 methodology description.
    """
    primary = part3_daily[part3_daily["experiment"].astype(str).str.lower() == "primary"].copy()
    if primary.empty:
        raise ValueError("No primary experiment rows in Part 3 daily discrimination report.")

    expected_models = {"extra_trees", "random_forest"}
    subset = primary[primary["model"].isin(expected_models)].copy()
    if set(subset["model"].unique()) != expected_models:
        raise ValueError("Part 3 regime source is missing Extra Trees or Random Forest.")

    # Verify the trend/risk regime labels are consistent across model rows.
    regime_consistency = subset.groupby("session_date").agg(
        fold_count=("fold_number", "nunique"),
        nifty_state_count=("nifty_trend_state", "nunique"),
        risk_state_count=("market_risk_state", "nunique"),
        model_count=("model", "nunique"),
    )
    if not (regime_consistency["fold_count"] == 1).all() or not (regime_consistency["nifty_state_count"] == 1).all() or not (regime_consistency["risk_state_count"] == 1).all() or not (regime_consistency["model_count"] == 2).all():
        raise ValueError("Part 3 market-regime labels are not consistent across model rows.")

    states = subset[subset["model"] == "extra_trees"].copy()
    states = states[["session_date", "fold_number", "nifty_trend_state", "market_risk_state"]]
    states["fold_number"] = states["fold_number"].astype(int)
    if states["session_date"].duplicated().any() or len(states) != 303:
        raise ValueError(f"Expected 303 unique primary validation regime dates; got {len(states)}.")

    for fold, (start, end, expected_n) in EXPECTED_FOLD_DATES.items():
        fold_dates = states[states["fold_number"] == fold]["session_date"]
        if len(fold_dates) != expected_n or (fold_dates.min(), fold_dates.max()) != (start, end):
            raise ValueError(f"Part 3 fold {fold} regime dates do not match the frozen validation plan.")

    market_map = market.set_index("session_date")["india_vix_close"]
    states["india_vix_close"] = states["session_date"].map(market_map)
    if states["india_vix_close"].isna().any():
        missing = states.loc[states["india_vix_close"].isna(), "session_date"].head(5).tolist()
        raise ValueError(f"Missing VIX values on validation dates: {missing}")

    threshold_rows = []
    thresholds = {}
    market_sorted = market.sort_values("session_date").copy()

    # Match Part 3.5: use the 120 sessions strictly before
    # each fold's validation start date.
    for fold, (validation_start, validation_end, expected_n) in (
        EXPECTED_FOLD_DATES.items()
    ):
        historical = market_sorted[
            market_sorted["session_date"] < validation_start
        ].copy()

        train = historical.tail(120).copy()

        if len(train) != 120:
            raise ValueError(
                f"Fold {fold}: expected 120 preceding VIX training "
                f"observations, found {len(train)}."
            )

        threshold = float(
            train["india_vix_close"].quantile(0.75)
        )
        thresholds[fold] = threshold

        threshold_rows.append({
            "fold_number": fold,
            "training_end_date": train["session_date"].max(),
            "training_observations": len(train),
            "first_training_date": train["session_date"].min(),
            "last_training_date": train["session_date"].max(),
            "vix_training_q75": threshold,
            "threshold_method": (
                "75th percentile of preceding 120 training-session "
                "VIX closes; no validation observations used"
            ),
        })

    states["vix_training_q75"] = (
        states["fold_number"].map(thresholds)
    )

    if states["vix_training_q75"].isna().any():
        raise ValueError(
            "Missing fold-specific VIX training threshold."
        )

    # Match Part 3.5's strict greater-than boundary.
    states["vix_level_regime"] = np.where(
        states["india_vix_close"] > states["vix_training_q75"],
        "elevated",
        "normal",
    )

    # Reconcile reconstructed q75 classifications to Part 3.5.
    # The saved report contains fold-level rows plus one pooled OOF row.
    vix_rows = sample_sizes[
        (sample_sizes["regime_type"].astype(str).str.strip()
         == "vix_level_vs_training_q75")
        & (sample_sizes["regime"].astype(str).str.strip() == "elevated")
    ].copy()

    actual_by_fold = (
        states.assign(
            _elevated=states["vix_level_regime"].eq("elevated")
        )
        .groupby("fold_number")["_elevated"]
        .sum()
        .astype(int)
        .to_dict()
    )

    # Production Part 3.5 reports distinguish fold rows from pooled OOF.
    # Retain the single-row branch for older isolated unit-test fixtures.
    if {"scope", "fold_number"}.issubset(vix_rows.columns):
        scopes = vix_rows["scope"].astype(str).str.strip().str.lower()

        fold_rows = vix_rows.loc[scopes.eq("fold")].copy()
        pooled_rows = vix_rows.loc[scopes.eq("pooled_oof")].copy()

        if len(pooled_rows) != 1:
            raise ValueError(
                "Part 3.5 must contain exactly one pooled elevated-VIX row; "
                f"found {len(pooled_rows)}."
            )

        if fold_rows.empty:
            raise ValueError(
                "Part 3.5 contains no fold-level elevated-VIX reference rows."
            )

        fold_rows["fold_number"] = pd.to_numeric(
            fold_rows["fold_number"], errors="raise"
        ).astype(int)

        fold_rows["validation_dates"] = pd.to_numeric(
            fold_rows["validation_dates"], errors="raise"
        ).astype(int)

        if fold_rows["fold_number"].duplicated().any():
            raise ValueError(
                "Part 3.5 contains duplicate fold-level elevated-VIX rows."
            )

        expected_by_fold = dict(
            zip(fold_rows["fold_number"], fold_rows["validation_dates"])
        )

        pooled_elevated = int(
            pd.to_numeric(
                pooled_rows["validation_dates"], errors="raise"
            ).iloc[0]
        )

        expected_total = sum(expected_by_fold.values())

        if expected_total != pooled_elevated:
            raise ValueError(
                "Part 3.5 fold-level elevated-VIX counts do not reconcile "
                f"with pooled OOF: folds={expected_total}, "
                f"pooled={pooled_elevated}."
            )

        # Folds with no elevated observations may be absent from the
        # sample-size report. Their expected count is therefore zero.
        for fold in sorted(states["fold_number"].astype(int).unique()):
            expected = int(expected_by_fold.get(fold, 0))
            actual = int(actual_by_fold.get(fold, 0))

            if actual != expected:
                raise ValueError(
                    "Elevated-VIX fold reconciliation failed for "
                    f"fold {fold}: expected {expected}, got {actual}."
                )

        actual_total = sum(actual_by_fold.values())

        if actual_total != pooled_elevated:
            raise ValueError(
                "Reconstructed elevated-VIX total does not reconcile "
                f"with pooled Part 3.5: expected {pooled_elevated}, "
                f"got {actual_total}."
            )

    else:
        # Compatibility path for isolated legacy test fixtures.
        if len(vix_rows) != 1:
            raise ValueError(
                "Legacy Part 3.5 fixture must contain exactly one "
                "elevated-VIX reference row."
            )

        expected_elevated = int(vix_rows["validation_dates"].iloc[0])
        actual_elevated = int(
            states["vix_level_regime"].eq("elevated").sum()
        )

        if actual_elevated != expected_elevated:
            raise ValueError(
                "Reconstructed elevated-VIX labels do not reconcile "
                "with legacy Part 3.5 fixture: "
                f"expected {expected_elevated}, got {actual_elevated}."
            )
    states = states.sort_values("session_date").reset_index(drop=True)
    return states, pd.DataFrame(threshold_rows)


def drawdown_episodes(
    dates: list[str],
    equity: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL_INR,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return drawdown episodes, high-water marks and percentage drawdowns.

    Peaks are tracked causally in one pass. No future equity observations are
    used to determine the peak for an earlier drawdown episode.
    """
    values = np.asarray(equity, dtype=float)
    if len(dates) != len(values) or len(values) == 0:
        raise ValueError("Drawdown dates and equity must be non-empty and equal length.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Equity values must be finite and positive.")
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be finite and positive.")

    hwm = np.maximum.accumulate(np.maximum(values, initial_capital))
    dd = values / hwm - 1.0
    episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    running_peak = float(initial_capital)
    running_peak_date = dates[0]
    running_peak_index = 0
    tolerance = max(0.01, initial_capital * 1e-10)

    for i, (day, value_raw) in enumerate(zip(dates, values)):
        value = float(value_raw)
        if active is not None:
            if value >= float(active["peak_equity_inr"]) - tolerance:
                episodes.append({
                    **active,
                    "recovery_date": day,
                    "recovered": True,
                    "underwater_sessions": int(i - active["start_index"]),
                    "peak_to_recovery_sessions": int(i - active["peak_index"]),
                    "calendar_days_to_recovery": int((pd.Timestamp(day) - pd.Timestamp(active["peak_date"])).days),
                    "end_date": day,
                    "status": "RECOVERED",
                })
                active = None
                if value > running_peak + tolerance:
                    running_peak = value
                    running_peak_date = day
                    running_peak_index = i
                continue

            current_dd = value / float(active["peak_equity_inr"]) - 1.0
            active["underwater_sessions"] += 1
            if current_dd < float(active["maximum_drawdown_pct"]) / 100.0:
                active["maximum_drawdown_pct"] = 100.0 * current_dd
                active["trough_date"] = day
                active["trough_equity_inr"] = value
            continue

        if value > running_peak + tolerance:
            running_peak = value
            running_peak_date = day
            running_peak_index = i
            continue

        if value < running_peak - tolerance:
            current_dd = value / running_peak - 1.0
            active = {
                "peak_date": running_peak_date,
                "peak_equity_inr": float(running_peak),
                "start_date": day,
                "start_index": i,
                "peak_index": running_peak_index,
                "trough_date": day,
                "trough_equity_inr": value,
                "maximum_drawdown_pct": 100.0 * current_dd,
                "underwater_sessions": 1,
            }

    if active is not None:
        episodes.append({
            **active,
            "recovery_date": None,
            "recovered": False,
            "underwater_sessions": int(len(values) - active["start_index"]),
            "peak_to_recovery_sessions": None,
            "calendar_days_to_recovery": None,
            "end_date": dates[-1],
            "status": "UNRECOVERED_AT_END",
        })
    return pd.DataFrame(episodes), hwm, dd


def calculate_daily_risk_metrics(equity_frame: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL_INR) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = equity_frame.sort_values("session_date").reset_index(drop=True).copy()
    dates = frame["session_date"].tolist()
    equity = frame["net_liquidation_equity_inr"].to_numpy(dtype=float)
    if not np.isclose(equity[0], initial_capital, rtol=1e-8, atol=0.05):
        raise ValueError(f"First daily-equity row does not start at initial capital: {equity[0]:.2f}.")

    prev = np.concatenate(([initial_capital], equity[:-1]))
    returns = equity / prev - 1.0
    # Match the existing Part 4.2.2 convention for risk statistics:
    # pct_change().dropna() excludes the initial zero-return seed observation.
    returns_for_stats = returns[1:] if len(returns) > 1 else np.asarray([], dtype=float)
    episodes, hwm, dd = drawdown_episodes(dates, equity, initial_capital)
    returns_series = pd.Series(returns_for_stats, dtype=float)
    q05 = float(returns_series.quantile(0.05)) if len(returns_series) else np.nan
    q01 = float(returns_series.quantile(0.01)) if len(returns_series) else np.nan
    cvar05 = float(returns_series[returns_series <= q05].mean()) if len(returns_series) else np.nan
    cvar01 = float(returns_series[returns_series <= q01].mean()) if len(returns_series) else np.nan
    mean_ret = float(returns_series.mean()) if len(returns_series) else np.nan
    std_ret = float(returns_series.std(ddof=1)) if len(returns_series) > 1 else 0.0
    downside = np.minimum(returns_for_stats, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    n_periods = max(1, len(equity) - 1)
    ending = float(equity[-1])
    cagr = (ending / initial_capital) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1.0 if ending > 0 else -1.0
    max_dd = float(np.min(dd))
    underwater_sessions = np.where(dd < -1e-10, 1, 0)
    longest_underwater = 0
    current = 0
    for x in underwater_sessions:
        current = current + 1 if x else 0
        longest_underwater = max(longest_underwater, current)

    recovered = episodes[episodes["recovered"] == True] if not episodes.empty else pd.DataFrame()
    unrecovered = episodes[episodes["recovered"] == False] if not episodes.empty else pd.DataFrame()
    risk = {
        "initial_capital_inr": float(initial_capital),
        "ending_net_equity_inr": ending,
        "net_total_return_pct": 100.0 * (ending / initial_capital - 1.0),
        "gross_total_return_before_costs_pct": 100.0 * (float(frame["gross_equity_before_costs_inr"].iloc[-1]) / initial_capital - 1.0),
        "net_cagr_pct": 100.0 * cagr,
        "daily_observations": int(len(frame)),
        "daily_return_observations": int(len(returns_for_stats)),
        "annualized_volatility_pct": 100.0 * std_ret * math.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe_rf_zero": mean_ret / std_ret * math.sqrt(TRADING_DAYS_PER_YEAR) if std_ret > 0 else np.nan,
        "sortino_rf_zero": mean_ret / downside_dev * math.sqrt(TRADING_DAYS_PER_YEAR) if downside_dev > 0 else np.nan,
        "max_drawdown_pct": 100.0 * max_dd,
        "ulcer_index_pct": 100.0 * float(np.sqrt(np.mean(np.square(dd)))),
        "daily_return_5th_percentile_pct": 100.0 * q05,
        "daily_cvar_95_loss_pct": -100.0 * cvar05,
        "daily_return_1st_percentile_pct": 100.0 * q01,
        "daily_cvar_99_loss_pct": -100.0 * cvar01,
        "worst_daily_return_pct": 100.0 * float(np.min(returns_for_stats)) if len(returns_for_stats) else np.nan,
        "best_daily_return_pct": 100.0 * float(np.max(returns_for_stats)) if len(returns_for_stats) else np.nan,
        "calmar_ratio": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "average_daily_exposure_pct": float(frame["daily_exposure_pct"].mean()),
        "median_daily_exposure_pct": float(frame["daily_exposure_pct"].median()),
        "maximum_daily_exposure_pct": float(frame["daily_exposure_pct"].max()),
        "days_exposure_above_50pct": int((frame["daily_exposure_pct"] > 50.0).sum()),
        "days_exposure_above_75pct": int((frame["daily_exposure_pct"] > 75.0).sum()),
        "recovered_drawdown_episodes": int(len(recovered)),
        "unrecovered_drawdown_episodes_at_end": int(len(unrecovered)),
        "longest_underwater_sessions": int(longest_underwater),
        "longest_recovered_peak_to_recovery_sessions": int(recovered["peak_to_recovery_sessions"].max()) if not recovered.empty else 0,
        "max_observed_drawdown_episode_sessions": int(episodes["underwater_sessions"].max()) if not episodes.empty else 0,
    }
    daily_risk = frame[["portfolio", "slippage_bps_per_side", "session_date", "cash_inr", "open_stock_legs", "raw_market_value_inr", "gross_equity_before_costs_inr", "net_liquidation_equity_inr", "daily_exposure_pct"]].copy()
    daily_risk["daily_return_pct"] = 100.0 * returns
    daily_risk["high_water_mark_inr"] = hwm
    daily_risk["drawdown_pct"] = 100.0 * dd
    daily_risk["underwater"] = dd < -1e-10
    return risk, daily_risk, episodes


def build_stock_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["portfolio", "slippage_bps_per_side"]
    for (portfolio, slip), group in trades.groupby(keys, sort=True):
        total_negative = float((-group.loc[group["net_pnl_inr"] < 0, "net_pnl_inr"]).sum())
        stock = group.groupby("instrument_key", as_index=False).agg(
            completed_stock_legs=("net_pnl_inr", "size"),
            signal_dates=("signal_date", "nunique"),
            gross_pnl_inr=("gross_pnl_inr", "sum"),
            net_pnl_inr=("net_pnl_inr", "sum"),
            fees_inr=("total_fees_inr", "sum"),
            slippage_inr=("total_slippage_inr", "sum"),
            average_trade_net_return_pct=("net_return_on_entry_cash_pct", "mean"),
            median_trade_net_return_pct=("net_return_on_entry_cash_pct", "median"),
            winning_legs_pct=("net_pnl_inr", lambda s: 100.0 * float((s > 0).mean())),
        )
        negative_by_stock = group.assign(
            negative_loss_inr=(-group["net_pnl_inr"].clip(upper=0.0))
        ).groupby("instrument_key")["negative_loss_inr"].sum()
        stock["negative_trade_loss_inr"] = stock["instrument_key"].map(negative_by_stock).fillna(0.0)
        stock["share_of_all_negative_trade_loss_pct"] = (
            100.0 * stock["negative_trade_loss_inr"] / total_negative if total_negative > 0 else 0.0
        )
        shares = stock["share_of_all_negative_trade_loss_pct"].to_numpy(float) / 100.0
        hhi = float(np.square(shares).sum())
        stock["portfolio_stock_loss_hhi"] = hhi
        stock["effective_number_of_loss_contributors"] = 1.0 / hhi if hhi > 0 else np.nan
        stock["portfolio"] = portfolio
        stock["slippage_bps_per_side"] = float(slip)
        rows.append(stock)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_fold_reports(trades: pd.DataFrame, daily_risk: pd.DataFrame, regime_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_rows = []
    for (portfolio, slip, fold), group in trades.groupby(["portfolio", "slippage_bps_per_side", "fold_number"], sort=True):
        trade_rows.append({
            "portfolio": portfolio,
            "slippage_bps_per_side": float(slip),
            "fold_number": int(fold),
            "fold_type": "signal_cohort_trade_attribution",
            "completed_stock_legs": int(len(group)),
            "signal_dates": int(group["signal_date"].nunique()),
            "unique_instruments": int(group["instrument_key"].nunique()),
            "gross_pnl_inr": float(group["gross_pnl_inr"].sum()),
            "net_pnl_inr": float(group["net_pnl_inr"].sum()),
            "fees_inr": float(group["total_fees_inr"].sum()),
            "slippage_inr": float(group["total_slippage_inr"].sum()),
            "net_pnl_contribution_pct_initial_capital": 100.0 * float(group["net_pnl_inr"].sum()) / INITIAL_CAPITAL_INR,
            "winning_legs_pct": 100.0 * float((group["net_pnl_inr"] > 0).mean()),
            "average_leg_net_return_pct": float(group["net_return_on_entry_cash_pct"].mean()),
            "median_leg_net_return_pct": float(group["net_return_on_entry_cash_pct"].median()),
            "worst_stock_leg_pnl_inr": float(group["net_pnl_inr"].min()),
            "best_stock_leg_pnl_inr": float(group["net_pnl_inr"].max()),
        })
    trade_fold = pd.DataFrame(trade_rows)

    # Calendar-period daily risk is a separate view. Dates inherit the most
    # recent validation-fold label. Positions can cross fold boundaries, so
    # this is explicitly not the same as signal-cohort attribution above.
    fmap = regime_map[["session_date", "fold_number"]].drop_duplicates().copy()
    fmap["_date_key"] = pd.to_datetime(fmap["session_date"], errors="raise")
    fmap = fmap.sort_values("_date_key")
    daily_parts = []
    for (portfolio, slip), group in daily_risk.groupby(["portfolio", "slippage_bps_per_side"], sort=True):
        g = group.sort_values("session_date").copy()
        g["_date_key"] = pd.to_datetime(g["session_date"], errors="raise")
        g = pd.merge_asof(
            g.sort_values("_date_key"),
            fmap[["_date_key", "fold_number"]].sort_values("_date_key"),
            on="_date_key", direction="backward",
        )
        for fold, fg in g.dropna(subset=["fold_number"]).groupby("fold_number", sort=True):
            rets = fg["daily_return_pct"].to_numpy(float) / 100.0
            q05 = float(np.quantile(rets, 0.05))
            daily_parts.append({
                "portfolio": portfolio,
                "slippage_bps_per_side": float(slip),
                "fold_number": int(fold),
                "fold_type": "calendar_period_mark_to_market_proxy",
                "calendar_dates": int(len(fg)),
                "start_date": fg["session_date"].min(),
                "end_date": fg["session_date"].max(),
                "net_calendar_period_return_pct_compounded": 100.0 * (float(np.prod(1.0 + rets)) - 1.0),
                "mean_daily_return_pct": float(np.mean(rets) * 100.0),
                "daily_volatility_pct": float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0) if len(rets) > 1 else 0.0,
                "daily_cvar_95_loss_pct": -100.0 * float(np.mean(rets[rets <= q05])),
                "average_exposure_pct": float(fg["daily_exposure_pct"].mean()),
                "days_exposure_above_75pct": int((fg["daily_exposure_pct"] > 75.0).sum()),
                "fold_assignment_method": "most recent validation signal date; concurrent holdings may originate in a previous fold",
            })
    calendar_fold = pd.DataFrame(daily_parts)
    return trade_fold, calendar_fold


def build_regime_trade_report(trades: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    joined = trades.merge(
        regimes[["session_date", "fold_number", "nifty_trend_state", "market_risk_state", "vix_level_regime", "india_vix_close", "vix_training_q75"]],
        left_on="signal_date", right_on="session_date", how="left", validate="many_to_one",
        suffixes=("", "_regime"),
    )
    if joined["nifty_trend_state"].isna().any() or joined["market_risk_state"].isna().any() or joined["vix_level_regime"].isna().any():
        raise ValueError("Some trades could not be assigned a signal-origin market regime.")
    fold_regime_col = "fold_number_regime" if "fold_number_regime" in joined.columns else "fold_number_y"
    if fold_regime_col not in joined.columns:
        raise ValueError("Could not identify the joined Part 3 fold-number column.")
    if (joined["fold_number"].astype(int) != joined[fold_regime_col].astype(int)).any():
        raise ValueError("Trade fold and Part 3 signal-date fold disagree.")

    rows = []
    regime_columns = [
        ("nifty_trend", "nifty_trend_state"),
        ("market_risk", "market_risk_state"),
        ("vix_level_vs_training_q75", "vix_level_regime"),
    ]
    for regime_type, column in regime_columns:
        for (portfolio, slip, regime), group in joined.groupby(["portfolio", "slippage_bps_per_side", column], sort=True):
            rows.append({
                "portfolio": portfolio,
                "slippage_bps_per_side": float(slip),
                "regime_type": regime_type,
                "regime": str(regime),
                "signal_dates": int(group["signal_date"].nunique()),
                "completed_stock_legs": int(len(group)),
                "unique_instruments": int(group["instrument_key"].nunique()),
                "gross_pnl_inr": float(group["gross_pnl_inr"].sum()),
                "net_pnl_inr": float(group["net_pnl_inr"].sum()),
                "fees_inr": float(group["total_fees_inr"].sum()),
                "slippage_inr": float(group["total_slippage_inr"].sum()),
                "net_pnl_contribution_pct_initial_capital": 100.0 * float(group["net_pnl_inr"].sum()) / INITIAL_CAPITAL_INR,
                "average_leg_net_return_pct": float(group["net_return_on_entry_cash_pct"].mean()),
                "median_leg_net_return_pct": float(group["net_return_on_entry_cash_pct"].median()),
                "winning_legs_pct": 100.0 * float((group["net_pnl_inr"] > 0).mean()),
                "worst_leg_pnl_inr": float(group["net_pnl_inr"].min()),
                "sample_warning": "DESCRIPTIVE_ONLY" if group["signal_date"].nunique() < 20 else "",
                "attribution_basis": "market regime on signal date; full seven-session trade P&L assigned to signal-origin regime",
            })
    return pd.DataFrame(rows)


def build_regime_exposure_report(daily_risk: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regime_columns = [
        ("nifty_trend", "nifty_trend_state"),
        ("market_risk", "market_risk_state"),
        ("vix_level_vs_training_q75", "vix_level_regime"),
    ]
    merged = daily_risk.merge(
        regimes[["session_date", "nifty_trend_state", "market_risk_state", "vix_level_regime"]],
        on="session_date", how="inner", validate="many_to_one",
    )
    for regime_type, column in regime_columns:
        for (portfolio, slip, regime), group in merged.groupby(["portfolio", "slippage_bps_per_side", column], sort=True):
            returns = group["daily_return_pct"].to_numpy(float)
            q05 = float(np.quantile(returns, 0.05))
            rows.append({
                "portfolio": portfolio,
                "slippage_bps_per_side": float(slip),
                "regime_type": regime_type,
                "regime": str(regime),
                "daily_observations": int(len(group)),
                "average_exposure_pct": float(group["daily_exposure_pct"].mean()),
                "median_exposure_pct": float(group["daily_exposure_pct"].median()),
                "maximum_exposure_pct": float(group["daily_exposure_pct"].max()),
                "days_exposure_above_50pct": int((group["daily_exposure_pct"] > 50.0).sum()),
                "days_exposure_above_75pct": int((group["daily_exposure_pct"] > 75.0).sum()),
                "mean_daily_return_pct": float(returns.mean()),
                "worst_daily_return_pct": float(returns.min()),
                "daily_cvar_95_loss_pct": -float(returns[returns <= q05].mean()),
                "sample_warning": "DESCRIPTIVE_ONLY" if len(group) < 20 else "",
                "attribution_basis": "daily mark-to-market date regime; excludes exit-tail dates without a validation regime row",
            })
    return pd.DataFrame(rows)


def build_prediction_reliability(part3_daily: pd.DataFrame) -> pd.DataFrame:
    primary = part3_daily[part3_daily["experiment"].astype(str).str.lower() == "primary"].copy()
    rows = []
    for model in ("extra_trees", "random_forest"):
        frame = primary[primary["model"] == model].sort_values("session_date").copy()
        if len(frame) != 303 or frame["session_date"].duplicated().any():
            raise ValueError(f"Expected 303 unique primary validation dates for {model} reliability history.")
        frame = frame.reset_index(drop=True)
        matured_ic = frame["cross_sectional_spearman_ic"].shift(RELIABILITY_MATURITY_LAG)
        matured_spread = frame["top_minus_bottom_actual_return"].shift(RELIABILITY_MATURITY_LAG)
        frame["matured_through_signal_date"] = frame["session_date"].shift(RELIABILITY_MATURITY_LAG)
        frame["trailing_matured_mean_ic_20"] = matured_ic.rolling(RELIABILITY_LOOKBACK, min_periods=RELIABILITY_LOOKBACK).mean()
        frame["trailing_matured_mean_top_bottom_spread_20"] = matured_spread.rolling(RELIABILITY_LOOKBACK, min_periods=RELIABILITY_LOOKBACK).mean()
        frame["matured_observation_count"] = matured_ic.rolling(RELIABILITY_LOOKBACK, min_periods=RELIABILITY_LOOKBACK).count()

        states = []
        for ic, spread in zip(frame["trailing_matured_mean_ic_20"], frame["trailing_matured_mean_top_bottom_spread_20"]):
            if not np.isfinite(ic) or not np.isfinite(spread):
                state = "insufficient_history"
            elif ic > 0 and spread > 0:
                state = "positive_trailing_reliability"
            elif ic <= 0 and spread <= 0:
                state = "weak_trailing_reliability"
            else:
                state = "mixed_trailing_reliability"
            states.append(state)
        frame["prediction_reliability_state"] = states
        frame["model"] = model
        frame["reliability_method"] = (
            "20 prior matured validation dates; outcomes shifted by seven validation sessions; "
            "positive requires both mean Spearman IC and mean top-bottom realised spread > 0; "
            "weak requires both <= 0"
        )
        rows.append(frame[[
            "model", "fold_number", "session_date", "matured_through_signal_date",
            "cross_sectional_spearman_ic", "top_minus_bottom_actual_return",
            "trailing_matured_mean_ic_20", "trailing_matured_mean_top_bottom_spread_20",
            "matured_observation_count", "prediction_reliability_state", "reliability_method",
        ]])
    return pd.concat(rows, ignore_index=True)


def build_reliability_exposure_report(daily_risk: pd.DataFrame, reliability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reliability_rows = []
    pair_rows = []
    for portfolio, model in PORTFOLIO_MODEL.items():
        rel = reliability[reliability["model"] == model].copy()
        base = daily_risk[daily_risk["portfolio"] == portfolio].merge(
            rel[["session_date", "prediction_reliability_state", "trailing_matured_mean_ic_20",
                 "trailing_matured_mean_top_bottom_spread_20", "matured_through_signal_date"]],
            on="session_date", how="left", validate="many_to_one",
        )
        base = base[base["prediction_reliability_state"].notna()].copy()
        for (slip, state), group in base.groupby(["slippage_bps_per_side", "prediction_reliability_state"], sort=True):
            returns = group["daily_return_pct"].to_numpy(float)
            q05 = float(np.quantile(returns, 0.05)) if len(returns) else np.nan
            reliability_rows.append({
                "portfolio": portfolio,
                "model": model,
                "slippage_bps_per_side": float(slip),
                "prediction_reliability_state": state,
                "daily_observations": int(len(group)),
                "average_exposure_pct": float(group["daily_exposure_pct"].mean()),
                "median_exposure_pct": float(group["daily_exposure_pct"].median()),
                "maximum_exposure_pct": float(group["daily_exposure_pct"].max()),
                "days_exposure_above_50pct": int((group["daily_exposure_pct"] > 50.0).sum()),
                "days_exposure_above_75pct": int((group["daily_exposure_pct"] > 75.0).sum()),
                "mean_daily_return_pct": float(returns.mean()),
                "worst_daily_return_pct": float(returns.min()),
                "daily_cvar_95_loss_pct": -float(returns[returns <= q05].mean()),
                "mean_trailing_matured_ic": float(group["trailing_matured_mean_ic_20"].mean()),
                "mean_trailing_matured_spread": float(group["trailing_matured_mean_top_bottom_spread_20"].mean()),
                "interpretation": "descriptive exposure association only; no position was reduced or filtered",
            })
        for slip, slip_group in base.groupby("slippage_bps_per_side", sort=True):
            means = slip_group.groupby("prediction_reliability_state")["daily_exposure_pct"].mean().to_dict()
            weak = means.get("weak_trailing_reliability", np.nan)
            positive = means.get("positive_trailing_reliability", np.nan)
            pair_rows.append({
                "portfolio": portfolio,
                "model": model,
                "slippage_bps_per_side": float(slip),
                "weak_state_average_exposure_pct": float(weak) if np.isfinite(weak) else np.nan,
                "positive_state_average_exposure_pct": float(positive) if np.isfinite(positive) else np.nan,
                "weak_minus_positive_exposure_pp": float(weak - positive) if np.isfinite(weak) and np.isfinite(positive) else np.nan,
                "diagnostic_only": True,
                "note": "Exposure comparison is descriptive; state is derived only from prior matured OOF ranking outcomes.",
            })

    # Equal-weight is not selected by a prediction model, so do not assign it
    # an artificial confidence/reliability state.
    for slip in SCENARIOS_BPS:
        g = daily_risk[(daily_risk["portfolio"] == "equal_weight_20_integer_redistributed") & np.isclose(daily_risk["slippage_bps_per_side"], slip)]
        if not g.empty:
            reliability_rows.append({
                "portfolio": "equal_weight_20_integer_redistributed",
                "model": "not_model_driven",
                "slippage_bps_per_side": float(slip),
                "prediction_reliability_state": "not_applicable",
                "daily_observations": int(len(g)),
                "average_exposure_pct": float(g["daily_exposure_pct"].mean()),
                "median_exposure_pct": float(g["daily_exposure_pct"].median()),
                "maximum_exposure_pct": float(g["daily_exposure_pct"].max()),
                "days_exposure_above_50pct": int((g["daily_exposure_pct"] > 50.0).sum()),
                "days_exposure_above_75pct": int((g["daily_exposure_pct"] > 75.0).sum()),
                "mean_daily_return_pct": float(g["daily_return_pct"].mean()),
                "worst_daily_return_pct": float(g["daily_return_pct"].min()),
                "daily_cvar_95_loss_pct": -float(g.loc[g["daily_return_pct"] <= g["daily_return_pct"].quantile(0.05), "daily_return_pct"].mean()),
                "mean_trailing_matured_ic": np.nan,
                "mean_trailing_matured_spread": np.nan,
                "interpretation": "equal-weight benchmark has no model prediction reliability state",
            })
    return pd.DataFrame(reliability_rows), pd.DataFrame(pair_rows)


def reconcile_existing_portfolio_metrics(
    calculated: pd.DataFrame,
    source_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Check that the new daily risk calculations reproduce Part 4.2.2 basics."""
    comparisons = [
        "net_total_return_pct",
        "gross_total_return_before_costs_pct",
        "max_drawdown_pct",
        "annualized_volatility_pct",
        "sharpe_rf_zero",
        "average_daily_exposure_pct",
    ]
    rows = []
    for portfolio in PORTFOLIOS:
        for slip in SCENARIOS_BPS:
            new_row = calculated[(calculated.portfolio == portfolio) & np.isclose(calculated.slippage_bps_per_side, slip)]
            old_row = source_metrics[(source_metrics.portfolio == portfolio) & np.isclose(source_metrics.slippage_bps_per_side, slip)]
            if len(new_row) != 1 or len(old_row) != 1:
                raise ValueError(f"Expected exactly one calculated/source metrics row for {portfolio}/{slip} bps.")
            new = new_row.iloc[0]
            old = old_row.iloc[0]
            diffs = {}
            for col in comparisons:
                if col not in new.index or col not in old.index:
                    raise ValueError(f"Cannot reconcile missing metric column {col}.")
                a, b = float(new[col]), float(old[col])
                diffs[col] = abs(a - b)
                tolerance = 1e-5 if col == "sharpe_rf_zero" else 0.01
                if not np.isclose(a, b, rtol=1e-8, atol=tolerance, equal_nan=True):
                    raise ValueError(f"Part 4.2.2 metric reconciliation failed for {portfolio}/{slip}/{col}: {a} vs {b}.")
            rows.append({
                "portfolio": portfolio,
                "slippage_bps_per_side": float(slip),
                "metrics_reconciled": ";".join(comparisons),
                "passed": True,
                "maximum_absolute_difference": max(diffs.values()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    from backend.app.ml.phase15.validation import part4_2_1_diagnostics as p421

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    protected, protected_fingerprint = p421._load_protection_helpers()
    protected_before = {str(path): protected_fingerprint(path) for path in protected}
    if len(protected) != 11:
        raise RuntimeError(f"Expected 11 protected production/V4 files; found {len(protected)}.")

    print("=" * 100)
    print("PHASE 15 PART 4.3 - RISK, DRAWDOWN, STABILITY AND REGIME ASSESSMENT")
    print("=" * 100)
    print(f"[1/8] Loading saved Part 4.2.2 run {PART422_RUN_ID}...")
    daily, trades, source_metrics, summary, part422_paths, schedule_reconciliation = load_part422_inputs()
    print(f"      daily-equity rows={len(daily):,}; trade legs={len(trades):,}; portfolios={len(PORTFOLIOS)}; scenarios={len(SCENARIOS_BPS)}")

    print("[2/8] Loading prior OOF ranking diagnostics and daily market context...")
    part3_daily, market, sample_sizes, context_paths = load_regime_sources()
    print(f"      Part 3 daily discrimination rows={len(part3_daily):,}; Part 2 market rows={len(market):,}")

    input_paths = {**part422_paths, **context_paths}
    source_before = {str(path): sha256_file(path) for path in input_paths.values()}

    print("[3/8] Building NIFTY, market-risk and fold-specific elevated-VIX states...")
    regimes, vix_thresholds = build_market_regimes(part3_daily, market, sample_sizes)
    print(f"      validation regime dates={len(regimes):,}; elevated-VIX dates={(regimes['vix_level_regime'] == 'elevated').sum():,}")

    print("[4/8] Calculating daily drawdown, recovery, volatility and tail-risk metrics...")
    risk_rows = []
    daily_risk_frames = []
    drawdown_frames = []
    for (portfolio, slip), group in daily.groupby(["portfolio", "slippage_bps_per_side"], sort=True):
        risk, daily_risk, episodes = calculate_daily_risk_metrics(group)
        risk.update({"portfolio": portfolio, "slippage_bps_per_side": float(slip)})
        risk_rows.append(risk)
        daily_risk_frames.append(daily_risk)
        if not episodes.empty:
            episodes.insert(0, "portfolio", portfolio)
            episodes.insert(1, "slippage_bps_per_side", float(slip))
            drawdown_frames.append(episodes)
    portfolio_risk = pd.DataFrame(risk_rows)
    daily_risk = pd.concat(daily_risk_frames, ignore_index=True)
    dd_episodes = pd.concat(drawdown_frames, ignore_index=True) if drawdown_frames else pd.DataFrame()
    metric_reconciliation = reconcile_existing_portfolio_metrics(portfolio_risk, source_metrics)

    print("      Part 4.2.2 core portfolio metrics reconciled: PASS")
    print("[5/8] Measuring stock-loss concentration and fold stability...")
    stock_concentration = build_stock_concentration(trades)
    fold_trade, fold_calendar = build_fold_reports(trades, daily_risk, regimes)

    print("[6/8] Assessing performance/exposure by market regime and matured prediction reliability...")
    regime_trade = build_regime_trade_report(trades, regimes)
    regime_exposure = build_regime_exposure_report(daily_risk, regimes)
    reliability = build_prediction_reliability(part3_daily)
    reliability_exposure, reliability_exposure_comparison = build_reliability_exposure_report(daily_risk, reliability)

    print("[7/8] Saving new research-only Part 4.3 reports...")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {
        "portfolio_risk_metrics": REPORT_DIR / f"phase15_part4_3_{run_id}_portfolio_risk_metrics.csv",
        "daily_risk_series": REPORT_DIR / f"phase15_part4_3_{run_id}_daily_risk_series.csv",
        "drawdown_episodes": REPORT_DIR / f"phase15_part4_3_{run_id}_drawdown_episodes.csv",
        "stock_risk_concentration": REPORT_DIR / f"phase15_part4_3_{run_id}_stock_risk_concentration.csv",
        "fold_trade_attribution": REPORT_DIR / f"phase15_part4_3_{run_id}_fold_trade_attribution.csv",
        "fold_calendar_risk": REPORT_DIR / f"phase15_part4_3_{run_id}_fold_calendar_risk.csv",
        "market_regime_trade_performance": REPORT_DIR / f"phase15_part4_3_{run_id}_market_regime_trade_performance.csv",
        "market_regime_exposure": REPORT_DIR / f"phase15_part4_3_{run_id}_market_regime_exposure.csv",
        "prediction_reliability_history": REPORT_DIR / f"phase15_part4_3_{run_id}_prediction_reliability_history.csv",
        "reliability_exposure": REPORT_DIR / f"phase15_part4_3_{run_id}_reliability_exposure.csv",
        "reliability_exposure_comparison": REPORT_DIR / f"phase15_part4_3_{run_id}_reliability_exposure_comparison.csv",
        "vix_thresholds": REPORT_DIR / f"phase15_part4_3_{run_id}_vix_thresholds.csv",
        "market_regime_calendar": REPORT_DIR / f"phase15_part4_3_{run_id}_market_regime_calendar.csv",
        "source_metric_reconciliation": REPORT_DIR / f"phase15_part4_3_{run_id}_source_metric_reconciliation.csv",
        "fixed_quantity_anchor_reconciliation": REPORT_DIR / f"phase15_part4_3_{run_id}_fixed_quantity_anchor_reconciliation.csv",
        "summary": REPORT_DIR / f"phase15_part4_3_{run_id}_summary.json",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {path}")

    reports = {
        "portfolio_risk_metrics": portfolio_risk,
        "daily_risk_series": daily_risk,
        "drawdown_episodes": dd_episodes,
        "stock_risk_concentration": stock_concentration,
        "fold_trade_attribution": fold_trade,
        "fold_calendar_risk": fold_calendar,
        "market_regime_trade_performance": regime_trade,
        "market_regime_exposure": regime_exposure,
        "prediction_reliability_history": reliability,
        "reliability_exposure": reliability_exposure,
        "reliability_exposure_comparison": reliability_exposure_comparison,
        "vix_thresholds": vix_thresholds,
        "market_regime_calendar": regimes,
        "source_metric_reconciliation": metric_reconciliation,
        "fixed_quantity_anchor_reconciliation": schedule_reconciliation,
    }
    for name, frame in reports.items():
        frame.to_csv(outputs[name], index=False)

    # Recheck every source and protected fingerprint before writing final summary.
    source_after = {str(path): sha256_file(path) for path in input_paths.values()}
    protected_after = {str(path): protected_fingerprint(path) for path in protected}
    if source_before != source_after:
        raise RuntimeError("Part 4.2.2/Part 2/Part 3 source reports changed during Part 4.3.")
    if protected_before != protected_after:
        raise RuntimeError("Protected production/V4 artifact fingerprints changed during Part 4.3.")

    summary = {
        "run_id": run_id,
        "phase": 15,
        "part": "4.3",
        "title": "Risk, Drawdown, Portfolio Stability and Market-Regime Assessment",
        "research_only": True,
        "source_runs": {
            "part4_2_2": PART422_RUN_ID,
            "part3_daily_discrimination": PART3_RUN_ID,
            "part2_daily_market_context": PART2_RUN_ID,
            "part3_5_regime_sample_sizes": PART3_5_RUN_ID,
        },
        "portfolios": list(PORTFOLIOS),
        "slippage_scenarios_bps_per_side": list(SCENARIOS_BPS),
        "risk_metrics": [
            "net return and CAGR", "annualized volatility", "Sharpe and Sortino (zero risk-free rate)",
            "maximum drawdown", "drawdown episode and recovery duration", "Ulcer Index",
            "daily 5th/1st percentile and CVaR", "exposure and time above exposure thresholds",
        ],
        "fold_methodology": {
            "trade_attribution": "Full stock-leg P&L assigned to the fold of its signal date.",
            "calendar_risk": "Daily mark-to-market dates inherit the most recent validation-fold label; overlapping positions can originate in prior folds.",
        },
        "market_regime_methodology": {
            "nifty_trend_and_market_risk": "Existing Part 3 daily discrimination labels, joined by signal date.",
            "elevated_vix": "India VIX close > fold-specific 75th percentile computed from the 120 sessions strictly before each validation start; threshold is frozen per fold.",
            "trade_attribution": "Full seven-session trade P&L attributed to market regime on the signal date.",
            "daily_exposure": "Daily exposure associated with the same-date regime; exit-tail dates without a validation signal regime are excluded from regime exposure summaries.",
        },
        "prediction_reliability_methodology": {
            "models": {"extra_trees_top5": "extra_trees", "random_forest_top5": "random_forest"},
            "metric_inputs": ["cross_sectional_spearman_ic", "top_minus_bottom_actual_return"],
            "lookback_matured_signal_dates": RELIABILITY_LOOKBACK,
            "maturity_lag_sessions": RELIABILITY_MATURITY_LAG,
            "positive_state": "Both trailing 20-matured-date mean IC and mean top-bottom realised spread are > 0.",
            "weak_state": "Both trailing 20-matured-date mean IC and mean top-bottom realised spread are <= 0.",
            "mixed_state": "The two trailing measures have different signs.",
            "insufficient_history": "Fewer than 20 matured observations available.",
            "important_limitation": "This is a past-only historical ranking-quality proxy, not model confidence, a calibrated probability, or a validated position-sizing rule. It does not alter historical exposure.",
        },
        "holdout": {
            "final_holdout_start": FINAL_HOLDOUT_START,
            "final_holdout_used": False,
            "maximum_part4_2_2_date": MAX_ALLOWED_DATE,
            "vix_thresholds_use_only_pre_validation_training_data": True,
        },
        "models_fitted": False,
        "predictions_modified": False,
        "hyperparameters_changed": False,
        "upstox_api_called": False,
        "source_reports_modified": False,
        "protected_artifact_count": len(protected),
        "protected_artifacts_modified": False,
        "source_fingerprints_unchanged": True,
        "part4_2_2_trade_source_reconciliation": {
            "fixed_quantity_20bps_anchor_matches": True,
            "trade_counts_match_portfolio_metrics": True,
            "net_pnl_matches_ending_equity": True,
            "gross_net_cost_ledger_reconciles": True,
        },
        "limitations": [
            "This is descriptive analysis of previously reviewed out-of-fold predictions and fixed-quantity historical replays, not independent final-holdout evidence.",
            "Seven-session target returns overlap; fold and market-regime groups are dependent and descriptive. No new significance tests or multiple-testing adjustment are applied.",
            "Drawdowns and tail metrics are based on simulated daily liquidation equity under the specified fee/slippage assumptions.",
            "Daily OHLC cannot reproduce order-book depth, partial fills, market impact, auction execution, corporate actions or dividends.",
            "Signal-origin regime attribution assigns each complete seven-session trade to its entry signal's regime; the market may change during the holding period.",
            "The trailing reliability state is retrospective monitoring research. It must not be used as a live risk gate without a separately frozen and forward-tested policy.",
            "NIFTY and VIX regime definitions are inherited or reconstructed from the earlier Phase 15 regime methodology; small groups are flagged descriptive-only.",
        ],
        "reports": {name: path.name for name, path in outputs.items()},
    }
    with outputs["summary"].open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("[8/8] Protected artifacts and source report fingerprints verified unchanged.")
    print("\n" + "=" * 100)
    print("PHASE 15 PART 4.3 COMPLETED")
    print("=" * 100)
    print(portfolio_risk[[
        "portfolio", "slippage_bps_per_side", "net_total_return_pct", "max_drawdown_pct",
        "longest_underwater_sessions", "unrecovered_drawdown_episodes_at_end",
        "annualized_volatility_pct", "sharpe_rf_zero", "sortino_rf_zero",
        "daily_cvar_95_loss_pct", "average_daily_exposure_pct",
    ]].round(4).to_string(index=False))
    print("\nDrawdown episodes:", len(dd_episodes))
    print("Stock concentration rows:", len(stock_concentration))
    print("Fold trade attribution rows:", len(fold_trade))
    print("Market regime trade rows:", len(regime_trade))
    print("Reliability history rows:", len(reliability))
    print("Reliability exposure rows:", len(reliability_exposure))
    print("Part 4.2.2 risk metrics reconciled: PASSED")
    print("Elevated-VIX dates reconciled to Part 3.5 sample counts: PASSED")
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Part 4.2.2 / Part 2 / Part 3 source reports: UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("\nReports:")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
