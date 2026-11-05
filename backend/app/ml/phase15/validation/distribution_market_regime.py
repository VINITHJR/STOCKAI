from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from backend.app.ml.phase15.validation.real_data_benchmark import (
    build_real_data_plan,
    discover_protected_files,
    fingerprint,
)


REPORT_DIR = Path("backend/app/ml/phase15/reports")
SOURCE_RUN_ID = "20261031_131021"
WINDOWS = (60, 120, 252)

TARGETS = {
    "primary": "next_open_to_close_return_7s",
    "forecast": "close_to_close_return_7s",
}

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "overnight_gap_return",
    "intraday_return",
    "daily_range_pct",
    "close_location",
    "realized_volatility_20d",
    "sma20_gap",
    "sma50_gap",
    "rsi_wilder_14",
    "macd_pct",
    "macd_signal_pct",
    "bb_width_20",
    "bb_percent_b_20",
    "atr_pct_14",
    "relative_volume_log_20",
    "trend_efficiency_20d",
    "nifty_return_1d",
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_realized_volatility_20d",
    "nifty_close_vs_sma20",
    "nifty_close_vs_sma50",
    "india_vix_close",
    "india_vix_return_1d",
    "india_vix_return_5d",
    "india_vix_close_vs_sma20",
    "relative_return_1d",
    "relative_return_5d",
    "relative_return_20d",
]

MARKET_FEATURES = [
    "nifty_return_1d",
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_realized_volatility_20d",
    "nifty_close_vs_sma20",
    "nifty_close_vs_sma50",
    "india_vix_close",
    "india_vix_return_1d",
    "india_vix_return_5d",
    "india_vix_close_vs_sma20",
]

EXPECTED_FOLDS = {
    1: ("2024-01-09", "2025-01-02", "2025-01-14", "2025-04-11", 4900, 1220),
    2: ("2024-01-09", "2025-04-01", "2025-04-15", "2025-07-10", 6120, 1220),
    3: ("2024-01-09", "2025-07-01", "2025-07-11", "2025-10-08", 7340, 1220),
    4: ("2024-01-09", "2025-09-26", "2025-10-09", "2026-01-05", 8560, 1200),
    5: ("2024-01-09", "2025-12-24", "2026-01-06", "2026-04-06", 9760, 1200),
}


def date_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        raise ValueError("Cannot calculate statistics on an empty sample.")

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "positive_pct": float(np.mean(values > 0) * 100),
        "negative_pct": float(np.mean(values < 0) * 100),
        "zero_pct": float(np.mean(values == 0) * 100),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def psi_score(training: np.ndarray, validation: np.ndarray, bins=10) -> float:
    """PSI using quantile boundaries fitted only on the training window."""

    if len(training) == 0 or len(validation) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    unique_edges = np.unique(np.quantile(training, quantiles))

    if len(unique_edges) == 1:
        return (
            0.0
            if np.allclose(validation, unique_edges[0])
            else float("nan")
        )

    if len(unique_edges) == 2:
        midpoint = float(np.mean(unique_edges))
        edges = np.array([-np.inf, midpoint, np.inf])
    else:
        edges = np.concatenate((
            [-np.inf],
            unique_edges[1:-1],
            [np.inf],
        ))

    train_counts = np.histogram(training, bins=edges)[0]
    validation_counts = np.histogram(validation, bins=edges)[0]

    smoothing = 0.5
    train_pct = (
        train_counts + smoothing
    ) / (
        train_counts.sum() + smoothing * len(train_counts)
    )
    validation_pct = (
        validation_counts + smoothing
    ) / (
        validation_counts.sum() + smoothing * len(validation_counts)
    )

    return float(np.sum(
        (validation_pct - train_pct)
        * np.log(validation_pct / train_pct)
    ))


def psi_level(value: float) -> str:
    if not np.isfinite(value):
        return "not_evaluable"
    if value >= 0.25:
        return "high"
    if value >= 0.10:
        return "moderate"
    return "low"


def load_and_validate_plan():
    plan = build_real_data_plan()
    panel = plan.panel.copy()

    required = {
        "instrument_key",
        "session_date",
        *FEATURES,
        *TARGETS.values(),
    }

    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            "Panel is missing columns: " + ", ".join(sorted(missing))
        )

    panel["session_date"] = pd.to_datetime(
        panel["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if len(plan.folds) != 5:
        raise ValueError(f"Expected five folds; received {len(plan.folds)}.")

    holdout = set(plan.final_holdout_indices)
    all_analysis_indices = set()

    for fold in plan.folds:
        number = int(fold.fold_number)

        if number not in EXPECTED_FOLDS:
            raise ValueError(f"Unexpected fold number: {number}")

        expected = EXPECTED_FOLDS[number]

        train_indices = list(fold.train_indices)
        validation_indices = list(fold.validation_indices)

        if set(train_indices).intersection(validation_indices):
            raise ValueError(f"Fold {number}: train/validation overlap.")

        if holdout.intersection(train_indices + validation_indices):
            raise ValueError(f"Fold {number}: final holdout overlap.")

        training = panel.iloc[train_indices]
        validation = panel.iloc[validation_indices]

        actual = (
            date_text(training["session_date"].min()),
            date_text(training["session_date"].max()),
            date_text(validation["session_date"].min()),
            date_text(validation["session_date"].max()),
            len(training),
            len(validation),
        )

        if actual != expected:
            raise ValueError(
                f"Fold {number} differs from original benchmark.\n"
                f"Expected: {expected}\nActual: {actual}"
            )

        if training["session_date"].nunique() != fold.train_session_count:
            raise ValueError(f"Fold {number}: training date count mismatch.")

        if validation["session_date"].nunique() != fold.validation_session_count:
            raise ValueError(f"Fold {number}: validation date count mismatch.")

        all_analysis_indices.update(train_indices)
        all_analysis_indices.update(validation_indices)

    return plan, panel, sorted(all_analysis_indices)


def reconcile_targets(panel: pd.DataFrame, plan) -> list[dict]:
    """Confirm rebuilt validation targets match the previous benchmark."""

    results = []

    for experiment, target in TARGETS.items():
        path = REPORT_DIR / (
            f"phase15_benchmark_{SOURCE_RUN_ID}_{experiment}_oof_predictions.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        saved = pd.read_csv(
            path,
            usecols=[
                "fold_number",
                "instrument_key",
                "session_date",
                "actual_return",
            ],
        )

        saved["session_date"] = pd.to_datetime(
            saved["session_date"], errors="raise"
        ).dt.strftime("%Y-%m-%d")

        duplicates = saved.groupby(
            ["fold_number", "instrument_key", "session_date"]
        )["actual_return"].nunique()

        if (duplicates > 1).any():
            raise ValueError(
                f"{experiment}: models disagree on saved actual returns."
            )

        saved = saved.drop_duplicates(
            ["fold_number", "instrument_key", "session_date"]
        )

        pieces = []

        for fold in plan.folds:
            part = panel.iloc[list(fold.validation_indices)][
                ["instrument_key", "session_date", target]
            ].copy()

            part["fold_number"] = int(fold.fold_number)
            part = part.rename(columns={target: "rebuilt_actual"})

            pieces.append(part)

        rebuilt = pd.concat(pieces, ignore_index=True)

        merged = rebuilt.merge(
            saved,
            on=["fold_number", "instrument_key", "session_date"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(
                f"{experiment}: saved and reconstructed samples differ."
            )

        max_diff = float(np.max(np.abs(
            merged["rebuilt_actual"].to_numpy(dtype=float)
            - merged["actual_return"].to_numpy(dtype=float)
        )))

        if not np.allclose(
            merged["rebuilt_actual"],
            merged["actual_return"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"{experiment}: target reconciliation failed; "
                f"maximum difference={max_diff}"
            )

        results.append({
            "experiment": experiment,
            "matched_samples": int(len(merged)),
            "maximum_absolute_difference": max_diff,
            "passed": True,
        })

    return results


def recent_window_target_report(panel, plan) -> pd.DataFrame:
    rows = []

    for fold in plan.folds:
        number = int(fold.fold_number)
        training_all = panel.iloc[list(fold.train_indices)]
        validation = panel.iloc[list(fold.validation_indices)]

        ordered_dates = sorted(training_all["session_date"].unique())

        for window_size in WINDOWS:
            if len(ordered_dates) < window_size:
                continue

            selected_dates = set(ordered_dates[-window_size:])
            training = training_all[
                training_all["session_date"].isin(selected_dates)
            ]

            for experiment, target in TARGETS.items():
                train_values = finite_values(training[target])
                validation_values = finite_values(validation[target])

                train_stats = stats(train_values)
                validation_stats = stats(validation_values)

                train_daily = (
                    training.groupby("session_date")[target].mean().to_numpy()
                )
                validation_daily = (
                    validation.groupby("session_date")[target].mean().to_numpy()
                )

                train_daily_stats = stats(train_daily)
                validation_daily_stats = stats(validation_daily)

                psi = psi_score(train_values, validation_values)
                ks = ks_2samp(train_values, validation_values).statistic

                rows.append({
                    "fold_number": number,
                    "window_sessions": window_size,
                    "experiment": experiment,
                    "target_column": target,
                    "training_start": date_text(training["session_date"].min()),
                    "training_end": date_text(training["session_date"].max()),
                    "validation_start": date_text(validation["session_date"].min()),
                    "validation_end": date_text(validation["session_date"].max()),
                    "training_stock_session_rows": train_stats["count"],
                    "validation_stock_session_rows": validation_stats["count"],
                    "training_daily_basket_mean": train_daily_stats["mean"],
                    "validation_daily_basket_mean": validation_daily_stats["mean"],
                    "daily_basket_mean_shift": (
                        validation_daily_stats["mean"] - train_daily_stats["mean"]
                    ),
                    "training_pooled_mean": train_stats["mean"],
                    "validation_pooled_mean": validation_stats["mean"],
                    "pooled_mean_shift": (
                        validation_stats["mean"] - train_stats["mean"]
                    ),
                    "training_pooled_median": train_stats["median"],
                    "validation_pooled_median": validation_stats["median"],
                    "training_positive_pct": train_stats["positive_pct"],
                    "validation_positive_pct": validation_stats["positive_pct"],
                    "training_negative_pct": train_stats["negative_pct"],
                    "validation_negative_pct": validation_stats["negative_pct"],
                    "training_std": train_stats["std"],
                    "validation_std": validation_stats["std"],
                    "validation_to_training_std_ratio": (
                        validation_stats["std"] / train_stats["std"]
                        if train_stats["std"] > 0 else np.nan
                    ),
                    "ks_distance": float(ks),
                    "psi": psi,
                    "psi_level": psi_level(psi),
                })

    return pd.DataFrame(rows)


def recent_window_feature_report(panel, plan) -> pd.DataFrame:
    rows = []

    for fold in plan.folds:
        number = int(fold.fold_number)
        training_all = panel.iloc[list(fold.train_indices)]
        validation = panel.iloc[list(fold.validation_indices)]

        ordered_dates = sorted(training_all["session_date"].unique())

        for window_size in WINDOWS:
            if len(ordered_dates) < window_size:
                continue

            selected_dates = set(ordered_dates[-window_size:])
            training = training_all[
                training_all["session_date"].isin(selected_dates)
            ]

            for feature in FEATURES:
                train_series = pd.to_numeric(
                    training[feature], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

                validation_series = pd.to_numeric(
                    validation[feature], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

                train_values = finite_values(train_series)
                validation_values = finite_values(validation_series)

                if not len(train_values) or not len(validation_values):
                    raise ValueError(
                        f"Fold {number}, window {window_size}, "
                        f"feature {feature}: empty observations."
                    )

                train_stats = stats(train_values)
                validation_stats = stats(validation_values)

                pooled_std = np.sqrt(
                    (train_stats["std"] ** 2 + validation_stats["std"] ** 2) / 2
                )

                psi = psi_score(train_values, validation_values)
                ks = ks_2samp(train_values, validation_values).statistic

                rows.append({
                    "fold_number": number,
                    "window_sessions": window_size,
                    "feature": feature,
                    "feature_family": (
                        "technical" if FEATURES.index(feature) < 18
                        else "market_context"
                    ),
                    "training_start": date_text(training["session_date"].min()),
                    "training_end": date_text(training["session_date"].max()),
                    "validation_start": date_text(validation["session_date"].min()),
                    "validation_end": date_text(validation["session_date"].max()),
                    "training_missing_pct": float(train_series.isna().mean() * 100),
                    "validation_missing_pct": float(
                        validation_series.isna().mean() * 100
                    ),
                    "missingness_shift_points": float(
                        (validation_series.isna().mean()
                         - train_series.isna().mean()) * 100
                    ),
                    "training_mean": train_stats["mean"],
                    "validation_mean": validation_stats["mean"],
                    "mean_shift": (
                        validation_stats["mean"] - train_stats["mean"]
                    ),
                    "training_median": train_stats["median"],
                    "validation_median": validation_stats["median"],
                    "median_shift": (
                        validation_stats["median"] - train_stats["median"]
                    ),
                    "training_std": train_stats["std"],
                    "validation_std": validation_stats["std"],
                    "standardized_mean_difference": (
                        (validation_stats["mean"] - train_stats["mean"])
                        / pooled_std if pooled_std > 0 else np.nan
                    ),
                    "ks_distance": float(ks),
                    "psi": psi,
                    "psi_level": psi_level(psi),
                })

    return pd.DataFrame(rows)


def build_daily_market_context(panel: pd.DataFrame,
                               analysis_indices: list[int]) -> pd.DataFrame:
    """Deduplicate common market features to one observation per session."""

    analysis = panel.iloc[analysis_indices]

    records = []

    for session_date, group in analysis.groupby("session_date", sort=True):
        record = {"session_date": session_date}

        for feature in MARKET_FEATURES:
            values = finite_values(group[feature])

            if len(values) == 0:
                raise ValueError(
                    f"Market feature {feature} is empty on {session_date}."
                )

            spread = float(values.max() - values.min())

            if spread > 1e-12:
                raise ValueError(
                    f"Market feature {feature} differs across stocks on "
                    f"{session_date}; max spread={spread}"
                )

            record[feature] = float(values[0])

        records.append(record)

    daily = pd.DataFrame(records).sort_values("session_date").reset_index(drop=True)

    # Transparent descriptive rules, not a trained or official regime classifier.
    bullish = (
        (daily["nifty_close_vs_sma50"] > 0)
        & (daily["nifty_return_20d"] > 0)
    )
    bearish = (
        (daily["nifty_close_vs_sma50"] < 0)
        & (daily["nifty_return_20d"] < 0)
    )

    daily["nifty_trend_state"] = np.select(
        [bullish, bearish],
        ["bullish", "bearish"],
        default="mixed",
    )

    daily["vix_above_sma20"] = (
        daily["india_vix_close_vs_sma20"] > 0
    )

    risk_on = bullish & ~daily["vix_above_sma20"]
    risk_off = bearish & daily["vix_above_sma20"]

    daily["market_risk_state"] = np.select(
        [risk_on, risk_off],
        ["risk_on", "risk_off"],
        default="mixed",
    )

    daily["nifty_volatility_20d"] = daily[
        "nifty_realized_volatility_20d"
    ]

    return daily


def regime_summary(
    daily: pd.DataFrame,
    panel: pd.DataFrame,
    plan,
) -> pd.DataFrame:
    rows = []

    for fold in plan.folds:
        number = int(fold.fold_number)
        training_all = panel.iloc[list(fold.train_indices)]
        validation = panel.iloc[list(fold.validation_indices)]

        training_dates = sorted(training_all["session_date"].unique())
        validation_dates = sorted(validation["session_date"].unique())

        for window_size in WINDOWS:
            if len(training_dates) < window_size:
                continue

            selected_dates = training_dates[-window_size:]
            train_daily = daily[
                daily["session_date"].isin(selected_dates)
            ].copy()

            validation_daily = daily[
                daily["session_date"].isin(validation_dates)
            ].copy()

            if len(train_daily) != window_size:
                raise ValueError(
                    f"Fold {number}: expected {window_size} daily "
                    f"training records; found {len(train_daily)}."
                )

            if len(validation_daily) != len(validation_dates):
                raise ValueError(
                    f"Fold {number}: validation daily records do not match."
                )

            # This threshold is derived only from the selected training window.
            vix_q75 = float(train_daily["india_vix_close"].quantile(0.75))

            for segment, sample in (
                ("recent_training", train_daily),
                ("validation", validation_daily),
            ):
                rows.append({
                    "fold_number": number,
                    "window_sessions": window_size,
                    "segment": segment,
                    "start_date": date_text(sample["session_date"].min()),
                    "end_date": date_text(sample["session_date"].max()),
                    "session_count": len(sample),
                    "nifty_bullish_pct": float(
                        (sample["nifty_trend_state"] == "bullish").mean() * 100
                    ),
                    "nifty_bearish_pct": float(
                        (sample["nifty_trend_state"] == "bearish").mean() * 100
                    ),
                    "nifty_mixed_pct": float(
                        (sample["nifty_trend_state"] == "mixed").mean() * 100
                    ),
                    "vix_above_sma20_pct": float(
                        sample["vix_above_sma20"].mean() * 100
                    ),
                    "vix_above_training_window_q75_pct": float(
                        (sample["india_vix_close"] > vix_q75).mean() * 100
                    ),
                    "risk_on_pct": float(
                        (sample["market_risk_state"] == "risk_on").mean() * 100
                    ),
                    "risk_off_pct": float(
                        (sample["market_risk_state"] == "risk_off").mean() * 100
                    ),
                    "risk_mixed_pct": float(
                        (sample["market_risk_state"] == "mixed").mean() * 100
                    ),
                    "nifty_return_5d_mean": float(
                        sample["nifty_return_5d"].mean()
                    ),
                    "nifty_return_20d_mean": float(
                        sample["nifty_return_20d"].mean()
                    ),
                    "nifty_close_vs_sma50_mean": float(
                        sample["nifty_close_vs_sma50"].mean()
                    ),
                    "nifty_realized_volatility_mean": float(
                        sample["nifty_realized_volatility_20d"].mean()
                    ),
                    "india_vix_close_mean": float(
                        sample["india_vix_close"].mean()
                    ),
                    "india_vix_return_5d_mean": float(
                        sample["india_vix_return_5d"].mean()
                    ),
                    "training_window_vix_q75": vix_q75,
                })

    return pd.DataFrame(rows)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    protected_files = sorted(discover_protected_files())

    if len(protected_files) != 11:
        raise RuntimeError(
            f"Expected 11 protected files, found {len(protected_files)}."
        )

    protected_before = {
        str(path): fingerprint(path)
        for path in protected_files
    }

    print("=" * 90)
    print("PHASE 15: RECENT-WINDOW AND MARKET-REGIME ANALYSIS")
    print("=" * 90)

    print("\nReconstructing the original benchmark plan...")
    plan, panel, analysis_indices = load_and_validate_plan()

    print("Original fold boundaries and sample counts: MATCHED")
    print("Final holdout: EXCLUDED")

    print("\nReconciling validation targets with saved benchmark...")
    reconciliation = reconcile_targets(panel, plan)

    for item in reconciliation:
        print(
            f"{item['experiment']}: {item['matched_samples']} matched; "
            f"max difference={item['maximum_absolute_difference']:.12g}"
        )

    print("\nBuilding recent-window target comparisons...")
    target_report = recent_window_target_report(panel, plan)

    print("Building recent-window drift for all 31 features...")
    feature_report = recent_window_feature_report(panel, plan)

    print("Building one-row-per-session NIFTY/VIX history...")
    daily_market = build_daily_market_context(panel, analysis_indices)

    print("Summarizing market regimes by fold and training window...")
    market_report = regime_summary(daily_market, panel, plan)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports = {
        "target": REPORT_DIR / f"phase15_part2_{run_id}_target_windows.csv",
        "features": REPORT_DIR / f"phase15_part2_{run_id}_feature_windows.csv",
        "daily_market": REPORT_DIR / f"phase15_part2_{run_id}_daily_market_context.csv",
        "market_summary": REPORT_DIR / f"phase15_part2_{run_id}_market_regime_summary.csv",
        "summary": REPORT_DIR / f"phase15_part2_{run_id}_summary.json",
    }

    for path in reports.values():
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {path}"
            )

    target_report.to_csv(reports["target"], index=False)
    feature_report.to_csv(reports["features"], index=False)
    daily_market.to_csv(reports["daily_market"], index=False)
    market_report.to_csv(reports["market_summary"], index=False)

    summary = {
        "run_id": run_id,
        "source_benchmark_run_id": SOURCE_RUN_ID,
        "analysis": "recent_window_and_market_regime",
        "recent_training_windows": list(WINDOWS),
        "target_columns": TARGETS,
        "feature_count": len(FEATURES),
        "fold_count": len(plan.folds),
        "analysis_panel_rows": len(analysis_indices),
        "daily_market_observations": len(daily_market),
        "target_report_rows": len(target_report),
        "feature_report_rows": len(feature_report),
        "market_summary_rows": len(market_report),
        "validation_target_reconciliation": reconciliation,
        "final_holdout_used": False,
        "models_fitted": False,
        "production_artifacts_modified": False,
        "market_regime_rules": {
            "bullish": "NIFTY close_vs_SMA50 > 0 AND NIFTY 20-day return > 0",
            "bearish": "NIFTY close_vs_SMA50 < 0 AND NIFTY 20-day return < 0",
            "mixed": "Neither bullish nor bearish condition",
            "risk_on": "Bullish NIFTY state AND VIX below its SMA20",
            "risk_off": "Bearish NIFTY state AND VIX above its SMA20",
            "risk_mixed": "All other combinations",
            "vix_high_threshold": (
                "75th percentile of India VIX close in the corresponding "
                "recent training window; applied unchanged to validation."
            ),
            "warning": (
                "These are transparent descriptive rules, not a trained "
                "or officially defined market-regime classifier."
            ),
        },
        "dependence_warning": (
            "Stock-session rows have overlapping targets and cross-stock "
            "dependence. Daily market context is deduplicated to one row "
            "per trading date."
        ),
        "reports": {
            key: path.name
            for key, path in reports.items()
        },
        "protected_artifact_count": len(protected_files),
    }

    with reports["summary"].open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    protected_after = {
        str(path): fingerprint(path)
        for path in protected_files
    }

    if protected_before != protected_after:
        raise RuntimeError(
            "Protected production/V4 fingerprints changed."
        )

    print("\n" + "=" * 90)
    print("A. RECENT-WINDOW TARGET COMPARISON")
    print("=" * 90)

    cols = [
        "fold_number",
        "window_sessions",
        "experiment",
        "training_daily_basket_mean",
        "validation_daily_basket_mean",
        "training_positive_pct",
        "validation_positive_pct",
        "training_negative_pct",
        "validation_negative_pct",
        "validation_to_training_std_ratio",
        "psi",
        "psi_level",
    ]

    print(target_report[cols].round(5).to_string(index=False))

    print("\n" + "=" * 90)
    print("B. FOLD 5: TOP FEATURE DRIFT BY RECENT WINDOW")
    print("=" * 90)

    for window_size in WINDOWS:
        subset = feature_report[
            (feature_report["fold_number"] == 5)
            & (feature_report["window_sessions"] == window_size)
        ].sort_values("psi", ascending=False)

        print(f"\nFOLD 5 / LAST {window_size} TRAINING SESSIONS")

        if subset.empty:
            print("No observations.")
            continue

        print(
            subset[
                [
                    "feature",
                    "feature_family",
                    "training_mean",
                    "validation_mean",
                    "standardized_mean_difference",
                    "ks_distance",
                    "psi",
                    "psi_level",
                ]
            ].head(12).round(5).to_string(index=False)
        )

    print("\n" + "=" * 90)
    print("C. MARKET-REGIME SUMMARY")
    print("=" * 90)

    summary_cols = [
        "fold_number",
        "window_sessions",
        "segment",
        "nifty_bullish_pct",
        "nifty_bearish_pct",
        "nifty_mixed_pct",
        "vix_above_sma20_pct",
        "vix_above_training_window_q75_pct",
        "risk_on_pct",
        "risk_off_pct",
        "risk_mixed_pct",
        "nifty_return_20d_mean",
        "india_vix_close_mean",
    ]

    print(
        market_report[summary_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("PHASE 15 PART 2 COMPLETED SUCCESSFULLY")
    print("=" * 90)

    for name, path in reports.items():
        print(f"{name}: {path}")

    print("\nProtected production/V4 artifacts: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("Existing benchmark reports: NOT MODIFIED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
