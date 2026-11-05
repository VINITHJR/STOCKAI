from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.phase15.validation.real_data_benchmark import (
    discover_protected_files,
    fingerprint,
)


REPORT_DIR = Path("backend/app/ml/phase15/reports")

BENCHMARK_RUN_ID = "20261031_131021"
PART2_RUN_ID = "20261031_144125"
PART3_RUN_ID = "20261031_145936"

EXPERIMENTS = {
    "primary": "next_open_to_close_return_7s",
    "forecast": "close_to_close_return_7s",
}

MODELS = [
    "zero_return",
    "train_mean",
    "ridge",
    "random_forest",
    "extra_trees",
]

FOLD_DATES = {
    1: ("2025-01-14", "2025-04-11", 61),
    2: ("2025-04-15", "2025-07-10", 61),
    3: ("2025-07-11", "2025-10-08", 61),
    4: ("2025-10-09", "2026-01-05", 60),
    5: ("2026-01-06", "2026-04-06", 60),
}

RECENT_VIX_WINDOW = 120
EXPECTED_INSTRUMENTS = 20

PREDICTION_COLUMNS = [
    "fold_number",
    "model",
    "instrument_key",
    "session_date",
    "target_column",
    "actual_return",
    "predicted_return",
]

MARKET_COLUMNS = [
    "session_date",
    "nifty_trend_state",
    "market_risk_state",
    "india_vix_close",
    "india_vix_return_5d",
    "india_vix_close_vs_sma20",
]

REGIME_COLUMNS = {
    "nifty_trend": "nifty_trend_state",
    "market_risk": "market_risk_state",
    "vix_5d_direction": "vix_5d_state",
    "vix_vs_sma20": "vix_sma_state",
    "vix_level_vs_training_q75": "vix_level_state",
}

METRICS_TO_RECONCILE = [
    "sample_count",
    "actual_mean",
    "predicted_mean",
    "mean_prediction_bias",
    "mae",
    "directional_accuracy_pct",
]


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan

    return float(
        pd.Series(x).corr(pd.Series(y), method=method)
    )


def calculate_metrics(group: pd.DataFrame) -> dict:
    actual = group["actual_return"].to_numpy(dtype=float)
    predicted = group["predicted_return"].to_numpy(dtype=float)

    if len(actual) == 0:
        raise ValueError("Cannot calculate metrics on an empty group.")

    error = predicted - actual

    active = (predicted != 0) & (actual != 0)

    directional_accuracy = (
        float(
            (
                np.sign(predicted[active])
                == np.sign(actual[active])
            ).mean() * 100
        )
        if active.any()
        else np.nan
    )

    return {
        "sample_count": int(len(actual)),
        "date_count": int(group["session_date"].nunique()),
        "instrument_count": int(group["instrument_key"].nunique()),

        "actual_mean": float(actual.mean()),
        "actual_median": float(np.median(actual)),

        "predicted_mean": float(predicted.mean()),
        "predicted_median": float(np.median(predicted)),

        "mean_prediction_bias": float(error.mean()),
        "median_prediction_bias": float(np.median(error)),

        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),

        "prediction_positive_pct": float(
            (predicted > 0).mean() * 100
        ),
        "prediction_negative_pct": float(
            (predicted < 0).mean() * 100
        ),
        "prediction_neutral_pct": float(
            (predicted == 0).mean() * 100
        ),

        "directional_coverage_pct": float(
            (predicted != 0).mean() * 100
        ),
        "directional_accuracy_pct": directional_accuracy,

        "pearson_correlation": safe_corr(
            predicted, actual, "pearson"
        ),
        "spearman_correlation": safe_corr(
            predicted, actual, "spearman"
        ),
    }


def load_predictions() -> dict[str, pd.DataFrame]:
    results = {}
    reference_keys = None
    reference_instruments = None

    for experiment, target in EXPERIMENTS.items():

        path = REPORT_DIR / (
            f"phase15_benchmark_{BENCHMARK_RUN_ID}_"
            f"{experiment}_oof_predictions.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path, usecols=PREDICTION_COLUMNS)

        df["fold_number"] = pd.to_numeric(
            df["fold_number"], errors="raise"
        ).astype(int)

        df["actual_return"] = pd.to_numeric(
            df["actual_return"], errors="raise"
        )

        df["predicted_return"] = pd.to_numeric(
            df["predicted_return"], errors="raise"
        )

        df["session_date"] = pd.to_datetime(
            df["session_date"], errors="raise"
        ).dt.strftime("%Y-%m-%d")

        if not np.isfinite(
            df[["actual_return", "predicted_return"]].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{experiment}: invalid prediction values.")

        if set(df["model"].unique()) != set(MODELS):
            raise ValueError(f"{experiment}: model list mismatch.")

        if set(df["target_column"].unique()) != {target}:
            raise ValueError(f"{experiment}: target mismatch.")

        if len(df) != 30300:
            raise ValueError(
                f"{experiment}: expected 30300 rows; got {len(df)}."
            )

        keys_columns = [
            "fold_number",
            "model",
            "instrument_key",
            "session_date",
        ]

        if df.duplicated(keys_columns).any():
            raise ValueError(f"{experiment}: duplicate prediction rows.")

        instruments = set(df["instrument_key"].unique())

        if len(instruments) != EXPECTED_INSTRUMENTS:
            raise ValueError(
                f"{experiment}: expected 20 stocks; got {len(instruments)}."
            )

        if reference_instruments is None:
            reference_instruments = instruments
        elif instruments != reference_instruments:
            raise ValueError("Experiments have different stock universes.")

        keys = set(
            map(
                tuple,
                df[keys_columns].itertuples(index=False, name=None),
            )
        )

        if reference_keys is not None and keys != reference_keys:
            raise ValueError("Experiment validation samples differ.")

        reference_keys = keys

        actual_unique = df.groupby(
            ["fold_number", "instrument_key", "session_date"]
        )["actual_return"].nunique()

        if (actual_unique > 1).any():
            raise ValueError(
                f"{experiment}: actual returns differ across models."
            )

        for model in MODELS:
            model_rows = df[df["model"] == model]

            if len(model_rows) != 6060:
                raise ValueError(
                    f"{experiment}/{model}: expected 6060 rows."
                )

            for fold, (start, end, expected_sessions) in FOLD_DATES.items():
                sample = model_rows[
                    model_rows["fold_number"] == fold
                ]

                if len(sample) != expected_sessions * EXPECTED_INSTRUMENTS:
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: row count mismatch."
                    )

                dates = (
                    sample["session_date"].min(),
                    sample["session_date"].max(),
                )

                if dates != (start, end):
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: date mismatch."
                    )

                counts = sample.groupby("instrument_key").size()

                if not (counts == expected_sessions).all():
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: stock count mismatch."
                    )

        results[experiment] = df

        print(
            f"{experiment}: {len(df)} rows validated; "
            f"{len(instruments)} stocks; five folds."
        )

    return results


def load_market_context() -> pd.DataFrame:
    path = REPORT_DIR / (
        f"phase15_part2_{PART2_RUN_ID}_daily_market_context.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    market = pd.read_csv(path, usecols=MARKET_COLUMNS)

    market["session_date"] = pd.to_datetime(
        market["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if market["session_date"].duplicated().any():
        raise ValueError(
            "Market context must contain one row per session."
        )

    numeric_columns = [
        "india_vix_close",
        "india_vix_return_5d",
        "india_vix_close_vs_sma20",
    ]

    for column in numeric_columns:
        market[column] = pd.to_numeric(
            market[column], errors="raise"
        )

    if not np.isfinite(
        market[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Market context contains invalid VIX values.")

    if market[MARKET_COLUMNS].isna().any().any():
        raise ValueError("Market context contains missing required values.")

    if not set(market["nifty_trend_state"]).issubset(
        {"bullish", "bearish", "mixed"}
    ):
        raise ValueError("Unexpected NIFTY regime values.")

    if not set(market["market_risk_state"]).issubset(
        {"risk_on", "risk_off", "mixed"}
    ):
        raise ValueError("Unexpected market-risk values.")

    return market.sort_values("session_date").reset_index(drop=True)


def build_fold_market_context(
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    context_parts = []
    threshold_rows = []

    for fold, (start, end, expected_sessions) in FOLD_DATES.items():

        # Select only dates strictly before validation.
        historical = market[
            market["session_date"] < start
        ].sort_values("session_date")

        training_window = historical.tail(RECENT_VIX_WINDOW)

        if len(training_window) != RECENT_VIX_WINDOW:
            raise ValueError(
                f"Fold {fold}: insufficient preceding VIX training sessions."
            )

        threshold = float(
            training_window["india_vix_close"].quantile(0.75)
        )

        validation = market[
            (market["session_date"] >= start)
            & (market["session_date"] <= end)
        ].copy()

        if validation["session_date"].nunique() != expected_sessions:
            raise ValueError(
                f"Fold {fold}: unexpected validation market-date count."
            )

        validation["fold_number"] = fold
        validation["vix_5d_state"] = np.where(
            validation["india_vix_return_5d"] > 0,
            "rising",
            "flat_or_falling",
        )

        validation["vix_sma_state"] = np.where(
            validation["india_vix_close_vs_sma20"] > 0,
            "above_sma20",
            "at_or_below_sma20",
        )

        validation["vix_level_state"] = np.where(
            validation["india_vix_close"] > threshold,
            "elevated",
            "normal",
        )

        context_parts.append(validation)

        threshold_rows.append({
            "fold_number": fold,
            "training_window_sessions": RECENT_VIX_WINDOW,
            "training_window_start": training_window["session_date"].min(),
            "training_window_end": training_window["session_date"].max(),
            "validation_start": start,
            "validation_end": end,
            "training_vix_q75": threshold,
            "validation_vix_mean": float(
                validation["india_vix_close"].mean()
            ),
            "validation_vix_above_training_q75_pct": float(
                (validation["india_vix_close"] > threshold).mean() * 100
            ),
        })

    context = pd.concat(context_parts, ignore_index=True)

    if context.duplicated(["fold_number", "session_date"]).any():
        raise ValueError("Duplicate fold/date market context.")

    return context, pd.DataFrame(threshold_rows)


def load_daily_discrimination() -> pd.DataFrame:
    path = REPORT_DIR / (
        f"phase15_part3_{PART3_RUN_ID}_daily_discrimination.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    columns = [
        "experiment",
        "fold_number",
        "model",
        "session_date",
        "cross_sectional_spearman_ic",
        "top_minus_bottom_actual_return",
    ]

    daily = pd.read_csv(path, usecols=columns)

    daily["fold_number"] = pd.to_numeric(
        daily["fold_number"], errors="raise"
    ).astype(int)

    daily["session_date"] = pd.to_datetime(
        daily["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    for column in (
        "cross_sectional_spearman_ic",
        "top_minus_bottom_actual_return",
    ):
        daily[column] = pd.to_numeric(
            daily[column], errors="coerce"
        )

    if len(daily) != 3030:
        raise ValueError(
            f"Expected 3030 daily discrimination rows; got {len(daily)}."
        )

    if daily.duplicated([
        "experiment",
        "fold_number",
        "model",
        "session_date",
    ]).any():
        raise ValueError("Duplicate daily discrimination rows.")

    return daily


def build_regime_performance(
    joined: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for experiment, df in joined.items():

        for regime_type, regime_column in REGIME_COLUMNS.items():

            # Individual fold performance.
            for (fold, model, regime), group in df.groupby(
                ["fold_number", "model", regime_column]
            ):
                rows.append({
                    "experiment": experiment,
                    "scope": "fold",
                    "fold_number": int(fold),
                    "model": model,
                    "regime_type": regime_type,
                    "regime": regime,
                    **calculate_metrics(group),
                })

            # Pooled OOF performance across all five folds.
            for (model, regime), group in df.groupby(
                ["model", regime_column]
            ):
                rows.append({
                    "experiment": experiment,
                    "scope": "pooled_oof",
                    "fold_number": "ALL",
                    "model": model,
                    "regime_type": regime_type,
                    "regime": regime,
                    **calculate_metrics(group),
                })

    return pd.DataFrame(rows)


def build_regime_sample_sizes(context: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for regime_type, column in REGIME_COLUMNS.items():

        for fold, group in context.groupby("fold_number"):
            for regime, sample in group.groupby(column):
                rows.append({
                    "scope": "fold",
                    "fold_number": int(fold),
                    "regime_type": regime_type,
                    "regime": regime,
                    "validation_dates": int(
                        sample["session_date"].nunique()
                    ),
                    "stock_observations": int(
                        sample["session_date"].nunique()
                        * EXPECTED_INSTRUMENTS
                    ),
                    "minimum_dates_for_descriptive_review": 20,
                    "sample_warning": (
                        "LOW_DATE_COUNT"
                        if sample["session_date"].nunique() < 20
                        else "DESCRIPTIVE_ONLY"
                    ),
                })

        for regime, sample in context.groupby(column):
            dates = int(sample["session_date"].nunique())

            rows.append({
                "scope": "pooled_oof",
                "fold_number": "ALL",
                "regime_type": regime_type,
                "regime": regime,
                "validation_dates": dates,
                "stock_observations": dates * EXPECTED_INSTRUMENTS,
                "minimum_dates_for_descriptive_review": 20,
                "sample_warning": (
                    "LOW_DATE_COUNT"
                    if dates < 20
                    else "DESCRIPTIVE_ONLY"
                ),
            })

    return pd.DataFrame(rows)


def build_discrimination_by_regime(
    daily: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:

    joined = daily.merge(
        context[
            [
                "fold_number",
                "session_date",
                *REGIME_COLUMNS.values(),
            ]
        ],
        on=["fold_number", "session_date"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not (joined["_merge"] == "both").all():
        raise ValueError(
            "Daily discrimination data could not be fully matched to market regimes."
        )

    joined = joined.drop(columns=["_merge"])
    rows = []

    for regime_type, column in REGIME_COLUMNS.items():

        for (experiment, fold, model, regime), group in joined.groupby(
            ["experiment", "fold_number", "model", column]
        ):
            ic = group["cross_sectional_spearman_ic"].dropna()
            spread = group["top_minus_bottom_actual_return"].dropna()

            rows.append({
                "experiment": experiment,
                "scope": "fold",
                "fold_number": int(fold),
                "model": model,
                "regime_type": regime_type,
                "regime": regime,
                "validation_dates": int(
                    group["session_date"].nunique()
                ),
                "valid_rank_ic_dates": int(len(ic)),
                "mean_daily_spearman_ic": (
                    float(ic.mean()) if len(ic) else np.nan
                ),
                "median_daily_spearman_ic": (
                    float(ic.median()) if len(ic) else np.nan
                ),
                "positive_rank_ic_dates_pct": (
                    float((ic > 0).mean() * 100)
                    if len(ic) else np.nan
                ),
                "mean_top_bottom_spread": (
                    float(spread.mean()) if len(spread) else np.nan
                ),
                "positive_spread_dates_pct": (
                    float((spread > 0).mean() * 100)
                    if len(spread) else np.nan
                ),
            })

        for (experiment, model, regime), group in joined.groupby(
            ["experiment", "model", column]
        ):
            ic = group["cross_sectional_spearman_ic"].dropna()
            spread = group["top_minus_bottom_actual_return"].dropna()

            rows.append({
                "experiment": experiment,
                "scope": "pooled_oof",
                "fold_number": "ALL",
                "model": model,
                "regime_type": regime_type,
                "regime": regime,
                "validation_dates": int(
                    group["session_date"].nunique()
                ),
                "valid_rank_ic_dates": int(len(ic)),
                "mean_daily_spearman_ic": (
                    float(ic.mean()) if len(ic) else np.nan
                ),
                "median_daily_spearman_ic": (
                    float(ic.median()) if len(ic) else np.nan
                ),
                "positive_rank_ic_dates_pct": (
                    float((ic > 0).mean() * 100)
                    if len(ic) else np.nan
                ),
                "mean_top_bottom_spread": (
                    float(spread.mean()) if len(spread) else np.nan
                ),
                "positive_spread_dates_pct": (
                    float((spread > 0).mean() * 100)
                    if len(spread) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def reconcile_previous_part3_reports(
    performance: pd.DataFrame,
) -> list[dict]:

    checks = []

    sources = {
        "nifty_trend": "nifty_regime_performance",
        "market_risk": "risk_regime_performance",
    }

    compare_columns = [
        "sample_count",
        "actual_mean",
        "predicted_mean",
        "mean_prediction_bias",
        "mae",
        "directional_accuracy_pct",
    ]

    for regime_type, filename_part in sources.items():

        path = REPORT_DIR / (
            f"phase15_part3_{PART3_RUN_ID}_{filename_part}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        saved = pd.read_csv(path)
        required = {
            "experiment",
            "fold_number",
            "model",
            "regime",
            *compare_columns,
        }

        missing = required - set(saved.columns)

        if missing:
            raise ValueError(
                f"{path.name}: missing columns {sorted(missing)}"
            )

        calculated = performance[
            (performance["scope"] == "fold")
            & (performance["regime_type"] == regime_type)
        ].copy()

        keys = [
            "experiment",
            "fold_number",
            "model",
            "regime",
        ]

        merged = saved.merge(
            calculated,
            on=keys,
            how="outer",
            suffixes=("_saved", "_recalculated"),
            indicator=True,
            validate="one_to_one",
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(
                f"Existing {regime_type} report has different regime groups."
            )

        for column in compare_columns:
            a = pd.to_numeric(
                merged[f"{column}_saved"], errors="coerce"
            ).to_numpy(dtype=float)

            b = pd.to_numeric(
                merged[f"{column}_recalculated"], errors="coerce"
            ).to_numpy(dtype=float)

            if not np.allclose(
                a, b,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{regime_type} reconciliation failed for {column}."
                )

        checks.append({
            "source_report": path.name,
            "matched_groups": int(len(merged)),
            "metrics_checked": len(compare_columns),
            "passed": True,
        })

    return checks


def main() -> int:

    print("=" * 100)
    print("PHASE 15 PART 3.5 - REGIME-CONDITIONED PERFORMANCE")
    print("=" * 100)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    protected_files = sorted(discover_protected_files())

    if len(protected_files) != 11:
        raise RuntimeError(
            f"Expected 11 protected artifacts; found {len(protected_files)}."
        )

    protected_before = {
        str(path): fingerprint(path)
        for path in protected_files
    }

    print("\n[1/8] Loading saved out-of-fold predictions...")
    predictions = load_predictions()

    print("\n[2/8] Loading Part 2 daily market context...")
    market = load_market_context()
    print("Available daily market observations:", len(market))

    print("\n[3/8] Building fold-specific VIX thresholds...")
    context, thresholds = build_fold_market_context(market)

    print("Validation market-context dates:", len(context))
    print("VIX thresholds use only preceding 120 training sessions.")

    print("\n[4/8] Joining prediction data to market regimes...")
    joined = {}

    for experiment, df in predictions.items():

        combined = df.merge(
            context,
            on=["fold_number", "session_date"],
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        if not (combined["_merge"] == "both").all():
            raise ValueError(
                f"{experiment}: market context missing for prediction dates."
            )

        combined = combined.drop(columns=["_merge"])

        if len(combined) != 30300:
            raise ValueError(
                f"{experiment}: unexpected joined row count."
            )

        joined[experiment] = combined

        print(
            f"{experiment}: {len(combined)} rows joined successfully."
        )

    print("\n[5/8] Calculating performance by regime...")
    performance = build_regime_performance(joined)
    sample_sizes = build_regime_sample_sizes(context)

    print("Performance report rows:", len(performance))
    print("Regime sample-size rows:", len(sample_sizes))

    print("\nReconciling previous NIFTY and market-risk reports...")
    reconciliation = reconcile_previous_part3_reports(performance)
    print(json.dumps(reconciliation, indent=2))

    print("\n[6/8] Calculating discrimination by regime...")
    daily_discrimination = load_daily_discrimination()

    discrimination = build_discrimination_by_regime(
        daily_discrimination,
        context,
    )

    print("Discrimination report rows:", len(discrimination))

    print("\n[7/8] Preparing new report files...")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports = {
        "regime_performance": REPORT_DIR / (
            f"phase15_part3_5_{run_id}_regime_performance.csv"
        ),
        "discrimination": REPORT_DIR / (
            f"phase15_part3_5_{run_id}_discrimination_by_regime.csv"
        ),
        "sample_sizes": REPORT_DIR / (
            f"phase15_part3_5_{run_id}_regime_sample_sizes.csv"
        ),
        "vix_thresholds": REPORT_DIR / (
            f"phase15_part3_5_{run_id}_vix_thresholds.csv"
        ),
        "summary": REPORT_DIR / (
            f"phase15_part3_5_{run_id}_summary.json"
        ),
    }

    for path in reports.values():
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {path}"
            )

    performance.to_csv(reports["regime_performance"], index=False)
    discrimination.to_csv(reports["discrimination"], index=False)
    sample_sizes.to_csv(reports["sample_sizes"], index=False)
    thresholds.to_csv(reports["vix_thresholds"], index=False)

    summary = {
        "run_id": run_id,
        "benchmark_run_id": BENCHMARK_RUN_ID,
        "part2_run_id": PART2_RUN_ID,
        "part3_run_id": PART3_RUN_ID,
        "analysis": "regime_conditioned_performance",
        "experiments": EXPERIMENTS,
        "models": MODELS,
        "folds": 5,
        "stocks": EXPECTED_INSTRUMENTS,
        "validation_dates": int(context["session_date"].nunique()),
        "vix_level_threshold": (
            "Per-fold 75th percentile calculated on the preceding "
            "120 training sessions only."
        ),
        "vix_direction": (
            "Rising when india_vix_return_5d > 0; otherwise flat_or_falling."
        ),
        "vix_sma_state": (
            "Above SMA20 when india_vix_close_vs_sma20 > 0."
        ),
        "regime_definitions": (
            "NIFTY and market-risk classifications are taken unchanged "
            "from the Part 2 descriptive rules."
        ),
        "validation_reconciliation": reconciliation,
        "final_holdout_used": False,
        "models_fitted": False,
        "hyperparameters_changed": False,
        "calibration_corrections_applied": False,
        "production_artifacts_modified": False,
        "statistical_warning": (
            "All regime summaries are descriptive. Seven-session targets "
            "overlap, stock returns are cross-correlated, and observations "
            "within market regimes are not independent. The 20-date warning "
            "is a descriptive sample-size screen, not a statistical power test."
        ),
        "reports": {
            name: path.name for name, path in reports.items()
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

    # ---------------------------------------------------------
    # Console summaries
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("A. PERFORMANCE BY NIFTY TREND")
    print("=" * 100)

    nifty = performance[
        (performance["scope"] == "pooled_oof")
        & (performance["regime_type"] == "nifty_trend")
    ]

    print(
        nifty[
            [
                "experiment", "model", "regime",
                "date_count", "sample_count",
                "actual_mean", "predicted_mean",
                "mean_prediction_bias", "mae",
                "directional_accuracy_pct",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("B. PERFORMANCE BY MARKET RISK STATE")
    print("=" * 100)

    risk = performance[
        (performance["scope"] == "pooled_oof")
        & (performance["regime_type"] == "market_risk")
    ]

    print(
        risk[
            [
                "experiment", "model", "regime",
                "date_count", "sample_count",
                "actual_mean", "predicted_mean",
                "mean_prediction_bias", "mae",
                "directional_accuracy_pct",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("C. PERFORMANCE WHEN INDIA VIX IS RISING")
    print("=" * 100)

    vix_rising = performance[
        (performance["scope"] == "pooled_oof")
        & (performance["regime_type"] == "vix_5d_direction")
    ]

    print(
        vix_rising[
            [
                "experiment", "model", "regime",
                "date_count", "sample_count",
                "mean_prediction_bias", "mae",
                "directional_accuracy_pct",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("D. PERFORMANCE UNDER ELEVATED VIX")
    print("=" * 100)

    vix_level = performance[
        (performance["scope"] == "pooled_oof")
        & (performance["regime_type"] == "vix_level_vs_training_q75")
    ]

    print(
        vix_level[
            [
                "experiment", "model", "regime",
                "date_count", "sample_count",
                "mean_prediction_bias", "mae",
                "directional_accuracy_pct",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("E. DISCRIMINATION BY MARKET RISK STATE")
    print("=" * 100)

    rank_risk = discrimination[
        (discrimination["scope"] == "pooled_oof")
        & (discrimination["regime_type"] == "market_risk")
    ]

    print(
        rank_risk[
            [
                "experiment", "model", "regime",
                "validation_dates",
                "mean_daily_spearman_ic",
                "positive_rank_ic_dates_pct",
                "mean_top_bottom_spread",
                "positive_spread_dates_pct",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("F. REGIME SAMPLE SIZES")
    print("=" * 100)

    pooled_sizes = sample_sizes[
        sample_sizes["scope"] == "pooled_oof"
    ]

    print(
        pooled_sizes[
            [
                "regime_type", "regime",
                "validation_dates",
                "stock_observations",
                "sample_warning",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("G. FOLD 5: ELEVATED VIX VERSUS NORMAL VIX")
    print("=" * 100)

    fold5_vix = performance[
        (performance["scope"] == "fold")
        & (performance["fold_number"] == 5)
        & (performance["regime_type"] == "vix_level_vs_training_q75")
    ]

    if not fold5_vix.empty:
        print(
            fold5_vix[
                [
                    "experiment", "model", "regime",
                    "date_count", "sample_count",
                    "actual_mean", "predicted_mean",
                    "mean_prediction_bias", "mae",
                    "directional_accuracy_pct",
                ]
            ].round(5).to_string(index=False)
        )
    else:
        print("No Fold 5 VIX groups available.")

    print("\n" + "=" * 100)
    print("PHASE 15 PART 3.5 COMPLETED SUCCESSFULLY")
    print("=" * 100)

    for name, path in reports.items():
        print(f"{name}: {path}")

    print("\nPrevious NIFTY/risk reports: RECONCILED")
    print("Protected production/V4 artifacts: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("Calibration corrections: NONE")
    print("Hyperparameters changed: NO")
    print("Existing benchmark reports: NOT MODIFIED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
