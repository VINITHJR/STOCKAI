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
MARKET_RUN_ID = "20261031_144125"

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

EXPECTED_FOLD_ROWS = {
    1: 1220,
    2: 1220,
    3: 1220,
    4: 1200,
    5: 1200,
}

EXPECTED_FOLD_DATES = {
    1: ("2025-01-14", "2025-04-11"),
    2: ("2025-04-15", "2025-07-10"),
    3: ("2025-07-11", "2025-10-08"),
    4: ("2025-10-09", "2026-01-05"),
    5: ("2026-01-06", "2026-04-06"),
}

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
]


def safe_corr(x, y, method="pearson") -> float:
    """Return correlation only when both variables have variation."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return float("nan")

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")

    return float(
        pd.Series(x).corr(pd.Series(y), method=method)
    )


def performance_metrics(group: pd.DataFrame) -> dict:
    """
    Descriptive regression and directional metrics.

    Bias convention:
        positive bias = predictions are higher than actual returns.
        negative bias = predictions are lower than actual returns.

    A prediction is neutral only when it is exactly zero.
    """

    actual = group["actual_return"].to_numpy(dtype=float)
    predicted = group["predicted_return"].to_numpy(dtype=float)

    if len(actual) == 0:
        raise ValueError("Cannot calculate metrics for an empty group.")

    error = predicted - actual

    pred_positive = predicted > 0
    pred_negative = predicted < 0
    pred_neutral = predicted == 0

    actual_positive = actual > 0
    actual_negative = actual < 0
    actual_neutral = actual == 0

    directional_rows = (
        (predicted != 0)
        & (actual != 0)
    )

    if directional_rows.any():
        directional_accuracy = float(
            (
                np.sign(predicted[directional_rows])
                == np.sign(actual[directional_rows])
            ).mean() * 100
        )
    else:
        directional_accuracy = float("nan")

    pred_std = float(np.std(predicted, ddof=1)) if len(predicted) > 1 else 0.0
    actual_std = float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0

    return {
        "sample_count": int(len(actual)),

        "actual_mean": float(np.mean(actual)),
        "actual_median": float(np.median(actual)),
        "actual_std": actual_std,

        "predicted_mean": float(np.mean(predicted)),
        "predicted_median": float(np.median(predicted)),
        "predicted_std": pred_std,

        "mean_prediction_bias": float(np.mean(error)),
        "median_prediction_bias": float(np.median(error)),
        "mean_absolute_bias": float(np.mean(np.abs(error))),

        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),

        "prediction_positive_pct": float(pred_positive.mean() * 100),
        "prediction_negative_pct": float(pred_negative.mean() * 100),
        "prediction_neutral_pct": float(pred_neutral.mean() * 100),

        "actual_positive_pct": float(actual_positive.mean() * 100),
        "actual_negative_pct": float(actual_negative.mean() * 100),
        "actual_neutral_pct": float(actual_neutral.mean() * 100),

        "directional_coverage_pct": float(
            (predicted != 0).mean() * 100
        ),
        "directional_accuracy_pct": directional_accuracy,

        "pearson_correlation": safe_corr(predicted, actual, "pearson"),
        "spearman_correlation": safe_corr(predicted, actual, "spearman"),
    }


def calibration_line(group: pd.DataFrame) -> dict:
    """
    Descriptive validation-only regression:

        actual_return = intercept + slope * predicted_return

    This does not create or apply a calibration correction.
    """

    actual = group["actual_return"].to_numpy(dtype=float)
    predicted = group["predicted_return"].to_numpy(dtype=float)

    if len(actual) < 2 or np.unique(predicted).size < 2:
        return {
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "calibration_fit_r2": np.nan,
        }

    design = np.column_stack([
        np.ones(len(predicted)),
        predicted,
    ])

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        actual,
        rcond=None,
    )

    intercept = float(coefficients[0])
    slope = float(coefficients[1])

    fitted = intercept + slope * predicted

    residual_ss = float(np.sum(np.square(actual - fitted)))
    total_ss = float(np.sum(np.square(actual - actual.mean())))

    r2 = (
        1.0 - residual_ss / total_ss
        if total_ss > 0
        else np.nan
    )

    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_fit_r2": float(r2),
    }


def load_and_validate_predictions() -> dict[str, pd.DataFrame]:
    results = {}
    reference_keys = None

    for experiment, expected_target in EXPERIMENTS.items():

        path = REPORT_DIR / (
            f"phase15_benchmark_{BENCHMARK_RUN_ID}_"
            f"{experiment}_oof_predictions.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path, usecols=PREDICTION_COLUMNS)

        if set(df["model"].unique()) != set(MODELS):
            raise ValueError(
                f"{experiment}: unexpected model set: "
                f"{sorted(df['model'].unique())}"
            )

        if set(df["target_column"].unique()) != {expected_target}:
            raise ValueError(
                f"{experiment}: unexpected target column."
            )

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
            raise ValueError(
                f"{experiment}: NaN or infinite prediction values detected."
            )

        if df.duplicated([
            "fold_number",
            "model",
            "instrument_key",
            "session_date",
        ]).any():
            raise ValueError(
                f"{experiment}: duplicate model/stock/session predictions."
            )

        if len(df) != 30300:
            raise ValueError(
                f"{experiment}: expected 30300 rows, got {len(df)}."
            )

        keys = set()

        for model in MODELS:
            model_rows = df[df["model"] == model]

            if len(model_rows) != 6060:
                raise ValueError(
                    f"{experiment}/{model}: expected 6060 observations, "
                    f"received {len(model_rows)}."
                )

            model_keys = set(
                zip(
                    model_rows["fold_number"],
                    model_rows["instrument_key"],
                    model_rows["session_date"],
                )
            )

            if keys and model_keys != keys:
                raise ValueError(
                    f"{experiment}/{model}: validation sample mismatch."
                )

            keys = model_keys

            for fold, expected_count in EXPECTED_FOLD_ROWS.items():
                fold_rows = model_rows[
                    model_rows["fold_number"] == fold
                ]

                if len(fold_rows) != expected_count:
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: "
                        f"expected {expected_count}, got {len(fold_rows)}."
                    )

                expected_start, expected_end = EXPECTED_FOLD_DATES[fold]

                actual_dates = (
                    fold_rows["session_date"].min(),
                    fold_rows["session_date"].max(),
                )

                if actual_dates != (expected_start, expected_end):
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: date mismatch."
                    )

        # Actual targets must be identical across models for each sample.
        actual_variation = df.groupby(
            ["fold_number", "instrument_key", "session_date"]
        )["actual_return"].nunique()

        if (actual_variation > 1).any():
            raise ValueError(
                f"{experiment}: different models have different actual returns."
            )

        if reference_keys is not None and keys != reference_keys:
            raise ValueError(
                f"{experiment}: sample keys differ from the other experiment."
            )

        reference_keys = keys
        results[experiment] = df

        print(
            f"{experiment}: {len(df)} rows; "
            f"5 models; 6060 observations per model; validation checks PASSED"
        )

    return results


def load_and_validate_market_context() -> pd.DataFrame:
    path = REPORT_DIR / (
        f"phase15_part2_{MARKET_RUN_ID}_daily_market_context.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    market = pd.read_csv(path, usecols=MARKET_COLUMNS)

    market["session_date"] = pd.to_datetime(
        market["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if market["session_date"].duplicated().any():
        raise ValueError(
            "Part 2 market file must contain exactly one row per session."
        )

    if market[MARKET_COLUMNS].isna().any().any():
        raise ValueError("Missing market-regime values detected.")

    valid_trends = {"bullish", "bearish", "mixed"}
    valid_risk = {"risk_on", "risk_off", "mixed"}

    if not set(market["nifty_trend_state"]).issubset(valid_trends):
        raise ValueError("Unexpected NIFTY trend-state values.")

    if not set(market["market_risk_state"]).issubset(valid_risk):
        raise ValueError("Unexpected market-risk-state values.")

    return market


def combine_predictions_with_market(
    predictions: dict[str, pd.DataFrame],
    market: pd.DataFrame,
) -> dict[str, pd.DataFrame]:

    combined = {}

    for experiment, df in predictions.items():

        joined = df.merge(
            market,
            on="session_date",
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        if not (joined["_merge"] == "both").all():
            missing_dates = sorted(
                joined.loc[
                    joined["_merge"] != "both",
                    "session_date",
                ].unique()
            )

            raise ValueError(
                f"{experiment}: market regime missing for dates "
                f"{missing_dates[:10]}"
            )

        joined = joined.drop(columns=["_merge"])

        combined[experiment] = joined

    return combined



def build_performance_reports(data: dict[str, pd.DataFrame]):
    fold_rows = []
    pooled_rows = []
    direction_rows = []
    calibration_bin_rows = []
    stock_rows = []
    regime_nifty_rows = []
    regime_risk_rows = []

    for experiment, df in data.items():

        # -----------------------------------------------------
        # 1. Fold-level performance and calibration
        # -----------------------------------------------------
        for (fold, model), group in df.groupby(
            ["fold_number", "model"]
        ):
            metrics = performance_metrics(group)
            line = calibration_line(group)

            fold_rows.append({
                "experiment": experiment,
                "fold_number": int(fold),
                "model": model,
                **metrics,
                **line,
            })

        # -----------------------------------------------------
        # 2. Pooled out-of-fold performance
        # -----------------------------------------------------
        for model, group in df.groupby("model"):
            metrics = performance_metrics(group)
            line = calibration_line(group)

            pooled_rows.append({
                "experiment": experiment,
                "model": model,
                **metrics,
                **line,
            })

        # -----------------------------------------------------
        # 3. Direction-conditioned realised returns
        # -----------------------------------------------------
        for (fold, model), group in df.groupby(
            ["fold_number", "model"]
        ):
            subset = group.copy()

            subset["prediction_direction"] = np.select(
                [
                    subset["predicted_return"] > 0,
                    subset["predicted_return"] < 0,
                ],
                [
                    "bullish",
                    "bearish",
                ],
                default="neutral",
            )

            for direction, direction_group in subset.groupby(
                "prediction_direction"
            ):
                metrics = performance_metrics(direction_group)

                direction_rows.append({
                    "experiment": experiment,
                    "fold_number": int(fold),
                    "model": model,
                    "prediction_direction": direction,
                    **metrics,
                })

        # -----------------------------------------------------
        # 4. Quantile calibration
        # -----------------------------------------------------
        # Kept completely separate from pooled model metrics.
        # Bins are descriptive and are not used to correct predictions.
        for (fold, model), group in df.groupby(
            ["fold_number", "model"]
        ):
            unique_predictions = int(
                group["predicted_return"].nunique()
            )

            if unique_predictions < 2:
                continue

            subset = group.copy()

            bin_count = min(10, unique_predictions)

            subset["prediction_bin"] = pd.qcut(
                subset["predicted_return"],
                q=bin_count,
                duplicates="drop",
            )

            for bin_label, bin_group in subset.groupby(
                "prediction_bin",
                observed=True,
                sort=True,
            ):
                actual = bin_group["actual_return"]
                predicted = bin_group["predicted_return"]

                calibration_bin_rows.append({
                    "experiment": experiment,
                    "fold_number": int(fold),
                    "model": model,
                    "prediction_bin": str(bin_label),
                    "sample_count": int(len(bin_group)),
                    "mean_predicted_return": float(predicted.mean()),
                    "median_predicted_return": float(predicted.median()),
                    "mean_actual_return": float(actual.mean()),
                    "median_actual_return": float(actual.median()),
                    "mean_prediction_bias": float(
                        (predicted - actual).mean()
                    ),
                    "mae": float(
                        np.abs(predicted - actual).mean()
                    ),
                    "actual_positive_pct": float(
                        (actual > 0).mean() * 100
                    ),
                })

        # -----------------------------------------------------
        # 5. Stock-level calibration
        # -----------------------------------------------------
        for scope, group_columns in (
            ("pooled_oof", ["model", "instrument_key"]),
            ("fold", ["fold_number", "model", "instrument_key"]),
        ):
            for keys, group in df.groupby(group_columns):
                if not isinstance(keys, tuple):
                    keys = (keys,)

                mapping = dict(zip(group_columns, keys))
                metrics = performance_metrics(group)

                stock_rows.append({
                    "experiment": experiment,
                    "scope": scope,
                    **mapping,
                    **metrics,
                })

        # -----------------------------------------------------
        # 6. NIFTY and market-risk regime performance
        # -----------------------------------------------------
        for regime_column, target_rows in (
            ("nifty_trend_state", regime_nifty_rows),
            ("market_risk_state", regime_risk_rows),
        ):
            for (fold, model, regime), group in df.groupby(
                ["fold_number", "model", regime_column]
            ):
                target_rows.append({
                    "experiment": experiment,
                    "fold_number": int(fold),
                    "model": model,
                    "regime": regime,
                    **performance_metrics(group),
                })

    return {
        "fold_metrics": pd.DataFrame(fold_rows),
        "pooled_metrics": pd.DataFrame(pooled_rows),
        "directional_calibration": pd.DataFrame(direction_rows),
        "calibration_bins": pd.DataFrame(calibration_bin_rows),
        "stock_calibration": pd.DataFrame(stock_rows),
        "nifty_regime_performance": pd.DataFrame(regime_nifty_rows),
        "risk_regime_performance": pd.DataFrame(regime_risk_rows),
    }


def build_discrimination_reports(data: dict[str, pd.DataFrame]):
    daily_rows = []

    for experiment, df in data.items():

        for (fold, model, session_date), group in df.groupby(
            ["fold_number", "model", "session_date"]
        ):
            predictions = group["predicted_return"].to_numpy(dtype=float)
            actual = group["actual_return"].to_numpy(dtype=float)

            if len(group) != 20:
                raise ValueError(
                    f"{experiment}/{model}/Fold {fold}/{session_date}: "
                    f"expected 20 stocks, got {len(group)}."
                )

            rank_ic = safe_corr(
                predictions,
                actual,
                method="spearman",
            )

            if np.unique(predictions).size >= 2:
                ranked = group.sort_values(
                    "predicted_return",
                    ascending=False,
                    kind="mergesort",
                )

                quartile_size = max(1, len(ranked) // 4)

                top = ranked.head(quartile_size)
                bottom = ranked.tail(quartile_size)

                top_mean = float(top["actual_return"].mean())
                bottom_mean = float(bottom["actual_return"].mean())

                top_bottom_spread = top_mean - bottom_mean
            else:
                top_mean = np.nan
                bottom_mean = np.nan
                top_bottom_spread = np.nan

            daily_rows.append({
                "experiment": experiment,
                "fold_number": int(fold),
                "model": model,
                "session_date": session_date,
                "nifty_trend_state": group["nifty_trend_state"].iloc[0],
                "market_risk_state": group["market_risk_state"].iloc[0],
                "cross_sectional_spearman_ic": rank_ic,
                "predicted_top_quartile_actual_mean": top_mean,
                "predicted_bottom_quartile_actual_mean": bottom_mean,
                "top_minus_bottom_actual_return": top_bottom_spread,
            })

    daily = pd.DataFrame(daily_rows)

    summary_rows = []

    for (experiment, fold, model), group in daily.groupby(
        ["experiment", "fold_number", "model"]
    ):
        ic = group["cross_sectional_spearman_ic"].dropna()
        spread = group["top_minus_bottom_actual_return"].dropna()

        summary_rows.append({
            "experiment": experiment,
            "fold_number": int(fold),
            "model": model,
            "session_count": int(len(group)),
            "valid_rank_ic_sessions": int(len(ic)),
            "mean_daily_spearman_ic": (
                float(ic.mean()) if len(ic) else np.nan
            ),
            "median_daily_spearman_ic": (
                float(ic.median()) if len(ic) else np.nan
            ),
            "positive_rank_ic_sessions_pct": (
                float((ic > 0).mean() * 100) if len(ic) else np.nan
            ),
            "valid_top_bottom_sessions": int(len(spread)),
            "mean_top_bottom_actual_return_spread": (
                float(spread.mean()) if len(spread) else np.nan
            ),
            "positive_top_bottom_spread_sessions_pct": (
                float((spread > 0).mean() * 100)
                if len(spread) else np.nan
            ),
        })

    return daily, pd.DataFrame(summary_rows)


def main() -> int:
    print("=" * 90)
    print("PHASE 15: PART 3 - MODEL PREDICTION CALIBRATION AND BIAS ANALYSIS")
    print("=" * 90)

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

    print("\n[1/6] Loading original out-of-fold prediction reports...")
    predictions = load_and_validate_predictions()

    print("\n[2/6] Loading Part 2 daily market-regime report...")
    market = load_and_validate_market_context()

    print("Daily market observations:", len(market))

    print("\n[3/6] Joining predictions to market regimes...")
    data = combine_predictions_with_market(predictions, market)

    for experiment, frame in data.items():
        print(
            f"{experiment}: {len(frame)} joined rows; "
            f"{frame['session_date'].nunique()} validation dates"
        )

    print("\n[4/6] Calculating bias, direction and calibration...")
    performance = build_performance_reports(data)

    print("[5/6] Calculating discrimination and regime-conditioned metrics...")
    daily_discrimination, discrimination_summary = (
        build_discrimination_reports(data)
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports = {
        "fold_metrics": REPORT_DIR / f"phase15_part3_{run_id}_fold_metrics.csv",
        "pooled_metrics": REPORT_DIR / f"phase15_part3_{run_id}_pooled_metrics.csv",
        "directional_calibration": REPORT_DIR / f"phase15_part3_{run_id}_directional_calibration.csv",
        "calibration_bins": REPORT_DIR / f"phase15_part3_{run_id}_calibration_bins.csv",
        "stock_calibration": REPORT_DIR / f"phase15_part3_{run_id}_stock_calibration.csv",
        "nifty_regime": REPORT_DIR / f"phase15_part3_{run_id}_nifty_regime_performance.csv",
        "risk_regime": REPORT_DIR / f"phase15_part3_{run_id}_risk_regime_performance.csv",
        "daily_discrimination": REPORT_DIR / f"phase15_part3_{run_id}_daily_discrimination.csv",
        "discrimination_summary": REPORT_DIR / f"phase15_part3_{run_id}_discrimination_summary.csv",
        "summary": REPORT_DIR / f"phase15_part3_{run_id}_summary.json",
    }

    for path in reports.values():
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {path}"
            )

    print("\n[6/6] Saving new read-only diagnostic reports...")

    performance["fold_metrics"].to_csv(
        reports["fold_metrics"], index=False
    )
    performance["pooled_metrics"].to_csv(
        reports["pooled_metrics"], index=False
    )
    performance["directional_calibration"].to_csv(
        reports["directional_calibration"], index=False
    )
    performance["calibration_bins"].to_csv(
        reports["calibration_bins"], index=False
    )
    performance["stock_calibration"].to_csv(
        reports["stock_calibration"], index=False
    )
    performance["nifty_regime_performance"].to_csv(
        reports["nifty_regime"], index=False
    )
    performance["risk_regime_performance"].to_csv(
        reports["risk_regime"], index=False
    )
    daily_discrimination.to_csv(
        reports["daily_discrimination"], index=False
    )
    discrimination_summary.to_csv(
        reports["discrimination_summary"], index=False
    )

    summary = {
        "run_id": run_id,
        "benchmark_run_id": BENCHMARK_RUN_ID,
        "market_context_run_id": MARKET_RUN_ID,
        "analysis": "prediction_calibration_bias_discrimination",
        "experiments": EXPERIMENTS,
        "models": MODELS,
        "validation_observations_per_model": 6060,
        "fold_count": 5,
        "instrument_count": 20,
        "final_holdout_used": False,
        "models_fitted": False,
        "calibration_correction_applied": False,
        "hyperparameters_changed": False,
        "production_artifacts_modified": False,
        "neutral_definition": (
            "A prediction is neutral only when predicted_return == 0 exactly. "
            "Small nonzero predictions are classified by their sign."
        ),
        "bias_definition": (
            "prediction bias = predicted_return - actual_return. "
            "Positive values indicate overprediction."
        ),
        "calibration_method": (
            "Descriptive validation-only conditional means, quantile bins "
            "and linear calibration slope/intercept. No correction is fitted "
            "or applied to future predictions."
        ),
        "discrimination_method": (
            "Daily cross-sectional Spearman rank correlation and "
            "top-quartile minus bottom-quartile realised return spread."
        ),
        "dependence_warning": (
            "Seven-session targets overlap across sessions and stocks "
            "are correlated. Pooled metrics and daily averages are "
            "descriptive, not independent-sample statistical tests."
        ),
        "market_regime_source": (
            "Part 2 daily market context report. NIFTY bullish/bearish/mixed "
            "and market risk_on/risk_off/mixed labels are descriptive rules."
        ),
        "report_files": {
            key: path.name for key, path in reports.items()
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
    # Console summary: pooled results and Fold 5
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("A. POOLED PREDICTION BIAS AND DIRECTION")
    print("=" * 90)

    display = [
        "experiment",
        "model",
        "actual_mean",
        "predicted_mean",
        "mean_prediction_bias",
        "mae",
        "rmse",
        "prediction_positive_pct",
        "prediction_negative_pct",
        "prediction_neutral_pct",
        "directional_accuracy_pct",
        "pearson_correlation",
        "spearman_correlation",
    ]

    print(
        performance["pooled_metrics"][display]
        .round(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("B. FOLD 5 PERFORMANCE")
    print("=" * 90)

    fold5 = performance["fold_metrics"][
        performance["fold_metrics"]["fold_number"] == 5
    ]

    print(
        fold5[
            [
                "experiment",
                "model",
                "actual_mean",
                "predicted_mean",
                "mean_prediction_bias",
                "prediction_positive_pct",
                "prediction_negative_pct",
                "directional_accuracy_pct",
                "calibration_slope",
                "spearman_correlation",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("C. NIFTY REGIME-CONDITIONED PERFORMANCE")
    print("=" * 90)

    fold5_regime = performance["nifty_regime_performance"][
        performance["nifty_regime_performance"]["fold_number"] == 5
    ]

    print(
        fold5_regime[
            [
                "experiment",
                "model",
                "regime",
                "sample_count",
                "actual_mean",
                "predicted_mean",
                "mean_prediction_bias",
                "mae",
                "directional_accuracy_pct",
            ]
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("D. DISCRIMINATION SUMMARY")
    print("=" * 90)

    print(
        discrimination_summary[
            discrimination_summary["fold_number"] == 5
        ].round(5).to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("PHASE 15 PART 3 COMPLETED SUCCESSFULLY")
    print("=" * 90)

    for name, path in reports.items():
        print(f"{name}: {path}")

    print("\nProtected production/V4 artifacts: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("Calibration corrections applied: NO")
    print("Hyperparameters changed: NO")
    print("Existing benchmark reports: NOT MODIFIED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
