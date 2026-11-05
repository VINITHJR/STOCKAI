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


ROOT = Path("backend/app/ml/phase15")
REPORT_DIR = ROOT / "reports"

SOURCE_RUN_ID = "20261031_131021"

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

# Original benchmark boundaries and sample counts.
# Abort rather than silently comparing different folds.
EXPECTED_FOLDS = {
    1: {
        "train_start": "2024-01-09",
        "train_end": "2025-01-02",
        "validation_start": "2025-01-14",
        "validation_end": "2025-04-11",
        "train_rows": 4900,
        "validation_rows": 1220,
    },
    2: {
        "train_start": "2024-01-09",
        "train_end": "2025-04-01",
        "validation_start": "2025-04-15",
        "validation_end": "2025-07-10",
        "train_rows": 6120,
        "validation_rows": 1220,
    },
    3: {
        "train_start": "2024-01-09",
        "train_end": "2025-07-01",
        "validation_start": "2025-07-11",
        "validation_end": "2025-10-08",
        "train_rows": 7340,
        "validation_rows": 1220,
    },
    4: {
        "train_start": "2024-01-09",
        "train_end": "2025-09-26",
        "validation_start": "2025-10-09",
        "validation_end": "2026-01-05",
        "train_rows": 8560,
        "validation_rows": 1200,
    },
    5: {
        "train_start": "2024-01-09",
        "train_end": "2025-12-24",
        "validation_start": "2026-01-06",
        "validation_end": "2026-04-06",
        "train_rows": 9760,
        "validation_rows": 1200,
    },
}


def date_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def distribution_stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        raise ValueError("Cannot calculate distribution statistics: no finite values.")

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "positive_pct": float(np.mean(values > 0) * 100),
        "negative_pct": float(np.mean(values < 0) * 100),
        "zero_pct": float(np.mean(values == 0) * 100),
    }


def population_stability_index(
    training: np.ndarray,
    validation: np.ndarray,
    bins: int = 10,
) -> float:
    """Use quantile boundaries derived only from training observations."""

    if len(training) == 0 or len(validation) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    unique_edges = np.unique(np.quantile(training, quantiles))

    if unique_edges.size == 1:
        # A constant training feature cannot define useful quantile bins.
        return 0.0 if np.allclose(validation, unique_edges[0]) else float("nan")

    if unique_edges.size == 2:
        midpoint = float(np.mean(unique_edges))
        edges = np.array([-np.inf, midpoint, np.inf])
    else:
        edges = np.concatenate((
            [-np.inf],
            unique_edges[1:-1],
            [np.inf],
        ))

    training_counts = np.histogram(training, bins=edges)[0]
    validation_counts = np.histogram(validation, bins=edges)[0]

    smoothing = 0.5
    training_pct = (
        training_counts + smoothing
    ) / (
        training_counts.sum() + smoothing * len(training_counts)
    )

    validation_pct = (
        validation_counts + smoothing
    ) / (
        validation_counts.sum() + smoothing * len(validation_counts)
    )

    return float(np.sum(
        (validation_pct - training_pct)
        * np.log(validation_pct / training_pct)
    ))


def drift_level(psi: float) -> str:
    if not np.isfinite(psi):
        return "not_evaluable"
    if psi >= 0.25:
        return "high"
    if psi >= 0.10:
        return "moderate"
    return "low"


def validate_plan(plan) -> pd.DataFrame:
    if not hasattr(plan, "panel") or not hasattr(plan, "folds"):
        raise TypeError("build_real_data_plan did not return a compatible WalkForwardPlan.")

    panel = plan.panel

    required = {
        "instrument_key",
        "session_date",
        *FEATURES,
        *TARGETS.values(),
    }

    missing = required - set(panel.columns)

    if missing:
        raise ValueError(
            "Reconstructed panel is missing required columns: "
            + ", ".join(sorted(missing))
        )

    panel = panel.copy()
    panel["session_date"] = pd.to_datetime(
        panel["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if len(plan.folds) != 5:
        raise ValueError(f"Expected five folds, received {len(plan.folds)}.")

    holdout_indices = set(plan.final_holdout_indices)

    for fold in plan.folds:
        number = int(fold.fold_number)

        if number not in EXPECTED_FOLDS:
            raise ValueError(f"Unexpected fold number: {number}")

        expected = EXPECTED_FOLDS[number]

        train_indices = list(fold.train_indices)
        validation_indices = list(fold.validation_indices)

        if holdout_indices.intersection(train_indices):
            raise ValueError(f"Fold {number} training overlaps final holdout.")

        if holdout_indices.intersection(validation_indices):
            raise ValueError(f"Fold {number} validation overlaps final holdout.")

        training = panel.iloc[train_indices]
        validation = panel.iloc[validation_indices]

        actual = {
            "train_start": date_text(training["session_date"].min()),
            "train_end": date_text(training["session_date"].max()),
            "validation_start": date_text(validation["session_date"].min()),
            "validation_end": date_text(validation["session_date"].max()),
            "train_rows": len(training),
            "validation_rows": len(validation),
        }

        if actual != expected:
            raise ValueError(
                f"Fold {number} does not match the original benchmark.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}\n"
                "Stopping to avoid comparing different samples."
            )

    return panel


def reconcile_original_validation_targets(
    panel: pd.DataFrame,
    plan,
) -> list[dict]:
    """Verify rebuilt validation targets against the saved OOF actual returns."""

    reconciliations = []

    for experiment, target_column in TARGETS.items():
        prediction_path = REPORT_DIR / (
            f"phase15_benchmark_{SOURCE_RUN_ID}_{experiment}_oof_predictions.csv"
        )

        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        saved = pd.read_csv(
            prediction_path,
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

        saved["actual_return"] = pd.to_numeric(
            saved["actual_return"], errors="raise"
        )

        keys = ["fold_number", "instrument_key", "session_date"]

        consistency = saved.groupby(keys)["actual_return"].nunique()

        if (consistency > 1).any():
            raise ValueError(
                f"{experiment}: saved OOF models disagree on actual returns."
            )

        saved_unique = saved.drop_duplicates(keys)

        expected_parts = []

        for fold in plan.folds:
            validation = panel.iloc[list(fold.validation_indices)]

            part = validation[
                ["instrument_key", "session_date", target_column]
            ].copy()

            part["fold_number"] = int(fold.fold_number)
            part = part.rename(columns={target_column: "rebuilt_actual_return"})

            expected_parts.append(part)

        rebuilt = pd.concat(expected_parts, ignore_index=True)

        comparison = rebuilt.merge(
            saved_unique,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not (comparison["_merge"] == "both").all():
            raise ValueError(
                f"{experiment}: rebuilt and saved OOF validation samples differ."
            )

        difference = np.abs(
            comparison["rebuilt_actual_return"].to_numpy(dtype=float)
            - comparison["actual_return"].to_numpy(dtype=float)
        )

        max_difference = float(difference.max()) if len(difference) else 0.0

        if not np.allclose(
            comparison["rebuilt_actual_return"],
            comparison["actual_return"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"{experiment}: validation target values do not match "
                f"the saved benchmark. Maximum difference: {max_difference}"
            )

        reconciliations.append({
            "experiment": experiment,
            "target_column": target_column,
            "matched_validation_samples": int(len(comparison)),
            "maximum_absolute_difference": max_difference,
            "exact_within_tolerance": True,
        })

    return reconciliations


def build_target_distribution_report(
    panel: pd.DataFrame,
    plan,
) -> pd.DataFrame:
    rows = []

    for fold in plan.folds:
        number = int(fold.fold_number)

        training = panel.iloc[list(fold.train_indices)]
        validation = panel.iloc[list(fold.validation_indices)]

        for experiment, target_column in TARGETS.items():
            train_values = finite_values(training[target_column])
            validation_values = finite_values(validation[target_column])

            train_stats = distribution_stats(train_values)
            validation_stats = distribution_stats(validation_values)

            train_std = train_stats["std"]

            ks = ks_2samp(
                train_values,
                validation_values,
            ).statistic

            psi = population_stability_index(
                train_values,
                validation_values,
            )

            rows.append({
                "fold_number": number,
                "experiment": experiment,
                "target_column": target_column,
                "training_start": date_text(training["session_date"].min()),
                "training_end": date_text(training["session_date"].max()),
                "validation_start": date_text(validation["session_date"].min()),
                "validation_end": date_text(validation["session_date"].max()),
                "training_count": train_stats["count"],
                "validation_count": validation_stats["count"],
                "training_mean": train_stats["mean"],
                "validation_mean": validation_stats["mean"],
                "mean_shift_validation_minus_training": (
                    validation_stats["mean"] - train_stats["mean"]
                ),
                "training_median": train_stats["median"],
                "validation_median": validation_stats["median"],
                "median_shift_validation_minus_training": (
                    validation_stats["median"] - train_stats["median"]
                ),
                "training_std": train_std,
                "validation_std": validation_stats["std"],
                "validation_to_training_std_ratio": (
                    validation_stats["std"] / train_std
                    if train_std > 0 else np.nan
                ),
                "training_positive_pct": train_stats["positive_pct"],
                "validation_positive_pct": validation_stats["positive_pct"],
                "positive_pct_shift_points": (
                    validation_stats["positive_pct"] - train_stats["positive_pct"]
                ),
                "training_negative_pct": train_stats["negative_pct"],
                "validation_negative_pct": validation_stats["negative_pct"],
                "negative_pct_shift_points": (
                    validation_stats["negative_pct"] - train_stats["negative_pct"]
                ),
                "training_q05": train_stats["q05"],
                "validation_q05": validation_stats["q05"],
                "training_q95": train_stats["q95"],
                "validation_q95": validation_stats["q95"],
                "standardized_mean_shift": (
                    (validation_stats["mean"] - train_stats["mean"]) / train_std
                    if train_std > 0 else np.nan
                ),
                "ks_distance": float(ks),
                "psi": psi,
                "psi_level": drift_level(psi),
            })

    return pd.DataFrame(rows)


def build_feature_drift_report(
    panel: pd.DataFrame,
    plan,
) -> pd.DataFrame:
    rows = []

    for fold in plan.folds:
        number = int(fold.fold_number)

        training = panel.iloc[list(fold.train_indices)]
        validation = panel.iloc[list(fold.validation_indices)]

        for feature in FEATURES:
            train_series = pd.to_numeric(
                training[feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

            validation_series = pd.to_numeric(
                validation[feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

            train_missing_pct = float(train_series.isna().mean() * 100)
            validation_missing_pct = float(validation_series.isna().mean() * 100)

            train_values = finite_values(train_series)
            validation_values = finite_values(validation_series)

            if len(train_values) == 0 or len(validation_values) == 0:
                raise ValueError(
                    f"Fold {number}, feature {feature}: no usable observations."
                )

            train_stats = distribution_stats(train_values)
            validation_stats = distribution_stats(validation_values)

            pooled_std = np.sqrt(
                (train_stats["std"] ** 2 + validation_stats["std"] ** 2) / 2
            )

            standardized_mean_difference = (
                (validation_stats["mean"] - train_stats["mean"]) / pooled_std
                if pooled_std > 0 else np.nan
            )

            ks = ks_2samp(
                train_values,
                validation_values,
            ).statistic

            psi = population_stability_index(
                train_values,
                validation_values,
            )

            family = (
                "technical"
                if FEATURES.index(feature) < 18
                else "market_context"
            )

            rows.append({
                "fold_number": number,
                "feature": feature,
                "feature_family": family,
                "training_count": len(train_values),
                "validation_count": len(validation_values),
                "training_missing_pct": train_missing_pct,
                "validation_missing_pct": validation_missing_pct,
                "missing_pct_shift_points": (
                    validation_missing_pct - train_missing_pct
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
                "standardized_mean_difference": standardized_mean_difference,
                "training_q05": train_stats["q05"],
                "validation_q05": validation_stats["q05"],
                "training_q95": train_stats["q95"],
                "validation_q95": validation_stats["q95"],
                "ks_distance": float(ks),
                "psi": psi,
                "psi_level": drift_level(psi),
            })

    return pd.DataFrame(rows)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    protected_files = sorted(discover_protected_files())

    if len(protected_files) != 11:
        raise RuntimeError(
            f"Expected 11 protected production/V4 artifacts; "
            f"found {len(protected_files)}. Stopping."
        )

    protected_before = {
        str(path): fingerprint(path)
        for path in protected_files
    }

    print("=" * 85)
    print("PHASE 15: TARGET DISTRIBUTION AND FEATURE DRIFT ANALYSIS")
    print("=" * 85)
    print("Reconstructing the original walk-forward plan...")
    print("This step fetches historical data through the existing Upstox service.")

    plan = build_real_data_plan()
    panel = validate_plan(plan)

    print("\nOriginal fold boundaries and sample counts: MATCHED")

    print("\nReconciling reconstructed validation targets with saved OOF results...")
    reconciliation = reconcile_original_validation_targets(panel, plan)

    for item in reconciliation:
        print(
            f"{item['experiment']}: "
            f"{item['matched_validation_samples']} samples matched; "
            f"maximum absolute difference="
            f"{item['maximum_absolute_difference']:.12g}"
        )

    print("\nCalculating training-versus-validation target distributions...")
    target_report = build_target_distribution_report(panel, plan)

    print("Calculating drift across all 31 features and five folds...")
    feature_report = build_feature_drift_report(panel, plan)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_path = REPORT_DIR / (
        f"phase15_drift_{run_id}_target_distribution.csv"
    )
    feature_path = REPORT_DIR / (
        f"phase15_drift_{run_id}_feature_drift.csv"
    )
    summary_path = REPORT_DIR / (
        f"phase15_drift_{run_id}_summary.json"
    )

    for path in (target_path, feature_path, summary_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {path}"
            )

    target_report.to_csv(target_path, index=False)
    feature_report.to_csv(feature_path, index=False)

    summary = {
        "run_id": run_id,
        "source_benchmark_run_id": SOURCE_RUN_ID,
        "analysis": "training_vs_validation_distribution_and_feature_drift",
        "instrument_count": len(plan.instrument_keys),
        "panel_rows": len(panel),
        "feature_count": len(FEATURES),
        "fold_count": len(plan.folds),
        "target_columns": TARGETS,
        "target_report_rows": len(target_report),
        "feature_report_rows": len(feature_report),
        "validation_target_reconciliation": reconciliation,
        "final_holdout_used": False,
        "models_fitted": False,
        "production_artifacts_modified": False,
        "psi_interpretation": {
            "low": "PSI < 0.10",
            "moderate": "0.10 <= PSI < 0.25",
            "high": "PSI >= 0.25",
            "note": "Heuristic monitoring bands, not statistical significance tests.",
        },
        "ks_note": (
            "KS distance is reported descriptively. No p-values are used "
            "because stock-session observations are dependent."
        ),
        "files": {
            "target_distribution": target_path.name,
            "feature_drift": feature_path.name,
        },
        "protected_artifact_count": len(protected_files),
    }

    with summary_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    protected_after = {
        str(path): fingerprint(path)
        for path in protected_files
    }

    if protected_before != protected_after:
        raise RuntimeError(
            "Protected artifact fingerprints changed. Investigate immediately."
        )

    print("\n" + "=" * 85)
    print("TARGET DISTRIBUTION SUMMARY")
    print("=" * 85)

    display_columns = [
        "fold_number",
        "experiment",
        "training_mean",
        "validation_mean",
        "training_positive_pct",
        "validation_positive_pct",
        "training_negative_pct",
        "validation_negative_pct",
        "validation_to_training_std_ratio",
        "psi",
        "psi_level",
    ]

    print(
        target_report[display_columns]
        .round(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 85)
    print("TOP FEATURE DRIFT PER FOLD")
    print("=" * 85)

    for fold_number in sorted(feature_report["fold_number"].unique()):
        subset = feature_report[
            feature_report["fold_number"] == fold_number
        ].copy()

        subset = subset.sort_values(
            "psi",
            ascending=False,
            na_position="last",
        )

        print(f"\nFOLD {fold_number}")

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
                    "missing_pct_shift_points",
                ]
            ]
            .head(12)
            .round(5)
            .to_string(index=False)
        )

    print("\n" + "=" * 85)
    print("PHASE 15 DISTRIBUTION AND FEATURE DRIFT ANALYSIS PASSED")
    print("=" * 85)
    print("Target report:", target_path)
    print("Feature report:", feature_path)
    print("Summary:", summary_path)
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
