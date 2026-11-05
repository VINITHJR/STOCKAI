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

ML_MODELS = [
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

EXPECTED_FOLD_SESSIONS = {
    1: 61,
    2: 61,
    3: 61,
    4: 60,
    5: 60,
}

EXPECTED_DATES = {
    1: ("2025-01-14", "2025-04-11"),
    2: ("2025-04-15", "2025-07-10"),
    3: ("2025-07-11", "2025-10-08"),
    4: ("2025-10-09", "2026-01-05"),
    5: ("2026-01-06", "2026-04-06"),
}

BIAS_TOLERANCE = 1e-12

INPUT_COLUMNS = [
    "fold_number",
    "model",
    "instrument_key",
    "session_date",
    "target_column",
    "actual_return",
    "predicted_return",
]

METRICS_TO_RECONCILE = [
    "sample_count",
    "actual_mean",
    "actual_median",
    "predicted_mean",
    "predicted_median",
    "mean_prediction_bias",
    "median_prediction_bias",
    "mae",
    "rmse",
    "directional_accuracy_pct",
]


def correlation(x, y, method="pearson"):
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

    if not len(actual):
        raise ValueError("Empty stock/model/fold sample.")

    error = predicted - actual

    active = (predicted != 0) & (actual != 0)

    directional_accuracy = (
        float(
            (np.sign(predicted[active]) == np.sign(actual[active])).mean()
            * 100
        )
        if active.any()
        else np.nan
    )

    return {
        "sample_count": int(len(actual)),

        "actual_mean": float(np.mean(actual)),
        "actual_median": float(np.median(actual)),
        "actual_std": (
            float(np.std(actual, ddof=1))
            if len(actual) > 1 else 0.0
        ),

        "predicted_mean": float(np.mean(predicted)),
        "predicted_median": float(np.median(predicted)),
        "predicted_std": (
            float(np.std(predicted, ddof=1))
            if len(predicted) > 1 else 0.0
        ),

        "mean_prediction_bias": float(np.mean(error)),
        "median_prediction_bias": float(np.median(error)),
        "mae": float(np.mean(np.abs(error))),
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

        "actual_positive_pct": float(
            (actual > 0).mean() * 100
        ),
        "actual_negative_pct": float(
            (actual < 0).mean() * 100
        ),

        "directional_coverage_pct": float(
            (predicted != 0).mean() * 100
        ),
        "directional_accuracy_pct": directional_accuracy,

        "pearson_correlation": correlation(predicted, actual, "pearson"),
        "spearman_correlation": correlation(predicted, actual, "spearman"),
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

        df = pd.read_csv(path, usecols=INPUT_COLUMNS)

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
            raise ValueError(f"{experiment}: non-finite return values.")

        if set(df["model"].unique()) != set(MODELS):
            raise ValueError(f"{experiment}: model list mismatch.")

        if set(df["target_column"].unique()) != {target}:
            raise ValueError(f"{experiment}: target column mismatch.")

        if len(df) != 30300:
            raise ValueError(
                f"{experiment}: expected 30300 rows; received {len(df)}."
            )

        if df.duplicated([
            "fold_number",
            "model",
            "instrument_key",
            "session_date",
        ]).any():
            raise ValueError(f"{experiment}: duplicate prediction samples.")

        instruments = set(df["instrument_key"].unique())

        if len(instruments) != 20:
            raise ValueError(
                f"{experiment}: expected 20 instruments; found {len(instruments)}."
            )

        if reference_instruments is None:
            reference_instruments = instruments
        elif instruments != reference_instruments:
            raise ValueError(
                "Primary and forecast instrument universes differ."
            )

        key_columns = [
            "fold_number",
            "model",
            "instrument_key",
            "session_date",
        ]

        keys = set(map(tuple, df[key_columns].itertuples(
            index=False, name=None
        )))

        if reference_keys is not None and keys != reference_keys:
            raise ValueError(
                f"{experiment}: sample keys differ from the other experiment."
            )

        reference_keys = keys

        actual_variation = df.groupby([
            "fold_number",
            "instrument_key",
            "session_date",
        ])["actual_return"].nunique()

        if (actual_variation > 1).any():
            raise ValueError(
                f"{experiment}: actual return differs across models."
            )

        for model in MODELS:
            model_rows = df[df["model"] == model]

            if len(model_rows) != 6060:
                raise ValueError(
                    f"{experiment}/{model}: expected 6060 rows."
                )

            for fold, expected_total in EXPECTED_FOLD_ROWS.items():
                fold_rows = model_rows[
                    model_rows["fold_number"] == fold
                ]

                if len(fold_rows) != expected_total:
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: "
                        f"row count mismatch."
                    )

                expected_start, expected_end = EXPECTED_DATES[fold]

                if (
                    fold_rows["session_date"].min(),
                    fold_rows["session_date"].max(),
                ) != (expected_start, expected_end):
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: date mismatch."
                    )

                per_stock_counts = fold_rows.groupby(
                    "instrument_key"
                ).size()

                if not (per_stock_counts == EXPECTED_FOLD_SESSIONS[fold]).all():
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: "
                        "stock session counts are inconsistent."
                    )

        if model == "zero_return":
            zero_rows = df[df["model"] == "zero_return"]

            if not np.all(
                zero_rows["predicted_return"].to_numpy(dtype=float) == 0
            ):
                raise ValueError(
                    f"{experiment}: zero_return predictions are not zero."
                )

        results[experiment] = df

        print(
            f"{experiment}: validated {len(df)} rows, "
            f"20 stocks, 5 models and 5 folds."
        )

    return results


def build_stock_metrics(data):
    fold_rows = []
    pooled_rows = []

    for experiment, df in data.items():

        for (fold, model, instrument), group in df.groupby(
            ["fold_number", "model", "instrument_key"]
        ):
            fold_rows.append({
                "experiment": experiment,
                "fold_number": int(fold),
                "model": model,
                "instrument_key": instrument,
                **calculate_metrics(group),
            })

        for (model, instrument), group in df.groupby(
            ["model", "instrument_key"]
        ):
            pooled_rows.append({
                "experiment": experiment,
                "model": model,
                "instrument_key": instrument,
                **calculate_metrics(group),
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(pooled_rows)


def reconcile_saved_part3_report(
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
) -> dict:
    path = REPORT_DIR / (
        f"phase15_part3_{PART3_RUN_ID}_stock_calibration.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    saved = pd.read_csv(path)

    required = {
        "experiment",
        "scope",
        "model",
        "instrument_key",
        "sample_count",
        *METRICS_TO_RECONCILE,
    }

    missing = required - set(saved.columns)

    if missing:
        raise ValueError(
            "Saved stock report missing columns: "
            + ", ".join(sorted(missing))
        )

    if len(saved) != 1200:
        raise ValueError(
            f"Expected 1200 saved stock rows; received {len(saved)}."
        )

    if set(saved["scope"].unique()) != {"pooled_oof", "fold"}:
        raise ValueError("Unexpected saved stock report scopes.")

    comparisons = []

    for scope in ("fold", "pooled_oof"):

        saved_part = saved[saved["scope"] == scope].copy()

        if scope == "fold":
            expected_part = fold_metrics.copy()
            keys = [
                "experiment",
                "fold_number",
                "model",
                "instrument_key",
            ]
        else:
            expected_part = pooled_metrics.copy()
            keys = [
                "experiment",
                "model",
                "instrument_key",
            ]

        merged = saved_part.merge(
            expected_part,
            on=keys,
            how="outer",
            suffixes=("_saved", "_recomputed"),
            indicator=True,
            validate="one_to_one",
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(
                f"Saved stock report sample keys do not match: {scope}."
            )

        for column in METRICS_TO_RECONCILE:
            left = pd.to_numeric(
                merged[f"{column}_saved"], errors="coerce"
            ).to_numpy(dtype=float)

            right = pd.to_numeric(
                merged[f"{column}_recomputed"], errors="coerce"
            ).to_numpy(dtype=float)

            if not np.allclose(
                left,
                right,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Stock report reconciliation failed for {scope}, "
                    f"metric {column}."
                )

        comparisons.append({
            "scope": scope,
            "matched_rows": int(len(merged)),
            "metrics_verified": len(METRICS_TO_RECONCILE),
            "passed": True,
        })

    return {
        "source_report": path.name,
        "saved_rows": int(len(saved)),
        "reconciliation": comparisons,
        "passed": True,
    }


def classify_bias(value):
    if value > BIAS_TOLERANCE:
        return "overprediction"
    if value < -BIAS_TOLERANCE:
        return "underprediction"
    return "approximately_unbiased"


def build_bias_consistency(pooled_metrics, fold_metrics):
    rows = []

    for (experiment, model, instrument), group in fold_metrics.groupby(
        ["experiment", "model", "instrument_key"]
    ):
        group = group.sort_values("fold_number")

        if group["fold_number"].nunique() != 5:
            raise ValueError(
                f"Missing folds for {experiment}/{model}/{instrument}."
            )

        biases = group["mean_prediction_bias"].to_numpy(dtype=float)
        classifications = [classify_bias(value) for value in biases]

        positive_count = classifications.count("overprediction")
        negative_count = classifications.count("underprediction")
        neutral_count = classifications.count("approximately_unbiased")

        if positive_count == 5:
            pattern = "consistent_overprediction_5_of_5"
        elif negative_count == 5:
            pattern = "consistent_underprediction_5_of_5"
        elif positive_count >= 4:
            pattern = "predominantly_overprediction"
        elif negative_count >= 4:
            pattern = "predominantly_underprediction"
        else:
            pattern = "mixed_or_near_unbiased"

        pooled_row = pooled_metrics[
            (pooled_metrics["experiment"] == experiment)
            & (pooled_metrics["model"] == model)
            & (pooled_metrics["instrument_key"] == instrument)
        ]

        if len(pooled_row) != 1:
            raise ValueError(
                f"Pooled stock metric missing for {instrument}."
            )

        pooled_row = pooled_row.iloc[0]

        rows.append({
            "experiment": experiment,
            "model": model,
            "instrument_key": instrument,
            "bias_pattern": pattern,
            "overprediction_folds": positive_count,
            "underprediction_folds": negative_count,
            "approximately_unbiased_folds": neutral_count,
            "fold1_bias": biases[0],
            "fold2_bias": biases[1],
            "fold3_bias": biases[2],
            "fold4_bias": biases[3],
            "fold5_bias": biases[4],
            "pooled_mean_prediction_bias": pooled_row["mean_prediction_bias"],
            "pooled_mae": pooled_row["mae"],
            "pooled_directional_accuracy_pct": (
                pooled_row["directional_accuracy_pct"]
            ),
        })

    return pd.DataFrame(rows)


def build_extra_trees_comparison(fold_metrics):
    rows = []

    for experiment in EXPERIMENTS:

        selected = fold_metrics[
            (fold_metrics["experiment"] == experiment)
            & (
                fold_metrics["model"].isin(
                    ["extra_trees", "zero_return", "train_mean"]
                )
            )
        ]

        wide = selected.pivot(
            index=["fold_number", "instrument_key"],
            columns="model",
            values="mae",
        ).reset_index()

        expected = {"extra_trees", "zero_return", "train_mean"}

        if not expected.issubset(wide.columns):
            raise ValueError(
                f"{experiment}: missing baseline/model stock MAE."
            )

        wide["improvement_vs_zero"] = (
            wide["zero_return"] - wide["extra_trees"]
        )

        wide["improvement_vs_train_mean"] = (
            wide["train_mean"] - wide["extra_trees"]
        )

        wide["beats_zero"] = (
            wide["improvement_vs_zero"] > BIAS_TOLERANCE
        )

        wide["beats_train_mean"] = (
            wide["improvement_vs_train_mean"] > BIAS_TOLERANCE
        )

        for instrument, group in wide.groupby("instrument_key"):
            rows.append({
                "experiment": experiment,
                "instrument_key": instrument,
                "fold_count": int(group["fold_number"].nunique()),
                "folds_beating_zero": int(group["beats_zero"].sum()),
                "folds_beating_train_mean": int(
                    group["beats_train_mean"].sum()
                ),
                "beats_zero_all_folds": bool(group["beats_zero"].all()),
                "beats_zero_at_least_4_folds": bool(
                    group["beats_zero"].sum() >= 4
                ),
                "mean_mae_improvement_vs_zero": float(
                    group["improvement_vs_zero"].mean()
                ),
                "median_mae_improvement_vs_zero": float(
                    group["improvement_vs_zero"].median()
                ),
                "mean_mae_improvement_vs_train_mean": float(
                    group["improvement_vs_train_mean"].mean()
                ),
                "fold5_mae_improvement_vs_zero": float(
                    group.loc[
                        group["fold_number"] == 5,
                        "improvement_vs_zero",
                    ].iloc[0]
                ),
            })

    return pd.DataFrame(rows)


def build_recurring_failures(fold_metrics):
    selected = fold_metrics[
        fold_metrics["model"].isin(ML_MODELS + ["zero_return"])
    ]

    wide = selected.pivot(
        index=["experiment", "fold_number", "instrument_key"],
        columns="model",
        values="mae",
    ).reset_index()

    if not set(ML_MODELS + ["zero_return"]).issubset(wide.columns):
        raise ValueError("Missing model or zero-return stock MAE.")

    rows = []

    for model in ML_MODELS:
        wide["excess_mae"] = wide[model] - wide["zero_return"]
        wide["worse_than_zero"] = (
            wide["excess_mae"] > BIAS_TOLERANCE
        )

        for (experiment, instrument), group in wide.groupby(
            ["experiment", "instrument_key"]
        ):
            worse = int(group["worse_than_zero"].sum())
            better = int(
                (group["excess_mae"] < -BIAS_TOLERANCE).sum()
            )

            rows.append({
                "experiment": experiment,
                "model": model,
                "instrument_key": instrument,
                "folds_worse_than_zero": worse,
                "folds_better_than_zero": better,
                "folds_tied_with_zero": 5 - worse - better,
                "mean_excess_mae_vs_zero": float(
                    group["excess_mae"].mean()
                ),
                "median_excess_mae_vs_zero": float(
                    group["excess_mae"].median()
                ),
                "recurring_failure_5_of_5": worse == 5,
                "recurring_failure_4_or_more": worse >= 4,
            })

    return pd.DataFrame(rows)


def build_failure_consensus(recurring_failures):
    rows = []

    for (experiment, instrument), group in recurring_failures.groupby(
        ["experiment", "instrument_key"]
    ):
        failed_models = group[
            group["recurring_failure_4_or_more"]
        ]["model"].tolist()

        rows.append({
            "experiment": experiment,
            "instrument_key": instrument,
            "models_failing_at_least_4_of_5_folds": len(failed_models),
            "model_names": ";".join(sorted(failed_models)),
            "total_worse_model_folds": int(
                group["folds_worse_than_zero"].sum()
            ),
            "models_tested": len(ML_MODELS),
        })

    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 95)
    print("PHASE 15 PART 3.4 - STOCK-LEVEL CALIBRATION ANALYSIS")
    print("=" * 95)

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

    print("\n[1/7] Loading saved out-of-fold predictions...")
    data = load_predictions()

    print("\n[2/7] Recalculating per-stock, per-fold metrics...")
    fold_metrics, pooled_metrics = build_stock_metrics(data)

    if len(fold_metrics) != 1000:
        raise RuntimeError(
            f"Expected 1000 fold-stock metric rows; got {len(fold_metrics)}."
        )

    if len(pooled_metrics) != 200:
        raise RuntimeError(
            f"Expected 200 pooled stock metric rows; "
            f"got {len(pooled_metrics)}."
        )

    print("Fold-stock metric rows:", len(fold_metrics))
    print("Pooled stock metric rows:", len(pooled_metrics))

    print("\n[3/7] Reconciling against previously saved Part 3 stock report...")
    saved_reconciliation = reconcile_saved_part3_report(
        fold_metrics,
        pooled_metrics,
    )

    print(json.dumps(saved_reconciliation, indent=2))

    print("\n[4/7] Classifying stock-level bias consistency...")
    bias_consistency = build_bias_consistency(
        pooled_metrics,
        fold_metrics,
    )

    print("Bias consistency rows:", len(bias_consistency))

    print("\n[5/7] Comparing Extra Trees against baselines...")
    extra_trees = build_extra_trees_comparison(fold_metrics)

    print("Extra Trees stock comparison rows:", len(extra_trees))

    print("\n[6/7] Identifying recurring stock-level failures...")
    recurring = build_recurring_failures(fold_metrics)
    failure_consensus = build_failure_consensus(recurring)

    print("Recurring failure rows:", len(recurring))
    print("Stock failure consensus rows:", len(failure_consensus))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports = {
        "fold_metrics": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_stock_fold_metrics.csv"
        ),
        "pooled_metrics": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_stock_pooled_metrics.csv"
        ),
        "bias_consistency": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_bias_consistency.csv"
        ),
        "extra_trees_comparison": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_extra_trees_comparison.csv"
        ),
        "recurring_failures": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_recurring_failures.csv"
        ),
        "failure_consensus": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_failure_consensus.csv"
        ),
        "summary": REPORT_DIR / (
            f"phase15_part3_4_{run_id}_summary.json"
        ),
    }

    for path in reports.values():
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    print("\n[7/7] Saving new read-only stock diagnostics...")

    fold_metrics.to_csv(reports["fold_metrics"], index=False)
    pooled_metrics.to_csv(reports["pooled_metrics"], index=False)
    bias_consistency.to_csv(reports["bias_consistency"], index=False)
    extra_trees.to_csv(reports["extra_trees_comparison"], index=False)
    recurring.to_csv(reports["recurring_failures"], index=False)
    failure_consensus.to_csv(reports["failure_consensus"], index=False)

    summary = {
        "run_id": run_id,
        "benchmark_run_id": BENCHMARK_RUN_ID,
        "part3_source_run_id": PART3_RUN_ID,
        "analysis": "stock_level_calibration_and_recurring_failures",
        "instruments": 20,
        "folds": 5,
        "experiments": list(EXPERIMENTS.keys()),
        "models": MODELS,
        "fold_stock_rows": len(fold_metrics),
        "pooled_stock_rows": len(pooled_metrics),
        "saved_part3_stock_report_reconciliation": saved_reconciliation,
        "bias_consistency_rule": (
            "Positive bias means overprediction; negative bias means "
            "underprediction. Consistent means 5/5 folds. Predominant "
            "means at least 4/5 folds."
        ),
        "extra_trees_comparison": (
            "Positive improvement means Extra Trees MAE is lower than "
            "the specified baseline MAE."
        ),
        "recurring_failure_rule": (
            "A model/stock is flagged when its MAE exceeds zero-return "
            "in at least 4 of 5 folds."
        ),
        "final_holdout_used": False,
        "models_fitted": False,
        "calibration_correction_applied": False,
        "hyperparameters_changed": False,
        "production_artifacts_modified": False,
        "protected_artifact_count": len(protected_files),
        "report_files": {
            key: path.name for key, path in reports.items()
        },
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
    # Console output
    # ---------------------------------------------------------
    print("\n" + "=" * 95)
    print("A. STOCK-LEVEL BIAS CONSISTENCY")
    print("=" * 95)

    bias_counts = (
        bias_consistency.groupby(
            ["experiment", "model", "bias_pattern"]
        )
        .size()
        .rename("stock_count")
        .reset_index()
    )

    print(
        bias_counts.sort_values(
            ["experiment", "model", "bias_pattern"]
        ).to_string(index=False)
    )

    print("\n" + "=" * 95)
    print("B. EXTRA TREES PERFORMANCE BY STOCK")
    print("=" * 95)

    for experiment in EXPERIMENTS:
        subset = extra_trees[
            extra_trees["experiment"] == experiment
        ].sort_values(
            "mean_mae_improvement_vs_zero",
            ascending=False,
        )

        print(f"\n{experiment.upper()}")

        print(
            subset[
                [
                    "instrument_key",
                    "folds_beating_zero",
                    "folds_beating_train_mean",
                    "mean_mae_improvement_vs_zero",
                    "fold5_mae_improvement_vs_zero",
                    "beats_zero_all_folds",
                ]
            ].round(6).to_string(index=False)
        )

    print("\n" + "=" * 95)
    print("C. RECURRING MODEL FAILURES")
    print("=" * 95)

    for experiment in EXPERIMENTS:
        subset = recurring[
            recurring["experiment"] == experiment
        ].sort_values(
            [
                "folds_worse_than_zero",
                "mean_excess_mae_vs_zero",
            ],
            ascending=[False, False],
        )

        print(f"\n{experiment.upper()}")

        print(
            subset[
                [
                    "model",
                    "instrument_key",
                    "folds_worse_than_zero",
                    "folds_better_than_zero",
                    "mean_excess_mae_vs_zero",
                    "recurring_failure_4_or_more",
                ]
            ].head(15).round(6).to_string(index=False)
        )

    print("\n" + "=" * 95)
    print("D. STOCKS WHERE MULTIPLE MODELS REPEATEDLY FAIL")
    print("=" * 95)

    print(
        failure_consensus.sort_values(
            [
                "experiment",
                "models_failing_at_least_4_of_5_folds",
                "total_worse_model_folds",
            ],
            ascending=[True, False, False],
        ).head(25).to_string(index=False)
    )

    print("\n" + "=" * 95)
    print("PHASE 15 PART 3.4 COMPLETED SUCCESSFULLY")
    print("=" * 95)

    for name, path in reports.items():
        print(f"{name}: {path}")

    print("\nSaved Part 3 stock report: RECONCILED")
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("Calibration corrections applied: NO")
    print("Hyperparameters changed: NO")
    print("Existing benchmark reports: NOT MODIFIED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
