from __future__ import annotations

import json
import math
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
    "extra_trees",
    "random_forest",
    "ridge",
]

FOLDS = {
    1: {"sessions": 61, "rows": 1220},
    2: {"sessions": 61, "rows": 1220},
    3: {"sessions": 61, "rows": 1220},
    4: {"sessions": 60, "rows": 1200},
    5: {"sessions": 60, "rows": 1200},
}

EXPECTED_DATES = {
    1: ("2025-01-14", "2025-04-11"),
    2: ("2025-04-15", "2025-07-10"),
    3: ("2025-07-11", "2025-10-08"),
    4: ("2025-10-09", "2026-01-05"),
    5: ("2026-01-06", "2026-04-06"),
}

BLOCK_LENGTHS = (7, 14)
BOOTSTRAP_REPS = 5000
RANDOM_SEED = 20261031
CONFIDENCE_LEVEL = 0.95

PREDICTION_COLUMNS = [
    "fold_number",
    "model",
    "instrument_key",
    "session_date",
    "target_column",
    "actual_return",
    "predicted_return",
]


def spearman_correlation(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan

    return float(
        pd.Series(x).corr(pd.Series(y), method="spearman")
    )


def fractional_tail_mean(
    scores: np.ndarray,
    actual: np.ndarray,
    k: int,
    highest: bool,
) -> float:
    """
    Calculate top/bottom-k realised return with fractional tie handling.

    If a score is tied at a portfolio boundary, the remaining portfolio
    weight is shared equally across all securities with that score.
    """

    if np.unique(scores).size < 2:
        return np.nan

    unique_scores = sorted(
        np.unique(scores),
        reverse=highest,
    )

    remaining = float(k)
    weighted_return = 0.0

    for score in unique_scores:
        mask = scores == score
        count = int(mask.sum())

        weight_per_stock = min(1.0, remaining / count)

        weighted_return += (
            float(actual[mask].sum()) * weight_per_stock
        )

        remaining -= count * weight_per_stock

        if remaining <= 1e-12:
            break

    if remaining > 1e-9:
        raise RuntimeError("Unable to construct fractional top/bottom basket.")

    return weighted_return / k


def build_daily_metrics(
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    One row per experiment, fold, model and trading date.

    All 20 stocks are kept together when calculating cross-sectional
    ranking and top/bottom portfolio spread.
    """

    rows = []

    for experiment, frame in predictions.items():

        for (fold, model, session_date), group in frame.groupby(
            ["fold_number", "model", "session_date"],
            sort=True,
        ):
            group = group.copy()

            if len(group) != 20:
                raise ValueError(
                    f"{experiment}/{model}/Fold {fold}/{session_date}: "
                    f"expected 20 stock rows, found {len(group)}."
                )

            if group["instrument_key"].nunique() != 20:
                raise ValueError(
                    f"{experiment}/{model}/{session_date}: "
                    "duplicate or missing stock instruments."
                )

            actual = group["actual_return"].to_numpy(dtype=float)
            predicted = group["predicted_return"].to_numpy(dtype=float)

            if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
                raise ValueError("Non-finite daily return observation.")

            unique_predictions = int(np.unique(predicted).size)

            # Cross-sectional rank correlation.
            ic = spearman_correlation(predicted, actual)

            # Reproduce the original Part 3 sorting method for reconciliation.
            ranked = group.sort_values(
                "predicted_return",
                ascending=False,
                kind="mergesort",
            )

            legacy_spread = np.nan

            if unique_predictions >= 2:
                top = ranked.head(5)
                bottom = ranked.tail(5)

                legacy_spread = float(
                    top["actual_return"].mean()
                    - bottom["actual_return"].mean()
                )

            # Detect score ties at either portfolio boundary.
            scores = group["predicted_return"].to_numpy(dtype=float)

            ordered_scores = np.sort(scores)
            bottom_boundary = ordered_scores[4]
            top_boundary = ordered_scores[-5]

            bottom_boundary_ties = int(
                np.sum(scores == bottom_boundary)
            )
            top_boundary_ties = int(
                np.sum(scores == top_boundary)
            )

            # Tie-aware top/bottom spread.
            # This is the primary spread used in bootstrap inference.
            if unique_predictions >= 2:
                top_mean = fractional_tail_mean(
                    scores, actual, k=5, highest=True
                )
                bottom_mean = fractional_tail_mean(
                    scores, actual, k=5, highest=False
                )

                tie_aware_spread = top_mean - bottom_mean
            else:
                tie_aware_spread = np.nan

            daily_mae = float(np.abs(predicted - actual).mean())
            daily_rmse = float(
                np.sqrt(np.mean(np.square(predicted - actual)))
            )

            rows.append({
                "experiment": experiment,
                "fold_number": int(fold),
                "model": model,
                "session_date": session_date,
                "stock_count": 20,
                "prediction_unique_count": unique_predictions,
                "daily_spearman_ic": ic,
                "top_bottom_spread_legacy": legacy_spread,
                "top_bottom_spread": tie_aware_spread,
                "daily_mae": daily_mae,
                "daily_rmse": daily_rmse,
                "top_boundary_tie_count": top_boundary_ties,
                "bottom_boundary_tie_count": bottom_boundary_ties,
            })

    result = pd.DataFrame(rows)

    if len(result) != 3030:
        raise ValueError(
            f"Expected 3030 daily metric rows; found {len(result)}."
        )

    if result.duplicated([
        "experiment", "fold_number", "model", "session_date"
    ]).any():
        raise ValueError("Duplicate daily metric keys.")

    return result


def load_and_validate_predictions():
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
            raise ValueError(f"{experiment}: invalid return values.")

        if len(df) != 30300:
            raise ValueError(
                f"{experiment}: expected 30300 observations; got {len(df)}."
            )

        if set(df["model"].unique()) != set(MODELS):
            raise ValueError(f"{experiment}: unexpected model list.")

        if set(df["target_column"].unique()) != {target}:
            raise ValueError(f"{experiment}: target mismatch.")

        keys = [
            "fold_number",
            "model",
            "instrument_key",
            "session_date",
        ]

        if df.duplicated(keys).any():
            raise ValueError(f"{experiment}: duplicate observations.")

        instruments = set(df["instrument_key"].unique())

        if len(instruments) != 20:
            raise ValueError(f"{experiment}: expected 20 stocks.")

        if reference_instruments is None:
            reference_instruments = instruments
        elif instruments != reference_instruments:
            raise ValueError("Stock universe differs between experiments.")

        key_set = set(
            map(tuple, df[keys].itertuples(index=False, name=None))
        )

        if reference_keys is not None and key_set != reference_keys:
            raise ValueError("Primary and forecast sample keys differ.")

        reference_keys = key_set

        actual_unique = df.groupby([
            "fold_number", "instrument_key", "session_date"
        ])["actual_return"].nunique()

        if (actual_unique > 1).any():
            raise ValueError("Actual return differs across models.")

        for fold, expected in FOLDS.items():
            expected_start, expected_end = EXPECTED_DATES[fold]

            for model in MODELS:
                subset = df[
                    (df["fold_number"] == fold)
                    & (df["model"] == model)
                ]

                if len(subset) != expected["rows"]:
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: row mismatch."
                    )

                if (
                    subset["session_date"].min(),
                    subset["session_date"].max(),
                ) != (expected_start, expected_end):
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: date mismatch."
                    )

                if not (
                    subset.groupby("instrument_key").size()
                    == expected["sessions"]
                ).all():
                    raise ValueError(
                        f"{experiment}/{model}/Fold {fold}: stock sessions mismatch."
                    )

        results[experiment] = df

        print(
            f"{experiment}: 30300 rows validated; "
            "6060 per model; five folds."
        )

    return results


def reconcile_part3_daily_report(daily: pd.DataFrame) -> dict:
    path = REPORT_DIR / (
        f"phase15_part3_{PART3_RUN_ID}_daily_discrimination.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    required = [
        "experiment",
        "fold_number",
        "model",
        "session_date",
        "cross_sectional_spearman_ic",
        "top_minus_bottom_actual_return",
    ]

    saved = pd.read_csv(path, usecols=required)

    saved["fold_number"] = pd.to_numeric(
        saved["fold_number"], errors="raise"
    ).astype(int)

    saved["session_date"] = pd.to_datetime(
        saved["session_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if len(saved) != 3030:
        raise ValueError(
            f"Expected 3030 saved Part 3 records; found {len(saved)}."
        )

    keys = ["experiment", "fold_number", "model", "session_date"]

    merged = daily.merge(
        saved,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "Daily discrimination samples differ from saved Part 3 report."
        )

    comparisons = [
        (
            "daily_spearman_ic",
            "cross_sectional_spearman_ic",
        ),
        (
            "top_bottom_spread_legacy",
            "top_minus_bottom_actual_return",
        ),
    ]

    for calculated_column, saved_column in comparisons:
        a = merged[calculated_column].to_numpy(dtype=float)
        b = pd.to_numeric(
            merged[saved_column], errors="coerce"
        ).to_numpy(dtype=float)

        if not np.allclose(
            a,
            b,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Part 3 daily report reconciliation failed: "
                f"{calculated_column}."
            )

    tie_days = int(
        (
            (daily["top_boundary_tie_count"] > 1)
            | (daily["bottom_boundary_tie_count"] > 1)
        ).sum()
    )

    return {
        "saved_report": path.name,
        "matched_daily_rows": int(len(merged)),
        "metrics_reconciled": 2,
        "passed": True,
        "days_with_top_or_bottom_boundary_ties": tie_days,
    }


def make_moving_block_indices(
    n: int,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Non-circular moving-block bootstrap.

    Blocks never wrap around from the end of one validation fold to its
    beginning. Every sampled block consists of consecutive trading dates.
    """

    if n < block_length:
        raise ValueError(
            f"Block length {block_length} exceeds date count {n}."
        )

    blocks_needed = math.ceil(n / block_length)

    starts = rng.integers(
        0,
        n - block_length + 1,
        size=(repetitions, blocks_needed),
    )

    offsets = np.arange(block_length)

    indices = (
        starts[:, :, None] + offsets[None, None, :]
    ).reshape(repetitions, -1)

    return indices[:, :n]


def bootstrap_distribution(
    values_by_fold: dict[int, np.ndarray],
    folds_to_include: list[int],
    indices_by_fold: dict[int, np.ndarray],
) -> np.ndarray:
    """Date-weighted mean over independently resampled fold blocks."""

    repetitions = next(
        iter(indices_by_fold.values())
    ).shape[0]

    total_sum = np.zeros(repetitions, dtype=float)
    total_count = np.zeros(repetitions, dtype=float)

    for fold in folds_to_include:
        values = np.asarray(
            values_by_fold[fold], dtype=float
        )

        indices = indices_by_fold[fold]

        if indices.shape[1] != len(values):
            raise ValueError(
                f"Fold {fold}: bootstrap indices and date array differ."
            )

        sampled = values[indices]

        valid = np.isfinite(sampled)

        total_sum += np.where(valid, sampled, 0.0).sum(axis=1)
        total_count += valid.sum(axis=1)

    result = np.divide(
        total_sum,
        total_count,
        out=np.full(repetitions, np.nan),
        where=total_count > 0,
    )

    return result


def point_estimate(
    values_by_fold: dict[int, np.ndarray],
    folds: list[int],
) -> float:
    values = np.concatenate([
        np.asarray(values_by_fold[fold], dtype=float)
        for fold in folds
    ])

    return (
        float(np.nanmean(values))
        if np.isfinite(values).any()
        else np.nan
    )


def summarize_bootstrap(
    values_by_fold: dict[int, np.ndarray],
    folds: list[int],
    indices_by_fold: dict[int, np.ndarray],
    *,
    zero_reference_test_applicable: bool = True,
) -> dict:
    """Summarise a block-bootstrap distribution.

    ``zero_reference_test_applicable`` must be False for standalone
    non-negative loss metrics such as MAE. Their confidence interval
    describes the level/uncertainty of the loss; testing whether that
    interval excludes zero is not a meaningful model-skill comparison.

    Keep it True for signed metrics (IC, spread) and for paired differences
    (including MAE differences), where zero is a meaningful no-difference
    reference.
    """
    estimate = point_estimate(values_by_fold, folds)

    bootstrap_values = bootstrap_distribution(
        values_by_fold,
        folds,
        indices_by_fold,
    )

    valid = bootstrap_values[np.isfinite(bootstrap_values)]

    if len(valid) < 100:
        raise RuntimeError(
            "Insufficient valid bootstrap replicates."
        )

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = float(np.quantile(valid, alpha / 2))
    upper = float(np.quantile(valid, 1 - alpha / 2))

    if zero_reference_test_applicable:
        positive_pct = float((valid > 0).mean() * 100)
        negative_pct = float((valid < 0).mean() * 100)
        ci_excludes_zero = bool(lower > 0 or upper < 0)
    else:
        # Blank these fields in the CSV: MAE > 0 is expected by definition.
        positive_pct = np.nan
        negative_pct = np.nan
        ci_excludes_zero = None

    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_sd": float(np.std(valid, ddof=1)),
        "bootstrap_positive_pct": positive_pct,
        "bootstrap_negative_pct": negative_pct,
        "valid_replicates": int(len(valid)),
        "zero_reference_test_applicable": bool(
            zero_reference_test_applicable
        ),
        "ci_excludes_zero": ci_excludes_zero,
    }


def build_model_bootstrap_reports(
    daily: pd.DataFrame,
    rng: np.random.Generator,
):
    rows = []
    index_cache = {}

    for experiment in EXPERIMENTS:

        exp_data = daily[daily["experiment"] == experiment]

        for block_length in BLOCK_LENGTHS:
            indices_by_fold = {}

            for fold in FOLDS:
                fold_data = exp_data[
                    exp_data["fold_number"] == fold
                ]

                dates = sorted(
                    fold_data["session_date"].unique()
                )

                expected = FOLDS[fold]["sessions"]

                if len(dates) != expected:
                    raise ValueError(
                        f"{experiment}/Fold {fold}: daily date count mismatch."
                    )

                indices_by_fold[fold] = make_moving_block_indices(
                    len(dates),
                    block_length,
                    BOOTSTRAP_REPS,
                    rng,
                )

            index_cache[(experiment, block_length)] = indices_by_fold

            for model in MODELS:
                for metric in (
                    "daily_mae",
                    "daily_spearman_ic",
                    "top_bottom_spread",
                ):
                    if metric != "daily_mae" and model not in ML_MODELS:
                        continue

                    values_by_fold = {}

                    for fold in FOLDS:
                        subset = exp_data[
                            (exp_data["fold_number"] == fold)
                            & (exp_data["model"] == model)
                        ].sort_values("session_date")

                        values_by_fold[fold] = subset[
                            metric
                        ].to_numpy(dtype=float)

                    scopes = [
                        ("fold", [fold])
                        for fold in FOLDS
                    ]
                    scopes.append(("pooled_oof", list(FOLDS.keys())))

                    for scope, included_folds in scopes:
                        result = summarize_bootstrap(
                            values_by_fold,
                            included_folds,
                            indices_by_fold,
                            zero_reference_test_applicable=(
                                metric != "daily_mae"
                            ),
                        )

                        date_count = sum(
                            len(values_by_fold[fold])
                            for fold in included_folds
                        )

                        rows.append({
                            "experiment": experiment,
                            "scope": scope,
                            "fold_number": (
                                included_folds[0]
                                if scope == "fold" else "ALL"
                            ),
                            "model": model,
                            "metric": metric,
                            "block_length_sessions": block_length,
                            "date_count": date_count,
                            **result,
                        })

    return pd.DataFrame(rows), index_cache


def get_paired_daily_arrays(
    daily: pd.DataFrame,
    experiment: str,
    fold: int,
    first_model: str,
    second_model: str,
    metric: str,
):
    subset = daily[
        (daily["experiment"] == experiment)
        & (daily["fold_number"] == fold)
        & (daily["model"].isin([first_model, second_model]))
    ]

    pivot = subset.pivot(
        index="session_date",
        columns="model",
        values=metric,
    ).sort_index()

    if first_model not in pivot.columns or second_model not in pivot.columns:
        raise ValueError("Paired model comparison is missing a model.")

    if len(pivot) != FOLDS[fold]["sessions"]:
        raise ValueError("Paired comparison does not contain all fold dates.")

    return (
        pivot[first_model].to_numpy(dtype=float)
        - pivot[second_model].to_numpy(dtype=float)
    )


def build_paired_comparison_report(
    daily: pd.DataFrame,
    index_cache,
) -> pd.DataFrame:
    rows = []

    pairs = [
        ("extra_trees", "random_forest", "daily_spearman_ic"),
        ("extra_trees", "random_forest", "top_bottom_spread"),
        ("extra_trees", "random_forest", "daily_mae"),
        ("extra_trees", "ridge", "daily_spearman_ic"),
        ("extra_trees", "ridge", "top_bottom_spread"),
        ("extra_trees", "ridge", "daily_mae"),
        ("random_forest", "ridge", "daily_spearman_ic"),
        ("random_forest", "ridge", "top_bottom_spread"),
        ("random_forest", "ridge", "daily_mae"),
        ("extra_trees", "zero_return", "daily_mae"),
        ("extra_trees", "train_mean", "daily_mae"),
        ("random_forest", "zero_return", "daily_mae"),
        ("random_forest", "train_mean", "daily_mae"),
        ("ridge", "zero_return", "daily_mae"),
        ("ridge", "train_mean", "daily_mae"),
    ]

    for experiment in EXPERIMENTS:

        for block_length in BLOCK_LENGTHS:
            indices_by_fold = index_cache[
                (experiment, block_length)
            ]

            for first_model, second_model, metric in pairs:

                differences_by_fold = {}

                for fold in FOLDS:
                    differences_by_fold[fold] = get_paired_daily_arrays(
                        daily,
                        experiment,
                        fold,
                        first_model,
                        second_model,
                        metric,
                    )

                scopes = [
                    ("fold", [fold])
                    for fold in FOLDS
                ]
                scopes.append(("pooled_oof", list(FOLDS.keys())))

                for scope, included_folds in scopes:
                    result = summarize_bootstrap(
                        differences_by_fold,
                        included_folds,
                        indices_by_fold,
                    )

                    if metric in (
                        "daily_spearman_ic",
                        "top_bottom_spread",
                    ):
                        direction = "higher_is_better"
                        first_better = (
                            result["bootstrap_positive_pct"]
                        )
                        confidence_favours_first = (
                            result["ci_lower"] > 0
                        )
                    else:
                        direction = "lower_is_better"
                        first_better = (
                            result["bootstrap_negative_pct"]
                        )
                        confidence_favours_first = (
                            result["ci_upper"] < 0
                        )

                    rows.append({
                        "experiment": experiment,
                        "scope": scope,
                        "fold_number": (
                            included_folds[0]
                            if scope == "fold" else "ALL"
                        ),
                        "first_model": first_model,
                        "second_model": second_model,
                        "metric": metric,
                        "difference_definition": (
                            f"{first_model} minus {second_model}"
                        ),
                        "interpretation_direction": direction,
                        "block_length_sessions": block_length,
                        **result,
                        "bootstrap_first_model_better_pct": first_better,
                        "ci_supports_first_model_superiority": bool(
                            confidence_favours_first
                        ),
                    })

    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 100)
    print("PHASE 15 PART 4.1.1 - STATISTICAL ROBUSTNESS VERIFICATION")
    print("=" * 100)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    protected_files = sorted(discover_protected_files())

    if len(protected_files) != 11:
        raise RuntimeError(
            f"Expected 11 protected production/V4 artifacts; "
            f"found {len(protected_files)}."
        )

    protected_before = {
        str(path): fingerprint(path)
        for path in protected_files
    }

    print("\n[1/6] Loading original OOF prediction reports...")
    predictions = load_and_validate_predictions()

    print("\n[2/6] Calculating daily cross-sectional metrics...")
    daily = build_daily_metrics(predictions)

    print("Daily model/date observations:", len(daily))

    print("\n[3/6] Reconciling metrics with saved Part 3 discrimination report...")
    reconciliation = reconcile_part3_daily_report(daily)

    print(json.dumps(reconciliation, indent=2))

    print("\n[4/6] Running moving-block bootstrap...")
    print("Bootstrap repetitions:", BOOTSTRAP_REPS)
    print("Block lengths:", BLOCK_LENGTHS)
    print("Confidence level:", CONFIDENCE_LEVEL)
    print("Random seed:", RANDOM_SEED)

    rng = np.random.default_rng(RANDOM_SEED)

    model_report, index_cache = build_model_bootstrap_reports(
        daily,
        rng,
    )

    print("Model uncertainty rows:", len(model_report))

    print("\n[5/6] Running paired model and baseline comparisons...")
    paired_report = build_paired_comparison_report(
        daily,
        index_cache,
    )

    print("Paired comparison rows:", len(paired_report))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports = {
        "daily_metrics": REPORT_DIR / (
            f"phase15_part4_1_{run_id}_daily_metrics.csv"
        ),
        "model_uncertainty": REPORT_DIR / (
            f"phase15_part4_1_{run_id}_model_uncertainty.csv"
        ),
        "paired_comparisons": REPORT_DIR / (
            f"phase15_part4_1_{run_id}_paired_comparisons.csv"
        ),
        "summary": REPORT_DIR / (
            f"phase15_part4_1_{run_id}_summary.json"
        ),
    }

    for path in reports.values():
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {path}"
            )

    print("\n[6/6] Saving new robustness reports...")

    daily.to_csv(reports["daily_metrics"], index=False)
    model_report.to_csv(reports["model_uncertainty"], index=False)
    paired_report.to_csv(reports["paired_comparisons"], index=False)

    summary = {
        "run_id": run_id,
        "benchmark_run_id": BENCHMARK_RUN_ID,
        "part3_run_id": PART3_RUN_ID,
        "analysis": "date_block_bootstrap_statistical_robustness",
        "experiments": EXPERIMENTS,
        "models": MODELS,
        "instruments": 20,
        "validation_dates": 303,
        "fold_count": 5,
        "bootstrap_method": (
            "Non-circular moving-block bootstrap, independently within "
            "each validation fold. Complete daily cross-sections remain "
            "together. Paired model differences use identical sampled dates."
        ),
        "block_lengths_sessions": list(BLOCK_LENGTHS),
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "random_seed": RANDOM_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "top_bottom_method": (
            "Top five versus bottom five stocks per date, ranked by "
            "prediction. Fractional allocation at tied score boundaries."
        ),
        "zero_return_ranking": (
            "Zero Return has constant predictions and cannot rank stocks. "
            "It is included only in paired MAE comparisons. Ranking skill "
            "is assessed against zero as a no-skill reference."
        ),
        "paired_comparisons": (
            "Extra Trees, Random Forest and Ridge are compared using "
            "date-paired differences. MAE is lower-is-better; IC and "
            "top-bottom spread are higher-is-better. Zero-reference "
            "interval tests are applied to paired differences."
        ),
        "standalone_metric_interpretation": {
            "daily_mae": (
                "Standalone MAE is a non-negative loss. Its bootstrap "
                "confidence interval describes the loss level and "
                "uncertainty; bootstrap sign percentages and whether the "
                "interval excludes zero are not reported because zero is "
                "not a meaningful no-skill reference for standalone MAE. "
                "Compare MAE models using paired date-wise MAE differences."
            ),
            "daily_spearman_ic": (
                "Zero is a no-association reference. Bootstrap sign "
                "percentages are the fraction of resamples above/below zero, "
                "not a posterior probability or a guarantee of future skill."
            ),
            "top_bottom_spread": (
                "Zero is a no-spread reference. Bootstrap sign percentages "
                "are the fraction of resamples above/below zero, not a "
                "posterior probability or a guarantee of future returns."
            ),
        },
        "reporting_revision": (
            "Standalone MAE rows leave bootstrap sign percentages and "
            "ci_excludes_zero blank. Paired MAE differences retain these "
            "zero-reference comparison fields."
        ),
        "uncertainty_warning": (
            "Bootstrap intervals are descriptive uncertainty estimates. "
            "Seven-session targets overlap and cross-stock dependence "
            "persists across dates. Block lengths 7 and 14 provide a "
            "sensitivity check, not a guarantee that dependence is fully "
            "captured. Results are exploratory and are not proof of "
            "trading profitability."
        ),
        "multiple_testing": (
            "No multiple-comparison adjustment is applied. Interpret "
            "many fold/model comparisons as exploratory."
        ),
        "part3_reconciliation": reconciliation,
        "final_holdout_used": False,
        "models_fitted": False,
        "hyperparameters_changed": False,
        "calibration_corrections_applied": False,
        "production_artifacts_modified": False,
        "protected_artifact_count": len(protected_files),
        "reports": {
            name: path.name for name, path in reports.items()
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
            "Protected production/V4 artifact fingerprints changed."
        )

    # ---------------------------------------------------------
    # Console summaries
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("A. POOLED MODEL ROBUSTNESS - 7 SESSION BLOCKS")
    print("=" * 100)

    pooled = model_report[
        (model_report["scope"] == "pooled_oof")
        & (model_report["block_length_sessions"] == 7)
        & (
            model_report["model"].isin(
                ["extra_trees", "random_forest", "ridge"]
            )
        )
    ]

    pooled_ranking = pooled[
        pooled["metric"].isin(
            ["daily_spearman_ic", "top_bottom_spread"]
        )
    ]
    pooled_mae = pooled[pooled["metric"] == "daily_mae"]

    print("Ranking metrics (zero is the no-skill reference):")
    print(
        pooled_ranking[
            [
                "experiment",
                "model",
                "metric",
                "estimate",
                "ci_lower",
                "ci_upper",
                "bootstrap_positive_pct",
                "ci_excludes_zero",
            ]
        ].round(6).to_string(index=False)
    )
    print("\nStandalone MAE (CI describes loss level; no zero-reference test):")
    print(
        pooled_mae[
            [
                "experiment",
                "model",
                "estimate",
                "ci_lower",
                "ci_upper",
                "bootstrap_sd",
                "valid_replicates",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("B. POOLED MODEL ROBUSTNESS - 14 SESSION BLOCKS")
    print("=" * 100)

    pooled14 = model_report[
        (model_report["scope"] == "pooled_oof")
        & (model_report["block_length_sessions"] == 14)
        & (
            model_report["model"].isin(
                ["extra_trees", "random_forest", "ridge"]
            )
        )
    ]

    pooled14_ranking = pooled14[
        pooled14["metric"].isin(
            ["daily_spearman_ic", "top_bottom_spread"]
        )
    ]
    pooled14_mae = pooled14[pooled14["metric"] == "daily_mae"]

    print("Ranking metrics (zero is the no-skill reference):")
    print(
        pooled14_ranking[
            [
                "experiment",
                "model",
                "metric",
                "estimate",
                "ci_lower",
                "ci_upper",
                "bootstrap_positive_pct",
                "ci_excludes_zero",
            ]
        ].round(6).to_string(index=False)
    )
    print("\nStandalone MAE (CI describes loss level; no zero-reference test):")
    print(
        pooled14_mae[
            [
                "experiment",
                "model",
                "estimate",
                "ci_lower",
                "ci_upper",
                "bootstrap_sd",
                "valid_replicates",
            ]
        ].round(6).to_string(index=False)
    )

    print("\nNote: bootstrap sign percentages are resample fractions, not probabilities.")
    print("Standalone MAE does not use a zero-reference significance test; paired MAE differences do.")

    print("\n" + "=" * 100)
    print("C. EXTRA TREES VERSUS RANDOM FOREST - PAIRED COMPARISONS")
    print("=" * 100)

    comparison = paired_report[
        (paired_report["scope"] == "pooled_oof")
        & (
            paired_report["first_model"] == "extra_trees"
        )
        & (
            paired_report["second_model"] == "random_forest"
        )
    ]

    print(
        comparison[
            [
                "experiment",
                "metric",
                "block_length_sessions",
                "estimate",
                "ci_lower",
                "ci_upper",
                "bootstrap_first_model_better_pct",
                "ci_supports_first_model_superiority",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("D. TREE MODELS VERSUS ZERO-RETURN - MAE")
    print("=" * 100)

    baseline = paired_report[
        (paired_report["scope"] == "pooled_oof")
        & (
            paired_report["second_model"] == "zero_return"
        )
        & (
            paired_report["metric"] == "daily_mae"
        )
    ]

    print(
        baseline[
            [
                "experiment",
                "first_model",
                "block_length_sessions",
                "estimate",
                "ci_lower",
                "ci_upper",
                "bootstrap_first_model_better_pct",
                "ci_supports_first_model_superiority",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("E. FOLD-LEVEL EXTRA TREES AND RANDOM FOREST RANKING")
    print("=" * 100)

    fold_ranking = model_report[
        (model_report["scope"] == "fold")
        & (model_report["block_length_sessions"] == 7)
        & (
            model_report["model"].isin(
                ["extra_trees", "random_forest"]
            )
        )
        & (
            model_report["metric"].isin(
                ["daily_spearman_ic", "top_bottom_spread"]
            )
        )
    ]

    print(
        fold_ranking[
            [
                "experiment",
                "fold_number",
                "model",
                "metric",
                "estimate",
                "ci_lower",
                "ci_upper",
                "ci_excludes_zero",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("PHASE 15 PART 4.1.1 COMPLETED SUCCESSFULLY")
    print("=" * 100)

    for name, path in reports.items():
        print(f"{name}: {path}")

    print("\nPart 3 daily discrimination: RECONCILED")
    print("Protected production/V4 files: 11 UNCHANGED")
    print("Final holdout: NOT USED")
    print("Models fitted: NO")
    print("Hyperparameters changed: NO")
    print("Calibration corrections: NONE")
    print("Existing benchmark reports: NOT MODIFIED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
