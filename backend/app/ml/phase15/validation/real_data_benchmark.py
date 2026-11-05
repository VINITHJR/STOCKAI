"""Run isolated Phase 15 real-market walk-forward benchmarks.

Experiments:
A. next_open_to_close_return_7s
B. close_to_close_return_7s

The final holdout is reserved and never fitted, predicted, or scored.
All output files are written only to Phase 15/reports.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.phase15.data.historical_loader import (
    HistoricalDataError,
    fetch_daily_candles,
)
from backend.app.ml.phase15.features.feature_builder import (
    PHASE15_FEATURE_COLUMNS,
    build_phase15_features,
)
from backend.app.ml.phase15.targets.target_builder import (
    build_phase15_training_frame,
)
from backend.app.ml.phase15.validation.walk_forward import (
    combine_training_frames,
    build_walk_forward_splits,
)
from backend.app.ml.phase15.validation.model_benchmark import (
    DEFAULT_MODELS,
    PRIMARY_TARGET,
    FORECAST_TARGET,
    benchmark_walk_forward,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path.cwd()
MODEL_DIR = ROOT / "backend/app/ml/models"
REPORT_DIR = ROOT / "backend/app/ml/phase15/reports"

KEY_FILE = MODEL_DIR / "lstm_instrument_keys.txt"

START_DATE = "2023-10-28"
END_DATE = "2026-10-28"

EXPECTED_STOCKS = 20
EXPECTED_SESSIONS = 744
EXPECTED_FEATURE_READY = 695
EXPECTED_TARGET_READY = 737
EXPECTED_ELIGIBLE = 688

MAX_HORIZON = 7
INITIAL_TRAIN_SIZE = 252
N_SPLITS = 5
FINAL_HOLDOUT_SIZE = 126

EXPECTED_VALIDATION_SIZES = [61, 61, 61, 60, 60]
EXPECTED_TRAINING_SIZES = [245, 306, 367, 428, 488]

NIFTY_KEY = "NSE_INDEX|Nifty 50"
VIX_KEY = "NSE_INDEX|India VIX"

EXPERIMENTS = (
    ("primary", PRIMARY_TARGET),
    ("forecast", FORECAST_TARGET),
)

PRODUCTION_FILES = (
    MODEL_DIR / "lstm_model_v2.keras",
    MODEL_DIR / "lstm_scalers_v2.pkl",
)


# =========================================================
# FILE PROTECTION
# =========================================================

def discover_protected_files() -> list[Path]:
    missing = [
        path for path in PRODUCTION_FILES
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing production files:\n"
            + "\n".join(str(path) for path in missing)
        )

    v4_files = sorted(
        path
        for path in MODEL_DIR.glob("lstm_v4_candidate*")
        if path.is_file()
    )

    if len(v4_files) < 3:
        raise FileNotFoundError(
            "Expected at least three V4 candidate artifacts; "
            f"found {len(v4_files)}."
        )

    return sorted(PRODUCTION_FILES) + v4_files


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    stat = path.stat()

    return {
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def verify_protected_files(
    protected_files: list[Path],
    baseline: dict,
) -> bool:
    try:
        current_files = discover_protected_files()

        if current_files != protected_files:
            print("Protected artifact inventory changed.")
            return False

        all_unchanged = True

        for path in protected_files:
            current = fingerprint(path)
            unchanged = current == baseline[path]

            print(
                f"{'UNCHANGED' if unchanged else 'CHANGED'}: "
                f"{path.name}"
            )

            if not unchanged:
                all_unchanged = False

        return all_unchanged

    except Exception:
        traceback.print_exc()
        return False


# =========================================================
# API RETRIES
# =========================================================

def fetch_with_retry(
    instrument_key: str,
    start_date: str,
    end_date: str,
    attempts: int = 3,
) -> dict:
    """Retry transient network timeouts only."""

    transient_markers = (
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionResetError",
        "RemoteDisconnected",
        "Connection aborted",
        "Temporary failure",
    )

    for attempt in range(1, attempts + 1):
        try:
            return fetch_daily_candles(
                instrument_key,
                start_date,
                end_date,
            )

        except HistoricalDataError as exc:
            message = str(exc)

            retryable = any(
                marker in message
                for marker in transient_markers
            )

            if not retryable or attempt == attempts:
                raise

            delay = 2 ** attempt

            print(
                f"Transient API error for {instrument_key}. "
                f"Retry {attempt + 1}/{attempts} "
                f"in {delay} seconds."
            )

            time.sleep(delay)


# =========================================================
# DATA PREPARATION
# =========================================================

def load_stock_keys() -> list[str]:
    if not KEY_FILE.is_file():
        raise FileNotFoundError(
            f"Instrument key file missing: {KEY_FILE}"
        )

    keys = []

    for line in KEY_FILE.read_text(
        encoding="utf-8"
    ).splitlines():
        key = line.split("#", 1)[0].strip()

        if key:
            keys.append(key)

    if len(keys) != EXPECTED_STOCKS:
        raise AssertionError(
            f"Expected {EXPECTED_STOCKS} stock keys; "
            f"found {len(keys)}."
        )

    if len(set(keys)) != len(keys):
        raise AssertionError("Duplicate instrument keys found.")

    return keys


def build_real_data_plan():
    """Fetch actual market data and build the walk-forward plan in memory."""

    print("\n[2/6] Loading the stock universe...")

    stock_keys = load_stock_keys()
    print(f"Unique stock instruments: {len(stock_keys)}")

    print("\n[3/6] Fetching NIFTY 50 and India VIX...")

    nifty_data = fetch_with_retry(
        NIFTY_KEY,
        START_DATE,
        END_DATE,
    )

    time.sleep(0.3)

    vix_data = fetch_with_retry(
        VIX_KEY,
        START_DATE,
        END_DATE,
    )

    nifty_calendar = [
        row["session_date"] for row in nifty_data["rows"]
    ]

    vix_calendar = [
        row["session_date"] for row in vix_data["rows"]
    ]

    if len(nifty_calendar) != EXPECTED_SESSIONS:
        raise AssertionError(
            f"NIFTY sessions={len(nifty_calendar)}; "
            f"expected {EXPECTED_SESSIONS}."
        )

    if len(vix_calendar) != EXPECTED_SESSIONS:
        raise AssertionError(
            f"India VIX sessions={len(vix_calendar)}; "
            f"expected {EXPECTED_SESSIONS}."
        )

    if nifty_calendar != vix_calendar:
        raise AssertionError(
            "NIFTY and India VIX calendars differ."
        )

    print(f"NIFTY sessions: {len(nifty_calendar)}")
    print(f"India VIX sessions: {len(vix_calendar)}")
    print("Benchmark calendars: MATCH")

    print("\n[4/6] Building 20 stock feature and target frames...")

    training_frames = {}

    for number, instrument_key in enumerate(stock_keys, start=1):
        print(
            f"[{number:02d}/{EXPECTED_STOCKS}] "
            f"{instrument_key}"
        )

        time.sleep(0.25)

        stock_data = fetch_with_retry(
            instrument_key,
            START_DATE,
            END_DATE,
        )

        stock_calendar = [
            row["session_date"]
            for row in stock_data["rows"]
        ]

        if len(stock_calendar) != EXPECTED_SESSIONS:
            raise AssertionError(
                f"{instrument_key}: expected {EXPECTED_SESSIONS} "
                f"sessions, received {len(stock_calendar)}."
            )

        if stock_calendar != nifty_calendar:
            raise AssertionError(
                f"{instrument_key}: calendar differs from NIFTY."
            )

        features = build_phase15_features(
            stock_data,
            nifty_data,
            vix_data,
        )

        expected_columns = [
            "session_date",
            *PHASE15_FEATURE_COLUMNS,
        ]

        if list(features.columns) != expected_columns:
            raise AssertionError(
                f"{instrument_key}: feature schema mismatch."
            )

        if len(features) != EXPECTED_SESSIONS:
            raise AssertionError(
                f"{instrument_key}: incorrect feature row count."
            )

        frame = build_phase15_training_frame(
            features,
            stock_data,
            horizons=(1, 3, 7),
        )

        counts = {
            "feature_ready": int(frame["feature_ready"].sum()),
            "targets_ready": int(frame["targets_ready"].sum()),
            "training_eligible": int(frame["training_eligible"].sum()),
        }

        expected_counts = {
            "feature_ready": EXPECTED_FEATURE_READY,
            "targets_ready": EXPECTED_TARGET_READY,
            "training_eligible": EXPECTED_ELIGIBLE,
        }

        if counts != expected_counts:
            raise AssertionError(
                f"{instrument_key}: readiness count mismatch. "
                f"Actual={counts}; expected={expected_counts}."
            )

        training_frames[instrument_key] = frame

        print(
            f"    features={counts['feature_ready']}, "
            f"targets={counts['targets_ready']}, "
            f"eligible={counts['training_eligible']}"
        )

    print("\nAll 20 stock frames passed.")

    print("\n[5/6] Building shared panel and walk-forward plan...")

    panel = combine_training_frames(training_frames)

    if len(panel) != EXPECTED_STOCKS * EXPECTED_SESSIONS:
        raise AssertionError(
            f"Combined panel has {len(panel)} rows; "
            f"expected {EXPECTED_STOCKS * EXPECTED_SESSIONS}."
        )

    if panel["instrument_key"].nunique() != EXPECTED_STOCKS:
        raise AssertionError("Combined panel does not contain 20 stocks.")

    eligible_dates = panel.loc[
        panel["training_eligible"],
        "session_date",
    ].nunique()

    if eligible_dates != EXPECTED_ELIGIBLE:
        raise AssertionError(
            f"Shared eligible dates={eligible_dates}; "
            f"expected {EXPECTED_ELIGIBLE}."
        )

    plan = build_walk_forward_splits(
        panel,
        n_splits=N_SPLITS,
        initial_train_size=INITIAL_TRAIN_SIZE,
        final_holdout_size=FINAL_HOLDOUT_SIZE,
        max_horizon=MAX_HORIZON,
    )

    actual_validation_sizes = [
        fold.validation_session_count
        for fold in plan.folds
    ]

    actual_training_sizes = [
        fold.train_session_count
        for fold in plan.folds
    ]

    if actual_validation_sizes != EXPECTED_VALIDATION_SIZES:
        raise AssertionError(
            f"Validation sizes differ: {actual_validation_sizes}"
        )

    if actual_training_sizes != EXPECTED_TRAINING_SIZES:
        raise AssertionError(
            f"Training sizes differ: {actual_training_sizes}"
        )

    if plan.final_train_session_count != 555:
        raise AssertionError("Unexpected final training date count.")

    if len(plan.final_train_indices) != 555 * EXPECTED_STOCKS:
        raise AssertionError("Unexpected final training row count.")

    if plan.final_holdout_session_count != FINAL_HOLDOUT_SIZE:
        raise AssertionError("Unexpected final holdout date count.")

    if len(plan.final_holdout_indices) != (
        FINAL_HOLDOUT_SIZE * EXPECTED_STOCKS
    ):
        raise AssertionError("Unexpected final holdout row count.")

    if plan.final_purged_session_count != MAX_HORIZON:
        raise AssertionError("Final training purge count mismatch.")

    if plan.holdout_embargo_session_count != MAX_HORIZON:
        raise AssertionError("Final holdout embargo count mismatch.")

    print(f"Combined panel rows: {len(panel)}")
    print(f"Combined instruments: {panel['instrument_key'].nunique()}")
    print(f"Shared eligible dates: {eligible_dates}")

    print("\nWalk-forward fold summary:")

    for fold in plan.folds:
        print(
            f"Fold {fold.fold_number}: "
            f"train={fold.train_session_count} dates/"
            f"{len(fold.train_indices)} rows; "
            f"validation={fold.validation_session_count} dates/"
            f"{len(fold.validation_indices)} rows"
        )

    print(
        f"Final holdout reserved: "
        f"{plan.final_holdout_start_date} to "
        f"{plan.final_holdout_end_date}"
    )

    return plan


# =========================================================
# BENCHMARK VALIDATION
# =========================================================

def validate_benchmark_result(
    result,
    plan,
    target_column: str,
) -> None:

    expected_models = set(DEFAULT_MODELS)

    expected_validation_rows = sum(
        len(fold.validation_indices)
        for fold in plan.folds
    )

    expected_prediction_rows = (
        expected_validation_rows * len(DEFAULT_MODELS)
    )

    expected_metric_rows = (
        N_SPLITS * len(DEFAULT_MODELS)
        + len(DEFAULT_MODELS)
    )

    expected_audit_rows = (
        N_SPLITS * len(DEFAULT_MODELS)
    )

    if result.target_column != target_column:
        raise AssertionError("Benchmark target metadata mismatch.")

    if result.feature_columns != tuple(PHASE15_FEATURE_COLUMNS):
        raise AssertionError("Benchmark feature schema mismatch.")

    if len(result.feature_columns) != 31:
        raise AssertionError("Benchmark did not use exactly 31 features.")

    if result.holdout_used:
        raise AssertionError(
            "Benchmark reports that the final holdout was used."
        )

    if set(result.predictions["model"]) != expected_models:
        raise AssertionError("Benchmark prediction model set mismatch.")

    if set(result.metrics["model"]) != expected_models:
        raise AssertionError("Benchmark metric model set mismatch.")

    if set(result.predictions["target_column"]) != {target_column}:
        raise AssertionError("Mixed target columns in prediction output.")

    if len(result.predictions) != expected_prediction_rows:
        raise AssertionError(
            f"Prediction rows={len(result.predictions)}; "
            f"expected {expected_prediction_rows}."
        )

    if len(result.metrics) != expected_metric_rows:
        raise AssertionError(
            f"Metric rows={len(result.metrics)}; "
            f"expected {expected_metric_rows}."
        )

    if len(result.fit_audit) != expected_audit_rows:
        raise AssertionError(
            f"Audit rows={len(result.fit_audit)}; "
            f"expected {expected_audit_rows}."
        )

    predictions = result.predictions

    if not np.isfinite(
        predictions["actual_return"].to_numpy(dtype=float)
    ).all():
        raise AssertionError("Actual returns contain non-finite values.")

    if not np.isfinite(
        predictions["predicted_return"].to_numpy(dtype=float)
    ).all():
        raise AssertionError("Predictions contain non-finite values.")

    # Check that all five models received exactly the same rows
    # within each fold.
    for fold in plan.folds:
        fold_predictions = predictions.loc[
            predictions["fold_number"] == fold.fold_number
        ]

        reference_keys = None

        for model_name in DEFAULT_MODELS:
            model_rows = fold_predictions.loc[
                fold_predictions["model"] == model_name
            ]

            keys = set(
                zip(
                    model_rows["instrument_key"].astype(str),
                    model_rows["session_date"].astype(str),
                )
            )

            if len(keys) != len(fold.validation_indices):
                raise AssertionError(
                    f"Fold {fold.fold_number}, {model_name}: "
                    "duplicate or missing validation keys."
                )

            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise AssertionError(
                    f"Fold {fold.fold_number}: models used different samples."
                )

        if len(reference_keys) != len(fold.validation_indices):
            raise AssertionError(
                f"Fold {fold.fold_number}: validation sample count mismatch."
            )

    audit = result.fit_audit

    if audit["final_holdout_used"].astype(bool).any():
        raise AssertionError("Audit reports final holdout usage.")

    if not (audit["feature_count"] == 31).all():
        raise AssertionError("Audit reports an incorrect feature count.")

    if (
        audit["maximum_training_target_end"]
        >= audit["validation_start_date"]
    ).any():
        raise AssertionError(
            "A training target crosses its validation boundary."
        )

    if (
        audit["maximum_validation_target_end"]
        >= plan.final_holdout_start_date
    ).any():
        raise AssertionError(
            "A validation target reaches the final holdout."
        )

    # Pooled metrics must contain one row per model.
    pooled = result.metrics.loc[
        result.metrics["scope"] == "pooled_oof"
    ]

    if len(pooled) != len(DEFAULT_MODELS):
        raise AssertionError("Pooled out-of-fold metrics are incomplete.")

    print(
        f"Validated {target_column}: "
        f"{len(result.predictions):,} predictions, "
        f"{len(result.metrics)} metric rows, "
        f"{len(result.fit_audit)} audit rows."
    )


# =========================================================
# EXPERIMENT EXECUTION
# =========================================================

def run_experiments(plan):
    results = {}

    identity_columns = [
        "fold_number",
        "model",
        "instrument_key",
        "session_date",
    ]

    for label, target_column in EXPERIMENTS:
        print("\n" + "=" * 68)
        print(f"EXPERIMENT: {label.upper()}")
        print(f"TARGET: {target_column}")
        print("=" * 68)

        result = benchmark_walk_forward(
            plan,
            target_column=target_column,
            model_names=DEFAULT_MODELS,
        )

        validate_benchmark_result(
            result,
            plan,
            target_column,
        )

        print("\nPooled out-of-fold metrics:")

        pooled = result.metrics.loc[
            result.metrics["scope"] == "pooled_oof",
            [
                "model",
                "sample_count",
                "mae",
                "rmse",
                "r2",
                "directional_accuracy_pct",
                "directional_coverage_pct",
            ],
        ]

        print(
            pooled.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

        results[label] = result

    # Both target experiments must evaluate identical sample keys.
    primary_keys = (
        results["primary"]
        .predictions[identity_columns]
        .sort_values(identity_columns)
        .reset_index(drop=True)
    )

    forecast_keys = (
        results["forecast"]
        .predictions[identity_columns]
        .sort_values(identity_columns)
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        primary_keys,
        forecast_keys,
    )

    print("\nExperiment sample alignment: PASSED")
    print("Primary and forecast predictions remain separate.")

    return results


# =========================================================
# REPORT OUTPUT
# =========================================================

def save_reports(plan, results, run_id, protected_count):
    """Save outputs only under the isolated Phase 15 reports directory."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_map = {}

    for label, result in results.items():
        file_map[label] = {
            "metrics": (
                f"phase15_benchmark_{run_id}_{label}_metrics.csv"
            ),
            "predictions": (
                f"phase15_benchmark_{run_id}_{label}_oof_predictions.csv"
            ),
            "fit_audit": (
                f"phase15_benchmark_{run_id}_{label}_fit_audit.csv"
            ),
        }

    summary_filename = (
        f"phase15_benchmark_{run_id}_summary.json"
    )

    destinations = [
        REPORT_DIR / name
        for group in file_map.values()
        for name in group.values()
    ] + [REPORT_DIR / summary_filename]

    collisions = [
        path for path in destinations
        if path.exists()
    ]

    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing benchmark reports:\n"
            + "\n".join(str(path) for path in collisions)
        )

    # Write the six experiment CSV files.
    for label, result in results.items():
        result.metrics.to_csv(
            REPORT_DIR / file_map[label]["metrics"],
            index=False,
        )

        result.predictions.to_csv(
            REPORT_DIR / file_map[label]["predictions"],
            index=False,
        )

        result.fit_audit.to_csv(
            REPORT_DIR / file_map[label]["fit_audit"],
            index=False,
        )

    fold_summary = []

    for fold in plan.folds:
        fold_summary.append({
            "fold_number": fold.fold_number,
            "training_start_date": fold.train_start_date,
            "training_end_date": fold.train_end_date,
            "training_session_count": fold.train_session_count,
            "training_row_count": len(fold.train_indices),
            "validation_start_date": fold.validation_start_date,
            "validation_end_date": fold.validation_end_date,
            "validation_session_count": fold.validation_session_count,
            "validation_row_count": len(fold.validation_indices),
            "purged_session_count": fold.purged_session_count,
            "purged_sample_count": fold.purged_sample_count,
        })

    summary = {
        "run_id": run_id,
        "data_start_date": START_DATE,
        "data_end_date": END_DATE,
        "instrument_count": EXPECTED_STOCKS,
        "sessions_per_instrument": EXPECTED_SESSIONS,
        "combined_panel_rows": EXPECTED_STOCKS * EXPECTED_SESSIONS,
        "feature_count": len(PHASE15_FEATURE_COLUMNS),
        "feature_columns": list(PHASE15_FEATURE_COLUMNS),
        "models": list(DEFAULT_MODELS),
        "fold_count": N_SPLITS,
        "folds": fold_summary,
        "final_training": {
            "start_date": plan.final_train_start_date,
            "end_date": plan.final_train_end_date,
            "session_count": plan.final_train_session_count,
            "row_count": len(plan.final_train_indices),
            "purged_session_count": plan.final_purged_session_count,
            "purged_sample_count": plan.final_purged_sample_count,
        },
        "final_holdout": {
            "start_date": plan.final_holdout_start_date,
            "end_date": plan.final_holdout_end_date,
            "session_count": plan.final_holdout_session_count,
            "reserved_only": True,
            "used_for_fitting": False,
            "used_for_prediction": False,
            "used_for_scoring": False,
            "validation_embargo_sessions": (
                plan.holdout_embargo_session_count
            ),
        },
        "experiments": {
            label: {
                "target_column": result.target_column,
                "holdout_used": result.holdout_used,
                "metrics_file": file_map[label]["metrics"],
                "predictions_file": file_map[label]["predictions"],
                "fit_audit_file": file_map[label]["fit_audit"],
            }
            for label, result in results.items()
        },
        "protected_production_and_v4_artifact_count": protected_count,
        "output_directory": str(REPORT_DIR),
        "model_training_type": "walk_forward_validation_only",
        "lstm_retrained": False,
    }

    summary_path = REPORT_DIR / summary_filename

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return destinations


# =========================================================
# MAIN
# =========================================================

def main() -> int:
    run_id = datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S"
    )

    protected_files = None
    baseline = None
    plan = None
    results = None

    test_failed = False
    integrity_ok = False

    print("=" * 68)
    print("PHASE 15 REAL-DATA MODEL BENCHMARK")
    print("=" * 68)

    try:
        print("\n[1/6] Fingerprinting protected production and V4 files...")

        protected_files = discover_protected_files()

        baseline = {
            path: fingerprint(path)
            for path in protected_files
        }

        print(f"Production files protected: {len(PRODUCTION_FILES)}")
        print(
            "V4 candidate files protected: "
            f"{len(protected_files) - len(PRODUCTION_FILES)}"
        )
        print(f"Total protected files: {len(protected_files)}")

        plan = build_real_data_plan()

        results = run_experiments(plan)

    except Exception:
        test_failed = True
        print("\nREAL-DATA BENCHMARK: FAILED")
        traceback.print_exc()

    finally:
        print("\n[PROTECTION] Rechecking protected files...")

        if protected_files is None or baseline is None:
            print("Integrity baseline unavailable.")
            integrity_ok = False
        else:
            integrity_ok = verify_protected_files(
                protected_files,
                baseline,
            )

        print(
            "\nProtected production/V4 artifacts: "
            f"{'UNCHANGED' if integrity_ok else 'FAILED/UNVERIFIED'}"
        )

    if test_failed or not integrity_ok:
        print("\nREPORTS NOT SAVED.")
        print("OVERALL RESULT: FAILED")
        return 1

    try:
        print("\nSaving benchmark outputs under Phase 15/reports...")

        report_files = save_reports(
            plan,
            results,
            run_id,
            len(protected_files),
        )

        # Recheck protected files after report generation too.
        if not verify_protected_files(
            protected_files,
            baseline,
        ):
            raise RuntimeError(
                "Protected artifact integrity changed after report saving."
            )

        print("\nGenerated report files:")

        for path in report_files:
            print(path)

        print("\n" + "=" * 68)
        print("REAL-DATA MODEL BENCHMARK: PASSED")
        print("=" * 68)
        print("Primary experiment: completed")
        print("Forecast experiment: completed")
        print("Five models per experiment: completed")
        print("Final holdout: untouched")
        print("Production LSTM retraining: NOT PERFORMED")
        print("V4 candidate artifacts: unchanged")
        print("Phase 15 reports: saved")

        return 0

    except Exception:
        print("\nREPORT SAVING OR FINAL INTEGRITY CHECK FAILED.")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
