"""Read-only integration test for the Phase 15 real-data pipeline."""

from __future__ import annotations

import hashlib
import sys
import time
import traceback
from pathlib import Path

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


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path.cwd()
MODEL_DIR = ROOT / "backend/app/ml/models"
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

NIFTY_KEY = "NSE_INDEX|Nifty 50"
VIX_KEY = "NSE_INDEX|India VIX"

PRODUCTION_FILES = [
    MODEL_DIR / "lstm_model_v2.keras",
    MODEL_DIR / "lstm_scalers_v2.pkl",
]


# =========================================================
# PROTECTED FILE FINGERPRINTS
# =========================================================

def discover_protected_files() -> list[Path]:
    """Include production files and every original V4 candidate variant."""

    missing_production = [
        path for path in PRODUCTION_FILES
        if not path.is_file()
    ]

    if missing_production:
        raise FileNotFoundError(
            "Missing production files:\n"
            + "\n".join(str(path) for path in missing_production)
        )

    v4_files = sorted(
        path for path in MODEL_DIR.glob("lstm_v4_candidate*")
        if path.is_file()
    )

    if len(v4_files) < 3:
        raise FileNotFoundError(
            "Expected at least three V4 candidate research artifacts. "
            f"Found {len(v4_files)}."
        )

    return PRODUCTION_FILES + v4_files


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


def read_stock_keys() -> list[str]:
    if not KEY_FILE.is_file():
        raise FileNotFoundError(
            f"Instrument key file not found: {KEY_FILE}"
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
            f"Expected {EXPECTED_STOCKS} stock keys; found {len(keys)}."
        )

    if len(set(keys)) != len(keys):
        raise AssertionError("Duplicate stock instrument keys found.")

    return keys


def fetch_with_retry(
    instrument_key: str,
    start_date: str,
    end_date: str,
    attempts: int = 3,
):
    """Retry only transient connection and read-timeout failures."""

    transient_errors = (
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
                for marker in transient_errors
            )

            if not retryable or attempt == attempts:
                raise

            delay = 2 ** attempt

            print(
                f"Transient API error for {instrument_key}: "
                f"{message}. Retry {attempt + 1}/{attempts} "
                f"in {delay} seconds."
            )

            time.sleep(delay)


# =========================================================
# INTEGRATION TEST
# =========================================================

def run_integration() -> None:
    print("=" * 68)
    print("PHASE 15 REAL-DATA INTEGRATION TEST")
    print("=" * 68)

    # 1. Protect production and all V4 candidate artifacts.
    print("\n[1/8] Fingerprinting protected files...")

    protected_files = discover_protected_files()

    before = {
        path: fingerprint(path)
        for path in protected_files
    }

    print(f"Production files protected: {len(PRODUCTION_FILES)}")
    print(
        "V4 candidate artifacts protected: "
        f"{len(protected_files) - len(PRODUCTION_FILES)}"
    )
    print(f"Total protected files: {len(protected_files)}")

    # 2. Read the instrument universe.
    print("\n[2/8] Loading stock instrument keys...")

    stock_keys = read_stock_keys()

    print(f"Unique stock instruments: {len(stock_keys)}")

    # 3. Fetch benchmark data.
    print("\n[3/8] Fetching NIFTY 50 and India VIX...")

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
        row["session_date"]
        for row in nifty_data["rows"]
    ]

    vix_calendar = [
        row["session_date"]
        for row in vix_data["rows"]
    ]

    if len(nifty_calendar) != EXPECTED_SESSIONS:
        raise AssertionError(
            f"NIFTY returned {len(nifty_calendar)} sessions; "
            f"expected {EXPECTED_SESSIONS}."
        )

    if len(vix_calendar) != EXPECTED_SESSIONS:
        raise AssertionError(
            f"India VIX returned {len(vix_calendar)} sessions; "
            f"expected {EXPECTED_SESSIONS}."
        )

    if nifty_calendar != vix_calendar:
        raise AssertionError(
            "NIFTY and India VIX calendars do not match."
        )

    print(f"NIFTY sessions: {len(nifty_calendar)}")
    print(f"India VIX sessions: {len(vix_calendar)}")
    print("Benchmark calendars: MATCH")

    # 4. Build individual stock training frames in memory.
    print("\n[4/8] Building all 20 stock training frames...")

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
                f"{instrument_key}: received {len(stock_calendar)} "
                f"sessions; expected {EXPECTED_SESSIONS}."
            )

        if stock_calendar != nifty_calendar:
            raise AssertionError(
                f"{instrument_key}: trading calendar differs from NIFTY."
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
                f"{instrument_key}: Phase 15 feature schema mismatch."
            )

        if len(features) != EXPECTED_SESSIONS:
            raise AssertionError(
                f"{instrument_key}: unexpected feature row count."
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
                f"{instrument_key}: unexpected readiness counts. "
                f"Actual={counts}; expected={expected_counts}."
            )

        if frame["session_date"].tolist() != stock_calendar:
            raise AssertionError(
                f"{instrument_key}: training frame dates changed order."
            )

        training_frames[instrument_key] = frame

        print(
            f"    rows={len(frame)}, "
            f"features_ready={counts['feature_ready']}, "
            f"targets_ready={counts['targets_ready']}, "
            f"eligible={counts['training_eligible']}"
        )

    print("\nAll 20 individual stock frames passed.")

    # 5. Combine and validate the shared panel.
    print("\n[5/8] Combining the stock training frames...")

    panel = combine_training_frames(training_frames)

    expected_panel_rows = EXPECTED_STOCKS * EXPECTED_SESSIONS

    if len(panel) != expected_panel_rows:
        raise AssertionError(
            f"Panel has {len(panel)} rows; "
            f"expected {expected_panel_rows}."
        )

    if panel["instrument_key"].nunique() != EXPECTED_STOCKS:
        raise AssertionError(
            "Combined panel does not contain exactly 20 instruments."
        )

    eligible_dates = panel.loc[
        panel["training_eligible"],
        "session_date",
    ].nunique()

    if eligible_dates != EXPECTED_ELIGIBLE:
        raise AssertionError(
            f"Shared eligible dates={eligible_dates}; "
            f"expected {EXPECTED_ELIGIBLE}."
        )

    print(f"Combined rows: {len(panel)}")
    print(f"Combined instruments: {panel['instrument_key'].nunique()}")
    print(f"Shared eligible dates: {eligible_dates}")

    # 6. Generate and validate five chronological folds.
    print("\n[6/8] Building five walk-forward folds...")

    plan = build_walk_forward_splits(
        panel,
        n_splits=N_SPLITS,
        initial_train_size=INITIAL_TRAIN_SIZE,
        final_holdout_size=FINAL_HOLDOUT_SIZE,
        max_horizon=MAX_HORIZON,
    )

    if len(plan.folds) != N_SPLITS:
        raise AssertionError(
            f"Expected {N_SPLITS} folds; received {len(plan.folds)}."
        )

    expected_validation_sizes = [61, 61, 61, 60, 60]

    actual_validation_sizes = [
        fold.validation_session_count
        for fold in plan.folds
    ]

    if actual_validation_sizes != expected_validation_sizes:
        raise AssertionError(
            "Validation fold sizes are different from expected: "
            f"{actual_validation_sizes}"
        )

    for fold in plan.folds:
        train = plan.panel.iloc[
            list(fold.train_indices)
        ]

        validation = plan.panel.iloc[
            list(fold.validation_indices)
        ]

        if train["instrument_key"].nunique() != EXPECTED_STOCKS:
            raise AssertionError(
                f"Fold {fold.fold_number}: missing training instruments."
            )

        if validation["instrument_key"].nunique() != EXPECTED_STOCKS:
            raise AssertionError(
                f"Fold {fold.fold_number}: missing validation instruments."
            )

        validation_targets = pd.to_datetime(
            validation["max_horizon_target_session"],
            errors="coerce",
        )

        if validation_targets.isna().any():
            raise AssertionError(
                f"Fold {fold.fold_number}: missing validation target dates."
            )

        if not (
            validation_targets
            < pd.Timestamp(plan.final_holdout_start_date)
        ).all():
            raise AssertionError(
                f"Fold {fold.fold_number}: validation targets "
                "cross the final holdout boundary."
            )

        if fold.purged_session_count != MAX_HORIZON:
            raise AssertionError(
                f"Fold {fold.fold_number}: unexpected purge session count."
            )

        expected_purged_rows = MAX_HORIZON * EXPECTED_STOCKS

        if fold.purged_sample_count != expected_purged_rows:
            raise AssertionError(
                f"Fold {fold.fold_number}: unexpected purge sample count."
            )

        print(
            f"Fold {fold.fold_number}: "
            f"training={fold.train_session_count} dates/"
            f"{len(fold.train_indices)} rows; "
            f"validation={fold.validation_session_count} dates/"
            f"{len(fold.validation_indices)} rows; "
            f"purged={fold.purged_session_count} dates/"
            f"{fold.purged_sample_count} rows; "
            f"validation={fold.validation_start_date} to "
            f"{fold.validation_end_date}"
        )

    # 7. Validate final training and sealed holdout.
    print("\n[7/8] Verifying final training and holdout...")

    final_train = plan.panel.iloc[
        list(plan.final_train_indices)
    ]

    final_holdout = plan.panel.iloc[
        list(plan.final_holdout_indices)
    ]

    if plan.final_train_session_count != 555:
        raise AssertionError(
            f"Final training dates={plan.final_train_session_count}; "
            "expected 555."
        )

    if len(plan.final_train_indices) != 555 * EXPECTED_STOCKS:
        raise AssertionError("Final training row count is incorrect.")

    if plan.final_holdout_session_count != FINAL_HOLDOUT_SIZE:
        raise AssertionError("Final holdout date count is incorrect.")

    if len(plan.final_holdout_indices) != (
        FINAL_HOLDOUT_SIZE * EXPECTED_STOCKS
    ):
        raise AssertionError("Final holdout row count is incorrect.")

    if plan.final_purged_session_count != MAX_HORIZON:
        raise AssertionError("Final training purge date count is incorrect.")

    expected_final_purged_rows = MAX_HORIZON * EXPECTED_STOCKS

    if plan.final_purged_sample_count != expected_final_purged_rows:
        raise AssertionError("Final training purge row count is incorrect.")

    holdout_start = pd.Timestamp(plan.final_holdout_start_date)

    final_train_target_dates = pd.to_datetime(
        final_train["max_horizon_target_session"],
        errors="coerce",
    )

    if final_train_target_dates.isna().any():
        raise AssertionError("Final training has missing target dates.")

    if not (final_train_target_dates < holdout_start).all():
        raise AssertionError(
            "Final training labels reach or cross the holdout."
        )

    if set(plan.final_train_indices).intersection(
        plan.final_holdout_indices
    ):
        raise AssertionError(
            "Final training and holdout indices overlap."
        )

    if final_holdout["instrument_key"].nunique() != EXPECTED_STOCKS:
        raise AssertionError("Final holdout is missing instruments.")

    if plan.holdout_embargo_session_count != MAX_HORIZON:
        raise AssertionError("Holdout embargo metadata is incorrect.")

    print(
        f"Final training: {plan.final_train_start_date} to "
        f"{plan.final_train_end_date}"
    )
    print(f"Final training dates: {plan.final_train_session_count}")
    print(f"Final training rows: {len(plan.final_train_indices)}")
    print(
        f"Final purge: {plan.final_purged_session_count} dates/"
        f"{plan.final_purged_sample_count} rows"
    )
    print(
        f"Final holdout: {plan.final_holdout_start_date} to "
        f"{plan.final_holdout_end_date}"
    )
    print(f"Final holdout dates: {plan.final_holdout_session_count}")
    print(f"Final holdout rows: {len(plan.final_holdout_indices)}")
    print("Final training/holdout overlap: NONE")

    # Summary.
    print("\n" + "=" * 68)
    print("REAL-DATA INTEGRATION: ALL CHECKS PASSED")
    print("=" * 68)

    print("Historical loader: PASSED")
    print("20-stock feature generation: PASSED")
    print("20-stock target generation: PASSED")
    print("Shared calendar: PASSED")
    print("Combined panel: PASSED")
    print("Five chronological folds: PASSED")
    print("Validation purging: PASSED")
    print("Validation embargo before holdout: PASSED")
    print("Final training purge: PASSED")
    print("Final holdout isolation: PASSED")


def main() -> int:
    before = None
    protected_files = []
    test_failed = False
    integrity_failed = False

    try:
        protected_files = discover_protected_files()

        print(
            f"Preflight: {len(protected_files)} protected files found."
        )

        before = {
            path: fingerprint(path)
            for path in protected_files
        }

        run_integration()

    except Exception:
        test_failed = True
        print("\nREAL-DATA INTEGRATION: FAILED")
        traceback.print_exc()

    finally:
        print("\n[PROTECTION] Checking production and V4 file integrity...")

        if before is None:
            integrity_failed = True
            print(
                "Fingerprint baseline was not created. "
                "Integrity result is UNVERIFIED."
            )
        else:
            try:
                after_files = discover_protected_files()

                if after_files != protected_files:
                    integrity_failed = True
                    print(
                        "Protected artifact file list changed."
                    )

                for path in protected_files:
                    if not path.is_file():
                        integrity_failed = True
                        print(f"MISSING: {path}")
                        continue

                    after = fingerprint(path)
                    unchanged = before[path] == after

                    print(
                        f"{'UNCHANGED' if unchanged else 'CHANGED'}: "
                        f"{path.name}"
                    )

                    if not unchanged:
                        integrity_failed = True

            except Exception:
                integrity_failed = True
                traceback.print_exc()

        if integrity_failed:
            print("\nPROTECTED ARTIFACT INTEGRITY: FAILED OR UNVERIFIED")
        else:
            print("\nPROTECTED ARTIFACT INTEGRITY: PASSED")

    if test_failed or integrity_failed:
        print("\nOVERALL RESULT: FAILED")
        return 1

    print("\nOVERALL RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
