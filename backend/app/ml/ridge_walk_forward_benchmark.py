"""Research-only Ridge benchmark using the exact V4 walk-forward folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from backend.app.ml.lstm_model_v4_candidate import LSTMPricePredictor
from backend.app.ml.lstm_walk_forward_v4_candidate import (
    FOLDS,
    HORIZONS,
    fetch_candles,
    make_samples,
    metric_block,
    read_keys,
)

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]

MODELS_DIR = Path("backend/app/ml/models")
REFERENCE_CSV = MODELS_DIR / (
    "lstm_v4_candidate_activation_diag_walk_forward_predictions.csv"
)
OUTPUT_DIR = MODELS_DIR / "research"
OUTPUT_CSV = OUTPUT_DIR / "lstm_ridge_v4_benchmark_predictions.csv"
OUTPUT_JSON = OUTPUT_DIR / "lstm_ridge_v4_benchmark_report.json"

KEY_COLUMNS = ["fold", "instrument_key", "base_index", "base_timestamp"]


def make_key_set(frame):
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing test-key columns: {missing}")

    keys = []
    for row in frame[KEY_COLUMNS].itertuples(index=False, name=None):
        fold, instrument, base_index, timestamp = row
        keys.append((
            str(fold),
            str(instrument),
            int(base_index),
            str(timestamp),
        ))

    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate fold/instrument/date keys found.")

    return set(keys)


def training_mean_mae(rows, horizon):
    actual = np.asarray(
        [row[f"actual_return_{horizon}d"] for row in rows], dtype=float
    )
    mean_prediction = np.asarray(
        [row[f"train_mean_return_{horizon}d"] for row in rows], dtype=float
    )
    valid = np.isfinite(actual) & np.isfinite(mean_prediction)

    if not valid.any():
        return None

    return float(np.mean(np.abs(actual[valid] - mean_prediction[valid])) * 100.0)


def baseline_block(rows):
    result = {}
    for horizon in HORIZONS:
        result[str(horizon)] = {
            "training_mean_return_mae_pct_points":
                round(training_mean_mae(rows, horizon), 6)
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instrument-file",
        default="backend/app/ml/models/lstm_instrument_keys.txt",
    )
    parser.add_argument("--days", type=int, default=1095)
    args = parser.parse_args()

    if args.days < 180:
        raise ValueError("--days must be at least 180.")

    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(
            f"V4 reference predictions are missing: {REFERENCE_CSV}"
        )

    reference = pd.read_csv(REFERENCE_CSV)
    expected_keys = make_key_set(reference)

    keys = read_keys(None, args.instrument_file)
    if len(keys) < 2:
        raise ValueError("At least two instrument keys are required.")

    print(f"Reference V4 test rows: {len(reference)}")
    print(f"Instruments requested: {len(keys)}")
    print(f"Ridge alpha candidates: {ALPHAS}")

    template = LSTMPricePredictor(
        sequence_length=60,
        prediction_horizons=HORIZONS,
    )

    datasets = {}
    fetch_errors = {}

    for index, key in enumerate(keys, 1):
        print(f"\n[{index}/{len(keys)}] Fetching {args.days} days: {key}")
        try:
            candles = fetch_candles(key, args.days)
            df = template._clean_candles(candles)

            if len(df) < 250:
                raise ValueError(
                    f"Only {len(df)} usable rows after indicators."
                )

            datasets[key] = {
                "df": df,
                "raw": template._relative_features(df),
            }
            print(f"  Usable rows: {len(df)}")

        except Exception as exc:
            fetch_errors[key] = str(exc)
            print(f"  SKIPPED: {exc}")

    if len(datasets) < 2:
        raise RuntimeError(
            f"Fewer than two stocks loaded. Errors: {fetch_errors}"
        )

    all_predictions = []
    fold_summaries = []
    generated_test_keys = []

    for fold in FOLDS:
        fold_name = fold["name"]
        print(f"\n========== {fold_name} ==========")

        fold_bounds = {}
        scaler_fit_rows = []

        for key, item in datasets.items():
            df = item["df"]
            n = len(df)

            train_end = int(n * fold["train_end"])
            val_end = int(n * fold["val_end"])
            test_end = int(n * fold["test_end"])

            fold_bounds[key] = (train_end, val_end, test_end)
            scaler_fit_rows.append(item["raw"][:train_end])

        # Match V4: fit feature scaler on training rows only, pooled across stocks.
        feature_scaler = StandardScaler()
        feature_scaler.fit(np.concatenate(scaler_fit_rows, axis=0))

        train_samples = []
        val_samples = []
        test_samples = []
        usable_keys = []

        for key, item in datasets.items():
            train_end, val_end, test_end = fold_bounds[key]
            scaled = feature_scaler.transform(item["raw"])

            tr, va, te = make_samples(
                key,
                item["df"],
                scaled,
                template.sequence_length,
                HORIZONS,
                train_end,
                val_end,
                test_end,
            )

            if not tr or not va or not te:
                fetch_errors[f"{key}:{fold_name}"] = (
                    f"Insufficient samples: train={len(tr)}, "
                    f"validation={len(va)}, test={len(te)}"
                )
                continue

            usable_keys.append(key)
            train_samples.extend(
                (x, y, {**meta, "fold": fold_name})
                for x, y, meta in tr
            )
            val_samples.extend(
                (x, y, {**meta, "fold": fold_name})
                for x, y, meta in va
            )
            test_samples.extend(
                (x, y, {**meta, "fold": fold_name})
                for x, y, meta in te
            )

        if len(usable_keys) < 2 or not train_samples or not val_samples or not test_samples:
            raise RuntimeError(f"{fold_name}: insufficient samples.")

        # Verify this fold's test windows against the saved V4 reference later.
        for _, _, meta in test_samples:
            generated_test_keys.append((
                str(meta["fold"]),
                str(meta["instrument_key"]),
                int(meta["base_index"]),
                str(meta["base_timestamp"]),
            ))

        # Keep the temporal sequence, flattening only for Ridge regression.
        X_train = np.asarray(
            [x for x, _, _ in train_samples], dtype=np.float32
        ).reshape(len(train_samples), -1)

        X_val = np.asarray(
            [x for x, _, _ in val_samples], dtype=np.float32
        ).reshape(len(val_samples), -1)

        X_test = np.asarray(
            [x for x, _, _ in test_samples], dtype=np.float32
        ).reshape(len(test_samples), -1)

        y_train_raw = np.asarray(
            [y for _, y, _ in train_samples], dtype=np.float64
        )
        y_val_raw = np.asarray(
            [y for _, y, _ in val_samples], dtype=np.float64
        )

        # Match V4: per-horizon target normalization based on training labels only.
        target_mean = np.mean(y_train_raw, axis=0)
        target_scale = np.maximum(np.std(y_train_raw, axis=0), 0.005)
        y_train_scaled = (y_train_raw - target_mean) / target_scale

        # Match V4: equal total training weight per instrument.
        train_keys = [
            meta["instrument_key"] for _, _, meta in train_samples
        ]
        counts = {key: train_keys.count(key) for key in set(train_keys)}

        sample_weight = np.asarray(
            [1.0 / counts[key] for key in train_keys],
            dtype=np.float64,
        )
        sample_weight *= len(sample_weight) / sample_weight.sum()

        test_predictions = np.zeros(
            (len(test_samples), len(HORIZONS)), dtype=np.float64
        )

        selected_alphas = {}
        validation_scores = {}

        for column, horizon in enumerate(HORIZONS):
            best_alpha = None
            best_score = float("inf")

            # Select alpha on validation only. Test data is not used for tuning.
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(
                    X_train,
                    y_train_scaled[:, column],
                    sample_weight=sample_weight,
                )

                val_scaled = model.predict(X_val)
                val_raw = val_scaled * target_scale[column] + target_mean[column]

                score = float(
                    np.mean(np.abs(val_raw - y_val_raw[:, column]))
                )

                if score < best_score:
                    best_score = score
                    best_alpha = alpha

            if best_alpha is None:
                raise RuntimeError(
                    f"No Ridge alpha selected for horizon {horizon}."
                )

            # Refit the selected model on training data only.
            final_model = Ridge(alpha=best_alpha, fit_intercept=True)
            final_model.fit(
                X_train,
                y_train_scaled[:, column],
                sample_weight=sample_weight,
            )

            test_scaled = final_model.predict(X_test)
            test_predictions[:, column] = (
                test_scaled * target_scale[column] + target_mean[column]
            )

            selected_alphas[str(horizon)] = float(best_alpha)
            validation_scores[str(horizon)] = {
                "selected_alpha": float(best_alpha),
                "validation_mae_pct_points": round(best_score * 100.0, 6),
            }

        if not np.isfinite(test_predictions).all():
            raise RuntimeError(f"{fold_name}: non-finite Ridge predictions.")

        fold_rows = []

        for index, (_, _, meta) in enumerate(test_samples):
            row = {
                "fold": fold_name,
                "instrument_key": meta["instrument_key"],
                "base_index": int(meta["base_index"]),
                "base_timestamp": str(meta["base_timestamp"]),
                "base_close": float(meta["base_close"]),
            }

            for column, horizon in enumerate(HORIZONS):
                prediction = float(test_predictions[index, column])
                actual = float(meta["actual_returns"][horizon])
                training_mean = float(target_mean[column])

                row[f"pred_return_{horizon}d"] = prediction
                row[f"actual_return_{horizon}d"] = actual
                row[f"pred_price_{horizon}d"] = (
                    float(meta["base_close"]) * (1.0 + prediction)
                )
                row[f"actual_price_{horizon}d"] = float(
                    meta["actual_prices"][horizon]
                )
                row[f"train_mean_return_{horizon}d"] = training_mean

            fold_rows.append(row)

        all_predictions.extend(fold_rows)

        fold_summary = {
            "fold": fold_name,
            "stocks_evaluated": len(usable_keys),
            "training_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "held_out_test_samples": len(test_samples),
            "feature_count_after_flattening": int(X_train.shape[1]),
            "selected_alphas": selected_alphas,
            "validation_alpha_selection": validation_scores,
            "test_metrics": {
                str(h): metric_block(fold_rows, h) for h in HORIZONS
            },
            "training_mean_baseline": baseline_block(fold_rows),
            "date_bounds": {
                key: {
                    "train_end": int(bounds[0]),
                    "validation_end": int(bounds[1]),
                    "test_end": int(bounds[2]),
                }
                for key, bounds in fold_bounds.items()
            },
        }

        fold_summaries.append(fold_summary)

        print(f"Stocks evaluated: {len(usable_keys)}")
        print(f"Train/validation/test: {len(train_samples)}/"
              f"{len(val_samples)}/{len(test_samples)}")
        print(f"Selected alpha by horizon: {selected_alphas}")

        for h in HORIZONS:
            metrics = fold_summary["test_metrics"][str(h)]
            mean_mae = fold_summary["training_mean_baseline"][str(h)][
                "training_mean_return_mae_pct_points"
            ]
            print(
                f"  {h}-session: Ridge MAE="
                f"{metrics['return_mae_pct_points']:.6f} pp | "
                f"zero baseline="
                f"{metrics['baseline_no_change_return_mae_pct_points']:.6f} pp | "
                f"training-mean baseline={mean_mae:.6f} pp"
            )

    # Critical fairness check: compare every generated test row to V4's saved rows.
    generated_frame = pd.DataFrame(
        generated_test_keys, columns=KEY_COLUMNS
    )
    generated_keys = make_key_set(generated_frame)

    if generated_keys != expected_keys:
        missing = list(expected_keys - generated_keys)[:5]
        extra = list(generated_keys - expected_keys)[:5]
        raise RuntimeError(
            "TEST DATE ALIGNMENT FAILED. Ridge results were NOT saved. "
            f"V4 reference rows={len(expected_keys)}, "
            f"Ridge generated rows={len(generated_keys)}. "
            f"Example missing keys={missing}; example extra keys={extra}. "
            "The fetched candle history or fold boundaries differ from V4."
        )

    pooled_metrics = {
        str(h): metric_block(all_predictions, h) for h in HORIZONS
    }
    pooled_training_mean = baseline_block(all_predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_predictions).to_csv(OUTPUT_CSV, index=False)

    report = {
        "status": "success",
        "model": "Ridge Regression",
        "research_only": True,
        "production_artifacts_modified": False,
        "reference_v4_csv": str(REFERENCE_CSV),
        "test_rows_exactly_match_v4": True,
        "matched_test_rows": len(generated_keys),
        "instruments_requested": len(keys),
        "instruments_loaded": len(datasets),
        "alpha_candidates": ALPHAS,
        "alpha_selection": (
            "Per-horizon alpha selected by validation MAE; "
            "test rows are not used for tuning."
        ),
        "input_representation": (
            "60-session x 14-feature V4 sequences flattened to 840 features."
        ),
        "target_scaling": (
            "Per-horizon training-label mean and standard deviation, "
            "with the same 0.005 scale floor as V4."
        ),
        "sample_weighting": (
            "Equal total training sample weight per instrument, matching V4."
        ),
        "fold_summaries": fold_summaries,
        "overall_metrics_pooled_test_windows": pooled_metrics,
        "pooled_training_mean_baseline": pooled_training_mean,
        "fetch_or_split_errors": fetch_errors,
        "predictions_csv": str(OUTPUT_CSV),
        "report_path": str(OUTPUT_JSON),
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nRIDGE WALK-FORWARD BENCHMARK COMPLETE")
    print(f"Exact V4 test-row match: YES ({len(generated_keys)} rows)")
    print(f"Predictions: {OUTPUT_CSV}")
    print(f"Report: {OUTPUT_JSON}")

    print("\nPOOLED TEST COMPARISON (MAE percentage points)")
    print(f"{'Horizon':<12} {'Ridge':>12} {'Zero baseline':>16} {'Train mean':>14}")

    for h in HORIZONS:
        ridge_mae = pooled_metrics[str(h)]["return_mae_pct_points"]
        zero_mae = pooled_metrics[str(h)][
            "baseline_no_change_return_mae_pct_points"
        ]
        mean_mae = pooled_training_mean[str(h)][
            "training_mean_return_mae_pct_points"
        ]
        print(
            f"{h}-session   {ridge_mae:>12.6f} "
            f"{zero_mae:>16.6f} {mean_mae:>14.6f}"
        )

    print("\nProduction artifacts modified: NO")


if __name__ == "__main__":
    main()
