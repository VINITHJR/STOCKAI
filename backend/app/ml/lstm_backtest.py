"""Chronological, held-out backtest for StockAI LSTM v2.

Run from the StockAI project root:
python -m backend.app.ml.lstm_backtest --instrument-keys "NSE_EQ|KEY1" "NSE_EQ|KEY2" --days 730 --epochs 40

This trains a TEMPORARY model in memory and does not overwrite production artifacts.
Split per instrument: first 70% train, next 15% validation, final 15% held-out test.
The validation segment selects the best epoch; only the final test segment is used
for the reported out-of-sample metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from backend.app.services.upstox_service import get_historical_candles
from backend.app.ml.lstm_model import LSTMPricePredictor, TENSORFLOW_AVAILABLE


def fetch_candles(instrument_key: str, days: int) -> List[dict]:
    end_date = date.today()
    start_date = end_date - timedelta(days=int(days))
    result = get_historical_candles(
        instrument_key=instrument_key,
        unit="days",
        interval="1",
        to_date=end_date.isoformat(),
        from_date=start_date.isoformat(),
    )
    raw = result.get("data", {}).get("candles", []) if isinstance(result, dict) else []
    candles = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        try:
            candles.append({
                "timestamp": str(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5] or 0),
            })
        except (TypeError, ValueError):
            continue
    if not candles:
        raise ValueError("No valid historical candles returned.")
    return candles


def _target_for_horizon(closes: np.ndarray, base_idx: int, horizon: int) -> float:
    return float(closes[base_idx + horizon] / closes[base_idx] - 1.0)


def _collect_split_samples(
    key: str,
    df,
    scaled_features: np.ndarray,
    sequence_length: int,
    horizons: Sequence[int],
    train_end: int,
    val_end: int,
) -> Tuple[list, list, list]:
    """Create windows without allowing target dates to cross split boundaries."""
    closes = df["close"].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].astype(str).to_numpy()
    max_horizon = max(horizons)
    n = len(df)
    train, val, test = [], [], []

    # base_idx is the last row in the input sequence. Horizon h targets
    # base_idx+h, matching the v2 training target definition.
    for base_idx in range(sequence_length - 1, n - max_horizon):
        start = base_idx - sequence_length + 1
        end_target = base_idx + max_horizon
        if closes[base_idx] <= 0:
            continue
        x = scaled_features[start:base_idx + 1].astype(np.float32)
        y = np.asarray(
            [_target_for_horizon(closes, base_idx, h) for h in horizons],
            dtype=np.float32,
        )
        meta = {
            "instrument_key": key,
            "base_timestamp": str(timestamps[base_idx]),
            "base_close": float(closes[base_idx]),
            "target_timestamps": [str(timestamps[base_idx + h]) for h in horizons],
            "actual_prices": [float(closes[base_idx + h]) for h in horizons],
            "actual_returns": [float(v) for v in y],
        }
        if end_target < train_end:
            train.append((x, y, meta))
        elif base_idx >= train_end - 1 and end_target < val_end:
            val.append((x, y, meta))
        elif base_idx >= val_end - 1 and end_target < n:
            test.append((x, y, meta))
    return train, val, test


def _direction(value: float, threshold: float = 0.0005) -> str:
    # 0.0005 return = 0.05%, the same deadband used by the live prediction output.
    if value > threshold:
        return "bullish"
    if value < -threshold:
        return "bearish"
    return "neutral"


def _safe_mean(values: Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def calculate_metrics(rows: List[dict], horizon_index: int, horizon: int) -> dict:
    actual_returns = np.asarray([r["actual_returns"][horizon_index] for r in rows], dtype=float)
    predicted_returns = np.asarray([r["predicted_returns"][horizon_index] for r in rows], dtype=float)
    actual_prices = np.asarray([r["actual_prices"][horizon_index] for r in rows], dtype=float)
    predicted_prices = np.asarray([r["predicted_prices"][horizon_index] for r in rows], dtype=float)
    base_prices = np.asarray([r["base_close"] for r in rows], dtype=float)

    return_errors_pct_points = (predicted_returns - actual_returns) * 100.0
    price_errors = predicted_prices - actual_prices
    baseline_price_errors = base_prices - actual_prices
    baseline_return_errors_pct_points = (-actual_returns) * 100.0

    actual_dirs = [_direction(x) for x in actual_returns]
    predicted_dirs = [_direction(x) for x in predicted_returns]
    directional_correct = sum(a == p for a, p in zip(actual_dirs, predicted_dirs))
    non_neutral = [i for i, a in enumerate(actual_dirs) if a != "neutral"]
    directional_non_neutral = (
        100.0 * sum(actual_dirs[i] == predicted_dirs[i] for i in non_neutral) / len(non_neutral)
        if non_neutral else None
    )

    return {
        "horizon_days": int(horizon),
        "test_samples": int(len(rows)),
        "return_mae_pct_points": round(float(np.mean(np.abs(return_errors_pct_points))), 6),
        "return_rmse_pct_points": round(float(np.sqrt(np.mean(return_errors_pct_points ** 2))), 6),
        "price_mae_inr": round(float(np.mean(np.abs(price_errors))), 4),
        "price_rmse_inr": round(float(np.sqrt(np.mean(price_errors ** 2))), 4),
        "directional_accuracy_percent_including_neutral": round(100.0 * directional_correct / len(rows), 3) if rows else None,
        "directional_accuracy_percent_actual_moves_only": round(directional_non_neutral, 3) if directional_non_neutral is not None else None,
        "baseline_no_change_return_mae_pct_points": round(float(np.mean(np.abs(baseline_return_errors_pct_points))), 6),
        "baseline_no_change_price_mae_inr": round(float(np.mean(np.abs(baseline_price_errors))), 4),
        "beats_no_change_baseline_on_return_mae": bool(
            np.mean(np.abs(return_errors_pct_points)) < np.mean(np.abs(baseline_return_errors_pct_points))
        ) if rows else False,
    }


def run_backtest(instrument_keys: List[str], days: int, epochs: int, batch_size: int) -> dict:
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is unavailable in this Python environment.")
    keys = list(dict.fromkeys(k.strip() for k in instrument_keys if k and k.strip()))
    if len(keys) < 2:
        raise ValueError("Provide at least two instruments. More instruments are strongly recommended.")

    predictor = LSTMPricePredictor(sequence_length=60, prediction_horizons=[1, 3, 7])
    datasets = []
    fetch_errors = {}
    for idx, key in enumerate(keys, start=1):
        print(f"[{idx}/{len(keys)}] Fetching {days} calendar days for {key} ...")
        try:
            candles = fetch_candles(key, days)
            df = predictor._clean_candles(candles)
            n = len(df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            minimum = predictor.sequence_length + max(predictor.prediction_horizons) + 20
            if n < minimum or train_end <= predictor.sequence_length + max(predictor.prediction_horizons):
                raise ValueError(f"Only {n} usable rows after indicators; increase --days.")
            raw_features = predictor._relative_features(df)
            datasets.append({
                "key": key, "df": df, "raw": raw_features,
                "train_end": train_end, "val_end": val_end,
            })
            print(f"  Usable rows: {n}; split indices train={train_end}, validation={val_end}, test={n}")
        except Exception as exc:
            fetch_errors[key] = str(exc)
            print(f"  SKIP: {exc}")

    if len(datasets) < 2:
        raise RuntimeError(f"At least two instruments must have enough history. Errors: {fetch_errors}")

    # Fit preprocessing ONLY on each instrument's training-period rows.
    predictor.feature_scaler.fit(np.concatenate([d["raw"][:d["train_end"]] for d in datasets], axis=0))

    train_samples, val_samples, test_samples = [], [], []
    split_summary = {}
    for d in datasets:
        scaled = predictor.feature_scaler.transform(d["raw"])
        train, val, test = _collect_split_samples(
            d["key"], d["df"], scaled, predictor.sequence_length,
            predictor.prediction_horizons, d["train_end"], d["val_end"],
        )
        if not train or not val or not test:
            fetch_errors[d["key"]] = (
                f"Insufficient split windows: train={len(train)}, validation={len(val)}, test={len(test)}. "
                "Increase --days."
            )
            print(f"  SKIP from evaluation: {fetch_errors[d['key']]}")
            continue
        train_samples.extend(train)
        val_samples.extend(val)
        test_samples.extend(test)
        split_summary[d["key"]] = {
            "usable_rows": int(len(d["df"])),
            "train_samples": len(train),
            "validation_samples": len(val),
            "held_out_test_samples": len(test),
            "test_start_timestamp": test[0][2]["base_timestamp"],
            "test_end_timestamp": test[-1][2]["base_timestamp"],
        }

    if len(split_summary) < 2:
        raise RuntimeError(f"At least two instruments need valid train/validation/test splits. Errors: {fetch_errors}")

    X_train = np.asarray([s[0] for s in train_samples], dtype=np.float32)
    y_train = np.asarray([s[1] for s in train_samples], dtype=np.float32)
    X_val = np.asarray([s[0] for s in val_samples], dtype=np.float32)
    y_val = np.asarray([s[1] for s in val_samples], dtype=np.float32)
    X_test = np.asarray([s[0] for s in test_samples], dtype=np.float32)

    print(f"\nTraining temporary model: train={len(X_train)}, validation={len(X_val)}, held-out test={len(X_test)}")
    # Intentionally train in memory and DO NOT call predictor.save().
    predictor.model = predictor.build_model((predictor.sequence_length, len(predictor.feature_columns)))
    history = predictor.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        shuffle=False,
        verbose=1,
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            )
        ],
    )
    predictor.is_trained = True
    pred_returns = np.asarray(predictor.model.predict(X_test, verbose=0), dtype=np.float64)
    pred_returns = np.clip(pred_returns, -0.5, 0.5)

    test_rows = []
    for idx, sample in enumerate(test_samples):
        meta = dict(sample[2])
        returns = pred_returns[idx].tolist()
        meta["predicted_returns"] = returns
        meta["predicted_prices"] = [
            float(meta["base_close"] * (1.0 + ret)) for ret in returns
        ]
        for h_idx, horizon in enumerate(predictor.prediction_horizons):
            meta[f"predicted_return_{horizon}d_percent"] = returns[h_idx] * 100.0
            meta[f"predicted_price_{horizon}d"] = meta["predicted_prices"][h_idx]
            meta[f"actual_return_{horizon}d_percent"] = meta["actual_returns"][h_idx] * 100.0
            meta[f"actual_price_{horizon}d"] = meta["actual_prices"][h_idx]
        test_rows.append(meta)

    overall_metrics = {
        str(h): calculate_metrics(test_rows, i, h)
        for i, h in enumerate(predictor.prediction_horizons)
    }
    per_instrument_metrics = {}
    for key in split_summary:
        rows = [r for r in test_rows if r["instrument_key"] == key]
        per_instrument_metrics[key] = {
            str(h): calculate_metrics(rows, i, h)
            for i, h in enumerate(predictor.prediction_horizons)
        }

    root = Path("backend/app/ml/models")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "lstm_v2_backtest_report.json"
    predictions_path = root / "lstm_v2_backtest_predictions.csv"
    report = {
        "status": "success",
        "model_version": 2,
        "method": "chronological holdout; 70% train / 15% validation / 15% final test per instrument",
        "production_artifacts_modified": False,
        "note": "This evaluates a newly trained temporary model, not the already-saved production model. Metrics are out-of-sample only relative to these chronological splits.",
        "target_type": "future_return",
        "horizons_days": predictor.prediction_horizons,
        "instruments_requested": keys,
        "instruments_evaluated": list(split_summary.keys()),
        "days_requested": int(days),
        "epochs_completed": len(history.history.get("loss", [])),
        "best_validation_loss": float(min(history.history.get("val_loss", [math.nan]))),
        "sample_counts": {
            "train": int(len(X_train)), "validation": int(len(X_val)), "held_out_test": int(len(X_test))
        },
        "split_summary": split_summary,
        "overall_metrics": overall_metrics,
        "per_instrument_metrics": per_instrument_metrics,
        "fetch_or_split_errors": fetch_errors,
        "report_path": str(report_path),
        "predictions_csv_path": str(predictions_path),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    if test_rows:
        fieldnames = []
        for row in test_rows:
            for name in row:
                if name not in fieldnames and not isinstance(row[name], (list, dict)):
                    fieldnames.append(name)
        with predictions_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in test_rows:
                writer.writerow({k: v for k, v in row.items() if k in fieldnames})

    print("\nBACKTEST COMPLETE")
    print(f"Report:      {report_path}")
    print(f"Predictions: {predictions_path}")
    print(json.dumps({
        "status": report["status"],
        "production_artifacts_modified": False,
        "sample_counts": report["sample_counts"],
        "overall_metrics": report["overall_metrics"],
        "report_path": str(report_path),
        "predictions_csv_path": str(predictions_path),
        "fetch_or_split_errors": fetch_errors,
    }, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological held-out backtest for StockAI LSTM v2")
    parser.add_argument("--instrument-keys", nargs="+", required=True)
    parser.add_argument("--days", type=int, default=730, help="Calendar days of historical data per stock; 730+ recommended")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_backtest(args.instrument_keys, args.days, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
